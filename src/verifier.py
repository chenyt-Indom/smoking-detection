"""VLM 二次复核：把「疑似异常」交给多模态大模型判一次。

为什么要有这一层
----------------
YOLO 只会说"这里有个像火/烟的东西"，它不懂**常识**：
灶台上的火苗是正常的、厂房顶上一片白光可能是灯、飘过来的白雾可能是蒸汽。
这些判断需要"位置 + 周围环境 + 时间变化"三样一起看 —— 正好是 VLM 的强项。

放在流水线哪一步
----------------
    ④ 时序判别（本地规则）说「疑似」  →  ⑤ VLM 复核  →  ⑥ 事件
只在④给出候选之后调用，不是每帧都调（1 秒/次 × 25fps 是不可能的）。

四条硬约束（少一条就会翻车）
--------------------------
1. **送 3 帧拼图，不送单帧** —— 单帧等于退回"单帧识别"，白烟和蒸汽分不开
2. **强制 JSON + 固定枚举** —— 不能让它自由发挥，否则无法自动决策
3. **限额 + 去重** —— 同一目标 2 分钟只审一次；每小时封顶，超了降级
4. **必须能降级** —— 超时/429/额度用完/返回脏数据 → 回退本地规则，不能因为云端挂了就不报警

实测（2026-10-29，deepseek-flash）：
    单次 0.9~1.2s，约 284 tokens（图 256 + 输出 28）；
    能正确区分 火苗 / 蒸汽 / 吊灯（靠"细杆支撑"这种环境线索）。
"""
from __future__ import annotations

import base64
import json
import os
import re
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .temporal import TemporalFeatures, Verdict as RuleVerdict, judge

try:
    import yaml
except Exception:            # pragma: no cover
    yaml = None

try:
    import cv2
except Exception:            # pragma: no cover
    cv2 = None


# ---------------------------------------------------------------- 配置
DEFAULT_CFG = {
    "enabled": True,
    "model": "deepseek-flash",
    "url": "https://api.deepseek.com/chat/completions",
    "api_key_env": "DEEPSEEK_KEY",
    "thinking": "disabled",
    "temperature": 0.1,
    "max_tokens": 300,
    "timeout_s": 15,
    "min_duration_s": 6.0,
    "min_conf": 0.35,
    "frames": 3,
    "frame_gap_s": 5.0,
    "hourly_limit": 200,
    "dedup_seconds": 120,
    "labels": ["fire", "smoke", "steam", "light", "other", "uncertain"],
    "accept_conf": 0.6,
}


def load_config(path: str | Path | None = None) -> dict:
    cfg = dict(DEFAULT_CFG)
    p = Path(path) if path else Path(__file__).resolve().parent.parent / "config" / "verifier.yaml"
    if yaml is not None and p.exists():
        try:
            loaded = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
            cfg.update({k: v for k, v in loaded.items() if v is not None})
        except Exception:
            pass
    return cfg


# ---------------------------------------------------------------- 结果结构
@dataclass
class Verdict2:
    """复核结论。source 标明是 VLM 给的还是降级到本地规则给的。"""

    label: str                     # fire / smoke / steam / light / other / uncertain
    confidence: float
    reason: str = ""
    risk: str = "low"              # high / medium / low
    source: str = "vlm"            # vlm | rule | skipped
    latency_s: float = 0.0
    raw: dict = field(default_factory=dict)
    needs_review: bool = False     # 置信度处在"像但不肯定"区间，建议人工过一眼

    @property
    def is_alarm(self) -> bool:
        """够格直接拉警报：明确是火/烟且置信度够。"""
        return self.label in ("fire", "smoke") and self.confidence >= 0.5

    @property
    def is_suspect(self) -> bool:
        """疑似但不够格报警：进"待确认"队列，不能直接丢。"""
        return (self.label in ("fire", "smoke") and self.confidence < 0.5) or self.needs_review


# ---------------------------------------------------------------- 图片编码
def _encode(img: np.ndarray, max_side: int = 640) -> str | None:
    """BGR ndarray → data URI（JPEG，限制长边，控制 token 与体积）。"""
    if img is None or img.size == 0:
        return None
    if cv2 is not None:
        h, w = img.shape[:2]
        s = min(1.0, max_side / max(h, w))
        if s < 1.0:
            img = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        if not ok:
            return None
        b64 = base64.b64encode(buf.tobytes()).decode()
    else:
        from io import BytesIO
        from PIL import Image
        bio = BytesIO()
        Image.fromarray(img[:, :, ::-1]).save(bio, format="JPEG", quality=85)
        b64 = base64.b64encode(bio.getvalue()).decode()
    return f"data:image/jpeg;base64,{b64}"


# ---------------------------------------------------------------- 提示词
SYSTEM = (
    "你是消防安防复核员。给你同一路监控的连续几帧画面（按时间先后排列），"
    "以及第一遍检测给出的候选框和时序统计。判断这个目标到底是什么。\n"
    "判断依据优先级：\n"
    "1) 有没有真实可燃物/火源形态（火焰有跳动、根部明亮、有热变形）\n"
    "2) 周围环境（厨房灶台、焊接工位、灯具、排气口、管道口）\n"
    "3) 时间变化（烟：缓慢累积、由白转灰黑；蒸汽：快速膨胀后迅速变淡）\n"
    "4) 位置（贴天花板的常亮光斑多半是灯；贴地面的白雾多半是蒸汽/水汽）\n"
    "严禁为了给结论而猜。看不出来就输出 uncertain。\n"
    "【判据优先级】图像看不清时，优先采信时序统计，不要凭「看起来像白色」就判蒸汽：\n"
    "· 面积增长 < 0.30/s（缓慢持续）→ 更像烟；> 0.55/s（几秒膨胀）→ 更像蒸汽\n"
    "· 亮度持续下降（负值明显）→ 烟（碳颗粒累积）；亮度上升/不变 → 蒸汽或灯\n"
    "· 边界清晰度快速变负 → 蒸汽扩散；变化不大 → 烟或火\n"
    "· 存续 > 25s → 几乎不可能是蒸汽（蒸汽短命）\n"
    "· 画面里能看到燃烧物/炉灶/焊接火花/焦黑痕迹 → 往烟倾斜"
)


def build_prompt(feat: TemporalFeatures | None, bbox: tuple | None,
                 frame_size: tuple | None, ctx_text: str = "") -> str:
    """把「框位置 + 时序统计 + 场景提示」写成文字，喂给 VLM。

    只给图片不给这些数字，模型只能靠猜 —— 实测这句话把准确率拉得很高。
    """
    lines = []

    if bbox is not None and frame_size is not None:
        x1, y1, x2, y2 = [float(v) for v in bbox[:4]]
        w, h = frame_size
        cx, cy = (x1 + x2) / 2 / max(w, 1), (y1 + y2) / 2 / max(h, 1)
        pos_v = "上部" if cy < 0.33 else ("中部" if cy < 0.67 else "下部")
        pos_h = "左侧" if cx < 0.33 else ("中部" if cx < 0.67 else "右侧")
        size = (x2 - x1) * (y2 - y1) / max(w * h, 1)
        lines.append(
            f"候选框：画面{pos_h}{pos_v}（相对中心 x={cx:.2f}, y={cy:.2f}），"
            f"占画面 {size * 100:.1f}%"
        )

    if feat is not None:
        hint = ("增长缓慢→偏烟" if feat.area_growth < 0.30
                else ("增长很快→偏蒸汽" if feat.area_growth > 0.55 else "增长中等→两可"))
        lines.append(
            "时序统计：持续 {:.1f}s；面积对数增长率 {:+.2f}/s（{}）；"
            "亮度变化 {:+.1f}/s（负=变暗，偏烟）；边界清晰度变化 {:+.3f}/s"
            "（负=越来越糊，偏蒸汽）；上升速度 {:.1f}px/s".format(
                feat.duration, feat.area_growth, hint, feat.gray_trend,
                feat.edge_trend, feat.rise_speed)
        )

    if ctx_text:
        lines.append(f"场景信息：{ctx_text}")

    lines.append(
        "只输出一个 JSON，不要任何解释文字：\n"
        '{"label":"fire|smoke|steam|light|other|uncertain",'
        '"confidence":0.0-1.0,"reason":"20字以内，必须指出画面里的依据",'
        '"risk":"high|medium|low"}'
    )
    return "\n".join(lines)


# ---------------------------------------------------------------- 复核器
class VlmVerifier:
    """带限额、去重、降级的 VLM 复核器。"""

    def __init__(self, cfg: dict | None = None, api_key: str | None = None):
        self.cfg = cfg or load_config()
        # 注意：这里必须用 is None 判断。
        # 写成 `api_key or os.environ.get(...)` 的话，显式传 "" 会被当成假值，
        # 又悄悄回退到环境变量 —— 测"无 Key 降级"时就会真的发请求（踩过）。
        self.api_key = (os.environ.get(self.cfg["api_key_env"], "")
                        if api_key is None else api_key)
        self._calls: list[float] = []          # 最近一小时调用时间戳
        self._last: dict[str, float] = {}      # 去重：key → 上次复核时间
        self._lock = threading.Lock()

    # ------------------------------------------------------------ 限额 / 去重
    def _quota_ok(self) -> bool:
        now = time.time()
        with self._lock:
            self._calls = [t for t in self._calls if now - t < 3600]
            return len(self._calls) < int(self.cfg["hourly_limit"])

    def _mark_call(self) -> None:
        with self._lock:
            self._calls.append(time.time())

    def _should_skip(self, key: str) -> bool:
        now = time.time()
        with self._lock:
            last = self._last.get(key)
            if last and now - last < float(self.cfg["dedup_seconds"]):
                return True
            self._last[key] = now
            return False

    # ------------------------------------------------------------ 主流程
    def verify(self, frames: list[np.ndarray], *, bbox: tuple | None = None,
               feat: TemporalFeatures | None = None, ctx_text: str = "",
               key: str = "") -> Verdict2:
        """frames：按时间顺序的画面（会均匀取样到 cfg.frames 张）。"""
        cfg = self.cfg
        if not cfg.get("enabled", True):
            return self._fallback(feat, bbox, frames, "复核已关闭")

        if feat is not None and feat.duration < float(cfg["min_duration_s"]):
            return self._fallback(feat, bbox, frames,
                                  f"持续 {feat.duration:.1f}s 未达 {cfg['min_duration_s']}s")

        if key and self._should_skip(key):
            return self._fallback(feat, bbox, frames, "同一目标已复核过（去重）")

        if not self.api_key:
            return self._fallback(feat, bbox, frames, "未配置 API Key")

        if not self._quota_ok():
            return self._fallback(feat, bbox, frames, "本小时调用额度已用完")

        imgs = [f for f in frames if f is not None]
        if not imgs:
            return self._fallback(feat, bbox, frames, "没有可用画面")

        pick = imgs if len(imgs) <= cfg["frames"] else [
            imgs[int(i * (len(imgs) - 1) / (cfg["frames"] - 1))] for i in range(cfg["frames"])
        ]
        uris = [u for u in (_encode(f) for f in pick) if u]
        if not uris:
            return self._fallback(feat, bbox, frames, "画面编码失败")

        size = (imgs[0].shape[1], imgs[0].shape[0])
        content = [{"type": "text", "text": build_prompt(feat, bbox, size, ctx_text)}]
        for u in uris:
            content.append({"type": "image_url", "image_url": {"url": u}})

        body = {
            "model": cfg["model"],
            "messages": [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": content},
            ],
            "temperature": cfg["temperature"],
            "max_tokens": cfg["max_tokens"],
            "thinking": {"type": cfg["thinking"]},
        }

        t0 = time.time()
        try:
            import requests
            r = requests.post(cfg["url"], json=body, timeout=cfg["timeout_s"],
                              headers={"Authorization": f"Bearer {self.api_key}",
                                       "Content-Type": "application/json"})
            latency = time.time() - t0
            self._mark_call()
            if r.status_code >= 400:
                return self._fallback(feat, bbox, frames,
                                      f"HTTP {r.status_code}", latency)
            msg = r.json().get("choices", [{}])[0].get("message", {})
            text = (msg.get("content") or "").strip()
            parsed = _parse_json(text)
            if not parsed:
                return self._fallback(feat, bbox, frames, "返回内容不是合法 JSON", latency)
            return self._to_verdict(parsed, latency)
        except Exception as exc:      # noqa: BLE001
            return self._fallback(feat, bbox, frames,
                                  f"{type(exc).__name__}", time.time() - t0)

    # ------------------------------------------------------------ 内部
    def _to_verdict(self, d: dict, latency: float) -> Verdict2:
        label = str(d.get("label", "uncertain")).strip().lower()
        if label not in self.cfg["labels"]:
            label = "uncertain"
        try:
            conf = float(d.get("confidence", 0))
        except (TypeError, ValueError):
            conf = 0.0
        conf = max(0.0, min(1.0, conf))
        # 分级处理，不要一刀切：
        #   conf < 0.35            → 真的拿不准，判 uncertain
        #   0.35 ≤ conf < 0.6      → 保留原标签，但标记"需人工确认"（needs_review）
        #   conf ≥ 0.6             → 正常采信
        # 为什么不直接把中间段丢掉：漏报火的代价远大于多看一张图。
        # 但也绝不把它当"高置信火情"直接拉警报。
        needs_review = False
        if conf < 0.35:
            label = "uncertain"
        elif conf < float(self.cfg["accept_conf"]):
            needs_review = True
            if label in ("fire", "smoke"):
                # 疑似火情：保留标签，风险降一档，交给事件层做"待确认"处理
                pass
        risk = str(d.get("risk", "")).strip().lower()
        if risk not in ("high", "medium", "low"):
            risk = "high" if label == "fire" else ("medium" if label == "smoke" else "low")
        if needs_review and risk == "high":
            risk = "medium"
        v = Verdict2(label=label, confidence=round(conf, 3),
                     reason=str(d.get("reason", ""))[:60], risk=risk,
                     source="vlm", latency_s=round(latency, 2), raw=d)
        v.needs_review = needs_review
        return v

    def _fallback(self, feat: TemporalFeatures | None, bbox, frames,
                  why: str, latency: float = 0.0) -> Verdict2:
        """降级到本地规则。规则也拿不准就老实说 uncertain。"""
        rv: RuleVerdict = judge(feat)
        label = rv.label if rv.label in ("smoke", "steam") else "uncertain"
        return Verdict2(label=label, confidence=float(rv.score or 0),
                        reason=(rv.reasons[0] if rv.reasons else why)[:60],
                        risk="medium" if label == "smoke" else "low",
                        source="rule", latency_s=round(latency, 2),
                        raw={"fallback_reason": why})


def _parse_json(text: str) -> dict | None:
    """容错解析：模型偶尔会包 ```json 或前后加话，这里只抠第一个完整对象。"""
    if not text:
        return None
    try:
        return json.loads(text)
    except Exception:
        pass
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None
