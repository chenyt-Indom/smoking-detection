"""异步 LLM 裁判：YOLO 出可疑 → 后台送审 → 裁决回来再定级。

为什么要独立成一层，而不是像 `verifier.py` 那样同步调用
------------------------------------------------------
这是设计上的硬约束，不是风格问题：

· **同步调用的代价**：`requests.post` 一次 1~1.5 秒。而流水线在 15fps 时每帧只有
  65ms 预算。同步调一次 = 硬卡掉 20 帧视频。卡住的这 1.5 秒里如果真起火，
  后面本来能看见的帧全丢了 —— 为了"看清楚"反而"看不见"，得不偿失。
· **所以这里全部异步**：`submit()` 只是把任务扔进线程池（微秒级返回），
  `poll()` 每帧收割已完成的结果。推理循环永远不等网络。

为什么只送「可疑」而不是每帧
----------------------------
用户的原话是不能在前台实时判断，太耗钱太慢。按 15fps 每帧送审算：
  15 帧/秒 × 3600 秒 × 约 300 token ≈ 1600 万 token/小时/路
这是纯烧钱。而火灾是**事件**不是**流**：一条轨迹只需要判一次。
一条烟审 1 次，成本降到万分之一，延迟还更低（不用排队）。

三张图而不是一张
----------------
单帧送审等于退回"单帧识别"——白烟和蒸汽在单帧上同色同形，模型只能瞎猜。
所以每次送：
  图1 整幅画面（提供环境：厨房灶台？车间？天花板？）
  图2 可疑区域放大（提供细节：有没有火舌、焦黑、颗粒感）
这两张图加起来 token 和一张 1080p 原图差不多，但信息量完全不同。

降级绝不阻塞告警
----------------
没 Key / 超额 / 超时 / 返回脏数据 → `source="rule"`，回退到本地规则层的判断。
**回退时会保守放行**（宁可报出来让人看一眼），因为漏报火的代价远大于多看一张图。
"""
from __future__ import annotations

import json
import os
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

# ---------------------------------------------------------------- 配置
DEFAULTS = {
    "enabled": True,
    "dry_run": False,              # True = 不发请求，返回假裁决（接线验证用）
    "url": "https://ai.ctaigw.cn/v1/chat/completions",
    "model": "",                   # 必填：平台模型广场里的具体模型名
    "api_key_env": "TELEAI_APPKEY",
    "temperature": 0.1,
    "max_tokens": 160,             # 只要一个短 JSON，给多了纯浪费
    "timeout_s": 8,
    "roi_max_side": 512,           # ROI 放大后的长边
    "full_max_side": 768,          # 整幅图的长边（太大既慢又贵）
    "jpeg_quality": 82,
    "max_workers": 2,              # 并发上限。别调高：并发多了反而被限流
    "queue_limit": 8,              # 排队超过这个数就跳过（宁可少审也别堆积）
    "hourly_limit": 300,           # 每小时调用上限，超了自动降级
    "dedup_seconds": 180,          # 同一目标 3 分钟内只审一次
    "accept_conf": 0.55,           # 采信门限
}

SYSTEM = (
    "你是消防安防值班员，正在看一路监控。图1是整幅画面（判断环境），"
    "图2是可疑区域放大（判断细节）。\n"
    "任务：判断这个可疑目标是否构成火灾风险。\n\n"
    "判据：\n"
    "- 厨房灶台、焊接工位、排气口、水壶上方的白色羽流 → 水汽/工艺气，风险低\n"
    "- 天花板常亮光斑、灯具、反光 → 灯光，风险低\n"
    "- 有明火舌、根部明亮、火光映在墙面、焦黑痕迹 → fire，风险高\n"
    "- 灰/黑色翻滚上升、贴天花板横向扩散、地面黑色堆积 → smoke，风险高\n"
    "- 脱离常见场景、看不出来 → uncertain，confidence 给低\n\n"
    "只输出一个 JSON，不要任何解释文字，不要 markdown 代码块：\n"
    '{"label":"fire|smoke|steam|light|other|uncertain",'
    '"risk":"high|medium|low","confidence":0.0-1.0,'
    '"reason":"20字以内，必须指出画面里的具体依据"}'
)


def load_config(path: str | Path | None = None) -> dict:
    """读 config/arbiter.yaml，缺的用 DEFAULTS 补。"""
    cfg = dict(DEFAULTS)
    p = Path(path) if path else Path(__file__).resolve().parent.parent / "config" / "arbiter.yaml"
    if p.exists():
        try:
            import yaml
            cfg.update({k: v for k, v in (yaml.safe_load(p.read_text(encoding="utf-8")) or {}).items()
                        if v is not None})
        except Exception:
            pass
    return cfg


def load_dotenv(path: str | Path | None = None) -> None:
    """把 .env 里的 KEY=VALUE 灌进环境变量（已存在的环境变量不覆盖）。

    只支持最简单的 KEY=VALUE，不引 python-dotenv —— 少一个依赖少一个部署坑。
    """
    p = Path(path) if path else Path(__file__).resolve().parent.parent / ".env"
    if not p.exists():
        return
    try:
        for ln in p.read_text(encoding="utf-8", errors="ignore").splitlines():
            ln = ln.strip()
            if not ln or ln.startswith("#") or "=" not in ln:
                continue
            k, v = ln.split("=", 1)
            k, v = k.strip(), v.strip().strip('"').strip("'")
            if k and k not in os.environ:
                os.environ[k] = v
    except OSError:
        pass


# ---------------------------------------------------------------- 结果
@dataclass
class ArbiterResult:
    """一次裁决的结论。属性和 Verdict2 有意保持一致，便于互换。"""

    label: str                     # fire / smoke / steam / light / other / uncertain
    risk: str = "low"              # high / medium / low
    confidence: float = 0.0
    reason: str = ""
    source: str = "llm"            # llm / rule / dry_run
    latency_s: float = 0.0
    raw: dict = field(default_factory=dict)

    @property
    def is_alarm(self) -> bool:
        """够格直接拉警报。"""
        return self.label in ("fire", "smoke") and self.confidence >= 0.55

    @property
    def is_dismiss(self) -> bool:
        """明确不是火灾风险，可以压掉。"""
        return self.label in ("steam", "light", "other")

    def __str__(self) -> str:
        return (f"{self.label}/{self.risk} conf={self.confidence:.2f} "
                f"[{self.source} {self.latency_s:.1f}s] {self.reason}")


def _encode(img: np.ndarray, max_side: int, quality: int) -> str | None:
    """BGR → data URI（JPEG）。控制长边是为了压 token，不是为了省流量。"""
    if img is None or img.size == 0:
        return None
    h, w = img.shape[:2]
    s = min(1.0, max_side / max(h, w))
    if s < 1.0:
        img = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        return None
    import base64
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode()


def _parse_json(text: str) -> dict | None:
    """模型经常裹一层 ```json，或者前面带一句话。抓第一个 { ... }。"""
    if not text:
        return None
    t = text.strip()
    if t.startswith("```"):
        t = t.split("\n", 1)[-1]
        if t.rstrip().endswith("```"):
            t = t.rstrip()[:-3]
    i, j = t.find("{"), t.rfind("}")
    if i < 0 or j <= i:
        return None
    try:
        d = json.loads(t[i:j + 1])
        return d if isinstance(d, dict) else None
    except json.JSONDecodeError:
        return None


def _crop(image: np.ndarray, box, margin: float = 0.35) -> np.ndarray | None:
    """按框裁图并外扩一点 —— 只看框内会丢掉"周围是什么环境"这个最强线索。"""
    h, w = image.shape[:2]
    x1, y1, x2, y2 = [float(v) for v in box[:4]]
    bw, bh = max(1.0, x2 - x1), max(1.0, y2 - y1)
    x1 = int(max(0, x1 - margin * bw)); x2 = int(min(w, x2 + margin * bw))
    y1 = int(max(0, y1 - margin * bh)); y2 = int(min(h, y2 + margin * bh))
    if x2 <= x1 or y2 <= y1:
        return None
    return image[y1:y2, x1:x2].copy()


# ---------------------------------------------------------------- 裁判
class LlmArbiter:
    """线程池版异步裁判。线程安全，可以直接挂到多路流水线上。"""

    def __init__(self, cfg: dict | None = None, api_key: str | None = None):
        # 显式传入的 cfg 只做**覆盖**，不整份替换 —— 否则少写一个键就 KeyError
        self.cfg = {**load_config(), **(cfg or {})}
        if api_key is None:
            load_dotenv()
            api_key = os.environ.get(self.cfg["api_key_env"], "")
        # 用 is None 判断，别写 `api_key or env`：显式传 "" 会被悄悄换回环境变量，
        # 测"无 Key 降级"时会真的发请求（verifier.py 里踩过同样的坑）。
        self.api_key = api_key or ""
        self._pool = ThreadPoolExecutor(max_workers=int(self.cfg["max_workers"]),
                                        thread_name_prefix="arbiter")
        self._futures: dict[str, Future] = {}
        self._calls: list[float] = []
        self._last: dict[str, float] = {}
        self._lock = threading.Lock()
        self.stats = {"submitted": 0, "llm": 0, "rule": 0, "skipped": 0,
                      "timeout": 0, "error": 0}

    # ------------------------------------------------------------ 额度 / 去重
    def _quota_ok(self) -> bool:
        now = time.time()
        with self._lock:
            self._calls = [t for t in self._calls if now - t < 3600]
            return len(self._calls) < int(self.cfg["hourly_limit"])

    def _mark_call(self) -> None:
        with self._lock:
            self._calls.append(time.time())

    def _dup(self, key: str) -> bool:
        now = time.time()
        with self._lock:
            last = self._last.get(key)
            if last is not None and now - last < float(self.cfg["dedup_seconds"]):
                return True
            self._last[key] = now
            return False

    # ------------------------------------------------------------ 对外
    def submit(self, key: str, frame: np.ndarray, box, *,
               rule_label: str = "uncertain", rule_score: float = 0.0,
               context: str = "", feat=None) -> tuple[bool, str]:
        """提交一次裁决。立即返回，不等网络。

        返回 (accepted, reason)：
          accepted=True  → 调用方应该**暂缓**定级，等 poll() 拿到结果再决定
          accepted=False → 调用方**立刻按原有规则逻辑走**（不要因为裁判不可用就漏报）
        """
        if not self.cfg.get("enabled", True):
            return False, "裁判已关闭"
        if not self.api_key and not self.cfg.get("dry_run"):
            return False, "未配置 API Key"
        if not self.cfg.get("model") and not self.cfg.get("dry_run"):
            return False, "未配置 model（去 config/arbiter.yaml 填）"
        if self._dup(key):
            return False, "同一目标已审过"
        if not self._quota_ok():
            self.stats["skipped"] += 1
            return False, "本小时额度已用完"
        with self._lock:
            alive = {k: f for k, f in self._futures.items() if not f.done()}
            self._futures = alive
            if len(alive) >= int(self.cfg["queue_limit"]):
                self.stats["skipped"] += 1
                return False, "排队已满"

        # 主线程只做裁剪（约 1ms），编码和网络都扔进线程池
        roi = _crop(frame, box)
        payload = (frame, roi, float(rule_score), str(rule_label), str(context), feat)
        with self._lock:
            self._futures[key] = self._pool.submit(self._run, key, payload)
        self.stats["submitted"] += 1
        return True, "已提交"

    def poll(self) -> list[tuple[str, ArbiterResult]]:
        """收割已完成的结果。每帧调一次，无结果时开销接近 0。"""
        out: list[tuple[str, ArbiterResult]] = []
        with self._lock:
            done = [(k, f) for k, f in self._futures.items() if f.done()]
            for k, _ in done:
                self._futures.pop(k, None)
        for k, f in done:
            try:
                out.append((k, f.result()))
            except Exception as exc:                     # noqa: BLE001
                out.append((k, ArbiterResult("uncertain", "low", 0.0,
                                             f"任务异常 {type(exc).__name__}", "rule")))
                self.stats["error"] += 1
        return out

    def pending(self) -> int:
        with self._lock:
            return sum(1 for f in self._futures.values() if not f.done())

    def close(self) -> None:
        self._pool.shutdown(wait=False)

    # ------------------------------------------------------------ 线程内
    def _run(self, key: str, payload) -> ArbiterResult:
        frame, roi, rule_score, rule_label, context, feat = payload
        cfg = self.cfg

        if cfg.get("dry_run"):
            time.sleep(0.05)                             # 模拟网络，验证异步不卡主线程
            self.stats["llm"] += 1
            return ArbiterResult(rule_label if rule_label != "uncertain" else "smoke",
                                 "high" if rule_label in ("fire", "smoke") else "low",
                                 max(rule_score, 0.6), "dry-run 假裁决", "dry_run", 0.05)

        imgs = []
        if roi is not None:
            imgs.append(_encode(roi, int(cfg["roi_max_side"]), int(cfg["jpeg_quality"])))
        imgs.append(_encode(frame, int(cfg["full_max_side"]), int(cfg["jpeg_quality"])))
        uris = [u for u in imgs if u]
        if not uris:
            return self._fallback(rule_label, rule_score, "画面编码失败")

        text = self._user_text(roi, frame, context, feat)
        content = [{"type": "text", "text": text}]
        for u in uris:
            content.append({"type": "image_url", "image_url": {"url": u}})

        body = {
            "model": cfg["model"],
            "messages": [{"role": "system", "content": SYSTEM},
                         {"role": "user", "content": content}],
            "temperature": cfg["temperature"],
            "max_tokens": cfg["max_tokens"],
        }

        t0 = time.time()
        try:
            import requests
            r = requests.post(cfg["url"], json=body, timeout=float(cfg["timeout_s"]),
                              headers={"Authorization": f"Bearer {self.api_key}",
                                       "Content-Type": "application/json",
                                       # 少了 User-Agent 会被平台安全护栏拦，
                                       # 而且错误长得跟鉴权失败一样
                                       "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})
            latency = time.time() - t0
            self._mark_call()
            if r.status_code >= 400:
                self.stats["error"] += 1
                return self._fallback(rule_label, rule_score,
                                      f"HTTP {r.status_code} {r.text[:100]}", latency)
            msg = r.json().get("choices", [{}])[0].get("message", {})
            parsed = _parse_json((msg.get("content") or "").strip())
            if not parsed:
                self.stats["error"] += 1
                return self._fallback(rule_label, rule_score, "返回不是合法 JSON", latency)
            self.stats["llm"] += 1
            return self._to_result(parsed, latency)
        except Exception as exc:                          # noqa: BLE001
            self.stats["timeout"] += 1
            return self._fallback(rule_label, rule_score,
                                  f"{type(exc).__name__}", time.time() - t0)

    @staticmethod
    def _user_text(roi, frame, context: str, feat) -> str:
        lines = []
        if roi is not None and frame is not None:
            fh, fw = frame.shape[:2]
            rh, rw = roi.shape[:2]
            lines.append(f"可疑区域占整幅画面约 {rw * rh / max(fw * fh, 1) * 100:.1f}%，"
                         f"位置在画面{_pos_text(rh, rw, fh, fw)}")
        if feat is not None:
            lines.append(f"该目标已持续 {feat.duration:.1f}s，"
                         f"面积对数增长率 {feat.area_growth:+.2f}/s，"
                         f"亮度变化 {feat.gray_trend:+.1f}/s，"
                         f"清晰度变化 {feat.edge_trend:+.3f}/s")
        lines.append(f"第一遍检测给出：{context or '疑似火/烟'}")
        lines.append("请判断是否有火灾风险。")
        return "\n".join(lines)

    def _to_result(self, d: dict, latency: float) -> ArbiterResult:
        label = str(d.get("label", "uncertain")).strip().lower()
        if label not in ("fire", "smoke", "steam", "light", "other", "uncertain"):
            label = "uncertain"
        risk = str(d.get("risk", "low")).strip().lower()
        if risk not in ("high", "medium", "low"):
            risk = "low"
        try:
            conf = max(0.0, min(1.0, float(d.get("confidence", 0))))
        except (TypeError, ValueError):
            conf = 0.0
        # 置信度太低时不要硬信标签 —— 但也不丢，交给调用方当"疑似"处理
        if conf < float(self.cfg["accept_conf"]) and label in ("fire", "smoke"):
            risk = "medium"
        return ArbiterResult(label, risk, conf, str(d.get("reason", ""))[:60], "llm", latency, d)

    def _fallback(self, rule_label: str, rule_score: float, why: str,
                  latency: float = 0.0) -> ArbiterResult:
        """裁判不可用 → 退回规则层。**保守放行**：宁可报出来让人看一眼。"""
        self.stats["rule"] += 1
        return ArbiterResult(rule_label or "uncertain", "medium", float(rule_score),
                             f"裁判不可用（{why}），按规则层结论处理", "rule", latency)


def _pos_text(rh: int, rw: int, fh: int, fw: int) -> str:
    """这里只能给出尺寸占比，位置在调用方已经通过框坐标表达，不重复算。"""
    return f"{'偏大' if rw * rh / max(fw * fh, 1) > 0.15 else '中等'}"
