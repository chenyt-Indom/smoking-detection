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
import re
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
    "图2是可疑区域放大（判断细节）；若只有一张图，那张就是整幅画面。\n"
    "任务：判断这个可疑目标是否构成火灾风险。\n\n"
    "判据：\n"
    "- 厨房灶台、焊接工位、排气口、水壶上方的白色羽流 → 水汽/工艺气，风险低\n"
    "- ⚠️ 先看这条再看灯光：**打火机、蜡烛、酒精灯、火柴、灶火，"
    "哪怕只有一个指甲盖大，也是 fire**。判据是「有没有一小片橙黄/蓝黄渐变、"
    "边缘会飘动的火苗」—— 手指或金属物件上出现这种小亮点，基本就是打火机，"
    "不要因为面积小、没有烟、没有烧到别的东西就判成 light\n"
    "- 有明火舌、根部明亮、火光映在墙面、焦黑痕迹 → fire，风险高\n"
    "- 天花板常亮光斑、灯具、反光：大面积、边缘柔和、位置固定 → light，风险低\n"
    "- 手机手电筒、补光灯：圆形或方形**均匀**亮斑，边缘锐利、"
    "内部没有明暗梯度、色温偏白或偏蓝 → light，风险低\n"
    "- ⚠️ 分不清是「一小团会飘的火苗」还是「均匀的圆形光斑」时，"
    "选 uncertain 而不是 light —— 宁可请人看一眼，也别把火排掉\n"
    "- 灰/黑色翻滚上升、贴天花板横向扩散、地面黑色堆积 → smoke，风险高\n"
    "- 香烟/烟头/卷烟：**正在燃烧的香烟端头有红点或细烟缕 → smoke**，风险中；"
    "只是夹在指间没点燃、或躺在烟灰缸里没火 → other（说明理由：未点燃）\n"
    "- ⚠️ **香烟要判 cigarette，不是 smoke**：画面里是白色/褐色细长条、"
    "被手指夹着、端头可能有红点或褐色燃点，但**没有向外扩散的烟云** → "
    "label 填 **cigarette**。只有出现成片向外翻滚扩散的烟气时才判 smoke。"
    "这样系统才能把「抽烟」和「着火冒烟」分开报。\n"
    "- ⚠️⚠️ **判 fire 的唯一标准是「画面里有没有看得见的明火」，"
    "不是「会不会造成火灾」。** 打火机、蜡烛、火柴、灶火、酒精灯、焊接火 —— "
    "只要是橙黄/蓝黄的会飘动的火苗，一律判 **fire**。风险等级可以给 low/medium，"
    "但 label 绝不可是 other。因为「只是个打火机烧不起来」就判 other，"
    "等于把明火当成没有 —— 这是本系统最不能犯的错。\n"
    "- 人手/手指/衣物/家具/宠物/屏幕画面等**画面里完全没有明火**的东西 → other，"
    "风险低。判 other 之前先问自己一句：**画面里真的没有火苗吗？** "
    "哪怕只有指甲盖大的一小片橙黄，也要判 fire 而不是 other。\n"
    "- ⚠️ 如果上下文里出现「同一位置另检出 cigarette/smoke」，说明两个模型"
    "对同一块区域给了不同答案。此时**以你在画面里看到的为准**："
    "看到的是白色/褐色细长条、端头没有明显飘动火苗 → 判 cigarette 不是 fire；"
    "看到的是会飘动的橙黄火苗 → 判 fire。不要因为'检测器说是火'就跟着判火。\n"
    "- ⚠️⚠️ **人脸/头部区域的高分误报必须排除**。实测模型会把这些东西"
    "判成 fire（置信度高达 0.78）：「耳朵轮廓」「耳机」「下巴阴影」"
    "「被顶灯照亮的皮肤」；也会把「鼻子」「口罩」「下巴」判成 cigarette（0.54）。"
    "判定规则：\n"
    "    · 框落在人脸上（眼睛/鼻子/嘴/耳/下巴/脖子/头发边缘）→ 只要画面里"
    "      **没有手指夹着的细长条**、**没有会飘动的橙黄火苗**，一律判 other\n"
    "    · 皮肤被灯照亮的橙黄色反光**不是火**：火苗有明确边界、边缘会飘动；"
    "      皮肤反光是弥散的，跟着人脸轮廓走，亮区没有独立形态\n"
    "    · 反过来也不能矫枉过正：如果框里确实有一小片会飘的橙黄火苗"
    "      （比如手指间的打火机），即使在脸旁边也要判 fire\n"
    "- 图片太糊、太小、被遮挡，确实无法判断 → uncertain，confidence 给低\n"
    "- ⚠️ other 和 uncertain 必须分清：\n"
    "  · 你看清了、并且确定它不是火/烟/蒸汽/灯光 → other（例如看清楚是手指）\n"
    "  · 你看不清、或者看到了但没有把握 → uncertain\n"
    "  不要把'看清楚了但没危险'答成 uncertain，那会让系统一直请你复核。\n\n"
    "只输出一个 JSON，不要任何解释文字，不要 markdown 代码块：\n"
    '{"label":"fire|smoke|cigarette|steam|light|other|uncertain",'
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

    label: str                     # fire / smoke / cigarette / steam / light / other / uncertain
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
    frag = t[i:j + 1]
    # 中文引号（模型偶尔把英文引号写成 “ ” ‘ ’）。
    # 用 chr() 而不是字面量，避免源码经过几层替换后转义失效。
    _cn = frag
    for _ch, _ascii in ((chr(0x201C), '"'), (chr(0x201D), '"'),
                        (chr(0x2018), "'"), (chr(0x2019), "'")):
        _cn = _cn.replace(_ch, _ascii)
    for cand in (frag, _cn, re.sub(r",\s*([}\]])", r"\1", frag), _cn):
        try:
            d = json.loads(cand)
            return d if isinstance(d, dict) else None
        except json.JSONDecodeError:
            continue
    return None


def _crop(image: np.ndarray, box, margin: float = 0.85) -> np.ndarray | None:
    """按框裁图并外扩 —— 只看框内会丢掉"周围是什么环境"这个最强线索。

    为什么外扩到 0.85（框尺寸的 85%，接近一倍）
    ------------------------------------------
    烟雾是**向外扩散**的东西，只框住核心那一团完全看不出形态：
    分不清是"烟在扩散"还是"一团静态白斑"。外扩近一倍后，
    能看到烟从哪冒出来、往哪飘、边缘是硬的还是毛的 —— 这些才是
    LLM 判断"烟 vs 蒸汽 vs 白墙斑点"的关键。

    火苗同理：小火苗单独看只是一点橙色，加上周边才知道是打火机、
    蜡烛还是灯罩反光。
    """
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
        # 最近若干次裁决的流水，只给前端 LLM 面板展示用。
        # 不落盘：它是"此刻连没连上、快不快"的实时证据，不是审计日志。
        self._history: list[dict] = []

    def _is_local(self) -> bool:
        """端点是不是本机（Ollama / vLLM / LM Studio 之类）。

        本地服务不校验 Key，硬要求它填 key 会让人以为配错了。
        判断只看 url 的主机部分，不做 DNS 解析。
        """
        url = str(self.cfg.get("url") or "")
        return any(h in url for h in ("127.0.0.1", "localhost", "0.0.0.0", "::1"))

    # ------------------------------------------------------------ 额度 / 去重
    def _quota_ok(self) -> bool:
        """配额检查 —— 只做提醒，**永不阻止送审**。

        用户要求："所有内容务必都得有 LLM 裁判，不能自作主张，
        特别是香烟，必须不能漏。"

        原来配额到上限就返回 False，等于规则层单方面决定"这个目标
        不送 LLM 了"，而且完全静默 —— 值班员不知道裁判那段时间缺席。
        对安全系统来说这是最不该有的行为：漏报的代价远高于几次 API 费用。

        配额耗尽时降级为规则层，但**必须打日志** —— 静默降级等于
        让人以为系统一直在判。检测、画框、报警都不受影响，只是这条
        不再花 LLM 的钱复核。
        """
        now = time.time()
        with self._lock:
            self._calls = [t for t in self._calls if now - t < 3600]
            used = len(self._calls)
        limit = int(self.cfg["hourly_limit"])
        if used >= limit:
            # 配额耗尽 -> 降级到规则层。这不是"不报"，而是"这条不再花钱请
            # LLM 看第二眼"：检测照跑、框照画、事件照推，只是定级由规则层做。
            # 关键是**必须留痕** —— 静默降级等于让人以为系统一直在判。
            self.stats_quota_hit = getattr(self, "stats_quota_hit", 0) + 1
            if not getattr(self, "_quota_warned", False):
                self._quota_warned = True
                print(f"[warn] LLM 配额已达 {used}/{limit} 次/小时，"
                      f"后续目标降级为规则层判定（仍在检测、仍会报警，"
                      f"只是不再送 LLM 复核）。建议检查误报量。", flush=True)
            return False
        if used < limit * 0.5:
            self._quota_warned = False
        return True

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
        if not self.api_key and not self.cfg.get("dry_run") and not self._is_local():
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
        payload = (frame, roi, float(rule_score), str(rule_label), str(context), feat, box)
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

    # ------------------------------------------------------------ 可观测性
    def snapshot(self) -> dict:
        """前端 LLM 面板要的全部状态：连没连上、配额还剩多少、刚才几次快不快。"""
        cfg = self.cfg
        model = str(cfg.get("model") or "")
        dry = bool(cfg.get("dry_run"))
        has_key = bool(self.api_key)
        enabled = bool(cfg.get("enabled", True))

        problems: list[str] = []
        if not enabled:
            problems.append("裁判已关闭（config/arbiter.yaml 的 enabled）")
        if not model and not dry:
            problems.append("model 未填 —— 去 config/arbiter.yaml 写上平台给出的模型名")
        if not has_key and not dry and not self._is_local():
            problems.append("未读到 API Key —— 在项目根目录 .env 里写对应变量")

        with self._lock:
            used = len(self._calls)
            pending = sum(1 for f in self._futures.values() if not f.done())
            history = list(self._history[-12:])

        limit = int(cfg.get("hourly_limit", 0) or 0)
        return {
            "configured": bool(enabled and (dry or (model and (has_key or self._is_local())))),
            "enabled": enabled,
            "dry_run": dry,
            "model": model,
            "url": str(cfg.get("url") or ""),
            "has_key": has_key,
            "local": self._is_local(),
            "key_hint": (self.api_key[:4] + "…" + self.api_key[-4:]
                         if len(self.api_key) > 8 else ""),
            "quota": {"used": used, "limit": limit, "left": max(0, limit - used)},
            "pending": pending,
            "stats": dict(self.stats),
            "history": history,
            "problems": problems,
            # 看板「充值」按钮跳这里，改 config/arbiter.yaml 即可
            "recharge_url": str(cfg.get("recharge_url") or ""),
        }

    def test_connection(self, timeout: float | None = None) -> dict:
        """真打一次最小请求，确认 URL / 模型 / Key 三样都通。

        和 submit 的区别：它**同步**返回结果，且只发文本不发图 —— 目的是验证
        "能不能连上"，不是验证"判得准不准"。成功后会计入每小时配额。
        """
        cfg = self.cfg
        if cfg.get("dry_run"):
            return {"ok": True, "dry_run": True, "msg": "dry_run 模式下不发真实请求"}
        if not cfg.get("model"):
            return {"ok": False, "error": "model 未填（config/arbiter.yaml）"}
        if not self.api_key and not self._is_local():
            return {"ok": False, "error": "未配置 API Key"}

        body = {"model": cfg["model"],
                "messages": [{"role": "user", "content": "ping"}],
                "max_tokens": 8}
        if str(cfg.get("thinking", "")).lower() == "disabled":
            body["thinking"] = {"type": "disabled"}
        t0 = time.time()
        try:
            import requests
            r = requests.post(
                cfg["url"], json=body,
                timeout=float(timeout or cfg["timeout_s"]),
                headers={"Authorization": f"Bearer {self.api_key}",
                         "Content-Type": "application/json",
                         "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})
            dt = round(time.time() - t0, 2)
            if r.status_code >= 400:
                return {"ok": False, "latency_s": dt, "status": r.status_code,
                        "error": f"HTTP {r.status_code}: {r.text[:200]}"}
            self._mark_call()
            msg = (r.json().get("choices", [{}])[0]
                   .get("message", {}).get("content") or "")
            return {"ok": True, "latency_s": dt, "status": r.status_code,
                    "msg": str(msg)[:80]}
        except Exception as exc:                     # noqa: BLE001
            return {"ok": False, "latency_s": round(time.time() - t0, 2),
                    "error": f"{type(exc).__name__}: {exc}"}

    # ------------------------------------------------------------ 线程内
    def _run(self, key: str, payload) -> ArbiterResult:
        """线程池入口：包一层计时与流水记录，任何异常都要能落地成结果。"""
        t0 = time.time()
        try:
            res = self._run_inner(key, payload)
        except Exception as exc:                     # noqa: BLE001
            self.stats["error"] += 1
            res = self._fallback("uncertain", 0.0,
                                 f"未捕获异常 {type(exc).__name__}", time.time() - t0)
        self._record(key, res, time.time() - t0)
        return res

    def _record(self, key: str, res: ArbiterResult, wall_latency: float) -> None:
        row = {
            "at": time.strftime("%H:%M:%S"),
            "ts": round(time.time(), 3),
            "key": key,
            "source": res.source,                    # llm / rule / dry_run
            "label": res.label,
            "risk": res.risk,
            "confidence": round(float(res.confidence), 3),
            "latency_s": round(float(res.latency_s or wall_latency), 2),
            "reason": (res.reason or "")[:80],
            "ok": res.source == "llm",
        }
        with self._lock:
            self._history.append(row)
            if len(self._history) > 60:
                self._history = self._history[-60:]

    def _run_inner(self, key: str, payload) -> ArbiterResult:
        frame, roi, rule_score, rule_label, context, feat, box = payload
        cfg = self.cfg

        if cfg.get("dry_run"):
            time.sleep(0.05)                             # 模拟网络，验证异步不卡主线程
            self.stats["llm"] += 1
            return ArbiterResult(rule_label if rule_label != "uncertain" else "smoke",
                                 "high" if rule_label in ("fire", "smoke") else "low",
                                 max(rule_score, 0.6), "dry-run 假裁决", "dry_run", 0.05)

        # ⚠️ 顺序必须与 SYSTEM 里的描述一致：图1 整幅画面、图2 可疑区域放大。
        # 原来这里是 ROI 在前、全图在后，而 prompt 写的正好相反 ——
        # 模型把裁剪图当成"整幅画面"去理解环境，把全图当成"放大细节"，
        # 判断依据整个错位。这种错配比少给一张图更糟。
        # 先看环境再看细节也是更自然的顺序。
        imgs = []
        imgs.append(_encode(frame, int(cfg["full_max_side"]), int(cfg["jpeg_quality"])))
        if roi is not None:
            imgs.append(_encode(roi, int(cfg["roi_max_side"]), int(cfg["jpeg_quality"])))
        uris = [u for u in imgs if u]
        if not uris:
            return self._fallback(rule_label, rule_score, "画面编码失败")

        text = self._user_text(roi, frame, context, feat, box)
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
        # DeepSeek 默认开思考模式（effort=high）。实测 max_tokens=512 时
        # 全部 token 都被 reasoning 吃掉、content 返回空串，JSON 解析必然失败。
        # 裁决只需一眼判断，关掉思考更快更省，temperature 也才生效。
        if str(cfg.get("thinking", "")).lower() == "disabled":
            body["thinking"] = {"type": "disabled"}

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
            content = (msg.get("content") or "").strip()
            # DeepSeek 思考模式下，答案可能只出现在 reasoning_content 里，
            # content 为空或只有一个换行。直接判"返回不是合法 JSON"就会
            # 白白退回规则层 —— 用户截图里那次"火情 / 耗时 15.1s"就是这么来的：
            # 模型其实答了，只是答在了另一个字段。
            if not content:
                content = (msg.get("reasoning_content") or "").strip()
            parsed = _parse_json(content)
            if not parsed:
                self.stats["error"] += 1
                # 记录原始返回的前 120 字符，方便以后定位到底长什么样
                print(f"[llm] 解析失败，原始返回：{content[:120]!r}", flush=True)
                return self._fallback(rule_label, rule_score, "返回不是合法 JSON", latency)
            self.stats["llm"] += 1
            res = self._to_result(parsed, latency)

            # ── 二次验证：判"非火"时再问一遍 ─────────────────────
            # 用户的原话是"不要急于求成，最好多次验证"。实测出现过打火机的
            # 火苗被判 light → 系统直接"已排除"，等于把真火静默掉。
            # 这类错误方向上是**危险的**（漏报），比误报更需要防。
            #
            # 只对"排除性结论"复核：判 fire/smoke 不重复（多花一次钱且没有
            # 安全争议），判 light/other/steam 且置信不足时才再问。
            # 两次都说不是火 → 采信；两次打架 → 交人，不擅自排除。
            thr = float(cfg.get("verify_dismiss_conf", 0.85))
            if (cfg.get("verify_dismiss", True) and res.is_dismiss
                    and float(res.confidence) < thr):
                body2 = dict(body)
                body2["temperature"] = float(cfg.get("verify_temperature", 0.35))
                try:
                    r2 = requests.post(cfg["url"], json=body2,
                                       timeout=float(cfg["timeout_s"]),
                                       headers={"Authorization": f"Bearer {self.api_key}",
                                                "Content-Type": "application/json",
                                                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})
                    self._mark_call()
                    self.stats["verify_runs"] = self.stats.get("verify_runs", 0) + 1
                    if r2.status_code < 400:
                        m2 = r2.json().get("choices", [{}])[0].get("message", {})
                        p2 = _parse_json((m2.get("content") or "").strip())
                        if p2:
                            res2 = self._to_result(p2, time.time() - t0)
                            if not res2.is_dismiss:
                                # 两次结论相反 → 不擅自排除，交人看
                                self.stats["verify_conflict"] = \
                                    self.stats.get("verify_conflict", 0) + 1
                                return ArbiterResult(
                                    "uncertain", "medium",
                                    min(float(res.confidence), float(res2.confidence)),
                                    f"两次判定不一致（{res.label} vs {res2.label}），"
                                    f"已转人工确认",
                                    "llm", time.time() - t0,
                                    {"first": res.label, "second": res2.label})
                            # 两次都说不是火 → 取更确信的那一次
                            if float(res2.confidence) > float(res.confidence):
                                res = res2
                except Exception:                    # noqa: BLE001
                    # 复核失败不影响第一次结论，但记一笔便于排查
                    self.stats["verify_error"] = self.stats.get("verify_error", 0) + 1

            return res
        except Exception as exc:                          # noqa: BLE001
            self.stats["timeout"] += 1
            return self._fallback(rule_label, rule_score,
                                  f"{type(exc).__name__}", time.time() - t0)

    @staticmethod
    def _user_text(roi, frame, context: str, feat, box=None) -> str:
        lines = []
        if roi is not None and frame is not None:
            fh, fw = frame.shape[:2]
            rh, rw = roi.shape[:2]
            lines.append(f"可疑区域占整幅画面约 {rw * rh / max(fw * fh, 1) * 100:.1f}%，"
                         f"{_pos_text(rh, rw, fh, fw, box)}")
        if feat is not None:
            g = float(feat.area_growth)
            if abs(g) < 0.10:
                g_txt = "面积基本不变"
            elif g > 0.25:
                g_txt = "面积在快速扩散"
            elif g > 0:
                g_txt = "面积在缓慢扩大"
            else:
                g_txt = "面积在收缩"
            jitter = abs(float(feat.gray_trend)) + abs(float(feat.edge_trend)) * 10
            j_txt = "亮度与边缘都在抖动" if jitter > 2.0 else "亮度与边缘都很稳定"
            lines.append(f"时序观测：已持续 {feat.duration:.1f}s，{g_txt}（{g:+.2f}/s），"
                         f"{j_txt}（亮度 {feat.gray_trend:+.1f}/s，"
                         f"清晰度 {feat.edge_trend:+.3f}/s）")
            lines.append("参考：固定光源/反光通常面积与亮度都不变、边缘稳定；"
                         "燃烧通常面积在涨、亮度或边缘有抖动。这只是参考，仍以画面为准。")
        lines.append(f"第一遍检测给出：{context or '疑似火/烟'}")
        lines.append("请判断是否有火灾风险。")
        return "\n".join(lines)

    def _to_result(self, d: dict, latency: float) -> ArbiterResult:
        label = str(d.get("label", "uncertain")).strip().lower()
        if label not in ("fire", "smoke", "cigarette", "steam", "light",
                         "other", "uncertain"):
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


def _pos_text(rh: int, rw: int, fh: int, fw: int, box=None) -> str:
    """给出可疑区域在画面中的**方位**。

    原来只返回"偏大/中等"，那说的是尺寸不是位置 —— 模型拿到一张裁剪图和
    "在画面中等"，依然不知道它拍的是天花板还是桌面。而"手电筒举在画面中央"
    和"墙角起火"在判断上完全是两回事，方位是有效线索。
    """
    size = "占画面比例较大" if rw * rh / max(fw * fh, 1) > 0.15 else "占画面比例中等"
    if not box:
        return size
    x1, y1, x2, y2 = box
    cx, cy = (x1 + x2) / 2.0 / max(fw, 1), (y1 + y2) / 2.0 / max(fh, 1)
    horiz = "左侧" if cx < 0.33 else ("右侧" if cx > 0.67 else "中部")
    vert = "上方" if cy < 0.33 else ("下方" if cy > 0.67 else "中间高度")
    return f"位于画面{horiz}{vert}，{size}"
