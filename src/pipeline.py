"""五级流水线编排：capture → detect → track → temporal → event。

这个文件补的是**最后一环**
--------------------------
在它之前，五个阶段的零件都齐了，但谁也调不动谁：
`infer.Detector` 没有任何调用方，`temporal.judge` 要的序列没人生产，
`verifier` 挂在旁边等待"疑似"信号却永远等不到。所以整条链路实际是断的。

现在按 ARCHITECTURE.md 的契约串起来：

    FramePacket ──detect──▶ Detection[] ──track──▶ Track[] ──judge──▶ Verdict ──▶ Event

三条设计原则
------------
1. **检测器只负责召回，精度交给后面几级**
   检测门限故意压低（`det_conf=0.20`），漏检比误检致命 —— 火情漏一次就是事故。
   误报由 track（有没有持续存在）+ temporal（是不是蒸汽）+ event（确认帧数/冷却）
   三层一起压。所以**不要**为了让告警干净去调高 `det_conf`。

2. **火和烟走两条不同的路，不共用一份逻辑**
   | | 火 fire | 烟 smoke |
   |---|---|---|
   | 靠什么判 | 单帧就够（颜色/形状很特征） | 必须看时序（单帧分不开蒸汽） |
   | 确认条件 | 连续 N 帧命中 | 时序判 smoke **且** 连续 N 帧 |
   | 告警级别 | alarm | 有火 → alarm；无火 → warning |
   火不需要等，等 1.5 秒可能就烧起来了；烟必须等，不等就必然被蒸汽打爆。

3. **云台转动时冻结事件，不清轨迹**
   `moving=True` 时检测框坐标已经不可比，此刻冒出的事件 100% 是假的。
   但轨迹不能清 —— 云台转回来还要接着跟，清了就断成两条。

Event 的够与不够
----------------
这一层只输出"结构化事件 + 截图"，不负责推送。
短信/界面/Webhook 放到调用方（`on_event` 回调）里做，避免把 IO 塞进推理循环。
"""
from __future__ import annotations

import json
import os
import time
from collections import deque
from dataclasses import dataclass, field

import cv2
import numpy as np

from .infer import Detection, Detector, iou
from .sources.base import CameraSource, FramePacket
from .temporal import Verdict, TrackletSample, extract_features, judge
from .tracker import IoUTracker, Track

# 阈值集中在这里。现场要调，改这里不用翻代码。
DEFAULTS = {
    # ── 检测门限：按类别分别设 ──────────────────────────────
    "det_conf": 0.20,          # 送进检测器的总门限（压很低，保召回）
    "fire_conf": 0.30,         # 火：框级门限
    "smoke_conf": 0.45,        # 烟：框级门限（烟的单帧置信更虚，要提高）

    # ── 事件确认 ────────────────────────────────────────────
    "fire_confirm": 3,         # 火：连续命中几帧才报（抗单帧闪烁）
    "smoke_confirm": 2,        # 烟：时序判为 smoke 后再连续几帧坐实

    # ── 去重（按**区域**，不按类别）──────────────────────────
    # 为什么不能按类别全局冷却：画面另一头真的第二处起火会被压掉 30 秒，
    # 安防上这是漏报。所以只压"同一块地方"的重复告警 —— 它防的是轨迹断裂后
    # 重新建轨（同一团火被报两次），不是防"又有新火"。
    "cooldown_sec": 30.0,
    "dedup_iou": 0.10,         # 与已有告警框 IoU 超过这个值 → 视为同一处
    "dedup_dist_px": 60.0,     # 或中心相距不到这么多像素 → 同一处
    # 或者：两个框各外扩 50% 后仍然相交 → 同一处火情。
    # 这一条是为"一片连成带的火被模型切成好几段"准备的：实测宽火焰带会被切成
    # 两个相距 56px 的框（中间并没有断开），不合并就会同一场火烧出两条告警。
    # 用相对尺寸而不是绝对像素，换分辨率不用重调。
    "dedup_grow": 0.5,

    # ── 静态误报抑制 ────────────────────────────────────────
    # 白墙高光、消防服、静态白布：长期不动 → 背景模型会把它吸进背景 → 前景占比趋 0。
    # 注意**不用于火**：烧久了火也会被中值背景吸收，拿它卡火会漏真火情。
    "fg_gate": 0.08,
    "fg_gate_min_dur": 12.0,

    # ── 火源先验 ────────────────────────────────────────────
    "fire_nearby_grow": 0.6,   # 烟框外扩 60% 后若含火框中心，就算「旁边有火」

    # ── 杂项 ────────────────────────────────────────────────
    "min_box_pct": 0.0002,     # 面积小于画面 0.02% 的框当噪点丢掉
    "snapshot_margin": 0.25,   # 截图外扩比例
    "snapshot_max_side": 640,  # 截图长边上限（存盘别存 4K）
}


# ------------------------------------------------------------------ 工具


def expand_box(box, grow: float):
    """按比例向四周外扩。"""
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    return (x1 - grow * w, y1 - grow * h, x2 + grow * w, y2 + grow * h)


def center_in(box, pt) -> bool:
    x1, y1, x2, y2 = box
    return x1 <= pt[0] <= x2 and y1 <= pt[1] <= y2


def crop_box(image: np.ndarray, box, margin: float, max_side: int):
    """按框裁图（带外扩），长边超限就等比缩小。返回新数组，不持有原图引用。"""
    h, w = image.shape[:2]
    x1, y1, x2, y2 = expand_box(box, margin)
    x1 = int(max(0, min(w - 1, x1)))
    y1 = int(max(0, min(h - 1, y1)))
    x2 = int(max(x1 + 1, min(w, x2)))
    y2 = int(max(y1 + 1, min(h, y2)))
    crop = image[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    ch, cw = crop.shape[:2]
    scale = max_side / max(ch, cw)
    if scale < 1.0:
        crop = cv2.resize(crop, (max(1, int(cw * scale)), max(1, int(ch * scale))),
                          interpolation=cv2.INTER_AREA)
    return crop.copy()


# ------------------------------------------------------------------ 契约


@dataclass
class Event:
    """一条告警。界面上要展示的东西全在这里。"""

    kind: str                     # "fire" / "smoke"
    level: str                    # "alarm"（最高级）/ "warning"
    cam_id: str
    track_id: int
    frame_idx: int
    t_first: float                # 该轨迹第一次出现的时间
    t_alarm: float                # 确认告警的时间
    score: float
    box: tuple[float, float, float, float]
    reasons: list[str] = field(default_factory=list)
    snapshot: np.ndarray | None = None
    meta: dict = field(default_factory=dict)

    @property
    def latency(self) -> float:
        """从"目标出现"到"报出来"用了多久。安防验收通常要求 < 10s。"""
        return max(0.0, self.t_alarm - self.t_first)

    def to_dict(self) -> dict:
        """不含图像，可直接落 JSON / 入库。"""
        return {
            "kind": self.kind, "level": self.level, "cam": self.cam_id,
            "track_id": self.track_id, "frame_idx": self.frame_idx,
            "t_first": round(self.t_first, 3), "t_alarm": round(self.t_alarm, 3),
            "latency": round(self.latency, 3), "score": round(self.score, 3),
            "box": [round(float(v), 1) for v in self.box],
            "reasons": list(self.reasons), **self.meta,
        }

    def __str__(self) -> str:
        return (f"[{self.level.upper()}] {self.kind} cam={self.cam_id} "
                f"tid={self.track_id} t={self.t_alarm:.1f}s 延迟={self.latency:.1f}s "
                f"score={self.score:.2f}")


@dataclass
class FrameResult:
    """一帧的处理结果。调试/可视化直接读它。"""

    idx: int
    ts: float
    dets: list[Detection]
    tracks: list[Track]
    verdicts: dict[int, Verdict]
    events: list[Event]
    frozen: bool = False          # 云台转动，本帧判定冻结
    nearby: dict[int, bool] = field(default_factory=dict)   # tid → 烟旁边有没有火


# ------------------------------------------------------------------ 主类


class Pipeline:
    """把五级串起来的编排器。无状态部分（检测器/阈值）构造时定，有状态部分是轨迹。"""

    def __init__(
        self,
        detector: Detector,
        *,
        tracker: IoUTracker | None = None,
        cfg: dict | None = None,
        temporal_thr: dict | None = None,
        verifier=None,
        sliced: bool = True,
        tile_grid: int = 2,
        overlap: float = 0.25,
    ) -> None:
        self.detector = detector
        self.tracker = tracker or IoUTracker()
        self.cfg = {**DEFAULTS, **(cfg or {})}
        self.temporal_thr = temporal_thr or {}
        self.verifier = verifier            # 可选：VLM 复核（默认关，离线可跑）
        self.sliced = sliced
        self.tile_grid = tile_grid
        self.overlap = overlap

        # 检测器门限统一压低，按类别过滤在下面做
        if hasattr(self.detector, "conf"):
            self.detector.conf = self.cfg["det_conf"]

        self._fire_streak: dict[int, int] = {}
        self._smoke_streak: dict[int, int] = {}
        self._fired: set[int] = set()        # 已报过的轨迹（同一条火只报一次）
        self._recent_alarms: list[tuple[float, str, tuple, int]] = []   # (t, kind, box, tid)
        self._recent: deque = deque(maxlen=8)   # 给 VLM 复核用的近几帧

        self.stats = {
            "frames": 0, "dets": 0, "tracks": 0,
            "fire_alarms": 0, "smoke_alarms": 0,
            "steam_suppressed": 0, "static_suppressed": 0,
            "frozen_frames": 0, "cooldown_blocked": 0,
        }

    # ------------------------------------------------------------ 内部
    @staticmethod
    def _kind_of(cls_name: str, cls_id: int) -> str:
        """把类别名映射到 fire / smoke。名字优先，因为不同数据集 id 可能不一样。"""
        n = (cls_name or "").lower()
        if "fire" in n or "flame" in n:
            return "fire"
        if "smoke" in n:
            return "smoke"
        return "fire" if cls_id == 0 else "smoke"

    def _fire_nearby(self, smoke_box, fire_dets: list[Detection]) -> tuple[bool, list[str]]:
        """烟框附近有没有火。这是时序判别里最强的先验，实现要对得起它的权重。"""
        grown = expand_box(smoke_box, self.cfg["fire_nearby_grow"])
        for d in fire_dets:
            if center_in(grown, d.center()) or iou(d.xyxy, grown) > 0:
                return True, [f"同场景检出火（conf {d.conf:.2f}，相距 {self._dist(smoke_box, d.xyxy):.0f}px）"]
        return False, []

    @staticmethod
    def _dist(a, b) -> float:
        acx, acy = (a[0] + a[2]) / 2, (a[1] + a[3]) / 2
        bcx, bcy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
        return float(np.hypot(acx - bcx, acy - bcy))

    def _can_alarm(self, kind: str, tid: int, ts: float, box) -> tuple[bool, str]:
        """按区域去重：同一块地方最近报过就不再报，别处新起火照报。"""
        if tid in self._fired:
            return False, "该轨迹已报过"

        cd = self.cfg["cooldown_sec"]
        self._recent_alarms = [r for r in self._recent_alarms if ts - r[0] < cd]
        g = self.cfg["dedup_grow"]
        for t0, k0, b0, _ in self._recent_alarms:
            if k0 != kind:
                continue
            same_place = (
                iou(b0, box) > self.cfg["dedup_iou"]                       # 明显重叠
                or self._dist(b0, box) < self.cfg["dedup_dist_px"]          # 贴在一起
                or iou(expand_box(b0, g), expand_box(box, g)) > 0           # 同一片火的碎片
            )
            if same_place:
                return False, f"冷却中（{ts - t0:.1f}s 前同一区域已报过）"
        return True, ""

    def _make_event(self, kind, level, tid, t, pkt, score, box, reasons) -> Event:
        self._fired.add(tid)
        self._recent_alarms.append((pkt.ts, kind, tuple(float(v) for v in box), tid))
        snap = crop_box(pkt.image, box, self.cfg["snapshot_margin"],
                        self.cfg["snapshot_max_side"])
        ev = Event(kind=kind, level=level, cam_id=pkt.cam_id, track_id=tid,
                   frame_idx=pkt.idx, t_first=t.first_t, t_alarm=pkt.ts,
                   score=float(score), box=tuple(float(v) for v in box),
                   reasons=list(reasons), snapshot=snap,
                   meta={"hits": t.hits, "n_samples": len(t.samples)})
        self.stats[f"{kind}_alarms"] += 1
        return ev

    def _prune(self) -> None:
        """轨迹结束就把连击计数一起清掉，否则长跑会一直攒内存。"""
        live = set(self.tracker.tracks)
        for d in (self._fire_streak, self._smoke_streak):
            for tid in [k for k in d if k not in live]:
                d.pop(tid, None)
        self._fired &= live

    # ------------------------------------------------------------ 单帧
    def process(self, pkt: FramePacket) -> FrameResult:
        """处理一帧。这是整条链路的唯一入口。"""
        cfg = self.cfg

        # ② detect —— 压低门限拿候选
        dets = self.detector.detect(pkt.image, sliced=self.sliced,
                                    tile_grid=self.tile_grid, overlap=self.overlap)
        h, w = pkt.image.shape[:2]
        min_area = cfg["min_box_pct"] * h * w
        dets = [d for d in dets if d.area >= min_area]

        # ③ track —— 串成轨迹（云台转动时内部自己冻结）
        tracks = self.tracker.update(dets, pkt.image, pkt.ts,
                                     moving=pkt.moving, fg_mask=pkt.fg_mask)

        self.stats["frames"] += 1
        self.stats["dets"] += len(dets)
        self.stats["tracks"] = len(tracks)
        self._recent.append(pkt.image)

        if pkt.moving:
            # 云台在转：坐标不可信，事件一律冻结（轨迹由 tracker 自己保活）
            self.stats["frozen_frames"] += 1
            self._prune()
            return FrameResult(pkt.idx, pkt.ts, dets, tracks, {}, [], frozen=True)

        fire_dets = [d for d in dets
                     if self._kind_of(d.cls_name, d.cls_id) == "fire"
                     and d.conf >= cfg["fire_conf"]]

        events: list[Event] = []
        verdicts: dict[int, Verdict] = {}
        nearby_map: dict[int, bool] = {}

        for t in tracks:
            if t.last_t != pkt.ts or not t.samples:
                continue                      # 本帧没匹配上，不推进判定
            kind = self._kind_of(t.cls_name, t.cls_id)
            conf = t.samples[-1].conf

            # ── 火：快路径，三帧命中就报，不等时序 ──────────────
            if kind == "fire":
                if conf < cfg["fire_conf"]:
                    self._fire_streak[t.tid] = 0
                    continue
                self._fire_streak[t.tid] = self._fire_streak.get(t.tid, 0) + 1
                if self._fire_streak[t.tid] < cfg["fire_confirm"]:
                    continue
                ok, why_not = self._can_alarm("fire", t.tid, pkt.ts, t.last_box)
                if not ok:
                    if "冷却" in why_not:
                        self.stats["cooldown_blocked"] += 1
                    continue
                events.append(self._make_event(
                    "fire", "alarm", t.tid, t, pkt, conf, t.last_box,
                    [f"连续 {self._fire_streak[t.tid]} 帧检出火焰（conf {conf:.2f}）"]))
                continue

            # ── 烟：慢路径，必须先过时序判别 ────────────────────
            if conf < cfg["smoke_conf"]:
                self._smoke_streak[t.tid] = 0
                continue

            feat = extract_features(t.samples)
            nearby, near_why = self._fire_nearby(t.last_box, fire_dets)
            v = judge(feat, thr=self.temporal_thr, has_fire_nearby=nearby)

            # 静态误报抑制：**从出生起**就不在前景里 → 墙面白斑/反光/贴纸。
            # 用 fg_ratio_first 而不是 ema：中值背景会把静止的烟吸进背景，
            # 用 ema 判定会在 20 秒后开始误杀真实的阴燃（实测量 109 帧被压掉）。
            if (v.label == "smoke" and pkt.fg_mask is not None and feat is not None
                    and feat.duration >= cfg["fg_gate_min_dur"]
                    and t.fg_ratio_first < cfg["fg_gate"]):
                self.stats["static_suppressed"] += 1
                v = Verdict("uncertain", v.score * 0.5,
                            v.reasons + [f"出生起就不在前景区（前景 {t.fg_ratio_first:.0%}），"
                                         f"疑似墙面白斑/反光"],
                            feat)

            verdicts[t.tid] = v
            nearby_map[t.tid] = nearby

            if v.label == "steam":
                self.stats["steam_suppressed"] += 1
                self._smoke_streak[t.tid] = 0
                continue
            if v.label != "smoke" or v.score < cfg["smoke_conf"]:
                self._smoke_streak[t.tid] = 0
                continue

            self._smoke_streak[t.tid] = self._smoke_streak.get(t.tid, 0) + 1
            if self._smoke_streak[t.tid] < cfg["smoke_confirm"]:
                continue

            ok, why_not = self._can_alarm("smoke", t.tid, pkt.ts, t.last_box)
            if not ok:
                if "冷却" in why_not:
                    self.stats["cooldown_blocked"] += 1
                continue

            level = "alarm" if nearby else "warning"
            reasons = list(near_why) + list(v.reasons)
            events.append(self._make_event("smoke", level, t.tid, t, pkt,
                                           v.score, t.last_box, reasons))

        self._prune()
        return FrameResult(pkt.idx, pkt.ts, dets, tracks, verdicts, events,
                           frozen=False, nearby=nearby_map)

    # ------------------------------------------------------------ 多帧
    def run(self, source: CameraSource, *, on_event=None, max_frames: int | None = None,
            every: int = 1, quiet: bool = False) -> dict:
        """把一路视频源跑到底。

        every：抽帧间隔（2 = 隔一帧跑一次）。烟雾是慢过程，抽帧几乎不影响效果，
               但吞吐直接翻倍 —— 现场算力不够时先调这个，别去降 imgsz。
        """
        t0 = time.time()
        all_events: list[Event] = []
        n = 0
        empty = 0

        while not source.closed:
            pkt = source.read()
            if pkt is None:
                empty += 1
                if empty > 1000:        # 源活着但一直没帧 → 别死等
                    break
                continue
            empty = 0

            if n % every == 0:
                res = self.process(pkt)
                for ev in res.events:
                    all_events.append(ev)
                    if not quiet:
                        print(ev, flush=True)
                    if on_event is not None:
                        on_event(ev, res)
            n += 1
            if max_frames is not None and n >= max_frames:
                break

        elapsed = max(1e-6, time.time() - t0)
        return {
            "frames": n, "processed": self.stats["frames"],
            "elapsed_s": round(elapsed, 1),
            "fps": round(self.stats["frames"] / elapsed, 2),
            "events": [e.to_dict() for e in all_events],
            "stats": dict(self.stats),
        }

    # ------------------------------------------------------------ 落盘
    @staticmethod
    def save_event(ev: Event, out_dir: str | os.PathLike) -> str | None:
        """把事件写成 jpg + jsonl。返回图片路径（没截图就返回 None）。

        文件名用**存盘时刻**的日历时间，不用 `ev.t_alarm` —— 后者是流内时间戳
        （从开流算的第几秒），拿去格式化会得到 1970 年。
        """
        os.makedirs(out_dir, exist_ok=True)
        stamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
        base = f"{ev.cam_id}_{ev.kind}_{stamp}_t{ev.track_id}_{ev.t_alarm:.0f}s"
        img_path = None
        if ev.snapshot is not None:
            img_path = os.path.join(str(out_dir), base + ".jpg")
            cv2.imwrite(img_path, ev.snapshot)
        with open(os.path.join(str(out_dir), "events.jsonl"), "a", encoding="utf-8") as f:
            rec = ev.to_dict()
            rec["snapshot"] = os.path.basename(img_path) if img_path else None
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        return img_path
