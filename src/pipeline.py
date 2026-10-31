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
from .temporal import (Verdict, TrackletSample, extract_features, judge,
                       judge_fire)
from .tracker import IoUTracker, Track

# 阈值集中在这里。现场要调，改这里不用翻代码。
DEFAULTS = {
    # ── 检测门限：按类别分别设 ──────────────────────────────
    "det_conf": 0.20,          # 送进检测器的总门限（压很低，保召回）
    "fire_conf": 0.30,         # 火：框级门限
    "smoke_conf": 0.45,        # 烟：框级门限（烟的单帧置信更虚，要提高）

    # ── 事件确认 ────────────────────────────────────────────
    # 火：连续命中几帧才报。从 3 提到 5 是为了让闪烁闸有足够样本：
    # 判"抖不抖"至少需要 4 个帧间差分（=5 帧）。3 帧时算不出，闸等于不存在。
    # 代价是 0.12s → 0.20s（25fps），对火灾尺度无关紧要。
    "fire_confirm": 5,
    "smoke_confirm": 2,        # 烟：时序判为 smoke 后再连续几帧坐实

    # ── 极快模式（烟跳过时序判别）──────────────────────────
    # 白烟和蒸汽在**单帧**上同色同形，要分开必须观察一段时间 —— 这是物理事实，
    # 不是代码能绕过的。所以"极快"和"能区分蒸汽"互斥，只能二选一。
    #
    # 适用：现场**没有蒸汽源** —— 机房、配电间、仓库、垃圾房、电池仓、档案室。
    #       这些地方出现上升的白色羽流，几乎只能是烟，没必要等 25 秒。
    # 不适用：厨房、锅炉房、洗浴间、有加湿器/冷却塔的车间 —— 开了就是误报机器。
    # 打开后烟的确认时间从 25s 降到 3 帧（25fps 下约 0.12s）。
    #
    # 开了之后是**分级**的，不是一上来就喊狼来了：
    #   notice  0.2s   极快模式命中 → 立刻推给值班员看画面（需人工确认）
    #   alarm   1.8s+  时序层也确认是烟 → 升级为正式告警
    #   cleared 5s+    时序层判成蒸汽 → 自动撤销刚才那条 notice
    # 这样"反应快"和"不误报"各自拿到各自该有的级别，不必二选一。
    "fast_smoke": False,
    "fast_smoke_confirm": 3,   # 直通模式下烟连续命中几帧就报

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

    # ── LLM 裁判 ────────────────────────────────────────────
    # 裁判判过的目标，**规则层不许再定级**。不设这条会出大问题：
    # 实测 LLM 判"蒸汽"压掉之后，下一帧裁判去重生效（不重复送审），
    # 规则路径就接手把这个目标重新报成 ALARM —— 等于裁判白判，还多花了一次钱。
    # 静默时长和裁判自己的去重窗口对齐，过了这个时间才允许重新送审。
    "llm_mute_sec": 180.0,
    # 被判为非风险的目标，判完之后又活了这么久 → 把人叫回来复核。
    # 真蒸汽几秒最多十几秒就散，撑过 30 秒本身就说明裁判可能看错了。
    # 这也是漏报的兜底：不给这条，一次误判就是一个完全静默的窗口。
    "dismiss_recheck_sec": 30.0,

    # 闪烁闸总开关。默认开。合成测试场景里要关掉 —— 贴图火是静止的，
    # 判它 static 是物理正确的，开着只会把"合成数据不真实"变成"测试失败"。
    "flicker_gate": True,

    # 检测策略。auto = 先粗筛、有东西才切片（空白帧 4.8 倍提速）；
    # slice = 每帧固定切片（最稳）；full = 不切片（小目标召回掉 13 点，仅排障用）。
    "detect_mode": "auto",

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


@dataclass
class _Pending:
    """已送审、等 LLM 裁决的目标。

    不能直接存 Track 对象：裁决回来时轨迹可能已经老化消失了，但它照样要出事件
    （火真的烧过，只是烧完了）。所以这里存的是**事件需要的最小快照**。
    """

    kind: str
    tid: int
    cam_id: str
    frame_idx: int
    ts: float
    t_first: float
    box: tuple[float, float, float, float]
    snapshot: np.ndarray | None
    score: float
    reasons: list[str]
    hits: int
    n_samples: int


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
        arbiter=None,
        sliced: bool | None = None,
        tile_grid: int = 2,
        overlap: float = 0.25,
    ) -> None:
        self.detector = detector
        self.tracker = tracker or IoUTracker()
        self.cfg = {**DEFAULTS, **(cfg or {})}
        # 检测策略：显式给了 sliced 就按它走（旧行为，测试脚本靠这个做 A/B）；
        # 什么都没给就走 auto —— 粗筛没东西就不切片，空白帧 4.8 倍提速。
        if "detect_mode" not in (cfg or {}):
            self.cfg["detect_mode"] = ("slice" if sliced else "full") \
                if sliced is not None else "auto"
        self.sliced = self.cfg["detect_mode"] == "slice"
        self.temporal_thr = temporal_thr or {}
        self.verifier = verifier            # 可选：同步 VLM 复核（旧路径）
        self.arbiter = arbiter              # 可选：异步 LLM 裁判（新路径，推荐）
        self.tile_grid = tile_grid
        self.overlap = overlap

        # 检测器门限统一压低，按类别过滤在下面做
        if hasattr(self.detector, "conf"):
            self.detector.conf = self.cfg["det_conf"]

        self._fire_streak: dict[int, int] = {}
        self._smoke_streak: dict[int, int] = {}
        self._notice_streak: dict[int, int] = {}
        self._noticed: set[int] = set()      # 已发 notice 的轨迹（一条轨迹只发一次）
        self._cleared: set[int] = set()      # notice 已被撤销的轨迹（防止反复发/撤）
        self._steam_noticed: set[int] = set()  # 已推过"降级为蒸汽"提示的轨迹
        self._static_noticed: set[int] = set() # 已推过"疑似灯光"提示的轨迹
        self._recheck_done: set[int] = set()   # 已对被判降级却久不消散的目标叫过人
        self._fired: set[int] = set()        # 已报过的轨迹（同一条火只报一次）
        self._recent_alarms: list[tuple[float, str, tuple, int]] = []   # (t, kind, box, tid)
        self._recent_notices: list[tuple[float, tuple]] = []             # (t, box)
        self._pending: dict[str, _Pending] = {}    # key=cam:tid → 等裁决的目标
        self._arbiter_done: dict[str, float] = {}  # key → 裁判给出结论的流内时刻
        self._dismissed: dict[str, float] = {}     # key → 被判"非风险"的时刻（只这些才复核）
        self._recent: deque = deque(maxlen=8)   # 给 VLM 复核用的近几帧

        self.stats = {
            "frames": 0, "dets": 0, "tracks": 0,
            "fire_alarms": 0, "smoke_alarms": 0,
            "steam_suppressed": 0, "static_suppressed": 0,
            "steam_ignored": 0,          # 极快模式下被忽略的蒸汽判定（用于事后校准）
            "notices": 0, "notice_cleared": 0, "steam_notices": 0,  # 提示 / 撤销 / 降级告知
            "recheck_notices": 0,        # 降级后久不消散、把人叫回来的次数
            "static_light_blocked": 0,   # 被闪烁闸拦下的疑似灯光（真火保留 = 真阳性）
            "llm_reviews": 0, "llm_alarms": 0, "llm_dismissed": 0,   # LLM 裁判
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
        """按区域去重：同一块地方报过就不再报，别处新起火照报。

        关键细节是**续期**：命中同区域记录时，把那条记录的时间戳刷成当前时刻。
        不续期的话冷却一到（30s）就重新放开，同一场火会被反复报 —— 因为模型把
        一团火切成好几条轨迹，每条轨迹都独立走完流程。实测过：一条火在 t=1.4s
        报过，t=31.4s 和 t=61.5s 又各报一次，`cooldown_blocked` 累计 1577 次。
        续期之后就变成：**只要这个位置还在烧，就只报一次；烧完了 30 秒后才允许重报。**
        """
        if tid in self._fired:
            return False, "该轨迹已报过"

        cd = self.cfg["cooldown_sec"]
        g = self.cfg["dedup_grow"]
        kept: list[tuple[float, str, tuple, int]] = []
        hit = None
        for rec in self._recent_alarms:
            t0, k0, b0, tid0 = rec
            if ts - t0 >= cd:
                continue                                   # 过期，丢掉
            if hit is None and k0 == kind and (
                    iou(b0, box) > self.cfg["dedup_iou"]        # 明显重叠
                    or self._dist(b0, box) < self.cfg["dedup_dist_px"]   # 贴在一起
                    or iou(expand_box(b0, g), expand_box(box, g)) > 0):  # 同一片火的碎片
                hit = len(kept)
            kept.append(rec)
        if hit is not None:
            t0, k0, b0, tid0 = kept[hit]
            kept[hit] = (ts, k0, b0, tid0)                 # 续期：目标还在烧
            self._recent_alarms = kept
            return False, f"同一区域仍在持续告警（{(ts - t0):.0f}s 前报过，已续期）"
        self._recent_alarms = kept
        return True, ""

    def _can_notice(self, ts: float, box) -> bool:
        """notice 也要按区域去重。

        notice 不进 `_fired`（否则正式告警永远升级不上来），但**不**去重的话，
        一团被模型切成 3 段的烟会吐出 3 条 notice —— 实测就是 4 条刷屏。
        所以它用自己的一套记录，和正式告警的冷却互不干扰。
        """
        cd = self.cfg["cooldown_sec"]
        g = self.cfg["dedup_grow"]
        kept: list[tuple[float, tuple]] = []
        hit = None
        for rec in self._recent_notices:
            t0, b0 = rec
            if ts - t0 >= cd:
                continue
            if hit is None and (iou(b0, box) > self.cfg["dedup_iou"]
                                or self._dist(b0, box) < self.cfg["dedup_dist_px"]
                                or iou(expand_box(b0, g), expand_box(box, g)) > 0):
                hit = len(kept)
            kept.append(rec)
        if hit is not None:
            kept[hit] = (ts, kept[hit][1])      # 同样续期，避免目标还在却反复提示
            self._recent_notices = kept
            return False
        self._recent_notices = kept
        return True

    def _make_event(self, kind, level, tid, t, pkt, score, box, reasons) -> Event:
        # notice / cleared 是临时提示，**不占用正式告警的去重额度** ——
        # 否则 notice 会把后面真正的 alarm 挡在冷却期外面，等于永远升级不了。
        if level in ("warning", "alarm"):
            self._fired.add(tid)
            self._recent_alarms.append((pkt.ts, kind, tuple(float(v) for v in box), tid))
        snap = crop_box(pkt.image, box, self.cfg["snapshot_margin"],
                        self.cfg["snapshot_max_side"])
        ev = Event(kind=kind, level=level, cam_id=pkt.cam_id, track_id=tid,
                   frame_idx=pkt.idx, t_first=t.first_t, t_alarm=pkt.ts,
                   score=float(score), box=tuple(float(v) for v in box),
                   reasons=list(reasons), snapshot=snap,
                   meta={"hits": t.hits, "n_samples": len(t.samples)})
        if level in ("warning", "alarm"):        # notice / cleared 不算告警，别污染统计
            self.stats[f"{kind}_alarms"] += 1
        return ev

    def _prune(self) -> None:
        """轨迹结束就把连击计数一起清掉，否则长跑会一直攒内存。"""
        live = set(self.tracker.tracks)
        for d in (self._fire_streak, self._smoke_streak, self._notice_streak):
            for tid in [k for k in d if k not in live]:
                d.pop(tid, None)
        self._fired &= live
        self._noticed &= live
        self._cleared &= live
        self._steam_noticed &= live
        self._static_noticed &= live
        self._recheck_done &= live
        for k in [k for k in self._arbiter_done if k.rsplit(":", 1)[-1].isdigit()
                  and int(k.rsplit(":", 1)[-1]) not in live]:
            self._arbiter_done.pop(k, None)
            self._dismissed.pop(k, None)

    # ------------------------------------------------------------ LLM 裁判
    def _defer(self, kind, tid, t, pkt, score, box, reasons) -> bool:
        """把定级交给 LLM 裁判。返回 True = 已送审/正在等，本帧不要发事件。

        返回 False 的两种情形都**必须**让调用方继续走规则路径：
          · 没挂裁判
          · 裁判拒收（没 Key / 超额 / 去重 / 排队满）
        否则 LLM 一挂告警就全停了 —— 这是最不能接受的事故。
        """
        if self.arbiter is None:
            return False
        key = f"{pkt.cam_id}:{tid}"
        if key in self._pending:              # 已在等裁决，别重复提交
            return True
        done = self._arbiter_done.get(key)
        if done is not None and (pkt.ts - done) < self.cfg["llm_mute_sec"]:
            return True                       # 裁判已判过，规则层闭嘴
        ok, _why = self.arbiter.submit(
            key, pkt.image, box, rule_label=kind, rule_score=score,
            context=f"{kind}，检测 conf {t.samples[-1].conf:.2f}，轨迹已命中 {t.hits} 帧",
            feat=extract_features(t.samples))
        if not ok:
            return False
        self._pending[key] = _Pending(
            kind=kind, tid=tid, cam_id=pkt.cam_id, frame_idx=pkt.idx, ts=pkt.ts,
            t_first=t.first_t, box=tuple(float(v) for v in box),
            snapshot=crop_box(pkt.image, box, self.cfg["snapshot_margin"],
                              self.cfg["snapshot_max_side"]),
            score=float(score), reasons=list(reasons),
            hits=t.hits, n_samples=len(t.samples))
        return True

    def _stale_recheck(self, t, pkt) -> bool:
        """被判为非风险的目标还在，就说明裁判可能错了 —— 把人叫回来。

        裁判说"这是蒸汽"只是**一次判决**，不是永久豁免。这一点必须写进代码里，
        否则一次误判就是一段盲区：`llm_mute_sec` 期间规则层完全闭嘴，真烟在烧
        也没人知道。

        判据取"**判降级之后又撑过了多久**"，不是目标总寿命：总寿命会把
        "刚判完立刻复核"也算成命中，两张卡片同一秒落地，等于没复核。
        真蒸汽几秒最多十几秒就散，撑过 30 秒本身就是证据。

        只对**判降级**的目标生效（`_dismissed`），不是所有裁判结论：
        "拿不准"推的是 notice，本来就等人看，不需要再复核一遍。

        这里**不重新调 LLM**：同一个问题问第二遍，答案大概率还是同一个，
        白花配额还多等 1.5 秒。该做的是把证据直接摆到人面前。
        """
        done = self._dismissed.get(f"{pkt.cam_id}:{t.tid}")
        if done is None:                       # 裁判没判降级，走正常路径
            return False
        if t.tid in self._recheck_done:        # 已叫过一次，别刷屏
            return False
        if pkt.ts - done < self.cfg["dismiss_recheck_sec"]:
            return False
        self._recheck_done.add(t.tid)
        return True

    def _resolve(self, key: str, res, pkt: FramePacket | None) -> Event | None:
        """把裁决结果变成事件（在下一帧开头收割，不在送审当帧）。"""
        info = self._pending.pop(key, None)
        if info is None:
            return None
        self.stats["llm_reviews"] += 1
        self._arbiter_done[key] = info.ts     # 这个目标从此由裁判说了算
        reasons = ([f"LLM 判为 {res.label}（risk={res.risk}，conf={res.confidence:.2f}）："
                    f"{res.reason}"] + info.reasons[:2])

        if res.is_dismiss:
            # ⚠️ 这里**不能** return None。判成蒸汽就一声不响，等于给真烟开了后门：
            # 规则层或 LLM 只要错一次，管理员屏幕上什么都没有，事故就发生在静默里。
            # 所以降级也要推一条，让前端留下痕迹、让人可以自己复核。
            self.stats["llm_dismissed"] += 1
            self._dismissed[key] = info.ts         # 记下降级时刻，供兜底复核计时
            return Event(kind=info.kind, level="cleared", cam_id=info.cam_id,
                         track_id=info.tid, frame_idx=info.frame_idx,
                         t_first=info.t_first,
                         t_alarm=pkt.ts if pkt is not None else info.ts,
                         score=float(res.confidence or info.score), box=info.box,
                         reasons=[f"LLM 判为{res.label}，已降级（非火灾风险）"] + info.reasons[:2],
                         snapshot=info.snapshot,
                         meta={"hits": info.hits, "n_samples": info.n_samples,
                               "llm_label": res.label, "llm_risk": res.risk,
                               "llm_source": res.source,
                               "llm_latency": round(res.latency_s, 2),
                               "dismissed": True})
        if res.label in ("fire", "smoke"):
            level = "alarm" if res.risk == "high" else "warning"
        else:                                  # uncertain → 不硬判，推给人看
            level = "notice"

        now_ts = pkt.ts if pkt is not None else info.ts
        if level != "notice" and not self._can_alarm(info.kind, info.tid, now_ts, info.box)[0]:
            return None

        ev = Event(kind=info.kind, level=level, cam_id=info.cam_id, track_id=info.tid,
                   frame_idx=info.frame_idx, t_first=info.t_first,
                   # 用**裁决生效时刻**而不是送审时刻：否则"确认延迟"会把等 LLM 的
                   # 那一两秒吞掉，数字看着漂亮但不符合实际
                   t_alarm=now_ts,
                   score=float(res.confidence or info.score), box=info.box,
                   reasons=reasons, snapshot=info.snapshot,
                   meta={"hits": info.hits, "n_samples": info.n_samples,
                         "llm_label": res.label, "llm_risk": res.risk,
                         "llm_source": res.source,
                         "llm_latency": round(res.latency_s, 2)})
        if level in ("warning", "alarm"):
            self._fired.add(info.tid)
            self._recent_alarms.append((info.ts, info.kind, info.box, info.tid))
            self.stats["llm_alarms"] += 1
            self.stats[f"{info.kind}_alarms"] += 1
        else:
            self._recent_notices.append((info.ts, info.box))
        return ev

    # ------------------------------------------------------------ 单帧
    def _detect(self, image):
        """按 detect_mode 选检测策略。

        · "auto"（默认）：先粗筛一帧，粗筛没东西就不切片。空白帧只花 10.7ms，
          可疑帧才付切片的钱 —— 真实枪机 99% 的帧是空的，省的就是这部分。
        · "slice"：每帧固定 2x2 切片，最稳但最贵（52ms/帧）。
        · "full"：不切片，最快（10.7ms）但小目标召回掉 13 个点，只用于排障对比。
        """
        mode = self.cfg.get("detect_mode", "slice")
        if mode == "auto" and hasattr(self.detector, "detect_auto"):
            return self.detector.detect_auto(
                image, tile_grid=self.tile_grid, overlap=self.overlap)
        return self.detector.detect(image, sliced=(mode == "slice"),
                                    tile_grid=self.tile_grid, overlap=self.overlap)

    def process(self, pkt: FramePacket) -> FrameResult:
        """处理一帧。这是整条链路的唯一入口。"""
        cfg = self.cfg

        # ② detect —— 压低门限拿候选
        dets = self._detect(pkt.image)
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

        # 先收割 LLM 裁决（上几帧送审的）。放在最前面：早一帧出结果就早一帧告警。
        if self.arbiter is not None:
            for key, res in self.arbiter.poll():
                ev = self._resolve(key, res, pkt)
                if ev is not None:
                    events.append(ev)

        for t in tracks:
            if t.last_t != pkt.ts or not t.samples:
                continue                      # 本帧没匹配上，不推进判定
            kind = self._kind_of(t.cls_name, t.cls_id)
            conf = t.samples[-1].conf

            # ── 火：快路径，但必过一道「闪烁闸」────────────────
            # 为什么火也要过闸：车位灯、路灯、招牌、玻璃反光在单帧上和火一模一样
            # （高亮、暖色、局部过曝），YOLO 必给高 conf。原来只看"连续 3 帧命中"，
            # 而灯是恒亮的 —— 3 帧稳过，等于没有任何拦截。实测过：合成的一块橙色
            # 亮斑能拿到 conf=0.87，比真火还高。
            # 火和灯唯一稳定的差别是**抖不抖**，所以这道闸只看帧间亮度抖动。
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
                fire_reasons = [f"连续 {self._fire_streak[t.tid]} 帧检出火焰（conf {conf:.2f}）"]

                fv = (judge_fire(extract_features(t.samples),
                                 area_px=t.samples[-1].area, thr=self.temporal_thr)
                      if cfg["flicker_gate"] else None)
                if fv is not None and fv.label == "static":
                    # ⚠️ 这里不能 continue 了事。灯也会烧到短路、火也可能烧得很稳，
                    # 判错一次就是漏报。所以降级留痕 + 送 LLM 复核，双保险。
                    self.stats["static_light_blocked"] += 1
                    if t.tid not in self._static_noticed and t.tid not in self._fired:
                        self._static_noticed.add(t.tid)
                        events.append(self._make_event(
                            "fire", "notice", t.tid, t, pkt, conf, t.last_box,
                            ["检出火焰但亮度恒定，疑似灯光/反光，已降级（**不是排除**），"
                             "已送人工复核"] + list(fv.reasons[:1])))
                        self._defer("fire", t.tid, t, pkt, conf, t.last_box,
                                    ["规则层疑为静态光源，请判断是火还是灯"])
                    continue

                if self._defer("fire", t.tid, t, pkt, conf, t.last_box, fire_reasons):
                    continue          # 已送审，等 LLM 裁决再定级
                events.append(self._make_event(
                    "fire", "alarm", t.tid, t, pkt, conf, t.last_box,
                    fire_reasons + ([] if (fv is None or fv.label != "fire")
                                    else list(fv.reasons[:1]))))
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

            # ── 极快模式的第一级：先发 notice，不等时序 ──────────
            # 之后**不 continue**，继续往下走正规判别：
            # 时序确认了 → 升级成 alarm；判成蒸汽 → 发 cleared 撤销这条 notice。
            if cfg["fast_smoke"]:
                self._notice_streak[t.tid] = self._notice_streak.get(t.tid, 0) + 1
                if (self._notice_streak[t.tid] >= cfg["fast_smoke_confirm"]
                        and t.tid not in self._noticed and self._can_notice(pkt.ts, t.last_box)):
                    self._noticed.add(t.tid)
                    self._recent_notices.append((pkt.ts, tuple(float(v) for v in t.last_box)))
                    self.stats["notices"] += 1
                    events.append(self._make_event(
                        "smoke", "notice", t.tid, t, pkt, conf, t.last_box,
                        [f"极快模式：连续 {self._notice_streak[t.tid]} 帧检出烟"
                         f"（conf {conf:.2f}），尚未做蒸汽判别，需人工确认"]))
                    # notice 之后立刻送审：这才是"YOLO 第一轮可疑就交给 LLM"的地方。
                    # notice 已经让值班员看到画面了，LLM 的结果用来升级或压掉。
                    self._defer("smoke", t.tid, t, pkt, conf, t.last_box,
                                [f"极快模式第一轮可疑（conf {conf:.2f}）"])

            if v.label == "steam":
                self.stats["steam_suppressed"] += 1
                self._smoke_streak[t.tid] = 0
                # ⚠️ 同样不能静默。规则层判蒸汽只是"倾向"，不是"结论"——
                # 面积/亮度这几个手工特征分不开真烟和蒸汽，判错一次就是漏报。
                # 所以每条轨迹降级时推一条 notice 到前端，标注判据，让人自己看。
                if t.tid not in self._steam_noticed and t.tid not in self._fired:
                    self._steam_noticed.add(t.tid)
                    self.stats["steam_notices"] += 1
                    events.append(self._make_event(
                        "smoke", "notice", t.tid, t, pkt, v.score, t.last_box,
                        ["规则层倾向判为蒸汽/水雾，已降级（不是排除）"] + list(v.reasons[:2])))
                    self._defer("smoke", t.tid, t, pkt, v.score, t.last_box,
                                ["规则层倾向蒸汽，请复核"])
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
            # 兜底复核：裁判判过"蒸汽"的东西还赖着不走 —— 那就不是蒸汽。
            # 不再问 LLM，直接推 notice 让值班员看一眼。
            if self._stale_recheck(t, pkt):
                self.stats["recheck_notices"] += 1
                held = pkt.ts - self._dismissed[f"{pkt.cam_id}:{t.tid}"]
                events.append(self._make_event(
                    "smoke", "notice", t.tid, t, pkt, v.score, t.last_box,
                    [f"判为非火灾风险后仍持续 {held:.0f}s 未消散（共 {pkt.ts - t.first_t:.0f}s），"
                     f"请人工确认是否误判"] + list(v.reasons[:2])))
                continue
            if self._defer("smoke", t.tid, t, pkt, v.score, t.last_box, reasons):
                continue                  # 已送审，等 LLM 裁决再定级
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
