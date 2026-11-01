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
from .light_memory import LightMemory
from .sources.base import CameraSource, FramePacket
from .temporal import (Verdict, TrackletSample, extract_features, judge,
                       judge_fire)
from .tracker import IoUTracker, Track

# 阈值集中在这里。现场要调，改这里不用翻代码。
DEFAULTS = {
    # ── 检测门限：按类别分别设 ──────────────────────────────
    "det_conf": 0.20,          # 送进检测器的总门限（压很低，保召回）
    # 火告警阈值。0.45 偏高：真火在视频里（距离远、被遮挡、单帧模糊）
    # 常只拿到 0.3~0.4，会掉进灰色候选档，看起来像"没发现"。
    # 0.35 是折中：既让真火进黄色档，又不至于把所有反光都拉进来。
    "fire_conf": 0.35,
    "smoke_conf": 0.30,        # 烟：告警阈值

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
    # 置信度闸门：LLM 只在"规则拿不准"的区间里用。
    #   高于 llm_sure_conf → 规则层自己就能定，花钱问模型纯浪费
    #   低于 llm_min_conf  → 模型看了多半也是噪声，救了也是误报
    # 两头都省，只留中间那段真正需要判断的。
    # 送审下限必须**低于**报警阈值，否则会出现"不够格送审、却够格报警"的漏洞：
    # 实测 conf 0.47 撞上 fire_conf 0.45，跳过裁判直接报了火情。
    "llm_min_conf": 0.18,
    # 免裁线：达到这个置信度才允许不送审直接定性。
    # 设太高浪费钱，设太低会让检测器单方面定性 —— 实测 0.85 时一张人脸被
    # 直接推成「火情」，裁判根本没参与。0.92 是"基本只可能是真火"的量级。
    "llm_sure_conf": 0.92,
    # 规则层置信度高于此值时，LLM 判"非风险"也**只能降级为待确认，不能直接排除**。
    # 依据：实测出现过 YOLO 给 0.95、LLM 一句 "light" 就把它抹掉的情况，
    # 而画面里确实有火 —— 两个信号冲突时应该交给人看，不该由模型单方面裁决。
    # 规则层置信达到这个值、而 LLM 判非火时，才升级成"待确认"交人。
    # 0.85 太低：实测开思考后 LLM 的理由相当扎实（"头发反光""无火苗轮廓"），
    # 这时还把结论推回给人重判一遍，等于裁判白请了。
    "llm_override_conf": 0.95,
    # 置信度高于此值时，即使该区域刚被判过"非火"也重新送审（防区域静音误伤真火）
    "llm_mute_bypass_conf": 0.80,
    # 固定光源预筛（纯本地、免费）。设为 False 则所有目标都送审，更准但更贵。
    "static_prefilter": True,
    # 光敏引擎：找出画面里所有亮斑，用灰框标出来（不参与追踪和告警）。
    # YOLO 只认识训练过的样子，远处/被遮挡/没见过的发光物它可能完全不出框；
    # 而"那里有光"用亮度阈值就能百分之百抓到。两者互补。
    "lightspot": True,
    "lightspot_v_min": 205,        # 亮度阈值：低于它的不算"发光"
    "lightspot_min_area": 12,      # 最小光斑面积，再小基本是噪点
    "lightspot_max_area_pct": 0.25,  # 超过画面 1/4 的是整片过曝，框它没意义
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

    # ── 已知非火灾区域的记忆（省钱层）────────────────────────────
    # 一盏路灯整夜亮着，YOLO 会把它切成一条条新轨迹，而送审键原来是
    # `机位:轨迹ID` —— 每换一条就重新问一次 LLM。实测这套记忆能把
    # 同一盏灯的 LLM 调用从"每次重连都问"压到"确认一次就够"。
    "light_memory": True,
    "light_memory_path": "data/light_memory.json",
    "light_memory_iou": 0.45,        # 归一化 IoU 超过这个就当同一处
    "light_memory_area_tol": 0.6,    # 面积变化 >60% 认为"已不是原来那东西"，记忆作废
    "light_memory_ttl": 604800.0,    # 7 天不出现就忘掉（场景可能已经改了）
    "light_memory_notice_sec": 3600.0,
    # 命中区域记忆时是否推"待确认"卡片。默认关：记忆仍在生效（省调用），
    # 只是不占事件流。要排查记忆是否正常工作时再打开。
    "light_memory_notice": False,   # 命中记忆时给前端留痕的最小间隔（默认一小时一条）

    # 同区域送审静音：键是**区域**不是轨迹 ID。哪怕记忆还没建立
    # （第一次见到这盏灯），也只在同一块地方问一次。
    "llm_region_mute_sec": 180.0,

    # ── 杂项 ────────────────────────────────────────────────
    # 最小框面积。0.0002 在 960x540 上是 104 像素，约 10x10 —— 远处的火
    # 刚好卡在这条线附近被当噪点丢掉。降到 0.00008（约 6x8）给小火留余地，
    # 真正的像素噪声由后面的 LLM 和连续帧确认来兜。
    "min_box_pct": 0.00003,
    "snapshot_margin": 0.85,   # 截图外扩比例（0.45 -> 0.85，烟雾/小火苗必须带周边才能判）
    # ROI 最小边长：框太小（比如 30x30 的火苗）时按比例扩到这个尺寸再裁，
    # 否则模型只能看到一个色块，判断不了它在人脸/墙面/半空中
    "snapshot_min_side": 220,
    "snapshot_max_side": 640,  # ROI 截图长边上限（存盘别存 4K）
    "snapshot_full_max_side": 900,  # 全景图长边上限，比 ROI 大一点才看得清位置
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


def full_view(image: np.ndarray, box, max_side: int = 800):
    """全图缩略 + 把检测框标出来。

    为什么要跟 ROI 一起存
    --------------------
    ROI 是放大看细节的（判断"这团东西是不是烟"），但它丢掉了空间上下文 ——
    事后审查时只看到一团模糊的白色，根本不知道它在房间哪个角落、旁边有没有
    其他东西。带上这张标了框的全景图，一眼就能定位，也方便判断"是不是窗外
    飘进来的雾"这类环境因素。
    """
    img = image.copy()
    h, w = img.shape[:2]
    x1, y1, x2, y2 = (int(round(v)) for v in expand_box(box, 0.0))
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w - 1, x2), min(h - 1, y2)
    if x2 > x1 and y2 > y1:
        thick = max(2, int(max(h, w) / 320))
        cv2.rectangle(img, (x1, y1), (x2, y2), (48, 48, 235), thick)
    scale = max_side / max(h, w)
    if scale < 1.0:
        img = cv2.resize(img, (max(1, int(w * scale)), max(1, int(h * scale))),
                         interpolation=cv2.INTER_AREA)
    return img


def crop_box(image: np.ndarray, box, margin: float, max_side: int,
             min_side: int = 220):
    """按框裁图（带外扩），长边超限就等比缩小。返回新数组，不持有原图引用。

    为什么要 min_side
    -----------------
    实测小火苗的检测框可能只有 30x30，按框裁出来就是一张 33x34 的小图：
    模型看到的只是一个白色小方块，判断不了它是火苗、反光还是手电筒。
    用户的原话是"可以根据周围环境进行判断"，那就得让裁剪图带足周边 ——
    所以框太小时按比例向四周扩到至少 min_side，让它能看到
    "这个亮点是在人脸上、在墙上、还是在半空中"。
    """
    h, w = image.shape[:2]
    x1, y1, x2, y2 = expand_box(box, margin)
    # 太小就围绕中心继续外扩，直到达到最小边长（受画面边界限制）
    cw, ch = max(1.0, x2 - x1), max(1.0, y2 - y1)
    if min(cw, ch) < min_side:
        k = max(min_side / cw, min_side / ch)
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        x1, x2 = cx - cw * k / 2.0, cx + cw * k / 2.0
        y1, y2 = cy - ch * k / 2.0, cy + ch * k / 2.0
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
    snapshot: np.ndarray | None = None          # ROI 裁剪（放大看细节）
    snapshot_full: np.ndarray | None = None     # 全图缩略（看空间位置）
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
    # 归一化坐标（0~1）+ 归一化面积。裁判判"不是火"时要靠它们记住这块地方，
    # 存像素坐标的话换分辨率整份记忆就废了。
    box_n: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    area_n: float = 0.0


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

        # 一致性校验：送审下限必须**低于**每一类的报警阈值。
        # 否则会存在这样的目标 —— 够格报警(fire_conf)却不够格送审(llm_min_conf)，
        # 于是规则层绕过裁判直接定性。实测就踩过：conf 0.47 撞上 0.45 报了火情，
        # 而 0.47 < 0.50 的送审下限，裁判根本没看到它。
        low = float(self.cfg.get("llm_min_conf", 0.35))
        for key in ("fire_conf", "smoke_conf"):
            if key in self.cfg and low > float(self.cfg[key]):
                fixed = max(0.05, float(self.cfg[key]) - 0.05)
                print(f"[warn] llm_min_conf({low}) 高于 {key}({self.cfg[key]})，"
                      f"会漏送一部分告警目标；已自动下调到 {fixed:.2f}", flush=True)
                self.cfg["llm_min_conf"] = fixed
                low = fixed

        self._fire_streak: dict[int, int] = {}
        self._smoke_streak: dict[int, int] = {}
        self._notice_streak: dict[int, int] = {}
        self._noticed: set[int] = set()      # 已发 notice 的轨迹（一条轨迹只发一次）
        self._cleared: set[int] = set()      # notice 已被撤销的轨迹（防止反复发/撤）
        self._steam_noticed: set[int] = set()  # 已推过"降级为蒸汽"提示的轨迹
        self._static_noticed: set[int] = set() # 已推过"疑似灯光"提示的轨迹
        self._recheck_done: set[int] = set()   # 已对被判降级却久不消散的目标叫过人
        # 被"固定光源预筛"拦下的轨迹：不送审、也不推事件。
        # 这类目标已满足"稳定 >8s + 面积不涨 + 亮度不抖 + 边缘稳定"，
        # 是典型的灯具/反光/手电筒常亮，再弹一条"待确认"纯属噪音。
        self._prefiltered: set[int] = set()
        self._fired: set[int] = set()        # 已报过的轨迹（同一条火只报一次）
        self._recent_alarms: list[tuple[float, str, tuple, int]] = []   # (t, kind, box, tid)
        self._recent_notices: list[tuple[float, tuple]] = []             # (t, box)
        self._pending: dict[str, _Pending] = {}    # key=cam:tid → 等裁决的目标
        # 被"合并送审"的轨迹：tid -> 时间戳。
        # 同一位置出现多个类别的框时，只把其中一条送给 LLM，其余记在这里
        # 表示"已经一起看过了"，避免同一块地方被送两次、判出两个矛盾结论。
        self._peer_suppressed: dict[int, float] = {}
        self._arbiter_done: dict[str, float] = {}  # key → 裁判给出结论的流内时刻
        self._dismissed: dict[str, float] = {}     # key → 被判"非风险"的时刻（只这些才复核）
        self._recent: deque = deque(maxlen=8)   # 给 VLM 复核用的近几帧

        # 实时检测帧率（滑动窗口）。和画面帧率是两个完全不同的量：
        # 画面跟摄像头走，检测跟算力走，之前混用同一个字段导致看不出瓶颈。
        self.det_fps = 0.0
        self._det_ts: deque = deque(maxlen=24)

        self.stats = {
            "frames": 0, "dets": 0, "tracks": 0,
            "fire_alarms": 0, "smoke_alarms": 0,
            "steam_suppressed": 0, "static_suppressed": 0,
            "steam_ignored": 0,          # 极快模式下被忽略的蒸汽判定（用于事后校准）
            "notices": 0, "notice_cleared": 0, "steam_notices": 0,  # 提示 / 撤销 / 降级告知
            "recheck_notices": 0,        # 降级后久不消散、把人叫回来的次数
            "static_light_blocked": 0,   # 被闪烁闸拦下的疑似灯光（真火保留 = 真阳性）
            "llm_reviews": 0, "llm_alarms": 0, "llm_dismissed": 0,   # LLM 裁判
            "llm_saved": 0,              # 因为"这块地方判过"而省掉的 LLM 调用（按轨迹计一次）
            "memory_hits": 0,            # 命中区域记忆（已知非火）的帧数
            "memory_reopened": 0,        # 记忆因外形突变而作废的次数
            "frozen_frames": 0, "cooldown_blocked": 0,
        }

        # 已确认"不是火"的区域记忆（落盘，跨重启有效）。这是省钱层：
        # 判定走它就不再调 LLM，一盏灯一晚一次就够。
        self.lights: LightMemory | None = None
        if self.cfg["light_memory"]:
            self.lights = LightMemory(
                self.cfg["light_memory_path"],
                iou_thr=float(self.cfg["light_memory_iou"]),
                area_tol=float(self.cfg["light_memory_area_tol"]),
                ttl=float(self.cfg["light_memory_ttl"]))
        self._region_judged: list[tuple[str, tuple, float]] = []   # (cam, box_n, ts)
        self._saved_tids: set[int] = set()   # 已计过"省一次调用"的轨迹，别按帧重复计
        self._mem_saved_ts = 0.0

    # ------------------------------------------------------------ 内部
    @staticmethod
    def _kind_of(cls_name: str, cls_id: int) -> str:
        """把类别名映射到 fire / smoke。名字优先，因为不同数据集 id 可能不一样。"""
        n = (cls_name or "").lower()
        if "fire" in n or "flame" in n:
            return "fire"
        # 香烟单独成一类：它有自己的模型、自己的阈值，
        # 但判定流程完全沿用烟的那套（时序 + LLM），不另起一套逻辑。
        if "cig" in n or "tobacco" in n:
            return "cigarette"
        if "smoke" in n:
            return "smoke"
        return "fire" if cls_id == 0 else "smoke"

    @staticmethod
    def _shape_ok(box, kind: str, img_w: int, img_h: int,
                  cfg: dict) -> bool:
        """按目标形态做物理约束 —— 专门用来挡"看起来像但物理上不可能"的框。

        为什么需要这一层
        ----------------
        香烟模型对白衣服、手指、窗帘边缘都会给 0.3~0.5 的置信度，
        光靠阈值压不干净（真烟的置信度也就 0.5 上下，压过头就漏了）。
        但香烟有确定的长宽比和尺寸范围：
          · 是细长条，长宽比 2~12
          · 一支烟在画面里占比很小（0.01%~2%）
        不符合这两条的，无论置信度多高都不可能是香烟。
        这是用物理常识补模型精度，比硬调阈值划算得多。
        """
        x1, y1, x2, y2 = (float(v) for v in box)
        w = max(1.0, x2 - x1)
        h = max(1.0, y2 - y1)
        area_pct = (w * h) / max(1.0, img_w * img_h)
        if kind == "cigarette":
            ar = max(w, h) / min(w, h)
            # 放宽到 1.5~15：原来 2.0 会把"侧对镜头、几乎成一条线"的烟
            # 以及"正对镜头、看起来偏方"的烟头挡掉。交付场景宁可多放进来。
            lo = float(cfg.get("cig_aspect_min", 1.5))
            hi = float(cfg.get("cig_aspect_max", 15.0))
            if not (lo <= ar <= hi):
                return False
            if not (float(cfg.get("cig_area_min", 0.00002))
                    <= area_pct <= float(cfg.get("cig_area_max", 0.05))):
                return False
        return True

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

        ⚠️ 返回 True 时**由本函数自己记账**。以前是各调用点自己 append，结果
        新加的分支忘了写，去重表永远是空的 = 等于没去重（实测一盏灯 8 段轨迹
        推了 8 张卡片）。记账这事必须和判定绑在一起，不能交给调用方。
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
        self._recent_notices.append((ts, tuple(float(v) for v in box)))
        return True

    def _make_event(self, kind, level, tid, t, pkt, score, box, reasons) -> Event:
        # notice / cleared 是临时提示，**不占用正式告警的去重额度** ——
        # 否则 notice 会把后面真正的 alarm 挡在冷却期外面，等于永远升级不了。
        if level in ("warning", "alarm"):
            self._fired.add(tid)
            self._recent_alarms.append((pkt.ts, kind, tuple(float(v) for v in box), tid))
        # 规则层**不允许**单方面把目标定性成"火情"。
        #
        # 实测事故：人脸被强光打到 conf 0.47，因为低于 llm_min_conf 没送去裁判，
        # 规则层就凭"连续 6 帧命中"直接推了「火情」——值班员看到的就是一条
        # 明明白白的误报。YOLO 只能回答"这里像火"，不能回答"这就是火"，
        # 定性必须由裁判做，或者置信度高到无需裁决。
        if level == "alarm" and float(score) < float(self.cfg.get("llm_sure_conf", 0.85)):
            self.stats["alarm_downgraded"] = self.stats.get("alarm_downgraded", 0) + 1
            level = "notice"
            reasons = [f"置信 {score:.2f} 不足以直接定性，待人工确认"] + list(reasons)[:1]

        snap = crop_box(pkt.image, box, self.cfg["snapshot_margin"],
                        self.cfg["snapshot_max_side"],
                        int(self.cfg.get("snapshot_min_side", 220)))
        # 同时留一张标了框的全景图：ROI 只能看"是什么"，全景才能看"在哪"
        full = full_view(pkt.image, box, self.cfg["snapshot_full_max_side"])
        ev = Event(kind=kind, level=level, cam_id=pkt.cam_id, track_id=tid,
                   frame_idx=pkt.idx, t_first=t.first_t, t_alarm=pkt.ts,
                   score=float(score), box=tuple(float(v) for v in box),
                   reasons=list(reasons), snapshot=snap, snapshot_full=full,
                   meta={"hits": t.hits, "n_samples": len(t.samples),
                         # 原图尺寸：导出正样本时要把像素框换算成 YOLO 归一化坐标
                         "img_w": int(pkt.image.shape[1]),
                         "img_h": int(pkt.image.shape[0])})
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
        self._saved_tids &= live
        self._recheck_done &= live
        self._prefiltered &= live
        for k in [k for k in self._arbiter_done if k.rsplit(":", 1)[-1].isdigit()
                  and int(k.rsplit(":", 1)[-1]) not in live]:
            self._arbiter_done.pop(k, None)
            self._dismissed.pop(k, None)
        # 区域记忆定期落盘（每 60 秒一次），别每帧写盘
        if self.lights is not None:
            now = time.time()
            if now - self._mem_saved_ts > 60.0:
                self._mem_saved_ts = now
                self.lights.save()

    # ------------------------------------------------------------ 区域记忆
    @staticmethod
    def _norm_box(box, h: int, w: int) -> tuple[tuple, float]:
        """像素框 → 归一化框 + 归一化面积。"""
        if h <= 0 or w <= 0:
            return (0.0, 0.0, 0.0, 0.0), 0.0
        x1, y1, x2, y2 = [float(v) for v in box]
        nb = (x1 / w, y1 / h, x2 / w, y2 / h)
        return nb, max(0.0, (nb[2] - nb[0]) * (nb[3] - nb[1]))

    def _region_muted(self, cam: str, box_n, ts: float) -> bool:
        """同一块地方刚被裁判判过吗？

        键必须是**区域**而不是轨迹 ID：路灯会被遮挡/失配切成一段段新轨迹，
        按 ID 去重等于没去重 —— 每段都重新问一次 LLM，一夜几十次。
        """
        mute = float(self.cfg["llm_region_mute_sec"])
        if mute <= 0:
            return False
        keep = []
        hit = False
        for c, b, t in self._region_judged:
            if ts - t > mute:
                continue                     # 过期，顺手清掉
            keep.append((c, b, t))
            if not hit and c == cam and iou(b, box_n) >= 0.5:
                hit = True
        self._region_judged = keep
        return hit

    def _count_saved(self, tid: int) -> None:
        """记一次"少问一嘴"。按轨迹只计一次 —— 按帧计会把 300 帧说成 300 次省单，
        那是自欺欺人：同一帧不省也不会去调。"""
        if tid not in self._saved_tids:
            self._saved_tids.add(tid)
            self.stats["llm_saved"] += 1

    def _memory_check(self, pkt: FramePacket, t: Track, box_n, area_n: float,
                      flicker_label: str | None) -> str | None:
        """这块地方是不是「已确认不是火」？是则返回理由（调用方别再送审/报警）。

        安全阀（很重要，别删）：外形变了就不认记忆。
          · 面积变化超 `light_memory_area_tol` → 不是原来那东西了
            （路灯被点着、火烧大了、相机被转了）
          · 闪烁升到 fire 档 → 灯不会抖，抖了就说明情况变了
        命中这两条就把记忆**作废**，恢复完整流程（该问 LLM 就问，该报就报）。
        记忆只配省"确认过的旧情况"的钱，不能替一个正在变化的目标下结论。
        """
        if self.lights is None:
            return None
        spot = self.lights.match(pkt.cam_id, box_n)
        if spot is None:
            return None

        changed = not self.lights.still_same(spot, box_n, area_n)
        if flicker_label == "fire":
            changed = True
        if changed:
            n = self.lights.forget(pkt.cam_id, box_n)
            self.stats["memory_reopened"] += n
            return None

        self.lights.touch(spot)
        self.stats["memory_hits"] += 1
        # 只统计"这条轨迹一次都没问过 LLM、全靠记忆挡下来的"
        if f"{pkt.cam_id}:{t.tid}" not in self._arbiter_done:
            self._count_saved(t.tid)
        # 留痕按**时间窗口**给，不按轨迹：一顿路灯一夜被切成几十段，
        # 每段推一条 notice 就是把屏幕刷满，那和误报没区别。
        if not self.lights.should_notice(spot, float(self.cfg["light_memory_notice_sec"])):
            return "silent"
        return (f"该位置此前已确认是{spot.label}（累计 {spot.hits} 次），"
                f"外形未变，本次不再送 LLM 复核")

    # ------------------------------------------------------------ LLM 裁判
    @staticmethod
    def _looks_static_light(feat) -> bool:
        """时序上看起来像固定光源（灯/反光/手电筒），纯本地判断，不花钱。

        判据来自实测：手电筒和灯具的特征是面积不涨、亮度不抖、边缘稳定，
        而燃烧会扩散、会跳。同时要求已经稳定存在足够久 —— 真火苗刚起那一两秒
        也可能很稳，不能凭这个就认定没事。

        ⚠️ 命中后**只降级为"待确认"，绝不排除**。漏检的代价远大于多问几次，
        所以这道闸只用来省"问模型"的钱，不用来省"看见"的灵敏度。
        """
        if feat is None:
            return False
        try:
            return (abs(float(feat.area_growth)) < 0.05
                    and abs(float(feat.gray_trend)) < 1.0
                    and abs(float(feat.edge_trend)) < 0.02
                    and float(feat.duration) > 8.0)
        except (TypeError, ValueError):
            return False

    def _defer(self, kind, tid, t, pkt, score, box, reasons) -> bool:
        """把定级交给 LLM 裁判。返回 True = 已送审/正在等，本帧不要发事件。

        返回 False 的两种情形都**必须**让调用方继续走规则路径：
          · 没挂裁判
          · 裁判拒收（没 Key / 超额 / 去重 / 排队满）
        否则 LLM 一挂告警就全停了 —— 这是最不能接受的事故。
        """
        if self.arbiter is None:
            return False

        # 已经被"合并送审"带走了 —— 不要再单独送一次
        _ps = self._peer_suppressed.get(tid)
        if _ps is not None and (pkt.ts - _ps) < 3.0:
            return True
        if _ps is not None:
            self._peer_suppressed.pop(tid, None)     # 过期清理

        key = f"{pkt.cam_id}:{tid}"
        if key in self._pending:              # 已在等裁决，别重复提交
            return True
        done = self._arbiter_done.get(key)
        if done is not None and (pkt.ts - done) < self.cfg["llm_mute_sec"]:
            return True                       # 裁判已判过，规则层闭嘴
        # 同一块地方刚判过就别再花钱 —— 键按区域，不按轨迹 ID。
        # 这条比上面那条更重要：路灯被切成新轨迹时，`done` 查不到（键变了），
        # 只有区域匹配才拦得住。
        h, w = pkt.image.shape[:2]
        box_n, area_n = self._norm_box(box, h, w)
        # 高置信目标无视区域静音：静音键是"区域"而不是"目标"，
        # 同一块地方刚才可能只是在墙上反光被判过 light，随后真火出现在同一位置
        # 就会被一起静音 —— 那是实打实的漏报。置信度够高就重新问一次。
        high_bypass = float(self.cfg.get("llm_mute_bypass_conf", 0.80))
        if score < high_bypass and self._region_muted(pkt.cam_id, box_n, pkt.ts):
            self._count_saved(tid)
            return True

        # ⓪ 同位置多类别：合并成一次送审
        #
        # 主模型会把白色香烟误检成 fire，香烟模型又会单独出一个 cigarette，
        # 两条轨迹框在同一块地方。如果各送一次，LLM 会看到两张几乎一样的图、
        # 分别判成"火"和"香烟"，业务上就成了自相矛盾的两个结论。
        # 合并一次送审、把两个类别都摆到它面前，它才能给出一个统一判断。
        _peers = []
        for _otid, _ot in getattr(self.tracker, "tracks", {}).items():
            if _otid == tid or _ot.last_box is None or not _ot.samples:
                continue
            if _ot.cls_name == t.cls_name:
                continue
            if iou(box, _ot.last_box) > 0.35:
                _peers.append(_ot)
        if _peers:
            _desc = "、".join(
                f"{p.cls_name}（置信 {p.samples[-1].conf:.2f}）" for p in _peers
            )
            reasons = list(reasons) + [
                f"⚠️ 同一位置另检出 {_desc}。这几个框高度重叠，很可能是"
                f"**同一个物体被不同模型判成了不同类别**。请只给一个结论："
                f"这到底是哪一种？（例如白色细长条端头有红点 = 香烟；"
                f"手指上有会飘动的橙黄火苗 = 火；两者都有 = 各自判断）"
            ]
            for _p in _peers:
                self._peer_suppressed[_p.tid] = pkt.ts

        # ① 固定光源预筛：纯本地、零成本。命中就交回规则层（最多到"待确认"），
        #    既不花钱，也不会把目标从屏幕上抹掉。
        if (self.cfg.get("static_prefilter", True)
                and score < float(self.cfg.get("llm_sure_conf", 0.92))
                and self._looks_static_light(extract_features(t.samples))):
            self.stats["static_prefiltered"] = self.stats.get("static_prefiltered", 0) + 1
            self._prefiltered.add(tid)
            return False

        # 硬门槛：连「画框/告警」的门槛都没到，就不该占用裁判。
        #
        # 用户观察到"框还是灰的就推给 DeepSeek 了"—— 根因是送审下限
        # llm_min_conf(0.30) 比 fire_conf(0.35) 还低，于是 0.30~0.35 的目标
        # 画的是灰框、却照样被送审。灰色框的语义是"只是看到了、不作主张"，
        # 那就不该花这笔钱，也不该在裁判那里刷存在感。
        floor = float(self.cfg.get(f"{kind}_conf", 0.35))
        if float(score) < floor:
            self.stats["below_floor_skipped"] = \
                self.stats.get("below_floor_skipped", 0) + 1
            return False

        # ② 置信度闸门（省钱的核心）：只在规则拿不准的区间才花钱
        sure = float(self.cfg.get("llm_sure_conf", 0.85))
        low = float(self.cfg.get("llm_min_conf", 0.50))
        if score >= sure or score < low:
            self.stats["llm_gated"] = self.stats.get("llm_gated", 0) + 1
            return False          # 交回规则层，一次 API 都不花

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
                              self.cfg["snapshot_max_side"],
                              int(self.cfg.get("snapshot_min_side", 220))),
            score=float(score), reasons=list(reasons),
            hits=t.hits, n_samples=len(t.samples),
            box_n=box_n, area_n=area_n)
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
        # 记住"这块地方刚判过"（区域键，跨轨迹有效）
        self._region_judged.append((info.cam_id, info.box_n, info.ts))
        if len(self._region_judged) > 200:
            self._region_judged = self._region_judged[-200:]
        # 判"不是火/烟"的，落盘记住这块地方：下次同位置的轨迹直接跳过 LLM
        if self.lights is not None and res.is_dismiss:
            self.lights.remember(info.cam_id, info.box_n, info.area_n, res.label)
            self.lights.save()
        reasons = ([f"LLM 判为 {res.label}（risk={res.risk}，conf={res.confidence:.2f}）："
                    f"{res.reason}"] + info.reasons[:2])

        # LLM 自己都说没把握、且置信度很低 → 静默丢弃，不弹卡片。
        # 这类事件的本质是"模型看了但没看出来"，推给人只是噪音。
        if (res.label == "uncertain"
                and float(res.confidence) < float(self.cfg.get("uncertain_silent_conf", 0.50))):
            self.stats["uncertain_silent"] = self.stats.get("uncertain_silent", 0) + 1
            return None

        if res.is_dismiss:
            # ⚠️ 这里**不能** return None。判成蒸汽就一声不响，等于给真烟开了后门：
            # 规则层或 LLM 只要错一次，管理员屏幕上什么都没有，事故就发生在静默里。
            # 所以降级也要推一条，让前端留下痕迹、让人可以自己复核。
            self.stats["llm_dismissed"] += 1
            self._dismissed[key] = info.ts         # 记下降级时刻，供兜底复核计时
            # 规则层非常肯定（>= llm_override_conf）而 LLM 说不是 → 两个模型打架。
            # 这时候不能听 LLM 的把它直接抹掉，降成"待确认"让人扫一眼。
            if float(info.score) >= float(self.cfg.get("llm_override_conf", 0.85)):
                self.stats["llm_overridden"] = self.stats.get("llm_overridden", 0) + 1
                return Event(kind=info.kind, level="notice", cam_id=info.cam_id,
                             track_id=info.tid, frame_idx=info.frame_idx,
                             t_first=info.t_first,
                             t_alarm=pkt.ts if pkt is not None else info.ts,
                             score=float(info.score), box=info.box,
                             reasons=[f"LLM 判为{res.label}，但检测置信 {info.score:.2f} 很高，"
                                      f"冲突未决，请人工确认"] + info.reasons[:1],
                             snapshot=info.snapshot,
                             meta={"hits": info.hits, "n_samples": info.n_samples,
                                   "llm_label": res.label, "llm_risk": res.risk,
                                   "llm_source": res.source,
                                   "llm_latency": round(res.latency_s, 2),
                                   "overridden": True})
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
        # ── 事件类别以 LLM 的判定为准，不再沿用检测类别 ──────────
        #
        # 用户的截图暴露了这个问题：LLM 明明判了 smoke（理由写得很准——
        # "手指夹白色卷烟，端头有褐色燃点，没有飘动火苗"），事件标签却还是
        # "火情"。因为 kind 一直用的是检测类别（主模型把香烟误检成 fire），
        # LLM 的结论只影响了 level，没影响 kind。
        #
        # 现在的规则：
        #   LLM 判 fire    -> kind 一定是 fire
        #   LLM 判 smoke   -> 保留检测给出的细分（cigarette 的还是 cigarette，
        #                     否则是 smoke）。这样香烟仍然显示"香烟"而不是"烟情"。
        #   其它（light/other/steam/uncertain）-> 保持原类别，只影响降级逻辑
        if res.label in ("fire", "smoke", "cigarette"):
            kind_out = res.label          # LLM 说了算，它现在能直接判 cigarette
        else:
            kind_out = info.kind

        if res.label in ("fire", "smoke", "cigarette"):
            level = "alarm" if res.risk == "high" else "warning"
        else:                                  # uncertain → 不硬判，推给人看
            level = "notice"

        now_ts = pkt.ts if pkt is not None else info.ts
        if level != "notice" and not self._can_alarm(info.kind, info.tid, now_ts, info.box)[0]:
            return None

        ev = Event(kind=kind_out, level=level, cam_id=info.cam_id, track_id=info.tid,
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
            # 形态不合物理常识的直接丢（只有香烟有约束，火/烟原样通过）
            h_img, w_img = pkt.image.shape[:2]
            # 形态校验**只做标记，不拦截**。
            #
            # 之前形态不符直接 continue，等于规则层替 LLM 下了"这不是香烟"
            # 的结论 —— 而形态判断偶尔会把真烟挡掉（侧对镜头几乎成一条线、
            # 正对镜头看着偏方、只露出一截）。用户明确要求"所有内容都得有
            # LLM 裁判、香烟必须不能漏"，所以这里改成记一个标记，
            # 把"形态可疑"作为提示一起送给 LLM 决断，而不是自己拍板。
            shape_ok = self._shape_ok(t.last_box or (0, 0, 1, 1), kind,
                                      w_img, h_img, cfg)
            if not shape_ok:
                self.stats["shape_flagged"] = self.stats.get("shape_flagged", 0) + 1
                # 形态不符的对象**不送 LLM**（省配额、压误报），
                # 但也**不静默丢弃**：
                #   · 它仍然留在 tracks 里 -> 画面照样画灰框，人看得见
                #   · 计入 shape_flagged 统计 -> 事后能查"今天挡了多少"
                #   · 一旦特征变化（变长/变细/面积涨）下一帧就会重新走完整流程
                # 这就是"过滤但不瞒报"：系统不替谁下结论，只是不花 LLM 的钱。
                continue
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

                # 附上"同一位置其他类别的检测"作为交叉证据。
                #
                # 主模型（fire_smoke）会把白色细长的香烟误检成 fire —— 这是
                # 两个模型训练数据不同带来的必然重叠。此时香烟模型往往也在
                # 同一位置给出 cigarette，但那是一条独立轨迹，LLM 原本看不到。
                # 把这条客观事实告诉它，让它自己综合判断，而不是由规则层
                # 擅自把 fire 改成 cigarette。
                for _d in dets:
                    if _d.cls_name == t.cls_name:
                        continue
                    ix = max(0.0, min(_d.xyxy[2], t.last_box[2]) - max(_d.xyxy[0], t.last_box[0]))
                    iy = max(0.0, min(_d.xyxy[3], t.last_box[3]) - max(_d.xyxy[1], t.last_box[1]))
                    _inter = ix * iy
                    _u = (_d.area + t.samples[-1].area - _inter)
                    if _u > 0 and _inter / _u > 0.3:
                        fire_reasons.append(
                            f"⚠️ 同一位置另检出 {_d.cls_name}（conf {_d.conf:.2f}），"
                            f"可能与火模型重复判定，请结合画面核对")
                        break

                fv = (judge_fire(extract_features(t.samples),
                                 area_px=t.samples[-1].area, thr=self.temporal_thr)
                      if cfg["flicker_gate"] else None)

                # 这块地方以前被确认过"不是火"吗（路灯/招牌/反光）？
                # 是 → 不再花钱问 LLM，一条轨迹只留一条提示。
                box_n, area_n = self._norm_box(t.last_box, h, w)
                mem = self._memory_check(pkt, t, box_n, area_n,
                                         fv.label if fv is not None else None)
                if mem:
                    # 记忆命中一律静默跳过，不再推"待确认"卡片。
                    # 区域记忆本身继续生效（省 LLM 调用），只是不占事件流 ——
                    # 用户已确认过这块地方是灯，反复提醒只是噪音。
                    # 外形一旦变化（面积变 >60% 或开始抖动），_memory_check 会返回
                    # None，走完整复核流程，不会漏。
                    if self.cfg.get("light_memory_notice", False):
                        if mem != "silent":
                            events.append(self._make_event(
                                "fire", "notice", t.tid, t, pkt, conf, t.last_box,
                                [mem]))
                    continue

                if fv is not None and fv.label == "static":
                    # ⚠️ 这里不能 continue 了事。灯也会烧到短路、火也可能烧得很稳，
                    # 判错一次就是漏报。所以降级留痕 + 送 LLM 复核，双保险。
                    self.stats["static_light_blocked"] += 1
                    if t.tid not in self._static_noticed and t.tid not in self._fired:
                        self._static_noticed.add(t.tid)
                        # 留痕也要按**区域**去重，不能按轨迹 ID。
                        # 路灯一夜被切成几十段是常态（遮挡/失配/conf 抖都会换 ID），
                        # 按 ID 记账就是几十张卡片 —— 用户原话"不要多次上报"。
                        # `_can_notice` 命中同区域会续期，所以"这块地方一直亮着"
                        # 只有一条；灯真灭了 30 秒之后才允许再提。
                        sent = self._defer("fire", t.tid, t, pkt, conf, t.last_box,
                                           ["规则层疑为静态光源，请判断是火还是灯"])
                        # 先送审、不提前发待确认：原来"发notice+同时送审"会让同一件事出两条卡片"
                        # （先蓝的待确认、再绿的已排除），看起来就是通知满天飞。送审失败才退化成待确认。
                        if not sent and self._can_notice(pkt.ts, t.last_box):
                            events.append(self._make_event(
                                "fire", "notice", t.tid, t, pkt, conf, t.last_box,
                                ["检出火焰但亮度恒定，疑似灯光/反光（裁判不可用，请人工确认）"] + list(fv.reasons[:1])))
                    continue          # static 目标已送审，等裁决
                if self._defer("fire", t.tid, t, pkt, conf, t.last_box, fire_reasons):
                    continue          # 已送审，等 LLM 裁决再定级
                if t.tid in self._prefiltered:
                    # 固定光源预筛拦下：静默跳过，不推"待确认"占事件流。
                    # 检测和追踪照常，特征一变（面积涨/亮度抖）下次自动走完整流程。
                    continue
                events.append(self._make_event(
                    "fire", "alarm", t.tid, t, pkt, conf, t.last_box,
                    fire_reasons + ([] if (fv is None or fv.label != "fire")
                                    else list(fv.reasons[:1]))))
                continue

            # ── 香烟：独立路径，直接送 LLM ──────────────────────
            #
            # 香烟**不能套用烟雾的时序判据**。那套判据是为"会膨胀、会上飘、
            # 边缘逐渐模糊"的烟云设计的：看面积增速、上飘趋势、边缘衰减。
            # 而一支香烟只有二十几像素、细长、不膨胀、不上飘 —— 套进去
            # 永远判不出 "smoke"，于是每个香烟目标都在后面那道
            # `if v.label != "smoke"` 处被清零，**从头到尾一次都没送过 LLM**。
            #
            # 这就是用户观察到的"香烟一直不触发 LLM"。
            # 正确做法：香烟过门槛就直接交给 LLM —— 它判断"这是香烟还是
            # 别的细长物"靠的是截图里的视觉形态，不需要时序统计量。
            if kind == "cigarette":
                if conf < cfg.get("cigarette_conf", 0.03):
                    self._smoke_streak[t.tid] = 0
                    continue
                self._smoke_streak[t.tid] = self._smoke_streak.get(t.tid, 0) + 1
                # 连续 2 帧坐实就送审（和烟一致），避免单帧噪声
                if self._smoke_streak[t.tid] < 2:
                    continue
                if self._defer(kind, t.tid, t, pkt, conf, t.last_box,
                               [f"连续 {self._smoke_streak[t.tid]} 帧检出香烟"
                                f"（conf {conf:.2f}），请判断是香烟还是别的细长物体"]):
                    continue
                # 裁判不可用 -> 降级为待确认，但**绝不静默丢弃**
                if self._can_notice(pkt.ts, t.last_box):
                    self.stats["notices"] += 1
                    events.append(self._make_event(
                        kind, "notice", t.tid, t, pkt, conf, t.last_box,
                        [f"连续 {self._smoke_streak[t.tid]} 帧检出香烟"
                         f"（conf {conf:.2f}），裁判不可用，请人工确认"]))
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

            # 这块地方以前判过"不是火灾风险"（蒸汽源/反光）？是就不再送审。
            # 放在时序判别之后：烟的记忆只在"规则层也不认为是烟"时才敢省，
            # 否则等于用旧结论捂住一个正在长大的羽流。
            if v.label != "smoke":
                box_n, area_n = self._norm_box(t.last_box, h, w)
                mem = self._memory_check(pkt, t, box_n, area_n, None)
                if mem == "silent":
                    continue
                if mem:
                    events.append(self._make_event(
                        kind, "notice", t.tid, t, pkt, v.score, t.last_box,
                        [mem, "不是排除：外形一旦变化会自动恢复复核"]))
                    continue

            # ── 极快模式的第一级：先发 notice，不等时序 ──────────
            # 之后**不 continue**，继续往下走正规判别：
            # 时序确认了 → 升级成 alarm；判成蒸汽 → 发 cleared 撤销这条 notice。
            if cfg["fast_smoke"]:
                self._notice_streak[t.tid] = self._notice_streak.get(t.tid, 0) + 1
                if (self._notice_streak[t.tid] >= cfg["fast_smoke_confirm"]
                        and t.tid not in self._noticed and self._can_notice(pkt.ts, t.last_box)):
                    self._noticed.add(t.tid)
                    self.stats["notices"] += 1
                    events.append(self._make_event(
                        kind, "notice", t.tid, t, pkt, conf, t.last_box,
                        [f"极快模式：连续 {self._notice_streak[t.tid]} 帧检出烟"
                         f"（conf {conf:.2f}），尚未做蒸汽判别，需人工确认"]))
                    # notice 之后立刻送审：这才是"YOLO 第一轮可疑就交给 LLM"的地方。
                    # notice 已经让值班员看到画面了，LLM 的结果用来升级或压掉。
                    self._defer(kind, t.tid, t, pkt, conf, t.last_box,
                                [f"极快模式第一轮可疑（conf {conf:.2f}）"])

            if v.label == "steam":
                self.stats["steam_suppressed"] += 1
                self._smoke_streak[t.tid] = 0
                # ⚠️ 同样不能静默。规则层判蒸汽只是"倾向"，不是"结论"——
                # 面积/亮度这几个手工特征分不开真烟和蒸汽，判错一次就是漏报。
                # 所以每条轨迹降级时推一条 notice 到前端，标注判据，让人自己看。
                if t.tid not in self._steam_noticed and t.tid not in self._fired:
                    self._steam_noticed.add(t.tid)
                    # 同静态灯：一个蒸汽口一夜也被切成几十段，按轨迹记账就是刷屏。
                    if self._can_notice(pkt.ts, t.last_box):
                        self.stats["steam_notices"] += 1
                        events.append(self._make_event(
                            kind, "notice", t.tid, t, pkt, v.score, t.last_box,
                            ["规则层倾向判为蒸汽/水雾，已降级（不是排除）"] + list(v.reasons[:2])))
                    self._defer(kind, t.tid, t, pkt, v.score, t.last_box,
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
                    kind, "notice", t.tid, t, pkt, v.score, t.last_box,
                    [f"判为非火灾风险后仍持续 {held:.0f}s 未消散（共 {pkt.ts - t.first_t:.0f}s），"
                     f"请人工确认是否误判"] + list(v.reasons[:2])))
                continue
            if self._defer(kind, t.tid, t, pkt, v.score, t.last_box, reasons):
                continue                  # 已送审，等 LLM 裁决再定级
            events.append(self._make_event("smoke", level, t.tid, t, pkt,
                                           v.score, t.last_box, reasons))

        self._prune()
        # ── 光敏引擎：补上 YOLO 漏掉的亮斑 ────────────────────
        # 只加进返回给画面的 dets，**不参与追踪、不参与判定**。
        # 它的语义是"这里有团光，但我不知道是什么"，所以一律灰框、
        # 置信度为 0，绝不允许它触发任何告警路径。
        # 画框用 **tracks 的 last_box**，不用原始 dets。
        #
        # 原始检测框每帧都在抖（YOLO 的回归本身就有几个像素的抖动，加上
        # 框大小变化，视觉上就是"一闪一闪"）。而 Track.last_box 是经过
        # Kalman 平滑的，位置连续、尺寸稳定 —— 用户要的"框稳定、平滑"
        # 本来就已经算好了，只是之前画的是原始框没用上。
        #
        # 顺带一个好处：轨迹有 coasting，目标被短暂遮挡时框不会立刻消失。
        # 只画**本帧匹配上**的轨迹（misses == 0）。
        #
        # 这是"框分裂"的彻底解法。原来把所有存活轨迹都画出来 —— 包括
        # 这一帧没匹配上的旧轨迹（它还要活 0.25 秒）和刚建立的新轨迹，
        # 于是屏幕上同时出现"停在原地的旧框"和"跟着目标走的新框"。
        #
        # 只画本帧有检测支撑的轨迹，屏幕上任何时刻一条轨迹最多一个框，
        # 物理上不可能分裂。代价是目标短暂丢失时框会立刻消失 ——
        # 对监控来说这比"两个框"更可接受（消失说明真的没看见）。
        draw_dets: list[Detection] = []
        for _t in tracks:
            # 允许刚好 1 帧没匹配上仍然画出来。
            #
            # 纯 misses==0 会出现"闪一下又消失"：只要某一帧检测抖动一下
            # 没接上，框就瞬间没了、下一帧又回来，肉眼看到的是闪烁。
            # 放 1 帧宽限能吃掉这种单帧抖动，而 2 帧以上的断档仍然会隐藏 ——
            # 那边通常意味着目标真的走了或真的没检出，继续画旧位置反而是错的。
            if _t.misses > 1:
                continue
            if _t.last_box is None or not _t.samples:
                continue
            draw_dets.append(Detection(_t.cls_id, _t.cls_name,
                                       float(_t.samples[-1].conf), _t.last_box))

        # 光敏引擎降频：每 2 帧跑一次。
        # 它只回答"哪里有亮光"，亮光不会在 33ms 内出现又消失，
        # 逐帧跑纯属浪费 GIL —— 而 GIL 正是画面卡顿的元凶。
        self._ls_n = getattr(self, "_ls_n", 0) + 1
        if cfg.get("lightspot", True) and self._ls_n % 2 == 0:
            try:
                from .lightspot import detect as _ls_detect
                spots = _ls_detect(
                    pkt.image,
                    v_min=int(cfg.get("lightspot_v_min", 205)),
                    min_area=int(cfg.get("lightspot_min_area", 12)),
                    max_area_pct=float(cfg.get("lightspot_max_area_pct", 0.25)),
                )
                if spots:
                    # 与已有检测框重叠的不要重复画
                    kept = []
                    for sp in spots:
                        sx1, sy1, sx2, sy2 = sp["box"]
                        dup = False
                        for d in dets:
                            dx1, dy1, dx2, dy2 = d.box
                            ix = max(0.0, min(sx2, dx2) - max(sx1, dx1))
                            iy = max(0.0, min(sy2, dy2) - max(sy1, dy1))
                            inter = ix * iy
                            union = ((sx2 - sx1) * (sy2 - sy1)
                                     + (dx2 - dx1) * (dy2 - dy1) - inter)
                            if union > 0 and inter / union > 0.35:
                                dup = True
                                break
                        if not dup:
                            kept.append(sp)
                    if kept:
                        draw_dets = list(dets) + [
                            Detection(cls_id=-1, cls_name=f"光:{sp['color']}",
                                      conf=0.0, box=sp["box"])
                            for sp in kept
                        ]
                        self.stats["lightspots"] = self.stats.get("lightspots", 0) + len(kept)
            except Exception:                 # noqa: BLE001
                pass                          # 光敏引擎坏了也绝不能拖垮主流程

        return FrameResult(pkt.idx, pkt.ts, draw_dets, tracks, verdicts, events,
                           frozen=False, nearby=nearby_map)

    # ------------------------------------------------------------ 多帧
    def run(self, source: CameraSource, *, on_event=None, max_frames: int | None = None,
            every: int = 1, quiet: bool = False, on_frame=None) -> dict:
        """把一路视频源跑到底。

        ⚠️ 检测跑在**独立线程**里，主循环只负责采集 + 推画面。

        为什么必须拆线程
        ----------------
        原来检测是在主循环里同步调用的：算一帧要 11~60ms，这段时间主循环
        完全停住、不能推画面 —— 表现就是"每次检测画面就卡一下"。
        用户的原话是"调用检测时会卡顿"，观察得完全准确。

        拆开之后：
          · 主循环每来一帧就推画面，**永不阻塞**
          · 检测线程从队列里拿最新帧处理，跟不上就丢帧（宁丢不积压）
          · 画面用最近一次检测结果，最多滞后一个检测周期（约 200ms）
            这种滞后对监控来说无所谓 —— 人眼看的是"发生了什么"，
            不是"精确到毫秒的同步"

        every：抽帧间隔（3 = 每 3 帧送一次检测）。烟/火都是慢过程。

        on_frame：每帧回调 (pkt, res)，给实时画面用。异常一律吞掉 ——
                  可视化挂了不能连累检测。
        """
        import queue as _q
        import threading as _th

        t0 = time.time()
        all_events: list[Event] = []
        n = 0
        empty = 0
        last_idx = -1
        last_det_t = 0.0
        force_detect_s = 0.6      # 重复帧期间最多隔这么久强制重检一次
        detq: _q.Queue = _q.Queue(maxsize=2)   # 只留最新两帧
        stop = _th.Event()
        shared: dict = {"res": None}          # 检测线程写，主线程读

        def _detect_loop():
            while not stop.is_set():
                try:
                    pkt = detq.get(timeout=0.1)
                except _q.Empty:
                    continue
                if pkt is None:
                    break
                try:
                    res = self.process(pkt)
                except Exception:                  # noqa: BLE001
                    continue
                shared["res"] = res
                self._last_dets = res.dets
                for ev in res.events:
                    all_events.append(ev)
                    if not quiet:
                        print(ev, flush=True)
                    if on_event is not None:
                        try:
                            on_event(ev, res)
                        except Exception:          # noqa: BLE001
                            pass

        th = _th.Thread(target=_detect_loop, name="detect", daemon=True)
        th.start()

        def _push(pkt):
            """推一帧画面。检测还没出结果时先推空框，别让画面黑着。"""
            if on_frame is None:
                return
            res = shared["res"]
            if res is None:
                res = FrameResult(pkt.idx, pkt.ts, [], [], {}, [])
            try:
                on_frame(pkt, res)
            except Exception:                      # noqa: BLE001
                pass

        try:
            while not source.closed:
                pkt = source.read()
                if pkt is None:
                    empty += 1
                    if empty > 1000:
                        break
                    time.sleep(0.002)
                    continue
                empty = 0

                # 重复帧：画面照推（否则前端冻住），并按周期强制重检
                if pkt.idx == last_idx:
                    _push(pkt)
                    if time.time() - last_det_t >= force_detect_s:
                        last_det_t = time.time()
                        try:
                            detq.put_nowait(pkt)
                        except _q.Full:
                            pass
                    time.sleep(0.002)
                    continue
                last_idx = pkt.idx

                # 抽帧送检：**不等结果**，丢进队列就继续
                if n % every == 0:
                    last_det_t = time.time()
                    try:
                        detq.put_nowait(pkt)
                    except _q.Full:
                        # 检测跟不上就丢这一帧。宁可少检一次，也不能让队列
                        # 积压 —— 那会导致检测永远在处理几秒前的旧画面。
                        self.stats["det_dropped"] = self.stats.get("det_dropped", 0) + 1

                _push(pkt)

                n += 1
                if max_frames is not None and n >= max_frames:
                    break
        finally:
            stop.set()
            th.join(timeout=2.0)


        elapsed = max(1e-6, time.time() - t0)
        if self.lights is not None:
            self.lights.save()          # 收尾必落盘：断电就白记了
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
