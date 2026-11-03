"""轨迹关联：把逐帧检测框串成 tracklet，喂给 `temporal.judge`。

为什么必须有这一层
------------------
`temporal.py` 要的是**一条轨迹上的时间序列**（`TrackletSample[]`），
但检测器只会吐逐帧的孤立框。中间这段没人做，所以时序判别一直是断链的。

这一层只做三件事，不做 Kalman：
1. 按类别做贪心 IoU 关联（不同类不互相抢，烟里的火不能被烟框吃掉）
2. 给每个匹配上的框算判别需要的像素统计（亮度/饱和度/边缘密度）
3. 维护轨迹生命周期（消失多久算结束）

为什么不上 Kalman / BoT-SORT
---------------------------
烟雾是**无定形**目标，Kalman 假设匀速直线运动，对膨胀扩散的烟反而引入错误预测；
本项目时序判别只看面积/亮度/清晰度的走向，不依赖精确轨迹预测。
先跑通规则版，等 tracklet 攒够数据再用 GBDT 替换判别器（M5），那时再谈更好的关联。

云台约束
--------
`moving=True` 时**冻结**：不新建轨迹、不追加样本、不让轨迹老化。
云台一转，同一条烟会被拉成多个轨迹，中途更新等于往判别器里灌脏数据。
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from .infer import Detection, iou
from .temporal import TrackletSample

MATCH_IOU = 0.15        # 关联门限；烟框逐帧抖动大，不能设高。
                        # 0.30 时目标一移动就匹配失败、另起一条新轨迹，
                        # 表现为"框分裂、旧框留在原地"。降到 0.15 让它更容易接上。
# ⚠️ 断帧容忍从 0.25 秒提到 2.0 秒。
# 0.25 秒在 10fps 下只有 2~3 帧 —— 目标稍微动快一点、被手挡一下、
# 或者检测本身偶尔漏一帧，轨迹就被判结束，下次出现时分配**新 ID**。
# 用户的原话："不要因为中间断过几帧就更换ID"。
# 2 秒（约 20 帧）能扛住短时遮挡和检测抖动，同时又不至于让已经离开的
# 目标一直占着 ID。
MAX_MISS_SEC = 1.0
                        # 目标移动但检测漏帧/匹配失败时，旧轨迹还能活多久。
                        # 原来 3.0 秒：配合 miss 预测，框会停住/漂移很久，
                        # 用户反馈"框停原地 + 飘"。现在 miss 不再预测（框诚实停住），
                        # 那停住的时间就要短 —— 缩到 1.5 秒让旧框快速消失，
                        # 又保留 1.5 秒抗遮挡/抗断帧（用户要求"断几帧不换 ID"）。
MAX_SAMPLES = 150       # 单条轨迹最多留这么多时间点，防止长跑内存爆
# ByteTrack 思想：低于此置信度的检测**不新建轨迹**（只允许匹配到已有轨迹）。
# 为什么需要：det_conf 压到 0.08 后，画面里会冒出大量 0.08~0.2 的噪声框，
# 每个都建一条轨迹，等于给"分裂/满屏框"喂料。真目标（火/烟）一旦成立
# 置信度会往上走，香烟目标分数天然低所以单列一个更低的值。
# 注意这只拦"新建轨迹"，不拦"匹配已有轨迹" —— 已确认的目标即使某一帧
# 置信度掉下去，仍能靠这个兜底接上，不会断 ID。
NEW_TRACK_MIN_CONF = {"fire": 0.15, "smoke": 0.20, "cigarette": 0.03}
# 灰度差多少才算"这一像素在变"。8 级足够滤掉压缩噪声和传感器噪声，
# 又远小于火焰跳动的幅度（实测火焰区域 65%~92% 的像素每帧都超过这个值）。
FLICKER_PIXEL_DELTA = 8


@dataclass

class _Kalman:
    """8 维 Kalman 滤波器：状态 = [cx, cy, w, h, vx, vy, vw, vh]。

    为什么必须要它
    --------------
    没有预测，匹配就只能拿"上一帧的框"去和"这一帧的检测"算 IOU。
    目标一移动，两个框就错开、IOU 归零 -> 匹配失败 -> 另起新轨迹 ->
    屏幕上出现两个框（旧的留在原地）。这就是"框分裂"的机械原因。

    Kalman 每帧先预测"目标现在大概在哪、多大"，再拿**预测框**去匹配，
    移动中的目标也能对上号。同时它对宽高也做平滑，所以框的大小会
    自适应地跟着目标变化，而不是逐帧跳变。

    为什么过程噪声给得比常规大
    --------------------------
    火和烟会膨胀、会抖动，不符合"匀速直线"假设。过程噪声大意味着
    更信任新观测、允许模型快速修正 —— 对无定形目标是合适的取舍。
    """

    __slots__ = ("x", "P")

    def __init__(self, box):
        x1, y1, x2, y2 = (float(v) for v in box)
        w = max(1.0, x2 - x1)
        h = max(1.0, y2 - y1)
        cx, cy = x1 + w / 2.0, y1 + h / 2.0
        self.x = np.array([cx, cy, w, h, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
        # 初始协方差：位置/尺寸不确定度按框大小的比例给
        self.P = np.diag([w * 0.1, h * 0.1, w * 0.1, h * 0.1,
                          w * 0.5, h * 0.5, w * 0.3, h * 0.3]) ** 2

    def predict(self, dt: float = 1.0):
        F = np.eye(8)
        F[0, 4] = F[1, 5] = F[2, 6] = F[3, 7] = dt
        self.x = F @ self.x
        # 过程噪声：位置小、尺寸大（允许膨胀/收缩）
        # 位置噪声小（压制抖动，框不乱跳），尺寸噪声大（允许膨胀/收缩），
        # 速度噪声**偏小**：火苗是"原地闪烁"、香烟是"手持微抖"，两者真实
        # 位移都很慢。速度噪声给大（之前 5）会让滤波器把"闪烁抖动"当成
        # "目标在动"，估出一条错误的速度，预测/校正时框就跟着飘 ——
        # 这正是"框漂移"的帮凶之一。压到 2.5，让滤波器更相信"慢速运动"，
        # 把抖动当噪声滤掉而不是当成速度追。
        q = np.diag([1.5, 1.5, 9.0, 9.0, 2.5, 2.5, 3.0, 3.0]) ** 2
        self.P = F @ self.P @ F.T + q
        return self.box()

    def correct(self, box):
        x1, y1, x2, y2 = (float(v) for v in box)
        w = max(1.0, x2 - x1)
        h = max(1.0, y2 - y1)
        z = np.array([x1 + w / 2.0, y1 + h / 2.0, w, h], dtype=np.float64)
        H = np.zeros((4, 8))
        H[0, 0] = H[1, 1] = H[2, 2] = H[3, 3] = 1.0
        # 观测噪声：检测框本身有几个像素的抖动，尺寸的抖动更大
        # 位置噪声从 2px 提到 5px：火苗闪烁、香烟手持微抖时，YOLO 的框
        # 帧间跳动幅度实测能到 ±8px，远大于 2px。之前把 R 设成 2，等于
        # 告诉滤波器"检测很准"，于是它把抖动当真位置、框跟着跳。提到 5
        # 更贴近真实抖动量级，滤波器会更信任历史估计、把跳动压下去。
        # 尺寸噪声 8→12：框的宽高抖动比位置更厉害，一并放大让尺寸更稳。
        R = np.diag([5.0, 5.0, 12.0, 12.0]) ** 2
        S = H @ self.P @ H.T + R
        try:
            K = self.P @ H.T @ np.linalg.inv(S)
        except np.linalg.LinAlgError:
            return self.box()
        self.x = self.x + K @ (z - H @ self.x)
        self.P = (np.eye(8) - K @ H) @ self.P
        # 宽高不能为负
        self.x[2] = max(4.0, self.x[2])
        self.x[3] = max(4.0, self.x[3])
        return self.box()

    def box(self):
        cx, cy, w, h = self.x[0], self.x[1], self.x[2], self.x[3]
        return (cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0)


@dataclass
class Track:
    """一条轨迹。`samples` 就是 temporal 层要的输入。"""

    tid: int
    cls_id: int
    cls_name: str
    samples: list[TrackletSample] = field(default_factory=list)
    last_box: tuple[float, float, float, float] | None = None
    last_t: float = 0.0
    first_t: float = 0.0
    hits: int = 0
    misses: int = 0
    fg_ratio_ema: float = 0.0     # 框落在前景掩码内的比例（指数滑动均值）
    # 轨迹**刚建立**时的前景占比（取前 3 次匹配的最大值）。
    # 为什么要单独存这个值：中值背景会把静止不动的东西慢慢吸进背景，
    # 所以一条静止的烟过 20 秒后 fg_ratio_ema 会趋 0 —— 而墙上本来就有的白斑
    # 是**从出生起**就不在前景里。只看 ema 分不出这两者，会把阴燃误杀。
    fg_ratio_first: float = 0.0
    # ── 闪烁检测用的**固定绝对坐标窗口** ──────────────────────────
    # 为什么不跟着检测框每帧变：YOLO 的框每帧宽高都不同，直接拿框内区域做
    # 逐像素差分会因为尺寸不一致而永远跳过（实测 flicker 恒为 0）。而且框在动，
    # 逐像素差分测到的就是"框在动"而不是"火在抖"。目标第一次出现时把窗口钉死，
    # 之后只在明显跑偏（中心偏移超过窗口尺寸）时才重钉 —— 测的始终是同一块空间。
    # Kalman 状态：用于"预测下一帧在哪"，匹配时用它而不是上一帧的框
    kf: "_Kalman | None" = None
    roi: "tuple[int, int, int, int] | None" = None
    prev_gray: "np.ndarray | None" = None
    prev_gray_global: float = 0.0

    @property
    def box(self):
        return self.last_box

    def duration(self) -> float:
        return max(0.0, self.last_t - self.first_t)


def box_stats(image: np.ndarray, box) -> tuple[float, float, float]:
    """框内 (平均亮度, 平均饱和度, 边缘密度)。temporal 的判别特征全靠这三个。"""
    h, w = image.shape[:2]
    x1 = int(max(0, min(w - 1, box[0])))
    y1 = int(max(0, min(h - 1, box[1])))
    x2 = int(max(x1 + 1, min(w, box[2])))
    y2 = int(max(y1 + 1, min(h, box[3])))
    crop = image[y1:y2, x1:x2]
    if crop.size == 0:
        return 128.0, 50.0, 0.0
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    sat = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)[:, :, 1]
    edges = cv2.Canny(gray, 60, 160)
    return float(gray.mean()), float(sat.mean()), float((edges > 0).mean())


def _lock_window(box, h: int, w: int, lo: int = 24, hi: int = 320):
    """按首次检测框钉下一块固定尺寸的窗口（绝对坐标）。"""
    x1, y1, x2, y2 = [int(round(v)) for v in box]
    bw = min(hi, max(lo, x2 - x1))
    bh = min(hi, max(lo, y2 - y1))
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    nx1 = max(0, min(w - bw, cx - bw // 2))
    ny1 = max(0, min(h - bh, cy - bh // 2))
    return nx1, ny1, min(w, nx1 + bw), min(h, ny1 + bh)


def _window_drifted(roi, box, ratio: float = 1.0) -> bool:
    """目标中心偏离窗口中心超过窗口尺寸的 ratio 倍 → 该重钉了。"""
    rx, ry = (roi[0] + roi[2]) / 2, (roi[1] + roi[3]) / 2
    bx, by = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    return abs(bx - rx) > (roi[2] - roi[0]) * ratio or abs(by - ry) > (roi[3] - roi[1]) * ratio


def fg_overlap(box, fg_mask: np.ndarray | None) -> float:
    """框面积中落在前景掩码内的比例。无掩码时返回 1.0（不过滤）。"""
    if fg_mask is None:
        return 1.0
    h, w = fg_mask.shape[:2]
    x1 = int(max(0, min(w - 1, box[0])))
    y1 = int(max(0, min(h - 1, box[1])))
    x2 = int(max(x1 + 1, min(w, box[2])))
    y2 = int(max(y1 + 1, min(h, box[3])))
    sub = fg_mask[y1:y2, x1:x2]
    if sub.size == 0:
        return 0.0
    return float((sub > 127).mean())


class IoUTracker:
    """贪心 IoU 关联。足够支撑规则版时序判别。"""

    def __init__(self, match_iou: float = MATCH_IOU, max_miss_sec: float = MAX_MISS_SEC):
        self.match_iou = match_iou
        self.max_miss_sec = max_miss_sec
        self.tracks: dict[int, Track] = {}
        self._next_id = 1
        self._prev_gray_full: "np.ndarray | None" = None

    def reset(self) -> None:
        self.tracks.clear()
        self._next_id = 1
        self._prev_gray_full = None

    def update(
        self,
        dets: list[Detection],
        image: np.ndarray,
        ts: float,
        moving: bool = False,
        fg_mask: np.ndarray | None = None,
    ) -> list[Track]:
        """喂一帧的检测结果，返回当前活跃轨迹。"""
        if moving:
            # 云台转动：冻结，不动任何轨迹
            return [t for t in self.tracks.values() if t.misses * 0.04 <= self.max_miss_sec]

        # 整幅灰度图一帧只算一次：既要它的均值（曝光基准），也要拿它算
        # 「整幅有多大比例的像素在变」—— 相机晃动/云影/自动曝光会让整幅一起变，
        # ROI 的变化量减掉这个基准，剩下的才是目标自己在动。
        # 实测 854x480 上 absdiff+计数 < 1ms，比逐框重算便宜得多。
        gray_full = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        g_global = float(gray_full.mean())
        frac_global = 0.0
        if self._prev_gray_full is not None and self._prev_gray_full.shape == gray_full.shape:
            frac_global = float(
                (cv2.absdiff(gray_full, self._prev_gray_full) > FLICKER_PIXEL_DELTA).mean())
        self._prev_gray_full = gray_full

        unmatched = {tid: t for tid, t in self.tracks.items()}
        used: set[int] = set()

        # ── 第一步：预测每条轨迹这一帧"应该在哪个位置、多大" ──────
        # 这是和外头专业追踪器对齐的关键一步。之前直接拿上一帧的框去
        # 算 IOU，目标一移动两个框就错开、IOU 归零 -> 匹配失败 -> 新轨迹。
        # 现在先用 Kalman 预测，再拿**预测框**匹配，移动中的目标也能接上。
        pred_box: dict[int, tuple] = {}
        for tid, t in self.tracks.items():
            if t.kf is None:
                if t.last_box is not None:
                    t.kf = _Kalman(t.last_box)
                else:
                    continue
            pred_box[tid] = t.kf.predict()

        # ── 第二步：贪心匹配（IOU 主判据 + 中心距离兜底）──────────
        #
        # 为什么光靠 IOU 不够：打火机、香烟这类小目标只有 20x20 像素，
        # 火苗一抖位置跳 10px，IOU 立刻掉到 0.1 以下 —— 明明还是同一团火，
        # 却因为"框不重叠"被判成新目标。实测一次打火冒出 #419/#421/#422
        # 三条轨迹、三张卡片，根子就在这里。
        #
        # 兜底判据：IOU 不够时看中心距离。容差取"框对角线的一半、至少 12px"，
        # 距离在容差内就给一个低分（0~0.1），确保它**永远排在纯 IOU 匹配之后** ——
        # 有正常重叠的目标不会被距离匹配抢走。
        def _score(pb, db) -> float:
            v = iou(pb, db)
            if v >= self.match_iou:
                return v
            pcx, pcy = (pb[0] + pb[2]) / 2.0, (pb[1] + pb[3]) / 2.0
            dcx, dcy = (db[0] + db[2]) / 2.0, (db[1] + db[3]) / 2.0
            pw, ph = pb[2] - pb[0], pb[3] - pb[1]
            dist = ((pcx - dcx) ** 2 + (pcy - dcy) ** 2) ** 0.5
            # 容差 = 框对角线全长（至少 28px）。
            #
            # 之前取 0.6 倍对角线，实测 20x20 的小火苗抖动 ±8px 时中心距离
            # 能到 17.8px，而容差只有 17.0 —— 差 0.8 像素就判成新目标，
            # 一次打火照样分裂成三条轨迹。小目标的抖动幅度本身就接近它的
            # 尺寸，容差必须按整条对角线给，不能打折。
            # 底线从 20 提到 32、并给对角线 1.3 倍余量：香烟手持移动时中心
            # 每帧能跳几十像素，容差太小就匹配失败 → 另起新轨迹 → 分裂
            # （用户反馈"香烟离开后分裂出一个新框"）。放宽到 1.3 倍对角线，
            # 让移动中的香烟更容易接回旧轨迹、不换 ID。
            tol = max(40.0, (pw * pw + ph * ph) ** 0.5 * 2.5)
            if dist <= tol:
                return 0.1 * (1.0 - dist / tol)      # 0~0.1，永远低于 IOU 阈值
            return 0.0

        pairs = []
        for tid, t in self.tracks.items():
            pb = pred_box.get(tid)
            if pb is None:
                continue
            for di, d in enumerate(dets):
                # ⚠️ 必须按**类名**判断同类，不能用 cls_id！
                # 主模型是 {0: fire, 1: smoke}，香烟模型是 {0: cigarette} ——
                # 火和香烟的 cls_id **都是 0**。之前用 `d.cls_id != t.cls_id`
                # 过滤，等于把"火"和"香烟"当成同一类：火检测(0)能匹配到
                # 香烟轨迹(0)、香烟检测(0)能匹配到火轨迹(0)，两条轨迹互抢、
                # 框互相"吞"，画面就出现"香烟和火混在一起、几乎重合"。
                # 这和 nms() 里"按类名分组"是同一个坑，这里漏改了。
                if str(d.cls_name).lower() != str(t.cls_name).lower():
                    continue
                v = _score(pb, d.xyxy)
                if v > 0.0:
                    pairs.append((v, tid, di))
        pairs.sort(key=lambda x: -x[0])

        for v, tid, di in pairs:
            if tid in used or di in used:
                continue
            t = self.tracks[tid]
            d = dets[di]
            smp, crop, roi = self._sample(image, d, ts, gray_full, g_global,
                                          frac_global, t.prev_gray, t.roi)
            t.samples.append(smp)
            # 目标跑出原来那块空间就重钉窗口（否则测的是"它走了"，不是"它在抖"）
            if t.roi is None or _window_drifted(t.roi, d.xyxy):
                t.roi, crop = roi, None
            t.prev_gray = crop
            t.prev_gray_global = g_global
            # 用检测框校正 Kalman，再把**平滑后的框**存为 last_box。
            # 这样画出来的是滤波结果，不是逐帧抖动的原始检测值。
            if t.kf is not None:
                t.last_box = t.kf.correct(d.xyxy)
            else:
                t.last_box = d.xyxy
            t.last_t = ts
            t.hits += 1
            t.misses = 0
            fr = fg_overlap(d.xyxy, fg_mask)
            t.fg_ratio_ema = 0.6 * t.fg_ratio_ema + 0.4 * fr if t.hits > 1 else fr
            if t.hits <= 3:                       # 出生期的峰值，用来看"它一开始是不是前景"
                t.fg_ratio_first = max(t.fg_ratio_first, fr)
            used.add(tid)
            used.add(di)
            unmatched.pop(tid, None)

        # 未匹配的轨迹老化
        for tid in list(unmatched):
            t = self.tracks[tid]
            t.misses += 1
            # miss 期间 last_box **停住（coast）**，不预测推进。
            #
            # 这是"框漂移偏离目标"的根因修复。之前试过"平均位移推进"和
            # "Kalman 预测推进"，两者都会在**目标其实静止、只是被漏检**时，
            # 把框沿一条有噪声的速度方向推出去（火苗闪烁、手持香烟微抖，
            # 速度估计本身就不可靠），下一帧重新检测到又跳回来 ——
            # 肉眼看到的就是"框飘出去又弹回来"。
            #
            # 匹配这一步本来就用 Kalman 预测框（pred_box）去接移动中的目标，
            # 所以显示框在这里停住**完全不影响跟上移动目标**：目标移动时被
            # 重新检测到，pred_box 预测在前、照样能匹配上，correct() 再把
            # 显示框拉回正确位置。停住只是"不画那个不确定的预测位置"而已。
            #
            # 代价：目标移动 + 恰好漏帧的那 1~2 帧，框会短暂停在原地，
            # 但这是"诚实停住"，比"漂到错误方向再弹回来"干净得多。
            # 火苗闪烁、香烟手持时置信度剧烈跳（0.06~0.81），漏帧都更常见，
            # 容忍更长（2秒）避免闪烁导致轨迹断裂 —— 轨迹断一次就换 ID，
            # 用户会看到"框消失又冒出新 ID"（目标丢失）。
            _miss_sec = {"fire": 2.0, "cigarette": 2.0}.get(
                str(t.cls_name).lower(), self.max_miss_sec)
            if (ts - t.last_t) > _miss_sec:
                self.tracks.pop(tid, None)

        # 未匹配的检测新建轨迹
        for di, d in enumerate(dets):
            if di in used:
                continue
            # ByteTrack：低置信度检测不新建轨迹，避免噪声框起一条轨迹造成分裂。
            if float(d.conf) < NEW_TRACK_MIN_CONF.get(str(d.cls_name).lower(), 0.0):
                continue
            tid = self._next_id
            self._next_id += 1
            t = Track(tid=tid, cls_id=d.cls_id, cls_name=d.cls_name,
                      last_box=d.xyxy, last_t=ts, first_t=ts, hits=1,
                      fg_ratio_ema=fg_overlap(d.xyxy, fg_mask))
            t.kf = _Kalman(d.xyxy)
            t.fg_ratio_first = t.fg_ratio_ema
            smp, crop, roi = self._sample(image, d, ts, gray_full, g_global,
                                          frac_global, None, None)
            t.samples.append(smp)
            t.roi = roi
            t.prev_gray = crop
            t.prev_gray_global = g_global
            self.tracks[tid] = t

        return list(self.tracks.values())

    @staticmethod
    def _sample(image: np.ndarray, d: Detection, ts: float,
                gray_full: np.ndarray | None = None,
                g_global: float = 0.0, frac_global: float = 0.0,
                prev_gray: np.ndarray | None = None,
                prev_roi: "tuple[int, int, int, int] | None" = None,
                ) -> tuple[TrackletSample, "np.ndarray | None",
                           "tuple[int, int, int, int] | None"]:
        """采一个时间点。顺带返回框内灰度小图，供下一帧算闪烁。"""
        g, s, e = box_stats(image, d.xyxy)
        cx, cy = d.center()

        flicker, crop, roi = 0.0, None, None
        if gray_full is not None:
            h, w = gray_full.shape[:2]
            roi = _lock_window(d.xyxy, h, w) if prev_roi is None else prev_roi
            x1, y1, x2, y2 = roi
            crop = gray_full[y1:y2, x1:x2].copy()
            if (prev_gray is not None and crop.size and prev_gray.shape == crop.shape):
                d_px = cv2.absdiff(crop, prev_gray)
                frac_roi = float((d_px > FLICKER_PIXEL_DELTA).mean())
                # 扣掉整幅的同步变化：相机被风吹得轻微晃动、云影飘过、自动曝光调整，
                # 都会让整幅一起变。不减掉这个基准，稳定的灯会被判成火 —— 正好搞反。
                flicker = max(0.0, frac_roi - frac_global)

        return TrackletSample(t=ts, area=d.area, cx=cx, cy=cy,
                              gray_mean=g, sat_mean=s, edge_density=e, conf=d.conf,
                              gray_global=g_global, flicker=flicker), crop, roi
