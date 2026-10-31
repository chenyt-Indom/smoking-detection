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

MATCH_IOU = 0.30        # 关联门限；烟框逐帧抖动大，不能设高
MAX_MISS_SEC = 0.8      # 连续这么久没匹配上就判轨迹结束
MAX_SAMPLES = 150       # 单条轨迹最多留这么多时间点，防止长跑内存爆
# 灰度差多少才算"这一像素在变"。8 级足够滤掉压缩噪声和传感器噪声，
# 又远小于火焰跳动的幅度（实测火焰区域 65%~92% 的像素每帧都超过这个值）。
FLICKER_PIXEL_DELTA = 8


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

        # 贪心：所有 (track, det) 对按 IoU 降序，逐个配对
        pairs = []
        for tid, t in self.tracks.items():
            if t.last_box is None:
                continue
            for di, d in enumerate(dets):
                if d.cls_id != t.cls_id:
                    continue
                v = iou(t.last_box, d.xyxy)
                if v >= self.match_iou:
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
            if (ts - t.last_t) > self.max_miss_sec:
                self.tracks.pop(tid, None)

        # 未匹配的检测新建轨迹
        for di, d in enumerate(dets):
            if di in used:
                continue
            tid = self._next_id
            self._next_id += 1
            t = Track(tid=tid, cls_id=d.cls_id, cls_name=d.cls_name,
                      last_box=d.xyxy, last_t=ts, first_t=ts, hits=1,
                      fg_ratio_ema=fg_overlap(d.xyxy, fg_mask))
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
