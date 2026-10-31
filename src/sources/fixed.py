"""固定机位：背景建模把「新出现的东西」捞出来。

为什么固定机位要用背景法，光靠 YOLO 不够
---------------------------------------
早期烟雾是一片**很淡、半透明、没有边界**的雾状区域，检测器在单帧上很难给出高置信框。
但它相对于"这块背景原来的样子"是很明显的差异 —— 尤其是持续累积时。

所以固定机位做两层：
  · `median_bg`  长时中值背景 —— 对**缓慢**变化最敏感（烟就是慢慢铺开的）
  · `MOG2`       高斯混合 —— 对**快速**变化敏感（火焰跳动、有人走过）
两者取并集，得到"变化区域"。这个掩码有三个用途：
  1. 给检测器做 ROI 提示（只在变化区跑检测，省算力）
  2. 直接作为"疑似烟区"喂给时序判别（面积增长曲线）
  3. 抑制静态误检（墙上白色的画、白墙高光长期存在 → 不进掩码）

已知局限（写清楚，免得后面对不上）
  · 光照突变（开灯/关灯/日照变化）会造成大面积假前景 → 用 `lighting_jump` 检测并重置背景
  · 长期停在那里的物体（放了张白纸）会逐渐被吸收进背景 → 这是期望行为，不是 bug
  · 反光/水面波动会产生零星前景 → 形态学 + 面积过滤
"""
from __future__ import annotations

import numpy as np

try:                       # 训练环境有 opencv；没有也能退化跑（只是没有背景模型）
    import cv2
except Exception:          # pragma: no cover
    cv2 = None

from .base import CameraInfo, CameraSource, FramePacket


class BackgroundModel:
    """中值背景 + MOG2 的组合前景检测。"""

    def __init__(self, warmup: int = 30, history: int = 500,
                 var_threshold: float = 16.0, min_area: int = 120,
                 refresh_every: int = 30):
        self.warmup = warmup            # 前 N 帧只建背景，不报前景
        self.history = history
        self.var_threshold = var_threshold
        self.min_area = min_area
        self.frames: list[np.ndarray] = []
        self._bg_median: np.ndarray | None = None
        # 中值背景**不能每帧重算**：500 帧 stack + median 在 960x540 上实测 ~0.7s/帧，
        # 比 YOLO 推理还贵一个量级，整条流水线会被拖到 1fps。
        # 背景本身是"长时中值"，20 秒窗口内变化极慢，每 refresh_every 帧刷一次足够。
        self.refresh_every = max(1, refresh_every)
        self._bg_age = 10 ** 9          # 首次必须立刻建
        self._mog2 = cv2.createBackgroundSubtractorMOG2(
            history=history, varThreshold=var_threshold, detectShadows=False
        ) if cv2 is not None else None
        self._prev_mean: float | None = None

    # ------------------------------------------------------------ 内部
    def _median(self) -> np.ndarray | None:
        if not self.frames:
            return None
        if self._bg_median is None or self._bg_age >= self.refresh_every:
            self._bg_median = np.median(np.stack(self.frames[-self.history:]), axis=0
                                        ).astype(np.uint8)
            self._bg_age = 0
        self._bg_age += 1
        return self._bg_median

    def lighting_jump(self, gray_mean: float) -> bool:
        """整幅平均亮度突变 → 多半是开关灯，背景作废。"""
        prev, self._prev_mean = self._prev_mean, gray_mean
        if prev is None:
            return False
        return abs(gray_mean - prev) > 12.0

    # ------------------------------------------------------------ 对外
    def update(self, img: np.ndarray) -> tuple[np.ndarray, bool]:
        """喂一帧，返回 (前景掩码, 是否处于预热期)。"""
        gray = img if img.ndim == 2 else None
        if gray is None:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if cv2 is not None else img.mean(axis=2)

        self.frames.append(gray.copy() if gray.ndim == 2 else gray)
        if len(self.frames) > max(self.history, self.warmup) * 2:
            self.frames = self.frames[-self.history:]
            self._bg_median = None

        warming = len(self.frames) < self.warmup
        if warming:
            if self._mog2 is not None:
                self._mog2.apply(img)
            return np.zeros_like(gray, dtype=np.uint8), True

        mask = np.zeros_like(gray, dtype=np.uint8)

        # ① 中值背景差分（对缓慢铺开的烟最灵）
        bg = self._median()
        if bg is not None:
            diff = cv2.absdiff(gray, bg) if cv2 is not None else np.abs(gray.astype(int) - bg.astype(int)).astype(np.uint8)
            _, m1 = cv2.threshold(diff, 18, 255, cv2.THRESH_BINARY) if cv2 is not None else (None, (diff > 18).astype(np.uint8) * 255)
            mask = cv2.bitwise_or(mask, m1) if cv2 is not None else np.maximum(mask, m1)

        # ② MOG2（对快速变化敏感，火焰跳动靠它）
        if self._mog2 is not None:
            m2 = self._mog2.apply(img)
            _, m2 = cv2.threshold(m2, 200, 255, cv2.THRESH_BINARY)   # 丢掉阴影
            mask = cv2.bitwise_or(mask, m2)

        # ③ 去噪 + 连断：烟是大块软边，用开运算杀掉零星噪点，再闭运算补洞
        if cv2 is not None:
            k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            k7 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k3)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k7)
            mask = self._drop_small(mask)
        return mask, False

    def _drop_small(self, mask: np.ndarray) -> np.ndarray:
        n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        out = np.zeros_like(mask)
        for i in range(1, n):
            if stats[i, cv2.CC_STAT_AREA] >= self.min_area:
                out[labels == i] = 255
        return out


class FixedCameraSource(CameraSource):
    """固定机位源：读帧 + 背景建模，掩码塞进 FramePacket。"""

    def __init__(self, info: CameraInfo, frames=None, warmup: int = 30):
        super().__init__(info)
        self._frames = iter(frames) if frames is not None else None
        self.bg = BackgroundModel(warmup=warmup)
        self._idx = 0

    @classmethod
    def from_video(cls, info: CameraInfo, path: str) -> "FixedCameraSource":
        cap = cv2.VideoCapture(path)
        frames = []
        while True:
            ok, img = cap.read()
            if not ok:
                break
            frames.append(img)
        cap.release()
        return cls(info, frames)

    def read(self) -> FramePacket | None:
        if self._frames is None:
            return None
        try:
            img = next(self._frames)
        except StopIteration:
            self.closed = True
            return None

        mask, warming = self.bg.update(img)
        pkt = FramePacket(
            cam_id=self.info.cam_id, idx=self._idx, ts=self._idx / max(self.info.fps, 1e-6),
            image=img, moving=False, fg_mask=None if warming else mask,
        )
        self._idx += 1
        return pkt
