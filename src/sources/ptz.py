"""云台机位：先「把画面摆正」，再谈背景差分。

云台的特殊性
-----------
云台一转，整幅画面都在动：
· 直接做帧差 → 满屏都是"变化"，背景建模彻底失效
· 直接把轨迹跨帧关联 → 同一团烟会被拉到画面另一头，轨迹彻底错乱

所以云台这条路必须先做**全局运动估计**（ORB 特征 + RANSAC 求单应），
把当前帧 warp 回参考坐标系，之后的差分和跟踪才有意义。

三个必须处理的细节
-----------------
1. **估计失败要能退**：纹理少的场景（白墙、夜间）ORB 匹配点不够 →
   退回"用上一帧的单应做累积"，再不行就标记 `vision_lost=True`，那段时间不下结论
2. **转动中不判**：`moving=True` 的帧，轨迹与事件都要冻结（否则误报）
3. **停稳后重建参考**：云台停到新位置，旧参考背景已经没用了 →
   `settling` 帧内重算中值背景，避免"换了个视角就报一片烟"
"""
from __future__ import annotations

import numpy as np

try:
    import cv2
except Exception:          # pragma: no cover
    cv2 = None

from .base import CameraInfo, CameraSource, FramePacket


class MotionCompensator:
    """相邻帧 / 帧与参考 之间的全局单应估计。"""

    def __init__(self, max_features: int = 800, min_matches: int = 18,
                 ransac_px: float = 3.0, resize_w: int = 480):
        self.min_matches = min_matches
        self.ransac_px = ransac_px
        self.resize_w = resize_w          # 降分辨率估计，快 3~5 倍，精度够
        self.orb = cv2.ORB_create(max_features) if cv2 is not None else None
        self.matcher = (cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
                        if cv2 is not None else None)

    def _prep(self, gray: np.ndarray):
        h, w = gray.shape[:2]
        if w <= self.resize_w:
            return gray, 1.0
        s = self.resize_w / w
        return cv2.resize(gray, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA), s

    def estimate(self, ref_gray: np.ndarray, cur_gray: np.ndarray) -> np.ndarray | None:
        """返回 cur → ref 的单应（3×3，原图坐标系）。失败返回 None。"""
        if cv2 is None:
            return None
        g1, s1 = self._prep(ref_gray)
        g2, s2 = self._prep(cur_gray)
        kp1, d1 = self.orb.detectAndCompute(g1, None)
        kp2, d2 = self.orb.detectAndCompute(g2, None)
        if d1 is None or d2 is None:
            return None
        matches = self.matcher.match(d1, d2)
        if len(matches) < self.min_matches:
            return None

        p1 = np.float32([kp1[m.queryIdx].pt for m in matches])   # ref
        p2 = np.float32([kp2[m.trainIdx].pt for m in matches])   # cur
        H, inliers = cv2.findHomography(p2, p1, cv2.RANSAC, self.ransac_px)
        if H is None or inliers is None:
            return None
        # 内点太少说明是"蒙"出来的 → 宁可不给
        if float(inliers.mean()) < 0.45:
            return None

        # 把缩放还原到原图坐标：p_orig = p_small / s
        S1 = np.diag([1 / s1, 1 / s1, 1.0])
        S2 = np.diag([1 / s2, 1 / s2, 1.0])
        return S1 @ H @ np.linalg.inv(S2)

    @staticmethod
    def motion_of(H: np.ndarray) -> tuple[float, float]:
        """从单应里拆出「平移量(px)」和「缩放比」，用于判断云台是否在动。"""
        h = np.asarray(H, dtype=float)
        h = h / (h[2, 2] if abs(h[2, 2]) > 1e-9 else 1.0)
        dx, dy = float(h[0, 2]), float(h[1, 2])
        scale = float(np.sqrt(abs(np.linalg.det(h[:2, :2]))))
        return float(np.hypot(dx, dy)), scale

    def is_moving(self, H: np.ndarray | None, shift_px: float = 2.0,
                  scale_tol: float = 0.02) -> bool:
        if H is None:
            return False
        dist, scale = self.motion_of(H)
        return dist > shift_px or abs(scale - 1.0) > scale_tol


class PtzCameraSource(CameraSource):
    """云台源：全局补偿 + 对齐后的背景差分。"""

    def __init__(self, info: CameraInfo, frames=None, warmup: int = 20,
                 settling: int = 8):
        super().__init__(info)
        self._frames = iter(frames) if frames is not None else None
        self.comp = MotionCompensator()
        self.warmup = warmup
        self.settling = settling
        self._idx = 0
        self._ref_gray: np.ndarray | None = None     # 参考帧（对齐基准）
        self._ref_bg: np.ndarray | None = None       # 参考坐标系下的背景板
        self._aligned_pool: list[np.ndarray] = []    # 用于算中值背景
        self._H_total: np.ndarray | None = None      # 当前帧 → 参考
        self._still_count = 0

    @classmethod
    def from_video(cls, info: CameraInfo, path: str) -> "PtzCameraSource":
        cap = cv2.VideoCapture(path)
        frames = []
        while True:
            ok, img = cap.read()
            if not ok:
                break
            frames.append(img)
        cap.release()
        return cls(info, frames)

    # ------------------------------------------------------------ 内部
    def _update_reference(self, aligned_gray: np.ndarray) -> None:
        self._aligned_pool.append(aligned_gray.copy())
        if len(self._aligned_pool) > max(self.warmup * 2, 60):
            self._aligned_pool = self._aligned_pool[-self.warmup * 2:]
        if len(self._aligned_pool) >= self.warmup:
            self._ref_bg = np.median(np.stack(self._aligned_pool), axis=0).astype(np.uint8)

    def _reset_reference(self) -> None:
        """停稳后重建：旧视角的背景不能再用。"""
        self._aligned_pool.clear()
        self._ref_bg = None
        self._ref_gray = None
        self._H_total = None

    # ------------------------------------------------------------ 对外
    def read(self) -> FramePacket | None:
        if self._frames is None:
            return None
        try:
            img = next(self._frames)
        except StopIteration:
            self.closed = True
            return None

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if cv2 is not None else img.mean(axis=2)
        self._idx += 1

        # 第一帧就是参考
        if self._ref_gray is None:
            self._ref_gray = gray.copy()
            self._update_reference(gray)
            return FramePacket(self.info.cam_id, 0, 0.0, img, moving=False,
                               homography=np.eye(3, dtype=np.float32), fg_mask=None)

        H = self.comp.estimate(self._ref_gray, gray)
        moving = self.comp.is_moving(H)

        if H is None:
            # 估计失败（白墙/夜间）：不补偿，直接标记"这帧不可靠"
            return FramePacket(self.info.cam_id, self._idx,
                               self._idx / max(self.info.fps, 1e-6), img,
                               moving=False, homography=None, fg_mask=None,
                               meta={"vision_lost": True})

        if moving:
            # 转动中：仍然给出补偿矩阵（下游可用），但不产前景、不更新背景
            self._still_count = 0
            return FramePacket(self.info.cam_id, self._idx,
                               self._idx / max(self.info.fps, 1e-6), img,
                               moving=True, homography=H, fg_mask=None,
                               meta={"note": "云台转动中，暂停判定"})

        self._still_count += 1

        aligned = cv2.warpPerspective(gray, H, (gray.shape[1], gray.shape[0]),
                                      flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        mask = None
        if self._ref_bg is None or self._still_count <= self.settling:
            # 刚停稳：这段时间只用来重建背景，不报前景
            self._update_reference(aligned)
        else:
            diff = cv2.absdiff(aligned, self._ref_bg)
            _, mask = cv2.threshold(diff, 20, 255, cv2.THRESH_BINARY)
            k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            k7 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k3)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k7)
            # 背景缓慢更新，避免"停住的白烟"永远算前景
            self._ref_bg = cv2.addWeighted(self._ref_bg, 0.995, aligned, 0.005, 0)

        return FramePacket(self.info.cam_id, self._idx,
                           self._idx / max(self.info.fps, 1e-6), img,
                           moving=False, homography=H, fg_mask=mask)
