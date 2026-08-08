"""
person_tracker.py - 人体毫秒级追踪器 v5
Kalman滤波 + 匈牙利匹配 + 平滑框过渡
v5: 提高追踪稳定性(max_missed=12) + 快速响应EMA + 高过程噪声
"""
import numpy as np
import cv2
from collections import OrderedDict
from typing import List, Tuple


class KalmanBoxTracker:
    """Kalman滤波追踪器 — 每个目标独立状态估计"""

    count = 0

    def __init__(self, bbox: List[int], frame_idx: int):
        self.kf = cv2.KalmanFilter(7, 4)
        self.kf.transitionMatrix = np.eye(7, dtype=np.float32)
        self.kf.transitionMatrix[0, 4] = 1.0
        self.kf.transitionMatrix[1, 5] = 1.0
        self.kf.transitionMatrix[2, 6] = 1.0
        self.kf.measurementMatrix = np.eye(4, 7, dtype=np.float32)
        self.kf.processNoiseCov = np.eye(7, dtype=np.float32) * 0.03
        self.kf.processNoiseCov[4, 4] = 0.35  # x方向速度噪声（更高=追踪快速移动不丢）
        self.kf.processNoiseCov[5, 5] = 0.35  # y方向速度噪声
        self.kf.processNoiseCov[6, 6] = 0.08  # 尺度变化噪声
        self.kf.measurementNoiseCov = np.eye(4, dtype=np.float32) * 0.10  # 极低=高度信任检测值

        x1, y1, x2, y2 = bbox
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        s = (x2 - x1) * (y2 - y1)
        r = (x2 - x1) / max(y2 - y1, 1)
        self.kf.statePre = np.array([[cx], [cy], [s], [r], [0], [0], [0]], dtype=np.float32)

        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.frame_idx = frame_idx
        self.last_seen = frame_idx
        self.missed = 0
        self.hits = 1
        self.bbox = bbox
        self.smoothed_bbox = bbox
        self.head_bbox = None      # 头部区域
        self.hand_roi = None       # 右手区域
        self.hand_left_roi = None  # 左手区域
        self.confidence = 0.0
        self._update_rois()

    def predict(self):
        """Kalman预测下一帧位置 — 带验证钳位"""
        pred = self.kf.predict()
        cx, cy, s, r = pred[0, 0], pred[1, 0], pred[2, 0], pred[3, 0]
        s = max(s, 1000)
        r = max(min(r, 5.0), 0.2)
        w = np.sqrt(s * r)
        h = s / max(w, 1)
        x1 = int(cx - w / 2)
        y1 = int(cy - h / 2)
        x2 = int(cx + w / 2)
        y2 = int(cy + h / 2)
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = max(x1 + 10, x2)
        y2 = max(y1 + 10, y2)
        self.bbox = [x1, y1, x2, y2]
        if self.smoothed_bbox:
            alpha = 0.55  # 激进响应（预测帧）
            self.smoothed_bbox = [
                int(alpha * self.bbox[i] + (1 - alpha) * self.smoothed_bbox[i])
                for i in range(4)
            ]
        else:
            self.smoothed_bbox = list(self.bbox)
        self._update_rois()

    def update(self, bbox: List[int], frame_idx: int):
        """用检测结果更新 Kalman"""
        x1, y1, x2, y2 = bbox
        if x2 <= x1 + 5 or y2 <= y1 + 5:
            return
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        s = (x2 - x1) * (y2 - y1)
        r = (x2 - x1) / max(y2 - y1, 1)
        self.kf.correct(np.array([[cx], [cy], [s], [r]], dtype=np.float32))
        self.bbox = [x1, y1, x2, y2]
        self.last_seen = frame_idx
        self.missed = 0
        self.hits += 1
        if self.smoothed_bbox:
            alpha = 0.60  # 激进响应（更新帧，检测值权重更高）
            self.smoothed_bbox = [
                int(alpha * bbox[i] + (1 - alpha) * self.smoothed_bbox[i])
                for i in range(4)
            ]
        else:
            self.smoothed_bbox = list(bbox)
        self._update_rois()

    def mark_missed(self):
        self.missed += 1

    def _update_rois(self):
        """精准 ROI 估算 — 头部+双手，严格贴合人体比例"""
        bx = self.smoothed_bbox if self.smoothed_bbox else self.bbox
        x1, y1, x2, y2 = bx
        pw = max(x2 - x1, 1)
        ph = max(y2 - y1, 1)

        # 收缩5%贴合人体边缘
        margin_x = int(pw * 0.05)
        margin_y = int(ph * 0.05)
        x1 += margin_x
        x2 -= margin_x
        y1 += margin_y
        y2 -= margin_y
        pw = max(x2 - x1, 1)
        ph = max(y2 - y1, 1)

        # 头部 ROI：人体上部 0%-18%，宽度 25%-60%（覆盖全角度：正面+侧面+背面）
        self.head_bbox = [
            x1 + int(pw * 0.25), y1 + int(ph * 0.00),
            x1 + int(pw * 0.60), y1 + int(ph * 0.18)
        ]
        # 右手 ROI：身体右侧 55%-95% 宽，下部 30%-55%（不超出身体边界）
        self.hand_roi = [
            x1 + int(pw * 0.55), y1 + int(ph * 0.30),
            x2,                 y1 + int(ph * 0.55)
        ]
        # 左手 ROI：身体左侧 5%-45% 宽，下部 30%-55%（不超出身体边界）
        self.hand_left_roi = [
            x1,                 y1 + int(ph * 0.30),
            x1 + int(pw * 0.45), y1 + int(ph * 0.55)
        ]


class PersonTracker:
    """人体追踪管理器"""

    def __init__(self, max_missed: int = 12, iou_thresh: float = 0.2):
        self._max_missed = max_missed
        self._iou_thresh = iou_thresh
        self._tracks: OrderedDict = OrderedDict()
        self._frame_idx = 0

    def update(self, detections: List[List]) -> List[KalmanBoxTracker]:
        self._frame_idx += 1

        # 保存预测前的bbox（用于 IoU 匹配，避免预测漂移）
        pre_predict_boxes = {tid: (t.smoothed_bbox if t.smoothed_bbox else t.bbox)
                             for tid, t in self._tracks.items()}

        for t in self._tracks.values():
            t.predict()
            t.mark_missed()

        if not detections:
            self._remove_stale()
            return self.get_active_tracks()

        det_boxes = np.array([d[:4] for d in detections])
        track_ids = list(self._tracks.keys())
        # 使用预测前的框做匹配（避免 Kalman 预测漂移导致 ID 切换）
        track_boxes = np.array([pre_predict_boxes[tid] for tid in track_ids])

        if len(track_boxes) > 0:
            iou_matrix = self._iou_matrix(track_boxes, det_boxes)
            matched, unmatched_tracks, unmatched_dets = self._associate(iou_matrix)
        else:
            matched, unmatched_tracks, unmatched_dets = [], [], list(range(len(detections)))

        for ti, di in matched:
            tid = track_ids[ti]
            det = detections[di]
            self._tracks[tid].update(det[:4], self._frame_idx)
            if len(det) > 4:
                self._tracks[tid].confidence = det[4]

        for di in unmatched_dets:
            det = detections[di]
            # 去重：检查是否与现有 track 高度重叠
            det_box = np.array([det[:4]])
            existing_boxes = np.array([t.smoothed_bbox if t.smoothed_bbox else t.bbox for t in self._tracks.values()])
            if len(existing_boxes) > 0:
                overlaps = self._iou_matrix(existing_boxes, det_box)
                if np.max(overlaps) > 0.5:  # IoU > 0.5 视为重复，跳过
                    continue
            tracker = KalmanBoxTracker(det[:4], self._frame_idx)
            tracker.confidence = det[4] if len(det) > 4 else 0.0
            self._tracks[tracker.id] = tracker

        self._remove_stale()
        return self.get_active_tracks()

    def get_active_tracks(self) -> List[KalmanBoxTracker]:
        # 允许missed<=5的track仍显示（~0.5秒内Kalman预测位置）
        return [t for t in self._tracks.values() if t.missed <= 5]

    def _remove_stale(self):
        to_remove = [tid for tid, t in self._tracks.items() if t.missed > self._max_missed]
        for tid in to_remove:
            del self._tracks[tid]

    def _associate(self, iou_matrix: np.ndarray) -> Tuple:
        matched = []
        unmatched_tracks = set(range(iou_matrix.shape[0]))
        unmatched_dets = set(range(iou_matrix.shape[1]))
        if iou_matrix.size == 0:
            return matched, unmatched_tracks, unmatched_dets
        flat_indices = np.argsort(iou_matrix.ravel())[::-1]
        for idx in flat_indices:
            ti = idx // iou_matrix.shape[1]
            di = idx % iou_matrix.shape[1]
            if iou_matrix[ti, di] >= self._iou_thresh:
                if ti in unmatched_tracks and di in unmatched_dets:
                    matched.append((ti, di))
                    unmatched_tracks.remove(ti)
                    unmatched_dets.remove(di)
        return matched, unmatched_tracks, unmatched_dets

    @staticmethod
    def _iou_matrix(track_boxes: np.ndarray, det_boxes: np.ndarray) -> np.ndarray:
        n_tracks, n_dets = len(track_boxes), len(det_boxes)
        iou = np.zeros((n_tracks, n_dets), dtype=np.float32)
        for i in range(n_tracks):
            tx1, ty1, tx2, ty2 = track_boxes[i]
            ta = max((tx2 - tx1) * (ty2 - ty1), 1)
            for j in range(n_dets):
                dx1, dy1, dx2, dy2 = det_boxes[j]
                x1 = max(tx1, dx1)
                y1 = max(ty1, dy1)
                x2 = min(tx2, dx2)
                y2 = min(ty2, dy2)
                inter = max(0, x2 - x1) * max(0, y2 - y1)
                da = max((dx2 - dx1) * (dy2 - dy1), 1)
                iou[i, j] = inter / (ta + da - inter + 1e-6)
        return iou