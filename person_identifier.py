"""
person_identifier.py - 人物重识别与智能去重
基于衣着颜色直方图 + 外貌特征，结合时间窗口去重
"""
import cv2
import numpy as np
import time
from collections import defaultdict


class PersonIdentifier:
    """人物特征提取与重识别"""

    def __init__(self, time_window=30, similarity_threshold=0.75):
        self._time_window = time_window
        self._sim_threshold = similarity_threshold
        self._recent_persons = defaultdict(list)  # camera_id -> [(person_id, features, timestamp)]

    def extract_features(self, frame, bbox):
        """提取人物特征（衣着颜色直方图 + 位置特征）"""
        x1, y1, x2, y2 = bbox
        h, w = frame.shape[:2]
        x1 = max(0, x1); y1 = max(0, y1)
        x2 = min(w, x2); y2 = min(h, y2)

        if x2 - x1 < 10 or y2 - y1 < 10:
            return self._empty_features()

        roi = frame[y1:y2, x1:x2]

        # 分成上中下三部分（头/上衣/裤子）
        h_roi = roi.shape[0]
        parts = {
            "upper": roi[:h_roi // 3, :],
            "middle": roi[h_roi // 3: 2 * h_roi // 3, :],
            "lower": roi[2 * h_roi // 3:, :],
        }

        features = {}
        for part_name, part in parts.items():
            if part.size == 0:
                continue
            # HSV 颜色直方图
            hsv = cv2.cvtColor(part, cv2.COLOR_BGR2HSV)
            for channel, ch_name in enumerate(["h", "s", "v"]):
                hist = cv2.calcHist([hsv], [channel], None, [16], [0, 256])
                hist = cv2.normalize(hist, hist).flatten()
                features[f"{part_name}_{ch_name}"] = hist.tolist()

        # 人体比例特征
        aspect_ratio = (x2 - x1) / max(y2 - y1, 1)
        features["aspect_ratio"] = aspect_ratio
        features["area"] = (x2 - x1) * (y2 - y1)
        features["center_x"] = (x1 + x2) / 2 / w
        features["center_y"] = (y1 + y2) / 2 / h

        return features

    def _empty_features(self):
        return {"aspect_ratio": 0, "area": 0, "center_x": 0, "center_y": 0}

    def find_similar_person(self, camera_id, features, current_time, db=None):
        """查找近期相似人物，返回 (matched_person_id, best_confidence) 或 (None, 0)"""
        now = current_time

        # 清理过期记录
        self._cleanup(now)

        # 先检查内存缓存
        for person_id, stored_features, ts in self._recent_persons.get(camera_id, []):
            if now - ts <= self._time_window:
                sim = self._cosine_similarity(features, stored_features)
                if sim > self._sim_threshold:
                    return person_id, sim

        # 再检查数据库
        if db:
            eid, conf = db.find_recent_similar_person(camera_id, features, self._time_window)
            if eid is not None:
                return f"db_{eid}", conf

        return None, 0

    def register_person(self, camera_id, person_id, features):
        """注册新人物"""
        now = time.time()
        self._recent_persons[camera_id].append((person_id, features, now))
        self._cleanup(now)

    def _cleanup(self, now):
        for cam_id in list(self._recent_persons.keys()):
            self._recent_persons[cam_id] = [
                (pid, feat, ts) for pid, feat, ts in self._recent_persons[cam_id]
                if now - ts <= self._time_window * 2
            ]

    @staticmethod
    def _cosine_similarity(f1, f2):
        """计算特征余弦相似度"""
        keys = set(f1.keys()) & set(f2.keys())
        if not keys:
            return 0.0

        vec1, vec2 = [], []
        for k in keys:
            v1 = f1[k]
            v2 = f2[k]
            if isinstance(v1, list):
                vec1.extend(v1)
                vec2.extend(v2)
            else:
                vec1.append(v1)
                vec2.append(v2)

        a = np.array(vec1, dtype=np.float32)
        b = np.array(vec2, dtype=np.float32)
        na = np.linalg.norm(a)
        nb = np.linalg.norm(b)
        if na == 0 or nb == 0:
            return 0.0
        return float(np.dot(a, b) / (na * nb))