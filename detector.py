"""
detector.py - 极速多级检测管线 v6
S1: 人体检测(YOLOv8n) → 追踪
S2: 姿态估计(YOLOv8n-pose) → 头部定位 + 双手定位（手腕关键点）
S3: 香烟检测(smoking.onnx) → 头部和手部区域
     数据增强(亮度/对比度微扰) 防过拟合
S4: 告警生成 → 仅输出香烟/烟盒检测结果
"""
import onnxruntime as ort
import numpy as np
import cv2
import time
import os
import random
from typing import List, Dict, Optional, Tuple
from person_tracker import PersonTracker


# COCO 17关键点索引
COCO_KPTS = {
    "nose": 0, "left_eye": 1, "right_eye": 2,
    "left_ear": 3, "right_ear": 4,
    "left_shoulder": 5, "right_shoulder": 6,
    "left_elbow": 7, "right_elbow": 8,
    "left_wrist": 9, "right_wrist": 10,
    "left_hip": 11, "right_hip": 12,
    "left_knee": 13, "right_knee": 14,
    "left_ankle": 15, "right_ankle": 16,
}
HEAD_KPT_IDS = [0, 1, 2, 3, 4]  # nose, eyes, ears
HAND_KPT_IDS = [9, 10]            # left_wrist, right_wrist


class MultiStageDetector:
    """极速检测管线 — 仅关注头部+手部+香烟"""

    PERSON_CLASS = 0
    SMOKING_CLASSES = {0: "cigarette", 1: "cigarette_pack"}

    def __init__(self, person_model_path: str, smoking_model_path: str,
                 pose_model_path: str = None,
                 conf_threshold: float = 0.50, iou_threshold: float = 0.50):
        self._conf = conf_threshold
        self._iou_thresh = iou_threshold
        self._pose_model_path = pose_model_path

        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        opts.intra_op_num_threads = 2
        opts.inter_op_num_threads = 1
        opts.enable_mem_pattern = True
        opts.enable_cpu_mem_arena = True

        self._person_session = ort.InferenceSession(
            person_model_path, opts, providers=["CPUExecutionProvider"])
        self._person_input_name = self._person_session.get_inputs()[0].name
        self._person_img_size = self._person_session.get_inputs()[0].shape[2]

        self._smoke_session = ort.InferenceSession(
            smoking_model_path, opts, providers=["CPUExecutionProvider"])
        self._smoke_input_name = self._smoke_session.get_inputs()[0].name
        self._smoke_img_size = self._smoke_session.get_inputs()[0].shape[2]
        smoke_out_shape = self._smoke_session.get_outputs()[0].shape
        self._smoke_num_classes = smoke_out_shape[1] - 4 if len(smoke_out_shape) >= 2 else 1

        # 姿态估计模型（头部定位 + 双手定位）
        self._pose_session = None
        self._pose_input_name = None
        self._pose_img_size = 320
        if pose_model_path and os.path.exists(pose_model_path):
            self._pose_session = ort.InferenceSession(
                pose_model_path, opts, providers=["CPUExecutionProvider"])
            self._pose_input_name = self._pose_session.get_inputs()[0].name
            self._pose_img_size = self._pose_session.get_inputs()[0].shape[2]
            self._pose_out_dim = self._pose_session.get_outputs()[0].shape[1]  # 56 for pose

        self._tracker = PersonTracker(max_missed=12, iou_thresh=0.2)
        self._min_person_area = 8000  # 提高最小面积过滤假阳性

        # 帧跳过
        self._person_frame_counters = {}
        self._skip_interval = 3  # 每3帧做一次香烟检测
        self._person_detect_skip = 1  # 每帧做人体检测（YOLOv8n极快，确保追踪不丢）

        # 头部框EMA平滑缓存（极速响应，减少延迟感）
        self._head_bbox_cache: Dict[int, List[int]] = {}
        self._head_ema_alpha = 0.80  # EMA平滑系数（0.80=丝滑流畅）

        # 姿态估计缓存（隔帧运行，减轻CPU负载）
        self._pose_cache = {}  # track_id -> kpts
        self._pose_skip = 2  # 每2帧运行pose（头部+手部唯一来源，需较高频率）

        # 性能统计
        self._inference_times: List[float] = []
        self._fps = 0.0
        self._frame_count = 0
        self._global_frame = 0

        # 心跳监控（用于检测AI是否卡死）
        self._last_detect_time = time.perf_counter()
        self._tracker_reset_interval = 300  # 每300帧重置追踪器，防止Kalman发散

        # 数据增强参数（防过拟合）
        self._aug_brightness_range = (-0.05, 0.05)  # 亮度微扰 ±5%
        self._aug_contrast_range = (0.95, 1.05)      # 对比度微扰 ±5%

    @property
    def heartbeat_age_sec(self) -> float:
        """距离上次检测的秒数（用于心跳监控）"""
        return time.perf_counter() - self._last_detect_time

    @property
    def avg_inference_time_ms(self) -> float:
        if not self._inference_times:
            return 0.0
        return sum(self._inference_times) / len(self._inference_times)

    @property
    def current_fps(self) -> float:
        return self._fps

    @property
    def backend_name(self) -> str:
        return "CPU(ONNX优化)"

    def detect(self, frame: np.ndarray) -> Dict:
        """极速检测：人体→姿态估计(头部+手部)→香烟"""
        t0 = time.perf_counter()
        self._last_detect_time = t0  # 更新心跳
        h, w = frame.shape[:2]
        self._global_frame += 1

        # 定期重置追踪器，防止Kalman滤波器长期发散导致追踪丢失
        if self._global_frame % self._tracker_reset_interval == 0:
            self._tracker = PersonTracker(max_missed=12, iou_thresh=0.2)
            self._person_frame_counters.clear()
            self._head_bbox_cache.clear()
            self._pose_cache.clear()

        # S1: 头部检测（V6训练: 直接找头, 无需pose→head估算）
        head_dets = []
        if self._global_frame % self._person_detect_skip == 0:
            head_dets = self._detect_persons_fast(frame)  # 现在输出的是头框
        tracked = self._tracker.update(head_dets)

        # S2: 姿态估计 → 仅用于手部定位（头部不再依赖pose）
        pose_kpts = None
        if self._pose_session is not None and tracked:
            if self._global_frame % self._pose_skip == 0:
                pose_kpts = self._detect_pose(frame)
                pose_kpts = self._assign_pose_to_tracks(tracked, pose_kpts, w, h)
                self._pose_cache = pose_kpts if pose_kpts else {}
            else:
                pose_kpts = self._pose_cache if self._pose_cache else None

        # S3: 头部+手部 香烟检测
        all_alerts = []
        all_heads = []
        all_hands = []
        alert_hand_ids = set()

        for track in tracked:
            tid = track.id
            if tid not in self._person_frame_counters:
                self._person_frame_counters[tid] = 0
            self._person_frame_counters[tid] += 1

            # 过滤1：降低门槛，立即显示（但超10帧未更新才丢弃）
            if track.missed > 10:
                continue

            # 过滤2：pose关键点验证 — 仅极端假阳性才丢弃
            if pose_kpts and tid in pose_kpts:
                kpts = pose_kpts[tid]
                valid_head = sum(1 for kpt_id in HEAD_KPT_IDS if kpts[kpt_id, 2] > 0.15)
                # 放宽：只在追了很久(>30帧)且头关键点全无时才认为是假阳性
                if valid_head < 1 and track.hits > 30:
                    continue
            # 注意：移除了过严的 elif 检查，允许新track在没有pose匹配时也能显示

            do_smoke = self._person_frame_counters[tid] % self._skip_interval == 0

            # 获取头部框：使用姿态关键点估算
            head_bbox = None
            if pose_kpts and tid in pose_kpts:
                kpts = pose_kpts[tid]
                head_bbox = self._kpts_to_head_bbox(kpts, w, h)

            # 兜底策略：人体框估算头部（关键点不可用时）
            use_fallback = False
            if not head_bbox:
                use_fallback = True
                bx1, by1, bx2, by2 = track.bbox
                body_h = by2 - by1
                body_w = bx2 - bx1
                if body_h > 20 and body_w > 20:
                    fb_h = max(body_h * 0.22, 40)
                    fb_w = max(fb_h * 0.65, 30)
                    fb_cx = (bx1 + bx2) / 2
                    fb_y1 = by1 - fb_h * 0.20
                    fb_y2 = by1 + fb_h * 0.90
                    fb_x1 = fb_cx - fb_w / 2
                    fb_x2 = fb_cx + fb_w / 2
                    head_bbox = [int(fb_x1), int(max(0, fb_y1)),
                                 int(fb_x2), int(min(h - 1, fb_y2))]
                    if head_bbox[2] - head_bbox[0] < 20 or head_bbox[3] - head_bbox[1] < 20:
                        head_bbox = None

            if head_bbox:
                hx1, hy1, hx2, hy2 = self._clip_roi(head_bbox, w, h)
                if tid in self._head_bbox_cache:
                    prev = self._head_bbox_cache[tid]
                    alpha = self._head_ema_alpha
                    hx1 = int(alpha * hx1 + (1 - alpha) * prev[0])
                    hy1 = int(alpha * hy1 + (1 - alpha) * prev[1])
                    hx2 = int(alpha * hx2 + (1 - alpha) * prev[2])
                    hy2 = int(alpha * hy2 + (1 - alpha) * prev[3])
                self._head_bbox_cache[tid] = [hx1, hy1, hx2, hy2]
                source_tag = "body_fallback" if use_fallback else "keypoints"
                all_heads.append({"bbox": [hx1, hy1, hx2, hy2], "track_id": tid, "source": source_tag})
                if do_smoke and hy2 > hy1 and hx2 > hx1:
                    roi = self._augment_roi(frame[hy1:hy2, hx1:hx2])
                    # ★ 浅色背景局部对比度增强: 亮度>180才触发, 强化烟-背景边缘
                    roi = self.enhance_cigarette_contrast(roi)
                    smoke_dets = self._detect_smoking_fast(roi)
                    for sd in smoke_dets:
                        all_alerts.append({
                            "label": sd["label"], "confidence": sd["confidence"],
                            "bbox": [int(hx1 + sd["bbox"][0]), int(hy1 + sd["bbox"][1]),
                                     int(hx1 + sd["bbox"][2]), int(hy1 + sd["bbox"][3])],
                            "track_id": tid, "roi_type": "head",
                        })
            elif tid in self._head_bbox_cache:
                hx1, hy1, hx2, hy2 = self._head_bbox_cache[tid]
                all_heads.append({"bbox": [hx1, hy1, hx2, hy2], "track_id": tid, "source": "cache_hold"})

            # 获取手部框（优先用姿态关键点）
            hand_boxes = []
            if pose_kpts and tid in pose_kpts:
                kpts = pose_kpts[tid]
                hand_boxes = self._kpts_to_hand_boxes(kpts, w, h)
            if not hand_boxes:
                # 降级：比例估算
                if track.hand_roi:
                    hand_boxes.append(("R", track.hand_roi))
                if track.hand_left_roi:
                    hand_boxes.append(("L", track.hand_left_roi))

            for side, hb in hand_boxes:
                hx1, hy1, hx2, hy2 = self._clip_roi(hb, w, h)
                roi_type = f"hand_{side.lower()}"
                if do_smoke and hy2 > hy1 and hx2 > hx1:
                    roi = self._augment_roi(frame[hy1:hy2, hx1:hx2])
                    # ★ 浅色背景局部对比度增强: 亮度>180才触发, 强化烟-背景边缘
                    roi = self.enhance_cigarette_contrast(roi)
                    smoke_dets = self._detect_smoking_fast(roi)
                    for sd in smoke_dets:
                        all_alerts.append({
                            "label": sd["label"], "confidence": sd["confidence"],
                            "bbox": [int(hx1 + sd["bbox"][0]), int(hy1 + sd["bbox"][1]),
                                     int(hx1 + sd["bbox"][2]), int(hy1 + sd["bbox"][3])],
                            "track_id": tid, "roi_type": roi_type,
                        })
                    if smoke_dets:
                        alert_hand_ids.add(f"{tid}_{side}")
                        all_hands.append({"bbox": [hx1, hy1, hx2, hy2], "track_id": tid, "side": side})

        # 清理
        active_ids = {t.id for t in tracked}
        self._person_frame_counters = {tid: cnt for tid, cnt in self._person_frame_counters.items() if tid in active_ids}
        self._head_bbox_cache = {tid: bbox for tid, bbox in self._head_bbox_cache.items() if tid in active_ids}
        self._pose_cache = {tid: kpts for tid, kpts in self._pose_cache.items() if tid in active_ids}

        # 4.2 过滤越界零件框 + 4.3 NMS去重
        person_boxes = [t.smoothed_bbox if t.smoothed_bbox else t.bbox for t in tracked]
        if person_boxes:
            all_heads = self._filter_parts_by_person(all_heads, person_boxes, max_offset=1.5)
            all_hands = self._filter_parts_by_person(all_hands, person_boxes, max_offset=1.5)
        all_heads = self._nms_parts(all_heads, iou_thresh=0.5)
        all_hands = self._nms_parts(all_hands, iou_thresh=0.5)

        # 性能统计
        elapsed = (time.perf_counter() - t0) * 1000
        self._inference_times.append(elapsed)
        if len(self._inference_times) > 50:
            self._inference_times = self._inference_times[-50:]
        self._frame_count += 1
        if self._frame_count >= 10:
            self._fps = 1000 / self.avg_inference_time_ms
            self._frame_count = 0

        return {
            "persons": [{"id": t.id, "bbox": t.smoothed_bbox if t.smoothed_bbox else t.bbox}
                        for t in tracked],
            "heads": all_heads,
            "hands": all_hands,
            "alerts": all_alerts,
        }

    # ── 姿态估计方法 ──

    def _detect_pose(self, frame: np.ndarray) -> List[Dict]:
        """YOLOv8n-pose 姿态估计 → 提取每人17个关键点"""
        h, w = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 1/255.0, (self._pose_img_size, self._pose_img_size),
                                     swapRB=True, crop=False)
        outputs = self._pose_session.run(None, {self._pose_input_name: blob})
        preds = outputs[0]  # (1, 56, N)
        if len(preds.shape) == 3:
            preds = np.transpose(preds[0], (1, 0))  # (N, 56)

        # 过滤低置信度
        scores = preds[:, 4]
        mask = scores > self._conf
        if not np.any(mask):
            return []

        indices = np.where(mask)[0]
        order = np.argsort(scores[indices])[::-1]
        indices = indices[order]

        # NMS
        boxes_raw = preds[:, :4]
        keep = self._nms(boxes_raw[indices], scores[indices])
        indices = indices[keep]

        scale_x = w / self._pose_img_size
        scale_y = h / self._pose_img_size

        persons = []
        for idx in indices:
            # 关键点: preds[idx, 5:] = 51个值 (17*3)
            kpts = preds[idx, 5:].reshape(17, 3)
            # 坐标缩放
            kpts[:, 0] *= scale_x
            kpts[:, 1] *= scale_y
            persons.append({
                "bbox": [
                    int((preds[idx, 0] - preds[idx, 2] / 2) * scale_x),
                    int((preds[idx, 1] - preds[idx, 3] / 2) * scale_y),
                    int((preds[idx, 0] + preds[idx, 2] / 2) * scale_x),
                    int((preds[idx, 1] + preds[idx, 3] / 2) * scale_y),
                ],
                "kpts": kpts,
                "score": float(scores[idx]),
            })
        return persons

    def _assign_pose_to_tracks(self, tracked: list, pose_persons: list,
                                img_w: int, img_h: int) -> Dict[int, np.ndarray]:
        """将姿态关键点按IoU匹配到追踪目标"""
        if not pose_persons:
            return {}
        track_boxes = np.array([t.smoothed_bbox if t.smoothed_bbox else t.bbox for t in tracked])
        pose_boxes = np.array([p["bbox"] for p in pose_persons])
        iou = self._iou_matrix(track_boxes, pose_boxes)
        assigned = {}
        used = set()
        for ti in range(len(tracked)):
            best_j = -1
            best_iou = 0.3  # 最小IoU阈值
            for pj in range(len(pose_persons)):
                if pj in used:
                    continue
                if iou[ti, pj] > best_iou:
                    best_iou = iou[ti, pj]
                    best_j = pj
            if best_j >= 0:
                assigned[tracked[ti].id] = pose_persons[best_j]["kpts"]
                used.add(best_j)
        return assigned

    @staticmethod
    def _kpts_to_head_bbox(kpts: np.ndarray, img_w: int, img_h: int) -> Optional[List[int]]:
        """从关键点提取头部边界框 v5 — 多策略融合估算，精确覆盖完整头部
        
        策略优先级：鼻子可见 > 眼睛可见 > 耳朵估算 > 肩宽推算
        头部比例：宽=耳朵到耳朵外轮廓, 高=头顶到下巴 (高宽比≈1.55)
        """
        img_area = img_w * img_h
        
        # 收集各关键点置信度
        nose_c = kpts[0, 2]
        left_eye_c = kpts[1, 2] if len(kpts) > 1 else 0
        right_eye_c = kpts[2, 2] if len(kpts) > 2 else 0
        left_ear_c = kpts[3, 2] if len(kpts) > 3 else 0
        right_ear_c = kpts[4, 2] if len(kpts) > 4 else 0
        
        # 有效头部关键点（用于位置和尺寸计算）
        valid = []
        weights = []
        for i in HEAD_KPT_IDS:
            x, y, c = kpts[i]
            if c > 0.15:
                valid.append([x, y])
                weights.append(c)
        valid = np.array(valid) if valid else np.empty((0, 2))
        weights = np.array(weights) if weights else np.empty(0)
        
        if len(valid) < 2:
            return None
        
        # 计算关键点跨度
        kw_raw = valid[:, 0].max() - valid[:, 0].min()
        kh_raw = valid[:, 1].max() - valid[:, 1].min()
        
        # 预估算头部尺寸（用于后续中心位置计算）
        if left_ear_c > 0.15 and right_ear_c > 0.15:
            # 双耳可见：耳朵间距 ≈ 头宽的65%（仅耳朵露在脸两侧外）
            ear_dist = abs(kpts[3, 0] - kpts[4, 0])
            est_head_w = ear_dist / 0.65
        elif left_eye_c > 0.15 and right_eye_c > 0.15:
            # 双眼可见：眼距 ≈ 头宽的45%
            eye_dist = abs(kpts[1, 0] - kpts[2, 0])
            est_head_w = eye_dist / 0.45
        elif (left_eye_c > 0.15 or right_eye_c > 0.15):
            # 单眼可见：用单眼位置和典型单侧眼距推算
            visible_x = kpts[1, 0] if left_eye_c > 0.15 else kpts[2, 0]
            # 假设鼻子在视野中附近，估侧脸宽度
            est_head_w = max(abs(visible_x - kpts[0, 0]) * 3.5, 50)
        else:
            # 回退：用关键点跨度估算
            est_head_w = max(kw_raw * 1.4, 60)

        est_head_w = max(est_head_w, 50)
        est_head_h = est_head_w * 1.25  # 精确头部高宽比
        
        # === 策略1: 鼻子可见 ===
        if nose_c > 0.20:
            nx, ny = kpts[0, 0], kpts[0, 1]
            # 鼻子在头部约40-50%高度处（从头顶往下）
            # 头部中心在鼻子下方约8%头高处
            head_cx = nx
            head_cy = ny + est_head_h * 0.08
            head_w = est_head_w
            head_h = est_head_h
        
        # === 策略2: 眼睛可见（鼻子不可见时） ===
        elif (left_eye_c > 0.15 or right_eye_c > 0.15):
            # 用可见眼睛均值作为位置参考
            eye_pts = []
            if left_eye_c > 0.15:
                eye_pts.append(kpts[1, :2])
            if right_eye_c > 0.15:
                eye_pts.append(kpts[2, :2])
            eye_pts = np.array(eye_pts)
            eye_mean = eye_pts.mean(axis=0)
            # 眼睛在头部约25-30%高度处
            head_cx = eye_mean[0]
            head_cy = eye_mean[1] + est_head_h * 0.22
            head_w = est_head_w
            head_h = est_head_h
        
        # === 策略3: 侧脸/背对（耳朵可见） ===
        elif len(valid) >= 2:
            cx = valid[:, 0].mean()
            cy = valid[:, 1].mean()
            head_cx = cx
            head_cy = cy + est_head_h * 0.15
            if len(valid) == 2 and kw_raw > 5:
                head_w = kw_raw / 0.50
            else:
                head_w = max(kw_raw * 1.6, 60)
            head_h = head_w * 1.55
        
        else:
            return None
        
        # 肩宽验证（防止异常放大，但不下限收紧）
        left_shoulder_c = kpts[5, 2] if len(kpts) > 5 else 0
        right_shoulder_c = kpts[6, 2] if len(kpts) > 6 else 0
        if left_shoulder_c > 0.15 and right_shoulder_c > 0.15:
            shoulder_dist = abs(kpts[5, 0] - kpts[6, 0])
            # 头宽应小于肩宽（典型头:肩 = 1:2.5），仅上限保护
            max_head_w = shoulder_dist * 0.55
            if head_w > max_head_w:
                head_w = max_head_w
                head_h = head_w * 1.30
        
        # 确保合理范围
        head_w = np.clip(head_w, 40, img_w * 0.6)
        head_h = np.clip(head_h, 55, img_h * 0.8)
        
        x1 = int(head_cx - head_w / 2)
        y1 = int(head_cy - head_h / 2)
        x2 = int(head_cx + head_w / 2)
        y2 = int(head_cy + head_h / 2)
        
        # 边界裁剪
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(img_w, x2)
        y2 = min(img_h, y2)
        
        if x2 <= x1 + 15 or y2 <= y1 + 15:
            return None
        return [x1, y1, x2, y2]

    @staticmethod
    def _kpts_to_hand_boxes(kpts: np.ndarray, img_w: int, img_h: int) -> List[Tuple[str, List[int]]]:
        """从关键点提取手部边界框"""
        boxes = []
        for kpt_id, side in [(9, "R"), (10, "L")]:  # wrist keypoints
            x, y, c = kpts[kpt_id]
            if c > 0.25:
                size = max(30, int(img_h * 0.06))
                x1 = max(0, int(x - size))
                y1 = max(0, int(y - size * 0.8))
                x2 = min(img_w, int(x + size))
                y2 = min(img_h, int(y + size * 1.2))
                if x2 > x1 + 10 and y2 > y1 + 10:
                    boxes.append((side, [x1, y1, x2, y2]))
        return boxes

    @staticmethod
    def _iou_matrix(track_boxes: np.ndarray, det_boxes: np.ndarray) -> np.ndarray:
        """计算IoU矩阵"""
        n_t, n_d = len(track_boxes), len(det_boxes)
        iou = np.zeros((n_t, n_d), dtype=np.float32)
        for i in range(n_t):
            tx1, ty1, tx2, ty2 = track_boxes[i]
            ta = max((tx2 - tx1) * (ty2 - ty1), 1)
            for j in range(n_d):
                dx1, dy1, dx2, dy2 = det_boxes[j]
                da = max((dx2 - dx1) * (dy2 - dy1), 1)
                ix1 = max(tx1, dx1)
                iy1 = max(ty1, dy1)
                ix2 = min(tx2, dx2)
                iy2 = min(ty2, dy2)
                inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
                iou[i, j] = inter / (ta + da - inter + 1e-6)
        return iou

    def _augment_roi(self, roi: np.ndarray) -> np.ndarray:
        """轻度数据增强（防过拟合）：随机亮度/对比度微扰"""
        if roi.size == 0:
            return roi
        roi = roi.astype(np.float32)
        # 亮度微扰
        brightness = random.uniform(*self._aug_brightness_range)
        roi += brightness * 255
        # 对比度微扰
        contrast = random.uniform(*self._aug_contrast_range)
        roi = roi * contrast
        roi = np.clip(roi, 0, 255).astype(np.uint8)
        return roi

    def enhance_cigarette_contrast(self, roi: np.ndarray) -> np.ndarray:
        """浅色背景局部对比度增强: 强化香烟与背景的边缘, 浅色背景下识别更准
        自适应策略: ROI平均亮度 > 180(浅色背景场景)才触发, 暗光场景不多跑一步
        1. CLAHE(clipLimit=3.0, tileGridSize=4x4) 在 LAB 色彩空间 L 通道局部直方图均衡
        2. 3×3 锐化核(kernel=[-1,-1,-1; -1,9,-1; -1,-1,-1]) 边缘增强
        调用位置: ROI裁剪+基础增强之后, 送抽烟检测模型之前
        """
        if roi is None or roi.size == 0:
            return roi
        # 自适应开关: 平均亮度>180(浅色背景)才增强, 暗光/正常光直接返回
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        if float(gray.mean()) <= 180.0:
            return roi
        # 1) CLAHE 在 LAB L 通道局部直方图均衡化
        lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))
        l = clahe.apply(l)
        roi_enh = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)
        # 2) 3×3 锐化核边缘增强
        kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]], dtype=np.float32)
        roi_enh = cv2.filter2D(roi_enh, -1, kernel)
        return np.clip(roi_enh, 0, 255).astype(np.uint8)

    @staticmethod
    def _filter_parts_by_person(parts: list, person_boxes: list, max_offset: float = 1.5) -> list:
        """4.2 过滤越界零件框 — 零件中心必须在人体框 max_offset 倍范围内"""
        if not person_boxes or not parts:
            return parts
        keep = []
        for part in parts:
            bx, by, bx2, by2 = part["bbox"]
            cx, cy = (bx + bx2) / 2, (by + by2) / 2
            ok = False
            for pb in person_boxes:
                px1, py1, px2, py2 = pb
                pw, ph = px2 - px1, py2 - py1
                if pw <= 0 or ph <= 0:
                    continue
                if (px1 - max_offset * pw <= cx <= px2 + max_offset * pw
                        and py1 - max_offset * ph <= cy <= py2 + max_offset * ph):
                    ok = True
                    break
            if ok:
                keep.append(part)
        return keep

    @staticmethod
    def _nms_parts(parts: list, iou_thresh: float = 0.5) -> list:
        """4.3 NMS零件去重 — 同一人重叠手部/头部框合并，IoU>0.5视为重复"""
        if len(parts) <= 1:
            return parts
        n = len(parts)
        areas = np.array([(p["bbox"][2] - p["bbox"][0]) * (p["bbox"][3] - p["bbox"][1])
                          for p in parts], dtype=np.float32)
        # 按 track_id 分组，组内 NMS
        groups = {}
        for i, p in enumerate(parts):
            tid = p.get("track_id", -1)
            groups.setdefault(tid, []).append(i)
        keep_all = []
        for indices in groups.values():
            order = sorted(indices, key=lambda i: areas[i], reverse=True)
            keep = []
            while order:
                best = order[0]
                keep.append(best)
                remaining = []
                for j in order[1:]:
                    x1 = max(parts[best]["bbox"][0], parts[j]["bbox"][0])
                    y1 = max(parts[best]["bbox"][1], parts[j]["bbox"][1])
                    x2 = min(parts[best]["bbox"][2], parts[j]["bbox"][2])
                    y2 = min(parts[best]["bbox"][3], parts[j]["bbox"][3])
                    inter = max(0, x2 - x1) * max(0, y2 - y1)
                    iou = inter / (areas[best] + areas[j] - inter + 1e-6)
                    if iou < iou_thresh:
                        remaining.append(j)
                order = remaining
            keep_all.extend(keep)
        return [parts[i] for i in sorted(keep_all)]

    def _detect_persons_fast(self, frame: np.ndarray) -> List[List]:
        h, w = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 1/255.0, (self._person_img_size, self._person_img_size),
                                     swapRB=True, crop=False)
        outputs = self._person_session.run(None, {self._person_input_name: blob})
        preds = outputs[0]
        if len(preds.shape) == 3:
            preds = np.transpose(preds[0], (1, 0))

        # YOLOv8 ONNX 输出格式: [cx, cy, w, h, cls_0, cls_1, ..., cls_79]
        # 仅保留 class 0 (person) 的检测结果
        boxes_raw = preds[:, :4]
        class_scores = preds[:, 4:]
        class_ids = np.argmax(class_scores, axis=1)
        max_scores = np.max(class_scores, axis=1)

        # 双重过滤：类别=person(0) 且 置信度达标
        person_mask = (class_ids == self.PERSON_CLASS) & (max_scores > self._conf)
        if not np.any(person_mask):
            return []

        indices = np.where(person_mask)[0]
        order = np.argsort(max_scores[indices])[::-1]
        indices = indices[order]
        keep = self._nms(boxes_raw[indices], max_scores[indices])
        indices = indices[keep]

        scale_x = w / self._person_img_size
        scale_y = h / self._person_img_size
        detections = []
        for idx in indices:
            cx, cy, bw, bh = boxes_raw[idx]
            x1 = int((cx - bw / 2) * scale_x)
            y1 = int((cy - bh / 2) * scale_y)
            x2 = int((cx + bw / 2) * scale_x)
            y2 = int((cy + bh / 2) * scale_y)
            area = (x2 - x1) * (y2 - y1)
            if area < self._min_person_area:
                continue
            # 人体宽高比过滤：排除横向条状物体（空调、家具等）
            bw_box = x2 - x1
            bh_box = y2 - y1
            if bh_box <= 0: continue
            ratio = bw_box / bh_box
            # 太宽（>2.0）= 横向物体；太高（<0.3）= 竖向条状
            if ratio > 2.0 or ratio < 0.3:
                continue
            detections.append([x1, y1, x2, y2, float(max_scores[idx])])
        return detections

    def _detect_smoking_fast(self, roi_frame: np.ndarray) -> List[Dict]:
        if roi_frame.size == 0:
            return []
        rh, rw = roi_frame.shape[:2]
        blob = cv2.dnn.blobFromImage(roi_frame, 1/255.0, (self._smoke_img_size, self._smoke_img_size),
                                     swapRB=True, crop=False)
        outputs = self._smoke_session.run(None, {self._smoke_input_name: blob})
        preds = outputs[0]
        if len(preds.shape) == 3:
            preds = np.transpose(preds[0], (1, 0))

        boxes_raw = preds[:, :4]
        scores = preds[:, 4:]
        if self._smoke_num_classes == 1:
            max_scores = scores[:, 0]
            class_ids = np.zeros(len(max_scores), dtype=np.int32)
        else:
            class_ids = np.argmax(scores, axis=1)
            max_scores = np.max(scores, axis=1)

        mask = max_scores > self._conf
        if not np.any(mask):
            return []

        indices = np.where(mask)[0]
        order = np.argsort(max_scores[indices])[::-1]
        indices = indices[order]
        keep = self._nms(boxes_raw[indices], max_scores[indices])
        indices = indices[keep]

        scale_x = rw / self._smoke_img_size
        scale_y = rh / self._smoke_img_size
        results = []
        for idx in indices:
            cls_id = int(class_ids[idx])
            label = self.SMOKING_CLASSES.get(cls_id, f"class_{cls_id}")
            cx, cy, bw, bh = boxes_raw[idx]
            x1 = int((cx - bw / 2) * scale_x)
            y1 = int((cy - bh / 2) * scale_y)
            x2 = int((cx + bw / 2) * scale_x)
            y2 = int((cy + bh / 2) * scale_y)
            results.append({
                "label": label, "confidence": float(max_scores[idx]),
                "bbox": [x1, y1, x2, y2],
            })
        return results

    def _nms(self, boxes: np.ndarray, scores: np.ndarray) -> List[int]:
        x1 = boxes[:, 0] - boxes[:, 2] / 2
        y1 = boxes[:, 1] - boxes[:, 3] / 2
        x2 = boxes[:, 0] + boxes[:, 2] / 2
        y2 = boxes[:, 1] + boxes[:, 3] / 2
        areas = (x2 - x1) * (y2 - y1)
        order = np.argsort(scores)[::-1]
        keep = []
        while len(order) > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
            iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
            order = order[1:][iou < self._iou_thresh]
        return keep

    @staticmethod
    def _clip_roi(roi, img_w, img_h):
        x1, y1, x2, y2 = roi
        return (max(0, min(x1, img_w - 1)), max(0, min(y1, img_h - 1)),
                max(1, min(x2, img_w)), max(1, min(y2, img_h)))