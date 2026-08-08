"""
decision_engine.py - 多层决策引擎 v2
空间推理: 区分真实抽烟 / 手近脸无烟 / 手持香烟未吸
时间确认: 连续帧确认 + 滑动窗口 + 冷却机制
"""
from collections import defaultdict
import time
from datetime import datetime
from typing import List, Dict, Optional, Tuple


class DecisionEngine:
    """抽烟告警决策引擎 — 空间+时间双重推理"""

    # 告警级别
    LEVEL_SMOKING = "smoking"       # 确认抽烟：香烟在嘴部ROI 或 手近脸+持烟
    LEVEL_SUSPICIOUS = "suspicious" # 可疑：持烟但手未贴近脸部
    LEVEL_HAND_NEAR = "hand_near"   # 仅手近脸：无香烟检测

    def __init__(self, config: dict):
        self._confirm = config["confirm_frames"]
        self._window = config["confirm_window_seconds"]
        self._cooldown = config["cooldown_seconds"]
        self._history: Dict[str, List[Tuple[float, str]]] = defaultdict(list)
        self._last_alert: Dict[str, float] = {}

    def evaluate(self, cam_id: str, detections: List[Dict]) -> Tuple[str, Optional[Dict]]:
        """
        空间+时间双重推理评估
        返回: (alert_level, detail)
           alert_level: "smoking" | "suspicious" | "hand_near" | None
        """
        now = time.time()

        if not detections:
            self._history[cam_id].append((now, None))
            return None, None

        # === 空间推理：分类每个告警 ===
        mouth_cigs = []    # 嘴部ROI检测到香烟 → 确认抽烟
        hand_cigs_near = []  # 手部ROI检测到香烟 + 手近脸 → 确认抽烟
        hand_cigs_far = []   # 手部ROI检测到香烟 + 手远离脸 → 可疑
        hand_near_no_cig = []  # 手近脸但无烟 → 仅手近脸

        for alert in detections:
            roi_type = alert.get("roi_type", "")
            overlap = alert.get("hand_mouth_overlap", 0.0)

            if roi_type in ("mouth", "head"):
                # 香烟在头部/嘴部区域 → 确认抽烟
                mouth_cigs.append(alert)
            elif roi_type in ("hand_r", "hand_l"):
                if overlap > 0.1:
                    # 手贴近嘴部 + 检测到香烟 → 确认抽烟
                    hand_cigs_near.append(alert)
                else:
                    # 手远离嘴部 + 检测到香烟 → 可疑（可能只是拿着烟）
                    hand_cigs_far.append(alert)

        # 手近脸但无香烟：通过 overlap 判断（detections 中没有香烟但评估时可能有）
        # 这里我们需要在 call site 检查是否有手近脸但无烟的情况

        # === 综合判定 ===
        if mouth_cigs or hand_cigs_near:
            # 确认抽烟
            level = self.LEVEL_SMOKING
            all_cigs = mouth_cigs + hand_cigs_near
            best = max(all_cigs, key=lambda d: d["confidence"])
            reason = []
            if mouth_cigs:
                reason.append("头部/嘴部区域检测到疑似香烟")
            if hand_cigs_near:
                reason.append("手近脸部且持疑似香烟")
            self._track(cam_id, now, level)
        elif hand_cigs_far:
            # 可疑：持烟但手不在脸部附近
            level = self.LEVEL_SUSPICIOUS
            best = max(hand_cigs_far, key=lambda d: d["confidence"])
            reason = ["手部检测到疑似香烟，但手未贴近脸部"]
            self._track(cam_id, now, level)
        else:
            # 有检测但都不符合抽烟条件
            return None, None

        # === 时间确认：连续帧 + 冷却 ===
        if self._confirm > 1:
            positive = sum(1 for _, lv in self._history[cam_id]
                          if lv and lv in (self.LEVEL_SMOKING, self.LEVEL_SUSPICIOUS))
            if positive < self._confirm:
                return None, None

        # 冷却检查
        last = self._last_alert.get(cam_id, 0)
        if now - last < self._cooldown:
            return None, None

        self._last_alert[cam_id] = now

        return level, {
            "cam_id": cam_id,
            "level": level,
            "confidence": best["confidence"],
            "label": best.get("label", ""),
            "roi_type": best.get("roi_type", ""),
            "hand_mouth_overlap": best.get("hand_mouth_overlap", 0.0),
            "reason": "; ".join(reason),
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "track_id": best.get("track_id", -1),
        }

    def _track(self, cam_id: str, now: float, level: str):
        self._history[cam_id].append((now, level))
        cutoff = now - self._window
        self._history[cam_id] = [(t, lv) for t, lv in self._history[cam_id] if t > cutoff]

    def reset(self, cam_id: str = None):
        if cam_id:
            self._history.pop(cam_id, None)
            self._last_alert.pop(cam_id, None)
        else:
            self._history.clear()
            self._last_alert.clear()