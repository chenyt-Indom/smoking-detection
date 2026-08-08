"""
alerter.py - 告警处理模块
触发告警时保存截图证据帧
"""
import cv2
import os


class Alerter:
    def __init__(self, config: dict):
        self._dir = config.get("screenshot_dir", "alerts")
        self._save = config.get("save_screenshot", True)
        os.makedirs(self._dir, exist_ok=True)

    def trigger(self, cam_name: str, detail: dict, frame):
        """触发告警 - 保存截图"""
        if self._save and frame is not None:
            ts = detail["timestamp"].replace(":", "").replace(" ", "_")
            filepath = os.path.join(self._dir, f"{cam_name}_{ts}.jpg")
            cv2.imwrite(filepath, frame)