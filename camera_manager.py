"""
camera_manager.py - 多摄像头极速帧采集管线
支持本地摄像头、RTSP 流、视频文件
每路独立线程采集，grab/retrieve 极速模式，毫秒级帧缓冲
"""
import cv2
import threading
import time
import numpy as np
from queue import Queue, Empty
from typing import Optional, List


class CameraManager:
    """多摄像头管理器"""

    def __init__(self, cameras_config: list):
        self._cameras = [Camera(cfg) for cfg in cameras_config]

    @property
    def active_cameras(self) -> List:
        return [c for c in self._cameras if c.enabled]

    @property
    def all_cameras(self) -> List:
        return self._cameras

    def add(self, config: dict):
        self._cameras.append(Camera(config))

    def remove(self, cam_id: str):
        cam = next((c for c in self._cameras if c.id == cam_id), None)
        if cam:
            cam.stop()
            self._cameras.remove(cam)

    def to_config_list(self) -> list:
        return [{
            "id": c.id, "name": c.name, "source": c.source, "enabled": c.enabled,
        } for c in self._cameras]

    def stop_all(self):
        for cam in self._cameras:
            cam.stop()


class Camera:
    """单个摄像头 - 独立采集线程（grab/retrieve 极速模式）"""

    def __init__(self, config: dict):
        self.id = config["id"]
        self.name = config["name"]
        self.source = config["source"]
        self.enabled = config.get("enabled", True)
        self.roi = config.get("roi")
        self._cap: Optional[cv2.VideoCapture] = None
        self._queue = Queue(maxsize=3)
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._fps = 0.0
        self._frame_count = 0
        self._last_fps_time = time.time()

    def start(self):
        source = int(self.source) if isinstance(self.source, int) else self.source
        # USB 摄像头使用 DSHOW 后端（Windows 优化），RTSP 流使用默认 FFMPEG 后端
        if isinstance(source, int):
            self._cap = cv2.VideoCapture(source, cv2.CAP_DSHOW)
        else:
            self._cap = cv2.VideoCapture(source)
        if not self._cap.isOpened():
            raise RuntimeError(f"无法打开摄像头: {self.name} ({self.source})")

        # USB 摄像头：极速模式 MJPG 编码 + 最大 FPS + 最优分辨率
        # RTSP 流：不强制设置编码/分辨率，使用流原始参数
        if isinstance(source, int):
            self._cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
            self._cap.set(cv2.CAP_PROP_FPS, 60)
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)    # 最小缓冲

        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self):
        """极速采集循环：grab/retrieve 分离减少解码阻塞"""
        while self._running and self._cap and self._cap.isOpened():
            grabbed = self._cap.grab()
            if not grabbed:
                continue
            ret, frame = self._cap.retrieve()
            if not ret:
                continue

            if self.roi:
                x, y, w, h = self.roi
                frame = frame[y:y + h, x:x + w]

            # 丢旧帧，只保留最新帧
            if self._queue.full():
                try:
                    self._queue.get_nowait()
                except Empty:
                    pass
            self._queue.put(frame)

            self._frame_count += 1
            now = time.time()
            elapsed = now - self._last_fps_time
            if elapsed >= 1.0:
                self._fps = self._frame_count / elapsed
                self._frame_count = 0
                self._last_fps_time = now

    def read(self) -> Optional[np.ndarray]:
        try:
            return self._queue.get_nowait()
        except Empty:
            return None

    def read_blocking(self, timeout: float = 0.1) -> Optional[np.ndarray]:
        try:
            return self._queue.get(timeout=timeout)
        except Empty:
            return None

    @property
    def fps(self) -> float:
        return self._fps

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=1.0)
        if self._cap:
            self._cap.release()