"""实时画面：把每帧的检测结果画成带框的图，供浏览器以 MJPEG 观看。

为什么框画在后端而不是前端叠加
------------------------------
前端叠加需要同时同步两条流（图像 + 框坐标）。网络一抖动，框就会和图错位几帧，
看起来像"框飘了"。直接在后端把框画死，前端一个 <img src="/stream"> 就能看，
弱网最差也只是掉帧，不会框图不匹配。

详细的框信息（类别、置信度、轨迹 ID）另外走 /api/state 给前端列表用，
所以"能不能看清框"和"能不能查细节"这两件事互不干扰。

为什么限流
----------
检测本身 16~19 fps，但画框 + JPEG 编码是纯 CPU 开销。如果不限流，
CPU 会被视频编码吃掉，反而拖慢检测。默认压到 12 fps，肉眼完全够。

中文标签说明
------------
cv2.putText 不支持中文，硬写会变成一串问号。这里类别名一律用英文大写
（FIRE / SMOKE），颜色才是主要的区分手段：火红、烟橙、其他灰。
"""
from __future__ import annotations

import threading
import time

import cv2
import numpy as np

# 类别 → 框颜色（BGR）。0=fire 1=smoke，与 dataset.yaml 一致
COLORS = {
    0: (48, 48, 235),      # 红：火（达到告警阈值）
    1: (40, 165, 255),     # 橙：烟（达到告警阈值）
}
# 未达告警阈值的候选框用黄色细线。
# 目的是把"看见了"和"要报警"在视觉上也分开 —— 用户能看到系统注意到那块区域了，
# 只是置信度还不足以惊动人。之前两者共用一个阈值，调高就再也看不见真火。
CAND_COLORS = {
    0: (0, 210, 255),
    1: (60, 230, 255),
}
DEFAULT_COLOR = (190, 190, 190)

FONT = cv2.FONT_HERSHEY_SIMPLEX


class LiveView:
    """最新帧的单一存放点：pipeline 写，HTTP 线程读。"""

    def __init__(self, *, max_width: int = 1280, jpeg_quality: int = 80,
                 fps_limit: float = 12.0, max_bytes: int = 90000,
                 fallback_quality: int = 50):
        self.max_width = max_width
        self.jpeg_quality = int(jpeg_quality)
        self.min_interval = 1.0 / max(fps_limit, 1.0)
        # 体积兜底：静止画面一帧约 26KB，但画面剧烈变化（开关灯、大幅移动）
        # 时压缩率骤降，实测能到 143KB —— 编码本身只多花 0.4ms，可传输和
        # 浏览器解码会明显变慢，前端串行拉取时表现为"画面一变就停一下"。
        # 超过上限就降一档质量重编，体积能砍一半，代价才 0.3ms。
        self.max_bytes = int(max_bytes)
        self.fallback_quality = int(fallback_quality)
        # 各类别的告警阈值：低于它的框画成黄色"候选"，高于它才是红色"告警"
        self.alarm_confs: dict[int, float] = {0: 0.55, 1: 0.60}

        self._cond = threading.Condition(threading.Lock())
        self._jpeg: bytes | None = None
        self._seq = 0
        self._state: dict = {"seq": 0, "dets": [], "counts": {}, "fps": 0.0}

        self._last_pub = 0.0
        self._n_pub = 0
        self._t_first = time.time()
        self.llm = None                      # 由外部注入 LlmArbiter，供 /api/llm 用

    # ------------------------------------------------------------ 写入
    def publish(self, image: np.ndarray, dets, *, frame_idx: int | None = None,
                cam_id: str = "", fps: float = 0.0, extra: dict | None = None) -> bool:
        """由 pipeline 每帧调用。返回本次是否真的编码了一帧（被限流时为 False）。"""
        now = time.time()
        if now - self._last_pub < self.min_interval:
            return False
        self._last_pub = now

        try:
            drawn = self._draw(image, dets, frame_idx, cam_id)
            ok, buf = cv2.imencode(".jpg", drawn,
                                   [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality])
            if not ok:
                return False
            if len(buf) > self.max_bytes:
                ok2, buf2 = cv2.imencode(
                    ".jpg", drawn,
                    [int(cv2.IMWRITE_JPEG_QUALITY), self.fallback_quality])
                if ok2 and len(buf2) < len(buf):
                    buf = buf2
            jpeg = buf.tobytes()
        except Exception:                      # noqa: BLE001  画框失败绝不能拖垮检测
            return False

        counts: dict[str, int] = {}
        det_list = []
        for d in dets:
            name = getattr(d, "cls_name", "obj")
            counts[name] = counts.get(name, 0) + 1
            det_list.append({
                "cls_id": int(getattr(d, "cls_id", -1)),
                "name": name,
                "conf": round(float(getattr(d, "conf", 0.0)), 4),
                "xyxy": [round(float(v), 1) for v in getattr(d, "xyxy", (0, 0, 0, 0))],
            })

        self._n_pub += 1
        with self._cond:
            self._jpeg = jpeg
            self._seq += 1
            self._state = {
                "seq": self._seq,
                "ts": round(now, 3),
                "frame_idx": frame_idx,
                "cam": cam_id,
                "size": [int(image.shape[1]), int(image.shape[0])],
                "dets": det_list,
                "counts": counts,
                "fps": round(fps, 2),
                "pub_fps": round(self._n_pub / max(1e-6, now - self._t_first), 2),
                **(extra or {}),
            }
            self._cond.notify_all()
        return True

    # ------------------------------------------------------------ 画框
    def _draw(self, image: np.ndarray, dets, frame_idx, cam_id) -> np.ndarray:
        img = image.copy()
        h, w = img.shape[:2]
        placed: list[tuple[int, int, int, int]] = []   # 已占用的标签区域，防重叠

        def collides(box) -> bool:
            x1, y1, x2, y2 = box
            return any(not (x2 <= a or x1 >= c or y2 <= b or y1 >= d)
                       for a, b, c, d in placed)

        for d in dets:
            x1, y1, x2, y2 = (int(round(v)) for v in getattr(d, "xyxy", (0, 0, 0, 0)))
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w - 1, x2), min(h - 1, y2)
            if x2 <= x1 or y2 <= y1:
                continue

            cid = int(getattr(d, "cls_id", -1))
            conf = float(getattr(d, "conf", 0.0))
            is_alarm = conf >= self.alarm_confs.get(cid, 0.55)
            color = (COLORS if is_alarm else CAND_COLORS).get(cid, DEFAULT_COLOR)
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2 if is_alarm else 1)

            name = str(getattr(d, "cls_name", "obj")).upper()
            label = f"{name} {conf:.2f}" if is_alarm else f"{name} {conf:.2f} ?"
            (tw, th), base = cv2.getTextSize(label, FONT, 0.5, 1)
            bh = th + base + 4

            # 默认贴在框的上沿；与已画的标签撞了就往下挪，最多挪 6 次
            ly = y1 - bh - 2
            for _ in range(6):
                if ly >= 0 and not collides((x1, ly, x1 + tw + 6, ly + bh)):
                    break
                ly += bh
            if ly < 0:
                ly = y1 + 2
            ly = max(0, min(h - bh - 1, ly))
            placed.append((x1, ly, x1 + tw + 6, ly + bh))

            cv2.rectangle(img, (x1, ly), (x1 + tw + 6, ly + bh), color, -1)
            cv2.putText(img, label, (x1 + 3, ly + th + 1), FONT, 0.5,
                        (255, 255, 255), 1, cv2.LINE_AA)

        # 左上角信息条：帧号 / 检测数 / 时间
        bar = f"f{frame_idx if frame_idx is not None else '-'}  det {len(dets)}  " \
              f"{time.strftime('%H:%M:%S')}"
        if cam_id:
            bar = f"{cam_id}  " + bar
        cv2.putText(img, bar, (8, 22), FONT, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(img, bar, (8, 22), FONT, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

        if w > self.max_width:                  # 控制带宽与解码开销
            scale = self.max_width / w
            img = cv2.resize(img, (self.max_width, int(h * scale)),
                             interpolation=cv2.INTER_AREA)
        return img

    # ------------------------------------------------------------ 读出
    def state(self) -> dict:
        with self._cond:
            return dict(self._state)

    def frame_bytes(self) -> bytes | None:
        """最新一帧的 JPEG 字节，供 /api/frame.jpg 逐帧拉取。

        为什么还要留这条路径：MJPEG（multipart/x-mixed-replace）在部分浏览器、
        内嵌 WebView 和预览面板里根本不会渲染，表现为"黑屏但接口有数据"。
        逐帧拉取是普通 img 请求，任何环境都能显示，代价只是多几个 HTTP 请求。
        """
        with self._cond:
            return self._jpeg

    def mjpeg_iter(self):
        """持续产出 multipart 分片。调用方负责写响应头。"""
        last = -1
        while True:
            with self._cond:
                self._cond.wait_for(lambda: self._seq != last, timeout=2.0)
                jpeg, seq = self._jpeg, self._seq
            if jpeg is None or seq == last:
                continue                        # 还没第一帧，或只是等超时
            last = seq
            yield (b"--frame\r\nContent-Type: image/jpeg\r\n"
                   + f"Content-Length: {len(jpeg)}\r\n\r\n".encode()
                   + jpeg + b"\r\n")

    # ------------------------------------------------------------ LLM 状态
    def llm_state(self) -> dict:
        """给 /api/llm 用。没注入 arbiter 时返回明确的未配置状态。"""
        if self.llm is None:
            return {"configured": False, "reason": "主程序未启用 LLM 裁判"}
        try:
            return self.llm.snapshot()
        except Exception as e:                  # noqa: BLE001
            return {"configured": False, "reason": f"读取失败: {e}"}
