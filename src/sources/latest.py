"""把摄像头采集放到独立线程，画面帧率不再被检测速度拖死。

为什么需要这一层
----------------
原来的循环是「读一帧 → 检测 → 读一帧」，检测一帧要 100ms，摄像头就读成 9fps。
哪怕源是 30fps 的，画面看起来也是卡顿的。

这里采集线程只管不停地读、只保留最新一帧；检测端每次取最新帧，
画面发布线程另走一条路也取最新帧。三条路互不等待：

    采集线程 ──▶ 最新帧槽 ──┬──▶ 检测（慢，9fps，可以跳帧）
                            └──▶ 画面发布（快，25fps，永远拿最新的）

代价是检测可能拿到"与上一帧相同"的画面（源比检测快时），这没问题 ——
检测本来就允许跳帧，烟雾是慢过程。
"""
from __future__ import annotations

import threading
import time


class LatestFrameSource:
    """包在任意 CameraSource 外面，让 read() 永远返回最新帧。"""

    def __init__(self, inner, *, log_every_s: float = 0.0):
        self.inner = inner
        self._cond = threading.Condition()
        self._pkt = None
        self._seq = 0
        self._stop = False
        self._read_errors = 0
        self._th = threading.Thread(target=self._loop, name="capture", daemon=True)
        self._th.start()

    # ---------------------------------------------------------- 采集线程
    def _loop(self) -> None:
        while not self._stop:
            try:
                pkt = self.inner.read()
            except Exception:                    # noqa: BLE001
                self._read_errors += 1
                time.sleep(0.01)
                continue
            if pkt is None:
                if getattr(self.inner, "closed", False):
                    break
                time.sleep(0.003)
                continue
            with self._cond:
                self._pkt = pkt
                self._seq += 1
                self._cond.notify_all()

    # ---------------------------------------------------------- 对外
    @property
    def closed(self) -> bool:
        return self._stop or bool(getattr(self.inner, "closed", False))

    def read(self):
        """检测端用：拿当前最新帧（可能和上次是同一帧）。"""
        with self._cond:
            return self._pkt

    def read_new(self, last_seq: int, timeout: float = 0.5):
        """画面端用：等一帧新的。返回 (pkt, seq)，超时则返回同一帧同上 seq。"""
        with self._cond:
            self._cond.wait_for(
                lambda: self._seq != last_seq or self._stop, timeout=timeout)
            return self._pkt, self._seq

    def read_blocking(self, timeout: float = 0.2):
        """给 pipeline 的 read() 用：没新帧就稍等，避免空转拉满 CPU。"""
        with self._cond:
            self._cond.wait_for(
                lambda: self._seq != getattr(self, "_last_given", -1) or self._stop,
                timeout=timeout)
            self._last_given = self._seq
            return self._pkt

    def close(self) -> None:
        self._stop = True
        with self._cond:
            self._cond.notify_all()
        try:
            self.inner.close()
        except Exception:                        # noqa: BLE001
            pass
