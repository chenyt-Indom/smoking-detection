"""采集层的数据契约：摄像头抽象 + 帧包。

为什么要把「固定机位」和「云台」分成两种源
----------------------------------------
它们的**背景会不一样**：

· 固定机位：背景长期不变 → 可以用背景建模（中值背景 / MOG2）把"新出现的东西"
  （包括**很淡、很早期的烟**）捞出来，这是检测器单帧看不出来的
· 云台：画面一直在动 → 背景建模直接失效（整幅都是"前景"），
  必须先估计全局运动、把画面**对齐回参考坐标**，差分才有意义

所以 FramePacket 里除了图像，还要带上：
  · `moving`      云台是否正在转动 —— 转动期间的轨迹必须丢弃（否则同一条烟会被拉花）
  · `homography`  相对参考帧的单应矩阵 —— 给后续模块做坐标统一
  · `fg_mask`     前景掩码 —— 固定机位/补偿后算出来的"变化区域"
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np


@dataclass
class CameraInfo:
    cam_id: str
    kind: str                     # "fixed" 固定机位 / "ptz" 云台
    name: str = ""
    fps: float = 25.0
    note: str = ""


@dataclass
class FramePacket:
    cam_id: str
    idx: int                      # 帧序号（自增）
    ts: float                     # 时间戳（秒，单调）
    image: np.ndarray             # BGR
    moving: bool = False          # 云台转动中（固定机位恒 False）
    homography: np.ndarray | None = None   # 当前帧 → 参考帧 的单应
    fg_mask: np.ndarray | None = None      # uint8 0/255 前景
    meta: dict = field(default_factory=dict)


class CameraSource(ABC):
    """所有摄像头源的统一接口。

    实现类只需要保证：`read()` 返回 None 表示这一路暂时没有新帧（不是结束）；
    结束信号用 `closed` 属性表达。
    """

    def __init__(self, info: CameraInfo):
        self.info = info
        self.closed = False

    @abstractmethod
    def read(self) -> FramePacket | None:
        ...

    def close(self) -> None:
        self.closed = True

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
