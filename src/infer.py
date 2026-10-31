"""检测器封装：整图推理 + 切片推理（SAHI 思路）。

为什么需要切片推理
------------------
v3 模型的诊断结论（2026-10-30，test 集 250 张实测）：

|          | 框面积中位 | 整图 640 召回 | 2x2 切片召回 |
|----------|-----------|--------------|-------------|
| fire     | 0.84%     | 0.521        | **0.601**   |
| smoke    | 6.71%     | 0.593        | 0.593       |

火的目标太小 —— 在 640 输入下只剩 20~30 个像素，被下采样吃掉了。
把同一张大图切成 2x2 带重叠的小块、每块各自放大到 640 再检测，
等效于「局部放大」，火召回直接 +8 个点；而烟本来就大，切片对它没有任何收益。

所以切片不是无脑开：**只在小目标（fire）掉得多的场景开**，代价是算力约 4 倍。

设计约束
--------
1. 切片 + 整图可以叠加（sliced=True 时同时跑整图，兜住跨切片的大目标，如整片烟）
2. 合并必须在**原图坐标系**做 NMS，否则同一个目标会被切片边界切成两半后重复上报
3. 全部纯 numpy，不引入 torchvision / sahi 依赖（现场部署少一个依赖少一个坑）
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np


@dataclass
class Detection:
    """一个检测框（坐标是原图像素，xyxy）。"""

    cls_id: int
    cls_name: str
    conf: float
    xyxy: tuple[float, float, float, float]

    @property
    def area(self) -> float:
        x1, y1, x2, y2 = self.xyxy
        return max(0.0, x2 - x1) * max(0.0, y2 - y1)

    def center(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.xyxy
        return (x1 + x2) / 2, (y1 + y2) / 2


def iou(a: Sequence[float], b: Sequence[float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix = min(ax2, bx2) - max(ax1, bx1)
    iy = min(ay2, by2) - max(ay1, by1)
    if ix <= 0 or iy <= 0:
        return 0.0
    inter = ix * iy
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / union if union > 0 else 0.0


def nms(dets: Iterable[Detection], iou_thr: float = 0.5) -> list[Detection]:
    """按类别做贪心 NMS（不同类之间不抑制：烟里的火不该被烟框吃掉）。"""
    out: list[Detection] = []
    by_cls: dict[int, list[Detection]] = {}
    for d in dets:
        by_cls.setdefault(d.cls_id, []).append(d)
    for _, group in by_cls.items():
        group = sorted(group, key=lambda d: -d.conf)
        keep: list[Detection] = []
        for d in group:
            if all(iou(d.xyxy, k.xyxy) < iou_thr for k in keep):
                keep.append(d)
        out.extend(keep)
    return sorted(out, key=lambda d: -d.conf)


class Detector:
    """YOLO 检测器封装，支持整图 / 切片两种推理方式。

    参数
    ----
    weights   : .pt 权重路径
    conf      : 置信度门限（部署建议 fire 0.30 / smoke 0.45，这里用统一值，可再按类过滤）
    imgsz     : 单次前向的输入尺寸。**不必等于训练尺寸** —— v5 实测（test 集）推理改 896
                时 fire mAP50-95 从 0.4194 升到 0.4332，比按训练尺寸 960 推理更好。
                原因：YOLO 推理可以做 letterbox 缩放，896 的框定位更贴边，960 反而摊薄了密度
    tile_grid : 切片数（2 表示 2x2）。1 = 不切片
    overlap   : 切片重叠比例，0.25 表示相邻切片重叠 25% 宽高（防目标正好落在切线）
    """

    def __init__(
        self,
        weights: str,
        conf: float = 0.35,
        iou: float = 0.5,
        imgsz: int = 896,
        device: str | int = 0,
        classes: Sequence[int] | None = None,
    ) -> None:
        from ultralytics import YOLO  # 延迟导入：只做标注审计时不必装 ultralytics

        self.model = YOLO(weights)
        self.names = self.model.names
        self.conf = conf
        self.iou = iou
        self.imgsz = imgsz
        self.device = device
        self.classes = list(classes) if classes is not None else None

    # ---------------------------------------------------------------- 内部
    def _predict(self, img: np.ndarray, offset: tuple[int, int] = (0, 0)) -> list[Detection]:
        r = self.model.predict(
            img, imgsz=self.imgsz, conf=self.conf, iou=self.iou,
            device=self.device, classes=self.classes, verbose=False,
        )[0]
        ox, oy = offset
        out = []
        for b in r.boxes:
            x1, y1, x2, y2 = b.xyxy[0].tolist()
            cid = int(b.cls)
            out.append(Detection(cid, self.names[cid], float(b.conf),
                                 (x1 + ox, y1 + oy, x2 + ox, y2 + oy)))
        return out

    def _tiles(self, h: int, w: int, grid: int, overlap: float):
        """生成 (x0, y0, x1, y1) 切片窗口，边缘会被裁到图内。"""
        th, tw = h / grid, w / grid
        ox, oy = int(tw * overlap), int(th * overlap)
        for gy in range(grid):
            for gx in range(grid):
                x0 = max(0, int(gx * tw) - ox)
                y0 = max(0, int(gy * th) - oy)
                x1 = min(w, int((gx + 1) * tw) + ox)
                y1 = min(h, int((gy + 1) * th) + oy)
                if x1 - x0 >= 32 and y1 - y0 >= 32:
                    yield x0, y0, x1, y1

    # ---------------------------------------------------------------- 对外
    def detect(
        self,
        image: np.ndarray,
        sliced: bool = False,
        tile_grid: int = 2,
        overlap: float = 0.25,
        keep_full_frame: bool = True,
    ) -> list[Detection]:
        """检测一帧。

        sliced=True 时：
          · keep_full_frame=True（默认）先跑一遍整图，兜住横跨多个切片的大烟团
          · 再跑 grid×grid 个重叠切片，各自放大到 imgsz 后检测
          · 所有框映射回原图坐标后统一 NMS
        """
        h, w = image.shape[:2]
        dets: list[Detection] = []

        if not sliced:
            return nms(self._predict(image), self.iou)

        if keep_full_frame:
            dets.extend(self._predict(image))
        for x0, y0, x1, y1 in self._tiles(h, w, tile_grid, overlap):
            crop = image[y0:y1, x0:x1]
            if crop.size == 0:
                continue
            dets.extend(self._predict(crop, offset=(x0, y0)))
        return nms(dets, self.iou)
