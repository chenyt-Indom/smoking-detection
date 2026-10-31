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
    def _predict(self, img: np.ndarray, offset: tuple[int, int] = (0, 0),
                 conf: float | None = None) -> list[Detection]:
        r = self.model.predict(
            img, imgsz=self.imgsz, conf=self.conf if conf is None else conf,
            iou=self.iou, device=self.device, classes=self.classes, verbose=False,
        )[0]
        ox, oy = offset
        out = []
        for b in r.boxes:
            x1, y1, x2, y2 = b.xyxy[0].tolist()
            cid = int(b.cls)
            out.append(Detection(cid, self.names[cid], float(b.conf),
                                 (x1 + ox, y1 + oy, x2 + ox, y2 + oy)))
        return out

    def _predict_batch(self, imgs: list[np.ndarray],
                       offsets: list[tuple[int, int]]) -> list[Detection]:
        """一次前向跑多张图。分级检测要补切好几块，逐块跑很亏 ——
        实测 batch=4 合计 23.3ms，逐块跑是 4×11.9ms（1.9 倍差距）。"""
        if not imgs:
            return []
        if len(imgs) == 1:
            return self._predict(imgs[0], offsets[0])
        rs = self.model.predict(
            list(imgs), imgsz=self.imgsz, conf=self.conf, iou=self.iou,
            device=self.device, classes=self.classes, verbose=False,
        )
        out = []
        for r, (ox, oy) in zip(rs, offsets):
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
        batched: bool = True,
    ) -> list[Detection]:
        """检测一帧。

        sliced=True 时：
          · keep_full_frame=True（默认）先跑一遍整图，兜住横跨多个切片的大烟团
          · 再跑 grid×grid 个重叠切片，各自放大到 imgsz 后检测
          · 所有框映射回原图坐标后统一 NMS

        batched=True 时把上面这些图**一次性**送进网络。逐张跑 5 次前向和
        批量跑 5 张，结果完全一样（同样输入、同样权重），但 GPU 只在一次
        kernel launch 里算完，实测 54.8ms → 28ms 左右。这个提速不拿召回换。
        """
        h, w = image.shape[:2]

        if not sliced:
            return nms(self._predict(image), self.iou)

        crops: list[np.ndarray] = []
        offs: list[tuple[int, int]] = []
        if keep_full_frame:
            crops.append(image)
            offs.append((0, 0))
        for x0, y0, x1, y1 in self._tiles(h, w, tile_grid, overlap):
            crop = image[y0:y1, x0:x1]
            if crop.size == 0:
                continue
            crops.append(crop)
            offs.append((x0, y0))

        if batched:
            return nms(self._predict_batch(crops, offs), self.iou)

        dets: list[Detection] = []
        for crop, off in zip(crops, offs):
            dets.extend(self._predict(crop, offset=off))
        return nms(dets, self.iou)

    def detect_auto(
        self,
        image: np.ndarray,
        trigger_conf: float = 0.08,
        tile_grid: int = 2,
        overlap: float = 0.25,
    ) -> list[Detection]:
        """分级检测：先粗筛一帧，**只有粗筛出了东西才切片**。

        为什么要这么干
        --------------
        真实枪机 99% 的帧是空的（停车场、走廊、仓库），而 `sliced=True` 是
        每帧固定跑 5 次前向（整图 + 2x2 切片），实测 52ms，整条流水线 18fps ——
        全是给空白帧付的钱。空白帧只值 1 次前向：10.7ms。

        为什么"要切就切全"，而不是只补可疑区域
        --------------------------------------
        试过围着候选框补 1~3 块高分辨率小片（更省），实测 fire 召回从 67.8%
        掉到 61.7%：候选框漏掉的区域就是永远补不到的区域，等于把"没看见"
        固化成了"没看见"。切片真正的价值是**整幅无死角地放大 3.3 倍**，
        不是把某块放得更大。所以慢路径原样保留 2x2，召回一个点都不许掉。

        为什么粗筛用 0.05 这么低的门限
        ------------------------------
        这张粗筛图同时充当整图那一路（结果按正常 conf 过滤后再合并），
        所以压低门限不会多花一次前向。门限越低，"有东西"的判定越松，
        越不容易把该切的帧漏掉 —— 而代价只是更多帧走慢路径。
        """
        h, w = image.shape[:2]
        coarse = self._predict(image, conf=trigger_conf)
        if not coarse:
            return []                  # 一片空白：不切片，这是快路径

        crops: list[np.ndarray] = []
        offs: list[tuple[int, int]] = []
        for x0, y0, x1, y1 in self._tiles(h, w, tile_grid, overlap):
            crop = image[y0:y1, x0:x1]
            if crop.size == 0:
                continue
            crops.append(crop)
            offs.append((x0, y0))
        # 粗筛那一帧就是"整图"那一路，别重复跑
        kept = [d for d in coarse if d.conf >= self.conf]
        return nms(kept + self._predict_batch(crops, offs), self.iou)
