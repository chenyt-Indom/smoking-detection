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
import cv2


@dataclass
class Detection:
    """一个检测框（坐标是原图像素，xyxy）。"""

    cls_id: int
    cls_name: str
    conf: float
    xyxy: tuple[float, float, float, float]
    # 轨迹 ID（-1 = 还没分配）。画框时显示成 #12 这种，
    # 方便肉眼确认"这是不是同一个目标"——用户提的需求：
    # "给方框一个id，不要因为中间断过几帧就更换ID"。
    tid: int = -1

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
    """按类别做贪心 NMS（不同类之间不抑制：烟里的火不该被烟框吃掉）。

    ⚠️ 分组用**类名**而不是 cls_id。
    主模型是 {0: fire, 1: smoke}，香烟模型是 {0: cigarette} —— 两者的 0 是
    不同的东西。按 id 分组会把"火"和"香烟"塞进同一组互相抑制，
    同一处既着火又有烟时会凭空删掉一个框。按类名分组才正确。
    """
    out: list[Detection] = []
    by_cls: dict[str, list[Detection]] = {}
    for d in dets:
        by_cls.setdefault(str(d.cls_name).lower(), []).append(d)
    for _, group in by_cls.items():
        group = sorted(group, key=lambda d: -d.conf)
        keep: list[Detection] = []
        for d in group:
            drop = False
            for k in keep:
                # ⚠️ 只用 IOU 会漏掉"小框整个包在大框里"的情况。
                # 一支烟的框 26x43 被另一个 40x70 的框整个罩住时，
                # 交集=26x43、并集=40x70，IOU 只有 ~0.35；若两个框尺寸差更多，
                # IOU 会掉到 0.2 以下 —— 低于 0.25 的门槛就不会被压掉，
                # 画面上就成了"同一支烟两个框"（用户截图里的 0.41 和 0.18）。
                #
                # 所以除了 IOU，还要看**包含比**：交集占较小框面积的比例。
                # 小框有 70% 以上落在大框里，就是重复检测，删掉小的那个。
                a = d.area
                b = k.area
                inter = max(0.0, min(d.xyxy[2], k.xyxy[2]) - max(d.xyxy[0], k.xyxy[0])) * \
                        max(0.0, min(d.xyxy[3], k.xyxy[3]) - max(d.xyxy[1], k.xyxy[1]))
                contain = inter / max(1.0, min(a, b))
                if iou(d.xyxy, k.xyxy) >= iou_thr or contain >= 0.70:
                    drop = True
                    break
            if not drop:
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
        iou: float = 0.45,
        imgsz: int = 896,
        device: str | int = 0,
        classes: Sequence[int] | None = None,
        extra_weights: str | None = None,
        extra_conf: float | None = None,
    ) -> None:
        from ultralytics import YOLO  # 延迟导入：只做标注审计时不必装 ultralytics

        self.model = YOLO(weights)
        self.names = self.model.names
        self.conf = conf
        self.iou = iou
        self.imgsz = imgsz
        self.device = device
        self.classes = list(classes) if classes is not None else None
        # 第二个模型（香烟）：与主模型分开推理，结果在 detect 里合并。
        # 香烟本体和明火是两套完全不同的训练数据，硬合成一个模型
        # 会互相干扰、两边精度都掉。
        self.extra = YOLO(extra_weights) if extra_weights else None
        self.extra_names = self.extra.names if self.extra else {}
        self.extra_conf = float(extra_conf if extra_conf is not None else conf)
        # 香烟模型降频：它要跑 5 次前向（整图 + 2x2 切片），逐帧跑会把
        # 检测帧率从 16 拖到 8.7。而香烟是慢过程（点烟、抽烟都在秒级），
        # 每 3 帧看一次完全够，中间复用上次结果。
        # 香烟模型**不再降频**（原来是 2）。
        # 降频的代价不是"慢"，而是**画框闪**：一半的帧根本没有香烟检测结果，
        # 画面上看就是框闪一下就没了。而用户要的是"先框出来、再判帧数"——
        # 框必须每帧都在，帧数门槛只用来决定"要不要报"，不该影响"画不画"。
        self.extra_every = 1
        # 切片频率：整图每帧跑（响应快），切片每 N 帧补一次（小目标召回）
        # 香烟改成每帧跑之后，切片通道让一让：3 -> 4。
        # 切片是主模型的补充通道（放大找小目标），4 帧一次够用，
        # 省下的算力给香烟模型，让它能每帧出框。
        # 切片 2x2 -> 3x3，每帧的放大倍数从 1.5x 提到 2.25x。
        #
        # 关键认知：**切片不等于放大**。2x2 切出来每块 640x360，
        # 送进 imgsz=960 只放大 1.5 倍 —— 一支 26x43 的香烟在切片里
        # 仍然只有 39x65，模型照样看不清。3x3 每块约 426x240，
        # 放大到 960 是 2.25 倍，香烟变成 58x97，这才进入"能认"的量级。
        # ⚠️ 切片参数要和**训练时的尺度**对齐，否则模型学的是"16x26 的小白条"，
        # 推理时看到的是"111x183 的大白条"，学到的特征用不上。
        #
        # 训练用 imgsz=960（整图），香烟在整图里约 20x32。
        # 因此推理这边用 2x2 切片 + imgsz 960：
        #     640x360 的块 -> 960，放大 1.5x -> 香烟 39x62
        # 和训练的 20x32 在同一量级。原来 3x3+1024 会放大 4.27x（111x183），
        # 与训练尺度差 5 倍以上，是"训练完反而更差"的隐藏原因。
        # 远距离检测：切片从 2x2 + 960 提到 3x3 + 1152。
        # 1280x720 切成 3x3 每块约 426x240，送到 1152 是 2.7 倍放大 ——
        # 一个 10x15 像素的远处小火苗会变成 27x40，进入"能认"的量级。
        # 香烟模型关掉后腾出的算力正好给这里。
        self.tile_imgsz = 960
        self.slice_every = 3
        # ⚠️ 这些缓存是**实例级状态**，而 Detector 在 run_multi 里是**多路共享**的
        # （一块 GPU 只建一份模型）。多路轮流调 detect_auto 时，A 路的切片缓存
        # 会被 B 路下一帧直接复用 —— 表现为 B 路上出现 A 路坐标的框、框乱跳。
        # 改成按 cache_key（cam_id）隔离的字典，每路各自记自己的切片/香烟缓存。
        self._slice_n: dict = {}
        self._slice_cache: dict = {}
        self._extra_n: dict = {}
        self._extra_cache: dict = {}
        # 模型开关（前端可自主切换）：主模型=火/烟，香烟模型=cigarette
        self.enable_main = True
        self.enable_extra = True

    def set_device(self, device: str | int) -> None:
        """切换推理设备：0/'' = GPU(cuda)，'cpu' = CPU。前端 CPU/GPU 开关用。"""
        self.device = device

    def _enhance(self, image: np.ndarray) -> np.ndarray:
        """光线自适应：CLAHE 在 LAB 亮度通道做局部直方图均衡。

        作用：压反光/过曝、提暗部，让模型在强光/逆光下也能看清目标。
        火光本身是高亮，CLAHE 只均衡局部对比、不会把它抹掉（clipLimit 保守）。
        """
        try:
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            l = clahe.apply(l)
            return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)
        except Exception:                    # noqa: BLE001
            return image

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

    def _predict_extra(self, image: np.ndarray,
                       crops: list[np.ndarray] | None = None,
                       offs: list[tuple[int, int]] | None = None,
                       cache_key: str = "default") -> list[Detection]:
        """第二个模型（香烟）。整图 + 复用主模型的切片一起跑。

        为什么香烟必须走切片
        --------------------
        香烟是这三类里最小的目标 —— 一支烟在 960x540 的画面里常常只有
        十几像素宽，整图推理时和噪点没区别，灵敏度自然上不去。
        主模型已经切好 2x2 的图了，直接拿来用，不用再切一遍，
        额外成本只是多几次前向，不额外增加裁剪开销。
        """
        if self.extra is None:
            return []
        self._extra_n[cache_key] = self._extra_n.get(cache_key, 0) + 1
        if self._extra_n[cache_key] % self.extra_every != 0:
            return list(self._extra_cache.get(cache_key, []))   # 用上次的结果，不重复推理
        out = []

        def _run(img, offset=(0, 0)):
            try:
                r = self.extra.predict(
                    img, imgsz=self.imgsz, conf=self.extra_conf,
                    iou=self.iou, device=self.device, verbose=False,
                )[0]
            except Exception:                 # noqa: BLE001
                return
            ox, oy = offset
            for b in r.boxes:
                x1, y1, x2, y2 = b.xyxy[0].tolist()
                cid = int(b.cls)
                out.append(Detection(cid, self.extra_names.get(cid, "cigarette"),
                                     float(b.conf),
                                     (x1 + ox, y1 + oy, x2 + ox, y2 + oy)))

        _run(image)
        # 不做切片：实测香烟模型整图 9.8ms、4 片切片 40ms。
        # 用户要求"检测帧率拉到最高"，这里是最划算的一刀 ——
        # 省掉 40ms/帧（总耗时的 40%），代价是远处极小的香烟可能漏。
        # 靠 cigarette_conf=0.03 的极低门限 + 每帧都检（every=1）来补偿。
        self._extra_cache[cache_key] = out
        return out

    def _predict_extra_batch(self, images: list[np.ndarray],
                             cache_keys: list[str]) -> list[list[Detection]]:
        """batch 香烟检测多路整图（多摄像头同时）。带降频 + 缓存，逻辑同单路版。"""
        n = len(images)
        if self.extra is None:
            return [[] for _ in range(n)]
        results: list = []
        need_idx: list[int] = []
        for i, ck in enumerate(cache_keys):
            self._extra_n[ck] = self._extra_n.get(ck, 0) + 1
            if self._extra_n[ck] % self.extra_every != 0:
                results.append(list(self._extra_cache.get(ck, [])))
            else:
                results.append(None)          # 标记需要推理
                need_idx.append(i)
        if need_idx:
            imgs = [images[i] for i in need_idx]
            rs = self.extra.predict(imgs, imgsz=self.imgsz, conf=self.extra_conf,
                                    iou=self.iou, device=self.device, verbose=False)
            # 旋转 90° 再检一次（补横向香烟）：模型对横向香烟召回低，旋转后横向变纵向。
            rot_imgs = [cv2.rotate(im, cv2.ROTATE_90_CLOCKWISE) for im in imgs]
            rs_rot = self.extra.predict(rot_imgs, imgsz=self.imgsz, conf=self.extra_conf,
                                        iou=self.iou, device=self.device, verbose=False)
            for j, i in enumerate(need_idx):
                out = []
                for b in rs[j].boxes:
                    x1, y1, x2, y2 = b.xyxy[0].tolist()
                    cid = int(b.cls)
                    out.append(Detection(cid, self.extra_names.get(cid, "cigarette"),
                                         float(b.conf), (x1, y1, x2, y2)))
                # 旋转检测结果坐标转回原图：90°顺时针 原图(x,y)->旋转图(h-1-y,x)
                # 反推：旋转图(x',y') -> 原图(y', h-1-x')
                h, w = imgs[j].shape[:2]
                for b in rs_rot[j].boxes:
                    x1, y1, x2, y2 = b.xyxy[0].tolist()
                    cid = int(b.cls)
                    out.append(Detection(cid, self.extra_names.get(cid, "cigarette"),
                                         float(b.conf),
                                         (y1, h - 1 - x2, y2, h - 1 - x1)))
                self._extra_cache[cache_keys[i]] = out
                results[i] = out
        return results

    def _predict_batch(self, imgs: list[np.ndarray],
                       offsets: list[tuple[int, int]]) -> list[Detection]:
        """一次前向跑多张图。分级检测要补切好几块，逐块跑很亏 ——
        实测 batch=4 合计 23.3ms，逐块跑是 4×11.9ms（1.9 倍差距）。"""
        if not imgs:
            return []
        if len(imgs) == 1:
            return self._predict(imgs[0], offsets[0])
        # 切片块本身小，用更高的 imgsz 才能把里面的小目标放大到"能认"的尺寸。
        # 整图仍用 self.imgsz（960）保速度，切片单独用 tile_imgsz（1280）。
        rs = self.model.predict(
            list(imgs), imgsz=getattr(self, "tile_imgsz", self.imgsz),
            conf=self.conf, iou=self.iou,
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
        tile_grid: int = 3,
        overlap: float = 0.3,
        keep_full_frame: bool = True,
        batched: bool = False,
        cache_key: str = "default",
    ) -> list[Detection]:
        """检测一帧。

        sliced=True 时：
          · keep_full_frame=True（默认）先跑一遍整图，兜住横跨多个切片的大烟团
          · 再跑 grid×grid 个重叠切片，各自放大到 imgsz 后检测
          · 所有框映射回原图坐标后统一 NMS

        batched=False（默认）逐张跑。**不要随手改成 True**：批量前向时
        ultralytics 只在单图用最省 padding 的 `auto=True`，批量统一补成方形，
        输入几何略有不同，检测结果会漂（实测 fire 召回 68.5% → 67.8%，
        fire 场景还会偶发多出一条烟告警）。而批量只省 1.08 倍（56.8→52.6ms），
        不值。真正需要批量的是 detect_auto 的慢路径，那里是显式传参的。
        """
        h, w = image.shape[:2]

        if not sliced:
            return nms(self._predict(image), 0.25)

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
            return nms(self._predict_batch(crops, offs), 0.25)

        dets: list[Detection] = []
        for crop, off in zip(crops, offs):
            dets.extend(self._predict(crop, offset=off))
        return nms(dets + self._predict_extra(image, cache_key=cache_key), 0.25)

    def detect_auto(
        self,
        image: np.ndarray,
        trigger_conf: float = 0.03,
        tile_grid: int = 2,
        overlap: float = 0.25,
        cache_key: str = "default",
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
        image = self._enhance(image)          # 光线自适应（压反光/提暗部）
        h, w = image.shape[:2]
        coarse = self._predict(image, conf=trigger_conf) if self.enable_main else []

        # ⚠️ **不再因为粗筛为空就跳过切片** —— 这是个死循环。
        #
        # 原来的逻辑：整图粗筛没东西 -> 不切片 -> 返回空。
        # 但实测（同一帧、同一模型）：
        #     整图  1 块   检出 0 个
        #     2x2   4 块   检出 6 个（fire 5, smoke 1）
        #     3x3   9 块   检出 13 个（smoke 8, fire 5）
        # **整图对香烟和小火苗完全无效**（目标只有 20~40px，缩到 960 后
        # 和噪点没区别）。而恰恰是这种帧，粗筛给的是"0 个"而不是"低分"，
        # 于是永远走快路径、永远不切片、永远检不出 ——
        # 这就是用户说的"很多细节 yolo 没有去主动识别"。
        #
        # 现在：粗筛只用来提供"整图那一路的检出"（有火/大目标时有用），
        # **不再作为切片的前置条件**。切片该跑就跑。
        if coarse is None:
            coarse = []

        crops: list[np.ndarray] = []
        offs: list[tuple[int, int]] = []
        if self.enable_main:
            for x0, y0, x1, y1 in self._tiles(h, w, tile_grid, overlap):
                crop = image[y0:y1, x0:x1]
                if crop.size == 0:
                    continue
                crops.append(crop)
                offs.append((x0, y0))
        # 粗筛那一帧就是"整图"那一路，别重复跑
        kept = [d for d in coarse if d.conf >= self.conf]

        # ── 切片降频：整图每帧都跑，切片隔几帧补一次 ──────────────
        # 实测耗时：整图 11ms、2x2 切片 40ms。用户要求"识别频率大幅拉高
        # 且切片必须保留"，唯一可行的办法就是拆开频率：
        #   · 整图每帧跑      -> 响应快，大目标不漏
        #   · 切片每 slice_every 帧跑 -> 小目标仍然有专门的放大通道
        # 平均耗时从 51ms 降到 11 + 40/N，N=3 时约 24ms，检测频率翻倍。
        self._slice_n[cache_key] = self._slice_n.get(cache_key, 0) + 1
        dets = list(kept)
        if self.enable_main:
            if self._slice_n[cache_key] % int(self.slice_every) == 0:
                dets += self._predict_batch(crops, offs)
                self._slice_cache[cache_key] = [d for d in dets if d not in kept]
            else:
                # 不切片的帧沿用上一次的切片结果（框位置由追踪器平滑，不会跳）
                dets += list(self._slice_cache.get(cache_key, []))

        # 香烟模型：整图 + 复用切片（它自己有降频）
        if self.enable_extra:
            dets += self._predict_extra(image, crops, offs, cache_key=cache_key)

        # ⚠️ 合并后必须做**更严格的**二级 NMS（0.25，而不是 0.5）。
        #
        # 原因：整图推理和 2x2 切片推理是**分开跑的**，各自内部已经 NMS 过，
        # 但同一团火会在两个通道里各出一个框 —— 它们的 IOU 常在 0.3~0.5 之间，
        # 用 0.5 的门槛压不掉，于是就出现"同一个打火机上两个红框"。
        # 0.25 能把这些跨通道的重复彻底合并，而真正的两个独立目标
        # （比如手和烟）IOU 远低于 0.25，不会被误合并。
        return nms(dets, 0.25)

    def detect_auto_batch(
        self,
        images: list[np.ndarray],
        cache_keys: list[str],
        trigger_conf: float = 0.03,
        tile_grid: int = 2,
        overlap: float = 0.25,
    ) -> list[list[Detection]]:
        """多摄像头同时检测：batch 推理多路帧，一次前向喂所有路。

        和 detect_auto 的区别：整图、香烟都合成 batch（GPU 一次前向跑多路），
        切片仍按路独立（每路自己的 crops 合成 batch）。这样多路不再"交替"，
        而是同一时刻一起检测。
        """
        n = len(images)
        images = [self._enhance(im) for im in images]

        # ── 1. 整图 batch（主模型粗筛）──
        coarses: list[list] = [[] for _ in range(n)]
        if self.enable_main:
            rs = self.model.predict(
                images, imgsz=self.imgsz, conf=trigger_conf, iou=self.iou,
                device=self.device, classes=self.classes, verbose=False)
            for i, r in enumerate(rs):
                coarses[i] = [
                    Detection(int(b.cls), self.names[int(b.cls)], float(b.conf),
                              tuple(b.xyxy[0].tolist()))
                    for b in r.boxes
                ]

        # ── 2. 切片 + 合并（每路独立，降频逻辑同 detect_auto）──
        results: list[list] = []
        for i in range(n):
            dets = [d for d in coarses[i] if d.conf >= self.conf]
            crops: list[np.ndarray] = []
            offs: list[tuple[int, int]] = []
            if self.enable_main:
                h, w = images[i].shape[:2]
                for x0, y0, x1, y1 in self._tiles(h, w, tile_grid, overlap):
                    crop = images[i][y0:y1, x0:x1]
                    if crop.size == 0:
                        continue
                    crops.append(crop)
                    offs.append((x0, y0))
            ck = cache_keys[i]
            self._slice_n[ck] = self._slice_n.get(ck, 0) + 1
            if self.enable_main:
                if self._slice_n[ck] % int(self.slice_every) == 0:
                    dets += self._predict_batch(crops, offs)
                    self._slice_cache[ck] = [d for d in dets if d not in coarses[i]]
                else:
                    dets += list(self._slice_cache.get(ck, []))
            results.append(dets)

        # ── 3. 香烟 batch（多路整图）──
        if self.enable_extra:
            extra_per_cam = self._predict_extra_batch(images, cache_keys)
            for i in range(n):
                results[i] += extra_per_cam[i]

        # ── 4. 每路二级 NMS ──
        return [nms(r, 0.25) for r in results]

