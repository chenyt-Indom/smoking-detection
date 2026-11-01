"""细长物候选检测 —— 只负责"找出像香烟的东西"，判定交给 LLM。

为什么需要它
------------
香烟模型（v28）是拿标准香烟数据训的：带过滤嘴、斜夹在指间、有烟头。
用户现场那根是**通体白色、一端尖锐、竖直举着**的形态，模型给不出框
（实测全图最高 0.10，手部 ROI 单独测只有 0.01~0.02）。

但它有很确定的**几何特征**：细长、浅色、有明确的边界线。
这些东西用传统 CV 抓得住，而且是**纯检索**——抓到之后送给 LLM 裁决，
由它看截图判断"这到底是不是香烟"。所以这不是"手工判定"，
是"手工找候选 + 模型判定"，两者分工明确。

为什么不做手部约束
------------------
用户明确要求："去掉手部啊，只弄香烟检测啊，找到类似于香烟的东西，
然后上传 LLM 进行裁决。"
而且实测 hand_v12 在当前光照下也检出 0 只手 —— 拿它当闸门只会漏得更多。

什么样的算候选
--------------
  · 长宽比 >= 3         细长
  · 长边 >= 25px        太小没信息量
  · 面积占比 < 3%       太粗的不是香烟
  · 亮度/颜色：浅色或高亮（香烟是白的，燃烧端是橙红）

输出统一是 [(box, tag), ...]，tag 只描述"看起来像什么"，
不构成任何结论。
"""
from __future__ import annotations

import cv2
import numpy as np


def _elongated_candidates(mask: np.ndarray, min_len: int, max_area_pct: float,
                          aspect_min: float):
    """从二值掩码里找细长连通域。"""
    h, w = mask.shape[:2]
    n, labels, stats, _cent = cv2.connectedComponentsWithStats(mask, 8)
    out = []
    max_area = h * w * max_area_pct
    for i in range(1, n):
        x, y, bw, bh, area = (int(v) for v in stats[i])
        if area < 30 or area > max_area:
            continue
        # ⚠️ 长宽比必须用**最小外接矩形**算，不能用 bbox。
        # 香烟通常是**斜着拿的**：一根 20px 粗、160px 长的斜烟，
        # 它的 bbox 可能是 120x110 —— 比值只有 1.1，直接被筛掉。
        # 用户现场那根就是这样：bbox 143x88（比值 1.6），
        # 但用最小外接矩形量出来是 168x20（比值 8.4），明显是细长物。
        comp = (labels[y:y + bh, x:x + bw] == i).astype(np.uint8)
        pts = cv2.findNonZero(comp)
        if pts is None:
            continue
        (_, _), (rw, rh), _ = cv2.minAreaRect(pts)
        long_side, short_side = max(rw, rh), max(1.0, min(rw, rh))
        if long_side < min_len:
            continue
        if long_side / short_side < aspect_min:
            continue
        out.append((float(x), float(y), float(x + bw - 1), float(y + bh - 1)))
    return out


def detect(image: np.ndarray, *, min_len: int = 25, aspect_min: float = 3.0,
           max_area_pct: float = 0.03, max_keep: int = 6):
    """找出所有"像香烟"的细长物候选。

    返回 [(box, tag), ...]，按"像不像香烟"粗略排序（只是排序，不是判定）。
    调用方应该把 box 全部送 LLM，由它看截图决定。
    """
    if image is None or image.size == 0:
        return []
    h, w = image.shape[:2]
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    v = hsv[:, :, 2]
    s = hsv[:, :, 1]

    found: list[tuple[tuple, str]] = []

    # ① 白色细长条：低饱和 + 高亮度。香烟本体就是这种。
    # 实测用户现场那根烟：V≈140、S≈80~100 —— 不是纯白，是"灰白"。
    # 原来的 S<=70 直接把它筛掉了。放宽到 S<=135、V>=105。
    white = cv2.inRange(hsv, (0, 0, 105), (180, 135, 255))
    # 核要小、迭代要少：实测 3x3 迭代 2 次会把细长的香烟和旁边的
    # 手臂/衣服一起聚类成一个大块，长宽比立刻降到 3 以下，反而被筛掉。
    # 2x2 迭代 1 次足够把被 JPEG 压缩断开的细条接上。
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    white = cv2.morphologyEx(white, cv2.MORPH_CLOSE, k, iterations=1)
    for b in _elongated_candidates(white, min_len, max_area_pct, aspect_min):
        found.append((b, "白色细长"))

    # ② 橙红细长条：燃烧端，或者整根偏黄的烟
    warm = cv2.inRange(hsv, (0, 60, 120), (30, 255, 255))
    warm = cv2.morphologyEx(warm, cv2.MORPH_CLOSE, k, iterations=2)
    for b in _elongated_candidates(warm, int(min_len * 0.7), max_area_pct, aspect_min):
        found.append((b, "橙红细长"))

    # ③ 褐色细长条（烟蒂、深色烟身）
    brown = cv2.inRange(hsv, (5, 40, 50), (25, 200, 170))
    brown = cv2.morphologyEx(brown, cv2.MORPH_CLOSE, k, iterations=2)
    for b in _elongated_candidates(brown, min_len, max_area_pct, aspect_min):
        found.append((b, "褐色细长"))

    if not found:
        return []

    # 去重：同一个东西可能同时命中白色和橙红（烟身白、端头红）
    found.sort(key=lambda t: -(t[0][2] - t[0][0]) * (t[0][3] - t[0][1]))
    kept: list[tuple[tuple, str]] = []

    def _iou(a, b) -> float:
        ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
        iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
        inter = ix * iy
        ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
        return inter / ua if ua > 0 else 0.0

    for box, tag in found:
        if any(_iou(box, kb) > 0.4 for kb, _ in kept):
            continue
        kept.append((box, tag))
        if len(kept) >= max_keep:
            break
    return kept
