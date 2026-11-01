"""光敏引擎 —— 只要在发光，就框出来，不管 YOLO 认不认得。

为什么需要它
------------
YOLO 只认识训练过的样子。远处的小火苗、被遮挡一半的火焰、没见过的打火机，
模型可能完全给不出框 —— 而"那里有光"这件事，用亮度阈值就能百分之百抓到。

所以这条通道的定位很明确：**不做识别，只做发现**。
它找出画面里所有明显发亮的区域，用一个灰框标出来，并给出主色调判断。
灰框的意思是"这里有一团光，但我不知道它是什么" —— 后续仍由 LLM 裁决。

为什么不直接用 YOLO 当检测器
---------------------------
两者是互补的，不是替代关系：
  · YOLO：认识形状和语义，但会漏小目标、漏没见过的形态
  · 光斑：不漏任何亮点，但分不清火和灯泡
合起来用，召回率接近"画面里有没有发光物"的物理上限。

颜色判别怎么做的
----------------
在 HSV 空间里统计亮斑内部的主色调和饱和度：
  · 橙红 + 高饱和 → 像火焰（火的色温低、饱和度高）
  · 白 + 低饱和   → 像灯泡/手电筒（白光三通道接近，饱和度低）
  · 蓝/青         → 像屏幕、LED 指示灯、冷色调补光
这只是**先验提示**，写进 context 交给 LLM，不作为最终结论。
"""
from __future__ import annotations

import cv2
import numpy as np

# 颜色分类用的色相区间（OpenCV 的 H 是 0~180）
_HUE_BANDS = [
    ("橙红", 0, 22),      # 火焰、白炽灯偏暖
    ("黄绿", 22, 45),     # 黄色补光、部分 LED
    ("青绿", 45, 95),
    ("蓝紫", 95, 145),    # 屏幕、LED 指示灯、手电筒冷白
    ("品红", 145, 180),   # 与 0 接壤，同样偏暖
]


def _classify_color(hsv_patch: np.ndarray) -> tuple[str, float, float]:
    """返回 (颜色名, 主色相, 平均饱和度)。饱和度高说明颜色纯，像火或彩色光源。"""
    h = hsv_patch[:, :, 0].astype(np.float32)
    s = hsv_patch[:, :, 1].astype(np.float32)
    v = hsv_patch[:, :, 2].astype(np.float32)
    # 只统计足够亮的像素，否则暗部会把色相拉偏
    m = v > max(60.0, float(v.mean()) * 0.6)
    if not np.any(m):
        return "未知", -1.0, 0.0
    hh, ss = h[m], s[m]
    # 圆形均值：色相是环形的，直接取算术平均会在 0/180 交界处出错
    ang = hh * (np.pi / 90.0)          # H 0~180 -> 0~2pi
    mean_deg = float(np.arctan2(np.sin(ang).mean(), np.cos(ang).mean()))
    mean_h = (mean_deg * (90.0 / np.pi)) % 180.0
    mean_s = float(ss.mean())
    if mean_s < 40:
        # 饱和度低 = 三通道接近 = 白光
        return "白光", mean_h, mean_s
    for name, lo, hi in _HUE_BANDS:
        if lo <= mean_h < hi:
            return name, mean_h, mean_s
    return "白光", mean_h, mean_s


def detect(image: np.ndarray, *, v_min: int = 205, min_area: int = 12,
           max_area_pct: float = 0.25, merge_gap: int = 6):
    """找出画面里所有明显发亮的区域。

    参数
    ----
    v_min       : 亮度阈值（0~255）。205 大约对应"明显比周围亮"的白炽级亮度。
                  调低会框住更多反光，调高只留强光源。
    min_area    : 最小连通域面积（像素）。12 大约是一个 3x4 的点 —— 再小基本是噪点。
    max_area_pct: 最大面积占画面比例。超过这个通常是整片过曝/大面积反光，
                  框它没有信息量。
    merge_gap   : 形态学闭运算的核大小，把断裂的亮斑连起来
                  （火焰常被压缩成几块，不连起来会框出一堆碎块）。

    返回 list[dict]，每项含 box / color / hue / sat / area。
    """
    if image is None or image.size == 0:
        return []
    h, w = image.shape[:2]
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    v = hsv[:, :, 2]

    # 亮度阈值。用 Otsu 自适应 + 固定下限取较大者：
    # 纯固定阈值在昏暗场景会一个都框不出来，纯 Otsu 在白天会框出一堆墙面。
    _, otsu = cv2.threshold(v, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    fixed = cv2.inRange(v, v_min, 255)
    mask = cv2.bitwise_and(otsu, fixed)

    if merge_gap > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (merge_gap, merge_gap))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,
                                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))

    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    out = []
    max_area = int(h * w * max_area_pct)
    for i in range(1, n):
        x, y, bw, bh, area = (int(v) for v in stats[i])
        if area < min_area or area > max_area:
            continue
        patch = hsv[y:y + bh, x:x + bw]
        color, hue, sat = _classify_color(patch)
        out.append({
            "box": (float(x), float(y), float(x + bw - 1), float(y + bh - 1)),
            "color": color, "hue": round(hue, 1), "sat": round(sat, 1),
            "area": area,
        })
    # 面积大的排前面：越大的光斑越可能是主线信息
    out.sort(key=lambda d: -d["area"])
    return out


def describe(spots, limit: int = 4) -> str:
    """把光斑结果压成一句话，塞进送审上下文。"""
    if not spots:
        return ""
    parts = [f"{s['color']}({s['area']}px)" for s in spots[:limit]]
    more = f" 等 {len(spots)} 处" if len(spots) > limit else ""
    return "画面另检出亮光：" + "、".join(parts) + more
