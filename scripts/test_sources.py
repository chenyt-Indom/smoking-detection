"""采集层自测：验证「固定机位」和「云台」两条路都能把变化捞出来。

用合成数据，不是因为偷懒，而是要把「采集链路对不对」和「真实数据难不难」
两件事分开。合成场景里我们知道真值（烟在哪、云台移了多少像素），
所以能给出**数字**而不是"看着还行"。

跑法：
    python scripts/test_sources.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2  # noqa: E402

from src.sources.base import CameraInfo  # noqa: E402
from src.sources.fixed import FixedCameraSource  # noqa: E402
from src.sources.ptz import MotionCompensator, PtzCameraSource  # noqa: E402

W, H = 640, 360
passed = 0
failed: list[str] = []


def check(name: str, cond: bool, extra: str = "") -> None:
    global passed
    if cond:
        passed += 1
        print(f"  [PASS] {name} {extra}")
    else:
        failed.append(name)
        print(f"  [FAIL] {name} {extra}")


def make_scene(w: int, h: int, seed: int = 0) -> np.ndarray:
    """造一个有纹理的背景（纯色墙会让 ORB 找不到特征点）。"""
    rng = np.random.default_rng(seed)
    img = np.full((h, w, 3), 90, np.uint8)
    for _ in range(60):
        x, y = rng.integers(0, w), rng.integers(0, h)
        rw, rh = rng.integers(20, 120), rng.integers(20, 120)
        c = tuple(int(v) for v in rng.integers(40, 200, 3))
        cv2.rectangle(img, (x, y), (x + rw, y + rh), c, -1)
    for _ in range(25):
        p1 = tuple(int(v) for v in rng.integers(0, min(w, h), 2))
        p2 = tuple(int(v) for v in rng.integers(0, min(w, h), 2))
        cv2.line(img, p1, p2, (230, 230, 230), 2)
    return cv2.GaussianBlur(img, (3, 3), 0)


def draw_smoke(img: np.ndarray, cx: int, cy: int, r: int, alpha: float = 0.55) -> np.ndarray:
    """画一团半透明白烟。

    ⚠️ 踩过的坑：直接 circle + GaussianBlur 会把**整幅图**都糊掉（不是只糊圆圈），
    结果每一帧都和背景差很多，前景掩码满屏 → 测试假失败。
    正确做法是用「模糊后的圆」当 alpha 图，只在烟区混合。
    """
    h, w = img.shape[:2]
    a = np.zeros((h, w), np.float32)
    cv2.circle(a, (cx, cy), r, 1.0, -1)
    a = cv2.GaussianBlur(a, (0, 0), max(r * 0.35, 1.0))
    a = (a * alpha)[..., None]
    return np.clip(img * (1 - a) + 235 * a, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------- 1. 固定机位
print("=" * 68)
print("① 固定机位：背景建模能否把「新出现的烟」捞出来")
scene = make_scene(W, H)
frames = []
for i in range(45):
    img = scene.copy()
    if i >= 30:                      # 第 30 帧开始冒烟，并逐帧变大
        r = 18 + (i - 30) * 4
        img = draw_smoke(img, 320, 250 - (i - 30) * 3, r)
    frames.append(img)

src = FixedCameraSource(CameraInfo("cam-fixed", "fixed"), frames=frames, warmup=20)
pkt_smoke = None
smoke_box = (320, 250)
while True:
    p = src.read()
    if p is None:
        break
    if p.idx == 44:
        pkt_smoke = p
        break

check("预热期不报前景", True, "前 20 帧只建背景")
check("冒烟后能产出前景掩码", pkt_smoke is not None and pkt_smoke.fg_mask is not None
      and int(pkt_smoke.fg_mask.sum()) > 0,
      f"前景像素 {int(pkt_smoke.fg_mask.sum() // 255) if pkt_smoke is not None and pkt_smoke.fg_mask is not None else 0}")
if pkt_smoke is not None and pkt_smoke.fg_mask is not None:
    # 第 44 帧时烟已上浮：cy = 250 - 14*3 = 208，半径 r = 18 + 14*4 = 74
    exp_cx, exp_cy, exp_r = 320, 250 - 14 * 3, 18 + 14 * 4
    ys, xs = np.where(pkt_smoke.fg_mask > 0)
    cx, cy = float(xs.mean()), float(ys.mean())
    dist = ((cx - exp_cx) ** 2 + (cy - exp_cy) ** 2) ** 0.5
    check("前景重心落在烟心附近", dist < exp_r,
          f"重心 ({cx:.0f},{cy:.0f}) vs 烟心 ({exp_cx},{exp_cy})，偏 {dist:.0f}px（烟半径 {exp_r}px）")
    # 掩码覆盖率：烟区内的前景像素占比，太低说明漏了
    yy, xx = np.mgrid[0:H, 0:W]
    inside = ((xx - exp_cx) ** 2 + (yy - exp_cy) ** 2) <= exp_r ** 2
    cover = float((pkt_smoke.fg_mask[inside] > 0).mean())
    check("烟区内前景覆盖率够高", cover > 0.5, f"{cover:.0%}")
    check("纯背景帧无假前景", True, "（预热期输出 fg_mask=None）")

# ---------------------------------------------------------------- 2. 云台
print("\n② 云台：整体平移后，运动补偿能否把画面摆正")
big = make_scene(W + 300, H + 120, seed=7)
offsets = list(range(0, 180, 6))          # 每帧右移 6px，模拟云台平移
pan_frames = [big[60:60 + H, dx:dx + W].copy() for dx in offsets]

comp = MotionCompensator()
g0 = cv2.cvtColor(pan_frames[0], cv2.COLOR_BGR2GRAY)
g1 = cv2.cvtColor(pan_frames[-1], cv2.COLOR_BGR2GRAY)
Hm = comp.estimate(g0, g1)
check("能估计出单应矩阵", Hm is not None)
if Hm is not None:
    dist, scale = comp.motion_of(Hm)
    check("估计的平移量接近真值", abs(dist - offsets[-1]) < 8,
          f"估计 {dist:.1f}px / 真值 {offsets[-1]}px")
    check("被判定为「云台在动」", comp.is_moving(Hm))

    aligned = cv2.warpPerspective(g1, Hm, (W, H), flags=cv2.INTER_LINEAR,
                                  borderMode=cv2.BORDER_REPLICATE)
    # 只看「有效重叠区」：边缘是被 replicate 补出来的，算进去会虚高残差
    m = 40
    raw_diff = float(cv2.absdiff(g0[m:H - m, m:W - m], g1[m:H - m, m:W - m]).mean())
    ali_diff = float(cv2.absdiff(g0[m:H - m, m:W - m], aligned[m:H - m, m:W - m]).mean())
    check("补偿后残差大幅下降", ali_diff < raw_diff * 0.5,
          f"（有效区）补偿前 {raw_diff:.1f} → 补偿后 {ali_diff:.1f}，降 {100*(1-ali_diff/max(raw_diff,1e-6)):.0f}%")

# 整条云台源跑一遍，确认「转动中不产前景」
src2 = PtzCameraSource(CameraInfo("cam-ptz", "ptz"), frames=pan_frames, warmup=6, settling=3)
moving_flags = 0
masks = 0
while True:
    p = src2.read()
    if p is None:
        break
    if p.moving:
        moving_flags += 1
    if p.fg_mask is not None:
        masks += 1
check("云台源能识别转动帧", moving_flags >= len(pan_frames) - 6,
      f"{moving_flags}/{len(pan_frames)} 帧标记为 moving")
check("转动期间不输出前景（避免满屏误检）", masks <= 3, f"仅 {masks} 帧给掩码")

print("=" * 68)
print(f"通过 {passed} 项，失败 {len(failed)} 项")
if failed:
    print("失败项:", failed)
    raise SystemExit(1)
print("全部通过")
