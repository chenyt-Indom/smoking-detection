"""在**真实连续帧**上验证「火焰闪烁闸」：真火不该被拦，恒定光源该被拦。

为什么单独一个脚本
------------------
`test_pipeline.py` 用的是合成场景，里面贴上去的"火"是一张静止贴图 ——
物理上就不该闪，闸判它 static 完全正确。所以在那边测闸没有意义
（只能把"合成数据不真实"显示成"测试失败"）。闸必须在真视频帧上验。

数据源
------
`E:/datasets/fire_smoke_v5/test/images/A_烟雾_火_fire_CV*.jpg`
共 506 张**逐帧连续**的纯火视频帧（同一段视频按帧号切出来的）。
连续帧是关键：跳跃采样测不出"帧间抖动"这件事本身。

判据
----
框内"变化像素占比"（灰度差 > 8），已扣掉整幅同步变化（相机晃动/云影/自动曝光）。
  · 真火（960x540 / 113 帧 / 664 条轨迹）：
      最小 0.200 · P10 0.212 · 中位 0.335 · 最大 0.599
  · 恒定光源：接近 0
阈值 `flicker_static=0.05` → 真火最弱一条下方仍有 4 倍余量，
同时恒亮光源照样拦得住。分辨率变了这个值会漂（854x480 那段最低到 0.148），
所以闸只做"降级 + LLM 复核"，绝不做"排除"。

用法
----
    python scripts/verify_flicker_gate.py
    python scripts/verify_flicker_gate.py --sliced 0     # 关切片，快一倍
"""
from __future__ import annotations

import argparse
import glob
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import cv2

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.infer import Detector                                    # noqa: E402
from src.tracker import IoUTracker                                # noqa: E402
from src.temporal import DEFAULTS, extract_features, judge_fire   # noqa: E402

WEIGHTS = ROOT / "models" / "fire_smoke_v5_26s960_best.pt"
DATA = Path("E:/datasets/fire_smoke_v5/test")
FPS = 25.0


def imread_u(p: str):
    """中文路径也能读。cv2.imread 在非 ASCII 路径上直接返回 None。"""
    return cv2.imdecode(np.fromfile(p, dtype=np.uint8), cv2.IMREAD_COLOR)


def pick_continuous(pattern: str, limit: int):
    """按帧号排序取同一段视频的连续帧（尺寸对齐，否则背景建模会炸）。

    边读边筛、够数就停 —— 目录里共有 500+ 张，全读进内存要 600MB，
    没必要为了取前 120 张把剩下的也解开。
    """
    files = sorted(glob.glob(pattern),
                   key=lambda p: int(re.search(r"CV(\d+)\.jpg$", p).group(1)))
    main: tuple | None = None
    out = []
    for p in files:
        im = imread_u(p)
        if im is None:
            continue
        if main is None:
            main = im.shape
        if im.shape != main:      # 混进别的视频的帧就丢掉，尺寸不一会炸背景建模
            continue
        out.append(im)
        if len(out) >= limit:
            break
    if not out:
        raise SystemExit(f"没读到图：{pattern}")
    return out, main


def collect(det, frames, wanted_cls: str, sliced: bool):
    """跑一遍追踪，收集每条火轨迹的 flicker 值和判定。"""
    trk = IoUTracker()
    ts = 0.0
    rows = []
    for f in frames:
        ts += 1.0 / FPS
        for tk in trk.update(det.detect(f, sliced=sliced, tile_grid=2), f, ts):
            if tk.cls_name != wanted_cls or len(tk.samples) < 6:
                continue
            feat = extract_features(tk.samples)
            if feat is None:
                continue
            v = judge_fire(feat, area_px=tk.samples[-1].area)
            rows.append((tk.tid, round(tk.samples[-1].area), feat.flicker, v.label))
    return rows


def synthetic_light(n: int, size: tuple[int, int]):
    """合成一个"完全恒定"的亮斑序列（只有传感器噪声），当对照基线。"""
    h, w = size
    rng = np.random.default_rng(7)
    base = np.full((h, w, 3), 40, np.uint8)
    base[:] = (46, 44, 42)
    out = []
    for _ in range(n):
        f = base.copy()
        cv2.ellipse(f, (w // 2, h // 2), (max(8, w // 12), max(8, h // 12)),
                    0, 0, 360, (238, 242, 248), -1)
        out.append(np.clip(f.astype(np.float32) + rng.normal(0, 2.5, f.shape),
                           0, 255).astype(np.uint8))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=120, help="取多少张连续帧")
    ap.add_argument("--sliced", type=int, default=1, help="1 开切片（更准，更慢）")
    args = ap.parse_args()

    det = Detector(str(WEIGHTS), imgsz=896, conf=0.25)
    print(f"模型 {WEIGHTS.name}  imgsz=896  sliced={bool(args.sliced)}")
    print(f"阈值 flicker_static={DEFAULTS['flicker_static']}  "
          f"flicker_fire={DEFAULTS['flicker_fire']}")

    # ── 真火：不该被拦 ────────────────────────────────────────
    frames, shape = pick_continuous(
        str(DATA / "images" / "A_烟雾_火_fire_CV*.jpg"), args.frames)
    print(f"\n真火连续帧 {len(frames)} 张  {shape[1]}x{shape[0]}")
    fire = collect(det, frames, "fire", bool(args.sliced))
    if not fire:
        print("  ⚠️ 没检出火轨迹，无法评估（换 --sliced 1 试试）")
        return 1
    fl = [r[2] for r in fire]
    cnt = Counter(r[3] for r in fire)
    blocked = cnt.get("static", 0)
    print(f"  轨迹样本 {len(fire)}   flicker 最小 {min(fl):.3f} / "
          f"P10 {np.percentile(fl, 10):.3f} / 中位 {np.median(fl):.3f} / 最大 {max(fl):.3f}")
    print(f"  判定 {dict(cnt)}")
    print(f"  真火被拦（判 static）: {blocked}/{len(fire)} = {blocked/len(fire):.1%}")

    # ── 恒定光源：该被拦 ──────────────────────────────────────
    light = synthetic_light(80, (shape[0], shape[1]))
    lrows = collect(det, light, "fire", False)
    if lrows:
        lf = [r[2] for r in lrows]
        lcnt = Counter(r[3] for r in lrows)
        print(f"\n合成恒定光源  轨迹样本 {len(lrows)}   "
              f"flicker 中位 {np.median(lf):.3f}   判定 {dict(lcnt)}")
        blocked_light = lcnt.get("static", 0)
        print(f"  恒定光源被拦: {blocked_light}/{len(lrows)} = "
              f"{blocked_light/len(lrows):.1%}")
    else:
        print("\n合成恒定光源：检测器没把它认成 fire —— "
              "这本身也说明不会误报（闸都用不上）")

    # ── 结论 ─────────────────────────────────────────────────
    ok = blocked / len(fire) <= 0.05
    print(f"\n{'=' * 54}")
    print(f"真火误拦率 {blocked/len(fire):.1%}（要求 ≤5%）  →  "
          f"{'✅ 通过' if ok else '❌ 不通过，阈值需要调宽'}")
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
