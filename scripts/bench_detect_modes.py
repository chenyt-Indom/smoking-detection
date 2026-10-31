"""分级检测 A/B：整图 / 固定切片 / 分级（粗筛+按需补切）。

量两件事，缺一不可：
  · **召回**（test 集上按 IoU≥0.5 逐框命中）—— 提速不能拿召回换
  · **耗时**（每帧均值 + p95）—— 只看均值会被个别大图带偏

另外单独测一段"空白画面"的耗时：真实枪机 99% 的帧是空的，
而固定切片是每帧都付 5 次前向的钱，这里的差距才是分级检测的价值所在。

用法
----
    python scripts/bench_detect_modes.py                # 默认 120 张，_v5 权重
    python scripts/bench_detect_modes.py --n 300
    python scripts/bench_detect_modes.py --trigger 0.05
"""
from __future__ import annotations

import argparse
import glob
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.infer import Detector, iou                                    # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
WEIGHTS = ROOT / "models" / "fire_smoke_v5_26s960_best.pt"
DATA = Path("E:/datasets/fire_smoke_v5/test")
NAMES = {0: "fire", 1: "smoke"}
WARM = 5            # 预热几次前向，把 CUDA/cuDNN 的首次开销排除掉


def imread_u(p: str):
    return cv2.imdecode(np.fromfile(p, dtype=np.uint8), cv2.IMREAD_COLOR)


def load_gt(img_path: str):
    p = Path(img_path)
    lbl = p.parent.parent / "labels" / (p.stem + ".txt")
    out = []
    if lbl.exists():
        for line in lbl.read_text(encoding="utf-8", errors="ignore").splitlines():
            t = line.split()
            if len(t) >= 5:
                c = int(float(t[0]))
                cx, cy, w, h = (float(v) for v in t[1:5])
                out.append((c, (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)))
    return out


def blank_scene(n: int):
    """一块墙面 + 缓慢曝光漂移。真实枪机绝大多数字段是这样的。"""
    yy, xx = np.mgrid[0:540, 0:960].astype(np.float32)
    base = 96 + 44 * (xx / 960)[..., None] + np.array([8, 6, 4], np.float32)
    base = np.clip(base, 0, 255).astype(np.uint8)
    for i in range(n):
        yield cv2.convertScaleAbs(base, alpha=1.0, beta=1.2 * np.sin(i / 60.0))


def run_mode(det: Detector, imgs, gts, sizes, fn):
    """返回 (每类命中/总数, 每帧耗时列表)。"""
    hit = {0: 0, 1: 0}
    tot = {0: 0, 1: 0}
    ms: list[float] = []
    for img, gt, (h, w) in zip(imgs, gts, sizes):
        t0 = time.perf_counter()
        dets = fn(det, img)
        ms.append((time.perf_counter() - t0) * 1000)
        for c, gb in gt:
            if c not in hit:
                continue
            tot[c] += 1
            gx = (gb[0] * w, gb[1] * h, gb[2] * w, gb[3] * h)
            if any(d.cls_id == c and iou(gx, d.xyxy) >= 0.5 for d in dets):
                hit[c] += 1
    return hit, tot, ms


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=str(WEIGHTS))
    ap.add_argument("--images", default=str(DATA / "images"))
    ap.add_argument("--n", type=int, default=120)
    ap.add_argument("--imgsz", type=int, default=896)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--grid", type=int, default=2)
    ap.add_argument("--trigger", type=float, action="append",
                    help="分级检测的粗筛触发线，可重复给多个值")
    a = ap.parse_args()
    triggers = a.trigger or [0.05, 0.08, 0.15]

    files = sorted(glob.glob(a.images + "/*.jpg"))[: a.n]
    imgs, gts, sizes = [], [], []
    for f in files:
        im = imread_u(f)
        if im is None:
            continue
        imgs.append(im)
        gts.append(load_gt(f))
        sizes.append(im.shape[:2])
    print(f"权重 {Path(a.weights).name}  imgsz={a.imgsz}  conf={a.conf}  "
          f"样本 {len(imgs)} 张  切片 {a.grid}x{a.grid}\n")

    det = Detector(a.weights, imgsz=a.imgsz, conf=a.conf)
    modes = [
        ("整图", lambda d, im: d.detect(im, sliced=False)),
        (f"切片{a.grid}x{a.grid}·逐张", lambda d, im: d.detect(
            im, sliced=True, tile_grid=a.grid, batched=False)),
        (f"切片{a.grid}x{a.grid}·批量", lambda d, im: d.detect(
            im, sliced=True, tile_grid=a.grid, batched=True)),
        (f"切片{a.grid}x{a.grid}·批量·无整图", lambda d, im: d.detect(
            im, sliced=True, tile_grid=a.grid, batched=True, keep_full_frame=False)),
    ]
    for tc in triggers:
        modes.append((f"分级·触发{tc:.2f}",
                      lambda d, im, tc=tc: d.detect_auto(im, trigger_conf=tc,
                                                         tile_grid=a.grid)))

    # 预热
    for im in imgs[:WARM]:
        det.detect(im, sliced=False)
        det.detect(im, sliced=True, tile_grid=a.grid, batched=True)
        det.detect_auto(im, trigger_conf=0.08)

    print(f"{'模式':<16}{'fire召回':>10}{'smoke召回':>11}"
          f"{'均值ms':>9}{'p95ms':>8}{'帧率':>8}{'空白帧ms':>10}")
    results = {}
    for name, fn in modes:
        hit, tot, ms = run_mode(det, imgs, gts, sizes, fn)
        # 空白画面单独计时
        bms = []
        for bi in blank_scene(20):
            t0 = time.perf_counter()
            fn(det, bi)
            bms.append((time.perf_counter() - t0) * 1000)
        rec = {c: (hit[c] / tot[c] if tot[c] else float("nan")) for c in (0, 1)}
        mean = float(np.mean(ms))
        p95 = float(np.percentile(ms, 95))
        results[name] = (rec, mean, float(np.mean(bms)))
        print(f"{name:<16}{rec[0]:>10.1%}{rec[1]:>11.1%}"
              f"{mean:>9.2f}{p95:>8.2f}{1000/mean:>8.1f}{np.mean(bms):>10.2f}" )
        del hit, tot, ms

    # ── 结论 ────────────────────────────────────────────────
    base_name = f"切片{a.grid}x{a.grid}·逐张"
    base = results[base_name]
    print(f"\n{'=' * 72}")
    print(f"对照基准「{base_name}」（= 现在的线上行为）的差值，正数=更好：")
    for name, (rec, mean, bmean) in results.items():
        if name == base_name:
            continue
        d0 = (rec[0] - base[0][0]) * 100
        d1 = (rec[1] - base[0][1]) * 100
        sp = base[1] / mean
        print(f"  {name:<20} fire {d0:+5.1f}pt  smoke {d1:+5.1f}pt  "
              f"耗时 {mean:5.1f}ms（{sp:.2f}x）  空白帧 {bmean:5.1f}ms")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
