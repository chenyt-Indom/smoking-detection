"""单次高分辨率整图 vs 5 次切片：找一个"一次前向"就能顶住召回的 imgsz。

背景：切片之所以有用，是因为它把局部放大送进网络。2x2 切 + 896 输入，
等效放大倍率约 3.3 倍。但代价是 5 次前向（整图 + 4 片），896 下每次 ~11.7ms，
合计 ~55ms，流水线只有 18fps。

于是有个更省的想法：**不切了，直接把整图按更大的 imgsz 送一次**。
像素多了，但只有 1 次前向、1 次 kernel launch。这个脚本就是量这条曲线：
召回随 imgsz 怎么涨、耗时怎么涨，和 5 次切片的基准比。

用法：
    python scripts/bench_imgsz_sweep.py --n 100
    python scripts/bench_imgsz_sweep.py --sizes 896 1152 1280 1536 1920
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


def blank() -> np.ndarray:
    yy, xx = np.mgrid[0:540, 0:960].astype(np.float32)
    return np.clip(96 + 44 * (xx / 960)[..., None] + np.array([8, 6, 4], np.float32),
                   0, 255).astype(np.uint8)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=str(WEIGHTS))
    ap.add_argument("--images", default=str(DATA / "images"))
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--sizes", type=int, nargs="+", default=[896, 1152, 1280, 1536, 1920])
    ap.add_argument("--grid", type=int, default=2)
    a = ap.parse_args()

    files = sorted(glob.glob(a.images + "/*.jpg"))[: a.n]
    imgs, gts = [], []
    for f in files:
        im = imread_u(f)
        if im is None:
            continue
        imgs.append(im)
        gts.append(load_gt(f))
    print(f"权重 {Path(a.weights).name}  conf={a.conf}  样本 {len(imgs)} 张\n")

    rows = []

    def score(det: Detector, fn) -> tuple[float, float, float, float]:
        hit = {0: 0, 1: 0}
        tot = {0: 0, 1: 0}
        ms = []
        for img, gt in zip(imgs, gts):
            h, w = img.shape[:2]
            t0 = time.perf_counter()
            dets = fn(img)
            ms.append((time.perf_counter() - t0) * 1000)
            for c, gb in gt:
                if c not in hit:
                    continue
                tot[c] += 1
                gx = (gb[0] * w, gb[1] * h, gb[2] * w, gb[3] * h)
                if any(d.cls_id == c and iou(gx, d.xyxy) >= 0.5 for d in dets):
                    hit[c] += 1
        r0 = hit[0] / tot[0] if tot[0] else float("nan")
        r1 = hit[1] / tot[1] if tot[1] else float("nan")
        b = blank()
        tb = []
        for _ in range(10):
            t0 = time.perf_counter()
            fn(b)
            tb.append((time.perf_counter() - t0) * 1000)
        return r0, r1, float(np.mean(ms)), float(np.mean(tb))

    # 基准：5 次切片 @896
    det0 = Detector(a.weights, imgsz=896, conf=a.conf)
    det0.detect(imgs[0], sliced=True, tile_grid=a.grid)     # 预热
    r = score(det0, lambda im: det0.detect(im, sliced=True, tile_grid=a.grid))
    rows.append((f"切片{a.grid}x{a.grid}@896", *r))

    for sz in a.sizes:
        det = Detector(a.weights, imgsz=sz, conf=a.conf)
        det.detect(imgs[0], sliced=False)                    # 预热
        r = score(det, lambda im, d=det: d.detect(im, sliced=False))
        rows.append((f"整图@{sz}", *r))

    print(f"{'模式':<20}{'fire召回':>10}{'smoke召回':>11}{'均值ms':>9}{'空白帧ms':>10}")
    for name, r0, r1, mean, bms in rows:
        print(f"{name:<20}{r0:>10.1%}{r1:>11.1%}{mean:>9.2f}{bms:>10.2f}")

    b0, b1, bm, _ = rows[0]
    print(f"\n{'=' * 62}\n对照「{rows[0][0]}」：")
    for name, r0, r1, mean, bms in rows[1:]:
        ok = "✅" if (r0 - b0) * 100 >= -1.0 and (r1 - b1) * 100 >= -1.0 else "❌ 召回掉太多"
        print(f"  {name:<16} fire {(r0-b0)*100:+5.1f}pt  smoke {(r1-b1)*100:+5.1f}pt  "
              f"{bm/mean:4.2f}x  {ok}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
