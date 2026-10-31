"""小面积火（<0.5% 画面）怎么提召回：把能拧的旋钮都量一遍。

为什么单独做这个
----------------
分段召回实测（`eval_smoldering.py`）：<0.5% 档只有 42%@0.45、57%@0.30，
而这块有 130 个 GT 框，是最大的一块漏报。笼统说"提高召回"没有意义 ——
要回答的是：**切片切多细、输入放多大，每一档能换多少召回、付多少毫秒。**

三个旋钮
--------
1. `tile_grid`：2x2 → 3x3 → 4x4。切片越多，每块被放大得越狠，小目标越清楚。
2. `imgsz`：每个 tile 送进网络的分辨率。
3. `overlap`：切片之间重叠多少（切细了目标容易落在缝上，重叠要跟着加）。

⚠️ 早先试过"只对可疑区域做第二次放大"（而不是切全），fire 召回掉了 6.7pt ——
因为候选框漏掉的区域永远补不到。所以这里只测**切全**，不测局部补。

跑法：
    python scripts/bench_small_object.py --n 200
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

DATA = Path("E:/datasets/fire_smoke_v5/test")
WEIGHTS = ROOT / "models" / "fire_smoke_v5_26s960_best.pt"
SMALL = 0.005                   # 小目标定义：占画面 <0.5%


def imread_u(p: Path):
    import cv2
    return cv2.imdecode(np.frombuffer(p.read_bytes(), np.uint8), cv2.IMREAD_COLOR)


def read_gt(txt: Path):
    out = []
    if not txt.is_file():
        return out
    for ln in txt.read_text(encoding="utf-8").splitlines():
        v = ln.split()
        if len(v) >= 5:
            c, cx, cy, w, h = int(float(v[0])), *map(float, v[1:5])
            out.append((c, (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2), w * h))
    return out


def iou(a, b) -> float:
    ix = min(a[2], b[2]) - max(a[0], b[0])
    iy = min(a[3], b[3]) - max(a[1], b[1])
    if ix <= 0 or iy <= 0:
        return 0.0
    inter = ix * iy
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200, help="取多少张含小面积火的图")
    ap.add_argument("--max-side", type=int, default=0, help=">0 时把图长边缩到这个值再测")
    a = ap.parse_args()

    from src.infer import Detector
    det = Detector(str(WEIGHTS), conf=0.15, imgsz=896, device=0)

    # 先挑出"含 <0.5% 火框"的图
    files = sorted((DATA / "images").glob("*.jpg"))
    picks = []
    for p in files:
        gts = read_gt(DATA / "labels" / (p.stem + ".txt"))
        if any(c == 0 and ar < SMALL for c, _b, ar in gts):
            picks.append(p)
        if len(picks) >= a.n:
            break
    print(f"含小面积火（<{SMALL * 100:.1f}%）的图：{len(picks)} 张\n")

    imgs = []
    for p in picks:
        im = imread_u(p)
        if im is None:
            continue
        if a.max_side and max(im.shape[:2]) > a.max_side:
            import cv2
            s = a.max_side / max(im.shape[:2])
            im = cv2.resize(im, (int(im.shape[1] * s), int(im.shape[0] * s)))
        imgs.append((p, im, read_gt(DATA / "labels" / (p.stem + ".txt"))))

    configs = [(2, 896, 0.25), (3, 896, 0.30), (4, 896, 0.30),
               (2, 1280, 0.25), (3, 1280, 0.30)]
    print(f"{'tile':>5}{'imgsz':>7}{'overlap':>9}"
          f"{'小目标@.30':>12}{'小目标@.20':>12}{'全火@.30':>10}{'ms/张':>8}")
    for grid, imgsz, ov in configs:
        det.imgsz = imgsz
        hit_s = {0.30: 0, 0.20: 0}
        tot_s = 0
        hit_all = 0
        tot_all = 0
        t0 = time.time()
        for p, im, gts in imgs:
            h, w = im.shape[:2]
            ds = det.detect(im, sliced=True, tile_grid=grid, overlap=ov)
            preds = [(d.conf, (d.xyxy[0] / w, d.xyxy[1] / h,
                               d.xyxy[2] / w, d.xyxy[3] / h))
                     for d in ds if d.cls_id == 0]
            for c, gb, ar in gts:
                if c != 0:
                    continue
                tot_all += 1
                ok30 = any(cf >= 0.30 and iou(gb, pb) >= 0.5 for cf, pb in preds)
                if ok30:
                    hit_all += 1
                if ar < SMALL:
                    tot_s += 1
                    if ok30:
                        hit_s[0.30] += 1
                    if any(cf >= 0.20 and iou(gb, pb) >= 0.5 for cf, pb in preds):
                        hit_s[0.20] += 1
        dt = (time.time() - t0) / max(1, len(imgs)) * 1000
        print(f"{grid:>5}{imgsz:>7}{ov:>9.2f}"
              f"{hit_s[0.30] / max(1, tot_s) * 100:>11.0f}%"
              f"{hit_s[0.20] / max(1, tot_s) * 100:>11.0f}%"
              f"{hit_all / max(1, tot_all) * 100:>9.0f}%{dt:>8.0f}")
    print(f"\n小目标框总数 {tot_s}（口径：GT 面积 <{SMALL * 100:.1f}% 的火焰框）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
