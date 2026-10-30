"""切片推理自测：直接对比「整图」与「切片」在同一批 test 图上的召回。

用法：
    python scripts/test_sliced_infer.py                       # 默认 v3 权重 + 40 张
    python scripts/test_sliced_infer.py --weights D:/fire_runs/fire_smoke_v4_26s960/weights/best.pt --n 60

判定标准（不是硬断言，避免 GPU/环境差异导致误报）：
  · fire 切片召回应 ≥ 整图召回（v3 实测 +8 点）
  · smoke 两者应基本持平（烟是大目标，切片无收益）
"""
from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.infer import Detector, iou  # noqa: E402


def load_gt(img_path: str):
    p = Path(img_path)
    lbl = p.parent.parent / "labels" / (p.stem + ".txt")
    out = []
    if lbl.exists():
        for line in lbl.read_text(encoding="utf-8").splitlines():
            t = line.split()
            if len(t) >= 5:
                c, cx, cy, w, h = int(float(t[0])), *[float(v) for v in t[1:5]]
                out.append((c, (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="D:/视觉安防系统/runs/fire_smoke_v3_26n/weights/best.pt")
    ap.add_argument("--images", default="E:/datasets/fire_smoke_v3/test/images")
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--grid", type=int, default=2)
    ap.add_argument("--overlap", type=float, default=0.25)
    a = ap.parse_args()

    det = Detector(a.weights, conf=a.conf, imgsz=640)
    files = sorted(glob.glob(a.images + "/*.jpg"))[: a.n]
    print(f"权重 {a.weights}\n样本 {len(files)} 张 · 切片 {a.grid}x{a.grid} 重叠 {a.overlap:.0%}\n")
    print(f"{'类别':<7}{'GT':>6}{'整图命中':>9}{'切片命中':>9}{'整图召回':>9}{'切片召回':>9}")

    agg = {0: {"gt": 0, "d": 0, "s": 0}, 1: {"gt": 0, "d": 0, "s": 0}}
    for f in files:
        import cv2
        import numpy as np

        img = cv2.imdecode(np.fromfile(f, dtype=np.uint8), cv2.IMREAD_COLOR)
        h, w = img.shape[:2]
        gts = load_gt(f)
        direct = det.detect(img, sliced=False)
        sliced = det.detect(img, sliced=True, tile_grid=a.grid, overlap=a.overlap)

        for c, gb in gts:
            if c not in agg:
                continue
            agg[c]["gt"] += 1
            gx = (gb[0] * w, gb[1] * h, gb[2] * w, gb[3] * h)
            if any(d.cls_id == c and iou(gx, d.xyxy) >= 0.5 for d in direct):
                agg[c]["d"] += 1
            if any(d.cls_id == c and iou(gx, d.xyxy) >= 0.5 for d in sliced):
                agg[c]["s"] += 1

    ok = True
    for c, name in [(0, "fire"), (1, "smoke")]:
        s = agg[c]
        if s["gt"] == 0:
            print(f"{name:<7}{0:>6}{'-':>9}{'-':>9}{'-':>9}{'-':>9}")
            continue
        rd, rs = s["d"] / s["gt"], s["s"] / s["gt"]
        flag = ""
        if name == "fire" and rs < rd:
            flag = "  ← 异常：切片反而掉了"
            ok = False
        print(f"{name:<7}{s['gt']:>6}{s['d']:>9}{s['s']:>9}{rd:>9.3f}{rs:>9.3f}{flag}")

    print("\n结论：", "切片对 fire 有增益或持平，符合预期" if ok else "切片未带来增益，检查 tile_grid/overlap")
    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
