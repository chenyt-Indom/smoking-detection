"""把 24,574 张外部烟图按「有没有火」分档，为数据集合并提供依据。

判据：用当前最好的双类模型 v5 对每张图跑一次检测，只取 fire 类的最高置信度。
      分档统计，输出清单，供后续决定哪些图可以进双类训练、哪些只能用于纯烟模型。

为什么要分档而不是一刀切阈值：
  之前实测 v5 在 conf 0.83 仍会把明亮白烟判成火（smoke_001590）。
  所以「高阈值就安全」不成立，必须看分布形状再定，而不是拍一个数。
"""
import argparse
import json
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(r"D:/datasets/smoke_zip_full")
WEIGHTS = r"D:/视觉安防系统/models/fire_smoke_v5_26s960_best.pt"

BINS = [(0.00, 0.05), (0.05, 0.15), (0.15, 0.25), (0.25, 0.50), (0.50, 1.01)]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--imgsz", type=int, default=960)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--out", default=r"D:/datasets/smoke_zip_full/fire_scan.json")
    a = ap.parse_args()

    imgs = sorted(str(p) for split in ("train", "val")
                  for p in (ROOT / split / "images").glob("*.jpg"))
    print(f"待扫描 {len(imgs)} 张", flush=True)

    model = YOLO(WEIGHTS)
    recs = []
    chunk_report = 2000

    for i in range(0, len(imgs), a.batch):
        chunk = imgs[i:i + a.batch]
        res = model.predict(chunk, imgsz=a.imgsz, conf=0.03, iou=0.7,
                            verbose=False, device=0, half=True)
        for path, r in zip(chunk, res):
            fire = [(float(c), [round(float(x), 1) for x in b])
                    for c, b in zip(r.boxes.cls.tolist(), r.boxes.xyxy.tolist())
                    if int(c) == 0]
            best = max((c for c, _ in fire), default=0.0)
            recs.append({"p": path, "fmax": round(best, 4), "nfire": len(fire)})
        if (i + a.batch) % chunk_report == 0:
            print(f"  {min(i + a.batch, len(imgs))}/{len(imgs)}", flush=True)

    Path(a.out).write_text(json.dumps(recs), encoding="utf-8")

    print("\n=== 按 fire 最高置信度分档 ===")
    tot = len(recs)
    for lo, hi in BINS:
        n = sum(1 for r in recs if lo <= r["fmax"] < hi)
        print("[{:.2f}, {:.2f})  {:>7} 张  {:>5.1f}%".format(lo, hi, n, n / tot * 100))

    print("\n合计 {} 张".format(tot))
    for thr in (0.05, 0.15, 0.25, 0.50, 0.80):
        n = sum(1 for r in recs if r["fmax"] < thr)
        print("  fire 最高分 < {:.2f} 的图：{:>6} 张（{:5.1f}%）".format(thr, n, n / tot * 100))

    multi = sum(1 for r in recs if r["nfire"] >= 3)
    print("\n火框 >= 3 个的图：{} 张".format(multi))
    print("清单已写入 {}".format(a.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
