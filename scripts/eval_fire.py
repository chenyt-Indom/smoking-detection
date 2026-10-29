"""在指定 split 上评估权重，输出每类指标。

Windows 注意：不要用 `python - <<EOF` 跑带多进程推理的代码 ——
ultralytics 的 dataloader 用 spawn，`__main__` 会变成 `<stdin>`，
直接报 `OSError: [Errno 22] Invalid argument: '...<stdin>'`。
所以必须落成 .py 文件再执行（这个坑记一下）。

跑法：
    python scripts/eval_fire.py --weights runs/fire_smoke_26n/weights/best.pt --split test
    python scripts/eval_fire.py --weights models/fire_smoke_26n_best.pt --split val --imgsz 960
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="runs/fire_smoke_26n/weights/best.pt")
    ap.add_argument("--data", default="E:/datasets/fire_smoke_v2/dataset.yaml")
    ap.add_argument("--split", default="test", choices=["train", "val", "valid", "test"])
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--conf", type=float, default=0.001, help="评估用低阈值，别卡掉低分框")
    a = ap.parse_args()

    from ultralytics import YOLO

    m = YOLO(a.weights)
    r = m.val(data=a.data, split=a.split, imgsz=a.imgsz, batch=a.batch,
              conf=a.conf, verbose=False, plots=True, project=str(ROOT / "runs"),
              name=f"eval_{a.split}")

    names = r.names if isinstance(r.names, dict) else dict(enumerate(r.names))
    print("=" * 60)
    print(f"权重 {a.weights}")
    print(f"数据 {a.data} · split={a.split} · imgsz={a.imgsz}")
    print("-" * 60)
    print(f"总体  mAP50 {r.box.map50:.4f}   mAP50-95 {r.box.map:.4f}"
          f"   P {r.box.mp:.4f}   R {r.box.mr:.4f}")
    for i, cname in names.items():
        if i < len(r.box.ap50):
            print(f"  {cname:6s} mAP50 {r.box.ap50[i]:.4f}   mAP50-95 {r.box.ap[i]:.4f}")
    print("=" * 60)

    out = ROOT / "runs" / f"eval_{a.split}_metrics.json"
    out.write_text(json.dumps({
        "weights": a.weights, "split": a.split, "imgsz": a.imgsz,
        "mAP50": round(float(r.box.map50), 4),
        "mAP50-95": round(float(r.box.map), 4),
        "precision": round(float(r.box.mp), 4),
        "recall": round(float(r.box.mr), 4),
        "per_class": {names[i]: round(float(r.box.ap50[i]), 4)
                      for i in range(len(r.box.ap50)) if i in names},
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"指标已存：{out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
