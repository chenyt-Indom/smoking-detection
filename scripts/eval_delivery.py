"""训练完成后的交付评估：阈值扫描 + 背景误报率 + 分尺寸 AP。

为什么不能只报一个 mAP50
-----------------------
mAP 是"把所有置信度都算进去"的综合分，**不能直接回答能不能用**。
安防要的是三个具体数字：
  ① 在某个人工设定的阈值下，漏报多少（Recall）、误报多少（Precision）
  ② 对着**纯背景画面**会不会瞎报（误报率）—— 决定值班员会不会关掉系统
  ③ 小目标（远处早期火情）单独看，会不会被整体平均分掩盖

用法：
    python scripts/eval_delivery.py --weights runs/fire_smoke_v3_26n/weights/best.pt \
        --bg-dir "E:/BaiduNetdiskDownload/Y_火灾+烟雾/背景负样本/images"
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="runs/fire_smoke_v3_26n/weights/best.pt")
    ap.add_argument("--data", default="E:/datasets/fire_smoke_v3/dataset.yaml")
    ap.add_argument("--bg-dir", default="E:/BaiduNetdiskDownload/Y_火灾+烟雾/背景负样本/images")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--thresholds", default="0.15,0.25,0.35,0.50")
    ap.add_argument("--skip-val", action="store_true")
    a = ap.parse_args()

    from ultralytics import YOLO

    m = YOLO(a.weights)
    report = {"weights": a.weights, "imgsz": a.imgsz}

    # ---------- ① 阈值扫描（同一模型，不同 conf 下的 P/R）----------
    if not a.skip_val:
        print("=" * 64)
        print("① 阈值扫描（test 集，看不同置信度门限下的性价比）")
        print(f"{'conf':>6} {'Precision':>10} {'Recall':>8} {'mAP50':>8} {'mAP50-95':>9}")
        rows = []
        for thr in [float(x) for x in a.thresholds.split(",")]:
            r = m.val(data=a.data, split="test", imgsz=a.imgsz, batch=32,
                      conf=thr, verbose=False, plots=False)
            row = {"conf": thr, "P": round(float(r.box.mp), 4),
                   "R": round(float(r.box.mr), 4),
                   "mAP50": round(float(r.box.map50), 4),
                   "mAP50-95": round(float(r.box.map), 4)}
            rows.append(row)
            print(f"{thr:>6.2f} {row['P']:>10.4f} {row['R']:>8.4f} "
                  f"{row['mAP50']:>8.4f} {row['mAP50-95']:>9.4f}")
        report["threshold_scan"] = rows

    # ---------- ② 背景误报率（最关键）----------
    bg_dir = Path(a.bg_dir)
    if bg_dir.exists():
        imgs = sorted(p for p in bg_dir.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
        if imgs:
            print("\n" + "=" * 64)
            print(f"② 纯背景误报测试：{len(imgs)} 张无火无烟的画面")
            fp = {"0.25": 0, "0.35": 0, "0.50": 0}
            t0 = time.time()
            # 分批预测，一次 64 张
            for i in range(0, len(imgs), 64):
                batch = [str(p) for p in imgs[i:i + 64]]
                for thr in (0.25, 0.35, 0.50):
                    res = m.predict(batch, imgsz=a.imgsz, conf=thr, verbose=False)
                    fp[f"{thr:.2f}"] += sum(1 for r in res if len(r.boxes))
            dt = time.time() - t0
            print(f"{'conf门限':>8} {'误报张数':>10} {'误报率':>9}   说明")
            for k, v in fp.items():
                rate = v / len(imgs) * 100
                tag = "可用" if rate < 5 else ("偏高" if rate < 15 else "不可用")
                print(f"{k:>8} {v:>10} {rate:>8.1f}%   {tag}")
            report["background_fp"] = {k: {"count": v, "rate": round(v / len(imgs), 4)}
                                       for k, v in fp.items()}
            report["background_n"] = len(imgs)

    # ---------- ③ 推理速度 ----------
    print("\n" + "=" * 64)
    print("③ 推理速度（单张 640）")
    import numpy as np
    dummy = np.zeros((480, 640, 3), dtype=np.uint8)
    for _ in range(3):
        m.predict(dummy, imgsz=a.imgsz, verbose=False)
    t0 = time.time()
    n = 20
    for _ in range(n):
        m.predict(dummy, imgsz=a.imgsz, verbose=False)
    ms = (time.time() - t0) / n * 1000
    print(f"  平均 {ms:.1f} ms/张（约 {1000/ms:.0f} FPS，含预处理与后处理）")
    report["latency_ms"] = round(ms, 2)

    out = ROOT / "runs" / "delivery_report.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n报告已存：{out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
