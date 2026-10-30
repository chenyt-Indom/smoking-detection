"""火情/烟雾检测 v4 训练：yolo26s @ 960 + 多尺度 + fire 过采样数据集。

相对 v3（yolo26n @ 640 + v3 数据集）改了三处，全部针对「fire 目标太小」这个诊断结论：
  1. 模型 n → s（2.4M → 9.5M 参数，v3 的 n 已收敛饱和，换大模型才有上升空间）
  2. 输入 640 → 960（fire 框面积中位数仅 0.84% 画面，28% 小于 0.25%，
     640 下这些小目标只剩 20~30px，小目标检测的常规解法就是提输入分辨率）
  3. 数据集 v3 → v4（含 fire 的图额外过采样一份，13,476 张 → 覆盖翻倍）
另外开 multi_scale=0.5，让模型对尺度不敏感，部署时切片/缩放都不至于掉太多。

注意：**推理时不要自己改分辨率**（v3 实测 960/1280 推理反而掉点）。
分辨率必须训练时就定下来，这也是这一版重训的主要原因。

用法：
    python scripts/train_v4.py                # 默认配置
    python scripts/train_v4.py --batch 6      # 显存吃紧时手动降
    python scripts/train_v4.py --dry-run      # 只校验数据与权重
"""
from __future__ import annotations

import argparse
import shutil
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def parse() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="yolo26s.pt")
    ap.add_argument("--data", default="E:/datasets/fire_smoke_v4/dataset.yaml")
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--imgsz", type=int, default=960)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--name", default="fire_smoke_v4_26s960")
    # 输出目录用纯英文：中文路径下 ultralytics 偶发把结果写到盘根
    ap.add_argument("--project", default="D:/fire_runs")
    ap.add_argument("--device", default="0")
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--multi-scale", type=float, default=0.5)
    ap.add_argument("--fraction", type=float, default=1.0, help="只用部分训练数据（标定/试跑用）")
    ap.add_argument("--close-mosaic", type=int, default=10, help="最后 N 轮关闭 mosaic")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    return ap.parse_args()


def main() -> int:
    a = parse()
    data = Path(a.data)
    if not data.exists():
        print(f"[x] 找不到数据配置：{data}")
        return 1

    n_train = len(list((data.parent / "train" / "images").glob("*")))
    print("=" * 70)
    print("  火情/烟雾 v4 · yolo26s @ 960")
    print(f"  数据   {data}（train {n_train} 张）")
    print(f"  配置   imgsz {a.imgsz} | batch {a.batch} | epochs {a.epochs} | multi_scale {a.multi_scale} | close_mosaic {a.close_mosaic}")
    print("=" * 70, flush=True)

    from ultralytics import YOLO

    model = YOLO(a.model)
    if a.dry_run:
        print("[dry-run] 权重加载成功，不训练")
        return 0

    kwargs = dict(
        data=str(data),
        epochs=a.epochs,
        imgsz=a.imgsz,
        batch=a.batch,
        device=a.device,
        workers=a.workers,
        project=a.project,
        name=a.name,
        exist_ok=True,
        resume=a.resume,
        multi_scale=a.multi_scale,
        fraction=a.fraction,
        # 小目标靠 mosaic 提高出现频率，末期关掉回归真实分布
        mosaic=1.0,
        close_mosaic=a.close_mosaic,
        # 火/烟对亮度、饱和度敏感
        hsv_h=0.015, hsv_s=0.7, hsv_v=0.5,
        degrees=5.0, translate=0.1, scale=0.5, shear=0.0,
        flipud=0.0,          # 禁止上下翻：烟只向上飘
        fliplr=0.5,
        erasing=0.2,
        patience=a.patience,
        verbose=True,
        seed=0,
        amp=True,
        plots=True,
    )

    t0 = time.time()
    try:
        model.train(**kwargs)
    except RuntimeError as exc:
        msg = str(exc).lower()
        if "out of memory" in msg or "cuda" in msg:
            nb = max(2, a.batch // 2)
            print(f"[!] 显存不足，自动降到 batch={nb} 重试…", flush=True)
            kwargs["batch"] = nb
            model = YOLO(a.model)
            model.train(**kwargs)
        else:
            raise

    print(f"\n训练结束，用时 {(time.time() - t0) / 60:.1f} 分钟", flush=True)
    best = Path(a.project) / a.name / "weights" / "best.pt"
    if best.exists():
        dst = ROOT / "models" / f"{a.name}_best.pt"
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(best, dst)
        print(f"最优权重：{best} → {dst}（{dst.stat().st_size / 1e6:.1f} MB）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
