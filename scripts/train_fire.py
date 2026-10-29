"""火情/烟雾检测训练脚本（YOLO26）。

沿用旧仓库 `train_offline.py` 的骨架（加载权重 → 训练 → 报出 best.pt），
升级到 YOLO26 + 2 类（fire / smoke）+ 按视频分段的干净数据集。

用法：
    python scripts/train_fire.py                     # 默认配置
    python scripts/train_fire.py --model yolo26s     # 换尺寸
    python scripts/train_fire.py --epochs 200 --batch 8
    python scripts/train_fire.py --dry-run           # 只检查数据和显存，不训练

为什么默认是 yolo26l 而不是 x：
    5070 Ti 笔记本 12GB，x 是 55.7M 参数，640 分辨率下 batch 只能开到 4~6，
    一次实验要跑 3 小时以上；l 是 24.8M，精度只低 2.5 个 mAP 点，速度却快一倍。
    先用 l 把链路和指标跑顺，再拿 x 冲上限才有意义。
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def parse() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="yolo26l.pt", help="yolo26n/s/m/l/x 或自训权重")
    ap.add_argument("--data", default="E:/datasets/fire_smoke_v2/dataset.yaml")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=12)
    ap.add_argument("--workers", type=int, default=2, help="Windows 上 8 个 worker 会吃掉 ~14GB 提交内存，2 个就够")
    ap.add_argument("--name", default="fire_smoke_26l")
    ap.add_argument("--project", default=str(ROOT / "runs"),
                    help="输出根目录。建议用纯英文路径（中文路径下 ultralytics 偶发把结果写到盘根）")
    ap.add_argument("--device", default="0")
    ap.add_argument("--patience", type=int, default=25)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    return ap.parse_args()


def main() -> int:
    a = parse()
    data = Path(a.data)
    if not data.exists():
        print(f"[x] 找不到数据集配置：{data}")
        return 1

    print("=" * 66)
    print("  火情 / 烟雾检测 · YOLO26")
    print(f"  数据   {data}")
    print(f"  模型   {a.model} | imgsz {a.imgsz} | batch {a.batch} | epochs {a.epochs}")
    print("=" * 66)

    # 数据自检：各类框数（底线：train≥1500 / valid≥300）
    try:
        import yaml
        cfg = yaml.safe_load(data.read_text(encoding="utf-8"))
        names = cfg.get("names") or []
        print(f"  类别：{names}")
    except Exception as exc:  # noqa: BLE001
        print(f"  [!] 读 dataset.yaml 失败：{exc}")

    from ultralytics import YOLO

    if a.dry_run:
        print("[dry-run] 不训练，仅验证权重与数据可加载")
        m = YOLO(a.model)
        print("  权重加载成功")
        return 0

    t0 = time.time()
    model = YOLO(a.model)

    try:
        results = model.train(
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
            # 小目标（fire 有 29% 的框小于 0.25% 画面）—— 保留 mosaic，末期关掉
            mosaic=1.0,
            close_mosaic=15,
            # 数据增强：火/烟对亮度与模糊很敏感，这几项给足
            hsv_h=0.015, hsv_s=0.7, hsv_v=0.5,
            degrees=5.0, translate=0.1, scale=0.5, shear=0.0,
            flipud=0.0, fliplr=0.5,
            patience=a.patience,
            verbose=True,
            seed=0,
            amp=True,
            plots=True,
        )
    except RuntimeError as exc:
        # 显存不够是最常见的失败原因，自动降一档再试一次
        msg = str(exc).lower()
        if "out of memory" in msg or "cuda" in msg:
            print(f"[!] 显存不足（{exc}），自动降到 batch={max(4, a.batch // 2)} 重试…")
            results = model.train(
                data=str(data), epochs=a.epochs, imgsz=a.imgsz,
                batch=max(4, a.batch // 2), device=a.device, workers=a.workers,
                project=a.project, name=a.name, exist_ok=True,
                mosaic=1.0, close_mosaic=15, patience=a.patience,
                verbose=True, seed=0, amp=True, plots=True,
            )
        else:
            raise

    dt = (time.time() - t0) / 60
    best = Path(a.project) / a.name / "weights" / "best.pt"
    print("=" * 66)
    print(f"训练结束，用时 {dt:.1f} 分钟")
    if best.exists():
        size = best.stat().st_size / 1024 / 1024
        print(f"最佳权重：{best}（{size:.1f} MB）")
        # 用 run 名做文件标识；别拿 a.model 拼（它可能是 runs/xxx/weights/last.pt，
        # split('.') 后带斜杠会把路径拼成多层，mkdir 又没加 parents=True → FileNotFoundError）
        tag = Path(a.model).stem if a.model.endswith(".pt") else a.model
        tag = tag.replace("last", a.name).replace("best", a.name)
        dst = ROOT / "models" / f"fire_smoke_{tag}.pt"
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(best, dst)
        print(f"已复制到：{dst}")
    else:
        print("[!] 没找到 best.pt，检查 runs 目录")
        return 1

    # 训练完顺手在 test 上跑一次，给出交付指标
    try:
        m = YOLO(str(best))
        metrics = m.val(data=str(data), split="test", imgsz=a.imgsz, verbose=False)
        print(f"test 指标：mAP50={metrics.box.map50:.4f}  mAP50-95={metrics.box.map:.4f}")
    except Exception as exc:  # noqa: BLE001
        print(f"[!] test 评估失败：{exc}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
