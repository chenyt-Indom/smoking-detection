"""香烟检测 · yolo26s 训练。

为什么单独一个脚本而不是改 train_v4.py
--------------------------------------
火情/烟雾和香烟是两套数据、两套目标尺度：
  · 火焰框面积中位数约 0.84% 画面
  · 香烟在 1280x720 里通常只有 0.02%~0.1%，比火焰还小一个量级
所以香烟要更高的训练分辨率、更严的增强尺度、以及专门的小目标策略。
混在一个脚本里两边都调不好。

当前数据
--------
  v14_train   33,604 张（历史最大，v28 就是在这上面训出来的）
  v30_regions  最近一版（区域裁剪，尺度更接近现场）
脚本默认用 v14_train 打底，可以用 --data 换成别的，或者用 --merge 把新样本并进来。

关键参数怎么定的
----------------
imgsz 1280
  香烟目标小，960 下很多样本的长边只有 20~30px。1280 是这块 12GB 显卡
  在 batch=4 下能稳住的上限。
batch 4
  1280 + yolo26s ≈ 9.5M 参数，batch 8 会 OOM（实测 12GB 上 batch 6 就开始
  报 CUDA out of memory，且整卡没有别的进程）。
multi_scale 0.3
  比火情的 0.5 小：香烟尺度变化本身不大，开太大反而让它在 1280 下
  频繁降到 900 以下，小目标直接糊掉。
mosaic 1.0 + close_mosaic 15
  mosaic 能让小目标在一张图里出现多次，对香烟很关键；末期关掉回归真实分布。

用法
----
    python scripts/train_cig.py --dry-run            # 只校验数据和权重
    python scripts/train_cig.py                      # 从 yolo26s.pt 从头训
    python scripts/train_cig.py --model models/smoke_cig_v28.pt --lr0 0.004
                                                     # 拿 v28 当预训练微调
    python scripts/train_cig.py --data D:/training_data/smoke/v14_plus/data.yaml
                                                     # 用并入了新样本的数据集
"""
from __future__ import annotations

import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def parse() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="yolo26s.pt",
                    help="从零训练用 yolo26s.pt；微调传已有权重（如 models/smoke_cig_v28.pt）")
    ap.add_argument("--data", default="D:/training_data/smoke/v14_train/data.yaml")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--name", default="cig_v31_26s1280")
    ap.add_argument("--project", default="D:/fire_runs")   # 纯英文路径
    ap.add_argument("--device", default="0")
    ap.add_argument("--patience", type=int, default=25)
    ap.add_argument("--multi-scale", type=float, default=0.3)
    ap.add_argument("--fraction", type=float, default=1.0)
    ap.add_argument("--close-mosaic", type=int, default=15)
    ap.add_argument("--lr0", type=float, default=0.01,
                    help="从零训练 0.01；从已有权重微调建议 0.003~0.005，"
                         "用 0.01 会把原能力冲掉")
    ap.add_argument("--optimizer", default="auto",
                    help="ultralytics 在 auto 时会忽略 lr0（强制 MuSGD lr=0.01）。"
                         "微调必须显式指定，如 --optimizer MuSGD")
    ap.add_argument("--warmup-bias-lr", type=float, default=0.0,
                    help="默认 0.1 是为从零训练准备的，微调时会在第一轮就把权重冲散")
    ap.add_argument("--freeze", type=int, default=0,
                    help="冻结前 N 层。微调少量新样本时设 10 左右能保住通用特征")
    ap.add_argument("--resume-path", default="", help="从 last.pt 续训")
    ap.add_argument("--dry-run", action="store_true")
    return ap.parse_args()


def main() -> int:
    a = parse()
    data = Path(a.data)
    if not data.exists():
        print(f"[x] 找不到数据配置：{data}")
        return 1

    # 统计训练集规模
    y = data.read_text(encoding="utf-8", errors="ignore")
    n_train = "未知"
    for line in y.splitlines():
        if line.strip().startswith("path:"):
            base = Path(line.split(":", 1)[1].strip())
            tr = base / "images" / "train"
            if tr.is_dir():
                n_train = f"{len(list(tr.glob('*.jpg'))) + len(list(tr.glob('*.png')))} 张"
            break

    print("=" * 72)
    print("  香烟检测 · yolo26s")
    print(f"  权重   {a.model}")
    print(f"  数据   {data}（train {n_train}）")
    print(f"  配置   imgsz {a.imgsz} | batch {a.batch} | epochs {a.epochs} "
          f"| multi_scale {a.multi_scale} | freeze {a.freeze}")
    print(f"  存档   {a.project}/{a.name}")
    print("=" * 72, flush=True)

    from ultralytics import YOLO

    model = YOLO(a.resume_path if a.resume_path else a.model)
    if a.dry_run:
        print("[dry-run] 权重与数据都正常，不训练")
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
        resume=bool(a.resume_path),
        multi_scale=a.multi_scale,
        fraction=a.fraction,
        close_mosaic=a.close_mosaic,
        # ⚠️ 增强参数必须**和推理时的尺度对齐**，否则前功尽弃。
        #
        # mosaic 1.0 会把 4 张拼成一张，每张缩到 1/4 大小 ——
        # 训练 imgsz 960 时香烟本来 20x32，拼完只剩 10x16，
        # 再叠加 scale ±50% 就是 5x8~15x24，而推理切片后是 39x62，
        # 差 5 倍以上。这正是"分辨率和 lr 都调对了还是不准"的第三个原因。
        mosaic=0.25,          # 偶尔拼一下见识多样性，但不能常态化
        scale=0.15,           # 尺度抖动收窄到 ±15%，保住目标像素
        degrees=8.0,          # 香烟持握倾斜有限，15 度会转成失真角度
        shear=2.0,
        perspective=0.0005,
        fliplr=0.5,
        flipud=0.0,           # 上下翻转会让香烟出现在不合常理的位置
        # erasing=0 是必须的：原来 0.4 会把香烟随机擦掉一块，
        # 但标注还在 —— 等于人为制造脏数据
        erasing=0.0,
        hsv_h=0.015,
        hsv_s=0.5,
        hsv_v=0.4,
        patience=a.patience,
        optimizer=a.optimizer,
        lr0=a.lr0,
        warmup_bias_lr=a.warmup_bias_lr,
        # 小目标密集，降低 NMS 前的 conf 让更多候选进入评估
        plots=True,
        val=True,
        save=True,
        seed=0,
    )
    if a.freeze:
        kwargs["freeze"] = a.freeze

    model.train(**kwargs)

    # 训练完自动把最佳权重拷到 models/，命名带版本号
    best = Path(a.project) / a.name / "weights" / "best.pt"
    if best.is_file():
        dst = ROOT / "models" / f"{a.name}_best.pt"
        dst.parent.mkdir(parents=True, exist_ok=True)
        import shutil
        shutil.copy2(best, dst)
        print(f"\n最佳权重已拷到 {dst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
