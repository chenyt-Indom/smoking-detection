"""把解压出来的大图组装成可直接训练的 YOLO 数据集。

为什么要单独一步而不是解压时就合并
---------------------------------
预标和原标分开存，是为了**复核前后都能用**：
  · 复核前：`--auto all`   全部预标当火（快，但带噪声，只适合跑一轮看趋势）
  · 复核后：`--reject bad.txt` 逐行列出要丢掉预标的文件名（人工标出来的）
这样同一个目录能出两版数据集，且**原始产物永不被覆盖**。

复核状态写在 dataset.yaml 的注释里 —— 训练完回头看指标，得知道标签是哪一档。

跑法：
    python scripts/build_smoke_zip_dataset.py                       # 只合并烟标，不要预标
    python scripts/build_smoke_zip_dataset.py --auto all             # 烟 + 全部火预标
    python scripts/build_smoke_zip_dataset.py --auto all --reject runs/prelabel_reject.txt
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="E:/datasets/smoke_zip_big")
    ap.add_argument("--out", default="E:/datasets/smoke_zip_ds")
    ap.add_argument("--auto", default="none", choices=["none", "all"],
                    help="火预标要不要进标签。none=只用原烟标（最保险）")
    ap.add_argument("--reject", default="",
                    help="每行一个 stem：这些人判为预标错的，丢掉它们的火预标")
    ap.add_argument("--val", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    src = Path(a.src)
    imgs = sorted((src / "images").glob("*.jpg"))
    if not imgs:
        raise SystemExit(f"没图：{src / 'images'}")

    reject = set()
    if a.reject and Path(a.reject).is_file():
        reject = {ln.strip().split(".")[0]
                  for ln in Path(a.reject).read_text(encoding="utf-8").splitlines()
                  if ln.strip() and not ln.startswith("#")}

    out = Path(a.out)
    (out / "labels").mkdir(parents=True, exist_ok=True)

    n_box = {0: 0, 1: 0}
    n_fire_img = 0
    kept = []
    for p in imgs:
        stem = p.stem
        merged: list[str] = []
        lab = src / "labels" / f"{stem}.txt"
        if lab.is_file():
            merged += [ln for ln in lab.read_text(encoding="utf-8").splitlines() if ln.strip()]
        if a.auto == "all" and stem not in reject:
            auto = src / "labels_auto" / f"{stem}.txt"
            if auto.is_file():
                fire = [ln for ln in auto.read_text(encoding="utf-8").splitlines() if ln.strip()]
                if fire:
                    n_fire_img += 1
                    merged += fire
        if not merged:                    # 一个框都没有的图不进训练集（空图另有负样本用途）
            continue
        for ln in merged:
            n_box[int(ln.split()[0])] += 1
        (out / "labels" / f"{stem}.txt").write_text("\n".join(merged) + "\n", encoding="utf-8")
        kept.append(p)

    rng = random.Random(a.seed)
    rng.shuffle(kept)
    nval = max(1, int(len(kept) * a.val))
    splits = {"val": kept[:nval], "train": kept[nval:]}
    for name, lst in splits.items():
        (out / f"{name}.txt").write_text(
            "\n".join(str(p.resolve()).replace("\\", "/") for p in lst) + "\n",
            encoding="utf-8")

    tag = ("火=自动预标，**未人工复核**，只适合跑一轮看趋势" if a.auto == "all"
           else "火=无标注（用这张表训练会把火当背景，别用于正式训练）")
    (out / "dataset.yaml").write_text(
        f"""# 外网烟雾集大图子集（{len(kept)} 张）
# 来源 Y-烟雾24000.zip，取最长边>=800 的 {len(imgs)} 张
# 烟标来自原数据集（已重映射 class 0=smoke → 1）
# {tag}
# 复核丢弃清单：{a.reject or '（无）'}  共丢 {len(reject)} 张的火预标
path: {out.resolve().as_posix()}
train: train.txt
val: val.txt
names:
  0: fire
  1: smoke
""", encoding="utf-8")

    print(f"图 {len(kept)} 张（原 {len(imgs)}，无框被剔除 {len(imgs) - len(kept)}）")
    print(f"  train {len(splits['train'])} / val {len(splits['val'])}")
    print(f"  框数：火 {n_box[0]}（{n_fire_img} 张图）  烟 {n_box[1]}")
    print(f"  模式 {a.auto}  丢弃预标 {len(reject)} 张")
    print(f"  输出 {out / 'dataset.yaml'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
