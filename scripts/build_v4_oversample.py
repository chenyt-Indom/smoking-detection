"""构建 fire_smoke_v4：在 v3 基础上对「含 fire 的图」做过采样（硬链接，不占额外磁盘）。

为什么这么做
------------
v3 诊断结论：fire 框数不少（27,137）但**面积只有烟的 1/8**（中位 0.84% vs 6.71%），
且含 fire 的图只有 13,476 张（smoke 20,612 张）。模型对 fire 的召回只有 0.52~0.59。

过采样不能把火变大，但能让每个 epoch 里 fire 出现得更频繁，
配合更大的输入尺寸（960）一起用，才是对症的。

做法
----
- train：v3 全部图片 + 含 fire 的图片再加一份（文件名加 `_fdup` 后缀，标注同步）
- valid / test：**原样硬链**，不动分布（评测口径必须和 v3 一致，才能对比）
- 全部用 os.link 硬链接，只在同盘生效，不消耗额外空间

用法：python scripts/build_v4_oversample.py
"""
from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

V3 = Path(r"E:/datasets/fire_smoke_v3")
V4 = Path(r"E:/datasets/fire_smoke_v4")


def link(src: Path, dst: Path) -> None:
    if dst.exists():
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def has_class(label: Path, cls: int) -> bool:
    if not label.exists():
        return False
    for line in label.read_text(encoding="utf-8").splitlines():
        t = line.split()
        if t and int(float(t[0])) == cls:
            return True
    return False


def main() -> int:
    if not V3.exists():
        print(f"[x] 找不到源数据集 {V3}")
        return 1

    for split in ("train", "valid", "test"):
        (V4 / split / "images").mkdir(parents=True, exist_ok=True)
        (V4 / split / "labels").mkdir(parents=True, exist_ok=True)

    total = fire_dup = 0
    for img in sorted((V3 / "train" / "images").glob("*")):
        lbl = V3 / "train" / "labels" / (img.stem + ".txt")
        link(img, V4 / "train" / "images" / img.name)
        if lbl.exists():
            link(lbl, V4 / "train" / "labels" / lbl.name)
        total += 1
        if has_class(lbl, 0):
            dup_img = V4 / "train" / "images" / f"{img.stem}_fdup{img.suffix}"
            dup_lbl = V4 / "train" / "labels" / f"{lbl.stem}_fdup.txt"
            link(img, dup_img)
            if lbl.exists():
                link(lbl, dup_lbl)
            fire_dup += 1

    for split in ("valid", "test"):
        for img in sorted((V3 / split / "images").glob("*")):
            lbl = V3 / split / "labels" / (img.stem + ".txt")
            link(img, V4 / split / "images" / img.name)
            if lbl.exists():
                link(lbl, V4 / split / "labels" / lbl.name)

    (V4 / "dataset.yaml").write_text(
        f"path: {V4.as_posix()}\n"
        "train: train/images\n"
        "val: valid/images\n"
        "test: test/images\n"
        "nc: 2\n"
        "names:\n"
        "  - fire\n"
        "  - smoke\n",
        encoding="utf-8",
    )
    (V4 / "classes.txt").write_text("fire\nsmoke\n", encoding="utf-8")

    n_train = len(list((V4 / "train" / "images").glob("*")))
    n_valid = len(list((V4 / "valid" / "images").glob("*")))
    n_test = len(list((V4 / "test" / "images").glob("*")))
    print(f"v3 train 原图 {total} 张，其中含 fire {fire_dup} 张 → 过采样后 train {n_train} 张")
    print(f"valid {n_valid} 张 / test {n_test} 张（与 v3 完全一致，仅硬链）")
    print(f"数据配置：{V4 / 'dataset.yaml'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
