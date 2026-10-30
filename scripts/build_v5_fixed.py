"""构建 fire_smoke_v5：在 v3 基础上换用「标注体检 + 修正」后的标签，并保留 v4 的火图过采样。

和 v4 的唯一差别
----------------
v4：标签原样照搬 v3（含 7,612 个有问题的 fire 框里被判定要改的那部分）
v5：train 里被判为「框内无火」的框删掉、判为「框太松」的框收紧到火区外接框，
    其余框、smoke 类、valid/test 全部原样。
    这样 v4 vs v5 就是「同一套训练配方、只有标签不同」的干净对照，
    第 4 轮就能看出差别，不必等 30 轮跑完。

为什么 valid / test 一律不动
----------------------------
评测口径一变，历史指标全部作废，v3/v4/v5 就没法比了。只修train。

做法
----
- train：v3 全部图片；含 fire 的图再硬链一份（`_fdup` 后缀），与 v4 完全一致
- 标注：train 里若有修正版就用修正版，否则硬链原标注；valid/test 全硬链
- 全部 os.link，同盘生效，不占额外磁盘

用法：
    python scripts/build_v5_fixed.py --dry-run
    python scripts/build_v5_fixed.py
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

V3 = Path(r"E:/datasets/fire_smoke_v3")
V5 = Path(r"E:/datasets/fire_smoke_v5")
FIX = Path(r"D:/视觉安防系统/runs/fire_label_fix/labels")


def link(src: Path, dst: Path) -> None:
    if dst.exists():
        dst.unlink()
    try:
        os.link(src, dst)
    except OSError:
        import shutil
        shutil.copy2(src, dst)


def has_fire(lb: Path) -> bool:
    if not lb.exists():
        return False
    for line in lb.read_text(encoding="utf-8", errors="ignore").splitlines():
        t = line.split()
        if t and abs(float(t[0])) < 0.5:
            return True
    return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    if not FIX.exists():
        print(f"[x] 找不到修正标注目录 {FIX}\n    先跑 propose_fire_fixes.py --write-labels")
        return 1
    fixed = {p.stem: p for p in FIX.glob("*.txt")}
    print(f"修正标注 {len(fixed)} 个")

    stat = {}
    for split in ("train", "valid", "test"):
        src_img = V3 / split / "images"
        src_lb = V3 / split / "labels"
        dst_img = V5 / split / "images"
        dst_lb = V5 / split / "labels"
        if not a.dry_run:
            dst_img.mkdir(parents=True, exist_ok=True)
            dst_lb.mkdir(parents=True, exist_ok=True)

        imgs = sorted(src_img.glob("*.jpg"))
        n_use_fix = n_dup = 0
        for p in imgs:
            lb = src_lb / (p.stem + ".txt")
            dst_l = dst_lb / (p.stem + ".txt")
            used_fix = split == "train" and p.stem in fixed
            if not a.dry_run:
                link(p, dst_img / p.name)
                if used_fix:
                    link(fixed[p.stem], dst_l)
                    n_use_fix += 1
                elif lb.exists():
                    link(lb, dst_l)
            elif used_fix:
                n_use_fix += 1

            # 火图过采样：与 v4 一致
            if split == "train" and has_fire(fixed[p.stem] if used_fix else lb):
                n_dup += 1
                if not a.dry_run:
                    link(p, dst_img / f"{p.stem}_fdup.jpg")
                    src_l = fixed[p.stem] if used_fix else lb
                    if src_l.exists():
                        link(src_l, dst_lb / f"{p.stem}_fdup.txt")

        stat[split] = (len(imgs), n_use_fix, n_dup)
        print(f"{split:<6} 图片 {len(imgs):>6}  用修正标注 {n_use_fix:>5}  火图复制 {n_dup:>6}"
              f"  → 合计 {len(imgs) + n_dup}")

    if a.dry_run:
        print("\n[dry-run] 没有写入任何文件")
        return 0
    print(f"\n数据集：{V5}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
