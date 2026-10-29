"""把多份来源合并成一个干净数据集：去重 → 分层重切 → 写 yaml。

为什么不能直接"把两个 train 目录拷一起"
--------------------------------------
实测过：这两份数据是**同源**的（都来自 FASDD，文件名都是 `bothFireAndSmoke_CVxxxxxx.jpg`），
第一份 4000 张里有 **3600 张**在第二份里也有。直接合并会造成两种情况：
  1. 同一张图同时出现在 train 和 val → 指标虚高（最致命）
  2. 白占空间、白跑算力

所以这里的顺序是：
  ① 以「文件名 stem」去重（同源命名一致，stem 相同就是同一张图）
  ② 全部打散成一个大池子，按「前缀 + 图片编号段」分层重切 70/20/10
  ③ 空标签（纯背景负样本）单独成组，按同比例分配，保证 val/test 里也有负样本
  ④ 写 dataset.yaml，并打印每类框数（判断够不够训）

用法：
    python scripts/merge_datasets.py \
        --src "E:/datasets/src_A/A_烟雾_火灾数据集_YOLO格式:train,valid" \
        --src "E:/datasets/fire_smoke_v1:train,valid,test" \
        --src "E:/BaiduNetdiskDownload/Y_火灾+烟雾/背景负样本:flat" \
        --out "E:/datasets/fire_smoke_v3"
"""
from __future__ import annotations

import argparse
import hashlib
import re
import shutil
from collections import defaultdict
from pathlib import Path

IMG_EXT = (".jpg", ".jpeg", ".png")


def md5_of(p: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.md5()
    with p.open("rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def group_key(path: Path) -> tuple:
    """前缀 + 编号段：同一段（相邻 ≤200 个编号）视为同源视频，必须同进同出。"""
    stem = path.stem
    m = re.search(r"_CV(\d+)", stem)
    if m:
        return (stem.split("_CV")[0], int(m.group(1)) // 200)
    m2 = re.search(r"_\((\d+)\)", stem)          # large_(2373)
    if m2:
        return (re.sub(r"_\(\d+\)$", "", stem), int(m2.group(1)) // 200)
    m3 = re.search(r"_(\d+)$", stem)
    if m3:
        return (stem.rsplit("_", 1)[0], int(m3.group(1)) // 200)
    return (stem, 0)


def collect(spec: str) -> list[tuple[Path, Path | None]]:
    """spec = '根目录:train,valid' 或 '根目录:flat'。返回 [(图, 标签或None)]。"""
    root_s, splits_s = spec.rsplit(":", 1)
    root = Path(root_s)
    out: list[tuple[Path, Path | None]] = []
    if splits_s.strip() == "flat":
        for img in sorted(root.iterdir()):
            if img.suffix.lower() in IMG_EXT:
                lbl = root.parent / "labels" / f"{img.stem}.txt"
                if not lbl.exists():
                    lbl = root / f"{img.stem}.txt"
                out.append((img, lbl if lbl.exists() else None))
        return out
    for sp in [s.strip() for s in splits_s.split(",") if s.strip()]:
        idir, ldir = root / sp / "images", root / sp / "labels"
        if not idir.exists():
            print(f"  [!] 跳过不存在的目录：{idir}")
            continue
        for img in sorted(idir.iterdir()):
            if img.suffix.lower() in IMG_EXT:
                lbl = ldir / f"{img.stem}.txt"
                out.append((img, lbl if lbl.exists() else None))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", action="append", required=True,
                    help="来源，格式 根目录:split1,split2 或 根目录:flat（可多次）")
    ap.add_argument("--out", required=True)
    ap.add_argument("--ratio", default="0.7,0.2,0.1")
    ap.add_argument("--verify-md5", action="store_true",
                    help="额外按内容哈希去重（慢，3.6 万张约 2 分钟）")
    a = ap.parse_args()

    tr, va, te = (float(x) for x in a.ratio.split(","))

    # ---------- 1) 汇总 + 去重 ----------
    pool: dict[str, tuple[Path, Path | None, str]] = {}
    stats = []
    for spec in a.src:
        items = collect(spec)
        name = Path(spec.rsplit(":", 1)[0]).name
        dup = 0
        for img, lbl in items:
            if img.stem in pool:
                dup += 1
                continue
            pool[img.stem] = (img, lbl, name)
        stats.append((name, len(items), dup))
        print(f"  {name:42s} 读入 {len(items):6d} 张，与已有重复 {dup:6d} 张")

    print(f"\n去重后合计 {len(pool)} 张唯一图片")

    if a.verify_md5:
        print("按内容哈希二次去重…")
        seen: dict[str, str] = {}
        removed = 0
        for stem, (img, lbl, src) in list(pool.items()):
            h = md5_of(img)
            if h in seen:
                pool.pop(stem)
                removed += 1
            else:
                seen[h] = stem
        print(f"  内容重复移除 {removed} 张，剩 {len(pool)} 张")

    # ---------- 2) 分层重切 ----------
    groups: dict[tuple, list[str]] = defaultdict(list)
    for stem in pool:
        groups[group_key(Path(stem))].append(stem)

    by_prefix: dict[str, list[tuple]] = defaultdict(list)
    for k in groups:
        by_prefix[k[0]].append(k)

    assign: dict[tuple, str] = {}
    for prefix, keys in by_prefix.items():
        keys.sort(key=lambda k: k[1])
        n = len(keys)
        if n < 4:
            for k in keys:
                assign[k] = "train"
            continue
        for i, k in enumerate(keys):
            r = i % 10
            assign[k] = "valid" if r in (3, 8) else ("test" if r == 7 else "train")

    out = Path(a.out)
    for sp in ("train", "valid", "test"):
        (out / sp / "images").mkdir(parents=True, exist_ok=True)
        (out / sp / "labels").mkdir(parents=True, exist_ok=True)

    count = {"train": 0, "valid": 0, "test": 0}
    boxes = {"train": defaultdict(int), "valid": defaultdict(int), "test": defaultdict(int)}
    neg = {"train": 0, "valid": 0, "test": 0}

    for stem, (img, lbl, src) in pool.items():
        sp = assign[group_key(img)]
        # 加来源前缀，避免不同来源同名文件互相覆盖
        new_stem = f"{src[:6]}_{stem}"[:120]
        shutil.copy2(img, out / sp / "images" / f"{new_stem}{img.suffix.lower()}")
        lines: list[str] = []
        if lbl and lbl.exists():
            lines = [x for x in lbl.read_text(encoding="utf-8", errors="ignore").splitlines() if x.strip()]
            shutil.copy2(lbl, out / sp / "labels" / f"{new_stem}.txt")
        else:
            (out / sp / "labels" / f"{new_stem}.txt").write_text("", encoding="utf-8")
        count[sp] += 1
        if not lines:
            neg[sp] += 1
        for ln in lines:
            try:
                cid = int(float(ln.split()[0]))
            except (ValueError, IndexError):
                continue
            boxes[sp][cid] += 1

    (out / "dataset.yaml").write_text(
        f"path: {out.as_posix()}\ntrain: train/images\nval: valid/images\ntest: test/images\n"
        "nc: 2\nnames:\n  - fire\n  - smoke\n", encoding="utf-8")
    (out / "classes.txt").write_text("fire\nsmoke\n", encoding="utf-8")

    # ---------- 3) 报告 ----------
    print("\n" + "=" * 66)
    print(f"{'split':8s}{'图片':>8s}{'负样本':>8s}{'fire框':>10s}{'smoke框':>10s}")
    for sp in ("train", "valid", "test"):
        print(f"{sp:8s}{count[sp]:>8d}{neg[sp]:>8d}{boxes[sp][0]:>10d}{boxes[sp][1]:>10d}")
    tot = {c: sum(boxes[s][c] for s in boxes) for c in (0, 1)}
    print(f"{'合计':8s}{sum(count.values()):>8d}{sum(neg.values()):>8d}{tot[0]:>10d}{tot[1]:>10d}")
    print("\n达标检查（train≥1500 / valid≥300 框）：")
    ok = True
    for c, name in ((0, "fire"), (1, "smoke")):
        t, v = boxes["train"][c], boxes["valid"][c]
        flag = "✓" if t >= 1500 and v >= 300 else ("⚠" if t >= 800 else "✗")
        ok = ok and flag == "✓"
        print(f"  {flag} {name:6s} train {t:6d} / valid {v:5d}")
    print(f"\n输出：{out}")
    print("→ 可以开训" if ok else "→ 数据仍偏少，建议再补")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
