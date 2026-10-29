"""数据集体检：把"能不能用"变成一串数字。

只看图片张数没意义 —— 决定训练效果的是**每类框数、框面积分布、有没有数据泄漏**。
这个脚本输出一份体检报告，用来决定「直接开训」还是「先修数据」。

跑法：
    python scripts/inspect_dataset.py E:/datasets/fire_smoke_v1
"""
from __future__ import annotations

import hashlib
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

SPLITS = ("train", "valid", "test")


def md5(p: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.md5()
    with p.open("rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def main() -> int:
    root = Path(sys.argv[1] if len(sys.argv) > 1 else "E:/datasets/fire_smoke_v1")
    if not root.exists():
        print(f"[x] 不存在：{root}")
        return 1

    names = {}
    cf = root / "classes.txt"
    if cf.exists():
        names = dict(enumerate(cf.read_text(encoding="utf-8").strip().splitlines()))

    print("=" * 74)
    print(f"数据集体检：{root}")
    print(f"类别定义：{names or '（缺 classes.txt）'}")

    per_split: dict[str, dict] = {}
    all_hashes: dict[str, list[tuple[str, str]]] = defaultdict(list)
    global_box_area: dict[int, list[float]] = defaultdict(list)

    for split in SPLITS:
        img_dir, lbl_dir = root / split / "images", root / split / "labels"
        if not img_dir.exists():
            continue
        imgs = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
        info = {
            "images": len(imgs),
            "empty_label": 0,
            "missing_label": 0,
            "boxes": Counter(),
            "areas": defaultdict(list),
            "sizes": Counter(),
            "bad_coord": 0,
            "zero_area": 0,
        }
        for p in imgs:
            lp = lbl_dir / f"{p.stem}.txt"
            if not lp.exists():
                info["missing_label"] += 1
                continue
            lines = [ln for ln in lp.read_text(encoding="utf-8", errors="ignore").splitlines() if ln.strip()]
            if not lines:
                info["empty_label"] += 1
                continue
            for ln in lines:
                parts = ln.split()
                if len(parts) < 5:
                    info["bad_coord"] += 1
                    continue
                try:
                    cid = int(float(parts[0]))
                    cx, cy, w, h = (float(v) for v in parts[1:5])
                except ValueError:
                    info["bad_coord"] += 1
                    continue
                if not (0 <= cx <= 1 and 0 <= cy <= 1) or w <= 0 or h <= 0:
                    info["bad_coord"] += 1
                    continue
                if w * h < 1e-6:
                    info["zero_area"] += 1
                    continue
                info["boxes"][cid] += 1
                info["areas"][cid].append(w * h)
                global_box_area[cid].append(w * h)
            all_hashes[md5(p)].append((split, p.name))

        per_split[split] = info
        # 图片尺寸抽样（全量太慢，抽 200 张）
        try:
            from PIL import Image
            for p in imgs[:: max(1, len(imgs) // 200)][:200]:
                with Image.open(p) as im:
                    info["sizes"][f"{im.width}x{im.height}"] += 1
        except Exception:
            pass

    # ---------------- 输出 ----------------
    print("\n【1】规模与标注完整性")
    print(f"{'split':8s}{'图片':>8s}{'空标签':>8s}{'缺标签':>8s}{'坐标异常':>10s}{'零面积':>8s}")
    for split, d in per_split.items():
        print(f"{split:8s}{d['images']:>8d}{d['empty_label']:>8d}{d['missing_label']:>8d}"
              f"{d['bad_coord']:>10d}{d['zero_area']:>8d}")

    print("\n【2】每类框数（mAP 看的是框，不是图）")
    print(f"{'split':8s}" + "".join(f"{names.get(c, str(c)):>12s}" for c in sorted(names)))
    for split, d in per_split.items():
        print(f"{split:8s}" + "".join(f"{d['boxes'].get(c, 0):>12d}" for c in sorted(names)))
    total = {c: sum(d["boxes"].get(c, 0) for d in per_split.values()) for c in sorted(names)}
    print(f"{'合计':8s}" + "".join(f"{total[c]:>12d}" for c in sorted(names)))

    print("\n【3】框面积分布（占整图比例）—— 判断是不是小目标为主")
    for c in sorted(global_box_area):
        a = np.array(global_box_area[c])
        if not len(a):
            continue
        q = np.percentile(a, [10, 50, 90])
        small = float((a < 0.01).mean() * 100)      # 小于 1% 画面
        tiny = float((a < 0.0025).mean() * 100)     # 小于 0.25%（32px@640）
        print(f"  {names.get(c, c):6s} n={len(a):6d}  中位 {q[1]*100:.2f}%  "
              f"P10 {q[0]*100:.2f}%  P90 {q[2]*100:.2f}%  | <1%: {small:.0f}%  <0.25%: {tiny:.0f}%")

    print("\n【4】图片尺寸（抽样 200 张）")
    for split, d in per_split.items():
        top = ", ".join(f"{k}×{v}" for k, v in d["sizes"].most_common(3))
        print(f"  {split:6s} {top}")

    print("\n【5】跨集重复图（数据泄漏检查）")
    dup = {h: v for h, v in all_hashes.items() if len({s for s, _ in v}) > 1}
    print(f"  完全相同的图出现在不同 split：{len(dup)} 组")
    for h, v in list(dup.items())[:5]:
        print(f"    {v[:3]}")

    # 文件名前缀 → 推断是否按场景分组切分
    print("\n【6】切分方式检查（同场景相邻帧是否被拆到不同集）")
    prefix_split: dict[str, set] = defaultdict(set)
    for split in SPLITS:
        d = root / split / "images"
        if not d.exists():
            continue
        for p in d.iterdir():
            key = p.stem.rsplit("_", 1)[0]      # bothFireAndSmoke_CV000006 → bothFireAndSmoke
            prefix_split[key].add(split)
    mixed = {k: v for k, v in prefix_split.items() if len(v) > 1}
    print(f"  场景前缀：{len(prefix_split)} 类 | 跨 split 混放的前缀：{len(mixed)} 类")
    for k, v in list(mixed.items())[:5]:
        print(f"    {k} → {sorted(v)}")

    # 相邻帧泄漏：CV 编号相近的图是否分散
    nums: dict[str, set] = defaultdict(set)
    for split in SPLITS:
        d = root / split / "images"
        if not d.exists():
            continue
        for p in d.iterdir():
            stem = p.stem
            if "_CV" in stem:
                try:
                    n = int(stem.split("_CV")[1].split("_")[0])
                except ValueError:
                    continue
                nums[stem.split("_CV")[0]].add((n, split))
    leak = 0
    for k, vals in nums.items():
        vals = sorted(vals)
        for (n1, s1), (n2, s2) in zip(vals, vals[1:]):
            if s1 != s2 and abs(n2 - n1) <= 30:      # 同视频 30 帧内跨集 → 高风险泄漏
                leak += 1
    print(f"  编号相邻(≤30)却跨 split 的帧对：{leak} 处  ← 越多说明切分越随机")

    print("\n【7】结论")
    tr, va = per_split.get("train", {}), per_split.get("valid", {})
    ok = True
    for c in sorted(names):
        tc, vc = tr.get("boxes", {}).get(c, 0), va.get("boxes", {}).get(c, 0)
        tag = "✓" if tc >= 1500 and vc >= 300 else ("⚠" if tc >= 800 and vc >= 150 else "✗")
        if tc < 800 or vc < 150:
            ok = False
        print(f"  {tag} {names.get(c, c):6s} train {tc:5d} 框 / valid {vc:4d} 框"
              f"（底线：train≥1500、valid≥300）")
    if len(dup):
        print(f"  ⚠ 有 {len(dup)} 组重复图，建议去重")
    if leak > 5:
        print(f"  ⚠ 存在相邻帧跨集泄漏，mAP 会虚高；建议按编号段重切")
    print("  → 可以开训" if ok else "  → 建议先补数据/修切分")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
