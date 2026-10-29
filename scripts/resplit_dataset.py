"""按「来源视频编号段」重新切分，消除相邻帧泄漏。

为什么必须重切
------------
买来的数据集是按**帧**随机分成 train/valid/test 的。体检发现：
编号相差 ≤30 的帧有 **897 处**被拆到不同 split —— 同一段视频的前后帧
几乎一样，这等于把答案提前泄漏给验证集，mAP 会虚高 10~20 个点。
这种分数在答辩/交付时一问就露馅。

做法
----
把 `前缀_CV编号` 里的编号排序，按 **连续编号段**（默认每 200 帧一段）整体
分配到 train/valid/test（80/10/10）。同一段视频的所有帧只会落在同一个 split。
文件名里没有 CV 编号的（other_xxx、middle_xxx 这类负样本）按文件名字典序
每 N 个取一段，同样整段分配。

输出到新目录，不动原始数据。
"""
from __future__ import annotations

import re
import shutil
import sys
from collections import defaultdict
from pathlib import Path

SPLITS = ("train", "valid", "test")
RATIO = (0.8, 0.1, 0.1)          # train / valid / test
SEG = 200                        # 每段的相邻编号跨度


def seg_of(num: int, seg: int = SEG) -> int:
    return num // seg


def main() -> int:
    src = Path(sys.argv[1] if len(sys.argv) > 1 else "E:/datasets/fire_smoke_v1")
    dst = Path(sys.argv[2] if len(sys.argv) > 2 else "E:/datasets/fire_smoke_v2")
    if not src.exists():
        print(f"[x] 源目录不存在：{src}")
        return 1

    # 1) 汇总所有图（以 train/valid/test 里实际存在的为准）
    items = []       # (path, group_key)
    for split in SPLITS:
        d = src / split / "images"
        if not d.exists():
            continue
        for p in sorted(d.iterdir()):
            if p.suffix.lower() not in (".jpg", ".jpeg", ".png"):
                continue
            m = re.search(r"_CV(\d+)", p.stem)
            if m:
                key = (p.stem.split("_CV")[0], seg_of(int(m.group(1))))
            else:
                # other_996 / middle_123 之类：按编号段，没有编号就归到同一段
                m2 = re.search(r"_(\d+)$", p.stem)
                if m2:
                    key = (p.stem.rsplit("_", 1)[0], seg_of(int(m2.group(1))))
                else:
                    key = (p.stem, 0)
            items.append((p, key))

    # 2) 以「段」为单位整体分配
    groups: dict[tuple, list[Path]] = defaultdict(list)
    for p, k in items:
        groups[k].append(p)

    # ⚠️ 不能"先填满 train、剩下给 val/test"：
    # 段与类别强相关（fire / smoke / both / other 各自独立编号），
    # 那样切出来 val 可能整片没烟、test 可能全是负样本（踩过：valid 402 fire 框 0 烟框）。
    # 正确做法：**按前缀分层**，每个前缀内部把自己的段按 8:1:1 轮转切。
    by_prefix: dict[str, list[tuple]] = defaultdict(list)
    for k in groups:
        by_prefix[k[0]].append(k)

    assign: dict[tuple, str] = {}
    for prefix, segs in by_prefix.items():
        segs.sort(key=lambda k: k[1])
        n = len(segs)
        if n < 4:                       # 段太少没法分层，整体进 train（多是有编号的零星负样本）
            for k in segs:
                assign[k] = "train"
            continue
        for i, k in enumerate(segs):
            r = i % 10
            # valid 20% / test 10% / train 70%：
            # 4000 张的数据集里，valid 少了框数不够（<300 框时 mAP 抖动 ±5 个点）
            assign[k] = "valid" if r in (3, 8) else ("test" if r == 7 else "train")

    keys = sorted(groups.keys())
    total = sum(len(v) for v in groups.values())
    cnt = {"train": 0, "valid": 0, "test": 0}
    for k in keys:
        cnt[assign[k]] += len(groups[k])

    # 3) 复制（图 + 同名标签）
    for s in SPLITS:
        (dst / s / "images").mkdir(parents=True, exist_ok=True)
        (dst / s / "labels").mkdir(parents=True, exist_ok=True)

    moved = {"train": 0, "valid": 0, "test": 0}
    for p, k in items:
        s = assign[k]
        lp = src / p.parent.parent.name / "labels" / f"{p.stem}.txt"
        if not lp.exists():
            # 换一种找法：同级 labels 目录
            lp = p.parent.parent / "labels" / f"{p.stem}.txt"
        shutil.copy2(p, dst / s / "images" / p.name)
        if lp.exists():
            # 注意括号：Path / "labels" / (p.stem + ".txt")，
            # 写成 Path / "labels" / p.stem + ".txt" 会先做除法再加字符串 → TypeError
            shutil.copy2(lp, dst / s / "labels" / f"{p.stem}.txt")
        moved[s] += 1

    # 4) 顺手把 classes.txt / dataset.yaml 写对
    (dst / "classes.txt").write_text("fire\nsmoke\n", encoding="utf-8")
    (dst / "dataset.yaml").write_text(
        f"path: {dst.as_posix()}\n"
        "train: train/images\nval: valid/images\ntest: test/images\n"
        "nc: 2\nnames:\n  - fire\n  - smoke\n",
        encoding="utf-8",
    )

    print(f"源：{src}（{total} 张）")
    print(f"新：{dst}")
    for s in SPLITS:
        print(f"  {s:6s} {moved[s]:5d} 张")
    print(f"段数 {len(keys)}（每段 ≤{SEG} 个连续编号），同段只进一个 split")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
