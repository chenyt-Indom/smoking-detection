"""把看板上标注的误报（硬负样本）并进训练集，生成一个新的 data.yaml。

思路：**不复制图片**，只改 yaml 的 train 列表。
--------------------------------------------
ultralytics 的 `train:` 支持给一个路径列表，多个目录会被一起扫。
所以硬负样本就留在 `hard_negatives/` 里原地用，不占第二份磁盘、也不会
出现"训练集里有一份、源目录改了一份"的漂移。

为什么硬负样本的标注是**空文件**
--------------------------------
YOLO 约定：一张图对应一个同名 .txt，里面一行都没有 = 这张图没有任何目标。
这正是"误报帧"要表达的意思 —— 这里没有火，也没有烟。比造一个负类干净：
不用改 nc，不用动 dataset 结构，也不会把"负类"和"背景"混成一个概念。

跑法
----
    python scripts/merge_hard_negatives.py                  # 只看统计，不写文件
    python scripts/merge_hard_negatives.py --apply           # 真的写 yaml
    python scripts/merge_hard_negatives.py --apply --oversample 3   # 负样本重复 3 次

为什么要 oversample
-------------------
41853 张训练图里塞 50 张负样本，等于没塞。把它们在 train 列表里重复出现
几次，是 ultralytics 里最简单有效的加权方式（不去重、只重复路径）。
建议比例：负样本数量占到训练集 0.5%~2% 起步，误报没压住再加。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
HN = ROOT / "hard_negatives"
BASE_YAML = Path("E:/datasets/fire_smoke_v5/dataset.yaml")


def read_yaml_simple(p: Path) -> dict:
    """只解析这个项目用到的那几个键，不为了一个 yaml 去装 pyyaml。"""
    out: dict = {}
    for ln in p.read_text(encoding="utf-8").splitlines():
        s = ln.split("#", 1)[0].rstrip()
        if not s or ":" not in s:
            continue
        k, v = s.split(":", 1)
        out[k.strip()] = v.strip()
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hn", default=str(HN), help="硬负样本目录")
    ap.add_argument("--base", default=str(BASE_YAML), help="原始 dataset.yaml")
    ap.add_argument("--out", default="", help="输出 yaml，默认 <base>_hard.yaml")
    ap.add_argument("--apply", action="store_true", help="不写文件只做检查")
    ap.add_argument("--oversample", type=int, default=1, help="负样本目录在 train 里出现几次")
    a = ap.parse_args()

    hn = Path(a.hn)
    img_dir, lbl_dir = hn / "images", hn / "labels"
    labels = sorted(lbl_dir.glob("*.txt")) if lbl_dir.is_dir() else []
    imgs = sorted(img_dir.glob("*.jpg")) if img_dir.is_dir() else []
    stems_l = {p.stem for p in labels}
    stems_i = {p.stem for p in imgs}
    paired = sorted(stems_i & stems_l)

    print(f"硬负样本目录 {hn}")
    print(f"  图片 {len(imgs)}  标注 {len(labels)}  成对 {len(paired)}")
    missing_lbl = sorted(stems_i - stems_l)
    missing_img = sorted(stems_l - stems_i)
    if missing_lbl:
        print(f"  ⚠️ {len(missing_lbl)} 张图没有标注文件，会被当**有目标但没框**，跳过："
              f"{missing_lbl[:3]}")
    if missing_img:
        print(f"  ⚠️ {len(missing_img)} 个标注没有对应图，跳过：{missing_img[:3]}")

    # 标了非空的要拦下来：硬负样本必须是空标注，否则它会自己制造误报
    nonempty = []
    for st in paired:
        if (lbl_dir / f"{st}.txt").read_text(encoding="utf-8").strip():
            nonempty.append(st)
    if nonempty:
        print(f"  ❌ {len(nonempty)} 个标注不是空的，这些不是负样本：{nonempty[:5]}")
        print("     （如果那一帧真的还有火/烟，它本来就不该标成误报）")
        return 2

    if not paired:
        print("\n还没有任何误报。先在 5050 看板上点几次「误报」再来。")
        return 0

    rep = hn / "reports.jsonl"
    if rep.is_file():
        rows = [json.loads(l) for l in rep.read_text(encoding="utf-8").splitlines() if l.strip()]
        fa = sum(1 for r in rows if r.get("verdict") == "false_alarm")
        ta = sum(1 for r in rows if r.get("verdict") == "true_alarm")
        print(f"\n人工反馈：误报 {fa} 条 / 属实 {ta} 条"
              f"（确认率 {ta / max(fa + ta, 1):.0%}）")

    base = Path(a.base)
    if not base.is_file():
        print(f"\n❌ 找不到原始 yaml：{base}")
        return 2
    cfg = read_yaml_simple(base)
    root = Path(cfg.get("path", "E:/datasets/fire_smoke_v5"))
    out = Path(a.out) if a.out else base.with_name(base.stem + "_hard.yaml")

    # train 用**列表**：多个目录一起扫。硬负样本原地引用，不复制。
    train_rel = []
    for v in [cfg.get("train", "train/images")]:
        train_rel.append(v)
    hn_rel = str(img_dir.resolve().relative_to(root.resolve())) \
        if img_dir.resolve().is_relative_to(root.resolve()) else str(img_dir.resolve())
    hn_rel = hn_rel.replace("\\", "/")       # yaml 里统一用正斜杠，Windows 也别用反斜杠
    for _ in range(max(1, a.oversample)):
        train_rel.append(hn_rel)

    text = [
        "# 由 scripts/merge_hard_negatives.py 生成，别手改（改了会被下次覆盖）",
        f"# 基线：{base}",
        f"# 追加 {len(paired)} 张硬负样本（空标注），oversample={max(1, a.oversample)}",
        f"path: {root.as_posix()}",
        "train:",
        *[f"  - {t}" for t in train_rel],
        f"val: {cfg.get('val', 'valid/images')}",
        f"test: {cfg.get('test', 'test/images')}",
        f"nc: {cfg.get('nc', '2')}",
        "names:",
    ]
    names = []
    if "names" in cfg:                       # 单行写法 names: ['fire','smoke'] 之类
        names = [n.strip(" []'\"") for n in cfg["names"].strip("[]").split(",") if n.strip()]
    text += [f"  - {n}" for n in names] or ["  - fire", "  - smoke"]

    print("\n生成内容：")
    print("\n".join("  " + ln for ln in text))
    if not a.apply:
        print("\n（未写盘。确认无误后加 --apply）")
        return 0
    out.write_text("\n".join(text) + "\n", encoding="utf-8")
    print(f"\n✅ 已写入 {out}")
    print(f"   训练时用：yolo train data={out} ...")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
