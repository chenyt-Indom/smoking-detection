"""从 Y-烟雾24000.zip 解压出纯烟单类数据集（验证「烟/火拆分」方案用）。

为什么单独拆一套
----------------
这批数据只有 smoke 标注、**图里的火是没标的**。在双类模型里放进训练集，
火区域会被当成背景，等于教模型「火不是火」——所以之前只能挑出 588 张
「整张图用 conf 0.15 都检不出火」的用，剩下 670 张全废掉。

拆成单类烟模型后，判据只有「有烟 / 无烟」，火在不在完全不影响。
**24,574 张可以全部用上。**

输出结构（ultralytics 标准布局）
    D:/datasets/smoke_all/
        images/{train,val}/*.jpg
        labels/{train,val}/*.txt   ← class 0 = smoke
        dataset.yaml

用法：
    python scripts/extract_smoke_all.py
    python scripts/extract_smoke_all.py --min-side 0      # 不过滤尺寸
    python scripts/extract_smoke_all.py --min-side 320    # 只要 >=320 的
"""
from __future__ import annotations

import argparse
import zipfile
from collections import Counter
from pathlib import Path

ZIP = Path("E:/BaiduNetdiskDownload/Y-烟雾24000.zip")
OUT = Path("D:/datasets/smoke_all")


def jpeg_size(buf: bytes) -> tuple[int, int] | None:
    """不解码只读 JPEG 头拿尺寸 —— 2 万多张全解码太慢。"""
    i, n = 2, len(buf)
    while i < n - 9:
        if buf[i] != 0xFF:
            i += 1
            continue
        m = buf[i + 1]
        if m in (0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF):
            h = (buf[i + 5] << 8) | buf[i + 6]
            w = (buf[i + 7] << 8) | buf[i + 8]
            return w, h
        if m in (0xD8, 0xD9) or 0xD0 <= m <= 0xD7:
            i += 2
            continue
        ln = (buf[i + 2] << 8) | buf[i + 3]
        i += 2 + ln
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-side", type=int, default=0,
                    help="最短边下限，小于它的跳过（0=不过滤）")
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()
    out = Path(a.out)

    for sp in ("train", "val"):
        (out / "images" / sp).mkdir(parents=True, exist_ok=True)
        (out / "labels" / sp).mkdir(parents=True, exist_ok=True)

    stats = Counter()
    kept = Counter()
    with zipfile.ZipFile(ZIP) as z:
        names = set(z.namelist())
        for sp in ("train", "val"):
            jpgs = sorted(n for n in names if n.startswith(f"{sp}/") and n.endswith(".jpg"))
            print(f"[{sp}] {len(jpgs)} 张", flush=True)
            for i, name in enumerate(jpgs, 1):
                stem = Path(name).name
                lab_zip = f"{sp}/labels/{Path(stem).stem}.txt"
                if lab_zip not in names:
                    stats["缺标签"] += 1
                    continue

                buf = z.read(name)
                if a.min_side > 0:
                    sz = jpeg_size(buf)
                    if not sz:
                        stats["尺寸解析失败"] += 1
                        continue
                    if min(sz) < a.min_side:
                        stats["尺寸过小"] += 1
                        continue

                # 原标签 class 0 = smoke，与单类烟模型一致，不需要重映射
                lab = z.read(lab_zip).decode("utf-8", "ignore").strip()
                if not lab:
                    stats["空标签"] += 1
                    continue
                # 校验：只允许 class 0，坐标在 [0,1]
                ok = True
                for line in lab.splitlines():
                    v = line.split()
                    if len(v) < 5 or int(float(v[0])) != 0:
                        ok = False
                        break
                    if not all(0.0 <= float(x) <= 1.0 for x in v[1:5]):
                        ok = False
                        break
                if not ok:
                    stats["标签异常"] += 1
                    continue

                (out / "images" / sp / stem).write_bytes(buf)
                (out / "labels" / sp / f"{Path(stem).stem}.txt").write_text(
                    lab + "\n", encoding="utf-8")
                kept[sp] += 1
                if i % 2000 == 0:
                    print(f"  {i}/{len(jpgs)} …", flush=True)

    yaml = f"""# 外部烟雾集（Y-烟雾24000）纯烟单类版
# 来源：E:/BaiduNetdiskDownload/Y-烟雾24000.zip 的 train/ 与 val/（原划分）
# 只保留 smoke 一类。原数据集的火**没有标注**，所以这份数据只适合训单类烟模型，
# 放进双类训练集会把火教成背景。
# 过滤：最短边 >= {a.min_side if a.min_side > 0 else '不限'}
path: {out.as_posix()}
train: images/train
val: images/val
nc: 1
names:
  0: smoke
"""
    (out / "dataset.yaml").write_text(yaml, encoding="utf-8")

    print(f"\n完成 → {out.as_posix()}")
    print(f"  images/train {kept['train']} | images/val {kept['val']}")
    if stats:
        print("  跳过统计:", dict(stats))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
