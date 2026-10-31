"""把 Y-烟雾24000.zip 完整解压成 YOLO 目录结构（只取已划分好的 train/ 与 val/）。

zip 里同时存在 JPEGImages/Annotations/YOLOLabels（原始 VOC+YOLO 全套）
和 train/、val/（官方已划分副本）。后者已经是 YOLO 目录结构，直接取用，
避免重复占空间。
"""
import zipfile
from pathlib import Path

SRC = Path(r"E:\BaiduNetdiskDownload\Y-烟雾24000.zip")
DST = Path(r"D:\datasets\smoke_zip_full")

z = zipfile.ZipFile(SRC)
names = z.namelist()

targets = [n for n in names
           if not n.endswith("/")
           and (n.startswith("train/images/") or n.startswith("train/labels/")
                or n.startswith("val/images/") or n.startswith("val/labels/"))]
print(f"待解压 {len(targets)} 个文件 -> {DST}")

DST.mkdir(parents=True, exist_ok=True)
done = 0
for n in targets:
    out = DST / n
    out.parent.mkdir(parents=True, exist_ok=True)
    with z.open(n) as f, open(out, "wb") as g:
        g.write(f.read())
    done += 1
    if done % 2000 == 0:
        print(f"  {done}/{len(targets)}", flush=True)

for split in ("train", "val"):
    ni = len(list((DST / split / "images").glob("*.jpg")))
    nl = len(list((DST / split / "labels").glob("*.txt")))
    print(f"{split}: images {ni} | labels {nl}")

print("DONE")
