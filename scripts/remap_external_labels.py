"""把外部烟雾集的标签从「class 0 = smoke」重映射为「class 1 = smoke」。

背景
----
外部集自带 classes.txt 只有一行 smoke，所以它的标签里 class 0 表示烟。
而本项目数据集约定 class 0 = fire、class 1 = smoke。
不重映射就直接混用，3 万多个烟框会全变成 fire 标签。

做法
----
新建 D:/datasets/smoke_zip_remap/{train,val}/labels/ 存放重映射后的标签；
images 用目录联接指向原图，不复制 789MB 图片。
"""
from pathlib import Path

SRC = Path(r"D:/datasets/smoke_zip_full")
DST = Path(r"D:/datasets/smoke_zip_remap")

n_file = n_box = n_empty = n_bad = 0
for split in ("train", "val"):
    src_dir = SRC / split / "labels"
    dst_dir = DST / split / "labels"
    dst_dir.mkdir(parents=True, exist_ok=True)
    for lp in src_dir.glob("*.txt"):
        out = []
        for line in lp.read_text(encoding="utf-8").splitlines():
            v = line.split()
            if not v:
                continue
            if len(v) < 5:
                n_bad += 1
                continue
            c = int(float(v[0]))
            if c != 0:
                n_bad += 1
                continue
            out.append("1 " + " ".join(v[1:5]))
            n_box += 1
        if not out:
            n_empty += 1
        (dst_dir / lp.name).write_text("\n".join(out) + ("\n" if out else ""),
                                       encoding="utf-8")
        n_file += 1

print(f"重映射标签 {n_file} 个文件 | 转换 {n_box} 个框 | 空标签 {n_empty} | 异常行 {n_bad}")
for split in ("train", "val"):
    print(f"  {split}: {len(list((DST / split / 'labels').glob('*.txt')))} 个标签")
