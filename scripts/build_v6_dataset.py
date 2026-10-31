"""组装 v6 数据集：v5 全量 + 外部烟雾集的**无火子集**（过采样）。

为什么只取无火子集
------------------
外部集 Y-烟雾24000 只有 smoke 标注，**图里的火是没标的**。双类训练时，
没标的火区域会被当成背景，等于教模型「火不是火」。

而「用高置信度预标把火补上」这条路也走不通 —— 实测 v5 在 **conf 0.83**
下仍把明亮白烟判成火（smoke_001590，面积 1.93%，框内是屋侧冒出的白烟）。
补上去的框里有一批是错的，训练时等于教模型「白烟也是火」，以后白烟误报只会更多。

所以只保留**整张图用 conf 0.15 都检不出任何火**的那部分（754 张），让它只贡献烟。

为什么过采样
------------
754 张放进 v5 的 41,853 张里只占 1.8%，训一轮看不出差别。而模型在这类
「远处大烟」场景的召回只有 37%（v5 自测 IoU0.5），是明确短板，所以复制多份
提高它在每个 epoch 里被采到的频率。每份独立采样、增强不同，不是简单重复。

标签路径注意
------------
ultralytics 靠「把 /images/ 换成 /labels/」找标签，所以清单里的路径必须落在
带同名 labels/ 的目录下。smoke_zip_ds/images 是指向 smoke_zip_big/images 的
目录联接，而 smoke_zip_ds/labels/ 才是**合并后**（含预标火框）的那份 ——
无火子集里两者等价，但换成有火的图就会读到错的标签，别混。

用法：
    python scripts/build_v6_dataset.py
    python scripts/build_v6_dataset.py --nofire-repeat 5
"""
from __future__ import annotations

import argparse
from pathlib import Path

V5 = Path("D:/datasets/fire_smoke_v5")
ZIPDS = Path("D:/datasets/smoke_zip_ds")


def read_list(p: Path) -> list[str]:
    if not p.is_file():
        return []
    return [l.strip() for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def is_fire_free(img_path: str) -> bool:
    """该图的标签里有没有 class 0(fire)。没有 = 无火子集可用。"""
    lp = ZIPDS / "labels" / f"{Path(img_path).stem}.txt"
    if not lp.is_file():
        return False
    for line in lp.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) >= 5 and int(float(parts[0])) == 0:
            return False
    return True


def labels_ok(img_path: str) -> bool:
    """按 ultralytics 的规则推导标签路径，确认存在。"""
    s = str(Path(img_path))
    lp = Path(s.replace("/images/", "/labels/").replace("\\images\\", "\\labels\\")).with_suffix(".txt")
    return lp.is_file()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="D:/datasets/fire_smoke_v6")
    ap.add_argument("--nofire-repeat", type=int, default=4, help="无火子集过采样倍数")
    ap.add_argument("--include-fire", action="store_true",
                    help="把有火预标的图也纳入（不推荐，见模块 docstring）")
    a = ap.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    v5_train = [str(p) for p in (V5 / "train" / "images").glob("*.jpg")]
    v5_valid = [str(p) for p in (V5 / "valid" / "images").glob("*.jpg")]
    v5_test = [str(p) for p in (V5 / "test" / "images").glob("*.jpg")]
    print(f"v5  train {len(v5_train)} | valid {len(v5_valid)} | test {len(v5_test)}")

    zip_train = read_list(ZIPDS / "train.txt")
    print(f"外部集 train {len(zip_train)} 条")

    nofire = [p for p in zip_train if is_fire_free(p)]
    withfire = [p for p in zip_train if not is_fire_free(p)]
    print(f"  其中无火 {len(nofire)} 张 / 有火预标 {len(withfire)} 张")

    if a.include_fire:
        print("[!] --include-fire 已开启：有火预标的图会进训练集，"
              "火框来自 conf>=0.55 预标，已知存在白烟误判，后果自负")
        extra = nofire + withfire
    else:
        extra = nofire
        print(f"[i] 有火预标的 {len(withfire)} 张**不纳入**（见 docstring）")

    extra_rep = []
    for _ in range(a.nofire_repeat):
        extra_rep.extend(extra)
    print(f"  过采样 x{a.nofire_repeat} -> {len(extra_rep)} 条")

    train_all = v5_train + extra_rep

    bad_img = [p for p in train_all if not Path(p).is_file()]
    bad_lab = [p for p in train_all if not labels_ok(p)]
    print(f"\n校验：缺图片 {len(bad_img)} | 缺标签 {len(bad_lab)}")
    for p in bad_img[:3]:
        print("  缺图:", p)
    for p in bad_lab[:3]:
        print("  缺标签:", p)
    if bad_img or bad_lab:
        print("[x] 校验失败，先修数据")
        return 1

    (out / "train.txt").write_text("\n".join(train_all) + "\n", encoding="utf-8")
    (out / "val.txt").write_text("\n".join(v5_valid) + "\n", encoding="utf-8")
    (out / "test.txt").write_text("\n".join(v5_test) + "\n", encoding="utf-8")
    (out / "dataset.yaml").write_text(f"""# v6 = v5 全量 + 外部烟雾集无火子集（x{a.nofire_repeat} 过采样）
# 外部集有火预标的 {len(withfire)} 张未纳入：火没标注，且高置信预标仍会误判白烟
# 生成脚本：scripts/build_v6_dataset.py
path: {out.as_posix()}
train: train.txt
val: val.txt
test: test.txt
nc: 2
names:
  0: fire
  1: smoke
""", encoding="utf-8")

    print(f"\n已写入 {out.as_posix()}")
    print(f"  train {len(train_all)} 条（v5 {len(v5_train)} + 无火x{a.nofire_repeat} {len(extra_rep)}）")
    print(f"  val   {len(v5_valid)} 条 | test {len(v5_test)} 条")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
