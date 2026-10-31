"""构建 v7：v5 全量 + 外部烟雾集「确认无火」子集。

外部集 Y-烟雾24000（24,574 张）的 classes.txt 只有 smoke 一行，标签 class 0 = 烟。
本项目约定 class 0 = fire / class 1 = smoke，所以外部标签必须先重映射
（见 scripts/remap_external_labels.py，产出 D:/datasets/smoke_zip_remap/），
否则 3 万多个烟框会全被当成 fire。

只取 v5 扫描 conf=0.03 检不出火的 21,645 张：
  有火的 2,929 张是真实火灾现场，火明显却没标注，
  进双类训练等于教模型「火不是目标」，fire 召回会无声塌掉。

外部样本占训练集 20%：无火组抽样约六成是合成白烟贴图，
全量灌入会让合成特征压过真实场景。
"""
import json
import random
from pathlib import Path

V6 = Path(r"D:/datasets/fire_smoke_v6")
SCAN = Path(r"D:/datasets/smoke_zip_full/fire_scan.json")
OUT = Path(r"D:/datasets/fire_smoke_v7")

TARGET_RATIO = 0.20
SEED = 20261031


def to_remap(p: str) -> str:
    """外部图路径改指向重映射后的标签目录（images 用联接，图不复制）。"""
    return p.replace("smoke_zip_full", "smoke_zip_remap")


def main() -> int:
    scan = json.loads(SCAN.read_text(encoding="utf-8"))
    nofire = [to_remap(r["p"]) for r in scan if r["nfire"] == 0]
    ext_tr = [p for p in nofire if "\\train\\images\\" in p]
    ext_va = [p for p in nofire if "\\val\\images\\" in p]
    print(f"无火图 {len(nofire)} 张 | 原属 train {len(ext_tr)} | 原属 val {len(ext_va)}")

    v5_tr = [l for l in (V6 / "train.txt").read_text(encoding="utf-8").splitlines()
             if l.strip() and "smoke_zip_ds" not in l]
    v5_va = [l for l in (V6 / "val.txt").read_text(encoding="utf-8").splitlines() if l.strip()]
    print(f"v5 原始：train {len(v5_tr)} | val {len(v5_va)}")

    need = int(len(v5_tr) * TARGET_RATIO / (1 - TARGET_RATIO))
    random.seed(SEED)
    pick = random.sample(ext_tr, need) if len(ext_tr) >= need else ext_tr[:]
    print(f"目标外部样本 {need} 张，实抽 {len(pick)} 张")

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "train.txt").write_text("\n".join(v5_tr + pick) + "\n", encoding="utf-8")
    (OUT / "val.txt").write_text("\n".join(v5_va) + "\n", encoding="utf-8")
    (OUT / "external_val.txt").write_text("\n".join(ext_va) + "\n", encoding="utf-8")

    (OUT / "dataset.yaml").write_text(
        "# v7 = v5 全量 + 外部烟雾集无火子集（20%，已重映射 class0->1）\n"
        "# 外部集无 fire 标注；v5 扫描 conf0.03 检不出火的 21,645 张才可用\n"
        "# 有火预标的 2,929 张已排除，防止污染 fire 类\n"
        f"path: {OUT.as_posix()}\n"
        "train: train.txt\n"
        "val: val.txt\n"
        f"test: {(V6 / 'test.txt').as_posix()}\n"
        "nc: 2\n"
        "names:/n"
        "  0: fire\n"
        "  1: smoke\n",
        encoding="utf-8")

    tot = len(v5_tr) + len(pick)
    print(f"\n=== v7 构成 ===")
    print(f"  train {tot}（v5 {len(v5_tr)} + 外部 {len(pick)}，外部 {len(pick)/tot:.1%}）")
    print(f"  val   {len(v5_va)}（与 v5 完全一致，指标可比）")
    print(f"  ext_val {len(ext_va)}（外部无火图，单独验收烟召回）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
