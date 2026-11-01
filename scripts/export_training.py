"""把人工复核结果导出成 YOLO 训练集 —— 主动学习闭环的最后一环。

整体闭环
--------
  ① 系统跑着，每条事件都带「模型判了什么」（llm_label）
  ② 你在看板上点「有火 / 无火」，等于给这条事件打了标签
  ③ 服务按四类自动分拣（notify.report 里做的）：
       TP  人工有火 + 模型有火  → 已学会，丢弃
       TN  人工无火 + 模型无火  → 已学会，丢弃
       FP  人工无火 + 模型有火  → **误报**，存 hard_negatives/（空标注）
       FN  人工有火 + 模型无火  → **漏报**，存 hard_negatives/positives/
  ④ 本脚本把这两堆样本转成标准 YOLO 目录，喂回训练

为什么只学 FP / FN
-----------------
两边都判对的样本，模型已经会了，再喂一遍没有信息增量 —— 只会让训练集膨胀、
拖慢训练。真正的收益只来自"模型判错的那部分"，这叫 hard example mining。

用法
----
    python scripts/export_training.py                 # 默认导出到 datasets/al/
    python scripts/export_training.py --out D:/al_v2  # 指定目录
    python scripts/export_training.py --dry           # 只看统计不写文件
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

REPORT_DIR = ROOT / "hard_negatives"
REPORTS = REPORT_DIR / "reports.jsonl"

# 模型类别顺序（和训练时的 data.yaml 保持一致）
CLS_ID = {"fire": 0, "smoke": 1}


def imread_cn(p: Path):
    """cv2.imread 在 Windows 中文路径下返回 None，必须走 fromfile + imdecode。"""
    return cv2.imdecode(np.fromfile(str(p), dtype=np.uint8), cv2.IMREAD_COLOR)


def load_rows() -> list[dict]:
    if not REPORTS.is_file():
        return []
    out = []
    for ln in REPORTS.read_text(encoding="utf-8", errors="ignore").splitlines():
        ln = ln.strip()
        if not ln:
            continue
        try:
            out.append(json.loads(ln))
        except json.JSONDecodeError:
            continue
    return out


def yolo_line(box, img_w, img_h, cls_id: int, sx: float, sy: float, out_w: float, out_h: float) -> str:
    """像素框（相对原图）→ YOLO 归一化坐标（相对导出后的图）。"""
    x1, y1, x2, y2 = (float(v) for v in box)
    cx = (x1 + x2) / 2.0 * sx / out_w
    cy = (y1 + y2) / 2.0 * sy / out_h
    bw = abs(x2 - x1) * sx / out_w
    bh = abs(y2 - y1) * sy / out_h
    cx = min(max(cx, 0.0), 1.0)
    cy = min(max(cy, 0.0), 1.0)
    bw = min(max(bw, 0.0), 1.0)
    bh = min(max(bh, 0.0), 1.0)
    return f"{cls_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "datasets" / "al"))
    ap.add_argument("--dry", action="store_true", help="只统计，不写文件")
    a = ap.parse_args()

    rows = load_rows()
    if not rows:
        print(f"[x] 没有反馈记录：{REPORTS}")
        print("    先在看看板上点几条「有火 / 无火」。")
        return 1

    # 同一 wall_ms 可能被点多次，取最后一次
    latest: dict[int, dict] = {}
    for r in rows:
        wid = r.get("wall_ms")
        if wid is not None:
            latest[wid] = r
    rows = list(latest.values())

    cnt = Counter(r.get("verdict_class") or "?" for r in rows)
    print(f"共 {len(rows)} 条人工反馈")
    for k in ("TP", "TN", "FP", "FN"):
        print(f"  {k}  {cnt.get(k, 0):4d}  "
              + {"TP": "都对，不学", "TN": "都对，不学",
                 "FP": "模型误报 → 负样本", "FN": "模型漏报 → 正样本"}[k])

    out = Path(a.out)
    img_tr, lbl_tr = out / "images" / "train", out / "labels" / "train"
    if not a.dry:
        for d in (img_tr, lbl_tr):
            d.mkdir(parents=True, exist_ok=True)

    # 目录按类别分开了（images/fire、images/smoke），统计也分开看
    stat = {"fire": [0, 0], "smoke": [0, 0]}     # [正样本, 负样本]

    n_pos = n_neg = 0
    for r in rows:
        vc = r.get("verdict_class")
        if vc not in ("FP", "FN"):
            continue
        saved = r.get("saved")
        if not saved:
            continue
        src = Path(saved)
        if not src.is_file():
            continue
        kind = str(r.get("kind") or "fire").lower()
        if kind not in ("fire", "smoke"):
            kind = "fire"

        if vc == "FP":
            # 负样本：空标注 = "这张图里没有目标"
            stem = f"neg_{r['wall_ms']}_{r.get('cam','')}"
            if not a.dry:
                shutil.copy2(src, img_tr / f"{stem}.jpg")
                (lbl_tr / f"{stem}.txt").write_text("", encoding="utf-8")
            n_neg += 1
            stat[kind][1] += 1
        else:
            # 正样本：用 YOLO 当时的框当伪标注
            img = imread_cn(src)
            if img is None:
                continue
            oh, ow = img.shape[:2]
            iw = float(r.get("img_w") or ow)
            ih = float(r.get("img_h") or oh)
            box = r.get("box")
            if not box or len(box) != 4:
                continue
            cls_id = CLS_ID.get(kind, 0)
            line = yolo_line(box, iw, ih, cls_id, ow / iw, oh / ih, ow, oh)
            stem = f"pos_{r['wall_ms']}_{r.get('cam','')}"
            if not a.dry:
                shutil.copy2(src, img_tr / f"{stem}.jpg")
                (lbl_tr / f"{stem}.txt").write_text(line + "\n", encoding="utf-8")
            n_pos += 1
            stat[kind][0] += 1

    print(f"\n导出：正样本 {n_pos} 张，负样本 {n_neg} 张"
          + ("（dry-run，未写文件）" if a.dry else f" → {out}"))
    for k, (pp, nn) in stat.items():
        if pp or nn:
            print(f"  {k:6s} 正 {pp:3d}  负 {nn:3d}")

    if not a.dry and (n_pos or n_neg):
        # 顺手生成一份 data.yaml，可以直接交给 ultralytics
        names = list(CLS_ID.keys())
        y = out / "data.yaml"
        y.write_text(
            f"path: {out.as_posix()}\n"
            "train: images/train\n"
            "val: images/train\n"      # 样本太少时先用训练集当验证集，看 loss 趋势
            f"nc: {len(names)}\n"
            f"names: {names}\n", encoding="utf-8")
        print(f"已写 {y}")
        print("\n下一步：把这两个目录并进主训练集，重训一轮")
        print(f"  {img_tr}  -> 你的 images/")
        print(f"  {lbl_tr}  -> 你的 labels/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
