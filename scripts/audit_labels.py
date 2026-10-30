"""标注审计：找出「火域标注不规范」的可疑样本，供人工复检。

为什么需要这个
--------------
v3 诊断（2026-10-30）：fire 的 mAP50 = 0.710 但 mAP50-95 只有 0.416，
落差 0.29（smoke 只有 0.21）。这说明模型**能大致找到火，但框不贴合**，
而框不贴合的常见根因是标注本身不一致：
  · 有的图只框了火点，有的把整片火域全框进去
  · 火和烟重叠时，火的框把烟也带进去了
  · 极小框（几个像素）是标注噪点
  · 重复标注（同一团火两个几乎重合的框）

这个脚本不改任何标注，只输出**可疑清单**（CSV + 汇总），由人工判断要不要修。

用法：
    python scripts/audit_labels.py                          # 默认审计 v3 train
    python scripts/audit_labels.py --split test             # 审计测试集
    python scripts/audit_labels.py --data E:/datasets/fire_smoke_v4
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

NAMES = {0: "fire", 1: "smoke"}

# 阈值（按画面面积比例，与分辨率无关）
TINY_PCT = 0.01        # < 0.01% 画面：约 8px@640，基本是噪点
HUGE_PCT = 30.0        # > 30% 画面：多半把烟/背景一起框进去了
OVERLAP_XCLS = 0.80    # fire 与 smoke 框 IoU > 0.8：高度重叠，疑似混标
OVERLAP_SAME = 0.85    # 同类框 IoU > 0.85：疑似重复标注
EDGE_MARGIN = 0.02     # 框越界容差（归一化坐标）


def iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix = min(ax2, bx2) - max(ax1, bx1)
    iy = min(ay2, by2) - max(ay1, by1)
    if ix <= 0 or iy <= 0:
        return 0.0
    inter = ix * iy
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / union if union > 0 else 0.0


def to_xyxy(cx, cy, w, h):
    return cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2


def audit(labels_dir: Path, out_csv: Path) -> dict:
    stats = defaultdict(lambda: {"n": 0, "tiny": 0, "huge": 0, "oob": 0})
    issues = []
    files = sorted(labels_dir.glob("*.txt"))
    for f in files:
        rows = []
        for line in f.read_text(encoding="utf-8", errors="ignore").splitlines():
            t = line.split()
            if len(t) < 5:
                continue
            rows.append((int(float(t[0])), *[float(v) for v in t[1:5]]))
        for i, (c, cx, cy, w, h) in enumerate(rows):
            name = NAMES.get(c, str(c))
            area = w * h * 100
            st = stats[name]
            st["n"] += 1
            if area < TINY_PCT:
                st["tiny"] += 1
                issues.append((f.name, "过小", name, f"面积 {area:.4f}%"))
            if area > HUGE_PCT:
                st["huge"] += 1
                issues.append((f.name, "过大", name, f"面积 {area:.1f}%"))
            x1, y1, x2, y2 = to_xyxy(cx, cy, w, h)
            if x1 < -EDGE_MARGIN or y1 < -EDGE_MARGIN or x2 > 1 + EDGE_MARGIN or y2 > 1 + EDGE_MARGIN:
                st["oob"] += 1
                issues.append((f.name, "越界", name, f"归一化框 ({x1:.3f},{y1:.3f},{x2:.3f},{y2:.3f})"))
            for j in range(i + 1, len(rows)):
                c2, cx2, cy2, w2, h2 = rows[j]
                ov = iou(to_xyxy(cx, cy, w, h), to_xyxy(cx2, cy2, w2, h2))
                if c != c2 and ov > OVERLAP_XCLS:
                    issues.append((f.name, "火烟重叠", f"{name}+{NAMES.get(c2, c2)}", f"IoU {ov:.2f}"))
                elif c == c2 and ov > OVERLAP_SAME:
                    issues.append((f.name, "重复框", name, f"IoU {ov:.2f}"))

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8-sig") as fh:
        wr = csv.writer(fh)
        wr.writerow(["图片", "问题", "类别", "详情"])
        wr.writerows(issues)

    return {"files": len(files), "stats": dict(stats), "issues": len(issues), "csv": str(out_csv)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="E:/datasets/fire_smoke_v3")
    ap.add_argument("--split", default="train")
    ap.add_argument("--out", default="D:/视觉安防系统/runs/label_audit.csv")
    a = ap.parse_args()

    labels = Path(a.data) / a.split / "labels"
    if not labels.exists():
        print(f"[x] 找不到 {labels}")
        return 1

    r = audit(labels, Path(a.out))
    print("=" * 62)
    print(f"审计目录：{labels}（{r['files']} 个标注文件）")
    print(f"{'类别':<8}{'框数':>8}{'过小':>8}{'过大':>8}{'越界':>8}")
    for name, st in r["stats"].items():
        print(f"{name:<8}{st['n']:>8}{st['tiny']:>8}{st['huge']:>8}{st['oob']:>8}")
    print("-" * 62)
    print(f"可疑记录 {r['issues']} 条 → {r['csv']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
