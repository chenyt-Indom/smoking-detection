"""把标注体检结果转成「建议标注」，出「原标注 vs 建议标注」对比图供人工确认。

为什么不直接删掉有问题的框
--------------------------
fire 本来就是弱类，删框只会让正样本更少。所以默认策略是**把框挪到真正的火区**：
  relocate  框内无火、但图里别处有火 → 移到最吻合的火区（把错标变成对的正样本）
  shrink    框比火区大很多 → 收紧到火区外接框 + 余量（直接改善 mAP50-95）
  drop      框内无火、且整图几乎没有火色 → 才删
  keep      判据正常 → 原样
smoke 类一律原样保留，只动 fire。

安全约束
--------
· 只读原标注，结果写到 proposals/ 下的新目录，**绝不覆盖** E:/datasets/fire_smoke_v3
· 收紧时限制最小尺寸（不低于原框线性尺寸的 30%），避免掩码碎片把框压成一条线
· 同一张图里挪到同一火区的重复框会被去重

输出
----
  proposals/labels/<图片名>.txt   建议标注（只含被改动的图）
  proposals/report.csv            逐框动作 + 改动前后坐标与面积比
  proposals/compare_XX.jpg        灰框=原标注，绿框=建议标注

用法
    python scripts/propose_fire_fixes.py --limit 200 --sheets 3
    python scripts/propose_fire_fixes.py --limit 0 --top 400 --sheets 34
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from find_bad_fire_labels import (  # noqa: E402
    FLAME_MIN_AREA_PCT, analyze, imwrite_u, load_thresholds, regions,
)

PAD = 0.12          # 建议框在火区外接框基础上外扩，占边长比例
MIN_SIDE_KEEP = 0.30  # 收紧后单边不得小于原框的 30%


def pad_box(b: tuple[int, int, int, int], w: int, h: int, pad: float = PAD):
    x1, y1, x2, y2 = b
    dx, dy = (x2 - x1) * pad, (y2 - y1) * pad
    return (max(0, int(x1 - dx)), max(0, int(y1 - dy)),
            min(w, int(x2 + dx)), min(h, int(y2 + dy)))


def iou_px(a, b) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def shrink_inside(mask: np.ndarray, box, w: int, h: int):
    """把框收紧到框内火焰像素的外接框。返回 None = 框内火像素太少，不动。"""
    x1, y1, x2, y2 = box
    sub = mask[y1:y2, x1:x2]
    ys, xs = np.nonzero(sub)
    if len(xs) < 12:
        return None
    fb = (x1 + int(xs.min()), y1 + int(ys.min()), x1 + int(xs.max()) + 1, y1 + int(ys.max()) + 1)
    nb = pad_box(fb, w, h)
    # 防止掩码碎片把框压得太小
    min_w = max(6, int((x2 - x1) * MIN_SIDE_KEEP))
    min_h = max(6, int((y2 - y1) * MIN_SIDE_KEEP))
    if nb[2] - nb[0] < min_w or nb[3] - nb[1] < min_h:
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        hw, hh = max(min_w, nb[2] - nb[0]) / 2, max(min_h, nb[3] - nb[1]) / 2
        nb = (max(0, int(cx - hw)), max(0, int(cy - hh)),
              min(w, int(cx + hw)), min(h, int(cy + hh)))
    return nb


def pick_region(box, regs, w, h):
    """挑与 GT 框最吻合的火区：先看 IoU，没有重叠就按中心距离取最近。"""
    if not regs:
        return None
    best, best_iou = None, 0.0
    for rx, ry, rw, rh in regs:
        r = (rx, ry, rx + rw, ry + rh)
        v = iou_px(box, r)
        if v > best_iou:
            best, best_iou = r, v
    if best is not None and best_iou > 0.01:
        return pad_box(best, w, h)
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    best = min(regs, key=lambda r: ((r[0] + r[2] / 2 - cx) ** 2 + (r[1] + r[3] / 2 - cy) ** 2))
    return pad_box((best[0], best[1], best[0] + best[2], best[1] + best[3]), w, h)


def plan(m: dict, img: np.ndarray, th: dict, strategy: str = "safe") -> dict:
    """对一张图算出每个 fire 框的动作。

    strategy="safe"（默认）：不挪框。框内无火 → 只删这个框；有火但偏大 → 收紧。
       2026-10-30 实测：relocate 会把框挪到橙色消防战斗服、城市灯光上，
       火焰色掩码分不清火与橙色衣物，所以 aggressive 策略只留作对照实验。
    strategy="relocate"：框内无火就挪到最吻合的火区（看看就好，别拿去训练）。
    """
    h, w = img.shape[:2]
    total = float(h * w)
    mask = m["_vis"][2]
    regs = regions(mask, FLAME_MIN_AREA_PCT / 100 * total)
    flame_pct = m["flame_pct"]

    out_boxes, acts = [], []
    for s in m["stats"]:
        box = s["box"]
        if s["judgeable"] and s["inside"] < th["low_inside"]:
            if strategy == "safe":
                acts.append(("drop", box, None))
                continue
            nb = pick_region(box, regs, w, h)
            if nb is None or flame_pct < 0.02:
                acts.append(("drop", box, None))
                continue
            acts.append(("relocate", box, nb))
        elif s["judgeable"] and s["tight"] > th["high_tight"]:
            nb = shrink_inside(mask, box, w, h)
            acts.append(("shrink", box, nb) if nb else ("keep", box, box))
        else:
            acts.append(("keep", box, box))
        nb = acts[-1][2]
        if nb is not None:
            out_boxes.append(nb)

    # 去重：挪动后高度重叠的框只留一个
    dedup = []
    for b in out_boxes:
        if all(iou_px(b, o) < 0.70 for o in dedup):
            dedup.append(b)
    return {"acts": acts, "boxes": dedup, "n_in": len(m["stats"])}


def to_norm(box, w: int, h: int) -> str:
    x1, y1, x2, y2 = box
    return (f"{(x1 + x2) / 2 / w:.6f} {(y1 + y2) / 2 / h:.6f} "
            f"{(x2 - x1) / w:.6f} {(y2 - y1) / h:.6f}")


def draw_pair(img: np.ndarray, orig_boxes, new_boxes) -> np.ndarray:
    vis = img.copy()
    for x1, y1, x2, y2 in orig_boxes:          # 灰 = 原标注
        cv2.rectangle(vis, (x1, y1), (x2, y2), (150, 150, 150), 1)
    for x1, y1, x2, y2 in new_boxes:           # 绿 = 建议标注
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 230, 0), 2)
    return vis


def make_sheets(items: list[tuple], outdir: Path, per: int, max_sheets: int) -> list[Path]:
    made = []
    cell, cols = 470, 3
    for si in range(min(max_sheets, (len(items) + per - 1) // per)):
        chunk = items[si * per:(si + 1) * per]
        if not chunk:
            break
        rowsn = (len(chunk) + cols - 1) // cols
        canvas = np.full((rowsn * cell, cols * cell, 3), 255, np.uint8)
        for i, (title, vis) in enumerate(chunk):
            hh, ww = vis.shape[:2]
            s = min(cell / ww, (cell - 26) / hh)
            im = cv2.resize(vis, (int(ww * s), int(hh * s)))
            r, c = divmod(i, cols)
            y0, x0 = r * cell + 26, c * cell
            canvas[y0:y0 + im.shape[0], x0:x0 + im.shape[1]] = im
            cv2.putText(canvas, title[:40], (x0 + 5, y0 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.46, (20, 20, 20), 1, cv2.LINE_AA)
        f = outdir / f"compare_{si + 1:02d}.jpg"
        imwrite_u(f, canvas)
        made.append(f)
    return made


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="E:/datasets/fire_smoke_v3")
    ap.add_argument("--split", default="train")
    ap.add_argument("--audit", default="D:/视觉安防系统/runs/fire_label_audit")
    ap.add_argument("--out", default="D:/视觉安防系统/runs/fire_label_fix")
    ap.add_argument("--limit", type=int, default=0, help="只处理体检报告里的前 N 张可疑图，0=全部")
    ap.add_argument("--sheets", type=int, default=5)
    ap.add_argument("--only", default="", help="对比图只渲染含这些动作的图，如 drop 或 drop,shrink")
    ap.add_argument("--strategy", default="safe", choices=["safe", "relocate"])
    ap.add_argument("--write-labels", action="store_true", help="实际写出建议标注文件")
    a = ap.parse_args()
    only = {s.strip() for s in a.only.split(",") if s.strip()}

    audit_dir = Path(a.audit)
    th = load_thresholds(audit_dir)
    audit_csv = audit_dir / "audit.csv"
    if not audit_csv.exists():
        print(f"[x] 先跑 find_bad_fire_labels.py 生成 {audit_csv}")
        return 1

    names, seen = [], set()
    with audit_csv.open(encoding="utf-8-sig") as fh:
        for r in csv.DictReader(fh):
            if r["image"] not in seen:
                seen.add(r["image"])
                names.append(r["image"])
    if a.limit:
        names = names[:a.limit]
    print(f"可疑图 {len(names)} 张，阈值 inside<{th['low_inside']} tight>{th['high_tight']}")

    base = Path(a.data) / a.split
    outdir = Path(a.out)
    (outdir / "labels").mkdir(parents=True, exist_ok=True)

    rows, sheets_items = [], []
    tally: dict[str, int] = {}
    for i, name in enumerate(names):
        p = base / "images" / name
        m = analyze(p, want_vis=True)
        if m is None:
            continue
        img = m["_vis"][0]              # 复用 analyze 已解码的图，不再二次读盘
        h, w = img.shape[:2]
        res = plan(m, img, th, a.strategy)

        lb = base / "labels" / (p.stem + ".txt")
        keep_smoke = []
        if lb.exists():
            for line in lb.read_text(encoding="utf-8", errors="ignore").splitlines():
                t = line.split()
                if len(t) >= 5 and abs(float(t[0]) - 1) < 0.5:
                    keep_smoke.append(line.strip())

        for act, ob, nb in res["acts"]:
            tally[act] = tally.get(act, 0) + 1
            rows.append({
                "image": name, "action": act,
                "orig": f"{ob[0]} {ob[1]} {ob[2]} {ob[3]}",
                "new": "" if nb is None else f"{nb[0]} {nb[1]} {nb[2]} {nb[3]}",
                "area_ratio": "" if nb is None else round(
                    ((nb[2] - nb[0]) * (nb[3] - nb[1])) /
                    max((ob[2] - ob[0]) * (ob[3] - ob[1]), 1), 2),
            })

        if a.write_labels:
            lines = ([f"0 {to_norm(b, w, h)}" for b in res["boxes"]] + keep_smoke)
            (outdir / "labels" / f"{p.stem}.txt").write_text(
                "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

        changed = sum(1 for act, _, _ in res["acts"] if act != "keep")
        orig_all = [b for b in (s["box"] for s in m["stats"])]
        acts_here = {act for act, _, _ in res["acts"]}
        if not only or (acts_here & only):
            vis = draw_pair(img, orig_all, res["boxes"])
            letter = {"drop": "删", "shrink": "收", "relocate": "挪", "keep": "留"}
            desc = "+".join(f"{letter[k]}{sum(1 for x, _, _ in res['acts'] if x == k)}"
                            for k in ("drop", "shrink", "relocate") if k in acts_here)
            title = f"#{i + 1} [{desc}] {name[:22]}"
            sheets_items.append((changed, title, vis))
        if (i + 1) % 100 == 0:
            print(f"  已处理 {i + 1}/{len(names)}", flush=True)

    with (outdir / "report.csv").open("w", newline="", encoding="utf-8-sig") as fh:
        wr = csv.DictWriter(fh, fieldnames=["image", "action", "orig", "new", "area_ratio"])
        wr.writeheader()
        wr.writerows(rows)

    # 对比图也按「改动量大」优先
    sheets_items.sort(key=lambda x: -x[0])
    made = make_sheets([(t, v) for _, t, v in sheets_items], outdir, per=6, max_sheets=a.sheets)

    print("=" * 70)
    print(f"处理 {len(names)} 张，逐框动作统计：")
    for k, v in sorted(tally.items(), key=lambda x: -x[1]):
        print(f"  {k:<9}{v:>7}")
    print(f"报告：{outdir / 'report.csv'}")
    if a.write_labels:
        print(f"建议标注：{outdir / 'labels'}（{len(list((outdir / 'labels').glob('*.txt')))} 个文件）")
    for f in made:
        print(f"对比图：{f}")
    print("\n灰框=原标注  绿框=建议标注（--write-labels 才会真正写文件）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
