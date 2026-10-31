"""对 smoke_zip 的大图重跑一次火预标，按不同 conf 阈值统计「可信火框」的量。

为什么必须做这一步
------------------
这批图的**火是没标的**。双类训练时，没标的火区域会被当成背景，
等于教模型「火不是火」——火召回会无声塌掉，而且训练曲线看不出来（
因为 val 集里这批图也是同样标法，指标反而"很好看"）。

反过来，预标本身也不可靠：抽检发现白烟画 F1.87%、黑烟画三个框、
消防员身上一堆 F0.03% 碎框。所以不是「标了就行」，得先量出
「哪个置信度以上的框还能信」。

做法
----
一次推理跑到 conf 0.15，之后在代码里按阈值筛（避免重复推理），
统计各阈值下：多少张图有火、共多少个火框、框面积分布。
再抽出 conf 最高的几张画出来看，用眼睛确认。

跑法：
    python scripts/reprelabel_smoke_zip.py
    python scripts/reprelabel_smoke_zip.py --sheet-conf 0.55 --n 6
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

IMGS = Path("D:/datasets/smoke_zip_big/images")
WEIGHTS = ROOT / "models" / "fire_smoke_v5_26s960_best.pt"

THRS = (0.30, 0.40, 0.50, 0.55, 0.60, 0.70)
BUCKETS = [(0.0, 0.001, "<0.1%  碎框"), (0.001, 0.005, "0.1~0.5%"),
           (0.005, 0.01, "0.5~1%"), (0.01, 0.03, "1~3%"), (0.03, 1.01, ">3%")]


def imread_u(p: Path):
    import cv2
    return cv2.imdecode(np.frombuffer(p.read_bytes(), np.uint8), cv2.IMREAD_COLOR)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--imgsz", type=int, default=960)
    ap.add_argument("--tile", type=int, default=2)
    ap.add_argument("--n", type=int, default=6, help="抽检图数量")
    ap.add_argument("--sheet-conf", type=float, default=0.55, help="抽检图画哪些 conf 以上的框")
    ap.add_argument("--out", default="D:/视觉安防系统/runs/reprelabel_check.jpg")
    a = ap.parse_args()

    from src.infer import Detector
    det = Detector(str(WEIGHTS), conf=0.15, imgsz=a.imgsz, device=0)

    imgs = sorted(IMGS.glob("*.jpg"))
    print(f"大图 {len(imgs)} 张 · imgsz {a.imgsz} · {a.tile}×{a.tile} 切片 · 推理到 conf 0.15\n")

    # 每张图存 [(conf, area_norm, xyxy)] 的火预测
    per_img: dict[str, list[tuple[float, float, tuple]]] = {}
    for i, p in enumerate(imgs, 1):
        im = imread_u(p)
        if im is None:
            continue
        h, w = im.shape[:2]
        ds = det.detect(im, sliced=True, tile_grid=a.tile, overlap=0.25)
        fires = []
        for d in ds:
            if d.cls_id != 0:
                continue
            x1, y1, x2, y2 = d.xyxy
            aw, ah = (x2 - x1) / w, (y2 - y1) / h
            fires.append((d.conf, max(0.0, aw) * max(0.0, ah), (x1, y1, x2, y2)))
        per_img[p.stem] = fires
        if i % 300 == 0:
            print(f"  {i}/{len(imgs)} …", flush=True)

    print(f"\n推理完成，共 {len(per_img)} 张\n")
    print("=" * 74)
    for t in THRS:
        n_img = sum(1 for f in per_img.values() if any(c >= t for c, _, _ in f))
        n_box = sum(1 for f in per_img.values() for c, _, _ in f if c >= t)
        print(f"{t:>6.2f} {n_img:>10d} {n_img/len(per_img)*100:>7.1f}% {n_box:>10d} {n_box/max(n_img,1):>8.2f}")
    print("=" * 74)

    # 高置信框的面积分布：如果 <0.1% 碎框占比很高，说明置信度不足以过滤伪影
    for t in (0.30, 0.55):
        cnt = Counter()
        for f in per_img.values():
            for c, ar, _ in f:
                if c < t:
                    continue
                for lo, hi, tag in BUCKETS:
                    if lo <= ar < hi:
                        cnt[tag] += 1
                        break
        tot = sum(cnt.values())
        if not tot:
            continue
        print(f"\nconf≥{t:.2f} 火框面积分布 (n={tot}):")
        for _, _, tag in BUCKETS:
            v = cnt.get(tag, 0)
            print(f"  {tag:14s} {v:6d}  {v/tot*100:5.1f}%")

    # ── 抽检图：挑「高置信火框最多」的图，最可能暴露假阳性 ──────────
    ranked = sorted(per_img.items(),
                    key=lambda kv: sum(1 for c, _, _ in kv[1] if c >= a.sheet_conf), reverse=True)
    pick = [(k, v) for k, v in ranked[: a.n] if any(c >= a.sheet_conf for c, _, _ in v)]
    if pick:
        from PIL import Image, ImageDraw, ImageFont
        font = ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", 15)
        CELL, COLS = 480, 3
        rows = (len(pick) + COLS - 1) // COLS
        sheet = Image.new("RGB", (COLS * CELL, rows * CELL), (24, 24, 28))
        for i, (stem, fires) in enumerate(pick):
            im = Image.open(IMGS / f"{stem}.jpg").convert("RGB")
            W, H = im.size
            im.thumbnail((CELL, CELL), Image.LANCZOS)
            sx, sy = im.width / W, im.height / H
            d = ImageDraw.Draw(im)
            shown = 0
            for c, ar, (x1, y1, x2, y2) in fires:
                if c < a.sheet_conf:
                    continue
                d.rectangle([x1 * sx, y1 * sy, x2 * sx, y2 * sy], outline=(255, 60, 60), width=2)
                d.text((x1 * sx + 2, max(0, y1 * sy - 17)), f"{c:.2f} {ar*100:.2f}%",
                       fill=(255, 60, 60), font=font)
                shown += 1
            ox, oy = (i % COLS) * CELL, (i // COLS) * CELL
            sheet.paste(im, (ox + (CELL - im.width) // 2, oy + (CELL - im.height) // 2))
            ImageDraw.Draw(sheet).text(
                (ox + 4, oy + 2), f"{stem}  {shown} fire@>={a.sheet_conf}",
                fill=(210, 210, 210), font=font)
        out = Path(a.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        sheet.save(out, quality=90)
        print(f"\n抽检图（红框=fire，标了 conf 与面积占比）→ {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
