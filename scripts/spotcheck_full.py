"""目视抽检 smoke_zip_full：把「模型说没火」和「模型说有火」的图各抽一批做成联络表。

目的：确认「conf 0.03 检不出 fire」这件事是否可信。
      如果无火组里其实藏着一堆火，那 21,645 张就不能进双类训练。
"""
import json
import random
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(r"D:/datasets/smoke_zip_full")
SCAN = json.loads((ROOT / "fire_scan.json").read_text(encoding="utf-8"))

CELL = 300
PAD = 4
LABEL_H = 20


def label_path(img: str) -> Path:
    s = img.replace("\\images\\", "\\labels\\").replace("/images/", "/labels/")
    return Path(s).with_suffix(".txt")


def sheet(picks, out, title):
    cols = 6
    rows = (len(picks) + cols - 1) // cols
    W = cols * (CELL + PAD) + PAD
    H = rows * (CELL + LABEL_H + PAD) + PAD
    canvas = Image.new("RGB", (W, H), (30, 30, 30))
    dr = ImageDraw.Draw(canvas)

    for i, rec in enumerate(picks):
        p = Path(rec["p"])
        try:
            im = Image.open(p).convert("RGB")
        except Exception:
            continue
        iw, ih = im.size

        boxes = []
        lp = label_path(rec["p"])
        if lp.is_file():
            for line in lp.read_text(encoding="utf-8").splitlines():
                v = line.split()
                if len(v) >= 5:
                    c, cx, cy, bw, bh = int(float(v[0])), *map(float, v[1:5])
                    boxes.append(("smoke", (cx - bw / 2) * iw, (cy - bh / 2) * ih,
                                  (cx + bw / 2) * iw, (cy + bh / 2) * ih))
                    _ = c

        im.thumbnail((CELL, CELL))
        sx, sy = im.size[0] / iw, im.size[1] / ih
        d = ImageDraw.Draw(im)
        for _n, x1, y1, x2, y2 in boxes:
            d.rectangle([x1 * sx, y1 * sy, x2 * sx, y2 * sy], outline=(0, 200, 255), width=2)

        cx = PAD + (i % cols) * (CELL + PAD)
        cy = PAD + (i // cols) * (CELL + LABEL_H + PAD)
        canvas.paste(im, (cx, cy))
        dr.text((cx + 2, cy + CELL + 4),
                f"{p.stem}  f={rec['nfire']}",
                fill=(230, 230, 230))

    canvas.save(out, quality=88)
    print(f"{title} -> {out}  ({len(picks)} 张)")


random.seed(42)
nofire = [r for r in SCAN if r["nfire"] == 0]
withfire = [r for r in SCAN if r["nfire"] >= 3]

sheet(random.sample(nofire, 24), ROOT / "sheet_nofire.jpg", "无火组")
sheet(random.sample(withfire, 12), ROOT / "sheet_withfire.jpg", "有火组")
