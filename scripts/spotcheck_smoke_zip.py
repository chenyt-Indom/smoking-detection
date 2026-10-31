"""抽检 smoke_zip_ds 的标签质量，拼一张联络表出来看。

为什么要这个：火框是模型自动预标的、**没人工复核**。如果预标本身是错的，
训练就是拿模型的错误预测当标准答案（自我强化），跑一轮除了浪费算力什么也证明不了。
花 30 秒看几张图，比事后分析半天曲线便宜得多。

用法：
    python scripts/spotcheck_smoke_zip.py                  # 随机 6 张有火预标的
    python scripts/spotcheck_smoke_zip.py --n 9 --seed 1
    python scripts/spotcheck_smoke_zip.py --only-fire      # 只看火框面积最小的
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

ROOT = Path("D:/datasets/smoke_zip_ds")
FONT = "C:/Windows/Fonts/msyh.ttc"
CELL = 460          # 每格边长
COLS = 3

CLS_NAME = {0: "fire", 1: "smoke"}
CLS_COLOR = {0: (255, 60, 60), 1: (60, 150, 255)}


def read_labels(stem: str) -> list[tuple[int, float, float, float, float]]:
    p = ROOT / "labels" / f"{stem}.txt"
    if not p.exists():
        return []
    out = []
    for line in p.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        out.append((int(float(parts[0])), *(float(v) for v in parts[1:5])))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--only-fire", action="store_true", help="按火框面积升序取最小的几张")
    ap.add_argument("--no-fire", action="store_true", help="改看**没有火预标**的图（验证是否真无火）")
    ap.add_argument("--out", default="D:/视觉安防系统/runs/spotcheck_smoke_zip.jpg")
    a = ap.parse_args()

    from PIL import Image, ImageDraw, ImageFont

    imgs = [Path(l.strip()) for l in (ROOT / "train.txt").read_text(encoding="utf-8").splitlines() if l.strip()]
    print(f"train 清单 {len(imgs)} 张")

    # 分两类：有 fire 预标的、没有的
    cands: list[tuple[Path, float, int, int]] = []
    nofire: list[tuple[Path, float, int, int]] = []
    for p in imgs:
        labs = read_labels(p.stem)
        fires = [w * h for c, _, _, w, h in labs if c == 0]
        if not fires:
            nofire.append((p, 0.0, 0, len(labs)))
            continue
        cands.append((p, min(fires), len(fires), len(labs)))

    print(f"有 fire 预标 {len(cands)} 张 / 无 fire 预标 {len(nofire)} 张")
    if not cands:
        print("[x] 没有任何 fire 预标，检查标签路径")
        return 1

    if a.no_fire:
        # 关键验证：这些图被当成"纯烟"用，如果里面其实有火（模型漏检），
        # 那就和"原数据集没标火"是同一个坑，只是换了个来源。
        random.seed(a.seed)
        pick = random.sample(nofire, min(a.n, len(nofire)))
        print("[模式] 只看无火预标的图 —— 若画面里有火还漏了，说明这 588 张也不能用")
    elif a.only_fire:
        cands.sort(key=lambda t: t[1])
        pick = cands[: a.n]
        print("按火框最小面积取前几张：")
    else:
        random.seed(a.seed)
        pick = random.sample(cands, min(a.n, len(cands)))

    rows = (len(pick) + COLS - 1) // COLS
    sheet = Image.new("RGB", (COLS * CELL, rows * CELL), (24, 24, 28))
    font = ImageFont.truetype(FONT, 15)

    for i, (path, min_fire, n_fire, n_all) in enumerate(pick):
        im = Image.open(path).convert("RGB")
        W, H = im.size
        im.thumbnail((CELL, CELL), Image.LANCZOS)
        # 缩放后按比例还原坐标
        sx, sy = im.width / W, im.height / H

        d = ImageDraw.Draw(im)
        for cls, cx, cy, bw, bh in read_labels(path.stem):
            x1 = (cx - bw / 2) * W * sx
            y1 = (cy - bh / 2) * H * sy
            x2 = (cx + bw / 2) * W * sx
            y2 = (cy + bh / 2) * H * sy
            col = CLS_COLOR.get(cls, (255, 255, 0))
            d.rectangle([x1, y1, x2, y2], outline=col, width=2)
            if cls == 0:
                # 标出面积占比，方便判断这框是不是太小/太碎
                d.text((x1 + 2, max(0, y1 - 17)), f"F{bw*bh*100:.2f}%", fill=col, font=font)

        ox, oy = (i % COLS) * CELL, (i // COLS) * CELL
        sheet.paste(im, (ox + (CELL - im.width) // 2, oy + 18 + (CELL - 18 - im.height) // 2))
        cap = f"{path.stem}  fire={n_fire} all={n_all} minF={min_fire*100:.2f}%"
        d2 = ImageDraw.Draw(sheet)
        d2.rectangle([ox, oy, ox + CELL, oy + 18], fill=(0, 0, 0))
        d2.text((ox + 4, oy + 1), cap, fill=(220, 220, 220), font=font)
        # 红=fire 蓝=smoke 图例写在第一格右上
        if i == 0:
            d2.text((ox + CELL - 118, oy + 1), "红=FIRE 蓝=SMOKE", fill=(255, 200, 120), font=font)

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out, quality=90)
    print(f"已输出 {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
