"""从 Y-烟雾24000.zip 里筛出大图、解压、并做「火」的自动预标。

三个决定，每个都有理由
---------------------
1. **只要大图**（最长边 ≥800）
   88.5% 是 240x240 缩略图（人为缩过，细节已丢），跟线上 `896 + 2x2 切片` 的
   输入分布差得远。全量 24574 张里大图约 2700 张，是这批素材里唯一值得进训练的。

2. **预标单独存 `labels_auto/`，绝不和 `labels/` 混**
   YOLO 里错一个框就是错一个监督信号。未复核的预标直接进训练集，等于用模型的
   错误循环喂自己。所以目录分开：人工复核后再合并。

3. **类别映射必须改**：他们的 txt 里 `0 = smoke`，我们的索引是 `0 = fire, 1 = smoke`。
   直接拷过去会把所有烟标成火 —— 这是最容易犯又最难发现的错。

产出
----
    <out>/images/            原图（只含大图）
    <out>/labels/            原始烟标，已重映射为 class 1
    <out>/labels_auto/       我们的火预标（class 0），待复核
    <out>/manifest.jsonl     每张一行的统计
    <out>/check_sheet.jpg    火预标核对表（--sheet 时）

跑法：
    python scripts/extract_smoke_zip_big.py --out E:/datasets/smoke_zip_big
"""
from __future__ import annotations

import argparse
import json
import sys
import zipfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

ZIP = Path(r"E:\BaiduNetdiskDownload\Y-烟雾24000.zip")
WEIGHTS = ROOT / "models" / "fire_smoke_v5_26s960_best.pt"
MIN_SIDE = 800                  # 最长边下限


def jpeg_size(buf: bytes) -> tuple[int, int] | None:
    """不解码只读 JPEG 头拿尺寸 —— 24574 张全解码太慢。"""
    i, n = 2, len(buf)
    while i < n - 9:
        if buf[i] != 0xFF:
            i += 1
            continue
        m = buf[i + 1]
        if m in (0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB):
            h = (buf[i + 5] << 8) | buf[i + 6]
            w = (buf[i + 7] << 8) | buf[i + 8]
            return w, h
        if m in (0xD8, 0xD9) or 0xD0 <= m <= 0xD7:
            i += 2
            continue
        i += 2 + ((buf[i + 2] << 8) | buf[i + 3])
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="E:/datasets/smoke_zip_big")
    ap.add_argument("--zip", default=str(ZIP))
    ap.add_argument("--min-side", type=int, default=MIN_SIDE)
    ap.add_argument("--fire-conf", type=float, default=0.45,
                    help="预标门限。宁可多标一点让复核删，也别漏（漏了就是假背景）")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--sheet", default="")
    a = ap.parse_args()

    import cv2
    out = Path(a.out)
    for d in ("images", "labels", "labels_auto"):
        (out / d).mkdir(parents=True, exist_ok=True)

    z = zipfile.ZipFile(a.zip)
    names = set(z.namelist())
    jpgs = sorted(n for n in names
                  if n.startswith("JPEGImages/") and n.lower().endswith(".jpg"))
    print(f"压缩包内 {len(jpgs)} 张，按最长边 ≥{a.min_side} 筛 …")

    big = []
    for n in jpgs:
        raw = z.read(n)
        wh = jpeg_size(raw)
        if wh and max(wh) >= a.min_side:
            big.append((n, raw, wh))
    if a.limit:
        big = big[:a.limit]
    print(f"大图 {len(big)} 张（占 {len(big) / len(jpgs) * 100:.1f}%）\n")

    from src.infer import Detector
    det = Detector(str(WEIGHTS), conf=0.20, imgsz=896, device=0)

    n_fire_img = 0
    n_fire_box = 0
    n_smoke_box = 0
    sheet_items = []
    with (out / "manifest.jsonl").open("w", encoding="utf-8") as mf:
        for k, (name, raw, (w, h)) in enumerate(big, 1):
            stem = Path(name).stem
            img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                continue
            (out / "images" / f"{stem}.jpg").write_bytes(raw)

            # 原始烟标 → class 1（他们是 0=smoke，我们是 0=fire/1=smoke）
            src = f"YOLOLabels/{stem}.txt"
            lines_out = []
            if src in names:
                for ln in z.read(src).decode("utf-8", "ignore").splitlines():
                    v = ln.split()
                    if len(v) >= 5:
                        lines_out.append("1 " + " ".join(v[1:5]))
                        n_smoke_box += 1
            (out / "labels" / f"{stem}.txt").write_text(
                "\n".join(lines_out) + ("\n" if lines_out else ""), encoding="utf-8")

            # 火预标
            ds = det.detect(img, sliced=True, tile_grid=2, overlap=0.25)
            fires = [d for d in ds if d.cls_id == 0 and d.conf >= a.fire_conf]
            alines = []
            for d in fires:
                cx = (d.xyxy[0] + d.xyxy[2]) / 2 / w
                cy = (d.xyxy[1] + d.xyxy[3]) / 2 / h
                bw = (d.xyxy[2] - d.xyxy[0]) / w
                bh = (d.xyxy[3] - d.xyxy[1]) / h
                alines.append(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
            (out / "labels_auto" / f"{stem}.txt").write_text(
                "\n".join(alines) + ("\n" if alines else ""), encoding="utf-8")
            if fires:
                n_fire_img += 1
                n_fire_box += len(fires)
                if len(sheet_items) < 12 and max(d.conf for d in fires) >= 0.60:
                    sheet_items.append((stem, img, fires, w, h))

            mf.write(json.dumps({
                "stem": stem, "w": w, "h": h,
                "smoke": len(lines_out), "fire_pre": len(fires),
                "fire_max": round(max((d.conf for d in fires), default=0.0), 3),
            }, ensure_ascii=False) + "\n")
            if k % 200 == 0:
                print(f"  {k}/{len(big)}  已预标火图 {n_fire_img} 张")

    print(f"\n完成：大图 {len(big)} 张")
    print(f"  原始烟框 {n_smoke_box} 个（已重映射为 class 1）")
    print(f"  火预标：{n_fire_img} 张图 / {n_fire_box} 个框"
          f"（{n_fire_img / max(1, len(big)) * 100:.1f}% 的图有火）")
    print(f"  输出目录 {out}")

    if a.sheet and sheet_items:
        from PIL import Image, ImageDraw, ImageFont
        F = ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", 15)
        FS = ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", 12)
        CW, CH, COLS = 300, 230, 4
        rows = (len(sheet_items) + COLS - 1) // COLS
        sheet = Image.new("RGB", (CW * COLS, (CH + 24) * rows), (255, 255, 255))
        dr = ImageDraw.Draw(sheet)
        for i, (stem, img, fires, w, h) in enumerate(sheet_items):
            vis = img.copy()
            for d in fires:
                cv2.rectangle(vis, (int(d.xyxy[0]), int(d.xyxy[1])),
                              (int(d.xyxy[2]), int(d.xyxy[3])), (40, 40, 230), 3)
            thumb = Image.fromarray(cv2.cvtColor(cv2.resize(vis, (CW, CH - 22)),
                                                cv2.COLOR_BGR2RGB))
            x, y = (i % COLS) * CW, (i // COLS) * (CH + 24)
            sheet.paste(thumb, (x, y + 22))
            dr.text((x + 3, y + 3), f"火预标 {max(d.conf for d in fires):.2f}  "
                                    f"{len(fires)} 框", font=F, fill=(170, 30, 30))
            dr.text((x + 3, y + 22 + CH - 20), stem, font=FS, fill=(70, 70, 70))
        sheet.save(a.sheet, quality=92)
        print(f"核对表已存 {a.sheet}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
