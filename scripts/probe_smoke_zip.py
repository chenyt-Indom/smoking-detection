"""外部烟雾集（Y-烟雾24000.zip）可用性评估。

用户原话："上面可能有火但是里面是没有标出来的" —— 这是**致命风险**：
YOLO 里没框 = 这张图那块**没有目标**。把带火的图只标烟喂进去，
等于在教模型"火是背景"，火召回会直接塌，而且塌得毫无征兆。

量四件事：
  ① 有多少张图，我们的火模型能在上面找到火（= 未标注的火）
  ② 他们的烟框面积分布 —— 是不是补到了我们缺的"小面积/淡烟"（阴燃）
  ③ 我们在他们标注上的召回：**同时报 IoU 0.3 / 0.5 / 0.7**
     （IoU 0.5 算不中不一定是漏检，也可能他们框太松。松框会污染我们的框尺度）
  ④ 出一张核对图：上半排"有火但没标注"，下半排"他们的大烟框"

检测参数：imgsz 640、**不切片**。图只有 240x240，切 2x2 等于把 120px 的块硬拉到
896（放大 7 倍）只会引入伪影 —— 这个尺寸不需要切片。

跑法：
    python scripts/probe_smoke_zip.py --n 400
    python scripts/probe_smoke_zip.py --n 400 --sheet runs/smoke_zip_check.jpg
"""
from __future__ import annotations

import argparse
import collections
import sys
import zipfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

ZIP = Path(r"E:\BaiduNetdiskDownload\Y-烟雾24000.zip")
WEIGHTS = ROOT / "models" / "fire_smoke_v5_26s960_best.pt"

BUCKETS = [(0.0, 0.005, "<0.5%"), (0.005, 0.01, "0.5~1%"), (0.01, 0.03, "1~3%"),
           (0.03, 0.10, "3~10%"), (0.10, 1.01, ">10%")]


def iou(a, b) -> float:
    ix = min(a[2], b[2]) - max(a[0], b[0])
    iy = min(a[3], b[3]) - max(a[1], b[1])
    if ix <= 0 or iy <= 0:
        return 0.0
    inter = ix * iy
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--zip", default=str(ZIP))
    ap.add_argument("--sheet", default="", help="输出核对图")
    a = ap.parse_args()

    import cv2
    z = zipfile.ZipFile(a.zip)
    allnames = set(z.namelist())
    jpgs = sorted(n for n in allnames
                  if n.startswith("JPEGImages/") and n.lower().endswith(".jpg"))
    step = max(1, len(jpgs) // a.n)
    picked = jpgs[::step][:a.n]
    print(f"共 {len(jpgs)} 张，均匀抽 {len(picked)} 张\n")

    from src.infer import Detector
    det = Detector(str(WEIGHTS), conf=0.15, imgsz=a.imgsz, device=0)

    n_img = 0
    fire_hit = {0.30: 0, 0.50: 0, 0.70: 0}
    box_total = 0
    buckets = collections.Counter()
    rec = {0.3: [0, 0], 0.5: [0, 0], 0.7: [0, 0]}
    bucket_hit = collections.Counter()
    fire_cases = []
    big_box_cases = []

    for name in picked:
        im = cv2.imdecode(np.frombuffer(z.read(name), np.uint8), cv2.IMREAD_COLOR)
        if im is None:
            continue
        n_img += 1
        h, w = im.shape[:2]
        stem = Path(name).stem
        lab = f"YOLOLabels/{stem}.txt"
        gts = []
        if lab in allnames:
            for ln in z.read(lab).decode("utf-8", "ignore").splitlines():
                v = ln.split()
                if len(v) >= 5:
                    cx, cy, bw, bh = map(float, v[1:5])
                    bb = ((cx - bw / 2) * w, (cy - bh / 2) * h,
                          (cx + bw / 2) * w, (cy + bh / 2) * h)
                    gts.append((bb, bw * bh))
        for _bb, ar in gts:
            box_total += 1
            for lo, hi, tag in BUCKETS:
                if lo <= ar < hi:
                    buckets[tag] += 1
                    break

        ds = det.detect(im, sliced=False)
        fires = [d for d in ds if d.cls_id == 0]
        mx = max((d.conf for d in fires), default=0.0)
        for t in fire_hit:
            if mx >= t:
                fire_hit[t] += 1
        smk = [(d.xyxy, d.conf) for d in ds if d.cls_id == 1 and d.conf >= 0.45]

        for bb, ar in gts:
            tag = next((tb for lo, hi, tb in BUCKETS if lo <= ar < hi), "?")
            for th in rec:
                rec[th][1] += 1
                if any(iou(bb, sb) >= th for sb, _c in smk):
                    rec[th][0] += 1
                    if th == 0.5:
                        bucket_hit[tag] += 1

        if mx >= 0.50 and len(fire_cases) < 6:
            fire_cases.append((stem, im, max(fires, key=lambda d: d.conf).xyxy, mx))
        if len(big_box_cases) < 6:
            if gts:
                biggest = max(gts, key=lambda x: x[1])
                if biggest[1] > 0.10:
                    big_box_cases.append((stem, im, biggest[0], biggest[1]))

    print("① 未标注的火（我们的火模型自己找到的）")
    print(f"{'门限':>8}{'命中图数':>10}{'占抽样':>10}")
    for t, v in fire_hit.items():
        print(f"{t:>8.2f}{v:>10d}{v / max(1, n_img) * 100:>9.1f}%")

    print(f"\n② 他们的烟框面积分布（共 {box_total} 个框）")
    for lo, hi, tag in BUCKETS:
        c = buckets[tag]
        print(f"  {tag:>8}  {c:>6d}  {c / max(1, box_total) * 100:5.1f}%")

    print("\n③ 我们在他们标注上的召回（三种 IoU 严格度）")
    for th, (h_, t_) in rec.items():
        print(f"  IoU {th:.1f}: {h_}/{t_} = {h_ / max(1, t_) * 100:.0f}%")
    print("  按面积档（IoU 0.5）：")
    for lo, hi, tag in BUCKETS:
        t_ = buckets[tag]
        if t_:
            print(f"    {tag:>8}  {bucket_hit[tag]:>5d}/{t_:<5d} "
                  f"{bucket_hit[tag] / t_ * 100:5.0f}%")

    if a.sheet and (fire_cases or big_box_cases):
        from PIL import Image, ImageDraw, ImageFont
        F = ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", 15)
        FS = ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", 12)
        CW, CH, COLS = 260, 210, 6
        sheet = Image.new("RGB", (CW * COLS, (CH + 26) * 2), (255, 255, 255))
        dr = ImageDraw.Draw(sheet)
        for i, (stem, im, bb, cf) in enumerate(fire_cases):
            vis = im.copy()
            cv2.rectangle(vis, (int(bb[0]), int(bb[1])), (int(bb[2]), int(bb[3])),
                          (40, 40, 230), 2)
            th = Image.fromarray(cv2.cvtColor(cv2.resize(vis, (CW, CH - 24)),
                                              cv2.COLOR_BGR2RGB))
            sheet.paste(th, (i * CW, 26))
            dr.text((i * CW + 3, 5), f"有火未标注 {cf:.2f}", font=F, fill=(180, 30, 30))
            dr.text((i * CW + 3, 26 + CH - 24), stem, font=FS, fill=(70, 70, 70))
        for i, (stem, im, bb, ar) in enumerate(big_box_cases):
            vis = im.copy()
            cv2.rectangle(vis, (int(bb[0]), int(bb[1])), (int(bb[2]), int(bb[3])),
                          (30, 150, 40), 2)
            th = Image.fromarray(cv2.cvtColor(cv2.resize(vis, (CW, CH - 24)),
                                              cv2.COLOR_BGR2RGB))
            sheet.paste(th, (i * CW, 26 + CH + 26))
            dr.text((i * CW + 3, CH + 26 + 5), f"他们的烟框 占画面 {ar * 100:.0f}%",
                    font=F, fill=(30, 120, 40))
            dr.text((i * CW + 3, 26 + CH + 26 + CH - 24), stem, font=FS, fill=(70, 70, 70))
        Path(a.sheet).parent.mkdir(parents=True, exist_ok=True)
        sheet.save(a.sheet, quality=92)
        print(f"\n核对图已存 {a.sheet}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
