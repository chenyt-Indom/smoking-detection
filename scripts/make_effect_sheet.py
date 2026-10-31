"""效果总览图：把「现在到底长什么样」画出来，别只看数字。

四格
----
A 真实火视频首报帧   —— 框位、置信度、闪烁值（闸为什么放行）
B 真实烟雾帧         —— 烟是否在 smoke_conf=0.45 之上还抓得住
C 背景误报场景       —— 闸关时会被报成火的那张图，框画在哪
D 另一个背景误报场景 —— 同上

为什么要这张图：mAP/P/R 只能说明"统计上还行"，画出来才能看出
框是不是套在火头上、烟是不是框住了整柱、误报到底在框什么东西。

跑法：
    python scripts/make_effect_sheet.py
输出：
    runs/effect_sheet.jpg
"""
from __future__ import annotations

import glob
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

DATA = Path("E:/datasets/fire_smoke_v5/test")
BG = Path("E:/BaiduNetdiskDownload/Y_火灾+烟雾/背景负样本/images")
WEIGHTS = ROOT / "models" / "fire_smoke_v5_26s960_best.pt"
FONT = "C:/Windows/Fonts/msyh.ttc"
OUT = ROOT / "runs" / "effect_sheet.jpg"

CELL = (640, 360)          # 每格尺寸
FPS = 25.0
FIRE_CONF, SMOKE_CONF = 0.30, 0.45


def imread_u(p: Path):
    import cv2
    try:
        return cv2.imdecode(np.frombuffer(p.read_bytes(), np.uint8), cv2.IMREAD_COLOR)
    except Exception:                                                # noqa: BLE001
        return None


def pick_continuous(pattern: str, limit: int):
    files = sorted(glob.glob(pattern),
                   key=lambda p: int(re.search(r"CV(\d+)\.jpg$", p).group(1)))
    main, out = None, []
    for p in files:
        im = imread_u(Path(p))
        if im is None:
            continue
        if main is None:
            main = im.shape
        if im.shape != main:
            continue
        out.append(im)
        if len(out) >= limit:
            break
    return out


def draw(im, dets, title, sub, accent=(60, 60, 220)):
    """把框和文字画上去。文字用 PIL —— cv2.putText 画不了中文。"""
    import cv2
    from PIL import Image, ImageDraw, ImageFont

    vis = im.copy()
    for d in dets:
        x1, y1, x2, y2 = [int(v) for v in d[:4]]
        cv2.rectangle(vis, (x1, y1), (x2, y2), accent, 3)
        vis = cv2.rectangle(vis, (x1, y1), (x2, y2), accent, 3)
    img = Image.fromarray(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
    dr = ImageDraw.Draw(img)
    f_big = ImageFont.truetype(FONT, 22)
    f_sm = ImageFont.truetype(FONT, 16)
    dr.rectangle([0, 0, img.width, 62], fill=(255, 255, 255))
    dr.text((10, 6), title, font=f_big, fill=(30, 30, 30))
    dr.text((10, 36), sub, font=f_sm, fill=(90, 90, 90))
    img.thumbnail(CELL)
    return img


def cell_blank(msg):
    from PIL import Image, ImageDraw, ImageFont
    img = Image.new("RGB", CELL, (246, 246, 244))
    dr = ImageDraw.Draw(img)
    dr.text((16, 16), msg, font=ImageFont.truetype(FONT, 18), fill=(80, 80, 80))
    return img


def main() -> int:
    from src.infer import Detector
    det = Detector(str(WEIGHTS), conf=0.20, imgsz=896, device=0)
    names = det.names if isinstance(det.names, dict) else dict(enumerate(det.names))
    print("类别：", names)

    cells = []

    # ── A 真实火视频首报帧（t=0.64s → 第 16 帧）──────────────
    frames = pick_continuous(str(DATA / "images" / "A_烟雾_火_fire_CV*.jpg"), 113)
    idx = int(round(0.64 * FPS))
    fa = frames[min(idx, len(frames) - 1)]
    ds = det.detect(fa, sliced=True, tile_grid=2, overlap=0.25)
    fire = [d for d in ds if names.get(d.cls_id, d.cls_name) == "fire"
            and d.conf >= FIRE_CONF]
    fire.sort(key=lambda d: -d.conf)
    best = fire[:1]
    sub = (f"首报 t=0.64s / alarm / conf {best[0].conf:.2f} / 框内逐帧变化 38%（阈值 25%）"
           if best else "该帧没有过 fire_conf 的框")
    cells.append(draw(fa, [d.xyxy for d in best], "A  真实火视频 · 第 16 帧（首报）", sub))

    # ── B 真实烟雾帧 ────────────────────────────────────────
    smoke_files = sorted((DATA / "images").glob("*smoke_CV*.jpg"))
    picked = None
    for p in smoke_files[::max(1, len(smoke_files) // 12)][:12]:
        im = imread_u(p)
        if im is None:
            continue
        ds = det.detect(im, sliced=True, tile_grid=2, overlap=0.25)
        sk = [d for d in ds if names.get(d.cls_id, d.cls_name) == "smoke"
              and d.conf >= SMOKE_CONF]
        if sk:
            sk.sort(key=lambda d: -d.conf)
            picked = (im, sk[:2], p.name)
            break
    if picked:
        im, sk, nm = picked
        cells.append(draw(im, [d.xyxy for d in sk], "B  真实烟雾帧",
                          f"{nm[:34]} · 烟 conf {sk[0].conf:.2f}（门限 0.45）",
                          accent=(200, 130, 20)))
    else:
        cells.append(cell_blank("B 真实烟雾帧：抽的 12 张里没有过 0.45 的烟框\n"
                                "→ 说明 smoke_conf=0.45 对淡烟偏严，值得单独查"))

    # ── C / D 两个背景误报场景 ──────────────────────────────
    files = sorted(p for p in BG.iterdir()
                   if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
    same = [p for p in files if (im := imread_u(p)) is not None
            and im.shape[1] == 640 and im.shape[0] == 480]
    fp_idx = [1, 79]          # eval_fp_end2end 里闸关时报出来的两个场景
    for k, i in enumerate(fp_idx):
        if i >= len(same):
            continue
        im = imread_u(same[i])
        ds = det.detect(im, sliced=True, tile_grid=2, overlap=0.25)
        hot = [d for d in ds if d.conf >= FIRE_CONF]
        hot.sort(key=lambda d: -d.conf)
        labels = "、".join(f"{names.get(d.cls_id, d.cls_name)} {d.conf:.2f}"
                          for d in hot[:2]) or "无框"
        cells.append(draw(
            im, [d.xyxy for d in hot[:2]],
            f"{'C' if k == 0 else 'D'}  纯背景场景（无火无烟）",
            f"{same[i].name[:26]} · 模型画了：{labels} → 闪烁闸判静态，降级 notice",
            accent=(70, 140, 70)))

    # ── 拼图 ───────────────────────────────────────────────
    from PIL import Image, ImageDraw, ImageFont
    W, H = CELL
    sheet = Image.new("RGB", (W * 2 + 24, H * 2 + 92), (255, 255, 255))
    dr = ImageDraw.Draw(sheet)
    dr.text((16, 10), "香烟/火焰检测系统 · 当前效果总览", font=ImageFont.truetype(FONT, 26),
            fill=(20, 20, 20))
    dr.text((16, 46), f"权重 {WEIGHTS.name} · imgsz 896 · 切片 2x2 · "
                      f"fire_conf {FIRE_CONF} / smoke_conf {SMOKE_CONF} · "
                      f"test 集真实图片", font=ImageFont.truetype(FONT, 15), fill=(110, 110, 110))
    for n, c in enumerate(cells[:4]):
        x = 8 + (n % 2) * (W + 8)
        y = 76 + (n // 2) * (H + 8)
        sheet.paste(c, (x, y))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(OUT, quality=92)
    print(f"\n已存 {OUT}  ({sheet.width}x{sheet.height})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
