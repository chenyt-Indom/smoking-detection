"""试跑 v3 火烟模型的训练效果。

1) 测试集抽样预测 -> 拼图 + 明细表（自己画框，避免 ultralytics 的排版问题）
2) 负样本误报测试：喂两段「没有人/头之外异常」的视频，统计是否误报 fire/smoke
"""
from __future__ import annotations

import glob
import json
import random
import sys
from pathlib import Path

import cv2
import numpy as np

WEIGHTS = r"D:/视觉安防系统/runs/fire_smoke_v3_26n/weights/best.pt"
TEST_IMGS = r"E:/datasets/fire_smoke_v3/test/images"
OUT = Path(r"E:/2026-10-30-08-23-36/v3_test")
NEG_VIDEOS = [
    r"D:/视觉安防系统/training_data/test_videos/test_crowd.mp4",
    r"D:/视觉安防系统/training_data/test_videos/test_heads_moving.mp4",
]
COLOR = {"fire": (0, 90, 255), "smoke": (255, 160, 0)}


def imread_u(p):
    return cv2.imdecode(np.fromfile(str(p), dtype=np.uint8), cv2.IMREAD_COLOR)


def imwrite_u(p, img, q=90):
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, q])
    buf.tofile(str(p))


def main() -> int:
    from ultralytics import YOLO

    OUT.mkdir(parents=True, exist_ok=True)
    model = YOLO(WEIGHTS)
    names = model.names

    # ---------- 1) 测试集抽样 ----------
    imgs = sorted(glob.glob(TEST_IMGS + "/*"))
    random.seed(2026)
    sel = random.sample(imgs, 16)
    res = model.predict(sel, imgsz=640, conf=0.25, verbose=False)

    cell = 420
    grid = np.full((cell * 4, cell * 4, 3), 250, np.uint8)
    rows = []
    for i, (r, path) in enumerate(zip(res, sel)):
        im = r.orig_img.copy()
        h, w = im.shape[:2]
        dets = []
        for b in r.boxes:
            cls = names[int(b.cls)]
            cf = float(b.conf)
            x1, y1, x2, y2 = [int(v) for v in b.xyxy[0].tolist()]
            c = COLOR.get(cls, (0, 200, 0))
            cv2.rectangle(im, (x1, y1), (x2, y2), c, 2)
            tag = f"{cls} {cf:.2f}"
            (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
            cv2.rectangle(im, (x1, max(0, y1 - th - 6)), (x1 + tw + 4, y1), c, -1)
            cv2.putText(im, tag, (x1 + 2, max(12, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
            dets.append((cls, round(cf, 3)))
        # GT 框数（同名 txt）
        gt = 0
        lbl = Path(str(path).replace("/images/", "/labels/").rsplit(".", 1)[0] + ".txt")
        if lbl.exists():
            gt = sum(1 for line in lbl.read_text().splitlines() if line.strip())
        rows.append({"img": Path(path).name, "gt": gt, "pred": len(dets), "dets": dets})

        imwrite_u(OUT / f"p{i:02d}.jpg", im)
        s = min(cell / w, cell / h)
        th_im = cv2.resize(im, (int(w * s), int(h * s)))
        gr, gc = divmod(i, 4)
        y, x = gr * cell, gc * cell
        grid[y:y + th_im.shape[0], x:x + th_im.shape[1]] = th_im
        cv2.putText(grid, f"#{i} gt={gt} pred={len(dets)}", (x + 6, y + cell - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (30, 30, 200), 1, cv2.LINE_AA)
    imwrite_u(OUT / "grid_test.jpg", grid)

    print("=" * 78)
    print(f"{'图':<22}{'GT框':>6}{'预测框':>8}   明细")
    for row in rows:
        print(f"{row['img'][:21]:<22}{row['gt']:>6}{row['pred']:>8}   {row['dets']}")

    # ---------- 2) 负样本视频误报 ----------
    neg = []
    for vp in NEG_VIDEOS:
        if not Path(vp).exists():
            continue
        cap = cv2.VideoCapture(vp)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        step = max(1, int(round(fps)))  # 每秒 1 帧
        idx = hit = total = 0
        worst = 0.0
        frames_to_save = []
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if idx % step == 0:
                total += 1
                r = model.predict(frame, imgsz=640, conf=0.25, verbose=False)[0]
                if len(r.boxes):
                    hit += 1
                    worst = max(worst, float(r.boxes.conf.max()))
                    if len(frames_to_save) < 2:
                        frames_to_save.append(r.plot()[:, :, ::-1])
            idx += 1
        cap.release()
        neg.append({"video": Path(vp).name, "frames": total, "fp_frames": hit,
                    "fp_rate": round(hit / max(total, 1), 4), "max_conf": round(worst, 3)})
        for j, fr in enumerate(frames_to_save):
            imwrite_u(OUT / f"neg_{Path(vp).stem}_{j}.jpg", np.ascontiguousarray(fr))
        print(f"[负样本] {Path(vp).name}: {total} 帧抽查，误报 {hit} 帧（{hit / max(total, 1):.1%}），最高 conf {worst:.2f}")

    (OUT / "report.json").write_text(
        json.dumps({"weights": WEIGHTS, "test_samples": rows, "negative_videos": neg},
                   ensure_ascii=False, indent=2), encoding="utf-8")
    print("=" * 78)
    print("输出目录：", OUT)
    return 0


if __name__ == "__main__":
    sys.exit(main())
