# eval_diag.py — 漏检分层诊断: 每根真烟卡在哪一层
# 层1: 头/手检测 → focus_regions 生成 | 层2: ROI裁剪放大SMOKE检出 | 层3: 交集判定 | 层4: 过滤链
import cv2, glob, os, sys
from ultralytics import YOLO

VAL = r"D:/training_data/smoke/v26_manual_orig/images/val"
HEAD = YOLO(r"D:\视觉安防系统\models\yolov8n_head_v7.pt", task='detect')
HAND = YOLO(r"D:\视觉安防系统\models\hand.pt", task='detect')
V30 = YOLO(r"D:\视觉安防系统\models\smoke_cig_v30.pt", task='detect')
HEAD.to('cuda'); HAND.to('cuda'); V30.to('cuda')
CONF_H, CONF_D, CONF_S = 0.55, 0.30, 0.06

def iou(b1, b2):
    ix1, iy1 = max(b1[0], b2[0]), max(b1[1], b2[1])
    ix2, iy2 = min(b1[2], b2[2]), min(b1[3], b2[3])
    inter = max(0, ix2-ix1) * max(0, iy2-iy1)
    u = (b1[2]-b1[0])*(b1[3]-b1[1]) + (b2[2]-b2[0])*(b2[3]-b2[1]) - inter
    return inter/u if u > 0 else 0

def detect(yolo, img, conf, imgsz):
    out = []
    r = yolo(img, conf=conf, iou=0.5, imgsz=imgsz, verbose=False)
    for rr in r:
        for b in rr.boxes:
            x1, y1, x2, y2 = b.xyxy[0].cpu().numpy()
            out.append((x1, y1, x2, y2))
    return out

def expand(b, W, H, kind):
    x1, y1, x2, y2 = b
    w_, h_ = x2-x1, y2-y1
    if kind == 'head':
        return (max(0,x1-w_*0.6), max(0,y1-h_*0.9), min(W,x2+w_*0.6), min(H,y2+h_*0.6))
    return (max(0,x1-w_*1.2), max(0,y1-h_*0.6), min(W,x2+w_*1.2), min(H,y2+h_*1.6))

# 分层统计
L = {'tot':0, 'no_region':0, 'region_no_det':0, 'det_no_touch':0, 'touch_filtered':0, 'pass_':0}
small_smoke = []   # 漏检烟在原始图里的尺寸
for f in sorted(glob.glob(os.path.join(VAL, '*.jpg'))):
    img = cv2.imread(f)
    if img is None: continue
    Hh, Ww = img.shape[:2]
    gt = []
    lb = f.replace('images','labels').replace('.jpg','.txt')
    if os.path.exists(lb):
        for line in open(lb):
            p = line.split()
            if len(p) >= 5:
                cx, cy, bw, bh = float(p[1])*Ww, float(p[2])*Hh, float(p[3])*Ww, float(p[4])*Hh
                gt.append((cx-bw/2, cy-bh/2, cx+bw/2, cy+bh/2))
    heads = detect(HEAD, img, CONF_H, 640)
    hands = detect(HAND, img, CONF_D, 640)
    regions = [expand(b, Ww, Hh, 'head') for b in heads] + [expand(b, Ww, Hh, 'hand') for b in hands]
    # 每根烟: 中心是否在区域内
    for (sx1, sy1, sx2, sy2) in gt:
        scx, scy = (sx1+sx2)/2, (sy1+sy2)/2
        L['tot'] += 1
        in_reg = any(rx1 <= scx <= rx2 and ry1 <= scy <= ry2 for (rx1,ry1,rx2,ry2) in regions)
        if not in_reg:
            L['no_region'] += 1
            small_smoke.append((sx2-sx1))
            continue
        # ROI 检测(与推理端相同: 每区域裁剪放大640跑V30)
        dets = []
        for (rx1, ry1, rx2, ry2) in regions[:4]:
            rw, rh = int(rx2-rx1), int(ry2-ry1)
            if rw < 24 or rh < 24: continue
            roi = img[int(ry1):int(ry2), int(rx1):int(rx2)]
            sc = 640.0/max(rw, rh)
            tw, th = int(rw*sc+0.5), int(rh*sc+0.5)
            big = cv2.resize(roi, (tw, th), interpolation=cv2.INTER_LINEAR)
            for d in detect(V30, big, CONF_S, 640):
                bx1, by1, bx2, by2 = d
                dets.append((rx1+bx1/sc, ry1+by1/sc, rx1+bx2/sc, ry1+by2/sc))
        # 该烟被检出?
        hit = any(iou(d, (sx1,sy1,sx2,sy2)) > 0.5 for d in dets)
        if not hit:
            L['region_no_det'] += 1
            small_smoke.append((sx2-sx1))
            continue
        # 交集判定
        touch = any(iou(d, (sx1,sy1,sx2,sy2)) > 0.5 and
                    (any(iou(d, hb) > 0.0 for hb in hands) or any(iou(d, hb2) > 0.0 for hb2 in heads))
                    for d in dets)
        if not touch:
            L['det_no_touch'] += 1
            continue
        L['pass_'] += 1

print("=== 漏检分层诊断(72张全图, v30管线) ===")
for k, v in L.items():
    print(f"  {k}: {v}")
import numpy as np
if small_smoke:
    a = np.array(small_smoke)
    print(f"漏检烟的原始宽度px: min={a.min():.0f} 中位={np.median(a):.0f} max={a.max():.0f}")
