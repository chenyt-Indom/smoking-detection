# eval_pipeline.py — 新旧检测管线 A/B(真实全图场景)
# 旧管线: 全图1024 V28 + 候选中心落在头/手扩张区域内才保留
# 新管线: 头/手框 → 裁剪扩张区域 → 放大640 → V30 → 框换算回全图
# 数据: v26_manual_orig/val (72张真实手持烟全图, 用户标注)
import cv2, glob, os
from ultralytics import YOLO

VAL = r"D:/training_data/smoke/v26_manual_orig/images/val"
HEAD = YOLO(r"D:\视觉安防系统\models\yolov8n_head_v7.pt", task='detect')
HAND = YOLO(r"D:\视觉安防系统\models\hand.pt", task='detect')
V28 = YOLO(r"D:\视觉安防系统\models\smoke_cig_v28.pt", task='detect')
V30 = YOLO(r"D:\视觉安防系统\models\smoke_cig_v30.pt", task='detect')
HEAD.to('cuda'); HAND.to('cuda'); V28.to('cuda'); V30.to('cuda')
CONF_H, CONF_D, CONF_S = 0.55, 0.38, 0.06

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
            out.append((float(b.conf[0]), x1, y1, x2, y2))
    return out

def expand(b, W, H, kind):
    x1, y1, x2, y2 = b
    w_, h_ = x2-x1, y2-y1
    if kind == 'head':
        return (max(0,x1-w_*0.6), max(0,y1-h_*0.9), min(W,x2+w_*0.6), min(H,y2+h_*0.6))
    return (max(0,x1-w_*1.2), max(0,y1-h_*0.6), min(W,x2+w_*1.2), min(H,y2+h_*1.6))

def pipeline_old(frame, regions, W, H):
    """旧: 全图V28, 候选中心在区域内"""
    cands = detect(V28, frame, CONF_S, 1024)
    out = []
    for c in cands:
        cx, cy = (c[1]+c[3])/2, (c[2]+c[4])/2
        if any(rx1 <= cx <= rx2 and ry1 <= cy <= ry2 for (rx1,ry1,rx2,ry2) in regions):
            out.append(c[1:])
    return out

def pipeline_new(frame, regions, W, H):
    """新: 区域裁剪放大640 → V30 → 换算回全图"""
    out = []
    for (rx1, ry1, rx2, ry2) in regions[:4]:
        rw, rh = rx2-rx1, ry2-ry1
        if rw < 24 or rh < 24: continue
        roi = frame[int(ry1):int(ry2), int(rx1):int(rx2)]
        sc = 640.0 / max(rw, rh)
        tw, th = int(rw*sc+0.5), int(rh*sc+0.5)
        if tw < 16 or th < 16: continue
        big = cv2.resize(roi, (tw, th), interpolation=cv2.INTER_LINEAR)
        for d in detect(V30, big, CONF_S, 640):
            bx1, by1, bx2, by2 = d[1:]
            out.append((rx1+bx1/sc, ry1+by1/sc, rx1+bx2/sc, ry1+by2/sc))
    return out

stats = {'old': {'det':0,'tp':0,'fp':0,'hit':0,'pos':0}, 'new': {'det':0,'tp':0,'fp':0,'hit':0,'pos':0}}
t_old = t_new = 0.0
import time
for f in sorted(glob.glob(os.path.join(VAL, '*.jpg')))[:40]:
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
    # 头/手检测(共用)
    heads = [b[1:] for b in detect(HEAD, img, CONF_H, 640)]
    hands = [b[1:] for b in detect(HAND, img, CONF_D, 640)]
    regions = [expand(b, Ww, Hh, 'head') for b in heads] + [expand(b, Ww, Hh, 'hand') for b in hands]
    for name, fn in (('old', pipeline_old), ('new', pipeline_new)):
        s = stats[name]
        t0 = time.time()
        dets = fn(img, regions, Ww, Hh) if name=='old' else fn(img, regions, Ww, Hh)
        dt = time.time()-t0
        if name=='old': t_old += dt
        else: t_new += dt
        s['det'] += len(dets)
        tp = sum(1 for d in dets if any(iou(d, g) > 0.5 for g in gt))
        s['tp'] += tp; s['fp'] += len(dets)-tp
        if gt and tp > 0: s['hit'] += 1
        if gt: s['pos'] += 1

print("\n=== 完整管线 A/B(真实全图, 前40张) ===")
for name in ('old', 'new'):
    s = stats[name]
    prec = s['tp']/max(1,s['det']); rec = s['tp']/max(1,s['pos'])
    hit = s['hit']/max(1,s['pos'])
    print(f"{name}: 框={s['det']} TP={s['tp']} FP={s['fp']} | precision={prec:.3f} recall={rec:.3f} 图检出率={hit:.3f}")
print(f"耗时: old={t_old:.2f}s new={t_new:.2f}s")
