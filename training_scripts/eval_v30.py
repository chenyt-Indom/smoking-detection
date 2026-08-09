# eval_v30.py — A/B 对比: 在 v30 验证集(区域图)上 V28 vs V30 检出效果
# 判定"区域→烟"训练是否真的有效(硬证据, 不看mAP数字, 看实际检出)
import cv2, glob, os, sys
from ultralytics import YOLO

VAL = r"D:/training_data/smoke/v30_regions/images/val"
V28 = YOLO(r"D:\视觉安防系统\models\smoke_cig_v28.pt", task='detect')
V30 = YOLO(r"D:/training_data/runs/detect/cig_v30/weights/best.pt", task='detect')
V28.to('cuda'); V30.to('cuda')
CONF = 0.06

def iou(b1, b2):
    ix1, iy1 = max(b1[0], b2[0]), max(b1[1], b2[1])
    ix2, iy2 = min(b1[2], b2[2]), min(b1[3], b2[3])
    inter = max(0, ix2-ix1) * max(0, iy2-iy1)
    u = (b1[2]-b1[0])*(b1[3]-b1[1]) + (b2[2]-b2[0])*(b2[3]-b2[1]) - inter
    return inter/u if u > 0 else 0

def run(model, img):
    out = []
    r = model(img, conf=CONF, iou=0.5, imgsz=640, verbose=False)
    for rr in r:
        for b in rr.boxes:
            x1, y1, x2, y2 = b.xyxy[0].cpu().numpy()
            out.append((float(b.conf[0]), x1, y1, x2, y2))
    return out

stats = {m: {'det': 0, 'tp': 0, 'fp': 0, 'hit': 0, 'pos': 0, 'conf_sum': 0.0} for m in ('V28', 'V30')}
n_imgs = 0
for f in sorted(glob.glob(os.path.join(VAL, '*.jpg'))):
    img = cv2.imread(f)
    if img is None: continue
    n_imgs += 1
    h, w = img.shape[:2]
    gt = []
    lb = f.replace('images', 'labels').replace('.jpg', '.txt')
    if os.path.exists(lb):
        for line in open(lb):
            p = line.split()
            if len(p) >= 5:
                cx, cy, bw, bh = float(p[1])*w, float(p[2])*h, float(p[3])*w, float(p[4])*h
                gt.append((cx-bw/2, cy-bh/2, cx+bw/2, cy+bh/2))
    for name, model in (('V28', V28), ('V30', V30)):
        s = stats[name]
        dets = run(model, img)
        s['det'] += len(dets)
        tp = 0
        for d in dets:
            s['conf_sum'] += d[0]
            if any(iou(d[1:], g) > 0.5 for g in gt):
                tp += 1
        s['tp'] += tp
        s['fp'] += len(dets) - tp
        if gt and tp > 0:
            s['hit'] += 1
        if gt:
            s['pos'] += 1
    if n_imgs % 25 == 0:
        print(f"  已处理 {n_imgs} 张...", flush=True)

print("\n=== A/B 对比结果(区域图验证集, conf=0.06, imgsz=640) ===")
for m in ('V28', 'V30'):
    s = stats[m]
    prec = s['tp'] / max(1, s['det'])
    rec = s['tp'] / max(1, s['pos'])
    hit = s['hit'] / max(1, s['pos'])
    avg_conf = s['conf_sum'] / max(1, s['det'])
    print(f"{m}: 检测框={s['det']} TP={s['tp']} FP={s['fp']} "
          f"| precision={prec:.3f} recall={rec:.3f} 含烟图检出率={hit:.3f} 平均conf={avg_conf:.3f}")
print(f"共 {n_imgs} 张验证图")
