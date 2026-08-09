# prep_handobj.py — 生成"手持物判断"训练数据
# 正样本(手拿烟): v26_manual_orig 用 hand.pt 裁剪手部ROI(含指尖/掌心区域)
# 输出: D:/training_data/handobj/with_obj/  (分类器正样本)
# 注意: 手框扩张一点(指尖夹烟处可能略出手框), 保存原始尺寸ROI
import cv2, glob, os, shutil
from pathlib import Path
from ultralytics import YOLO

HAND = YOLO(r'D:\视觉安防系统\models\hand.pt', task='detect')
HAND.to('cuda')
SRC = Path(r'D:/training_data/smoke/v26_manual_orig/images/train')
DST = Path(r'D:/training_data/handobj/with_obj')
DST.mkdir(parents=True, exist_ok=True)
CONF = 0.25

n_saved = n_img = 0
for f in sorted(glob.glob(str(SRC / '*.jpg'))):
    img = cv2.imread(f)
    if img is None: continue
    n_img += 1
    Hh, Ww = img.shape[:2]
    r = HAND(img, conf=CONF, iou=0.5, imgsz=640, verbose=False)
    boxes = []
    for rr in r:
        for b in rr.boxes:
            boxes.append(b.xyxy[0].cpu().numpy())
    for i, (x1, y1, x2, y2) in enumerate(boxes):
        w_, h_ = x2-x1, y2-y1
        if w_ < 40 or h_ < 40: continue   # 太小的手(远距)跳过
        # 扩张15%包含指尖/掌心周边
        ex = 0.15
        ax1, ay1 = max(0, int(x1-w_*ex)), max(0, int(y1-h_*ex))
        ax2, ay2 = min(Ww, int(x2+w_*ex)), min(Hh, int(y2+h_*ex))
        crop = img[ay1:ay2, ax1:ax2]
        name = f'{Path(f).stem}_h{i}.jpg'
        cv2.imwrite(str(DST / name), crop)
        n_saved += 1

print(f'手持烟正样本: {n_img} 张图 → {n_saved} 个手ROI → {DST}')
