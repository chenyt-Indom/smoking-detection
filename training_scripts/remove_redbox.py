# remove_redbox.py — 擦除采集图中的红框(模型画上的检测框)
# 红框位置来自 labels txt(归一化坐标), 用inpaint修复
import cv2, numpy as np, os
from pathlib import Path

SRC = Path(r'D:\training_data\smoke\fp_capture3')
DST = Path(r'D:\training_data\smoke\fp_capture3_clean')
n = 0
for lblp in sorted(SRC.rglob('*.txt')):
    # labels/*.txt → images/*.jpg
    rel = lblp.relative_to(SRC)
    parts = list(rel.parts)
    for i, p in enumerate(parts):
        if p == 'labels':
            parts[i] = 'images'
    imgp = SRC.joinpath(*parts).with_suffix('.jpg')
    img = cv2.imread(str(imgp))
    if img is None:
        print('跳过(无图):', imgp)
        continue
    h, w = img.shape[:2]
    mask = np.zeros(img.shape[:2], np.uint8)
    for line in lblp.read_text().strip().splitlines():
        p = line.split()
        if len(p) < 5:
            continue
        cx, cy, bw, bh = float(p[1])*w, float(p[2])*h, float(p[3])*w, float(p[4])*h
        x1, y1 = int(cx-bw/2), int(cy-bh/2)
        x2, y2 = int(cx+bw/2), int(cy+bh/2)
        cv2.rectangle(mask, (x1, y1), (x2, y2), 255, 6)   # 掩码覆盖红框线(2px粗, 6px保险)
    if mask.max() > 0:
        img = cv2.inpaint(img, mask, 5, cv2.INPAINT_TELEA)
    out = DST / rel.parent.parent / imgp.name   # 保持 images/xxx.jpg 结构
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), img)
    n += 1
print(f'红框擦除完成: {n} 张 -> {DST}')
