"""防过拟合训练 v3 — 修复版"""
import cv2, numpy as np, random, time, shutil
from pathlib import Path
from ultralytics import YOLO

OUT = Path("D:/视觉安防系统/training_data/datasets/synthetic")
for d in OUT.glob("**/*"):
    if d.is_file(): d.unlink()
for s in ["train/images","train/labels","val/images","val/labels"]:
    (OUT / s).mkdir(parents=True, exist_ok=True)

N_TRAIN, N_VAL = 8000, 2000
W, H = 640, 640
np.random.seed(123456)
random.seed(123456)

def make_bg():
    tp = random.randint(0, 5)
    if tp == 0:
        b = np.random.randint(50, 220, (H, W, 3), dtype=np.uint8)
        b = cv2.blur(b, (5, 5))
    elif tp == 1:
        b = np.zeros((H, W, 3), dtype=np.uint8)
        for y in range(H):
            v = int(80 + 100 * y / H)
            b[y, :] = np.clip(v + np.random.randint(-15, 15, 3), 20, 240)
    elif tp == 2:
        b = np.random.randint(80, 180, (H, W, 3), dtype=np.uint8)
    elif tp == 3:
        c1 = np.random.randint(60, 200, 3); c2 = np.random.randint(60, 200, 3)
        b = np.tile(c1, (H, W, 1)).astype(np.uint8)
        for py in range(0, H, 40):
            for px in range(0, W, 40):
                if (px // 40 + py // 40) % 2:
                    b[py:py+40, px:px+40] = c2
    elif tp == 4:
        b = np.random.randint(30, 200, (H, W, 3), dtype=np.uint8)
        for _ in range(random.randint(5, 15)):
            rx, ry = random.randint(0, W-40), random.randint(0, H-40)
            b[ry:ry+random.randint(20, 80), rx:rx+random.randint(20, 120)] = np.random.randint(60, 220, 3)
    else:
        b = np.full((H, W, 3), np.random.randint(50, 200), dtype=np.uint8)
    if random.random() < 0.3:
        b = cv2.blur(b, (3, 3))
    return b

def person(bg, cx, cy, s, clr, aug=True):
    bw, bh = max(12, int(36*s)), max(18, int(90*s))
    hr, lw = max(7, int(13*s)), max(4, bw//3)
    x1, x2 = cx-bw, cx+bw
    y1, y2 = cy-bh//2-hr*2, cy+bh
    if x1 < 3 or x2 >= W-3 or y1 < 3 or y2 >= H-3:
        return None
    
    bclr = (int(clr[0]), int(clr[1]), int(clr[2]))
    hclr = (int(min(230, clr[0]+random.randint(-15,20))), int(min(230, clr[1]+random.randint(-15,20))), int(min(230, clr[2]+random.randint(-15,20))))
    
    cv2.rectangle(bg, (cx-bw, cy-bh//2), (cx+bw, cy+bh//2), bclr, -1)
    cv2.rectangle(bg, (cx-lw, cy+bh//2), (cx, cy+bh), bclr, -1)
    cv2.rectangle(bg, (cx, cy+bh//2), (cx+lw, cy+bh), bclr, -1)
    cv2.circle(bg, (cx, cy-bh//2-hr), hr, hclr, -1)
    
    if aug and random.random() < 0.15:
        ox, oy = random.randint(cx-bw, cx+bw-10), random.randint(cy-bh//2, cy+bh-10)
        cv2.rectangle(bg, (ox, oy), (ox+random.randint(5,25), oy+random.randint(5,40)), np.random.randint(40,200,3).tolist(), -1)
    
    ncx, ncy = (x1+x2)/2/W, (y1+y2)/2/H
    nw, nh = (x2-x1)/W, (y2-y1)/H
    if nw < 0.005 or nh < 0.005: return None
    return f"0 {ncx:.6f} {ncy:.6f} {nw:.6f} {nh:.6f}"

def gen(idx, split, seed_off=0):
    b = make_bg(); la, u = [], []
    for _ in range(random.randint(1, 10)):
        for __ in range(30):
            x = random.randint(40, W-40); y = random.randint(80, H-60)
            sc = random.uniform(0.35, 2.2)
            clr = (random.randint(40,210), random.randint(40,210), random.randint(40,210))
            if all(((x-px)**2+(y-py)**2)**0.5 > (sc+ps)*35 for px, py, ps in u):
                break
        lab = person(b, x, y, sc, clr, aug=(split=="train"))
        if lab: la.append(lab); u.append((x, y, sc))
    cv2.imwrite(str(OUT/split/f"images/s_{idx:06d}.jpg"), b, [cv2.IMWRITE_JPEG_QUALITY, 90])
    with open(OUT/split/f"labels/s_{idx:06d}.txt", "w") as f:
        f.write('\n'.join(la) + ('\n' if la else ''))
    return len(la)

print("🎨 生成10000张数据...")
t0 = time.time(); tb = 0
for i in range(N_TRAIN): tb += gen(i, "train", 0)
for i in range(N_VAL): tb += gen(i, "val", 99999)
print(f"✅ {N_TRAIN+N_VAL}张/{tb}框/{time.time()-t0:.0f}s")

with open(OUT/"dataset.yaml", "w") as f:
    f.write(f"path: {OUT}\ntrain: train/images\nval: val/images\nnames:\n  0: person\nnc: 1\n")

print("\n🛡️ 防过拟合训练")
model = YOLO("yolov8n.pt")
model.train(
    data=str(OUT/"dataset.yaml"),
    epochs=20, imgsz=640, batch=32, name="person_antioverfit",
    project="D:/视觉安防系统/training_data/logs", exist_ok=True, device=0,
    workers=0, patience=5, single_cls=True, save=True,
    dropout=0.15, weight_decay=0.001, label_smoothing=0.1,
    augment=True, hsv_h=0.04, hsv_s=0.9, hsv_v=0.6,
    degrees=30, translate=0.25, scale=0.8, shear=8,
    flipud=0.3, fliplr=0.5, mosaic=1.0, mixup=0.4, copy_paste=0.3,
    erasing=0.5, perspective=0.002,
    lr0=0.0005, lrf=0.00001, optimizer="AdamW",
    warmup_epochs=2, close_mosaic=5, cos_lr=True,
    amp=True, seed=123456, verbose=True,
)

pt = "D:/视觉安防系统/training_data/logs/person_antioverfit/weights/best.pt"
if os.path.exists(pt):
    m = YOLO(pt)
    m.export(format="onnx", imgsz=640, half=True, simplify=True, opset=12, dynamic=False)
    shutil.copy2(pt.replace(".pt", ".onnx"), "D:/视觉安防系统/models/yolov8n.onnx")
    print("\n✅ 已部署!")
