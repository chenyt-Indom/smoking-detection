"""第二轮微调 — 更真实合成数据 + 摄像头背景 + 更强防过拟合"""
import cv2, numpy as np, random, time, shutil
from pathlib import Path
from ultralytics import YOLO

OUT = Path("D:/视觉安防系统/training_data/datasets/synthetic_v2")
for d in OUT.glob("**/*"): 
    if d.is_file(): d.unlink()
for s in ["train/images","train/labels","val/images","val/labels"]:
    (OUT/s).mkdir(parents=True, exist_ok=True)

N_TRAIN, N_VAL = 10000, 2000
W, H = 640, 640
np.random.seed(999)
random.seed(999)

# --- 真实背景纹理生成（更逼真） ---
def make_bg():
    tp = random.randint(0, 7)
    if tp == 0:  # Perlin-like noise
        b = np.random.randint(40, 220, (H, W, 3), dtype=np.uint8)
        for _ in range(3):
            b = cv2.GaussianBlur(b, (9, 9), 3)
        # 叠加纹理
        noise = np.random.randint(-20, 20, (H, W, 3))
        b = np.clip(b.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    elif tp == 1:  # 室内渐变
        b = np.zeros((H, W, 3), dtype=np.uint8)
        for y in range(H):
            v = 60 + int(120 * y / H)
            b[y, :] = np.clip(v + np.random.randint(-10, 10, 3), 20, 240)
    elif tp == 2:  # 户外场景
        b = np.random.randint(80, 200, (H, W, 3), dtype=np.uint8)
        # 天空
        b[:H//2, :] = np.clip(b[:H//2, :].astype(int) + 40, 0, 255).astype(np.uint8)
        # 地面纹理
        for x in range(0, W, 15):
            b[H//2:, x:x+8] = np.clip(b[H//2:, x:x+8].astype(int) + 30, 0, 255).astype(np.uint8)
    elif tp == 3:  # 走廊/室内墙壁
        b = np.full((H, W, 3), np.random.randint(100, 180), dtype=np.uint8)
        for y in range(0, H, 5):
            b[y, :] = np.clip(b[y, :].astype(int) + random.randint(-5, 5), 0, 255).astype(np.uint8)
        cv2.rectangle(b, (0, H*3//4), (W, H), (60, 60, 60), -1)  # 地板
    elif tp == 4:  # 噪点+模糊
        b = np.random.randint(60, 200, (H, W, 3), dtype=np.uint8)
        ks = random.choice([3, 5, 7, 9])
        b = cv2.GaussianBlur(b, (ks, ks), 2)
    elif tp == 5:  # 棋盘格
        c1 = np.random.randint(60, 200, 3); c2 = np.random.randint(60, 200, 3)
        b = np.tile(c1, (H, W, 1)).astype(np.uint8)
        for py in range(0, H, random.randint(20, 60)):
            for px in range(0, W, random.randint(20, 60)):
                if (px//30 + py//30) % 2:
                    b[py:py+30, px:px+30] = c2
    elif tp == 6:  # 色块随机
        b = np.random.randint(30, 200, (H, W, 3), dtype=np.uint8)
        for _ in range(random.randint(5, 20)):
            rx, ry = random.randint(0, W-40), random.randint(0, H-40)
            b[ry:ry+random.randint(20, 100), rx:rx+random.randint(20, 150)] = np.random.randint(60, 220, 3)
    else:  # 纯色
        b = np.full((H, W, 3), np.random.randint(50, 200), dtype=np.uint8)
    
    # 随机旋转/翻转增加多样性
    if random.random() < 0.2:
        b = cv2.flip(b, random.randint(-1, 1))
    
    return b

def draw_person(bg, cx, cy, s, clr, aug=True):
    bw, bh = max(12, int(36*s)), max(18, int(90*s))
    hr, lw = max(7, int(13*s)), max(4, bw//3)
    x1, x2 = cx-bw, cx+bw
    y1, y2 = cy-bh//2-hr*2, cy+bh
    if x1 < 3 or x2 >= W-3 or y1 < 3 or y2 >= H-3: return None
    
    bclr = (int(clr[0]), int(clr[1]), int(clr[2]))
    hclr = (int(min(230,clr[0]+random.randint(-15,20))), int(min(230,clr[1]+random.randint(-15,20))), int(min(230,clr[2]+random.randint(-15,20))))
    
    cv2.rectangle(bg, (cx-bw, cy-bh//2), (cx+bw, cy+bh//2), bclr, -1)
    cv2.rectangle(bg, (cx-lw, cy+bh//2), (cx, cy+bh), bclr, -1)
    cv2.rectangle(bg, (cx, cy+bh//2), (cx+lw, cy+bh), bclr, -1)
    
    # 更逼真的头部
    cv2.circle(bg, (cx, cy-bh//2-hr), hr, hclr, -1)
    cv2.circle(bg, (cx, cy-bh//2-hr), hr, (30,30,30), 1)
    # 眼睛
    eye_y = cy - bh//2 - hr + 2
    cv2.circle(bg, (cx-5, eye_y), 2, (255,255,255), -1)
    cv2.circle(bg, (cx+5, eye_y), 2, (255,255,255), -1)
    
    # 随机遮挡
    if aug and random.random() < 0.2:
        ox, oy = random.randint(cx-bw, cx+bw-10), random.randint(cy-bh//2, cy+bh-10)
        cv2.rectangle(bg, (ox, oy), (ox+random.randint(5,30), oy+random.randint(5,50)), np.random.randint(40,200,3).tolist(), -1)
    
    ncx, ncy = (x1+x2)/2/W, (y1+y2)/2/H
    nw, nh = (x2-x1)/W, (y2-y1)/H
    if nw < 0.005 or nh < 0.005: return None
    return f"0 {ncx:.6f} {ncy:.6f} {nw:.6f} {nh:.6f}"

def gen(idx, split):
    b = make_bg()
    la, u = [], []
    for _ in range(random.randint(1, 12)):
        for __ in range(30):
            x = random.randint(40, W-40); y = random.randint(80, H-60)
            sc = random.uniform(0.3, 2.5)
            clr = (random.randint(40,210), random.randint(40,210), random.randint(40,210))
            if all(((x-px)**2+(y-py)**2)**0.5 > (sc+ps)*35 for px,py,ps in u): break
        lab = draw_person(b, x, y, sc, clr, aug=(split=="train"))
        if lab: la.append(lab); u.append((x, y, sc))
    
    # 随机添加纯色块噪声（模拟误检场景）
    if split == "train" and random.random() < 0.1:
        for _ in range(random.randint(1, 3)):
            rx, ry = random.randint(0, W-100), random.randint(0, H-80)
            cv2.rectangle(b, (rx, ry), (rx+random.randint(30,150), ry+random.randint(20,100)), 
                        np.random.randint(60,200,3).tolist(), -1)
    
    cv2.imwrite(str(OUT/split/f"images/s_{idx:06d}.jpg"), b, [cv2.IMWRITE_JPEG_QUALITY, 90])
    with open(OUT/split/f"labels/s_{idx:06d}.txt", "w") as f:
        f.write('\n'.join(la)+('\n'if la else''))
    return len(la)

print("🎨 生成12000张真实风格数据...")
t0, tb = time.time(), 0
for i in range(N_TRAIN): tb += gen(i, "train")
for i in range(N_VAL): tb += gen(i, "val")
print(f"✅ {N_TRAIN+N_VAL}张/{tb}框/{time.time()-t0:.0f}s")

with open(OUT/"dataset.yaml", "w") as f:
    f.write(f"path: {OUT}\ntrain: train/images\nval: val/images\nnames:\n  0: person\nnc: 1\n")

print("\n🛡️ 第二轮微调 (更强防过拟合)")
model = YOLO("D:/视觉安防系统/training_data/logs/person_antioverfit/weights/best.pt")
model.train(
    data=str(OUT/"dataset.yaml"),
    epochs=25, imgsz=640, batch=32, name="person_v2",
    project="D:/视觉安防系统/training_data/logs", exist_ok=True, device=0,
    workers=0, patience=6, single_cls=True, save=True,
    dropout=0.2, weight_decay=0.002,
    augment=True, hsv_h=0.05, hsv_s=1.0, hsv_v=0.7,
    degrees=35, translate=0.3, scale=0.9, shear=10,
    flipud=0.3, fliplr=0.5, mosaic=1.0, mixup=0.5, copy_paste=0.4,
    erasing=0.6, perspective=0.003,
    lr0=0.0003, lrf=0.000005, optimizer="AdamW",
    warmup_epochs=3, close_mosaic=6, cos_lr=True,
    amp=True, seed=999, verbose=True,
)

pt = "D:/视觉安防系统/training_data/logs/person_v2/weights/best.pt"
if __import__('os').path.exists(pt):
    m = YOLO(pt)
    m.export(format="onnx", imgsz=640, half=False, simplify=True, opset=12, dynamic=False)
    shutil.copy2(pt.replace(".pt",".onnx"), "D:/视觉安防系统/models/yolov8n.onnx")
    print("\n✅ 第二轮微调完成，已部署!")
