"""
V3 微调：摄像头真实数据 + 侧脸/半身/遮挡合成数据
目标：提升侧脸、半身、部分出镜的检出率
"""
import cv2, numpy as np, random, time, os, shutil, sys
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, 'D:/视觉安防系统')
from detector import MultiStageDetector
import yaml

# ============================================================
# 1. 从摄像头录像提取真实标注数据
# ============================================================
print("=" * 60)
print("📹 阶段1: 提取摄像头真实标注数据")
print("=" * 60)

with open('D:/视觉安防系统/config.yaml') as f:
    cfg = yaml.safe_load(f)

detector = MultiStageDetector(
    person_model_path='D:/视觉安防系统/models/yolov8n.onnx',
    smoking_model_path='D:/视觉安防系统/models/smoking.onnx',
    pose_model_path='D:/视觉安防系统/models/yolov8n-pose.onnx',
    conf_threshold=0.30,  # 降低阈值捕获更多样本
    iou_threshold=0.45,
)

REAL_OUT = Path("D:/视觉安防系统/training_data/datasets/real_head")
for s in ["images", "labels"]:
    (REAL_OUT / s).mkdir(parents=True, exist_ok=True)

video_paths = [
    "D:/verify_out/v2_live.mp4",
    "D:/verify_out/v2_track2.mp4",
]

real_count = 0
for vp in video_paths:
    if not os.path.exists(vp):
        print(f"  跳过: {vp} (不存在)")
        continue
    cap = cv2.VideoCapture(vp)
    fi = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # 每5帧取一帧，避免重复
        if fi % 5 == 0:
            h, w = frame.shape[:2]
            result = detector.detect(frame)
            heads = result.get('heads', [])
            if heads:
                labels = []
                for hh in heads:
                    bx1, by1, bx2, by2 = hh['bbox']
                    # 转为YOLO归一化格式
                    cx = ((bx1 + bx2) / 2) / w
                    cy = ((by1 + by2) / 2) / h
                    bw = (bx2 - bx1) / w
                    bh = (by2 - by1) / h
                    if 0 < bw < 1 and 0 < bh < 1:
                        labels.append(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
                if labels:
                    real_count += 1
                    cv2.imwrite(
                        str(REAL_OUT / "images" / f"real_{real_count:06d}.jpg"),
                        frame, [cv2.IMWRITE_JPEG_QUALITY, 90]
                    )
                    with open(REAL_OUT / "labels" / f"real_{real_count:06d}.txt", "w") as lf:
                        lf.write("\n".join(labels) + "\n")
        fi += 1
    cap.release()
    print(f"  {vp}: 提取完成")

print(f"✅ 真实标注数据: {real_count} 帧")
print(f"   路径: {REAL_OUT}")

# ============================================================
# 2. 生成增强合成数据（侧脸/半身/遮挡）
# ============================================================
print(f"\n{'='*60}")
print("🎨 阶段2: 生成增强合成数据（侧脸/半身/遮挡）")
print("=" * 60)

SYNTH_OUT = Path("D:/视觉安防系统/training_data/datasets/synthetic_v3")
for s in ["train/images", "train/labels", "val/images", "val/labels"]:
    (SYNTH_OUT / s).mkdir(parents=True, exist_ok=True)

np.random.seed(123)
random.seed(123)
W, H = 640, 640
N_TRAIN = 8000
N_VAL = 1600

def random_bg():
    """丰富背景类型"""
    t = random.randint(0, 7)
    if t == 0:
        return np.random.randint(30, 220, (H, W, 3), dtype=np.uint8)
    elif t == 1:
        b = np.zeros((H, W, 3), dtype=np.uint8)
        for y in range(H):
            v = int(80 + 60 * np.sin(y * 0.02) + 40 * np.sin(y * 0.005))
            b[y, :] = np.clip(v, 20, 240)
        return b
    elif t == 2:
        b = np.random.randint(40, 200, (H, W, 3), dtype=np.uint8)
        return cv2.GaussianBlur(b, (9, 9), 4)
    elif t == 3:
        b = np.random.randint(50, 180, (H, W, 3), dtype=np.uint8)
        for i in range(0, W, 30):
            b[:, i:i + 8] = np.clip(b[:, i:i + 8].astype(int) + 30, 0, 255).astype(np.uint8)
        return b
    elif t == 4:
        b = np.zeros((H, W, 3), dtype=np.uint8)
        c1 = np.random.randint(80, 180, 3)
        c2 = np.random.randint(60, 200, 3)
        for y in range(0, H, 50):
            for x in range(0, W, 50):
                c = c1 if (x // 50 + y // 50) % 2 == 0 else c2
                b[y:y + 50, x:x + 50] = c
        return b
    elif t == 5:
        # 室外天空+地面
        b = np.zeros((H, W, 3), dtype=np.uint8)
        horizon = random.randint(250, 400)
        b[:horizon, :] = np.random.randint(100, 200, 3)
        b[horizon:, :] = np.random.randint(40, 140, 3)
        return b
    elif t == 6:
        # 走廊
        b = np.full((H, W, 3), np.random.randint(60, 160), dtype=np.uint8)
        # 消失点
        vp = (W // 2, H // 3)
        for i in range(10):
            angle = random.uniform(-np.pi / 4, np.pi / 4)
            x2 = int(vp[0] + 800 * np.cos(angle))
            y2 = int(vp[1] + 800 * np.sin(angle))
            cv2.line(b, vp, (x2, y2), (200, 200, 200), 1)
        return b
    else:
        return np.full((H, W, 3), np.random.randint(50, 220, 3), dtype=np.uint8)


def draw_person_v3(bg, cx, cy, scale, color, pose_type="front"):
    """绘制人物：支持正面/侧脸/半身/遮挡"""
    bw = max(15, int(38 * scale))
    bh = max(20, int(95 * scale))
    hr = max(8, int(14 * scale))
    lw = max(5, bw // 3)

    x1_body = cx - bw
    x2_body = cx + bw
    y1_head = cy - bh // 2 - hr * 2
    y2_body = cy + bh

    if x1_body < 2 or x2_body >= W - 2 or y1_head < 2 or y2_body >= H - 2:
        return None

    # 随机颜色微变
    dc = np.random.randint(-20, 20, 3)
    body_c = tuple(int(x) for x in np.clip(np.array(color) + dc, 30, 230))
    dc2 = np.random.randint(-15, 25, 3)
    head_c = tuple(int(x) for x in np.clip(np.array(color) + dc2, 30, 230))

    if pose_type == "front":
        # 完整正面人体
        cv2.rectangle(bg, (cx - bw, cy - bh // 2), (cx + bw, cy + bh // 2), body_c, -1)
        cv2.rectangle(bg, (cx - lw, cy + bh // 2), (cx, cy + bh), body_c, -1)
        cv2.rectangle(bg, (cx, cy + bh // 2), (cx + lw, cy + bh), body_c, -1)
        cv2.circle(bg, (cx, cy - bh // 2 - hr), hr, head_c, -1)
        # 眼睛（正面）
        eye_y = cy - bh // 2 - hr
        cv2.circle(bg, (cx - hr // 3, eye_y - 1), 2, (255, 255, 255), -1)
        cv2.circle(bg, (cx + hr // 3, eye_y - 1), 2, (255, 255, 255), -1)

        # 完整人体 bbox
        x1, y1 = x1_body, y1_head
        x2, y2 = x2_body, y2_body

    elif pose_type == "side":
        # 侧脸：头部偏移，身体窄化
        side_shift = int(bw * 0.5)
        # 侧脸身体
        cv2.rectangle(bg, (cx - bw + side_shift, cy - bh // 2),
                      (cx + bw - side_shift, cy + bh // 2), body_c, -1)
        # 侧脸头部（偏移到一侧）
        head_cx = cx + int(bw * 0.3)
        cv2.circle(bg, (head_cx, cy - bh // 2 - hr), hr, head_c, -1)
        # 侧脸轮廓线
        cv2.ellipse(bg, (head_cx, cy - bh // 2 - hr), (hr + 2, hr + 3), 0,
                    0, 180, (40, 40, 40), 1)

        x1 = min(x1_body, head_cx - hr)
        y1 = y1_head
        x2 = max(x2_body, head_cx + hr)
        y2 = y2_body

    elif pose_type == "half":
        # 半身（仅上半身+头部，模拟下半身被遮挡）
        half_h = bh // 2 + hr * 2
        cv2.rectangle(bg, (cx - bw, cy - bh // 2), (cx + bw, cy), body_c, -1)
        cv2.circle(bg, (cx, cy - bh // 2 - hr), hr, head_c, -1)
        eye_y = cy - bh // 2 - hr
        cv2.circle(bg, (cx - hr // 3, eye_y - 1), 2, (255, 255, 255), -1)
        cv2.circle(bg, (cx + hr // 3, eye_y - 1), 2, (255, 255, 255), -1)

        x1, y1 = x1_body, y1_head
        x2, y2 = x2_body, cy + 1  # 半身下边界

    elif pose_type == "edge":
        # 边缘部分出镜：人只有一半在画面内
        # 实际画在边界外也画，但标签只标可见部分
        cv2.rectangle(bg, (cx - bw, cy - bh // 2), (cx + bw, cy + bh // 2), body_c, -1)
        cv2.circle(bg, (cx, cy - bh // 2 - hr), hr, head_c, -1)

        # bbox裁剪到画面内
        x1 = max(0, x1_body)
        y1 = max(0, y1_head)
        x2 = min(W - 1, x2_body)
        y2 = min(H - 1, y2_body)

        if x2 <= x1 + 15 or y2 <= y1 + 15:
            return None

    elif pose_type == "back":
        # 背对：纯色圆头，无眼睛
        cv2.rectangle(bg, (cx - bw, cy - bh // 2), (cx + bw, cy + bh // 2), body_c, -1)
        cv2.rectangle(bg, (cx - lw, cy + bh // 2), (cx, cy + bh), body_c, -1)
        cv2.rectangle(bg, (cx, cy + bh // 2), (cx + lw, cy + bh), body_c, -1)
        # 背对头部（较深色、无眼睛）
        back_head = tuple(int(x) for x in np.clip(np.array(head_c) - 20, 20, 220))
        cv2.circle(bg, (cx, cy - bh // 2 - hr), hr, back_head, -1)

        x1, y1 = x1_body, y1_head
        x2, y2 = x2_body, y2_body

    else:
        # small - 远处小人
        small_s = scale * 0.5
        small_bw = max(8, int(18 * scale))
        small_bh = max(10, int(45 * scale))
        small_hr = max(5, int(7 * scale))
        cv2.rectangle(bg, (cx - small_bw, cy - small_bh // 2),
                      (cx + small_bw, cy + small_bh // 2), body_c, -1)
        cv2.circle(bg, (cx, cy - small_bh // 2 - small_hr), small_hr, head_c, -1)
        x1, y1 = cx - small_bw, cy - small_bh // 2 - small_hr * 2
        x2, y2 = cx + small_bw, cy + small_bh

    # 归一化标签
    ncx = ((x1 + x2) / 2) / W
    ncy = ((y1 + y2) / 2) / H
    nw = (x2 - x1) / W
    nh = (y2 - y1) / H

    if nw < 0.005 or nh < 0.005 or ncx < 0 or ncx > 1 or ncy < 0 or ncy > 1:
        return None

    return f"0 {ncx:.6f} {ncy:.6f} {nw:.6f} {nh:.6f}"


def generate_v3(idx, split):
    bg = random_bg()
    labels = []
    used = []
    n_people = random.randint(1, 10)

    for _ in range(n_people):
        for attempt in range(30):
            cx = random.randint(40, W - 40)
            cy = random.randint(80, H - 60)
            scale = random.uniform(0.35, 2.0)

            # 姿势分布：偏向侧脸和半身（针对性训练）
            r = random.random()
            if r < 0.25:
                pose = "front"
            elif r < 0.50:
                pose = "side"  # 25% 侧脸
            elif r < 0.70:
                pose = "half"  # 20% 半身
            elif r < 0.85:
                pose = "edge"  # 15% 边缘出镜
            elif r < 0.95:
                pose = "back"  # 10% 背对
            else:
                pose = "small"  # 5% 远处小人

            color = (random.randint(50, 200), random.randint(50, 200), random.randint(50, 200))

            ok = True
            for px, py, ps in used:
                if abs(cx - px) < (scale + ps) * 42 and abs(cy - py) < (scale + ps) * 120:
                    ok = False
                    break
            if ok:
                break

        lab = draw_person_v3(bg, cx, cy, scale, color, pose)
        if lab:
            labels.append(lab)
            used.append((cx, cy, scale))

    # 15%概率添加随机遮挡块
    if random.random() < 0.15:
        n_blocks = random.randint(1, 4)
        for _ in range(n_blocks):
            bx = random.randint(0, W - 50)
            by = random.randint(0, H - 50)
            bw_b = random.randint(30, 120)
            bh_b = random.randint(30, 120)
            block_c = tuple(int(x) for x in np.random.randint(40, 200, 3))
            cv2.rectangle(bg, (bx, by), (bx + bw_b, by + bh_b), block_c, -1)

    img_path = SYNTH_OUT / split / "images" / f"synth_{idx:06d}.jpg"
    lbl_path = SYNTH_OUT / split / "labels" / f"synth_{idx:06d}.txt"
    cv2.imwrite(str(img_path), bg, [cv2.IMWRITE_JPEG_QUALITY, 85])
    with open(lbl_path, "w") as f:
        f.write("\n".join(labels) + ("\n" if labels else ""))
    return len(labels)


t0 = time.time()
total_boxes = 0

for i in range(N_TRAIN):
    total_boxes += generate_v3(i, "train")
    if i % 2000 == 0:
        e = time.time() - t0
        print(f"  训练: {i}/{N_TRAIN} ({total_boxes}框, {e:.0f}s)")

for i in range(N_VAL):
    total_boxes += generate_v3(i, "val")

e = time.time() - t0
print(f"✅ 合成数据: {N_TRAIN + N_VAL}张, {total_boxes}框, {e:.0f}秒")

# ============================================================
# 3. 合并数据集（合成数据 + 真实数据）
# ============================================================
print(f"\n{'='*60}")
print("🔗 阶段3: 合并数据集")
print("=" * 60)

# 把真实数据混入训练集（全部用于训练，因为量少）
real_imgs = list((REAL_OUT / "images").glob("*.jpg"))
print(f"  真实数据: {len(real_imgs)}帧")

# 复制到合成训练目录
imported = 0
for rimg in real_imgs:
    rlbl = REAL_OUT / "labels" / f"{rimg.stem}.txt"
    if not rlbl.exists():
        continue
    # 重命名避免冲突
    new_name = f"real_{imported:06d}"
    shutil.copy2(str(rimg), str(SYNTH_OUT / "train" / "images" / f"{new_name}.jpg"))
    shutil.copy2(str(rlbl), str(SYNTH_OUT / "train" / "labels" / f"{new_name}.txt"))
    imported += 1

print(f"✅ 已合并: {imported}张真实数据 → 训练集")
print(f"  总训练集: {N_TRAIN + imported}张")

# 写数据集配置
ds_yaml = SYNTH_OUT / "dataset.yaml"
with open(ds_yaml, "w") as f:
    f.write(f"path: {SYNTH_OUT}\n")
    f.write("train: train/images\n")
    f.write("val: val/images\n")
    f.write("names:\n  0: person\nnc: 1\n")

print(f"\n  配置: {ds_yaml}")

# ============================================================
# 4. V3 训练（VRAM 优化的 batch + 强增强 + 高 dropout）
# ============================================================
print(f"\n{'='*60}")
print("🚀 阶段4: V3 微调训练（侧脸/半身/遮挡 专项）")
print("=" * 60)

# 从 V2 最佳模型继续
v2_pt = "D:/视觉安防系统/training_data/logs/person_v2/weights/best.pt"
model_path = v2_pt if os.path.exists(v2_pt) else "yolov8n.pt"
print(f"  起点: {model_path}")

from ultralytics import YOLO

model = YOLO(model_path)
results = model.train(
    data=str(ds_yaml),
    epochs=30,
    imgsz=640,
    batch=24,
    name="person_v3",
    project="D:/视觉安防系统/training_data/logs",
    exist_ok=True,
    device=0,
    workers=0,
    patience=8,
    save=True,
    single_cls=True,
    # ⚡ 强数据增强 — 侧脸/遮挡针对性
    augment=True,
    hsv_h=0.05, hsv_s=0.9, hsv_v=0.6,
    degrees=35.0, translate=0.25, scale=0.8,
    shear=8.0, perspective=0.0015,
    flipud=0.25, fliplr=0.5,
    mosaic=1.0, mixup=0.6, copy_paste=0.3,
    erasing=0.7,
    # ⚡ 防过拟合
    dropout=0.25,
    label_smoothing=0.12,
    optimizer="AdamW", lr0=0.0003, lrf=0.00003,
    momentum=0.937, weight_decay=0.002,
    warmup_epochs=5,
    close_mosaic=8,
    amp=True, seed=42,
    verbose=True,
)

# ============================================================
# 5. 导出ONNX并部署
# ============================================================
print(f"\n{'='*60}")
print("📦 阶段5: 导出ONNX并部署")
print("=" * 60)

pt_path = "D:/视觉安防系统/training_data/logs/person_v3/weights/best.pt"
if os.path.exists(pt_path):
    export_model = YOLO(pt_path)
    export_model.export(format="onnx", imgsz=640, half=False, simplify=True, opset=12, dynamic=False)

    src = pt_path.replace(".pt", ".onnx")
    dst = "D:/视觉安防系统/models/yolov8n.onnx"
    if os.path.exists(src):
        shutil.copy2(src, dst)
        sz = os.path.getsize(dst) / 1024
        print(f"✅ V3模型已部署: {dst} ({sz:.0f}KB)")
    else:
        print(f"⚠️ ONNX未找到，尝试搜索...")
        import glob
        for f in glob.glob("D:/视觉安防系统/training_data/logs/person_v3/weights/*.onnx"):
            shutil.copy2(f, dst)
            print(f"  找到: {f} -> {dst}")

    # 读取训练指标
    results_csv = "D:/视觉安防系统/training_data/logs/person_v3/results.csv"
    if os.path.exists(results_csv):
        with open(results_csv) as f:
            lines = f.readlines()
            last = lines[-1].strip().split(",")
            print(f"\n  mAP50={float(last[6]):.3f}  mAP50-95={float(last[7]):.3f}")
else:
    print("⚠️ 训练未生成best.pt，检查日志")

print("\n✅ V3训练流水线完成！")
