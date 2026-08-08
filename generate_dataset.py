"""
generate_dataset.py - 离线生成香烟+烟盒目标检测数据集
2类: cigarette(0), cigarette_pack(1)
包含逼真香烟、烟盒、多种背景场景
"""
import os, cv2, numpy as np, random, math
from pathlib import Path


def create_output_dirs():
    for d in ["datasets/smoking/images/train", "datasets/smoking/images/val",
              "datasets/smoking/labels/train", "datasets/smoking/labels/val"]:
        os.makedirs(d, exist_ok=True)


def make_gradient_h(w, h, c1, c2):
    g = np.zeros((h, w, 3), dtype=np.float32)
    for i in range(w):
        t = i / max(w - 1, 1)
        g[:, i] = np.array(c1) * (1 - t) + np.array(c2) * t
    return g.astype(np.uint8)


def make_gradient_v(w, h, c1, c2):
    g = np.zeros((h, w, 3), dtype=np.float32)
    for i in range(h):
        t = i / max(h - 1, 1)
        g[i, :] = np.array(c1) * (1 - t) + np.array(c2) * t
    return g.astype(np.uint8)


def overlay_pixels(img, obj, mask, ox, oy):
    """把 obj 通过 mask 覆盖到 img 上"""
    ih, iw = img.shape[:2]
    oh, ow = obj.shape[:2]
    for py in range(oh):
        for px in range(ow):
            if mask[py, px] > 0:
                ty, tx = oy + py, ox + px
                if 0 <= ty < ih and 0 <= tx < iw:
                    img[ty, tx] = obj[py, px]


def get_object_bbox(obj, obj_w, obj_h):
    """从绘制对象中提取边界框"""
    gray = cv2.cvtColor(obj, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 5, 255, cv2.THRESH_BINARY)
    coords = cv2.findNonZero(mask)
    if coords is None:
        return None
    bx, by, bw, bh = cv2.boundingRect(coords)
    return obj[by:by + bh, bx:bx + bw], mask[by:by + bh, bx:bx + bw], bx, by, bw, bh


# ============================================================
#  香烟绘制
# ============================================================
def draw_cigarette(img, cx, cy, length, thickness, angle=0):
    """绘制逼真香烟，返回 bbox (x1,y1,x2,y2)"""
    S = 4
    L = int(length * S)
    T = int(thickness * S)
    cig = np.zeros((T * 3, L * 3, 3), dtype=np.uint8)

    filter_start = int(L * 0.72)
    ash_end = int(L * 0.05)

    # 烟身
    body_grad = make_gradient_v(L - filter_start, T, (250, 248, 240), (210, 205, 195))
    for i in range(0, L - filter_start, 6):
        n = random.randint(-5, 5)
        body_grad[:, max(0, i - 1):min(L - filter_start, i + 2)] = np.clip(
            body_grad[:, max(0, i - 1):min(L - filter_start, i + 2)].astype(np.int16) + n, 0, 255).astype(np.uint8)
    cig[T:T * 2, filter_start:L] = body_grad

    # 过滤嘴
    fc1, fc2 = random.choice([((180, 120, 60), (140, 80, 30)),
                               ((200, 150, 80), (160, 100, 40)),
                               ((220, 170, 100), (180, 130, 70))])
    fg = make_gradient_h(filter_start, T, fc1, fc2)
    for _ in range(random.randint(8, 20)):
        cv2.circle(fg, (random.randint(0, filter_start - 1), random.randint(2, T - 3)),
                   random.randint(1, 3), (random.randint(100, 160), random.randint(60, 100), random.randint(20, 60)), -1)
    cig[T:T * 2, 0:filter_start] = fg

    # 燃烧头
    cig[T:T * 2, L - ash_end:L] = (random.randint(60, 100), random.randint(55, 95), random.randint(50, 90))
    glow = (random.randint(30, 80), random.randint(30, 60), random.randint(180, 255))
    for gy in range(T):
        a = 1.0 - abs(gy - T / 2) / (T / 2)
        for gx in range(ash_end + 2):
            aa = a * (1.0 - gx / (ash_end + 2))
            cig[T + gy, L - ash_end + gx] = tuple(
                int(cig[T + gy, L - ash_end + gx, c] * (1 - aa) + glow[c] * aa) for c in range(3))

    # 金色环
    rx = filter_start
    cig[T:T * 2, rx:rx + max(2, T // 3)] = (random.randint(30, 80), random.randint(160, 220), random.randint(180, 240))

    if angle != 0:
        M = cv2.getRotationMatrix2D((L // 2, T * 3 // 2), angle, 1.0)
        cig = cv2.warpAffine(cig, M, (L * 3, T * 3), borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
    cig = cv2.resize(cig, (L * 3, T * 3), interpolation=cv2.INTER_AREA)

    r = get_object_bbox(cig, L * 3, T * 3)
    if r is None:
        return cx, cy, cx + length, cy + thickness
    crop, mask, bx, by, bw, bh = r
    ox, oy = cx - bw // 2, cy - bh // 2
    overlay_pixels(img, crop, mask, ox, oy)
    return ox, oy, ox + bw, oy + bh


def draw_smoke(img, x1, y1, x2, y2, angle=0):
    rad = math.radians(angle)
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    sx = int(cx - math.cos(rad) * (x2 - x1) * 0.45)
    sy = int(cy - math.sin(rad) * (y2 - y1) * 0.45)
    layer = np.zeros_like(img, dtype=np.float32)
    for i in range(random.randint(4, 12)):
        lx = sx + random.randint(-10, 10)
        ly = sy - random.randint(5, 30 + i * 8)
        r = random.randint(4, 12 + i * 2)
        a = max(0.1, 1.0 - i * 0.08)
        c = (random.randint(180, 230), random.randint(180, 230), random.randint(190, 240))
        cv2.circle(layer, (lx, ly), r, (c[0] * a, c[1] * a, c[2] * a), -1)
    layer = cv2.GaussianBlur(layer, (15, 15), 8)
    m = layer > 0
    img[m] = np.clip(img[m].astype(np.float32) * 0.6 + layer[m] * 0.4, 0, 255).astype(np.uint8)


# ============================================================
#  烟盒绘制
# ============================================================
def draw_cigarette_pack(img, cx, cy, width, height, angle=0):
    """绘制逼真烟盒，返回 bbox"""
    S = 4
    W = int(width * S)
    H = int(height * S)
    pack = np.zeros((H * 3, W * 3, 3), dtype=np.uint8)

    # 烟盒品牌颜色方案
    brands = [
        # (主色, 副色, 品牌名)
        ((30, 30, 180), (220, 220, 240), "中华"),       # 红色中华
        ((30, 30, 160), (230, 200, 150), "中华"),       # 深红中华
        ((200, 200, 220), (30, 30, 120), "玉溪"),       # 白色玉溪
        ((50, 50, 50), (180, 150, 100), "利群"),        # 深色利群
        ((40, 40, 140), (220, 200, 180), "芙蓉王"),      # 蓝芙蓉王
        ((30, 30, 30), (200, 180, 140), "黄鹤楼"),       # 黑黄鹤楼
        ((180, 180, 200), (30, 30, 100), "云烟"),       # 白蓝云烟
        ((20, 20, 100), (220, 200, 160), "红塔山"),      # 蓝红塔山
    ]
    primary, secondary, brand = random.choice(brands)

    # 主色填充
    pack[:] = primary

    # 3D 立体效果：顶部高亮，底部阴影
    for i in range(H):
        f = 0.75 + 0.25 * (1.0 - i / H)
        pack[i, :] = np.clip(pack[i, :] * f, 0, 255).astype(np.uint8)
    for i in range(H):
        f = 1.0 - 0.15 * (i / H)
        pack[i, :W // 8] = np.clip(pack[i, :W // 8] * f, 0, 255).astype(np.uint8)

    # 副色带（品牌标识区域）
    stripe_h = H // 5
    stripe_y = H // 4
    for i in range(H):
        for j in range(W):
            if stripe_y <= i < stripe_y + stripe_h and W // 6 <= j < W * 5 // 6:
                t = (i - stripe_y) / stripe_h
                if t < 0.1 or t > 0.9:
                    pack[i, j] = tuple(int(p * 0.3 + s * 0.7) for p, s in zip(primary, secondary))
                else:
                    pack[i, j] = secondary

    # 品牌文字（仿宋体方块）
    text_color = (255, 255, 255) if sum(primary) < 400 else (30, 30, 30)
    text_y = stripe_y + stripe_h // 2
    # 用矩形模拟汉字
    char_w = W // 6
    for ci, ch in enumerate(brand[:2]):
        tx = W // 3 + ci * char_w + random.randint(-2, 2)
        ty = text_y + random.randint(-3, 3)
        # 绘制方块模拟文字
        cv2.putText(pack, ch, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8 * S, text_color, max(2, S // 2), cv2.LINE_AA)

    # 边缘线
    cv2.rectangle(pack, (2, 2), (W - 2, H - 2), (0, 0, 0), max(1, S // 4))

    # 金色装饰线
    gold_y = H * 2 // 3
    cv2.line(pack, (W // 6, gold_y), (W * 5 // 6, gold_y),
             (50, 180, 220), max(1, S // 4))

    # 条码区域
    barcode_y = H * 3 // 4
    barcode_h = H // 8
    for i in range(barcode_y, barcode_y + barcode_h):
        for j in range(W // 4, W * 3 // 4, random.randint(2, 5)):
            if random.random() < 0.5:
                pack[i, j] = (255, 255, 255)

    if angle != 0:
        M = cv2.getRotationMatrix2D((W // 2, H * 3 // 2), angle, 1.0)
        pack = cv2.warpAffine(pack, M, (W * 3, H * 3),
                              borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
    pack = cv2.resize(pack, (W * 3, H * 3), interpolation=cv2.INTER_AREA)

    r = get_object_bbox(pack, W * 3, H * 3)
    if r is None:
        return cx, cy, cx + width, cy + height
    crop, mask, bx, by, bw, bh = r
    ox, oy = cx - bw // 2, cy - bh // 2
    overlay_pixels(img, crop, mask, ox, oy)
    return ox, oy, ox + bw, oy + bh


# ============================================================
#  背景生成
# ============================================================
def generate_background(size=416):
    t = random.choice(["indoor_wall", "outdoor", "dark", "textured", "gradient"])
    img = np.zeros((size, size, 3), dtype=np.uint8)

    if t == "indoor_wall":
        b = random.randint(180, 240)
        img[:] = (b + random.randint(-10, 10), b + random.randint(-15, 5), b + random.randint(-10, 10))
        for i in range(size):
            img[i, :] = np.clip(img[i, :] * (0.85 + 0.15 * i / size), 0, 255).astype(np.uint8)

    elif t == "outdoor":
        sc = (random.randint(180, 255), random.randint(180, 240), random.randint(100, 180))
        gc = (random.randint(60, 140), random.randint(80, 160), random.randint(40, 100))
        hz = random.randint(size // 3, size * 2 // 3)
        for i in range(size):
            if i < hz:
                rt = i / max(hz, 1)
                img[i, :] = tuple(int(s * (1 - rt) + g * rt) for s, g in zip(sc, gc))
            else:
                img[i, :] = gc

    elif t == "dark":
        b = random.randint(20, 60)
        img[:] = (b + random.randint(-5, 10), b + random.randint(-5, 10), b + random.randint(-5, 10))
        lx, ly = random.randint(0, size), random.randint(0, size)
        for i in range(size):
            for j in range(size):
                d = math.sqrt((i - ly) ** 2 + (j - lx) ** 2)
                f = max(0.3, 1.0 - d / (size * 1.2))
                img[i, j] = np.clip(img[i, j] * (0.5 + 0.5 * f), 0, 255).astype(np.uint8)

    elif t == "textured":
        b = random.randint(100, 200)
        img[:] = (b, b, b)
        n = np.random.randint(-20, 20, (size, size), dtype=np.int16)
        for c in range(3):
            img[:, :, c] = np.clip(img[:, :, c].astype(np.int16) + n, 0, 255).astype(np.uint8)
        img = cv2.GaussianBlur(img, (5, 5), 2)

    else:
        c1 = tuple(random.randint(30, 200) for _ in range(3))
        c2 = tuple(random.randint(30, 200) for _ in range(3))
        img = make_gradient_v(size, size, c1, c2) if random.random() < 0.5 else make_gradient_h(size, size, c1, c2)

    n = np.random.randint(-5, 5, (size, size, 3), dtype=np.int16)
    img = np.clip(img.astype(np.int16) + n, 0, 255).astype(np.uint8)
    return img


# ============================================================
#  样本生成
# ============================================================
def bbox_to_yolo(x1, y1, x2, y2, img_size):
    x1 = max(0, min(x1, img_size - 1))
    y1 = max(0, min(y1, img_size - 1))
    x2 = max(1, min(x2, img_size))
    y2 = max(1, min(y2, img_size))
    cx = ((x1 + x2) / 2) / img_size
    cy = ((y1 + y2) / 2) / img_size
    w = (x2 - x1) / img_size
    h = (y2 - y1) / img_size
    return cx, cy, w, h


def generate_cigarette_sample(img_size=416):
    """生成香烟样本 (class_id=0)"""
    img = generate_background(img_size)
    length = random.randint(35, 100)
    thickness = random.randint(7, 16)
    angle = random.randint(-60, 60)
    cx = random.randint(img_size // 4, img_size * 3 // 4)
    cy = random.randint(img_size // 4, img_size * 3 // 4)
    x1, y1, x2, y2 = draw_cigarette(img, cx, cy, length, thickness, angle)
    draw_smoke(img, x1, y1, x2, y2, angle)
    cx_n, cy_n, w_n, h_n = bbox_to_yolo(x1, y1, x2, y2, img_size)
    return img, [(0, cx_n, cy_n, w_n, h_n)]


def generate_pack_sample(img_size=416):
    """生成烟盒样本 (class_id=1)"""
    img = generate_background(img_size)
    pw = random.randint(40, 90)
    ph = random.randint(55, 110)
    angle = random.randint(-45, 45)
    cx = random.randint(img_size // 4, img_size * 3 // 4)
    cy = random.randint(img_size // 4, img_size * 3 // 4)
    x1, y1, x2, y2 = draw_cigarette_pack(img, cx, cy, pw, ph, angle)
    cx_n, cy_n, w_n, h_n = bbox_to_yolo(x1, y1, x2, y2, img_size)
    return img, [(1, cx_n, cy_n, w_n, h_n)]


def generate_mixed_sample(img_size=416):
    """生成同时包含香烟和烟盒的样本"""
    img = generate_background(img_size)
    labels = []

    # 香烟在左侧
    cx1 = random.randint(img_size // 6, img_size // 3)
    cy1 = random.randint(img_size // 3, img_size * 2 // 3)
    x1, y1, x2, y2 = draw_cigarette(img, cx1, cy1,
                                     random.randint(35, 80), random.randint(7, 14),
                                     random.randint(-40, 40))
    draw_smoke(img, x1, y1, x2, y2, 0)
    labels.append((0,) + bbox_to_yolo(x1, y1, x2, y2, img_size))

    # 烟盒在右侧
    cx2 = random.randint(img_size * 2 // 3, img_size * 5 // 6)
    cy2 = random.randint(img_size // 3, img_size * 2 // 3)
    bx1, by1, bx2, by2 = draw_cigarette_pack(img, cx2, cy2,
                                              random.randint(40, 75), random.randint(55, 95),
                                              random.randint(-30, 30))
    labels.append((1,) + bbox_to_yolo(bx1, by1, bx2, by2, img_size))

    return img, labels


def generate_negative_sample(img_size=416):
    img = generate_background(img_size)
    for _ in range(random.randint(1, 5)):
        t = random.choice(["rect", "circle", "line", "sq"])
        c = tuple(random.randint(20, 220) for _ in range(3))
        if t == "rect":
            x, y = random.randint(0, img_size - 60), random.randint(0, img_size - 40)
            cv2.rectangle(img, (x, y), (x + random.randint(20, 120), y + random.randint(15, 80)), c, -1)
        elif t == "circle":
            cv2.circle(img, (random.randint(30, img_size - 30), random.randint(30, img_size - 30)),
                       random.randint(10, 50), c, -1)
        elif t == "line":
            x = random.randint(0, img_size)
            cv2.line(img, (x, random.randint(0, img_size)),
                     (x + random.randint(-150, 150), random.randint(0, img_size)), c, random.randint(1, 4))
        else:
            x, y = random.randint(0, img_size - 30), random.randint(0, img_size - 30)
            s = random.randint(10, 40)
            cv2.rectangle(img, (x, y), (x + s, y + s), c, -1)
    return img


# ============================================================
#  主函数
# ============================================================
def generate_dataset(num_train=3000, num_val=600):
    print("=" * 60)
    print("  香烟+烟盒目标检测数据集生成器")
    print("  类别: 0=cigarette(香烟) 1=cigarette_pack(烟盒)")
    print("=" * 60)
    create_output_dirs()

    def gen_split(count, prefix, split):
        print(f"\n生成{split} ({count} 张)...")
        for i in range(count):
            r = random.random()
            if r < 0.25:
                img, labels = generate_cigarette_sample()
            elif r < 0.45:
                img, labels = generate_pack_sample()
            elif r < 0.60:
                img, labels = generate_mixed_sample()
            else:
                img = generate_negative_sample()
                labels = []

            name = f"{prefix}_{i:05d}"
            cv2.imwrite(f"datasets/smoking/images/{split}/{name}.jpg", img,
                        [cv2.IMWRITE_JPEG_QUALITY, 92])
            if labels:
                with open(f"datasets/smoking/labels/{split}/{name}.txt", "w") as f:
                    f.write("\n".join(f"{cid} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"
                                      for cid, cx, cy, w, h in labels) + "\n")
            else:
                Path(f"datasets/smoking/labels/{split}/{name}.txt").touch()
            if (i + 1) % 500 == 0:
                print(f"  进度: {i + 1}/{count}")

    gen_split(num_train, "cig", "train")
    gen_split(num_val, "cig_val", "val")

    yaml = f"""path: {os.path.abspath('datasets/smoking')}
train: images/train
val: images/val
nc: 2
names: ['cigarette', 'cigarette_pack']
"""
    with open("datasets/smoking/data.yaml", "w") as f:
        f.write(yaml)

    print(f"\n数据集生成完成! 训练:{num_train} 验证:{num_val}")
    print(f"  类别: ['cigarette', 'cigarette_pack']")
    print("=" * 60)


if __name__ == "__main__":
    generate_dataset(num_train=3000, num_val=600)