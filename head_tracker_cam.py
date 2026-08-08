"""
V7-Head + V14-Smoke — 全屏香烟检测
V7: 头部检测 (绿色框) | V14: 全屏香烟 (红色框)
"""
import cv2, numpy as np, time
from ultralytics import YOLO

HEAD = YOLO(r"D:\视觉安防系统\models\yolov8n_head_v7.pt", task='detect')
SMOKE = YOLO(r"D:\视觉安防系统\models\smoke_cig_v24.pt", task='detect')
HEAD.to('cuda'); SMOKE.to('cuda')
print("✅ V7-Head + V24-Smoke | 头绿框 | 烟红框(全屏) | Q退出")

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
W, H = int(cap.get(3)), int(cap.get(4))

GREEN = (0, 255, 100); RED = (0, 0, 255)
fc = 0; t0 = time.time()

# ==================== 光线自适应预处理 ====================
# 应对复杂/多变光照(暗光、强逆光、灯光闪烁、过曝)下的稳定检测
# 策略: 增强帧只用于YOLO模型推理(模型对光照鲁棒),
#       颜色过滤规则仍用原帧(保持已调好的几十条过滤规则行为不变)
LIGHT_MODE = 'auto'      # 'auto'=自动 | 'off'=关闭 | 数值如 1.4=固定提亮 gamma
_luma_ema = None         # 亮度指数滑动平均(防灯光闪烁抖动)

def light_adapt(frame):
    """按环境亮度自动做 CLAHE 局部对比度 + gamma 全局亮度校正"""
    global _luma_ema
    if LIGHT_MODE == 'off':
        return frame
    # 1) 亮度统计(指数滑动平均, 抗闪烁)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    mean = float(gray.mean())
    if _luma_ema is None:
        _luma_ema = mean
    else:
        _luma_ema = 0.9 * _luma_ema + 0.1 * mean   # 慢跟踪, 防闪烁跳变
    luma = _luma_ema
    # 2) CLAHE 局部对比度增强(L通道, 提升暗部/逆光细节)
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clip = 2.5 if luma < 120 else 1.5          # 暗光更强局部增强
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(8, 8))
    l = clahe.apply(l)
    out = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)
    # 3) 全局亮度 gamma 校正(暗光提亮 / 过曝压回)
    if isinstance(LIGHT_MODE, (int, float)):
        gamma = float(LIGHT_MODE)
    elif luma < 75:        # 很暗(关灯/夜视) → 强提亮
        gamma = 0.55
    elif luma < 110:       # 偏暗 → 提亮
        gamma = 0.75
    elif luma > 190:       # 过曝 → 压暗
        gamma = 1.30
    elif luma > 160:       # 偏亮 → 微压
        gamma = 1.12
    else:
        gamma = 1.0
    if abs(gamma - 1.0) > 0.02:
        lut = np.array([((i / 255.0) ** (1.0 / gamma)) * 255 for i in range(256)], dtype=np.uint8)
        out = cv2.LUT(out, lut)
    return out

class KalmanBox:
    def __init__(self, bbox, pn=0.05, mn=0.1):
        # pn: processNoise(信运动, 大=快跟随) | mn: measurementNoise(信检测, 小=更贴检测)
        self.kf = cv2.KalmanFilter(6, 4)
        # 阻尼预测: 速度贡献 1.0 → 0.5 (方向突变时预测不过冲)
        self.kf.transitionMatrix = np.array([[1,0,0,0,0.5,0],[0,1,0,0,0,0.5],[0,0,1,0,0,0],[0,0,0,1,0,0],[0,0,0,0,1,0],[0,0,0,0,0,1]], np.float32)
        self.kf.measurementMatrix = np.eye(4, 6, dtype=np.float32)
        self.kf.processNoiseCov = np.eye(6, dtype=np.float32) * pn
        self.kf.measurementNoiseCov = np.eye(4, dtype=np.float32) * mn
        self.kf.errorCovPost = np.eye(6, dtype=np.float32)
        cx = (bbox[0] + bbox[2]) / 2; cy = (bbox[1] + bbox[3]) / 2
        w = bbox[2] - bbox[0]; h = bbox[3] - bbox[1]
        self.kf.statePost = np.array([[cx], [cy], [w], [h], [0], [0]], np.float32)
        self.bbox = bbox; self.lost = 0; self.confirmed = 0
    def predict(self):
        p = self.kf.predict().flatten()
        self.bbox = (float(p[0]-p[2]/2), float(p[1]-p[3]/2), float(p[0]+p[2]/2), float(p[1]+p[3]/2))
        return self.bbox
    def update(self, bbox):
        cx = (bbox[0] + bbox[2]) / 2; cy = (bbox[1] + bbox[3]) / 2
        w = bbox[2] - bbox[0]; h = bbox[3] - bbox[1]
        # 方向突变检测: 测量与预测中心差 >70px(烟变向) → 速度清零防过冲
        pred = self.kf.statePre.flatten()
        if (cx - pred[0])**2 + (cy - pred[1])**2 > 70**2:
            self.kf.statePost[4] = 0; self.kf.statePost[5] = 0
        self.kf.correct(np.array([[cx], [cy], [w], [h]], np.float32)); self.lost = 0
        self.confirmed += 1
        # 速度限幅 ±15px/帧 (防止连续同向加速导致预测过冲)
        s = self.kf.statePost.flatten()
        s[4] = float(np.clip(s[4], -15, 15)); s[5] = float(np.clip(s[5], -15, 15))
        self.kf.statePost = s.reshape(-1, 1)
        p = s
        self.bbox = (float(p[0]-p[2]/2), float(p[1]-p[3]/2), float(p[0]+p[2]/2), float(p[1]+p[3]/2))

def iou(a, b):
    ix1 = max(a[0], b[0]); iy1 = max(a[1], b[1]); ix2 = min(a[2], b[2]); iy2 = min(a[3], b[3])
    iw = max(0, ix2-ix1); ih = max(0, iy2-iy1)
    aa = (a[2]-a[0])*(a[3]-a[1]); bb = (b[2]-b[0])*(b[3]-b[1])
    return iw*ih/(aa+bb-iw*ih+1e-6)

tracks = []; smoke_tracks = []; next_id = 0
low_suspects = []   # 低分稳定框记忆: [(bbox, 连续帧数)] — 耳后/口边烟连续2帧确认建轨
import os
AUTO_SAVE = r'D:\training_data\smoke\fp_auto'   # 误检帧自动采集(检出烟时保存)
os.makedirs(AUTO_SAVE, exist_ok=True)
last_auto_save = 0

while True:
    ret, frame = cap.read()
    if not ret or frame is None:
        print('[重连] 摄像头断帧, 尝试重连...')
        cap.release()
        ok = False
        for attempt in range(10):
            time.sleep(1)
            cap = cv2.VideoCapture(0)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            ret, frame = cap.read()
            if ret and frame is not None and frame.size > 0:
                ok = True; break
        if not ok:
            print('[重连] 失败, 退出'); break

    for t in tracks: t[0].predict()
    for st in smoke_tracks: st[0].predict()

    # --- 光线自适应: 生成增强帧(仅用于模型推理) ---
    frame_enh = light_adapt(frame)

    # --- V7 头检测 (低阈值, 远距小头也检出) ---
    res = HEAD(frame_enh, conf=0.3, iou=0.5, verbose=False)
    det_heads = []
    for r in res:
        for box in r.boxes:
            b = box.xyxy[0].cpu().numpy()
            x1, y1, x2, y2 = float(b[0]), float(b[1]), float(b[2]), float(b[3])
            bw, bh = x2-x1, y2-y1
            if bw < 30 or bh < 30: continue
            ar = bw/bh if bh > 0 else 0
            if ar < 0.6 or ar > 1.5: continue
            if y2 > H * 0.85: continue
            det_heads.append((x1, y1, x2, y2))

    matched_ids = set(); matched_det = set()
    for j, dh in enumerate(det_heads):
        best_i, best_iou = -1, 0.15
        for i, t in enumerate(tracks):
            if t[1] in matched_ids: continue
            iou_val = iou(t[0].bbox, dh)
            if iou_val < 0.15:
                kf = t[0].kf; ps = kf.statePre.flatten()
                pb = (ps[0]-ps[2]/2, ps[1]-ps[3]/2, ps[0]+ps[2]/2, ps[1]+ps[3]/2)
                iou_val = iou(pb, dh)
            if iou_val > best_iou: best_iou = iou_val; best_i = i
        if best_i >= 0:
            tracks[best_i][0].update(dh); matched_ids.add(tracks[best_i][1]); matched_det.add(j)
    for t in tracks:
        if t[1] not in matched_ids: t[0].lost += 1
    for j, dh in enumerate(det_heads):
        if j not in matched_det and len(tracks) < 20:
            tracks.append([KalmanBox(dh), next_id]); next_id += 1
    tracks = [t for t in tracks if t[0].lost < 5]

    # --- V22 全屏烟检 + 智能过滤(距离机制) + ByteTrack低分池 ---
    raw_smoke = []
    low_pool = []   # ByteTrack 低分候选池 (被conf过滤但≥0.15, 仅用于维持已有轨迹)
    head_list = [(t[0].bbox, t[0].bbox[2]-t[0].bbox[0]) for t in tracks
                 if (t[0].bbox[2]-t[0].bbox[0]) > 0 and (t[0].bbox[3]-t[0].bbox[1]) > 0]
    sr = SMOKE(frame_enh, conf=0.08, iou=0.45, verbose=False)
    for r in sr:
        for box in r.boxes:
            conf = float(box.conf[0])
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            w, h = x2 - x1, y2 - y1
            if min(w, h) < 3: continue
            area_ratio = (w * h) / (W * H)
            aspect = max(w, h) / min(w, h)
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2

            # ===== 距离机制: 头框宽 → 距离分级 → 阈值 + 尺寸一致性校验 =====
            nearest_hw = None
            best_d2 = 1e18
            for hb, hw0 in head_list:
                hcx, hcy = (hb[0]+hb[2])/2, (hb[1]+hb[3])/2
                d2 = (cx-hcx)**2 + (cy-hcy)**2
                if d2 < best_d2:
                    best_d2 = d2; nearest_hw = hw0
            if nearest_hw is not None:
                hw = nearest_hw
                # 距离分级阈值(圆头/竖直烟需要低conf通过, V24对正向小目标conf低)
                # 近距降阈值让圆头烟过; 吸管/手机靠后续"无滤嘴细长"/"手持非烟物"规则挡
                if hw >= 150: dt = 0.32   # 近(<1m): 圆头/竖直烟需要0.30+
                elif hw >= 60: dt = 0.32   # 中(1-3m)
                else: dt = 0.27            # 远(>3m)
                if conf < dt:
                    # ByteTrack 低分池: 被距离阈值拒但≥0.15的合法框 → 供第二级关联(遮挡恢复)
                    # 门槛: 细长(aspect≥1.8挡手机1.78) + 面积/形状约束
                    if conf >= 0.15 and area_ratio < 0.08 and aspect >= 1.8 and aspect <= 6.0:
                        low_pool.append((float(x1), float(y1), float(x2), float(y2)))
                    continue
                # 尺寸一致性: 烟像素宽≈头宽×0.05(±余量); 胳膊/烟盒/手机≈0.25-0.40被拒
                if hw < 60:                       # 远距严格校验
                    if w > hw * 0.12: continue     # 框宽>烟应有宽度2.4倍 → 胳膊/烟盒/手机
                    if w < hw * 0.01: continue     # 远距太窄 → 噪点
                elif hw < 150:                    # 中距: 放宽(手持烟框含手指)
                    if w > hw * 0.22: continue     # 超0.22 → 胳膊/烟盒
            else:
                if conf < 0.40:
                    if conf >= 0.15 and area_ratio < 0.08 and aspect >= 1.8 and aspect <= 6.0:
                        low_pool.append((float(x1), float(y1), float(x2), float(y2)))
                    continue           # 无头部检出 → 极保守

            # 形状: 细长条(圆头/竖直需 conf≥0.30 例外); 极细长/大面积排除
            if conf < 0.30 and aspect < 1.3: continue  # 低置信须细长, 圆头/竖直需≥0.30
            if aspect > 6.0: continue                   # 极细长(窗帘褶/线)
            if area_ratio > 0.08: continue              # 大面积(整只手/物体)

            # 颜色过滤: 烟纸特征 vs 肤色占比分档判定
            xi1, yi1 = max(0, int(x1)), max(0, int(y1))
            xi2, yi2 = min(W, int(x2)), min(H, int(y2))
            if xi2 - xi1 < 4 or yi2 - yi1 < 4: continue
            crop = frame[yi1:yi2, xi1:xi2]
            hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
            pix = max(1, crop.shape[0] * crop.shape[1])
            skin = np.sum(cv2.inRange(hsv, (0,25,50), (45,180,255)) > 0) / pix
            pure_white = np.sum(cv2.inRange(hsv, (0,0,200), (180,25,255)) > 0) / pix
            saturated = np.sum(cv2.inRange(hsv, (0,60,60), (180,255,255)) > 0) / pix
            paper = np.sum(cv2.inRange(hsv, (0,0,150), (180,40,255)) > 0) / pix
            bright_high = np.sum(cv2.inRange(hsv, (0,0,220), (180,255,255)) > 0) / pix
            if pure_white > 0.85: continue        # 整框纯白(白笔/管子/纸卷) → 丢
            if saturated > 0.50: continue         # 大半鲜艳色(绿窗帘/彩物) → 丢
            if bright_high > 0.45: continue       # 高亮均匀面>45%(手机屏幕/亮面反光) → 丢
            glint = np.sum(cv2.inRange(hsv, (0,0,235), (180,60,255)) > 0) / pix
            if glint > 0.08 and area_ratio > 0.01: continue  # 金属/边框高反光(手机侧面) → 丢
            # ===== 烟特征检查 (核心防线): 真烟必须有白纸/滤嘴/亮度特征 =====
            # 暗色低纹理框(不白不棕不亮) = 手/杯子/阴影/暗吸管 → 丢
            brown = np.sum(cv2.inRange(hsv, (8,40,60), (40,200,200)) > 0) / pix  # 滤嘴棕
            paper_mid = np.sum(cv2.inRange(hsv, (0,0,140), (180,45,255)) > 0) / pix  # 烟纸白亮
            if pure_white < 0.12 and brown < 0.05 and paper_mid < 0.30 and bright_high < 0.30:
                continue    # 无烟特征(非白非棕非亮) → 非烟物体 → 丢
            # ===== 吸管/玻璃制品: 细长+纯白+反光 = 透明塑料/玻璃 =====
            # 吸管特征: aspect>3细长, pure_white>0.5(白塑料), glint>0.10(边缘反光)
            if aspect > 3.0 and pure_white > 0.50 and glint > 0.10:
                continue    # 吸管/玻璃管/试管 → 丢
            # ===== 无滤嘴细长物拒检 (吸管/笔/白管 vs 真烟): =====
            # 真烟必有棕色滤嘴(brown≥3%)或高亮烟头; 吸管全白无滤嘴
            # 去掉conf限制(吸管conf常>0.55也必须挡); 手持白烟滤嘴可见时brown≥3%保留
            if aspect > 2.5 and brown < 0.04:
                continue    # 任何conf的细长无滤嘴物 → 吸管/笔/塑料管 → 丢
            # ===== 手持非烟物拒检 (核心: 手+吸管/手机误判根因) =====
            # 实验发现: 手放过去, 手+吸管/手机被V24判烟(训练数据手持烟=手+烟)
            # 区分: 框内主要是手(skin>0.55) + 无滤嘴特征(brown<0.04) → 手持吸管/手机/笔
            # 注意: 烟在嘴前/脸前(吸烟动作)框内含部分脸肤色, skin通常0.3-0.5 → 保留
            # 手持真烟: 框内含滤嘴(brown≥4%) → 保留
            if skin > 0.55 and brown < 0.04:
                continue    # 手持无滤嘴物 → 吸管/手机/笔 → 丢
            # ===== 手机/书本特征 (V24高置信误检防线) =====
            # 手机: 非细长(宽高比0.45-1.3) + 中大面积(>0.3%) + 内部有亮屏区(高亮>15%)或暗屏区(暗>30%)
            if aspect < 1.3 and area_ratio > 0.003:
                bright_mid = np.sum(cv2.inRange(hsv, (0,0,140), (180,60,255)) > 0) / pix
                dark = np.sum(cv2.inRange(hsv, (0,0,0), (180,50,120)) > 0) / pix
                if bright_mid > 0.30: continue    # 大片亮屏区(手机亮屏/平板) → 丢
                if dark > 0.55: continue          # 大片暗区(手机黑屏/书本) → 丢


            if area_ratio < 0.01:
                # 远距离小目标: 手指含背景肤色占比仍>20%, 烟纸白肤色<10%
                if skin > 0.20: continue
            else:
                # 近距离: 肤色主导=手指(绷直/弯曲); 手持烟肤色通常<50%
                if skin > 0.80: continue
                if skin > 0.50 and paper < skin * 0.5: continue

            # 头框重叠: 中心在头框内的小框 = 脸上特征(鼻/嘴) 排除
            in_face = False
            for t in tracks:
                hx1, hy1, hx2, hy2 = t[0].bbox
                h_area = (hx2 - hx1) * (hy2 - hy1)
                if h_area <= 0: continue
                # 重叠面积
                ix1 = max(x1, hx1); iy1 = max(y1, hy1)
                ix2 = min(x2, hx2); iy2 = min(y2, hy2)
                inter = max(0, ix2-ix1) * max(0, iy2-iy1)
                # 情况A: 检测框几乎覆盖头框(>50%) = 整个头被当烟 → 丢
                if inter / h_area > 0.5:
                    in_face = True; break
                # 情况B: 中心在头框内的极小框(<5%)
                #   细长(aspect≥2.5) = 耳后/口边烟 → 放行(夹耳后需求)
                #   非细长 = 脸上特征(鼻/嘴/眼) → 丢
                if hx1 <= cx <= hx2 and hy1 <= cy <= hy2:
                    if (w * h) / h_area < 0.05 and aspect < 2.5:
                        in_face = True; break
            if in_face: continue
            # 兜底: 无头框时, 大而圆的检测框(占画面3%+ 且 近圆形) = 头/大圆物 → 丢
            if area_ratio > 0.03 and aspect < 1.5:
                continue

            raw_smoke.append((float(x1), float(y1), float(x2), float(y2)))

    # 合并重叠检测框: 同一烟分裂的框合成一个(iou或中心近都合并)
    merged = []
    for sb in sorted(raw_smoke, key=lambda b: -(b[2]-b[0])*(b[3]-b[1])):
        dup = False
        scx, scy = (sb[0]+sb[2])/2, (sb[1]+sb[3])/2
        for mb in merged:
            if iou(sb, mb) > 0.3:
                dup = True; break
            mcx, mcy = (mb[0]+mb[2])/2, (mb[1]+mb[3])/2
            if ((scx-mcx)**2 + (scy-mcy)**2) ** 0.5 < 40:
                dup = True; break
        if not dup:
            merged.append(sb)
    raw_smoke = merged

    # === ByteTrack 两级关联 (Kalman预测 + 高分/低分双层) ===
    sm_matched = set(); sm_det = set()
    # Level 1: 高分检测框 → 现有轨迹 (IoU优先, 移动快时中心距补偿)
    for j, sb in enumerate(raw_smoke):
        best_i, best_score = -1, 0.0
        scx, scy = (sb[0]+sb[2])/2, (sb[1]+sb[3])/2
        for i, st in enumerate(smoke_tracks):
            if st[1] in sm_matched: continue
            tb = st[0].bbox
            tcx, tcy = (tb[0]+tb[2])/2, (tb[1]+tb[3])/2
            iou_val = iou(tb, sb)
            dist = ((scx-tcx)**2 + (scy-tcy)**2) ** 0.5
            # 综合评分: IoU优先; 移动快时IoU低但中心近 → 距离补偿(120px)
            score = iou_val
            if iou_val < 0.15 and dist < 120:
                score = 0.15 + 0.15 * (1 - dist / 120)
            if score > best_score:
                best_score = score; best_i = i
        if best_i >= 0 and best_score >= 0.15:
            smoke_tracks[best_i][0].update(sb); sm_matched.add(smoke_tracks[best_i][1]); sm_det.add(j)
    # Level 2 (ByteTrack核心): 低分池 → 未匹配轨迹 (遮挡/远距恢复, 更严格IoU)
    for sb in low_pool:
        if len(smoke_tracks) == 0: break
        best_i, best_score = -1, 0.30
        scx, scy = (sb[0]+sb[2])/2, (sb[1]+sb[3])/2
        for i, st in enumerate(smoke_tracks):
            if st[1] in sm_matched: continue
            tb = st[0].bbox
            tcx, tcy = (tb[0]+tb[2])/2, (tb[1]+tb[3])/2
            iou_val = iou(tb, sb)
            dist = ((scx-tcx)**2 + (scy-tcy)**2) ** 0.5
            score = iou_val
            if iou_val < 0.25 and dist < 55:   # 低分恢复: 位置强约束, 避免误关联
                score = 0.25 + 0.10 * (1 - dist / 55)
            if score > best_score:
                best_score = score; best_i = i
        if best_i >= 0 and best_score >= 0.30:
            smoke_tracks[best_i][0].update(sb); sm_matched.add(smoke_tracks[best_i][1])
    # 未匹配轨迹: lost+1 (Kalman预测维持)
    for st in smoke_tracks:
        if st[1] not in sm_matched: st[0].lost += 1
    # 低分稳定框建轨 (耳后/口边烟等新目标, conf低但连续2帧同位置 = 真目标)
    if len(smoke_tracks) < 6:
        for sb in low_pool:
            # 只处理未匹配过轨迹的低分框
            already = False
            scx, scy = (sb[0]+sb[2])/2, (sb[1]+sb[3])/2
            for st in smoke_tracks:
                tb = st[0].bbox
                tcx, tcy = (tb[0]+tb[2])/2, (tb[1]+tb[3])/2
                if ((scx-tcx)**2 + (scy-tcy)**2) ** 0.5 < 40:
                    already = True; break
            if already: continue
            # 与历史低分框匹配: 连续2帧同一位置 → 建轨
            hit = False
            for k, (lb, cnt) in enumerate(low_suspects):
                lcx, lcy = (lb[0]+lb[2])/2, (lb[1]+lb[3])/2
                if ((scx-lcx)**2 + (scy-lcy)**2) ** 0.5 < 30:
                    low_suspects[k] = (sb, cnt + 1)
                    if cnt + 1 >= 2:   # 连续2帧稳定 → 建轨前手机/手持物校验
                        # 手机特征: 亮屏/暗屏/肤色占比高 → 拒绝建轨
                        xi1, yi1 = max(0, int(sb[0])), max(0, int(sb[1]))
                        xi2, yi2 = min(W, int(sb[2])), min(H, int(sb[3]))
                        reject = False
                        if xi2 - xi1 >= 6 and yi2 - yi1 >= 6:
                            crop = frame[yi1:yi2, xi1:xi2]
                            hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
                            pix = crop.shape[0] * crop.shape[1]
                            bright = np.sum(cv2.inRange(hsv, (0,0,200), (180,255,255)) > 0) / pix
                            dark = np.sum(cv2.inRange(hsv, (0,0,0), (180,50,120)) > 0) / pix
                            skin = np.sum(cv2.inRange(hsv, (0,25,50), (45,180,255)) > 0) / pix
                            if bright > 0.45 or dark > 0.55 or skin > 0.50:
                                reject = True
                        if not reject:
                            smoke_tracks.append([KalmanBox(sb, pn=0.30, mn=0.05), next_id + 5000])
                            next_id += 1
                        low_suspects.pop(k)
                    hit = True; break
            if not hit:
                low_suspects.append((sb, 1))
    low_suspects = [(lb, cnt) for lb, cnt in low_suspects if cnt < 8]  # 8帧未确认则遗忘
    # 新轨迹: 仅高分框创建 (低分池不建新轨, 防误检污染); 烟用快跟随Kalman(减少漂移)
    for j, sb in enumerate(raw_smoke):
        if j not in sm_det and len(smoke_tracks) < 6:
            smoke_tracks.append([KalmanBox(sb, pn=0.30, mn=0.05), next_id + 5000]); next_id += 1
    for st in smoke_tracks:
        # 漂移抑制: 预测框出画面直接淘汰
        bx1, by1, bx2, by2 = st[0].bbox
        if bx2 < 0 or by2 < 0 or bx1 > W or by1 > H:
            st[0].lost = 99
    # 未确认轨道(噪声)快速清除: confirmed<2 的10帧即清(熬过波动防闪); 稳定轨道20帧
    smoke_tracks = [st for st in smoke_tracks
                    if st[0].lost < (20 if st[0].confirmed >= 2 else 10)]

    # 误检帧自动采集: 有确认烟轨时保存整帧(限频), 供筛选负样本
    if smoke_tracks and time.time() - last_auto_save >= 0.5:
        last_auto_save = time.time()
        cv2.imwrite(os.path.join(AUTO_SAVE, f'auto_{int(last_auto_save)}.jpg'), frame)

    # --- 绘制 ---
    for t in tracks:
        x1, y1, x2, y2 = map(int, t[0].bbox)
        cv2.rectangle(frame, (x1,y1), (x2,y2), GREEN, 2)
        lb = f"#{t[1]}"
        (tw,th),_ = cv2.getTextSize(lb, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)
        cv2.rectangle(frame, (x1,y1-14), (x1+tw+4,y1), GREEN, -1)
        cv2.putText(frame, lb, (x1+2,y1-3), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,0), 1)

    for st in smoke_tracks:
        # 显示: 稳定轨全显; 新轨(lost<3)也显(减少"刚出现就消失"的闪烁)
        sw, sh = st[0].bbox[2]-st[0].bbox[0], st[0].bbox[3]-st[0].bbox[1]
        if (sw*sh)/(W*H) < 0.01 and st[0].confirmed < 1 and st[0].lost >= 3:
            continue
        sx1, sy1, sx2, sy2 = map(int, st[0].bbox)
        cv2.rectangle(frame, (sx1,sy1), (sx2,sy2), RED, 3)
        cv2.putText(frame, "CIG", (sx1, sy2+18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, RED, 2)

    cv2.putText(frame, f"V14-Smoke | {len(tracks)} heads | {len(smoke_tracks)} cigs | {fc/max(1,time.time()-t0):.0f}fps",
                (4,18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, GREEN, 2)
    cv2.imshow("Head + Smoke Detection", frame)
    fc += 1
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release(); cv2.destroyAllWindows()
