"""
V7-Head + V14-Smoke — 全屏香烟检测
V7: 头部检测 (绿色框) | V14: 全屏香烟 (红色框)
"""
import cv2, numpy as np, time, torch
from ultralytics import YOLO

HEAD = YOLO(r"D:\视觉安防系统\models\yolov8n_head_v7.pt", task='detect')
SMOKE = YOLO(r"D:\视觉安防系统\models\smoke_cig_v30.pt", task='detect')
HEAD.to('cuda'); SMOKE.to('cuda')
# ★★ 手部检测模型(新架构: 手掌蓝色框 + 香烟聚焦头/手区域)
#   hand.pt 由 COCO 手部数据集训练, 训练完成后放入 models/ 自动启用
#   hand.pt 不存在时 HAND_READY=False → 禁用手部功能, 烟检回退全屏
try:
    HAND = YOLO(r"D:\视觉安防系统\models\hand.pt", task='detect')
    HAND.to('cuda')
    HAND_READY = True
    print("✅ 手部模型已加载(hand.pt)")
except Exception:
    HAND_READY = False
    print("⚠️ hand.pt 未就绪, 手部功能禁用, 烟检回退全屏")
# ★ 半精度(fp16)推理加速: 仅 SMOKE 用(烟模型大, 提速明显)
#   HEAD 保持 fp32: 头部小目标(遮挡/半脑袋)在fp16下精度损失→丢失, 头推理仅6ms不拖帧率
try:
    SMOKE.model.half()
except Exception:
    print("⚠️ SMOKE fp16转换失败, 继续用fp32")
print("✅ V7-Head + V30-Smoke | 头绿框 | 手蓝框 | 烟红框(头/手区域聚焦) | Q退出")

# ★ ROI放大检测的结果模拟对象(兼容 ultralytics box 接口)
#   手/头区域裁图放大后独立跑SMOKE, 检出框需换算回全图坐标再进入过滤链
class _RoiBox:
    __slots__ = ('xyxy', 'conf')
    def __init__(self, x1, y1, x2, y2, c):
        self.xyxy = [torch.tensor([float(x1), float(y1), float(x2), float(y2)])]
        self.conf = [torch.tensor(float(c))]
class _RoiRes:
    __slots__ = ('boxes',)
    def __init__(self, boxes):
        self.boxes = boxes

cap = cv2.VideoCapture(0)
# 单实例锁: 防止多实例抢同一个摄像头导致"显示混乱/头框丢失"
try:
    import ctypes
    _kernel32 = ctypes.windll.kernel32
    _hMutex = _kernel32.CreateMutexW(None, False, "Global\\HeadTrackerCam_SingleInstance_v1")
    _lastErr = _kernel32.GetLastError()
    if _lastErr == 183:   # ERROR_ALREADY_EXISTS
        print("[ERROR] 另一个追踪器实例已在运行! 请先关掉旧的再启动。")
        import time; time.sleep(5); exit(1)
    _SINGLE_INSTANCE_MUTEX = _hMutex
except Exception as _e:
    _SINGLE_INSTANCE_MUTEX = None
    print(f"[WARN] 单实例锁创建失败: {_e}")
# ★ 摄像头 1280x720(恢复原分辨率): 540p源图放大进640输入会模糊头部细节→置信度波动丢失
#   720p源→640是缩小(细节保留), 头部遮挡/半脑袋小目标识别恢复
#   帧率靠 fp16 + CLAHE 4x4 + 移除副检测 保证(720p下light_adapt ~7ms, 总预算~27ms)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_FPS, 60)   # ★ 请求60FPS输出(摄像头支持则生效, 不支持自动回落)
W, H = int(cap.get(3)), int(cap.get(4))

GREEN = (0, 255, 100); RED = (0, 0, 255)
fc = 0; t0 = time.time()

# ==================== 光线自适应预处理 ====================
# 应对复杂/多变光照(暗光、强逆光、灯光闪烁、过曝)下的稳定检测
# 策略: 增强帧只用于YOLO模型推理(模型对光照鲁棒),
#       颜色过滤规则仍用原帧(保持已调好的几十条过滤规则行为不变)
LIGHT_MODE = 'auto'      # 'auto'=自动 | 'off'=关闭 | 数值如 1.4=固定提亮 gamma
_luma_ema = None         # 亮度指数滑动平均(防灯光闪烁抖动)
_clahe_obj = None        # CLAHE 对象缓存(帧率优化: 避免每帧重建)
_last_gamma = None       # 上次 gamma 值(仅变化时重建 LUT)
_lut_cache = None        # gamma LUT 缓存

def light_adapt(frame):
    """光线自适应 v3: 减少灰蒙蒙(避免 4 重压暗叠加)
    ① 过曝感知: 过曝占比>8% 禁用CLAHE + 强压gamma(防过曝误检)
    ② 局部亮度平衡(明暗不统一时启用, 目标130偏亮→ 减少灰蒙蒙)
    ③ 暗光 CLAHE 提纹(烟纸纤维凸显) — tileGridSize 8x8(更柔和, 避免刷平细节)
    ④ gamma: 正常情况 1.0(不压暗, 由显示层 0.70 统一压, 避免双重叠加)
    ⑤ 动态光变快速响应(luma_ema 0.7)
    """
    global _luma_ema
    if LIGHT_MODE == 'off':
        return frame
    # 1) 亮度统计(指数滑动平均, 快速响应光变)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    mean = float(gray.mean())
    if _luma_ema is None:
        _luma_ema = mean
    else:
        _luma_ema = 0.7 * _luma_ema + 0.3 * mean
    luma = _luma_ema
    # 2) 过曝占比 + 明暗差异检测
    overexp_mask = gray > 240
    pct_overexposed = float(overexp_mask.mean())
    # 明暗差异: 8x6 分块亮度极差(>70 灰阶 → 光线不统一, 需亮度平衡)
    gh, gw = gray.shape
    bh, bw = max(1, gh//6), max(1, gw//8)
    block_means = []
    for by in range(0, gh - bh + 1, bh):
        for bx in range(0, gw - bw + 1, bw):
            block_means.append(float(gray[by:by+bh, bx:bx+bw].mean()))
    bright_range = (max(block_means) - min(block_means)) if block_means else 0.0
    # 3) LAB 空间处理
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    lf = l.astype(np.float32)
    if pct_overexposed > 0.08:
        # 过曝场景: 不做局部增强(防刷平过曝区), 只靠 gamma 强压
        out = frame.copy()
    else:
        # ③ 局部亮度平衡(明暗不统一 → 均衡): L -= 背景估计 + 目标灰度
        #    bg(大核均值) 代表局部背景亮度; L-bg 消除明暗差
        #    目标130(原105→130: 偏亮, 减少灰蒙蒙)
        if bright_range > 70.0:
            bg = cv2.GaussianBlur(lf, (0, 0), 21)
            target = 130.0     # ★ 105→130: 平衡后偏亮, 保留更多明暗细节
            lf = lf - bg + target
            lf = np.clip(lf, 0, 255)
        # ④ CLAHE 局部对比度(提纹) — tileGridSize 8x8(原4x4→8x8: 更柔和, 避免刷平)
        global _clahe_obj
        if luma < 120:
            clip = 3.0    # 暗光强提纹
        else:
            clip = 1.5
        if _clahe_obj is None or abs(_clahe_obj.getClipLimit() - clip) > 0.01:
            # ★ 8x8→5x5: 更精细的局部增强 — 8x8块太大(720p下~90px), 烟纸细纹理被平均刷平
            #   5x5(~57px)局部直方图均衡保留细纹理(烟纸纤维/滤嘴纹路), 细节更明显
            _clahe_obj = cv2.createCLAHE(clipLimit=clip, tileGridSize=(5, 5))
        lf = _clahe_obj.apply(lf.astype(np.uint8)).astype(np.float32)
        # ⑤ ★ 边缘对比度增强(Unsharp Mask) — 治"白背景烟边缘梯度不足"
        #    用户实测: 白烟在浅色背景(白门/白墙)上识别不出 — 边缘梯度仅5-15灰阶, 模型看不见
        #    Unsharp: lf + amount*(lf - blur) → 边缘过冲, 梯度放大, 烟轮廓凸显
        #    ★ 用户要求加强: amount 1.5→2.8(更强边缘), std阈值 60→80(更多浅色场景触发)
        #    触发条件: 高亮(luma>140) + 低梯度(std<80) = 均匀浅色背景场景
        #    正常/暗光场景 amount≈0 不触发, 避免噪声放大
        if luma > 140:
            _gstd = float(np.std(gray))
            if _gstd < 80.0:
                _blur = cv2.GaussianBlur(lf, (0, 0), 3)
                lf = lf + 2.8 * (lf - _blur)   # ★ amount=2.8(原1.5): 更强边缘强化
                lf = np.clip(lf, 0, 255)
        # ⑥ ★★ 分块清晰度补偿 — 平均化全屏检测精力(治"区域检测不均")
        #    实测: 画面中上部 Laplacian 方差 1-37(模糊/白墙低对比), 下部 682-825(清晰)
        #    → 中上部的小烟 conf 暴跌 → 漏检(同一根烟换个位置就识别不出)
        #    补偿原理: 计算整帧梯度图, 低梯度(平坦/模糊)区域做"额外Unsharp锐化",
        #    把弱检测区信号拉齐到与清晰区一致 → 全屏检测精力平均化
        #    方法: 梯度掩膜(低梯度=1, 高梯度=0) × 额外锐化量, 掩膜高斯平滑防接缝
        _gx = cv2.Sobel(lf, cv2.CV_32F, 1, 0, ksize=3)
        _gy = cv2.Sobel(lf, cv2.CV_32F, 0, 1, ksize=3)
        _grad = np.abs(_gx) + np.abs(_gy)
        _low_mask = (_grad < 10.0).astype(np.float32)        # 低清晰度区域掩膜
        _low_mask = cv2.GaussianBlur(_low_mask, (0, 0), 15)  # 平滑(防块接缝)
        _low_mask = _low_mask / max(1e-6, float(_low_mask.max()))  # 归一化0-1
        _b3 = cv2.GaussianBlur(lf, (0, 0), 3)
        # ★ amount 1.2→0.6: 低清晰区额外锐化减弱 — 过强会把烟纸细纹理反复钝化
        #   补偿目的只是"拉平区域差异", 0.6 足够且保留纹理细节
        lf = lf + 0.6 * (lf - _b3) * _low_mask              # 低清晰区额外锐化(减弱)
        lf = np.clip(lf, 0, 255)
        out = cv2.cvtColor(cv2.merge([lf.astype(np.uint8), a, b]), cv2.COLOR_LAB2BGR)
    # 5) gamma 校正: 正常情况 1.0(原1.10→1.0: 不在 light_adapt 压暗, 统一交给显示层 0.70)
    if isinstance(LIGHT_MODE, (int, float)):
        gamma = float(LIGHT_MODE)
    elif pct_overexposed > 0.20:    # 极过曝 → 极强压
        gamma = 2.2
    elif pct_overexposed > 0.08:    # 过曝 → 强压
        gamma = 1.8
    elif luma < 75:        # 很暗 → 温和提亮
        gamma = 0.75
    elif luma < 110:       # 偏暗 → 微提
        gamma = 0.88
    elif luma > 150:
        # ★ 白背景特化(治"白烟在白背景识别不出"): 高亮+均匀(白墙/白桌面)
        gstd = float(np.std(gray))
        if gstd < 50:
            gamma = 1.6    # 白背景场景: 强压
        else:
            gamma = 1.30
    else:
        gamma = 1.0        # ★ 原1.10→1.0: 正常情况不压暗(避免双重叠加灰蒙蒙)
    if abs(gamma - 1.0) > 0.02:
        global _last_gamma, _lut_cache
        if _last_gamma is None or abs(_last_gamma - gamma) > 0.01:
            _lut_cache = np.array([((i / 255.0) ** (1.0 / gamma)) * 255 for i in range(256)], dtype=np.uint8)
            _last_gamma = gamma
        out = cv2.LUT(out, _lut_cache)
    return out

# ==================== 长期追踪签名 (Re-ID) ====================
# 烟丢失后, 在预测位置附近找"位置近+尺寸像+颜色像"的候选, 恢复原ID继续追踪
SIG_HIST_SIZE = 64       # 签名直方图bin数
SIG_RECOVER_RANGE = 2.5  # 找回搜索范围 = 丢失时框宽 × 此系数
SIG_RECOVER_FRAMES = 10  # 最长找回帧数(~0.5秒), 超时清轨防幽灵(原30帧=1.5s太久)

def extract_sig(frame, bbox):
    """提取目标外观签名: HSV直方图 + 尺寸指纹
    bbox: (x1,y1,x2,y2) 原帧坐标系
    返回 dict(hist=归一化直方图, w=宽, h=高, nz=非零bin数) 或 None
    纯色/低纹理区域(黑屏/白墙)返回None → 防止"空直方图"被误判为相似
    """
    x1, y1, x2, y2 = map(int, bbox)
    Hf, Wf = frame.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(Wf, x2), min(Hf, y2)
    if x2 - x1 < 3 or y2 - y1 < 3:
        return None
    crop = frame[y1:y2, x1:x2]
    # 纯色/低纹理拒绝(鲁棒方案): 灰度方差低 = 无区分度
    # 真实烟: 滤嘴棕+烟纸白+阴影 → 方差高; 黑屏/白墙/纯色物 → 方差≈0
    gray_std = float(np.std(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)))
    if gray_std < 8.0:
        return None   # 整块无纹理 → 拒绝(防黑屏/白墙/纯色误匹配)
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [SIG_HIST_SIZE, SIG_HIST_SIZE], [0, 180, 0, 256])
    cv2.normalize(hist, hist)
    return {'hist': hist, 'w': float(x2 - x1), 'h': float(y2 - y1)}

def sig_similarity(a, b):
    """宽容相似度: 位置约束在主流程做, 这里只比外观(颜色+尺寸)
    返回 (颜色相似度0-1, 尺寸是否量级对)
    """
    if a is None or b is None:
        return 0.0, False
    # 颜色: 直方图相关性 (CV_COMP_CORREL, 1=完全一致, 0=无关, 负=相反)
    # 双签名都必须有足够纹理(extract_sig已过滤纯色) → 相关性可靠
    try:
        corr = cv2.compareHist(a['hist'], b['hist'], cv2.HISTCMP_CORREL)
        corr = max(0.0, corr)   # 负相关无意义, 截断到0
    except Exception:
        corr = 0.0
    # 尺寸量级: 宽高比是否都是"细长/相似量级" (±35%内算量级对)
    ar_a = a['w'] / max(1e-6, a['h'])
    ar_b = b['w'] / max(1e-6, b['h'])
    size_ok = (0.65 <= ar_a / max(1e-6, ar_b) <= 1.55) and \
              (0.5 <= a['w'] / max(1e-6, b['w']) <= 2.0)
    return corr, size_ok

# ==================== 轻量材质识别 (烟纸 vs 塑料) ====================
# 原理: 烟纸=哑光+纤维纹理+漫反射(纹理密/反光稳)
#       塑料吸管=光滑+无纹理+镜面反射(纹理平/反光闪烁)
# 无需新模型, 用GLCM纹理统计 + 高光点密度 + 反光稳定性 区分

def texture_density(gray):
    """GLCM纹理密度: 局部灰度变化量(能量/对比度综合)
    返回 0-1 值: 高=纹理密(纸/织物), 低=光滑(塑料/玻璃)
    """
    if gray.size == 0:
        return 0.0
    # 梯度幅值均值 = 简易纹理强度(塑料光滑→梯度小, 纸有纹理→梯度大)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    # 归一化: 典型纸纹理梯度均值~25-60, 塑料<15
    td = float(mag.mean()) / 80.0
    return min(1.0, td)

def glint_density(crop):
    """镜面高光密度: 高亮且低饱和像素占比(真正的镜面反光点)
    ★ 阈值V≥235: 区分"镜面反光"(塑料/玻璃/金属→V≥235高光点) 
      与"漫反射白"(烟纸/纸张→V≈200-230均匀, 不算反光)
    塑料/玻璃/金属: 反光点高密度; 纸: 漫反射无反光点
    """
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    pix = max(1, crop.shape[0] * crop.shape[1])
    # 镜面高光: 极高亮度(V≥235) + 低饱和(S<60) = 白亮反射点(非漫反射)
    glint = np.sum(cv2.inRange(hsv, (0, 0, 235), (180, 60, 255)) > 0) / pix
    return glint

def is_paper_material(crop, area_ratio, last_glint=None):
    """材质判定: 返回 (是否纸材质, 纹理密度, 高光密度, 更新后高光记忆)
    ★ 判定: 纸必须"有纹理 AND 低反光"同时满足(哑光+纤维纹理+漫反射)
      - 塑料吸管: 纹理平 + 高反光 → 拒
      - 白墙/门: 边缘梯度高但整体反光强 → 拒(误检重灾区)
      - 玻璃/金属: 高光密度高 → 拒
      - 真烟纸: 纤维纹理 + 哑光低反光 → 过
    远距小目标(area_ratio<0.002)纹理不可靠 → 放行靠形状+颜色+高conf
    """
    if crop is None or crop.size < 16:
        return True, 0.0, 0.0, last_glint   # 太小无法判断, 放行
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    td = texture_density(gray)
    glint = glint_density(crop)
    # 小目标也查材质(原0.002阈值跳过太宽, 细长塑料吸管area_ratio小被漏过)
    #   太小的框(<6px)Sobel不可靠才跳过
    if crop.shape[0] < 6 or crop.shape[1] < 6:
        return True, td, glint, last_glint
    # 核心判据(AND): 纸 = 有纹理(≥0.10) 且 低反光(<0.25)
    # ★ 门槛已放宽(用户反馈光线影响材质判定): 原0.15/0.15 过严, 光线变化时误杀真烟
    is_paper = (td >= 0.06) and (glint < 0.30)   # ★ td 0.10→0.06 / glint 0.25→0.30 (应对光线过滤纹理变化, 减小材质判定严格度)
    if not is_paper:
        # 双保险: 有纹理且帧间反光稳定(波动<0.05) → 漫反射=纸(塑料反光会闪烁)
        if last_glint is not None and last_glint > 0:
            if td >= 0.06 and abs(glint - last_glint) < 0.05:
                is_paper = True
    return is_paper, td, glint, glint

def try_recover_smoke(frame, st, all_boxes, W, H):
    """丢失找回核心: 在预测位置附近搜索, 位置近+签名匹配 → 恢复原ID
    st = [KalmanBox, track_id]; 返回匹配到的bbox或None
    优先级: 位置 > 尺寸 > 颜色 (外观差距大也能找回)
    """
    kf = st[0]
    if kf.lost < 1 or kf.lost > SIG_RECOVER_FRAMES:
        return None
    tb = kf.bbox
    tcx, tcy = (tb[0] + tb[2]) / 2, (tb[1] + tb[3]) / 2
    tw, th = tb[2] - tb[0], tb[3] - tb[1]
    # 搜索中心: 用"最后确认位置"作锚点(比漂移的预测框更可信), 再加小速度外推
    lg = kf.last_good
    lgcx, lgcy = (lg[0]+lg[2])/2, (lg[1]+lg[3])/2
    kf_ps = kf.kf.statePost.flatten()
    vx, vy = float(kf_ps[4]), float(kf_ps[5])
    # 锚点 = 最后确认位置 + 速度外推(限1帧, 防止跟着漂移框走远)
    ax = lgcx + vx; ay = lgcy + vy
    # 搜索范围: 随丢失帧数扩大(每帧+20%), 封顶40%画面; 至少覆盖烟宽3倍
    search_r = min(max(40.0, tw * SIG_RECOVER_RANGE) * (1.0 + 0.20 * kf.lost), W * 0.4)
    best_b, best_score = None, 0.0
    for b in all_boxes:
        bcx, bcy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
        d = ((bcx - ax) ** 2 + (bcy - ay) ** 2) ** 0.5
        if d > search_r:
            continue   # 位置太远: 不可能是同一个烟
        # 位置近 → 计算外观相似度(统一用原帧, 与主流程签名一致)
        bw, bh = b[2] - b[0], b[3] - b[1]
        sig_b = extract_sig(frame, b)
        # 双通道判定: 位置是硬门槛, 外观里"尺寸量级对"或"颜色像"任一命中
        if kf.sig is not None and sig_b is not None:
            corr, size_ok = sig_similarity(kf.sig, sig_b)
        else:
            corr, size_ok = 0.0, True   # 无签名: 只靠位置+尺寸
        # 尺寸粗校验(用头宽比例常识): 烟应小而细长, 排除大物体
        aspect = max(bw, bh) / max(1e-6, min(bw, bh))
        if aspect < 1.3 and bw * bh > (W * H) * 0.01:
            size_ok = False   # 大而近圆 → 手/脸/物体
        # 高灵敏度: 位置近(距离<0.5搜索窗)的候选降低分数门槛
        pos_w = 1.0 - d / max(search_r, 1e-6)
        score = pos_w * 0.6 + (corr if size_ok else 0.0) * 0.4
        if score > best_score:
            best_score, best_b = score, b
    # 阈值: 位置很近(前50%窗)时放宽到0.30, 否则0.35
    if best_b is not None:
        best_d = ((best_b[0]+best_b[2])/2 - ax) ** 2 + ((best_b[1]+best_b[3])/2 - ay) ** 2
        best_d = best_d ** 0.5
        if best_score >= (0.30 if best_d <= search_r * 0.5 else 0.35):
            return best_b
    return None

class KalmanBox:
    def __init__(self, bbox, pn=0.05, mn=0.1, sig=None):
        # pn: processNoise(信运动, 大=快跟随) | mn: measurementNoise(信检测, 小=更贴检测)
        self.kf = cv2.KalmanFilter(6, 4)
        # 阻尼预测: 速度贡献 1.0 → 0.5 (方向突变时预测不过冲)
        self.kf.transitionMatrix = np.array([[1,0,0,0,1.0,0],[0,1,0,0,0,1.0],[0,0,1,0,0,0],[0,0,0,1,0,0],[0,0,0,0,1,0],[0,0,0,0,0,1]], np.float32)  # ★速度贡献1.0(满速跟随)
        self.kf.measurementMatrix = np.eye(4, 6, dtype=np.float32)
        self.kf.processNoiseCov = np.eye(6, dtype=np.float32) * pn
        self.kf.measurementNoiseCov = np.eye(4, dtype=np.float32) * mn
        self.kf.errorCovPost = np.eye(6, dtype=np.float32)
        cx = (bbox[0] + bbox[2]) / 2; cy = (bbox[1] + bbox[3]) / 2
        w = bbox[2] - bbox[0]; h = bbox[3] - bbox[1]
        self.kf.statePost = np.array([[cx], [cy], [w], [h], [0], [0]], np.float32)
        self.bbox = bbox; self.lost = 0; self.confirmed = 0
        self.disp_bbox = bbox   # 显示用平滑框(EMA, 防瞬移/跳变)
        # ===== 长期追踪签名 (Re-ID): 丢失后找回用 =====
        self.sig = sig          # 外观签名 (None=尚未采集)
        self.sig_pool = []      # 历史签名池(最近N个代表外观, 容忍外观演变)
        self.candidates = []    # 丢失期间的低分候选 [(bbox, 相似度)] 供恢复验证
        self.last_good = bbox   # 最后确认位置(丢失期间防预测飘移的锚点)
    def predict(self):
        # 丢失期间漂移抑制: 速度按丢失帧数衰减(0.7^lost)
        # ★ 回拉只对"低速烟"生效(疑似停住→黄框停附近防飘);
        #   烟仍在移动(速度快)→ 不回拉, 预测继续外推(黄框跟着走, 不卡住)
        if self.lost > 0:
            s = self.kf.statePost.flatten()
            raw_speed = abs(s[4]) + abs(s[5])     # 衰减前速度(判断烟是否在移动)
            decay = 0.7 ** min(self.lost, 5)      # 丢失越久速度贡献越小
            s[4] *= decay; s[5] *= decay
            self.kf.statePost = s.reshape(-1, 1)
            if raw_speed < 8.0:                   # 原本就低速(烟疑似停住) → 回拉防飘
                # 回拉系数: lost=1拉回30%, lost=5+拉回80% → 停在最后确认位置附近
                pull = min(0.30 + 0.10 * self.lost, 0.80)
                lg = self.last_good
                lgcx, lgcy = (lg[0]+lg[2])/2, (lg[1]+lg[3])/2
                s[0] = s[0] * (1 - pull) + lgcx * pull
                s[1] = s[1] * (1 - pull) + lgcy * pull
                self.kf.statePost = s.reshape(-1, 1)
            # 高速(烟在移动) → 不回拉, 靠速度外推继续走
        p = self.kf.predict().flatten()
        self.bbox = (float(p[0]-p[2]/2), float(p[1]-p[3]/2), float(p[0]+p[2]/2), float(p[1]+p[3]/2))
        # 显示平滑: 丢失期间显示框缓慢跟随预测(不跳变)
        alpha = 0.50 if self.lost > 0 else 0.85  # ★预测期显示紧跟
        d = self.disp_bbox
        self.disp_bbox = (
            d[0]*(1-alpha) + self.bbox[0]*alpha, d[1]*(1-alpha) + self.bbox[1]*alpha,
            d[2]*(1-alpha) + self.bbox[2]*alpha, d[3]*(1-alpha) + self.bbox[3]*alpha)
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
        s[4] = float(np.clip(s[4], -45, 45)); s[5] = float(np.clip(s[5], -45, 45))  # ★限幅±45(快速平移不截断)
        self.kf.statePost = s.reshape(-1, 1)
        p = s
        self.bbox = (float(p[0]-p[2]/2), float(p[1]-p[3]/2), float(p[0]+p[2]/2), float(p[1]+p[3]/2))
        # 显示平滑: EMA 混合检测框与旧显示框, 消除快速移动后的瞬移
        # 丢失重获时瞬移最大 → 检测置信度低(lost>0刚恢复)时用更重平滑
        alpha = 0.50 if self.lost > 0 else 0.85  # ★显示紧跟检测(不滞后)
        d = self.disp_bbox
        self.disp_bbox = (
            d[0]*(1-alpha) + self.bbox[0]*alpha, d[1]*(1-alpha) + self.bbox[1]*alpha,
            d[2]*(1-alpha) + self.bbox[2]*alpha, d[3]*(1-alpha) + self.bbox[3]*alpha)
        self.last_good = bbox   # 更新最后确认位置
        self.candidates = []   # 恢复关联后清空候选
    def update_sig(self, new_sig, max_pool=6):
        """更新外观签名: 多帧累积(0.7旧+0.3新) + 历史池记录代表外观
        注意: sig是dict(hist数组+w/h), 平滑只作用于hist; 尺寸取最近观测
        """
        if self.sig is None:
            self.sig = new_sig
        else:
            # 直方图平滑累积(数组可加权), 尺寸/宽高取最近值
            self.sig['hist'] = self.sig['hist'] * 0.7 + new_sig['hist'] * 0.3
            self.sig['w'] = new_sig['w']
            self.sig['h'] = new_sig['h']
        self.sig_pool.append(dict(new_sig))   # 深拷贝: 防外部引用篡改
        if len(self.sig_pool) > max_pool:
            self.sig_pool.pop(0)

def iou(a, b):
    ix1 = max(a[0], b[0]); iy1 = max(a[1], b[1]); ix2 = min(a[2], b[2]); iy2 = min(a[3], b[3])
    iw = max(0, ix2-ix1); ih = max(0, iy2-iy1)
    aa = (a[2]-a[0])*(a[3]-a[1]); bb = (b[2]-b[0])*(b[3]-b[1])
    return iw*ih/(aa+bb-iw*ih+1e-6)

tracks = []; smoke_tracks = []; hand_tracks = []; next_id = 0   # ★ hand_tracks: 手部追踪(蓝色框)
import os
AUTO_SAVE = r'D:\training_data\smoke\fp_auto'   # 误检帧自动采集(检出烟时保存)
os.makedirs(AUTO_SAVE, exist_ok=True)
last_auto_save = 0

# ★ 长记忆已取消(用户17:19): 删除 smoke_history/low_suspects — 检测框只来自本帧
while True:
    ret, frame = cap.read()
    if not ret or frame is None:
        # ★ 轻量重试(治"一会识别一会丢失"): 偶发抓帧失败(MSMF抖动/短暂占用)
        #   先快速重读3次(50ms级), 成功即继续, 不触发重型重连(避免卡1秒中断检测)
        retry_ok = False
        for _r in range(3):
            time.sleep(0.05)
            ret, frame = cap.read()
            if ret and frame is not None and frame.size > 0:
                retry_ok = True; break
        if not retry_ok:
            # 连续失败 → 重型重连(摄像头被占用/驱动卡死)
            print('[重连] 摄像头持续断帧, 尝试重连...')
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
    for ht in hand_tracks: ht[0].predict()   # ★ 手部追踪预测(新架构)

    # ★★ 光线自适应暂停使用(用户要求 11:35): 全部关闭但代码保留, 可随时恢复
    #   恢复方法: 取消下行注释, 改为 frame_enh = light_adapt(frame)
    #   暂停原因: 光线锐化/过滤后香烟细节纹理不够明显, 先回退原帧对比
    #   影响: 检测用原帧(无CLAHE/Unsharp/分块补偿), 暗光/白背景增强全部失效
    frame_enh = frame

    # --- V7 头检测 (原始帧, conf=0.40 提升: 防手掌/圆形物被误认为头部) ---
    # 用户反馈(14:05): 手掌有时被识别成头部 → conf 0.30→0.40(更严格)
    # 手部有独立 hand.pt 检测(蓝色框), 头模型不再需要低conf容忍手掌
    res = HEAD(frame, conf=0.55, iou=0.5, verbose=False)
    det_heads = []
    for r in res:
        for box in r.boxes:
            b = box.xyxy[0].cpu().numpy()
            x1, y1, x2, y2 = float(b[0]), float(b[1]), float(b[2]), float(b[3])
            bw, bh = x2-x1, y2-y1
            if bw < 10 or bh < 10: continue          # 极小下限(原30, 放宽)
            ar = bw/bh if bh > 0 else 0
            if bw < 60:                               # 小/中头: 宽高比检查
                if ar < 0.5 or ar > 1.8: continue
            # 大头(≥60px): 不限制宽高比(贴镜头视角变形) — 取消限制
            # 底部过滤: 完全去掉(贴镜头/仰角都识别)
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
    tracks = [t for t in tracks if t[0].lost < 5]   # 头轨迹容忍(昨天配置, 恢复)

    # --- ★★ 手部检测+追踪(新架构): 手掌蓝色框, 框随手掌变化动态更新 ---
    #   hand.pt 由 COCO 手部数据训练; 与头部同样用 KalmanBox 追踪, 流畅稳定
    det_hands = []
    if HAND_READY:
        try:
            res_h = HAND(frame, conf=0.30, iou=0.5, verbose=False)
            for r in res_h:
                for box in r.boxes:
                    b = box.xyxy[0].cpu().numpy()
                    x1, y1, x2, y2 = float(b[0]), float(b[1]), float(b[2]), float(b[3])
                    bw, bh = x2-x1, y2-y1
                    if bw < 20 or bh < 20: continue   # 最小手框(过滤噪声)
                    if bw > W*0.7 or bh > H*0.7: continue  # 超大框(误检)
                    # ★★ 肤色验证(治椅子/木纹/红色物体被误判为手) v2:
                    #   人手: 肤色占比高(≥25%) + 高饱和红木占比低(皮肤S 90-140, 红木S>150)
                    #   椅背红木: 纯红木S>150占比高(暖光下V也高会命中肤色) → 用高饱和红排除
                    _hsx1, _hsy1 = max(0,int(x1)), max(0,int(y1))
                    _hsx2, _hsy2 = min(W,int(x2)), min(H,int(y2))
                    if _hsx2 - _hsx1 >= 8 and _hsy2 - _hsy1 >= 8:
                        _hcrop = frame[_hsy1:_hsy2, _hsx1:_hsx2]
                        _hhsv = cv2.cvtColor(_hcrop, cv2.COLOR_BGR2HSV)
                        _hpix = _hcrop.shape[0] * _hcrop.shape[1]
                        _hskin = float(np.sum(cv2.inRange(_hhsv, (0,25,50), (45,180,255)) > 0)) / _hpix
                        if _hskin < 0.25:
                            continue   # 框内肤色<25% → 非人手(椅子/木纹) → 拒
                        # 高饱和红木排除: 红/棕(H 0-30) + S>150(极饱和纯红木, 肤色S一般<150)
                        _hsat_red = float(np.sum(cv2.inRange(_hhsv, (0,150,40), (30,255,255)) > 0)) / _hpix
                        if _hsat_red > 0.45:   # 阈值0.45: 红木2(0.56)拒, 深肤色手(0.45)过
                            continue   # 极饱和红棕占比>35% → 红木椅背/木制品 → 拒
                    det_hands.append((x1, y1, x2, y2))
        except Exception:
            pass
    # ★★ 手部检测去重(治"方框分裂/重叠"): 同一只手被模型输出多个框(微小偏移/双检)
    #   按面积降序, IoU>0.3 或中心距<手宽×0.8 → 保留大框去重小框
    det_hands_dedup = []
    for _dh in sorted(det_hands, key=lambda b: -(b[2]-b[0])*(b[3]-b[1])):
        _dup = False
        for _dh2 in det_hands_dedup:
            if iou(_dh, _dh2) > 0.3:
                _dup = True; break
            _d = ((_dh[0]+_dh[2])/2 - (_dh2[0]+_dh2[2])/2)**2 + ((_dh[1]+_dh[3])/2 - (_dh2[1]+_dh2[3])/2)**2
            if _d < ((_dh2[2]-_dh2[0]) * 0.8) ** 2:
                _dup = True; break
        if not _dup:
            det_hands_dedup.append(_dh)
    det_hands = det_hands_dedup
    # 手部关联追踪(与头部同逻辑: IoU匹配 + Kalman预测)
    h_matched_ids = set(); h_matched_det = set()
    for j, dh in enumerate(det_hands):
        best_i, best_iou = -1, 0.15
        for i, ht in enumerate(hand_tracks):
            if ht[1] in h_matched_ids: continue
            iou_val = iou(ht[0].bbox, dh)
            if iou_val < 0.15:
                kf = ht[0].kf; ps = kf.statePre.flatten()
                pb = (ps[0]-ps[2]/2, ps[1]-ps[3]/2, ps[0]+ps[2]/2, ps[1]+ps[3]/2)
                iou_val = iou(pb, dh)
            if iou_val > best_iou: best_iou = iou_val; best_i = i
        if best_i >= 0:
            hand_tracks[best_i][0].update(dh)
            h_matched_ids.add(hand_tracks[best_i][1]); h_matched_det.add(j)
    for ht in hand_tracks:
        if ht[1] not in h_matched_ids: ht[0].lost += 1
    for j, dh in enumerate(det_hands):
        if j not in h_matched_det and len(hand_tracks) < 10:
            hand_tracks.append([KalmanBox(dh, pn=0.50, mn=0.05), next_id + 1000]); next_id += 1
    hand_tracks = [ht for ht in hand_tracks if ht[0].lost < 5]

    # --- V22 全屏烟检 + 智能过滤(距离机制) + ByteTrack低分池 ---
    raw_smoke = []
    low_pool = []   # ByteTrack 低分候选池 (被conf过滤但≥0.15, 仅用于维持已有轨迹)
    head_list = [(t[0].bbox, t[0].bbox[2]-t[0].bbox[0]) for t in tracks
                 if (t[0].bbox[2]-t[0].bbox[0]) > 0 and (t[0].bbox[3]-t[0].bbox[1]) > 0]
    # ★ 强光源mask预计算(每帧1次): 原帧灰度>235 = 过曝白光区(灯具/强光源/光晕)
    #   真烟不会出现在纯过曝白光区, 检测框内过曝占比高 → 直接拒(治强光源误检)
    _over_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _over_mask = _over_gray > 235
    # ★★ 新架构(用户要求): 香烟检测注意力只放在"头部+手部"
    #   在头框/手框的扩张区域内找烟(烟常在嘴边/手持/耳后), 大幅减少全屏误检(椅子/门框/桌腿)
    #   头框扩张: 上下左右各扩(脸前/嘴前/耳后) | 手框扩张: 更大(手持烟/挥烟)
    #   ★ 用户要求(14:14): 严格 — focus_regions 改用【本帧原始检测det_heads/det_hands】
    #   不用 tracks/hand_tracks(它们含lost<5的历史轨迹, 检测不到时仍"虚假存在" → 误判烟)
    #   本帧没检测到 → focus_regions空 → 烟识别关闭(符合"无头无手绝不识别")
    focus_regions = []
    for (hx1, hy1, hx2, hy2) in det_heads:           # 本帧头检测(不含历史)
        hw_, hh_ = hx2-hx1, hy2-hy1
        focus_regions.append((max(0, hx1-hw_*0.6), max(0, hy1-hh_*0.9),
                              min(W, hx2+hw_*0.6), min(H, hy2+hh_*0.6)))
    for (hx1, hy1, hx2, hy2) in det_hands:            # 本帧手检测(不含历史)
        hw_, hh_ = hx2-hx1, hy2-hy1
        focus_regions.append((max(0, hx1-hw_*1.2), max(0, hy1-hh_*0.6),
                              min(W, hx2+hw_*1.2), min(H, hy2+hh_*1.6)))
    # 无头无手: focus_regions为空 → 本帧不做烟检测(严格限定区域, 不识别区域外)
    # ★★ 新架构(用户需求): ROI 裁剪放大独立检测 — "放大方框看附近小区域有没有香烟"
    #   头框/手框出现 → 裁剪扩张区域ROI → 放大到640 → 独立跑SMOKE → 框坐标换算回全图
    #   相比"全图1024+中心过滤": 区域小图被放大, 有效分辨率翻倍, 小烟/被挡一半的烟更好检出
    #   区域上限4个(多人场景防性能下降); 无头无手 → sr空 → 烟识别关闭
    sr = []
    if focus_regions:
        _roi_budget = 4
        for (frx1, fry1, frx2, fry2) in focus_regions:
            if _roi_budget <= 0: break
            # ★ 坐标转int再切片(模型输出float, numpy切片必须整数)
            _frx1, _fry1 = max(0, int(frx1)), max(0, int(fry1))
            _frx2, _fry2 = min(W, int(frx2)), min(H, int(fry2))
            _frw, _frh = _frx2-_frx1, _fry2-_fry1
            if _frw < 24 or _frh < 24: continue
            _roi = frame_enh[_fry1:_fry2, _frx1:_frx2]
            _scale = 640.0 / max(_frw, _frh)      # 放大系数(区域小→放大倍数大)
            _tw, _th = int(_frw*_scale+0.5), int(_frh*_scale+0.5)
            if _tw < 16 or _th < 16: continue
            try:
                _roi_big = cv2.resize(_roi, (_tw, _th), interpolation=cv2.INTER_LINEAR)
                _rr = SMOKE(_roi_big, conf=0.06, iou=0.5, imgsz=640, verbose=False)
                for _rrc in _rr:
                    for _b2 in _rrc.boxes:
                        _bc = float(_b2.conf[0])
                        _bx1, _by1, _bx2, _by2 = _b2.xyxy[0].cpu().numpy()
                        # 放大图坐标 → 换算回全图坐标(以_int裁剪原点为基准)
                        _ox1 = _frx1 + _bx1/_scale; _oy1 = _fry1 + _by1/_scale
                        _ox2 = _frx1 + _bx2/_scale; _oy2 = _fry1 + _by2/_scale
                        sr.append(_RoiRes([_RoiBox(_ox1, _oy1, _ox2, _oy2, _bc)]))
                _roi_budget -= 1
            except Exception:
                pass
    else:
        sr = []   # 无头无手 → 本帧不检测烟
    for r in sr:
        for box in r.boxes:
            conf = float(box.conf[0])
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            w, h = x2 - x1, y2 - y1
            if min(w, h) < 3: continue
            area_ratio = (w * h) / (W * H)
            aspect = max(w, h) / min(w, h)
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2

            # ===== ★★ 区域聚焦(严格版, 用户强调): 候选中心必须在头/手聚焦区域内 =====
            # 香烟识别功能只存在于"头框/手框出现后"的框内及其附近
            # 区域外一律拒(即使候选是烟/即使靠近已确认烟轨) — "装没看见"
            _in_focus = False
            for (frx1, fry1, frx2, fry2) in focus_regions:
                if frx1 <= cx <= frx2 and fry1 <= cy <= fry2:
                    _in_focus = True; break
            if not _in_focus:
                continue   # ★ 严格: 不在头/手区域 → 一律拒(无任何豁免)

            # ===== ★★★ 必须与蓝框(手)/绿框(头)有交集(用户17:07要求): 否则不算烟 =====
            #   替代原"接触手"约束(v3): 红框必须与蓝框或绿框有交集(IoU>0)
            #   ① 嘴前/嘴边烟: 在头框(绿)内 → 有交集 → 放行(修复"头框内烟检测不到")
            #   ② 手持烟: 在手框(蓝)内/重叠 → 放行
            #   ③ 窗格栅/门框/桌面/远处烟: 与任何头/手框无交集 → 拒
            _touch_hand = False; _touch_head = False
            for (_tbx1, _tby1, _tbx2, _tby2) in det_hands:
                if iou((x1, y1, x2, y2), (_tbx1, _tby1, _tbx2, _tby2)) > 0.0:
                    _touch_hand = True; break
            for (_tbx1, _tby1, _tbx2, _tby2) in det_heads:
                if iou((x1, y1, x2, y2), (_tbx1, _tby1, _tbx2, _tby2)) > 0.0:
                    _touch_head = True; break
            if not (_touch_hand or _touch_head):
                continue   # ★ 严格: 红框未与蓝/绿框有交集 → 一律拒(不管多像烟)

            # ===== 强光源直接过滤: 整框几乎全过曝 → 直接拒 =====
            # 治"强光源/灯具/过曝区被误判为烟"
            # 双保险: 过曝占比>25% 且 暗区(<150)占比<15% = 纯光晕/纯白光区 → 拒
            #   (真烟被强光直射时烟身过曝但背景/手指暗区多 → 不误杀)
            if area_ratio > 0.002:
                ox1, oy1 = max(0, int(x1)), max(0, int(y1))
                ox2, oy2 = min(W, int(x2)), min(H, int(y2))
                if ox2 - ox1 >= 4 and oy2 - oy1 >= 4:
                    _roi = _over_mask[oy1:oy2, ox1:ox2]
                    if float(_roi.mean()) > 0.25:
                        _rg = _over_gray[oy1:oy2, ox1:ox2]
                        if float((_rg < 150).mean()) < 0.15:
                            continue   # 整框几乎全过曝(纯光晕/强光源) → 拒

            # ===== ★ 低纹理细长拒(用户实测: 椅子边/门框/桌腿误检) =====
            # 用户截图(11:59): 室内椅子上误出红框2个 — V28把"细长棕色纯色物体"当烟
            # 区分: 椅子边/门框/桌腿 = "纯色均匀, Laplacian 方差极低(<10)"
            #       烟纸纤维纹理丰富, Laplacian 方差 > 20
            # 光线暂停后无CLAHE, 几何防线挡不住纯色细长, 必须用纹理密度判定
            # 注: 白背景烟(像素少)lap可能低, 用 aspect 联合 + 面积下限避免误伤
            if area_ratio < 0.05 and aspect >= 2.0:
                _qcrop = frame[max(0,int(y1)):min(H,int(y2)), max(0,int(x1)):min(W,int(x2))]
                if _qcrop.size >= 16:
                    _qg = cv2.cvtColor(_qcrop, cv2.COLOR_BGR2GRAY)
                    _lap_var = cv2.Laplacian(_qg, cv2.CV_64F).var()
                    # 纯色均匀细长物体(椅子边/门框) → 拒
                    if _lap_var < 8.0:
                        continue   # 低纹理纯色细长 = 椅子边/门框/桌腿 → 拒

            # ===== ★ 纯形状识别模式(用户要求): 停用材质关+颜色关 =====
            # 光线照射/远距会使材质(纹理/反光)与颜色(滤嘴棕/烟纸白)判断不可靠
            # → 回归最稳妥: 只用图形特征(细长条 aspect + 距离尺寸一致性)识别
            # conf 保持模型原始值(不靠颜色加权), 弱信号由形状关+尺寸校验兜底

            # ===== 距离机制: 头框宽 → 距离分级 → 阈值 + 尺寸一致性校验 =====
            nearest_hw = None
            best_d2 = 1e18
            nearest_hw_dist = 1e18   # ★ 烟到最近人头中心的距离(px)
            for hb, hw0 in head_list:
                hcx, hcy = (hb[0]+hb[2])/2, (hb[1]+hb[3])/2
                d2 = (cx-hcx)**2 + (cy-hcy)**2
                if d2 < best_d2:
                    best_d2 = d2; nearest_hw = hw0
                    nearest_hw_dist = d2 ** 0.5
            if nearest_hw is not None:
                hw = nearest_hw
                if hw >= 150: dt = 0.06   # 近距
                elif hw >= 60: dt = 0.06
                else: dt = 0.06            # 远距(统一0.06)
                if conf < dt:
                    # ★★ 疑似目标 2x 放大复核(用户要求): 低置信但形状像烟的候选
                    #   原理: 弱信号烟在远/小/模糊时 conf<0.06, 但放大2倍后细节更清晰,
                    #   模型 conf 提升 → 复核通过放行(治"漏识别"); 纯噪声放大后仍低 → 拒
                    _recheck_ok = False
                    if conf >= 0.03 and 2.0 <= aspect <= 5.5:   # 只复核"有信号+像烟"的候选
                        _rx1 = max(0, int(x1) - int(w)); _ry1 = max(0, int(y1) - int(h))
                        _rx2 = min(W, int(x2) + int(w)); _ry2 = min(H, int(y2) + int(h))
                        if _rx2 - _rx1 >= 8 and _ry2 - _ry1 >= 8:
                            _roi = frame_enh[_ry1:_ry2, _rx1:_rx2]
                            if _roi.size > 0:
                                try:
                                    _rr = SMOKE(_roi, conf=0.10, iou=0.5, imgsz=480, verbose=False)
                                    for _rrc in _rr:
                                        for _b2 in _rrc.boxes:
                                            if float(_b2.conf[0]) >= 0.18:   # 放大后确认烟
                                                _recheck_ok = True; break
                                        if _recheck_ok: break
                                except Exception:
                                    pass
                    if not _recheck_ok:
                        # 复核失败 → 低分池(供遮挡恢复) 或 直接拒
                        if conf >= 0.05 and area_ratio < 0.05 and aspect >= 2.0 and aspect <= 5.5:
                            low_pool.append((float(x1), float(y1), float(x2), float(y2)))
                        continue
                    # ★ 复核通过: 放大后确认是烟 → 提升 conf 放行(继续走形状/肤色把关)
                    conf = max(conf, 0.18)
                # ★ 尺寸一致性校验 — 只对"烟靠近人头"生效(烟-头距离 < 头宽×3)
                #   白墙区烟离人头远(>头宽×3): 跳过尺寸校验, 靠形状关+面积兜底
                #   (修复: 画面中央人头近+白墙区烟远 → 头宽比例失衡误杀白墙烟)
                if nearest_hw_dist < hw * 3.0:
                    if hw < 60:                       # 远距严格校验
                        if w > hw * 0.12: continue     # 框宽>烟应有宽度2.4倍 → 胳膊/烟盒/手机
                        if w < hw * 0.01: continue     # 远距太窄 → 噪点
                    elif hw < 150:                    # 中距: 放宽(手持烟框含手指)
                        if w > hw * 0.22: continue     # 超0.22 → 胳膊/烟盒
                # else: 烟远离人头 → 不做尺寸约束(形状关+面积上限兜底)
            else:
                # ★ 无头部参考时, 仍让弱信号进入过滤链(治白墙+远距小烟漏检)
                #   原 conf<0.40 直接continue → 白背景烟 conf 0.14-0.17 直接被拒
                #   现与 dt 一致: conf<0.06 才拒, 靠"形状关+面积"几何兜底防误检
                if conf < 0.06:
                    if conf >= 0.04 and area_ratio < 0.08 and aspect >= 1.8 and aspect <= 6.0:
                        low_pool.append((float(x1), float(y1), float(x2), float(y2)))
                    continue           # conf 太低才拒绝

            # ===== 形状关(距离感知): 真烟细长条, 防大面积/极细长异常 =====
            # ★ 近距(hw≥150)放宽: 手持烟框内含手, 框更大/宽高比更低 → 不能按远距标准误杀
            #   远距(纯烟条): 细长小框, 严格; 近距(手+烟): 允许更大/更宽
            is_near = nearest_hw is not None and nearest_hw >= 150
            if conf < 0.25 and aspect < (1.0 if is_near else 1.3): continue  # 近距允许近方形(手含烟)
            if aspect > 6.0: continue                   # 极细长(窗帘褶/线)
            if area_ratio > (0.18 if is_near else 0.08): continue  # 近距框含手, 面积上限放宽到18%
            # ★ 无头参考时绝对宽度兜底: 远处胳膊/手宽(>40px)而烟窄(<30px)
            #   之前尺寸校验被"远离人头跳过"后, 胳膊全放行 → 加绝对宽度上限
            if nearest_hw is None and w > 40:
                continue   # 无头参考 + 框宽>40px = 胳膊/手/物体 → 拒

            # ===== D规则豁免: 已确认烟轨附近 → 跳过剩余过滤 =====
            # 真烟已连续确认, 位置吻合的框大概率还是它(即使近距手指入框肤色高)
            # "追踪一旦确认, 规则让位给追踪" → 根治近距误杀
            near_confirmed = False
            for stt in smoke_tracks:
                if stt[0].confirmed < 2: continue
                tbb = stt[0].bbox
                tcx2, tcy2 = (tbb[0]+tbb[2])/2, (tbb[1]+tbb[3])/2
                if ((cx-tcx2)**2 + (cy-tcy2)**2) ** 0.5 < max(60.0, (tbb[2]-tbb[0])*3.0):
                    near_confirmed = True; break

            # ===== 肤色主导拒(所有框生效, 不豁免已确认轨!) =====
            # ★ 根治"手指误建轨后永远豁免": 远距手指肤色0.74-0.89(实测)
            #   真烟肤色<0.40(白纸+滤嘴稀释) → 肤色主导拒对真烟无害
            #   放在D规则豁免之前, 即使已确认烟轨也检查(手指不该被追踪豁免)
            xi1, yi1 = max(0, int(x1)), max(0, int(y1))
            xi2, yi2 = min(W, int(x2)), min(H, int(y2))
            if xi2 - xi1 >= 6 and yi2 - yi1 >= 6:
                _crop = frame[yi1:yi2, xi1:xi2]
                _hsv = cv2.cvtColor(_crop, cv2.COLOR_BGR2HSV)
                _pix = _crop.shape[0] * _crop.shape[1]
                _skin = np.sum(cv2.inRange(_hsv, (0,25,50), (45,180,255)) > 0) / _pix
                if _skin > 0.55:
                    # ★ 口边烟豁免(用户17:07): 与头框交集+细长(aspect≥2.0) = 嘴上烟(含唇色) → 放行
                    if not (_touch_head and aspect >= 2.0):
                        continue   # 肤色主导(>55%) = 手/胳膊/脸 → 拒(不管形状, 不豁免!)

            if not near_confirmed:
                # ===== 手指组合拒(仅新目标): 肤色较高+粗短 =====
                # 实测均值: 手指 aspect≈1.9粗短 肤色≈0.35 | 香烟 aspect≈2.4细长
                # 只对"新目标"检查(D规则豁免已确认烟轨)
                if xi2 - xi1 >= 6 and yi2 - yi1 >= 6:
                    if _skin > 0.30 and aspect < 2.2 and not (_touch_head and aspect >= 2.0):
                        continue   # 手指(肤色较高 + 粗短 aspect<2.2) → 拒(头框内细长=口边烟豁免)

            # (材质关/颜色关已停用 — 纯形状识别模式, D规则豁免仅预留)

            # ===== 头框重叠 + 头附近区域: 中心在头框内/下方的极小框 = 脸上特征(鼻/嘴/下巴/喉结) =====
            # ★ 已确认烟轨豁免: 烟移到嘴前/耳后(头框内)是正常吸烟姿态, 不能被脸上特征过滤误杀
            # (这就是"烟移到屏幕中间人脸处→框卡住不动"的根因, 现修复)
            if not near_confirmed:
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
                    #   细长(aspect≥2.0) = 耳后/口边烟 → 放行(口边烟框含唇 aspect常2.0-2.5)
                    #   非细长 = 脸上特征(鼻/嘴/眼) → 丢
                    if hx1 <= cx <= hx2 and hy1 <= cy <= hy2:
                        if (w * h) / h_area < 0.05 and aspect < 2.0:
                            in_face = True; break
                if in_face: continue
            # 兜底: 无头框时, 大而圆的检测框(占画面3%+ 且 近圆形) = 头/大圆物 → 丢
            if area_ratio > 0.03 and aspect < 1.5:
                continue

            # ===== 电子设备拒(用户17:17): 手机/耳机被误判为烟 =====
            #   拿在手里/戴在耳边会通过"交集"判定, 用屏幕/机身特征区分:
            #   ① 手机亮屏: 白亮像素多 + 非细长 + 面积>0.4% → 拒 (烟白但细长, 不触发)
            #   ② 彩色屏幕(壁纸/内容): 高饱和彩像素>50% → 拒 (烟滤嘴棕色占比小)
            #   ③ 黑屏手机/黑耳机: 黑色像素>55% → 拒 (烟纸白, 黑占比极低)
            if xi2 - xi1 >= 6 and yi2 - yi1 >= 6:
                _ehsv = cv2.cvtColor(frame[yi1:yi2, xi1:xi2], cv2.COLOR_BGR2HSV)
                _epix = max(1, (yi2-yi1) * (xi2-xi1))
                _ew = float(np.sum(cv2.inRange(_ehsv, (0,0,180), (180,60,255)) > 0)) / _epix
                _eb = float(np.sum(cv2.inRange(_ehsv, (0,0,0), (180,255,70)) > 0)) / _epix
                _ec = float(np.sum(cv2.inRange(_ehsv, (0,80,80), (180,255,255)) > 0)) / _epix
                if area_ratio > 0.004 and _ew > 0.45 and aspect < 3.0:
                    continue   # 大片亮白 + 非细长 = 手机亮屏 → 拒
                if _ec > 0.50:
                    continue   # 高饱和彩主导 = 彩色屏幕/彩色耳机 → 拒
                if _eb > 0.55:
                    continue   # 黑色主导 = 黑屏手机/黑耳机 → 拒

            raw_smoke.append((float(x1), float(y1), float(x2), float(y2)))

    # ★ 合并重叠检测框(方向感知): 同一烟分裂的框合成一个
    #   烟是细长条, 一根烟被检测成两段时: 重叠区小(iou常<0.15-0.3) 且 中心距大(>40px)
    #   旧阈值(0.3/40px)太严 → 分裂框不合并 → 多框重叠!
    #   方向感知: 竖直烟看横向对齐(dx<较小宽×2.5) + 沿纵向距离<高×1.0
    #             水平烟对称; 旁边小物短边偏移大 → 不误合并
    merged = []
    for sb in sorted(raw_smoke, key=lambda b: -(b[2]-b[0])*(b[3]-b[1])):
        dup = False
        scx, scy = (sb[0]+sb[2])/2, (sb[1]+sb[3])/2
        for mb in merged:
            if iou(sb, mb) > 0.15:
                dup = True; break
            mcx, mcy = (mb[0]+mb[2])/2, (mb[1]+mb[3])/2
            mw = max(sb[2]-sb[0], mb[2]-mb[0])
            mh = max(sb[3]-sb[1], mb[3]-mb[1])
            sw = min(sb[2]-sb[0], mb[2]-mb[0])
            dx = abs(scx-mcx); dy = abs(scy-mcy)
            dist = (dx**2 + dy**2) ** 0.5
            if mh >= mw:
                # 竖直烟: 横向对齐 + 沿纵向距离<高×1.0
                if dx < max(15.0, sw*2.5) and dist < max(60.0, mh*1.0):
                    dup = True; break
            else:
                # 水平烟: 纵向对齐 + 沿横向距离<宽×1.0
                if dy < max(15.0, sw*2.5) and dist < max(60.0, mw*1.0):
                    dup = True; break
        if not dup:
            merged.append(sb)
    raw_smoke = merged

    # ★★ 长记忆已取消(用户17:19): 删除历史框快照(smoke_history)/低分池融合
    #   检测框只来自本帧 raw_smoke, 无跨帧历史 → 丢失立即删框

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
            # C速度外推双窗口: Kalman速度预测下一帧位置, 快移动接"速度窗"
            kf_ps = st[0].kf.statePost.flatten()
            vx, vy = float(kf_ps[4]), float(kf_ps[5])
            px, py = tcx + vx, tcy + vy          # 速度外推预测中心
            dist_vel = ((scx-px)**2 + (scy-py)**2) ** 0.5
            # 综合评分: IoU优先; 移动快时IoU低但中心近 → 距离补偿(120px)
            score = iou_val
            if iou_val < 0.15 and dist < 120:
                score = 0.15 + 0.15 * (1 - dist / 120)
            # 速度窗补偿: 快移时用速度外推距离评分(比位置窗更准)
            if iou_val < 0.15 and dist_vel < 130:  # ★速度窗130(快移可关联)
                s2 = 0.15 + 0.25 * (1 - dist_vel / 130)  # ★快移评分提升
                if s2 > score: score = s2
            if score > best_score:
                best_score = score; best_i = i
        if best_i >= 0 and best_score >= 0.15:
            smoke_tracks[best_i][0].update(sb); sm_matched.add(smoke_tracks[best_i][1]); sm_det.add(j)
            # 长期追踪: 采集/更新外观签名 (用原帧, 颜色准确)
            sig = extract_sig(frame, sb)
            if sig is not None:
                smoke_tracks[best_i][0].update_sig(sig)
    # ★★ 长记忆已取消(用户17:19): 删除Level2低分匹配/长期找回/低分建轨
    #   检测到就显示, 丢失(lost>=1)立即删框, 无任何历史/预测残留
    # 未匹配轨迹: lost+1 (下帧清理时 lost>=1 立即删除)
    for st in smoke_tracks:
        if st[1] not in sm_matched: st[0].lost += 1
    # ===== (已删除: 长期追踪找回 try_recover_smoke — 长记忆取消) =====
    # ===== (已删除: 低分稳定框建轨 low_suspects — 长记忆取消) =====
    # 新轨迹: 仅高分框创建 (低分池不建新轨, 防误检污染); 烟用快跟随Kalman(减少漂移)
    for j, sb in enumerate(raw_smoke):
        if j not in sm_det and len(smoke_tracks) < 6:
            _sig1 = extract_sig(frame, sb)
            smoke_tracks.append([KalmanBox(sb, pn=0.80, mn=0.05, sig=_sig1), next_id + 5000]); next_id += 1
    for st in smoke_tracks:
        # 漂移抑制: 预测框出画面直接淘汰
        bx1, by1, bx2, by2 = st[0].bbox
        if bx2 < 0 or by2 < 0 or bx1 > W or by1 > H:
            st[0].lost = 99
    # ★ 防闪缓冲(用户17:30): 连续丢3帧才删轨(滤单帧检测波动), 真消失0.15s内清除
    smoke_tracks = [st for st in smoke_tracks if st[0].lost < 3]

    # 误检帧自动采集: 有确认烟轨时保存整帧(限频), 供筛选负样本
    if smoke_tracks and time.time() - last_auto_save >= 0.5:
        last_auto_save = time.time()
        cv2.imwrite(os.path.join(AUTO_SAVE, f'auto_{int(last_auto_save)}.jpg'), frame)

    # --- 显示帧准备: ★★ 光线过滤暂停(用户要求 11:35): 直接显示原帧, 不压暗 ---
    #   恢复方法: 取消下行注释, 改回 _disp = cv2.convertScaleAbs(frame_enh, alpha=0.62, beta=0)
    _disp = frame_enh.copy()

    # --- 绘制(在 _disp 上, 覆盖涂暗层, 标注框颜色保持鲜艳) ---
    # ★★ 手部蓝色框(新架构): 追踪框随手掌大小动态变化(Kalman更新)
    for ht in hand_tracks:
        hx1, hy1, hx2, hy2 = map(int, ht[0].disp_bbox if hasattr(ht[0], 'disp_bbox') else ht[0].bbox)
        cv2.rectangle(_disp, (hx1,hy1), (hx2,hy2), (255,0,0), 2)   # 蓝色
        cv2.putText(_disp, "HAND", (hx1, hy1-6), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,0,0), 1)

    for t in tracks:
        x1, y1, x2, y2 = map(int, t[0].bbox)
        cv2.rectangle(_disp, (x1,y1), (x2,y2), GREEN, 2)
        lb = f"#{t[1]}"
        (tw,th),_ = cv2.getTextSize(lb, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)
        cv2.rectangle(_disp, (x1,y1-14), (x1+tw+4,y1), GREEN, -1)
        cv2.putText(_disp, lb, (x1+2,y1-3), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,0), 1)

    for st in smoke_tracks:
        # ★★ 严格区域(用户强调): 烟轨中心必须在头/手聚焦区域内才显示, 否则"装没看见"
        #   即使香烟就在镜头中, 不在头/手框内 → 不显示(区域外禁止识别功能)
        _scx = (st[0].disp_bbox[0]+st[0].disp_bbox[2])/2
        _scy = (st[0].disp_bbox[1]+st[0].disp_bbox[3])/2
        _s_in = False
        for (frx1, fry1, frx2, fry2) in focus_regions:
            if frx1 <= _scx <= frx2 and fry1 <= _scy <= fry2:
                _s_in = True; break
        if not _s_in:
            continue   # 不在头/手区域 → 不显示(装没看见)
        # ★★ 红框必须与蓝框/绿框有交集(用户17:07要求): 一脱离立即不显示, 无缓冲
        _touch_now = False
        _dbb = st[0].disp_bbox
        for (_tbx1, _tby1, _tbx2, _tby2) in list(det_hands) + list(det_heads):
            if iou(_dbb, (_tbx1, _tby1, _tbx2, _tby2)) > 0.0:
                _touch_now = True; break
        if not _touch_now:
            continue   # ★ 红框已脱离蓝/绿框 → 立即消失(不显示预测框)
        # 显示: 稳定轨全显; 新轨(lost<3)也显(减少"刚出现就消失"的闪烁)
        sw, sh = st[0].disp_bbox[2]-st[0].disp_bbox[0], st[0].disp_bbox[3]-st[0].disp_bbox[1]
        # ★★ 防闪缓冲(用户17:30): 丢失<3帧(单帧检测波动)保持显示, 连续丢3帧(~0.15s)才消失
        #   解决"一闪一闪": 每帧检测conf波动导致偶发漏检, 零缓冲会直接闪; 3帧缓冲滤掉波动
        if st[0].lost >= 3:
            continue
        if (sw*sh)/(W*H) < 0.01 and st[0].confirmed < 1 and st[0].lost >= 1:
            continue
        sx1, sy1, sx2, sy2 = map(int, st[0].disp_bbox)
        if st[0].lost > 0:
            cv2.rectangle(_disp, (sx1,sy1), (sx2,sy2), (0,165,255), 2)
            cv2.putText(_disp, "CIG?", (sx1, sy2+18), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,165,255), 1)
        else:
            cv2.rectangle(_disp, (sx1,sy1), (sx2,sy2), RED, 3)
            cv2.putText(_disp, "CIG", (sx1, sy2+18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, RED, 2)

    # 状态行(画在 _disp 上, 覆盖涂暗)
    cv2.putText(_disp, f"H:{len(tracks)} C:{len(smoke_tracks)} {fc/max(1,time.time()-t0):.0f}fps",
                (4,18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, GREEN, 2)

    # ★ 显示 960x540(原640x360): 提高视频分辨率, 画面更清晰(处理能力有余裕)
    _disp = cv2.resize(_disp, (960, 540))
    cv2.imshow("Head + Smoke Detection", _disp)
    fc += 1
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release(); cv2.destroyAllWindows()
