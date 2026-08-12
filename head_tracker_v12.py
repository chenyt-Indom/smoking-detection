"""
V12-重构版 — YOLOv12s × 3 + BoT-SORT 专业追踪 + DLSS 显示
手/头/烟 全部 YOLOv12s | BoT-SORT(ultralytics botsort) | Real-ESRGAN超分 + RIFE插帧(显示层)
检测用真实帧(准确), 显示用 DLSS 增强(丝滑+高清)
"""
import cv2, numpy as np, time, torch, os, threading
from ultralytics import YOLO

# ---- 15:31 完全使用 V12 模型(COCO预训练微调)检测: hand_v12/head_v12/smoke_v12 ----
#   废掉 pose 方案; 头/手用 V12 检测模型(完整框, 像 V8 机制)
try:
    HEAD = YOLO(r"D:\视觉安防系统\models\head_v12.pt", task='detect')
    print("✅ 头模型 head_v12 加载(YOLOv12s微调)")
except Exception:
    HEAD = YOLO(r"D:\training_data\yolo12s.pt", task='detect')
    print("⚠️ head_v12 未就绪, 用 yolo12s 预训练")
HEAD.to('cuda')
try:
    # ★ 16:16 TensorRT: engine 优先(推理快2-3x, fp16)
    import os as _os
    if _os.path.exists(r"D:\视觉安防系统\models\smoke_v12.engine"):
        SMOKE = YOLO(r"D:\视觉安防系统\models\smoke_v12.engine", task='detect')
        print("✅ 烟模型 smoke_v12 TensorRT engine 加载(2-3x)")
    else:
        SMOKE = YOLO(r"D:\视觉安防系统\models\smoke_v12.pt", task='detect')
        print("✅ 烟模型 smoke_v12 加载")
except Exception:
    SMOKE = YOLO(r"D:\training_data\yolo12s.pt", task='detect')
    print("⚠️ smoke_v12 未就绪, 用 yolo12s 预训练")
try:
    SMOKE.to('cuda')
except Exception:
    pass   # ★ TensorRT engine 已绑定GPU, 不支持.to (16:16)
try:
    # ★ 20:31 手模型换回 hand.pt(V8 验证版): 用户实拍数据训练, 实拍能识别
    #   hand_v12(11k白背景)实拍分布偏移失败(只出巨型假框) → 弃用, 回退V8模型
    HAND = YOLO(r"D:\视觉安防系统\models\hand.pt", task='detect')
    print("✅ 手模型 hand.pt 加载(V8验证版, 用户实拍数据训练)")
    HAND_READY = True
except Exception:
    HAND_READY = False
    print("⚠️ hand_v12 不可用, 手部禁用")
if HAND_READY:
    HAND.to('cuda')
    try:
        HAND.model.half()
    except Exception:
        pass
# ★★ 18:40 MediaPipe Hands 手部关键点(21点): 精准手掌定位/手势/大小自适应
#   非YOLO检测模型(手部关键点库, hand_landmarker.task 7.8MB 本地)
#   21点外接框 = 完美贴合手掌(不含手臂), 握拳框小/张开框大(手势自适应)
MP_HANDS = None
_mp_ref = None
try:
    # ★ 20:33 恢复启用(独立线程隔离卡死, 见 _mp_worker)
    import mediapipe as _mp
    from mediapipe.tasks import python as _mp_python
    from mediapipe.tasks.python import vision as _mp_vision
    _mp_ref = _mp
    if _os.path.exists(r"D:\training_data\hand_landmarker.task"):
        _mp_opts = _mp_vision.HandLandmarkerOptions(
            base_options=_mp_python.BaseOptions(model_asset_path=r"D:\training_data\hand_landmarker.task"),
            num_hands=4, min_hand_detection_confidence=0.2, min_tracking_confidence=0.3)   # ★ 08-12 0.4→0.2/0.3(部分遮挡手也能检出)
        MP_HANDS = _mp_vision.HandLandmarker.create_from_options(_mp_opts)
        print("✅ MediaPipe Hands 已加载(21点手部关键点, 精准手掌)")
    else:
        print("⚠️ hand_landmarker.task 缺失, MediaPipe 未启用")
except Exception as _e:
    MP_HANDS = None
    print(f"⚠️ MediaPipe Hands 未加载: {type(_e).__name__} {str(_e)[:80]}")
# ★★ 23:52 MiDaS 单目深度估计(空间立体感知): 深度图判断"谁在前"
#   手/胳膊挡脸 → 手区域深度小(近), 脸区域深度大(远) → 头框保持, 告别2D猜测
#   ★ 00:08 懒初始化: MiDaS 在系统环境初始化会卡死(与MediaPipe并发冲突, 单测正常)
#   → 初始化移入深度线程内部(后台), 卡死只影响深度线程, 不阻塞主系统
DEPTH = None   # (model, transform) 或 None(线程内懒初始化)
_depth_map = None      # 128x72 深度图
_depth_time = 0.0      # 深度图时间戳
_depth_frame_cur = None
_depth_lock = threading.Lock()
_depth_inited = False
try:
    SMOKE.model.half()
except Exception:
    pass
# fp16: 手/烟(大目标/大模型)用, 头保持 fp32(小目标精度)
try:
    SMOKE.model.half()
except Exception:
    pass
print("✅ V12-重构 | YOLOv12s×3 + BoT-SORT | 头绿框 | 手蓝框 | 烟红框 | DLSS显示 | Q退出")

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
# ★ 采集优化(重构版): DSHOW后端(低延迟) + 缓冲1帧(降卡顿); 摄像头硬件上限~20fps
try:
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)   # 17:03 1080p采集(实测25.6fps)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)   # 17:03 1080p
except Exception:
    pass
# ★ 17:03 1080p可缩放窗口(高清显示, 窗口自适应屏幕)
cv2.namedWindow("Head + Smoke Detection", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Head + Smoke Detection", 1280, 720)
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
# ★ 摄像头分辨率(重构版): 640x360 帧率优先(实测640x480仅19-21fps, 720p更低)
#   清晰度交给 DLSS 显示层(Real-ESRGAN 超分2x), 检测输入 640 已够
#   采集优化已在上面(DSHOW + 640x360 + 缓冲1帧)
W, H = int(cap.get(3)), int(cap.get(4))

GREEN = (0, 255, 100); RED = (0, 0, 255)
fc = 0; t0 = time.time(); _gframe = 0   # ★ _gframe: pose 帧计数(14:26)

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

# ==================== DLSS 显示管线(Real-ESRGAN 超分 + RIFE 插帧) ====================
# 检测用真实帧(准确), 显示画面 DLSS 增强(丝滑+高清)
# 惰性初始化: 库/权重就绪自动启用, 未就绪跳过(不影响主系统)
_sr_model = None
_rife_model = None
_prev_disp = None
_last_frame_size = None   # ★ 超分后帧尺寸追踪(变化时 reset BoT-SORT)

def init_dlss():
    """初始化超分(Real-ESRGAN via spandrel)与插帧(RIFE)。
    14:35 spandrel 替代 realesrgan(basicsr 构建失败); 
    超分标准模型不实时 → 用于 ROI 降频增强 + 显示间歇刷新"""
    global _sr_model, _rife_model
    ok_sr, ok_rife = False, False
    try:
        if os.path.exists(r'D:/training_data/RealESRGAN_x4plus.pth'):
            from spandrel import ModelLoader
            _sr_model = ModelLoader().load_from_file(r'D:/training_data/RealESRGAN_x4plus.pth')
            # ★ 16:45 CPU常驻(不占GPU): 超分仅显示低频用, CPU够; GPU让给检测(BOTSORT/engine)
            _sr_model = _sr_model.cpu().half()
            _sr_model.eval()
            ok_sr = True
            print('✅ DLSS超分: Real-ESRGAN(spandrel) 已启用(ROI降频增强)')
    except Exception as e:
        print(f'⚠️ DLSS超分未启用: {e}')
    try:
        # ★ 16:15 RIFE 已就绪(ncnn-vulkan exe 下载完成, 431MB)
        if os.path.exists(r'D:/training_data/rife-ncnn/rife-ncnn-vulkan-20221029-windows/rife-ncnn-vulkan.exe'):
            _rife_model = True   # exe 方式(标记可用)
            ok_rife = True
            print('✅ DLSS插帧: RIFE(ncnn-vulkan) 已启用')
    except Exception as e:
        print(f'⚠️ DLSS插帧未启用: {e}')
    return ok_sr, ok_rife

# ★ DLSS 显示初始化(库/权重就绪自动启用, 未就绪跳过) — 必须在函数定义之后调用
init_dlss()

# ★ 14:35 超分增强(烟ROI): 标准 Real-ESRGAN 不实时(ROI 0.2-0.5s) → 降频(每1.5s一次) + 缓存
_sr_roi_cache = None      # (超分ROI, 原ROI坐标, 时间戳)
_sr_roi_key = None        # ROI key(避免重复)
_last_sr_time = 0.0

def sr_enhance_roi(frame, x1, y1, x2, y2, key):
    """ROI 超分(降频缓存): 返回超分ROI或None(未到时间/失败)"""
    global _sr_roi_cache, _sr_roi_key, _last_sr_time
    if _sr_model is None:
        return None
    now = time.time()
    if now - _last_sr_time < 1.5:
        return None   # 降频
    _last_sr_time = now
    try:
        x1, y1 = max(0, int(x1)), max(0, int(y1))
        x2, y2 = min(frame.shape[1], int(x2)), min(frame.shape[0], int(y2))
        if x2-x1 < 32 or y2-y1 < 32:
            return None
        crop = frame[y1:y2, x1:x2]
        t = torch.from_numpy(crop.transpose(2,0,1)[None].astype(np.float32)/255.0).cuda().half()
        with torch.no_grad():
            up = _sr_model(t)[0]
        up = (up.clamp(0,1).permute(1,2,0).cpu().numpy()*255).astype(np.uint8)
        _sr_roi_cache = (up, (x1, y1, x2, y2))
        return up
    except Exception:
        return None

# ★★ 16:29 超分异步线程(展示超分效果, 不阻塞主循环): 后台每3s超分当前帧 → _sr_latest
_sr_latest = None
_sr_latest_frame = None
_sr_thread = None
_sr_demo = None          # 16:58 效果展示: (原ROI, 超分ROI) 对比缓存
_rife_demo = None        # 16:58 效果展示: (原帧, 插帧) 对比缓存
_demo_a = None           # RIFE演示用前一帧
_demo_b = None
def _sr_worker():
    """★ 16:58 超分效果展示(CPU后台10s/次): 只超分中央ROI(320x180) → _sr_demo对比窗
    CPU常驻不占GPU, 主循环只贴图不阻塞"""
    global _sr_latest, _sr_latest_frame, _sr_demo
    while not _stop_flag.is_set():
        time.sleep(10.0)
        try:
            _cur = _sr_latest_frame
            if _cur is None or _sr_model is None:
                continue
            _h, _w = _cur.shape[:2]
            _rx1, _ry1 = max(0, _w//2-160), max(0, _h//2-90)
            _rx2, _ry2 = min(_w, _w//2+160), min(_h, _h//2+90)
            _roi = _cur[_ry1:_ry2, _rx1:_rx2]
            _t = torch.from_numpy(_roi.transpose(2,0,1)[None].astype(np.float32)/255.0)
            with torch.no_grad():
                _up = _sr_model(_t)[0]   # CPU推理
            _sr_roi = (np.clip(_up.permute(1,2,0).numpy(),0,1)*255).astype(np.uint8)
            _sr_demo = (_roi.copy(), _sr_roi)
        except Exception:
            pass

def _rife_worker():
    """★ 16:58 RIFE效果展示(后台10s/次): 用最近两帧插帧 → _rife_demo对比窗"""
    global _rife_demo, _demo_a, _demo_b
    while not _stop_flag.is_set():
        time.sleep(10.0)
        try:
            if _demo_a is None or _demo_b is None:
                continue
            import subprocess
            cv2.imwrite(r'D:/training_data/_d_a.png', _demo_a)
            cv2.imwrite(r'D:/training_data/_d_b.png', _demo_b)
            subprocess.run([
                r'D:/training_data/rife-ncnn/rife-ncnn-vulkan-20221029-windows/rife-ncnn-vulkan.exe',
                '-0', r'D:/training_data/_d_a.png', '-1', r'D:/training_data/_d_b.png',
                '-o', r'D:/training_data/_d_mid.png', '-g', '0'],
                timeout=10, capture_output=True)
            _mid = cv2.imread(r'D:/training_data/_d_mid.png')
            if _mid is not None:
                _rife_demo = (_demo_b.copy(), _mid)
        except Exception:
            pass

# ★ 17:13 演示线程彻底禁用(卡顿元凶3): 16:58 加的对比窗已移除但线程没停
#   RIFE 线程每10s启2s子进程插帧 / 超分线程每10s占CPU → 周期性卡顿
#   用户只需流畅1080p: 超分/RIFE 均不实时, 全部停用(代码保留, 离线可用)
_stop_flag = threading.Event()   # ★ 全局停止标志(线程用)
if False and _sr_model is not None:
    _sr_thread = threading.Thread(target=_sr_worker, daemon=True)
    _sr_thread.start()
    print("✅ 超分演示线程已启动(CPU/10s, 效果对比窗)")
if False and _rife_model is not None:
    _rife_thread = threading.Thread(target=_rife_worker, daemon=True)
    _rife_thread.start()
    print("✅ RIFE演示线程已启动(10s, 效果对比窗)")

def dlss_super_resolve(frame):
    """★ 超分前置: 采集帧 → Real-ESRGAN 超分 2x → 720p
    超分帧同时用于检测(小目标细节恢复)与显示。失败返回原帧。"""
    if _sr_model is None or frame is None:
        return frame
    try:
        _sr_model.half()
        _up, _ = _sr_model.enhance(frame, outscale=2)
        return _up
    except Exception:
        return frame

def dlss_interp(prev_frame, curr_frame):
    """★ 16:15 RIFE 插帧(ncnn-vulkan exe, 本地GPU): 前后帧生成中间帧 → 显示丝滑
    降尺寸到320x180加速(插帧只求视觉流畅, 显示时resize回)"""
    global _rife_model
    if prev_frame is None or curr_frame is None:
        return None
    try:
        import subprocess
        _h1, _w1 = prev_frame.shape[:2]
        _small = lambda f: cv2.resize(f, (320, 180), interpolation=cv2.INTER_AREA)
        cv2.imwrite(r'D:/training_data/_rife_a.png', _small(prev_frame))
        cv2.imwrite(r'D:/training_data/_rife_b.png', _small(curr_frame))
        subprocess.run([
            r'D:/training_data/rife-ncnn/rife-ncnn-vulkan-20221029-windows/rife-ncnn-vulkan.exe',
            '-0', r'D:/training_data/_rife_a.png', '-1', r'D:/training_data/_rife_b.png',
            '-o', r'D:/training_data/_rife_mid.png', '-g', '0'],
            timeout=8, capture_output=True)
        mid = cv2.imread(r'D:/training_data/_rife_mid.png')
        if mid is not None:
            return cv2.resize(mid, (_w1, _h1))
        return None
    except Exception:
        return None


# ==================== POSE 关键点机制(V8沿用, 用户14:26) ====================
#   头/手不训练检测模型, 用预训练 YOLOv8n-pose 关键点直接在画面找(不偏离预训练)
#   COCO pose 17关键点: 0鼻 1左眼 2右眼 3左耳 4右耳 5/6肩 7/8肘 9/10腕 11/12髋 ...
import onnxruntime as ort

_pose_session = None
_pose_input_name = None
_POSE_SIZE = 320
try:
    _pose_session = ort.InferenceSession(r"D:\视觉安防系统\models\yolov8n-pose.onnx",
                                         providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    _pose_input_name = _pose_session.get_inputs()[0].name
    _POSE_SIZE = int(_pose_session.get_inputs()[0].shape[2])
    print(f"✅ POSE 已加载(yolov8n-pose.onnx, {_POSE_SIZE})")
except Exception as _e:
    _pose_session = None
    print(f"⚠️ POSE 加载失败: {_e}")

HEAD_KPT_IDS = (0, 1, 2, 3, 4)   # 鼻/眼/耳

# ★ 15:07 检测框时间平滑(V12 稳定性修复核心): pose 关键点逐帧抖动 → 帧间 EMA
#   V8 用模型每帧输出稳定框; pose 输入抖动, 必须加这一层才能达到同等稳定
_prev_head_boxes = []
_head_w_ema = None   # ★ 15:20 头宽时间EMA(防多策略切换跳变)
_head_cx_ema = None   # ★ 15:27 头中心x EMA
_head_cy_ema = None   # ★ 15:27 头中心y EMA
_prev_hand_boxes = []
_mp_prev_boxes = []   # ★ 20:16 MediaPipe 手框时间EMA(治漂移)

def smooth_boxes(new_boxes, prev_boxes, alpha=0.60):
    """新检测框与上一帧框 EMA 平滑(位置/大小, 防关键点抖动导致框跳变)
    alpha=1 完全用新框; alpha=0.6 偏新但滤抖动。返回 (平滑框, 更新后的prev)"""
    if not prev_boxes:
        return new_boxes, new_boxes
    out = []
    used = set()
    for nb in new_boxes:
        ncx = (nb[0] + nb[2]) / 2; ncy = (nb[1] + nb[3]) / 2
        nw = nb[2] - nb[0]
        best_i, best_d = -1, nw * 0.6   # 中心距 < 0.6倍宽 才算同一目标
        for i, pb in enumerate(prev_boxes):
            if i in used:
                continue
            pcx = (pb[0] + pb[2]) / 2; pcy = (pb[1] + pb[3]) / 2
            d = ((ncx - pcx) ** 2 + (ncy - pcy) ** 2) ** 0.5
            if d < best_d:
                best_d = d; best_i = i
        if best_i >= 0:
            used.add(best_i)
            pb = prev_boxes[best_i]
            out.append(tuple(pb[k] * (1 - alpha) + nb[k] * alpha for k in range(4)))
        else:
            out.append(nb)
    return out, new_boxes

def detect_pose(frame):
    """全帧姿态估计 → [(bbox, kpts17), ...]"""
    if _pose_session is None:
        return []
    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (_POSE_SIZE, _POSE_SIZE), swapRB=True, crop=False)
    outs = _pose_session.run(None, {_pose_input_name: blob})
    preds = outs[0]
    if len(preds.shape) == 3:
        preds = np.transpose(preds[0], (1, 0))   # (N, 56)
    scores = preds[:, 4]
    mask = scores > 0.40
    if not np.any(mask):
        return []
    idxs = np.where(mask)[0]
    idxs = idxs[np.argsort(scores[idxs])[::-1]]
    # NMS
    boxes = preds[:, :4]
    keep = []
    for i in idxs:
        _ov = False
        for j in keep:
            ix1 = max(boxes[i,0]-boxes[i,2]/2, boxes[j,0]-boxes[j,2]/2)
            iy1 = max(boxes[i,1]-boxes[i,3]/2, boxes[j,1]-boxes[j,3]/2)
            ix2 = min(boxes[i,0]+boxes[i,2]/2, boxes[j,0]+boxes[j,2]/2)
            iy2 = min(boxes[i,1]+boxes[i,3]/2, boxes[j,1]+boxes[j,3]/2)
            _iw = max(0, ix2-ix1); _ih = max(0, iy2-iy1)
            _ar1 = boxes[i,2]*boxes[i,3]; _ar2 = boxes[j,2]*boxes[j,3]
            if _iw*_ih / max(1e-6, min(_ar1, _ar2)) > 0.5:
                _ov = True; break
        if not _ov:
            keep.append(i)
    sx = w / _POSE_SIZE; sy = h / _POSE_SIZE
    persons = []
    for idx in keep:
        kpts = preds[idx, 5:].reshape(17, 3)
        kpts[:, 0] *= sx; kpts[:, 1] *= sy
        persons.append(([
            int((preds[idx,0]-preds[idx,2]/2)*sx), int((preds[idx,1]-preds[idx,3]/2)*sy),
            int((preds[idx,0]+preds[idx,2]/2)*sx), int((preds[idx,1]+preds[idx,3]/2)*sy),
        ], kpts))
    return persons

def kpts_to_head_bbox(kpts, w, h):
    """关键点 → 头部框(多策略, 移植自 detector.py _kpts_to_head_bbox)
    ★15:20 头宽时间EMA: 多策略切换(双耳/双眼/单眼/跨度)导致 est_w 跳变 → 框大小忽大忽小"""
    global _head_w_ema
    valid, weights = [], []
    for i in HEAD_KPT_IDS:
        x, y, c = kpts[i]
        if c > 0.15:
            valid.append([x, y]); weights.append(c)
    if len(valid) < 2:
        return None
    valid = np.array(valid); weights = np.array(weights)
    kw_raw = valid[:, 0].max() - valid[:, 0].min()
    kh_raw = valid[:, 1].max() - valid[:, 1].min()
    if kpts[3, 2] > 0.15 and kpts[4, 2] > 0.15:
        est_w = abs(kpts[3,0]-kpts[4,0]) / 0.65
    elif kpts[1, 2] > 0.15 and kpts[2, 2] > 0.15:
        est_w = abs(kpts[1,0]-kpts[2,0]) / 0.45
    elif kpts[1, 2] > 0.15 or kpts[2, 2] > 0.15:
        vx = kpts[1,0] if kpts[1,2] > 0.15 else kpts[2,0]
        est_w = max(abs(vx-kpts[0,0]) * 3.5, 50)
    else:
        est_w = max(kw_raw * 1.4, 60)
    # ★ 15:20 头宽 EMA(0.75旧+0.25新): 滤策略切换跳变; 变化率限幅(单帧±15%)
    if _head_w_ema is None:
        _head_w_ema = est_w
    else:
        _prev_w = _head_w_ema
        _head_w_ema = 0.75 * _head_w_ema + 0.25 * est_w
        if abs(_head_w_ema - _prev_w) > _prev_w * 0.15:   # 限幅: 单帧变化≤15%
            _head_w_ema = _prev_w + (_head_w_ema - _prev_w) / abs(_head_w_ema - _prev_w) * _prev_w * 0.15
    est_w = _head_w_ema
    # ★★ 15:27 框中心 cx/cy 也 EMA(0.75旧+0.25新): 关键点位置每帧波动 → 框位置抖(15:26偏右)
    cx_raw = float(np.average(valid[:, 0], weights=weights))
    cy_raw = float(np.average(valid[:, 1], weights=weights))
    global _head_cx_ema, _head_cy_ema
    if _head_cx_ema is None:
        _head_cx_ema, _head_cy_ema = cx_raw, cy_raw
    else:
        # 位置 EMA + 单帧限幅(头宽×0.25, 防止关键点瞬移)
        for _cur, _ema, _name in [(cx_raw, _head_cx_ema, '_head_cx_ema'), (cy_raw, _head_cy_ema, '_head_cy_ema')]:
            _new = 0.75 * _ema + 0.25 * _cur
            if abs(_new - _ema) > est_w * 0.25:
                _new = _ema + (_new - _ema) / abs(_new - _ema) * est_w * 0.25
            if _name == '_head_cx_ema':
                _head_cx_ema = _new
            else:
                _head_cy_ema = _new
    cx, cy = _head_cx_ema, _head_cy_ema
    est_w = min(est_w, w * 0.35)
    est_h = est_w * 1.55
    x1 = max(0, int(cx - est_w / 2)); x2 = min(w, int(cx + est_w / 2))
    y1 = max(0, int(cy - est_h * 0.48)); y2 = min(h, int(cy + est_h * 0.52))
    if x2 - x1 < 12 or y2 - y1 < 12:
        return None
    return (x1, y1, x2, y2)

def kpts_to_hand_bbox(kpts, w, h):
    """手腕关键点(9/10) → 手框(★15:16 扩大: V8 模型输出完整手掌bbox, pose只有手腕)
    手长(手腕→指尖) ≈ 头宽×1.1; 以手腕为起点, 沿"手肘→手腕"方向外扩覆盖手指"""
    # 先取头宽作为手的尺寸参考
    hb = kpts_to_head_bbox(kpts, w, h)
    ref_w = (hb[2]-hb[0]) if hb else 80
    hand_len = ref_w * 1.15    # 手指+手掌长度 ≈ 头宽×1.15
    hand_wid = ref_w * 0.75    # 手掌宽度
    for wid, eid in ((9, 7), (10, 8)):   # 手腕→肘部方向(手指在手腕向外的延伸)
        wx, wy, wc = kpts[wid]
        if wc < 0.25:
            continue
        ex, ey, ec = kpts[eid]
        if ec > 0.15 and (abs(ex-wx) > 5 or abs(ey-wy) > 5):
            # 方向向量: 肘→腕(手朝外方向), 归一化
            dx, dy = wx - ex, wy - ey
            dl = (dx*dx + dy*dy) ** 0.5
            ux, uy = dx/dl, dy/dl
            # 手框起点=手腕, 终点=手腕+手长(沿肘→腕方向)
            fx, fy = wx + ux * hand_len, wy + uy * hand_len
            x1 = max(0, int(min(wx, fx) - hand_wid * 0.4))
            x2 = min(w, int(max(wx, fx) + hand_wid * 0.4))
            y1 = max(0, int(min(wy, fy) - hand_wid * 0.4))
            y2 = min(h, int(max(wy, fy) + hand_wid * 0.4))
        else:
            # 无肘部参考: 以手腕为中心对称扩(覆盖手指方向)
            x1 = max(0, int(wx - hand_len * 0.5)); x2 = min(w, int(wx + hand_len * 0.5))
            y1 = max(0, int(wy - hand_len * 0.5)); y2 = min(h, int(wy + hand_len * 0.5))
        if x2 - x1 >= 15 and y2 - y1 >= 15:
            return (x1, y1, x2, y2)
    return None


def detect_hand_mp(frame):
    """★ 18:40 MediaPipe 21点手部关键点 → 手掌外接框(精准贴合, 不含手臂)
    21点(指尖×5+指节×10+手腕+手掌)外接框: 握拳框小/张开框大(手势大小自适应)
    ★ 20:16 框级时间EMA: 21点外接框对关键点抖动敏感(min/max点一跳→框边跳→漂移)
    帧间匹配(中心距<0.6宽)后 EMA(0.55旧+0.45新) 位置+尺寸 → 消除漂移
    返回 [(x1,y1,x2,y2), ...]; 未启用/无手时返回 []"""
    global _mp_prev_boxes
    if MP_HANDS is None or frame is None:
        return []
    try:
        _h, _w = frame.shape[:2]
        _rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        _mimg = _mp_ref.Image(image_format=_mp_ref.ImageFormat.SRGB, data=_rgb)
        _res = MP_HANDS.detect(_mimg)
        _boxes = []
        if _res.hand_landmarks:
            for _lm in _res.hand_landmarks:
                _xs = [p.x for p in _lm]; _ys = [p.y for p in _lm]
                _x1 = max(0, int(min(_xs) * _w)); _y1 = max(0, int(min(_ys) * _h))
                _x2 = min(_w, int(max(_xs) * _w)); _y2 = min(_h, int(max(_ys) * _h))
                # 4% padding: 贴合手掌不留白(完美不多余)
                _pw, _ph = (_x2 - _x1) * 0.04, (_y2 - _y1) * 0.04
                _x1 = max(0, int(_x1 - _pw)); _y1 = max(0, int(_y1 - _ph))
                _x2 = min(_w, int(_x2 + _pw)); _y2 = min(_h, int(_y2 + _ph))
                if _x2 - _x1 >= 15 and _y2 - _y1 >= 15:
                    _boxes.append((_x1, _y1, _x2, _y2))
        # ★ 20:16 框级时间EMA(治漂移): 与上帧框匹配后平滑, 不匹配(新出现/瞬移)直接用
        if _mp_prev_boxes:
            _out = []
            _used = set()
            for _b in _boxes:
                _bc = ((_b[0]+_b[2])/2, (_b[1]+_b[3])/2)
                _bw2 = _b[2] - _b[0]
                _bi, _bd = -1, _bw2 * 0.6
                for _i, _pb in enumerate(_mp_prev_boxes):
                    if _i in _used: continue
                    _pc = ((_pb[0]+_pb[2])/2, (_pb[1]+_pb[3])/2)
                    _d = ((_bc[0]-_pc[0])**2 + (_bc[1]-_pc[1])**2) ** 0.5
                    if _d < _bd:
                        _bd = _d; _bi = _i
                if _bi >= 0:
                    _used.add(_bi)
                    _pb = _mp_prev_boxes[_bi]
                    _out.append(tuple(int(_pb[k]*0.55 + _b[k]*0.45) for k in range(4)))
                else:
                    _out.append(_b)
            _boxes = _out
        _mp_prev_boxes = _boxes
        return _boxes
    except Exception:
        return []


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
        self.pn = pn   # ★ 02:21 记录pn(区分目标: 手=0.50外推约束, 烟=0.80高速外推)
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
        # ★ 18:48 上下文激活: 该头/手框内确认过香烟 → True
        #   激活后框内"非手自带物体"直接归类为烟(烟转向外形剧变不再丢失)
        self.smoke_activated = False
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
            # ★★ 02:21 手丢失外推约束(治"手遮挡消失后框乱飘"): 手丢失时预测按
            #   速度惯性外推(手没了框还飘走) → 外推超 1.5手宽 回拉最后确认位置
            #   + 速度减半(框停在最后确认位置, 不乱飘)
            #   [手专用 pn=0.50; 烟 pn=0.80 保持原行为]
            if self.pn == 0.50 and self.last_good is not None:
                _lgc = ((self.last_good[0]+self.last_good[2])/2, (self.last_good[1]+self.last_good[3])/2)
                _w_ref = max(20.0, self.last_good[2]-self.last_good[0])
                s = self.kf.statePost.flatten()
                _d_now = ((s[0]-_lgc[0])**2 + (s[1]-_lgc[1])**2) ** 0.5
                if _d_now > _w_ref * 1.5:
                    _pull = min(0.5, (_d_now - _w_ref*1.5) / (_w_ref*1.5))
                    s[0] = s[0]*(1-_pull) + _lgc[0]*_pull
                    s[1] = s[1]*(1-_pull) + _lgc[1]*_pull
                    s[4] *= 0.5; s[5] *= 0.5
                    self.kf.statePost = s.reshape(-1, 1)
        p = self.kf.predict().flatten()
        self.bbox = (float(p[0]-p[2]/2), float(p[1]-p[3]/2), float(p[0]+p[2]/2), float(p[1]+p[3]/2))
        # 显示平滑: 丢失期间显示框缓慢跟随预测(不跳变)
        # ★ 20:12 速度自适应: 快速移动(位移>0.5手宽/帧)→α0.92紧跟(不拖尾)
        #   中速→0.82 | 静止→0.60(稳定) | 丢失期→0.50(慢跟随预测)
        alpha = 0.50
        if self.lost <= 0:
            d = self.disp_bbox
            _cx1 = (self.bbox[0]+self.bbox[2])/2; _cy1 = (self.bbox[1]+self.bbox[3])/2
            _cx2 = (d[0]+d[2])/2; _cy2 = (d[1]+d[3])/2
            _disp_dist = ((_cx1-_cx2)**2 + (_cy1-_cy2)**2) ** 0.5
            _w_avg = max(15.0, (self.bbox[2]-self.bbox[0] + (d[2]-d[0])) / 2.0)
            _speed = _disp_dist / _w_avg
            if _speed > 0.5: alpha = 0.97    # 快移: 极限紧跟(17:47 0.92→0.97)
            elif _speed > 0.2: alpha = 0.82  # 中速
            else: alpha = 0.60               # 静止: 稳定
        d = self.disp_bbox
        self.disp_bbox = (
            d[0]*(1-alpha) + self.bbox[0]*alpha, d[1]*(1-alpha) + self.bbox[1]*alpha,
            d[2]*(1-alpha) + self.bbox[2]*alpha, d[3]*(1-alpha) + self.bbox[3]*alpha)
        return self.bbox
    def update(self, bbox):
        cx = (bbox[0] + bbox[2]) / 2; cy = (bbox[1] + bbox[3]) / 2
        w = bbox[2] - bbox[0]; h = bbox[3] - bbox[1]
        # ★★ 02:18 尺寸时间EMA(治"快速运动时大小抖动/瞬间变又恢复"): hand.pt 在
        #   运动模糊帧输出框尺寸忽大忽小 → 直接进KF → 框大小跳变
        #   [只平滑尺寸, 位置cx/cy不动] w/h 0.7旧+0.3新:
        #     手靠近/远离(渐进) → 平滑跟随; 模糊抖动 → 被滤
        _se = getattr(self, '_size_ema', None)
        if _se is None:
            _se = (float(w), float(h))
        else:
            _se = (_se[0]*0.7 + w*0.3, _se[1]*0.7 + h*0.3)
        self._size_ema = _se
        w, h = _se[0], _se[1]
        # ★★ 00:46 方向突变检测修正(治快速移动跟不准): 原逻辑"偏移>70px就速度清零"
        #   是给烟变向设计的 → 手快速甩动(帧间位移100-200px)被误判突变 → 速度清零
        #   → 预测不外推 → 框跟不上手。修正: 仅"反向移动"才清零(测量在预测反方向=真突变),
        #     同向快速移动(甩手)不清零, Kalman速度持续外推 → 框紧跟手
        pred = self.kf.statePre.flatten()
        _dx = cx - pred[0]; _dy = cy - pred[1]
        _vx = pred[4]; _vy = pred[5]
        if (_dx*_vx + _dy*_vy) < 0 and (_dx*_dx + _dy*_dy) > 70**2:
            self.kf.statePost[4] = 0; self.kf.statePost[5] = 0
        self.kf.correct(np.array([[cx], [cy], [w], [h]], np.float32)); self.lost = 0
        self.confirmed += 1
        # 速度限幅 ±200px/帧 (17:47 用户: 还要更快 → 120→200, 极限跟速)
        s = self.kf.statePost.flatten()
        s[4] = float(np.clip(s[4], -200, 200)); s[5] = float(np.clip(s[5], -200, 200))
        self.kf.statePost = s.reshape(-1, 1)
        p = s
        self.bbox = (float(p[0]-p[2]/2), float(p[1]-p[3]/2), float(p[0]+p[2]/2), float(p[1]+p[3]/2))
        # 显示平滑: EMA 混合检测框与旧显示框, 消除快速移动后的瞬移
        # 丢失重获时瞬移最大 → 检测置信度低(lost>0刚恢复)时用更重平滑
        # ★ 18:08 防抖: alpha 0.85→0.72, 烟框更稳定不"一抖一抖"(代价:轻微滞后)
        alpha = 0.50 if self.lost > 0 else 0.72  # ★显示紧跟检测(不滞后)
        d = self.disp_bbox
        self.disp_bbox = (
            d[0]*(1-alpha) + self.bbox[0]*alpha, d[1]*(1-alpha) + self.bbox[1]*alpha,
            d[2]*(1-alpha) + self.bbox[2]*alpha, d[3]*(1-alpha) + self.bbox[3]*alpha)
        self.last_good = bbox   # 更新最后确认位置
        self.candidates = []   # 恢复关联后清空候选
        self._last_det = ((bbox[0]+bbox[2])/2, (bbox[1]+bbox[3])/2)   # ★ 08-12 上帧匹配检测中心(匹配惯性用)
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

def hand_skin_ok(frame, x1, y1, x2, y2, W, H):
    """★ 08-12 肤色验证(抽取自检测链, 低分候选升级建轨复用):
    人手: 肤色占比≥25% + 高饱和红木占比≤45% (治椅子/木纹误判)"""
    _hsx1, _hsy1 = max(0,int(x1)), max(0,int(y1))
    _hsx2, _hsy2 = min(W,int(x2)), min(H,int(y2))
    if _hsx2 - _hsx1 < 8 or _hsy2 - _hsy1 < 8:
        return False
    try:
        _hcrop = frame[_hsy1:_hsy2, _hsx1:_hsx2]
        _hhsv = cv2.cvtColor(_hcrop, cv2.COLOR_BGR2HSV)
        _hpix = _hcrop.shape[0] * _hcrop.shape[1]
        _hskin = float(np.sum(cv2.inRange(_hhsv, (0,25,50), (45,180,255)) > 0)) / _hpix
        if _hskin < 0.25:
            return False   # 肤色<25% → 非人手
        _hsat_red = float(np.sum(cv2.inRange(_hhsv, (0,150,40), (30,255,255)) > 0)) / _hpix
        if _hsat_red > 0.45:
            return False   # 高饱和红木 → 拒
        return True
    except Exception:
        return False

# ==================== ★★ 08-12 新架构: 物品检测 + 最清晰帧截图 + 后台识别 ====================
#   摒弃实时香烟检测: 实时只判断"头框/手框内有无非人体物品"(有→框变红),
#   有物品时每ID只截"最清晰一帧"入库, 后台用超分+香烟模型识别静态图

def _roi_clarity(roi):
    """ROI清晰度评分: Laplacian方差(越高越清晰/细节越多)"""
    try:
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        return float(cv2.Laplacian(gray, cv2.CV_64F).var())
    except Exception:
        return 0.0

def _roi_has_object(frame, x1, y1, x2, y2, kind):
    """检测ROI内是否有非人体天然物品(烟/吸管/手持物)。
    原理: 肤色掩码 → 内部"非肤色连通块"= 可能的物品(排除接触边缘的背景块)
    kind='hand': 手框内非肤色块(烟/杯子/手机等手持物)
    kind='head': 嘴部区域(头框下部中央)非肤色块(嘴含的烟/吸管)
    返回 True/False
    """
    try:
        Hf, Wf = frame.shape[:2]
        x1, y1 = max(0, int(x1)), max(0, int(y1))
        x2, y2 = min(Wf, int(x2)), min(Hf, int(y2))
        if x2-x1 < 18 or y2-y1 < 18:
            return False
        if kind == 'head':
            # 嘴部区域: 头框下部 55%~90% 高度, 中央 70% 宽度(脸下部=嘴/下巴)
            _my1 = y1 + int((y2-y1)*0.55); _my2 = y1 + int((y2-y1)*0.90)
            _mx1 = x1 + int((x2-x1)*0.15); _mx2 = x2 - int((x2-x1)*0.15)
            if _my2 - _my1 < 8 or _mx2 - _mx1 < 8:
                return False
            roi = frame[_my1:_my2, _mx1:_mx2]
        else:
            roi = frame[y1:y2, x1:x2]
        if roi.shape[0] < 8 or roi.shape[1] < 8:
            return False
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        nonskin = ~(cv2.inRange(hsv, (0,25,50), (45,180,255)) > 0)
        # 开运算去细小噪点(手指缝/纹理阴影)
        nonskin = cv2.morphologyEx(nonskin.astype(np.uint8), cv2.MORPH_OPEN,
                                   np.ones((3,3), np.uint8)) > 0
        n, labels, stats, _ = cv2.connectedComponentsWithStats(nonskin.astype(np.uint8), 8)
        total = roi.shape[0] * roi.shape[1]
        max_area = 0
        max_dim = 0.0     # 最大"维度占比"(细长物: 烟杆/吸管某维度占ROI比例大)
        for li in range(1, n):
            x, y, w, h, area = stats[li]
            if area < 25: continue
            touch_edge = (x <= 0 or y <= 0 or x+w >= roi.shape[1]-1 or y+h >= roi.shape[0]-1)
            if touch_edge and area > total * 0.30:
                continue   # 大面积接触边缘 = 背景延伸 → 排除
            # 烟/吸管伸出嘴/手外会接触边缘(小块) → 保留
            max_dim = max(max_dim, w / roi.shape[1], h / roi.shape[0])
            if area > max_area:
                max_area = area
        # 三判据: ①内部非肤色块占比(粗物) ②非肤色总占比(细长物) ③最大维度占比(竖烟/横吸管)
        block_ratio = max_area / max(1, total)
        nonskin_ratio = float(np.sum(nonskin)) / max(1, total)
        if kind == 'head':
            return block_ratio > 0.08 or nonskin_ratio > 0.10 or max_dim > 0.40
        else:
            return block_ratio > 0.12 or nonskin_ratio > 0.25 or max_dim > 0.45
    except Exception:
        return False

def _super_resolve(img):
    """后台超分辨率增强(Real-ESRGAN CPU): 静态截图识别前清晰化, 返回超分图或None
    注意: _sr_model 为 half(fp16), 输入必须 .half() 否则 RuntimeError;
    CPU推理慢 → 输入长边限制≤256(超分4x后≤1024, 单张2-5秒)"""
    if _sr_model is None:
        return None
    try:
        h, w = img.shape[:2]
        scale = min(1.0, 256.0 / max(h, w))
        if scale < 1.0:
            img = cv2.resize(img, (max(1, int(w*scale)), max(1, int(h*scale))))
        t = torch.from_numpy(img.transpose(2,0,1)[None].astype(np.float32)/255.0).half()
        with torch.no_grad():
            up = _sr_model(t)[0]
        return (np.clip(up.permute(1,2,0).cpu().numpy(), 0, 1) * 255).astype(np.uint8)
    except Exception:
        return None

def _try_capture_best(frame, x1, y1, x2, y2, obj_type, obj_id):
    """每ID只保存"最清晰一帧"截图入库(Laplacian方差评分, 更高才覆盖)"""
    try:
        Hf, Wf = frame.shape[:2]
        x1, y1 = max(0, int(x1)-6), max(0, int(y1)-6)
        x2, y2 = min(Wf, int(x2)+6), min(Hf, int(y2)+6)
        if x2-x1 < 24 or y2-y1 < 24:
            return
        roi = frame[y1:y2, x1:x2]
        clarity = _roi_clarity(roi)
        key = (obj_type, obj_id)
        prev = _best_caps.get(key)
        if prev is not None and clarity <= prev[0] + 2.0:
            return   # 未超过历史最好(±2容差) → 不覆盖(避免抖动反复写盘)
        _best_caps[key] = (clarity, time.time())
        path = os.path.join(CAP_DIR, f"{obj_type}_{obj_id}.jpg")
        cv2.imwrite(path, roi)
        _obj_db.upsert_cap(obj_type, obj_id, path, clarity)
    except Exception:
        pass

def _analyzer_worker():
    """★ 08-12 后台识别线程: 扫描数据库未处理截图 → 超分清晰化 → 香烟模型静态识别 → 写回
    摒弃实时香烟检测: 识别只用"最清晰帧"(静态图精度远高于实时流弱帧)"""
    while not _stop_flag.is_set():
        time.sleep(2.0)
        try:
            items = _obj_db.get_unprocessed(3)
            for it in items:
                try:
                    p = it['cap_path']
                    if not os.path.exists(p):
                        _obj_db.mark_processed(it['id'], 0, 0.0, None)
                        continue
                    img = cv2.imread(p)
                    if img is None:
                        continue
                    sup_path = None
                    infer_img = img
                    # 1) 超分清晰化(2x~4x, 识别输入用)
                    sup = _super_resolve(img)
                    if sup is not None:
                        sup_path = p.replace('.jpg', '_sr.jpg')
                        cv2.imwrite(sup_path, sup)
                        infer_img = sup
                    # 2) 香烟模型静态识别
                    best_conf = 0.0
                    try:
                        res = SMOKE(infer_img, conf=0.25, iou=0.5, imgsz=640, verbose=False)
                        for r in res:
                            if r.boxes is None: continue
                            for box in r.boxes:
                                c = float(box.conf.cpu().numpy()[0])
                                if c > best_conf:
                                    best_conf = c
                    except Exception:
                        pass
                    result = 1 if best_conf >= 0.25 else 0
                    _obj_db.mark_processed(it['id'], result, best_conf, sup_path)
                except Exception:
                    try:
                        _obj_db.mark_processed(it['id'], 0, 0.0, None)
                    except Exception:
                        pass
        except Exception:
            pass


def light_source_mask(gray):
    """★08-10 15:23 大面积光源掩码: 分辨"白色内容(烟/白物)"与"大面积光源(窗/灯/过曝)"
    ① 亮度>200 亮区 → 15x15闭运算合并成区域
    ② 每个连通区域:
       - 面积 < 画面1.5% → 白色小块(可能是白烟/白物体) → 保留, 不算光源
       - 内部梯度低(Laplacian方差<60) = 均匀发光区 → 光源
    返回 bool mask(True=光源)
    """
    h, w = gray.shape
    bright = (gray > 200).astype(np.uint8)
    kernel = np.ones((15, 15), np.uint8)
    bright = cv2.morphologyEx(bright, cv2.MORPH_CLOSE, kernel)
    n, _, stats, _ = cv2.connectedComponentsWithStats(bright, 8)
    mask = np.zeros((h, w), dtype=bool)
    for i in range(1, n):
        x, y, bw, bh, area = stats[i]
        if area < 0.015 * h * w:      # 小块 → 白色内容(烟/白物), 不算光源
            continue
        roi = gray[y:y+bh, x:x+bw]
        lap = cv2.Laplacian(roi, cv2.CV_64F)
        if float(lap.var()) < 60.0:   # 内部均匀(低梯度) = 大面积发光源
            mask[y:y+bh, x:x+bw] = True
    return mask

tracks = []; smoke_tracks = []; hand_tracks = []; next_id = 0   # ★ hand_tracks: 手部追踪(蓝色框)
_prev_low_pool = []   # ★★ 08-12 上一帧手低分候选池(低分候选连续2帧确认建轨, 提升识别灵敏度)
import os
AUTO_SAVE = r'D:\training_data\smoke\fp_auto'   # 误检帧自动采集(检出烟时保存)
os.makedirs(AUTO_SAVE, exist_ok=True)
last_auto_save = 0

# ★★ 重构版: 多线程采集(采集线程持续读帧 → 有界队列; 处理主线程取最新帧)
#   摄像头只被采集线程访问(线程安全), 处理耗时不再拖采集 → 帧率稳定
import threading, queue as _queue
_frame_q = _queue.Queue(maxsize=2)
_stop_flag = threading.Event()

# ★★ 20:33 MediaPipe 独立线程(21点精准手掌, 卡死自动降级):
#   实测 mediapipe detect 与摄像头流并发会卡死 → 放独立线程隔离, 主循环只取最新结果
#   - MediaPipe 正常 → 21点外接框(精准手掌/手势/大小)
#   - 结果超时(>0.8s未更新=线程卡死) → 主循环自动回退 hand.pt/pose, 系统不崩
_mp_latest_boxes = []          # [(x1,y1,x2,y2), ...] 最新手掌框
_mp_latest_time = 0.0          # 结果时间戳
_mp_frame_cur = None           # 最新帧(线程取)
_mp_lock = threading.Lock()
def _mp_worker():
    global _mp_latest_boxes, _mp_latest_time, _mp_frame_cur, _mp_prev_boxes
    while True:
        time.sleep(0.02)
        try:
            with _mp_lock:
                _fr = _mp_frame_cur
            if _fr is None or MP_HANDS is None:
                continue
            _h, _w = _fr.shape[:2]
            _rgb = cv2.cvtColor(_fr, cv2.COLOR_BGR2RGB)
            _mimg = _mp_ref.Image(image_format=_mp_ref.ImageFormat.SRGB, data=_rgb)
            _res = MP_HANDS.detect(_mimg)
            _boxes = []
            if _res.hand_landmarks:
                for _lm in _res.hand_landmarks:
                    _xs = [p.x for p in _lm]; _ys = [p.y for p in _lm]
                    _x1 = max(0, int(min(_xs)*_w)); _y1 = max(0, int(min(_ys)*_h))
                    _x2 = min(_w, int(max(_xs)*_w)); _y2 = min(_h, int(max(_ys)*_h))
                    # ★ 21:25 用户要求: 手框适当放大一圈(4%→12% padding, 每边外扩)
                    _pw, _ph = (_x2-_x1)*0.12, (_y2-_y1)*0.12
                    _x1 = max(0, int(_x1-_pw)); _y1 = max(0, int(_y1-_ph))
                    _x2 = min(_w, int(_x2+_pw)); _y2 = min(_h, int(_y2+_ph))
                    if _x2-_x1 >= 15 and _y2-_y1 >= 15:
                        _boxes.append((_x1, _y1, _x2, _y2))
            # ★ 21:18 帧间EMA(治漂移, 20:33重写线程时丢失): 21点外接框对关键点抖动敏感,
            #   min/max点一跳→框边跳→漂移; 与上帧框匹配后 EMA
            #   ★ 21:59 治偶发漂移: 匹配阈值 0.6→1.0宽(检测跳变也平滑过渡, 不直接换新框)
            #   + 权重 0.55→0.65旧(更稳; 快速移动由Kalman预测兜底跟手)
            if _mp_prev_boxes:
                _out = []
                _used = set()
                for _b in _boxes:
                    _bc = ((_b[0]+_b[2])/2, (_b[1]+_b[3])/2)
                    _bw2 = _b[2] - _b[0]
                    _bi, _bd = -1, _bw2 * 1.0
                    for _i, _pb in enumerate(_mp_prev_boxes):
                        if _i in _used: continue
                        _pc = ((_pb[0]+_pb[2])/2, (_pb[1]+_pb[3])/2)
                        _d = ((_bc[0]-_pc[0])**2 + (_bc[1]-_pc[1])**2) ** 0.5
                        if _d < _bd:
                            _bd = _d; _bi = _i
                    if _bi >= 0:
                        _used.add(_bi)
                        _pb = _mp_prev_boxes[_bi]
                        # ★ 01:43 权重 0.65→0.55(治"不稳定/滞后"): 0.65旧框太重 →
                        #   手移动时框拖后腿(跟不上); 0.55 更跟手, 抖动由 Kalman 兜底
                        _out.append(tuple(int(_pb[k]*0.55 + _b[k]*0.45) for k in range(4)))
                    else:
                        _out.append(_b)
                _boxes = _out
            _mp_prev_boxes = _boxes
            with _mp_lock:
                _mp_latest_boxes = _boxes
                _mp_latest_time = time.time()
        except Exception:
            pass
if MP_HANDS is not None:   # ★★ 08-12 10:44 MediaPipe关节验证器(不产新框,防分裂): 治伸直手识别不出+框小臂
    threading.Thread(target=_mp_worker, daemon=True).start()
    print("✅ MediaPipe独立线程已启动(21点关节验证器: 治伸直手/框小臂, 卡死自动降级hand.pt)")

# ★★ 23:52 MiDaS 深度线程(空间立体感知): 每0.5s对最新帧算深度图(128x72)
#   独立线程, 不阻塞主循环; 深度图用于"手/胳膊挡头"判定(谁在前)
#   ★ 00:08 懒初始化: 线程内首次运行才加载模型(系统环境加载会卡死 → 后台线程隔离,
#   卡死只停深度线程, 主系统照常; 加载成功才启用深度判定)
def _depth_worker():
    global _depth_map, _depth_time, _depth_frame_cur, DEPTH, _depth_inited
    while True:
        time.sleep(0.5)
        try:
            if DEPTH is None and not _depth_inited:
                _depth_inited = True
                try:
                    import sys as _sys
                    import torch.hub as _hub
                    try:
                        _hub._validate_not_a_forked_repo = lambda *a, **k: None
                    except Exception:
                        pass
                    _sys.path.insert(0, r'D:/training_data/midas_src/MiDaS-master')
                    from midas.midas_net_custom import MidasNet_small
                    import torchvision.transforms as _T
                    if _os.path.exists(r"D:\training_data\midas_small.pt"):
                        _m_model = MidasNet_small(path=r"D:\training_data\midas_small.pt", features=64,
                                                  backbone='efficientnet_lite3', exportable=True,
                                                  non_negative=True, blocks={'expand': True})
                        _m_model.eval().cuda().half()
                        _m_tr = _T.Compose([_T.ToTensor(), _T.Resize((256, 256)),
                                            _T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
                        DEPTH = (_m_model, _m_tr)
                        print("✅ MiDaS 深度估计已加载(空间立体感知, 后台线程)")
                    else:
                        print("⚠️ midas_small.pt 缺失, 深度感知未启用")
                except Exception as _e:
                    print(f"⚠️ MiDaS 加载失败(不阻塞系统): {type(_e).__name__} {str(_e)[:60]}")
                continue
            if DEPTH is None:
                continue
            with _depth_lock:
                _fr = _depth_frame_cur
            if _fr is None:
                continue
            _m, _tr = DEPTH
            _rgb = _fr[:, :, ::-1].copy()
            _x = _tr(_rgb).unsqueeze(0).cuda().half()
            with torch.no_grad():
                _pred = _m(_x)
            _d = torch.nn.functional.interpolate(
                _pred.unsqueeze(1), size=(72, 128), mode='bilinear', align_corners=False,
            ).squeeze().float().cpu().numpy()
            with _depth_lock:
                _depth_map = _d
                _depth_time = time.time()
        except Exception:
            pass
if True:
    threading.Thread(target=_depth_worker, daemon=True).start()
    print("✅ MiDaS 深度线程已启动(懒初始化, 卡死不阻塞系统)")

def _capture_worker():
    """★ 17:13 帧率修复: 1080p 只出现在采集环节!
    采集 1080p(细节/清晰) → 立即降采样 720p 入队 → 主循环全程 720p 处理
    (检测 letterbox 640/画框/ROI/显示与 480p 系统同级负载 → 帧率恢复)
    720p 由 1080p 降采样得来, 清晰度仍远好于 480p 源放大"""
    while not _stop_flag.is_set():
        _ok, _fr = cap.read()
        if not _ok or _fr is None:
            time.sleep(0.01)
            continue
        if _fr.shape[1] > 1280:
            _fr = cv2.resize(_fr, (1280, 720), interpolation=cv2.INTER_AREA)
        if _frame_q.full():
            try:
                _frame_q.get_nowait()   # 丢旧帧(保最新)
            except _queue.Empty:
                pass
        _frame_q.put(_fr)

_th_cap = threading.Thread(target=_capture_worker, daemon=True)
_th_cap.start()

# ==================== ★★ 08-12 新架构初始化: 物品截图数据库 + Web帧队列 + 后台识别 ====================
from object_db import ObjectDB
_obj_db = ObjectDB()                                  # 物品事件数据库(每ID最清晰一帧)
CAP_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'alerts', 'obj_caps')
os.makedirs(CAP_DIR, exist_ok=True)
_web_queue = _queue.Queue(maxsize=2)                  # 已发布显示帧(Web MJPEG流, 丢旧保新)
_best_caps = {}                                       # (type,id) -> (clarity, ts) 每ID最清晰记录
threading.Thread(target=_analyzer_worker, daemon=True).start()   # 后台超分+香烟识别线程
from web_server import start_web_server
start_web_server(_web_queue, _obj_db, CAP_DIR)                   # Web前端(线程, 端口5050)
print("✅ 新架构: 物品检测(框变红) + 最清晰帧截图入库 + 后台超分/香烟识别 已启用")

# ★ 长记忆已取消(用户17:19): 删除 smoke_history/low_suspects — 检测框只来自本帧
while True:
    # 从采集队列取帧(500ms超时, 摄像头断帧时重连)
    try:
        frame = _frame_q.get(timeout=0.5)
    except _queue.Empty:
        frame = None
    if frame is None or frame.size == 0:
        # 连续失败 → 重型重连(摄像头被占用/驱动卡死)
        print('[重连] 摄像头持续断帧, 尝试重连...')
        _stop_flag.set()
        try:
            _th_cap.join(timeout=2)
        except Exception:
            pass
        cap.release()
        ok = False
        for attempt in range(10):
            time.sleep(1)
            try:
                cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)   # 17:03 1080p采集(实测25.6fps)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)   # 17:03 1080p
            except Exception:
                cap = cv2.VideoCapture(0)
            if cap.isOpened():
                _stop_flag.clear()
                _th_cap = threading.Thread(target=_capture_worker, daemon=True)
                _th_cap.start()
                try:
                    frame = _frame_q.get(timeout=1.0)
                except _queue.Empty:
                    frame = None
            if frame is not None and frame.size > 0:
                ok = True; break
        if not ok:
            print('[重连] 失败, 退出'); break

    for t in tracks: t[0].predict()
    for st in smoke_tracks: st[0].predict()
    for ht in hand_tracks: ht[0].predict()   # ★ 手部追踪预测(新架构)

    # ★★ 17:13 超分前置彻底禁用(卡帧元凶): Real-ESRGAN CPU 每帧 1080p 超分需 1s+
    #   之前只禁了线程/显示, 主循环这个调用一直没禁 → 每帧卡 1 秒
    #   超分不实时(实测全帧 1.1s), 实时场景不可用; 保留代码, 离线可用
    if False and _sr_model is not None:
        frame = dlss_super_resolve(frame)
        if frame is not None:
            W, H = frame.shape[1], frame.shape[0]
            # 帧尺寸变化 → 追踪器 reset(跟踪器的持久化状态按原尺寸)
            if _last_frame_size is not None and _last_frame_size != (W, H):
                try:
                    HAND.reset_tracking()
                except Exception:
                    pass
            _last_frame_size = (W, H)

    # ★★ 光线自适应暂停使用(用户要求 11:35): 全部关闭但代码保留, 可随时恢复
    #   恢复方法: 取消下行注释, 改为 frame_enh = light_adapt(frame)
    #   暂停原因: 光线锐化/过滤后香烟细节纹理不够明显, 先回退原帧对比
    #   影响: 检测用原帧(无CLAHE/Unsharp/分块补偿), 暗光/白背景增强全部失效
    frame_enh = frame
    # ★ 20:33 MediaPipe 线程取帧(21点手部检测)
    with _mp_lock:
        _mp_frame_cur = frame
    # ★ 23:52 MiDaS 深度线程取帧(空间感知)
    with _depth_lock:
        _depth_frame_cur = frame
    # ★ 18:36 手检测补充通道: pose 手腕(隔帧, 预训练零训练) — hand_v12 实拍分布偏移失败(只出巨型假框)
    #   14:26 原始机制: 预训练 pose 直接找手(手腕关键点9/10); 15:31 因头画框不准废弃, 手腕定位本身可靠
    if _gframe % 2 == 0:
        _pose_persons = detect_pose(frame)
        _pose_cache2 = _pose_persons
    else:
        _pose_persons = _pose_cache2 if '_pose_cache2' in globals() else []
    _gframe += 1
    # ★ 16:29 超分线程取帧(异步展示超分效果)
    _sr_latest_frame = frame
    # ★ 16:58 RIFE演示: 缓存最近两帧小图(后台插帧对比窗用)
    _demo_a = _demo_b
    _demo_b = cv2.resize(frame, (320, 180), interpolation=cv2.INTER_AREA) if frame is not None else None

    # --- 头检测(15:31 完全用 V12 模型, 废掉 pose): head_v12 模型检测(完整头框) ---
    #   ★ 15:51 BoT-SORT 全面接入: HEAD.track(botsort) — 模型检测 + BOTSORT追踪(ReID/GMC/低分恢复)
    #   验证: 实拍图 15/15 检出(BOTSORT 工作正常)
    #   ★ 17:31 conf 0.55→0.50: 手误检靠"头框宽高比钳制"处理(见下方), 0.55会致遮挡时头检测
    #   时有时无(忽闪), 0.50更连续稳定; 误检由BOTSORT低分恢复+追踪过滤兜底
    #   ★★ 23:26 彻查结论: BOTSORT双层(内部宽松关联+外层update)放大手/胳膊挡脸的偏框污染
    #     → 回归 V8 直接检测(单层 KalmanBox), 与 V8 逻辑一致(用户: 对比V8, 像V8一样稳)
    res = HEAD(frame, conf=0.50, iou=0.5, imgsz=640, verbose=False)
    det_heads = []
    for r in res:
        if r.boxes is None: continue
        for box in r.boxes:
            b = box.xyxy[0].cpu().numpy()
            x1, y1, x2, y2 = float(b[0]), float(b[1]), float(b[2]), float(b[3])
            bw, bh = x2-x1, y2-y1
            if bw < 10 or bh < 10: continue          # 极小下限(V8机制)
            ar = bw/bh if bh > 0 else 0
            if bw < 60:                               # 小/中头: 宽高比检查(V8机制)
                if ar < 0.5 or ar > 1.8: continue
            det_heads.append((x1, y1, x2, y2))


    # ★ 15:07 头框时间平滑(pose 关键点抖动 → EMA), 之后进追踪
    det_heads, _prev_head_boxes = smooth_boxes(det_heads, _prev_head_boxes)

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
            # ★★ 23:26 彻查: 移除 17:41 加的"中心距 2.5头宽 通道"(V8 没有此机制)
            #   它是头框"慢慢偏移"的根因: 手/胳膊挡脸偏框 IoU<0.15 但中心距<2.5头宽
            #   → 偏框匹配成功→update→头轨被逐步拖动→慢慢偏移(补丁压不住)
            #   V8 靠纯 IoU 0.15 严格匹配, 偏框天然被拒 → 头框稳定。回归 V8 逻辑
            if iou_val > best_iou: best_iou = iou_val; best_i = i
        if best_i >= 0:
            _tb = tracks[best_i][0].bbox
            _tw, _th = _tb[2]-_tb[0], _tb[3]-_tb[1]
            _nw, _nh = dh[2]-dh[0], dh[3]-dh[1]
            # ★★ 00:15 回归 V8 精神(详细对比 V8 后清理): V8 头上零防御, 靠"纯IoU严格匹配"
            #   天然拒绝遮挡偏框(IoU<0.15 不匹配→不update→头框不动)。V12 模型(head_v12)弱,
            #   手挡头边缘时偏框与旧轨部分重叠 IoU>0.15 会匹配上 → 需要唯一等价拦截:
            #   [预测一致性拦截] 新框中心 vs Kalman预测中心(已含速度外推)偏移 > 0.4头宽
            #     = 遮挡偏框 → 不 update(用预测框, 头框保持)
            #     快速转头: 预测已外推 → 新框≈预测位置 → 偏移小 → 正常 update ✅
            #   [保险] 连续拦截>20帧强制 update(防长时间遮挡预测漂移)
            #   (手挡头锁定/深度判定/尺寸钳制已删除 — 均依赖手轨或延迟高, 与 V8 行为不一致)
            _ncx3 = (dh[0]+dh[2])/2; _ncy3 = (dh[1]+dh[3])/2
            _pcx3 = (_tb[0]+_tb[2])/2; _pcy3 = (_tb[1]+_tb[3])/2
            _dist3 = ((_ncx3-_pcx3)**2 + (_ncy3-_pcy3)**2) ** 0.5
            _skip3 = getattr(tracks[best_i][0], '_skip_upd', 0)
            if _tw > 15 and _th > 15 and _dist3 > _tw * 0.4 and _skip3 < 20:
                # 遮挡偏框: 不 update, 用预测框(头框保持原位, 不飘)
                _htrk3 = tracks[best_i][0]
                _htrk3._skip_upd = _skip3 + 1
                _htrk3.lost = 0
                _htrk3.disp_bbox = _htrk3.bbox   # 硬跟预测框(无EMA滞后)
                matched_ids.add(tracks[best_i][1]); matched_det.add(j)
                continue
            tracks[best_i][0]._skip_upd = 0
            tracks[best_i][0].update(dh); matched_ids.add(tracks[best_i][1]); matched_det.add(j)
    for t in tracks:
        if t[1] not in matched_ids: t[0].lost += 1
    for j, dh in enumerate(det_heads):
        if j in matched_det: continue
        # ★★ 17:31 新建轨抑制(治头遮挡时框分裂): 与任何现有头轨中心距 < 1.5×最大头宽
        #   → 视为同一头(遮挡导致IoU小未匹配) → 不新建, 顺手喂给最近轨
        #   ★ 00:03 治"挡头时头框飘": 偏框经"顺手喂给"直接update旧轨 → 旧轨被拖走
        #     → 喂给前加偏框校验(速度自适应): 旧轨速度小+偏移大 = 偏框 → 不喂(丢弃, 不新建)
        _dup_track = False
        for _t in tracks:
            _tb = _t[0].bbox
            _d2 = ((dh[0]+dh[2])/2 - (_tb[0]+_tb[2])/2)**2 + ((dh[1]+dh[3])/2 - (_tb[1]+_tb[3])/2)**2
            _tw_max = max(dh[2]-dh[0], _tb[2]-_tb[0])
            if _d2 < (_tw_max * 1.5) ** 2:
                _dup_track = True
                if _t[1] not in matched_ids:
                    _tv = tracks[tracks.index(_t)][0].kf.statePost.flatten()
                    _tvel2 = (_tv[4]**2 + _tv[5]**2) ** 0.5
                    _tdist2 = _d2 ** 0.5
                    _tthr2 = max(_tw_max * 0.6, _tvel2 * 1.5 + 15)
                    if _tdist2 <= _tthr2:   # 偏移在速度自适应阈值内 = 正常帧 → 喂
                        tracks[tracks.index(_t)][0].update(dh)
                        matched_ids.add(_t[1])
                break
        if not _dup_track and len(tracks) < 20:
            # ★★ 17:32 新建轨互斥(治"手独立出现被误检成头"): 新头框与上帧手轨重叠
            #   ★ 22:03 收严 0.25→0.4(治"头手靠近头消失"): 手在头旁轻微重叠(0.25-0.4)不再拦头
            #     → 头轨丢失重建时不会被旁边的手挡住; 只有手基本盖住头框(IoU>0.4)才拦
            #   ⚠️ 不影响已有头轨: 手挡头前时头轨已存在, 保留(不删不压) → 挡头仍识别
            _hand_ovl = False
            for _ht in hand_tracks:
                if iou(dh, _ht[0].bbox) > 0.40:
                    _hand_ovl = True; break
            if not _hand_ovl:
                tracks.append([KalmanBox(dh), next_id]); next_id += 1
    tracks = [t for t in tracks if t[0].lost < 5]   # 头轨迹容忍(昨天配置, 恢复)

    # --- ★★ 手部检测(01:47 完全沿用 V8: hand.pt 单源检测 + V8 过滤链) ---
    #   [V8内容] hand.pt(用户实拍训练) → 尺寸过滤 → 低分池 → 肤色验证
    #     → 去重1.2 → ReID+Kalman → 低分池维持 → 局部找回 → 建轨抑制1.5 → lost<5
    #   [移除, V8没有不掺入] MediaPipe 21点主通道 / 锐化输入 / 抑制2.5动态 / lost≥5续显
    det_hands = []
    hand_low_pool = []
    if HAND_READY:
        try:
            res_h = HAND(frame, conf=0.12, iou=0.5, imgsz=640, verbose=False)
            for r in res_h:
                for box in r.boxes:
                    b = box.xyxy[0].cpu().numpy()
                    conf = float(box.conf.cpu().numpy()[0]) if box.conf is not None else 1.0
                    x1, y1, x2, y2 = float(b[0]), float(b[1]), float(b[2]), float(b[3])
                    bw, bh = x2-x1, y2-y1
                    if bw < 20 or bh < 20: continue   # 最小手框(过滤噪声)
                    if bw > W*0.7 or bh > H*0.7: continue  # 超大框(误检)
                    # ★★ 08-12 02:27 宽高比过滤(治"框成胳膊"): hand.pt 在"搜索不到手"时
                    #   常把"手+前臂"一起框(细长条)。手掌框接近方形(w/h 0.7~1.8),
                    #   手臂框细长(比例>2.2) → 拒
                    #   ★★ 08-12 10:30 2.2→4.0: 手掌伸直对准相机(水平)时是"宽长条"轮廓,
                    #     宽高比 3-4 → 旧阈值拒掉 → 手伸平识别不出。4.0 覆盖伸直手, 仍挡细长手臂
                    #   ★★ 08-12 10:41 2.2-4.0区间按面积区分"伸直手掌"vs"手+小臂":
                    #     伸直手掌: 面积小(≈头面积0.06-0.15); 手+小臂: 面积大(≈0.3头以上)
                    #     → 面积>0.30×头框 → 手+小臂(拒); 面积小 → 伸直手掌(保留)
                    _ar = max(bw, bh) / max(15.0, min(bw, bh))
                    if _ar > 4.0:
                        continue   # 细长条(手臂/物体) → 非手掌
                    if _ar > 2.2:
                        _hd_area_ref = 0
                        for _hd in det_heads:
                            _ha = (_hd[2]-_hd[0]) * (_hd[3]-_hd[1])
                            if _ha > _hd_area_ref:
                                _hd_area_ref = _ha
                        if _hd_area_ref > 0 and (bw*bh) > _hd_area_ref * 0.30:
                            continue   # 面积接近头 = 手+小臂(框到胳膊) → 拒
                    # ★ 低分候选(0.12-0.25): 先查"连续2帧确认"可否升级建轨(灵敏度)
                    #   ★★ 08-12 治"手出现识别慢": 手刚出现/运动模糊时 conf 常在
                    #   0.12-0.25, 只进池不建轨 → 要等 conf 升到 0.25+ 才出框(延迟几百ms)
                    #   [新] 上一帧同位置(<0.8手宽)也有低分候选(连续2帧) + 肤色验证
                    #        通过 → 立即升级为正常检测(建轨) → 手出现即出框
                    if conf < 0.25:
                        hand_low_pool.append((x1, y1, x2, y2, conf))
                        if _prev_low_pool:
                            for _plp in _prev_low_pool:
                                _plw = max(_plp[2]-_plp[0], bw, 20.0)
                                _pc = ((_plp[0]+_plp[2])/2 - (x1+x2)/2)**2 + ((_plp[1]+_plp[3])/2 - (y1+y2)/2)**2
                                if _pc < (_plw * 0.8) ** 2:
                                    if hand_skin_ok(frame, x1, y1, x2, y2, W, H):
                                        det_hands.append((x1, y1, x2, y2))
                                    break
                        continue
                    # ★★ 肤色验证(V8机制, 治椅子/木纹/红色物体误判)
                    #   ★ 高置信(≥0.60)跳过(hand.pt 高置信误杀少)
                    if conf < 0.60:
                        if not hand_skin_ok(frame, x1, y1, x2, y2, W, H):
                            continue
                    det_hands.append((x1, y1, x2, y2))
        except Exception:
            pass
    # ★★ 08-12 10:44 MediaPipe关节验证器(治伸直手识别不出+框小臂):
    #   对宽高比>2.2 的可疑 hand.pt 框用 MediaPipe 21点验证:
    #     找到匹配 → 替换为 MediaPipe 手掌框(精确, 天然不含小臂) → 治伸直手+治框小臂
    #     未匹配 → 拒该框(可能是小臂误检) → 治框小臂
    #   ★★ 08-12 10:52 匹配放宽(IoU>0.2→IoU>0.1或中心距<0.8手宽):
    #     手肘框(手+小臂)中心在手臂上, 与手掌框IoU可能仅0.1-0.2 → 原0.2漏配 → 手肘框没被替换
    #   正常宽高比(≤2.2)直接用 hand.pt 框(MediaPipe 外接框可能偏小, 不用)
    #   ★★ 08-12 10:52 补充通道(治"半遮挡手识别不出"): MediaPipe检出的手掌框
    #     未被 hand.pt 覆盖(hand.pt对部分遮挡手检不出) → 直接补充为检测;
    #     与已有框"中心距<1.2手宽或IoU>0.3"=同手 → 替换为MediaPipe框(精确, 防分裂)
    #   卡死/超时(_mpb空或时间过期)→ 不处理, 降级纯hand.pt
    try:
        with _mp_lock:
            _mpb = list(_mp_latest_boxes)
        if _mpb and (time.time() - _mp_latest_time) < 0.5:
            _new_det = []
            for _dh in det_hands:
                _dhw = _dh[2]-_dh[0]; _dhh = _dh[3]-_dh[1]
                _dh_ar = max(_dhw, _dhh) / max(15.0, min(_dhw, _dhh))
                if _dh_ar <= 2.2:
                    _new_det.append(_dh); continue  # 正常宽高比 → 直接用
                _matched_mp = None
                for _mb in _mpb:
                    _dwi = max(_dhw, _dhh, 20.0)
                    _mdc = ((_dh[0]+_dh[2])/2-(_mb[0]+_mb[2])/2)**2 + \
                           ((_dh[1]+_dh[3])/2-(_mb[1]+_mb[3])/2)**2
                    if iou(_dh, _mb) > 0.1 or _mdc < (_dwi * 0.8) ** 2:
                        _matched_mp = _mb; break
                if _matched_mp is not None:
                    _new_det.append(_matched_mp)  # 替换为精确手掌框(治伸直手+治小臂)
                # else: 拒(不加入) — MediaPipe有结果但无匹配=小臂误检
            det_hands = _new_det
            # 补充通道: MediaPipe 检出的手掌框(部分遮挡/伸直手 hand.pt 检不出的)
            for _mb in _mpb:
                _mb_dup = False
                for _k, _dh in enumerate(det_hands):
                    _mwi = max(_mb[2]-_mb[0], _dh[2]-_dh[0], 20.0)
                    _mdc = ((_mb[0]+_mb[2])/2-(_dh[0]+_dh[2])/2)**2 + \
                           ((_mb[1]+_mb[3])/2-(_dh[1]+_dh[3])/2)**2
                    if _mdc < (_mwi * 1.2) ** 2 or iou(_mb, _dh) > 0.3:
                        det_hands[_k] = _mb   # 同手 → 替换为MediaPipe精确框(防分裂)
                        _mb_dup = True
                        break
                if not _mb_dup:
                    det_hands.append(_mb)     # 新增(半遮挡手也识别)
    except Exception:
        pass
    # ★ 去重(★ 08-12 只按IoU>0.3合并同手双检框): 移除"中心距<1.2手宽"合并
    #   [旧] 中心距<1.2手宽即合并 → 两只手平行靠近时中心距近但框不重叠 → 第二只手框被合并掉
    #   [新] 只合并"真正重叠"的双检框(IoU>0.3) → 两手靠近各自框保留 → 双手都有框
    det_hands_dedup = []
    for _dh in sorted(det_hands, key=lambda b: -(b[2]-b[0])*(b[3]-b[1])):
        _dup = False
        for _dh2 in det_hands_dedup:
            if iou(_dh, _dh2) > 0.3:
                _dup = True; break
        if not _dup:
            det_hands_dedup.append(_dh)
    det_hands = det_hands_dedup
    # ★★ 20:15 ReID: 为每个手检测框提取外观签名(HSV直方图), 供追踪匹配
    _hand_sigs = [extract_sig(frame, _dh) for _dh in det_hands]
    # ★★ 08-12 手部关联追踪重构(治"遮挡/靠近时框漂移、框错、ID交换"):
    #   [位置为主] IoU + 中心距惩罚(>1.5手宽惩罚, >2.5手宽拒绝) — 遮挡/误检跳飞不入轨
    #   [受限外观] 检测框同时接近≥2个活跃轨(两手靠近) → 禁用外观通道
    #              (两手肤色一致, 外观=赌博, 是ID交换元凶; 单手快速移动仍启用救回)
    #   [全局匹配] 分数降序贪心(一轨一框, 一框一轨) → 防两手靠近时轨交换手
    #   [丢失轨隔离] lost≥2的轨不参与主匹配(留给局部找回通道), 不抢新检测
    h_matched_ids = set(); h_matched_det = set()
    _h_scores = []
    for j, dh in enumerate(det_hands):
        _dhc = ((dh[0]+dh[2])/2, (dh[1]+dh[3])/2)
        _dhw = max(20.0, dh[2]-dh[0])
        for i, ht in enumerate(hand_tracks):
            if ht[0].lost >= 2: continue      # 丢失2帧轨不抢主匹配
            _hb = ht[0].bbox
            _kfp = ht[0].kf.statePre.flatten()   # ★ 08-12 预测中心(含速度外推)
            _pb = (_kfp[0]-_kfp[2]/2, _kfp[1]-_kfp[3]/2, _kfp[0]+_kfp[2]/2, _kfp[1]+_kfp[3]/2)
            # ★★ 08-12 预测优先匹配(治"两手晃动/交换/靠近时框乱跟"):
            #   [旧] 主要用"当前框IoU" → 两手靠近时检测框互相重叠 → IoU无法区分 → 换手/抖动
            #   [新] 预测框IoU(速度外推后"手应该在哪")权重略高 → 两手交叉时检测框与
            #        "本手预测"更贴合 → 匹配正确, 不换手
            iou_cur = iou(_hb, dh)
            iou_pred = iou(_pb, dh)
            iou_val = max(iou_cur, iou_pred * 1.15)
            # ★ 位置突变惩罚: 检测框中心与【预测中心】距离归一化(>1.5手宽惩罚, >2.5拒)
            _dist = ((_dhc[0]-_kfp[0])**2 + (_dhc[1]-_kfp[1])**2) ** 0.5
            _w_ref = max(_dhw, _hb[2]-_hb[0], 20.0)
            _dr = _dist / _w_ref
            if _dr > 2.5:
                continue
            _pos_pen = max(0.0, _dr - 1.5) * 0.5
            # ★ 受限外观: 该检测框周围1.5手宽内的活跃轨数量
            _near_n = 0
            for _ht2 in hand_tracks:
                if _ht2[0].lost >= 2: continue
                _hb2 = _ht2[0].bbox
                _d2 = ((_dhc[0]-(_hb2[0]+_hb2[2])/2)**2 + (_dhc[1]-(_hb2[1]+_hb2[3])/2)**2) ** 0.5
                if _d2 < _w_ref * 1.5:
                    _near_n += 1
            _app = 0.0
            # ★★ 08-12 外观收严(治"两手靠近→第二只手没框"): 画面≥2检测框(两手
            #   场景) → 禁用外观通道。两手肤色一致, 外观会把第二只手的检测框吸给
            #   第一只手轨(IoU低但外观像也匹配) → 第二只手永远建不了轨
            if len(det_hands) <= 1 and _near_n <= 1 and ht[0].sig is not None and _hand_sigs[j] is not None:
                _corr, _size_ok = sig_similarity(ht[0].sig, _hand_sigs[j])
                if _size_ok:
                    _app = max(0.0, _corr)   # 相关性[-1,1] → [0,1]
            # ★★ 08-12 匹配惯性(治"两手晃动/靠近时框跳变"): 与上帧本轨匹配的检测
            #   框位置接近 → 加分。两手各自晃动时检测框跟随各自的手 → 轨被惯性
            #   粘住正确的手, 即使两框靠近也不互相抢
            _inertia = 0.0
            _ld = getattr(ht[0], '_last_det', None)
            if _ld is not None:
                _ldd = ((_dhc[0]-_ld[0])**2 + (_dhc[1]-_ld[1])**2) ** 0.5
                _inertia = max(0.0, 1.0 - _ldd / (_w_ref * 2.0)) * 0.5
            # ★★ 08-12 距离得分: 检测框离预测中心越近得分越高(快速移动IoU≈0时的
            #   位置依据, 防快速晃动帧"无匹配"丢轨)
            _dist_score = max(0.0, 1.0 - _dr / 1.5) * 0.35
            _score = iou_val + _dist_score + _app * 0.8 - _pos_pen + _inertia
            if _score > 0.15:
                _h_scores.append((_score, i, j))
    for _msc, best_i, j in sorted(_h_scores, key=lambda x: -x[0]):
        if j in h_matched_det or hand_tracks[best_i][1] in h_matched_ids: continue
        dh = det_hands[j]
        # ★★ 02:27 手摸头防护(治"摸头时手被头发遮挡→框变形/识别不出"): 手摸头时
        #   手框与头轨重叠, hand.pt 框被头发/脸污染(尺寸骤变)
        #   [条件] 手框与头轨重叠(IoU>0.1) 且 检测框尺寸与轨偏差>40% → 污染框
        #   [行为] 用预测框维持(不update) → 框稳定不随头发变形
        #   [不误伤] 手挡头(正常手势)尺寸不骤变 → 正常update; 手在头旁不重叠 → 正常
        _tb_h = hand_tracks[best_i][0].bbox
        _tw_h = _tb_h[2]-_tb_h[0]; _th_h = _tb_h[3]-_tb_h[1]
        _nw_h = dh[2]-dh[0]; _nh_h = dh[3]-dh[1]
        _size_jump = (_tw_h > 15 and _th_h > 15 and
                      (_nw_h > _tw_h*1.4 or _nh_h > _th_h*1.4 or _nw_h < _tw_h*0.6 or _nh_h < _th_h*0.6))
        _near_head2 = False
        if _size_jump:
            for _t in tracks:
                if iou(dh, _t[0].bbox) > 0.1:
                    _near_head2 = True; break
        # ★★ 08-12 合并框判据修正(治"两手握紧/靠近→框挤压暴涨/双框同一手"):
        #   [旧] IoU>0.3 → 合并框很大时与轨IoU反而<0.3 → 漏判 → 轨被拉向中间 ❌
        #   [新] "检测框覆盖另一活跃轨中心(±0.25手宽余量)" 或 IoU>0.3 → 合并框
        #   [行为] 用预测框维持(不吸收暴涨尺寸) → 框不被挤大/拉向中间
        #   [恢复] 手分开后检测框分离 → 自动恢复正常跟踪
        _merge_box = False
        for _ht2 in hand_tracks:
            if _ht2 is hand_tracks[best_i] or _ht2[1] in h_matched_ids: continue
            if _ht2[0].lost >= 2: continue          # 对方已丢失(领地空出) → 不算合并
            _b2 = _ht2[0].bbox
            _c2x = (_b2[0]+_b2[2])/2; _c2y = (_b2[1]+_b2[3])/2
            _pad = _tw_h * 0.25
            if (dh[0]-_pad) <= _c2x <= (dh[2]+_pad) and (dh[1]-_pad) <= _c2y <= (dh[3]+_pad):
                _merge_box = True; break
            # ★★ 08-12 10:44 IoU 0.3→0.4(收紧防临界抖动)+ 加退出保护(hysteresis):
            #   原 0.3 临界时反复触发/退出 → 合并框位置在"完全预测/正常update"间
            #   切换 → 抖且大小不稳定。0.4 明确重叠才触发, 退出后轨对象保护3帧
            #   仍走完全预测, 4帧后才恢复正常update → 临界不抖
            if iou(dh, _b2) > 0.4:
                _merge_box = True; break
        _merge_protect = getattr(hand_tracks[best_i][0], '_merge_protect', 0)
        if (_size_jump and _near_head2) or _merge_box or _merge_protect > 0:
            # ★★ 08-12 位置-尺寸分离更新(治"手碰头/手碰手时挤压排斥漂移"):
            #   [旧] 维持预测框(不吸收任何测量) → 手长期靠Kalman外推 → 框被头/
            #        另一只手"顶住"(排斥感) + 预测误差累积(漂移) ❌
            #   [新] 位置半跟随检测(检测中心与KF预测中心取平均: 手动框动不被顶住,
            #        污染框也不完全拉飞), 尺寸保持轨值(不被头/合并框撑大) →
            #        框自然跟随且大小稳定 → 无排斥无挤压无漂移
            #   ★★ 08-12 10:30 合并框时位置完全用预测中心(不漂向中间, 治"框抖动/大小不稳定"):
            #     两手重叠/合并时, 检测框中心在两手中间, 半跟随会让框每帧微微漂向中间
            #     → 抖动+大小被合并框影响 → 完全用预测中心 + 尺寸保持 = 稳定不动 ✅
            #   ★★ 08-12 10:44 退出保护(hysteresis): merge_box 触发后设 _merge_protect=3,
            #     后续3帧即使 IoU<0.4 不触发, 也走完全预测 → 临界不抖, 第4帧才恢复
            if _merge_box:
                hand_tracks[best_i][0]._merge_protect = 3
            elif _merge_protect > 0:
                hand_tracks[best_i][0]._merge_protect = _merge_protect - 1
            hand_tracks[best_i][0].lost = 0
            _kfp2 = hand_tracks[best_i][0].kf.statePre.flatten()
            if (_merge_box or _merge_protect > 0) and not (_size_jump and _near_head2):
                # 合并框/退出保护期: 位置完全用预测中心(稳定)
                _cx2 = _kfp2[0]; _cy2 = _kfp2[1]
            else:
                # 手摸头: 半跟随(检测与预测平均, 跟手走)
                _cx2 = ((dh[0]+dh[2])/2 + _kfp2[0]) * 0.5
                _cy2 = ((dh[1]+dh[3])/2 + _kfp2[1]) * 0.5
            _clx = _cx2 - _tw_h*0.5; _cly = _cy2 - _th_h*0.5
            _crx = _cx2 + _tw_h*0.5; _cry = _cy2 + _th_h*0.5
            hand_tracks[best_i][0].update((_clx, _cly, _crx, _cry))
        else:
            # ★★ 08-12 手肘框防护(治"剧烈运动/变换角度时框到手肘"): hand.pt 会
            #   把"手+前臂"一起框(尺寸明显大于手掌), 中心偏向前臂 → 轨被拉到手肘
            #   [判据] 检测框面积>轨1.5x 且 与预测中心偏移>0.8手宽 → 手肘/污染框
            #   [行为] 半跟随(位置取检测与预测平均, 尺寸保持轨) → 框不漂到手肘
            #   位置突变>2.0手宽(跳飞误检)也走半跟随(不硬维持, 避免长期不吸收漂移)
            _kfp = hand_tracks[best_i][0].kf.statePre.flatten()
            _mcd = ((dh[0]+dh[2])/2 - _kfp[0])**2 + ((dh[1]+dh[3])/2 - _kfp[1])**2
            _ww2 = max(_tw_h, _nw_h, 20.0)
            _elbow_box = ((_nw_h*_nh_h) > max(20.0, _tw_h*_th_h)*1.5) and _mcd > (_ww2*0.8)**2
            if _elbow_box or _mcd > (_ww2*2.0)**2:
                hand_tracks[best_i][0].lost = 0
                _cx2 = ((dh[0]+dh[2])/2 + _kfp[0]) * 0.5
                _cy2 = ((dh[1]+dh[3])/2 + _kfp[1]) * 0.5
                _clx = _cx2 - _tw_h*0.5; _cly = _cy2 - _th_h*0.5
                _crx = _cx2 + _tw_h*0.5; _cry = _cy2 + _th_h*0.5
                hand_tracks[best_i][0].update((_clx, _cly, _crx, _cry))
            else:
                hand_tracks[best_i][0].update(dh)
        h_matched_ids.add(hand_tracks[best_i][1]); h_matched_det.add(j)
    for ht in hand_tracks:
        if ht[1] not in h_matched_ids: ht[0].lost += 1
    # ★★ 20:12 低分池维持(快速移动不消失): 未匹配的手轨, 用低分候选(conf 0.12-0.25)
    #   中与预测位置重叠的 → update + lost 复位(运动模糊帧的弱检测也能续轨)
    if hand_low_pool:
        for _ht in hand_tracks:
            if _ht[1] in h_matched_ids: continue
            if _ht[0].lost <= 0: continue      # 只救"刚丢失"的轨
            _pb = _ht[0].bbox
            _best_low = None; _best_iou_low = 0.20
            for _lp in hand_low_pool:
                _io = iou(_pb, _lp[:4])
                if _io > _best_iou_low:
                    # ★★ 08-12 低分池互斥(治"两手握紧→双框同一手"): 低分候选若与
                    #   另一活跃手轨重叠(IoU>0.3) → 那是另一只手的框(或合并框),
                    #   吸过来=两个轨粘同一只手 → 跳过该候选
                    _steal = False
                    for _ht2 in hand_tracks:
                        if _ht2 is _ht or _ht2[1] in h_matched_ids: continue
                        if _ht2[0].lost >= 2: continue
                        if iou(_ht2[0].bbox, _lp[:4]) > 0.3:
                            _steal = True; break
                    if _steal: continue
                    # ★★ 08-12 中心距验证(治"两手一前一后打圈→丢失轨乱飘"): 丢失轨
                    #   只接受"离预测中心≤0.3手宽"的低分候选(原0.5太松, 前面手弱检测
                    #   落在后面手预测附近会被误喂 → 框被带偏乱飘)
                    _lpc = ((_lp[0]+_lp[2])/2 - (_pb[0]+_pb[2])/2)**2 + ((_lp[1]+_lp[3])/2 - (_pb[1]+_pb[3])/2)**2
                    _lpw = max(_pb[2]-_pb[0], _lp[2]-_lp[0], 20.0)
                    if _lpc > (_lpw * 0.3) ** 2:
                        continue
                    _best_iou_low = _io; _best_low = _lp
            if _best_low is not None:
                _ht[0].update(_best_low[:4])
                _ht[0].lost = 0
                h_matched_ids.add(_ht[1])
    # ★★ 20:12 局部找回(快速移动不消失): 仍未匹配的轨(lost≥1), 预测位置裁剪放大再检
    #   仅对"追踪中的手"局部搜索, 不产生全局新检测 → 不会误检爆炸
    #   ★ 02:21 治"遮挡后框乱飘": 找回框 conf 0.15→0.30 + 中心距验证(≤0.5手宽)
    #     [旧] 遮挡处重检到背景物体(conf≥0.15) → update → 轨被带偏 → 乱飘 ❌
    #     [新] 只认"高置信且离预测位置近"的找回 → 误检不入轨 ✅
    for _ht in hand_tracks:
        if _ht[1] in h_matched_ids: continue
        if _ht[0].lost > 2: continue      # 最多救2帧(超时放弃, 防幽灵)
        _pb = _ht[0].bbox
        _hw3, _hh3 = _pb[2]-_pb[0], _pb[3]-_pb[1]
        if _hw3 < 20 or _hh3 < 20: continue
        _ex1, _ey1 = max(0, int(_pb[0]-_hw3*0.6)), max(0, int(_pb[1]-_hh3*0.6))
        _ex2, _ey2 = min(W, int(_pb[2]+_hw3*0.6)), min(H, int(_pb[3]+_hh3*0.6))
        if _ex2-_ex1 < 24 or _ey2-_ey1 < 24: continue
        try:
            _crop3 = frame[_ey1:_ey2, _ex1:_ex2]
            _rr3 = HAND(_crop3, conf=0.25, iou=0.5, imgsz=480, verbose=False)   # ★ 08-12 0.30→0.25(模糊帧弱检也能找回)
            _found3 = None
            for _r in _rr3:
                for _b in _r.boxes:
                    _bx1, _by1, _bx2, _by2 = _b.xyxy[0].cpu().numpy()
                    _sc3 = 480.0 / max(_ex2-_ex1, _ey2-_ey1)
                    _gx1 = _ex1 + _bx1/_sc3; _gy1 = _ey1 + _by1/_sc3
                    _gx2 = _ex1 + _bx2/_sc3; _gy2 = _ey1 + _by2/_sc3
                    if _gx2-_gx1 < 15 or _gy2-_gy1 < 15: continue
                    # ★ 02:21 中心距验证: 找回框须离预测位置 ≤0.5手宽(否则是背景误检)
                    #   ★ 08-12 0.8→0.5手宽: 两手一前一后时找回常命中"前面手"的检测
                    #     (快速移动消失由主匹配的"距离得分+惯性"兜底, 找回可收紧防乱飘)
                    #   ★★ 08-12 10:44 0.5→0.3手宽: 更严, 前面手检测不被误救(根治乱飘)
                    _gc = ((_gx1+_gx2)/2 - (_pb[0]+_pb[2])/2)**2 + ((_gy1+_gy2)/2 - (_pb[1]+_pb[3])/2)**2
                    if _gc > (_hw3 * 0.3) ** 2:
                        continue
                    _found3 = (_gx1, _gy1, _gx2, _gy2)
                    break
                if _found3 is not None: break
            if _found3 is not None:
                # ★★ 08-12 找回互斥(治"两手靠近→另一只手被抢"): 找回框若与另一
                #   活跃手轨重叠(IoU>0.3) → 那是另一只手的检测(局部重检常命中
                #   合并框/对方的手) → 放弃找回, 本轨正常丢失(防两个轨粘同一只手)
                _steal2 = False
                for _ht2 in hand_tracks:
                    if _ht2 is _ht or _ht2[1] in h_matched_ids: continue
                    if _ht2[0].lost >= 2: continue
                    if iou(_ht2[0].bbox, _found3) > 0.3:
                        _steal2 = True; break
                if not _steal2:
                    _ht[0].update(_found3)
                    _ht[0].lost = 0
                    h_matched_ids.add(_ht[1])
        except Exception:
            pass
    for j, dh in enumerate(det_hands):
        if j not in h_matched_det:
            # ★★ 19:54 新建轨抑制(治"手移动→方框分裂"): 新检测框与任何现有手轨
            #   中心距 < 1.5×最大手宽 → 视为同一只手(运动模糊导致未匹配) → 不新建
            #   (否则快速移动时 IoU 小未匹配 → 新建第二条轨 → 双框)
            #   ★ 01:47 完全沿用V8: 固定 1.5手宽(19:54 V8 原版), 移除 2.5/动态0.8
            _suppress_r = 1.5
            _dup_track = False
            # ★★ 08-12 取消"手间距离限制"(治"两手靠近→其中一只手没框"):
            #   [旧] 新检测框与任何手轨中心距<1.5手宽 → 视为同手 → 抑制建轨
            #        → 两手靠近时第二只手永远建不了轨 ❌
            #   [新] 若该检测框与画面中【另一检测框】中心距<1.5手宽(两只手靠在一起,
            #        且非同一只手——同手双检已被IoU去重合并) → 突破抑制, 第二只手
            #        立即建轨出框; 只有单独一个检测框时(快速移动/同手)才保留抑制
            _near_other_det = False
            for _dk, _dh2 in enumerate(det_hands):
                if _dk == j: continue
                _d2d = ((dh[0]+dh[2])/2-(_dh2[0]+_dh2[2])/2)**2 + ((dh[1]+dh[3])/2-(_dh2[1]+_dh2[3])/2)**2
                if _d2d < (max(dh[2]-dh[0], _dh2[2]-_dh2[0], 20.0) * 1.5) ** 2:
                    _near_other_det = True; break
            if not _near_other_det:
                for _ht in hand_tracks:
                    if _ht[0].lost >= 1: continue   # ★★ 08-12 丢失轨不被顺手更新(治乱飘: 错误检测喂给丢失轨 → 框被带偏)
                    _hb = _ht[0].bbox
                    _d2 = ((dh[0]+dh[2])/2 - (_hb[0]+_hb[2])/2)**2 + ((dh[1]+dh[3])/2 - (_hb[1]+_hb[3])/2)**2
                    _hw_max = max(dh[2]-dh[0], _hb[2]-_hb[0])
                    if _d2 < (_hw_max * _suppress_r) ** 2:
                        _dup_track = True
                        # ★ 顺手把它匹配给最近的轨(加速收敛, 框立刻跟上手)
                        # ★★ 08-12 顺手更新防合并框: 检测框若覆盖其他活跃轨中心(±0.25手宽)
                        #   → 是合并框/另一只手 → 不顺手更新(否则绕过合并框保护把轨拉偏)
                        if _ht[1] not in h_matched_ids:
                            _steal3 = False
                            _hba = _ht[0].bbox
                            _hw_ref = max(dh[2]-dh[0], _hba[2]-_hba[0], 20.0)
                            for _ht3 in hand_tracks:
                                if _ht3 is _ht or _ht3[1] in h_matched_ids: continue
                                if _ht3[0].lost >= 2: continue
                                _b3 = _ht3[0].bbox
                                _c3x = (_b3[0]+_b3[2])/2; _c3y = (_b3[1]+_b3[3])/2
                                _pad3 = _hw_ref * 0.25
                                if (dh[0]-_pad3) <= _c3x <= (dh[2]+_pad3) and (dh[1]-_pad3) <= _c3y <= (dh[3]+_pad3):
                                    _steal3 = True; break
                            if not _steal3:
                                hand_tracks[hand_tracks.index(_ht)][0].update(dh)
                                h_matched_ids.add(_ht[1])
                        break
            if not _dup_track and len(hand_tracks) < 10:
                # ★ ReID: 新建轨携带外观签名(后续匹配用)
                hand_tracks.append([KalmanBox(dh, pn=0.50, mn=0.05, sig=_hand_sigs[j] if j < len(_hand_sigs) else None),
                                    next_id + 1000]); next_id += 1
    # ★ 01:47 完全沿用V8: 手轨丢失容忍 lost<5(15:35 V8原版)
    hand_tracks = [ht for ht in hand_tracks if ht[0].lost < 5]
    # ★★ 08-12 记录本帧低分候选(供下一帧"连续2帧确认升级建轨")
    _prev_low_pool = [lp[:4] for lp in hand_low_pool]
    # ★ 17:31 手/头互斥已移除(用户要求: 手挡头前时头仍需识别)
    #   手挡头时两框都保留(互斥会删真头, 不可取); 头稳定性靠下方"建轨抑制+尺寸钳制"解决

    # ==================== ★★ 08-12 摒弃实时香烟检测(新架构: 后台静态识别) ====================
    #   实时只做"头/手框物品检测"(有物→框变红); 香烟识别移到后台对"最清晰截图"超分+识别
    _light_mask = np.zeros(frame.shape[:2], dtype=bool)   # 烟检停用 → 光源掩码置空(显示层不标紫)
    if False:
        # --- V22 全屏烟检 + 智能过滤(距离机制) + ByteTrack低分池 ---
        raw_smoke = []
        low_pool = []   # ByteTrack 低分候选池 (被conf过滤但≥0.15, 仅用于维持已有轨迹)
        head_list = [(t[0].bbox, t[0].bbox[2]-t[0].bbox[0]) for t in tracks
                     if (t[0].bbox[2]-t[0].bbox[0]) > 0 and (t[0].bbox[3]-t[0].bbox[1]) > 0]
        # ★ 强光源mask预计算(每帧1次): 原帧灰度>235 = 过曝白光区(灯具/强光源/光晕)
        #   真烟不会出现在纯过曝白光区, 检测框内过曝占比高 → 直接拒(治强光源误检)
        _over_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _over_mask = _over_gray > 235
        # ★★ 15:23 大面积光源掩码(区域级): 窗/灯/过曝发光区 vs 白色小块(烟/白物)
        #   过滤链中: 候选框内"光源占比高" → 拒(只留白色内容给检测)
        _light_mask = light_source_mask(_over_gray)
        # ★★ 新架构(用户要求): 香烟检测注意力只放在"头部+手部"
        #   在头框/手框的扩张区域内找烟(烟常在嘴边/手持/耳后), 大幅减少全屏误检(椅子/门框/桌腿)
        #   头框扩张: 上下左右各扩(脸前/嘴前/耳后) | 手框扩张: 更大(手持烟/挥烟)
        #   ★ 稳定性(18:08): focus_regions 改用【平滑追踪框 tracks/hand_tracks】
        #     本帧检测框每帧跳几px → ROI区域跟着抖 → 烟框"一抖一抖"
        #     追踪框经Kalman平滑, 区域稳定 → 烟框稳定
        #   ★ 严格性兜底: "交集判定"(候选烟必须与本帧det_heads/det_hands有交集)不变,
        #     无头无手(本帧) → 候选全拒 → 依然"无头无手绝不识别"
        focus_regions = []   # (rx1, ry1, rx2, ry2, kind)  kind='head'/'hand'
        for t in tracks:                                  # 平滑头追踪框
            hx1, hy1, hx2, hy2 = t[0].bbox
            hw_, hh_ = hx2-hx1, hy2-hy1
            focus_regions.append((max(0, hx1-hw_*0.6), max(0, hy1-hh_*0.9),
                                  min(W, hx2+hw_*0.6), min(H, hy2+hh_*0.6), 'head'))
        for ht in hand_tracks:                            # 平滑手追踪框
            hx1, hy1, hx2, hy2 = ht[0].bbox
            hw_, hh_ = hx2-hx1, hy2-hy1
            focus_regions.append((max(0, hx1-hw_*1.2), max(0, hy1-hh_*0.6),
                                  min(W, hx2+hw_*1.2), min(H, hy2+hh_*1.6), 'hand'))
        # 无头无手: focus_regions为空 → 本帧不做烟检测(严格限定区域, 不识别区域外)
        # ★★ 新架构(用户需求): ROI 裁剪放大独立检测 — "放大方框看附近小区域有没有香烟"
        #   头框/手框出现 → 裁剪扩张区域ROI → 放大到768 → SMOKE批量推理 → 框坐标换算回全图
        #   相比"全图1024+中心过滤": 区域小图被放大, 有效分辨率翻倍, 小烟/被挡一半的烟更好检出
        #   ★性能拉满(18:05): 所有ROI收集后一次batch前向(替代逐区域串行), 频率翻倍
        #   ★质量提升: 放大目标640→768(细节+20%), 区域上限4→6(多人)
        #   无头无手 → sr空 → 烟识别关闭
        sr = []
        if focus_regions:
            # ★ 16:29 帧率优化: 烟隔帧推理(烟移动慢, 追踪由BOTSORT预测续帧) + 缓存上一帧结果
            if _gframe % 2 == 0:
                try:
                    _rr_all = SMOKE.track(frame_enh, persist=True, tracker='botsort.yaml',
                                          conf=0.12, iou=0.5, imgsz=640, verbose=False)
                    _smoke_det_cache = []
                    for _rrc in _rr_all:
                        if _rrc.boxes is None: continue
                        for _b2 in _rrc.boxes:
                            _bcx = float(_b2.conf[0])
                            _bx1, _by1, _bx2, _by2 = _b2.xyxy[0].cpu().numpy()
                            _smoke_det_cache.append((_bx1, _by1, _bx2, _by2, _bcx))
                except Exception:
                    _smoke_det_cache = []
                for _s in _smoke_det_cache:
                    sr.append(_RoiRes([_RoiBox(_s[0], _s[1], _s[2], _s[3], _s[4])]))
            else:
                for _s in (_smoke_det_cache if '_smoke_det_cache' in dir() else []):
                    sr.append(_RoiRes([_RoiBox(_s[0], _s[1], _s[2], _s[3], _s[4])]))
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
                for (frx1, fry1, frx2, fry2, _) in focus_regions:
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
                # ★ 18:10 防断续: 交集判定用【本帧检测 + 平滑追踪框】
                #   本帧头/手检测帧间波动(漏1-2帧) → 追踪框(lost<5)兜底, 烟不闪断
                #   真消失(连续丢帧, 追踪框也清) → 交集失败 → 烟框消失(不残留)
                for (_tbx1, _tby1, _tbx2, _tby2) in list(det_hands) + [ht[0].bbox for ht in hand_tracks]:
                    if iou((x1, y1, x2, y2), (_tbx1, _tby1, _tbx2, _tby2)) > 0.0:
                        _touch_hand = True; break
                for (_tbx1, _tby1, _tbx2, _tby2) in list(det_heads) + [t[0].bbox for t in tracks]:
                    if iou((x1, y1, x2, y2), (_tbx1, _tby1, _tbx2, _tby2)) > 0.0:
                        _touch_head = True; break
                if not (_touch_hand or _touch_head):
                    continue   # ★ 严格: 红框未与蓝/绿框有交集 → 一律拒(不管多像烟)

                # ===== ★★ 上下文激活(用户18:48): 已激活的手/头框内 → 直接归类为烟 =====
                #   某手/头框ID内确认过一次烟 → 激活 → 框内"非手自带物体"直接算烟
                #   应对: 烟转向时外形剧变(像素重排)导致检测框消失/漏判
                _in_activated = False
                for _ht in hand_tracks:
                    if _ht[0].smoke_activated and iou((x1, y1, x2, y2), _ht[0].bbox) > 0.05:
                        _in_activated = True; break
                for _t in tracks:
                    if _t[0].smoke_activated and iou((x1, y1, x2, y2), _t[0].bbox) > 0.05:
                        _in_activated = True; break
                if _in_activated:
                    raw_smoke.append((float(x1), float(y1), float(x2), float(y2)))
                    continue   # ★ 已激活框内 → 直接归类, 不再推导/过滤

                # ===== ★ 快速放行(用户18:48): 在框内 + 满足任一充分条件 → 直接归类 =====
                #   充分条件: 细长(aspect 1.8-6) + 有纹理(非纯色) → 就是烟, 跳过复杂过滤
                #   (门框/椅背=纯色细长 → 低纹理排除; 手机近方形 → 不触发, 走原过滤链)
                if 1.8 <= aspect <= 6.0 and area_ratio < 0.18:
                    _qcrop = frame[max(0,int(y1)):min(H,int(y2)), max(0,int(x1)):min(W,int(x2))]
                    if _qcrop.size >= 16:
                        _qg = cv2.cvtColor(_qcrop, cv2.COLOR_BGR2GRAY)
                        _lap_var = cv2.Laplacian(_qg, cv2.CV_64F).var()
                        if _lap_var >= 8.0:
                            raw_smoke.append((float(x1), float(y1), float(x2), float(y2)))
                            continue   # ★ 细长+有纹理+在框内 → 直接归类为烟

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
                        # ★★ 15:23 大面积光源过滤(区域级): 候选框内"光源掩码"占比>35%
                        #   → 大面积光源(窗/灯/过曝区) → 拒; 白色小块(烟/白物)不在掩码 → 放行
                        _lr = _light_mask[oy1:oy2, ox1:ox2]
                        if float(_lr.mean()) > 0.35:
                            continue   # 框内大面积光源主导 → 过滤掉光源, 不是白色内容

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

                # ===== ★★ 手持区域肤色互斥(用户19:44: 持物是烟检测的开关) =====
                #   候选与手框"部分重叠"(IoU 0.03-0.4) = 潜在手持烟 → 必须"非肤色"
                #   空手: 手指/手掌被当烟 → 框内肤色高(≥0.45) → 拒(未持物绝不框烟)
                #   持物: 真烟白色纸+棕滤嘴 → 框内肤色低(<0.45) → 放行(等于"拍到真烟才持物")
                _hand_partial = False
                for (_hbx1, _hby1, _hbx2, _hby2) in list(det_hands) + [ht[0].bbox for ht in hand_tracks]:
                    _hiou = iou((x1, y1, x2, y2), (_hbx1, _hby1, _hbx2, _hby2))
                    # ★ 19:47 扩大: IoU>0.01(任何交集) 即触发检查, 防红框伸出框外绕过互斥
                    if 0.01 < _hiou <= 0.4:
                        _hand_partial = True; break
                if _hand_partial and _skin >= 0.45:
                    continue   # ★ 手框内疑似烟但肤色主导 = 手指/手掌被当烟 → 拒(未持物不框烟)

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

        # ★★ 18:48 上下文激活: 本帧确认的烟 → 激活其所属手/头框(ID)
        #   该框内后续出现"非手自带物体" → 直接归类为烟(快速放行分支)
        for _sb in raw_smoke:
            for _ht in hand_tracks:
                if iou(_sb, _ht[0].bbox) > 0.05:
                    _ht[0].smoke_activated = True
            for _t in tracks:
                if iou(_sb, _t[0].bbox) > 0.05:
                    _t[0].smoke_activated = True

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

    # ★★ 15:23 光源可视化: 大面积光源区域画紫色半透明(分辨"光源"vs"白色内容")
    #   被过滤的光源 → 紫色; 白色小块(烟/白物)不在掩码 → 正常处理
    _disp_overlay = _disp.copy()
    _disp_overlay[_light_mask] = (255, 0, 255)   # 紫
    _disp = cv2.addWeighted(_disp, 0.82, _disp_overlay, 0.18, 0)

    # --- 绘制(在 _disp 上, 覆盖涂暗层, 标注框颜色保持鲜艳) ---
    # ★★ 手部蓝色框(新架构): 追踪框随手掌大小动态变化(Kalman更新)
    #   15:35 防分裂: lost>=2 的旧轨不画(快速移动时旧位置残影立刻消失, 不再与新位置双框)
    #   ★ 21:36 治"快速移动框消失": 运动模糊→MediaPipe失败→lost增长→lost>=2隐藏=消失
    #   放宽到 lost>=4: 模糊1-3帧时用Kalman预测位置继续显示(不消失), 第4帧才隐藏
    # ★ 01:47 完全沿用V8: 显示隐藏 lost>=2(15:35 V8原版, 残影快速隐藏防双框)
    for ht in hand_tracks:
        if ht[0].lost >= 2:
            continue   # 丢失2帧隐藏(V8机制)
        # ★★ 08-12 乱飘兜底(治"两手一前一后打圈→丢失轨乱飘"): 丢失期间(lost≥1)
        #   若显示框已漂离最后确认位置>1.5手宽 → 隐藏。即使前面手的检测漏网喂入,
        #   显示也不允许框飘远(宁可消失, 不乱飘)
        if ht[0].lost >= 1:
            _dchk = ht[0].disp_bbox
            _lgck = getattr(ht[0], 'last_good', None)
            if _lgck is not None:
                _ddc = ((_dchk[0]+_dchk[2])/2 - (_lgck[0]+_lgck[2])/2)**2 + \
                       ((_dchk[1]+_dchk[3])/2 - (_lgck[1]+_lgck[3])/2)**2
                _wwc = max(_dchk[2]-_dchk[0], 20.0)
                if _ddc > (_wwc * 1.5) ** 2:
                    continue   # 丢失框已飘远 → 隐藏(不乱飘)
        hx1, hy1, hx2, hy2 = map(int, ht[0].disp_bbox if hasattr(ht[0], 'disp_bbox') else ht[0].bbox)
        # ★★ 08-12 新架构: 手框内物品检测(非人体天然物: 烟/杯子/手机等手持物)
        #   连续2帧有物才确认(防单帧误检闪烁); 有物 → 框变红 + 最清晰帧截图入库
        _has_obj = _roi_has_object(frame, hx1, hy1, hx2, hy2, 'hand')
        _oc = getattr(ht[0], '_obj_cnt', 0)
        _oc = (_oc + 1) if _has_obj else 0
        ht[0]._obj_cnt = _oc
        _obj_ok = _oc >= 2
        if _obj_ok:
            _try_capture_best(frame, hx1, hy1, hx2, hy2, 'hand', ht[1])
        _hcol = RED if _obj_ok else (255, 0, 0)   # 有物红框, 无物蓝框
        cv2.rectangle(_disp, (hx1,hy1), (hx2,hy2), _hcol, 2)
        _htxt = "有物" if _obj_ok else "空手"
        cv2.putText(_disp, f"H{ht[1]}:{_htxt}", (hx1, hy1-6), cv2.FONT_HERSHEY_SIMPLEX, 0.4, _hcol, 1)

    for t in tracks:
        x1, y1, x2, y2 = map(int, t[0].bbox)
        # ★★ 08-12 新架构: 头框嘴部物品检测(嘴含的烟/吸管等非人体天然物)
        _has_obj = _roi_has_object(frame, x1, y1, x2, y2, 'head')
        _oc = getattr(t[0], '_obj_cnt', 0)
        _oc = (_oc + 1) if _has_obj else 0
        t[0]._obj_cnt = _oc
        _obj_ok = _oc >= 2
        if _obj_ok:
            _try_capture_best(frame, x1, y1, x2, y2, 'head', t[1])
        _hcol = RED if _obj_ok else GREEN   # 有物红框, 无物绿框
        cv2.rectangle(_disp, (x1,y1), (x2,y2), _hcol, 2)
        lb = f"#{t[1]}"
        (tw,th),_ = cv2.getTextSize(lb, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)
        cv2.rectangle(_disp, (x1,y1-14), (x1+tw+4,y1), _hcol, -1)
        cv2.putText(_disp, lb, (x1+2,y1-3), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,0), 1)

    for st in smoke_tracks:
        # ★★ 严格区域(用户强调): 烟轨中心必须在头/手聚焦区域内才显示, 否则"装没看见"
        #   即使香烟就在镜头中, 不在头/手框内 → 不显示(区域外禁止识别功能)
        _scx = (st[0].disp_bbox[0]+st[0].disp_bbox[2])/2
        _scy = (st[0].disp_bbox[1]+st[0].disp_bbox[3])/2
        _s_in = False
        for (frx1, fry1, frx2, fry2, _) in focus_regions:
            if frx1 <= _scx <= frx2 and fry1 <= _scy <= fry2:
                _s_in = True; break
        if not _s_in:
            continue   # 不在头/手区域 → 不显示(装没看见)
        # ★★ 红框必须与蓝框/绿框有交集(用户17:07要求): 脱离立即不显示
        #   18:10 防断续: 交集判定含平滑追踪框(本帧漏检1-2帧不闪断, 真消失才消失)
        _touch_now = False
        _dbb = st[0].disp_bbox
        for (_tbx1, _tby1, _tbx2, _tby2) in list(det_hands) + list(det_heads) + \
                [ht[0].bbox for ht in hand_tracks] + [t[0].bbox for t in tracks]:
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
        # ★ 统一红色(用户17:34): 防闪缓冲内(lost 0-2)不区分颜色, 消除红/黄交替闪烁
        #   原: lost>0画黄框CIG?, lost=0画红框 → 检测抖动导致红黄交替闪
        cv2.rectangle(_disp, (sx1,sy1), (sx2,sy2), RED, 3)
        cv2.putText(_disp, "CIG", (sx1, sy2+18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, RED, 2)

    # ★ 16:29 技术效果可视化: 状态面板(模型看到什么 + 各技术状态)
    _fps_now = fc / max(1, time.time() - t0)
    cv2.putText(_disp, f"H:{len(tracks)} C:{len(smoke_tracks)} {_fps_now:.0f}fps",
                (4, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, GREEN, 2)
    cv2.putText(_disp, f"TRT:{'ON' if _os.path.exists(r'D:/视觉安防系统/models/smoke_v12.engine') else 'OFF'} "
                       f"BOTSORT:ON SR:{'ON' if _sr_model is not None else 'OFF'} "
                       f"RIFE:{'ON' if _rife_model is not None else 'OFF'}",
                (4, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 2)

    # ★ 显示(17:06 帧率修复): 采集1080p(检测细节好), 显示缩到720p(锐化+imshow轻, 帧率稳)
    #   1080p全帧锐化/imshow 20-40ms/帧 → 帧率骤降; 720p显示视觉仍清晰
    _disp = cv2.resize(_disp, (1280, 720), interpolation=cv2.INTER_LANCZOS4) if _disp.shape[1] > 1280 else _disp
    # ★ 轻量锐化(Unsharp, ~2ms 720p): 边缘更清晰(感知锐度提升, 不引入伪影)
    _disp = cv2.addWeighted(_disp, 1.5, cv2.GaussianBlur(_disp, (0, 0), 2.0), -0.5, 0)
    _show = _disp
    # ★★ 08-12 Web前端帧发布: 显示帧压缩入队(丢旧保新), MJPEG流用
    try:
        _jpeg = cv2.imencode('.jpg', _show, [cv2.IMWRITE_JPEG_QUALITY, 70])[1].tobytes()
        if _web_queue.full():
            try:
                _web_queue.get_nowait()
            except _queue.Empty:
                pass
        _web_queue.put(_jpeg)
    except Exception:
        pass
    # 16:59 单窗口: 主画面=模型实际看到的(检测输入帧+框+状态), 无额外窗口/对比窗
    cv2.imshow("Head + Smoke Detection", _show)
    _prev_disp = _disp.copy()
    fc += 1
    if cv2.waitKey(1) & 0xFF == ord('q'): break

_stop_flag.set()   # 停采集线程(重构版)
try:
    _th_cap.join(timeout=2)
except Exception:
    pass
cap.release(); cv2.destroyAllWindows()
