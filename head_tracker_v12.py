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
    HAND = YOLO(r"D:\视觉安防系统\models\hand_v12.pt", task='detect')
    print("✅ 手模型 hand_v12 加载")
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
            if _speed > 0.5: alpha = 0.92    # 快移: 紧跟
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
        # ★ 18:08 防抖: alpha 0.85→0.72, 烟框更稳定不"一抖一抖"(代价:轻微滞后)
        alpha = 0.50 if self.lost > 0 else 0.72  # ★显示紧跟检测(不滞后)
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
import os
AUTO_SAVE = r'D:\training_data\smoke\fp_auto'   # 误检帧自动采集(检出烟时保存)
os.makedirs(AUTO_SAVE, exist_ok=True)
last_auto_save = 0

# ★★ 重构版: 多线程采集(采集线程持续读帧 → 有界队列; 处理主线程取最新帧)
#   摄像头只被采集线程访问(线程安全), 处理耗时不再拖采集 → 帧率稳定
import threading, queue as _queue
_frame_q = _queue.Queue(maxsize=2)
_stop_flag = threading.Event()

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
    # ★ 16:29 超分线程取帧(异步展示超分效果)
    _sr_latest_frame = frame
    # ★ 16:58 RIFE演示: 缓存最近两帧小图(后台插帧对比窗用)
    _demo_a = _demo_b
    _demo_b = cv2.resize(frame, (320, 180), interpolation=cv2.INTER_AREA) if frame is not None else None

    # --- 头检测(15:31 完全用 V12 模型, 废掉 pose): head_v12 模型检测(完整头框) ---
    #   ★ 15:51 BoT-SORT 全面接入: HEAD.track(botsort) — 模型检测 + BOTSORT追踪(ReID/GMC/低分恢复)
    #   验证: 实拍图 15/15 检出(BOTSORT 工作正常)
    res = HEAD.track(frame, persist=True, tracker='botsort.yaml', conf=0.45, iou=0.5, imgsz=640, verbose=False)
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
            if iou_val > best_iou: best_iou = iou_val; best_i = i
        if best_i >= 0:
            tracks[best_i][0].update(dh); matched_ids.add(tracks[best_i][1]); matched_det.add(j)
    for t in tracks:
        if t[1] not in matched_ids: t[0].lost += 1
    for j, dh in enumerate(det_heads):
        if j not in matched_det and len(tracks) < 20:
            tracks.append([KalmanBox(dh), next_id]); next_id += 1
    tracks = [t for t in tracks if t[0].lost < 5]   # 头轨迹容忍(昨天配置, 恢复)

    # --- ★★ 手部检测(15:31 完全用 V12 模型, 废掉 pose): hand_v12 模型检测(完整手掌框) ---
    #   V8 机制: 模型输出(conf 0.12起) → 尺寸过滤 → 肤色验证 → 低分池 → 去重 → 追踪
    det_hands = []
    hand_low_pool = []
    if HAND_READY:
        try:
            # ★ 15:51 BoT-SORT 全面接入: HAND.track(botsort) — 低分恢复/ReID外观匹配
            res_h = HAND.track(frame, persist=True, tracker='botsort.yaml',
                               conf=0.25, iou=0.5, imgsz=640, verbose=False)
            for r in res_h:
                if r.boxes is None: continue
                for box in r.boxes:
                    b = box.xyxy[0].cpu().numpy()
                    conf = float(box.conf.cpu().numpy()[0]) if box.conf is not None else 1.0
                    x1, y1, x2, y2 = float(b[0]), float(b[1]), float(b[2]), float(b[3])
                    bw, bh = x2-x1, y2-y1
                    if bw < 20 or bh < 20: continue   # 最小手框(V8机制)
                    if bw > W*0.7 or bh > H*0.7: continue  # 超大框(V8机制)
                    if conf < 0.25:
                        hand_low_pool.append((x1, y1, x2, y2, conf))
                        continue
                    # ★★ 肤色验证(V8机制, 治椅子/木纹误判)
                    _hsx1, _hsy1 = max(0,int(x1)), max(0,int(y1))
                    _hsx2, _hsy2 = min(W,int(x2)), min(H,int(y2))
                    if _hsx2 - _hsx1 >= 8 and _hsy2 - _hsy1 >= 8:
                        _hcrop = frame[_hsy1:_hsy2, _hsx1:_hsx2]
                        _hhsv = cv2.cvtColor(_hcrop, cv2.COLOR_BGR2HSV)
                        _hpix = _hcrop.shape[0] * _hcrop.shape[1]
                        _hskin = float(np.sum(cv2.inRange(_hhsv, (0,25,50), (45,180,255)) > 0)) / _hpix
                        if _hskin < 0.25:
                            continue   # 肤色<25% → 非人手
                        _hsat_red = float(np.sum(cv2.inRange(_hhsv, (0,150,40), (30,255,255)) > 0)) / _hpix
                        if _hsat_red > 0.45:
                            continue   # 高饱和红木 → 拒
                    det_hands.append((x1, y1, x2, y2))
        except Exception:
            pass
    # ★ 15:07 手框时间平滑(模型输出已稳, 平滑防偶发跳变), 之后进去重/追踪
    det_hands, _prev_hand_boxes = smooth_boxes(det_hands, _prev_hand_boxes)
    # ★★ 手部检测去重(治"方框分裂/重叠"): 同一只手被模型输出多个框(微小偏移/双检)
    #   ★ 15:35 中心距 0.8→1.2 手宽: 快速移动时运动模糊导致双检框中心距大, 放宽合并
    det_hands_dedup = []
    for _dh in sorted(det_hands, key=lambda b: -(b[2]-b[0])*(b[3]-b[1])):
        _dup = False
        for _dh2 in det_hands_dedup:
            if iou(_dh, _dh2) > 0.3:
                _dup = True; break
            _d = ((_dh[0]+_dh[2])/2 - (_dh2[0]+_dh2[2])/2)**2 + ((_dh[1]+_dh[3])/2 - (_dh2[1]+_dh2[3])/2)**2
            if _d < ((_dh2[2]-_dh2[0]) * 1.2) ** 2:
                _dup = True; break
        if not _dup:
            det_hands_dedup.append(_dh)
    det_hands = det_hands_dedup
    # ★★ 20:15 ReID: 为每个手检测框提取外观签名(HSV直方图), 供追踪匹配
    _hand_sigs = [extract_sig(frame, _dh) for _dh in det_hands]
    # 手部关联追踪(BoTSORT式: 位置IoU + 外观ReID双通道匹配, 快速移动不消失)
    h_matched_ids = set(); h_matched_det = set()
    for j, dh in enumerate(det_hands):
        best_i, best_score = -1, 0.15
        for i, ht in enumerate(hand_tracks):
            if ht[1] in h_matched_ids: continue
            iou_val = iou(ht[0].bbox, dh)
            if iou_val < 0.15:
                kf = ht[0].kf; ps = kf.statePre.flatten()
                pb = (ps[0]-ps[2]/2, ps[1]-ps[3]/2, ps[0]+ps[2]/2, ps[1]+ps[3]/2)
                iou_val = iou(pb, dh)
            # ★ ReID 外观通道: 快速移动IoU≈0但同一只手肤色/纹理一致 → 外观补偿匹配
            _app = 0.0
            if ht[0].sig is not None and _hand_sigs[j] is not None:
                _corr, _size_ok = sig_similarity(ht[0].sig, _hand_sigs[j])
                if _size_ok:
                    _app = max(0.0, _corr)   # 相关性[-1,1] → [0,1]
            # 组合分 = 位置IoU + 外观(权重0.8): 保留原IoU通道(≥0.15即匹配),
            # 同时 IoU低但外观像(如0.05+0.3)也匹配 → 快速移动帧不丢轨
            _score = iou_val + _app * 0.8
            if _score > best_score: best_score = _score; best_i = i
        if best_i >= 0:
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
                    _best_iou_low = _io; _best_low = _lp
            if _best_low is not None:
                _ht[0].update(_best_low[:4])
                _ht[0].lost = 0
                h_matched_ids.add(_ht[1])
    # ★★ 20:12 局部找回(快速移动不消失): 仍未匹配的轨(lost≥1), 预测位置裁剪放大再检
    #   仅对"追踪中的手"局部搜索(conf 0.15), 不产生全局新检测 → 不会误检爆炸
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
            _rr3 = HAND(_crop3, conf=0.15, iou=0.5, imgsz=480, verbose=False)
            _found3 = None
            for _r in _rr3:
                for _b in _r.boxes:
                    _bx1, _by1, _bx2, _by2 = _b.xyxy[0].cpu().numpy()
                    _sc3 = 480.0 / max(_ex2-_ex1, _ey2-_ey1)
                    _gx1 = _ex1 + _bx1/_sc3; _gy1 = _ey1 + _by1/_sc3
                    _gx2 = _ex1 + _bx2/_sc3; _gy2 = _ey1 + _by2/_sc3
                    if _gx2-_gx1 < 15 or _gy2-_gy1 < 15: continue
                    _found3 = (_gx1, _gy1, _gx2, _gy2)
                    break
                if _found3 is not None: break
            if _found3 is not None:
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
            _dup_track = False
            for _ht in hand_tracks:
                _hb = _ht[0].bbox
                _d2 = ((dh[0]+dh[2])/2 - (_hb[0]+_hb[2])/2)**2 + ((dh[1]+dh[3])/2 - (_hb[1]+_hb[3])/2)**2
                _hw_max = max(dh[2]-dh[0], _hb[2]-_hb[0])
                if _d2 < (_hw_max * 1.5) ** 2:
                    _dup_track = True
                    # ★ 顺手把它匹配给最近的轨(加速收敛, 框立刻跟上手)
                    if _ht[1] not in h_matched_ids:
                        hand_tracks[hand_tracks.index(_ht)][0].update(dh)
                        h_matched_ids.add(_ht[1])
                    break
            if not _dup_track and len(hand_tracks) < 10:
                # ★ ReID: 新建轨携带外观签名(后续匹配用)
                hand_tracks.append([KalmanBox(dh, pn=0.50, mn=0.05, sig=_hand_sigs[j] if j < len(_hand_sigs) else None),
                                    next_id + 1000]); next_id += 1
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
    for ht in hand_tracks:
        if ht[0].lost >= 2:
            continue   # ★ 旧残影快速隐藏(防"手移动快→方框分裂")
        hx1, hy1, hx2, hy2 = map(int, ht[0].disp_bbox if hasattr(ht[0], 'disp_bbox') else ht[0].bbox)
        cv2.rectangle(_disp, (hx1,hy1), (hx2,hy2), (255,0,0), 2)   # 蓝色
        cv2.putText(_disp, "HAND", (hx1, hy1-6), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,0,0), 1)
# ★★ 19:35 手持物判断v5: 只认"拍到真烟", 排除"手本身被当烟"
        #   ✅ 持物 = 已确认烟轨(confirmed≥3, 当前帧在手框内) 且 与手框"部分重叠"
        #   ★ 互斥判定(关键): 烟框与手框 IoU 必须 ∈ (0.03, 0.4]
        #     - 手持真烟: 烟框小(细长), 部分嵌入手里 → IoU≈0.05-0.3 → 持物 ✅
        #     - 手/手指被当烟(空手误检): 烟框≈手框 → IoU>0.4 → 排除 ❌
        #     - 烟离手(嘴前/桌面): 无交集 IoU=0 → 不算手持 ✅
        _holding = False
        for _st in smoke_tracks:
            if _st[0].confirmed < 3: continue        # 连续≥3帧稳定(滤短时误检)
            if _st[0].lost > 0: continue             # 当前帧还在(防历史轨残影)
            _sbb = _st[0].bbox
            _iou_h = iou(_sbb, (hx1, hy1, hx2, hy2))
            if 0.03 < _iou_h <= 0.4:                 # 部分重叠 = 手持真烟
                _holding = True; break
        _htxt = "持物" if _holding else "空手"
        _hcol = (0, 255, 255) if _holding else (255, 0, 0)
        cv2.putText(_disp, _htxt, (hx1, hy2+16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, _hcol, 1)

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
