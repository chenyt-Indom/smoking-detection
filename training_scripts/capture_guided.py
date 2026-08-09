"""引导式误检帧采集 v2 - 分阶段提示用户做动作, 保存检出帧供筛选"""
import cv2, os, time
from pathlib import Path
from ultralytics import YOLO

SMOKE = YOLO(r'D:\视觉安防系统\models\smoke_cig_v30.pt')
SMOKE.to('cuda')

OUT_ROOT = Path('D:/training_data/smoke/fp_capture3')
OUT_ROOT.mkdir(parents=True, exist_ok=True)

# 阶段: (名称, 时长秒, 动作提示)
STAGES = [
    ("stage1_face", 40,  "① 正脸对镜头, 慢慢左右转头"),
    ("stage2_phone", 40, "② 拿手机, 转各种角度/亮屏/黑屏"),
    ("stage3_objects", 40, "③ 拿书/笔/笔盒等长条物晃动"),
    ("stage4_arm", 40,  "④ 伸手/伸胳膊/挥手(不拿烟)"),
    ("stage5_far", 40,  "⑤ 走到画面远处, 转身/伸手"),
    ("stage6_cig", 40,  "⑥ 拿真烟! 横/竖/斜/圆头各种角度"),
    ("stage7_smoke", 40, "⑦ 模拟吸烟: 烟含嘴里, 手夹烟"),
    ("stage8_free", 40, "⑧ 自由活动: 正常做事, 走动"),
]

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print('❌ 摄像头打开失败'); exit(1)
W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

print(f'🎥 引导式采集启动: 8阶段×40秒 = {sum(s[1] for s in STAGES)}秒')
print(f'   跟着画面上的提示做动作, 程序自动记录检出帧\n')

saved_total = 0
for stage_name, dur, hint in STAGES:
    out_img = OUT_ROOT / stage_name / 'images'
    out_lbl = OUT_ROOT / stage_name / 'labels'
    out_img.mkdir(parents=True, exist_ok=True)
    out_lbl.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    last_save = 0
    saved = 0
    print(f'▶ {hint}  ({dur}s)')
    while time.time() - t0 < dur:
        ok, frame = cap.read()
        if not ok: break
        # 显示提示
        remain = int(dur - (time.time() - t0))
        cv2.putText(frame, f'{hint}  [{remain}s]', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        res = SMOKE(frame, conf=0.10, iou=0.45, verbose=False)
        boxes = res[0].boxes
        now = time.time()
        if len(boxes) > 0 and now - last_save >= 0.5:
            last_save = now
            # ★ 画框只在"显示帧"上画, 保存用原始frame(无红框) — 修复红框入图污染
            disp = frame.copy()
            for b in boxes:
                x1, y1, x2, y2 = map(int, b.xyxy[0].cpu().numpy())
                cv2.rectangle(disp, (x1, y1), (x2, y2), (0, 0, 255), 2)
            name = f'{stage_name}_{saved:03d}.jpg'
            cv2.imwrite(str(out_img / name), frame)   # 保存原始帧(无红框)
            lines = []
            for b in boxes:
                x1, y1, x2, y2 = b.xyxy[0].cpu().numpy()
                cx = ((x1+x2)/2)/W; cy = ((y1+y2)/2)/H
                bw = (x2-x1)/W; bh = (y2-y1)/H
                lines.append(f'0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}')
            (out_lbl / name.replace('.jpg', '.txt')).write_text('\n'.join(lines) + '\n')
            saved += 1

        # 显示带框帧(便于用户看到模型在检什么)
        cv2.imshow('采集(红框=模型检出)', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cap.release(); cv2.destroyAllWindows()
            print(f'\n⏹ 提前结束'); exit(0)
    saved_total += saved
    print(f'  ✓ {stage_name}: 保存 {saved} 张')

cap.release()
cv2.destroyAllWindows()
print(f'\n✅ 采集完成! 共 {saved_total} 张检出帧 → {OUT_ROOT}')
print(f'   接下来: 我筛选真烟(正样本)与误检(负样本)加入训练')
