"""V3 动态追踪测试 30s - 移动/运动模糊/快速变向"""
import cv2, time, sys
sys.path.insert(0, 'D:/视觉安防系统')
from detector import MultiStageDetector

detector = MultiStageDetector(
    person_model_path='D:/视觉安防系统/models/yolov8n.onnx',
    smoking_model_path='D:/视觉安防系统/models/smoking.onnx',
    pose_model_path='D:/视觉安防系统/models/yolov8n-pose.onnx',
    conf_threshold=0.35, iou_threshold=0.45,
)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
time.sleep(1)

print("="*55)
print("  V3 动态追踪 30秒")
print("  请做: 慢慢转头→快速摇头→走近→走远→变身高→点头")
print("="*55)

frames = []
heads_log = []
fps_log = []
t0 = time.time()
DURATION = 30
sample_int = 6  # 每6帧保存一张图

while time.time() - t0 < DURATION:
    ret, frame = cap.read()
    if not ret: break
    t1 = time.time()
    result = detector.detect(frame)
    dt = time.time() - t1

    heads = result.get('heads', [])
    heads_log.append(len(heads))
    fps_log.append(dt)

    # 画框
    for hh in heads:
        bx1,by1,bx2,by2 = [int(x) for x in hh['bbox']]
        cv2.rectangle(frame, (bx1,by1), (bx2,by2), (0,255,0), 2)
        tid = hh.get('track_id', '?')
        cv2.putText(frame, f'Head#{tid}', (bx1,by1-8),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

    elapsed = time.time() - t0
    cv2.putText(frame, f't={elapsed:.0f}s heads={len(heads)}',
               (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)

    # 采样保存
    if int(elapsed * 10) % 8 == 0 and len(frames) < 10:
        frames.append((f't{elapsed:.1f}s', frame.copy()))

cap.release()

# 统计
total = len(heads_log)
detected = sum(1 for h in heads_log if h > 0)
det_rate = detected/total*100 if total else 0
avg_fps = 1.0/(sum(fps_log)/len(fps_log)) if fps_log else 0

print(f"\n总帧数: {total}")
print(f"检出帧: {detected}/{total} ({det_rate:.1f}%)")
print(f"平均FPS: {avg_fps:.1f}")

import os
os.makedirs('D:/verify_out/v3_dynamic', exist_ok=True)
for i, (name, frame) in enumerate(frames):
    p = f'D:/verify_out/v3_dynamic/dyn_{i}_{name}.jpg'
    cv2.imwrite(p, frame)
    print(f'  saved {p}')

print(f"\n{'✅' if det_rate>=85 else '⚠️'} 动态追踪{'稳定' if det_rate>=85 else '波动'}")