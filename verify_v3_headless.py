"""V3角度验证-无头录制10秒"""
import cv2, time, json, sys
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

print("录制10秒...请转动头部:正面→左→右→低头→半身→背对")
frames = 0
heads_total = 0
heads_by_frame = []
start = time.time()

while time.time() - start < 10:
    ret, frame = cap.read()
    if not ret: break
    result = detector.detect(frame)
    hds = result.get('heads', [])
    heads_total += len(hds)
    heads_by_frame.append(len(hds))
    frames += 1

cap.release()

det_rate = sum(1 for h in heads_by_frame if h > 0) / max(frames, 1) * 100
print(f"\n总帧数: {frames}")
print(f"有头帧: {sum(1 for h in heads_by_frame if h>0)}/{frames}")
print(f"检出率: {det_rate:.1f}%")
print(f"平均头数: {heads_total/max(frames,1):.2f}")
if det_rate >= 85:
    print("\n✅ V3各角度追踪正常!")
elif det_rate >= 60:
    print(f"\n🟡 检出率{det_rate:.0f}%，侧脸有改善但非完美")
else:
    print(f"\n⚠️ 检出率{det_rate:.0f}%，仍需优化")
