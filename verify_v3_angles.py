"""V3 角度验证 - 正面/侧脸/半身/背对/边缘实时测试"""
import cv2, time, sys, yaml
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

print("\n" + "="*55)
print("  V3 角度验证 - 请依次做以下动作:")
print("  1.正面  2.左侧脸  3.右侧脸  4.半身  5.背对  6.边缘")
print("  每角度保持3秒，按Q退出")
print("="*55)

fps_times = []
frame_count = 0
fps = 0

while True:
    ret, frame = cap.read()
    if not ret: break

    frame_count += 1
    t0 = time.time()
    result = detector.detect(frame)
    dt = time.time() - t0
    fps_times.append(dt)
    if len(fps_times) > 30: fps_times = fps_times[-30:]
    fps = 1.0 / (sum(fps_times)/len(fps_times)) if fps_times else 0

    heads = result.get('heads', [])
    persons = result.get('persons', [])
    h, w = frame.shape[:2]

    # 画人体框
    for p in persons:
        if len(p) >= 4:
            bx1, by1, bx2, by2 = p[:4]
            cv2.rectangle(frame, (int(bx1), int(by1)), (int(bx2), int(by2)), (128,128,128), 1)

    # 画头部绿框 + 角度估算
    for hh in heads:
        bx1, by1, bx2, by2 = hh['bbox']
        cv2.rectangle(frame, (int(bx1), int(by1)), (int(bx2), int(by2)), (0,255,0), 2)

        tid = hh.get('track_id', '?')
        cv2.putText(frame, f"Head#{tid}", (int(bx1), int(by1)-8),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

        # 角度估算
        kpts = hh.get('keypoints', None)
        if kpts is not None and kpts.shape[0] >= 5:
            nose = kpts[0]
            eye_l = kpts[1]
            eye_r = kpts[2]
            confidence = (float(eye_l[2]) + float(eye_r[2])) / 2
            angle_text = ""
            color = (0,255,0)
            if confidence > 0.3:
                if float(nose[2]) > 0.3 and float(eye_l[2]) > 0.3 and float(eye_r[2]) > 0.3:
                    angle_text = "正面"
                elif float(eye_l[2]) > 0.3 and float(eye_r[2]) < 0.2:
                    angle_text = "左侧脸"
                    color = (255,255,0)
                elif float(eye_r[2]) > 0.3 and float(eye_l[2]) < 0.2:
                    angle_text = "右侧脸"
                    color = (255,255,0)
                elif float(nose[2]) > 0.3 and float(eye_l[2]) < 0.2 and float(eye_r[2]) < 0.2:
                    angle_text = "背对/半身"
                    color = (0,200,255)
                cv2.putText(frame, angle_text, (int(bx1), int(by2)+16),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # 状态栏
    cv2.putText(frame, f"FPS:{fps:.0f} Heads:{len(heads)}", (8, 20),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)

    cv2.imshow("V3 Angle Test", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
