"""
V7-Head + Smoke — 双模型管道
V7: 头部检测 (绿色框)
吸烟检测: 头部ROI内找香烟 (红色框)
"""
import cv2, numpy as np, time
from ultralytics import YOLO

HEAD = YOLO(r"D:\视觉安防系统\models\yolov8n_head_v7.pt", task='detect')
HEAD.to('cuda')
print("✅ V7-Head-Smoke | 头部绿色 | 香烟红色 | Q退出")

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
W, H = int(cap.get(3)), int(cap.get(4))

GREEN = (0, 255, 100)
RED   = (0, 0, 255)
fc = 0; t0 = time.time()

class KalmanBox:
    def __init__(self,bbox):
        self.kf=cv2.KalmanFilter(6,4)
        self.kf.transitionMatrix=np.array([
            [1,0,0,0,1,0],[0,1,0,0,0,1],[0,0,1,0,0,0],
            [0,0,0,1,0,0],[0,0,0,0,1,0],[0,0,0,0,0,1]],np.float32)
        self.kf.measurementMatrix=np.eye(4,6,dtype=np.float32)
        self.kf.processNoiseCov=np.eye(6,dtype=np.float32)*0.05
        self.kf.measurementNoiseCov=np.eye(4,dtype=np.float32)*0.1
        self.kf.errorCovPost=np.eye(6,dtype=np.float32)
        cx=(bbox[0]+bbox[2])/2;cy=(bbox[1]+bbox[3])/2
        w=bbox[2]-bbox[0];h=bbox[3]-bbox[1]
        self.kf.statePost=np.array([[cx],[cy],[w],[h],[0],[0]],np.float32)
        self.bbox=bbox;self.lost=0
    def predict(self):
        p=self.kf.predict().flatten()
        self.bbox=(float(p[0]-p[2]/2),float(p[1]-p[3]/2),float(p[0]+p[2]/2),float(p[1]+p[3]/2))
        return self.bbox
    def update(self,bbox):
        cx=(bbox[0]+bbox[2])/2;cy=(bbox[1]+bbox[3])/2
        w=bbox[2]-bbox[0];h=bbox[3]-bbox[1]
        self.kf.correct(np.array([[cx],[cy],[w],[h]],np.float32));self.lost=0
        p=self.kf.statePost.flatten()
        self.bbox=(float(p[0]-p[2]/2),float(p[1]-p[3]/2),float(p[0]+p[2]/2),float(p[1]+p[3]/2))

def iou(a,b):
    ix1=max(a[0],b[0]);iy1=max(a[1],b[1]);ix2=min(a[2],b[2]);iy2=min(a[3],b[3])
    iw=max(0,ix2-ix1);ih=max(0,iy2-iy1)
    aa=(a[2]-a[0])*(a[3]-a[1]);bb=(b[2]-b[0])*(b[3]-b[1])
    return iw*ih/(aa+bb-iw*ih+1e-6)

tracks=[];next_id=0

while True:
    ret,frame=cap.read()
    if not ret:break

    for t in tracks:t[0].predict()

    # V7 头检测
    res=HEAD(frame,conf=0.3,iou=0.5,verbose=False)
    det_heads=[]
    for r in res:
        for box in r.boxes:
            b=box.xyxy[0].cpu().numpy()
            x1,y1,x2,y2=float(b[0]),float(b[1]),float(b[2]),float(b[3])
            if x2-x1<8 or y2-y1<8:continue
            det_heads.append((x1,y1,x2,y2))

    # 匹配+新轨迹
    matched_ids=set();matched_det=set()
    for j,dh in enumerate(det_heads):
        best_i,best_iou=-1,0.15
        for i,t in enumerate(tracks):
            if t[1] in matched_ids:continue
            iou_val=iou(t[0].bbox,dh)
            if iou_val<0.15:
                kf=t[0].kf;ps=kf.statePre.flatten()
                pb=(ps[0]-ps[2]/2,ps[1]-ps[3]/2,ps[0]+ps[2]/2,ps[1]+ps[3]/2)
                iou_val=iou(pb,dh)
            if iou_val>best_iou:best_iou=iou_val;best_i=i
        if best_i>=0:
            tracks[best_i][0].update(dh)
            matched_ids.add(tracks[best_i][1]);matched_det.add(j)

    for t in tracks:
        if t[1] not in matched_ids:t[0].lost+=1

    for j,dh in enumerate(det_heads):
        if j not in matched_det and len(tracks)<20:
            tracks.append([KalmanBox(dh),next_id]);next_id+=1

    tracks=[t for t in tracks if t[0].lost<5]

    # 烟检: 逐头检测 (模型训练中, 暂时返回空)
    smoke_heads = set()  # 检出香烟的轨迹ID
    # TODO: smoke_heads = detect_cigarettes(frame, tracks)  ← 训练完成后启用

    # 绘制: 绿色头框 + 红框(仅检出烟的轨迹)
    for t in tracks:
        x1,y1,x2,y2=map(int,t[0].bbox)
        # 绿色头框(始终显示)
        cv2.rectangle(frame,(x1,y1),(x2,y2),GREEN,2)
        lb=f"#{t[1]}"
        (tw,th),_=cv2.getTextSize(lb,cv2.FONT_HERSHEY_SIMPLEX,0.35,1)
        cv2.rectangle(frame,(x1,y1-14),(x1+tw+4,y1),GREEN,-1)
        cv2.putText(frame,lb,(x1+2,y1-3),cv2.FONT_HERSHEY_SIMPLEX,0.35,(0,0,0),1)
        # 红框: 仅检出烟时覆盖全头
        if t[1] in smoke_heads:
            cv2.rectangle(frame,(x1,y1),(x2,y2),RED,3)  # 3px粗红框
            cv2.putText(frame,"SMOKE",(x1+2,y2-4),
                        cv2.FONT_HERSHEY_SIMPLEX,0.4,RED,2)

    cv2.putText(frame,f"V7-Smoke | {len(tracks)} heads | {fc/max(1,time.time()-t0):.0f}fps",
                (4,18),cv2.FONT_HERSHEY_SIMPLEX,0.5,GREEN,2)
    cv2.imshow("V7 Head + Smoke Detection",frame)
    fc+=1
    if cv2.waitKey(1)&0xFF==ord('q'):break

cap.release();cv2.destroyAllWindows()