"""离线训练香烟+烟盒检测模型 - 2类: cigarette(0), cigarette_pack(1)"""
from ultralytics import YOLO
import os, shutil, glob

print("=" * 60)
print("  香烟+烟盒目标检测模型训练")
print("  类别: 0=cigarette(香烟) 1=cigarette_pack(烟盒)")
print("=" * 60)

if not os.path.exists("yolov8n.pt"):
    print("下载 YOLOv8n 预训练权重...")
    import requests
    url = "https://gitee.com/arfus/yolov8n_pt2rknn/raw/master/yolov8n.pt"
    resp = requests.get(url, timeout=60, verify=False, stream=True)
    with open("yolov8n.pt", "wb") as f:
        for chunk in resp.iter_content(65536):
            f.write(chunk)
    print(f"下载完成: {os.path.getsize('yolov8n.pt')/1024/1024:.1f} MB")

print("加载 YOLOv8n 预训练权重...")
model = YOLO("yolov8n.pt")

print("开始训练 (2类: cigarette + cigarette_pack)...")
model.train(data="datasets/smoking/data.yaml", epochs=30, imgsz=416,
            batch=16, name="smoking_det", project="runs", exist_ok=True,
            workers=0, verbose=True)

best_pt = None
for f in glob.glob("runs/**/best.pt", recursive=True):
    best_pt = f; break

if not best_pt:
    print("未找到 best.pt"); exit(1)

print(f"训练完成! 最佳模型: {best_pt} ({os.path.getsize(best_pt)/1024/1024:.1f} MB)")

print("导出 ONNX...")
model = YOLO(best_pt)
model.export(format="onnx", imgsz=416, half=True)

onnx_file = best_pt.replace(".pt", ".onnx")
if os.path.exists(onnx_file):
    os.makedirs("models", exist_ok=True)
    shutil.copy(onnx_file, "models/smoking.onnx")
    print(f"ONNX: models/smoking.onnx ({os.path.getsize('models/smoking.onnx')/1024/1024:.1f} MB)")

print("=" * 60)
print("  全部完成! 类别: cigarette, cigarette_pack")
print("=" * 60)