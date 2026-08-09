"""手部检测模型训练: YOLOv8n, 数据来自 COCO-hand + TV-hand
用法: python train_hand.py
目标: 手掌检测精准, 与头部一样流畅追踪
"""
import yaml, os

BASE = r'D:/training_data/hand_yolo'

# 数据集配置
data_yaml = {
    'path': BASE,
    'train': 'images/train',
    'val': 'images/val',
    'nc': 1,
    'names': {0: 'hand'},
}
with open(os.path.join(BASE, 'data.yaml'), 'w', encoding='utf-8') as f:
    yaml.safe_dump(data_yaml, f, allow_unicode=True)
print('data.yaml 已生成:', os.path.join(BASE, 'data.yaml'))

from ultralytics import YOLO

model = YOLO('yolov8n.pt')   # 预训练起点
model.train(
    data=os.path.join(BASE, 'data.yaml'),
    epochs=100,
    imgsz=640,
    batch=16,
    name='hand_v1',
    device=0,
    workers=0,
    patience=20,
    optimizer='AdamW',
    lr0=0.001,
    verbose=True,
)
print('\n✅ 手部模型训练完成!')
