"""V26 - 仅757张人工标注(V24起点, 300轮, 慢学习率, 在线增强)
数据: 仅本次人工标注的757张原图(无离线副本)
  train 685 (正499/负186) | val 72 (正49/负23)
  - 负样本(吸管/手机/背景帧, 无框)自动作为背景学习 → 模型学会"这些不是烟"
特点:
1. 防背景死记: 在线增强 erasing/scale/translate/shear/degrees 扰动, 框外背景不参与梯度
2. 运动模糊适应: 无离线副本, 依赖 V24 已有能力 + 在线增强(动态模糊由检测端Kalman+多帧补偿)
3. 光线自适应: 训练时 hsv_h/s/v 在线强增强; 推理端 head_tracker_cam.light_adapt() 已接入
4. 保留V24记忆: V24权重起点 + freeze=4(浅层冻结)
5. 慢学习率: lr0=0.0003, 300轮, patience=60
"""
from ultralytics import YOLO

model = YOLO(r'D:\视觉安防系统\models\smoke_cig_v24.pt')
print('V26: V24起点 + 757张人工标注(无副本) | train 685 / val 72 | 300轮')

model.train(
    data=r"D:/training_data/smoke/v26_manual_orig/data.yaml",
    epochs=300,
    imgsz=640,
    batch=24,
    name='cig_v26',
    patience=60,
    save=True,
    save_period=20,
    device=0,
    workers=0,
    lr0=0.0003,
    lrf=0.002,
    optimizer='AdamW',
    cos_lr=True,
    warmup_epochs=5,
    close_mosaic=15,
    amp=True,
    freeze=4,
    # --- 在线增强: 防背景死记 + 光线 + 形状扰动 ---
    mosaic=0.8,
    mixup=0.15,
    copy_paste=0.1,
    scale=0.8,
    translate=0.15,
    shear=12,
    degrees=180,
    perspective=0.0005,
    flipud=0.1,
    fliplr=0.5,
    hsv_h=0.06,
    hsv_s=0.9,
    hsv_v=0.8,
    erasing=0.4,
    crop_fraction=0.9,
)
print("\n✅ V26 DONE! runs/detect/cig_v26/weights/best.pt")
