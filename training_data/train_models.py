#!/usr/bin/env python3
"""
YOLOv8 模型训练脚本
- 人体检测模型: YOLOv8n 微调
- 头部检测模型: YOLOv8n 微调
- 自动导出 FP16 ONNX
- 自动评估 mAP
"""

import os, sys, time, yaml, shutil, argparse
from pathlib import Path
from datetime import datetime

# 添加训练日志目录
BASE_DIR = Path(r"D:/视觉安防系统/training_data")
LOG_DIR = BASE_DIR / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

def validate_dataset(data_yaml: str) -> bool:
    """验证数据集完整性"""
    with open(data_yaml, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    
    base = Path(cfg['path'])
    train_img = base / cfg['train']
    val_img = base / cfg['val']
    
    n_train = len(list(train_img.glob("*.jpg"))) if train_img.exists() else 0
    n_val = len(list(val_img.glob("*.jpg"))) if val_img.exists() else 0
    
    print(f"  训练集: {n_train} 张, 验证集: {n_val} 张")
    
    if n_train == 0:
        print("  ❌ 训练集为空！")
        return False
    return True

def train_head_model(epochs: int = 200, batch: int = 32):
    """训练头部检测模型 —— 重点训练，更多轮次 + 更强数据增强"""
    print("\n" + "="*60)
    print("🎯 训练头部检测模型 (YOLOv8n Head Detection)")
    print("="*60)
    
    from ultralytics import YOLO
    
    data_yaml = str(BASE_DIR / "configs" / "head_detection.yaml")
    
    if not validate_dataset(data_yaml):
        return False
    
    # 从头训练（不加载预训练权重以获得专门头部检测能力）
    # 实际上微调预训练权重效果更好
    model = YOLO("yolov8n.pt")  # 从 COCO 预训练开始
    
    start_time = time.time()
    
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=640,
        batch=batch,
        name="head_detection",
        project=str(LOG_DIR),
        exist_ok=True,
        device=0,  # GPU
        workers=8,
        patience=50,  # 50 轮无改善则早停
        save=True,
        save_period=10,
        
        # === 强数据增强（适应复杂环境） ===
        augment=True,
        hsv_h=0.015,   # 色调增强
        hsv_s=0.7,     # 饱和度增强
        hsv_v=0.4,     # 亮度增强
        degrees=15.0,  # 旋转增强
        translate=0.1, # 平移增强
        scale=0.5,     # 缩放增强（模拟不同距离）
        shear=2.0,     # 剪切增强
        perspective=0.0005,  # 透视增强
        flipud=0.2,    # 上下翻转（监控摄像头可能出现）
        fliplr=0.5,    # 左右翻转
        mosaic=1.0,    # 马赛克增强（模拟人群遮挡）
        mixup=0.2,     # 混合增强
        copy_paste=0.1,  # 复制粘贴增强
        
        # === 优化器设置 ===
        optimizer="AdamW",
        lr0=0.001,
        lrf=0.0001,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3,
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,
        
        # === 损失权重 ===
        box=7.5,
        cls=0.5,
        dfl=1.5,
        
        # === 其他 ===
        close_mosaic=10,  # 最后 10 轮关闭 mosaic 增强
        resume=False,
        amp=True,          # 自动混合精度
        fraction=1.0,
        single_cls=True,   # 单类别任务
        verbose=True,
        seed=42,
        deterministic=True,
        plots=True,        # 生成训练曲线图
    )
    
    elapsed = time.time() - start_time
    print(f"\n⏱️  训练耗时: {elapsed/60:.1f} 分钟")
    
    # 导出 ONNX
    export_model(
        str(LOG_DIR / "head_detection" / "weights" / "best.pt"),
        str(BASE_DIR / "models" / "yolov8n_head.onnx")
    )
    
    return True

def train_person_model(epochs: int = 150, batch: int = 32):
    """训练人体检测模型"""
    print("\n" + "="*60)
    print("🕺 训练人体检测模型 (YOLOv8n Person Detection)")
    print("="*60)
    
    from ultralytics import YOLO
    
    data_yaml = str(BASE_DIR / "configs" / "person_detection.yaml")
    
    if not validate_dataset(data_yaml):
        return False
    
    model = YOLO("yolov8n.pt")  # COCO 预训练
    
    start_time = time.time()
    
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=640,
        batch=batch,
        name="person_detection",
        project=str(LOG_DIR),
        exist_ok=True,
        device=0,
        workers=8,
        patience=30,
        save=True,
        save_period=10,
        
        # 数据增强
        augment=True,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=10.0,
        translate=0.1,
        scale=0.5,
        shear=1.0,
        perspective=0.0003,
        flipud=0.1,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.15,
        copy_paste=0.1,
        
        # 优化器
        optimizer="AdamW",
        lr0=0.001,
        lrf=0.0001,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3,
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,
        
        # 损失权重
        box=7.5,
        cls=0.5,
        dfl=1.5,
        
        close_mosaic=10,
        resume=False,
        amp=True,
        fraction=1.0,
        single_cls=True,
        verbose=True,
        seed=42,
        deterministic=True,
        plots=True,
    )
    
    elapsed = time.time() - start_time
    print(f"\n⏱️  训练耗时: {elapsed/60:.1f} 分钟")
    
    # 导出 ONNX
    export_model(
        str(LOG_DIR / "person_detection" / "weights" / "best.pt"),
        str(BASE_DIR / "models" / "yolov8n_person.onnx")
    )
    
    return True

def export_model(pt_path: str, onnx_path: str):
    """导出 ONNX 模型"""
    print(f"\n📦 导出 ONNX: {onnx_path}")
    from ultralytics import YOLO
    model = YOLO(pt_path)
    model.export(
        format="onnx",
        imgsz=640,
        half=True,   # FP16
        simplify=True,
        opset=12,
        dynamic=False,
    )
    # ultralytics 导出时文件名为相同目录下的 best.onnx
    src_onnx = str(Path(pt_path).parent / "best.onnx")
    if os.path.exists(src_onnx):
        shutil.copy2(src_onnx, onnx_path)
        print(f"  ✅ ONNX 已保存: {onnx_path}")
    else:
        print(f"  ⚠️ 未找到导出文件: {src_onnx}")

def evaluate_model(pt_path: str, data_yaml: str):
    """评估模型性能"""
    print(f"\n📊 评估模型: {pt_path}")
    from ultralytics import YOLO
    model = YOLO(pt_path)
    metrics = model.val(data=data_yaml, imgsz=640, device=0)
    
    print(f"  mAP50:    {metrics.box.map50:.4f}")
    print(f"  mAP50-95: {metrics.box.map:.4f}")
    print(f"  Precision: {metrics.box.mp:.4f}")
    print(f"  Recall:    {metrics.box.mr:.4f}")
    
    return {
        "mAP50": float(metrics.box.map50),
        "mAP50-95": float(metrics.box.map),
        "precision": float(metrics.box.mp),
        "recall": float(metrics.box.mr),
    }

def main():
    parser = argparse.ArgumentParser(description="YOLOv8 模型训练")
    parser.add_argument("--model", choices=["head", "person", "all"], default="all")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--person-epochs", type=int, default=150)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--eval-only", action="store_true", help="仅评估，不训练")
    args = parser.parse_args()
    
    results = {}
    
    if args.eval_only:
        print("📊 仅评估模式")
        head_pt = str(LOG_DIR / "head_detection" / "weights" / "best.pt")
        person_pt = str(LOG_DIR / "person_detection" / "weights" / "best.pt")
        
        if os.path.exists(head_pt):
            evaluate_model(head_pt, str(BASE_DIR / "configs" / "head_detection.yaml"))
        if os.path.exists(person_pt):
            evaluate_model(person_pt, str(BASE_DIR / "configs" / "person_detection.yaml"))
        return
    
    if args.model in ("head", "all"):
        train_head_model(epochs=args.epochs, batch=args.batch)
        head_pt = str(LOG_DIR / "head_detection" / "weights" / "best.pt")
        if os.path.exists(head_pt):
            results["head"] = evaluate_model(
                head_pt,
                str(BASE_DIR / "configs" / "head_detection.yaml")
            )
    
    if args.model in ("person", "all"):
        train_person_model(epochs=args.person_epochs, batch=args.batch)
        person_pt = str(LOG_DIR / "person_detection" / "weights" / "best.pt")
        if os.path.exists(person_pt):
            results["person"] = evaluate_model(
                person_pt,
                str(BASE_DIR / "configs" / "person_detection.yaml")
            )
    
    # 保存训练结果
    result_file = LOG_DIR / f"training_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.yaml"
    with open(result_file, 'w', encoding='utf-8') as f:
        yaml.dump({
            "timestamp": datetime.now().isoformat(),
            "models": results,
            "epochs": {"head": args.epochs, "person": args.person_epochs},
            "batch_size": args.batch,
        }, f, allow_unicode=True)
    print(f"\n📝 训练结果已保存: {result_file}")

if __name__ == "__main__":
    main()
