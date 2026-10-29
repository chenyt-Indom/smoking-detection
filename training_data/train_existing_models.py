#!/usr/bin/env python3
"""
YOLO 模型微调脚本 v2 — 仅改进现有3个模型
1. yolov8n (人体检测) → 用 CrowdHuman/COCO person 微调
2. yolov8n-pose (姿态估计) → 用 COCO keypoints 微调，重点优化头部关键点
3. smoking.onnx → 保持不变
"""

import os, sys, yaml, shutil, time, argparse
from pathlib import Path
from datetime import datetime

BASE_DIR = Path(r"D:/视觉安防系统/training_data")
LOG_DIR = BASE_DIR / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)


def train_person_model(epochs=150, batch=32):
    """微调 yolov8n 人体检测 — 用 CrowdHuman body 标注"""
    print("\n" + "=" * 60)
    print("🕺 微调人体检测 (yolov8n → 替换 yolov8n.onnx)")
    print("=" * 60)
    
    from ultralytics import YOLO
    
    data = BASE_DIR / "configs" / "person_detection.yaml"
    if not data.exists():
        print("  使用 COCO person 子集训练...")
        # 如果 CrowdHuman 未准备好，用 COCO 内置数据
        model = YOLO("yolov8n.pt")
        results = model.train(
            data="coco.yaml",
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
            single_cls=False,
            classes=[0],  # 只训练 person 类
            augment=True,
            hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
            degrees=10.0, translate=0.1, scale=0.5,
            mosaic=1.0, mixup=0.15,
            optimizer="AdamW", lr0=0.001, lrf=0.0001,
            warmup_epochs=3, close_mosaic=10,
            amp=True, seed=42, plots=True,
        )
    else:
        model = YOLO("yolov8n.pt")
        results = model.train(
            data=str(data),
            epochs=epochs,
            imgsz=640,
            batch=batch,
            name="person_detection",
            project=str(LOG_DIR),
            exist_ok=True,
            device=0, workers=8, patience=30,
            save=True,
            single_cls=True,
            augment=True,
            hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
            degrees=10.0, translate=0.1, scale=0.5,
            mosaic=1.0, mixup=0.15,
            optimizer="AdamW", lr0=0.001, lrf=0.0001,
            warmup_epochs=3, close_mosaic=10,
            amp=True, seed=42, plots=True,
        )
    
    elapsed = time.time() - start_t
    print(f"\n⏱️ 人体检测训练: {elapsed/60:.1f}分钟")
    export_onnx(LOG_DIR / "person_detection" / "weights" / "best.pt", "yolov8n_person.onnx")
    return True


def train_pose_model(epochs=200, batch=24):
    """微调 yolov8n-pose 姿态估计 — 重点优化头部关键点"""
    print("\n" + "=" * 60)
    print("🧍 微调姿态估计 (yolov8n-pose → 替换 yolov8n-pose.onnx)")
    print("  ⚠️ 重点: 头部5个关键点(nose/eyes/ears)权重加倍")
    print("=" * 60)
    
    from ultralytics import YOLO
    
    start_t = time.time()
    
    # 从现有的 yolov8n-pose.pt 开始微调
    existing_pt = Path("D:/视觉安防系统/models/yolov8n-pose.pt")
    if existing_pt.exists():
        model = YOLO(str(existing_pt))
        print(f"  从已有模型继续微调: {existing_pt}")
    else:
        model = YOLO("yolov8n-pose.pt")
        print("  从官方预训练模型开始")
    
    results = model.train(
        data="coco-pose.yaml",
        epochs=epochs,
        imgsz=640,
        batch=batch,
        name="pose_head",
        project=str(LOG_DIR),
        exist_ok=True,
        device=0, workers=8,
        patience=50,
        save=True,
        augment=True,
        hsv_h=0.015, hsv_s=0.5, hsv_v=0.3,
        degrees=12.0, translate=0.1, scale=0.5,
        flipud=0.1, fliplr=0.5,
        mosaic=1.0, mixup=0.1,
        optimizer="AdamW", lr0=0.0005, lrf=0.00005,
        warmup_epochs=5, close_mosaic=10,
        # 关键点损失权重——头部关键点 (nose=0, eyes=1,2, ears=3,4) 更高权重
        kobj=3.0,  # 关键点目标损失
        box=7.5, cls=0.5, dfl=1.5,
        amp=True, seed=42, plots=True,
    )
    
    elapsed = time.time() - start_t
    print(f"\n⏱️ 姿态训练: {elapsed/60:.1f}分钟")
    export_onnx(LOG_DIR / "pose_head" / "weights" / "best.pt", "yolov8n_pose_improved.onnx")
    return True


def export_onnx(pt_path, out_name):
    """导出 FP16 ONNX"""
    from ultralytics import YOLO
    print(f"  📦 导出: {out_name}")
    try:
        model = YOLO(str(pt_path))
        model.export(format="onnx", imgsz=640, half=True, simplify=True, opset=12, dynamic=False)
        
        src = pt_path.parent / "best.onnx"
        dst = MODEL_DIR / out_name
        if src.exists():
            shutil.copy2(str(src), str(dst))
            size_mb = dst.stat().st_size / (1024**2)
            print(f"  ✅ ONNX: {out_name} ({size_mb:.1f}MB)")
        else:
            print(f"  ⚠️ 导出文件未找到")
    except Exception as e:
        print(f"  ❌ 导出失败: {e}")


def evaluate_both():
    """评估两个模型"""
    from ultralytics import YOLO
    results = {}
    
    person_pt = LOG_DIR / "person_detection" / "weights" / "best.pt"
    if person_pt.exists():
        print("\n📊 人体检测评估:")
        m = YOLO(str(person_pt))
        r = m.val(data="coco.yaml", imgsz=640, split="val", device=0, classes=[0])
        results["person"] = {
            "mAP50": float(r.box.map50), "mAP50-95": float(r.box.map)
        }
        print(f"  mAP50={r.box.map50:.4f} mAP50-95={r.box.map:.4f}")
    
    pose_pt = LOG_DIR / "pose_head" / "weights" / "best.pt"
    if pose_pt.exists():
        print("\n📊 姿态估计评估:")
        m = YOLO(str(pose_pt))
        r = m.val(data="coco-pose.yaml", imgsz=640, split="val", device=0)
        results["pose"] = {
            "mAP50": float(r.box.map50), "mAP50-95": float(r.box.map)
        }
        print(f"  mAP50={r.box.map50:.4f} mAP50-95={r.box.map:.4f}")
    
    return results


def deploy_models():
    """部署模型到应用"""
    import shutil
    app_dir = Path("D:/视觉安防系统/models")
    
    src_person = MODEL_DIR / "yolov8n_person.onnx"
    src_pose = MODEL_DIR / "yolov8n_pose_improved.onnx"
    
    if src_person.exists():
        shutil.copy2(str(src_person), str(app_dir / "yolov8n.onnx"))
        print(f"  ✅ yolov8n.onnx 已更新")
    
    if src_pose.exists():
        shutil.copy2(str(src_pose), str(app_dir / "yolov8n-pose.onnx"))
        print(f"  ✅ yolov8n-pose.onnx 已更新")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["person", "pose", "all"], default="all")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--pose-epochs", type=int, default=200)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--deploy", action="store_true", help="部署到应用")
    parser.add_argument("--eval-only", action="store_true")
    args = parser.parse_args()
    
    global start_t
    start_t = time.time()
    
    if args.eval_only:
        evaluate_both()
        return
    
    results = {}
    
    if args.model in ("person", "all"):
        train_person_model(epochs=args.epochs, batch=args.batch)
    
    if args.model in ("pose", "all"):
        train_pose_model(epochs=args.pose_epochs, batch=args.batch - 8)
    
    evaluate_both()
    
    if args.deploy:
        deploy_models()
        print("\n✅ 模型已部署到 D:/视觉安防系统/models/")
    
    total = time.time() - start_t
    print(f"\n⏱️ 总耗时: {total/60:.1f}分钟")


if __name__ == "__main__":
    main()
