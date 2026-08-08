"""
model_downloader.py - 抽烟检测模型准备工具
支持从 Roboflow 下载预训练数据集，自动训练 YOLOv8 并导出 ONNX

使用方法:
    # 方式1: 从 Roboflow 下载数据集并训练
    python model_downloader.py --roboflow-api-key YOUR_KEY

    # 方式2: 使用已有数据集训练
    python model_downloader.py --dataset ./smoking_dataset

    # 方式3: 下载预训练 YOLOv8n 作为人体检测模型
    python model_downloader.py --download-yolov8n

注意: 此工具仅在首次部署时需要联网下载数据集和预训练权重。
      训练完成后，生成的 ONNX 模型完全本地运行，无需网络。
"""
import os
import sys
import argparse
import subprocess


def download_yolov8n():
    """下载 YOLOv8n 预训练模型并导出 ONNX（人体检测用）"""
    print("=" * 50)
    print("  下载 YOLOv8n 并导出 ONNX")
    print("=" * 50)

    try:
        from ultralytics import YOLO
    except ImportError:
        print("❌ 请先安装 ultralytics: pip install ultralytics")
        sys.exit(1)

    os.makedirs("models", exist_ok=True)
    target = "models/yolov8n.onnx"

    if os.path.exists(target):
        print(f"✅ {target} 已存在，跳过")
        return

    print("→ 下载 YOLOv8n.pt ...")
    model = YOLO("yolov8n.pt")
    print("→ 导出 ONNX (FP16, 416x416)...")
    model.export(format="onnx", imgsz=416, half=True)
    # 移动文件
    src = "yolov8n.onnx"
    if os.path.exists(src):
        os.rename(src, target)
        print(f"✅ 已生成: {target}")
    print(f"✅ YOLOv8n ONNX 导出完成")


def train_from_roboflow(api_key: str):
    """从 Roboflow 下载抽烟检测数据集并训练"""
    print("=" * 50)
    print("  从 Roboflow 下载数据集并训练抽烟检测模型")
    print("=" * 50)

    try:
        from roboflow import Roboflow
        from ultralytics import YOLO
    except ImportError:
        print("❌ 请先安装依赖: pip install roboflow ultralytics")
        sys.exit(1)

    # 下载数据集
    print("→ 连接 Roboflow...")
    rf = Roboflow(api_key=api_key)

    # 推荐的抽烟检测数据集（Roboflow Universe 上公开可用）
    # 数据集: Smoking Detection (包含 cigarette, person, smoke, vape 等类别)
    print("→ 搜索抽烟检测数据集...")
    print("  提示: 请在 Roboflow Universe 搜索 'smoking detection'")
    print("  推荐数据集: Smoking_person (多类别), Cigarette Detection 等")
    print()
    print("  请手动执行以下步骤:")
    print("  1. 访问 https://universe.roboflow.com/")
    print("  2. 搜索 'smoking detection' 或 'cigarette detection'")
    print("  3. 下载 YOLOv8 格式数据集")
    print("  4. 解压到 datasets/ 目录")
    print()
    print("  或者使用以下代码自动下载:")
    print()
    print('  project = rf.workspace("WORKSPACE").project("PROJECT")')
    print('  dataset = project.version(VERSION).download("yolov8")')
    print()
    print("  然后运行: python model_downloader.py --dataset ./datasets/xxx")
    print("=" * 50)


def train_from_dataset(dataset_path: str):
    """使用本地数据集训练抽烟检测模型"""
    print("=" * 50)
    print("  训练抽烟检测模型")
    print("=" * 50)

    try:
        from ultralytics import YOLO
    except ImportError:
        print("❌ 请先安装 ultralytics: pip install ultralytics")
        sys.exit(1)

    # 检查数据集
    data_yaml = os.path.join(dataset_path, "data.yaml")
    if not os.path.exists(data_yaml):
        print(f"❌ 数据集配置文件不存在: {data_yaml}")
        print("  请确保数据集目录包含 data.yaml")
        sys.exit(1)

    print(f"✅ 数据集: {dataset_path}")
    print(f"→ 加载 YOLOv8n 预训练权重...")
    model = YOLO("yolov8n.pt")

    print("→ 开始训练...")
    print("  参数: epochs=50, imgsz=416, batch=16")
    model.train(
        data=data_yaml,
        epochs=50,
        imgsz=416,
        batch=16,
        name="smoking_detection",
        project="runs",
        exist_ok=True,
    )

    # 导出 ONNX
    print("→ 训练完成，导出 ONNX 模型...")
    best_pt = "runs/smoking_detection/weights/best.pt"
    if os.path.exists(best_pt):
        model = YOLO(best_pt)
        model.export(format="onnx", imgsz=416, half=True)
        os.makedirs("models", exist_ok=True)
        if os.path.exists("runs/smoking_detection/weights/best.onnx"):
            os.rename("runs/smoking_detection/weights/best.onnx", "models/smoking.onnx")
            print("✅ 模型已导出: models/smoking.onnx")
        print("✅ 全部完成！")
    else:
        print("⚠️ 训练权重未找到，请检查训练日志")


def print_guide():
    """打印完整使用指南"""
    print("""
╔══════════════════════════════════════════════════════╗
║        AI 抽烟检测系统 - 模型准备指南                ║
╠══════════════════════════════════════════════════════╣
║                                                      ║
║  方式一: 从 Roboflow 获取公开数据集训练              ║
║  ────────────────────────────────────────             ║
║  1. 访问 https://universe.roboflow.com/              ║
║  2. 搜索 "smoking detection"                         ║
║  3. 下载 YOLOv8 格式数据集                           ║
║  4. 解压后运行:                                      ║
║     python model_downloader.py --dataset ./数据集目录  ║
║                                                      ║
║  方式二: 使用已有数据集训练                          ║
║  ─────────────────────────────                       ║
║  python model_downloader.py --dataset ./smoking_data  ║
║                                                      ║
║  方式三: 下载人体检测模型 (YOLOv8n)                  ║
║  ──────────────────────────────────                  ║
║  python model_downloader.py --download-yolov8n        ║
║                                                      ║
║  输出文件:                                           ║
║  models/smoking.onnx  - 抽烟检测模型                  ║
║  models/yolov8n.onnx  - 人体检测模型                  ║
║                                                      ║
╚══════════════════════════════════════════════════════╝
""")


def main():
    parser = argparse.ArgumentParser(description="抽烟检测模型准备工具")
    parser.add_argument("--dataset", type=str, help="本地数据集路径")
    parser.add_argument("--roboflow-api-key", type=str, help="Roboflow API Key")
    parser.add_argument("--download-yolov8n", action="store_true", help="下载 YOLOv8n 人体检测模型")
    parser.add_argument("--guide", action="store_true", help="显示使用指南")

    args = parser.parse_args()

    if args.guide or len(sys.argv) == 1:
        print_guide()
        return

    if args.download_yolov8n:
        download_yolov8n()

    if args.roboflow_api_key:
        train_from_roboflow(args.roboflow_api_key)

    if args.dataset:
        train_from_dataset(args.dataset)


if __name__ == "__main__":
    main()