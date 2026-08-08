"""main.py - AI 抽烟检测系统主程序入口（完整版）"""
import yaml, sys, os
from camera_manager import CameraManager
from camera_scanner import CameraScanner
from detector import MultiStageDetector
from decision_engine import DecisionEngine
from alerter import Alerter
from logger import Logger
from database import FeedbackDB
from reinforcement import ReinforcementLearner
from gui import DetectionGUI


def load_config(path="config.yaml"):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def print_banner():
    print("=" * 60)
    print("   AI 抽烟检测系统 - 安防智能体")
    print("   多级检测 | 强化学习 | 智能去重 | 管理员审核")
    print("   完全本地推理 | 零网络依赖")
    print("=" * 60)


def main():
    print_banner()
    cfg = load_config()

    # 自动扫描摄像头
    print("[System] 正在扫描摄像头...")
    scanned = CameraScanner.scan_all()
    if scanned:
        # 合并到配置中
        existing_ids = {c["id"] for c in cfg.get("cameras", [])}
        for cam in scanned:
            if cam["id"] not in existing_ids:
                cfg["cameras"].append({
                    "id": cam["id"],
                    "name": cam["name"],
                    "source": cam["source"],
                    "enabled": True,
                })
        print(f"[System] 共发现 {len(scanned)} 个摄像头")

    print("[System] 正在初始化本地数据库...")
    db = FeedbackDB("alerts/feedback.db")
    rl = ReinforcementLearner(db)

    print("[System] 正在初始化多级检测引擎...")
    detector = MultiStageDetector(
        person_model_path=cfg["model"]["person_model"],
        smoking_model_path=cfg["model"]["smoking_model"],
        pose_model_path=cfg["model"].get("pose_model", "models/yolov8n-pose.onnx"),
        conf_threshold=cfg["model"].get("confidence_threshold", 0.35),
        iou_threshold=cfg["model"].get("iou_threshold", 0.45),
    )
    print(f"[System] 检测引擎就绪 - 后端: {detector.backend_name}")

    decision = DecisionEngine(cfg["decision"])
    alerter = Alerter(cfg["alert"])
    logger = Logger(cfg["logging"])
    cameras = CameraManager(cfg["cameras"])

    logger.info(f"系统启动 - 推理后端: {detector.backend_name}")
    logger.info(f"活跃摄像头: {len(cameras.active_cameras)} 个")
    logger.info(f"数据库: alerts/feedback.db")

    # 启动 GUI
    gui = DetectionGUI(cfg, cameras, detector, decision, alerter, logger, db, rl)
    gui.run()


if __name__ == "__main__":
    main()