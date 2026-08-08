"""
headless_mode.py - 后台无界面运行模式
无需打开GUI窗口，所有检测流程正常运行
截图自动保存，支持查看运行日志
"""
import yaml, os, time, sys, signal, threading
import cv2
from datetime import datetime
from camera_manager import CameraManager
from camera_scanner import CameraScanner
from detector import MultiStageDetector
from decision_engine import DecisionEngine
from alerter import Alerter
from logger import Logger
from database import FeedbackDB
from reinforcement import ReinforcementLearner
from person_identifier import PersonIdentifier


class HeadlessRunner:
    """后台运行器"""

    def __init__(self, config_path="config.yaml"):
        with open(config_path, "r", encoding="utf-8") as f:
            self._cfg = yaml.safe_load(f)

        # 初始化各模块
        self._db = FeedbackDB("alerts/feedback.db")
        self._rl = ReinforcementLearner(self._db)
        self._logger = Logger(self._cfg["logging"])
        self._alerter = Alerter(self._cfg["alert"])

        self._detector = MultiStageDetector(
            person_model_path=self._cfg["model"]["person_model"],
            smoking_model_path=self._cfg["model"]["smoking_model"],
            pose_model_path=self._cfg["model"].get("pose_model", "models/yolov8n-pose.onnx"),
            conf_threshold=self._cfg["model"].get("confidence_threshold", 0.35),
            iou_threshold=self._cfg["model"].get("iou_threshold", 0.45),
        )
        self._decision = DecisionEngine(self._cfg["decision"])
        self._person_id = PersonIdentifier(time_window=30)

        self._cameras = CameraManager(self._cfg["cameras"])
        self._running = False
        self._alert_count = 0
        self._frame_count = 0

    def start(self):
        """启动后台检测"""
        self._running = True
        self._logger.info("=" * 50)
        self._logger.info(f"后台模式启动 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self._logger.info(f"推理后端: {self._detector.backend_name}")
        self._logger.info(f"活跃摄像头: {len(self._cameras.active_cameras)} 个")
        self._logger.info(f"模型: 人体检测 + 香烟/烟盒检测(2类)")
        self._logger.info(f"数据库: alerts/feedback.db")
        self._logger.info("=" * 50)

        # 启动摄像头
        for cam in self._cameras.active_cameras:
            try:
                cam.start()
                self._logger.info(f"摄像头已启动: {cam.name}")
            except RuntimeError as e:
                self._logger.info(f"摄像头启动失败: {cam.name} - {e}")

        interval = self._cfg.get("performance", {}).get("inference_interval", 1)
        frame_idx = 0
        last_alert_time = {}
        last_stats_time = time.time()

        try:
            while self._running:
                for cam in self._cameras.active_cameras:
                    frame = cam.read()
                    if frame is None:
                        continue

                    frame_idx += 1

                    if frame_idx % interval != 0:
                        continue

                    # 多级检测
                    result = self._detector.detect(frame)

                    # 过滤告警
                    valid_alerts = []
                    for alert in result["alerts"]:
                        label = alert["label"]
                        conf = alert["confidence"]
                        tid = alert["track_id"]

                        # 自适应阈值
                        rl_threshold = self._rl.get_threshold(label)
                        if conf < rl_threshold:
                            continue

                        # 智能去重
                        person_key = f"{cam.id}_{tid}"
                        now = time.time()
                        if person_key in last_alert_time:
                            lt, lc, _ = last_alert_time[person_key]
                            if now - lt < 30 and conf <= lc:
                                continue
                        last_alert_time[person_key] = (now, conf, alert["bbox"])
                        valid_alerts.append(alert)

                    # 决策链
                    should_alert, detail = self._decision.evaluate(cam.id, valid_alerts)
                    if should_alert and valid_alerts:
                        best = max(valid_alerts, key=lambda a: a["confidence"])

                        # 提取人物特征
                        person_features = {}
                        track_id = best.get("track_id", 0)
                        for p in result["persons"]:
                            if p["id"] == track_id:
                                person_features = self._person_id.extract_features(frame, p["bbox"])
                                break

                        # 保存截图
                        alert_dir = "alerts"
                        os.makedirs(alert_dir, exist_ok=True)
                        ts = time.strftime("%Y%m%d_%H%M%S")
                        img_path = os.path.join(alert_dir, f"{cam.id}_{best['label']}_{ts}.jpg")
                        cv2.imwrite(img_path, frame)

                        # 存入数据库
                        self._db.add_evidence(
                            cam.id, cam.name, f"person_{track_id}",
                            img_path, best["label"], best["confidence"],
                            best["bbox"], person_features,
                        )

                        self._alert_count += 1
                        self._logger.info(
                            f"告警 #{self._alert_count}: {cam.name} "
                            f"{best['label']} 置信度{best['confidence']:.0%} -> {img_path}"
                        )

                    self._frame_count += 1

                    # 每30秒输出统计
                    if time.time() - last_stats_time > 30:
                        stats = self._db.get_stats_for_display()
                        rl_stats = self._rl.get_stats()
                        self._logger.info(
                            f"运行统计: 处理{self._frame_count}帧 | "
                            f"告警{self._alert_count}次 | "
                            f"准确率{stats['accuracy']} | "
                            f"待审{stats['pending']} | "
                            f"推理{self._detector.avg_inference_time_ms:.1f}ms"
                        )
                        last_stats_time = time.time()

                    time.sleep(0.001)

        except KeyboardInterrupt:
            self._logger.info("收到中断信号，正在停止...")
        finally:
            self.stop()

    def stop(self):
        self._running = False
        for cam in self._cameras.active_cameras:
            cam.stop()
        stats = self._db.get_stats_for_display()
        self._logger.info(f"后台模式已停止 - 共处理{self._frame_count}帧, 告警{self._alert_count}次")
        self._logger.info(f"准确率: {stats['accuracy']}, 待审核: {stats['pending']}")


def run_headless(config_path="config.yaml"):
    runner = HeadlessRunner(config_path)
    runner.start()


if __name__ == "__main__":
    config = sys.argv[1] if len(sys.argv) > 1 else "config.yaml"
    run_headless(config)