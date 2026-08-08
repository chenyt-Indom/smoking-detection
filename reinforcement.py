"""
reinforcement.py - 强化学习反馈机制 v2
基于管理员反馈动态调整检测阈值 + 置信度校准 + 自动重训练
"""
import json, os, time
import numpy as np
from datetime import datetime


class ReinforcementLearner:
    """强化学习：根据反馈持续调整检测参数"""

    def __init__(self, db, save_path="models/rl_params.json"):
        self._db = db
        self._save_path = save_path
        self._params = self._load_params()
        self._feedback_buffer = []
        # 置信度历史：从持久化参数中恢复（长期记忆）
        self._confidence_history = self._params.get("_confidence_history", {})
        if not isinstance(self._confidence_history, dict):
            self._confidence_history = {}

    def _load_params(self):
        if os.path.exists(self._save_path):
            with open(self._save_path, "r") as f:
                return json.load(f)
        return {
            "base_thresholds": {"cigarette": 0.35, "cigarette_pack": 0.35},
            "adaptive_thresholds": {"cigarette": 0.35, "cigarette_pack": 0.35},
            "learning_rate": 0.05,
            "reward_history": [],
            "total_episodes": 0,
            "total_feedbacks": 0,
            "last_retrain_at": None,
            "_confidence_history": {},
        }

    def save_params(self):
        os.makedirs(os.path.dirname(self._save_path), exist_ok=True)
        self._params["total_episodes"] += 1
        # 持久化置信度历史（长期记忆：只保留最近50条避免文件过大）
        for label in list(self._confidence_history.keys()):
            if len(self._confidence_history[label]) > 50:
                self._confidence_history[label] = self._confidence_history[label][-50:]
        self._params["_confidence_history"] = self._confidence_history
        with open(self._save_path, "w") as f:
            json.dump(self._params, f, indent=2)

    def get_threshold(self, label):
        """获取自适应阈值"""
        return self._params["adaptive_thresholds"].get(label, 0.35)

    def process_feedback(self, label, was_correct, confidence):
        """处理单条反馈 — 即时调整阈值"""
        self._feedback_buffer.append({
            "label": label,
            "was_correct": was_correct,
            "confidence": confidence,
            "timestamp": datetime.now().isoformat(),
        })

        # 记录置信度历史用于校准
        if label not in self._confidence_history:
            self._confidence_history[label] = []
        self._confidence_history[label].append((confidence, was_correct))
        if len(self._confidence_history[label]) > 100:
            self._confidence_history[label] = self._confidence_history[label][-100:]

        self._params["total_feedbacks"] = self._params.get("total_feedbacks", 0) + 1

        # 即时微调阈值（单条反馈）
        self._immediate_adjust(label, was_correct, confidence)

        # 每收集 5 条反馈做一次批量调整
        if len(self._feedback_buffer) >= 5:
            self._adjust_thresholds()

    def _immediate_adjust(self, label, was_correct, confidence):
        """即时微调：基于单条反馈小步调整"""
        lr = self._params["learning_rate"] * 0.2  # 即时调整学习率更小
        current = self._params["adaptive_thresholds"].get(label, 0.35)

        if was_correct:
            # 正确检测 → 略微降低阈值（更敏感）
            new_threshold = max(0.20, current - lr * 0.3)
        else:
            # 误报 → 提高阈值（更严格）
            new_threshold = min(0.70, current + lr * 0.5)

        self._params["adaptive_thresholds"][label] = round(new_threshold, 4)

    def _adjust_thresholds(self):
        """批量调整：基于最近反馈的准确率"""
        lr = self._params["learning_rate"]

        for label in ["cigarette", "cigarette_pack"]:
            label_feedbacks = [f for f in self._feedback_buffer if f["label"] == label]
            if not label_feedbacks:
                continue

            correct = sum(1 for f in label_feedbacks if f["was_correct"])
            total = len(label_feedbacks)
            accuracy = correct / max(total, 1)

            current = self._params["adaptive_thresholds"].get(label, 0.35)

            if accuracy >= 0.9:
                new_threshold = max(0.20, current - lr * 0.5)
            elif accuracy >= 0.7:
                new_threshold = current
            elif accuracy >= 0.5:
                new_threshold = min(0.60, current + lr)
            else:
                new_threshold = min(0.70, current + lr * 2)

            self._params["adaptive_thresholds"][label] = round(new_threshold, 4)

        # 记录奖励历史
        total_correct = sum(1 for f in self._feedback_buffer if f["was_correct"])
        self._params["reward_history"].append({
            "episode": self._params["total_episodes"],
            "correct": total_correct,
            "total": len(self._feedback_buffer),
            "accuracy": total_correct / max(len(self._feedback_buffer), 1),
        })

        if len(self._params["reward_history"]) > 100:
            self._params["reward_history"] = self._params["reward_history"][-100:]

        self._feedback_buffer = []
        self.save_params()

    def calibrate_confidence(self, label, raw_confidence):
        """置信度校准：基于历史反馈调整置信度"""
        if label not in self._confidence_history or len(self._confidence_history[label]) < 10:
            return raw_confidence

        history = self._confidence_history[label]
        # 计算该置信度区间的历史准确率
        correct_same = sum(1 for c, w in history if abs(c - raw_confidence) < 0.05 and w)
        total_same = sum(1 for c, w in history if abs(c - raw_confidence) < 0.05)
        if total_same >= 3:
            calibrated = correct_same / total_same
            # 混合原始置信度和校准置信度
            return raw_confidence * 0.6 + calibrated * 0.4
        return raw_confidence

    def get_optimal_threshold(self, label, target_precision=0.85):
        """基于历史数据计算最优阈值（达到目标精度）"""
        if label not in self._confidence_history or len(self._confidence_history[label]) < 20:
            return self.get_threshold(label)

        history = sorted(self._confidence_history[label], key=lambda x: x[0])
        for conf, was_correct in history:
            # 计算高于该阈值的精度
            above = [(c, w) for c, w in history if c >= conf]
            if not above:
                continue
            precision = sum(1 for _, w in above if w) / len(above)
            if precision >= target_precision:
                return conf
        return self.get_threshold(label)

    def should_retrain(self):
        """判断是否需要重新训练模型"""
        if len(self._params["reward_history"]) < 10:
            return False

        recent_5 = self._params["reward_history"][-5:]
        avg_acc = sum(r["accuracy"] for r in recent_5) / len(recent_5)

        # 准确率低于60% 或 反馈量超过50条且准确率低于70%
        if avg_acc < 0.6:
            return True
        if self._params.get("total_feedbacks", 0) > 50 and avg_acc < 0.7:
            return True

        return False

    def get_retrain_recommendation(self):
        """获取重训练建议"""
        if not self.should_retrain():
            return None

        recommendations = []
        for label in ["cigarette", "cigarette_pack"]:
            if label in self._confidence_history:
                history = self._confidence_history[label]
                if len(history) >= 10:
                    correct = sum(1 for _, w in history if w)
                    total = len(history)
                    acc = correct / max(total, 1)
                    if acc < 0.7:
                        recommendations.append({
                            "label": label,
                            "accuracy": acc,
                            "samples": total,
                            "threshold": self.get_threshold(label),
                            "suggested_threshold": self.get_optimal_threshold(label),
                        })

        return recommendations if recommendations else None

    def get_stats(self):
        """获取强化学习统计"""
        recent = self._params["reward_history"][-10:] if self._params["reward_history"] else []
        recent_acc = sum(r["accuracy"] for r in recent) / len(recent) if recent else 0.0

        return {
            "thresholds": self._params["adaptive_thresholds"],
            "total_episodes": self._params["total_episodes"],
            "total_feedbacks": self._params.get("total_feedbacks", 0),
            "recent_accuracy": recent_acc,
            "learning_rate": self._params["learning_rate"],
            "should_retrain": self.should_retrain(),
        }

    def get_learning_curve(self):
        """获取学习曲线数据"""
        return self._params.get("reward_history", [])