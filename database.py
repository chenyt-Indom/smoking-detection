"""
database.py - 本地 SQLite 数据库
存储截图证据、管理员反馈、模型评分、强化学习数据
v2: 图片BLOB长期保存 + 删除 + 标注 + 全量查询
"""
import sqlite3, os, json, time
import numpy as np
from datetime import datetime
from threading import Lock


class FeedbackDB:
    """本地反馈数据库"""

    def __init__(self, db_path="alerts/feedback.db"):
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self._db_path = db_path
        self._lock = Lock()
        self._init_tables()

    def _init_tables(self):
        with self._lock:
            conn = sqlite3.connect(self._db_path)
            c = conn.cursor()

            # 截图证据表
            c.execute("""
                CREATE TABLE IF NOT EXISTS evidence (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    camera_id TEXT NOT NULL,
                    camera_name TEXT,
                    person_id TEXT,
                    image_path TEXT NOT NULL,
                    image_blob BLOB,
                    label TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    bbox TEXT,
                    person_features TEXT,
                    timestamp TEXT NOT NULL,
                    admin_judgment TEXT DEFAULT 'pending',
                    admin_score INTEGER DEFAULT 0,
                    admin_note TEXT,
                    admin_annotation TEXT,
                    judged_at TEXT,
                    created_at TEXT DEFAULT (datetime('now','localtime'))
                )
            """)

            # 迁移：为旧数据库添加缺失的列
            cols = {row[1] for row in c.execute("PRAGMA table_info(evidence)")}
            if "image_blob" not in cols:
                c.execute("ALTER TABLE evidence ADD COLUMN image_blob BLOB")
            if "admin_annotation" not in cols:
                c.execute("ALTER TABLE evidence ADD COLUMN admin_annotation TEXT")
            if "person_features" not in cols:
                c.execute("ALTER TABLE evidence ADD COLUMN person_features TEXT")

            # 强化学习反馈表
            c.execute("""
                CREATE TABLE IF NOT EXISTS feedback (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    evidence_id INTEGER,
                    camera_id TEXT,
                    label TEXT,
                    confidence REAL,
                    was_correct INTEGER,
                    created_at TEXT DEFAULT (datetime('now','localtime')),
                    FOREIGN KEY (evidence_id) REFERENCES evidence(id)
                )
            """)

            # 模型评分表
            c.execute("""
                CREATE TABLE IF NOT EXISTS model_scores (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    label TEXT NOT NULL,
                    total_correct INTEGER DEFAULT 0,
                    total_incorrect INTEGER DEFAULT 0,
                    accuracy REAL DEFAULT 0.0,
                    confidence_threshold REAL DEFAULT 0.35,
                    updated_at TEXT DEFAULT (datetime('now','localtime'))
                )
            """)

            # 人物特征缓存表
            c.execute("""
                CREATE TABLE IF NOT EXISTS person_cache (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    person_id TEXT NOT NULL,
                    camera_id TEXT,
                    features TEXT NOT NULL,
                    last_seen TEXT DEFAULT (datetime('now','localtime')),
                    detection_count INTEGER DEFAULT 1
                )
            """)

            conn.commit()
            conn.close()

    def add_evidence(self, camera_id, camera_name, person_id, image_path,
                     label, confidence, bbox, person_features):
        """添加截图证据，同时保存图片BLOB用于长期存储"""
        with self._lock:
            conn = sqlite3.connect(self._db_path)
            c = conn.cursor()

            # 读取图片二进制数据用于长期存储
            image_blob = None
            if os.path.exists(image_path):
                try:
                    with open(image_path, "rb") as f:
                        image_blob = f.read()
                except Exception:
                    pass

            c.execute("""
                INSERT INTO evidence (camera_id, camera_name, person_id, image_path,
                    image_blob, label, confidence, bbox, person_features, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (camera_id, camera_name, person_id, image_path, image_blob,
                  label, confidence, json.dumps(bbox), json.dumps(person_features),
                  datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
            eid = c.lastrowid
            conn.commit()
            conn.close()
            return eid

    def judge_evidence(self, evidence_id, is_correct, note=""):
        """管理员判断：正确=+1分，错误=-1分"""
        with self._lock:
            conn = sqlite3.connect(self._db_path)
            c = conn.cursor()

            judgment = "correct" if is_correct else "incorrect"
            score = 1 if is_correct else -1
            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            c.execute("""
                UPDATE evidence SET admin_judgment=?, admin_score=?, admin_note=?, judged_at=?
                WHERE id=?
            """, (judgment, score, note, now, evidence_id))

            c.execute("SELECT label, confidence, camera_id FROM evidence WHERE id=?", (evidence_id,))
            row = c.fetchone()
            if row:
                label, confidence, cam_id = row
                c.execute("""
                    INSERT INTO feedback (evidence_id, camera_id, label, confidence, was_correct)
                    VALUES (?, ?, ?, ?, ?)
                """, (evidence_id, cam_id, label, confidence, 1 if is_correct else 0))
                self._update_model_score(c, label, is_correct)

            conn.commit()
            conn.close()

    def annotate_evidence(self, evidence_id, annotation):
        """管理员标注截图"""
        with self._lock:
            conn = sqlite3.connect(self._db_path)
            c = conn.cursor()
            c.execute("UPDATE evidence SET admin_annotation=? WHERE id=?", (annotation, evidence_id))
            conn.commit()
            conn.close()

    def delete_evidence(self, evidence_id):
        """删除证据记录和关联的图片文件"""
        with self._lock:
            conn = sqlite3.connect(self._db_path)
            c = conn.cursor()
            c.execute("SELECT image_path FROM evidence WHERE id=?", (evidence_id,))
            row = c.fetchone()
            if row and row[0] and os.path.exists(row[0]):
                try:
                    os.remove(row[0])
                except Exception:
                    pass
            c.execute("DELETE FROM feedback WHERE evidence_id=?", (evidence_id,))
            c.execute("DELETE FROM evidence WHERE id=?", (evidence_id,))
            conn.commit()
            conn.close()

    def get_evidence_by_id(self, evidence_id):
        """获取单条证据"""
        with self._lock:
            conn = sqlite3.connect(self._db_path)
            conn.row_factory = sqlite3.Row
            c = conn.cursor()
            c.execute("SELECT * FROM evidence WHERE id=?", (evidence_id,))
            row = c.fetchone()
            conn.close()
            return dict(row) if row else None

    def get_all_evidence(self, filters=None, limit=100, offset=0):
        """获取所有证据（支持筛选）"""
        filters = filters or {}
        with self._lock:
            conn = sqlite3.connect(self._db_path)
            conn.row_factory = sqlite3.Row
            c = conn.cursor()

            query = "SELECT * FROM evidence WHERE 1=1"
            params = []

            if "label" in filters and filters["label"]:
                query += " AND label=?"
                params.append(filters["label"])
            if "camera_id" in filters and filters["camera_id"]:
                query += " AND camera_id=?"
                params.append(filters["camera_id"])
            if "judgment" in filters and filters["judgment"]:
                query += " AND admin_judgment=?"
                params.append(filters["judgment"])
            if "date_from" in filters and filters["date_from"]:
                query += " AND timestamp >= ?"
                params.append(filters["date_from"])
            if "date_to" in filters and filters["date_to"]:
                query += " AND timestamp <= ?"
                params.append(filters["date_to"])

            query += " ORDER BY created_at DESC LIMIT ? OFFSET ?"
            params.extend([limit, offset])

            c.execute(query, params)
            rows = [dict(r) for r in c.fetchall()]
            conn.close()
            return rows

    def get_evidence_count(self, filters=None):
        """获取证据总数"""
        filters = filters or {}
        with self._lock:
            conn = sqlite3.connect(self._db_path)
            c = conn.cursor()

            query = "SELECT COUNT(*) FROM evidence WHERE 1=1"
            params = []

            if "label" in filters and filters["label"]:
                query += " AND label=?"
                params.append(filters["label"])
            if "camera_id" in filters and filters["camera_id"]:
                query += " AND camera_id=?"
                params.append(filters["camera_id"])
            if "judgment" in filters and filters["judgment"]:
                query += " AND admin_judgment=?"
                params.append(filters["judgment"])

            c.execute(query, params)
            count = c.fetchone()[0]
            conn.close()
            return count

    def get_image_blob(self, evidence_id):
        """从数据库获取图片BLOB（长期存储）"""
        with self._lock:
            conn = sqlite3.connect(self._db_path)
            c = conn.cursor()
            c.execute("SELECT image_blob FROM evidence WHERE id=?", (evidence_id,))
            row = c.fetchone()
            conn.close()
            if row and row[0]:
                return row[0]
            return None

    def get_all_labels(self):
        """获取所有标签类型"""
        with self._lock:
            conn = sqlite3.connect(self._db_path)
            c = conn.cursor()
            c.execute("SELECT DISTINCT label FROM evidence")
            labels = [r[0] for r in c.fetchall()]
            conn.close()
            return labels

    def get_all_cameras(self):
        """获取所有摄像头"""
        with self._lock:
            conn = sqlite3.connect(self._db_path)
            c = conn.cursor()
            c.execute("SELECT DISTINCT camera_id, camera_name FROM evidence")
            cameras = [{"id": r[0], "name": r[1]} for r in c.fetchall()]
            conn.close()
            return cameras

    def _update_model_score(self, cursor, label, is_correct):
        """更新模型评分"""
        cursor.execute("SELECT * FROM model_scores WHERE label=?", (label,))
        row = cursor.fetchone()
        if row:
            if is_correct:
                cursor.execute("""
                    UPDATE model_scores SET total_correct=total_correct+1,
                    accuracy=CAST(total_correct+1 AS REAL)/(total_correct+total_incorrect+1),
                    updated_at=datetime('now','localtime')
                    WHERE label=?
                """, (label,))
            else:
                cursor.execute("""
                    UPDATE model_scores SET total_incorrect=total_incorrect+1,
                    accuracy=CAST(total_correct AS REAL)/(total_correct+total_incorrect+1),
                    updated_at=datetime('now','localtime')
                    WHERE label=?
                """, (label,))
        else:
            acc = 1.0 if is_correct else 0.0
            tc = 1 if is_correct else 0
            ti = 0 if is_correct else 1
            cursor.execute("""
                INSERT INTO model_scores (label, total_correct, total_incorrect, accuracy)
                VALUES (?, ?, ?, ?)
            """, (label, tc, ti, acc))

    def get_confidence_threshold(self, label):
        """根据模型评分动态调整置信度阈值"""
        with self._lock:
            conn = sqlite3.connect(self._db_path)
            c = conn.cursor()
            c.execute("SELECT accuracy, confidence_threshold FROM model_scores WHERE label=?", (label,))
            row = c.fetchone()
            conn.close()
            if row:
                accuracy, threshold = row
                if accuracy >= 0.9:
                    return max(0.25, threshold - 0.02)
                elif accuracy >= 0.8:
                    return threshold
                elif accuracy >= 0.6:
                    return min(0.55, threshold + 0.03)
                else:
                    return min(0.65, threshold + 0.05)
            return 0.35

    def get_pending_evidence(self, limit=50):
        """获取待审核的截图"""
        with self._lock:
            conn = sqlite3.connect(self._db_path)
            conn.row_factory = sqlite3.Row
            c = conn.cursor()
            c.execute("""
                SELECT * FROM evidence WHERE admin_judgment='pending'
                ORDER BY created_at DESC LIMIT ?
            """, (limit,))
            rows = [dict(r) for r in c.fetchall()]
            conn.close()
            return rows

    def get_recent_judged(self, limit=20):
        """获取最近已审核的记录"""
        with self._lock:
            conn = sqlite3.connect(self._db_path)
            conn.row_factory = sqlite3.Row
            c = conn.cursor()
            c.execute("""
                SELECT * FROM evidence WHERE admin_judgment!='pending'
                ORDER BY judged_at DESC LIMIT ?
            """, (limit,))
            rows = [dict(r) for r in c.fetchall()]
            conn.close()
            return rows

    def get_model_stats(self):
        """获取模型统计信息"""
        with self._lock:
            conn = sqlite3.connect(self._db_path)
            conn.row_factory = sqlite3.Row
            c = conn.cursor()
            c.execute("SELECT * FROM model_scores")
            rows = [dict(r) for r in c.fetchall()]
            conn.close()
            return rows

    def get_total_stats(self):
        """获取总体统计"""
        with self._lock:
            conn = sqlite3.connect(self._db_path)
            c = conn.cursor()
            c.execute("SELECT COUNT(*) FROM evidence WHERE admin_judgment='correct'")
            correct = c.fetchone()[0]
            c.execute("SELECT COUNT(*) FROM evidence WHERE admin_judgment='incorrect'")
            incorrect = c.fetchone()[0]
            c.execute("SELECT COUNT(*) FROM evidence WHERE admin_judgment='pending'")
            pending = c.fetchone()[0]
            conn.close()
            total = correct + incorrect
            acc = correct / max(total, 1)
            return {"correct": correct, "incorrect": incorrect, "pending": pending,
                    "accuracy": acc, "total": total}

    def get_feedback_stats(self):
        """获取反馈统计数据（用于学习曲线）"""
        with self._lock:
            conn = sqlite3.connect(self._db_path)
            c = conn.cursor()
            c.execute("""
                SELECT label, was_correct, COUNT(*) as cnt
                FROM feedback GROUP BY label, was_correct
            """)
            rows = c.fetchall()
            conn.close()
            stats = {}
            for label, was_correct, cnt in rows:
                if label not in stats:
                    stats[label] = {"correct": 0, "incorrect": 0}
                if was_correct:
                    stats[label]["correct"] = cnt
                else:
                    stats[label]["incorrect"] = cnt
            return stats

    def find_recent_similar_person(self, camera_id, person_features, time_window=30):
        """查找近期相似人物（用于去重）"""
        with self._lock:
            conn = sqlite3.connect(self._db_path)
            c = conn.cursor()
            now = datetime.now()
            c.execute("""
                SELECT id, person_features, confidence, created_at FROM evidence
                WHERE camera_id=? AND admin_judgment='pending'
                ORDER BY created_at DESC LIMIT 20
            """, (camera_id,))
            rows = c.fetchall()
            conn.close()

            for eid, stored_features, conf, created_at in rows:
                if stored_features:
                    stored = json.loads(stored_features)
                    similarity = self._feature_similarity(person_features, stored)
                    if similarity > 0.75:
                        return eid, conf
            return None, 0

    @staticmethod
    def _feature_similarity(f1, f2):
        """计算两个特征向量的余弦相似度"""
        if not f1 or not f2:
            return 0.0
        keys = set(f1.keys()) & set(f2.keys())
        if not keys:
            return 0.0
        vec1, vec2 = [], []
        for k in keys:
            v1, v2 = f1[k], f2[k]
            if isinstance(v1, list):
                vec1.extend(v1)
                vec2.extend(v2)
            else:
                vec1.append(v1)
                vec2.append(v2)
        a = np.array(vec1, dtype=np.float32)
        b = np.array(vec2, dtype=np.float32)
        na = np.linalg.norm(a)
        nb = np.linalg.norm(b)
        if na == 0 or nb == 0:
            return 0.0
        return float(np.dot(a, b) / (na * nb))

    def replace_evidence_image(self, evidence_id, new_image_path, new_confidence):
        """替换已有证据的截图（用更高置信度的替换）"""
        with self._lock:
            conn = sqlite3.connect(self._db_path)
            c = conn.cursor()
            image_blob = None
            if os.path.exists(new_image_path):
                try:
                    with open(new_image_path, "rb") as f:
                        image_blob = f.read()
                except Exception:
                    pass
            c.execute("""
                UPDATE evidence SET image_path=?, image_blob=?, confidence=?
                WHERE id=?
            """, (new_image_path, image_blob, new_confidence, evidence_id))
            conn.commit()
            conn.close()

    def get_stats_for_display(self):
        """获取用于前端显示的统计"""
        stats = self.get_total_stats()
        scores = self.get_model_stats()
        return {
            "total_correct": stats["correct"],
            "total_incorrect": stats["incorrect"],
            "pending": stats["pending"],
            "accuracy": f"{stats['accuracy']:.1%}",
            "model_scores": scores,
        }