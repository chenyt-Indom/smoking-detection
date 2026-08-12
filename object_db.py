# -*- coding: utf-8 -*-
"""
object_db.py — 物品事件数据库 (08-12 新架构)
每个头/手ID只保留"最清晰的一帧"截图 + 后台香烟识别结果
"""
import sqlite3, os, time

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "alerts", "object_events.db")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS object_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    obj_type TEXT NOT NULL,        -- 'head' / 'hand'
    obj_id INTEGER NOT NULL,       -- 轨ID
    ts REAL NOT NULL,              -- 截图时间戳
    cap_path TEXT NOT NULL,        -- 原始截图路径(最清晰帧)
    clarity REAL DEFAULT 0,        -- 清晰度评分(Laplacian方差)
    super_path TEXT DEFAULT NULL,  -- 超分辨率增强后路径
    smoke_result INTEGER DEFAULT -1,  -- -1=未处理 0=无烟 1=有烟
    smoke_conf REAL DEFAULT 0,
    processed INTEGER DEFAULT 0,   -- 后台是否已识别
    UNIQUE(obj_type, obj_id)
)
"""


class ObjectDB:
    def __init__(self, db_path=DB_PATH):
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.db_path = db_path
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute(_SCHEMA)
        self._conn.commit()
        self._lock = threading_lock()

    # ---- 写 ----
    def upsert_cap(self, obj_type, obj_id, cap_path, clarity):
        """记录/更新某ID的最清晰截图(同ID覆盖, 重置识别状态)"""
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO object_events "
                "(obj_type, obj_id, ts, cap_path, clarity, smoke_result, smoke_conf, processed) "
                "VALUES (?,?,?,?,?, -1, 0, 0)",
                (obj_type, obj_id, time.time(), cap_path, float(clarity)))
            self._conn.commit()

    def mark_processed(self, event_id, smoke_result, smoke_conf, super_path):
        with self._lock:
            self._conn.execute(
                "UPDATE object_events SET smoke_result=?, smoke_conf=?, super_path=?, processed=1 WHERE id=?",
                (int(smoke_result), float(smoke_conf), super_path, int(event_id)))
            self._conn.commit()

    # ---- 读 ----
    def get_unprocessed(self, limit=3):
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM object_events WHERE processed=0 ORDER BY ts ASC LIMIT ?",
                (int(limit),)).fetchall()
            return [dict(r) for r in rows]

    def get_all_events(self, limit=100):
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM object_events ORDER BY ts DESC LIMIT ?",
                (int(limit),)).fetchall()
            return [dict(r) for r in rows]

    def get_stats(self):
        with self._lock:
            row = self._conn.execute(
                "SELECT COUNT(*) AS total, SUM(processed) AS done, "
                "SUM(CASE WHEN smoke_result=1 THEN 1 ELSE 0 END) AS has_smoke "
                "FROM object_events").fetchone()
            return {"total": row["total"] or 0, "processed": row["done"] or 0,
                    "has_smoke": row["has_smoke"] or 0}

    def close(self):
        with self._lock:
            self._conn.close()


# 线程锁工厂(sqlite连接跨线程需外部锁)
import threading as _th
def threading_lock():
    return _th.Lock()
