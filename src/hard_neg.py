"""自动收集硬负样本（误报帧）—— 让主动学习不再依赖人工一条条点。

为什么需要它
------------
统计今天的事件：fire 2236 条里 **1469 条被 LLM 判成"非风险"**，也就是
66% 是误报（主因是皮肤高光 / 白衣服被当火）。但 `hard_negatives/` 里只有
7 张火类负样本 —— 因为负样本**只在人工于点看板上点"误报"时才写入**
（src/notify.py 的反馈接口），不点就一张都不采集。

结果就是：误报天天有，训练集天天不长，同一个错误永远修不掉。

这个模块做的事
--------------
监听 `level == "cleared"`（已被判为"非风险"）的事件，自动把 ROI 裁剪存进
`hard_negatives/images/<kind>/` 并配一个**空标注**（YOLO 约定：空 txt = 图里
没有任何目标）。LLM 已经替我们做过"这里不是火"的判断，这个标签是可信的。

三个必须有的保护（否则会把自己坑了）
------------------------------------
1. **去重**：摄像头对着静止场景时，同一个误报会连着触发几百次。
   用 16x16 均值哈希比对，Hamming 距离 <= 6 视为同一帧，直接跳过。
   哈希持久化到 `auto_hashes.txt`，重启后不会重复采集。
2. **限量**：默认总量 800 张、每小时 120 张。防止跑一晚上塞满磁盘。
3. **只收"确认是误报"的**：`level` 必须是 `cleared`（LLM 明确否定），
   `notice`/`warning` 这种"还没定论"的一律不收 —— 把真火当负样本会把
   模型教坏，这比漏采严重得多。

写入格式与人工样本完全一致
--------------------------
    hard_negatives/images/fire/hn_auto_<ts>_<cam>.jpg
    hard_negatives/labels/fire/hn_auto_<ts>_<cam>.txt   (空文件)

`scripts/merge_hard_negatives.py` 会连同人工样本一起并进训练集。
"""
from __future__ import annotations

import json
import queue
import threading
import time
from pathlib import Path

import cv2
import numpy as np

# 只采这三类。light/steam 之类本来就不该进火烟数据集。
KINDS = ("fire", "smoke", "cigarette")

# 均值哈希的边长。16x16 在"区分不同场景"和"容忍轻微压缩噪声"之间比较平衡。
_HASH_SIDE = 16
# 汉明距离 <= 这个值就认为是同一帧（越小越严格）
DUP_HAMMING = 6


def _ahash(img: np.ndarray) -> int:
    """16x16 灰度均值哈希，返回一个 256 位整数。"""
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    small = cv2.resize(g, (_HASH_SIDE, _HASH_SIDE), interpolation=cv2.INTER_AREA)
    bits = (small > small.mean()).flatten()
    v = 0
    for b in bits:
        v = (v << 1) | int(bool(b))
    return v


def _hamming(a: int, b: int) -> int:
    return int(a ^ b).bit_count()


class HardNegCollector:
    """把"已判非风险"的 ROI 自动存成硬负样本。线程安全、不阻塞检测主循环。"""

    def __init__(self, root: str | Path, *,
                 max_total: int = 800,
                 max_per_hour: int = 120,
                 min_side: int = 24,
                 enabled: bool = True) -> None:
        self.root = Path(root)
        self.img_root = self.root / "images"
        self.lbl_root = self.root / "labels"
        self.max_total = int(max_total)
        self.max_per_hour = int(max_per_hour)
        self.min_side = int(min_side)
        self.enabled = bool(enabled)

        self._q: queue.Queue = queue.Queue(maxsize=256)
        self._hashes: list[int] = []
        self._recent_ts: list[float] = []
        self._total = 0
        self.stats = {"saved": 0, "dup": 0, "skipped": 0, "capped": 0, "errors": 0}

        if self.enabled:
            self.root.mkdir(parents=True, exist_ok=True)
            self._load_hashes()
            threading.Thread(target=self._loop, name="hard-neg", daemon=True).start()

    # ------------------------------------------------------------ 持久化
    @property
    def _hash_file(self) -> Path:
        return self.root / "auto_hashes.txt"

    def _load_hashes(self) -> None:
        """载入历史哈希 + 统计已有自动样本数，避免重启后重复采。"""
        try:
            if self._hash_file.is_file():
                for line in self._hash_file.read_text(encoding="utf-8").splitlines():
                    line = line.strip()
                    if line.isdigit():
                        self._hashes.append(int(line))
            for k in KINDS:
                d = self.img_root / k
                if d.is_dir():
                    self._total += sum(1 for _ in d.glob("hn_auto_*.jpg"))
        except OSError:
            pass

    def _remember(self, h: int) -> None:
        self._hashes.append(h)
        if len(self._hashes) > 4000:            # 只留最近的，文件别无限长
            self._hashes = self._hashes[-4000:]
        try:
            with open(self._hash_file, "a", encoding="utf-8") as f:
                f.write(f"{h}\n")
        except OSError:
            self.stats["errors"] += 1

    # ------------------------------------------------------------ 外部接口
    def submit(self, roi, kind: str, cam: str, score: float = 0.0,
               reason: str = "") -> None:
        """投递一个候选误报。**只入队，不做 IO** —— 调用方在检测主循环里。"""
        if not self.enabled or roi is None:
            return
        k = str(kind or "").lower()
        if k not in KINDS:
            return
        try:
            self._q.put_nowait((roi, k, str(cam), float(score), str(reason)))
        except queue.Full:
            self.stats["skipped"] += 1

    # ------------------------------------------------------------ 工作线程
    def _capped(self, now: float) -> bool:
        if self._total >= self.max_total:
            return True
        self._recent_ts = [t for t in self._recent_ts if now - t < 3600]
        return len(self._recent_ts) >= self.max_per_hour

    def _loop(self) -> None:
        while True:
            item = self._q.get()
            try:
                if item is None:
                    return
                self._handle(*item)
            except Exception as exc:              # noqa: BLE001
                self.stats["errors"] += 1
                print(f"[hard-neg] 采集异常 {type(exc).__name__}: {exc}", flush=True)
            finally:
                self._q.task_done()

    def _handle(self, roi, kind: str, cam: str, score: float, reason: str) -> None:
        h, w = roi.shape[:2]
        if min(h, w) < self.min_side:
            self.stats["skipped"] += 1
            return
        now = time.time()
        if self._capped(now):
            self.stats["capped"] += 1
            return
        hh = _ahash(roi)
        if any(_hamming(hh, old) <= DUP_HAMMING for old in self._hashes[-1500:]):
            self.stats["dup"] += 1
            return

        img_dir = self.img_root / kind
        lbl_dir = self.lbl_root / kind
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)
        stem = f"hn_auto_{int(now * 1000)}_{cam}"
        try:
            ok, buf = cv2.imencode(".jpg", roi, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
            if not ok:
                self.stats["errors"] += 1
                return
            (img_dir / f"{stem}.jpg").write_bytes(buf.tobytes())
            # 空标注 = 这张图没有目标（YOLO 约定）
            (lbl_dir / f"{stem}.txt").write_text("", encoding="utf-8")
        except OSError:
            self.stats["errors"] += 1
            return

        self._remember(hh)
        self._recent_ts.append(now)
        self._total += 1
        self.stats["saved"] += 1
        try:
            with open(self.root / "auto_log.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps({
                    "ts": round(now, 3), "kind": kind, "cam": cam,
                    "score": round(score, 3), "reason": reason[:200],
                    "file": str(img_dir / f"{stem}.jpg"),
                }, ensure_ascii=False) + "\n")
        except OSError:
            pass
