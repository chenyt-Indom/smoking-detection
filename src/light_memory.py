"""已确认"不是火"的区域记忆：同一盏路灯不要每晚重新花钱问 LLM。

要解决的事
----------
现场最贵的不是推理，是**重复推理**。一盏路灯整夜亮着，YOLO 会因为遮挡、
失配、曝光变化把它切成一条又一条轨迹，而原来送审的键是 `机位:轨迹ID` ——
每换一条轨迹就重新问一次 LLM。一夜下来几十上百次，答案每次都是"这是灯"。

三层防线（从便宜到贵）
--------------------
1. `llm_mute_sec` + 区域匹配（在 pipeline 里）：同一块地方刚判过，就别再问。
   进程内有效，重启即忘 —— 挡住同一班次内的抖动。
2. **本模块**：判过"非火灾风险"的区域落盘记住。下次这条轨迹出现，
   直接按"已知非风险"处理，一次 API 都不调。
3. 外形突变就作废（`still_same`）：面积翻了、或者开始抖了，说明这块地方
   已经不是原来那个东西了（灯烧起来了 / 反光换了个角度），立刻恢复完整流程。
   这是这套机制的安全阀 —— 记忆只能省"确认过的旧情况"的钱，
   绝不能替一个**正在变化**的目标下结论。

为什么记归一化坐标
------------------
换分辨率、换码流都要能对上。存像素坐标的话，同一个机位改一次分辨率，
整份记忆全废。
"""
from __future__ import annotations

import json
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class LightSpot:
    """一处"已知不是火"的地方。坐标是归一化的 0~1。"""
    cam: str
    box: tuple[float, float, float, float]
    label: str                  # LLM 给的标签：light / steam / other
    hits: int = 1               # 被确认过几次
    first: float = 0.0          # 首次确认（wall clock）
    last: float = 0.0           # 最近一次确认/命中
    area: float = 0.0           # 归一化面积，用来判断"还是不是原来那个东西"

    @property
    def center(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.box
        return (x1 + x2) / 2, (y1 + y2) / 2


def _iou(a, b) -> float:
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, x2 - x1), max(0.0, y2 - y1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


class LightMemory:
    """落盘的区域记忆。线程安全（判定在采集线程，落盘可能被别处触发）。"""

    def __init__(self, path: str | Path | None = "data/light_memory.json", *,
                 iou_thr: float = 0.45, area_tol: float = 0.6,
                 ttl: float = 7 * 86400.0, max_spots: int = 500):
        self.path = Path(path) if path else None
        self.iou_thr = iou_thr
        self.area_tol = area_tol
        self.ttl = ttl
        self.max_spots = max_spots
        self._spots: list[LightSpot] = []
        self._lock = threading.Lock()
        self._dirty = False
        self.stats = {"remembered": 0, "hit": 0, "reopened": 0, "forgotten": 0}
        self.load()

    # ---------------------------------------------------------------- 落盘
    def load(self) -> int:
        if self.path is None or not self.path.is_file():
            return 0
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return 0
        now = time.time()
        with self._lock:
            self._spots = [LightSpot(cam=r["cam"], box=tuple(r["box"]),
                                     label=r.get("label", "light"),
                                     hits=int(r.get("hits", 1)),
                                     first=float(r.get("first", now)),
                                     last=float(r.get("last", now)),
                                     area=float(r.get("area", 0.0)))
                           for r in raw.get("spots", [])
                           if now - float(r.get("last", now)) <= self.ttl]
        return len(self._spots)

    def save(self) -> None:
        if self.path is None:
            return
        with self._lock:
            if not self._dirty:
                return
            payload = {"version": 1, "saved": time.strftime("%Y-%m-%d %H:%M:%S"),
                       "spots": [asdict(s) for s in self._spots]}
            self._dirty = False
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=1),
                           encoding="utf-8")
            tmp.replace(self.path)          # 原子替换，别留半个文件
        except OSError:
            with self._lock:
                self._dirty = True

    # ---------------------------------------------------------------- 查询
    def match(self, cam: str, box_n, area_n: float = 0.0) -> LightSpot | None:
        """这块地方记过吗？记过就返回那一条。"""
        now = time.time()
        with self._lock:
            for s in self._spots:
                if s.cam != cam:
                    continue
                if now - s.last > self.ttl:
                    continue
                if _iou(s.box, box_n) >= self.iou_thr:
                    return s
        return None

    def still_same(self, spot: LightSpot, box_n, area_n: float) -> bool:
        """还是原来那个东西吗？

        面积变化超过 `area_tol` 就认为不是了。这条判据不好省：路灯被点着、
        垃圾桶里的火烧大、或者相机被人转了一下，都会让面积明显变。
        误判成"还是那盏灯"的代价是漏一次真火，所以宁可从宽（宽=更容易作废）。
        """
        if spot.area <= 0 or area_n <= 0:
            return True                     # 缺数据就不作数（别拿猜的当判据）
        r = area_n / spot.area
        return (1.0 / (1.0 + self.area_tol)) <= r <= (1.0 + self.area_tol)

    # ---------------------------------------------------------------- 写入
    def remember(self, cam: str, box_n, area_n: float, label: str) -> LightSpot:
        """确认"这里不是火"，记一笔。已有记录就累加次数、刷新时间。"""
        now = time.time()
        with self._lock:
            for s in self._spots:
                if s.cam == cam and _iou(s.box, box_n) >= self.iou_thr:
                    s.hits += 1
                    s.last = now
                    s.label = label or s.label
                    s.area = area_n or s.area
                    self._dirty = True
                    self.stats["remembered"] += 1
                    return s
            spot = LightSpot(cam=cam, box=tuple(float(v) for v in box_n),
                             label=label or "light", hits=1,
                             first=now, last=now, area=area_n)
            self._spots.append(spot)
            if len(self._spots) > self.max_spots:
                # 留最近用过的，别让文件无限长
                self._spots.sort(key=lambda s: s.last, reverse=True)
                self._spots = self._spots[: self.max_spots]
            self._dirty = True
            self.stats["remembered"] += 1
            return spot

    def touch(self, spot: LightSpot) -> None:
        with self._lock:
            spot.last = time.time()
            spot.hits += 1
            self._dirty = True
            self.stats["hit"] += 1

    def forget(self, cam: str, box_n) -> int:
        """作废这一块地方的记忆（外形变了 / 人工否定）。"""
        with self._lock:
            keep = [s for s in self._spots
                    if not (s.cam == cam and _iou(s.box, box_n) >= self.iou_thr)]
            n = len(self._spots) - len(keep)
            if n:
                self._spots = keep
                self._dirty = True
                self.stats["reopened"] += n
            return n

    def clear(self) -> int:
        with self._lock:
            n = len(self._spots)
            self._spots = []
            self._dirty = True
            return n

    def __len__(self) -> int:
        return len(self._spots)

    def summary(self) -> list[dict]:
        """给看板/命令行看的摘要。"""
        with self._lock:
            return [{"cam": s.cam, "label": s.label, "hits": s.hits,
                     "box": [round(v, 3) for v in s.box],
                     "age_h": round((time.time() - s.last) / 3600, 1)}
                    for s in sorted(self._spots, key=lambda s: -s.last)[:20]]
