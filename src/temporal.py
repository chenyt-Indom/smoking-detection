"""时序判别：把「逐帧检测框」升级成「视频级判定」。

为什么必须有这一层
------------------
用户的核心要求是 **白烟/黑烟要检测，但要和蒸汽、水雾分开**。
这件事在**单帧**上做不到 —— 白烟和蒸汽同色、同形状、都是上升羽流，
单帧分类器只能靠背景瞎猜，误报率必然高。

但它们在**时间维度**上差别很大：

| 特征 | 烟（含白烟） | 蒸汽 / 水雾 |
|---|---|---|
| 面积增长 | 慢而持续（燃烧持续供烟） | 快，几秒内膨胀后迅速变淡 |
| 存续时间 | 长（数十秒~分钟） | 短（几秒~十几秒，随水汽消散） |
| 颜色走向 | 白 → 灰 → 深灰/黑（碳颗粒累积） | 始终偏白，单调变淡 |
| 边界清晰度 | 相对稳定（烟有纹理） | 越来越模糊（水汽扩散） |
| 垂直速度 | 随浮力上升，较稳定 | 上升更快、更"抖" |
| 与火源关系 | 常伴随火焰/高温区 | 与火源无关，多在壶口/管道口 |

所以架构是：**检测器负责"看见"，这一层负责"确认是什么"**。
这样既保住 YOLO26 的实时性，又把不可解的单帧问题变成可算的时序特征。

设计取舍
--------
· 用**手工特征 + 轻量分类器**，不用 3D CNN：样本量小、要可解释（能说清"为什么判成蒸汽"）
· 判别器先给**规则版**（零样本可用，冷启动），积累数据后再训练 GBDT 覆盖它
· 所有阈值集中在 `DEFAULTS` 里，方便按现场调，不要散落在代码各处
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import numpy as np


@dataclass
class TrackletSample:
    """轨迹上的一个时间点（一般来自检测框）。

    只存判别需要的量，不存图像 —— 视频长跑时内存才不会爆。
    """

    t: float                     # 时间戳（秒）
    area: float                  # 框面积（像素²）
    cx: float                    # 中心 x
    cy: float                    # 中心 y
    gray_mean: float             # 框内平均亮度 0~255
    sat_mean: float              # 框内平均饱和度 0~255（烟/蒸汽都低，用于排除彩色干扰）
    edge_density: float          # 边缘密度 0~1（清晰度，蒸汽会快速下降）
    conf: float = 1.0            # 检测置信度


@dataclass
class TemporalFeatures:
    """从一条轨迹算出来的判别特征。"""

    duration: float              # 存续时长（秒）
    area_growth: float           # 面积增长率（1/秒，对数域斜率）
    area_cv: float               # 面积波动系数（抖动程度）
    rise_speed: float            # 垂直上升速度（像素/秒，向上为正）
    drift_speed: float           # 水平漂移速度（烟常随风漂）
    gray_trend: float            # 亮度变化趋势（负=变暗，烟常见）
    sat_trend: float             # 饱和度趋势
    edge_trend: float            # 清晰度趋势（负=越来越糊，蒸汽常见）
    n: int                       # 样本数
    span_ratio: float            # 有效时长占理论时长的比例（轨迹连续性）


def _slope(ts: np.ndarray, ys: np.ndarray) -> float:
    """最小二乘斜率；样本不足返回 0。"""
    if len(ts) < 2:
        return 0.0
    t = ts - ts.mean()
    denom = float((t * t).sum())
    if denom <= 1e-9:
        return 0.0
    return float((t * (ys - ys.mean())).sum() / denom)


def extract_features(samples: list[TrackletSample]) -> TemporalFeatures | None:
    """把轨迹点压成特征。样本太少（< 4）返回 None —— 那时任何判别都是瞎猜。"""
    if len(samples) < 4:
        return None

    ts = np.array([s.t for s in samples], dtype=float)
    order = np.argsort(ts)
    samples = [samples[i] for i in order]
    ts = ts[order]

    area = np.array([s.area for s in samples], dtype=float)
    cy = np.array([s.cy for s in samples], dtype=float)
    cx = np.array([s.cx for s in samples], dtype=float)
    gray = np.array([s.gray_mean for s in samples], dtype=float)
    sat = np.array([s.sat_mean for s in samples], dtype=float)
    edge = np.array([s.edge_density for s in samples], dtype=float)

    duration = float(ts[-1] - ts[0])
    if duration <= 0:
        return None

    # 面积在**对数域**取斜率：几何增长时线性回归会被大面积的晚期样本带偏
    log_area = np.log(np.maximum(area, 1.0))
    area_growth = _slope(ts, log_area)

    area_cv = float(area.std() / max(area.mean(), 1.0))

    # 图像坐标 y 向下，所以"向上"是 cy 减少
    rise_speed = -_slope(ts, cy)
    drift_speed = abs(_slope(ts, cx))

    # 采样间隔中位数 → 判断轨迹连续性（丢帧多则特征不可信）
    gaps = np.diff(ts)
    span_ratio = 1.0
    if len(gaps):
        med = float(np.median(gaps))
        if med > 0:
            span_ratio = min(1.0, float(len(ts) * med / max(duration, 1e-6)))

    return TemporalFeatures(
        duration=duration,
        area_growth=area_growth,
        area_cv=area_cv,
        rise_speed=rise_speed,
        drift_speed=drift_speed,
        gray_trend=_slope(ts, gray),
        sat_trend=_slope(ts, sat),
        edge_trend=_slope(ts, edge),
        n=len(samples),
        span_ratio=span_ratio,
    )


# ---------------------------------------------------------------- 判别
# 阈值集中在这里。现场不同（室内/室外/光照）需要按数据微调，
# 建议调完写进 config/temporal.yaml，别改代码。
DEFAULTS = {
    "min_duration": 1.5,        # 少于 1.5 秒不下结论（信息不够）
    # 走向类特征（亮度/清晰度）的最小观测窗口。
    # 为什么单列：轨迹刚建立时框还在长，框内均值被"框变大把背景吃进来"主导，
    # 实测合成序列里 1.6s 就能算出 -9.5/s 的亮度斜率 —— 那是几何变化，不是烟在变黑。
    # 不加这条，任何刚出现的烟都会在 2 秒内被判成 smoke，等于绕过时序判别。
    "trend_min_duration": 5.0,
    "area_growth_steam": 0.55,  # 面积对数增长率 > 这个值，倾向蒸汽/水雾
    "area_growth_smoke": 0.25,  # 低于这个值更像烟（缓慢持续）
    "edge_drop_steam": -0.35,   # 清晰度快速下降 → 蒸汽
    "gray_drop_smoke": -0.8,    # 明显变暗 → 烟（白烟转灰黑）
    "short_life": 8.0,          # 存续 < 8 秒更像蒸汽（烟一般更久）
    "long_life": 25.0,          # 存续 > 25 秒几乎是烟/火
    "conf_smoke": 0.55,         # 判定为烟的置信度门限
    "conf_steam": 0.55,         # 判定为蒸汽的门限
}


@dataclass
class Verdict:
    label: str                   # "smoke" / "steam" / "uncertain"
    score: float                 # 0~1，对该判定的把握
    reasons: list[str] = field(default_factory=list)   # 人话解释，便于现场调参
    features: TemporalFeatures | None = None


def judge(feat: TemporalFeatures | None, *, thr: dict | None = None,
          has_fire_nearby: bool = False) -> Verdict:
    """规则判别器（冷启动版）。

    has_fire_nearby：同一时刻附近是否存在 fire 框 —— 这是**最强的先验**，
    旁边有火就基本是烟（水雾很少和明火同现）。所以它直接大幅加权。
    """
    if feat is None:
        return Verdict("uncertain", 0.0, ["轨迹样本不足（<4 帧）"])

    t = {**DEFAULTS, **(thr or {})}
    smoke = steam = 0.0
    why: list[str] = []

    if feat.duration < t["min_duration"]:
        return Verdict("uncertain", 0.0,
                       [f"存续仅 {feat.duration:.1f}s，不足 {t['min_duration']}s，先不下结论"],
                       feat)

    # ① 面积增长：蒸汽膨胀快，烟缓慢
    if feat.area_growth >= t["area_growth_steam"]:
        steam += 0.35
        why.append(f"面积增长快（{feat.area_growth:.2f}/s），像水汽膨胀")
    elif feat.area_growth <= t["area_growth_smoke"]:
        smoke += 0.30
        why.append(f"面积增长缓慢（{feat.area_growth:.2f}/s），符合持续供烟")

    # ② 清晰度变化：蒸汽越来越糊，烟纹理相对稳定
    # ③ 亮度走向：烟会由白转灰/黑，蒸汽只会变淡（亮度上升或平稳）
    # 这两条都是"走向"，观测窗口不够时斜率不可信（见 trend_min_duration 的注释）
    if feat.duration < t["trend_min_duration"]:
        why.append(f"观测仅 {feat.duration:.1f}s，暂不看亮度/清晰度走向"
                   f"（需 {t['trend_min_duration']:.0f}s）")
    else:
        if feat.edge_trend <= t["edge_drop_steam"]:
            steam += 0.25
            why.append(f"边界快速变糊（{feat.edge_trend:.2f}/s），是水汽扩散的典型表现")
        elif feat.edge_trend > 0:
            smoke += 0.10

        if feat.gray_trend <= t["gray_drop_smoke"]:
            smoke += 0.35
            why.append(f"整体变暗（{feat.gray_trend:.2f}/s），白烟转灰黑")
        elif feat.gray_trend >= 0.5:
            steam += 0.20
            why.append(f"整体变亮（{feat.gray_trend:.2f}/s），更像水汽消散")

    # ④ 存续时长：烟长、蒸汽短
    if feat.duration < t["short_life"]:
        steam += 0.15
        why.append(f"存续 {feat.duration:.0f}s 偏短")
    elif feat.duration > t["long_life"]:
        smoke += 0.20
        why.append(f"存续 {feat.duration:.0f}s 偏长")

    # ⑤ 火源先验（最强）
    if has_fire_nearby:
        smoke += 0.45
        why.append("同场景存在明火，烟的可能性大幅上升")
    else:
        steam += 0.10

    # ⑥ 轨迹连续性：丢帧多则降权，宁可"不确定"也不要误报
    if feat.span_ratio < 0.5:
        smoke *= 0.6
        steam *= 0.6
        why.append(f"轨迹不连续（覆盖率 {feat.span_ratio:.0%}），结论打折")

    # 归一化到 0~1：两方分数差距越大越有把握
    total = smoke + steam
    if total <= 1e-6:
        return Verdict("uncertain", 0.0, why or ["特征不足以区分"], feat)

    if smoke >= steam:
        score = smoke / total * min(1.0, smoke / 0.75)
        label = "smoke" if score >= t["conf_smoke"] else "uncertain"
    else:
        score = steam / total * min(1.0, steam / 0.75)
        label = "steam" if score >= t["conf_steam"] else "uncertain"

    if label == "uncertain":
        why.append(f"双方接近（烟 {smoke:.2f} / 汽 {steam:.2f}），不硬判")

    return Verdict(label, round(score, 3), why, feat)


def judge_batch(seqs: Iterable[list[TrackletSample]], **kw) -> list[Verdict]:
    return [judge(extract_features(s), **kw) for s in seqs]
