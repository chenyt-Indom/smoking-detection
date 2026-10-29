"""时序判别的自测：用合成轨迹验证「烟 / 蒸汽」能不能被分开。

没有真实数据之前，这一步很重要 —— 它证明**判别逻辑本身**是对的，
把后面调参的变量缩小到"特征提取准不准"，而不是"逻辑通不通"。

跑法：
    python scripts/test_temporal.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.temporal import TrackletSample, extract_features, judge  # noqa: E402


def make_seq(kind: str, fps: float = 10.0, dur: float = 30.0,
             x0: float = 320, y0: float = 400, seed: int = 0):
    """合成一条轨迹。

    smoke：缓慢膨胀、亮度逐渐下降、清晰度稳定、存续久
    steam：快速膨胀、亮度上升、清晰度快速下降、存续短
    """
    rng = np.random.default_rng(seed)
    n = int(fps * dur)
    ts = np.arange(n) / fps

    if kind == "smoke":
        base = 900.0
        area = base * np.exp(0.18 * ts)                 # 对数增长率 ≈0.18
        gray = 210 - 3.2 * ts                           # 逐渐变暗
        edge = 0.30 + 0.002 * ts                        # 纹理基本稳定
        rise = 2.2
    else:  # steam
        base = 600.0
        area = base * np.exp(0.85 * ts)                 # 对数增长率 ≈0.85
        gray = 200 + 2.0 * ts                           # 越来越淡/亮
        edge = 0.34 - 0.02 * ts                         # 快速变糊
        rise = 3.5

    out = []
    for i, t in enumerate(ts):
        out.append(TrackletSample(
            t=float(t),
            area=float(area[i] * (1 + rng.normal(0, 0.03))),
            cx=float(x0 + 0.4 * t + rng.normal(0, 1.2)),
            cy=float(y0 - rise * t + rng.normal(0, 1.2)),
            gray_mean=float(np.clip(gray[i] + rng.normal(0, 2), 0, 255)),
            sat_mean=float(np.clip(18 + rng.normal(0, 3), 0, 255)),
            edge_density=float(np.clip(edge[i] + rng.normal(0, 0.01), 0, 1)),
            conf=0.8,
        ))
    return out


def main() -> int:
    cases = [
        ("纯白烟（无火）", make_seq("smoke", seed=1), False, "smoke"),
        ("纯白烟（旁边有火）", make_seq("smoke", seed=2), True, "smoke"),
        ("蒸汽", make_seq("steam", dur=12, seed=3), False, "steam"),
        ("蒸汽（旁边有火，干扰）", make_seq("steam", dur=12, seed=4), True, None),
        ("样本太少（2 帧）", make_seq("smoke")[:2], False, "uncertain"),
    ]

    ok = 0
    print("=" * 66)
    for name, seq, fire, expect in cases:
        feat = extract_features(seq)
        v = judge(feat, has_fire_nearby=fire)
        flag = "PASS" if (expect is None or v.label == expect) else "FAIL"
        if flag == "PASS":
            ok += 1
        print(f"[{flag}] {name}")
        if feat:
            print(f"       特征：时长 {feat.duration:.1f}s · 面积增长 {feat.area_growth:+.2f}/s · "
                  f"亮度 {feat.gray_trend:+.2f}/s · 清晰度 {feat.edge_trend:+.3f}/s")
        print(f"       判定：{v.label}  score={v.score}")
        for r in v.reasons:
            print(f"         · {r}")

    print("=" * 66)
    print(f"通过 {ok}/{len(cases)}")
    return 0 if ok == len(cases) else 1


if __name__ == "__main__":
    raise SystemExit(main())
