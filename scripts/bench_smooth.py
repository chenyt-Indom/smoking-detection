"""框平滑参数扫描：在合成轨迹上量化「准不准 / 抖不抖 / 滞后多少」。

为什么需要这个脚本
------------------
用户反馈"不够精准，追踪也不平滑"，但如果只凭肉眼在摄像头前调参，永远
说不清到底是 Kalman 太钝（滞后）、还是 EMA 太重（跟不上），也无法比较
两套参数哪个更好。这里把真实链路的两段平滑**原样搬进受控实验**：

    带噪检测框  ->  _Kalman(predict/correct)  ->  LiveView._smooth  ->  输出框

Kalman 和 EMA 都直接 import 项目里的实现（只把常量改成可注入），
所以扫出来的最优组合能直接落回源码，不会出现"测的和跑的不是一回事"。

度量三个互相冲突的指标
----------------------
  精准  center_err  : 输出框中心与真值中心的平均距离（px）—— 越小越"准"
  平滑  jerk        : 中心位置二阶差分绝对值均值（px/帧²）—— 越小越"稳"
  滞后  lag         : 互相关求出的帧延迟 —— 越小越"跟手"

三者天然打架：想压 jerk 就得加重历史，加重就滞后，滞后就"不准"。
所以不取单一最优，而是找**帕累托前沿**（没有别的组合在三个指标上同时更好）。

用法
----
    python scripts/bench_smooth.py                # 扫描 + 推荐
    python scripts/bench_smooth.py --top 15       # 多看几名
"""
from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import liveview as lv          # noqa: E402
from src import tracker as tk           # noqa: E402

# ── 目标画像（尺寸/噪声量级都取自线上实测）────────────────────────────
# 12fps 是当前检测帧率（引擎 + 双路）；平滑的"每帧步长"和它直接相关。
FPS = 12.0
N_FRAME = 240

# 低分框的定位误差放大系数（上限）。
# 这是个**假设**，不是实测值 —— 所以不当成唯一真理：脚本会用 0 和 0.6
# 两套假设各跑一遍，只有**两套下都不差**的参数才算过关。只信一套会过拟合。
CONF_NOISE_K = 0.6

PROFILES = {
    # 火苗：几乎不动，但 YOLO 给的框会因闪烁忽大忽小、中心 ±4px 抖
    "fire": dict(
        w0=34, h0=42, drift=(0.12, 0.10),      # 每帧真位移（px）—— 蔓延很慢
        pos_sigma=4.0, size_sigma=6.0,
        size_osc=0.22,                          # 闪烁导致框尺寸摆动 ±22%
        dropout=0.15,                           # 漏检率
        conf=(0.35, 0.85),                      # 火一般分数不低
    ),
    # 香烟：手持，慢速漂移 + 偶发一下抖动；框很小，归一化抖动最大
    "cigarette": dict(
        w0=16, h0=44, drift=(1.8, 1.2),
        pos_sigma=3.0, size_sigma=4.0,
        size_osc=0.10,
        dropout=0.25,
        conf=(0.05, 0.40),                      # 香烟分数天然很低
    ),
    # 烟：缓慢扩散，框大，形态本来就在变
    "smoke": dict(
        w0=120, h0=140, drift=(0.6, 0.4),
        pos_sigma=5.0, size_sigma=8.0,
        size_osc=0.18,
        dropout=0.18,
        conf=(0.30, 0.80),
    ),
}


def make_sequence(name: str, seed: int = 7, k_override: float | None = None):
    """生成 (gt_boxes, det_boxes)。det 为 None 表示这一帧漏检。"""
    p = PROFILES[name]
    _kbase = CONF_NOISE_K if k_override is None else k_override
    rng = np.random.default_rng(seed)
    w, h = float(p["w0"]), float(p["h0"])
    cx, cy = 640.0, 360.0
    vx, vy = 0.0, 0.0
    gt, det, confs = [], [], []
    for i in range(N_FRAME):
        # 真值：慢速随机游走的漂移
        vx = 0.9 * vx + float(rng.normal(0, p["drift"][0]))
        vy = 0.9 * vy + float(rng.normal(0, p["drift"][1]))
        cx += vx
        cy += vy
        # 尺寸缓慢增长（火的蔓延 / 烟的扩散）
        w *= 1.0 + float(rng.normal(0, 0.002))
        h *= 1.0 + float(rng.normal(0, 0.002))
        g = (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)
        gt.append(g)

        if rng.random() < p["dropout"]:
            det.append(None)
            confs.append(0.0)
            continue
        # 置信度：类内随机，低分框的定位误差更大（用 noise_k 放大）
        cf = float(rng.uniform(*p["conf"]))
        # conf 越低 -> 定位越差。放大倍数**封顶 1.6x**：
        # 真实 YOLO 的低分框确实更飘，但没有"分数 0.05 就飘 3 倍"那么夸张，
        # 模型设太狠会把结论推向"越平滑越好"，那是自欺欺人。
        _k = 1.0 + max(0.0, (0.5 - cf)) / 0.5 * _kbase
        confs.append(cf)
        # 观测 = 真值 + 闪烁尺寸摆动 + 位置/尺寸高斯噪声
        osc = 1.0 + float(rng.uniform(-p["size_osc"], p["size_osc"]))
        dw = max(4.0, w * osc + float(rng.normal(0, p["size_sigma"] * _k)))
        dh = max(4.0, h * osc + float(rng.normal(0, p["size_sigma"] * _k)))
        dcx = cx + float(rng.normal(0, p["pos_sigma"] * _k))
        dcy = cy + float(rng.normal(0, p["pos_sigma"] * _k))
        det.append((dcx - dw / 2, dcy - dh / 2, dcx + dw / 2, dcy + dh / 2))
    return gt, det, confs


def run_chain(gt, det, kf_q, kf_r, base, gain, use_kf=True, use_ema=True,
              cls_name="fire", confs=None, conf_floor=1.0):
    """跑一遍 Kalman + EMA 链路，返回输出框序列（漏检帧沿用上一帧，模拟 coast）。"""
    # 保存现场 -> 注入参数 -> 跑完还原。
    # 必须还原：这些是**模块全局**，不还原的话下一次调用读到的
    # 就是上一次写进去的值（踩过一次：两组参数跑出完全一样的数字）。
    _q0, _r0 = tk.KF_Q_DIAG, tk.KF_R_DIAG
    _b0, _g0 = lv.SMOOTH_BASE, lv.SMOOTH_GAIN
    _c0 = lv.SMOOTH_CONF_FLOOR
    tk.KF_Q_DIAG = tuple(kf_q)
    tk.KF_R_DIAG = tuple(kf_r)
    lv.SMOOTH_BASE = dict(base)
    lv.SMOOTH_GAIN = dict(gain)
    lv.SMOOTH_CONF_FLOOR = float(conf_floor)

    kf = None
    raw = []
    for box in det:
        if box is None:
            raw.append(None)              # coast：不推进，保留上一帧
            continue
        if use_kf:
            if kf is None:
                kf = tk._Kalman(box)
                b = kf.box()
            else:
                kf.predict()
                b = kf.correct(box)
        else:
            b = box
        raw.append(b)

    # 漏检帧用上一帧输出补位，保证序列连续（线上画框也是这么做的）
    confs = list(confs) if confs is not None else [0.5] * len(raw)
    filled, fconf, last, lconf = [], [], None, 0.5
    for b, c in zip(raw, confs):
        if b is None:
            b, c = last, lconf
        else:
            last, lconf = b, c
        filled.append(b)
        fconf.append(c)

    if not use_ema:
        # ⚠️ 这里也要还原！曾经漏掉这一条 return 分支，结果"只 Kalman"
        # 那一行把 KF_R 留在了旧值，后面第三节读到的就不是源码里的真值。
        tk.KF_Q_DIAG, tk.KF_R_DIAG = _q0, _r0
        lv.SMOOTH_BASE, lv.SMOOTH_GAIN = _b0, _g0
        lv.SMOOTH_CONF_FLOOR = _c0
        return [b for b in filled if b is not None]

    view = lv.LiveView.__new__(lv.LiveView)
    view._prev_boxes = []
    out = []
    for b, c in zip(filled, fconf):
        if b is None:
            continue
        dets = [_Box(b, cls_name, c)]
        sm = view._smooth(dets)
        out.append(sm[0][0])
    tk.KF_Q_DIAG, tk.KF_R_DIAG = _q0, _r0
    lv.SMOOTH_BASE, lv.SMOOTH_GAIN = _b0, _g0
    lv.SMOOTH_CONF_FLOOR = _c0
    return out


class _Box:
    """最小可用的 Detection 替身：_smooth 只读 xyxy / cls_name / conf。"""

    __slots__ = ("xyxy", "cls_name", "conf")

    def __init__(self, b, name="fire", conf=0.5):
        self.xyxy = b
        self.cls_name = name
        self.conf = conf


def centers(boxes):
    return np.array([[(b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0] for b in boxes])


def metrics(gt, out):
    """gt 与 out 都从"第一次有输出"的位置对齐（前面几帧还在建立轨迹）。"""
    n = min(len(gt), len(out))
    g = centers(gt[len(gt) - n:])
    o = centers(out[len(out) - n:])
    err = float(np.linalg.norm(o - g, axis=1).mean())
    # jerk：位置的二阶差分。这里**只算有真实检测的连续段**，
    # 因为漏检帧的输出本来就是"沿用上一帧"，会人为拉低 jerk，掩盖真实抖动。
    d2 = np.diff(o, n=2, axis=0)
    jerk = float(np.abs(d2).mean()) if len(d2) else 0.0
    # 滞后：把输出序列整体前后平移，找与真值误差最小的一档
    best_lag, best_e = 0, 1e9
    for lag in range(0, 12):
        if lag >= n:
            break
        e = np.linalg.norm(o[lag:] - g[:len(g) - lag], axis=1).mean()
        if e < best_e:
            best_e, best_lag = e, lag
    return dict(err=err, jerk=jerk, lag=best_lag, lag_err=float(best_e))


# 每类目标**独立扫**：字典本身就是按类取值的，一个类的参数不会影响另一个。
# 混在一起扫只会让"火的偏好"污染"香烟的偏好"。
BASE_GRID = [0.15, 0.22, 0.30, 0.38, 0.45, 0.55, 0.65]
GAIN_GRID = [0.0, 0.6, 1.2, 1.8, 2.6, 3.5]
KF_GRID = [
    (2.0, 2.0, 8.0, 8.0),
    (3.0, 3.0, 8.0, 8.0),
    (4.0, 4.0, 10.0, 10.0),
    (5.0, 5.0, 12.0, 12.0),
]
KF_Q = (1.5, 1.5, 9.0, 9.0, 2.5, 2.5, 3.0, 3.0)

# 当前线上参数（导入时快照一份，之后 run_chain 改全局也不会影响这里）
CUR_BASE = dict(lv.SMOOTH_BASE)
CUR_GAIN = dict(lv.SMOOTH_GAIN)
CUR_KF_R = tuple(tk.KF_R_DIAG)
CUR_KF_Q = tuple(tk.KF_Q_DIAG)
CUR_CONF_FLOOR = float(getattr(lv, "SMOOTH_CONF_FLOOR", 1.0))
# 上一版参数（用户反馈"不精准/不平滑"的那一版）
OLD = {
    "fire":      dict(base=0.15, gain=0.0),
    "cigarette": dict(base=0.35, gain=1.2),
    "smoke":     dict(base=0.45, gain=1.4),
}


def filled_det(det):
    out, last = [], None
    for b in det:
        if b is None:
            b = last
        else:
            last = b
        out.append(b)
    return [b for b in out if b is not None]


def eval_one(k, base, gain, kf_r, conf_floor=1.0):
    gt, det, cfs = make_sequence(k)
    b, g = dict(CUR_BASE), dict(CUR_GAIN)
    b[k], g[k] = base, gain
    out = run_chain(gt, det, KF_Q, kf_r, b, g, cls_name=k,
                    confs=cfs, conf_floor=conf_floor)
    return metrics(gt, out)


# 两套噪声假设：只用一套调参必然过拟合，所以两套都跑，取交集才算过关。
K_GRID = (0.0, CONF_NOISE_K)
K_NAME = {0.0: "A(噪声无关)", CONF_NOISE_K: f"B(低分更飘×{1 + CONF_NOISE_K:.1f})"}

# 上一版线上参数（用户反馈"不精准/不平滑"的那一版）与其 KF
OLD = {"fire": (0.15, 0.0), "cigarette": (0.35, 1.2), "smoke": (0.45, 1.4)}
OLD_KF_R = (5.0, 5.0, 12.0, 12.0)


def seq(k, kk):
    return make_sequence(k, k_override=kk)


def ev(k, base, gain, kf_r, floor, kk, cache):
    gt, det, cfs = cache[(k, kk)]
    b, g = dict(CUR_BASE), dict(CUR_GAIN)
    b[k], g[k] = base, gain
    out = run_chain(gt, det, CUR_KF_Q, kf_r, b, g, cls_name=k,
                    confs=cfs, conf_floor=floor)
    return metrics(gt, out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", type=int, default=5)
    args = ap.parse_args()

    cache = {(k, kk): seq(k, kk) for k in PROFILES for kk in K_GRID}

    # ── 一、基线：YOLO 框本身有多抖 ────────────────────────────
    print("=" * 104)
    print("一、基线（误差 = 与真值中心的平均距离 px；抖动 = 中心二阶差分 px/帧²）")
    print("-" * 104)
    print(f"{'假设':<20}" + "".join(f"{k:>26}" for k in PROFILES))
    for kk in K_GRID:
        cells = []
        for k in PROFILES:
            gt, det, _c = cache[(k, kk)]
            m = metrics(gt, filled_det(det))
            cells.append(f"裸检测 {m['err']:.2f}/{m['jerk']:.2f}")
        print(f"{K_NAME[kk]:<20}" + "".join(f"{c:>26}" for c in cells))
    print("  平滑要做的事：把这个误差和抖动都压下来 —— 注意是**两个都要**。")
    print("  只压抖动不看误差，就会得到「框很稳但落在目标后面」的效果。")

    # ── 二、逐类扫参（两套假设的误差取平均做目标函数）──────────
    print()
    print("=" * 104)
    print("二、逐类扫参（目标 = 两套假设下 (误差 + 0.4×抖动) 的平均）")
    print("-" * 104)
    for k in PROFILES:
        rows = []
        for (b, g) in itertools.product(BASE_GRID, GAIN_GRID):
            ms = [ev(k, b, g, CUR_KF_R, CUR_CONF_FLOOR, kk, cache) for kk in K_GRID]
            loss = sum(m["err"] + 0.4 * m["jerk"] for m in ms) / len(ms)
            rows.append((loss, b, g, ms))
        rows.sort(key=lambda r: r[0])
        print(f"\n【{k}】KF_R={CUR_KF_R[0]:.0f}/{CUR_KF_R[2]:.0f}  "
              f"线上 α={CUR_BASE[k]:.2f} g={CUR_GAIN[k]:.1f}")
        head_txt = f"  {'alpha':>6}{'gain':>6}" + "".join(
            f"{K_NAME[kk] + ' 误差/抖动':>24}" for kk in K_GRID) + f"{'loss':>9}"
        print(head_txt)
        for loss, b, g, ms in rows[:args.top]:
            cells = "".join(f"{m['err']:12.2f}/{m['jerk']:.2f}".rjust(24) for m in ms)
            print(f"  {b:>6.2f}{g:>6.1f}{cells}{loss:>9.3f}")

    # ── 三、线上参数 vs 上一版（两套假设交叉验证）─────────────
    print()
    print("=" * 104)
    print("三、线上参数实测（直接读源码常量）vs 上一版 —— 两套假设都要更好才算真改进")
    print("-" * 104)
    print(f"KF_Q_DIAG = {CUR_KF_Q}")
    print(f"KF_R_DIAG = {CUR_KF_R}      （上一版 {OLD_KF_R}）")
    print(f"SMOOTH_BASE      = {CUR_BASE}")
    print(f"SMOOTH_GAIN      = {CUR_GAIN}")
    print(f"SMOOTH_CONF_FLOOR = {CUR_CONF_FLOOR}")
    print("-" * 104)
    print(f"{'类别':<11}{'方案':<8}" + "".join(f"{K_NAME[kk] + ' 误差/抖动':>24}" for kk in K_GRID))
    for k in PROFILES:
        o = [ev(k, *OLD[k], OLD_KF_R, CUR_CONF_FLOOR, kk, cache) for kk in K_GRID]
        n = [ev(k, CUR_BASE[k], CUR_GAIN[k], CUR_KF_R, CUR_CONF_FLOOR, kk, cache)
             for kk in K_GRID]
        fmt = lambda ms: "".join(f"{m['err']:12.2f}/{m['jerk']:.2f}".rjust(24) for m in ms)  # noqa: E731
        print(f"{k:<11}{'上一版':<8}{fmt(o)}")
        print(f"{'':<11}{'现在':<8}{fmt(n)}")
        d = "  ".join(
            f"{K_NAME[kk].split('(')[0]} 误差{(o[i]['err'] - n[i]['err']) / o[i]['err'] * 100:+.0f}%"
            f" 抖动{(o[i]['jerk'] - n[i]['jerk']) / o[i]['jerk'] * 100:+.0f}%"
            for i, kk in enumerate(K_GRID))
        print(f"{'':<11}{'改善':<8}{d}")
    print("=" * 104)
    print("注：「抖动」绝对值才是观感，别只看百分比。火 0.44px/帧² 和香烟 2.57")
    print("    在屏幕上完全是两回事（火的框 34x42，香烟只有 16x44）。")
    print("=" * 104)


if __name__ == "__main__":
    main()
