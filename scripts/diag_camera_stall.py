"""摄像头停帧诊断：判断"画面突变时卡住"是摄像头本身还是软件。

怎么用
------
1. 先关掉正在跑的监控（它会独占摄像头）：
       taskkill /F /IM python.exe
   或者关掉 FireMonitor.exe 的窗口

2. 运行本脚本，**在它运行的 30 秒里，做那个会让画面卡住的动作**
   （开关灯、快速挥手、走动、开关手电筒……）

3. 看最后的结论：
   - 「摄像头侧」尖峰集中在帧差大 / 亮度跳变大的时刻 → 是自动曝光/增益在重新收敛
   - 完全没有尖峰 → 摄像头输出是稳的，问题在软件链路
   - 尖峰和帧差无关，随机分布 → 更像 USB 带宽或驱动调度

为什么这样能区分
----------------
摄像头在光线突变时，自动曝光要重新算积分时间，输出会暂停几百毫秒。
这段时间里 `cap.read()` 就是拿不到新帧。如果尖峰全都落在"亮度跳变"的帧上，
基本可以断定是摄像头的自动曝光行为 —— 软件侧改不了，只能接受或改曝光模式。

同理，如果尖峰全都落在"帧间差异大"的帧上（画面在大幅运动），
那是 JPEG/带宽吃紧，可以通过降分辨率或降质量缓解。
"""
from __future__ import annotations

import argparse
import time

import cv2
import numpy as np


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--seconds", type=float, default=30.0)
    ap.add_argument("--w", type=int, default=640)
    ap.add_argument("--h", type=int, default=480)
    ap.add_argument("--spike-ms", type=float, default=100.0,
                    help="超过多少毫秒算一次停帧（默认 100）")
    a = ap.parse_args()

    cap = cv2.VideoCapture(a.camera, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print(f"[x] 摄像头 {a.camera} 打不开。是不是监控还在跑？先关掉它。")
        return 1
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, a.w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, a.h)
    cap.set(cv2.CAP_PROP_FPS, 30)
    for _ in range(15):
        cap.read()                       # 预热，跳过刚打开的抖动

    print("=" * 62)
    print(f"  开始采集 {a.seconds:.0f} 秒")
    print("  👉 现在请在镜头前做那个会让画面卡住的动作（开关灯 / 快速挥手）")
    print("=" * 62)

    prev_gray = None
    rec: list[tuple[float, float, float, float]] = []   # ts, gap_ms, diff, bright
    last_ts = time.time()
    t_end = time.time() + a.seconds

    while time.time() < t_end:
        ok, frame = cap.read()
        if not ok:
            continue
        now = time.time()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.int16)
        diff = float(np.abs(gray - prev_gray).mean()) if prev_gray is not None else 0.0
        bright = float(gray.mean())
        rec.append(((now - last_ts) * 1000.0, diff, bright, now))
        prev_gray = gray
        last_ts = now
    cap.release()

    if len(rec) < 10:
        print("[x] 样本太少，可能摄像头没出帧")
        return 1

    gaps = np.array([r[0] for r in rec])
    diffs = np.array([r[1] for r in rec])
    brights = np.array([r[2] for r in rec])

    n = len(gaps)
    avg_gap = float(gaps.mean())
    print(f"\n共 {n} 帧 / {a.seconds:.0f} 秒")
    print(f"  帧率      {1000/avg_gap:.1f} fps（平均间隔 {avg_gap:.0f}ms）")
    print(f"  间隔      中位 {np.median(gaps):.0f}ms  P90 {np.percentile(gaps,90):.0f}ms  "
          f"P99 {np.percentile(gaps,99):.0f}ms  最大 {gaps.max():.0f}ms")
    print(f"  帧间差异  中位 {np.median(diffs):.2f}  最大 {diffs.max():.2f}")
    print(f"  亮度      {brights.min():.0f} ~ {brights.max():.0f}（中位 {np.median(brights):.0f}）")

    spike_idx = [i for i in range(n) if gaps[i] > a.spike_ms]
    print(f"\n停帧（间隔 > {a.spike_ms:.0f}ms）：{len(spike_idx)} 次")

    if not spike_idx:
        print("\n结论：摄像头输出是稳的，这轮没复现停帧。")
        print("      说明问题只在特定条件下出现，或者不在摄像头侧。")
        return 0

    print(f"{'序号':<6}{'间隔ms':<10}{'帧差':<10}{'亮度':<10}{'亮度变化'}")
    bright_delta = []
    diff_of_spike = []
    for k, i in enumerate(spike_idx[:15], 1):
        d_b = abs(brights[i] - brights[i - 1]) if i > 0 else 0.0
        bright_delta.append(d_b)
        diff_of_spike.append(float(diffs[i]))
        print(f"{k:<6}{gaps[i]:<10.0f}{diffs[i]:<10.2f}{brights[i]:<10.0f}{d_b:+.0f}")

    # 对照：非尖峰帧的平均帧差 / 亮度变化
    normal = [i for i in range(n) if gaps[i] <= a.spike_ms]
    n_diff = float(np.mean([diffs[i] for i in normal])) if normal else 0.0
    n_bdl = float(np.mean([abs(brights[i] - brights[i-1]) for i in normal if i > 0])) if normal else 0.0
    s_diff = float(np.mean(diff_of_spike)) if diff_of_spike else 0.0
    s_bdl = float(np.mean(bright_delta)) if bright_delta else 0.0

    print(f"\n尖峰时刻  平均帧差 {s_diff:.2f}  平均亮度跳变 {s_bdl:.1f}")
    print(f"正常时刻  平均帧差 {n_diff:.2f}  平均亮度跳变 {n_bdl:.1f}")

    print("\n" + "=" * 62)
    if s_bdl > max(3.0, n_bdl * 3):
        print("结论：停帧与【亮度跳变】高度相关 → 摄像头自动曝光在重新收敛。")
        print("      这是硬件行为，软件改不了。对策：固定光照 / 关掉自动曝光 / 换摄像头。")
    elif s_diff > max(3.0, n_diff * 2.5):
        print("结论：停帧与【画面大幅运动】相关 → JPEG 体积或 USB 带宽吃紧。")
        print("      对策：已经做了体积兜底，可以再把分辨率降到 480x360 试试。")
    else:
        print("结论：停帧与画面变化都不相关 → 更像驱动调度或 USB 供电波动。")
        print("      对策：换个 USB 口（优先主板后置口，别用前面板/HUB），关掉其他占 USB 带宽的设备。")
    print("=" * 62)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
