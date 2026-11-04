"""量化「画面一直闪」到底有多严重。

为什么需要这个脚本
------------------
用户的原话是"一直闪""框乱跳"，但这类主观描述没法比较两个版本。
这里把"闪"拆成两个可测的量，直接从运行中的看板接口采样：

  1. 框数变化率 —— 同一路、同一类别，相邻两次采样之间框的数量变了多少次。
     这是最直观的"闪"：屏幕上本来有 2 个框，下一帧变 1 个、再下一帧变 3 个。
  2. 位置抖动 —— 把相邻采样里 IoU 最好的一对框配对，看中心位移有多大。

同时还报检测 fps / 推流 fps，因为"闪"和帧率低经常是同一个原因。

用法
----
    python scripts/bench_flicker.py             # 两路各测 12 秒
    python scripts/bench_flicker.py --secs 20
    python scripts/bench_flicker.py --tag 加TRT前   # 打标签，方便前后对比

⚠️ 测之前确认只有一个实例在跑。两个实例抢同一批摄像头时，这个数字会
   翻好几倍，看起来像算法问题，其实是进程重复（这个坑已经踩过一次）。
"""
from __future__ import annotations

import argparse
import collections
import json
import statistics
import time
import urllib.request

BASE = "http://127.0.0.1:5050"
CLASSES = ("fire", "smoke", "cigarette")


def _opener():
    # 显式关掉代理：本机回环走代理会超时（这台机器上装过全局代理）
    return urllib.request.build_opener(urllib.request.ProxyHandler({}))


def _state(op, cam: str) -> dict | None:
    try:
        with op.open(f"{BASE}/api/state?cam={cam}", timeout=2) as r:
            return json.loads(r.read().decode())
    except Exception:                       # noqa: BLE001
        return None


def _iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / u if u > 0 else 0.0


def measure(cam: str, secs: float) -> dict:
    op = _opener()
    counts: dict[str, list[int]] = {k: [] for k in CLASSES}
    moves: dict[str, list[float]] = {k: [] for k in CLASSES}
    det_fps: list[float] = []
    pub_fps: list[float] = []
    prev: dict[str, list] = {}
    t0 = time.time()
    n = 0
    last_seq = -1
    while time.time() - t0 < secs:
        s = _state(op, cam)
        if s:
            # ⚠️ 只在**新帧**上比较。原先每 60ms 采一次样，而检测帧率会变
            # （5.5fps 的版本一次检测结果会被采到两三次，看起来"很稳"；
            # 10fps 的版本每次采样都是新结果，变化率自然更高）。
            # 不比 seq 就会得出"加 TensorRT 让画面更闪"的假结论。
            _seq = int(s.get("seq") or -1)
            if _seq == last_seq:
                time.sleep(0.02)
                continue
            last_seq = _seq
            n += 1
            det_fps.append(float(s.get("fps") or 0))
            pub_fps.append(float(s.get("pub_fps") or 0))
            cur: dict[str, list] = {k: [] for k in CLASSES}
            for d in s.get("dets") or []:
                nm = str(d.get("name"))
                if nm in cur:
                    cur[nm].append(d.get("xyxy"))
            for k in CLASSES:
                counts[k].append(len(cur[k]))
                # 配对：贪心取 IoU 最大的一对，量中心位移 / 框对角线（归一化）
                used: set[int] = set()
                for b in cur[k]:
                    best, bi = -1, 0.3
                    for i, p in enumerate(prev.get(k) or []):
                        if i in used:
                            continue
                        v = _iou(p, b)
                        if v > bi:
                            bi, best = v, i
                    if best >= 0:
                        used.add(best)
                        p = prev[k][best]
                        dx = (b[0] + b[2] - p[0] - p[2]) / 2.0
                        dy = (b[1] + b[3] - p[1] - p[3]) / 2.0
                        diag = max(1.0, ((b[2] - b[0]) ** 2 + (b[3] - b[1]) ** 2) ** 0.5)
                        moves[k].append((dx * dx + dy * dy) ** 0.5 / diag)
            prev = cur
        time.sleep(0.06)
    return dict(n=n, counts=counts, moves=moves, det_fps=det_fps, pub_fps=pub_fps)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--secs", type=float, default=12.0)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    print("=" * 88)
    print(f"画面闪烁基准 {args.tag}   （每路 {args.secs:.0f} 秒）")
    print("=" * 88)
    summary = {}
    for cam in ("cam_0", "cam_1"):
        m = measure(cam, args.secs)
        if not m["n"]:
            print(f"{cam}: 拿不到数据（服务没起来？）")
            continue
        print(f"\n【{cam}】采样 {m['n']} 次  检测 {statistics.mean(m['det_fps']):.1f} fps  "
              f"推流 {statistics.mean(m['pub_fps']):.1f} fps")
        print(f"  {'类别':<11}{'框数均值':>9}{'波动':>7}{'变化率':>9}{'位置抖动':>10}")
        for k in CLASSES:
            v = m["counts"][k]
            mv = m["moves"][k]
            flips = (sum(1 for i in range(1, len(v)) if v[i] != v[i - 1]) / max(1, len(v) - 1))
            jit = statistics.mean(mv) if mv else 0.0
            flag = ""
            if v and max(v) > 0 and flips > 0.25:
                flag = "  <- 闪"
            print(f"  {k:<11}{statistics.mean(v):>9.2f}{statistics.pstdev(v):>7.2f}"
                  f"{flips * 100:>8.0f}%{jit:>10.3f}{flag}")
            summary[(cam, k)] = (flips, jit)
    print("\n  变化率 = 相邻采样框数不同的比例（越小越稳）；"
          "位置抖动 = 归一化中心位移（越小越稳）")
    print("=" * 88)

    # 供跨版本对比：把关键数字压成一行
    line = " ".join(f"{c}:{k[:3]}={v[0] * 100:.0f}%/{v[1]:.3f}"
                    for (c, k), v in summary.items())
    print(f"对比用：{line}")


if __name__ == "__main__":
    main()
