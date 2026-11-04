"""同一批帧上做后端 A/B：.pt / FP16 引擎 / FP32 引擎 谁的框更稳。

为什么非要做这个
----------------
"画面一直闪"这个问题上，在线采样的结论**不可信** —— 用户一会儿在抽烟、
一会儿走开，同一路摄像头的框数天然在变。之前就因此得出过相反结论：
同一个指标，一次说"加 TRT 更闪"，换一次采样又说"基本不闪"。

唯一靠谱的办法：先把帧抓到磁盘，再让三个后端**跑同一批帧**，
逐帧比较检出数量和框位置。输入完全一致，差异只可能来自后端。

指标
----
  框数变化率 : 相邻两帧框数不同的比例（越小越稳）
  位置位移   : 同一目标（IoU>0.3 配对）在两帧间的中心位移 / 框对角线
  平均框数   : 检出能力（太少=漏，太多=噪）

用法
----
    python scripts/bench_backend_ab.py
    python scripts/bench_backend_ab.py --frames 60 --cam cam_1
"""
from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
FRAMES = ROOT / "logs" / "ab_frames"

# 三个后端：主模型 + 香烟模型各一份
BACKENDS = {
    ".pt  (FP32 原生)": dict(
        main=r"D:\trt_export\main_pt.pt",
        extra=r"D:\trt_export\cig_pt.pt"),
    "引擎 FP16": dict(
        main=r"D:\trt_export\main_fp16.engine",
        extra=r"D:\trt_export\cig_fp16.engine"),
    "引擎 FP32": dict(
        main=r"D:\视觉安防系统\models\fire_smoke_v5_26s960_best.engine",
        extra=r"D:\视觉安防系统\models\cig_v32_smoking_best.engine"),
}


def iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / u if u > 0 else 0.0


def run_backend(tag: str, cfg: dict, imgs: list, cam: str) -> dict:
    from src.infer import Detector

    det = Detector(cfg["main"], conf=0.05, imgsz=960, device=0,
                   extra_weights=cfg["extra"], extra_conf=0.03)
    per: dict[str, list] = {"fire": [], "smoke": [], "cigarette": []}
    for im in imgs:
        # 走生产同一条路径（含切片、降频缓存、二级 NMS）
        out = det.detect_auto_batch([im], [cam], tile_grid=3, overlap=0.25)[0]
        cur = {"fire": [], "smoke": [], "cigarette": []}
        for d in out:
            n = str(d.cls_name).lower()
            if n in cur:
                cur[n].append(d.xyxy)
        for k in cur:
            per[k].append(cur[k])
    return per


def summarize(name: str, per: dict) -> dict:
    res = {}
    for k, boxes in per.items():
        counts = [len(b) for b in boxes]
        flips = (sum(1 for i in range(1, len(counts)) if counts[i] != counts[i - 1])
                 / max(1, len(counts) - 1))
        moves = []
        for i in range(1, len(boxes)):
            used = set()
            for b in boxes[i]:
                best, bi = -1, 0.3
                for j, p in enumerate(boxes[i - 1]):
                    if j in used:
                        continue
                    v = iou(p, b)
                    if v > bi:
                        bi, best = v, j
                if best >= 0:
                    used.add(best)
                    p = boxes[i - 1][best]
                    dx = (b[0] + b[2] - p[0] - p[2]) / 2.0
                    dy = (b[1] + b[3] - p[1] - p[3]) / 2.0
                    dg = max(1.0, ((b[2] - b[0]) ** 2 + (b[3] - b[1]) ** 2) ** 0.5)
                    moves.append((dx * dx + dy * dy) ** 0.5 / dg)
        res[k] = dict(mean=statistics.mean(counts) if counts else 0,
                      flips=flips, move=statistics.mean(moves) if moves else 0)
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=40)
    ap.add_argument("--cam", default="cam_0")
    args = ap.parse_args()

    d = FRAMES / args.cam
    files = sorted(d.glob("*.jpg"))[:args.frames]
    if not files:
        print(f"[x] 没有帧：{d}（先跑抓帧）")
        return
    imgs = [cv2.imread(str(f)) for f in files]
    imgs = [im for im in imgs if im is not None]
    print("=" * 92)
    print(f"后端 A/B —— {args.cam}，{len(imgs)} 帧（同一批帧跑三个后端）")
    print("=" * 92)

    results = {}
    for tag, cfg in BACKENDS.items():
        print(f"  跑 {tag} ...", flush=True)
        per = run_backend(tag, cfg, imgs, args.cam)
        results[tag] = summarize(tag, per)

    print()
    print(f"{'后端':<18}{'类别':<11}{'平均框数':>9}{'框数变化率':>12}{'位置位移':>10}")
    print("-" * 92)
    for tag in BACKENDS:
        r = results[tag]
        for k in ("fire", "smoke", "cigarette"):
            m = r[k]
            print(f"{tag if k == 'fire' else '':<18}{k:<11}{m['mean']:>9.2f}"
                  f"{m['flips'] * 100:>11.0f}%{m['move']:>10.3f}")
        print()
    print("=" * 92)
    print("解读：同样的帧，变化率越低=框越不容易闪；平均框数越多=召回越高（也可能是噪）。")


if __name__ == "__main__":
    main()
