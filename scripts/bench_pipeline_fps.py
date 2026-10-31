"""端到端帧率 A/B：整条流水线（检测+追踪+时序+事件）在三种检测策略下的 fps。

为什么不只看检测耗时：检测只占一半。切片一旦生效，轨迹会变多、每帧要算的
样本也多，追踪+判别的开销跟着涨（实测 11ms → 26ms）。所以必须端到端量。

两组输入：
  · 真火   —— 那段 113 帧**逐帧连续**的真实火视频帧（有火，切片会全开）
  · 空白   —— 合成的墙面 + 曝光漂移（真实枪机绝大多数字段长这样）

用法：
    python scripts/bench_pipeline_fps.py
    python scripts/bench_pipeline_fps.py --frames 113 --repeat 1
"""
from __future__ import annotations

import argparse
import glob
import re
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.infer import Detector                                   # noqa: E402
from src.pipeline import Pipeline                                # noqa: E402
from src.sources.base import CameraInfo                          # noqa: E402
from src.sources.fixed import FixedCameraSource                  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
WEIGHTS = ROOT / "models" / "fire_smoke_v5_26s960_best.pt"
DATA = Path("E:/datasets/fire_smoke_v5/test")
FPS = 25.0


def imread_u(p: str):
    return cv2.imdecode(np.fromfile(p, dtype=np.uint8), cv2.IMREAD_COLOR)


def pick_continuous(pattern: str, limit: int) -> list[np.ndarray]:
    files = sorted(glob.glob(pattern),
                   key=lambda p: int(re.search(r"CV(\d+)\.jpg$", p).group(1)))
    main, out = None, []
    for p in files:
        im = imread_u(p)
        if im is None:
            continue
        if main is None:
            main = im.shape
        if im.shape != main:
            continue
        out.append(im)
        if len(out) >= limit:
            break
    return out


def blank_frames(n: int) -> list[np.ndarray]:
    yy, xx = np.mgrid[0:540, 0:960].astype(np.float32)
    base = np.clip(96 + 44 * (xx / 960)[..., None] + np.array([8, 6, 4], np.float32),
                   0, 255).astype(np.uint8)
    return [cv2.convertScaleAbs(base, alpha=1.0, beta=1.2 * np.sin(i / 60.0))
            for i in range(n)]


def run(pipe_cfg: dict, frames: list[np.ndarray], warm: int) -> tuple[float, dict]:
    det = Detector(str(WEIGHTS), imgsz=896, conf=0.25)
    pipe = Pipeline(det, cfg=pipe_cfg)
    src = FixedCameraSource(CameraInfo(cam_id="bench", kind="fixed", fps=FPS),
                            frames=iter(frames), warmup=warm)
    pipe.run(src, max_frames=len(frames))       # 预热（含 CUDA 初始化）
    pipe = Pipeline(det, cfg=pipe_cfg)          # 状态清零，重量一次
    src = FixedCameraSource(CameraInfo(cam_id="bench", kind="fixed", fps=FPS),
                            frames=iter(frames), warmup=warm)
    t0 = time.perf_counter()
    res = pipe.run(src, max_frames=len(frames))
    el = time.perf_counter() - t0
    return len(frames) / max(el, 1e-9), res["stats"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=113)
    ap.add_argument("--blank", type=int, default=60)
    a = ap.parse_args()

    fire = pick_continuous(str(DATA / "images" / "A_烟雾_火_fire_CV*.jpg"), a.frames)
    blanks = blank_frames(a.blank)
    print(f"真火连续帧 {len(fire)} 张 {fire[0].shape[1]}x{fire[0].shape[0]}  ·  "
          f"空白帧 {len(blanks)} 张\n")

    modes = [("slice（现状）", {"detect_mode": "slice"}),
             ("auto（分级）", {"detect_mode": "auto"}),
             ("full（不切片）", {"detect_mode": "full"})]
    print(f"{'策略':<16}{'真火 fps':>10}{'flicker拦':>10}{'空白 fps':>10}{'提速':>8}")
    base = None
    for name, cfg in modes:
        f1, s1 = run(cfg, fire, warm=0)
        f2, _ = run(cfg, blanks, warm=0)
        if base is None:
            base = f2
        print(f"{name:<16}{f1:>10.1f}{s1['static_light_blocked']:>10}"
              f"{f2:>10.1f}{f2/base:>7.2f}x")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
