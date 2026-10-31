"""真实火视频的端到端验收：报不报、多快报、报几条。

为什么要单独测这一段
--------------------
`eval_fire.py` 报的是 mAP/P/R，回答"框画得准不准"；
`eval_fp_end2end.py` 回答"没火的时候会不会瞎报"。
还差一个：**真火递进来，流水线多久开第一枪、会不会被自己的闸拦掉**。

这里用的是 test 集里**同一段视频的连续帧**（`A_烟雾_火_fire_CV*.jpg`），
所以时间轴是真实的，延迟可以直接读秒。

跑法：
    python scripts/eval_fire_video.py                 # 闪烁闸开（线上配置）
    python scripts/eval_fire_video.py --no-gate       # 对照
"""
from __future__ import annotations

import argparse
import glob
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.pipeline import Pipeline                                    # noqa: E402
from src.sources.base import CameraInfo                              # noqa: E402
from src.sources.fixed import FixedCameraSource                      # noqa: E402

DATA = Path("E:/datasets/fire_smoke_v5/test")
WEIGHTS = ROOT / "models" / "fire_smoke_v5_26s960_best.pt"
FPS = 25.0


def imread_u(p: Path):
    import cv2
    try:
        return cv2.imdecode(np.frombuffer(p.read_bytes(), np.uint8), cv2.IMREAD_COLOR)
    except Exception:                                                # noqa: BLE001
        return None


def pick_continuous(pattern: str, limit: int):
    """按帧号排序取同一段视频的连续帧。尺寸不一的丢掉 —— 混进别的视频会炸背景建模。"""
    files = sorted(glob.glob(pattern),
                   key=lambda p: int(re.search(r"CV(\d+)\.jpg$", p).group(1)))
    main, out = None, []
    for p in files:
        im = imread_u(Path(p))
        if im is None:
            continue
        if main is None:
            main = im.shape
        if im.shape != main:
            continue
        out.append(im)
        if len(out) >= limit:
            break
    if not out:
        raise SystemExit(f"没读到图：{pattern}")
    return out


def run(frames, gate: bool, imgsz: int):
    from src.infer import Detector
    det = Detector(str(WEIGHTS), conf=0.20, imgsz=imgsz, device=0)
    evs: list = []
    pipe = Pipeline(det, cfg={
        "detect_mode": "auto",
        "flicker_gate": gate,
        "light_memory": False,
        "fast_smoke": False,
    }, arbiter=None)
    src = FixedCameraSource(CameraInfo(cam_id="real_fire", kind="fixed", fps=FPS),
                            frames=iter(frames), warmup=0)
    pipe.run(src, max_frames=len(frames), quiet=True,
             on_event=lambda ev, _r: evs.append(ev))
    return evs, pipe.stats


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=113)
    ap.add_argument("--imgsz", type=int, default=896)
    ap.add_argument("--no-gate", action="store_true")
    a = ap.parse_args()

    frames = pick_continuous(str(DATA / "images" / "A_烟雾_火_fire_CV*.jpg"), a.frames)
    print(f"真实火视频连续帧 {len(frames)} 张  "
          f"{frames[0].shape[1]}x{frames[0].shape[0]}  = {len(frames) / FPS:.1f} 秒")
    print(f"权重 {WEIGHTS.name}  imgsz={a.imgsz}  闪烁闸{'关' if a.no_gate else '开'}\n")

    evs, st = run(frames, not a.no_gate, a.imgsz)
    formal = [e for e in evs if e.level in ("warning", "alarm")]
    notice = [e for e in evs if e.level == "notice"]
    cleared = [e for e in evs if e.level == "cleared"]

    print(f"正式告警 {len(formal)} 条 / notice {len(notice)} 条 / 撤销 {len(cleared)} 条")
    print(f"静态光源拦截 {st['static_light_blocked']} 帧  "
          f"蒸汽抑制 {st['steam_suppressed']} 帧  同区域冷却 {st['cooldown_blocked']} 次")
    if formal:
        e = formal[0]
        print(f"\n首次告警：t={e.t_alarm:.2f}s  级别 {e.level}  score={e.score:.2f}")
        print(f"  判据：{'；'.join(e.reasons[:3])}")
        if len(formal) > 1:
            print(f"  后续 {len(formal) - 1} 条："
                  + ", ".join(f"{x.t_alarm:.1f}s" for x in formal[1:6]))
    else:
        print("\n❌ 整段视频一条正式告警都没有 —— 这是漏报，不能上线")
        for e in notice[:4]:
            print(f"  · [notice] t={e.t_alarm:.1f}s {e.reasons[0][:60]}")
    lat = formal[0].t_alarm if formal else float("nan")
    print(f"\n端到端首报延迟 {lat:.2f}s（含 {len(frames) / FPS:.0f}s 素材内的确认帧数）")
    return 0 if formal else 2


if __name__ == "__main__":
    raise SystemExit(main())
