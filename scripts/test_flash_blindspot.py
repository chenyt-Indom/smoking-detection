"""闪烁闸对「本身就在闪的假火」到底拦不拦得住 —— 实测记录。

原本的担心
----------
闸的判据是"框内像素是否逐帧跳变"。这招对恒亮的灯（路灯/招牌/反光）有效，
但**警灯爆闪、电焊弧光、霓虹 PWM、电视里的火焰画面**本身就在闪 ——
按理说会被判成 fire 档直接放行，一路报成正式告警。

`eval_fp_end2end.py` 里那张消防车只报成 notice，是因为它被冻结成 60 帧重复
播放，红蓝爆闪在帧间一动不动（测试artifact），不能当结论。

实测结论（2026-10-31）
---------------------
把那张消防车按 5Hz、±40% / ±70% / ±90% 三档调亮压暗，闸**全部拦住**：
`static_light_blocked` 41~82 帧，正式告警 0 条。

原因不是闸变聪明了，而是**面积**：判据要求"框内 ≥25% 的像素在跳"，
而车灯在检测框里只占几个百分点，怎么闪都够不到那条线。担心不成立。

**还没验证的边界**：判据是框内占比 ≥25%，所以如果光源**大面积**闪烁
（近距离电焊弧光、电视屏幕里的火铺满画面），理论上会穿过去。
对这份素材做 blob 叠加试过，但目标框只有 33x21px，亮斑画不进去、测不出来 ——
所以这一条只能说"未验证"，不编数字。要验证得找一段真实电焊/霓虹视频。

跑法：
    python scripts/test_flash_blindspot.py            # 默认 ±40%
    python scripts/test_flash_blindspot.py --depth 0.9
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.pipeline import Pipeline                                    # noqa: E402
from src.sources.base import CameraInfo                              # noqa: E402
from src.sources.fixed import FixedCameraSource                      # noqa: E402

BG = Path("E:/BaiduNetdiskDownload/Y_火灾+烟雾/背景负样本/images")
WEIGHTS = ROOT / "models" / "fire_smoke_v5_26s960_best.pt"
FPS = 25.0


def imread_u(p: Path):
    import cv2
    return cv2.imdecode(np.frombuffer(p.read_bytes(), np.uint8), cv2.IMREAD_COLOR)


def find_truck() -> Path:
    """找那张消防车（纯背景里最容易骗过模型的夜间强光场景）。"""
    files = sorted(p for p in BG.iterdir() if p.suffix.lower() in (".jpg", ".png"))
    same = [p for p in files if (im := imread_u(p)) is not None
            and im.shape[1] == 640 and im.shape[0] == 480]
    return same[1]


def flash_frames(img, n: int, hz: float = 5.0, depth: float = 0.4):
    """只让**高饱和的暖色高光**明暗跳变，模拟爆闪灯。

    为什么挑像素而不是整幅调亮度：整幅一起变会被流水线的"整幅同步变化补偿"
    直接扣掉（`frac_global`），那就不是在测闸了。
    """
    import cv2
    b, r = img[:, :, 0].astype(np.int16), img[:, :, 2].astype(np.int16)
    hot = (r > 140) & (r > b + 30)                       # 红色高光 = 警灯/尾灯
    hot = cv2.dilate(hot.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
    period = max(2, int(round(FPS / hz)))
    for i in range(n):
        on = (i % period) < (period / 2)
        f = (1.0 + depth) if on else (1.0 - depth)
        out = img.astype(np.float32)
        out[hot] = np.clip(out[hot] * f, 0, 255)
        yield out.astype(np.uint8)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=75)
    ap.add_argument("--hz", type=float, default=5.0)
    ap.add_argument("--depth", type=float, default=0.4)
    ap.add_argument("--no-gate", action="store_true")
    a = ap.parse_args()

    from src.infer import Detector
    p = find_truck()
    img = imread_u(p)
    print(f"素材：{p.name}  {img.shape[1]}x{img.shape[0]}")
    print(f"模拟爆闪 {a.hz:.0f}Hz  幅度 ±{a.depth * 100:.0f}%  "
          f"{a.frames} 帧 = {a.frames / FPS:.1f} 秒  闪烁闸{'关' if a.no_gate else '开'}\n")

    det = Detector(str(WEIGHTS), conf=0.20, imgsz=896, device=0)
    evs: list = []
    pipe = Pipeline(det, cfg={
        "detect_mode": "auto",
        "flicker_gate": not a.no_gate,
        "light_memory": False,
        "fast_smoke": False,
    }, arbiter=None)
    src = FixedCameraSource(CameraInfo(cam_id="flash", kind="fixed", fps=FPS),
                            frames=flash_frames(img, a.frames, a.hz, a.depth), warmup=0)
    pipe.run(src, max_frames=a.frames, quiet=True,
             on_event=lambda ev, _r: evs.append(ev))

    formal = [e for e in evs if e.level in ("warning", "alarm")]
    notice = [e for e in evs if e.level == "notice"]
    print(f"正式告警 {len(formal)} 条 / notice {len(notice)} 条  "
          f"静态拦截 {pipe.stats['static_light_blocked']} 帧")
    for e in formal[:3]:
        print(f"  · [{e.kind}] {e.level} t={e.t_alarm:.2f}s score={e.score:.2f} "
              f"| {e.reasons[0][:56]}")
    if formal:
        print("\n结论：会闪的假火穿过了闸 —— 只能靠 LLM 裁判或硬负样本重训，"
              "调阈值没用（会开始误杀真火）")
    else:
        print("\n结论：拦住了。闸的判据是「框内 ≥25% 像素在跳」，"
              "车灯在框里只占几个百分点，怎么闪都够不到 —— 担心不成立")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
