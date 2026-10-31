"""反光实测：水洼/玻璃/湿地面这类"亮斑会动"的假火，闸到底拦不拦得住。

为什么反光要单独测
------------------
普通灯光是**恒亮**的，闪烁闸一测就判静态。反光不一样：
  · 水面波纹 → 亮斑在**抖**，而且抖得不规则
  · 玻璃幕墙 / 车窗 → 亮斑**跟着车/云动**
两类都满足"框内像素在跳"，正是闸放行的条件。所以它比路灯危险。

脚本分两段
----------
A 筛场景：按"近白高光占比"给 640x480 背景图排序，取最像反光的一批跑流水线
  （`bright = V>235 且 S<60` —— 反光多是过曝的近白高光，比单纯"亮"更贴切）
B 让亮斑真的动：拿反光最强那张，给高光区叠加 2~3Hz 的波纹式扰动，
  闸开 / 闸关各跑一遍。**关闸对照不能省** —— 否则分不清"闸拦住了"和"模型根本没报"。

跑法：
    python scripts/eval_reflection.py --n 15
    python scripts/eval_reflection.py --n 15 --ripple-hz 3
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


def glare_ratio(img) -> float:
    """近白高光占比：过曝且低饱和 —— 反光/眩光的典型长相。"""
    import cv2
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    v, s = hsv[:, :, 2], hsv[:, :, 1]
    return float(((v > 235) & (s < 60)).mean())


def frames_static(img, n: int, rng):
    base = img.astype(np.int16)
    for _ in range(n):
        yield np.clip(base + rng.normal(0, 1.0, base.shape), 0, 255).astype(np.uint8)


def frames_ripple(img, n: int, hz: float, rng, box=None):
    """在**模型自己框出来的那块区域**叠一个随波纹游走的亮度扰动。

    为什么用检测框而不是"近白高光"：第一版按近白像素取掩码，结果海面日落那张
    高光只占 0.7%、扰动根本没落在框里，闸开/关都 0 条 —— 测了个寂寞。
    要问的是"这块反光如果在抖，闸放不放行"，那扰动就必须落在闸真正看的那块窗口上。

    为什么不整幅调亮度：整幅一起变会被流水线的"整幅同步变化补偿"扣掉。
    """
    import cv2
    h, w = img.shape[:2]
    if box is not None:
        mask = np.zeros((h, w), np.uint8)
        x1, y1, x2, y2 = [int(v) for v in box]
        cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)
    else:
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask = ((hsv[:, :, 2] > 235) & (hsv[:, :, 1] < 60)).astype(np.uint8)
        mask = cv2.dilate(mask, np.ones((7, 7), np.uint8))
        if mask.sum() < 200:
            mask = cv2.dilate(mask, np.ones((21, 21), np.uint8))
    m = mask.astype(bool)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    period = max(2, int(round(FPS / hz)))
    for i in range(n):
        phase = 2 * np.pi * (i % period) / period
        # 斜向波纹：空间上缓慢起伏 + 时间上 0~1 摆动，像水面反光在游
        wave = 0.5 + 0.5 * np.sin(6.0 * (xx / w) + 4.0 * (yy / h) + phase)
        gain = 0.55 + 0.9 * wave
        out = img.astype(np.float32)
        out[m] = np.clip(out[m] * gain[m][:, None], 0, 255)
        yield np.clip(out + rng.normal(0, 1.0, out.shape), 0, 255).astype(np.uint8)


def run(frames, n: int, gate: bool, cam: str):
    from src.infer import Detector
    det = Detector(str(WEIGHTS), conf=0.20, imgsz=896, device=0)
    evs: list = []
    pipe = Pipeline(det, cfg={
        "detect_mode": "auto",
        "flicker_gate": gate,
        "light_memory": False,
        "fast_smoke": False,
    }, arbiter=None)
    src = FixedCameraSource(CameraInfo(cam_id=cam, kind="fixed", fps=FPS),
                            frames=frames, warmup=0)
    pipe.run(src, max_frames=n, quiet=True,
             on_event=lambda ev, _r: evs.append(ev))
    return evs, pipe.stats


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=15, help="A 段取几个反光场景")
    ap.add_argument("--hold", type=int, default=60, help="B 段每个场景多少帧")
    ap.add_argument("--ripple-hz", type=float, default=2.5)
    ap.add_argument("--bg-dir", default=str(BG))
    ap.add_argument("--picks", default="", help="手工指定的场景序号，逗号分隔")
    ap.add_argument("--sheet", default="", help="只生成 97 张的联络表到该路径")
    a = ap.parse_args()

    files = sorted(p for p in Path(a.bg_dir).iterdir()
                   if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
    same = [p for p in files if (im := imread_u(p)) is not None
            and im.shape[1] == 640 and im.shape[0] == 480]
    if a.picks:
        idx = [int(x) for x in a.picks.split(",") if x.strip()]
        top = [(glare_ratio(imread_u(same[i])), same[i]) for i in idx]
        print(f"A 段：人工指定的 {len(top)} 个反光场景（联络表序号 {idx}）")
    else:
        scored = sorted(((glare_ratio(imread_u(p)), p) for p in same), reverse=True)
        top = scored[:a.n]
        print(f"A 段：640x480 背景 {len(same)} 张，取近白高光占比最高的 {len(top)} 张")
    print(f"  近白高光占比 {top[0][0] * 100:.1f}% ~ {top[-1][0] * 100:.1f}%\n")

    rng = np.random.default_rng(0)
    tot_alarm = tot_notice = 0
    scenes = set()
    for i, (g, p) in enumerate(top):
        im = imread_u(p)
        evs, st = run(frames_static(im, a.hold, rng), a.hold, True, f"g{i}")
        fa = [e for e in evs if e.level in ("warning", "alarm")]
        nt = [e for e in evs if e.level == "notice"]
        tot_alarm += len(fa)
        tot_notice += len(nt)
        if fa:
            scenes.add(i)
        flag = "⚠️ 报出正式告警" if fa else ("降级 notice" if nt else "无事件")
        print(f"  [{i:02d}] 高光 {g * 100:5.1f}%  {p.name[:28]:28s}  {flag}")
    print(f"\n  A 段合计：正式告警 {tot_alarm} 条 / notice {tot_notice} 条，"
          f"场景误报率 {len(scenes)}/{len(top)} = {len(scenes) / len(top) * 100:.0f}%")

    print(f"\nB 段：给高光区叠 {a.ripple_hz:.1f}Hz 波纹扰动（亮斑真的在动）")
    worst = top[0][1]
    im = imread_u(worst)
    from src.infer import Detector
    _d = Detector(str(WEIGHTS), conf=0.20, imgsz=896, device=0)
    _n = _d.names if isinstance(_d.names, dict) else dict(enumerate(_d.names))
    _hot = sorted([d for d in _d.detect(im, sliced=True, tile_grid=2, overlap=0.25)
                   if _n.get(d.cls_id, d.cls_name) == "fire"], key=lambda d: -d.conf)
    box = _hot[0].xyxy if _hot else None
    print(f"  素材 {worst.name}  近白高光 {glare_ratio(im) * 100:.1f}%  "
          + (f"模型火框 conf {_hot[0].conf:.2f}" if _hot else "模型没给框 → B 段无意义"))
    for gate in (True, False):
        evs, st = run(frames_ripple(im, a.hold * 2, a.ripple_hz, rng, box),
                      a.hold * 2, gate, f"rip_{int(gate)}")
        fa = [e for e in evs if e.level in ("warning", "alarm")]
        nt = [e for e in evs if e.level == "notice"]
        print(f"  闸{'开' if gate else '关'}：正式告警 {len(fa)} 条 / notice {len(nt)} 条  "
              f"静态拦截 {st['static_light_blocked']} 帧")
        for e in fa[:2]:
            print(f"    · {e.level} t={e.t_alarm:.2f}s score={e.score:.2f} "
                  f"| {e.reasons[0][:50]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
