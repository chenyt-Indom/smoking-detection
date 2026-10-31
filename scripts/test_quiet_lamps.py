"""固定光源"别一直上报"的自测 —— 专测**没有 LLM** 的那条路。

为什么单独测这个
----------------
`test_light_memory.py` 测的是"记忆命中就不花钱问 LLM"，但它挂了一个假裁判。
现实里最常见的情形恰恰是**裁判不可用**（没配 model / 没网 / 额度用完），
这时候：

  · 记忆永远学不到东西 —— 它只在裁判判"降级"时才落盘；
  · 静态灯靠闪烁闸压成 notice，但降级留痕那一段是按**轨迹 ID** 记账的。

路灯一夜被切成几十段是常态（遮挡、失配、conf 抖动都会换 ID），
按 ID 记账 = 每段一张卡片。用户说的"不要多次上报"就落在这条路上。

断言
----
1. 8 条轨迹的同一盏灯，notice 卡片 ≤ 1 张。
2. 仍然留了痕（不能一条都没有 —— 降级不是排除，屏幕上必须看得见）。
3. 正式告警 0 条（灯不是火）。

跑法：
    python scripts/test_quiet_lamps.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.infer import Detection                                   # noqa: E402
from src.pipeline import Pipeline                                 # noqa: E402
from src.sources.base import CameraInfo                           # noqa: E402
from src.sources.fixed import FixedCameraSource                   # noqa: E402

W, H = 960, 540
FPS = 25.0
WARM = 30
CX, CY = 420, 250

# 8 段：每段 1 秒（25 帧），中间断 1.6 秒（40 帧）——
# MAX_MISS_SEC 是 0.8 秒，1.6 秒的断档必然把轨迹切成新的 ID。
SEG, GAP, NSEG = 25, 40, 8
PLAN = [(WARM + i * (SEG + GAP), WARM + i * (SEG + GAP) + SEG, 40)
        for i in range(NSEG)]
TOTAL = WARM + NSEG * SEG + (NSEG - 1) * GAP + 5


def scene() -> np.ndarray:
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    base = 96 + 44 * (xx / W)[..., None] + np.array([8, 6, 4], np.float32)
    return np.clip(base, 0, 255).astype(np.uint8)


def frames_lamp():
    """恒亮的暖白亮斑 —— 和火在单帧上长得一模一样，唯一差别是不抖。"""
    base = scene()
    for i in range(TOTAL):
        img = base.copy()
        for a, b, r in PLAN:
            if a <= i < b:
                cv2.ellipse(img, (CX, CY), (r, int(r * 0.7)), 0, 0, 360,
                            (238, 242, 248), -1)
        yield img


class Stub:
    def __init__(self):
        self.names = {0: "fire", 1: "smoke"}
        self.conf = 0.20
        self.i = 0
        self.tids_boxes = []

    def detect(self, image, sliced=False, **kw):
        i = self.i
        self.i += 1
        for a, b, r in PLAN:
            if a <= i < b:
                return [Detection(0, "fire", 0.82,
                                  (CX - r, CY - r * 0.7, CX + r, CY + r * 0.7))]
        return []


def main() -> int:
    ok = True
    evs: list = []
    seen_tracks: set = set()

    pipe = Pipeline(Stub(), cfg={
        "flicker_gate": True,
        "light_memory": True,          # 开着也没用：没有裁判就学不到东西
        "light_memory_path": "",       # 空路径 → 不落盘，别污染现场记忆
        "detect_mode": "full",
    }, arbiter=None)                   # ← 关键：没有 LLM

    src = FixedCameraSource(CameraInfo(cam_id="cam_lamp", kind="fixed", fps=FPS),
                            frames=frames_lamp(), warmup=WARM)
    # 记下所有出现过的轨迹 ID，证明轨迹确实被切成了多段
    orig = pipe.process

    def spy(pkt):
        r = orig(pkt)
        for t in r.tracks:
            seen_tracks.add(t.tid)
        return r

    pipe.process = spy
    pipe.run(src, max_frames=TOTAL, quiet=True,
             on_event=lambda ev, _r: evs.append(ev))

    st = pipe.stats
    notices = [e for e in evs if e.level == "notice"]
    alarms = [e for e in evs if e.level in ("warning", "alarm")]

    print(f"轨迹数 {len(seen_tracks)}  静态灯拦截 {st['static_light_blocked']} 帧")
    print(f"notice {len(notices)} 张  正式告警 {len(alarms)} 条")

    a = len(seen_tracks) >= 4
    print(f"  {'✅' if a else '❌'} 轨迹确实被切成了多段（{len(seen_tracks)} 段）—— 前提成立")
    ok &= a

    b = len(notices) <= 1
    print(f"  {'✅' if b else '❌'} {len(seen_tracks)} 段轨迹只推 1 张卡片（实际 {len(notices)} 张）")
    ok &= b

    c = len(notices) >= 1
    print(f"  {'✅' if c else '❌'} 仍留了痕（降级不是排除，屏幕上要看得见）")
    ok &= c

    d = len(alarms) == 0
    print(f"  {'✅' if d else '❌'} 灯没有变成正式告警（{len(alarms)} 条）")
    ok &= d

    print("\n" + "=" * 58)
    print("结论：", "✅ 全部通过" if ok else "❌ 有不通过项")
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
