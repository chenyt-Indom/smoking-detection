"""区域记忆自测：同一盏路灯反复出现，不能反复花钱问 LLM。

三个必须成立的断言
------------------
1. 被判"灯"之后，**同一位置换了一条新轨迹**（现实里路灯天天被切成好几段）
   不再提交 LLM，只留一条提示。
2. 重启进程（新建 Pipeline 指向同一个记忆文件）依然记得 —— 否则每晚重新交学费。
3. 灯**外形变了**（这里用面积涨 3 倍模拟"灯被点着了 / 火烧大了"）必须立刻
   作废记忆、重新送审。这条是安全阀，不成立就绝不能上线。

跑法：
    python scripts/test_light_memory.py
"""
from __future__ import annotations

import sys
import tempfile
import time
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


def scene() -> np.ndarray:
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    base = 96 + 44 * (xx / W)[..., None] + np.array([8, 6, 4], np.float32)
    return np.clip(base, 0, 255).astype(np.uint8)


def frames_lamp(plan, n: int = 420):
    """plan: [(start, end, 半径)]，不在任何区间里的帧就是空场。"""
    base = scene()
    for i in range(n):
        img = base.copy()
        img = cv2.convertScaleAbs(img, alpha=1.0, beta=0.8 * np.sin(i / 90.0))
        for a, b, r in plan:
            if a <= i < b:
                cv2.ellipse(img, (CX, CY), (r, int(r * 0.7)), 0, 0, 360,
                            (238, 242, 248), -1)
        yield img


class Stub:
    """受控检测器：框由 plan 决定，专测记忆层，不掺检测噪声。"""

    def __init__(self, plan):
        self.plan = plan
        self.names = {0: "fire", 1: "smoke"}
        self.conf = 0.20
        self.i = 0

    def detect(self, image, sliced=False, **kw):
        i = self.i
        self.i += 1
        for a, b, r in self.plan:
            if a <= i < b:
                return [Detection(0, "fire", 0.82,
                                  (CX - r, CY - r * 0.7, CX + r, CY + r * 0.7))]
        return []


class Arrb:
    """假裁判：一律判"灯"。记下被提交了几次 —— 这就是"花了多少钱"。"""

    def __init__(self, delay: float = 0.2):
        from src.arbiter import ArbiterResult
        self.res = ArbiterResult("light", "low", 0.9, "假裁判：这是一盏灯", "llm", delay)
        self.delay = delay
        self.submitted = 0
        self.keys: list[str] = []
        self._q: dict[str, float] = {}
        self.stats = {"submitted": 0}

    def submit(self, key, frame, box, **kw):
        if key in self._q:
            return False, "同一目标已审过"
        self._q[key] = time.time() + self.delay
        self.submitted += 1
        self.stats["submitted"] += 1
        self.keys.append(key)
        return True, "已提交"

    def poll(self):
        now, out = time.time(), []
        for k in [k for k, due in self._q.items() if now >= due]:
            self._q.pop(k, None)
            out.append((k, self.res))
        return out

    def pending(self):
        return len(self._q)

    def close(self):
        pass


def run(plan, mem_path: str, cfg_extra: dict | None = None):
    arb = Arrb()
    evs: list = []
    cfg = {"flicker_gate": True, "light_memory": True,
           "light_memory_path": mem_path,
           "cooldown_sec": 0.01, "dedup_grow": 0.5,
           "llm_region_mute_sec": 5.0, "detect_mode": "full",
           "light_memory_notice_sec": 3600.0}
    cfg.update(cfg_extra or {})
    pipe = Pipeline(Stub(plan), cfg=cfg, arbiter=arb)
    src = FixedCameraSource(CameraInfo(cam_id="cam_lamp", kind="fixed", fps=FPS),
                            frames=frames_lamp(plan), warmup=WARM)
    pipe.run(src, max_frames=420, quiet=True,
             on_event=lambda ev, _r: evs.append(ev))
    return arb, pipe.stats, evs


def main() -> int:
    ok = True
    tmp = Path(tempfile.mkdtemp()) / "light_memory.json"

    # 灯亮 → 灭 → 再亮（第二条轨迹）。中间断 2 秒 > MAX_MISS_SEC，轨迹一定换了新的。
    plan = [(WARM, 160, 40), (210, 420, 40)]
    arb, st, evs = run(plan, str(tmp))
    print(f"[第一次出现] LLM 提交 {arb.submitted} 次  轨迹键 {arb.keys}")
    print(f"  记忆命中 {st['memory_hits']} 帧  llm_saved {st['llm_saved']}  "
          f"memory_reopened {st['memory_reopened']}")

    a = arb.submitted == 1
    print(f"  {'✅' if a else '❌'} 只问了 1 次（第二次出现换了轨迹也不再问）")
    ok &= a

    b = st["memory_hits"] > 0 and st["llm_saved"] > 0
    print(f"  {'✅' if b else '❌'} 第二次出现命中记忆（memory_hits={st['memory_hits']}）")
    ok &= b

    txt = tmp.read_text(encoding="utf-8") if tmp.is_file() else "{}"
    c = "light" in txt
    print(f"  {'✅' if c else '❌'} 记忆已落盘：{tmp}")
    ok &= c

    # 不该刷屏：第二次出现（t>8s）一条 notice 都不该有 —— 留痕按小时给，不按轨迹
    late = [e for e in evs if e.t_alarm > 8.0 and e.level == "notice"]
    g = len(late) == 0
    print(f"  {'✅' if g else '❌'} 第二次出现不再刷 notice（多余 {len(late)} 条）")
    ok &= g

    # 重启：新 Pipeline + 同一个记忆文件
    arb2, st2, _ = run([(WARM, 420, 40)], str(tmp))
    d = arb2.submitted == 0
    print(f"[重启后]  LLM 提交 {arb2.submitted} 次 {'✅' if d else '❌'}"
          f"（应为 0：记忆跨进程有效）")
    ok &= d

    # 安全阀 A：灯稍微变大（面积 ×1.69）—— 区域还匹配得上，但 `still_same`
    # 必须判"不是原来那东西"，把记忆作废并重新送审。
    arb3, st3, _ = run([(WARM, 420, 52)], str(tmp))
    e = st3["memory_reopened"] >= 1 and arb3.submitted >= 1
    print(f"[外形略变] LLM 提交 {arb3.submitted} 次  reopened={st3['memory_reopened']} "
          f"{'✅' if e else '❌'}（应≥1：面积涨 1.69 倍要重新判）")
    ok &= e

    # 安全阀 B：灯变成一大团火（面积 ×9）—— 连区域都匹配不上了，更该重新问
    arb4, st4, _ = run([(WARM, 420, 120)], str(tmp))
    f = arb4.submitted >= 1
    print(f"[外形突变] LLM 提交 {arb4.submitted} 次 {'✅' if f else '❌'}（应≥1）")
    ok &= f

    print("\n" + "=" * 58)
    print("结论：", "✅ 全部通过" if ok else "❌ 有不通过项，别上线")
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
