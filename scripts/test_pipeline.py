"""M4 端到端验证：把五级流水线真正跑起来，量「确认延迟」和「蒸汽抑制」。

这个测试能证明什么、不能证明什么（先说清楚，免得后面对不上）
------------------------------------------------------------
能证明：
  · capture → detect → track → temporal → event 五级真的连通了，事件能出来
  · 火走快路径、烟走时序路径，两条路的分支逻辑对
  · 「旁边有火」这个先验把白烟从 uncertain 推到 smoke（有 A/B 对照）
  · 只有烟没有火时，**不会**在 2 秒内被误确认（trend_min_duration 那条门）
  · 快速膨胀 + 变糊的羽流判成 steam 并被丢掉（蒸汽误报防线）

不能证明（要现场数据才行）：
  · 真实误报率 —— 背景是合成的，墙不会自己冒烟
  · 真实漏报率 —— 素材是从 test 集抠的真火，检测器本来就认得
所以这里只看**延迟**和**逻辑正确性**，不拿它当精度验收。

关于 steam 场景为什么用假检测器
--------------------------------
先试过用真 YOLO 跑合成蒸汽：220 帧只出了 32 个框、0 条轨迹 —— 检测器根本不认
合成出来的羽流，场景全程空转，判定层一次都没被调用，等于什么都没测。
所以改成「真实图像 + 受控检测框」：图像仍是真的（亮度/边缘密度由真像素算出来），
只有"框在哪、多大"由代码给定，这样 temporal 的特征是真实的，判定逻辑能被确定性地打到。

用法
----
    python scripts/test_pipeline.py                     # 跑全部场景
    python scripts/test_pipeline.py --scene fire        # 只跑火
    python scripts/test_pipeline.py --sliced 0          # 关切片对比
    python scripts/test_pipeline.py --scene smoke_only --n-mul 2.4   # 拉长到 30s，测阴燃
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import NamedTuple

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.infer import Detection, Detector                          # noqa: E402
from src.pipeline import Pipeline                                  # noqa: E402
from src.sources.base import CameraInfo                            # noqa: E402
from src.sources.fixed import FixedCameraSource                    # noqa: E402


class FakeArbiter:
    """固定答案的假裁判：验证 LLM 裁决真的会改变事件级别，且不卡主线程。

    不联网，所以能把"裁判说蒸汽"和"裁判说火"两个分支都确定性地跑到。
    """

    def __init__(self, label: str, risk: str, conf: float, delay: float = 0.3):
        from src.arbiter import ArbiterResult
        self.res = ArbiterResult(label, risk, conf, f"假裁判：{label}", "llm", delay)
        self.delay = delay
        self._q: dict[str, float] = {}
        self._done: dict[str, object] = {}
        self.stats = {"submitted": 0, "llm": 0, "rule": 0, "skipped": 0,
                      "timeout": 0, "error": 0}

    def submit(self, key, frame, box, **kw) -> tuple[bool, str]:
        if key in self._q or key in self._done:
            return False, "同一目标已审过"
        self._q[key] = time.time() + self.delay
        self.stats["submitted"] += 1
        return True, "已提交"

    def poll(self):
        now = time.time()
        ready = [k for k, due in self._q.items() if now >= due]
        out = []
        for k in ready:
            self._q.pop(k, None)
            self._done[k] = self.res
            out.append((k, self.res))
        return out

    def pending(self) -> int:
        return len(self._q)

    def close(self) -> None:
        pass


def make_arbiter(kind: str):
    """--llm 参数 → 裁判实例。"""
    if not kind or kind == "off":
        return None
    if kind == "dry":
        from src.arbiter import LlmArbiter
        cfg = {"dry_run": True, "enabled": True}
        return LlmArbiter(cfg=cfg, api_key="dry")
    if kind == "dismiss":       # 判成蒸汽 → 应该压掉，0 告警
        return FakeArbiter("steam", "low", 0.85)
    if kind == "alarm":         # 判成火 → 应该升级成 alarm
        return FakeArbiter("fire", "high", 0.9)
    if kind == "uncertain":     # 拿不准 → 应该降为 notice（推给人看）
        return FakeArbiter("uncertain", "low", 0.2)
    raise SystemExit(f"未知 --llm 取值：{kind}")

WEIGHTS = ROOT / "models" / "fire_smoke_v5_26s960_best.pt"
DATA = Path("E:/datasets/fire_smoke_v5/test")
CACHE = ROOT / "runs" / "pipeline_patches"
SCENE_W, SCENE_H = 960, 540
FPS = 25.0
WARMUP_FRAMES = 30          # 前 30 帧空场，给背景建模预热

# 场景清单。expect=None 表示只观测不下断言；need 是必须达到的统计下限，
# 不设这个的话"没测到判定层"也会被当成通过（第一版就踩了这个坑）。
SCENES = {
    "fire": dict(desc="火（快路径）", expect={"fire"}),
    "fire_smoke": dict(desc="火+烟（火源先验）", expect={"fire", "smoke"}),
    "smoke_only": dict(desc="只有烟无火（对照）", expect=None),
    "steam": dict(desc="蒸汽（负样本，应 0 告警）", expect=set(),
                  need={"steam_suppressed": 1}),
    "lamp": dict(desc="车位灯（恒定亮斑，应 0 正式告警）", expect=set(),
                 need={"static_light_blocked": 1}),
}


# ---------------------------------------------------------------- 素材
def _parse_labels(path: Path) -> list[tuple[int, float, float, float, float]]:
    out = []
    try:
        for ln in path.read_text(encoding="utf-8", errors="ignore").splitlines():
            p = ln.split()
            if len(p) >= 5:
                out.append((int(float(p[0])), *map(float, p[1:5])))
    except OSError:
        pass
    return out


class Sprite(NamedTuple):
    """一块合成素材：图像 + **目标物体在素材里的位置** + 来源标注名。

    为什么要把物体位置一路带下来：素材是带上下文裁的（`_crop` 的 margin 让边长
    变 3 倍）。贴的时候若按"整块素材"定尺寸，标称 1.2% 的火会实打实占掉 11%
    画面 —— 尺度全错，合成场景就不再是它声称的那个场景。
    """
    img: np.ndarray
    obj: tuple
    name: str


# 复验点：素材必须**在它真正会被贴到的位置上**仍然被认成对的那一类。
# 原先只有一个固定复验点（seed 7 的墙、画面下方），结果挑出来的烟素材贴到
# seed 11 的墙、画面上部之后，检测器把它判成了 **fire** —— 整个 fire_smoke
# 场景只剩火、烟一条轨迹都没有。素材"在原图里是烟"不等于"贴上去还是烟"。
PROBE_SPOTS = {
    0: [(7, 0.30, 0.62), (11, 0.36, 0.55)],    # 火：fire 场景 / fire_smoke 场景
    1: [(11, 0.36, 0.36), (23, 0.48, 0.44)],   # 烟：fire_smoke·smoke_only / 蒸汽
}


def pick_patch(det: Detector, cls_id: int, target_pct: float, other_cls: int,
               margin: float = 0.25) -> Sprite:
    """从 test 集里挑一块**检测器确实认得出**的真火/真烟，当合成素材。

    不直接信标注：标注可能偏，也可能框太糊。所以候选按面积接近度排序后，
    真的拿检测器跑一遍整图，conf 够高才采用 —— 否则测试的前提就是假的。

    `margin` 是四周各外扩多少个框宽。原来取 1.0（边长变 3 倍），那时粘贴按
    "整块素材"缩放到原尺寸，多出来的上下文看不出问题。改成按物体定尺度之后，
    上下文也跟着放大 —— 火素材的实际着色范围比火本身宽一倍，把上方的烟盖掉了：
    同一个烟素材从 smoke 0.93 掉到 0.22，烟永久确认不了。
    """
    CACHE.mkdir(parents=True, exist_ok=True)
    tag = f"cls{cls_id}_pct{target_pct}_m{margin}"
    # 缓存随裁剪参数走：margin 变了素材就变了，沿用旧缓存会拿到旧尺度的图
    npz = CACHE / f"patch_v4_{tag}.npz"
    if npz.exists():
        z = np.load(npz, allow_pickle=True)
        return Sprite(z["patch"], tuple(z["obj"].tolist()), str(z["name"]))

    img_dir, lbl_dir = DATA / "images", DATA / "labels"
    cands = []
    for lf in sorted(os.listdir(lbl_dir)):
        if not lf.endswith(".txt"):
            continue
        rows = _parse_labels(lbl_dir / lf)
        if not rows or any(r[0] == other_cls for r in rows):
            continue                      # 素材里别混进另一类，否则分不清是谁触发的
        for r in rows:
            if r[0] == cls_id:
                cands.append((abs(r[3] * r[4] - target_pct), lf, r))
    cands.sort(key=lambda x: x[0])

    for _, lf, r in cands[:40]:
        img_p = img_dir / (lf[:-4] + ".jpg")
        if not img_p.exists():
            continue
        img = cv2.imread(str(img_p))
        if img is None:
            continue
        dets = det.detect(img, sliced=False)
        hit = [d for d in dets if d.cls_id == cls_id and d.conf >= 0.5]
        if not hit:
            continue
        # 素材里不能同时冒出另一类：拿"火+烟"的整团裁下来当火，烟会跟着贴进场景，
        # 让 fire 场景凭空多一条 smoke 告警（实测过），测出来的东西就没意义了。
        if any(d.cls_id == other_cls and d.conf > 0.35 for d in dets):
            continue
        d = max(hit, key=lambda x: x.conf)
        patch, rect = _crop(img, d.xyxy, margin)
        if patch is None or min(patch.shape[:2]) < 40:
            continue
        # 物体在素材里的坐标（裁剪时整体平移了 rect[0]/rect[1]），再夹进素材边界
        ph, pw = patch.shape[:2]
        obj = (max(0.0, d.xyxy[0] - rect[0]), max(0.0, d.xyxy[1] - rect[1]),
               min(float(pw), d.xyxy[2] - rect[0]), min(float(ph), d.xyxy[3] - rect[1]))
        if obj[2] - obj[0] < 8 or obj[3] - obj[1] < 8:
            continue
        # 素材本身得"亮得起来"。第一版没这道筛选，挑到一块平均亮度只有 11.9 的
        # 夜间火（黑背景占 95%），贴到画面上就是一条黑带 + 几个火点 ——
        # 既不像火，闪烁闸也测不出东西（噪声根本没地方加）。
        if float(cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY).mean()) < 45:
            continue
        # 最后一道：**按最终尺度、贴到它真正会去的位置上再验一次**。
        # 原图里干净不等于贴上去也干净 —— 换了背景之后检测器可能把火焰边缘
        # 认成烟，也可能把烟认成火（实测过，后者的后果更隐蔽：fire_smoke
        # 场景里烟一条轨迹都没有，测试却"跑通了"）。
        ok = True
        for seed, fx, fy in PROBE_SPOTS[cls_id]:
            probe = make_scene(random.Random(seed))
            paste_obj(probe, patch, obj, SCENE_W * fx, SCENE_H * fy, target_pct)
            vd = det.detect(probe, sliced=False)
            if any(d.cls_id == other_cls and d.conf > 0.35 for d in vd):
                ok = False
                break
            if not any(d.cls_id == cls_id and d.conf >= 0.45 for d in vd):
                ok = False
                break
        if not ok:
            continue
        np.savez_compressed(npz, patch=patch, obj=np.array(obj), name=lf)
        print(f"  素材 cls={cls_id} ← {lf}  conf={d.conf:.2f}  "
              f"素材 {pw}x{ph}  物体 {obj[2]-obj[0]:.0f}x{obj[3]-obj[1]:.0f}")
        return Sprite(patch, obj, lf)
    raise RuntimeError(f"没找到可用的 cls={cls_id} 素材")


def _crop(img, box, margin):
    """按框裁一块带上下文的素材。返回 (素材, 裁剪矩形)。

    裁剪矩形要一起返回 —— 后面要用它把"物体在素材里的位置"算出来，
    否则没法按物体（而不是整块素材）定粘贴尺寸。
    """
    h, w = img.shape[:2]
    x1, y1, x2, y2 = box
    bw, bh = x2 - x1, y2 - y1
    cx1 = int(max(0, x1 - margin * bw)); cx2 = int(min(w, x2 + margin * bw))
    cy1 = int(max(0, y1 - margin * bh)); cy2 = int(min(h, y2 + margin * bh))
    if cx2 <= cx1 or cy2 <= cy1:
        return None, None
    return img[cy1:cy2, cx1:cx2].copy(), (cx1, cy1, cx2, cy2)


# ---------------------------------------------------------------- 合成
# 配色必须是**冷色/中性**（B ≥ G ≥ R）。
# 第一版用三个通道独立随机，结果生成了一块橙色矩形，被 YOLO 判成火 conf=0.87
# —— 比场景里真的那簇火还高，直接把"火场景报了 2 次"搞成假阳性。
# 火焰色相在 HSV 的 h≤28，需要 R 明显大于 G，卡住 R≤G 就不会再撞上。
PALETTE_BGR = [
    (162, 162, 162), (172, 150, 130), (140, 162, 176), (110, 122, 136),
    (96, 106, 116), (186, 190, 194), (128, 138, 150), (74, 82, 90),
]


def make_scene(rng: random.Random) -> np.ndarray:
    """一面"墙"：渐变 + 几块矩形（门窗/柜子）+ 噪声。有结构才能让切片有意义。"""
    yy, xx = np.mgrid[0:SCENE_H, 0:SCENE_W].astype(np.float32)
    base = 96 + 44 * (xx / SCENE_W)[..., None] + np.array([8, 6, 4], np.float32)
    base = np.clip(base + 0 * yy[..., None], 0, 255).astype(np.float32)

    for _ in range(6):
        w = rng.randint(60, 220); h = rng.randint(50, 180)
        x = rng.randint(0, SCENE_W - w); y = rng.randint(0, SCENE_H - h)
        col = np.array(rng.choice(PALETTE_BGR), np.float32)
        base[y:y + h, x:x + w] = base[y:y + h, x:x + w] * 0.35 + col * 0.65

    base += np.random.normal(0, 3.0, base.shape).astype(np.float32)
    return np.clip(base, 0, 255).astype(np.uint8)


def paste(canvas: np.ndarray, patch: np.ndarray, cx: float, cy: float,
          scale: float, blur: float = 0.0, feather: float = 0.14):
    """把素材软边贴到画布上。硬边会像贴纸，检测器容易被边带偏。

    返回实际贴上去的矩形 (x1,y1,x2,y2)，蒸汽场景要靠它给出"检测框"。
    """
    ph, pw = patch.shape[:2]
    nw, nh = max(8, int(pw * scale)), max(8, int(ph * scale))
    p = cv2.resize(patch, (nw, nh),
                   interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
    if blur > 0.2:
        k = int(blur) * 2 + 1
        p = cv2.GaussianBlur(p, (k, k), 0)

    x1 = int(round(cx - nw / 2)); y1 = int(round(cy - nh / 2))
    x1 = max(0, min(SCENE_W - nw, x1)); y1 = max(0, min(SCENE_H - nh, y1))
    roi = canvas[y1:y1 + nh, x1:x1 + nw]
    if roi.shape[:2] != (nh, nw):
        return None

    a = np.ones((nh, nw), np.float32)
    k = max(3, int(min(nh, nw) * feather))
    ramp = np.linspace(0.0, 1.0, k, dtype=np.float32)
    a[:k, :] *= ramp[:, None]; a[-k:, :] *= ramp[::-1, None]
    a[:, :k] *= ramp[None, :]; a[:, -k:] *= ramp[::-1][None, :]
    a3 = a[..., None]
    roi[:] = (roi.astype(np.float32) * (1 - a3) + p.astype(np.float32) * a3).astype(np.uint8)
    return (x1, y1, x1 + nw, y1 + nh)


def obj_scale(patch: np.ndarray, obj: tuple, target_pct: float) -> float:
    """算出"让物体占画面 target_pct"所需的粘贴缩放比。"""
    x1, y1, x2, y2 = obj
    ow, oh = max(1.0, x2 - x1), max(1.0, y2 - y1)
    scale = (target_pct * SCENE_W * SCENE_H / (ow * oh)) ** 0.5
    ph, pw = patch.shape[:2]
    # 整块素材不能超出画布，否则 paste() 会因为 roi 尺寸对不上直接放弃
    return max(0.05, min(scale, SCENE_W * 0.95 / pw, SCENE_H * 0.95 / ph))


def paste_obj(canvas: np.ndarray, patch: np.ndarray, obj: tuple,
              cx: float, cy: float, target_pct: float, blur: float = 0.0):
    """把素材贴到画布上，并让**目标物体本身**占到画面的 target_pct。

    为什么不直接 `paste(patch, scale=1.0)`：素材是带上下文裁出来的，`_crop` 的
    margin 让边长变成 3 倍。标称 1.2% 的火贴上去实际占 11%；烟更夸张 —— 挑中的
    检测框几乎盖满原图，整块贴出来有 27% 画面，把旁边的火整个盖住了。
    `fire_smoke` 场景因此一次都没检出火、0 告警，根因就在这。

    返回**物体**在画布上的矩形（不是整块素材的），合成时要用它对位。
    """
    x1, y1, x2, y2 = obj
    ow, oh = max(1.0, x2 - x1), max(1.0, y2 - y1)
    scale = obj_scale(patch, obj, target_pct)
    ph, pw = patch.shape[:2]
    # paste() 是按**素材中心**对齐的，而物体不在素材正中，所以补一个偏移，
    # 让物体中心（而不是素材中心）落在 (cx, cy)
    rect = paste(canvas, patch,
                 cx + (pw / 2 - (x1 + x2) / 2) * scale,
                 cy + (ph / 2 - (y1 + y2) / 2) * scale,
                 scale, blur)
    if rect is None:
        return None
    # rect 是**素材**的左上角 → 物体左上角 = 素材左上角 + 物体在素材内的偏移×scale
    ox = rect[0] + x1 * scale
    oy = rect[1] + y1 * scale
    return (ox, oy, ox + ow * scale, oy + oh * scale)


def gen_fire(fire: Sprite, n: int):
    """场景1：先空场，第 WARMUP 帧起一小簇火固定在那儿（模拟垃圾桶起火）。"""
    rng = random.Random(7)
    scene = make_scene(rng)
    for i in range(n):
        img = scene.copy()
        img = cv2.convertScaleAbs(img, alpha=1.0, beta=1.2 * np.sin(i / 60.0))  # 光照慢漂
        if i >= WARMUP_FRAMES:
            jitter = 1.0 + 0.03 * np.sin(i / 3.0)          # 火在跳，尺寸微抖
            paste_obj(img, fire.img, fire.obj,
                      SCENE_W * 0.30, SCENE_H * 0.62, 0.012 * jitter)
        yield img


def gen_smoke(fire: Sprite | None, smoke: Sprite, n: int, with_fire: bool):
    """场景2/3：烟（可带火）。背景与随机种子完全一致，只有"有没有火"这一个变量。

    两组的差值，就是「旁边有火」这个先验的真实贡献 —— 这是本项目最核心的接线：
    白烟和蒸汽同色同形，单帧分不开，能靠的先验就剩"旁边有没有明火"。

    粘贴顺序有讲究：**先烟后火**。反过来的话烟柱会把底部的火盖掉
    （烟素材带了很多上下文，尺寸不小），火一次都检不出来，
    这个场景就测不出"火+烟"了。物理上也该是火在前、烟在火上方升起。
    """
    rng = random.Random(11)
    scene = make_scene(rng)
    for i in range(n):
        img = scene.copy()
        img = cv2.convertScaleAbs(img, alpha=1.0, beta=1.2 * np.sin(i / 60.0))
        if i >= WARMUP_FRAMES:
            grow = 1.0 + 0.10 * min(1.0, (i - WARMUP_FRAMES) / 100.0)   # 缓慢铺开
            srect = paste_obj(img, smoke.img, smoke.obj,
                              SCENE_W * 0.36, SCENE_H * 0.36, 0.02 * grow)
            if with_fire:
                # 火摆在烟柱**正下方并留一道缝**。放进去（哪怕只压住一点）都会
                # 把烟下半截吃掉：实测火盖上去之后，同一个烟素材从 smoke 0.93
                # 变成 fire 0.88，整条烟轨迹消失。
                # 位置仍落在「火源先验」的外扩范围里（fire_nearby_grow=0.6，
                # 烟框往下再多算 60% 高，够宽松）。
                if srect is None:
                    fx, fy = SCENE_W * 0.36, SCENE_H * 0.62
                else:
                    fs = obj_scale(fire.img, fire.obj, 0.012)
                    fh = (fire.obj[3] - fire.obj[1]) * fs
                    fx = (srect[0] + srect[2]) / 2
                    fy = srect[3] + 0.10 * (srect[3] - srect[1]) + fh / 2
                paste_obj(img, fire.img, fire.obj, fx, fy, 0.012)
        yield img


# ── 蒸汽：图像是真的，检测框是给定的 ─────────────────────────────
STEAM_SCALE = (0.25, 1.20)      # 面积比 (1.2/0.25)²≈23 倍 → 对数增长 ~0.7/s


class StubDetector:
    """受控假检测器。只用于确定性地打判定层，不参与任何精度评估。"""

    def __init__(self, box_fn, cls_id: int = 1, conf: float = 0.80):
        self.box_fn = box_fn
        self.names = {0: "fire", 1: "smoke"}
        self.conf = 0.20
        self.cls_id = cls_id
        self.fixed_conf = conf
        self.i = 0

    def detect(self, image, sliced=False, **kw) -> list[Detection]:
        b = self.box_fn(self.i)
        self.i += 1
        if b is None:
            return []
        name = self.names[self.cls_id]
        return [Detection(self.cls_id, name, self.fixed_conf, b)]


def steam_scale(i: int, n: int) -> float:
    k = min(1.0, max(0.0, (i - WARMUP_FRAMES) / max(1, n - WARMUP_FRAMES)))
    a, b = STEAM_SCALE
    return a + (b - a) * k


def gen_steam(smoke_patch, n: int):
    """蒸汽：几秒内膨胀 + 越来越糊。期望 0 告警且 steam_suppressed>0。"""
    rng = random.Random(23)
    scene = make_scene(rng)
    for i in range(n):
        img = scene.copy()
        img = cv2.convertScaleAbs(img, alpha=1.0, beta=1.2 * np.sin(i / 60.0))
        if i >= WARMUP_FRAMES:
            s = steam_scale(i, n)
            blur = 0.8 + 3.2 * min(1.0, (i - WARMUP_FRAMES) / max(1, n - WARMUP_FRAMES))
            paste(img, smoke_patch, SCENE_W * 0.48, SCENE_H * 0.44, s, blur)
        yield img


def steam_box(i: int, n: int, patch_wh: tuple[int, int]):
    """让"检测框"跟着合成的羽流一起长，这样面积增长是真的（由图像驱动）。"""
    if i < WARMUP_FRAMES:
        return None
    pw, ph = patch_wh
    s = steam_scale(i, n)
    cx, cy = SCENE_W * 0.48, SCENE_H * 0.44
    half_w, half_h = pw * s / 2, ph * s / 2
    return (cx - half_w, cy - half_h, cx + half_w, cy + half_h)


# ── 灯光：图像是真的，检测框给定，光斑恒定不闪 ──────────────────
def gen_lamp(n: int):
    """夜间停车场：一块恒定的亮斑（车位灯 / 路灯 / 招牌反光）。

    YOLO 在单帧上确实会把这种亮斑认成 fire（实测合成橙色亮斑拿到 conf 0.87，
    比真火还高），所以这里直接用受控检测器给框，专测「闪烁闸」这一层。

    关键：光斑**完全不闪**。真火实测框内 65%~92% 的像素逐帧在跳，
    这里应该是 0 —— 这就是闸要拦的东西。
    """
    rng = random.Random(31)
    scene = make_scene(rng)
    scene = np.clip(scene.astype(np.float32) * 0.35, 0, 255).astype(np.uint8)   # 夜间
    for i in range(n):
        img = scene.copy()
        # 只留极轻的整幅漂移（自动曝光），光斑本身恒定
        img = cv2.convertScaleAbs(img, alpha=1.0, beta=0.8 * np.sin(i / 90.0))
        if i >= WARMUP_FRAMES:
            cv2.ellipse(img, (int(SCENE_W * 0.42), int(SCENE_H * 0.46)),
                        (48, 34), 0, 0, 360, (238, 242, 248), -1)
        yield img


def lamp_box(i: int):
    if i < WARMUP_FRAMES:
        return None
    cx, cy = SCENE_W * 0.42, SCENE_H * 0.46
    return (cx - 48, cy - 34, cx + 48, cy + 34)


# ---------------------------------------------------------------- 先验自检
def check_fire_nearby() -> bool:
    """先验逻辑的单元检查。这条错了，白烟和蒸汽就彻底分不开。"""
    from src.pipeline import DEFAULTS

    p = Pipeline.__new__(Pipeline)
    p.cfg = dict(DEFAULTS)
    smoke = (100, 100, 300, 300)

    def fire(x1, y1, x2, y2):
        return Detection(0, "fire", 0.6, (x1, y1, x2, y2))

    cases = [
        ("火在烟框内", fire(150, 150, 200, 200), True),
        ("火紧挨烟下方", fire(180, 310, 230, 360), True),
        ("火在画面另一头", fire(800, 700, 860, 760), False),
    ]
    ok = True
    for name, d, want in cases:
        got, _ = p._fire_nearby(smoke, [d])
        print(f"   先验 {'✅' if got == want else '❌'} {name}: 期望 {want} / 实得 {got}")
        ok &= got == want
    return ok


# ---------------------------------------------------------------- 跑
def run_scene(name: str, det: Detector, args) -> dict:
    spec = SCENES[name]
    expect = spec["expect"]
    n = 320
    stub = None
    # 裁判的答案会改变应得结果，期望值要跟着改，否则测出来的"失败"是假的
    if args.llm in ("dismiss", "uncertain"):
        expect = set()          # 判蒸汽 → 压掉；拿不准 → 只发 notice（不算告警）

    if name == "fire":
        fire = pick_patch(det, cls_id=0, target_pct=0.012, other_cls=1)
        n, frames = 150, gen_fire(fire, 150)
    elif name == "fire_smoke":
        fire = pick_patch(det, cls_id=0, target_pct=0.012, other_cls=1)
        smoke = pick_patch(det, cls_id=1, target_pct=0.02, other_cls=0)
        frames = gen_smoke(fire, smoke, n, with_fire=True)
    elif name == "smoke_only":
        smoke = pick_patch(det, cls_id=1, target_pct=0.02, other_cls=0)
        frames = gen_smoke(None, smoke, n, with_fire=False)
    else:
        smoke = pick_patch(det, cls_id=1, target_pct=0.02, other_cls=0)
        n = 160
        pwh = (smoke.img.shape[1], smoke.img.shape[0])
        if name == "steam":
            frames = gen_steam(smoke.img, n)
            stub = StubDetector(lambda i: steam_box(i, n, pwh))
        else:                                       # lamp：光斑恒定，给定检测框
            frames = gen_lamp(n)
            stub = StubDetector(lamp_box, cls_id=0, conf=0.82)

    n = int(n * args.n_mul)
    if args.n_mul != 1.0:                     # 拉长时重新生成，保持时间轴一致
        if name == "smoke_only":
            frames = gen_smoke(None, smoke, n, with_fire=False)
        elif name == "fire_smoke":
            frames = gen_smoke(fire, smoke, n, with_fire=True)
        elif name == "steam":
            frames = gen_steam(smoke.img, n)
            stub = StubDetector(lambda i: steam_box(i, n, pwh))
        elif name == "lamp":
            frames = gen_lamp(n)
            stub = StubDetector(lamp_box, cls_id=0, conf=0.82)

        else:
            frames = gen_fire(fire, n)

    info = CameraInfo(cam_id=f"sim_{name}", kind="fixed", fps=FPS)
    src = FixedCameraSource(info, frames=frames, warmup=WARMUP_FRAMES)
    arb = make_arbiter(args.llm)
    # 闪烁闸在合成场景里关掉：贴上去的"火"是一张静止贴图（物理上就不该闪），
    # 闸判它 static 完全正确，开着测不出任何东西，只会把"合成数据不真实"
    # 显示成"测试失败"。真实火能不能过闸，由 scripts/verify_flicker_gate.py
    # 在 506 张**逐帧连续**的真火视频帧上单独验证。
    gate = name not in ("fire", "fire_smoke")
    pipe = Pipeline(stub or det, sliced=bool(args.sliced) and stub is None, tile_grid=2,
                    cfg={"fast_smoke": bool(args.fast), "flicker_gate": gate}, arbiter=arb)

    events: list = []
    trace: list = []

    def _hook(ev, r):
        events.append(ev)
        if args.trace:
            for tid, v in r.verdicts.items():
                if v.label != "uncertain" or (v.features and v.features.duration > 4):
                    trace.append({"t": round(r.ts, 1), "tid": tid, "label": v.label,
                                  "score": v.score, "nearby": r.nearby.get(tid),
                                  "dur": round(v.features.duration, 1) if v.features else 0,
                                  "reasons": v.reasons})

    t0 = time.time()
    res = pipe.run(src, on_event=_hook, max_frames=n, every=args.every)
    el = time.time() - t0
    stats = res["stats"]

    if args.save:
        for ev in events:
            Pipeline.save_event(ev, ROOT / "runs" / "pipeline_events")

    got = {e.kind for e in events if e.level in ("warning", "alarm")}
    checks = {"事件种类": got == expect if expect is not None else None}
    for k, v in (spec.get("need") or {}).items():
        checks[f"{k}≥{v}"] = stats.get(k, 0) >= v
    judged = [v for v in checks.values() if v is not None]
    ok = None if (expect is None and not (spec.get("need"))) else all(judged)

    n_notice = sum(1 for e in events if e.level == "notice")
    n_clear = sum(1 for e in events if e.level == "cleared")

    out = {
        "scene": name, "desc": spec["desc"], "frames": n,
        "elapsed_s": round(el, 1), "fps": round(n / el, 2),
        "sliced": bool(args.sliced) and stub is None, "stub_detector": stub is not None,
        "fast_smoke": bool(args.fast),
        "llm": args.llm,
        "llm_stats": getattr(arb, "stats", None) if arb else None,
        "events": [e.to_dict() for e in events], "n_events": len(events),
        "n_notice": n_notice, "n_cleared": n_clear,
        "expect_kinds": sorted(expect) if expect is not None else "n/a(观测)",
        "got_kinds": sorted(got), "checks": checks, "ok": ok, "stats": stats,
        "t_appear": round(WARMUP_FRAMES / FPS, 2), "verdict_trace": trace[-40:],
    }

    print(f"\n── 场景 {name}（{spec['desc']}）  {n} 帧 / {el:.0f}s / {n / el:.1f} fps"
          f"{'  [受控检测器]' if stub else ''}")
    for e in events:
        print(f"   {e}")
        for r in e.reasons[:4]:
            print(f"       · {r}")
    if not events:
        print("   （无告警）")
    elif n_notice or n_clear:
        print(f"   （另有 notice {n_notice} 条 / 自动撤销 {n_clear} 条）")
    for k, v in checks.items():
        print(f"   {k}: {'观测' if v is None else ('✅' if v else '❌')}")
    print(f"   统计：{stats}")
    if trace:
        step = max(1, len(trace) // 8)
        print("   时序判别演化：")
        for r in trace[::step][-8:]:
            why = r["reasons"][-1] if r["reasons"] else ""
            print(f"     t={r['t']:>5.1f}s tid={r['tid']} 存续{r['dur']:>4.1f}s "
                  f"{r['label']:<9} {r['score']:.2f} 火邻={r['nearby']}  {why[:36]}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="all", choices=["all", *SCENES])
    ap.add_argument("--sliced", type=int, default=1, help="1 开切片 0 关")
    ap.add_argument("--every", type=int, default=1, help="抽帧间隔")
    ap.add_argument("--imgsz", type=int, default=896)
    ap.add_argument("--n-mul", type=float, default=1.0, help="序列长度倍数（拉长测长期行为）")
    ap.add_argument("--save", type=int, default=1, help="1 保存告警截图")
    ap.add_argument("--trace", type=int, default=1, help="1 打印时序判别演化")
    ap.add_argument("--fast", type=int, default=0, help="1 开极快模式（烟跳过时序判别）")
    ap.add_argument("--llm", default="off",
                    choices=["off", "dry", "dismiss", "alarm", "uncertain"],
                    help="LLM 裁判：off=不用 / dry=假接口 / dismiss=判蒸汽 / alarm=判火 / uncertain=拿不准")
    ap.add_argument("--out", default=str(ROOT / "runs" / "pipeline_test.json"))
    args = ap.parse_args()

    det = Detector(str(WEIGHTS), conf=0.20, imgsz=args.imgsz, device=0)
    print(f"模型 {WEIGHTS.name}  imgsz={args.imgsz}  sliced={bool(args.sliced)}  "
          f"every={args.every}  n_mul={args.n_mul}  类别={det.names}")
    print("先验逻辑自检：")
    prior_ok = check_fire_nearby()

    names = list(SCENES) if args.scene == "all" else [args.scene]
    results = [run_scene(nm, det, args) for nm in names]

    Path(args.out).write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    judged = [r for r in results if r["ok"] is not None]
    npass = sum(bool(r["ok"]) for r in judged)
    print(f"\n{'=' * 62}\n先验自检 {'通过' if prior_ok else '失败'}   "
          f"场景 {npass}/{len(judged)} 通过（另 {len(results) - len(judged)} 个仅观测）"
          f"   → {args.out}")


if __name__ == "__main__":
    main()
