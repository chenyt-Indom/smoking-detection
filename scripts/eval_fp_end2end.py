"""端到端误报评估：把纯背景画面当「固定枪机一直盯着」，看会不会报出来。

为什么不能只看单帧误报率
------------------------
`eval_delivery.py` 报的 6.3% 是「这张图里有没有一个框」，那只是**第一道门**。
值班员真正感受到的，是三道门之后还剩几条**正式告警**：
  ① 框级门限（fire_conf 0.30 / smoke_conf 0.45）
  ② 连 N 帧确认 + 同区域冷却
  ③ 闪烁闸（亮度恒定 → 降级成 notice，不是告警）

这个脚本把三道门一起算进去，并且 A/B 掉闪烁闸，看它到底值几个点。

方法（以及它的边界，别过度解读）
--------------------------------
1051 张纯背景图里等间隔抽 N 张，每张**连续保持 HOLD 帧**再换下一张 ——
模拟固定机位盯着一动不动的场景。帧间只加 σ=1.0 的传感器噪声。

σ 不能开大：真实压缩流里静止画面的 `flicker` 实测是 **0.000**（车位灯场景
160 帧）。σ=3 会让 5.9% 的像素超过 8 灰度差门限，把"一动不动"硬生生测成
"在抖"，那是在给闪烁闸造一个不存在的敌人。

所以这里报的是**场景级**误报率：「N 个纯背景场景各盯 HOLD/25 秒，几个报了
正式告警」。不做"每小时多少次"的换算 —— N 个互不相干的场景挤在几分钟里，
按小时外推只会得到一个没有意义的大数。

跑法：
    python scripts/eval_fp_end2end.py --n 40 --hold 100
    python scripts/eval_fp_end2end.py --n 40 --hold 100 --no-gate   # 关掉闪烁闸做对照
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
    """带中文/括号路径的读图。cv2.imread 在 Windows 上遇到非 ASCII 会返回 None。"""
    import cv2
    try:
        buf = np.frombuffer(p.read_bytes(), dtype=np.uint8)
        return cv2.imdecode(buf, cv2.IMREAD_COLOR)
    except Exception:                                                # noqa: BLE001
        return None


def frames_of(imgs, hold: int, rng):
    """每张图连续放 hold 帧，帧间加一点点噪声（真实传感器不会是纯拷贝）。"""
    for im in imgs:
        base = im.astype(np.int16)
        for _ in range(hold):
            noisy = base + rng.normal(0, 1.0, base.shape).astype(np.int16)
            yield np.clip(noisy, 0, 255).astype(np.uint8)


def run(imgs, hold: int, gate: bool, imgsz: int,
        fire_conf: float | None = None, smoke_conf: float | None = None):
    """一个场景一套干净的流水线状态。

    为什么不是所有场景塞进同一次 run：`_can_alarm`/`_can_notice` 的冷却按**区域**
    记账，不同场景的误检框只要位置碰巧重叠，后一个场景就被前排的冷却压住了 ——
    那是把误报率评低，不是真实表现。每个场景从零开始才等价于"一台枪机盯这一个场景"。
    """
    from src.infer import Detector
    det = Detector(str(WEIGHTS), conf=0.20, imgsz=imgsz, device=0)
    rng = np.random.default_rng(0)
    evs: list = []
    st_acc: dict = {}
    for i, im in enumerate(imgs):
        cfg = {
            "detect_mode": "auto",          # 和线上一致
            "flicker_gate": gate,
            "light_memory": False,          # 别读也别写现场那份记忆文件
            "fast_smoke": False,
        }
        if fire_conf is not None:
            cfg["fire_conf"] = fire_conf
        if smoke_conf is not None:
            cfg["smoke_conf"] = smoke_conf
        pipe = Pipeline(det, cfg=cfg, arbiter=None)   # 没有 LLM：最坏情况，全靠规则层
        src = FixedCameraSource(CameraInfo(cam_id=f"fp_{i}", kind="fixed", fps=FPS),
                                frames=frames_of([im], hold, rng), warmup=0)
        one: list = []
        pipe.run(src, max_frames=hold, quiet=True,
                 on_event=lambda ev, _r: one.append(ev))
        evs.extend(one)
        for k, v in pipe.stats.items():
            if isinstance(v, int):
                st_acc[k] = st_acc.get(k, 0) + v
    return evs, st_acc


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40, help="抽多少个背景场景")
    ap.add_argument("--hold", type=int, default=100, help="每个场景连续多少帧")
    ap.add_argument("--imgsz", type=int, default=896)
    ap.add_argument("--bg-dir", default=str(BG))
    ap.add_argument("--gate-only", action="store_true",
                    help="只跑闸开那一种（对比阈值时省一半时间，闸关的结果上次已有）")
    ap.add_argument("--fire-conf", type=float, default=None,
                    help="覆盖流水线的 fire_conf（默认 0.30）")
    ap.add_argument("--smoke-conf", type=float, default=None,
                    help="覆盖流水线的 smoke_conf（默认 0.45）")
    ap.add_argument("--size", default="640x480",
                    help="只取这个尺寸的图（WxH）。**不要改成拉伸对齐** —— "
                         "拉伸会改目标尺度，评出来的不是同一件事")
    a = ap.parse_args()

    files = sorted(p for p in Path(a.bg_dir).iterdir()
                   if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
    if not files:
        raise SystemExit(f"没读到背景图：{a.bg_dir}")
    wt, ht = (int(v) for v in a.size.lower().split("x"))
    # 尺寸必须一致：BackgroundModel 逐像素差分，尺寸一变就直接炸
    same = [p for p in files if (im := imread_u(p)) is not None
            and im.shape[1] == wt and im.shape[0] == ht]
    if not same:
        raise SystemExit(f"没有 {a.size} 的背景图，换个 --size")
    step = max(1, len(same) // a.n)
    picked = same[::step][:a.n]
    imgs = [imread_u(p) for p in picked]
    secs = len(imgs) * a.hold / FPS
    print(f"背景场景 {len(imgs)} 个 × {a.hold / FPS:.0f} 秒 = 等效 {secs / 60:.1f} 分钟无火视频"
          f"（{a.size}，共 {len(same)} 张可用）")
    print(f"权重 {WEIGHTS.name}  imgsz={a.imgsz}  检测策略 auto\n")

    rows = []
    for gate in ((True,) if a.gate_only else (True, False)):
        evs, st = run(imgs, a.hold, gate, a.imgsz, a.fire_conf, a.smoke_conf)
        formal = [e for e in evs if e.level in ("warning", "alarm")]
        notice = [e for e in evs if e.level == "notice"]
        # 每个场景一个独立 cam_id（fp_0、fp_1…），所以撞上误报的**场景数**直接可数
        scenes = {e.cam_id for e in formal}
        rows.append((gate, formal, notice, st, len(scenes)))
        tag = "开" if gate else "关"
        print(f"── 闪烁闸{tag} ─────────────────────────────")
        print(f"  正式告警 {len(formal)} 条 / notice {len(notice)} 条   "
              f"静态光源拦截 {st.get('static_light_blocked', 0)} 帧")
        print(f"  场景误报率 {len(scenes)}/{len(imgs)} = "
              f"{len(scenes) / max(1, len(imgs)) * 100:.0f}%")
        for e in formal[:6]:
            print(f"    · [{e.kind}] {e.level} {e.cam_id} "
                  f"t={e.t_alarm:.1f}s score={e.score:.2f}")
        print()

    on = rows[0]
    if len(rows) > 1:
        off = rows[1]
        print("=" * 62)
        print(f"闪烁闸把正式告警从 {len(off[1])} 条压到 {len(on[1])} 条"
              f"（notice 从 {len(off[2])} 涨到 {len(on[2])} 条 —— 降级不是排除）")
    else:
        print("=" * 62)
        print(f"闸开：正式告警 {len(on[1])} 条 / notice {len(on[2])} 条")
    print("注意：这里每张图都是不同场景，别把条数按小时外推。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
