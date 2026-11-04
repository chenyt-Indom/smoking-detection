"""后端数值稳定性测试：同一张图 + 真实视频扰动，判定会不会翻。

为什么需要它（而不是在线采样）
------------------------------
"框一直闪"的在线指标被场景内容污染得厉害：用户走开一下，框数就归零，
指标立刻"变好"。之前已经因此得出过两次相反结论。

这里换一个**受控**的做法：
  拿一张有目标的验证集图片 -> 生成 K 个"看起来几乎一样"的扰动版本
  （JPEG 重压缩 / 轻微高斯噪声 / 轻微模糊，模拟视频编码抖动和传感器噪声）
  -> 三个后端各跑一遍 -> 数"同一张图的不同版本，检出数会不会变"。

如果某个后端在数值上更脆，它的判定就会在扰动下翻来覆去 —— 这正是
画面闪烁的根源（置信度贴着门槛，一点点数值差就跨过去）。

指标
----
  翻动率 : 每个图片组里，检出数与"多数结果"不同的版本占比。越低越稳。
  平均框数、平均最高置信度：辅助判断是不是靠"少检"换来的稳定。

用法
----
    python scripts/bench_backend_stability.py
    python scripts/bench_backend_stability.py --groups 10 --variants 6
"""
from __future__ import annotations

import argparse
import random
import statistics
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# 主模型的后端（香烟模型同法，但先聚焦主模型 —— 闪得最厉害的是 smoke）
BACKENDS = {
    ".pt  原生FP32": r"D:\trt_export\main_pt.pt",
    "引擎 FP16": r"D:\trt_export\main_fp16.engine",
    "引擎 FP32": r"D:\视觉安防系统\models\fire_smoke_v5_26s960_best.engine",
}
def safe_imread(path) -> "np.ndarray | None":
    """读图必须绕开 cv2.imread —— Windows 上它打不开含中文的路径。

    验证集文件名形如 `A_烟雾_火_bothFireAndSmoke_CV001509.jpg`，
    直接 cv2.imread 会返回 None 并在 stderr 刷 findDecoder 警告，
    于是三个后端都"检出 0 个框"，看起来像模型全瞎了（实际是没读到图）。
    用 np.fromfile + cv2.imdecode 走字节流就能正常读。
    """
    try:
        buf = np.fromfile(str(path), dtype=np.uint8)
        return cv2.imdecode(buf, cv2.IMREAD_COLOR) if buf.size else None
    except Exception:                      # noqa: BLE001
        return None


IMG_DIRS = [
    Path(r"E:\datasets\fire_smoke_v5\test\images"),
    Path(r"D:\training_data\smoke\v32_smoking\images"),
]


def variants(im: np.ndarray, k: int) -> list[np.ndarray]:
    """生成 k 个"肉眼几乎无差"的扰动版本，模拟视频流里的编码/传感器抖动。"""
    out = []
    rng = np.random.default_rng(0)
    for i in range(k):
        v = im.copy()
        if i % 3 == 1:
            v = cv2.GaussianBlur(v, (3, 3), 0.6)          # 轻微模糊
        if i % 3 == 2:
            n = rng.normal(0, 3.0, v.shape).astype(np.int16)   # 传感器噪声
            v = np.clip(v.astype(np.int16) + n, 0, 255).astype(np.uint8)
        q = [92, 85, 75][i % 3]                           # JPEG 重压缩
        ok, buf = cv2.imencode(".jpg", v, [int(cv2.IMWRITE_JPEG_QUALITY), q])
        if ok:
            v = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        out.append(v)
    return out


def pick_images(n: int, cam_hint: str = "") -> list[Path]:
    """挑有标注、且标注里确实有目标的图，避免拿空白图白测。"""
    cands: list[Path] = []
    for d in IMG_DIRS:
        if not d.is_dir():
            continue
        lbl = d.parent / "labels"
        if not lbl.is_dir():
            lbl = d.parent.parent / "labels" / d.name
        for p in sorted(d.glob("*.jpg"))[:400]:
            lf = (lbl / (p.stem + ".txt")) if lbl.is_dir() else None
            if lf is not None and lf.is_file() and lf.stat().st_size > 8:
                cands.append(p)
    random.Random(7).shuffle(cands)
    return cands[:n]


def count_boxes(m, imgs: list[np.ndarray], conf: float) -> list[int]:
    """m 是已加载的模型 —— 千万别在循环里重复 YOLO(path)，引擎重载一次好几秒。"""
    out = []
    for im in imgs:
        r = m.predict(im, imgsz=960, conf=conf, iou=0.45, device=0, verbose=False)[0]
        out.append(len(r.boxes))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--groups", type=int, default=10)
    ap.add_argument("--variants", type=int, default=6)
    ap.add_argument("--conf", type=float, default=0.05)
    args = ap.parse_args()

    files = pick_images(args.groups)
    if not files:
        print("[x] 没找到带标注的图片")
        return
    print("=" * 78)
    print(f"后端数值稳定性：{len(files)} 张图 × {args.variants} 个扰动版本，"
          f"conf={args.conf}")
    print("=" * 78)
    print("  取图：")
    for f in files[:5]:
        print(f"    {f}")
    if len(files) > 5:
        print(f"    … 另 {len(files) - 5} 张")

    groups = [[safe_imread(f)] for f in files]
    groups = [g for g in groups if g[0] is not None]
    if not groups:
        print("[x] 一张图都读不出来（检查路径/权限）")
        return
    perturbed = [variants(g[0], args.variants) for g in groups]

    print()
    print(f"{'后端':<16}{'翻动率':>9}{'平均框数':>10}{'检出为0的组':>13}{'最慢组耗时':>12}")
    print("-" * 78)
    from ultralytics import YOLO
    rows = {}
    for tag, path in BACKENDS.items():
        model = YOLO(path)
        flips_total = 0.0
        counts_all = []
        zero_groups = 0
        import time
        worst = 0.0
        for grp in perturbed:
            t0 = time.time()
            cs = count_boxes(model, grp, args.conf)
            worst = max(worst, time.time() - t0)
            counts_all.extend(cs)
            if max(cs) == 0:
                zero_groups += 1
                continue
            mode = Counter(cs).most_common(1)[0][0]
            flips_total += sum(1 for c in cs if c != mode) / len(cs)
        n_ok = max(1, len(perturbed) - zero_groups)
        rows[tag] = dict(flip=flips_total / n_ok,
                         mean=statistics.mean(counts_all) if counts_all else 0,
                         zero=zero_groups, worst=worst)
        r = rows[tag]
        print(f"{tag:<16}{r['flip'] * 100:>8.0f}%{r['mean']:>10.2f}"
              f"{r['zero']:>13}{r['worst']:>11.2f}s")
    print("-" * 78)
    best = min(rows, key=lambda t: (rows[t]["flip"], -rows[t]["mean"]))
    print(f"翻动率最低：{best}（{rows[best]['flip'] * 100:.0f}%）")
    print("翻动率=同一个目标在几乎相同的输入下，检出数变来变去的比例 —— 越小越不闪。")
    print("=" * 78)


if __name__ == "__main__":
    main()
