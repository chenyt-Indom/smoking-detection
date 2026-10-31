"""阴燃诊断：烟到底在哪一档漏掉，降门限又能捞回多少。

为什么不能笼统说"提高召回"
--------------------------
烟召回 68% 这个数字掩盖了两件事：
  ① **小面积/淡烟**（阴燃早期、远处）漏得最狠，而这类恰恰最该早发现
  ② `smoke_conf=0.45` 是个统一门限，它同时卡住了"浓烟"和"淡烟"

所以这里按 **GT 框面积分档**统计召回，再看把门限从 0.45 降到 0.30/0.20
能捞回多少 —— 并且用纯背景图量出**降门限的代价**（每类各自的误报率）。
只报召回不报代价的建议不能要。

口径
----
· 检测用**完整切片**（要切就切全，和线上一致），一次跑到 conf 0.15，
  之后在代码里按不同门限筛，避免重复推理
· 匹配：**同一张图内**、同类、IoU ≥ 0.5。跨图匹配是错的（归一化坐标下
  任意两个框都可能"重上"，第一版就写错了）

跑法：
    python scripts/eval_smoldering.py --n 500
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

DATA = Path("E:/datasets/fire_smoke_v5/test")
BG = Path("E:/BaiduNetdiskDownload/Y_火灾+烟雾/背景负样本/images")
WEIGHTS = ROOT / "models" / "fire_smoke_v5_26s960_best.pt"

# 面积分档（占画面比例）。阴燃早期基本落在 <1% 这两档
BUCKETS = [(0.0, 0.005, "<0.5%  早期/远处"),
           (0.005, 0.01, "0.5~1% 小"),
           (0.01, 0.03, "1~3%   中"),
           (0.03, 0.10, "3~10%  大"),
           (0.10, 1.01, ">10%   贴脸")]
THRS = (0.45, 0.30, 0.20)
NAMES = {0: "火焰", 1: "烟"}


def imread_u(p: Path):
    import cv2
    return cv2.imdecode(np.frombuffer(p.read_bytes(), np.uint8), cv2.IMREAD_COLOR)


def read_gt(txt: Path) -> list[tuple[int, tuple]]:
    """YOLO 标注 → [(cls, (x1,y1,x2,y2) 归一化)]。"""
    out = []
    if not txt.is_file():
        return out
    for ln in txt.read_text(encoding="utf-8").splitlines():
        v = ln.split()
        if len(v) < 5:
            continue
        c, cx, cy, w, h = int(float(v[0])), *map(float, v[1:5])
        out.append((c, (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)))
    return out


def iou(a, b) -> float:
    ix = min(a[2], b[2]) - max(a[0], b[0])
    iy = min(a[3], b[3]) - max(a[1], b[1])
    if ix <= 0 or iy <= 0:
        return 0.0
    inter = ix * iy
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=500, help="test 集取多少张（均匀抽）")
    ap.add_argument("--imgsz", type=int, default=896)
    ap.add_argument("--bg-n", type=int, default=400, help="背景图取多少张量代价")
    a = ap.parse_args()

    from src.infer import Detector
    det = Detector(str(WEIGHTS), conf=0.15, imgsz=a.imgsz, device=0)

    imgs = sorted((DATA / "images").glob("*.jpg"))
    step = max(1, len(imgs) // a.n)
    picked = imgs[::step][:a.n]
    print(f"test 集 {len(imgs)} 张，均匀抽 {len(picked)} 张 · imgsz {a.imgsz} · 完整切片\n")

    # 每张图：[(cls, box_n, area)] 的 GT + [(cls, conf, box_n)] 的预测，**成对保存**
    items: list[tuple[list, list]] = []
    for p in picked:
        im = imread_u(p)
        if im is None:
            continue
        h, w = im.shape[:2]
        gts = [(c, b, (b[2] - b[0]) * (b[3] - b[1]))
               for c, b in read_gt(DATA / "labels" / (p.stem + ".txt"))]
        pl = [(d.cls_id, d.conf,
               (d.xyxy[0] / w, d.xyxy[1] / h, d.xyxy[2] / w, d.xyxy[3] / h))
              for d in det.detect(im, sliced=True, tile_grid=2, overlap=0.25)]
        items.append((gts, pl))

    def recall(cls: int, thr: float, lo: float, hi: float) -> tuple[int, int]:
        """返回 (命中, 总数)，只在**同一张图内**匹配。"""
        hit = tot = 0
        for gts, pl in items:
            cand = [(c, pb) for c, cf, pb in pl if c == cls and cf >= thr]
            for c, gb, ar in gts:
                if c != cls or not (lo <= ar < hi):
                    continue
                tot += 1
                if any(iou(gb, pb) >= 0.5 for _c, pb in cand):
                    hit += 1
        return hit, tot

    for cls in (1, 0):
        print(f"── {NAMES[cls]} 分段召回 ─────────────────────────────")
        print(f"{'面积档':22s}" + "".join(f"{'conf ' + format(t, '.2f'):>12}" for t in THRS))
        for lo, hi, tag in BUCKETS:
            hits = [recall(cls, t, lo, hi) for t in THRS]
            if hits[0][1] == 0:
                continue
            row = f"{tag:22s}"
            for h_, t_ in hits:
                row += f"{h_ / t_ * 100:10.0f}% "
            print(row + f"  (n={hits[0][1]})")
        hits = [recall(cls, t, 0.0, 9.9) for t in THRS]
        row = f"{'合计':22s}"
        for h_, t_ in hits:
            row += f"{h_ / max(1, t_) * 100:10.0f}% "
        print(row + f"  (n={hits[0][1]})\n")

    # ── 降门限的代价：背景图分类型误报 ─────────────────────
    bgs = sorted(p for p in Path(BG).iterdir() if p.suffix.lower() in (".jpg", ".png"))
    bstep = max(1, len(bgs) // a.bg_n)
    bpick = bgs[::bstep][:a.bg_n]
    print(f"── 代价：{len(bpick)} 张纯背景图上的分类型误报 ──────────")
    print(f"{'conf':>6}" + "".join(f"{NAMES[c]:>16}" for c in (0, 1)))
    for t in THRS:
        cnt = {0: 0, 1: 0}
        for p in bpick:
            im = imread_u(p)
            if im is None:
                continue
            ds = det.detect(im, sliced=True, tile_grid=2, overlap=0.25)
            for c in (0, 1):
                if any(d.cls_id == c and d.conf >= t for d in ds):
                    cnt[c] += 1
        print(f"{t:>6.2f}" + "".join(
            f"{cnt[c]:>8d} ({cnt[c] / len(bpick) * 100:4.1f}%)" for c in (0, 1)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
