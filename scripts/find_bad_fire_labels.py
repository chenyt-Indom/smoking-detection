"""火域标注体检：找出「框错物体」「框太松」「漏标火区」的样本，并给出建议火域框。

为什么不用人工逐张看 13,476 张
------------------------------
v4 抽检（2026-10-30）发现的标注问题高度集中：
  · 大片火只标了零散火点（森林火、草地火）→ 模型学不到火域边界
  · 火框框到消防员/装备/烟上 → 直接注入错样本
  · 火框比实际火焰大一大圈 → mAP50 还能看，mAP50-95 上不去
这些样本是少数，但决定指标上限。

三个判定量（互相独立，不再重复计同一个信号）
--------------------------------------------
  inside  = GT 火框内火焰像素 / 框面积
            低 → 框里几乎没有火，疑似框错物体（框到人、装备、纯烟）
  tight   = GT 框面积 / 框内火焰像素的最小外接框面积
            高 → 框比火大一圈，靠 mAP50 蒙过去、mAP50-95 掉点
  uncover = 框外火焰像素 / 框内火焰像素
            高 → 大片火区没标，模型学不到边界

阈值怎么来的（关键）
--------------------
不猜。先跑 `--calibrate` 在随机样本上统计这三个量的真实分布，
按分位数定阈值（例如 inside 取 P8 → 只把最差的 8% 推给人工），
写进 thresholds.json；正式跑时自动加载。
这样不同数据集、不同火焰色掩码质量下都能自动适配。

输出：
  · calibrate.csv / thresholds.json   标定结果
  · audit.csv                          可疑样本 + 问题类型 + 严重度
  · sheet_*.jpg                        对比图（橙=GT火框，红=框错，绿=建议补框，黄=框太松）

用法：
    python scripts/find_bad_fire_labels.py --calibrate 600
    python scripts/find_bad_fire_labels.py --limit 300
    python scripts/find_bad_fire_labels.py --split train --limit 0    # 0 = 全量
"""
from __future__ import annotations

import argparse
import csv
import heapq
import json
import random
import sys
from pathlib import Path

import cv2
import numpy as np

# ---- 火焰色掩码参数 ----
FLAME_MIN_AREA_PCT = 0.005    # 火焰候选连通域最小面积（占画面 %），只用于「建议补框」
MIN_JUDGE_PCT = 0.5           # 框小于画面这个比例就不做「框内无火」判定
                              # 实测（2026-10-30）：自采的打火机/小火堆样本，框占画面仅
                              # 0.06%~1.35%、火像素 0~35 个，硬判会把这些正确标注全删掉。
                              # 宁可漏判小框，也不能误删小目标火的正样本。

# ---- 判定阈值：会被 thresholds.json 覆盖 ----
DEFAULTS = {
    "low_inside": 0.02,       # inside < 该值 → 疑似框错物体（框里没有火）
    "high_tight": 2.8,        # tight  > 该值 → 疑似框太松
    "high_uncover": 20.0,     # 框外火焰 / 框内火焰 > 该值
    "min_uncover_pct": 8.0,   # 且框外火焰至少占画面这么多 %，压掉掩码噪声
    "src": "builtin",
}

# 分位标定：flag 掉最差的一小撮。均带 floor，避免分布有质量堆积时算出退化阈值
CALIB = {
    # low_inside 是**绝对判据**：框内火焰像素不到 1% 基本等于框里没火，不按分位标定。
    # 曾经按 P10 标成 0.0744，把「框里有 7% 火焰」的正常框也判成框错，实测误删小目标火标注。
    "low_inside": ("fixed", 0.01),
    "high_tight": ("p90", 2.0),         # P90 与 2.0 取大者
    "high_uncover": ("p90", 20.0),
    "min_uncover_pct": ("p85", 4.0),
}
LOW_INSIDE_MAX_RATE = 0.25   # 固定阈值下命中率超过这个比例，说明掩码在本数据集失效，改回 P5 兜底


def imread_u(p: Path) -> np.ndarray | None:
    data = np.fromfile(str(p), dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR) if data.size else None


def imwrite_u(p: Path, img: np.ndarray, q: int = 90) -> None:
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, q])
    buf.tofile(str(p))


def flame_mask(bgr: np.ndarray, relaxed: bool = False) -> np.ndarray:
    """火焰色掩码：橙红高饱和 + 高亮白心。只用 3x3 闭运算补小孔，不做开运算。

    两档灵敏度，分别服务两个不同的问题：
      relaxed=True  「这个框里到底有没有火」—— 宁可多算，漏判会把正确标注误删。
                    实测把 s/v 门槛从 110/140 降到 90/110，暗场景小火苗的识别率从 0% 提到 44%。
      relaxed=False 「火到底有多大」—— 必须准，否则棕色土壤/木头混进来会把
                    火区外接框撑大、让「框太松」判不出来。
    """
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    if relaxed:
        warm = (((h <= 35) | (h >= 165)) & (s >= 90) & (v >= 110))
        core = ((v >= 215) & (s <= 90))
    else:
        warm = (((h <= 28) | (h >= 165)) & (s >= 110) & (v >= 140))
        core = ((v >= 235) & (s <= 70))                   # 火焰白心
    m = ((warm | core).astype(np.uint8)) * 255
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    return cv2.morphologyEx(m, cv2.MORPH_CLOSE, k)


def regions(mask: np.ndarray, min_area: float) -> list[tuple[int, int, int, int]]:
    n, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    return [(stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP],
             stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT])
            for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] >= min_area]


def read_labels(p: Path) -> list[tuple[int, float, float, float, float]]:
    out = []
    lb = p.parent.parent / "labels" / (p.stem + ".txt")
    if not lb.exists():
        return out
    for line in lb.read_text(encoding="utf-8", errors="ignore").splitlines():
        t = line.split()
        if len(t) >= 5:
            out.append((int(float(t[0])), *[float(x) for x in t[1:5]]))
    return out


def analyze(img_path: Path, want_vis: bool = False) -> dict | None:
    """算原始度量，不做判定。返回 None = 图中没有 fire 标注或读图失败。"""
    img = imread_u(img_path)
    if img is None:
        return None
    h, w = img.shape[:2]
    total = float(h * w)
    labels = read_labels(img_path)
    fire = [(cx, cy, bw, bh) for c, cx, cy, bw, bh in labels if c == 0]
    if not fire:
        return None

    mask = flame_mask(img)                  # 严格：算火有多大
    mask_r = flame_mask(img, relaxed=True)  # 高灵敏：只用来判断框里有没有火

    boxes = []
    for cx, cy, bw, bh in fire:
        x1 = int(max(0, (cx - bw / 2) * w))
        y1 = int(max(0, (cy - bh / 2) * h))
        x2 = int(min(w, (cx + bw / 2) * w))
        y2 = int(min(h, (cy + bh / 2) * h))
        if x2 > x1 and y2 > y1:
            boxes.append((x1, y1, x2, y2))
    if not boxes:
        return None

    # ---- 逐框：inside & tight（逐框算，画图时才能只标出真正有问题的那个框）----
    min_px = max(12.0, 0.0002 * total)          # 少于这点火像素，紧致度没有统计意义
    stats = []
    for x1, y1, x2, y2 in boxes:
        sub = mask[y1:y2, x1:x2]
        area = float((x2 - x1) * (y2 - y1))
        n_r = float(np.count_nonzero(mask_r[y1:y2, x1:x2]))   # 高灵敏计数
        n = float(np.count_nonzero(sub))                      # 严格计数
        judgeable = (area / total * 100) >= MIN_JUDGE_PCT
        tight = 0.0
        if n >= min_px and judgeable:
            ys, xs = np.nonzero(sub)
            tight = area / max(float((xs.max() - xs.min() + 1) * (ys.max() - ys.min() + 1)), 1.0)
        stats.append({"box": (x1, y1, x2, y2), "inside": n_r / area,
                      "tight": tight, "judgeable": judgeable})

    # ---- 整图：covered / uncovered ----
    cover = np.zeros((h, w), np.uint8)
    for x1, y1, x2, y2 in boxes:
        cover[y1:y2, x1:x2] = 255
    flame_in = float(np.count_nonzero(cv2.bitwise_and(mask, cover)))
    flame_total = float(np.count_nonzero(mask))
    uncovered = max(0.0, flame_total - flame_in)
    tights = [s["tight"] for s in stats if s["tight"] > 0]
    judged = [s["inside"] for s in stats if s["judgeable"]]

    m = {
        "image": img_path.name,
        "n_fire": len(boxes),
        "n_judged": len(judged),
        "box_pct": sum((x2 - x1) * (y2 - y1) for x1, y1, x2, y2 in boxes) / total * 100,
        "flame_pct": flame_total / total * 100,
        "inside": min(judged) if judged else 1.0,     # 没有可判框 → 视为正常
        "inside_mean": float(np.mean(judged)) if judged else 1.0,
        "tight": max(tights) if tights else 0.0,
        "uncover": uncovered / max(flame_in, 1.0),
        "uncover_pct": uncovered / total * 100,
        "stats": stats,
    }
    if want_vis:
        m["_vis"] = (img, cover, mask)
    return m


def pct(values: list[float], q: float) -> float:
    return float(np.percentile(values, q)) if values else 0.0


def has_fire_label(p: Path) -> bool:
    """只读 txt 不解码图片——用来在 imread 之前把一半无火样本筛掉。"""
    lb = p.parent.parent / "labels" / (p.stem + ".txt")
    if not lb.exists():
        return False
    try:
        for line in lb.read_text(encoding="utf-8", errors="ignore").splitlines():
            t = line.split()
            if t and abs(float(t[0])) < 0.5:      # class 0 = fire
                return True
    except Exception:
        return False
    return False


def fire_images(base: Path) -> tuple[list[Path], int]:
    all_imgs = sorted((base / "images").glob("*.jpg"))
    return [p for p in all_imgs if has_fire_label(p)], len(all_imgs)


def collect(base: Path, n: int, seed: int) -> tuple[list[dict], int]:
    fire, total = fire_images(base)
    print(f"  含火图 {len(fire)} / 全部 {total} 张", flush=True)
    if n and n < len(fire):
        random.Random(seed).shuffle(fire)
        fire = fire[:n]
    out = []
    for i, p in enumerate(fire):
        r = analyze(p)
        if r:
            out.append(r)
        if (i + 1) % 300 == 0:
            print(f"  分析 {i + 1}/{len(fire)}", flush=True)
    return out, len(fire)


def do_calibrate(base: Path, n: int, seed: int, outdir: Path) -> int:
    print(f"标定：随机抽 {n} 张（seed={seed}）统计火焰度量分布", flush=True)
    rows, n_checked = collect(base, n, seed)
    if not rows:
        print("[x] 没有含火标注的样本，检查 --data / --split")
        return 1
    outdir.mkdir(parents=True, exist_ok=True)
    calib_fields = ["image", "n_fire", "box_pct", "flame_pct", "inside",
                    "inside_mean", "tight", "uncover", "uncover_pct"]
    with (outdir / "calibrate.csv").open("w", newline="", encoding="utf-8-sig") as fh:
        wr = csv.DictWriter(fh, fieldnames=calib_fields, extrasaction="ignore")
        wr.writeheader()
        wr.writerows(rows)

    inside = [r["inside"] for r in rows]
    tight = [r["tight"] for r in rows if r["tight"] > 0]
    unc = [r["uncover"] for r in rows]
    upct = [r["uncover_pct"] for r in rows]
    series = {"inside": inside, "tight": tight, "uncover": unc, "uncover_pct": upct}

    print("=" * 74)
    print(f"含火样本 {len(rows)} 张（随机抽自 {n_checked} 张含火图）")
    print(f"{'指标':<14}{'P10':>9}{'P50':>9}{'P85':>9}{'P90':>9}{'max':>9}")
    for name, arr in series.items():
        print(f"{name:<14}{pct(arr,10):>9.3f}{pct(arr,50):>9.3f}{pct(arr,85):>9.3f}"
              f"{pct(arr,90):>9.3f}{max(arr) if arr else 0:>9.3f}")
    print(f"{'flame_pct':<14}{pct([r['flame_pct'] for r in rows],10):>9.2f}"
          f"{pct([r['flame_pct'] for r in rows],50):>9.2f}"
          f"{pct([r['flame_pct'] for r in rows],85):>9.2f}"
          f"{pct([r['flame_pct'] for r in rows],90):>9.2f}")

    qmap = {"p10": 10, "p85": 85, "p90": 90}
    metric_of = {"low_inside": "inside", "high_tight": "tight",
                 "high_uncover": "uncover", "min_uncover_pct": "uncover_pct"}
    thr = {"src": f"calibrated on {len(rows)} fire imgs ({base.parent.name}/{base.name})"}
    for key, (q, floor) in CALIB.items():
        arr = series[metric_of[key]]
        if q == "fixed":
            rate = sum(1 for v in arr if v < floor) / max(len(arr), 1)
            if rate > LOW_INSIDE_MAX_RATE:      # 掩码在本数据集太弱，固定阈值会全命中 → 退回 P5
                thr[key] = round(pct(arr, 5), 4)
                print(f"[!] 固定阈值 {floor} 命中率 {rate * 100:.0f}% 过高，low_inside 改用 P5")
            else:
                thr[key] = floor
                print(f"    low_inside 固定 {floor}，命中率 {rate * 100:.1f}%")
        else:
            thr[key] = round(max(pct(arr, qmap[q]), floor), 4)
    (outdir / "thresholds.json").write_text(json.dumps(thr, indent=2), encoding="utf-8")
    print("-" * 74)
    print(f"阈值 → inside<{thr['low_inside']}   tight>{thr['high_tight']}   "
          f"uncover>{thr['high_uncover']} 且 uncover_pct>={thr['min_uncover_pct']}%")
    print(f"写入 {outdir / 'thresholds.json'}")
    return 0


def load_thresholds(outdir: Path) -> dict:
    f = outdir / "thresholds.json"
    if f.exists():
        try:
            d = json.loads(f.read_text(encoding="utf-8"))
            return {**DEFAULTS, **d}
        except Exception as e:
            print(f"[!] thresholds.json 读取失败（{e}），用内置默认值")
    return dict(DEFAULTS)


def _clip(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


def judge(m: dict, th: dict) -> list[tuple[str, str]]:
    """返回 [(问题类型, 说明)]。三个判据互相独立，不再重复计同一信号。"""
    issues = []
    n_wrong = sum(1 for s in m["stats"] if s["judgeable"] and s["inside"] < th["low_inside"])
    if n_wrong:
        issues.append(("疑似框错物体",
                       f"{n_wrong}/{m['n_fire']} 个框内无火（最少 {m['inside'] * 100:.2f}%）"))
    n_loose = sum(1 for s in m["stats"] if s["tight"] > th["high_tight"])
    if n_loose:
        issues.append(("疑似框太松",
                       f"{n_loose}/{m['n_fire']} 个框偏大（最松 {m['tight']:.1f} 倍，伤 mAP50-95）"))
    # uncover 只作辅助：掩码会把暖色背景算进来，必须同时满足绝对面积门槛
    if m["uncover"] > th["high_uncover"] and m["uncover_pct"] >= th["min_uncover_pct"]:
        issues.append(("疑似漏标火区",
                       f"框外火焰是框内 {m['uncover']:.0f} 倍、占画面 {m['uncover_pct']:.1f}%"))
    return issues


def severity(m: dict, th: dict) -> float:
    """连续严重度 0~1，用来排序。三个分量都不在 1.0 处饱和，保证同一类问题内部也能排先后。"""
    li = max(th["low_inside"], 1e-6)
    ht = max(th["high_tight"], 1.0)
    s_in = _clip(1.0 - m["inside"] / li)                       # 框里越没火越重
    s_tg = _clip((m["tight"] - 1.0) / max(ht * 1.5 - 1.0, 1e-6))  # 框越松越重
    s_uc = _clip(m["uncover_pct"] / 25.0)                      # 漏标面积越大越重
    return round(min(1.0, 0.50 * s_in + 0.25 * s_tg + 0.25 * s_uc), 3)


def draw(m: dict, issues: list[tuple[str, str]], th: dict) -> np.ndarray:
    """逐框上色：红=框内无火，黄=框偏大，橙=看着正常，绿=建议补充的火区。"""
    img, cover, mask = m["_vis"]
    vis = img.copy()
    for s in m["stats"]:
        x1, y1, x2, y2 = s["box"]
        if s["judgeable"] and s["inside"] < th["low_inside"]:
            col = (0, 0, 255)
        elif s["judgeable"] and s["tight"] > th["high_tight"]:
            col = (0, 215, 255)
        else:
            col = (0, 140, 255)
        cv2.rectangle(vis, (x1, y1), (x2, y2), col, 2)
    if any("漏标" in k for k, _ in issues):
        h, w = img.shape[:2]
        for rx, ry, rw, rh in regions(mask, FLAME_MIN_AREA_PCT / 100 * h * w):
            if cover[ry:ry + rh, rx:rx + rw].mean() < 60:
                cv2.rectangle(vis, (rx, ry), (rx + rw, ry + rh), (0, 200, 0), 2)
    return vis


def write_sheets(vis_list: list[tuple], outdir: Path, per_sheet: int, max_sheets: int) -> list[Path]:
    """已按严重度排序的 (score, seq, stem, vis)，每 per_sheet 张拼一张对比图。"""
    made = []
    cell, cols = 460, 3
    for si in range(min(max_sheets, (len(vis_list) + per_sheet - 1) // per_sheet)):
        chunk = vis_list[si * per_sheet:(si + 1) * per_sheet]
        if not chunk:
            break
        rowsn = (len(chunk) + cols - 1) // cols
        canvas = np.full((rowsn * cell, cols * cell, 3), 255, np.uint8)
        for i, (_sc, _seq, stem, vis) in enumerate(chunk):
            hh, ww = vis.shape[:2]
            s = min(cell / ww, (cell - 26) / hh)
            im = cv2.resize(vis, (int(ww * s), int(hh * s)))
            r, c = divmod(i, cols)
            y0, x0 = r * cell + 26, c * cell
            canvas[y0:y0 + im.shape[0], x0:x0 + im.shape[1]] = im
            cv2.putText(canvas, f"#{si * per_sheet + i + 1} {_sc:.2f} {stem[:24]}",
                        (x0 + 5, y0 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (20, 20, 20), 1, cv2.LINE_AA)
        f = outdir / f"sheet_{si + 1:02d}.jpg"
        imwrite_u(f, canvas)
        made.append(f)
    return made


def do_audit(base: Path, n: int, seed: int, outdir: Path, top: int, max_sheets: int) -> int:
    th = load_thresholds(outdir)
    print(f"阈值：inside<{th['low_inside']}   tight>{th['high_tight']}   "
          f"uncover>{th['high_uncover']} 且 uncover_pct>={th['min_uncover_pct']}%")
    print(f"来源：{th.get('src')}", flush=True)

    imgs, total_all = fire_images(base)
    print(f"含火图 {len(imgs)} / 全部 {total_all} 张", flush=True)
    if n and n < len(imgs):
        random.Random(seed).shuffle(imgs)
        imgs = imgs[:n]

    outdir.mkdir(parents=True, exist_ok=True)
    rows, n_fire_img, n_flagged = [], 0, 0
    # 有界堆：只留最严重的 top 张画布，避免全量跑时几千张图把内存吃干净
    heap: list[tuple] = []
    for i, p in enumerate(imgs):
        m = analyze(p, want_vis=True)
        if m is not None:
            n_fire_img += 1
            issues = judge(m, th)
            if issues:
                n_flagged += 1
                sc = severity(m, th)
                rows.extend([{"image": m["image"], "issue": k, "detail": d,
                              "score": sc,
                              "inside": round(m["inside"], 4), "tight": round(m["tight"], 2),
                              "uncover": round(m["uncover"], 1),
                              "uncover_pct": round(m["uncover_pct"], 2),
                              "box_pct": round(m["box_pct"], 3),
                              "n_fire": m["n_fire"]} for k, d in issues])
                if top <= 0:
                    heap.append((sc, n_flagged, p.stem, draw(m, issues, th)))
                elif len(heap) < top:
                    heapq.heappush(heap, (sc, n_flagged, p.stem, draw(m, issues, th)))
                elif sc > heap[0][0]:
                    heapq.heapreplace(heap, (sc, n_flagged, p.stem, draw(m, issues, th)))
        m = None
        if (i + 1) % 500 == 0:
            print(f"  分析 {i + 1}/{len(imgs)}，可疑 {n_flagged}", flush=True)

    rows.sort(key=lambda r: (-r["score"], r["image"]))
    with (outdir / "audit.csv").open("w", newline="", encoding="utf-8-sig") as fh:
        wr = csv.DictWriter(fh, fieldnames=["rank", "image", "issue", "detail", "score",
                                            "inside", "tight", "uncover", "uncover_pct",
                                            "box_pct", "n_fire"])
        wr.writeheader()
        wr.writerows([{"rank": i + 1, **r} for i, r in enumerate(rows)])

    vis_list = sorted(heap, key=lambda x: (-x[0], x[1]))
    sheets = write_sheets(vis_list, outdir, per_sheet=12, max_sheets=max_sheets)

    print("=" * 74)
    print(f"分析 {n_fire_img} 张含火图，命中可疑 {n_flagged} 张"
          f"（{n_flagged / max(n_fire_img, 1) * 100:.1f}%）")
    kinds: dict[str, int] = {}
    for r in rows:
        kinds[r["issue"]] = kinds.get(r["issue"], 0) + 1
    for k, v in sorted(kinds.items(), key=lambda x: -x[1]):
        print(f"  {k}: {v}")
    print(f"输出：{outdir / 'audit.csv'}")
    for f in sheets:
        print(f"对比图：{f}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="E:/datasets/fire_smoke_v4")
    ap.add_argument("--split", default="train")
    ap.add_argument("--limit", type=int, default=300, help="抽查张数，0=全量")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--calibrate", type=int, default=0,
                    help="标定模式：随机抽 N 张统计分布，写 thresholds.json")
    ap.add_argument("--top", type=int, default=300, help="对比图只保留最严重的前 N 张，0=全部")
    ap.add_argument("--sheets", type=int, default=5, help="最多出几张对比图（每张 12 图）")
    ap.add_argument("--out", default="D:/视觉安防系统/runs/fire_label_audit")
    a = ap.parse_args()

    base = Path(a.data) / a.split
    if not (base / "images").exists():
        print(f"[x] 找不到 {base / 'images'}")
        return 1
    outdir = Path(a.out)
    if a.calibrate:
        return do_calibrate(base, a.calibrate, a.seed, outdir)
    return do_audit(base, a.limit, a.seed, outdir, a.top, a.sheets)


if __name__ == "__main__":
    sys.exit(main())
