"""训练实时监控：在终端里刷新显示每一轮的进度、指标和 ETA。

为什么不直接用旧仓库的 monitor_v5.py
----------------------------------
那个脚本是按**列的位置**解析 results.csv 的（`d[4]` 当 dfl、`d[7]` 当 mAP50），
而 YOLO26 的列顺序变了（第 5 列是 l1_loss，不是 dfl），位置一错整张表就串了。
这里改成**按表头名取值**，换模型/换版本都不会错。

用法：
    python scripts/monitor_train.py                  # 自动找最新的 runs/*/results.csv
    python scripts/monitor_train.py --run fire_smoke_26n
    python scripts/monitor_train.py --interval 2     # 刷新间隔（秒）
    python scripts/monitor_train.py --once           # 只打印一次（给 AI/日志用）
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


# 可能存放训练结果的目录（ultralytics 续训时会用 checkpoint 里记录的路径，
# 不一定听 --project，所以这里统一扫一遍，谁最新看谁）
CANDIDATE_ROOTS = [
    ROOT / "runs",                  # 仓库内（续训常常落在这里）
    Path("D:/fire_smoke_train/runs"),  # 纯英文训练目录
    Path("D:/fire_runs"),           # 火烟 v4 起：输出直接放这里（不带 detect 层级）
]


def find_latest_run(runs_dir: Path | None = None) -> Path | None:
    roots = [runs_dir] if runs_dir else CANDIDATE_ROOTS
    best, best_ts = None, -1.0
    for runs in roots:
        if not runs or not runs.exists():
            continue
        for p in runs.iterdir():
            if not p.is_dir():
                continue
            f = p / "results.csv"
            # 刚启动、第一轮还没跑完的任务没有 results.csv ——
            # 只看它的话会一直显示上一版旧训练，所以用目录时间兜底。
            ts = f.stat().st_mtime if f.exists() else p.stat().st_mtime
            if ts > best_ts:
                best, best_ts = p, ts
    return best


def read_rows(csv_path: Path) -> list[dict]:
    try:
        with csv_path.open(encoding="utf-8") as f:
            return list(csv.DictReader(f))
    except Exception:
        return []


def g(row: dict, *keys, default=None):
    """按表头名取值，兼容不同版本的前缀差异。"""
    for k in keys:
        if k in row and row[k] not in ("", None):
            try:
                return float(row[k])
            except ValueError:
                return row[k]
    return default


def fmt_eta(sec: float) -> str:
    if sec <= 0 or sec != sec:
        return "—"
    m, s = divmod(int(sec), 60)
    h, m = divmod(m, 60)
    return f"{h}h{m:02d}m" if h else f"{m}m{s:02d}s"


def render(run: Path, rows: list[dict], epochs_total: int | None) -> str:
    out = []
    bar = "=" * 92
    now = time.strftime("%H:%M:%S")
    out.append(bar)
    out.append(f"  火情/烟雾 训练监控   {now}   运行目录：{run.name}")

    # 顶部信息：模型、数据、关键超参（从 args.yaml 取）
    args_p = run / "args.yaml"
    if args_p.exists():
        try:
            import yaml
            a = yaml.safe_load(args_p.read_text(encoding="utf-8")) or {}
            out.append(f"  模型 {a.get('model', '?')} | imgsz {a.get('imgsz')} | batch {a.get('batch')} | "
                       f"epochs {a.get('epochs')} | 优化器 {a.get('optimizer', 'auto')}")
            out.append(f"  数据 {a.get('data')}")
        except Exception:
            pass
    out.append(bar)

    if not rows:
        out.append("  等待第一轮结果…（刚启动要 1~2 分钟：建缓存 + AMP 检查）")
        return "\n".join(out)

    total = epochs_total or len(rows)
    out.append(f"  {'Ep':>4s} {'box':>7s} {'cls':>7s} {'P':>7s} {'R':>7s} {'mAP50':>8s} "
               f"{'mAP50-95':>9s} {'趋势':>5s} {'耗时':>7s}")
    out.append("  " + "-" * 88)

    maps = []
    best_ep, best_map, best_row = 0, -1.0, None
    for i, r in enumerate(rows):
        ep = int(g(r, "epoch", default=0) or 0)
        box = g(r, "train/box_loss", default=0.0)
        cls = g(r, "train/cls_loss", default=0.0)
        p = g(r, "metrics/precision(B)", default=0.0)
        rec = g(r, "metrics/recall(B)", default=0.0)
        m50 = g(r, "metrics/mAP50(B)", default=0.0)
        m95 = g(r, "metrics/mAP50-95(B)", default=0.0)
        t_cum = g(r, "time", default=0.0) or 0.0
        # ⚠️ results.csv 的 time 是**累计**秒数，不是本轮耗时。
        # 之前直接拿它当每轮时间，ETA 算出 43 小时（错的）。这里用相邻两行差分。
        t = t_cum - (g(rows[i - 1], "time", default=0.0) or 0.0) if i else t_cum
        maps.append(m95)
        if m95 > best_map:
            best_map, best_ep, best_row = m95, ep, r

        if i == 0:
            arrow = "  --"
        else:
            d = m95 - maps[i - 1]
            arrow = " ▲▲" if d > 0.005 else (" ▲" if d > 0.001 else
                    (" ▬" if d > -0.001 else (" ▼" if d > -0.005 else " ▼▼")))
        star = " 🏆" if ep == best_ep and best_map > 0 else ""
        out.append(f"  {ep:>4d} {box:7.4f} {cls:7.4f} {p:7.3f} {rec:7.3f} {m50:8.4f} "
                   f"{m95:9.4f} {arrow:>5s} {t:6.1f}s{star}")

    # 底部统计
    done = len(rows)
    out.append("  " + "-" * 88)
    cum = [g(r, "time", default=0.0) or 0.0 for r in rows]
    deltas = [cum[i] - cum[i - 1] for i in range(1, len(cum)) if cum[i] > cum[i - 1]]
    avg_t = (sum(deltas) / len(deltas)) if deltas else (cum[-1] / max(done, 1))
    eta = (total - done) * avg_t
    pct = done / max(total, 1) * 100
    out.append(f"  进度 {done}/{total}（{pct:.0f}%）  平均 {avg_t:.1f}s/轮  ETA {fmt_eta(eta)}")
    out.append(f"  最佳：第 {best_ep} 轮  mAP50-95 = {best_map:.4f}")

    if len(maps) >= 6:
        recent, earlier = sum(maps[-3:]) / 3, sum(maps[-6:-3]) / 3
        trend = "上升 ↑" if recent > earlier + 0.002 else ("走平 →" if recent > earlier - 0.002 else "下降 ↓（留意过拟合）")
        out.append(f"  最近 3 轮均值 {recent:.4f} vs 前 3 轮 {earlier:.4f} → {trend}")

    # 显存 / GPU 利用率（Windows 有 nvidia-smi 才显示）
    try:
        import subprocess
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=3)
        if r.returncode == 0:
            u, mu, mt, temp = [x.strip() for x in r.stdout.strip().split(",")[:4]]
            out.append(f"  GPU {u}%  显存 {int(mu)/1024:.1f}/{int(mt)/1024:.1f} GB  {temp}℃")
    except Exception:
        pass

    if (run / "weights" / "best.pt").exists():
        mb = (run / "weights" / "best.pt").stat().st_size / 1024 / 1024
        out.append(f"  best.pt 已保存（{mb:.1f} MB）→ {run / 'weights' / 'best.pt'}")
    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="", help="runs/ 下的目录名，空=自动找最新")
    ap.add_argument("--runs-dir", default="", help="runs 根目录（默认仓库下 runs/；训练放别处时指定）")
    ap.add_argument("--interval", type=float, default=3.0)
    ap.add_argument("--once", action="store_true")
    a = ap.parse_args()

    # 不指定 --runs-dir 时走 CANDIDATE_ROOTS 全量扫描（只给 ROOT/runs 会漏掉 D:/fire_runs 的新训练）
    runs_root = Path(a.runs_dir) if a.runs_dir else None
    run = (runs_root / a.run) if (runs_root and a.run) else find_latest_run(runs_root)
    if run is None or not run.exists():
        print(f"[x] 没找到训练目录（runs/ 下没有 results.csv）。先启动训练。")
        return 1

    epochs_total = None
    args_p = run / "args.yaml"
    if args_p.exists():
        try:
            import yaml
            epochs_total = int((yaml.safe_load(args_p.read_text(encoding="utf-8")) or {}).get("epochs") or 0) or None
        except Exception:
            pass

    if a.once:
        print(render(run, read_rows(run / "results.csv"), epochs_total))
        return 0

    try:
        while True:
            os.system("cls" if os.name == "nt" else "clear")
            rows = read_rows(run / "results.csv")
            print(render(run, rows, epochs_total))
            print("\n  Ctrl+C 退出监控（不影响训练）")
            time.sleep(a.interval)
    except KeyboardInterrupt:
        print("\n已退出监控（训练仍在后台跑）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
