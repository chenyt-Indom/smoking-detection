"""v4 训练实时面板：进度 + 指标 + GPU + ETA，5 秒刷新。

为什么训练时看不到输出
----------------------
训练是以「后台任务」方式启动的（stdout 重定向到 train_v4b.log），没有控制台窗口，
关掉终端也不会中断训练。想看实时状态，跑这个脚本即可（只读，不影响训练）。

用法：
    python scripts/watch_v4.py
    python scripts/watch_v4.py --dir D:/fire_runs/fire_smoke_v4_26s960 --log D:/fire_runs/train_v4b.log
"""
from __future__ import annotations

import argparse
import csv
import os
import re
import subprocess
import sys
import time
from pathlib import Path


def gpu_stats() -> str:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        ).stdout.strip().split("\n")[0]
        util, used, total, temp, power = [x.strip() for x in out.split(",")]
        bar_n = int(float(util) / 100 * 24)
        return (f"GPU  [{'#' * bar_n}{'.' * (24 - bar_n)}] {util:>3}%  "
                f"显存 {used}/{total} MiB  {temp}°C  {float(power):.0f}W")
    except Exception:
        return "GPU  [nvidia-smi 不可用]"


def progress_line(log: Path) -> str:
    """从日志里取最后一行迭代进度。"""
    if not log.exists():
        return "日志尚未生成"
    try:
        with log.open("rb") as f:
            f.seek(0, os.SEEK_END)
            f.seek(max(0, f.tell() - 8192))
            tail = f.read().decode("utf-8", errors="ignore")
    except Exception:
        return "日志读取失败"
    lines = [ln.strip() for ln in tail.replace("\r", "\n").split("\n") if ln.strip()]
    # 日志里带 ANSI 控制符（\x1b[K 之类），先剥掉再匹配
    ansi = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
    for ln in reversed(lines):
        ln = ansi.sub("", ln).strip()
        if re.match(r"^\d+/\d+\s", ln):
            return ln
    return "（等待第一个迭代）"


def metrics(dir_: Path) -> list[str]:
    f = dir_ / "results.csv"
    if not f.exists():
        return ["（首轮尚未结束，指标还未写入）"]
    rows = list(csv.DictReader(f.open(encoding="utf-8")))
    if not rows:
        return ["（暂无数据）"]
    out = []
    best = max(rows, key=lambda r: float(r.get("metrics/mAP50(B)") or 0))
    for r in rows[-3:]:
        out.append(
            f"ep{int(float(r['epoch'])):>3}  mAP50={float(r['metrics/mAP50(B)']):.4f}  "
            f"mAP50-95={float(r['metrics/mAP50-95(B)']):.4f}  "
            f"P={float(r['metrics/precision(B)']):.3f}  R={float(r['metrics/recall(B)']):.3f}"
        )
    out.append(f"最佳：ep{int(float(best['epoch']))}  mAP50={float(best['metrics/mAP50(B)']):.4f}")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="D:/fire_runs/fire_smoke_v4_26s960")
    ap.add_argument("--log", default="D:/fire_runs/train_v4b.log")
    ap.add_argument("--once", action="store_true")
    a = ap.parse_args()
    d, log = Path(a.dir), Path(a.log)

    while True:
        os.system("cls" if os.name == "nt" else "clear")
        print("=" * 74)
        print("  火情/烟雾 v4 · yolo26s @960 · multi_scale 0.5")
        print("=" * 74)
        print(progress_line(log))
        print("-" * 74)
        print(gpu_stats())
        print("-" * 74)
        for line in metrics(d):
            print(line)
        print("-" * 74)
        print(f"权重目录：{d / 'weights'}")
        print("Ctrl+C 退出（退出不会中断训练）")
        if a.once:
            return 0
        try:
            time.sleep(5)
        except KeyboardInterrupt:
            return 0


if __name__ == "__main__":
    sys.exit(main())
