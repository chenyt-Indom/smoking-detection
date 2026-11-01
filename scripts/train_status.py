"""看训练进度 —— 一条命令给出当前 epoch、loss、mAP、ETA。

训练的输出被重定向到日志文件，终端里那个动态进度条不会刷新
（进度条靠 \\r 回车原地重绘，写进文件后就挤成一行了）。
所以单独做个小工具，从日志 + results.csv 里提取真正有用的数字。

用法：
    python scripts/train_status.py                 # 默认看 cig_v31_u960
    python scripts/train_status.py --name xxx      # 看别的训练
    python scripts/train_status.py --watch 30      # 每 30 秒刷新一次
"""
from __future__ import annotations

import argparse
import csv
import re
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def parse() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", default="D:/fire_runs")
    ap.add_argument("--name", default="cig_v31_u960")
    ap.add_argument("--log", default="")
    ap.add_argument("--watch", type=int, default=0, help="每 N 秒刷新（0=只打印一次）")
    return ap.parse_args()


def snapshot(project: str, name: str, log: str) -> str:
    run = Path(project) / name
    csv_path = run / "results.csv"
    lines: list[str] = []

    # ── 从 results.csv 拿每轮的正式指标（比解析进度条可靠） ──
    rows: list[dict] = []
    if csv_path.is_file():
        try:
            with csv_path.open(encoding="utf-8", errors="ignore") as f:
                rows = list(csv.DictReader(f))
        except Exception:
            rows = []

    total_epochs = 0
    args_yaml = run / "args.yaml"
    if args_yaml.is_file():
        m = re.search(r"^epochs:\s*(\d+)", args_yaml.read_text(encoding="utf-8", errors="ignore"), re.M)
        if m:
            total_epochs = int(m.group(1))

    if rows:
        last = rows[-1]
        def g(k: str) -> str:
            v = (last.get(k) or "").strip()
            return v if v else "-"

        cur = len(rows)
        lines.append(f"  轮次      {cur}/{total_epochs or '?'}")
        # ultralytics 的列名带前后空格，这里是去掉空格的版本
        def pick(*keys: str) -> str:
            for k, v in last.items():
                kk = (k or "").strip()
                if kk in keys:
                    return (v or "").strip() or "-"
            return "-"

        lines.append(f"  box_loss  {pick('train/box_loss')}")
        lines.append(f"  cls_loss  {pick('train/cls_loss')}")
        lines.append(f"  mAP50     {pick('metrics/mAP50(B)')}")
        lines.append(f"  mAP50-95  {pick('metrics/mAP50-95(B)')}")
        lines.append(f"  precision {pick('metrics/precision(B)')}")
        lines.append(f"  recall    {pick('metrics/recall(B)')}")

        # 最佳 mAP50 出现在哪一轮
        best_v, best_e = -1.0, 0
        for i, r in enumerate(rows, 1):
            for k, v in r.items():
                if (k or "").strip() == "metrics/mAP50(B)":
                    try:
                        fv = float(v)
                    except (TypeError, ValueError):
                        continue
                    if fv > best_v:
                        best_v, best_e = fv, i
        if best_e:
            lines.append(f"  最佳      第 {best_e} 轮  mAP50={best_v:.4f}")
    else:
        lines.append("  results.csv 还没生成（第 1 轮验证结束后才会有指标）")

    # ── 日志尾部：判断是否在跑、卡在哪 ──
    lp = Path(log) if log else ROOT / "logs" / "train_v31.log"
    if lp.is_file():
        txt = lp.read_text(encoding="utf-8", errors="ignore")
        # 进度条被 \r 挤在一起，取最后一段
        tail = txt[-4000:].replace("\r", "\n")
        m = re.findall(r"(\d+)/(\d+)\s+([\d.]+)s/it", tail)
        if m:
            i, n, s = m[-1]
            lines.append(f"  当前batch {i}/{n}   {s}s/it")
        if "OOM" in tail or "out of memory" in tail.lower():
            lines.append("  ⚠️ 显存不足")
        if "Traceback" in tail:
            lines.append("  ⚠️ 有异常，看日志尾部")
        mt = lp.stat().st_mtime
        age = time.time() - mt
        lines.append(f"  日志更新  {time.strftime('%H:%M:%S', time.localtime(mt))}"
                     f"（{age:.0f} 秒前）{'  ← 可能已停止' if age > 300 else ''}")

    return "\n".join(lines)


def main() -> int:
    a = parse()
    while True:
        print("=" * 62)
        print(f"  {a.name}")
        print("=" * 62)
        print(snapshot(a.project, a.name, a.log))
        if not a.watch:
            return 0
        time.sleep(a.watch)


if __name__ == "__main__":
    raise SystemExit(main())
