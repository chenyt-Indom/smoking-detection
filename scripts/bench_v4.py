"""微基准：只测「单迭代耗时 + 显存峰值」，不做验证、不用全量数据。

跑法：python scripts/bench_v4.py
输出：每个配置的 s/it、显存峰值、折算的单轮耗时与 30 轮总时长
"""
from __future__ import annotations

import re
import subprocess
import sys
import tempfile
from pathlib import Path

PY = sys.executable

CONFIGS = [(6, 960), (8, 960)]

CODE = r"""
import sys
from ultralytics import YOLO
bs, sz = int(sys.argv[1]), int(sys.argv[2])
m = YOLO('yolo26s.pt')
m.train(
    data='E:/datasets/fire_smoke_v4/dataset.yaml',
    epochs=1, imgsz=sz, batch=bs, fraction=0.004, val=False, plots=False,
    workers=4, device=0, project='D:/fire_runs/bench', name=f'b{bs}_{sz}',
    exist_ok=True, verbose=True, seed=0, amp=True, multi_scale=0.5,
    mosaic=1.0, close_mosaic=20,
)
"""

N_TRAIN = 41857


def run(bs: int, sz: int):
    with tempfile.NamedTemporaryFile("w+", suffix=".log", delete=False, encoding="utf-8", errors="ignore") as fh:
        log = fh.name
    with open(log, "w", encoding="utf-8", errors="ignore") as out:
        subprocess.run([PY, "-c", CODE, str(bs), str(sz)], stdout=out, stderr=subprocess.STDOUT, timeout=900)
    txt = Path(log).read_text(encoding="utf-8", errors="ignore").replace("\r", "\n")
    sp = [float(x) for x in re.findall(r"([\d.]+)s/it", txt)]
    mm = [float(x) for x in re.findall(r"\s([\d.]+)G\s+[\d.]+\s+[\d.]+\s+[\d.]+\s+\d+\s+\d+:", txt)]
    sit = sorted(sp[len(sp) // 2:])[max(0, len(sp[len(sp) // 2:]) // 2)] if sp else None
    return bs, sz, sit, (max(mm) if mm else None)


def main() -> int:
    print(f"{'batch':>6}{'imgsz':>7}{'显存G':>8}{'s/it':>8}{'迭代/轮':>9}{'分钟/轮':>9}{'30轮小时':>10}", flush=True)
    for bs, sz in CONFIGS:
        try:
            b, s, sit, mem = run(bs, sz)
        except subprocess.TimeoutExpired:
            print(f"{bs:>6}{sz:>7}   超时", flush=True)
            continue
        if not sit:
            print(f"{bs:>6}{sz:>7}{mem if mem else '-':>8}   未取到速度", flush=True)
            continue
        iters = N_TRAIN // b
        mpe = iters * sit / 60
        print(f"{b:>6}{s:>7}{mem:>8}{sit:>8.3f}{iters:>9}{mpe:>9.1f}{mpe * 30 / 60:>10.1f}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
