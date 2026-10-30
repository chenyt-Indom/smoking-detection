"""隔离测试：multi_scale 是否拖慢训练（同机、同数据、只改这一个参数）。"""
from __future__ import annotations

import io
import re
import sys
from contextlib import redirect_stdout

from ultralytics import YOLO

DATA = "E:/datasets/fire_smoke_v4/dataset.yaml"


def one(ms: float) -> tuple[float, float | None]:
    buf = io.StringIO()
    m = YOLO("yolo26s.pt")
    with redirect_stdout(buf):
        m.train(
            data=DATA, epochs=1, imgsz=960, batch=6, fraction=0.002, val=False,
            plots=False, workers=4, device=0, project="D:/fire_runs/bench_ms",
            name=f"ms{ms}", exist_ok=True, verbose=True, seed=0, amp=True,
            multi_scale=ms, mosaic=1.0,
        )
    txt = buf.getvalue().replace("\r", "\n")
    sp = [float(x) for x in re.findall(r"([\d.]+)s/it", txt)]
    mm = [float(x) for x in re.findall(r"\s([\d.]+)G\s+[\d.]+\s+[\d.]+\s+[\d.]+\s+\d+\s+\d+:", txt)]
    if not sp:
        return -1.0, (max(mm) if mm else None)
    tail = sp[len(sp) // 3:]
    return round(sorted(tail)[len(tail) // 2], 3), (max(mm) if mm else None)


if __name__ == "__main__":
    for ms in (0.0, 0.5):
        sit, mem = one(ms)
        print(f"RESULT multi_scale={ms}: {sit} s/it, 显存峰值 {mem} G", flush=True)
