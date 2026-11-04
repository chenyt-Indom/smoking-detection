"""导出 FP32 的 TensorRT 引擎（固定 imgsz=960、固定 batch=1）。

为什么单独写一个脚本
-------------------
线上默认用的是 FP16 引擎（`half=True`），因为快。但实测发现 FP16 会让
**置信度落在门槛附近的检测**在相邻帧之间翻来覆去地进出 —— 表现就是
"框一直闪"。用户选了这条路：宁可慢一点，也要和 .pt 一样的判定稳定性。

所以导出 FP32 引擎，命名规则不变（同目录同名），`_prefer_engine()` 会自动
挑到它，不需要改任何配置或代码。

⚠️ 三个硬约束（都踩过）
0. **必须在纯 ASCII 路径下导**。TensorRT 的 ONNX 解析器打不开含中文的路径 ——
   本项目根目录是 `D://视觉安防系统`，直接在原地导会得到
   `RuntimeError: failed to load ONNX file`（而 ONNX 其实已经生成好了，
   报错信息里还写着"Input file cannot be found"，极具误导性）。
   所以脚本先把 .pt 拷到 `D://trt_export//` 再导出，最后把 .engine 搬回来。
1. **必须 dynamic=False**。TensorRT 11 + ultralytics 的动态 batch 会
   `CUDA error: an illegal memory access was encountered`，实测崩过两次。
   引擎因此固定 batch=1，批量场景由 Detector._bp() 在调用侧逐张循环。
2. **必须 imgsz 与线上一致（960）**。引擎是定形编译的，尺寸不符会
   `AssertionError: input size ... not equal to max model size`，
   而且该异常会被上层 try/except 吞掉，表现成"检测全空"而不是报错。

用法
----
    python scripts/export_fp32_engine.py            # 两个模型都导
    python scripts/export_fp32_engine.py --only cig
"""
from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "models"

TARGETS = {
    "main": MODELS / "fire_smoke_v5_26s960_best.pt",
    "cig": MODELS / "cig_v32_smoking_best.pt",
}
IMGSZ = 960


def export_one(tag: str, pt: Path) -> bool:
    from ultralytics import YOLO

    eng = pt.with_suffix(".engine")
    backup_dir = MODELS / "_engine_fp16"
    # 先备份旧的（FP16）引擎：万一 FP32 太慢或效果不好，改个名就能回退
    if eng.is_file():
        backup_dir.mkdir(exist_ok=True)
        dst = backup_dir / eng.name
        if not dst.exists():
            shutil.move(str(eng), str(dst))
            print(f"  [备份] 旧引擎 -> {dst.relative_to(ROOT)}")
        else:
            eng.unlink()          # 备份已存在，直接删掉待重导
            print("  [备份] 已存在同名备份，删除待重导的旧引擎")

    # 关键：拷到 ASCII 临时目录再导（TensorRT 读不了中文路径）
    tmp = Path("D:/trt_export")
    tmp.mkdir(parents=True, exist_ok=True)
    tmp_pt = tmp / pt.name
    shutil.copy2(str(pt), str(tmp_pt))
    print(f"  [导出] {pt.name} -> FP32 engine（imgsz={IMGSZ}, batch=1, half=False）",
          flush=True)
    print(f"         在 ASCII 路径 {tmp} 下编译（TensorRT 不支持中文路径）", flush=True)
    t0 = time.time()
    m = YOLO(str(tmp_pt))
    out = Path(str(m.export(format="engine", imgsz=IMGSZ, half=False,
                            dynamic=False, batch=1, device=0)))
    ok = out.is_file()
    if ok:
        # 搬回 models/ 并覆盖
        shutil.copy2(str(out), str(eng))
        size = eng.stat().st_size / 1e6
        # FP32 引擎体积应约为 FP16 的 1.8~2.0 倍，差太多说明 half 没生效
        ratio = size / 20.8 if tag == "main" else size / 25.5
        print(f"  [完成] {eng.name}  {size:.1f}MB  {time.time() - t0:.0f}s  "
              f"（约为 FP16 的 {ratio:.1f} 倍；<1.5 倍要怀疑 half 没生效）",
              flush=True)
    else:
        print(f"  [失败] {time.time() - t0:.0f}s", flush=True)
    return ok


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=list(TARGETS), default=None)
    args = ap.parse_args()

    tags = [args.only] if args.only else list(TARGETS)
    print("=" * 78)
    print("导出 FP32 TensorRT 引擎（目标是消掉 FP16 带来的门槛抖动）")
    print("=" * 78)
    failed = []
    for tag in tags:
        pt = TARGETS[tag]
        if not pt.is_file():
            print(f"  [跳过] 找不到 {pt}")
            failed.append(tag)
            continue
        print(f"\n【{tag}】{pt.name}")
        if not export_one(tag, pt):
            failed.append(tag)
    print()
    print("=" * 78)
    if failed:
        print(f"失败：{failed}")
        sys.exit(1)
    print("全部导出成功。重启监控即可生效（检测器会自动优先用 .engine）。")
    print("回退到 FP16：把 models/_engine_fp16/*.engine 拷回 models/ 覆盖。")


if __name__ == "__main__":
    main()
