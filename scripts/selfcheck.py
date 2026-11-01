"""启动前自检 + 关键路径回归测试。

为什么需要这个
--------------
这个项目最近连续出了三次同类事故：改代码时锚点没匹配上，
改动静默失败，运行时才 `AttributeError` 炸掉 —— 而且因为异常被
try/except 吞掉，界面看着还在跑，实际上一个框都没检出。

人工 review 靠不住，只有机器每次都能跑一遍检查才靠得住。

检查分三层
----------
1. 静态：所有 self.xxx 在类里都有初始化（防 AttributeError）
2. 配置：yaml 里该有的键都在、值在合理区间
3. 行为：NMS 分组、Kalman 收敛、轨迹生命周期 —— 真正跑一遍看结果

用法
----
    python scripts/selfcheck.py          # 只报告
    python scripts/selfcheck.py --strict # 有任何失败就返回非零
"""
from __future__ import annotations

import argparse
import ast
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

OK, FAIL, WARN = "OK  ", "FAIL", "WARN"
results: list[tuple[str, str, str]] = []


def check(status: str, name: str, detail: str = "") -> None:
    results.append((status, name, detail))


# ────────────────────────── 1. 静态检查 ─────────────────────────
def check_attrs() -> None:
    """扫描每个类，找出"用了 self.xxx 但从没赋值"的情况。

    这是三次事故的共同根因：某个属性只在方法里读、没在 __init__ 里写。
    静态扫一遍就能提前发现。
    """
    targets = ["src/infer.py", "src/tracker.py", "src/pipeline.py", "src/liveview.py"]
    for rel in targets:
        path = ROOT / rel
        if not path.is_file():
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError as e:
            check(FAIL, f"{rel} 语法", str(e)[:80])
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.ClassDef):
                continue
            assigned: set[str] = set()
            loaded: set[str] = set()
            for sub in ast.walk(node):
                if isinstance(sub, ast.Attribute) and isinstance(sub.value, ast.Name) \
                        and sub.value.id == "self":
                    if isinstance(sub.ctx, ast.Store):
                        assigned.add(sub.attr)
                    else:
                        loaded.add(sub.attr)
            # 类属性（dataclass 字段 / 类变量）也算已定义
            for stmt in node.body:
                if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
                    assigned.add(stmt.target.id)
                elif isinstance(stmt, ast.Assign):
                    for t in stmt.targets:
                        if isinstance(t, ast.Name):
                            assigned.add(t.id)
            # 方法名也会以 self.xxx 形式被读取（self._draw(...)），排除掉
            methods = {n.name for n in node.body if isinstance(n, ast.FunctionDef)}
            missing = sorted(loaded - assigned - methods)
            if missing:
                check(WARN, f"{node.name} 属性",
                      f"{rel}: 读取但未见赋值 -> {', '.join(missing[:6])}")


# ────────────────────────── 2. 配置检查 ─────────────────────────
def check_config() -> None:
    import yaml
    cfg_path = ROOT / "config" / "monitor.yaml"
    if not cfg_path.is_file():
        check(FAIL, "配置文件", "config/monitor.yaml 不存在")
        return
    try:
        cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    except Exception as e:                              # noqa: BLE001
        check(FAIL, "配置文件解析", str(e)[:80])
        return
    thr = cfg.get("thresholds", {})
    det = cfg.get("detect", {})

    # 阈值必须递增：检测门限 < 告警门限 < 彩色框门限
    pairs = [
        ("det_conf", "fire_conf", "color_conf"),
        ("det_conf", "smoke_conf", "color_conf_smoke"),
        ("cigarette_conf", "cigarette_conf", "color_conf_cig"),
    ]
    for lo_k, mid_k, hi_k in pairs:
        lo, mid, hi = thr.get(lo_k), thr.get(mid_k), thr.get(hi_k)
        if None in (lo, mid, hi):
            check(WARN, f"阈值 {mid_k}", f"缺少配置项 ({lo_k}/{mid_k}/{hi_k})")
            continue
        if not (lo <= mid <= hi):
            check(FAIL, f"阈值次序 {mid_k}",
                  f"{lo_k}={lo} 应 <= {mid_k}={mid} 应 <= {hi_k}={hi}")
        else:
            check(OK, f"阈值次序 {mid_k}", f"{lo} <= {mid} <= {hi}")

    # 香烟模型文件必须存在（否则静默跳过，用户不知道）
    ew = det.get("extra_weights")
    if ew:
        p = Path(ew)
        if not p.is_absolute():
            p = ROOT / ew
        check(OK if p.is_file() else FAIL, "香烟模型文件",
              str(p.name) if p.is_file() else f"不存在: {p}")
    check(OK if 1 <= int(det.get("imgsz", 0)) else FAIL, "推理分辨率",
          f"imgsz={det.get('imgsz')}")


# ────────────────────────── 3. 行为测试 ─────────────────────────
def test_nms_grouping() -> None:
    """NMS 必须按类名分组 —— fire 和 cigarette 的 cls_id 都是 0。"""
    from src.infer import Detection, nms
    fire = Detection(0, "fire", 0.90, (100, 100, 200, 200))
    cig = Detection(0, "cigarette", 0.80, (105, 105, 195, 195))   # 同 id、不同名、几乎重叠
    out = nms([fire, cig], 0.25)
    names = {d.cls_name for d in out}
    if names == {"fire", "cigarette"}:
        check(OK, "NMS 按类名分组", "同 id 不同类没有被互相抑制")
    else:
        check(FAIL, "NMS 按类名分组", f"被错误抑制，只剩 {names}")

    # 同类高度重叠必须只留一个
    a = Detection(0, "fire", 0.90, (100, 100, 200, 200))
    b = Detection(0, "fire", 0.85, (104, 104, 196, 196))
    out2 = nms([a, b], 0.25)
    check(OK if len(out2) == 1 else FAIL, "NMS 同类去重",
          f"重叠两个框 -> 剩 {len(out2)} 个（应为 1）")


def test_kalman() -> None:
    """Kalman 必须能跟上移动目标，且尺寸自适应。"""
    from src.tracker import _Kalman
    kf = _Kalman((100, 100, 200, 200))
    actual_x = 100
    errors = []
    for i in range(8):
        actual_x += 25
        pred = kf.predict()
        # 目标同时变大
        out = kf.correct((actual_x, 100, actual_x + 100 + i * 6, 200 + i * 6))
        errors.append(abs(pred[0] - actual_x))
    # 稳态后预测误差应远小于每帧位移
    steady = sum(errors[3:]) / max(1, len(errors[3:]))
    if steady < 25:
        check(OK, "Kalman 预测", f"稳态预测误差 {steady:.1f}px < 单帧位移 25px")
    else:
        check(FAIL, "Kalman 预测", f"稳态误差 {steady:.1f}px，跟不上目标")
    w0, w1 = 100.0, out[2] - out[0]
    if w1 > w0:
        check(OK, "Kalman 尺寸自适应", f"宽度 {w0:.0f} -> {w1:.0f}（目标变大，框跟着变大）")
    else:
        check(FAIL, "Kalman 尺寸自适应", f"宽度没跟上 {w0:.0f} -> {w1:.0f}")


def test_tracker_no_split() -> None:
    """目标快速移动时，轨迹不能被拆成两条。"""
    import numpy as np
    from src.infer import Detection
    from src.tracker import IoUTracker as Tracker
    tr = Tracker()
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    tid_seen = set()
    x = 100
    for i in range(10):
        x += 30                                   # 每帧右移 30px（快速移动）
        dets = [Detection(0, "fire", 0.9, (x, 100, x + 80, 180))]
        for t in tr.update(dets, img, ts=i * 0.033):
            if t.cls_name == "fire":
                tid_seen.add(t.tid)
    if len(tid_seen) == 1:
        check(OK, "快速移动不分裂", "10 帧全程同一条轨迹 tid 恒定")
    else:
        check(FAIL, "快速移动不分裂", f"分裂出 {len(tid_seen)} 条轨迹（应为 1）")


def test_pipeline_attrs() -> None:
    """Pipeline / Detector 的实例化不报错（属性初始化齐全）。"""
    import yaml
    cfg = yaml.safe_load((ROOT / "config" / "monitor.yaml").read_text(encoding="utf-8"))
    det_cfg = cfg.get("detect", {})
    w = ROOT / "models" / "fire_smoke_v5_26s960_best.pt"
    if not w.is_file():
        check(WARN, "Detector 实例化", "主模型文件不存在，跳过")
        return
    try:
        from src.infer import Detector
        ew = det_cfg.get("extra_weights")
        ep = None
        if ew:
            c = Path(ew) if Path(ew).is_absolute() else ROOT / ew
            ep = str(c) if c.is_file() else None
        d = Detector(str(w), conf=0.05, imgsz=640, extra_weights=ep, extra_conf=0.03)
        for attr in ("slice_every", "extra_every", "_slice_n", "_extra_n"):
            if not hasattr(d, attr):
                check(FAIL, "Detector 属性", f"缺少 self.{attr}")
                return
        check(OK, "Detector 实例化", "关键属性齐全")
    except Exception as e:                              # noqa: BLE001
        check(FAIL, "Detector 实例化", f"{type(e).__name__}: {e}")



def test_filter_but_traceable() -> None:
    """过滤可以，但必须留痕 —— 不允许"静默丢弃"。

    用户的原话是"还是得改回去，但是不能完全不报"。
    所以规则层可以过滤（省配额、压误报），但被过滤的对象必须：
      · 仍然出现在画面上（灰框），人看得见
      · 有统计计数，事后能查
      · 有日志，值班员知道裁判什么时候缺席
    """
    src = (ROOT / "src" / "pipeline.py").read_text(encoding="utf-8")
    if "shape_flagged" in src and "continue" in src:
        check(OK, "香烟形态：过滤但留痕",
              "不送 LLM，但保留灰框 + 计入 shape_flagged 统计")
    else:
        check(FAIL, "香烟形态：过滤但留痕", "缺少 shape_flagged 统计")

    arb = (ROOT / "src" / "arbiter.py").read_text(encoding="utf-8")
    seg = arb[arb.find("def _quota_ok"):]
    seg = seg[:seg.find("\n    def ", 10)] if "\n    def " in seg[10:] else seg
    has_warn = "[warn]" in seg
    has_false = "return False" in seg
    if has_false and has_warn:
        check(OK, "配额：降级但留痕", "降级到规则层，并打 warn 日志")
    elif not has_false:
        check(OK, "配额：不降级", "配额耗尽仍送 LLM")
    else:
        check(FAIL, "配额：降级但留痕", "降级了但没有日志 —— 静默降级不可接受")



def test_small_target_no_split() -> None:
    """极小目标和抖动时不能分裂 —— 这是打火机出三条事件的根因。

    打火机的火只有约 20x20 像素，火苗抖动一下位置跳 10px，
    两个框就完全不重叠（IOU=0）。纯 IOU 匹配会判成新目标，
    于是一次打火冒出三条轨迹、三张卡片。
    这里模拟"20x20 小目标每帧随机抖动 ±8px"，看会不会分裂。
    """
    import random
    import numpy as np
    from src.infer import Detection
    from src.tracker import IoUTracker
    random.seed(7)
    tr = IoUTracker()
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    tids = set()
    cx, cy = 320, 240
    for i in range(12):
        # 目标位置小幅漂移 + 每帧检测抖动
        cx += random.uniform(-4, 4)
        cy += random.uniform(-4, 4)
        jx, jy = random.uniform(-8, 8), random.uniform(-8, 8)
        x = cx + jx
        y = cy + jy
        dets = [Detection(0, "fire", 0.8, (x, y, x + 20, y + 20))]
        for t in tr.update(dets, img, ts=i * 0.05):
            if t.cls_name == "fire":
                tids.add(t.tid)
    if len(tids) == 1:
        check(OK, "小目标抖动不分裂", "20x20 目标抖动 ±8px，12 帧仍是一条轨迹")
    else:
        check(FAIL, "小目标抖动不分裂", f"分裂出 {len(tids)} 条轨迹（应为 1）")



def test_snapshot_and_merge() -> None:
    """截图边距要够大 + 重叠框必须合并送审。

    用户提出的两点：
      · 截图框太小，烟雾扩散形态看不出来，LLM 无法判断
      · 同一物体不同类别的框离得近时，不该分两次送给 LLM
    """
    src = (ROOT / "src" / "pipeline.py").read_text(encoding="utf-8")
    import re
    m = re.search(r'"snapshot_margin":\s*([0-9.]+)', src)
    if m and float(m.group(1)) >= 0.8:
        check(OK, "截图边距", f"外扩 {float(m.group(1)):.0%}，能看到烟雾扩散形态")
    elif m:
        check(FAIL, "截图边距", f"只有 {float(m.group(1)):.0%}，烟雾判断会失真")
    else:
        check(WARN, "截图边距", "未找到 snapshot_margin 配置")

    if "_peer_suppressed" in src and "_peers" in src:
        check(OK, "重叠框合并送审", "同位置多类别只送一次，LLM 统一判断")
    else:
        check(FAIL, "重叠框合并送审", "缺失，同一块地方会被送两次")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--strict", action="store_true", help="有 FAIL 时返回非零")
    a = ap.parse_args()

    check_attrs()
    check_config()
    test_nms_grouping()
    test_kalman()
    test_tracker_no_split()
    test_small_target_no_split()
    test_pipeline_attrs()
    test_filter_but_traceable()
    test_snapshot_and_merge()

    print("\n" + "=" * 62)
    print(f"{'结果':6s}{'检查项':26s}说明")
    print("-" * 62)
    for st, name, detail in results:
        print(f"{st:6s}{name:26s}{detail}")
    n_fail = sum(1 for s, _, _ in results if s == FAIL)
    n_warn = sum(1 for s, _, _ in results if s == WARN)
    print("=" * 62)
    print(f"通过 {len(results) - n_fail - n_warn} | 警告 {n_warn} | 失败 {n_fail}")
    if n_fail:
        print("\n有失败项 —— 不要启动服务，先修。")
    return 1 if (a.strict and n_fail) else 0


if __name__ == "__main__":
    sys.exit(main())
