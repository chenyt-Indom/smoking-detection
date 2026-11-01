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



def test_all_go_to_llm() -> None:
    """过阈值的目标必须全部送 LLM —— 规则层不得自作主张拦截。

    用户明确要求："所有内容务必都得有 LLM 裁判，不能自作主张，
    特别是香烟，必须不能漏。"

    这里检查两个具体的拦截点是否已经放开：
      · 香烟形态校验（原来是 continue，现在只标记）
      · LLM 配额耗尽（原来降级到规则层，现在照送）
    """
    src = (ROOT / "src" / "pipeline.py").read_text(encoding="utf-8")
    if "shape_flagged" in src and "shape_rejected" not in src:
        check(OK, "香烟形态不拦截", "形态不符只标记，仍送 LLM")
    else:
        check(FAIL, "香烟形态不拦截", "仍在用 shape_rejected 直接 continue")

    arb = (ROOT / "src" / "arbiter.py").read_text(encoding="utf-8")
    # _quota_ok 必须无条件返回 True
    seg = arb[arb.find("def _quota_ok"):]
    seg = seg[:seg.find("def ", 10)] if "def " in seg[10:] else seg
    if "return True" in seg and "return False" not in seg:
        check(OK, "配额不拦截送审", "配额耗尽只是提醒，仍送 LLM")
    else:
        check(FAIL, "配额不拦截送审", "_quota_ok 仍会返回 False 拦截")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--strict", action="store_true", help="有 FAIL 时返回非零")
    a = ap.parse_args()

    check_attrs()
    check_config()
    test_nms_grouping()
    test_kalman()
    test_tracker_no_split()
    test_pipeline_attrs()
    test_all_go_to_llm()

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
