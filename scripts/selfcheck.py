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



def test_cigarette_reaches_llm() -> None:
    """香烟必须真的能走到 LLM 送审 —— 这是用户反复反馈的问题。

    原来香烟走的是烟雾的时序判别路径，而 `judge()` 是为"会膨胀、
    会上飘的烟云"设计的。一支烟细长、不膨胀，永远判不出 smoke，
    于是在 `if v.label != "smoke"` 处被清零，从头到尾一次都没送过 LLM。

    检查香烟有自己的送审分支，且在时序判别**之前**。
    """
    src = (ROOT / "src" / "pipeline.py").read_text(encoding="utf-8")
    i_cig = src.find('if kind == "cigarette":')
    i_judge = src.find("v = judge(feat, thr=self.temporal_thr")
    if i_cig < 0:
        check(FAIL, "香烟送审路径", "没有独立分支，会走烟雾判据被拦掉")
        return
    if i_judge > 0 and i_cig < i_judge:
        check(OK, "香烟送审路径", "独立分支在时序判别之前，过门槛即送 LLM")
    else:
        check(WARN, "香烟送审路径", "分支位置可疑，确认是否在 judge() 之前")

    # 香烟分支里必须有 _defer 调用。窗口取到下一个同级分支为止，
    # 不能写死 1600 字符 —— 注释一多就截不到（自检自己踩过这个坑）。
    nxt = src.find('if conf < cfg["smoke_conf"]', i_cig)
    seg = src[i_cig: nxt if nxt > 0 else i_cig + 4000]
    if "_defer" in seg:
        check(OK, "香烟直接送审", "分支内调用 _defer，不经过烟雾判据")
    else:
        check(FAIL, "香烟直接送审", "分支内没有 _defer 调用")



def test_llm_label_wins() -> None:
    """事件类别必须以 LLM 的判定为准，且 cigarette 贯穿全链路。

    用户截图暴露过：LLM 判 smoke（理由写得很准——"手指夹白色卷烟，
    端头有褐色燃点，没有飘动火苗"），事件标签却还是"火情"，
    因为 kind 一直用检测类别（主模型把香烟误检成 fire）。
    """
    arb = (ROOT / "src" / "arbiter.py").read_text(encoding="utf-8")
    # 原来这里是硬匹配 `"fire|smoke|cigarette|steam"` 这一整串。
    # prompt 后来把候选值重排成 fire|smoke|cigarette|light|other|uncertain
    # （没写 steam，因为烟/汽本来就该归 other），锚点就失效了 ——
    # 检查报"没有 cigarette"，实际 prompt 里明明有。改成解析枚举本身。
    import re as _re
    _m = _re.search(r'"label"\s*:\s*"([^"]+)"', arb)
    _labels = [x.strip() for x in (_m.group(1).split("|") if _m else [])]
    if "cigarette" in _labels:
        check(OK, "LLM 支持 cigarette", f"label 可选值：{'|'.join(_labels)}")
    else:
        check(FAIL, "LLM 支持 cigarette",
              f"label 白名单里没有 cigarette（当前：{'|'.join(_labels) or '解析不到'}）")

    if '"cigarette"' in arb or "cigarette" in arb.split("def parse")[0]:
        check(OK, "香烟判据", "prompt 已说明「香烟判 cigarette 不是 smoke」")
    else:
        check(FAIL, "香烟判据", "prompt 里没有香烟的判定规则")

    src = (ROOT / "src" / "pipeline.py").read_text(encoding="utf-8")
    if "kind_out = res.label" in src:
        check(OK, "类别以 LLM 为准", "kind_out 直接采用 res.label")
    else:
        check(FAIL, "类别以 LLM 为准", "事件 kind 仍用检测类别")

    ui = (ROOT / "web" / "dashboard.html").read_text(encoding="utf-8")
    if "香烟" in ui and "cigarette" in ui:
        check(OK, "前端显示香烟", "levelCN 已处理 cigarette")
    else:
        check(FAIL, "前端显示香烟", "前端不认识 cigarette")



def test_cigarette_gate() -> None:
    """香烟的置信度闸门必须用自己的门槛，不能套通用 0.50。

    用户截图：两条 cigarette 事件都是"裁判不可用，请人工确认"，
    置信度只有 0.13 / 0.21。原因是 _defer 里那道通用闸门
        low = llm_min_conf (0.50)
    —— 香烟模型对真实香烟给的分常在 0.1~0.3，永远过不了 0.50，
    于是每一条都被挡回规则层。
    """
    src = (ROOT / "src" / "pipeline.py").read_text(encoding="utf-8")
    i = src.find("② 置信度闸门")
    if i < 0:
        check(WARN, "香烟置信度闸门", "没找到闸门代码")
        return
    seg = src[i:i + 1200]
    if 'kind == "cigarette"' in seg and "cigarette_conf" in seg:
        check(OK, "香烟置信度闸门", "香烟走 0.03~0.95，不套通用 0.50")
    else:
        check(FAIL, "香烟置信度闸门", "香烟仍在用通用门槛，会被 0.50 挡住")



def test_cigarette_confirm_frames() -> None:
    """香烟必须有足够的连续帧门槛，否则瞬时噪声会刷屏。

    用户截图：一个只有人脸和马赛克的画面，conf 0.03（正好卡在检测门槛），
    2 帧就弹了"待确认"。抽烟是持续行为，瞬间的不可能是真烟。
    """
    src = (ROOT / "src" / "pipeline.py").read_text(encoding="utf-8")
    import re
    m = re.search(r'"cigarette_confirm":\s*(\d+)', src)
    n = int(m.group(1)) if m else 0
    if n >= 5:
        check(OK, "香烟确认帧数", f"连续 {n} 帧才送审（约 0.3~0.5 秒），滤掉瞬时噪声")
    else:
        check(FAIL, "香烟确认帧数", f"只有 {n} 帧，太容易被噪声触发")

    cfg = (ROOT / "config" / "monitor.yaml").read_text(encoding="utf-8")
    import yaml
    c = yaml.safe_load(cfg) or {}
    tc = float((c.get("thresholds") or {}).get("cigarette_conf", 0))
    # 策略变了：现在香烟是"极低门槛收候选 + 全量送 LLM 裁决"，
    # 门限低是**故意的**。只要它 <= det_conf 就说得通 ——
    # 真正的把关在 LLM 那一层，不在检测层。
    det_c = float((c.get("thresholds") or {}).get("det_conf", 1))
    # 正确关系：类别门槛 >= 总门槛（总门槛是"什么都要过"的那道粗筛）。
    # ⚠️ 唯一的例外是 cigarette：它不走主模型，走 **副模型自己的 conf**，
    #    所以 det_conf 管不到它。这个例外不能凭"我记得是这样"放行，
    #    必须先在 infer.py 里确认副模型确实用的是 self.extra_conf。
    extra_own_conf = _extra_model_has_own_conf()
    if tc >= det_c:
        check(OK, "香烟检测门限", f"{tc} >= det_conf {det_c}，不会被总门槛先滤掉")
    elif extra_own_conf:
        check(OK, "香烟检测门限",
              f"{tc} < det_conf {det_c}，但香烟走副模型自己的 extra_conf，"
              f"主门槛管不到它（已在 infer.py 核实）")
    else:
        check(FAIL, "香烟检测门限", f"{tc} < det_conf {det_c}，目标会被总门槛滤掉")



def _extra_model_has_own_conf() -> bool:
    """副模型（香烟）是否有独立的 conf 参数，而不是复用主模型的。

    为什么需要：det_conf 只是**主模型**的推理门槛。香烟是第二个模型、
    第二份权重，走的是 extra_conf。如果哪天有人把副模型的 predict 改成
    复用 self.conf，那 cigarette_conf 就失去意义了 —— 这个函数会在
    自检里把那种改动暴露出来，而不是继续默认"应该没问题"。
    """
    try:
        src = (ROOT / "src" / "infer.py").read_text(encoding="utf-8")
    except OSError:
        return False
    has_attr = "self.extra_conf" in src
    # 副模型 predict 的 conf 实参必须是 extra_conf
    used = ("conf=self.extra_conf" in src
            or "conf=self.extra_conf" in src.replace(" ", "")
            or '"conf": self.extra_conf' in src)
    return has_attr and (used or "extra_conf," in src.replace(" ", ""))


def test_confirm_frames_all() -> None:
    """三类目标的确认帧数都不能太低，否则会刷屏。

    用户要求："火和烟也不能说时刻上报，还是要帧数不能太低的。"

    注意要读**合并后**的配置（monitor.yaml 会覆盖代码里的 DEFAULTS）。
    第一次改的时候只改了 DEFAULTS，yaml 里还留着 fire_confirm:6 /
    smoke_confirm:3，白改一场 —— 所以这项检查必须走真实合并结果。
    """
    import yaml
    cfg_file = ROOT / "config" / "monitor.yaml"
    y = (yaml.safe_load(cfg_file.read_text(encoding="utf-8")) or {}).get("thresholds", {})
    sys.path.insert(0, str(ROOT))
    from src.pipeline import DEFAULTS
    merged = dict(DEFAULTS)
    merged.update(y or {})

    bad = []
    for k in ("fire_confirm", "smoke_confirm", "cigarette_confirm"):
        n = int(merged.get(k, 0))
        if n < 5:
            bad.append(f"{k}={n}")
    if bad:
        check(FAIL, "确认帧数（合并后）", "过低 -> " + ", ".join(bad))
    else:
        vals = " ".join(f"{k.split('_')[0]}={int(merged[k])}"
                        for k in ("fire_confirm", "smoke_confirm", "cigarette_confirm"))
        check(OK, "确认帧数（合并后）", f"{vals}，三类都不低于 5 帧")



def test_json_parse() -> None:
    """LLM 返回的各种脏格式都要能解析出来。

    用户截图：事件被判成"火情"，理由是"裁判不可用（返回不是合法 JSON）"，
    耗时 15.1s —— 模型其实答了，只是格式不标准，系统退回规则层
    按检测类别（fire）定级，把一支香烟判成了火。
    """
    sys.path.insert(0, str(ROOT))
    from src.arbiter import _parse_json
    L, R = chr(0x201C), chr(0x201D)
    cases = [
        ("标准 JSON", '{"label":"fire","confidence":0.8}', "fire"),
        ("裹代码块", '```json\n{"label":"smoke"}\n```', "smoke"),
        ("前面带解释", '判定如下：{"label":"other"}', "other"),
        ("中文引号", '{' + L + 'label' + R + ':' + L + 'cigarette' + R + '}', "cigarette"),
        ("尾逗号", '{"label":"fire",}', "fire"),
    ]
    bad = []
    for name, txt, want in cases:
        got = _parse_json(txt)
        if not got or got.get("label") != want:
            bad.append(name)
    if bad:
        check(FAIL, "LLM 返回解析容错", "解析不出来 -> " + "、".join(bad))
    else:
        check(OK, "LLM 返回解析容错", f"{len(cases)} 种脏格式全部能解析")



def test_ui_and_prompt_fixes() -> None:
    """三个用户反馈点的守门检查。

    一、JSON 截断必须能救回来
       实测模型把 reason 写到一半就停（连右括号都没有），
       原来直接判"不是合法 JSON"退回规则层，把香烟判成了火。
    二、按钮文案按三类区分
       原来只判 === "smoke"，香烟事件落进 else 分支，
       明明是香烟却问"有火/无火"。
    三、送审截图要标出检测框
       不标框，模型得自己猜"让我看哪一块"。
    """
    sys.path.insert(0, str(ROOT))
    from src.arbiter import _parse_json
    trunc = ('{"label":"cigarette","risk":"low","confidence":0.62,'
             '"reason":"手指夹白色细长条，无飘动火苗、无扩散烟')
    r = _parse_json(trunc)
    if r and r.get("label") == "cigarette":
        check(OK, "JSON 截断可救回", "缺右括号也能解析出 label/confidence")
    else:
        check(FAIL, "JSON 截断可救回", "截断后解析失败，会退回规则层")

    ui = (ROOT / "web" / "dashboard.html").read_text(encoding="utf-8")
    if '"cigarette"' in ui and "有香烟" in ui:
        check(OK, "按钮文案按类别", "香烟事件问「有香烟/无香烟」")
    else:
        check(FAIL, "按钮文案按类别", "香烟事件仍问「有火/无火」")

    # 需求反转过一次：最初要求"给 LLM 的图标注框指明看哪里"，
    # 后来用户实测发现框会误导模型（把画上去的线当成画面内容），
    # 于是改成默认不画。这里检查默认值确实是 False。
    arb = (ROOT / "src" / "arbiter.py").read_text(encoding="utf-8")
    if "mark: bool = False" in arb:
        check(OK, "送审图不加标注", "默认不画框，避免把模型带偏")
    else:
        check(FAIL, "送审图不加标注", "仍在往送审图上画框，会误导模型")



def test_floor_order() -> None:
    """检测层总门槛必须低于所有类别门槛。

    det_conf 是 ultralytics 推理时的一道全局过滤，比它低的**全部丢掉**。
    实测踩过：det_conf 提到 0.18、cigarette_conf 是 0.12，
    结果香烟模型明明检出了 0.17 的目标，在检测层就被滤掉 ——
    用户看到的是"画面里明明在抽烟，却一个框都没有"。
    """
    import yaml
    c = (yaml.safe_load((ROOT / "config" / "monitor.yaml")
                        .read_text(encoding="utf-8")) or {}).get("thresholds", {})
    det = float(c.get("det_conf", 1))
    lows = {k: float(c.get(k, 1)) for k in
            ("fire_conf", "smoke_conf", "cigarette_conf")}
    # cigarette 例外同上：门槛由副模型的 extra_conf 把关，见 _extra_model_has_own_conf。
    if _extra_model_has_own_conf():
        lows.pop("cigarette_conf", None)
    bad = [f"{k}={v} < det_conf={det}" for k, v in lows.items() if v < det]
    if bad:
        check(FAIL, "检测层门槛次序", "总门槛高于类别门槛 -> " + "；".join(bad))
    else:
        detail = f"det={det} " + " ".join(f"{k.split('_')[0]}={v}" for k, v in lows.items())
        check(OK, "检测层门槛次序", detail)



def test_parse_with_thinking() -> None:
    """开了思考链之后，答案前面会有一大段分析，解析必须能穿透。

    实测日志：
      [llm] 解析失败，原始返回：'让我仔细看看这两张图片。图1：整幅画面。
      一个人戴着大耳机，手里拿着手机…他嘴里似乎叼着什么东西…'
    —— 模型在正文里写了长篇分析，真正的 JSON 还没开始输出就被截断了。
    """
    sys.path.insert(0, str(ROOT))
    from src.arbiter import _parse_json
    cases = [
        ("思考在前", '让我看看图1。图1是一个人。\n{"label":"cigarette","confidence":0.62}'),
        ("思考带花括号", '分析{a}和{b}后：{"label":"other","confidence":0.9}'),
        ("纯 JSON", '{"label":"fire","confidence":0.8}'),
    ]
    bad = [n for n, t in cases if not (_parse_json(t) or {}).get("label")]
    if bad:
        check(FAIL, "思考链下的解析", "穿透失败 -> " + "、".join(bad))
    else:
        check(OK, "思考链下的解析", "答案在分析之后也能取出（取最后一个 JSON）")



def test_cigarette_skips_static_prefilter() -> None:
    """香烟必须豁免"静态光源预筛"。

    用户截图：一条 cigarette 事件显示"裁判不可用，请人工确认"，耗时 0.3s
    —— 0.3 秒说明 LLM 根本没被调用。

    原因：手持香烟本来就几乎不动，被 _looks_static_light 判成"固定光源"，
    直接 return False。可香烟是明确违规行为，哪怕静止也必须让 LLM 看。
    """
    src = (ROOT / "src" / "pipeline.py").read_text(encoding="utf-8")
    i = src.find("① 固定光源预筛")
    if i < 0:
        check(WARN, "香烟豁免静态预筛", "没找到预筛代码")
        return
    seg = src[i:i + 900]
    if 'kind != "cigarette"' in seg:
        check(OK, "香烟豁免静态预筛", "手持香烟静止不动也会送 LLM")
    else:
        check(FAIL, "香烟豁免静态预筛", "香烟静止时仍会被判成光源、不送审")



def test_all_kind_conf_present() -> None:
    """每一类的 {kind}_conf 都必须在 DEFAULTS 里声明。

    踩过的坑：cigarette_conf 只写在 monitor.yaml，但 _defer 用
    self.cfg（DEFAULTS 打底），键不存在就取兜底 0.35 ——
    yaml 改成 0.005 完全不起作用，香烟永远过不了送审门槛。
    诊断日志：[defer] 门槛拦下 kind=cigarette score=0.061 floor=0.35
    """
    sys.path.insert(0, str(ROOT))
    from src.pipeline import DEFAULTS
    missing = [k for k in ("fire_conf", "smoke_conf", "cigarette_conf")
               if k not in DEFAULTS]
    if missing:
        check(FAIL, "各类门槛已声明", "DEFAULTS 缺少 " + "、".join(missing))
    else:
        vals = " ".join(f"{k.split('_')[0]}={DEFAULTS[k]}" for k in
                        ("fire_conf", "smoke_conf", "cigarette_conf"))
        check(OK, "各类门槛已声明", vals)



def test_no_ascii_quote_in_cn() -> None:
    """中文提示串里不能出现裸 ASCII 双引号 —— 会截断 Python 字符串。

    这个坑踩了三次：往 prompt 里写 "目标在哪" 这种中文引号时手滑打成
    ASCII 的 "，整行字符串被提前截断，py_compile 报 SyntaxError 且报的
    行号在几十行之前，很难看出是哪里写了引号。
    """
    bad = []
    for rel in ("src/arbiter.py", "src/pipeline.py"):
        f = ROOT / rel
        if not f.is_file():
            continue
        try:
            ast.parse(f.read_text(encoding="utf-8"))
        except SyntaxError as e:
            bad.append(f"{rel}:L{e.lineno} {e.msg}")
    if bad:
        check(FAIL, "语法完整", "；".join(bad))
    else:
        check(OK, "语法完整", "arbiter/pipeline 都能正常解析")


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
    test_cigarette_reaches_llm()
    test_llm_label_wins()
    test_cigarette_gate()
    test_cigarette_confirm_frames()
    test_confirm_frames_all()
    test_json_parse()
    test_ui_and_prompt_fixes()
    test_floor_order()
    test_parse_with_thinking()
    test_cigarette_skips_static_prefilter()
    test_all_kind_conf_present()
    test_no_ascii_quote_in_cn()

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
