"""多路摄像头监控：一个检测器，轮流跑多路。

为什么不是"每路一个进程"
------------------------
GPU 只有一块，多进程各自加载一份模型会直接把 12GB 显存吃爆，而且 CUDA 上下文
不能跨进程共享。所以这里走单进程多路：Detector 只建一个，各路轮流推理。

代价与取舍
----------
检测是串行的，N 路时每路的检测频率约为单路的 1/N。这是物理约束，不是实现问题。
好消息是烟雾是慢过程，5fps 的检测频率对"十几秒才铺开"的烟完全够用；
画面则走独立发布线程，不受检测频率限制（跟摄像头走）。

配置全在 config/monitor.yaml：摄像头列表、检测参数、阈值、LLM 闸门。
"""
from __future__ import annotations

import argparse
import sys
import threading
import time
from pathlib import Path

import cv2
import yaml

# 资源根目录：源码运行 = 项目根（scripts 的上一级）；PyInstaller 打包 = exe 所在目录
ROOT = (Path(sys.executable).resolve().parent if getattr(sys, "frozen", False)
        else Path(__file__).resolve().parents[1])
sys.path.insert(0, str(ROOT))

from src.arbiter import LlmArbiter, load_dotenv                    # noqa: E402
from src.infer import Detector                                    # noqa: E402
from src.liveview import LiveView                                 # noqa: E402
from src.notify import Notifier                                   # noqa: E402
from src.pipeline import Pipeline                                 # noqa: E402
from src.sources.base import CameraInfo                           # noqa: E402
from src.sources.fixed import FixedCameraSource                   # noqa: E402
from src.sources.latest import LatestFrameSource                   # noqa: E402

WEIGHTS = ROOT / "models" / "fire_smoke_v5_26s960_best.pt"
LEVEL_TAG = {"alarm": "火情", "warning": "烟情", "notice": "待确认", "cleared": "已排除"}


def load_cfg() -> dict:
    p = ROOT / "config" / "monitor.yaml"
    if not p.is_file():
        print(f"[x] 缺少配置 {p}")
        raise SystemExit(2)
    return yaml.safe_load(p.read_text(encoding="utf-8")) or {}


def port_busy(host: str, port: int) -> bool:
    """看板端口是否被占用 —— 第二道防重复启动（跨会话也有效）。"""
    import socket
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind((host, port))
        return False
    except OSError:
        return True
    finally:
        s.close()


def acquire_lock():
    """单实例锁（内核互斥量）。多路也只有一个进程，锁的语义不变。"""
    if sys.platform == "win32":
        try:
            import ctypes
            from ctypes import wintypes
            k32 = ctypes.WinDLL("kernel32", use_last_error=True)
            k32.CreateMutexW.restype = wintypes.HANDLE
            k32.CreateMutexW.argtypes = [wintypes.LPCVOID, wintypes.BOOL,
                                         wintypes.LPCWSTR]
            # ⚠️ 用 Global\ 而不是 Local\：Local 只在同一个登录会话内互斥。
            # 实测出现过两个实例（不同解释器启动、落在不同 session）都拿到锁，
            # 同时抢同一批摄像头 —— 画面疯狂闪烁、帧率减半，用户以为算法坏了。
            h = k32.CreateMutexW(None, True, "Global\\FireSmokeMonitor_Single")
            if ctypes.get_last_error() == 183:
                return None
            return h
        except Exception:                    # noqa: BLE001
            return True
    return True


class Worker:
    """一路摄像头：采集线程 + 独立流水线状态 + 独立画面发布线程。"""

    def __init__(self, cam: dict, detector, view: LiveView, pipe_cfg: dict,
                 arbiter, pub_interval: float, cig_conf: float = 0.03):
        self.id = str(cam["id"])
        self.name = str(cam.get("name") or self.id)
        self.index = int(cam["index"])
        self.view = view
        self.pipe_cfg = pipe_cfg
        self.pub_interval = pub_interval
        # 本路香烟检出的最低置信度。**按路设**，不是全局一个值 ——
        # 实测依据：cam_0 是 640x480，一支烟只有十几像素，香烟模型在那里
        # 给出的中位置信度只有 0.06、67 次检出里 0 次 >=0.20，全是噪声；
        # 这些噪声框会建轨迹、被送 LLM 复核（白烧额度）、还把界面数量刷得乱跳。
        # 而 cam_1 是 1280x720，香烟像素翻倍，才真正检得出。
        self.cig_conf = float(cig_conf)
        self.last_idx = -1
        self._dbg = 0
        self._stop = threading.Event()

        backend = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY
        cap = cv2.VideoCapture(self.index, backend)
        if not cap.isOpened():
            raise RuntimeError(f"摄像头 {self.index}（{self.name}）打不开")
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(cam.get("width", 640)))
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(cam.get("height", 480)))
        cap.set(cv2.CAP_PROP_FPS, 30)
        self.cap = cap
        print(f"  {self.id}（{self.name}）"
              f"{int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x"
              f"{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}"
              f" 已打开", flush=True)

        info = CameraInfo(cam_id=self.id, kind="fixed", fps=30.0)

        def frames():
            fails = 0
            while not self._stop.is_set():
                ok, img = cap.read()
                if not ok:
                    fails += 1
                    if fails > 300:
                        print(f"[{self.id}] 连续读帧失败，判定掉线", flush=True)
                        break
                    time.sleep(0.01)
                    continue
                fails = 0
                yield img

        self.src = LatestFrameSource(FixedCameraSource(info, frames=frames()))
        # 每路一个独立 Pipeline：轨迹/时序/区域记忆都是**按路**隔离的，
        # 共用一份会把 A 摄像头的轨迹和 B 摄像头的混在一起。
        self.pipe = Pipeline(detector, sliced=pipe_cfg.pop("sliced", False),
                             cfg=pipe_cfg, arbiter=arbiter)
        self.pipe._last_dets = []

        self._pub = threading.Thread(target=self._publish_loop,
                                     name=f"pub-{self.id}", daemon=True)
        self._pub.start()

    def _publish_loop(self) -> None:
        last_seq, last_pub = -1, 0.0
        n, t0 = 0, time.time()
        while not self._stop.is_set():
            pkt, seq = self.src.snapshot()
            now = time.time()
            if pkt is not None and seq != last_seq and (now - last_pub) >= self.pub_interval:
                last_seq, last_pub = seq, now
                n += 1
                try:
                    self.view.publish(pkt.image, getattr(self.pipe, "_last_dets", None) or [],
                                      frame_idx=pkt.idx, cam_id=self.id,
                                      fps=getattr(self.pipe, "det_fps", 0.0))
                except Exception:            # noqa: BLE001
                    pass
            time.sleep(0.004)

    def read_frame(self):
        """读一帧（不检测）。返回 pkt 或 None（无新帧/重复帧）。"""
        pkt = self.src.read()
        if pkt is None:
            return None
        if pkt.idx == self.last_idx:
            return None
        self.last_idx = pkt.idx
        return pkt

    def step_with(self, pkt, dets):
        """用外部 batch 检测的结果处理一帧，返回事件列表。"""
        # 香烟按**本路**门槛再筛一次（见 self.cig_conf）。
        # 在进追踪器之前筛，所以低分噪声不会建轨迹、不会进告警、不会送 LLM。
        # 只影响 cigarette —— 火/烟的判定链路一个字都没动。
        if self.cig_conf > 0:
            dets = [d for d in dets
                    if not ("cig" in str(getattr(d, "cls_name", "")).lower()
                            and float(getattr(d, "conf", 0.0)) < self.cig_conf)]
        res = self.pipe.process(pkt, dets)
        self.pipe._last_dets = res.dets
        self.pipe._det_ts.append(time.time())
        if len(self.pipe._det_ts) >= 4:
            span = self.pipe._det_ts[-1] - self.pipe._det_ts[0]
            if span > 0:
                self.pipe.det_fps = (len(self.pipe._det_ts) - 1) / span
        return res.events

    def step(self):
        """单路检测（保留兼容）：读帧 → 检测 → 处理。"""
        pkt = self.read_frame()
        if pkt is None:
            return []
        res = self.pipe.process(pkt)
        self.pipe._last_dets = res.dets
        self.pipe._det_ts.append(time.time())
        if len(self.pipe._det_ts) >= 4:
            span = self.pipe._det_ts[-1] - self.pipe._det_ts[0]
            if span > 0:
                self.pipe.det_fps = (len(self.pipe._det_ts) - 1) / span
        return res.events

    def close(self) -> None:
        self._stop.set()
        try:
            self.src.close()
        except Exception:                    # noqa: BLE001
            pass
        try:
            self.cap.release()
        except Exception:                    # noqa: BLE001
            pass


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "config" / "monitor.yaml"))
    ap.add_argument("--port", type=int, default=None)
    ap.add_argument("--host", default=None)
    ap.add_argument("--llm", default="auto", choices=["auto", "off", "dry"])
    ap.add_argument("--no-browser", action="store_true",
                    help="启动后不自动打开浏览器看板")
    a = ap.parse_args()

    lock = acquire_lock()
    if lock is None:
        print("[x] 已经有一个监控进程在运行，先关掉旧的。")
        return 4
    # 第二道保险：端口被占说明还有别的副本（安装版/别的解释器）在跑。
    # 两个实例同时抓摄像头 = 画面一直闪、帧率减半，必须拦住。
    _ui = (yaml.safe_load(Path(a.config).read_text(encoding="utf-8")) or {}).get("ui") or {}
    _p = int(_ui.get("port", 5050))
    if port_busy(str(_ui.get("host", "127.0.0.1")), _p):
        print(f"[x] 端口 {_p} 已被占用 —— 还有另一个监控实例在跑（或安装版开着）。")
        print("    两个实例同时抓摄像头会让画面一直闪、帧率减半，请先关掉那个。")
        return 5

    cfg = yaml.safe_load(Path(a.config).read_text(encoding="utf-8")) or {}
    cams = [c for c in (cfg.get("cameras") or []) if c.get("enabled", True)]
    if not cams:
        print("[x] monitor.yaml 里没有启用的摄像头")
        return 2

    det_cfg = cfg.get("detect") or {}
    thr = cfg.get("thresholds") or {}
    llm_cfg = cfg.get("llm") or {}
    view_cfg = cfg.get("view") or {}
    ui_cfg = cfg.get("ui") or {}

    load_dotenv()

    # ── 共享检测器：整机只有一份模型 ──
    print(f"\n加载模型 {WEIGHTS.name} ...", flush=True)
    # 香烟模型（可选）：和主模型是两份权重，分开推理后合并结果
    extra_w = det_cfg.get("extra_weights") or ""
    extra_path = None
    if extra_w:
        cand = Path(extra_w)
        if not cand.is_absolute():
            cand = WEIGHTS.parent.parent / extra_w      # 相对仓库根目录
        if cand.is_file():
            extra_path = str(cand)
            print(f"加载香烟模型 {cand.name} ...", flush=True)
        else:
            print(f"[warn] 香烟模型不存在，跳过：{cand}", flush=True)

    _device = det_cfg.get("device", 0)
    if _device == "cpu":
        _device = "cpu"
    elif not _device or _device == "auto":
        _device = 0          # 默认 GPU(cuda:0)
    detector = Detector(str(WEIGHTS), conf=float(thr.get("det_conf", 0.20)),
                        imgsz=int(det_cfg.get("imgsz", 640)), device=_device,
                        extra_weights=extra_path,
                        extra_conf=float(thr.get("cigarette_conf", 0.20)))

    # ── LLM 裁判（全局共享，各路共用同一个配额与去重表）──
    arb = None
    if a.llm != "off":
        llm_over = {
            "hourly_limit": int(llm_cfg.get("hourly_limit", 60)),
            "dedup_seconds": float(llm_cfg.get("dedup_seconds", 900)),
        }
        if a.llm == "dry":
            llm_over["dry_run"] = True
        arb = LlmArbiter(cfg=llm_over)

    # ── 每路一个画面视图 ──
    views = {
        str(c["id"]): LiveView(
            max_width=int(view_cfg.get("max_width", 900)),
            jpeg_quality=int(view_cfg.get("jpeg_quality", 72)),
            fps_limit=float(view_cfg.get("fps_limit", 30)),
            max_bytes=int(view_cfg.get("max_bytes", 90000)),
        ) for c in cams
    }
    for v in views.values():
        v.llm = arb
        # 画面分级：低于告警阈值的画黄色候选框，高于才是红色告警框
        v.alarm_confs = {0: float(thr.get("fire_conf", 0.45)),
                         1: float(thr.get("smoke_conf", 0.50))}
        # 彩色框阈值：比告警阈值高，保证"检测宽松但画面干净"
        v.color_confs = {0: float(thr.get("color_conf", 0.45)),
                         1: float(thr.get("color_conf_smoke", 0.55)),
                         2: float(thr.get("color_conf_cig", 0.55))}

    nf = Notifier(out_dir=ROOT / "alerts",
                  host=a.host or ui_cfg.get("host", "127.0.0.1"),
                  port=int(a.port or ui_cfg.get("port", 5050)),
                  lives=views)
    nf.detector = detector                  # 供 /api/models 切换模型开关
    url = nf.start()

    # ── 每路独立流水线 ──
    pub_interval = 1.0 / max(1.0, float(view_cfg.get("fps_limit", 30)))
    pipe_cfg = {
        "detect_mode": "auto" if det_cfg.get("sliced") else "full",
        "fast_smoke": bool(det_cfg.get("fast_smoke", False)),
        "det_conf": float(thr.get("det_conf", 0.20)),
        "fire_conf": float(thr.get("fire_conf", 0.55)),
        "smoke_conf": float(thr.get("smoke_conf", 0.60)),
        "fire_confirm": int(thr.get("fire_confirm", 6)),
        "smoke_confirm": int(thr.get("smoke_confirm", 3)),
        "cooldown_sec": float(thr.get("cooldown_sec", 60)),
        "dedup_iou": float(thr.get("dedup_iou", 0.10)),
        "dedup_grow": float(thr.get("dedup_grow", 0.5)),
        "fg_gate": float(thr.get("fg_gate", 0.08)),
        "fg_gate_min_dur": float(thr.get("fg_gate_min_dur", 12.0)),
        # LLM 置信度闸门：只在规则拿不准的区间才花钱
        "llm_min_conf": float(llm_cfg.get("min_conf", 0.50)),
        "llm_sure_conf": float(llm_cfg.get("sure_conf", 0.85)),
        "llm_mute_sec": float(llm_cfg.get("dedup_seconds", 300)),
        # 高置信目标无视静音 / 无视 LLM 否定，防止真火被这两条省钱的闸误伤
        "llm_mute_bypass_conf": float(llm_cfg.get("mute_bypass_conf", 0.80)),
        "llm_override_conf": float(llm_cfg.get("override_conf", 0.85)),
    }

    workers: list[Worker] = []
    print("\n接入摄像头：", flush=True)
    for c in cams:
        pc = dict(pipe_cfg)
        try:
            _cig = float(c.get("cigarette_conf",
                                thr.get("cigarette_conf", 0.03)))
            workers.append(Worker(c, detector, views[str(c["id"])], pc,
                                  arb, pub_interval, cig_conf=_cig))
        except RuntimeError as e:
            print(f"  [跳过] {e}", flush=True)
    if not workers:
        print("[x] 没有可用摄像头")
        return 3

    print(f"\n看板：{url}    共 {len(workers)} 路")
    print(f"检测 imgsz={det_cfg.get('imgsz', 640)} every={det_cfg.get('every', 2)} "
          f"sliced={det_cfg.get('sliced', False)}  "
          f"阈值 fire={pipe_cfg['fire_conf']} smoke={pipe_cfg['smoke_conf']}")
    print("按 Ctrl+C 停止\n" + "-" * 70, flush=True)

    # 自动打开浏览器看板（双击 exe 后无需手动输网址）。
    # 用独立线程延迟 1.5s 打开，等端口就绪、模型加载完。
    if not a.no_browser:
        def _open_browser() -> None:
            time.sleep(1.5)
            try:
                import webbrowser
                webbrowser.open(url)
            except Exception:               # noqa: BLE001
                pass
        threading.Thread(target=_open_browser, daemon=True).start()

    every = max(1, int(det_cfg.get("every", 2)))

    def on_event(ev, cam_id: str):
        nf.push(ev)
        tag = LEVEL_TAG.get(ev.level, ev.level)
        print(f"[{time.strftime('%H:%M:%S')}] {cam_id} {tag:<4} {ev.kind:<5} "
              f"tid={ev.track_id} score={ev.score:.2f} | "
              f"{(ev.reasons[0][:40] if ev.reasons else '')}", flush=True)

    stop = threading.Event()
    tick = 0
    _tile_grid = getattr(workers[0].pipe, "tile_grid", 2) if workers else 2
    _overlap = getattr(workers[0].pipe, "overlap", 0.25) if workers else 0.25
    try:
        while not stop.is_set():
            # ── 多摄像头同时检测：同一时刻收集所有路的帧，batch 一次推理 ──
            pkts = [w.read_frame() for w in workers]
            valid = [(i, pkt) for i, pkt in enumerate(pkts) if pkt is not None]
            if valid:
                images = [pkt.image for _, pkt in valid]
                cache_keys = [workers[i].id for i, _ in valid]
                try:
                    dets_list = detector.detect_auto_batch(
                        images, cache_keys, tile_grid=_tile_grid, overlap=_overlap)
                except Exception as exc:          # noqa: BLE001
                    print(f"[batch] 检测异常 {type(exc).__name__}: {exc}", flush=True)
                    dets_list = [[] for _ in valid]
                for (i, pkt), dets in zip(valid, dets_list):
                    try:
                        for ev in workers[i].step_with(pkt, dets):
                            on_event(ev, workers[i].id)
                    except Exception as exc:      # noqa: BLE001
                        print(f"[{workers[i].id}] 处理异常 {type(exc).__name__}: {exc}", flush=True)
            tick += 1
            if not valid:
                time.sleep(0.002)
    except KeyboardInterrupt:
        print("\n手动停止")
    finally:
        for w in workers:
            w.close()
        nf.stop()
        if arb is not None:
            arb.close()
        print("已退出")
    return 0


if __name__ == "__main__":
    sys.exit(main())
