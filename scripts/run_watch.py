"""实时监控主程序：YOLO 发现可疑 → 立即推前端 → 后台 LLM 裁决 → 结论也推前端。

架构（三条并行的路径，互不阻塞）
--------------------------------
    视频帧 ──▶ YOLO 检测 ──▶ 轨迹 ──▶ 规则层
                                        │
                     ┌──────────────────┼──────────────────┐
                     ▼                  ▼                  ▼
              ① 0.2s 推 notice    ② 异步送 LLM        ③ 定时推心跳
              （管理员立刻看到）   （1~2s 后出结论）    （证明系统活着）
                                        │
                                        ▼
                                 推 alarm / cleared

关键点：LLM **不阻塞**任何一帧，也**不回灌**检测链路。
它的结论只是给管理员的第二个信号，和规则层的信号并列推出去。

用法
----
    python scripts/run_watch.py --demo            # 合成序列演示（不用摄像头）
    python scripts/run_watch.py --camera 0        # 摄像头
    python scripts/run_watch.py --video xx.mp4    # 视频文件
常用参数：
    --fast 1        开极快模式（烟不等时序，0.2s 出手）
    --port 5050     看板端口
    --every 2       抽帧间隔
"""
from __future__ import annotations

import argparse
import sys
import threading
import time
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
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


def acquire_single_instance_lock():
    """同一时间只允许一个监控进程。

    为什么必须加这把锁
    ------------------
    Windows 允许多个进程同时打开同一个摄像头，但它们会互相踩数据 ——
    表现是画面花屏（蓝白条纹 + 绿色块）和严重卡顿，看起来像解码错误。
    这个坑排查起来极费时间，所以直接在入口拦住。

    ⚠️ 不能用 msvcrt 文件锁：锁文件是 0 字节时，两个进程都能成功锁上第 1 个字节
    （实测确认，结果就是两个实例一起抢摄像头）。改用内核互斥量 —— 它是进程级的，
    进程退出操作系统自动释放，既不会残留也不会被绕过。
    """
    if sys.platform == "win32":
        try:
            import ctypes
            from ctypes import wintypes
            k32 = ctypes.WinDLL("kernel32", use_last_error=True)
            k32.CreateMutexW.restype = wintypes.HANDLE
            k32.CreateMutexW.argtypes = [wintypes.LPCVOID, wintypes.BOOL,
                                         wintypes.LPCWSTR]
            handle = k32.CreateMutexW(None, True, "Local\\FireSmokeMonitor_Single")
            if ctypes.get_last_error() == 183:      # ERROR_ALREADY_EXISTS
                return None
            return handle          # 保持引用；进程结束由内核自动释放
        except Exception:          # noqa: BLE001  拿不到互斥量就别挡着启动
            return True

    # 非 Windows：退回文件锁
    lock_path = ROOT / "data" / "monitor.lock"
    try:
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        fh = open(lock_path, "w")
        fh.write("0")
        fh.flush()
    except OSError:
        return True
    try:
        import fcntl
        fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        fh.close()
        return None
    return fh


def build_source(args):
    if args.demo:
        # 复用测试脚本的合成序列：先空场 30 帧，然后火+烟出现
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "tp", ROOT / "scripts" / "test_pipeline.py")
        tp = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(tp)

        det = Detector(str(WEIGHTS), conf=0.20, imgsz=args.imgsz, device=0)
        fire = tp.pick_patch(det, cls_id=0, target_pct=0.012, other_cls=1)
        smoke = tp.pick_patch(det, cls_id=1, target_pct=0.02, other_cls=0)
        n = int(600 * args.n_mul)
        info = CameraInfo(cam_id="demo_01", kind="fixed", fps=tp.FPS)
        src = FixedCameraSource(info, frames=tp.gen_smoke(fire, smoke, n, with_fire=True),
                                warmup=tp.WARMUP_FRAMES)
        return src, det, n

    info = CameraInfo(cam_id=f"cam_{args.camera if args.camera is not None else 'file'}",
                      kind="fixed", fps=args.fps)
    if args.video:
        src = FixedCameraSource.from_video(info, args.video)
    else:
        # Windows 上必须显式指定 DirectShow：默认的 MSMF 后端对不少 USB 摄像头
        # 会"能打开但只返回黑帧"，画面正中还带一个带锁的摄像头占位图标。
        # 单独用 cv2.VideoCapture(0) 测试时同样会踩到，所以不是权限问题。
        backend = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY
        cap = cv2.VideoCapture(args.camera, backend)
        if not cap.isOpened():
            print(f"[x] 摄像头 {args.camera} 打不开。换一个索引试试（--camera 1），"
                  f"或确认没有被其他程序占用", flush=True)
            raise SystemExit(3)
        # 分辨率按摄像头**实际能力**选，不是越高越好。
        # 实测两台设备差异巨大：
        #   Rapoo 外接  ：720p 只有 10fps，640x480 才 16.7fps（YUY2 顶满 USB 带宽）
        #   Integrated  ：1280x720 能跑 29fps，天然支持压缩格式
        # 默认给 1280x720，用 --width/--height 可覆盖。
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
        cap.set(cv2.CAP_PROP_FPS, 30)

        def _frames():
            """持续读帧。

            ⚠️ 关键：read() 偶发失败**不能**直接 break。
            摄像头在自动曝光/白平衡重新收敛的那几百毫秒里会返回 (False, None)，
            早先这里一失败就结束生成器，采集线程随之永久退出 —— 画面从此定格，
            而且不会自愈。表现就是"画风突变后停帧"。
            现在改成连续失败到阈值才认输。
            """
            fails = 0
            while True:
                ok, img = cap.read()
                if not ok:
                    fails += 1
                    if fails % 30 == 1:
                        print(f"[cam] 读帧失败 {fails} 次（相机可能在调曝光，稍等）",
                              flush=True)
                    if fails > 300:          # 连续约 3 秒拿不到才判定设备掉线
                        print("[cam] 连续读帧失败，判定摄像头掉线，退出采集", flush=True)
                        break
                    time.sleep(0.01)
                    continue
                if fails:
                    print(f"[cam] 已恢复（此前连续失败 {fails} 次）", flush=True)
                    fails = 0
                yield img
            cap.release()

        src = FixedCameraSource(info, frames=_frames())
    det = Detector(str(WEIGHTS), conf=0.20, imgsz=args.imgsz, device=0)
    return src, det, None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", action="store_true", help="用合成序列演示")
    ap.add_argument("--camera", type=int, default=None, help="摄像头编号")
    ap.add_argument("--video", default=None, help="视频文件")
    ap.add_argument("--fps", type=float, default=25.0)
    ap.add_argument("--width", type=int, default=1280, help="采集宽度，默认 1280")
    ap.add_argument("--height", type=int, default=720, help="采集高度，默认 720")
    ap.add_argument("--imgsz", type=int, default=896)
    ap.add_argument("--every", type=int, default=2, help="抽帧间隔")
    ap.add_argument("--fast", type=int, default=1, help="1=开极快模式")
    ap.add_argument("--sliced", type=int, default=1)
    ap.add_argument("--detect", default="auto", choices=["auto", "slice", "full"],
                    help="auto=粗筛没东西就不切片（空白帧 45.9fps / 切片只有 16.3）")
    ap.add_argument("--forget-lights", type=int, default=0,
                    help="1=清空「已确认不是火」的区域记忆（现场布置变了就用它）")
    ap.add_argument("--llm", default="auto", choices=["auto", "off", "dry"],
                    help="auto=按 config/arbiter.yaml；dry=不发请求只验证接线")
    ap.add_argument("--n-mul", type=float, default=1.0)
    ap.add_argument("--port", type=int, default=5050)
    ap.add_argument("--host", default="127.0.0.1",
                    help="0.0.0.0 = 允许局域网访问（手机/另一台电脑能看）")
    ap.add_argument("--webhook", default="", help="额外转发到企微/钉钉等")
    ap.add_argument("--max-frames", type=int, default=None)
    args = ap.parse_args()

    if not (args.demo or args.video or args.camera is not None):
        print("要么 --demo，要么 --camera 0，要么 --video 文件")
        return 2

    lock = acquire_single_instance_lock()
    if lock is None:
        print("[x] 已经有一个监控进程在运行。")
        print("    多个进程同时打开摄像头会导致画面花屏 + 严重卡顿。")
        print("    请先关掉旧的（任务管理器结束 python.exe / FireMonitor.exe），再启动这个。")
        return 4

    load_dotenv()
    raw_src, det, plan = build_source(args)
    # 采集放进独立线程：检测 9fps、画面 25fps，互不拖累
    src = LatestFrameSource(raw_src)

    # ── LLM 裁判：异步，缺 Key/缺模型时自动降级，绝不阻塞告警 ──
    arb = None
    if args.llm != "off":
        cfg = {"dry_run": True} if args.llm == "dry" else None
        arb = LlmArbiter(cfg=cfg)
        if args.llm == "auto" and not arb.cfg.get("model"):
            print("⚠️  config/arbiter.yaml 里 model 没填 → 裁判会自动降级到规则层")
        if args.llm == "auto" and not arb.api_key:
            print("⚠️  没读到 TELEAI_APPKEY（放 .env 里）→ 裁判会自动降级")

    # ── 实时画面：后端画框后压成 JPEG，前端逐帧拉取 ──
    # fps_limit 要跟上源帧率，否则画面会明显卡顿；20~25 是 CPU 编码和流畅度的平衡点
    live = LiveView(max_width=1280, jpeg_quality=72, fps_limit=30.0)
    live.llm = arb                       # /api/llm 读它的 snapshot()

    # ── 前端推送 ──
    nf = Notifier(out_dir=ROOT / "alerts", host=args.host, port=args.port,
                  webhook_url=args.webhook, live=live)
    url = nf.start()
    base_url = url[:-1] if url.endswith("/") else url

    pipe = Pipeline(det, sliced=bool(args.sliced), tile_grid=2,
                    cfg={"fast_smoke": bool(args.fast), "detect_mode": args.detect},
                    arbiter=arb)

    print(f"\n看板已启动：{url}")
    print(f"模型 {WEIGHTS.name}  imgsz={args.imgsz}  every={args.every}  "
          f"detect={args.detect}  fast_smoke={bool(args.fast)}  LLM={args.llm}")

    # ── 「已确认不是火」的区域记忆：同一盏灯不再反复花钱问 LLM ──
    lights = getattr(pipe, "lights", None)
    if args.forget_lights:
        if lights is not None:
            n = lights.clear()
            lights.save()
            print(f"已清空区域记忆 {n} 处（{lights.path}）")
        else:
            print("区域记忆是关着的（cfg.light_memory=False）")
    if lights is not None:
        print(f"区域记忆 {len(lights)} 处（{lights.path}）"
              f" —— 命中则跳过 LLM，外形一变自动作废")
    print("按 Ctrl+C 停止\n" + "-" * 68)

    def on_event(ev, res):
        nf.push(ev)
        tag = LEVEL_TAG.get(ev.level, ev.level)
        print(f"[{time.strftime('%H:%M:%S')}] {tag:<4} {ev.kind:<5} "
              f"tid={ev.track_id} t={ev.t_alarm:.1f}s 延迟={ev.latency:.1f}s "
              f"score={ev.score:.2f} | {ev.reasons[0][:44] if ev.reasons else ''}")

    # 实时画面：独立线程，固定节奏取最新帧 + 最近一次检测框。
    #
    # 不用「等新帧通知」那种写法：采集线程的 notify 会被检测线程的 GIL 占压，
    # 等收到通知时往往已过几百毫秒，画面就是一顿一顿的。
    # 这里每秒按固定间隔去取"此刻最新的一帧"，有新的就发，没有就跳过，
    # 峰值延迟由 sleep 粒度决定（约 5ms），不会积累。
    stop_pub = threading.Event()
    PUB_INTERVAL = 1.0 / 30.0          # 目标 30fps，实际受摄像头上限约束

    def publish_loop():
        last_seq, last_pub = -1, 0.0
        n, t0 = 0, time.time()
        while not stop_pub.is_set():
            ta = time.time()
            pkt, seq = src.snapshot()
            tb = time.time()
            now = tb
            if pkt is not None and seq != last_seq and (now - last_pub) >= PUB_INTERVAL:
                gap = (now - last_pub) * 1000 if last_pub else 0
                prev_seq = last_seq
                last_seq, last_pub = seq, now
                n += 1
                t1 = time.time()
                try:
                    # fps 传的是**检测帧率**，pub_fps 由 LiveView 自己算画面帧率，
                    # 两个指标分开才能看出到底是哪个环节慢
                    live.publish(pkt.image, getattr(pipe, "_last_dets", None) or [],
                                 frame_idx=pkt.idx, cam_id=getattr(pkt, "cam_id", ""),
                                 fps=getattr(pipe, "det_fps", 0.0))
                except Exception:        # noqa: BLE001
                    pass
                cost = (time.time() - t1) * 1000
                # 卡顿归因：间隔长要分清是"取帧被锁"还是"画框编码慢"还是"线程没被调度"
                if gap > 200:
                    print(f"[slow] gap={gap:.0f}ms seq={last_seq}(+{last_seq-prev_seq}) 取帧={1000*(tb-ta):.1f}ms "
                          f"发布={cost:.1f}ms", flush=True)
                elif cost > 80:
                    print(f"[slow] 发布耗时 {cost:.0f}ms（画框+编码）", flush=True)
            time.sleep(0.004)              # 4ms 轮询，兼顾延迟与 CPU

    pub_th = threading.Thread(target=publish_loop, name="publish", daemon=True)
    pub_th.start()

    # 压掉 GC 带来的偶发卡顿：模型和检测器都加载完了，对象图基本稳定，
    # 把已有对象冻结出扫描范围，再放宽阈值，避免每几百毫秒就来一次回收停顿。
    # 之前实测出现过 480ms 的单帧间隔，多半就是 GC + 编码撞在一起。
    try:
        import gc
        gc.collect()
        gc.freeze()
        gc.set_threshold(2000, 50, 50)
    except Exception:                    # noqa: BLE001
        pass

    # 关键：把 GIL 切换间隔从默认 5ms 压到 1ms。
    #
    # 探针实测：画面线程取帧 0.0ms、发布 2.4ms，两次发布却相隔 2700ms ——
    # 说明线程根本没被调度，不是它自己慢，是抢不到 GIL。
    # 主线程每帧要做 numpy/opencv 的密集运算（letterbox、NMS 后处理、画框），
    # 这些是 C 代码但不释放 GIL；默认 5ms 的切换粒度下，画面线程一次只能挤进
    # 一个很窄的窗口，攒够一次发布要等好几帧。
    try:
        sys.setswitchinterval(0.001)
    except Exception:                    # noqa: BLE001
        pass

    try:
        out = pipe.run(src, on_event=on_event, every=args.every,
                       max_frames=args.max_frames)
        print("-" * 68)
        print(f"帧 {out['frames']}  处理 {out['processed']}  {out['fps']} fps")
        print(f"事件 {out['n_events'] if 'n_events' in out else len(out['events'])}"
              f"  统计 {out['stats']}")
        if arb is not None:
            print(f"LLM  {arb.stats}")
    except KeyboardInterrupt:
        print("\n手动停止")
    finally:
        stop_pub.set()
        if not args.demo:
            pass
        nf.stop()
        if arb is not None:
            arb.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
