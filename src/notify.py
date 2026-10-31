"""告警推送：把事件**立刻**推到管理员浏览器（SSE），不经过任何队列积压。

为什么用 SSE 而不是 WebSocket / 轮询
------------------------------------
· 轮询：管理员看到的延迟 = 轮询间隔，想做 1 秒内必到就得每秒打一次接口，
  成本高且延迟不稳定。
· WebSocket：要双向、要心跳、要处理重连握手，对"只往一个方向推告警"是过度设计。
· **SSE（Server-Sent Events）**：浏览器原生 EventSource 支持，单向下行正好对口，
  断线自动重连，纯标准库几十行就能实现 —— 不引入 FastAPI/uvicorn 依赖。

为什么不用框架
--------------
现场部署的机器经常是离线的、装不了新包的。这个模块**只用标准库**，
`python src/notify.py` 就能起来，少一个依赖少一个部署事故。

推送顺序保证
------------
事件先落盘（截图 + jsonl）再广播。宁可晚 3 毫秒，也不要出现"前端有告警、磁盘没证据"——
事后追责的时候截图比日志有用得多。
"""
from __future__ import annotations

import json
import os
import queue
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

import cv2

# 告警级别 → 前端配色语义（前端只认这四种）
LEVELS = ("alarm", "warning", "notice", "cleared")


class Notifier:
    """事件推送器：落盘 + 广播 SSE + 可选 webhook。"""

    def __init__(self, out_dir: str | os.PathLike = "alerts", *,
                 host: str = "127.0.0.1", port: int = 5050,
                 webhook_url: str = "", max_history: int = 300,
                 dashboard: str | os.PathLike | None = None,
                 report_dir: str | os.PathLike | None = None,
                 live=None):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.host, self.port = host, port
        self.webhook_url = webhook_url
        self.max_history = max_history
        # 实时画面（src/liveview.py）。为 None 时 /stream 返回 503，
        # 其余功能（事件流、反馈）照常工作。
        self.live = live
        self.dashboard = Path(dashboard) if dashboard else (
            Path(__file__).resolve().parent.parent / "web" / "dashboard.html")
        # 人工标注的误报落在这里，等 scripts/merge_hard_negatives.py 回流进训练集
        self.report_dir = Path(report_dir) if report_dir else (
            self.out_dir.parent / "hard_negatives")

        self._clients: list[queue.Queue] = []
        self._history: list[dict] = []
        self._lock = threading.Lock()
        self._httpd: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None
        # 事件落盘队列：把磁盘 IO 从检测主循环里彻底摘出去
        self._wq: queue.Queue = queue.Queue(maxsize=2000)
        self._wthread = threading.Thread(target=self._writer_loop, name="event-writer",
                                         daemon=True)
        self._wthread.start()
        self.stats = {"pushed": 0, "clients": 0, "errors": 0}

    def _writer_loop(self) -> None:
        """后台串行写 jsonl。串行是为了保持事件顺序。"""
        path = self.out_dir / "events.jsonl"
        while True:
            rec = self._wq.get()
            if rec is None:
                self._wq.task_done()
                break
            try:
                with open(path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            except OSError:
                self.stats["errors"] += 1
            finally:
                self._wq.task_done()

    # ------------------------------------------------------------ 推送
    def push(self, ev, extra: dict | None = None) -> dict:
        """把一条 Event 变成前端载荷：存图 → 组装 → 广播。返回落地的 dict。"""
        rec = ev.to_dict()
        rec["wall"] = time.strftime("%Y-%m-%d %H:%M:%S")
        rec["wall_ms"] = int(time.time() * 1000)
        rec["level"] = rec.get("level") if rec.get("level") in LEVELS else "notice"

        snap_url = None
        if getattr(ev, "snapshot", None) is not None:
            ts = time.strftime("%Y%m%d_%H%M%S", time.localtime())
            name = f"{rec['wall_ms']}_{ev.cam_id}_{ev.kind}_{ev.level}.jpg"
            p = self.out_dir / name
            # 落盘必须异步：1280x720 的 JPEG 写盘本身只要几毫秒，但
            # Windows Defender 实时扫描会把它拖到几百毫秒甚至更久，
            # 而 push() 是在检测主循环里同步调用的 —— 一卡就是整条链路卡。
            # 先写临时文件再原子 rename，前端不会读到写了一半的图。
            snap_url = f"/snapshots/{name}"
            threading.Thread(target=self._write_snapshot, args=(p, ev.snapshot),
                             daemon=True).start()
        rec["snapshot"] = snap_url
        if extra:
            rec.update(extra)

        # jsonl 追加改成后台队列。
        # 实测卡顿特征：画面和采集"同时"停 1~2 秒，而编码只要 2ms —— 是整个进程被
        # 挡住了。同步写 jsonl 时，磁盘 IO + Windows Defender 实时扫描能拖几百毫秒，
        # 而 push() 在检测主循环里调用，一卡全线停。写盘不该挡推送。
        try:
            self._wq.put_nowait(rec)
        except queue.Full:
            self.stats["errors"] += 1

        self._broadcast(rec)
        if self.webhook_url:
            threading.Thread(target=self._post_webhook, args=(rec,), daemon=True).start()
        self.stats["pushed"] += 1
        return rec

    def _write_snapshot(self, path: Path, img) -> None:
        """后台线程写截图。先写 .tmp 再 replace，保证读到的一定是完整文件。"""
        tmp = path.with_name(path.name + ".tmp")
        ts = time.time()
        try:
            cv2.imwrite(str(tmp), img)
            os.replace(tmp, path)
            cost = (time.time() - ts) * 1000
            if cost > 200:
                print(f"[slow] 截图落盘 {cost:.0f}ms（后台线程，已不阻塞检测）", flush=True)
        except Exception:                        # noqa: BLE001
            self.stats["errors"] += 1
            try:
                tmp.unlink(missing_ok=True)
            except OSError:
                pass

    def _broadcast(self, rec: dict) -> None:
        with self._lock:
            self._history.append(rec)
            if len(self._history) > self.max_history:
                self._history = self._history[-self.max_history:]
            dead = []
            for q in self._clients:
                try:
                    q.put_nowait(rec)
                except queue.Full:
                    dead.append(q)
            for q in dead:
                self._clients.remove(q)

    def _post_webhook(self, rec: dict) -> None:
        """可选：转发到企业微信/钉钉/自建接口。失败不影响本地推送。"""
        try:
            import requests
            requests.post(self.webhook_url, json=rec, timeout=5)
        except Exception:                            # noqa: BLE001
            self.stats["errors"] += 1

    # ------------------------------------------------------------ 人工反馈
    def report(self, body: dict) -> dict:
        """看板上点「误报」/「属实」→ 落成一条**硬负样本**，等回流进训练集。

        为什么这才是灯光误报的正解
        --------------------------
        闪烁闸（物理先验）能拦掉恒亮光源，但它拦不住"会闪的假火"：电焊弧光、
        夕阳在玻璃上跳动、红色霓虹的 PWM 调光、电视里的火焰画面。这类东西只有
        拿真实误报帧去训才治得住。灯是开放集合 —— 标完车位灯，下周加个招牌照样报。

        为什么存成**空标注**的图
        ------------------------
        YOLO 里一个空 .txt 就是"这张图没有任何目标"，比任何负类标签都干净。
        截图是检测框外扩 25% 的裁剪（见 snapshot_margin），和切片推理喂进网络的
        那块区域尺度接近 —— 直接当负样本是合适的。

        为什么要去重
        ------------
        同一条告警可能被反复点、或者前端重试。按 wall_ms 定名，重复提交只更新
        reports.jsonl 里的一行，不会在训练集里塞 50 张一样的图。
        """
        verdict = str(body.get("verdict") or "")
        if verdict not in ("false_alarm", "true_alarm"):
            return {"ok": False, "error": "verdict 只能是 false_alarm / true_alarm"}

        wid = body.get("wall_ms")
        with self._lock:
            rec = next((r for r in reversed(self._history)
                        if r.get("wall_ms") == wid), None)
        if rec is None:
            return {"ok": False, "error": "找不到这条事件（可能已滚出历史，刷新页面重试）"}

        row = {"wall_ms": wid, "wall": rec.get("wall"), "verdict": verdict,
               "kind": rec.get("kind"), "level": rec.get("level"),
               "cam": rec.get("cam"), "score": rec.get("score"),
               "reasons": (rec.get("reasons") or [])[:2],
               "note": str(body.get("note") or "")[:200],
               "at": time.strftime("%Y-%m-%d %H:%M:%S")}
        saved = None

        if verdict == "false_alarm":
            snap = rec.get("snapshot") or ""
            src = self.out_dir / unquote(snap[len("/snapshots/"):]) \
                if snap.startswith("/snapshots/") else None
            if src is None or not src.is_file():
                return {"ok": False, "error": "这条事件没有截图，存不了负样本"}
            img_dir = self.report_dir / "images"
            lbl_dir = self.report_dir / "labels"
            img_dir.mkdir(parents=True, exist_ok=True)
            lbl_dir.mkdir(parents=True, exist_ok=True)
            stem = f"hn_{wid}_{rec.get('cam')}_{rec.get('kind')}"
            # 去重：同一 wall_ms 已经存过就不覆盖（人工可能补点 note）
            if not (img_dir / f"{stem}.jpg").exists():
                (img_dir / f"{stem}.jpg").write_bytes(src.read_bytes())
                (lbl_dir / f"{stem}.txt").write_text("", encoding="utf-8")
            saved = str(img_dir / f"{stem}.jpg")
            self.stats["hard_negatives"] = self.stats.get("hard_negatives", 0) + 1

        # 反馈流水账：既做留证，也算"人工复核后确认率"
        try:
            self.report_dir.mkdir(parents=True, exist_ok=True)
            row["saved"] = saved
            with open(self.report_dir / "reports.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        except OSError:
            self.stats["errors"] += 1
        return {"ok": True, "verdict": verdict, "saved": saved}

    # ------------------------------------------------------------ HTTP
    def start(self) -> str:
        """起推送服务（后台线程），返回看板地址。"""
        if self._httpd is not None:
            return self.url
        notifier = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"
            # 空闲连接必须尽快回收。
            # ThreadingHTTPServer 是"一个连接一个线程"，而看板每 30ms 拉一帧、
            # 每帧 URL 都带不同的 ?n= 参数，keep-alive 下连接会一直挂着。
            # 实测累积到 300+ 个 ESTABLISHED 线程，几百个线程哄抢 GIL，
            # 采集线程和发布线程一起被饿死 —— 表现就是画面每隔一两秒停一次。
            timeout = 5

            def log_message(self, *a):               # 静音，别把控制台刷爆
                pass

            def handle(self):
                """吞掉客户端断开引发的连接异常。

                HTTP/1.1 是长连接：一个请求处理完，服务器会继续读下一行请求头。
                浏览器关标签页、或者 MJPEG/SSE 流被主动断开时，这一步会抛
                ConnectionAbortedError(WinError 10053)，而标准库的 socketserver
                不会吞它，会在日志里打一整段 traceback。功能没坏，但很吵，
                而且看日志的人会以为服务出错。这里统一收掉。
                """
                try:
                    super().handle()
                except (ConnectionAbortedError, ConnectionResetError,
                        BrokenPipeError, TimeoutError):
                    pass

            def _send(self, code: int, ctype: str, body: bytes, extra=None):
                self.send_response(code)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                for k, v in (extra or {}).items():
                    self.send_header(k, v)
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):                        # noqa: N802
                path = urlparse(self.path).path

                if path in ("/", "/index.html"):
                    if not notifier.dashboard.exists():
                        self._send(404, "text/plain; charset=utf-8",
                                   "缺少 web/dashboard.html".encode())
                        return
                    self._send(200, "text/html; charset=utf-8",
                               notifier.dashboard.read_bytes())
                    return

                if path == "/api/recent":
                    with notifier._lock:
                        body = json.dumps(notifier._history[-80:], ensure_ascii=False)
                    self._send(200, "application/json; charset=utf-8", body.encode())
                    return

                if path == "/stream":
                    # MJPEG：每一帧都是一个独立 JPEG 分片，浏览器 <img> 直接能放
                    if notifier.live is None:
                        self._send(503, "text/plain; charset=utf-8",
                                   "实时画面未启用（主程序没接 LiveView）".encode())
                        return
                    self.send_response(200)
                    self.send_header("Content-Type",
                                     "multipart/x-mixed-replace; boundary=frame")
                    self.send_header("Cache-Control", "no-store")
                    self.send_header("Connection", "keep-alive")
                    self.send_header("X-Accel-Buffering", "no")
                    self.end_headers()
                    try:
                        for chunk in notifier.live.mjpeg_iter():
                            self.wfile.write(chunk)
                            self.wfile.flush()
                    except (BrokenPipeError, ConnectionResetError, OSError):
                        pass
                    return

                if path == "/api/frame.jpg":
                    jpeg = (notifier.live.frame_bytes()
                            if notifier.live is not None else None)
                    if jpeg is None:
                        self._send(503, "text/plain; charset=utf-8", b"no frame yet")
                    else:
                        self._send(200, "image/jpeg", jpeg)
                    return

                if path == "/api/state":
                    st = notifier.live.state() if notifier.live is not None else {}
                    self._send(200, "application/json; charset=utf-8",
                               json.dumps(st, ensure_ascii=False).encode())
                    return

                if path == "/api/llm":
                    st = (notifier.live.llm_state() if notifier.live is not None
                          else {"configured": False, "reason": "实时视图未启用"})
                    self._send(200, "application/json; charset=utf-8",
                               json.dumps(st, ensure_ascii=False).encode())
                    return

                if path == "/events":
                    # SSE：先补发最近历史，再持续推新事件
                    self.send_response(200)
                    self.send_header("Content-Type", "text/event-stream; charset=utf-8")
                    self.send_header("Cache-Control", "no-cache")
                    self.send_header("Connection", "keep-alive")
                    self.send_header("X-Accel-Buffering", "no")   # 让 nginx 别缓冲
                    self.end_headers()
                    q: queue.Queue = queue.Queue(maxsize=500)
                    with notifier._lock:
                        notifier._clients.append(q)
                        hist = list(notifier._history[-40:])
                    notifier.stats["clients"] = len(notifier._clients)
                    try:
                        for rec in hist:
                            self.wfile.write(f"data: {json.dumps(rec, ensure_ascii=False)}\n\n"
                                             .encode())
                        self.wfile.flush()
                        while True:
                            try:
                                rec = q.get(timeout=15)
                                payload = f"data: {json.dumps(rec, ensure_ascii=False)}\n\n"
                            except queue.Empty:
                                payload = ": ping\n\n"           # 心跳，防中间设备断连
                            self.wfile.write(payload.encode())
                            self.wfile.flush()
                    except (BrokenPipeError, ConnectionResetError, OSError):
                        pass
                    finally:
                        with notifier._lock:
                            if q in notifier._clients:
                                notifier._clients.remove(q)
                            notifier.stats["clients"] = len(notifier._clients)
                    return

                if path.startswith("/snapshots/"):
                    name = unquote(path[len("/snapshots/"):])
                    f = (notifier.out_dir / name)
                    # 防目录穿越：只允许 out_dir 下的普通文件
                    try:
                        f.resolve().relative_to(notifier.out_dir.resolve())
                    except (ValueError, OSError):
                        self._send(403, "text/plain", b"forbidden")
                        return
                    if f.is_file():
                        self._send(200, "image/jpeg", f.read_bytes())
                    else:
                        self._send(404, "text/plain", b"not found")
                    return

                self._send(404, "text/plain", b"not found")

            def do_POST(self):                       # noqa: N802
                path = urlparse(self.path).path

                if path == "/api/llm/test":
                    llm = getattr(notifier.live, "llm", None) if notifier.live else None
                    if llm is None:
                        self._send(503, "application/json; charset=utf-8",
                                   b'{"ok":false,"error":"\\u672a\\u542f\\u7528 LLM"}')
                        return
                    try:
                        res = llm.test_connection()
                    except Exception as exc:         # noqa: BLE001
                        res = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
                    self._send(200 if res.get("ok") else 502,
                               "application/json; charset=utf-8",
                               json.dumps(res, ensure_ascii=False).encode())
                    return

                if path != "/api/report":
                    self._send(404, "text/plain", b"not found")
                    return
                try:
                    n = int(self.headers.get("Content-Length") or 0)
                    body = json.loads(self.rfile.read(n) or b"{}")
                    if not isinstance(body, dict):
                        raise ValueError
                except (ValueError, OSError):
                    self._send(400, "application/json; charset=utf-8",
                               b'{"ok":false,"error":"bad json"}')
                    return
                res = notifier.report(body)
                self._send(200 if res.get("ok") else 400,
                           "application/json; charset=utf-8",
                           json.dumps(res, ensure_ascii=False).encode())

        self._httpd = ThreadingHTTPServer((self.host, self.port), Handler)
        self._httpd.daemon_threads = True
        self._thread = threading.Thread(target=self._httpd.serve_forever,
                                        name="notifier", daemon=True)
        self._thread.start()
        return self.url

    @property
    def url(self) -> str:
        host = "127.0.0.1" if self.host in ("0.0.0.0", "::") else self.host
        return f"http://{host}:{self.port}/"

    def stop(self) -> None:
        if self._httpd is not None:
            self._httpd.shutdown()
            self._httpd = None
