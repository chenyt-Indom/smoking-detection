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
                 dashboard: str | os.PathLike | None = None):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.host, self.port = host, port
        self.webhook_url = webhook_url
        self.max_history = max_history
        self.dashboard = Path(dashboard) if dashboard else (
            Path(__file__).resolve().parent.parent / "web" / "dashboard.html")

        self._clients: list[queue.Queue] = []
        self._history: list[dict] = []
        self._lock = threading.Lock()
        self._httpd: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None
        self.stats = {"pushed": 0, "clients": 0, "errors": 0}

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
            try:
                cv2.imwrite(str(p), ev.snapshot)     # 先落盘，再广播
                snap_url = f"/snapshots/{name}"
            except Exception:                        # noqa: BLE001
                self.stats["errors"] += 1
        rec["snapshot"] = snap_url
        if extra:
            rec.update(extra)

        # jsonl 追加（长期留证）
        try:
            with open(self.out_dir / "events.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        except OSError:
            self.stats["errors"] += 1

        self._broadcast(rec)
        if self.webhook_url:
            threading.Thread(target=self._post_webhook, args=(rec,), daemon=True).start()
        self.stats["pushed"] += 1
        return rec

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

    # ------------------------------------------------------------ HTTP
    def start(self) -> str:
        """起推送服务（后台线程），返回看板地址。"""
        if self._httpd is not None:
            return self.url
        notifier = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *a):               # 静音，别把控制台刷爆
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
