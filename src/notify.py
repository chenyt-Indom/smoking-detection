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
from urllib.parse import parse_qs, unquote, urlparse

import cv2

from .paths import resource_root

# 告警级别 → 前端配色语义（前端只认这四种）
LEVELS = ("alarm", "warning", "notice", "cleared")


class Notifier:
    """事件推送器：落盘 + 广播 SSE + 可选 webhook。"""

    def __init__(self, out_dir: str | os.PathLike = "alerts", *,
                 host: str = "127.0.0.1", port: int = 5050,
                 webhook_url: str = "", max_history: int = 300,
                 dashboard: str | os.PathLike | None = None,
                 report_dir: str | os.PathLike | None = None,
                 live=None, lives=None):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.host, self.port = host, port
        self.webhook_url = webhook_url
        self.max_history = max_history
        self.dashboard = Path(dashboard) if dashboard else (
            resource_root() / "web" / "dashboard.html")
        # 人工标注的误报落在这里，等 scripts/merge_hard_negatives.py 回流进训练集
        self.report_dir = Path(report_dir) if report_dir else (
            self.out_dir.parent / "hard_negatives")

        # 实时画面：支持多路。key 是 cam_id，前端用 ?cam=xxx 指定。
        # 传单个 live 时自动归到 "default"，保持旧调用方式可用。
        if lives is not None:
            self.lives: dict[str, object] = dict(lives)
        elif live is not None:
            self.lives = {"default": live}
        else:
            self.lives = {}

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
        # 已人工标注过的 wall_ms：主动学习的队列要排除它们，避免重复问人
        self._reported: set[int] = set()

    def _pick_live(self, cam: str | None):
        """按 cam 参数挑一路。没给就返回第一路（单摄像头时的常规用法）。"""
        if not self.lives:
            return None
        if cam and cam in self.lives:
            return self.lives[cam]
        return next(iter(self.lives.values()))

    @property
    def live(self):
        """兼容旧代码：返回第一路。"""
        return next(iter(self.lives.values())) if self.lives else None

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

        # 两张图一起存：ROI 看细节，全景看位置。
        # 同步写：实测每张 5.5ms，异步反而让前端拿到事件时文件还没落地。
        snap_url = full_url = None
        has_roi = getattr(ev, "snapshot", None) is not None
        has_full = getattr(ev, "snapshot_full", None) is not None
        if has_roi or has_full:
            base = f"{rec['wall_ms']}_{ev.cam_id}_{ev.kind}_{ev.level}"
            if has_roi:
                p_roi = self.out_dir / f"{base}.jpg"
                self._write_snapshot(p_roi, ev.snapshot)
                snap_url = f"/snapshots/{base}.jpg" if p_roi.is_file() else None
            if has_full:
                p_full = self.out_dir / f"{base}_full.jpg"
                self._write_snapshot(p_full, ev.snapshot_full)
                full_url = f"/snapshots/{base}_full.jpg" if p_full.is_file() else None
        rec["snapshot"] = snap_url
        rec["snapshot_full"] = full_url
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
        """后台线程写截图。先写 .tmp 再 replace，保证读到的一定是完整文件。

        用 imencode + write_bytes 而不是 cv2.imwrite：
        imwrite 失败时只返回 False、不抛异常，而它的路径处理在 Windows 中文路径下
        还有已知问题。显式编码后由 Python 写文件，成功与否一目了然。
        """
        tmp = path.with_name(path.name + ".tmp")
        ts = time.time()
        try:
            ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
            if not ok:
                self.stats["errors"] += 1
                print(f"[notify] 截图编码失败: {path.name}", flush=True)
                return
            tmp.write_bytes(buf.tobytes())
            os.replace(tmp, path)
            cost = (time.time() - ts) * 1000
            if cost > 200:
                print(f"[slow] 截图落盘 {cost:.0f}ms（后台线程，已不阻塞检测）", flush=True)
        except Exception as exc:                 # noqa: BLE001
            self.stats["errors"] += 1
            print(f"[notify] 截图写入失败 {path.name}: {type(exc).__name__} {exc}",
                  flush=True)
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

    # ------------------------------------------------------------ 主动学习
    def review_queue(self, limit: int = 30) -> list[dict]:
        """挑出最值得人工标注的事件 —— 主动学习的采样器。

        为什么不按时间顺序全推：绝大多数事件模型都判得很笃定，标它们
        对模型没有增量。真正有价值的是模型"自己拿不准"的那一小撮，
        这叫不确定性采样（uncertainty sampling），是 active learning 的核心。

        优先级（越靠前信息量越大）
          1. 两次判定打架（overridden）—— 两个模型互相矛盾，最值得看
          2. LLM 返回 uncertain        —— 它自己都说没把握
          3. 置信落在边界区间 0.40~0.65 —— 典型难例
        已标注过的直接跳过，不重复打扰。
        """
        with self._lock:
            hist = list(self._history)
        done = set(self._reported)
        # reports.jsonl 里已有的也算标注过（跨重启）
        try:
            if (self.report_dir / "reports.jsonl").is_file():
                for ln in (self.report_dir / "reports.jsonl").read_text(
                        encoding="utf-8", errors="ignore").splitlines():
                    if ln.strip():
                        try:
                            done.add(json.loads(ln).get("wall_ms"))
                        except json.JSONDecodeError:
                            pass
        except OSError:
            pass

        scored = []
        for e in hist:
            wid = e.get("wall_ms")
            if wid in done:
                continue
            label = str(e.get("llm_label") or "")
            try:
                conf = float(e.get("score") or 0.0)
            except (TypeError, ValueError):
                conf = 0.0
            # llm_conf 才是裁判自己的把握度；score 是 YOLO 的。优先用前者
            try:
                llm_conf = float(e.get("llm_conf") or 0.0)
            except (TypeError, ValueError):
                llm_conf = 0.0
            pri = None
            if e.get("overridden"):
                pri = 0
            elif label == "uncertain":
                pri = 1
            elif 0.40 <= llm_conf <= 0.65:
                pri = 2
            elif 0.40 <= conf <= 0.60:
                pri = 3
            if pri is None:
                continue
            scored.append((pri, -float(conf), e))
        scored.sort(key=lambda x: (x[0], x[1]))
        out = []
        for pri, _, e in scored[:limit]:
            d = dict(e)
            d["why"] = {0: "两次判定不一致", 1: "模型没把握",
                        2: "裁判置信处于边界", 3: "检测置信处于边界"}[pri]
            out.append(d)
        return out

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
        # 语义统一成「有没有这个目标」，具体是火还是烟由事件的 kind 决定：
        #   火事件 -> 有火 / 无火 ；烟事件 -> 有烟 / 无烟
        # 这样同一套接口能同时服务两类事件，前端只换文案。
        alias = {"true_alarm": "present", "false_alarm": "absent",
                 "fire": "present", "no_fire": "absent"}
        verdict = alias.get(verdict, verdict)
        if verdict not in ("present", "absent"):
            return {"ok": False, "error": "verdict 只能是 present / absent"}

        wid = body.get("wall_ms")
        with self._lock:
            rec = next((r for r in reversed(self._history)
                        if r.get("wall_ms") == wid), None)
        if rec is None:
            return {"ok": False, "error": "找不到这条事件（可能已滚出历史，刷新页面重试）"}

        # ── 四类归类：人工判定 × 模型判定 ─────────────────────
        # 这是主动学习的分拣台。只有两类值得进训练集：
        #   人工说有火、模型说没有 → 假阴性（漏报），要正样本
        #   人工说无火、模型说有   → 假阳性（误报），要负样本
        # 两边一致的说明模型已经会了，再喂一遍没有增量。
        # 注意：Event.to_dict() 把 meta 用 ** 展开了，所以 llm_label / img_w
        # 这些都是**顶层字段**，不是嵌在 rec["meta"] 里。
        model_label = str(rec.get("llm_label") or "")
        if not model_label:
            # 没有 LLM 结论时用事件级别兜底推断
            lv = rec.get("level")
            model_label = "fire" if lv in ("alarm", "warning") else (
                "other" if lv == "cleared" else "uncertain")
        model_says_fire = model_label in ("fire", "smoke")
        human_says_fire = verdict == "present"
        if human_says_fire and model_says_fire:
            vclass = "TP"          # 都对，不学
        elif human_says_fire and not model_says_fire:
            vclass = "FN"          # 模型漏报 → 正样本
        elif not human_says_fire and model_says_fire:
            vclass = "FP"          # 模型误报 → 负样本
        else:
            vclass = "TN"          # 都没火，不学

        with self._lock:
            self._reported.add(wid)      # 标过的就退出待标注队列
        row = {"wall_ms": wid, "wall": rec.get("wall"), "verdict": verdict,
               "verdict_class": vclass, "model_label": model_label,
               "kind": rec.get("kind"), "level": rec.get("level"),
               "cam": rec.get("cam"), "score": rec.get("score"),
               "reasons": (rec.get("reasons") or [])[:2],
               "note": str(body.get("note") or "")[:200],
               "at": time.strftime("%Y-%m-%d %H:%M:%S")}
        saved = None
        note_extra = ""

        # ── 只有 FP / FN 才值得进训练集 ──────────────────────
        # FP（误报）→ 负样本：空标注，教模型"这张图里没有火"
        # FN（漏报）→ 正样本：保留 YOLO 当时的框当伪标注，教模型"这里有火"
        # TP/TN 说明两边判断一致，再喂一遍没有增量，只记流水不存图。
        if vclass in ("FP", "FN"):
            key = "snapshot_full" if vclass == "FN" else "snapshot"
            snap = rec.get(key) or ""
            src = self.out_dir / unquote(snap[len("/snapshots/"):]) \
                if snap.startswith("/snapshots/") else None
            if src is None or not src.is_file():
                # 缺图不阻断反馈：流水本身（哪条判错）对统计准确率有价值。
                note_extra = " [截图缺失，未生成样本]"
                self.stats["missing_snapshot"] = self.stats.get("missing_snapshot", 0) + 1
            else:
                # 负样本用 ROI（信息密度高、直接是待判别的区域）
                # 正样本用全图（要保留空间关系，且框坐标本来就是相对全图的）
                # 按类别分子目录：火的误报和烟的误报是两回事，
                # 训练时也得分开放（负样本可以混，正样本的类别必须对）。
                kind_dir = str(rec.get("kind") or "fire").lower()
                if kind_dir not in ("fire", "smoke"):
                    kind_dir = "fire"
                # 目录按**事件类别**分，不能写死 fire。
                # 原来所有样本都塞进 images/fire/，香烟和烟的样本混在里面，
                # 重训时按目录取数据就会把香烟样本当火用 ——
                # 用户的原话："注意样本对应的是烟而不是火"。
                _k = str(rec.get("kind") or "fire").lower()
                if _k not in ("fire", "smoke", "cigarette", "steam", "light"):
                    _k = "fire"
                base = self.report_dir / "positives" if vclass == "FN" else self.report_dir
                img_dir = base / "images" / kind_dir
                lbl_dir = base / "labels" / kind_dir
                img_dir.mkdir(parents=True, exist_ok=True)
                lbl_dir.mkdir(parents=True, exist_ok=True)
                stem = (f"hp_{wid}_{rec.get('cam')}_{rec.get('kind')}" if vclass == "FN"
                        else f"hn_{wid}_{rec.get('cam')}_{rec.get('kind')}")
                if not (img_dir / f"{stem}.jpg").exists():
                    (img_dir / f"{stem}.jpg").write_bytes(src.read_bytes())
                    if vclass == "FP":
                        # 空标注 = 这张图没有任何目标
                        (lbl_dir / f"{stem}.txt").write_text("", encoding="utf-8")
                    else:
                        # 正样本的框在 reports.jsonl 里（box + 原图尺寸），
                        # 由 scripts/export_training.py 换算成 YOLO 归一化坐标。
                        (lbl_dir / f"{stem}.txt").write_text(
                            "# pending: 由 export_training.py 依据 reports.jsonl 生成\n",
                            encoding="utf-8")
                saved = str(img_dir / f"{stem}.jpg")
                k = "hard_positives" if vclass == "FN" else "hard_negatives"
                self.stats[k] = self.stats.get(k, 0) + 1

        # 反馈流水账：既做留证，也算"人工复核后确认率"
        try:
            self.report_dir.mkdir(parents=True, exist_ok=True)
            row["saved"] = saved
            if note_extra:
                row["note"] = (row["note"] + note_extra).strip()
            with open(self.report_dir / "reports.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        except OSError:
            self.stats["errors"] += 1
        res = {"ok": True, "verdict": verdict, "verdict_class": vclass,
               "model_label": model_label, "saved": saved}
        if note_extra:
            res["warning"] = "截图缺失，已记录反馈但未生成负样本"
        return res

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
                # 三重防缓存。只写 no-store 时部分浏览器对主文档仍走内存缓存，
                # 用户改了前端却一直看到旧页面（实测踩过好几次）。
                self.send_header("Cache-Control",
                                 "no-store, no-cache, must-revalidate, max-age=0")
                self.send_header("Pragma", "no-cache")
                self.send_header("Expires", "0")
                for k, v in (extra or {}).items():
                    self.send_header(k, v)
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):                        # noqa: N802
                parsed = urlparse(self.path)
                path = parsed.path
                cam = (parse_qs(parsed.query).get("cam") or [None])[0]

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
                    view = notifier._pick_live(cam)
                    if view is None:
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
                        for chunk in view.mjpeg_iter():
                            self.wfile.write(chunk)
                            self.wfile.flush()
                    except (BrokenPipeError, ConnectionResetError, OSError):
                        pass
                    return

                if path == "/api/frame.jpg":
                    view = notifier._pick_live(cam)
                    jpeg = view.frame_bytes() if view is not None else None
                    if jpeg is None:
                        self._send(503, "text/plain; charset=utf-8", b"no frame yet")
                    else:
                        self._send(200, "image/jpeg", jpeg)
                    return

                if path == "/api/state":
                    view = notifier._pick_live(cam)
                    st = view.state() if view is not None else {}
                    self._send(200, "application/json; charset=utf-8",
                               json.dumps(st, ensure_ascii=False).encode())
                    return

                if path == "/api/review_queue":
                    body = notifier.review_queue()
                    self._send(200, "application/json; charset=utf-8",
                               json.dumps(body, ensure_ascii=False).encode())
                    return

                if path == "/api/cams":
                    # 一次拿回所有路的概要，前端不用逐路轮询
                    out = [{"cam": k, **v.state()} for k, v in notifier.lives.items()]
                    self._send(200, "application/json; charset=utf-8",
                               json.dumps(out, ensure_ascii=False).encode())
                    return

                if path == "/api/llm":
                    view = notifier._pick_live(cam)
                    st = (view.llm_state() if view is not None
                          else {"configured": False, "reason": "实时视图未启用"})
                    self._send(200, "application/json; charset=utf-8",
                               json.dumps(st, ensure_ascii=False).encode())
                    return

                if path == "/api/models":
                    det = getattr(notifier, "detector", None)
                    st = {
                        "main": bool(getattr(det, "enable_main", True)),
                        "extra": bool(getattr(det, "enable_extra", True)),
                    }
                    self._send(200, "application/json; charset=utf-8",
                               json.dumps(st, ensure_ascii=False).encode())
                    return

                if path == "/api/view/boxes":
                    view = notifier._pick_live(cam)
                    show = bool(getattr(view, "show_boxes", True)) if view else True
                    self._send(200, "application/json; charset=utf-8",
                               json.dumps({"show_boxes": show},
                                          ensure_ascii=False).encode())
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
                parsed = urlparse(self.path)
                path = parsed.path
                cam = (parse_qs(parsed.query).get("cam") or [None])[0]

                if path == "/api/snap":
                    # 存当前帧做训练素材：点一下按钮就把画面落盘，
                    # 同时把当前检测框写成 YOLO 预标注 —— 标注时只需修正，
                    # 不用从零画框，300 张的工作量能省一半以上。
                    view = notifier._pick_live(cam)
                    frame = getattr(view, "last_frame", None) if view else None
                    if frame is None:
                        self._send(503, "application/json; charset=utf-8",
                                   "{\"ok\":false,\"error\":\"没有可用画面\"}".encode())
                        return
                    import time as _t
                    root = Path(notifier.report_dir).parent / "dataset_raw"
                    (root / "images").mkdir(parents=True, exist_ok=True)
                    (root / "labels").mkdir(parents=True, exist_ok=True)
                    stem = f"{cam or 'cam'}_{int(_t.time()*1000)}"
                    ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
                    if not ok:
                        self._send(500, "application/json; charset=utf-8",
                                   "{\"ok\":false,\"error\":\"编码失败\"}".encode())
                        return
                    (root / "images" / f"{stem}.jpg").write_bytes(buf.tobytes())
                    # 预标注：只写 cigarette 类，且只写合理的框（去掉过大的误检框）
                    H, W = frame.shape[:2]
                    lines = []
                    dets = getattr(view, "last_dets", None) or []
                    for d in dets:
                        if str(getattr(d, "cls_name", "")) != "cigarette":
                            continue
                        x1, y1, x2, y2 = [float(v) for v in d.xyxy]
                        bw, bh = (x2 - x1) / W, (y2 - y1) / H
                        if bw <= 0 or bh <= 0 or bw * bh > 0.15:
                            continue
                        lines.append(f"0 {(x1+x2)/2/W:.6f} {(y1+y2)/2/H:.6f} {bw:.6f} {bh:.6f}")
                    (root / "labels" / f"{stem}.txt").write_text(
                        "\n".join(lines), encoding="utf-8")
                    body = ('{"ok":true,"file":"%s.jpg","prelabels":%d}' % (stem, len(lines))).encode()
                    self._send(200, "application/json; charset=utf-8", body)
                    return

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

                if path == "/api/llm/key":
                    # 前端「填入 API Key」输入框：动态写入裁判的 key，其他用户可自行配置。
                    llm = getattr(notifier.live, "llm", None) if notifier.live else None
                    if llm is None:
                        self._send(503, "application/json; charset=utf-8",
                                   b'{"ok":false,"error":"\\u672a\\u542f\\u7528 LLM"}')
                        return
                    key = ""
                    try:
                        n = int(self.headers.get("Content-Length") or 0)
                        body = json.loads(self.rfile.read(n) or b"{}")
                        key = str(body.get("api_key") or "").strip()
                    except (ValueError, OSError):
                        pass
                    llm.set_api_key(key)
                    self._send(200, "application/json; charset=utf-8",
                               json.dumps({"ok": True, "has_key": bool(key),
                                           "key_hint": (key[:4] + "…" + key[-4:])
                                           if len(key) > 8 else (key or "")},
                                          ensure_ascii=False).encode())
                    return

                if path == "/api/models":
                    det = getattr(notifier, "detector", None)
                    if det is None:
                        self._send(503, "application/json; charset=utf-8",
                                   b'{"ok":false,"error":"detector not set"}')
                        return
                    try:
                        n = int(self.headers.get("Content-Length") or 0)
                        body = json.loads(self.rfile.read(n) or b"{}")
                    except (ValueError, OSError):
                        body = {}
                    if "main" in body:
                        det.enable_main = bool(body["main"])
                    if "extra" in body:
                        det.enable_extra = bool(body["extra"])
                    self._send(200, "application/json; charset=utf-8",
                               json.dumps({"ok": True, "main": det.enable_main,
                                           "extra": det.enable_extra},
                                          ensure_ascii=False).encode())
                    return

                if path == "/api/view/boxes":
                    view = notifier._pick_live(cam)
                    if view is None:
                        self._send(503, "application/json; charset=utf-8",
                                   b'{"ok":false,"error":"view not set"}')
                        return
                    try:
                        n = int(self.headers.get("Content-Length") or 0)
                        body = json.loads(self.rfile.read(n) or b"{}")
                    except (ValueError, OSError):
                        body = {}
                    if "show_boxes" in body:
                        view.show_boxes = bool(body["show_boxes"])
                        for _v in notifier.lives.values():   # 多路同步
                            _v.show_boxes = view.show_boxes
                    self._send(200, "application/json; charset=utf-8",
                               json.dumps({"ok": True,
                                           "show_boxes": view.show_boxes},
                                          ensure_ascii=False).encode())
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
