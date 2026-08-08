"""
gui.py - 桌面 GUI 界面（重写版）
修复视频显示 + 管理员面板 + 强化学习 + 智能去重
v3: 显示/检测分离架构 — 毫秒级检测 + 高帧率显示
v4: 4.4 绘制优化 — 半透明叠加 + 颜色按ID + 抗锯齿 + Z-order
"""
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import cv2, threading, time, os, numpy as np
from queue import Queue, Empty
from person_identifier import PersonIdentifier

# 4.4 track_id 颜色哈希 — 不同ID不同颜色，每帧稳定
PALETTE = [
    (56, 189, 248), (34, 197, 94), (251, 146, 60), (244, 63, 94),
    (168, 85, 247), (14, 165, 233), (132, 204, 22), (255, 200, 50),
    (100, 180, 255), (255, 100, 180), (180, 255, 100), (200, 150, 255),
]


def _color_for(track_id: int) -> tuple:
    return PALETTE[track_id % len(PALETTE)]


def _draw_box(img, box, label, track_id, color=None, thickness=1, conf=None):
    """4.4 半透明叠加 + 抗锯齿 + 颜色按ID"""
    x1, y1, x2, y2 = map(int, box)
    if color is None:
        color = _color_for(track_id)
    # 半透明填充 (alpha=0.08)
    overlay = img.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
    cv2.addWeighted(overlay, 0.08, img, 0.92, 0, img)
    # 边框 (1px, 抗锯齿)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness, cv2.LINE_AA)
    # 标签背景 + 文字
    text = f"{label}#{track_id}" + (f" {conf:.0%}" if conf else "")
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
    ty = max(y1 - 4, th + 4)
    cv2.rectangle(img, (x1, ty - th - 4), (x1 + tw + 6, ty), color, -1, cv2.LINE_AA)
    cv2.putText(img, text, (x1 + 3, ty - 3),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)


class DetectionGUI:
    def __init__(self, config, camera_manager, detector, decision_engine,
                 alerter, logger, db, reinforcement):
        self._cfg = config["ui"]
        self._perf_cfg = config.get("performance", {})
        self._model_cfg = config.get("model", {})
        self._cameras = camera_manager
        self._detector = detector
        self._decision = decision_engine
        self._alerter = alerter
        self._logger = logger
        self._db = db
        self._rl = reinforcement
        self._running = False
        self._selected_cam = None
        self._alert_records = []
        self._frame_count = 0
        self._person_id = PersonIdentifier(time_window=30)
        self._photo_ref = None
        # 异步检测
        self._detect_queue = Queue(maxsize=2)  # 缓冲2帧，检测线程不丢帧
        self._detect_cache = {}  # cam_id -> last result
        self._detect_cache_lock = threading.Lock()
        self._gui_queue = Queue(maxsize=2)  # GUI渲染队列，缓冲2帧保证流畅
        # 心跳监控
        self._detect_heartbeat = time.perf_counter()  # 检测线程心跳
        self._heartbeat_lock = threading.Lock()
        self._detect_thread = None
        self._build_window()

    def _build_window(self):
        self._root = tk.Tk()
        self._root.title(self._cfg["title"])
        w, h = self._cfg["window_size"]
        self._root.geometry(f"{w}x{h}")
        self._root.configure(bg="#F5F8FC")

        # ===== 顶部标题栏 =====
        topbar = tk.Frame(self._root, bg="#1E5AA8", height=50)
        topbar.pack(fill=tk.X)
        tk.Label(topbar, text="AI 抽烟检测系统 - 安防智能体",
                 fg="white", bg="#1E5AA8",
                 font=("Microsoft YaHei", 16, "bold")).pack(side=tk.LEFT, padx=15, pady=10)

        self._perf_label = tk.Label(topbar, text="", fg="#B0C4DE", bg="#1E5AA8",
                                    font=("Consolas", 9))
        self._perf_label.pack(side=tk.LEFT, padx=10)

        # 管理员面板按钮
        tk.Button(topbar, text="审核面板", bg="#F59E0B", fg="white",
                  font=("Microsoft YaHei", 10),
                  command=self._open_admin).pack(side=tk.RIGHT, padx=5, pady=10)

        self._btn_start = tk.Button(topbar, text="▶ 开始检测", bg="#059669", fg="white",
                                    font=("Microsoft YaHei", 11), command=self.start)
        self._btn_start.pack(side=tk.RIGHT, padx=(5, 5), pady=10)

        self._btn_stop = tk.Button(topbar, text="■ 停止检测", bg="#DC2626", fg="white",
                                   font=("Microsoft YaHei", 11), command=self.stop,
                                   state=tk.DISABLED)
        self._btn_stop.pack(side=tk.RIGHT, padx=5, pady=10)

        # ===== 主体 =====
        body = tk.Frame(self._root, bg="#F5F8FC")
        body.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        # 左侧摄像头列表
        left = tk.Frame(body, bg="white", width=240, relief=tk.RIDGE, bd=1)
        left.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 5))
        left.pack_propagate(False)
        tk.Label(left, text="摄像头列表", bg="white",
                 font=("Microsoft YaHei", 12, "bold"),
                 anchor="w").pack(fill=tk.X, padx=10, pady=(10, 5))

        # 摄像头列表可滚动区域
        cam_canvas = tk.Canvas(left, bg="white", highlightthickness=0)
        cam_scrollbar = tk.Scrollbar(left, orient=tk.VERTICAL, command=cam_canvas.yview)
        self._cam_list_frame = tk.Frame(cam_canvas, bg="white")
        self._cam_list_frame.bind("<Configure>",
            lambda e: cam_canvas.configure(scrollregion=cam_canvas.bbox("all")))
        cam_canvas.create_window((0, 0), window=self._cam_list_frame, anchor="nw")
        cam_canvas.configure(yscrollcommand=cam_scrollbar.set)
        cam_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        cam_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # 添加摄像头按钮
        add_btn_frame = tk.Frame(left, bg="white")
        add_btn_frame.pack(fill=tk.X, padx=10, pady=(5, 10))
        self._add_cam_btn = tk.Button(add_btn_frame, text="+ 添加摄像头", bg="#1E5AA8", fg="white",
                                       font=("Microsoft YaHei", 10, "bold"),
                                       command=self._show_add_camera_dialog)
        self._add_cam_btn.pack(fill=tk.X, ipady=4)

        # 右侧视频画面
        right = tk.Frame(body, bg="#1A1A1A")
        right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        self._video_label = tk.Label(right, bg="#1A1A1A")
        self._video_label.pack(fill=tk.BOTH, expand=True)

        # 底部告警记录
        bottom = tk.Frame(self._root, bg="white", height=120, relief=tk.RIDGE, bd=1)
        bottom.pack(fill=tk.X, padx=10, pady=(0, 10))
        bottom.pack_propagate(False)
        tk.Label(bottom, text="告警记录", bg="white",
                 font=("Microsoft YaHei", 12, "bold"),
                 anchor="w").pack(fill=tk.X, padx=10, pady=(5, 0))
        self._alert_list = tk.Text(bottom, height=4, bg="white",
                                   font=("Consolas", 10),
                                   state=tk.DISABLED, wrap=tk.WORD)
        self._alert_list.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        # 管理员面板引用
        self._admin_panel = None
        from admin_panel import AdminPanel
        self._admin_panel = AdminPanel(self._db, self._rl, self._logger,
                                       on_judge_callback=self._on_admin_judge)

    def _open_admin(self):
        if self._admin_panel:
            self._admin_panel.show()

    def _on_admin_judge(self, evidence_id, is_correct):
        """管理员判断后更新阈值"""
        new_thresholds = self._rl.get_stats()["thresholds"]
        self._logger.info(f"RL阈值更新: {new_thresholds}")

    def _refresh_camera_list(self):
        for w in self._cam_list_frame.winfo_children():
            w.destroy()
        for cam in self._cameras.all_cameras:
            # 每行一个摄像头：开关按钮 + 名称标签
            row = tk.Frame(self._cam_list_frame, bg="white")
            row.pack(fill=tk.X, pady=3, padx=5)

            # 开关按钮
            toggle_text = "ON" if cam.enabled else "OFF"
            toggle_color = "#059669" if cam.enabled else "#DC2626"
            btn_toggle = tk.Button(row, text=toggle_text, bg=toggle_color, fg="white",
                                   font=("Microsoft YaHei", 9, "bold"), width=4,
                                   relief=tk.FLAT,
                                   command=lambda c=cam: self._toggle_camera(c))
            btn_toggle.pack(side=tk.LEFT, padx=(0, 5))

            # 摄像头名称（可点击选择）
            name_color = "#059669" if cam.enabled else "#9CA3AF"
            status_icon = "●" if cam.enabled else "○"
            lbl = tk.Label(row, text=f"{status_icon} {cam.name}", bg="white", fg=name_color,
                           font=("Microsoft YaHei", 10), anchor="w", cursor="hand2")
            lbl.pack(side=tk.LEFT, fill=tk.X, expand=True)
            lbl.bind("<Button-1>", lambda e, c=cam: self._select_camera(c))

    def _toggle_camera(self, cam):
        """切换摄像头开关状态"""
        if self._running:
            # 如果正在运行，需要停止再启动摄像头
            if cam.enabled:
                cam.stop()
                cam.enabled = False
                self._logger.info(f"摄像头已关闭: {cam.name}")
            else:
                cam.enabled = True
                try:
                    cam.start()
                    self._logger.info(f"摄像头已开启: {cam.name}")
                except RuntimeError as e:
                    cam.enabled = False
                    self._logger.info(f"摄像头开启失败: {e}")
        else:
            cam.enabled = not cam.enabled
            self._logger.info(f"摄像头{'开启' if cam.enabled else '关闭'}: {cam.name}")

        # 如果关闭的是当前选中的摄像头，切换到第一个活跃的
        if not cam.enabled and self._selected_cam == cam.id:
            active = self._cameras.active_cameras
            self._selected_cam = active[0].id if active else None

        self._save_camera_config()
        self._refresh_camera_list()

    def _save_camera_config(self):
        """保存摄像头配置到 config.yaml"""
        import yaml
        config_path = "config.yaml"
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f)
            cfg["cameras"] = self._cameras.to_config_list()
            with open(config_path, "w", encoding="utf-8") as f:
                yaml.dump(cfg, f, allow_unicode=True, default_flow_style=False, sort_keys=False)
        except Exception as e:
            self._logger.info(f"保存摄像头配置失败: {e}")

    def _select_camera(self, cam):
        if cam.enabled:
            self._selected_cam = cam.id

    def _show_add_camera_dialog(self):
        """显示添加摄像头对话框"""
        dialog = tk.Toplevel(self._root)
        dialog.title("添加摄像头")
        dialog.geometry("420x280")
        dialog.configure(bg="#F5F8FC")
        dialog.resizable(False, False)
        dialog.transient(self._root)
        dialog.grab_set()

        # 居中显示
        dialog.update_idletasks()
        rx = self._root.winfo_rootx()
        ry = self._root.winfo_rooty()
        rw = self._root.winfo_width()
        rh = self._root.winfo_height()
        dw = 420
        dh = 280
        x = rx + (rw - dw) // 2
        y = ry + (rh - dh) // 2
        dialog.geometry(f"+{x}+{y}")

        # 标题
        tk.Label(dialog, text="添加新摄像头", bg="#F5F8FC",
                 font=("Microsoft YaHei", 14, "bold")).pack(pady=(15, 10))

        # 类型选择
        type_frame = tk.Frame(dialog, bg="#F5F8FC")
        type_frame.pack(fill=tk.X, padx=30, pady=5)
        tk.Label(type_frame, text="类型:", bg="#F5F8FC",
                 font=("Microsoft YaHei", 10), width=8, anchor="w").pack(side=tk.LEFT)
        cam_type_var = tk.StringVar(value="USB")
        type_combo = ttk.Combobox(type_frame, textvariable=cam_type_var,
                                   values=["USB", "RTSP"], state="readonly", width=25)
        type_combo.pack(side=tk.LEFT)

        # 名称输入
        name_frame = tk.Frame(dialog, bg="#F5F8FC")
        name_frame.pack(fill=tk.X, padx=30, pady=5)
        tk.Label(name_frame, text="名称:", bg="#F5F8FC",
                 font=("Microsoft YaHei", 10), width=8, anchor="w").pack(side=tk.LEFT)
        name_var = tk.StringVar(value="")
        name_entry = tk.Entry(name_frame, textvariable=name_var, font=("Microsoft YaHei", 10), width=27)
        name_entry.pack(side=tk.LEFT)

        # 源地址输入
        src_frame = tk.Frame(dialog, bg="#F5F8FC")
        src_frame.pack(fill=tk.X, padx=30, pady=5)
        tk.Label(src_frame, text="源地址:", bg="#F5F8FC",
                 font=("Microsoft YaHei", 10), width=8, anchor="w").pack(side=tk.LEFT)
        src_var = tk.StringVar(value="0")
        src_entry = tk.Entry(src_frame, textvariable=src_var, font=("Microsoft YaHei", 10), width=27)
        src_entry.pack(side=tk.LEFT)

        hint_text = "USB: 输入数字索引(如 0, 1, 2)\nRTSP: 输入完整地址(如 rtsp://...)"
        tk.Label(dialog, text=hint_text, bg="#F5F8FC", fg="#6B7280",
                 font=("Microsoft YaHei", 8), justify=tk.LEFT).pack(pady=(5, 10))

        # 按钮
        btn_frame = tk.Frame(dialog, bg="#F5F8FC")
        btn_frame.pack(pady=10)

        error_var = tk.StringVar()
        error_label = tk.Label(dialog, textvariable=error_var, bg="#F5F8FC", fg="#DC2626",
                               font=("Microsoft YaHei", 9))

        def do_add():
            name = name_var.get().strip()
            source = src_var.get().strip()
            cam_type = cam_type_var.get()

            if not name:
                name = f"摄像头 {source}"

            if cam_type == "USB":
                try:
                    source = int(source)
                except ValueError:
                    error_var.set("USB摄像头源地址必须是数字")
                    error_label.pack()
                    return

            cam_id = f"cam_{len(self._cameras.all_cameras) + 1:02d}"
            self._cameras.add({
                "id": cam_id,
                "name": name,
                "source": source,
                "enabled": True,
            })
            self._save_camera_config()
            self._logger.info(f"已添加摄像头: {name} ({source})")
            self._refresh_camera_list()
            dialog.destroy()

        tk.Button(btn_frame, text="添加", bg="#059669", fg="white",
                  font=("Microsoft YaHei", 10, "bold"), width=10,
                  command=do_add).pack(side=tk.LEFT, padx=5)
        tk.Button(btn_frame, text="取消", bg="#9CA3AF", fg="white",
                  font=("Microsoft YaHei", 10), width=10,
                  command=dialog.destroy).pack(side=tk.LEFT, padx=5)

    def start(self):
        self._running = True
        self._btn_start.config(state=tk.DISABLED)
        self._btn_stop.config(state=tk.NORMAL)
        self._refresh_camera_list()
        self._logger.info("检测已启动（毫秒级实时安防管线）")
        # 清空队列（防止上次运行残留数据）
        self._clear_queue(self._detect_queue)
        self._clear_queue(self._gui_queue)
        with self._detect_cache_lock:
            self._detect_cache.clear()
        threading.Thread(target=self._display_loop, daemon=True).start()
        threading.Thread(target=self._gui_render_worker, daemon=True).start()
        self._detect_thread = threading.Thread(target=self._detection_worker, daemon=True)
        self._detect_thread.start()
        self._update_perf()

    def stop(self):
        self._running = False
        self._btn_start.config(state=tk.NORMAL)
        self._btn_stop.config(state=tk.DISABLED)
        # 等待线程退出
        time.sleep(0.15)
        # 清空队列
        self._clear_queue(self._detect_queue)
        self._clear_queue(self._gui_queue)
        # 停止所有摄像头
        for cam in self._cameras.all_cameras:
            try:
                cam.stop()
            except Exception:
                pass
        # 清空检测缓存
        with self._detect_cache_lock:
            self._detect_cache.clear()
        self._logger.info("检测已停止")

    @staticmethod
    def _clear_queue(q):
        """安全清空队列"""
        try:
            while True:
                q.get_nowait()
        except Empty:
            pass

    def _update_perf(self):
        if not self._running:
            return
        backend = self._detector.backend_name
        inf_ms = self._detector.avg_inference_time_ms
        stats = self._db.get_stats_for_display()

        # 心跳健康检查：检测线程是否卡死
        with self._heartbeat_lock:
            heartbeat_age = time.perf_counter() - self._detect_heartbeat
        detector_age = self._detector.heartbeat_age_sec

        status_text = f"后端:{backend} | 推理:{inf_ms:.1f}ms | "
        if heartbeat_age > 8.0 or detector_age > 8.0:
            # AI超过8秒无响应，尝试恢复
            status_text += "⚠ AI无响应"
            self._logger.info(f"[健康检查] 检测心跳{heartbeat_age:.1f}s, 推理心跳{detector_age:.1f}s — 尝试恢复")
            self._restart_detection_worker()
        else:
            status_text += f"准确率:{stats['accuracy']} | 待审:{stats['pending']}"

        self._perf_label.config(text=status_text)
        self._root.after(1000, self._update_perf)

    def _restart_detection_worker(self):
        """重启检测工作线程"""
        if self._detect_thread and self._detect_thread.is_alive():
            return  # 线程还在运行，不需要重启
        self._logger.info("[恢复] 重新启动检测线程...")
        self._clear_queue(self._detect_queue)
        with self._detect_cache_lock:
            self._detect_cache.clear()
        with self._heartbeat_lock:
            self._detect_heartbeat = time.perf_counter()
        self._detect_thread = threading.Thread(target=self._detection_worker, daemon=True)
        self._detect_thread.start()

    # ================================================================
    #  检测工作线程 — 异步 YOLO 推理，1ms 超时极速轮询
    #  v2: 添加异常保护，防止检测线程静默崩溃导致方框冻结
    # ================================================================
    def _detection_worker(self):
        """独立检测线程：极速轮询 → 推理 → 即时告警推送"""
        last_alert_time = {}
        stall_count = 0
        while self._running:
            try:
                item = self._detect_queue.get(timeout=0.001)  # 1ms超时，毫秒级响应
            except Empty:
                stall_count += 1
                # 如果超过5秒没收到新帧，发出警告
                if stall_count > 5000:
                    self._logger.info("[心跳] 检测队列空闲超过5秒，等待新帧...")
                    stall_count = 0
                continue

            stall_count = 0
            cam, frame = item
            t0 = time.perf_counter()

            try:
                # 多级检测
                result = self._detector.detect(frame)
            except Exception as e:
                # 检测异常：记录错误，清空该摄像头缓存，防止显示冻结方框
                self._logger.info(f"[检测异常] {cam.name}: {e}")
                with self._detect_cache_lock:
                    self._detect_cache.pop(cam.id, None)
                continue

            # 更新心跳
            with self._heartbeat_lock:
                self._detect_heartbeat = time.perf_counter()

            # 更新缓存（供显示线程使用，带时间戳用于TTL过期检查）
            with self._detect_cache_lock:
                self._detect_cache[cam.id] = (result, time.perf_counter())

            # RL 阈值过滤
            valid_alerts = []
            for alert in result["alerts"]:
                if alert["confidence"] < self._rl.get_threshold(alert["label"]):
                    continue
                valid_alerts.append(alert)

            # 人物去重
            deduped_alerts = []
            for alert in valid_alerts:
                person_key = f"{cam.id}_{alert['track_id']}"
                now = time.time()
                if person_key in last_alert_time:
                    lt, lc, _ = last_alert_time[person_key]
                    if now - lt < 30:
                        if alert["confidence"] > lc:
                            last_alert_time[person_key] = (now, alert["confidence"], alert["bbox"])
                        continue
                last_alert_time[person_key] = (now, alert["confidence"], alert["bbox"])
                deduped_alerts.append(alert)

            # 决策链：空间推理 + 时间确认
            if deduped_alerts:
                level, detail = self._decision.evaluate(cam.id, deduped_alerts)
                if level:
                    self._push_alert(cam, frame, result, deduped_alerts, detail, level)

            # 记录毫秒级性能
            elapsed = (time.perf_counter() - t0) * 1000
            if elapsed > 20:
                self._logger.info(f"检测耗时: {elapsed:.1f}ms")

    def _push_alert(self, cam, frame, result, deduped_alerts, detail, level):
        """即时推送告警：保存截图 + 写入数据库 + 更新前端"""
        best_alert = max(deduped_alerts, key=lambda a: a["confidence"])
        detail["detected_label"] = best_alert["label"]
        detail["confidence"] = best_alert["confidence"]
        detail["level"] = level

        # 可疑级别（hand_cigs_far）和确认抽烟（smoking）都入库
        # 仅手近脸（hand_near）不入库，仅前端提示
        if level == "hand_near":
            self._root.after(0, lambda c=cam, d=detail: self._on_alert(c, d))
            self._logger.info(f"手近脸提示: {cam.name}")
            return

        # 提取人物特征
        person_features = {}
        track_id = best_alert.get("track_id", 0)
        for p in result["persons"]:
            if p["id"] == track_id:
                person_features = self._person_id.extract_features(frame, p["bbox"])
                break

        # DB 级去重
        if person_features:
            similar_id, similar_conf = self._db.find_recent_similar_person(
                cam.id, person_features, time_window=30)
            if similar_id and best_alert["confidence"] <= similar_conf:
                return
            if similar_id and best_alert["confidence"] > similar_conf:
                alert_dir = "alerts"
                os.makedirs(alert_dir, exist_ok=True)
                ts = time.strftime("%Y%m%d_%H%M%S")
                img_path = os.path.join(alert_dir, f"{cam.id}_{best_alert['label']}_{ts}.jpg")
                cv2.imwrite(img_path, frame)
                self._db.replace_evidence_image(similar_id, img_path, best_alert["confidence"])
                self._logger.info(f"替换证据: #{similar_id} -> {best_alert['confidence']:.0%}")
                return

        # 保存截图
        alert_dir = "alerts"
        os.makedirs(alert_dir, exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        label = best_alert["label"]
        img_path = os.path.join(alert_dir, f"{cam.id}_{label}_{ts}.jpg")
        cv2.imwrite(img_path, frame)

        # 写入数据库（含 BLOB 长期存储 + 级别标记）
        self._db.add_evidence(
            cam.id, cam.name, f"person_{track_id}",
            img_path, label, best_alert["confidence"],
            best_alert["bbox"], person_features)

        # 更新前端告警列表
        self._root.after(0, lambda c=cam, d=detail: self._on_alert(c, d))
        level_cn = "确认抽烟" if level == "smoking" else "可疑"
        self._logger.info(f"[{level_cn}] {cam.name} {label} {best_alert['confidence']:.0%}")

    # ================================================================
    #  显示主循环 — 极速帧读取 + 速度插值追踪 + 检测框绘制
    #  v10: 速度插值 — 方框追踪帧率与视频帧率一致
    #       检测结果用于更新运动模型，显示时用速度预测插值位置
    # ================================================================
    def _display_loop(self):
        """显示主循环：读取帧 → resize → 速度插值预测 → 叠加检测框 → 推送 GUI 渲染队列"""
        started_cams = set()
        for cam in self._cameras.active_cameras:
            try:
                cam.start()
                started_cams.add(cam.id)
            except RuntimeError as e:
                self._logger.info(str(e))

        active = self._cameras.active_cameras
        if active and not self._selected_cam:
            self._selected_cam = active[0].id

        disp_w = self._perf_cfg.get("display_width", 960)
        disp_h = self._perf_cfg.get("display_height", 540)
        RESULT_TTL = 1.5  # 检测结果超过1.5秒即丢弃
        frame_idx = 0
        last_fps_update = time.perf_counter()
        fps_frame_count = 0
        display_fps = 0.0
        none_frame_count = 0
        _last_result_ts = {}  # cam_id -> float (上次检测结果的时间戳)
        # 运动模型：cam_id -> {track_id -> {"bbox": [x1,y1,x2,y2], "vx": float, "vy": float, "ts": float}}
        _head_motion = {}

        while self._running:
            for cam in self._cameras.active_cameras:
                if cam.id not in started_cams:
                    try:
                        cam.start()
                        started_cams.add(cam.id)
                    except RuntimeError as e:
                        self._logger.info(str(e))
                        continue

                frame = cam.read()
                if frame is None:
                    none_frame_count += 1
                    if none_frame_count == 1 or none_frame_count % 100 == 0:
                        self._logger.info(f"[警告] {cam.name} 连续{none_frame_count}帧为空，摄像头可能断开")
                    continue
                none_frame_count = 0

                frame_idx += 1

                # 每帧都推送检测（非阻塞，队列满则丢弃）
                try:
                    self._detect_queue.put_nowait((cam, frame.copy()))
                except Exception:
                    pass

                # 只为选中摄像头做显示处理
                if cam.id == self._selected_cam:
                    now = time.perf_counter()

                    # 检查检测线程是否存活
                    detect_thread_alive = self._detect_thread and self._detect_thread.is_alive()

                    # 从缓存取最新检测结果
                    result = None
                    result_ts = 0.0
                    with self._detect_cache_lock:
                        cache_entry = self._detect_cache.get(cam.id)
                    if cache_entry is not None:
                        result, result_ts = cache_entry

                    # 判断是否为新检测结果（时间戳不同）
                    last_ts = _last_result_ts.get(cam.id, 0.0)
                    is_new_result = (result is not None and result_ts > last_ts + 0.001)

                    # 更新运动模型（新检测结果到达时）
                    if is_new_result and result:
                        _last_result_ts[cam.id] = result_ts
                        cam_motion = _head_motion.setdefault(cam.id, {})
                        active_tids = set()

                        for h in result.get("heads", []):
                            tid = h["track_id"]
                            x1, y1, x2, y2 = h["bbox"]
                            if x2 <= x1 + 10 or y2 <= y1 + 10:
                                continue
                            active_tids.add(tid)
                            cx = (x1 + x2) / 2.0
                            cy = (y1 + y2) / 2.0

                            if tid in cam_motion:
                                prev = cam_motion[tid]
                                dt = result_ts - prev["ts"]
                                if dt > 0.01:
                                    # 计算速度（EMA平滑）
                                    prev_cx = (prev["bbox"][0] + prev["bbox"][2]) / 2.0
                                    prev_cy = (prev["bbox"][1] + prev["bbox"][3]) / 2.0
                                    raw_vx = (cx - prev_cx) / dt
                                    raw_vy = (cy - prev_cy) / dt
                                    prev["vx"] = prev.get("vx", 0.0) * 0.35 + raw_vx * 0.65
                                    prev["vy"] = prev.get("vy", 0.0) * 0.35 + raw_vy * 0.65
                                prev["bbox"] = [x1, y1, x2, y2]
                                prev["ts"] = result_ts
                            else:
                                cam_motion[tid] = {
                                    "bbox": [x1, y1, x2, y2],
                                    "vx": 0.0, "vy": 0.0,
                                    "ts": result_ts,
                                }

                        # 清理已经不存在的 track_id
                        stale = [tid for tid in cam_motion if tid not in active_tids]
                        for tid in stale:
                            del cam_motion[tid]

                    # TTL与线程存活检查
                    result_age = now - result_ts if result_ts > 0 else 999
                    result_expired = result_age > RESULT_TTL
                    if result_expired or not detect_thread_alive:
                        result = None

                    # resize 到显示尺寸
                    display_frame = cv2.resize(frame, (disp_w, disp_h), interpolation=cv2.INTER_LINEAR)
                    scale_x = disp_w / frame.shape[1]
                    scale_y = disp_h / frame.shape[0]

                    # 绘制头部追踪框（速度插值：检测帧率 ≠ 显示帧率也能平滑追踪）
                    cam_motion = _head_motion.get(cam.id, {})
                    if cam_motion and not result_expired and detect_thread_alive:
                        for tid, motion in cam_motion.items():
                            dt = now - motion["ts"]
                            # 速度预测当前位置
                            bbox = motion["bbox"]
                            vx = motion.get("vx", 0.0)
                            vy = motion.get("vy", 0.0)
                            bw = bbox[2] - bbox[0]
                            bh = bbox[3] - bbox[1]
                            pred_cx = (bbox[0] + bbox[2]) / 2.0 + vx * dt
                            pred_cy = (bbox[1] + bbox[3]) / 2.0 + vy * dt

                            x1 = int((pred_cx - bw / 2.0) * scale_x)
                            y1 = int((pred_cy - bh / 2.0) * scale_y)
                            x2 = int((pred_cx + bw / 2.0) * scale_x)
                            y2 = int((pred_cy + bh / 2.0) * scale_y)

                            if x2 <= x1 + 10 or y2 <= y1 + 10:
                                continue
                            cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            cv2.putText(display_frame, f"Head#{tid}",
                                        (x1, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX,
                                        0.5, (0, 255, 0), 2)

                    # FPS + 状态显示
                    if self._perf_cfg.get("display_fps", True):
                        detect_fps = self._detector.current_fps
                        status = "LIVE" if detect_thread_alive else "DEAD"
                        tracking = "INTERP" if cam_motion and not result_expired else "IDLE"
                        cv2.putText(display_frame,
                                    f"Display:{display_fps:.0f}fps | Detect:{detect_fps:.0f}fps | Age:{result_age:.1f}s | {tracking}",
                                    (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

                    # 推送 GUI 渲染
                    try:
                        self._gui_queue.put_nowait(display_frame)
                    except Exception:
                        pass

                    # 显示 FPS 统计
                    fps_frame_count += 1
                    if now - last_fps_update >= 1.0:
                        display_fps = fps_frame_count / (now - last_fps_update)
                        last_fps_update = now
                        fps_frame_count = 0

                self._frame_count += 1

        # 循环退出时清理摄像头
        for cam in self._cameras.all_cameras:
            try:
                cam.stop()
            except Exception:
                pass

    # ================================================================
    #  GUI 渲染线程 — 仅做 BGR→RGB 转换 + PhotoImage（帧已在 display_loop 中 resize）
    # ================================================================
    def _gui_render_worker(self):
        """独立 GUI 渲染线程：从队列取帧 → BGR→RGB → PhotoImage → 更新 Tkinter"""
        while self._running:
            try:
                frame = self._gui_queue.get(timeout=0.01)
            except Empty:
                continue
            try:
                # 帧已在 display_loop 中 resize 到目标尺寸，这里直接转换
                img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                photo = ImageTk.PhotoImage(Image.fromarray(img))
                self._photo_ref = photo
                self._root.after(0, self._safe_update_video, photo)
            except Exception:
                pass
        # 退出时清理引用
        self._photo_ref = None

    def _safe_update_video(self, photo):
        """安全更新视频帧"""
        try:
            self._video_label.config(image=photo)
            self._video_label.image = photo
        except Exception:
            pass

    def _on_alert(self, cam, detail):
        max_history = self._cfg["alert_history_max"]
        label = detail.get("detected_label", "检测")
        conf = detail.get("confidence", 0)
        record = f"[{detail['timestamp']}] {cam.name}  {label}  置信度{conf:.0%}"
        self._alert_records.append(record)
        if len(self._alert_records) > max_history:
            self._alert_records = self._alert_records[-max_history:]

        self._alert_list.config(state=tk.NORMAL)
        self._alert_list.delete(1.0, tk.END)
        self._alert_list.insert(tk.END, "\n".join(self._alert_records))
        self._alert_list.see(tk.END)
        self._alert_list.config(state=tk.DISABLED)

    def run(self):
        self._refresh_camera_list()
        self._root.mainloop()