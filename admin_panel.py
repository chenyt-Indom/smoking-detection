"""
admin_panel.py - 管理员审核面板 v2
标签页：待审核 | 全部记录 | 模型统计
功能：判断正误、标注、删除、筛选、学习曲线、图片长期存储
"""
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from PIL import Image, ImageTk
import os, io, json, threading, time
from datetime import datetime


class AdminPanel:
    """管理员审核面板 - 独立窗口"""

    def __init__(self, db, reinforcement, logger, on_judge_callback=None):
        self._db = db
        self._rl = reinforcement
        self._logger = logger
        self._on_judge = on_judge_callback
        self._current_evidence = None
        self._photo_ref = None
        self._window = None
        self._build()

    def _build(self):
        self._window = tk.Toplevel()
        self._window.title("管理员审核面板 - AI抽烟检测系统")
        self._window.geometry("1100x750")
        self._window.configure(bg="#F5F8FC")

        # 顶部标题栏
        top = tk.Frame(self._window, bg="#1E5AA8", height=48)
        top.pack(fill=tk.X)
        tk.Label(top, text="管理员审核面板 - 截图证据管理",
                 fg="white", bg="#1E5AA8",
                 font=("Microsoft YaHei", 14, "bold")).pack(side=tk.LEFT, padx=15, pady=10)

        self._stats_label = tk.Label(top, text="", fg="#B0C4DE", bg="#1E5AA8",
                                     font=("Consolas", 10))
        self._stats_label.pack(side=tk.RIGHT, padx=15, pady=10)

        # 标签页
        self._notebook = ttk.Notebook(self._window)
        self._notebook.pack(fill=tk.BOTH, expand=True, padx=8, pady=5)

        # === 标签1: 待审核 ===
        self._tab_pending = tk.Frame(self._notebook, bg="#F5F8FC")
        self._notebook.add(self._tab_pending, text="  待审核  ")
        self._build_pending_tab()

        # === 标签2: 全部记录 ===
        self._tab_all = tk.Frame(self._notebook, bg="#F5F8FC")
        self._notebook.add(self._tab_all, text="  全部记录  ")
        self._build_all_records_tab()

        # === 标签3: 模型统计 ===
        self._tab_stats = tk.Frame(self._notebook, bg="#F5F8FC")
        self._notebook.add(self._tab_stats, text="  模型统计  ")
        self._build_stats_tab()

        self._refresh()

    # ================================================================
    #  标签1: 待审核
    # ================================================================
    def _build_pending_tab(self):
        body = tk.Frame(self._tab_pending, bg="#F5F8FC")
        body.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # 左侧: 截图预览
        left = tk.Frame(body, bg="white", width=500, relief=tk.RIDGE, bd=1)
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        left.pack_propagate(False)
        tk.Label(left, text="截图证据", bg="white",
                 font=("Microsoft YaHei", 12, "bold"),
                 anchor="w").pack(fill=tk.X, padx=10, pady=(8, 3))
        self._image_label = tk.Label(left, bg="#1A1A1A", text="暂无待审核截图",
                                     font=("Microsoft YaHei", 12), fg="#888")
        self._image_label.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        self._info_text = tk.Text(left, height=7, bg="white", font=("Consolas", 10))
        self._info_text.pack(fill=tk.X, padx=10, pady=5)

        # 右侧: 操作
        right = tk.Frame(body, bg="white", width=300, relief=tk.RIDGE, bd=1)
        right.pack(side=tk.RIGHT, fill=tk.Y)
        right.pack_propagate(False)

        tk.Label(right, text="判断操作", bg="white",
                 font=("Microsoft YaHei", 12, "bold"),
                 anchor="w").pack(fill=tk.X, padx=10, pady=(10, 10))

        self._btn_correct = tk.Button(right, text="✓ 判断正确 (+1分)", bg="#059669", fg="white",
                                       font=("Microsoft YaHei", 11, "bold"),
                                       command=self._judge_correct,
                                       state=tk.DISABLED, height=2)
        self._btn_correct.pack(fill=tk.X, padx=15, pady=5)

        self._btn_wrong = tk.Button(right, text="✗ 判断错误 (-1分)", bg="#DC2626", fg="white",
                                     font=("Microsoft YaHei", 11, "bold"),
                                     command=self._judge_wrong,
                                     state=tk.DISABLED, height=2)
        self._btn_wrong.pack(fill=tk.X, padx=15, pady=5)

        ttk.Separator(right, orient=tk.HORIZONTAL).pack(fill=tk.X, padx=15, pady=8)

        tk.Label(right, text="管理员备注:", bg="white",
                 font=("Microsoft YaHei", 10), anchor="w").pack(fill=tk.X, padx=15, pady=(0, 2))
        self._note_entry = tk.Text(right, height=4, font=("Microsoft YaHei", 10))
        self._note_entry.pack(fill=tk.X, padx=15, pady=5)

        ttk.Separator(right, orient=tk.HORIZONTAL).pack(fill=tk.X, padx=15, pady=8)

        tk.Label(right, text="标注信息:", bg="white",
                 font=("Microsoft YaHei", 10), anchor="w").pack(fill=tk.X, padx=15, pady=(0, 2))
        self._annotation_entry = tk.Text(right, height=3, font=("Microsoft YaHei", 10))
        self._annotation_entry.pack(fill=tk.X, padx=15, pady=5)

        btn_frame = tk.Frame(right, bg="white")
        btn_frame.pack(fill=tk.X, padx=15, pady=10)
        tk.Button(btn_frame, text="保存标注", bg="#3B82F6", fg="white",
                  font=("Microsoft YaHei", 9), command=self._save_annotation).pack(side=tk.LEFT, padx=2)
        tk.Button(btn_frame, text="刷新", bg="#6B7280", fg="white",
                  font=("Microsoft YaHei", 9), command=self._refresh).pack(side=tk.LEFT, padx=2)
        tk.Button(btn_frame, text="跳过", bg="#9CA3AF", fg="white",
                  font=("Microsoft YaHei", 9), command=self._skip).pack(side=tk.RIGHT, padx=2)
        tk.Button(btn_frame, text="删除", bg="#DC2626", fg="white",
                  font=("Microsoft YaHei", 9), command=self._delete_current).pack(side=tk.RIGHT, padx=2)

    # ================================================================
    #  标签2: 全部记录
    # ================================================================
    def _build_all_records_tab(self):
        # 顶部筛选栏
        filter_bar = tk.Frame(self._tab_all, bg="#E8EDF4", height=40)
        filter_bar.pack(fill=tk.X, padx=5, pady=(5, 0))
        filter_bar.pack_propagate(False)

        tk.Label(filter_bar, text="筛选:", bg="#E8EDF4", font=("Microsoft YaHei", 10)).pack(side=tk.LEFT, padx=5)

        tk.Label(filter_bar, text="标签:", bg="#E8EDF4", font=("Microsoft YaHei", 9)).pack(side=tk.LEFT, padx=(10, 2))
        self._filter_label = ttk.Combobox(filter_bar, values=["全部"], state="readonly", width=12)
        self._filter_label.pack(side=tk.LEFT, padx=2)
        self._filter_label.set("全部")

        tk.Label(filter_bar, text="判定:", bg="#E8EDF4", font=("Microsoft YaHei", 9)).pack(side=tk.LEFT, padx=(10, 2))
        self._filter_judgment = ttk.Combobox(filter_bar, values=["全部", "pending", "correct", "incorrect"],
                                             state="readonly", width=10)
        self._filter_judgment.pack(side=tk.LEFT, padx=2)
        self._filter_judgment.set("全部")

        tk.Label(filter_bar, text="摄像头:", bg="#E8EDF4", font=("Microsoft YaHei", 9)).pack(side=tk.LEFT, padx=(10, 2))
        self._filter_camera = ttk.Combobox(filter_bar, values=["全部"], state="readonly", width=14)
        self._filter_camera.pack(side=tk.LEFT, padx=2)
        self._filter_camera.set("全部")

        tk.Button(filter_bar, text="搜索", bg="#3B82F6", fg="white",
                  font=("Microsoft YaHei", 9), command=self._search_records).pack(side=tk.LEFT, padx=10)

        # 分页
        self._page_label = tk.Label(filter_bar, text="", bg="#E8EDF4", font=("Microsoft YaHei", 9))
        self._page_label.pack(side=tk.RIGHT, padx=5)
        tk.Button(filter_bar, text=">", bg="#6B7280", fg="white", font=("Microsoft YaHei", 8), width=2,
                  command=self._next_page).pack(side=tk.RIGHT, padx=1)
        tk.Button(filter_bar, text="<", bg="#6B7280", fg="white", font=("Microsoft YaHei", 8), width=2,
                  command=self._prev_page).pack(side=tk.RIGHT, padx=1)

        self._page_offset = 0
        self._page_size = 20

        # 记录列表（Treeview + 滚动条）
        list_frame = tk.Frame(self._tab_all, bg="white")
        list_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        columns = ("id", "时间", "摄像头", "标签", "置信度", "判定", "备注")
        self._record_tree = ttk.Treeview(list_frame, columns=columns, show="headings", height=15)
        for col in columns:
            self._record_tree.heading(col, text=col)
            w = 60 if col == "id" else 140 if col == "时间" else 100 if col in ("标签", "判定") else 80
            self._record_tree.column(col, width=w, anchor="center")

        scrollbar = ttk.Scrollbar(list_frame, orient=tk.VERTICAL, command=self._record_tree.yview)
        self._record_tree.configure(yscrollcommand=scrollbar.set)
        self._record_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self._record_tree.bind("<Double-1>", self._on_record_double_click)

        # 操作按钮
        btn_bar = tk.Frame(self._tab_all, bg="#F5F8FC")
        btn_bar.pack(fill=tk.X, padx=5, pady=5)
        tk.Button(btn_bar, text="查看详情", bg="#3B82F6", fg="white",
                  font=("Microsoft YaHei", 9), command=self._view_record_detail).pack(side=tk.LEFT, padx=3)
        tk.Button(btn_bar, text="删除选中", bg="#DC2626", fg="white",
                  font=("Microsoft YaHei", 9), command=self._delete_selected).pack(side=tk.LEFT, padx=3)
        tk.Button(btn_bar, text="导出CSV", bg="#059669", fg="white",
                  font=("Microsoft YaHei", 9), command=self._export_csv).pack(side=tk.LEFT, padx=3)

    # ================================================================
    #  标签3: 模型统计
    # ================================================================
    def _build_stats_tab(self):
        body = tk.Frame(self._tab_stats, bg="#F5F8FC")
        body.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # 模型评分
        score_frame = tk.LabelFrame(body, text="模型评分", bg="white",
                                    font=("Microsoft YaHei", 11, "bold"), padx=10, pady=10)
        score_frame.pack(fill=tk.X, pady=(0, 10))
        self._score_text = tk.Text(score_frame, height=6, bg="white", font=("Consolas", 10))
        self._score_text.pack(fill=tk.X)

        # 学习曲线
        curve_frame = tk.LabelFrame(body, text="强化学习状态", bg="white",
                                     font=("Microsoft YaHei", 11, "bold"), padx=10, pady=10)
        curve_frame.pack(fill=tk.BOTH, expand=True)
        self._curve_text = tk.Text(curve_frame, height=10, bg="white", font=("Consolas", 10))
        self._curve_text.pack(fill=tk.BOTH, expand=True)

        btn_frame = tk.Frame(body, bg="#F5F8FC")
        btn_frame.pack(fill=tk.X, pady=5)
        tk.Button(btn_frame, text="刷新统计", bg="#3B82F6", fg="white",
                  font=("Microsoft YaHei", 10), command=self._update_scores).pack(side=tk.LEFT)

    # ================================================================
    #  刷新
    # ================================================================
    def _refresh(self):
        pending = self._db.get_pending_evidence()
        if pending:
            self._load_evidence(pending[0])
        else:
            self._current_evidence = None
            self._image_label.config(image="", text="暂无待审核截图\n\n所有截图已审核完毕!",
                                     font=("Microsoft YaHei", 14), fg="#059669")
            self._btn_correct.config(state=tk.DISABLED)
            self._btn_wrong.config(state=tk.DISABLED)
            self._info_text.config(state=tk.NORMAL)
            self._info_text.delete(1.0, tk.END)
            self._info_text.insert(1.0, "等待新的检测截图...")
            self._info_text.config(state=tk.DISABLED)
        self._update_scores()
        self._update_filter_combos()
        self._search_records()

    def _update_filter_combos(self):
        labels = ["全部"] + self._db.get_all_labels()
        self._filter_label["values"] = labels
        cameras = ["全部"] + [c["name"] for c in self._db.get_all_cameras()]
        self._filter_camera["values"] = cameras

    def _load_evidence(self, evidence):
        self._current_evidence = evidence
        img_path = evidence["image_path"]

        # 尝试从文件加载，失败则从数据库BLOB加载
        img = None
        if os.path.exists(img_path):
            try:
                img = Image.open(img_path)
            except Exception:
                img = None

        if img is None:
            blob = self._db.get_image_blob(evidence["id"])
            if blob:
                try:
                    img = Image.open(io.BytesIO(blob))
                except Exception:
                    img = None

        if img:
            img.thumbnail((480, 400))
            photo = ImageTk.PhotoImage(img)
            self._image_label.config(image=photo, text="")
            self._image_label.image = photo
            self._photo_ref = photo
        else:
            self._image_label.config(image="", text="图片加载失败")

        info = (
            f"ID: #{evidence['id']}\n"
            f"摄像头: {evidence.get('camera_name', 'N/A')}\n"
            f"时间: {evidence.get('timestamp', 'N/A')}\n"
            f"检测类型: {evidence.get('label', 'N/A')}\n"
            f"置信度: {evidence.get('confidence', 0):.1%}\n"
            f"人物ID: {evidence.get('person_id', 'N/A')}\n"
            f"标注: {evidence.get('admin_annotation') or '(无)'}"
        )
        self._info_text.config(state=tk.NORMAL)
        self._info_text.delete(1.0, tk.END)
        self._info_text.insert(1.0, info)
        self._info_text.config(state=tk.DISABLED)

        self._btn_correct.config(state=tk.NORMAL)
        self._btn_wrong.config(state=tk.NORMAL)

        # 加载已有标注
        self._annotation_entry.delete(1.0, tk.END)
        if evidence.get("admin_annotation"):
            self._annotation_entry.insert(1.0, evidence["admin_annotation"])

    def _judge_correct(self):
        self._do_judge(True)

    def _judge_wrong(self):
        self._do_judge(False)

    def _do_judge(self, is_correct):
        if not self._current_evidence:
            return
        eid = self._current_evidence["id"]
        note = self._note_entry.get(1.0, tk.END).strip()
        self._db.judge_evidence(eid, is_correct, note)

        label = self._current_evidence["label"]
        conf = self._current_evidence["confidence"]
        self._rl.process_feedback(label, is_correct, conf)

        result = "正确" if is_correct else "错误"
        self._logger.info(f"管理员判断: #{eid} -> {result} ({label})")
        self._note_entry.delete(1.0, tk.END)

        if self._on_judge:
            self._on_judge(eid, is_correct)
        self._refresh()

    def _save_annotation(self):
        if not self._current_evidence:
            return
        annotation = self._annotation_entry.get(1.0, tk.END).strip()
        if annotation:
            self._db.annotate_evidence(self._current_evidence["id"], annotation)
            self._logger.info(f"已保存标注: #{self._current_evidence['id']}")

    def _skip(self):
        if not self._current_evidence:
            return
        pending = self._db.get_pending_evidence()
        if len(pending) > 1:
            idx = next((i for i, e in enumerate(pending) if e["id"] == self._current_evidence["id"]), -1)
            next_idx = (idx + 1) % len(pending)
            self._load_evidence(pending[next_idx])
        else:
            self._refresh()

    def _delete_current(self):
        if not self._current_evidence:
            return
        if messagebox.askyesno("确认删除", f"确定删除 #{self._current_evidence['id']} 吗？\n图片文件和数据库记录都将被删除。"):
            self._db.delete_evidence(self._current_evidence["id"])
            self._logger.info(f"已删除: #{self._current_evidence['id']}")
            self._refresh()

    def _delete_selected(self):
        sel = self._record_tree.selection()
        if not sel:
            return
        item = self._record_tree.item(sel[0])
        eid = item["values"][0]
        if messagebox.askyesno("确认删除", f"确定删除 #{eid} 吗？"):
            self._db.delete_evidence(eid)
            self._logger.info(f"已删除: #{eid}")
            self._search_records()

    # ================================================================
    #  全部记录: 搜索、分页
    # ================================================================
    def _search_records(self):
        filters = {}
        label = self._filter_label.get()
        if label and label != "全部":
            filters["label"] = label
        judgment = self._filter_judgment.get()
        if judgment and judgment != "全部":
            filters["judgment"] = judgment
        camera = self._filter_camera.get()
        if camera and camera != "全部":
            cameras = self._db.get_all_cameras()
            for c in cameras:
                if c["name"] == camera:
                    filters["camera_id"] = c["id"]
                    break

        total = self._db.get_evidence_count(filters)
        records = self._db.get_all_evidence(filters, limit=self._page_size, offset=self._page_offset)

        self._record_tree.delete(*self._record_tree.get_children())
        for r in records:
            judgment = r.get("admin_judgment", "pending")
            j_display = {"pending": "待审", "correct": "正确", "incorrect": "错误"}.get(judgment, judgment)
            self._record_tree.insert("", "end", values=(
                r["id"],
                (r.get("timestamp") or "")[:19],
                r.get("camera_name", ""),
                r.get("label", ""),
                f"{r.get('confidence', 0):.1%}",
                j_display,
                (r.get("admin_note") or "")[:30],
            ))

        total_pages = max(1, (total + self._page_size - 1) // self._page_size)
        current_page = self._page_offset // self._page_size + 1
        self._page_label.config(text=f"共 {total} 条 | 第 {current_page}/{total_pages} 页")

    def _next_page(self):
        self._page_offset += self._page_size
        self._search_records()

    def _prev_page(self):
        self._page_offset = max(0, self._page_offset - self._page_size)
        self._search_records()

    def _on_record_double_click(self, event):
        self._view_record_detail()

    def _view_record_detail(self):
        sel = self._record_tree.selection()
        if not sel:
            return
        item = self._record_tree.item(sel[0])
        eid = item["values"][0]
        evidence = self._db.get_evidence_by_id(eid)
        if not evidence:
            return

        # 弹出详情窗口
        detail = tk.Toplevel(self._window)
        detail.title(f"证据详情 #{eid}")
        detail.geometry("700x550")
        detail.configure(bg="#F5F8FC")

        # 图片
        img_frame = tk.Frame(detail, bg="white", height=350)
        img_frame.pack(fill=tk.X, padx=10, pady=10)
        img_frame.pack_propagate(False)

        img = None
        if os.path.exists(evidence["image_path"]):
            try:
                img = Image.open(evidence["image_path"])
            except Exception:
                pass
        if img is None:
            blob = self._db.get_image_blob(eid)
            if blob:
                try:
                    img = Image.open(io.BytesIO(blob))
                except Exception:
                    pass

        if img:
            img.thumbnail((650, 330))
            photo = ImageTk.PhotoImage(img)
            lbl = tk.Label(img_frame, image=photo, bg="white")
            lbl.image = photo
            lbl.pack(pady=5)

        # 信息
        info_frame = tk.Frame(detail, bg="white")
        info_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        info = (
            f"ID: #{evidence['id']}\n"
            f"摄像头: {evidence.get('camera_name', '')}\n"
            f"时间: {evidence.get('timestamp', '')}\n"
            f"检测类型: {evidence.get('label', '')}\n"
            f"置信度: {evidence.get('confidence', 0):.1%}\n"
            f"人物ID: {evidence.get('person_id', '')}\n"
            f"判定: {evidence.get('admin_judgment', 'pending')}\n"
            f"分数: {evidence.get('admin_score', 0)}\n"
            f"备注: {evidence.get('admin_note') or ''}\n"
            f"标注: {evidence.get('admin_annotation') or ''}\n"
            f"创建时间: {evidence.get('created_at', '')}\n"
            f"判定时间: {evidence.get('judged_at', '')}"
        )
        info_text = tk.Text(info_frame, font=("Consolas", 10), height=12)
        info_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        info_text.insert(1.0, info)
        info_text.config(state=tk.DISABLED)

    def _export_csv(self):
        path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV", "*.csv")])
        if not path:
            return
        records = self._db.get_all_evidence(limit=5000)
        with open(path, "w", encoding="utf-8-sig") as f:
            f.write("ID,时间,摄像头,标签,置信度,判定,备注,标注\n")
            for r in records:
                f.write(f"{r['id']},{r.get('timestamp','')},{r.get('camera_name','')},"
                        f"{r.get('label','')},{r.get('confidence',0):.3f},"
                        f"{r.get('admin_judgment','')},{(r.get('admin_note') or '')},{(r.get('admin_annotation') or '')}\n")
        self._logger.info(f"已导出CSV: {path}")

    def _update_scores(self):
        stats = self._db.get_stats_for_display()
        rl_stats = self._rl.get_stats()
        fb_stats = self._db.get_feedback_stats()

        self._stats_label.config(
            text=f"正确:{stats['total_correct']} | 错误:{stats['total_incorrect']} | "
                 f"准确率:{stats['accuracy']} | 待审:{stats['pending']}"
        )

        # 模型评分
        score_text = "模型评分:\n"
        score_text += f"{'标签':<20} {'准确率':<10} {'正确':<8} {'错误':<8}\n"
        score_text += "-" * 50 + "\n"
        for s in stats["model_scores"]:
            score_text += f"{s['label']:<20} {s['accuracy']:<10.1%} "
            score_text += f"+{s['total_correct']:<7} -{s['total_incorrect']:<7}\n"
        score_text += f"\n自适应阈值:\n"
        for label, th in rl_stats["thresholds"].items():
            score_text += f"  {label}: {th:.3f}\n"

        self._score_text.config(state=tk.NORMAL)
        self._score_text.delete(1.0, tk.END)
        self._score_text.insert(1.0, score_text)
        self._score_text.config(state=tk.DISABLED)

        # 强化学习曲线
        curve_text = "强化学习状态:\n"
        curve_text += f"  学习轮次: {rl_stats['total_episodes']}\n"
        curve_text += f"  学习率: {rl_stats['learning_rate']}\n"
        curve_text += f"  近期准确率: {rl_stats['recent_accuracy']:.1%}\n\n"

        curve_text += "反馈统计:\n"
        curve_text += f"{'标签':<20} {'正确':<8} {'错误':<8} {'准确率':<10}\n"
        curve_text += "-" * 50 + "\n"
        for label, data in fb_stats.items():
            total = data["correct"] + data["incorrect"]
            acc = data["correct"] / max(total, 1)
            curve_text += f"{label:<20} {data['correct']:<8} {data['incorrect']:<8} {acc:<10.1%}\n"

        curve_text += "\n学习历史（最近10轮）:\n"
        recent = self._rl.get_learning_curve()[-10:]
        for r in recent:
            bar = "█" * int(r["accuracy"] * 20)
            curve_text += f"  轮次{r['episode']:>3}: {bar} {r['accuracy']:.1%} ({r['correct']}/{r['total']})\n"

        self._curve_text.config(state=tk.NORMAL)
        self._curve_text.delete(1.0, tk.END)
        self._curve_text.insert(1.0, curve_text)
        self._curve_text.config(state=tk.DISABLED)

    def show(self):
        if self._window:
            self._window.deiconify()
            self._refresh()

    def hide(self):
        if self._window:
            self._window.withdraw()