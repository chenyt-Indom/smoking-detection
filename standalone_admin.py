"""standalone_admin.py - 独立管理员审核面板"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from database import FeedbackDB
from reinforcement import ReinforcementLearner
from admin_panel import AdminPanel
from logger import Logger
import tkinter as tk

db = FeedbackDB("alerts/feedback.db")
rl = ReinforcementLearner(db)
logger = Logger({"level": "INFO", "file": "logs/app.log", "max_size_mb": 20})
root = tk.Tk()
root.withdraw()
panel = AdminPanel(db, rl, logger)
root.mainloop()