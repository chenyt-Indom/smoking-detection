"""统一资源路径定位：兼容源码运行和 PyInstaller 打包。

源码运行时项目根是 D:\\视觉安防系统；打包成 exe 后，模型/web/config
这些资源随 exe 放在同一目录，所以根目录 = exe 所在目录。
"""
from __future__ import annotations

import sys
from pathlib import Path


def resource_root() -> Path:
    """返回项目/资源根目录（源码 = 项目根，打包 = exe 所在目录）。"""
    if getattr(sys, "frozen", False):          # PyInstaller 打包后 frozen=True
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent.parent
