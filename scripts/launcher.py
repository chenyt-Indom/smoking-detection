"""桌面启动器：双击 → 起服务 → 自动打开看板。

打包成 exe 后放在桌面，用户不需要记任何命令。

为什么要单独做一个启动器而不是直接打包 run_watch.py
----------------------------------------------------
run_watch 依赖 torch + ultralytics + opencv + CUDA，PyInstaller 打进去要 2GB 以上，
而且 CUDA 的 DLL 很容易漏、启动还慢。启动器只做三件事（找解释器、起子进程、开浏览器），
打出来几 MB，秒开。代价是依赖本机已有可用的 Python 环境 —— 对"自己这台机器用"是最划算的。

配置文件
--------
项目里的 config/launcher.json，不存在就用默认值：
    {"camera": 0, "port": 5050, "llm": "auto"}
camera 换成 1 就是笔记本内置摄像头；llm 填 off 则完全不走大模型。
"""
from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import time
import webbrowser
from pathlib import Path

PROJECT = Path(r"D:\视觉安防系统")

# 按优先级找解释器。第一项是本机现成的训练环境。
PY_CANDIDATES = [
    r"C:\Users\19853\.workbuddy\binaries\python\envs\training\Scripts\python.exe",
    r"C:\Users\19853\AppData\Local\Programs\Python\Python313\python.exe",
    r"C:\Users\19853\AppData\Local\Programs\Python\Python312\python.exe",
]

DEFAULTS = {"camera": 0, "port": 5050, "llm": "auto", "host": "127.0.0.1"}


def load_cfg() -> dict:
    cfg = dict(DEFAULTS)
    p = PROJECT / "config" / "launcher.json"
    if p.is_file():
        try:
            cfg.update(json.loads(p.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError):
            pass
    return cfg


def find_python() -> str | None:
    for c in PY_CANDIDATES:
        if Path(c).is_file():
            return c
    # 退一步：PATH 里的 python
    for name in ("python.exe", "python"):
        from shutil import which
        w = which(name)
        if w:
            return w
    return None


def port_open(port: int, host: str = "127.0.0.1", timeout: float = 0.6) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(timeout)
        return s.connect_ex((host, port)) == 0


def _setup_console() -> None:
    """把控制台切到 UTF-8。

    Windows 控制台默认代码页 936（GBK），打印中文或 ✓ → … 这类字符会直接抛
    UnicodeEncodeError 把启动器打挂。打包成 exe 后连 PYTHONIOENCODING 都不好使，
    只能在这里用 SetConsoleOutputCP + reconfigure 双管齐下。
    """
    if os.name == "nt":
        try:
            import ctypes
            ctypes.windll.kernel32.SetConsoleOutputCP(65001)
            ctypes.windll.kernel32.SetConsoleCP(65001)
        except Exception:                    # noqa: BLE001
            pass
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:                    # noqa: BLE001
            pass


def main() -> int:
    _setup_console()
    print("=" * 56)
    print("  火情烟雾智能监控 · 启动器")
    print("=" * 56)

    if not PROJECT.is_dir():
        print(f"[x] 找不到项目目录：{PROJECT}")
        input("按回车退出…")
        return 1

    cfg = load_cfg()
    port = int(cfg["port"])
    url = f"http://{cfg['host']}:{port}/"

    # 已经在跑就直接开浏览器，避免端口冲突
    if port_open(port):
        print(f"服务已在运行 → 直接打开 {url}")
        webbrowser.open(url)
        time.sleep(1)
        return 0

    py = find_python()
    if not py:
        print("[x] 没找到可用的 Python。请先确认本机 Python 环境。")
        input("按回车退出…")
        return 1

    script = PROJECT / "scripts" / "run_watch.py"
    if not script.is_file():
        print(f"[x] 找不到主程序：{script}")
        input("按回车退出…")
        return 1

    cmd = [py, "-u", str(script),
           "--camera", str(cfg["camera"]),
           "--llm", str(cfg["llm"]),
           "--port", str(port),
           "--host", str(cfg["host"])]
    print(f"解释器 : {py}")
    print(f"摄像头 : {cfg['camera']}   大模型: {cfg['llm']}   端口: {port}")
    print("-" * 56)

    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"
    try:
        proc = subprocess.Popen(cmd, cwd=str(PROJECT), env=env)
    except OSError as e:
        print(f"[x] 启动失败：{e}")
        input("按回车退出…")
        return 1

    # 等端口起来（模型加载 + 摄像头打开通常 10~25 秒）
    print("正在启动，等看板就绪…")
    ready = False
    for i in range(120):
        if proc.poll() is not None:
            print("[x] 子进程已退出，看下面日志找原因")
            break
        if port_open(port):
            ready = True
            break
        if i and i % 10 == 0:
            print(f"  …已等 {i // 2}s")
        time.sleep(0.5)

    if ready:
        print(f"\n✓ 看板就绪：{url}")
        webbrowser.open(url)
        print("（关闭本窗口即停止监控；也可以按 Ctrl+C）\n")
    else:
        print("\n⚠ 端口没起来，但仍保持运行，请看上面的报错\n")

    try:
        proc.wait()
    except KeyboardInterrupt:
        print("\n正在停止…")
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
    print("已退出。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
