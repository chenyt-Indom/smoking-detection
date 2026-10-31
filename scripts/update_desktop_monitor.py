"""把桌面监控入口切到通用版 monitor.py。

原来桌面的 bat 调的是 C:\\Users\\19853\\.workbuddy\\skills\\training-monitor\\universal_monitor.py，
现在统一指向 D:\\yolo-training-monitor\\monitor.py（git 仓库里的那一份），
以后只维护一处代码，GitHub 上就是本机在跑的版本。

编码：GBK + CRLF + chcp 936，中文才能在 cmd 窗口正常显示。
"""
from pathlib import Path

DESKTOP = Path(r"C:\Users\19853\Desktop")

BAT = r"""@echo off
chcp 936 >nul
title 火情烟雾 · 训练监控
set PYTHONIOENCODING=utf-8

set PY=C:\Users\19853\.workbuddy\binaries\python\envs\training\Scripts\python.exe
set MON=D:\yolo-training-monitor\monitor.py
set RUNS=D:\fire_runs

if not exist "%PY%" (
  echo [错误] 找不到 Python:
  echo        %PY%
  pause
  exit /b 1
)

if not exist "%MON%" (
  echo [错误] 找不到监控脚本:
  echo        %MON%
  echo.
  echo 如果这个目录被移走了，改本文件里的 MON 变量指向新的 monitor.py。
  pause
  exit /b 1
)

REM --runs-dir 指定训练输出根目录；去掉它则自动探测当前目录下的 runs\detect
"%PY%" -u "%MON%" --runs-dir "%RUNS%"

echo.
echo 监控已退出。训练进程不受影响。
pause
"""

path = DESKTOP / "火情烟雾训练监控.bat"
path.write_bytes(BAT.replace("\r\n", "\n").replace("\n", "\r\n").encode("gbk"))
print(f"已写入 {path}  ({path.stat().st_size} 字节, gbk+crlf)")
