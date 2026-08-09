@echo off
title Training Monitor (Universal)
cd /d "D:\training_data"

set PY=C:\Users\19853\.workbuddy\binaries\python\envs\training\Scripts\python.exe
if not exist "%PY%" (
    echo [ERROR] Python env not found: %PY%
    pause
    exit /b 1
)

if not exist "C:\Users\19853\.workbuddy\skills\training-monitor\universal_monitor.py" (
    echo [ERROR] Monitor script not found
    pause
    exit /b 1
)

echo Starting universal training monitor...
"%PY%" -u "C:\Users\19853\.workbuddy\skills\training-monitor\universal_monitor.py"

echo.
echo Monitor exited.
pause
