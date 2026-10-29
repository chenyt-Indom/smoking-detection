@echo off
chcp 936 >nul
title 火情烟雾训练监控
cd /d "%~dp0"

set PY=C:/Users/19853/.workbuddy/binaries/python/envs/training/Scripts/python.exe
if not exist "%PY%" (
  echo [错误] 找不到 Python 环境：%PY%
  pause & exit /b 1
)

"%PY%" -u scripts\monitor_train.py %*

echo.
echo 监控已退出（训练在另一个窗口里继续，不受影响）
pause
