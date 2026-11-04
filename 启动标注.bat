@echo off
chcp 936 >nul
title 香烟标注工具 · 端口 5077
cd /d "%~dp0"

set "PY=C:/Users/19853/.workbuddy/binaries/python/envs/training/Scripts/python.exe"
if not exist "%PY%" (
  echo.
  echo [x] 找不到 Python 训练环境：
  echo     %PY%
  echo     用记事本改本文件里的 PY= 那一行即可。
  echo.
  pause
  exit /b 1
)

rem 端口已被占用就别再起一个（否则标注服务会互相抢端口）
netstat -ano | findstr ":5077" | findstr "LISTENING" >nul 2>&1
if not errorlevel 1 (
  echo 标注服务已经在运行，直接打开页面。
  start "" "http://127.0.0.1:5077/"
  exit /b 0
)

echo ============================================================
echo  正在启动香烟标注服务，浏览器会自动打开
echo  关闭本窗口 = 停止服务（标注内容已落盘，不会丢）
echo ============================================================
echo.
"%PY%" scripts\annotate_cig.py --port 5077 --open

echo.
echo 服务已停止。标注内容保存在：
echo   D:/training_data/smoke/v32_smoking/labels_fixed   (轴对齐，可直接重训)
echo   D:/training_data/smoke/v32_smoking/labels_obb     (旋转框，留给 -obb 模型)
echo 原始 labels\ 一个字节都没动。
echo.
pause
