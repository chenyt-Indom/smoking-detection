@echo off
chcp 936 >nul
title 火情烟雾 训练
cd /d "%~dp0"

set PY=C:/Users/19853/.workbuddy/binaries/python/envs/training/Scripts/python.exe
set RUNS=D:/fire_smoke_train/runs
set NAME=fire_smoke_26n

if not exist "%RUNS%" mkdir "%RUNS%"

echo ============================================================
echo   火情 / 烟雾检测训练（YOLO26）
echo   输出：%RUNS%\%NAME%
echo   另开一个窗口双击「训练监控.bat」可实时看进度
echo ============================================================
echo.

if exist "%RUNS%\%NAME%\weights\last.pt" (
  echo [续训] 发现 last.pt，从断点继续...
  "%PY%" -u scripts\train_fire.py --model "%RUNS%\%NAME%\weights\last.pt" --project "%RUNS%" --name %NAME% --epochs 80 --batch 32 --imgsz 640 --resume %*
) else (
  echo [新训] 没有断点，从 yolo26n.pt 开始...
  "%PY%" -u scripts\train_fire.py --model yolo26n.pt --project "%RUNS%" --name %NAME% --epochs 80 --batch 32 --imgsz 640 %*
)

echo.
echo 训练结束或中断，按任意键关闭窗口
pause
