@echo off
chcp 936 >nul
title 火情烟雾 · 续训
cd /d "D:/视觉安防系统"
set PY=C:/Users/19853/.workbuddy/binaries/python/envs/training/Scripts/python.exe
if not exist "%PY%" ( echo [错误] 找不到 Python & pause & exit /b 1 )

echo 正在找最新的训练任务…
"%PY%" -c "import sys;sys.path.insert(0,'.');from scripts.monitor_train import find_latest_run;r=find_latest_run();print(r.name if r else '');open('.last_run','w').write(r.name if r else '')" > .last_run_tmp 2>nul
set /p RUNNAME=<.last_run_tmp
if "%RUNNAME%"=="" ( echo [错误] 没找到任何 runs\*\results.csv & pause & exit /b 1 )
echo 续训：%RUNNAME%
echo.

if not exist "runs\%RUNNAME%\weights\last.pt" (
  echo [错误] runs\%RUNNAME%\weights\last.pt 不存在，无法续训
  pause & exit /b 1
)

"%PY%" -u scripts\train_fire.py --model "runs\%RUNNAME%\weights\last.pt" --name %RUNNAME% --epochs 80 --batch 32 --workers 2 --imgsz 640 --resume
echo.
echo 结束，按任意键关闭
pause
