@echo off
title 视觉安防系统
set PYTHON=C:\Users\19853\AppData\Local\Programs\Python\Python312\python.exe
cd /d "D:\视觉安防系统"

echo ============================================
echo   视觉安防系统
echo ============================================
echo.
echo [1] 启动带界面的监控模式
echo [2] 启动后台无界面模式
echo [3] 打开管理员审核面板
echo [4] 退出
echo.
set /p choice="请选择 (1-4): "

if "%choice%"=="1" goto gui
if "%choice%"=="2" goto headless
if "%choice%"=="3" goto admin
if "%choice%"=="4" goto end
goto end

:gui
echo 启动带界面监控模式...
%PYTHON% main.py
goto end

:headless
echo 启动后台无界面模式...
echo 系统将在后台运行，截图保存到 alerts 目录
echo 按 Ctrl+C 停止
%PYTHON% headless_mode.py
goto end

:admin
echo 启动管理员审核面板...
%PYTHON% standalone_admin.py
goto end

:end
echo 再见!
pause
