@echo off
chcp 65001 >nul
echo ==========================================
echo   V32 头部追踪 - 摄像头实时检测
echo   COCO + 卡尔曼滤波
echo   按 Q 键退出
echo ==========================================
echo.
cd /d "D:\视觉安防系统"
"C:\Users\19853\.workbuddy\binaries\python\envs\training\Scripts\python.exe" -u "D:\视觉安防系统\head_tracker_cam.py"
pause