@echo off
chcp 936 >nul
cd /d "D:/视觉安防系统"
call scripts\auto_save.bat
echo [%date% %time%] 正在停止训练进程... >> logs\autosave.log
taskkill /F /IM python.exe >nul 2>&1
timeout /t 5 /nobreak >nul
call scripts\auto_save.bat
echo [%date% %time%] 训练已停止并完成最终备份 >> logs\autosave.log
