@echo off
chcp 936 >nul
cd /d "D:/视觉安防系统"

set RUN=runs\fire_smoke_v3_26n
for /f "tokens=1-4 delims=/ " %%a in ("%date%") do set D=%%a%%b%%c
set T=%time:~0,2%%time:~3,2%
set T=%T: =0%
set DEST=backups\v3_%D%_%T%

if not exist "%RUN%\weights\best.pt" (
  echo [%date% %time%] 没找到 best.pt，跳过 >> logs\autosave.log
  goto :eof
)

mkdir "%DEST%" 2>nul
copy /y "%RUN%\weights\best.pt" "%DEST%\" >nul
copy /y "%RUN%\weights\last.pt" "%DEST%\" >nul
copy /y "%RUN%\results.csv" "%DEST%\" >nul
copy /y "%RUN%\*.png" "%DEST%\" >nul
echo [%date% %time%] 已备份到 %DEST% >> logs\autosave.log
