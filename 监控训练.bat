@echo off
chcp 65001 >nul
title Fire-Smoke v4 Monitor
cd /d "%~dp0"
"C://Users//19853//.workbuddy//binaries//python//envs//training//Scripts//python.exe" scripts\watch_v4.py
pause
