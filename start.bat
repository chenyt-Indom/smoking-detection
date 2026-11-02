@echo off
chcp 65001 >nul
cd /d D:/视觉安防系统
echo 清代理后启动（沙箱代理会让 LLM 请求超时）
set http_proxy=
set https_proxy=
set HTTP_PROXY=
set HTTPS_PROXY=
set all_proxy=
set ALL_PROXY=
set NO_PROXY=*
set no_proxy=*
set PYTHONIOENCODING=utf-8
"C:/Users/19853/.workbuddy/binaries/python/envs/training/Scripts/python.exe" -u scripts\run_multi.py --llm auto
pause
