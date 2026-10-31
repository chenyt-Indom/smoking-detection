"""重写桌面三个入口文件，使其与当前 v7 训练一致。

为什么必须重写
--------------
旧的「火情烟雾启动训练.bat」还是 26n / 640 / 80 轮的老参数，
项目路径里还带着一个空格（D:/视觉安防系 统/runs），检查点目录也和变量对不上。
现在 v7 正用 26s / 960 / 20 轮在跑，误点会再起一个训练抢显存，两个一起崩。

编码约定
--------
.bat 用 GBK + CRLF —— 配合 chcp 936，中文才能在 cmd 窗口正常显示。
.txt 用 UTF-8 with BOM —— 记事本能正确识别，不会像旧版那样满屏乱码。
"""
from pathlib import Path

DESKTOP = Path(r"C:\Users\19853\Desktop")

START_BAT = r"""@echo off
chcp 936 >nul
title 火情烟雾 v7 · 启动训练

echo ==================================================
echo  火情烟雾 v7c · 启动训练
echo  数据 D:\datasets\fire_smoke_v7  (52316 张)
echo  起点 models\fire_smoke_v5_26s960_best.pt
echo  20 轮 / 960 / batch 6 / lr0 0.004 / MuSGD / warmup 0
echo  预计 9-10 小时
echo ==================================================
echo.

REM 防重复启动：已有 python 在跑时直接拒绝，避免两个训练抢显存
tasklist /NH /FI "IMAGENAME eq python.exe" 2>nul | find /I "python.exe" >nul
if not errorlevel 1 (
  echo [拒绝] 检测到 python 进程正在运行，可能训练还没结束。
  echo        重复启动会抢占显存，两个都会崩。
  echo        想看进度请双击「火情烟雾训练监控.bat」。
  echo.
  pause
  exit /b 1
)

cd /d "D:\视觉安防系统"
set PY=C:\Users\19853\.workbuddy\binaries\python\envs\training\Scripts\python.exe

"%PY%" -u scripts\train_v4.py --model models\fire_smoke_v5_26s960_best.pt --data D:\datasets\fire_smoke_v7\dataset.yaml --epochs 20 --imgsz 960 --batch 6 --workers 4 --lr0 0.004 --optimizer MuSGD --warmup-bias-lr 0 --patience 8 --close-mosaic 5 --multi-scale 0.25 --name fire_smoke_v7c_26s960 --project D:\fire_runs

echo.
echo 训练结束/中断，按任意键关闭
pause
"""

RESUME_BAT = r"""@echo off
chcp 936 >nul
title 火情烟雾 v7 · 续训

cd /d "D:\视觉安防系统"
set PY=C:\Users\19853\.workbuddy\binaries\python\envs\training\Scripts\python.exe
set LAST=D:\fire_runs\fire_smoke_v7c_26s960\weights\last.pt

if not exist "%LAST%" (
  echo [错误] 没找到断点文件：
  echo        %LAST%
  echo.
  echo 说明：第一轮还没跑完时不会写出 last.pt。
  echo       这种情况请用「火情烟雾启动训练.bat」重新开始。
  echo.
  pause
  exit /b 1
)

tasklist /NH /FI "IMAGENAME eq python.exe" 2>nul | find /I "python.exe" >nul
if not errorlevel 1 (
  echo [拒绝] 已有 python 进程在跑，先关掉再续训。
  echo.
  pause
  exit /b 1
)

echo 从断点续训：%LAST%
"%PY%" -u scripts\train_v4.py --resume-path "%LAST%" --data D:\datasets\fire_smoke_v7\dataset.yaml --name fire_smoke_v7c_26s960 --project D:\fire_runs

echo.
echo 训练结束/中断，按任意键关闭
pause
"""

README = """火情烟雾检测 · 桌面入口说明
================================

【当前训练】
  版本      fire_smoke_v7c_26s960
  数据集    D:\\datasets\\fire_smoke_v7   （52,316 张）
  起点权重  models\\fire_smoke_v5_26s960_best.pt
  输出目录  D:\\fire_runs\\fire_smoke_v7c_26s960
  参数      20 轮 / 960 / batch 6 / lr0 0.004 / MuSGD / warmup_bias_lr 0
  预计      9~10 小时

  ⚠ 不要去掉 --optimizer MuSGD 和 --warmup-bias-lr 0
    ultralytics 的 optimizer=auto 会忽略 lr0，强制用 0.01，
    并且把 warmup 起点也改掉。微调时 lr 大 2.5 倍会把已有能力冲散。

【三个入口】
  1) 火情烟雾训练监控.bat
     每 3 秒刷新：每轮 loss / P / R / mAP50 / mAP50-95、
     趋势箭头、ETA、GPU 占用与显存。
     第一轮验证跑完前，显示实时步数进度条。
     关掉监控窗口不会中断训练。

  2) 火情烟雾启动训练.bat
     从头开始训练。启动前会检查有没有 python 在跑，
     有就拒绝退出，避免两个训练抢显存。

  3) 火情烟雾续训.bat
     关机或中断后，从 last.pt 断点继续，优化器和轮次都会恢复。
     第一轮没跑完时还没有 last.pt，此时要用启动脚本重来。

【数据集 v7 的构成】
  v7 = v5 全量 41,853 张 + 外部烟雾集无火子集 10,463 张
  外部集来自 E:\\BaiduNetdiskDownload\\Y-烟雾24000.zip（24,574 张）
  该集只有 smoke 一类标注，class 0 已重映射为 class 1
  其中 2,929 张有火但无火标注的图已排除，防止污染 fire 类

【训练完成后看什么】
  best.pt                        当前最优权重
  results.csv                    每轮指标（监控读的就是它）
  results.png / confusion_matrix.png   训练结束自动生成
"""


def write(path: Path, text: str, encoding: str) -> None:
    # 统一 CRLF，Windows 下 cmd / 记事本才不会把行粘在一起
    path.write_bytes(text.replace("\r\n", "\n").replace("\n", "\r\n").encode(encoding))
    print(f"已写入 {path.name}  ({path.stat().st_size} 字节, {encoding})")


for name, text in [
    ("火情烟雾启动训练.bat", START_BAT),
    ("火情烟雾续训.bat", RESUME_BAT),
]:
    write(DESKTOP / name, text, "gbk")

# UTF-8 带 BOM：记事本靠 BOM 判断编码，没有 BOM 的老版本会当 ANSI 读成乱码
(DESKTOP / "火情烟雾·说明.txt").write_bytes(
    b"\xef\xbb\xbf" + README.replace("\n", "\r\n").encode("utf-8"))
print(f"已写入 火情烟雾·说明.txt  ({(DESKTOP / '火情烟雾·说明.txt').stat().st_size} 字节, utf-8-sig)")
