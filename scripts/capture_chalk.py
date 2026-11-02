"""拍摄粉笔训练素材 —— 按引导依次拍 8 个姿势。

为什么单独做一个脚本，而不是用看板的"存样本"按钮
--------------------------------------------------
看板存的是"当前报警帧"，姿势和场景都是随机碰到的；而训练需要的是
**按姿态维度覆盖**：正面、侧面、举到嘴边、垂在身侧……每种都要有。
所以这里做成"引导式"：屏幕上方告诉你现在该摆什么姿势，你摆好后按空格，
一张一张攒，攒够自动切下一个姿势。

用法
----
    python scripts/capture_chalk.py                 # 用内置摄像头（1280x720）
    python scripts/capture_chalk.py --camera 0      # 换成外接那颗
    python scripts/capture_chalk.py --per 25        # 每个姿势拍 25 张

操作
----
    空格  存一张
    n     跳过当前姿势，切下一个
    q     退出

输出
----
    dataset_chalk/images/<姿势>_<序号>.jpg
    dataset_chalk/labels/<姿势>_<序号>.txt      （空的，等标注）
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "dataset_chalk"

# 8 个姿势，覆盖"持握方式 × 角度 × 距离"这几个会影响检测的维度。
# 顺序从最典型到最边缘 —— 时间不够时至少把前面的拍完。
POSES = [
    ("正面举胸前", "粉笔横在胸前，手指自然夹住中段，镜头正对"),
    ("侧面45度", "身体侧过去一点，粉笔和镜头成 45 度角"),
    ("举到嘴边", "模拟抽烟姿势，粉笔一端凑到嘴唇前"),
    ("垂在身侧", "手自然垂下，粉笔朝下"),
    ("只露一半", "用另一只手挡住粉笔一半，只露出端头"),
    ("靠近镜头", "把粉笔举到离镜头 30cm 左右，占画面较大"),
    ("远离镜头", "退到 2 米外，粉笔只有一点点大"),
    ("平放桌上", "粉笔横放在桌面上，镜头俯拍"),
]


def parse() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", type=int, default=2, help="摄像头序号（实测 2 = 内置 1280x720）")
    ap.add_argument("--per", type=int, default=25, help="每个姿势拍多少张")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    return ap.parse_args()


def main() -> int:
    a = parse()
    (OUT / "images").mkdir(parents=True, exist_ok=True)
    (OUT / "labels").mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(a.camera, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, a.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, a.height)
    if not cap.isOpened():
        print(f"[x] 打不开摄像头 {a.camera}")
        print("    常见原因：安防服务正在运行，占着摄像头。")
        print("    先关掉运行中的 run_multi（黑窗口或 start.bat），再跑这个脚本。")
        print("    或者换序号试试：--camera 1  --camera 2")
        return 1

    win = "capture chalk  |  SPACE=save   N=next pose   Q=quit"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, 960, 540)

    idx = 0          # 当前姿势序号
    count = 0        # 当前姿势已拍张数
    total = 0
    flash_until = 0.0

    print("=" * 60)
    for i, (name, tip) in enumerate(POSES, 1):
        print(f"  {i}. {name:8s}  {tip}")
    print("=" * 60)
    print("  空格存图 / n 下一个姿势 / q 退出")
    print("=" * 60)

    while idx < len(POSES):
        ok, frame = cap.read()
        if not ok:
            break
        name, tip = POSES[idx]
        show = frame.copy()
        h, w = show.shape[:2]

        # 顶部：当前姿势 + 进度
        cv2.rectangle(show, (0, 0), (w, 92), (0, 0, 0), -1)
        cv2.putText(show, f"[{idx+1}/{len(POSES)}] {name}   {count}/{a.per}",
                    (16, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 120), 2)
        cv2.putText(show, tip, (16, 76), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (200, 200, 200), 1)

        # 底部：累计
        cv2.putText(show, f"total saved: {total}", (16, h - 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1)

        # 存图后的白闪反馈，让人知道确实存下了
        if time.time() < flash_until:
            ov = show.copy()
            cv2.rectangle(ov, (0, 0), (w, h), (255, 255, 255), -1)
            show = cv2.addWeighted(ov, 0.35, show, 0.65, 0)

        cv2.imshow(win, show)
        k = cv2.waitKey(1) & 0xFF

        if k in (ord("q"), 27):
            break
        if k == ord("n"):
            idx += 1
            count = 0
            continue
        if k == 32:                      # 空格
            stem = f"{idx+1:02d}_{name}_{count+1:03d}"
            ok2, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
            if ok2:
                (OUT / "images" / f"{stem}.jpg").write_bytes(buf.tobytes())
                (OUT / "labels" / f"{stem}.txt").write_text("", encoding="utf-8")
                count += 1
                total += 1
                flash_until = time.time() + 0.12
                if count >= a.per:
                    # 拍够了，稍微停一下再切下一个，免得手滑多拍
                    time.sleep(0.35)
                    idx += 1
                    count = 0

    cap.release()
    cv2.destroyAllWindows()
    print(f"\n  完成：共存 {total} 张到 {OUT / 'images'}")
    print(f"  下一步：用 X-AnyLabeling 打开 {OUT / 'images'} 标注，")
    print(f"         类别名填 cigarette（粉笔当香烟的正样本用）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
