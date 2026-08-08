"""head_tracker 安全启动器: 确保单实例, 防止假死副本抢资源"""
import os, sys, time, subprocess

LOCK = os.path.join(os.path.dirname(__file__), '.tracker.lock')

def acquire():
    try:
        fd = os.open(LOCK, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(fd, str(os.getpid()).encode())
        os.close(fd)
        return True
    except FileExistsError:
        return False

def release():
    try:
        os.remove(LOCK)
    except OSError:
        pass

if __name__ == '__main__':
    if not acquire():
        print("[ERROR] 已有追踪器在运行(检测到锁文件), 请先关闭旧窗口!")
        time.sleep(3)
        sys.exit(1)
    try:
        py = r"C:\Users\19853\.workbuddy\binaries\python\envs\training\Scripts\python.exe"
        script = os.path.join(os.path.dirname(__file__), "head_tracker_cam.py")
        p = subprocess.Popen([py, "-u", script], cwd=os.path.dirname(__file__))
        p.wait()
    finally:
        release()
