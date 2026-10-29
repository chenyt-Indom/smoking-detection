#!/usr/bin/env python3
"""下载测试视频 - 用于验证头部追踪效果"""
import os, sys, subprocess, argparse
from pathlib import Path

# Pexels 免费视频链接（CC0 许可，可直接使用）
# 人群走路、移动场景
TEST_VIDEOS = [
    {
        "name": "crowd_walking_1",
        "url": "https://www.pexels.com/video/854260/download/",
        "desc": "街道人群行走 - 多人移动场景"
    },
    {
        "name": "people_walking_street",
        "url": "https://www.pexels.com/video/855564/download/",
        "desc": "十字路口行人 - 复杂城市环境"
    },
    {
        "name": "single_walking_close",
        "url": "https://www.pexels.com/video/855778/download/",
        "desc": "单人近景行走 - 头部清晰可见"
    },
    {
        "name": "multi_person_hallway",
        "url": "https://www.pexels.com/video/855273/download/",
        "desc": "走廊多人通过 - 室内环境"
    },
]

def download_video(url, output_path, name):
    """下载视频文件"""
    import urllib.request
    import ssl
    
    # 绕过 SSL 验证（某些环境需要）
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    
    print(f"  下载: {name} ...")
    try:
        req = urllib.request.Request(url, headers={
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        with urllib.request.urlopen(req, context=ctx, timeout=120) as response:
            data = response.read()
            with open(output_path, 'wb') as f:
                f.write(data)
            size_mb = len(data) / (1024*1024)
            print(f"  ✅ 完成: {size_mb:.1f} MB")
            return True
    except Exception as e:
        print(f"  ⚠️ 下载失败: {e}")
        return False

def generate_synthetic_video(output_dir: Path):
    """如果网络下载失败，生成合成测试视频"""
    import numpy as np
    import cv2
    
    output_path = output_dir / "synthetic_test.mp4"
    print(f"  生成合成测试视频: {output_path}")
    
    # 创建一个模拟行人移动的合成视频
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, 30, (640, 480))
    
    for i in range(300):  # 10秒@30fps
        frame = np.ones((480, 640, 3), dtype=np.uint8) * 200
        
        # 模拟多个移动的人
        for j in range(3):
            x = int(100 + j * 150 + 5 * np.sin((i + j * 50) * 0.05))
            y = int(200 + 30 * np.cos((i + j * 40) * 0.04))
            # 画人体
            cv2.ellipse(frame, (x, y + 40), (20, 40), 0, 0, 360, (100, 100, 100), -1)
            # 画头部
            cv2.circle(frame, (x, y - 20), 15, (80, 80, 80), -1)
        
        out.write(frame)
    
    out.release()
    print(f"  ✅ 合成视频已生成")
    return str(output_path)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="D:/视觉安防系统/training_data/test_videos")
    args = parser.parse_args()
    
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("📹 下载测试视频...")
    success_count = 0
    
    for video in TEST_VIDEOS:
        out_path = output_dir / f"{video['name']}.mp4"
        if out_path.exists():
            print(f"  ⏭️ 已存在: {video['name']}")
            success_count += 1
            continue
        
        if download_video(video['url'], out_path, video['desc']):
            success_count += 1
    
    if success_count == 0:
        print("\n⚠️ 网络下载失败，生成合成测试视频...")
        synthetic = generate_synthetic_video(output_dir)
        
        # 也可以使用摄像头录制作为备选
        print("💡 也可以使用摄像头录制一段测试视频")
    else:
        print(f"\n✅ 成功下载 {success_count}/{len(TEST_VIDEOS)} 个测试视频")
    
    # 列出所有测试视频
    videos = list(output_dir.glob("*.mp4")) + list(output_dir.glob("*.avi")) + list(output_dir.glob("*.mov"))
    print(f"\n📁 测试视频目录 ({len(videos)} 个文件):")
    for v in videos:
        size = v.stat().st_size / (1024**2)
        print(f"  {v.name}: {size:.1f} MB")

if __name__ == "__main__":
    main()
