#!/usr/bin/env python3
"""通过 HuggingFace 镜像下载 CrowdHuman 数据集"""
import os, sys
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from huggingface_hub import hf_hub_download
from pathlib import Path

# 需要下载的文件
FILES = [
    "CrowdHuman_train01.zip",
    "CrowdHuman_train02.zip", 
    "CrowdHuman_train03.zip",
    "CrowdHuman_val.zip",
    "annotation_train.odgt",
    "annotation_val.odgt",
]

REPO_ID = "Dhruvvv26/CrowdHuman"
SAVE_DIR = Path(r"D:/视觉安防系统/training_data/CrowdHuman_source")
SAVE_DIR.mkdir(parents=True, exist_ok=True)

for filename in FILES:
    print(f"下载: {filename} ...")
    try:
        local_path = hf_hub_download(
            repo_id=REPO_ID,
            filename=filename,
            repo_type="dataset",
            local_dir=str(SAVE_DIR),
            resume_download=True,
        )
        size_mb = os.path.getsize(local_path) / (1024**2)
        print(f"  ✅ 完成: {filename} ({size_mb:.1f} MB)")
    except Exception as e:
        print(f"  ❌ 失败: {filename} - {e}")

print("\n下载完成!")
# 列出下载的文件
for f in sorted(SAVE_DIR.glob("*")):
    size = f.stat().st_size / (1024**3) if f.stat().st_size > 1024**3 else f.stat().st_size / (1024**2)
    unit = "GB" if f.stat().st_size > 1024**3 else "MB"
    print(f"  {f.name}: {size:.1f} {unit}")
