#!/usr/bin/env python3
"""
CrowdHuman 数据集解压脚本
将 HuggingFace 下载的 zip 文件解压到正确位置
"""

import zipfile, os, shutil, argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

def extract_zip(zip_path: Path, output_dir: Path) -> bool:
    """解压单个 zip 文件"""
    print(f"  📦 解压: {zip_path.name} ...")
    try:
        with zipfile.ZipFile(zip_path, 'r') as zf:
            zf.extractall(output_dir)
        print(f"  ✅ 完成: {zip_path.name}")
        return True
    except Exception as e:
        print(f"  ❌ 失败: {zip_path.name} - {e}")
        return False

def extract_crowdhuman(source_dir: str, output_dir: str):
    """解压所有 CrowdHuman zip 文件并组织图片"""
    source = Path(source_dir)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    
    # 寻找所有 zip 文件
    zip_files = list(source.glob("CrowdHuman_*.zip"))
    print(f"找到 {len(zip_files)} 个 zip 文件:")
    for zf in zip_files:
        print(f"  - {zf.name} ({zf.stat().st_size / (1024**3):.2f} GB)")
    
    if not zip_files:
        print("❌ 未找到 CrowdHuman zip 文件")
        # 检查是否有已解压的图片
        jpgs = list(source.glob("**/*.jpg"))
        print(f"   但找到 {len(jpgs)} 张 jpg 图片（可能已解压）")
        return False
    
    # 并行解压
    print("\n开始解压...")
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = {executor.submit(extract_zip, zf, output): zf for zf in zip_files}
        for future in as_completed(futures):
            zf = futures[future]
            try:
                future.result()
            except Exception as e:
                print(f"  ❌ {zf.name}: {e}")
    
    # 组织图片：将所有图片平铺到 Images/ 目录
    images_dir = output / "Images"
    images_dir.mkdir(exist_ok=True)
    
    jpg_count = 0
    for jpg in output.rglob("*.jpg"):
        dest = images_dir / jpg.name
        if not dest.exists():
            try:
                shutil.move(str(jpg), str(dest))
                jpg_count += 1
            except:
                shutil.copy2(str(jpg), str(dest))
                jpg_count += 1
    
    print(f"\n✅ 解压完成: {jpg_count} 张图片在 {images_dir}")
    
    # 复制标注文件
    for odgt_file in ["annotation_train.odgt", "annotation_val.odgt"]:
        src = source / odgt_file
        if src.exists():
            shutil.copy2(str(src), str(output / odgt_file))
            print(f"  ✅ 复制: {odgt_file}")
    
    return True

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="./CrowdHuman_source", help="HuggingFace 下载目录")
    parser.add_argument("--output", default="./CrowdHuman", help="解压输出目录")
    args = parser.parse_args()
    
    extract_crowdhuman(args.source, args.output)

if __name__ == "__main__":
    main()
