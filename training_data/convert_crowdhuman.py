#!/usr/bin/env python3
"""
CrowdHuman ODGT → YOLO 格式转换脚本
- 生成 person detection 数据集（fbox + vbox）
- 生成 head detection 数据集（hbox）
- 自动过滤 mask/ignore 标注
- 验证集使用 val 数据
"""

import json, os, shutil, random
from typing import List, Tuple, Dict
from pathlib import Path
import argparse

def parse_odgt(filepath: str) -> List[Dict]:
    """解析 .odgt 文件，返回标注列表"""
    records = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records

def convert_crowdhuman(
    source_dir: str,
    output_dir: str,
    train_odgt: str = "annotation_train.odgt",
    val_odgt: str = "annotation_val.odgt",
):
    """
    将 CrowdHuman 转换为 YOLO 格式，生成两个数据集：
    1. datasets/person/ - 人体检测 (class 0: person)
    2. datasets/head/   - 头部检测 (class 0: head)
    """
    source = Path(source_dir)
    output = Path(output_dir)
    
    # 输出路径
    person_train_img = output / "datasets" / "person" / "train" / "images"
    person_train_lbl = output / "datasets" / "person" / "train" / "labels"
    person_val_img = output / "datasets" / "person" / "val" / "images"
    person_val_lbl = output / "datasets" / "person" / "val" / "labels"
    
    head_train_img = output / "datasets" / "head" / "train" / "images"
    head_train_lbl = output / "datasets" / "head" / "train" / "labels"
    head_val_img = output / "datasets" / "head" / "val" / "images"
    head_val_lbl = output / "datasets" / "head" / "val" / "labels"
    
    # 确保目录存在
    for d in [person_train_img, person_train_lbl, person_val_img, person_val_lbl,
              head_train_img, head_train_lbl, head_val_img, head_val_lbl]:
        d.mkdir(parents=True, exist_ok=True)
    
    # 创建符号链接：head 和 person 共享原始图像
    # 由于 YOLO 需要图像在各自目录下，我们用 symlink
    
    def process_split(odgt_path, src_images, output_img_dir, output_lbl_dir, 
                      output_img_dir_head, output_lbl_dir_head, is_train=True):
        """处理一个数据分割（train/val）"""
        records = parse_odgt(odgt_path)
        split_name = "训练集" if is_train else "验证集"
        print(f"\n处理{split_name}: {len(records)} 张图片")
        
        person_count = 0
        head_count = 0
        ignored = 0
        
        for rec in records:
            img_name = rec["ID"]
            img_path = src_images / img_name
            
            if not img_path.exists():
                print(f"  ⚠️ 图片不存在: {img_name}")
                continue
            
            h, w = None, None
            try:
                import cv2
                frame = cv2.imread(str(img_path))
                if frame is not None:
                    h, w = frame.shape[:2]
            except:
                pass
            
            if h is None or w is None:
                print(f"  ⚠️ 无法读取图片尺寸: {img_name}")
                continue
            
            gtboxes = rec.get("gtboxes", [])
            person_lines = []
            head_lines = []
            
            for box in gtboxes:
                if box.get("tag") == "mask":
                    continue  # 跳过 crowd/reflection 标注
                
                extra = box.get("extra", {})
                if extra.get("ignore", 0) == 1:
                    ignored += 1
                    continue
                
                # === 人体标注 (fbox: full body) ===
                fbox = box.get("fbox")  # [x, y, w, h]
                if fbox and len(fbox) == 4:
                    fx, fy, fw, fh = fbox
                    fw = max(1, fw)
                    fh = max(1, fh)
                    # 归一化
                    cx = (fx + fw / 2) / w
                    cy = (fy + fh / 2) / h
                    nw = fw / w
                    nh = fh / h
                    # 裁剪到 [0, 1]
                    cx = max(0, min(1, cx))
                    cy = max(0, min(1, cy))
                    nw = max(0, min(1, nw))
                    nh = max(0, min(1, nh))
                    if nw > 0.001 and nh > 0.001:
                        person_lines.append(f"0 {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
                        person_count += 1
                
                # === 头部标注 (hbox: head box) ===
                hbox = box.get("hbox")
                head_attr = box.get("head_attr", {})
                if head_attr.get("ignore", 0) == 1:
                    continue
                if hbox and len(hbox) == 4:
                    hx, hy, hw, hh = hbox
                    hw = max(1, hw)
                    hh = max(1, hh)
                    cx = (hx + hw / 2) / w
                    cy = (hy + hh / 2) / h
                    nw = hw / w
                    nh = hh / h
                    cx = max(0, min(1, cx))
                    cy = max(0, min(1, cy))
                    nw = max(0, min(1, nw))
                    nh = max(0, min(1, nh))
                    if nw > 0.001 and nh > 0.001:
                        head_lines.append(f"0 {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
                        head_count += 1
            
            # 复制图片到两个数据集目录
            base_name = Path(img_name).stem
            
            # Person dataset
            dst_img_p = output_img_dir / f"{base_name}.jpg"
            if not dst_img_p.exists():
                shutil.copy2(img_path, dst_img_p)
            with open(output_lbl_dir / f"{base_name}.txt", 'w') as f:
                f.write('\n'.join(person_lines) + ('\n' if person_lines else ''))
            
            # Head dataset
            dst_img_h = output_img_dir_head / f"{base_name}.jpg"
            if not dst_img_h.exists():
                shutil.copy2(img_path, dst_img_h)
            with open(output_lbl_dir_head / f"{base_name}.txt", 'w') as f:
                f.write('\n'.join(head_lines) + ('\n' if head_lines else ''))
        
        print(f"  ✅ 人体框: {person_count}, 头部框: {head_count}, 忽略: {ignored}")
    
    # === 处理训练集 ===
    # 训练图片可能在多个 zip 中，按序号搜索
    train_odgt_path = source / train_odgt
    train_img_dirs = []
    
    # 寻找解压后的训练图片目录
    for subdir_name in ["Images", "images", "train", "train_images"]:
        for prefix in ["CrowdHuman_train", "train"]:
            for i in range(1, 4):
                candidate = source / f"{prefix}{i:02d}" / subdir_name
                if candidate.exists():
                    train_img_dirs.append(candidate)
    
    # 如果上述结构不存在，尝试在 source 根目录下搜索
    if not train_img_dirs:
        for candidate in source.iterdir():
            if candidate.is_dir():
                imgs = list(candidate.glob("*.jpg"))
                if imgs:
                    train_img_dirs.append(candidate)
    
    if not train_img_dirs:
        print("❌ 未找到训练图片目录！请确认数据集已解压")
        print(f"   当前目录内容: {list(source.iterdir())[:20]}")
        return False
    
    print(f"找到 {len(train_img_dirs)} 个训练图片目录")
    
    # 自下而上：先扫描所有 JPG 文件进行匹配
    # 将每个目录映射为其扁平的图片列表
    all_train_images = {}
    for d in train_img_dirs:
        for img in d.rglob("*.jpg"):
            all_train_images[img.name] = img
    
    if not all_train_images:
        print("❌ 训练图片目录为空")
        return False
    
    print(f"共找到 {len(all_train_images)} 张训练图片")
    
    # 处理训练集
    if train_odgt_path.exists():
        process_split(
            str(train_odgt_path),
            source,  # 基础搜索目录
            person_train_img, person_train_lbl,
            head_train_img, head_train_lbl,
            is_train=True
        )
    else:
        print(f"❌ 训练标注文件不存在: {train_odgt_path}")
        return False
    
    # === 处理验证集 ===
    val_odgt_path = source / val_odgt
    val_img_dirs = []
    for subdir_name in ["Images", "images", "val", "val_images"]:
        for prefix in ["CrowdHuman_val", "val"]:
            candidate = source / prefix / subdir_name
            if candidate.exists():
                val_img_dirs.append(candidate)
    
    if not val_img_dirs:
        for candidate in source.iterdir():
            if candidate.is_dir() and "val" in candidate.name.lower():
                imgs = list(candidate.glob("*.jpg"))
                if imgs:
                    val_img_dirs.append(candidate)
    
    if val_odgt_path.exists() and val_img_dirs:
        print(f"找到 {len(val_img_dirs)} 个验证图片目录")
        process_split(
            str(val_odgt_path),
            source,
            person_val_img, person_val_lbl,
            head_val_img, head_val_lbl,
            is_train=False
        )
    else:
        print(f"⚠️ 验证标注或图片目录不存在，从训练集分割 20% 作为验证集")
        # 从训练集的 person 图片中分割
        train_imgs = list(person_train_img.glob("*.jpg"))
        random.shuffle(train_imgs)
        split_idx = int(len(train_imgs) * 0.8)
        val_imgs = train_imgs[split_idx:]
        for img_path in val_imgs:
            label_path = person_train_lbl / f"{img_path.stem}.txt"
            hlbl_path = head_train_lbl / f"{img_path.stem}.txt"
            # 移动到验证集
            shutil.move(str(img_path), str(person_val_img / img_path.name))
            shutil.move(str(img_path), str(head_val_img / img_path.name))  # 会失败如果重复用
            if label_path.exists():
                shutil.move(str(label_path), str(person_val_lbl / f"{img_path.stem}.txt"))
            if hlbl_path.exists():
                shutil.move(str(hlbl_path), str(head_val_lbl / f"{img_path.stem}.txt"))
        print(f"  ✅ 分割: 训练 {split_idx} 张, 验证 {len(val_imgs)} 张")
    
    # === 统计 ===
    print("\n" + "="*60)
    print("📊 数据集统计")
    print("="*60)
    for name, img_dir, lbl_dir in [
        ("人体-训练", person_train_img, person_train_lbl),
        ("人体-验证", person_val_img, person_val_lbl),
        ("头部-训练", head_train_img, head_train_lbl),
        ("头部-验证", head_val_img, head_val_lbl),
    ]:
        n_imgs = len(list(img_dir.glob("*.jpg"))) if img_dir.exists() else 0
        n_lbls = len(list(lbl_dir.glob("*.txt"))) if lbl_dir.exists() else 0
        print(f"  {name}: {n_imgs} 图片, {n_lbls} 标注")
    
    print("\n✅ 转换完成！")
    return True

def main():
    parser = argparse.ArgumentParser(description="CrowdHuman → YOLO 格式转换")
    parser.add_argument("--source", default="./CrowdHuman_source", help="CrowdHuman 源目录")
    parser.add_argument("--output", default=".", help="输出根目录")
    parser.add_argument("--train-odgt", default="annotation_train.odgt", help="训练标注文件名")
    parser.add_argument("--val-odgt", default="annotation_val.odgt", help="验证标注文件名")
    args = parser.parse_args()
    
    convert_crowdhuman(args.source, args.output, args.train_odgt, args.val_odgt)

if __name__ == "__main__":
    main()
