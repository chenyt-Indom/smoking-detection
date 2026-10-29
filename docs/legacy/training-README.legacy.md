# 训练资产 (V26)

## 文件说明
- `cap25_final.json` — 本次757帧人工标注 (8阶段: s1横置/s2圆头/s3嘴前/s4耳后/s5吸管/s6手机/s7背景/s8自由)
  - 键: `阶段/文件名.jpg`; 值: `{type: pos|neg, boxes: [[cx,cy,w,h]归一化]}`
- `train_v26.py` — V26训练脚本 (V24起点, freeze=4, lr0=0.0003, 300轮)
  - 数据: 仅757张原图(无副本) | 在线增强: hsv/erasing/scale/degrees
  - 负样本帧(无框)自动作为背景学习 → 教模型吸管/手机/背景不是烟
- `make_annotator.py` — 自包含HTML标注工具生成器

## 训练数据来源
- 原图: `D:\training_data\smoke\cap_v25\` (8个子目录)
- 数据集构建: 按阶段90/10划分 train 685 / val 72

## 模型记忆链
V24(3.2万张) → V26(757张人工标注) → V27(计划: +4000张精选)

## 光线自适应 (head_tracker_cam.py)
- `light_adapt()`: 亮度EMA抗闪烁 + CLAHE局部对比度 + 自动gamma(暗光提亮/过曝压回)
- 增强帧仅用于模型推理, 颜色过滤规则仍用原帧
