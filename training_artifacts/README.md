# 训练产出物（training_artifacts）

本目录存放**训练参数、逐轮指标、评估曲线与最优权重**，供复现与交付验收对照。
**训练集 / 验证集 / 测试集图片本身不入库**（3.7 万张、约 14 GB，且为二进制素材）。

最后更新：2026-10-30

---

## 一、这一版是什么

| 项 | 值 |
|---|---|
| 实验名 | `fire_smoke_v3_26n` |
| 模型 | YOLO26n（2.4M 参数） |
| 权重 | `fire_smoke_v3_26n/weights/fire_smoke_v3_26n_best.pt`（5.4 MB） |
| 输入 | 640 × 640，batch 32，80 轮，patience 25 |
| 数据集 | `E:/datasets/fire_smoke_v3`（FASDD 两批去重合并 37,818 张唯一图） |
| 划分 | train 28,381 / valid 6,496 / test 2,920（按编号段分层，防相邻帧泄漏） |
| 类别 | `fire`、`smoke`（2 类） |
| 硬件/耗时 | RTX 5070 Ti Laptop 12GB，约 5.5 小时 |

**训练集不在本仓库**，只保留它的规模与切分方式（上表），以便核对指标口径。
重新训练请把数据放回 `E:/datasets/fire_smoke_v3/{train,valid,test}/{images,labels}`，
再执行 `python scripts/train_fire.py --model yolo26n.pt --data E:/datasets/fire_smoke_v3/dataset.yaml --name fire_smoke_v3_26n`。

---

## 二、指标（口径见下）

### test 集（2,920 张，未参与训练/调参）

| 指标 | 值 |
|---|---|
| mAP50 | **0.7646** |
| mAP50-95 | **0.5112** |
| Precision | 0.7619 |
| Recall | 0.6726 |

### 分类别（mAP50-95）

| 类别 | mAP50-95 | 说明 |
|---|---|---|
| `smoke` | **0.6062** | 稳定：置信度普遍 0.79~0.95 |
| `fire` | **0.4161** | 明显偏弱：置信度仅 0.31~0.52，大片火场只框出零星火点 |

> `fire` 是当前短板。推测原因：火焰标注偏「火点」而非「整片火域」，且火焰形态多变（远景小火/整片山火）。
> 下一步优先补火域完整框，其次加小目标火样本。

### 误报（1,051 张纯背景图，无火无烟）

| conf 门限 | 误报张数 | 误报率 |
|---|---|---|
| 0.25 | 48 | 4.6% |
| 0.35 | 22 | **2.1%** |
| 0.50 | 10 | **1.0%** |

对比上一版 v2（4,000 张数据）：**背景被判为 fire 的比例 74% → 2%**，误报问题基本解决。

### 速度

10.4 ms/张（约 96 FPS，含预处理与后处理，RTX 5070 Ti Laptop，FP32 PyTorch）。

### 非火场景误报抽测（2026-10-30 复核）

| 视频 | 抽帧 | 误报帧 | 最高 conf |
|---|---|---|---|
| `test_crowd.mp4`（人群） | 10 | 0 | 0.00 |
| `test_heads_moving.mp4`（人头走动） | 10 | 0 | 0.00 |

复测脚本：`scripts/test_v3_effect.py`（同时输出 16 张 test 抽样预测拼图与明细）。

---

## 三、文件清单

```
training_artifacts/
├─ README.md                       ← 本文件
├─ delivery_report.json            ← 阈值扫描 + 背景误报统计 + 延迟
├─ eval_test_metrics.json          ← v2 baseline 在 test 上的官方指标
├─ fire_smoke_v3_26n/              ← 当前交付版
│  ├─ args.yaml                    ← ★ 完整训练参数（可直接喂给 model.train）
│  ├─ results.csv                  ← 逐 epoch：loss / P / R / mAP50 / mAP50-95 / lr
│  ├─ labels.jpg                   ← 标注分布（框中心热力图 + 宽高比 + 数量）
│  ├─ val_batch0~2_{labels,pred}.jpg ← 验证集标注 vs 预测对照（定性检查）
│  ├─ curves/
│  │  ├─ results.png               ← 总曲线（loss + 四项指标）
│  │  ├─ confusion_matrix.png      ← 混淆矩阵（计数）
│  │  ├─ confusion_matrix_normalized.png ← 混淆矩阵（归一化）
│  │  └─ Box{P,R,F1,PR}_curve.png  ← 精度/召回/F1/PR 曲线
│  └─ weights/
│     └─ fire_smoke_v3_26n_best.pt ← ★ 最优权重（5.4 MB）
└─ fire_smoke_26n_v2baseline/      ← 上一版（v2，4,000 张）留作对照
   ├─ results.csv / results.png
   └─ confusion_matrix_normalized.png
```

★ = 复现/部署必需；其余为验收证据。

---

## 四、关键训练参数（摘自 args.yaml）

| 参数 | 值 | 为什么这么设 |
|---|---|---|
| `epochs` | 80 | 28k 数据下 80 轮已收敛（best 在 ep67） |
| `patience` | 25 | 连续 25 轮无提升即早停 |
| `batch` | 32 | 12GB 显存下的极大值 |
| `imgsz` | 640 | 与部署分辨率一致 |
| `mosaic` / `close_mosaic` | 1.0 / 15 | 小目标（<0.25% 画面占 28%）靠 mosaic 提召回，末期关掉回归真实分布 |
| `hsv_h/s/v` | 0.015 / 0.7 / 0.5 | 火/烟对亮度与饱和度极敏感，给足 |
| `degrees` | 5.0 | 火焰无固定朝向，轻微旋转即可 |
| `fliplr` | 0.5 | 烟羽流左右对称，可翻 |
| `flipud` | 0.0 | **禁止**上下翻：烟只向上飘，翻了就是反物理 |
| `amp` | true | 混合精度，省显存约 30% |
| `optimizer` | auto | 由 ultralytics 按数据量选 |
| `seed` / `deterministic` | 0 / true | 结果可复现 |

完整值见 `fire_smoke_v3_26n/args.yaml`。

---

## 五、已知短板（下一版要打的点）

1. **`fire` 召回偏低**：大片火场只框零星火点，mAP50-95 仅 0.416；
2. **无 `steam` 负类**：蒸汽/水雾会误报为 `smoke`，需靠下游时序判别兜底；
3. **小目标召回偏低**：<0.25% 画面的目标占 28%；
4. **跨域风险**：换新场地/新相机预计掉 10~20 点 mAP；
5. **框贴合度一般**：部分 `smoke` 框把山体、建筑一起吞进去（mAP50-95 低于 mAP50 的原因）。
