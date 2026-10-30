# 安检视觉：火情 / 烟雾 视频检测

> 从摄像头视频里检测**火焰**与**烟雾（白烟 / 灰烟 / 黑烟）**，并把**蒸汽、水雾**排除掉。
> **视频级判定**，不做单帧分类。

## 为什么这件事必须做在视频上

白烟和蒸汽在单帧图像里是同一种东西：同为白色、同为半透明、同为上升羽流。
任何单帧模型都只能靠背景瞎猜 —— 这是同类产品误报率高的根因。

一旦有时间维度，两者差别很清楚：

| 特征 | 烟 | 蒸汽 / 水雾 |
|---|---|---|
| 面积增长 | 慢而持续 | 几秒内快速膨胀再变淡 |
| 存续时长 | 数十秒~分钟 | 几秒~十几秒 |
| 亮度走向 | 白 → 灰 → 黑（碳颗粒） | 始终偏白，只会变淡 |
| 边界清晰度 | 相对稳定（有纹理） | 越来越糊 |
| 与火源关系 | 常伴明火 | 与火无关（壶口/管道口） |

**结论：检测器负责"看见"，时序模块负责"确认是什么"。**


## 训练结果（v3 · 3.78 万张合并数据）

数据集：FASDD 两批（去重后 37,818 张唯一图，含 1,051 张纯背景负样本）
划分：train 28,381 / valid 6,496 / test 2,920（**按编号段分层切分，防相邻帧泄漏**）
模型：YOLO26n（2.4M 参数）· 640 分辨率 · batch 32 · 80 轮

| 指标 | val（最佳 ep67） | test（2,920 张） |
|---|---|---|
| mAP50 | 0.7761 | **0.7646** |
| mAP50-95 | 0.5274 | 0.5112 |
| Precision | — | 0.762 |
| Recall | — | 0.673 |

**误报率**（1,051 张纯背景画面，无火无烟）：

| conf 门限 | 误报张数 | 误报率 |
|---|---|---|
| 0.25 | 48 | 4.6% |
| 0.35 | 22 | **2.1%** |
| 0.50 | 10 | **1.0%** |

**速度**：10.4 ms/张（约 96 FPS，含预处理与后处理，RTX 5070 Ti Laptop）

> 训练参数、逐轮指标、曲线、混淆矩阵与**最优权重**见 [`training_artifacts/`](training_artifacts/README.md)。
> 训练集（37,818 张 / 约 14 GB）不入库，只保留规模与切分口径。

对比上一版（4,000 张）：mAP50 0.606 → 0.765，提升 **+15.9 点**；同模型未换，只加数据。

已知短板：无 `steam` 负类（蒸汽会误报）、小目标（<0.25% 画面）占 28% 召回偏低、
跨域（换新场地）预计掉 10~20 点。

## 目录

```
src/
  sources/            采集层：两种相机两种背景策略
    base.py           FramePacket 契约（moving / homography / fg_mask）
    fixed.py          固定机位：中值背景 + MOG2 → 变化区域
    ptz.py            云台：ORB 全局单应补偿 → 对齐后差分
  temporal.py         时序判别：轨迹特征 → smoke / steam / uncertain（带人话理由）
config/
  dataset.yaml        3 类数据配置：fire / smoke / steam
  train_l.yaml        YOLO26l 训练配置（12GB 显存安全档）
  temporal.yaml       现场阈值（改这里，不改代码）
scripts/
  test_sources.py     采集层自测（11 项，合成数据）
  test_temporal.py    时序判别自测（5 项，合成轨迹）
  test_v3_effect.py   交付模型实跑：test 抽样预测 + 非火视频误报抽测
training_artifacts/   训练产出物（参数/指标/曲线/最优权重；训练集不入库）
  README.md           产出清单、指标口径、关键超参、已知短板
docs/
  ARCHITECTURE.md     五级流水线、双模采集、评估规范、里程碑
```

## 当前进度

| 阶段 | 状态 | 证据 |
|---|---|---|
| M0 旧项目归档 | ✅ | tag `smoking-final-20261029` |
| M1 时序判别器 | ✅ | `scripts/test_temporal.py` 5/5（含"蒸汽旁边有火"干扰场景） |
| M1.5 双模采集层 | ✅ | `scripts/test_sources.py` 11/11（烟心误差 2px、云台位移误差 0.4px） |
| M2 数据管线 | ⏳ | 待做：D-Fire + 自建 steam，**按视频切分** |
| M3 baseline 训练 | ⏳ | YOLO26l |
| M4 视频推理打通 | ⏳ | 五级流水线 |
| M5 时序分类器替换规则版 | ⏳ | 用积累的 tracklet 训 GBDT |

## 跑一下自测

```bash
# 环境：Python 3.13 + torch 2.11(cu128) + ultralytics 8.4.115 + opencv
python scripts/test_sources.py     # 采集层
python scripts/test_temporal.py    # 时序判别
```

## 关于旧版本（香烟检测）

代码已从本分支移除，但**没有丢**：

```bash
git checkout smoking-final-20261029      # 完整回到旧版
git checkout smoking-final-20261029 -- detector.py    # 只取某个文件
```

旧版权重（`models/*.pt`）不再进 git，仍在本地 `models/` 目录里。

## 许可

检测框架 YOLO26 属 **AGPL-3.0**（教学/自用无碍；商用需开源或购买 Ultralytics 商业授权）。
