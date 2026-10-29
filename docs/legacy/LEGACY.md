# 旧系统参考存档（香烟检测 · smoking-detection）

> 本目录只保留**日后需要参考的配置与结论**，不保留代码实现。
> 完整代码在 tag `smoking-final-20261029` 与 `master` 分支里，随时可取。

## 一、为什么弃用

旧系统是「香烟/吸烟行为检测」，最终形态是：
**YOLO 检测头/手 → 在头手框内判"非肤色物体" → 最佳帧截图 + 超分 → Web 前端展示**。

弃用的原因不是精度不够，而是**目标变了**：现在要做安检场景的**火情/烟雾**检测。
两类任务的数据、指标、误报代价完全不同，继续在旧代码上改只会背一堆历史包袱。

## 二、旧版运行参数（原样保留，可直接对照）

| 段落 | 参数 | 值 | 新系统是否复用 |
|---|---|---|---|
| cameras | 多相机列表（id/name/source/enabled） | 2 路 USB | ✅ 结构直接沿用 |
| model | confidence_threshold | **0.50** | ✅ 同量级，烟雾建议更低（0.25~0.35） |
| model | iou_threshold | 0.5 | ✅ |
| decision | confirm_frames | **1** | ⚠️ 新系统要 **N≥8**（视频级确认） |
| decision | confirm_window_seconds | 0.5 | ✅ 概念沿用 |
| decision | cooldown_seconds | **3** | ✅ 建议新系统放宽到 30~60 |
| performance | inference_interval | 1 | ✅ 每帧都推理 |
| performance | max_queue_size | 3 | ✅ 丢旧帧防积压 |
| performance | warmup_frames | 10 | ✅ 新系统固定机位需要 20~30 |
| performance | display 960×540 | — | 参考 |
| alert | save_screenshot / screenshot_dir | true / alerts | ✅ |
| logging | level / file / max_size_mb | INFO / logs/app.log / 20 | ✅ |
| ui | primary_color | `#1E5AA8` | 可沿用 |

> 关键差异：旧系统 `confirm_frames=1`（单帧定论），新系统必须多帧确认 —— 这正是
> "不要单帧识别"的直接落地。

## 三、旧系统踩过的坑（从备份文件名与提交记录可查）

这些坑在新系统里**会以另一种形式重现**，提前记下：

| 坑 | 旧系统表现 | 新系统对应风险 |
|---|---|---|
| 双目标互斥 | 手/头框互相覆盖 → "手头互斥""手挡头锁定" | 火/烟框重叠时归属混乱 |
| 防闪烁 | "防闪缓冲""框断续""模糊续显" | 烟雾被快速帧间抖动 → 事件反复触发 |
| 偏框/漂移 | "偏框防御 v4""中心 EMA""位置钳制" | 云台转动时框被拉花 |
| 阈值敏感 | "conf010""手阈值038""接触放宽" | 烟雾半透明，置信度天然偏低 |
| 速度与精度拉扯 | "CPU 超分""RIFE 补帧""engine 导出" | 视频流水线延迟 |
| 轨迹关联 | "BOTSORT""烟BOTSORT""高速跟手" | 烟团分裂/合并导致 ID 跳变 |

**共性教训**：目标形态多变（手、烟、火、雾都是无定形的）时，
**单帧规则永远调不完**，必须靠时序与多帧一致性兜底 —— 这也是新系统一开始就把
"时序判别"做成独立模块的原因。

## 四、旧系统用过的模型（仅作记录）

- 人体/头部：`yolov8n.onnx`、`head_v12.pt`（YOLOv12s 微调）
- 手部：`hand.pt`（11k_Hands + 自采 1054 张）
- 香烟：`smoke_cig_v14~v30.pt`（最终 V30，mAP50 0.797）
- 辅助：`yolov8n-pose.onnx`、MediaPipe 21 点、MiDaS 深度、Real-ESRGAN 超分
- 追踪：BoT-SORT + KalmanBox + 位移感知 EMA

> 这些权重仍在本地 `models/` 目录（已从 git 移除，不再随仓库分发）。

## 五、如何恢复完整旧版

```bash
# 整个仓库回到旧版
git checkout smoking-final-20261029

# 只取某个文件（不切换分支）
git checkout smoking-final-20261029 -- detector.py

# 上面的备份文件也都在这两个提交里
git log --all --oneline | head
```

## 六、新系统保留了哪些旧思路

1. **多相机 + 每路独立 worker**（旧 config 的 cameras 结构）
2. **决策层参数化**（confirm_frames / window / cooldown）
3. **性能节流**（inference_interval / max_queue_size / warmup）
4. **告警落盘**（截图 + 事件记录）
5. **"宁可慢一点，别误报"** —— 旧系统调了三个月阈值才稳定，这条经验直接继承
