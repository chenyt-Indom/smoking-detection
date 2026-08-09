"""V29 - mAP50-95 定位精度专项训练
基于 V28 best.pt 续训, 引入新技术:

★ 新技术1: Wise-IoU v3 (动态非单调聚焦损失)
   - 替代默认 CIoU: 对低质量样本动态降权, 聚焦高质量框回归
   - 显著提升 mAP50-95 (框定位精度), 业界验证
★ 新技术2: 数据集扩充 +35%
   - v27_mix(4802) + v26_manual(4237手动高精度) + stage6_cig(72)
   - 负样本 2208 张(防误检) → 共 8233 train
★ 新技术3: imgsz 768→896 (小目标像素+36%, 框回归更精细)
★ 新技术4: DFL 3.0 + box 9.5 (更用力精修边框)
★ 新技术5: 数据增强强化 (copy_paste 0.2 / erasing 0.5 / mosaic 0.8)
"""
import torch
import torch.nn as nn
from ultralytics import YOLO
from ultralytics.utils import metrics
from ultralytics.utils import loss as ULoss
import ultralytics.utils.loss as loss_mod

# ============ Wise-IoU v3 损失实现 ============
def bbox_iou_wiou(box1, box2, xywh=False, eps=1e-7):
    """Wise-IoU v3: 返回 (iou, wiou_loss)
    WIoUv3 = r * R_wiou * (1 - IoU)
      - R_wiou = exp(中心距²/外接框对角线²): 距离惩罚
      - β = 离群度(当前IoU损失/平均), r = 动态非单调聚焦系数
    """
    # 转 xyxy
    if xywh:
        (x1, y1, w1, h1), (x2, y2, w2, h2) = box1.chunk(4, -1), box2.chunk(4, -1)
        b1_x1, b1_x2 = x1 - w1 / 2, x1 + w1 / 2
        b1_y1, b1_y2 = y1 - h1 / 2, y1 + h1 / 2
        b2_x1, b2_x2 = x2 - w2 / 2, x2 + w2 / 2
        b2_y1, b2_y2 = y2 - h2 / 2, y2 + h2 / 2
    else:
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[..., 0], box1[..., 1], box1[..., 2], box1[..., 3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[..., 0], box2[..., 1], box2[..., 2], box2[..., 3]

    # 交集
    inter = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * \
            (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)
    # 并集
    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1
    union = w1 * h1 + w2 * h2 - inter + eps
    iou = inter / union

    # 外接框
    cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)
    ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)
    c2 = cw ** 2 + ch ** 2 + eps  # 外接框对角线²

    # 中心距离²
    rho2 = ((b2_x1 + b2_x2) - (b1_x1 + b1_x2)) ** 2 / 4 + \
           ((b2_y1 + b2_y2) - (b1_y1 + b1_y2)) ** 2 / 4

    # WIoUv1: 距离惩罚 R_wiou = exp(rho2 / c2)
    R_wiou = torch.exp(rho2 / c2)
    wiou = 1.0 - iou  # 基础 IoU 损失(尚未乘R)

    return iou, R_wiou * wiou  # (iou, 距离加权损失)


class WIoUBboxLoss(ULoss.BboxLoss):
    """Wise-IoU v3 版 BboxLoss: 动态非单调聚焦, 提升 mAP50-95"""

    def forward(self, pred_dist, pred_bboxes, anchor_points, target_bboxes,
                target_scores, target_scores_sum, fg_mask, imgsz, stride):
        weight = target_scores[fg_mask].sum(-1, keepdim=True)
        pb = pred_bboxes[fg_mask]
        tb = target_bboxes[fg_mask]

        # WIoUv3: 基础 + 距离惩罚
        iou, wiou_loss = bbox_iou_wiou(pb, tb, xywh=False)
        # 动态非单调聚焦: β = 离群度, r = β/(δ·α^(β-δ))
        with torch.no_grad():
            beta = wiou_loss.detach() / (wiou_loss.detach().mean() + 1e-7)
            alpha, delta = 1.9, 3.0
            r = beta / (delta * torch.pow(alpha, (beta - delta).clamp(-10, 10)))
            r = r.clamp(0, 10)  # 防爆
        loss_iou = (r * wiou_loss * weight).sum() / target_scores_sum

        # DFL loss (保持原样)
        target_ltrb = ULoss.bbox2dist(anchor_points, target_bboxes, self.dfl_loss.reg_max - 1)
        loss_dfl = self.dfl_loss(pred_dist[fg_mask].view(-1, self.dfl_loss.reg_max),
                                 target_ltrb[fg_mask]) * weight
        loss_dfl = loss_dfl.sum() / target_scores_sum
        return loss_iou, loss_dfl


# 替换 ultralytics 默认 BboxLoss 为 WIoU 版
loss_mod.BboxLoss = WIoUBboxLoss
print('✅ Wise-IoU v3 损失已注入')

# ============ 训练配置 ============
model = YOLO(r'D:\training_data\runs\detect\cig_v28\weights\best.pt')
print('V29: V28起点 + WIoUv3 + imgsz896 + 数据+35% | mAP50-95 专项')

model.train(
    data=r"D:/training_data/smoke/v29_mix/data.yaml",
    epochs=150,
    imgsz=896,           # ★ 768→896: 小目标像素+36%
    batch=12,            # 896输入显存↑, 降batch(12GB显存)
    name='cig_v29',
    patience=45,
    save=True,
    save_period=25,
    device=0,
    workers=0,
    lr0=0.0002,
    lrf=0.002,
    optimizer='AdamW',
    cos_lr=True,
    warmup_epochs=5,
    close_mosaic=18,
    amp=True,
    freeze=4,            # 保留V28记忆
    box=9.5,             # ★ 8.5→9.5: 更用力框回归
    dfl=3.0,             # ★ 2.5→3.0: DFL精修边框
    # --- 增强 (强化) ---
    mosaic=0.8,
    mixup=0.2,
    copy_paste=0.2,
    scale=0.9,
    translate=0.15,
    shear=12,
    degrees=180,
    perspective=0.0005,
    flipud=0.1,
    fliplr=0.5,
    hsv_h=0.08,
    hsv_s=1.0,
    hsv_v=0.9,
    erasing=0.5,
    crop_fraction=0.9,
)
print("\n✅ V29 DONE! runs/detect/cig_v29/weights/best.pt")
