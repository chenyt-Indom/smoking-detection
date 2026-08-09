"""V30 - "区域→烟" 专项训练(头/手区域里的烟检测)
数据集: smoke/v30_regions — 由 prepare_v30.py 从 v26_manual_orig(757张手动标注)生成
  输入 = 头框/手框扩张区域裁图(与推理端ROI裁剪放大一致)
  输出 = 区域内香烟框
目标: 模型专精"手上/嘴边的烟", 推理端 ROI 放大输入与训练分布匹配 → 手持烟/小烟检出质变

技术: 继承 V29 的 Wise-IoU v3 损失(定位精度), 起点 V28 best.pt
"""
import torch
import torch.nn as nn
from ultralytics import YOLO
from ultralytics.utils import metrics
from ultralytics.utils import loss as ULoss
import ultralytics.utils.loss as loss_mod

# ============ Wise-IoU v3 损失实现(同V29) ============
def bbox_iou_wiou(box1, box2, xywh=False, eps=1e-7):
    if xywh:
        (x1, y1, w1, h1), (x2, y2, w2, h2) = box1.chunk(4, -1), box2.chunk(4, -1)
        b1_x1, b1_x2 = x1 - w1 / 2, x1 + w1 / 2
        b1_y1, b1_y2 = y1 - h1 / 2, y1 + h1 / 2
        b2_x1, b2_x2 = x2 - w2 / 2, x2 + w2 / 2
        b2_y1, b2_y2 = y2 - h2 / 2, y2 + h2 / 2
    else:
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[..., 0], box1[..., 1], box1[..., 2], box1[..., 3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[..., 0], box2[..., 1], box2[..., 2], box2[..., 3]
    inter = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * \
            (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)
    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1
    union = w1 * h1 + w2 * h2 - inter + eps
    iou = inter / union
    cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)
    ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)
    c2 = cw ** 2 + ch ** 2 + eps
    rho2 = ((b2_x1 + b2_x2) - (b1_x1 + b1_x2)) ** 2 / 4 + \
           ((b2_y1 + b2_y2) - (b1_y1 + b1_y2)) ** 2 / 4
    R_wiou = torch.exp(rho2 / c2)
    wiou = 1.0 - iou
    return iou, R_wiou * wiou


class WIoUBboxLoss(ULoss.BboxLoss):
    def forward(self, pred_dist, pred_bboxes, anchor_points, target_bboxes,
                target_scores, target_scores_sum, fg_mask, imgsz, stride):
        weight = target_scores[fg_mask].sum(-1, keepdim=True)
        pb = pred_bboxes[fg_mask]
        tb = target_bboxes[fg_mask]
        iou, wiou_loss = bbox_iou_wiou(pb, tb, xywh=False)
        with torch.no_grad():
            beta = wiou_loss.detach() / (wiou_loss.detach().mean() + 1e-7)
            alpha, delta = 1.9, 3.0
            r = beta / (delta * torch.pow(alpha, (beta - delta).clamp(-10, 10)))
            r = r.clamp(0, 10)
        loss_iou = (r * wiou_loss * weight).sum() / target_scores_sum
        target_ltrb = ULoss.bbox2dist(anchor_points, target_bboxes, self.dfl_loss.reg_max - 1)
        loss_dfl = self.dfl_loss(pred_dist[fg_mask].view(-1, self.dfl_loss.reg_max),
                                 target_ltrb[fg_mask]) * weight
        loss_dfl = loss_dfl.sum() / target_scores_sum
        return loss_iou, loss_dfl


loss_mod.BboxLoss = WIoUBboxLoss
print('✅ Wise-IoU v3 损失已注入')

# ============ 训练配置 ============
# 起点: V28 best(当前部署, mAP95=0.495 优于V29)
model = YOLO(r'D:\training_data\runs\detect\cig_v28\weights\best.pt')
print('V30: V28起点 + 区域数据集v30_regions + WIoUv3 | 手持/嘴边烟专项')

model.train(
    data=r"D:/training_data/smoke/v30_regions/data.yaml",
    epochs=120,
    imgsz=640,           # ★ 与推理端ROI放大尺寸一致(区域图为主, 640足够)
    batch=24,            # 640输入显存占用小, batch可大
    name='cig_v30',
    patience=30,
    save=True,
    save_period=25,
    device=0,
    workers=0,
    lr0=0.0003,
    lrf=0.002,
    optimizer='AdamW',
    cos_lr=True,
    warmup_epochs=4,
    close_mosaic=15,
    amp=True,
    freeze=4,            # 保留V28记忆
    box=9.5,
    dfl=3.0,
    # --- 增强(区域图场景: 烟在手上/嘴边, 旋转+平移为主) ---
    mosaic=0.7,
    mixup=0.15,
    copy_paste=0.15,
    scale=0.8,
    translate=0.2,
    shear=10,
    degrees=45,
    flipud=0.1,
    fliplr=0.5,
    hsv_h=0.08,
    hsv_s=1.0,
    hsv_v=0.9,
    erasing=0.4,
    crop_fraction=0.9,
)
print("\n✅ V30 DONE! runs/detect/cig_v30/weights/best.pt")
