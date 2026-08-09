# prepare_v30.py — 生成 v30"区域→烟"训练集
# 思路: 用户需求"训练模型识别把香烟拿在手上时应该去哪找香烟"
#   用 head/hand 模型检测 v26_manual_orig(757张手动标注)每张图的头框/手框,
#   烟框与头/手扩张区域配对 → 裁剪区域ROI → 输出 (区域图 → 区域内烟框) 数据集
#   推理端(ROI裁剪放大独立检测)输入与训练分布一致, 模型专精"手上/嘴边的烟"
# 用法: python prepare_v30.py [--head-conf 0.35] [--hand-conf 0.25]
import cv2, numpy as np, os, sys, random
from ultralytics import YOLO
from pathlib import Path

SRC = Path(r"D:\training_data\smoke\v26_manual_orig")
OUT = Path(r"D:\training_data\smoke\v30_regions")
HEAD = YOLO(r"D:\视觉安防系统\models\yolov8n_head_v7.pt", task='detect')
HAND = YOLO(r"D:\视觉安防系统\models\hand.pt", task='detect')
HEAD.to('cuda'); HAND.to('cuda')

CONF_H = float(sys.argv[sys.argv.index('--head-conf')+1]) if '--head-conf' in sys.argv else 0.35
CONF_D = float(sys.argv[sys.argv.index('--hand-conf')+1]) if '--hand-conf' in sys.argv else 0.25
MAX_NEG_PER_IMG = 1      # 每张图最多1个负样本(区域无烟)
NEG_RATIO = 0.8          # 负样本总体保留比例(防正负失衡)

def detect(yolo, img, conf, W, H):
    """返回 (x1,y1,x2,y2) 像素框列表"""
    out = []
    try:
        r = yolo(img, conf=conf, iou=0.5, imgsz=640, verbose=False)
        for rr in r:
            for b in rr.boxes:
                bx1, by1, bx2, by2 = b.xyxy[0].cpu().numpy()
                if bx2-bx1 < 15 or by2-by1 < 15: continue
                out.append((float(bx1), float(by1), float(bx2), float(by2)))
    except Exception:
        pass
    return out

def expand_region(b, W, H, kind):
    """与 head_tracker_cam.py 相同的扩张参数"""
    x1, y1, x2, y2 = b
    w_, h_ = x2-x1, y2-y1
    if kind == 'head':
        return (max(0,x1-w_*0.6), max(0,y1-h_*0.9), min(W,x2+w_*0.6), min(H,y2+h_*0.6))
    else:
        return (max(0,x1-w_*1.2), max(0,y1-h_*0.6), min(W,x2+w_*1.2), min(H,y2+h_*1.6))

def main():
    random.seed(42)
    stats = {'imgs':0, 'head_det':0, 'hand_det':0, 'smoke_boxes':0,
             'paired_pos':0, 'neg_roi':0, 'pos_out':0, 'neg_out':0,
             'smoke_unmatched':0}
    for split in ('train','val'):
        (OUT/'images'/split).mkdir(parents=True, exist_ok=True)
        (OUT/'labels'/split).mkdir(parents=True, exist_ok=True)
    img_idx = 0
    for split in ('train','val'):
        img_dir = SRC/'images'/split
        lbl_dir = SRC/'labels'/split
        for imgf in sorted(os.listdir(img_dir)):
            if not imgf.lower().endswith(('.jpg','.jpeg','.png')): continue
            imgp = img_dir/imgf
            img = cv2.imread(str(imgp))
            if img is None: continue
            Hh, Ww = img.shape[:2]
            stats['imgs'] += 1
            # 1) 检测头/手
            heads = detect(HEAD, img, CONF_H, Ww, Hh)
            hands = detect(HAND, img, CONF_D, Ww, Hh)
            stats['head_det'] += len(heads); stats['hand_det'] += len(hands)
            # 2) 读烟标注(归一化→像素)
            lbp = lbl_dir/(imgf.rsplit('.',1)[0]+'.txt')
            smokes = []
            if lbp.exists():
                for line in lbp.read_text().strip().splitlines():
                    p = line.split()
                    if len(p) < 5: continue
                    cx, cy, w_, h_ = float(p[1])*Ww, float(p[2])*Hh, float(p[3])*Ww, float(p[4])*Hh
                    smokes.append((cx-w_/2, cy-h_/2, cx+w_/2, cy+h_/2))
            stats['smoke_boxes'] += len(smokes)
            # 3) 扩张区域(头+手, 按面积排序大的优先)
            regions = []
            for hb in heads: regions.append(expand_region(hb, Ww, Hh, 'head'))
            for db in hands: regions.append(expand_region(db, Ww, Hh, 'hand'))
            regions.sort(key=lambda r: -(r[2]-r[0])*(r[3]-r[1]))
            # 4) 烟框 → 区域配对(中心在区域内), 一个烟最多配一个区域(先到先得)
            smoke_used = [False]*len(smokes)
            per_img_neg = 0   # 本张图已写负样本数(每张最多MAX_NEG_PER_IMG个)
            for ri, (rx1, ry1, rx2, ry2) in enumerate(regions):
                roi_smokes = []
                for si, (sx1, sy1, sx2, sy2) in enumerate(smokes):
                    if smoke_used[si]: continue
                    scx, scy = (sx1+sx2)/2, (sy1+sy2)/2
                    if not (rx1 <= scx <= rx2 and ry1 <= scy <= ry2): continue
                    # 烟框大部分要在区域内(相交≥60%), 否则训练时信息不完整
                    ix1, iy1 = max(rx1,sx1), max(ry1,sy1)
                    ix2, iy2 = min(rx2,sx2), min(ry2,sy2)
                    inter = max(0,ix2-ix1)*max(0,iy2-iy1)
                    sarea = (sx2-sx1)*(sy2-sy1)
                    if sarea <= 0 or inter/sarea < 0.6: continue
                    smoke_used[si] = True
                    roi_smokes.append((max(ix1,rx1), max(iy1,ry1), min(ix2,rx2), min(iy2,ry2)))
                # 裁剪 ROI 原图
                rw, rh = int(rx2-rx1), int(ry2-ry1)
                if rw < 24 or rh < 24: continue
                roi_img = img[int(ry1):int(ry2), int(rx1):int(rx2)]
                # 正/负样本
                if roi_smokes:
                    name = f"{imgf.rsplit('.',1)[0]}_r{ri}.jpg"
                    cv2.imwrite(str(OUT/'images'/split/name), roi_img)
                    lines = []
                    for (sx1, sy1, sx2, sy2) in roi_smokes:
                        cxx = ((sx1+sx2)/2 - rx1)/rw; cyy = ((sy1+sy2)/2 - ry1)/rh
                        ww_ = (sx2-sx1)/rw; hh_ = (sy2-sy1)/rh
                        lines.append(f"0 {cxx:.6f} {cyy:.6f} {ww_:.6f} {hh_:.6f}")
                    (OUT/'labels'/split/(name.rsplit('.',1)[0]+'.txt')).write_text('\n'.join(lines))
                    stats['pos_out'] += 1
                else:
                    # 负样本: 每张图最多MAX_NEG_PER_IMG个 + 总体比例控制
                    if per_img_neg >= MAX_NEG_PER_IMG: continue
                    if random.random() > NEG_RATIO: continue
                    per_img_neg += 1
                    name = f"{imgf.rsplit('.',1)[0]}_r{ri}.jpg"
                    cv2.imwrite(str(OUT/'images'/split/name), roi_img)
                    (OUT/'labels'/split/(name.rsplit('.',1)[0]+'.txt')).write_text('')
                    stats['neg_out'] += 1
            stats['smoke_unmatched'] += len(smokes) - sum(smoke_used)
    # 统计未配对烟(中心不在任何区域)
    print("=== v30 数据生成完成 ===")
    for k, v in stats.items():
        print(f"  {k}: {v}")
    print(f"  输出目录: {OUT}")

if __name__ == '__main__':
    main()
