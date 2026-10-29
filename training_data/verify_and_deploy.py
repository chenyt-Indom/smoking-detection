#!/usr/bin/env python3
"""自动验证头部追踪系统 — 测试视频 → 检测 → 评分 → 报告"""

import sys, os, yaml, time, cv2, numpy as np
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, "D:/视觉安防系统")
from detector import MultiStageDetector
from person_tracker import PersonTracker

# 安防级标准
STANDARDS = {
    "head_detection_rate": 0.85,    # 头部检出率 > 85%
    "tracking_continuity": 0.80,     # 追踪连续性 > 80%
    "box_stability": 0.15,           # 框抖动 < 15% (IoU变化标准差)
    "avg_frame_time_ms": 50,         # 每帧耗时 < 50ms
}

APP_DIR = Path("D:/视觉安防系统")

def load_config():
    with open(APP_DIR / "config.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def verify_head_tracking(video_path: str) -> dict:
    """运行头部追踪测试"""
    cfg = load_config()
    print(f"\n{'='*60}")
    print(f"🎯 头部追踪验证: {Path(video_path).name}")
    print(f"{'='*60}")
    
    detector = MultiStageDetector(
        person_model_path=str(APP_DIR / cfg["model"]["person_model"]),
        smoking_model_path=str(APP_DIR / cfg["model"]["smoking_model"]),
        pose_model_path=str(APP_DIR / cfg["model"].get("pose_model", "models/yolov8n-pose.onnx")),
        conf_threshold=cfg["model"].get("confidence_threshold", 0.35),
        iou_threshold=cfg["model"].get("iou_threshold", 0.45),
    )
    
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    print(f"  视频: {total_frames}帧 @ {fps:.0f}fps")
    
    # 统计
    head_history = defaultdict(list)  # track_id -> [(frame, bbox, head_center), ...]
    head_detected_frames = 0
    total_head_boxes = 0
    frame_times = []
    total_tracks_seen = set()
    
    for frame_idx in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break
        
        t0 = time.perf_counter()
        result = detector.detect(frame)
        dt = (time.perf_counter() - t0) * 1000
        frame_times.append(dt)
        
        heads = result.get("heads", [])
        if heads:
            head_detected_frames += 1
        
        for h in heads:
            tid = h["track_id"]
            bbox = h["bbox"]
            cx = (bbox[0] + bbox[2]) / 2
            cy = (bbox[1] + bbox[3]) / 2
            head_history[tid].append((frame_idx, bbox, (cx, cy)))
            total_head_boxes += 1
            total_tracks_seen.add(tid)
        
        if frame_idx % 100 == 0:
            print(f"  进度: {frame_idx}/{total_frames} ({100*frame_idx/total_frames:.0f}%) "
                  f"- {len(heads)}个头部追踪中")
    
    cap.release()
    
    # 计算指标
    n_frames = frame_idx + 1
    detection_rate = head_detected_frames / n_frames if n_frames > 0 else 0
    avg_time = np.mean(frame_times) if frame_times else 0
    
    # 追踪连续性: 每个track_id的帧数比例
    continuities = []
    for tid, records in head_history.items():
        if len(records) > 1:
            first_frame = records[0][0]
            last_frame = records[-1][0]
            span = last_frame - first_frame + 1
            cont = len(records) / span if span > 0 else 0
            continuities.append(cont)
    avg_continuity = np.mean(continuities) if continuities else 0
    
    # 框稳定性: 相邻帧间IoU变化的标准差
    iou_changes = []
    for tid, records in head_history.items():
        for i in range(1, len(records)):
            _, b1, _ = records[i-1]
            _, b2, _ = records[i]
            # 计算IoU
            x1 = max(b1[0], b2[0]); y1 = max(b1[1], b2[1])
            x2 = min(b1[2], b2[2]); y2 = min(b1[3], b2[3])
            inter = max(0, x2-x1) * max(0, y2-y1)
            a1 = (b1[2]-b1[0]) * (b1[3]-b1[1])
            a2 = (b2[2]-b2[0]) * (b2[3]-b2[1])
            iou = inter / max(1, a1 + a2 - inter)
            iou_changes.append(1 - iou)  # 变化量 = 1 - IoU
    
    iou_std = np.std(iou_changes) if iou_changes else 1.0
    
    # 评分
    metrics = {
        "video": Path(video_path).name,
        "total_frames": n_frames,
        "total_tracks": len(total_tracks_seen),
        "head_detection_rate": detection_rate,
        "tracking_continuity": avg_continuity,
        "box_stability_iou_std": iou_std,
        "avg_frame_time_ms": avg_time,
        "total_head_boxes": total_head_boxes,
    }
    
    # 达标检查
    checks = {
        "头部检出率": (detection_rate >= STANDARDS["head_detection_rate"],
                       f"{detection_rate:.1%} >= {STANDARDS['head_detection_rate']:.0%}"),
        "追踪连续性": (avg_continuity >= STANDARDS["tracking_continuity"],
                       f"{avg_continuity:.1%} >= {STANDARDS['tracking_continuity']:.0%}"),
        "框稳定性": (iou_std <= STANDARDS["box_stability"],
                     f"{iou_std:.3f} <= {STANDARDS['box_stability']}"),
        "帧处理速度": (avg_time <= STANDARDS["avg_frame_time_ms"],
                       f"{avg_time:.1f}ms <= {STANDARDS['avg_frame_time_ms']}ms"),
    }
    
    all_pass = all(v[0] for v in checks.values())
    
    print(f"\n📊 测试结果:")
    for name, (passed, detail) in checks.items():
        icon = "✅" if passed else "❌"
        print(f"  {icon} {name}: {detail}")
    
    print(f"\n📈 详细指标:")
    print(f"  检出帧率: {detection_rate:.1%}")
    print(f"  追踪连续性: {avg_continuity:.1%}")
    print(f"  框稳定性(IoU std): {iou_std:.4f}")
    print(f"  平均帧时: {avg_time:.1f}ms")
    print(f"  追踪人数: {len(total_tracks_seen)}")
    print(f"  总头部框数: {total_head_boxes}")
    
    if all_pass:
        print(f"\n✅ 安防级标准全部达标！")
    else:
        print(f"\n⚠️ 部分指标未达标，需要调整")
    
    return metrics, all_pass

def main():
    test_videos = [
        "D:/视觉安防系统/training_data/test_videos/test_heads_moving.mp4",
        "D:/视觉安防系统/training_data/test_videos/test_crowd.mp4",
    ]
    
    all_results = []
    all_pass = True
    
    for vp in test_videos:
        if not os.path.exists(vp):
            print(f"⚠️ 测试视频不存在: {vp}")
            continue
        metrics, passed = verify_head_tracking(vp)
        all_results.append(metrics)
        if not passed:
            all_pass = False
    
    # 保存结果
    import json
    result_file = "D:/视觉安防系统/training_data/verification_result.json"
    with open(result_file, "w", encoding="utf-8") as f:
        json.dump({
            "all_pass": all_pass,
            "standards": STANDARDS,
            "results": all_results,
        }, f, indent=2, ensure_ascii=False)
    print(f"\n📝 验证结果已保存: {result_file}")
    
    if all_pass:
        # 部署模型
        deploy()
    
    return all_pass

def deploy():
    """部署优化后的模型和应用"""
    print(f"\n{'='*60}")
    print(f"🚀 部署优化后的系统到 D盘应用")
    print(f"{'='*60}")
    
    import shutil
    
    # detector.py 已修改，同步到 git 仓库
    src_detector = "D:/视觉安防系统/detector.py"
    dst_detector = "C:/Users/19853/WorkBuddy/2026-08-02-11-30-04/smoking-detection/detector.py"
    shutil.copy2(src_detector, dst_detector)
    print(f"  ✅ detector.py 已同步到git仓库")
    
    print(f"\n✅ 部署完成!")
    print(f"   D:/视觉安防系统/detector.py 已优化头部框估算")
    print(f"   视频显示: 仅显示绿色头部追踪框")
    print(f"   追踪: 速度插值 + EMA平滑 + 多策略头部定位")

if __name__ == "__main__":
    main()
