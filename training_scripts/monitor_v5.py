"""V5 训练监控 - 完整详情"""
import os, time, subprocess

CSV = r"D:\training_data\logs\person_v5\results.csv"
start = time.time()

while True:
    os.system('cls')
    elapsed = (time.time() - start) / 60
    print("=" * 78)
    print(f"  V5 训练监控  {time.strftime('%H:%M:%S')}  已监控 {elapsed:.0f}分")
    print(f"  30轮 | dropout=0.30 wd=0.003 | patience=10 | warmup(3)+close(3)")
    print("=" * 78)

    if os.path.exists(CSV):
        lines = [l.strip().split(',') for l in open(CSV).readlines()[1:]]
        # 表头
        print(f"  {'Ep':<4s} {'box':>7s} {'cls':>7s} {'dfl':>7s} {'P':>6s} {'R':>6s} {'mAP50':>7s} {'mAP95':>7s} {'v_box':>7s} {'趋势':>6s}")
        print("  " + "-" * 76)

        maps = []; best_m = 0
        for i, d in enumerate(lines):
            e    = int(d[0])
            box  = float(d[2])
            cls  = float(d[3])
            dfl  = float(d[4])
            P    = float(d[5])
            R    = float(d[6])
            m50  = float(d[7])
            m95  = float(d[8])
            vbox = float(d[9]) if len(d) > 9 else 0
            maps.append(m95)

            # 趋势箭头
            if i >= 1:
                diff = m95 - maps[i-1]
                if diff > 0.005: arrow = " ▲▲"
                elif diff > 0.001: arrow = " ▲ "
                elif diff > -0.001: arrow = " ▬ "
                elif diff > -0.005: arrow = " ▼ "
                else: arrow = " ▼▼"
            else:
                arrow = " --"

            tag = " 🏆" if m95 >= max(maps) and len(maps) > 1 else ""
            print(f"  {e:<4d} {box:7.4f} {cls:7.4f} {dfl:7.4f} {P:6.2%} {R:6.2%} {m50:7.4f} {m95:7.4f} {vbox:7.4f} {arrow}{tag}")

        # 底部统计
        if len(maps) >= 2:
            inc = sum(1 for i in range(1, len(maps)) if maps[i] > maps[i-1] + 0.001)
            dec = sum(1 for i in range(1, len(maps)) if maps[i] < maps[i-1] - 0.001)
            flat = len(maps) - 1 - inc - dec
            print(f"\n  {inc}升 {dec}降 {flat}平", end="  ")

            # 过拟合: box_loss持续降但mAP连续降
            boxes = [float(d[2]) for d in lines]
            box_dropping = boxes[-1] < boxes[-2]
            map_dropping = len(maps) >= 3 and maps[-1] < maps[-2] and maps[-2] < maps[-3]
            if maps[-1] > maps[0] and boxes[-1] < boxes[0]:
                print("✅ 健康 (box↓ + mAP↑)")
            elif map_dropping and box_dropping:
                print("⚠️ 过拟合 (box↓ 但 mAP连降)")
            else:
                print("🟡 观察中")

            best = max(maps); best_ep = lines[maps.index(best)][0]
            ep = int(lines[-1][0])
            print(f"  🏆 最佳 E{best_ep}  mAP50-95={best:.4f}")
            print(f"  📊 {ep}/30 完成", end="")
            if ep > 0 and ep < 30:
                avg_min_per_epoch = elapsed / ep
                remaining = int((30 - ep) * avg_min_per_epoch)
                print(f"  |  均值 {avg_min_per_epoch:.1f}分/轮  |  剩余 ~{remaining}分  |  预计 {time.strftime('%H:%M', time.localtime(time.time() + remaining*60))}")
            else:
                print()

    else:
        print(f"\n  🔄 Epoch 1 训练中 (Cache + AMP pass)")

    print(f"\n  V4→V5 | 10000合成+30真实 | mosaic=0.5 mixup=0.4 erasing=0.8")
    print(f"  SCUT-HEAD下载中 (V6混入真人数据)")
    time.sleep(3)
