"""V4 实时监控 - 含续训阶段"""
import os, time, subprocess

CSV = r"D:\training_data\logs\person_v4\results.csv"
start = time.time()

while True:
    os.system('cls')
    elapsed = (time.time() - start) / 60
    print("=" * 58)
    print(f"  V4 训练监控  {time.strftime('%H:%M:%S')}  已监控 {elapsed:.0f}分")
    print(f"  [相位1: E1-25 完成] → [相位2: E26-40 续训中]")
    print("=" * 58)

    if os.path.exists(CSV):
        lines = [l.strip().split(',') for l in open(CSV).readlines()]
        if len(lines) > 1:
            data = lines[1:]
            maps = [float(d[8]) for d in data]
            boxes = [float(d[2]) for d in data]
            n = len(maps)
            ep = int(data[-1][0])

            # 分两阶段显示
            phase1 = [d for d in data if int(d[0]) <= 25]
            phase2 = [d for d in data if int(d[0]) > 25]

            if phase1:
                print(f"\n  ── 阶段1 (E1-E25) ──")
                # 仅显示关键点：首、最佳、尾
                key_eps = [phase1[0], phase1[-1]]
                best_p1 = max(phase1, key=lambda d: float(d[8]))
                if best_p1 not in key_eps:
                    key_eps.append(best_p1)
                key_eps.sort(key=lambda d: int(d[0]))
                for d in key_eps:
                    e, b, m = d[0], float(d[2]), float(d[8])
                    tag = "🏆" if d == best_p1 else ""
                    bar = "█" * int(m * 28)
                    print(f"  E{e:>2s}  box={b:.4f}  mAP50-95={m:.4f}  {bar}  {tag}")
                if len(phase1) > 3:
                    v0 = float(phase1[0][8])
                    v1 = float(phase1[-1][8])
                    gain = v1 - v0
                    print(f"  E1→E25: {gain:+.4f}  ({int(gain*100)}% 提升)")

            if phase2:
                print(f"\n  ── 阶段2 (E26-E40) 续训中 ──")
                for d in phase2[-8:]:
                    e, b, m = d[0], float(d[2]), float(d[8])
                    bar = "█" * int(m * 28)
                    tag = "🆕最新" if d == phase2[-1] else ""
                    print(f"  E{e:>2s}  box={b:.4f}  mAP50-95={m:.4f}  {bar}  {tag}")

                if len(phase2) >= 2:
                    p2maps = [float(d[8]) for d in phase2]
                    p2gain = p2maps[-1] - p2maps[0]
                    print(f"  E26→E{ep}: {p2gain:+.4f}")

                # 全局最佳
                best_all = max(data, key=lambda d: float(d[8]))
                best_e, best_m = best_all[0], float(best_all[8])
                print(f"\n  🏆 全局最佳: E{best_e}  mAP50-95={best_m:.4f}  差距: {best_m-maps[-1]:.4f}")

            # 过拟合检测
            if n >= 5 and ep > 25:
                recent_maps = maps[-5:]
                recent_boxes = boxes[-5:]
                inc = sum(1 for i in range(4) if recent_maps[i+1] > recent_maps[i] + 0.001)
                if inc >= 3 and recent_boxes[-1] < recent_boxes[0]:
                    print(f"  ✅ 健康: {inc}/4升+mAP↑+box↓")
                elif inc <= 1:
                    print(f"  ⚠️ 停滞: {inc}/4升 → 可能到顶了")

            remaining = 40 - ep if ep < 40 else 0
            if remaining > 0:
                eta = int(remaining * 3.0)
                print(f"\n  {ep}/40 完成 | 剩余 ~{eta}分钟")

    else:
        print("\n  ⏳ 训练启动中...")

    print(f"\n  dropout=0.25 | mixup=0.65 | erasing=0.75")
    print(f"  12060张 | 42166框 | workers=0 cache=True")
    print(f"  ⚠ CSV重写: 阶段1数据已丢失, 仅显示续训阶段2")
    time.sleep(3)
