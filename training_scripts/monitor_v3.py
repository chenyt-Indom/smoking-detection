"""V3 训练实时监控"""
import os, time, subprocess

CSV = r"D:\training_data\logs\person_v3\results.csv"
LOG = r"D:\training_data\logs\person_v3"
start = time.time()

while True:
    os.system('cls')
    elapsed = (time.time() - start) / 60
    
    print("=" * 55)
    print(f"  V3 训练监控  {time.strftime('%H:%M:%S')}  已运行 {elapsed:.0f}分钟")
    print("=" * 55)

    if os.path.exists(CSV):
        lines = [l.strip().split(',') for l in open(CSV).readlines()]
        if len(lines) > 1:
            data = lines[1:]
            for d in data:
                ep = d[0]
                box = float(d[2])
                m50 = float(d[7])
                m95 = float(d[8])
                bar = "█" * int(m95 * 25)
                print(f"  E{ep:>2s} box={box:.4f} mAP50={m50:.4f} mAP50-95={m95:.4f} {bar}")
            
            maps = [float(d[8]) for d in data]
            if len(maps) >= 3:
                inc = sum(1 for i in range(1,len(maps)) if maps[i]>maps[i-1])
                dec = len(maps)-1-inc
                print(f"\n  {'🛡️ 无过拟合' if inc>=dec else '⚠️ 关注!'}  上升{inc}次 下降{dec}次")
            
            d = data[-1]
            ep = int(d[0])
            eta = int((25-ep)*2.7)
            print(f"\n  {ep}/25 完成  剩余 ~{eta}分钟")
        else:
            print("  ⏳ 首轮验证写入中...")
    else:
        # 检查进程
        try:
            out = subprocess.check_output('nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits',shell=False).decode()
            gpu, vram = out.strip().split(', ')
            print(f"  🔄 GPU:{gpu}%  VRAM:{vram}MB  Epoch 1 训练中...")
        except:
            print("  ⏳ 启动中...")

    print(f"\n  workers=0  cache=True  batch=64  dropout=0.25")
    print(f"  数据: 8030+1600+30真实 | 25轮 | ~1小时")
    time.sleep(3)
