"""云端 DeepSeek vs 本地 Qwen3-VL 的裁判 A/B 对比。

为什么要做这个
--------------
"火苗 vs 手机手电筒"是细粒度判别，8B 和 550B 的差距可能正好体现在这里。
不实测就切换，等于拿生产环境试错。

测试素材
--------
· hard_negatives/images/  —— 用户手动点过"误报"的图，**已知不是火**
· alerts/*.jpg           —— 现场事件截图，覆盖火苗/灯光/反光/白墙

判定口径
--------
每张图分别喂给两个模型，比较 label 是否一致。
不一致的挑出来人工看 —— 那才是真正需要决策的地方。
（全一致说明 8B 够用，可以放心切本地省钱。）

用法
----
    python scripts/ab_arbiter.py                 # 默认各取 15 张
    python scripts/ab_arbiter.py --n 30 --out ab.json
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.arbiter import LlmArbiter                      # noqa: E402

LOCAL_URL = "http://127.0.0.1:11434/v1/chat/completions"


def collect(n: int, seed: int = 7):
    """收集测试图：先放已知误报，再用事件截图补足。"""
    picks: list[tuple[Path, str]] = []
    hn = sorted((ROOT / "hard_negatives" / "images").glob("*.jpg"))
    for p in hn[:n]:
        picks.append((p, "已知误报"))
    need = n - len(picks)
    if need > 0:
        al = sorted((ROOT / "alerts").glob("*.jpg"))
        random.seed(seed)
        if len(al) > need:
            al = random.sample(al, need)
        for p in al:
            tag = "事件_" + (p.stem.rsplit("_", 1)[-1] if "_" in p.stem else "?")
            picks.append((p, tag))
    return picks


def judge_one(arb: LlmArbiter, img, box) -> dict:
    """同步跑一次裁决（走 arbiter 自己的编码/提示词，保证口径一致）。"""
    key = f"ab:{time.time_ns()}"
    ok, why = arb.submit(key, img, box, rule_label="uncertain", rule_score=0.0,
                         context="A/B 对比测试")
    if not ok:
        return {"label": "submit_fail", "error": why, "latency": 0.0}
    t0 = time.time()
    while time.time() - t0 < 60:
        for _k, res in arb.poll():
            return {"label": res.label, "risk": res.risk,
                    "conf": round(float(res.confidence), 2),
                    "latency": round(float(res.latency_s), 2),
                    "reason": (res.reason or "")[:50],
                    "source": res.source}
        time.sleep(0.2)
    return {"label": "timeout", "error": "60s 未返回", "latency": 60.0}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=15)
    ap.add_argument("--out", default="logs/ab_result.json")
    a = ap.parse_args()

    cloud = LlmArbiter(cfg={"hourly_limit": 9999, "dedup_seconds": 0})
    local = LlmArbiter(cfg={"url": LOCAL_URL, "model": "qwen3-vl:8b",
                            "api_key_env": "NO_NEED", "timeout_s": 60,
                            "hourly_limit": 9999, "dedup_seconds": 0})

    for tag, arb in (("云端", cloud), ("本地", local)):
        s = arb.snapshot()
        print(f"  {tag}: model={s['model']} local={s['local']} "
              f"configured={s['configured']}")

    picks = collect(a.n)
    print(f"\n共 {len(picks)} 张，开始对比…\n")

    rows = []
    agree = 0
    t_cloud = t_local = 0.0
    for i, (p, tag) in enumerate(picks, 1):
        img = cv2.imread(str(p))
        if img is None:
            print(f"  [{i}] 读不了 {p.name}")
            continue
        h, w = img.shape[:2]
        # 截图本身已经是 ROI，全图也用同一张：它没有原始画面上下文，
        # 但两个模型面对的条件完全一样，对比是公平的
        box = (0, 0, w - 1, h - 1)
        rc = judge_one(cloud, img, box)
        rl = judge_one(local, img, box)
        same = rc.get("label") == rl.get("label")
        agree += int(same)
        t_cloud += rc.get("latency", 0)
        t_local += rl.get("latency", 0)
        rows.append({"file": p.name, "tag": tag, "cloud": rc, "local": rl,
                     "same": same})
        flag = "一致" if same else "**分歧**"
        print(f"  [{i:>2}] {tag:8s} {p.name[:34]:34s} "
              f"云={rc.get('label'):8s} 本={rl.get('label'):8s} {flag}")

    n = max(len(rows), 1)
    print(f"\n{'='*66}")
    print(f"  一致 {agree}/{len(rows)}（{agree/n*100:.0f}%）")
    print(f"  平均延迟  云端 {t_cloud/n:.2f}s   本地 {t_local/n:.2f}s")
    diffs = [r for r in rows if not r["same"]]
    if diffs:
        print(f"\n  分歧 {len(diffs)} 条（这些才需要人工定夺）：")
        for r in diffs[:10]:
            print(f"    {r['file'][:36]}")
            print(f"      云: {r['cloud'].get('label')} "
                  f"{r['cloud'].get('reason','')}")
            print(f"      本: {r['local'].get('label')} "
                  f"{r['local'].get('reason','')}")
    print(f"{'='*66}")

    out = ROOT / a.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n明细已写入 {out}")
    cloud.close()
    local.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
