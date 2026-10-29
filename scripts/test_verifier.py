"""VLM 复核器自测：合成 4 类场景，验证判得住、判得对、降级可用。

验收标准（比"准确率"更贴近安防）：
  · 真火必须判成 fire        —— 漏报是最不能接受的
  · 非火（蒸汽/灯/烟）不许判成 fire —— 误报会让值班员关掉系统
  · 没 Key / 超额 / 去重 时必须降级而不是崩

跑法：
    python scripts/test_verifier.py            # 需要 DEEPSEEK_KEY
    python scripts/test_verifier.py --offline  # 只测降级路径，不花额度
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2  # noqa: E402

from src.temporal import TrackletSample, extract_features  # noqa: E402
from src.verifier import VlmVerifier, load_config  # noqa: E402

W, H = 640, 480
passed = 0
failed: list[str] = []


def check(name: str, cond: bool, extra: str = "") -> None:
    global passed
    if cond:
        passed += 1
        print(f"  [PASS] {name} {extra}")
    else:
        failed.append(name)
        print(f"  [FAIL] {name} {extra}")


def bgr(rgb: tuple) -> tuple:
    """OpenCV 用 BGR，写颜色时按 RGB 思考再翻过来，避免"橙色火苗画成蓝色"。"""
    r, g, b = rgb
    return (b, g, r)


def base_scene() -> np.ndarray:
    img = np.full((H, W, 3), (58, 62, 70), np.uint8)
    cv2.rectangle(img, (60, 340), (580, 470), (110, 105, 98), -1)   # 台面
    cv2.rectangle(img, (0, 0), (W, 60), (80, 84, 92), -1)           # 天花板
    return img


def soft_blob(img, cx, cy, r, color, alpha):
    a = np.zeros(img.shape[:2], np.float32)
    cv2.circle(a, (cx, cy), int(r), 1.0, -1)
    a = cv2.GaussianBlur(a, (0, 0), max(r * 0.4, 1.0)) * alpha
    a3 = a[..., None]
    return np.clip(img * (1 - a3) + np.array(color, np.float32) * a3, 0, 255).astype(np.uint8)


def gen(kind: str, n: int = 3) -> list[np.ndarray]:
    """生成 n 帧（时间递增），用来给 VLM 看"变化"。"""
    out = []
    for i in range(n):
        img = base_scene()
        t = i / max(n - 1, 1)
        if kind == "fire":                      # 灶台火苗：三簇橙黄 + 帧间跳动（真实火焰在跳）
            cx, cy = 320, 300
            rng = np.random.default_rng(11 + i)
            for k, (r, col) in enumerate([(46, bgr((255, 90, 10))), (30, bgr((255, 170, 30))),
                                          (16, bgr((255, 245, 200)))]):
                rr = r * (1 + 0.18 * np.sin(t * 9 + k) + rng.normal(0, 0.06))
                tilt = int(rng.normal(0, 4))
                cv2.ellipse(img, (cx + tilt, int(cy - r * 0.4)), (int(rr), int(rr * 1.7)),
                            0, 0, 360, col, -1)
            # 火星/尖端：让"根部明亮、边缘发散"更明显
            for _ in range(6):
                sx = cx + int(rng.integers(-26, 26))
                sy = cy - int(rng.integers(40, 95))
                cv2.circle(img, (sx, sy), int(rng.integers(1, 3)), bgr((255, 200, 120)), -1)
            cv2.rectangle(img, (250, 330), (390, 350), (70, 70, 75), -1)   # 灶台/炉架
        elif kind == "steam":                   # 壶口白汽：快速变粗、变淡
            for j in range(5):
                r = 22 + t * 40 + j * 6
                cy = 320 - j * 34 - t * 30
                img = soft_blob(img, 320, int(cy), r, (232, 236, 240), 0.55 - 0.08 * j)
            cv2.rectangle(img, (270, 320), (370, 360), (150, 150, 155), -1)  # 水壶
        elif kind == "smoke":                   # 灰烟：偏深、带湍流、上飘变暗（和蒸汽的主要区别）
            rng = np.random.default_rng(21 + i)
            for j in range(5):
                r = 26 + t * 20 + j * 9
                cy = 310 - j * 38 - t * 22
                cx = 330 + int(rng.normal(0, 7))       # 左右摆动 = 湍流
                # 深灰偏暖：燃烧产生的烟不是纯白
                img = soft_blob(img, cx, int(cy), r, (118, 114, 108), 0.55 - 0.06 * j)
            # 烟柱底部（靠近火源）更浓
            img = soft_blob(img, 328, 330, 34, (96, 92, 88), 0.55)
        elif kind == "lamp":                    # 吊灯：圆盘 + 细杆，位置贴天花板
            cv2.line(img, (320, 60), (320, 140), (95, 95, 100), 6)
            cv2.ellipse(img, (320, 150), (58, 26), 0, 0, 360, bgr((250, 248, 235)), -1)
            img = soft_blob(img, 320, 165, 40, bgr((255, 250, 220)), 0.35)
        out.append(img)
    return out


def fake_features(kind: str):
    """按类型造一条轨迹，喂给复核器做时序描述（模拟跟踪层输出）。"""
    rng = np.random.default_rng(3)
    dur = {"fire": 12.0, "smoke": 22.0, "steam": 10.0, "lamp": 30.0}[kind]
    n = int(dur * 5)
    s = []
    for i in range(n):
        t = i / 5
        if kind == "smoke":
            area = 900 * np.exp(0.20 * t); gray = 200 - 2.4 * t; edge = 0.28
        elif kind == "steam":
            area = 700 * np.exp(0.75 * t); gray = 210 + 1.6 * t; edge = 0.33 - 0.018 * t
        elif kind == "fire":
            area = 800 * np.exp(0.12 * t); gray = 190 - 1.0 * t; edge = 0.36
        else:
            area = 1200 * np.ones(n)[i]; gray = 240.0; edge = 0.30
        s.append(TrackletSample(t=t, area=float(area), cx=320 + rng.normal(0, 1),
                                cy=300 - 2 * t, gray_mean=float(gray), sat_mean=18.0,
                                edge_density=float(edge), conf=0.8))
    return extract_features(s)


def main() -> int:
    offline = "--offline" in sys.argv
    cfg = load_config()
    print("=" * 70)
    print(f"VLM 复核自测 | 模型 {cfg['model']} | thinking={cfg['thinking']} | "
          f"帧数 {cfg['frames']} | 限额 {cfg['hourly_limit']}/h")

    if offline:
        v = VlmVerifier(cfg, api_key="")
        r = v.verify(gen("fire"), bbox=(260, 240, 380, 340), feat=fake_features("fire"))
        check("无 Key 时降级到本地规则", r.source == "rule", r.reason)
        v2 = VlmVerifier({**cfg, "enabled": False})
        r2 = v2.verify(gen("fire"), feat=fake_features("fire"))
        check("复核关闭时跳过", r2.source == "rule", r2.raw.get("fallback_reason", ""))
        print("=" * 70)
        print(f"通过 {passed} 项，失败 {len(failed)} 项（offline 模式）")
        return 0 if not failed else 1

    v = VlmVerifier(cfg)

    # 短轨迹不送审（省额度）
    short = fake_features("fire")
    v_short = VlmVerifier({**cfg, "min_duration_s": 999})
    r_short = v_short.verify(gen("fire"), feat=short)
    check("持续不足不送审", r_short.source == "rule", r_short.raw.get("fallback_reason", ""))

    cases = [
        ("fire", "火苗（灶台）", {"fire"}, (260, 190, 380, 340)),
        ("smoke", "灰烟（上飘）", {"smoke", "other", "uncertain"}, (280, 140, 390, 320)),
        ("steam", "蒸汽（壶口）", {"steam", "other", "uncertain"}, (250, 130, 390, 360)),
        ("lamp", "吊灯", {"light", "other"}, (255, 120, 385, 200)),
    ]

    for kind, name, expect, box in cases:
        frames = gen(kind)
        feat = fake_features(kind)
        r = v.verify(frames, bbox=box, feat=feat, ctx_text="室内厂房，画面下方有操作台")
        ok = r.label in expect
        # 最重要的判据：非 fire 场景绝不能被判成 fire
        if kind != "fire" and r.label == "fire":
            ok = False
        check(f"{name}", ok,
              f"→ {r.label} ({r.confidence}) {r.source} {r.latency_s}s | {r.reason}")

    # 去重：同一 key 第二次应被跳过
    before = v.verify(gen("fire"), feat=fake_features("fire"), key="track-42")
    after = v.verify(gen("fire"), feat=fake_features("fire"), key="track-42")
    check("同一目标去重", after.source == "rule", after.raw.get("fallback_reason", ""))

    print("=" * 70)
    print(f"通过 {passed} 项，失败 {len(failed)} 项")
    if failed:
        print("失败项:", failed)
        return 1
    print("全部通过")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
