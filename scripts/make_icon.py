"""生成应用图标：深色圆角底 + 橙红火焰。

为什么要自己画：桌面快捷方式如果用的是 .bat 默认图标，一眼看不出这是什么，
双击时还容易误认。一个 16~256 全尺寸的 ico 能让快捷方式在任务栏、任务管理器、
Alt+Tab 里都能被认出来。

画法用超采样：先在 1024 画，再缩到目标尺寸，边缘才不会有锯齿。
火焰用二次贝塞尔描外轮廓 + 椭圆底，比直接画三角形像得多。
"""
import math
from pathlib import Path

from PIL import Image, ImageDraw

S = 1024          # 超采样画布边长
OUT = Path(__file__).resolve().parent / "app.ico"


def quad(p0, p1, p2, n=60):
    """二次贝塞尔采样。"""
    pts = []
    for i in range(n + 1):
        t = i / n
        u = 1 - t
        pts.append((u * u * p0[0] + 2 * u * t * p1[0] + t * t * p2[0],
                    u * u * p0[1] + 2 * u * t * p1[1] + t * t * p2[1]))
    return pts


def flame(cx, cy, w, h, lean=0.0):
    """火焰轮廓：左右两条贝塞尔 + 底部半椭圆。

    控制点必须贴近中线，顶部才会收成尖角；控制点一往外撇就变成水滴了。
    lean 让火苗略微倾斜，比正中对称更像真的在烧。
    """
    top = (cx + w * lean, cy - h * 0.52)
    left = quad((cx - w * 0.44, cy + h * 0.27),
                (cx - w * 0.50, cy - h * 0.12),
                top)
    right = quad(top,
                 (cx + w * 0.58, cy - h * 0.14),
                 (cx + w * 0.44, cy + h * 0.27))
    bottom = []
    for i in range(1, 40):
        a = math.pi * (i / 40.0)
        bottom.append((cx + w * 0.44 * math.cos(a),
                       cy + h * 0.27 + h * 0.22 * math.sin(a)))
    return left + right + bottom


img = Image.new("RGBA", (S, S), (0, 0, 0, 0))
d = ImageDraw.Draw(img)

# 外形：深色圆角方块 + 细边
pad = int(S * 0.03)
rad = int(S * 0.235)
d.rounded_rectangle([pad, pad, S - pad, S - pad], radius=rad,
                    fill=(18, 23, 32, 255), outline=(58, 70, 88, 255),
                    width=int(S * 0.018))

cx, cy = S * 0.50, S * 0.55
d.polygon(flame(cx, cy, S * 0.50, S * 0.62, lean=0.03), fill=(239, 68, 68, 255))
d.polygon(flame(cx + S * 0.012, cy + S * 0.055, S * 0.31, S * 0.40, lean=0.05),
          fill=(249, 115, 22, 255))
d.polygon(flame(cx + S * 0.022, cy + S * 0.100, S * 0.15, S * 0.22, lean=0.07),
          fill=(253, 224, 71, 255))

# 底部一道地平线，让构图稳一点
d.rounded_rectangle([S * 0.28, S * 0.845, S * 0.72, S * 0.878],
                    radius=int(S * 0.016), fill=(59, 130, 246, 210))

sizes = [256, 128, 64, 48, 32, 16]
img.resize((256, 256), Image.LANCZOS).save(
    OUT, format="ICO", sizes=[(s, s) for s in sizes])
print(f"已生成 {OUT}  ({OUT.stat().st_size} 字节, 尺寸 {sizes})")

# 顺手存一张 png 便于查看
img.resize((256, 256), Image.LANCZOS).save(OUT.with_suffix(".png"))
print(f"预览图 {OUT.with_suffix('.png')}")
