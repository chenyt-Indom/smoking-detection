"""生成火预标复核页：一个离线 HTML，键盘翻图、标记误标、导出剔除清单。

为什么做成离线 HTML 而不是起服务
--------------------------------
机器现在会不定期蓝屏（E: 那块 X800 掉盘导致 0x10133）。起服务意味着多一个进程、
多一份状态；而纯静态页面 + localStorage 能把进度留在浏览器里，**崩了也不丢**。
双击打开就能用，不需要 Python 在跑。

复核什么东西
------------
红框 = 我们的火预标（class 0），绿框 = 原数据集的烟标（class 1，已重映射）。
判断标准只有一条：**红框里到底有没有火**。
  · 有火 → 保留（哪怕框大了一点、或者一张图上很多个框，都先留着）
  · 没火（红色招牌/车漆/砖墙/夕阳/只是烟）→ 标为误标

已知失败模式（看着像但通常不是火）：
  · 消防车红色车身、红色横幅、红砖外墙
  · 黄昏的橙红色天空、夕阳照的玻璃
  · 只有烟没有焰（这种情况该保留烟标，不该有火标）
  · 同一团火被切成十几个小框 —— 这种**不算错**，先留着

导出
----
按键 `d` 下载 `prelabel_reject.txt`（每行一个 stem，含被剔除的），
直接喂给：`python scripts/build_smoke_zip_dataset.py --auto all --reject 那个文件`

跑法：
    python scripts/make_prelabel_review.py --src D:/datasets/smoke_zip_big
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def read_boxes(path: Path) -> list[list[float]]:
    out = []
    if not path.is_file():
        return out
    for ln in path.read_text(encoding="utf-8").splitlines():
        v = ln.split()
        if len(v) >= 5:
            out.append([int(float(v[0]))] + [round(float(x), 5) for x in v[1:5]])
    return out


HTML = """<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<title>火预标复核</title>
<style>
  body{margin:0;font:14px/1.5 "Microsoft YaHei",system-ui,sans-serif;background:#f6f6f4;color:#222}
  header{display:flex;align-items:center;gap:14px;padding:10px 16px;background:#fff;
         border-bottom:.5px solid #ddd;position:sticky;top:0;z-index:5}
  h1{font-size:16px;font-weight:500;margin:0}
  .bar{flex:1;height:6px;background:#e6e6e2;border-radius:3px;overflow:hidden;max-width:280px}
  .bar>i{display:block;height:100%;background:#378ADD;width:0}
  .stat{color:#666;font-size:13px}
  main{max-width:1180px;margin:0 auto;padding:16px}
  .stage{position:relative;background:#111;border-radius:10px;overflow:hidden;
         display:flex;justify-content:center}
  canvas{max-width:100%;display:block}
  .row{display:flex;gap:10px;margin-top:12px;align-items:center;flex-wrap:wrap}
  button{font:inherit;padding:9px 16px;border-radius:8px;border:.5px solid #c9c9c4;
         background:#fff;cursor:pointer}
  button:hover{background:#f0f0ec}
  .ok{border-color:#1D9E75;color:#0F6E56}
  .no{border-color:#A32D2D;color:#A32D2D}
  .k{color:#888;font-size:12px}
  #done{display:none;background:#fff;border-radius:10px;padding:24px}
  code{background:#eee;padding:2px 6px;border-radius:4px;font-size:12px}
</style></head><body>
<header>
  <h1>火预标复核</h1>
  <div class="bar"><i id="prog"></i></div>
  <span class="stat" id="stat"></span>
</header>
<main>
  <div class="stage"><canvas id="cv"></canvas></div>
  <div class="row">
    <button class="ok" id="bOk">有火，保留 (→)</button>
    <button class="no" id="bNo">不是火，剔除 (x)</button>
    <button id="bUndo">撤销 (z)</button>
    <button id="bDl">导出剔除清单 (d)</button>
    <span class="k">红=火预标　绿=原烟标　进度自动存在浏览器里，崩了也不丢</span>
  </div>
  <div id="done"><h2 style="font-size:15px">这一批看完了</h2>
    <p>点上面的「导出剔除清单」，得到 <code>prelabel_reject.txt</code>，然后：</p>
    <p><code>python scripts/build_smoke_zip_dataset.py --auto all --reject 该文件</code></p>
    <button id="bReset">重新开始（清空进度）</button>
  </div>
</main>
<script>
const ITEMS = __ITEMS__;
const KEY = 'fire_prelabel_review_v1';
let idx = 0, verdict = {};
try { const s = JSON.parse(localStorage.getItem(KEY) || '{}');
      verdict = s.verdict || {}; idx = s.idx || 0; } catch(e){}
if (idx >= ITEMS.length) idx = 0;

const cv = document.getElementById('cv'), ctx = cv.getContext('2d');
const img = new Image();

function save(){ localStorage.setItem(KEY, JSON.stringify({verdict, idx})); }

function paint(){
  const it = ITEMS[idx];
  cv.width = img.naturalWidth; cv.height = img.naturalHeight;
  const W = img.naturalWidth, H = img.naturalHeight;
  ctx.drawImage(img, 0, 0);
  const scale = Math.max(2, W / 500);
  for (const b of it.boxes){
    const [c, cx, cy, bw, bh] = b;
    const x = (cx - bw / 2) * W, y = (cy - bh / 2) * H;
    ctx.lineWidth = scale;
    ctx.strokeStyle = c === 0 ? '#ff3b30' : '#22c55e';
    ctx.strokeRect(x, y, bw * W, bh * H);
  }
  const nFire = it.boxes.filter(b => b[0] === 0).length;
  const nSmoke = it.boxes.filter(b => b[0] === 1).length;
  document.getElementById('stat').textContent =
    (idx + 1) + ' / ' + ITEMS.length + '　' + it.stem +
    '　火框 ' + nFire + '　烟框 ' + nSmoke;
  const done = Object.keys(verdict).length;
  document.getElementById('prog').style.width = (done / ITEMS.length * 100) + '%';
}

function load(){
  const it = ITEMS[idx];
  const want = 'images/' + it.stem + '.jpg';
  // 同一张图再读一次时浏览器不会再触发 onload（缓存命中），必须自己重绘，
  // 否则「撤销」回到上一张会卡在灰屏上。
  if (img.getAttribute('src') === want && img.complete) { paint(); return; }
  paint_dummy();
  img.onload = paint;
  img.src = want;
}
function paint_dummy(){ ctx.fillStyle = '#111'; cv.width = 900; cv.height = 500;
  ctx.fillRect(0, 0, 900, 500); }

function mark(v){
  verdict[ITEMS[idx].stem] = v;
  idx = Math.min(idx + 1, ITEMS.length - 1);
  save();
  if (Object.keys(verdict).length >= ITEMS.length){
    document.getElementById('done').style.display = 'block';
  }
  load();
}

document.getElementById('bOk').onclick = () => mark('ok');
document.getElementById('bNo').onclick = () => mark('reject');
document.getElementById('bUndo').onclick = () => {
  idx = Math.max(0, idx - 1); delete verdict[ITEMS[idx].stem]; save(); load();
};
document.getElementById('bReset').onclick = () => {
  if (!confirm('清空所有复核进度？')) return;
  verdict = {}; idx = 0; localStorage.removeItem(KEY); load();
};
document.getElementById('bDl').onclick = () => {
  const bad = Object.keys(verdict).filter(k => verdict[k] === 'reject');
  const blob = new Blob([bad.join('\\n') + '\\n'], {type:'text/plain'});
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = 'prelabel_reject.txt';
  a.click();
};

addEventListener('keydown', e => {
  if (e.key === 'ArrowRight' || e.key === 'k') mark('ok');
  else if (e.key === 'x' || e.key === 'X') mark('reject');
  else if (e.key === 'z' || e.key === 'Z') document.getElementById('bUndo').click();
  else if (e.key === 'd' || e.key === 'D') document.getElementById('bDl').click();
});

load();
</script></body></html>
"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="D:/datasets/smoke_zip_big")
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    src = Path(a.src)
    items = []
    for p in sorted((src / "labels_auto").glob("*.txt")):
        auto = read_boxes(p)
        if not auto:                       # 没有火预标的图不用复核
            continue
        smoke = read_boxes(src / "labels" / p.name)
        items.append({"stem": p.stem, "boxes": auto + smoke})
    if not items:
        raise SystemExit(f"没找到带火预标的图：{src / 'labels_auto'}")

    html = HTML.replace("__ITEMS__", json.dumps(items, ensure_ascii=False))
    out = Path(a.out) if a.out else src / "review.html"
    out.write_text(html, encoding="utf-8")
    n_box = sum(len(i["boxes"]) for i in items)
    print(f"待复核 {len(items)} 张，{n_box} 个框")
    print(f"复核页 {out}")
    print("双击用浏览器打开；红=火预标 绿=原烟标；→保留 / x剔除 / z撤销 / d导出")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
