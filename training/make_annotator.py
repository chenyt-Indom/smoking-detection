# -*- coding: utf-8 -*-
"""
make_annotator.py — 图像标注工具生成器

把一个目录(或文件列表)里的图片打包成一个"自包含"HTML 标注页:
- 图片以 base64 内嵌, 单文件可用(无需服务器)
- 拖拽画框标注正样本 / 一键标负样本 / 删除标注
- 导出 JSON, 格式: {"path": {"type": "pos|neg", "boxes": [[cx,cy,w,h], ...]}}

用法:
  python make_annotator.py --images "D:/images" --out annotate.html --label "批次1"
  python make_annotator.py --files a.jpg b.jpg c.jpg --out annotate.html --label "demo"

可选:
  --img-ext jpg,png   图片扩展名(默认 jpg)
  --json-name cap25.json  导出文件名(默认 out 同名 .json)
"""
import argparse, base64, json, os, re, sys

def collect_images(path, exts):
    if os.path.isdir(path):
        files = []
        for root, _, names in os.walk(path):
            for n in sorted(names):
                if n.lower().endswith(tuple('.' + e.lower() for e in exts)):
                    files.append(os.path.join(root, n))
        return files
    return [path]

def main():
    ap = argparse.ArgumentParser(description='Generate a self-contained annotation page')
    ap.add_argument('--images', help='image directory')
    ap.add_argument('--files', nargs='*', help='explicit image files')
    ap.add_argument('--out', required=True, help='output html path')
    ap.add_argument('--label', default='标注批次', help='batch label shown in title')
    ap.add_argument('--img-ext', default='jpg', help='extensions, comma separated')
    ap.add_argument('--json-name', default=None, help='export json filename')
    args = ap.parse_args()

    if args.files:
        files = args.files
    elif args.images:
        files = collect_images(args.images, args.img_ext.split(','))
    else:
        sys.exit('need --images or --files')

    if not files:
        sys.exit('no images found')

    # 相对路径作为键(相对当前目录, 或仅文件名)
    keys = []
    for f in files:
        try:
            rel = os.path.relpath(f, os.getcwd()).replace('\\', '/')
            keys.append(rel if not rel.startswith('..') else os.path.basename(f))
        except ValueError:
            keys.append(os.path.basename(f))

    imgs = {}
    for k, f in zip(keys, files):
        with open(f, 'rb') as fp:
            b64 = base64.b64encode(fp.read()).decode()
        ext = os.path.splitext(f)[1].lstrip('.').lower()
        mime = 'png' if ext == 'png' else 'jpeg'
        imgs[k] = f'data:image/{mime};base64,{b64}'

    # 模板(与仓库内 annotator_template.html 保持一致逻辑, 这里内联一个干净版本)
    batch_id = re.sub(r'[^a-zA-Z0-9_\-]', '_', os.path.splitext(os.path.basename(args.out))[0])
    json_name = args.json_name or (batch_id + '.json')

    html = r'''<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="UTF-8">
<title>__LABEL__ 标注页</title>
<style>
  body { font-family: "Microsoft YaHei"; background:#1e1e1e; color:#eee; margin:0; padding:16px; }
  h3 { margin: 4px 0; }
  .box { margin-bottom: 22px; border:1px solid #555; padding:12px; background:#252525; }
  .wrap { position:relative; display:inline-block; max-width:100%; }
  .wrap img { max-width:100%; display:block; cursor:crosshair; }
  .wrap canvas { position:absolute; left:0; top:0; pointer-events:none; max-width:100%; }
  .status { font-size:13px; margin-top:4px; }
  .ok { color:#30a46c; font-weight:bold; }
  .no { color:#e5484d; font-weight:bold; }
  .btn { background:#0a84ff; color:#fff; border:none; padding:6px 14px; border-radius:4px;
         cursor:pointer; font-size:13px; margin:4px 4px 0 0; }
  .btn.green { background:#30a46c; }
  .btn.red { background:#e5484d; }
  #nav { position:fixed; bottom:12px; right:16px; background:#333; padding:8px 14px; border-radius:8px; z-index:99; }
  #counter { position:fixed; top:8px; right:14px; background:#333; padding:6px 12px; border-radius:6px; font-size:13px; }
</style>
</head>
<body>
<div id="counter">已标 0/__N__</div>
<h3>__LABEL__: 拖拽画框(从烟左上角拖到右下角) 或 点"这张没烟"</h3>
<div style="color:#aaa; font-size:13px; margin-bottom:14px;">图片已内嵌,无需服务器,直接在这里操作即可</div>

<div id="frames"></div>

<div id="nav">
  <button class="btn" style="font-size:14px;" onclick="exportJSON()">📤 导出标注</button>
</div>

<script>
const IMGS = __IMGS__;
</script>
<script>
const FRAMES = Object.keys(IMGS);
const SAVE_KEY = '__BATCHID__';
const EXPORT_NAME = '__JSONNAME__';
const state = {};
try { const s = localStorage.getItem(SAVE_KEY); if (s) Object.assign(state, JSON.parse(s)); } catch(e){}
const container = document.getElementById('frames');

FRAMES.forEach(fname => {
  const box = document.createElement('div');
  box.className = 'box';
  box.innerHTML = `
    <h3>${fname}</h3>
    <div class="wrap"><img src="${IMGS[fname]}"></div>
    <div class="status">未标注</div>
    <button class="btn green" onclick="markNo('${fname}')">这张没烟</button>
    <button class="btn red" onclick="delMark('${fname}')">🗑 删除标注</button>
  `;
  container.appendChild(box);
  setupDrag(box, fname);
});

function setupDrag(box, fname) {
  const img = box.querySelector('img');
  const wrap = box.querySelector('.wrap');
  let drag = null;
  img.addEventListener('mousedown', e => {
    const rect = img.getBoundingClientRect();
    drag = {sx:(e.clientX-rect.left)/rect.width, sy:(e.clientY-rect.top)/rect.height};
    e.preventDefault();
  });
  window.addEventListener('mousemove', e => {
    if (!drag) return;
    const rect = img.getBoundingClientRect();
    draw(box, fname, drag.sx, drag.sy, (e.clientX-rect.left)/rect.width, (e.clientY-rect.top)/rect.height);
  });
  window.addEventListener('mouseup', e => {
    if (!drag) return;
    const rect = img.getBoundingClientRect();
    const x2 = (e.clientX-rect.left)/rect.width, y2 = (e.clientY-rect.top)/rect.height;
    if (Math.abs(x2-drag.sx) < 0.02 && Math.abs(y2-drag.sy) < 0.02) { drag = null; return; }
    if (!state[fname] || state[fname].type !== 'pos') state[fname] = {type:'pos', boxes:[]};
    state[fname].boxes.push({x1:drag.sx, y1:drag.sy, x2:x2, y2:y2});
    save(); updateStatus(box, fname); draw(box, fname, 0,0,0,0);
    drag = null;
  });
}

function draw(box, fname, x1,y1,x2,y2) {
  const img = box.querySelector('img');
  const wrap = box.querySelector('.wrap');
  let canvas = wrap.querySelector('canvas');
  if (!canvas) { canvas = document.createElement('canvas'); wrap.appendChild(canvas); }
  canvas.width = img.naturalWidth; canvas.height = img.naturalHeight;
  const ctx = canvas.getContext('2d');
  ctx.clearRect(0,0,canvas.width,canvas.height);
  const s = state[fname];
  if (s && s.type==='pos') {
    s.boxes.forEach(b => {
      ctx.strokeStyle='#30a46c'; ctx.lineWidth=3;
      ctx.strokeRect(b.x1*canvas.width, b.y1*canvas.height, (b.x2-b.x1)*canvas.width, (b.y2-b.y1)*canvas.height);
    });
  }
  if (x2>0) {
    ctx.strokeStyle='#ffb800'; ctx.lineWidth=3; ctx.setLineDash([6,4]);
    ctx.strokeRect(Math.min(x1,x2)*canvas.width, Math.min(y1,y2)*canvas.height,
                   Math.abs(x2-x1)*canvas.width, Math.abs(y2-y1)*canvas.height);
    ctx.setLineDash([]);
  }
}

function updateStatus(box, fname) {
  const el = box.querySelector('.status');
  const s = state[fname];
  if (!s) { el.textContent='未标注'; el.className='status'; }
  else if (s.type==='neg') { el.innerHTML='<span class="no">✗ 无烟(负样本)</span>'; }
  else { el.innerHTML = `<span class="ok">✓ 有烟 ${s.boxes.length}个框</span>`; }
  document.getElementById('counter').textContent = `已标 ${Object.keys(state).length}/${FRAMES.length}`;
}

function markNo(fname) {
  state[fname] = {type:'neg', boxes:[]};
  save(); updateStatus(container.querySelectorAll('.box')[[...FRAMES].indexOf(fname)], fname);
  const wrap = container.querySelectorAll('.box')[[...FRAMES].indexOf(fname)].querySelector('.wrap');
  const c = wrap.querySelector('canvas'); if (c) c.remove();
}

function delMark(fname) {
  delete state[fname];
  save();
  const box = container.querySelectorAll('.box')[[...FRAMES].indexOf(fname)];
  updateStatus(box, fname);
  const c = box.querySelector('.wrap canvas'); if (c) c.remove();
}

function save() { try { localStorage.setItem(SAVE_KEY, JSON.stringify(state)); } catch(e){} }

function exportJSON() {
  const out = {};
  for (const fname of FRAMES) {
    const s = state[fname];
    if (!s) continue;
    if (s.type === 'neg') { out[fname] = {type:'neg', boxes:[]}; }
    else {
      const boxes = s.boxes.map(b => [
        (b.x1+b.x2)/2, (b.y1+b.y2)/2, Math.abs(b.x2-b.x1), Math.abs(b.y2-b.y1)
      ]);
      out[fname] = {type:'pos', boxes};
    }
  }
  const blob = new Blob([JSON.stringify(out, null, 1)], {type:'application/json'});
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob); a.download = EXPORT_NAME; a.click();
  alert(`已导出 ${Object.keys(out).length}/${FRAMES.length} 帧 → ${EXPORT_NAME} (在下载目录)`);
}

FRAMES.forEach(fname => {
  const boxes = container.querySelectorAll('.box');
  const idx = FRAMES.indexOf(fname);
  updateStatus(boxes[idx], fname);
  if (state[fname] && state[fname].type==='pos') draw(boxes[idx], fname, 0,0,0,0);
});
</script>
</body>
</html>
'''

    html = (html.replace('__LABEL__', args.label)
                .replace('__N__', str(len(files)))
                .replace('__IMGS__', json.dumps(imgs, ensure_ascii=False))
                .replace('__BATCHID__', batch_id)
                .replace('__JSONNAME__', json_name))

    with open(args.out, 'w', encoding='utf-8') as f:
        f.write(html)
    print(f'✅ 已生成 {args.out}  ({len(files)} 帧, {os.path.getsize(args.out)/1024:.0f} KB)')

if __name__ == '__main__':
    main()
