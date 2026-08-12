# -*- coding: utf-8 -*-
"""
web_server.py — 视觉安防系统 Web 前端 (纯标准库, 零依赖)
以线程方式运行在 head_tracker_v12.py 进程内, 端口 5050
功能:
  /            主界面(实时视频流 + 头/手ID状态 + 物品事件库 + 识别结果)
  /video_feed  MJPEG 实时视频流
  /api/events  物品事件列表(原图/超分/香烟识别结果) JSON
  /api/stats   统计信息 JSON
  /caps/<file> 截图文件(原始帧/超分图)
"""
import json, os, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import urllib.parse

# ===== 运行时由 head_tracker_v12.py 注入 =====
WEB_QUEUE = None     # 帧队列(每帧 jpeg bytes)
OBJ_DB = None        # ObjectDB 实例
CAP_DIR = None       # 截图目录

PORT = 5050

# 1x1 黑色占位图(无帧时显示)
_PLACEHOLDER_JPEG = bytes.fromhex(
    'FFD8FFE000104A46494600010100000100010000FFDB004300080606070605080707070909080A0C'
    '140D0C0B0B0C1912130F141D1A1F1E1D1A1C1C20242E2720222C231C1C2837292C30313434341F'
    '27393D38323C2E333432FFC0000B080001000101011100FFC4001F00000105010101010101000000'
    '00000000000102030405060708090A0BFFC400B5100002010303020403050504040000017D010203'
    '00041105122131410613516107227114328191A1082342B1C11552D1F02433627282090A16171819'
    '1A25262728292A3435363738393A434445464748494A535455565758595A636465666768696A7374'
    '75767778797A838485868788898A92939495969798999AA2A3A4A5A6A7A8A9AAB2B3B4B5B6B7B8B9'
    'BAC2C3C4C5C6C7C8C9CAD2D3D4D5D6D7D8D9DAE1E2E3E4E5E6E7E8E9EAF1F2F3F4F5F6F7F8F9FA'
    'FFC4001F0100030101010101010101010000000000000102030405060708090A0BFFC400B5110002'
    '0102040403040705040400010277000102031104052131061241510761711322328108144291A1B1'
    'C109233352F0156272D10A162434E125F11718191A262728292A35363738393A434445464748494A'
    '535455565758595A636465666768696A737475767778797A82838485868788898A92939495969798'
    '999AA2A3A4A5A6A7A8A9AAB2B3B4B5B6B7B8B9BAC2C3C4C5C6C7C8C9CAD2D3D4D5D6D7D8D9DAE2'
    'E3E4E5E6E7E8E9EAF2F3F4F5F6F7F8F9FAFFDA000C03010002110311003F0003F80000FFD9')

INDEX_HTML = """<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="utf-8">
<title>视觉安防系统 · 物品检测全流程看板</title>
<style>
:root{--bg:#0f1420;--card:#1a2130;--line:#2a3550;--txt:#e8ecf4;--dim:#8a94a8;--red:#ff4d5e;--grn:#2ecc71;--blu:#4da3ff;--yel:#ffc933}
*{box-sizing:border-box;margin:0;padding:0}
body{background:var(--bg);color:var(--txt);font-family:"Microsoft YaHei",system-ui,sans-serif;padding:16px}
h1{font-size:20px;margin-bottom:12px}
h1 span{color:var(--dim);font-size:13px;font-weight:normal;margin-left:10px}
.row{display:flex;gap:14px;flex-wrap:wrap;align-items:flex-start}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px;margin-bottom:14px}
.video-box{width:640px;max-width:100%}
.video-box img{width:100%;border-radius:8px;display:block;background:#000}
.badges{display:flex;gap:10px;flex-wrap:wrap;margin:10px 0}
.badge{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:8px 14px;font-size:13px}
.badge b{font-size:18px;display:block}
.b-red{color:var(--red)} .b-grn{color:var(--grn)} .b-yel{color:var(--yel)} .b-blu{color:var(--blu)}
table{width:100%;border-collapse:collapse;font-size:13px}
th,td{padding:7px 8px;text-align:left;border-bottom:1px solid var(--line)}
th{color:var(--dim);font-weight:normal;font-size:12px}
td img{width:110px;height:78px;object-fit:cover;border-radius:5px;display:block;background:#000}
.t-red{color:var(--red);font-weight:bold}.t-grn{color:var(--grn);font-weight:bold}.t-dim{color:var(--dim)}
.tag{display:inline-block;padding:1px 8px;border-radius:10px;font-size:11px;margin-right:4px}
.tag-h{background:#2a3550;color:var(--blu)}.tag-d{background:#2a3550;color:var(--yel)}
.legend{font-size:12px;color:var(--dim);margin-top:8px;line-height:1.7}
.legend .dot{display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:4px;vertical-align:middle}
.empty{padding:24px;text-align:center;color:var(--dim)}
@keyframes blink{50%{opacity:.35}}
.live{color:var(--grn);font-weight:bold;animation:blink 1.2s infinite}
</style>
</head>
<body>
<h1>🛡 视觉安防系统 · 物品检测全流程<span>实时检测 → 最清晰帧截图 → 超分 → 香烟识别</span></h1>

<div class="row">
  <div class="card video-box">
    <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px">
      <b>实时画面</b><span class="live" id="liveTag">● 直播中</span>
    </div>
    <img src="/video_feed" alt="实时视频流">
    <div class="legend">
      <span><span class="dot" style="background:#2ecc71"></span>头框(无物)</span>
      <span><span class="dot" style="background:#4da3ff"></span>手框(空手)</span>
      <span><span class="dot" style="background:#ff4d5e"></span>检测到物品(嘴含物/手持物)</span>
    </div>
  </div>

  <div style="flex:1;min-width:300px">
    <div class="card">
      <b>实时状态</b>
      <div class="badges">
        <div class="badge"><b id="stTotal">0</b>已采集ID<span class="b-blu" style="margin-left:4px;font-size:11px">(头+手)</span></div>
        <div class="badge"><b id="stDone">0</b>已识别</div>
        <div class="badge"><b id="stSmoke" class="b-red">0</b>检出香烟</div>
      </div>
      <div class="legend">流程: 头/手框检出非人体物品 → 框变红 → 每ID保存最清晰一帧 → 后台Real-ESRGAN超分 → 香烟模型识别静态图</div>
    </div>
    <div class="card">
      <b>流程说明</b>
      <div class="legend">
        1️⃣ <b>实时层</b>: 完全摒弃实时香烟检测。只在头框嘴部/手框内检测<b>非肤色物品块</b>(烟/吸管/杯子/手机等), 有物 → 框变红<br>
        2️⃣ <b>截图层</b>: 每个头/手ID只保留<b>最清晰一帧</b>(Laplacian评分), 存入数据库<br>
        3️⃣ <b>识别层</b>: 后台对截图做<b>Real-ESRGAN超分</b>清晰化, 再用香烟模型识别静态图 → 结果写回
      </div>
    </div>
  </div>
</div>

<div class="card">
  <b>物品事件库 <span id="evCount" class="t-dim" style="font-size:12px"></span></b>
  <table>
    <thead><tr><th>ID</th><th>截图(最清晰帧)</th><th>超分后</th><th>清晰度</th><th>香烟识别</th><th>时间</th></tr></thead>
    <tbody id="evBody"><tr><td colspan="6" class="empty">等待采集...</td></tr></tbody>
  </table>
</div>

<script>
function fmtTs(t){if(!t)return'-';var d=new Date(t*1000);return d.toLocaleString('zh-CN',{hour12:false})}
function esc(s){return String(s==null?'':s).replace(/[&<>"]/g,function(c){return{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]})}
async function loadStats(){
  try{
    var r=await fetch('/api/stats');var s=await r.json();
    document.getElementById('stTotal').textContent=s.total||0;
    document.getElementById('stDone').textContent=s.processed||0;
    document.getElementById('stSmoke').textContent=s.has_smoke||0;
  }catch(e){}
}
async function loadEvents(){
  try{
    var r=await fetch('/api/events');var ev=await r.json();
    document.getElementById('evCount').textContent='('+ev.length+'条)';
    var body=document.getElementById('evBody');
    if(!ev.length){body.innerHTML='<tr><td colspan="6" class="empty">暂无采集, 请让手进入画面或做出吸烟动作</td></tr>';return}
    body.innerHTML=ev.map(function(e){
      var tag=e.obj_type==='head'?'<span class="tag tag-h">头</span>':'<span class="tag tag-d">手</span>';
      var cap='/caps/'+encodeURIComponent(e.cap_path.split(/[\\\\/]/).pop());
      var sup=e.super_path?'/caps/'+encodeURIComponent(e.super_path.split(/[\\\\/]/).pop()):'';
      var res='<span class="t-dim">待识别</span>';
      if(e.processed){res=e.smoke_result==1?'<span class="t-red">🚬 检出香烟 (conf '+Number(e.smoke_conf).toFixed(2)+')</span>':'<span class="t-grn">无</span>'}
      return '<tr><td>'+tag+e.obj_id+'</td>'
        +'<td><img src="'+cap+'?v='+e.id+'" title="清晰度'+Number(e.clarity).toFixed(0)+'"></td>'
        +'<td>'+(sup?'<img src="'+sup+'?v='+e.id+'">':'<span class="t-dim">-</span>')+'</td>'
        +'<td>'+Number(e.clarity).toFixed(0)+'</td>'
        +'<td>'+res+'</td>'
        +'<td class="t-dim">'+fmtTs(e.ts)+'</td></tr>';
    }).join('');
  }catch(e){}
}
loadStats();loadEvents();
setInterval(loadStats,2000);setInterval(loadEvents,3000);
</script>
</body>
</html>
"""


class Handler(BaseHTTPRequestHandler):
    protocol_version = 'HTTP/1.1'

    def log_message(self, *a):
        pass

    def _send_bytes(self, data, ctype, status=200):
        try:
            self.send_response(status)
            self.send_header('Content-Type', ctype)
            self.send_header('Content-Length', str(len(data)))
            self.send_header('Cache-Control', 'no-store')
            self.end_headers()
            self.wfile.write(data)
        except Exception:
            pass

    def do_GET(self):
        try:
            parsed = urllib.parse.urlparse(self.path)
            path = parsed.path
            if path in ('/', '/index.html'):
                self._send_bytes(INDEX_HTML.encode('utf-8'), 'text/html; charset=utf-8')
            elif path == '/video_feed':
                self._stream_mjpeg()
            elif path == '/api/events':
                data = OBJ_DB.get_all_events(100) if OBJ_DB else []
                self._send_bytes(json.dumps(data, ensure_ascii=False).encode('utf-8'), 'application/json')
            elif path == '/api/stats':
                stats = OBJ_DB.get_stats() if OBJ_DB else {}
                stats['queue'] = WEB_QUEUE.qsize() if WEB_QUEUE else -1
                self._send_bytes(json.dumps(stats, ensure_ascii=False).encode('utf-8'), 'application/json')
            elif path.startswith('/caps/'):
                fname = os.path.basename(urllib.parse.unquote(path))
                fp = os.path.join(CAP_DIR or '', fname)
                if os.path.exists(fp):
                    with open(fp, 'rb') as f:
                        self._send_bytes(f.read(), 'image/jpeg')
                else:
                    self._send_bytes(b'not found', 'text/plain', 404)
            else:
                self._send_bytes(b'404', 'text/plain', 404)
        except Exception:
            try:
                self._send_bytes(b'err', 'text/plain', 500)
            except Exception:
                pass

    def _stream_mjpeg(self):
        """MJPEG 流(长连接): 从队列取帧, 超时发占位图保持连接"""
        try:
            self.send_response(200)
            self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=frame')
            self.send_header('Cache-Control', 'no-store')
            self.end_headers()
            while True:
                try:
                    jpg = WEB_QUEUE.get(timeout=3.0)
                except Exception:
                    jpg = None
                if jpg is None:
                    jpg = _PLACEHOLDER_JPEG
                try:
                    self.wfile.write(b'--frame\r\nContent-Type: image/jpeg\r\n'
                                     b'Content-Length: ' + str(len(jpg)).encode() + b'\r\n\r\n')
                    self.wfile.write(jpg)
                    self.wfile.write(b'\r\n')
                    self.wfile.flush()
                except Exception:
                    break
        except Exception:
            pass


def start_web_server(queue, obj_db, cap_dir, port=PORT):
    """在独立线程启动 HTTP 服务(非阻塞)"""
    global WEB_QUEUE, OBJ_DB, CAP_DIR, PORT
    WEB_QUEUE, OBJ_DB, CAP_DIR, PORT = queue, obj_db, cap_dir, port
    srv = ThreadingHTTPServer(('0.0.0.0', port), Handler)
    import threading
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    print(f"🌐 Web 前端已启动: http://localhost:{port}")
    return srv
