#!/usr/bin/env python3
"""
page_FlitGame - 样本筛选页面（模块）
提供后端函数和 HTML 模板，供 page_trainer 导入使用。
也可独立运行（python -m WebUI.page_flit_game）。
"""
import argparse
import base64
import json
import time
import cv2
import numpy as np
from pathlib import Path
from http.server import BaseHTTPRequestHandler, HTTPServer
from socketserver import ThreadingMixIn
from urllib.parse import urlparse, parse_qs

# ------------------------------------------------------------------
# 后端逻辑（公开接口，page_trainer 导入使用）
# ------------------------------------------------------------------

WORKSPACE = 'workspace'


def scan_aligned_dirs(workspace=None):
    """扫描 workspace 下所有 aligned 目录，返回 [{path, name, count}]"""
    ws = Path(workspace or WORKSPACE).resolve()
    dirs = []
    if not ws.exists():
        return dirs
    for p in ws.rglob('aligned'):
        if p.is_dir() and p.parent.parent == ws:
            files = sorted([f for f in p.iterdir()
                           if f.suffix.lower() in ('.jpg', '.jpeg', '.png')])
            dirs.append({
                'path': str(p),
                'name': str(p.relative_to(ws)),
                'count': len(files),
            })
    return dirs


def list_samples(dir_path, offset, count):
    """返回从 offset 开始的 count 个样本（base64 512x512 JPEG）。"""
    p = Path(dir_path)
    files = sorted([f for f in p.iterdir()
                    if f.suffix.lower() in ('.jpg', '.jpeg', '.png')])
    total = len(files)
    selected = files[offset:offset + count]

    samples = []
    for f in selected:
        try:
            img_array = np.fromfile(str(f), dtype=np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            if img is None:
                continue
            if img.shape[0] != 512 or img.shape[1] != 512:
                img = cv2.resize(img, (512, 512))
            ok, jpg = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if ok:
                samples.append({
                    'filename': f.name,
                    'path': str(f),
                    'data': base64.b64encode(jpg.tobytes()).decode(),
                })
        except Exception:
            continue

    return samples, total


def move_to_trash(file_paths):
    """移动文件到 aligned_trash 目录，返回成功移动的数量。"""
    moved = 0
    for fp_str in file_paths:
        fp = Path(fp_str)
        if not fp.exists():
            continue
        trash_dir = fp.parent.parent / (fp.parent.name + '_trash')
        trash_dir.mkdir(parents=True, exist_ok=True)
        dst = trash_dir / fp.name
        try:
            fp.rename(dst)
            moved += 1
        except Exception:
            pass
    return moved


def count_files(dir_path):
    """统计目录中图片文件总数。"""
    p = Path(dir_path)
    if not p.exists():
        return 0
    return len([f for f in p.iterdir()
                if f.suffix.lower() in ('.jpg', '.jpeg', '.png')])


# ------------------------------------------------------------------
# HTML / CSS / JS 前端
# ------------------------------------------------------------------

HTML = r"""<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>DFL 样本筛选</title>
	<style>
		@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;450;500;550;600&display=swap');
		*{box-sizing:border-box;margin:0;padding:0}
		::-webkit-scrollbar{width:6px;height:6px}
		::-webkit-scrollbar-track{background:transparent}
		::-webkit-scrollbar-thumb{background:rgba(255,255,255,.08);border-radius:3px;transition:background .15s}
		::-webkit-scrollbar-thumb:hover{background:rgba(255,255,255,.14)}
		*{scrollbar-width:thin;scrollbar-color:rgba(255,255,255,.08) transparent}
		::selection{background:rgba(91,91,214,.35);color:#fff}
		body{background:#0a0a0b;color:rgba(255,255,255,.8);font-family:'Inter',-apple-system,sans-serif;font-size:13px;font-weight:450;height:100vh;overflow:hidden;display:flex;flex-direction:column;-webkit-font-smoothing:antialiased}
		header{display:flex;align-items:center;gap:10px;height:40px;padding:0 14px;background:#0d0d0e;border-bottom:1px solid rgba(255,255,255,.06);flex-shrink:0}
		header select,header input,header button{font-size:11px;font-family:inherit}
		header label{display:flex;align-items:center;gap:4px;font-size:11px;color:rgba(255,255,255,.35)}
		.back-btn{display:inline-flex;align-items:center;gap:4px;padding:4px 10px;background:rgba(255,255,255,.04);border:1px solid rgba(255,255,255,.06);border-radius:5px;color:rgba(255,255,255,.45);text-decoration:none;font-size:11px;font-weight:500;transition:all .12s}
		.back-btn:hover{background:rgba(255,255,255,.07);color:rgba(255,255,255,.65)}
		select{padding:3px 8px;background:#0d0d0e;border:1px solid rgba(255,255,255,.08);border-radius:5px;color:rgba(255,255,255,.65);font-size:12px;font-family:inherit;outline:none;transition:border-color .12s}
		select:focus{border-color:rgba(91,91,214,.4)}
		input[type=number]{padding:3px 6px;width:56px;background:#0d0d0e;border:1px solid rgba(255,255,255,.08);border-radius:5px;color:rgba(255,255,255,.65);font-size:12px;font-family:inherit;outline:none;transition:border-color .12s;text-align:center}
		input[type=number]:focus{border-color:rgba(91,91,214,.4)}
		button{padding:4px 10px;background:rgba(255,255,255,.04);border:1px solid rgba(255,255,255,.08);border-radius:5px;color:rgba(255,255,255,.55);cursor:pointer;font-size:11px;font-family:inherit;font-weight:500;transition:all .12s}
		button:hover{background:rgba(255,255,255,.07);color:rgba(255,255,255,.75)}
		button:disabled{opacity:.3;cursor:default;background:rgba(255,255,255,.02)}
		#gallery{flex:1;overflow-y:auto;display:flex;flex-wrap:wrap;gap:10px;padding:14px;align-content:flex-start}
		.img-card{position:relative;width:calc(50% - 5px);border-radius:8px;overflow:hidden;border:2px solid transparent;cursor:pointer;transition:all .15s;background:#0d0d0e}
		.img-card:hover{border-color:rgba(255,255,255,.08)}
		.img-card.marked{border-color:#5b5bd6;box-shadow:0 0 12px rgba(91,91,214,.15)}
		.img-card img{width:100%;height:auto;display:block}
		.img-card .overlay{position:absolute;inset:0;background:rgba(91,91,214,.25);backdrop-filter:blur(2px);-webkit-backdrop-filter:blur(2px);display:none;align-items:center;justify-content:center}
		.img-card.marked .overlay{display:flex}
		.img-card .overlay span{color:#fff;font-size:13px;font-weight:500;background:rgba(0,0,0,.5);padding:4px 12px;border-radius:5px}
		.img-card .label{position:absolute;bottom:0;left:0;right:0;padding:4px 8px;background:linear-gradient(transparent,rgba(0,0,0,.7));color:rgba(255,255,255,.7);font-size:10px;font-weight:500;pointer-events:none}
		.empty-msg{width:100%;text-align:center;padding:40px 20px;color:rgba(255,255,255,.2);font-size:13px}
		#footer{display:flex;align-items:center;gap:12px;height:36px;padding:0 14px;background:#0d0d0e;border-top:1px solid rgba(255,255,255,.06);flex-shrink:0}
		#stats{display:flex;gap:16px;font-size:11px;color:rgba(255,255,255,.35)}
		#stats b{color:rgba(255,255,255,.6);font-weight:500}
		#settle-btn{padding:5px 14px;background:linear-gradient(135deg,#5b5bd6,#8b5cf6);border:none;border-radius:5px;color:#fff;font-size:11px;font-weight:500;cursor:pointer;margin-left:auto;transition:opacity .12s}
		#settle-btn:hover{opacity:.9;box-shadow:0 2px 8px rgba(91,91,214,.25)}
		#settle-btn:disabled{opacity:.3;cursor:default;box-shadow:none}
		.modal-overlay{position:fixed;inset:0;background:rgba(0,0,0,.6);backdrop-filter:blur(4px);-webkit-backdrop-filter:blur(4px);z-index:1000;display:none;align-items:center;justify-content:center}
		.modal-overlay.open{display:flex}
		.modal-box{min-width:320px;max-width:420px;background:#0d0d0e;border:1px solid rgba(255,255,255,.08);border-radius:10px;box-shadow:0 16px 48px rgba(0,0,0,.5);padding:20px;text-align:center}
		.modal-box h2{font-size:14px;font-weight:550;color:rgba(255,255,255,.8);margin-bottom:8px}
		.modal-box p{font-size:12px;color:rgba(255,255,255,.45);margin-bottom:16px}
		.modal-box input{width:100%;padding:8px 12px;background:rgba(255,255,255,.02);border:1px solid rgba(255,255,255,.08);border-radius:6px;color:rgba(255,255,255,.75);font-size:13px;font-family:inherit;text-align:center;outline:none;transition:border-color .12s}
		.modal-box input:focus{border-color:rgba(91,91,214,.4);box-shadow:0 0 0 2px rgba(91,91,214,.08)}
		.modal-box .modal-err{font-size:12px;color:#ef4444;margin-top:8px;display:none}
		.modal-box .modal-actions{display:flex;gap:8px;margin-top:16px}
		.modal-box .modal-actions button{flex:1;padding:7px 12px;border-radius:6px;cursor:pointer;font-size:12px;font-weight:500;font-family:inherit;border:none;transition:opacity .12s}
		.btn-apply{background:linear-gradient(135deg,#5b5bd6,#8b5cf6)!important;color:#fff!important}
		.btn-apply:hover{opacity:.9!important;box-shadow:0 2px 8px rgba(91,91,214,.25)!important}
		.btn-reset{background:rgba(255,255,255,.04)!important;color:rgba(255,255,255,.45)!important}
		.btn-reset:hover{background:rgba(255,255,255,.07)!important;color:rgba(255,255,255,.65)!important}
		@media(max-width:600px){header{padding:0 10px;gap:6px}}
	</style>
</head>
<body>
<!-- Password modal (for settle action) -->
<div class="modal-overlay" id="pwd-modal">
  <div class="modal-box">
    <h2 id="pwd-modal-title">&#128274; 需要密码确认</h2>
    <p id="pwd-modal-desc">结算操作需要输入密码</p>
    <input id="pwd-input" type="password" placeholder="请输入密码" autocomplete="off">
    <div class="modal-err" id="pwd-err">密码错误，请重试</div>
    <div class="modal-actions">
      <button class="btn-reset" id="pwd-cancel-btn" onclick="closePwdModal()">取消</button>
      <button class="btn-apply" id="pwd-confirm-btn" onclick="confirmPwd()">确认</button>
    </div>
  </div>
</div>
<header>
  <a href="/Trainer" class="back-btn">&larr; 返回训练</a>
  <select id="dir-select"><option value="">-- 选择目录 --</option></select>
  <label>每批
    <select id="count-select">
      <option value="2">2</option>
      <option value="4" selected>4</option>
      <option value="6">6</option>
      <option value="8">8</option>
    </select>
  </label>
  <label>跳至 <input type="number" id="jump-input" value="0" min="0"></label>
  <button class="back-btn" id="jump-btn">跳转</button>
</header>
<div id="gallery"><div class="empty-msg">请选择目录</div></div>
<div id="footer">
  <div id="stats">
    <span>已处理: <b id="s-done">0</b></span>
    <span>剩余: <b id="s-remain">0</b></span>
    <span>已移动: <b id="s-trashed">0</b></span>
  </div>
  <button id="settle-btn" disabled>&#9654; 结算本批</button>
</div>
<script>
var _dir = '';
var _offset = 0;
var _count = 4;
var _total = 0;
var _trashed = 0;
var _marked = {};
var _currentFiles = [];

// ---- 目录列表 ----
fetch('/FlitGame/api/dirs').then(function(r){return r.json()}).then(function(dirs){
  var sel = document.getElementById('dir-select');
  dirs.forEach(function(d){
    var opt = document.createElement('option');
    opt.value = d.path;
    opt.textContent = d.name + ' (' + d.count + ')';
    sel.appendChild(opt);
  });
});

// ---- 事件绑定 ----
var _authenticated = false;

document.getElementById('dir-select').addEventListener('change', function(){
  if(!_authenticated){
    // show password modal for dir selection
    _pendingDir = this.value;
    this.value = '';
    document.getElementById('pwd-modal-title').textContent = '🔒 需要密码';
    document.getElementById('pwd-modal-desc').textContent = '选择目录需要输入密码';
    document.getElementById('pwd-confirm-btn').style.background = '#2a5a3a';
    document.getElementById('pwd-confirm-btn').style.borderColor = '#3a7a5a';
    document.getElementById('pwd-confirm-btn').style.color = '#fff';
    document.getElementById('pwd-cancel-btn').style.display = 'block';
    document.getElementById('pwd-modal').classList.add('open');
    document.getElementById('pwd-input').value = '';
    document.getElementById('pwd-err').style.display = 'none';
    document.getElementById('pwd-input').focus();
    return;
  }
  _dir = this.value;
  _offset = 0;
  _marked = {};
  if(_dir) loadSamples();
  else document.getElementById('gallery').innerHTML = '<div class="empty-msg">请选择目录</div>';
});

document.getElementById('count-select').addEventListener('change', function(){
  _count = parseInt(this.value);
  _offset = 0;
  _marked = {};
  if(_dir) loadSamples();
});

document.getElementById('jump-btn').addEventListener('click', function(){
  var v = parseInt(document.getElementById('jump-input').value);
  if(v >= 0 && v !== _offset){
    _offset = v;
    _marked = {};
    if(_dir) loadSamples();
  }
});
document.getElementById('jump-input').addEventListener('keydown', function(e){
  if(e.key === 'Enter') document.getElementById('jump-btn').click();
});

// ---- 加载样本 ----
function loadSamples(){
  var gallery = document.getElementById('gallery');
  gallery.innerHTML = '<div class="empty-msg">加载中...</div>';
  document.getElementById('settle-btn').disabled = true;

  fetch('/FlitGame/api/samples?dir=' + encodeURIComponent(_dir) + '&offset=' + _offset + '&count=' + _count)
    .then(function(r){return r.json()})
    .then(function(data){
      _total = data.total;
      _currentFiles = data.samples;
      renderGallery();
      updateStats();
      document.getElementById('settle-btn').disabled = data.samples.length === 0;
    })
    .catch(function(){
      gallery.innerHTML = '<div class="empty-msg">加载失败，请重试</div>';
    });
}

// ---- 渲染图片网格 ----
function renderGallery(){
  var gallery = document.getElementById('gallery');
  gallery.innerHTML = '';
  if(!_currentFiles || !_currentFiles.length){
    gallery.innerHTML = '<div class="empty-msg">没有更多样本了</div>';
    return;
  }
  _currentFiles.forEach(function(item, idx){
    var card = document.createElement('div');
    card.className = 'img-card' + (_marked[item.path] ? ' marked' : '');
    card.dataset.idx = idx;

    var img = document.createElement('img');
    img.src = 'data:image/jpeg;base64,' + item.data;
    img.alt = item.filename;

    var overlay = document.createElement('div');
    overlay.className = 'overlay';
    var span = document.createElement('span');
    span.textContent = '⚠️ 即将排除这个样本';
    overlay.appendChild(span);

    var label = document.createElement('div');
    label.className = 'label';
    label.textContent = item.filename;

    card.appendChild(img);
    card.appendChild(overlay);
    card.appendChild(label);

    card.addEventListener('click', function(){
      var path = _currentFiles[parseInt(this.dataset.idx)].path;
      if(_marked[path]){
        delete _marked[path];
        this.classList.remove('marked');
      } else {
        _marked[path] = true;
        this.classList.add('marked');
      }
    });

    gallery.appendChild(card);
  });
}

// ---- 更新统计 ----
function updateStats(){
  var done = _offset;
  var remain = Math.max(0, _total - _offset);
  document.getElementById('s-done').textContent = done;
  document.getElementById('s-remain').textContent = remain;
  document.getElementById('s-trashed').textContent = _trashed;
}

// ---- 结算按钮 ----
// Password modal
var _pendingMarked = null;
var _pendingDir = null;
function openPwdModal() {
  document.getElementById('pwd-modal-title').textContent = '✅ 结算确认';
  document.getElementById('pwd-modal-desc').textContent = '输入密码将标记的图片移入回收站';
  document.getElementById('pwd-confirm-btn').style.background = '#2a5a3a';
  document.getElementById('pwd-confirm-btn').style.borderColor = '#3a7a5a';
  document.getElementById('pwd-confirm-btn').style.color = '#fff';
  document.getElementById('pwd-cancel-btn').style.display = 'block';
  document.getElementById('pwd-modal').classList.add('open');
  document.getElementById('pwd-input').value = '';
  document.getElementById('pwd-err').style.display = 'none';
  document.getElementById('pwd-input').focus();
}
function closePwdModal() {
  document.getElementById('pwd-modal').classList.remove('open');
  document.getElementById('pwd-confirm-btn').style.background = '';
  document.getElementById('pwd-confirm-btn').style.borderColor = '';
  document.getElementById('pwd-confirm-btn').style.color = '';
  _pendingMarked = null;
  _pendingDir = null;
}
document.getElementById('pwd-input').addEventListener('keydown', function(e) {
  if (e.key === 'Enter') confirmPwd();
});
async function confirmPwd() {
  var pwd = document.getElementById('pwd-input').value;
  var err = document.getElementById('pwd-err');
  try {
    var r = await fetch('/check-password', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({password: pwd}),
    });
    var res = await r.json();
    if (res.ok) {
      // save before closePwdModal nulls them
      var marked = _pendingMarked;
      var dir = _pendingDir;
      closePwdModal();
      if (dir !== null) {
        _authenticated = true;
        _pendingDir = null;
        document.getElementById('dir-select').value = dir;
        _dir = dir;
        _offset = 0;
        _marked = {};
        if(_dir) loadSamples();
        else document.getElementById('gallery').innerHTML = '<div class="empty-msg">请选择目录</div>';
      } else if (marked) {
        doSettle(marked, pwd);
      }
    } else {
      err.style.display = 'block';
    }
  } catch(e) {
    err.textContent = '请求失败: ' + e.message;
    err.style.display = 'block';
  }
}
function doSettle(markedPaths, pwd) {
  var btn = document.getElementById('settle-btn');
  btn.disabled = true;
  btn.textContent = '⏳ 处理中...';
  fetch('/FlitGame/api/trash', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({paths: markedPaths, password: pwd})
  })
  .then(function(r){return r.json()})
  .then(function(result){
    _trashed += result.moved;
    _offset += _count;
    _marked = {};
    loadSamples();
    btn.textContent = '▶ 结算本批';
  })
  .catch(function(){
    btn.disabled = false;
    btn.textContent = '▶ 结算本批';
  });
}

document.getElementById('settle-btn').addEventListener('click', function(){
  var markedPaths = Object.keys(_marked);
  if(markedPaths.length === 0){
    _offset += _count;
    _marked = {};
    loadSamples();
    return;
  }
  if (_authenticated) {
    doSettle(markedPaths);
    return;
  }
  _pendingMarked = markedPaths;
  openPwdModal();
});
</script>
</body>
</html>"""


# ------------------------------------------------------------------
# 独立 HTTP 服务（仅 __main__ 使用）
# ------------------------------------------------------------------

class ThreadingHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True


class _Handler(BaseHTTPRequestHandler):
    """独立模式下的 HTTP handler（集成模式下由 page_trainer 接管）"""
    def log_message(self, *a): pass

    def _json(self, data, status=200):
        body = json.dumps(data, ensure_ascii=False).encode()
        self.send_response(status)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path
        params = parse_qs(parsed.query)

        if path == '/api/dirs':
            self._json(scan_aligned_dirs())
        elif path == '/api/samples':
            dir_path = params.get('dir', [None])[0]
            if not dir_path:
                self._json({'samples': [], 'total': 0})
                return
            offset = int(params.get('offset', ['0'])[0])
            count = int(params.get('count', ['4'])[0])
            samples, total = list_samples(dir_path, offset, count)
            self._json({'samples': samples, 'total': total})
        elif path == '/api/stats':
            dir_path = params.get('dir', [None])[0]
            if not dir_path:
                self._json({'total': 0})
                return
            total = count_files(dir_path)
            trash_dir = Path(dir_path).parent / (Path(dir_path).name + '_trash')
            trashed = count_files(str(trash_dir)) if trash_dir.exists() else 0
            self._json({'total': total, 'trashed': trashed})
        else:
            body = HTML.encode()
            self.send_response(200)
            self.send_header('Content-Type', 'text/html; charset=utf-8')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    def do_POST(self):
        if self.path == '/api/trash':
            try:
                content_len = int(self.headers.get('Content-Length', 0))
                raw = self.rfile.read(content_len)
                data = json.loads(raw)
                moved = move_to_trash(data.get('paths', []))
                self._json({'moved': moved})
            except Exception as e:
                self._json({'error': str(e)}, 400)
        else:
            self._json({'error': 'not found'}, 404)


def start_standalone_server(port=6790):
    """启动独立 HTTP 服务器（自测用）。"""
    ip = '127.0.0.1'
    try:
        import socket
        ip = socket.gethostbyname(socket.gethostname())
    except Exception:
        pass
    print(f'[FlitGame] 样本筛选: http://{ip}:{port}')
    server = ThreadingHTTPServer(('0.0.0.0', port), _Handler)
    import threading
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DFL 样本筛选页面')
    parser.add_argument('--workspace', default='workspace',
                        help='workspace 目录路径')
    parser.add_argument('--port', type=int, default=6790,
                        help='HTTP 端口 (默认 6790)')
    args = parser.parse_args()
    WORKSPACE = args.workspace

    server = start_standalone_server(port=args.port)
    print(f'[FlitGame] 按 Ctrl+C 停止')
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.shutdown()
        print('[FlitGame] 已停止')
