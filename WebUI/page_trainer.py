#!/usr/bin/env python3
import argparse
import base64
import json
import os
import time
import threading
import webbrowser
from pathlib import Path

import ipaddress
import urllib.request
import urllib.error
import json as _json

_geo_cache = {}

def _get_ip_location(ip_str):
    if ip_str in _geo_cache:
        return _geo_cache[ip_str]
    try:
        ip_obj = ipaddress.ip_address(ip_str)
        if ip_obj.is_private or ip_obj.is_loopback:
            _geo_cache[ip_str] = '本地'
            return '本地'
    except Exception:
        pass
    try:
        req = urllib.request.Request(
            f'http://ip-api.com/json/{ip_str}?lang=zh-CN',
            headers={'User-Agent': 'DFL-WebUI/1.0'},
        )
        with urllib.request.urlopen(req, timeout=3) as resp:
            data = _json.loads(resp.read())
            if data.get('status') == 'success':
                loc = f"{data.get('city', '')} {data.get('regionName', '')}".strip() or '未知'
                _geo_cache[ip_str] = loc
                return loc
    except Exception:
        pass
    _geo_cache[ip_str] = '未知'
    return '未知'

# numpy-aware JSON encoder for model options with numpy types
class _NumpyEncoder(json.JSONEncoder):
    def default(self, o):
        import numpy as np
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, (np.ndarray,)):
            return o.tolist()
        if isinstance(o, (np.bool_,)):
            return bool(o)
        return super().default(o)
from http.server import BaseHTTPRequestHandler, HTTPServer
from socketserver import ThreadingMixIn
from urllib.parse import urlparse

try:
    from WebUI import page_flit_game as _flit
except ImportError:
    try:
        import page_flit_game as _flit
    except ImportError:
        _flit = None

try:
    from WebUI import page_settings as _settings_mod
except ImportError:
    try:
        import page_settings as _settings_mod
    except ImportError:
        _settings_mod = None

try:
    from samplelib import page_sample_viz as _viz_mod
except ImportError:
    try:
        import page_sample_viz as _viz_mod
    except ImportError:
        _viz_mod = None

try:
    from WebUI import page_merger_studio as _merger_mod
except ImportError:
    try:
        import page_merger_studio as _merger_mod
    except ImportError:
        _merger_mod = None

try:
    from core.leras import nn
    nn.initialize_main_env()
except Exception:
    pass

try:
    from WebUI.studio_backend import StudioBackend
    _studio_be = StudioBackend()
    from WebUI import studio_settings as _studio_settings
    _ws_dir = os.getcwd()
    _studio_settings.init(_ws_dir)
    if _studio_be is not None:
        _studio_be.set_workspace(_ws_dir)
    # [MergerStudio] logs suppressed
    # print(f"[MergerStudio] Workspace: {_ws_dir}")
    # print(f"[MergerStudio] Cache: {os.path.join(_ws_dir, "studio_cache")}")
except ImportError:
    _studio_be = None
    _ws_dir = os.getcwd()

class ThreadingHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True

_mem_cache = {}
_mem_cache_lock = threading.Lock()
_preview_requested = threading.Event()
_save_requested = threading.Event()
_close_requested = threading.Event()

def update_cache(data: dict):
    with _mem_cache_lock:
        _mem_cache.update(data)

def is_preview_requested() -> bool:
    return _preview_requested.is_set()

def clear_preview_request():
    _preview_requested.clear()

def is_save_requested() -> bool:
    return _save_requested.is_set()

def clear_save_request():
    _save_requested.clear()

def is_close_requested() -> bool:
    return _close_requested.is_set()

def clear_close_request():
    _close_requested.clear()

_settings_pending = None
_settings_lock = threading.Lock()

_model_pending = None
_model_lock = threading.Lock()

WEBUI_AUTH = os.environ.get('DFL_WEBUI_PASSWORD', '')

def set_password(pwd: str):
    global WEBUI_AUTH
    WEBUI_AUTH = pwd

def get_settings_update():
    global _settings_pending
    with _settings_lock:
        val = _settings_pending
        _settings_pending = None
    return val

def get_model_options_update():
    global _model_pending
    with _model_lock:
        val = _model_pending
        _model_pending = None
    if val is not None:
        print(f'[WebUI-Trace] get_model_options_update 收到: {val}')
    return val

def _read_cache():
    with _mem_cache_lock:
        return dict(_mem_cache)

def _scan_models(model_dir=None):
    """扫描模型目录中的已保存模型实例，从 _data.dat 读取真实配置"""
    from datetime import datetime
    import io
    import os
    import pickle

    class _TorchStub:
        """替换 torch 类型的占位符，接受任意 pickle 操作。"""
        def __new__(cls, *args, **kwargs):
            return super().__new__(cls)
        def __init__(self, *args, **kwargs):
            pass
        def __setstate__(self, state):
            pass

    class _ModelDataUnpickler(pickle.Unpickler):
        def find_class(self, module, name):
            if module == 'torch' or module.startswith('torch.'):
                return _TorchStub
            return super().find_class(module, name)

    results = []
    base = Path(model_dir or MODEL_DIR)
    if not base.exists():
        return results
    seen = set()
    for f in base.iterdir():
        if not f.is_file() or not f.name.endswith('_data.dat'):
            continue
        name = f.name[:-len('_data.dat')]
        parts = name.rsplit('_', 1)
        if len(parts) != 2:
            continue
        base_name, class_name = parts
        if base_name in seen:
            continue
        seen.add(base_name)

        options = {}
        training_iter = 0
        try:
            model_data = _ModelDataUnpickler(io.BytesIO(f.read_bytes())).load()
            options = model_data.get('options', {})
            # 只移除可能泄漏的 _TorchStub 占位符，保留 numpy 等合法类型
            options = {k: v for k, v in options.items() if type(v).__name__ != '_TorchStub'}
            training_iter = model_data.get('iter', 0)
        except Exception as e:
            import traceback
            print(f'[ERROR] WebUI _scan_models failed: {e}')
            traceback.print_exc()

        archi_val = options.get('archi', '')
        archi_type = archi_val.split('-')[0] if archi_val else ''

        stat = f.stat()
        results.append({
            'name': base_name,
            'class_name': class_name,
            'type': archi_type.upper() if archi_type else class_name,
            'full_name': name,
            'resolution': options.get('resolution', '?'),
            'archi': archi_val or '?',
            'face_type': options.get('face_type', '?'),
            'ae_dims': options.get('ae_dims', '?'),
            'e_dims': options.get('e_dims', '?'),
            'd_dims': options.get('d_dims', '?'),
            'd_mask_dims': options.get('d_mask_dims', '?'),
            'use_bf16': options.get('use_bf16', False),
            'iter': training_iter,
            'mtime': datetime.fromtimestamp(stat.st_mtime).isoformat(),
            'size': stat.st_size,
        })
    results.sort(key=lambda x: x['mtime'], reverse=True)
    return results

# SSE broadcast
_lock = threading.Lock()
_clients = []
_last_payload = None

def _broadcast(data: str):
    global _last_payload
    msg = ('data: ' + data + '\n\n').encode()
    with _lock:
        _last_payload = msg
        dead = []
        for wfile in _clients:
            try:
                wfile.write(msg)
                wfile.flush()
            except Exception:
                dead.append(wfile)
        for d in dead:
            _clients.remove(d)

# Poller: only broadcasts iter/speed/last-loss — no images, no full history
MODEL_DIR = '.'
POLL_INTERVAL = 2
_iter_times = []

def _poller():
    global _iter_times
    prev_iter = -1
    while True:
        try:
            cache = _read_cache()
            cur_iter = cache.get('iter', 0)
            loss_history = cache.get('loss_history', [])
            now = time.time()

            _iter_times.append((now, cur_iter))
            _iter_times = [(t, i) for t, i in _iter_times if now - t <= 60]
            speed = 0.0
            if len(_iter_times) >= 2:
                dt = _iter_times[-1][0] - _iter_times[0][0]
                di = _iter_times[-1][1] - _iter_times[0][1]
                if dt > 0 and di > 0:
                    speed = di / dt

            if cur_iter != prev_iter:
                prev_iter = cur_iter
                last_loss = loss_history[-1] if loss_history else []
                payload = json.dumps({
                    'iter': cur_iter,
                    'loss': last_loss,
                    'speed': round(speed, 3),
                    'ts': int(now),
                })
                _broadcast(payload)
        except Exception:
            pass
        time.sleep(POLL_INTERVAL)

HTML = """<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>DFL Monitor</title>
	<style>
	/* font: system geometric stack (minimalist, no remote fetch) */
	*{box-sizing:border-box;margin:0;padding:0}
	html,body{height:100%}body{overflow:hidden;background:#101013;display:flex;flex-direction:column;color:rgba(255,255,255,.8);font-family:'SF Pro Display', 'Segoe UI Variable', 'Segoe UI',-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;font-size:13px;font-weight:450;line-height:1.5;-webkit-font-smoothing:antialiased;-moz-osx-font-smoothing:grayscale;user-select:none}
	::selection{background:rgba(122,158,159,.35);color:#d2d3d5}
	::-webkit-scrollbar{width:6px;height:6px}
	::-webkit-scrollbar-track{background:transparent}
	::-webkit-scrollbar-thumb{background:rgba(255,255,255,.055);border-radius:3px;transition:background .15s}
	::-webkit-scrollbar-thumb:hover{background:rgba(255,255,255,.14)}
	*{scrollbar-width:thin;scrollbar-color:rgba(255,255,255,.055) transparent}
	header{display:flex;align-items:center;gap:12px;height:40px;padding:0 14px;background:#16171a;border-bottom:1px solid rgba(255,255,255,.04);flex-shrink:0}
	header h1{font-size:13px;font-weight:550;color:rgba(255,255,255,.8);letter-spacing:-.01em}
	#dot{width:8px;height:8px;border-radius:50%;background:rgba(255,255,255,.12);flex-shrink:0;transition:background .4s}
	#dot.live{background:linear-gradient(135deg,#6f8f90,#67878a);box-shadow:0 0 8px rgba(122,158,159,.4)}
	.modal-overlay{position:fixed;inset:0;background:rgba(0,0,0,.6);backdrop-filter:blur(4px);-webkit-backdrop-filter:blur(4px);z-index:1000;display:none;align-items:center;justify-content:center}
	.modal-overlay.open{display:flex}
	.modal-box{min-width:320px;max-width:420px;background:#16171a;border:1px solid rgba(255,255,255,.055);border-radius:10px;box-shadow:0 16px 48px rgba(0,0,0,.5);padding:20px;text-align:center}
	.modal-box h2{font-size:14px;font-weight:550;color:rgba(255,255,255,.8);margin-bottom:8px}
	.modal-box p{font-size:12px;color:rgba(255,255,255,.45);margin-bottom:16px}
	.modal-box input{width:100%;padding:8px 12px;background:rgba(255,255,255,.02);border:1px solid rgba(255,255,255,.055);border-radius:6px;color:rgba(255,255,255,.75);font-size:13px;font-family:inherit;text-align:center;outline:none;transition:border-color .12s}
	.modal-box input:focus{border-color:rgba(122,158,159,.4);box-shadow:0 0 0 2px rgba(122,158,159,.08)}
	.modal-box .modal-err{font-size:12px;color:#a65d5d;margin-top:8px;display:none}
	.modal-box .modal-actions{display:flex;gap:8px;margin-top:16px}
	.modal-box .modal-actions button{flex:1;padding:7px 12px;border-radius:6px;cursor:pointer;font-size:12px;font-weight:500;font-family:inherit;border:none;transition:opacity .12s}
	#stats{display:flex;gap:16px;margin-left:auto;font-size:12px;color:rgba(255,255,255,.35);flex-wrap:wrap;align-items:center}
	#stats span b{color:rgba(255,255,255,.65);font-weight:500}
	button{padding:5px 12px;background:rgba(255,255,255,.04);border:1px solid rgba(255,255,255,.055);border-radius:6px;color:rgba(255,255,255,.55);cursor:pointer;font-size:11px;font-family:inherit;font-weight:500;transition:all .12s}
	button:hover{background:rgba(255,255,255,.07);color:rgba(255,255,255,.75)}
	button:disabled{opacity:.3;cursor:default;background:rgba(255,255,255,.02)}
	#btn-auto-refresh.active{background:rgba(122,158,159,.15);border-color:rgba(122,158,159,.3);color:#8faeaf}
	.btn-apply{background:linear-gradient(135deg,#6f8f90,#67878a)!important;color:#d2d3d5!important}
	.btn-apply:hover{opacity:.9!important;box-shadow:0 2px 8px rgba(122,158,159,.25)!important}
	.btn-reset{background:rgba(255,255,255,.04)!important;color:rgba(255,255,255,.45)!important}
	.btn-reset:hover{background:rgba(255,255,255,.07)!important;color:rgba(255,255,255,.65)!important}
	main{flex:1;min-height:0;overflow-y:auto;padding:16px 20px;display:flex;flex-direction:column;gap:16px}
	.card{background:#16171a;border:1px solid rgba(255,255,255,.04);border-radius:8px;padding:14px}
	.card h2{font-size:10px;font-weight:600;text-transform:uppercase;letter-spacing:.06em;color:rgba(255,255,255,.25);margin-bottom:10px}
	#chart-wrap{height:200px}
	canvas#chart{width:100%;height:100%}
	#legend{display:flex;flex-wrap:wrap;gap:6px;margin-top:8px}
	.leg{display:flex;align-items:center;gap:4px;font-size:11px;color:rgba(255,255,255,.35)}
	.leg-dot{width:8px;height:8px;border-radius:50%}
	#prev-tabs{display:flex;gap:4px;margin-bottom:10px;flex-wrap:wrap}
	.prev-tab{padding:4px 12px;background:rgba(255,255,255,.04);border:1px solid rgba(255,255,255,.04);border-radius:5px;color:rgba(255,255,255,.35);cursor:pointer;font-size:11px;font-family:inherit;font-weight:500;transition:all .12s}
	.prev-tab:hover{background:rgba(255,255,255,.04);color:rgba(255,255,255,.55)}
	.prev-tab.active{background:rgba(122,158,159,.12);border-color:rgba(122,158,159,.25);color:rgba(255,255,255,.75)}
	.preview-wrap{display:flex;flex-direction:column;align-items:center;min-width:0;width:100%}
	.preview-img-wrap{position:relative;display:inline-block;width:100%;line-height:0}
	#preview-img{width:100%;height:auto;border-radius:5px;border:1px solid rgba(255,255,255,.04);cursor:zoom-in;display:block;box-shadow:0 4px 12px rgba(0,0,0,.3)}
	#preview-labels{position:absolute;top:0;left:0;width:100%;height:100%;pointer-events:none;overflow:hidden}
	#prev-hint{font-size:11px;color:rgba(255,255,255,.2);margin-top:6px}
	#log{font-size:12px;color:rgba(255,255,255,.3);font-family:'SF Pro Display', 'Segoe UI Variable', 'Segoe UI',monospace;max-height:80px;overflow-y:auto;line-height:1.6;background:rgba(0,0,0,.15);border-radius:5px;padding:8px 10px;border:1px solid rgba(255,255,255,.04)}
	#log div:last-child{color:rgba(255,255,255,.65)}
	#lightbox{display:none;position:fixed;inset:0;background:rgba(0,0,0,.7);backdrop-filter:blur(6px);-webkit-backdrop-filter:blur(6px);z-index:999;align-items:center;justify-content:center;cursor:zoom-out}
	#lightbox.open{display:flex}
	#lightbox img{max-width:92vw;max-height:92vh;border-radius:6px;object-fit:contain;box-shadow:0 8px 32px rgba(0,0,0,.5)}
	@media(max-width:600px){header{padding:0 10px;gap:8px}main{padding:10px}.card{padding:10px}#stats{gap:8px}}
	/* --- sidebar --- */
	#sidebar{position:fixed;top:0;left:-260px;width:260px;height:100vh;background:rgba(13,13,14,.85);backdrop-filter:blur(16px);-webkit-backdrop-filter:blur(16px);border-right:1px solid rgba(255,255,255,.04);z-index:200;transition:left .25s ease;display:flex;flex-direction:column;padding:0;overflow-y:auto}
	#sidebar.open{left:0}
	#sidebar .shead{padding:14px 16px;border-bottom:1px solid rgba(255,255,255,.04)}
	#sidebar .shead h2{font-size:13px;font-weight:550;color:rgba(255,255,255,.8)}
	#sidebar .shead .ssub{font-size:11px;color:rgba(255,255,255,.25);margin-top:4px}
	#sidebar .shead .ssub span{color:rgba(255,255,255,.5)}
	#sidebar .snav{flex:1;padding:8px 0}
	#sidebar .snav a{display:flex;align-items:center;gap:8px;padding:9px 16px;color:rgba(255,255,255,.4);text-decoration:none;font-size:13px;font-weight:450;transition:all .12s;border-left:2px solid transparent}
	#sidebar .snav a:hover{background:rgba(255,255,255,.03);color:rgba(255,255,255,.65);border-left-color:rgba(122,158,159,.4)}
	#sidebar .snav a.active{color:rgba(255,255,255,.8);border-left-color:#7b7be6;background:rgba(122,158,159,.06)}
	#sidebar-overlay{position:fixed;inset:0;background:rgba(0,0,0,.4);z-index:199;display:none}
	#sidebar-overlay.open{display:block}
	#hamburger{display:inline-flex;align-items:center;justify-content:center;width:30px;height:30px;background:rgba(255,255,255,.04);border:1px solid rgba(255,255,255,.04);border-radius:5px;color:rgba(255,255,255,.4);cursor:pointer;font-size:16px;line-height:1;padding:0;flex-shrink:0;transition:all .12s}
	#hamburger:hover{background:rgba(255,255,255,.07);color:rgba(255,255,255,.65)}
	</style>
</head>
<body>
<!-- Password modal -->
<div class="modal-overlay" id="pwd-modal">
  <div class="modal-box">
    <h2 id="pwd-modal-title">&#128274; 需要密码确认</h2>
    <p id="pwd-modal-desc">请输入密码</p>
    <input id="pwd-input" type="password" placeholder="请输入密码" autocomplete="off">
    <div class="modal-err" id="pwd-err">密码错误，请重试</div>
    <div class="modal-actions">
      <button class="btn-reset" onclick="closePwdModal()" id="pwd-cancel-btn">取消</button>
      <button class="btn-apply" id="pwd-confirm-btn" onclick="confirmPwd()">确认</button>
    </div>
  </div>
</div>
<div id="sidebar-overlay"></div>
<div id="sidebar">
  <div class="shead">
    <h2>DFL Torch</h2>
    <div class="ssub">Iter: <span id="sb-iter">--</span> &middot; <span id="sb-speed">-- it/s</span></div>
    <div class="ssub" id="sb-model" style="margin-top:2px;font-size:10px;color:#6b6c70"></div>
  </div>
  <div class="snav">
    <a href="/Trainer" class="active">&#9707; 训练监控</a>
    <a href="/FlitGame">&#9673; 样本筛选</a>
    <a href="/Settings">&#9881; 参数调整</a>
  </div>
</div>
<header>
  <button id="hamburger" title="菜单">&#9776;</button>
  <div id="dot"></div>
  <h1>DFL Training Monitor</h1>
  <div id="stats">
    <span>Iter: <b id="s-iter">&#8212;</b></span>
    <span>Speed: <b id="s-speed">&#8212;</b> it/s</span>
    <span>Loss: <b id="s-loss">&#8212;</b></span>
    <span id="s-time" style="color:#6b6c70"></span>
    <button id="btn-refresh" onclick="requestPreview()">&#21047;&#26032;&#39044;&#35272;</button>
    <button id="btn-auto-refresh" onclick="toggleAutoRefresh()">&#9632; 自动</button>
    <button id="btn-save" onclick="requestSave()">&#20445;&#23384;</button>
    <button id="btn-quit" onclick="requestQuit()">&#36864;&#20986;</button>
  </div>
</header>
<main>
  <div class="card">
    <h2>Loss Curves</h2>
    <div id="chart-wrap"><canvas id="chart"></canvas></div>
    <div id="legend"></div>
  </div>
  <div class="card">
    <h2>Preview</h2>
    <div id="prev-tabs"></div>
    <div class="preview-wrap">
      <div class="preview-img-wrap">
        <img id="preview-img" src="" alt="&#28857;&#20987;&#21047;&#26032;&#39044;&#35272;&#33719;&#21462;&#22270;&#20687;" style="display:none">
        <div id="preview-labels"></div>
      </div>
      <span id="prev-hint">&#28857;&#20987;&#8220;&#21047;&#26032;&#39044;&#35272;&#8221;&#33719;&#21462;&#22270;&#20687;</span>
    </div>
  </div>
  <div class="card">
    <h2>Event Log <span id="log-toggle" onclick="toggleLogScroll()" style="font-size:11px;color:#6b6c70;cursor:pointer;float:right;font-weight:400;text-transform:none;letter-spacing:0">&#9660; 自动</span></h2>
    <div id="log"></div>
  </div>
</main>
<div id="lightbox"><img id="lb-img"></div>
<script>
const COLORS = ['#7eb8f7','#f7a07e','#7ef7a0','#f7e07e','#c07ef7','#f77eb8','#7ef7f0'];
let hiddenLossIndices = new Set();
let lossHistory = [], lossNames = [];
let zoomStart = 0, zoomEnd = 1;
let dragging = false, dragX0 = 0, zoomS0 = 0, zoomE0 = 0;

const canvas = document.getElementById('chart');
const ctx = canvas.getContext('2d');

function resizeCanvas() {
  const wrap = document.getElementById('chart-wrap');
  canvas.width = wrap.clientWidth * devicePixelRatio;
  canvas.height = wrap.clientHeight * devicePixelRatio;
  canvas.style.width = wrap.clientWidth + 'px';
  canvas.style.height = wrap.clientHeight + 'px';
  drawChart();
}

function drawChart() {
  const W = canvas.width, H = canvas.height;
  ctx.clearRect(0, 0, W, H);
  ctx.fillStyle = '#0f0f13';
  ctx.fillRect(0, 0, W, H);
  if (lossHistory.length < 2) {
    ctx.fillStyle = '#444';
    ctx.font = (14*devicePixelRatio) + 'px monospace';
    ctx.fillText('No data yet...', 20*devicePixelRatio, 40*devicePixelRatio);
    return;
  }
  const PAD = {l:48*devicePixelRatio, r:12*devicePixelRatio, t:12*devicePixelRatio, b:28*devicePixelRatio};
  const cW = W-PAD.l-PAD.r, cH = H-PAD.t-PAD.b;
  const i0 = Math.floor(zoomStart*(lossHistory.length-1));
  const i1 = Math.ceil(zoomEnd*(lossHistory.length-1));
  const slice = lossHistory.slice(i0, i1+1);
  if (slice.length < 2) return;
  const nLoss = slice[0].length;
  const tail = slice.slice(Math.floor(slice.length*0.1));
  let yMax = 0;
  tail.forEach(r => r.forEach(v => { if (v > yMax) yMax = v; }));
  yMax = Math.max(yMax*1.1, 0.001);
  ctx.strokeStyle = '#1e1e2e'; ctx.lineWidth = 1;
  for (let g = 0; g <= 4; g++) {
    const y = PAD.t + cH*(1-g/4);
    ctx.beginPath(); ctx.moveTo(PAD.l, y); ctx.lineTo(PAD.l+cW, y); ctx.stroke();
    ctx.fillStyle = '#6b6c70';
    ctx.font = (10*devicePixelRatio) + 'px monospace';
    ctx.fillText((yMax*g/4).toFixed(4), 2, y+4*devicePixelRatio);
  }
  ctx.fillStyle = '#6b6c70'; ctx.font = (10*devicePixelRatio) + 'px monospace';
  for (let g = 0; g <= 4; g++) {
    const idx = Math.round(i0+(i1-i0)*g/4);
    ctx.fillText(idx, PAD.l+cW*g/4-10*devicePixelRatio, H-6*devicePixelRatio);
  }
  for (let p = 0; p < nLoss; p++) {
    if (hiddenLossIndices.has(p)) continue;
    ctx.beginPath();
    ctx.strokeStyle = COLORS[p%COLORS.length];
    ctx.lineWidth = 1.5*devicePixelRatio;
    ctx.globalAlpha = 0.85;
    slice.forEach((row, i) => {
      const x = PAD.l+(i/(slice.length-1))*cW;
      const y = PAD.t+cH*(1-Math.min(1, row[p]/yMax));
      i === 0 ? ctx.moveTo(x,y) : ctx.lineTo(x,y);
    });
    ctx.stroke();
  }
  ctx.globalAlpha = 1;
}

canvas.addEventListener('wheel', e => {
  e.preventDefault();
  const rect = canvas.getBoundingClientRect();
  const fx = (e.clientX-rect.left)/rect.width;
  const span = zoomEnd-zoomStart;
  const factor = e.deltaY > 0 ? 1.15 : 0.87;
  let newSpan = Math.min(1, Math.max(0.02, span*factor));
  let center = zoomStart+fx*span;
  zoomStart = Math.max(0, center-fx*newSpan);
  zoomEnd = Math.min(1, zoomStart+newSpan);
  if (zoomEnd === 1) zoomStart = 1-newSpan;
  drawChart();
}, {passive:false});
canvas.addEventListener('mousedown', e => { dragging=true; dragX0=e.clientX; zoomS0=zoomStart; zoomE0=zoomEnd; });
window.addEventListener('mousemove', e => {
  if (!dragging) return;
  const rect = canvas.getBoundingClientRect();
  const dx = (e.clientX-dragX0)/rect.width*(zoomE0-zoomS0);
  zoomStart = Math.max(0, zoomS0-dx);
  zoomEnd = Math.min(1, zoomE0-dx);
  drawChart();
});
window.addEventListener('mouseup', () => { dragging=false; });
canvas.addEventListener('dblclick', () => { zoomStart=0; zoomEnd=1; drawChart(); });
window.addEventListener('resize', resizeCanvas);
resizeCanvas();

function handleData(d) {
  document.getElementById('s-iter').textContent = d.iter.toLocaleString();
  document.getElementById('s-speed').textContent = d.speed > 0 ? d.speed.toFixed(2) : '—';
  document.getElementById('s-time').textContent = new Date(d.ts*1000).toLocaleTimeString();
  // sidebar stats
  document.getElementById('sb-iter').textContent = d.iter.toLocaleString();
  document.getElementById('sb-speed').textContent = d.speed > 0 ? d.speed.toFixed(2) + ' it/s' : '— it/s';
  if (d.loss && d.loss.length) {
    lossHistory.push(d.loss);
    if (lossHistory.length > 10000) lossHistory.shift();
    if (lossNames.length !== d.loss.length) {
      lossNames = d.loss.length === 5 ? ['src', 'dst', 'd_gan', 'g_gan', 'opt'] : ['src', 'dst', 'opt'];
      document.getElementById('legend').innerHTML = lossNames.map((n,i) =>
        '<div class="leg" data-idx="'+i+'" style="cursor:pointer"><div class="leg-dot" style="background:'+COLORS[i%COLORS.length]+'"></div><span>'+n+'</span></div>'
      ).join('');
      document.querySelectorAll('#legend .leg').forEach(function(el){el.addEventListener('click',function(){toggleLossLine(parseInt(this.dataset.idx));});});
    }
    document.getElementById('s-loss').textContent = d.loss.map(v => v.toFixed(4)).join(' / ');
    if (zoomEnd >= 0.98) zoomEnd = 1;
    drawChart();
  }
}

async function loadLossHistory() {
  try {
    const r = await fetch('/loss-history');
    const h = await r.json();
    if (h && h.length) {
      lossHistory = h.slice(-10000);
      if (lossHistory.length) {
        lossNames = lossHistory[0].length === 5 ? ['src', 'dst', 'd_gan', 'g_gan', 'opt'] : ['src', 'dst', 'opt'];
        document.getElementById('legend').innerHTML = lossNames.map((n,i) =>
          `<span class="leg" style="cursor:pointer" data-idx="${i}"><span class="leg-dot" style="background:${COLORS[i%COLORS.length]}"></span><span>${n}</span></span>`
        ).join('');
        document.querySelectorAll('#legend .leg').forEach(function(el){el.addEventListener('click',function(){toggleLossLine(parseInt(this.dataset.idx));});});
        drawChart();
      }
    }
  } catch(_) {}
}

let _previewPending = false;
let _allPreviews = [];
let _currentView = 0;
let _previewLabels = null;

function switchView(idx) {
  _currentView = idx;
  document.querySelectorAll('.prev-tab').forEach((btn, i) => {
    btn.classList.toggle('active', i === idx);
  });
  const img = document.getElementById('preview-img');
  const labelsEl = document.getElementById('preview-labels');
  labelsEl.innerHTML = '';
  _labelRects = [];
  if (_allPreviews[idx]) {
    img.style.display = 'block';
    document.getElementById('prev-hint').textContent = _allPreviews[idx].name + ' — ' + new Date().toLocaleTimeString();
    img.onload = function() { renderLabels(); };
    img.src = 'data:image/jpeg;base64,' + _allPreviews[idx].data;
    requestAnimationFrame(function() {
      if (img.naturalWidth) renderLabels();
    });
  }
}

let _labelRects = [];

// —— 背景自适应文字颜色 ——
// 预览图是 data: URL，canvas 不会被污染，可以离屏读像素：
// 每个标签按自己脚下那块区域的平均亮度决定用白字还是黑字（深背景→白字，浅背景→黑字）。
var _lumCanvas = null, _lumCtx = null, _lumData = null, _lumSrc = null;

function _buildLumMap(img) {
  try {
    if (!_lumCanvas) {
      _lumCanvas = document.createElement('canvas');
      _lumCtx = _lumCanvas.getContext('2d', { willReadFrequently: true });
    }
    if (_lumSrc === img && _lumData) return;
    _lumCanvas.width = img.naturalWidth;
    _lumCanvas.height = img.naturalHeight;
    _lumCtx.drawImage(img, 0, 0);
    _lumData = _lumCtx.getImageData(0, 0, _lumCanvas.width, _lumCanvas.height);
    _lumSrc = img;
  } catch (e) { _lumData = null; _lumSrc = null; }
}

// 图像坐标 (x0,y0,w,h) 区域的平均亮度（Rec.709 加权，0-255）；读不到返回 null
function _regionLum(x0, y0, w, h) {
  if (!_lumData) return null;
  x0 = Math.max(0, Math.floor(x0)); y0 = Math.max(0, Math.floor(y0));
  var x1 = Math.min(_lumCanvas.width - 1, Math.floor(x0 + w));
  var y1 = Math.min(_lumCanvas.height - 1, Math.floor(y0 + h));
  if (x1 <= x0 || y1 <= y0) return null;
  var d = _lumData.data, sum = 0, n = 0;
  var stepX = Math.max(1, ((x1 - x0) / 48) | 0);
  var stepY = Math.max(1, ((y1 - y0) / 16) | 0);
  for (var y = y0; y <= y1; y += stepY) {
    for (var x = x0; x <= x1; x += stepX) {
      var i = (y * _lumCanvas.width + x) * 4;
      sum += 0.2126 * d[i] + 0.7152 * d[i + 1] + 0.0722 * d[i + 2];
      n++;
    }
  }
  return n ? sum / n : null;
}

function renderLabels() {
  const img = document.getElementById('preview-img');
  const container = document.getElementById('preview-labels');
  container.innerHTML = '';
  _labelRects = [];
  if (!_previewLabels) return;
  var tabName = (_allPreviews[_currentView] || {}).name || '';
  var isMorph = tabName.indexOf('morph') !== -1;
  const iw = img.naturalWidth, ih = img.naturalHeight;
  if (!iw || !ih) return;
  var nCols = isMorph ? 3 : (_previewLabels.n_cols || 5);
  var nRows = _previewLabels.n_samples || Math.round(ih / (iw / nCols));
  if (!nRows || nRows <= 0) return;
  if (isMorph) nRows = nRows * 2;
  const pw = container.offsetWidth, ph = container.offsetHeight;
  const cellW = iw / nCols, cellH = ih / nRows;
  const sx = pw / iw, sy = ph / ih;
  const srcF = _previewLabels.src_fnames || [];
  const dstF = _previewLabels.dst_fnames || [];
  const _colSpec = (_previewLabels.col_spec && _previewLabels.col_spec.length === nCols)
    ? _previewLabels.col_spec : null;
  // 调色列的实际算法名（rct / mt / sot / lut / ...），标签上如实显示
  const _gradeName = _previewLabels.grade_name || '';
  const sLoss = _previewLabels.src_loss;
  const dLoss = _previewLabels.dst_loss;
  const sLossVec = _previewLabels.src_loss_vec || null;
  const dLossVec = _previewLabels.dst_loss_vec || null;
  const pad = Math.max(4, cellW / 50);
  const fs = Math.max(10, cellW / 22);
  _buildLumMap(img);
  if (_lumCtx) _lumCtx.font = '700 ' + fs + 'px sans-serif';
  const baseStyle = 'position:absolute;font-size:' + (fs * sx) + 'px;font-family:sans-serif;font-weight:700;white-space:nowrap;pointer-events:none;line-height:1;bottom:auto;';

  function fmtLoss(v) {
    return (v !== undefined && v !== 0) ? v.toFixed(4) : '';
  }

  for (let r = 0; r < nRows; r++) {
    var rowLabel;
    if (isMorph) {
      var sampleIdx = Math.floor(r / 2);
      var rowInSample = r % 2;
      var dRaw = sampleIdx < dstF.length ? dstF[sampleIdx] : '';
      rowLabel = function(col) {
        if (col === 0) return dRaw;
        if (rowInSample === 0) return 'morph ' + [25, 50][col - 1];
        return 'morph ' + [65, 75, 100][col - 1];
      };
    } else {
      var sRaw = r < srcF.length ? srcF[r] : '';
      var dRaw = r < dstF.length ? dstF[r] : '';
      var rowIdx = r;
      if (nCols === 9) {
        rowLabel = function(col) {
          if (col === 0) return sRaw;
          if (col === 1) return fmtLoss(sLossVec && sLossVec.length ? sLossVec[rowIdx] : sLoss);
          if (col === 2) return 'pred';
          return '';
        };
      } else {
        rowLabel = function(col) {
          // 模型可自报列规格（随机偏色开启时会多出 cast / rct 两组列）
          if (_colSpec) {
            const t = _colSpec[col];
            if (t === 'src') return sRaw;
            if (t === 'dst') return dRaw;
            if (t === 'src_loss') return fmtLoss(sLossVec && sLossVec.length ? sLossVec[rowIdx] : sLoss);
            if (t === 'dst_loss') return fmtLoss(dLossVec && dLossVec.length ? dLossVec[rowIdx] : dLoss);
            if (t === 'pred') return 'pred';
            if (t === 'cast') return 'cast';
            if (t === 'grade') return (_gradeName || 'rct') + ' pred';
            return '';
          }
          switch (col) {
            case 0: return sRaw;
            case 1: return fmtLoss(sLossVec && sLossVec.length ? sLossVec[rowIdx] : sLoss);
            case 2: return dRaw;
            case 3: return fmtLoss(dLossVec && dLossVec.length ? dLossVec[rowIdx] : dLoss);
            case 4: return 'pred';
          }
          return '';
        };
      }
    }

    const cellTop = (r * cellH + cellH - pad) * sy;
    for (var col = 0; col < nCols; col++) {
      var txt = rowLabel(col);
      if (!txt) continue;
      // 标签在图像坐标系里的占位，用来采样它背后那块背景的亮度
      var bx = col * cellW + pad;
      var by = (r * cellH + cellH - pad) - fs;
      var bw = _lumCtx ? _lumCtx.measureText(txt).width : txt.length * fs * 0.6;
      var lum = _regionLum(bx - 2, by - 2, bw + 4, fs + 4);
      var lightBg = lum !== null && lum > 140;
      var el = document.createElement('span');
      el.textContent = txt;
      el.style.cssText = baseStyle
        + ';left:' + ((col * cellW + pad) * sx) + 'px;top:' + (cellTop - fs * sx) + 'px;'
        + (lightBg
          ? 'color:#101013;text-shadow:0 1px 2px rgba(255,255,255,.85),0 0 6px rgba(255,255,255,.7)'
          : 'color:#fff;text-shadow:0 1px 2px rgba(0,0,0,.85),0 0 6px rgba(0,0,0,.7)');
      container.appendChild(el);
      _labelRects.push(el);
    }
  }
}

function requestPreview() {
  if (_previewPending) return;
  const btn = document.getElementById('btn-refresh');
  btn.disabled = true;
  btn.textContent = '请求中...';
  fetch('/request-preview').then(() => {
    addLog('已发送预览请求，等待训练进程响应...');
    _previewPending = true;
    _pollPreview();
  });
}

function _pollPreview() {
  let _retryCount = 0;
  const MAX_RETRIES = 600; // 500ms x 600 = 300s，覆盖长 iter（100s+）的预览渲染等待，与 trainctl 窗口一致
  const _try = () => {
    fetch('/preview').then(r => r.json()).then(resp => {
      const previews = resp.previews || resp;
      if (previews.length) {
        _previewPending = false;
        _previewLabels = resp.labels || null;
        const btn = document.getElementById('btn-refresh');
        btn.disabled = false;
        btn.textContent = '刷新预览';
        _allPreviews = previews;
        // Dynamically create tab buttons
        const container = document.getElementById('prev-tabs');
        container.innerHTML = '';
        previews.forEach((p, i) => {
          const tab = document.createElement('button');
          tab.className = 'prev-tab' + (i === 0 ? ' active' : '');
          tab.dataset.view = i;
          tab.textContent = p.name;
          tab.addEventListener('click', () => switchView(i));
          container.appendChild(tab);
        });
        if (_currentView >= previews.length) _currentView = 0;
        switchView(_currentView);
      } else if (++_retryCount < MAX_RETRIES) {
        setTimeout(_try, 500);
      } else {
        _previewPending = false;
        document.getElementById('btn-refresh').disabled = false;
        document.getElementById('btn-refresh').textContent = '刷新预览';
        addLog('预览生成超时，请重试');
      }
    }).catch(() => {
      if (++_retryCount < MAX_RETRIES) {
        setTimeout(_try, 1000);
      } else {
        _previewPending = false;
        document.getElementById('btn-refresh').disabled = false;
        document.getElementById('btn-refresh').textContent = '刷新预览';
        addLog('预览请求失败');
      }
    });
  };
  _try();
}

function requestSave() {
  _pendingAction = 'save';
  document.getElementById('pwd-modal-title').textContent = '✅ 保存模型';
  document.getElementById('pwd-modal-desc').textContent = '输入密码保存当前训练进度';
  document.getElementById('pwd-confirm-btn').style.background = '#2a5a3a';
  document.getElementById('pwd-confirm-btn').style.borderColor = '#3a7a5a';
  document.getElementById('pwd-confirm-btn').style.color = '#d2d3d5';
  document.getElementById('pwd-cancel-btn').style.display = 'block';
  openPwdModal();
}
function _doSave(pwd) {
  const btn = document.getElementById('btn-save');
  btn.disabled = true;
  btn.textContent = '保存中...';
  fetch('/request-save', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({auth: pwd}),
  }).then(() => {
    addLog('已发送保存请求...');
    setTimeout(() => { btn.disabled = false; btn.textContent = '保存'; }, 3000);
  });
}

function requestQuit() {
  _pendingAction = 'quit';
  document.getElementById('pwd-modal-title').textContent = '❌ 退出确认';
  document.getElementById('pwd-modal-desc').textContent = '输入密码保存并退出训练';
  document.getElementById('pwd-confirm-btn').style.background = '#5a2a2a';
  document.getElementById('pwd-confirm-btn').style.borderColor = '#7a3a3a';
  document.getElementById('pwd-confirm-btn').style.color = '#d2d3d5';
  document.getElementById('pwd-cancel-btn').style.display = 'block';
  openPwdModal();
}
function _doQuit(pwd) {
  if (!confirm('确认保存并退出训练？')) return;
  const btn = document.getElementById('btn-quit');
  btn.disabled = true;
  fetch('/request-quit', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({auth: pwd}),
  }).then(function(){addLog('已发送退出请求，正在保存并退出...');});
}

function addLog(msg) {
  const log = document.getElementById('log');
  const d = document.createElement('div');
  d.textContent = '[' + new Date().toLocaleTimeString() + '] ' + msg;
  log.appendChild(d);
  if (log.children.length > 50) log.removeChild(log.firstChild);
  log.scrollTop = log.scrollHeight;
}

// Password modal
let _pendingAction = null;
function openPwdModal() {
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
  _pendingAction = null;
}
document.getElementById('pwd-input').addEventListener('keydown', e => {
  if (e.key === 'Enter') confirmPwd();
});
async function confirmPwd() {
  const pwd = document.getElementById('pwd-input').value;
  const err = document.getElementById('pwd-err');
  try {
    const r = await fetch('/check-auth', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({auth: pwd}),
    });
    const res = await r.json();
    if (res.ok) { const action = _pendingAction;
      closePwdModal();
      if (action === 'save') _doSave(pwd);
      else if (action === 'quit') _doQuit(pwd);
    } else {
      err.style.display = 'block';
    }
  } catch(e) {
    err.textContent = '请求失败: ' + e.message;
    err.style.display = 'block';
  }
}

function connect() {
  const es = new EventSource('/events');
  es.onopen = () => { document.getElementById('dot').classList.add('live'); addLog('Connected'); };
  es.onerror = () => {
    document.getElementById('dot').classList.remove('live');
    es.close();
    setTimeout(connect, 3000);
  };
  es.onmessage = e => { try { handleData(JSON.parse(e.data)); } catch(_) {} };
}
connect();
loadLossHistory();

const lb = document.getElementById('lightbox');
lb.addEventListener('click', () => lb.classList.remove('open'));
document.addEventListener('keydown', e => { if (e.key === 'Escape') lb.classList.remove('open'); });
document.getElementById('preview-img').addEventListener('click', e => {
  document.getElementById('lb-img').src = e.target.src;
  lb.classList.add('open');
});

// ---- sidebar ----
const sidebar = document.getElementById('sidebar');
const overlay = document.getElementById('sidebar-overlay');
const hamburger = document.getElementById('hamburger');
function openSidebar(){ sidebar.classList.add('open'); overlay.classList.add('open'); }
function closeSidebar(){ sidebar.classList.remove('open'); overlay.classList.remove('open'); }
hamburger.addEventListener('click', openSidebar);
overlay.addEventListener('click', closeSidebar);

// ========== Keyboard Shortcuts ==========
document.addEventListener('keydown', function(e){
  // Sidebar close (always works)
  if (e.key === 'Escape') { closeSidebar(); return; }
  // Ignore if typing in an input
  if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;
  if (e.key === ' ' || e.key === 'Space') { e.preventDefault(); requestPreview(); }
  if (e.ctrlKey && e.key === 's') { e.preventDefault(); requestSave(); }
  if (e.ctrlKey && e.key === 'q') { e.preventDefault(); requestQuit(); }
});

// ========== Auto-Refresh Preview ==========
let _autoRefreshTimer = null;
function toggleAutoRefresh() {
  const btn = document.getElementById('btn-auto-refresh');
  if (_autoRefreshTimer) {
    clearInterval(_autoRefreshTimer);
    _autoRefreshTimer = null;
    btn.textContent = '■ 自动';
    btn.classList.remove('active');
    addLog('自动刷新已关闭');
  } else {
    _autoRefreshTimer = setInterval(requestPreview, 8000);
    btn.textContent = '■ 自动 ON';
    btn.classList.add('active');
    addLog('自动刷新已开启 (每 8s)');
    requestPreview(); // trigger immediately
  }
}

// ========== Log Auto-Scroll Control ==========
let _logAutoScroll = true;
function toggleLogScroll() {
  _logAutoScroll = !_logAutoScroll;
  const el = document.getElementById('log-toggle');
  if (_logAutoScroll) {
    el.innerHTML = '▼ 自动';
    el.style.color = '#6b6c70';
  } else {
    el.innerHTML = '▼ 暂停';
    el.style.color = '#f77';
  }
}
// Override addLog to respect _logAutoScroll
const _origAddLog = addLog;
addLog = function(msg) {
  const log = document.getElementById('log');
  const d = document.createElement('div');
  d.textContent = '[' + new Date().toLocaleTimeString() + '] ' + msg;
  log.appendChild(d);
  if (log.children.length > 50) log.removeChild(log.firstChild);
  if (_logAutoScroll) log.scrollTop = log.scrollHeight;
}

// ========== Load Model Info for Sidebar ==========
async function loadModelInfo() {
  try {
    const [optsR, modelsR] = await Promise.all([
      fetch('/current-model-options'),
      fetch('/api/models'),
    ]);
    const opts = await optsR.json();
    const models = await modelsR.json();
    const parts = [];
    if (opts.model_name) parts.push(opts.model_name);
    if (opts.resolution) parts.push(opts.resolution + 'px');
    if (opts.face_type) parts.push(opts.face_type);
    if (opts.archi) parts.push(opts.archi);
    document.getElementById('sb-model').textContent = parts.join(' | ');
    // Update sidebar header with model count
    if (models.length > 0) {
      var hdr = document.querySelector('#sidebar .shead h2');
      if (hdr) hdr.textContent = 'DFL Torch (' + models.length + ')';
    }
  } catch(_) {}
}
loadModelInfo();

// ── 右键跳转：预览图上右键 → 定位样本行 → 复制路径 / 打开所在文件夹（2026-10-14）──
(function() {
  var menu = null;
  function closeMenu() { if (menu) { menu.remove(); menu = null; } }
  document.addEventListener('click', closeMenu);
  window.addEventListener('blur', closeMenu);

  function buildMenu(e, img, row) {
    var F = _previewLabels || {};
    var srcF = F.src_fnames || [], dstF = F.dst_fnames || [];
    if (!srcF.length && !dstF.length) return;
    var nRows = F.n_samples || 4;
    if (!nRows || nRows <= 0) return;
    row = Math.max(0, Math.min(nRows - 1, row));
    var sPath = row < srcF.length ? String(srcF[row]) : '';
    var dPath = row < dstF.length ? String(dstF[row]) : '';
    closeMenu();
    menu = document.createElement('div');
    menu.style.cssText = 'position:fixed;z-index:99999;background:#17171b;border:1px solid #34343c;'
      + 'border-radius:8px;box-shadow:0 8px 24px rgba(0,0,0,.5);padding:6px 0;min-width:260px;font-size:12px;';
    menu.style.left = Math.min(e.clientX, window.innerWidth - 290) + 'px';
    menu.style.top = Math.min(e.clientY, window.innerHeight - 160) + 'px';
    function item(label, sub, onClick) {
      var it = document.createElement('div');
      it.style.cssText = 'padding:7px 14px;cursor:pointer;color:#d5d6da;display:flex;flex-direction:column;gap:2px;';
      it.onmouseenter = function() { it.style.background = '#26262e'; };
      it.onmouseleave = function() { it.style.background = ''; };
      var l1 = document.createElement('div'); l1.textContent = label; l1.style.fontWeight = '600';
      it.appendChild(l1);
      if (sub) { var l2 = document.createElement('div'); l2.textContent = sub; l2.style.cssText = 'color:#8a8b93;font-size:10px;max-width:250px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;'; it.appendChild(l2); }
      it.onclick = function(ev) { ev.stopPropagation(); onClick(); closeMenu(); };
      menu.appendChild(it);
      return it;
    }
    function copyText(txt) {
      if (navigator.clipboard && navigator.clipboard.writeText) navigator.clipboard.writeText(txt);
      else { var ta = document.createElement('textarea'); ta.value = txt; document.body.appendChild(ta); ta.select(); document.execCommand('copy'); ta.remove(); }
    }
    var header = document.createElement('div');
    header.textContent = '\u7b2c ' + (row + 1) + ' \u884c\u6837\u672c';
    header.style.cssText = 'padding:6px 14px 4px;color:#6f7076;font-size:10px;border-bottom:1px solid #26262e;margin-bottom:4px;';
    menu.appendChild(header);
    if (sPath) {
      item('\u590d\u5236 SRC \u8def\u5f84', sPath, function() { copyText(sPath); });
      item('\u6253\u5f00 SRC \u6240\u5728\u6587\u4ef6\u5939', '', function() {
        fetch('/reveal-path', { method: 'POST', headers: {'Content-Type':'application/json'}, body: JSON.stringify({ path: sPath }) });
      });
    }
    if (dPath) {
      item('\u590d\u5236 DST \u8def\u5f84', dPath, function() { copyText(dPath); });
      item('\u6253\u5f00 DST \u6240\u5728\u6587\u4ef6\u5939', '', function() {
        fetch('/reveal-path', { method: 'POST', headers: {'Content-Type':'application/json'}, body: JSON.stringify({ path: dPath }) });
      });
    }
    document.body.appendChild(menu);
  }

  // 原图（内嵌预览）：直接按行高换算
  var wrap = document.querySelector('.preview-img-wrap');
  if (wrap) {
    wrap.addEventListener('contextmenu', function(e) {
      e.preventDefault();
      var img = document.getElementById('preview-img');
      if (!img || img.style.display === 'none') return;
      var F = _previewLabels || {};
      var nCols = F.n_cols || 5;
      var nRows = F.n_samples || Math.round(img.naturalHeight / (img.naturalWidth / nCols));
      if (!nRows || nRows <= 0) return;
      var rect = img.getBoundingClientRect();
      var cellH = rect.height / nRows;
      buildMenu(e, img, Math.floor((e.clientY - rect.top) / cellH));
    });
  }

  // 放大层（lightbox）：object-fit:contain 需要先把点击坐标映射回图片实际显示区
  var lb = document.getElementById('lightbox');
  if (lb) {
    lb.addEventListener('contextmenu', function(e) {
      e.preventDefault();
      var img = document.getElementById('lb-img');
      if (!img || !img.naturalWidth) return;
      var F = _previewLabels || {};
      var nCols = F.n_cols || 5;
      var nRows = F.n_samples || Math.round(img.naturalHeight / (img.naturalWidth / nCols));
      if (!nRows || nRows <= 0) return;
      var rect = img.getBoundingClientRect();
      // contain 的实际显示尺寸
      var scale = Math.min(rect.width / img.naturalWidth, rect.height / img.naturalHeight);
      var dispW = img.naturalWidth * scale, dispH = img.naturalHeight * scale;
      var offX = (rect.width - dispW) / 2, offY = (rect.height - dispH) / 2;
      var y = e.clientY - rect.top - offY;
      if (y < 0 || y > dispH) return;
      buildMenu(e, img, Math.floor(y / (dispH / nRows)));
    });
  }
})();

window.addEventListener('resize', function() {
  var img = document.getElementById('preview-img');
  if (img.style.display !== 'none') renderLabels();
});

</script>
</body>
</html>"""

class Handler(BaseHTTPRequestHandler):
    def _check_pwd(self, pwd):
        return pwd == WEBUI_AUTH

    def _read_json_body(self):
        """读取并解析 JSON body，返回 dict 或 None。"""
        try:
            content_len = int(self.headers.get('Content-Length', 0))
            raw = self.rfile.read(content_len)
            return json.loads(raw)
        except Exception:
            return None

    def _require_pwd(self, data):
        """检查 data['auth'] 是否匹配，不匹配时发 401。"""
        if data and data.get('auth', '') == WEBUI_AUTH:
            return True
        self._send_json({'ok': False, 'error': '密码错误'}, 401)
        return False

    def _real_client_ip(self):
        # Check X-Forwarded-For first (reverse proxy / tunnel)
        forwarded = self.headers.get('X-Forwarded-For', '')
        if forwarded:
            ip = forwarded.split(',')[0].strip()
            if ip:
                return ip
        return self.client_address[0]

    def log_message(self, fmt, *args):
        try:
            # 只打印 4xx/5xx 错误请求，正常 2xx 静默
            is_error = args[1] if len(args) > 1 else None
            if is_error is not None and str(is_error)[0] in ('4', '5'):
                ip = self._real_client_ip()
                port = self.client_address[1]
                loc = _get_ip_location(ip)
                print(f'[WebUI] {ip}（{port}）（{loc}）{args[0] if args else fmt} [HTTP {is_error}]')
        except Exception:
            pass

    def _ok(self):
        self.send_response(200)
        self.send_header('Content-Type', 'text/plain')
        self.end_headers()
        self.wfile.write(b'ok')

    def _send_html(self, html_bytes):
        self.send_response(200)
        self.send_header('Content-Type', 'text/html; charset=utf-8')
        self.send_header('Cache-Control', 'no-cache, no-store, must-revalidate')
        self.send_header('Content-Length', str(len(html_bytes)))
        self.end_headers()
        try:
            self.wfile.write(html_bytes)
        except (ConnectionAbortedError, ConnectionResetError, BrokenPipeError, OSError):
            pass

    def _send_json(self, data, status=200):
        if status >= 400 and isinstance(data, dict) and 'error' in data:
            import traceback
            print(f'[Studio Error] status={status} {data["error"]}')
            traceback.print_exc()
        body = json.dumps(data, ensure_ascii=False, cls=_NumpyEncoder).encode()
        self.send_response(status)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        try:
            self.wfile.write(body)
        except (ConnectionAbortedError, ConnectionResetError, BrokenPipeError, OSError):
            pass

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path

        # --- 训练监控器 API ---
        if path == '/events':
            self._sse()
        elif path == '/reveal-path':
            data = self._read_json_body()
            if not data:
                self._send_json({'ok': False}, 400)
                return
            import subprocess, os
            p = str(data.get('path', ''))
            # 安全面：只允许定位已训练素材常见根下的路径，防任意 explorer 调用
            allowed = any(p.lower().startswith(r.lower()) for r in (
                'd:\\deepfake', 'f:\\dfl-pytorch', 'f:\\dfl', 'c:\\tools') if r)
            if not (p and os.path.exists(p) and allowed):
                self._send_json({'ok': False, 'error': 'path not allowed'}, 403)
                return
            try:
                if os.path.isfile(p):
                    subprocess.Popen(['explorer', '/select,', os.path.normpath(p)])
                else:
                    subprocess.Popen(['explorer', os.path.normpath(p)])
                self._send_json({'ok': True})
            except Exception as e:
                self._send_json({'ok': False, 'error': str(e)}, 500)
        elif path == '/request-preview':
            _preview_requested.set()
            self._ok()
        elif path == '/preview':
            with _mem_cache_lock:
                raw = _mem_cache.pop('previews', {})
                lbl = _mem_cache.pop('preview_labels', {})
            if raw:
                _preview_requested.clear()
            previews = [{'name': n, 'data': base64.b64encode(d).decode()} for n, d in raw.items()]
            self._send_json({'previews': previews, 'labels': lbl})
        # --- FlitGame API ---
        elif _flit is not None and path == '/FlitGame/api/dirs':
            self._send_json(_flit.scan_aligned_dirs())
        elif _flit is not None and path == '/FlitGame/api/samples':
            from urllib.parse import parse_qs
            params = parse_qs(parsed.query)
            dir_path = params.get('dir', [None])[0]
            if not dir_path:
                self._send_json({'samples': [], 'total': 0})
                return
            offset = int(params.get('offset', ['0'])[0])
            count = int(params.get('count', ['4'])[0])
            samples, total = _flit.list_samples(dir_path, offset, count)
            self._send_json({'samples': samples, 'total': total})
        elif _flit is not None and path == '/FlitGame/api/stats':
            from urllib.parse import parse_qs
            params = parse_qs(parsed.query)
            dir_path = params.get('dir', [None])[0]
            if not dir_path:
                self._send_json({'total': 0})
                return
            total = _flit.count_files(dir_path)
            trash_dir = Path(dir_path).parent / (Path(dir_path).name + '_trash')
            trashed = _flit.count_files(str(trash_dir)) if trash_dir.exists() else 0
            self._send_json({'total': total, 'trashed': trashed})
        # --- Settings API ---
        elif path == '/current-model-options':
            with _mem_cache_lock:
                opts = _mem_cache.get('model_options', {})
            self._send_json(opts)
        elif path == '/current-settings':
            with _mem_cache_lock:
                src = _mem_cache.get('settings_src', {})
            # Return flat settings (same for both src/dst)
            self._send_json(src)
        # --- 模型发现 API ---
        elif path == '/api/models':
            self._send_json(_scan_models())
        # --- 页面路由 ---
        elif path in ('/', '/Trainer'):
            self._send_html(HTML.encode())
        elif _flit is not None and path == '/FlitGame':
            self._send_html(_flit.HTML.encode())
        elif path == '/loss-history':
            with _mem_cache_lock:
                history = _mem_cache.get('loss_history', [])
            self._send_json(history[-1000:])
        elif _settings_mod is not None and path == '/Settings':
            self._send_html(_settings_mod.HTML.encode())
        # --- SampleViz API ---
        elif _viz_mod is not None and path == '/SampleViz/api/dirs':
            self._send_json(_viz_mod.scan_aligned_dirs())
        elif _viz_mod is not None and path == '/SampleViz/api/data':
            from urllib.parse import parse_qs
            params = parse_qs(parsed.query)
            dir_path = params.get('dir', [None])[0]
            if not dir_path:
                self._send_json({'error': 'missing dir'})
                return
            data = _viz_mod.get_angle_data(dir_path)
            self._send_json(data)
        elif _viz_mod is not None and path == '/SampleViz/api/progress':
            self._send_json(_viz_mod.get_progress())
        elif _viz_mod is not None and path == '/SampleViz':
            self._send_html(_viz_mod.HTML.encode())
        elif _merger_mod is not None and path in ('/MergerStudio', '/merger-studio'):
            self._send_html(_merger_mod.HTML.encode())
        elif _studio_be is not None and path == '/studio/models':
            import os as _os2
            model_dir = _os2.path.join(MODEL_DIR) if MODEL_DIR else '.'
            dfm_files = []
            if _os2.path.isdir(model_dir):
                for f in sorted(_os2.listdir(model_dir)):
                    if f.endswith('.dfm'):
                        fp = _os2.path.join(model_dir, f)
                        dfm_files.append({'name': f, 'size': _os2.path.getsize(fp)})
            self._send_json(dfm_files)
        elif _studio_be is not None and path == '/studio/frame':
            from urllib.parse import parse_qs
            params = parse_qs(parsed.query)
            sid = params.get('sid', [None])[0]
            t = params.get('t', [None])[0]
            if not sid or t is None:
                self._send_json({'error': 'missing sid or t'}, 400)
                return
            try:
                jpg = _studio_be.extract_frame(sid, float(t))
                if jpg:
                    self.send_response(200)
                    self.send_header('Content-Type', 'image/jpeg')
                    self.send_header('Content-Length', str(len(jpg)))
                    self.send_header('Cache-Control', 'public, max-age=3600')
                    self.end_headers()
                    self.wfile.write(jpg)
                else:
                    self._send_json({'error': 'extraction failed'}, 500)
            except Exception as ex:
                self._send_json({'error': str(ex)}, 500)
        elif _studio_be is not None and path == '/studio/stream':
            from urllib.parse import parse_qs
            params = parse_qs(parsed.query)
            sid = params.get('sid', [None])[0]
            t = params.get('t', ['0'])[0]
            if not sid:
                self._send_json({'error': 'missing sid'}, 400)
                return
            proc = _studio_be.start_stream(sid, float(t))
            if not proc:
                self._send_json({'error': 'stream start failed'}, 500)
                return
            try:
                self.send_response(200)
                self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=ffmpeg')
                self.send_header('Cache-Control', 'no-cache')
                self.end_headers()
                while True:
                    chunk = proc.stdout.read(4096)
                    if not chunk:
                        break
                    try:
                        self.wfile.write(chunk)
                    except Exception:
                        break
            except Exception:
                pass
            finally:
                _studio_be.stop_stream(sid)
        elif _studio_be is not None and path == '/studio/duration':
            from urllib.parse import parse_qs
            params = parse_qs(parsed.query)
            sid = params.get('sid', [None])[0]
            if not sid:
                self._send_json({'error': 'missing sid'}, 400)
                return
            dur = _studio_be.probe_duration(sid)
            self._send_json({'duration': dur})
        elif _studio_be is not None and path == '/studio/thumbs':
            from urllib.parse import parse_qs
            params = parse_qs(parsed.query)
            sid = params.get('sid', [None])[0]
            n_str = params.get('n', [None])[0]
            w_str = params.get('w', [None])[0]
            h_str = params.get('h', [None])[0]
            if not sid or n_str is None or w_str is None or h_str is None:
                self._send_json({'error': 'missing sid, n, w, or h'}, 400)
                return
            try:
                n, w, h = int(n_str), int(w_str), int(h_str)
                if n < 1 or n > 200:
                    self._send_json({'error': 'n out of range (1-200)'}, 400)
                    return
                jpg = _studio_be.extract_thumbstrip(sid, n, w, h)
                if jpg:
                    self.send_response(200)
                    self.send_header('Content-Type', 'image/jpeg')
                    self.send_header('Content-Length', str(len(jpg)))
                    self.send_header('Cache-Control', 'public, max-age=3600')
                    self.end_headers()
                    self.wfile.write(jpg)
                else:
                    self._send_json({'error': 'strip extraction failed'}, 500)
            except Exception as ex:
                self._send_json({'error': str(ex)}, 500)
        elif _studio_be is not None and path == '/studio/thumbs-pregen':
            from urllib.parse import parse_qs
            params = parse_qs(parsed.query)
            sid = params.get('sid', [None])[0]
            w_str = params.get('w', [None])[0]
            h_str = params.get('h', [None])[0]
            if not sid or w_str is None or h_str is None:
                self._send_json({'error': 'missing sid, w, or h'}, 400)
                return
            try:
                w, h = int(w_str), int(h_str)
                jpg = _studio_be.get_pregen_strip(sid, w, h)
                if jpg:
                    self.send_response(200)
                    self.send_header('Content-Type', 'image/jpeg')
                    self.send_header('Content-Length', str(len(jpg)))
                    self.send_header('Cache-Control', 'public, max-age=3600')
                    self.end_headers()
                    self.wfile.write(jpg)
                else:
                    self._send_json({'error': 'pregen strip failed'}, 500)
            except Exception as ex:
                self._send_json({'error': str(ex)}, 500)
        elif _studio_be is not None and path == '/studio/settings':
            self._send_json(_studio_be.get_settings())
        elif _merger_mod is not None and path == '/merger-studio/logo':
            import os as _os
            logo_path = _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), 'ui', 'img', 'logo_new.png')
            try:
                with open(logo_path, 'rb') as _f:
                    data = _f.read()
                self.send_response(200)
                self.send_header('Content-Type', 'image/png')
                self.send_header('Content-Length', str(len(data)))
                self.end_headers()
                self.wfile.write(data)
            except Exception:
                self.send_response(404)
                self.end_headers()
        elif _studio_be is not None and path == '/studio/cache-list':
            self._send_json(_studio_be.list_cache())

        elif _studio_be is not None and path == '/studio/batch-export-status-v2':
            from urllib.parse import parse_qs
            params = parse_qs(parsed.query)
            task_id = params.get('task_id', [None])[0]
            if not task_id:
                self._send_json({'error': 'missing task_id'}, 400)
                return
            self._send_json(_studio_be.get_batch_status(task_id))

        elif _studio_be is not None and path == '/studio/frames-strip':
            from urllib.parse import parse_qs
            params = parse_qs(parsed.query)
            ck = params.get('cache_key', [None])[0]
            st = params.get('start', ['0'])[0]
            ct = params.get('count', ['10'])[0]
            w = params.get('w', ['160'])[0]
            h = params.get('h', ['90'])[0]
            if not ck:
                self._send_json({'error': 'missing cache_key'}, 400)
                return
            try:
                data = _studio_be.get_frames_strip(ck, int(st), int(ct), int(w), int(h))
                if data:
                    self.send_response(200)
                    self.send_header('Content-Type', 'image/jpeg')
                    self.send_header('Content-Length', str(len(data)))
                    self.send_header('Cache-Control', 'public, max-age=3600')
                    self.end_headers()
                    self.wfile.write(data)
                else:
                    self._send_json({'error': 'no frames found'}, 404)
            except Exception as ex:
                self._send_json({'error': str(ex)}, 500)
        else:
            self._send_html(HTML.encode())

    def do_POST(self):
        path = urlparse(self.path).path
        if _flit is not None and path == '/FlitGame/api/trash':
            try:
                data = self._read_json_body()
                if not self._require_pwd(data):
                    return
                moved = _flit.move_to_trash(data.get('paths', []))
                self._send_json({'moved': moved})
            except Exception as e:
                self._send_json({'error': str(e)}, 400)
        elif path == '/check-auth':
            try:
                data = self._read_json_body() or {}
                ok = data.get('auth', '') == WEBUI_AUTH
                self._send_json({'ok': ok})
            except Exception as e:
                self._send_json({'ok': False, 'error': str(e)}, 400)
        elif _studio_be is not None and path == '/studio/upload':
            try:
                content_len = int(self.headers.get('Content-Length', 0))
                raw = self.rfile.read(content_len)
                from urllib.parse import unquote
                filename = unquote(self.headers.get('X-Filename', 'video.mp4'))
                sid = _studio_be.upload_video(raw, filename)
                dur = _studio_be.probe_duration(sid)
                self._send_json({'sid': sid, 'filename': filename, 'duration': dur})
            except Exception as ex:
                self._send_json({'error': str(ex)}, 500)
        elif _studio_be is not None and path == '/studio/load-model':
            try:
                data = self._read_json_body() or {}
                model_dir = MODEL_DIR or '.'
                dfm_name = data.get('name', '')
                if not dfm_name:
                    self._send_json({'error': 'missing model name'}, 400)
                    return
                dfm_path = os.path.join(model_dir, dfm_name)
                if not os.path.exists(dfm_path):
                    self._send_json({'error': 'model file not found'}, 400)
                    return
                info = _studio_be.load_model(dfm_path)
                self._send_json(info)
            except Exception as ex:
                self._send_json({'error': str(ex)}, 500)
        elif _studio_be is not None and path == '/studio/settings':
            try:
                data = self._read_json_body() or {}
                result = _studio_be.update_settings(data)
                self._send_json(result)
            except Exception as ex:
                self._send_json({'error': str(ex)}, 500)

        elif _studio_be is not None and path == '/studio/batch-export':
            try:
                data = self._read_json_body() or {}
                sid = data.get('sid', '')
                if not sid:
                    self._send_json({'error': 'missing sid'}, 400)
                    return
                model_info = _studio_be.get_loaded_model()
                if not model_info.get('loaded'):
                    self._send_json({'error': 'no model loaded'}, 400)
                    return
                settings = _studio_be.get_settings()
                task_id = _studio_be.start_batch_export(
                    sid, model_info['path'],
                    data.get('output_dir', ''),
                    float(data.get('start_sec', 0)),
                    float(data.get('end_sec', 0)),
                    settings
                )
                self._send_json({'task_id': task_id})
            except Exception as ex:
                self._send_json({'error': str(ex)}, 500)

        elif _studio_be is not None and path == '/studio/batch-export-status':
            from urllib.parse import parse_qs
            params = parse_qs(urlparse(self.path).query)
            task_id = params.get('task_id', [None])[0]
            if not task_id:
                self._send_json({'error': 'missing task_id'}, 400)
                return
            self._send_json(_studio_be.get_batch_status(task_id))

        elif _studio_be is not None and path == '/studio/cancel-batch-export':
            try:
                data = self._read_json_body() or {}
                task_id = data.get('task_id', '')
                if not task_id:
                    self._send_json({'error': 'missing task_id'}, 400)
                    return
                _studio_be.cancel_batch_export(task_id)
                self._send_json({'status': 'cancelled'})
            except Exception as ex:
                self._send_json({'error': str(ex)}, 500)

        elif _studio_be is not None and path == '/studio/batch-export-v2':
            try:
                data = self._read_json_body() or {}
                sid = data.get('sid', '')
                if not sid:
                    self._send_json({'error': 'missing sid'}, 400)
                    return
                start_sec = float(data.get('start_sec', 0))
                end_sec = float(data.get('end_sec', 0))
                settings = data.get('settings', None)
                need_pause = data.get('need_pause_for_filter', False)
                options = {
                    'start_sec': start_sec,
                    'end_sec': end_sec,
                    'output_dir': data.get('output_dir', ''),
                    'format': data.get('output_format', 'jpg'),
                    'jpeg_quality': int(data.get('jpeg_quality', 95)),
                    'settings': settings,
                    'need_pause_for_filter': need_pause,
                }
                task_id = _studio_be.start_export_v2(sid, options)
                self._send_json({'task_id': task_id})
            except Exception as ex:
                self._send_json({'error': str(ex)}, 500)

        elif _studio_be is not None and path == '/studio/cancel-batch-export-v2':
            try:
                data = self._read_json_body() or {}
                task_id = data.get('task_id', '')
                if not task_id:
                    self._send_json({'error': 'missing task_id'}, 400)
                    return
                _studio_be.cancel_batch_export(task_id)
                self._send_json({'status': 'cancelled'})
            except Exception as ex:
                self._send_json({'error': str(ex)}, 500)

        elif path == '/studio/cluster-faces':
            try:
                data = self._read_json_body() or {}
                aligned_dir = data.get('aligned_dir', '')
                if not aligned_dir:
                    self._send_json({'error': 'missing aligned_dir'}, 400)
                    return
                eps = float(data.get('eps', 0.5))
                min_samples = int(data.get('min_samples', 3))
                from FacesetProcessor.Filter import FaceIDFilter
                f = FaceIDFilter(Path(aligned_dir))
                result = f.filter_by_face_id(eps, min_samples)
                if isinstance(result, dict):
                    self._send_json({'clustered': result.get('clustered', True), 'groups': result.get('groups', 0)})
                elif isinstance(result, bool):
                    self._send_json({'clustered': result, 'groups': 0})
                else:
                    self._send_json({'clustered': True, 'groups': int(result)})
            except Exception as ex:
                self._send_json({'error': str(ex)}, 500)

        elif path == '/studio/merge-aligned':
            try:
                data = self._read_json_body() or {}
                aligned_dir = data.get('aligned_dir', '')
                if not aligned_dir:
                    self._send_json({'error': 'missing aligned_dir'}, 400)
                    return
                from FacesetProcessor.Filter import FaceIDFilter
                f = FaceIDFilter(Path(aligned_dir))
                f.merge_subfolders_to_aligned()
                self._send_json({'merged': True})
            except Exception as ex:
                self._send_json({'error': str(ex)}, 500)

        elif _studio_be is not None and path == '/studio/open-directory':
            try:
                data = self._read_json_body() or {}
                dir_path = data.get('path', '')
                if not dir_path:
                    self._send_json({'error': 'missing path'}, 400)
                    return
                _studio_be.open_directory(dir_path)
                self._send_json({'opened': True})
            except Exception as ex:
                self._send_json({'error': str(ex)}, 500)

        elif _studio_be is not None and path == '/studio/continue-batch-export':
            try:
                data = self._read_json_body() or {}
                task_id = data.get('task_id', '')
                if not task_id:
                    self._send_json({'error': 'missing task_id'}, 400)
                    return
                from WebUI.studio_backend import _batch_tasks, _batch_lock
                with _batch_lock:
                    task = _batch_tasks.get(task_id)
                if not task:
                    self._send_json({'error': 'task not found'}, 404)
                    return
                task._resume_event.set()
                self._send_json({'continued': True, 'task_id': task_id})
            except Exception as ex:
                self._send_json({'error': str(ex)}, 500)

        elif _studio_be is not None and path == '/studio/extract-cache':
            try:
                data = self._read_json_body() or {}
                sid = data.get('sid', '')
                if not sid:
                    self._send_json({'error': 'missing sid'}, 400)
                    return
                cache_info = _studio_be.check_cache(sid)
                if cache_info['hit']:
                    self._send_json({'status': 'cached', 'cache_dir': cache_info['cache_dir']})
                    return
                video_path = _studio_be.get_video_path(sid)
                if not video_path:
                    self._send_json({'error': 'no video loaded'}, 400)
                    return
                cache_key = _studio_be.get_cache_key_for_sid(sid)
                if not cache_key:
                    self._send_json({'error': 'cache key generation failed'}, 400)
                    return
                cache_dir = os.path.join(_ws_dir, 'studio_cache', cache_key)
                _studio_be.extract_thumb_cache(sid, cache_dir)
                self._send_json({'status': 'done', 'cache_dir': cache_dir})
            except Exception as ex:
                self._send_json({'error': str(ex)}, 500)
        elif _studio_be is not None and path == '/studio/export-frame':
            try:
                data = self._read_json_body() or {}
                sid = data.get('sid', '')
                time_sec = float(data.get('time_sec', 0))
                if not sid:
                    self._send_json({'error': 'missing sid'}, 400)
                    return
                jpg_bytes = _studio_be.export_frame(sid, time_sec)
                self.send_response(200)
                self.send_header('Content-Type', 'image/jpeg')
                self.send_header('Content-Length', str(len(jpg_bytes)))
                self.send_header('Content-Disposition', 'attachment; filename="frame.jpg"')
                self.end_headers()
                self.wfile.write(jpg_bytes)
            except Exception as ex:
                self._send_json({'error': str(ex)}, 500)

        elif _studio_be is not None and path == '/studio/analyze-frame':
            try:
                data = self._read_json_body() or {}
                sid = data.get('sid', '')
                if not sid:
                    self._send_json({'error': 'missing sid'}, 400)
                    return
                model_info = _studio_be.get_loaded_model()
                if not model_info.get('loaded'):
                    self._send_json({'error': 'no model loaded'}, 400)
                    return
                settings = _studio_be.get_settings()
                settings['time_sec'] = data.get('time_sec', 0)
                result = _studio_be.analyze_frame(sid, model_info['path'], settings)
                self._send_json(result)
            except Exception as ex:
                self._send_json({'error': str(ex)}, 500)

        elif _studio_be is not None and path == '/studio/recomposite':
            try:
                data = self._read_json_body() or {}
                sid = data.get('sid', '')
                if not sid:
                    self._send_json({'error': 'missing sid'}, 400)
                    return
                model_info = _studio_be.get_loaded_model()
                if not model_info.get('loaded'):
                    self._send_json({'error': 'no model loaded'}, 400)
                    return
                settings = data.get('settings', None)
                if not settings:
                    settings = _studio_be.get_settings()
                settings['time_sec'] = data.get('time_sec', 0)
                result = _studio_be.recomposite(sid, model_info['path'], settings)
                self._send_json(result)
            except Exception as ex:
                self._send_json({'error': str(ex)}, 500)

        elif path == '/request-save':
            data = self._read_json_body()
            if not self._require_pwd(data):
                return
            _save_requested.set()
            self._ok()
        elif path == '/request-quit':
            data = self._read_json_body()
            if not self._require_pwd(data):
                return
            _save_requested.set()
            _close_requested.set()
            self._ok()
        elif path == '/update-model-options':
            try:
                data = self._read_json_body()
                if not self._require_pwd(data):
                    return
                data.pop('auth', None)
                print(f'[WebUI-Trace] /update-model-options 收到: {data}')
                global _model_pending
                with _model_lock:
                    _model_pending = data
                self._send_json({'ok': True})
            except Exception as e:
                print(f'[WebUI-Trace] /update-model-options 错误: {e}')
                self._send_json({'ok': False, 'error': str(e)}, 400)
        elif path == '/update-settings':
            try:
                data = self._read_json_body()
                if not self._require_pwd(data):
                    return
                data.pop('auth', None)
                # Apply same settings to both SRC and DST generators
                payload = {'src': data, 'dst': data}
                global _settings_pending
                with _settings_lock:
                    _settings_pending = payload
                self._send_json({'ok': True})
            except Exception as e:
                self._send_json({'ok': False, 'error': str(e)}, 400)
        elif _studio_be is not None and path == '/studio/cache-delete':
            try:
                data = self._read_json_body() or {}
                key = data.get('key', '')
                if not key:
                    self._send_json({'error': 'missing key'}, 400)
                    return
                ok = _studio_be.delete_cache(key)
                self._send_json({'deleted': ok})
            except Exception as ex:
                self._send_json({'error': str(ex)}, 500)
        else:
            self._send_json({'error': 'not found'}, 404)

    def _sse(self):
        self.send_response(200)
        self.send_header('Content-Type', 'text/event-stream')
        self.send_header('Cache-Control', 'no-cache')
        self.end_headers()
        ip = self._real_client_ip()
        port = self.client_address[1]
        loc = _get_ip_location(ip)
        print(f'[WebUI]{ip}（{port}）（{loc}）connected')
        with _lock:
            if _last_payload:
                try:
                    self.wfile.write(_last_payload)
                    self.wfile.flush()
                except Exception:
                    return
            _clients.append(self.wfile)
        try:
            while True:
                time.sleep(15)
                self.wfile.write(b': ping\n\n')
                self.wfile.flush()
        except Exception:
            pass
        finally:
            with _lock:
                if self.wfile in _clients:
                    _clients.remove(self.wfile)
            print(f'[WebUI]{ip}（{port}）（{loc}）disconnected')


def start_webui_server(port=8080, model_dir='.', poll_interval=2, auth=None, host='127.0.0.1'):
    """在后台线程中启动 WebUI HTTP 服务器并立即返回。

    Args:
        port: HTTP 监听端口 (默认 8080)。
        model_dir: 模型保存目录路径（saved_models_path），用于扫描已有模型实例。
        poll_interval: SSE 轮询间隔秒数 (默认 2)。
        auth: WebUI 访问和保存/退出密码。None 时使用环境变量 DFL_WEBUI_PASSWORD（未设置则为空）。
        host: 监听地址 (默认 127.0.0.1，仅本地访问)。
    Returns:
        ThreadingHTTPServer 实例 (调用方可在退出时 shutdown())。
    """
    global MODEL_DIR, POLL_INTERVAL, WEBUI_AUTH
    if auth is not None:
        WEBUI_AUTH = auth
    # Normalize to absolute path
    MODEL_DIR = str(Path(model_dir).resolve())
    POLL_INTERVAL = poll_interval

    # 自动推导 workspace：model_dir 的父目录即 workspace
    _workspace = str(Path(MODEL_DIR).parent)
    if _flit is not None:
        _flit.WORKSPACE = _workspace
    if _viz_mod is not None:
        _viz_mod.WORKSPACE = _workspace

    threading.Thread(target=_poller, daemon=True).start()

    print(f'\n[WebUI] http://127.0.0.1:{port}')

    server = ThreadingHTTPServer((host, port), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-dir', default='.')
    parser.add_argument('--port', type=int, default=8080)
    parser.add_argument('--interval', type=int, default=2)
    parser.add_argument('--host', default='127.0.0.1', help='监听地址 (默认 127.0.0.1)')
    args = parser.parse_args()

    server = start_webui_server(host=args.host, port=args.port, model_dir=args.model_dir, poll_interval=args.interval)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.shutdown()
