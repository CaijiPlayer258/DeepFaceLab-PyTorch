#!/usr/bin/env python3
"""
独立样本角度可视化服务器。
双击运行后浏览器访问 http://localhost:8899

用法:
    python samplelib/serve_angle_view.py               # 默认端口 8899
    python samplelib/serve_angle_view.py --port 8080
"""
import sys, json, os
from pathlib import Path
from urllib.parse import urlparse, parse_qs

_HERE = Path(__file__).resolve().parent
_BASE = _HERE.parent
sys.path.insert(0, str(_BASE))

_DEFAULT_ALIGNED = str((_BASE / 'workspace' / 'data_src' / 'aligned').resolve())

import page_sample_viz as viz

HTML = viz.HTML
# 统一换行符（viz.HTML 使用 CRLF）
HTML = HTML.replace('\r\n', '\n')
# 去掉 loadDirs() 调用（本服务器没有 /api/dirs 端点，也不再有下拉框）
HTML = HTML.replace('\nloadDirs();', '\n// loadDirs — removed')
# 移除对已删除元素（btn-load、dir-select）的 JS 引用，防止脚本在注册 canvas 点击前崩溃
HTML = HTML.replace(
    "document.getElementById('btn-load').addEventListener('click', loadData);",
    "// document.getElementById('btn-load').addEventListener('click', loadData); — removed"
)
HTML = HTML.replace(
    "dirSelect.addEventListener('change', () => {\n  if (dirSelect.value) loadData();\n});",
    "// dirSelect.addEventListener('change', () => { if (dirSelect.value) loadData(); }); — removed"
)
# 替换前端 API 路径
HTML = HTML.replace('/SampleViz/api/data', '/api/data')
HTML = HTML.replace('/SampleViz/api/progress', '/api/progress')
HTML = HTML.replace('/SampleViz', '/')
HTML = HTML.replace('/Trainer', '/')

# ── 替换工具栏：去掉下拉框，换成路径输入框 ─────────────────
# 找到 toolbar div，在它里面加入路径输入
_old_toolbar = '''    <label for="dir-select">对齐目录</label>
    <select id="dir-select"><option value="">— 选择目录 —</option></select>
    <button class="btn-refresh" id="btn-load">加载</button>'''
_new_toolbar = '''    <input id="path-input" type="text" placeholder="输入 aligned 目录路径..."
      style="flex:2;min-width:300px;padding:3px 8px;background:#0d0d0e;border:1px solid rgba(255,255,255,.08);border-radius:5px;color:rgba(255,255,255,.65);font-size:11px;font-family:inherit;outline:none"
      value="''' + _DEFAULT_ALIGNED + '''">
    <button class="btn-refresh" id="btn-load-path" onclick="loadPath()">加载</button>'''

if _old_toolbar in HTML:
    HTML = HTML.replace(_old_toolbar, _new_toolbar)
else:
    if __name__ == '__main__':
        print("[Viz] WARNING: toolbar pattern not found!", file=sys.stderr)

# ── 注入 JS：替换 loadData + 路径输入事件 ────────────────
# 注入图片显示弹窗 + 点击事件
# 将 canvas-wrap 改为左右并排布局：散点图 + 人脸缩略图
_canvas_close = HTML.find('</canvas>')
if _canvas_close > 0:
    _panel = '''<div id="face-panel" style="display:none;flex-direction:column;align-items:center;justify-content:center;flex:1;min-width:300px;border-left:1px solid rgba(255,255,255,.06);padding:12px">
      <img id="img-display" style="max-width:560px;max-height:560px;width:100%;border-radius:6px;margin-bottom:8px;object-fit:contain">
      <div id="img-info" style="font-size:11px;color:rgba(255,255,255,.45);text-align:center"></div>
    </div>\n  '''
    # 在 </canvas> 后插入 face-panel（不截断后续内容！）
    _insert_at = _canvas_close + len('</canvas>')
    HTML = HTML[:_insert_at] + '\n  ' + _panel + HTML[_insert_at:]

# 在 canvas-wrap 上加 flex-row
HTML = HTML.replace('class="canvas-wrap"',
                    'class="canvas-wrap" style="display:flex;flex-direction:row"')

# 在 mode-select 后插入偏移量输入框
_mode_end = HTML.find('</select>')
if _mode_end > 0:
    # 找到 mode-select 对应的 </select> 后面的位置
    _offset_html = '''    <span class="range-group" style="margin-left:12px"><label>Pitch&#160;偏移</label>
      <input id="pitch-offset" type="number" value="20" step="1" style="width:60px">
    </span>
    <span class="range-group"><label>Yaw&#160;偏移</label>
      <input id="yaw-offset" type="number" value="0" step="1" style="width:60px">
    </span>
'''
    _insert_off = _mode_end + len('</select>')
    HTML = HTML[:_insert_off] + '\n' + _offset_html + HTML[_insert_off:]

# 替换工具提示显示方式为点击显示在右侧面板
_path_js = '''
var rawData = null;

function loadPath() {
  rawData = null;
  var dir = document.getElementById('path-input').value.trim();
  if (!dir) { document.getElementById('status').textContent = '请输入目录路径'; return; }
  doLoadData(dir);
}
function showFaceImage(fname, yaw, pitch) {
  document.getElementById('face-panel').style.display = 'flex';
  document.getElementById('img-display').src = '/api/image?path=' + encodeURIComponent(fname);
  document.getElementById('img-info').textContent = fname + '  Yaw:' + yaw.toFixed(1) + '  Pitch:' + pitch.toFixed(1);
}

// 偏移量处理：override render 以拦截数据加载，存储原始值并应用偏移
var _origRender = render;
render = function(data) {
  if (!rawData && data && data.yaws) {
    rawData = { filenames: data.filenames.slice(), yaws: data.yaws.slice(), pitches: data.pitches.slice() };
  }
  applyOffsets();
  _origRender(currentData);
};

function applyOffsets() {
  if (!rawData || !currentData) return;
  var po = parseFloat(document.getElementById('pitch-offset').value) || 0;
  var yo = parseFloat(document.getElementById('yaw-offset').value) || 0;
  for (var i = 0; i < rawData.yaws.length; i++) {
    currentData.yaws[i] = rawData.yaws[i] + yo;
    currentData.pitches[i] = rawData.pitches[i] + po;
  }
}

// 偏移量输入实时更新
document.getElementById('pitch-offset').addEventListener('input', function() {
  if (rawData) { applyOffsets(); sampleCount.textContent = currentData.yaws.length; _origRender(currentData); }
});
document.getElementById('yaw-offset').addEventListener('input', function() {
  if (rawData) { applyOffsets(); sampleCount.textContent = currentData.yaws.length; _origRender(currentData); }
});

// canvas click to show face in right panel
canvas.addEventListener('click', function(e) {
  var rect = canvas.getBoundingClientRect();
  var mx = (e.clientX - rect.left) * 600 / rect.width;
  var my = (e.clientY - rect.top) * 600 / rect.height;
  var p = findPoint(mx, my);
  console.log('[Viz] click at', mx.toFixed(0), my.toFixed(0), 'found:', p ? p.name : 'null');
  if (p) showFaceImage(p.name, p.yaw, p.pitch);
});
document.getElementById('path-input').addEventListener('keydown', function(e) {
  if (e.key === 'Enter') loadPath();
});
var loadData = loadPath;
// dir-select was removed, make references to it return null safely
var _origGI = document.getElementById;
document.getElementById = function(id) {
  if (id === 'dir-select') return null;
  return _origGI.call(document, id);
};
'''
# 在脚本结束前注入自定义 JS（替换 loadData + 点击事件）
_script_end = HTML.rfind('</script>')
if _script_end > 0:
    HTML = HTML[:_script_end] + _path_js + '\n' + HTML[_script_end:]


def make_handler():
    from http.server import BaseHTTPRequestHandler

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            parsed = urlparse(self.path)
            path = parsed.path

            if path == '/' or path == '/index.html':
                self.send_response(200)
                self.send_header('Content-Type', 'text/html; charset=utf-8')
                self.end_headers()
                self.wfile.write(HTML.encode('utf-8'))

            elif path == '/api/progress':
                self.send_json(viz.get_progress())

            elif path == '/api/image':
                params = parse_qs(parsed.query)
                img_path = params.get('path', [None])[0]
                img_dir = getattr(self.server, '_img_dir', None)
                if img_path and img_dir:
                    fp = Path(img_dir) / Path(img_path).name
                    if fp.exists():
                        self.send_response(200)
                        self.send_header('Content-Type', 'image/jpeg')
                        self.end_headers()
                        self.wfile.write(fp.read_bytes())
                        return
                self.send_error(404)

            elif path == '/api/data':
                params = parse_qs(parsed.query)
                dir_path = params.get('dir', [None])[0]
                if not dir_path:
                    self.send_json({'error': 'missing dir'})
                    return
                self.server._img_dir = dir_path
                import tqdm
                _pbar_data = {'pbar': None}
                def _progress(current, total, fname):
                    if _pbar_data['pbar'] is None:
                        _pbar_data['pbar'] = tqdm.tqdm(total=total, desc="计算角度", unit="img", ascii=True)
                    _pbar_data['pbar'].n = current
                    _pbar_data['pbar'].refresh()
                    # 更新共享状态供前端轮询
                    viz._viz_progress['current'] = current
                    viz._viz_progress['total'] = total
                    viz._viz_progress['fname'] = fname
                data = viz.get_angle_data(dir_path, on_progress=_progress)
                if _pbar_data['pbar'] is not None:
                    _pbar_data['pbar'].close()
                self.send_json(data)

            else:
                self.send_error(404)

        def send_json(self, obj):
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(obj, ensure_ascii=False).encode('utf-8'))

        def log_message(self, fmt, *args):
            msg = str(args[0]) if args else ''
            if '/api/progress' not in msg:
                print(f"[{self.address_string()}] {fmt % args}")

    return Handler


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='样本角度可视化服务器')
    parser.add_argument('--port', type=int, default=8899, help='端口 (默认 8899)')
    args = parser.parse_args()

    from http.server import ThreadingHTTPServer
    handler = make_handler()
    server = ThreadingHTTPServer(('0.0.0.0', args.port), handler)
    url = f"http://localhost:{args.port}"
    print()
    print("=" * 55)
    print(f"  样本角度可视化服务器已启动")
    print(f"  浏览器访问: {url}")
    print(f"  按 Ctrl+C 停止服务器")
    print("=" * 55)
    print()
    try:
        import webbrowser
        webbrowser.open(url)
    except Exception:
        pass
    print(f"[SampleViz] 按 Ctrl+C 停止")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[SampleViz] 已停止")
        server.server_close()
