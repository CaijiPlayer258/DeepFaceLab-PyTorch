# -*- coding: utf-8 -*-
"""faceid_browser —— iOS 相册风格 faceid 分组浏览/删除工具

用法：
  python tools/faceid_browser.py --root <含 faceid_group_xxx 子夹的目录> [--port 8100]

功能：
  首页 = 每个子目录显示一张封面（iOS 相册网格），显示组名+张数
  点击封面 → 展开该组全部脸图
  图片交互：单击选中（高亮框）/ Shift+点击范围多选 / Ctrl+A 全选
  删除：选中图移到 <root>/.trash/（可恢复），按组名+文件名保留路径
  快捷键：Delete=删除选中，Esc=取消选中，Backspace=返回上级
"""
import argparse
import sys
import json
import os
import shutil
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import unquote, urlparse

HTML = """<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>FaceID Group Browser</title>
<style>
* { margin:0; padding:0; box-sizing:border-box; }
body { background:#111; color:#ddd; font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif; min-height:100vh; }
#topbar { position:sticky; top:0; z-index:100; background:#1a1a1e; padding:10px 16px; display:flex; align-items:center; gap:12px; border-bottom:1px solid #2a2a30; }
#topbar h1 { font-size:16px; font-weight:600; flex:1; }
#btn-back { display:none; background:#2a2a30; border:none; color:#ccc; padding:6px 14px; border-radius:6px; cursor:pointer; font-size:13px; }
#btn-back:hover { background:#3a3a42; }
#btn-del { background:#8b2020; border:none; color:#fff; padding:6px 16px; border-radius:6px; cursor:pointer; font-size:13px; display:none; }
#btn-del:hover { background:#a82828; }
#sel-count { font-size:12px; color:#888; display:none; }
#grid { display:grid; gap:4px; padding:4px; }
#grid.groups { grid-template-columns: repeat(auto-fill, minmax(180px, 1fr)); }
#grid.faces { grid-template-columns: repeat(auto-fill, minmax(120px, 1fr)); }
.group-card { position:relative; cursor:pointer; border-radius:8px; overflow:hidden; aspect-ratio:1; background:#1a1a1e; transition:transform .15s; }
.group-card:hover { transform:scale(1.02); }
.group-card img { width:100%; height:100%; object-fit:cover; display:block; }
.group-card .glabel { position:absolute; bottom:0; left:0; right:0; background:linear-gradient(transparent,rgba(0,0,0,.8)); padding:8px 10px 6px; }
.group-card .gname { font-size:12px; font-weight:600; color:#fff; }
.group-card .gcount { font-size:10px; color:#aaa; }
.face-item { position:relative; cursor:pointer; border-radius:4px; overflow:hidden; aspect-ratio:1; background:#1a1a1e; border:2px solid transparent; transition:border-color .1s; }
.face-item img { width:100%; height:100%; object-fit:cover; display:block; pointer-events:none; }
.face-item.selected { border-color:#4a9eff; }
.face-item.selected::after { content:'✓'; position:absolute; top:4px; right:4px; background:#4a9eff; color:#fff; width:18px; height:18px; border-radius:50%; display:flex; align-items:center; justify-content:center; font-size:11px; font-weight:700; }
.face-item .fname { position:absolute; bottom:0; left:0; right:0; font-size:9px; color:rgba(255,255,255,.6); padding:2px 4px; background:linear-gradient(transparent,rgba(0,0,0,.6)); overflow:hidden; text-overflow:ellipsis; white-space:nowrap; pointer-events:none; }
#status { position:fixed; bottom:12px; right:12px; background:#2a2a30; padding:8px 16px; border-radius:8px; font-size:12px; color:#aaa; z-index:100; }
</style>
</head>
<body>
<div id="topbar">
  <button id="btn-back" onclick="goBack()">← 返回</button>
  <h1 id="title">FaceID Groups</h1>
  <span id="sel-count"></span>
  <button id="btn-merge" onclick="mergeSelected()" style="display:none;background:#2a6a2a;border:none;color:#fff;padding:6px 16px;border-radius:6px;cursor:pointer;font-size:13px;">⇄ 合并到主簇</button>
  <button id="btn-del" onclick="deleteSelected()">🗑 删除选中</button>
</div>
<div id="grid" class="groups"></div>
<div id="status"></div>
<script>
const API = window.__API__ || '';
let currentGroup = null;
let selected = new Set();
let lastClickIdx = -1;
let allFaces = [];

async function api(path, opts) {
  const r = await fetch(API + path, opts);
  return r.json();
}

// ── 首页：分组列表 ──
async function loadGroups() {
  currentGroup = null;
  selected.clear();
  document.getElementById('btn-back').style.display = 'none';
  document.getElementById('btn-del').style.display = 'none';
  document.getElementById('sel-count').style.display = 'none';
  document.getElementById('title').textContent = 'FaceID Groups';
  document.getElementById('grid').className = 'groups';
  const data = await api('/api/groups');
  const grid = document.getElementById('grid');
  grid.innerHTML = '';
  const groups = data.groups || [];
  groups.sort((a, b) => b.count - a.count);
  for (const g of groups) {
    const card = document.createElement('div');
    card.className = 'group-card';
    card.innerHTML = `<img src="/api/thumb/${encodeURIComponent(g.name)}" loading="lazy">
      <div class="glabel"><div class="gname">${g.name}</div><div class="gcount">${g.count} 张</div></div>`;
    card.onclick = () => openGroup(g.name);
    grid.appendChild(card);
  }
  document.getElementById('status').textContent = `${groups.length} 组`;
  // 首页不显示合并按钮（合并只在线组内视图有）
  const bm = document.getElementById('btn-merge');
  if (bm) bm.style.display = 'none';
}

// ── 组内视图：全部脸图 ──
async function openGroup(name) {
  currentGroup = name;
  selected.clear();
  lastClickIdx = -1;
  document.getElementById('btn-back').style.display = 'block';
  document.getElementById('btn-del').style.display = 'block';
  document.getElementById('title').textContent = name;
  document.getElementById('grid').className = 'faces';
  const data = await api(`/api/group/${encodeURIComponent(name)}`);
  allFaces = data.faces || [];
  const grid = document.getElementById('grid');
  grid.innerHTML = '';
  allFaces.forEach((f, i) => {
    const el = document.createElement('div');
    el.className = 'face-item';
    el.dataset.idx = i;
    el.innerHTML = `<img src="/api/face/${encodeURIComponent(name)}/${encodeURIComponent(f)}" loading="lazy">
      <div class="fname">${f}</div>`;
    el.addEventListener('click', (e) => faceClick(e, i));
    grid.appendChild(el);
  });
  document.getElementById('status').textContent = `${allFaces.length} 张`;
  updateSelUI();
}

function faceClick(e, idx) {
  e.stopPropagation();
  const el = e.currentTarget;
  if (e.shiftKey && lastClickIdx >= 0) {
    // Shift 范围选
    const a = Math.min(lastClickIdx, idx), b = Math.max(lastClickIdx, idx);
    for (let i = a; i <= b; i++) selected.add(i);
  } else if (e.ctrlKey || e.metaKey) {
    // Ctrl 切换
    if (selected.has(idx)) selected.delete(idx); else selected.add(idx);
    lastClickIdx = idx;
  } else {
    // 单击 = 单选切换
    if (selected.has(idx) && selected.size === 1) { selected.delete(idx); }
    else { selected.clear(); selected.add(idx); }
    lastClickIdx = idx;
  }
  // 同步 DOM
  document.querySelectorAll('.face-item').forEach((el2, i) => {
    el2.classList.toggle('selected', selected.has(i));
  });
  updateSelUI();
}

function updateSelUI() {
  const n = selected.size;
  const sc = document.getElementById('sel-count');
  sc.textContent = n > 0 ? `已选 ${n} 张` : '';
  sc.style.display = n > 0 ? 'block' : 'none';
  document.getElementById('btn-del').style.display = (currentGroup && n > 0) ? 'block' : 'none';
  // 合并按钮：在组内视图始终显示（选中的图可以从当前组合并到主簇）
  const bm = document.getElementById('btn-merge');
  if (bm) bm.style.display = (currentGroup && !currentGroup.includes('merged') && !currentGroup.startsWith('.')) ? 'block' : 'none';
}

async function mergeSelected() {
  if (!currentGroup || selected.size === 0) return;
  const names = Array.from(selected).map(i => allFaces[i]);
  const target = 'merged';
  if (!confirm(`将 ${names.length} 张从 ${currentGroup} 合并到 ${target}/ ？`)) return;
  const r = await api('/api/merge', {
    method: 'POST', headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({ groups: [currentGroup], target: target })
  });
  if (r.moved !== undefined) {
    console.log(`合并 ${r.moved} 张`);
    openGroup(currentGroup); // 刷新当前视图
  }
}

async function deleteSelected() {
  if (!currentGroup || selected.size === 0) return;
  const names = Array.from(selected).map(i => allFaces[i]);
  if (!confirm(`确定删除 ${names.length} 张？（移到 .trash/ 可恢复）`)) return;
  const r = await api('/api/delete', {
    method: 'POST', headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({ group: currentGroup, faces: names })
  });
  if (r.deleted !== undefined) {
    console.log(`删除 ${r.deleted} 张`);
    openGroup(currentGroup); // 刷新
  }
}

function goBack() {
  if (currentGroup) loadGroups();
}

document.addEventListener('keydown', (e) => {
  if (e.key === 'Escape') { selected.clear(); updateSelUI(); document.querySelectorAll('.face-item.selected').forEach(el => el.classList.remove('selected')); }
  if (e.key === 'Backspace' && currentGroup) goBack();
  if ((e.key === 'Delete' || e.key === 'Backspace') && selected.size > 0 && currentGroup) { e.preventDefault(); deleteSelected(); }
});

loadGroups();
</script>
</body>
</html>"""


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _j(self, obj, code=200):
        b = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.send_header('Content-Length', str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def do_GET(self):
        p = unquote(urlparse(self.path).path)

        if p == '/' or p == '/index.html':
            b = HTML.encode()
            self.send_response(200)
            self.send_header('Content-Type', 'text/html; charset=utf-8')
            self.send_header('Content-Length', str(len(b)))
            self.end_headers()
            self.wfile.write(b)
        elif p == '/api/groups':
            groups = []
            for d in sorted(os.listdir(self.server.root)):
                gd = os.path.join(self.server.root, d)
                if not os.path.isdir(gd) or d.startswith('.'):
                    continue
                imgs = [f for f in os.listdir(gd) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                if imgs:
                    groups.append({'name': d, 'count': len(imgs)})
            self._j({'groups': groups})
        elif p == '/api/thumb':
            # 返回该组第一张图的缩略图
            group = p.split('/')[-1]
            gd = os.path.join(self.server.root, group)
            if os.path.isdir(gd):
                imgs = sorted(f for f in os.listdir(gd) if f.lower().endswith(('.png', '.jpg', '.jpeg')))
                if imgs:
                    fp = os.path.join(gd, imgs[0])
                    with open(fp, 'rb') as f:
                        data = f.read()
                    self.send_response(200)
                    self.send_header('Content-Type', 'image/jpeg' if fp.endswith('.jpg') else 'image/png')
                    self.send_header('Content-Length', str(len(data)))
                    self.end_headers()
                    self.wfile.write(data)
                    return
            self.send_response(404)
            self.end_headers()
        elif p.startswith('/api/group/'):
            group = p.split('/')[-1]
            gd = os.path.join(self.server.root, group)
            faces = sorted(f for f in os.listdir(gd) if f.lower().endswith(('.png', '.jpg', '.jpeg'))) if os.path.isdir(gd) else []
            self._j({'faces': faces})
        elif p.startswith('/api/face/'):
            parts = p.split('/')
            group = parts[2]
            face = parts[3]
            fp = os.path.join(self.server.root, group, face)
            if os.path.isfile(fp):
                with open(fp, 'rb') as f:
                    data = f.read()
                ct = 'image/png' if fp.endswith('.png') else 'image/jpeg'
                self.send_response(200)
                self.send_header('Content-Type', ct)
                self.send_header('Content-Length', str(len(data)))
                self.end_headers()
                self.wfile.write(data)
            else:
                self.send_response(404)
                self.end_headers()
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        p = unquote(urlparse(self.path).path)
        if p == '/api/delete':
            cl = int(self.headers.get('Content-Length', 0))
            data = json.loads(self.rfile.read(cl))
            group = data.get('group', '')
            faces = data.get('faces', [])
            trash = os.path.join(self.server.root, '.trash', group)
            os.makedirs(trash, exist_ok=True)
            n = 0
            for f in faces:
                src = os.path.join(self.server.root, group, f)
                if os.path.isfile(src):
                    dst = os.path.join(trash, f)
                    if os.path.exists(dst):
                        stem, ext = os.path.splitext(f)
                        dst = os.path.join(trash, f"{stem}_{n}{ext}")
                    shutil.move(src, dst)
                    n += 1
            self._j({'deleted': n})
        elif p == '/api/merge':
            cl = int(self.headers.get('Content-Length', 0))
            data = json.loads(self.rfile.read(cl))
            src_groups = data.get('groups', [])
            target = data.get('target', '')
            # 合并：把 src_groups 的全部图片移到 target 目录
            moved = 0
            for sg in src_groups:
                sd = os.path.join(self.server.root, sg)
                if not os.path.isdir(sd):
                    continue
                td = os.path.join(self.server.root, target)
                os.makedirs(td, exist_ok=True)
                for f in os.listdir(sd):
                    src = os.path.join(sd, f)
                    dst = os.path.join(td, f)
                    if os.path.exists(dst):
                        stem, ext = os.path.splitext(f)
                        dst = os.path.join(td, f"{stem}_{sg[:8]}{ext}")
                    shutil.move(src, dst)
                    moved += 1
                # 删空源夹
                try:
                    os.rmdir(sd)
                except OSError:
                    pass
            self._j({'moved': moved, 'target': target})


import shutil


def main():
    ap = argparse.ArgumentParser(description='FaceID 分组浏览器')
    ap.add_argument('--root', required=True, help='含 faceid_group_xxx 子夹的目录')
    ap.add_argument('--port', type=int, default=8100)
    a = ap.parse_args()

    root = os.path.abspath(a.root)
    if not os.path.isdir(root):
        sys.exit(f'目录不存在: {root}')

    server = HTTPServer(('127.0.0.1', a.port), Handler)
    server.root = root
    print(f'[faceid_browser] http://127.0.0.1:{a.port}  root={root}')
    import webbrowser
    threading.Timer(1.0, lambda: webbrowser.open(f'http://127.0.0.1:{a.port}')).start()
    server.serve_forever()


if __name__ == '__main__':
    import sys
    main()
