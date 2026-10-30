#!/usr/bin/env python3
"""
page_sample_viz - 样本可视化页面（模块）
展示训练集人脸角度 (yaw × pitch) 散点分布图。
"""
import json
import math
import numpy as np
from pathlib import Path
from multiprocessing import cpu_count
import concurrent.futures

try:
    import h5py
except ImportError:
    h5py = None

_HERE = Path(__file__).resolve().parent
_BASE = _HERE.parent
import sys
sys.path.insert(0, str(_BASE))

from DFLIMG import DFLIMG
from facelib.LandmarksProcessor import estimate_pitch_yaw_roll

WORKSPACE = 'workspace'

# 进度跟踪（线程安全，单线程访问）
_viz_progress = {'current': 0, 'total': 0, 'fname': ''}

def _progress_cb(current, total, fname):
    global _viz_progress
    _viz_progress['current'] = current
    _viz_progress['total'] = total
    _viz_progress['fname'] = fname

def get_progress():
    return dict(_viz_progress)

# ------------------------------------------------------------------
# HDF5 缓存读写
# ------------------------------------------------------------------

def _safe_name(filename):
    return filename.replace('/', '_SLASH_').replace('\\', '_BSLASH_')

_CACHE_VERSION = 4

def _load_cache(metadata_h5):
    """加载 HDF5 缓存，返回 {filename: (yaw, pitch)} 字典。"""
    if not metadata_h5.exists() or h5py is None:
        return {}
    cache = {}
    try:
        with h5py.File(metadata_h5, 'r') as f:
            cv = f.attrs.get('cache_version', 0)
            if cv != _CACHE_VERSION:
                return {}
            for safe_name in f.keys():
                grp = f[safe_name]
                orig = grp.attrs.get('__original_filename__', safe_name)
                yaw = grp.attrs.get('yaw', None)
                pitch = grp.attrs.get('pitch', None)
                if yaw is not None and pitch is not None:
                    cache[orig] = (float(yaw), float(pitch))
    except Exception:
        pass
    return cache

def _save_cache_entry(metadata_h5, filename, yaw, pitch):
    """增量写入一条缓存记录。"""
    if h5py is None:
        return
    try:
        with h5py.File(metadata_h5, 'a') as f:
            f.attrs['cache_version'] = _CACHE_VERSION
            sname = _safe_name(filename)
            if sname in f:
                grp = f[sname]
            else:
                grp = f.create_group(sname)
                grp.attrs['__original_filename__'] = filename
            grp.attrs['yaw'] = yaw
            grp.attrs['pitch'] = pitch
    except Exception:
        pass

def _rebuild_cache(metadata_h5, data):
    """全量重建 HDF5 缓存。"""
    if h5py is None:
        return
    try:
        with h5py.File(metadata_h5, 'w') as f:
            f.attrs['cache_version'] = _CACHE_VERSION
            for filename, (yaw, pitch) in data.items():
                sname = _safe_name(filename)
                grp = f.create_group(sname)
                grp.attrs['__original_filename__'] = filename
                grp.attrs['yaw'] = yaw
                grp.attrs['pitch'] = pitch
    except Exception:
        pass

# ------------------------------------------------------------------
# 角度计算（多进程 worker 需为模块级函数）
# ------------------------------------------------------------------

def _compute_angles_worker(args):
    """
    多进程 worker：计算单张 aligned 人脸的 yaw/pitch。
    args = (filepath_str,) 或 (filepath_str, method_str)
    返回 (filename, (yaw_deg, pitch_deg)) 或 (filename, None)。
    """
    filepath_str = args if isinstance(args, str) else args[0]

    try:
        import sys, math
        from pathlib import Path
        _here = Path(__file__).resolve().parent
        _base = _here.parent
        if str(_base) not in sys.path:
            sys.path.insert(0, str(_base))

        from DFLIMG import DFLIMG
        from facelib.LandmarksProcessor import estimate_pitch_yaw_roll
        import numpy as np

        dflimg = DFLIMG.load(filepath_str)
        if dflimg is None or not dflimg.has_data():
            return (Path(filepath_str).name, None)

        landmarks = dflimg.get_landmarks()
        shape = dflimg.get_shape()
        if landmarks is None or shape is None:
            return (Path(filepath_str).name, None)

        h, w = shape[:2]
        size = max(h, w)

        # 68 点 solvePnP 方法
        pitch_rad, yaw_rad, _ = estimate_pitch_yaw_roll(landmarks, size=size)
        yaw_deg = float(yaw_rad * 180.0 / math.pi)
        pitch_deg = float(pitch_rad * 180.0 / math.pi)

        return (Path(filepath_str).name, (yaw_deg, pitch_deg))
    except Exception:
        return (Path(filepath_str).name, None)

# ------------------------------------------------------------------
# 公开接口
# ------------------------------------------------------------------

def scan_aligned_dirs(workspace=None):
    """扫描 workspace 下所有含 aligned* 的目录，返回 [{path, name, count}]。"""
    ws = Path(workspace or WORKSPACE).resolve()
    dirs = []
    if not ws.exists():
        return dirs
    seen = set()
    for p in ws.rglob('*'):
        if p.is_dir() and p.name.startswith('aligned') and str(p) not in seen:
            seen.add(str(p))
            files = sorted([f for f in p.iterdir()
                           if f.suffix.lower() in ('.jpg', '.jpeg', '.png')])
            if files:
                dirs.append({
                    'path': str(p),
                    'name': str(p.relative_to(ws)),
                    'count': len(files),
                })
    return dirs

def get_angle_data(dir_path, on_progress=None):
    """
    获取 aligned 目录所有人脸的 (yaw, pitch) 数据。
    优先读 HDF5 缓存，缺失的自动计算并补充缓存。

    Args:
        dir_path: 人脸集目录
        on_progress: 进度回调
        method: '68pt' (solvePnP) 或 'reference' (对称性分析)
    返回 {filenames: [...], yaws: [...], pitches: [...]}。
    """
    p = Path(dir_path)
    if not p.exists():
        return {'filenames': [], 'yaws': [], 'pitches': []}

    files = sorted([f for f in p.iterdir()
                    if f.suffix.lower() in ('.jpg', '.jpeg', '.png')])
    if not files:
        return {'filenames': [], 'yaws': [], 'pitches': []}

    metadata_h5 = p / 'metadata.h5'

    # 1. 加载已有缓存（方法不一致时自动失效）
    cache = _load_cache(metadata_h5)
    if cache:
        print(f"[缓存] 已加载 {len(cache)} 条数据")

    # 2. 检查哪些文件需要计算
    cached_fnames = set(cache.keys())
    all_fnames = {f.name for f in files}
    new_fnames = all_fnames - cached_fnames
    # 剔除缓存中已经不存在的文件
    stale = cached_fnames - all_fnames
    if stale:
        for name in stale:
            cache.pop(name, None)

    # 3. 多进程计算新文件的 yaw/pitch
    if new_fnames:
        total_new = len(new_fnames)
        worker_count = min(cpu_count(), total_new, 16)  # 最多 16 进程
        file_list = sorted(new_fnames)
        file_tasks = [str(p / fname) for fname in file_list]

        done = 0
        cb = on_progress or _progress_cb

        if worker_count <= 1:
            # 单进程回退
            for idx, fname in enumerate(file_list):
                cb(idx + 1, total_new, fname)
                result = _compute_angles_worker(file_tasks[idx])
                fname_result, angles = result
                if angles is not None:
                    yaw, pitch = angles
                    cache[fname] = (yaw, pitch)
                    _save_cache_entry(metadata_h5, fname, yaw, pitch)
        else:
            with concurrent.futures.ProcessPoolExecutor(max_workers=worker_count) as pool:
                fut_map = {pool.submit(_compute_angles_worker, task): task for task in file_tasks}
                for fut in concurrent.futures.as_completed(fut_map):
                    done += 1
                    try:
                        fname_result, angles = fut.result()
                    except Exception:
                        fname_result = Path(fut_map[fut]).name
                        angles = None
                    cb(done, total_new, fname_result)
                    if angles is not None:
                        yaw, pitch = angles
                        cache[fname_result] = (yaw, pitch)
                        _save_cache_entry(metadata_h5, fname_result, yaw, pitch)

        # 如果有文件被移除，重写缓存（低频操作）
        if stale:
            _rebuild_cache(metadata_h5, cache)

    # 4. 组装返回数据
    fnames = []
    yaws = []
    pitches = []
    for fname in files:
        name = fname.name
        if name in cache:
            yaw, pitch = cache[name]
            fnames.append(name)
            yaws.append(yaw)
            pitches.append(pitch)

    return {
        'filenames': fnames,
        'yaws': yaws,
        'pitches': pitches,
    }


# ==================================================================
# HTML 前端 —— Canvas 散点图
# ==================================================================

HTML = r"""<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>DFL 样本可视化</title>
	<style>
		@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;450;500;550;600&display=swap');
		*{box-sizing:border-box;margin:0;padding:0}
		::-webkit-scrollbar{width:6px;height:6px}
		::-webkit-scrollbar-track{background:transparent}
		::-webkit-scrollbar-thumb{background:rgba(255,255,255,.08);border-radius:3px;transition:background .15s}
		::-webkit-scrollbar-thumb:hover{background:rgba(255,255,255,.14)}
		*{scrollbar-width:thin;scrollbar-color:rgba(255,255,255,.08) transparent}
		::selection{background:rgba(91,91,214,.35);color:#fff}
		body{background:#0a0a0b;color:rgba(255,255,255,.8);font-family:'Inter',-apple-system,sans-serif;font-size:13px;font-weight:450;min-height:100vh;padding:16px;display:flex;flex-direction:column;-webkit-font-smoothing:antialiased}
		header{display:flex;align-items:center;gap:10px;height:40px;padding:0 14px;background:#0d0d0e;border-bottom:1px solid rgba(255,255,255,.06);flex-shrink:0;margin:-16px -16px 12px}
		header h1{font-size:13px;font-weight:550;color:rgba(255,255,255,.8);letter-spacing:-.01em}
		.back-btn{display:inline-flex;align-items:center;gap:4px;padding:4px 10px;background:rgba(255,255,255,.04);border:1px solid rgba(255,255,255,.06);border-radius:5px;color:rgba(255,255,255,.45);text-decoration:none;font-size:11px;font-weight:500;transition:all .12s}
		.back-btn:hover{background:rgba(255,255,255,.07);color:rgba(255,255,255,.65)}
		#main-content{flex:1;display:flex;flex-direction:column;gap:8px}
		.toolbar{display:flex;align-items:center;gap:8px;flex-wrap:wrap}
		.toolbar label{font-size:11px;color:rgba(255,255,255,.35)}
		.toolbar select{padding:3px 8px;background:#0d0d0e;border:1px solid rgba(255,255,255,.08);border-radius:5px;color:rgba(255,255,255,.65);font-size:11px;font-family:inherit;outline:none;transition:border-color .12s}
		.toolbar select:focus{border-color:rgba(91,91,214,.4)}
		.range-group{display:flex;align-items:center;gap:4px}
		.range-group label{font-size:11px;color:rgba(255,255,255,.35);margin-right:2px}
		.range-group input[type=number]{width:52px;padding:3px 4px;background:#0d0d0e;border:1px solid rgba(255,255,255,.08);border-radius:5px;color:rgba(255,255,255,.65);font-size:11px;font-family:inherit;text-align:center;outline:none;transition:border-color .12s}
		.range-group input[type=number]:focus{border-color:rgba(91,91,214,.4)}
		.stat{font-size:11px;color:rgba(255,255,255,.35);margin-left:4px}
		button{padding:4px 10px;background:rgba(255,255,255,.04);border:1px solid rgba(255,255,255,.08);border-radius:5px;color:rgba(255,255,255,.55);cursor:pointer;font-size:11px;font-family:inherit;font-weight:500;transition:all .12s}
		button:hover{background:rgba(255,255,255,.07);color:rgba(255,255,255,.75)}
		button:disabled{opacity:.3;cursor:default}
		.btn-refresh{padding:4px 12px;background:rgba(255,255,255,.04);border:1px solid rgba(255,255,255,.08);border-radius:5px;color:rgba(255,255,255,.55);cursor:pointer;font-size:11px;font-family:inherit;font-weight:500;transition:all .12s}
		.btn-refresh:hover{background:rgba(255,255,255,.07);color:rgba(255,255,255,.75)}
		.progress-wrap{display:none;align-items:center;gap:10px;padding:8px 12px;background:#0d0d0e;border:1px solid rgba(255,255,255,.06);border-radius:6px}
		.progress-wrap.active{display:flex}
		.progress-text{font-size:11px;color:rgba(255,255,255,.35);flex-shrink:0;min-width:120px}
		.progress-bar{flex:1;height:4px;background:rgba(255,255,255,.06);border-radius:2px;overflow:hidden}
		.progress-bar .fill{height:100%;background:linear-gradient(90deg,#5b5bd6,#8b5cf6);border-radius:2px;transition:width .3s;width:0%}
		.canvas-wrap{flex:1;display:flex;align-items:center;justify-content:center;background:#0d0d0e;border:1px solid rgba(255,255,255,.06);border-radius:8px;padding:12px;min-height:400px}
		#scatter{max-width:100%;max-height:100%}
		#status{text-align:center;font-size:12px;color:rgba(255,255,255,.2);padding:20px}
		#tooltip{display:none;position:fixed;background:#0d0d0e;border:1px solid rgba(255,255,255,.08);border-radius:6px;padding:6px 10px;font-size:11px;color:rgba(255,255,255,.65);pointer-events:none;z-index:100;box-shadow:0 4px 12px rgba(0,0,0,.3)}
		#tooltip b{color:rgba(255,255,255,.8)}
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
		.btn-refresh:hover{background:rgba(255,255,255,.07);color:rgba(255,255,255,.75)}
		@media(max-width:600px){body{padding:10px}header{margin:-10px -10px 8px}.toolbar{gap:4px}}
	</style>
</head>
<body>
<!-- Password modal (hidden until Load clicked) -->
</header>
<main id="main-content">
  <div class="toolbar">
    <label for="dir-select">对齐目录</label>
    <select id="dir-select"><option value="">— 选择目录 —</option></select>
    <button class="btn-refresh" id="btn-load">加载</button>
    <span class="stat">样本数: <span id="sample-count">0</span></span>
  </div>
  <div class="toolbar">
    <span class="range-group"><label>Yaw</label>
      <input id="yaw-min" type="number" value="-100" step="5">
      <span style="color:#444">~</span>
      <input id="yaw-max" type="number" value="100" step="5">
    </span>
    <span class="range-group"><label>Pitch</label>
      <input id="pitch-min" type="number" value="-100" step="5">
      <span style="color:#444">~</span>
      <input id="pitch-max" type="number" value="100" step="5">
    </span>
    <select id="mode-select" onchange="_renderMode()">
      <option value="heatmap">热力图</option>
      <option value="dots">点图</option>
    </select>
    <span class="range-group" id="opacity-group" style="display:none">
      <label>透明度</label>
      <input id="dot-opacity" type="number" value="0.25" min="0" max="1" step="0.05" style="width:56px">
    </span>
    <button class="btn-refresh" id="btn-export" onclick="exportPNG()" style="margin-left:auto">&#10515; 导出 PNG</button>
  </div>
  <div class="progress-wrap" id="progress-wrap">
    <div class="progress-text" id="progress-text">计算角度中...</div>
    <div class="progress-bar"><div class="fill" id="progress-fill"></div></div>
  </div>
  <div class="canvas-wrap">
    <canvas id="scatter" width="600" height="600"></canvas>
  </div>
  <div id="status">请选择对齐目录并点击加载</div>
</main>
<div id="tooltip"></div>
<script>
const canvas = document.getElementById('scatter');
const ctx = canvas.getContext('2d');
const tooltip = document.getElementById('tooltip');
const statusEl = document.getElementById('status');
const sampleCount = document.getElementById('sample-count');
const dirSelect = document.getElementById('dir-select');
const progressWrap = document.getElementById('progress-wrap');
const progressFill = document.getElementById('progress-fill');
const progressText = document.getElementById('progress-text');

const W = 600, H = 600;
const MARGIN = 40;
const PLOT_L = MARGIN, PLOT_R = W - MARGIN;
const PLOT_T = MARGIN, PLOT_B = H - MARGIN;
const PLOT_W = PLOT_R - PLOT_L;
const PLOT_H = PLOT_B - PLOT_T;

let currentData = null;
let hoveredPoint = null;

// Range refs
const yawMin = document.getElementById('yaw-min');
const yawMax = document.getElementById('yaw-max');
const pitchMin = document.getElementById('pitch-min');
const pitchMax = document.getElementById('pitch-max');

function getRange() {
  return {
    ymin: parseFloat(yawMin.value) || -100,
    ymax: parseFloat(yawMax.value) || 100,
    pmin: parseFloat(pitchMin.value) || -100,
    pmax: parseFloat(pitchMax.value) || 100,
  };
}

// Coord transforms
function toScreen(yaw, pitch) {
  const r = getRange();
  const x = PLOT_L + (yaw - r.ymin) / (r.ymax - r.ymin) * PLOT_W;
  const y = PLOT_B - (pitch - r.pmin) / (r.pmax - r.pmin) * PLOT_H;
  return [x, y];
}
function toData(sx, sy) {
  const r = getRange();
  const yaw = (sx - PLOT_L) / PLOT_W * (r.ymax - r.ymin) + r.ymin;
  const pitch = (PLOT_B - sy) / PLOT_H * (r.pmax - r.pmin) + r.pmin;
  return [yaw, pitch];
}
function dotRadiusPx() {
  const r = getRange();
  return Math.max(0.3, 0.2 / (r.ymax - r.ymin) * PLOT_W);
}

function niceStep(range) {
  const rough = range / 6;
  const mag = Math.pow(10, Math.floor(Math.log10(rough)));
  if (rough / mag < 1.5) return mag;
  if (rough / mag < 3.5) return 2 * mag;
  if (rough / mag < 7.5) return 5 * mag;
  return 10 * mag;
}

function drawAxes() {
  const r = getRange();
  ctx.strokeStyle = '#3a3a5a';
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(PLOT_L, PLOT_T); ctx.lineTo(PLOT_L, PLOT_B);
  ctx.lineTo(PLOT_R, PLOT_B);
  ctx.stroke();
  // Yaw grid
  ctx.fillStyle = '#666';
  ctx.font = '10px monospace';
  ctx.textAlign = 'center';
  ctx.textBaseline = 'top';
  const yawStep = niceStep(r.ymax - r.ymin);
  for (let v = Math.ceil(r.ymin / yawStep) * yawStep; v <= r.ymax; v += yawStep) {
    const x = PLOT_L + (v - r.ymin) / (r.ymax - r.ymin) * PLOT_W;
    ctx.fillStyle = '#3a3a5a';
    ctx.fillRect(x, PLOT_T, 1, PLOT_H);
    ctx.fillStyle = '#666';
    ctx.fillText(v.toFixed(v % 1 === 0 ? 0 : 1), x, PLOT_B + 4);
  }
  // Pitch grid
  ctx.textAlign = 'right';
  ctx.textBaseline = 'middle';
  const pitchStep = niceStep(r.pmax - r.pmin);
  for (let v = Math.ceil(r.pmin / pitchStep) * pitchStep; v <= r.pmax; v += pitchStep) {
    const y = PLOT_B - (v - r.pmin) / (r.pmax - r.pmin) * PLOT_H;
    ctx.fillStyle = '#3a3a5a';
    ctx.fillRect(PLOT_L, y, PLOT_W, 1);
    ctx.fillStyle = '#666';
    ctx.fillText(v.toFixed(v % 1 === 0 ? 0 : 1), PLOT_L - 6, y);
  }
  // Labels
  ctx.fillStyle = '#888';
  ctx.font = '12px sans-serif';
  ctx.textAlign = 'center';
  ctx.textBaseline = 'top';
  ctx.fillText('Yaw (°)', W / 2, H - 4);
  ctx.save();
  ctx.translate(12, H / 2);
  ctx.rotate(-Math.PI / 2);
  ctx.textAlign = 'center';
  ctx.textBaseline = 'bottom';
  ctx.fillText('Pitch (°)', 0, 0);
  ctx.restore();
  // Center cross
  const [cx, cy] = toScreen(0, 0);
  ctx.strokeStyle = '#4a4a6a';
  ctx.setLineDash([3, 3]);
  ctx.beginPath();
  ctx.moveTo(cx, PLOT_T); ctx.lineTo(cx, PLOT_B);
  ctx.moveTo(PLOT_L, cy); ctx.lineTo(PLOT_R, cy);
  ctx.stroke();
  ctx.setLineDash([]);
}

function drawHeatmap(data) {
  const r = getRange();
  const yaws = data.yaws;
  const pitches = data.pitches;
  const n = yaws.length;
  if (n === 0) return;

  // Rasterize to density grid
  const GRID = 60;
  const grid = new Float32Array(GRID * GRID);
  const sigma = Math.min(3, Math.max(1, Math.floor(n / 50))); // 自适应模糊半径
  const sigma2 = sigma * sigma * 2;

  for (let i = 0; i < n; i++) {
    const gx = (yaws[i] - r.ymin) / (r.ymax - r.ymin) * GRID;
    const gy = (r.pmax - pitches[i]) / (r.pmax - r.pmin) * GRID;
    const ix = Math.floor(Math.min(gx, GRID - 1));
    const iy = Math.floor(Math.min(gy, GRID - 1));
    if (ix < 0 || ix >= GRID || iy < 0 || iy >= GRID) continue;
    // Gaussian blur contribution
    for (let dy = -sigma; dy <= sigma; dy++) {
      for (let dx = -sigma; dx <= sigma; dx++) {
        const cx = ix + dx, cy = iy + dy;
        if (cx < 0 || cx >= GRID || cy < 0 || cy >= GRID) continue;
        const dist2 = dx * dx + dy * dy;
        grid[cy * GRID + cx] += Math.exp(-dist2 / sigma2);
      }
    }
  }

  // Normalize
  let maxVal = 0;
  for (let i = 0; i < grid.length; i++) {
    if (grid[i] > maxVal) maxVal = grid[i];
  }
  if (maxVal === 0) maxVal = 1;

  // Draw density as heatmap
  const cellW = PLOT_W / GRID;
  const cellH = PLOT_H / GRID;
  for (let gy = 0; gy < GRID; gy++) {
    for (let gx = 0; gx < GRID; gx++) {
      const v = grid[gy * GRID + gx] / maxVal;
      if (v < 0.005) continue;  // 低阈值确保小样本也能显示
      const x = PLOT_L + gx * cellW;
      const y = PLOT_T + gy * cellH;
      // Blue (cold) -> Red (hot)
      const r = Math.min(255, Math.round(v * 255));
      const b = Math.min(255, Math.round((1 - v) * 255));
      const g = Math.min(255, Math.round(128 - Math.abs(v - 0.5) * 200));
      ctx.fillStyle = `rgba(${r},${g},${b},${Math.min(0.8, v * 0.5 + 0.3)})`;
      ctx.fillRect(x, y, cellW + 0.5, cellH + 0.5);
    }
  }
}

function drawDots(data) {
  // 精确坐标点图：半径 0.2°（数据坐标），透明度可调
  const yaws = data.yaws;
  const pitches = data.pitches;
  const n = yaws.length;
  if (n === 0) return;
  const rad = dotRadiusPx();
  const alpha = parseFloat(document.getElementById('dot-opacity').value) || 0.25;
  ctx.fillStyle = `rgba(120,160,255,${alpha})`;
  for (let i = 0; i < n; i++) {
    const [x, y] = toScreen(yaws[i], pitches[i]);
    if (x < PLOT_L || x > PLOT_R || y < PLOT_T || y > PLOT_B) continue;
    ctx.beginPath();
    ctx.arc(x, y, rad, 0, Math.PI * 2);
    ctx.fill();
  }
}

function drawHovered(p) {
  if (!p) return;
  const [x, y] = toScreen(p.yaw, p.pitch);
  ctx.strokeStyle = '#fff';
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.arc(x, y, 5, 0, Math.PI * 2);
  ctx.stroke();
  ctx.fillStyle = 'rgba(255,255,255,0.2)';
  ctx.beginPath();
  ctx.arc(x, y, 12, 0, Math.PI * 2);
  ctx.fill();
}

function render(data) {
  ctx.clearRect(0, 0, W, H);
  drawAxes();
  if (data && data.yaws.length) {
    const mode = document.getElementById('mode-select').value;
    if (mode === 'heatmap') {
      drawHeatmap(data);
    } else {
      drawDots(data);
    }
    drawHovered(hoveredPoint);
  }
}

function updateOpacityGroup() {
  const show = document.getElementById('mode-select').value === 'dots';
  document.getElementById('opacity-group').style.display = show ? 'flex' : 'none';
}

function exportPNG() {
  if (!currentData || !currentData.yaws.length) {
    statusEl.textContent = '没有数据可导出';
    setTimeout(() => { if (currentData) statusEl.textContent = ''; }, 2000);
    return;
  }
  const link = document.createElement('a');
  link.download = 'scatter_' + new Date().toISOString().slice(0,19).replace(/[:-]/g, '') + '.png';
  link.href = canvas.toDataURL('image/png');
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
}

// Hit test
function findPoint(mx, my) {
  if (!currentData) return null;
  const threshold = 6;
  let best = null, bestDist = threshold;
  for (let i = 0; i < currentData.yaws.length; i++) {
    const [x, y] = toScreen(currentData.yaws[i], currentData.pitches[i]);
    const dx = mx - x, dy = my - y;
    const d = Math.sqrt(dx * dx + dy * dy);
    if (d < bestDist) {
      bestDist = d;
      best = { idx: i, yaw: currentData.yaws[i], pitch: currentData.pitches[i], name: currentData.filenames[i] };
    }
  }
  return best;
}

canvas.addEventListener('mousemove', e => {
  const rect = canvas.getBoundingClientRect();
  const scaleX = W / rect.width;
  const scaleY = H / rect.height;
  const mx = (e.clientX - rect.left) * scaleX;
  const my = (e.clientY - rect.top) * scaleY;
  const p = findPoint(mx, my);
  hoveredPoint = p;
  render(currentData);
  if (p) {
    tooltip.innerHTML = `<b>${p.name}</b><br>Yaw: ${p.yaw.toFixed(1)}°<br>Pitch: ${p.pitch.toFixed(1)}°`;
    tooltip.style.display = 'block';
    tooltip.style.left = (e.clientX + 14) + 'px';
    tooltip.style.top = (e.clientY - 10) + 'px';
    canvas.style.cursor = 'pointer';
  } else {
    tooltip.style.display = 'none';
    canvas.style.cursor = 'crosshair';
  }
});

canvas.addEventListener('mouseleave', () => {
  hoveredPoint = null;
  tooltip.style.display = 'none';
  canvas.style.cursor = 'crosshair';
  render(currentData);
});

// Load dir list
async function loadDirs() {
  try {
    const r = await fetch('/SampleViz/api/dirs');
    const dirs = await r.json();
    dirSelect.innerHTML = '<option value="">— 选择目录 —</option>' +
      dirs.map(d => `<option value="${d.path}">${d.name} (${d.count} 张)</option>`).join('');
  } catch(e) {
    statusEl.textContent = '加载目录列表失败: ' + e.message;
  }
}

// Load angle data
async function doLoadData(dir) {
  if (!dir) return;
  statusEl.textContent = '加载中...';
  var _btn = document.getElementById('btn-load'); if (_btn) _btn.disabled = true;
  progressWrap.classList.add('active');
  progressFill.style.width = '0%';
  progressText.textContent = '计算角度中...';

  try {
    const poll = setInterval(async () => {
      try {
        const pr = await fetch('/SampleViz/api/progress?' + Date.now());
        const p = await pr.json();
        if (p.total > 0) {
          const pct = Math.min(100, p.current / p.total * 100);
          progressFill.style.width = pct + '%';
          progressText.textContent = `计算角度中... ${p.current}/${p.total} (${p.fname || ''})`;
        }
      } catch(_) {}
    }, 500);

    progressText.textContent = '计算角度中...';
    const r = await fetch('/SampleViz/api/data?dir=' + encodeURIComponent(dir));
    if (r.status === 200) {
      const _sz = parseInt(r.headers.get('X-Cache-Status') || '0');
    }
    const data = await r.json();
    clearInterval(poll);

    progressWrap.classList.remove('active');
    var _btn = document.getElementById('btn-load'); if (_btn) _btn.disabled = false;
    statusEl.textContent = '';

    if (data.error) {
      statusEl.textContent = '错误: ' + data.error;
      return;
    }

    currentData = data;
    sampleCount.textContent = data.yaws.length;
    render(data);
  } catch(e) {
    clearInterval(poll);
    progressWrap.classList.remove('active');
    var _btn = document.getElementById('btn-load'); if (_btn) _btn.disabled = false;
    statusEl.textContent = '加载失败: ' + e.message;
  }
}

function loadData() {
  const dir = document.getElementById('dir-select').value;
  if (!dir) return;
  doLoadData(dir);
}
document.getElementById('btn-load').addEventListener('click', loadData);
dirSelect.addEventListener('change', () => {
  if (dirSelect.value) loadData();
});
// Range inputs: re-render on change
[yawMin, yawMax, pitchMin, pitchMax].forEach(el => {
  el.addEventListener('input', () => { if (currentData) render(currentData); });
});
function _renderMode() {
  updateOpacityGroup();
  if (currentData) { try { render(currentData); } catch(e) { console.error(e); } }
}
document.getElementById('dot-opacity').addEventListener('input', () => {
  if (currentData && document.getElementById('mode-select').value === 'dots') render(currentData);
});


updateOpacityGroup();
loadDirs();
</script>
</body>
</html>"""


if __name__ == '__main__':
    # ═══════════════════════════════════════════════════════════════
    # 独立模式：命令行角度统计工具
    # 用法: python page_sample_viz.py <faceset_dir>
    # ═══════════════════════════════════════════════════════════════
    import sys, argparse

    parser = argparse.ArgumentParser(description='DFL 样本角度分布统计')
    parser.add_argument('dir', nargs='?', default=None,
                        help='人脸集目录（含 aligned JPG 文件）')
    parser.add_argument('--csv', action='store_true',
                        help='输出 CSV 格式')
    parser.add_argument('--stats-only', action='store_true',
                        help='只输出统计汇总，不逐条打印')
    args = parser.parse_args()

    target_dir = args.dir
    if not target_dir:
        # 未指定目录时扫描 workspace
        dirs = scan_aligned_dirs()
        if not dirs:
            print("未找到 aligned 目录，请指定目录路径")
            sys.exit(1)
        print("可用目录:")
        for i, d in enumerate(dirs, 1):
            print(f"  {i}. {d['name']} ({d['count']} 张)")
        print(f"\n用法: python {Path(__file__).name} <目录路径>")
        sys.exit(0)

    target_dir = Path(target_dir)
    if not target_dir.exists():
        print(f"[!] 目录不存在: {target_dir}")
        sys.exit(1)

    print(f"正在扫描 {target_dir} ...")
    print(f"角度计算方法: {args.method}")
    data = get_angle_data(target_dir, method=args.method)
    n = len(data['filenames'])

    if n == 0:
        print("[!] 未找到有效样本（可能没有 DFLJPG 格式的图片）")
        sys.exit(1)

    yaws = np.array(data['yaws'])
    pitches = np.array(data['pitches'])

    print(f"\n样本数: {n}")
    print(f"Yaw 范围:  {yaws.min():.1f}° ~ {yaws.max():.1f}°  (均值: {yaws.mean():.1f}° ± {yaws.std():.1f}°)")
    print(f"Pitch 范围: {pitches.min():.1f}° ~ {pitches.max():.1f}°  (均值: {pitches.mean():.1f}° ± {pitches.std():.1f}°)")

    # 角度覆盖评估（通用标准）
    yaw_coverage = abs(yaws.max() - yaws.min())
    pitch_coverage = abs(pitches.max() - pitches.min())
    print(f"\n角度覆盖:")
    print(f"  Yaw:   {yaw_coverage:.0f}° ({yaws.min():.0f}° ~ {yaws.max():.0f}°)")
    print(f"  Pitch: {pitch_coverage:.0f}° ({pitches.min():.0f}° ~ {pitches.max():.0f}°)")

    if yaw_coverage >= 60:
        print("  [OK] Yaw 覆盖良好 (>=60)")
    elif yaw_coverage >= 30:
        print("  [WARN] Yaw 覆盖一般 (30~60)，建议补充侧脸")
    else:
        print("  [BAD] Yaw 覆盖不足 (<30)，需要大量补充侧脸")

    if pitch_coverage >= 20:
        print("  [OK] Pitch 覆盖良好 (>=20)")
    elif pitch_coverage >= 10:
        print("  [WARN] Pitch 覆盖一般 (10~20)，建议补充仰/俯脸")
    else:
        print("  [BAD] Pitch 覆盖不足 (<10)，需要大量补充仰/俯脸")

    # 极值点采样（最左、最右、最上、最下各5张）
    print(f"\n极端角度采样:")
    for label, values in [('最左侧脸 (Yaw最小)', yaws),
                            ('最右侧脸 (Yaw最大)', yaws),
                            ('最仰脸 (Pitch最大)', pitches),
                            ('最俯脸 (Pitch最小)', pitches)]:
        indices = np.argsort(values)
        if '最大' in label or '仰' in label:
            idx = indices[-5:]  # 取最大5个
        else:
            idx = indices[:5]   # 取最小5个
        files = [data['filenames'][i] for i in idx]
        vals = [values[i] for i in idx]
        print(f"  {label}:")
        for f, v in zip(files, vals):
            print(f"    {f}: {v:.1f}°")

    # CSV 输出
    if args.csv:
        print("\n=== CSV ===")
        print("filename,yaw,pitch")
        for i in range(n):
            print(f"{data['filenames'][i]},{data['yaws'][i]:.2f},{data['pitches'][i]:.2f}")

    print("\n完成")
