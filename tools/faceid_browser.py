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
import re
import shutil
import threading
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from urllib.parse import unquote, urlparse, parse_qs

HTML = """<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>FaceID Group Browser</title>
<style>
* { margin:0; padding:0; box-sizing:border-box; }
body { background:#111; color:#ddd; font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif; min-height:100vh; overflow-x:hidden; }
#topbar { position:sticky; top:0; z-index:100; background:#1a1a1e; padding:10px 16px; display:flex; align-items:center; gap:6px; border-bottom:1px solid #2a2a30; }
#topbar h1 { font-size:16px; font-weight:600; flex:1; }
.iconbtn.dim { opacity:.35; cursor:default; }
.iconbtn.active { background:rgba(69,162,255,.18); }
canvas.lm-cv { position:absolute; inset:0; width:100%; height:100%; pointer-events:none; }
.iconbtn.dim:hover { background:none; }
.iconbtn.dim:hover::after { opacity:0; }
#btn-back:hover { background:#3a3a42; }
#btn-del:hover { background:#a82828; }
#sel-count { font-size:12px; color:#888; display:none; }
#grid { display:grid; gap:4px; padding:4px; }
#grid.groups { grid-template-columns: repeat(auto-fill, minmax(180px, 1fr)); }
#grid.faces { grid-template-columns: repeat(auto-fill, minmax(var(--fs, 120px), 1fr)); }
.group-card { position:relative; cursor:pointer; overflow:hidden; aspect-ratio:1; background:#1a1a1e; transition:transform .15s; }
.group-card:hover { transform:scale(1.02); }
.group-card.gsel { outline:3px solid #4a9eff; outline-offset:-3px; }
#size-slider { width:110px; accent-color:#fff; }
#topbar button { border-radius:0 !important; }
#topbar button.iconbtn, #topbar #zc-grid { border-radius:8px !important; }
#zoomctl { display:flex; align-items:center; gap:2px; }
.iconbtn { background:none; border:none; color:#45a2ff; width:30px; height:28px; display:flex; align-items:center; justify-content:center; cursor:pointer; padding:0; position:relative; }
.iconbtn:hover { background:rgba(255,255,255,.12); }
.iconbtn svg { display:block; }
.iconbtn::after { content:attr(data-tip); position:absolute; top:calc(100% + 8px); left:50%; transform:translateX(-50%); background:#2a2a2e; border:1px solid #4a4a50; color:#fff; font-size:12px; padding:5px 10px; border-radius:8px; white-space:nowrap; opacity:0; pointer-events:none; transition:opacity .12s; z-index:300; }
.iconbtn:hover::after { opacity:1; }
#modal-mask { position:fixed; inset:0; z-index:400; background:rgba(0,0,0,.45); display:none; align-items:center; justify-content:center; opacity:0; transition:opacity .15s; }
#modal-card { background:#1c1c1e; border:1px solid #2a2a30; border-radius:14px; min-width:280px; max-width:420px; padding:18px 20px 14px; transform:scale(.95); transition:transform .15s; }
#modal-text { color:#fff; font-size:14px; line-height:1.5; margin-bottom:16px; word-break:break-all; }
#modal-btns { display:flex; gap:10px; justify-content:flex-end; }
#modal-btns button { border:none; cursor:pointer; font-size:13px; padding:7px 18px; border-radius:8px; }
#modal-cancel { background:#3a3a40; color:#fff; }
#modal-ok { background:#0a84ff; color:#fff; }
#modal-input { display:none; width:100%; background:#2a2a30; border:1px solid #4a4a50; color:#fff; font-size:14px; padding:8px 10px; border-radius:8px; margin-bottom:16px; outline:none; }
#ctxmenu { position:fixed; z-index:500; display:none; background:rgba(38,40,46,.88); backdrop-filter:blur(20px) saturate(1.4); -webkit-backdrop-filter:blur(20px) saturate(1.4); border:1px solid rgba(255,255,255,.14); border-radius:10px; padding:5px; min-width:210px; box-shadow:0 8px 30px rgba(0,0,0,.5); }
.ctx-item { display:flex; align-items:center; gap:10px; padding:6px 10px; color:#eee; font-size:13px; border-radius:6px; cursor:pointer; }
.ctx-item:hover { background:rgba(255,255,255,.12); }
.ctx-item svg { color:#45a2ff; flex:none; }
.ctx-has-sub { position:relative; }
#ctx-sub { position:absolute; left:calc(100% + 2px); top:-5px; display:none; background:rgba(38,40,46,.92); backdrop-filter:blur(20px); border:1px solid rgba(255,255,255,.14); border-radius:10px; padding:5px; min-width:170px; box-shadow:0 8px 30px rgba(0,0,0,.5); }
.ctx-has-sub:hover #ctx-sub { display:block; }
.ctx-item .chev { margin-left:auto; color:#8a8a92; }
#lightbox { position:fixed; inset:0; z-index:450; display:none; background:rgba(0,0,0,.92); opacity:0; transition:opacity .22s; }
#lb-img { position:absolute; left:50%; top:50%; transform:translate(-50%,-50%); max-width:92vw; max-height:92vh; object-fit:contain; }
.group-card img { width:100%; height:100%; object-fit:cover; display:block; }
.group-card .glabel { position:absolute; bottom:0; left:0; right:0; background:linear-gradient(transparent,rgba(0,0,0,.8)); padding:8px 10px 6px; }
.group-card .gname { font-size:12px; font-weight:600; color:#fff; }
.group-card .gcount { font-size:10px; color:#aaa; }
.face-item { position:relative; cursor:pointer; overflow:hidden; aspect-ratio:1; background:#1a1a1e; border:2px solid transparent; transition:border-color .1s; }
.face-item img { width:100%; height:100%; object-fit:cover; display:block; pointer-events:none; }
.face-item.selected { border-color:#4a9eff; }
.face-item .fname { position:absolute; bottom:0; left:0; right:0; font-size:9px; color:rgba(255,255,255,.6); padding:2px 4px; background:linear-gradient(transparent,rgba(0,0,0,.6)); overflow:hidden; text-overflow:ellipsis; white-space:nowrap; pointer-events:none; }
#status { position:fixed; bottom:12px; right:12px; background:#2a2a30; padding:8px 16px; font-size:12px; color:#aaa; z-index:100; }
#hover-preview-wrap { position:fixed; z-index:200; display:none; filter:drop-shadow(0 12px 40px rgba(0,0,0,.55)); }
#hover-preview { background:#1a1a1e; padding:12px; width:100%; height:100%; flex-direction:column; opacity:0; transform:scale(.92); transform-origin:top left; transition:opacity .14s ease, transform .14s cubic-bezier(.2,.7,.3,1); }
#hover-preview.hp-show { opacity:1; transform:scale(1); }
#hover-preview .hp-grid { display:grid; gap:4px; overflow-y:auto; overflow-x:hidden; min-height:0; align-content:start; }
#hover-preview .hp-grid img { width:100%; aspect-ratio:1; object-fit:cover; background:#111; }
#hover-preview .hp-grid::-webkit-scrollbar { width:4px; }
#hover-preview .hp-grid::-webkit-scrollbar-thumb { background:rgba(255,255,255,.35); border-radius:999px; }
#hover-preview .hp-grid::-webkit-scrollbar-thumb:hover { background:rgba(255,255,255,.6); }
#hover-preview .hp-grid::-webkit-scrollbar-track { background:transparent; }
</style>
</head>
<body>
<div id="topbar">
  <button id="btn-back" class="iconbtn" onclick="goBack()" data-tip="返回" style="display:none;"><svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><path d="M15 5 L8 12 L15 19"/></svg></button>
  <div id="zoomctl" style="display:none;">
    <button id="zc-grid" class="iconbtn" data-tip="恢复自动尺寸"><svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M4 4 h16 v16 h-16 Z M9.3 4 v16 M14.7 4 v16 M4 9.3 h16 M4 14.7 h16"/></svg></button>
    <button id="zc-minus" class="iconbtn" data-tip="缩小"><svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><path d="M5 12 h14"/></svg></button>
    <input type="range" id="size-slider" min="0" max="6" step="1" oninput="applySize(this.value)">
    <button id="zc-plus" class="iconbtn" data-tip="放大"><svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><path d="M12 5 v14 M5 12 h14"/></svg></button>
  </div>
  <h1 id="title">FaceID Groups</h1>
  <span id="sel-count"></span>
  <button id="btn-merge-groups" class="iconbtn" onclick="mergeGroups()" data-tip="并入主簇" style="display:none;"><svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M4 4 L12 12 M20 4 L12 12 M12 12 V20 M8 16 L12 20 L16 16"/></svg></button>
  <button id="btn-newfolder-groups" class="iconbtn" onclick="groupsToNew()" data-tip="移到新文件夹" style="display:none;"><svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><path d="M12 5 v14 M5 12 h14"/></svg></button>
  <button id="btn-trash-groups" class="iconbtn" onclick="trashGroups()" data-tip="永久删除组" style="display:none;"><svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M4 7 h16 M10 7 V5 a1 1 0 0 1 1-1 h2 a1 1 0 0 1 1 1 v2 M6.5 7 l1 13 h9 l1-13 M10 11 v5 M14 11 v5"/></svg></button>
  <button id="btn-landmarks" class="iconbtn" onclick="toggleLandmarks()" data-tip="显示标记点" style="display:none;"><svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><ellipse cx="12" cy="18.4" rx="7.5" ry="2.8"/><path d="M12 3 a4.2 4.2 0 0 1 4.2 4.2 c0 2.8-4.2 7-4.2 7 s-4.2-4.2-4.2-7 A4.2 4.2 0 0 1 12 3 Z"/><circle cx="12" cy="7.2" r="1.5"/></svg></button>
  <button id="btn-newfolder" class="iconbtn" onclick="moveFacesToNew()" data-tip="移到新文件夹" style="display:none;"><svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><path d="M12 5 v14 M5 12 h14"/></svg></button>
  <button id="btn-merge" class="iconbtn" onclick="mergeSelected()" data-tip="合并选中到主簇" style="display:none;"><svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M4 4 L12 12 M20 4 L12 12 M12 12 V20 M8 16 L12 20 L16 16"/></svg></button>
  <button id="btn-del" class="iconbtn" onclick="deleteSelected()" data-tip="删除选中" style="display:none;"><svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M4 7 h16 M10 7 V5 a1 1 0 0 1 1-1 h2 a1 1 0 0 1 1 1 v2 M6.5 7 l1 13 h9 l1-13 M10 11 v5 M14 11 v5"/></svg></button>
</div>
<div id="grid" class="groups"></div>
<div id="status"></div>
<div id="hover-preview-wrap"><div id="hover-preview"></div></div>
<div id="modal-mask"><div id="modal-card"><div id="modal-text"></div><input id="modal-input" type="text"><div id="modal-btns"><button id="modal-cancel">取消</button><button id="modal-ok">确认</button></div></div></div>
<div id="ctxmenu"></div>
<div id="lightbox"><img id="lb-img" alt=""></div>
<script>
const API = window.__API__ || '';
let currentGroup = null;
let selected = new Set();
let lastClickIdx = -1;
let allFaces = [];
let groupSel = new Set();
let lastGroup = null;
let allGroups = [];
let lastGroupIdx = -1;
let groupCards = [];
const physKeys = { ctrl: false, shift: false, meta: false };

async function api(path, opts) {
  const r = await fetch(API + path, opts);
  return r.json();
}

// ── 首页：分组列表 ──
async function loadGroups() {
  currentGroup = null;
  selected.clear();
  groupSel.clear(); lastGroup = null; lastGroupIdx = -1;
  document.getElementById('btn-back').style.display = 'none';
  document.getElementById('btn-del').style.display = 'none';
  document.getElementById('sel-count').style.display = 'none';
  document.getElementById('zoomctl').style.display = 'none';
  document.getElementById('btn-merge-groups').style.display = 'none';
  const btg = document.getElementById('btn-trash-groups');
  btg.style.display = 'flex';
  btg.classList.add('dim');
  const bng = document.getElementById('btn-newfolder-groups');
  bng.style.display = 'flex';
  bng.classList.add('dim');
  document.getElementById('title').textContent = 'FaceID Groups';
  document.getElementById('grid').className = 'groups';
  const data = await api('/api/groups');
  const grid = document.getElementById('grid');
  grid.innerHTML = '';
  const groups = data.groups || [];
  groups.sort((a, b) => b.count - a.count);
  allGroups = groups;
  groupCards = [];
  groups.forEach((g, i) => {
    const card = document.createElement('div');
    card.className = 'group-card';
    card.innerHTML = `<img src="/api/thumb/${encodeURIComponent(g.name)}" loading="lazy">
      <div class="glabel"><div class="gname">${g.name}</div><div class="gcount">${g.count} 张</div></div>`;
    card.onclick = () => {
      if (physKeys.ctrl || physKeys.meta) toggleGroupSel(g.name, card, i);
      else if (physKeys.shift && groupSel.size > 0) rangeGroupSel(i);
      else if (groupSel.size > 0) toggleGroupSel(g.name, card, i);
      else openGroup(g.name);
    };
    card.onmouseenter = (e) => startHoverPreview(g.name, e);
    card.onmouseleave = () => scheduleHpClose();
    card.addEventListener('contextmenu', (e) => showGroupCtx(e, g.name));
    groupCards.push({ name: g.name, el: card });
    grid.appendChild(card);
  });
  document.getElementById('status').textContent = `${groups.length} 组`;
  const bm = document.getElementById('btn-merge');
  if (bm) bm.style.display = 'none';
}

let hpTimer = null, hpCloseTimer = null, hpHideTimer = null, hpOnPv = false;
const HP_SIZES = [90, 130, 180, 250, 340, 480, 640];
const hpCache = {};

function startHoverPreview(name, e) {
  clearTimeout(hpCloseTimer);
  clearTimeout(hpTimer);
  hpTimer = setTimeout(() => showHoverPreview(name, e), 550);
}

function endHoverPreview() {
  clearTimeout(hpTimer);
  clearTimeout(hpCloseTimer);
  hpOnPv = false;
  const pv = document.getElementById('hover-preview');
  const wp = document.getElementById('hover-preview-wrap');
  if (pv && wp) {
    pv.classList.remove('hp-show');
    clearTimeout(hpHideTimer);
    hpHideTimer = setTimeout(() => { if (!hpOnPv) { pv.style.display = 'none'; wp.style.display = 'none'; } }, 150);
  }
}

function scheduleHpClose() {
  clearTimeout(hpCloseTimer);
  hpCloseTimer = setTimeout(() => { if (!hpOnPv) endHoverPreview(); }, 150);
}

function squirclePath(w, h, d, n) {
  d = Math.min(d, w / 2, h / 2);
  const seg = 10;
  const pts = [];
  const corner = (cx, cy, a0, a1) => {
    for (let i = 0; i <= seg; i++) {
      const t = a0 + (a1 - a0) * i / seg;
      const c = Math.cos(t), s = Math.sin(t);
      const x = cx + d * Math.sign(c) * Math.pow(Math.abs(c), 2 / n);
      const y = cy + d * Math.sign(s) * Math.pow(Math.abs(s), 2 / n);
      pts.push((pts.length === 0 ? 'M' : 'L') + x.toFixed(2) + ' ' + y.toFixed(2));
    }
  };
  corner(d, d, Math.PI, Math.PI * 1.5);
  corner(w - d, d, Math.PI * 1.5, Math.PI * 2);
  corner(w - d, h - d, 0, Math.PI * 0.5);
  corner(d, h - d, Math.PI * 0.5, Math.PI);
  return pts.join(' ') + ' Z';
}

async function showHoverPreview(name, e) {
  const pv = document.getElementById('hover-preview');
  if (!pv) return;
  let faces = hpCache[name];
  if (!faces) {
    try { const data = await api(`/api/group/${encodeURIComponent(name)}`); faces = hpCache[name] = (data.faces || []); }
    catch (err) { return; }
  }
  if (!faces.length) return;
  const cells = faces.map(f => `<img src="/api/face/${encodeURIComponent(name)}/${encodeURIComponent(f)}?t=1" loading="lazy">`).join('');
  pv.innerHTML = `<div class="hp-grid">${cells}</div>`;
  pv.querySelector('.hp-grid').onclick = () => openGroup(name);
  pv.onmouseenter = () => { hpOnPv = true; clearTimeout(hpCloseTimer); };
  pv.onmouseleave = () => { hpOnPv = false; endHoverPreview(); };
  const n = faces.length;
  const slv = document.getElementById('size-slider');
  const cell = (slv && slv.value !== '' && HP_SIZES[Number(slv.value)]) ? HP_SIZES[Number(slv.value)] : 96;
  const cgap = 4, pad = 24;
  const W_max = Math.min(1080, innerWidth * 0.92), H_max = Math.min(760, innerHeight * 0.84);
  const colsMax = Math.max(1, Math.floor((W_max - pad - cgap) / (cell + cgap)));
  const cols = Math.min(colsMax, n);
  const rows = Math.ceil(n / cols);
  const w = Math.min(W_max, cols * (cell + cgap) - cgap + pad);
  const h = Math.min(H_max, rows * (cell + cgap) - cgap + pad + 2);
  pv.querySelector('.hp-grid').style.gridTemplateColumns = `repeat(${cols}, minmax(0, 1fr))`;
  const gap = 3;
  let x = e.clientX + gap, y = e.clientY + gap;
  let flipX = false, flipY = false;
  if (x + w > innerWidth - 6) { x = Math.max(6, e.clientX - w - gap); flipX = true; }
  if (y + h > innerHeight - 6) { y = Math.max(6, e.clientY - h - gap); flipY = true; }
  const wp = document.getElementById('hover-preview-wrap');
  pv.style.transformOrigin = (flipX ? 'right' : 'left') + ' ' + (flipY ? 'bottom' : 'top');
  wp.style.width = w + 'px'; wp.style.height = h + 'px';
  wp.style.left = x + 'px'; wp.style.top = y + 'px';
  pv.style.clipPath = `path("${squirclePath(w, h, 26, 5)}")`;
  clearTimeout(hpHideTimer);
  wp.style.display = 'block';
  pv.style.display = 'flex';
  void pv.offsetHeight;
  pv.classList.add('hp-show');
}

function toggleGroupSel(name, card, idx) {
  if (groupSel.has(name)) { groupSel.delete(name); card.classList.remove('gsel'); }
  else { groupSel.add(name); lastGroup = name; card.classList.add('gsel'); if (idx !== undefined) lastGroupIdx = idx; }
  updateGroupSelUI();
}

function rangeGroupSel(idx) {
  if (lastGroupIdx < 0) lastGroupIdx = idx;
  const a = Math.min(lastGroupIdx, idx), b = Math.max(lastGroupIdx, idx);
  for (let i = a; i <= b; i++) {
    const gc = groupCards[i];
    if (!gc) continue;
    groupSel.add(gc.name);
    gc.el.classList.add('gsel');
  }
  lastGroupIdx = idx;
  updateGroupSelUI();
}

function updateGroupSelUI() {
  const target = mainClusterName();
  const bm = document.getElementById('btn-merge-groups');
  if (bm) {
    const can = groupSel.size >= 1 && target && Array.from(groupSel).some(n => n !== target);
    bm.style.display = can ? 'flex' : 'none';
    bm.setAttribute('data-tip', `并入 ${target}`);
  }
  const btg = document.getElementById('btn-trash-groups');
  if (btg) {
    const hasSrc = groupSel.size > 0 && Array.from(groupSel).some(n => n !== 'faceid_noise');
    btg.classList.toggle('dim', !hasSrc);
  }
  const bng = document.getElementById('btn-newfolder-groups');
  if (bng) bng.classList.toggle('dim', groupSel.size === 0);
  const sc = document.getElementById('sel-count');
  if (groupSel.size > 0) { sc.textContent = `已选 ${groupSel.size} 组 → 将并入 ${target || '主簇'}`; sc.style.display = 'block'; }
  else sc.style.display = 'none';
}

function mainClusterName() {
  if (allGroups.some(g => g.name === 'faceid_group_000')) return 'faceid_group_000';
  return allGroups.length > 0 ? allGroups[0].name : null;
}

function shortName(n) { return n ? n.replace('faceid_', '') : ''; }

async function mergeGroups() {
  const target = mainClusterName();
  if (!target) return;
  const sources = Array.from(groupSel).filter(n => n !== target);
  if (sources.length === 0) return;
  if (!await showConfirm(`把 ${sources.length} 个组（${sources.join('、')}）并入 ${target}/ ？`)) return;
  const r = await api('/api/merge', {
    method: 'POST', headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({ groups: sources, target: target })
  });
  if (r.moved !== undefined) { console.log(`组级合并 ${r.moved} 张`); loadGroups(); }
}

async function trashGroups() {
  const sources = Array.from(groupSel);
  if (sources.length === 0) return;
  const total = sources.reduce((a, nm) => a + ((allGroups.find(g => g.name === nm) || {}).count || 0), 0);
  if (!await showConfirm(`永久删除 ${sources.length} 个组（${sources.join('、')}）共 ${total} 张？（不可恢复）`)) return;
  const r = await api('/api/delete-groups', {
    method: 'POST', headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({ groups: sources })
  });
  if (r.deleted !== undefined) { console.log(`永久删除 ${r.deleted} 张`); loadGroups(); }
}

async function groupsToNew() {
  const sources = Array.from(groupSel);
  if (sources.length === 0) return;
  const target = await showPrompt(`把 ${sources.length} 个组（${sources.join('、')}）的全部图移到新文件夹，文件夹名：`, 'faceid_group_new');
  if (!target) return;
  const r = await api('/api/merge', {
    method: 'POST', headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({ groups: sources, target: target })
  });
  if (r.moved !== undefined) { console.log(`移入新夹 ${r.target} 共 ${r.moved} 张`); loadGroups(); }
}

// ── 组内视图：全部脸图 ──
async function openGroup(name) {
  currentGroup = name;
  selected.clear();
  lastClickIdx = -1;
  document.getElementById('btn-merge-groups').style.display = 'none';
  document.getElementById('btn-trash-groups').style.display = 'none';
  document.getElementById('btn-newfolder-groups').style.display = 'none';
  endHoverPreview();
  document.getElementById('btn-back').style.display = 'flex';
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
    el.title = f;
    el.innerHTML = `<img src="/api/face/${encodeURIComponent(name)}/${encodeURIComponent(f)}?t=1" loading="lazy">`;
    el.addEventListener('click', (e) => faceClick(e, i));
    el.addEventListener('dblclick', () => openLightbox(el.querySelector('img'), `/api/face/${encodeURIComponent(name)}/${encodeURIComponent(f)}`));
    el.addEventListener('contextmenu', (e) => showCtxMenu(e, name, f));
    grid.appendChild(el);
  });
  document.getElementById('status').textContent = `${allFaces.length} 张`;
  if (lmShow) document.querySelectorAll('.face-item').forEach(el => lmObserver.observe(el));
  initSlider();
  updateSelUI();
}

function initSlider() {
  const zc = document.getElementById('zoomctl');
  const sl = document.getElementById('size-slider');
  document.documentElement.style.removeProperty('--fs');
  zc.style.display = 'flex';
  requestAnimationFrame(() => {
    const first = document.querySelector('.face-item');
    if (first) {
      const w = Math.round(first.getBoundingClientRect().width);
      sl.value = nearestSizeIdx(w);
    }
  });
}

function nearestSizeIdx(w) {
  let best = 0, bd = Infinity;
  HP_SIZES.forEach((s, i) => { const d = Math.abs(s - w); if (d < bd) { bd = d; best = i; } });
  return best;
}

function applySize(idx) {
  document.documentElement.style.setProperty('--fs', HP_SIZES[idx] + 'px');
}

function stepSize(d) {
  const sl = document.getElementById('size-slider');
  sl.value = Math.max(0, Math.min(HP_SIZES.length - 1, Number(sl.value) + d));
  applySize(sl.value);
}

function resetAutoSize() {
  document.documentElement.style.removeProperty('--fs');
}

function faceClick(e, idx) {
  e.stopPropagation();
  const el = e.currentTarget;
  if (physKeys.shift && lastClickIdx >= 0) {
    // Shift 范围选
    const a = Math.min(lastClickIdx, idx), b = Math.max(lastClickIdx, idx);
    for (let i = a; i <= b; i++) selected.add(i);
  } else if (physKeys.ctrl || physKeys.meta) {
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
  const blm = document.getElementById('btn-landmarks');
  if (blm) blm.style.display = currentGroup ? 'flex' : 'none';
  const bd = document.getElementById('btn-del');
  bd.style.display = currentGroup ? 'flex' : 'none';
  bd.classList.toggle('dim', n === 0);
  const bnf = document.getElementById('btn-newfolder');
  if (bnf) {
    bnf.style.display = currentGroup ? 'flex' : 'none';
    bnf.classList.toggle('dim', n === 0);
  }
  // 合并按钮：在组内视图常驻显示，未选中时置灰
  const bm = document.getElementById('btn-merge');
  if (bm) {
    bm.style.display = (currentGroup && currentGroup !== mainClusterName()) ? 'flex' : 'none';
    bm.classList.toggle('dim', n === 0);
  }
}

async function mergeSelected() {
  if (!currentGroup || selected.size === 0) return;
  const names = Array.from(selected).map(i => allFaces[i]);
  const target = mainClusterName() || 'faceid_group_000';
  if (!await showConfirm(`将 ${names.length} 张从 ${currentGroup} 合并到 ${target}/ ？`)) return;
  const r = await api('/api/move-faces', {
    method: 'POST', headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({ group: currentGroup, faces: names, target: target })
  });
  if (r.moved !== undefined) {
    console.log(`合并 ${r.moved} 张`);
    openGroup(currentGroup); // 刷新当前视图
  }
}

async function deleteSelected() {
  if (!currentGroup || selected.size === 0) return;
  const names = Array.from(selected).map(i => allFaces[i]);
  const msg = `永久删除 ${names.length} 张？（不可恢复）`;
  if (!await showConfirm(msg)) return;
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
  if (modalOpen) return;
  if (e.key === 'Escape') { if (document.getElementById('lightbox').style.display === 'block') { closeLightbox(); return; } hideCtxMenu(); selected.clear(); updateSelUI(); document.querySelectorAll('.face-item.selected').forEach(el => el.classList.remove('selected')); groupSel.clear(); lastGroup = null; document.querySelectorAll('.group-card.gsel').forEach(el => el.classList.remove('gsel')); updateGroupSelUI(); endHoverPreview(); }
  if (e.key === 'Backspace' && currentGroup) goBack();
  if ((e.key === 'Delete' || e.key === 'Backspace') && selected.size > 0 && currentGroup) { e.preventDefault(); deleteSelected(); }
});

document.addEventListener('keydown', (e) => {
  if (e.key === 'Shift') physKeys.shift = true;
  else if (e.key === 'Control') physKeys.ctrl = true;
  else if (e.key === 'Meta') physKeys.meta = true;
});
document.addEventListener('keyup', (e) => {
  if (e.key === 'Shift') physKeys.shift = false;
  else if (e.key === 'Control') physKeys.ctrl = false;
  else if (e.key === 'Meta') physKeys.meta = false;
});
window.addEventListener('blur', () => { physKeys.shift = physKeys.ctrl = physKeys.meta = false; });

let lbFromRect = null;

function openLightbox(thumbEl, bigSrc) {
  const lb = document.getElementById('lightbox');
  const im = document.getElementById('lb-img');
  lbFromRect = thumbEl.getBoundingClientRect();
  im.src = bigSrc;
  lb.style.display = 'block';
  lb.style.opacity = '0';
  const go = () => {
    const tr = im.getBoundingClientRect();
    if (!tr.width || !tr.height) return;
    const sx = lbFromRect.width / tr.width, sy = lbFromRect.height / tr.height;
    const s0 = Math.max(sx, sy) * 0.97;
    const dx = (lbFromRect.left + lbFromRect.width / 2) - (tr.left + tr.width / 2);
    const dy = (lbFromRect.top + lbFromRect.height / 2) - (tr.top + tr.height / 2);
    im.style.transition = 'none';
    im.style.transform = `translate(calc(-50% + ${dx}px), calc(-50% + ${dy}px)) scale(${s0})`;
    void im.offsetHeight;
    lb.style.opacity = '1';
    im.style.transition = 'transform .34s cubic-bezier(.2,.75,.3,1.1)';
    im.style.transform = 'translate(-50%,-50%) scale(1.015)';
    setTimeout(() => {
      im.style.transition = 'transform .18s cubic-bezier(.3,.7,.4,1)';
      im.style.transform = 'translate(-50%,-50%) scale(1)';
    }, 350);
  };
  if (im.complete && im.naturalWidth) go(); else im.onload = go;
}

function closeLightbox() {
  const lb = document.getElementById('lightbox');
  const im = document.getElementById('lb-img');
  lb.style.opacity = '0';
  const tr = im.getBoundingClientRect();
  if (lbFromRect && tr.width && tr.height) {
    const sx = lbFromRect.width / tr.width, sy = lbFromRect.height / tr.height;
    const s = Math.max(sx, sy);
    const dx = (lbFromRect.left + lbFromRect.width / 2) - (tr.left + tr.width / 2);
    const dy = (lbFromRect.top + lbFromRect.height / 2) - (tr.top + tr.height / 2);
    im.style.transition = 'transform .2s cubic-bezier(.4,.6,.4,1)';
    im.style.transform = `translate(calc(-50% + ${dx}px), calc(-50% + ${dy}px)) scale(${s})`;
  }
  setTimeout(() => { lb.style.display = 'none'; im.style.transform = 'translate(-50%,-50%)'; }, 210);
}

function showCtxMenu(e, group, face) {
  e.preventDefault();
  e.stopPropagation();
  window._ctx = { group, face };
  const subs = (allGroups || []).filter(g => !/^faceid_group_[0-9]+$/.test(g.name) && g.name !== 'faceid_noise' && g.name !== 'merged' && !g.name.startsWith('trash:'));
  const subHtml = subs.length
    ? subs.map(g => `<div class="ctx-item" data-target="${g.name}"><span>${g.name}</span></div>`).join('')
    : '<div class="ctx-item" style="color:#888;cursor:default;">先用 ＋ 新建文件夹</div>';
  openCtxMenu(e.clientX, e.clientY, `
    <div class="ctx-item" data-act="reveal"><svg width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M3 7 q0-2 2-2 h4 l2 2 h8 q2 0 2 2 v9 q0 2-2 2 H5 q-2 0-2-2 Z"/></svg><span>在文件夹中显示</span></div>
    <div class="ctx-item ctx-has-sub"><svg width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M3 7 q0-2 2-2 h4 l2 2 h8 q2 0 2 2 v9 q0 2-2 2 H5 q-2 0-2-2 Z"/><path d="M9 12 h6 M13 10 l2 2 -2 2"/></svg><span>移动到文件夹</span><svg class="chev" width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><path d="M9 5 l7 7 -7 7"/></svg><div id="ctx-sub">${subHtml}</div></div>
  `);
}

function showGroupCtx(e, name) {
  e.preventDefault();
  e.stopPropagation();
  window._ctxGroup = name;
  openCtxMenu(e.clientX, e.clientY, `
    <div class="ctx-item" data-act="rename"><svg width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 20 h9"/><path d="M16.5 3.5 a2.1 2.1 0 0 1 3 3 L7 19 l-4 1 1-4 Z"/></svg><span>重命名</span></div>
  `);
}

function openCtxMenu(x, y, html) {
  const m = document.getElementById('ctxmenu');
  m.innerHTML = html;
  m.style.display = 'block';
  const mw = m.offsetWidth, mh = m.offsetHeight;
  if (x + mw > innerWidth - 6) x = Math.max(6, innerWidth - mw - 6);
  if (y + mh > innerHeight - 6) y = Math.max(6, innerHeight - mh - 6);
  m.style.left = x + 'px'; m.style.top = y + 'px';
}

async function doMoveFaces(target) {
  const c = window._ctx;
  if (!c) return;
  const names = selected.size ? Array.from(selected).map(i => allFaces[i]) : [c.face];
  const r = await api('/api/move-faces', {
    method: 'POST', headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({ group: c.group, faces: names, target: target })
  });
  if (r.moved !== undefined) { selected.clear(); openGroup(c.group); }
}

async function renameGroupAction() {
  const name = window._ctxGroup;
  if (!name || name.includes(':')) return;
  const nn = await showPrompt(`重命名「${name}」为：`, name);
  if (!nn || nn === name) return;
  const r = await api('/api/rename-group', {
    method: 'POST', headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({ group: name, new: nn })
  });
  if (r.ok) loadGroups();
  else document.getElementById('status').textContent = `重命名失败: ${r.error || ''}`;
}

function hideCtxMenu() { document.getElementById('ctxmenu').style.display = 'none'; }

let lmShow = false;
const lmCache = {};
const lmObserver = new IntersectionObserver((ents) => {
  for (const en of ents) {
    if (!en.isIntersecting) continue;
    lmObserver.unobserve(en.target);
    drawLandmarks(en.target);
  }
}, { rootMargin: '300px' });

async function drawLandmarks(el) {
  const idx = Number(el.dataset.idx);
  const f = allFaces[idx];
  if (!f) return;
  const ckey = currentGroup + '/' + f;
  let lms = lmCache[ckey];
  if (lms === undefined) {
    try {
      const r = await api(`/api/landmarks?g=${encodeURIComponent(currentGroup)}&f=${encodeURIComponent(f)}`);
      lms = lmCache[ckey] = (r.ok && r.landmarks && r.landmarks.length) ? { pts: r.landmarks, w: r.w, h: r.h } : null;
    } catch (err) { lms = lmCache[ckey] = null; }
  }
  if (!lms || !lmShow) return;
  const img = el.querySelector('img');
  if (!img) return;
  if (!img.naturalWidth) { img.addEventListener('load', () => drawLandmarks(el), { once: true }); return; }
  let cv = el.querySelector('canvas.lm-cv');
  if (!cv) { cv = document.createElement('canvas'); cv.className = 'lm-cv'; el.appendChild(cv); }
  const W = el.clientWidth, H = el.clientHeight;
  cv.width = W; cv.height = H;
  const ctx = cv.getContext('2d');
  const sx = W / lms.w, sy = H / lms.h;
  ctx.clearRect(0, 0, W, H);
  const r = Math.max(1.4, W * 0.014);
  ctx.fillStyle = '#ff1744';
  for (const pt of lms.pts) {
    ctx.beginPath();
    ctx.arc(pt[0] * sx, pt[1] * sy, r, 0, 6.2832);
    ctx.fill();
  }
}

function toggleLandmarks() {
  lmShow = !lmShow;
  document.getElementById('btn-landmarks').classList.toggle('active', lmShow);
  if (!currentGroup) return;
  if (!lmShow) {
    document.querySelectorAll('canvas.lm-cv').forEach(c => c.remove());
    return;
  }
  document.querySelectorAll('.face-item').forEach(el => lmObserver.observe(el));
}

let modalOpen = false;

function showConfirm(text) {
  return new Promise((resolve) => {
    const mask = document.getElementById('modal-mask');
    const card = document.getElementById('modal-card');
    const inp = document.getElementById('modal-input');
    inp.style.display = 'none';
    document.getElementById('modal-text').textContent = text;
    modalOpen = true;
    mask.style.display = 'flex';
    void mask.offsetHeight;
    mask.style.opacity = '1';
    card.style.transform = 'scale(1)';
    const close = (val) => {
      mask.style.opacity = '0';
      card.style.transform = 'scale(.95)';
      setTimeout(() => { mask.style.display = 'none'; }, 160);
      document.getElementById('modal-ok').onclick = null;
      document.getElementById('modal-cancel').onclick = null;
      mask.onclick = null;
      document.removeEventListener('keydown', onKey);
      modalOpen = false;
      resolve(val);
    };
    const onKey = (e) => { if (e.key === 'Escape') close(false); else if (e.key === 'Enter') close(true); };
    document.getElementById('modal-ok').onclick = () => close(true);
    document.getElementById('modal-cancel').onclick = () => close(false);
    mask.onclick = (e) => { if (e.target === mask) close(false); };
    document.addEventListener('keydown', onKey);
    document.getElementById('modal-ok').focus();
  });
}

function showPrompt(text, defval = '') {
  return new Promise((resolve) => {
    const mask = document.getElementById('modal-mask');
    const card = document.getElementById('modal-card');
    const inp = document.getElementById('modal-input');
    document.getElementById('modal-text').textContent = text;
    inp.style.display = 'block';
    inp.value = defval;
    modalOpen = true;
    mask.style.display = 'flex';
    void mask.offsetHeight;
    mask.style.opacity = '1';
    card.style.transform = 'scale(1)';
    const close = (val) => {
      mask.style.opacity = '0';
      card.style.transform = 'scale(.95)';
      setTimeout(() => { mask.style.display = 'none'; inp.style.display = 'none'; }, 160);
      document.getElementById('modal-ok').onclick = null;
      document.getElementById('modal-cancel').onclick = null;
      mask.onclick = null;
      document.removeEventListener('keydown', onKey);
      modalOpen = false;
      resolve(val);
    };
    const submit = () => { const v = inp.value.trim(); if (v) close(v); };
    const onKey = (e) => { if (e.key === 'Escape') close(null); else if (e.key === 'Enter') submit(); };
    document.getElementById('modal-ok').onclick = submit;
    document.getElementById('modal-cancel').onclick = () => close(null);
    mask.onclick = (e) => { if (e.target === mask) close(null); };
    document.addEventListener('keydown', onKey);
    inp.focus();
    inp.select();
  });
}

async function moveFacesToNew() {
  if (!currentGroup || selected.size === 0) return;
  const names = Array.from(selected).map(i => allFaces[i]);
  const target = await showPrompt(`把 ${names.length} 张移到新文件夹，文件夹名：`, 'faceid_group_new');
  if (!target) return;
  const r = await api('/api/move-faces', {
    method: 'POST', headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({ group: currentGroup, faces: names, target: target })
  });
  if (r.moved !== undefined) { console.log(`移动 ${r.moved} 张到 ${r.target}`); openGroup(currentGroup); }
}

document.getElementById('ctxmenu').onclick = async (e) => {
  const it = e.target.closest('.ctx-item');
  if (!it) return;
  if (it.dataset.target) { hideCtxMenu(); doMoveFaces(it.dataset.target); return; }
  const act = it.dataset.act;
  if (!act) return;
  hideCtxMenu();
  if (act === 'reveal') {
    const c = window._ctx;
    if (c) await api('/api/reveal', { method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(c) });
  } else if (act === 'rename') {
    renameGroupAction();
  }
};
document.addEventListener('mousedown', (e) => {
  const m = document.getElementById('ctxmenu');
  if (m.style.display === 'block' && !m.contains(e.target)) hideCtxMenu();
});
document.getElementById('lightbox').onclick = () => closeLightbox();
document.getElementById('zc-minus').onclick = () => stepSize(-1);
document.getElementById('zc-plus').onclick = () => stepSize(1);
document.getElementById('zc-grid').onclick = () => resetAutoSize();
loadGroups();
</script>
</body>
</html>"""


def resolve_group(root, name):
    """组名→目录；trash:<名> 映射到 .trash/<名> 子夹"""
    if name.startswith('trash:'):
        sub = re.sub(r'[<>:"/\\|?*\x00-\x1f]+', '_', name[6:]).strip().rstrip(' .')
        return os.path.join(root, '.trash', sub)
    return os.path.join(root, name)


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
        parsed = urlparse(self.path)
        p = unquote(parsed.path)
        want_thumb = 't' in parse_qs(parsed.query)

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
            trash_dir = os.path.join(self.server.root, '.trash')
            if os.path.isdir(trash_dir):
                for d in sorted(os.listdir(trash_dir)):
                    td = os.path.join(trash_dir, d)
                    if not os.path.isdir(td):
                        continue
                    imgs = [f for f in os.listdir(td) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                    if imgs:
                        groups.append({'name': 'trash:' + d, 'count': len(imgs)})
            self._j({'groups': groups})
        elif p.startswith('/api/thumb/'):
            # 返回该组第一张图的缩略图（缓存到 <root>/.thumbs/）
            group = p.split('/')[-1]
            gd = resolve_group(self.server.root, group)
            if os.path.isdir(gd):
                imgs = sorted(f for f in os.listdir(gd) if f.lower().endswith(('.png', '.jpg', '.jpeg')))
                if imgs:
                    fp = os.path.join(gd, imgs[0])
                    try:
                        data = thumb_bytes(fp, self.server.root, 'thumb_' + group)
                        self.send_response(200)
                        self.send_header('Content-Type', 'image/jpeg')
                        self.send_header('Content-Length', str(len(data)))
                        self.end_headers()
                        self.wfile.write(data)
                        return
                    except Exception:
                        pass
            self.send_response(404)
            self.end_headers()
        elif p.startswith('/api/group/'):
            group = p.split('/')[-1]
            gd = resolve_group(self.server.root, group)
            faces = sorted(f for f in os.listdir(gd) if f.lower().endswith(('.png', '.jpg', '.jpeg'))) if os.path.isdir(gd) else []
            self._j({'faces': faces})
        elif p.startswith('/api/landmarks'):
            qs = parse_qs(urlparse(self.path).query)
            group = qs.get('g', [''])[0]
            face = qs.get('f', [''])[0]
            fp = os.path.join(resolve_group(self.server.root, group), face)
            root_real = os.path.realpath(self.server.root)
            fp_real = os.path.realpath(fp)
            if not fp_real.startswith(root_real) or not os.path.isfile(fp_real):
                self._j({'ok': False})
                return
            res = _read_dfld_landmarks(fp_real)
            if res is None:
                self._j({'ok': False, 'landmarks': [], 'w': 0, 'h': 0})
            else:
                self._j({'ok': True, 'landmarks': res[0], 'w': res[1], 'h': res[2]})
        elif p.startswith('/api/face/'):
            parts = p.split('/')
            group = parts[3]
            face = '/'.join(parts[4:])
            fp = os.path.join(resolve_group(self.server.root, group), face)
            if os.path.isfile(fp):
                if want_thumb:
                    try:
                        import hashlib
                        key = 'face_' + hashlib.md5((group + '/' + face).encode('utf-8', 'surrogatepass')).hexdigest()
                        data = thumb_bytes(fp, self.server.root, key)
                        self.send_response(200)
                        self.send_header('Content-Type', 'image/jpeg')
                        self.send_header('Content-Length', str(len(data)))
                        self.end_headers()
                        self.wfile.write(data)
                        return
                    except Exception:
                        pass
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
            n = 0
            for f in faces:
                src = os.path.join(resolve_group(self.server.root, group), f)
                if os.path.isfile(src):
                    os.remove(src)
                    n += 1
            self._j({'deleted': n})
        elif p == '/api/delete-groups':
            cl = int(self.headers.get('Content-Length', 0))
            data = json.loads(self.rfile.read(cl))
            groups = data.get('groups', [])
            n = 0
            for g in groups:
                gd = resolve_group(self.server.root, g)
                if not os.path.isdir(gd):
                    continue
                for f in os.listdir(gd):
                    fp = os.path.join(gd, f)
                    if os.path.isfile(fp):
                        os.remove(fp)
                        n += 1
                try:
                    os.rmdir(gd)
                except OSError:
                    pass
            self._j({'deleted': n})
        elif p == '/api/reveal':
            cl = int(self.headers.get('Content-Length', 0))
            data = json.loads(self.rfile.read(cl))
            group = data.get('group', '')
            face = data.get('face', '')
            fp = os.path.join(resolve_group(self.server.root, group), face)
            root_real = os.path.realpath(self.server.root)
            fp_real = os.path.realpath(fp)
            if not fp_real.startswith(root_real) or not os.path.isfile(fp_real):
                self._j({'ok': False, 'error': 'not found'})
                return
            import subprocess
            try:
                subprocess.Popen(['explorer.exe', '/select,', fp_real])
                self._j({'ok': True})
            except Exception as e:
                self._j({'ok': False, 'error': str(e)})
        elif p == '/api/rename-group':
            cl = int(self.headers.get('Content-Length', 0))
            data = json.loads(self.rfile.read(cl))
            group = data.get('group', '')
            import re as _re3
            new = _re3.sub(r'[<>:"/\\|?*\x00-\x1f]+', '_', data.get('new', '')).strip().rstrip(' .')
            if not new or new.startswith('.') or ':' in group or group.startswith('.'):
                self._j({'ok': False, 'error': 'invalid name'})
                return
            src = resolve_group(self.server.root, group)
            dst = os.path.join(self.server.root, new)
            if not os.path.isdir(src) or os.path.exists(dst):
                self._j({'ok': False, 'error': 'invalid'})
                return
            try:
                os.rename(src, dst)
                self._j({'ok': True})
            except OSError as e2:
                self._j({'ok': False, 'error': str(e2)})
        elif p == '/api/move-faces':
            cl = int(self.headers.get('Content-Length', 0))
            data = json.loads(self.rfile.read(cl))
            group = data.get('group', '')
            faces = data.get('faces', [])
            import re as _re
            target = _re.sub(r'[<>:"/\\|?*\x00-\x1f]+', '_', data.get('target', '')).strip().rstrip(' .')
            if not target or target.startswith('.'):
                self._j({'moved': 0, 'error': 'invalid name'})
                return
            td = os.path.join(self.server.root, target)
            os.makedirs(td, exist_ok=True)
            n = 0
            for f in faces:
                src = os.path.join(resolve_group(self.server.root, group), f)
                dst = os.path.join(td, f)
                if os.path.isfile(src):
                    if os.path.exists(dst):
                        stem, ext = os.path.splitext(f)
                        dst = os.path.join(td, f"{stem}_{n}{ext}")
                    shutil.move(src, dst)
                    n += 1
            self._j({'moved': n, 'target': target})
        elif p == '/api/merge':
            cl = int(self.headers.get('Content-Length', 0))
            data = json.loads(self.rfile.read(cl))
            src_groups = data.get('groups', [])
            target = data.get('target', '')
            import re as _re2
            target = _re2.sub(r'[<>:"/\\|?*\x00-\x1f]+', '_', target).strip().rstrip(' .')
            if not target or target.startswith('.'):
                self._j({'moved': 0, 'error': 'invalid name'})
                return
            # 合并：把 src_groups 的全部图片移到 target 目录
            moved = 0
            for sg in src_groups:
                sd = resolve_group(self.server.root, sg)
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


def _read_dfld_landmarks(fp):
    """轻量解析 PNG dFLd 块，返回 (landmarks, 原图宽, 原图高) 或 None"""
    import pickle as _pickle
    import struct as _struct
    try:
        with open(fp, 'rb') as f:
            data = f.read()
        if not data.startswith(b'\x89PNG\r\n\x1a\n'):
            return None
        ow = _struct.unpack('>I', data[16:20])[0]
        oh = _struct.unpack('>I', data[20:24])[0]
        pos = 8
        while pos + 8 <= len(data):
            ln = _struct.unpack('>I', data[pos:pos + 4])[0]
            ctype = data[pos + 4:pos + 8]
            if ctype == b'IDAT':
                return None
            if ctype == b'dFLd':
                d = _pickle.loads(data[pos + 8:pos + 8 + ln])
                lms = d.get('landmarks')
                if lms is None:
                    return None
                if hasattr(lms, 'tolist'):
                    lms = lms.tolist()
                return (lms, ow, oh)
            pos += 12 + ln
    except Exception:
        return None
    return None


def thumb_bytes(fp, root, key, size=320):
    """读取或生成 320px JPEG 缩略图，原子缓存到 <root>/.thumbs/<key>.jpg"""
    cache_dir = os.path.join(root, '.thumbs')
    cp = os.path.join(cache_dir, key + '.jpg')
    if os.path.isfile(cp):
        try:
            os.utime(cp, None)
        except OSError:
            pass
        with open(cp, 'rb') as f:
            return f.read()
    import io
    from PIL import Image
    with Image.open(fp) as im:
        im = im.convert('RGB')
        im.thumbnail((size, size))
        buf = io.BytesIO()
        im.save(buf, 'JPEG', quality=88)
        data = buf.getvalue()
    os.makedirs(cache_dir, exist_ok=True)
    tmp = cp + f'.{os.getpid()}.tmp'
    with open(tmp, 'wb') as f:
        f.write(data)
    os.replace(tmp, cp)
    return data


def _cleanup_thumbs(root, max_bytes=200 * 1024 * 1024, ttl_days=7):
    """缓存维护：删超期未访问 + 超配额时按最旧淘汰"""
    import time as _time
    td = os.path.join(root, '.thumbs')
    if not os.path.isdir(td):
        return
    now = _time.time()
    removed = 0
    freed = 0
    entries = []
    for fn in os.listdir(td):
        fp = os.path.join(td, fn)
        if not os.path.isfile(fp):
            continue
        try:
            st = os.stat(fp)
        except OSError:
            continue
        if now - st.st_mtime > ttl_days * 86400:
            try:
                os.remove(fp)
                removed += 1
                freed += st.st_size
            except OSError:
                pass
        else:
            entries.append((st.st_mtime, st.st_size, fp))
    total = sum(e[1] for e in entries)
    if total > max_bytes:
        entries.sort()
        for mt, sz, fp in entries:
            if total <= max_bytes:
                break
            try:
                os.remove(fp)
                removed += 1
                freed += sz
                total -= sz
            except OSError:
                pass
    if removed:
        print(f'[thumbs] 清理 {removed} 个缓存，释放 {freed / 1048576:.1f} MB', flush=True)


def _start_thumbs_maintenance(root):
    def _loop():
        import time as _time
        while True:
            try:
                _cleanup_thumbs(root)
            except Exception:
                pass
            _time.sleep(24 * 3600)
    threading.Thread(target=_loop, daemon=True).start()


def _kill_existing_instances():
    """杀掉已存在的 faceid_browser 实例（固定单实例，防多服务并存）"""
    import subprocess as _sp
    me = os.getpid()
    try:
        out = _sp.check_output(
            ['powershell', '-NoProfile', '-Command',
             "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | "
             "Where-Object { $_.CommandLine -match 'faceid_browser' } | "
             "ForEach-Object { $_.ProcessId }"],            timeout=15).decode()
        pids = [int(x) for x in out.split() if x.isdigit()]
    except Exception:
        pids = []
    for pid in pids:
        if pid == me:
            continue
        try:
            _sp.run(['taskkill', '/PID', str(pid), '/T', '/F'], capture_output=True)
        except Exception:
            pass
    if pids:
        import time as _t
        _t.sleep(1.5)


def main():
    ap = argparse.ArgumentParser(description='FaceID 分组浏览器')
    ap.add_argument('--root', required=True, help='含 faceid_group_xxx 子夹的目录')
    ap.add_argument('--port', type=int, default=8104)
    a = ap.parse_args()

    root = os.path.abspath(a.root)
    if not os.path.isdir(root):
        sys.exit(f'目录不存在: {root}')

    _kill_existing_instances()

    server = ThreadingHTTPServer(('127.0.0.1', a.port), Handler)
    server.root = root
    _start_thumbs_maintenance(root)
    print(f'[faceid_browser] http://127.0.0.1:{a.port}  root={root}')
    import webbrowser
    threading.Timer(1.0, lambda: webbrowser.open(f'http://127.0.0.1:{a.port}')).start()
    server.serve_forever()


if __name__ == '__main__':
    import sys
    main()
