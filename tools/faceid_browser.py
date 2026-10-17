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
import collections
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
body { background:#111; color:#ddd; font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif; height:100vh; overflow:hidden; display:flex; flex-direction:column; }
#topbar { position:sticky; top:0; z-index:100; flex:none; background:#1a1a1e; padding:10px 16px; display:flex; align-items:center; gap:6px; border-bottom:1px solid #2a2a30; }
#topbar h1 { font-size:16px; font-weight:600; flex:1; }
.iconbtn.dim { opacity:.35; cursor:default; }
.iconbtn.active { background:rgba(69,162,255,.18); }
#btn-selectmode { width:auto; padding:0 10px; font-size:12px; font-weight:600; }
.face-item.mismatch { border-color:#ff453a !important; }
.face-item.dupe { border-color:#ff1744 !important; }
.face-item .cnum { position:absolute; top:2px; left:2px; z-index:2; font-size:9px; line-height:1.3; padding:1px 4px; border-radius:6px; background:rgba(0,0,0,.6); color:#ff8a8a; pointer-events:none; }
.face-item .cnum.keep { color:#7fc4ff; }
#modal-progress { height:6px; background:#2a2a30; border-radius:3px; margin-bottom:14px; overflow:hidden; }
#modal-progress-fill { height:100%; background:#45a2ff; transition:width .3s; }
canvas.lm-cv { position:absolute; inset:0; width:100%; height:100%; pointer-events:none; }
.iconbtn.dim:hover { background:none; }
.iconbtn.dim:hover::after { opacity:0; }
#btn-back:hover { background:#3a3a42; }
#btn-del:hover { background:#a82828; }
#sel-count { font-size:12px; color:#888; display:none; }
#title { flex:1; min-width:0; margin:0; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
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
.ctx-flip #ctx-sub { left:auto; right:calc(100% + 2px); }
.ctx-item.danger, .ctx-item.danger svg { color:#ff453a; }
.ctx-item .chev { margin-left:auto; color:#8a8a92; }
#grid-sentinel { height: 1px; }
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
#status { position:fixed; bottom:12px; right:12px; background:#2a2a30; padding:8px 16px; font-size:12px; color:#aaa; z-index:100; display:none; }
#hover-preview-wrap { position:fixed; z-index:200; display:none; filter:drop-shadow(0 12px 40px rgba(0,0,0,.55)); }
#hover-preview { background:#1a1a1e; padding:12px; width:100%; height:100%; flex-direction:column; opacity:0; transform:scale(.92); transform-origin:top left; transition:opacity .14s ease, transform .14s cubic-bezier(.2,.7,.3,1); }
#hover-preview.hp-show { opacity:1; transform:scale(1); }
#hover-preview .hp-grid { display:grid; gap:4px; overflow-y:auto; overflow-x:hidden; min-height:0; align-content:start; }
#hover-preview .hp-grid img { width:100%; aspect-ratio:1; object-fit:cover; background:#111; }
#hover-preview .hp-grid::-webkit-scrollbar { width:4px; }
#hover-preview .hp-grid::-webkit-scrollbar-thumb { background:rgba(255,255,255,.35); border-radius:999px; }
#hover-preview .hp-grid::-webkit-scrollbar-thumb:hover { background:rgba(255,255,255,.6); }
#hover-preview .hp-grid::-webkit-scrollbar-track { background:transparent; }
html { height:100%; }
html, body { -webkit-user-select:none; user-select:none; }
#scrollarea { flex:1; min-height:0; overflow-y:auto; overflow-x:hidden; scrollbar-gutter:stable; }
#modal-input { -webkit-user-select:text; user-select:text; }
img { -webkit-user-drag:none; }
</style>
</head>
<body>
<div id="topbar">
  <button id="btn-back" class="iconbtn" onclick="goBack()" data-tip="返回" style="display:none;"><svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><path d="M15 5 L8 12 L15 19"/></svg></button>
  <div id="zoomctl" style="display:none;">
    <button id="zc-grid" class="iconbtn" data-tip="恢复自动尺寸"><svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M4 4 h16 v16 h-16 Z M9.3 4 v16 M14.7 4 v16 M4 9.3 h16 M4 14.7 h16"/></svg></button>
    <button id="zc-minus" class="iconbtn" data-tip="缩小"><svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><path d="M5 12 h14"/></svg></button>
    <input type="range" id="size-slider" min="0" max="2" step="1" oninput="applySize(this.value)">
    <button id="zc-plus" class="iconbtn" data-tip="放大"><svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><path d="M12 5 v14 M5 12 h14"/></svg></button>
  </div>
  <h1 id="title">FaceID Groups</h1>
  <span id="sel-count"></span>
  <button id="btn-selectmode" class="iconbtn" onclick="toggleSelectMode()" data-tip="选择模式：单击即多选">Select</button>
  <button id="btn-landmarks" class="iconbtn" onclick="toggleLandmarks()" data-tip="显示标记点"><svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><ellipse cx="12" cy="18.4" rx="7.5" ry="2.8"/><path d="M12 3 a4.2 4.2 0 0 1 4.2 4.2 c0 2.8-4.2 7-4.2 7 s-4.2-4.2-4.2-7 A4.2 4.2 0 0 1 12 3 Z"/><circle cx="12" cy="7.2" r="1.5"/></svg></button>
  <button id="btn-newfolder" class="iconbtn" onclick="ctxNewFolder()" data-tip="移到新文件夹"><svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><path d="M12 5 v14 M5 12 h14"/></svg></button>
  <button id="btn-merge" class="iconbtn" onclick="ctxMergeMain()" data-tip="合并选中到主簇"><svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M4 4 L12 12 M20 4 L12 12 M12 12 V20 M8 16 L12 20 L16 16"/></svg></button>
  <button id="btn-sort" class="iconbtn" onclick="showSortMenu(event)" data-tip="排序"><svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M4 6 h9 M4 12 h6 M4 18 h9"/><path d="M18 5 v14 M15 16 l3 3 3-3"/></svg></button>
  <button id="btn-dedupe" class="iconbtn" onclick="dedupeOrMoveMarked()" data-tip="查重"><svg id="dd-icon-find" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="8" y="3" width="13" height="13" rx="1"/><path d="M16 16 v3 a1 1 0 0 1-1 1 H4 a1 1 0 0 1-1-1 V8 a1 1 0 0 1 1-1 h3"/></svg><svg id="dd-icon-move" style="display:none" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M3 7 q0-2 2-2 h4 l2 2 h8 q2 0 2 2 v9 q0 2-2 2 H5 q-2 0-2-2 Z"/><path d="M9 12 h6 M13 10 l2 2 -2 2"/></svg></button>
  <button id="btn-refcheck" class="iconbtn" onclick="refcheckOrMoveMarked()" data-tip="指定人脸检测"><svg id="rc-icon-find" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="9" cy="7.5" r="3.2"/><path d="M3 20 c0-3.3 2.7-5.8 6-5.8 h1.5"/><circle cx="16.5" cy="16.5" r="4"/><path d="M19.8 19.8 L22 22"/></svg><svg id="rc-icon-move" style="display:none" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="9" cy="7.5" r="3.2"/><path d="M3 20 c0-3.3 2.7-5.8 6-5.8 h1.5"/><path d="M14 15 h7 M18 12 l3 3 -3 3"/></svg></button>
  <button id="btn-del" class="iconbtn" onclick="ctxDeleteSel()" data-tip="删除选中"><svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M4 7 h16 M10 7 V5 a1 1 0 0 1 1-1 h2 a1 1 0 0 1 1 1 v2 M6.5 7 l1 13 h9 l1-13 M10 11 v5 M14 11 v5"/></svg></button>
</div>
<div id="scrollarea"><div id="grid" class="groups"></div><div id="grid-sentinel"></div></div>
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
let gridObs = null;
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
  document.getElementById('sel-count').style.display = 'none';
  setSelectMode(false);
  document.getElementById('zoomctl').style.display = 'none';
  document.getElementById('title').textContent = 'FaceID Groups';
  document.getElementById('grid').className = 'groups';
  const _sen0 = document.getElementById('grid-sentinel');
  if (_sen0) _sen0.style.display = 'none';
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
    const gLabel = g.root ? '（根目录）' : g.name;
    card.innerHTML = `<img src="/api/thumb/${encodeURIComponent(g.name)}" loading="lazy">
      <div class="glabel"><div class="gname">${gLabel}</div><div class="gcount">${g.count} 张</div></div>`;
    card.onclick = () => {
      if (physKeys.ctrl || physKeys.meta) toggleGroupSel(g.name, card, i);
      else if (physKeys.shift && groupSel.size > 0) rangeGroupSel(i);
      else if (selectMode || groupSel.size > 0) toggleGroupSel(g.name, card, i);
      else openGroup(g.name);
    };
    card.onmouseenter = () => { hpHoverGroup = g.name; };
    card.onmouseleave = () => { hpHoverGroup = null; };
    if (g.name !== '.') card.addEventListener('contextmenu', (e) => showGroupCtx(e, g.name));
    groupCards.push({ name: g.name, el: card });
    grid.appendChild(card);
  });
  updateSelUI();
}

let hpTimer = null, hpCloseTimer = null, hpHideTimer = null, hpOnPv = false;
const HP_SIZES = [180, 250, 340];
const hpCache = {};
let hpHoverGroup = null;

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
  // 预览格子尺寸 = 网页图片当前尺寸（--fs，未设置时用 CSS 默认 120px）
  let cell = parseInt(getComputedStyle(document.documentElement).getPropertyValue('--fs')) || 0;
  if (!cell) {
    const slv = document.getElementById('size-slider');
    cell = (slv && slv.value !== '' && HP_SIZES[Number(slv.value)]) ? HP_SIZES[Number(slv.value)] : 120;
  }
  const cgap = 4, pad = 24;
  const W_max = innerWidth * 0.94, H_max = innerHeight * 0.86;
  const colsMax = Math.max(1, Math.floor((W_max - pad - cgap) / (cell + cgap)));
  const cols = Math.min(colsMax, n);
  const rows = Math.ceil(n / cols);
  const w = Math.min(W_max, cols * (cell + cgap) - cgap + pad);
  const h = Math.min(H_max, rows * (cell + cgap) - cgap + pad + 2);
  pv.querySelector('.hp-grid').style.gridTemplateColumns = `repeat(${cols}, minmax(0, 1fr))`;
  const gap = 3;
  let x, y, flipX = false, flipY = false;
  if (e) {
    x = e.clientX + gap; y = e.clientY + gap;
    if (x + w > innerWidth - 6) { x = Math.max(6, e.clientX - w - gap); flipX = true; }
    if (y + h > innerHeight - 6) { y = Math.max(6, e.clientY - h - gap); flipY = true; }
  } else {
    x = Math.max(6, (innerWidth - w) / 2); y = Math.max(6, (innerHeight - h) / 2);
  }
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
  updateSelUI();
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
  if (r.moved !== undefined) { invalidateCaches([...sources, target]); console.log(`组级合并 ${r.moved} 张`); loadGroups(); }
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
  if (r.deleted !== undefined) { invalidateCaches(sources); console.log(`永久删除 ${r.deleted} 张`); loadGroups(); }
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
  if (r.moved !== undefined) { invalidateCaches([...sources, target]); console.log(`移入新夹 ${r.target} 共 ${r.moved} 张`); loadGroups(); }
}

// ── 组内视图：全部脸图 ──
async function openGroup(name) {
  endHoverPreview();
  currentGroup = name;
  selected.clear();
  lastClickIdx = -1;
  endHoverPreview();
  document.getElementById('btn-back').style.display = 'flex';
  document.getElementById('title').textContent = name === '.' ? '（根目录）' : name;
  document.getElementById('grid').className = 'faces';
  const data = await api(`/api/group/${encodeURIComponent(name)}`);
  allFaces = data.faces || [];
  if (sortState && sortState.group === name) {
    const _set = new Set(allFaces);
    const _ordered = sortState.faces.filter(f => _set.has(f));
    const _seen = new Set(_ordered);
    allFaces = _ordered.concat(allFaces.filter(f => !_seen.has(f)));
  }
  // 有查重结果（内存或落盘）→ 取回，并让同簇图片相邻显示
  let ddRes = null;
  if (dedupeResult && dedupeResult.group === name) ddRes = { groups: dedupeResult.clusters || [] };
  else {
    try {
      const st = await api('/api/dedupe-status');
      if (st && st.result && !st.result.error && st.result.group === name) ddRes = st.result;
    } catch (err) { }
  }
  const clusterNo = {};
  if (ddRes) {
    dedupeResult = { group: name, red: new Set(), clusters: ddRes.groups || [] };
    for (const g of dedupeResult.clusters) for (const d of g.dupes) dedupeResult.red.add(d);
    dedupeResult.clusters.forEach((g, ci) => {
      if (g.keep) clusterNo[g.keep] = [ci + 1, true];
      for (const d of g.dupes) clusterNo[d] = [ci + 1, false];
    });
    const inCluster = new Set(Object.keys(clusterNo));
    const ordered = [];
    for (const g of dedupeResult.clusters) for (const n of [g.keep, ...g.dupes]) if (inCluster.has(n) && !ordered.includes(n)) ordered.push(n);
    allFaces = ordered.concat(allFaces.filter(f => !inCluster.has(f)));
  }
  const grid = document.getElementById('grid');
  grid.innerHTML = '';
  const sentinel = document.getElementById('grid-sentinel');
  const wantDedupe = !!(dedupeResult && dedupeResult.group === currentGroup);
  const wantRef = !!(refResult && refResult.group === currentGroup);
  const makeFaceEl = (f, i) => {
    const el = document.createElement('div');
    el.className = 'face-item';
    el.dataset.idx = i;
    el.title = f;
    if (clusterNo[f]) el.innerHTML = `<span class="cnum${clusterNo[f][1] ? ' keep' : ''}">${clusterNo[f][0]}</span>`;
    el.innerHTML += `<img src="/api/face/${encodeURIComponent(name)}/${encodeURIComponent(f)}?t=1" loading="lazy">`;
    el.addEventListener('click', (e) => faceClick(e, i));
    el.addEventListener('dblclick', () => openLightbox(el.querySelector('img'), `/api/face/${encodeURIComponent(name)}/${encodeURIComponent(f)}`));
    el.addEventListener('contextmenu', (e) => showCtxMenu(e, name, f));
    if (wantDedupe && dedupeResult.red.has(f)) el.classList.add('dupe');
    if (wantRef && refResult.mismatch.has(f)) el.classList.add('mismatch');
    if (selected.has(i)) el.classList.add('selected');
    if (lmShow) lmObserver.observe(el);
    return el;
  };
  const BATCH = 240;
  let rendered = 0;
  const renderBatch = () => {
    const end = Math.min(rendered + BATCH, allFaces.length);
    const frag = document.createDocumentFragment();
    for (let i = rendered; i < end; i++) frag.appendChild(makeFaceEl(allFaces[i], i));
    grid.appendChild(frag);
    rendered = end;
    if (sentinel) sentinel.style.display = rendered < allFaces.length ? 'block' : 'none';
    if (rendered >= allFaces.length && gridObs) { gridObs.disconnect(); gridObs = null; }
  };
  renderBatch();
  if (gridObs) { gridObs.disconnect(); gridObs = null; }
  gridObs = new IntersectionObserver((ents) => {
    if (ents.some((x) => x.isIntersecting)) renderBatch();
  }, { root: document.getElementById('scrollarea'), rootMargin: '600px 0px' });
  if (sentinel) gridObs.observe(sentinel);
  initSlider();
  refreshGroupsSilently();
  updateSelUI();
  maybeResumeRefcheck();
}

function initSlider() {
  const zc = document.getElementById('zoomctl');
  const sl = document.getElementById('size-slider');
  zc.style.display = 'flex';
  if (sliderGroup === currentGroup) return;
  sliderGroup = currentGroup;
  document.documentElement.style.removeProperty('--fs');
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

let selectMode = false;
let sliderGroup = null;
function setSelectMode(on) {
  selectMode = on;
  const b = document.getElementById('btn-selectmode');
  if (b) b.classList.toggle('active', on);
}
function toggleSelectMode() { setSelectMode(!selectMode); }

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
  } else if (selectMode) {
    // Select 模式：单击切换，累积多选
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
  if (currentGroup) {
    sc.textContent = n > 0 ? `已选 ${n}/${allFaces.length} 张` : '';
    sc.style.display = n > 0 ? 'block' : 'none';
  } else if (groupSel.size > 0) {
    sc.textContent = `已选 ${groupSel.size} 组 → 将并入 ${mainClusterName() || '主簇'}`;
    sc.style.display = 'block';
  } else {
    sc.textContent = '';
    sc.style.display = 'none';
  }
  // 统一工具条：所有按钮常显，按情境置灰（内外视图完全一致）
  const anySel = currentGroup ? n > 0 : groupSel.size > 0;
  for (const id of ['btn-merge', 'btn-newfolder', 'btn-del']) {
    const b = document.getElementById(id);
    if (b) b.classList.toggle('dim', !anySel);
  }
  const bmRoot = document.getElementById('btn-merge');
  if (bmRoot && currentGroup === '.') {
    bmRoot.classList.add('dim');
    bmRoot.setAttribute('data-tip', '单夹模式：无需合并主簇');
  }
  const blm = document.getElementById('btn-landmarks');
  if (blm) blm.classList.toggle('dim', !currentGroup);
  const bsm = document.getElementById('btn-selectmode');
  if (bsm) bsm.classList.remove('dim');
  const bdd = document.getElementById('btn-dedupe');
  if (bdd) {
    const hasRed = !!(dedupeResult && dedupeResult.group === currentGroup && dedupeResult.red.size > 0);
    bdd.dataset.mode = hasRed ? 'move' : 'find';
    bdd.classList.toggle('dim', !hasRed && !currentGroup);
    bdd.setAttribute('data-tip', hasRed ? `把 ${dedupeResult.red.size} 张标红重复图移到 repeated/` : '查重');
    const icF = document.getElementById('dd-icon-find');
    const icM = document.getElementById('dd-icon-move');
    if (icF) icF.style.display = hasRed ? 'none' : 'block';
    if (icM) icM.style.display = hasRed ? 'block' : 'none';
  }
  const brc = document.getElementById('btn-refcheck');
  if (brc) {
    const hasMm = !!(refResult && refResult.group === currentGroup && refResult.mismatch.size > 0);
    brc.dataset.mode = hasMm ? 'move' : 'find';
    brc.classList.toggle('dim', !currentGroup);
    brc.setAttribute('data-tip', hasMm ? `把 ${refResult.mismatch.size} 张不匹配脸移到 mismatch/` : '指定人脸检测（先选中 1 张参考脸）');
    const rF = document.getElementById('rc-icon-find');
    const rM = document.getElementById('rc-icon-move');
    if (rF) rF.style.display = hasMm ? 'none' : 'block';
    if (rM) rM.style.display = hasMm ? 'block' : 'none';
  }
  const bsort = document.getElementById('btn-sort');
  if (bsort) {
    const sact = !!(sortState && sortState.group === currentGroup);
    bsort.classList.toggle('active', sact);
    bsort.classList.toggle('dim', !currentGroup);
    bsort.setAttribute('data-tip', sact ? (sortState.by === 'blur' ? '排序：模糊度（最糊在前）' : '排序：直方图相似度（最离群在前）') : '排序');
  }
}

function ctxMergeMain() { if (currentGroup === '.') return; if (currentGroup) return mergeSelected(); return mergeGroups(); }
function ctxNewFolder() { if (currentGroup) return moveFacesToNew(); return groupsToNew(); }
function ctxDeleteSel() { if (currentGroup) return deleteSelected(); return trashGroups(); }

async function mergeSelected() {
  if (!currentGroup || selected.size === 0) return;
  if (mainClusterName() === currentGroup) return;
  const names = Array.from(selected).map(i => allFaces[i]);
  const target = mainClusterName() || 'faceid_group_000';
  if (!await showConfirm(`将 ${names.length} 张从 ${currentGroup} 合并到 ${target}/ ？`)) return;
  const r = await api('/api/move-faces', {
    method: 'POST', headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({ group: currentGroup, faces: names, target: target })
  });
  if (r.moved !== undefined) { invalidateCaches([currentGroup, target]); openGroup(currentGroup); // 刷新当前视图
  }
}

async function deleteSelected() {
  if (!currentGroup || selected.size === 0) return;
  const names = Array.from(selected).map(i => allFaces[i]);
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
  if (currentGroup && (e.ctrlKey || e.metaKey) && (e.key === 'a' || e.key === 'A')) {
    e.preventDefault();
    selected.clear();
    for (let i = 0; i < allFaces.length; i++) selected.add(i);
    document.querySelectorAll('.face-item').forEach(el => el.classList.add('selected'));
    updateSelUI();
    return;
  }
  if (currentGroup && (e.ctrlKey || e.metaKey) && (e.key === 'i' || e.key === 'I')) {
    e.preventDefault();
    const inv = new Set();
    for (let i = 0; i < allFaces.length; i++) if (!selected.has(i)) inv.add(i);
    selected.clear();
    inv.forEach(i => selected.add(i));
    document.querySelectorAll('.face-item').forEach(el => el.classList.toggle('selected', selected.has(Number(el.dataset.idx))));
    updateSelUI();
    return;
  }
  if (e.key === 'Escape') { if (document.getElementById('lightbox').style.display === 'block') { closeLightbox(); return; } hideCtxMenu(); selected.clear(); updateSelUI(); document.querySelectorAll('.face-item.selected').forEach(el => el.classList.remove('selected')); groupSel.clear(); lastGroup = null; document.querySelectorAll('.group-card.gsel').forEach(el => el.classList.remove('gsel')); updateGroupSelUI(); endHoverPreview(); }
  if (e.key === 'Backspace' && currentGroup) goBack();
  if (e.key === 'Delete' && selected.size > 0 && currentGroup) { e.preventDefault(); deleteSelected(); }
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

// 目标文件夹白名单：只允许「新建的/重命名的」文件夹（自动簇与流程目录不可作目标）
const AUTO_DIR_NAMES = new Set(['faceid_noise', 'merged', 'groups', 'repeated', 'blurry', 'low_quality', 'medium_quality', 'high_quality']);
function curatedFolders() {
  return (allGroups || []).filter(g => !/^faceid_group_[0-9]+$/.test(g.name) && !AUTO_DIR_NAMES.has(g.name) && !g.name.startsWith('trash:'));
}

function showCtxMenu(e, group, face) {
  e.preventDefault();
  e.stopPropagation();
  window._ctx = { group, face };
  const mc = mainClusterName();
  const mainItem = (mc && mc !== group)
    ? `<div class="ctx-item" data-act="move-main"><svg width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M4 4 L12 12 M20 4 L12 12 M12 12 V20 M8 16 L12 20 L16 16"/></svg><span>移动到主簇</span></div>`
    : '';
  const subs = curatedFolders();
  const subHtml = subs.length
    ? subs.map(g => `<div class="ctx-item" data-target="${g.name}"><span>${g.name}</span></div>`).join('')
    : '<div class="ctx-item" style="color:#888;cursor:default;">先用 ＋ 新建文件夹</div>';
  openCtxMenu(e.clientX, e.clientY, `
    ${mainItem}
    <div class="ctx-item" data-act="reveal"><svg width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M3 7 q0-2 2-2 h4 l2 2 h8 q2 0 2 2 v9 q0 2-2 2 H5 q-2 0-2-2 Z"/></svg><span>在文件夹中显示</span></div>
    <div class="ctx-item ctx-has-sub"><svg width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M3 7 q0-2 2-2 h4 l2 2 h8 q2 0 2 2 v9 q0 2-2 2 H5 q-2 0-2-2 Z"/><path d="M9 12 h6 M13 10 l2 2 -2 2"/></svg><span>移动到文件夹</span><svg class="chev" width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><path d="M9 5 l7 7 -7 7"/></svg><div id="ctx-sub">${subHtml}</div></div>
    <div class="ctx-item danger" data-act="del"><svg width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M4 7 h16 M10 7 V5 a1 1 0 0 1 1-1 h2 a1 1 0 0 1 1 1 v2 M6.5 7 l1 13 h9 l1-13 M10 11 v5 M14 11 v5"/></svg><span>删除</span></div>
  `);
}

function showGroupCtx(e, name) {
  e.preventDefault();
  e.stopPropagation();
  window._ctxGroup = name;
  const mc = mainClusterName();
  const mainItem = (mc && mc !== name)
    ? `<div class="ctx-item" data-act="merge-main"><svg width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M4 4 L12 12 M20 4 L12 12 M12 12 V20 M8 16 L12 20 L16 16"/></svg><span>合并到主簇</span></div>`
    : '';
  const subs = curatedFolders().filter(g => g.name !== name);
  const subHtml = subs.length
    ? subs.map(g => `<div class="ctx-item" data-gtarget="${g.name}"><span>${g.name}</span></div>`).join('')
    : '<div class="ctx-item" style="color:#888;cursor:default;">先用「移到新文件夹」建目标夹</div>';
  openCtxMenu(e.clientX, e.clientY, `
    ${mainItem}
    <div class="ctx-item ctx-has-sub"><svg width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M3 7 q0-2 2-2 h4 l2 2 h8 q2 0 2 2 v9 q0 2-2 2 H5 q-2 0-2-2 Z"/><path d="M9 12 h6 M13 10 l2 2 -2 2"/></svg><span>合并到文件夹</span><svg class="chev" width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><path d="M9 5 l7 7 -7 7"/></svg><div id="ctx-sub">${subHtml}</div></div>
    <div class="ctx-item" data-act="rename"><svg width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 20 h9"/><path d="M16.5 3.5 a2.1 2.1 0 0 1 3 3 L7 19 l-4 1 1-4 Z"/></svg><span>重命名</span></div>
    <div class="ctx-item danger" data-act="del-group"><svg width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M4 7 h16 M10 7 V5 a1 1 0 0 1 1-1 h2 a1 1 0 0 1 1 1 v2 M6.5 7 l1 13 h9 l1-13 M10 11 v5 M14 11 v5"/></svg><span>删除文件夹</span></div>
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

function invalidateCaches(groups) {
  for (const g of groups) delete hpCache[g];
  if (groups.length) api('/api/invalidate', { method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({ groups }) });
}

async function doMoveFaces(target) {
  const c = window._ctx;
  if (!c) return;
  const names = selected.size ? Array.from(selected).map(i => allFaces[i]) : [c.face];
  const r = await api('/api/move-faces', {
    method: 'POST', headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({ group: c.group, faces: names, target: target })
  });
  if (r.moved !== undefined) { selected.clear(); invalidateCaches([c.group, target]); openGroup(c.group); }
}

async function deleteGroupAction() {
  const name = window._ctxGroup;
  if (!name) return;
  const info = (allGroups.find(g => g.name === name) || {}).count || 0;
  if (!await showConfirm(`永久删除文件夹「${name}」及其中 ${info} 张图？（不可恢复）`)) return;
  const r = await api('/api/delete-groups', {
    method: 'POST', headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({ groups: [name] })
  });
  if (r.deleted !== undefined) { invalidateCaches([name]); loadGroups(); }
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
  else flashStatus(`重命名失败: ${r.error || ''}`);
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
let modalHideTimer = null;

function showConfirm(text) {
  return new Promise((resolve) => {
    clearTimeout(modalHideTimer);
    const mask = document.getElementById('modal-mask');
    const card = document.getElementById('modal-card');
    const inp = document.getElementById('modal-input');
    inp.style.display = 'none';
    const bar = document.getElementById('modal-progress');
    if (bar) bar.remove();
    document.getElementById('modal-btns').style.display = 'flex';
    document.getElementById('modal-text').textContent = text;
    modalOpen = true;
    mask.style.display = 'flex';
    void mask.offsetHeight;
    mask.style.opacity = '1';
    card.style.transform = 'scale(1)';
    const close = (val) => {
      mask.style.opacity = '0';
      card.style.transform = 'scale(.95)';
      clearTimeout(modalHideTimer);
      modalHideTimer = setTimeout(() => { mask.style.display = 'none'; }, 160);
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
    clearTimeout(modalHideTimer);
    const mask = document.getElementById('modal-mask');
    const card = document.getElementById('modal-card');
    const inp = document.getElementById('modal-input');
    const bar = document.getElementById('modal-progress');
    if (bar) bar.remove();
    document.getElementById('modal-btns').style.display = 'flex';
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
      clearTimeout(modalHideTimer);
      modalHideTimer = setTimeout(() => { mask.style.display = 'none'; inp.style.display = 'none'; }, 160);
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

let dedupeResult = null;

function showProgressModal(text) {
  clearTimeout(modalHideTimer);
  const mask = document.getElementById('modal-mask');
  const card = document.getElementById('modal-card');
  const inp = document.getElementById('modal-input');
  inp.style.display = 'none';
  document.getElementById('modal-btns').style.display = 'none';
  document.getElementById('modal-text').textContent = text;
  let bar = document.getElementById('modal-progress');
  if (!bar) {
    bar = document.createElement('div');
    bar.innerHTML = '<div id="modal-progress-fill"></div>';
    bar.id = 'modal-progress';
    document.getElementById('modal-text').after(bar);
  }
  bar.firstChild.style.width = '0%';
  modalOpen = true;
  mask.style.display = 'flex';
  void mask.offsetHeight;
  mask.style.opacity = '1';
  card.style.transform = 'scale(1)';
}

function updateProgressModal(text, pct) {
  document.getElementById('modal-text').textContent = text;
  const fill = document.getElementById('modal-progress-fill');
  if (fill) fill.style.width = Math.max(0, Math.min(100, pct)) + '%';
}

function hideProgressModal() {
  const mask = document.getElementById('modal-mask');
  const card = document.getElementById('modal-card');
  mask.style.opacity = '0';
  card.style.transform = 'scale(.95)';
  clearTimeout(modalHideTimer);
  modalHideTimer = setTimeout(() => {
    mask.style.display = 'none';
    const bar = document.getElementById('modal-progress');
    if (bar) bar.remove();
    document.getElementById('modal-btns').style.display = 'flex';
  }, 160);
  modalOpen = false;
}

async function startDedupe() {
  if (!currentGroup) return;
  if (!await showConfirm(`在「${currentGroup}」内查重（感知哈希相似度 ≥ 0.98）？`)) return;
  const r0 = await api('/api/dedupe', {
    method: 'POST', headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({ group: currentGroup })
  });
  if (!r0.ok) { await showConfirm(r0.error === 'busy' ? '查重已在进行中' : '查重启动失败'); return; }
  showProgressModal('查重中…');
  let sawRunning = false, polls = 0;
  const timer = setInterval(async () => {
    let st;
    try { st = await api('/api/dedupe-status'); } catch (err) { return; }
    polls++;
    if (st.running) {
      sawRunning = true;
      const label = st.phase === 'hashing' ? '计算哈希' : '两两比对';
      updateProgressModal(`${label} ${st.done}/${st.total}`, st.total ? st.done / st.total * 100 : 0);
      return;
    }
    if (!sawRunning && !st.result && polls < 6) return;  // 启动竞态：再等几拍
    clearInterval(timer);
    hideProgressModal();
    if (!st.result || st.result.error) { await showConfirm('查重失败: ' + (st.result && st.result.error || '未知')); return; }
    applyDedupeResult(st.result);
    await openGroup(currentGroup);
    await showConfirm(`检出 ${st.result.dupe_groups} 组重复 / ${st.result.dupes_total} 张重复图，已按簇相邻排列并红框标出（簇内第一张保留）`);
  }, 500);
}

function applyDedupeResult(res) {
  dedupeResult = { group: currentGroup, red: new Set(), clusters: res.groups || [] };
  for (const g of res.groups) for (const d of g.dupes) dedupeResult.red.add(d);
  document.querySelectorAll('.face-item').forEach(el => {
    const f = allFaces[Number(el.dataset.idx)];
    el.classList.toggle('dupe', dedupeResult.red.has(f));
  });
  updateSelUI();
}

async function moveMarkedDupes() {
  if (!dedupeResult || dedupeResult.group !== currentGroup || dedupeResult.red.size === 0) return;
  const names = Array.from(dedupeResult.red);
  if (!await showConfirm(`把 ${names.length} 张标红重复图移到 repeated/ ？`)) return;
  const r = await api('/api/move-faces', {
    method: 'POST', headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({ group: currentGroup, faces: names })
  });
  if (r.moved !== undefined) {
    dedupeResult = null;
    try { await api('/api/dedupe-clear', { method: 'POST' }); } catch (err) { }
    invalidateCaches([currentGroup, 'repeated']);
    openGroup(currentGroup);
  }
}

let sortState = null;  // {group, by, faces}
let refResult = null;  // {group, mismatch:Set}
let refPollTimer = null;

function showSortMenu(e) {
  if (!currentGroup) { flashStatus('先进一个组'); return; }
  e.preventDefault();
  e.stopPropagation();
  const m = document.getElementById('ctxmenu');
  const cur = (sortState && sortState.group === currentGroup) ? sortState.by : 'default';
  const item = (k, label) => `<div class="ctx-item${cur === k ? ' active' : ''}" data-sort="${k}"><span>${label}</span></div>`;
  m.innerHTML = item('default', '默认顺序') + item('blur', '模糊度 ↑ 最糊在前') + item('hist', '直方图相似度 ↑ 最离群在前');
  m.style.display = 'block';
  const w = m.offsetWidth, hgt = m.offsetHeight;
  let x = e.clientX, y = e.clientY;
  if (x + w > innerWidth - 8) x = innerWidth - w - 8;
  if (y + hgt > innerHeight - 8) y = innerHeight - hgt - 8;
  m.style.left = Math.max(8, x) + 'px';
  m.style.top = Math.max(8, y) + 'px';
  const hide = (ev) => { if (!m.contains(ev.target)) { m.style.display = 'none'; document.removeEventListener('mousedown', hide); } };
  setTimeout(() => document.addEventListener('mousedown', hide), 0);
}

async function applySort(by) {
  if (!currentGroup) return;
  if (by === 'default') {
    sortState = null;
    await openGroup(currentGroup);
    flashStatus('已恢复默认顺序');
    return;
  }
  flashStatus(by === 'blur' ? '正在按模糊度排序…' : '正在按直方图相似度排序…');
  const r = await api('/api/sort', { method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({ group: currentGroup, by: by }) });
  if (!r.faces) { flashStatus('排序失败'); return; }
  sortState = { group: currentGroup, by: by, faces: r.faces };
  await openGroup(currentGroup);
  flashStatus(by === 'blur' ? '已按模糊度排序（最糊在前，共 ' + r.faces.length + ' 张）' : '已按直方图相似度排序（最离群在前）');
}

async function refcheckOrMoveMarked() {
  const brc = document.getElementById('btn-refcheck');
  if (!currentGroup) { flashStatus('先进一个组，选中一张图作参考'); return; }
  if (brc && brc.dataset.mode === 'move') {
    const names = Array.from(refResult.mismatch);
    if (!await showConfirm(`把 ${names.length} 张不匹配的脸移到 mismatch/ ？`)) return;
    const r = await api('/api/move-faces', {
      method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({ group: currentGroup, faces: names, target: 'mismatch' })
    });
    if (r.moved !== undefined) {
      refResult = null;
      await api('/api/refcheck-clear', { method: 'POST' });
      invalidateCaches([currentGroup, 'mismatch']);
      flashStatus(`已移走 ${r.moved} 张到 mismatch/`);
      openGroup(currentGroup);
    }
    return;
  }
  if (selected.size !== 1) { flashStatus('先选中恰好 1 张图作为参考脸'); return; }
  const ref = allFaces[Array.from(selected)[0]];
  if (!await showConfirm(`用「${ref}」作为参考脸，检测组内不匹配的脸？`)) return;
  const r0 = await api('/api/refcheck', { method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({ group: currentGroup, ref: ref }) });
  if (!r0.ok) { flashStatus(r0.error === 'busy' ? '检测已在进行中…' : '启动失败: ' + (r0.error || '')); return; }
  showProgressModal('人脸比对中…');
  pollRefcheck();
}

function pollRefcheck() {
  if (refPollTimer) { clearInterval(refPollTimer); refPollTimer = null; }
  let idleTicks = 0;
  refPollTimer = setInterval(async () => {
    let st;
    try { st = await api('/api/refcheck-status'); } catch (e) { return; }
    if (st.running) { idleTicks = 0; updateProgressModal(`比对 ${st.done}/${st.total}`, st.total ? st.done / st.total * 100 : 0); return; }
    if (!st.result) {
      idleTicks++;
      if (idleTicks < 20) return;
      clearInterval(refPollTimer); refPollTimer = null;
      hideProgressModal();
      await showConfirm('检测失败：服务长时间未返回结果，请重试或看服务日志');
      return;
    }
    clearInterval(refPollTimer); refPollTimer = null;
    hideProgressModal();
    if (st.result.error) { await showConfirm('检测失败: ' + st.result.error); return; }
    applyRefResult(st.result);
    await showConfirm(`检出 ${st.result.mismatch.length} 张不匹配 / ${st.result.matched} 张匹配（阈值 ${st.result.threshold}）。再点一次该按钮把它们移到 mismatch/`);
  }, 500);
}

function applyRefResult(res) {
  const ms = new Set(res.mismatch || []);
  refResult = { group: res.group || currentGroup, mismatch: ms };
  document.querySelectorAll('.face-item').forEach(el => {
    const f = allFaces[Number(el.dataset.idx)];
    el.classList.toggle('mismatch', ms.has(f));
  });
  updateSelUI();
}

async function maybeResumeRefcheck() {
  if (!currentGroup) return;
  let st;
  try { st = await api('/api/refcheck-status'); } catch (e) { return; }
  if (!st) return;
  if (st.running && st.group === currentGroup) { showProgressModal('人脸比对中…'); pollRefcheck(); return; }
  if (st.result && !st.result.error && st.result.group === currentGroup) { applyRefResult(st.result); }
}

async function dedupeOrMoveMarked() {
  const bdd = document.getElementById('btn-dedupe');
  if (bdd && bdd.dataset.mode === 'move') return moveMarkedDupes();
  // 已有结果（内存或落盘）就直接恢复，不重跑
  try {
    const st = await api('/api/dedupe-status');
    if (st && st.running) { flashStatus('查重已在进行中…'); return; }
    if (st && st.result && !st.result.error && st.result.group === currentGroup) { applyDedupeResult(st.result); await openGroup(currentGroup); return; }
  } catch (err) { }
  return startDedupe();
}

async function refreshGroupsSilently() {
  try {
    const d = await api('/api/groups');
    allGroups = (d.groups || []).sort((a, b) => b.count - a.count);
  } catch (err) {}
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
  if (r.moved !== undefined) { console.log(`移动 ${r.moved} 张到 ${r.target}`); await refreshGroupsSilently(); openGroup(currentGroup); }
}

async function moveFacesToMain() {
  const c = window._ctx;
  if (!c) return;
  const target = mainClusterName();
  if (!target || target === c.group) return;
  const names = selected.size ? Array.from(selected).map(i => allFaces[i]) : [c.face];
  if (!await showConfirm(`将 ${names.length} 张从 ${c.group} 移动到主簇 ${target}/ ？`)) return;
  const r = await api('/api/move-faces', {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ group: c.group, faces: names, target: target })
  });
  if (r.moved !== undefined) { selected.clear(); invalidateCaches([c.group, target]); openGroup(c.group); }
}

async function ctxDelete() {
  const c = window._ctx;
  if (!c) return;
  const names = selected.size ? Array.from(selected).map(i => allFaces[i]) : [c.face];
  if (!await showConfirm(`永久删除 ${names.length} 张？（不可恢复）`)) return;
  const r = await api('/api/delete', {
    method: 'POST', headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({ group: c.group, faces: names })
  });
  if (r.deleted !== undefined) { selected.clear(); invalidateCaches([c.group]); openGroup(c.group); }
}

async function mergeGroupTo(target) {
  const src = window._ctxGroup;
  if (!src || src === target || src.includes(':')) return;
  const info = (allGroups.find(g => g.name === src) || {}).count || 0;
  if (!await showConfirm(`把「${src}」（${info} 张）并入「${target}」？`)) return;
  const r = await api('/api/merge', {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ groups: [src], target: target })
  });
  if (r.moved !== undefined) { invalidateCaches([src, target]); window._ctxGroup = null; loadGroups(); }
  else flashStatus(`合并失败: ${r.error || ''}`);
}

// 二级列表贴边自动翻转：会超出右缘时改为向左弹出，保证始终完整可见
document.getElementById('ctxmenu').addEventListener('mouseover', (e) => {
  const item = e.target.closest('.ctx-has-sub');
  for (const el of document.querySelectorAll('#ctxmenu .ctx-has-sub.ctx-flip')) el.classList.remove('ctx-flip');
  if (!item) return;
  const sub = item.querySelector('#ctx-sub');
  if (!sub) return;
  const prev = sub.style.display;
  sub.style.display = 'block';
  const r = sub.getBoundingClientRect();
  sub.style.display = prev;
  if (r.width && r.right > innerWidth - 6) item.classList.add('ctx-flip');
});

let stTimer = null;
function flashStatus(msg) {
  const st = document.getElementById('status');
  st.textContent = msg; st.style.display = 'block';
  clearTimeout(stTimer); stTimer = setTimeout(() => { st.style.display = 'none'; }, 4000);
}

// Space：屏幕中央预览悬停中的文件夹；再按/Esc 关闭（输入框内不拦截）
async function undoDelete() {
  const r = await api('/api/undo-delete', { method: 'POST' });
  if (!r.ok) { flashStatus(r.error === 'empty' ? '没有可撤销的删除' : '撤销失败: ' + (r.error || '')); return; }
  flashStatus(`已恢复 ${r.face} → ${r.group}（还可撤销 ${r.left} 张）`);
  invalidateCaches([r.group]);
  if (currentGroup === r.group) { openGroup(currentGroup); }
  else if (!currentGroup) { loadGroups(); }
}

document.addEventListener('keydown', (e) => {
  const tg = e.target && e.target.tagName;
  if (tg === 'INPUT' || tg === 'TEXTAREA') return;
  if ((physKeys.ctrl || physKeys.meta) && (e.key === 'z' || e.key === 'Z') && !e.shiftKey) {
    if (modalOpen) return;
    e.preventDefault();
    undoDelete();
  }
});

document.addEventListener('keydown', (e) => {
  const tg = e.target && e.target.tagName;
  if (tg === 'INPUT' || tg === 'TEXTAREA') return;
  if (e.key === 'Escape') { endHoverPreview(); return; }
  if (e.code !== 'Space' && e.key !== ' ') return;
  const pv = document.getElementById('hover-preview');
  if (pv && pv.style.display === 'flex') { endHoverPreview(); e.preventDefault(); return; }
  if (currentGroup) return;
  if (!hpHoverGroup) return;
  e.preventDefault();
  showHoverPreview(hpHoverGroup, null);
});

document.getElementById('ctxmenu').onclick = async (e) => {
  const it = e.target.closest('.ctx-item');
  if (!it) return;
  if (it.dataset.sort) { hideCtxMenu(); applySort(it.dataset.sort); return; }
  if (it.dataset.target) { hideCtxMenu(); doMoveFaces(it.dataset.target); return; }
  if (it.dataset.gtarget) { hideCtxMenu(); mergeGroupTo(it.dataset.gtarget); return; }
  const act = it.dataset.act;
  if (!act) return;
  hideCtxMenu();
  if (act === 'reveal') {
    const c = window._ctx;
    if (c) await api('/api/reveal', { method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(c) });
  } else if (act === 'rename') {
    renameGroupAction();
  } else if (act === 'move-main') {
    moveFacesToMain();
  } else if (act === 'del') {
    ctxDelete();
  } else if (act === 'del-group') {
    deleteGroupAction();
  } else if (act === 'merge-main') {
    const mc = mainClusterName();
    if (mc && window._ctxGroup && mc !== window._ctxGroup) mergeGroupTo(mc);
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
    """组名→目录；'.'=根目录自身；trash:<名> 映射到 .trash/<名> 子夹"""
    if name == '.':
        return root
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
            root_imgs = [x for x in os.listdir(self.server.root)
                         if x.lower().endswith(('.png', '.jpg', '.jpeg'))
                         and os.path.isfile(os.path.join(self.server.root, x))]
            if root_imgs:
                groups.insert(0, {'name': '.', 'count': len(root_imgs), 'root': True})
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
            src = 'dfld'
            if res is None:
                res = _fan_landmarks(fp_real)
                src = 'fan'
            if res is None:
                self._j({'ok': False, 'landmarks': [], 'w': 0, 'h': 0})
            else:
                self._j({'ok': True, 'landmarks': res[0], 'w': res[1], 'h': res[2], 'source': src})
        elif p == '/api/refcheck-status':
            self._j(dict(_ref_state))
        elif p == '/api/dedupe-status':
            st = dict(_dedupe_state)
            if not st.get('running') and st.get('result') is None:
                try:
                    fp = os.path.join(self.server.root, '.dedupe_result.json')
                    if os.path.isfile(fp):
                        with open(fp, 'r', encoding='utf-8') as fh:
                            st['result'] = json.load(fh)
                except Exception:
                    pass
            self._j(st)
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
                    _remember_deleted(group, f, src)
                    os.remove(src)
                    n += 1
            self._j({'deleted': n})
        elif p == '/api/delete-groups':
            cl = int(self.headers.get('Content-Length', 0))
            data = json.loads(self.rfile.read(cl))
            groups = [g for g in data.get('groups', []) if g != '.']
            n = 0
            for g in groups:
                gd = resolve_group(self.server.root, g)
                if not os.path.isdir(gd):
                    continue
                for f in os.listdir(gd):
                    fp = os.path.join(gd, f)
                    if os.path.isfile(fp):
                        _remember_deleted(g, f, fp)
                        os.remove(fp)
                        n += 1
                try:
                    os.rmdir(gd)
                except OSError:
                    pass
            self._j({'deleted': n})
        elif p == '/api/undo-delete':
            if not _undo_buf:
                self._j({'ok': False, 'error': 'empty'})
                return
            item = _undo_buf.pop()
            gd = resolve_group(self.server.root, item['group'])
            try:
                os.makedirs(gd, exist_ok=True)
                dst = os.path.join(gd, item['face'])
                if os.path.exists(dst):
                    stem, ext = os.path.splitext(item['face'])
                    dst = os.path.join(gd, f"{stem}_restored{ext}")
                with open(dst, 'wb') as f:
                    f.write(item['data'])
                self._j({'ok': True, 'group': item['group'], 'face': os.path.basename(dst), 'left': len(_undo_buf)})
            except OSError as e:
                self._j({'ok': False, 'error': str(e)})
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
        elif p == '/api/refcheck':
            cl = int(self.headers.get('Content-Length', 0))
            data = json.loads(self.rfile.read(cl))
            if _ref_state.get('running'):
                self._j({'ok': False, 'error': 'busy'})
                return
            group = data.get('group', '')
            ref = data.get('ref', '')
            threshold = float(data.get('threshold', 0.4))
            gd = resolve_group(self.server.root, group)
            if not os.path.isdir(gd) or not os.path.isfile(os.path.join(gd, ref)):
                self._j({'ok': False, 'error': 'no ref'})
                return
            _ref_state.update(running=True, phase='starting', done=0, total=0, result=None, group=group)
            threading.Thread(target=_refcheck_worker, args=(self.server.root, group, ref, threshold), daemon=True).start()
            self._j({'ok': True})
        elif p == '/api/refcheck-status':
            self._j(dict(_ref_state))
        elif p == '/api/refcheck-clear':
            _ref_state.update(running=False, phase='', done=0, total=0, result=None, group=None)
            self._j({'ok': True})
        elif p == '/api/sort':
            cl = int(self.headers.get('Content-Length', 0))
            data = json.loads(self.rfile.read(cl))
            group = data.get('group', '')
            by = data.get('by', 'blur')
            gd = resolve_group(self.server.root, group)
            faces = sorted(f for f in os.listdir(gd) if f.lower().endswith(('.png', '.jpg', '.jpeg'))) if os.path.isdir(gd) else []
            import numpy as _np
            import cv2 as _cv2
            vals = []
            if by == 'hist':
                hists = []
                for f in faces:
                    try:
                        img = _cv2.imdecode(_np.fromfile(os.path.join(gd, f), dtype=_np.uint8), _cv2.IMREAD_GRAYSCALE)
                    except Exception:
                        img = None
                    if img is None:
                        hists.append((f, None))
                        continue
                    hh = _cv2.calcHist([img], [0], None, [64], [0, 256]).flatten()
                    hh = hh / (hh.sum() + 1e-8)
                    hists.append((f, hh))
                valid = [hh for _, hh in hists if hh is not None]
                if valid:
                    mean = _np.mean(_np.array(valid), 0)
                    for f, hh in hists:
                        if hh is None:
                            vals.append((f, 0.0))
                            continue
                        c = _np.corrcoef(hh, mean)[0, 1]
                        vals.append((f, float(c) if not _np.isnan(c) else 0.0))
                vals.sort(key=lambda t: t[1])
            else:
                for f in faces:
                    try:
                        img = _cv2.imdecode(_np.fromfile(os.path.join(gd, f), dtype=_np.uint8), _cv2.IMREAD_GRAYSCALE)
                    except Exception:
                        img = None
                    if img is None:
                        vals.append((f, 0.0))
                        continue
                    vals.append((f, float(_cv2.Laplacian(img, _cv2.CV_64F).var())))
                vals.sort(key=lambda t: t[1])
            self._j({'faces': [f for f, _ in vals], 'by': by, 'metrics': [round(v, 4) for _, v in vals]})
        elif p == '/api/invalidate':
            cl = int(self.headers.get('Content-Length', 0))
            data = json.loads(self.rfile.read(cl))
            groups = data.get('groups', [])
            td = os.path.join(self.server.root, '.thumbs')
            n = 0
            if os.path.isdir(td):
                targets = {'thumb_' + g + '.jpg' for g in groups}
                for fn in os.listdir(td):
                    if fn in targets:
                        try:
                            os.remove(os.path.join(td, fn))
                            n += 1
                        except OSError:
                            pass
            self._j({'invalidated': n})
        elif p == '/api/dedupe':
            cl = int(self.headers.get('Content-Length', 0))
            data = json.loads(self.rfile.read(cl))
            if _dedupe_state.get('running'):
                self._j({'ok': False, 'error': 'busy'})
                return
            group = data.get('group', '')
            threshold = float(data.get('threshold', 0.98))
            gd = resolve_group(self.server.root, group)
            if not os.path.isdir(gd):
                self._j({'ok': False, 'error': 'no dir'})
                return
            threading.Thread(target=_dedupe_worker, args=(self.server.root, group, threshold), daemon=True).start()
            self._j({'ok': True})
        elif p == '/api/dedupe-clear':
            _dedupe_state.update(running=False, phase='', done=0, total=0, result=None)
            try:
                fp = os.path.join(self.server.root, '.dedupe_result.json')
                if os.path.isfile(fp):
                    os.remove(fp)
            except Exception:
                pass
            self._j({'ok': True})
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
            src_groups = [g for g in data.get('groups', []) if g != '.']
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


_undo_buf = collections.deque(maxlen=10)


def _remember_deleted(group, face, fp):
    """删除前内存缓存文件内容（最多 10 张，进程内存，不落盘）"""
    try:
        with open(fp, 'rb') as f:
            data = f.read()
        _undo_buf.append({'group': group, 'face': face, 'data': data})
    except OSError:
        pass


_dedupe_state = {'running': False, 'phase': '', 'done': 0, 'total': 0, 'result': None}


_ref_state = {'running': False, 'phase': '', 'done': 0, 'total': 0, 'result': None}
_arc_sess = [None]
_fan_sess = [None]
_lm_cache = {}


def _fan_landmarks(fp):
    """无 dFLd 时的兑底：FAN onnx 现场检测 68 点（返回 (pts, w, h, 'fan') 或 None）"""
    import numpy as _np
    import cv2 as _cv2
    try:
        buf = _np.fromfile(fp, dtype=_np.uint8)
        img = _cv2.imdecode(buf, _cv2.IMREAD_COLOR)
    except Exception:
        return None
    if img is None:
        return None
    h, w = img.shape[:2]
    import hashlib as _hl
    try:
        ck = 'lm_' + _hl.md5(fp.encode('utf-8', 'surrogatepass')).hexdigest() + f'_{w}x{h}_{int(os.path.getmtime(fp))}'
        if ck in _lm_cache:
            return _lm_cache[ck]
    except OSError:
        ck = None
    try:
        if _fan_sess[0] is None:
            import onnxruntime as _ort
            mp = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'modelhub', 'onnx', 'FAN', '2DFAN.onnx')
            _fan_sess[0] = _ort.InferenceSession(mp, providers=['CPUExecutionProvider'])
        center = _np.array([w / 2.0, h / 2.0], dtype=_np.float32)
        scale = max(w, h) / 195.0
        h2 = 200.0 * scale
        sx = 256.0 / h2
        tx = 256.0 * (-center[0] / h2 + 0.5)
        ty = 256.0 * (-center[1] / h2 + 0.5)
        m = _np.array([[sx, 0, tx], [0, sx, ty]], dtype=_np.float32)
        crop = _cv2.warpAffine(img, m, (256, 256), flags=_cv2.INTER_LINEAR)
        rgb = _cv2.cvtColor(crop, _cv2.COLOR_BGR2RGB)
        x = (rgb.astype(_np.float32) / 255.0).transpose(2, 0, 1)[None]
        s = _fan_sess[0]
        pred = s.run(None, {s.get_inputs()[0].name: x})[0]
        a = pred[0]
        ch, ah, aw = a.shape
        flat = a.reshape((ch, ah * aw))
        idx = flat.argmax(1)
        # heatmap 是 64×64 网格，反变换用 heatmap 分辨率（不是 256）
        sx64 = float(aw) / h2
        tx64 = float(aw) * (-center[0] / h2 + 0.5)
        ty64 = float(aw) * (-center[1] / h2 + 0.5)
        pts = []
        for i in range(ch):
            pX = int(idx[i] % aw)
            pY = int(idx[i] // aw)
            cx = pX + 0.5
            cy = pY + 0.5
            ox = float((cx - tx64) / sx64)
            oy = float((cy - ty64) / sx64)
            pts.append([ox, oy])
        res = (pts, w, h, 'fan')
        if ck:
            if len(_lm_cache) > 4000:
                _lm_cache.clear()
            _lm_cache[ck] = res
        return res
    except Exception:
        return None


def _arc_embed(fp):
    """ArcFace（w600k_mbf）512 维归一化 embedding"""
    import numpy as _np
    import cv2 as _cv2
    if _arc_sess[0] is None:
        import onnxruntime as _ort
        mp = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'modelhub', 'onnx', 'ArcFace', 'w600k_mbf.onnx')
        _arc_sess[0] = _ort.InferenceSession(mp, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    try:
        buf = _np.fromfile(fp, dtype=_np.uint8)
        img = _cv2.imdecode(buf, _cv2.IMREAD_COLOR)
    except Exception:
        return None
    if img is None:
        return None
    rs = _cv2.resize(img, (112, 112))
    rgb = _cv2.cvtColor(rs, _cv2.COLOR_BGR2RGB)
    x = (rgb.astype(_np.float32) / 255.0 - 0.5) / 0.5
    x = _np.transpose(x, (2, 0, 1))[None]
    s = _arc_sess[0]
    emb = s.run(None, {s.get_inputs()[0].name: x})[0].flatten()
    return emb / (_np.linalg.norm(emb) + 1e-8)


def _refcheck_worker(root, group, ref_face, threshold):
    try:
        import numpy as _np
        gd = resolve_group(root, group)
        ref_emb = _arc_embed(os.path.join(gd, ref_face))
        if ref_emb is None:
            _ref_state.update(running=False, phase='error', result={'error': '参考图读取失败'})
            return
        faces = sorted(f for f in os.listdir(gd) if f.lower().endswith(('.png', '.jpg', '.jpeg')))
        _ref_state.update(running=True, phase='checking', done=0, total=len(faces), result=None)
        mismatch = []
        matched = 0
        for i, fn in enumerate(faces):
            emb = _arc_embed(os.path.join(gd, fn))
            if emb is not None:
                sim = float(_np.dot(ref_emb, emb))
                if sim < threshold:
                    mismatch.append(fn)
                else:
                    matched += 1
            _ref_state['done'] = i + 1
        _ref_state.update(running=False, phase='done', result={
            'mismatch': mismatch, 'matched': matched, 'ref': ref_face, 'threshold': threshold, 'group': group})
    except Exception as e:
        _ref_state.update(running=False, phase='error', result={'error': str(e)})


def _dedupe_worker(root, group, threshold):
    import cv2 as _cv2
    import numpy as _np
    try:
        gd = resolve_group(root, group)
        faces = sorted(f for f in os.listdir(gd) if f.lower().endswith(('.png', '.jpg', '.jpeg')))
        total = len(faces)
        _dedupe_state.update(running=True, phase='hashing', done=0, total=total, result=None)
        hs = []
        for i, fn in enumerate(faces):
            fp = os.path.join(gd, fn)
            img = None
            try:
                buf = _np.fromfile(fp, dtype=_np.uint8)
                img = _cv2.imdecode(buf, _cv2.IMREAD_COLOR)
            except Exception:
                img = None
            h = None
            if img is not None:
                gray = _cv2.cvtColor(img, _cv2.COLOR_BGR2GRAY)
                rs = _cv2.resize(gray, (32, 32), interpolation=_cv2.INTER_AREA)
                low = _cv2.dct(_np.float32(rs))[:8, :8]
                h = (low > _np.mean(low)).astype(int).flatten()
            hs.append((fn, h))
            _dedupe_state['done'] = i + 1
        _dedupe_state['phase'] = 'comparing'
        parent = list(range(len(hs)))

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        for i in range(len(hs)):
            if hs[i][1] is None:
                continue
            for j in range(i + 1, len(hs)):
                if hs[j][1] is None:
                    continue
                dist = int(_np.sum(hs[i][1] != hs[j][1]))
                if (64 - dist) / 64.0 >= threshold:
                    ri, rj = find(i), find(j)
                    if ri != rj:
                        parent[ri] = rj
            _dedupe_state['done'] = i + 1
        clusters = {}
        for i in range(len(hs)):
            if hs[i][1] is None:
                continue
            clusters.setdefault(find(i), []).append(hs[i][0])
        dupes = []
        out_groups = []
        for members in clusters.values():
            if len(members) > 1:
                out_groups.append({'keep': members[0], 'dupes': members[1:]})
                dupes.extend(members[1:])
        _dedupe_state.update(running=False, phase='done', result={
            'group': group, 'total_images': total, 'dupes_total': len(dupes),
            'dupe_groups': len(out_groups), 'groups': out_groups})
        try:
            with open(os.path.join(root, '.dedupe_result.json'), 'w', encoding='utf-8') as fh:
                json.dump(_dedupe_state['result'], fh, ensure_ascii=False)
        except Exception:
            pass
    except Exception as e:
        _dedupe_state.update(running=False, phase='error', result={'error': str(e)})


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


def thumb_bytes(fp, root, key, size=400):
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
             "ForEach-Object { $_.ProcessId }"],
            timeout=15).decode()
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
