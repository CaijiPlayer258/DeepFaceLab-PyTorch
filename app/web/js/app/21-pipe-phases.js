'use strict';
/* 分趟流程图（等宽列 + 连线 + 帧流动画）。

  用户口径：
    · 列不是写死的六段 —— 后端 status.pipeline.nodes 给几个就画几个，**等宽**；
    · 必须有连线（帧从一格流到下一格）；
    · 必须有「帧在流」的视觉：连线上跑的光点；
    · 节点名用真实名字（TinyMog#0 / <模型>#0 / XSegLite#0 / Compose#0 / <编码器>#0）；
    · 池竖直、闸门关着时后半段灰、扫描期不给百分比。
*/
(function () {
  var NS = "http://www.w3.org/2000/svg";
  var W = 1200, H = 300, PAD = 40, TOP = 92, BOXH = 46;
  function mk(tag, attrs) {
    var el = document.createElementNS(NS, tag);
    for (var k in (attrs || {})) if (attrs[k] != null) el.setAttribute(k, attrs[k]);
    return el;
  }

  function draw(svg, d) {
    while (svg.firstChild) svg.removeChild(svg.firstChild);
    var pl = d.pipeline || {}; var nodes = pl.nodes || [];
    if (!nodes.length) return false;
    var cur = String(d.work_phase || d.phase || '');
    var pi = d.pass_info || {};
    var scanning = (cur === 'scan') || (d.pool_frozen === false);
    var gateOpen = (cur === 'merge' || cur === 'encode');
    var faces = (d.pool || {}).faces || 0;
    var total = (d.pool || {}).faces_total || 0;
    var pct = total ? Math.min(1, faces / total) : 0;

    var n = nodes.length;
    var colW = (W - PAD * 2) / n;          // 等宽：列宽只由节点数决定
    var bw = Math.min(colW - 18, 104);     // 盒子略窄于列，留出连线的空隙
    var defs = mk('defs');
    var m = mk('marker', { id: 'fArrow', viewBox: '0 0 8 8', refX: 7, refY: 4,
                          markerWidth: 5, markerHeight: 5, orient: 'auto' });
    m.appendChild(mk('path', { d: 'M0,0 L8,4 L0,8 z', 'class': 'farrowhead' }));
    defs.appendChild(m); svg.appendChild(defs);
    var g = mk('g'); svg.appendChild(g);

    var cx = [], states = [], midY = TOP + BOXH / 2;
    for (var i = 0; i < n; i++) {
      var nd = nodes[i];
      var x = PAD + i * colW + colW / 2;
      cx.push(x);
      var on = (nd.id === 'dec' || nd.id === 'det' || nd.id === 'align') ? (cur === 'scan')
             : (nd.id === 'dfm' || nd.id === 'xse' || nd.id === 'pool') ? (cur === 'pass')
             : (nd.id === 'gate' || nd.id === 'cmp' || nd.id === 'enc' || nd.id === 'out') ? gateOpen
             : false;
      states.push(on);
      var label = String(nd.label || nd.id);
      if (nd.id === 'dfm' && pi.label) label = pi.label + '#0';
      if (nd.id === 'pool') {
        /* 池：竖着画在最下面一行，不占列 */
        continue;
      }
      if (nd.id === 'gate') {
        g.appendChild(mk('rect', { x: x - 26, y: TOP + 5, width: 52, height: 36, rx: 7,
                                   'class': 'fnode gate' + (gateOpen ? ' on' : ' off') }));
        var gt = mk('text', { x: x, y: TOP + 28, 'text-anchor': 'middle', 'class': 't-lane' });
        gt.textContent = (gateOpen ? '开' : '关'); g.appendChild(gt);
        var gb = mk('text', { x: x, y: TOP - 10, 'text-anchor': 'middle', 'class': 't-stage' });
        gb.textContent = '闸门'; g.appendChild(gb);
        continue;
      }
      g.appendChild(mk('rect', { x: x - bw / 2, y: TOP, width: bw, height: BOXH, rx: 8,
                                 'class': 'fnode' + (on ? ' on' : ' off') }));
      var t = mk('text', { x: x, y: TOP + BOXH / 2 + 4, 'text-anchor': 'middle', 'class': 't-lane' });
      t.textContent = label; g.appendChild(t);
    }

    /* 连线：相邻两列之间一条带箭头的曲线 + 跑动的光点 */
    for (var j = 0; j < n - 1; j++) {
      var a = cx[j], b2 = cx[j + 1];
      var x1 = a + bw / 2, x2 = b2 - bw / 2;
      var isPool = nodes[j].id === 'pool' || nodes[j + 1].id === 'pool';
      if (isPool) continue;                                  // 池那条单独画
      if (x2 - x1 < 6) continue;
      var dpath = 'M ' + x1 + ' ' + midY + ' C ' + (x1 + (x2 - x1) / 2) + ' ' + (midY - 12) + ', ' +
                  (x1 + (x2 - x1) / 2) + ' ' + (midY + 12) + ', ' + x2 + ' ' + midY;
      g.appendChild(mk('path', { d: dpath, 'class': 'fedge', 'marker-end': 'url(#fArrow)' }));
      /* 帧流动：沿同一条路径跑 3 个点（CSS 动画驱动 dashoffset 更省，但 animateMotion 最稳） */
      for (var k2 = 0; k2 < 3; k2++) {
        var dot = mk('circle', { r: 2.4, 'class': 'fdot' + (states[j + 1] ? ' on' : '') });
        var anim = mk('animateMotion', { dur: '2.4s', repeatCount: 'indefinite',
                                        begin: (k2 * 0.8) + 's', path: dpath });
        dot.appendChild(anim); g.appendChild(dot);
      }
    }

    /* 池：竖着，落在「遮罩」列与「闸门」列之间，画在盒子下方 */
    var poolIdx = -1;
    for (var q = 0; q < n; q++) if (nodes[q].id === 'pool') poolIdx = q;
    if (poolIdx >= 0) {
      var pxp = cx[poolIdx];
      var ph = 64, pw = 30;
      var py = midY - ph / 2;
      /* 前后连线（和别的列一样，水平进出） */
      if (poolIdx > 0) {
        var xa = cx[poolIdx - 1] + bw / 2, xb = pxp - pw / 2;
        if (xb - xa > 4) {
          g.appendChild(mk('path', { d: 'M ' + xa + ' ' + midY + ' L ' + xb + ' ' + midY,
                                     'class': 'fedge', 'marker-end': 'url(#fArrow)' }));
        }
      }
      if (poolIdx < n - 1) {
        var xc = pxp + pw / 2, xd = cx[poolIdx + 1] - bw / 2;
        if (xd - xc > 4) {
          g.appendChild(mk('path', { d: 'M ' + xc + ' ' + midY + ' L ' + xd + ' ' + midY,
                                     'class': 'fedge', 'marker-end': 'url(#fArrow)' }));
        }
      }
      g.appendChild(mk('rect', { x: pxp - pw / 2, y: py, width: pw, height: ph, rx: 6, 'class': 'fpool' }));
      var fh = ph * pct;
      g.appendChild(mk('rect', { x: pxp - pw / 2, y: py + ph - fh, width: pw, height: fh, rx: 6, 'class': 'fpoolwater' }));
      var pt = mk('text', { x: pxp, y: py - 8, 'text-anchor': 'middle', 'class': 't-stage' });
      pt.textContent = '人脸暂存池'; g.appendChild(pt);
      var pl1 = mk('text', { x: pxp, y: py + ph + 16, 'text-anchor': 'middle', 'class': 't-count' });
      pl1.textContent = scanning ? (faces + ' 张') : (faces + ' / ' + total + ' 张'); g.appendChild(pl1);
      var pl2 = mk('text', { x: pxp, y: py + ph + 30, 'text-anchor': 'middle', 'class': 't-count' });
      pl2.textContent = scanning ? '已发现（还在涨）' : (Math.round(pct * 1000) / 10) + '%'; g.appendChild(pl2);
      var pl3 = mk('text', { x: pxp, y: py + ph + 44, 'text-anchor': 'middle', 'class': 't-count' });
      pl3.textContent = pi.i ? ('趟 ' + pi.i + '/' + pi.n) : ''; g.appendChild(pl3);
    }
    return true;
  }
  window.mgPipeFlowDraw = draw;

  function syncSub(d) {
    var sub = document.getElementById('mgPipeSub');
    if (!sub || !d || !d.mode) return;
    if (d.mode === 'passes') sub.textContent = '分趟 · ' + (d.mode_why || '') + ' · 闸门后才合成';
    else sub.textContent = '快速导出 · ' + (d.mode_why || '');
  }
  var last = '';
  function tick() {
    var svg = document.getElementById('mgPipeSvg');
    if (!svg || document.visibilityState === 'hidden') return;
    fetch('/api/merge/export/status').then(function (r) { return r.json(); }).then(function (d) {
      syncSub(d);
      if (!d || !d.pipeline || d.pipeline.shape !== 'passes') return;
      var sig = [d.work_phase, (d.pool || {}).faces, (d.pass_info || {}).i].join('|');
      if (sig === last) return;
      last = sig;
      draw(svg, d);
    }).catch(function () {});
  }
  setInterval(tick, 1000);
  tick();
})();
