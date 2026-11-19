'use strict';
/* 分趟流程图：**经典那套视觉**（列头 + lane + 收腰曲线 + 光点 + 成片杯），
   但列由后端 status.pipeline.nodes 决定、等宽。池在自己那一列里竖着画。
   16-pipe-draw 里的 KEYS 是分片内的私有变量，window 上够不到，所以这里按同一套
   常量与画法复刻一遍（W=1200 / PADL=56 / TOPY=58 / 列等分），保证观感一致。 */
(function () {
  var NS = 'http://www.w3.org/2000/svg';
  var W = 1200, H = 300, PADL = 56, TOPY = 58, BOXH = 26;
  function mk(tag, a) {
    var e = document.createElementNS(NS, tag);
    for (var k in (a || {})) if (a[k] != null) e.setAttribute(k, a[k]);
    return e;
  }

  function draw(svg, d) {
    while (svg.firstChild) svg.removeChild(svg.firstChild);
    var nodes = ((d.pipeline || {}).nodes) || [];
    if (!nodes.length) return false;
    var cur = String(d.work_phase || '');
    var pi = d.pass_info || {};
    var scanning = (cur === 'scan') || (d.pool_frozen === false);
    var gateOpen = (cur === 'merge' || cur === 'encode');
    var faces = (d.pool || {}).faces || 0, total = (d.pool || {}).faces_total || 0;
    var pct = total ? Math.min(1, faces / total) : 0;
    var n = nodes.length;
    var colW = (W - PADL * 2) / n;
    var g = mk('g'); svg.appendChild(g);
    var defs = mk('defs');
    var mk2 = mk('marker', { id: 'a1', viewBox: '0 0 8 8', refX: 7, refY: 4, markerWidth: 5, markerHeight: 5, orient: 'auto' });
    mk2.appendChild(mk('path', { d: 'M0,0 L8,4 L0,8 z', 'class': 'farrowhead' }));
    defs.appendChild(mk2); svg.appendChild(defs);

    var cx = [], on = [], yBox = TOPY + 6;
    for (var i = 0; i < n; i++) {
      var nd = nodes[i], x = PADL + i * colW;
      cx.push(x);
      var st = (nd.id === 'dec' || nd.id === 'det' || nd.id === 'align') ? (cur === 'scan')
             : (nd.id === 'dfm' || nd.id === 'xse') ? (cur === 'pass')
             : (nd.id === 'pool') ? (cur === 'pass' || cur === 'scan')
             : (gateOpen);
      on.push(st);
      var lbl = String(nd.label || nd.id);
      if (nd.id === 'dfm' && pi.label) lbl = pi.label + '#0';
      var t1 = mk('text', { x: x, y: 24, 'text-anchor': 'middle', 'class': 't-stage' + (st ? ' on' : '') });
      t1.textContent = lbl; g.appendChild(t1);
      var cnt = '';
      if (nd.id === 'pool') cnt = scanning ? (faces + ' 张') : (faces + ' / ' + total);
      else if (nd.id === 'out') cnt = gateOpen ? (d.vid && d.vid.frames_done ? d.vid.frames_done + ' 帧' : '') : '';
      else if (pi.i && nd.id === 'dfm') cnt = '趟 ' + pi.i + '/' + pi.n;
      if (cnt) {
        var t2 = mk('text', { x: x, y: 40, 'text-anchor': 'middle', 'class': 't-count' });
        t2.textContent = cnt; g.appendChild(t2);
      }
      if (nd.id === 'pool') {
        /* 池：竖着，占满这一列的竖向空间 */
        var ph = 118, pw = 30, py = yBox + 6;
        g.appendChild(mk('rect', { x: x - pw / 2, y: py, width: pw, height: ph, rx: 7, 'class': 'lane pool' }));
        var fh = ph * pct;
        g.appendChild(mk('rect', { x: x - pw / 2, y: py + ph - fh, width: pw, height: fh, rx: 7, 'class': 'water poolwater' }));
        var pl = mk('text', { x: x, y: py + ph + 18, 'text-anchor': 'middle', 'class': 't-count' });
        pl.textContent = scanning ? '已发现（还在涨）' : (Math.round(pct * 1000) / 10) + '%';
        g.appendChild(pl);
        continue;
      }
      g.appendChild(mk('rect', { x: x - 37, y: yBox, width: 74, height: BOXH, rx: 7,
                                 'class': 'lane' + (st ? '' : ' off') }));
      var tl = mk('text', { x: x, y: yBox + BOXH / 2 + 3.2, 'text-anchor': 'middle', 'class': 't-lane' });
      var stateTxt = (nd.id === 'gate') ? (gateOpen ? '开' : '关')
                    : !st ? '等待' : (cur === 'scan' ? '扫描中'
                    : (cur === 'pass' ? '换脸中' : '合成中'));
      tl.textContent = stateTxt;
      g.appendChild(tl);
    }

    /* 收腰曲线：每两列之间从盒右 → 腰点 → 盒左（经典图的画法） */
    var midY = yBox + BOXH / 2;
    for (var j = 0; j < n - 1; j++) {
      var x1 = cx[j] + 37, x2 = cx[j + 1] - 37;
      if (x2 - x1 < 4) continue;
      var wx = (x1 + x2) / 2;
      var d1 = 'M ' + x1 + ' ' + midY + ' C ' + wx + ' ' + midY + ', ' + wx + ' ' + midY + ', ' + x2 + ' ' + midY;
      g.appendChild(mk('path', { d: d1, 'class': 'edge' }));
      for (var q = 0; q < 2; q++) {
        var dot = mk('circle', { r: 2.2, 'class': 'pulse' + (on[j + 1] ? ' on' : '') });
        dot.appendChild(mk('animateMotion', { dur: '2.2s', repeatCount: 'indefinite',
                                           begin: (q * 1.1) + 's', path: d1 }));
        g.appendChild(dot);
      }
    }

    /* 成片杯：最后一列右侧的竖桶 */
    var ocx = W - PADL + 6;
    g.appendChild(mk('rect', { x: ocx - 22, y: yBox - 6, width: 44, height: 150, rx: 8, 'class': 'outcup' }));
    var oh = 150 * (gateOpen ? 1 : 0);
    if (oh > 0) g.appendChild(mk('rect', { x: ocx - 22, y: yBox - 6 + 150 - oh, width: 44, height: oh, rx: 8, 'class': 'water out' }));
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
