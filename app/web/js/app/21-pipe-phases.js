'use strict';
/* 分趟：在**经典六列图**里加一只「人脸暂存池」。
   位置：没选 XSeg → 插在 dfm(换脸) 列之后；选了 XSeg → 插在 xseg(遮罩) 列之后。
   做法：不碰 16-pipe-draw 的渲染，只在两列之间的空隙里补一个竖的水箱 + 三行读数。
   为什么这样：那几个列是由 KEYS 画出来的，改动它就要动 300 行的老渲染器；
   而用户要的只是「多一只池子」，补在列间缝隙里既不动老图，也不会挤乱它。 */
(function () {
  var NS = 'http://www.w3.org/2000/svg';

  function poolAnchor() {
    var svg = document.getElementById('mgPipeSvg');
    if (!svg) return null;
    var heads = [].slice.call(svg.querySelectorAll('text.t-stage'));
    if (!heads.length) return null;
    var xs = heads.map(function (t) { return { x: parseFloat(t.getAttribute('x')), t: t }; });
    var i = -1;
    for (var k = 0; k < xs.length; k++) {
      var txt = String(xs[k].t.textContent || '');
      if (/换脸|#0/.test(txt) && i < 0) i = k;          /* 先记住换脸那列 */
    }
    var wantXseg = (window.__mgPoolAfter || 'xseg') === 'xseg';
    var idx = 0;
    for (var j = 0; j < xs.length; j++) {
      var s2 = String(xs[j].t.textContent || '');
      if (wantXseg && /遮罩|XSeg/i.test(s2)) { idx = j; break; }
      if (!wantXseg && /换脸/.test(s2)) { idx = j; break; }
    }
    var x = (xs[idx].x + xs[idx + 1].x) / 2;
    var top = 70, h = 150;
    return { x: x, top: top, h: h };
  }

  function drawPool(d) {
    var svg = document.getElementById('mgPipeSvg');
    if (!svg) return;
    var old = svg.querySelector('g[data-pool-overlay]');
    if (old) svg.removeChild(old);
    if (!d || d.mode !== 'passes') return;
    var a = poolAnchor();
    if (!a) return;
    var faces = (d.pool || {}).faces || 0;
    var total = (d.pool || {}).faces_total || 0;
    var scanning = (String(d.work_phase || '') === 'scan') || (d.pool_frozen === false);
    var pct = total ? Math.min(1, faces / total) : 0;
    var pw = 28;
    var g = document.createElementNS(NS, 'g');
    g.setAttribute('data-pool-overlay', '1');
    function el(tag, at) {
      var e = document.createElementNS(NS, tag);
      for (var k in at) if (at[k] != null) e.setAttribute(k, at[k]);
      return e;
    }
    g.appendChild(el('rect', { x: a.x - pw / 2, y: a.top, width: pw, height: a.h, rx: 7, 'class': 'lane pool' }));
    var fh = a.h * pct;
    if (fh > 1) {
      g.appendChild(el('rect', { x: a.x - pw / 2, y: a.top + a.h - fh, width: pw, height: fh, rx: 7,
                                 'class': 'water poolwater',
                                 style: 'fill:#4ea1ff;fill-opacity:.55' }));
    }
    var t0 = el('text', { x: a.x, y: a.top - 10, 'text-anchor': 'middle', 'class': 't-stage' });
    t0.textContent = '人脸暂存池'; g.appendChild(t0);
    var t1 = el('text', { x: a.x, y: a.top + a.h + 16, 'text-anchor': 'middle', 'class': 't-count' });
    t1.textContent = scanning ? (faces + ' 张') : (faces + ' / ' + total); g.appendChild(t1);
    var t2 = el('text', { x: a.x, y: a.top + a.h + 30, 'text-anchor': 'middle', 'class': 't-count' });
    t2.textContent = scanning ? '已发现（还在涨）' : (Math.round(pct * 1000) / 10) + '%'; g.appendChild(t2);
    var pi = d.pass_info || {};
    var t3 = el('text', { x: a.x, y: a.top + a.h + 44, 'text-anchor': 'middle', 'class': 't-count' });
    t3.textContent = pi.i ? ('趟 ' + pi.i + '/' + pi.n) : ''; g.appendChild(t3);
    svg.appendChild(g);
  }

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
      if (d && d.mode === 'passes') window.__mgPoolAfter = (d.mask_on === false) ? 'dfm' : 'xseg';
      /* 分趟这条路没有自己的轮询驱动经典图 —— 这里把 status 交给它，
         否则 lane 永远是「— 帧」、光点一次都不放（用户看到的就是"没动画"）。 */
      if (d && d.mode === 'passes') {
        try {
          if (typeof mgpStatusPaint === 'function') mgpStatusPaint(d);
          else if (typeof window.mgpPipe === 'function') window.mgpPipe(d);
        } catch (e) { /* 渲染器内部报错不影响轮询 */ }
      }
      var sig = [d.mode, d.work_phase, (d.pool || {}).faces, (d.pass_info || {}).i, (d.pass_info || {}).label].join('|');
      if (sig === last) return;
      last = sig;
      drawPool(d);
    }).catch(function () {});
  }
  setInterval(tick, 1000);
  tick();
})();
