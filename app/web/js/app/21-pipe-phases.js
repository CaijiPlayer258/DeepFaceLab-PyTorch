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
    var t0 = el('text', { x: a.x, y: a.top - 10, 'text-anchor': 'middle', 'class': 't-stage', style: 'fill:#e8ecf5' });
    t0.textContent = '人脸暂存池'; g.appendChild(t0);
    var t1 = el('text', { x: a.x, y: a.top + a.h + 16, 'text-anchor': 'middle', 'class': 't-count', style: 'fill:#c9d2e4' });
    t1.textContent = scanning ? (faces + ' 张') : (faces + ' / ' + total); g.appendChild(t1);
    var t2 = el('text', { x: a.x, y: a.top + a.h + 30, 'text-anchor': 'middle', 'class': 't-count', style: 'fill:#c9d2e4' });
    t2.textContent = scanning ? '已发现（还在涨）' : (Math.round(pct * 1000) / 10) + '%'; g.appendChild(t2);
    var pi = d.pass_info || {};
    var t3 = el('text', { x: a.x, y: a.top + a.h + 44, 'text-anchor': 'middle', 'class': 't-count', style: 'fill:#c9d2e4' });
    t3.textContent = pi.i ? ('趟 ' + pi.i + '/' + pi.n) : ''; g.appendChild(t3);
    svg.appendChild(g);
  }

  /* 池水单独一步：不管上面 paint 成没成，这一笔都要画上（数字对、水位也得对）。 */
  function paintPoolWater(d) {
    var svg = document.getElementById('mgPipeSvg');
    if (!svg) return;
    var old = svg.querySelector('g[data-pool-water]');
    if (old) svg.removeChild(old);
    if (!d || d.mode !== 'passes') return;
    var faces = (d.pool || {}).faces || 0, total = (d.pool || {}).faces_total || 0;
    /* 趟交界时后端会先发一帧 pool=0（这一趟还没写完）—— 只增不减，别让数字闪。 */
    if (faces < (window.__mgPoolMax || 0)) faces = window.__mgPoolMax;
    else window.__mgPoolMax = faces;
    /* 闸门开了：池子开始放水 —— 未合成的部分才是「还在池里」。 */
    var _cmpNow = ((d.stages || {}).compose || {}).ticks || 0;
    if ((d.work_phase === 'merge' || d.work_phase === 'encode') && total) {
      faces = Math.max(0, total - _cmpNow);
    }
    if (!total) return;
    var g = document.createElementNS('http://www.w3.org/2000/svg', 'g');
    g.setAttribute('data-pool-water', '1');
    /* 池子的矩形就在图上（我自己画的那只），按它算水位 */
    var tank = svg.querySelector('rect.pool');
    if (!tank) return;
    var x = +tank.getAttribute('x'), y = +tank.getAttribute('y');
    var w = +tank.getAttribute('width'), h = +tank.getAttribute('height');
    var pct = Math.max(0, Math.min(1, faces / total));
    var fh = h * pct;
    if (fh > 1) {
      var r = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
      r.setAttribute('x', x); r.setAttribute('y', y + h - fh);
      r.setAttribute('width', w); r.setAttribute('height', fh);
      r.setAttribute('rx', 7);
      r.setAttribute('fill', '#4ea1ff');
      r.setAttribute('fill-opacity', '0.55');
      g.appendChild(r);
    }
    svg.appendChild(g);
  }

  /* 分趟：池子必须是**真的一列**，连线才会 遮罩 → 池 → 合成。
     这里只做两件事：切成列位置、把 pool 当作一个 stage 喂给 paint()（含水位）。 */
  var poolAt = null;
  function keepPool(d) {
    window.__mgPoolDebug = window.__mgPoolDebug || { calls: 0, last: null };
    window.__mgPoolDebug.calls += 1;
    window.__mgPoolDebug.last = { mode: d && d.mode, ph: d && d.work_phase,
                                  faces: d && (d.pool || {}).faces,
                                  total: d && (d.pool || {}).faces_total };
    if (!d || d.mode !== 'passes') return;
    var want = (d.mask_on === false) ? 'dfm' : 'xseg';
    if (poolAt !== want) {
      if (typeof window.mgPoolColumn === 'function' && window.mgPoolColumn(want)) poolAt = want;
    }
    var faces = (d.pool || {}).faces || 0, total = (d.pool || {}).faces_total || 0;
    var pi = d.pass_info || {};
    d.pool = d.pool || {};
    d.stages = d.stages || {};
    /* 池的「出库」只在闸门开了之后才算数：
         · 扫描/换脸阶段：ticks 固定 0 —— 池里在攒，不往合成流（否则会有假光点）；
         · 合流阶段：ticks 跟 compose 走 —— 每合成一帧才出库一帧。 */
    var _cmp = ((d.stages || {}).compose || {}).ticks || 0;
    var _ph = String(d.work_phase || '');
    var _flow = (_ph === 'merge' || _ph === 'encode') ? _cmp : 0;
    /* 池列上那个数字只由 paint 写，这里只决定「它该是几」：
         · 扫描/装载/换脸：只增不减（趟交界后端会发一帧 0，别让数字跳）
         · 合流/编码：total − compose（放水）
         · 其它（收工/未开单）：保持上一次的数，不回 0
       以前我这边也写一遍文字，两边格式不同就变成 457 ⇄ 0 一直闪。 */
    var _shown = Number(window.__mgPoolShown || 0);
    if (_ph === 'merge' || _ph === 'encode') {
      _shown = Math.max(0, total - _cmp);
    } else if (_ph === 'scan' || _ph === 'load' || _ph === 'pass') {
      _shown = Math.max(_shown, faces);
    }
    window.__mgPoolShown = _shown;
    d.stages.pool = { ticks: _flow, lanes: ['池'], inflight: 0,
                      level: total ? (_shown / total) : 0 };
    window.__mgPoolStage = d.stages.pool;
  }

  /* 池列自己的读数和的水位：paint 只填 encode 那一杯，池这杯要自己填。
     几何照抄 17-water 的做法：整列当一个杯子，scaleY(level)。 */
  /* 池列自己的读数和水位。
     lane 方块画在 layers.box 那一层（不在 col 里），所以按 data-lane="pool#0" 找；
     水位就用同一块 rect 的 y/h 再叠一块 scaleY(level) 的蓝面。 */
  function paintPoolColumn(d) {
    var svg = document.getElementById('mgPipeSvg');
    if (!svg) return;
    var faces = (d.pool || {}).faces || 0;
    var total = (d.pool || {}).faces_total || 0;
    var lv = total ? Math.max(0, Math.min(1, faces / total)) : 0;
    var col = svg.querySelector('g.col[data-k="pool"]');
    if (col) {
      var lt = col.querySelector('text.t-lane');
      if (lt && (lt.textContent || '').trim() === '—') lt.textContent = '池';
    }
    var lane = svg.querySelector('rect[data-lane^="pool#"]');
    if (!lane) return;
    var key = 'pool-water';
    var w = svg.querySelector('rect[data-pool-water="' + key + '"]');
    if (!w) {
      w = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
      w.setAttribute('data-pool-water', key);
      w.setAttribute('fill', '#4ea1ff');
      w.setAttribute('fill-opacity', '0.55');
      w.setAttribute('rx', lane.getAttribute('rx') || '7');
      lane.parentNode.insertBefore(w, lane.nextSibling);
    }
    w.setAttribute('x', lane.getAttribute('x'));
    w.setAttribute('y', lane.getAttribute('y'));
    w.setAttribute('width', lane.getAttribute('width'));
    w.setAttribute('height', lane.getAttribute('height'));
    w.style.transformOrigin = 'center bottom';
    w.style.transformBox = 'fill-box';
    w.style.transform = 'scaleY(' + (Math.round(lv * 1000) / 1000) + ')';
  }
  var finished = false;
  function finishOnce(d) {
    if (!d || d.mode !== 'passes') return;
    if (d.running) { finished = false; return; }
    /* 刚开单那一拍 running 还是 False、ok 还是 None —— 那不是失败，是还没开始。 */
    if (d.ok === null || d.ok === undefined) return;
    if (finished) return;
    finished = true;
    var okAll = !!d.ok && !d.cancelled;
    try {
      if (okAll && typeof window.toast === 'function') {
        toast("导出完成", String(d.out || '').slice(-60), "success");
      } else if (!okAll && typeof window.toast === 'function') {
        toast(d.cancelled ? "已取消导出" : "导出失败",
              String(d.error || d.msg || '').slice(0, 120), d.cancelled ? "info" : "error");
      }
    } catch (e) {}
    try {
      if (typeof window.mgpResultShow === 'function') {
        window.mgpResultShow(okAll, {
          phase: okAll ? "done" : (d.cancelled ? "cancelled" : "failed"),
          out_video: d.out, out: d.out, fused: false,
          error: d.error || d.msg || ""
        });
      }
    } catch (e) {}
  }

  /* 模型预热（phase=load）：换脸列蓝色闪烁，告诉用户「在装载，还没开始推理」。 */
  function paintLoading(d) {
    var svg = document.getElementById('mgPipeSvg');
    if (!svg) return;
    var col = svg.querySelector('g.col[data-k="dfm"]');
    var lanes = svg.querySelectorAll('rect[data-lane^="dfm#"]');
    var on = !!(d && d.work_phase === 'load');
    if (col) { if (on) col.classList.add('loading'); else col.classList.remove('loading'); }
    [].slice.call(lanes).forEach(function (r) {
      if (on) r.classList.add('loading'); else r.classList.remove('loading');
    });
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
    if (!svg) return;
    fetch('/api/merge/export/status').then(function (r) { return r.json(); }).then(function (d) {
      syncSub(d);
      finishOnce(d);
      window.__mgMode = (d && d.mode) || '';
      /* 分趟这条路以前没人启动 app 自己的轮询（mgFusedPoll），所以 MG_PIPE.paint 一次都没跑过，
         图上的 lane/曲线/光点全是占位。这里只要它还在跑，就确保那条轮询在跑。 */
      if (d && d.mode === 'passes' && d.running && !window.__mgFusedRunning) {
        window.__mgFusedRunning = true;
        if (typeof window.mgFusedStartPoll === 'function') window.mgFusedStartPoll();
      }
      if (d && d.mode === 'passes') window.__mgPoolAfter = (d.mask_on === false) ? 'dfm' : 'xseg';
      /* 分趟这条路没有自己的轮询驱动经典图 —— 这里把 status 交给它，
         否则 lane 永远是「— 帧」、光点一次都不放（用户看到的就是"没动画"）。 */
      if (d && d.mode === 'passes') {
        try {
          keepPool(d);
      var _painted = false;
          if (typeof window.mgPassesPaint === 'function') _painted = window.mgPassesPaint(d);
      paintPoolColumn(d);
      paintLoading(d);
          setTimeout(function () { paintPoolColumn(d); }, 80);
          if (!_painted && typeof mgpStatusPaint === 'function') mgpStatusPaint(d);
          /* 经典图里未点亮的 lane 用的是 --ink-3，深底上几乎看不见；
             分趟这条路把字提亮一档（只改 fill，不动布局）。 */
          var _svg = document.getElementById('mgPipeSvg');
          if (_svg) {
            /* 列标题下那行计数自己写：paint() 的 counts 通道在分趟这条路上不认
               （它走 MGP 的 idx/counts，不是 stages），这里是唯一的真数来源。 */
            var _sg = d.stages || {};
            var _want = ["frames", "det", "dfm", "xseg", "compose", "encode"];
            var _titles = [].slice.call(_svg.querySelectorAll('text.t-stage'));
            var _counts = [].slice.call(_svg.querySelectorAll('text.t-count'));
            _titles.forEach(function (t, i) {
              var k = _want[i];
              if (!k || !_counts[i]) return;
              var v = (_sg[k] || {}).ticks;
              if (typeof v === 'number') _counts[i].textContent = v + ' 帧';
            });
            var _laneName = {
              frames: '解码', det: 'TinyMog#0',
              dfm: ((d.pass_info || {}).label || '模型') + '#0',
              xseg: 'XSegLite#0', compose: 't0', encode: 'e0'
            };
            var _lanes = [].slice.call(_svg.querySelectorAll('text.t-lane'));
            _lanes.forEach(function (t, i) {
              var k = _want[i];
              if (!k) return;
              var _cur = (t.textContent || '').trim();
              if (!_cur || _cur === '—' || _cur === '–' || _cur === '-') {
                t.textContent = _laneName[k] || k;
              }
            });
            /* 深底上看不清的根因：这些文字用的是 --ink-2/--ink-3（次要色）。
               分趟这条路统一提亮：标题 #e8ecf5、计数 #b9c2d6、其余 #c9d2e4。 */
            [].slice.call(_svg.querySelectorAll('text')).forEach(function (t) {
              var c = t.getAttribute('class') || '';
              if (c.indexOf('t-stage') >= 0) t.style.fill = '#e8ecf5';
              else if (c.indexOf('t-count') >= 0) t.style.fill = '#b9c2d6';
              else t.style.fill = '#c9d2e4';
            });
          }
          else if (typeof window.mgpPipe === 'function') window.mgpPipe(d);
        } catch (e) { /* 渲染器内部报错不影响轮询 */ }
      }
      var sig = [d.mode, d.work_phase, (d.pool || {}).faces, (d.pass_info || {}).i, (d.pass_info || {}).label].join('|');
      if (sig === last) return;
      last = sig;
    }).catch(function () {});
  }
  setInterval(tick, 1000);
  tick();
})();
