'use strict';
window.__shard21_loaded = (window.__shard21_loaded || 0) + 1;
/* 分相位流水线渲染（多模型换多人脸这条路）。

  为什么单独一个渲染器：fused 那张六列图是**并发视图**（六个阶段同时在跑），
  而 passes 这条路是**串行相位**：扫描 → 逐模型趟（一次只装一个模型，N 趟）→ 闸门 → 合流。
  六列图画不出「一次只跑一个模型」「池只写不读/只读不写」这两件事，
  所以后端发 pipeline.shape=passes 时，这里改画相位图。

  数据全部来自 status：pipeline.phases + phase/pass/pool 三个计数。前端不自己算。 */

(function () {
  function esc(s) { return String(s == null ? "" : s); }

  function chip(x, y, text, on) {
    var w = Math.max(56, text.length * 7.2 + 16);
    var g = document.createElementNS("http://www.w3.org/2000/svg", "g");
    var r = document.createElementNS("http://www.w3.org/2000/svg", "rect");
    r.setAttribute("x", x); r.setAttribute("y", y); r.setAttribute("width", w);
    r.setAttribute("height", 24); r.setAttribute("rx", 6);
    r.setAttribute("class", on ? "phchip on" : "phchip");
    var t = document.createElementNS("http://www.w3.org/2000/svg", "text");
    t.setAttribute("x", x + w / 2); t.setAttribute("y", y + 16);
    t.setAttribute("text-anchor", "middle"); t.setAttribute("class", "t-lane");
    t.textContent = text;
    g.appendChild(r); g.appendChild(t);
    return g;
  }

  function draw(svg, d) {
    var pl = d.pipeline || {};
    var phases = pl.phases || [];
    var cur = String(d.work_phase || d.phase || "");
    while (svg.firstChild) svg.removeChild(svg.firstChild);
    var W = 1200, H = 340;
    var n = phases.length || 4;
    var bw = 250, gap = (W - 80 - n * bw) / Math.max(1, n - 1);
    for (var i = 0; i < phases.length; i++) {
      var ph = phases[i];
      var x = 40 + i * (bw + gap);
      var on = (ph.id === cur) || (ph.id === "pass" && cur === "pass");
      var g = document.createElementNS("http://www.w3.org/2000/svg", "g");
      var box = document.createElementNS("http://www.w3.org/2000/svg", "rect");
      box.setAttribute("x", x); box.setAttribute("y", 70);
      box.setAttribute("width", bw); box.setAttribute("height", 150);
      box.setAttribute("rx", 10);
      box.setAttribute("class", on ? "phbox on" : "phbox");
      g.appendChild(box);
      var t1 = document.createElementNS("http://www.w3.org/2000/svg", "text");
      t1.setAttribute("x", x + 14); t1.setAttribute("y", 96);
      t1.setAttribute("class", "t-stage");
      var lab = esc(ph.label);
      if (ph.id === "pass" && ph.count) lab += " ×" + ph.count;
      t1.textContent = lab; g.appendChild(t1);
      var steps = ph.steps || [];
      for (var j = 0; j < steps.length; j++) {
        g.appendChild(chip(x + 14, 112 + j * 24 > 190 ? 190 : 112 + j * 26, esc(steps[j]), on));
      }
      /* 该相位的计数：只有当前相位有真实数字，其它显示占位 */
      var info = "";
      if (ph.id === "pass" && (d.pass_info || d.pass)) {
        var _ps = d.pass_info || d.pass;
        info = "第 " + _ps.i + "/" + _ps.n + " 趟" + (_ps.label ? " · " + _ps.label : "");
      } else if (ph.id === "merge" && d.vid) {
        info = (d.vid.frames_done || 0) + "/" + (d.vid.frames_total || 0) + " 帧";
      } else if (ph.id === "scan") {
        info = on ? "进行中" : (cur ? "已完成" : "待开始");
      } else if (ph.id === "gate") {
        info = (cur === "merge") ? "已打开" : "关闭（等所有趟结束）";
      }
      if (info) {
        var t2 = document.createElementNS("http://www.w3.org/2000/svg", "text");
        t2.setAttribute("x", x + 14); t2.setAttribute("y", 208);
        t2.setAttribute("class", "t-count");
        t2.textContent = info; g.appendChild(t2);
      }
      svg.appendChild(g);
    }
    /* 人脸池：只在 pass/merge 之间存在，画在闸门下方 */
    var pool = d.pool;
    if (pool && pool.faces_total) {
      var pt = document.createElementNS("http://www.w3.org/2000/svg", "text");
      pt.setAttribute("x", W / 2); pt.setAttribute("y", 268);
      pt.setAttribute("text-anchor", "middle"); pt.setAttribute("class", "t-count");
      pt.textContent = "人脸暂存池（落盘）：" + pool.faces + " / " + pool.faces_total + " 张脸";
      svg.appendChild(pt);
    }
  }

  /* 真正的入口是 window.mgpPipe（页面每拍都调它），不是 18-bypass 里那个同名 paint。
     之前挂错地方 = 前端"啥动静没有"。这里包一层：passes 走相位图，其余原样交回去。 */
  window.mgPipePhases = function (d) {
    var svg = document.getElementById("mgPipeSvg");
    if (!svg) return false;
    try { draw(svg, d || {}); return true; } catch (e) { return false; }
  };

  /* 用 defineProperty 接管 window.mgpPipe：它由 19-eff-drawer.js 赋值，
     而分片按文件名拼接，21 号在 19 号之后执行 —— 直接赋值会被后写的原函数覆盖
     （这就是「副标题一直不变」的原因）。setter 拿住原函数，谁赋值都包在外层。 */
  var _realPipe = null;

  function _syncSub(d) {
    var sub = document.getElementById("mgPipeSub");
    if (!sub) return;
    if (d.mode === "passes") {
      sub.textContent = "分趟 · " + (d.mode_why || "逐模型") + " · 池在闸门后才合成";
    } else if (d.mode === "fused") {
      sub.textContent = "快速导出 · " + (d.mode_why || "单模型");
    }
  }
  window.mgPipeSubSync = _syncSub;

  function _wrappedPipe(d) {
    d = d || {};
    try { _syncSub(d); } catch (e) { /* 纯展示 */ }
    try {
      if (d.pipeline && d.pipeline.shape === "passes" && window.mgPipePhases(d)) return;
    } catch (e) { /* 失败就退回原渲染器 */ }
    if (_realPipe) _realPipe(d);
  }

  /* 真正被进度页调用的入口是 MG_PIPE.paint（19-eff-drawer.js: MG_PIPE.paint(d)），
     不是 window.mgpPipe —— 之前两轮都挂错了地方，所以「怎么都不生效」。
     分片共享同一个 IIFE 作用域，这里能直接拿到 MG_PIPE。 */
  try {
    var _MP = window.MG_PIPE;
    if (_MP && _MP.paint) {
      var _origPaint = _MP.paint;
      _MP.paint = function (d) {
        try { _syncSub(d || {}); } catch (e) { /* 纯展示 */ }
        try {
          if (d && d.pipeline && d.pipeline.shape === "passes" && window.mgPipePhases(d)) return;
        } catch (e) { /* 退回原渲染器 */ }
        return _origPaint.call(_MP, d);
      };
    }
  } catch (e) { /* 没有 MG_PIPE 就只保留轮询侧那一道保险 */ }

  try {
    Object.defineProperty(window, "mgpPipe", {
      configurable: true,
      get: function () { return _wrappedPipe; },
      set: function (fn) { _realPipe = fn; }
    });
  } catch (e) {
    _realPipe = window.mgpPipe;
    window.mgpPipe = _wrappedPipe;
  }
})();

/* ------------------------------------------------------------------
   自洽渲染器：直接读 /api/merge/export/status 自己画。
   为什么要这样：MG_PIPE 在分片作用域里（window 上没有），前几轮包装全部落空；
   这里不再依赖任何内部符号，只依赖 DOM id（#mgPipeSvg / #mgPipeSub）。
   只在「导出进度」这块可见时轮询，其它页面不打扰后端。
   ------------------------------------------------------------------ */
(function () {
  var TICK = 1000;
  var lastSig = "";

  function visible(el) {
    return !!(el && (el.offsetWidth || el.offsetHeight || el.getClientRects().length));
  }

  function render(d) {
    var svg = document.getElementById("mgPipeSvg");
    if (!svg || !d || !d.pipeline || d.pipeline.shape !== "passes") return false;
    if (window.mgPipePhases && window.mgPipePhases(d)) return true;
    return false;
  }

  function syncSub(d) {
    var sub = document.getElementById("mgPipeSub");
    if (!sub || !d) return;
    if (d.mode === "passes") {
      sub.textContent = "分趟 · " + (d.mode_why || "") + " · 闸门后才合成";
    } else if (d.mode === "fused") {
      sub.textContent = "快速导出 · " + (d.mode_why || "");
    }
  }

  function tick() {
    var svg = document.getElementById("mgPipeSvg");
    if (!svg) return;                          // 没有这块就别发请求
    if (document.visibilityState === "hidden") return;
    fetch("/api/merge/export/status").then(function (r) { return r.json(); }).then(function (d) {
      var sig = [d.mode, (d.pipeline || {}).shape, d.work_phase,
                 (d.pool || {}).faces, (d.pass_info || {}).i].join("|");
      syncSub(d);
      if (sig === lastSig) return;
      lastSig = sig;
      render(d);
    }).catch(function () { /* 网络抖动忽略 */ });
  }

  setInterval(tick, TICK);
  tick();
})();
