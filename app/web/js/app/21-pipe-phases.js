'use strict';
/* 分趟导出的「旁路」叠层：**保留经典六列图**，只在它上面补几笔。

  用户口径：在之前那版经典图（七等分列 + lane 方块 + 贝塞尔曲线 + 成片杯）上改，
  加几条旁路，而不是另画一套表格。
  这里只做三件事：
    1. 在「遮罩」和「合成」两列之间插一个人脸暂存池（竖的）+ 闸门；
    2. 从池子到合成列画一条虚线旁路（和经典图里无脸帧那条同风格）；
    3. 左上角写一行相位（趟 i/N · 模型名）。
  几何照抄 16-pipe-draw.js 的常量，保证落在同一套坐标里。
*/
(function () {
  var NS = "http://www.w3.org/2000/svg";
  var W = 1200, PADL = 56, TOPY = 58, NCOL = 7;
  var COLW = (W - PADL * 2) / NCOL;
  function mk(tag, attrs) {
    var el = document.createElementNS(NS, tag);
    for (var k in (attrs || {})) if (attrs[k] != null) el.setAttribute(k, attrs[k]);
    return el;
  }

  function overlay(d) {
    var svg = document.getElementById("mgPipeSvg");
    if (!svg) return false;
    var old = svg.querySelector("g[data-pool-overlay]");
    if (old) svg.removeChild(old);
    if (!d || !d.pipeline || d.pipeline.shape !== "passes") return false;

    var cur = String(d.work_phase || "");
    var pi = d.pass_info || {};
    var scanning = (cur === "scan") || (d.pool_frozen === false);
    var gateOpen = (cur === "merge" || cur === "encode");
    var faces = (d.pool || {}).faces || 0;
    var total = (d.pool || {}).faces_total || 0;
    var pct = total ? Math.min(1, faces / total) : 0;

    var g = mk("g", { "data-pool-overlay": "1" });
    var x0 = PADL + 3 * COLW, x1 = PADL + 4 * COLW;
    var px = (x0 + x1) / 2 - 17, py = TOPY + 44, pw = 34, ph = 96;

    g.appendChild(mk("path", {
      d: "M " + (x0 + 40) + " " + (py + ph / 2) + " L " + (px - 6) + " " + (py + ph / 2),
      "class": "edge poolin" }));
    g.appendChild(mk("rect", { x: px, y: py, width: pw, height: ph, rx: 6, "class": "lane pool" }));
    var fh = Math.max(0, ph * pct);
    g.appendChild(mk("rect", { x: px, y: py + (ph - fh), width: pw, height: fh, rx: 6,
                               "class": "water poolwater" }));
    var t = mk("text", { x: px + pw / 2, y: py - 8, "text-anchor": "middle", "class": "t-stage" });
    t.textContent = "人脸暂存池"; g.appendChild(t);
    var l1 = mk("text", { x: px + pw / 2, y: py + ph + 16, "text-anchor": "middle", "class": "t-count" });
    l1.textContent = scanning ? (faces + " 张") : (faces + " / " + total + " 张"); g.appendChild(l1);
    var l2 = mk("text", { x: px + pw / 2, y: py + ph + 30, "text-anchor": "middle", "class": "t-count" });
    l2.textContent = scanning ? "已发现（还在涨）" : (Math.round(pct * 1000) / 10) + "%"; g.appendChild(l2);
    var l3 = mk("text", { x: px + pw / 2, y: py + ph + 44, "text-anchor": "middle", "class": "t-count" });
    l3.textContent = pi.i ? ("趟 " + pi.i + "/" + pi.n + (pi.label ? " · " + pi.label : "")) : "";
    g.appendChild(l3);

    var gx = px + pw + 14, gy = py + ph / 2 - 16;
    g.appendChild(mk("path", { d: "M " + (px + pw) + " " + (py + ph / 2) + " L " + gx + " " + (py + ph / 2),
                               "class": "edge" }));
    g.appendChild(mk("rect", { x: gx, y: gy, width: 30, height: 32, rx: 6,
                               "class": "lane gate" + (gateOpen ? " on" : "") }));
    var gt = mk("text", { x: gx + 15, y: gy + 20, "text-anchor": "middle", "class": "t-lane" });
    gt.textContent = gateOpen ? "开" : "关"; g.appendChild(gt);

    g.appendChild(mk("path", {
      d: "M " + (gx + 30) + " " + (py + ph / 2) +
         " C " + (x1 - 70) + " " + (py + ph / 2 + 40) + ", " + (x1 - 30) + " " + (TOPY + 200) +
         ", " + (x1 - 10) + " " + (TOPY + 200),
      "class": "edge bypass poolbypass" + (gateOpen ? "" : " dim") }));

    var badge = mk("text", { x: PADL, y: TOPY - 22, "class": "t-stage" });
    badge.textContent = "阶段 · " + (cur === "scan" ? "扫描" : (cur === "pass" ? "逐模型趟" : "合流")) +
                       (pi.n ? "（" + (pi.i || 0) + "/" + pi.n + " 趟）" : "");
    g.appendChild(badge);
    svg.appendChild(g);
    return true;
  }
  window.mgPipePoolOverlay = overlay;

  var last = '';
  function syncSub(d) {
    var sub = document.getElementById("mgPipeSub");
    if (!sub || !d || !d.mode) return;
    if (d.mode === "passes") sub.textContent = "分趟 · " + (d.mode_why || "") + " · 闸门后才合成";
    else sub.textContent = "快速导出 · " + (d.mode_why || "");
  }
  function tick() {
    var svg = document.getElementById("mgPipeSvg");
    if (!svg || document.visibilityState === "hidden") return;
    fetch("/api/merge/export/status").then(function (r) { return r.json(); }).then(function (d) {
      syncSub(d);
      var sig = [d.mode, d.work_phase, (d.pool || {}).faces, (d.pass_info || {}).i].join("|");
      if (sig === last) return;
      last = sig;
      overlay(d);
    }).catch(function () {});
  }
  setInterval(tick, 1000);
  tick();
})();
