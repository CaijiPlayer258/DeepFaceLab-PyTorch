'use strict';
/* 分趟导出的流程图（多模型换多人脸）。

  要求（用户口径，逐条落实）：
  · 显示**后端真实**的流水线，不是装饰；节点由后端 status.pipeline 决定；
  · 立刻显示，不等管线完全启动；
  · 节点名用真实名字：TinyMog#0 / <模型>#0 / XSegLite#0 / Compose#0 / <编码器>#0；
  · 人脸暂存池**竖着**画，标题在上，下面三行（张数 / 百分比 / 阶段 i/N）；
  · 闸门：所有趟跑完才开，关着时后面整段灰；
  · 无 XSeg 时遮罩那格整格消失，池直接接在换脸后面；
  · 它必须像**流程图**：有连线、有方向，不是几块卡片。
*/

(function () {
  var NS = "http://www.w3.org/2000/svg";
  function mk(tag, attrs) {
    var el = document.createElementNS(NS, tag);
    for (var k in (attrs || {})) if (attrs[k] != null) el.setAttribute(k, attrs[k]);
    return el;
  }

  /* 一个环节：圆角矩形 + 名字 + 副字 */
  function node(g, x, y, w, h, label, sub, on) {
    var state = on ? "on" : "";
    g.appendChild(mk("rect", { x: x, y: y, width: w, height: h, rx: 8,
                              "class": "fnode" + (state ? " " + state : "") }));
    var t = mk("text", { x: x + w / 2, y: y + (sub ? h / 2 - 2 : h / 2 + 4),
                          "text-anchor": "middle", "class": "t-lane" });
    t.textContent = label; g.appendChild(t);
    if (sub) {
      var s2 = mk("text", { x: x + w / 2, y: y + h / 2 + 13, "text-anchor": "middle",
                            "class": "t-count" });
      s2.textContent = sub; g.appendChild(s2);
    }
  }

  /* 带箭头的连线 */
  function arrow(g, x1, y1, x2, y2) {
    g.appendChild(mk("path", { d: "M " + x1 + " " + y1 + " L " + x2 + " " + y2,
                               "class": "fedge", "marker-end": "url(#fArrow)" }));
  }

  function pool(g, x, y, w, h, faces, total, stageTxt, scanning) {
    g.appendChild(mk("rect", { x: x, y: y, width: w, height: h, rx: 8, "class": "fpool" }));
    var pct = total ? Math.min(1, faces / total) : 0;
    var fillH = Math.max(0, h * pct);
    g.appendChild(mk("rect", { x: x, y: y + (h - fillH), width: w, height: fillH,
                               rx: 8, "class": "fpoolwater" }));
    var ttl = mk("text", { x: x + w / 2, y: y - 12, "text-anchor": "middle", "class": "t-stage" });
    ttl.textContent = "人脸暂存池"; g.appendChild(ttl);
    var l1 = mk("text", { x: x + w + 12, y: y + 30, "class": "t-count" });
    l1.textContent = scanning ? (faces + " 张") : (faces + " / " + (total || "?") + " 张");
    g.appendChild(l1);
    var l2 = mk("text", { x: x + w + 12, y: y + 48, "class": "t-count" });
    l2.textContent = scanning ? "占比未知" : (total ? (Math.round(pct * 1000) / 10) + "%" : "—");
    g.appendChild(l2);
    var l3 = mk("text", { x: x + w + 12, y: y + 66, "class": "t-count" });
    l3.textContent = stageTxt || ""; g.appendChild(l3);
  }

  function draw(svg, d) {
    while (svg.firstChild) svg.removeChild(svg.firstChild);
    var defs = mk("defs");
    var m = mk("marker", { id: "fArrow", viewBox: "0 0 8 8", refX: 7, refY: 4,
                            markerWidth: 5, markerHeight: 5, orient: "auto" });
    m.appendChild(mk("path", { d: "M0,0 L8,4 L0,8 z", "class": "farrowhead" }));
    defs.appendChild(m); svg.appendChild(defs);
    var g = mk("g"); svg.appendChild(g);

    var cur = String(d.work_phase || d.phase || "");
    var maskOn = (d.mask_on !== false);
    var pi = d.pass_info || {};
    var nPass = pi.n || 2;
    var label = pi.label || "模型";
    var poolFaces = (d.pool || {}).faces || 0;
    var poolTotal = (d.pool || {}).faces_total || 0;
    var gateOpen = (cur === "merge" || cur === "encode");
    var scanning = (cur === "scan") || (d.pool_frozen === false);

    var NW = 92, NH = 40, GAP = 30;
    var rows = [{ y: 60, label: "① 扫描（每个模型之前只做一次）", on: cur === "scan" },
                { y: 178, label: "② 逐模型趟（一次只装一个模型，串行）", on: cur === "pass" }];
    for (var r = 0; r < rows.length; r++) {
      var band = rows[r];
      g.appendChild(mk("rect", { x: 20, y: band.y - 30, width: 1160, height: 96, rx: 10,
                                 "class": "fband" + (band.on ? " on" : "") }));
      var bt = mk("text", { x: 34, y: band.y - 12, "class": "t-stage" + (band.on ? " on" : "") });
      bt.textContent = band.label; g.appendChild(bt);
    }

    /* ---- ① 扫描 ---- */
    var y = rows[0].y + 4, x = 40;
    node(g, x, y, NW, NH, "解码", null, cur === "scan");
    x += NW; arrow(g, x + 2, y + NH / 2, x + GAP - 2, y + NH / 2); x += GAP;
    node(g, x, y, NW + 16, NH, "TinyMog#0", "检测", cur === "scan");
    x += NW + 16; arrow(g, x + 2, y + NH / 2, x + GAP - 2, y + NH / 2); x += GAP;
    node(g, x, y, NW + 16, NH, "68 点", "对齐 / 归组", cur === "scan");
    x += NW + 16; arrow(g, x + 2, y + NH / 2, x + GAP - 2, y + NH / 2); x += GAP;
    node(g, x, y, NW + 26, NH, "分配表", "index.json", cur === "scan");

    /* ---- ② 逐模型趟 → 池 → 闸门 → 合流 ---- */
    y = rows[1].y + 4; x = 40;
    node(g, x, y, NW, NH, "解码", null, cur === "pass");
    x += NW; arrow(g, x + 2, y + NH / 2, x + GAP - 2, y + NH / 2); x += GAP;
    node(g, x, y, NW, NH, "对齐", "68 点", cur === "pass");
    x += NW; arrow(g, x + 2, y + NH / 2, x + GAP - 2, y + NH / 2); x += GAP;
    node(g, x, y, NW + 26, NH, label + "#0", "换脸", cur === "pass");
    x += NW + 26;
    if (maskOn) {
      arrow(g, x + 2, y + NH / 2, x + GAP - 2, y + NH / 2); x += GAP;
      node(g, x, y, NW + 16, NH, "XSegLite#0", "遮罩", cur === "pass");
      x += NW + 16;
    }

    /* 池：竖着，标签在**右侧**三行，闸门再往右让开 */
    var pw = 34, ph = 72, px = x + 22, py = y + 2;
    arrow(g, x + 2, y + NH / 2, px - 6, y + NH / 2);
    pool(g, px, py, pw, ph, poolFaces, poolTotal,
         scanning ? "" : (pi.i ? ("阶段 " + pi.i + "/" + pi.n) : ""), scanning);

    var gx = px + pw + 118, gy = y;
    arrow(g, px + pw + 2, y + NH / 2, gx - 4, gy + NH / 2);
    node(g, gx, gy, 58, NH, "闸门", gateOpen ? "已开" : "关闭", gateOpen);

    x = gx + 58; arrow(g, x + 2, gy + NH / 2, x + GAP - 2, gy + NH / 2); x += GAP;
    node(g, x, gy, NW + 16, NH, "Compose#0", "每帧混一次", gateOpen);
    x += NW + 16; arrow(g, x + 2, gy + NH / 2, x + GAP - 2, gy + NH / 2); x += GAP;
    node(g, x, gy, NW + 40, NH, "h264_nvenc#0", "编码", gateOpen);
    x += NW + 40; arrow(g, x + 2, gy + NH / 2, x + GAP - 2, gy + NH / 2); x += GAP;
    node(g, x, gy, 58, NH, "成片", null, gateOpen);

    if (!gateOpen) {
      var tip = mk("text", { x: 40, y: 268, "class": "t-count" });
      tip.textContent = "闸门关着：等 " + nPass + " 趟写完池子才开合流（此刻合成 0 帧是正常的）";
      g.appendChild(tip);
    }
  }
  window.mgPipePhases = function (d) {
    var svg = document.getElementById("mgPipeSvg");
    if (!svg) return false;
    try { draw(svg, d || {}); return true; } catch (e) { return false; }
  };

  /* 自洽轮询：不依赖 MG_PIPE 等内部符号，只依赖 DOM。 */
  var lastSig = "";
  function syncSub(d) {
    var sub = document.getElementById("mgPipeSub");
    if (!sub || !d) return;
    if (d.mode === "passes") sub.textContent = "分趟 · " + (d.mode_why || "") + " · 闸门后才合成";
    else if (d.mode === "fused") sub.textContent = "快速导出 · " + (d.mode_why || "");
  }
  function tick() {
    var svg = document.getElementById("mgPipeSvg");
    if (!svg || document.visibilityState === "hidden") return;
    fetch("/api/merge/export/status").then(function (r) { return r.json(); }).then(function (d) {
      syncSub(d);
      if (!d || !d.pipeline || d.pipeline.shape !== "passes") return;
      var sig = [d.work_phase, (d.pool || {}).faces, (d.pass_info || {}).i, (d.pass_info || {}).label].join("|");
      if (sig === lastSig) return;
      lastSig = sig;
      window.mgPipePhases(d);
    }).catch(function () {});
  }
  setInterval(tick, 1000);
  tick();
})();
