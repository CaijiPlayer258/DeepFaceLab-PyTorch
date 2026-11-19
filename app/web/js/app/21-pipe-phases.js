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
  function node(g, x, y, w, h, label, sub, state) {
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
    var ttl = mk("text", { x: x + w / 2, y: y - 8, "text-anchor": "middle", "class": "t-stage" });
    ttl.textContent = "人脸暂存池"; g.appendChild(ttl);
    var l1 = mk("text", { x: x + w / 2, y: y + h + 16, "text-anchor": "middle", "class": "t-count" });
    l1.textContent = scanning ? (faces + " 张") : (faces + " / " + (total || "?") + " 张");
    var l2 = mk("text", { x: x + w / 2, y: y + h + 30, "text-anchor": "middle", "class": "t-count" });
    l2.textContent = scanning ? "占比未知" : (total ? (Math.round(pct * 1000) / 10) + "%" : "—");
    g.appendChild(l2);
    var l3 = mk("text", { x: x + w / 2, y: y + h + 44, "text-anchor": "middle", "class": "t-count" });
    l3.textContent = stageTxt || ""; g.appendChild(l3);
  }

  function draw(svg, d) {
    var W = 1200, H = 340;
    while (svg.firstChild) svg.removeChild(svg.firstChild);
    var defs = mk("defs");
    var m = mk("marker", { id: "fArrow", viewBox: "0 0 8 8", refX: 7, refY: 4,
                            markerWidth: 6, markerHeight: 6, orient: "auto" });
    m.appendChild(mk("path", { d: "M0,0 L8,4 L0,8 z", "class": "farrowhead" }));
    defs.appendChild(m); svg.appendChild(defs);
    var g = mk("g"); svg.appendChild(g);

    var cur = String(d.work_phase || d.phase || "");
    var maskOn = (d.mask_on !== false);   // 后端说本单没有遮罩模型 -> 遮罩那格整格不画
    var pi = d.pass_info || {};
    var nPass = pi.n || 2;
    var passLabel = pi.label || "模型";
    var poolFaces = (d.pool || {}).faces || 0;
    var poolTotal = (d.pool || {}).faces_total || 0;
    var gateOpen = (cur === "merge" || cur === "encode");

    /* 两行固定布局：宽度先算好再画，保证 1200 内放得下（上一版溢出，成片被切掉）。 */
    var NW = 104, NH = 46, GAP = 24, PAD = 28;
    var rowY = [58, 168];
    function rowLabel(y, txt, on) {
      var t = mk("text", { x: PAD, y: y - 14, "class": "t-stage" });
      t.textContent = txt;
      if (on) t.setAttribute("style", "fill:var(--accent,#4ea1ff)");
      g.appendChild(t);
    }

    /* ---- ① 扫描（每个模型之前只做一次） ---- */
    rowLabel(rowY[0], "① 扫描（一次）", cur === "scan");
    var x = PAD, y = rowY[0];
    node(g, x, y, NW, NH, "解码", null, cur === "scan" ? "on" : "");
    x += NW; arrow(g, x + 2, y + NH / 2, x + GAP - 2, y + NH / 2); x += GAP;
    node(g, x, y, NW + 12, NH, "TinyMog#0", "检测", cur === "scan" ? "on" : "");
    x += NW + 12; arrow(g, x + 2, y + NH / 2, x + GAP - 2, y + NH / 2); x += GAP;
    node(g, x, y, NW, NH, "68 点", "对齐/归组", cur === "scan" ? "on" : "");
    x += NW; arrow(g, x + 2, y + NH / 2, x + GAP - 2, y + NH / 2); x += GAP;
    node(g, x, y, NW + 20, NH, "分配表", "index.json", cur === "scan" ? "on" : "");

    /* ---- ② 逐模型趟（串行）→ 池 → 闸门 → 合流 → 成片 ---- */
    rowLabel(rowY[1], "② 逐模型趟（一次只装一个模型）", cur === "pass");
    x = PAD; y = rowY[1];
    node(g, x, y, NW, NH, "解码", null, cur === "pass" ? "on" : "");
    x += NW; arrow(g, x + 2, y + NH / 2, x + GAP - 2, y + NH / 2); x += GAP;
    node(g, x, y, NW, NH, "对齐", "68 点", cur === "pass" ? "on" : "");
    x += NW; arrow(g, x + 2, y + NH / 2, x + GAP - 2, y + NH / 2); x += GAP;
    node(g, x, y, NW + 28, NH, passLabel + "#0", "换脸", cur === "pass" ? "on" : "");
    x += NW + 28;
    if (maskOn) {
      arrow(g, x + 2, y + NH / 2, x + GAP - 2, y + NH / 2); x += GAP;
      node(g, x, y, NW + 16, NH, "XSegLite#0", "遮罩", cur === "pass" ? "on" : "");
      x += NW + 16;
    }

    /* 池：竖着挂在遮罩之后，标签在池子右侧（上一版压在下面，看不清） */
    var pw = 40, ph = 96, px = x + 18, py = y - 6;
    arrow(g, x + 2, y + NH / 2, px - 4, y + NH / 2);
    var scanning = (cur === "scan") || (d.pool_frozen === false);
    pool(g, px, py, pw, ph, poolFaces, poolTotal,
         scanning ? "已发现（还在涨）" : (pi.i ? ("阶段 " + pi.i + "/" + pi.n) : ""),
         scanning);

    /* 闸门 */
    var gx = px + pw + 96, gy = y;
    arrow(g, px + pw + 2, y + NH / 2, gx - 2, gy + NH / 2);
    node(g, gx, gy, 72, NH, "闸门", gateOpen ? "已开" : "关闭",
         gateOpen ? "on" : "off");

    /* 合流 → 编码 → 成片 */
    x = gx + 72; arrow(g, x + 2, gy + NH / 2, x + GAP - 2, gy + NH / 2); x += GAP;
    node(g, x, gy, NW + 16, NH, "Compose#0", "每帧混一次", gateOpen ? "on" : "off");
    x += NW + 16; arrow(g, x + 2, gy + NH / 2, x + GAP - 2, gy + NH / 2); x += GAP;
    node(g, x, gy, NW + 40, NH, "h264_nvenc#0", "编码", gateOpen ? "on" : "off");
    x += NW + 40; arrow(g, x + 2, gy + NH / 2, x + GAP - 2, gy + NH / 2); x += GAP;
    node(g, x, gy, 64, NH, "成片", null, gateOpen ? "on" : "off");

    /* 闸门关着时补一句人话（用户要能看懂「为什么后面不动」） */
    if (!gateOpen) {
      var tip = mk("text", { x: PAD, y: 306, "class": "t-count" });
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
