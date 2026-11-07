/* TF 时代（DFL 1.x / 2.0）dat 的迁移提示 —— 参数页顶部那一行，可展开明细。
 *
 * 为什么单独一个文件：app.js 是个几千行的巨型 IIFE，从里面抽单个函数既不好改、
 * 也没法单独跑断言。这里只依赖 document.createElement / appendChild / textContent，
 * 所以 Node 里塞一个极小的 DOM 就能直接验证「该出现时出现、新格式时一行都不画」——
 * require() 它，喂一个 {detected:...} 的对象，看返回值 / childNodes 就够了。
 *
 * 数据来自 /api/models/{name}/config 的 legacy 字段（后端 server/_legacy.py）：
 *   { detected, counts, report, reasons, weights, needs_convert, hint, note, command }
 */
(function (root) {
  "use strict";

  function el(tag, cls, text) {
    var e = document.createElement(tag);
    if (cls) e.className = cls;
    if (text !== undefined && text !== null) e.textContent = String(text);
    return e;
  }

  function clear(node) {
    while (node && node.firstChild) node.removeChild(node.firstChild);
  }

  /* 「迁移 3 · 改键 5 · 忽略 2 …」：计数为 0 的项不占地方，一整行读得完 */
  function countsLine(counts) {
    var order = ["迁移", "改键", "忽略", "补默认", "无法识别"];
    var parts = [];
    counts = counts || {};
    for (var i = 0; i < order.length; i++) {
      var n = Number(counts[order[i]] || 0);
      if (n > 0) parts.push(order[i] + " " + n);
    }
    return parts.join(" · ");
  }

  function copyText(text, btn) {
    var done = function () {
      var old = btn.textContent;
      btn.textContent = "已复制";
      setTimeout(function () { btn.textContent = old; }, 1200);
    };
    try {
      if (root.navigator && root.navigator.clipboard && root.navigator.clipboard.writeText) {
        root.navigator.clipboard.writeText(text).then(done, function () {});
        return;
      }
    } catch (e) { /* 落到下面的兜底 */ }
    try {
      var ta = document.createElement("textarea");
      ta.value = text;
      ta.style.position = "fixed";
      ta.style.opacity = "0";
      document.body.appendChild(ta);
      ta.select();
      document.execCommand("copy");
      document.body.removeChild(ta);
      done();
    } catch (e) { /* 复制失败就让用户手动框选，别再弹一个报错盖住原文 */ }
  }

  function commandBlock(cmd) {
    var wrap = el("div", "lgCmd");
    wrap.appendChild(el("pre", "lgCmdText", cmd));
    var btn = el("button", "mini lgCopy", "复制");
    btn.type = "button";
    btn.addEventListener("click", function () { copyText(cmd, btn); });
    wrap.appendChild(btn);
    return wrap;
  }

  function detailBlock(info) {
    var box = el("div", "lgDetail");
    box.hidden = true;

    var reasons = info.reasons || [];
    if (reasons.length) {
      box.appendChild(el("div", "lgHd", "判定依据"));
      var ul = el("ul", "lgReasons");
      for (var i = 0; i < reasons.length; i++) ul.appendChild(el("li", "", reasons[i]));
      box.appendChild(ul);
    }

    var report = info.report || [];
    if (report.length) {
      box.appendChild(el("div", "lgHd", "迁移报告（" + report.length + " 行）"));
      box.appendChild(el("pre", "lgReport", report.join("\n")));
    }

    if (info.command) {
      box.appendChild(el("div", "lgHd", "转权重：粘到 PowerShell 里跑"));
      box.appendChild(commandBlock(info.command));
    }
    return box;
  }

  /* 画（或清空）参数页顶部的那条提示。返回 true = 画了，false = 新格式 / 没问题 */
  function renderLegacyNotice(host, info) {
    if (!host) return false;
    clear(host);
    if (!info || (!info.detected && !info.needs_convert)) {
      host.hidden = true;
      if (host.style) host.style.display = "none";
      return false;
    }
    host.hidden = false;
    if (host.style) host.style.display = "";

    var counts = info.counts || {};
    var w = info.weights || { torch: 0, tf: 0 };
    var title = info.hint || (info.detected ? "这份 dat 来自 TF 版本，参数已自动迁移" : "");
    if (!title) return false;

    var card = el("div", "alert warn legacyNotice");
    card.setAttribute("data-legacy", info.detected ? "detected" : "weights");
    card.setAttribute("data-needs-convert", info.needs_convert ? "1" : "0");

    var hd = el("div", "alhd");
    hd.appendChild(el("b", "", title));
    var toggle = el("button", "mini lgToggle", "明细");
    toggle.type = "button";
    hd.appendChild(toggle);
    card.appendChild(hd);

    var sub = el("div", "albd lgSub");
    var bits = [];
    var cl = countsLine(counts);
    if (cl) bits.push(cl);
    bits.push("权重 torch " + Number(w.torch || 0) + " · TF " + Number(w.tf || 0));
    sub.textContent = bits.join(" · ");
    card.appendChild(sub);

    if (info.note) card.appendChild(el("div", "lgNote", info.note));

    var detail = detailBlock(info);
    card.appendChild(detail);
    toggle.addEventListener("click", function () {
      detail.hidden = !detail.hidden;
      toggle.textContent = detail.hidden ? "明细" : "收起";
    });

    host.appendChild(card);
    return true;
  }

  function clearLegacyNotice(host) {
    if (!host) return;
    clear(host);
    host.hidden = true;
    if (host.style) host.style.display = "none";
  }

  var API = {
    render: renderLegacyNotice,
    clear: clearLegacyNotice,
    countsLine: countsLine
  };
  root.LegacyNotice = API;
  root.renderLegacyNotice = renderLegacyNotice;   // app.js 里少写一层
  if (typeof module !== "undefined" && module.exports) module.exports = API;
})(typeof window !== "undefined" ? window
   : (typeof globalThis !== "undefined" ? globalThis : this));
