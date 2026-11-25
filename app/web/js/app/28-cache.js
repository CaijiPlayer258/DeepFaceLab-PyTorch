(function () {
  "use strict";

  var $ = function (id) { return document.getElementById(id); };
  function fmt(b) {
    if (!b) return "0 B";
    if (b < 1024) return b + " B";
    if (b < 1048576) return (b / 1024).toFixed(1) + " KB";
    return (b / 1048576).toFixed(1) + " MB";
  }

  var LAST = null;

  function render(d) {
    LAST = d;
    var box = $("cacheBox"), tag = $("cacheTag");
    if (!box) return;
    var items = (d && d.items) || [];
    var total = 0;
    var rows = items.map(function (it) {
      total += it.bytes || 0;
      var cls = (it.bytes || it.count) ? "" : " style=\"color:var(--ink-3)\"";
      var cnt = it.count ? it.count + " 项" : "无";
      var tip = (it.sample || []).join("  ·  ");
      return "<div style=\"display:flex;gap:10px;align-items:baseline;padding:4px 0\"" + cls + ">"
        + "<span style=\"width:120px;flex:none\">" + it.label + "</span>"
        + "<span class=\"mono\" style=\"width:78px;flex:none;text-align:right\">" + fmt(it.bytes || 0) + "</span>"
        + "<span style=\"width:56px;flex:none;color:var(--ink-3)\">" + cnt + "</span>"
        + "<span style=\"flex:1;min-width:0;color:var(--ink-3);overflow:hidden;text-overflow:ellipsis;white-space:nowrap\""
        + " title=\"" + tip.replace(/"/g, "&quot;") + "\">" + (it.note || "") + "</span>"
        + "</div>";
    }).join("");
    box.innerHTML = rows || "没有可清理的缓存";
    if (tag) {
      tag.textContent = total ? fmt(total) : "干净";
      tag.className = "tag " + (total ? "t-warn" : "t-gray");
    }
  }

  function scan() {
    var box = $("cacheBox");
    if (box) box.textContent = "正在读取缓存清单…";
    fetch("/api/cache/scan").then(function (r) { return r.json(); })
      .then(render)
      .catch(function (e) {
        if (box) box.textContent = "读取失败：" + String((e && e.message) || e);
      });
  }

  function clean() {
    var total = LAST ? (LAST.items || []).reduce(function (a, it) { return a + (it.bytes || 0); }, 0) : 0;
    /* confirmBox 是 5 参数 (title, desc, okText, onOk, danger) —— 少传 okText 会把
       回调函数当成按钮文案印出来（之前在别的页面踩过一次）。 */
    confirmBox("清理缓存",
      "会删掉：" + fmt(total) + " 的字节码 / 预览缓存 / 合成日志 / 旧版备份 / 流水线临时文件。"
      + " 模型、数据集、成片都不动。清完建议重启一次应用。",
      "清理",
      function () {
        var btn = $("cacheClean");
        if (btn) { btn.disabled = true; btn.textContent = "清理中…"; }
        fetch("/api/cache/clean", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: "{}"
        }).then(function (r) { return r.json(); })
          .then(function (d) {
            toast("缓存已清理", "释放 " + fmt(d.bytes || 0)
              + (d.restart_required ? " · 重启一次才生效" : ""), "success");
            scan();
          })
          .catch(function (e) { toast("清理失败", String((e && e.message) || e), "error"); })
          .then(function () {
            var b = $("cacheClean");
            if (b) { b.disabled = false; b.textContent = "清理"; }
          });
      });
  }

  function wire() {
    var s = $("cacheScan"), c = $("cacheClean");
    if (s) s.addEventListener("click", scan);
    if (c) c.addEventListener("click", clean);
    scan();
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", wire);
  } else {
    wire();
  }
})();
