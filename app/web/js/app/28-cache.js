/* 设置页「缓存」卡片：扫描 / 清理。

   注意：**不要**在外面再包 (function(){…})() —— 所有 shard 是同一个大 IIFE 的片段
   （见 00-core.js 开头的目录说明）。自己再包一层的话，confirmBox / toast 这些
   定义在上一层的函数就看不见了，于是 clean() 会退回 window.confirm —— 而 Electron
   里原生 confirm 是被忽略的，表现就是「点清理没反应」。
   标识符一律带 cache 前缀，避免和别的 shard 撞名（$ 用 00-core.js 那个）。 */

var CACHE_LAST = null;

function cacheFmt(b) {
  if (!b) return "0 B";
  if (b < 1024) return b + " B";
  if (b < 1048576) return (b / 1024).toFixed(1) + " KB";
  return (b / 1048576).toFixed(1) + " MB";
}

function cacheRender(d) {
  CACHE_LAST = d;
  var box = $("cacheBox"), tag = $("cacheTag");
  if (!box) return;
  var items = (d && d.items) || [];
  var total = 0;
  var rows = items.map(function (it) {
    total += it.bytes || 0;
    var cnt = it.count ? it.count + " 项" : "无";
    var tip = String((it.sample || []).join("  ·  ")).replace(/"/g, "&quot;");
    var dim = (it.bytes || it.count) ? "" : "color:var(--ink-3);";
    return "<div style=\"display:flex;gap:10px;align-items:baseline;padding:4px 0;" + dim + "\">"
      + "<span style=\"width:120px;flex:none\">" + it.label + "</span>"
      + "<span class=\"mono\" style=\"width:78px;flex:none;text-align:right\">" + cacheFmt(it.bytes || 0) + "</span>"
      + "<span style=\"width:56px;flex:none;color:var(--ink-3)\">" + cnt + "</span>"
      + "<span style=\"flex:1;min-width:0;color:var(--ink-3);overflow:hidden;text-overflow:ellipsis;white-space:nowrap\""
      + " title=\"" + tip + "\">" + (it.note || "") + "</span>"
      + "</div>";
  }).join("");
  box.innerHTML = rows || "没有可清理的缓存";
  if (tag) {
    tag.textContent = total ? cacheFmt(total) : "干净";
    tag.className = "tag " + (total ? "t-yellow" : "t-gray");
  }
}

function cacheScan() {
  var box = $("cacheBox");
  if (box) box.textContent = "正在读取缓存清单…";
  fetch("/api/cache/scan").then(function (r) { return r.json(); })
    .then(cacheRender)
    .catch(function (e) {
      var b = $("cacheBox");
      if (b) b.textContent = "读取失败：" + String((e && e.message) || e);
    });
}

function cacheDoClean() {
  var btn = $("cacheClean");
  if (btn) { btn.disabled = true; btn.textContent = "清理中…"; }
  fetch("/api/cache/clean", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: "{}"
  })
    .then(function (r) { return r.json(); })
    .then(function (d) {
      toast("缓存已清理", "释放 " + cacheFmt(d.bytes || 0)
        + (d.restart_required ? " · 重启一次才生效" : ""), "success");
      cacheScan();
    })
    .catch(function (e) { toast("清理失败", String((e && e.message) || e), "error"); })
    .then(function () {
      var b = $("cacheClean");
      if (b) { b.disabled = false; b.textContent = "清理"; }
    });
}

function cacheCleanClick() {
  var total = CACHE_LAST
    ? (CACHE_LAST.items || []).reduce(function (a, it) { return a + (it.bytes || 0); }, 0)
    : 0;
  var msg = "会删掉 " + cacheFmt(total) + " 的字节码 / 预览缓存 / 合成日志 / 旧版备份 / 流水线临时文件。"
    + " 模型、数据集、成片都不动。清完建议重启一次应用。";
  if (typeof confirmBox === "function") {
    confirmBox("清理缓存", msg, "清理", cacheDoClean);
    return;
  }
  cacheDoClean();          /* 兜底：连确认框都没有就直接清（删的全是可重建的） */
}

(function cacheWire() {
  function bind() {
    var s = $("cacheScan"), c = $("cacheClean");
    if (s) s.addEventListener("click", cacheScan);
    if (c) c.addEventListener("click", cacheCleanClick);
    cacheScan();
  }
  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", bind);
  } else {
    bind();
  }
})();
