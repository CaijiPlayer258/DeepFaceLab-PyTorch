/* 输出页（#p-output）：把后端 stdout/stderr 回显出来，方便截图上报。

为什么单独一页：用户遇到报错时手上只有截图，而 Electron 那份
_electron_backend.log 每次启动都被 "w" 截断。这一页让用户自己就能把完整日志
截下来 / 复制出去。

增量靠**累计行号**（/api/output/tail 的 total）：环形缓冲淘汰旧行不影响行号，
所以拿最后一条的 n 当下次的 since 就能稳定续读，不需要 diff 内容。
*/
(function () {
  "use strict";

  var OUT = {
    on: false,       // 是否在本页
    timer: null,
    since: 0,        // 已经拿到的最大行号
    rows: [],        // 本地保留的行（跟后端一致，最多 MAX）
    max: 6000,
    failed: 0
  };
  var POLL_MS = 900;

  function $(id) { return document.getElementById(id); }

  function pane() { return $("outPane"); }

  /* 危险/警告行的判定：宁可把「失败」当红，也别让真正的报错混在灰字里 */
  function classify(s) {
    var low = s.toLowerCase();
    if (/traceback|\berror\b|exception|失败|错误|退出码|killed|out of memory|fatal|无法|未能/
        .test(low) || /(^|\s)(E|CRITICAL)\s/.test(s)) return "e";
    if (/\bwarn(ing)?\b|警告|被丢弃|跳过|degrad|fallback|回退|降档/.test(low)) return "w";
    if (s.charAt(0) === "[") return "d";
    return "";
  }

  function esc(s) {
    return String(s).replace(/[&<>"']/g, function (c) {
      return { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c];
    });
  }

  function renderAll() {
    var el = pane();
    if (!el) return;
    var buf = [];
    for (var i = 0; i < OUT.rows.length; i++) {
      var r = OUT.rows[i];
      var cls = classify(r.s);
      buf.push('<span class="ln ' + cls + '"><span class="no">' + r.n + "</span>" + esc(r.s) + "</span>");
    }
    el.innerHTML = buf.join("");
    scrollToEnd();
  }

  function appendRows(rows) {
    var el = pane();
    if (!el) return;
    var buf = [];
    for (var i = 0; i < rows.length; i++) {
      var cls = classify(rows[i].s);
      buf.push('<span class="ln ' + cls + '"><span class="no">' + rows[i].n + "</span>" + esc(rows[i].s) + "</span>");
    }
    el.insertAdjacentHTML("beforeend", buf.join(""));
    // 本地也裁一下，别让 DOM 无限长
    while (el.childNodes.length > OUT.max) el.removeChild(el.firstChild);
    scrollToEnd();
  }

  function scrollToEnd() {
    var el = pane();
    var auto = $("outAuto");
    if (!el || (auto && !auto.checked)) return;
    el.scrollTop = el.scrollHeight;
  }

  function stat(text) {
    var s = $("outStat");
    if (s) s.textContent = text;
  }

  function pull() {
    if (!OUT.on) return;
    fetch("/api/output/tail?since=" + OUT.since + "&limit=2000", { cache: "no-store" })
      .then(function (r) { return r.ok ? r.json() : Promise.reject(new Error("HTTP " + r.status)); })
      .then(function (d) {
        OUT.failed = 0;
        if (d && d.max_lines) OUT.max = d.max_lines;
        var rows = (d && d.lines) || [];
        if (rows.length) {
          OUT.since = rows[rows.length - 1].n;
          OUT.rows = OUT.rows.concat(rows);
          if (OUT.rows.length > OUT.max) OUT.rows = OUT.rows.slice(-OUT.max);
          appendRows(rows);
        }
        var total = (d && d.total) || OUT.since;
        stat("共 " + total + " 行 · 已显示 " + OUT.rows.length +
             " · 文件 " + (d && d.file ? d.file : "?"));
      })
      .catch(function (e) {
        OUT.failed++;
        stat("读取失败（第 " + OUT.failed + " 次）：" + (e && e.message ? e.message : e));
      });
  }

  function start() {
    OUT.on = true;
    if (OUT.timer) clearInterval(OUT.timer);
    stat("正在连接…");
    pull();
    OUT.timer = setInterval(pull, POLL_MS);
  }

  function stop() {
    OUT.on = false;
    if (OUT.timer) { clearInterval(OUT.timer); OUT.timer = null; }
  }

  /* 只清屏幕和本地缓冲，后端文件不动（用户可能还要拿去上报） */
  function clearView() {
    OUT.rows = [];
    OUT.since = 0;
    var el = pane();
    if (el) el.innerHTML = "";
    stat("已清屏（文件仍保留）");
    // 从当前末尾续读：把 since 拉到最新，避免下一轮把历史全灌回来
    fetch("/api/output/tail?since=0&limit=1", { cache: "no-store" })
      .then(function (r) { return r.json(); })
      .then(function (d) { OUT.since = (d && d.total) || 0; })
      .catch(function () {});
  }

  function copyAll() {
    var text = OUT.rows.map(function (r) { return r.n + "  " + r.s; }).join("\n");
    var done = function () { stat("已复制 " + OUT.rows.length + " 行到剪贴板"); };
    try {
      if (navigator.clipboard && navigator.clipboard.writeText) {
        navigator.clipboard.writeText(text).then(done, function () { fallbackCopy(text, done); });
        return;
      }
    } catch (e) {}
    fallbackCopy(text, done);
  }

  function fallbackCopy(text, done) {
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
    } catch (e) { stat("复制失败：" + (e && e.message ? e.message : e)); }
  }

  /* 进/出这一页时开关轮询 —— 不在这一页就不该一直拉日志 */
  window.__outEnter = function () { start(); };
  window.__outLeave = function () { stop(); };

  function bind() {
    var c = $("outClear"), cp = $("outCopy"), w = $("outWrap");
    if (c) c.addEventListener("click", clearView);
    if (cp) cp.addEventListener("click", copyAll);
    if (w) w.addEventListener("change", function () {
      var el = pane();
      if (el) el.classList.toggle("nowrap", !w.checked);
    });
    var a = $("outAuto");
    if (a) a.addEventListener("change", function () { if (a.checked) scrollToEnd(); });
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", bind);
  } else {
    bind();
  }

  /* 暴露给路由：00-core.js 的 go() 里调 */
  window.LCSOutput = { enter: start, leave: stop };

  /* 首屏直接落在 #output（刷新、或直接打开这个地址）时，go() 早就跑完了
     —— 那时 LCSOutput 还不存在，enter() 没被调用过。这里补一次。
     用 rAF 等一帧，让 60-tail.js 里的初始路由先落定。 */
  function boot() {
    try {
      var k = String(location.hash || "").replace(/^#/, "");
      if (k === "output") start();
    } catch (e) {}
  }
  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", function () {
      requestAnimationFrame(boot);
    });
  } else {
    requestAnimationFrame(boot);
  }
})();
