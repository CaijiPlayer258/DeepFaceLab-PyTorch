/* 引擎编译页 —— 自成一体，不依赖 app.js 的内部函数。
 *
 * 为什么单独一个文件：之前把这段逻辑塞进 app.js，被插到了 SSE 回调里（作用域错），
 * 后来又依赖 go() 的时机，反复出问题。这里改成"自己能跑"：
 *   · 页面元素在就渲染，不在就什么都不做（每 3 秒看一眼，成本极低）
 *   · 不依赖路由、不依赖 app.js 的任何变量
 */
(function () {
  "use strict";

  var INTERVAL_IDLE = 3000;
  var INTERVAL_BUILD = 1000;
  var timer = null;
  var interval = INTERVAL_IDLE;
  var lastHtml = "";

  function $(id) { return document.getElementById(id); }

  function toast(a, b) {
    if (typeof window.toast === "function") { window.toast(a, b); return; }
    console.log("[engine]", a, b || "");
  }

  function rowsHtml(models, busyName) {
    return models.map(function (m) {
      var size = m.size_mb ? (m.size_mb + " MB") : "—";
      var esize = m.engine_mb ? (m.engine_mb + " MB") : "—";
      /* 状态列只有两种东西：能编就给按钮，不能编就是一个字 —— 不再重复"未编译" */
      var st;
      if (busyName === m.name) {
        st = '<span class="engspin" title="正在编译"></span>';
      } else if (m.state === "ready") {
        st = '<span class="st-ok">已编译</span>';
      } else if (m.state === "missing" || m.state === "stale") {
        st = '<button class="mini" data-eng="' + m.name + '">编译</button>';
      } else if (m.state === "no-gpu") {
        st = '<span class="st-none">无 GPU</span>';
      } else {
        st = '<span class="st-none">无模型</span>';
      }
      return "<tr>" +
        '<td class="eng-name">' + m.name + "</td>" +
        '<td class="eng-md5">' + (m.md5 || "—") + "</td>" +
        '<td class="eng-num">' + size + "</td>" +
        '<td class="eng-num">' + esize + "</td>" +
        '<td class="eng-st">' + st + "</td>" +
        "</tr>";
    }).join("");
  }

  function render(d) {
    var models = (d && d.models) || [];

    var box = $("engList");
    if (!box) return;                       /* 不在这个页面，跳过渲染 */
    if ($("engSm")) $("engSm").textContent = d.sm ? ("SM " + d.sm) : "无 GPU";

    var job = d.job || {};
    var busyName = job.running ? job.name : "";
    var allBtn = $("engAll");
    if (allBtn) {
      allBtn.disabled = !!job.running;
      allBtn.style.opacity = job.running ? ".55" : "";
      allBtn.style.pointerEvents = job.running ? "none" : "";
    }
    var html = models.length ? rowsHtml(models, busyName) : '<div class="hint">没有找到任何模型</div>';
    if (html !== lastHtml) {
      lastHtml = html;
      box.innerHTML = html;
      Array.prototype.forEach.call(box.querySelectorAll("[data-eng]"), function (b) {
        b.addEventListener("click", function () { build(b.dataset.eng); });
      });
    }

  }

  function load() {
    return fetch("/api/engines").then(function (r) { return r.json(); })
      .then(function (d) {
        watchJob(d);
        render(d);
        var busy = !!(d.job && d.job.running);
        var want = busy ? INTERVAL_BUILD : INTERVAL_IDLE;
        if (want !== interval) { interval = want; start(); }
        return d;
      })
      .catch(function (e) {
        var box = $("engList");
        if (box) box.innerHTML = '<div class="hint">读取引擎状态失败：' + e + "</div>";
      });
  }

  function start() {
    if (timer) clearInterval(timer);
    timer = setInterval(load, interval);
  }

  /* 记住上一次看到的编译任务状态：从 running 变 false 时给个结果提示 */
  var wasRunning = "";
  var lastError = "";

  function watchJob(d) {
    var job = d.job || {};
    if (job.running) { wasRunning = job.name || "?"; }
    else if (wasRunning) {
      lastError = job.error || "";
      if (lastError) { toast("编译失败", wasRunning + " · " + lastError); }
      else { toast("编译完成", wasRunning); }
      wasRunning = "";
    }
  }

  function build(name) {
    fetch("/api/engines/build", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name: name })
    }).then(function (r) { return r.json(); }).then(function (d) {
      if (d && d.ok) { load(); }
      else { toast("编译失败", (d && d.detail) || "未知错误"); }
    }).catch(function () {});
  }

  function buildAll() {
    load().then(function (d) {
      var todo = ((d && d.models) || []).filter(function (m) {
        return m.state === "missing" || m.state === "stale";
      });
      if (!todo.length) { toast("无需编译", "所有可用模型都已有引擎"); return; }
      var i = 0;
      (function next() {
        if (i >= todo.length) { toast("编译完成", "共 " + todo.length + " 个模型"); load(); return; }
        var name = todo[i++].name;
        fetch("/api/engines/build", {
          method: "POST", headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ name: name })
        }).catch(function () {});
        var wait = setInterval(function () {
          fetch("/api/engines").then(function (r) { return r.json(); }).then(function (st) {
            if (!st.job || !st.job.running) { clearInterval(wait); next(); }
          }).catch(function () { clearInterval(wait); });
        }, 1500);
      })();
    });
  }

  /* 绑定按钮（元素是静态的，脚本在页面末尾执行，一定拿得到） */
  function bind() {
    var a = $("engAll");
    if (a && !a.__engBound) { a.__engBound = true; a.addEventListener("click", buildAll); }
  }

  function boot() {
    bind();
    load();
    start();
    /* 静态 HTML 之后可能被重新渲染（切换页面只切 class，不会重建），这里再兜一次 */
    document.addEventListener("click", function (e) {
      var nav = e.target && e.target.closest ? e.target.closest('[data-go="tool-engine"]') : null;
      if (nav) { bind(); load(); }
    }, true);
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", boot);
  } else {
    boot();
  }
})();
