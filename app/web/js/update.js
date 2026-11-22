/* 自动更新：检查 GitHub Release，只拉代码包。
   只有一个按钮：没更新时是「检查更新」，发现新版本后同一个按钮变成「下载并安装」。 */
(function () {
  "use strict";
  var btn = document.getElementById("updCheck");
  var state = document.getElementById("updState");
  var notes = document.getElementById("updNotes");
  var local = document.getElementById("updLocal");
  if (!btn) return;

  /* release notes 是 markdown，以前直接塞进 <pre> 里显示：##、-、` 全露出来，
     字体也是等宽，和整个界面不搭。这里做一个够用的子集渲染。 */
  function esc(t) {
    return t.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
  }
  function inline(t) {
    return esc(t)
      .replace(/`([^`]+)`/g, "<code>$1</code>")
      .replace(/\*\*([^*]+)\*\*/g, "<b>$1</b>");
  }
  function mdToHtml(md) {
    var out = [], list = null;
    function closeList() { if (list) { out.push("</ul>"); list = null; } }
    md.split(/\r?\n/).forEach(function (line) {
      var m;
      if (/^\s*$/.test(line)) { closeList(); return; }
      if ((m = line.match(/^#{1,3}\s+(.*)$/))) { closeList(); out.push("<h4>" + inline(m[1]) + "</h4>"); return; }
      if ((m = line.match(/^\s*[-*]\s+(.*)$/))) {
        if (!list) { out.push("<ul>"); list = 1; }
        out.push("<li>" + inline(m[1]) + "</li>");
        return;
      }
      if ((m = line.match(/^\s*\d+\.\s+(.*)$/))) {
        if (!list) { out.push("<ul>"); list = 1; }
        out.push("<li>" + inline(m[1]) + "</li>");
        return;
      }
      closeList();
      out.push("<p>" + inline(line) + "</p>");
    });
    closeList();
    return out.join("");
  }
  var INFO = null;
  var MODE = "check";        // check | install

  function setState(t) { if (state) state.textContent = t; }
  function setBtn(t) { btn.textContent = t; }

  fetch("/api/update/version").then(function (r) { return r.json(); })
    .then(function (j) { if (local) local.textContent = "当前 " + (j.human || ("v" + j.version)); })
    .catch(function () {});

  btn.addEventListener("click", function () {
    if (MODE === "install") { doInstall(); return; }
    if (MODE === "restart") { doRestart(); return; }
    doCheck();
  });

  function doCheck() {
    btn.disabled = true;
    setState("正在检查…");
    fetch("/api/update/check").then(function (r) { return r.json(); }).then(function (j) {
      btn.disabled = false;
      INFO = j;
      if (!j.ok) { setState("检查失败：" + (j.error || "网络不可用")); return; }
      if (notes) {
        notes.style.display = "block";
        notes.innerHTML = mdToHtml((j.notes || "").slice(0, 4000)) || '<p class="upd-empty">(该版本没有说明)</p>';
      }
      if (j.behind) {
        MODE = "install";
        setBtn("下载并安装");
        setState("发现新版本 " + (j.remote || j.tag || ""));
      } else {
        MODE = "check";
        setBtn("检查更新");
        setState("已是最新（" + j.local + "）");
      }
    }).catch(function (e) {
      btn.disabled = false;
      setState("检查失败：" + e);
    });
  }

  var POLL = 0;

  function stopPoll() { if (POLL) { clearInterval(POLL); POLL = 0; } }

  function paintProgress() {
    fetch("/api/update/progress").then(function (r) { return r.json(); }).then(function (p) {
      if (!p || !p.running) return;
      var pct = Math.max(0, Math.min(100, p.pct || 0));
      var bar = document.getElementById("updProg");
      if (bar) { bar.classList.remove("hide"); bar.firstElementChild.style.width = pct + "%"; }
      var t = p.msg || "处理中…";
      if (p.total) t += "  " + (p.done / 1048576).toFixed(1) + " / " + (p.total / 1048576).toFixed(1) + " MB";
      setState(t);
    }).catch(function () {});
  }
  function doInstall() {
    if (!INFO || !INFO.behind) return;
    btn.disabled = true;
    setState("正在下载并安装…");
    stopPoll();
    POLL = setInterval(paintProgress, 400);
    fetch("/api/update/apply", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ asset: INFO.asset ? INFO.asset.name : null })
    }).then(function (r) { return r.json(); }).then(function (j) {
      stopPoll();
      var bar0 = document.getElementById("updProg");
      if (bar0) { bar0.firstElementChild.style.width = "100%"; }
      btn.disabled = false;
      if (!j.ok) { setState("安装失败：" + (j.error || "")); return; }
      if (!j.installed) { setState(j.message || "已是最新"); return; }
      /* 装完就别再让人点「检查更新」了：这时唯一有意义的动作是重启 */
      MODE = "restart";
      setBtn("立刻重启");
      setState("已安装 " + j.files + " 个文件，重启后生效" + (j.backup ? "（备份：" + j.backup + "）" : ""));
    }).catch(function (e) {
      stopPoll();
      btn.disabled = false;
      setState("安装失败：" + e);
    });
  }
})();
