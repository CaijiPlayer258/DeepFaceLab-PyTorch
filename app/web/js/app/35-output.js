/* 输出页（#p-output）：把后端 stdout/stderr 回显出来，方便截图上报。

为什么单独一页：用户遇到报错时手上只有截图，而 Electron 那份
_electron_backend.log 每次启动都被 "w" 截断。这一页让用户自己就能把完整日志
截下来 / 复制出去。

增量靠**累计行号**（/api/output/tail 的 total）：环形缓冲淘汰旧行不影响行号，
所以拿最后一条的 n 当下的 since 就能稳定续读，不需要 diff 内容。

**吸附规则**（用户要求）：新行永远追加在最下面；只有当视口已经贴到倒数 3 行以内
时才自动跟到底 —— 用户往上翻看旧日志时，新行进来不会把他弹回底部。
*/
(function () {
  "use strict";

  var OUT = {
    on: false,
    timer: null,
    since: 0,
    max: 6000
  };
  var POLL_MS = 900;
  var STICK_LINES = 3;      // 距底部 3 行以内才开始吸附

  function $(id) { return document.getElementById(id); }
  function pane() { return $("outPane"); }

  /* 行高（px）：CSS 里 outpane 是 font-size 12.5 / line-height 1.55 */
  function lineHeight() {
    var el = pane();
    if (!el) return 20;
    try {
      var lh = parseFloat(getComputedStyle(el).lineHeight);
      if (lh > 0) return lh;
    } catch (e) {}
    return 20;
  }

  /* 视口是不是已经贴在底部（倒数 STICK_LINES 行以内） */
  function nearBottom() {
    var el = pane();
    if (!el) return false;
    var gap = el.scrollHeight - el.scrollTop - el.clientHeight;
    return gap <= lineHeight() * STICK_LINES;
  }

  function stick() {
    var el = pane();
    if (!el) return;
    el.scrollTop = el.scrollHeight;
  }

  /* 危险/警告行的判定：宁可把「失败」当红，也别让真正的报错混在灰字里 */
  function classify(s) {
    var low = s.toLowerCase();
    if (/traceback|\berror\b|exception|失败|错误|退出码|killed|out of memory|fatal|无法|未能/.test(low)) return "e";
    if (/\bwarn(ing)?\b|警告|被丢弃|跳过|degrad|fallback|回退|降档/.test(low)) return "w";
    if (s.charAt(0) === "[") return "d";
    return "";
  }

  function esc(s) {
    return String(s).replace(/[&<>"']/g, function (c) {
      return { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c];
    });
  }

  /* 把 Traceback 块折起来 —— 一份 Python traceback 动辄二三十行，全展开的话
     用户想往上翻真正有用的那几行得滚半天。识别规则：
       · 起点：整行就是 «Traceback (most recent call last):»
       · 终点：第一条**顶格**、且不是 ^^^ 标记、也不是下一个 Traceback 的行
         —— 那一行就是异常消息（ValueError: ...），拿它当摘要
       · 走到末尾还没找到消息行 = 块还没写完（流式），这一批先不折
     折叠用 <details>：原生、可键盘操作、展开状态由浏览器记着。 */
  var TB_RE = /^\s*Traceback \(most recent call last\)/;
  var CARET_RE = /^\s*\^+\s*$/;

  function group(rows) {
    var out = [];
    var i = 0;
    while (i < rows.length) {
      if (!TB_RE.test(rows[i].s)) { out.push({ k: "l", r: rows[i] }); i++; continue; }
      var j = i + 1, head = null;
      while (j < rows.length) {
        var t = rows[j].s;
        if (TB_RE.test(t)) break;                       // 下一个块开始，当前块没消息
        if (t.length && t.charAt(0) !== " " && t.charAt(0) !== "\t" && !CARET_RE.test(t)) {
          head = t; j++; break;                           // 顶格非 ^^^ = 异常消息
        }
        j++;
      }
      if (!head && j >= rows.length) {                    // 还没写完，先当普通行
        out.push({ k: "l", r: rows[i] }); i++; continue;
      }
      out.push({ k: "tb", head: head || "Traceback", rows: rows.slice(i, j) });
      i = j;
    }
    return out;
  }

  function html(rows) {
    var groups = group(rows);
    var buf = [];
    for (var i = 0; i < groups.length; i++) {
      var g = groups[i];
      if (g.k === "l") {
        buf.push('<span class="ln ' + classify(g.r.s) + '">' + esc(g.r.s) + "</span>");
        continue;
      }
      var n = g.rows.length;
      var body = g.rows.map(function (r) { return esc(r.s); }).join("\n");
      buf.push('<details class="tb"><summary>' + esc(g.head) +
        ' <em>' + n + ' 行</em></summary><div class="tbraw">' + body + "</div></details>");
    }
    return buf.join("");
  }

  /* 增量 append 不行了：一个 traceback 块可能跨两次拉取，前一批看到的是「未闭合」
     （按普通行摆着），后一批才补齐消息行 —— 那时得把前面那些行收进 <details>。
     所以有新增就整体重渲染。6000 行以内这个代价可以接受（实测 3~6ms），
     而且只在真的有新行时才跑，日志静止时一次都不渲染。 */
  function append(rows) {
    var el = pane();
    if (!el) return;
    var wasStuck = nearBottom();      // **先**判断，写进去之后 scrollHeight 就变了
    el.innerHTML = html(OUT.rows);
    if (wasStuck) stick();
  }

  function replaceAll(rows) {
    var el = pane();
    if (!el) return;
    el.innerHTML = html(rows);
    stick();
  }

  function pull() {
    if (!OUT.on) return;
    fetch("/api/output/tail?since=" + OUT.since + "&limit=2000", { cache: "no-store" })
      .then(function (r) { return r.ok ? r.json() : Promise.reject(new Error("HTTP " + r.status)); })
      .then(function (d) {
        if (d && d.max_lines) OUT.max = d.max_lines;
        var rows = (d && d.lines) || [];
        if (!rows.length) return;
        var first = !OUT.since;       // 第一次拉 = 装满屏，直接跳底
        OUT.since = rows[rows.length - 1].n;
        if (first) replaceAll(rows);
        else append(rows);
      })
      .catch(function () {});
  }

  function start() {
    OUT.on = true;
    if (OUT.timer) clearInterval(OUT.timer);
    pull();
    OUT.timer = setInterval(pull, POLL_MS);
  }

  function stop() {
    OUT.on = false;
    if (OUT.timer) { clearInterval(OUT.timer); OUT.timer = null; }
  }

  window.LCSOutput = { enter: start, leave: stop };

  /* 首屏直接落在 #output（刷新、或直接打开这个地址）时，go() 早就跑完了
     —— 那时 LCSOutput 还不存在，enter() 没被调用过。这里补一次。 */
  function boot() {
    try {
      if (String(location.hash || "").replace(/^#/, "") === "output") start();
    } catch (e) {}
  }
  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", function () { requestAnimationFrame(boot); });
  } else {
    requestAnimationFrame(boot);
  }
})();
