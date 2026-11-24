/* 输出页（#p-output）：把后端 stdout/stderr 回显出来，方便截图上报。

为什么单独一页：用户遇到报错时手上只有截图，而 Electron 那份
_electron_backend.log 每次启动都被 "w" 截断。这一页让用户自己就能把完整日志
截下来 / 复制出去。

增量靠**累计行号**（/api/output/tail 的 total）：环形缓冲淘汰旧行不影响行号。

## 交互设计（没有按钮，没有快捷键，全用手势）

  · 点任意一条**错误行**        → 只看错误（再点退出）
  · 双击任意一条**非错误行**    → 按它的 [前缀] 过滤（再双击退出）
  · 点底部「N 条新输出」        → 滚到**绝对底部**（scrollHeight，不是第一条新行）
  · 滚到底（3 行内）            → 恢复吸附，新输出行消失

错误行只响应单击、其他行只响应双击 —— 这样不用为 dblclick 加 250ms 延迟。

## 折叠

  · Traceback 块（有异常消息行）        → <details>，摘要就是 ValueError: ...
  · [vram] 显存规划块（6 行一次）       → <details>，摘要取最后那行的结论
  · 连续完全相同的行                    → <details> 带 ×N

因为块可能跨两次拉取（前一批看着没写完），渲染走**整体重绘**；只在 rows 变化时才跑。
*/
(function () {
  "use strict";

  var OUT = {
    on: false,
    timer: null,
    since: 0,
    max: 6000,
    rows: [],
    seenN: 0,          // 滚到底那一刻的最后行号（未读 = 当前lastN - seenN）
    mode: null         // null | "err" | {prefix:"[vram]"}
  };
  var POLL_MS = 900;
  var STICK_LINES = 3;

  function $(id) { return document.getElementById(id); }
  function pane() { return $("outPane"); }
  function bar() { return $("outBar"); }

  function lineHeight() {
    var el = pane();
    if (!el) return 20;
    try {
      var lh = parseFloat(getComputedStyle(el).lineHeight);
      if (lh > 0) return lh;
    } catch (e) {}
    return 20;
  }

  function nearBottom() {
    var el = pane();
    if (!el) return false;
    return (el.scrollHeight - el.scrollTop - el.clientHeight) <= lineHeight() * STICK_LINES;
  }

  function stick() {
    var el = pane();
    if (!el) return;
    el.scrollTop = el.scrollHeight;
    OUT.seenN = OUT.rows.length ? OUT.rows[OUT.rows.length - 1].n : 0;
  }

  /* ---------- 行分类 ---------- */
  var E_RE = /traceback|\berror\b|exception|失败|错误|退出码|killed|out of memory|fatal|无法|未能/;
  var W_RE = /\bwarn(ing)?\b|警告|被丢弃|跳过|degrad|fallback|回退|降档/;

  function classify(s) {
    var low = s.toLowerCase();
    if (E_RE.test(low)) return "e";
    if (W_RE.test(low)) return "w";
    if (s.charAt(0) === "[") return "d";
    return "";
  }

  function esc(s) {
    return String(s).replace(/[&<>"']/g, function (c) {
      return { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c];
    });
  }

  /* 行首的 [xxx] 标签，双击时用它做过滤维度 */
  function tagOf(s) {
    var m = /^\[([A-Za-z0-9_\-\/]+)\]/.exec(s);
    return m ? m[1] : null;
  }

  /* ---------- 分组 ---------- */
  var TB_RE = /^\s*Traceback \(most recent call last\)/;
  var CARET_RE = /^\s*\^+\s*$/;
  var VRAM_HEAD = /^\[vram\]\s*预览：\s*模型 参数量/;

  function groupRows(rows) {
    var out = [];
    var i = 0;
    while (i < rows.length) {
      var s = rows[i].s;

      /* ① Traceback 块 */
      if (TB_RE.test(s)) {
        var j = i + 1, head = null;
        while (j < rows.length) {
          var t = rows[j].s;
          if (TB_RE.test(t)) break;
          if (t.length && t.charAt(0) !== " " && t.charAt(0) !== "\t" && !CARET_RE.test(t)) {
            head = t; j++; break;
          }
          j++;
        }
        if (!head && j >= rows.length) { out.push({ k: "l", r: rows[i] }); i++; continue; }
        out.push({ k: "tb", head: head || "Traceback", rows: rows.slice(i, j) });
        i = j; continue;
      }

      /* ② [vram] 显存规划块：从「模型 参数量」到最后那条「采用」 */
      if (VRAM_HEAD.test(s)) {
        var j2 = i + 1, last = null;
        while (j2 < rows.length) {
          if (!/^\[vram\]/.test(rows[j2].s)) break;
          last = rows[j2];
          j2++;
          if (/采用/.test(last.s)) break;
        }
        if (j2 - i >= 2) {
          var tail = (last && /采用/.test(last.s)) ? last.s : rows[j2 - 1].s;
          out.push({
            k: "vb",
            head: "显存规划 · " + tail.replace(/^\[vram\]\s*预览：\s*/, ""),
            rows: rows.slice(i, j2)
          });
          i = j2; continue;
        }
        out.push({ k: "l", r: rows[i] }); i++; continue;
      }

      /* ③ 连续完全相同的行 */
      var c = 1;
      while (i + c < rows.length && rows[i + c].s === s && !TB_RE.test(s) && !VRAM_HEAD.test(s)) c++;
      if (c >= 2) {
        out.push({ k: "rep", n: c, s: s, rows: rows.slice(i, i + c) });
        i += c; continue;
      }

      out.push({ k: "l", r: rows[i] });
      i++;
    }
    return out;
  }

  /* ---------- 过滤 ---------- */
  /* 整组保留：组里任意一行命中，整块留下（不然 traceback 只剩一行 ValueError，上下文全丢） */
  function groupHit(g, mode) {
    if (!mode) return true;
    var list = g.k === "l" ? [g.r.s] : g.rows.map(function (r) { return r.s; });
    if (mode === "err") {
      for (var i = 0; i < list.length; i++) if (classify(list[i]) === "e") return true;
      return false;
    }
    if (mode.prefix) {
      var p = mode.prefix;
      for (var k = 0; k < list.length; k++) if (list[k].indexOf(p) === 0) return true;
      return false;
    }
    return true;
  }

  function applyFilter(groups, mode) {
    if (!mode) return groups;
    return groups.filter(function (g) { return groupHit(g, mode); });
  }

  /* ---------- 渲染 ---------- */
  function blockHtml(cls, head, n, rows, tagAttr) {
    var body = rows.map(function (r) { return esc(r.s); }).join("\n");
    return '<details class="' + cls + '"><summary' + (tagAttr || "") + ">" + esc(head) +
      ' <em>' + n + ' 行</em></summary><div class="tbraw">' + body + '</div></details>';
  }

  function renderGroups(groups) {
    var buf = [];
    for (var i = 0; i < groups.length; i++) {
      var g = groups[i];
      if (g.k === "l") {
        var cls = classify(g.r.s);
        var tag = tagOf(g.r.s);
        buf.push('<span class="ln ' + cls + '"' + (tag ? ' data-tag="' + esc(tag) + '"' : "") +
          ">" + esc(g.r.s) + "</span>");
      } else if (g.k === "tb") {
        buf.push(blockHtml("tb", g.head, g.rows.length, g.rows));
      } else if (g.k === "vb") {
        buf.push(blockHtml("vb", g.head, g.rows.length, g.rows));
      } else if (g.k === "rep") {
        buf.push(blockHtml("rep", g.s, g.n, g.rows));
      }
    }
    return buf.join("");
  }

  /* 状态条（只在该模式激活时存在） */
  function renderBar(groups) {
    var el = bar();
    if (!el) return;
    if (!OUT.mode) { el.classList.add("hide"); el.innerHTML = ""; return; }
    var label = OUT.mode === "err"
      ? "只看错误"
      : "只看 " + (OUT.mode.prefix || "");
    var n = groups.filter(function (g) { return groupHit(g, OUT.mode); }).length;
    el.innerHTML = '<span class="bt">' + esc(label) + " · " + n + " 组</span>" +
      '<span class="bx" data-act="all">‹ 查看全部</span>';
    el.classList.remove("hide");
  }

  /* 「N 条新输出」必须渲染在 .outpane **里面** —— position:sticky 相对最近的
     滚动祖先定位，放在外面（兄弟节点）它就退化成普通的 static 块，既不贴底也不滚。 */
  function unreadCount() {
    var lastN = OUT.rows.length ? OUT.rows[OUT.rows.length - 1].n : 0;
    return Math.max(0, lastN - OUT.seenN);
  }

  function newbarHtml() {
    var u = unreadCount();
    return u ? '<div class="newbar" data-act="jump">' + u + " 条新输出  ↓</div>" : "";
  }

  function render() {
    var el = pane();
    if (!el) return;
    var groups = groupRows(OUT.rows);
    renderBar(groups);
    el.innerHTML = renderGroups(applyFilter(groups, OUT.mode)) + newbarHtml();
  }

  /* ---------- 拉取 ---------- */
  function pull() {
    if (!OUT.on) return;
    fetch("/api/output/tail?since=" + OUT.since + "&limit=2000", { cache: "no-store" })
      .then(function (r) { return r.ok ? r.json() : Promise.reject(new Error("HTTP " + r.status)); })
      .then(function (d) {
        if (d && d.max_lines) OUT.max = d.max_lines;
        var rows = (d && d.lines) || [];
        if (!rows.length) return;
        var wasStuck = nearBottom() || OUT.seenN === 0;
        OUT.since = rows[rows.length - 1].n;
        OUT.rows = OUT.rows.concat(rows);
        if (OUT.rows.length > OUT.max) OUT.rows = OUT.rows.slice(-OUT.max);
        render();
        if (wasStuck) stick(); else render();
      })
      .catch(function () {});
  }

  function start() {
    OUT.on = true;
    if (OUT.timer) clearInterval(OUT.timer);
    OUT.since = 0;
    OUT.rows = [];
    OUT.seenN = 0;
    OUT.mode = null;
    pull();
    OUT.timer = setInterval(pull, POLL_MS);
  }

  function stop() {
    OUT.on = false;
    if (OUT.timer) { clearInterval(OUT.timer); OUT.timer = null; }
  }

  /* ---------- 手势 ---------- */
  function bind() {
    var el = pane();
    if (el) {
      el.addEventListener("click", function (ev) {
        var t = ev.target;
        /* 点错误行 → 只看错误 */
        while (t && t !== el && !t.classList) t = t.parentNode;
        if (!t || t === el) return;
        if (t.classList.contains("ln") && t.classList.contains("e")) {
          OUT.mode = (OUT.mode === "err") ? null : "err";
          render(); stick();
        }
      });

      el.addEventListener("dblclick", function (ev) {
        var t = ev.target;
        while (t && t !== el && !t.classList) t = t.parentNode;
        if (!t || t === el) return;
        /* 错误行不参与双击（它已经被单击占用了） */
        if (t.classList.contains("e")) return;
        var tag = t.getAttribute && t.getAttribute("data-tag");
        if (!tag) return;
        var m = "[" + tag + "]";
        OUT.mode = (OUT.mode && OUT.mode.prefix === m) ? null : { prefix: m };
        render(); stick();
      });
    }

    /* 「N 条新输出」每次重绘都是新节点 → 用委托，绑在容器上 */
    if (el) {
      el.addEventListener("click", function (ev) {
        var t = ev.target;
        if (!t || !t.getAttribute || t.getAttribute("data-act") !== "jump") return;
        ev.stopPropagation();
        el.scrollTo({ top: el.scrollHeight, behavior: "smooth" });
        OUT.seenN = OUT.rows.length ? OUT.rows[OUT.rows.length - 1].n : 0;
        setTimeout(render, 420);
      });
    }

    var b = bar();
    if (b) {
      b.addEventListener("click", function () {
        OUT.mode = null;
        render(); stick();
      });
    }

    if (el) {
      el.addEventListener("scroll", function () {
        if (nearBottom()) {
          var lastN = OUT.rows.length ? OUT.rows[OUT.rows.length - 1].n : 0;
          if (OUT.seenN !== lastN) { OUT.seenN = lastN; render(); }
        }
      }, { passive: true });
    }
  }

  /* ---------- 生命周期 ---------- */
  window.LCSOutput = { enter: start, leave: stop };

  function boot() {
    try {
      if (String(location.hash || "").replace(/^#/, "") === "output") start();
    } catch (e) {}
    bind();
  }
  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", function () { requestAnimationFrame(boot); });
  } else {
    requestAnimationFrame(boot);
  }
})();
