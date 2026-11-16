/* ============================================================================
 * 权重分包门控 —— 工具 → 遮罩绘制
 *   weights/ 按功能分包：core 必需（缺了这页没法用），sam 可选（约 3.4 GB）。
 *   后端接口（并行开发中，没上线时会 404 —— 一律按"未知"处理，不报错、不遮挡）：
 *     GET  /api/weights/status        → { ok, packages:{ core:{installed}, sam:{installed,size_mb} } }
 *     POST /api/weights/fetch         { group:"sam" } → { ok, job }
 *     GET  /api/weights/fetch/status  → { ok, running, pct, line, error }
 *   遮罩绘制那三块控件（SAM / GROUNDED SAM2 / BISENET）长在 iframe
 *   /mask-editor/index.html 里，是独立文档 —— app.css 打不进去，
 *   所以这里同源直连 contentDocument：置灰走 disabled + 半透明，
 *   提示条也插在它自己的 .right-panel 里，配色全部借编辑器已有的 token
 *   （--m-line / --m-panel-2 / --m-ink-2，按钮复用 .mkbtn，红沿用 #ef4444）。
 * ============================================================================ */
/* ==== WKG-START ==== 这段可以单独切出来喂假 DOM 跑状态机，别在里面引用外部变量 */
(function () {
  "use strict";

  var SIZE_MB = 3455;   /* 后端没给 size_mb 时的兜底值（= 3.4 GB） */

  /* ------------------------------------------------------------ 纯状态机 */

  /** /api/weights/status 的响应 → {core,sam,samMB}；形状不对返回 null（= 未知）。 */
  function wgParse(d) {
    if (!d || typeof d !== "object") return null;
    var pk = d.packages;
    if (!pk || typeof pk !== "object") return null;
    var core = pk.core, sam = pk.sam;
    if (!core || typeof core.installed !== "boolean") return null;
    var mb = (sam && typeof sam.size_mb === "number" && sam.size_mb > 0) ? sam.size_mb : SIZE_MB;
    return { core: core.installed === true, sam: !!(sam && sam.installed === true), samMB: mb };
  }

  /** 3455 → "3.4 GB" */
  function wgSize(mb) {
    var g = (Number(mb) || SIZE_MB) / 1024;
    return g.toFixed(1) + " GB";
  }

  /** 按钮文案：只有下载中带百分比，失败/中断回到「重试」。 */
  function wgLabel(phase, pct) {
    if (phase === "downloading") return "下载中 " + pct + "%";
    if (phase === "failed") return "重试";
    return "下载";
  }

  /**
   * 状态机：status（wgParse 的结果）+ 运行时（{phase,pct,err}）→ 视图
   *   unknown     接口没就绪 / 读失败 → 什么都不做，维持现状
   *   nocore      core 没装（理论上不该）→ 顶部红条 + 「打开数据集」之外主要操作置灰
   *   missing     sam 没装 → 三块置灰 + 上方提示条 + 「下载」
   *   downloading 正在下 → 三块继续置灰 + 按钮「下载中 42%」
   *   failed      失败/中断 → 三块继续置灰 + 按钮「重试」+ 一行错误
   *   ready       sam 装好 → 解锁，提示条撤掉
   */
  function wgView(st, rt) {
    rt = rt || {};
    var phase = "unknown";
    if (st) {
      if (!st.core) phase = "nocore";
      else if (st.sam) phase = "ready";
      else if (rt.phase === "downloading") phase = "downloading";
      else if (rt.phase === "failed") phase = "failed";
      else phase = "missing";
    }
    var pct = Math.round(Number(rt.pct) || 0);
    if (pct < 0) pct = 0;
    if (pct > 100) pct = 100;
    var gate = (phase === "missing" || phase === "downloading" || phase === "failed");
    return {
      phase: phase,
      gate: gate,
      label: wgLabel(phase, pct),
      pct: pct,
      size: wgSize(st ? st.samMB : SIZE_MB),
      err: phase === "failed" ? String(rt.err || "下载中断，点「重试」再来一次") : "",
      core: phase === "nocore"
    };
  }

  /* -------------------------------------------------------- 画到 iframe 里 */

  function wgFrame() { return document.getElementById("mkFrame"); }

  /** 同源 iframe 的文档；拿不到（还没 load / 跨域）返回 null。 */
  function wgDoc() {
    var f = wgFrame();
    if (!f) return null;
    try {
      var d = f.contentDocument || (f.contentWindow && f.contentWindow.document);
      return (d && typeof d.createElement === "function") ? d : null;
    } catch (e) { return null; }
  }

  /** 按标题文字找右栏那三块（"SAM" / "GROUNDED SAM2" / "BISENET"）。 */
  function wgSection(doc, title) {
    var secs = doc.querySelectorAll(".right-panel .panel-section") || [];
    for (var i = 0; i < secs.length; i++) {
      var t = secs[i].querySelector(".panel-section__title");
      if (t && String(t.textContent || "").trim().toUpperCase() === title) return secs[i];
    }
    return null;
  }

  /**
   * 置灰 / 还原：disabled 挡住点击，opacity + pointer-events 让它看着也用不了。
   * key 是"谁上的锁"：core 一层、sam 一层，各记各的基线，
   * 否则整页锁还原时会把三块刚上的锁一起解掉（右栏锁包含那三块）。
   */
  function wgLockBox(el, on, key) {
    if (!el) return;
    var k = key || "data-wkgwas";
    var ctl = el.querySelectorAll("button, input, select, textarea") || [];
    for (var i = 0; i < ctl.length; i++) {
      var c = ctl[i];
      if (on) {
        if (c.getAttribute(k) === null) c.setAttribute(k, c.disabled ? "1" : "0");
        c.disabled = true;
      } else if (c.getAttribute(k) !== null) {
        c.disabled = c.getAttribute(k) === "1";
        c.removeAttribute(k);
      }
    }
    if (el.style) {
      el.style.opacity = on ? "0.45" : "";
      el.style.pointerEvents = on ? "none" : "";
    }
  }

  /** 三块上方那条提示条（懒建；建好之后每帧只改文字）。 */
  function wgBox(doc) {
    var bar = doc.getElementById("wkgSamBar");
    if (bar) return bar;
    bar = doc.createElement("div");
    bar.id = "wkgSamBar";
    bar.style.cssText =
      "display:flex;align-items:center;gap:6px;flex-wrap:wrap;margin:0 0 8px;" +
      "padding:8px 10px;border:1px solid var(--m-line);border-radius:9px;" +
      "background:var(--m-panel-2);color:var(--m-ink-2);font-size:10.5px;line-height:1.45";

    var txt = doc.createElement("div");
    txt.id = "wkgSamTxt";
    txt.style.cssText = "flex:1 1 auto;min-width:0";
    txt.textContent = "遮罩绘制需要额外模型（约 3.4 GB）";
    bar.appendChild(txt);

    var btn = doc.createElement("button");
    btn.id = "wkgSamBtn";
    btn.type = "button";
    btn.className = "mkbtn";                       /* 复用编辑器里「打开」那颗按钮 */
    btn.style.cssText = "height:24px;padding:0 10px;font-size:10.5px";
    btn.textContent = "下载";
    btn.onclick = function () { wgStart(); };
    bar.appendChild(btn);

    var err = doc.createElement("div");
    err.id = "wkgSamErr";
    err.style.cssText = "display:none;flex:1 1 100%;font-size:10px;color:#ef4444";
    bar.appendChild(err);
    return bar;
  }

  /** 插到第一块（SAM）前面；找不到就追加到右栏末尾。 */
  function wgPut(doc, bar, before) {
    var host = before && before.parentNode;
    if (!host) {
      host = doc.querySelector(".right-panel") || doc.body;
      before = null;
    }
    if (!host || bar.parentNode === host) return;
    host.insertBefore(bar, before || null);
  }

  /** core 缺包时页面顶部的红条。 */
  function wgCoreBar(doc) {
    var cb = doc.getElementById("wkgCoreBar");
    if (cb) return cb;
    cb = doc.createElement("div");
    cb.id = "wkgCoreBar";
    cb.style.cssText =
      "display:flex;align-items:center;gap:8px;padding:9px 14px;" +
      "font-size:11.5px;line-height:1.4;color:#ef4444;" +
      "background:rgba(239,68,68,0.10);border-bottom:1px solid rgba(239,68,68,0.28)";
    cb.textContent = "缺少基础模型，请先运行 modelhub.fetch --group core";
    var shell = doc.querySelector(".app-shell") || doc.body;
    if (shell) shell.insertBefore(cb, shell.firstChild || null);
    return cb;
  }

  /**
   * 把视图贴到编辑器文档上。doc 可以是真 Document，也可以是假 DOM。
   * 找不到那三块（编辑器改版）就整体不动，别把别人的布局搅了。
   */
  function wgPaintDoc(doc, view) {
    if (!doc || !view) return false;

    /* core 缺包管的是整页，必须先落锁/解锁 —— 放到三块后面会被它们的还原盖掉 */
    var cb = doc.getElementById("wkgCoreBar");
    if (view.core) {
      wgCoreBar(doc);
      wgLockBox(doc.querySelector(".toolbar"), true, "data-wkgcore");
      wgLockBox(doc.querySelector(".mode-bar"), true, "data-wkgcore");
      wgLockBox(doc.querySelector(".right-panel"), true, "data-wkgcore");
    } else if (cb && cb.parentNode) {
      cb.parentNode.removeChild(cb);
      wgLockBox(doc.querySelector(".toolbar"), false, "data-wkgcore");
      wgLockBox(doc.querySelector(".mode-bar"), false, "data-wkgcore");
      wgLockBox(doc.querySelector(".right-panel"), false, "data-wkgcore");
    }

    var titles = ["SAM", "GROUNDED SAM2", "BISENET"];
    var lockSecs = view.core || view.gate;   /* core 缺时也锁着，别让两层互相解 */
    var first = null, i, sec;
    for (i = 0; i < titles.length; i++) {
      sec = wgSection(doc, titles[i]);
      if (sec && !first) first = sec;
      wgLockBox(sec, lockSecs, "data-wkgsec");
    }

    var bar = doc.getElementById("wkgSamBar");
    if (view.gate && first) {
      bar = wgBox(doc);
      var txt = doc.getElementById("wkgSamTxt");
      if (txt) txt.textContent = "遮罩绘制需要额外模型（约 " + view.size + "）";
      var btn = doc.getElementById("wkgSamBtn");
      if (btn) {
        btn.textContent = view.label;                        /* 下载 / 下载中 42% / 重试 */
        btn.disabled = (view.phase === "downloading");       /* 下载中不许连点 */
      }
      var err = doc.getElementById("wkgSamErr");
      if (err) {
        err.textContent = view.err || "";
        err.style.display = view.err ? "block" : "none";
      }
      wgPut(doc, bar, first);
    } else if (bar && bar.parentNode) {
      bar.parentNode.removeChild(bar);
    }
    return true;
  }

  /* ---------------------------------------------------------------- 运行时 */

  var RT = { status: null, phase: "idle", pct: 0, err: "", timer: 0 };

  function wgPaint() {
    var doc = wgDoc();
    if (!doc) return false;
    try { return wgPaintDoc(doc, wgView(RT.status, RT)); } catch (e) { return false; }
  }

  function wgStop() {
    if (RT.timer) { clearInterval(RT.timer); RT.timer = 0; }
  }

  /** 失败/中断收口：轮询停掉、按钮回到「重试」、提示条给一行错误。 */
  function wgFail(msg) {
    wgStop();
    RT.phase = "failed";
    RT.err = String(msg || "下载中断");
    wgPaint();
  }

  /** GET status —— 读不到（404 / 网络失败）就当"未知"，维持现状。 */
  function wgLoad() {
    return fetch("/api/weights/status")
      .then(function (r) { return r.ok ? r.json() : null; })
      .then(function (d) {
        RT.status = wgParse(d);
        wgPaint();
        return RT.status;
      })
      .catch(function () { RT.status = null; wgPaint(); return null; });
  }

  /** 跑完/停了之后重新读 status，控件按真实安装状态恢复（不用刷新页面）。 */
  function wgDone(d) {
    return fetch("/api/weights/status")
      .then(function (r) { return r.ok ? r.json() : null; })
      .then(function (j) {
        RT.status = wgParse(j);
        if (RT.status && RT.status.sam) {
          RT.phase = "idle"; RT.err = ""; RT.pct = 100;
          wgPaint();
          return;
        }
        wgFail((d && (d.error || d.detail || d.line)) || "下载没有完成，已中断");
      })
      .catch(function () { wgFail("下载停了但读不到安装状态，点「重试」再确认一次"); });
  }

  /** 1000ms 一跳：running 就一直报进度，停了就收口。 */
  function wgTick() {
    return fetch("/api/weights/fetch/status")
      .then(function (r) { return r.ok ? r.json() : null; })
      .then(function (d) {
        if (!d || typeof d !== "object") throw new Error("下载状态接口没有响应");
        RT.pct = Math.round(Number(d.pct) || 0);
        if (d.running) { wgPaint(); return; }
        wgStop();
        if (d.error) { wgFail(d.error); return; }
        return wgDone(d);
      })
      .catch(function (e) { wgFail((e && e.message) || "下载中断"); });
  }

  /** 点「下载」：POST 起任务 → 1s 轮询，pct 直接写到按钮文字上。 */
  function wgStart() {
    if (RT.phase === "downloading") return;          /* 已经在下，别再起一个 */
    RT.phase = "downloading";
    RT.pct = 0;
    RT.err = "";
    wgPaint();                                        /* 立刻变「下载中 0%」并置灰 */
    fetch("/api/weights/fetch", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ group: "sam" })
    }).then(function (r) {
      if (!r.ok) throw new Error(r.status === 404 ? "下载接口还没就绪，稍后再试" : ("下载接口返回 " + r.status));
      return r.json();
    }).then(function (d) {
      if (d && d.ok === false) throw new Error(String(d.error || d.detail || "下载任务没能启动"));
      wgStop();
      RT.timer = setInterval(wgTick, 1000);
    }).catch(function (e) { wgFail((e && e.message) || "下载启动失败"); });
  }

  /** 进入「工具 → 遮罩绘制」时调用。 */
  function wgEnter() {
    var f = wgFrame();
    if (f && !f.dataset.wkgBound) {
      f.dataset.wkgBound = "1";
      /* 换目录 / 首次加载都会重设 src，画好的状态要重新贴一次 */
      f.addEventListener("load", function () { wgPaint(); });
    }
    wgPaint();          /* 先把上次读到的状态贴上去，切页回来不闪 */
    return wgLoad();    /* 再按后端最新状态刷一遍 */
  }

  /* 刷新 / 深链接直接落在这一页时：主 IIFE 的 go() 比这段先跑，那时 window.WKG 还不存在，
     所以这里补一次（有 hash 就当作"进页"）。 */
  try {
    document.addEventListener("DOMContentLoaded", function () {
      var h = "";
      try { h = String((window.location && window.location.hash) || "").replace(/^#/, ""); } catch (e2) {}
      if (h === "tool-mask") wgEnter();
    });
  } catch (e3) {}

  window.WKG = { enter: wgEnter, refresh: wgLoad, start: wgStart, paint: wgPaint };
  /* 单测入口：纯函数 + 画布函数都能直接喂假文档 */
  window.WKG._test = { parse: wgParse, view: wgView, label: wgLabel, size: wgSize, paint: wgPaintDoc };
})();
/* ==== WKG-END ==== */
