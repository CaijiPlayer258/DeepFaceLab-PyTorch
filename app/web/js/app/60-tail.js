/* ============================================================================
 * 设置 → 权重包 —— 「缺哪个包、缺几个、点一下能下」
 *   GET  /api/weights/status        → { packages:{ core:{installed,missing,missing_mb,…} } }
 *   POST /api/weights/fetch         { group, url_base } → { ok, job }
 *   GET  /api/weights/fetch/status  → { running, pct, line, error }
 *   在这块 UI 之前，缺 errface_xgb.ubj 只能靠命令行 modelhub.fetch 解决，
 *   界面上说「缺模型」却没地方点 —— 至少要有地方能看、能下、能知道下到哪一步。
 * ============================================================================ */
(function () {
  "use strict";

  var TIMER = 0, BUSY = "";

  function $(id) { return document.getElementById(id); }

  /* 后端来的动态句子也要进词典（切英文不能留中文） */
  function addDict(zh, en) {
    try {
      var d = window.I18N && window.I18N.dict;
      if (d && zh && d[zh] === undefined) d[zh] = en || zh;
    } catch (e) {}
  }
  function say(zh, en) { addDict(zh, en); return zh; }

  function size(mb) {
    var n = Number(mb) || 0;
    if (!n) return "";
    return n >= 1024 ? (n / 1024).toFixed(1) + " GB" : n.toFixed(1) + " MB";
  }

  function stateText(p) {
    if (!p) return say("未检测", "unknown");
    if (p.installed) return say("已装", "installed");
    var miss = Number(p.missing) || 0;
    var sz = size(p.missing_mb);
    return say("缺 " + miss + " 个文件" + (sz ? "（" + sz + "）" : ""),
               miss + " files missing" + (sz ? " (" + sz + ")" : ""));
  }

  function row(key, pkg) {
    var el = document.createElement("div");
    el.style.cssText = "display:flex;align-items:center;gap:10px;padding:5px 0;font-size:12px";

    var name = document.createElement("span");
    name.style.cssText = "width:64px;flex:0 0 64px;color:var(--ink-2);font-weight:600";
    name.textContent = key;                     /* core / sam 是包名，不翻 */
    el.appendChild(name);

    var st = document.createElement("span");
    st.style.cssText = "flex:1 1 auto;min-width:0;color:var(--ink-3);font-size:11.5px";
    st.textContent = stateText(pkg);
    st.title = st.textContent;
    el.appendChild(st);

    if (!pkg || !pkg.installed) {
      var b = document.createElement("button");
      b.className = "mini";
      b.type = "button";
      b.textContent = "下载";
      b.disabled = !!BUSY;
      b.addEventListener("click", function () { fetchGroup(key, b); });
      el.appendChild(b);
    }
    return el;
  }

  function paint(d) {
    var box = $("pkgBox"), tag = $("pkgTag");
    if (!box) return;
    box.innerHTML = "";
    box.className = "";
    if (!d || !d.ok) {
      box.className = "hint";
      box.textContent = say("读不到权重清单：", "Cannot read the weight list: ") +
                        String((d && (d.error || d.detail)) || "weights/manifest.json");
      if (tag) { tag.textContent = say("未知", "unknown"); tag.className = "tag t-gray"; }
      return;
    }
    var pk = d.packages || {};
    var keys = Object.keys(pk);
    if (!keys.length) {
      box.className = "hint";
      box.textContent = say("清单里没有功能包", "The list declares no packs");
      return;
    }
    keys.sort(function (a, b) { return (pk[a].required ? 0 : 1) - (pk[b].required ? 0 : 1); });
    keys.forEach(function (k) { box.appendChild(row(k, pk[k])); });
    var miss = keys.filter(function (k) { return !pk[k].installed; });
    if (tag) {
      tag.textContent = miss.length ? say("缺 " + miss.length + " 个包", miss.length + " packs missing")
                                    : say("齐全", "all installed");
      tag.className = "tag " + (miss.length ? "t-yellow" : "t-green");
    }
    var msg = $("pkgMsg");
    if (msg && !BUSY) {
      msg.textContent = d.fetch && d.fetch.ready === false
        ? say("下载器不可用：" + (d.fetch.hint || "modelhub.fetch 不在"), "")
        : "";
    }
  }

  function load() {
    var box = $("pkgBox");
    if (box) { box.className = "hint"; box.textContent = "正在读取权重清单…"; }
    return fetch("/api/weights/status")
      .then(function (r) { return r.ok ? r.json() : null; })
      .then(function (d) { paint(d); return d; })
      .catch(function () { paint(null); return null; });
  }

  function setMsg(txt) {
    var m = $("pkgMsg");
    if (m) m.textContent = txt || "";
  }

  function stopTick() { if (TIMER) { clearInterval(TIMER); TIMER = 0; } }

  function tick(group) {
    fetch("/api/weights/fetch/status").then(function (r) { return r.ok ? r.json() : null; })
      .then(function (s) {
        if (!s || typeof s !== "object") throw new Error("no status");
        var line = s.line || "";
        setMsg(s.pct + "% · " + line);
        if (s.running) return;
        stopTick();
        BUSY = "";
        if (s.error) { setMsg(say("下载失败：" + s.error, "Download failed: " + s.error)); load(); return; }
        setMsg(say("下载完成，重新检测…", "Download finished — re-checking…"));
        load();
      })
      .catch(function () { stopTick(); BUSY = ""; setMsg(say("读不到下载进度", "Cannot read download progress")); });
  }

  function fetchGroup(group, btn) {
    if (BUSY) return;
    var base = String(($("pkgUrl") && $("pkgUrl").value) || "").trim();
    if (!base) {
      /* 清单里没有 url，必须给镜像 —— 别让用户点了才在日志里看到这句 */
      setMsg(say("先填镜像地址：下载器需要 https:// 前缀的权重站（清单里没有内置 url）。",
                 "Enter a mirror first: the downloader needs an https:// weight site (the list has no built-in url)."));
      return;
    }
    BUSY = group;
    if (btn) { btn.disabled = true; btn.textContent = "准备中…"; }
    setMsg("…");
    fetch("/api/weights/fetch", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ group: group, url_base: base })
    }).then(function (r) {
      return r.json().then(function (d) {
        if (!r.ok) throw new Error((d && (d.detail || d.error)) || ("HTTP " + r.status));
        return d;
      });
    }).then(function () {
      stopTick();
      TIMER = setInterval(function () { tick(group); }, 1000);
      tick(group);
    }).catch(function (e) {
      BUSY = "";
      if (btn) { btn.disabled = false; btn.textContent = "下载"; }
      setMsg(say("下载没能启动：" + ((e && e.message) || e), ""));
    });
  }

  function enter() {
    var box = $("pkgBox");
    if (!box) return null;
    if (!box.dataset.pkBound) {
      box.dataset.pkBound = "1";
      var r = $("pkgRefresh");
      if (r) r.addEventListener("click", function () { load(); });
      var u = $("pkgUrl");
      if (u) {
        try { u.value = window.localStorage.getItem("dfl-weights-mirror") || ""; } catch (e) {}
        u.addEventListener("change", function () {
          try { window.localStorage.setItem("dfl-weights-mirror", String(u.value || "").trim()); } catch (e) {}
        });
      }
    }
    return load();
  }

  window.WPK = { enter: enter, refresh: load, _test: { stateText: stateText, size: size } };

  /* 刷新 / 深链接直接落在设置页：主 IIFE 的 go() 在这段之前就跑过了，
     那时 window.WPK 还不存在，这里补一次（和 WKG 对 tool-mask 的处理一致）。 */
  try {
    document.addEventListener("DOMContentLoaded", function () {
      var h = "";
      try { h = String((window.location && window.location.hash) || "").replace(/^#/, ""); } catch (e) {}
      if (h === "settings") enter();
    });
  } catch (e2) {}
})();

