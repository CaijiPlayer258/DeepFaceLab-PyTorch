/* 合成工作室 · 人脸数据库（导出流程的第二步）。
 *
 * 位置：app/web/js/face_db.js，由 index.html 在 app.js 之后单独引入。
 * 职责：
 *   · 消费 /api/merge/faces/* 的 8 个接口，渲染组卡片 / 128 缩略图 / 回收站；
 *   · 组间拖拽（Pointer Events，松手才发 move，拖动中实时高亮落点）；
 *   · Ctrl / Shift 多选、批量 move / trash、键盘 M / T；
 *   · 从当前帧提取（extract）并在多脸时让用户勾选；
 *   · 「继续导出」只把控制权交回 app.js，自己绝不发 /export 请求。
 *
 * 不依赖任何框架；所有文案走 index.html 或 i18n.js 词典，错误一律 toast。
 */
(function () {
  "use strict";

  var API = "/api/merge/faces";
  var THUMB_FAIL_GAP = 4000;

  var state = {
    open: false,
    project: "",
    mode: "export",
    frame: null,
    allowExtract: true,
    dfm: "",
    dfmNames: [],
    data: null,
    loaded: false,
    loading: false,
    busy: false,
    extracting: false,
    selected: {},
    anchor: "",
    activeGroup: "",
    order: [],
    seq: 0,
    sizes: {},
    onContinue: null,
    onCancel: null,
    modal: null,
    modalPrev: null,
    drag: null,
    suppressClick: false,
    lastThumbToast: 0
  };

  function $(id) { return document.getElementById(id); }
  function one(sel, root) { return (root || document).querySelector(sel); }
  function all(sel, root) { return Array.prototype.slice.call((root || document).querySelectorAll(sel)); }

  function make(tag, cls, text) {
    var n = document.createElement(tag);
    if (cls) n.className = cls;
    if (text != null) n.textContent = String(text);
    return n;
  }

  function esc(s) {
    return String(s == null ? "" : s).replace(/[&<>"']/g, function (c) {
      return ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c];
    });
  }

  function toast(title, sub, type) {
    if (typeof window.toast === "function") window.toast(title, sub || "", type || "info");
    else if (window.console) console.warn("[face-db]", title, sub || "");
  }

  function projectName(p) {
    return String(p || "").split(/[\\/]/).filter(Boolean).pop() || "—";
  }

  function frameLabel(n) {
    var en = !!(window.I18N && window.I18N.get && window.I18N.get() === "en");
    if (n == null || n === "" || isNaN(Number(n))) return en ? "Frame —" : "帧 —";
    return en ? ("Frame " + n) : ("帧 " + n);
  }

  function apiUrl(extra) {
    var q = "?project=" + encodeURIComponent(state.project);
    if (extra) q += "&" + extra;
    return API + q;
  }

  /* fetch JSON：HTTP 层错误直接抛；业务层 ok:false 由 must() 决定是否抛。
     extract 的 no_face 是 200 + ok:false，所以不能在这里一刀切。 */
  function call(path, opts, timeoutMs) {
    opts = opts || {};
    opts.headers = Object.assign({ "Content-Type": "application/json" }, opts.headers || {});
    var ctl = (typeof AbortController === "function") ? new AbortController() : null;
    var timer = ctl && timeoutMs ? setTimeout(function () { try { ctl.abort(); } catch (e) {} }, timeoutMs) : null;
    if (ctl) opts.signal = ctl.signal;
    return fetch(path, opts).then(function (r) {
      return r.text().then(function (txt) {
        var d = {};
        try { d = txt ? JSON.parse(txt) : {}; } catch (e) { d = { error: txt.slice(0, 160) }; }
        if (timer) { clearTimeout(timer); timer = null; }
        if (!r.ok) throw new Error(d.error || d.detail || ("HTTP " + r.status));
        return d;
      });
    }, function (e) {
      if (timer) { clearTimeout(timer); timer = null; }
      throw new Error(String((e && e.message) || e));
    });
  }

  function must(d, fallback) {
    if (!d || d.ok === false) {
      throw new Error(String((d && (d.detail || d.error)) || fallback || "请求失败"));
    }
    return d;
  }

  function post(path, body) {
    return call(path, { method: "POST", body: JSON.stringify(body || {}) }, 30000);
  }

  function del(path, body) {
    return call(path, { method: "DELETE", body: JSON.stringify(body || {}) }, 30000);
  }

  function thumbFailed() {
    var now = Date.now();
    if (now - state.lastThumbToast < THUMB_FAIL_GAP) return;
    state.lastThumbToast = now;
    toast("缩略图加载失败", "人脸记录还在，稍后刷新或重试即可", "warn");
  }

  /* 一张缩略图：加载失败时保留占位块，绝不显示破图标。 */
  function makeThumb(src, extraClass) {
    var box = make("div", "fdb-thumb" + (extraClass ? " " + extraClass : ""));
    var fb = make("div", "fdb-fallback");
    fb.appendChild(make("b", null, "缩略图"));
    fb.appendChild(make("span", null, "加载失败"));
    if (src) {
      var img = document.createElement("img");
      img.alt = "";
      img.loading = "lazy";
      img.decoding = "async";
      img.draggable = false;          /* 图片默认 draggable 会触发原生拖拽，pointer 流被 cancel */
      img.src = src;
      img.addEventListener("error", function () {
        box.parentNode && box.parentNode.classList && box.parentNode.classList.add("is-broken");
        box.classList.add("is-broken");
        thumbFailed();
      });
      img.addEventListener("load", function () {
        box.parentNode && box.parentNode.classList && box.parentNode.classList.remove("is-broken");
        box.classList.remove("is-broken");
      });
      box.appendChild(img);
    } else {
      box.classList.add("is-broken");
      box.parentNode && box.parentNode.classList && box.parentNode.classList.add("is-broken");
    }
    box.appendChild(fb);
    return box;
  }

  function faceThumb(face) {
    return face.thumb_url || (API + "/thumb?project=" + encodeURIComponent(state.project) +
      "&id=" + encodeURIComponent(face.id));
  }

  /* ---------- 当前帧 / DFM ---------- */
  function currentFrame() {
    if (state.frame != null && state.frame !== "") return Number(state.frame) || 0;
    if (!state.project) return null;
    var mv = window.__mv || {};
    if (!(Number(mv.frames) > 0)) return null;
    if (window.MergeTimeline && window.MergeTimeline.getFrame) {
      var n = window.MergeTimeline.getFrame();
      if (typeof n === "number" && isFinite(n)) return Math.max(0, Math.round(n));
    }
    return 0;
  }

  function selectedDfms() {
    var out = [];
    try {
      if (typeof window.mgModelsPayload === "function") {
        (window.mgModelsPayload() || []).forEach(function (m) {
          var n = String((m && m.name) || "").trim();
          if (n && out.indexOf(n) < 0) out.push(n);
        });
      }
    } catch (e) { /* 模型列表还没画出来时走组名兜底 */ }
    var groups = (state.data && state.data.groups) || [];
    groups.forEach(function (g) {
      if (!g.orphan && g.dfm && out.indexOf(g.dfm) < 0) out.push(g.dfm);
    });
    return out;
  }

  function collectDfmNames() {
    var out = [];
    try {
      if (typeof window.mgModelsPayload === "function") {
        (window.mgModelsPayload() || []).forEach(function (m) {
          var n = String((m && m.name) || "").trim();
          if (n && out.indexOf(n) < 0) out.push(n);
        });
      }
    } catch (e) { /* ignore */ }
    var groups = (state.data && state.data.groups) || [];
    groups.forEach(function (g) {
      var n = String(g.dfm || "").trim();
      if (n && out.indexOf(n) < 0) out.push(n);
    });
    return out;
  }

  function refreshDfmOptions(prefer) {
    var sel = $("fdbDfm");
    if (!sel) return;
    state.dfmNames = collectDfmNames();
    var keep = prefer != null ? String(prefer) : String(sel.value || state.dfm || "");
    var opts = [""].concat(state.dfmNames);
    sel.innerHTML = "";
    opts.forEach(function (name) {
      var o = document.createElement("option");
      o.value = name;
      o.textContent = name || "不指定（未分组）";
      sel.appendChild(o);
    });
    if (opts.indexOf(keep) < 0) keep = opts[1] || "";
    sel.value = keep;
    state.dfm = keep;
    /* select 被 app.js 换成了 .dd 替身：替身的文字只在 change 时同步。
       这里用原生 change 通知它，否则选项换了、按钮还显示上一轮的空白。 */
    try { sel.dispatchEvent(new Event("change", { bubbles: true })); } catch (e) {}
  }

  function syncDfmToGroup(gid) {
    var group = groupById(gid);
    if (group && !group.orphan && group.dfm) {
      state.dfm = group.dfm;
      var sel = $("fdbDfm");
      if (sel && Array.prototype.some.call(sel.options, function (o) { return o.value === group.dfm; })) {
        sel.value = group.dfm;
        try { sel.dispatchEvent(new Event("change", { bubbles: true })); } catch (e) {}
      }
    }
  }

  function groupById(gid) {
    var groups = (state.data && state.data.groups) || [];
    for (var i = 0; i < groups.length; i++) if (groups[i].id === gid) return groups[i];
    return null;
  }

  function groupLabel(gid) {
    if (!gid) return "未分组";
    var g = groupById(gid);
    return g ? (g.name || g.dfm || gid) : "未分组";
  }

  function groupSize(g) {
    if (!g) return 512;
    var n = Number(g.size);
    if (isFinite(n) && n > 0) return n;
    var cached = Number(state.sizes[g.dfm]);
    return isFinite(cached) && cached > 0 ? cached : 512;
  }

  /* ---------- 数据加载 ---------- */
  function applyData(d) {
    if (!d || !Array.isArray(d.groups) || !Array.isArray(d.faces)) {
      throw new Error("返回的人脸库格式不对");
    }
    state.data = d;
    state.loaded = true;
    state.loading = false;
    state.order = [];
    var groups = {};
    d.groups.forEach(function (g) { groups[g.id] = g; });
    d.faces.forEach(function (f) {
      if (!f.trashed) state.order.push(f.id);
    });
    state.order = state.order.filter(function (id) {
      var f = faceById(id);
      return f && !f.trashed;
    });
    // 选中集合只保留还活跃的 id，避免删/移之后残留
    Object.keys(state.selected).forEach(function (id) {
      var f = faceById(id);
      if (!f || f.trashed) delete state.selected[id];
    });
    if (state.anchor) {
      var a = faceById(state.anchor);
      if (!a || a.trashed) state.anchor = "";
    }
    d.groups.forEach(function (g) {
      var n = Number(g.size);
      if (isFinite(n) && n > 0 && g.dfm) state.sizes[g.dfm] = n;
    });
    render();
  }

  function failLoad(e) {
    state.loading = false;
    state.loaded = false;
    var box = $("fdbError");
    var msg = $("fdbErrorMsg");
    if (msg) msg.textContent = String((e && e.message) || e || "未知错误").slice(0, 180);
    if (box) box.hidden = false;
    var ld = $("fdbLoading");
    if (ld) ld.hidden = true;
    toast("人脸库加载失败", String((e && e.message) || e || "未知错误").slice(0, 160), "error");
    updateActions();
  }

  function load() {
    if (!state.project) { failLoad(new Error("没有项目路径")); return; }
    var seq = ++state.seq;
    state.loading = true;
    state.loaded = false;
    var ld = $("fdbLoading");
    var er = $("fdbError");
    if (ld) ld.hidden = false;
    if (er) er.hidden = true;
    updateActions();
    return call(apiUrl(), { cache: "no-store" }, 30000).then(function (d) {
      if (seq !== state.seq) return;
      applyData(must(d, "读取失败"));
    }).catch(function (e) {
      if (seq !== state.seq) return;
      failLoad(e);
    });
  }

  /* 进页面先把当前模型选择同步成组：取消勾选的 dfm 会变 orphan，而不是把脸带走。 */
  function openLoad() {
    var dfms = selectedDfms();
    if (!dfms.length) { return load(); }
    var seq = ++state.seq;
    state.loading = true;
    state.loaded = false;
    var ld = $("fdbLoading");
    var er = $("fdbError");
    if (ld) ld.hidden = false;
    if (er) er.hidden = true;
    updateActions();
    return post(API + "/groups", { project: state.project, dfms: dfms }).then(function (d) {
      if (seq !== state.seq) return;
      applyData(must(d, "同步分组失败"));
    }).catch(function (e) {
      if (seq !== state.seq) return;
      toast("分组同步失败", String((e && e.message) || e).slice(0, 160), "error");
      return load();
    });
  }

  function faceById(id) {
    var faces = (state.data && state.data.faces) || [];
    for (var i = 0; i < faces.length; i++) if (faces[i].id === id) return faces[i];
    return null;
  }

  function activeFaces() {
    return ((state.data && state.data.faces) || []).filter(function (f) { return !f.trashed; });
  }

  function trashedFaces() {
    return ((state.data && state.data.faces) || []).filter(function (f) { return f.trashed; });
  }

  /* ---------- 选择 ---------- */
  function selectedIds() {
    var out = [];
    state.order.forEach(function (id) { if (state.selected[id]) out.push(id); });
    Object.keys(state.selected).forEach(function (id) {
      if (out.indexOf(id) < 0) {
        var f = faceById(id);
        if (f && !f.trashed) out.push(id);
      }
    });
    return out;
  }

  function setSelected(map) {
    state.selected = {};
    Object.keys(map || {}).forEach(function (id) { if (map[id]) state.selected[id] = true; });
    updateSelection();
  }

  function selectOne(id, ev) {
    if (!id) return;
    var ctrl = !!(ev && (ev.ctrlKey || ev.metaKey));
    var shift = !!(ev && ev.shiftKey);
    if (shift && state.anchor) {
      var a = state.order.indexOf(state.anchor);
      var b = state.order.indexOf(id);
      if (a >= 0 && b >= 0) {
        var lo = Math.min(a, b), hi = Math.max(a, b);
        var next = {};
        for (var i = lo; i <= hi; i++) next[state.order[i]] = true;
        setSelected(next);
        return;
      }
    }
    if (ctrl) {
      var cur = Object.assign({}, state.selected);
      if (cur[id]) delete cur[id]; else cur[id] = true;
      setSelected(cur);
    } else {
      var only = {};
      only[id] = true;
      setSelected(only);
    }
    state.anchor = id;
  }

  function updateSelection() {
    all(".fdb-face").forEach(function (c) {
      var on = !!state.selected[c.getAttribute("data-face-id")];
      c.classList.toggle("is-selected", on);
      c.setAttribute("aria-selected", on ? "true" : "false");
    });
    updateActions();
  }

  function updateActions() {
    var ids = selectedIds();
    var n = ids.length;
    var trashN = trashedFaces().length;
    var dis = state.busy || state.loading || !state.loaded;
    var sc = $("fdbSelCount");
    if (sc) sc.textContent = String(n);
    ["fdbMoveBtn", "fdbTrashBtn", "fdbClearSel"].forEach(function (id) {
      var b = $(id); if (b) b.disabled = dis || n === 0;
    });
    var allBtn = $("fdbPickAll");
    if (allBtn) allBtn.disabled = dis || state.order.length === 0;
    var continueBtn = $("fdbContinue");
    if (continueBtn) continueBtn.disabled = dis;
    var rAll = $("fdbRestoreAll");
    if (rAll) rAll.disabled = dis || trashN === 0;
    var pAll = $("fdbPurgeAll");
    if (pAll) pAll.disabled = dis || trashN === 0;
    updateExtract();
  }

  function updateExtract() {
    var wrap = $("fdbExtractWrap");
    var btn = $("fdbExtract");
    if (wrap) wrap.hidden = !state.allowExtract;
    if (!btn) return;
    if (!state.allowExtract) { btn.disabled = true; return; }
    var reason = "";
    if (!state.project) reason = "还没有项目";
    else if (currentFrame() == null) reason = "还没有当前帧";
    else if (state.loading) reason = "人脸库还在读取";
    else if (state.extracting) reason = "正在提取";
    else if (state.busy) reason = "还有操作在进行";
    btn.disabled = !!reason;
    btn.title = reason;
    btn.textContent = state.extracting ? "提取中…" : "从当前帧提取人脸";
  }

  /* ---------- 渲染：组 / 人脸 / 回收站 / 统计 ---------- */
  function render() {
    var focusBefore = document.activeElement && document.activeElement.getAttribute
      ? document.activeElement.getAttribute("data-face-id") : "";
    renderUngrouped();
    renderGroups();
    renderTrash();
    renderStats();
    renderMoveOptions();
    refreshDfmOptions(state.dfm);
    updateSelection();
    updateExtract();
    var empty = $("fdbEmpty");
    if (empty) empty.hidden = activeFaces().length !== 0;
    var ld = $("fdbLoading");
    if (ld) ld.hidden = !state.loading;
    if (focusBefore) {
      var next = one('.fdb-face[data-face-id="' + cssAttr(focusBefore) + '"]');
      if (next && next.focus) { try { next.focus({ preventScroll: true }); } catch (e) { next.focus(); } }
    }
  }

  function cssAttr(s) {
    return String(s).replace(/\\/g, "\\\\").replace(/"/g, '\\"');
  }

  function makeFace(face, gid) {
    var card = make("div", "fdb-face");
    card.setAttribute("data-face-id", face.id);
    card.setAttribute("data-group", gid || "");
    card.setAttribute("draggable", "false");
    card.setAttribute("tabindex", "0");
    card.setAttribute("role", "option");
    card.setAttribute("aria-selected", state.selected[face.id] ? "true" : "false");
    if (state.selected[face.id]) card.classList.add("is-selected");
    var srcFrame = face.source && face.source.frame;
    var label = frameLabel(srcFrame);
    var aria = [label, face.id, groupLabel(gid)].join(" · ");
    card.setAttribute("aria-label", aria);
    card.title = face.id + " · " + label + " · " + groupLabel(gid);
    var thumb = makeThumb(faceThumb(face));
    if (thumb.classList.contains("is-broken")) card.classList.add("is-broken");
    card.appendChild(thumb);
    var sel = make("span", "fdb-selmark", "✓");
    sel.setAttribute("aria-hidden", "true");
    card.appendChild(sel);
    var meta = make("div", "fdb-meta");
    var fr = make("span", "fdb-frame", label);
    var id = make("span", "fdb-id", face.id);
    meta.appendChild(fr); meta.appendChild(id);
    card.appendChild(meta);
    return card;
  }

  function renderUngrouped() {
    var grid = $("fdbUngroupedGrid");
    var count = $("fdbUngroupedCount");
    if (!grid) return;
    var faces = activeFaces().filter(function (f) {
      return !f.group || !groupById(f.group);
    });
    grid.innerHTML = "";
    faces.forEach(function (f) { grid.appendChild(makeFace(f, "")); });
    if (count) count.textContent = String(faces.length);
    if (!faces.length) {
      var hint = make("span", "fdb-lane-empty", "拖动到此处可取消分组");
      grid.appendChild(hint);
    }
  }

  function renderGroups() {
    var box = $("fdbGroups");
    if (!box) return;
    box.innerHTML = "";
    var groups = (state.data && state.data.groups) || [];
    var active = activeFaces();
    groups.forEach(function (g) {
      var card = make("section", "fdb-group" + (g.orphan ? " is-orphan" : ""));
      card.setAttribute("data-group", g.id);
      card.setAttribute("data-drop-group", g.id);
      card.setAttribute("aria-label", String(g.name || g.dfm || g.id));
      var hd = make("div", "fdb-group-hd");
      var line = make("div", "fdb-group-line");
      line.appendChild(make("span", "fdb-group-name", g.name || g.dfm || g.id));
      if (g.orphan) line.appendChild(make("span", "fdb-orphan-tag", "已取消选择"));
      hd.appendChild(line);
      var meta = make("div", "fdb-group-meta");
      var n = active.filter(function (f) { return f.group === g.id; }).length;
      var cnt = make("span", "fdb-count");
      cnt.appendChild(make("b", null, n));
      cnt.appendChild(make("span", null, "张脸"));
      meta.appendChild(cnt);
      var sz = groupSize(g);
      var size = make("span", "fdb-size", sz + "×" + sz);
      size.title = "DFM 输入尺寸";
      meta.appendChild(size);
      hd.appendChild(meta);
      card.appendChild(hd);
      var grid = make("div", "fdb-group-grid");
      grid.setAttribute("role", "listbox");
      grid.setAttribute("aria-multiselectable", "true");
      grid.setAttribute("aria-label", String(g.name || g.dfm || g.id) + " 人脸");
      active.filter(function (f) { return f.group === g.id; }).forEach(function (f) {
        grid.appendChild(makeFace(f, g.id));
      });
      if (!grid.children.length) {
        var empty = make("div", "fdb-group-empty", "拖到这里，或从当前帧提取");
        empty.setAttribute("data-drop-group", g.id);
        grid.appendChild(empty);
      }
      card.appendChild(grid);
      box.appendChild(card);
    });
  }

  function renderTrash() {
    var list = $("fdbTrashList");
    var empty = $("fdbTrashEmpty");
    if (!list) return;
    var faces = trashedFaces();
    list.innerHTML = "";
    faces.forEach(function (f) {
      var row = make("div", "fdb-trash-item");
      row.setAttribute("data-face-id", f.id);
      var th = make("div", "fdb-trash-thumb");
      var src = faceThumb(f);
      var img = document.createElement("img");
      img.alt = ""; img.loading = "lazy"; img.src = src;
      var fb = make("span", "fdb-fallback", "×");
      img.addEventListener("error", function () { th.classList.add("is-broken"); thumbFailed(); });
      img.addEventListener("load", function () { th.classList.remove("is-broken"); });
      th.appendChild(img); th.appendChild(fb);
      row.appendChild(th);
      var main = make("div", "fdb-trash-main");
      var nm = make("div", "fdb-trash-name", (f.source && frameLabel(f.source.frame)) + " · " + f.id);
      nm.title = String((f.source && f.source.video) || "");
      var sub = make("div", "fdb-trash-sub", groupLabel(f.group));
      main.appendChild(nm); main.appendChild(sub); row.appendChild(main);
      var acts = make("div", "fdb-trash-acts");
      var rb = make("button", "mini", "还原");
      rb.type = "button"; rb.setAttribute("data-act", "restore");
      var db = make("button", "mini fdb-danger", "彻底删除");
      db.type = "button"; db.setAttribute("data-act", "delete");
      acts.appendChild(rb); acts.appendChild(db);
      row.appendChild(acts);
      list.appendChild(row);
    });
    if (empty) empty.hidden = faces.length !== 0;
    var c = $("fdbTrashCount");
    if (c) c.textContent = String(faces.length);
  }

  function renderStats() {
    var faces = (state.data && state.data.faces) || [];
    var groups = (state.data && state.data.groups) || [];
    var gids = {};
    groups.forEach(function (g) { gids[g.id] = true; });
    var active = faces.filter(function (f) { return !f.trashed; });
    var grouped = active.filter(function (f) { return f.group && gids[f.group]; });
    var un = active.length - grouped.length;
    var tr = faces.length - active.length;
    function put(id, n) { var e = $(id); if (e) e.textContent = String(n); }
    put("fdbStatTotal", faces.length);
    put("fdbStatGrouped", grouped.length);
    put("fdbStatUngrouped", un);
    put("fdbStatTrash", tr);
    put("fdbTrashCount", tr);
    put("fdbUngroupedCount", active.filter(function (f) { return !f.group || !gids[f.group]; }).length);
  }

  function renderMoveOptions() {
    var sel = $("fdbMoveTarget");
    if (!sel) return;
    var keep = sel.value;
    var groups = (state.data && state.data.groups) || [];
    sel.innerHTML = "";
    var o0 = document.createElement("option");
    o0.value = ""; o0.textContent = "未分组";
    sel.appendChild(o0);
    groups.forEach(function (g) {
      var o = document.createElement("option");
      o.value = g.id;
      o.textContent = (g.name || g.dfm || g.id) + (g.orphan ? "（已取消选择）" : "");
      sel.appendChild(o);
    });
    var ok = false;
    Array.prototype.forEach.call(sel.options, function (o) { if (o.value === keep) ok = true; });
    sel.value = ok ? keep : (groups[0] ? groups[0].id : "");
    try { sel.dispatchEvent(new Event("change", { bubbles: true })); } catch (e) {}
  }

  /* ---------- 组间投放 ---------- */
  function apiMove(ids, group) {
    return post(API + "/move", { project: state.project, ids: ids, group: group }).then(function (d) {
      return must(d, "移动失败");
    });
  }

  function moveFaces(ids, group) {
    ids = (ids || []).filter(Boolean);
    if (!ids.length) { toast("没有选中人脸", "先点一张，或按住 Ctrl 多选", "warn"); return Promise.resolve(); }
    if (state.busy) return Promise.resolve();
    state.busy = true;
    updateActions();
    var target = groupLabel(group);
    return apiMove(ids, group).then(function (d) {
      toast("已移动 " + d.updated + " 张人脸", "目标：" + target, "success");
      state.selected = {};
      state.anchor = "";
      return load();
    }).catch(function (e) {
      toast("移动失败", String((e && e.message) || e).slice(0, 160), "error");
    }).then(function () {
      state.busy = false;
      updateActions();
    }, function () {
      state.busy = false;
      updateActions();
    });
  }

  /* ---------- 回收站 ---------- */
  function apiTrash(ids) {
    return post(API + "/trash", { project: state.project, ids: ids }).then(function (d) {
      return must(d, "回收站操作失败");
    });
  }

  function trashFaces(ids) {
    ids = (ids || []).filter(Boolean);
    if (!ids.length) { toast("没有选中人脸", "先选一张再移入回收站", "warn"); return Promise.resolve(); }
    if (state.busy) return Promise.resolve();
    state.busy = true;
    updateActions();
    return apiTrash(ids).then(function (d) {
      toast("已移入回收站", "共 " + d.updated + " 张，可随时还原", "success");
      state.selected = {};
      state.anchor = "";
      return load();
    }).catch(function (e) {
      toast("移入回收站失败", String((e && e.message) || e).slice(0, 160), "error");
    }).then(function () { state.busy = false; updateActions(); },
            function () { state.busy = false; updateActions(); });
  }

  function restoreFaces(ids) {
    ids = (ids || []).filter(Boolean);
    if (!ids.length || state.busy) return Promise.resolve();
    state.busy = true;
    updateActions();
    return post(API + "/restore", { project: state.project, ids: ids }).then(function (d) {
      toast("已还原", "共 " + must(d, "还原失败").updated + " 张", "success");
      return load();
    }).catch(function (e) {
      toast("还原失败", String((e && e.message) || e).slice(0, 160), "error");
    }).then(function () { state.busy = false; updateActions(); },
            function () { state.busy = false; updateActions(); });
  }

  function deleteFaces(ids) {
    ids = (ids || []).filter(Boolean);
    if (!ids.length || state.busy) return Promise.resolve();
    state.busy = true;
    updateActions();
    return del(API, { project: state.project, ids: ids }).then(function (d) {
      toast("已彻底删除", "共 " + must(d, "删除失败").count + " 张，无法再还原", "success");
      return load();
    }).catch(function (e) {
      toast("彻底删除失败", String((e && e.message) || e).slice(0, 160), "error");
    }).then(function () { state.busy = false; updateActions(); },
            function () { state.busy = false; updateActions(); });
  }

  function askDelete(ids) {
    ids = (ids || []).filter(Boolean);
    if (!ids.length) return;
    confirmDialog(
      ids.length > 1 ? "彻底删除 " + ids.length + " 张人脸？" : "彻底删除这张人脸？",
      "文件和缩略图都会被删除，不能还原。",
      "彻底删除",
      function () { deleteFaces(ids); },
      true
    );
  }

  /* ---------- 提取 ---------- */
  function rememberSize(dfm, size) {
    var n = Number(size);
    if (dfm && isFinite(n) && n > 0) state.sizes[dfm] = n;
  }

  function extractFaces() {
    var btn = $("fdbExtract");
    if (!btn || btn.disabled || state.extracting) return;
    var frame = currentFrame();
    if (frame == null) { toast("没有当前帧", "先在合成器里打开项目或视频", "warn"); return; }
    var dfm = String(($("fdbDfm") || {}).value || "");
    state.extracting = true;
    updateActions();
    var old = btn.textContent;
    btn.textContent = "提取中…";
    call(API + "/extract", {
      method: "POST",
      body: JSON.stringify({ project: state.project, frame: frame, dfm: dfm })
    }, 120000).then(function (d) {
      if (d && d.ok === false) {
        if (d.reason === "no_face") {
          toast("没有检测到人脸", "换一帧再试，或确认这一帧真的有脸", "warn");
          return null;
        }
        throw new Error(d.detail || d.error || "提取失败");
      }
      must(d, "提取失败");
      rememberSize(d.dfm || dfm, d.size);
      var faces = d.faces || [];
      if (faces.length <= 1) {
        toast("已提取 " + faces.length + " 张人脸",
          "加入：" + groupLabel(d.group || ""), "success");
        return load();
      }
      return openExtractPicker(d, dfm);
    }).catch(function (e) {
      toast("提取失败", String((e && e.message) || e).slice(0, 160), "error");
    }).then(function () {
      state.extracting = false;
      btn.textContent = old || "从当前帧提取人脸";
      updateActions();
    }, function () {
      state.extracting = false;
      btn.textContent = old || "从当前帧提取人脸";
      updateActions();
    });
  }

  function openExtractPicker(d, dfm) {
    var body = make("div", "fdb-extract-picker");
    var targetRow = make("label", "fdb-pick-target");
    targetRow.appendChild(make("span", null, "加入"));
    var sel = document.createElement("select");
    sel.className = "mgsel";
    sel.id = "fdbExtractTarget";
    var groups = (state.data && state.data.groups) || [];
    var o0 = document.createElement("option");
    o0.value = ""; o0.textContent = "未分组";
    sel.appendChild(o0);
    groups.forEach(function (g) {
      if (g.orphan) return;
      var o = document.createElement("option");
      o.value = g.id; o.textContent = g.name || g.dfm || g.id;
      sel.appendChild(o);
    });
    sel.value = d.group || "";
    targetRow.appendChild(sel);
    body.appendChild(targetRow);
    var grid = make("div", "fdb-check-grid");
    (d.faces || []).forEach(function (f, i) {
      var lab = make("label", "fdb-check");
      var cb = document.createElement("input");
      cb.type = "checkbox"; cb.checked = true; cb.value = f.id;
      lab.appendChild(cb);
      if (f.thumb || f.thumb_url) {
        var img = document.createElement("img");
        img.alt = ""; img.src = f.thumb || faceThumb(f);
        img.addEventListener("error", function () { img.style.visibility = "hidden"; });
        lab.appendChild(img);
      } else {
        lab.appendChild(make("span", "fdb-check-ph", "?"));
      }
      lab.appendChild(make("span", null, (i + 1) + " · " + f.id));
      grid.appendChild(lab);
    });
    body.appendChild(grid);
    openModal({
      title: "选择提取结果",
      desc: "取消勾选的脸会放进回收站（可还原），不会留在组里。",
      body: body,
      okText: "加入人脸库",
      focus: grid.querySelector("input") || sel,
      onOk: function () {
        var checked = [];
        all('input[type="checkbox"]', body).forEach(function (cb) { if (cb.checked) checked.push(cb.value); });
        var unchecked = (d.faces || []).map(function (f) { return f.id; })
          .filter(function (id) { return checked.indexOf(id) < 0; });
        var target = sel.value || "";
        var cur = d.group || "";
        state.busy = true;
        updateActions();
        var chain = Promise.resolve();
        if (target !== cur && checked.length) {
          chain = chain.then(function () { return apiMove(checked, target); });
        }
        if (unchecked.length) {
          chain = chain.then(function () { return apiTrash(unchecked); });
        }
        chain.then(function () {
          toast("已加入 " + checked.length + " 张人脸", "目标：" + groupLabel(target), "success");
          return load();
        }).catch(function (e) {
          toast("提取结果处理失败", String((e && e.message) || e).slice(0, 160), "error");
          return load();
        }).then(function () { state.busy = false; updateActions(); },
                function () { state.busy = false; updateActions(); });
      }
    });
  }

  /* ---------- 二次确认 / 组选择 / 提取选择共用弹层 ---------- */
  function openModal(opts) {
    opts = opts || {};
    var modal = $("fdbModal");
    if (!modal) return;
    var title = $("fdbModalTitle"), desc = $("fdbModalDesc"), body = $("fdbModalBody");
    var ok = $("fdbModalOk"), cancel = $("fdbModalCancel");
    if (title) title.textContent = opts.title || "确认";
    if (desc) desc.textContent = opts.desc || "";
    if (body) { body.innerHTML = ""; if (opts.body) body.appendChild(opts.body); }
    if (ok) {
      ok.hidden = !opts.onOk;
      ok.textContent = opts.okText || "确定";
      ok.classList.toggle("fdb-danger", !!opts.danger);
    }
    state.modalPrev = document.activeElement;
    state.modal = { onOk: opts.onOk || null, onCancel: opts.onCancel || null, focus: opts.focus || null };
    modal.hidden = false;
    var focusEl = opts.focus || cancel;
    setTimeout(function () {
      if (focusEl && focusEl.focus) { try { focusEl.focus(); } catch (e) {} }
    }, 30);
  }

  function closeModal(cancelIt) {
    var m = state.modal;
    var modal = $("fdbModal");
    if (modal) modal.hidden = true;
    state.modal = null;
    var focus = state.modalPrev;
    state.modalPrev = null;
    if (focus && focus.focus && document.contains(focus)) {
      try { focus.focus({ preventScroll: true }); } catch (e) { focus.focus(); }
    }
    if (cancelIt && m && m.onCancel) {
      try { m.onCancel(); } catch (e) { /* 用户取消，无需再报 */ }
    }
  }

  function confirmDialog(title, desc, okText, onOk, danger) {
    openModal({ title: title, desc: desc, okText: okText, onOk: onOk, danger: danger, focus: $("fdbModalCancel") });
  }

  function openGroupPicker(ids) {
    var body = make("div", "fdb-modal-list");
    var curGroups = {};
    ids.forEach(function (id) {
      var f = faceById(id);
      if (f) curGroups[f.group || ""] = true;
    });
    var rows = [{ id: "", name: "未分组", orphan: false }].concat((state.data && state.data.groups) || []);
    rows.forEach(function (g) {
      var b = make("button", "fdb-pick" + (curGroups[g.id] ? " is-current" : ""));
      b.type = "button";
      b.setAttribute("data-group", g.id || "");
      b.appendChild(make("span", null, g.name || g.dfm || "未分组"));
      b.appendChild(make("small", null, g.id ? g.id : "group=\"\""));
      b.addEventListener("click", function () {
        closeModal(false);
        moveFaces(ids, g.id || "");
      });
      body.appendChild(b);
    });
    openModal({
      title: "移动 " + ids.length + " 张人脸",
      desc: "选择目标分组；未分组 = 从当前组移出。",
      body: body,
      okText: "取消",
      focus: body.querySelector("button"),
      onOk: function () { closeModal(false); }
    });
    var ok = $("fdbModalOk");
    if (ok) ok.hidden = true;
  }

  /* ---------- 拖拽：Pointer Events，落点实时高亮 ---------- */
  function boardPoint(x, y) {
    var el = document.elementFromPoint(x, y);
    return el && el.closest ? el.closest("[data-drop-group]") : null;
  }

  function clearDrop() {
    all(".fdb-lane.is-drop, .fdb-group.is-drop").forEach(function (z) { z.classList.remove("is-drop"); });
    var tip = $("fdbDropTip");
    if (tip) tip.hidden = true;
    var note = one(".fdb-drop-note.is-live");
    if (note) note.classList.remove("is-live");
  }

  function updateDrop(x, y) {
    var zone = boardPoint(x, y);
    clearDrop();
    if (!zone) return;
    zone.classList.add("is-drop");
    var gid = zone.getAttribute("data-drop-group") || "";
    var label = groupLabel(gid);
    var ids = state.drag ? state.drag.ids : [];
    var tip = $("fdbDropTip");
    if (tip) {
      tip.hidden = false;
      tip.textContent = "移到 " + label + " · " + ids.length + " 张";
      tip.style.left = Math.min(window.innerWidth - 180, x + 16) + "px";
      tip.style.top = Math.min(window.innerHeight - 34, y + 16) + "px";
    }
    if (zone.classList.contains("fdb-lane")) {
      var note = zone.querySelector(".fdb-drop-note");
      if (note) note.classList.add("is-live");
    }
  }

  function startDrag(e) {
    var d = state.drag;
    if (!d) return;
    d.moved = true;
    var cur = Object.assign({}, state.selected);
    if (!cur[d.id]) {
      cur = {}; cur[d.id] = true;
      setSelected(cur);
      state.anchor = d.id;
    }
    d.ids = selectedIds();
    if (!d.ids.length) d.ids = [d.id];
    var source = d.card;
    var ghost = source.cloneNode(true);
    ghost.className = "fdb-ghost";
    ghost.setAttribute("aria-hidden", "true");
    ghost.style.left = (e.clientX + 12) + "px";
    ghost.style.top = (e.clientY + 12) + "px";
    document.body.appendChild(ghost);
    d.ghost = ghost;
    all(".fdb-face").forEach(function (c) {
      if (d.ids.indexOf(c.getAttribute("data-face-id")) >= 0) c.classList.add("is-dragging");
    });
    document.body.classList.add("fdb-dragging");
    d.card.setPointerCapture && d.card.setPointerCapture(e.pointerId);
    state.suppressClick = true;
  }

  function onPointerDown(e) {
    if (!state.open || state.modal || e.button !== 0) return;
    if (!e.target.closest) return;
    var card = e.target.closest(".fdb-face");
    if (!card) return;
    if (e.target.closest("button, a, input, select, textarea")) return;
    state.drag = {
      id: card.getAttribute("data-face-id"),
      card: card,
      x: e.clientX,
      y: e.clientY,
      pointerId: e.pointerId,
      moved: false,
      ids: []
    };
  }

  function onPointerMove(e) {
    var d = state.drag;
    if (!d) return;
    var dx = e.clientX - d.x, dy = e.clientY - d.y;
    if (!d.moved && Math.sqrt(dx * dx + dy * dy) < 6) return;
    if (!d.moved) {
      try { startDrag(e); }
      catch (err) {
        state.drag = null;
        state.lastDragError = String((err && err.message) || err);
        if (window.console) console.error("[face-db] drag start failed", err);
        toast("拖拽启动失败", state.lastDragError.slice(0, 140), "error");
        return;
      }
    }
    e.preventDefault();
    if (d.ghost) {
      d.ghost.style.left = (e.clientX + 12) + "px";
      d.ghost.style.top = (e.clientY + 12) + "px";
    }
    var board = $("fdbBoard");
    if (board) {
      var r = board.getBoundingClientRect();
      if (e.clientX < r.left + 36) board.scrollLeft -= 14;
      else if (e.clientX > r.right - 36) board.scrollLeft += 14;
    }
    updateDrop(e.clientX, e.clientY);
  }

  function endDrag(e) {
    var d = state.drag;
    if (!d) return;
    var moved = d.moved;
    var ids = d.ids.slice();
    var zone = moved ? boardPoint(e.clientX, e.clientY) : null;
    var target = zone ? (zone.getAttribute("data-drop-group") || "") : null;
    if (d.ghost) d.ghost.remove();
    document.body.classList.remove("fdb-dragging");
    all(".fdb-face.is-dragging").forEach(function (c) { c.classList.remove("is-dragging"); });
    clearDrop();
    try { d.card.releasePointerCapture && d.card.releasePointerCapture(d.pointerId); } catch (err) {}
    state.drag = null;
    if (moved) {
      setTimeout(function () { state.suppressClick = false; }, 250);
      if (target !== null) {
        var allSame = ids.every(function (id) {
          var f = faceById(id);
          return f && (f.group || "") === target;
        });
        if (allSame) toast("已经在目标分组", groupLabel(target), "info");
        else moveFaces(ids, target);
      } else {
        toast("没有放到分组上", "拖到组卡片或顶部「未分组」才会移动", "info");
      }
    }
  }

  function onPointerCancel() {
    var d = state.drag;
    if (!d) return;
    if (d.ghost) d.ghost.remove();
    document.body.classList.remove("fdb-dragging");
    all(".fdb-face.is-dragging").forEach(function (c) { c.classList.remove("is-dragging"); });
    clearDrop();
    state.drag = null;
    setTimeout(function () { state.suppressClick = false; }, 0);
  }

  /* ---------- 键盘 ---------- */
  function typing(e) {
    var t = e.target || {};
    var tag = String(t.tagName || "").toUpperCase();
    return tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT" || t.isContentEditable;
  }

  function focusedFaceId() {
    var a = document.activeElement;
    return (a && a.closest && a.closest(".fdb-face")) ? a.closest(".fdb-face").getAttribute("data-face-id") : "";
  }

  function keyTargetIds() {
    var ids = selectedIds();
    if (ids.length) return ids;
    var id = focusedFaceId();
    if (id) {
      var oneMap = {}; oneMap[id] = true;
      setSelected(oneMap);
      state.anchor = id;
      return [id];
    }
    return [];
  }

  function onKeyDown(e) {
    if (!state.open) return;
    if (state.modal) {
      if (e.key === "Escape") { e.preventDefault(); closeModal(true); return; }
      if (e.key === "Tab") {
        var card = one(".fdb-modal-card");
        if (!card) return;
        var list = all('button:not([hidden]), input:not([type="hidden"]), select, [tabindex="0"]', card)
          .filter(function (x) { return !x.disabled && x.offsetParent !== null; });
        if (!list.length) return;
        var first = list[0], last = list[list.length - 1];
        if (e.shiftKey && document.activeElement === first) { e.preventDefault(); last.focus(); }
        else if (!e.shiftKey && document.activeElement === last) { e.preventDefault(); first.focus(); }
      }
      return;
    }
    if (!isVisible()) return;
    if (typing(e)) return;
    var key = String(e.key || "").toLowerCase();
    var face = focusedFaceId();
    if (key === "m") {
      var ids = keyTargetIds();
      if (!ids.length) { toast("先选中人脸", "点一张，或用 Tab 聚焦后按 M", "warn"); return; }
      e.preventDefault();
      openGroupPicker(ids);
      return;
    }
    if (key === "t") {
      var ids2 = keyTargetIds();
      if (!ids2.length) { toast("先选中人脸", "点一张，或用 Tab 聚焦后按 T", "warn"); return; }
      e.preventDefault();
      trashFaces(ids2);
      return;
    }
    if (key === "escape") {
      setSelected({}); state.anchor = ""; return;
    }
    if (key === "a" && (e.ctrlKey || e.metaKey)) {
      var allMap = {};
      state.order.forEach(function (id) { allMap[id] = true; });
      setSelected(allMap);
      e.preventDefault();
      return;
    }
    if (face && (key === "enter" || key === " " || e.key === "Spacebar")) {
      e.preventDefault();
      selectOne(face, e);
      return;
    }
    if (face && (key === "arrowright" || key === "arrowdown" || key === "arrowleft" || key === "arrowup" ||
                 key === "home" || key === "end")) {
      var cards = all(".fdb-face");
      var idx = cards.indexOf(document.activeElement.closest(".fdb-face"));
      if (idx < 0) return;
      var next = idx;
      if (key === "arrowright" || key === "arrowdown") next = Math.min(cards.length - 1, idx + 1);
      if (key === "arrowleft" || key === "arrowup") next = Math.max(0, idx - 1);
      if (key === "home") next = 0;
      if (key === "end") next = cards.length - 1;
      if (next !== idx) { e.preventDefault(); cards[next].focus(); }
    }
  }

  /* ---------- 页面打开 / 关闭 ---------- */
  function isVisible() {
    var view = $("mgFaceDb");
    if (!view || !view.classList.contains("in")) return false;
    return view.offsetParent !== null;
  }

  function updateHeader() {
    var project = $("fdbProject");
    if (project) {
      project.textContent = projectName(state.project);
      project.title = state.project;
    }
    var mode = $("fdbMode");
    if (mode) mode.textContent = state.mode === "export-project" ? "项目导出" : "常规导出";
    var frame = $("fdbFrame");
    var f = currentFrame();
    var en = !!(window.I18N && window.I18N.get && window.I18N.get() === "en");
    if (frame) frame.textContent = f == null ? (en ? "Current frame —" : "当前帧 —")
                                             : ((en ? "Current frame " : "当前帧 ") + f);
  }

  function open(opts) {
    opts = opts || {};
    if (!opts.project) {
      toast("还没有素材", "先导入视频或打开 DFL 项目", "warn");
      return false;
    }
    closeModal(false);
    state.open = true;
    state.project = String(opts.project);
    state.mode = opts.mode === "export-project" ? "export-project" : "export";
    state.onContinue = typeof opts.onContinue === "function" ? opts.onContinue : null;
    state.onCancel = typeof opts.onCancel === "function" ? opts.onCancel : null;
    state.allowExtract = opts.allowExtract !== false;
    state.frame = (opts.frame != null) ? opts.frame : currentFrameSnapshot();
    state.selected = {};
    state.anchor = "";
    state.activeGroup = "";
    state.data = null;
    state.loaded = false;
    state.loading = false;
    state.busy = false;
    state.extracting = false;
    state.dfm = selectedDfms()[0] || "";
    state.seq++;
    updateHeader();
    var empty = $("fdbEmpty"); if (empty) empty.hidden = true;
    var er = $("fdbError"); if (er) er.hidden = true;
    var ld = $("fdbLoading"); if (ld) ld.hidden = false;
    if (window.mgStep) window.mgStep("facedb");
    refreshDfmOptions(state.dfm);
    var firstGroup = null;
    openLoad();
    setTimeout(function () {
      var board = $("fdbBoard");
      if (board && board.focus) { try { board.focus({ preventScroll: true }); } catch (e) { board.focus(); } }
    }, 60);
    updateActions();
    return true;
  }

  function currentFrameSnapshot() {
    var mv = window.__mv || {};
    if (!(Number(mv.frames) > 0)) return null;
    if (window.MergeTimeline && window.MergeTimeline.getFrame) {
      var n = window.MergeTimeline.getFrame();
      if (typeof n === "number" && isFinite(n)) return Math.max(0, Math.round(n));
    }
    return 0;
  }

  function cancel() {
    var cb = state.onCancel;
    state.open = false;
    state.onCancel = null;
    state.onContinue = null;
    closeModal(false);
    if (cb) cb();
    else if (window.mgStep) window.mgStep("main");
  }

  function askCancel() {
    confirmDialog(
      "返回会取消本次导出？",
      "已经整理的人脸组会保留；这次不会开始导出。",
      "返回",
      function () { cancel(); },
      false
    );
  }

  function proceed() {
    if (state.loading) { toast("还在读取人脸库", "等列表出来再继续", "info"); return; }
    if (!state.loaded) { toast("人脸库没有就绪", "先处理加载错误再继续", "warn"); return; }
    if (state.busy) { toast("还有操作在进行", "等当前操作完成", "info"); return; }
    var cb = state.onContinue;
    if (!cb) { toast("没有导出动作", "请从合成器的导出按钮进入", "warn"); return; }
    /* 继续 = 离开人脸库；如果 app 那边的请求同步失败、页面还停在这里，
       1.2 秒后把状态翻回来，别让键盘/面包屑以为已经关掉了。 */
    state.open = false;
    try { cb(); }
    catch (e) { state.open = true; toast("导出没有启动", String((e && e.message) || e).slice(0, 160), "error"); return; }
    setTimeout(function () { if (isVisible()) state.open = true; }, 1200);
  }

  /* ---------- 事件绑定 ---------- */
  function bind() {
    var board = $("fdbBoard");
    if (board) {
      board.addEventListener("click", function (e) {
        if (state.suppressClick) {
          state.suppressClick = false;
          e.preventDefault();
          e.stopPropagation();
          return;
        }
        var face = e.target.closest && e.target.closest(".fdb-face");
        if (face) {
          var id = face.getAttribute("data-face-id");
          var gid = face.getAttribute("data-group") || "";
          if (gid) { state.activeGroup = gid; syncDfmToGroup(gid); }
          selectOne(id, e);
          try { face.focus({ preventScroll: true }); } catch (err) {}
          return;
        }
        var group = e.target.closest && e.target.closest(".fdb-group");
        if (group) {
          state.activeGroup = group.getAttribute("data-group") || "";
          syncDfmToGroup(state.activeGroup);
        }
      });
      board.addEventListener("pointerdown", onPointerDown);
      board.addEventListener("dragstart", function (e) {
        /* 拖拽自己实现；原生 dragstart 只会把 pointer 流 cancel 掉。 */
        if (e.target.closest && e.target.closest(".fdb-face")) e.preventDefault();
      });
    }
    document.addEventListener("pointermove", onPointerMove, { passive: false });
    document.addEventListener("pointerup", endDrag);
    document.addEventListener("pointercancel", onPointerCancel);
    document.addEventListener("keydown", onKeyDown);

    var close = $("fdbClose");
    if (close) close.addEventListener("click", askCancel);
    var cont = $("fdbContinue");
    if (cont) cont.addEventListener("click", proceed);
    var ext = $("fdbExtract");
    if (ext) ext.addEventListener("click", extractFaces);
    var retry = $("fdbRetry");
    if (retry) retry.addEventListener("click", load);

    var pickAll = $("fdbPickAll");
    if (pickAll) pickAll.addEventListener("click", function () {
      var m = {};
      state.order.forEach(function (id) { m[id] = true; });
      setSelected(m);
    });
    var clearSel = $("fdbClearSel");
    if (clearSel) clearSel.addEventListener("click", function () { setSelected({}); state.anchor = ""; });
    var move = $("fdbMoveBtn");
    if (move) move.addEventListener("click", function () {
      moveFaces(selectedIds(), ($("fdbMoveTarget") || {}).value || "");
    });
    var trash = $("fdbTrashBtn");
    if (trash) trash.addEventListener("click", function () { trashFaces(selectedIds()); });

    var trashList = $("fdbTrashList");
    if (trashList) trashList.addEventListener("click", function (e) {
      var btn = e.target.closest && e.target.closest("button[data-act]");
      if (!btn) return;
      var row = btn.closest(".fdb-trash-item");
      if (!row) return;
      var id = row.getAttribute("data-face-id");
      if (btn.getAttribute("data-act") === "restore") restoreFaces([id]);
      else askDelete([id]);
    });
    var rAll = $("fdbRestoreAll");
    if (rAll) rAll.addEventListener("click", function () {
      restoreFaces(trashedFaces().map(function (f) { return f.id; }));
    });
    var pAll = $("fdbPurgeAll");
    if (pAll) pAll.addEventListener("click", function () {
      askDelete(trashedFaces().map(function (f) { return f.id; }));
    });

    var modal = $("fdbModal");
    if (modal) {
      modal.addEventListener("click", function (e) {
        if (e.target === modal) closeModal(true);
      });
    }
    var mCancel = $("fdbModalCancel");
    if (mCancel) mCancel.addEventListener("click", function () { closeModal(true); });
    var mOk = $("fdbModalOk");
    if (mOk) mOk.addEventListener("click", function () {
      var m = state.modal;
      if (!m) return;
      var fn = m.onOk;
      closeModal(false);
      if (fn) { try { fn(); } catch (e) { toast("操作失败", String(e && e.message || e), "error"); } }
    });

    var dfm = $("fdbDfm");
    if (dfm) dfm.addEventListener("change", function () { state.dfm = dfm.value; });
  }

  window.FaceDB = {
    open: open,
    close: cancel,
    requestClose: askCancel,
    refresh: load,
    isOpen: function () { return state.open; },
    currentProject: function () { return state.project; },
    /* 调试/验收用：不改变状态，只把内部开关读出来。 */
    debug: function () {
      return {
        open: state.open,
        loading: state.loading,
        busy: state.busy,
        lastDragError: state.lastDragError || "",
        drag: state.drag ? { id: state.drag.id, ids: state.drag.ids.slice(), moved: state.drag.moved } : null,
        selected: Object.keys(state.selected),
        groups: ((state.data && state.data.groups) || []).length,
        faces: ((state.data && state.data.faces) || []).length
      };
    }
  };

  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", bind);
  else bind();
})();
