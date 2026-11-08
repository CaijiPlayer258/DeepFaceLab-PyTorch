/* 合成工作室 · 人脸数据库（导出流程的第二步）。
 *
 * 位置：app/web/js/face_db.js，由 index.html 在 app.js 之后单独引入。
 * 职责：
 *   · 消费 /api/merge/faces/*，渲染组卡片 / 128 缩略图 / 回收站；
 *   · 拖拽（Pointer Events，松手才发请求，拖动中实时高亮落点）：
 *       组卡片 = move，回收站面板 = trash；拖到别处什么都不做；
 *   · Ctrl / Shift 多选、全选 / 清空、键盘 T（回收站）、Esc（取消选择）；
 *   · 新脸怎么落库由「＋」走 /api/merge/faces/add，落进哪一组靠 addTarget()
 *     （= 下拉「提取到」选中的组 / 第一个组 / 第一个模型）；
 *   · 「继续导出」只把控制权交回 app.js，自己绝不发 /export 请求。
 *
 * 界面上没有「未分组」区、没有移动按钮、没有提取按钮：这个页面只做
 * 「看 + 选 + 拖」。
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
    selected: {},
    anchor: "",
    activeGroup: "",
    order: [],
    seq: 0,
    sizes: {},
    onContinue: null,
    onCancel: null,
    browseOnly: false,
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

  /* ---------- DFM ---------- */
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
    var opts = state.dfmNames.slice();
    sel.innerHTML = "";
    opts.forEach(function (name) {
      var o = document.createElement("option");
      o.value = name;
      o.textContent = name;
      sel.appendChild(o);
    });
    /* 一个模型都没有：给一个禁用的占位，别让下拉空着像个坏控件。
       这时「＋」会提示先选模型 —— 界面里不允许再出现「未分组」这个选项。 */
    if (!opts.length) {
      var ph = document.createElement("option");
      ph.value = "";
      ph.textContent = "—";
      ph.disabled = true;
      sel.appendChild(ph);
    }
    if (opts.indexOf(keep) < 0) keep = opts[0] || "";
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

  /* 组名。空组 id（group=""，老数据）在界面上不再出现「未分组」这个词：
     返回空串，调用方拼 aria/title 时自然少一段。 */
  function groupLabel(gid) {
    if (!gid) return "";
    var g = groupById(gid);
    return g ? (g.name || g.dfm || gid) : "";
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
    /* 后端回的是解析后的**绝对**项目目录（空 project 时就是默认库
       <工作区>/merge）。采纳它：之后缩略图/移动/回收站/分组同步都带这条真路径，
       前端不再自己猜「默认库在哪」。 */
    if (d.project && String(d.project) !== state.project) {
      state.project = String(d.project);
      updateHeader();
    }
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

  /* 首屏就失败时把上一份库的卡片清掉：错误态底下压着别人家 6 张脸，
     用户会以为「库还在、只是报了个错」。刷新失败（已经有数据）则保留。 */
  function clearBoard() {
    ["fdbGroups", "fdbTrashList"].forEach(function (id) {
      var n = $(id);
      if (n) n.innerHTML = "";
    });
    ["fdbTrashCount", "fdbStatTotal", "fdbStatTrash"].forEach(function (id) {
      var n = $(id);
      if (n) n.textContent = "0";
    });
    state.order = [];
    state.selected = {};
    var empty = $("fdbEmpty");
    if (empty) empty.hidden = true;
  }

  function failLoad(e) {
    state.loading = false;
    state.loaded = false;
    if (!state.data) clearBoard();
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
    /* 不再要求 state.project 非空：空串就是默认库，后端 resolve_library 会认，
       请求里 project= 空着也能拿到库内容。 */
    var seq = ++state.seq;
    state.loading = true;
    state.loaded = false;
    var ld = $("fdbLoading");
    var er = $("fdbError");
    if (ld) ld.hidden = false;
    if (er) er.hidden = true;
    updateActions();
    return call(apiUrl(), { cache: "no-store" }, 30000).then(function (d) {
      /* 同一帧同一张脸只留一张：这次 GET 把多出来的挪进了回收站（可还原）。
         提示挂在**响应**上而不是 applyData 上 —— 两个 GET 并发时第一份数据会被
         state.seq 丢掉，丢掉的那份才是带 deduped 的。 */
      var ded = Number(d && d.deduped) || 0;
      if (ded > 0) {
        toast("已把 " + ded + " 张同帧重复的脸移入回收站",
              "同一帧同一张脸只留最新的一张 · 回收站里能还原", "info");
      }
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
    var clearBtn = $("fdbClearSel");
    if (clearBtn) clearBtn.disabled = dis || n === 0;
    var allBtn = $("fdbPickAll");
    if (allBtn) allBtn.disabled = dis || state.order.length === 0;
    /* 底栏只剩「继续导出」这一颗，而且只在导出流程里出现：
       关闭入口只有右上角那一个（browse 模式 = 点它直接回合成视图）。 */
    var continueBtn = $("fdbContinue");
    if (continueBtn) {
      continueBtn.hidden = !!state.browseOnly;
      continueBtn.disabled = dis;
    }
    var rAll = $("fdbRestoreAll");
    if (rAll) rAll.disabled = dis || trashN === 0;
    var pAll = $("fdbPurgeAll");
    if (pAll) pAll.disabled = dis || trashN === 0;
    updateAddTarget();
  }

  /* 「提取到 <dfm>」不再配一颗提取按钮：新脸由合成器框上的「＋」落库
     （/api/merge/faces/add），这里只剩「选中哪个组」这一件事。
     一个模型都没有时下拉禁用，「＋」那边会提示先选模型。 */
  function updateAddTarget() {
    var wrap = $("fdbExtractWrap");
    if (wrap) wrap.hidden = !state.allowExtract;
    var sel = $("fdbDfm");
    if (!sel) return;
    var none = !state.dfmNames.length;
    sel.disabled = none || state.loading || state.busy;
    sel.title = none ? "工作区里还没有模型：先在合成器里选一个 SAEHD 模型"
                     : "新「＋」加进来的脸落进这个组";
  }

  /* ---------- 渲染：组 / 人脸 / 回收站 / 统计 ---------- */
  function render() {
    var focusBefore = document.activeElement && document.activeElement.getAttribute
      ? document.activeElement.getAttribute("data-face-id") : "";
    renderGroups();
    renderTrash();
    renderStats();
    refreshDfmOptions(state.dfm);
    updateSelection();
    updateAddTarget();
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
    var aria = [label, face.id, groupLabel(gid)].filter(Boolean).join(" · ");
    card.setAttribute("aria-label", aria);
    card.title = aria;
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

  /* 老库里 group="" 的脸不再单占一块（「未分组」这个概念已经从界面上消失）：
     顺手画进第一个非 orphan 的组卡片里，拖一下就能真正落进那个组。 */
  function adoptGroupId() {
    var groups = (state.data && state.data.groups) || [];
    for (var i = 0; i < groups.length; i++) if (!groups[i].orphan) return groups[i].id;
    return groups[0] ? groups[0].id : "";
  }

  function renderGroups() {
    var box = $("fdbGroups");
    if (!box) return;
    box.innerHTML = "";
    var groups = (state.data && state.data.groups) || [];
    var active = activeFaces();
    var adopt = adoptGroupId();
    groups.forEach(function (g) {
      var card = make("section", "fdb-group" + (g.orphan ? " is-orphan" : ""));
      card.setAttribute("data-group", g.id);
      card.setAttribute("data-drop-group", g.id);
      card.setAttribute("aria-label", String(g.name || g.dfm || g.id));
      var mine = active.filter(function (f) {
        if (f.group === g.id) return true;
        return g.id === adopt && (!f.group || !groupById(f.group));
      });
      /* 组头一行装完：组名 + 张数 + 输入尺寸（384×384） */
      var hd = make("div", "fdb-group-hd");
      hd.appendChild(make("span", "fdb-group-name", g.name || g.dfm || g.id));
      if (g.orphan) hd.appendChild(make("span", "fdb-orphan-tag", "已取消选择"));
      var cnt = make("span", "fdb-count");
      cnt.appendChild(make("b", null, mine.length));
      cnt.appendChild(make("span", null, "张脸"));
      hd.appendChild(cnt);
      var sz = groupSize(g);
      var size = make("span", "fdb-size", sz + "×" + sz);
      size.title = "DFM 输入尺寸";
      hd.appendChild(size);
      card.appendChild(hd);
      var grid = make("div", "fdb-group-grid");
      grid.setAttribute("role", "listbox");
      grid.setAttribute("aria-multiselectable", "true");
      grid.setAttribute("aria-label", String(g.name || g.dfm || g.id) + " 人脸");
      mine.forEach(function (f) { grid.appendChild(makeFace(f, g.id)); });
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
    var tr = 0;
    faces.forEach(function (f) { if (f.trashed) tr++; });
    function put(id, n) { var e = $(id); if (e) e.textContent = String(n); }
    put("fdbStatTotal", faces.length);
    put("fdbStatTrash", tr);
    put("fdbTrashCount", tr);
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

  /* ---------- 二次确认弹层（彻底删除前问一句） ---------- */
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

  /* ---------- 拖拽：Pointer Events，落点实时高亮 ---------- */
  /* 落点只有两种：组卡片（move）和回收站面板（trash）。
     未分组区已经删掉，拖到别处一律没有行为。 */
  function dropZoneAt(x, y) {
    var el = document.elementFromPoint(x, y);
    if (!el || !el.closest) return null;
    var trash = el.closest("[data-drop-trash]");
    if (trash) return { kind: "trash", el: trash, group: null };
    var group = el.closest("[data-drop-group]");
    if (group) return { kind: "group", el: group, group: group.getAttribute("data-drop-group") || "" };
    return null;
  }

  function clearDrop() {
    all(".fdb-group.is-drop, .fdb-trash.is-drop").forEach(function (z) { z.classList.remove("is-drop"); });
    var tip = $("fdbDropTip");
    if (tip) tip.hidden = true;
  }

  function updateDrop(x, y) {
    var zone = dropZoneAt(x, y);
    clearDrop();
    if (!zone) return;
    zone.el.classList.add("is-drop");
    var ids = state.drag ? state.drag.ids : [];
    var tip = $("fdbDropTip");
    if (!tip) return;
    tip.hidden = false;
    tip.textContent = zone.kind === "trash"
      ? "移入回收站 · " + ids.length + " 张"
      : "移到 " + groupLabel(zone.group) + " · " + ids.length + " 张";
    tip.style.left = Math.min(window.innerWidth - 180, x + 16) + "px";
    tip.style.top = Math.min(window.innerHeight - 34, y + 16) + "px";
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
    var zone = moved ? dropZoneAt(e.clientX, e.clientY) : null;
    if (d.ghost) d.ghost.remove();
    document.body.classList.remove("fdb-dragging");
    all(".fdb-face.is-dragging").forEach(function (c) { c.classList.remove("is-dragging"); });
    clearDrop();
    try { d.card.releasePointerCapture && d.card.releasePointerCapture(d.pointerId); } catch (err) {}
    state.drag = null;
    if (moved) {
      setTimeout(function () { state.suppressClick = false; }, 250);
      /* 拖到空白处：什么都不做（未分组区没了，不再有「取消分组」这个落点） */
      if (!zone) return;
      if (zone.kind === "trash") { trashFaces(ids); return; }
      var target = zone.group;
      var allSame = ids.every(function (id) {
        var f = faceById(id);
        return f && (f.group || "") === target;
      });
      if (allSame) toast("已经在目标分组", groupLabel(target), "info");
      else moveFaces(ids, target);
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

  /* 顶栏只留「人脸数据库 + 已选/全选/清空 + 提取到 + 关闭」：
     项目名副标题、模式徽标、当前帧这些小字都删了（合成器里本来就有）。 */
  function updateHeader() {
    var el = $("fdbSelCount");
    if (el) el.textContent = String(selectedIds().length);
  }

  function open(opts) {
    opts = opts || {};
    /* project 空串 = 视频模式：**不是错误**。后端把空 project 解析成默认库
       <工作区>/merge（不存在就建），列表/缩略图/移动/回收站全都认这一条。
       以前这里直接 return false，于是合成器点「人脸数据库」什么都不会发生。 */
    closeModal(false);
    state.open = true;
    state.project = String(opts.project || "");
    state.mode = opts.mode === "export-project" ? "export-project" : "export";
    state.onContinue = typeof opts.onContinue === "function" ? opts.onContinue : null;
    state.onCancel = typeof opts.onCancel === "function" ? opts.onCancel : null;
    /* 没有 onContinue = 只是来看库的（合成器顶部入口）：不当导出流程拦，
       关闭就是回合成视图，也不再问「返回会取消本次导出？」 */
    state.browseOnly = !state.onContinue;
    state.allowExtract = opts.allowExtract !== false;
    state.frame = (opts.frame != null) ? opts.frame : currentFrameSnapshot();
    state.selected = {};
    state.anchor = "";
    state.activeGroup = "";
    state.data = null;
    state.loaded = false;
    state.loading = false;
    state.busy = false;
    state.dfm = selectedDfms()[0] || "";
    state.seq++;
    updateHeader();
    var cont = $("fdbContinue");
    if (cont) {
      /* 关闭只有一个入口：右上角。底部这颗只在导出流程里叫「继续导出」。 */
      cont.textContent = "继续导出";
      cont.hidden = !!state.browseOnly;
      cont.title = "";
    }
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
    /* 只看库（没有 onContinue）就没有「导出」可取消：直接回合成视图，不弹确认 */
    if (state.browseOnly) { cancel(); return; }
    confirmDialog(
      "返回会取消本次导出？",
      "已经整理的人脸组会保留；这次不会开始导出。",
      "返回",
      function () { cancel(); },
      false
    );
  }

  function proceed() {
    /* 顶部入口进来的「关闭」：不导出，只回合成视图（加载中/出错也允许走） */
    if (state.browseOnly) { cancel(); return; }
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

  /* ---------- 「＋」落库目标：当前选中的组 ---------- */
  /* 界面上已经没有「未分组」，所以新脸必须落进一个真实存在的组：
       1) 页面开着 → 用户在下拉「提取到」里选的那一组（点组卡片也会把它选上）；
       2) 没选 / 没开 → 库里第一个非 orphan 的组；
       3) 库里一个组都没有 → 合成器里选中的第一个模型（后端按 dfm 顺手建组）；
       4) 连模型都没有 → null，前端提示「先选模型」。
     每次点「＋」都现查一次 /api/merge/faces（同一个请求在飞时共用），**不做**
     时间缓存：刚建好组、刚选好模型的两秒里给出旧答案会让「＋」白点一次。 */
  var TGT = { at: 0, project: null, data: null, inflight: null };

  function pickTarget(data) {
    var groups = ((data && data.groups) || []).filter(function (g) { return g && g.dfm; });
    var models = collectDfmNames();
    var want = String(state.dfm || "");
    var hit = null;
    groups.forEach(function (g) { if (!hit && String(g.dfm) === want) hit = g; });
    if (!hit && want && models.indexOf(want) >= 0) {
      return { dfm: want, group: "", label: want };
    }
    if (!hit) {
      groups.forEach(function (g) { if (!hit && !g.orphan) hit = g; });
      if (!hit) hit = groups[0] || null;
    }
    if (hit) return { dfm: String(hit.dfm), group: hit.id, label: hit.name || hit.dfm };
    var first = models[0] || "";
    return first ? { dfm: first, group: "", label: first } : null;
  }

  /* 只为了问「库里有哪几个组」：不渲染、不改状态；同一个请求在飞时共用。 */
  function libraryFor(project) {
    if (TGT.inflight && TGT.project === project) return TGT.inflight;
    TGT.project = project;
    TGT.inflight = call(API + "?project=" + encodeURIComponent(project),
                        { cache: "no-store" }, 15000).then(function (d) {
      TGT.data = d; TGT.at = Date.now(); TGT.inflight = null;
      return d;
    }, function (e) {
      TGT.inflight = null; TGT.data = null;
      throw e;
    });
    return TGT.inflight;
  }

  function addTarget(project) {
    project = String(project == null ? state.project : project);
    /* 页面开着时用手里这份真数据（用户刚点过组卡片 / 换过下拉） */
    if (state.open && state.loaded && state.data) {
      return Promise.resolve(pickTarget(state.data));
    }
    return libraryFor(project).then(pickTarget, function () { return pickTarget(null); });
  }

  window.FaceDB = {
    open: open,
    close: cancel,
    requestClose: askCancel,
    refresh: load,
    isOpen: function () { return state.open; },
    /* 合成器框上的「＋」用：告诉它这张新脸该落进哪个组（null = 先选模型） */
    addTarget: addTarget,
    selectedDfm: function () { return String(state.dfm || ""); },
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
