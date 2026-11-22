  /* ---------- 主页：根据真实数据推下一步 ---------- */
  function ago(ts) {
    if (!ts) return "";
    var d = Date.now() / 1000 - ts;
    if (d < 3600) return Math.max(1, Math.round(d / 60)) + " 分钟前";
    if (d < 86400) return Math.round(d / 3600) + " 小时前";
    return Math.round(d / 86400) + " 天前";
  }

  function greeting() {
    var hh = new Date().getHours();
    if (hh < 5) return "夜深了";
    if (hh < 11) return "早上好";
    if (hh < 13) return "中午好";
    if (hh < 18) return "下午好";
    if (hh < 23) return "晚上好";
    return "夜深了";
  }

  function renderHome(s, models) {
    var src = (s.datasets && s.datasets.src && s.datasets.src.count) || 0;
    var dst = (s.datasets && s.datasets.dst && s.datasets.dst.count) || 0;
    var list = models || [];
    var latest = list[0];
    var exported = list.filter(function (m) { return m.has_dfm; }).length;

    /* 下一步 */
    var label, desc, btn, act;
    if (!src && !dst) {
      label = "先导入一份人脸集";
      desc = "src 与 dst 都还是空的。提取器可以从视频或图片里裁出对齐后的人脸。";
      btn = "打开提取器"; act = function () { go("extract"); };
    } else if (!list.length) {
      label = "创建第一个模型";
      desc = "数据集已就绪（src " + fmt(src) + " / dst " + fmt(dst) + "），接下来建模型开始训练。";
      btn = "新建模型"; act = function () { go("train"); };
    } else {
      label = "继续训练 " + latest.name;
      desc = "停在 iter " + fmt(latest.iter) + "，上次训练于 " + (ago(latest.trained_at) || "未知") + "。";
      btn = "继续训练"; act = function () { openCfg(latest); };
    }
    if ($("hello")) $("hello").textContent = greeting() + (AUTH.logged_in && AUTH.name ? "，" + AUTH.name : "");
    var nb = $("nextBtn");
    if (nb) { nb.textContent = btn; nb.onclick = act; }


    /* 流水线 */
    if ($("pvData")) $("pvData").innerHTML = "<b>" + fmt(src) + "</b> src · <b>" + fmt(dst) + "</b> dst";
    if ($("pvModel")) $("pvModel").innerHTML = list.length
      ? "<b>" + list.length + "</b> 个模型 · 最新 " + latest.name + " · iter " + fmt(latest.iter)
      : "还没有模型";
    if ($("pvMerge")) $("pvMerge").innerHTML = exported
      ? "<b>" + exported + "</b> 个已导出 DFM"
      : "还没有导出过 DFM";

    /* 最近活动 */
    var box = $("homeRecent");
    if (box) {
      var rows = list.slice(0, 4);
      box.innerHTML = rows.length
        ? rows.map(function (m) {
            return '<div class="act"><div class="ai"><svg class="ic" style="width:12px;height:12px;opacity:1"><use href="#i-cpu"/></svg></div>' +
              '<div class="an">' + m.name + " · iter " + fmt(m.iter) + "</div>" +
              '<div class="at">' + ago(m.trained_at) + "</div></div>";
          }).join("")
        : '<div class="empty-note">还没有任何训练记录</div>';
    }
    if ($("homeVer")) {
      /* 首页右下角只放大版本：v4.1-1122 那串日期留在关于页的标签里 */
      var _hv = String(s.version || "");
      var _hm = _hv.match(/^(\d+\.\d+)-/);
      $("homeVer").textContent = "v" + (_hm ? _hm[1] : _hv.replace(/^v/, ""));
    }
    if (window.I18N) window.I18N.refresh();
  }

  function loadState() {
    fetch("/api/state").then(function (r) { return r.json(); }).then(function (s) {
      STATE = s;
      /* 预览模式：后端带 preview=true 时，协议门和登录校验都放行，
         好让设计工作台那种独立会话直接看到真实界面。 */
      if (s && s.preview) {
        window.__preview = true;
        window.__loggedIn = true;
        if (window.__legalSync) window.__legalSync();
      }
      var v = el("aboutVer");
      if (v) {
        /* 4.1-1122 / 4.1-1122-b2：大版本留在正文，日期做成蓝色标签挂在右边，
           和引擎编译页的状态标签一个样式（.tag.t-blue）。 */
        var _raw = String(s.version || "");
        var _m = _raw.match(/^(\d+\.\d+)-(\d{2})(\d{2})(?:-b(\d+))?/);
        if (_m) {
          v.textContent = "v" + _m[1];
          var _tag = el("aboutVerTag");
          if (_tag) {
            _tag.textContent = _m[2] + "-" + _m[3] + (_m[4] ? " · B" + _m[4] : "");
            _tag.classList.remove("hide");
            _tag.title = "构建日期 " + _m[2] + "-" + _m[3];
          }
        } else {
          v.textContent = _raw ? ("v" + _raw.replace(/^v/, "")) : "—";
        }
      }
      renderDatasets(s.datasets || { src: { count: 0, exists: false }, dst: { count: 0, exists: false } });
      loadModels();
      renderMerge(s.models || []);
      renderPorts(s.ports || {});
      syncDatasetBadge();
      renderRuntime(s);
      renderLastModel(s.models || []);
      if (window.__invalidateShotCache) window.__invalidateShotCache();
      if (window.I18N) window.I18N.refresh();
    }).catch(function (e) { console.warn("state load failed", e); });
  }
  window.loadState = loadState;
  window.refreshUI = function () { loadState(); };

  /* ---------- 下载进程：按实际任务渲染 ---------- */
  var BILI_JOBS = [];     // { bvid, title, state, speed }
  window.BILI_JOBS = BILI_JOBS;   // 调试 / 任务页共用
  function renderProcs() {
    var box = $("jobList");
    var badge = $("jobBadge");
    var summary = $("jobSummary");
    var running = BILI_JOBS.filter(function (j) { return j.state !== "done"; }).length;

    if (badge) {
      badge.textContent = running > 99 ? "99+" : String(running);
      badge.hidden = running === 0;
    }
    var summaryText = BILI_JOBS.length
      ? (running ? (running + " 个进行中 / 共 " + BILI_JOBS.length + " 个") : (BILI_JOBS.length + " 个已完成"))
      : "空闲";
    if (summary) summary.textContent = summaryText;
    var top = $("jobSummaryTop");
    if (top) top.textContent = summaryText;
    if (!box) return;
    box.innerHTML = "";
    if (!BILI_JOBS.length) {
      box.innerHTML = '<div class="empty-note" id="jobEmpty">当前没有任务</div>';
      return;
    }
    function row(j) {
      var d = document.createElement("div");
      d.className = "job" + (j.state === "done" ? " done" : (j.state === "failed" ? " failed" : ""));
      var tail;
      if (j.state === "done") {
        tail = '<span class="jobstate done">已完成</span>';
      } else if (j.state === "failed") {
        tail = '<span class="jobstate" style="color:#ef4444" title="' +
          String(j.error || "").replace(/"/g, "&quot;") + '">失败' +
          (j.error ? "：" + String(j.error).slice(0, 60) : "") + "</span>";
      } else if (j.state === "queued") {
        tail = '<span class="jobstate">排队中</span>';
      } else if (j.state === "cancelled") {
        tail = '<span class="jobstate">已取消</span>';
      } else {
        tail = '<span class="jobbar"><i style="width:' + Math.round(j.pct || 0) + '%"></i></span>' +
          '<span class="jobpct">' + Math.round(j.pct || 0) + "%</span>" +
          '<span class="jobstate">' + (j.speed || "0.0 MB/s") + "</span>";
      }
      d.innerHTML =
        '<i class="jobdot"></i>' +
        '<span class="jobname">' + (j.title || j.bvid || "任务") + "</span>" +
        '<span class="jobkind">' + (j.kind || "B 站下载") + "</span>" + tail;
      return d;
    }
    var runningList = BILI_JOBS.filter(function (j) { return j.state === "downloading"; });
    var queuedList = BILI_JOBS.filter(function (j) { return j.state === "queued"; });
    var doneList = BILI_JOBS.filter(function (j) { return j.state === "done"; });

    runningList.forEach(function (j) { box.appendChild(row(j)); });
    queuedList.slice(0, 5).forEach(function (j) { box.appendChild(row(j)); });
    if (queuedList.length > 5) {
      var more = document.createElement("div");
      more.className = "job more";
      more.textContent = "还有 " + (queuedList.length - 5) + " 个在排队";
      box.appendChild(more);
    }
    if (doneList.length) {
      var dsum = document.createElement("div");
      dsum.className = "job more done";
      dsum.textContent = "已完成 " + doneList.length + " 个";
      box.appendChild(dsum);
    }
  }
  window.renderProcs = renderProcs;

  /* ---------- 视频列表：内置全选 / 计数 / 下载 ---------- */
  var BILI_VIDEOS = [];   // { bvid, title, coop, checked }
  function biliSelCount() {
    return BILI_VIDEOS.filter(function (v) { return v.checked; }).length;
  }
  function syncBiliSel() {
    var n = biliSelCount();
    var sel = $("biliSel");
    if (sel) sel.textContent = String(n);
    var all = $("biliAll");
    if (all) {
      all.checked = BILI_VIDEOS.length > 0 && n === BILI_VIDEOS.length;
      all.indeterminate = n > 0 && n < BILI_VIDEOS.length;
    }
    var start = $("biliStart");
    if (start) {
      start.disabled = n === 0;
      start.textContent = n ? ("下载选中（" + n + "）") : "下载选中";
    }
  }
  function renderBili(list) {
    if (Array.isArray(list)) BILI_VIDEOS = list;
    var body = $("biliBody");
    var empty = $("biliEmpty");
    var cnt = $("biliCount");
    if (!body) return;
    if (cnt) cnt.textContent = BILI_VIDEOS.length + " 个";
    body.innerHTML = "";
    if (!BILI_VIDEOS.length) {
      if (empty) empty.style.display = "";
      syncBiliSel();
      return;
    }
    if (empty) empty.style.display = "none";
    BILI_VIDEOS.forEach(function (v, i) {
      var tr = document.createElement("tr");
      tr.innerHTML = '<td><input type="checkbox"' + (v.checked ? " checked" : "") + "></td>" +
        "<td>" + (biliEsc(v.title) || "—") + "</td>" +
        '<td class="mono">' + (biliEsc(v.bvid) || "—") + "</td>" +
        '<td class="mono">' + (biliEsc(v.length) || "—") + "</td>" +
        "<td>" + (v.coop ? "是" : "否") + "</td>";
      var cb = tr.querySelector("input");
      cb.addEventListener("change", function () {
        BILI_VIDEOS[i].checked = cb.checked;
        syncBiliSel();
      });
      body.appendChild(tr);
    });
    syncBiliSel();
  }
  window.renderBili = renderBili;

  /* ---------- 视频列表：走后端真实接口 /api/bili/up/videos ----------
     以前这里是个纯前端假列表：本地随机生成 60~160 条假 bvid / 假标题，
     还配了个定时器假装在分页，一条网络请求都不发。现在按页拉真数据。 */
  var BILI_PAGE = 30;
  var BILI_TIMER = null;
  var DL_TIMER = null;
  var DL_THREADS = 4;
  var BILI_QUALITY = "best";
  var BILI_PAUSED = false;

  function biliEsc(s) {
    return String(s == null ? "" : s).replace(/[&<>"']/g, function (c) {
      return { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c];
    });
  }

  /* FastAPI 的 HTTPException detail 可能是字符串，也可能是 {code,message} */
  function biliErrText(body, status) {
    var d = body;
    if (d && typeof d === "object" && d.detail !== undefined) d = d.detail;
    var msg = "";
    if (d && typeof d === "object") msg = d.message || d.msg || "";
    else if (typeof d === "string") msg = d;
    if (!msg) msg = "HTTP " + status;
    return msg;
  }

  function biliFetchList(uid) {
    if (BILI_TIMER) { clearTimeout(BILI_TIMER); BILI_TIMER = null; }
    var bar = $("biliFetchBar"), fill = $("biliFetchFill"), txt = $("biliFetchText");
    var btn = $("biliFetch"), owner = $("biliOwner");
    var page = 1, total = 0, mock = false;
    BILI_VIDEOS.length = 0;
    renderBili([]);
    if (bar) bar.hidden = false;
    if (btn) { btn.disabled = true; btn.textContent = "获取中…"; }
    if (owner) owner.textContent = "UID " + uid;

    function fail(msg) {
      if (fill) fill.style.width = "0%";
      if (txt) txt.textContent = "失败：" + msg;
      if (btn) { btn.disabled = false; btn.textContent = "获取视频列表"; }
      toast("获取视频列表失败", msg, "error");
    }

    function step() {
      fetch("/api/bili/up/videos", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ mid: uid, page: page, ps: BILI_PAGE })
      }).then(function (r) {
        return r.text().then(function (t) {
          var d = null;
          try { d = JSON.parse(t || "null"); } catch (e) { d = t; }
          return { ok: r.ok, status: r.status, body: d };
        });
      }).then(function (res) {
        var d = res.body || {};
        if (!res.ok) { fail(biliErrText(d, res.status)); return; }
        var items = (d && d.items) || [];
        mock = !!(d && d.mock);
        total = (d && d.total) || 0;
        items.forEach(function (v) {
          BILI_VIDEOS.push({
            bvid: v.bvid || "",
            title: v.title || "",
            length: v.length || "",
            coop: !!v.is_coop,
            checked: false
          });
        });
        renderBili();
        var pct = total ? Math.round(BILI_VIDEOS.length / total * 100) : 0;
        if (fill) fill.style.width = Math.min(100, pct) + "%";
        if (txt) {
          txt.textContent = "第 " + page + " 页 · 已取 " + BILI_VIDEOS.length +
            (total ? " / " + total : "") + (mock ? "（占位数据）" : "");
        }
        if (d && d.has_more && items.length) {
          page += 1;
          BILI_TIMER = setTimeout(step, 400);
        } else {
          BILI_TIMER = null;
          if (bar) setTimeout(function () { bar.hidden = true; }, 900);
          if (btn) { btn.disabled = false; btn.textContent = "获取视频列表"; }
          if (fill) fill.style.width = "0%";
          toast(mock ? "获取完成（占位数据）" : "获取完成",
                "UID " + uid + " · 共 " + BILI_VIDEOS.length + " 个视频", "success");
        }
      }).catch(function (e) {
        fail("请求失败：" + e);
      });
    }
    step();
  }
  window.biliFetchList = biliFetchList;

  /* ---------- 下载：真队列，进度来自 /api/bili/batch/status ---------- */
  function biliConcurrency() { return DL_THREADS; }
  function biliSetThreads(n) { DL_THREADS = n; }
  window.biliSetThreads = biliSetThreads;
  function biliSetQuality(q) { BILI_QUALITY = q || "best"; }
  window.biliSetQuality = biliSetQuality;

  function biliStopPoll() {
    if (DL_TIMER) { clearInterval(DL_TIMER); DL_TIMER = null; }
  }

  function biliApplyStatus(s) {
    var items = (s && s.items) || [];
    BILI_JOBS.length = 0;
    items.forEach(function (it) {
      var v = BILI_VIDEOS.filter(function (x) { return x.bvid === it.bvid; })[0] || {};
      BILI_JOBS.push({
        bvid: it.bvid,
        title: v.title || it.bvid,
        state: it.state,
        pct: it.pct || 0,
        speed: it.speed ? (Number(it.speed).toFixed(1) + " MB/s") : "",
        error: it.error || "",
        path: it.path || ""
      });
    });
    renderProcs();
    var pauseBtn = $("biliPause");
    if (pauseBtn) pauseBtn.textContent = (s && s.paused) ? "继续" : "暂停";
    if (s && !s.running) {
      biliStopPoll();
      var failed = items.filter(function (i) { return i.state === "failed"; }).length;
      if (failed) {
        toast("下载结束，但有失败", failed + " 个失败：见任务列表里的错误信息", "error");
      } else if (items.length) {
        toast("下载完成", "全部 " + items.length + " 个视频已下载", "success");
      }
    }
  }

  function biliPollStatus() {
    fetch("/api/bili/batch/status").then(function (r) { return r.json(); })
      .then(biliApplyStatus).catch(function () { });
  }

  function biliStartDownload(picked) {
    if (!picked || !picked.length) return;
    var bvids = picked.map(function (v) { return v.bvid; });
    toast("开始下载", "已选 " + bvids.length + " 个视频", "info");
    fetch("/api/bili/batch", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ bvids: bvids, workers: biliConcurrency(), quality: BILI_QUALITY })
    }).then(function (r) {
      return r.text().then(function (t) {
        var d = null;
        try { d = JSON.parse(t || "null"); } catch (e) { d = t; }
        return { ok: r.ok, status: r.status, body: d };
      });
    }).then(function (res) {
      if (!res.ok) { toast("下载启动失败", biliErrText(res.body, res.status), "error"); return; }
      BILI_JOBS.length = 0;
      biliPollStatus();
      biliStopPoll();
      DL_TIMER = setInterval(biliPollStatus, 800);
    }).catch(function (e) {
      toast("下载启动失败", String(e), "error");
    });
  }
  window.biliStartDownload = biliStartDownload;

  function biliTogglePause() {
    BILI_PAUSED = !BILI_PAUSED;
    fetch("/api/bili/batch/pause", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ paused: BILI_PAUSED })
    }).then(function (r) { return r.json(); }).then(biliApplyStatus).catch(function () { });
    return BILI_PAUSED;
  }
  window.biliTogglePause = biliTogglePause;

  function biliStopAll() {
    BILI_PAUSED = false;
    fetch("/api/bili/batch/cancel", { method: "POST" })
      .then(function (r) { return r.json(); })
      .then(biliApplyStatus)
      .catch(function () { biliStopPoll(); });
  }
  window.biliStopAll = biliStopAll;

  (function bindBili() {
    var pauseBtn = $("biliPause");
    if (pauseBtn && !pauseBtn.dataset.bound) {
      pauseBtn.dataset.bound = "1";
      pauseBtn.addEventListener("click", function () {
        var paused = window.biliTogglePause ? window.biliTogglePause() : false;
        pauseBtn.textContent = paused ? "继续" : "暂停";
      });
    }
    var stopBtn = $("biliStop");
    if (stopBtn && !stopBtn.dataset.bound) {
      stopBtn.dataset.bound = "1";
      stopBtn.addEventListener("click", function () {
        confirmBox("终止下载", "会清空当前队列，已下载的文件保留。", "确定", function () {
          if (window.biliStopAll) window.biliStopAll();
          if (pauseBtn) pauseBtn.textContent = "暂停";
          go("tool-bili");
        });
      });
    }
    var qualSeg = $("biliQuality");
    if (qualSeg && !qualSeg.dataset.bound) {
      qualSeg.dataset.bound = "1";
      qualSeg.addEventListener("click", function (e) {
        var b = e.target.closest("b");
        if (!b) return;
        Array.prototype.forEach.call(qualSeg.children, function (x) { x.classList.remove("on"); });
        b.classList.add("on");
        var t = b.textContent.trim();
        if (window.biliSetQuality) window.biliSetQuality(/最高/.test(t) ? "best" : (/1080/.test(t) ? "1080" : "720"));
      });
    }
    var threadSeg = $("biliThreads");
    if (threadSeg && !threadSeg.dataset.bound) {
      threadSeg.dataset.bound = "1";
      threadSeg.addEventListener("click", function (e) {
        var b = e.target.closest("b");
        if (!b) return;
        Array.prototype.forEach.call(threadSeg.children, function (x) { x.classList.remove("on"); });
        b.classList.add("on");
        if (window.biliSetThreads) window.biliSetThreads(parseInt(b.textContent, 10) || 4);
      });
    }

    var fetchBtn = $("biliFetch");
    if (fetchBtn && !fetchBtn.dataset.bound) {
      fetchBtn.dataset.bound = "1";
      fetchBtn.addEventListener("click", function () {
        var uid = ($("biliUid") || {}).value || "";
        uid = String(uid).trim();
        if (!/^\d+$/.test(uid)) {
          toast("UID 不合法", "请输入纯数字的 UP 主 UID", "error");
          return;
        }
        biliFetchList(uid);
      });
    }
    var all = $("biliAll");
    if (all && !all.dataset.bound) {
      all.dataset.bound = "1";
      all.addEventListener("change", function () {
        BILI_VIDEOS.forEach(function (v) { v.checked = all.checked; });
        renderBili();
      });
    }
    var start = $("biliStart");
    if (start && !start.dataset.bound) {
      start.dataset.bound = "1";
      start.addEventListener("click", function () {
        var picked = BILI_VIDEOS.filter(function (v) { return v.checked; });
        if (!picked.length) return;
        biliStartDownload(picked);
        go("bili-jobs");
      });
    }
  })();

  /* ---------- 角度可视化：坐标轴 + 散点 / 格子图 ---------- */
  var ANGLE_PTS = null;
  function anglePoints() {
    if (ANGLE_PTS) return ANGLE_PTS;
    var pts = [];
    var rnd = (function (s) { return function () { s = (s * 1103515245 + 12345) & 0x7fffffff; return s / 0x7fffffff; }; })(42);
    function gauss(mu, sd) {
      var u = 1 - rnd(), v = rnd();
      return mu + sd * Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
    }
    for (var i = 0; i < 900; i++) {
      var yaw = gauss(0, 22);
      var pitch = gauss(2, 14);
      if (yaw < -90 || yaw > 90 || pitch < -90 || pitch > 90) continue;
      pts.push({ y: yaw, p: pitch });
    }
    ANGLE_PTS = pts;
    return pts;
  }

  var ANGLE_MODE = "scatter";

  function drawAngle() {
    var box = $("angleBox");
    if (!box) return;
    var pts = anglePoints();
    var W = 1160, H = 420, L = 46, R = 18, T = 16, Bm = 36;
    var pw = W - L - R, ph = H - T - Bm;
    function X(v) { return L + ((v + 90) / 180) * pw; }
    function Y(v) { return T + ph - ((v + 90) / 180) * ph; }

    var s = '<svg viewBox="0 0 ' + W + " " + H + '" width="100%" height="100%" preserveAspectRatio="xMidYMid meet">';

    /* 网格 + 刻度 */
    [-90, -60, -30, 0, 30, 60, 90].forEach(function (t) {
      s += '<line x1="' + X(t) + '" y1="' + T + '" x2="' + X(t) + '" y2="' + (T + ph) + '" stroke="var(--line)" stroke-width="1"/>';
      s += '<line x1="' + L + '" y1="' + Y(t) + '" x2="' + (L + pw) + '" y2="' + Y(t) + '" stroke="var(--line)" stroke-width="1"/>';
      s += '<text x="' + X(t) + '" y="' + (T + ph + 16) + '" fill="var(--ink-3)" font-size="10" text-anchor="middle">' + t + "°</text>";
      s += '<text x="' + (L - 8) + '" y="' + (Y(t) + 3) + '" fill="var(--ink-3)" font-size="10" text-anchor="end">' + t + "</text>";
    });
    /* 坐标轴 */
    s += '<line x1="' + L + '" y1="' + (T + ph) + '" x2="' + (L + pw) + '" y2="' + (T + ph) + '" stroke="var(--line-strong)" stroke-width="1"/>';
    s += '<line x1="' + L + '" y1="' + T + '" x2="' + L + '" y2="' + (T + ph) + '" stroke="var(--line-strong)" stroke-width="1"/>';
    s += '<text x="' + (L + pw) + '" y="' + (H - 6) + '" fill="var(--ink-3)" font-size="10" text-anchor="end">偏航 yaw</text>';
    s += '<text x="6" y="' + (T + 10) + '" fill="var(--ink-3)" font-size="10">俯仰</text>';

    if (ANGLE_MODE === "scatter") {
      pts.forEach(function (pt) {
        s += '<circle cx="' + X(pt.y).toFixed(1) + '" cy="' + Y(pt.p).toFixed(1) + '" r="2.2" fill="var(--ink-2)" opacity="0.45"/>';
      });
    } else {
      var N = 24, cellW = pw / N, cellH = ph / N;
      var grid = [], max = 0;
      for (var i = 0; i < N; i++) { grid.push(new Array(N).fill(0)); }
      pts.forEach(function (pt) {
        var ix = Math.min(N - 1, Math.floor(((pt.y + 90) / 180) * N));
        var iy = Math.min(N - 1, Math.floor(((pt.p + 90) / 180) * N));
        grid[iy][ix]++;
        if (grid[iy][ix] > max) max = grid[iy][ix];
      });
      for (var yy = 0; yy < N; yy++) {
        for (var xx = 0; xx < N; xx++) {
          if (!grid[yy][xx]) continue;
          var a = 0.12 + 0.85 * (grid[yy][xx] / max);
          s += '<rect x="' + (L + xx * cellW).toFixed(1) + '" y="' + (T + ph - (yy + 1) * cellH).toFixed(1) +
            '" width="' + (cellW - 1).toFixed(1) + '" height="' + (cellH - 1).toFixed(1) +
            '" fill="var(--ink-2)" opacity="' + a.toFixed(2) + '"/>';
        }
      }
    }

    s += "</svg>";
    box.innerHTML = s;

    var hint = document.querySelector("#p-tool-angle .cardhd .hint");
    if (hint) hint.textContent = pts.length + " 张 · " + (ANGLE_MODE === "scatter" ? "散点图" : "格子图");
  }
  window.drawAngle = drawAngle;
