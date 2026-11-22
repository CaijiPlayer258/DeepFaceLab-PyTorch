/* 工具 · 视频切帧：ffmpeg 把视频导出成帧序列。
 *
 * 后端在 app/server/backend_tools/frames.py：单任务、-progress 管道读进度、
 * 输出 image2 的 %06d.jpg（和 DFL 的帧序列约定一致）。
 *
 * 这一页只做三件事：收参数、开任务、画进度。参数记在 localStorage 里，
 * 下次进来接着用 —— 和 B 站下载 / 色域迁移同一套习惯。
 */
(function () {
  var KEY = "dfl-frames-opts";
  var TIMER = 0;

  function $id(s) { return document.getElementById(s); }

  function opts() {
    var o = { fmt: "jpg", q: 2, from: "", to: "", dst: "" };
    try {
      var raw = JSON.parse(localStorage.getItem(KEY) || "{}");
      for (var k in o) if (raw[k] != null) o[k] = raw[k];
    } catch (e) {}
    return o;
  }
  function save(o) {
    try { localStorage.setItem(KEY, JSON.stringify(o)); } catch (e) {}
  }

  function bind() {
    var dst = $id("frDst");
    if (dst) dst.value = opts().dst || "";

    var fmtSeg = $id("frFmtSeg");
    if (fmtSeg && !fmtSeg.dataset.bound) {
      fmtSeg.dataset.bound = "1";
      fmtSeg.addEventListener("click", function (e) {
        var b = e.target.closest("b");
        if (!b) return;
        fmtSeg.querySelectorAll("b").forEach(function (x) { x.classList.remove("on"); });
        b.classList.add("on");
        var o = opts(); o.fmt = b.getAttribute("data-v"); save(o);
      });
    }

    /* 源视频：选文件（不是目录），走 desktop.pickFile */
    var pick = $id("frSrcPick");
    if (pick && !pick.dataset.bound) {
      pick.dataset.bound = "1";
      pick.addEventListener("click", function (e) {
        e.stopPropagation();
        var api = window.desktop;
        if (!api || !api.pickFile) {
          if (window.toast) window.toast("选择视频", "网页版拿不到本地路径，请直接填写", "info");
          return;
        }
        api.pickFile("选择视频", [
          { name: "视频", extensions: ["mp4", "mkv", "mov", "avi", "wmv", "flv", "webm", "m4v", "ts", "mpg", "mpeg"] },
          { name: "全部", extensions: ["*"] }
        ]).then(function (p) {
          if (!p) return;
          var inp = $id("frSrc");
          if (inp) { inp.value = p; inp.dispatchEvent(new Event("change", { bubbles: true })); }
        }).catch(function (err) {
          if (window.toast) window.toast("打开失败", String((err && err.message) || err), "error");
        });
      });
    }

    /* 输出目录：目录选择器 */
    var dstBox = dst && dst.closest(".pathinp");
    if (dstBox && !dstBox.dataset.pick) {
      dstBox.dataset.pick = "1";
      var pill = dstBox.querySelector(".iconpill");
      if (pill) pill.addEventListener("click", function (e) {
        e.stopPropagation();
        if (window.pickFolderFor) window.pickFolderFor(dstBox);
      });
    }

    ["frQ", "frFrom", "frTo"].forEach(function (id) {
      var e = $id(id);
      if (!e || e.dataset.bound) return;
      e.dataset.bound = "1";
      e.addEventListener("change", function () {
        var o = opts();
        if (id === "frQ") o.q = parseInt(e.value, 10) || 2;
        if (id === "frFrom") o.from = e.value;
        if (id === "frTo") o.to = e.value;
        save(o);
      });
    });

    var btn = $id("btnFrames");
    if (btn && !btn.dataset.bound) { btn.dataset.bound = "1"; btn.addEventListener("click", start); }
    var stop = $id("frStop");
    if (stop && !stop.dataset.bound) {
      stop.dataset.bound = "1";
      stop.addEventListener("click", function () {
        stop.disabled = true;
        fetch("/api/tools/frames/stop", { method: "POST" }).catch(function () {}).then(function () {
          stop.disabled = false;
          poll();
        });
      });
    }

    /* 首次进入：恢复上次的参数 */
    var o = opts();
    if ($id("frQ")) $id("frQ").value = o.q;
    if ($id("frFrom")) $id("frFrom").value = o.from || "";
    if ($id("frTo")) $id("frTo").value = o.to || "";
    if (fmtSeg) [].forEach.call(fmtSeg.children, function (b) {
      b.classList.toggle("on", b.getAttribute("data-v") === o.fmt);
    });
    poll();
  }

  function start() {
    var src = ($id("frSrc") && $id("frSrc").value || "").trim();
    if (!src) { if (window.toast) window.toast("还没选视频", "先选一个要切帧的视频文件", "warn"); return; }
    var o = opts();
    var body = {
      src: src,
      dst: ($id("frDst") && $id("frDst").value || "").trim(),
      mode: "all", fmt: o.fmt, quality: o.q,
      from: parseFloat(($id("frFrom") && $id("frFrom").value) || 0) || 0,
      to: parseFloat(($id("frTo") && $id("frTo").value) || 0) || 0
    };
    o.dst = body.dst; save(o);

    var btn = $id("btnFrames");
    if (btn) { btn.disabled = true; btn.textContent = "启动中…"; }
    fetch("/api/tools/frames", {
      method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body)
    }).then(function (r) { return r.json().then(function (d) { return { ok: r.ok, d: d }; }); })
      .then(function (res) {
        if (!res.ok) throw new Error((res.d && (res.d.detail || res.d.error)) || "启动失败");
        if ($id("frProg")) $id("frProg").classList.remove("hide");
        if (window.toast) window.toast("开始切帧", "输出到 " + (res.d.dst || ""), "success");
        poll();
      })
      .catch(function (e) {
        if (window.toast) window.toast("启动失败", String((e && e.message) || e), "error");
      })
      .then(function () { if (btn) { btn.disabled = false; btn.textContent = "开始切帧"; } });
  }

  function setTxt(id, v) { var e = $id(id); if (e) e.textContent = v; }

  function paint(d) {
    if (!d) return;
    var running = !!d.running;
    var card = $id("frProg");
    if (card && (running || d.done || d.error)) card.classList.remove("hide");
    var pct = Math.max(0, Math.min(100, d.pct || 0));
    if ($id("frBar")) $id("frBar").style.width = pct.toFixed(1) + "%";
    setTxt("frPctTxt", (d.pct || 0).toFixed(0) + "%");
    setTxt("frDone", d.done || 0);
    setTxt("frTotal", "/ " + (d.total || 0));
    setTxt("frFps", d.fps || 0);
    setTxt("frElapsed", Math.round(d.elapsed || 0));
    setTxt("frEta", d.eta || "—");
    setTxt("frSpeed", d.speed ? ("×" + d.speed) : "");
    setTxt("frTitle", running ? "正在切帧" : (d.error ? "切帧失败" : "切帧完成"));
    setTxt("frMsg", d.error || d.msg || "");
    if ($id("frStop")) $id("frStop").classList.toggle("hide", !running);
  }

  function poll() {
    if (TIMER) { clearTimeout(TIMER); TIMER = 0; }
    fetch("/api/tools/frames/status").then(function (r) { return r.json(); }).then(function (d) {
      paint(d);
      if (d && d.running) TIMER = setTimeout(poll, 500);
      else if (d && d.msg) {
        if (d.error && window.toast) window.toast("切帧失败", d.error, "error");
        else if (d.pct >= 100) window.toast && window.toast("切帧完成", "共 " + d.done + " 帧", "success");
      }
    }).catch(function () { TIMER = setTimeout(poll, 1500); });
  }

  window.framesBind = bind;
  if (document.readyState !== "loading") {
    if ($id("btnFrames")) bind();
  } else {
    document.addEventListener("DOMContentLoaded", function () { if ($id("btnFrames")) bind(); });
  }
})();
