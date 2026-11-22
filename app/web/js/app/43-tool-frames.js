/* 工具 · 视频切帧：ffmpeg 把视频原样拆成帧序列（不抽帧、不补帧）。
 *
 * 后端 app/server/backend_tools/frames.py：单任务、-progress 管道读进度、
 * 输出 image2 的 %06d.jpg（和 DFL 的帧序列约定一致）。
 *
 * 界面分两级，和人脸提取一样：
 *   #p-tool-frames      参数（源视频 / 输出 / 格式 / 质量 / 时间段）
 *   #p-tool-frames-run  进度（速度曲线 + 四格统计 + 中断）
 */
(function () {
  var KEY = "dfl-frames-opts";
  var TIMER = 0;
  var HIST = [{ pct: 0, spd: 0 }];   /* [{ pct, spd }] 画速度曲线（起点先占一位，保证能画出线段） */

  function $id(s) { return document.getElementById(s); }
  function setTxt(id, v) { var e = $id(id); if (e) e.textContent = v; }

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

  /* ---------------- 配置页 ---------------- */

  function bind() {
    var o = opts();
    if ($id("frDst")) $id("frDst").value = o.dst || "";

    var fmtSeg = $id("frFmtSeg");
    if (fmtSeg && !fmtSeg.dataset.bound) {
      fmtSeg.dataset.bound = "1";
      fmtSeg.addEventListener("click", function (e) {
        var b = e.target.closest("b");
        if (!b) return;
        fmtSeg.querySelectorAll("b").forEach(function (x) { x.classList.remove("on"); });
        b.classList.add("on");
        var t = opts(); t.fmt = b.getAttribute("data-v"); save(t);
      });
    }
    if (fmtSeg) [].forEach.call(fmtSeg.children, function (b) {
      b.classList.toggle("on", b.getAttribute("data-v") === o.fmt);
    });

    /* 源视频：选文件（desktop.pickFile），网页版退回手填 */
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
    var dst = $id("frDst");
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
        var t = opts();
        if (id === "frQ") t.q = parseInt(e.value, 10) || 2;
        if (id === "frFrom") t.from = e.value;
        if (id === "frTo") t.to = e.value;
        save(t);
      });
    });
    if ($id("frQ")) $id("frQ").value = o.q;
    if ($id("frFrom")) $id("frFrom").value = o.from || "";
    if ($id("frTo")) $id("frTo").value = o.to || "";

    var btn = $id("btnFrames");
    if (btn && !btn.dataset.bound) { btn.dataset.bound = "1"; btn.addEventListener("click", start); }

    /* 已经有任务在跑：直接进运行页，别让用户以为点不动 */
    fetch("/api/tools/frames/status").then(function (r) { return r.json(); }).then(function (d) {
      if (d && d.running && window.go) window.go("tool-frames-run");
      else paintIdle(d);
    }).catch(function () {});
  }

  function paintIdle(d) {
    if (!d) return;
    if (d.done || d.error) {
      /* 上次跑过：进运行页看结果，配置页保持干净 */
      if ($id("frRunCard")) $id("frRunCard").classList.remove("hide");
    }
  }

  /* ---------------- 启动 ---------------- */

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
        HIST = [];
        if ($id("frRunCard")) $id("frRunCard").classList.remove("hide");
        if (window.go) window.go("tool-frames-run");
        poll();
      })
      .catch(function (e) {
        if (window.toast) window.toast("启动失败", String((e && e.message) || e), "error");
      })
      .then(function () { if (btn) { btn.disabled = false; btn.textContent = "开始切帧"; } });
  }

  /* ---------------- 运行页 ---------------- */

  function drawGraph() {
    var cv = $id("frGraph");
    if (!cv || !cv.getContext) return;
    cv.style.display = HIST.length >= 1 ? "block" : "none";
    var dpr = window.devicePixelRatio || 1;
    var W = cv.clientWidth || 640, H = cv.clientHeight || 170;
    if (cv.width !== Math.round(W * dpr) || cv.height !== Math.round(H * dpr)) {
      cv.width = Math.round(W * dpr);
      cv.height = Math.round(H * dpr);
    }
    var g = cv.getContext("2d");
    g.setTransform(dpr, 0, 0, dpr, 0, 0);
    g.clearRect(0, 0, W, H);

    var cs = getComputedStyle(document.documentElement);
    var cInk3 = (cs.getPropertyValue("--ink-3") || "#8b8b98").trim();
    var cLine = (cs.getPropertyValue("--line") || "rgba(128,128,128,.2)").trim();
    var cBrand = (cs.getPropertyValue("--brand") || "#5d6ccf").trim();
    var cFont = (cs.getPropertyValue("--font") || "system-ui").trim();

    var padL = 54, padR = 14, padT = 12, padB = 22;
    var plotW = W - padL - padR, plotH = H - padT - padB;

    var peak = 1;
    HIST.forEach(function (p) { if (p.spd > peak) peak = p.spd; });
    var maxSpd = Math.max(10, Math.ceil(peak * 1.2 / 10) * 10);

    g.font = "10px " + cFont;
    g.textAlign = "right";
    g.textBaseline = "middle";
    [0, 0.5, 1].forEach(function (k) {
      var y = padT + plotH * (1 - k);
      g.strokeStyle = cLine;
      g.lineWidth = 1;
      g.beginPath(); g.moveTo(padL, y + 0.5); g.lineTo(W - padR, y + 0.5); g.stroke();
      g.fillStyle = cInk3;
      g.fillText(String(Math.round(maxSpd * k)), padL - 10, y);
    });
    g.textAlign = "center";
    g.textBaseline = "top";
    [0, 50, 100].forEach(function (pct) {
      var x = padL + plotW * pct / 100;
      g.strokeStyle = cLine;
      g.beginPath(); g.moveTo(x + 0.5, padT); g.lineTo(x + 0.5, padT + plotH); g.stroke();
      g.fillStyle = cInk3;
      g.fillText(pct + "%", x, padT + plotH + 5);
    });

    if (HIST.length < 2) return;
    g.strokeStyle = cBrand;
    g.lineWidth = 1.6;
    g.beginPath();
    HIST.forEach(function (p, i) {
      var x = padL + plotW * Math.min(100, Math.max(0, p.pct)) / 100;
      var y = padT + plotH * (1 - Math.min(1, p.spd / maxSpd));
      if (i === 0) g.moveTo(x, y); else g.lineTo(x, y);
    });
    g.stroke();
  }

  function paint(d) {
    if (!d) return;
    var running = !!d.running;

    /* 按钮状态先定：后面的画图万一抛异常，也不能让「中断」在完成之后还亮着 */
    var stop = $id("frStop"), back = $id("frBack");
    if (stop) {
      stop.hidden = !running;
      stop.style.setProperty("display", running ? "inline-flex" : "none", "important");
    }
    if (back) {
      back.hidden = running;
      back.style.setProperty("display", running ? "none" : "inline-flex", "important");
    }

    var card = $id("frRunCard");
    if (card && (running || d.done || d.error)) card.classList.remove("hide");

    setTxt("frDone", d.done || 0);
    setTxt("frTotal", "/ " + (d.total || 0));
    setTxt("frFps", d.fps || 0);
    setTxt("frElapsed", Math.round(d.elapsed || 0));
    setTxt("frEta", d.eta || "—");
    setTxt("frPctTxt", (d.pct || 0).toFixed(0) + "%" + (d.speed ? (" · ×" + d.speed) : ""));
    setTxt("frTitle", running ? "正在切帧" : (d.error ? "切帧失败" : "切帧完成"));
    setTxt("frMsg", d.error || d.msg || "");

    if (running) {
      var last = HIST[HIST.length - 1];
      if (!last || d.pct > last.pct + 0.3 || (d.fps || 0) !== last.spd) {
        HIST.push({ pct: d.pct || 0, spd: d.fps || 0 });
        if (HIST.length > 400) HIST.shift();
      }
    } else if (d.done) {
      /* 跑完把终点补上，曲线才走到 100% */
      var tail = HIST[HIST.length - 1];
      if (!tail || tail.pct < 99) HIST.push({ pct: 100, spd: d.fps || (tail ? tail.spd : 0) });
    }
    try { drawGraph(); } catch (e) { /* 画不出曲线不影响数字 */ }
  }
  function poll() {
    if (TIMER) { clearTimeout(TIMER); TIMER = 0; }
    fetch("/api/tools/frames/status").then(function (r) { return r.json(); }).then(function (d) {
      paint(d);
      if (d && d.running) TIMER = setTimeout(poll, 500);
      else if (d && (d.done || d.error)) {
        if (d.error) { if (window.toast) window.toast("切帧失败", d.error, "error"); }
        else if (window.toast) window.toast("切帧完成", "共 " + d.done + " 帧", "success");
      }
    }).catch(function () { TIMER = setTimeout(poll, 1500); });
  }

  function bindRun() {
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
    var back = $id("frBack");
    if (back && !back.dataset.bound) {
      back.dataset.bound = "1";
      back.addEventListener("click", function () { if (window.go) window.go("tool-frames"); });
    }
    /* 进运行页时如果任务已经在跑（或刚跑完），立刻画一帧别等 500ms */
    fetch("/api/tools/frames/status").then(function (r) { return r.json(); }).then(function (d) {
      paint(d);
      if (d && d.running) poll();
    }).catch(function () {});
  }


  /* 页面是启动时按 location.hash 打开的，那时本分片还没执行 ——
     再补一次进入，否则运行页永远停在初始状态（卡片 hidden、两个按钮都不受控）。 */
  function autoEnter() {
    var h = (location.hash || "").replace(/^#/, "");
    if (h === "tool-frames-run") bindRun();
    else if (h === "tool-frames") bind();
    else if ($id("btnFrames")) bind();
  }
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", autoEnter);
  else autoEnter();
  window.framesBind = bind;
  window.framesRunEnter = bindRun;
})();
