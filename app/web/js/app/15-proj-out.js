    /* ============================================================
       导出项目（标准 DFL 项目模式）
       ------------------------------------------------------------
       <项目>/000000.jpg + aligned/ → 逐帧合成 → <项目>/merged/000000.jpg
       → 按原 fps/音轨编码成 <项目>.mp4。和上面那个「导出」（4.0 MergeStudio
       一口气出 mp4）并列：断点续跑、失败帧可追、产物可一张张复查。
       ============================================================ */
    var MGPJ_POLL = null;
    window.__mgpjExporting = false;

    function mgpj(id) { return document.getElementById(id); }

    function mgpjSetList(boxId, listId, items, render) {
      var box = mgpj(boxId), list = mgpj(listId);
      if (!box || !list) return;
      list.innerHTML = "";
      var n = (items || []).length;
      box.hidden = n === 0;
      for (var i = 0; i < n; i++) {
        var li = document.createElement("li");
        var r = render ? render(items[i]) : { b: "", t: String(items[i]) };
        if (r.b) {
          var b = document.createElement("b");
          b.textContent = r.b;
          li.appendChild(b);
        }
        var sp = document.createElement("span");
        sp.textContent = r.t;
        li.appendChild(sp);
        list.appendChild(li);
      }
    }

    function mgpjRender(d) {
      var total = Number(d.total) || 0;
      var done = Number(d.done) || 0;
      var p = Math.max(0, Math.min(100, Number(d.pct) || 0));
      var fill = mgpj("mgpjFill"), pctEl = mgpj("mgpjPct");
      var fr = mgpj("mgpjFrames"), sp = mgpj("mgpjSpeed"), eta = mgpj("mgpjEta");
      var fileEl = mgpj("mgpjFile"), outEl = mgpj("mgpjOut"), hint = mgpj("mgpjHint");
      if (fill) fill.style.width = p + "%";
      if (pctEl) pctEl.textContent = p + "%";
      if (fr) fr.textContent = total ? ("已处理 " + done + " / " + total + " 帧")
                                    : (d.msg || "准备中…");
      var rate = Number(d.frame_fps) || 0;
      if (sp) sp.textContent = rate > 0 ? (rate.toFixed(1) + " 帧/秒") : ("— 帧/秒");
      if (eta) {
        var left = Number(d.eta_s);
        eta.textContent = d.running
          ? (left > 0 ? ("剩余 " + left + " 秒") : "剩余 —")
          : (d.phase === "done" ? "完成" : "已停");
      }
      if (fileEl) {
        var proj = String(d.project || "");
        fileEl.textContent = proj ? (proj.split(/[\\/]/).pop() + " · merged/") : "—";
      }
      if (outEl && d.out_video) outEl.textContent = String(d.out_video);
      if (hint) {
        var bits = [];
        if (d.msg) bits.push(String(d.msg));
        if (Number(d.swapped)) bits.push("换脸 " + d.swapped + " 帧");
        if (Number(d.copied)) bits.push("原样 " + d.copied + " 帧");
        if (Number(d.skipped)) bits.push("续跑跳过 " + d.skipped + " 帧");
        if (d.encode && d.encode.seconds) bits.push("编码 " + d.encode.seconds + " 秒");
        hint.textContent = bits.join(" · ");
      }
      mgpjSetList("mgpjWarnBox", "mgpjWarn", d.warnings, null);
      mgpjSetList("mgpjFailBox", "mgpjFailed", d.failed_frames, function (f) {
        return { b: String((f && f.frame) || "?"), t: String((f && f.error) || "失败") };
      });
      var cnt = mgpj("mgpjFailCount");
      if (cnt) cnt.textContent = String((d.failed_frames || []).length);
    }

    function mgpjStopPoll() {
      if (MGPJ_POLL) { clearInterval(MGPJ_POLL); MGPJ_POLL = null; }
    }

    function mgpjTerminal(d) {
      mgpjStopPoll();
      window.__mgpjExporting = false;
      var okAll = d.phase === "done";
      var cancelled = d.phase === "cancelled";
      var openBtn = mgpj("mgpjOpen");
      if (openBtn) openBtn.disabled = !(okAll && d.out_video);
      var cb = mgpj("mgpjCancel");
      if (cb) { cb.disabled = false; cb.textContent = "返回合成器"; }
      var eta = mgpj("mgpjEta");
      if (eta) eta.textContent = okAll ? "完成" : (cancelled ? "已取消" : "失败");
      if (okAll) {
        toast("导出项目完成", String(d.out_video || ""), "success");
      } else if (cancelled) {
        toast("已取消导出项目", "merged/ 里已经写好的帧会留着，下次接着跑", "info");
      } else {
        toast("导出项目失败", String(d.error || "").slice(0, 120), "error");
      }
    }

    function mgpjPoll() {
      mgpjStopPoll();
      MGPJ_POLL = setInterval(function () {
        fetch("/api/merge/export-project/status").then(function (r) { return r.json(); })
          .then(function (d) {
            mgpjRender(d);
            if (!d.running) mgpjTerminal(d);
          }).catch(function () {});
      }, 1000);
    }

    function mgpjStart(projectDir) {
      if (window.__mgpjExporting) { toast("正在导出项目", "上一次还没结束", "info"); return; }
      projectDir = String(projectDir || window.__mergeInputDir || "");
      if (!projectDir) {
        mgToastErr("pj-nosrc", "还没有项目目录",
                   "当前是视频模式：视频还没抽帧 + 对齐成项目，导出项目要项目目录", "warn", 6000);
        return;
      }
      var p = (window.MergePages && window.MergePages.settings) ? window.MergePages.settings() : {};
      var body = {
        project_dir: projectDir,
        models: mgFaceModelsPayload(),
        model: (document.querySelector("#mgModels .mgitem.on .nm") || {}).textContent || "",
        thresh: 0.5,
        settings: mgMergeSettings(p)
      };
      /* 检测器：p.detector 有就直接用（调用方点名的），否则读 #mgDetector；都没有就不发。 */
      var detP = window.mgDetectorBody ? window.mgDetectorBody(p) : "";
      if (detP) body.detector = detP;
      /* 输出路径：填的是 .mp4 就当文件名；填的是目录就放 <项目名>_merged.mp4 */
      var outRaw = String(p.output_path || "").trim();
      if (/^[A-Za-z]:[\\/]/.test(outRaw)) {
        body.out_video = /\.mp4$/i.test(outRaw)
          ? outRaw
          : (outRaw.replace(/[\\/]+$/, "") + "\\" +
             String(projectDir).split(/[\\/]/).pop() + "_merged.mp4");
      }

      window.__mgpjExporting = true;
      var cb0 = mgpj("mgpjCancel");
      if (cb0) { cb0.disabled = false; cb0.textContent = "取消导出"; }
      var open0 = mgpj("mgpjOpen");
      if (open0) open0.disabled = true;
      mgpjSetList("mgpjWarnBox", "mgpjWarn", [], null);
      mgpjSetList("mgpjFailBox", "mgpjFailed", [], null);
      var fill0 = mgpj("mgpjFill");
      if (fill0) fill0.style.width = "0%";
      if (window.mgStep) window.mgStep("project");

      fetch("/api/merge/export-project", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body)
      }).then(function (r) {
        return r.json().then(function (d) {
          if (!r.ok || d.ok === false) throw new Error(d.detail || d.error || "启动失败");
          return d;
        });
      }).then(function (d) {
        mgpjRender(d);
        toast("开始导出项目", String(d.out_video || "").slice(-46), "info");
        mgpjPoll();
      }).catch(function (e) {
        window.__mgpjExporting = false;
        toast("导出项目未启动", String((e && e.message) || e).slice(0, 140), "error");
      });
    }
    window.mgpjStart = mgpjStart;

    var pjBtn = $("mgExportProject");
    if (pjBtn && !pjBtn.dataset.bound) {
      pjBtn.dataset.bound = "1";
      pjBtn.addEventListener("click", function () {
        mgNeedProjectDir().then(function (hit) {
          if (!hit) return;                 /* 已经弹过一条说明模式与缺件的提示 */
          window.__mergeInputDir = hit.dir;
          mgOpenFaceDb("export-project", function () { mgpjStart(hit.dir); }, hit.dir);
        });
      });
    }

    var pjCancel = $("mgpjCancel");
    if (pjCancel && !pjCancel.dataset.bound) {
      pjCancel.dataset.bound = "1";
      pjCancel.addEventListener("click", function () {
        if (!window.__mgpjExporting) { if (window.mgStep) window.mgStep("main"); return; }
        pjCancel.disabled = true;
        fetch("/api/merge/export-project/cancel", { method: "POST" })
          .then(function (r) { return r.json(); })
          .then(function (d) {
            toast(d.ok ? "正在取消" : "取消失败",
                  d.ok ? "跑完当前这一帧就停" : String(d.error || d.reason || ""),
                  d.ok ? "info" : "error");
          })
          .catch(function () {})
          .then(function () { setTimeout(function () { pjCancel.disabled = false; }, 2000); });
      });
    }

    var pjOpen = $("mgpjOpen");
    if (pjOpen && !pjOpen.dataset.bound) {
      pjOpen.dataset.bound = "1";
      pjOpen.addEventListener("click", function () {
        var path = (mgpj("mgpjOut") || {}).textContent || "";
        if (!path || path === "—") return;
        fetch("/api/merge/reveal", {
          method: "POST", headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ path: path })
        }).then(function (r) { return r.json(); }).then(function (d) {
          if (!d.ok) toast("打不开所在目录", String(d.error || "").slice(0, 110), "error");
        }).catch(function (e) {
          toast("打不开所在目录", String((e && e.message) || e).slice(0, 110), "error");
        });
      });
    }
