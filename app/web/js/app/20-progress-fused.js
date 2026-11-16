    /* ============================================================
       快速导出（融合流水线，后端 fused=1）—— 视频模式的**默认**导出路径
       ------------------------------------------------------------
       一次 POST /api/merge/export（body 带 fused:1），后端把
       解码 → 检测 → 对齐/归组 → dfm → XSeg → 合成 → 编码 六段同时在内存里跑，
       段间只有有界队列。和分段链的三个差别，界面都要如实说出来：
         · 不建 <名字>_cache/（一个中间文件都不落，帧只在 %TEMP% 里经过）；
         · **不支持续跑**：取消就是整段重跑（分段链才有阶段游标）；
         · 进度没有"到第几步"这回事 —— 六段全程同时在跑，只有帧进度。
       所以进度页走「六步都显示进行中 + 帧进度」这一档（用户给的两个选项之一）。
       ============================================================ */
    var MGP_FUSED_TIMER = null;
    window.__mgFusedRunning = false;

    /* （六段的列标题状态由上面 MG_PIPE 那段里的 mgpFusedSteps 负责，这里不再重复定义） */

    function mgpFusedPaint(d) {
      d = d || {};
      if (d.eff) mgpEffRender(d.eff, d.eff_warnings);
      var v = d.vid || {};
      var total = Number(v.total || d.total || 0);
      var done = Number(v.done || d.frames_done || 0);
      var rate = Number(v.frame_fps || 0);
      var wasCancel = !!d.cancelled && !d.ok;
      var okAll = !!d.ok && !wasCancel;
      var p = Math.max(0, Math.min(100, Number(d.pct) || 0));
      if (!d.running && !okAll && !p) {
        p = Math.max(0, Math.min(100, Number(d.pct_at_failure) || 0));
      }
      var fr = $("mgpFrames");
      if (fr) {
        if (total) {
          /* 速度只写在标题右侧那一个地方（#mgPipeRight）：这里再来一次就是同一行里
             两个「11.3 帧/秒」（用户看到的噪音之一）。 */
          fr.textContent = done + " / " + total + " 帧";
        } else {
          fr.textContent = d.msg || "准备中…";
        }
      }
      var budget = (v.frame_budget != null) ? v.frame_budget : d.frame_budget;
      var eta = $("mgpEta");
      if (eta) {
        if (d.running) {
          var bits = [];
          if (v.eta_s != null) bits.push("剩余 " + v.eta_s + " 秒");
          else if (d.msg) bits.push(String(d.msg).slice(0, 40));
          if (budget != null) bits.push("帧缓冲 " + budget);
          eta.textContent = bits.join(" · ") || "剩余 —";
        } else if (okAll) {
          eta.textContent = "完成";
        } else if (wasCancel) {
          eta.textContent = "已取消（不支持续跑）";
        } else {
          eta.textContent = String(d.error || "已停").slice(0, 48);
        }
      }
      /* 文件名栏：把「融合流水线」这件事直接写出来（六步全亮却不标一句会让人以为卡住） */
      if (d.video) {
        mgpSetFile(String(d.video).split(/[\\/]/).pop() + " · 融合流水线",
                   "六段同时在内存里跑，不建 <名字>_cache/；不支持续跑（取消即整段重跑）");
      }
      mgpFusedSteps(okAll ? "done" : (d.running ? "on" : (wasCancel ? "idle" : "err")));
      /* 流水线实况：这一份 status 就是那张图的唯一输入（stages → lane/曲线/光点） */
      MG_PIPE.paint(d);
    }

    function mgFusedStopPoll() {
      if (MGP_FUSED_TIMER) { clearInterval(MGP_FUSED_TIMER); MGP_FUSED_TIMER = null; }
    }

    function mgFusedPoll() {
      mgFusedStopPoll();
      /* fired：setInterval 不等异步回调 —— status 那一发要是慢过 500ms，会同时压着
         两三个 tick 一起回来，每个都走一遍终态分支 = 同一条「导出完成」弹 2~3 次。
         终态只许收一次口（这也是「连点/抖动不叠 toast」的一部分）。 */
      var fired = false;
      MGP_FUSED_TIMER = setInterval(function () {
        if (fired) return;
        fetch("/api/merge/export/status").then(function (r) { return r.json(); })
          .then(function (d) {
            if (fired) return;
            mgpFusedPaint(d);
            if (d.running) return;
            fired = true;
            mgFusedStopPoll();
            window.__mgFusedRunning = false;
            window.__mergeExporting = false;
            var wasCancel = !!d.cancelled && !d.ok;
            var okAll = !!d.ok && !wasCancel;
            var cb = $("mgpCancel");
            if (cb) { cb.disabled = false; cb.textContent = "取消导出"; }
            if (okAll) {
              mgpChainDone();            /* 收尾：六个圆点同时轻微发光一次 */
              toast("导出完成", String(d.out || "").slice(-60), "success");
            } else if (d.vram_oom) {
              /* 强制尝试撞 OOM：后端已经把现场收干净（放掉本进程会话 + 重算当下的账），
                 这里只要把界面搬回被拒态 —— 用户看的必须是「确实不够 / 已回滚」，
                 而不是一句莫名的「导出失败」。 */
              window.mgVramReject(d.vram_oom, window.__mgForceRetry);
              return;
            } else if (wasCancel) {
              /* 关键差异：分段链说"接着跑"，快速导出只能说"整段重跑" */
              toast("已取消导出", "快速导出不支持续跑，需整段重跑", "info");
            } else {
              toast("导出失败", String(d.error || d.msg || "").slice(0, 140), "error");
            }
            if (typeof window.mgpResultShow === "function") {
              window.mgpResultShow(!!okAll, {
                phase: okAll ? "done" : (wasCancel ? "cancelled" : "failed"),
                out_video: d.out, out: d.out, cache_removed: false,
                fused: true, error: d.error || d.msg || ""
              });
            }
          }).catch(function () { /* 网络抖一下就等下一拍 */ });
      /* 250ms：和 MG_PIPE 的 POLL_MS 同一个窗口（一拍的光点就铺在这 250ms 里）。
         以前 500ms 时，20 帧/秒的一段一次要补 10 个 tick，全在同一瞬间放出来，
         肉眼就是「每半秒跳一波」；4 次/秒的 status 请求在后端只是读一次 qsize，不贵。 */
      }, 250);
    }

    function mgFusedExport(payload) {
      payload = payload || {};
      var video = mgVidPath();
      if (!video) {
        toast("没有素材", "先在工作台导入一段视频", "warn");
        return;
      }
      if (window.__mergeExporting || window.__mgStageRunning || window.__mgVramChecking) {
        mgToastOnce("merge-busy", "正在导出", "上一次还没结束", "info");
        return;
      }
      var body = {
        video: video,
        fused: 1,                       /* ★ 默认路径：融合流水线 */
        faces_project: String(window.__mergeFaceLib || ""),
        model: payload.model ||
               (document.querySelector("#mgModels .mgitem.on .nm") || {}).textContent || "",
        models: (window.mgExportModelsPayload ? window.mgExportModelsPayload() : mgModelsPayload()),
        thresh: 0.5,
        settings: mgMergeSettings(payload)
      };
      /* 同上：检测器有真值才发；quality 没有就不发（别再替用户写死 "high"）。 */
      var detF = window.mgDetectorBody ? window.mgDetectorBody(payload) : "";
      if (detF) body.detector = detF;
      if (payload.align_quality) body.quality = payload.align_quality;
      if (payload.output_path && /^[A-Za-z]:[\\/]/.test(payload.output_path)) {
        body.output_dir = payload.output_path;
      }
      var frameFrom = parseInt(payload.frame_from, 10);
      if (!isNaN(frameFrom) && frameFrom > 0) body.start_frame = frameFrom;
      var frameTo = parseInt(payload.frame_to, 10);
      if (!isNaN(frameTo) && frameTo > 0) body.end_frame = frameTo;
      mgPerfBody(body, payload);          /* 性能那两个下拉：只有非「自动」才带键 */

      /* ★ 显存预检（dry_run）：装不下就**在这里**收口 —— 不进进度页、不起 job。
         以前是先把六步画成「并行中」，再把六步打红 + 三条一样的 toast。 */
      window.__mgVramChecking = true;
      mgVramPreflight(body, function (proceed) {
        window.__mgVramChecking = false;
        if (!proceed) return;                    /* 已被拒：toast + 拒绝态都出来了 */
        mgFusedStart(payload, body);
      });
    }
    /* 预检通过之后真正开跑（内容和以前一模一样，只是被包进了回调里） */
    function mgFusedStart(payload, body) {
      window.__mergeExporting = true;
      window.__mgFusedRunning = true;
      window.__mgStageRunning = false;
      mgpStopPoll();
      mgpReset();
      mgpResultHide();
      if (window.mgStep) window.mgStep("progress");
      var cb = $("mgpCancel");
      if (cb) { cb.disabled = false; cb.textContent = "取消导出"; cb.hidden = false; }
      mgpFusedPaint({ running: true, msg: "启动快速导出…", vid: {} });
      document.dispatchEvent(new CustomEvent("merge:export-start", { detail: payload }));

      fetch("/api/merge/export", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body)
      }).then(function (r) {
        return r.json().then(function (d) { return { status: r.status, ok: r.ok, d: d || {} }; });
      }).then(function (res) {
        var d = res.d || {};
        /* 二次防线：预检和真请求之间显存可能被别人抢走 —— 这里同样按「被拒」收口，
           不许只弹一条 toast 就把六步留在「并行中」。 */
        if (res.status === 400 && d.need_mb != null) { mgVramReject(d); return; }
        if (!res.ok || d.ok === false) {
          throw new Error(d.error || d.detail || ("HTTP " + res.status));
        }
        if (d.fused === false) { mgpFusedPaint({ running: true, msg: "后端走了分段链…", vid: {} }); }
        mgFusedPoll();
      }).catch(function (e) {
        window.__mergeExporting = false;
        window.__mgFusedRunning = false;
        mgFusedStopPoll();
        toast("导出未启动", String((e && e.message) || e).slice(0, 140), "error");
      });
    }
    window.mgFusedExport = mgFusedExport;
    window.mgpFusedPaint = mgpFusedPaint;


    /* 演示：没有真任务时把四步动画跑一遍（保留给调试/截图用） */
    function mgDemoProgress() {
      if (window.__mergeExporting || window.__mgStageRunning) return;
      var k = 0;
      if (MGP_TIMER) clearInterval(MGP_TIMER);
      mgpReset();
      mgpPaint(0, { state: "on" });
      MGP_TIMER = setInterval(function () {
        k += 1;
        if (k > MGP_STAGES.length) { clearInterval(MGP_TIMER); MGP_TIMER = null; return; }
        var demoTips = ["486 帧", "512 张脸", "512 张换脸", "512 张遮罩", "486/486", "crf16"];
        mgpPaint(k - 1, { state: "done", tip: demoTips[k - 1] || "" });
        if (k < MGP_STAGES.length) mgpPaint(k, { state: "on" });

      }, 700);
    }
    window.mgDemoProgress = mgDemoProgress;

    var cancelBtn = $("mgpCancel");
    if (cancelBtn && !cancelBtn.dataset.bound) {
      cancelBtn.dataset.bound = "1";
      cancelBtn.addEventListener("click", function () {
        if (!window.__mergeExporting) { if (window.mgStep) window.mgStep("settings"); return; }
        cancelBtn.disabled = true;
        /* 快速导出（融合流水线）：只有 /export/cancel 这一条取消路，
           取消后清理临时帧、整段作废（没有游标可续）。 */
        if (window.__mgFusedRunning) {
          fetch("/api/merge/export/cancel", { method: "POST" })
            .then(function (r) { return r.json(); })
            .then(function (d) {
              toast(d.ok ? "正在取消" : "取消失败",
                    d.ok ? "快速导出不支持续跑：取消后已写出的临时帧会被清掉，需整段重跑"
                         : String(d.error || d.reason || ""),
                    d.ok ? "info" : "error");
            })
            .catch(function () {})
            .then(function () { setTimeout(function () { cancelBtn.disabled = false; }, 2500); });
          return;
        }
        if (window.__mgStageRunning) {
          fetch("/api/merge/stage/cancel", { method: "POST" })
            .then(function (r) { return r.json(); })
            .then(function (d) {
              toast(d.ok ? "正在取消" : "取消失败",
                    d.ok ? ("跑完这一帧就停：" + (d.stage || "")) : String(d.error || ""),
                    d.ok ? "info" : "error");
            })
            .catch(function () {})
            .then(function () { setTimeout(function () { cancelBtn.disabled = false; }, 2000); });
          return;
        }
        fetch("/api/merge/export/cancel", { method: "POST" })
          .then(function (r) { return r.json(); })
          .then(function (d) {
            toast(d.ok ? "正在取消" : "取消失败", d.ok ? "等它退出，几秒" : String(d.reason || ""), d.ok ? "info" : "error");
          })
          .catch(function () {})
          .then(function () { setTimeout(function () { cancelBtn.disabled = false; }, 2500); });
      });
    }

    var toSet = $("mgToSettings");
    if (toSet && !toSet.dataset.bound) {
      toSet.dataset.bound = "1";
      toSet.addEventListener("click", function () { if (window.mgStep) window.mgStep("settings"); });
    }
    /* 合成参数：界面上的控件 → 后端 settings（/export 和 /export-project 共用这一份） */
    function mgMergeSettings(payload) {
      payload = payload || {};
      return {
        mode: payload.mode || "overlay",
        /* 界面上是两个 seg：遮罩来源（DFM / XSeg / XSegLite）× 遮罩模式（prd×dst / prd / dst）。
           这里合成 MergeStudio 的 mask_mode 整数：
             DFM   → 4 / 2 / 1      XSeg → 8 / 6 / 7 */
        mask_mode: (function () {
          var src = (document.querySelector('#p-merge .seg[data-k="seg"] b.on') || {}).textContent || "DFM";
          var md = (document.querySelector('#p-merge .seg[data-k="mask"] b.on') || {}).textContent || "prd×dst";
          md = md.trim();
          var isX = src.trim().indexOf("XSeg") === 0;
          if (isX) return md === "dst" ? 7 : (md === "prd" ? 6 : 8);
          return md === "dst" ? 1 : (md === "prd" ? 2 : 4);
        })(),
        erode_mask: payload.erode_mask,
        blur_mask: payload.blur_mask,
        color_transfer: payload.color_transfer,
        face_type: payload.face_type,
        image_format: payload.image_format || "jpg"
      };
    }
    window.mgMergeSettings = mgMergeSettings;

    /* 两个性能旋钮 -> 请求体。
         ★ 选「自动」时**不带这个键**（不是发 "auto"、更不是发一个替用户选好的具体值）：
           带了就等于替用户做了决定，而他以后看到的界面仍是「自动」—— 那种
           「界面上设一套、实际跑另一套」正是这轮要根治的东西。
         ★ 前端不发 auto，后端也有兜底：body 里没有这两个键 = 用 env / 后端默认
           （见 app/server/backend_merge/knobs.py 的三条口径）。 */
    function mgPerfBody(body, payload) {
      body = body || {};
      payload = payload || {};
      var dt = String(payload.dfm_dtype == null ? "auto" : payload.dfm_dtype).trim().toLowerCase();
      if (dt && dt !== "auto") body.dfm_dtype = dt;
      var cb = String(payload.compose_backend == null ? "auto" : payload.compose_backend).trim().toLowerCase();
      if (cb === "gpu") body.compose_gpu = 1;
      else if (cb === "cpu") body.compose_gpu = 0;
      return body;
    }
    window.mgPerfBody = mgPerfBody;

    /* 真跑一次合成。工作台里的「导出」会先走到导出设置页，
       两个「开始导出」（#mgStartExport2 在工作台、#mgStartExport 在子页）共用这一份逻辑。 */
    function mgRunExport(payload) {
      payload = payload || {};
      document.dispatchEvent(new CustomEvent("merge:export-start", { detail: payload }));

      var inputDir = window.__mergeInputDir || "";
      /* 视频模式：只要打开了源视频就能导出，不需要先抽帧 + 对齐成项目。
         两种输入互斥 —— 有项目目录就走老路（帧目录），否则带 video 走后端
         的视频分支（见 routes_merge._start_video_export）。 */
      var video = mgVidPath();
      if (!inputDir && !video) {
        toast("没有素材", "先在工作台导入一段视频（或打开一个 DFL 项目）", "warn");
        return;
      }
      if (window.__mergeExporting) {
        toast("正在合成", "上一次还没结束", "info");
        return;
      }
      if (video && !inputDir) {
        /* 视频模式，两条路：
           · 快速导出（融合流水线，默认）：一次 POST /api/merge/export（fused=1），
             后端六段同时在内存里跑 —— 不建 <名字>_cache/、快约 2.8 倍，但**不支持续跑**；
           · 分段链（关掉「快速导出」才走）：切帧 → 切脸 → 换脸 → 遮罩 → 合成 → 导出，
             每步独立 API/独立产物，失败/取消能从断点接着跑（排查、对照用这一条）。 */
        if (payload.fast_export !== false) { mgFusedExport(payload); return; }
        mgStageExport(payload);
        return;
      }
      window.__mergeExporting = true;
      if (typeof window.mgpResultHide === "function") window.mgpResultHide();

      var body = {
        input_dir: inputDir,
        project: inputDir,
        video: video,                       /* 视频模式：后端从源视频逐帧导出 */
        /* 人脸库目录：空 = 默认库（<工作区>/merge）。视频模式下没有项目，
           库有组就按 embedding 判、没组就全部用左栏选中的 dfm（后端判定）。 */
        faces_project: String(window.__mergeFaceLib || ""),
        model: (document.querySelector("#mgModels .mgitem.on .nm") || {}).textContent || "",
        /* 多选契约：每个选中模型一条 {name, faces}；faces=[] = 全部脸。
           导出用**全部选中项**（mgExportModelsPayload），不按预览那套按脸分摊。 */
        models: (window.mgExportModelsPayload ? window.mgExportModelsPayload() : mgModelsPayload()),
        keep_cache: !!payload.keep_cache,
        /* 检测器：这条（项目模式，走 4.0 子进程）以前**根本不发**，后端也不读，
           等于界面选什么都不算数、引擎按自己的默认 YOLOv8 跑，抽屉却写 tinymog。
           现在和视频/分段两条路一样，把下拉的值发出去；读不到就**不发**（不塞空串）。 */
        workers: parseInt(payload.num_workers, 10) || 0,
        encoder: payload.encoder || "h264_nvenc",
        hwaccel: payload.hwaccel === "" ? "off" : (payload.hwaccel || "auto"),
        settings: mgMergeSettings(payload)
      };
      var detE = window.mgDetectorBody ? window.mgDetectorBody(payload) : "";
      if (detE) body.detector = detE;
      if (payload.output_path) {
        if (/^[A-Za-z]:[\\/]/.test(payload.output_path)) {
          body.output_dir = payload.output_path;
        } else {
          /* 相对路径后端不收：以前静默丢掉，用户会以为填的已生效 */
          toast("输出路径被忽略", "只认 Windows 绝对路径，这次按默认命名输出", "warn");
        }
      }
      mgPerfBody(body, payload);          /* 性能那两个下拉：只有非「自动」才带键 */

      fetch("/api/merge/export", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body)
      }).then(function (r) {
        return r.json().then(function (d) {
          if (!r.ok) throw new Error(d.detail || "启动失败");
          return d;
        });
      }).then(function (d) {
        toast("已开始合成", String(d.output || d.input_dir || "").slice(-46), "info");
        if (window.mgStep) window.mgStep("progress");
        pollMergeExport();
      }).catch(function (e) {
        window.__mergeExporting = false;
        toast("合成未启动", String((e && e.message) || e).slice(0, 120), "error");
      });
    }
    window.mgRunExport = mgRunExport;

    var startEx = $("mgStartExport2");
    if (startEx && !startEx.dataset.bound) {
      startEx.dataset.bound = "1";
      startEx.addEventListener("click", function () {
        mgRunExport((window.MergePages && window.MergePages.settings) ? window.MergePages.settings() : {});
      });
    }

    /* 导出设置页底部的「开始导出」：**整条导出链唯一的 POST 入口**。
       0 个模型直接挡回去（页面上这颗按钮本来就该是禁的，这里再兜一层）。
       参数一律从 MergePages.settings() 现取：编码器 / 质量 / 格式 / 每帧进程 /
       保留 cache / 帧范围，都是用户在设置页上改完的那一份，不会被重置。 */
    var startGo = $("mgStartExport");
    if (startGo && !startGo.dataset.bound) {
      startGo.dataset.bound = "1";
      startGo.addEventListener("click", function () {
        if (!mgSyncExportGate()) {
          mgToastErr("export-nosel", "先选一个模型",
                     "回合成器左栏「模型」里点一个（可多选），再回来导出", "warn", 4000);
          return;
        }
        var p = (window.MergePages && window.MergePages.settings)
          ? window.MergePages.settings() : {};
        mgRunExport(p);
      });
    }

    /* 导出设置页的「取消」：回合成器，参数一个都不动（控件值原样留着，下次进来还是它） */
    var cancelSet = $("mgCancelExport");
    if (cancelSet && !cancelSet.dataset.bound) {
      cancelSet.dataset.bound = "1";
      cancelSet.addEventListener("click", function () {
        mgStep("main");
      });
    }

    /* 「全片」：把帧范围清回默认（起始 0、结束留空） */
    var frameAll = $("mgFrameAll");
    if (frameAll && !frameAll.dataset.bound) {
      frameAll.dataset.bound = "1";
      frameAll.addEventListener("click", function () {
        var a = $("mgFrameFrom"), b = $("mgFrameTo");
        if (a) a.value = "0";
        if (b) b.value = "";
        if (window.MergePages && window.MergePages.syncStatus) window.MergePages.syncStatus();
      });
    }

    /* 结果面板：返回导出设置（参数都还在）/ 再次导出（同一份参数再跑一遍） */
    var toSet2 = $("mgpToSettings");
    if (toSet2 && !toSet2.dataset.bound) {
      toSet2.dataset.bound = "1";
      toSet2.addEventListener("click", function () {
        mgpResultHide();
        if (window.mgpRejectHide) window.mgpRejectHide();   /* 拒绝态（六步收起）也一起复位 */
        mgStep("settings");
        mgFillExportSettings();
      });
    }
    var againBtn = $("mgpAgain");
    if (againBtn && !againBtn.dataset.bound) {
      againBtn.dataset.bound = "1";
      againBtn.addEventListener("click", function () {
        if (window.__mergeExporting || window.__mgStageRunning) {
          toast("正在导出", "上一次还没结束", "info");
          return;
        }
        if (!mgSyncExportGate()) {
          mgpResultHide();
          mgStep("settings");
          mgToastErr("export-nosel", "先选一个模型", "回合成器左栏选一个模型", "warn", 4000);
          return;
        }
        mgpResultHide();
        mgRunExport((window.MergePages && window.MergePages.settings)
          ? window.MergePages.settings() : {});
      });
    }

    var tlBox = $("mergeTracks");
    if (tlBox && !tlBox.dataset.frameBound) {
      tlBox.dataset.frameBound = "1";
      /* 时间轴说帧变了 → 统一走 mgFrameChange：先清框，检测一回来就画框，
         合成随后只换底图。以前这里是「先 mgCmpNow（含检测 + dfm + 合成，~400ms）
         再 mvDetect」，于是框跟着合成那一发一起到；视频模式那条分支更彻底 ——
         连清框都没有，播放/拖动时旧框一直挂在画面上（「播放后旧人脸框不清除」）。 */
      tlBox.addEventListener("mtl:frame", function (e) {
        var n = (e.detail && e.detail.frame) || 0;
        if (window.mgFrameChange) { window.mgFrameChange(n, { why: "timeline" }); return; }
        if (window.mvClearBoxes) window.mvClearBoxes();
        if (window.mgCmpNow) window.mgCmpNow({ frame: n });
        else window.mvFrame(n);
        if (window.mgBoxesOn && window.mgBoxesOn()) window.mvDetect(n);
      });
      /* 时间轴自己按 setInterval 播放（帧目录模式）：按播放的那一下也要立刻清框，
         别等第一拍（33ms 后）才清 —— 那一拍之前画面上还是上一帧的框。 */
      tlBox.addEventListener("mtl:play", function (e) {
        var on = !!(e.detail && e.detail.playing);
        var n = (window.MergeTimeline && window.MergeTimeline.getFrame)
          ? window.MergeTimeline.getFrame() : 0;
        if (on) {
          /* 播放开始：只清框，不补发检测 —— 下一拍（33ms 后）就是新帧，
             那一发会在 180ms 后回来，而那时画面已经走远了。 */
          if (window.mgInvalidateBoxes) window.mgInvalidateBoxes("play", n);
          return;
        }
        /* 暂停：这一帧定住了，清干净重来一发（检测 + 合成），框和换脸结果都补齐 */
        if (window.mgFrameChange) window.mgFrameChange(n, { why: "pause" });
      });
    }


    var addProj = $("mgAddProject");
    if (addProj && !addProj.dataset.bound) {
      addProj.dataset.bound = "1";
      addProj.addEventListener("click", function () { window.mvOpenProject(); });
    }

    var addV = $("mgAddVideo");
    if (addV && !addV.dataset.bound) {
      addV.dataset.bound = "1";
    }
    if (addV) addV.addEventListener("click", function () {
      /* 桌面版直接选一个视频文件（比"先选目录再列文件"少一步） */
      if (window.desktop && window.desktop.pickFile) {
        window.desktop.pickFile("选择视频").then(function (p) {
          if (p) window.mgPrepare(p);
        });
        return;
      }
      if (!(window.desktop && window.desktop.pickDir)) {
        toast("导入视频", "网页版拿不到本地路径，请把视频放进工作区", "info");
        return;
      }
      window.desktop.pickDir("选择视频所在目录").then(function (dir) {
        if (!dir) return;
        fetch("/api/merge/videos?dir=" + encodeURIComponent(dir))
          .then(function (r) { return r.json(); })
          .then(function (d) {
            var box = $("mgVideos");
            var empty = $("mgVideosEmpty");
            var list = (d.videos || []).slice(0, 60);
            box.innerHTML = "";
            if (!list.length) {
              if (empty) { empty.style.display = ""; empty.textContent = "这个目录里没有视频"; }
              toast("没有找到视频", dir, "info");
              return;
            }
            if (empty) empty.style.display = "none";
            (d.videos || []).forEach(function (v) { if (!v.dir) v.dir = dir; });
            list.forEach(function (v) {
              var row = document.createElement("div");
              row.className = "mgitem";
              row.dataset.path = v.path;
              row.innerHTML = '<span class="nm">' + v.name + '</span>';
              row.addEventListener("click", function () {
                /* 点视频 = 准备这个素材（抽帧 + 对齐），不是只看看首帧 */
                window.mgPrepare(v.path);
              });
              box.appendChild(row);
            });
            toast("已导入 " + list.length + " 个视频", dir, "success");
          });
      });
    });
  }

  window.bindMergeUI = bindMergeUI;

  /* 内置 DFM 导出：起后台任务，轮询状态，导出完刷新列表 */
  var DFM_POLL = null;

  function exportDfm(name, btn) {
    if (btn) { btn.disabled = true; btn.textContent = "导出中"; }
    toast("开始导出", name + " → .dfm（要加载模型，可能几分钟）", "info");
    fetch("/api/models/" + encodeURIComponent(name) + "/export", { method: "POST" })
      .then(function (r) {
        if (!r.ok) return r.json().then(function (d) { throw new Error(d.detail || "启动失败"); });
        return r.json();
      })
      .then(function () { pollExport(name, btn); })
      .catch(function (e) {
        if (btn) { btn.disabled = false; btn.textContent = "导出"; }
        toast("导出失败", String((e && e.message) || e), "error");
      });
  }

  function pollExport(name, btn) {
    if (DFM_POLL) clearInterval(DFM_POLL);
    DFM_POLL = setInterval(function () {
      fetch("/api/models/export/status").then(function (r) { return r.json(); }).then(function (d) {
        if (btn) {
          var last = (d.log && d.log.length) ? d.log[d.log.length - 1] : "";
          btn.textContent = d.running ? (last ? last.slice(0, 6) : "导出中") : "导出";
        }
        if (!d.running) {
          clearInterval(DFM_POLL); DFM_POLL = null;
          if (d.ok) {
            toast("导出完成", name + " 已生成 .dfm", "success");
            loadModels();
            /* 列表不重绘就一直看不到新出现的 DFM 标记；选中由 renderMerge 按名字保住 */
            if (window.MODELS) renderMerge(window.MODELS);
          } else {
            if (btn) { btn.disabled = false; btn.textContent = "导出"; }
            toast("导出失败", d.error || "看控制台日志", "error");
            console.warn("export log", (d.log || []).join("\n"));
          }
        }
      }).catch(function () {});
    }, 1200);
  }
  var MERGE_POLL = null;
  function pollMergeExport() {
    if (MERGE_POLL) clearInterval(MERGE_POLL);
    /* 新一轮导出：把上一轮留下的终态按钮字样（返回设置/重新导出）复位 */
    var cb0 = document.getElementById("mgpCancel");
    if (cb0) { cb0.textContent = "取消导出"; cb0.disabled = false; }
    /* 和 mgFusedPoll 同一个道理：setInterval 不等异步回调，慢一拍就会几个 tick 同时
       走终态分支 → 同一条结果 toast 弹好几次。终态只收一次口。 */
    var fired = false;
    MERGE_POLL = setInterval(function () {
      if (fired) return;
      fetch("/api/merge/export/status").then(function (r) { return r.json(); }).then(function (d) {
        if (fired) return;
        var frEl = document.getElementById("mgpFrames");
        var msg = document.getElementById("mgpEta");
        /* 终态语义：后端 ok 是「任务成功」标志，取消时是 ok=false + cancelled=true。
           以前拿 !running 当成功，取消和失败都画成满格 100%。
           注意「取消来晚、成片已生成」：后端 ok=true + cancelled=true + phase="done"，
           成片真实存在，这种情况不算取消（否则会把有产物的一次显示成已取消）。 */
        var wasCancel = !!d.cancelled && !d.ok;
        var okAll = !!d.ok && !wasCancel;
        var last = (d.log && d.log.length) ? d.log[d.log.length - 1] : "";
        /* 后端上报的是真实阶段进度（pct / msg / 帧数），进度页直接照它画 */
        var text = d.msg || last;
        var p = Math.max(0, Math.min(100, Number(d.pct) || 0));
        /* 失败/取消时后端把 pct 归 0、把停下来的位置放进 pct_at_failure，
           不补这一步，「停在当前 pct」会画成 0%。 */
        if (!d.running && !okAll && !p) {
          p = Math.max(0, Math.min(100, Number(d.pct_at_failure) || 0));
        }
        if (msg) msg.textContent = text ? String(text).slice(0, 60) : "合成中…";
        if (d.out && String(d.out) !== "—") {
          if (window.mgpSetFile) window.mgpSetFile(String(d.out).split(/[\\/]/).pop());
        }

        /* 阶段直接用后端的 @@P stage：0 搭目录/载模型 · 3 换脸 · 5 合成 · 6 编码 */
        var srv = Number(d.stage);
        var stage;
        if (!isNaN(srv) && srv >= 0) {
          stage = srv >= 6 ? 2 : (srv >= 3 ? 1 : 0);
        } else {
          stage = p < 20 ? 0 : p < 92 ? 1 : p < 100 ? 2 : 3;
        }
        if (!d.running) stage = okAll ? 3 : stage;
        var done = /Merging frames (\d+)\/(\d+)/.exec(text || "");
        if (frEl) {
          if (done) frEl.textContent = done[1] + " / " + done[2] + " 帧";
          else if (!d.running && okAll) frEl.textContent = "完成";
        }
        /* 四步动画（复用「模型转换」的 Stepper）：后端 stage 0..6
           -> 0 切帧 / 1 切脸 / 2 合成 / 3 导出。老的四行「—」已经删了。 */
        var si = srv >= 6 ? 3 : (srv >= 3 ? 2 : (srv >= 2 ? 1 : 0));
        mgpPaint(si, {
          state: d.running ? "on" : (okAll ? "done" : "err"),
          tip: (si === 2 && done) ? (done[1] + "/" + done[2]) : (si === 3 ? "crf16 · yuv420p" : "")
        });

        if (!d.running) {
          if (fired) return;
          fired = true;
          clearInterval(MERGE_POLL); MERGE_POLL = null;
          window.__mergeExporting = false;
          /* 终态文案：成功「完成」· 取消「已取消」· 失败给 d.error 前 60 字（不再只闪 3.6 秒的 toast） */
          if (msg) {
            msg.textContent = okAll ? "完成"
              : (wasCancel ? "已取消" : String(d.error || d.msg || "合成失败").slice(0, 60));
          }
          /* 终态出口统一交给结果面板（返回导出设置 / 再次导出），
             取消键自己藏起来 —— 不再让同一颗按钮在「取消导出 / 返回设置 / 重新导出」
             三种身份之间来回变。也不再自动跳回工作台：那会把刚出来的结果盖掉。 */
          var cbEnd = document.getElementById("mgpCancel");
          if (cbEnd) { cbEnd.disabled = false; cbEnd.textContent = "取消导出"; }
          toast(okAll ? "合成完成" : (wasCancel ? "已取消合成" : "合成失败"),
                okAll ? ("输出 " + d.out)
                      : (wasCancel ? "已停在你按取消时的位置" : String(d.error || "").slice(0, 120)),
                okAll ? "success" : (wasCancel ? "info" : "error"));
          if (typeof window.mgpResultShow === "function") {
            window.mgpResultShow(!!okAll, {
              phase: okAll ? "done" : (wasCancel ? "cancelled" : "failed"),
              out_video: d.out, error: d.error || d.msg || ""
            });
          }
        }
      }).catch(function () {});
    }, 1500);
  }
  window.pollMergeExport = pollMergeExport;

  window.exportDfm = exportDfm;

  function renderMerge(list) {
    var box = el("mgModels");
    if (!box) return;
    /* 重绘前记住当前选中的模型名：以前每行都按 i===0 重置，
       轮询一刷新，用户选好的模型就被换成第一个了 */
    var keepSel = mgSelNames();       /* 多选：整串选中项都要保住，不能只记第一个 */
    var keepRow = null;
    box.innerHTML = "";
    if (!list.length) {
      box.innerHTML = '<div class="empty-note">还没有可导出的模型</div>';
      return;
    }
    list.forEach(function (m) {
      var r = document.createElement("div");
      r.className = "mgitem";
      /* 有 .dfm 就是「已导出」：/api/state 给的是 has_export，/api/models 给的是 has_dfm，
         两个来源都认 —— 否则同一条记录在两个接口里会一个显示 DFM、一个显示「导出」，
         点了也只会 404。只有 .dfm、没有训练目录的条目本来就没有权重可导出。 */
      var exported = !!(m.has_export || m.has_dfm);
      if (m.dfm) r.title = m.dfm;
      /* 「已导出」标记：以前靠那个 DFM 徽标（.tag.ok）判定，徽标按要求删了，
         判定改挂到行类名上 —— 自动选中仍然优先停在已导出的模型，行为不变。 */
      if (exported) r.classList.add("has-dfm");
      r.innerHTML = '<span class="nm">' + m.name + '</span>' +
        
        /* 徽标（face all / face N）不再显示：元素留着但恒 hidden，下面 bdEl 的点击切换绑定不删 */
        '<span class="tag n mgface" hidden></span>' +
        (exported ? '' : '<button class="mini mgexp" title="导出为 .dfm">导出</button>');
      if (!exported) {
        var expBtn = r.querySelector(".mgexp");
        if (expBtn) expBtn.addEventListener("click", function (e) {
          e.stopPropagation();
          exportDfm(m.name, expBtn);
        });
      }
      /* 这一行在列表里的下标：Shift 范围选择要用（append 之前取，正好是它自己的位置） */
      var rowIdx = box.children.length;
      /* 选中判定全交给 mgPickRow：Ctrl/Cmd 切单条、Shift 选范围、普通点击仍是单选 */
      r.addEventListener("click", function (e) { mgPickRow(box, rowIdx, e); });
      /* 徽章：点它只切「管哪张脸」，不碰选中态 */
      var bdEl = r.querySelector(".mgface");
      if (bdEl) bdEl.addEventListener("click", function (e) {
        e.stopPropagation();
        mgCycleFaces(m.name);
      });
      box.appendChild(r);
      if (keepSel.indexOf(m.name) >= 0) keepRow = r;   /* 只用来判断「有没有命中」 */
    });
    /* 按名字恢复整串多选；一个都没命中（模型真的没了）才退回第一个 */
    Array.prototype.forEach.call(box.children, function (x) {
      var nm = String(((x.querySelector(".nm") || {}).textContent) || "").trim();
      if (nm && keepSel.indexOf(nm) >= 0) x.classList.add("on");
    });
    if (!keepRow) {
      /* 一个都没命中：优先选「已经导出过 .dfm」的那一行。后端合成只认 .dfm，
         默认停在一条没有 .dfm 的模型上 → 每次合成 404 → 开箱就是一屏空白。 */
      keepRow = Array.prototype.filter.call(box.children, function (x) {
        return x.classList.contains("has-dfm");
      })[0] || box.children[0] || null;
      if (keepRow) keepRow.classList.add("on");
    } else if (MG_SEL_AUTO) {
      /* 选中是「自动挑的」（用户没点过）：模型列表会分两拨到（/api/models 先，
         /api/state 后），第一拨里可能只有没导出的条目，名字一保住就再也换不掉，
         于是永远停在一条没有 .dfm 的模型上。这里允许自动选中升到有 .dfm 的那条。 */
      var onRows = Array.prototype.filter.call(box.children, function (x) {
        return x.classList.contains("on");
      });
      var hasDfm = onRows.some(function (x) { return x.classList.contains("has-dfm"); });
      var pref = Array.prototype.filter.call(box.children, function (x) {
        return x.classList.contains("has-dfm");
      })[0];
      if (!hasDfm && pref) {
        onRows.forEach(function (x) { x.classList.remove("on"); });
        pref.classList.add("on");
      }
    }
    mgSyncFaces();        /* 徽章跟着恢复后的选中项走 */
    mgWarmSelected();     /* 渲染完预热：多选时每个选中模型各打一枪（__mgWarm 自带 10 秒去重） */
    mgSyncExportGate();   /* 列表从 0 条变 N 条（或反过来）→ 导出入口跟着变 */
  }

  function renderDatasets(ds) {
    var box = el("dsStats");
    if (!box) return;
    function line(name, d) {
      var suffix = d.exists ? (d.count + " 张") : "不存在";
      return "workspace" + BS + name + BS + " <span class=\"n\">" + suffix + "</span>";
    }
    box.innerHTML = line("data_src", ds.src) + "<br>" + line("data_dst", ds.dst);
  }

  function renderPorts(p) {
    [["toolMaskPort", p.mask], ["toolAnglePort", p.angle]].forEach(function (kv) {
      var e = el(kv[0]);
      if (e) e.textContent = kv[1];
    });
  }

  function renderRuntime(s) {
    var rt = s.runtime || {};
    var ui = el("aboutUI");
    if (ui) {
      var m = (navigator.userAgent.match(/Electron\/([\d.]+)/) || [])[1];
      var c = (navigator.userAgent.match(/Chrome\/([\d.]+)/) || [])[1];
      ui.textContent = m ? ("Electron " + m + " · Chromium " + c) : "Browser";
    }
    var bk = el("aboutBackend");
    if (bk) bk.textContent = "Python " + (rt.python || "?") + " · 127.0.0.1:" + (rt.port || "?");
    var ws = el("aboutWorkspace");
    if (ws) ws.textContent = rt.workspace || "—";

    var st = s.ports_status || {};

    var url = el("svcurl");
    if (url) {
      var p = (s.ports || {}).merge || 8001;
      url.textContent = url.textContent.replace(/127\.0\.0\.1:\d+/, "127.0.0.1:" + p);
    }
  }

  function renderLastModel(list) {
    var card = el("lastModelCard");
    if (!card) return;
    if (!list.length) { card.classList.add("hide"); return; }
    card.classList.remove("hide");
    var m = list[0];
    var n = el("lastModelName"), it = el("lastModelIter");
    if (n) n.textContent = m.name;
    if (it) it.textContent = m.iter ? ("iter " + fmt(m.iter)) : m.state;
  }

  function loadModels() {
    fetch("/api/models").then(function (r) { return r.json(); }).then(function (d) {
      MODELS = d.models || [];
      window.MODELS = MODELS;      /* 重新绑定，别让外部还指着最初那个空数组 */
      renderTrainer(MODELS);
      renderMerge(MODELS);      /* 合成页的模型列表也吃这份新数据（has_export 才是新的） */
      if (STATE) renderHome(STATE, MODELS);
    }).catch(function (e) { console.warn("models load failed", e); });
  }
  window.loadModels = loadModels;
