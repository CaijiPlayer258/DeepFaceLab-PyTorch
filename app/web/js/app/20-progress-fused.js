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

  /* ---------- 面部 68 点：编辑器保存过就用保存的 ---------- */
  var FACE_PTS = null;
  function facePts() {
    if (FACE_PTS) return FACE_PTS;
    FACE_PTS = L68.map(function (p) { return [p[0], p[1], p[2]]; });
    return FACE_PTS;
  }
  /* 保存在工作区里（face68.json），换端口、换机器都还在。
     判据是 r.ok：后端没有这份数据时回 404 + detail（以前是 200 + points:null，
     只能靠 d.points == null 猜），这里把 detail 摆到界面上，别让它静默回内置坐标。 */
  function loadFace() {
    return fetch("/api/face").then(function (r) {
      if (!r.ok) {
        return r.json().catch(function () { return {}; }).then(function (d) {
          toast("这一帧没有关键点",
                String((d && (d.detail || d.error)) || ("HTTP " + r.status)).slice(0, 120),
                "info");
          return null;
        });
      }
      return r.json();
    }).then(function (d) {
      if (d && d.ok !== false && d.points && d.points.length === 68) {
        FACE_PTS = d.points;
        HEAD = null;
      }
    }).catch(function (e) { console.warn("face load failed", e); });
  }
  window.loadFace = loadFace;
  window.facePts = facePts;

  function syncDatasetBadge() {
    /* 只刷新「当前选中目录这一层」的张数。
       以前直接拿 STATE.datasets.src.count —— 那数的是 data_src/aligned，
       选 data_src/1 也照样显示 1203 张，用户拿它跟自己的 670 张一对就炸了。 */
    if (window.dsRefreshCount) { try { window.dsRefreshCount(); } catch (e) {} }
  }
  window.syncDatasetBadge = syncDatasetBadge;

  /* ---------- 路径框拖放 ---------- */
  function pathOfFile(f) {
    if (window.desktop && window.desktop.pathForFile) return window.desktop.pathForFile(f);
    return f.path || "";                      /* 浏览器兜底（普通网页无路径） */
  }

  function onPathDrop(box, path) {
    var val = box.querySelector(".pathval");
    if (!val) return;
    val.value = path;
    val.title = path;
    /* 数据集顶部的框：更新路径并刷新张数 */
    if (box.id === "dsPathBox") {
      DSB.dir = path;
      /* 换了目录就立刻问后端「这一层几张」—— 别再用 /api/state 里那个
         永远指向 data_src/aligned 的 datasets.src.count。 */
      dsRefreshCount();
      toast("数据集已更换", path, "success");
    } else {
      toast("路径已填入", path, "success");
    }
  }

  /* 所有路径框点一下都能选目录（文件选择器接上后换成真对话框） */
  /* 图标 → 系统目录选择器 → 写回输入框 */
  function pickFolderFor(box) {
    var api = window.desktop;
    if (!api || !api.pickDir) {
      toast("选择目录", "网页版拿不到本地路径，请直接填写或拖入文件夹", "info");
      return;
    }
    var title = box.id === "dsPathBox" ? "选择数据集目录" : "选择目录";
    api.pickDir(title).then(function (p) {
      if (!p) return;
      var inp = box.querySelector(".pathval");
      if (inp) { inp.value = p; inp.dispatchEvent(new Event("change", { bubbles: true })); }
      if (box.id === "dsPathBox") {
        DSB.dir = p;
        loadState();
        toast("数据集已更换", p, "success");
      } else {
        toast("路径已填入", p, "success");
      }
    }).catch(function (e) {
      toast("打开失败", String((e && e.message) || e), "error");
    });
  }
  window.pickFolderFor = pickFolderFor;

  /* 关闭拼写检查：路径、UID、数字都会被画红波浪线，纯干扰 */
  function noSpell() {
    document.querySelectorAll("input, textarea").forEach(function (el) {
      if (!el.hasAttribute("spellcheck")) el.spellcheck = false;
      if (!el.hasAttribute("autocomplete")) el.autocomplete = "off";
    });
  }
  window.noSpell = noSpell;

  function bindPickPaths() {
    document.querySelectorAll(".pathinp").forEach(function (box) {
      if (box.dataset.pick) return;
      box.dataset.pick = "1";
      /* 只有文件夹图标才唤起选择器；点文字区是编辑 */
      var icon = box.querySelector(".iconpill");
      if (icon) {
        /* 用 onclick 属性而不是 addEventListener：重复绑定只保留最后一个，
           否则每次进页面都会叠加一个 handler，点一下弹一串 toast */
        icon.onclick = function (e) {
          e.stopPropagation();
          e.preventDefault();
          pickFolderFor(box);
        };
      }
    });
  }
  window.bindPickPaths = bindPickPaths;

  function bindDropPaths() {
    document.querySelectorAll(".pathinp").forEach(function (box) {
      if (box.dataset.drop) return;
      box.dataset.drop = "1";
      var depth = 0;

      box.addEventListener("dragenter", function (e) {
        e.preventDefault();
        depth++;
        box.classList.add("dragover");
      });
      box.addEventListener("dragover", function (e) {
        e.preventDefault();
        if (e.dataTransfer) e.dataTransfer.dropEffect = "copy";
      });
      box.addEventListener("dragleave", function (e) {
        e.preventDefault();
        depth = Math.max(0, depth - 1);
        if (!depth) box.classList.remove("dragover");
      });
      box.addEventListener("drop", function (e) {
        e.preventDefault();
        e.stopPropagation();
        depth = 0;
        box.classList.remove("dragover");
        var f = e.dataTransfer && e.dataTransfer.files && e.dataTransfer.files[0];
        if (!f) return;
        var p = pathOfFile(f);
        if (!p) {
          box.classList.add("bad-drop");
          setTimeout(function () { box.classList.remove("bad-drop"); }, 900);
          toast("无法读取路径", "请把文件夹从资源管理器拖进来", "error");
          return;
        }
        onPathDrop(box, p);
      });
    });
  }
  window.bindDropPaths = bindDropPaths;

  /* 拖文件进窗口时，别让 Electron 直接导航过去 */
  document.addEventListener("dragover", function (e) { e.preventDefault(); });
  document.addEventListener("drop", function (e) { e.preventDefault(); });

  /* ---------- 3D 人脸编辑器（debug 页） ---------- */
  var ED = { pts: null, sel: 0, yaw: 0.35, pitch: -0.12, drag: null };

  /* Blender 式数值拖拽：按住左右拖动改值，上下也吃；位移超过 3px 才进入拖拽，
     否则当普通点击去聚焦输入 */
  function edBindDrag(input, opts) {
    if (!input || input.dataset.dragBound) return;
    input.dataset.dragBound = "1";
    var box = input.closest(".numdrag");
    var step = (opts && opts.step) || 0.005;
    var startX = 0, startY = 0, startVal = 0, active = false, moved = false;

    function clamp(v) {
      if (opts && opts.min !== undefined && v < opts.min) v = opts.min;
      if (opts && opts.max !== undefined && v > opts.max) v = opts.max;
      return v;
    }
    function apply(v) {
      input.value = (Math.round(v * 1000) / 1000).toFixed(3);
      input.dispatchEvent(new Event("input", { bubbles: true }));
    }

    input.addEventListener("pointerdown", function (e) {
      if (e.button !== 0) return;
      startX = e.clientX; startY = e.clientY;
      startVal = parseFloat(input.value) || 0;
      active = true; moved = false;
      input.setPointerCapture(e.pointerId);
    });

    input.addEventListener("pointermove", function (e) {
      if (!active) return;
      var dx = e.clientX - startX, dy = e.clientY - startY;
      if (!moved && Math.abs(dx) + Math.abs(dy) < 3) return;
      if (!moved) {
        moved = true;
        if (box) box.classList.add("dragging");
        input.blur();
      }
      e.preventDefault();
      /* 左右为主，上下反向也生效：右/上 = 加，左/下 = 减 */
      var delta = (dx - dy) * step;
      apply(clamp(startVal + delta));
    });

    function end(e) {
      if (!active) return;
      active = false;
      if (box) box.classList.remove("dragging");
      try { input.releasePointerCapture(e.pointerId); } catch (_) {}
      if (moved) {
        input.dispatchEvent(new Event("change", { bubbles: true }));
        if (window.edDraw) window.edDraw();
      } else {
        input.focus();
        input.select();
      }
    }
    input.addEventListener("pointerup", end);
    input.addEventListener("pointercancel", end);
    input.addEventListener("keydown", function (e) {
      if (e.key === "ArrowUp" || e.key === "ArrowDown") {
        e.preventDefault();
        var v = (parseFloat(input.value) || 0) + (e.key === "ArrowUp" ? 1 : -1) * step * (e.shiftKey ? 20 : 4);
        apply(clamp(v));
        if (window.edDraw) window.edDraw();
      }
    });
  }
  window.edBindDrag = edBindDrag;

  function edInit() {
    if (ED.pts) return;
    ED.pts = facePts().map(function (p) { return [p[0], p[1], p[2]]; });
    var list = $("edList");
    if (list && !list.childElementCount) {
      for (var i = 0; i < 68; i++) {
        var b = document.createElement("b");
        b.textContent = i;
        b.setAttribute("data-i", i);
        b.addEventListener("click", function () { edSelect(+this.getAttribute("data-i")); });
        list.appendChild(b);
      }
    }
    edSelect(0);
  }

  function edSelect(i) {
    ED.sel = i;
    var list = $("edList");
    if (list) Array.prototype.forEach.call(list.children, function (n, k) {
      n.classList.toggle("on", k === i);
    });
    if ($("edSel")) $("edSel").textContent = i + 1;
    var p = ED.pts[i];
    if ($("edX")) $("edX").value = p[0].toFixed(3);
    if ($("edY")) $("edY").value = p[1].toFixed(3);
    if ($("edZ")) $("edZ").value = p[2].toFixed(3);
    edDraw();
  }

  function edApply() {
    var p = ED.pts[ED.sel];
    ["X", "Y", "Z"].forEach(function (ax, k) {
      var v = parseFloat(($("ed" + ax) || {}).value);
      if (!isNaN(v)) p[k] = v;
    });
    edDraw();
  }

  /* 68 点的标准连线（iBUG 300-W 拓扑）—— 连起来才看得出哪条是睫毛、哪圈是唇 */
  var ED_EDGES = (function () {
    var e = [];
    function chain(a, b, closed) {
      for (var i = a; i < b; i++) e.push([i, i + 1]);
      if (closed) e.push([b, a]);
    }
    chain(0, 16);            // 下颌线
    chain(17, 20);           // 左眉
    chain(22, 25);           // 右眉
    chain(27, 29);           // 鼻梁
    chain(31, 34);           // 鼻底
    chain(36, 41, true);     // 左眼（闭合）
    chain(42, 47, true);     // 右眼（闭合）
    chain(48, 59, true);     // 外唇（闭合）
    chain(60, 67, true);     // 内唇（闭合）
    return e;
  })();

  /* 分组着色：眉眼唇用不同色调，方便分辨 */
  function edEdgeColor(a, b) {
    var lo = Math.min(a, b), hi = Math.max(a, b);
    if (lo >= 36 && hi <= 41) return "#ff5f5f";   // 左眼（睫毛在这一圈的上缘 36-39）
    if (lo >= 42 && hi <= 47) return "#ff5f5f";   // 右眼
    if (lo >= 48) return "#f0b03a";               // 唇
    if ((lo >= 17 && hi <= 21) || (lo >= 22 && hi <= 26)) return "#57c25e";  // 眉
    if (lo >= 27 && hi <= 35) return "#5b93ee";   // 鼻
    return "#9a9aa6";                             // 轮廓
  }

  function edDraw() {
    var cv = $("edCanvas");
    if (!cv || !ED.pts) return;
    var g = cv.getContext("2d");
    var W = cv.width, H = cv.height, S = 96, CAM = 4.2;
    g.clearRect(0, 0, W, H);

    var cy = Math.cos(ED.yaw), sy = Math.sin(ED.yaw);
    var cp = Math.cos(ED.pitch), sp = Math.sin(ED.pitch);
    var accent = getComputedStyle(document.documentElement).getPropertyValue("--ink-2").trim() || "#4F46E5";

    /* 地面网格，帮助判断深度 */
    g.strokeStyle = accent; g.globalAlpha = 0.08; g.lineWidth = 1;
    for (var gz = -1; gz <= 1; gz += 0.5) {
      g.beginPath();
      for (var gx = -1.4; gx <= 1.4; gx += 0.1) {
        var xx = gx * cy + gz * sy, zz = -gx * sy + gz * cy;
        var yy = -1.9 * cp - zz * sp, z2 = -1.9 * sp + zz * cp;
        var kk = CAM / (CAM - z2 * 0.9);
        var px = W / 2 + xx * S * kk, py = H / 2 - yy * S * kk;
        if (gx === -1.4) g.moveTo(px, py); else g.lineTo(px, py);
      }
      g.stroke();
    }
    g.globalAlpha = 1;

    var proj = ED.pts.map(function (p) {
      var x = p[0] * cy + p[2] * sy;
      var z = -p[0] * sy + p[2] * cy;
      var y = p[1] * cp - z * sp;
      var z2 = p[1] * sp + z * cp;
      var k = CAM / (CAM - z2 * 0.9);
      return { x: W / 2 + x * S * k, y: H / 2 - y * S * k, z: z2 };
    });

    /* 坐标轴 */
    [[1, 0, 0, "#e05252"], [0, 1, 0, "#4caf50"], [0, 0, 1, "#4f7ce0"]].forEach(function (ax) {
      var p2 = [ax[0] * 1.5, ax[1] * 1.5, ax[2] * 1.5];
      var x = p2[0] * cy + p2[2] * sy, z = -p2[0] * sy + p2[2] * cy;
      var y = p2[1] * cp - z * sp, z2 = p2[1] * sp + z * cp;
      var k = CAM / (CAM - z2 * 0.9);
      g.strokeStyle = ax[3]; g.globalAlpha = 0.5;
      g.beginPath();
      g.moveTo(W / 2, H / 2);
      g.lineTo(W / 2 + x * S * k, H / 2 - y * S * k);
      g.stroke();
    });
    g.globalAlpha = 1;

    /* 先连线：按两端平均深度排一遍，远处的淡、近处的实 */
    var lines = ED_EDGES.map(function (e) {
      var a = proj[e[0]], b = proj[e[1]];
      return { a: a, b: b, z: (a.z + b.z) / 2, c: edEdgeColor(e[0], e[1]) };
    }).sort(function (p, q) { return p.z - q.z; });
    lines.forEach(function (L) {
      var depth = Math.max(0, Math.min(1, (L.z + 0.8) / 1.6));
      g.strokeStyle = L.c;
      g.globalAlpha = 0.5 + depth * 0.5;
      g.lineWidth = 1.1 + depth * 1.1;
      g.beginPath();
      g.moveTo(L.a.x, L.a.y);
      g.lineTo(L.b.x, L.b.y);
      g.stroke();
    });
    g.globalAlpha = 1;

    proj.forEach(function (p, i) {
      var depth = Math.max(0, Math.min(1, (p.z + 0.6) / 1.2));
      g.beginPath();
      g.arc(p.x, p.y, i === ED.sel ? 6 : 2 + depth * 2.4, 0, Math.PI * 2);
      g.fillStyle = i === ED.sel ? "#e05252" : accent;
      g.globalAlpha = i === ED.sel ? 1 : 0.35 + depth * 0.6;
      g.fill();
    });
    g.globalAlpha = 1;

    /* 当前选中点标个序号，改坐标时能对上 */
    if (ED.sel >= 0 && proj[ED.sel]) {
      var sp2 = proj[ED.sel];
      g.fillStyle = "#e05252";
      g.font = "600 10px Inter, sans-serif";
      g.textAlign = "center";
      g.fillText(String(ED.sel), sp2.x, sp2.y - 10);
    }
  }

  function edBind() {
    var cv = $("edCanvas");
    if (!cv || cv.dataset.bound) return;
    cv.dataset.bound = "1";

    cv.addEventListener("mousedown", function (e) {
      ED.drag = { x: e.clientX, y: e.clientY, yaw: ED.yaw, pitch: ED.pitch, moved: false };
    });
    window.addEventListener("mousemove", function (e) {
      if (!ED.drag) return;
      var dx = e.clientX - ED.drag.x, dy = e.clientY - ED.drag.y;
      if (Math.abs(dx) + Math.abs(dy) > 3) ED.drag.moved = true;
      ED.yaw = ED.drag.yaw + dx * 0.01;
      ED.pitch = Math.max(-1.2, Math.min(1.2, ED.drag.pitch + dy * 0.01));
      edDraw();
    });
    window.addEventListener("mouseup", function (e) {
      if (!ED.drag) return;
      var moved = ED.drag.moved;
      ED.drag = null;
      if (moved) return;
      /* 没拖动 = 点选最近的点 */
      var r = cv.getBoundingClientRect();
      var mx = (e.clientX - r.left) * cv.width / r.width;
      var my = (e.clientY - r.top) * cv.height / r.height;
      var cy = Math.cos(ED.yaw), sy = Math.sin(ED.yaw);
      var cp = Math.cos(ED.pitch), sp = Math.sin(ED.pitch);
      var best = -1, bestD = 1e9;
      ED.pts.forEach(function (p, i) {
        var x = p[0] * cy + p[2] * sy, z = -p[0] * sy + p[2] * cy;
        var y = p[1] * cp - z * sp, z2 = p[1] * sp + z * cp;
        var k = 4.2 / (4.2 - z2 * 0.9);
        var px = cv.width / 2 + x * 96 * k, py = cv.height / 2 - y * 96 * k;
        var d = (px - mx) * (px - mx) + (py - my) * (py - my);
        if (d < bestD) { bestD = d; best = i; }
      });
      if (best >= 0 && bestD < 900) edSelect(best);
    });

    ["edX", "edY", "edZ"].forEach(function (id) {
      var el = $(id);
      if (el) el.addEventListener("input", edApply);
    });
    /* 三条轴都支持按住拖动改值 */
    if (window.edBindDrag) {
      window.edBindDrag($("edX"), { step: 0.006, min: -3, max: 3 });
      window.edBindDrag($("edY"), { step: 0.004, min: -2, max: 2 });
      window.edBindDrag($("edZ"), { step: 0.006, min: -3, max: 3 });
    }
    if ($("edReset")) $("edReset").addEventListener("click", function () {
      fetch("/api/face", { method: "DELETE" }).then(function () {
        FACE_PTS = null;
        HEAD = null;
        ED.pts = facePts().map(function (p) { return [p[0], p[1], p[2]]; });
        edSelect(ED.sel);
        toast("已重置", "回到内置坐标", "info");
      });
    });
    if ($("edSave")) $("edSave").addEventListener("click", function () {
      fetch("/api/face", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ points: ED.pts })
      }).then(function (r) { return r.json(); }).then(function (d) {
        if (!d || !d.ok) throw new Error("服务端拒绝");
        FACE_PTS = ED.pts.map(function (p) { return [p[0], p[1], p[2]]; });
        HEAD = null;
        toast("已保存", "隐藏模式的脸已更新", "success");
      }).catch(function (e) {
        toast("保存失败", String((e && e.message) || e), "error");
      });
    });
    if ($("edExport")) $("edExport").addEventListener("click", function () {
      var s = "[" + ED.pts.map(function (p) {
        return "[" + p.map(function (v) { return v.toFixed(4); }).join(",") + "]";
      }).join(",") + "]";
      if (navigator.clipboard) navigator.clipboard.writeText(s);
      console.log(s);
      toast("已复制", "68 点坐标在剪贴板里", "success");
    });
  }

  /* ---------- 隐私幕布：68 点人脸会跟着鼠标看 ---------- */
  /* ---------- 隐私幕布：3D 头部模型只用来算 68 个特征点 ---------- */
  var HEAD = null;

  /* 参数化头部曲面：u = 纬度（-1 下巴 .. 1 头顶），v = 经度（0 正面中央） */
  function surf(u, v) {
    var lat = u * Math.PI / 2, lon = v;
    var x = Math.cos(lat) * Math.sin(lon);
    var y = Math.sin(lat);
    var z = Math.cos(lat) * Math.cos(lon);

    var front = Math.max(0, Math.cos(lon));
    x *= 0.78 - 0.14 * front;

    if (u < 0) {
      var t = -u;
      x *= 1 - 0.52 * t * t;
      z *= 1 - 0.18 * t * t;
      y -= 0.05 * t * t;
    }
    /* 眉骨 / 颧骨 */
    if (u > 0.28 && u < 0.55 && front > 0.5) z += 0.05;
    if (u > -0.02 && u < 0.26 && Math.abs(x) > 0.30 && front > 0.4) z += 0.035;
    /* 鼻脊 */
    if (front > 0.5) {
      z += 0.16 * Math.exp(-Math.pow(x / 0.10, 2)) * Math.exp(-Math.pow((u + 0.03) / 0.40, 2));
      /* 眼窝 */
      z -= 0.07 * Math.exp(-(Math.pow((Math.abs(x) - 0.30) / 0.14, 2) + Math.pow((u - 0.13) / 0.16, 2)));
    }
    return { x: x * 0.86, y: y * 1.04, z: z * 0.88 };
  }

  /* 68 点三维坐标：由 InsightFace/FaceMesh 在真实人像上跑出来的结果，
     以瞳距为单位归一化，原点在双眼中心，+y 朝上、+z 朝前 */
  var L68 = [[-1.1981,0.0865,0.0770],[-1.1138,-0.3111,0.0832],[-1.0835,-0.6150,0.0795],[-0.9789,-0.8799,0.0690],[-0.8349,-1.1234,0.0574],[-0.6307,-1.3352,0.0409],[-0.4610,-1.5138,0.0293],[-0.2243,-1.6625,0.0176],[0.0206,-1.7874,0.0075],[0.2679,-1.6647,0.0177],[0.4995,-1.5145,0.0289],[0.6759,-1.3354,0.0416],[0.8877,-1.1216,0.0580],[1.0258,-0.8820,0.0693],[1.1289,-0.6249,0.0807],[1.1603,-0.3106,0.0825],[1.2480,0.0865,0.0770],[-0.9050,0.3341,-0.0081],[-0.7302,0.4627,-0.0230],[-0.5323,0.4844,-0.0317],[-0.5037,0.3836,-0.0249],[-0.2488,0.4128,-0.0338],[0.2986,0.4129,-0.0338],[0.5535,0.3843,-0.0249],[0.5822,0.4843,-0.0317],[0.7800,0.4606,-0.0230],[0.9549,0.3326,-0.0081],[0.0722,-0.0848,-0.0111],[-0.0162,-0.0824,0.0122],[0.0702,-0.4492,0.0130],[-0.0112,-0.4476,0.0363],[-0.0824,-0.7697,-0.0340],[-0.0755,-0.7916,-0.0339],[0.0249,-0.7898,0.0212],[0.1254,-0.7916,-0.0339],[0.1322,-0.7697,-0.0340],[-0.6782,-0.0794,-0.0093],[-0.5307,0.0572,-0.0121],[-0.4134,0.0603,-0.0119],[-0.3589,0.0099,-0.0109],[-0.4380,-0.0260,-0.0109],[-0.5806,-0.0583,-0.0114],[0.3193,-0.0083,-0.0100],[0.4494,0.0713,-0.0121],[0.5684,0.0617,-0.0122],[0.6405,-0.0219,-0.0109],[0.5784,-0.0391,-0.0110],[0.4439,-0.0276,-0.0107],[-0.1561,-1.1024,-0.0128],[-0.1501,-1.0488,-0.0252],[-0.0779,-1.0051,-0.0323],[0.0295,-0.9947,-0.0345],[0.1362,-1.0061,-0.0325],[0.2035,-1.0493,-0.0262],[0.2236,-1.0992,-0.0208],[0.2143,-1.1605,-0.0222],[0.1300,-1.2072,-0.0247],[0.0214,-1.2275,-0.0256],[-0.0972,-1.2054,-0.0243],[-0.2225,-1.1710,-0.0216],[-0.1513,-1.0911,-0.0130],[-0.0601,-1.0852,-0.0241],[0.0457,-1.0896,-0.0265],[0.1239,-1.0859,-0.0255],[0.1676,-1.1036,-0.0205],[0.1342,-1.1228,-0.0214],[0.0069,-1.1310,-0.0223],[-0.1511,-1.1280,-0.0207]];

  function landmarks() {
    if (HEAD) return HEAD;
    var src = facePts();
    /* 脸上重心偏下（下巴比眉低很多），往上提一点再压缩纵向，画面才居中 */
    var P = src.map(function (p) { return { x: p[0] * 0.92, y: (p[1] + 0.64) * 0.86, z: -p[2] * 3 }; });
    var groups = [
      [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16],
      [17,18,19,20,21], [22,23,24,25,26],
      [27,28,29,30], [31,32,33,34,35],
      [36,37,38,39,40,41,36], [42,43,44,45,46,47,42],
      [48,49,50,51,52,53,54,55,56,57,58,59,48],
      [60,61,62,63,64,65,66,67,60]
    ];
    HEAD = { pts: P, groups: groups };
    return HEAD;
  }

  function buildFace() {
    var box = $("cface");
    if (!box) return;

    var W = 320, H = 380, S = 118, CAM = 4.2;
    var cv = document.createElement("canvas");
    var dpr = window.devicePixelRatio || 1;
    cv.width = W * dpr; cv.height = H * dpr;
    cv.style.width = W + "px"; cv.style.height = H + "px";
    var g = cv.getContext("2d");
    g.scale(dpr, dpr);
    box.innerHTML = "";
    box.appendChild(cv);
    box.dataset.done = "1";

    var M = landmarks();
    var accent = getComputedStyle(document.documentElement).getPropertyValue("--ink-2").trim() || "#4F46E5";

    function render(yaw, pitch) {
      g.clearRect(0, 0, W, H);
      var cy = Math.cos(yaw), sy = Math.sin(yaw), cp = Math.cos(pitch), sp = Math.sin(pitch);

      var pts = M.pts.map(function (p) {
        var x = p.x * cy + p.z * sy;
        var z = -p.x * sy + p.z * cy;
        var y = p.y * cp - z * sp;
        var z2 = p.y * sp + z * cp;
        var k = CAM / (CAM - z2 * 0.9);
        return { x: W / 2 + x * S * k, y: H / 2 - y * S * k, z: z2, k: k };
      });

      /* 极淡的连线：只为了让脸成形，不抢点的注意力 */
      g.strokeStyle = accent;
      g.lineWidth = 1;
      g.globalAlpha = 0.14;
      M.groups.forEach(function (grp) {
        g.beginPath();
        grp.forEach(function (idx, j) {
          var p = pts[idx];
          if (j) g.lineTo(p.x, p.y); else g.moveTo(p.x, p.y);
        });
        g.stroke();
      });

      /* 68 个点：近的大而实，远的细而淡 */
      pts.forEach(function (p) {
        var depth = Math.max(0, Math.min(1, (p.z + 0.75) / 1.5));
        g.globalAlpha = 0.22 + depth * 0.78;
        g.beginPath();
        g.arc(p.x, p.y, 1.5 + depth * 2.1, 0, Math.PI * 2);
        g.fillStyle = accent;
        g.fill();
      });
      g.globalAlpha = 1;
    }
    render(0, 0);

    var raf = 0, mx = 0, my = 0;
    document.addEventListener("mousemove", function (e) {
      if (curtainHidden()) return;
      var r = box.getBoundingClientRect();
      mx = Math.max(-1, Math.min(1, (e.clientX - (r.left + r.width / 2)) / (window.innerWidth / 2)));
      my = Math.max(-1, Math.min(1, (e.clientY - (r.top + r.height / 2)) / (window.innerHeight / 2)));
      if (raf) return;
      raf = requestAnimationFrame(function () {
        raf = 0;
        render(mx * 0.55, my * 0.32);
      });
    });
  }

  function curtainHidden() {
    var c = $("curtain");
    return !c || c.style.display === "none";
  }

  function setCurtain(on) {
    var c = $("curtain");
    if (!c) return;
    c.style.display = on ? "grid" : "none";
    if (on) {
      /* 幕布只盖内容区：左边界跟着侧边栏走，不切断侧栏的标题行 */
      var side = document.querySelector(".side");
      if (side) c.style.left = Math.round(side.getBoundingClientRect().right) + "px";
    }
    if (on) {
      buildFace();
      var svg = document.querySelector("#cface svg");
      if (svg) svg.style.transform = "";
      if ($("cpass")) { $("cpass").value = ""; setTimeout(function () { $("cpass").focus(); }, 60); }
      if ($("cerr")) $("cerr").textContent = "";
    }
  }
  window.setCurtain = setCurtain;

  function hidePass() {
    try { return localStorage.getItem("dfl-hidepass") || ""; } catch (e) { return ""; }
  }

  function tryUnlock() {
    var want = hidePass();
    var got = ($("cpass") || {}).value || "";
    var err = $("cerr");
    if (!want) {
      if (err) err.textContent = "还没有设置密码，去「设置 → 隐私密码」里设一个";
      return;
    }
    if (got === want) { setCurtain(false); return; }
    if (err) err.textContent = "密码不对";
    if ($("cpass")) { $("cpass").value = ""; $("cpass").focus(); }
  }

  /* ---------- 账号 ---------- */
  var AUTH = { logged_in: false, name: "", onboarded: false };

  /* 头像：优先用 B 站返回的图片，失败或没有就退回首字母 */
  function paintAvatar(url, name) {
    var av = $("uAvatar");
    if (!av) return;
    var letter = (name || "用户").slice(0, 1).toUpperCase();
    function fallback() {
      av.textContent = letter;
      av.classList.remove("has-img");
    }
    var src = (url || "").trim();
    if (!src) { fallback(); return; }
    if (src.indexOf("//") === 0) src = "https:" + src;
    src = src.replace(/^http:/, "https:");
    var img = new Image();
    img.alt = "";
    img.referrerPolicy = "no-referrer";
    img.onload = function () {
      av.textContent = "";
      av.appendChild(img);
      av.classList.add("has-img");
    };
    img.onerror = fallback;
    img.src = src;
  }

  function applyAuth(a) {
    /* 允许无参调用：同意协议后要用"当前登录态"重绘一次。
       以前无参会把 __loggedIn 打成 false，于是又弹回扫码页。 */
    if (a) AUTH = a;
    a = AUTH;
    window.__loggedIn = !!(a && a.logged_in);
    window.__userName = (a && a.name) || "";
    /* 始终用保存的偏好：默认值是 auto，等价于"没选过就跟随系统" */
    var want = THEME.saved;
    if (want !== THEME.mode) {
      THEME.mode = want;
      applyTheme(resolveTheme());
      syncThemeSeg();
    }
    var box = $("userBox");
    if (box) {
      box.style.display = AUTH.logged_in ? "" : "none";
      if (AUTH.logged_in) {
        var nm = AUTH.name || "用户";
        $("uName").textContent = nm;
        paintAvatar(AUTH.avatar, nm);
      }
    }
    var w = $("welcome");
    if (w) {
      /* 未登录当然要看扫码页；已登录但还没同意协议的，也先停在这里读协议 ——
         协议现在长在欢迎页里，藏在主页面后面等于没给用户看。 */
      var needLegal = !!AUTH.logged_in && !(window.__legalAccepted && window.__legalAccepted());
      /* 未登录看扫码页；已登录但还没同意协议的，也先停在这里（协议卡盖在扫码页上） */
      var show = !AUTH.logged_in || needLegal;
      w.style.display = show ? "grid" : "none";
      /* 欢迎页一藏起来就停 shader，否则它会在后台空转、拖慢整机 */
      if (!show && window.__welcome3d && window.__welcome3d.stop) {
        try { window.__welcome3d.stop(); } catch (e) {}
      }
      if (show && !BILI.key) setTimeout(biliQr, 120);
      /* 退出登录后欢迎页重新出现：shader 之前被 stop() 掐了，得重新接上，
         否则登录页背景是一片静止的深色（看不到动画）。 */
      if (show && window.__welcome3d && window.__welcome3d.start) {
        try { window.__welcome3d.start(); } catch (e) {}
      }
    }
    if ($("hello") && AUTH.logged_in && AUTH.name) {
      $("hello").textContent = greeting() + "，" + AUTH.name;
    }
  }

  /* ---------- B 站扫码登录 ---------- */
  var BILI = { timer: 0, key: "", busy: false };

  window.applyAuth = applyAuth;

  function biliSetState(text) {
    if ($("wQrState")) $("wQrState").textContent = text;
  }

  function biliStop() {
    if (BILI.timer) { clearInterval(BILI.timer); BILI.timer = 0; }
    BILI.busy = false;
  }

  /* 二维码块状态：loading（占位）/ ready（可扫）/ expired（模糊+重试）/ error */
  function biliQrState(state, hintText) {
    var box = $("wQr"), hint = $("wQrHint"), img = $("wQrImg");
    if (box) box.className = "wqr" + (state === "ready" ? "" : " " + state);
    if (hint) {
      hint.style.display = state === "ready" ? "none" : "";
      if (hintText != null) hint.textContent = hintText;
    }
    if (img && state === "loading") { img.style.display = "none"; img.removeAttribute("src"); }
  }

  function biliQr() {
    biliStop();
    var img = $("wQrImg");
    biliQrState("loading", "正在获取…");
    biliSetState("正在向 B 站申请二维码…");

    fetch("/api/bili/qr", { method: "POST" })
      .then(function (r) { return r.json().then(function (d) { return { ok: r.ok, d: d }; }); })
      .then(function (x) {
        if (!x.ok || !x.d.qr) throw new Error(x.d.detail || "生成失败");
        BILI.key = x.d.key;
        if (img) { img.src = "data:image/png;base64," + x.d.qr; img.style.display = "block"; }
        biliQrState("ready");
        biliSetState("用手机 B 站 App 扫码，扫完在手机上确认。");
        BILI.timer = setInterval(biliPoll, 2000);
      })
      .catch(function (e) {
        biliQrState("error", "点这里重试");
        biliSetState("无法获取二维码：" + ((e && e.message) || e));
      });
  }

  function biliPoll() {
    if (BILI.busy || !BILI.key) return;
    BILI.busy = true;
    fetch("/api/bili/poll", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ key: BILI.key })
    }).then(function (r) { return r.json(); }).then(function (d) {
      BILI.busy = false;
      if (d.status === "ok") {
        biliStop();
        biliSetState("登录成功，欢迎 " + (d.name || "") + "…");
        fetch("/api/auth/login", {
          method: "POST", headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ uid: d.uid || 0, name: d.name || "用户", avatar: d.avatar || "" })
        }).then(function () {
          AUTH = { logged_in: true, name: d.name || "用户", onboarded: true };
          playLoginIntro(function () {
            applyAuth(AUTH);
            toast("登录成功", "欢迎，" + (d.name || ""), "success");
            loadState();
          });
        });
        return;
      }
      if (d.status === "expired" || d.status === "error") {
        biliStop();
        biliQrState("expired", "已过期 · 点这里刷新");
        biliSetState(d.message || "二维码已过期");
        return;
      }
      biliSetState(d.message || "等待扫码…");
    }).catch(function () { BILI.busy = false; });
  }
  if ($("wQr")) $("wQr").addEventListener("click", function () {
    var cls = this.className || "";
    if (cls.indexOf("expired") >= 0 || cls.indexOf("error") >= 0) biliQr();
  });
  window.biliQr = biliQr;

  /* 登录成功：欢迎页淡出，logo 缩小归位到侧边栏 */
  function playLoginIntro(done) {
    var w = $("welcome");
    var from = document.querySelector(".wlogo");
    var to = $("brandLogo");
    if (!w || !from || !to) { if (done) done(); return; }

    var a = from.getBoundingClientRect();
    var b = to.getBoundingClientRect();
    if (!b.width) { if (done) done(); return; }

    /* 克隆一枚 logo，从欢迎页的位置飞到侧边栏 */
    var fly = document.createElement("img");
    fly.className = "flyLogo";
    fly.src = from.src;
    fly.style.cssText = "position:fixed;left:" + a.left + "px;top:" + a.top + "px;width:" + a.width +
      "px;height:" + a.height + "px;border-radius:16px;z-index:400;pointer-events:none;will-change:transform,width,height,border-radius";
    document.body.appendChild(fly);

    /* 原本的 logo 先藏起来，落位时再出现 */
    to.style.transition = "opacity .18s ease";
    to.style.opacity = "0";

    /* 卡片内容整体上浮淡出 */
    var card = w.querySelector(".wcard");
    if (card) {
      card.style.transition = "opacity .28s ease, transform .32s cubic-bezier(.25,.46,.45,.94)";
      card.style.opacity = "0";
      card.style.transform = "translateY(-10px)";
    }

    /* 背景淡出，露出主界面 */
    w.style.transition = "background-color .42s ease .06s";
    w.style.backgroundColor = "transparent";

    requestAnimationFrame(function () {
      fly.animate([
        { left: a.left + "px", top: a.top + "px", width: a.width + "px", height: a.height + "px", borderRadius: "16px" },
        { left: b.left + "px", top: b.top + "px", width: b.width + "px", height: b.height + "px", borderRadius: "8px" }
      ], { duration: 520, easing: "cubic-bezier(.22,.75,.25,1)", fill: "forwards" });
    });

    setTimeout(function () {
      w.style.display = "none";
      w.style.backgroundColor = "";
      fly.remove();
      to.style.opacity = "1";
      if (card) { card.style.transition = ""; card.style.opacity = ""; card.style.transform = ""; }
      setTimeout(function () { to.style.transition = ""; }, 200);
      if (done) done();
    }, 540);
  }
  window.playLoginIntro = playLoginIntro;

  function authLogin(name) {
    fetch("/api/auth/login", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ uid: 258000000 + Math.floor(Math.random() * 999), name: name })
    }).then(function (r) { return r.json(); }).then(function (a) {
      AUTH = { logged_in: true, name: a.name, onboarded: true };
      playLoginIntro(function () {
        applyAuth(AUTH);
        toast("登录成功", "欢迎，" + a.name, "success");
        loadState();
      });
    });
  }

  function authSkip() {
    fetch("/api/auth/skip", { method: "POST" }).then(function () {
      AUTH.onboarded = true;
      applyAuth(AUTH);
    });
  }

  /* ---------- 通用确认弹层 ---------- */
  var CF = { ok: null, prev: null };

  function confirmBox(title, desc, okText, onOk, danger) {
    var box = $("confirmBox");
    if (!box) return;
    CF.ok = onOk;
    CF.prev = document.activeElement;
    if ($("cfTitle")) $("cfTitle").textContent = title;
    if ($("cfDesc")) $("cfDesc").textContent = desc || "";
    var okBtn = $("cfOk");
    if (okBtn) {
      okBtn.textContent = okText || "确定";
      okBtn.classList.toggle("danger", !!danger);
    }
    box.style.display = "grid";
    /* 焦点先给「取消」，避免回车误确认 */
    setTimeout(function () { if ($("cfCancel")) $("cfCancel").focus(); }, 30);
  }

  function closeConfirm() {
    var box = $("confirmBox");
    if (box) box.style.display = "none";
    CF.ok = null;
    if (CF.prev && CF.prev.focus) { try { CF.prev.focus(); } catch (e) {} }
    CF.prev = null;
  }

  function authLogout() {
    confirmBox(
      "退出登录？",
      "会清除本机的 B 站授权，下次使用需要重新扫码。训练和数据都不受影响。",
      "退出",
      function () {
            Promise.all([
          fetch("/api/auth/logout", { method: "POST" }),
          fetch("/api/bili/logout", { method: "POST" })
        ]).then(function () {
          BILI.key = "";
          applyAuth({ logged_in: false, name: "", onboarded: false });
          /* 退出登录 = 连"我同意过协议"一起作废：下次扫码进来必须重新读一遍。
             同时把浏览器侧的 cookie 也清掉（后端会删掉 cookie 文件，这里补前端那份）。 */
          try { localStorage.removeItem("dfl-legal-v3"); } catch (e) {}
          try {
            document.cookie.split(";").forEach(function (c) {
              var k = c.split("=")[0].trim();
              if (!k) return;
              document.cookie = k + "=; expires=Thu, 01 Jan 1970 00:00:00 GMT; path=/";
              document.cookie = k + "=; expires=Thu, 01 Jan 1970 00:00:00 GMT; path=/; domain=" + location.hostname;
            });
          } catch (e) {}
          toast("已退出登录", "B 站授权已清除", "info");
        });
      },
      true
    );
  }

  function loadAuth() {
    return fetch("/api/auth").then(function (r) { return r.json(); }).then(applyAuth).catch(function () {});
  }

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
    if ($("helloSub")) $("helloSub").textContent = desc;
    if ($("nextHint")) $("nextHint").textContent = label;
    var nb = $("nextBtn");
    if (nb) { nb.textContent = btn; nb.onclick = act; }

    /* 成绩单 */
    var totalIter = 0;
    list.forEach(function (m) { totalIter += (m.iter || 0); });
    if ($("stIter")) $("stIter").textContent = fmt(totalIter);
    if ($("stModels")) $("stModels").textContent = list.length;
    if ($("stFaces")) $("stFaces").textContent = fmt(src + dst);
    if ($("stExports")) $("stExports").textContent = exported;

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
    if ($("homeVer")) $("homeVer").textContent = "v" + (s.version || "");
    if (window.I18N) window.I18N.refresh();
  }

  function loadState() {
    fetch("/api/state").then(function (r) { return r.json(); }).then(function (s) {
      STATE = s;
      var v = el("aboutVer");
      if (v) v.textContent = s.version;
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

  /* ---------- 参数设置（照 4.0 training_config_page 的分组） ---------- */
  var CFG_GROUPS = [
    { t: "模型架构参数", note: "创建后不可修改", ro: true, rows: [
      ["模型名称", "name"], ["分辨率", "resolution"], ["archi", "archi"],
      ["e_dims", "e_dims"], ["ae_dims", "ae_dims"], ["d_dims", "d_dims"], ["d_mask_dims", "d_mask_dims"]
    ]},
    { t: "数据集参数", rows: [
      { k: "src", l: "src 人脸集", t: "path", v: "workspace\\data_src\\" },
      { k: "dst", l: "dst 人脸集", t: "path", v: "workspace\\data_dst\\" },
      { k: "face_type", l: "人脸类型", t: "sel", ro: true, o: ["whole_face", "full_face", "half_face", "midfull_face", "head"], v: "whole_face",
        d: "新建模型时按这一项确定，建完就固定在模型里" }
    ]},
    { t: "训练参数", rows: [
      { k: "batch_size", l: "批次大小", t: "num", v: "8", d: "每次训练的样本数量" },
      { k: "target_batch", l: "目标批次", t: "num", v: "0",
        d: "想要的等效批次；= 批次大小 × 累积次数。0 = 关闭累积" },
      { k: "target_iter", l: "目标迭代数", t: "num", v: "0", d: "0 表示无限训练" },
      { k: "use_bf16", l: "训练精度", t: "seg", o: ["bf16", "fp32"], v: "bf16" },
      { k: "gradient_checkpointing", l: "梯度检查点", t: "sw", v: 0, d: "省显存，略慢" },
      { k: "clipgrad", l: "梯度裁剪", t: "sw", v: 1, d: "防止梯度爆炸" },
      { k: "models_opt_on_gpu", l: "优化器放显存", t: "sw", v: 1, d: "关掉可省显存" },
      { k: "use_fast_generator", l: "快速生成器", t: "sw", v: 0 },
      { k: "write_preview_history", l: "写入预览历史", t: "sw", v: 0, d: "预览图存到 _history" },
      { k: "pretrain", l: "预训练", t: "sw", v: 0, d: "只练通用特征" },
      { k: "pretrain_single_decoder", l: "预训练单解码器", t: "sw", v: 0, d: "预训练时只用一个解码器" },
      { k: "crash_threshold", l: "崩溃阈值", t: "num", v: "0.0", d: "loss 超过它视为崩溃，不覆盖备份" },
      { k: "log_code_stats", l: "Inter code 分布日志", t: "num", v: "0", d: "0 = 关闭，每 N 迭代输出一次" },
      { k: "max_backups", l: "最大备份数", t: "num", v: "3" },
      /* 下面三项引擎一直在读、也一直写在 dat 里，但参数页以前没有对应控件 ——
         用户只能直接改 dat 才能动它们，改前的摘要也不显示，等于隐形参数。 */
      { k: "nan_guard", l: "NaN 梯度保护", t: "sw", v: 0,
        d: "每步扫一遍梯度，出现 NaN 就跳过这一步（约慢 4%）" },
      { k: "write_train_log", l: "训练日志 CSV", t: "sw", v: 0,
        d: "在模型目录写 <模型名>_train_log.csv" },
      { k: "eager_mode", l: "即时模式(XLA)", t: "sw", v: 0,
        d: "仅 SAEHD；需要 torch_xla，普通训练保持关闭" }
    ]},
    { t: "优化器参数", rows: [
      { k: "optimizer", l: "优化器", t: "seg", o: ["adam", "adabelief", "lion"], v: "adam" },
      { k: "lr", l: "学习率", t: "num", v: "0.00005" },
      { k: "lr_policy", l: "学习率策略", t: "seg", o: ["None", "CosineAnnealingLR"], v: "None" },
      { k: "lr_cos", l: "余弦周期", t: "num", v: "0", d: "0 = 关闭" },
      { k: "lr_total_steps", l: "学习率总步数", t: "num", v: "0", d: "0 = 关闭" },
      { k: "lr_plateau_factor", l: "平台衰减系数", t: "seg", o: ["1.0", "0.5", "0.7", "0.3"], v: "1.0", d: "1.0 = 关闭" },
      { k: "lr_plateau_patience", l: "平台耐心", t: "num", v: "0", d: "连续 N 轮不降才衰减" },
      { k: "lr_plateau_min_ratio", l: "平台最小比例", t: "num", v: "0.1" },
      { k: "lr_warmup_steps", l: "LR 预热步数", t: "num", v: "0", d: "0 = 关闭；前 N 步线性升温" },
      { k: "lr_dropout", l: "LR Dropout", t: "seg", o: ["n", "y", "cpu"], v: "n" },
      { k: "pia_cycle_power", l: "PIA 周期", t: "num", v: "1.0" }
    ]},
    { t: "GAN 训练参数", rows: [
      { k: "gan_power", l: "GAN 强度", t: "num", v: "0.0", d: "0 = 关闭 GAN" },
      { k: "gan_patch_size", l: "GAN 块大小", t: "num", v: "32" },
      { k: "gan_dims", l: "GAN 维度", t: "num", v: "16" }
    ]},
    { t: "补丁参数", note: "0 = 关闭", rows: [
      { k: "true_face_power", l: "真脸强度", t: "num", v: "0.0", d: "CodeDiscriminator 真脸损失权重" },
      { k: "face_style_power", l: "面部风格强度", t: "num", v: "0.0", d: "风格迁移中面部的权重" },
      { k: "bg_style_power", l: "背景风格强度", t: "num", v: "0.0", d: "风格迁移中背景的权重" }
    ]},
    { t: "数据增强参数", rows: [
      { k: "rotation_range", l: "旋转范围", t: "num", v: "10" },
      { k: "scale_range", l: "缩放范围", t: "num", v: "0.15" },
      { k: "t_range", l: "偏移范围", t: "num", v: "0.05" },
      { k: "random_warp", l: "随机扭曲", t: "sw", v: 1,
        d: "对输入图像做弹性形变、重建目标不变：让模型在五官被扭曲的情况下也能解码出正常形状的人脸，减少 dst 五官形状对 src 解码结果的影响" },
      { k: "random_hsv_power", l: "随机 HSV 强度", t: "num", v: "0.0",
        d: "0.0~0.3 · 随机调整输入的 HSV：让模型从输入就学习更丰富的色彩，解出的人脸颜色也更丰富" },
      { k: "ct_mode", l: "颜色迁移", t: "sel", o: ["none", "rct", "lct", "mkl", "mkl-m", "idt", "idt-m", "sot-m", "mix-m", "mt", "lut"], v: "none",
        d: "把 src 的颜色映射到 dst 上：通过改变输入数据，让模型学习调色后的 src 图像，生成更贴合 dst 色彩的人脸" },
      { k: "random_occlusion", l: "随机遮挡", t: "sw", v: 0,
        d: "随机遮挡输入图像的一部分、重建目标不变：让模型在信息缺失（被挡住）的情况下也能解码出人脸" },
      { k: "random_noise", l: "随机噪点", t: "sw", v: 0,
        d: "给输入图像随机加噪声、重建目标不变：让模型在噪点严重时也能解码出清晰的人脸" },
      { k: "random_color_alg", l: "偏色算法", t: "sel", o: ["none", "direct", "rct", "mt", "lct"], v: "none",
        d: "对输入图像随机偏色、重建目标不变：让模型无论输入什么颜色的人脸，都只解码出正常颜色的人脸" },
      { k: "random_color_power", l: "偏色强度", t: "num", v: "0.0", d: "0.0~1.0" },
      { k: "edge_enhance_power", l: "边缘增强", t: "num", v: "0.0", d: "0.0~20.0" },
      { k: "eyes_scale", l: "眼睛缩放", t: "num", v: "1.0", d: "0.25~2.0" }
    ]},
    { t: "冻结层参数", rows: [
      { k: "freeze_encoder", l: "冻结编码器", t: "sw", v: 0 },
      { k: "freeze_decoder_mask", l: "冻结遮罩解码器", t: "sw", v: 0 },
      { k: "freeze_inter_AB", l: "冻结 inter_AB", t: "sw", v: 0 },
      { k: "freeze_inter_B", l: "冻结 inter_B", t: "sw", v: 0 }
    ]}
  ];


  /* ------------------------------------------------------------
   * 冻结层：按架构只留相关的行
   *   df     → 冻结中间 / 冻结输出
   *   liae   → 冻结 inter_AB / 冻结 inter_B
   *   amp    → inter_AB + inter_B + 冻结源 + 冻结目标
   *   DFLarge / LIAELarge 在 4.1 已经移除，它们的参数（瓶颈 / A遮罩 / B遮罩 /
   *   共享 / 瓶颈B）一律不显示 —— 留着只会让人以为能开。
   * null = 所有架构都有。
   * ------------------------------------------------------------ */
  var FREEZE_ARCH = {
    freeze_encoder: null,
    freeze_decoder_mask: null,
    freeze_inter_AB: ["liae", "amp"],
    freeze_inter_B: ["liae", "amp"],
    freeze_inter: ["df"],
    freeze_decoder_dst: ["df"],
    freeze_inter_src: ["amp"],
    freeze_inter_dst: ["amp"],
    freeze_bottleneck: [],
    freeze_decoderA_mask: [],
    freeze_decoderB_mask: [],
    freeze_bottleneck_AB: [],
    freeze_bottleneck_B: []
  };

  function archOf(archi) {
    var a = String(archi || "").toLowerCase();      /* liae-ud / df-ud / amp */
    if (a.indexOf("liae") === 0) return "liae";
    if (a.indexOf("df") === 0) return "df";
    if (a.indexOf("amp") === 0) return "amp";
    return "";
  }

  function filterFreezeRows(root, archi) {
    if (!root) return;
    var arch = archOf(archi);
    if (!arch) return;                              /* 拿不到架构就别乱删 */
    Array.prototype.forEach.call(root.querySelectorAll(".cfgrow[data-k]"), function (row) {
      var k = row.getAttribute("data-k");
      if (!(k in FREEZE_ARCH)) return;
      var allow = FREEZE_ARCH[k];
      if (allow === null) return;
      if (allow.indexOf(arch) < 0) row.remove();
    });
    /* 清完如果那张卡空了，就连卡一起收掉 */
    Array.prototype.forEach.call(root.querySelectorAll(".card"), function (c) {
      if (!c.querySelector(".cfgrow")) c.remove();
    });
  }

  window.__filterFreezeRows = filterFreezeRows;   /* mount() 在另一段 IIFE 里 */

  /* 训练中：给能热改的行挂一个「热」标记（只在这个模型真的在训练时才显示） */
  function markHotRows() {
    var box = $("cfgBox");
    if (!box) return;
    var m = window.__cfgModel;
    var lt = window.__lastTrain;
    var live = !!(lt && lt.running && m && lt.name === m.name);
    /* 训练在跑时主按钮的语义变了：不是「开始训练」，而是「应用参数」 */
    var startBtn = $("cfgStart");
    if (startBtn) startBtn.textContent = live ? "应用参数" : "开始训练";
    box.querySelectorAll(".cfgrow").forEach(function (row) {
      var lb = row.querySelector(".cl");
      if (!lb) return;
      /* 每一行都留一个固定宽的槽：有没有「热」都占同样的位置，
         这样说明文字的左边缘在整页是一条竖线（不然只有带热字的行被顶出去）。 */
      var slot = row.querySelector(".hotslot");
      if (!slot) {
        slot = document.createElement("span");
        slot.className = "hotslot";
        lb.insertAdjacentElement("afterend", slot);
      }
      var k = row.getAttribute("data-k");
      var tag = slot.querySelector(".hottag");
      var want = !!(live && window.HOT_KEYS && k && window.HOT_KEYS[k]);
      if (!want) {
        if (tag) tag.remove();
        row.classList.remove("hot");
        return;
      }
      row.classList.add("hot");
      if (!tag) {
        var s = document.createElement("span");
        s.className = "hottag";
        s.textContent = "热";
        s.title = "训练中改这一项会立即生效";
        slot.appendChild(s);
      }
    });
  }
  window.markHotRows = markHotRows;

  function openCfg(m) {
    var box = $("cfgBox");
    if (!box) return;
    window.__cfgModel = m || window.__cfgModel || null;   /* 冻结层过滤要知道当前架构 */
    window.__cfgArchi = (m && m.archi) || window.__cfgArchi || "";

    /* 补充参数不再塞进 CFG_GROUPS：TrainExtra.mount() 会把它们并进同名卡片，
       这里只管渲染原有的分组，避免同一批参数出现两遍。 */

    box.innerHTML = "";

    /* 上一个模型的迁移提示不能留在页面上：先清掉，等 /config 回来再按这次的结果画。
       （新格式 dat 的 legacy.detected=false，一路清空，页面上什么都不多出来。） */
    try {
      if (window.LegacyNotice) window.LegacyNotice.clear(document.getElementById("cfgLegacy"));
    } catch (e) { /* 提示层失败不能挡住参数页 */ }

    CFG_GROUPS.forEach(function (g) {
      /* 标了 ro 的分组（模型架构参数）内容和顶部"模型结构"只读卡完全重复：
         名字/分辨率/archi/e_dims/ae_dims/d_dims/d_mask_dims 一模一样。
         以前两处都画，页面上连着两块一样的信息，看着就是"混乱"。 */
      if (g.ro) return;
      var card = document.createElement("div");
      card.className = "card";
      var hd = document.createElement("div");
      hd.className = "cardhd";
      hd.innerHTML = "<b>" + g.t + "</b>" + (g.note ? '<span class="hint">' + g.note + "</span>" : "");
      card.appendChild(hd);

      (g.rows || []).forEach(function (r) {
        var row = document.createElement("div");
        row.className = "cfgrow" + ((g.ro || (!Array.isArray(r) && r.ro)) ? " ro" : "");

        /* 每行都带上 data-k：既给搜索/定位用，也让补齐块能按锚点插入
           （之前行上没有 data-k，after:"dst" 找不到锚点，只能落到卡片末尾）。 */
        if (r.k) row.setAttribute("data-k", r.k);

        if (!Array.isArray(r) && r.t === "path") {
          row.className += " cfgpathrow";
          /* 值用 <input>：路径行本身就该能直接改字（以前是 <span>，
             点上去没反应，看起来像"编辑框但打不了字"）。 */
          row.innerHTML = '<div class="inp mono pathinp cfgpath" style="flex:1 1 auto;width:auto;min-width:0">' +
            '<span class="lb">' + r.l + "</span>" +
            '<input class="pathval" spellcheck="false" autocomplete="off">' +
            '<span class="pill iconpill" title="选择目录"><svg class="ic"><use href="#i-folder"/></svg></span>' +
            "</div>";
          var pv = row.querySelector(".pathval");
          pv.value = r.v || "";
          if (r.k) pv.dataset.cfg = r.k;
          row.querySelector(".iconpill").addEventListener("click", function () {
            toast("选择目录", "文件选择器还没接", "info");
          });
          card.appendChild(row);
          return;
        }

        if (Array.isArray(r)) {
          var val = m ? (m[r[1]] || "—") : "—";
          row.innerHTML = '<span class="cl">' + r[0] + '</span><span class="cv">' + val + "</span>";
          card.appendChild(row);
          return;
        }

        row.innerHTML = '<span class="cl">' + r.l + "</span>" +
          '<span class="cd">' + (r.d || "") + "</span>" +
          '<span class="cc"></span>';
        var cc = row.querySelector(".cc");

        if (r.t === "num") {
          var inp = document.createElement("input");
          inp.className = "inp";
          inp.value = r.v;
          if (r.k) inp.dataset.cfg = r.k;
          inp.dataset.v0 = String(inp.value);
          cc.appendChild(inp);
        } else if (r.t === "seg") {
          var seg = document.createElement("div");
          seg.className = "seg";
          if (r.k) seg.dataset.cfg = r.k;          /* 之前漏了这行：seg 既不被回填也不进 patch */
          r.o.forEach(function (o, i) {
            var b = document.createElement("b");
            b.textContent = o;
            b.dataset.v = o;                       /* 业务值取这个，别取 textContent（会被翻译） */
            if (i === 0) { b.className = "on"; seg.dataset.v0 = String(o); }
            b.addEventListener("click", function () {
              Array.prototype.forEach.call(seg.children, function (x) { x.classList.remove("on"); });
              b.classList.add("on");
            });
            seg.appendChild(b);
          });
          cc.appendChild(seg);
        } else if (r.t === "sel") {
          var sel = document.createElement("select");
          sel.className = "mgsel";
          if (r.k) sel.dataset.cfg = r.k;
          (r.o || []).forEach(function (o) {
            var op = document.createElement("option");
            /* o 可以是字符串，也可以是 { v, l }：显示名与存的值分开 */
            var ov = (o && typeof o === "object") ? o.v : o;
            var ol = (o && typeof o === "object") ? o.l : o;
            op.value = String(ov);
            op.textContent = ol;
            sel.appendChild(op);
          });
          cc.appendChild(sel);
          /* 选中值要在**挂到文档之后**设：先设 value 再 append，浏览器会把选中态重置掉，
             结果是下拉框空着、提交时 face_type 变空串。 */
          sel.value = String(r.v);
          if (sel.selectedIndex < 0) sel.selectedIndex = 0;
          sel.dataset.v0 = String(sel.value);
          if (r.ro) {
            /* 结构性参数（face_type）：值照样从 dat 回显，但不许改 ——
               以前这里是可点的下拉框，点了保存却永远存不进去（假象）。 */
            sel.disabled = true;
            sel.dataset.ro = "1";
            sel.title = "创建后不可改：要换人脸类型只能新建模型";
          }
        } else if (r.t === "sw") {
          var sw = document.createElement("div");
          sw.className = "sw" + (r.v ? " on" : "");
          if (r.k) sw.dataset.cfg = r.k;
          sw.dataset.v0 = sw.classList.contains("on") ? "1" : "0";
          /* data-bound 必须打上：TrainExtra.mount() 会给「所有没绑过的 .sw」再挂一个
             click → 两个监听各 toggle 一次，等于抵消 —— 表现就是"开关拨不动"。 */
          sw.setAttribute("data-bound", "1");
          sw.addEventListener("click", function () { sw.classList.toggle("on"); });
          cc.appendChild(sw);
        }
        card.appendChild(row);
      });

      box.appendChild(card);
    });

    /* 结构卡在这里补一次：以前只靠 window.openCfg 的外层包装，
       而那个包装是在脚本加载时就抓走的 —— 只要 openCfg 定义得比包装晚，
       或者别的模块又包了一层，结构卡就不会出现（页面上只剩搜索框 + 参数卡）。
       现在由 openCfg 自己负责，谁调用都有效。 */
    try { filterFreezeRows(box, (m && m.archi) || window.__cfgArchi || ""); } catch (e) { /* 过滤失败就全留着 */ }

    try { if (window.__enhanceSelects) window.__enhanceSelects(box); } catch (e) { /* 增强失败不影响页面 */ }

    try {
      if (typeof nmReadonlyPanel === "function") {
        nmReadonlyPanel(m || window.__cfgModel || (window.MODELS || [])[0] || null);
      }
    } catch (e) { /* 结构卡失败不影响参数页 */ }

    var sub = $("cfgSub");
    if (sub) sub.textContent = m ? ("正在配置 " + m.name + "，确认后开始训练。") : "训练前确认参数。";
    if ($("cfgHint") && m) $("cfgHint").textContent = "当前 iter " + fmt(m.iter) + " · " + (m.archi || "");
    if (window.I18N) window.I18N.refresh();
    go("train-cfg");

    /* 回填一批控件：opts = /config 返回的 dat options。
       onlyNew=true 时只填「还没有基线」的行 —— TrainExtra 的「设备」卡是
       openCfg 之后 80ms 才挂上来的（见 mount 里的 setTimeout），这批行在第一次
       回填时还不存在，不补一次的话它们永远停在写死的默认值上：
       dat 里 device="auto"，下拉框却一直显示 "0"，用户以为改完没生效。 */
    function fillCfgRows(opts, onlyNew) {
      if (!opts) return 0;
      var filled = 0;
      box.querySelectorAll("[data-cfg]").forEach(function (el) {
        if (onlyNew && el.dataset.v0 !== undefined) return;
        var k = el.dataset.cfg, v = opts[k];
        if (k === "target_batch" && (v === undefined || v === null || Number(v) === 0)) {
          var _b0 = Number(opts.batch_size) || 1, _a0 = Number(opts.accum_steps) || 1;
          v = _a0 > 1 ? _b0 * _a0 : 0;
        }
        if (v === undefined) {
          /* dat 里没有这个键（老模型没存过 / 新建时没写）：
             标记一下，别给它记基线 —— 否则用户改成界面默认值会被判成"没改"，
             这个值永远存不进模型（"最大备份数填 3 不生效"就是这么来的）。 */
          el.dataset.missing = "1";
          return;
        }
        delete el.dataset.missing;
        filled++;
        if (el.tagName === "INPUT") el.value = v;
        else if (el.tagName === "SELECT") {
          /* 短码必须照 4.0 引擎的原表（Model_SAEHD:195 就是 ["h","mf","f","wf","head"]）。
             以前这里写的是 wf/ff/hf：face_type="f" 的模型在框里显示成第一项
             whole_face —— 只读行显示错的值比空白更误导。ff/hf 留着容忍早期写过的值。 */
          var _ALIAS = { h: "half_face", mf: "midfull_face", f: "full_face",
                         wf: "whole_face", head: "head",
                         ff: "full_face", hf: "half_face" };
          var sv = String(v);
          el.value = sv;
          if (el.selectedIndex < 0 && _ALIAS[sv.toLowerCase()]) el.value = _ALIAS[sv.toLowerCase()];
          /* 设备：必须在下面那句兜底之前过一遍守卫 —— 一旦 selectedIndex<0 被改成 0，
             「存的是 GPU 1 但本机只有 GPU 0」就变成了「用户选了自动」，看不出来了。 */
          if (el.dataset.cfg === "device" && window.__devGuard) window.__devGuard(el, sv);
          if (el.selectedIndex < 0) el.selectedIndex = 0;
        }
        else if (el.classList.contains("sw")) el.classList.toggle("on", !!v);
        else if (el.classList.contains("seg")) {
          /* 按 data-v 比，不看 textContent —— 切到英文后文案变了，比文本永远不匹配 */
          var hit = false;
          Array.prototype.forEach.call(el.children, function (b) {
            var bv = String(b.dataset.v !== undefined ? b.dataset.v : b.textContent);
            /* 数字要按数值比：dat 里存的是 float 1.0，分段控件的 token 是字符串 "1.0"，
               直接比字符串会一个都不亮（表现是这一行看着"没选中"）。 */
            var same = bv === String(v) ||
              (bv !== "" && String(v).trim() !== "" &&
               isFinite(Number(bv)) && isFinite(Number(v)) && Number(bv) === Number(v));
            if (same) hit = true;
            b.classList.toggle("on", same);
          });
          /* 存的是布尔（比如 use_bf16 -> 1/0），按 开/关 兜一次 */
          if (!hit && el.children.length === 2) {
            var onIdx = (v === true || v === 1 || v === "1") ? 0 : 1;
            Array.prototype.forEach.call(el.children, function (b, i) {
              b.classList.toggle("on", i === onIdx);
            });
          }
        }
      });

      /* 「已读入 N 项」按当前页面上真实填上的行数算：补充块（TrainExtra）的行
         可能比这次回填晚到，晚到的那批由下面的 __fillCfgRows 再补一次。 */
      var hint = document.getElementById("cfgHint");
      if (hint) {
        /* 先记下原始文案：openCfg 每重画一次都会拉一遍配置，直接往后追加会叠成
           "已读入 46 项真实参数 · 已读入 46 项真实参数"。 */
        if (!hint.dataset.base) {
          hint.dataset.base = (hint.textContent || "").replace(/\s*·\s*已读入 \d+ 项真实参数\s*$/, "");
        }
        var total = 0;
        box.querySelectorAll("[data-cfg]").forEach(function (el) {
          if (el.dataset.missing !== "1" && el.dataset.v0 !== undefined) total++;
        });
        hint.textContent = hint.dataset.base + " · 已读入 " + Math.max(filled, total) + " 项真实参数";
      }
      return filled;
    }
    window.__fillCfgRows = fillCfgRows;   /* TrainExtra 补挂的行要用它再填一次 */
    window.__cfgOpts = null;

    /* 拉真实的 *_data.dat options 回填 —— 页面上的值不再是写死的默认值 */
    if (m) {
      fetch("/api/models/" + encodeURIComponent(m.name) + "/config")
        .then(function (r) { return r.ok ? r.json() : null; })
        .then(function (d) {
          if (!d || !d.options) return;
          var opts = d.options;
          window.__cfgOpts = opts;         /* 给晚挂的行（设备卡）留一份 */
          fillCfgRows(opts, false);
          /* 回填完真实参数，把「初始值」重记一遍：之后只有用户真改过的才算改过 */
          if (window.syncCfgBaseline) window.syncCfgBaseline();

          /* TF 时代（DFL 1.x/2.0）的 dat：后端（server/_legacy.py）已经把 options 翻成
             4.1 的键，这里只负责把「迁了几项、还要不要转权重」放在页面顶部，
             一行 + 可展开明细。新格式 dat 传进来的 legacy.detected=false，这行不会出现。 */
          try {
            if (window.LegacyNotice) {
              window.LegacyNotice.render(document.getElementById("cfgLegacy"), d.legacy);
            }
          } catch (e) { console.warn("legacy notice failed", e); }

          /* 设备下拉的选项是 /api/devices 回来后才填的，可能在这次回填之后：
             等它到货时把「还没有基线」的行再填一遍（否则设备行永远停在默认值）。 */
          setTimeout(function () {
            var n = fillCfgRows(window.__cfgOpts, true);
            if (n && window.syncCfgBaseline) window.syncCfgBaseline();
          }, 0);
        })
        .catch(function (e) { console.warn("config load failed", e); });
    }

    /* 补充参数并由 TrainExtra.mount() 并入同名卡片。
       以前是靠包装 window.openCfg 来实现的，但那份包装没生效（__trainExtra 一直是 false），
       于是每次重渲染都把补充项弄丢。直接在这里调，顺序上不可能错。 */
    if (window.TrainExtra && typeof window.TrainExtra.mount === "function") {
      try { window.TrainExtra.mount(); } catch (e) {}
    }
    /* 补充区的行是上面 mount() 才插进来的，第一次标记扫不到 —— 渲染完再补一次 */
    setTimeout(function () { if (window.markHotRows) window.markHotRows(); }, 0);
  }
  window.openCfg = openCfg;

  /* ---------- 训练页控制按钮：暂停 / 继续 · 立即保存 · 停止 ---------- */
  function trainCtl(path, okMsg) {
    /* 带上当前模型名：后端会核对"这是不是正在训练的那个"，
       避免界面切到别的模型后点暂停，把另一个训练不声不响停掉。 */
    var _m = (window.__lastTrain && window.__lastTrain.name)
          || (window.__cfgModel && window.__cfgModel.name) || "";
    return fetch(path, {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name: _m })
    })
      .then(function (r) { return r.json().then(function (d) {
        if (!r.ok) throw new Error(d.detail || "请求失败");
        return d;
      }); })
      .then(function (d) {
        if (okMsg) toast(okMsg, String((d && d.msg) || "").slice(0, 80), "success");
        return d;
      })
      .catch(function (e) { toast("操作失败", String((e && e.message) || e).slice(0, 120), "error"); });
  }

  (function bindRunButtons() {
    document.addEventListener("click", function (e) {
      window.__runClick = (window.__runClick || 0) + 1;
      var t = e.target && e.target.closest ? e.target : null;
      window.__runClickId = t && t.id;
      if (!t || !t.id) return;
      if (t.id === "runPause") {
        var paused = t.textContent.trim() === "继续";
        trainCtl(paused ? "/api/train/resume" : "/api/train/pause",
                 paused ? "已继续" : "已暂停");
        t.textContent = paused ? "暂停" : "继续";
      } else if (t.id === "runSave") {
        t.disabled = true;
        var oldTxt = t.textContent;
        t.textContent = "保存中…";
        trainCtl("/api/train/save", "已请求保存").then(function (d) {
          t.textContent = (d && d.ok) ? "已保存" : oldTxt;
          setTimeout(function () { t.disabled = false; t.textContent = oldTxt; }, 2000);
        });
      } else if (t.id === "runStop") {
        /* 用应用自己的弹窗：原生 confirm 在 Electron 里是系统对话框，
           和整站风格完全不搭（而且键盘焦点会丢）。 */
        confirmBox("停止训练？", "会先让训练进程保存再退出，可能要等十几秒。",
          "停止训练", function () {
            t.disabled = true;
            t.textContent = "停止中…";
            trainCtl("/api/train/stop", "训练已停止").then(function () {
              t.disabled = false;
              t.textContent = "停止";
            });
          }, true);
      }
    });
  })();

  /* ---------- 训练中（骨架：数据先用真实模型 + 模拟值） ---------- */
  /* 训练状态：真进程在跑就按它刷新，没跑就显示模型里最后的 iter */
  var TRAIN_POLL = null;
  function _num(n) { return Number(n || 0).toLocaleString("en-US"); }

  /* 训练状态的唯一来源：徽章、圆点、状态条都读它，避免三处各写一套对不上。
     返回 { text, tone }，tone ∈ blue / green / yellow / red / gray */
  function trainState(d) {
    if (!d) return { text: "连接中", tone: "blue" };
    if (!d.running) {
      return d.ok === false ? { text: "训练失败", tone: "red" }
                            : { text: "已停止", tone: "gray" };
    }
    if (d.stopping) return { text: "正在停止", tone: "red" };
    if (d.paused) return { text: "已暂停", tone: "yellow" };
    if (!d.iter_seen) return { text: "加载中", tone: "blue" };
    return { text: "训练中", tone: "green" };
  }

  /* 显存估算（参考值，误差 ±15%）。
     模型部分按 参数×字节 精确算，激活按 128px/batch8 的实测值线性缩放，
     再加上 cuDNN workspace / CUDA context / 碎片这三项测不出来的固定开销。 */
  /* 显存估算（参考值，误差 ±15%）。
     锚定在一个实测点上：liae-ud / ae256 / e64 / d64 / bf16 / batch8 / R128
     真实训练峰值 = 5.7 GB（nvidia-smi 采样）。分成三块：
       静态  1.26 GB（权重+梯度+优化器动量，实测模型 78.6M 参数 × 16B）
       固定  2.5  GB（cuDNN workspace + CUDA context + 分配器碎片，测不出来）
       激活  1.94 GB（随 batch 线性、随分辨率平方）
     参数个数不去公式推算 —— SAEHD 的 conv 栈宽度和 e_dims 不是线性关系，
     硬推会得出 "2M 参数" 这种离谱数字（真实 78.6M）。 */
  /* 显存估算。
     旧公式把「权重/梯度/优化器状态」乘了 (resolution/128)^2 —— 完全错了：
     分辨率不决定参数量（参数量只由 ae/e/d/d_mask 和架构决定），只影响激活。
     结果 512/ae512 的模型被估成 46.7 GB（实际 8.9 GB），吓得不敢建。
     新公式用「参数量推固定开销 + 经验激活项」，用实测值标定：
       512/ae512/e64/d128/bs1/检查点  -> 实际 8.9 GB
       640/ae512/e64/d128/bs1          -> 实际 11.8 GB
       1024/ae256/e128/d128/bs1/单解码 -> 实际 18.0 GB
     误差约 ±30%，比原来的 5 倍偏差强得多。 */
  function vramEstimate(o) {
    var R = Number(o.resolution) || 128;
    var B = Number(o.batch_size) || 1;
    var ae = Number(o.ae_dims) || 256;
    var eD = Number(o.e_dims) || 64;
    var dD = Number(o.d_dims) || 128;
    var ckpt = !!(o.gradient_checkpointing || o.use_gradient_checkpointing);
    var archi = String(o.archi || "");
    var dmD = Number(o.d_mask_dims) || 32;
    var flags = String(o.archi || "df-udt").toLowerCase();
    var hasT = flags.indexOf("t") >= 0, hasD = flags.indexOf("d") >= 0;
    // ★ 分辨率确实影响参数量，而且是通过 inter 的两个 Dense 层**平方级**影响：
    //   out_res   = R / (32 if -t else 16)     ← encoder 展平后的空间尺寸
    //   lowest    = R / (32 if -d else 16)     ← inter 输出的最低分辨率
    //   dense1 = (e_ch*8 * out_res^2) * ae
    //   dense2 = ae * (lowest^2 * ae)
    // 例：1024 且不带 -t 时 out_res=64，dense1 直接到 10.7 亿参数（就是那个 1073.7M 的张量）。
    var e8 = eD * 8;
    var outRes = Math.floor(R / (hasT ? 32 : 16));
    var lowest = Math.floor(R / (hasD ? 32 : 16));
    var interM = (e8 * outRes * outRes * ae + ae
                  + ae * lowest * lowest * ae + lowest * lowest * ae) / 1e6;
    // encoder + 两个 decoder：与分辨率无关（卷积核/通道数固定），只随 e / d / mask 变
    var encM = 15.65 * Math.pow(Math.max(eD, 16) / 64, 2);
    var decM = 2 * (123.9 * Math.pow(Math.max(dD, 16) / 128, 2)
                    + 8.63 * Math.pow(Math.max(dmD, 8) / 32, 1.5));
    var paramsM = interM + encM + decM;
    // 每个参数占的字节：权重 4 + 梯度 4 + 优化器状态（AdaBelief/Adam 8，Lion 4）
    var opt = String(o.optimizer || "adabelief").toLowerCase();
    var optB = (opt === "lion") ? 4 : 8;
    var perParam = 4 + 4 + optB;
    // 单解码器支路：dst 解码器不参与前向/反向，梯度与状态只算 src 那半边
    var singleDec = !!(o.pretrain && o.pretrain_single_decoder);
    var fixedGb = paramsM * perParam / 1024 * (singleDec ? 0.72 : 1.0);
    var actGb = 1.0 * Math.pow(R / 512, 2) * Math.max(B, 1)
                    * Math.sqrt(Math.max(ae, 32) / 512) * (ckpt ? 1.0 : 2.2);
    var sysGb = 0.9;
    return { total: fixedGb + actGb + sysGb, act: actGb, stat: fixedGb + sysGb };
  }
  window.vramEstimate = vramEstimate;

  /* 显卡总量来自现成的 /api/gpu（routes_models.py）。只取一次，缓存在 window。 */
  function loadGpuInfo(cb) {
    if (window.GPU_INFO) { cb(window.GPU_INFO); return; }
    fetch("/api/gpu").then(function (r) { return r.json(); }).then(function (d) {
      var g = (d && d.gpus && d.gpus[0]) || null;
      window.GPU_INFO = g;
      cb(g);
    }).catch(function () { window.GPU_INFO = null; cb(null); });
  }
  window.loadGpuInfo = loadGpuInfo;

  /* 从参数页读当前值：键名是 data-cfg（和 collectCfgPatch 同一套），
     值可能是 input / select / .sw（开关）/ .seg（分段）四种形态。 */
  window.CFG_VALS = function () {
    var out = {};
    var box = document.getElementById("cfgBox");
    if (!box) return out;
    box.querySelectorAll("[data-cfg]").forEach(function (el) {
      var k = el.dataset.cfg;
      if (!k) return;
      if (el.tagName === "INPUT") out[k] = el.value;
      else if (el.tagName === "SELECT") out[k] = el.value;
      else if (el.classList.contains("sw")) out[k] = el.classList.contains("on");
      else if (el.classList.contains("seg")) {
        var on = el.querySelector("b.on");
        if (on) out[k] = (on.dataset.v !== undefined ? on.dataset.v : on.textContent.trim());
      }
    });
    return out;
  };

  function paintVram() {
    var box = document.getElementById("cfgVram");
    if (!box) return;
    var o = (window.CFG_VALS && window.CFG_VALS()) || {};
    /* 参数页没有 resolution 这一行（分辨率在新建模型时就定了，之后改不了），
       只能从当前打开的模型里取；取不到就按 128 估，并标注出来。 */
    var m = window.__cfgModel || {};
    if (!o.resolution) o.resolution = m.resolution || m.res || 0;
    var assumed = false;
    if (!o.resolution) { o.resolution = 128; assumed = true; }
    /* ★ 新建模型时，分辨率和 ae 维度在「新建模型 · 结构」卡里，不在 cfgBox 的
       参数行上（那些行没有 data-cfg）。以前这里读不到就退回上一个/第一个模型的
       resolution —— 于是明明要建 256 的模型，底栏却按 384 估出 "22.6 GB 放不下"，
       跟同一行提示"建议 256 / batch 8，约 3.5 GB"自相矛盾，吓得人不敢建。 */
    if (document.getElementById("nmStruct")) {
      var _nr = document.getElementById("nmRes");
      var _na = document.getElementById("nmAe");
      if (_nr && _nr.value) { o.resolution = Number(_nr.value); assumed = false; }
      if (_na && _na.value) { o.ae_dims = Number(_na.value); }
    }
    box.style.display = "";
    var r = vramEstimate(o);
    var gpu = window.GPU_INFO || {};
    if (!window.GPU_INFO) loadGpuInfo(function () { paintVram(); });
    var total = gpu.total_gb || 0;
    var pct = total ? (r.total / total) * 100 : 0;
    box.className = "vram" + (total && r.total > total ? " bad" : (total && pct > 82 ? " warn" : ""));
    box.innerHTML = "预计显存 <b>" + r.total.toFixed(1) + " GB</b>" +
      (total ? " / 显卡 " + total.toFixed(1) + " GB" : "") +
      (total && r.total > total ? " <b style=\"color:var(--bad)\">放不下，请降低分辨率或批次</b>" : "") +
      (assumed ? " <span style=\"color:var(--ink-3)\">按 128px 估</span>" : "") +
      "<span class=\"vh\">估算值（±30%），训练页显示的才是实测占用</span>";
  }
  window.paintVram = paintVram;

  /* paintVram 的全局刷新：参数页任何输入/分段切换后重新估算 */
  document.addEventListener("input", function () {
    if (document.getElementById("cfgVram")) window.paintVram();
  }, true);
  document.addEventListener("click", function (e) {
    var t = e.target;
    if (t && t.closest && t.closest("#cfgBox .seg")) setTimeout(function () { window.paintVram(); }, 0);
  }, true);

  window.trainState = trainState;

  /* 训练页的常驻失败横幅。toast 会自己消失，而"要改配置"类的错误必须一直摆着。 */
  function showRunAlert(title, body, actions) {
    var box = document.getElementById("runAlert");
    if (!box) return;
    var t = document.getElementById("runAlertTitle");
    var b = document.getElementById("runAlertBody");
    var a = document.getElementById("runAlertAct");
    if (t) t.textContent = title || "训练已停止";
    if (b) b.textContent = String(body || "").slice(0, 600);
    if (a) {
      a.innerHTML = "";
      (actions || []).forEach(function (it) {
        var btn = document.createElement("button");
        btn.className = "mini";
        btn.textContent = it.label;
        btn.addEventListener("click", it.onClick);
        a.appendChild(btn);
      });
    }
    box.classList.remove("hide");
  }
  window.showRunAlert = showRunAlert;

  function hideRunAlert() {
    var box = document.getElementById("runAlert");
    if (box) box.classList.add("hide");
  }
  window.hideRunAlert = hideRunAlert;
  (function () {
    var x = document.getElementById("runAlertX");
    if (x) x.addEventListener("click", hideRunAlert);
  })();

  function pollTrain(m) {
    /* 调用点有时不给 m（例如「启动成功后 pollTrain()」），那样右侧统计就永远不刷新。
       记住最近一次训练界面用的模型，后续轮询都拿它。 */
    if (m && m.name) window.__runModel = m;
    var rm = (m && m.name) ? m : window.__runModel;
    if (TRAIN_POLL) clearInterval(TRAIN_POLL);
    function tick() {
      fetch("/api/train/status").then(function (r) { return r.json(); }).then(function (d) {
        window.__lastTrain = d;
        /* 模块加载失败的常驻横幅：内容是后端给的 warnings[]（状态，不是一次性事件），
           显隐由 js/loadbanner.js 决定 —— 训练没在跑就不显示。 */
        if (window.LCSLoadBanner) window.LCSLoadBanner.update(d);
        var iterEl = document.getElementById("runIter");
        if (iterEl && d.iter) iterEl.textContent = Number(d.iter).toLocaleString();
        /* 「点训练 → 等进程起来」这段窗口里，后端还会说 running:false。
           这时候不能把徽章改成「已停止」—— 那会让用户以为刚才那下没生效。 */
        var startingIdle = !!(window.__trainStarting
          && (Date.now() - window.__trainStarting) < 20000
          && !d.running && d.ok !== false);
        var state = startingIdle ? { text: "启动中", tone: "blue" } : trainState(d);
        var stage = document.getElementById("runStage");
        if (stage) {
          stage.textContent = state.text;
          stage.className = "tag t-" + state.tone;
        }
        var dot = document.querySelector("#p-run .live");
        var dotCls = "live s-" + state.tone;
        if (dot && dot.className !== dotCls) dot.className = dotCls;
        var pbtn = document.getElementById("runPause");
        if (pbtn) pbtn.textContent = d.paused ? "继续" : "暂停";
        /* 进程崩了只显示"已停止"会让人以为是自己停的，把真实原因打出来 */
        if (!d.running && d.ok === false && d.error) {
          var logEl0 = document.getElementById("runLog");
          if (logEl0) logEl0.textContent = String(d.error).slice(0, 400);
          /* 显存不足这类错误要**主动弹出来**：只写进日志区，用户以为训练正常停了。 */
          var _errTxt = String(d.error);
          /* 常驻横幅：OOM 必须让用户看到"要改什么"，不能只是闪一下。 */
          if (window.showRunAlert) {
            if (_errTxt.indexOf("显存不足") >= 0) {
              window.showRunAlert("显存不足，训练已停止", _errTxt);
            } else if (_errTxt) {
              window.showRunAlert("训练已停止", _errTxt);
            }
          }
          if (_errTxt.indexOf("显存不足") >= 0 && window.__oomToasted !== _errTxt) {
            window.__oomToasted = _errTxt;
            toast("显存不足", _errTxt.split("\n").slice(0, 3).join(" "), "error", 12000);
          }
        }
        if (d.running && window.hideRunAlert) window.hideRunAlert();
        /* 真进度：iter / 速度 / 已跑 / 剩余（后端 /api/train/status 里都算好了） */
        function _hms(sec) {
          sec = Math.max(0, Math.round(sec));
          var h = Math.floor(sec / 3600), m = Math.floor((sec % 3600) / 60), s = sec % 60;
          return h ? (h + "h " + m + "m") : (m ? (m + "m " + s + "s") : (s + "s"));
        }
        /* __trainStarting：刚点下「训练」的这几秒，模型里的旧迭代数不能拿来充数 */
        var starting = window.__trainStarting && (Date.now() - window.__trainStarting < 600000);
        if (starting && d.running && d.iter_seen) window.__trainStarting = 0;
        if (starting && !d.running && Date.now() - window.__trainStarting > 8000) window.__trainStarting = 0;
        var idleIter = d.iter || (window.__trainStarting ? 0 : ((rm && rm.iter) || 0));
        if ($("runIter")) $("runIter").textContent = Number(idleIter).toLocaleString("en-US");
        if ($("runTarget")) {
          var tg = d.target_iter || (rm && rm.target_iter) || 0;
          $("runTarget").textContent = tg ? Number(tg).toLocaleString("en-US") : "∞";
        }
        if ($("runSpeed")) {
          $("runSpeed").textContent = (!d.running && !window.__trainStarting) ? "—"
            : (d.paused ? "已暂停"
              : ((d.it_s || 0) > 0 ? (Number(d.it_s).toFixed(2) + " it/s") : "测速中…"));
        }
        /* 学习率与两个 loss：训练日志里每轮都带着，直接照抄比另算可信 */
        if ($("stLr")) {
          $("stLr").textContent = d.lr ? Number(d.lr).toExponential(2) : (d.running ? "—" : "未在训练");
        }
        if (d.history && d.history.length) {
          /* 实时曲线：每个轮询（2 秒）重画一次，曲线才会跟着走 */
          var hs = [], hd = [];
          d.history.forEach(function (h) { hs.push(h[1]); hd.push(h[2]); });
          var box = document.getElementById("chartBox");
          var sig = hs.length + ":" + hs[hs.length - 1] + ":" + hd[hd.length - 1];
          if (box && sig !== window.__chartSig) {
            window.__chartSig = sig;
            window.__chartLive = true;
            box.innerHTML = chartSVG({ src: hs, dst: hd });
            if ($("chartHint")) $("chartHint").textContent = "最近 " + hs.length + " iter（实时）";
          }
          if ($("chartAvg")) {
            var tail = d.history.slice(-100);
            var sum = 0;
            tail.forEach(function (h) { sum += (h[1] + h[2]) / 2; });
            $("chartAvg").textContent = (sum / tail.length).toFixed(4);
          }
        }
        var sparkSrc = (rm && rm.spark && rm.spark.src) || [];
        var sparkDst = (rm && rm.spark && rm.spark.dst) || [];
        if ($("runLossSrc")) {
          $("runLossSrc").textContent = (d.loss_src != null) ? Number(d.loss_src).toFixed(4)
            : (!d.running && !window.__trainStarting && sparkSrc.length ? Number(sparkSrc[sparkSrc.length - 1]).toFixed(4) : "—");
        }
        if ($("runLossDst")) {
          $("runLossDst").textContent = (d.loss_dst != null) ? Number(d.loss_dst).toFixed(4)
            : (!d.running && !window.__trainStarting && sparkDst.length ? Number(sparkDst[sparkDst.length - 1]).toFixed(4) : "—");
        }
        if ($("runElapsed")) $("runElapsed").textContent = d.elapsed_s ? ("已跑 " + _hms(d.elapsed_s)) : "—";
        /* 有目标才显示剩余时间；没目标时进度行已经是「X / 不限 iter」，
           再写一句「未设目标迭代」就是同一件事说两遍。 */
        var hasTarget = !!(d.target_iter || (rm && rm.target_iter));
        if ($("runEta")) {
          $("runEta").textContent = d.eta_s != null ? ("约 " + _hms(d.eta_s) + " 剩余") : "";
          $("runEta").style.display = (hasTarget || d.eta_s != null) ? "" : "none";
        }
        if ($("runEtaDot")) {
          $("runEtaDot").style.display = (hasTarget || d.eta_s != null) ? "" : "none";
        }

        /* 右侧统计：全按 /api/models/{name}/stats 的真实值填，取不到就显示 — */
        if (rm && rm.name && !window.__statsBusy) {
          window.__statsBusy = true;
          fetch("/api/models/" + encodeURIComponent(rm.name) + "/stats")
            .then(function (r) { return r.json(); }).then(function (s) {
              window.__statsBusy = false;
              if (!s || s.HTTP) return;
              if (s.gpu) {
                if ($("stGpu")) $("stGpu").textContent = "显存 " + s.gpu.used_gb + " / " + s.gpu.total_gb + " GB";
                if ($("stGpuBar")) $("stGpuBar").style.width =
                  Math.min(100, Math.round(100 * s.gpu.used_gb / (s.gpu.total_gb || 1))) + "%";
                if ($("stUtil")) $("stUtil").textContent = (s.gpu.util == null ? "—" : s.gpu.util + "%");
                if ($("stUtilBar")) $("stUtilBar").style.width = (s.gpu.util == null ? 0 : s.gpu.util) + "%";
              }
              if ($("stCpu")) $("stCpu").textContent = (s.cpu_percent == null ? "—" : s.cpu_percent + "%");
              if ($("stCpuBar")) $("stCpuBar").style.width = (s.cpu_percent || 0) + "%";
              if ($("stRam")) {
                $("stRam").textContent = (s.ram_used_gb == null) ? "—"
                  : (s.ram_used_gb + " / " + s.ram_total_gb + " GB · " + Math.round(s.ram_percent) + "%");
              }
              if ($("stRamBar")) $("stRamBar").style.width = (s.ram_percent || 0) + "%";
              if ($("stDisk")) {
                var parts = [];
                if (s.disk_free_gb != null) parts.push("剩 " + s.disk_free_gb + " GB");
                if (s.model_gb != null) parts.push("模型 " + s.model_gb + " GB");
                $("stDisk").textContent = parts.join(" · ") || "—";
              }
              if ($("stSave") && s.last_save_ts) {
                var mins = Math.max(0, Math.round((Date.now() / 1000 - s.last_save_ts) / 60));
                $("stSave").textContent = mins < 1 ? "刚刚" : (mins < 60 ? mins + " 分钟前" : Math.floor(mins / 60) + " 小时前");
              }
              if ($("stNext")) {
                if (!s.backup_interval) {
                  $("stNext").textContent = "关闭";
                } else {
                  /* 倒计时：4.0 是在「距上次备份 >= backup_interval」时备份，
                     所以按真实的上次备份迭代数算，而不是取模猜（重启后锚点会变）。 */
                  var lt2 = window.__lastTrain || {};
                  var curIt = lt2.iter || s.iter || 0;
                  var lastBak = lt2.last_backup_iter || 0;
                  var left = lastBak > 0
                    ? Math.max(0, s.backup_interval - (curIt - lastBak))
                    : (s.backup_interval - (curIt % s.backup_interval));
                  $("stNext").textContent = lastBak > 0
                    ? ("还有 " + _num(left) + " iter")
                    : ("还有 " + _num(left) + " iter（约）");
                }
              }
              if ($("stBak")) {
                $("stBak").textContent = s.has_backup_dir
                  ? (s.backup_count + (s.max_backups ? " / " + s.max_backups : "") + " 份")
                  : (s.max_backups ? "0 / " + s.max_backups + " 份" : "—");
              }
            }).catch(function () { window.__statsBusy = false; });
        }

        /* 状态条只在「徽章说不清」的时候出现：
           加载模型 / 正在保存 / 正在停止 / 训练失败。
           训练中、已暂停、已停止都不显示 —— 徽章已经写着状态，上面那行已经有迭代数。 */
        var act = "", cls = "", sub = "";
        var showStrip = false;
        if (startingIdle) {
          showStrip = true;
          act = "启动中…";
          cls = "busy";
          sub = "4.0 进程正在起来，模型加载要一会儿";
        } else if (!d.running) {
          if (d.ok === false) {
            showStrip = true;
            cls = "bad";
            sub = String(d.error || "").slice(0, 70);
          }
        } else if (d.stopping) {
          showStrip = true; act = "正在停止…"; cls = "busy"; sub = "等它保存完再退出";
        } else if (d.saving) {
          showStrip = true; act = "正在保存…"; cls = "busy"; sub = "下一个迭代后落盘";
        } else if (!d.paused && !d.iter_seen) {
          showStrip = true; cls = "busy";
          var hint = "";
          (d.log || []).slice().reverse().some(function (l) {
            if (!l || l.charAt(0) === "[") return false;
            hint = String(l).replace(/^[\s·]+/, "").slice(0, 34);
            return true;
          });
          act = "加载模型…";
          sub = [d.elapsed_s ? ("已等 " + _hms(d.elapsed_s)) : "", hint].filter(Boolean).join(" · ");
        }

        var strip = document.getElementById("runStat");
        if (strip) {
          strip.className = "runstat " + cls;
          strip.style.display = showStrip ? "" : "none";
          var tEl = document.getElementById("runStatText");
          var sEl = document.getElementById("runStatSub");
          if (tEl) { tEl.textContent = act; tEl.style.display = act ? "" : "none"; }
          if (sEl) sEl.textContent = sub;
        }
        /* 第一行迭代出现 = 训练真的开始了：这时候把占位图换成实时预览（每次开训只自动一次） */
        if (d.running && d.iter_seen) {
          if (!window.__pvAutoDone) {
            window.__pvAutoDone = true;
            if (window.pvAutoAfterStart) window.pvAutoAfterStart();
          }
        } else if (!d.running) {
          window.__pvAutoDone = false;   /* 下次开训再自动一次 */
        }
        /* 没在训练也别把定时器清掉：降频到 5 秒继续看。
           否则在别处（或另一个窗口）开训之后，这个页面永远停在「已停止」。 */
        var want = d.running ? 2000 : 5000;
        if (window.__pollEvery !== want) {
          window.__pollEvery = want;
          if (TRAIN_POLL) clearInterval(TRAIN_POLL);
          TRAIN_POLL = setInterval(tick, want);
        }
      }).catch(function () {});
    }
    /* 先立刻打一枪：进页面那两秒不该停在「连接中…」这种中间态上 */
    window.__pollEvery = 2000;
    tick();
    if (TRAIN_POLL) clearInterval(TRAIN_POLL);
    TRAIN_POLL = setInterval(tick, 2000);
  }
  window.pollTrain = pollTrain;

  function startTraining(name) {
    return fetch("/api/train/start", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name: name })
    }).then(function (r) {
      return r.json().then(function (d) {
        if (!r.ok) throw new Error(d.detail || "启动失败");
        return d;
      });
    });
  }
  window.startTraining = startTraining;
  window.openRun = openRun;

  /* ---------- 训练预览：单图 + 手动刷新 ---------- */
  var PV_TABS = ["原图", "遮罩", "输入", "合并"];

  function pvShow(idx) {
    var st = window.__pv || {};
    idx = Math.max(0, Math.min(3, parseInt(idx, 10) || 0));
    st.idx = idx;
    window.__pv = st;
    var tabs = document.querySelectorAll("#pvTabs b");
    Array.prototype.forEach.call(tabs, function (b) {
      b.classList.toggle("on", Number(b.dataset.v) === idx);
    });
    var img = document.getElementById("pvImg");
    var empty = document.getElementById("pvEmpty");
    var crop = document.getElementById("pvCrop");
    var src = st.imgs && st.imgs[idx] ? st.imgs[idx] : "";
    if (crop) crop.style.display = "none";
    if (img) {
      if (src) {
        img.src = src;
        img.style.display = "";
        if (empty) empty.style.display = "none";
      } else {
        img.removeAttribute("src");
        img.style.display = "none";
        if (empty) {
          empty.style.display = "";
          empty.textContent = "还没有预览图 —— 训练跑起来后点「刷新」";
        }
      }
    }
    var hint = document.getElementById("prevHint");
    if (hint) {
      var bits = [];
      if (st.imgs && st.imgs[idx]) {
        if (st.iter) bits.push(_num(st.iter) + " iter");
        if (st.at) {
          var sec = Math.max(0, Math.round((Date.now() - st.at) / 1000));
          bits.push(sec < 60 ? (sec + "s 前") : (Math.floor(sec / 60) + "m 前"));
        }
      }
      hint.textContent = bits.join(" · ");
    }
  }
  window.pvShow = pvShow;

  function pvFetch(silent) {
    var btn = document.getElementById("pvRefresh");
    if (btn && !silent) { btn.disabled = true; btn.textContent = "等新图…"; }
    /* wait=8：后端会一直等到 4.0 真的吐出一张新图（最多 8 秒）。
       不带这个参数只会立刻返回缓存 —— 点刷新像没反应，就是这个原因。
       进页面那次也给 6 秒：训练在跑的话通常一进来就有图，不用先点一下。 */
    return fetch("/api/train/preview?wait=" + (silent ? "6" : "8")).then(function (r) { return r.json(); }).then(function (j) {
      var list = j.previews || [];
      var st = window.__pv || (window.__pv = { imgs: [], labels: [], iter: 0, at: 0, idx: 0 });
      if (list.length) {
        st.imgs = list.map(function (p) { return p.data ? ("data:image/jpeg;base64," + p.data) : ""; });
        st.labels = list.map(function (p) { return p.name || ""; });
        st.at = Date.now();
      }
      st.iter = j.iter || (window.__lastTrain || {}).iter || 0;
      st.fresh = !!j.fresh;
      pvShow(st.idx || 0);
      if (btn && !silent) { btn.disabled = false; btn.textContent = "刷新"; }
      if (!silent) {
        if (!list.length) toast("预览还没准备好", "4.0 通常几十秒后才有第一张", "info");
        else if (!j.fresh) toast("没拿到新图", "训练那边还没出下一张，稍后再点一次", "info");
      }
      return { n: list.length, fresh: !!j.fresh };
    }).catch(function () {
      if (btn && !silent) { btn.disabled = false; btn.textContent = "刷新"; }
      return { n: 0, fresh: false };
    });
  }
  window.pvFetch = pvFetch;

  /* 开训后自动拉一次预览：4.0 要跑出第一张才会给图，一次可能落空，
     所以最多重试 3 次、每次隔 12 秒。只在「本次开训」自动这一次。 */
  function pvAutoAfterStart() {
    var tries = 0;
    (function attempt() {
      tries += 1;
      pvFetch(true).then(function (r) {
        if (r && r.fresh) return;
        if (tries < 3) setTimeout(attempt, 12000);
      });
    })();
  }
  window.pvAutoAfterStart = pvAutoAfterStart;

  document.addEventListener("click", function (e) {
    var t = e.target;
    if (!t) return;
    if (t.id === "pvRefresh") { pvFetch(false); return; }
    var tab = t.closest ? t.closest("#pvTabs b") : null;
    if (tab) { pvShow(Number(tab.dataset.v)); }
  });

  /* 点「开始训练」之后立刻把界面打回「加载模型…」：
     4.0 要几十秒才出第一行迭代，这段时间显示旧数字（或「训练中」）都是假的。 */
  function trainStartingUi() {
    window.__trainStarting = Date.now();
    window.__pvAutoDone = false;
    var dot0 = document.querySelector("#p-run .live");
    if (dot0) dot0.className = "live s-blue";
    if ($("runStat")) {
      $("runStat").className = "runstat busy";
      if ($("runStatText")) $("runStatText").textContent = "启动中…";
      if ($("runStatSub")) $("runStatSub").textContent = "";
    }
    if ($("runStage")) {
      $("runStage").textContent = "启动中";
      $("runStage").className = "tag t-blue";
    }
    if ($("runIter")) $("runIter").textContent = "0";
    if ($("runTarget")) $("runTarget").textContent = "∞";
    if ($("runSpeed")) $("runSpeed").textContent = "测速中…";
    if ($("runElapsed")) $("runElapsed").textContent = "—";
    if ($("runEta")) { $("runEta").textContent = ""; $("runEta").style.display = "none"; }
    if ($("runEtaDot")) $("runEtaDot").style.display = "none";
    if ($("runLossSrc")) $("runLossSrc").textContent = "—";
    if ($("runLossDst")) $("runLossDst").textContent = "—";
    window.__lastTrain = null;
  }
  window.trainStartingUi = trainStartingUi;

  /* 与 4.0 的 _HOT_KEYS 对齐：训练中改这些会立即生效 */
  var HOT_KEYS = {
    crash_threshold: 1, backup_interval: 1, max_backups: 1,
    face_style_power: 1, bg_style_power: 1,
    masked_training: 1, blur_out_mask: 1, eyes_mouth_prio: 1, uniform_yaw: 1,
    lr: 1, lr_cos: 1, lr_dropout: 1,
    random_src_flip: 1, random_dst_flip: 1, random_hsv_power: 1,
    random_warp: 1, random_occlusion: 1, random_noise: 1,
    random_color_power: 1, random_color_alg: 1,
    ct_mode: 1,
    rotation_range: 1, scale_range: 1, t_range: 1,
    eyes_scale: 1, mouth_scale: 1, edge_enhance_power: 1,
    log_code_stats: 1
  };
  window.HOT_KEYS = HOT_KEYS;

  /* 把参数页上改过的值收集成一个 patch（「保存」和「开始训练」共用） */
  window.collectCfgPatch = collectCfgPatch;
  /* 回填真实参数之后，把「初始值」重记一遍：只有和它不同的项才算改过 */
  function syncCfgBaseline() {
    var box = $("cfgBox");
    if (!box) return;
    box.querySelectorAll("[data-cfg]").forEach(function (el) {
      if (el.dataset.missing === "1") {
        /* 模型里没有这个键 → 不记基线；用户真改过（dirty）时 collectCfgPatch 会把它写进去 */
        delete el.dataset.v0;
        return;
      }
      if (el.tagName === "INPUT") el.dataset.v0 = String(el.value);
      else if (el.tagName === "SELECT") el.dataset.v0 = String(el.value);
      else if (el.classList.contains("sw")) el.dataset.v0 = el.classList.contains("on") ? "1" : "0";
      else if (el.classList.contains("seg")) {
        var on = el.querySelector("b.on");
        if (on) el.dataset.v0 = String(on.dataset.v !== undefined ? on.dataset.v : on.textContent.trim());
      }
    });
  }
  window.syncCfgBaseline = syncCfgBaseline;

  /* 目标批次 → 累积次数：向下取整（不能整除时用警告提示实际等效值） */
  function accumFromTarget(batchSize, target) {
    var b = Math.max(1, Math.floor(Number(batchSize) || 1));
    var t = Math.floor(Number(target) || 0);
    if (t <= b) return 1;
    return Math.max(1, Math.floor(t / b));
  }
  window.accumFromTarget = accumFromTarget;

  /* 目标批次校验：不能整除时用红字说明实际等效批次，而不是拦着不让存 */
  function validateTargetBatch() {
    var box = $("cfgBox");
    if (!box) return;
    var bEl = box.querySelector("[data-cfg=batch_size]");
    var tEl = box.querySelector("[data-cfg=target_batch]");
    if (!bEl || !tEl) return;
    var b = Math.max(1, Math.floor(Number(bEl.value) || 1));
    var t = Math.floor(Number(tEl.value) || 0);
    var row = tEl.closest(".cfgrow");
    var cd = row && row.querySelector(".cd");
    if (!cd) return;
    var base = "想要的等效批次；= 批次大小 × 累积次数。0 = 关闭累积";
    var eff = accumFromTarget(b, t) * b;
    if (t === 0) {
      cd.textContent = base;
      cd.style.color = "";
      return;
    }
    if (t < b) {
      cd.textContent = "目标批次小于批次大小（" + b + "），不会累积；请填 ≥ " + b + " 或直接填 0";
      cd.style.color = "var(--bad)";
      return;
    }
    if (t % b !== 0) {
      cd.textContent = "⚠ " + t + " 不能被 " + b + " 整除：实际会按 " +
                      (eff / b) + " 次累积，等效批次 " + eff + "（不是 " + t + "）";
      cd.style.color = "var(--warn)";
      return;
    }
    cd.textContent = base + "（当前 " + b + " × " + (eff / b) + " = " + eff + "）";
    cd.style.color = "";
  }
  window.validateTargetBatch = validateTargetBatch;


  /* 用户真的动过某一行 → 打 dirty 标记。
     原因：collectCfgPatch 只报"和基线不同"的行，而基线是回填 dat 时设的；
     dat 里没有的键（max_backups / backup_interval / crash_threshold 等）永远没有基线，
     于是用户填了值也会被静默丢掉 —— 这就是"最大备份数填 3 却不生效"的根因。 */
  (function () {
    function mark(e) {
      var t = e.target;
      var el = t && t.closest ? t.closest("[data-cfg]") : null;
      if (el) el.dataset.dirty = "1";
    }
    ["input", "change", "click"].forEach(function (ev) {
      document.addEventListener(ev, mark, true);
    });
  })();

  /* all=true：新建模型时用 —— 此时模型还不存在，没有"基线"可谈，页面上的值就是用户要的。 */
  function collectCfgPatch(all) {
    var box = $("cfgBox");
    var patch = {};
    if (!box) return patch;
    /* 只有这 7 个是「创建后不可改」的结构参数（和 app.py 的 _CFG_STRUCTURAL、
       routes_models._STRUCTURAL_KEYS 三处必须一致）。
       src / dst 是可改的数据集路径（引擎读 options.src/dst）；
       gan_power / true_face_power / face_style_power / models_opt_on_gpu 是每次训练
       都能改的损失权重与放置开关 —— 以前被错列在这里，改了静默丢弃、刷新回默认。 */
    var STRUCT = ["resolution", "archi", "e_dims", "ae_dims", "d_dims", "d_mask_dims",
                  "face_type", "name"];
    box.querySelectorAll("[data-cfg]").forEach(function (el) {
      var k = el.dataset.cfg;
      if (!k || STRUCT.indexOf(k) >= 0 || el.dataset.ro === "1") return;
      if (el.tagName === "INPUT") {
        var v = el.value.trim();
        if (v === "") return;
        patch[k] = !isNaN(Number(v)) ? Number(v) : v;
      } else if (el.tagName === "SELECT") {
        patch[k] = el.value;
      } else if (el.classList.contains("sw")) {
        patch[k] = el.classList.contains("on");
      } else if (el.classList.contains("seg")) {
        var on = el.querySelector("b.on");
        if (on) {
          var sv = (on.dataset.v !== undefined ? on.dataset.v : on.textContent.trim());
          var mm = /^([0-9.]+)\s*[（(].*[）)]$/.exec(String(sv));
          patch[k] = mm ? mm[1] : sv;
        }
      }
      /* 只报真的改过的：没动过的行不写回、也不热推。
         以前保存会把整页默认值一起推过去，模型里没有的键
         （crash_threshold / max_backups / log_code_stats…）就被写成 UI 默认值。 */
      if (Object.prototype.hasOwnProperty.call(patch, k) && !all) {
        /* 没有基线 = 这一行不是我们画的（或还没回填完）。以当前值为基线并跳过，
           宁可这一次不报，也不要把界面默认值当成用户的修改写进模型。 */
        if (el.dataset.v0 === undefined) {
          if (el.dataset.dirty === "1") {
            /* 用户主动改过：保留这次的值为"用户修改"，并把当前值记成新基线，
               这样同一个键不会每次都重复推送。 */
            el.dataset.v0 = el.classList.contains("sw")
              ? (el.classList.contains("on") ? "1" : "0")
              : String(patch[k]);
          } else {
            el.dataset.v0 = el.classList.contains("sw")
              ? (el.classList.contains("on") ? "1" : "0")
              : String(patch[k]);
            delete patch[k];
          }
        } else {
          var nowV = el.classList.contains("sw")
            ? (el.classList.contains("on") ? "1" : "0")
            : String(patch[k]);
          /* dat 里没有这个键 + 用户动过 → 就算值刚好等于界面默认值也要写进去。
             否则"把最大备份数设成默认的 3"永远存不进模型（这就是用户报的那个 bug）。 */
          var _force = (el.dataset.missing === "1" && el.dataset.dirty === "1");
          if (!_force && nowV === String(el.dataset.v0)) delete patch[k];
        }
      }
    });
    /* 目标批次存在 → 换算累积次数（引擎只认 accum_steps） */
    if (patch.target_batch !== undefined || patch.batch_size !== undefined) {
      var _cb = $("cfgBox");
      var _bEl = _cb && _cb.querySelector("[data-cfg=batch_size]");
      var _tEl = _cb && _cb.querySelector("[data-cfg=target_batch]");
      var _b2 = patch.batch_size !== undefined ? patch.batch_size : Number((_bEl || {}).value || 1);
      var _t2 = patch.target_batch !== undefined ? patch.target_batch : Number((_tEl || {}).value || 0);
      patch.accum_steps = accumFromTarget(_b2, _t2);
    }
    return patch;
  }
  window.collectCfgPatch = collectCfgPatch;

  /* 委托监听：批次大小 / 目标批次 改动时立刻校验（页面重绘后也不用重新绑） */
  document.addEventListener("input", function (e) {
    var el = e.target;
    if (!el || !el.dataset || (el.dataset.cfg !== "batch_size" && el.dataset.cfg !== "target_batch")) return;
    validateTargetBatch();
  }, true);
  document.addEventListener("change", function (e) {
    var el = e.target;
    if (!el || !el.dataset || (el.dataset.cfg !== "batch_size" && el.dataset.cfg !== "target_batch")) return;
    validateTargetBatch();
  }, true);
  window.addEventListener("load", function () { setTimeout(validateTargetBatch, 600); });


  function openRun(m) {
    /* 进页面先把占位文案清掉：没连上后端时不该显示 "1.82 it/s · 正式训练" 这种假数据 */
    if ($("runSpeed")) $("runSpeed").textContent = "测速中…";
    if ($("runElapsed")) $("runElapsed").textContent = "—";
    if ($("runEta")) { $("runEta").textContent = ""; $("runEta").style.display = "none"; }
    if ($("runEtaDot")) $("runEtaDot").style.display = "none";
    if ($("runLossSrc")) $("runLossSrc").textContent = "—";
    if ($("runLossDst")) $("runLossDst").textContent = "—";
    if ($("stLr")) $("stLr").textContent = "—";
    /* 已经在训练的话，别把标签重置成「连接中」再等下一次轮询 —— 会闪一下。
       只有确实没在跑时才写成「加载模型…」（从点「训练」到第一枪状态回来的这一瞬）。 */
    var lt = window.__lastTrain;
    var sameRun = !!(lt && lt.running && (!m || !m.name || lt.name === m.name));
    if (!sameRun) {
      if ($("runStage")) {
        $("runStage").textContent = "连接中";
        $("runStage").className = "tag t-blue";
      }
      if ($("runStat")) {
        $("runStat").className = "runstat busy";
        if ($("runStatText")) $("runStatText").textContent = "加载模型…";
        if ($("runStatSub")) $("runStatSub").textContent = "";
      }
      var dot1 = document.querySelector("#p-run .live");
      if (dot1) dot1.className = "live s-blue";
    }
    if (window.pollTrain) window.pollTrain(m);

    var box = $("chartBox");
    var nameEl = $("runName");
    if (nameEl) nameEl.textContent = m ? m.name : "模型";

    /* 预览只放一张，按分类切换；刷新是手动的 —— 自动刷新会打断观察
       （刚看清遮罩边缘，图就被下一轮换掉了）。进入页面时先拉一次，
       之后只有点「刷新」才更新。 */
    /* 不再拿模型里那张 4×5 拼图当占位：没拉到实时图就空着，提示去刷新 */
    window.__pv = { imgs: [], labels: [], iter: 0, at: 0, idx: 0,
                    bs: (m && m.batch_size) || 4, modelIter: (m && m.iter) || 0 };
    /* 没在训练时，运行条读模型里保存的值 —— 显示 0 iter / 一堆破折号看着像没加载出来 */
    if (m && m.iter && !(window.__lastTrain && window.__lastTrain.running)) {
      if ($("runIter")) $("runIter").textContent = _num(m.iter);
      if ($("runTarget")) $("runTarget").textContent = m.target_iter ? _num(m.target_iter) : "∞";
      if ($("runSpeed")) $("runSpeed").textContent = "—";
      var sp = (m.spark && m.spark.src) || [];
      var dp = (m.spark && m.spark.dst) || [];
      if ($("runLossSrc")) $("runLossSrc").textContent = sp.length ? Number(sp[sp.length - 1]).toFixed(4) : "—";
      if ($("runLossDst")) $("runLossDst").textContent = dp.length ? Number(dp[dp.length - 1]).toFixed(4) : "—";
    }
    if ($("prevHint")) {
      $("prevHint").textContent = m && m.batch_size ? ("bs " + m.batch_size) : "—";
    }
    /* 不再拿模型里存的那张 preview.jpg 当占位 —— 它是 4 行 × 5 列的拼图，
       单独摆在预览区里看着像「四张脸拼在一起」。宁可空着，提示去刷新。 */
    pvShow(0);
    if (window.pollTrain) window.pollTrain(m);
    pvFetch(true);

    if (box && !(m && m.spark && m.spark.src && m.spark.src.length)) {
      box.innerHTML = '<div class="chartempty">还没有 loss 记录 —— 跑几十个迭代就有了</div>';
    }
    /* loss 曲线：先用模型里存的历史占位，训练跑起来之后改画实时 history */
    window.__chartLive = false;
    if (box && m && m.spark && m.spark.src && m.spark.src.length) {
      box.innerHTML = chartSVG(m.spark);
      var avg = m.spark.src.slice(-100);
      var mean = avg.reduce(function (a, b) { return a + b; }, 0) / (avg.length || 1);
      if ($("chartAvg")) $("chartAvg").textContent = mean.toFixed(4);
      if ($("chartHint")) $("chartHint").textContent = "最近 " + m.spark.src.length + " iter（模型里存的）";
    }
    go("run");
  }

  function chartSVG(spark) {
    var s = (spark && spark.src) || [], d = (spark && spark.dst) || [];
    if (!s.length && !d.length) return "";
    /* 两条曲线共用一套纵轴，才能直接比高低 */
    var all = s.concat(d);
    var min = Math.min.apply(null, all), max = Math.max.apply(null, all);
    var span = (max - min) || 1;
    var W = 600, H = 160;

    function pts(arr, smooth) {
      var out = [];
      var src = smooth ? movingAvg(arr, 8) : arr;
      for (var i = 0; i < src.length; i++) {
        var x = (i / (src.length - 1 || 1)) * W;
        var y = H - ((src[i] - min) / span) * (H - 10) - 5;
        out.push(x.toFixed(1) + "," + y.toFixed(1));
      }
      return out.join(" ");
    }
    function poly(arr, color, w, opacity) {
      if (!arr || arr.length < 2) return "";
      return '<polyline fill="none" stroke="' + color + '" stroke-width="' + w + '"' +
        (opacity ? ' opacity="' + opacity + '"' : '') +
        ' vector-effect="non-scaling-stroke" stroke-linejoin="round" points="' + pts(arr, false) + '"/>';
    }
    function polyS(arr, color) {
      if (!arr || arr.length < 2) return "";
      return '<polyline fill="none" stroke="' + color + '" stroke-width="1.8"' +
        ' vector-effect="non-scaling-stroke" stroke-linejoin="round" stroke-linecap="round" points="' + pts(arr, true) + '"/>';
    }
    /* 三条水平参考线：0% / 50% / 100%，很淡，只用来判断幅度 */
    var grid = "";
    [0.25, 0.5, 0.75].forEach(function (f) {
      var y = (H * f).toFixed(1);
      grid += '<line x1="0" y1="' + y + '" x2="' + W + '" y2="' + y +
        '" stroke="var(--line)" stroke-width="1" vector-effect="non-scaling-stroke"/>';
    });
    return '<svg viewBox="0 0 ' + W + " " + H + '" preserveAspectRatio="none">' + grid +
      poly(d, "var(--chart-dst)", ".8", ".35") +
      poly(s, "var(--chart-src)", ".8", ".35") +
      polyS(d, "var(--chart-dst)") +
      polyS(s, "var(--chart-src)") +
      "</svg>";
  }

  function movingAvg(arr, n) {
    var out = [], sum = 0;
    for (var i = 0; i < arr.length; i++) {
      sum += arr[i];
      if (i >= n) sum -= arr[i - n];
      out.push(sum / Math.min(i + 1, n));
    }
    return out;
  }

  /* ---------- 通知调试 ---------- */
  function dbgToast(n) {
    var title = ($("dbgTitle") || {}).value || "通知";
    var sub = ($("dbgSub") || {}).value || "";
    var ms = parseInt(($("dbgMs") || {}).value, 10);
    var typeEl = document.querySelector("#dbgType > b.on");
    var type = typeEl ? typeEl.getAttribute("data-t") : "success";
    var count = n || 1;
    for (var i = 0; i < count; i++) {
      (function (k) {
        setTimeout(function () {
          var t = count > 1 ? title + " #" + (k + 1) : title;
          toast(t, sub, type, ms);
          updateDbgCount();
        }, k * 120);
      })(i);
    }
  }

  function updateDbgCount() {
    var el = $("dbgCount");
    if (!el) return;
    var n = document.querySelectorAll("#toastroot .toast").length;
    /* 这行是运行时拼的：英文模式下拼完要再翻一次，否则永远是中文 */
    var en = false;
    try { en = window.I18N && window.I18N.get() === "en"; } catch (e) {}
    el.textContent = en ? (n + " messages") : ("当前 " + n + " 条");
  }

  if ($("dbgOne")) {
    $("dbgOne").addEventListener("click", function () { dbgToast(1); });
    $("dbgThree").addEventListener("click", function () { dbgToast(3); });
    $("dbgFive").addEventListener("click", function () { dbgToast(5); });
    $("dbgClear").addEventListener("click", function () {
      document.querySelectorAll("#toastroot .toast").forEach(function (t) { dismissToast(t); });
      updateDbgCount();
    });
    $("dbgLong").addEventListener("click", function () {
      toast("这是一个很长很长的通知标题，用来测试换行和气泡的最大宽度会不会撑破",
        "副标题也很长：workspace\\data_src\\ 里还有 12,408 张没处理，预计需要 18 分钟", "warn", 4200);
      updateDbgCount();
    });
    document.querySelectorAll("#dbgType > b").forEach(function (b) {
      b.addEventListener("click", function () {
        document.querySelectorAll("#dbgType > b").forEach(function (x) { x.classList.remove("on"); });
        b.classList.add("on");
      });
    });
    setInterval(updateDbgCount, 600);
  }

  /* ---------- 侧边栏「工具」分组 ---------- */
  var toolGrp = document.getElementById("toolgrp");
  var toolHd = document.getElementById("toolgrpHd");
  var toolBody = document.getElementById("toolgrpBody");

  /* ---------- 折叠态：工具组飞出层 ---------- */
  var FLY = { timer: 0, pinned: false };

  function flyShow(pin) {
    var fly = $("toolFly"), hd = $("toolgrpHd");
    if (!fly || !hd) return;
    if (!document.querySelector(".app").classList.contains("collapsed")) return;
    if (pin) FLY.pinned = true;
    var r = hd.getBoundingClientRect();
    fly.style.left = Math.round(r.right + 6) + "px";
    fly.style.top = Math.round(r.top - 6) + "px";
    var fh = fly.offsetHeight || 200;
    var maxTop = window.innerHeight - fh - 12;
    if (r.top - 6 + fh > maxTop) fly.style.top = Math.max(12, maxTop) + "px";
    fly.classList.add("on");
  }

  function flyHide(delay, force) {
    if (FLY.pinned && !force) return;      /* 钉住时鼠标移开也不收 */
    clearTimeout(FLY.timer);
    FLY.timer = setTimeout(function () {
      var fly = $("toolFly");
      if (fly) fly.classList.remove("on");
      if (force) FLY.pinned = false;
    }, delay === undefined ? 160 : delay);
  }

  function flyToggle() {
    var fly = $("toolFly");
    if (!fly) return;
    if (FLY.pinned && fly.classList.contains("on")) {
      FLY.pinned = false;
      flyHide(0, true);                     /* 再点一次 = 收起 */
    } else {
      flyShow(true);                        /* 第一次点 = 钉住 */
    }
  }

  function flyBind() {
    var hd = $("toolgrpHd"), fly = $("toolFly");
    if (!hd || !fly || hd.dataset.fly) return;
    hd.dataset.fly = "1";

    hd.addEventListener("mouseenter", function () {
      clearTimeout(FLY.timer);
      flyShow(false);
    });
    hd.addEventListener("mouseleave", function () { flyHide(); });
    fly.addEventListener("mouseenter", function () { clearTimeout(FLY.timer); });
    fly.addEventListener("mouseleave", function () { flyHide(); });

    hd.addEventListener("click", function (e) {
      if (!document.querySelector(".app").classList.contains("collapsed")) return;
      e.stopPropagation();
      flyToggle();
    }, true);

    /* 点空白处：解锁并收起 */
    document.addEventListener("click", function (e) {
      if (!fly.classList.contains("on")) return;
      if (fly.contains(e.target) || hd.contains(e.target)) return;
      FLY.pinned = false;
      flyHide(0, true);
    });

    fly.querySelectorAll(".nav.sub").forEach(function (n) {
      n.addEventListener("click", function () {
        go(n.getAttribute("data-go"));
        FLY.pinned = false;
        flyHide(0, true);
      });
    });

    window.addEventListener("resize", function () { FLY.pinned = false; flyHide(0, true); });
  }
  window.flyBind = flyBind;

  /* Esc：关浮层、关隐私幕布（幕布需要密码，Esc 只收起未解锁的临时层） */
  document.addEventListener("keydown", function (e) {
    if (e.key !== "Escape") return;
    var fly = $("toolFly");
    if (fly && fly.classList.contains("on")) {
      FLY.pinned = false;
      flyHide(0, true);
      e.preventDefault();
    }
  });

  function setToolOpen(open) {
    if (!toolGrp || !toolBody) return;
    toolGrp.classList.toggle("closed", !open);
    if (open) {
      toolBody.style.height = "auto";
      toolBody.style.height = toolBody.scrollHeight + "px";   // 展开时现量
      /* 动画结束后交还给 auto，避免多次展开累计出多余空白 */
      setTimeout(function () {
        if (!toolGrp.classList.contains("closed")) toolBody.style.height = "auto";
      }, 260);
    } else {
      toolBody.style.height = "0px";
    }
    try { window.localStorage.setItem("dfl-toolgrp", open ? "1" : "0"); } catch (e) {}
  }
  window.setToolOpen = setToolOpen;

  if (toolHd) {
    toolHd.addEventListener("click", function () {
      setToolOpen(toolGrp.classList.contains("closed"));
    });
  }
  if (toolBody) {
    toolBody.style.height = toolBody.scrollHeight + "px";
    try {
      if (window.localStorage.getItem("dfl-toolgrp") === "0") setToolOpen(false);
    } catch (e) {}

  }

  /* 外部链接：交给系统浏览器，不在 app 里开新窗口 */
  function openExt(url) {
    var br = api();
    if (br && br.openExternal) { br.openExternal(url); return; }
    window.open(url, "_blank");
  }
  window.openExt = openExt;
  document.addEventListener("click", function (e) {
    var a = e.target && e.target.closest ? e.target.closest(".lnk") : null;
    if (!a) return;
    e.preventDefault();
    var url = a.getAttribute("data-url") || a.getAttribute("href");
    if (url) openExt(url);
  });


  if ($("setPass")) {
    $("setPass").value = hidePass();
    $("setPassSave").addEventListener("click", function () {
      try { localStorage.setItem("dfl-hidepass", $("setPass").value.trim()); } catch (e) {}
      toast("已保存", $("setPass").value.trim() ? "隐藏模式密码已更新" : "密码已清空", "success");
    });
  }

  flyBind();
  bindDropPaths();

  if ($("runHide")) $("runHide").addEventListener("click", function () { setCurtain(true); });
  if ($("cunlock")) $("cunlock").addEventListener("click", tryUnlock);
  if ($("cpass")) $("cpass").addEventListener("keydown", function (e) { if (e.key === "Enter") tryUnlock(); });


  function syncWelcomeCap() {
    var en = false;
    try { en = window.I18N && window.I18N.get() === "en"; } catch (e) {}
    var A = $("wCapA"), B = $("wCapB");
    if (A) A.textContent = en ? "Scan with the " : "使用";
    if (B) B.textContent = en ? "app to sign in" : "APP 扫码授权登录";
  }
  window.__syncWelcomeCap = syncWelcomeCap;
  syncWelcomeCap();

  document.querySelectorAll("#wLang b").forEach(function (b) {
    b.addEventListener("click", function () {
      var lg = b.getAttribute("data-lang");
      document.querySelectorAll("#wLang b").forEach(function (x) { x.classList.toggle("on", x === b); });
      if (window.I18N) window.I18N.apply(lg);
      if (window.__syncWelcomeCap) window.__syncWelcomeCap();
    });
  });


  if ($("cfCancel")) $("cfCancel").addEventListener("click", closeConfirm);
  if ($("cfOk")) $("cfOk").addEventListener("click", function () {
    var fn = CF.ok;
    closeConfirm();
    if (fn) fn();
  });
  if ($("confirmBox")) $("confirmBox").addEventListener("click", function (e) {
    if (e.target === $("confirmBox")) closeConfirm();
  });
  document.addEventListener("keydown", function (e) {
    if (e.key === "Escape" && $("confirmBox") && $("confirmBox").style.display !== "none") {
      closeConfirm();
      e.preventDefault();
    }
  });


  if ($("aboutLogo")) $("aboutLogo").addEventListener("click", function () { go("debug"); });
  if ($("edCanvas")) { edInit(); edBind(); edDraw(); }

  if ($("wSkip")) $("wSkip").addEventListener("click", authSkip);
  if ($("uLogout")) $("uLogout").addEventListener("click", authLogout);

  /* 头像菜单：折叠态下 [退出] 按钮被隐藏，点头像弹一个小菜单，至少还有条出路。
     展开态下也保留（点头像 = 快捷退出），点别处或按 Esc 关掉。 */
  (function () {
    var av = $("uAvatar"), menu = $("uMenu"), item = $("uMenuLogout");
    if (!av || !menu) return;
    /* 菜单挪到 body 下：侧栏 .side 是 overflow:auto（还会生成包含块），
       留在里面会被裁掉，弹不出来。 */
    document.body.appendChild(menu);

    function close() { menu.classList.remove("on"); }
    function place() {
      var r = av.getBoundingClientRect();
      menu.style.left = Math.round(r.right + 8) + "px";
      menu.style.top = Math.round(r.top - 4) + "px";
    }
    function toggle(e) {
      e.stopPropagation();
      /* 只有折叠态才弹菜单：展开态旁边就有 [退出] 按钮，
         再弹一个"退出登录"就是同一件事说两遍。 */
      var app = document.querySelector(".app");
      if (!app || !app.classList.contains("collapsed")) { close(); return; }
      var on = menu.classList.toggle("on");
      if (on) place();
    }
    av.addEventListener("click", toggle);
    av.addEventListener("keydown", function (e) {
      if (e.key === "Enter" || e.key === " ") { e.preventDefault(); toggle(e); }
    });
    if (item) item.addEventListener("click", function (e) {
      e.stopPropagation();
      close();
      authLogout();
    });
    document.addEventListener("click", close);
    document.addEventListener("keydown", function (e) { if (e.key === "Escape") close(); });
  })();

  /* 「保存」：只写参数，不动训练。训练中改的增强类参数会被后端热推给 4.0，立即生效。 */
  var cfgSave = $("cfgSave");
  if (cfgSave) cfgSave.addEventListener("click", function () {
    var m = window.__cfgModel || (window.MODELS || [])[0] || MODELS[0];
    if (!m) { toast("保存失败", "还没选中模型", "error"); return; }
    var patch = collectCfgPatch();
    if (!Object.keys(patch).length) { toast("没有改动", "参数还是原来的值", "info"); return; }
    cfgSave.disabled = true;
    cfgSave.textContent = "保存中…";
    fetch("/api/models/" + encodeURIComponent(m.name) + "/config", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ options: patch })
    }).then(function (r) {
      return r.json().then(function (d) {
        if (!r.ok) throw new Error(d.detail || "保存失败");
        return d;
      });
    }).then(function (d) {
      cfgSave.disabled = false;
      cfgSave.textContent = "保存";
      /* 老 dat 一保存就落成新格式了：迁移那行撤掉，转换提示（如果还没有 .pth）留着 */
      try {
        if (window.LegacyNotice) window.LegacyNotice.render(document.getElementById("cfgLegacy"), d.legacy);
      } catch (e2) { /* 提示层失败不影响保存结果 */ }
      var hot = d.hot || [], cold = d.need_restart || [];
      if (d.hot_applied) {
        toast("已保存 · " + hot.length + " 项即时生效",
          (cold.length ? "要重启训练才生效：" + cold.slice(0, 4).join("、") : "全部立即生效"),
          "success");
      } else if (hot.length && d.hot_error) {
        toast("已保存，但没推给训练进程", String(d.hot_error).slice(0, 90), "error");
      } else if (!d.count) {
        toast("没有改动", "参数和文件里的一致", "info");
      } else {
        toast("已保存", cold.length ? ("要重启训练才生效：" + cold.slice(0, 4).join("、")) : "下次启动训练时生效",
          "success");
      }
    }).catch(function (e) {
      cfgSave.disabled = false;
      cfgSave.textContent = "保存";
      toast("保存失败", String((e && e.message) || e).slice(0, 110), "error");
    });
  });

  var cfgStart = $("cfgStart");
  if (cfgStart) cfgStart.addEventListener("click", function () {
    var m = window.__cfgModel || (window.MODELS || [])[0] || MODELS[0];
    var box = document.getElementById("cfgBox");
    if (!m) {
      /* 模型列表还没到位：现拉一次再走 */
      fetch("/api/models").then(function (r) { return r.json(); }).then(function (d) {
        MODELS = d.models || [];
        window.MODELS = MODELS;
        if (window.__cfgStartRetry) return;
        window.__cfgStartRetry = 1;
        var m2 = window.__cfgModel || MODELS[0];
        window.__cfgStartRetry = 0;
        if (m2) { window.__cfgModel = m2; document.getElementById("cfgStart").click(); }
      }).catch(function () { toast("保存失败", "拿不到模型列表", "error"); });
      return;
    }
    if (!box) { openRun(m); return; }

    /* 把页面上改过的值收集起来，写回模型的 _data.dat。
       结构性字段（分辨率 / 架构 / 各维度 / face_type）创建后写不进去，
       传上去后端会整单拒绝 —— 之前"点开始训练没反应"就是死在这一步。 */
    var patch = collectCfgPatch();

    /* 训练已经在跑：这次点击只当作「应用参数」，别再去 startTraining ——
       那会拿 409「已经在训练」当失败，弹一条「训练未启动」，让人以为参数没生效。 */
    var lt0 = window.__lastTrain;
    var liveNow = !!(lt0 && lt0.running && m && lt0.name === m.name);

    cfgStart.disabled = true;
    fetch("/api/models/" + encodeURIComponent(m.name) + "/config", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ options: patch })
    }).then(function (r) {
      return r.json().then(function (d) {
        if (!r.ok) throw new Error(d.detail || "保存失败");
        return d;
      });
    }).then(function (d) {
      cfgStart.disabled = false;
      if (liveNow) {
        /* 训练中：已经热推了，就地提示，不跳页面、不重启 */
        if (d.count) {
          toast(d.hot_applied ? "已应用 · 立即生效" : "已保存",
            d.hot_applied
              ? ("改动 " + d.count + " 项，" + (d.need_restart && d.need_restart.length
                  ? (d.need_restart.length + " 项要重启才生效") : "全部即时生效"))
              : (d.need_restart && d.need_restart.length
                  ? ("要重启训练才生效：" + d.need_restart.slice(0, 4).join("、"))
                  : "下次启动训练时生效"),
            "success");
        } else {
          toast("没有改动", "参数还是原来的值", "info");
        }
        if (window.markHotRows) window.markHotRows();
        return;
      }
      try {
        if (window.LegacyNotice) window.LegacyNotice.render(document.getElementById("cfgLegacy"), d.legacy);
      } catch (e2) { /* 同上 */ }
      if (d.count) toast("参数已保存", m.name + " · 改动 " + d.count + " 项", "success");
      openRun(m);
      if (window.trainStartingUi) window.trainStartingUi();
      /* 参数落盘之后真的拉起 4.0 的训练进程 */
      if (window.startTraining) {
        window.startTraining(m.name)
          .then(function () {
            toast("训练已启动", m.name + " · 4.0 进程已拉起", "success");
            if (window.trainStartingUi) window.trainStartingUi();
            if (window.pollTrain) window.pollTrain(m);
          })
          .catch(function (e) {
            toast("训练未启动", String((e && e.message) || e).slice(0, 110), "error");
          });
      }
    }).catch(function (e) {
      cfgStart.disabled = false;
      toast("保存失败", String((e && e.message) || e), "error");
    });
  });
  /* ---------- 远程访问（把整个界面开到 0.0.0.0） ---------- */
  var rSw = document.getElementById("remoteSw");
  var rState = document.getElementById("remoteState");
  var rUrl = document.getElementById("remoteUrl");
  var rPort = document.getElementById("remotePort");

  function paintRemote(s) {
    if (!s) return;
    if (rSw) rSw.classList.toggle("on", !!s.enabled);
    if (rState) {
      rState.textContent = s.enabled ? ("已启用 · 端口 " + s.port) : "未启用";
      rState.style.color = s.enabled ? "var(--ok)" : "";
    }
    if (rUrl) rUrl.value = s.enabled ? s.url : "—";
    if (rPort && s.enabled) rPort.value = s.port;
  }

  function loadRemote() {
    fetch("/api/remote").then(function (r) { return r.json(); })
      .then(paintRemote).catch(function () {});
  }
  window.__loadRemote = loadRemote;

  if (rSw && !rSw.dataset.bound) {
    rSw.dataset.bound = "1";
    rSw.addEventListener("click", function () {
      var want = !rSw.classList.contains("on");
      var p = rPort ? (parseInt(rPort.value, 10) || 8002) : 8002;
      rSw.style.opacity = ".5";
      fetch("/api/remote", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ enabled: want, port: p })
      }).then(function (r) {
        return r.json().then(function (d) {
          if (!r.ok) throw new Error(d.detail || "设置失败");
          return d;
        });
      }).then(function (d) {
        rSw.style.opacity = "";
        paintRemote(d);
        toast(want ? "远程访问已开启" : "远程访问已关闭",
              want ? ("浏览器打开 " + d.url) : "只在本机可用了",
              want ? "success" : "info");
      }).catch(function (e) {
        rSw.style.opacity = "";
        toast("设置失败", String((e && e.message) || e).slice(0, 110), "error");
      });
    });
  }
  loadRemote();

  var cfgReset = $("cfgReset");
  if (cfgReset) cfgReset.addEventListener("click", function () {
    openCfg(window.__cfgModel || (window.MODELS || [])[0] || MODELS[0]);
  });

  document.querySelectorAll("#p-tool-angle .dbgrow .seg b").forEach(function (b, i) {
    b.addEventListener("click", function () {
      ANGLE_MODE = i === 0 ? "scatter" : "grid";
      drawAngle();
    });
  });
  drawAngle();

  var procHint = document.querySelector("#p-tool-bili .cardhd .hint");
  if (procHint && procHint.textContent.indexOf("并发") >= 0) procHint.textContent = "并发 " + BILI_JOBS.length;

  var backBtn = $("runBack");
  if (backBtn) backBtn.addEventListener("click", function () { go("train"); });

  loadFace();
  loadAuth().then(function () { renderHome(STATE || { datasets: { src: {}, dst: {} } }, MODELS); });

  renderSteps();
  /* 有 hash 就开到那一页（刷新 / 重启后回到原页），否则回主页 */
  var _bootKey = hashKey();
  go(_bootKey && PAGE_IDS[_bootKey] ? _bootKey : "home");
  loadState();
  if (window.I18N) {
    syncLangSeg(window.I18N.get());
    window.I18N.refresh();
  }
})();
