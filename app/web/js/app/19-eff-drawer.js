    /* ============================================================
       生效配置（后端 [eff] 的回显）—— 进度页顶部的横幅 + 只读小面板
       ------------------------------------------------------------
       为什么要有这一块：检测器那个 bug 里日志写着 retinaface_10g、界面写着 TinyMog，
       两边各自都对，就是没人对得上账。后端现在每一单都回一份 eff（字段 + 来源）和
       eff_warnings[]，这里只负责摆出来：
         · 横幅：请求体/界面默认值对不上时出现（可关闭，换一单重新出现）；
         · 面板：这次真正生效的 14 个字段，值 + 来源（界面 / 推断 / 默认）。
       ============================================================ */
    var MGP_EFF = { eff: null, warns: [], dismissed: false, open: false, paintKey: "" };
    var MGP_EFF_LABEL = [
      ["detector", "检测器"], ["mask_source", "遮罩来源"], ["ct_mode", "色域迁移"],
      ["blend", "融合方式"], ["erode", "遮罩腐蚀"], ["blur", "遮罩羽化"],
      ["models", "模型"], ["faces_project", "人脸库"], ["library_enabled", "人脸库启用"],
      ["fast_export", "快速导出"], ["frame_range", "帧范围"], ["workers", "并发"],
      ["batch", "批大小"], ["jpeg_quality", "JPEG 质量"],
      /* 14 → 16：设置页「性能」区那两个下拉。它们的来源多一档 "auto" ——
         界面选了「自动」，档位交给后端定（不是「默认」：用户确实选过）。 */
      ["dfm_dtype", "推理精度"], ["compose_backend", "合成后端"]
    ];

    /* 来源标签：ui→界面 / inferred→推断 / auto→自动 / 其余→默认 */
    function mgpEffSrcText(s) {
      if (s === "ui") return "界面";
      if (s === "inferred") return "推断";
      if (s === "auto") return "自动";
      return "默认";
    }
    function mgpEffFmt(v) {
      if (v === true) return "开";
      if (v === false) return "关";
      if (v === null || v === undefined || v === "") return "—";
      if (Object.prototype.toString.call(v) === "[object Array]") {
        return v.length ? v.join("、") : "—";
      }
      return String(v);
    }
    function mgpEffHideWarn() {
      var box = $("mgpEffWarn");
      if (box) { box.hidden = true; box.classList.add("hide"); }
    }
    /* 两条警告要能**同时看见**：用户完全可能一边踩了「检测器是按质量推断的」、
       一边踩了「合成后端从 gpu 退回 cpu」。以前是 texts.join('\n') 塞进一个
       textContent —— .albd 虽然 pre-line，但验收读 DOM 只拿得到一个字符串，
       分不清「一条长文案」和「两条」，人也容易只读到第一行。现在一条一个 <div>。 */
    function mgpEffShowWarn(warns) {
      var box = $("mgpEffWarn");
      if (!box) return;
      var texts = [];
      for (var i = 0; i < (warns || []).length; i++) {
        var w = warns[i] || {};
        var t = w && w.text ? String(w.text) : String(w || "");
        if (t && texts.indexOf(t) < 0) texts.push(t);      /* 同一句话只摆一行 */
      }
      if (!texts.length || MGP_EFF.dismissed) { mgpEffHideWarn(); return; }
      var title = $("mgpEffWarnTitle"), body = $("mgpEffWarnBody");
      if (title) title.textContent = "生效配置与界面可能不一致（" + texts.length + " 项）";
      if (body) {
        body.innerHTML = "";
        for (var j = 0; j < texts.length; j++) {
          var line = document.createElement("div");
          line.className = "mgeff-wl";
          line.setAttribute("data-i", String(j));
          line.textContent = texts[j];
          body.appendChild(line);
        }
      }
      box.hidden = false;
      box.classList.remove("hide");
    }
    /* 警告列表按文案去重合并（保序）：后端 eff_warnings[] 是**累积**的，
       但不同调用给回来的可能只是一部分（POST 响应 / 换阶段 / 轮询先到）——
       整体替换会把已经在横幅上的那条抹掉。 */
    function mgpEffMergeWarns(prev, next) {
      var out = [], seen = {}, i, t;
      var push = function (w) {
        t = (w && w.text) ? String(w.text) : String(w || "");
        if (!t || seen[t]) return;
        seen[t] = 1;
        out.push(w);
      };
      for (i = 0; i < (prev || []).length; i++) push(prev[i]);
      for (i = 0; i < (next || []).length; i++) push(next[i]);
      return out;
    }
    /* 画抽屉里的 14 个格子。force=false 时按 paintKey 判重：
       status 每 500ms 轮询一次，同一份配置不重画 DOM。 */
    function mgpEffPaint(force) {
      var grid = $("mgpEffGrid"), routeEl = $("mgpEffRoute");
      var eff = MGP_EFF.eff;
      if (!grid || !eff) return false;
      var paintKey = String(eff.route || "") + "|" + String(eff.at || "") + "|" +
                     String(eff.kind || "") + "|" + String((MGP_EFF.warns || []).length) +
                     "|" + String(eff.compose_why || "");
      if (!force && MGP_EFF.paintKey === paintKey) return true;
      MGP_EFF.paintKey = paintKey;
      if (routeEl) {
        var line = String(eff.route || "") + (eff.kind ? (" · " + String(eff.kind)) : "");
        routeEl.textContent = line;
        routeEl.title = line;
      }
      grid.innerHTML = "";
      for (var i = 0; i < MGP_EFF_LABEL.length; i++) {
        var key = MGP_EFF_LABEL[i][0], name = MGP_EFF_LABEL[i][1];
        var src = eff[key + "_src"] || "default";
        var item = document.createElement("div");
        item.className = "mgeff-i";
        item.setAttribute("data-k", key);
        var k = document.createElement("span"); k.className = "k"; k.textContent = name;
        var b = document.createElement("b"); b.className = "v";
        b.textContent = mgpEffFmt(eff[key]);
        /* 被降级（选了 gpu 但实际跑 cpu 之类）：值仍是真实生效值，另挂「已降级」小标，
           原因放 title —— 光靠顶部横幅，用户点开抽屉会以为「我选的就是 cpu」。 */
        var why = (key === "compose_backend" && eff.compose_why) ? String(eff.compose_why) : "";
        var s = document.createElement("i"); s.className = "s " + src;
        s.textContent = mgpEffSrcText(src);
        if (eff.src_note && eff.src_note[key]) s.title = String(eff.src_note[key]);
        item.appendChild(k); item.appendChild(b);
        if (why) {
          var d = document.createElement("em"); d.className = "d"; d.textContent = "已降级";
          d.title = why;
          item.appendChild(d);
          s.className = "s downgraded";
          s.textContent = "降级";
          s.title = why;
          item.classList.add("downgraded");
        }
        item.appendChild(s);
        grid.appendChild(item);
      }
      return true;
    }

    /* 打开抽屉：**点开才渲染**这 14 个格子（常驻面板已经拆掉了，不占版面也不占渲染） */
    function mgpEffOpen() {
      var box = $("mgpEffDrawer"), btn = $("mgpEffOpen");
      if (!box || !MGP_EFF.eff) return;
      MGP_EFF.open = true;
        if (btn) { btn.hidden = true; btn.style.display = "none"; }   /* 抽屉开了就收起右下角入口 */
      mgpEffPaint(true);
      box.hidden = false;
      if (btn) btn.setAttribute("aria-expanded", "true");
      var sheet = $("mgpEffSheet");
      if (sheet && sheet.focus) { try { sheet.focus({ preventScroll: true }); } catch (e) { /* 老内核 */ } }
    }
    function mgpEffClose(giveFocus) {
      var box = $("mgpEffDrawer"), btn = $("mgpEffOpen");
      MGP_EFF.open = false;
      if (box) box.hidden = true;
          btn.hidden = false; btn.style.display = "";
      if (btn) {
        btn.setAttribute("aria-expanded", "false");
        if (giveFocus && btn.focus) {
          try { btn.focus({ preventScroll: true }); } catch (e) { /* 老内核 */ }
        }
      }
    }

    /* 后端回显 -> 横幅 + 抽屉。eff / warns 都来自 status（或 POST 响应），只读。
       抽屉关着的时候只记数据、不动 DOM —— 14 个字段不再常驻进度页。 */
    function mgpEffRender(eff, warns) {
      if (eff) MGP_EFF.eff = eff;
      /* 合并（不是整体替换）：换一次 status 只带回一部分警告时，不能把已经在横幅上的那条吞掉。
         换一单才清空 —— 那在 mgpReset 里（warns: []）。 */
      if (warns && warns.length) MGP_EFF.warns = mgpEffMergeWarns(MGP_EFF.warns, warns);
      eff = MGP_EFF.eff;
      /* 没有回显就别给一个点开是空的入口 */
      var openBtn = $("mgpEffOpen");
      if (openBtn) openBtn.disabled = !eff;
      if (!eff) { mgpEffClose(false); return; }
      mgpEffShowWarn(MGP_EFF.warns.length ? MGP_EFF.warns : (eff.warnings || []));
      if (MGP_EFF.open) mgpEffPaint(false);
    }
    window.mgpEffRender = mgpEffRender;
    window.mgpEffOpen = mgpEffOpen;
    window.mgpEffClose = mgpEffClose;
    window.__mgpEff = function () { return MGP_EFF; };

    /* 横幅的 × ：只关本次挂载（换一单在 mgpReset 里复位，和训练页横幅同一个口径） */
    (function bindEffWarnX() {
      var x = $("mgpEffWarnX");
      if (x && !x.dataset.bound) {
        x.dataset.bound = "1";
        x.addEventListener("click", function () {
          MGP_EFF.dismissed = true;
          mgpEffHideWarn();
        });
      }
    })();

    /* 右下角入口 + 抽屉的三个出口：点入口开、点 × 关、点遮罩关、Esc 关 */
    (function bindEffDrawer() {
      var open = $("mgpEffOpen"), close = $("mgpEffClose"), mask = $("mgpEffMask");
      if (open && !open.dataset.bound) {
        open.dataset.bound = "1";
        open.addEventListener("click", function () { mgpEffOpen(); });
      }
      if (close && !close.dataset.bound) {
        close.dataset.bound = "1";
        close.addEventListener("click", function () { mgpEffClose(true); });
      }
      if (mask && !mask.dataset.bound) {
        mask.dataset.bound = "1";
        mask.addEventListener("click", function () { mgpEffClose(true); });
      }
      if (!document.__mgpEffEsc) {
        document.__mgpEffEsc = 1;
        document.addEventListener("keydown", function (e) {
          if (!e || e.key !== "Escape") return;
          var box = $("mgpEffDrawer");
          if (box && !box.hidden) { e.stopPropagation(); mgpEffClose(true); }
        }, true);
      }
    })();

    /* 文件名栏：有名字才显示。以前占位是一根「—」，标题就变成「正在导出 —」——
       既不是文件名也不是状态（用户点名的噪音）。全部写入点都走这里。 */
    function mgpSetFile(txt, tip) {
      var el = $("mgpFile");
      if (!el) return;
      var s = String(txt == null ? "" : txt);
      el.textContent = s;
      el.hidden = !s;
      if (tip !== undefined) { if (tip) el.title = tip; else el.removeAttribute("title"); }
    }
    window.mgpSetFile = mgpSetFile;

    function mgpReset() {
      mgpRejectHide();
      /* 图整张回到「还没开单」：光点/点亮/卡住/计数/列标题状态全清。
         结构（lane 数与曲线）留着 —— 下一单的 status.stages 一到，签名对得上就复用，
         对不上才重建（切 lane 数是重建的唯一理由）。 */
      MG_PIPE.reset();
      var sub = $("mgPipeSub");
      if (sub) sub.textContent = "六段并行 · 每格一条 lane";
      var fr = $("mgpFrames"); if (fr) fr.textContent = "—";
      var eta = $("mgpEta"); if (eta) eta.textContent = "剩余 —";
      mgpSetFile("");
      /* 生效配置：换一单就重新开始 —— 上一单点过的「关闭横幅」不该延续到这一单 */
      MGP_EFF = { eff: null, warns: [], dismissed: false, open: false, paintKey: "" };
      mgpEffClose(false);
      var effGrid = $("mgpEffGrid"); if (effGrid) effGrid.innerHTML = "";
      var openBtn = $("mgpEffOpen"); if (openBtn) openBtn.disabled = true;
      mgpEffHideWarn();
    }
    window.mgpReset = mgpReset;
    /* 重连/恢复和验收脚本要用的同一套入口（改状态机只改一处，别各写一份）：
       mgpPaint/mgpFusedSteps = 列标题状态，mgpPipe = 喂一拍 status（stages 那份实况），
       __mgpPipe = 只读快照（验收要断言光点/卡住/曲线条数，别去扒 SVG 内部结构）。 */
    window.mgpPaint = mgpPaint;
    window.mgpFusedSteps = mgpFusedSteps;
    window.mgpChainDone = mgpChainDone;
    window.mgpPipe = function (d) { MG_PIPE.paint(d || {}); };
    window.mgpPipeReset = function () { MG_PIPE.reset(); };
    window.__mgpPipe = function () { return MG_PIPE.debug(); };
    /* 截图专用：把光点钉在曲线上（0 = 恢复）。只影响动画时钟，不影响任何判定。 */
    window.mgpPipeFreeze = function (at) {
      var v = Number(at) || 0;
      MG_PIPE.freeze(v);
      return v;
    };
    /* 一进来就把六列摆出来（lane 是「—」占位）：这块面板不该在开单之前是一个空白框。
       真正的 lanes 一到（status.stages），签名变了会自动重建 —— 所以这里只画骨架。 */
    MG_PIPE.paint({});

    /* 终态结果面板（进度页）：导出中藏起来，只留「取消导出」；
       完成/失败/取消才顶上来，给「返回导出设置」+「再次导出」两个出口。 */
    function mgpResultHide() {
      var box = $("mgpResult");
      if (box) box.hidden = true;
      /* 底部那排出口单独成块（用户：按钮放页面下面），一起藏 */
      var acts = $("mgpResultActs"); if (acts) acts.hidden = true;
      var cb = $("mgpCancel");
      if (cb) cb.hidden = false;
      return box;
    }
    function mgpResultShow(okAll, d) {
    /* 结果态（成功/失败）都收起「仍然尝试」—— 它只属于「被拒」那一屏 */
    (function () { var fb = $("mgpForceTry"); if (fb) { fb.hidden = true; fb.style.display = "none"; }
      var rz = $("mgpResult"); if (rz) rz.classList.remove("is-rej"); })();
      d = d || {};
      var box = $("mgpResult");
      /* 底部那排出口：终态才出现（文本在标题下、按钮在页面底部） */
      var actsBox = $("mgpResultActs"); if (actsBox) actsBox.hidden = false;
      var txt = $("mgpResultText");
      /* 上一轮可能是「被拒绝」态（六步/进度条收起来了）：这里要恢复回来，
         否则一次拒绝之后再跑成功，进度页还是只剩一行字。 */
      mgpRejectHide();
      var cb = $("mgpCancel");
      if (cb) cb.hidden = true;               /* 终态不再有「取消导出」 */
      /* 标题：终态就别再写「正在导出」了 —— 结果面板写着「导出完成」、标题写着
         「正在导出」，同一屏两句话打架。 */
      var ttl2 = $("mgpHdTitle");
      if (ttl2) {
        ttl2.textContent = okAll ? "导出完成"
          : (d.phase === "cancelled" ? "已取消" : "导出失败");
      }
      if (txt) {
        var bits = [];
        if (okAll) {
          /* 页标题已经写着「导出完成」，结果行不再重复这三个字（用户：省下一点文字） */
          if (d.out_video || d.out) bits.push(String(d.out_video || d.out));
          if (d.cache_removed) bits.push("cache 已清理");
          if (d.fused) bits.push("快速导出（融合流水线）· 未建 cache");
          /* 坏帧降级：成片出来了，但跳过的那几帧要当面说清楚（帧号在 <成片>.log 里） */
          if (Number(d.skipped_count) || 0) {
            bits.push("跳过 " + d.skipped_count + " 帧（坏帧降级 · 详见成片日志）");
          }
        } else if (d.phase === "cancelled") {
          bits.push("已取消");
          /* 分段链有阶段游标、能接着跑；快速导出没有 —— 别把那句照抄过来 */
          bits.push(d.fused ? "快速导出不支持续跑，需整段重跑"
                            : "已完成的阶段留着，下次从这里接着跑");
        } else {
          bits.push("导出失败");
          if (d.error) bits.push(String(d.error).slice(0, 160));
        }
        txt.textContent = bits.join(" · ");
      }
      if (box) box.hidden = false;
      return box;
    }
    window.mgpResultHide = mgpResultHide;
    window.mgpResultShow = mgpResultShow;

    /* 阶段状态 -> 总进度条 + 帧数/速度 + 当前步右侧小字 */
    function mgpStatusPaint(d, idx) {
      d = d || {};
      if (d.eff) mgpEffRender(d.eff, d.eff_warnings);
      var p = Math.max(0, Math.min(100, Number(d.stage_pct) || 0));
      var overall = Math.round((idx * 100 + p) / MGP_STAGES.length);
      if (d.phase === "done") overall = Math.round(((idx + 1) * 100) / MGP_STAGES.length);
      var total = Number(d.frames_total) || 0, done = Number(d.frames_done) || 0;
      var rate = Number(d.frame_fps) || 0;
      var fr = $("mgpFrames");
      if (fr) {
        fr.textContent = total
          ? (done + " / " + total + " 帧")
          : (d.msg || "准备中…");
      }
      var eta = $("mgpEta");
      if (eta) {
        if (d.running) eta.textContent = (d.eta_s != null ? ("剩余 " + d.eta_s + " 秒") : (d.msg || "剩余 —"));
        else eta.textContent = (d.phase === "done" ? "完成"
          : (d.phase === "cancelled" ? "已取消" : "已停"));
      }
      /* 坏帧降级（DFL_FUSED_SKIP_BAD=1）：真跳过过帧才改这句副标题 ——
         没有跳过时一个字符都不动，默认路径的界面和现在逐像素相同。 */
      var skp = Number(d.skipped_count) || 0;
      var subEl = $("mgPipeSub");
      if (subEl && skp) {
        subEl.textContent = "跳过 " + skp + " 帧（坏帧降级 · 详见成片日志）";
      }
      var st = MGP_STAGES[idx] || "";
      var counts = d.counts || {};
      var tip = "";
      if (st === "frames") tip = (Number(counts.frames) || 0) + " 帧";
      else if (st === "faces") tip = (Number(counts.faces) || 0) + " 张脸";
      else if (st === "pred") tip = (Number(counts.pred) || 0) + " 张换脸";
      else if (st === "mask") tip = ((d.mask_source === "DFM") ? "用模型遮罩" : (d.mask_source || "")) + " · " + (Number(counts.masks) || 0) + " 张";
      else if (st === "merge") tip = total ? (done + "/" + total) : (d.msg || "");
      else if (st === "encode") tip = "crf16 · yuv420p";
      if (d.cache_dir) mgpSetFile(String(d.cache_dir).split(/[\\/]/).pop());
      var state = (d.phase === "failed" ? "err" : (d.phase === "done" ? "done" : "on"));
      mgpPaint(idx, { state: state, tip: tip });
      /* 分段链也有流水线实况：/stage/status 里的 stages_live（和 /export/status 的
         stages 是同一个 _StageMeter，只是那个响应里的 stages 早被每段元数据占着） */
      MG_PIPE.paint(d);
    }

    var MGP_POLL = null;
    var MGP_RETRY = {};              /* 每个阶段只重试一次（网络抖动，不是后端报错） */
    var MGP_TRACE = [];              /* 每个阶段 POST 的留痕（排障用：window.__mgpTrace） */
    window.__mgpTrace = MGP_TRACE;
    window.__mgStageRunning = false;

    function mgpTrace(o) {
      MGP_TRACE.push(o);
      if (MGP_TRACE.length > 40) MGP_TRACE.shift();
    }

    function mgpStopPoll() {
      if (MGP_POLL) { clearTimeout(MGP_POLL); MGP_POLL = null; }
    }

    function mgpTerminal(okAll, d, idx) {
      mgpStopPoll();
      window.__mgStageRunning = false;
      window.__mergeExporting = false;
      var cb = $("mgpCancel");
      /* 终态：取消键自己藏起来（mgpResultShow 里 hidden），出口改成结果面板那两颗 */
      if (cb) { cb.disabled = false; cb.textContent = "取消导出"; }
      var d2 = d || {};
      if (d2.vram_oom) {
        /* 强制尝试撞 OOM（分段链）：回到被拒态，别把六个阶段画成红的 */
        window.mgVramReject(d2.vram_oom, window.__mgForceRetry);
        return;
      }
      if (okAll) {
        mgpPaint(MGP_STAGES.length - 1, { state: "done", tip: "mp4" });
        mgpChainDone();
        var eta = $("mgpEta"); if (eta) eta.textContent = "完成";
        toast("导出完成", String(d2.out_video || "").slice(-60) || "六个阶段全部完成", "success");
      } else if (d2.phase === "cancelled") {
        /* 取消：当前这一步不是 done，也不该继续摆出「进行中」的亮蓝色 —— 收回 idle（灰） */
        mgpPaint(idx, { state: "idle", tip: "已取消" });
        toast("已取消导出", "已完成的阶段留着，下次从这里接着跑", "info");
      } else {
        mgpPaint(idx, { state: "err", tip: "失败" });
        var eta2 = $("mgpEta");
        if (eta2) eta2.textContent = String(d2.error || "失败").slice(0, 60);
        toast("导出失败", String(d2.error || "").slice(0, 140), "error");
      }
      /* 结果面板：完成 / 取消 / 失败三种终态都要有「返回导出设置 + 再次导出」 */
      mgpResultShow(okAll, d2);
    }

    function mgpPollStage(idx, body, next) {
      MGP_POLL = setTimeout(function () {
        fetch("/api/merge/stage/status").then(function (r) { return r.json(); }).then(function (d) {
          mgpStatusPaint(d, idx);
          if (d.running) { mgpPollStage(idx, body, next); return; }
          if (d.phase === "done") {
            var tip = "";
            if (MGP_STAGES[idx] === "frames") tip = ((d.counts || {}).frames || 0) + " 帧";
            else if (MGP_STAGES[idx] === "faces") tip = ((d.counts || {}).faces || 0) + " 张脸";
            mgpPaint(idx, { state: "done", tip: tip });
            next();
            return;
          }
          mgpTerminal(false, d, idx);
        }).catch(function () { mgpPollStage(idx, body, next); });
      /* 分段链保持 600ms：/stage/status 每次要 listdir 四个产物目录（frames/aligned/
         pred/mask），帧数上千时加频率不划算；而且分段链是**串行**的，图上不会出现
         「同一时刻一大堆交接」那种密度问题。 */
      }, 600);
    }

    function mgpRunStage(idx, body, next) {
      var st = MGP_STAGES[idx];
      if (!st) { next(); return; }
      mgpPaint(idx, { state: "on", tip: "" });
      /* 换阶段时把上一段的帧数/速度清掉，别让人以为新阶段已经「完成」 */
      var fr0 = $("mgpFrames"); if (fr0) fr0.textContent = "—";
      var eta0 = $("mgpEta"); if (eta0) eta0.textContent = "启动 " + (idx + 1) + "/" + MGP_STAGES.length + "…";
      fetch("/api/merge/stage/" + st, {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body)
      }).then(function (r) {
        return r.json().then(function (d) {
          /* 显存拒绝是「这一单没开始」，不是「这个阶段失败」：不标红、不重试，
             直接走被拒态（一行提示 + 返回导出设置）。 */
          if (r.status === 400 && d && d.need_mb != null) throw { __vram: d };
          if (!r.ok || d.ok === false) throw new Error(d.error || d.detail || ("HTTP " + r.status));
          return d;
        });
      }).then(function (d) {
        mgpTrace({ stage: st, ev: "ok", skipped: !!d.skipped });
        if (d.skipped) {
          var tip = (MGP_STAGES[idx] === "faces") ? (((d.counts || {}).faces || 0) + " 张脸")
                  : (MGP_STAGES[idx] === "pred") ? (((d.counts || {}).pred || 0) + " 张换脸")
                  : (MGP_STAGES[idx] === "mask") ? (((d.counts || {}).masks || 0) + " 张遮罩")
                  : (((d.counts || {}).frames || 0) + " 帧");
          if (MGP_STAGES[idx] === "mask" && d.mask_source) tip = (d.mask_source === "DFM" ? "用模型遮罩" : d.mask_source) + " · " + tip;
          mgpPaint(idx, { state: "done", tip: tip || "已有产物" });
          next();
          return;
        }
        mgpStatusPaint(d, idx);
        mgpPollStage(idx, body, next);
      }).catch(function (e) {
        if (e && e.__vram) {                 /* 起 job 之前就被显存挡下：一次提示，不标红 */
          mgpTrace({ stage: st, ev: "vram-refuse", msg: mgVramLine(e.__vram) });
          mgVramReject(e.__vram);
          return;
        }
        var msg = String((e && e.message) || e);
        /* 长任务跑完那一拍偶尔掉一次连接（网络层抖动）：重试一次。
           后端明确报的错（HTTP 4xx/5xx 带 error 文本）不重试，免得把真错误刷掉。 */
        if (!/^HTTP /.test(msg) && !/^(合成|切帧|切脸|换脸|遮罩|导出)/.test(msg) && !MGP_RETRY[st]) {
          MGP_RETRY[st] = 1;
          mgpTrace({ stage: st, ev: "retry", msg: msg.slice(0, 120) });
          mgpPaint(idx, { state: "on", tip: "重试…" });
          setTimeout(function () { mgpRunStage(idx, body, next); }, 900);
          return;
        }
        mgpTrace({ stage: st, ev: "error", msg: msg.slice(0, 200) });
        mgpPaint(idx, { state: "err", tip: "失败" });
        mgpTerminal(false, { phase: "failed", error: (e && e.message) || String(e) }, idx);
      });
      mgpTrace({ stage: st, ev: "post" });
    }

    /* 视频模式导出 = 四个阶段串起来跑（每步一个 API，失败/取消都能续跑） */
    function mgStageExport(payload) {
      payload = payload || {};
      var video = mgVidPath();
      if (!video) {
        toast("没有素材", "先在工作台导入一段视频", "warn");
        return;
      }
      if (window.__mergeExporting || window.__mgStageRunning || window.__mgVramChecking) {
        /* 连点：2 秒内只弹一条（幂等 —— 上一次还在跑/还在预检，再点不该叠 toast） */
        mgToastOnce("merge-busy", "正在导出", "上一次还没结束", "info");
        return;
      }
      var body = {
        video: video,
        faces_project: String(window.__mergeFaceLib || ""),
        /* payload.model 优先（导出设置页/验收脚本可以直接点名一个 dfm），
           没有就用左栏选中的那个模型 */
        model: payload.model ||
               (document.querySelector("#mgModels .mgitem.on .nm") || {}).textContent || "",
        /* 导出用**左栏选中的全部模型**（models[] 一个不落）；预览那边才按 faces 分摊。
           以前这里走的是 mgFaceModelsPayload()，人脸库没分配时它是空数组 ——
           导出的 models 就成了 []，等于一个模型都没带。 */
        models: (window.mgExportModelsPayload ? window.mgExportModelsPayload() : mgModelsPayload()),
        /* cache 默认不保留（后端成功导出后会删掉 <名字>_cache/）；导出设置页勾「保留」才拦住 */
        keep_cache: !!payload.keep_cache,
        thresh: 0.5,
        settings: mgMergeSettings(payload),
        force: false
      };
      /* 检测器：缺省**不发这个键**（发空串会被后端当成"没人说过"，正是老的静默回落）。 */
      var det = window.mgDetectorBody ? window.mgDetectorBody(payload) : "";
      if (det) body.detector = det;
      /* quality/align_quality：页面上早就没有这个控件了。以前这里硬写 "high"，
         等于每次都替用户表态「要 retinaface_10g」—— 后端 #4 修好之后，只有**真给了**
         才按档推。没人给就不发，让后端按 detector / 默认档走。 */
      if (payload.align_quality) body.quality = payload.align_quality;
      if (payload.output_path && /^[A-Za-z]:[\\/]/.test(payload.output_path)) {
        body.output_dir = payload.output_path;
      }
      /* 帧范围：默认全片。只有真改过才带上，别给后端塞一对它不一定认的默认值。 */
      var frameFrom = parseInt(payload.frame_from, 10);
      if (!isNaN(frameFrom) && frameFrom > 0) body.start_frame = frameFrom;
      var frameTo = parseInt(payload.frame_to, 10);
      if (!isNaN(frameTo) && frameTo > 0) body.end_frame = frameTo;
      mgPerfBody(body, payload);          /* 性能那两个下拉：只有非「自动」才带键 */

      /* ★ 显存预检（dry_run）：分段链有六个阶段，但显存是**一条链共用的**。
         在这里问一次，装不下就整条拒绝 —— 不然六个阶段会各失败一次
         （六步全红 + 好几条一样的 toast，用户看到的是"跑过又失败"）。 */
      window.__mgVramChecking = true;
      mgVramPreflight(body, function (proceed) {
        window.__mgVramChecking = false;
        if (!proceed) return;
        mgStageStart(payload, body);
      });
    }
    /* 预检通过之后真正开始六阶段链 */
    function mgStageStart(payload, body) {
      window.__mergeExporting = true;
      window.__mgStageRunning = true;
      mgpStopPoll();
      mgpReset();
      mgpResultHide();                    /* 上一轮的结果面板先收起来 */
      if (window.mgStep) window.mgStep("progress");
      var cb = $("mgpCancel");
      if (cb) { cb.disabled = false; cb.textContent = "取消导出"; cb.hidden = false; }
      document.dispatchEvent(new CustomEvent("merge:export-start", { detail: payload }));

      var idx = 0;
      function next() {
        idx += 1;
        if (idx >= MGP_STAGES.length) {
          /* 全部跑完：拿一次终态（成片路径 / cache 是否已删）再收尾 */
          fetch("/api/merge/stage/status").then(function (r) { return r.json(); })
            .then(function (d) { mgpTerminal(true, d || {}, MGP_STAGES.length - 1); })
            .catch(function () { mgpTerminal(true, {}, MGP_STAGES.length - 1); });
          return;
        }
        mgpRunStage(idx, body, next);
      }
      mgpPaint(0, { state: "on" });
      mgpRunStage(0, body, next);
    }
    window.mgStageExport = mgStageExport;
