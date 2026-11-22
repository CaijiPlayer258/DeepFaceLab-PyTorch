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

        /* 引擎的常驻提示（[notice]，比如预训练结束重置了 inter）：蓝色信息条，
           训练结束后仍然保留，直到下次开训或手动关掉。 */
        var nEl = document.getElementById("trainNotice");
        if (nEl) {
          var nTxt = d && d.notice ? String(d.notice) : "";
          if (nTxt) {
            nEl.textContent = nTxt;
            nEl.style.display = "";
          } else if (!d || !d.running) {
            nEl.style.display = "none";
          }
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
    /* 先把界面切到训练页并打回「启动中」，再发请求：
       /api/train/start 里要释放合成会话、写 dat、探测设备，可能好几秒，
       以前要等它返回才跳转，用户会以为按钮没反应。 */
    try {
      window.__trainStarting = Date.now();
      if (window.go) window.go("run");
      if (window.trainStartingUi) window.trainStartingUi();
    } catch (e) {}
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
