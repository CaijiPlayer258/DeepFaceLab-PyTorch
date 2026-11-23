        /* ---------------- 终态：一口气收尾，然后彻底停 ---------------- */
        /* started 之外还要 age>=1500ms：POST 刚发出时后端还挂着上一轮的
           pct=100/running=false，那是残帧，不是这一轮的完成。 */
        if (maskIsDone(p) && (started || age >= 1500)) {
          window.__maskDone = true;
          /* 最终平均速度 = 总张数 / 总耗时。后端 rate 就是 done/(now-t0)，
             拿得到就照抄；读不到才退回前端自己量的 done/elapsed。 */
          var avg = null;
          if (done > 0) {
            avg = (Number(p.speed) > 0) ? Number(p.speed)
                  : (elapsed > 0.3 ? done / elapsed : null);
          }
          window.__maskFinal = { speed: avg, done: done, total: total,
                                 elapsed: elapsed, msg: p.msg || "" };
          dsRunPaint({ pct: 100, msg: p.msg, done: total || done, total: total || done,
                       speed: avg, elapsed: elapsed, eta: null, running: false, ok: p.ok });
          dsFreezeGraph(avg);                                /* 末端压成平线并冻结采样 */
          maskResetBtn(btn);
          setNavRunning("dataset", false, 0);
          toast("应用遮罩完成", p.msg || "已完成", "success");
          return;                                            /* 不再 setTimeout：就此定格 */
        }

        /* ---------------- 运行中 ---------------- */
        idle = (running || pct > 0 || done > 0) ? 0 : idle + 1;
        if (idle >= 6 && age > 3000) {                       /* ≈3 秒什么都没有：别空转 */
          window.__maskDone = true;
          setNavRunning("dataset", false, 0);
          maskResetBtn(btn);
          dsRunPaint({ pct: 0, msg: "任务没有启动（后端没有进度）", done: 0, total: 0,
                       speed: null, elapsed: elapsed, eta: null, running: false, ok: false });
          toast("应用遮罩", "后端没有进度，任务可能没起来", "error");
          return;
        }
        dsRunPaint({ pct: pct, msg: p.msg, done: done, total: total,
                     speed: (Number(p.speed) > 0) ? Number(p.speed)
                            : (elapsed > 0.3 && done ? done / elapsed : null),
                     elapsed: elapsed, eta: p.eta, running: true });
        if (btn) { btn.disabled = true; btn.textContent = "应用中…"; }
        setTimeout(tick, next);
      }).catch(function () {
        if (window.__maskDone) return;                       /* 网络抖一下不算完成 */
        setTimeout(tick, 1000);
      });
    };
    setTimeout(tick, 200);
  }
  window.maskPoll = maskPoll;

  /* 手动改路径框 → 同步 DSB.dir（选目录那条路已经会写） */
  (function () {
    var el = document.getElementById("dsPathVal");
    if (!el || el.dataset.dsBound) return;
    el.dataset.dsBound = "1";
    el.addEventListener("input", function () {
      if (window.DSB) DSB.dir = String(el.value || "").trim();
    });
    el.addEventListener("change", function () {
      if (window.DSB) DSB.dir = String(el.value || "").trim();
    });
  })();

  window.dsMask = dsMask;

  function runStep(i) {
    var k = STEP_SEL[i] || 0;

    /* ---------- 过滤器 ---------- */
    if (i === 0) {
      var mode = k === 1 ? "faceid" : (k === 2 ? "repeated" : (k === 3 ? "errface" : "blur"));
      if (mode === "errface" && erOff()) { erExplain(); return; }
      var body = { mode: mode, dir: "data_src" };
      if (mode === "blur") body.percent = PARAM_VAL["0-0-0"] !== undefined ? PARAM_VAL["0-0-0"] : 10;
      else if (mode === "repeated") body.similarity = PARAM_VAL["0-2-0"] !== undefined ? PARAM_VAL["0-2-0"] : 0.98;
      else {
        body.threshold = PARAM_VAL["0-3-0"] !== undefined ? PARAM_VAL["0-3-0"] : 0.5;
      body.percent = PARAM_VAL["0-3-1"] !== undefined ? PARAM_VAL["0-3-1"] : 0;
      body.eps = PARAM_VAL["0-1-0"] !== undefined ? PARAM_VAL["0-1-0"] : 0.40;
        body.min_samples = PARAM_VAL["0-1-1"] !== undefined ? PARAM_VAL["0-1-1"] : 2;
      }
      toast("开始扫描", (mode === "blur" ? "模糊" : (mode === "errface" ? "错脸" : "重复图")) + " · 正在统计…", "info");
      dsRunShow(mode === "blur" ? "正在扫描清晰度"
    : (mode === "repeated" ? "正在查找重复图"
    : (mode === "errface" ? "正在检测错脸" : "正在聚类人脸身份")));
      fetch("/api/dataset/scan", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body)
      }).then(function (r) { return r.json(); }).then(function (d) {
        toast("扫描完成", "共 " + d.total + " 张，命中 " + d.count + " 张", d.count ? "success" : "info");
        dsRunPaint({ pct: 100, msg: "共 " + d.total + " 张，命中 " + d.count + " 张",
                     done: d.total, total: d.total, speed: 0, eta: 0, running: false });
      }).catch(function (e) {
        toast("扫描失败", String((e && e.message) || e), "error");
      });
      return;
    }

    /* ---------- 排序器 ---------- */
    if (i === 1) {
      var BY = ["blur", "yaw", "origname", "phash", "hist"];
      var by = BY[k] || "blur";
      toast("开始排序", MODULES[1].modes[k] + " · 正在处理…", "info");
      dsRunShow("正在排序");
      fetch("/api/dataset/sort", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ by: by, dir: "data_src" })
      }).then(function (r) { return r.json(); }).then(function (d) {
        toast("排序完成", "已重排 " + d.count + " 张", "success");
        dsRunPaint({ pct: 100, msg: "已按「" + MODULES[1].modes[k] + "」重排 " + d.count + " 张",
                     done: d.count, total: d.count, speed: 0, eta: 0, running: false });
      }).catch(function (e) {
        toast("排序失败", String((e && e.message) || e), "error");
      });
      return;
    }

    toast(MODULES[i].n, "这一节的后端还没接", "info");
  }

  function dsModeKey() {
    /* 以前只认「重复图」，人脸身份和错脸都掉进了 blur —— 选错脸却在删模糊图 */
    var n = MODULES[0].modes[STEP_SEL[0]] || "";
    if (n.indexOf("身份") >= 0) return "faceid";
    if (n.indexOf("重复") >= 0) return "repeated";
    if (n.indexOf("错脸") >= 0 || n.indexOf("坏图") >= 0) return "errface";
    if (n.indexOf("合并") >= 0) return "merge";
    /* 遮挡率是**排序器**：不筛不删，算完给你看排序 + 一个阈值手动移走 */
    if (n.indexOf("遮挡") >= 0) return "occlusion";
    return "blur";
  }

  /* 数据集处理：扫描 → 确认移走 → 撤销，三段式 */
  var DSB = { stage: 0, hits: [], last: [], dir: "data_src", mode: "blur" };

  function dsBtn() { return document.getElementById("dsBtn"); }
  function dsNum() { return document.querySelector('#dsSteps [data-res="num"]'); }
  function dsLbl() { return document.querySelector('#dsSteps [data-res="lbl"]'); }
  function dsBar() { return document.querySelector('#dsSteps [data-res="a"]'); }

  /* 站点卡片上的「开始处理」按钮：真实按钮带 data-step（见 renderStation）。
     '#dsSteps .step:nth-child(n) .btn' 那套选择器从来没有匹配到过，别再用 ——
     应用遮罩就是因为 btn 恒为 null，完成之后按钮一直停在「应用中…」。 */
  function dsStepBtn(i) {
    return document.querySelector('#dsSteps button[data-step="' + i + '"]') ||
           document.querySelector('#dsSteps .dsacts .btn') || null;
  }

  function dsPaint(num, lbl, pct) {
    var n = dsNum(), l = dsLbl(), b = dsBar();
    if (n && num != null) n.textContent = num;
    if (l && lbl != null) l.textContent = lbl;
    if (b && pct != null) b.style.width = Math.max(2, Math.min(100, pct)) + "%";
  }

  function dsDirArg() {
    /* 后端只认 workspace 下的相对目录；拖进来的绝对路径要先裁掉前缀 */
    var inp = document.getElementById("dsPathVal");
    var raw = String((inp && inp.value) || DSB.dir || "data_src").trim();
    if (!raw) return "data_src";
    var p = raw.replace(/\//g, "\\");
    var i = p.toLowerCase().lastIndexOf("workspace\\");
    if (i >= 0) p = p.slice(i + 10);
    p = p.replace(/^\\+/, "");
    if (p.indexOf(":") >= 0) return "data_src";
    var parts = p.split("\\").filter(Boolean);
    var k = parts.indexOf("data_src");
    if (k >= 0) parts = parts.slice(k);
    return parts.join("/") || "data_src";
  }

  /* ---------- 「这一层有几张」只有一个来源：后端 /api/dataset/count ----------
     路径框右边那个「N 张」以前读 /api/state 的 datasets.src.count，而它永远
     数 data_src/aligned —— 选中 data_src/1（670 张）也照样显示 1203 张，
     运行面板还拿这个数预填成「已读取 1203 张」，跟聚类实际吃的 670 张对不上。
     现在面板一打开、路径一变就问后端要「选中目录这一层」的数。 */
  var DS_COUNT = { arg: "", dir: "", count: 0, skipped: [], recursive: 0 };
  var DS_WARMED = false;

  function dsWarmFaceid() {
    /* 人脸身份第一次点击要 1.5~2 秒加载引擎；面板打开时就提前热起来，
       用户真点下去只剩 0.2~0.3 秒的活。失败无所谓，点击时还会自己起。 */
    if (DS_WARMED) return;
    DS_WARMED = true;
    try {
      fetch("/api/dataset/faceid/prewarm", {
        method: "POST", headers: { "Content-Type": "application/json" }, body: "{}"
      }).catch(function () { DS_WARMED = false; });
    } catch (e) { DS_WARMED = false; }
  }
  window.dsWarmFaceid = dsWarmFaceid;

  function dsRefreshCount(dirArg) {
    var arg = (dirArg == null ? dsDirArg() : dirArg) || "data_src";
    /* 读目录期间把「N 张」变成骨架条：这一块在等后端扫盘，
         其余按钮照常可点 —— 不要用全屏遮挡表示加载。 */
    var tagEl = document.getElementById("dsPathCount");
    if (tagEl) { tagEl.classList.add("sk"); tagEl.textContent = ""; }
    var restore = function () { if (tagEl) tagEl.classList.remove("sk"); };
    return fetch("/api/dataset/count", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ dir: arg })
    }).then(function (r) { return r.json(); }).then(function (d) {
      restore();
      if (!d || d.count == null) return null;
      DS_COUNT = { arg: arg, dir: d.dir || "", count: d.count || 0,
                   skipped: d.skipped_dirs || [], recursive: d.recursive_would_be || 0 };
      var el = document.getElementById("dsPathCount");
      if (el) el.textContent = DS_COUNT.count + " 张";
      dsWarmFaceid();
      return DS_COUNT;
    }).catch(function () {
      if (tagEl) { tagEl.classList.remove("sk"); tagEl.textContent = "—"; }
      return null;
    });
  }
  window.dsRefreshCount = dsRefreshCount;

  var DS_TICK = 0;

  function dsTickStop() {
    if (DS_TICK) { clearInterval(DS_TICK); DS_TICK = 0; }
  }

  function dsTickStart(msg, total) {
    dsTickStop();
    var t0 = Date.now(), pct = 0;
    DS_TICK = setInterval(function () {
      /* 单次阻塞请求拿不到中间进度：百分比不装样子，只把「已用时间」走起来 */
      dsRunPaint({ pct: 0, done: 0, total: total || 0, speed: null,
                   elapsed: (Date.now() - t0) / 1000, msg: msg, running: true, noSample: true });
    }, 220);
  }

  /* 人脸身份：引擎每处理一个 batch 就把进度写进临时文件，这里 400ms 读一次 */
  var DS_POLL = 0, DS_STAGE = "", DS_T0 = 0, DS_MODE = "";

  function dsPollStop() {
    if (DS_POLL) { clearInterval(DS_POLL); DS_POLL = 0; }
  }

  function dsPollStart() {
    dsPollStop();
    DS_STAGE = "";
    DS_T0 = Date.now();
    DS_POLL = setInterval(function () {
      fetch("/api/dataset/faceid/progress").then(function (r) { return r.json(); }).then(function (p) {
        if (!p || !p.ok) return;
        var stage = p.stage || "";
        var done = p.done || 0, total = p.total || 0;
        var sec = (Date.now() - DS_T0) / 1000;
        if (stage !== DS_STAGE) {
          DS_STAGE = stage;
          DS_HIST = [];                    /* 每个阶段各自一条曲线 */
          dsDrawGraph();
          if (stage === "embed") dsRunShow(DS_MODE === "blur" ? "正在统计清晰度" : (DS_MODE === "repeated" ? "正在计算哈希" : "特征提取中"));
          else if (stage === "cluster") dsRunShow("正在聚类");
          sec = 0; DS_T0 = Date.now();
        }
        var pct = total ? done / total * 100 : 0;
        var spdTxt = (sec > 0.3 && done > 0) ? (Math.round(done / sec) + " 张/秒") : "—";
        var spd = (stage === "embed" && sec > 0.3 && done > 0) ? done / sec : null;
        var _msg2 = stage === "cluster" ? "按特征分组…" : (DS_MODE === "blur" ? "计算清晰度…" : (DS_MODE === "repeated" ? "计算感知哈希…" : "提取 ArcFace 特征…"));
        dsRunPaint({ pct: pct, done: done, total: total, speed: spd, elapsed: sec, running: true, msg: _msg2 });
      }).catch(function () {});
    }, 400);
  }

  /* 运行页明细表：扫完/移完把命中项列出来，用户才知道刚才动了哪些文件。
     以前只有数字，没有任何明细，文件被静默移进 _trash。 */


  function dsRunFinish(msg, done, total, t0) {
    /* 扫描太快（命中缓存）时只有一两个采样点，曲线会是一根斜线甚至空着；
       补上起点和终点，形态和提取器一致。 */
    if (DS_HIST.length < 2) {
      /* 扫描太快（缓存命中）时只有一两个采样点。补起终点时**用测得的速度铺平**，
         而不是从 0 起步 —— 否则画出来是一根贴底的水平线接一根竖直尖峰，很怪。 */
      var _spd = (DS_HIST.length && DS_HIST[0].spd) || 0;
      DS_HIST = [{ pct: 0, spd: _spd }, { pct: 100, spd: _spd }];
      dsDrawGraph();
    }
    dsTickStop();
    var sec = t0 ? (Date.now() - t0) / 1000 : 0;
    dsRunPaint({ pct: 100, done: done, total: total, speed: sec >= 0.3 ? done / sec : null,   /* 太快（缓存命中）时速度没意义，显示 — */
                 elapsed: sec.toFixed(1), msg: msg, running: false });

  }

  function dsRunFail(msg, e) {
    dsTickStop();
    var txt = (e && (e.message || e)) + "";
    dsRunPaint({ pct: 100, done: 0, total: 0, speed: 0, elapsed: 0,
                 msg: "失败：" + txt, running: false, ok: false });
    toast("处理失败", txt, "error");

  }

  /* 把界面上的参数映射成后端要的 body —— 之前写死 threshold=10、percent 不传，
     错脸模式永远 0 命中，模糊模式则固定砍 10% */
  function dsScanBody(mode, dirArg) {
    var body = { mode: mode, dir: dirArg };
    function num(k, dft) {
      var v = (typeof PARAM_VAL !== "undefined" ? PARAM_VAL : {})[k];   /* 同作用域，不要走 window */
      return (v === undefined || v === null || v === "") ? dft : Number(v);
    }
    if (mode === "blur") body.percent = num("0-0-0", 10);
    else if (mode === "repeated") body.similarity = num("0-2-0", 0.98);
    else if (mode === "errface") {
      body.threshold = num("0-3-0", 0.5);
      body.percent = num("0-3-1", 0);
    } else if (mode === "faceid") {
      body.eps = num("0-1-0", 0.40);
      body.min_samples = num("0-1-1", 2);
      /* 0 = 只移噪声/杂点（安全，默认）；1 = 只保留最大簇那个身份 */
      /* 人脸身份：每个身份簇一个目录，全部移动 */
      body.split_by_identity = true;
    }
    return body;
  }

  /* 「移走命中项」独立出来：扫描完自动接上，不再让用户点第二次 */
  function dsApply(btn, dirArg, t0) {
    if (btn) { btn.disabled = true; }
    dsTickStart("移动到 _trash…", DSB.hits.length);
    fetch("/api/dataset/apply", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ mode: DSB.mode, dir: dirArg,
                             names: DSB.hits.map(function (x) { return x.name; }) })
    }).then(function (r) { return r.json(); }).then(function (d) {
      if (d && d.detail) throw new Error(String(d.detail));
      var bad = (d.failed || []).length;
      /* 不做撤销：移完即结算，状态回到初始 */
      var _movedHits = DSB.hits.slice();
      DSB.stage = 0; DSB.last = []; DSB.hits = [];
      if (btn) { btn.textContent = "开始处理"; btn.disabled = false; }
      dsPaint(d.moved, "张已移入 _trash", 100);
      var _mvSec = DS_T0 ? Math.max(0.2, (Date.now() - DS_T0) / 1000) : 0;
      toast(bad ? "部分未移走" : "已移走 " + d.moved + " 张",
            "在 _trash 里" + (bad ? " · " + bad + " 张失败" : ""),
            bad ? "warn" : "success");
      dsRunFinish("已移走 " + d.moved + " 张", d.moved, d.moved + bad, t0);
    }).catch(function (e) {
      if (btn) { btn.textContent = "开始处理"; btn.disabled = false; }
      dsRunFail("移动失败", e);
    });
  }

  function dsRun() {
    var mode = dsModeKey();
    DSB.mode = mode;
    var btn = dsBtn();
    var dirArg = dsDirArg();
    var t0 = Date.now();

    /* 遮挡率（排序器）：不筛不删，走自己那条路（算完只排版 + 给个阈值） */
    if (mode === "occlusion") { occScan(btn, dirArg, false); return; }
    occReset();                       /* 换到别的模式：上一轮的排序网格收起来 */

    /* 错脸检测是「能力」：没装就直说怎么装，一个请求都不发（以前这里一路打到
       /api/dataset/scan 吃 409，界面只显示「扫描失败」，看着像坏了）。 */
    if (mode === "errface" && erOff()) { erExplain(); return; }


    /* 合并子文件夹：一次性搬完，没有扫描阶段 */
    if (mode === "merge") {
      if (btn) btn.disabled = true;
      dsRunShow("正在合并子文件夹");
      dsTickStart("把子目录里的图搬回根目录…", 0);
      fetch("/api/dataset/merge-subdirs", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ dir: dirArg })
      }).then(function (r) { return r.json(); }).then(function (d) {
        if (d && d.detail) throw new Error(String(d.detail));
        if (btn) { btn.textContent = "开始处理"; btn.disabled = false; }
        dsPaint(d.merged, "张已并回根目录", 100);
        toast("合并完成",
              "并入 " + d.merged + " 张 · 重名改名 " + (d.renamed || 0) + " 张 · 删除空目录 " +
              ((d.removed || []).length) + " 个",
              (d.errors ? "warn" : "success"), 8000);
        dsRunFinish("合并 " + d.merged + " 张，清理 " + ((d.removed || []).length) + " 个空目录",
                    d.merged, d.merged, t0);
      }).catch(function (e) {
        if (btn) { btn.textContent = "开始处理"; btn.disabled = false; }
        dsRunFail("合并失败", e);
      });
      return;
    }

    /* 扫描（扫完直接移走） */
    if (btn) { btn.disabled = true; }
    dsPaint(null, "正在统计…", 0);
    DSB.stage = 0;
    dsRunShow(MODULES[DS_TAB].n + " · 扫描中");
    /* 先把总数填上：人脸身份要等引擎，光显示「0 / 0」像是卡死。
       **只认后端给的「这个目录这一层」的张数** —— 以前从路径框的
       「N 张」标签里抠数字，而那个数永远数的是 data_src/aligned，
       选中 data_src/1（670 张）也照样显示 1203 张。 */
    var knownTotal = (DS_COUNT.arg === dirArg) ? DS_COUNT.count : 0;
    if (!knownTotal) dsRefreshCount(dirArg);      /* 数还没到手：先要，别拿别的目录凑 */
    if (knownTotal > 0) dsRunPaint({ pct: 2, done: 0, total: knownTotal, speed: null, elapsed: 0,
                                    msg: "已读取 " + knownTotal + " 张" });
    if (mode === "faceid" || mode === "blur" || mode === "repeated") { DS_MODE = mode; dsPollStart(); }
    dsTickStart(mode === "faceid" ? "正在提取人脸特征并聚类…（这一步要跑引擎进程，通常 5~10 秒）"
               : (mode === "errface" ? "正在读取 68 点并分类…" : "正在统计数据集…"), knownTotal);
    fetch("/api/dataset/scan", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify(dsScanBody(mode, dirArg))
    }).then(function (r) { return r.json(); }).then(function (d) {
      dsPollStop();
      /* 能力没装：后端回 200 + unavailable（不是 409）。这里只解释，不进入移动阶段 */
      if (d && d.unavailable) {
        erExplain(d);
        dsTickStop();
        if (btn) { btn.textContent = "开始处理"; btn.disabled = false; }
        return;
      }
      if (d && d.detail) throw new Error(String(d.detail));
      DSB.hits = d.hits || [];
      var sec = ((Date.now() - t0) / 1000).toFixed(1);
      var hd0 = document.querySelector("#dsSteps .card .cardhd .hint");
      if (hd0) {
        /* 三件事分开写清楚：这一层共几张、引擎实际吃了几张、哪些子目录被跳过。
           以前只有一个数，而且可能是别的目录的。 */
        var _nf = (d.info && d.info.n_files != null) ? d.info.n_files : null;
        var _skd = (d.skipped_dirs || []).filter(function (x) { return x.images > 0; });
        var _txt = "共 " + d.total + " 张";
        if (mode === "faceid" && _nf != null) _txt += "，实际参与聚类 " + _nf + " 张";
        _txt += " · 命中 " + (d.count || 0) + " 张";
        if (_skd.length) _txt += " · 跳过 " + _skd.map(function (x) { return x.name + " " + x.images; }).join("、") + " 张";
        hd0.textContent = _txt;
      }
      var sk = (d.info && d.info.skipped_n) || 0;
      var est = (d.info && d.info.estimated_n) || 0;
      if (mode === "errface" && est > 0) {
        toast("已现场补点", est + " 张没有元数据的脸，用 2D106 重新检测了关键点", "info", 6000);
      }
      if (mode === "errface" && sk > 0) {
        var first = ((d.info && d.info.skipped) || [])[0];
        toast("部分图片未参与错脸判定",
              sk + " 张缺有效 landmarks" + (first ? " · 例：" + first.file + " → " + first.why : ""),
              "warn", 7000);
      }
      if (mode === "faceid" && d.info && d.info.split) {
        var sp = d.info.split, dirs = sp.by_dir || {};
        var parts = Object.keys(dirs).sort().map(function (k) { return k + " " + dirs[k] + " 张"; });
        toast("已按身份分文件夹",
              (parts.length ? parts.join(" · ") : "没有可分的组") + "（都在 _by_id/ 下）",
              "success", 9000);
        DSB.hits = [];
        DSB.stage = 0;
        if (btn) { btn.textContent = "开始处理"; btn.disabled = false; }
        dsPaint(sp.moved, "张已分到 _by_id/ 下的 " + Object.keys(dirs).length + " 个目录", 100);
        dsRunFinish("分成 " + Object.keys(dirs).length + " 组，移走 " + sp.moved + " 张", sp.moved, sp.moved, t0);
        return;
      }
      if (mode === "faceid" && d.info && d.info.clusters) {
        var cl = d.info.clusters, ids = Object.keys(cl).length, noisy = d.info.noise || 0;
        var mainN = (d.info.main && d.info.main.n) || 0;
        var otherN = Math.max(0, (d.total || 0) - mainN);
        toast("身份聚类完成",
              "主体 " + mainN + " 张 · 其他身份 " + otherN + " 张已移出 · " + ids + " 簇" +
              (noisy ? " · 噪声 " + noisy : ""),
              otherN ? "warn" : "success", 8000);
      }
      if (!DSB.hits.length) {
        DSB.stage = 0;
        if (btn) { btn.textContent = "开始处理"; btn.disabled = false; }
        dsPaint(0, "共 " + d.total + " 张，没有命中 · " + sec + " 秒", 0);
        toast("扫描完成", "没有需要处理的图片", "info");
        dsRunFinish("扫描完成 · 没有命中", d.total, d.total, t0);
        return;
      }
      /* 默认移走：扫完立刻进入移动阶段，同一次点击完成 */
      DSB.stage = 1;
      dsRunPaint({ pct: 100, done: d.total, total: d.total, speed: null,
                   elapsed: (Date.now() - t0) / 1000,
                   msg: "扫描完成 · 命中 " + d.count + " 张，正在移走…", running: true });
      dsApply(btn, dirArg, t0);
    }).catch(function (e) {
      dsPollStop();
      if (btn) { btn.textContent = "开始处理"; btn.disabled = false; }
      dsRunFail("扫描失败", e);
    });
  }


  /* ---------- 遮挡率排序器（排序，不是过滤器） ----------
     口径全在后端（app/server/_occlusion.py）：
       occ = 1 - sum(mask[凸包]) / count(凸包)，凸包 = 68 点里去掉眉毛 17..26
     100% = 眼睛及以下整块被挡住，0% = 一点没挡。
     这里只做三件事：排版（角标 + 统计）、阈值实时计数、把超阈值的移进 _trash。 */
  var OCC = { data: null, dir: "", thr: 0.30, last: null, busy: false, bound: false };
  var OCC_MAXCELLS = 1200;          /* 网格上限：再多 DOM 就卡，超出用一行字说明 */

  var OCC_REASON = {
    no_landmarks: "没有 68 点 landmarks（不是 DFLJPG）",
    no_mask: "没有 xseg 遮罩（APP15 里没有 xseg_mask）",
    hull_too_small: "凸包面积小于图面积 0.5%（点集失效）",
    unreadable: "文件读不出来或图片解不开"
  };

  function occPct(v) {
    return (v == null || isNaN(Number(v))) ? "—" : (Math.round(Number(v) * 100) + "%");
  }
  function occBadgeCls(v) {
    return v >= 0.5 ? "t-red" : (v >= 0.25 ? "t-yellow" : "t-green");
  }
  function occEl(id) { return document.getElementById(id); }

  /* 收起面板（换模式 / 开始新一轮时调用）：数据也清掉，免得角标和统计对不上 */
  function occReset() {
    OCC.data = null;
    OCC.last = null;
    var card = occEl("occCard");
    if (card) card.classList.add("hide");
    var g = occEl("occGrid");
    if (g) g.innerHTML = "";
    var un = occEl("occUnranked");
    if (un) un.innerHTML = "";
    var undo = occEl("occUndo");
    /* 用 hidden 属性而不是 .hide：.btn 的 display 带 !important，
       和 .hide 同权重、后出现，靠 classList 根本收不起来（.mini 踩过同一个坑） */
    if (undo) undo.hidden = true;
  }
  window.occReset = occReset;

  function occBind() {
    if (OCC.bound) return;
    OCC.bound = true;
    var sl = occEl("occThr"), num = occEl("occThrNum");
    function apply(v, from) {
      var n = Math.round(Number(v));
      if (!isFinite(n)) n = 30;
      n = Math.max(0, Math.min(100, n));
      OCC.thr = n / 100;
      if (sl && from !== "sl") sl.value = n;
      if (num && from !== "num") num.value = n;
      occCount();
    }
    if (sl) sl.addEventListener("input", function () { apply(sl.value, "sl"); });
    if (num) num.addEventListener("input", function () { apply(num.value, "num"); });
    if (occEl("occMove")) occEl("occMove").addEventListener("click", occAskMove);
    if (occEl("occUndo")) occEl("occUndo").addEventListener("click", occUndo);
    apply(OCC.thr * 100, "");
  }

  /* 超阈值 = 严格大于（和后端 move 的判据一致，数目才可能一模一样） */
  function occHits() {
    if (!OCC.data) return [];
    return (OCC.data.items || []).filter(function (it) {
      return it.valid && it.occ > OCC.thr;
    });
  }

  function occCount() {
    var n = occHits().length;
    var thr = Math.round(OCC.thr * 100);
    var c = occEl("occCnt");
    if (c) c.textContent = "超过 " + thr + "% 的有 " + n + " 张";
    var mv = occEl("occMove");
    if (mv) {
      mv.textContent = "移走这 " + n + " 张";
      mv.disabled = (n === 0) || OCC.busy;
    }
    /* 会在网格上标出来的那一批：描一圈，用户一眼看到"移走的是哪几张" */
    var cells = document.querySelectorAll("#occGrid .occcell");
    Array.prototype.forEach.call(cells, function (cd) {
      cd.classList.toggle("hit", Number(cd.dataset.occ) > OCC.thr);
    });
  }

  function occStat(k, v, unit) {
    return '<div class="stat"><div class="k">' + k + '</div><div class="v">' + v +
           (unit ? '<small>' + unit + '</small>' : '') + '</div></div>';
  }

  function occRender(d) {
    occBind();
    var card = occEl("occCard");
    if (!card) return;
    card.classList.remove("hide");
    var st = d.stats || {};
    var unranked = st.unranked || 0;

    var hint = occEl("occHint");
    if (hint) {
      hint.textContent = "共 " + (d.total || 0) + " 张 · 有效 " + (d.valid || 0) +
                         " · 未参与 " + unranked + " · 用时 " + (d.ms || 0) + " 毫秒";
    }

    var stats = occEl("occStats");
    if (stats) {
      stats.innerHTML =                       /* occPct 自带 %，单位别再拼一遍 */
        occStat("中位", occPct(st.median), "") +
        occStat("p90", occPct(st.p90), "") +
        occStat("最大", occPct(st.max), "") +
        occStat("未参与", String(unranked), "张");
    }

    var grid = occEl("occGrid");
    if (grid) {
      grid.innerHTML = "";
      var list = (d.items || []).filter(function (it) { return it.valid; });
      var shown = list.slice(0, OCC_MAXCELLS);
      var frag = document.createDocumentFragment();
      var q = "/api/dataset/thumb?dir=" + encodeURIComponent(OCC.dir) + "&s=96&name=";
      shown.forEach(function (it) {
        var cell = document.createElement("div");
        cell.className = "occcell";
        cell.dataset.occ = it.occ;
        cell.title = it.name + " · 遮挡 " + occPct(it.occ) +
                     "（凸包占图 " + occPct(it.hull_ratio) + "）";
        var img = document.createElement("img");
        img.loading = "lazy";
        img.decoding = "async";
        img.alt = "";
        img.src = q + encodeURIComponent(it.name);
        var tag = document.createElement("span");
        tag.className = "tag occb " + occBadgeCls(it.occ);
        tag.textContent = occPct(it.occ);
        cell.appendChild(img);
        cell.appendChild(tag);
        frag.appendChild(cell);
      });
      grid.appendChild(frag);
      /* 上一轮那行说明先摘掉：rescan 之后它要么重复、要么说的是旧张数 */
      var old = document.querySelector("#occCard .occmore");
      if (old && old.parentNode) old.parentNode.removeChild(old);
      if (list.length > shown.length) {
        var more = document.createElement("div");
        more.className = "hint occmore";
        more.textContent = "网格只画前 " + shown.length + " 张（共 " + list.length + " 张），排序与统计是全量的";
        grid.parentNode.insertBefore(more, grid.nextSibling);
      }
    }

    var un = occEl("occUnranked");
    if (un) {
      un.innerHTML = "";
      var bad = (d.items || []).filter(function (it) { return !it.valid; });
      if (bad.length) {
        var by = {};
        bad.forEach(function (it) { (by[it.reason] = by[it.reason] || []).push(it.name); });
        var hd = document.createElement("div");
        hd.className = "cardhd";
        hd.innerHTML = '<b>未参与排序 ' + bad.length + ' 张</b>' +
                       '<span class="hint">没有凸包/遮罩的图算不出遮挡率，不参与统计</span>';
        un.appendChild(hd);
        Object.keys(by).sort().forEach(function (k) {
          var row = document.createElement("div");
          row.className = "occunrow";
          var names = by[k];
          var eg = names.slice(0, 3).join("、") + (names.length > 3 ? " …" : "");
          row.innerHTML = '<span class="tag t-gray">' + names.length + '</span>' +
                          '<span class="ocr">' + (OCC_REASON[k] || k) + '</span>' +
                          '<span class="hint">例：' + eg + '</span>';
          un.appendChild(row);
        });
      }
    }
    occCount();
  }

  function occScan(btn, dirArg, quiet) {
    if (OCC.busy) return;
    OCC.busy = true;
    OCC.dir = dirArg || OCC.dir || "data_src";
    var t0 = Date.now();
    if (btn) btn.disabled = true;
    if (!quiet) {
      occReset();
      dsRunShow("遮挡率排序");
      dsTickStart("读 landmarks 与 XSeg 遮罩…", 0);
    }
    fetch("/api/dataset/occlusion", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ dir: OCC.dir, action: "scan" })
    }).then(function (r) { return r.json(); }).then(function (d) {
      OCC.busy = false;
      dsTickStop();
      if (btn) { btn.textContent = "开始处理"; btn.disabled = false; }
      if (!d || d.ok !== true) throw new Error((d && d.detail) || "扫描失败");
      OCC.data = d;
      occRender(d);
      var un = (d.stats && d.stats.unranked) || 0;
      if (!quiet) {
        dsRunFinish("有效 " + (d.valid || 0) + " 张 · 未参与 " + un + " 张", d.total, d.total, t0);
        toast("遮挡率已排好",
              (d.valid || 0) + " 张可排序" + (un ? " · " + un + " 张缺数据" : "") +
              " · " + (d.ms || 0) + " 毫秒",
              (d.valid ? "success" : "warn"), 6000);
      }
    }).catch(function (e) {
      OCC.busy = false;
      dsTickStop();
      if (btn) { btn.textContent = "开始处理"; btn.disabled = false; }
      /* 失败也必须落到终态（running=false），否则进度条永远停在"处理中"。
         quiet 是"移走/还原之后的重扫"，那一次的运行卡片已经是完成态，
         不能再被改写成失败 —— 只弹一句提示。 */
      if (quiet) toast("重扫失败", String((e && e.message) || e), "error");
      else dsRunFail("遮挡率扫描失败", e);
    });
  }
  window.occScan = occScan;

  function occAskMove() {
    var hits = occHits();
    if (!hits.length || OCC.busy) return;
    var thr = Math.round(OCC.thr * 100);
    confirmBox("移走这 " + hits.length + " 张？",
               "遮挡率都超过 " + thr + "%。文件只会移动到 _trash 里（不删除），" +
               "移完可以点「撤销」原地还原。",
               "移走", function () { occDoMove(); }, true);
  }

  function occDoMove() {
    if (OCC.busy) return;
    OCC.busy = true;
    var mv = occEl("occMove");
    if (mv) { mv.disabled = true; mv.textContent = "正在移走…"; }
    fetch("/api/dataset/occlusion", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ dir: OCC.dir, action: "move", threshold: OCC.thr, target: "_trash" })
    }).then(function (r) { return r.json(); }).then(function (d) {
      OCC.busy = false;
      if (!d || d.ok !== true) throw new Error((d && d.detail) || "移动失败");
      OCC.last = d;
      var mn = String(d.manifest || "").split(/[\\/]/).pop();
      var bad = (d.failed || []).length;
      toast("已移走 " + d.moved + " 张",
            "在 _trash 里 · 清单 " + (mn || "—") + (bad ? " · " + bad + " 张失败" : ""),
            bad ? "warn" : "success", 9000);
      var undo = occEl("occUndo");
      if (undo) undo.hidden = false;
      dsRefreshCount(OCC.dir);
      occScan(null, OCC.dir, true);          /* 重新扫：网格/统计/计数跟着目录状态走 */
    }).catch(function (e) {
      OCC.busy = false;
      toast("移动失败", String((e && e.message) || e), "error");
      occCount();
    });
  }

  function occUndo() {
    if (OCC.busy || !OCC.last) return;
    var items = (OCC.last.items || []).map(function (x) {
      return { name: x.name, trash: x.trash };
    });
    if (!items.length) return;
    OCC.busy = true;
    fetch("/api/dataset/restore", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ dir: OCC.dir, items: items })
    }).then(function (r) { return r.json(); }).then(function (d) {
      OCC.busy = false;
      if (!d || d.ok !== true) throw new Error((d && d.detail) || "还原失败");
      var bad = (d.failed || []).length;
      toast("已还原 " + d.restored + " 张", bad ? bad + " 张没还原（可能重名了）" : "文件都回原位了",
            bad ? "warn" : "success", 7000);
      OCC.last = null;
      var undo = occEl("occUndo");
      if (undo) undo.hidden = true;
      dsRefreshCount(OCC.dir);
      occScan(null, OCC.dir, true);
    }).catch(function (e) {
      OCC.busy = false;
      toast("还原失败", String((e && e.message) || e), "error");
    });
  }

  var DS_OPEN_AT = 0;
  function dsOpen() {
    var now = Date.now();
    if (now - DS_OPEN_AT < 800) return;      /* 防误触 / 重复触发 */
    DS_OPEN_AT = now;
    dsRefreshCount();                        /* 顺手把「这一层几张」和引擎都热起来 */
    /* 没有空态了：路径框和工作区常在，这里只负责重新选目录 */
    var path = $("dspath"), work = $("dswork");
    if (path) path.classList.add("in");
    if (work) work.classList.add("in");
    if (window.toast) toast("选择数据集", "文件选择器还没接 · 也可以直接把文件夹拖到这个框里", "info");
  }
  function dsReset() {
    /* 重置 = 回到第一个模块，不动路径框 */
    var l2 = $("lvl2"), l3 = $("lvl3");
    if (l2) l2.classList.remove("in");
    if (l3) l3.classList.remove("in");
    renderSteps();
  }
  window.dsOpen = dsOpen;
  window.dsReset = dsReset;

  var btnEx = $("btnExtract");
  var es = null;

  function showProgress(on) {
    var p = $("exProgress");
    if (!p) return;
    if (on) {
      p.classList.remove("hide");
      if (!reduced()) {
        p.animate([{ opacity: 0, transform: "translateY(6px)" }, { opacity: 1, transform: "none" }],
          { duration: 200, easing: PAGE_EASE });
      }
    } else {
      p.classList.add("hide");
    }
  }

  /* 秒 → 1:23 / 12 秒 */
  function fmtEta(sec) {
    sec = Math.max(0, Math.round(sec || 0));
    if (sec < 60) return sec + " 秒";
    var m = Math.floor(sec / 60), s2 = sec % 60;
    if (m < 60) return m + ":" + (s2 < 10 ? "0" : "") + s2;
    return Math.floor(m / 60) + " 小时 " + (m % 60) + " 分";
  }

  var STAGE_NAME = ["抽帧", "人脸对齐", "应用遮罩", "完成"];

  /* 速度曲线：横轴 = 进度（0~100%），纵轴 = 帧/秒，带刻度和网格 */
  var EX_HIST = [];          /* [{pct, fps}] */

  function drawSpeedGraph() {
    var cv = document.getElementById("exGraph");
    if (!cv || !cv.getContext) return;
    var dpr = window.devicePixelRatio || 1;
    var W = cv.clientWidth || 640;
    var H = cv.clientHeight || 180;
    if (cv.width !== Math.round(W * dpr) || cv.height !== Math.round(H * dpr)) {
      cv.width = Math.round(W * dpr);
      cv.height = Math.round(H * dpr);
    }
    var g = cv.getContext("2d");
    g.setTransform(dpr, 0, 0, dpr, 0, 0);
    g.clearRect(0, 0, W, H);

    /* 颜色/字体都跟随主题，别写死 rgba(0,0,0)：深色模式会看不见 */
    var cs = getComputedStyle(document.documentElement);
    var cInk3 = (cs.getPropertyValue("--ink-3") || "#8b8b98").trim();
    var cLine = (cs.getPropertyValue("--line") || "rgba(0,0,0,.08)").trim();
    var cBrand = (cs.getPropertyValue("--brand") || "#5d6ccf").trim();
    var cFont = (cs.getPropertyValue("--font") || "system-ui").trim();

    var padL = 58, padR = 14, padT = 12, padB = 24;
    var plotW = W - padL - padR, plotH = H - padT - padB;

    var peak = 1;
    EX_HIST.forEach(function (p) { if (p.fps > peak) peak = p.fps; });
    if (EX_D && EX_D.fps > peak) peak = EX_D.fps;
    /* 顶部留 20% 余量：曲线不会顶到框，视觉上不憋 */
    var maxFps = Math.max(10, Math.ceil(peak * 1.2 / 10) * 10);

    /* 网格 + 纵轴刻度（0 / 一半 / 最大值） */
    g.font = "10px " + cFont;
    g.textAlign = "right";
    g.textBaseline = "middle";
    [0, 0.25, 0.5, 0.75, 1].forEach(function (k) {
      var y = padT + plotH * (1 - k);
      g.strokeStyle = cLine;
      g.lineWidth = 1;
      g.beginPath(); g.moveTo(padL, y + 0.5); g.lineTo(W - padR, y + 0.5); g.stroke();
      g.fillStyle = cInk3;
      g.fillText(String(Math.round(maxFps * k)), padL - 10, y);
    });

    /* 横轴刻度（0% 25% 50% 75% 100%） */
    g.textAlign = "center";
    g.textBaseline = "top";
    [0, 25, 50, 75, 100].forEach(function (pct) {
      var x = padL + plotW * pct / 100;
      g.strokeStyle = cLine;
      g.beginPath(); g.moveTo(x + 0.5, padT); g.lineTo(x + 0.5, padT + plotH); g.stroke();
      g.fillStyle = cInk3;
      g.fillText(pct + "%", x, padT + plotH + 6);
    });

    if (EX_HIST.length < 1) return;

    /* 末端用缓动中的实时值补一个点：两包 SSE 之间线条也会继续长 */
    var pts = EX_HIST.slice();
    var lastPct = pts.length ? pts[pts.length - 1].pct : -1;
    if (EX_D && EX_D.pct > lastPct + 0.05) {
      pts.push({ pct: EX_D.pct, fps: EX_D.fps });
    }
    if (pts.length < 2) return;

    function pt(p) {
      return [padL + plotW * Math.min(100, Math.max(0, p.pct)) / 100,
              padT + plotH * (1 - Math.min(1, p.fps / maxFps))];
    }

    /* 填充 */
    g.beginPath();
    var first = pt(pts[0]);
    g.moveTo(first[0], padT + plotH);
    pts.forEach(function (p) { var q = pt(p); g.lineTo(q[0], q[1]); });
    var last = pt(pts[pts.length - 1]);
    g.lineTo(last[0], padT + plotH);
    g.closePath();
    g.fillStyle = "rgba(93,108,207,0.14)";
    g.fill();

    /* 折线 */
    g.beginPath();
    pts.forEach(function (p, i) { var q = pt(p); i ? g.lineTo(q[0], q[1]) : g.moveTo(q[0], q[1]); });
    g.strokeStyle = cBrand;
    g.lineWidth = 1.8;
    g.lineJoin = "round";
    g.stroke();
  }

  /* 数值缓动：SSE 每 ~0.5s 推一次，直接赋值会一跳一跳。
     这里用 rAF 把显示值平滑逼近目标值，图表末端也跟着动。 */
  var EX_T = null;                       /* 目标值 */
  var EX_D = { pct: 0, fps: 0, faces: 0, done: 0, faceRate: 0 };
  var EX_RAF = 0;

  function exEase(cur, target, k) {
    var d = target - cur;
    if (Math.abs(d) < 0.01) return target;
    return cur + d * k;
  }

  function exTick() {
    if (!EX_T) { EX_RAF = 0; return; }
    try {
    var k = 0.06;   /* 追平约 0.6 秒，视觉上是滑过去的 */
    EX_D.pct = exEase(EX_D.pct, EX_T.pct, k);
    EX_D.fps = exEase(EX_D.fps, EX_T.fps, k);
    EX_D.faces = exEase(EX_D.faces, EX_T.faces, k);
    EX_D.done = exEase(EX_D.done, EX_T.done, k);
    EX_D.faceRate = exEase(EX_D.faceRate, EX_T.faceRate, k);

    if ($("exFps")) $("exFps").textContent = EX_D.fps.toFixed(1);
    if ($("exFaceRate")) $("exFaceRate").textContent = EX_D.faceRate.toFixed(1);
    if ($("exFaces")) $("exFaces").textContent = Math.round(EX_D.faces);
    if ($("exFiles")) $("exFiles").textContent = Math.round(EX_D.done);
    if ($("exFilesTotal")) $("exFilesTotal").textContent = "/ " + (EX_T.total || 0);

    drawSpeedGraph();

    var settled = Math.abs(EX_D.pct - EX_T.pct) < 0.4 &&
                  Math.abs(EX_D.fps - EX_T.fps) < 0.2 &&
                  Math.abs(EX_D.done - EX_T.done) < 1;
    if (settled && !EX_T.running) { EX_RAF = 0; return; }
    EX_RAF = requestAnimationFrame(exTick);
    } catch (err) {
      /* 绘制/取值出错也不能让循环停摆，否则界面就“冻”住了 */
      console.error("exTick", err);
      EX_RAF = requestAnimationFrame(exTick);
    }
  }

  function exKick() {
    if (!EX_RAF) EX_RAF = requestAnimationFrame(exTick);
  }

  /* ---------- 数据处理运行页 ---------- */
  function dsRunEl(id) { return document.getElementById(id); }

  function dsRunShow(title) {
    var card = dsRunEl("dsRunCard");
    if (card) card.classList.remove("hide");
    var t = dsRunEl("dsRunTitle");
    if (t) t.textContent = title || "正在处理";
    var st = dsRunEl("btnDsRunStop");
    if (st) { st.classList.remove("hide"); st.disabled = false; st.textContent = "中断"; st.dataset.mode = "stop"; }
    if (dsRunEl("dsRunMsg")) dsRunEl("dsRunMsg").textContent = "准备中…";
    if (dsRunEl("dsRunDone")) dsRunEl("dsRunDone").textContent = "0";
    if (dsRunEl("dsRunTotal")) dsRunEl("dsRunTotal").textContent = "/ 0";
    if (dsRunEl("dsRunSpeed")) dsRunEl("dsRunSpeed").textContent = "0";
    if (dsRunEl("dsRunElapsed")) dsRunEl("dsRunElapsed").textContent = "0";
    if (dsRunEl("dsRunEta")) dsRunEl("dsRunEta").textContent = "—";
    if (dsRunEl("dsRunRate")) dsRunEl("dsRunRate").textContent = "";
    DS_HIST = [];                 /* 新一次运行从零开始，别把上一轮的曲线留着 */
    DS_HIST_FROZEN = false;       /* 采样闩一起清，新一轮才采得到样 */
    DS_HIST_FINAL = false;        /* 定稿闩一起清，新曲线才有资格被补端点 */
    dsDrawGraph();
    go("dataset-run");
  }

  /* ---------- 速度曲线采样 ---------- */
  var DS_HIST_MAX = 240;        /* 最多 240 点，页面开着久了也不会无限涨 */

  /* 采一个点。返回 true 表示 DS_HIST 变了、需要重画。
     规则：
     - 第一个真实采样点之前固定插入 (0%,0) —— 曲线从原点出发，不是从半空开始；
     - x 只增不减：乱序旧帧直接丢；同一百分比用最新真实值替换，不堆竖线；
     - 0% 的瞬时速度没有信息量（那是计时器刚起跳），丢掉，起点已经是 (0,0)。 */
  function dsHistPush(pct, spd) {
    if (DS_HIST_FROZEN) return false;
    var y = Number(spd) || 0;
    if (!(y > 0)) return false;
    var x = Math.max(0, Math.min(100, Number(pct) || 0));
    var changed = false;
    if (!DS_HIST.length) { DS_HIST.push({ pct: 0, spd: 0 }); changed = true; }
    if (x <= 0) return changed;                 /* 0% 起跳点：起点已经钉好 */
    var last = DS_HIST[DS_HIST.length - 1];
    if (x < last.pct) return changed;           /* 迟到的旧帧：别把 x 拉回去 */
    if (x === last.pct) {
      if (last.spd === y) return changed;
      last.spd = y;                             /* 同一进度：保留最新真实值 */
      return true;
    }
    DS_HIST.push({ pct: x, spd: y });
    /* 超上限先丢第二个点：起点 (0,0) 必须留在图上 */
    if (DS_HIST.length > DS_HIST_MAX) DS_HIST.splice(1, 1);
    return true;
  }

  function dsRunPaint(p) {
    if (!p) return;
    /* 采样：给速度曲线用（最多 240 点，页面开着久了也不会无限涨） */
    /* 只采「有真实速度」的点：起手那几帧 speed 是 null，采进去曲线会从 0 起飞。
       第一个真实采样点之前先钉一个 (0%,0) 起点（dsHistPush 负责），曲线必定从原点
       出发 —— 以前秒杀任务只采到完成帧，曲线就只剩 x=100 上的一根竖线。
       收尾帧只采一次（采完立刻定稿），定稿之后一个点都不再采 —— 完成后每轮算出来
       的速度只会更小，再采下去就是在 x=100 上继续往下画，用户看到的就是
       「跑完了曲线还在掉」。 */
    if (p.pct != null && !p.noSample && !DS_HIST_FROZEN && Number(p.speed) > 0 &&
        dsHistPush(p.pct, p.speed)) {
      dsDrawGraph();
    }
    if (dsRunEl("dsRunMsg")) dsRunEl("dsRunMsg").textContent = p.msg || "";
    if (dsRunEl("dsRunDone")) dsRunEl("dsRunDone").textContent = p.done || 0;
    if (dsRunEl("dsRunTotal")) dsRunEl("dsRunTotal").textContent = "/ " + (p.total || 0);
    if (dsRunEl("dsRunSpeed")) dsRunEl("dsRunSpeed").textContent = (p.speed == null) ? "—" : dsFmtSpd(p.speed);
    if (dsRunEl("dsRunElapsed")) dsRunEl("dsRunElapsed").textContent = dsFmtSec(p.elapsed);
    if (dsRunEl("dsRunEta")) dsRunEl("dsRunEta").textContent = (p.eta ? Math.round(p.eta) + "" : "—");
    if (dsRunEl("dsRunRate")) {
      var bits = [];
      if (p.speed) bits.push(dsFmtSpd(p.speed) + " 张/秒");
      if (p.eta) bits.push("剩 " + Math.round(p.eta) + " 秒");
      dsRunEl("dsRunRate").textContent = bits.join(" · ");
    }
    if (!p.running) {
      /* 终态一次性钉死：标题、返回按钮、剩余时间。之后的帧不会再改动这些。 */
      var t = dsRunEl("dsRunTitle");
      if (t) t.textContent = (p.ok === false) ? "处理失败" : "处理完成";
      var st = dsRunEl("btnDsRunStop");
      if (st) {
        st.dataset.mode = "back";
        st.classList.remove("danger");
        st.classList.remove("hide");
        st.disabled = false;
        st.textContent = "返回";
      }
      if (dsRunEl("dsRunEta")) dsRunEl("dsRunEta").textContent = "—";
      DS_HIST_FROZEN = true;          /* 终态：此后不再采样（历史本身不动） */
    }

  }
  window.dsRunPaint = dsRunPaint;

  var DS_HIST = [];
  /* 采样闩：dsRunPaint 见到终态帧就置 true，不再追加采样；dsRunShow 清回 false。
     它只挡「之后的追加」，不动已有历史。 */
  var DS_HIST_FROZEN = false;
  /* 定稿闩：dsFreezeGraph 只允许把一条曲线补完一次，重复调用不改任何数 */
  var DS_HIST_FINAL = false;

  /* 速度显示：大数取整，小数一位 —— 之前把 2524.1090146750525 原样贴出来了 */
  function dsFmtSpd(v) {
    var n = Number(v) || 0;
    return n >= 100 ? String(Math.round(n)) : n.toFixed(1);
  }

  function dsFmtSec(s) {
    var n = Number(s) || 0;
    return n < 10 ? n.toFixed(2) : n.toFixed(1);
  }

  function dsDrawGraph() {
    var cv = document.getElementById("dsRunGraph");
    if (!cv || !cv.getContext) return;
    /* 采样不足（单次阻塞请求）时不画空图表 */
    cv.style.display = DS_HIST.length >= 1 ? "block" : "none";
    var dpr = window.devicePixelRatio || 1;
    var W = cv.clientWidth || 640, H = cv.clientHeight || 180;
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

    var padL = 58, padR = 14, padT = 12, padB = 24;
    var plotW = W - padL - padR, plotH = H - padT - padB;

    var peak = 1;
    DS_HIST.forEach(function (p) { if (p.spd > peak) peak = p.spd; });
    var maxSpd = Math.max(10, Math.ceil(peak * 1.2 / 10) * 10);

    g.font = "10px " + cFont;
    g.textAlign = "right";
    g.textBaseline = "middle";
    [0, 0.25, 0.5, 0.75, 1].forEach(function (k) {
      var y = padT + plotH * (1 - k);
      g.strokeStyle = cLine;
      g.lineWidth = 1;
      g.beginPath(); g.moveTo(padL, y + 0.5); g.lineTo(W - padR, y + 0.5); g.stroke();
      g.fillStyle = cInk3;
      g.fillText(String(Math.round(maxSpd * k)), padL - 10, y);
    });

    g.textAlign = "center";
    g.textBaseline = "top";
    [0, 25, 50, 75, 100].forEach(function (pct) {
      var x = padL + plotW * pct / 100;
      g.strokeStyle = cLine;
      g.beginPath(); g.moveTo(x + 0.5, padT); g.lineTo(x + 0.5, padT + plotH); g.stroke();
      g.fillStyle = cInk3;
      g.fillText(pct + "%", x, padT + plotH + 6);
    });

    if (DS_HIST.length < 2) return;
    function pt(p) {
      return [padL + plotW * Math.min(100, Math.max(0, p.pct)) / 100,
              padT + plotH * (1 - Math.min(1, p.spd / maxSpd))];
    }
    g.beginPath();
    var first = pt(DS_HIST[0]);
    g.moveTo(first[0], padT + plotH);
    DS_HIST.forEach(function (p) { var q = pt(p); g.lineTo(q[0], q[1]); });
    var lastP = pt(DS_HIST[DS_HIST.length - 1]);
    g.lineTo(lastP[0], padT + plotH);
    g.closePath();
    g.fillStyle = "rgba(93,108,207,0.14)";
    g.fill();

    g.beginPath();
    DS_HIST.forEach(function (p, i) { var q = pt(p); i ? g.lineTo(q[0], q[1]) : g.moveTo(q[0], q[1]); });
    g.strokeStyle = cBrand;
    g.lineWidth = 1.8;
    g.lineJoin = "round";
    g.stroke();
  }
  /* 曲线定稿：补齐端点 → 稀疏时按 done 比例补中间点 → 冻结采样。
     三条规矩：
     1) 只补点，不改写已有历史 —— 冻结只影响之后的追加（以前把最后一个点直接
        改成平均值，历史就这么被改掉了）；
     2) 采样太稀（<3 点，例如 669 张 / 473 张每秒 ≈ 1.4 秒的秒杀任务，500ms 一次
        只来得及采到完成帧）时，从 (0,0) 线性补到 (100, 最终平均速度)；中间点的
        速度按 done 比例插值，只有真实端点数据，不编造任何抖动；
     3) 完成后不再追加，重复调用不改历史的任何一个数。 */
  var DS_SYNTH_STEPS = 4;       /* 稀疏曲线 4 段：0 / 25 / 50 / 75 / 100 */

  function dsFreezeGraph(spd) {
    if (DS_HIST_FINAL) { dsDrawGraph(); return; }    /* 已定稿：再来一次也不动历史 */
    var v = Number(spd);
    if (!isFinite(v) || v < 0) v = 0;

    /* 起点：固定 (0,0)。缺就补，已有的点一个不动。 */
    if (!DS_HIST.length) DS_HIST.push({ pct: 0, spd: 0 });
    else if (DS_HIST[0].pct > 0) DS_HIST.unshift({ pct: 0, spd: 0 });

    /* 太稀：真实数据只有 (0,0) 加最多一个采样点，用终值铺一条直线过去 */
    if (DS_HIST.length < 3) {
      var endSpd = v > 0 ? v : (DS_HIST.length > 1 ? DS_HIST[DS_HIST.length - 1].spd : 0);
      if (endSpd > 0) {
        var line = [];
        for (var k = 0; k <= DS_SYNTH_STEPS; k++) {
          line.push({ pct: 100 * k / DS_SYNTH_STEPS, spd: endSpd * k / DS_SYNTH_STEPS });
        }
        DS_HIST = line;
      }
    }

    /* 终点：追加 (100, 最终平均)。最后一个真实采样点保持原值；同 x 再补一个同值点，
       最后一段斜率是 0（平收尾），不会在完成后再往下掉。 */
    var last = DS_HIST[DS_HIST.length - 1];
    var tailSpd = v > 0 ? v : last.spd;
    DS_HIST.push({ pct: 100, spd: tailSpd });
    if (DS_HIST.length > DS_HIST_MAX) DS_HIST.splice(1, DS_HIST.length - DS_HIST_MAX);

    DS_HIST_FROZEN = true;
    DS_HIST_FINAL = true;
    dsDrawGraph();
  }

  window.dsRunShow = dsRunShow;

  /* 「中断 / 返回」：这个按钮以前没绑任何事件，点了没反应 */
  (function bindDsRunStop() {
    var st = dsRunEl("btnDsRunStop");
    if (!st) return;
    st.addEventListener("click", function () {
      dsTickStop();
      if (st.dataset.mode === "back") { go("dataset"); return; }
      go("dataset");
      toast("已离开运行页", "扫描/移动会在后台继续完成", "info");
    });
  })();

  /* 侧栏「任务运行中」标记：百分比胶囊（展开时在右侧，折叠时贴图标角） */
  function setNavRunning(key, on, pct) {
    document.querySelectorAll('.nav[data-go="' + key + '"]').forEach(function (n) {
      var pill = n.querySelector(".navpct");
      if (!on) {
        n.classList.remove("running");
        if (pill) pill.remove();
        return;
      }
      if (!pill) {
        pill = document.createElement("span");
        pill.className = "navpct";
        n.appendChild(pill);
      }
      var v = (pct === null || pct === undefined) ? "" : (Math.round(pct) + "%");
      if (pill.textContent !== v) pill.textContent = v;
      n.classList.add("running");
    });
  }
  window.setNavRunning = setNavRunning;

  function setExtractUI(s) {
    /* 失败态：后端 Job.fail 给的是 ok=false + error（app.py）。以前这里只看
       running —— 线程一死就写「提取完成」，用户以为跑完了，这就是事故现场。 */
    var failed = (s.ok === false || !!s.error);
    if (!s.running) {
      /* 结束：运行视图仍可查看，但侧栏点「提取器」应回到参数页 */
      delete SUB_MEMORY["extract"];
      if (LAST_PAGE["extract"] === "extract-run") LAST_PAGE["extract"] = "extract";
      /* 结束：标题改「提取完成」，按钮从「中断提取」变成「返回」 */
      var hd0 = document.querySelector("#exProgress .xferhd b");
      if (hd0) hd0.textContent = failed ? "提取失败" : "提取完成";
      var sb0 = document.getElementById("btnExStop");
      if (sb0) {
        sb0.dataset.mode = "back";
        sb0.classList.remove("danger", "pri", "hide");
        sb0.disabled = false;
        sb0.textContent = "返回";
      }
    }
    /* 显示值不再直接写 DOM：把它交给缓动循环，避免每 0.5s 一跳 */
    /* 人脸速度与帧速度同口径：分子是**本次运行**新落盘的脸（后端 faces_new），
       分母是 SSE 里的 elapsed（和 fps 用的是同一个 JOB.t0）。有后端的 face_rate
       就用它，没有（老后端）才退回 faces/elapsed。 */
    var faceRate = (typeof s.face_rate === "number" && s.elapsed > 0.3)
      ? s.face_rate
      : ((s.elapsed > 0.5 && s.faces) ? (s.faces / s.elapsed) : 0);
    EX_T = {
      pct: s.pct || 0, fps: s.fps || 0, faces: s.faces || 0,
      done: s.done || 0, total: s.total || 0,
      faceRate: faceRate, running: !!s.running
    };
    if (typeof s.fps === "number") {
      var lastP = EX_HIST.length ? EX_HIST[EX_HIST.length - 1].pct : -1;
      if (s.pct > lastP || EX_HIST.length === 0) {
        EX_HIST.push({ pct: s.pct || 0, fps: s.fps || 0 });
        if (EX_HIST.length > 600) EX_HIST.shift();
      }
    }
    var st = document.getElementById("exStage");
    if (st) st.textContent = s.running ? (STAGE_NAME[s.stage] || "") : "";
    setNavRunning("extract", !!s.running, s.pct || 0);
    exKick();
  }

  var EX_DST = "";   /* 本次提取的真实输出目录（后端解析后的绝对路径） */

  function subscribe() {
    if (es) es.close();
    es = new EventSource("/api/extract/progress");
    es.onmessage = function (ev) {
      var s = JSON.parse(ev.data);
      setExtractUI(s);
      /* 失败优先于「完成」：ok===false 或 error 非空就是失败 —— 红条，别报完成 */
      if (s.ok === false || s.error) {
        es.close(); es = null;
        toast("提取失败", String(s.error || s.msg || "").replace(/^提取失败：/, ""), "error", 8000);
        if (btnEx) { btnEx.disabled = false; btnEx.textContent = "开始提取"; }
        var sbF = document.getElementById("btnExStop");
        if (sbF) {
          sbF.dataset.mode = "back";
          sbF.classList.remove("danger", "hide");
          sbF.disabled = false;
          sbF.textContent = "返回";
        }
        setTimeout(function () {
          var pF = $("exProgress");
          if (pF) pF.classList.remove("hide");
        }, 0);
        return;
      }
      if (!s.running && s.pct >= 100) {
        es.close(); es = null;
        toast("提取完成",
              (s.faces ? (s.faces + " 张人脸") : ((s.total || 0) + " 个文件")) +
              (EX_DST ? " → " + EX_DST : ""));



  if (btnEx) { btnEx.disabled = false; btnEx.textContent = "开始提取"; }
        setTimeout(function () {
          var p = $("exProgress");
          if (p) p.classList.remove("hide");
        }, 400);
      }
    };
    es.onerror = function () { if (es) { es.close(); es = null; } };
  }

  /* 提取运行页：中断提取（只有一个按钮，放在进度下面） */
  (function () {
    var stop = document.getElementById("btnExStop");
    if (!stop || stop.dataset.bound) return;
    stop.dataset.bound = "1";
    stop.addEventListener("click", function () {
      if (stop.dataset.mode === "back") {          /* 任务已结束：直接回参数页 */
        var p = document.getElementById("exProgress");
        if (p) p.classList.add("hide");
        var b0 = document.getElementById("btnExtract");
        if (b0) { b0.disabled = false; b0.textContent = "开始提取"; }
        go("extract");
        return;
      }
      stop.disabled = true;
      stop.textContent = "正在中断…";
      fetch("/api/extract/cancel", { method: "POST", headers: { "Content-Type": "application/json" }, body: "{}" })
        .catch(function () {})
        .finally(function () {
          stop.disabled = false;
          stop.textContent = "返回";
          stop.dataset.mode = "back";
          stop.classList.remove("danger");
          var b = document.getElementById("btnExtract");
          if (b) { b.disabled = false; b.textContent = "开始提取"; }
        });
    });
  })();

  /* 速度测试页 */
  


  if (btnEx) {
    btnEx.addEventListener("click", function () {
      if (btnEx.disabled) return;
      btnEx.disabled = true;
      btnEx.textContent = "正在提取…";
      showProgress(true);
      /* 上一轮结束会留下「提取完成」，新任务要复位 */
      var hd = document.querySelector("#exProgress .xferhd b");
      if (hd) hd.textContent = "正在提取人脸";
      EX_T = { pct: 0, fps: 0, faces: 0, done: 0, total: 0, faceRate: 0, running: true };
      EX_D = { pct: 0, fps: 0, faces: 0, done: 0, faceRate: 0 };
      EX_HIST.length = 0;
      exKick();
      var sb = document.getElementById("btnExStop");
      if (sb) {
        sb.dataset.mode = "stop";
        sb.classList.remove("hide");
        sb.classList.add("danger");
        sb.disabled = false;
        sb.textContent = "中断提取";
      }
      go("extract-run");
      /* 用 id 取，别依赖 .pathval 的先后顺序 */
      var srcEl = document.getElementById("exSrc");
      var dstEl = document.getElementById("exDst");
      var src = srcEl ? String(srcEl.value || "").trim() : "";
      var dst = dstEl ? String(dstEl.value || "").trim() : "";
      /* 帧跳跃 N 表示每 N 帧取一张 —— 源视频 24 fps 时，fps 参数就是 24/N */
      var skipInp = document.getElementById("exSkip");
      var skip = skipInp ? Math.max(1, parseInt(skipInp.value, 10) || 1) : 1;
      /* 不再假设源是 24fps：把"每 N 帧取 1"直接交给后端，由它按源视频真实帧率换算 */

      /* 质量档 / 检测角度 / 帧跳跃：三个都是真参数，一起发给后端 */
      var qSeg = document.getElementById("exDetSeg");
      var qOn = qSeg ? qSeg.querySelector("b.on") : null;
      /* 检测器：RetinaFace(det_10g) 准，TinyMog 快。质量档不再影响分辨率。 */
      var quality = qOn && /TinyMog/i.test(qOn.textContent) ? "fast" : "high";

      var aSeg = document.getElementById("exAngleSeg");
      var aOn = aSeg ? aSeg.querySelector("b.on") : null;
      var rotate = aOn ? (parseInt(aOn.textContent, 10) || 0) : 0;

      if (!src) {
        btnEx.disabled = false;
        btnEx.textContent = "开始提取";
        showProgress(false);
        toast("缺少输入", "先在「输入」里填一个视频文件，或把文件拖进框里", "error");
        return;
      }

      fetch("/api/extract/start", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          src: src, dst: dst, skip: skip,
          rotate: rotate, quality: quality, align: true
        })
      }).then(function (r) {
        if (!r.ok) return r.json().then(function (d) { throw new Error(d.detail || "启动失败"); });
        return r.json();
      }).then(function (d) {
        EX_DST = d.dst || "";
        toast("开始抽帧", d.src + " → " + d.dst, "info");
        subscribe();
      }).catch(function (e) {
        btnEx.disabled = false;
        btnEx.textContent = "开始提取";
        showProgress(false);
        /* 启动失败时后端根本没有任务：那颗「中断提取」必须一起收掉。
           以前只复位了主按钮，页面上会留一个红按钮，点了没有任何反应。 */
        var sbx = document.getElementById("btnExStop");
        if (sbx) {
          sbx.dataset.mode = "stop";
          sbx.classList.add("hide");
          sbx.classList.remove("danger");
          sbx.textContent = "中断提取";
        }
        toast("启动失败", String((e && e.message) || e), "error");
      });
    });
  }

  var mOn = false;
  var btnMerge = $("btnmerge");
  if (btnMerge) {
    btnMerge.addEventListener("click", function () {
      mOn = !mOn;
      $("svcdot").classList.toggle("on", mOn);
      $("svctitle").textContent = mOn ? "合成服务运行中" : "合成服务未启动";
      $("svcurl").textContent = mOn
        ? "http://127.0.0.1:8001 · 已在浏览器打开"
        : "启动后在浏览器里打开工作台，模型在网页里再选";
      btnMerge.className = mOn ? "btn ghost lg" : "btn lg";
      btnMerge.style.cssText = "display:inline-flex;align-items:center;gap:8px;padding:14px 30px;font-size:14.5px";
      btnMerge.innerHTML = mOn
        ? '<svg class="ic" style="stroke:var(--ink-2);opacity:1"><use href="#i-power"/></svg>停止服务'
        : '<svg class="ic" style="stroke:var(--on-accent);opacity:1"><use href="#i-play"/></svg>启动合成器';
      toast(mOn ? "合成服务已启动" : "合成服务已停止", mOn ? "端口 8001" : "");
      if (window.I18N) window.I18N.refresh();
    });
  }

  window.winMin = function () { var a = api(); if (a) a.minimize(); };
  window.winMax = function () { var a = api(); if (a) a.maximize(); };
  window.winClose = function () { var a = api(); if (a) a.close(); };

  /* ---------------- 数据（全部来自 /api/state） ---------------- */
  var STATE = null;
  function el(id) { return document.getElementById(id); }
  function fmt(n) { try { return Number(n).toLocaleString("en-US"); } catch (e) { return String(n); } }
  var BS = "&#92;";

  function sparkSVG(spark) {
    var s = (spark && spark.src) || [];
    var d = (spark && spark.dst) || [];
    if (!s.length) return "";
    var all = s.concat(d);
    var min = Math.min.apply(null, all);
    var max = Math.max.apply(null, all);
    var span = (max - min) || 1;
    var W = 100, H = 40;
    function poly(arr) {
      var pts = [];
      for (var i = 0; i < arr.length; i++) {
        var x = (i / (arr.length - 1 || 1)) * W;
        var y = H - ((arr[i] - min) / span) * H;
        pts.push(x.toFixed(2) + "," + y.toFixed(2));
      }
      return pts.join(" ");
    }
    return '<svg viewBox="0 0 ' + W + " " + H + '" preserveAspectRatio="none">' +
      '<polyline fill="none" stroke="var(--ink-2)" stroke-width="1" vector-effect="non-scaling-stroke" points="' + poly(s) + '"/>' +
      '<polyline fill="none" stroke="var(--ok)" stroke-width="1" vector-effect="non-scaling-stroke" opacity=".8" points="' + poly(d) + '"/>' +
      "</svg>";
  }

  function modelCard(m, idx) {
    var el = document.createElement("div");
    el.className = "mcard2";
    /* 名字写在属性上：动态刷新时靠它认「这张卡对应哪个模型」 */
    el.setAttribute("data-model", m.name);
    el.style.animationDelay = Math.min(idx, 8) * 45 + "ms";

    var params = [];
    if (m.resolution) params.push(m.resolution);
    if (m.ae_dims) params.push("ae" + m.ae_dims);
    if (m.e_dims || m.d_dims) params.push("e" + m.e_dims + "/d" + m.d_dims);
    if (m.face_type) params.push(m.face_type);

    var kv = [];
    if (m.iter) kv.push('<span class="tag n">iter ' + fmt(m.iter) + "</span>");
    if (m.batch_size) kv.push('<span class="tag n">bs ' + m.batch_size + "</span>");
    if (m.bf16) kv.push('<span class="tag n">bf16</span>');
    if (m.optimizer) kv.push('<span class="tag n">' + m.optimizer + "</span>");
    if (m.weights === "npy") kv.push('<span class="tag warn">npy · 需转换</span>');
    /* DFM 徽标已按要求去掉 */

    /* 没有 loss 曲线就不画那块占位：空卡片比一行“暂无 loss 记录”干净 */
    var chart = (m.spark && m.spark.src && m.spark.src.length)
      ? '<div class="lchart">' + sparkSVG(m.spark) +
        '<div class="lg"><span><i style="background:var(--ink-2)"></i>src</span>' +
        '<span><i style="background:var(--ok)"></i>dst</span>' +
        '<span style="margin-left:auto">最近 ' + m.spark.src.length + " iter</span></div></div>"
      : '';

    /* 预览图不可点：以前是 <a target="_blank">，在 Electron 壳里会被
       setWindowOpenHandler 丢给 shell.openExternal —— 点一下模型缩略图就弹外部浏览器，
       非常莫名其妙。这里改成纯 <div>，只负责显示。想看大图去「参数」页或训练页。 */
    var prev = m.preview_url
      ? '<div class="mprev"><img loading="lazy" src="' + m.preview_url + '" alt=""></div>'
      : '';

    el.innerHTML = chart + prev +
      '<div class="mbody">' +
        '<div class="nm">' + m.name + ' <span class="tag n">' + (m.archi || (m.dfm_only ? "dfm" : "")) + "</span></div>" +
        (params.length ? '<div class="meta">' + params.join(" · ") + "</div>" : "") +
        (kv.length ? '<div class="kv">' + kv.join("") + "</div>" : "") +
        (m.dfm_only
          /* 只有 .dfm：没有训练目录，训练/参数/导出/删除都无从下手，整块不渲染 */
          ? ''
          : '<div class="acts"><button class="mini pri" data-act="run">训练</button>' +
            '<button class="mini" data-act="cfg">参数</button>' +
            '<button class="mini" data-act="export">导出</button>' +
            '<button class="mini" data-act="del">删除</button></div>') +
      "</div>";
    var runBtn = el.querySelector('[data-act="run"]');
    var _legacy = (m.weights === "npy");
    if (runBtn && _legacy) {
      /* 4.0 的老模型（.npy 权重）不能直接训：按钮变成「转换」，点了进转换子页 */
      runBtn.textContent = "转换";
      runBtn.addEventListener("click", function (e) {
        /* 卡片上还有别的委托监听（点了会跳训练界面），这里必须拦下来 */
        if (e) { e.preventDefault(); e.stopPropagation(); if (e.stopImmediatePropagation) e.stopImmediatePropagation(); }
        if (window.cvOpen) window.cvOpen(m.name);
        else { window.__cvPick = m.name; setTimeout(function () { go("train-convert"); }, 0); }
      });
    }
    if (runBtn && !_legacy) {
      /* 已经在训练的模型，按钮就是「回到训练界面」的入口 —— 不能再禁用，
         否则从监控界面退回列表之后就没有路回去了。 */
      if (m.state === "训练中" || m.state === "已暂停") runBtn.textContent = m.state;
    }
    if (runBtn) runBtn.addEventListener("click", function () {
      /* 「训练」就该真的开训：先把界面切到这个模型（曲线/预览），再拉进程。
         只有首页置顶那张卡才叫「继续训练」—— 它指的是最近训练过的那个模型。 */
      openRun(m);
      if (runBtn.__busy) return;
      if (m.state === "训练中" || m.state === "已暂停") return;  /* 已经在跑：只是回来看 */
      runBtn.__busy = true;
      runBtn.textContent = "启动中…";
      if (window.trainStartingUi) window.trainStartingUi();
      window.startTraining(m.name).then(function () {
        runBtn.textContent = "训练中";
        runBtn.__busy = false;
        if (window.trainStartingUi) window.trainStartingUi();
        if (window.pollTrain) window.pollTrain(m);
      }).catch(function (e) {
        var msg = String((e && e.message) || e);
        runBtn.__busy = false;
        /* 这个模型本来就在训练（从列表里重复点）——不是失败，接着看就行 */
        if (msg.indexOf("已经在训练") >= 0) {
          runBtn.textContent = "训练中";
          if (window.pollTrain) window.pollTrain(m);
          return;
        }
        runBtn.textContent = "训练";
        if (window.toast) window.toast("启动失败", msg.slice(0, 120), "error");
      });
    });

    var cfgBtn = el.querySelector('[data-act="cfg"]');
    if (cfgBtn) cfgBtn.addEventListener("click", function () {
      window.__cfgModel = m;
      openCfg(m);
      go("train-cfg");
    });

    var expBtn = el.querySelector('[data-act="export"]');
    if (expBtn) expBtn.addEventListener("click", function () {
      if (expBtn.disabled) return;
      expBtn.disabled = true;
      expBtn.textContent = "导出中…";
      toast("导出 DFM", m.name + " · 要加载模型，几分钟", "info");
      fetch("/api/models/" + encodeURIComponent(m.name) + "/export", { method: "POST" })
        .then(function (r) { return r.json().then(function (d) {
          if (!r.ok) throw new Error(d.detail || "启动失败");
          return d;
        }); })
        .then(function () {
          var t = setInterval(function () {
            fetch("/api/models/export/status").then(function (r) { return r.json(); }).then(function (d) {
              if (d.running) {
                var last = (d.log && d.log.length) ? d.log[d.log.length - 1] : "";
                expBtn.textContent = last ? String(last).slice(0, 6) : "导出中…";
                return;
              }
              clearInterval(t);
              expBtn.disabled = false;
              expBtn.textContent = "导出";
              if (d.ok) {
                toast("导出完成", m.name + ".dfm 已生成", "success");
                loadModels();
              } else {
                toast("导出失败", String(d.error || "").slice(0, 110), "error");
              }
            }).catch(function () {});
          }, 1500);
        })
        .catch(function (e) {
          expBtn.disabled = false;
          expBtn.textContent = "导出";
          toast("导出失败", String((e && e.message) || e).slice(0, 110), "error");
        });
    });

    var delBtn = el.querySelector('[data-act="del"]');
    if (delBtn) delBtn.addEventListener("click", function () {
      confirmBox("删除模型",
                "会移除 " + m.name + " 的全部权重与配置，无法恢复。",
                "删除",
                function () {
        fetch("/api/models/" + encodeURIComponent(m.name) + "/delete", { method: "POST" })
          .then(function (r) { return r.json().then(function (d) {
            if (!r.ok) throw new Error(d.detail || "删除失败");
            return d;
          }); })
          .then(function (d) {
            toast("已删除", m.name + " · 移除 " + (d.count || 0) + " 个文件", "success");
            loadModels();
          })
          .catch(function (e) {
            toast("删除失败", String((e && e.message) || e).slice(0, 110), "error");
          });
                }, true);
    });
    return el;
  }

  var MODELS = [];
  window.MODELS = MODELS;
  var MSORT = "time";

  var MQUERY = "";

  function bindTrainerTop() {
    var s = document.getElementById("msearch");
    if (s && !s.dataset.bound) {
      s.dataset.bound = "1";
      s.addEventListener("input", function () {
        MQUERY = String(s.value || "").trim().toLowerCase();
        renderTrainer(MODELS);
      });
    }
    var nb = document.getElementById("btnNewModel");
    if (nb && !nb.dataset.bound) {
      nb.dataset.bound = "1";
      nb.addEventListener("click", function () {
        if (window.openNewModel) { window.openNewModel(); return; }
        confirmBox("新建模型", "正在加载显卡信息…", "好", function () {});
      });
    }
  }
  window.bindTrainerTop = bindTrainerTop;

  function renderTrainer(list) {
    var box = $("mgrid");
    if (!box) return;
    MODELS = list || [];
    box.innerHTML = "";
    if (!MODELS.length) {
      box.innerHTML = '<div class="empty-note">workspace' + "\\" + 'model 里还没有模型</div>';
      return;
    }
    var sorted = MODELS.slice();
    if (MQUERY) sorted = sorted.filter(function (m) { return String(m.name).toLowerCase().indexOf(MQUERY) >= 0; });
    if (MSORT === "iter") sorted.sort(function (a, b) { return b.iter - a.iter; });
    else if (MSORT === "name") sorted.sort(function (a, b) { return a.name.localeCompare(b.name); });
    else sorted.sort(function (a, b) { return (b.trained_at || 0) - (a.trained_at || 0); });
    if (!sorted.length) {
      box.innerHTML = '<div class="empty-note">没有匹配「' + MQUERY + '」的模型</div>';
      return;
    }
    sorted.forEach(function (m, i) { box.appendChild(modelCard(m, i)); });
  /* 卡片是动态插入的，切到英文后再渲染会是中文 —— 渲染完统一过一遍词典 */
  try { if (window.I18N && window.I18N.get() === "en") window.I18N.refresh(); } catch (e) {}
    if (window.I18N) window.I18N.refresh();
    /* 这一遍就是最新的：把签名记下来，下一拍轮询不会因为「还没记过」再重建一次 */
    _mSig = modelsSig(MODELS);
  }
  /* 暴露给外部：新建 / 删除模型之后要立刻重画「模型选择」列表 */
  window.renderTrainer = renderTrainer;

  /* ------------------------------------------------------------------
     模型列表动态刷新（在「模型选择」页时每 3 秒看一次 /api/models）：
       · 某模型的 .dat 没了 → 这张卡先淡出（.mgone），320ms 后真的移出 DOM，
         再整体重排一次，剩下的卡片位置自然补上；
       · 数据没变化就不动 DOM —— 否则每 3 秒重建一遍，鼠标悬停、滚动全被打断。
     ------------------------------------------------------------------ */
  var _mSig = "";
  function modelsSig(list) {
    return (list || []).map(function (m) {
      return [m.name, m.iter, m.state, m.has_export ? 1 : 0].join(":");
    }).join("|");
  }
  function syncTrainer(list) {
    var box = $("mgrid");
    if (!box) return;
    var names = {};
    (list || []).forEach(function (m) { names[m.name] = 1; });
    var gone = [];
    Array.prototype.forEach.call(box.querySelectorAll("[data-model]"), function (el) {
      if (!names[el.getAttribute("data-model")]) gone.push(el);
    });
    if (!gone.length) { renderTrainer(list); return; }
    gone.forEach(function (el) { el.classList.add("mgone"); });
    setTimeout(function () {
      gone.forEach(function (el) { if (el.parentNode) el.parentNode.removeChild(el); });
      renderTrainer(list);
    }, 320);
  }
  function pollModels() {
    if (!document.getElementById("p-train")) return;
    if (!$("mgrid") || document.getElementById("p-train").classList.contains("hide")) return;
    fetch("/api/models").then(function (r) { return r.json(); }).then(function (d) {
      var list = (d && d.models) || [];
      var sig = modelsSig(list);
      if (sig === _mSig) return;          /* 没变化：一个节点都不碰 */
      _mSig = sig;
      syncTrainer(list);
    }).catch(function () {});
  }
  window.__pollModels = pollModels;
  setInterval(pollModels, 3000);
