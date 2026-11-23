  /* ---------- 合成工作室：真实视频预览（后端 /api/merge/*） ---------- */
  /* 所有状态都挂在 window 上：之前用模块内 var，出现过「传 300 却按 0 取帧」的诡异串台 */
  window.__mv = { path: "", video: "", t: 0, frames: 0, fps: 30, opening: false, log: [] };

  /* ---------- 时间轴宿主入口：重进页面 keep / 换素材重建 ---------- */
  /* 判断依据是 key = (素材路径|总帧数)：
     相同 → 还是同一个素材，只是又切回合成页，keep：保住播放头和人脸轨；
     不同 → 换了素材（或帧数变了），整条重建：播放头归零、旧人脸轨清掉。
     没素材时 total 传 0 —— 以前用 300 兜底，屏幕上那 300 帧是假数据。 */
  window.__mtlKey = window.__mtlKey || "";
  function mgTimelineSync(total, fps) {
    if (!window.MergeTimeline || !window.MergeTimeline.init) return null;
    var mv = window.__mv || {};
    total = Math.max(0, Math.floor(parseInt(total, 10) || 0));
    var useFps = (fps && fps > 0) ? fps : (mv.fps || 30);
    var key = String(mv.path || "") + "|" + total;
    var keep = total > 0 && key === window.__mtlKey;
    var out = null;
    try {
      out = window.MergeTimeline.init(total, useFps, { keep: keep });
    } catch (e) { console.warn("timeline init", e); }
    window.__mtlKey = key;
    return out;
  }

  window.mvStage = function (dataUrl) {
    var st = document.getElementById("mgStage");
    if (!st) return false;
    /* 只认老路径自己插的帧图：否则 #mgPreview（合成 overlay）会被当成它，把帧图画进合成层 */
    var img = st.querySelector("img.mgfrm");
    if (!img) {
      img = document.createElement("img");
      img.className = "mgfrm";
      img.alt = "";
      st.appendChild(img);
    }
    img.style.display = "block";
    img.onload = function () {
      /* 「框」开着就补画一次：图是异步换的，画早了会按旧尺寸画在空气里 */
      if (window.mgBoxesOn && window.mgBoxesOn()) window.mvRedrawBoxes();
    };
    img.src = dataUrl;
    var hint = document.getElementById("mgStageHint");
    if (hint) hint.style.display = "none";
    return true;
  };

  /* 预览取图的序号保护：快速拖时间轴 / 连续播放时会有多个请求同时在飞，
     900px 的 JPEG 谁后到谁覆盖 —— 手指停在 240 帧，画面可能停在 180 帧。
     这里给每个视图一个自增序号，回来时序号不是最新的就直接丢弃。 */
  var MV_SEQ = { orig: 0, swap: 0, det: 0 };

  /* 成片预览：契约要求 /api/merge/merged 也接受合成参数，
     所以先按 POST + {frame, settings, models} 发；后端还是老版本（404/405/501）
     再退回老的 GET /api/merge/merged/<n>?w=900。 */
  window.mvMerged = function (n) {
    n = parseInt(n, 10) || 0;
    var seq = ++MV_SEQ.swap;
    var body = {
      /* 超分开着时要原尺寸：SR 在父进程做，先缩到 900 再超分等于白做（脸只有 ~170px） */
      frame: n, idx: n, n: n, w: (window.mgSrLevel && window.mgSrLevel() > 0 ? 1920 : 900),
      /* 和合成请求同一个口径：有「不换脸」的脸才要无损，否则 jpg（快 3 倍） */
      no_swap_faces: (window.mgFaceNoSwapIds ? window.mgFaceNoSwapIds() : []),
      fmt: ((window.mgFaceNoSwapIds && window.mgFaceNoSwapIds().length) ? "png" : "jpg"),
      model: (window.mgFaceSingleModel ? window.mgFaceSingleModel() : ""),
      models: (window.mgFaceModelsPayload ? window.mgFaceModelsPayload() : []),
      /* 超分滑块：顶层也发一份（后端两个口径都认；settings 里那份是合成参数口径） */
      sr: (window.mgSrLevel ? window.mgSrLevel() : 0),
      settings: (window.mgCmpSettings ? window.mgCmpSettings() : {})
    };
    function legacy() {
      return fetch("/api/merge/merged/" + n + "?w=900").then(function (r) {
        return r.json().then(function (d) {
          if (!r.ok) throw new Error(d.detail || "取不到成片");
          return d;
        });
      });
    }
    fetch("/api/merge/merged", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body)
    }).then(function (r) {
      if (r.status === 404 || r.status === 405 || r.status === 501) return legacy();
      return r.json().then(function (d) {
        if (!r.ok) throw new Error((d && d.detail) || ("HTTP " + r.status));
        return d;
      });
    }).then(function (d) {
      if (seq !== MV_SEQ.swap) return;      /* 过期响应，丢掉 */
      if (d.png) window.mvStage(d.png);
    }).catch(function (e) {
      if (seq !== MV_SEQ.swap) return;
      toast("成片预览", String((e && e.message) || e).slice(0, 90), "info");
      var hint = document.getElementById("mgStageHint");
      if (hint) { hint.style.display = ""; hint.textContent = "还没有成片 —— 先导出一次"; }
    });
  };

  window.mvFrame = function (n) {
    var st = window.__mv;
    n = parseInt(n, 10) || 0;
    /* 视频模式由 <video> 自己出画面，不再逐帧取 JPEG（帧目录 / 视频流挂了时才走这里） */
    if (mgVidPath() && !(window.__mgVideo || {}).streamDown) return Promise.resolve();
    if (!st.path) { st.log.push("no-path " + n); return Promise.resolve(); }
    var seq = ++MV_SEQ.orig;
    /* 这一层是「底图」：合成结果走 #mgPreview 那层 overlay，底图永远是原帧。
       以前这里不带 raw=1，拿回来的是**合成图**，于是「原图」一态下面垫着的
       其实是换过脸的图 —— 原图视图会露出换脸结果。 */
    return fetch("/api/merge/frame/" + n + "?w=900&raw=1&fmt=png").then(function (r) { return r.json(); })
      .then(function (d) {
        st.log.push("req " + n + " -> " + (d.png ? d.png.length : "none") + " (idx " + d.idx + ")");
        if (seq !== MV_SEQ.orig) return;    /* 过期响应，丢掉 */
        if (d.png) window.mvStage(d.png);
      })
      .catch(function (err) { st.log.push("err " + n + " " + err.message); });
  };

  window.mvOpen = function (path, name) {
    var st = window.__mv;
    if (st.opening) return;
    st.opening = true;
    toast("打开视频", (name || "") + " · 正在解析", "info");
    fetch("/api/merge/open", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ path: path })
    }).then(function (r) { return r.json(); }).then(function (d) {
      st.opening = false;
      if (d.frames == null) { toast("打开失败", d.detail || "未知错误", "error"); return; }
      st.path = d.path; st.frames = d.frames; st.fps = d.fps || 30;
      /* 有源视频就走 <video> 播放器；纯帧目录（DFL 项目）继续走老的逐帧取图 */
      var nextVideo = (d.mode === "project") ? "" : (d.video || path);
      /* 换了素材就不要再续上一个视频的播放位置；同一个文件重开才用 window.__mv.t */
      if (nextVideo !== (window.__mgVideo || {}).src) st.t = 0;
      st.video = nextVideo;
      if (window.mgStageEmpty) window.mgStageEmpty(false);
      mgTimelineSync(d.frames, d.fps || 30);
      if (window.mgClearFaceAssign) window.mgClearFaceAssign();   /* 先清脸分配，再取帧 */
      if (st.video) {
        window.mgAttachVideo(st.video);
      } else {
        window.mgDetachVideo();
        window.mvFrame(0);
        if (window.mgCmpNow) window.mgCmpNow({ frame: 0 });
      }
      window.mvAnalyze(24);
      var tot = document.getElementById("mtlTotal");
      if (tot) tot.textContent = d.frames;
      if (d.mode === "project") {
        toast("已打开项目", (name || d.name) + " · " + d.frames + " 张 · " +
              (d.with_landmarks || 0) + " 张带 landmark", "success");
      } else {
        toast("已打开", (name || d.name) + " · " + d.frames + " 帧 · " + Math.round(d.fps) + " fps · " + d.width + "×" + d.height, "success");
      }
      var vl = document.getElementById("mgVideos");
      if (vl) {
        Array.prototype.forEach.call(vl.children, function (x) {
          x.classList.toggle("on", x.dataset.path === path);
        });
      }
      /* 视频已打开：用户接下来一定会拖时间轴/暂停看效果，先把模型热起来（多选时逐个） */
      window.mgWarmSelected();
    }).catch(function (err) {
      st.opening = false;
      toast("打开失败", String((err && err.message) || err), "error");
    });
  };

  /* 打开一个 DFL 项目目录（data_dst / aligned / 裸帧目录都认） */
  window.mvOpenProject = async function (path) {
    try {
      var p = path;
      if (!p) {
        if (!(window.desktop && window.desktop.pickDir)) {
          toast("打开项目", "网页版拿不到本地路径", "info");
          return;
        }
        p = await window.desktop.pickDir("选择 DFL 项目目录（data_dst）");
      }
      if (!p) return;

      var r1 = await fetch("/api/merge/project", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ path: p })
      });
      var info = await r1.json();
      if (!r1.ok) throw new Error(info.detail || "读不了这个目录");

      var r2 = await fetch("/api/merge/open", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ path: p })
      });
      var d2 = await r2.json();
      if (!r2.ok) throw new Error(d2.detail || "打开失败");

      window.__mergeInputDir = p;
      if (window.mgDflModeSync) setTimeout(window.mgDflModeSync, 0);   /* 素材换了：刷「沿用 DFL」可用性 */
      /* 「打开项目」这条路从来没写过 window.__mv：预览取图一直是 no-path，
         时间轴也拿不到这次的帧数。顺手补上（keep/重建的 key 也靠它才算得对）。 */
      var mv = window.__mv;
      mv.path = d2.path || p;
      /* 后端若连源视频一起给了就用播放器；只有帧目录就退回逐帧取图 */
      mv.video = String(d2.video || "");
      if (mv.video) window.mgAttachVideo(mv.video);
      else if (window.mgDetachVideo) window.mgDetachVideo();
      mv.frames = d2.frames || 0;
      mv.fps = (d2.fps && d2.fps > 0) ? d2.fps : 30;
      mgTimelineSync(mv.frames, mv.fps);
      window.mvFrame(0);
      var tot = document.getElementById("mtlTotal");
      if (tot) tot.textContent = d2.frames;

      toast("已打开项目",
            String(p).split(/[\\/]/).pop() + " · " + d2.frames + " 张 · " +
            (d2.with_landmarks || 0) + " 张带 landmark",
            "success");
    } catch (e) {
      toast("打开失败", String((e && e.message) || e).slice(0, 110), "error");
      console.error("[mvOpenProject]", e);
    }
  };

  /* ---------- 素材准备：视频 → 抽帧 + 对齐 → 可合成项目 ---------- */
  /* 4.1 的提取器只落对齐脸、不落原帧，而合成必须要有原帧才能把脸贴回去。
     后端 /api/merge/prepare 补了这条路，这里负责把它接进界面。 */
  window.__mergePrep = { running: false, project: "", timer: null, fails: 0 };

  function mgPrepLine(text, pct) {
    var box = document.getElementById("mgPrep");
    if (!box) return;
    if (!text) { box.style.display = "none"; return; }
    box.style.display = "";
    var lb = box.querySelector(".lb");
    var bar = box.querySelector(".bar i");
    if (lb) lb.textContent = text;
    if (bar && typeof pct === "number") {
      bar.style.width = Math.max(2, Math.min(100, pct)) + "%";
    }
  }

  /* 素材准备完成后，把项目挂到左栏素材列表上（否则一直显示"还没有导入素材"） */
  function mgShowPrepared(project, frames) {
    var box = document.getElementById("mgVideos");
    var empty = document.getElementById("mgVideosEmpty");
    if (!box || !project) return;
    var name = String(project).split(/[\\/]/).pop();
    var row = null;
    Array.prototype.forEach.call(box.children, function (x) {
      if (x.dataset.path === project) row = x;
    });
    if (!row) {
      row = document.createElement("div");
      row.className = "mgitem";
      row.dataset.path = project;
      row.innerHTML = '<span class="nm">' + name + '</span><span class="tag">' + (frames || 0) + ' 帧</span>';
      row.addEventListener("click", function () {
      if (window.mgDflModeSync) setTimeout(window.mgDflModeSync, 0);   /* 素材换了：刷「沿用 DFL」可用性 */
        window.__mergeInputDir = project;
        window.mgOpenPrepared(project, frames, 0);
      });
      box.appendChild(row);
    }
    Array.prototype.forEach.call(box.children, function (x) {
      x.classList.toggle("on", x === row);
    });
    if (empty) empty.style.display = "none";
    /* 素材就绪 = 舞台上马上会有画面：舞台空态收起 */
    if (window.mgStageEmpty) window.mgStageEmpty(false);
  }
  window.mgShowPrepared = mgShowPrepared;

  /* 点「已就绪项目」= 让服务端把素材真正切过去（POST /api/merge/open）。
     以前这里只 GET /api/merge/info 读缓存：缓存里还是上一个项目，所以画面和
     总帧数都不换。以 open 回来的 frames/fps 为准，失败必须说出来。 */
  window.mgOpenPrepared = function (project, frames, aligned) {
    var st = window.__mv;
    if (!project) {
      toast("打开素材", "这个项目没有路径，打不开", "error");
      return;
      if (window.mgDflModeSync) setTimeout(window.mgDflModeSync, 0);   /* 素材换了：刷「沿用 DFL」可用性 */
    }
    window.__mergeInputDir = project;
    fetch("/api/merge/open", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ path: project })
    }).then(function (r) {
      return r.json().then(function (d) {
        if (!r.ok || d.frames == null) throw new Error(d.detail || "打开失败");
        return d;
      });
    }).then(function (d) {
      st.path = d.path || project;
      /* 后端若连源视频一起给了就用播放器；只有帧目录就走逐帧取图 */
      st.video = String(d.video || "");
      if (st.video) window.mgAttachVideo(st.video);
      else if (window.mgDetachVideo) window.mgDetachVideo();
      st.frames = d.frames || 0;
      st.fps = (d.fps && d.fps > 0) ? d.fps : 30;
      mgTimelineSync(st.frames, st.fps);
      /* 重新点开同一个素材时 key 会撞上、走 keep，播放头可能停在上次的位置，
         这里明确归零，跟下面要取的 0 帧对齐 */
      if (window.MergeTimeline && window.MergeTimeline.setFrame) {
        try { window.MergeTimeline.setFrame(0); } catch (e) {}
      }
      var tot = document.getElementById("mtlTotal");
      if (tot) tot.textContent = st.frames;
      mgShowPrepared(project, st.frames);
      /* 脸分配属于「上一份素材」：先清掉再取帧/合成，否则第一发请求还带着旧分配 */
      if (window.mgClearFaceAssign) window.mgClearFaceAssign();
      /* 底图 = 原帧（mvFrame 走 raw=1），合成结果另走 overlay 那一层。
         以前只画底图，底图又是合成图，所以「原图」看起来像合成；现在两层分开，
         打开项目后必须主动合成一次，合成态才有换脸结果。 */
      window.mvFrame(0);
      /* 换素材 = 帧要变了：清框 / 检测 / 合成全走同一条链路（见 mgFrameChange）。
         「框」默认是开的，所以换完素材这一发里就带着检测 —— 一进来就能看见框；
         没人脸的那一帧则连合成都不发，直接回原图。 */
      window.mgFrameChange(0, { why: "open" });
      if (window.mvAnalyze) window.mvAnalyze(24);
      toast("素材已就绪",
            st.frames + " 帧 · " + (aligned || d.with_landmarks || 0) + " 张对齐脸",
            "success");
      /* 素材就绪 = 马上就要合成，趁机让后端把模型载入（多选时逐个，失败静默） */
      window.mgWarmSelected();
    }).catch(function (e) {
      /* 以前 catch 里空着还报「素材已就绪」—— 失败了不能装没事 */
      toast("打开素材失败", String((e && e.message) || e).slice(0, 110), "error");
    });
  };

  window.mgPrepare = function (video) {
    if (!video) return;
    var st = window.__mergePrep;
    if (st.running) { toast("正在准备", "等这一段处理完再来", "info"); return; }
    st.running = true;
    st.fails = 0;
    /* 这里不再清 st.project / window.__mergeInputDir：上一次已就绪的项目要留着，
       准备失败时用户还得回到它。只有下面成功分支才覆盖。 */
    mgPrepLine("复制视频…", 0);
    /* 视频名可能是 encodeURIComponent 过的（拖入/URL 带过来的），先解一次再显示，
       否则 toast 里是一串 %E4%B9%9D… */
    var _nm = String(video).split(/[\\/]/).pop();
    try { _nm = decodeURIComponent(_nm); } catch (e) {}
    toast("导入视频", _nm + " · 复制到工作区", "info");

    fetch("/api/merge/prepare", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ video: video })
    }).then(function (r) {
      return r.json().then(function (d) {
        if (!r.ok) throw new Error(d.detail || "启动失败");
        return d;
      });
    }).then(function () {
      if (st.timer) clearInterval(st.timer);
      st.timer = setInterval(function () {
        fetch("/api/merge/prepare/status").then(function (r) { return r.json(); })
          .then(function (d) {
            st.fails = 0;
            mgPrepLine((d.stage ? d.stage + " · " : "") + (d.msg || ""), d.pct);
            if (d.running) return;
            clearInterval(st.timer); st.timer = null; st.running = false;
            if (d.ok) {
              st.project = d.project || "";
              window.__mergeInputDir = st.project;
              mgPrepLine("", 0);
              window.mgOpenPrepared(st.project, d.frames, d.aligned);
            } else {
              mgPrepLine("准备失败：" + String(d.error || "").slice(0, 80), 100);
              toast("准备失败", String(d.error || "未知错误").slice(0, 120), "error");
            }
          }).catch(function () {
            /* 状态接口一直连不上时别无限轮询：st.running 会永久卡在 true，
               之后连准备素材的按钮都点不动。40 次（≈28 秒）后收摊报错。 */
            st.fails = (st.fails || 0) + 1;
            if (st.fails <= 40) return;
            clearInterval(st.timer); st.timer = null; st.running = false; st.fails = 0;
            mgPrepLine("准备失败：状态轮询中断", 100);
            toast("准备失败", "状态接口连续 40 次没有响应，已停止等待", "error");
          });
      }, 700);
    }).catch(function (e) {
      st.running = false;
      mgPrepLine("准备失败：" + String((e && e.message) || e).slice(0, 80), 100);
      toast("准备失败", String((e && e.message) || e).slice(0, 120), "error");
    });
  };

  window.mvList = function (dir) {
    var box = document.getElementById("mgVideos");
    if (!box) return;
    var url = "/api/merge/videos" + (dir ? "?dir=" + encodeURIComponent(dir) : "");
    fetch(url).then(function (r) { return r.json(); }).then(function (d) {
      var list = (d.videos || []);
      box.innerHTML = "";
      /* 列表是「工作区里的视频」，不含已经抽好帧的项目 —— 重新拉一次别把当前项目弄丢 */
      if (window.__mergeInputDir) {
        try { mgShowPrepared(window.__mergeInputDir, (window.__mv || {}).frames || 0); } catch (e) {}
      }
      var empty = document.getElementById("mgVideosEmpty");
      if (!list.length) {
        /* 空状态看的是 box 里到底有没有行：上面已经按 __mergeInputDir 把「已就绪项目」
           塞进去了，只按接口返回的 list 判，会把「还没有导入素材」压在那行下面。 */
        if (empty) empty.style.display = box.children.length ? "none" : "";
        return;
      }
      /* 下面要追加视频行：先用老规矩把空状态收掉，别让它杵在列表头上 */
      if (empty) empty.style.display = "none";
      list.slice(0, 60).forEach(function (v) {
        var row = document.createElement("div");
        row.className = "mgitem";
        row.dataset.path = v.path;
        row.innerHTML = '<span class="nm">' + v.name + '</span>';
        row.addEventListener("click", function () { window.window.mvOpen(v.path, v.name); });
        box.appendChild(row);
      });
    }).catch(function () {});
  };


  /* ---------- 检测视图：RetinaFace 出框，画在 canvas 叠加层上 ---------- */
  window.mvOverlay = function () {
    var st = document.getElementById("mgStage");
    if (!st) return null;
    var cv = st.querySelector("canvas.mgovl");
    if (!cv) {
      cv = document.createElement("canvas");
      cv.className = "mgovl";
      st.appendChild(cv);
    }
    return cv;
  };

  /* ---------- 帧时序：t0（帧要变）/ t_detect / t_draw / t_frame ----------
     只给验收和排查读（window.mgTiming()）。全部取 performance.now()，同页可直接相减。 */
  var MG_FRAME_T = { frame: -1, why: "", t0: 0, tDetect: 0, tDraw: 0, tFrame: 0,
                     boxes: 0, chips: 0, noFace: false };

  function mgNowT() {
    return (window.performance && window.performance.now) ? window.performance.now() : Date.now();
  }

  /* 检测请求的两把保险：
       seq —— 谁后发谁说了算（老回包一律丢掉，绝不把上一帧的框画到新画面上）
       gen —— 「清过框」的代数：同一帧清框之后要重新检测，gen 也用来做同帧去重 */
  var MG_DET = { seq: 0, gen: 0, lastFrame: -1, lastGen: -1, last: null,
                 inflightFrame: -1, inflightGen: -1, inflight: null };

  window.mgTiming = function () {
    var t = MG_FRAME_T;
    return { frame: t.frame, why: t.why, t0: t.t0, t_detect: t.tDetect, t_draw: t.tDraw,
             t_frame: t.tFrame, boxes: t.boxes, chips: t.chips, no_face: !!t.noFace,
             detect_ms: t.tDetect ? Math.round((t.tDetect - t.t0) * 10) / 10 : null,
             draw_ms: (t.tDetect && t.tDraw) ? Math.round((t.tDraw - t.tDetect) * 10) / 10 : null,
             frame_after_draw_ms: (t.tDraw && t.tFrame) ? Math.round((t.tFrame - t.tDraw) * 10) / 10 : null };
  };
  window.mgTimingMark = function (key) {
    if (key === "t_frame") MG_FRAME_T.tFrame = mgNowT();
    return MG_FRAME_T[key];
  };

  /* 「帧要变了」的唯一入口：同步清掉 canvas 描边 + chip 层，并记下 t0。
     播放开始 / seek / 换帧 / 换素材都必须先走这里 —— 宁可暂时没有框，
     也不许把一个上一帧的框留在新画面上。 */
  window.mgInvalidateBoxes = function (why, frame) {
    if (frame != null) MG_FRAME_T.frame = Math.max(0, parseInt(frame, 10) || 0);
    MG_FRAME_T.why = String(why || "");
    MG_FRAME_T.t0 = mgNowT();
    MG_FRAME_T.tDetect = 0;
    MG_FRAME_T.tDraw = 0;
    MG_FRAME_T.tFrame = 0;
    MG_FRAME_T.boxes = 0;
    MG_FRAME_T.chips = 0;
    MG_FRAME_T.noFace = false;
    window.mvClearBoxes();
    return MG_FRAME_T;
  };

  window.mvClearBoxes = function () {
    window.__mvBoxes = null;
    /* 清框 = 这一代作废：在飞的检测回来也不许画，下一次检测重新发 */
    MG_DET.seq++;
    MG_DET.gen++;
    MG_DET.lastFrame = -1;
    MG_DET.last = null;
    MG_DET.inflightFrame = -1;
    MG_DET.inflightGen = -1;
    MG_DET.inflight = null;
    /* 框、chip、＋ 三样是同一份数据的三种画法：清框必须连 DOM 层一起清，
       否则会留下一排指向上一个素材的 chip。 */
    mgResetFaces();
    MG_FRAME_T.chips = 0;
    var cv = document.querySelector("#mgStage canvas.mgovl");
    if (!cv) return;
    var ctx = cv.getContext("2d");
    ctx.clearRect(0, 0, cv.width, cv.height);
  };

  /* 图片是异步换的：换完再画一次，否则 canvas 会按旧尺寸画在空气里 */
  window.mvRedrawBoxes = function (tries) {
    var b = window.__mvBoxes;
    if (!b) return;
    tries = tries == null ? 8 : tries;
    window.mvDrawBoxes(b.boxes, b.w, b.h);
    if (window.__mvPending && tries > 0) {
      /* 版面还没就绪（刚切页/刚换图），下一帧再试，最多 8 次 */
      requestAnimationFrame(function () { window.mvRedrawBoxes(tries - 1); });
    }
  };

  /* 画框：只描边、不填充。
     以前框里压一层 rgba(128,128,140,.14)，脸被盖住看不见换没换成功；现在画布上
     除了 2px 描边一个像素都不落（验收里按像素读：框内 = 全透明）。
     颜色 = 这张脸分到的 dfm 的专属色；没分配 = 灰色。 */
  /* 框外侧那 2px 深色对比线：浅色描边压在亮画面上时靠它立住（实测把「近白帧」上的
     可辨识对比从 1.0:1 拉到 4~6:1），暗画面上它和底色一样深，等于没画。 */
  var MG_BOX_HALO = "rgba(0,0,0,.55)";

  window.mvDrawBoxes = function (boxes, ow, oh) {
    var cv = window.mvOverlay();
    if (!cv || !ow || !oh) return;
    window.__mvBoxes = { boxes: boxes, w: ow, h: oh };
    var st = document.getElementById("mgStage");
    /* canvas 和 img 在 CSS 里是同一块内缩盒子（inset:6px），所以直接按舞台量；
       量到 0 说明页面还没排好版（隐藏/首帧），把框留着，等 img.onload 再画一次。 */
    var r = st.getBoundingClientRect();
    if (r.width < 20 || r.height < 20) {
      window.__mvPending = true;
      return;
    }
    window.__mvPending = false;
    cv.width = Math.round(r.width) - 12;
    cv.height = Math.round(r.height) - 12;
    var ctx = cv.getContext("2d");
    ctx.clearRect(0, 0, cv.width, cv.height);

    /* 和 img 的 object-fit: contain 用同一套算法，框才不会错位 */
    var s = Math.min(cv.width / ow, cv.height / oh);
    var ox = (cv.width - ow * s) / 2;
    var oy = (cv.height - oh * s) / 2;

    var rects = [], raw = [];
    ctx.lineWidth = 2;                     /* 契约：2px */
    (boxes || []).forEach(function (b, i) {
      var x1 = b.box[0], y1 = b.box[1], x2 = b.box[2], y2 = b.box[3];
      /* +0.5 让 2px 线正好压在像素格上，缩放后也是实心两象素、不发虚 */
      var rx = Math.round(ox + x1 * s) + 0.5, ry = Math.round(oy + y1 * s) + 0.5;
      var rw = Math.round((x2 - x1) * s), rh = Math.round((y2 - y1) * s);
      var name = mgFaceChipModel(i);
      /* 被「点成原图」的那张脸：框线走虚线（其它框还是实线）——点一下就能看出
         「这张脸现在没换」。虚线只在框线上，框内一样一个像素都不落。 */
      var rawView = (typeof mgFaceIsRawView === "function") && mgFaceIsRawView(i);
      ctx.setLineDash(rawView ? [6, 4] : []);
      ctx.strokeStyle = MG_BOX_HALO;
      ctx.strokeRect(rx - 2, ry - 2, rw + 4, rh + 4);
      ctx.strokeStyle = name ? mgModelColor(name) : mgUnassignedColor();
      ctx.strokeRect(rx, ry, rw, rh);
      ctx.setLineDash([]);
      rects[i] = { x: rx - 0.5, y: ry - 0.5, w: rw, h: rh };
      /* kps68 是检测回包带的 68 点：点「＋」时原样交给 /faces/add 落库 */
      raw[i] = { box: [x1, y1, x2, y2], score: b.score, kps68: b.kps68 || null };
    });
    mgSetRects(rects, raw);
  };

  /* 窗口尺寸变了：canvas 的位图尺寸只在 mvDrawBoxes 里算过一次，之后全靠 CSS 拉伸，
     拉伸完框就错位。防抖 120ms，等布局稳下来再按新尺寸重画一次。 */
  var MV_RESIZE_T = null;
  window.addEventListener("resize", function () {
    if (MV_RESIZE_T) clearTimeout(MV_RESIZE_T);
    MV_RESIZE_T = setTimeout(function () {
      MV_RESIZE_T = null;
      var b = window.__mvBoxes;
      if (!b) return;
      var page = document.getElementById("p-merge");
      if (!page || !page.classList || !page.classList.contains("on")) return;
      if (!window.mgBoxesOn || !window.mgBoxesOn()) return;
      if (!document.querySelector("#mgStage canvas.mgovl")) return;
      window.mvDrawBoxes(b.boxes, b.w, b.h);
    }, 120);
  });

  window.mvAnalyze = function (step) {
    var mv = window.__mv || {};
    if (!mv.path) { toast("先打开视频", "分析需要先选一个视频", "info"); return; }
    step = step || 12;
    toast("开始分析", "每隔 " + step + " 帧检测一次，稍等…", "info");
    var t0 = Date.now();
    fetch("/api/merge/analyze?step=" + step + "&model=" + encodeURIComponent(mgDetectorName()))
      .then(function (r) { return r.json(); })
      .then(function (d) {
        if (!d.marks) { toast("分析失败", d.detail || "未知错误", "error"); return; }
        var faces = [];
        d.marks.forEach(function (m) {
          if (m.count > 0) {
            faces.push({ s: m.idx, e: Math.min(d.total, m.idx + d.step), n: m.count });
          }
        });
        if (window.MergeTimeline && window.MergeTimeline.setData) {
          window.MergeTimeline.setData({ faces: faces });
        }
        toast("分析完成",
          d.scanned + " 个采样点 · " + faces.length + " 处有脸 · 用时 " + ((Date.now() - t0) / 1000).toFixed(1) + " 秒",
          "success");
      })
      .catch(function (e) { toast("分析失败", String((e && e.message) || e), "error"); });
  };

  function mgDetectorName() {
    var sel = document.getElementById("mgDetector");
    /* 默认 TinyMog：500m 的权重不存在，以前那个兜底值会让后端去找一个没有的模型 */
    return (sel && sel.value) || "TinyMog";
  }
  window.mgDetectorName = mgDetectorName;

  /* 检测器进请求体：**要么不带这个键，要么带一个真值**。
     以前四条导出路径都写 `detector: payload.detector || ""`：导出设置页没这个元素时
     发出去的是空串，后端 _detector_val("") 把它当成「谁都没说」→ 落进默认档。
     界面选什么都白选，日志里也看不出「前端根本没发」。
     现在：payload 里有就用；没有就直接读 #mgDetector 的 value；连元素都没有 →
     **不发这个键**（返回 ""），并 console.warn 一句（这不是错误，是「认不出界面」）。 */
  function mgDetectorBody(payload) {
    var v = String((payload && payload.detector) || "").trim();
    if (v) return v;
    var sel = document.getElementById("mgDetector");
    v = String((sel && sel.value) || "").trim();
    if (v) return v;
    if (window.console && console.warn) {
      console.warn("[merge] #mgDetector 不在页面上（或没有值）：这次不传 detector，"
                   + "后端将用默认检测器（tinymog）");
    }
    return "";
  }
  window.mgDetectorBody = mgDetectorBody;

  /* 检测：**一回来就画框 + chip**，绝不和合成串在一起。
     返回 Promise<{ok, boxes, width, height, frame}>
        · 调用方（mgFrameChange）用它决定「有人脸才发合成」；
        · 没人脸时也 resolve（ok:true, boxes:[]），不是失败。
     同一帧 + 同一代只发一发：进页面、合成回包、暂停都会重复问同一帧，
     多打几发只会让后端白跑检测；清过框（gen 变了）才会重新问。 */
  window.mvDetect = function (n, opt) {
    var mv = window.__mv || {};
    n = Math.max(0, parseInt(n, 10) || 0);
    if (!mv.path) return Promise.resolve({ ok: false, boxes: [], why: "no-source", frame: n });
    opt = opt || {};
    var gen = MG_DET.gen;
    if (!opt.force) {
      if (MG_DET.inflight && MG_DET.inflightFrame === n && MG_DET.inflightGen === gen) {
        return MG_DET.inflight;                       /* 同一帧正在飞：复用那一发 */
      }
      if (MG_DET.last && MG_DET.lastFrame === n && MG_DET.lastGen === gen) {
        return Promise.resolve(MG_DET.last);          /* 这一代已经检测过：直接给结果 */
      }
    }
    var seq = ++MG_DET.seq;             /* 拖时间轴时检测请求会连发，回来晚的直接丢 */
    var t0 = mgNowT();
    var p = fetch("/api/merge/detect/" + n + "?model=" + encodeURIComponent(mgDetectorName()) + "&thresh=0.5")
      .then(function (r) { return r.json(); })
      .then(function (d) {
        if (seq !== MG_DET.seq) return { ok: false, boxes: [], why: "stale", frame: n };
        MG_DET.inflight = null;
        MG_DET.inflightFrame = -1;
        /* 回来得太晚（播放中一帧 33ms，检测 180ms）→ 这一帧已经不是当前帧了。
           绝不把它的框画到新画面上：宁可这一拍没有框。 */
        var cur = -1;
        try { cur = mgCmpFrameIdx(); } catch (e) { cur = -1; }
        if (cur >= 0 && cur !== n) {
          return { ok: false, boxes: [], why: "outdated", frame: n, cur: cur };
        }
        var boxes = (d && d.boxes) || [];
        var res = { ok: !!(d && d.boxes), frame: n, gen: gen, boxes: boxes,
                    width: (d && d.width) || 0, height: (d && d.height) || 0,
                    t_request: t0, t_detect: mgNowT() };
        MG_DET.last = res;
        MG_DET.lastFrame = n;
        MG_DET.lastGen = gen;
        MG_FRAME_T.frame = n;
        MG_FRAME_T.tDetect = res.t_detect;
        MG_FRAME_T.boxes = boxes.length;
        if (!boxes.length) {
          if (d && d.detail) toast("检测失败", String(d.detail).slice(0, 90), "error");
          return res;
        }
        /* 就是这里画框：回包到 → 描边落画布 → chip 层同一份数据重建。
           合成（dfm 推理 + 遮罩 + 编码，几百毫秒）此刻可能还在飞，
           画面先有框 + 上一张底图，合成回来只替换底图。 */
        window.mvDrawBoxes(boxes, res.width, res.height);
        if (window.__mvBoxes) window.__mvBoxes.frame = n;   /* 屏幕上的框属于哪一帧 */
        MG_FRAME_T.tDraw = mgNowT();
        var layer = document.getElementById("mgFaceLayer");
        MG_FRAME_T.chips = layer ? layer.children.length : 0;
        return res;
      })
      .catch(function (e) {
        if (seq !== MG_DET.seq) return { ok: false, boxes: [], why: "stale", frame: n };
        MG_DET.inflight = null;
        MG_DET.inflightFrame = -1;
        return { ok: false, boxes: [], why: "error", detail: String((e && e.message) || e), frame: n };
      });
    MG_DET.inflight = p;
    MG_DET.inflightFrame = n;
    MG_DET.inflightGen = gen;
    return p;
  };


  window.MergeVideo = { list: window.mvList, open: window.mvOpen, frame: window.mvFrame, state: window.__mv };

  /* ---------- 真实视频播放器：<video> 播原片，暂停时才取一张合成帧 ---------- */
  /* 两种模式互不干扰：
       video 模式   —— window.__mv.video 有值（源文件路径）：<video src="/api/merge/video?path=...">
                      播原片，播放中不请求预览；拖动结束/暂停才 POST /api/merge/preview。
       project 模式 —— 只有 DFL 帧目录（window.__mv.path，没有 .video）：继续走老的 mvFrame 逐帧 JPEG。
     后端（/api/merge/video、/api/merge/preview）在并行开发：404/501/超时一律降级，不弹错误风暴。 */
  window.__mgVideo = {
    src: "",           /* 已挂到 <video> 上的源文件路径 */
    png: "",           /* overlay 上当前那张合成帧 dataURL */
    wait: null,        /* preview 防抖 timer（120ms） */
    seq: 0,            /* preview 序号：过期响应直接丢 */
    blockUntil: 0,     /* 失败冷却到什么时候（3 秒内不再发同类请求） */
    warned: false,     /* 「轻提示」只给一次，不刷屏 */
    playing: false,
    streamDown: false, /* /api/merge/video 挂了（404/501）：退回逐帧预览 */
    syncing: 0,        /* >0：这一拍帧事件来自视频自身，别再回灌 video */
    ctl: null,         /* 当前这一发预览的 AbortController（切帧时 abort） */
    retryTimer: null   /* 超时后自动重试的定时器（新请求/切帧就作废） */
  };

  var MG_PLAY_SVG = '<svg viewBox="0 0 16 16" width="14" height="14" aria-hidden="true"><path d="M5.4 3.6 12.4 8l-7 4.4V3.6Z" fill="currentColor"/></svg>';
  var MG_PAUSE_SVG = '<svg viewBox="0 0 16 16" width="14" height="14" aria-hidden="true"><rect x="4.6" y="3.8" width="2.6" height="8.4" rx="0.8" fill="currentColor"/><rect x="8.8" y="3.8" width="2.6" height="8.4" rx="0.8" fill="currentColor"/></svg>';

  function mgVid() { return document.getElementById("mgVideo"); }
  function mgPrevImg() { return document.getElementById("mgPreview"); }
  function mgVidPath() { var mv = window.__mv || {}; return String(mv.video || ""); }
  /* 只有「真源视频」才走播放器；帧目录（DFL 项目）走老的逐帧图片 */
  function mgVideoMode() { return !!mgVidPath(); }
  window.mgVideoMode = mgVideoMode;

  /* 舞台中间的小字提示（传空 = 收起来，不占位）。
     标记位是 hidden 属性（HTML 里默认就是 hidden）：空态那句话由 #mgStageEmpty 负责，
     这里只放**临时状态**（无可用人脸 / 合成失败 / 视频流不可用…）。 */
  function mgHint(text) {
    var h = document.getElementById("mgStageHint");
    if (!h) return;
    if (!text) { h.style.display = "none"; h.hidden = true; return; }
    h.textContent = text;
    h.hidden = false;
    h.style.display = "";
  }
  window.mgHint = mgHint;

  /* 舞台上现在到底有没有素材：视频（__mv.video）、帧目录（__mv.path）、
     或已经准备完的项目（__mergeInputDir）。三者皆空 = 干净工作区。 */
  function mgHasSource() {
    var mv = window.__mv || {};
    return !!(mv.video || mv.path || window.__mergeInputDir);
  }
  window.mgHasSource = mgHasSource;

function mgStageEmpty(on) {
    /* 空态文案与按钮已按用户要求全部去掉：没有素材时舞台就只留棋盘底，不盖任何文字。
       函数保留成空壳，是因为 mvOpen / mgShowPrepared / bindMergeUI 三处仍在调它。 */
    return null;
  }
  window.mgStageEmpty = mgStageEmpty;

  /* 底图只有一种：合成结果（#mgPreview）压在原片上。
     「合成 / 原图」不再是一个整帧视图 —— 它变成**每张脸**的事：点某张脸的框，
     只把那张脸切回原像素（请求体 no_swap_faces），其它脸照常换。
     所以这里只剩「有没有合成图可贴」这一件事。 */

  function mgSetPlayIcon(on) {
    var b = document.getElementById("mtlPlay");
    if (b) b.innerHTML = on ? MG_PAUSE_SVG : MG_PLAY_SVG;
  }

  /* 合成图有就贴、没有就退回 video/帧图（不空屏）。检测框**始终显示**：
     它现在是「点框切这张脸的合成/原图」的唯一入口，界面上没有开关了。 */
  function mgApplyView() {
    var st = window.__mgVideo || {};
    var p = mgPrevImg();
    if (p) {
      var show = !!st.png && !st.playing;
      p.style.display = show ? "block" : "none";
    }
    /* 两种视图（合成 / 调试）都保留框和 chip：点框切单脸原图在哪一态下都要能用 */
    var cv = document.querySelector("#mgStage canvas.mgovl");
    if (cv) cv.style.display = "";
    var layer = document.getElementById("mgFaceLayer");
    if (layer) layer.style.display = "";
    if (window.mvRedrawBoxes) window.mvRedrawBoxes();
  }
  window.mgApplyView = mgApplyView;

  /* 只关显示、保留上一帧：播放中/拖动中画面退回 video，停下再贴回来 */
  function mgHideOverlay() {
    var p = mgPrevImg();
    if (p) p.style.display = "none";
  }

  window.mgShowOverlay = function (src) {
    var st = window.__mgVideo, p = mgPrevImg();
    st.png = src || "";
    if (p && src) {
      p.src = src;
      /* 合成图是异步解码的：解完再补画一次框。mgApplyView 已经画过一遍，但那次
         量到的是「旧图还没换掉」的版面（切页/首帧时会差几像素），补这一发才对齐。 */
      p.onload = function () {
        if (window.mvRedrawBoxes) window.mvRedrawBoxes();
        mgSyncFaceCards();
      };
    }
    mgApplyView();
    mgSyncFaceCards();
    if (src) mgHint("");
  };

  /* 把 <video> 挂到源文件上（同一个文件不重挂：重进页面时 currentTime 还留着） */
  window.mgAttachVideo = function (path) {
    var st = window.__mgVideo, v = mgVid();
    if (!v || !path) return null;
    st.src = path;
    /* 换素材：上一份素材的框 / chip 立刻清掉（不等新素材的检测回来） */
    if (window.mgInvalidateBoxes) window.mgInvalidateBoxes("attach", mgCmpFrameIdx());
    st.streamDown = false;
    v.style.display = "block";
    /* 老路径插的帧图要让位：底是 video 画面 */
    Array.prototype.forEach.call(document.querySelectorAll("#mgStage img"), function (im) {
      if (im.id !== "mgPreview") im.style.display = "none";
    });
    mgBindVideo(v);
    if (v.getAttribute("data-src") !== path) {
      v.setAttribute("data-src", path);
      v.src = "/api/merge/video?path=" + encodeURIComponent(path);
    }
    st.playing = false;
    mgSetPlayIcon(false);
    mgApplyView();
    return v;
  };

  /* 切到帧目录（DFL 项目）时把播放器收掉，回到 mvFrame 那条老路 */
  window.mgDetachVideo = function () {
    var st = window.__mgVideo, v = mgVid();
    st.src = "";
    if (window.mgInvalidateBoxes) window.mgInvalidateBoxes("detach", mgCmpFrameIdx());
    st.png = "";
    st.playing = false;
    st.streamDown = false;
    mgHideOverlay();
    mgPrevFlag("");          /* 退回逐帧模式：别把「仍在解码…」留在舞台上 */
    mgSetPlayIcon(false);
    if (v) {
      try { v.pause(); } catch (e) {}
      v.removeAttribute("data-src");
      v.removeAttribute("src");
      try { v.load(); } catch (e) {}
      v.style.display = "none";
    }
    /* 检测框 canvas 在视频模式下被藏过（非检测视图），回项目模式要放出来 */
    var cv = document.querySelector("#mgStage canvas.mgovl");
    if (cv) cv.style.display = "";
    mgHint("");
  };

  /* 播放中按屏幕刷新率推播放头：timeupdate 只有 ~4Hz，只靠它播放头会一跳好几帧 */
  var MG_HEAD_RAF = null;
  function mgHeadLoop() {
    MG_HEAD_RAF = null;
    var v = mgVid();
    if (!v || v.paused) return;
    window.mgSyncTimeline();
    mgHeadLoopArm();
  }
  function mgHeadLoopArm() {
    if (MG_HEAD_RAF) return;
    if (typeof requestAnimationFrame !== "function") return;
    MG_HEAD_RAF = requestAnimationFrame(mgHeadLoop);
  }
  function mgHeadLoopStop() {
    if (MG_HEAD_RAF && typeof cancelAnimationFrame === "function") cancelAnimationFrame(MG_HEAD_RAF);
    MG_HEAD_RAF = null;
  }

  /* <video> 事件：播放中只推播放头，停下来才要合成帧 */
  function mgBindVideo(v) {
    if (v.getAttribute("data-mgbound")) return;
    v.setAttribute("data-mgbound", "1");

    v.addEventListener("loadedmetadata", function () {
      var mv = window.__mv || {};
      /* 后端没给总帧数就自己按 duration × fps 算，时间轴才有长度可拖 */
      if (!(mv.frames > 0) && isFinite(v.duration) && v.duration > 0) {
        mv.frames = Math.round(v.duration * (mv.fps || 30));
        mgTimelineSync(mv.frames, mv.fps || 30);
        var tot = document.getElementById("mtlTotal");
        if (tot) tot.textContent = mv.frames;
      }
      mgHint("");
      window.mgRestoreTime();
      window.mgSyncTimeline();
      if (v.paused) window.mgPreviewSoon();
    });

    v.addEventListener("timeupdate", function () {
      window.__mv.t = v.currentTime;
      if (!v.paused) window.mgSyncTimeline();
    });

    v.addEventListener("play", function () {
      /* 播放开始 = 帧马上要变：先同步清框（canvas + chip 层），
         新检测没回来之前宁可没有框，也不留一个停在上一帧位置的框 */
      if (window.mgInvalidateBoxes) window.mgInvalidateBoxes("play", mgCmpFrameIdx());
      /* 播放中不请求预览（省性能）：overlay 收起，直接看原片帧 */
      window.__mgVideo.playing = true;
      mgHideOverlay();
      mgHint("");
      mgSetPlayIcon(true);
      window.mgSyncTimeline();
      mgHeadLoopArm();
    });

    v.addEventListener("pause", function () {
      window.__mgVideo.playing = false;
      window.__mv.t = v.currentTime;
      mgHeadLoopStop();
      mgSetPlayIcon(false);
      /* 停下来 = 这一帧定住了：框 + 合成一起补齐（mgFrameChange 里有这道门） */
      if (!window.__mgVideo.streamDown) window.mgFrameChange(mgCmpFrameIdx(), { why: "pause" });
    });

    v.addEventListener("seeked", function () {
      window.__mv.t = v.currentTime;
      if (!v.paused) return;                  /* 播放中的 seek 不算「停下来」 */
      if (!window.__mgVideo.streamDown) window.mgPreviewSoon();
    });

    /* 播到结尾：ended 不一定再补一个 pause，这里手动收尾，静静取一张尾帧 */
    v.addEventListener("ended", function () {
      var st = window.__mgVideo;
      st.playing = false;
      window.__mv.t = v.currentTime;
      mgHeadLoopStop();
      mgSetPlayIcon(false);
      if (!st.streamDown) window.mgFrameChange(mgCmpFrameIdx(), { why: "ended" });
    });

    v.addEventListener("error", function () {
      var st = window.__mgVideo;
      if (!st.src) return;            /* 正在卸载 source（切项目模式），不算失败 */
      /* /api/merge/video 还没就绪（404/501）或编码浏览器不吃：退回逐帧 JPEG */
      st.streamDown = true;
      v.style.display = "none";
      mgHint("视频流接口不可用 —— 已退回逐帧预览（/api/merge/video）");
      var f = document.getElementById("mtlFrame");
      window.mvFrame(parseInt(f ? f.value : 0, 10) || 0);
    });
  }

  /* 回到上次停下的位置（window.__mv.t）：重进合成页/刚挂上 source 时用 */
  window.mgRestoreTime = function () {
    var v = mgVid();
    var t = Number((window.__mv || {}).t) || 0;
    if (!v || !(t > 0)) return;
    if (isFinite(v.duration) && v.duration > 0) t = Math.min(t, Math.max(0, v.duration - 0.02));
    if (Math.abs(v.currentTime - t) <= 0.04) return;
    try { v.currentTime = t; } catch (e) {}
  };

  /* 视频自己走时钟 → 时间轴只跟着画播放头（syncing 里禁止回灌 video） */
  window.mgSyncTimeline = function () {
    var v = mgVid(), st = window.__mgVideo;
    if (!v || !mgVideoMode()) return;
    if (!window.MergeTimeline || !window.MergeTimeline.setFrame) return;
    var fps = (window.__mv || {}).fps || 30;
    var n = Math.round(v.currentTime * fps);
    if (window.MergeTimeline.getFrame && window.MergeTimeline.getFrame() === n) return;
    st.syncing++;
    try { window.MergeTimeline.setFrame(n); } catch (e) { /* 时间轴还没 init */ }
    st.syncing--;
  };

  /* 时间轴 → video：拖动、逐帧、播放头都改成写 currentTime */
  window.mgVideoSeek = function (n) {
    var v = mgVid(), st = window.__mgVideo;
    if (!v || !mgVideoMode()) return;
    if (st.syncing > 0) return;               /* 这一拍是视频反向同步，别再打回去 */
    if (st.streamDown) return;                /* 已经退回逐帧预览 */
    n = Math.max(0, parseInt(n, 10) || 0);
    var fps = (window.__mv || {}).fps || 30;
    var t = n / fps;
    var dur = (isFinite(v.duration) && v.duration > 0) ? v.duration : 0;
    if (dur > 0 && t > dur - 0.001) t = Math.max(0, dur - 0.001);
    if (!v.paused) { try { v.pause(); } catch (e) {} }   /* 拖动即暂停：要的是定格 */
    window.__mv.t = t;
    mgHideOverlay();
    if (Math.abs(v.currentTime - t) > 1e-3) {
      try { v.currentTime = t; } catch (e) {}            /* seeked 里再取预览 */
    } else {
      window.mgPreviewSoon();
    }
  };

  /* 播放/暂停：视频模式下由 <video> 自己走时钟，时间轴的 setInterval 步进不参与 */
  window.mgTogglePlay = function () {
    var v = mgVid();
    if (!v || !mgVideoMode() || window.__mgVideo.streamDown) return;
    if (v.paused) {
      var p = null;
      try { p = v.play(); } catch (e) { p = null; }
      if (p && p.catch) p.catch(function () { mgSetPlayIcon(false); });
    } else {
      try { v.pause(); } catch (e) {}
    }
  };

  /* 右栏当前参数 → 预览请求（字段名和后端契约一致：video / t / model / settings）。
     新契约的 settings 由 mgCmpSettings() 组装（ct_mode / blend / mask_mode /
     mask_erode / mask_blur），这里只保留老预览接口要的那一组做兜底。 */
  function mgPreviewSettings() {
    var s = {};
    try {
      s = (window.MergePages && window.MergePages.settings) ? window.MergePages.settings() : {};
    } catch (e) { s = {}; }
    var neo = (window.mgCmpSettings ? window.mgCmpSettings() : {});
    var out = {
      mode: s.mode || neo.blend || "seamless",
      detector: s.detector || "TinyMog",
      face_type: s.face_type || "whole_face",
      /* 默认值同步：色域迁移 = mt、融合 = seamless */
      mask_source: s.mask_source || "XSegLite",
      mask_mode: s.mask_mode || "prd×dst",
      color_transfer: s.color_transfer || "mt",
      erode_mask: s.erode_mask || 0,
      blur_mask: (s.blur_mask == null ? 4 : s.blur_mask)
    };
    /* 新字段一并带上：后端认哪个用哪个，互不覆盖 */
    Object.keys(neo).forEach(function (k) { out[k] = neo[k]; });
    return out;
  }

  /* ---------- 合成器预热：趁用户还在看，先让后端把模型读进显存 ---------- */
  /* 预览接口有 5 秒多的冷启动（第一次「暂停 → 出片」全花在 load model 上）。
     在几个明显马上要用到模型的时机提前打一枪 POST /api/merge/preview/warm，
     等用户真暂停时后端已经是热的。后端还在并行开发，没这个接口就是 404 ——
     一律静默当没发生。原则：fire-and-forget，不 await、不显示 loading、
     失败不 toast、不碰 /api/merge/preview 的正常流程。 */
  var MG_WARM_AT = {};          /* 模型名 → 上次请求时间戳，用来做 10 秒去重 */
  var MG_WARM_GAP = 10000;
  window.__mgWarm = function (model) {
    var name = String(model == null ? "" : model).trim();
    if (!name) {
      /* 调用点没给名字（打开视频 / 进页面）时，沿用预览那套「当前选中模型」的取法 */
      try { name = String(mgPreviewModel() || "").trim(); } catch (e) { name = ""; }
    }
    if (!name) return;          /* 还没有可选模型（列表没渲染出来）：什么都不做 */
    var now = Date.now();
    /* 同一个模型 10 秒内只发一次：进页面、打开视频、切模型经常连着触发三次，
       重复请求只会让后端把刚载入的权重又丢一遍。换了模型则立刻放行。 */
    if (MG_WARM_AT[name] && (now - MG_WARM_AT[name]) < MG_WARM_GAP) return;
    MG_WARM_AT[name] = now;
    /* 3 秒超时：后端慢/卡住时不留下半开的连接；超时走 catch，什么都不做 */
    try {
      var ctrl = (typeof AbortController === "function") ? new AbortController() : null;
      var timer = ctrl ? setTimeout(function () { try { ctrl.abort(); } catch (e) {} }, 3000) : null;
      fetch("/api/merge/preview/warm", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ model: name }),
        signal: ctrl ? ctrl.signal : undefined
      }).then(function (r) {
        if (timer) { clearTimeout(timer); timer = null; }
        return r.json();
      }).catch(function () {});   /* 404 / 断网 / 超时中止：静默，绝不冒泡 */
    } catch (e) {}
  };
  function mgPreviewModel() {
    var el = document.querySelector("#mgModels .mgitem.on .nm");
    var name = el ? String(el.textContent || "").trim() : "";
    if (name) return name;
    /* 卡片还没渲染出来时退回右栏参数里的同名项 */
    try {
      if (window.MergePages && window.MergePages.settings) {
        name = String(window.MergePages.settings().model || "").trim();
      }
    } catch (e) { name = ""; }
    return name;
  }

  /* ---------- 模型多选 + 每个模型负责哪张脸 ---------- */
  /* 契约：settings.models = [{name, faces:[...]}]，faces 是从 0 起的脸序号，
     faces=[] 或缺省 = 全部脸。多选只是界面层的事：选中态仍然挂在 .mgitem.on 上，
     允许多条同时 .on；脸分配记在 MG_FACES[name] 里（只存用户点过徽章的），
     没点过就按「选中顺序」现算默认值 —— 这样列表轮询重绘也不会把分配弄丢。 */
  var MG_FACES = {};        /* 模型名 → [脸序号]；[] = 全部脸 */
  var MG_ANCHOR = null;     /* Shift+点击的上一次落点，范围选择从它开始 */
  var MG_SEL_AUTO = true;   /* true = 当前选中是列表自动挑的（用户还没点过） */
  var MG_FACE_N = 3;        /* 徽章循环档位：all → face 0 → face 1 → face 2 → all */

  function mgRows() {
    var box = el("mgModels");
    if (!box) return [];
    return Array.prototype.filter.call(box.children, function (x) {
      return x.classList && x.classList.contains("mgitem");
    });
  }
  /* 选中项（按列表顺序）：顺序就是脸的默认分配顺序 */
  function mgSelNames() {
    return mgRows().filter(function (r) { return r.classList.contains("on"); })
      .map(function (r) { return String(((r.querySelector(".nm") || {}).textContent) || "").trim(); })
      .filter(function (n) { return !!n; });
  }
  /* 没点过徽章时的默认：第一个选中管全部脸，第二个起依次 face 1、face 2 … */
  function mgDefaultFaces(pos) { return pos <= 0 ? [] : [pos]; }
  function mgFacesCur(name) {
    if (Object.prototype.hasOwnProperty.call(MG_FACES, name)) return MG_FACES[name] || [];
    var pos = mgSelNames().indexOf(name);
    return mgDefaultFaces(pos < 0 ? 0 : pos);
  }
  function mgFaceKey(faces) { return (!faces || !faces.length) ? "" : String(faces[0]); }
  function mgFaceStates() {
    var st = [[]];                                  /* 空数组 = 全部脸 */
    for (var i = 0; i < MG_FACE_N; i++) st.push([i]);
    return st;
  }
  /* 点徽章循环：all → face 0 → face 1 → face 2 → all */
  function mgCycleFaces(name) {
    var cur = mgFaceKey(mgFacesCur(name));
    var st = mgFaceStates(), at = -1;
    for (var i = 0; i < st.length; i++) if (mgFaceKey(st[i]) === cur) { at = i; break; }
    MG_FACES[name] = st[(at + 1) % st.length];       /* at = -1（档位外）也回到 all */
    mgSyncFaces();
  }
  /* 按列表顺序维护「谁管哪张脸」：徽标（face all / face N）按要求不再显示，
     这里只更新 dataset.faces —— 点击切换的绑定与 MG_FACES 状态都保留 */
  function mgSyncFaces() {
    mgRows().forEach(function (r) {
      var bd = r.querySelector(".mgface");
      var name = String(((r.querySelector(".nm") || {}).textContent) || "").trim();
      if (!bd) return;
      var faces = mgFacesCur(name);
      bd.hidden = true;                       /* 不再显示：hidden 恒为真（绑定/状态照旧） */
      bd.title = "";                          /* 徽标不显示了，tooltip 一并去掉 */
      bd.dataset.faces = mgFaceKey(faces);
    });
  }
  /* 选中判定：**点一下就是切换**（多选，0 / 1 / N 个都合法）；Shift 从锚点到本条整段选中。
     以前普通点击是单选（点一行把别的全清掉），想同时用两个模型只能按 Ctrl —— 用户不知道
     有这个手势，看到的永远是单选。现在取消单选语义：点一次选中，再点一次取消。
     Ctrl/Cmd 与普通点击同义（留给肌肉记忆，不再是「多选的唯一入口」）；
     也不再有「最后一条不许取消」的守卫 —— 0 个选中是正常状态，导出入口自己会禁用并提示。 */
  function mgPickRow(box, idx, ev) {
    MG_SEL_AUTO = false;                  /* 用户自己点了：以后不许再自动改选中 */
    var rows = Array.prototype.filter.call(box.children, function (x) {
      return x.classList && x.classList.contains("mgitem");
    });
    var row = rows[idx];
    if (!row) return;
    var shift = !!(ev && ev.shiftKey);
    if (shift) {
      var a = (MG_ANCHOR == null) ? idx : MG_ANCHOR;
      a = Math.max(0, Math.min(rows.length - 1, a));
      var lo = Math.min(a, idx), hi = Math.max(a, idx);
      rows.forEach(function (x, i) { x.classList.toggle("on", i >= lo && i <= hi); });
    } else {
      row.classList.toggle("on");         /* 切换：选中 ↔ 取消 */
      MG_ANCHOR = idx;
    }
    mgSyncFaces();
    /* 选中项变了 = chip 上那个名字变了（单选显示它、多选显示「第一个 +N」、0 个显示
       「选择模型」）。这里同步一次，不必等合成回包 —— 单模型回包只带 model/dfm、
       不带 faces[]，等它反而会让 chip 停在旧名字上。 */
    mgSyncFaceCards();
    if (window.mvRedrawBoxes) window.mvRedrawBoxes();
    mgWarmSelected();
    mgSyncExportGate();                    /* 选中项变了 → 导出入口跟着启用/禁用 */
    /* 左栏换模型 = 换「整帧用哪个 dfm」：右边参数一动就会重算，这里以前不动，
       于是"选了模型画面没反应"，得再去点一下别的东西才刷新。 */
    if (window.mgCmpSoon) window.mgCmpSoon();
  }
  /* 多选时每个选中的模型都预热一遍（__mgWarm 自带 10 秒去重，重复调用不要钱） */
  function mgWarmSelected() {
    var names = mgSelNames();
    if (!names.length) { try { window.__mgWarm(); } catch (e) {} return; }
    names.forEach(function (n) { try { window.__mgWarm(n); } catch (e) {} });
  }
  window.mgWarmSelected = mgWarmSelected;
  /* 组装请求用：models=[{name, faces}]，顺序 = 列表里的选中顺序 */
  function mgModelsPayload() {
    return mgSelNames().map(function (n) {
      /* 单模型也带 models，faces=[] 等价于「全部脸」；老字段 model 照旧保留 */
      return { name: n, faces: (mgFacesCur(n) || []).slice() };
    });
  }
  window.mgModelsPayload = mgModelsPayload;

  /* 导出/合成请求用的 models：**左栏选中的全部模型**，一个不落。
     预览那一路（mgModelsPayload）的 faces 还按「第几个选中管第几张脸」的默认分摊；
     导出不玩分摊 —— 没在画面上手动指过脸就是 faces: []（= 这个模型管全部脸），
     手动指过的沿用画面上的分配。这样「选 2 个 → models 长度 2」是稳的，
     不会因为人脸库还没分配就退化成空数组（以前导出走的就是 mgFaceModelsPayload，
     它一张脸都没分配时返回 []，等于一个模型都没带）。 */
  function mgExportModelsPayload() {
    var sel = mgSelNames();
    var byFace = [];
    try {
      byFace = (window.mgFaceModelsPayload && window.mgFaceModelsPayload()) || [];
    } catch (e) { byFace = []; }
    var assigned = {};
    byFace.forEach(function (x) { if (x && x.name) assigned[x.name] = (x.faces || []).slice(); });
    /* 以左栏选中为准：分配表只用来给选中项补 faces，**不引入没选中的模型**。
       以前是「分配表 ∪ 选中」，一旦画面上给别的模型指过脸（或早先选过多模型、
       分配记录还留着），导出请求里就多带一个模型 —— 后端据此判成「多模型」，
       本来单模型的一单被拆成两趟，整段视频白跑一遍。 */
    var out = [];
    sel.forEach(function (n) {
      out.push({ name: n, faces: (assigned[n] || []).slice() });
    });
    if (!out.length) {
      var seen = {};
      byFace.forEach(function (x) {
        if (!x || !x.name || seen[x.name]) return;
        seen[x.name] = 1;
        out.push({ name: x.name, faces: (x.faces || []).slice() });
      });
    }
    return out;
  }
  window.mgExportModelsPayload = mgExportModelsPayload;

  /* 导出入口的开关：0 个模型 → 按钮禁用 + 「先选一个模型」。
     返回当前选中个数（调用方可以直接拿来判断）。 */
  function mgSyncExportGate() {
    var n = mgSelNames().length;
    var btn = document.getElementById("mgExport");
    if (btn) {
      btn.disabled = (n === 0);
      btn.title = n ? "" : "先选一个模型";
      btn.setAttribute("aria-disabled", n === 0 ? "true" : "false");
    }
    ["mgExportHint", "mgStartHint"].forEach(function (id) {
      var h = document.getElementById(id);
      if (!h) return;
      h.hidden = (n !== 0);
      h.textContent = "先选一个模型";
    });
    return n;
  }
  window.mgSyncExportGate = mgSyncExportGate;


  /* 暂停/拖动结束 → 取一张合成帧。120ms 防抖：拖动会连着「结束」好几次 */
  window.mgPreviewSoon = function () {
    var st = window.__mgVideo;
    if (st.wait) clearTimeout(st.wait);
    st.wait = setTimeout(function () { st.wait = null; window.mgPreviewNow(); }, 120);
  };

  /* ---------- 预览请求：超时自动重试一次 ----------
     冷启动（ffmpeg 首帧解码慢 / 机械盘 / 4K 高码率）时 5 秒根本不够：以前前端一 abort 就只能等
     用户下一次 seek —— 表现就是「第一次拖没反应，再拖一次才好」。
     现在：第一次 5 秒超时 → 不清画面、不弹错，后台按「当前」帧号再发一次（超时放宽到 15 秒）；
     两次都超时才提示一次，提示里带「重试」。
     并发：新的预览一定 ++seq 并 abort 上一发；落地前再比一次 seq + 帧号，
     上一帧的结果绝不会贴到新帧上。 */
  var MG_PREV_TMO = 5000;          /* 首次请求超时（毫秒） */
  var MG_PREV_TMO_RETRY = 15000;   /* 自动重试的超时（毫秒）：放宽，够一次冷解码 */
  var MG_PREV_COOL = 3000;         /* 失败冷却（毫秒）：只挡用户主动连拖 */
  var MG_PREV_GAP = 200;           /* 首次超时 → 自动重试之间的间隔 */

  /* 舞台左下角一枚「仍在解码…」小胶囊：pointer-events:none，不遮画面、不吃拖动。
     呼吸灯走 Web Animations，不动 app.css。 */
  function mgPrevFlag(text) {
    var stage = document.getElementById("mgStage");
    if (!stage || !stage.appendChild) return;
    var el = document.getElementById("mgPrevChip");
    if (!text) { if (el) el.style.opacity = "0"; return; }
    if (!el) {
      el = document.createElement("div");
      el.id = "mgPrevChip";
      el.style.cssText = "position:absolute;left:10px;bottom:10px;z-index:5;pointer-events:none;" +
        "display:flex;align-items:center;gap:6px;padding:3px 9px;border-radius:999px;" +
        "font-size:11px;line-height:1.6;color:#fff;background:rgba(12,16,24,.55);" +
        "box-shadow:0 1px 6px rgba(0,0,0,.18);opacity:0;transition:opacity .16s ease";
      var dot = document.createElement("i");
      dot.style.cssText = "width:6px;height:6px;border-radius:50%;background:#7dd3fc;flex:none";
      var tx = document.createElement("span");
      tx.id = "mgPrevChipTx";
      tx.style.cssText = "font-size:11px;line-height:1.6;color:#fff";
      el.appendChild(dot);
      el.appendChild(tx);
      stage.appendChild(el);
      /* 呼吸灯：WAAPI，不需要改样式表 */
      if (dot.animate) {
        try {
          dot.animate([{ opacity: 0.25 }, { opacity: 1 }, { opacity: 0.25 }],
                      { duration: 1100, iterations: Infinity, easing: "ease-in-out" });
        } catch (e) { /* 没有 WAAPI 就留个静态点，功能不受影响 */ }
      }
    }
    var span = el.querySelector ? el.querySelector("span") : null;
    if (span && span.textContent !== text) span.textContent = text;
    el.style.opacity = "1";
  }
  window.mgPrevFlag = mgPrevFlag;

  /* 两次都失败：右下角提示一次，带「重试」按钮（跳过冷却 + 重新计数提示） */
  function mgPrevFailTip(isTimeout, detail) {
    var st = window.__mgVideo;
    if (st.warned) return;                        /* 同一轮失败只提示一次 */
    st.warned = true;
    var el = null;
    try {
      el = toast(isTimeout ? "预览超时" : "预览不可用",
                 isTimeout ? (detail || "这一帧解码太慢，已自动重试一次")
                           : String(detail || "").slice(0, 80),
                 isTimeout ? "warn" : "error", 12000);
    } catch (e) { el = null; }
    if (!el || !el.querySelector) return;
    var tt = el.querySelector(".tt");
    var host = tt ? tt.parentNode : null;
    if (!host || !host.appendChild) return;
    var btn = document.createElement("button");
    btn.type = "button";
    btn.textContent = "重试";
    btn.style.cssText = "margin-top:6px;padding:2px 10px;font-size:11px;line-height:1.6;" +
      "border-radius:6px;border:1px solid currentColor;background:transparent;" +
      "color:var(--accent);cursor:pointer";
    btn.addEventListener("click", function () {
      try { dismissToast(el); } catch (e) {}
      st.blockUntil = 0;
      st.warned = false;
      window.mgPreviewNow({ force: true });
    });
    host.appendChild(btn);
  }
  window.mgPrevFailTip = mgPrevFailTip;

  window.mgPreviewNow = function (opt) {
    opt = opt || {};
    var st = window.__mgVideo, v = mgVid();
    if (!v || !mgVideoMode()) return Promise.resolve(null);
    if (st.streamDown) return Promise.resolve(null);
    if (!v.paused) return Promise.resolve(null);                   /* 播放中不要预览 */
    /* 帧刚变、检测还没结论时先别合成：没人脸的那一帧根本不该跑 dfm
       （mgFrameChange 拿到 detect 结果后会把这道门打开） */
    if (!MG_CHAIN.gateOpen) return Promise.resolve(null);
    /* 只有一条合成路：/api/merge/frame（带 settings / models）。这条老预览接口
       只在「后端连 POST /frame 都不认」时才会走到（见 mgCmpLegacy）。 */
    /* 3 秒冷却只挡「用户主动连拖」：自动重试（retry）和手动重试（force）都不受影响 */
    if (!opt.force && !opt.retry && Date.now() < st.blockUntil) return Promise.resolve(null);

    var seq = ++st.seq;
    var fps = (window.__mv || {}).fps || 30;
    if (opt.force) st.warned = false;                              /* 手动重试：允许再提示一次 */
    if (st.retryTimer) { clearTimeout(st.retryTimer); st.retryTimer = null; }
    if (st.ctl) { try { st.ctl.abort(); } catch (e) {} st.ctl = null; }   /* 切帧：上一发真的掐掉 */
    mgPrevFlag("");

    function frameIdx() { return Math.round(v.currentTime * fps); }
    function body(t) {
      var idx = frameIdx();
      return {
        /* 新契约：/api/merge/frame 与 /api/merge/merged 都收 frame + settings + models。
           老预览接口只认 video/t，多带的字段它自己会忽略。 */
        frame: idx, idx: idx, n: idx,
        video: mgVidPath(),
        t: t,
        /* 自动分配要读的人脸库（空串 = 默认库） */
        project: window.__mergeInputDir || "",
        model: mgFaceSingleModel(),
        /* 多选契约：models=[{name, faces}]；单模型时 faces=[]（= 全部脸，向后兼容） */
        models: (window.mgFaceModelsPayload ? window.mgFaceModelsPayload() : mgModelsPayload()),
        settings: mgPreviewSettings()
      };
    }

    /* round 0 = 首次（5 秒），round 1 = 自动重试（15 秒）。每一发都重读 currentTime：
       重试必须带「当前」帧号，不能沿用上一发写死的 t。 */
    function send(round) {
      var ctrl = (typeof AbortController === "function") ? new AbortController() : null;
      var timedOut = false;
      var t = Math.round(v.currentTime * 1000) / 1000;
      var idx = frameIdx();
      var timer = ctrl ? setTimeout(function () {
        timedOut = true;
        try { ctrl.abort(); } catch (e) {}
      }, round > 0 ? MG_PREV_TMO_RETRY : MG_PREV_TMO) : null;
      st.ctl = ctrl;

      return fetch("/api/merge/preview", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body(t)),
        signal: ctrl ? ctrl.signal : undefined
      }).then(function (r) {
        return r.json().then(function (d) {
          if (!r.ok || !d || d.ok !== true) {
            var err = new Error((d && d.detail) || ("HTTP " + r.status));
            err.status = r.status;
            throw err;
          }
          return d;
        });
      }).then(function (d) {
        if (timer) clearTimeout(timer);
        if (seq !== st.seq) return null;            /* 过期响应，丢掉 */
        if (frameIdx() !== idx) {                   /* 帧已经换了：别贴到新帧上 */
          mgPrevFlag("");
          return null;
        }
        st.ctl = null;
        st.warned = false;
        mgPrevFlag("");
        window.__mv.lastPreviewMs = d.ms;
        if (d.has_face === false || !d.png) {
          st.png = "";
          mgHideOverlay();
          mgHint("这一帧没有检测到人脸");
          return null;
        }
        mgHint("");
        window.mgShowOverlay(d.png);
        if (window.mgTimingMark) window.mgTimingMark("t_frame");
        /* 这张是「左栏那一个模型换全部脸」的老结果（/api/merge/preview 不认人脸库）；
           合成视图要的是「每张脸按人脸库自动分配 dfm」。紧接着让 /api/merge/frame
           补一张正确的 —— 它的缓存键没变就直接命中，不会重复渲染。
           从 /frame 兜底回来那次（fromFallback）不再触发，否则两条路互相踢皮球。 */
        if (!(opt && opt.fromFallback) && window.mgCmpSoon) window.mgCmpSoon();
        /* 框不在这里画：检测是「帧变了」那一刻单独并发发的（见 mgFrameChange）。
           这一发只负责把合成图贴上去 —— 谁先回来谁先渲染，谁也不等谁。 */
        return d;
      }).catch(function (e) {
        if (timer) clearTimeout(timer);
        if (seq !== st.seq) return null;            /* 已被新请求取代：不提示、不冷却 */
        var isTimeout = !!(e && e.name === "AbortError") || timedOut;
        if (isTimeout && round === 0) {
          /* 第一次超时：画面保持上一帧，后台换 15 秒再来一次 */
          mgPrevFlag("仍在解码…（自动重试）");
          if (st.retryTimer) clearTimeout(st.retryTimer);
          st.retryTimer = setTimeout(function () {
            st.retryTimer = null;
            if (seq !== st.seq) return;               /* 这期间用户已经切帧：不再发 */
            if (!v.paused || st.streamDown) { mgPrevFlag(""); return; }
            var p = send(1);
            if (p && p.catch) p.catch(function () {});
          }, MG_PREV_GAP);
          return null;
        }
        /* 第二次也超时 / 非超时错误：清状态 → 3 秒冷却 → 提示一次 */
        mgPrevFlag("");
        st.ctl = null;
        st.blockUntil = Date.now() + MG_PREV_COOL;
        var secs = Math.round((round > 0 ? MG_PREV_TMO_RETRY : MG_PREV_TMO) / 1000);
        mgPrevFailTip(isTimeout,
          isTimeout ? ("这一帧解码超过 " + secs + " 秒，已自动重试一次仍没出来")
                    : ((e && e.message) || String(e)));
        return null;
      });
    }

    return send(0);
  };

  /* 进合成页：已有视频就把上次停下的时间点接回来（window.__mv.t） */
  window.mgResumeVideo = function () {
    var st = window.__mgVideo, v = mgVid();
    var path = mgVidPath();
    if (!path || !v) return;
    if (st.src !== path) { window.mgAttachVideo(path); return; }   /* loadedmetadata 里恢复 t */
    mgHint("");
    mgApplyView();
    if (v.readyState >= 1) window.mgRestoreTime();
    if (v.paused) window.mgPreviewSoon();
  };

  /* 视频模式下把播放键/空格拦下来交给 <video>：时间轴那套 setInterval 步进会和视频时钟打架。
     捕获阶段拦在 #p-merge / window 上，项目模式原样放行（老行为不变）。 */
  function mgBindTransport() {
    var page = document.getElementById("p-merge");
    if (page && !page.getAttribute("data-mgtransport")) {
      page.setAttribute("data-mgtransport", "1");
      page.addEventListener("click", function (e) {
        /* 视频流挂了、退回逐帧预览时，播放键还给时间轴自己处理 */
        if (!mgVideoMode() || (window.__mgVideo || {}).streamDown) return;
        var b = (e.target && e.target.closest) ? e.target.closest("#mtlPlay") : null;
        if (!b) return;
        e.preventDefault();
        e.stopPropagation();
        window.mgTogglePlay();
      }, true);
    }
    if (window.__mgKeyBound) return;
    window.__mgKeyBound = true;
    window.addEventListener("keydown", function (e) {
      if (!mgVideoMode() || (window.__mgVideo || {}).streamDown) return;
      var pg = document.getElementById("p-merge");
      if (!pg || !pg.classList || !pg.classList.contains("on")) return;
      /* 人脸库打开时空格归卡片选择，不要顺手把后台视频播起来 */
      var fdbView = document.getElementById("mgFaceDb");
      if (fdbView && fdbView.classList.contains("in")) return;
      if (e.ctrlKey || e.metaKey || e.altKey) return;
      var t = e.target || {};
      var tag = String(t.tagName || "").toUpperCase();
      if (tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT" || t.isContentEditable) return;
      if (e.key !== " " && e.key !== "Spacebar") return;
      e.preventDefault();
      e.stopPropagation();
      window.mgTogglePlay();
    }, true);
  }
  window.mgBindTransport = mgBindTransport;

