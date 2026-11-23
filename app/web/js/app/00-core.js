/* DeepFaceLab 4.1 — 前端。所有数据都来自本地 /api，界面本身不含任何算法。 */
/* ==========================================================================
   分片目录（app/web/js/app/，后端按文件名顺序拼成 /static/js/app.js）：
     00-core 基础设施 · 01-dataset 数据集页 · 02-merge-preview 合成器预览
     10-merge-settings 合成参数 · 15-proj-out 项目导出 · 16-pipe-draw 流水线绘制
     17-water 缓存水位 · 18-bypass 无脸旁路 · 19-eff-drawer 生效配置抽屉
     20-progress-fused 融合导出进度 · 21-face68-3d-privacy 68点/3D/隐私幕布
     22-account-bili 账号与B站 · 23-home-bili-dl 主页/下载 · 25-train-cfg 训练参数
     26-train-live 训练页控制与预览 · 27-tools-remote 工具组/远程访问
     30-studio-timeline 多轨时间轴 · 31-studio-faces 人脸库导出 · 32-mask-page 遮罩绘制
     40-train-extra 训练参数补齐 · 41-dataset-cfg 引擎/设置/欢迎页 · 50-misc · 60-tail
   外层是**一个 IIFE 跨所有分片**，所以分片单独 node --check 会报括号不匹配；
   改完请跑 node app/tools/js/check_bundle.js（它会报出具体分片与行号）。
   拼接逐字节等价于拆分前的 app.js（拆分时校验过 roundtrip）。

   下面是原来的分节目录（Ctrl+F 用）：
   用 Ctrl+F 搜下面任一条标题即可跳转；分区横幅本身保留在原文里。

     · hash 路由（#p-key）
     · 顶栏面包屑
     · 语言
     · 数据（全部来自 /api/state）
     · 合成工作室：真实视频预览（后端 /api/merge/*）
     · 素材准备：视频 → 抽帧 + 对齐 → 可合成项目
     · 检测视图：RetinaFace 出框，画在 canvas 叠加层上
     · 面部 68 点：编辑器保存过就用保存的
     · 路径框拖放
     · 3D 人脸编辑器（debug 页）
     · 隐私幕布：68 点人脸会跟着鼠标看
     · 隐私幕布：3D 头部模型只用来算 68 个特征点
     · 账号
     · B 站扫码登录
     · 通用确认弹层
     · 主页：根据真实数据推下一步
     · 下载进程：按实际任务渲染
     · 视频列表：内置全选 / 计数 / 下载
     · 模拟：分段拉取（B 站接口一次只给 20 条，且要限流）
     · 模拟：下载推进
     · 角度可视化：坐标轴 + 散点 / 格子图
     · 参数设置（照 4.0 training_config_page 的分组）
     · 训练页控制按钮：暂停 / 继续 · 立即保存 · 停止
     · 训练中（骨架：数据先用真实模型 + 模拟值）
     · 训练预览：单图 + 手动刷新
     · 通知调试
     · 侧边栏「工具」分组
     · 折叠态：工具组飞出层
     · 远程访问（把整个界面开到 0.0.0.0）
     · 占位数据：形状与 4.0 的 cutSegments / angleSegments 一致
     · 渲染
     · 对外 API
     · 播放（占位驱动；4.0 由 video 元素驱动）
     · 交互
     · 初始化
     · 合成器参数记忆
     · 事件绑定
     · 1.45 设备下拉：显示真名而不是序号
     · 1.5 界面构建时间
     · 2. 参数页搜索
     · 3. 训练页「继续训练」置顶
     · 挂钩页面切换
     · 权重分包门控（工具 → 遮罩绘制）
   ========================================================================== */

(function () {
  "use strict";

  var PAGE_IDS = {
    home: "p-home", extract: "p-extract", "extract-run": "p-extract-run", dataset: "p-dataset", train: "p-train",
    run: "p-run", "train-cfg": "p-train-cfg",
    "tool-mask": "p-tool-mask", "tool-bili": "p-tool-bili",
    "tool-color": "p-tool-color", "tool-frames": "p-tool-frames", "tool-frames-run": "p-tool-frames-run", "tool-angle": "p-tool-angle", "tool-engine": "p-tool-engine",
    "train-convert": "p-train-convert",
    "dataset-run": "p-dataset-run",
    "bili-jobs": "p-bili-jobs",
    debug: "p-debug",
    merge: "p-merge", settings: "p-settings", about: "p-about"
  };

  function $(id) { return document.getElementById(id); }
  function api() {
    if (window.desktop) return window.desktop;
    return window.pywebview ? window.pywebview.api : null;
  }

  var PAGE_EASE = "cubic-bezier(0.25, 0.46, 0.45, 0.94)";
  function reduced() {
    try { return window.matchMedia("(prefers-reduced-motion: reduce)").matches; } catch (e) { return false; }
  }

  /* ---------- hash 路由（#p-key） ---------- */
  function hashKey() {
    return String(location.hash || "").replace(/^#/, "");
  }

  window.addEventListener("hashchange", function () {
    if (HASH_SYNCING) return;
    var k = hashKey();
    if (k && PAGE_IDS[k]) go(k, { keepHash: true });
  });

  /* ---------- 顶栏面包屑 ---------- */
  var CRUMBS = {
    train: [{ t: "模型选择" }],
    run: [{ t: "模型选择", go: "train" }, { t: "参数设置", go: "train-cfg" }, { t: "训练界面" }],
    "train-cfg": [{ t: "模型选择", go: "train" }, { t: "参数设置" }],
    "train-convert": [{ t: "模型选择", go: "train" }, { t: "模型转换" }],
    "tool-mask": [{ t: "工具" }, { t: "遮罩绘制" }],
    "tool-bili": [{ t: "工具" }, { t: "B 站下载" }],
    "tool-color": [{ t: "工具" }, { t: "色域迁移" }],
    "tool-frames": [{ t: "工具" }, { t: "视频切帧" }],
    "tool-frames-run": [{ t: "工具" }, { t: "视频切帧", go: "tool-frames" }, { t: "切帧中" }],
    "tool-angle": [{ t: "工具" }, { t: "角度可视化" }],
    "tool-engine": [{ t: "工具" }, { t: "引擎编译" }],
    debug: [{ t: "关于", go: "about" }, { t: "调试" }],
    "bili-jobs": [{ t: "工具" }, { t: "B 站下载", go: "tool-bili" }, { t: "下载中" }],
    merge: [{ t: "合成工作室" }]
  };

  /* 子页面归属哪个导航项 */
  var NAV_ALIAS = { run: "train", "train-cfg": "train", "extract-run": "extract", "dataset-run": "dataset", "tool-frames-run": "tool-frames", debug: "about" };

  function renderCrumbs(key) {
    var box = $("crumbs");
    if (!box) return;
    var trail = (CRUMBS[key] || []).slice();
    /* 从训练界面退回模型列表后，顶栏要留一条「训练界面」的路 ——
       否则只能靠卡片上的按钮，而那个按钮以前还会被禁用，直接走不回去。 */
    if (key === "train" && window.__runModel) trail.push({ t: "训练界面", run: true });
    box.innerHTML = "";
    if (!trail.length) return;
    trail.forEach(function (c, i) {
      /* 英文模式下面包屑也要走字典 */
      var lang = (window.I18N && window.I18N.get) ? window.I18N.get() : "zh";
      var label = c.t;
      if (lang === "en" && window.I18N && window.I18N.dict && window.I18N.dict[c.t]) {
        label = window.I18N.dict[c.t];
      }
      /* 注意：这里重建成新对象时要把 run/step 一起带上 ——
         以前只留了 t 和 go，导致「训练界面」那节虽然画出来了却没有绑定点击。 */
      c = { t: label, go: c.go, run: c.run, step: c.step };
      if (i > 0) {
        var sep = document.createElement("span");
        sep.className = "crumb-sep";
        sep.textContent = "›";
        box.appendChild(sep);
      }
      var el = document.createElement("span");
      el.className = "crumb" + (i === trail.length - 1 ? " on" : "");
      el.textContent = c.t;
      el.setAttribute("data-i18n-skip", "1");
      if (c.run) {
        el.classList.add("link");
        el.addEventListener("click", function () {
          if (window.openRun) window.openRun(window.__runModel);
          else go("run");
        });
      } else if (c.step && i < trail.length - 1) {
        el.classList.add("link");
        el.addEventListener("click", function () {
          /* 人脸库还在时点「合成工作室」= 返回并取消导出，必须走二次确认；
             直接 mgStep('main') 会绕过 FaceDB 的关闭确认。 */
          var fdbView = document.getElementById("mgFaceDb");
          if (c.step === "main" && window.FaceDB && window.FaceDB.isOpen && window.FaceDB.isOpen() &&
              fdbView && fdbView.classList.contains("in")) {
            window.FaceDB.requestClose();
            return;
          }
          if (window.mgStep) window.mgStep(c.step);
        });
      } else if (c.go && i < trail.length - 1) {
        el.classList.add("link");
        el.addEventListener("click", function () { go(c.go); });
      }
      box.appendChild(el);
    });
  }

  /* 每个导航组的最后位置：点「训练器」应回到上次待的界面，而不是永远回列表 */
  var LAST_PAGE = {};
  var PAGE_GROUP = { run: "train", "train-cfg": "train", "extract-run": "extract", "dataset-run": "dataset", "tool-mask": "tools",
                     "tool-bili": "tools", "tool-color": "tools", "tool-frames": "tools", "tool-frames-run": "tools", "tool-angle": "tools", "train-convert": "train",
                     debug: "about" };

  function groupOf(key) { return PAGE_GROUP[key] || key; }

  /* 子页记忆：从侧边栏回到「训练器 / B 站下载」时，恢复上次停留的子页 */
  var SUB_PARENT = { run: "train", "train-cfg": "train", "train-convert": "train", "extract-run": "extract", "dataset-run": "dataset", "bili-jobs": "tool-bili" };
  var SUB_MEMORY = {};
  window.SUB_MEMORY = SUB_MEMORY;

  var HASH_SYNCING = false;

  /* 模型转换页：设计稿（convert-page-live.html）里没有列表 —— 进来要么是在跑的那张卡，
     要么什么都没有（从模型卡片点「转换」时，那边已经设好 __cvPick 并由 start() 直接接管）。 */
  /* ---- 转换：直接调接口 + 自己轮询。不依赖文件末尾那个 IIFE 里的 __cvStart，
         它历史上挂没挂上不稳，导致点「转换」只弹一句「转换器还没就绪」。 ---- */
  var CV_TIMER = null;
  function cvApi(path, opts) {
    return fetch(path, opts).then(function (r) {
      if (!r.ok) {
        return r.text().then(function (t) {
          throw new Error("HTTP " + r.status + (t ? " · " + t.slice(0, 300) : ""));
        });
      }
      return r.json();
    });
  }
  function cvPoll() {
    if (CV_TIMER) clearTimeout(CV_TIMER);
    CV_TIMER = setTimeout(function () {
      cvApi("/api/convert/status").then(function (st) {
        if (window.__cvPaint) window.__cvPaint(st);
        if (st && st.running) { cvPoll(); } else { CV_TIMER = null; }
      }).catch(function () { CV_TIMER = null; });
    }, 400);
  }
  function cvRun(name) {
    cvApi("/api/convert/start", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name: name })
    }).then(function () {
      if (window.toast) toast("转换已开始", name, "success");
      cvPoll();
    }).catch(function (e) {
      var msg = String((e && e.message) || e);
      if (window.toast) toast("转换启动失败", msg, "error");
      if (window.__cvPaint) window.__cvPaint({ step: "error", name: name, modules:
        [{ name: "启动转换", state: "bad", reason: msg }] });
    });
  }

  function cvListPaint() {
    var card = document.getElementById("cvCard");
    if (!card) return;

    /* 卡片常显 + 空闲态：直接操作 DOM，不依赖 window.__cvPaint
       （那个 export 在文件末尾的 IIFE 里，历史上挂没挂上不稳）。 */
    card.style.display = "";
    card.classList.remove("bad");
    var nm = document.getElementById("cvName");
    if (nm) nm.textContent = "未开始";
    var mt = document.getElementById("cvMeta");
    if (mt) mt.textContent = "";
    var bd = document.getElementById("cvBad");
    if (bd) bd.className = "cvbadge";
    ["cvS1", "cvS2", "cvS3"].forEach(function (sid, idx) {
      var el = document.getElementById(sid);
      if (!el) return;
      el.className = "cvstep";
      var ic = el.querySelector(".cvic");
      if (ic) ic.textContent = String(idx + 1);
    });
    ["cvRows1", "cvRows2"].forEach(function (rid) {
      var el = document.getElementById(rid);
      if (el) el.innerHTML = "";
    });

    var pick = window.__cvPick;
    if (pick) {
      window.__cvPick = "";
      if (nm) nm.textContent = pick;
      cvRun(pick);
      return;
    }

    fetch("/api/convert/status").then(function (r) { return r.json(); }).then(function (st) {
      if (!window.__cvPaint || !st) return;
      if (st.running || st.step === "error" || st.step === "done") window.__cvPaint(st);
    }).catch(function (e) {
      if (!window.__cvPaint) return;
      window.__cvPaint({ step: "error", name: "状态读取失败", modules:
                        [{ name: "/api/convert/status", state: "bad", reason: String((e && e.message) || e) }] });
    });
  }


  function go(key) {
    var id = PAGE_IDS[key];
    if (!id) return;
    if (window.__invalidateShotCache) window.__invalidateShotCache();
    document.querySelectorAll(".page").forEach(function (p) { p.classList.remove("on"); });
    var el = document.getElementById(id);
    if (el) {
      el.classList.add("on");
      if (!reduced()) {
        el.animate(
          [{ opacity: 0, transform: "translateY(6px)" }, { opacity: 1, transform: "none" }],
          { duration: 220, easing: PAGE_EASE }
        );
      }
    }
    /* 地址栏同步页面 key：#train-cfg 这种。好处有两个：
       刷新/重启能回到同一页；外部（比如自动化截图）可以直接打开某一页。 */
    try {
      if (location.hash.slice(1) !== key) {
        history.replaceState(null, "", "#" + key);
        HASH_SYNCING = true;
        setTimeout(function () { HASH_SYNCING = false; }, 0);
      }
    } catch (e) {}
    LAST_PAGE[groupOf(key)] = key;
    if (SUB_PARENT[key]) SUB_MEMORY[SUB_PARENT[key]] = key;
    var active = NAV_ALIAS[key] || key;
    document.querySelectorAll(".nav:not(.sub)").forEach(function (n) {
      n.classList.toggle("on", n.getAttribute("data-go") === active);
    });
    renderCrumbs(key);
    /* 「模块加载失败」横幅是**训练页的状态**：离开就收起，重新进来按最新状态再画一次
       （所以点 × 只是本次挂载内关掉，不是永久记住 —— 见 js/loadbanner.js）。 */
    if (window.LCSLoadBanner) {
      if (key === "run") window.LCSLoadBanner.enter();
      else window.LCSLoadBanner.leave();
    }
    if (key === "debug") { edInit(); edBind(); edDraw(); }
    if (key === "dataset") { renderSteps(); syncDatasetBadge(); erLoad(); }
    if (key === "train") bindTrainerTop();
    /* 模型转换页：进来就把列表拉一次。
       不依赖任何分片末尾的初始化（那种写法历史上被吞过好几次），
       也不依赖 window.__cvBoot —— 这里自己拉、自己画、自己绑按钮。 */
    if (key === "train-convert") cvListPaint();
    if (key === "settings" && window.WPK) window.WPK.enter();
    if (key === "train-cfg") {
      if (window.markHotRows) window.markHotRows();
      var cb = document.getElementById("cfgBox");
  setTimeout(function(){ if (window.paintVram) window.paintVram(); }, 0);
      if (cb && !cb.children.length) {
        var cm = window.__cfgModel || (MODELS && MODELS[0]) || null;
        if (cm) openCfg(cm);
      }
    }
    if (key === "tool-frames" && window.framesBind) window.framesBind();
    if (key === "tool-frames-run" && window.framesRunEnter) window.framesRunEnter();
    if (key === "tool-mask") {
      var mk = document.getElementById("mkFrame");
      if (mk && !mk.dataset.themeBound) {
        mk.dataset.themeBound = "1";
        mk.addEventListener("load", function () {
          try {
            var th = document.documentElement.getAttribute("data-theme") || "light";
            mk.contentWindow.postMessage({ dflTheme: th }, "*");
          } catch (e) {}
        });
      }
      if (mk && mk.src.indexOf("dir=") < 0) {
        var ws = (STATE && STATE.runtime && STATE.runtime.workspace) || "";
        var dir = ws ? (ws + "\\data_src\\aligned") : "workspace\\data_src\\aligned";
        mk.src = "/mask-editor/index.html?dir=" + encodeURIComponent(dir);
      }
      /* 权重分包门控：进页按 /api/weights/status 决定 SAM 那三块能不能用 */
      if (window.WKG && window.WKG.enter) window.WKG.enter();
      document.querySelector(".main").classList.add("no-scroll");
    }
    if (key === "merge") {
      bindMergeUI();
      if (window.mgLoadPrefs) window.mgLoadPrefs();
      /* mgLoadPrefs 会把「模式」按上次的偏好写回下拉 —— 写完了再判一次门控，
         否则「上次是 seamless、这次记忆里 gpu」这一路会漏过去。 */
      if (window.mgGateCmpBackend) window.mgGateCmpBackend();
      var right = document.querySelector("#p-merge .mgright");
      if (right && !right.dataset.prefBound) {
        right.dataset.prefBound = "1";
        right.addEventListener("change", function () { if (window.mgSavePrefs) window.mgSavePrefs(); });
        right.addEventListener("click", function (e) {
          if (e.target.closest && e.target.closest(".seg b")) {
            setTimeout(function () { if (window.mgSavePrefs) window.mgSavePrefs(); }, 0);
          }
        });
      }
      if (window.mgStep) window.mgStep(window.__mergeStep || "main");
      document.querySelector(".main").classList.add("no-scroll");
    } else if (key !== "tool-mask") {
      var mm = document.querySelector(".main");
      if (mm) mm.classList.remove("no-scroll");
    }
    noSpell();
    bindDropPaths();
    /* 路径框右侧那颗文件夹图标：以前只有数据处理页的 renderSteps 里绑过一次，
       提取器 / 工具页进页面时压根没绑 —— 点图标什么都不发生。
       每次切页补一次幂等绑定（box.dataset.pick 守卫），点下去走全局 pickFolderFor。 */
    bindPickPaths();
    /* 同步工具子项高亮 */
    document.querySelectorAll("#toolFly .nav.sub").forEach(function (n) {
      n.classList.toggle("on", n.getAttribute("data-go") === key);
    });
    document.querySelectorAll("#toolgrpBody .nav.sub").forEach(function (n) {
      n.classList.toggle("on", n.getAttribute("data-go") === key);
    });
    var main = document.querySelector(".main");
    if (main) main.scrollTop = 0;
  }
  document.querySelectorAll(".nav[data-go]").forEach(function (n) {
    n.addEventListener("click", function () {
      var target = n.getAttribute("data-go");
      /* 有子页记忆就回子页（B 站下载 → 下载中 / 训练器 → 训练界面） */
      var sub = SUB_MEMORY[target];
      if (sub && PAGE_IDS[sub]) { go(sub); return; }
      var last = LAST_PAGE[target];
      /* 只恢复"本次会话内真的打开过、且内容还在"的页面 */
      if (last && last !== target && PAGE_IDS[last]) {
        go(last);
      } else {
        go(target);
      }
    });
  });
  window.go = go;

  /* 文档就绪后再兜一次：刷新/直达 #train-convert 时不会经过 go() */
  document.addEventListener("DOMContentLoaded", function () {
    setTimeout(function () {
      if (location.hash.indexOf("train-convert") >= 0 && window.__cvBoot) {
        try { window.__cvBoot(); } catch (e) {}
      }
    }, 200);
  });

  /* ---------- 主题模式 ----------
     auto  = 跟随系统（默认，未登录时强制）
     light / dark = 用户在设置里显式选定
     存储键只用 dfl-theme-mode；dfl-theme 仍写入解析后的实际主题，给桌面壳用。 */
  var THEME = { mode: "auto", saved: "auto" };

  function systemTheme() {
    try {
      return (window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches) ? "dark" : "light";
    } catch (e) { return "light"; }
  }

  function resolveTheme() {
    return THEME.mode === "auto" ? systemTheme() : THEME.mode;
  }

  function syncThemeSeg() {
    var seg = document.getElementById("themeSeg");
    if (!seg) return;
    Array.prototype.forEach.call(seg.querySelectorAll("b"), function (b) {
      b.classList.toggle("on", b.getAttribute("data-mode") === THEME.mode);
    });
  }

  /* 设置里的"外观"改的是用户偏好，选完立即生效。
     默认值是 auto（跟随系统）—— 只要用户没手动选过，未登录也一直是跟随系统；
     但一旦手动选了，就按他选的来，不该因为没登录而"点了没反应"。 */
  function setThemeMode(mode, opts) {
    if (mode !== "light" && mode !== "dark") mode = "auto";
    THEME.saved = mode;
    try { window.localStorage.setItem("dfl-theme-mode", mode); } catch (e) {}
    THEME.mode = mode;
    applyTheme(resolveTheme(), opts);
    syncThemeSeg();
  }
  window.setThemeMode = setThemeMode;

  function applyTheme(t, opts) {
    var root = document.documentElement;
    var instant = opts && opts.instant;
    /* 圆形扩散（toggleTheme）自己就是完整的过渡：快照压在上面按 mask 揭开，
       底下的新主题必须"瞬间"就位。这里再叠 0.24s 的 CSS 过渡，
       中间帧就会出现"卡片已经白了、字还是旧的浅灰"这种错位。
       所以只有不走扩散的调用路径（比如启动时恢复主题）才挂 .theme-fade */
    if (!instant && root.getAttribute("data-theme") !== t) {
      root.classList.add("theme-fade");
      clearTimeout(window.__themeFadeT);
      window.__themeFadeT = setTimeout(function () { root.classList.remove("theme-fade"); }, 300);
    }
    root.setAttribute("data-theme", t);
    /* 遮罩编辑器是 iframe，主题要单独推一份 */
    var _mk = document.getElementById("mkFrame");
    if (_mk) { try { _mk.contentWindow.postMessage({ dflTheme: t }, "*"); } catch (e) {} }
    var label = document.getElementById("themeLabel");
    if (label) label.textContent = (t === "dark") ? "浅色模式" : "深色模式";
    try { window.localStorage.setItem("dfl-theme", t); } catch (e) {}
    var br = window.desktop || window.pywebview && window.pywebview.api;
    if (br && br.setTheme) { try { br.setTheme(t); } catch (e) {} }
    if (window.I18N) window.I18N.refresh();
  }

  window.applyTheme = applyTheme;

  /* 统一的 View Transition 入口：新的一次会主动跳过上一次，
     这样连续点击时每次都能从原点重新扩散一圈，而不是排队或吞掉点击 */
  function beginVT(mutate) {
    /* 不要手动 skipTransition —— 手动跳过之后立刻再 startViewTransition，
       浏览器会拒绝这一次调用，于是第二下点击就退化成淡入淡出。
       交给浏览器自己处理：新的过渡出现时它会自动跳过旧的。 */
    var vt;
    try {
      vt = document.startViewTransition(mutate);
    } catch (e) {
      return null;
    }
    try { vt.ready.catch(function () {}); } catch (e) {}
    try { vt.finished.catch(function () {}); } catch (e) {}
    return vt;
  }

  /* 兜底：不支持 View Transition 时用整体淡出淡入，保证不是硬切 */
  function fadeSwap(mutate) {
    var root = document.querySelector(".app");
    if (!root || reduced()) { mutate(); return; }
    var out = root.animate([{ opacity: 1 }, { opacity: 0.45 }], { duration: 110, easing: "ease-out" });
    out.onfinish = function () {
      mutate();
      root.animate([{ opacity: 0.45 }, { opacity: 1 }], { duration: 180, easing: "ease-out" });
    };
  }

  /* 文字类切换（语言）：整页交叉淡入淡出。背景色不变，圆形扩散看不出来 */
  function crossFadeSwap(mutate, duration) {
    var dur = duration || 320;
    if (!document.startViewTransition || reduced()) { fadeSwap(mutate); return; }
    var vt = beginVT(mutate);
    if (!vt) { fadeSwap(mutate); return; }
    vt.ready.then(function () {
      document.documentElement.animate(
        [{ opacity: 1 }, { opacity: 0 }],
        { duration: Math.round(dur * 0.6), easing: "ease-out",
          pseudoElement: "::view-transition-old(root)" }
      );
      document.documentElement.animate(
        [{ opacity: 0 }, { opacity: 1 }],
        { duration: dur, easing: "cubic-bezier(0.25, 0.46, 0.45, 0.94)",
          pseudoElement: "::view-transition-new(root)" }
      );
    });
  }

  /* 通用：以某个元素为圆心，圆形扩散到整屏；缓动是二次方缓出 */
  function circleReveal(originEl, mutate, duration, offsetX) {
    var r = originEl ? originEl.getBoundingClientRect() : null;
    var cx, cy;
    if (r && (r.width || r.height)) {
      cx = r.left + (offsetX == null ? r.width / 2 : offsetX);
      cy = r.top + r.height / 2;
    } else {
      cx = 40; cy = innerHeight - 60;
    }
    if (!document.startViewTransition || reduced()) { fadeSwap(mutate); return; }
    var far = Math.hypot(Math.max(cx, innerWidth - cx), Math.max(cy, innerHeight - cy));
    var vt = beginVT(mutate);
    if (!vt) { fadeSwap(mutate); return; }
    vt.ready.then(function () {
      document.documentElement.animate(
        {
          clipPath: [
            "circle(0px at " + cx + "px " + cy + "px)",
            "circle(" + far + "px at " + cx + "px " + cy + "px)"
          ]
        },
        {
          duration: duration || 520,
          easing: "cubic-bezier(0.25, 0.46, 0.45, 0.94)",
          pseudoElement: "::view-transition-new(root)"
        }
      );
    });
  }
  window.circleReveal = circleReveal;

  /* 一圈扩散：以旧主题的底色铺开再淡出。可以同时存在多圈，
     每次点击都独立生成一层，不再依赖 View Transition（它同一时刻只允许一层） */
  function spawnRipple(cx, cy, color) {
    var layer = document.getElementById("rippleLayer");
    if (!layer) {
      layer = document.createElement("div");
      layer.id = "rippleLayer";
      document.body.appendChild(layer);
    }
    var el = document.createElement("div");
    el.className = "ripple";
    var far = Math.hypot(Math.max(cx, innerWidth - cx), Math.max(cy, innerHeight - cy));
    el.style.left = cx + "px";
    el.style.top = cy + "px";
    el.style.width = el.style.height = (far * 2) + "px";
    el.style.borderColor = color;
    layer.appendChild(el);
    var anim = el.animate(
      [
        { transform: "translate(-50%, -50%) scale(0)", opacity: 0.85, borderWidth: "2px" },
        { transform: "translate(-50%, -50%) scale(1)", opacity: 0, borderWidth: "1px" }
      ],
      { duration: 640, easing: "cubic-bezier(0.25, 0.46, 0.45, 0.94)", fill: "forwards" }
    );
    anim.onfinish = function () { el.remove(); try { layoutToasts(); } catch (e) {} };
    return el;
  }
  window.spawnRipple = spawnRipple;

  /* 揭幕：每次点击生成一层「点击前主题」的整页快照，挖一个扩散的圆。
     关键：新层插到已有层的【下面】—— 老层在最上面，它的边界永远不会被盖掉。
     这样不需要冻结/烘培，也就不存在"点太快把画面卡成一整块遮罩"的问题。 */
  var SHOT_TOKEN = 0;
  var SHOT_CACHE = {};          /* { dark: dataURL, light: dataURL } */

  /* 界面内容变了（切页面、数据刷新），缓存的底图就作废 */
  window.__invalidateShotCache = function () { SHOT_CACHE = {}; };

  function toggleTheme() {
    var btn = document.getElementById("themeToggle");
    var r = btn ? btn.getBoundingClientRect() : { left: 40, top: innerHeight - 60, height: 0 };
    var cx = r.left + 15;
    var cy = r.top + r.height / 2;

    var prev = document.documentElement.getAttribute("data-theme") || "light";
    var next = prev === "dark" ? "light" : "dark";
    /* 手动切换 = 显式偏好，立即生效，下次启动也按这个来 */
    THEME.saved = next; THEME.mode = next;
    try { window.localStorage.setItem("dfl-theme-mode", next); } catch (e) {}
    syncThemeSeg();

    if (reduced()) { applyTheme(next); return; }

    /* 圆形揭幕默认开。桌面壳用整页位图（capturePage），浏览器没有这能力，
       退回 View Transition 的 clip-path 圆形（同样是以点击处为圆心扩散），
       所以两边看到的是同一个效果，只是实现不同。
       关掉：localStorage.setItem('dfl-theme-wipe','0')。 */
    var wantWipe = true;
    try { if (window.localStorage.getItem("dfl-theme-wipe") === "0") wantWipe = false; } catch (e) {}
    var shot = (wantWipe && window.desktop && window.desktop.snapshot) ? window.desktop.snapshot : null;
    if (!shot) {
      /* 浏览器：没有整页位图可裁，但 View Transition 能从点击点裁一个圆出来，
         效果和桌面壳那套一致（同一个缓动、同样 520ms）。 */
      if (wantWipe && window.circleReveal) {
        circleReveal(btn, function () { applyTheme(next, { instant: true }); }, 520, 15);
        return;
      }
      applyTheme(next);
      return;
    }

    var root = document.documentElement;
    var my = ++SHOT_TOKEN;

    function snapSwap() {
      root.classList.add("theme-snap");
      applyTheme(next, { instant: true });
      void root.offsetHeight;
      requestAnimationFrame(function () { root.classList.remove("theme-snap"); });
    }

    /* 扩散结束后顺手把这个主题的底图备好 —— 那时屏幕上没有任何 vt-shot，
       拍出来的是干净的界面。下次连点就能直接命中缓存，一次截图都不用等。 */
    function prefetch(theme) {
      if (SHOT_CACHE[theme]) return;
      if (document.querySelector(".vt-shot")) return;
      shot().then(function (u) { if (u && !SHOT_CACHE[theme]) SHOT_CACHE[theme] = u; })
            .catch(function () {});
    }

    function start(dataUrl) {
      if (my !== SHOT_TOKEN) return;
      if (!dataUrl) { applyTheme(next); return; }

      var layer = document.createElement("div");
      layer.className = "vt-shot";
      layer.style.backgroundImage = "url(" + dataUrl + ")";
      layer.style.setProperty("--cx", cx + "px");
      layer.style.setProperty("--cy", cy + "px");

      /* 新层插到已有层的下面：旧圈继续按自己的节奏扩，绝不打断 */
      var first = document.querySelector(".vt-shot");
      if (first) document.body.insertBefore(layer, first);
      else document.body.appendChild(layer);

      root.classList.add("theme-snap");
      applyTheme(next, { instant: true });
      void layer.offsetHeight;

      requestAnimationFrame(function () {
        requestAnimationFrame(function () {
          root.classList.remove("theme-snap");
          var far = Math.hypot(Math.max(cx, innerWidth - cx), Math.max(cy, innerHeight - cy));
          var anim = layer.animate(
            [{ "--wipe": "0px" }, { "--wipe": far + "px" }],
            { duration: 520, easing: "cubic-bezier(0.25, 0.46, 0.45, 0.94)", fill: "forwards" }
          );
          anim.onfinish = function () {
            layer.remove();
            prefetch(next);
          };
        });
      });
    }

    /* ① 缓存命中：同步启动，零延迟 */
    if (SHOT_CACHE[prev]) { start(SHOT_CACHE[prev]); return; }

    /* ② 屏幕上还有 vt-shot：这时截图会把那个圆焊进底图，不能拍。
          但也不能直接换色 —— 那样就没有圈了（用户会觉得"这次没动画"）。
          正确做法是等那一圈自己扩完（最多 500ms），拿到干净画面再拍。 */
    function shootWhenClean() {
      var waited = 0;
      (function wait() {
        if (my !== SHOT_TOKEN) return;
        if (document.querySelector(".vt-shot")) {
          if (waited >= 500) { snapSwap(); return; }   /* 兜底：别卡太久 */
          waited += 24;
          setTimeout(wait, 24);
          return;
        }
        requestAnimationFrame(function () {
          requestAnimationFrame(function () {
            if (my !== SHOT_TOKEN) return;
            shot().then(function (u) {
              if (my !== SHOT_TOKEN) return;
              if (u) SHOT_CACHE[prev] = u;
              start(u);
            }).catch(function () {
              if (my === SHOT_TOKEN) applyTheme(next);
            });
          });
        });
      })();
    }

    shootWhenClean();
  }

  window.toggleTheme = toggleTheme;

  /* View Transition 运行期间，点击的 target 不再是按钮本身（命中被过渡层接管），
     按钮上的监听收不到事件。这里在 document 捕获阶段按坐标兜底，
     保证过渡进行中再点也能触发一次新的扩散。 */
  document.addEventListener("click", function (e) {
    var btn = document.getElementById("themeToggle");
    if (!btn) return;
    if (btn === e.target || btn.contains(e.target)) return;   // 正常路径已处理
    if (!e.clientX && !e.clientY) return;                     // 程序化 click
    var r = btn.getBoundingClientRect();
    if (e.clientX >= r.left && e.clientX <= r.right && e.clientY >= r.top && e.clientY <= r.bottom) {
      toggleTheme();
    }
  }, true);

  /* 侧边栏折叠 */
  var APPR = document.querySelector(".app");
  var sideBtn = document.getElementById("brandLogo");
  function setCollapsed(on) {
    if (!APPR) return;
    APPR.classList.toggle("collapsed", !!on);
    if (sideBtn) sideBtn.title = on ? "展开侧边栏" : "折叠侧边栏";
    try { window.localStorage.setItem("dfl-side", on ? "1" : "0"); } catch (e) {}
    /* 工具分组：展开侧边栏后重新量高度（折叠时子项是 display:none，量出来是 0） */
    if (!on) {
      setTimeout(function () {
        var b = document.getElementById("toolgrpBody");
        var g = document.getElementById("toolgrp");
        if (b && g && !g.classList.contains("closed")) b.style.height = b.scrollHeight + "px";
      }, 280);
    }
    if (window.I18N) window.I18N.refresh();
  }
  window.setCollapsed = setCollapsed;
  if (sideBtn) {
    sideBtn.addEventListener("click", function () {
      setCollapsed(!APPR.classList.contains("collapsed"));
    });
  }
  try {
    if (window.localStorage.getItem("dfl-side") === "1") setCollapsed(true);
  } catch (e) {}
  var themeBtn = document.getElementById("themeToggle");
  if (themeBtn) themeBtn.addEventListener("click", toggleTheme);

  /* ---------------- 语言 ---------------- */
  function syncLangSeg(lang) {
    document.querySelectorAll("#langseg > b").forEach(function (b) {
      b.classList.toggle("on", b.getAttribute("data-lang") === lang);
    });
  }
  document.querySelectorAll("#langseg > b").forEach(function (b) {
    b.addEventListener("click", function () {
      if (!window.I18N) return;
      var target = b.getAttribute("data-lang");
      if (target === window.I18N.get()) return;
      crossFadeSwap(function () {
        syncLangSeg(window.I18N.apply(target));
      }, 320);
    });
  });
  if (window.I18N) syncLangSeg(window.I18N.get());
  try {
    var _m = window.localStorage.getItem("dfl-theme-mode");
    THEME.saved = (_m === "light" || _m === "dark") ? _m : "auto";
  } catch (e) { THEME.saved = "auto"; }
  /* 启动时还不知道登录状态 —— 一律先跟随系统；登录后再套用用户的显式选择 */
  THEME.mode = "auto";
  applyTheme(resolveTheme(), { instant: true });
  syncThemeSeg();

  /* 系统主题变了：只有 auto 模式跟着变 */
  try {
    var _mq = window.matchMedia("(prefers-color-scheme: dark)");
    var _onSys = function () { if (THEME.mode === "auto") applyTheme(resolveTheme()); };
    if (_mq.addEventListener) _mq.addEventListener("change", _onSys);
    else if (_mq.addListener) _mq.addListener(_onSys);
  } catch (e) {}

  /* 设置里的三档：跟随系统 / 浅色 / 深色 */
  document.addEventListener("click", function (e) {
    var b = e.target && e.target.closest ? e.target.closest("#themeSeg b") : null;
    if (!b) return;
    setThemeMode(b.getAttribute("data-mode"));
  });

  document.querySelectorAll(".seg").forEach(function (seg) {
    var kids = seg.querySelectorAll(":scope > b");
    Array.prototype.forEach.call(kids, function (b) {
      b.addEventListener("click", function () {
        Array.prototype.forEach.call(kids, function (x) { x.classList.remove("on"); });
        b.classList.add("on");
      });
    });
  });

  var TOAST_META = {
    success: { icon: "#i-check", color: "var(--ok)" },
    info: { icon: "#i-info", color: "var(--accent)" },
    warn: { icon: "#i-warn", color: "var(--warn)" },
    error: { icon: "#i-error", color: "var(--bad)" }
  };

  /* 退场：淡出 + 轻微下移 + 高度塌陷，避让的兄弟节点平滑上移 */
  function dismissToast(el) {
    if (!el || el.__leaving) return;
    el.__leaving = true;
    try { el.setAttribute("data-leaving", "1"); } catch (e) {}
    var cs = getComputedStyle(el);
    var h = el.getBoundingClientRect().height;
    var pt = parseFloat(cs.paddingTop) || 0;
    var pb = parseFloat(cs.paddingBottom) || 0;
    var gap = 8;   /* #toastroot 的 gap，塌陷时用负边距抵掉 */

    el.style.pointerEvents = "none";
    var anim = el.animate(
      [
        { opacity: 1, transform: "none", height: h + "px",
          paddingTop: pt + "px", paddingBottom: pb + "px", marginBottom: "0px" },
        { opacity: 0, transform: "translateY(6px) scale(.98)", height: "0px",
          paddingTop: "0px", paddingBottom: "0px", marginBottom: (-gap) + "px" }
      ],
      { duration: 200, easing: "cubic-bezier(.25,.46,.45,.94)", fill: "forwards" }
    );
    el.style.overflow = "hidden";
    el.style.minHeight = "0";
    anim.onfinish = function () { el.remove(); try { layoutToasts(); } catch (e2) {} };
  }
  window.dismissToast = dismissToast;

  /* ---- 3D 堆叠布局 ----
     toast 高度不一，位移只能在运行时算：
       收起：最新一条在底，其余每条向上错开 14px、缩小 5%、变淡 22%
       悬停：按真实高度累加，展开成完整列表（间距 8px） */
  var TOAST_STACK_STEP = 14;
  var TOAST_STACK_GAP = 8;
  function layoutToasts() {
    var root = $("toastroot");
    if (!root) return;
    var items = [].slice.call(root.querySelectorAll(".toast:not([data-leaving])"));
    var hover = root.matches(":hover");
    // DOM 顺序是「旧 → 新」，从最新往回排
    var offset = 0;
    for (var k = items.length - 1, i = 0; k >= 0; k--, i++) {
      var el = items[k];
      var h = el.offsetHeight || 56;
      el.style.zIndex = String(100 - i);
      if (hover) {
        el.style.bottom = offset + "px";
        el.style.transform = "none";
        el.style.opacity = "1";
        offset += h + TOAST_STACK_GAP;
      } else {
        el.style.bottom = (i * TOAST_STACK_STEP) + "px";
        el.style.transform = "scale(" + (1 - i * 0.05).toFixed(3) + ")";
        el.style.opacity = String(Math.max(0.35, 1 - i * 0.22));
      }
    }
  }
  window.layoutToasts = layoutToasts;

  function toast(title, sub, type, ms) {
    var root = $("toastroot");
    if (!root) return null;
    var meta = TOAST_META[type] || TOAST_META.success;
    /* error 的图标颜色直接给死：走 var(--bad) 在这套 sprite 上解析过一次绿的，很怪 */
    var _tint = (type === "error") ? "#ff6467" : meta.color;
    var life = (typeof ms === "number" && ms > 400) ? ms : 3600;
    /* 报错不自动消失、也不画底部倒计时条：用户通常只会截个图，
       倒计时一到就没了等于把唯一的线索丢掉。要关自己点 ×。 */
    var sticky = (type === "error");

    var el = document.createElement("div");
    el.className = "toast";
    el.setAttribute("data-type", type || "success");
    el.innerHTML =
      '<svg class="ic ti" style="stroke:' + _tint + ';opacity:1"><use href="' + meta.icon + '"/></svg>' +
      '<div><div class="tt">' + title + '</div><div class="ts">' + (sub || "") + '</div></div>' +
      '<button class="tx">×</button>' + (sticky ? "" : '<div class="tb"><i></i></div>');
    el.querySelector(".tx").addEventListener("click", function () { dismissToast(el); });
    var _tb = el.querySelector(".tb i");
    if (_tb) _tb.style.animationDuration = life + "ms";
    root.appendChild(el);
    /* shadcn 的 toast 有 TOAST_LIMIT（默认只留最新几条），不会无限堆 */
    var alive = root.querySelectorAll(".toast:not([data-leaving])");
    if (alive.length > 3) {
      for (var _i = 0; _i < alive.length - 3; _i++) dismissToast(alive[_i]);
    }
    // 先按收起态排一次，随后 measure 完再排一次（新节点刚插入时 offsetHeight 可能为 0）
    layoutToasts();
    requestAnimationFrame(layoutToasts);
    if (window.I18N) window.I18N.refresh();
    if (!sticky) setTimeout(function () { dismissToast(el); }, life);
    return el;
  }
  window.toast = toast;

  (function () {
    var root = $("toastroot");
    if (!root) return;
    root.addEventListener("mouseenter", layoutToasts);
    root.addEventListener("mouseleave", layoutToasts);
  })();

  var MODULES = [
    { n: "过滤器", d: "模糊 · 身份 · 重复 · 错脸 · 遮挡 · 合并",
      modes: ["模糊", "人脸身份", "重复图", "错脸", "合并子文件夹", "遮挡率（排序）"],
      modeDesc: ["高频/低频能量比 E(b0)/E(b2) 判清晰度，按分位淘汰最差一批（顺带清掉打不开的文件）",
                 "ArcFace 特征向量聚类，同一组≈同一个人（顺带清掉打不开的文件）",
                 "感知哈希(phash)检测高度相似，剔除重复帧（顺带清掉打不开的文件）",
      "68 点几何分类判错检（需要数据集用 2D106 真点重提过）",
                 "把所有子目录（含多层）里的图片搬回根目录，清掉空出来的文件夹；跳过 _trash 和隐藏目录",
                 "68 点凸包（不含眉毛）内 XSeg 遮罩的软面积占比：100% = 整张脸被挡住，0% = 一点没挡。只排序，移走要你确认"],
      out: "" },
    { n: "排序器", d: "清晰度 · 角度 · 时间",
      modes: ["清晰度", "角度", "原序", "感知哈希", "直方图"],
      modeDesc: ["越清晰排越前（blur）", "按偏航角从小到大（pose/yaw）",
                 "恢复原始文件顺序（reset）", "按感知哈希相近度排序（phash）",
                 "按直方图相似度排序（histogram）"],
      out: "" },
    { n: "应用遮罩", d: "XSeg 推理",
      modes: ["XSeg 推理"],
      modeDesc: ["用训练好的遮罩模型推理"],
      out: "" }
  ];

  /* ---------- 错脸检测（errface）能力门控 ----------
     缺 errface_xgb.ubj 属于「权重没装」，不是「功能坏了」。后端 /api/errface/info
     回 {available, reason, install}（中英各一份），这里据此把「错脸」选项置灰 +
     title 说明，点上去只解释「未安装 + 怎么装」，不再打到 /api/dataset/scan 吃 409。 */
  var ERRFACE = { known: false, ok: true, reason: "", reason_en: "", install: "", install_en: "", model: "" };
  var ERRFACE_IDX = 3;            /* MODULES[0].modes 里「错脸」的下标（改数组记得同步） */

  /* 后端来的动态文案也要进词典：切英文时这句不能留中文 */
  function erI18n(zh, en) {
    try {
      var d = window.I18N && window.I18N.dict;
      if (!d || !zh) return;
      if (d[zh] === undefined) d[zh] = en || zh;
    } catch (e) {}
  }

  function erOff() { return ERRFACE.known && !ERRFACE.ok; }

  function erHint(st) {
    var s = st || ERRFACE;
    return [s.reason, s.install].filter(Boolean).join(" · ");
  }
  function erHintEn(st) {
    var s = st || ERRFACE;
    return [s.reason_en, s.install_en].filter(Boolean).join(" · ");
  }

  function erApply(d) {
    if (!d || typeof d !== "object" || typeof d.available !== "boolean") {
      ERRFACE.known = false;                  /* 读不到就当"未知"：不置灰、不挡路 */
      return ERRFACE;
    }
    ERRFACE.known = true;
    ERRFACE.ok = d.available === true;
    ERRFACE.reason = String(d.reason || "");
    ERRFACE.reason_en = String(d.reason_en || "");
    ERRFACE.install = String(d.install || "");
    ERRFACE.install_en = String(d.install_en || "");
    ERRFACE.model = String(d.model || "");
    erI18n(erHint(), erHintEn());
    return ERRFACE;
  }

  /* 点「错脸」但没装：把「缺什么 / 多大 / 怎么装」说清楚，这不是报错 */
  function erExplain(d, btn) {
    if (d) erApply(d);
    var zh = erHint(d) || "缺 errface_xgb.ubj";
    erI18n(zh, erHintEn(d));
    if (btn) { btn.textContent = "开始处理"; btn.disabled = false; }
    if (typeof dsRunPaint === "function") {
      dsRunPaint({ pct: 0, done: 0, total: 0, speed: 0, elapsed: 0,
                   msg: "错脸检测未安装 · 见右下角提示", running: false });
    }
    toast("错脸检测未安装", zh, "warn", 12000);
    erLoad();                     /* 顺手刷一次：用户可能刚把文件拷进去 */
  }
  window.erfaceExplain = erExplain;

  function erLoad() {
    return fetch("/api/errface/info")
      .then(function (r) { return r.ok ? r.json() : null; })
      .then(function (d) {
        var before = ERRFACE.known + "|" + ERRFACE.ok;
        erApply(d);
        if ((ERRFACE.known + "|" + ERRFACE.ok) !== before) {
          try { if (typeof renderSteps === "function" && $("dsSteps")) renderSteps(); } catch (e) {}
        }
        return ERRFACE;
      })
      .catch(function () { ERRFACE.known = false; return ERRFACE; });
  }
  window.__erfaceReload = erLoad;
  window.ERRFACE_CAP = ERRFACE;   /* 自动化/排查用：页面里直接看能力状态 */
  window.__erfaceOff = erOff;

  /* 三段流水线同时展开：模块 → 模式 → 参数，各自独立成节 */
  var STEP_SEL = [0, 0, 0];      /* 每节选中的模式 */
  var PARAM_SEL = {};           /* "模块-模式-参数序号" → 选中下标 */
  var PARAM_VAL = {};           /* 同上 → 数字输入的值 */

  /* 每个模块 + 模式的参数组（"模块-模式" 索引） */
  var PARAM_MAP = {
    "0-0": [],   /* blur：能量比 + 分位，无可调参数 */

    /* faceid / repeated —— 默认值照抄 4.0 Qt GUI（位置聚类已移除，4.0 时期几乎没人用） */
    "0-1": [["邻域半径", { num: true, val: 0.40, min: 0.05, max: 1, step: 0.01 }],
            ["最小样本", { num: true, val: 2, min: 2, max: 50, step: 1 }]],
    "0-2": [["相似度阈值", { num: true, val: 0.98, min: 0.5, max: 1, step: 0.01 }]],
    "0-3": [["判定阈值", { num: true, val: 0.5, min: 0.1, max: 0.99, step: 0.01 }],
            ["最差 %（0=按阈值）", { num: true, val: 0, min: 0, max: 50, step: 0.5 }]],

    "1-0": [],   /* 排序没有分辨率参数 */
    "1-1": [],
    "1-2": [],
    "2-0": [["遮罩类型", ["XSegLite", "XSeg"]], ["遮罩尺寸", ["256", "512"]]],
  };

  /* 用训练参数页那套：分组卡片 + 行（标签 / 说明 / 右侧控件） */
  function segHTML(groups) {
    return groups.map(function (g) { return g; });
  }

  /* options 超过 3 个时改用下拉：分段控件排到 4 项以上会挤成一排小按钮，
     一眼看不清也容易点错。只有数据集页的「模式」走这条路 ——
     配置页的保存逻辑是按 .seg b.on 读取的，不能全局替换。 */
  /* meta[i] = {label, title, onBlock} —— 把某一项标成「未安装」：
     原生 option 置灰 + title，强化下拉框读 sl.__off 决定点了要不要拦。 */
  function makeSelect(options, sel, onPick, meta) {
    var sl = document.createElement("select");
    sl.className = "mgsel";
    options.forEach(function (o, i) {
      var op = document.createElement("option");
      var mm = meta && meta[i];
      op.value = String(i);
      op.textContent = (mm && mm.label) || o;
      if (mm) {
        op.disabled = true;                 /* 原生下拉里也点不动 */
        op.title = mm.title || "";
        sl.__off = sl.__off || {};
        sl.__off[i] = mm;
      }
      if (i === sel) op.selected = true;
      sl.appendChild(op);
    });
    sl.addEventListener("change", function () {
      if (onPick) onPick(Number(sl.value) || 0);
    });
    /* 和合成器/参数页同一个下拉组件：enhanceSelects 会把它包成
       "按钮 + 弹层"（.dd），原生 select 只留作数据源。 */
    var host = document.createElement("div");
    host.className = "segbx";
    host.appendChild(sl);
    try { if (window.__enhanceSelects) window.__enhanceSelects(host); } catch (e) {}
    return host;
  }

  function makeSeg(options, sel, onPick) {
    var seg = document.createElement("div");
    seg.className = "seg";
    options.forEach(function (o, i) {
      var b = document.createElement("b");
      b.textContent = o;
      if (i === sel) b.className = "on";
      b.addEventListener("click", function () {
        Array.prototype.forEach.call(seg.children, function (x) { x.classList.remove("on"); });
        b.classList.add("on");
        if (onPick) onPick(i);
      });
      seg.appendChild(b);
    });
    return seg;
  }

  function renderStation(i) {
    var m = MODULES[i];
    var k = STEP_SEL[i];
    var card = document.createElement("div");
    card.className = "card";
    card.dataset.station = i;

    var hd = document.createElement("div");
    hd.className = "cardhd";
    var hintText = i === 0 ? "未扫描" : (m.out || "");
    hd.innerHTML = "<b>" + m.n + "</b><span class=\"hint\">" + hintText + "</span>";
    card.appendChild(hd);

    /* 行 1：模式选择 —— 只有一个模式时这一行是废话，不渲染 */
    if (m.modes.length > 1) {
    var r1 = document.createElement("div");
    r1.className = "cfgrow";
    r1.innerHTML = '<span class="cl">模式</span><span class="cd">' + (m.modeDesc[k] || "") + '</span><span class="cc"></span>';
    /* 错脸检测没装：这一项置灰 + title 说明，点上去只弹「怎么装」 */
    var meta = null;
    if (i === 0 && erOff()) {
      meta = {};
      meta[ERRFACE_IDX] = {
        label: "错脸（未安装）",
        title: erHint(),
        onBlock: function () { erExplain(); }
      };
    }
    r1.querySelector(".cc").appendChild(
      (m.modes.length > 3 ? makeSelect : makeSeg)(m.modes, k, function (idx) {
      STEP_SEL[i] = idx;
      /* 换了模式：上一轮的排序网格留着只会误导（角标是旧数据），直接收起 */
      if (i === 0 && typeof window.occReset === "function") window.occReset();
      var fresh = renderStation(i);
      card.replaceWith(fresh);
      /* 只让「模式」下面的参数行弹一下：模式行本身不动，避免整卡片一起闪 */
      if (!reduced()) {
        var rows = Array.prototype.slice.call(fresh.querySelectorAll(".cfgrow")).slice(1);
        rows.forEach(function (r, k) {
          r.animate(
            [{ opacity: 0, transform: "translateY(6px)" },
             { opacity: 1, transform: "none" }],
            { duration: 170, delay: k * 25, easing: PAGE_EASE });
        });
      }
    }, meta));
    card.appendChild(r1);
    }

    /* 行 2..n：参数 */
    (PARAM_MAP[i + "-" + k] || PARAM_MAP[i + "-0"] || []).forEach(function (g, gi) {
      var row = document.createElement("div");
      row.className = "cfgrow";
      var key = i + "-" + k + "-" + gi;
      row.innerHTML = '<span class="cl">' + g[0] + '</span><span class="cd"></span><span class="cc"></span>';
      var cc = row.querySelector(".cc");
      if (g[1] && g[1].num) {           /* 数字输入：小数/整数都行 */
        var spec = g[1];
        var wrap = document.createElement("span");
        wrap.style.display = "inline-flex";
        wrap.style.alignItems = "center";
        wrap.style.gap = "6px";
        var inp = document.createElement("input");
        inp.type = "number";
        inp.className = "numin mono";
        inp.min = spec.min; inp.max = spec.max; inp.step = spec.step || 1;
        inp.value = (PARAM_VAL[key] !== undefined) ? PARAM_VAL[key] : spec.val;
        inp.style.width = "84px";
        inp.addEventListener("input", function () {
          var v = parseFloat(inp.value);
          PARAM_VAL[key] = isNaN(v) ? spec.val : v;
        });
        wrap.appendChild(inp);
        if (spec.unit) {
          var u = document.createElement("span");
          u.textContent = spec.unit;
          u.style.fontSize = "12px";
          u.style.color = "var(--ink-3)";
          wrap.appendChild(u);
        }
        cc.appendChild(wrap);
      } else {
        cc.appendChild(makeSeg(g[1], PARAM_SEL[key] || 0, function (idx) {
          PARAM_SEL[key] = idx;
        }));
      }
      card.appendChild(row);
    });

    /* 结果区改到二级界面（#p-dataset-run），这里不再内联 */

    /* 底部动作 */
    var acts = document.createElement("div");
    acts.className = "dsacts";
    acts.innerHTML = '<button class="btn"' + (i === 0 ? ' id="dsBtn"' : '') +
      ' data-step="' + i + '">开始处理</button>';
    acts.querySelector("button").addEventListener("click", function () {
      if (i === 0) { dsRun(); return; }
      if (i === 2) { dsMask(); return; }
      toast(MODULES[i].n, "这一节的后端还没接", "info");
    });
    card.appendChild(acts);

    return card;
  }

  var DS_TAB = 0;          /* 数据处理当前选中的工具 */

  function renderSteps() {
    var box = $("dsSteps");
    if (!box) return;
    box.innerHTML = "";

    /* 一张卡片 + 顶部左上的切换条（原来三个卡片并排，布局重复度太高） */
    var card = document.createElement("div");
    card.className = "card";

    var hd = document.createElement("div");
    hd.className = "cardhd";
    var seg = makeSeg(MODULES.map(function (m) { return m.n; }), DS_TAB, function (idx) {
      DS_TAB = idx;
      renderSteps();
    });
    hd.appendChild(seg);
    card.appendChild(hd);

    /* 直接把工具卡片嵌进来，只去掉它自己的卡片外壳 ——
       之前把子节点搬进容器，原有的 .card 内边距/布局丢了，行被挤成竖排 */
    var only = renderStation(DS_TAB);
    only.style.border = "0";
    only.style.padding = "0";
    only.style.margin = "0";
    only.style.background = "transparent";
    only.dataset.station = String(DS_TAB);
    if (!reduced()) {
      only.animate(
        [{ opacity: 0, transform: "translateY(4px)" }, { opacity: 1, transform: "none" }],
        { duration: 160, easing: PAGE_EASE });
    }
    /* 卡片头已经在顶部切换条里了，这里再显示一遍工具名是重复信息 */
    var hdInner = only.querySelector(".cardhd");
    if (hdInner) hdInner.style.display = "none";
    card.appendChild(only);

    box.appendChild(card);
    if (window.I18N) window.I18N.refresh();
    bindDropPaths();
    bindPickPaths();
    noSpell();
  }
  window.renderSteps = renderSteps;
  erLoad();                        /* 进页就知道「错脸」能不能用，别等点了才说 */

  /* 第 3 站：应用遮罩（XSeg 推理） */
  function dsMask() {
    var model = (PARAM_SEL["2-0-0"] === 1) ? "xseg" : "lite";
    var res = (PARAM_SEL["2-0-1"] === 1) ? 512 : 256;
    /* 直接读输入框的实际内容——用户手动输入时 DSB.dir 不会跟着变，这里不能信它 */
    var inp = document.getElementById("dsPathVal");
    var dir = String((inp && inp.value) || (window.DSB && DSB.dir) || "data_src").trim();
    /* 站点卡片上的按钮带 data-step（renderStation 里加的）。以前用的
       '#dsSteps .step:nth-child(3) .btn' 这个结构根本不存在，选择器恒为 null ——
       收尾分支（复位按钮 / 完成 toast）一次都跑不到，这就是「按钮没复位」的根因。 */
    var btn = dsStepBtn(2);
    if (btn) { btn.disabled = true; btn.textContent = "应用遮罩中…"; }
    toast("应用遮罩", dir + " · " + (model === "xseg" ? "XSeg" : "XSegLite") + " · " + res + "px · 8 线程", "info");
    dsRunShow("正在应用遮罩");
    window.__maskT0 = Date.now();
    window.__maskDone = false;          /* 新一轮：解开上一次的完成闩 */
    window.__maskFinal = null;
    fetch("/api/dataset/mask", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ dir: dir, workers: 8, model: model, res: res })
    }).then(function (r) {
      /* 后端崩成 500 时回的是纯文本 Internal Server Error，直接 r.json() 会抛
         "Unexpected token 'I'..." —— 那句话说不出任何有用的信息。先看状态码，
         把原文截一段带上，至少知道是哪一步挂了。 */
      if (!r.ok) {
        return r.text().then(function (t) {
          var head = String(t || "").trim().slice(0, 120);
          throw new Error("HTTP " + r.status + (head ? " · " + head : ""));
        });
      }
      return r.json();
    }).then(function (d) {
      if (d && d.ok === false) throw new Error(d.detail || "启动失败");
      toast("应用遮罩已开始", "共 " + (d.workers || 8) + " 线程 · " + (d.res || res) + "px", "success");
      maskPoll(btn);
    }).catch(function (e) {
      var msg = String((e && e.message) || e);
      window.__maskDone = true;         /* 起不来也要把界面和按钮放开 */
      toast("应用遮罩失败", msg, "error");
      maskResetBtn(btn);
      dsRunPaint({ pct: 0, msg: "启动失败：" + msg, done: 0, total: 0,
                   speed: null, elapsed: 0, eta: null, running: false, ok: false });
    });
  }
  /* 应用遮罩进度：一次运行只收尾一次。
     /api/dataset/mask/progress 的 speed 来自 JOB.snapshot() 的 done / (now - t0)：
     跑完之后 t0 不再前进、done 也不再涨，于是每轮重算出来的速度只会越来越小。
     以前这里 500ms 无条件重绘、也从不真的停表，界面就停在 100% 一边「掉速」，
     按钮也停在「应用中…」。现在：收到终态 → 定格最终平均速度 + 冻结曲线 +
     复位按钮 + 彻底停轮询；之后后端再推什么旧帧（running=true / pct<100）都丢掉。 */
  function maskIsDone(p) {
    if (!p) return false;
    if (p.running) return false;                /* 还在跑：pct=100 只可能是上一轮的残留 */
    if (p.ok === false) return true;            /* 失败也是终态，按钮同样要放开 */
    var pct = Number(p.pct) || 0;
    var done = Number(p.done) || 0;
    var total = Number(p.total) || 0;
    /* 只认「真的跑满」的终态：done>0 但没跑满的帧，是上一次失败/中断留下的残值，
       直接当完成会把「没跑起来」说成「处理完成」。 */
    return pct >= 100 || (total > 0 && done >= total) || Number(p.stage) === 3;
  }

  function maskResetBtn(btn) {
    var b = btn || dsStepBtn(2);
    if (b) { b.disabled = false; b.textContent = "开始处理"; }
    return b;
  }

  var MASK_POLL_FAST_MS = 2000;   /* 任务开头这 2 秒用 200ms 采样，之后回落 500ms */

  function maskPoll(btn) {
    if (!window.__maskT0) window.__maskT0 = Date.now();   /* 没给起点也不能让 age 永远算 0 */
    var started = false;      /* 见过 running=true 才算这一轮真的跑起来了 */
    var idle = 0;             /* 连续「没跑、没进度」的轮数 */

    var tick = function () {
      if (window.__maskDone) return;                        /* 已定稿：不再轮询 */
      fetch("/api/dataset/mask/progress").then(function (r) { return r.json(); }).then(function (p) {
        if (window.__maskDone) return;                      /* 飞行途中的旧包：以第一次 completed 为准 */
        p = p || {};
        var pct = Math.max(0, Math.min(100, Number(p.pct) || 0));
        var done = Number(p.done) || 0;
        var total = Number(p.total) || 0;
        var running = !!p.running;
        var age = Date.now() - (window.__maskT0 || Date.now());    /* 本轮已跑多久（ms） */
        var elapsed = age / 1000;
        if (running) started = true;
        setNavRunning("dataset", running, pct);
        /* 轮询节奏：前 2 秒 200ms（秒杀任务 1.4 秒，500ms 只够采 2~3 个点），
           之后 500ms，长任务不持续高频打请求。 */
        var next = age < MASK_POLL_FAST_MS ? 200 : 500;


/* ============================================================================
 * 训练器 · 模型转换（.npy -> .pth，Stepper）
 * 从 41-dataset-cfg.js 挪过来的：那个分片在 head 里同步执行时，前面若有一步抛错，
 * 后面这段就永远跑不到（getElementById 全是 null → 空卡片）。放在 00 号最先执行的分片里，
 * 且等 DOM 就绪再初始化。
 * ========================================================================== */
  /* ---------------- 训练器 · 模型转换（.npy -> .pth，Stepper） ----------------
     二级页：面包屑「模型选择 › 模型转换」，左上角有返回。
     以前它是独立 page，进去只能靠侧栏跳走，Ctrl+R 后 hash 还停在转换页 ——
     用户会以为「回不去了，必须重启」。 */
  (function () {
    var $ = function (id) { return document.getElementById(id); };
    var timer = null;

    function esc(s) {
      return String(s == null ? "" : s).replace(/[&<>"]/g, function (c) {
        return ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c];
      });
    }

    function renderList(items) {
      var box = $("cvList");
      if (!box) return;
      if (!items.length) {
        /* 没有待转换的老模型时，这张卡整个收起来 —— 留一个空框在最上面很扎眼 */
        var _cardBox = box.closest && box.closest(".card");
        if (_cardBox) _cardBox.style.display = "none";
        box.innerHTML = "";
          return;
      }
      var _cardOn = box.closest && box.closest(".card");
      if (_cardOn) _cardOn.style.display = "";
      box.innerHTML = items.map(function (it, i) {
        /* kind=modules 是 4.0 后期的分模块 npy（_encoder.npy 那套）：不需要转换，
           导出 dfm 的脚本直接读它 —— 这里别给一个点了就报错的「转换」按钮。 */
        var tag = it.kind === "modules"
          ? '<span class="tag">可直接导出</span>'
          : (it.converted
             ? '<span class="tag">已转换</span>'
             : '<button class="mini" data-cv="' + i + '">转换</button>');
               '<span class="m">' + esc(it.arch || "?") + ' · ' + esc(it.sizeText) + '</span>' +
               '<span class="sp"></span>' + tag + '</div>';
      }).join("");
      window.__cvItems = items;
      box.querySelectorAll("[data-cv]").forEach(function (b) {
        b.addEventListener("click", function () { start(items[+b.getAttribute("data-cv")]); });
      });
    }

    function load() {
      fetch("/api/convert/list").then(function (r) { return r.json(); })
        .then(function (j) {
          var items = (j && j.items) || [];
          renderList(items);
          /* 从模型卡片「转换」按钮进来时，直接开这一个模型 */
          var pick = window.__cvPick;
          if (pick) {
            /* 卡片上的名字是 _cvprobe，转换列表里是 _cvprobe_SAEHD —— 按前缀匹配 */
            var hit = items.filter(function (it) {
              return it.name === pick || it.name.indexOf(pick + "_") === 0;
            })[0];
            window.__cvPick = "";
            if (hit && !hit.converted) setTimeout(function () { start(hit); }, 60);
          }
        })
        .catch(function (e) {
          /* 失败别装成「没有老模型」：那样用户以为目录是空的，其实接口挂了 */
          var box = $("cvList");
          if (box) {
            box.innerHTML = '<div class="cvempty">读不到转换列表：' +
                            String((e && e.message) || e) + "</div>";
          }
        });
    }

    /* 圆点的内容：数字 / ✓ / ! / 旋转圈（照 convert-page-live.html 的 .ic） */
    var SPIN_SVG = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.6" ' +
                   'stroke-linecap="round"><path d="M21 12a9 9 0 1 1-6.2-8.6"/></svg>';
    function step(el, cls, mark) {
      var node = $(el);
      if (!node) return;
      node.className = "cvstep " + (cls || "");
      var ic = node.querySelector(".cvic");
      if (ic) ic.innerHTML = (mark === "spin") ? SPIN_SVG : String(mark);
    }

    /* 一行 = 一个模块：mk · 名字 · 项数；bad 时下面追加一行原因（设计稿的 .reason） */
    function cvRowsHtml(mods, isErr) {
      return (mods || []).map(function (m) {
        var on = m.state === "ok" ? "ok"
               : (m.state === "run" ? "run"
               : ((m.state === "err" || m.state === "bad") ? "bad" : ""));
        var mk = on === "ok" ? "✓" : (on === "run" ? "›" : (on === "bad" ? "!" : "·"));
        var cnt;
        if (on === "ok") cnt = (m.total || m.items || 0) + " 项";
        else if (on === "run") cnt = (m.items || 0) + " / " + (m.total || "?") + " 项";
        else if (on === "bad") cnt = "失败";
        else cnt = m.total ? (m.total + " 项") : "—";
        var html = '<div class="cvrowi ' + on + (on === "bad" ? " shake" : "") + '">' +
                   '<span class="mk">' + mk + '</span>' +
                   '<span class="n">' + esc(m.name) + '</span>' +
                   '<span class="c">' + cnt + '</span>';
        if (on === "bad" && m.reason) {
          html += '<div class="cvreason">' + esc(m.reason) + "</div>";
        }
        return html + "</div>";
      }).join("");
    }

    function paint(st) {
      var card = $("cvCard");
      if (!card) return;
      st = st || {};
      card.style.display = "";
      var mods = st.modules || [];
      $("cvName").textContent = st.name || "未开始";
      var meta = $("cvMeta");
      if (meta) {
        var bits = [];
        if (st.arch) bits.push(String(st.arch).toUpperCase());
        if (mods.length) bits.push(mods.length + " 个模块");
        meta.textContent = bits.join(" · ");
      }
      var isErr = st.step === "error";
      var isIdle = !st.step || st.step === "idle";
      var badge = $("cvBad");
      if (badge) badge.className = "cvbadge" + (isErr ? " on" : "");
      card.classList.toggle("bad", isErr);

      if (isIdle) {
        step("cvS1", "", "1"); step("cvS2", "", "2"); step("cvS3", "", "3");
      } else if (st.step === "load") {
        step("cvS1", "on open", "spin"); step("cvS2", "", "2"); step("cvS3", "", "3");
      } else if (st.step === "convert") {
        step("cvS1", "done", "✓"); step("cvS2", "on open", "spin"); step("cvS3", "", "3");
      } else if (st.step === "done") {
        step("cvS1", "done", "✓"); step("cvS2", "done", "✓"); step("cvS3", "done", "✓");
      } else if (isErr) {
        step("cvS1", "done", "✓"); step("cvS2", "err open", "!"); step("cvS3", "", "3");
      }

      var r1 = $("cvRows1"), r2 = $("cvRows2");
      if (r1) r1.innerHTML = cvRowsHtml(mods, isErr);
      if (r2) r2.innerHTML = cvRowsHtml(mods, isErr);
    }

    function poll() {
      fetch("/api/convert/status").then(function (r) { return r.json(); }).then(function (st) {
        paint(st);
        if (st.running) { timer = setTimeout(poll, 400); }
        else { clearTimeout(timer); timer = null; load(); }
      }).catch(function () { timer = setTimeout(poll, 900); });
    }

    function start(it) {
      $("cvCard").style.display = "";
      step("cvS1", "on", "1"); step("cvS2", "", "2"); step("cvS3", "", "3");
      if ($("cvRows")) $("cvRows").innerHTML = "";
      fetch("/api/convert/start", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ name: it.name })
      }).then(function (r) { return r.json(); }).then(function (j) {
        if (j && j.ok) { if (timer) clearTimeout(timer); timer = setTimeout(poll, 300); }
        else { paint({ step: "error", name: it.name, error: (j && j.detail) || "启动失败", modules: [] }); }
      }).catch(function (e) {
        paint({ step: "error", name: it.name, error: String(e), modules: [] });
      });
    }

    var _go = window.go;
    if (typeof _go === "function" && !_go.__cv) {
      var wrapped = function (k) {
        var r = _go.apply(this, arguments);
        if (k === "train-convert") { setTimeout(load, 30); }
        return r;
      };
      wrapped.__cv = true;
      window.go = wrapped;
    }
    /* 先进页面放个占位：以前这里是纯空白，看不清到底是「没跑」还是「没数据」 */
    /* 这个分片是在 <head> 里同步加载的，此刻 #cvList 还没解析出来 ——
       以前在这里直接 load()，getElementById 拿到 null，renderList 里 `if (!box) return`
       就什么都不画，页面永远是一张空卡片。等 DOM 就绪再跑。 */
    function cvBoot() {
      var box = $("cvList");
      if (box && !box.innerHTML) {
        box.innerHTML = '<div class="cvempty">正在读取 workspace/model …</div>';
      }
      load();
    }
    /* 挂在 IIFE 顶层（不是 cvBoot 里面）：cvBoot 只有在 DOMContentLoaded 时才会跑，
       而路由里的 cvListPaint 是随时会被调的 —— export 写在函数体内就永远是 undefined。 */
    window.__cvBoot = cvBoot;
    window.__cvPaint = paint;
    window.__cvStart = start;
    if (document.readyState === "loading") {
      document.addEventListener("DOMContentLoaded", cvBoot);
    } else {
      cvBoot();
    }
  })();
