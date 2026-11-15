/* DeepFaceLab 4.1 — 前端。所有数据都来自本地 /api，界面本身不含任何算法。 */
/* ==========================================================================
   分节目录（纯注释，代码没有拆成多文件，理由见 app/docs/08-SERVER-MAP.md）：
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
    "tool-color": "p-tool-color", "tool-angle": "p-tool-angle", "tool-engine": "p-tool-engine",
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
    "tool-mask": [{ t: "工具" }, { t: "遮罩绘制" }],
    "tool-bili": [{ t: "工具" }, { t: "B 站下载" }],
    "tool-color": [{ t: "工具" }, { t: "色域迁移" }],
    "tool-angle": [{ t: "工具" }, { t: "角度可视化" }],
    "tool-engine": [{ t: "工具" }, { t: "引擎编译" }],
    debug: [{ t: "关于", go: "about" }, { t: "调试" }],
    "bili-jobs": [{ t: "工具" }, { t: "B 站下载", go: "tool-bili" }, { t: "下载中" }],
    merge: [{ t: "合成工作室" }]
  };

  /* 子页面归属哪个导航项 */
  var NAV_ALIAS = { run: "train", "train-cfg": "train", "extract-run": "extract", "dataset-run": "dataset", debug: "about" };

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
                     "tool-bili": "tools", "tool-color": "tools", "tool-angle": "tools", "train-convert": "train",
                     debug: "about" };

  function groupOf(key) { return PAGE_GROUP[key] || key; }

  /* 子页记忆：从侧边栏回到「训练器 / B 站下载」时，恢复上次停留的子页 */
  var SUB_PARENT = { run: "train", "train-cfg": "train", "train-convert": "train", "extract-run": "extract", "dataset-run": "dataset", "bili-jobs": "tool-bili" };
  var SUB_MEMORY = {};
  window.SUB_MEMORY = SUB_MEMORY;

  var HASH_SYNCING = false;

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
    anim.onfinish = function () { el.remove(); };
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

    var shot = (window.desktop && window.desktop.snapshot) ? window.desktop.snapshot : null;
    if (!shot) { applyTheme(next); return; }

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
    anim.onfinish = function () { el.remove(); };
  }
  window.dismissToast = dismissToast;

  function toast(title, sub, type, ms) {
    var root = $("toastroot");
    if (!root) return null;
    var meta = TOAST_META[type] || TOAST_META.success;
    var life = (typeof ms === "number" && ms > 400) ? ms : 3600;

    var el = document.createElement("div");
    el.className = "toast";
    el.setAttribute("data-type", type || "success");
    el.innerHTML =
      '<svg class="ic ti" style="stroke:' + meta.color + ';opacity:1"><use href="' + meta.icon + '"/></svg>' +
      '<div><div class="tt">' + title + '</div><div class="ts">' + (sub || "") + '</div></div>' +
      '<button class="tx">×</button><div class="tb"><i></i></div>';
    el.querySelector(".tx").addEventListener("click", function () { dismissToast(el); });
    el.querySelector(".tb i").style.animationDuration = life + "ms";
    root.appendChild(el);
    if (window.I18N) window.I18N.refresh();
    setTimeout(function () { dismissToast(el); }, life);
    return el;
  }
  window.toast = toast;

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
    }).then(function (r) { return r.json(); }).then(function (d) {
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
    return fetch("/api/dataset/count", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ dir: arg })
    }).then(function (r) { return r.json(); }).then(function (d) {
      if (!d || d.count == null) return null;
      DS_COUNT = { arg: arg, dir: d.dir || "", count: d.count || 0,
                   skipped: d.skipped_dirs || [], recursive: d.recursive_would_be || 0 };
      var el = document.getElementById("dsPathCount");
      if (el) el.textContent = DS_COUNT.count + " 张";
      dsWarmFaceid();
      return DS_COUNT;
    }).catch(function () { return null; });
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
    el.style.animationDelay = Math.min(idx, 8) * 45 + "ms";

    var params = [];
    if (m.resolution) params.push(m.resolution);
    if (m.ae_dims) params.push("ae" + m.ae_dims);
    if (m.e_dims || m.d_dims) params.push("e" + m.e_dims + "/d" + m.d_dims);
    if (m.face_type) params.push(m.face_type);

    var kv = [];
    kv.push('<span class="tag n">iter ' + fmt(m.iter) + "</span>");
    if (m.batch_size) kv.push('<span class="tag n">bs ' + m.batch_size + "</span>");
    if (m.bf16) kv.push('<span class="tag n">bf16</span>');
    if (m.optimizer) kv.push('<span class="tag n">' + m.optimizer + "</span>");
    if (m.weights === "npy") kv.push('<span class="tag warn">npy · 需转换</span>');
    /* DFM 徽标已按要求去掉 */

    var chart = m.spark && m.spark.src && m.spark.src.length
      ? '<div class="lchart">' + sparkSVG(m.spark) +
        '<div class="lg"><span><i style="background:var(--ink-2)"></i>src</span>' +
        '<span><i style="background:var(--ok)"></i>dst</span>' +
        '<span style="margin-left:auto">最近 ' + m.spark.src.length + " iter</span></div></div>"
      : '<div class="lchart"><div class="lg">暂无 loss 记录</div></div>';

    /* 预览图不可点：以前是 <a target="_blank">，在 Electron 壳里会被
       setWindowOpenHandler 丢给 shell.openExternal —— 点一下模型缩略图就弹外部浏览器，
       非常莫名其妙。这里改成纯 <div>，只负责显示。想看大图去「参数」页或训练页。 */
    var prev = m.preview_url
      ? '<div class="mprev"><img loading="lazy" src="' + m.preview_url + '" alt=""></div>'
      : '<div class="mprev empty">无预览</div>';

    el.innerHTML = chart + prev +
      '<div class="mbody">' +
        '<div class="nm">' + m.name + ' <span class="tag n">' + (m.archi || "—") + "</span></div>" +
        '<div class="meta">' + (params.join(" · ") || "—") + "</div>" +
        '<div class="kv">' + kv.join("") + "</div>" +
        (m.dfm_only
          /* 只有 .dfm、没有训练目录：训练/参数/导出/删除全都无从下手，别摆假按钮 */
          ? '<div class="acts">' +
            '<span class="hint" style="margin-left:8px">只有 .dfm，没有训练目录</span></div>'
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
      confirmBox("删除模型", "会移除 " + m.name + " 的全部权重与配置，无法恢复。", function () {
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
      });
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
  }
  /* 暴露给外部：新建 / 删除模型之后要立刻重画「模型选择」列表 */
  window.renderTrainer = renderTrainer;

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
      frame: n, idx: n, n: n, w: 900,
      /* 和合成请求同一个口径：有「不换脸」的脸才要无损，否则 jpg（快 3 倍） */
      no_swap_faces: (window.mgFaceNoSwapIds ? window.mgFaceNoSwapIds() : []),
      fmt: ((window.mgFaceNoSwapIds && window.mgFaceNoSwapIds().length) ? "png" : "jpg"),
      model: (window.mgFaceSingleModel ? window.mgFaceSingleModel() : ""),
      models: (window.mgFaceModelsPayload ? window.mgFaceModelsPayload() : []),
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
    mgPrepLine("抽帧…", 0);
    toast("准备素材", String(video).split(/[\\/]/).pop() + " · 抽帧 + 对齐", "info");

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
    if (!sel.length) return byFace;       /* 0 个选中：这条不该发生（导出入口已禁用），兜底放行库分配 */
    return sel.map(function (n) {
      return {
        name: n,
        faces: Object.prototype.hasOwnProperty.call(assigned, n) ? assigned[n] : []
      };
    });
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


  /* ============================================================
     合成视图 · 检测框 / 每张脸的模型分配 / 参数实时预览 / 加进人脸库
     ------------------------------------------------------------
     以前「检测」和「合成」是两个页面级视图，互相切换：看框的时候看不到换脸
     结果，看结果时又没有框。现在只有一个画面（换脸结果），框只是叠加层，
     由「框」开关控制显隐；每个框右上角外侧是这张脸用哪个 dfm 的 chip，
     右下角外侧是「＋ 加进人脸数据库」。

     后端契约（后端并行改，字段对不上时用旧字段兜底）：
       POST /api/merge/frame   { frame|n, video|t, settings, models, model }
       POST /api/merge/merged  { frame, settings, models }
       settings = { ct_mode, blend, mask_mode, mask_erode, mask_blur }
       models   = [{ name, faces: [脸序号] }]        ← 沿用导出协议
       POST /api/merge/faces/add     { project?, kps68, box, dfm, frame, source }
                                        ← 点「＋」走这条：已知是哪张脸，68 点仿射
                                          直接落盘，不重检测、不要求有项目
       POST /api/merge/faces/extract { project, frame, dfm }   ← 整帧提取（老接口）
     ============================================================ */
  var MG_COLOR_CARD = ["#e8590c", "#1c7ed6", "#2f9e44", "#a61e4d",
                       "#7048e8", "#0c8599", "#d6336c", "#f08c00"];
  var MG_COLORS = {};          /* dfm 名 → 颜色；同一模型任何一帧都是这个色 */
  var MG_COLORS_KEY = "dfl-merge-model-colors";
  var MG_FACE_MODEL = {};      /* 脸序号 → dfm 名；**用户手动指定**的（会随请求发回去） */
  /* 脸序号 → {dfm, group, groupName, similarity, matched}：**后端按 embedding 自动判定**
     的结果（每次合成回包里的 faces）。和 MG_FACE_MODEL 分开放是有意的 ——
     这份不能当输入发回去，否则第一次判错就会被当成"用户指定"永远错下去。 */
  var MG_FACE_AUTO = {};
  var MG_FACE_NOSWAP = {};     /* 脸序号 → true：这张脸「不换脸」，请求里走 no_swap_faces */
  /* 脸序号 → true：这张脸被「点框」切成了原图（框线虚线 + 框角小标签）。
     和 MG_FACE_NOSWAP 走同一个契约键 no_swap_faces，区别只是它不改模型分配：
     用户只是想在这一帧上对比「换没换」，不是把这张脸从合成里摘出去。
     记住是**用户偏好**：切帧/切素材都不重置（存 localStorage）。 */
  var MG_FACE_RAWVIEW = {};
  var MG_FACE_RAWVIEW_KEY = "dfl-merge-face-raw";
  var MG_FACE_BUSY = {};       /* 脸序号 → true：这一张正在重算（画细进度环） */
  var MG_RECTS = [];           /* 脸序号 → 画布内矩形（chip / ＋ 的定位依据） */
  var MG_FACE_RAW = [];        /* 脸序号 → 原图坐标里的 box（提取人脸时回传） */
  var MG_CMP = { seq: 0, ctl: null, wait: null, lastMs: 0 };

  (function loadColors() {
    try {
      var raw = window.localStorage.getItem(MG_COLORS_KEY);
      if (!raw) return;
      var o = JSON.parse(raw);
      Object.keys(o || {}).forEach(function (k) {
        if (typeof o[k] === "string" && o[k]) MG_COLORS[k] = o[k];
      });
    } catch (e) { /* 无痕模式 / 关掉存储：内存里随机一份照样能用 */ }
  })();
  function mgSaveColors() {
    try { window.localStorage.setItem(MG_COLORS_KEY, JSON.stringify(MG_COLORS)); } catch (e) {}
  }

  /* 点框切「这张脸的原图」的偏好：按脸序号记，切帧/重开页面都还在 */
  (function loadRawView() {
    try {
      var raw = window.localStorage.getItem(MG_FACE_RAWVIEW_KEY);
      if (!raw) return;
      var o = JSON.parse(raw);
      Object.keys(o || {}).forEach(function (k) {
        var i = parseInt(k, 10);
        if (!isNaN(i) && o[k]) MG_FACE_RAWVIEW[i] = true;
      });
    } catch (e) { /* 无痕模式：内存里照样能用，只是不跨刷新 */ }
  })();
  function mgSaveRawView() {
    try {
      window.localStorage.setItem(MG_FACE_RAWVIEW_KEY, JSON.stringify(MG_FACE_RAWVIEW));
    } catch (e) {}
  }

  /* 固定色卡里随机抽一个；抽过就不再变（= 同一模型跨帧同色） */
  function mgModelColor(name) {
    var n = String(name || "");
    if (!n) return "";
    if (MG_COLORS[n]) return MG_COLORS[n];
    var used = {};
    Object.keys(MG_COLORS).forEach(function (k) { used[MG_COLORS[k]] = 1; });
    var free = MG_COLOR_CARD.filter(function (c) { return !used[c]; });
    var pool = free.length ? free : MG_COLOR_CARD;
    var c = pool[Math.floor(Math.random() * pool.length)];
    MG_COLORS[n] = c;
    mgSaveColors();
    return c;
  }
  window.mgModelColor = mgModelColor;

  /* 未分配的脸 = 灰色描边。以前取 --ink-3（深色主题 #6d6d78）：压在近黑的帧上
     实测对比只有 1.0~1.3:1 —— 框其实画了，肉眼和截图都看不出来。
     中间那版是 rgba(255,255,255,.55)：亮度够了，但**半透明白**压在脸上就是用户报的
     「框上有一层白蒙版」。现在用等亮度的**不透明**浅灰（#9aa0ae，压在 #1a1a1a 上仍然
     6:1），亮背景那一头由 mvDrawBoxes 里的 2px 深色外描边兜底（近白帧实测 4.4~5.0:1）。
     规矩：描边只准用不透明色 —— 框里一个像素都不许被半透明层糊到。 */
  function mgUnassignedColor() {
    return "#9aa0ae";
  }
  window.mgUnassignedColor = mgUnassignedColor;

  /* 后端自动判定的模型名（没有就是空串） */
  function mgFaceAuto(i) {
    var a = MG_FACE_AUTO[i];
    return (a && a.dfm) ? String(a.dfm) : "";
  }
  window.mgFaceAuto = mgFaceAuto;

  /* 这张脸现在用哪个 dfm：手动指定 > 后端自动判定 > 空（灰框） */
  function mgFaceModel(i) {
    if (Object.prototype.hasOwnProperty.call(MG_FACE_MODEL, i)) return String(MG_FACE_MODEL[i] || "");
    return mgFaceAuto(i);
  }
  window.mgFaceModel = mgFaceModel;

  /* 左栏「当前选中的模型」= 这一帧真正在用的那个 dfm（单选时整帧都归它）。
     0 个选中 = 空串。顺序就是列表顺序（多选时第一个是 mgCmpBody 发出去的 model）。
     **不**退回 MergePages.settings().model：那个是参数页的下拉值，用户在第 0 个选中时
     看到的应该是「还没选」，不是一个他以为自己没选过的模型。 */
  function mgFaceSelNames() {
    try { return mgSelNames() || []; } catch (e) { return []; }
  }
  window.mgFaceSelNames = mgFaceSelNames;

  /* chip 上那个模型名到底是谁（只用于上色 / 文案，不参与请求体）：
     用户手动指定 > 后端自动判定(faces[]) > 左栏选中的第一个。都没有 = 空串（灰框）。 */
  function mgFaceChipModel(i) {
    var nm = mgFaceModel(i);
    if (nm) return nm;
    if (mgFaceIsNoSwap(i)) return "";
    var a = MG_FACE_AUTO[i];
    if (a && a.matched === false) return "";
    var sel = mgFaceSelNames();
    return sel.length ? String(sel[0] || "") : "";
  }
  window.mgFaceChipModel = mgFaceChipModel;

  /* chip 上的字：手动指定 > 后端自动判定的 dfm > 不换脸 > 没匹配上 > 左栏选中的模型 > 选择模型。
     「没匹配」= 后端算完了但谁的组都不像（< 阈值），这张脸会被保持原像素。
     关键的一条：**后端没回 faces[] 不等于没有模型**。单模型预览/合成回包只带
     model / dfm（那才是这一帧真正生效的 dfm），chip 以前只认 faces[]，于是一个模型
     换得好好的画面，框上却写着「选择模型」（用户报的就是这个）。现在单选用左栏
     选中的那个名字，多选用「第一个 + 剩余个数」。 */
  function mgFaceChipText(i) {
    var nm = mgFaceModel(i);
    if (nm) return nm;
    if (mgFaceIsNoSwap(i)) return "不换脸";
    var a = MG_FACE_AUTO[i];
    if (a && a.matched === false) return "没匹配";
    var sel = mgFaceSelNames();
    if (sel.length === 1) return String(sel[0] || "");
    if (sel.length > 1) return String(sel[0] || "") + " +" + (sel.length - 1);
    return "选择模型";
  }
  window.mgFaceChipText = mgFaceChipText;

  /* 「不换脸」= 契约里的 no_swap_faces：后端会把这张脸原样留在帧里（连一个像素都
     不动），而不是"没分配 → 兜给第 0 个模型"。序号就是检测顺序，从 0 起。 */
  function mgFaceIsNoSwap(i) { return !!MG_FACE_NOSWAP[i]; }
  window.mgFaceIsNoSwap = mgFaceIsNoSwap;

  /* 这张脸现在显示的是「原图」（点框切过来的） */
  function mgFaceIsRawView(i) { return !!MG_FACE_RAWVIEW[i]; }
  window.mgFaceIsRawView = mgFaceIsRawView;

  /* 请求体的 no_swap_faces = 「不换脸」的脸 ∪ 「点成原图」的脸。
     两件事在契约上是同一件：这张脸保持原像素。 */
  function mgFaceNoSwapIds() {
    var out = {}, k;
    [MG_FACE_NOSWAP, MG_FACE_RAWVIEW].forEach(function (m) {
      Object.keys(m || {}).forEach(function (key) {
        var i = parseInt(key, 10);
        if (!isNaN(i) && m[key]) out[i] = true;
      });
    });
    return Object.keys(out).map(function (x) { return parseInt(x, 10); })
      .filter(function (i) { return !isNaN(i); })
      .sort(function (x, y) { return x - y; });
  }
  window.mgFaceNoSwapIds = mgFaceNoSwapIds;
  /* 换素材/换帧不影响这里（和 MG_FACE_MODEL 一样是"这一轮合成"的设置） */

  /* 检测框始终显示：它是「点框切这张脸的合成/原图」的唯一入口，界面上没有开关。
     函数留着是因为调用点一大片（都在问"要不要画框"），现在答案恒为"要"。 */
  function mgBoxesOn() { return true; }
  window.mgBoxesOn = mgBoxesOn;

  /* 视图切换（合成 / 调试）已经删掉：合成视图就是唯一视图。
     右上角那个 #mgViewSwitch、MG_VIEW 状态、按视图预热与缓存，全都不要了 ——
     现在合成只有一条路（mgCmpNow），每一发回包只贴当前这一帧。
     「调试」下沉到**每张脸**：框上的「调试」按钮打开那张脸的六宫格
     （见 mgFaceDebug* / /api/merge/face-debug），舞台不动、底图不动。 */


  /* 一次请求的底稿：settings / models / 帧号 / 不换脸的脸，全在这儿。
     合成视图只有一种，所以这里就是**完整**的请求体（六宫格那条接口拿它去掉
     idx 再补一个 face 就是自己的 body，参数永远和画面一致）。 */
  function mgCmpBody(frame) {
    frame = Math.max(0, parseInt(frame, 10) || 0);
    var mv = window.__mv || {};
    var v = mgVid();
    var fps = mv.fps || 30;
    var t = (v && mgVideoMode()) ? (Math.round(v.currentTime * 1000) / 1000) : (frame / fps);
    var ns = mgFaceNoSwapIds();
    return {
      frame: frame, idx: frame, n: frame,
      video: mgVidPath(), t: t, w: 900,
      /* 人脸库跟着当前项目走（空串 = 默认库 <工作区>/merge） */
      project: window.__mergeInputDir || "",
      /* 点过「不换脸」/ 点成原图的那几张脸：后端按这个列表原样留下 */
      no_swap_faces: ns,
      /* 有「不换脸」的脸就要无损（png）：那几张脸要和原帧逐像素相同 */
      fmt: (ns.length ? "png" : "jpg"),
      model: mgFaceSingleModel(),
      models: mgFaceModelsPayload(),
      settings: mgCmpSettings()
    };
  }
  window.mgCmpBody = mgCmpBody;

  /* ---------- 点框 = 只切这一张脸的「合成 / 原图」 ----------
     契约上就是 no_swap_faces：把这张脸的序号加进/移出请求体，后端把它原样留在
     帧里（其它脸照常换）。视觉反馈分两半：
       · 立刻：这条框线变虚线 + 框角出现「原图」小标签（不等网络）；
       · 随后：整帧重算回来，这张脸是原像素、其它脸还是换过的。
     状态按脸序号记在 localStorage 里（用户偏好），切帧不重置。 */
  function mgToggleFaceRaw(i, opt) {
    i = parseInt(i, 10);
    if (isNaN(i) || i < 0) return false;
    if (MG_FACE_RAWVIEW[i]) delete MG_FACE_RAWVIEW[i];
    else MG_FACE_RAWVIEW[i] = true;
    mgSaveRawView();
    mgSyncFaceCards();
    if (window.mvRedrawBoxes) window.mvRedrawBoxes();   /* 实线/虚线立刻换 */
    if (opt && opt.silent) return mgFaceIsRawView(i);
    var p = (window.mgCmpNow ? window.mgCmpNow() : null);
    if (p && p.catch) p.catch(function () {});
    return mgFaceIsRawView(i);
  }
  window.mgToggleFaceRaw = mgToggleFaceRaw;

  /* 画布坐标里的点命中哪个框；没有就 -1 */
  function mgFaceHitAt(x, y) {
    var stage = document.getElementById("mgStage");
    if (!stage || !MG_RECTS.length) return -1;
    var cv = stage.querySelector("canvas.mgovl");
    if (!cv) return -1;
    var r = cv.getBoundingClientRect();
    if (!r.width || !r.height) return -1;
    var px = (x - r.left) * (cv.width / r.width);
    var py = (y - r.top) * (cv.height / r.height);
    for (var i = 0; i < MG_RECTS.length; i++) {
      var b = MG_RECTS[i];
      if (!b || !(b.w > 0) || !(b.h > 0)) continue;
      if (px >= b.x && px <= b.x + b.w && py >= b.y && py <= b.y + b.h) return i;
    }
    return -1;
  }
  window.mgFaceHitAt = mgFaceHitAt;

  /* 绑一次：按下→抬起位移 < 4px 才算「点框」（拖拽/选择一律不算）。
     点在 chip / ＋ 上的不算 —— 它们是框外的按钮，有自己的事要干。 */
  var MG_PICK = { id: null, x: 0, y: 0, t: 0 };
  function mgBindFacePick() {
    var stage = document.getElementById("mgStage");
    if (!stage || window.__mgFacePickBound) return;
    window.__mgFacePickBound = 1;
    stage.setAttribute("data-face-pick", "1");
    var PICK_SLOP = 4;          /* 点击判定阈值（像素）：位移 >= 4 就是拖拽 */

    /* 绑在 document 上而不是舞台上：合成图 / 帧图 / <video> 三层的 pointer-events
       各不相同，绑在舞台上时点画面未必落到舞台自己身上（实测无头 Edge 里
       elementFromPoint 命中的是舞台里的图，不是舞台）。命中判定自己做，
       只有落在某个框里才切，点别处一律不动。 */
    /* 点在其它控件上不算：左右栏、时间轴、顶部的「人脸数据库」胶囊、toast…
       它们和舞台在同一个坐标系里，光靠"点落在框的矩形里"会误判。 */
    var PICK_SKIP = ".mgfc, button, a, input, select, label, #toastroot, " +
                    ".mgleft, .mgright, .mtl, .mgdbentry, .mgviews, .mgfgrid";

    document.addEventListener("pointerdown", function (e) {
      if (e.button != null && e.button !== 0) { MG_PICK.id = null; return; }
      var t = e.target;
      if (t && t.closest && t.closest(PICK_SKIP)) { MG_PICK.id = null; return; }
      MG_PICK.id = (e.pointerId == null ? 1 : e.pointerId);
      MG_PICK.x = e.clientX;
      MG_PICK.y = e.clientY;
      MG_PICK.t = Date.now();
    });

    document.addEventListener("pointerup", function (e) {
      var id = (e.pointerId == null ? 1 : e.pointerId);
      if (MG_PICK.id !== id) return;
      MG_PICK.id = null;
      var dx = (e.clientX || 0) - MG_PICK.x;
      var dy = (e.clientY || 0) - MG_PICK.y;
      if (Math.sqrt(dx * dx + dy * dy) >= PICK_SLOP) return;   /* 拖拽，不算点击 */
      if (Date.now() - MG_PICK.t > 1500) return;               /* 长按，不算点击 */
      var i = mgFaceHitAt(e.clientX, e.clientY);
      if (i < 0) return;
      e.preventDefault();
      mgToggleFaceRaw(i);
    });

    document.addEventListener("pointercancel", function () { MG_PICK.id = null; });
  }
  window.mgBindFacePick = mgBindFacePick;
  /* 测试/自动化用：每个框在页面坐标里的位置 + 现在是不是「原图」状态。
     只读，不改任何行为。 */
  window.mgFaceRects = function () {
    var stage = document.getElementById("mgStage");
    var cv = stage && stage.querySelector("canvas.mgovl");
    if (!cv) return [];
    var r = cv.getBoundingClientRect();
    if (!r.width || !cv.width) return [];
    var sx = r.width / cv.width, sy = r.height / cv.height;
    var out = [];
    MG_RECTS.forEach(function (b, i) {
      if (!b || !(b.w > 0) || !(b.h > 0)) return;
      out.push({ i: i,
                 x: r.left + b.x * sx, y: r.top + b.y * sy,
                 w: b.w * sx, h: b.h * sy,
                 cx: r.left + (b.x + b.w / 2) * sx,
                 cy: r.top + (b.y + b.h / 2) * sy,
                 raw: mgFaceIsRawView(i) });
    });
    return out;
  };
  function mgSetRects(rects, raw) {
    MG_RECTS = rects || [];
    MG_FACE_RAW = raw || [];
    mgRenderFaces();
  }

  /* ---------- 单脸调试：六宫格（3 列 × 2 行） ----------
     用户点某张脸框上的「调试」→ 舞台换成这张脸的特写六宫格：
       第 1 行  (1) mask(dfm-pred)      (2) mask(dst)    (3) merged(mask(dfm-pred), mask(dst))
       第 2 行  (4) dfm-pred            (5) dst-original (6) merged
     图从后端 GET/POST /api/merge/face-debug 来（**六张独立图**，前端自己排版，
     所以点任意一格就能放大）。后端复用预览 worker 里那一份 MergeMaskedFace，
     参数和画面用的是同一份 mgCmpBody（去掉 idx 再补 face）。
     六宫格是**盖在舞台上的一层**：底下的合成图、描边 canvas、chip 一个都不动，
     所以点「返回」= 把这层收起来，画面和进来之前逐像素一致。 */
  var MG_FDBG_CELLS = [
    ["mask-prd", "mask(dfm-pred)"],
    ["mask-dst", "mask(dst)"],
    ["mask-both", "merged(mask(dfm-pred), mask(dst))"],
    ["dfm-pred", "dfm-pred"],
    ["dst-original", "dst-original"],
    ["merged", "merged"]
  ];
  var MG_FDBG = { face: -1, seq: 0, data: null, zoom: "", busy: false, err: "" };

  function mgFdbgEl() { return document.getElementById("mgFaceGrid"); }

  /* 这张脸的六宫格请求体：和画面上那一发**同一套参数**，
     只把 idx（老接口里是帧号）摘掉 —— 这个接口的 face 才是「第几张脸」。 */
  function mgFdbgBody(i) {
    var frame = mgCmpFrameIdx();
    var base = window.mgCmpBody ? (window.mgCmpBody(frame) || {}) : {};
    var body = {};
    Object.keys(base).forEach(function (k) { body[k] = base[k]; });
    delete body.idx;
    body.frame = frame;
    body.n = frame;
    body.face = parseInt(i, 10) || 0;
    body.panel_size = 256;
    return body;
  }
  window.mgFdbgBody = mgFdbgBody;

  function mgFdbgSkeleton(i, frame) {
    var el = mgFdbgEl();
    if (!el) return null;
    el.innerHTML = "";
    var hd = document.createElement("div");
    hd.className = "mgfg-hd";
    var back = document.createElement("button");
    back.type = "button";
    back.id = "mgFaceGridBack";
    back.className = "mgfg-back";
    back.textContent = "← 返回";
    back.title = "回到合成视图（Esc）";
    var tt = document.createElement("span");
    tt.className = "mgfg-tt";
    tt.textContent = "";
    var sub = document.createElement("span");
    sub.className = "mgfg-sub";
    sub.id = "mgfgSub";
    sub.textContent = "正在算第 " + frame + " 帧…";
    hd.appendChild(back);
    hd.appendChild(tt);
    hd.appendChild(sub);
    el.appendChild(hd);

    var body = document.createElement("div");
    body.className = "mgfg-body";
    body.id = "mgfgBody";
    MG_FDBG_CELLS.forEach(function (pair, k) {
      var cell = document.createElement("figure");
      cell.className = "mgfg-cell";
      cell.setAttribute("data-k", pair[0]);
      cell.setAttribute("data-no", String(k + 1));
      cell.innerHTML =
        '<figcaption><span class="mgfg-nm">' + pair[1] + '</span></figcaption>' +
        '<div class="mgfg-im"><img alt="' + pair[1] + '"></div>';
      body.appendChild(cell);
    });
    el.appendChild(body);
    return el;
  }

  function mgFdbgFill(d) {
    var el = mgFdbgEl();
    if (!el) return;
    var by = {};
    ((d && d.panels) || []).forEach(function (p) { if (p && p.key) by[p.key] = p; });
    Array.prototype.forEach.call(el.querySelectorAll(".mgfg-cell"), function (cell) {
      var k = cell.getAttribute("data-k");
      var p = by[k] || null;
      var img = cell.querySelector("img");
      if (p && p.png) {
        img.src = p.png;
        cell.classList.add("has");
        cell.setAttribute("data-w", String(p.w || 0));
        cell.setAttribute("data-h", String(p.h || 0));
      } else {
        img.removeAttribute("src");
        cell.classList.remove("has");
      }
    });
    var sub = document.getElementById("mgfgSub");
    if (sub) {
      var st = (d && d.stats) || {};
      var bits = [];
      if (d && d.face && d.face.model) bits.push("模型 " + d.face.model);
      if (st.mask_prd != null) bits.push("遮罩覆盖 " + Math.round(Number(st.mask_prd) * 100) + "%");
      if (d && d.frozen) bits.push("冻结参数");
      if (d && d.ms != null) bits.push(Number(d.ms) + "ms");
      if (d && d.cached) bits.push("缓存");
      sub.textContent = "";
    }
  }
  window.mgFdbgFill = mgFdbgFill;

  function mgFaceDebugOpen(i) {
    i = parseInt(i, 10);
    if (isNaN(i) || i < 0) return null;
    var el = mgFdbgEl();
    if (!el) return null;
    var frame = mgCmpFrameIdx();
    MG_FDBG.face = i;
    MG_FDBG.zoom = "";
    MG_FDBG.data = null;
    MG_FDBG.err = "";
    mgFdbgSkeleton(i, frame);
    el.hidden = false;
    el.setAttribute("aria-hidden", "false");
    el.classList.remove("zoom");
    var st = document.getElementById("mgStage");
    if (st) st.setAttribute("data-face-debug", String(i));
    mgFaceDebugBind();
    var seq = ++MG_FDBG.seq;
    MG_FDBG.busy = true;
    fetch("/api/merge/face-debug", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(mgFdbgBody(i))
    }).then(function (r) {
      return r.json().then(function (d) {
        if (!r.ok || !d || d.ok === false) {
          throw new Error((d && (d.detail || d.error)) || ("HTTP " + r.status));
        }
        return d;
      }, function () { throw new Error("HTTP " + r.status); });
    }).then(function (d) {
      if (seq !== MG_FDBG.seq) return null;      /* 已经关了/换了另一张脸：丢掉 */
      MG_FDBG.busy = false;
      MG_FDBG.data = d;
      mgFdbgFill(d);
      return d;
    }).catch(function (e) {
      if (seq !== MG_FDBG.seq) return null;
      MG_FDBG.busy = false;
      MG_FDBG.err = String((e && e.message) || e);
      var sub = document.getElementById("mgfgSub");
      if (sub) sub.textContent = "六宫格取图失败：" + MG_FDBG.err.slice(0, 90);
      if (window.mgHint) window.mgHint("");
      if (window.toast) {
        try { toast("六宫格取图失败", MG_FDBG.err.slice(0, 90), "warn", 5000); } catch (x) {}
      }
      return null;
    });
    return el;
  }
  window.mgFaceDebugOpen = mgFaceDebugOpen;

  function mgFaceDebugClose() {
    MG_FDBG.seq++;                 /* 在飞的那一发回来也不要了 */
    MG_FDBG.face = -1;
    MG_FDBG.data = null;
    MG_FDBG.busy = false;
    MG_FDBG.zoom = "";
    var el = mgFdbgEl();
    if (el) {
      el.hidden = true;
      el.setAttribute("aria-hidden", "true");
      el.classList.remove("zoom", "on");
      el.innerHTML = "";
    }
    var st = document.getElementById("mgStage");
    if (st) st.removeAttribute("data-face-debug");
    return true;
  }
  window.mgFaceDebugClose = mgFaceDebugClose;

  /* 点格子放大 / 返回 / Esc —— 只绑一次（网格节点会重建，事件挂在容器上） */
  function mgFaceDebugBind() {
    var el = mgFdbgEl();
    if (!el || el.dataset.bound) return;
    el.dataset.bound = "1";
    el.addEventListener("click", function (e) {
      var t = e.target;
      if (t && t.closest && t.closest(".mgfg-back")) {
        e.preventDefault();
        mgFaceDebugClose();
        return;
      }
      var cell = t && t.closest ? t.closest(".mgfg-cell") : null;
      if (!cell || !el.contains(cell)) return;
      var k = String(cell.getAttribute("data-k") || "");
      if (!cell.classList.contains("has")) return;
      /* 再点一次同一格 = 缩回去 */
      if (MG_FDBG.zoom === k) {
        MG_FDBG.zoom = "";
        el.classList.remove("zoom");
        Array.prototype.forEach.call(el.querySelectorAll(".mgfg-cell.on"), function (c) {
          c.classList.remove("on");
        });
        return;
      }
      MG_FDBG.zoom = k;
      el.classList.add("zoom");
      Array.prototype.forEach.call(el.querySelectorAll(".mgfg-cell"), function (c) {
        c.classList.toggle("on", c === cell);
      });
    });
    if (!window.__mgFdbgKeyBound) {
      window.__mgFdbgKeyBound = 1;
      document.addEventListener("keydown", function (e) {
        if (e.key !== "Escape") return;
        var box = mgFdbgEl();
        if (!box || box.hidden) return;
        e.preventDefault();
        e.stopPropagation();
        if (MG_FDBG.zoom) {          /* 先退出放大，再退六宫格 */
          MG_FDBG.zoom = "";
          box.classList.remove("zoom");
          Array.prototype.forEach.call(box.querySelectorAll(".mgfg-cell.on"), function (c) {
            c.classList.remove("on");
          });
          return;
        }
        mgFaceDebugClose();
      });
    }
  }
  window.mgFaceDebugBind = mgFaceDebugBind;

  /* 测试/自动化用（只读）：六宫格现在开着没、哪张脸、六格都拿到图没有 */
  window.mgFaceDebugState = function () {
    var el = mgFdbgEl();
    if (!el) return { present: false };
    var cells = Array.prototype.map.call(el.querySelectorAll(".mgfg-cell"), function (c) {
      var im = c.querySelector("img");
      var r = c.getBoundingClientRect();
      return { k: c.getAttribute("data-k"), no: c.getAttribute("data-no"),
               nm: String((c.querySelector(".mgfg-nm") || {}).textContent || ""),
               has: c.classList.contains("has"),
               src: im ? String(im.getAttribute("src") || "").slice(0, 32) : "",
               nw: im ? im.naturalWidth : 0, nh: im ? im.naturalHeight : 0,
               w: Math.round(r.width), h: Math.round(r.height) };
    });
    return { present: true, open: !el.hidden, face: MG_FDBG.face,
             zoom: MG_FDBG.zoom, busy: MG_FDBG.busy, err: MG_FDBG.err,
             rect: (function () { var r = el.getBoundingClientRect();
                                 return [r.x, r.y, r.width, r.height]; })(),
             cells: cells };
  };


  /* 换素材了：脸序号是「这一帧的检测顺序」，把上一份素材的分配带过去毫无意义
     （3 张脸的项目点了 face 2 不换脸，再开一个 1 张脸的视频，那张脸就永远不换）。 */
  function mgClearFaceAssign() {
    MG_FACE_MODEL = {};
    MG_FACE_NOSWAP = {};
    MG_FACE_AUTO = {};
    mgResetFaces();
    /* 换了素材 → 刚才那份「能不能拿到这一帧的图」的缓存作废 */
    if (window.mgSourceForget) window.mgSourceForget();
  }
  window.mgClearFaceAssign = mgClearFaceAssign;

  function mgResetFaces() {
    MG_RECTS = [];
    MG_FACE_RAW = [];
    var layer = document.getElementById("mgFaceLayer");
    if (layer) layer.innerHTML = "";
    /* 脸都没了：还开着的六宫格指的就是不存在的脸，收掉 */
    if (window.mgFaceDebugClose) window.mgFaceDebugClose();
  }
  window.mgResetFaces = mgResetFaces;

  /* ---------- chip 层：每张脸一个小卡片 + 一个 ＋，全部是 DOM ----------
     关键：**脸没变就不重建**。
     以前这里无条件 layer.innerHTML = "" 再重建。而这张图每次合成回包都会重画
     （mgShowOverlay → mgApplyView → mvRedrawBoxes，加上 p.onload 再补一发、检测回包再来一发，
     一次换脸 = 3 次重画），重建会把**打开着的下拉菜单**连同里面的 .ddopt 一起扔掉：
     用户的 mousedown 落在旧节点上、mouseup/click 落在新节点或外层上 —— 浏览器把 click
     派发给两者共同的祖先，选项上的监听器一次都没跑，于是 MG_FACE_MODEL 没写、
     chip 还是「选择模型」、重开下拉也没有高亮（用户报的就是这个）。
     现在：脸序号列表一致 → 只原地更新位置/文字/颜色/忙碌环（下拉不受影响）；真的换了脸数才重建。 */
  function mgRenderFaces() {
    var layer = document.getElementById("mgFaceLayer");
    if (!layer) return;
    if (!mgBoxesOn()) { layer.innerHTML = ""; return; }
    var want = [], i, r;
    for (i = 0; i < MG_RECTS.length; i++) {
      r = MG_RECTS[i];
      if (!r || !(r.w > 0) || !(r.h > 0)) continue;
      want.push(i);
    }
    var cards = Array.prototype.slice.call(layer.querySelectorAll(".mgfc"));
    var same = (cards.length === want.length);
    for (i = 0; same && i < cards.length; i++) {
      if (parseInt(cards[i].getAttribute("data-face"), 10) !== want[i]) same = false;
    }
    if (same) { mgSyncFaceCards(); return; }   /* ← 原地更新，别炸掉打开着的下拉 */
    layer.innerHTML = "";
    for (i = 0; i < want.length; i++) layer.appendChild(mgMakeFaceCard(want[i], MG_RECTS[want[i]]));
    mgSyncFaceCards();
  }
  window.mgRenderFaces = mgRenderFaces;

  function mgCloseChipMenus(except) {
    Array.prototype.forEach.call(document.querySelectorAll("#mgFaceLayer .mgfdd.open"), function (d) {
      if (d !== except) d.classList.remove("open");
    });
  }

  function mgMakeFaceCard(i, r) {
    /* 颜色跟着 chip 上显示的那个名字走：单选/多选时那一帧的全部脸确实都在用
       左栏的 dfm，chip 不该是灰的（以前只有 MG_FACE_MODEL / faces[] 才算"已分配"）。 */
    var mname = mgFaceChipModel(i);
    var color = mname ? mgModelColor(mname) : "";

    /* 零尺寸锚点落在「框的右上角」：left 要加宽度，否则 anchor 停在左边线，
       chip 会跑到框外左侧、压到左栏上去（点都点不到） */
    var card = document.createElement("div");
    card.className = "mgfc" + (MG_FACE_BUSY[i] ? " busy" : "");
    card.setAttribute("data-face", String(i));
    /* 框角小标签「原图」：这张脸被点成原图时才露出来（别的框一个像素都不变）。
       pointer-events:none —— 它盖在画面上，点它要继续算「点框」。 */
    var vtag = document.createElement("i");
    vtag.className = "mgvtag";
    vtag.textContent = "原图";
    vtag.hidden = !mgFaceIsRawView(i);
    card.style.left = (r.x + r.w) + "px";
    card.style.top = r.y + "px";

    /* 右上角外侧：模型 chip（.dd / .ddbtn / .ddopt 是项目里的主题化下拉） */
    var dd = document.createElement("div");
    dd.className = "dd mgfdd";
    var btn = document.createElement("button");
    btn.type = "button";
    btn.className = "ddbtn mgfb" + (color ? " has" : "");
    btn.title = mname ? ("这张脸用 " + mname + "（点击换）") : "给这张脸选一个模型";
    if (color) { btn.style.borderColor = color; btn.style.color = color; }
    var txt = document.createElement("span");
    txt.className = "ddtxt";
    txt.textContent = mgFaceChipText(i);
    btn.appendChild(txt);

    var menu = document.createElement("div");
    menu.className = "ddmenu mgfmenu";
    btn.addEventListener("click", function (e) {
      e.stopPropagation();
      var open = dd.classList.contains("open");
      mgCloseChipMenus(null);
      if (open) { dd.classList.remove("open"); return; }
      menu.innerHTML = "";
      /* 选项 = 左栏已选中的 dfm + 「不换脸」 */
      var names = mgSelNames();
      var opts = names.concat([""]);
      if (!names.length) {
        var none = document.createElement("div");
        none.className = "ddopt off";
        none.textContent = "左栏还没选模型";
        menu.appendChild(none);
      }
      opts.forEach(function (nm) {
        var it = document.createElement("div");
        /* 「不换脸」那一项的高亮看的是 MG_FACE_NOSWAP，不是"有没有分配" */
        it.className = "ddopt" + ((nm ? ((mgFaceModel(i) || "") === nm) : mgFaceIsNoSwap(i)) ? " on" : "");
        it.textContent = nm || "不换脸";
        if (nm) it.style.color = mgModelColor(nm);
        it.addEventListener("click", function (ev) {
          ev.stopPropagation();
          dd.classList.remove("open");
          mgAssignFace(i, nm);
        });
        menu.appendChild(it);
      });
      dd.classList.add("open");
    });
    dd.appendChild(btn);
    dd.appendChild(menu);

    /* 重算中的细进度环（只圈这一张脸，不整屏 loading） */
    var ring = document.createElement("i");
    ring.className = "mgfring";
    ring.hidden = !MG_FACE_BUSY[i];
    if (!ring.hidden && ring.animate) {
      try {
        ring.animate([{ transform: "rotate(0deg)" }, { transform: "rotate(360deg)" }],
                     { duration: 720, iterations: Infinity, easing: "linear" });
      } catch (e) { /* 没有 WAAPI 就留一个静态环 */ }
    }
    dd.appendChild(ring);

    /* 右下角外侧，和 ＋ 并排：这张脸的六宫格调试（3 列 × 2 行特写）。
       位置靠右对齐、在 ＋ 左边一格 —— 两个按钮同高同圆角，视觉上是一对。 */
    var dbg = document.createElement("button");
    dbg.type = "button";
    dbg.className = "mgfdbg";
    dbg.textContent = "调试";
    dbg.title = "看这张脸的六宫格（遮罩 / 预测脸 / 原脸 / 合成）";
    dbg.setAttribute("data-tip", "看这张脸的六宫格");
    dbg.setAttribute("aria-label", "看这张脸的六宫格");
    dbg.style.top = Math.round(r.h + 3) + "px";
    dbg.addEventListener("click", function (e) {
      e.stopPropagation();
      mgFaceDebugOpen(i);
    });

    /* 最右下角外侧：加进人脸数据库 */
    var plus = document.createElement("button");
    plus.type = "button";
    plus.className = "mgplus";
    plus.textContent = "+";
    plus.title = "把当前人脸添加到人脸数据库";
    plus.setAttribute("data-tip", "把当前人脸添加到人脸数据库");
    plus.setAttribute("aria-label", "把当前人脸添加到人脸数据库");
    plus.style.top = Math.round(r.h + 3) + "px";
    plus.addEventListener("click", function (e) {
      e.stopPropagation();
      mgAddFaceToDb(i, plus);
    });

    card.appendChild(vtag);
    card.appendChild(dd);
    card.appendChild(dbg);
    card.appendChild(plus);
    return card;
  }

  /* 只改已存在卡片的外观（换色 / 换字 / 进度环），不重建 DOM：
     重建会把正在 hover 的下拉关掉，也会让动画闪。 */
  function mgSyncFaceCards() {
    var layer = document.getElementById("mgFaceLayer");
    if (!layer) return;
    Array.prototype.forEach.call(layer.querySelectorAll(".mgfc"), function (card) {
      var i = parseInt(card.getAttribute("data-face"), 10);
      /* 卡片不再重建，位置就得在这里跟着当前帧的框走（切帧/换图后 chip 要贴回框的右上角） */
      var rc = MG_RECTS[i];
      if (rc && rc.w > 0 && rc.h > 0) {
        card.style.left = (rc.x + rc.w) + "px";
        card.style.top = rc.y + "px";
        var pl = card.querySelector(".mgplus");
        if (pl) pl.style.top = Math.round(rc.h + 3) + "px";
        /* 「调试」和 ＋ 同一行（并排），高度跟着框走 */
        var dg = card.querySelector(".mgfdbg");
        if (dg) dg.style.top = Math.round(rc.h + 3) + "px";
      }
      var nm = mgFaceChipModel(i);
      var color = nm ? mgModelColor(nm) : "";
      var btn = card.querySelector(".mgfb");
      var tx = card.querySelector(".ddtxt");
      if (tx) tx.textContent = mgFaceChipText(i);
      if (btn) {
        btn.classList.toggle("has", !!color);
        btn.style.borderColor = color || "";
        btn.style.color = color || "";
        btn.title = nm ? ("这张脸用 " + nm + "（点击换）") : "给这张脸选一个模型";
      }
      var ring = card.querySelector(".mgfring");
      if (ring) ring.hidden = !MG_FACE_BUSY[i];
      card.classList.toggle("busy", !!MG_FACE_BUSY[i]);
      var vt = card.querySelector(".mgvtag");
      if (vt) vt.hidden = !mgFaceIsRawView(i);
    });
  }

  /* ---------- 每张脸分到哪个模型 → models payload ---------- */
  function mgFaceModelsPayload() {
    var byName = {}, order = [];
    Object.keys(MG_FACE_MODEL).forEach(function (k) {
      var i = parseInt(k, 10), nm = String(MG_FACE_MODEL[k] || "");
      if (!nm || isNaN(i)) return;
      if (!byName[nm]) { byName[nm] = []; order.push(nm); }
      byName[nm].push(i);
    });
    /* 一张都没手动分配 → 发空数组：**让后端按人脸 embedding 自动决定每张脸用哪个
       dfm**（/api/merge/frame、/api/merge/merged 的自动分配）。
       以前这里退回「左栏选中 = 全部脸」，那等于用同一个模型换所有人 ——
       和「组里的人脸 = 那个组的 dfm」的语义正好相反。 */
    if (!order.length) return [];
    return order.map(function (nm) {
      return { name: nm, faces: byName[nm].sort(function (a, b) { return a - b; }) };
    });
  }
  window.mgFaceModelsPayload = mgFaceModelsPayload;

  function mgFaceSingleModel() {
    var keys = Object.keys(MG_FACE_MODEL);
    if (keys.length) {
      var k = keys.sort(function (a, b) { return Number(a) - Number(b); })[0];
      return String(MG_FACE_MODEL[k] || "");
    }
    try { return mgPreviewModel(); } catch (e) { return ""; }
  }
  window.mgFaceSingleModel = mgFaceSingleModel;

  /* ---------- 右栏参数 → 请求里的 settings ---------- */
  var MG_MASK_INT = { "prd": 1, "dst": 2, "prd×dst": 3 };   /* 位标志：1=prd 2=dst 3=both */
  function mgSegVal(key, dflt) {
    var b = document.querySelector('#p-merge .seg[data-k="' + key + '"] b.on');
    if (!b) return dflt;
    var dv = b.getAttribute("data-v");
    return (dv == null) ? String(b.textContent || "").trim() : String(dv);
  }
  function mgNumVal(id, dflt) {
    var el = document.getElementById(id);
    var v = parseFloat(el ? el.value : "");
    return isFinite(v) ? v : dflt;
  }
  /* 融合方式只有这 4 个（和「模式」下拉的选项、后端白名单三处必须一致）。 */
  var MG_MERGE_BLENDS = ["overlay", "hist-match", "seamless", "seamless-hist-match"];

  function mgCmpSettings() {
    var ct = String((document.getElementById("mgColor") || {}).value || "mt");
    var mm = mgSegVal("mask", "prd×dst") || "prd×dst";
    var maskInt = (MG_MASK_INT[mm] == null) ? 3 : MG_MASK_INT[mm];
    var erode = mgNumVal("mgErode", 0);
    var blur = mgNumVal("mgBlur", 4);
    /* 融合方式的真源 = 「模式」下拉（4 个）。右边那个两档分段（叠加/无缝）
       只是它的快捷方式，两边由 mgSyncBlendSeg 保持同一个值。
       以前 blend 只读分段、mode 读下拉，而后端以 blend 为准 —— 下拉里选
       hist-match 画面一点不动（用户报的「选了没反应」就是这个）。 */
    var modeSel = String((document.getElementById("mgMode") || {}).value || "");
    if (MG_MERGE_BLENDS.indexOf(modeSel) < 0) modeSel = "";
    var blend = modeSel || mgSegVal("blend", "seamless");
    if (MG_MERGE_BLENDS.indexOf(blend) < 0) blend = "seamless";
    return {
      /* —— 新契约 —— */
      ct_mode: ct,
      blend: blend,
      mask_mode: maskInt,
    /* 遮罩来源单独发：后端只认这个字段（mask_mode 整数分不清 XSeg 与 XSegLite） */
    mask_source: mgSegVal("seg", "XSegLite"),
      mask_erode: erode,
      mask_blur: blur,
      /* —— 旧字段兜底：后端还是老版本时认这一组 —— */
      mode: modeSel || blend,
      color_transfer: ct,
      mask_source: mgSegVal("seg", "XSegLite"),
      mask_style: mm,
      erode_mask: erode,
      blur_mask: blur,
      detector: mgDetectorName(),
      face_type: String((document.getElementById("mgFaceType") || {}).value || "whole_face")
    };
  }
  window.mgCmpSettings = mgCmpSettings;

  /* 「模式」下拉 ↔ 右边的「融合方式」两档分段：同一个值，两边一起亮。
     overlay → 叠加，其它三种（hist-match / seamless / seamless-hist-match）→ 无缝，
     因为分段只有这两档 —— 它就是个快捷方式，真值永远在下拉里。 */
  function mgSyncBlendSeg(val) {
    var v = String(val || "");
    if (MG_MERGE_BLENDS.indexOf(v) < 0) return;
    var seg = document.querySelector('#p-merge .seg[data-k="blend"]');
    /* 融合方式分段已按用户要求删掉（与「模式」下拉重复），有元素才同步，没有就什么都不做。 */
    if (!seg) return;
    var want = (v === "overlay") ? "overlay" : "seamless";
    Array.prototype.forEach.call(seg.querySelectorAll("b"), function (b) {
      b.classList.toggle("on", (b.getAttribute("data-v") || "") === want);
    });
  }
  window.mgSyncBlendSeg = mgSyncBlendSeg;

  function mgCmpFrameIdx() {
    if (window.MergeTimeline && window.MergeTimeline.getFrame) {
      var n = window.MergeTimeline.getFrame();
      if (typeof n === "number" && isFinite(n)) return Math.max(0, Math.round(n));
    }
    var f = document.getElementById("mtlFrame");
    return Math.max(0, parseInt(f ? f.value : 0, 10) || 0);
  }
  window.mgCmpFrameIdx = mgCmpFrameIdx;

  /* 画面右下角的小字「正在合成…」：不遮画面、不吃点击 */
  function mgCmpNote(on, text) {
    var el = document.getElementById("mgCmpNote");
    if (!el) return;
    if (!on) { el.hidden = true; return; }
    el.textContent = text || "正在合成…";
    el.hidden = false;
  }
  window.mgCmpNote = mgCmpNote;

  function mgCmpBusyAll(ids) {
    MG_FACE_BUSY = {};
    (ids || []).forEach(function (i) { MG_FACE_BUSY[i] = true; });
    mgSyncFaceCards();
  }
  function mgCmpBusyClear() {
    MG_FACE_BUSY = {};
    mgSyncFaceCards();
  }

  /* ---------- 重算当前帧 ----------
     主路：POST /api/merge/frame（settings + models 分配）
     兜底：POST 不认（404/405/501）→ 老 GET /api/merge/frame/<n>?w=900
           再不行（视频模式）→ 老 POST /api/merge/preview
     每次新请求都 ++seq 并 abort 上一发：旧结果永远盖不到新画面上。 */
  function mgCmpNow(opt) {
    opt = opt || {};
    var mv = window.__mv || {};
    var frame = (opt.frame == null) ? mgCmpFrameIdx() : Math.max(0, parseInt(opt.frame, 10) || 0);
    if (!mv.path && !mgVidPath()) return Promise.resolve(null);

    var seq = ++MG_CMP.seq;
    if (MG_CMP.ctl) { try { MG_CMP.ctl.abort(); } catch (e) {} MG_CMP.ctl = null; }
    /* 请求体全在 mgCmpBody 里：合成只有这一种出图方式（没有视图字段可加了） */
    var body = mgCmpBody(frame);
    var ctrl = (typeof AbortController === "function") ? new AbortController() : null;
    MG_CMP.ctl = ctrl;
    mgCmpNote(true);
    var t0 = (window.performance && performance.now) ? performance.now() : Date.now();

    return fetch("/api/merge/frame", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
      signal: ctrl ? ctrl.signal : undefined
    }).then(function (r) {
      return r.json().then(function (d) {
        if (!r.ok || !d) {
          var err = new Error((d && d.detail) || ("HTTP " + r.status));
          err.status = r.status;
          /* 结构化错误（后端自己的 {ok:false,error}）和「路由不存在」要分得开：
             前者是参数/模型真错了，必须说出来；后者才轮到老接口兜底。 */
          err.api = !!(d && typeof d.ok === "boolean");
          throw err;
        }
        return d;
      }, function () {
        var err = new Error("HTTP " + r.status);
        err.status = r.status;
        err.api = false;                 /* 响应体不是 JSON：多半是老后端/没这个路由 */
        throw err;
      });
    }).then(function (d) {
      if (seq !== MG_CMP.seq) return null;            /* 过期响应：丢掉 */
      MG_CMP.ctl = null;
      MG_CMP.lastMs = ((window.performance && performance.now) ? performance.now() : Date.now()) - t0;
      mgCmpBusyClear();
      mgCmpNote(false);
      var png = d.png || d.image || d.data || d.frame_png || "";
      /* 过期响应上面已经丢了：到这儿就一定是当前这一帧，直接贴 */
      if (png) { window.mgShowOverlay(png); mgHint(""); }
      /* 合成图到了：它只换底图，框一个都不动（框是 detect 那条线画的） */
      if (window.mgTimingMark) window.mgTimingMark("t_frame");
      /* 每张脸被判给了哪个组 → chip 上直接显示（用户手动改则覆盖） */
      mgApplyFacePlan(d);
      return d;
    }).catch(function (e) {
      if (seq !== MG_CMP.seq) return null;
      if (e && e.name === "AbortError") return null;   /* 被新请求取代，不算失败 */
      MG_CMP.ctl = null;
      mgCmpBusyClear();
      mgCmpNote(false);
      var stt = e && e.status;
      if (stt === 404 || stt === 405 || stt === 501 || stt === 400 || stt === 422) {
        /* 后端自己给的 4xx（{ok:false,error}）= 参数/模型/检测器真错了。
           以前这里一并退回老 GET，老 GET 不吃 settings/model —— 结果是"点了个
           没导出 .dfm 的模型 / 传了个不存在的检测器"，画面悄悄换成**默认参数**那一张，
           错误被吞掉。现在只有「路由不存在」（响应体不是我们的结构化 JSON）才兜底。 */
        if (!(e && e.api)) return mgCmpLegacy(frame);
        mgCmpNote(false);
        mgHint(String(e.message || "合成失败").slice(0, 80));
        return null;
      }
      mgCmpNote(false);
      mgHint(String((e && e.message) || "合成失败").slice(0, 80));
      return null;
    });
  }
  window.mgCmpNow = mgCmpNow;

  function mgCmpLegacy(frame) {
    var seq = MG_CMP.seq;                    /* 不 ++：还是同一发 */
    mgCmpNote(true, "正在合成…（旧接口）");
    return fetch("/api/merge/frame/" + frame + "?w=900").then(function (r) {
      if (!r.ok) throw new Error("HTTP " + r.status);
      return r.json();
    }).then(function (d) {
      if (seq !== MG_CMP.seq) return null;
      mgCmpNote(false);
      if (d && d.png) { window.mgShowOverlay(d.png); mgHint(""); }
      return d;
    }).catch(function () {
      if (seq !== MG_CMP.seq) return null;
      mgCmpNote(false);
      /* 逐帧接口也没有：视频模式退回老的预览接口（它自己也带 settings/models） */
      if (mgVideoMode() && window.mgPreviewNow) {
        return window.mgPreviewNow({ force: true, fromFallback: true });
      }
      return null;
    });
  }

  /* 参数改一下 → 150ms 防抖 → 重算一击。连拖滑块 10 次只发最后一次。 */
  function mgCmpSoon() {
    if (MG_CMP.wait) clearTimeout(MG_CMP.wait);
    MG_CMP.wait = setTimeout(function () {
      MG_CMP.wait = null;
      mgCmpNow();
    }, 150);
  }
  window.mgCmpSoon = mgCmpSoon;

  /* 「原图」那一态：后端 /api/merge/frame 默认出的是合成结果，
     原片要显式 raw=1（契约里 raw=1 或 compose=0 就是不合成）。
     老后端不认 raw → 404/400 时退回 mvFrame 那条路，画面不至于空着。 */
  window.mgRawFrame = function (n) {
    n = Math.max(0, parseInt(n, 10) || 0);
    var seq = ++MV_SEQ.orig;
    return fetch("/api/merge/frame/" + n + "?w=900&raw=1&fmt=png").then(function (r) {
      if (!r.ok) throw new Error("HTTP " + r.status);
      return r.json();
    }).then(function (d) {
      if (seq !== MV_SEQ.orig) return null;
      if (d && d.png) window.mvStage(d.png);
      return d;
    }).catch(function () { return window.mvFrame(n); });
  };

  /* ---------- 底图跟帧：项目 / 帧目录模式下一帧一发原图 ----------
     底图是 img.mgfrm（原片），合成结果贴在上面的 #mgPreview。以前底图只在本帧
     「无人脸」时才换（mgNoFaceFrame → mgRawFrame）：有人脸就只贴合成层，于是
       · 合成慢一拍（项目模式 300~500ms）时画面钉在上一帧；
       · 合成失败 / 没选模型 / 被取消时，画面**永远**停在上一帧 —— 看起来就是
         「点时间轴不刷新」；
       · 时间轴播放时更彻底：播放中本来就不发合成，整屏冻在第一帧。
     现在底图和合成解耦：帧一变先换底图（只解码 + 缩放 + 编码，不检测不换脸），
     合成到了再盖上去。单发在飞时只记「最新目标」，回来补最后一帧 —— 拖时间轴
     连发几十帧也只补最后一张，不会排队排到画面走远。 */
  var MG_BASE = { want: -1, busy: false };
  function mgBaseFrame(n) {
    if (mgVidPath()) return;                  /* 视频模式底图是 <video> 自己 */
    MG_BASE.want = Math.max(0, parseInt(n, 10) || 0);
    if (MG_BASE.busy) return;                 /* 在飞：记下最新目标，回来再补 */
    mgBasePump();
  }
  function mgBasePump() {
    if (MG_BASE.want < 0) return;
    var n = MG_BASE.want;
    MG_BASE.want = -1;
    MG_BASE.busy = true;
    function done() {
      MG_BASE.busy = false;
      if (MG_BASE.want >= 0) mgBasePump();
    }
    var p = null;
    try { p = window.mgRawFrame(n); } catch (e) { p = null; }
    if (p && p.then) p.then(done, done); else done();
  }
  window.mgBaseFrame = mgBaseFrame;

  /* ---------- 帧变化编排：清框 → 检测（画框）→ 合成（只换底图） ----------
     一条帧变化只从这里走。三件事，不是两次等待：
       ① 同步清框（canvas 描边 + chip 层），这一代检测一并作废；
       ② 立刻发 /api/merge/detect —— 回来就画框 + chip，t_draw 紧跟 t_detect；
       ③ 检测结论「有人脸」才发合成（项目模式 /api/merge/frame，视频模式走预览），
          合成回包只替换底图，框一个都不碰。
     为什么合成要等检测这一拍：本帧没人脸时合成毫无意义（后端也只是回原图），
     而且先发 frame 会把它（内部含检测 + dfm + 遮罩 + 编码，~400ms）排在
     自己那一发检测前面 —— 看起来就像「检测很慢」，其实是被合成堵住了。
     检测本身是 3~20ms 的活，等它一拍不亏。
     兜底：检测冷启动（首次加载 onnx 要几秒）时不能让画面钉住 —— MG_DET_GATE
     到点就照发合成；那次检测晚回来照样把框画上（帧没换的话）。 */
  /* 等检测结论的兜底：第一档 800ms，如果那一发检测**还在飞**就再等一档，
     最多等到 3000ms。只在检测彻底挂住（或冷启动加载 onnx 要好几秒）时才放行合成 ——
     后端一忙，检测 700ms 才回来是常事，那种情况必须等它，不然又白跑一轮 dfm。 */
  var MG_DET_GATE = 800;
  var MG_DET_GATE_MAX = 3000;
  var MG_CHAIN = { seq: 0, frame: -1, gate: null, gateOpen: true, gateFrom: 0 };

  function mgComposeFrame(n) {
    var videoMode = !!(window.mgVideoMode && window.mgVideoMode() &&
                       !(window.__mgVideo || {}).streamDown);
    if (videoMode) {
      /* 视频模式：底图是 <video> 自己，合成结果只是贴上去的那一张 overlay */
      if (window.mgPreviewSoon) window.mgPreviewSoon();
      return null;
    }
    return window.mgCmpNow ? window.mgCmpNow({ frame: n }) : null;
  }

  /* 本帧无人脸：不发合成，底图换成原图，框 / chip 都不画 */
  function mgNoFaceFrame(n) {
    MG_FRAME_T.noFace = true;
    if (window.mgShowOverlay) window.mgShowOverlay("");     /* 撤掉上一帧留下的合成层 */
    if (window.mgCmpNote) window.mgCmpNote(false);
    if (window.mgHint) window.mgHint("本帧无人脸");
    var videoMode = !!(window.mgVideoMode && window.mgVideoMode());
    if (!videoMode && window.mgRawFrame) {
      /* 轻量取帧：GET /api/merge/frame/<n>?raw=1 —— 只解码 + 编码，不检测不换脸。
         它是「原图」，不是合成请求（raw=1 那条分支在 _frame_payload 最前面就返回了）。 */
      var p = null;
      try { p = window.mgRawFrame(n); } catch (e) { p = null; }
      /* 原图落地会把舞台中间的小字收起来（mvStage 里 hint.display=none），
         所以状态行等它回来再补一次 —— 帧/无人脸结论没变才补。 */
      if (p && p.then) {
        p.then(function () {
          if (MG_CHAIN.frame === n && MG_FRAME_T.noFace && window.mgHint) window.mgHint("本帧无人脸");
        }, function () {});
      }
    }
    return true;
  }
  window.mgNoFaceFrame = mgNoFaceFrame;

  function mgFrameChange(n, opt) {
    opt = opt || {};
    var mv = window.__mv || {};
    n = Math.max(0, parseInt(n, 10) || 0);
    if (!mv.path && !mgVidPath()) return null;
    /* 换帧了：开着的那张脸的六宫格是上一帧的，直接收掉（别让人对着旧证据调参） */
    if (window.mgFaceDebugClose) window.mgFaceDebugClose();
    /* 同一帧的连环召唤（进页面 init 归零 + open 补帧会连着叫三次）并成一发：
       上一发还在飞（gate 关着）就不用再清一次框、再打一发检测 */
    if (!opt.force && MG_CHAIN.frame === n && !MG_CHAIN.gateOpen) return null;
    var seq = ++MG_CHAIN.seq;
    MG_CHAIN.frame = n;
    MG_CHAIN.gateOpen = false;               /* 合成这道门先关上，等检测结论 */
    if (MG_CHAIN.gate) { clearTimeout(MG_CHAIN.gate); MG_CHAIN.gate = null; }

    /* ① 帧要变了：先把框清干净（同步，不等任何回包） */
    window.mgInvalidateBoxes(opt.why || "frame", n);

    /* ①-b 底图立刻跟到这一帧。合成还在路上（或根本不会来：无人脸 / 没模型 /
       失败 / 播放中）时，画面也必须先动 —— 否则时间轴点了像没点。 */
    if (!mgVidPath()) {
      mgHideOverlay();                       /* 上一帧的合成图不能盖着新底图 */
      mgBaseFrame(n);
    }

    /* 播放中：只清框 + 一发「在飞就跳过」的检测，绝不发合成。
       一帧一发 4 秒的合成会把 worker 队列塞死（原来项目模式播放就是这样），
       画面反而更卡；而检测结果几乎必然过期（见 mvDetect 的 outdated 分支），
       所以这一拍宁可没有框 —— 停下来那一发才是准的。 */
    var playing = !!((window.MergeTimeline && window.MergeTimeline.isPlaying &&
                      window.MergeTimeline.isPlaying()) ||
                     (window.__mgVideo && window.__mgVideo.playing));
    if (playing) {
      MG_CHAIN.gateOpen = true;              /* 没有待发的合成，门开着 */
      if (!MG_DET.inflight) {
        try { window.mvDetect(n); } catch (e) {}
      }
      return null;
    }

    /* 视频模式：先让 <video> 走到这一帧（画面本体）；seeked 里的预览被上面的门挡住 */
    if (window.mgVideoMode && window.mgVideoMode() && !(window.__mgVideo || {}).streamDown) {
      try { window.mgVideoSeek(n); } catch (e) {}
    }

    var fired = false;
    function openGate() {
      MG_CHAIN.gateOpen = true;
      if (MG_CHAIN.gate) { clearTimeout(MG_CHAIN.gate); MG_CHAIN.gate = null; }
    }
    function composeOnce() {
      if (fired || seq !== MG_CHAIN.seq) return;
      fired = true;
      openGate();
      mgComposeFrame(n);
    }

    /* ② 检测先发（3~20ms 的活）：回来就画框，然后才轮到合成 */
    var det = null;
    if (!opt.skipDetect) {
      try { det = window.mvDetect(n); } catch (e) { det = null; }
    }
    function afterDetect(res) {
      if (seq !== MG_CHAIN.seq) return;               /* 已经换到别的帧了：这一发作废 */
      if (res && res.ok && !(res.boxes || []).length) {
        fired = true;
        openGate();
        mgNoFaceFrame(n);                             /* ③-a 没人脸：不合成，回原图 */
        return;
      }
      composeOnce();                                  /* ③-b 有人脸：发合成 */
    }
    if (det && det.then) det.then(afterDetect, function () { afterDetect(null); });
    else afterDetect(null);
    MG_CHAIN.gateFrom = mgNowT();
    armGate();
    return det;

    function armGate() {
      MG_CHAIN.gate = setTimeout(function () {
        MG_CHAIN.gate = null;
        if (seq !== MG_CHAIN.seq) return;
        var waiting = !!(MG_DET.inflight && MG_DET.inflight === det);
        if (waiting && (mgNowT() - MG_CHAIN.gateFrom) < MG_DET_GATE_MAX) {
          armGate();                      /* 检测还在飞：再等一档 */
          return;
        }
        composeOnce();                    /* 检测挂了/太慢：照发，画面不能一直钉着 */
      }, MG_DET_GATE);
    }
  }
  window.mgFrameChange = mgFrameChange;

  /* 进页面 / 切回合成页：把当前帧补齐。同一帧已经画过框就不清（重进页面不该闪一下） */
  function mgCmpSync(opt) {
    opt = opt || {};
    var mv = window.__mv || {};
    if (!mv.path && !mgVidPath()) return;
    var n = mgCmpFrameIdx();
    if (MG_CHAIN.frame === n && MG_FRAME_T.tDraw) {
      MG_CHAIN.gateOpen = true;
      mgComposeFrame(n);
      return;
    }
    mgFrameChange(n, { why: "sync", skipDetect: !!opt.skipDetect });
  }
  window.mgCmpSync = mgCmpSync;

  /* ---------- 后端自动判定的结果 → 画面上每张脸的 chip ---------- */
  /* 合成回包里的 faces 是「这一帧每张脸被判给了哪个组/dfm + 相似度」的**权威答案**
     （auto=true 表示来自人脸库的 embedding 对比，false 表示用户手动指定过）。
     整份替换而不是合并：这一帧有几张脸、各自像谁，答案就在这一次响应里；
     上一帧的残留必须清掉，否则换人之后 chip 还挂着上一个模型。 */
  function mgApplyFacePlan(d) {
    var list = (d && d.faces) || [];
    if (!list.length) {
      /* 这一帧没有 faces = 走了单模型老路径（库里还没有组 / 老后端）。
         上一帧自动判出来的 chip 必须清掉，否则换人之后还挂着上个模型。 */
      if (d && d.ok && Object.keys(MG_FACE_AUTO).length) {
        MG_FACE_AUTO = {};
        mgSyncFaceCards();
        if (window.mvRedrawBoxes) window.mvRedrawBoxes();
      }
      return;
    }
    var next = {};
    list.forEach(function (f) {
      if (!f) return;
      var raw = (f.idx != null) ? f.idx : f.face;
      var i = parseInt(raw, 10);
      if (isNaN(i) || i < 0) return;
      var sim = (f.similarity == null) ? null : Number(f.similarity);
      next[i] = {
        dfm: String(f.dfm || f.model || ""),
        group: String(f.group || ""),
        groupName: String(f.group_name || ""),
        similarity: (sim == null || isNaN(sim)) ? null : sim,
        matched: f.matched !== false,
        auto: f.auto !== false
      };
    });
    MG_FACE_AUTO = next;
    mgSyncFaceCards();
    if (window.mvRedrawBoxes) window.mvRedrawBoxes();   /* 框色跟着判出来的模型走 */
  }
  window.mgApplyFacePlan = mgApplyFacePlan;

  /* ---------- 分配模型 → 立刻重算 ---------- */
  function mgAssignFace(i, name) {
    name = String(name || "");
    /* 手动指定 = 覆盖自动判定：把这一张从 MG_FACE_AUTO 里摘掉，
       否则下一次回包之前 chip 还会显示自动判出来的那个名字。 */
    delete MG_FACE_AUTO[i];
    if (name) { MG_FACE_MODEL[i] = name; delete MG_FACE_NOSWAP[i]; }
    else { delete MG_FACE_MODEL[i]; MG_FACE_NOSWAP[i] = true; }   /* 「不换脸」→ no_swap_faces */
    MG_FACE_BUSY[i] = true;
    /* 先用 canvas 换色 + 卡片换字（不等网络），再发重算 */
    window.mvRedrawBoxes();
    mgSyncFaceCards();
    var p = mgCmpNow();
    if (p && p.catch) p.catch(function () {});
  }
  window.mgAssignFace = mgAssignFace;

  /* ---------- 「＋」：把这张脸加进人脸数据库 ----------
     不再走 /api/merge/faces/extract（那个要 project、要整帧重检测，视频模式下
     没项目就直接 400）。现在点「＋」= 「我已经知道是哪张脸了，把这块存进库」：
     前端把这一帧的 68 点（/api/merge/detect 回包里就带着，和 extract 同一条
     2D106 链路）连同 frame/dfm 一起 POST /api/merge/faces/add，后端用**同一条
     68 点仿射**裁成该 dfm 的输入尺寸落盘。所以同一帧同一张脸，走 extract 和
     走 add 存下来的 PNG 逐像素相同。
     project 可有可无：没有就用默认库 <工作区>/merge，后端自己建。 */
  var MG_ADD = {};            /* 脸序号 → {busy, queue}：连点排队，一次只飞一发 */
  var MG_TOAST_SEEN = {};     /* 同一条提示 2 秒内只弹一次（错误不刷屏） */

  function mgToastOnce(key, title, sub, type, ms) {
    var now = Date.now();
    if (now - (MG_TOAST_SEEN[key] || 0) < 2000) return null;
    MG_TOAST_SEEN[key] = now;
    return toast(title, sub, type, ms);
  }
  window.mgToastOnce = mgToastOnce;

  /* 同一次操作里只允许出现**一条**错误：2 秒内的第二、第三条直接丢掉。
     以前「缺少 68 点」和「还没有素材」各用各的 key，两条会一起弹，还互相矛盾
     （画面里明明有视频）。现在「点＋」和「导出」的前置失败都走这里。 */
  var MG_ERR_AT = 0;
  function mgToastErr(key, title, sub, type, ms) {
    var now = Date.now();
    if (now - MG_ERR_AT < 2000) return null;
    if (now - (MG_TOAST_SEEN[key] || 0) < 2000) return null;
    MG_ERR_AT = now;
    MG_TOAST_SEEN[key] = now;
    /* toast() 是拼 innerHTML 的：服务端消息里带路径，尖括号别让它变成标签 */
    return toast(title, String(sub || "").replace(/[<>]/g, ""), type || "warn", ms || 5000);
  }
  window.mgToastErr = mgToastErr;

  /* 「现在到底能不能拿到这一帧的图」—— 问后端，不问变量。
     判定权在 /api/merge/faces/source：项目模式看项目帧、视频模式看合成器打开着
     的素材，两样都没有才算「没有素材」。以前只看 window.__mergeInputDir ——
     视频模式下它永远是空串。导出前的前置检查（mgNeedProjectDir）只认这个口径。
     结果缓存 1.2 秒（连点、重试不重复发请求）；
     老后端没这个接口就退回本地判断，绝不因为问不到就把用户拦下来。 */
  var MG_SRC = { at: 0, data: null, inflight: null };
  function mgSourceForget() {
    MG_SRC.at = 0; MG_SRC.data = null; MG_SRC.inflight = null;
  }
  window.mgSourceForget = mgSourceForget;

  function mgSrcHint(info) {
    info = info || {};
    var why = String(info.why || "");
    if (why) return why;                 /* 后端已经说了「当前是哪种模式、缺哪一个」 */
    if (info.mode === "project") return "当前是项目模式：项目里这一帧取不到图";
    if (info.mode === "video") return "当前是视频模式：还没打开素材 —— 先在左栏导入一段视频";
    return "当前既没有打开视频、也没有项目目录 —— 先导入一段视频或打开 DFL 项目";
  }

  function mgSourceInfo() {
    var now = Date.now();
    if (MG_SRC.data && now - MG_SRC.at < 1200) return Promise.resolve(MG_SRC.data);
    if (MG_SRC.inflight) return MG_SRC.inflight;
    var project = window.__mergeInputDir || "";
    var frame = mgCmpFrameIdx();
    MG_SRC.inflight = fetch("/api/merge/faces/source?project=" + encodeURIComponent(project) +
                            "&frame=" + frame)
      .then(function (r) { return r.json(); })
      .then(function (d) {
        return {
          ok: !!(d && d.ok !== false),
          available: !!(d && d.available),
          mode: String((d && d.mode) || ""),
          project: String((d && d.project) || ""),
          default_project: String((d && d.default_project) || ""),
          material: String((d && d.material) || ""),
          why: String((d && d.why) || ""),
          frame: frame
        };
      })
      .catch(function () {
        var video = mgVidPath();
        return { ok: false, available: !!(video || project),
                 mode: video ? "video" : (project ? "project" : "none"),
                 project: project, default_project: "", material: video,
                 why: "", frame: frame };
      })
      .then(function (info) {
        MG_SRC.data = info; MG_SRC.at = Date.now(); MG_SRC.inflight = null;
        return info;
      });
    return MG_SRC.inflight;
  }
  window.mgSourceInfo = mgSourceInfo;

  /* 导出（整片 / 标准项目）还要一个真正的项目目录：视频模式得先抽帧 + 对齐。 */
  function mgNeedProjectDir() {
    return mgSourceInfo().then(function (info) {
      var dir = String(info.project || window.__mergeInputDir || "");
      if (dir) return { info: info, dir: dir };
      if (!info.available) {
        mgToastErr("need-source", "还没有素材", mgSrcHint(info), "warn", 5000);
        return null;
      }
      mgToastErr("need-project", "当前是视频模式：还没有能导出的项目",
                 "视频还没抽帧 + 对齐成项目（先在左栏导入并等它准备完），导出要项目目录",
                 "warn", 6000);
      return null;
    });
  }
  window.mgNeedProjectDir = mgNeedProjectDir;


  function mgPlusBtn(i) {
    var layer = document.getElementById("mgFaceLayer");
    if (!layer) return null;
    return layer.querySelector('.mgfc[data-face="' + i + '"] .mgplus');
  }

  /* 连点 10 次要真的加 10 张（用户的意图），所以按钮不 disabled ——
     只是排队：请求在飞的时候点，就把这几次记进 queue，落地后接着发下一发。 */
  function mgAddFaceToDb(i, btn) {
    i = parseInt(i, 10);
    if (isNaN(i)) return;
    var q = MG_ADD[i] || (MG_ADD[i] = { busy: false, queue: 0 });
    q.queue += 1;
    var b0 = btn || mgPlusBtn(i);
    if (b0) b0.classList.add("busy");
    if (q.busy) return;
    mgAddFaceNext(i);
  }
  window.mgAddFaceToDb = mgAddFaceToDb;

  /* 后端给的业务失败（200 + ok:false）：一条提示、说清原因，绝不叠加第二条。 */
  function mgAddFaceFail(d) {
    var why = String((d && d.reason) || "");
    var detail = String((d && d.detail) || "").slice(0, 140);
    if (why === "no_landmarks68") {
      mgToastErr("add-no68", "这张脸没算出 68 点",
                 detail || "后端跑不出 2D106 的 68 点 —— 换一帧或重新检测再点", "warn", 5000);
    } else if (why === "no_face") {
      mgToastErr("add-noface", "这一帧没检测到脸", detail || "换一帧再点", "warn", 5000);
    } else if (why === "detect_failed") {
      mgToastErr("add-detfail", "服务端检测失败", detail || "检测器没跑起来", "error", 5000);
    } else if (why === "no_group_match") {
      /* 这张脸谁的组都不像（相似度都 < 阈值）→ 后端没落库，等用户手动指定 */
      mgToastErr("add-nogroup", "这张脸和现有分组都不像",
                 "请手动拖到某个组：先在框上的 chip 里给它选一个模型，再点＋",
                 "warn", 6000);
    } else {
      mgToastErr("add-" + why, "没有加到人脸库",
                 detail || why || "后端没有给出原因", "warn", 5000);
    }
  }

  function mgAddFaceNext(i) {
    var q = MG_ADD[i];
    if (!q) return;
    var btn = mgPlusBtn(i);
    if (q.queue <= 0) {
      if (btn) btn.classList.remove("busy");
      return;
    }
    q.queue -= 1;
    q.busy = true;
    if (btn) btn.classList.add("busy");

    var rec = MG_FACE_RAW[i] || null;
    var box = rec && rec.box ? rec.box.slice() : null;
    var kps68 = rec && rec.kps68 ? rec.kps68 : null;
    var has68 = !!(kps68 && kps68.length === 68);
    var frame = mgCmpFrameIdx();
    var video = mgVidPath() || "";

    function done() {
      q.busy = false;
      var b = mgPlusBtn(i);
      if (b && q.queue <= 0) b.classList.remove("busy");
      if (q.queue > 0) mgAddFaceNext(i);
    }

    /* 只把我们**真的知道**的东西发出去：
         有 kps68 → 直接带（快路径，前端算好的 68 点原样落库）；
         没有 / 不是 68 个 → 就不带，让后端从这一帧的图上自己跑
         「检测 → 2D106 → 68 点仿射裁剪」。
       绝不因为手里没有 kps68 就在这里弹错返回 —— 老后端（/api/merge/detect
       不回 kps68）时那样等于＋什么都没干，而用户看到的是「缺少 68 点」和
       「还没有素材」两条自相矛盾的提示（画面里明明有视频）。 */
    /* 落进哪一组**不由前端决定**（「提取到 <dfm>」下拉已经删掉）：
       后端会算这张脸的 embedding，和每个组的均值向量比余弦，像谁就归谁
       （响应里的 group / group_name / similarity / matched）。
       唯一带过去的偏好是「这张脸在框上的 chip 里被手动指定过模型」——
       那只是**匹配不上时的兜底**，不是目标组。 */
    var payload = {
      /* 有项目用项目库；没有留空 → 后端用默认库 <工作区>/merge 自己建 */
      project: window.__mergeInputDir || "",
      frame: frame,
      /* 同一帧只留一张脸：idx = 这张脸在这一帧里是第几张（多脸帧各存各的） */
      idx: i,
      detector: mgDetectorName() || "",
      source: { video: video, frame: frame }
    };
    var chip = String(MG_FACE_MODEL[i] || "");
    if (chip) payload.dfm = chip;
    if (has68) payload.kps68 = kps68;
    if (box) payload.box = box;

    fetch("/api/merge/faces/add", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    }).then(function (r) {
      return r.json().then(function (d) {
        /* no_face / no_landmarks68 / detect_failed / no_group_match 这类是业务结果
           （200 + ok:false），当成正常返回交给下面按 reason 说话，
           别报成「失败 HTTP 200」。 */
        if (d && d.ok === false && d.reason) return d;
        if (r.ok && d && d.ok !== false) return d;
        var err = new Error((d && (d.detail || d.error)) || ("HTTP " + r.status));
        err.status = r.status;
        throw err;
      }, function () {
        var err = new Error("HTTP " + r.status);
        err.status = r.status;
        throw err;
      });
    }).then(function (d) {
      done();
      if (d && d.ok === false) {
        mgAddFaceFail(d);
        return;
      }
      var faces = (d && d.faces) || [];
      var hit = faces[0] || null;
      var fid = (hit && hit.id) ? String(hit.id) : "";
      /* landmarks_from=server 说明这 68 点是后端自己补算的（前端手里没有） */
      var how = (d && d.landmarks_from === "server") ? " · 服务端补算 68 点" : "";
      /* 落进哪一组是后端按 embedding 判的：group_name 组名、similarity 余弦相似度。
         auto=true = 「和组均值比出来最像的那个」；chip 兜底时 source="chip"。 */
      var where = String((d && d.group_name) || "");
      var sim = (d && d.similarity != null) ? Number(d.similarity) : null;
      var why = (sim == null || isNaN(sim)) ? "" : (" · 相似度 " + sim.toFixed(2));
      var by = (d && d.auto) ? "（自动判定）"
             : ((d && d.source === "chip") ? "（按 chip 的模型）" : "");
      /* 后端按 (video, frame, idx) 去重：已有就回 replaced=true —— 那是**更新**，
         不能再报「已加入」（否则连点 5 次会看到 5 条「已加入」却只有 1 张脸）。 */
      if (d && d.replaced) {
        toast("已更新该帧的人脸", fid + (where ? " · " + where : "") + by + why + how,
              "success");
      } else {
        toast("已加入人脸数据库", (where ? fid + " · " + where : fid) + by + why +
              (d && d.created ? " · 新建了库" : "") + how, "success");
        /* 角标只数「新增」：更新没有让库变大，不该涨 */
        mgDbBadgeBump(1);
      }
      /* chip 立刻显示「这张脸被判给了谁」，不等下一次整帧重算 */
      if (where && d && d.auto) {
        MG_FACE_AUTO[i] = { dfm: String(d.group_dfm || ""), group: String(d.group || ""),
                            groupName: where, similarity: sim, matched: true, auto: true };
        mgSyncFaceCards();
        if (window.mvRedrawBoxes) window.mvRedrawBoxes();
      }
      mgFlyToDb(i, box);
      /* 人脸库开着就地刷新，别让用户以为没加进去 */
      try { if (window.FaceDB && window.FaceDB.refresh && window.FaceDB.isOpen()) window.FaceDB.refresh(); } catch (e) {}
    }).catch(function (e) {
      done();
      /* 拿不到图时后端会把「当前是视频模式/项目模式、缺的是哪一个」写进 detail */
      mgToastErr("add-err", "添加失败", String((e && e.message) || e).slice(0, 140), "error");
    });
  }
  window.mgAddFaceNext = mgAddFaceNext;

  function mgDbBadgeBump(delta) {
    var el = document.getElementById("mgDbBadge");
    if (!el) return 0;
    var n = Math.max(0, (parseInt(el.textContent, 10) || 0) + (delta == null ? 1 : delta));
    el.textContent = String(n);
    el.hidden = n === 0;
    return n;
  }
  window.mgDbBadgeBump = mgDbBadgeBump;

  /* 从当前合成帧里裁这一块当飞行缩略图（纯 dataURL，画布不会跨域污染） */
  function mgCropThumb(i) {
    try {
      var img = document.getElementById("mgPreview");
      if (!img || !img.naturalWidth || img.style.display === "none") return "";
      var r = MG_RECTS[i];
      if (!r) return "";
      var stage = document.getElementById("mgStage");
      var sr = stage.getBoundingClientRect();
      var ix = img.getBoundingClientRect();
      /* 和 object-fit: contain 用同一套映射：屏幕坐标 → 图片像素 */
      var s = Math.min(ix.width / img.naturalWidth, ix.height / img.naturalHeight);
      var ox = ix.left + (ix.width - img.naturalWidth * s) / 2;
      var oy = ix.top + (ix.height - img.naturalHeight * s) / 2;
      var sx = ((sr.left + 6 + r.x) - ox) / s;
      var sy = ((sr.top + 6 + r.y) - oy) / s;
      var sw = r.w / s, sh = r.h / s;
      var S = 44;
      var cv = document.createElement("canvas");
      cv.width = S; cv.height = S;
      var ctx = cv.getContext("2d");
      ctx.drawImage(img, sx, sy, sw, sh, 0, 0, S, S);
      return cv.toDataURL("image/png");
    } catch (e) { return ""; }
  }

  /* 飞入动画：WAAPI，不引库。缩略图从框的位置抛物线飞向顶部「人脸数据库」入口。 */
  function mgFlyToDb(i, box) {
    var entry = document.getElementById("mgDbEntry");
    var stage = document.getElementById("mgStage");
    if (!entry || !stage) return null;
    var sr = stage.getBoundingClientRect();
    var r = MG_RECTS[i] || { x: 0, y: 0, w: 40, h: 40 };
    var S = 44;
    var from = {
      left: sr.left + 6 + r.x + Math.max(0, (r.w - S) / 2),
      top: sr.top + 6 + r.y + Math.max(0, (r.h - S) / 2)
    };
    var to = entry.getBoundingClientRect();
    var fly = document.createElement("div");
    fly.className = "mgfly";
    fly.id = "mgFlyThumb";
    var src = mgCropThumb(i);
    if (src) {
      var im = document.createElement("img");
      im.src = src;
      im.alt = "";
      fly.appendChild(im);
    } else {
      fly.classList.add("ph");
    }
    fly.style.left = Math.round(from.left) + "px";
    fly.style.top = Math.round(from.top) + "px";
    document.body.appendChild(fly);

    var dx = (to.left + to.width / 2) - (from.left + S / 2);
    var dy = (to.top + to.height / 2) - (from.top + S / 2);
    if (fly.animate) {
      var anim = fly.animate([
        { transform: "translate(0,0) scale(1)", opacity: 1, offset: 0 },
        { transform: "translate(" + (dx * 0.62) + "px," + (dy * 0.62 - 34) + "px) scale(.66)", opacity: 1, offset: 0.6 },
        { transform: "translate(" + dx + "px," + dy + "px) scale(.1)", opacity: 0, offset: 1 }
      ], { duration: 640, easing: "cubic-bezier(.4,0,.2,1)" });
      anim.onfinish = function () { if (fly.parentNode) fly.parentNode.removeChild(fly); };
      /* 落点回弹：告诉眼睛「加到这个入口了」 */
      if (entry.animate) {
        try {
          entry.animate([{ transform: "scale(1)" }, { transform: "scale(1.1)" }, { transform: "scale(1)" }],
                        { duration: 300, delay: 560, easing: "ease-out" });
        } catch (e) {}
      }
    } else {
      setTimeout(function () { if (fly.parentNode) fly.parentNode.removeChild(fly); }, 640);
    }
    return fly;
  }
  window.mgFlyToDb = mgFlyToDb;

  /* ---------- 事件绑定（合成器主视图） ---------- */
  function mgBindCmp() {
    var page = document.getElementById("p-merge");
    if (!page) return;

    /* 点框切这张脸的合成/原图：舞台上的指针判定（阈值 4px），只绑一次 */
    if (window.mgBindFacePick) window.mgBindFacePick();
    /* 六宫格上的「返回」/ 点格放大：只绑一次 */
    if (window.mgFaceDebugBind) window.mgFaceDebugBind();

    /* 「模式」下拉（4 个融合方式）和右边那个两档分段共用同一个值：动哪个，
       另一个跟着高亮。以前 blend 只读分段、下拉写 mode，后端以 blend 为准 ——
       下拉里选 hist-match 画面一动不动。 */
    var modeEl = document.getElementById("mgMode");
    if (modeEl && !modeEl.dataset.blendBound) {
      modeEl.dataset.blendBound = "1";
      modeEl.addEventListener("change", function () { mgSyncBlendSeg(modeEl.value); });
      mgSyncBlendSeg(modeEl.value);
    }
    var bseg = page.querySelector('.seg[data-k="blend"]');
    if (bseg && !bseg.dataset.blendBound) {
      bseg.dataset.blendBound = "1";
      bseg.addEventListener("click", function (e) {
        var b = e.target && e.target.closest ? e.target.closest("b") : null;
        if (!b || !bseg.contains(b)) return;
        var v = b.getAttribute("data-v") || "";
        var sel = document.getElementById("mgMode");
        if (sel && MG_MERGE_BLENDS.indexOf(v) >= 0) sel.value = v;
      });
    }

    /* 参数实时预览：右栏任何控件动一下就防抖重算 */
    var right = page.querySelector(".mgright");
    if (right && !right.dataset.cmpBound) {
      right.dataset.cmpBound = "1";
      right.addEventListener("change", mgCmpSoon);
      right.addEventListener("input", mgCmpSoon);
      right.addEventListener("click", function (e) {
        var b = e.target && e.target.closest ? e.target.closest(".seg b") : null;
        if (!b) return;
        var seg = b.closest(".seg[data-k]");
        if (!seg) return;
        var k = seg.getAttribute("data-k");
        if (k === "blend" || k === "mask" || k === "seg" || k === "detect") mgCmpSoon();
      });
    }

    /* 点页面空白处收起 chip 下拉 */
    if (!page.dataset.cmpDocBound) {
      page.dataset.cmpDocBound = "1";
      document.addEventListener("click", function () { mgCloseChipMenus(null); });
      document.addEventListener("keydown", function (e) {
        if (e.key === "Escape") mgCloseChipMenus(null);
      });
    }

    /* 顶部「人脸数据库」入口：**不再**在这里判「有没有素材」。
       project 空着是合法的（视频模式）—— 后端把空串解析成默认库 <工作区>/merge，
       真打不开时由 face_db.js 把后端的 msg 弹出来。以前这里只判 window.__mergeInputDir：
       视频模式下它永远是空串，于是画面里明明有视频，点「人脸数据库」也弹
       「还没有素材」（「＋」那边再弹一条就自相矛盾了）。 */
    var entry = document.getElementById("mgDbEntry");
    if (entry && !entry.dataset.bound) {
      entry.dataset.bound = "1";
      entry.addEventListener("click", function () {
        if (!window.FaceDB || !window.FaceDB.open) {
          toast("人脸数据库没有加载", "face_db.js 版本不匹配，刷新页面再试", "error");
          return;
        }
        window.FaceDB.open({
          project: window.__mergeInputDir || "",   /* 空 = 后端落到默认库 */
          mode: "export",
          allowExtract: true,
          onCancel: function () { if (window.mgStep) window.mgStep("main"); }
        });
      });
    }


    /* 色域迁移存着老值（lut / idt / sot-m …）时回落到 mt，别让下拉空着 */
    var color = document.getElementById("mgColor");
    if (color) {
      var ok = Array.prototype.some.call(color.options, function (o) { return o.value === color.value; });
      if (!ok) {
        color.value = "mt";
        try { color.dispatchEvent(new Event("change", { bubbles: true })); } catch (e) {}
      }
    }

    /* 模式下拉只剩 4 个**融合方式**：本地存着老值（debug / original / raw-*）时
       回落到 seamless，别让下拉空着、也别把视图名当融合方式发给后端（会 400）。 */
    var modeSel = document.getElementById("mgMode");
    if (modeSel) {
      var okm = Array.prototype.some.call(modeSel.options, function (o) { return o.value === modeSel.value; });
      if (!okm) {
        modeSel.value = "seamless";
        try { modeSel.dispatchEvent(new Event("change", { bubbles: true })); } catch (e) {}
      }
    }
  }
  window.mgBindCmp = mgBindCmp;

  /* 遮罩绘制：把 MaskProcessor 的 WebUI 直接嵌进来 */
  function mkOpenEditor() {
    var inp = document.getElementById("mkDir");
    var frame = document.getElementById("mkFrame");
    var empty = document.getElementById("mkEmpty");
    var st = document.getElementById("mkState");
    if (!inp || !frame) return;
    var dir = String(inp.value || "").trim();
    if (st) { st.classList.remove("on"); st.innerHTML = "<i></i>连接中…"; }
    fetch("/api/project/open", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ path: dir })
    }).then(function (r) {
      if (!r.ok) return r.json().then(function (d) { throw new Error(d.detail || "打开失败"); });
      return r.json();
    }).then(function (d) {
      frame.src = "/mask-editor/index.html?dir=" + encodeURIComponent(d.path);
      if (empty) empty.classList.add("hide");
      if (st) { st.classList.add("on"); st.innerHTML = "<i></i>" + d.count + " 张"; }
      toast("已打开", d.path + " · " + d.count + " 张", "success");
    }).catch(function (e) {
      if (st) { st.classList.remove("on"); st.innerHTML = "<i></i>未连接"; }
      toast("打开失败", String((e && e.message) || e), "error");
    });
  }
  window.mkOpenEditor = mkOpenEditor;

  function bindMaskEditor() {
    var btn = document.getElementById("mkOpen");
    if (btn && !btn.dataset.bound) {
      btn.dataset.bound = "1";
      btn.addEventListener("click", mkOpenEditor);
    }
    var inp = document.getElementById("mkDir");
    if (inp && !inp.dataset.bound) {
      inp.dataset.bound = "1";
      inp.addEventListener("keydown", function (e) { if (e.key === "Enter") mkOpenEditor(); });
    }
  }
  window.bindMaskEditor = bindMaskEditor;

  function bindMergeUI() {
    var page = $("p-merge");
    if (!page) return;

    /* 时间轴：每次进合成页都重绘一次（元素可能被重建），幂等。
       是「重进页面」还是「换素材」交给 mgTimelineSync 的 key=(path|总帧数) 判断：
       相同 → keep（保住播放头和人脸轨），不同 → 整条重建。
       没素材时 total 就是 0 —— 以前用 300 兜底，画出来的是假帧。 */
    setTimeout(function () {
      window.mvList();
      var mv = window.__mv || {};
      mgTimelineSync(mv.path ? (mv.frames || 0) : 0, mv.fps || 30);
      /* 视频模式：已有的视频要接回上次停下的位置（window.__mv.t） */
      if (window.mgResumeVideo) window.mgResumeVideo();
      /* 进合成页时若已经有视频/素材：用户多半马上会暂停看效果，先热身。
         每次进页面都判一次；重复请求由 __mgWarm 的 10 秒去重挡住 */
      /* 选模型即加载：不等打开视频。模型选好了、视频还没选，也应该在后台把 DFM 装进显存
         （多选时每个选中模型都热一遍，__mgWarm 自带 10 秒去重） */
      window.mgWarmSelected();
      /* 内容不足一屏时把栈拉满，刻度尺才不会只铺一半 */
      var st = document.querySelector(".mtl-stack");
      var sc = document.querySelector(".mtl-scroll");
      if (st && sc) st.style.width = Math.max(st.scrollWidth, sc.clientWidth) + "px";
      /* 合成视图：chip / 参数实时预览的绑定（幂等），有素材就把框和画面都补齐 */
      if (window.mgBindCmp) window.mgBindCmp();
      if (window.mgApplyView) window.mgApplyView();
      if (window.mgCmpSync) window.mgCmpSync();
      /* 舞台空态：干净工作区进来 = 中央一句话 + 两个入口；已有素材 = 收起 */
      if (window.mgStageEmpty) window.mgStageEmpty();
    }, 60);

    if (page.dataset.bound) return;
    page.dataset.bound = "1";
    /* 视频模式：播放键 / 空格交给 <video> 的时钟（幂等，重复进入不会重复绑） */
    if (window.mgBindTransport) window.mgBindTransport();
    page.querySelectorAll(".seg").forEach(function (seg) {
      seg.addEventListener("click", function (e) {
        var b = e.target.closest("b");
        if (!b || !seg.contains(b)) return;
        Array.prototype.forEach.call(seg.children, function (x) { x.classList.remove("on"); });
        b.classList.add("on");
      });
    });
    /* 「合成 / 原图」不再有分段控件，也不再是整帧视图：
       画面永远是「合成图压在原片上」，点某张脸的框只把那张脸切回原像素
       （= 请求体里的 no_swap_faces，见 mgToggleFaceRaw / mgBindFacePick）。 */

    /* 导出流程：工作台（预览/剪辑）→ 人脸库（分组）→ 导出设置 → 导出中 */
    var MG_STEPS = ["main", "facedb", "settings"];
    function mgStep(name) {
      var step = document.getElementById("mgStep");
      var back = document.getElementById("mgBack");
      page.querySelectorAll(".mgview").forEach(function (v) {
        v.classList.toggle("in", v.dataset.view === name);
      });
      if (step) {
        var label = { facedb: "人脸库", settings: "导出设置",
                      project: "导出项目", progress: "导出进度" }[name] || "";
        step.textContent = label;
        step.hidden = !label;
      }
      if (back) back.hidden = (name === "main");
      /* 面包屑跟着走：合成工作室 › 人脸数据库 › 导出设置 */
      var trail = [{ t: "合成工作室", step: "main" }];
      if (name === "facedb") trail.push({ t: "人脸数据库", step: "facedb" });
      if (name === "settings") trail = [
        { t: "合成工作室", step: "main" },
        { t: "导出设置", step: "settings" }
      ];
      if (name === "progress") trail = [
        { t: "合成工作室", step: "main" },
        { t: "导出设置", step: "settings" },
        { t: "导出进度", step: "progress" }
      ];
      if (name === "project") trail = [
        { t: "合成工作室", step: "main" },
        { t: "导出项目", step: "project" }
      ];
      CRUMBS.merge = trail;
      renderCrumbs("merge");
      window.__mergeStep = name;
      if (window.SUB_MEMORY) window.SUB_MEMORY["merge"] = null;   /* 合成器用 mgStep，不走 go */
      /* 一律交给 MergePages.show：name==="main" 时它会清 .mgp-on，
      否则回主视图后主容器还被 .mgp-on 的 CSS 藏着 */
      if (window.MergePages) window.MergePages.show(name);
      document.querySelector("#p-merge .mgbody") &&
        (document.querySelector("#p-merge .mgbody").style.display = (name === "main") ? "" : "none");
    }
    window.mgStep = mgStep;

    var backBtn2 = document.getElementById("mgBack");
    if (backBtn2 && !backBtn2.dataset.bound) {
      backBtn2.dataset.bound = "1";
      backBtn2.addEventListener("click", function () {
        var cur = (page.querySelector(".mgview.in") || {}).dataset;
        var at = cur ? cur.view : "main";
        /* 人脸库这一页归 face_db.js 管：必须走它的 requestClose()（=「返回会取消
           本次导出？」那条确认），直接 mgStep 会把 state.open 留在 true ——
           导出流程里等于绕过确认，浏览模式又说不上「已经关掉」。 */
        if (at === "facedb" && window.FaceDB && window.FaceDB.requestClose) {
          window.FaceDB.requestClose();
          return;
        }
        var idx = MG_STEPS.indexOf(at);
        mgStep(idx <= 1 ? "main" : "facedb");
      });
    }

    /* 两条导出都先经过人脸库：整理完点「继续导出」才回调真正发请求。
       face_db.js 自己不发 /export；它只把控制权交回这里，参数组装仍然只有下面
       mgRunExport / mgpjStart 这一份，避免两套导出契约。 */
    function mgOpenFaceDb(mode, proceed, project) {
      project = String(project || window.__mergeInputDir || "");
      if (!project) {
        mgToastErr("open-db-noproj", "还没有项目目录",
                   "当前是视频模式：后端没给能放人脸库的目录（先导入一段视频或打开 DFL 项目）",
                   "warn", 5000);
        return;
      }
      if (!window.FaceDB || !window.FaceDB.open) {
        toast("人脸库没有加载", "face_db.js 版本不匹配，刷新页面再试", "error");
        return;
      }
      window.FaceDB.open({
        project: project,
        mode: mode,
        allowExtract: true,
        onContinue: proceed,
        onCancel: function () { if (window.mgStep) window.mgStep("main"); }
      });
    }

    /* ---------- 导出：先设置页，再开跑 ----------
       以前的「导出」是**直接开跑**：点下去就 mgRunExport，参数全是默认值，想改只能
       先手动摸到导出设置页去（还得知道有那一页）。现在入口只做两件事：
       ① 校验「有没有选模型」（0 个 = 按钮本来就是禁用的，这里再兜一层）；
       ② 把这一页的只读回显填好，切到导出设置页。**一个请求都不发。**
       真正 POST 阶段请求的是下面 #mgStartExport 那一下。 */
    function mgDefOutPath() {
      /* 只认「输出路径」框里用户手填的值。这里不能读 MergePages.settings().output_path：
         那条现在会把「输出目录」里改过的值也当 output_path 返回，默认值就会跟着自己跑
         （改一版 → 默认值变成上一版 → 再改又叠一层）。 */
      var outEl = document.getElementById("mgOutPath");
      var manual = String((outEl && outEl.value) || "").trim();
      if (manual) return manual;
      var dir = String(window.__mergeInputDir || "");
      if (dir) {
        var base = dir.split(/[\\/]/).filter(Boolean).pop() || "project";
        return dir.replace(/[\\/]+$/, "") + "\\" + base + "_merged.mp4";
      }
      var video = mgVidPath();
      if (video) return video.replace(/\.[^./\\]+$/, "") + "_merged.mp4";
      return "（默认：源视频名 _merged.mp4，与源视频同目录）";
    }

    /* 打开/刷新导出设置页：只读回显 + 当前参数，不动任何请求 */
    function mgFillExportSettings() {
      var names = mgSelNames();
      var box = document.getElementById("mgSelModels");
      if (box) {
        box.innerHTML = "";
        if (!names.length) {
          var em = document.createElement("span");
          em.className = "mgexp-models-empty";
          em.textContent = "还没选模型";
          box.appendChild(em);
        } else {
          names.forEach(function (n) {
            var chip = document.createElement("span");
            chip.className = "mgexp-chip";
            chip.textContent = n;
            box.appendChild(chip);
          });
        }
      }
      var out = document.getElementById("mgDefOut");
      if (out) {
        out.value = mgDefOutPath();
        /* 每次进导出设置页都把「用户改过没有」清掉：这一格刚被重算成默认值，
           继续把它当用户填的目录发出去，就会悄悄改掉上一次的输出位置。 */
        window.__mgDefOutEdited = false;
      }
      var total = 0;
      try {
        total = parseInt(String((window.__mv || {}).frames || 0), 10) || 0;
      } catch (e) { total = 0; }
      if (!total) {
        var tf = document.getElementById("mtlTotal");
        total = parseInt(String((tf && tf.textContent) || "0"), 10) || 0;
      }
      /* 右边那颗「全片」按钮本身就是「恢复整段」，这行字只补「一共多少帧」；
         总帧数还不知道（视频还没探测完）就干脆不显示，别写成「全片 全片」。 */
      var tot = document.getElementById("mgFrameTotal");
      if (tot) {
        if (total > 0) { tot.textContent = "全片 · " + total + " 帧"; tot.hidden = false; }
        else { tot.textContent = ""; tot.hidden = true; }
      }
      var to = document.getElementById("mgFrameTo");
      if (to) to.placeholder = total > 0 ? String(Math.max(0, total - 1)) : "末帧";
      var cnt = document.getElementById("mgSettingsCount");
      if (cnt) {
        cnt.textContent = names.length
          ? (names.length + " 个模型 · 全片")
          : "没有选中模型";
      }
      if (window.MergePages && window.MergePages.syncStatus) window.MergePages.syncStatus();
      mgSyncExportGate();
    }
    window.mgFillExportSettings = mgFillExportSettings;

    function mgOpenExportSettings() {
      if (!mgSyncExportGate()) {
        mgToastErr("export-nosel", "先选一个模型",
                   "左栏「模型」里点一个（可多选），导出要用它换脸", "warn", 4000);
        return false;
      }
      mgFillExportSettings();
      mgStep("settings");
      return true;
    }
    window.mgOpenExportSettings = mgOpenExportSettings;

    var exBtn = $("mgExport");
    if (exBtn && !exBtn.dataset.bound) {
      exBtn.dataset.bound = "1";
      exBtn.addEventListener("click", function () {
        /* 只切页；不解析项目、不请求后端 —— 断言「点导出 → 0 个 /api/merge/stage」靠这一条 */
        mgOpenExportSettings();
      });
    }

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
        detector: p.detector || "",
        thresh: 0.5,
        settings: mgMergeSettings(p)
      };
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

    /* ============================================================
       六阶段状态机 + 流水线实况（SVG）
       ------------------------------------------------------------
       这一页有两条后端的路，但只有一张图：
         · fused（快速导出，默认）：六段**并行**，只有帧进度；
         · 分段链（frames→faces→pred→mask→merge→encode）：串行，一段一段来。
       图本身（6 列 lane + 贝塞尔曲线 + 光点）的数据**全部**来自 status：
         status.stages[key] = {ticks, inflight, lanes[], ms?, batch?}
       前端一条都不写死 —— lane 有几条就是图上几个方块（合成池 8 线程 = t0..t7，
       编码 6 线程 = e0..e5），ticks 只增、光点按**增量**放，inflight 就是列间那句
       列标题的「N 帧」。没有这份数据时（老后端 / 还没开单）列是占位的「—」，不瞎猜。
       点亮：某条 lane 的列 tick 了 → 那条 lane 瞬间到峰值、1.15s 暗回常态。
       卡住：某列 >600ms 没 tick 且还没完成 → 整列琥珀；恢复自动复原。
       ============================================================ */
    var MGP_STAGES = ["frames", "faces", "pred", "mask", "merge", "encode"];
    /* 分段链的阶段名 → status.stages 的 key（两套名字，只在这张表里对一次） */
    var MG_KEY = { frames: "frames", faces: "det", pred: "dfm",
                   mask: "xseg", merge: "compose", encode: "encode" };

    var MG_PIPE = (function () {
      var NS = "http://www.w3.org/2000/svg";
      var KEYS = ["frames", "det", "dfm", "xseg", "compose", "encode"];
      /* 最后一列不叫「导出」了：这一段真的只是把合成好的帧**编码/缓存**出来，
         真正写出成片是后面 ffmpeg 那一趟（也计时到这一段）。用户看到的「导出」两个字
         出现在这一列会以为进度已经到写视频了，所以叫「缓存帧」。后端/日志里那个
         内部名 encode 不动。 */
      var NAME = { frames: "切帧", det: "切脸", dfm: "换脸",
                   xseg: "遮罩", compose: "合成", encode: "缓存帧" };
      /* viewBox 尺寸定死（和 UI 设计稿同一比例）：进度卡铺满内容区（≈1200~1400px），
         所以 1200×340 渲染出来接近 1:1～1.15:1，9.5px 的 lane 名字不会缩成 6px。
         高度不变 → 换 lane 数也不会让页面跳动。 */
      var W = 1200, H = 340, PADL = 56, TOPY = 58, BOXW = 74, GAP = 7;
      /* 右侧留 84px 给「成片」那一格（水位杯 + 标题）。它**不是第七列**：没有 lane、
         没有 t-stage，只画一杯水 —— 六列仍然是六列（验收按 text.t-stage 数）。 */
      var RGT = 0, OUTW = 56;   /* 成片杯现在吃满一格宽度 */
      /* 光点密度由**两个物理上限**决定，不再按「每次轮询硬切 10 个」：
           · PULSE_PER_S：每秒最多放多少个（按 POLL_MS 折算成每次轮询的预算）；
           · MAX_PULSE  ：同屏最多几个在飞（≈ 40 帧/秒 × 8 张脸 × 0.6s 飞行时间的极端值）。
         常见档（20~30 帧/秒、每帧 1~4 张脸）一条都不丢；只有极端帧率/人脸数才按比例稀释。
         以前 TICK_MAX=10 把「20 帧/秒」也一起砍了（500ms × 20fps = 每段正好 10 个），
         而且 10 个还在同一瞬间一次性放出来 —— 肉眼就是「每半秒跳一波」。 */
      var MAX_PULSE = 200;     /* 同屏在飞上限（丢最旧的） */
      var PULSE_PER_S = 240;   /* 每秒最多放几个光点 */
      var POLL_MS = 250;       /* 和轮询同一个窗口：一拍的光点铺在这 250ms 里 */
      var LIT_MS = 1250;       /* 1.15s 动画 + 余量，之后摘 .lit 回稳态 */
      var STALL_MS = 600;      /* 某列多久没 tick 算卡住 */

      var geo = {};            /* key -> {cx, off, lanes:[{y,rect,txt}], count, queue, col} */
      var edgesIn = {};        /* "si:ai" -> {d, el}：源 lane → 腰点 */
      var edgesOut = {};       /* "si:bi" -> {d, el}：腰点 → 目标 lane（和 edgesIn 共用腰点） */
      /* "in:out:q" / "out:out" -> d：编码列 → 成片的 n+1 条。**以前这个变量没声明**，
         只要 geo.out 先于这段 IIFE 被建出来（下面那版就是把杯子提前了），
         第一句 edges[...] 就是 ReferenceError → build() 直接抛出去，
         svg 一个子节点都没挂上、drawBypass() 永远走不到。 */
      var edges = {};
      var waists = {};         /* si -> {x, y}：这一对相邻列的腰点（进出共用） */
      var pulses = [];         /* 活跃光点 {c, path, L, t0, dur} */
      var pool = [];           /* 复用起来的 <circle>：不每个 tick 重建 DOM */
      var flashes = [];        /* 待点亮 [{k, i}]（合到下一帧一起做，减少强制重排） */
      var raf = 0;
      var sig = "";            /* lanes 的签名：变了才重建结构 */
      var prev = {};           /* key -> 上次的 ticks */
      var prevNf;              /* 上次的 det.no_face_ticks（旁路光点按它的增量放） */
      var lastAt = {};         /* key -> ticks 上次变化的时刻 */
      var stallNow = {};       /* key -> 现在卡没卡 */
      var state = {};          /* key -> "" / on / done / err（列标题颜色） */
      var seq = 0;             /* 光点序号：多 lane 时按它轮转 */
      var lastFlashed = [];    /* 最近一拍真正点亮过的 lane（验收拍照时按它重挂） */
      var lastSpawn = 0;       /* 最近一拍实际放出去几个光点（验收要看「按预算稀释」有没有生效） */
      var lastBypass = 0;      /* 其中走旁路的几个（无脸帧） */
      var layers = null;
      /* 旁路几何的「版本号」：build() 每重建一次结构就 +1（lane 数/列宽/杯子位置全变了）。
         drawBypass() 画完把当时的版本记下来，setBypass() 一比就知道要不要重画 ——
         以前它无条件只改读数，旁路因此会冻在**旧坐标**上。 */
      var bypassGen = 0, bypassDrawnGen = -1;
      /* 截图专用：把光点的时钟按住（>0 = 定格在这个时刻），只给验收拍照用。
         光点只有 470–750ms 的命，一张 headless 截图往返就可能比它长 —— 拍照时
         用 mgpPipeFreeze() 把它们钉在曲线上，运行时代码路径完全不看这个值。 */
      var freezeAt = 0;

      function nowMs() {
        return (window.performance && window.performance.now)
          ? window.performance.now() : Date.now();
      }
      function reduceMotion() {
        try {
          return !!(window.matchMedia &&
                    window.matchMedia("(prefers-reduced-motion: reduce)").matches);
        } catch (e) { return false; }
      }
      function mk(tag, attrs) {
        var e = document.createElementNS(NS, tag), k;
        for (k in (attrs || {})) {
          if (Object.prototype.hasOwnProperty.call(attrs, k)) e.setAttribute(k, attrs[k]);
        }
        return e;
      }
      /* 相邻两列之间只有一个**腰点**：x = 两列中线，y = 两列所有 lane 的 y 的中点。
         源侧：每条源 lane → 腰点（一条曲线）；目标侧：腰点 → 每条目标 lane（一条曲线）。
         于是「每个源 lane × 每个目标 lane」的交叉（合成 8 × 导出 6 = 48 条）变成
         8 + 6 = 14 条，进出共用同一个点 —— 视觉上是「先收束到一点、再散开」。
         单段仍是三次贝塞尔；两个控制点分别与两端同 y，所以**两端切线都水平**，
         两段在腰点处方向也连续，拼起来是一整条平滑的流（不是折线、也不打结）。 */
      function arc(a, b) {
        var dx = b.x - a.x;
        return "M" + a.x + "," + a.y +
          " C" + (a.x + dx * 0.55) + "," + a.y +
          " " + (b.x - dx * 0.55) + "," + b.y +
          " " + b.x + "," + b.y;
      }
      function lanesOf(st) {
        var out = {}, i, k, row;
        for (i = 0; i < KEYS.length; i++) {
          k = KEYS[i];
          row = (st && st[k]) || {};
          out[k] = (row.lanes && row.lanes.length) ? row.lanes.slice(0) : [];
        }
        return out;
      }
      function signature(lanes) {
        var parts = [], i;
        for (i = 0; i < KEYS.length; i++) parts.push((lanes[KEYS[i]] || []).join(","));
        return parts.join("|");
      }
      /* status.stages 的「实况」形状：必须有 lanes 或 ticks。
         分段链的 /stage/status 也带一个 stages（那是每段的元数据），别混。 */
      function isLive(st) {
        if (!st || !st.frames) return false;
        return !!(st.frames.lanes || typeof st.frames.ticks === "number");
      }

      function build(lanes) {
        var svg = $("mgPipeSvg");
        if (!svg) return false;
        while (svg.firstChild) svg.removeChild(svg.firstChild);
        layers = { defs: mk("defs", {}), edge: mk("g", { "class": "edges" }),
                   bypass: mk("g", { "class": "bypass" }), box: mk("g", {}),
                   lbl: mk("g", {}), pulse: mk("g", { "class": "pulses" }) };
        geo = {}; edgesIn = {}; edgesOut = {}; edges = {}; waists = {};
        pulses = []; pool = [];
        bypassD = "";
        bLabel = null;
        var colW = (W - PADL * 2) / KEYS.length;   /* 七等分：6 个阶段 + 成片各占一格（成片不再被挤在最右边） */
        var space = H - TOPY - 46;
        var i, j, k;
        for (i = 0; i < KEYS.length; i++) {
          k = KEYS[i];
          var cx = PADL + i * colW;
          var ln = (lanes[k] || []).slice(0);
          var off = !ln.length;                 /* 这一单不用这一段：占位，不假装有线程 */
          if (off) ln = ["—"];
          var n = ln.length;
          var laneH = Math.min(24, Math.max(14, space / n - GAP));
          var y0 = TOPY + (space - (n * laneH + (n - 1) * GAP)) / 2;
          var col = mk("g", { "class": "col" });
          col.setAttribute("data-k", k);
          var t1 = mk("text", { x: cx, y: 24, "text-anchor": "middle", "class": "t-stage" });
          t1.textContent = NAME[k] || k;
          col.appendChild(t1);
          var t2 = mk("text", { x: cx, y: 40, "text-anchor": "middle", "class": "t-count" });
          t2.textContent = "— 帧";
          col.appendChild(t2);
          layers.lbl.appendChild(col);
          geo[k] = { cx: cx, off: off, lanes: [], count: t2, col: col };
          /* 水位只填「缓存帧」这一列 —— 别的列是线程/队列，没有「攒了多少帧」这回事 */
          var cup = (k === "encode" && !off);
          for (j = 0; j < n; j++) {
            var y = y0 + j * (laneH + GAP);
            var rect = mk("rect", { x: cx - BOXW / 2, y: y, width: BOXW, height: laneH,
                                    rx: 7, "class": off ? "lane off" : "lane" });
            rect.setAttribute("data-lane", k + "#" + j);   /* key#序号：排障/拍照直接定位 */
            layers.box.appendChild(rect);
            var water = null;
            if (cup) {
              /* 水面：一块和方块同样大小的 rect，用 clip-path 拿到圆角，再 scaleY(level)。
                 夹在方块和它的名字之间（层级正好：水盖住底色、不盖住 t-lane 字）。 */
              var cid = "mgpW" + i + "_" + j;
              var cp = mk("clipPath", { id: cid });
              cp.appendChild(mk("rect", { x: cx - BOXW / 2, y: y, width: BOXW,
                                          height: laneH, rx: 7 }));
              layers.defs.appendChild(cp);
              /* rx=0：圆角由 clipPath 给。给水面自己 rx 的话，scaleY 会把圆角压扁，
                 顶边变成圆弧 —— 看着像「一小块色块」而不是「一条水平水面线」。 */
              water = mk("rect", { x: cx - BOXW / 2, y: y, width: BOXW, height: laneH,
                                   "class": "water", "data-level": "0" });
              /* clip-path 必须挂在**不被缩放**的外层 g 上：直接挂在水面上时，
                 clip 用的是元素自身的用户坐标系，会跟着 scaleY 一起被压扁 ——
                 结果是「一颗越来越小的药丸」，而不是「一条水平的水面线」。 */
              var wg = mk("g", { "clip-path": "url(#" + cid + ")" });
              wg.appendChild(water);
              layers.box.appendChild(wg);
            }
            var tx = mk("text", { x: cx, y: y + laneH / 2 + 3.2, "text-anchor": "middle",
                                  "class": "t-lane" });
            tx.textContent = ln[j];
            layers.box.appendChild(tx);
            geo[k].lanes.push({ y: y + laneH / 2, y0: y, h: laneH,
                                rect: rect, txt: tx, water: water });
          }
        }
        /* 六列之后的「成片」：一只窄水杯（缓存里的水倒进这里）。它不进 KEYS、没有 lane，
           所以 text.t-stage 仍然只有 6 个、rect.lane 仍然只数真线程。 */
        var ocx = PADL + KEYS.length * colW;
        var ocid = "mgpWout";
        var ocp = mk("clipPath", { id: ocid });
        ocp.appendChild(mk("rect", { x: ocx - OUTW / 2, y: TOPY, width: OUTW,
                                     height: space, rx: 7 }));
        layers.defs.appendChild(ocp);
        /* 壳子用 .outcup 而不是 .lane：rect.lane 只数真线程（验收按它数 lane），
           给它挂 lane 会让「lane 方块数 == status 里的线程数」这条断言多出一个 */
        layers.box.appendChild(mk("rect", { x: ocx - OUTW / 2, y: TOPY, width: OUTW,
                                            height: space, rx: 7, "class": "outcup" }));
        /* 杯子里的**两层**水（以前只有一层：蓝面直接画 encode，于是「编码一开始蓝面就突然满」）：
             蓝面 .water.out = 已合成 = cache + out = compose / 总帧   ← 帧已经从合成那列出来了
             绿面 .encflow  = 已编码 = out           = encode / 总帧   ← 帧已经变成视频了
           绿面画在蓝面**之上**、半透明（让蓝透出来）：两层之间的那一段蓝就是「合成好了、
           还没编码」的帧。编码阶段绿面从下往上追蓝面 —— 看起来就是「水被倒进视频里」。
           绿面用 .encflow 而不是再加一块 .water：验收按 rect.water 的条数断言「水位只填
           两处（缓存列 n 条 + 成片 1 条）」，多一块 .water 会把它数成 8 条。 */
        var owater = mk("rect", { x: ocx - OUTW / 2, y: TOPY, width: OUTW, height: space,
                                  "class": "water out", "data-level": "0" });
        var oenc = mk("rect", { x: ocx - OUTW / 2, y: TOPY, width: OUTW, height: space,
                                "class": "encflow", "data-level": "0" });
        var owg = mk("g", { "clip-path": "url(#" + ocid + ")" });
        owg.appendChild(owater);
        owg.appendChild(oenc);
        layers.box.appendChild(owg);
        /* 水面线：蓝色填充的顶边原本糊在底色里，「水在哪」没有明确边界。这条 1.5px 的
           半透明白线压在蓝面顶边上，就是那条边界。线本体停在杯口（y1 = y2 = TOPY），
           位置全部交给 translateY(space × (1 − 蓝面level)) —— 和蓝面同一条 .3s 补间，
           所以它永远贴着水位走，不会自己跳一格。挂在 clip 组**外面**：满格时它就在最上沿，
           塞进 clip 里会被切掉半条描边，1.5px 变成 0.75px 的灰边。 */
        var osurf = mk("line", { x1: ocx - OUTW / 2 + 1, x2: ocx + OUTW / 2 - 1,
                                 y1: TOPY, y2: TOPY, "class": "out-surf",
                                 "data-level": "0", "data-y": String(TOPY) });
        layers.box.appendChild(osurf);
        var otxt = mk("text", { x: ocx, y: 24, "text-anchor": "middle", "class": "t-out" });
        otxt.textContent = "成片";
        layers.lbl.appendChild(otxt);
        /* top/span = 杯子的杯口与杯高：setWater 用它们算水面线的 translateY */
        geo.out = { cx: ocx, water: owater, enc: oenc, surf: osurf, top: TOPY, span: space };
  /* 编码列 → 成片：与其它列同构（每条 lane 收束到腰点，再一条进杯）。
     成片不是一个 lane 列，所以单独画：n 条入腰 + 1 条腰→杯，都走 arc()。 */
  (function () {
    var E = geo.encode, O = geo.out;
    if (!E || !O || E.off || !E.lanes.length) return;
    var ys = [], q;
    for (q = 0; q < E.lanes.length; q++) ys.push(E.lanes[q].y);
    var wy = (Math.min.apply(null, ys) + Math.max.apply(null, ys)) / 2;
    var x1 = E.cx + BOXW / 2;
    var x2 = O.cx - OUTW / 2;
    var wx = (x1 + x2) / 2;
    E.waist = { x: wx, y: wy };
    for (q = 0; q < E.lanes.length; q++) {
      var din2 = arc({ x: x1, y: E.lanes[q].y }, { x: wx, y: wy }, (x1 + wx) / 2);
      var pin2 = mk("path", { d: din2, fill: "none", stroke: "var(--line-strong)",
        "stroke-width": 1, opacity: ".75" });
      layers.edge.appendChild(pin2);
      edges["in:out:" + q] = din2;
    }
    var dcup = arc({ x: wx, y: wy }, { x: x2, y: wy }, (wx + x2) / 2);
    var pcup = mk("path", { d: dcup, fill: "none", stroke: "var(--line-strong)",
      "stroke-width": 1, opacity: ".75" });
    layers.edge.appendChild(pcup);
    edges["out:out"] = dcup;
    /* 旁路要「并进主路」：把**腰点**和**这段进杯曲线**都挂到 geo.out 上，
       让 bypassGeom() 复用同一份数字（同一个腰点、同一条三次贝塞尔）。
       各算一条「差不多」的曲线在杯口碰一下，差 0.1px 就会被看成两条线，
       读者不会觉得那是「汇入主流」——所以尾段必须是这一个 d 的子串。 */
    O.waist = { x: wx, y: wy };
    O.tailD = dcup;
    O.tailC = String(dcup).replace(/^\s*M[^C]*/, "");   /* 去掉开头的 M，只剩那段 C */
  })();

        /* 连线：源侧 lane → 腰点、腰点 → 目标侧 lane（相邻两列共用同一个腰点）。
           空列（off）不连线，光点会跳过它。 */
        for (i = 0; i < KEYS.length - 1; i++) {
          var A = geo[KEYS[i]], B = geo[KEYS[i + 1]];
          if (A.off || B.off) continue;
          var ys = [];
          for (j = 0; j < A.lanes.length; j++) ys.push(A.lanes[j].y);
          for (j = 0; j < B.lanes.length; j++) ys.push(B.lanes[j].y);
          var waist = { x: (A.cx + B.cx) / 2,
                        y: (Math.min.apply(null, ys) + Math.max.apply(null, ys)) / 2 };
          waists[i] = waist;
          var sx = A.cx + BOXW / 2, tx = B.cx - BOXW / 2;
          for (var ai = 0; ai < A.lanes.length; ai++) {
            var din = arc({ x: sx, y: A.lanes[ai].y }, waist);
            var pin = mk("path", { d: din, fill: "none", stroke: "var(--line-strong)",
                                   "stroke-width": 1, opacity: ".75" });
            layers.edge.appendChild(pin);
            edgesIn[i + ":" + ai] = { d: din, el: pin };
          }
          for (var bi = 0; bi < B.lanes.length; bi++) {
            var dout = arc(waist, { x: tx, y: B.lanes[bi].y });
            var pout = mk("path", { d: dout, fill: "none", stroke: "var(--line-strong)",
                                    "stroke-width": 1, opacity: ".75" });
            layers.edge.appendChild(pout);
            edgesOut[i + ":" + bi] = { d: dout, el: pout };
          }
        }
        svg.appendChild(layers.defs);
        svg.appendChild(layers.edge);
        svg.appendChild(layers.bypass);  /* 旁路在连线之上、方块之下（不挡文字） */
        svg.appendChild(layers.box);
        svg.appendChild(layers.lbl);
        svg.appendChild(layers.pulse);   /* 光点在最上层 */
        sig = signature(lanes);
        lenCache = {};                   /* 连线/旁路的 d 都变了：量过的长度作废 */
        /* 结构重建后把上一拍的读数补回去：水位/旁路是「数据」，不是「结构」 */
        setWater(waterLv.cache, waterLv.out);
        /* 注：没有 edges["out:out"] 就没有「腰点 → 成片」那条实线，也就没有可并的
           汇合点 —— 那种情况下 bypassGeom() 直接返回 null（宁可不画，也不自己猜落点）。 */
        /* 结构重建 = 坐标全变：先升版本，再按新坐标重画（setBypass 靠这个版本号判定） */
        bypassGen += 1;
        drawBypass();
        return true;
      }

      /* ============================================================
         水位（进度）：只填「缓存帧」那一列 + 右侧「成片」那一格
         ------------------------------------------------------------
         两个数都来自 status.stages，前端不编：
           cache = (compose.ticks − encode.ticks) / 总帧数   ← 缓存里**现存**的帧
           out   = encode.ticks / 总帧数                     ← 已经编码出来的帧
         为什么是差值：帧从合成那一列流进缓存、再被编码那一列取走。合成 72 而编码 40
         就是「缓存里还压着 32 帧」—— 这就是缓存水位该显示的东西（用户口径：
         用 encode 与 compose 的差值推「已倒出多少」，不要新造后端字段）。
         快速导出（fused）是边合成边编码，这个差值就只有队列里那几十帧 —— 按真实比例，
         不做「先攒满再倒」的假动画。
         成片那杯是**两层**（就是用户要的「绿追蓝」）：
           blue  = cache + out = compose / 总帧   ← 蓝面：已经合成出来的帧
           green = out         = encode / 总帧    ← 绿面：已经编码成视频的帧（半透明，在蓝面之上）
         绿面永远 ≤ 蓝面；差值在收尾那一拍可能瞬时为负，setWater 里夹一刀。
         收工那一拍 compose = encode = 总帧 → 两层都到 1、两块面重合。
         ============================================================ */
      var waterLv = { cache: 0, out: 0, blue: 0, green: 0 };
      function wkey(v) { return Math.round(v * 1000) / 1000; }
      function setWater(cache, out) {
        cache = Math.max(0, Math.min(1, Number(cache) || 0));
        out = Math.max(0, Math.min(1, Number(out) || 0));
        waterLv.cache = cache;
        waterLv.out = out;
        var g = geo.encode;
        if (g && g.lanes.length) {
          var first = g.lanes[0], last = g.lanes[g.lanes.length - 1];
          var top = first.y0, bot = last.y0 + last.h;
          var span = Math.max(1, bot - top);
          /* 整列当一个杯子：水位线 y = 杯底 − level×杯高；每条 lane 只显示
             「自己这一段里被淹到的部分」→ 视觉上是一杯水往上漫，而不是六条一样高的条。 */
          var levelY = bot - cache * span;
          for (var i = 0; i < g.lanes.length; i++) {
            var ln = g.lanes[i];
            if (!ln.water) continue;
            var k = Math.max(0, Math.min(1, (ln.y0 + ln.h - levelY) / ln.h));
            ln.water.setAttribute("data-level", String(wkey(k)));
            ln.water.style.transform = "scaleY(" + wkey(k) + ")";
          }
        }
        /* 成片杯的两层水：蓝面 = 已合成（cache + out = compose/总帧）、绿面 = 已编码（out）。
           以前蓝面直接画 out —— 于是「编码一开始蓝面就突然满」，看不出水在被倒进视频。
           绿面夹在蓝面以内：encode 追不过 compose（差值瞬时为负时也不倒挂）。 */
        var blue = Math.max(0, Math.min(1, cache + out));
        var green = Math.min(blue, Math.max(0, Math.min(1, out)));
        waterLv.blue = blue;
        waterLv.green = green;
        var O = geo.out;
        if (O && O.water) {
          O.water.setAttribute("data-level", String(wkey(blue)));
          O.water.style.transform = "scaleY(" + wkey(blue) + ")";
        }
        if (O && O.enc) {
          O.enc.setAttribute("data-level", String(wkey(green)));
          O.enc.style.transform = "scaleY(" + wkey(green) + ")";
        }
        if (O && O.surf) {
          /* 水面线贴蓝面顶边：y = 杯顶 + 杯高 × (1 − blue)。线本体画在杯口（y1=y2=TOPY），
             所以位移就是 杯高 × (1 − blue)，和蓝面同一条 .3s 补间 —— 线跟着水走，不自己跳。
             data-y / data-level 是给验收读的绝对位置（不用再从 computed matrix 里反推）。 */
          var dy = (Number(O.span) || 0) * (1 - blue);
          O.surf.setAttribute("data-level", String(wkey(blue)));
          O.surf.setAttribute("data-y", String(wkey((Number(O.top) || 0) + dy)));
          O.surf.style.transform = "translateY(" + wkey(dy) + "px)";
          /* 空杯不画线：level=0 时那条线正好落在杯底内沿，看着像「底下还压着一层水」 */
          O.surf.style.opacity = blue > 0.004 ? "1" : "0";
        }
        /* 标题区那颗小杯跟成片**同一个数**（蓝面 = 已合成）：它的 aria-label 写的是「已完成帧」 */
        var cup = $("mgPipeCupFill");
        if (cup) {
          cup.setAttribute("data-level", String(wkey(blue)));
          cup.style.transform = "scaleY(" + wkey(blue) + ")";
        }
      }

      /* ============================================================
         无脸旁路：切脸之后直接汇入「缓存帧」，绕开 换脸 / 遮罩 / 合成
         ------------------------------------------------------------
         事实：一张脸都没检出的帧不需要 dfm/xseg，合成也是原帧直通，最后只走编码。
         所以图上从 det 右侧拉一条**淡线**沉到所有 lane 下面、再升到编码列 —— 线上写
         「无脸 N 帧」。不画的话，「切脸 1439 / 换脸 1204」这 235 帧看着像丢了
         （用户就是这么理解的，还自己在截图上补了一条红线）。
         只有这一单真有 no_face_ticks>0 时才出现；0 就没有这条线，也不留空节点。
         ============================================================ */
      var bypassN = 0, bypassD = "", bLabel = null;
      function bypassGeom() {
        var A = geo.det, B = geo.encode, O = geo.out;
        if (!A || !B || A.off || B.off || !A.lanes.length || !B.lanes.length) return null;
        /* 没有 O.waist / O.tailC 就不画：那条「腰点 → 成片」的实线是汇合点，
           它不存在时旁路就无处可并（以前这里会退到 B.cx 上另算一个落点 ——
           于是「旁路」和主路各走各的，这正是用户说「看着和主路没关系」的由来）。 */
        if (!O || !O.waist || !O.tailC) return null;
        var a0 = A.lanes[0];
        var x1 = A.cx + BOXW / 2, y1 = a0.y;
        /* 汇合点 = 主路腰点：x = (缓存帧右缘 + 杯左缘) / 2，y = 编码 lane 的中线。
           末段直接拼 O.tailC（主路 out:out 去掉 M 的那一段）——**同一份坐标**，
           所以虚线的尾巴就压在进杯实线上，看起来是「汇进去、一起倒进成片」，
           而不是两条线在杯口各画各的。 */
        var wx = O.waist.x, wy = O.waist.y;
        /* 沉到**所有列**的 lane 底下（再往下 20px）：只看首尾两列不够 —— 中间那列
           合成有 8 条 lane，比切脸/编码都深，旁路会从它最后两条中间穿过去。 */
        var bot = 0, i2, g2, l2;
        for (i2 = 0; i2 < KEYS.length; i2++) {
          g2 = geo[KEYS[i2]];
          if (!g2 || g2.off) continue;
          for (l2 = 0; l2 < g2.lanes.length; l2++) {
            bot = Math.max(bot, g2.lanes[l2].y0 + g2.lanes[l2].h);
          }
        }
        var yb = Math.min(H - 12, (bot || (a0.y0 + a0.h)) + 20);
        /* 「竖起来」的那一段要留在**缓存帧列右缘之外的空档**里（xr → wx 这一段，
           999.7~1116 之间）。以前是从 mx 一路斜拉到腰点，曲线中段正好从 e4/e5
           两个方块背后穿过去 —— 方块盖在旁路上面，只看得到几个角上的虚线段，
           读者根本读不出「升上来、汇进去」这件事。 */
        var xr = B.cx + BOXW / 2 + 12;       /* 刚出缓存帧列：竖起来的起点 */
        var mx = (x1 + xr) / 2;              /* 谷底水平段的中点（标签挂这儿） */
        var r = function (v) { return Math.round(v * 10) / 10; };
        var rel = function (a, b) { return arc(a, b).replace(/^M[^C]*/, ""); };
        return { /* ① det 右缘 →（沉到谷底）→ 空档里竖直升起 → 腰点
                    每段两端切线都水平，和主路同一个 arc() 约定，接上去不打折 */
                 d: "M" + r(x1) + "," + r(y1) +
                    rel({ x: x1, y: y1 }, { x: xr, y: yb }) +
                    rel({ x: xr, y: yb }, { x: wx, y: wy }) +
                    /* ② 腰点 → 杯左缘：主路那一段的原文（含控制点），一字不改地接上 */
                    O.tailC,
                 label: { x: mx, y: yb - 6 },
                 merge: { x: wx, y: wy },                     /* 汇合点 = 主路腰点 */
                 end: { x: O.cx - OUTW / 2, y: wy } };        /* 终点 = 杯左缘 */
      }
      function drawBypass() {
        if (!layers) return;
        /* 「我已经按这一版结构算过了」——即便下面因为 bypassN=0 提前返回也要记，
           否则 setBypass 会以为没画过、每一拍都重建 DOM。 */
        bypassDrawnGen = bypassGen;
        var g = layers.bypass;
        while (g.firstChild) g.removeChild(g.firstChild);
        bypassD = "";
        bLabel = null;
        if (!bypassN) return;
        var bg = bypassGeom();
        if (!bg) return;
        bypassD = bg.d;
        var p = mk("path", { d: bg.d });        /* 颜色/虚线/透明度全在 CSS 里 */
        p.setAttribute("data-bypass", "no-face");
        g.appendChild(p);
        bLabel = mk("text", { x: bg.label.x, y: bg.label.y, "text-anchor": "middle",
                              "class": "t-bypass" });
        bLabel.textContent = "无脸 " + bypassN + " 帧";
        g.appendChild(bLabel);
      }
      function setBypass(n) {
        n = Math.max(0, Math.floor(Number(n) || 0));
        var had = bypassN > 0;
        var shown = !!(layers && layers.bypass && layers.bypass.firstChild);
        bypassN = n;
        /* 已经画着**而且结构没重建过**，才只改读数（250ms 一拍，不重建 DOM）。
           以前这里是无条件 return —— 只要先画过一次，之后 build() 哪怕把列宽、
           杯子位置全换了，旁路的 d 也冻在旧坐标上（用户改了两版都「没被画出来」，
           另一半原因就是它）。bypassDrawnGen !== bypassGen 就说明坐标已经作废。 */
        if (n > 0 && shown && bypassDrawnGen === bypassGen) {
          if (bLabel && bLabel.textContent !== ("无脸 " + n + " 帧")) {
            bLabel.textContent = "无脸 " + n + " 帧";
          }
          return;
        }
        if (n === 0 && !shown && !had) return;   /* 本来就没有：不碰 DOM */
        drawBypass();
      }

      function setCount(k, ticks) {
        var g = geo[k];
        if (!g || !g.count) return;
        var txt = (ticks === null || ticks === undefined) ? "— 帧" : (ticks + " 帧");
        if (g.count.textContent !== txt) g.count.textContent = txt;
      }
      function markStall(k, on) {
        on = !!on;
        if (stallNow[k] === on) return;
        stallNow[k] = on;
        var g = geo[k];
        if (!g) return;
        for (var i = 0; i < g.lanes.length; i++) {
          if (on) g.lanes[i].rect.classList.add("stall");
          else g.lanes[i].rect.classList.remove("stall");
        }
      }
      function setStates(m) {
        var changed = false, i, k;
        for (i = 0; i < KEYS.length; i++) {
          k = KEYS[i];
          var st = (m && m[k]) || "";
          if (state[k] === st) continue;
          state[k] = st;
          changed = true;
          var g = geo[k];
          if (g && g.col) {
            g.col.classList.remove("on", "done", "err");
            if (st) g.col.classList.add(st);
          }
        }
        return changed;
      }
      /* 点亮也铺在同一个窗口里：at = 这一次交接「应该」发生的时刻（不会早于现在）。
         同一帧里同一条 lane 只闪一次（5 帧叠在 1/60 秒里看不出差别，但会白跑 4 次强制重排）。 */
      function flashOne(k, i, at) {
        flashes.push({ k: k, i: i, at: at || 0 });
      }
      function recycle(p) {
        p.c.setAttribute("opacity", "0");
        /* 坐标也清掉：池里的圆点不该带着上一次飞行的残坐标（探针/截图都可能看见它） */
        p.c.removeAttribute("cx");
        p.c.removeAttribute("cy");
        pool.push(p.c);
        /* 光点的临时 path 用完就删（它只是给 getPointAtLength 当尺子，
           不进 DOM 的可见层，也不留垃圾） */
        if (p.path && p.path.parentNode) p.path.parentNode.removeChild(p.path);
        p.path = null;
      }
      /* d = 源 lane→腰点 那段 + 腰点→目标 lane 那段（第二段以 "M 腰点" 开头，
         位置正好落在腰点上，所以拼起来的路径是**连续**的，光点从源 lane 一路走到目标 lane）。 */
      var lastSeen = {};      /* 这一拍要亮哪几条 lane（去重，给拍照用） */
      function rememberFlash(k, i) {
        var id = k + "#" + i;
        if (lastSeen[id]) return;
        lastSeen[id] = 1;
        lastFlashed.push({ k: k, i: i });
        if (lastFlashed.length > 64) lastFlashed.shift();
      }
      var lenCache = {};      /* 同一条「源→腰→目标」路径的长度只量一次 */
      function addPulse(key, d, startAt) {
        if (!layers) return;
        if (pulses.length >= MAX_PULSE) recycle(pulses.shift());   /* 超上限丢最旧的 */
        var path = mk("path", { d: d, fill: "none", stroke: "none" });
        layers.pulse.appendChild(path);
        var L = lenCache[key];
        if (L === undefined) {
          /* getTotalLength() 会逼一次 SVG 布局；同一条路径反复量就是纯浪费（240 个/秒时很致命） */
          L = path.getTotalLength();
          lenCache[key] = L;
        }
        var c = pool.pop() || mk("circle", { r: 2.4, fill: "var(--ok)" });
        if (!c.parentNode) layers.pulse.appendChild(c);
        c.setAttribute("opacity", "0");          /* 还没到起飞时刻：先藏着 */
        pulses.push({ c: c, path: path, L: L, t0: startAt,
                      dur: 470 + Math.random() * 280 });
      }
      /* 一次 tick：点亮发起的那条 lane，再沿曲线放一个光点过去。
         多 lane 的阶段（合成/编码）拿不到「是哪一条 lane 干的」—— status 里不带这个，
         每帧都带一份会让 500ms 的那次轮询白白变大。所以按 tick 序号**轮转**，
         保证光点始终落在**真实存在的 lane** 上（不是编出来的位置）。 */
      function spawn(k, n, sched) {
        var si = KEYS.indexOf(k);
        if (si < 0) return;
        if (si === KEYS.length - 1) {
          for (var z = 0; z < n; z++) {
            seq += 1;
            var li = seq % Math.max(1, geo[k].lanes.length);
            flashOne(k, li, sched());
            rememberFlash(k, li);
          }
          return;
        }
        var to = si + 1;
        while (to < KEYS.length && geo[KEYS[to]] && geo[KEYS[to]].off) to += 1;
        var A = geo[k];
        var B = (to < KEYS.length) ? geo[KEYS[to]] : null;
        for (var j = 0; j < n; j++) {
          seq += 1;
          var ai = A.lanes.length ? (seq % A.lanes.length) : 0;
          var at = sched();          /* 这一发「应该」发生的时刻（铺在轮询窗口里） */
          flashOne(k, ai, at);
          rememberFlash(k, ai);
          if (!B || reduceMotion()) continue;
          var bi = B.lanes.length ? ((seq * 7 + 3) % B.lanes.length) : 0;
          /* 出边要挂在**去的那一列左边那个**腰点上：to 跳过空列时 to-1 != si，
             以前一律写 si 会取到别的腰点（只在有空列时才错，且会取到 undefined 而不画）。 */
          var ein = edgesIn[si + ":" + ai], eout = edgesOut[(to - 1) + ":" + bi];
          if (ein && eout) addPulse(si + ":" + ai + ":" + bi, ein.d + " " + eout.d, at);
        }
      }
      function kick() {
        if (raf) return;
        raf = window.requestAnimationFrame(tick);
      }
      function tick(t) {
        raf = 0;
        if (freezeAt > 0) t = freezeAt;      /* 截图定格（见 freezeAt 的说明） */
        /* ① 点亮：摘类 → 强制重排 → 挂类（动画才会重放；衰减途中再来一次会重新点亮，
              不同 lane 各挂各的，互不打断）。合到一帧里做，减少强制重排次数。 */
        /* 点亮是铺在窗口里的：只处理 at <= t 的那些，没到点的留到下一帧（不丢） */
        var keep = [];
        while (flashes.length) {
          var f = flashes.shift();
          if (f.at && f.at > t) { keep.push(f); continue; }
          var g = geo[f.k];
          var lane = g && g.lanes[f.i];
          if (!lane) continue;
          if (t - (lane.lastLit || -1e9) < 16) continue;   /* 同一帧里同一条 lane 只闪一次 */
          lane.lastLit = t;
          lane.rect.classList.remove("lit");
          void lane.rect.getBoundingClientRect();
          lane.rect.classList.add("lit");
          /* 每条 lane 自己记「这是第几次亮」：1.25s 后只允许**自己那一次**摘类。
             没有这个守卫时，同一 lane 在 1.25s 内又亮一次，前一次的定时器会把
             后一次的 .lit 提前摘掉（动画被掐断、图上忽然全灭）—— 而「叠加」正是
             用户要的效果：衰减中再来一次应该重新点亮，不是提前熄掉。 */
          lane.gen = (lane.gen || 0) + 1;
          /* 注意把 lane 本身也捕进来：只捕 rect 的话闭包里那个 lane 是外层循环变量，
             循环结束后指向最后一条 lane —— 守卫会比错对象，个别 lane 就一直亮着。 */
          (function (ln, rect, mine) {
            window.setTimeout(function () {
              if (mine !== ln.gen) return;        /* 已经被更新的一次点亮接管 */
              rect.classList.remove("lit");
            }, LIT_MS);
          })(lane, lane.rect, lane.gen);
        }
        if (keep.length) flashes = keep.concat(flashes);
        /* ② 光点：沿曲线 getPointAtLength 走；只写 cx/cy/opacity，不读布局。 */
        for (var i = pulses.length - 1; i >= 0; i--) {
          var p = pulses[i];
          var kk = (t - p.t0) / p.dur;
          if (kk < 0) {                     /* 还没到起飞时刻：藏在源 lane 边上 */
            if (p.c.getAttribute("opacity") !== "0") p.c.setAttribute("opacity", "0");
            continue;
          }
          kk = Math.min(1, kk);
          var pt = p.path.getPointAtLength(p.L * kk);
          p.c.setAttribute("cx", pt.x.toFixed(1));
          p.c.setAttribute("cy", pt.y.toFixed(1));
          p.c.setAttribute("opacity", (0.9 * (1 - kk * 0.5)).toFixed(2));
          if (kk >= 1) { recycle(p); pulses.splice(i, 1); }
        }
        /* 定格时只画这一帧就停（不然 pulses 永远不结束 = 白烧 rAF） */
        if (!freezeAt && (pulses.length || flashes.length)) kick();
      }

      /* 喂一拍 status：列标题计数、点亮、光点、卡住，全在这里
         （inflight 后端照给，只是界面不再显示「在飞 N」） */
      function paint(d) {
        d = d || {};
        var right = $("mgPipeRight");
        var rate = Number((d.vid || {}).frame_fps || d.frame_fps || 0);
        if (right) right.textContent = rate > 0 ? (rate.toFixed(1) + " 帧/秒") : "— 帧/秒";
        var live = isLive(d.stages) ? d.stages
                 : (isLive(d.stages_live) ? d.stages_live : null);
        if (!live) {
          /* 还没开单 / 后端还没起流水线（模型加载那几秒）：先把六列摆出来当占位，
             别让这块是个空白框；真 lanes 一到，签名变了会自动重建。 */
          if (!layers) build(lanesOf(null));
          for (var z = 0; z < KEYS.length; z++) setCount(KEYS[z], null);
          return;
        }
        var lanes = lanesOf(live);
        if (signature(lanes) !== sig) build(lanes);

        /* ---- 水位 + 无脸旁路：两个读数都只用 status 里的真数字 ----
           水位 = 这一单攒了多少帧（缓存列）/ 已经变成成片的多少帧（成片杯）；
           旁路 = 这一单有多少帧根本没有脸（det.no_face_ticks，后端低成本计数）。
           收工那一拍：缓存清零、成片满格（cache 目录删了/帧都进了成片）。 */
        var framesTotal = Number(((d.vid || {}).total) || d.frames_total || d.total || 0);
        var composed = Math.max(0, Number((live.compose || {}).ticks) || 0);
        var coded = Math.max(0, Number((live.encode || {}).ticks) || 0);
        var cacheLv = framesTotal > 0 ? ((composed - coded) / framesTotal) : 0;
        var outLv = framesTotal > 0 ? (coded / framesTotal) : 0;
        var finished = !d.running &&
                       (d.ok === true || d.phase === "done" || d.state === "done");
        if (finished) { cacheLv = 0; outLv = 1; }
        setWater(cacheLv, outLv);
        var nf = Math.max(0, Number((live.det || {}).no_face_ticks) || 0);
        var dnf = (prevNf === undefined) ? 0 : Math.max(0, nf - prevNf);
        prevNf = nf;
        setBypass(nf);

        var t = nowMs();
        var running = !!d.running;
        var wants = [], total = 0, i, k, row, ticks, delta, n;
        for (i = 0; i < KEYS.length; i++) {
          k = KEYS[i]; row = live[k] || {};
          ticks = Math.max(0, Number(row.ticks) || 0);
          setCount(k, ticks);
          delta = (prev[k] === undefined) ? 0 : (ticks - prev[k]);
          if (delta > 0) lastAt[k] = t;
          else if (lastAt[k] === undefined) lastAt[k] = t;  /* 第一次看到：从这儿算「多久没动」 */
          prev[k] = ticks;
          wants.push(delta > 0 ? delta : 0);
          if (delta > 0) total += delta;
          /* 卡住 = 还在跑 + 这一段没完成 + 已经 >600ms 没 tick（tick 一恢复就自动复原） */
          markStall(k, running && state[k] !== "done" &&
                       (t - (lastAt[k] || t)) > STALL_MS);
        }
        /* 这一拍放多少个光点：按 PULSE_PER_S 折算的预算**按比例**稀释（各段比例不变），
           再把这个总数的每一发均匀铺在 POLL_MS 窗口里 —— 20 帧/秒时就是每 250/5 = 50ms
           一发，肉眼是连续流，而不是「每半秒跳一波」。 */
        var budget = Math.max(4, Math.round(PULSE_PER_S * POLL_MS / 1000));
        var scale = total > budget ? (budget / total) : 1;
        var plan = [], wantSpawn = 0;
        for (i = 0; i < KEYS.length; i++) {
          n = Math.max(0, Math.round(wants[i] * scale));
          plan.push(n);
          wantSpawn += n;
        }
        lastSpawn = wantSpawn;
        /* 无脸帧也真的走旁路：按 no_face_ticks 的增量在旁路线上放几个点（同一份 status
           的真实数字）。预算被主路占满时就不放 —— 旁路是配角，不能把主路的点挤掉。 */
        var bSpawn = 0;
        if (dnf > 0 && bypassD && !reduceMotion()) {
          bSpawn = Math.max(0, Math.min(budget - wantSpawn,
                                        Math.max(1, Math.round(dnf * scale))));
        }
        lastBypass = bSpawn;
        if (wantSpawn + bSpawn > 0) {
          var slotMs = POLL_MS / (wantSpawn + bSpawn);
          var slot = 0;
          var sched = function () { slot += 1; return t + slot * slotMs; };
          lastSeen = {};
          for (i = 0; i < KEYS.length; i++) {
            if (plan[i] > 0) spawn(KEYS[i], plan[i], sched);
          }
          for (i = 0; i < bSpawn; i++) addPulse("bypass", bypassD, sched());
        }
        kick();
      }

      function reset() {
        var i;
        freezeAt = 0;
        lastFlashed = [];
        lastSeen = {};
        lastSpawn = 0;
        lastBypass = 0;
        prevNf = undefined;      /* 换一单：旁路光点不补放历史增量 */
        setWater(0, 0);
        setBypass(0);
        kick();                       /* 上一张照片可能把时钟按住了：重开后要重新起 ticker */
        for (i = 0; i < pulses.length; i++) recycle(pulses[i]);
        pulses = [];
        flashes = [];
        prev = {};
        lastAt = {};
        seq = 0;
        for (i = 0; i < KEYS.length; i++) {
          stallNow[KEYS[i]] = false;
          markStall(KEYS[i], false);
          setCount(KEYS[i], null);
          /* 换一单 / 重连：把上一单残留的 .lit 也摘掉（否则它要等各自的 1.25s
             定时器才消失，图上会有一片「没人点也亮着」的方块） */
          var g = geo[KEYS[i]];
          if (g) {
            for (var j = 0; j < g.lanes.length; j++) g.lanes[j].rect.classList.remove("lit", "stall");
          }
        }
        setStates({});
        var right = $("mgPipeRight");
        if (right) right.textContent = "— 帧/秒";
      }

      function debug() {
        var out = { keys: KEYS, sig: sig, pulses: pulses.length, pool: pool.length,
                    stalled: {}, ticks: {}, state: {}, lanes: {},
                    edges: 0, edges_in: 0, edges_out: 0, waists: {},
                    last_flashed: lastFlashed, last_spawn: lastSpawn,
                    last_bypass: lastBypass,
                    /* cache/out 是**口径源**（差值 / encode），blue/green 是成片杯真正画出来的两层 */
                    water: { cache: waterLv.cache, out: waterLv.out,
                             blue: waterLv.blue, green: waterLv.green },
                    bypass: bypassN, bypass_drawn: !!bypassD,
                    /* 旁路原文 + 两个关键点：验收/排障直接读，不用再从像素里猜。
                       merge = 与主路汇合的腰点；end = 杯左缘（贴着进杯实线走完最后一段）；
                       out_tail_c = 主路 out:out 去掉 M 的那一段 —— bypass_d 必须**以它结尾**，
                       否则就还是「两条线各画各的」。 */
                    bypass_d: bypassD,
                    bypass_merge: (bypassD && geo.out && geo.out.waist)
                      ? { x: geo.out.waist.x, y: geo.out.waist.y } : null,
                    bypass_end: (bypassD && geo.out && geo.out.waist)
                      ? { x: geo.out.cx - OUTW / 2, y: geo.out.waist.y } : null,
                    out_tail_c: (geo.out && geo.out.tailC) || "",
                    bypass_gen: bypassGen, bypass_drawn_gen: bypassDrawnGen,
                    max_pulse: MAX_PULSE, pulse_per_s: PULSE_PER_S, poll_ms: POLL_MS,
                    built: !!layers };
        var i, k;
        for (i = 0; i < KEYS.length; i++) {
          k = KEYS[i];
          out.stalled[k] = !!stallNow[k];
          out.ticks[k] = (prev[k] === undefined ? null : prev[k]);
          out.state[k] = state[k] || "";
          out.lanes[k] = geo[k] ? geo[k].lanes.length : 0;
        }
        for (i in edgesIn) { if (Object.prototype.hasOwnProperty.call(edgesIn, i)) out.edges_in += 1; }
        for (i in edgesOut) { if (Object.prototype.hasOwnProperty.call(edgesOut, i)) out.edges_out += 1; }
        out.edges = out.edges_in + out.edges_out;      /* 画出来的连线总条数 */
        for (i in waists) {
          if (Object.prototype.hasOwnProperty.call(waists, i)) {
            out.waists[i] = { x: Math.round(waists[i].x * 100) / 100,
                              y: Math.round(waists[i].y * 100) / 100 };
          }
        }
        return out;
      }

      function freeze(at) {
        freezeAt = (Number(at) > 0) ? Number(at) : 0;
        if (!freezeAt) kick();
      }

      return { paint: paint, reset: reset, setStates: setStates, debug: debug,
               freeze: freeze, KEYS: KEYS, isLive: isLive };
    })();

    /* 列标题的状态（和分段链那套 mgpPaint 保持同一个入参口径） */
    function mgpWant(opts) {
      if (opts.state === "err") return "err";
      if (opts.state === "done") return "done";
      if (opts.state === "idle") return "";
      return "on";
    }
    /* 画到第 upto 步：前面全部 done，当前步 on / done / err / idle */
    function mgpPaint(upto, opts) {
      opts = opts || {};
      var st = mgpWant(opts);
      var m = {}, k;
      for (k = 0; k < MGP_STAGES.length; k++) {
        m[MG_KEY[MGP_STAGES[k]]] = (k < upto) ? "done" : (k === upto ? st : "");
      }
      MG_PIPE.setStates(m);
    }
    /* 融合流水线：六段并行，六个列标题同一个状态 */
    function mgpFusedSteps(st) {
      var v = (st === "done" ? "done" : (st === "err" ? "err" : (st === "idle" ? "" : "on")));
      var m = {}, k;
      for (k = 0; k < MGP_STAGES.length; k++) m[MG_KEY[MGP_STAGES[k]]] = v;
      MG_PIPE.setStates(m);
    }
    /* 整条链跑完：老接口保留（调用点在这页有好几处），只在图上写一句结论。
       不再「六个点一起发光」—— 发光那一套随竖排六步一起去掉了。 */
    function mgpChainDone() {
      var sub = $("mgPipeSub");
      if (sub) sub.textContent = "六段全部完成";
    }
    function mgpRejectShow(d, onForce) {
      d = d || {};
      var box = $("mgpResult");
      var txt = $("mgpResultText");
      var cb = $("mgpCancel");
      if (cb) cb.hidden = true;
      ["mgpStepsBox", "mgpFoot"].forEach(function (id) {
        var el = $(id);
        if (el) el.hidden = true;
      });
      var ttl = $("mgpHdTitle");
      /* 强制尝试撞 OOM 那一档是「跑过才失败」，别再说「未启动」—— 用户会以为没动过 */
      if (ttl) ttl.textContent = d.vram_oom ? "确实不够（已回滚）" : "导出未启动";
      if (txt) {
        var line = mgVramLine(d);
        var tip = mgVramAdvice(d);
        txt.textContent = tip ? (line + " · " + tip) : line;
      }
      /* 「仍然尝试（可能 OOM）」：只有后端说 can_force 且有重试入口时才亮出来。
         用户口径：明说不理解为什么被拒，那他更想**试**一次 —— 拒绝面板得给这条路。 */
      var fb = $("mgpForceTry");
      if (fb) {
        var can = (typeof onForce === "function") && d.can_force !== false;
        fb.hidden = !can;
        /* mgpRejectHide / mgpResultShow 会把 display:none 写成**内联**样式；这里必须清掉，
           否则「被拒 → 离开 → 再被拒」之后那颗按钮再也回不来（内联样式压过类规则）。
           以前它能一直挂着，靠的正是「.btn 的 display 压过了 [hidden]」这个 bug —— 修好
           [hidden] 之后，这条显式的清场就成了必须的。 */
        fb.style.display = can ? "" : "none";
        fb.disabled = false;
        fb.title = "按最小档真跑一次；真 OOM 会自动回滚到这一页，不会把后端搞坏";
      }
      if (box) { box.hidden = false; box.classList.add("is-rej"); }
      return box;
    }
    /* 回到「正常进度页」的样子（用户改完设置再点一次时先把拒绝态收起来） */
    function mgpRejectHide() {
      var box = $("mgpResult");
      if (box) box.classList.remove("is-rej");
      var fb0 = $("mgpForceTry");
      if (fb0) { fb0.hidden = true; fb0.style.display = "none"; }   /* 离开被拒态就收起「仍然尝试」 */
      var ttl = $("mgpHdTitle");
      if (ttl) ttl.textContent = "正在导出";
      ["mgpStepsBox", "mgpFoot"].forEach(function (id) {
        var el = $(id);
        if (el) el.hidden = false;
      });
    }
    window.mgpRejectShow = mgpRejectShow;
    window.mgpRejectHide = mgpRejectHide;

    /* 显存数字 -> 「需要 2.44 GB，可用 0.60 GB」（后端 400 里的 need_mb / avail_mb） */
    function mgGb(mb) {
      var n = Number(mb);
      if (!isFinite(n) || n < 0) return "";
      return (n / 1024).toFixed(2) + " GB";
    }
    /* 一行话：优先用后端的 summary（它把「预留」和「本程序 / 其它程序 / 空闲」
       都摊在同一行里 —— 老文案只给 need/avail 两个数，用户当然算不出那 1.37GB 去哪了）。
       后端没给（老版本后端）就按同一口径在本地拼一遍，字段都来自 400 的顶层。 */
    function mgVramLine(d) {
      d = d || {};
      if (d.vram_oom) {
        return String(d.vram_oom.summary || d.vram_oom.msg || "强制尝试失败：确实不够，已回滚")
          .split("\n")[0];
      }
      if (d.summary) return String(d.summary).split("\n")[0];
      var need = mgGb(d.need_mb), avail = mgGb(d.avail_mb);
      if (!need || !avail) {
        return String(d.error || d.detail || "显存不足").split("\n")[0];
      }
      var bits = ["需要 " + need];
      if (d.free_mb != null) {
        bits.push("空闲 " + mgGb(d.free_mb) +
                  " 里预留 " + (d.reserve_mb != null ? mgGb(d.reserve_mb) : "?") +
                  " 给驱动/碎片 = 实际可用 " + avail);
      } else {
        bits.push("实际可用 " + avail);
      }
      if (d.own_mb != null || d.other_mb != null) {
        bits.push("本程序 " + mgGb(d.own_mb) +
                  " / 其它程序 " + (d.other_mb == null ? "?" : mgGb(d.other_mb)) +
                  " / 空闲 " + (d.free_mb == null ? "?" : mgGb(d.free_mb)));
      }
      return "显存不足：" + bits.join(" · ");
    }
    function mgVramAdvice(d) {
      var s = (d && d.suggest) || [];
      return String(s[0] || "").trim();
    }
    /* 同一条拒绝 4 秒内只弹一次（连点导出不叠 toast） */
    var MG_VRAM_TOAST = { at: 0, text: "" };
    function mgVramToast(d) {
      var line = mgVramLine(d);
      var now = Date.now();
      if (line === MG_VRAM_TOAST.text && (now - MG_VRAM_TOAST.at) < 4000) return null;
      MG_VRAM_TOAST.at = now;
      MG_VRAM_TOAST.text = line;
      return toast(line, mgVramAdvice(d) || "关掉占 GPU 的程序后重试", "error", 7000);
    }
    /* 被拒的统一收口：**一次**提示 + 拒绝态 + 复位三个忙闲位（还能再点一次）。
       被拒也要进进度页：用户口径是「只显示一行错误 + 返回导出设置」——
       停在设置页虽然也能再点，但就没有那个明确的出口和结论了。 */
    function mgVramReject(d, onForce) {
      d = d || {};
      window.__mergeExporting = false;
      window.__mgStageRunning = false;
      window.__mgFusedRunning = false;
      window.__mgVramChecking = false;
      var cb = $("mgpCancel");
      if (cb) { cb.disabled = false; cb.textContent = "取消导出"; }
      if (window.mgStep) window.mgStep("progress");
      /* onForce：这颗「仍然尝试（可能 OOM）」点下去干什么（见 mgVramPreflight）。
         不给就不亮 —— 预览这种没有重试入口的地方不会画出一颗死按钮。 */
      if (d.vram_oom) {
        /* 强制尝试撞 OOM：**不是**「未启动」，是「跑过、确实不够、已回滚」。
           文案里必须带上后端的 summary（需要在 / 可用 / 差多少），否则用户以为白点了一次。 */
        if (window.mgpRejectShow) window.mgpRejectShow(d, onForce);
        toast("确实不够，已回滚", String(d.summary || d.msg || "").slice(0, 200), "error", 9000);
        return false;
      }
      mgpRejectShow(d, onForce);
      mgVramToast(d);
      return false;
    }
    window.mgVramReject = mgVramReject;

    /* 「仍然尝试（可能 OOM）」的点击口：重试回调由 mgVramPreflight 挂在 window 上 */
    var mgForceBtn = $("mgpForceTry");
    if (mgForceBtn && !mgForceBtn.dataset.bound) {
      mgForceBtn.dataset.bound = "1";
      mgForceBtn.addEventListener("click", function () {
        var fn = window.__mgForceRetry;
        if (typeof fn !== "function") return;
        mgForceBtn.disabled = true;
        var old = mgForceBtn.textContent;
        mgForceBtn.textContent = "正在尝试…";
        try { fn(); } catch (e) { /* 重试自己挂了：留在被拒态，别把界面弄坏 */ }
        setTimeout(function () { mgForceBtn.disabled = false; mgForceBtn.textContent = old; }, 2500);
      });
    }

    /* 起 job 之前的显存预检：POST 同一份 body + dry_run=1。
       后端只算账（_vram_plan：不加载模型、不建 job、不开线程）：
         · 200 → 照常开跑；
         · 400 + need_mb/avail_mb/suggest → 这里就收口（**一次** toast + 拒绝态）；
         · 别的失败 → 不挡路，交给真请求自己报（免得两套错误文案）。
       回调 cb(proceed)：proceed=false 表示已经处理完（别开跑）。 */
    function mgVramPreflight(body, cb) {
      var probe = {};
      for (var k in body) {
        if (Object.prototype.hasOwnProperty.call(body, k)) probe[k] = body[k];
      }
      probe.dry_run = 1;
      var fired = false;                      /* 回调只许走一次：重复进 mgFusedStart 会发两次 POST */
      fetch("/api/merge/export", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify(probe)
      }).then(function (r) {
        return r.json().then(function (d) { return { status: r.status, ok: r.ok, d: d || {} }; });
      }).then(function (res) {
        if (fired) return;
        var d = res.d || {};
        if (res.ok && d.ok !== false) { fired = true; cb(true, d); return; }
        if (res.status === 400 && d.need_mb != null) {
          fired = true;
          /* 强制尝试：把 force=1 写进**调用方那一份 body**（同一个对象引用），
             于是真请求（mgFusedStart / mgStageStart）也会带着它 —— 否则预检放行、
             真请求又被 400，用户看到的是「点了按钮还是不行」。 */
          var forceRetry = function () {
            body.force = 1;
            window.__mgVramChecking = true;
            mgVramPreflight(body, function (proceed2, d2) {
              window.__mgVramChecking = false;
              cb(proceed2, d2);
            });
          };
          window.__mgForceRetry = forceRetry;
          mgVramReject(d, forceRetry);        /* 一次 toast + 拒绝态 + 复位忙闲位 */
          cb(false, d);
          return;
        }
        fired = true;
        cb(true, d);                          /* 别的错误交给真请求自己报 */
      }).catch(function () {                  /* 预检自己挂了：别把用户挡在门外 */
        if (fired) return;
        fired = true;
        cb(true, {});
      });
    }
    window.mgVramPreflight = mgVramPreflight;

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
    function mgpEffShowWarn(warns) {
      var box = $("mgpEffWarn");
      if (!box) return;
      var texts = [];
      for (var i = 0; i < (warns || []).length; i++) {
        var w = warns[i] || {};
        var t = w && w.text ? String(w.text) : String(w || "");
        if (t) texts.push(t);
      }
      if (!texts.length || MGP_EFF.dismissed) { mgpEffHideWarn(); return; }
      var title = $("mgpEffWarnTitle"), body = $("mgpEffWarnBody");
      if (title) title.textContent = "生效配置与界面可能不一致（" + texts.length + " 项）";
      if (body) body.textContent = texts.join("\n");
      box.hidden = false;
      box.classList.remove("hide");
    }
    /* 画抽屉里的 14 个格子。force=false 时按 paintKey 判重：
       status 每 500ms 轮询一次，同一份配置不重画 DOM。 */
    function mgpEffPaint(force) {
      var grid = $("mgpEffGrid"), routeEl = $("mgpEffRoute");
      var eff = MGP_EFF.eff;
      if (!grid || !eff) return false;
      var paintKey = String(eff.route || "") + "|" + String(eff.at || "") + "|" +
                     String(eff.kind || "") + "|" + String((MGP_EFF.warns || []).length);
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
        var s = document.createElement("i"); s.className = "s " + src;
        s.textContent = mgpEffSrcText(src);
        if (eff.src_note && eff.src_note[key]) s.title = String(eff.src_note[key]);
        item.appendChild(k); item.appendChild(b); item.appendChild(s);
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
      if (warns && warns.length) MGP_EFF.warns = warns;
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
          bits.push("导出完成");
          if (d.out_video || d.out) bits.push(String(d.out_video || d.out));
          if (d.cache_removed) bits.push("cache 已清理");
          if (d.fused) bits.push("快速导出（融合流水线）· 未建 cache");
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
        detector: payload.detector || "",
        thresh: 0.5,
        settings: mgMergeSettings(payload),
        quality: payload.align_quality || "high",
        force: false
      };
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
        detector: payload.detector || "",
        thresh: 0.5,
        settings: mgMergeSettings(payload),
        quality: payload.align_quality || "high"
      };
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
        workers: parseInt(payload.num_workers, 10) || 0,
        encoder: payload.encoder || "h264_nvenc",
        hwaccel: payload.hwaccel === "" ? "off" : (payload.hwaccel || "auto"),
        settings: mgMergeSettings(payload)
      };
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
  /* 保存在工作区里（face68.json），换端口、换机器都还在 */
  function loadFace() {
    return fetch("/api/face").then(function (r) { return r.json(); }).then(function (d) {
      if (d && d.points && d.points.length === 68) {
        FACE_PTS = d.points;
        HEAD = null;
      }
    }).catch(function () {});
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

/* ===== 合成工作室 · 多轨时间轴（A）===== */
/* ============================================================
   合成工作室 · PR 风格多轨时间轴（片段）
   合并位置：app/web/js/ 独立引入（放在 app.js 之前或之后均可，勿改 app.js）
   暴露：window.MergeTimeline = { init(totalFrames), setFrame(n), getFrame() }

   参数命名全部来自 4.0 MergeStudio，未编造：
     state.totalFrames / state.currentFrame / state.fps
     state.cutSegments[{ start, end }]              （timeline.js 233 行）
     state.angleSegments[{ start, end, angles }]    （timeline.js 269 行）
     角度默认值 '0,90,180,270'                       （timeline.js 254 行）
     时间码格式 mm:ss.cs = 00:00.00                  （app.js updateTimecode）
     缩放档位 1 / 0.5 / 0.25 / 0.125                （index.html #res-select）
       → pxPerFrame 2 / 1 / 0.5 / 0.25（本片段映射，见 PPF）
     4.0 的 zoom-in / zoom-out 每次 ×2 / ÷2，范围 0.25 ~ 8
   老式写法：只用 var + function，不用 ES6 class / 箭头函数。
   ============================================================ */
(function () {
  'use strict';

  /* 4.0 四档缩放（值本身即 #res-select 的 option value） */
  var ZOOM_ORDER = [1, 0.5, 0.25, 0.125];
  var PPF = { '1': 2, '0.5': 1, '0.25': 0.5, '0.125': 0.25 };

  var RULER_STEP = 500;   /* 刻度尺每 500px 一个数字 */
  /* 视频轨以前铺 40 个均匀「缩略占位块」，纯装饰、和真实帧无关 ——
     用户明确要求不画（见 buildData）。这里不再保留占位块常量。 */

  var PLAY_SVG = '<svg viewBox="0 0 16 16" width="14" height="14" aria-hidden="true"><path d="M5.4 3.6 12.4 8l-7 4.4V3.6Z" fill="currentColor"/></svg>';
  var PAUSE_SVG = '<svg viewBox="0 0 16 16" width="14" height="14" aria-hidden="true"><rect x="4.6" y="3.8" width="2.6" height="8.4" rx="0.8" fill="currentColor"/><rect x="8.8" y="3.8" width="2.6" height="8.4" rx="0.8" fill="currentColor"/></svg>';

  var S = {
    ready: false,
    total: 0,
    frame: 0,
    fps: 30,          /* 4.0 state.fps；init 第二参数可覆盖 */
    zoom: 1,          /* 当前缩放档位，对应 PPF */
    ppf: 2,           /* pxPerFrame */
    dragging: false,
    playing: false,
    timer: null,
    data: null
  };

  function $(id) {
    return document.getElementById(id);
  }

  function clampFrame(n) {
    n = Math.round(n);
    if (!isFinite(n)) n = 0;
    if (n < 0) n = 0;
    if (S.total > 0 && n > S.total - 1) n = S.total - 1;
    return n;
  }

  function contentWidth() {
    return Math.max(1, Math.ceil(S.total * S.ppf));
  }

  /* ---------- 占位数据：形状与 4.0 的 cutSegments / angleSegments 一致 ---------- */
  function buildData(total) {
    /* 四条轨一律从「没有块」开始：
         · 视频轨 —— 以前那 40 个均匀占位块（缩略图条）是纯装饰，和真实帧一点关系
           都没有，而且一眼看上去像是「已经切好了片段」。用户要求直接不画，
           所以 video 恒为 []，只留刻度尺（renderRuler）和播放头。
         · 人脸 / 剪切 / 角度 —— 还没接后端就不造假数据，空轨道比假的红块诚实。
       有真实数据时由 MergeTimeline.setData({faces, cuts, angles}) 填。 */
    return { video: [], faces: [], cuts: [], angles: [] };
  }

  /* ---------- 渲染 ---------- */
  function addBlock(host, cls, s, e, title) {
    if (!host) return;
    var left = s * S.ppf;
    var w = (e - s) * S.ppf;
    var el = document.createElement('div');
    el.className = 'mtl-blk ' + cls;
    el.style.left = left + 'px';
    el.style.width = Math.max(2, w) + 'px';
    if (title) el.title = title;
    host.appendChild(el);
  }

  function clear(host) {
    if (host) host.innerHTML = '';
  }

  function renderRuler(w) {
    var host = $('mtlRuler');
    if (!host) return;
    clear(host);
    var px, num, tk, label;
    for (px = 0; px <= w; px += RULER_STEP) {
      num = Math.round(px / S.ppf);
      tk = document.createElement('div');
      tk.className = 'mtl-tk';
      tk.style.left = px + 'px';
      label = document.createElement('b');
      label.textContent = num;
      tk.appendChild(label);
      host.appendChild(tk);
    }
  }

  function renderVideo() {
    var host = $('mtlVideo');
    var i, d = S.data;
    if (!host || !d) return;
    clear(host);
    /* 没有块就整行收起（左边的「视频」标签一起收）—— 和其他三条轨同一套规矩。
       video 恒为 []，所以这一轨平时不出现；将来 setData({video:[…]}) 给了真数据
       会自己长回来。 */
    hideRow(host, !d.video.length);
    for (i = 0; i < d.video.length; i++) {
      addBlock(host, 'mtl-blk--video', d.video[i].s, d.video[i].e, '帧 ' + d.video[i].s + ' ~ ' + d.video[i].e);
    }
  }

  function renderFaces() {
    var host = $('mtlFaces');
    var i, d = S.data;
    if (!host || !d) return;
    clear(host);
    hideRow(host, !d.faces.length);
    for (i = 0; i < d.faces.length; i++) {
      addBlock(host, 'mtl-blk--face', d.faces[i].s, d.faces[i].e, '人脸 帧 ' + d.faces[i].s + ' ~ ' + d.faces[i].e);
    }
  }

  /* 轨道和它左边的标签名在两个不同的容器里，用序号对齐 */
  function hideRow(host, empty) {
    var stack = $('mtlStack');
    if (!host || !stack) return;
    var idx = Array.prototype.indexOf.call(stack.children, host);
    var names = document.querySelectorAll('#p-merge .mtl-name');
    if (idx >= 0 && names[idx]) names[idx].style.display = empty ? 'none' : '';
    host.style.display = empty ? 'none' : '';
  }

  function renderCut() {
    var host = $('mtlCut');
    var i, d = S.data;
    if (!host || !d) return;
    clear(host);
    hideRow(host, !d.cuts.length);
    for (i = 0; i < d.cuts.length; i++) {
      addBlock(host, 'mtl-blk--cut', d.cuts[i].s, d.cuts[i].e, '剪切 帧 ' + d.cuts[i].s + ' ~ ' + d.cuts[i].e);
    }
  }

  function renderAngle() {
    var host = $('mtlAngle');
    var i, d = S.data;
    if (!host || !d) return;
    clear(host);
    hideRow(host, !d.angles.length);
    for (i = 0; i < d.angles.length; i++) {
      addBlock(host, 'mtl-blk--angle', d.angles[i].s, d.angles[i].e, '角度 ' + d.angles[i].angles);
    }
  }

  function renderAll() {
    var stack = $('mtlStack');
    if (!stack) return;
    var w = contentWidth();
    stack.style.width = w + 'px';
    renderRuler(w);
    renderVideo();
    renderFaces();
    renderCut();
    renderAngle();
    placeHead();
    var tot = $('mtlTotal');
    if (tot) tot.textContent = String(S.total);
  }

  /* 播放头定位：只写 transform，不动 left。
     播放时一秒钟 30 帧，改 left 会让浏览器每帧重排整条栈；translateX 只重合成。
     x = 帧 × 每帧像素（ppf），和刻度、轨道块、缩放共用同一个 S.ppf。
     CSS 里 .mtl-head 有 margin-left:-1px：2px 线的**中心**正好落在 x 上。 */
  function placeHead() {
    var head = $('mtlHead');
    if (!head) return;
    var x = S.frame * S.ppf;
    head.style.transform = 'translate3d(' + x + 'px,0,0)';
    head.setAttribute('data-x', String(x));
  }

  /* 时间码：算法与 4.0 app.js updateTimecode 一致 → mm:ss.cs */
  function updateTimecode(idx) {
    var el = $('mtlTime');
    if (!el || !(S.fps > 0)) return;
    var s = idx / S.fps;
    var m = Math.floor(s / 60);
    s = Math.floor(s % 60);
    var cs = Math.floor((idx / S.fps % 1) * 100);
    el.textContent = (m < 10 ? '0' : '') + m + ':' + (s < 10 ? '0' : '') + s + '.' + (cs < 10 ? '0' : '') + cs;
  }

  function syncZoomSeg() {
    var seg = $('mtlZoomSeg');
    if (!seg) return;
    var items = seg.getElementsByTagName('b');
    var i, z;
    for (i = 0; i < items.length; i++) {
      z = parseFloat(items[i].getAttribute('data-zoom'));
      if (z === S.zoom) items[i].className = 'on';
      else items[i].className = '';
    }
  }

  function ensureVisible(px) {
    /* 拖动中绝不能自动滚：滚动改 mtlStack 的 rect.left，下一帧 frameAt
       算出的帧号更大 → 再滚 → 正反馈，播放头会自己"飞"到末尾 */
    if (S && S.dragging) return;
    var sc = $('mtlScroll');
    if (!sc || !sc.clientWidth) return;
    var pad = 24;
    if (px < sc.scrollLeft) sc.scrollLeft = Math.max(0, px - pad);
    else if (px > sc.scrollLeft + sc.clientWidth - pad) sc.scrollLeft = px - sc.clientWidth + pad;
  }

  /* 通知宿主（如 app.js）：帧变化 */
  function fireFrame(n) {
    var root = $('mergeTracks');
    if (!root || typeof window.CustomEvent !== 'function') return;
    try {
      root.dispatchEvent(new CustomEvent('mtl:frame', { bubbles: true, detail: { frame: n, total: S.total } }));
    } catch (err) { /* 忽略：宿主未监听 */ }
  }

  /* ---------- 对外 API ---------- */
  function setFrame(n) {
    if (!S.ready) return 0;
    S.frame = clampFrame(n);
    placeHead();
    var inp = $('mtlFrame');
    if (inp) inp.value = String(S.frame);
    var tot = $('mtlTotal');
    if (tot) tot.textContent = String(S.total);
    updateTimecode(S.frame);
    ensureVisible(S.frame * S.ppf);
    fireFrame(S.frame);
    return S.frame;
  }

  function getFrame() {
    return S.frame;
  }

  function applyZoom(z) {
    z = parseFloat(z);
    if (!isFinite(z) || !PPF[String(z)]) z = 1;
    var sc = $('mtlScroll');
    var centerPx = 0;
    if (sc && sc.clientWidth) centerPx = sc.scrollLeft + sc.clientWidth / 2;
    var centerFrame = Math.round(centerPx / S.ppf);
    S.zoom = z;
    S.ppf = PPF[String(z)];
    syncZoomSeg();
    renderAll();
    if (sc && sc.clientWidth) {
      sc.scrollLeft = Math.max(0, Math.round(centerFrame * S.ppf - sc.clientWidth / 2));
    }
    setFrame(S.frame);
  }

  function zoomStep(dir) {
    /* dir -1 = 放大（走向 1:1），+1 = 缩小（走向 1/8） */
    var i, idx = 0;
    for (i = 0; i < ZOOM_ORDER.length; i++) {
      if (ZOOM_ORDER[i] === S.zoom) idx = i;
    }
    idx = idx + dir;
    if (idx < 0) idx = 0;
    if (idx > ZOOM_ORDER.length - 1) idx = ZOOM_ORDER.length - 1;
    applyZoom(ZOOM_ORDER[idx]);
  }

  /* ---------- 播放（占位驱动；4.0 由 video 元素驱动） ---------- */
  function setPlayIcon(on) {
    var btn = $('mtlPlay');
    if (btn) btn.innerHTML = on ? PAUSE_SVG : PLAY_SVG;
  }

  /* 播放状态变化也往外说一声：宿主拿它来「播放瞬间清框」——
     帧目录模式由时间轴自己 setInterval 步进，第一拍要 33ms 之后才到，
     光等 mtl:frame 会让旧框多挂一帧。 */
  function firePlay(on) {
    var root = $('mergeTracks');
    if (!root || typeof window.CustomEvent !== 'function') return;
    try {
      root.dispatchEvent(new CustomEvent('mtl:play', { bubbles: true, detail: { playing: !!on, frame: S.frame } }));
    } catch (err) { /* 忽略：宿主未监听 */ }
  }

  /* announce=1 才通知宿主「停了」：拖动/逐帧/换素材这些调用点后面紧跟一次
     setFrame（自己会开一条新链路），再补一发「暂停」只会白跑一趟检测 + 合成。 */
  function stopPlay(announce) {
    var was = S.playing;
    if (S.timer) {
      clearInterval(S.timer);
      S.timer = null;
    }
    S.playing = false;
    setPlayIcon(false);
    if (was && announce) firePlay(false);
  }

  function startPlay() {
    if (S.total <= 0) return;
    firePlay(true);                 /* 先通知：播放开始 = 帧马上要变 */
    S.playing = true;               /* 先立状态：下面 setFrame(0) 也要按「播放中」处理 */
    if (S.frame >= S.total - 1) setFrame(0);
    setPlayIcon(true);
    var step = Math.max(30, Math.round(1000 / Math.max(1, S.fps)));
    S.timer = setInterval(function () {
      if (S.frame >= S.total - 1) {
        stopPlay();
        return;
      }
      setFrame(S.frame + 1);
    }, step);
  }

  function togglePlay() {
    if (S.playing) stopPlay(true);
    else startPlay();
  }

  /* ---------- 交互 ---------- */
  /* 抓住播放头拖动时，指针和播放头之间会差几个像素；不补偿的话，
     1/4 缩放下（ppf=0.5）一按下就跳 8 帧。 */
  var HEAD_GRAB = 0;

  function headHit(t) {
    var h = $('mtlHead');
    return !!(h && t && (t === h || (h.contains && h.contains(t))));
  }

  function frameAt(clientX) {
    var stack = $('mtlStack');
    if (!stack) return -1;
    var rect = stack.getBoundingClientRect();
    return clampFrame((clientX - HEAD_GRAB - rect.left) / S.ppf);
  }

  function onDown(e) {
    var sc = $('mtlScroll');
    if (!sc || e.button !== 0) return;
    if (e.target === sc) return;          /* 点在滚动条本体上：交给原生滚动 */
    HEAD_GRAB = 0;
    if (headHit(e.target)) {
      /* 抓住把手：记下指针相对播放头的偏移，拖动时保持这个相对位置 */
      var stack0 = $('mtlStack');
      var hx = stack0 ? (stack0.getBoundingClientRect().left + S.frame * S.ppf) : e.clientX;
      HEAD_GRAB = e.clientX - hx;
    }
    S.dragging = true;
    stopPlay();
    var root = $('mergeTracks');
    if (root) root.className = root.className.indexOf('is-dragging') < 0 ? root.className + ' is-dragging' : root.className;
    setFrame(frameAt(e.clientX));
    e.preventDefault();
  }

  function onMove(e) {
    if (!S.dragging) return;
    setFrame(frameAt(e.clientX));
    e.preventDefault();
  }

  function onUp() {
    if (!S.dragging) return;
    S.dragging = false;
    HEAD_GRAB = 0;
    var root = $('mergeTracks');
    if (root) root.className = root.className.replace(/\s*is-dragging/g, '');
  }

  function bindEvents() {
    var sc = $('mtlScroll');
    if (sc) {
      sc.addEventListener('mousedown', onDown);
      /* 4.0 行为：滚轮纵向滚动 → 时间轴横向滚动 */
      sc.addEventListener('wheel', function (e) {
        sc.scrollLeft += e.deltaY;
        e.preventDefault();
      }, { passive: false });
    }
    document.addEventListener('mousemove', onMove);
    document.addEventListener('mouseup', onUp);

    var inp = $('mtlFrame');
    if (inp) {
      /* 边打边跳：只有「是数字并且不越界」才真的 setFrame，
         半截输入（''、'-'）、非数字（abc）、超尾帧都不动画面 */
      inp.addEventListener('input', function () {
        var v = parseInt(this.value, 10);
        if (isNaN(v) || v < 0 || v > S.total - 1) return;
        setFrame(v);
      });
      inp.addEventListener('change', function () {
        var v = parseInt(this.value, 10);
        if (!isNaN(v)) setFrame(v);
        /* 非法输入以前会原样留在框里：change 末尾一律拨回当前帧 */
        this.value = String(S.frame);
      });
      inp.addEventListener('keydown', function (e) {
        if (e.key === 'Enter') {
          var v = parseInt(this.value, 10);
          if (!isNaN(v)) setFrame(v);
          this.value = String(S.frame);
          this.blur();
        } else if (e.key === 'ArrowLeft') {
          e.preventDefault();
          stopPlay();
          setFrame(S.frame - (e.shiftKey ? 10 : 1));
          this.value = String(S.frame);
        } else if (e.key === 'ArrowRight') {
          e.preventDefault();
          stopPlay();
          setFrame(S.frame + (e.shiftKey ? 10 : 1));
          this.value = String(S.frame);
        } else if (e.key === 'Escape') {
          e.preventDefault();
          this.value = String(S.frame);
          this.blur();
        }
      });
    }

    var prev = $('mtlPrev');
    if (prev) prev.addEventListener('click', function () { stopPlay(); setFrame(S.frame - 1); });

    var next = $('mtlNext');
    if (next) next.addEventListener('click', function () { stopPlay(); setFrame(S.frame + 1); });

    var play = $('mtlPlay');
    if (play) play.addEventListener('click', togglePlay);

    var zo = $('mtlZoomOut');
    if (zo) zo.addEventListener('click', function () { zoomStep(1); });

    var zi = $('mtlZoomIn');
    if (zi) zi.addEventListener('click', function () { zoomStep(-1); });

    var seg = $('mtlZoomSeg');
    if (seg) {
      seg.addEventListener('click', function (e) {
        var b = e.target;
        if (!b || b.tagName !== 'B') return;
        applyZoom(b.getAttribute('data-zoom'));
      });
    }

    /* 页面级快捷键：只在合成页激活（#p-merge 带 .on）时生效，且不抢输入控件/可编辑区的键。
       ←/→ 逐帧（Shift 十帧）；空格播放/暂停。 */
    document.addEventListener('keydown', function (e) {
      var page = document.getElementById('p-merge');
      if (!page || !page.classList || !page.classList.contains('on')) return;
      if (e.ctrlKey || e.metaKey || e.altKey) return;
      var t = e.target || {};
      var tag = String(t.tagName || '').toUpperCase();
      if (tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT' || t.isContentEditable) return;
      if (e.key === 'ArrowLeft') {
        e.preventDefault();
        stopPlay();
        setFrame(S.frame - (e.shiftKey ? 10 : 1));
      } else if (e.key === 'ArrowRight') {
        e.preventDefault();
        stopPlay();
        setFrame(S.frame + (e.shiftKey ? 10 : 1));
      } else if (e.key === ' ' || e.key === 'Spacebar') {
        e.preventDefault();
        togglePlay();
      }
    });
  }

  /* ---------- 初始化 ----------
     opts.keep = true：重进合成页的「续上」——保住播放头 S.frame 和已经算好的
     人脸轨 S.data.faces，只按新的 total 重排视频轨和刻度；事件不重复绑定。
     换素材时不要传 keep：人脸轨属于上一个素材，必须清掉，播放头归零。 */
  function init(totalFrames, fps, opts) {
    var root = $('mergeTracks');
    if (!root) return null;
    var keep = !!(opts && opts.keep);
    var oldFrame = S.frame;
    var oldFaces = (S.data && S.data.faces) ? S.data.faces : null;
    S.total = Math.max(0, Math.floor(isFinite(totalFrames) ? totalFrames : 0));
    if (fps && fps > 0) S.fps = fps;
    if (!S.ready) {
      bindEvents();
      S.ready = true;
    }
    stopPlay();
    S.data = buildData(S.total);
    if (keep && oldFaces) S.data.faces = oldFaces;
    /* keep 但没素材（total 0）时没有帧可停，回 0 */
    S.frame = (keep && S.total > 0) ? clampFrame(oldFrame) : 0;
    renderAll();
    return setFrame(S.frame);
  }

  window.MergeTimeline = {
    /* 验收/排查用：当前每帧像素、播放头的逻辑 x（= 帧 × ppf）、缩放档位 */
    getPpf: function () { return S.ppf; },
    getZoom: function () { return S.zoom; },
    getHeadX: function () { return S.frame * S.ppf; },
    isPlaying: function () { return !!S.playing; },
    seek: function (n) { stopPlay(); return setFrame(n); },
    /* 用真实数据覆盖演示数据（人脸轨、剪切段、角度带） */
    setData: function (partial) {
      if (!S.data || !partial) return;
      if (partial.video) S.data.video = partial.video;
      if (partial.faces) S.data.faces = partial.faces;
      if (partial.cuts) S.data.cuts = partial.cuts;
      if (partial.angles) S.data.angles = partial.angles;
      renderAll();
    },
    getData: function () { return S.data; },
    init: init,
    setFrame: setFrame,
    getFrame: getFrame
  };

  /* 占位自启动：宿主未显式调用 init 时先把空轴渲染出来（total 0）。
     以前这里按 300 帧渲染演示数据 —— 素材还没打开就显示 300 帧，全是假的；
     宿主随后调用 MergeTimeline.init(state.totalFrames) 会整条重绘覆盖。 */
  document.addEventListener('DOMContentLoaded', function () {
    if (!S.ready) init(0);
  });
})();

/* ===== 合成工作室 · 人脸库/导出（B）===== */
/* ============================================================
   合成工作室 · 页内二级页控制（人脸库 / 导出设置）
   暴露：window.MergePages = { show(name), settings(), ... }
   参数名全部取自 4.0：
     routes_export.py ExportStartRequest
       image_format 'jpg' | 'png' · encoder 'h264_nvenc' | 'libx264'
       hwaccel 'auto' | '' · output_path str · num_workers 0~16
   本文件只写 DOM，不发请求；点击「开始导出」派发事件交给主 app。
   ============================================================ */
(function () {
  'use strict';

  var WORKERS_MIN = 0;                 // num_workers 下限（0 = 自动）
  var WORKERS_MAX = 16;                // num_workers 上限

  function one(sel, root) { return (root || document).querySelector(sel); }
  function all(sel, root) {
    return Array.prototype.slice.call((root || document).querySelectorAll(sel));
  }

  var MergePages = {
    current: 'main',

    /* 切换页面：name ∈ 'facedb' | 'settings' | 'main'
       'main' 表示收起所有二级页，回到合成主视图 */
    show: function (name) {
      name = String(name || 'main');
      var target = null;
      all('.mgview').forEach(function (el) {
        /* 页容器在 index.html 里是 .mgview[data-view]（main / facedb / settings /
           progress / project），这里以前只认 data-page —— 于是 .on 和 .mgp-on
           从来没挂上过、current 恒为 'main'，全靠 mgStep 的 .in 兜着。两种都认。 */
        var view = el.getAttribute('data-view') || el.getAttribute('data-page') || '';
        var on = (view === name) && (name !== 'main');
        el.classList.toggle('on', on);
        if (on) {
          target = el;
          el.setAttribute('aria-hidden', 'false');
        } else {
          el.setAttribute('aria-hidden', 'true');
        }
      });

      /* 主视图让位（CSS：#p-merge.mgp-on > .mgbody / .mgtl 隐藏）。
         name==='main' 一律清掉：主视图自己也带 data-view='main'，留着 .mgp-on
         就会命中 [data-view='main']{display:none}，回主视图直接一片空白。 */
      var merge = document.getElementById('p-merge');
      if (merge) merge.classList.toggle('mgp-on', name !== 'main' && !!target);

      this.current = (name === 'main' || !target) ? 'main' : name;
      return this.current;
    },

    /* 这里以前有 buildFaceGrid(n) / setFaceCount(n)：往 #mgFaceGrid 里塞 n 个
       「序号 + 棋盘格」占位缩略图，再 grid.hidden = (n === 0) 把它显示出来。
       那是 4.0 人脸库的**旧占位网格**的渲染器，页面重做成 .fdb（face_db.js 画真数据）之后
       只剩它自己还在跑 —— 而 #mgFaceGrid 这个 id 现在属于**舞台里的单脸六宫格**
       （index.html 里 .mgfgrid#mgFaceGrid），于是 getElementById 抓到的正是舞台那一块：
       初始化时 24 个 .mgp-face 被写进舞台，hidden 又被抹成 false，
       .mgfgrid 是 position:absolute/inset:0 的 flex column —— 合成器一打开，
       舞台正中就顶上来一列 1..24 的编号空方块，把画面和空态全盖住。
       人脸库现在只有一侧真源（face_db.js + /api/faces），占位网格连同渲染器一起删掉：
       不生成，不是隐藏。别再往 #mgFaceGrid 里写任何人脸库的东西。 */

    /* 读取某个 seg 当前选中项的 data-v（没有 data-v 时退化成文字） */
    readSeg: function (key) {
      var seg = one('#p-merge .mgview .seg[data-k="' + key + '"]');
      if (!seg) return '';
      var on = seg.querySelector('b.on');
      if (!on) return '';
      var v = on.getAttribute('data-v');
      return v === null ? on.textContent.trim() : v;
    },

    /* 采集导出参数：字段名与 /api/export/start 一致 */
    settings: function () {
      var out = one('#mgOutPath');
      /* 「输出目录」这一行现在是可编辑的路径框（原来是 readonly 回显）。
         默认值仍由 mgFillExportSettings 填（= 后端本来就会用的位置），
         所以只有用户真改过（__mgDefOutEdited）且填的是 Windows 绝对路径，
         才把它当 output_path 发出去 —— 不动它就和以前完全一致。 */
      var dirOut = one('#mgDefOut');
      var dirVal = dirOut ? String(dirOut.value || '').trim() : '';
      var dirUsed = !!window.__mgDefOutEdited && /^[A-Za-z]:[\\/]|^\\\\/.test(dirVal);
      var workers = one('#mgWorkers');
      var nw = parseInt(workers ? workers.value : '0', 10);
      if (isNaN(nw)) nw = 0;
      nw = Math.min(WORKERS_MAX, Math.max(WORKERS_MIN, nw));
      var byId = function (id) {
        var el = document.getElementById(id);
        return el ? String(el.value || '').trim() : '';
      };
      var segOf = function (dk) {
        var seg = document.querySelector('#p-merge .seg[data-k="' + dk + '"] b.on');
        return seg ? seg.textContent.trim() : '';
      };
      var tiny = function (label) {
        var rows = document.querySelectorAll('#p-merge .mgright .mgrow');
        for (var i = 0; i < rows.length; i++) {
          var f = rows[i].querySelector('.f');
          if (f && f.textContent.indexOf(label) >= 0) {
            var inp = rows[i].querySelector('input');
            return inp ? (parseFloat(inp.value) || 0) : 0;
          }
        }
        return 0;
      };
      return {
        image_format: this.readSeg('image_format') || 'jpg',
        encoder: this.readSeg('encoder') || 'h264_nvenc',
        hwaccel: this.readSeg('hwaccel'),
        output_path: (out ? out.value.trim() : '') || (dirUsed ? dirVal : ''),
        num_workers: nw,
        /* 检测 */
        detector: byId('mgDetector'),
        face_type: byId('mgFaceType'),
        redetect: segOf('detect'),
        /* 合成 */
        mode: byId('mgMode'),
        mask_source: segOf('seg'),
        mask_mode: segOf('mask'),
        color_transfer: byId('mgColor'),
        /* 融合方式的真源 = 「模式」下拉（4 个）；分段只是快捷方式 */
        blend: (function () {
          var m = String(byId('mgMode') || '');
          if (m === 'overlay' || m === 'hist-match' || m === 'seamless'
              || m === 'seamless-hist-match') return m;
          return segOf('blend') || 'seamless';
        })(),
        /* 新契约：ct_mode / blend / mask_mode(int) / mask_erode / mask_blur。
           导出和预览共用这一份，别让两条路各说一套。
           （blend 在上面只写一次 —— 同一个对象里写两次，后面的会把前面的盖掉。） */
        ct_mode: byId('mgColor') || 'mt',
        mask_mode_num: ({ 'prd': 1, 'dst': 2, 'prd×dst': 3 })[segOf('mask')] || 3,
        mask_erode: tiny('遮罩腐蚀'),
        mask_blur: tiny('遮罩羽化'),
        erode_mask: tiny('遮罩腐蚀'),
        blur_mask: tiny('遮罩羽化'),
        model: (document.querySelector('#mgModels .mgitem.on .nm') || {}).textContent || '',
        /* 多选契约：models=[{name, faces}]，导出事件 detail 里也带上。
           合成视图里按框分配过的，以那个为准（按脸分配优先于左栏多选）。
           导出真正发请求时用的是 mgExportModelsPayload()（全部选中项），
           这里留的是预览那一路的口径。 */
        models: (window.mgFaceModelsPayload ? window.mgFaceModelsPayload() : mgModelsPayload()),
        /* 导出设置页新增的两项：
           keep_cache —— 默认 false（成功导出后后端会删掉 <名字>_cache/，勾「保留」才留）；
           frame_from / frame_to —— 默认 0 / 空 = 全片，填了才随阶段请求带出去。 */
        keep_cache: String(this.readSeg('keep_cache') || '') === '1',
        /* 快速导出（融合流水线，后端 fused=1）：**默认开**。
           关掉 = 回退六阶段分段链（每步独立产物、可续跑），留给排查/对照。
           老页面没有这一项时按「开」处理 —— 默认行为不能因为少了个控件就变。 */
        fast_export: (function () {
          var seg = document.querySelector('#p-merge .mgview .seg[data-k="fast_export"]');
          if (!seg) return true;
          var on = seg.querySelector('b.on');
          var v = on ? on.getAttribute('data-v') : '';
          return String(v === null ? '1' : v) !== '';
        })(),
        frame_from: Math.max(0, parseInt(byId('mgFrameFrom'), 10) || 0),
        frame_to: Math.max(0, parseInt(byId('mgFrameTo'), 10) || 0),
        /* 性能那两个下拉（data-cfg 记忆的同一份值）：
             dfm_dtype      auto / fp32 / bf16 / bf16w    → body.dfm_dtype（auto 时不带）
             compose_backend auto / cpu / gpu             → body.compose_gpu（1 / 0，auto 时不带）
           组装请求体在 mgPerfBody（app.js 里那一份），这里只报「用户选了什么」。 */
        dfm_dtype: (function () {
          var el = document.getElementById('mgDfmDtype');
          return el ? String(el.value || 'auto') : 'auto';
        })(),
        compose_backend: (function () {
          var el = document.getElementById('mgCmpBackend');
          return el ? String(el.value || 'auto') : 'auto';
        })()
      };
    },

    /* 底部状态文字跟着当前设置走 */
    syncStatus: function (text) {
      var el = document.getElementById('mgExportStatus');
      if (!el) return;
      if (typeof text === 'string') {
        el.textContent = text;
        return;
      }
      var s = this.settings();
      var fmt = s.image_format === 'png' ? 'PNG' : 'JPEG';
      var dec = s.hwaccel ? '硬解' : '软解';
      var wk = s.num_workers > 0 ? s.num_workers + ' 进程' : '自动进程';
      el.textContent = '就绪 · ' + fmt + ' · ' + s.encoder + ' · ' + dec + ' · ' + wk;
    }
  };

  /* ---------- 合成器参数记忆 ---------- */
  var MG_PREF_KEY = "dfl-merge-prefs";
  function mgSavePrefs() {
    try {
      var p = (window.MergePages && window.MergePages.settings) ? window.MergePages.settings() : null;
      if (p) localStorage.setItem(MG_PREF_KEY, JSON.stringify(p));
    } catch (e) {}
  }
  function mgLoadPrefs() {
    var raw = null;
    try { raw = localStorage.getItem(MG_PREF_KEY); } catch (e) {}
    if (!raw) return;
    var p;
    try { p = JSON.parse(raw); } catch (e) { return; }
    var byId = { detector: "mgDetector", face_type: "mgFaceType", mode: "mgMode", color_transfer: "mgColor" };
    Object.keys(byId).forEach(function (k) {
      var el = document.getElementById(byId[k]);
      if (el && p[k] != null) el.value = p[k];
    });
    /* 存着老的模式值（debug / original / raw-*）时下拉会变成空值 → 回落到 seamless；
       再把「融合方式」两档分段同步成同一个值（否则两边亮的不一致）。 */
    var modeEl = document.getElementById("mgMode");
    if (modeEl) {
      var known = ["overlay", "hist-match", "seamless", "seamless-hist-match"];
      if (known.indexOf(String(modeEl.value || "")) < 0) modeEl.value = "seamless";
      if (window.mgSyncBlendSeg) window.mgSyncBlendSeg(modeEl.value);
    }
    /* seg 类按文字匹配 */
    var segMap = { detect: p.redetect, seg: p.mask_source, mask: p.mask_mode };
    Object.keys(segMap).forEach(function (dk) {
      var v = segMap[dk];
      if (!v) return;
      var seg = document.querySelector('#p-merge .seg[data-k="' + dk + '"]');
      if (!seg) return;
      Array.prototype.forEach.call(seg.children, function (b) {
        b.classList.toggle("on", b.textContent.trim() === String(v).trim());
      });
    });
    /* 数字行 */
    var numMap = { "遮罩腐蚀": p.erode_mask, "遮罩羽化": p.blur_mask };
    Object.keys(numMap).forEach(function (label) {
      if (numMap[label] == null) return;
      var rows = document.querySelectorAll('#p-merge .mgright .mgrow');
      for (var i = 0; i < rows.length; i++) {
        var f = rows[i].querySelector('.f');
        if (f && f.textContent.indexOf(label) >= 0) {
          var inp = rows[i].querySelector('input');
          if (inp) inp.value = numMap[label];
          break;
        }
      }
    });
  }
  window.mgSavePrefs = mgSavePrefs;
  window.mgLoadPrefs = mgLoadPrefs;

  /* ---------- 导出设置页的下拉记忆（data-cfg）----------
     和训练器参数页同一套键名约定：控件上打 data-cfg="dfm_dtype"，值按这个键存取。
     训练页那份记忆是模型 .dat 里的（后端给回来的）；导出设置这两颗是**用户偏好**，
     和主题 / 侧栏一样存本机 localStorage —— 刷新、关掉再开，还是上次选的那一档。
     为什么必须有：这两个值会改变画质 / 显存 / 速度。刷新一下就悄悄回到「自动」，
     用户会以为「设了没用」，然后去翻日志找原因。 */
  var MG_CFG_MEM_KEY = "dfl-mg-cfg-memory";
  function mgCfgMemRead() {
    try {
      var raw = localStorage.getItem(MG_CFG_MEM_KEY);
      var obj = raw ? JSON.parse(raw) : null;
      return (obj && typeof obj === "object") ? obj : {};
    } catch (e) { return {}; }
  }
  function mgCfgMemSave(key, val) {
    key = String(key || "");
    if (!key) return;
    var mem = mgCfgMemRead();
    mem[key] = String(val == null ? "" : val);
    try { localStorage.setItem(MG_CFG_MEM_KEY, JSON.stringify(mem)); } catch (e) {}
  }
  function mgCfgMemApply() {
    var mem = mgCfgMemRead();
    var els = document.querySelectorAll('#p-merge .mgview [data-cfg]');
    Array.prototype.forEach.call(els, function (el) {
      var k = el.getAttribute('data-cfg');
      if (!k || !Object.prototype.hasOwnProperty.call(mem, k)) return;
      var v = String(mem[k]);
      if (el.tagName === 'SELECT') {
        var hit = false;
        for (var i = 0; i < el.options.length; i++) {
          if (String(el.options[i].value) === v) { hit = true; break; }
        }
        /* 存的值不在选项里（老版本存的）就保留当前默认 —— 别把控件置成空值 */
        if (hit) el.value = v;
      } else if (String(el.value) !== v) {
        el.value = v;
      }
    });
  }
  window.mgCfgMemApply = mgCfgMemApply;
  window.mgCfgMemSave = mgCfgMemSave;
  window.__mgCfgMem = mgCfgMemRead;          /* 验收读快照用（只读） */
  (function bindCfgMem() {
    var view = document.querySelector('#p-merge .mgview[data-view="settings"]');
    if (!view || view.dataset.cfgMemBound) return;
    view.dataset.cfgMemBound = '1';
    view.addEventListener('change', function (e) {
      var el = e.target && e.target.closest ? e.target.closest('[data-cfg]') : null;
      if (!el) return;
      mgCfgMemSave(el.getAttribute('data-cfg'), el.value);
    });
  })();
  mgCfgMemApply();

  /* ---------- 事件绑定 ---------- */

  // seg 点击：只处理 .mgview 内的，避免和主页面其它 seg 抢事件
  function bindSeg() {
    document.addEventListener('click', function (e) {
      var b = e.target && e.target.closest ? e.target.closest('.mgview .seg > b') : null;
      if (!b) return;
      var seg = b.parentElement;
      all('b', seg).forEach(function (item) { item.classList.toggle('on', item === b); });
      MergePages.syncStatus();
    });
  }

  function bindActions() {
    var go = document.getElementById('mgStartExport');
    if (go) {
      go.addEventListener('click', function () {
        var payload = MergePages.settings();
        // 交由主 app 调 POST /api/export/start
        document.dispatchEvent(new CustomEvent('merge:export-start', { detail: payload }));
        MergePages.syncStatus('已提交导出 · ' + payload.image_format + ' · ' + payload.encoder);
      });
    }

    var compute = document.getElementById('mgComputeEmb');
    if (compute) {
      compute.addEventListener('click', function () {
        var btn = compute;
        var old = btn.textContent;
        btn.textContent = '计算中…';
        btn.disabled = true;
        // 交由主 app 调 POST /api/face/compute-embeddings
        document.dispatchEvent(new CustomEvent('merge:compute-embeddings', { detail: {} }));
        setTimeout(function () { btn.textContent = old; btn.disabled = false; }, 1200);
      });
    }

    /* 导出设置页两个路径框（输出目录 / 输出路径）右边那颗文件夹图标：
       走全局同一套 pickFolderFor（系统目录选择器 → 写回 .pathval 并派发 change）。
       以前 #mgOutPick 只派发 'merge:pick-output-dir'，而那条事件**没有任何监听者**，
       点了等于没点；现在两行图标绑同一份逻辑，行为一致。
       用 onclick 属性（不是 addEventListener）：bindPickPaths 也写这个属性，
       谁后绑定谁生效，不会一次点击弹两个对话框。 */
    function mgBindOutPickers() {
      var icons = document.querySelectorAll(
        '#p-merge .mgview[data-view="settings"] .pathinp .pill.iconpill');
      Array.prototype.forEach.call(icons, function (icon) {
        icon.onclick = function (e) {
          e.preventDefault();
          e.stopPropagation();
          var box = icon.closest ? icon.closest('.pathinp') : null;
          if (window.pickFolderFor) { window.pickFolderFor(box); return; }
          document.dispatchEvent(new CustomEvent('merge:pick-output-dir', { detail: {} }));
        };
      });
    }
    mgBindOutPickers();
    window.mgBindOutPickers = mgBindOutPickers;

    /* 「输出目录」被手改过才参与导出（默认值只是回显，不动它 = 与以前一致） */
    var defOut = document.getElementById('mgDefOut');
    if (defOut && !defOut.dataset.dirtyBound) {
      defOut.dataset.dirtyBound = '1';
      var markDirty = function () {
        window.__mgDefOutEdited = String(defOut.value || '').trim() !== '';
      };
      defOut.addEventListener('input', markDirty);
      defOut.addEventListener('change', markDirty);   /* 目录选择器写回来也走这条 */
    }

    var workers = document.getElementById('mgWorkers');
    if (workers) {
      workers.addEventListener('input', function () {
        var v = parseInt(workers.value, 10);
        if (isNaN(v)) { MergePages.syncStatus(); return; }
        if (v < WORKERS_MIN) { workers.value = String(WORKERS_MIN); }
        if (v > WORKERS_MAX) { workers.value = String(WORKERS_MAX); }
        MergePages.syncStatus();
      });
    }

    var out = document.getElementById('mgOutPath');
    if (out) out.addEventListener('input', function () { MergePages.syncStatus(); });
  }

  function init() {
    MergePages.show('main');          // 默认全部隐藏，停在主视图
    MergePages.syncStatus();
    bindSeg();
    bindActions();
    /* 记忆里的两档要**先**回填：syncStatus 只读格式/编码/进程，但用户在设置页
       第一眼就得看到上次选的精度/后端，而不是等某次渲染才补上。 */
    try { if (window.mgCfgMemApply) window.mgCfgMemApply(); } catch (e) {}
  }

  window.MergePages = MergePages;

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();

/* ===== 遮罩绘制页（子智能体 C）===== */
/* ===== 训练参数补齐（子智能体 D）===== */
/* ============================================================
 * 4.1 训练参数 · 补齐片段（与 app.js 的 CFG_GROUPS 同构）
 * ------------------------------------------------------------
 * 只放「4.0 有、4.1 CFG_GROUPS 缺」的参数行，不重复已有项。
 * 每个 k 都能在 4.0 源码里找到出处，逐条见文末 CITE 表 / audit.md 第四节。
 *
 * 拼法 A（推荐，零改动风险）
 *   在 app.js 的 CFG_GROUPS 定义之后追加：
 *     window.TRAIN_EXTRA_GROUPS.forEach(function (g) { CFG_GROUPS.push(g); });
 *   —— 需要本文件先于该行加载，或把这几个组直接拷进 CFG_GROUPS 数组里。
 *
 * 拼法 B（完全不碰 app.js）
 *   index.html 里加：
 *     <link rel="stylesheet" href="/static/css/train-params.css">
 *     <script src="/static/js/train-params.js"></script>
 *   然后调用 window.TrainExtra.init()。
 *   init() 会挂钩 window.openCfg，等它渲染完原有分组后，再把这一块补进 #cfgBox。
 *
 * ⚠ 关于 select
 *   device / preview_grade_alg 的候选项都会超过 4 个，按「≤4 项用 seg，>4 项用 select」
 *   必须用 select；而现有 renderer 只有 num / seg / sw / path 四种。
 *   所以这两行单独放在 TRAIN_EXTRA_SELECT_GROUPS（t:"sel"），由本文件的 renderGroups() 渲染。
 *   如果你要把它们并进 CFG_GROUPS，请给 openCfg() 加一个 t:"sel" 分支，
 *   做法见文件末尾 OPENCFG_SEL_PATCH 注释里的四行代码。
 * ============================================================ */
(function () {
  "use strict";

  /* ------------------------------------------------------------
   * 一、可直接 splice 进 CFG_GROUPS 的组（只用到 num / seg / sw / path）
   * ------------------------------------------------------------ */
  var EXTRA_GROUPS = [
    {
      t: "训练参数 · 补充",
      note: "4.0 training_config_page 同名项",
      rows: [
        { k: "masked_training", l: "遮罩训练", t: "sw", v: 1, d: "启用后只在遮罩区域内计算损失" },
        { k: "blur_out_mask", l: "遮罩羽化", t: "sw", v: 0, d: "对输入的 mask 外周做高斯模糊" },
        { k: "eyes_mouth_prio", l: "嘴眼优先", t: "sw", v: 0, d: "优先训练嘴部和眼部区域" },
        { k: "backup_interval", l: "备份间隔", t: "num", v: "1000", d: "每 N 次迭代自动保存一次，0 = 关闭" }
      ]
    },
    {
      t: "补丁参数 · 补充",
      note: "0 = 关闭",
      rows: [
        { k: "vgg_perceptual_power", l: "VGG感知", t: "num", v: "0.0", d: "VGG16 特征 L1 损失，0~100，50 约等于与重建损失等权" }
      ]
    },
    {
      t: "数据增强 · 补充",
      rows: [
        { k: "mouth_scale", l: "嘴巴范围", t: "num", v: "1.0", d: "配合嘴眼优先，1.0 = 原始范围，0.25~2.0" },
        { k: "random_src_flip", l: "SRC翻转", t: "sw", v: 0, d: "src 图像随机水平翻转" },
        { k: "random_dst_flip", l: "DST翻转", t: "sw", v: 0, d: "dst 图像随机水平翻转" },
        { k: "uniform_yaw", l: "均匀偏航", t: "sw", v: 0, d: "侧脸少的 faceset 用，让 yaw 分布均匀" }
      ]
    },
    {
      t: "冻结层 · 补充",
      note: "按模型架构生效",
      rows: [
        { k: "freeze_inter", l: "冻结中间", t: "sw", v: 0, d: "SAEHD / DFSingle / PIA 的 DF 架构 inter" },
        { k: "freeze_decoder_dst", l: "冻结输出", t: "sw", v: 0, d: "冻结 decoder_dst，仅 DF 架构有这一支" },
        { k: "freeze_inter_src", l: "冻结源", t: "sw", v: 0, d: "仅 AMP 的源中间层" },
        { k: "freeze_inter_dst", l: "冻结目标", t: "sw", v: 0, d: "仅 AMP 的目标中间层" },
        { k: "freeze_bottleneck", l: "冻结瓶颈", t: "sw", v: 0, d: "仅 DeepFakeLarge 的 Bottleneck" },
        { k: "freeze_decoderA_mask", l: "冻结A遮罩", t: "sw", v: 0, d: "仅 DeepFakeLarge 的 src 解码器遮罩分支" },
        { k: "freeze_decoderB_mask", l: "冻结B遮罩", t: "sw", v: 0, d: "仅 DeepFakeLarge 的 dst 解码器遮罩分支" },
        { k: "freeze_bottleneck_AB", l: "冻结共享", t: "sw", v: 0, d: "仅 LIAELarge 的共享瓶颈 Bottleneck_AB" },
        { k: "freeze_bottleneck_B", l: "冻结瓶颈B", t: "sw", v: 0, d: "仅 LIAELarge 的目标瓶颈 Bottleneck_B" }
      ]
    },
    {
      /* 预训练集是路径，属于"数据集从哪来"，不是优化器的事 ——
         单独成组并进「数据集参数」，插在 dst 人脸集后面（after: "dst"）。 */
      t: "数据集参数 · 补充",
      rows: [
        { k: "pretrain_dataset_path", l: "预训练集", t: "path", v: "", after: "dst", full: true,
          d: "多身份人脸集目录，留空 = 用 dst 人脸集兜底" }
      ]
    }
  ];

  /* ------------------------------------------------------------
   * 二、必须用 select 的两行（候选项 > 4）
   * ------------------------------------------------------------ */
  var EXTRA_SELECT_GROUPS = [
    {
      t: "设备",
      pin: "top",                 /* 用哪个设备训练是最需要一眼看到的信息，别沉到页尾 */
      noheader: true,             /* 标题挪到行里，卡片不再单独占一行标题 */
      rows: [
        {
          /* 一行就够：左边"设备"，右边下拉框 */
          /* 默认 "自动选择"：dat 里没有 device 这个键时，引擎就是自动挑卡。
             以前这里写死 "0"，看起来像"用户已经选了 0 号卡"——是骗人的。 */
          k: "device", l: "设备", t: "sel", v: "auto", d: "", titlelabel: true,
          /* 选项在运行时由 /api/devices 填（GPU 型号 / CPU 线程数），
             这里的只是网还没回来时的兜底 */
          o: [{ v: "auto", l: "自动选择" }, { v: "cpu", l: "CPU" }, { v: "0", l: "GPU 0" }]
        }
      ]
    },
    {
      t: "数据增强 · 补充",
      note: "需要 select",
      rows: [
        {
          k: "preview_grade_alg", l: "预览调色", t: "sel", v: "same",
          d: "只影响「合并预览」第 5 列的显示，不影响训练和真正的合成",
          o: ["same", "none", "rct", "mt", "lct", "mkl", "idt", "sot", "lut"]
        }
      ]
    }
  ];

  /* ------------------------------------------------------------
   * 三、渲染 / 绑定（无业务逻辑，只做控件形态）
   * ------------------------------------------------------------ */
  var FOLDER_SVG = '<span class="pill iconpill" title="选择目录">' +
    '<svg class="ic"><use href="#i-folder"/></svg></span>';

  function mk(tag, cls) {
    var e = document.createElement(tag);
    if (cls) e.className = cls;
    return e;
  }

  /* 建一行：标签列 68px，控件与说明同在第二列 */
  function buildRow(r) {
    var row = mk("div", "cfgrow");
    row.setAttribute("data-k", r.k || "");
    if (r.k) row.title = r.k + (r.d ? " · " + r.d : "");

    /* nolabel：卡片标题已经说明了这是什么，行内再写一遍就是重复 */
    if (!r.nolabel) {
      var cl = mk("span", "cl");
      /* titlelabel：这一行没有卡片标题，标签本身充当标题 —— 用标题的字体和颜色 */
      if (r.titlelabel) cl.className = "cl ttl";
      cl.textContent = r.l || "";
      row.appendChild(cl);
    }

    var cc = mk("div", "cc");

    if (r.t === "path") {
      row.className += " cfgpathrow";
      var box = mk("div", "inp mono pathinp cfgpath");
      if (r.full) {
        /* full：和宿主「src 人脸集 / dst 人脸集」完全同款 —— 标签在框内、整行就是输入框。
           预训练集放在它们下面，三种路径长得一致才不会显得突兀。 */
        row.innerHTML = "";
        box.style.cssText = "flex:1 1 auto;width:auto;min-width:0";   /* 铺满整行，和宿主 src/dst 一致 */
        var lb = mk("span", "lb");
        lb.textContent = r.l || "";
        box.appendChild(lb);
        var val2 = document.createElement("input");
        val2.className = "pathval";
        val2.spellcheck = false;
        val2.autocomplete = "off";
        val2.value = r.v || "";
        if (r.k) val2.dataset.cfg = r.k;
        box.appendChild(val2);
        box.innerHTML += FOLDER_SVG;
        /* 直接挂在行上，不套 .cc —— .cc 是"控件靠右"的容器（宽 auto），
           把整行的路径框塞进去只会缩成一小截。 */
        row.appendChild(box);
        if (r.d) row.title = r.d;
        return row;
      }
      var val = mk("span", "val");
      val.textContent = r.v || "未选择";
      box.appendChild(val);
      box.innerHTML += FOLDER_SVG;
      cc.appendChild(box);
      row.appendChild(cc);
      row.appendChild(hint(r));
      return row;
    }

    if (r.t === "num") {
      var inp = mk("input", "inp");
      /* data-cfg 必须打上：没有它 openCfg 回填扫不到、collectCfgPatch 也不报，
         表现就是"这个值改了刷新就回默认"（补齐块的 num/seg/sw 以前全都漏了）。 */
      if (r.k) inp.dataset.cfg = r.k;
      inp.value = r.v == null ? "" : String(r.v);
      inp.setAttribute("inputmode", "decimal");
      inp.spellcheck = false;
      cc.appendChild(inp);
    } else if (r.t === "seg") {
      var seg = mk("div", "seg");
      if (r.k) seg.dataset.cfg = r.k;
      (r.o || []).forEach(function (o, i) {
        var b = mk("b");
        b.textContent = o;
        b.dataset.v = String(o);            /* 业务值取这个，别取 textContent（会被翻译） */
        if (String(o) === String(r.v) || (i === 0 && r.v == null)) b.className = "on";
        seg.appendChild(b);
      });
      cc.appendChild(seg);
    } else if (r.t === "sel") {
      var sel = mk("select", "mgsel");
      if (r.k) sel.dataset.cfg = r.k;
      var opts = (r.k === "device" && window.__DEVOPTS) ? window.__DEVOPTS : (r.o || []);
      opts.forEach(function (o) {
        var op = document.createElement("option");
        var ov = (o && typeof o === "object") ? o.v : o;
        var ol = (o && typeof o === "object") ? o.l : o;
        op.value = String(ov);
        op.textContent = ol;
        if (String(ov) === String(r.v)) op.selected = true;
        sel.appendChild(op);
      });
      if (sel.selectedIndex < 0) sel.selectedIndex = 0;
      cc.appendChild(sel);
      /* 设备行：控件旁边常驻「当前可用 GPU 列表」（值由 /api/devices 提供） */
      if (r.k === "device" && window.__devGuard) window.__devGuard(sel);
    } else if (r.t === "sw") {
      var sw = mk("div", "sw" + (r.v ? " on" : ""));
      if (r.k) sw.dataset.cfg = r.k;
      sw.dataset.v0 = sw.classList.contains("on") ? "1" : "0";
      sw.setAttribute("role", "switch");
      sw.tabIndex = 0;
      cc.appendChild(sw);
    }

    /* 顺序必须是 标签 → 说明 → 控件，和宿主页面（renderCfg）一致。
       以前是 标签 → 控件 → 说明，并进"训练参数"卡之后开关会跑到文字左边，
       整行看着像往左弯了一下。 */
    row.appendChild(hint(r));
    row.appendChild(cc);
    return row;
  }

  function hint(r) {
    var cd = mk("span", "cd");
    cd.textContent = r.d || "";
    return cd;
  }

  /* 一组 = 一张 .card（.cardhd + N 行 .cfgrow） */
  function buildGroup(g) {
    var card = mk("div", "card");
    if (g.noheader) {          /* 标题挪进行里，卡片本身不要标题 */
      (g.rows || []).forEach(function (r) { card.appendChild(buildRow(r)); });
      return card;
    }
    var hd = mk("div", "cardhd");
    var t = mk("b");
    t.textContent = g.t || "";
    hd.appendChild(t);
    if (g.note) {
      var n = mk("span", "hint");
      n.textContent = g.note;
      hd.appendChild(n);
    }
    card.appendChild(hd);
    (g.rows || []).forEach(function (r) { card.appendChild(buildRow(r)); });
    return card;
  }

  function bind(root) {
    if (!root) return;
    /* seg：单选切换 */
    Array.prototype.forEach.call(root.querySelectorAll(".seg"), function (seg) {
      if (seg.getAttribute("data-bound")) return;
      seg.setAttribute("data-bound", "1");
      seg.addEventListener("click", function (ev) {
        var b = ev.target;
        if (!b || b.tagName !== "B") return;
        Array.prototype.forEach.call(seg.children, function (x) { x.classList.remove("on"); });
        b.classList.add("on");
      });
    });
    /* sw：开关切换 */
    Array.prototype.forEach.call(root.querySelectorAll(".sw"), function (sw) {
      if (sw.getAttribute("data-bound")) return;
      sw.setAttribute("data-bound", "1");
      var flip = function () { sw.classList.toggle("on"); };
      sw.addEventListener("click", flip);
      sw.addEventListener("keydown", function (ev) {
        if (ev.key === " " || ev.key === "Enter") { ev.preventDefault(); flip(); }
      });
    });
    /* sel：只补焦点样式，取值交给业务层读 */
    Array.prototype.forEach.call(root.querySelectorAll(".mgsel"), function (sel) {
      if (sel.getAttribute("data-bound")) return;
      sel.setAttribute("data-bound", "1");
    });
  }

  function renderGroups(box, groups) {
    if (!box) return;
    (groups || []).forEach(function (g) { box.appendChild(buildGroup(g)); });
  }

  /* ------------------------------------------------------------
   * 四、对外接口
   * ------------------------------------------------------------ */
  var api = {
    GROUPS: EXTRA_GROUPS,
    SELECT_GROUPS: EXTRA_SELECT_GROUPS,
    renderGroups: renderGroups,
    bind: bind
  };

  /* 补齐参数不再单独成卡：优先并进已有的同名分组（训练参数 / 补丁参数 / 数据增强参数…）。
     只有找不到归属的组（比如 PIA）才退回独立卡片。 */
  /* 少数补充组没有同名卡片，显式指个归属 */
  var MERGE_INTO = { "PIA": "优化器参数" };

  function findCardByTitle(box, base) {
    var cards = box.querySelectorAll(".card");
    var want = [base, base + "参数"];
    var fallback = null;
    for (var i = 0; i < cards.length; i++) {
      var b = cards[i].querySelector(".cardhd b");
      if (!b) continue;
      var t = (b.textContent || "").trim();
      if (want.indexOf(t) >= 0) return cards[i];
      if (!fallback && t.indexOf(base) === 0) fallback = cards[i];
    }
    return fallback;
  }

  api.mount = function () {
    var box = document.getElementById("cfgBox");
    if (!box) return false;

    /* 可重复调用：先清掉上一次并入的行和残留的包装层。
       以前只在开头判断"有没有 train-extra 包装层"，结果 openCfg 重画后
       包装层消失、行还留着 → 每重画一次就多一份，页面越来越乱。 */
    Array.prototype.forEach.call(box.querySelectorAll('[data-extra="1"]'), function (r) {
      if (r.parentNode) r.parentNode.removeChild(r);
    });
    Array.prototype.forEach.call(box.querySelectorAll('[data-part="train-extra"]'), function (w) {
      if (w.parentNode) w.parentNode.removeChild(w);
    });

    var groups = (EXTRA_GROUPS || []).concat(EXTRA_SELECT_GROUPS || []);
    var seen = {};
    var leftover = [];

    function mergeInto(card, g) {
      /* 宿主卡里已经有的键不再补：Host 的"设备"卡自带 device 行，
         补齐块里也有一个 device 行，不加这层判断就会出现两行一样的 */
      var have = {};
      Array.prototype.forEach.call(card.querySelectorAll(".cfgrow[data-k]"), function (r) {
        var k = r.getAttribute("data-k");
        if (k) have[k] = 1;
      });
      (g.rows || []).forEach(function (r) {
        if (r.k && have[r.k]) return;
        var row = buildRow(r);
        row.setAttribute("data-extra", "1");   /* 打标记，下次 mount 好清理 */
        /* after: "<data-k>" —— 插到那一行后面，而不是整卡末尾。
           例：预训练集要跟在 dst 人脸集下面，别排到人脸类型之后。 */
        var anchor = r.after ? card.querySelector('.cfgrow[data-k="' + r.after + '"]') : null;
        if (anchor && anchor.nextSibling) card.insertBefore(row, anchor.nextSibling);
        else if (anchor) card.appendChild(row);
        else card.appendChild(row);
      });
      bind(card);
    }

    /* 同名组先归并 rows 再并卡：EXTRA_GROUPS 和 EXTRA_SELECT_GROUPS 里各有一个
       "数据增强 · 补充"，以前第二组被整组跳过 —— preview_grade_alg 因此从没出现在页面上。
       归并后顺序不变（先出现的组在前），同一张卡仍然只并一次。 */
    var byBase = [], order = {};
    groups.forEach(function (g) {
      var base = String(g.t || "").split(" · ")[0].trim();
      base = MERGE_INTO[base] || base;
      if (order[base] === undefined) {
        order[base] = byBase.length;
        /* pin / noheader 要跟着走：丢了它们设备卡就不置顶、还会多出一个标题 */
        byBase.push({ base: base, g: { t: g.t, note: g.note, pin: g.pin, noheader: g.noheader, rows: [] } });
      }
      var slot = byBase[order[base]];
      (g.rows || []).forEach(function (r) { slot.g.rows.push(r); });
    });
    byBase.forEach(function (it) {
      var base = it.base;
      if (seen[base]) return;
      seen[base] = 1;
      var card = findCardByTitle(box, base);
      if (card) mergeInto(card, it.g);
      else leftover.push({ base: base, g: it.g });
    });

    if (!leftover.length) {
      if (window.I18N && window.I18N.refresh) window.I18N.refresh();
      return true;
    }

    /* 宿主还有卡是稍后才挂上来的（"设备"卡就是 openCfg 之后才出现的），
       等一拍再找一次；仍找不到才自己建卡，避免和宿主重复。 */
    setTimeout(function () {
      var still = [];
      leftover.forEach(function (it) {
        var card = findCardByTitle(box, it.base);
        if (card) mergeInto(card, it.g);
        else still.push(it.g);
      });
      if (!still.length) return;

      /* pin: "top" 的组（设备）插到「训练参数」前面 ——
         "用什么设备训练"是用户最需要一眼确认的，不该排在页尾。 */
      var pinned = still.filter(function (g) { return g.pin; });
      var normal = still.filter(function (g) { return !g.pin; });

      function makeWrap(list) {
        var w = mk("div", "train-extra");
        w.setAttribute("data-part", "train-extra");
        list.forEach(function (g) { w.appendChild(buildGroup(g)); });
        bind(w);
        try { if (window.__enhanceSelects) window.__enhanceSelects(w); } catch (e) {}
        return w;
      }

      if (pinned.length) {
        var wp = makeWrap(pinned);
        var anchor = findCardByTitle(box, "训练参数") || box.firstChild;
        box.insertBefore(wp, anchor);
      }
      if (normal.length) box.appendChild(makeWrap(normal));
      var archNow = (window.__cfgModel && window.__cfgModel.archi) || window.__cfgArchi || "";
      try { if (window.__filterFreezeRows) window.__filterFreezeRows(box, archNow); } catch (e) {}

      /* 关键：这批行是 openCfg 的 /config 回填**之后**才挂上来的（晚 80ms），
         第一次回填扫不到它们 —— 于是「设备」下拉永远停在写死的默认值，
         dat 里 device="auto" 页面上却显示 "0"。这里补一次，只填还没基线的行。 */
      try {
        if (window.__fillCfgRows && window.__cfgOpts) {
          var _n = window.__fillCfgRows(window.__cfgOpts, true);
          if (_n && window.syncCfgBaseline) window.syncCfgBaseline();
        }
      } catch (e) {}
      if (window.I18N && window.I18N.refresh) window.I18N.refresh();
    }, 80);

    if (window.I18N && window.I18N.refresh) window.I18N.refresh();
    return true;
  };

  /* 渲染完由 openCfg 直接调 TrainExtra.mount()（见 openCfg 末尾）。
     这里以前还包了一层 window.openCfg，但那份包装从来没被调用过
     —— 留着只会让人以为有两套挂载机制，已经删掉。 */

  window.TRAIN_EXTRA_GROUPS = EXTRA_GROUPS;
  window.TRAIN_EXTRA_SELECT_GROUPS = EXTRA_SELECT_GROUPS;
  window.TrainExtra = api;
})();

/* ------------------------------------------------------------
 * CITE —— 每个 k 的 4.0 出处（参数名一律照抄，不做任何编造）
 * ------------------------------------------------------------
 * masked_training          models/ModelBase.py:835           training_config_page.py:465
 * blur_out_mask            models/ModelBase.py:836           training_config_page.py:475
 * prioritize_mouth_eyes    models/ModelBase.py:837           training_config_page.py:455
 *                          （写进模型时映射为 option eyes_mouth_prio，ModelBase.py:837）
 * backup_interval          models/ModelBase.py:336-338,867   mainscripts/Trainer.py:505
 * device                   training_config_page.py:305-377,1508-1512
 * vgg_perceptual_power     models/ModelBase.py:848           training_config_page.py:868
 * mouth_scale              models/ModelBase.py:840           training_config_page.py:1055
 * random_src_flip          models/ModelBase.py:833           training_config_page.py:1087
 * random_dst_flip          models/ModelBase.py:834           training_config_page.py:1097
 * uniform_yaw_distribution models/ModelBase.py:841           training_config_page.py:1107
 *                          （模型 option 名 uniform_yaw，Model_SAEHD/Model.py:47）
 * preview_grade_alg        models/ModelBase.py:850           training_config_page.py:1069
 * freeze_inter             Model_SAEHD/Model_pytorch.py:140   training_config_page.py:1145
 * freeze_decoder_dst       Model_SAEHD/Model_pytorch.py:144   training_config_page.py:1205
 * freeze_inter_src         Model_AMP/Model_pytorch.py:56      training_config_page.py:1175
 * freeze_inter_dst         Model_AMP/Model_pytorch.py:57      training_config_page.py:1185
 * freeze_bottleneck        Model_DeepFakeLarge/Model_pytorch.py:404  training_config_page.py:1217
 * freeze_decoderA_mask     Model_DeepFakeLarge/Model_pytorch.py:397  training_config_page.py:1227
 * freeze_decoderB_mask     Model_DeepFakeLarge/Model_pytorch.py:398  training_config_page.py:1237
 * freeze_bottleneck_AB     Model_LIAELarge/Model_pytorch.py:376      training_config_page.py:1249
 * freeze_bottleneck_B      Model_LIAELarge/Model_pytorch.py:377      training_config_page.py:1259
 * pretrain_dataset_path    training_config_page.py:261-274,1500
 *
 * ------------------------------------------------------------
 * OPENCFG_SEL_PATCH —— 若要把 EXTRA_SELECT_GROUPS 并进 CFG_GROUPS，
 * 在 openCfg() 里 t === "seg" 那个分支后面补一段：
 *
 *     } else if (r.t === "sel") {
 *       var sel = document.createElement("select");
 *       sel.className = "mgsel";
 *       r.o.forEach(function (o) {
 *         var op = document.createElement("option");
 *         op.textContent = o;
 *         if (String(o) === String(r.v)) op.selected = true;
 *         sel.appendChild(op);
 *       });
 *       cc.appendChild(sel);
 *     }
 * ------------------------------------------------------------ */

/* ============================================================
 * UX 补强（4.1）
 *   1) 新建模型对话框：显卡探测 + 档位推荐 + 一步创建并开训
 *   2) 参数页搜索框（不折叠，只过滤）
 *   3) 训练页「继续训练」置顶卡
 * 独立 IIFE，只用 window 上已有的 go/toast/startTraining/pollTrain，
 * 不动上面那几个大 IIFE 的内部结构。
 * ============================================================ */
(function () {
  "use strict";
  function $(id) { return document.getElementById(id); }
  function nfmt(n) { try { return Number(n).toLocaleString("en-US"); } catch (e) { return String(n); } }
  /* 动态插入的块不会自动翻译（i18n 只在启动和手动 refresh 时扫一遍），
     所以每个渲染函数结尾都要叫一次。中文模式下是空操作。 */
  function i18nRefresh() {
    try { if (window.I18N && window.I18N.get() === "en") window.I18N.refresh(); } catch (e) {}
  }

  /* ---------------- 1. 新建模型：直接进"空模型的参数页" ----------------
   * 不做弹窗。点「新建模型」就跳到参数设置页，顶上多一块「模型结构」：
   * 名字 / 架构 / 子分支 / 算子 / 分辨率 / 四个维度。
   * 脸型不在这块卡里：「数据集参数」卡已经有一项（whole_face / full_face …），
   * 两处同时出现就是重复项。创建时 nmSubmit() 从那一项取，再转成 dat 认的短码。
   * 结构参数只在这里出现（训练参数页原本没有这些），底部按钮临时变成「创建模型」。
   * 下拉项和合法值全部来自后端 /api/model_spec（源头是 4.0 的 core/archi_names.py）。
   */
  var NEW = { on: false, spec: null, prevSub: "", fetched: false };

  /* 面板要"立刻出现"：不能等 /api/model_spec（那里面有 nvidia-smi，约 70-200ms）。
     先用这份静态兜底画出来，后端数据到了再替换数值。 */
  var SPEC_FALLBACK = {
    archi: [{ name: "DF" }, { name: "LIAE" }, { name: "AMP" },
            { name: "DF Lite" }, { name: "LIAE Lite" }],
    subarchi: [{ v: "-u" }, { v: "-ud" }, { v: "-ut" }, { v: "-udt" },
               { v: "-d" }, { v: "-dt" }, { v: "-t" }],
    kernel: [{ label: "dw5" }],
    default_kernel: "dw5",
    face_types: [{ v: "h", label: "半脸（只要脸，不要额头下巴）" },
                 { v: "mf", label: "中全脸" }, { v: "f", label: "全脸" },
                 { v: "wf", label: "整头（含额头和下巴，推荐）" },
                 { v: "head", label: "头肩（要脖子和肩膀）" }],
    dims: { ae_dims: 256, e_dims: 64, d_dims: 128, d_mask_dims: 42 },
    dims_liae: { ae_dims: 352, e_dims: 102, d_dims: 102, d_mask_dims: 32 },
    limits: { resolution: [64, 640], ae_dims: [64, 1024],
              e_dims: [16, 256], d_dims: [16, 256], d_mask_dims: [8, 128] },
    recommend: { resolution: 256, batch_size: 4, need_gb: "-", preset: "fast" },
    gpu: null
  };

  function nmPanel() { return $("nmStruct"); }

  function nmRow(label, inner) {
    return '<div style="margin-top:12px"><div class="hint" style="margin-bottom:6px">' + label +
      "</div>" + inner + "</div>";
  }

  function nmRenderPanel() {
    var box = $("cfgBox");
    if (!box || !NEW.spec) return;
    var host = nmPanel();
    if (!host) {
      host = document.createElement("div");
      host.id = "nmStruct";
      host.className = "card";
      /* ★ 必须插进 #cfgBox 里面。
         以前是 "sbox.parentNode" 再 insertBefore —— 而搜索框外面那层 wrapper 是
         position:sticky(top:-2px)，结果整张「新建模型 · 结构」卡跟着一起吸顶，
         像 Excel 冻结首行那样压在参数列表上，页面看起来像"滚不动"。
         放进 cfgBox 后它跟普通卡片一样随页面滚动。 */
      if (box.firstChild) box.insertBefore(host, box.firstChild);
      else box.appendChild(host);
    }
    var S = NEW.spec;
    function opt(list, valueKey, textKey, selected) {
      return list.map(function (it) {
        var v = it[valueKey], t = it[textKey];
        return '<option value="' + v + '"' + (String(v) === String(selected) ? " selected" : "") +
          ">" + t + "</option>";
      }).join("");
    }
    function row(label, desc, control) {
      return '<div class="cfgrow"><span class="cl">' + label + "</span>" +
        '<span class="cd">' + desc + "</span>" +
        '<span class="cc">' + control + "</span></div>";
    }
    function num(id, v, lo, hi, step) {
      return '<input class="inp" id="' + id + '" type="number" value="' + v +
        '" min="' + lo + '" max="' + hi + '" step="' + step + '">';
    }
    var d = S.dims;
    host.innerHTML =
      '<div class="cardhd"><b>新建模型 · 结构</b>' +
      '<span class="hint">定完结构，点下面的「创建模型」就会开始训练</span></div>' +
      row("名字", "模型目录名，建完就固定", '<input class="inp" id="nmName" placeholder="例如 我的第一个模型" autocomplete="off">') +
      row("架构", "DF：双解码器　LIAE：双 inter。Lite 版换掉 5×5 稠密卷积",
          '<select class="mgsel" id="nmArchi">' + opt(S.archi, "name", "name", "DF Lite") + "</select>") +
      row("子分支", "u = 像素归一化　d = 可学习上采样　t = 编码器多一次下采样",
          '<select class="mgsel" id="nmSub">' + opt(S.subarchi, "v", "v", "-udt") + "</select>") +
      '<div id="nmKernelWrap">' +
      row("算子", "Lite 架构的卷积实现；dw5 最快且保住 5×5 感受野",
          '<select class="mgsel" id="nmKernel">' + opt(S.kernel, "label", "label", S.default_kernel) + "</select>") +
      "</div>" +
      row("分辨率", "对齐后的人脸边长，越大越吃显存",
          num("nmRes", S.recommend.resolution, S.limits.resolution[0], S.limits.resolution[1], 32)) +
      row("编码器宽度", "ae_dims：瓶颈层宽度，决定“能装下多少长相”",
          num("nmAe", d.ae_dims, S.limits.ae_dims[0], S.limits.ae_dims[1], 8)) +
      row("编码器维度", "e_dims：encoder 内部通道",
          num("nmE", d.e_dims, S.limits.e_dims[0], S.limits.e_dims[1], 8)) +
      row("解码器维度", "d_dims：decoder 内部通道",
          num("nmD", d.d_dims, S.limits.d_dims[0], S.limits.d_dims[1], 8)) +
      row("遮罩维度", "mask：解码器里遮罩分支的通道数",
          num("nmMask", d.d_mask_dims, S.limits.d_mask_dims[0], S.limits.d_mask_dims[1], 8)) +
      '<div class="cfgrow"><span class="cl"></span><span class="cd"></span><span class="cc">' +
      '<span class="hint" id="nmNote"></span></span></div>' +
      '<div class="cfgrow"><span class="cl"></span><span class="cd"></span><span class="cc">' +
      '<span class="hint" id="nmErr" style="color:var(--bad)"></span></span></div>' +
      '<div class="cfgrow"><span class="cl"></span><span class="cd"></span><span class="cc">' +
      '<button class="mini" id="nmCancel">退出新建</button></span></div>';

    nmSyncArchi();
    i18nRefresh();
    $("nmArchi").addEventListener("change", nmSyncArchi);
    $("nmCancel").addEventListener("click", function () { nmEnter(false); });
    var nm = $("nmName");
    if (nm && !nm.value) {
      var dt = new Date();
      nm.value = "模型_" + (dt.getMonth() + 1) + dt.getDate() + "_" +
        (dt.getHours() < 10 ? "0" : "") + dt.getHours() + (dt.getMinutes() < 10 ? "0" : "") + dt.getMinutes();
    }
  }

  function nmSyncArchi() {
    var S = NEW.spec;
    if (!S) return;
    var name = $("nmArchi").value;
    var a = S.archi.filter(function (x) { return x.name === name; })[0] || {};
    var wrap = $("nmKernelWrap");
    if (wrap) wrap.style.display = a.lite ? "" : "none";
    // 维度默认值跟着架构走：df 一套、liae 一套（都是本机跑过的配置）
    var d = (a.base === "liae") ? S.dims_liae : S.dims;
    if ($("nmAe")) $("nmAe").value = d.ae_dims;
    if ($("nmE")) $("nmE").value = d.e_dims;
    if ($("nmD")) $("nmD").value = d.d_dims;
    if ($("nmMask")) $("nmMask").value = d.d_mask_dims;
    nmNote();
  }

  function nmNote() {
    var el = $("nmNote");
    if (!el || !NEW.spec) return;
    var r = NEW.spec.recommend || {};
    var g = NEW.spec.gpu;
    var txt = "按你这张卡的可用显存，建议 " + (r.resolution || 256) + " 分辨率 / batch " +
      (r.batch_size || 4) + "（训练显存约 " + (r.need_gb || "-") + " GB）";
    if (g) txt += "。显卡 " + g.name.replace("NVIDIA GeForce ", "") + " · " + g.total_gb + " GB";
    el.textContent = txt;
  }

  /* 现有模型：同一块结构信息，但只读 —— 结构建好就固定了，
     用户还是想看到"我这个模型是 384 的 liae-ud、ae 352"。 */
  function nmReadonlyPanel(m) {
    if (!m || NEW.on) return;
    var box = $("cfgBox");
    if (!box) return;
    var host = $("nmInfo");
    if (!host) {
      host = document.createElement("div");
      host.id = "nmInfo";
      host.className = "card";
      /* 同上：不能插进吸顶的搜索框 wrapper，否则整卡跟着吸顶 */
      if (box.firstChild) box.insertBefore(host, box.firstChild);
      else box.appendChild(host);
    }
    function ro(label, value) {
      var v = (value === undefined || value === null || value === "") ? "—" : value;
      return '<div class="cfgrow"><span class="cl">' + label + '</span><span class="cv">' + v + "</span></div>";
    }
    var ap = m.archi_parsed || {};
    var kernel = String(ap.kernel || "").split(" - ")[0];
    host.innerHTML =
      '<div class="cardhd"><b>模型结构</b>' +
      '<span class="hint">创建后不可修改，要换结构请新建模型</span></div>' +
      ro("名字", m.name) +
      ro("架构", (ap.display || m.archi) + (ap.sub ? " " + ap.sub : "") +
         (ap.lite && kernel ? " · " + kernel : "")) +
      ro("分辨率", m.resolution) +
      ro("编码器宽度", m.ae_dims) +
      ro("编码器维度", m.e_dims) +
      ro("解码器维度", m.d_dims) +
      ro("遮罩维度", m.mask_dims) +
      ro("已训练", nfmt(m.iter || 0) + " iter · " + (m.optimizer || "—") +
         (m.weights ? " · 权重 " + m.weights : ""));
    i18nRefresh();
  }

  function nmEnter(on) {
    NEW.on = !!on;
    var startBtn = $("cfgStart");
    var subEl = $("cfgSub");
    if (!NEW.on) {
      var host = nmPanel();
      if (host) host.remove();
      nmReadonlyPanel(window.__cfgModel || (window.MODELS || [])[0] || null);
      if (startBtn) startBtn.textContent = "开始训练";
      if (subEl) subEl.textContent = "训练前确认参数。带 · 的是本次会话才会生效的项。";
      return;
    }
    if (subEl) subEl.textContent = "先定模型结构（分辨率 / 维度 / 架构），创建后这些就固定了；其余训练参数随时能改。";
    var info = $("nmInfo");
    if (info) info.remove();
    if (startBtn) startBtn.textContent = "创建模型";
    if (!NEW.spec) NEW.spec = SPEC_FALLBACK;
    nmRenderPanel();                    /* 立刻出现，不等网络 */
    if (NEW.fetched) return;
    NEW.fetched = true;
    fetch("/api/model_spec").then(function (r) { return r.json(); }).then(function (d) {
      NEW.spec = d;
      if (!NEW.on) return;
      /* 重画会重建 DOM，把用户已经填/选的东西先存下来，画完再放回去 */
      var FIELDS = ["nmName", "nmArchi", "nmSub", "nmKernel", "nmRes",
                    "nmAe", "nmE", "nmD", "nmMask"];
      var keep = {};
      FIELDS.forEach(function (k) { if ($(k)) keep[k] = $(k).value; });
      /* 用兜底值时那几个数字就是默认值，后端到了要换成推荐值；
         用户若手动改过（和兜底不同），就保留用户的选择 */
      var untouched = {
        nmRes: !keep.nmRes || keep.nmRes === String(SPEC_FALLBACK.recommend.resolution),
        nmAe: !keep.nmAe || keep.nmAe === String(SPEC_FALLBACK.dims.ae_dims),
        nmE: !keep.nmE || keep.nmE === String(SPEC_FALLBACK.dims.e_dims),
        nmD: !keep.nmD || keep.nmD === String(SPEC_FALLBACK.dims.d_dims),
        nmMask: !keep.nmMask || keep.nmMask === String(SPEC_FALLBACK.dims.d_mask_dims)
      };
      nmRenderPanel();
      ["nmName", "nmArchi", "nmSub", "nmKernel"].forEach(function (k) {
        if ($(k) && keep[k] !== undefined) $(k).value = keep[k];
      });
      var custom = !(untouched.nmRes && untouched.nmAe && untouched.nmE &&
                     untouched.nmD && untouched.nmMask);
      if (custom) {
        ["nmRes", "nmAe", "nmE", "nmD", "nmMask"].forEach(function (k) {
          if ($(k) && keep[k] !== undefined && !untouched[k]) $(k).value = keep[k];
        });
      } else {
        nmSyncArchi();      /* 一个都没动过 → 按当前架构的默认维度重设 */
      }
      nmNote();
    }).catch(function () {
      if ($("nmErr")) $("nmErr").textContent = "读模型规格失败（后端没起？用的是默认值）";
    });
  }

  function openNewModel() {
    nmBindCreate();
    /* ★ 先确保参数行已经渲染出来。
       以前是直接 go("train-cfg")，而 go() 只在「#cfgBox 里一张卡都没有」时才调 openCfg；
       新建面板自己往 #cfgBox 插了一张「模型结构」卡，于是 go() 认为"已经渲染过"，
       参数行（批次大小 / 预训练 / 预训练集路径 / 优化器 …）**永远不出现** ——
       用户在新建流程里根本没法设这些，建出来的模型只能吃推荐值。 */
    var box = document.getElementById("cfgBox");
    var hasRows = box && box.querySelector(".card:not(#nmStruct):not(#nmInfo)");
    if (!hasRows) {
      var m0 = window.__cfgModel || (window.MODELS || [])[0] || null;
      if (m0 && window.openCfg) { try { window.openCfg(m0); } catch (e) { if (window.go) window.go("train-cfg"); } }
      else if (window.go) window.go("train-cfg");
    } else if (window.go) {
      window.go("train-cfg");
    }
    setTimeout(function () { nmEnter(true); nmBindCreate(); }, 80);
  }

  /* 脸型只有一个入口：「数据集参数」卡里的 face_type 下拉（whole_face / full_face …）。
     创建模型时要把它翻成 dat 认的短码 —— Model_SAEHD 是拿 'wf'/'f'/'h'/'mf'/'head'
     直接查 FaceType 表的，发全名过去，训练一启动就 KeyError。 */
  var FACE_SHORT = { whole_face: "wf", full_face: "f", half_face: "h",
                     midfull_face: "mf", head: "head" };

  function nmFaceType() {
    var sel = document.querySelector("#cfgBox [data-cfg=face_type]");
    var v = sel ? String(sel.value || "").trim() : "";
    return FACE_SHORT[v] || v || "wf";      /* 空着或认不出就给 4.1 的老默认 wf */
  }

  function nmSubmit() {
    var name = ($("nmName").value || "").trim();
    if (!name) { $("nmErr").textContent = "起个名字"; $("nmName").focus(); return; }
    /* 一次请求带全：spec（结构，创建后不可改）+ options（训练参数，页面上填的）。
       后端用同一套 _clean_options 解析，返回"实际生效的全部 options"。
       前端不再需要"创建成功后再发一次 config"这种补丁式做法。 */
    var body = {
      name: name,
      spec: {
        archi_display: $("nmArchi").value,
        subarchi: $("nmSub").value,
        kernel_label: $("nmKernel") ? $("nmKernel").value : "",
        face_type: nmFaceType(),
        resolution: parseInt($("nmRes").value, 10) || undefined,
        ae_dims: parseInt($("nmAe").value, 10) || undefined,
        e_dims: parseInt($("nmE").value, 10) || undefined,
        d_dims: parseInt($("nmD").value, 10) || undefined,
        d_mask_dims: parseInt($("nmMask").value, 10) || undefined
      },
      options: (window.collectCfgPatch ? window.collectCfgPatch(true) : {}),
      /* 批次大小：页面上的值已经在 options 里了；这里只在它缺失时用显卡推荐值兜底 */
      batch_size: (function () {
        var el = document.querySelector("#cfgBox [data-cfg=batch_size]");
        var v = el ? parseInt(el.value, 10) : 0;
        if (v > 0) return v;
        return (NEW.spec && NEW.spec.recommend && NEW.spec.recommend.batch_size) || 4;
      })()
    };
    var btn = $("cfgStart");
    btn.disabled = true;
    btn.textContent = "创建中…";
    fetch("/api/models/create", {
      method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body)
    }).then(function (r) {
      return r.json().then(function (d) { if (!r.ok) throw new Error(d.detail || "创建失败"); return d; });
    }).then(function (d) {
      btn.textContent = "启动训练…";
      return fetch("/api/train/start", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ name: d.name })
      }).then(function (r) {
        return r.json().then(function (x) { if (!r.ok) throw new Error(x.detail || "训练没起来"); return d; });
      });
    }).then(function (d) {
      btn.disabled = false;
      btn.textContent = "开始训练";
      if (window.toast) window.toast("已创建并开始训练", d.name + " · " + d.options.archi + " · " + d.options.resolution + " 分辨率", "success");
      /* 后端会顺手把超范围的参数夹进合法区间（分辨率上限 640 等）。
         以前是静默的 —— 填 1024 建出来是 640，人还以为在训 1024。现在先说清楚。 */
      if (d.clamped && Object.keys(d.clamped).length) {
        var _nm = { resolution: "分辨率", ae_dims: "编码器宽度", e_dims: "编码器维度",
                    d_dims: "解码器维度", d_mask_dims: "遮罩维度" };
        var _txt = Object.keys(d.clamped).map(function (k) {
          var c = d.clamped[k];
          return (_nm[k] || k) + " " + c.requested + "→" + c.used;
        }).join("，");
        if (window.toast) window.toast("部分参数超出范围，已自动调整", _txt, "info");
      }
      /* ★ 关键：把「当前模型」切到刚建的这个再进运行页。
         以前这里是 go("run") —— 运行页不认模型，顶部标题还停在旧值（甚至
         是 HTML 里的占位 "384wf"），曲线/统计也按 "列表里第一个" 取，
         于是用户看到的是"点新建，却在训练第一个模型"。
         现在重拉一次模型列表，按名字找回这一条，openRun(它)。 */
      fetch("/api/models").then(function (r) { return r.json(); }).then(function (mm) {
        var list = (mm && mm.models) || [];
        var fresh = null;
        for (var i = 0; i < list.length; i++) {
          if (list[i] && list[i].name === d.name) { fresh = list[i]; break; }
        }
        if (list.length) window.MODELS = list;
        /* 立刻重画「模型选择」列表：以前新建完要切页/刷新才看得到新模型 */
        try { if (window.renderTrainer && window.MODELS) window.renderTrainer(window.MODELS); } catch (e) {}
        if (fresh) { window.__cfgModel = fresh; window.__runModel = fresh; }
        nmEnter(false);
        if (fresh && window.openRun) window.openRun(fresh);
        else if (window.go) window.go("run");
        if (window.pollTrain) window.pollTrain(fresh || undefined);
      }).catch(function () {
        nmEnter(false);
        if (window.go) window.go("run");
        if (window.pollTrain) window.pollTrain();
      });
    }).catch(function (e) {
      btn.disabled = false;
      btn.textContent = "创建模型";
      if ($("nmErr")) $("nmErr").textContent = String((e && e.message) || e);
    });
  }

  // 接管底部按钮：新建模式下先拦住原「开始训练」，改走创建。
  // 幂等 —— 页面结构可能还没就绪（脚本在 head 里），所以 openNewModel 和
  // DOMContentLoaded 都会再调一次。
  function nmBindCreate() {
    var b = $("cfgStart");
    if (!b || b.__nmBound) return;
    b.__nmBound = true;
    b.addEventListener("click", function (e) {
      if (!NEW.on) return;
      e.preventDefault();
      e.stopImmediatePropagation();
      nmSubmit();
    }, true);
  }
  nmBindCreate();

  window.openNewModel = openNewModel;

  /* 挂钩 openCfg：每次渲染完训练参数，把「模型结构（只读）」补在最上面 */
  var _openCfg = window.openCfg;
  if (typeof _openCfg === "function") {
    window.openCfg = function (m) {
      var r = _openCfg.apply(this, arguments);
      var mm = m || window.__cfgModel || (window.MODELS || [])[0] || null;
      setTimeout(function () { nmReadonlyPanel(mm); }, 60);
      return r;
    };
  }


  /* ---------------- 1.4 主题化下拉框（替掉原生 select 的弹层） ----------------
     原生 <select> 的弹出层是操作系统画的：蓝色高亮、没有过渡，跟界面两个世界。
     这里把它藏起来、用按钮 + 列表替身；value / change 仍然走原生元素，
     所以回填、提交参数的代码一行都不用改。 */
  function enhanceSelect(sel) {
    if (!sel || sel.dataset.dd) return;
    sel.dataset.dd = "1";
    var doll = function (tag, cls) { var el = document.createElement(tag); el.className = cls; return el; };
    var wrap = doll("div", "dd");
    sel.parentNode.insertBefore(wrap, sel);
    wrap.appendChild(sel);            /* 原生 select 留在 DOM 里，只隐藏 */
    sel.classList.add("ddnative");

    var btn = doll("button", "ddbtn");
    btn.type = "button";
    var menu = doll("div", "ddmenu");

    var txt = doll("span", "ddtxt");     /* 文字单独放一层，才能省略号截断 */
    function cur() {
      var o = sel.options[sel.selectedIndex];
      return o ? o.textContent : "";
    }
    function off(i) { return (sel.__off && sel.__off[i]) || null; }
    function sync() {
      txt.textContent = cur();
      /* 选中项本身是「未安装」时，按钮的 tooltip 要能说明原因，不能只回一遍标签文字 */
      var o = sel.options[sel.selectedIndex], mm = o ? off(sel.selectedIndex) : null;
      btn.title = mm ? (mm.title || o.textContent) : cur();
      btn.classList.toggle("off", !!mm);
    }
    function build() {
      menu.innerHTML = "";
      Array.prototype.forEach.call(sel.options, function (o, i) {
        var mm = off(i);
        var it = doll("div", "ddopt" + (i === sel.selectedIndex ? " on" : "") + (mm ? " off" : ""));
        it.textContent = o.textContent;
        it.title = mm ? (mm.title || o.textContent) : o.textContent;
        if (mm) it.setAttribute("aria-disabled", "true");
        it.addEventListener("click", function (ev) {
          ev.stopPropagation();
          /* 未安装：不切换，只解释（清掉原生 select 被选中的可能） */
          if (mm) { if (mm.onBlock) mm.onBlock(); sync(); close(); return; }
          sel.selectedIndex = i;
          sel.dispatchEvent(new Event("change", { bubbles: true }));
          sync();
          close();
        });
        menu.appendChild(it);
      });
    }
    function open() {
      build();
      wrap.classList.add("open");
      var on = menu.querySelector(".ddopt.on");
      if (on) on.scrollIntoView({ block: "nearest" });
    }
    function close() { wrap.classList.remove("open"); }

    btn.addEventListener("click", function (e) {
      e.stopPropagation();
      if (wrap.classList.contains("open")) close(); else open();
    });
    document.addEventListener("click", function () { close(); });
    document.addEventListener("keydown", function (e) { if (e.key === "Escape") close(); });
    sel.addEventListener("change", sync);

    btn.appendChild(txt);
    wrap.appendChild(btn);
    wrap.appendChild(menu);
    sync();
  }

  function enhanceSelects(root) {
    if (!root) return;
    Array.prototype.forEach.call(root.querySelectorAll("select.mgsel"), enhanceSelect);
  }
  /* openCfg 在另一个 IIFE 里（app.js 分了好几段作用域），拿不到这个函数，
     挂到 window 上给它用 —— 以前直接调 enhanceSelects，ReferenceError 被
     try/catch 吞掉，下拉框就一直是原生的。 */
  window.__enhanceSelects = enhanceSelects;

  /* ---------------- 1.45 设备下拉：显示真名而不是序号 ---------------- */
  function buildDeviceOpts(d) {
    var out = [{ v: "auto", l: "自动选择" }];
    if (d && d.cpu) out.push({ v: "cpu", l: String(d.cpu) });   /* 后端已经带了 "CPU (32 线程)"，别再叠一个前缀 */
    ((d && d.gpus) || []).forEach(function (g) {
      var mem = g.mem_mb ? " · " + Math.round(g.mem_mb / 1024) + " GB" : "";
      out.push({ v: String(g.idx), l: "GPU " + g.idx + " · " + g.name + mem });
    });
    if (d && d.gpus && d.gpus.length > 1) {
      out.push({
        v: d.gpus.map(function (g) { return g.idx; }).join(","),
        l: "全部 GPU（" + d.gpus.length + " 张）"
      });
    }
    return out;
  }

  /* 「当前可用 GPU 列表」直接写在下拉框旁边。
     以前这些信息只藏在选项文字里：存的值一旦不在列表里（换机器、拔了显卡、
     从别的工程拷了 dat），下拉框就静默跳到「自动选择」，用户完全看不出
     自己的选择已经被换掉了 —— 和「选了卡却不生效」是同一类问题。 */
  function devAvailableText(d) {
    var parts = [];
    ((d && d.gpus) || []).forEach(function (g) {
      var mem = g.mem_mb ? " · " + Math.round(g.mem_mb / 1024) + " GB" : "";
      parts.push("GPU " + g.idx + " " + (g.name || "") + mem);
    });
    if (d && d.cpu) parts.push(String(d.cpu));
    if (!parts.length) return "本机没有可用 GPU";
    return "可用：" + parts.join(" ｜ ");
  }

  /* want 给了就用它（回填 dat 时浏览器已经把无效值丢掉，sel.value 读不回来） */
  function devGuard(sel, want) {
    if (!sel) return;
    var d = window.__DEVINFO || {};
    var stale = sel.querySelector('option[data-invalid="1"]');
    if (stale) stale.parentNode.removeChild(stale);
    var cur = (want === undefined || want === null) ? String(sel.value || "") : String(want);
    /* 只对 GPU 序号（"0" / "0,1"）做拦截：auto / cpu 永远合法 */
    var numeric = /^[0-9]+(,[0-9]+)*$/.test(cur);
    var known = false;
    Array.prototype.forEach.call(sel.options, function (o) { if (o.value === cur) known = true; });
    if (numeric && !known) {
      /* 禁用它比默默忽略好：这条 disabled 选项让「本机没有这张卡」变成看得见的事实 */
      var op = document.createElement("option");
      op.value = cur;
      op.textContent = "⚠ GPU " + cur + "（本机不存在）";
      op.disabled = true;
      op.setAttribute("data-invalid", "1");
      sel.insertBefore(op, sel.firstChild);
      sel.value = cur;
    }
    var row = sel.closest ? sel.closest(".cfgrow") : null;
    if (!row) return;
    var h = row.querySelector(".devhint");
    if (!h) {
      h = document.createElement("span");
      h.className = "devhint";
      (sel.parentNode || row).appendChild(h);
    }
    var txt = devAvailableText(d);
    h.textContent = txt;
    h.title = txt;
  }
  window.__devGuard = devGuard;

  var DEVOPTS = null;
  function applyDevices(d) {
    window.__DEVINFO = d || null;
    DEVOPTS = buildDeviceOpts(d);
    window.__DEVOPTS = DEVOPTS;
    /* 已经有下拉框的话就地换选项，不用整页重画 */
    Array.prototype.forEach.call(document.querySelectorAll('select[data-cfg="device"]'), function (sel) {
      var keep = sel.value || "0";
      sel.innerHTML = "";
      DEVOPTS.forEach(function (o) {
        var op = document.createElement("option");
        op.value = String(o.v);
        op.textContent = o.l;
        sel.appendChild(op);
      });
      sel.value = keep;
      if (sel.selectedIndex < 0) sel.selectedIndex = 0;
      /* 旧值不在新列表里：留一条 disabled 的说明，别让下拉框自己换成「自动选择」 */
      devGuard(sel, keep);
      sel.dispatchEvent(new Event("change", { bubbles: true }));
    });
  }
  fetch("/api/devices").then(function (r) { return r.json(); })
    .then(applyDevices).catch(function () { });

  /* ---------------- 1.5 界面构建时间 ---------------- */
  /* 排查"改了看不到"时最需要的东西：把服务端文件的修改时间显示出来，
     一眼就能判断窗口是不是旧版本（旧 = 时间比最近一次改动早）。 */
  (function showBuild() {
    var el = function () { return document.getElementById("buildTag"); };
    function paint(iso) {
      var t = el();
      /* 挪到关于页的表格里了：只要时间本身，不用再带「界面构建」四个字 */
      if (t) t.textContent = (iso || "—") + " · 客户端 " + new Date().toLocaleTimeString();
    }
    fetch("/api/build").then(function (r) { return r.json(); }).then(function (d) {
      window.__buildIso = d.iso || "";
      paint(d.iso);
    }).catch(function () { paint(""); });
    document.addEventListener("DOMContentLoaded", function () { paint(window.__buildIso || ""); });
  })();

  /* ---------------- 2. 参数页搜索 ---------------- */
  function ensureCfgSearch() {
    var box = $("cfgBox");
    if (!box || $("cfgSearch")) return;
    var wrap = document.createElement("div");
    /* 吸顶：参数页要滚很久，搜索框必须一直够得着。
       背景取 --panel（和 .main 同色），否则滚动时下面的内容会从缝里透出来。 */
    wrap.style.cssText = "position:sticky;top:-2px;z-index:6;background:var(--panel);" +
      "padding:8px 0;margin:0";
    wrap.innerHTML =
      '<div class="trsearch">' +
      '  <input id="cfgSearch" placeholder="搜索参数（例如 遮罩 / batch / 学习率 / 翻转）" spellcheck="false" autocomplete="off">' +
      '  <svg class="tri" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.5">' +
      '    <circle cx="7" cy="7" r="4.5"/><path d="M10.5 10.5 14 14" stroke-linecap="round"/>' +
      "  </svg>" +
      "</div>";
    box.parentNode.insertBefore(wrap, box);
    $("cfgSearch").addEventListener("input", filterCfg);
    i18nRefresh();
  }

  function filterCfg() {
    var box = $("cfgBox");
    var inp = $("cfgSearch");
    if (!box || !inp) return;
    var q = String(inp.value || "").trim().toLowerCase();
    var cards = box.querySelectorAll(".card");
    for (var i = 0; i < cards.length; i++) {
      /* 结构卡（新建 / 只读）不参与搜索过滤：它们是页面的上下文，
         过滤掉会让人以为"新建模型的面板不见了"。 */
      if (cards[i].id === "nmStruct" || cards[i].id === "nmInfo") continue;
      var rows = cards[i].querySelectorAll(".cfgrow");
      var shown = 0;
      for (var j = 0; j < rows.length; j++) {
        var hit = !q || (rows[j].textContent || "").toLowerCase().indexOf(q) >= 0;
        rows[j].style.display = hit ? "" : "none";
        if (hit) shown++;
      }
      cards[i].style.display = (!q || shown > 0) ? "" : "none";
    }
  }

  /* ---------------- 3. 训练页「继续训练」置顶 ---------------- */
  function ensureResumeCard() {
    var grid = $("mgrid");
    if (!grid || $("resumeCard")) return;
    var d = document.createElement("div");
    d.id = "resumeCard";
    d.style.cssText = "margin:10px 0 14px";
    grid.parentNode.insertBefore(d, grid);
  }

  function refreshResume() {
    ensureResumeCard();
    var host = $("resumeCard");
    if (!host) return;
    fetch("/api/models").then(function (r) { return r.json(); }).then(function (d) {
      var m = (d.models || [])[0];
      if (!m) { host.innerHTML = ""; return; }
      var when = m.trained_at ? Math.max(0, Math.round((Date.now() / 1000 - m.trained_at) / 60)) : 0;
      var ago = when > 1440 ? Math.floor(when / 1440) + " 天前" : (when > 60 ? Math.floor(when / 60) + " 小时前" : when + " 分钟前");
      host.innerHTML =
        '<div class="card" style="display:flex;align-items:center;gap:12px;padding:12px 14px">' +
        '  <div style="flex:1;min-width:0">' +
        '    <b style="font-size:14px">继续训练 · ' + m.name + '</b>' +
        '    <div class="hint" style="margin-top:4px">iter ' + nfmt(m.iter) +
        ' · <span>上次训练</span> <span>' + ago + '</span>' +
        (m.has_preview ? "" : " · 还没有预览图（跑一会儿就有了）") + '</div>' +
        '  </div>' +
        '  <button class="btn pri" id="resumeGo">继续训练</button>' +
        '</div>';
      i18nRefresh();
      var btn = $("resumeGo");
      if (btn) btn.addEventListener("click", function () {
        btn.disabled = true;
        btn.textContent = "启动中…";
        window.startTraining(m.name).then(function () {
          if (window.go) window.go("run");
          if (window.pollTrain) window.pollTrain();
        }).catch(function (e) {
          btn.disabled = false;
          btn.textContent = "继续训练";
          if (window.toast) window.toast("启动失败", String((e && e.message) || e).slice(0, 120), "error");
        });
      });
    }).catch(function () {});
  }

  /* ---------------- 挂钩页面切换 ---------------- */
  var _go = window.go;
  if (typeof _go === "function") {
    window.go = function (key) {
      var r = _go.apply(this, arguments);
      setTimeout(function () {
        if (key === "train-cfg") { ensureCfgSearch(); filterCfg(); }
        if (key === "train") { refreshResume(); }
        /* 回到训练界面：轮询可能已经停了（上一个页面 idle 时被清掉/超时），
           这里重新拉一次，顺便把预览也刷一下。 */
        if (key === "run" && window.__runModel && window.pollTrain) {
          window.pollTrain(window.__runModel);
        }
      }, 60);
      return r;
    };
  }
  document.addEventListener("DOMContentLoaded", function () {
    setTimeout(function () { ensureCfgSearch(); refreshResume(); nmBindCreate(); }, 400);
  });
})();

/* ============================================================================
 * 步进器（.num）：宽度跟着内容走 + 让 −/+ 真的能点
 *   原来 input 固定 56px，能塞下 5 位数字 —— 而实际取值范围是 1~120（24fps 下
 *   帧跳跃超过 120 就撞到 0.2 fps 的下限，再大没有意义），空着一大截看着像毛坯。
 *   −/+ 两个按钮之前根本没有事件处理，点了没反应。
 * ============================================================================ */
(function () {
  "use strict";
  /* 用 canvas 量文本会踩两个坑：字体简写拼错时 canvas 静默退回默认 10px sans-serif
     （量出来偏窄，文字就被截断，就是这个 bug）；网页字体还没加载完时也一样偏窄。
     改成用一个隐藏的镜像 span 让浏览器自己排版，宽度直接取它的 offsetWidth，
     最后再用 input.scrollWidth 兜一次底 —— 三保险。 */
  var mirror = null;

  function mirrorWidth(str, cs) {
    if (!mirror) {
      mirror = document.createElement("span");
      mirror.setAttribute("aria-hidden", "true");
      mirror.style.cssText = "position:absolute;left:-99999px;top:0;visibility:hidden;" +
        "white-space:pre;pointer-events:none";
      document.body.appendChild(mirror);
    }
    /* cs.font 是浏览器算好的完整简写，比手工拼安全 */
    mirror.style.font = cs.font || (cs.fontSize + " " + cs.fontFamily);
    mirror.style.fontSize = cs.fontSize;
    mirror.style.fontFamily = cs.fontFamily;
    mirror.style.fontWeight = cs.fontWeight;
    mirror.style.fontStyle = cs.fontStyle;
    mirror.style.letterSpacing = cs.letterSpacing;
    mirror.textContent = str || "0";
    return mirror.getBoundingClientRect().width;
  }

  function bounds(inp) {
    var min = inp.min !== "" ? parseInt(inp.min, 10) : 1;
    var max = inp.max !== "" ? parseInt(inp.max, 10) : 999;
    if (isNaN(min)) min = 1;
    if (isNaN(max)) max = 999;
    return [min, max];
  }

  function sizeNum(inp) {
    if (!inp || inp.tagName !== "INPUT") return;
    var cs = getComputedStyle(inp);
    var pad = (parseFloat(cs.paddingLeft) || 0) + (parseFloat(cs.paddingRight) || 0);
    var bd = (parseFloat(cs.borderLeftWidth) || 0) + (parseFloat(cs.borderRightWidth) || 0);

    var text = String(inp.value || "0");
    var need = mirrorWidth(text, cs) + pad + bd;
    /* +2 给光标和内建的滚动余量 */
    var w = Math.ceil(Math.min(200, Math.max(38, need + 2)));
    inp.style.width = w + "px";

    /* 兜底：如果浏览器排版出来仍然放不下（比如字体刚加载完），按差值再补一次 */
    if (inp.scrollWidth > inp.clientWidth + 1) {
      var grow = inp.scrollWidth - inp.clientWidth;
      inp.style.width = Math.ceil(Math.min(200, w + grow + 1)) + "px";
    }
  }

  function sizeAll(root) {
    var list = (root || document).querySelectorAll(".num input");
    for (var i = 0; i < list.length; i++) sizeNum(list[i]);
  }

  function clampAndSize(inp) {
    var b = bounds(inp);
    var v = parseInt(inp.value, 10);
    if (isNaN(v)) v = b[0];
    v = Math.max(b[0], Math.min(b[1], v));
    inp.value = String(v);
    sizeNum(inp);
  }

  document.addEventListener("input", function (e) {
    if (e.target && e.target.matches && e.target.matches(".num input")) sizeNum(e.target);
  }, true);

  document.addEventListener("change", function (e) {
    if (e.target && e.target.matches && e.target.matches(".num input")) clampAndSize(e.target);
  }, true);

  document.addEventListener("click", function (e) {
    var btn = e.target && e.target.closest ? e.target.closest(".num .nb") : null;
    if (!btn || btn.disabled) return;
    var inp = btn.parentNode ? btn.parentNode.querySelector("input") : null;
    if (!inp) return;
    var step = parseInt(btn.dataset.d, 10) || 1;
    var b = bounds(inp);
    var cur = parseInt(inp.value, 10);
    if (isNaN(cur)) cur = b[0];
    inp.value = String(Math.max(b[0], Math.min(b[1], cur + step)));
    sizeNum(inp);
    inp.dispatchEvent(new Event("change", { bubbles: true }));
  }, true);

  sizeAll();
  document.addEventListener("DOMContentLoaded", function () { sizeAll(); });
  window.addEventListener("load", function () { sizeAll(); });
})();

/* ============================================================================
 * 数据集体检（One Shot）
 *   一次扫描把清晰度 / 曝光 / 纯色 / 重复 / 坏图全算出来，一次给出"该删哪些"。
 *   4.0 那边是 12 种排序各跑一遍全量（每遍都要重新解码），这里只解码一次。
 * ============================================================================ */
(function () {
  "use strict";
  var M = { el: null, timer: 0, result: null, dir: "data_src" };

  function $(id) { return document.getElementById(id); }
  function nf(n) { try { return Number(n).toLocaleString("en-US"); } catch (e) { return String(n); } }
  function i18n() { try { if (window.I18N && window.I18N.get() === "en") window.I18N.refresh(); } catch (e) {} }

  function ensure() {
    if (M.el) return M.el;
    var d = document.createElement("div");
    d.id = "osMask";
    d.style.cssText = "position:fixed;left:0;top:0;right:0;bottom:0;background:rgba(6,8,12,.66);" +
      "display:none;align-items:center;justify-content:center;z-index:9999";
    d.innerHTML =
      '<div class="card" style="width:520px;max-width:94vw;padding:18px">' +
      '  <div style="display:flex;align-items:center;gap:10px">' +
      '    <b style="font-size:15px">一键体检</b>' +
      '    <span class="hint" id="osSub" style="margin-left:auto">—</span>' +
      '  </div>' +
      '  <div style="display:flex;align-items:center;gap:8px;margin-top:14px">' +
      '    <span class="hint">强度</span>' +
      '    <div class="seg" id="osLevel" style="flex:1">' +
      '      <b data-lv="strict" class="on">严格</b><b data-lv="normal">标准</b><b data-lv="loose">宽松</b>' +
      '    </div>' +
      '  </div>' +
      '  <div style="display:flex;align-items:center;gap:8px;margin-top:8px">' +
      '    <span class="hint" title="规则之外，再按综合分砍掉尾部">模型轨</span>' +
      '    <div class="seg" id="osScore" style="flex:1">' +
      '      <b data-sp="0">关</b><b data-sp="5" class="on">额外 5%</b><b data-sp="10">额外 10%</b>' +
      '    </div>' +
      '  </div>' +
      '  <div class="hint" id="osProg" style="margin-top:12px">准备中…</div>' +
      '  <div style="height:5px;border-radius:999px;background:var(--line);overflow:hidden;margin-top:8px">' +
      '    <i id="osBar" style="display:block;height:100%;width:0;background:var(--accent);border-radius:999px;transition:width .15s ease"></i>' +
      '  </div>' +
      '  <div id="osBody" style="margin-top:16px"></div>' +
      '  <div style="display:flex;gap:8px;justify-content:flex-end;margin-top:18px">' +
      '    <button class="mini" id="osCancel">关闭</button>' +
      '    <button class="btn pri" id="osApply" style="display:none">清理</button>' +
      '  </div>' +
      '</div>';
    document.body.appendChild(d);
    M.el = d;
    d.addEventListener("click", function (e) { if (e.target === d) close(); });
    $("osCancel").addEventListener("click", close);
    return d;
  }

  function close() {
    if (M.timer) { clearInterval(M.timer); M.timer = 0; }
    if (M.el) M.el.style.display = "none";
  }

  function ro(label, value) {
    return '<div style="display:flex;justify-content:space-between;padding:5px 0;font-size:12.5px">' +
      '<span style="color:var(--ink-3)">' + label + "</span><b>" + value + "</b></div>";
  }

  function renderResult(p, elapsed) {
    var c = p.counts || {};
    var pct = p.total ? Math.round(1000 * p.trash_count / p.total) / 10 : 0;
    var avg = p.avg || {};
    var worst = (p.worst || []).slice(0, 5).map(function (w) {
      return '<div style="display:flex;justify-content:space-between;font-size:12px;padding:3px 0">' +
        '<span style="color:var(--ink-3);overflow:hidden;text-overflow:ellipsis;white-space:nowrap;max-width:52%">' +
        w.name + '</span><b>' + w.q + ' <span style="color:var(--ink-3);font-weight:400">(锐 ' +
        w.blur + ' · 净 ' + w.noise + ' · 光 ' + w.exp + ')</span></b></div>';
    }).join("");

    $("osBody").innerHTML =
      '<div style="display:flex;gap:10px;margin-bottom:12px">' +
      '  <div style="flex:1;padding:10px 12px;border-radius:10px;background:var(--panel-2)">' +
      '    <div class="hint">平均质量分</div>' +
      '    <div style="font-size:22px;font-weight:700;line-height:1.2">' + (avg.q_total || 0) + "</div></div>" +
      '  <div style="flex:2;padding:10px 12px;border-radius:10px;background:var(--panel-2)">' +
      '    <div class="hint">四维</div>' +
      '    <div style="font-size:12.5px;margin-top:3px">锐度 ' + (avg.q_blur || 0) +
      " · 噪点 " + (avg.q_noise || 0) + " · 曝光 " + (avg.q_exp || 0) +
      " · 遮挡 " + (p.occ_count != null ? (p.occ_count + " 张(" + p.occ_pct + "%)") : "—") + "</div></div>" +
      (function () {
        var bs = p.buckets || [];
        if (!bs.length) return "";
        var rows = bs.slice(0, 6).map(function (b) {
          return '<div style="display:flex;justify-content:space-between;font-size:12px;padding:2px 0">' +
            '<span style="color:var(--ink-3)">' + b.yaw + " · " + b.pitch + "</span>" +
            "<b>" + nf(b.count) + ' <span style="color:var(--ink-3);font-weight:400">留 ' + nf(b.keep) + "</span></b></div>";
        }).join("");
        return '<div style="margin-bottom:10px"><div class="hint" style="margin-bottom:4px">姿态分布（按张数）</div>' + rows + "</div>";
      })() +
      ro("总数", nf(p.total)) +
      ro("清晰度阈值 p" + p.blur_pct, p.thr_sharp) +
      ro("重复阈值", "汉明 ≤ " + p.dup_hamming + " · 像素差 ≤ " + p.dup_mae) +
      ro("模糊", nf(c.blur || 0)) +
      ro("纯色 / 空图", nf(c.flat || 0)) +
      ro("曝光异常", nf(c.exposure || 0)) +
      ro("遮挡（hull 未被 XSeg 覆盖）", nf(c.occ || 0)),
      ro("重复", nf(c.dup || 0)) +
      ro("打不开", nf(c.unreadable || 0)) +
      ro("模型轨（规则之外额外）", nf(c.score || 0)) +
      ((p.rescued && p.rescued.length) ? ro("稀缺角度救回", nf(p.rescued.length)) : "") +
      '<div style="margin-top:10px;padding-top:10px;border-top:1px solid var(--line)">' +
      ro("建议清理", nf(p.trash_count) + " 张（" + pct + "%）") +
      "</div>" +
      '<div class="hint" style="margin-top:10px">清理 = 移到 _trash 子目录，随时可以恢复；' +
      "本次扫描 " + elapsed + " s</div>" +
      (worst ? '<div style="margin-top:12px;padding-top:10px;border-top:1px solid var(--line)">' +
        '<div class="hint" style="margin-bottom:4px">最差 5 张（分数 · 三维）</div>' + worst + "</div>" : "");

    var btn = $("osApply");
    btn.style.display = "";
    btn.textContent = "清理 " + nf(p.trash_count) + " 张";
    btn.disabled = p.trash_count === 0;
    btn.onclick = function () { apply(p); };
    i18n();
  }

  function apply(p) {
    var btn = $("osApply");
    btn.disabled = true;
    btn.textContent = "清理中…";
    fetch("/api/dataset/apply", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ mode: "oneshot", dir: M.dir, names: p.trash })
    }).then(function (r) { return r.json(); }).then(function (d) {
      if (window.toast) window.toast("已清理", nf(d.moved || 0) + " 张移到 _trash（可恢复）", "success");
      close();
      if (window.refreshState) window.refreshState();
      if (window.dsRun) { try { window.dsRun(); } catch (e) {} }
    }).catch(function (e) {
      btn.disabled = false;
      btn.textContent = "重试";
      if (window.toast) window.toast("清理失败", String((e && e.message) || e).slice(0, 120), "error");
    });
  }

  function openOneShot(dir) {
    ensure();
    M.dir = dir || "data_src";
    M.result = null;
    M.el.style.display = "flex";
    $("osSub").textContent = M.dir;
    $("osBody").innerHTML = "";
    var lvBox = $("osLevel");
    if (lvBox && !lvBox.__bound) {
      lvBox.__bound = true;
      lvBox.addEventListener("click", function (e) {
        var b = e.target.closest ? e.target.closest("b") : null;
        if (!b) return;
        Array.prototype.forEach.call(lvBox.children, function (x) { x.classList.remove("on"); });
        b.classList.add("on");
        M.level = b.dataset.lv;
      });
    }
    var spBox = $("osScore");
    if (spBox && !spBox.__bound) {
      spBox.__bound = true;
      spBox.addEventListener("click", function (e) {
        var b = e.target.closest ? e.target.closest("b") : null;
        if (!b) return;
        Array.prototype.forEach.call(spBox.children, function (x) { x.classList.remove("on"); });
        b.classList.add("on");
        M.scorePct = parseFloat(b.dataset.sp);
      });
    }
    M.scorePct = (M.scorePct == null) ? 5 : M.scorePct;
    M.level = M.level || "strict";
    $("osProg").textContent = "扫描中…";
    $("osBar").style.width = "0";
    $("osApply").style.display = "none";

    fetch("/api/dataset/oneshot", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ dir: M.dir, level: M.level || "strict",
                             score_pct: M.scorePct == null ? 5 : M.scorePct })
    }).then(function (r) {
      return r.json().then(function (d) { if (!r.ok) throw new Error(d.detail || "启动失败"); return d; });
    }).then(function (d) {
      $("osSub").textContent = M.dir + " · " + nf(d.total) + " 张";
      if (M.timer) clearInterval(M.timer);
      M.timer = setInterval(poll, 400);
      poll();
    }).catch(function (e) {
      $("osProg").textContent = String((e && e.message) || e);
    });
  }

  function poll() {
    fetch("/api/dataset/oneshot/status").then(function (r) { return r.json(); }).then(function (d) {
      var total = d.total || 1, done = d.done || 0;
      var pct = Math.min(100, Math.round(1000 * done / total) / 10);
      $("osBar").style.width = pct + "%";
      $("osProg").textContent = "已扫 " + nf(done) + " / " + nf(total) + " 张 · " + d.elapsed + " s";
      if (d.error) {
        if (M.timer) { clearInterval(M.timer); M.timer = 0; }
        $("osProg").textContent = "失败：" + d.error;
        return;
      }
      if (!d.running && d.result) {
        if (M.timer) { clearInterval(M.timer); M.timer = 0; }
        $("osProg").textContent = "完成 · " + nf(d.result.total) + " 张 · " + d.elapsed + " s";
        $("osBar").style.width = "100%";
        renderResult(d.result, d.elapsed);
      }
    }).catch(function () {});
  }

  window.openOneShot = openOneShot;
})();


  /* ---------- 引擎编译页 ----------
     后端早就有 /api/engines 和 /api/engines/build，但前端从来没调用过，
     所以页面永远是空表格（用户看到的「并没有」）。 */
  (function () {
    var tbody = document.getElementById("engList");
    if (!tbody || tbody.dataset.bound) return;
    tbody.dataset.bound = "1";

    var stateText = { ready: "已编译", stale: "需重编", missing: "未编译" };
    var stateCls = { ready: "ok", stale: "warn", missing: "" };

    function row(m) {
      /* 状态列只放一个东西：
           已编译 → 绿色标签
           其余   → 按钮本身就是状态（点击即编译）
         不再「文字 + 按钮」并排，省掉一半宽度。 */
      var cell;
      if (m.state === "ready") {
        cell = '<span class="tag ok">已编译</span>';
      } else {
        var text = m.state === "stale" ? "更新" : "编译";
        var tip = m.state === "stale" ? "引擎需更新，点击重新编译" : "点击编译 " + m.name;
        cell = '<button class="mini" data-eng="' + m.name + '" title="' + tip + '">' + text + '</button>';
      }
      return '<tr>' +
        '<td>' + m.name + '</td>' +
        '<td class="mono" style="font-size:11px;color:var(--ink-3)">' + (m.md5 || "—") + '</td>' +
        '<td class="eng-num">' + (m.size_mb != null ? m.size_mb + ' MB' : '—') + '</td>' +
        '<td class="eng-num">' + (m.engine_mb != null ? m.engine_mb + ' MB' : '—') + '</td>' +
        '<td class="eng-st">' + cell + '</td>' +
        '</tr>';
    }

    /* 只在内容真的变了才写 DOM —— 每 2.5 秒无条件重写整张表会让标题和行闪烁 */
    var lastRows = "", lastSm = "";

    function paint(st) {
      var html = (st.models || []).map(row).join("") ||
        '<tr><td colspan="6" class="hint">没有可编译的模型</td></tr>';
      if (html !== lastRows) { tbody.innerHTML = html; lastRows = html; }

      var sm = document.getElementById("engSm");
      var job = st.job || {};
      var text = job.running
        ? ('正在编译 ' + (job.name || "") + ' · ' + (job.pct || 0) + '%')
        : ('GPU SM ' + (st.sm || "?") + ' · 引擎目录 ' + (st.dir || ""));
      if (sm && sm.textContent !== text) { sm.textContent = text; }
    }

    function load() {
      return fetch("/api/engines").then(function (r) { return r.json(); })
        .then(paint).catch(function () {});
    }

    function build(name) {
      fetch("/api/engines/build", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ name: name })
      }).then(function (r) { return r.json(); }).then(function () {
        toast("开始编译", name, "info");
        load();
      }).catch(function (e) {
        toast("编译失败", String((e && e.message) || e), "error");
      });
    }

    tbody.addEventListener("click", function (e) {
      var t = e.target.closest("[data-eng]");
      if (t) build(t.getAttribute("data-eng"));
    });

    var all = document.getElementById("engAll");
    if (all) {
      all.addEventListener("click", function () {
        fetch("/api/engines").then(function (r) { return r.json(); }).then(function (st) {
          var todo = (st.models || []).filter(function (m) { return m.state !== "ready"; });
          if (!todo.length) { toast("全部编译", "所有引擎都已就绪", "info"); return; }
          var i = 0;
          (function next() {
            if (i >= todo.length) { load(); return; }
            var name = todo[i++].name;
            fetch("/api/engines/build", {
              method: "POST", headers: { "Content-Type": "application/json" },
              body: JSON.stringify({ name: name })
            }).then(function () {
              var t = setInterval(function () {
                fetch("/api/engines").then(function (r) { return r.json(); }).then(function (s) {
                  paint(s);
                  if (!(s.job && s.job.running)) { clearInterval(t); next(); }
                });
              }, 1500);
            });
          })();
        });
      });
    }

    load();
    setInterval(function () {
      var page = document.getElementById("p-tool-engine");
      if (page && page.offsetParent !== null) load();
    }, 2500);
  })();


  /* 合成器的 4 个 select 也走同一个增强：
     页面是静态 HTML，之前没经过 __enhanceSelects，所以弹层一直是系统原生的。
     这里在启动时跑一次，并在每次切到「合成器」时再跑一次
     （合成器的面板可能被重绘，重绘会换掉新的 select 节点）。 */
  function enhanceMergeSelects() {
    var host = document.getElementById("p-merge");
    if (!host || !window.__enhanceSelects) return;
    try { window.__enhanceSelects(host); } catch (e) { console.warn("[dd] merge enhance failed", e); }
  }

  (function () {
    if (document.readyState === "loading") {
      document.addEventListener("DOMContentLoaded", function () { setTimeout(enhanceMergeSelects, 0); });
    } else {
      setTimeout(enhanceMergeSelects, 0);
    }
    /* 给模型卡片用：直接开某个模型的转换 */
    window.cvOpen = function (name) {
      window.__cvPick = name;
      if (typeof window.go === "function") window.go("train-convert");
      else load();
    };

    var _go = window.go;
    if (typeof _go === "function" && !_go.__mergeDd) {
      var wrapped = function (k) {
        var r = _go.apply(this, arguments);
        if (k === "merge") setTimeout(enhanceMergeSelects, 30);
        return r;
      };
      wrapped.__mergeDd = true;
      window.go = wrapped;
      window.go = wrapped;
    }
  })();


  /* ---------------- 工具：模型转换（.npy -> .pth，Stepper） ---------------- */
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
        box.innerHTML = '<div class="cvempty">workspace/model 里没有 *_weights.npy。把老版本的 .npy 模型放进去再刷新。</div>';
        return;
      }
      box.innerHTML = items.map(function (it, i) {
        var tag = it.converted
          ? '<span class="tag">已转换</span>'
          : '<button class="mini" data-cv="' + i + '">转换</button>';
        return '<div class="cvrow"><span class="n">' + esc(it.name) + '</span>' +
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
        .catch(function () { renderList([]); });
    }

    function step(el, cls, mark) {
      var node = $(el);
      if (!node) return;
      node.className = "cvst " + (cls || "");
      var ic = node.querySelector(".cvic");
      if (ic) ic.textContent = mark;
    }

    function paint(st) {
      var card = $("cvCard");
      if (!card) return;
      card.style.display = "";
      $("cvName").textContent = st.name || "model";
      $("cvMeta").textContent = st.modules && st.modules.length ? (st.modules.length + " 个模块") : "";
      $("cvBad").className = "cvbad" + (st.step === "error" ? " on" : "");

      if (st.step === "load") {
        step("cvS1", "on", "1"); step("cvS2", "", "2"); step("cvS3", "", "3");
      } else if (st.step === "convert") {
        step("cvS1", "done", "✓"); step("cvS2", "on open", "2"); step("cvS3", "", "3");
      } else if (st.step === "done") {
        step("cvS1", "done", "✓"); step("cvS2", "done", "✓"); step("cvS3", "done", "✓");
      } else if (st.step === "error") {
        step("cvS1", "done", "✓"); step("cvS2", "err open", "!"); step("cvS3", "", "3");
      }

      var rows = $("cvRows");
      if (rows) {
        var html = (st.modules || []).map(function (m) {
          var state = m.state === "ok" ? "ok" : (st.step === "error" && m.state !== "ok" ? "wait" : (m.state === "run" ? "run" : "wait"));
          var mk = state === "ok" ? "✓" : (state === "run" ? "›" : "·");
          var cnt = state === "ok" ? "完成" : (state === "run" ? "转换中" : "等待");
          return '<div class="cvrowi ' + state + '"><span class="mk">' + mk + '</span>' +
                 '<span class="n">' + esc(m.name) + '</span><span class="c">' + cnt + '</span></div>';
        }).join("");
        if (st.step === "error" && st.error) {
          html += '<div class="cvrowi bad"><span class="mk">!</span><span class="n">' +
                  esc(st.error) + '</span><span class="c">失败</span></div>';
        }
        rows.innerHTML = html;
      }
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
    setTimeout(function () { if ($("p-train-convert")) load(); }, 400);
  })();

  /* ---------- 设置页 · B 站账号（未登录 → 打开欢迎页扫码） ---------- */
  (function () {
    function gid(id) { return document.getElementById(id); }
    function render(d) {
      var tag = gid("biliTag"), info = gid("biliInfo"), btn = gid("btnBiliAction");
      if (!tag) return;
      var on = !!(d && d.logged_in);
      tag.textContent = on ? "已登录" : "未登录";
      tag.className = on ? "tag ok" : "tag";
      if (info) info.textContent = on
        ? ((d.name || "已登录") + (d.uid ? " · UID " + d.uid : ""))
        : "登录后可下载 1080P / 会员清晰度";
      if (btn) { btn.textContent = on ? "退出登录" : "去登录"; btn.dataset.on = on ? "1" : ""; }
    }
    function load() {
      fetch("/api/bili/me").then(function (r) { return r.json(); })
        .then(render).catch(function () { render(null); });
    }
    window.loadBili = load;

    var btn = gid("btnBiliAction");
    if (btn) btn.addEventListener("click", function () {
      if (btn.dataset.on === "1") {
        btn.disabled = true;
        Promise.all([
          fetch("/api/bili/logout", { method: "POST" }),
          fetch("/api/auth/logout", { method: "POST" })
        ]).then(function () {
          btn.disabled = false;
          if (window.toast) window.toast("B 站账号", "已退出登录", "success");
          location.reload();
        }).catch(function () { btn.disabled = false; });
      } else {
        /* 直接打开欢迎页的扫码界面 */
        var w = gid("welcome");
        if (w) { w.style.display = "grid"; if (window.biliQr) window.biliQr(); }
      }
    });

    var _goS = window.go;
    if (typeof _goS === "function" && !_goS.__biliCard) {
      var wrap = function (k) { var r = _goS.apply(this, arguments); if (k === "settings") setTimeout(load, 40); return r; };
      wrap.__biliCard = true;
      window.go = wrap;
    }
    setTimeout(load, 900);
  })();

  /* ---------- 欢迎页：三张卡渐进披露（选中展开，其余压暗） ---------- */
  (function () {
    var wrap = document.querySelector(".wfeats");
    if (!wrap) return;
    wrap.addEventListener("click", function (e) {
      var btn = e.target.closest ? e.target.closest("[data-wf]") : null;
      if (!btn) return;
      var on = btn.classList.contains("on");
      wrap.querySelectorAll("[data-wf]").forEach(function (b) { b.classList.remove("on"); });
      if (!on) btn.classList.add("on");
    });
  })();

  /* ---------- 欢迎页：直接显示扫码 ---------- */
  (function () {
    var tried = false;
    function kick() {
      if (tried) return;
      var w = document.getElementById("welcome");
      if (!w || w.style.display === "none") return;
      tried = true;
      if (window.biliQr) window.biliQr();
    }
    document.addEventListener("DOMContentLoaded", kick);
    var timer = setInterval(function () {
      kick();
      if (tried) clearInterval(timer);
    }, 400);
    setTimeout(function () { clearInterval(timer); }, 15000);
  })();

  /* ---------- 欢迎页：让玻璃条对齐背景图里的条带 ---------- */
  (function () {
    var IMG_W = 1193, IMG_H = 576, BAND_T = 224, BAND_B = 384;
    function sync() {
      var w = document.getElementById("welcome");
      if (!w) return;
      var vw = w.clientWidth || innerWidth;
      var vh = w.clientHeight || innerHeight;
      var scale = Math.max(vw / IMG_W, vh / IMG_H);
      var dh = IMG_H * scale, ox = (vw - IMG_W * scale) / 2, oy = (vh - dh) / 2;
      w.style.setProperty("--band-top", (oy + BAND_T * scale) + "px");
      w.style.setProperty("--band-h", ((BAND_B - BAND_T) * scale) + "px");
    }
    addEventListener("resize", sync);
    document.addEventListener("DOMContentLoaded", sync);
    sync();
    window.__bandSync = sync;
  })();


  /* ---------- 使用许可 / 免责声明 ---------- */
  (function () {
    var KEY = "dfl-legal-v3";   /* 协议位置/流程变了，旧的一次性勾选不再作数 */
    var gate = document.getElementById("legalGate");
    var checks = [].slice.call(document.querySelectorAll("[data-legal-check]"));
    var okBtn = document.getElementById("legalAccept");
    var noBtn = document.getElementById("legalDecline");
    if (!gate || !checks.length || !okBtn) return;

    function accepted() {
      try { return localStorage.getItem(KEY) === "1"; } catch (e) { return false; }
    }
    function open() {
      gate.style.display = "flex";
      /* 协议在登录页上展示时，把扫码卡和品牌字收起来，只留背景动画 */
      var wl = document.getElementById("welcome");
      if (wl) wl.classList.add("legal-on");
      document.documentElement.classList.add("legal-open");
    }
    function close() {
      gate.style.display = "none";
      var wl2 = document.getElementById("welcome");
      if (wl2) wl2.classList.remove("legal-on");
      document.documentElement.classList.remove("legal-open");
    }

    function sync() {
      /* 协议在**扫码登录之后**才弹：没登录时先好好扫码，别一进来就堵着。 */
      var must = !!window.__loggedIn && !accepted();
      if (must && gate.style.display === "none") open();
      if (!must && gate.style.display !== "none") close();
    }
    window.__legalSync = sync;
    window.__legalAccepted = accepted;

    /* 三条都必须勾选 */
    function refreshOk() {
      okBtn.disabled = !checks.every(function (c) { return c.checked; });
    }
    checks.forEach(function (c) { c.addEventListener("change", refreshOk); });
    refreshOk();

    okBtn.addEventListener("click", function () {
      try {
        localStorage.setItem(KEY, "1");
        localStorage.setItem(KEY + ".at", new Date().toISOString());
      } catch (e) {}
      close();
      /* 同意之后用当前登录态重绘一次：欢迎页该收起来了，别让用户再扫一次码 */
      try { if (window.applyAuth) window.applyAuth(); } catch (e) {}
      if (window.__afterLegal) { var f = window.__afterLegal; window.__afterLegal = null; f(); }
    });

    noBtn.addEventListener("click", function () {
      Promise.all([
        fetch("/api/auth/logout", { method: "POST" }),
        fetch("/api/bili/logout", { method: "POST" })
      ]).catch(function () {}).then(function () {
        try { localStorage.removeItem(KEY); } catch (e) {}
        try {
          document.cookie.split(";").forEach(function (c) {
            var k = c.split("=")[0].trim();
            if (k) document.cookie = k + "=; expires=Thu, 01 Jan 1970 00:00:00 GMT; path=/";
          });
        } catch (e) {}
        location.reload();
      });
    });

    setInterval(sync, 600);
    sync();
  })();

/* ============================================================================
 * 权重分包门控 —— 工具 → 遮罩绘制
 *   weights/ 按功能分包：core 必需（缺了这页没法用），sam 可选（约 3.4 GB）。
 *   后端接口（并行开发中，没上线时会 404 —— 一律按"未知"处理，不报错、不遮挡）：
 *     GET  /api/weights/status        → { ok, packages:{ core:{installed}, sam:{installed,size_mb} } }
 *     POST /api/weights/fetch         { group:"sam" } → { ok, job }
 *     GET  /api/weights/fetch/status  → { ok, running, pct, line, error }
 *   遮罩绘制那三块控件（SAM / GROUNDED SAM2 / BISENET）长在 iframe
 *   /mask-editor/index.html 里，是独立文档 —— app.css 打不进去，
 *   所以这里同源直连 contentDocument：置灰走 disabled + 半透明，
 *   提示条也插在它自己的 .right-panel 里，配色全部借编辑器已有的 token
 *   （--m-line / --m-panel-2 / --m-ink-2，按钮复用 .mkbtn，红沿用 #ef4444）。
 * ============================================================================ */
/* ==== WKG-START ==== 这段可以单独切出来喂假 DOM 跑状态机，别在里面引用外部变量 */
(function () {
  "use strict";

  var SIZE_MB = 3455;   /* 后端没给 size_mb 时的兜底值（= 3.4 GB） */

  /* ------------------------------------------------------------ 纯状态机 */

  /** /api/weights/status 的响应 → {core,sam,samMB}；形状不对返回 null（= 未知）。 */
  function wgParse(d) {
    if (!d || typeof d !== "object") return null;
    var pk = d.packages;
    if (!pk || typeof pk !== "object") return null;
    var core = pk.core, sam = pk.sam;
    if (!core || typeof core.installed !== "boolean") return null;
    var mb = (sam && typeof sam.size_mb === "number" && sam.size_mb > 0) ? sam.size_mb : SIZE_MB;
    return { core: core.installed === true, sam: !!(sam && sam.installed === true), samMB: mb };
  }

  /** 3455 → "3.4 GB" */
  function wgSize(mb) {
    var g = (Number(mb) || SIZE_MB) / 1024;
    return g.toFixed(1) + " GB";
  }

  /** 按钮文案：只有下载中带百分比，失败/中断回到「重试」。 */
  function wgLabel(phase, pct) {
    if (phase === "downloading") return "下载中 " + pct + "%";
    if (phase === "failed") return "重试";
    return "下载";
  }

  /**
   * 状态机：status（wgParse 的结果）+ 运行时（{phase,pct,err}）→ 视图
   *   unknown     接口没就绪 / 读失败 → 什么都不做，维持现状
   *   nocore      core 没装（理论上不该）→ 顶部红条 + 「打开数据集」之外主要操作置灰
   *   missing     sam 没装 → 三块置灰 + 上方提示条 + 「下载」
   *   downloading 正在下 → 三块继续置灰 + 按钮「下载中 42%」
   *   failed      失败/中断 → 三块继续置灰 + 按钮「重试」+ 一行错误
   *   ready       sam 装好 → 解锁，提示条撤掉
   */
  function wgView(st, rt) {
    rt = rt || {};
    var phase = "unknown";
    if (st) {
      if (!st.core) phase = "nocore";
      else if (st.sam) phase = "ready";
      else if (rt.phase === "downloading") phase = "downloading";
      else if (rt.phase === "failed") phase = "failed";
      else phase = "missing";
    }
    var pct = Math.round(Number(rt.pct) || 0);
    if (pct < 0) pct = 0;
    if (pct > 100) pct = 100;
    var gate = (phase === "missing" || phase === "downloading" || phase === "failed");
    return {
      phase: phase,
      gate: gate,
      label: wgLabel(phase, pct),
      pct: pct,
      size: wgSize(st ? st.samMB : SIZE_MB),
      err: phase === "failed" ? String(rt.err || "下载中断，点「重试」再来一次") : "",
      core: phase === "nocore"
    };
  }

  /* -------------------------------------------------------- 画到 iframe 里 */

  function wgFrame() { return document.getElementById("mkFrame"); }

  /** 同源 iframe 的文档；拿不到（还没 load / 跨域）返回 null。 */
  function wgDoc() {
    var f = wgFrame();
    if (!f) return null;
    try {
      var d = f.contentDocument || (f.contentWindow && f.contentWindow.document);
      return (d && typeof d.createElement === "function") ? d : null;
    } catch (e) { return null; }
  }

  /** 按标题文字找右栏那三块（"SAM" / "GROUNDED SAM2" / "BISENET"）。 */
  function wgSection(doc, title) {
    var secs = doc.querySelectorAll(".right-panel .panel-section") || [];
    for (var i = 0; i < secs.length; i++) {
      var t = secs[i].querySelector(".panel-section__title");
      if (t && String(t.textContent || "").trim().toUpperCase() === title) return secs[i];
    }
    return null;
  }

  /**
   * 置灰 / 还原：disabled 挡住点击，opacity + pointer-events 让它看着也用不了。
   * key 是"谁上的锁"：core 一层、sam 一层，各记各的基线，
   * 否则整页锁还原时会把三块刚上的锁一起解掉（右栏锁包含那三块）。
   */
  function wgLockBox(el, on, key) {
    if (!el) return;
    var k = key || "data-wkgwas";
    var ctl = el.querySelectorAll("button, input, select, textarea") || [];
    for (var i = 0; i < ctl.length; i++) {
      var c = ctl[i];
      if (on) {
        if (c.getAttribute(k) === null) c.setAttribute(k, c.disabled ? "1" : "0");
        c.disabled = true;
      } else if (c.getAttribute(k) !== null) {
        c.disabled = c.getAttribute(k) === "1";
        c.removeAttribute(k);
      }
    }
    if (el.style) {
      el.style.opacity = on ? "0.45" : "";
      el.style.pointerEvents = on ? "none" : "";
    }
  }

  /** 三块上方那条提示条（懒建；建好之后每帧只改文字）。 */
  function wgBox(doc) {
    var bar = doc.getElementById("wkgSamBar");
    if (bar) return bar;
    bar = doc.createElement("div");
    bar.id = "wkgSamBar";
    bar.style.cssText =
      "display:flex;align-items:center;gap:6px;flex-wrap:wrap;margin:0 0 8px;" +
      "padding:8px 10px;border:1px solid var(--m-line);border-radius:9px;" +
      "background:var(--m-panel-2);color:var(--m-ink-2);font-size:10.5px;line-height:1.45";

    var txt = doc.createElement("div");
    txt.id = "wkgSamTxt";
    txt.style.cssText = "flex:1 1 auto;min-width:0";
    txt.textContent = "遮罩绘制需要额外模型（约 3.4 GB）";
    bar.appendChild(txt);

    var btn = doc.createElement("button");
    btn.id = "wkgSamBtn";
    btn.type = "button";
    btn.className = "mkbtn";                       /* 复用编辑器里「打开」那颗按钮 */
    btn.style.cssText = "height:24px;padding:0 10px;font-size:10.5px";
    btn.textContent = "下载";
    btn.onclick = function () { wgStart(); };
    bar.appendChild(btn);

    var err = doc.createElement("div");
    err.id = "wkgSamErr";
    err.style.cssText = "display:none;flex:1 1 100%;font-size:10px;color:#ef4444";
    bar.appendChild(err);
    return bar;
  }

  /** 插到第一块（SAM）前面；找不到就追加到右栏末尾。 */
  function wgPut(doc, bar, before) {
    var host = before && before.parentNode;
    if (!host) {
      host = doc.querySelector(".right-panel") || doc.body;
      before = null;
    }
    if (!host || bar.parentNode === host) return;
    host.insertBefore(bar, before || null);
  }

  /** core 缺包时页面顶部的红条。 */
  function wgCoreBar(doc) {
    var cb = doc.getElementById("wkgCoreBar");
    if (cb) return cb;
    cb = doc.createElement("div");
    cb.id = "wkgCoreBar";
    cb.style.cssText =
      "display:flex;align-items:center;gap:8px;padding:9px 14px;" +
      "font-size:11.5px;line-height:1.4;color:#ef4444;" +
      "background:rgba(239,68,68,0.10);border-bottom:1px solid rgba(239,68,68,0.28)";
    cb.textContent = "缺少基础模型，请先运行 modelhub.fetch --group core";
    var shell = doc.querySelector(".app-shell") || doc.body;
    if (shell) shell.insertBefore(cb, shell.firstChild || null);
    return cb;
  }

  /**
   * 把视图贴到编辑器文档上。doc 可以是真 Document，也可以是假 DOM。
   * 找不到那三块（编辑器改版）就整体不动，别把别人的布局搅了。
   */
  function wgPaintDoc(doc, view) {
    if (!doc || !view) return false;

    /* core 缺包管的是整页，必须先落锁/解锁 —— 放到三块后面会被它们的还原盖掉 */
    var cb = doc.getElementById("wkgCoreBar");
    if (view.core) {
      wgCoreBar(doc);
      wgLockBox(doc.querySelector(".toolbar"), true, "data-wkgcore");
      wgLockBox(doc.querySelector(".mode-bar"), true, "data-wkgcore");
      wgLockBox(doc.querySelector(".right-panel"), true, "data-wkgcore");
    } else if (cb && cb.parentNode) {
      cb.parentNode.removeChild(cb);
      wgLockBox(doc.querySelector(".toolbar"), false, "data-wkgcore");
      wgLockBox(doc.querySelector(".mode-bar"), false, "data-wkgcore");
      wgLockBox(doc.querySelector(".right-panel"), false, "data-wkgcore");
    }

    var titles = ["SAM", "GROUNDED SAM2", "BISENET"];
    var lockSecs = view.core || view.gate;   /* core 缺时也锁着，别让两层互相解 */
    var first = null, i, sec;
    for (i = 0; i < titles.length; i++) {
      sec = wgSection(doc, titles[i]);
      if (sec && !first) first = sec;
      wgLockBox(sec, lockSecs, "data-wkgsec");
    }

    var bar = doc.getElementById("wkgSamBar");
    if (view.gate && first) {
      bar = wgBox(doc);
      var txt = doc.getElementById("wkgSamTxt");
      if (txt) txt.textContent = "遮罩绘制需要额外模型（约 " + view.size + "）";
      var btn = doc.getElementById("wkgSamBtn");
      if (btn) {
        btn.textContent = view.label;                        /* 下载 / 下载中 42% / 重试 */
        btn.disabled = (view.phase === "downloading");       /* 下载中不许连点 */
      }
      var err = doc.getElementById("wkgSamErr");
      if (err) {
        err.textContent = view.err || "";
        err.style.display = view.err ? "block" : "none";
      }
      wgPut(doc, bar, first);
    } else if (bar && bar.parentNode) {
      bar.parentNode.removeChild(bar);
    }
    return true;
  }

  /* ---------------------------------------------------------------- 运行时 */

  var RT = { status: null, phase: "idle", pct: 0, err: "", timer: 0 };

  function wgPaint() {
    var doc = wgDoc();
    if (!doc) return false;
    try { return wgPaintDoc(doc, wgView(RT.status, RT)); } catch (e) { return false; }
  }

  function wgStop() {
    if (RT.timer) { clearInterval(RT.timer); RT.timer = 0; }
  }

  /** 失败/中断收口：轮询停掉、按钮回到「重试」、提示条给一行错误。 */
  function wgFail(msg) {
    wgStop();
    RT.phase = "failed";
    RT.err = String(msg || "下载中断");
    wgPaint();
  }

  /** GET status —— 读不到（404 / 网络失败）就当"未知"，维持现状。 */
  function wgLoad() {
    return fetch("/api/weights/status")
      .then(function (r) { return r.ok ? r.json() : null; })
      .then(function (d) {
        RT.status = wgParse(d);
        wgPaint();
        return RT.status;
      })
      .catch(function () { RT.status = null; wgPaint(); return null; });
  }

  /** 跑完/停了之后重新读 status，控件按真实安装状态恢复（不用刷新页面）。 */
  function wgDone(d) {
    return fetch("/api/weights/status")
      .then(function (r) { return r.ok ? r.json() : null; })
      .then(function (j) {
        RT.status = wgParse(j);
        if (RT.status && RT.status.sam) {
          RT.phase = "idle"; RT.err = ""; RT.pct = 100;
          wgPaint();
          return;
        }
        wgFail((d && (d.error || d.detail || d.line)) || "下载没有完成，已中断");
      })
      .catch(function () { wgFail("下载停了但读不到安装状态，点「重试」再确认一次"); });
  }

  /** 1000ms 一跳：running 就一直报进度，停了就收口。 */
  function wgTick() {
    return fetch("/api/weights/fetch/status")
      .then(function (r) { return r.ok ? r.json() : null; })
      .then(function (d) {
        if (!d || typeof d !== "object") throw new Error("下载状态接口没有响应");
        RT.pct = Math.round(Number(d.pct) || 0);
        if (d.running) { wgPaint(); return; }
        wgStop();
        if (d.error) { wgFail(d.error); return; }
        return wgDone(d);
      })
      .catch(function (e) { wgFail((e && e.message) || "下载中断"); });
  }

  /** 点「下载」：POST 起任务 → 1s 轮询，pct 直接写到按钮文字上。 */
  function wgStart() {
    if (RT.phase === "downloading") return;          /* 已经在下，别再起一个 */
    RT.phase = "downloading";
    RT.pct = 0;
    RT.err = "";
    wgPaint();                                        /* 立刻变「下载中 0%」并置灰 */
    fetch("/api/weights/fetch", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ group: "sam" })
    }).then(function (r) {
      if (!r.ok) throw new Error(r.status === 404 ? "下载接口还没就绪，稍后再试" : ("下载接口返回 " + r.status));
      return r.json();
    }).then(function (d) {
      if (d && d.ok === false) throw new Error(String(d.error || d.detail || "下载任务没能启动"));
      wgStop();
      RT.timer = setInterval(wgTick, 1000);
    }).catch(function (e) { wgFail((e && e.message) || "下载启动失败"); });
  }

  /** 进入「工具 → 遮罩绘制」时调用。 */
  function wgEnter() {
    var f = wgFrame();
    if (f && !f.dataset.wkgBound) {
      f.dataset.wkgBound = "1";
      /* 换目录 / 首次加载都会重设 src，画好的状态要重新贴一次 */
      f.addEventListener("load", function () { wgPaint(); });
    }
    wgPaint();          /* 先把上次读到的状态贴上去，切页回来不闪 */
    return wgLoad();    /* 再按后端最新状态刷一遍 */
  }

  /* 刷新 / 深链接直接落在这一页时：主 IIFE 的 go() 比这段先跑，那时 window.WKG 还不存在，
     所以这里补一次（有 hash 就当作"进页"）。 */
  try {
    document.addEventListener("DOMContentLoaded", function () {
      var h = "";
      try { h = String((window.location && window.location.hash) || "").replace(/^#/, ""); } catch (e2) {}
      if (h === "tool-mask") wgEnter();
    });
  } catch (e3) {}

  window.WKG = { enter: wgEnter, refresh: wgLoad, start: wgStart, paint: wgPaint };
  /* 单测入口：纯函数 + 画布函数都能直接喂假文档 */
  window.WKG._test = { parse: wgParse, view: wgView, label: wgLabel, size: wgSize, paint: wgPaintDoc };
})();
/* ==== WKG-END ==== */

/* ============================================================================
 * 设置 → 权重包 —— 「缺哪个包、缺几个、点一下能下」
 *   GET  /api/weights/status        → { packages:{ core:{installed,missing,missing_mb,…} } }
 *   POST /api/weights/fetch         { group, url_base } → { ok, job }
 *   GET  /api/weights/fetch/status  → { running, pct, line, error }
 *   在这块 UI 之前，缺 errface_xgb.ubj 只能靠命令行 modelhub.fetch 解决，
 *   界面上说「缺模型」却没地方点 —— 至少要有地方能看、能下、能知道下到哪一步。
 * ============================================================================ */
(function () {
  "use strict";

  var TIMER = 0, BUSY = "";

  function $(id) { return document.getElementById(id); }

  /* 后端来的动态句子也要进词典（切英文不能留中文） */
  function addDict(zh, en) {
    try {
      var d = window.I18N && window.I18N.dict;
      if (d && zh && d[zh] === undefined) d[zh] = en || zh;
    } catch (e) {}
  }
  function say(zh, en) { addDict(zh, en); return zh; }

  function size(mb) {
    var n = Number(mb) || 0;
    if (!n) return "";
    return n >= 1024 ? (n / 1024).toFixed(1) + " GB" : n.toFixed(1) + " MB";
  }

  function stateText(p) {
    if (!p) return say("未检测", "unknown");
    if (p.installed) return say("已装", "installed");
    var miss = Number(p.missing) || 0;
    var sz = size(p.missing_mb);
    return say("缺 " + miss + " 个文件" + (sz ? "（" + sz + "）" : ""),
               miss + " files missing" + (sz ? " (" + sz + ")" : ""));
  }

  function row(key, pkg) {
    var el = document.createElement("div");
    el.style.cssText = "display:flex;align-items:center;gap:10px;padding:5px 0;font-size:12px";

    var name = document.createElement("span");
    name.style.cssText = "width:64px;flex:0 0 64px;color:var(--ink-2);font-weight:600";
    name.textContent = key;                     /* core / sam 是包名，不翻 */
    el.appendChild(name);

    var st = document.createElement("span");
    st.style.cssText = "flex:1 1 auto;min-width:0;color:var(--ink-3);font-size:11.5px";
    st.textContent = stateText(pkg);
    st.title = st.textContent;
    el.appendChild(st);

    if (!pkg || !pkg.installed) {
      var b = document.createElement("button");
      b.className = "mini";
      b.type = "button";
      b.textContent = "下载";
      b.disabled = !!BUSY;
      b.addEventListener("click", function () { fetchGroup(key, b); });
      el.appendChild(b);
    }
    return el;
  }

  function paint(d) {
    var box = $("pkgBox"), tag = $("pkgTag");
    if (!box) return;
    box.innerHTML = "";
    box.className = "";
    if (!d || !d.ok) {
      box.className = "hint";
      box.textContent = say("读不到权重清单：", "Cannot read the weight list: ") +
                        String((d && (d.error || d.detail)) || "weights/manifest.json");
      if (tag) { tag.textContent = say("未知", "unknown"); tag.className = "tag t-gray"; }
      return;
    }
    var pk = d.packages || {};
    var keys = Object.keys(pk);
    if (!keys.length) {
      box.className = "hint";
      box.textContent = say("清单里没有功能包", "The list declares no packs");
      return;
    }
    keys.sort(function (a, b) { return (pk[a].required ? 0 : 1) - (pk[b].required ? 0 : 1); });
    keys.forEach(function (k) { box.appendChild(row(k, pk[k])); });
    var miss = keys.filter(function (k) { return !pk[k].installed; });
    if (tag) {
      tag.textContent = miss.length ? say("缺 " + miss.length + " 个包", miss.length + " packs missing")
                                    : say("齐全", "all installed");
      tag.className = "tag " + (miss.length ? "t-yellow" : "t-green");
    }
    var msg = $("pkgMsg");
    if (msg && !BUSY) {
      msg.textContent = d.fetch && d.fetch.ready === false
        ? say("下载器不可用：" + (d.fetch.hint || "modelhub.fetch 不在"), "")
        : "";
    }
  }

  function load() {
    var box = $("pkgBox");
    if (box) { box.className = "hint"; box.textContent = "正在读取权重清单…"; }
    return fetch("/api/weights/status")
      .then(function (r) { return r.ok ? r.json() : null; })
      .then(function (d) { paint(d); return d; })
      .catch(function () { paint(null); return null; });
  }

  function setMsg(txt) {
    var m = $("pkgMsg");
    if (m) m.textContent = txt || "";
  }

  function stopTick() { if (TIMER) { clearInterval(TIMER); TIMER = 0; } }

  function tick(group) {
    fetch("/api/weights/fetch/status").then(function (r) { return r.ok ? r.json() : null; })
      .then(function (s) {
        if (!s || typeof s !== "object") throw new Error("no status");
        var line = s.line || "";
        setMsg(s.pct + "% · " + line);
        if (s.running) return;
        stopTick();
        BUSY = "";
        if (s.error) { setMsg(say("下载失败：" + s.error, "Download failed: " + s.error)); load(); return; }
        setMsg(say("下载完成，重新检测…", "Download finished — re-checking…"));
        load();
      })
      .catch(function () { stopTick(); BUSY = ""; setMsg(say("读不到下载进度", "Cannot read download progress")); });
  }

  function fetchGroup(group, btn) {
    if (BUSY) return;
    var base = String(($("pkgUrl") && $("pkgUrl").value) || "").trim();
    if (!base) {
      /* 清单里没有 url，必须给镜像 —— 别让用户点了才在日志里看到这句 */
      setMsg(say("先填镜像地址：下载器需要 https:// 前缀的权重站（清单里没有内置 url）。",
                 "Enter a mirror first: the downloader needs an https:// weight site (the list has no built-in url)."));
      return;
    }
    BUSY = group;
    if (btn) { btn.disabled = true; btn.textContent = "准备中…"; }
    setMsg("…");
    fetch("/api/weights/fetch", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ group: group, url_base: base })
    }).then(function (r) {
      return r.json().then(function (d) {
        if (!r.ok) throw new Error((d && (d.detail || d.error)) || ("HTTP " + r.status));
        return d;
      });
    }).then(function () {
      stopTick();
      TIMER = setInterval(function () { tick(group); }, 1000);
      tick(group);
    }).catch(function (e) {
      BUSY = "";
      if (btn) { btn.disabled = false; btn.textContent = "下载"; }
      setMsg(say("下载没能启动：" + ((e && e.message) || e), ""));
    });
  }

  function enter() {
    var box = $("pkgBox");
    if (!box) return null;
    if (!box.dataset.pkBound) {
      box.dataset.pkBound = "1";
      var r = $("pkgRefresh");
      if (r) r.addEventListener("click", function () { load(); });
      var u = $("pkgUrl");
      if (u) {
        try { u.value = window.localStorage.getItem("dfl-weights-mirror") || ""; } catch (e) {}
        u.addEventListener("change", function () {
          try { window.localStorage.setItem("dfl-weights-mirror", String(u.value || "").trim()); } catch (e) {}
        });
      }
    }
    return load();
  }

  window.WPK = { enter: enter, refresh: load, _test: { stateText: stateText, size: size } };

  /* 刷新 / 深链接直接落在设置页：主 IIFE 的 go() 在这段之前就跑过了，
     那时 window.WPK 还不存在，这里补一次（和 WKG 对 tool-mask 的处理一致）。 */
  try {
    document.addEventListener("DOMContentLoaded", function () {
      var h = "";
      try { h = String((window.location && window.location.hash) || "").replace(/^#/, ""); } catch (e) {}
      if (h === "settings") enter();
    });
  } catch (e2) {}
})();

