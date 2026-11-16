/* DeepFaceLab 4.1 — 前端。所有数据都来自本地 /api，界面本身不含任何算法。 */
/* ==========================================================================
   分节目录 —— 代码**已按页拆成 app/web/js/app/ 下的 15 个分片**，由后端拼成 /static/js/app.js：
     00-core（基础设施）· 10-merge-settings · 15-proj-out · 16-pipe-draw · 17-water · 18-bypass
     · 19-eff-drawer · 20-progress-fused · 30-studio-timeline · 31-studio-faces · 32-mask-page
     · 40-train · 41-dataset-cfg · 50-misc · 60-tail
   外层是**一个 IIFE 跨所有分片**，所以分片单独 node --check 会报"括号不匹配" ——
   改完请跑 node app/tools/js/check_bundle.js（会报出具体分片与行号）。
   拼接是逐字节等价的（拆分时校验过 roundtrip），运行语义与原 app.js 完全相同。

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

