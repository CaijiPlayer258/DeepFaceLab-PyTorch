/* DeepFaceLab 4.1 — 前端。所有数据都来自本地 /api，界面本身不含任何算法。 */
(function () {
  "use strict";

  var PAGE_IDS = {
    home: "p-home", extract: "p-extract", dataset: "p-dataset", train: "p-train",
    run: "p-run", "train-cfg": "p-train-cfg",
    "tool-mask": "p-tool-mask", "tool-bili": "p-tool-bili",
    "tool-color": "p-tool-color", "tool-angle": "p-tool-angle", "tool-engine": "p-tool-engine",
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
  var NAV_ALIAS = { run: "train", "train-cfg": "train", debug: "about" };

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
        el.addEventListener("click", function () { if (window.mgStep) window.mgStep(c.step); });
      } else if (c.go && i < trail.length - 1) {
        el.classList.add("link");
        el.addEventListener("click", function () { go(c.go); });
      }
      box.appendChild(el);
    });
  }

  /* 每个导航组的最后位置：点「训练器」应回到上次待的界面，而不是永远回列表 */
  var LAST_PAGE = {};
  var PAGE_GROUP = { run: "train", "train-cfg": "train", "tool-mask": "tools",
                     "tool-bili": "tools", "tool-color": "tools", "tool-angle": "tools",
                     debug: "about" };

  function groupOf(key) { return PAGE_GROUP[key] || key; }

  /* 子页记忆：从侧边栏回到「训练器 / B 站下载」时，恢复上次停留的子页 */
  var SUB_PARENT = { run: "train", "train-cfg": "train", "bili-jobs": "tool-bili" };
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
    if (key === "debug") { edInit(); edBind(); edDraw(); }
    if (key === "dataset") { renderSteps(); syncDatasetBadge(); }
    if (key === "train") bindTrainerTop();
    if (key === "train-cfg") {
      var cb = document.getElementById("cfgBox");
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
    { n: "过滤器", d: "模糊 · 坏图 · 重复",
      modes: ["模糊筛选", "重复图", "坏图"],
      modeDesc: ["按清晰度阈值挑出糊片", "相似度聚类，每组留一张", "打不开或没检出人脸"],
      out: "" },
    { n: "排序器", d: "清晰度 · 角度 · 时间",
      modes: ["清晰度", "角度", "时间"],
      modeDesc: ["越清晰排越前", "按偏航角从小到大", "按拍摄时间先后"],
      out: "" },
    { n: "遮罩应用", d: "XSeg 推理 · 平滑",
      modes: ["XSeg 推理", "遮罩平滑"],
      modeDesc: ["用训练好的遮罩模型推理", "羽化边缘，去掉锯齿"],
      out: "" }
  ];

  /* 三段流水线同时展开：模块 → 模式 → 参数，各自独立成节 */
  var STEP_SEL = [0, 0, 0];      /* 每节选中的模式 */

  /* 每个模块 + 模式的参数组（"模块-模式" 索引） */
  var PARAM_MAP = {
    "0-0": [["强度档位", ["10%", "30%", "60%"]], ["处理方式", ["移到 _trash", "直接删除"]]],
    "0-1": [["相似阈值", ["90%", "95%", "98%"]], ["处理方式", ["移到 _trash", "直接删除"]]],
    "0-2": [["判定阈值", ["低", "中", "高"]], ["处理方式", ["移到 _trash", "直接删除"]]],
    "1-0": [["目标分辨率", ["256", "384", "512"]]],
    "1-1": [["分组数", ["8", "16", "32"]]],
    "1-2": [["分段数", ["10", "20", "40"]]],
    "2-0": [["模型", ["XSeg", "XSegLite"]], ["阈值", ["0.3", "0.5", "0.7"]]],
    "2-1": [["平滑强度", ["1", "2", "3"]]]
  };

  /* 用训练参数页那套：分组卡片 + 行（标签 / 说明 / 右侧控件） */
  function segHTML(groups) {
    return groups.map(function (g) { return g; });
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

    /* 行 1：模式选择 */
    var r1 = document.createElement("div");
    r1.className = "cfgrow";
    r1.innerHTML = '<span class="cl">模式</span><span class="cd">' + (m.modeDesc[k] || "") + '</span><span class="cc"></span>';
    r1.querySelector(".cc").appendChild(makeSeg(m.modes, k, function (idx) {
      STEP_SEL[i] = idx;
      var fresh = renderStation(i);
      card.replaceWith(fresh);
    }));
    card.appendChild(r1);

    /* 行 2..n：参数 */
    (PARAM_MAP[i + "-" + k] || PARAM_MAP[i + "-0"] || []).forEach(function (g) {
      var row = document.createElement("div");
      row.className = "cfgrow";
      row.innerHTML = '<span class="cl">' + g[0] + '</span><span class="cd"></span><span class="cc"></span>';
      row.querySelector(".cc").appendChild(makeSeg(g[1], 0));
      card.appendChild(row);
    });

    /* 结果区（只有第一站有真实数据） */
    if (i === 0) {
      var res = document.createElement("div");
      res.className = "dsres";
      res.innerHTML =
        '<div class="dsresnum" data-res="num">—</div>' +
        '<div class="dsreslbl" data-res="lbl">等待扫描</div>' +
        '<div class="dsresbar"><i data-res="a"></i></div>';
      card.appendChild(res);
    }

    /* 底部动作 */
    var acts = document.createElement("div");
    acts.className = "dsacts";
    acts.innerHTML = '<button class="btn"' + (i === 0 ? ' id="dsBtn"' : '') +
      ' data-step="' + i + '">开始处理</button>';
    acts.querySelector("button").addEventListener("click", function () {
      if (i === 0) { dsRun(); return; }
      toast(MODULES[i].n, "这一节的后端还没接", "info");
    });
    card.appendChild(acts);

    return card;
  }

  function renderSteps() {
    var box = $("dsSteps");
    if (!box) return;
    box.innerHTML = "";
    MODULES.forEach(function (m, i) { box.appendChild(renderStation(i)); });
    if (window.I18N) window.I18N.refresh();
    bindDropPaths();
    bindPickPaths();
    noSpell();
  }
  window.renderSteps = renderSteps;

  function runStep(i) {
    if (i !== 0) {
      toast(MODULES[i].n, "这一节的后端还没接", "info");
      return;
    }
    var k = STEP_SEL[0];
    var mode = k === 1 ? "dup" : (k === 2 ? "bad" : "blur");
    toast("开始扫描", MODULES[0].modes[k] + " · 正在统计…", "info");
    fetch("/api/dataset/scan", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ mode: mode, dir: "data_src", threshold: mode === "dup" ? 6 : 10 })
    }).then(function (r) { return r.json(); }).then(function (d) {
      toast("扫描完成", "共 " + d.total + " 张，命中 " + d.count + " 张", d.count ? "success" : "info");
      var sec0 = document.querySelectorAll("#dsSteps .step")[0];
      if (sec0) {
        var num = sec0.querySelector('[data-res="num"]');
        var lbl = sec0.querySelector('[data-res="lbl"]');
        var bar = sec0.querySelector('[data-res="a"]');
        if (num) num.textContent = d.count;
        if (lbl) lbl.textContent = "张将被移走 · 共 " + d.total + " 张";
        if (bar) bar.style.width = (d.total ? Math.max(2, d.count / d.total * 100) : 0) + "%";
      }
    }).catch(function (e) {
      toast("扫描失败", String((e && e.message) || e), "error");
    });
  }

  function dsModeKey() {
    var names = MODULES[0].modes[STEP_SEL[0]] || "";
    if (names.indexOf("重复") >= 0) return "dup";
    if (names.indexOf("坏图") >= 0) return "bad";
    return "blur";
  }

  /* 数据集处理：扫描 → 确认移走 → 撤销，三段式 */
  var DSB = { stage: 0, hits: [], last: [], dir: "data_src", mode: "blur" };

  function dsBtn() { return document.getElementById("dsBtn"); }
  function dsNum() { return document.querySelector('#dsSteps [data-res="num"]'); }
  function dsLbl() { return document.querySelector('#dsSteps [data-res="lbl"]'); }
  function dsBar() { return document.querySelector('#dsSteps [data-res="a"]'); }

  function dsPaint(num, lbl, pct) {
    var n = dsNum(), l = dsLbl(), b = dsBar();
    if (n && num != null) n.textContent = num;
    if (l && lbl != null) l.textContent = lbl;
    if (b && pct != null) b.style.width = Math.max(2, Math.min(100, pct)) + "%";
  }

  function dsRun() {
    var mode = dsModeKey();
    DSB.mode = mode;
    var btn = dsBtn();

    if (DSB.stage === 2) {                    /* 撤销 */
      fetch("/api/dataset/restore", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ dir: DSB.dir, items: DSB.last || [] })
      }).then(function (r) { return r.json(); }).then(function (d) {
        DSB.stage = 0; DSB.last = []; DSB.hits = [];
        if (btn) btn.textContent = "开始处理";
        dsPaint("—", "等待扫描", 0);
        var hdR = document.querySelector("#dsSteps .card .cardhd .hint");
        if (hdR) hdR.textContent = "未扫描";
        toast("已撤销", d.restored + " 张回到原目录", "success");
      }).catch(function (e) { toast("撤销失败", String(e.message || e), "error"); });
      return;
    }

    if (DSB.stage === 1) {                    /* 确认移走 */
      if (btn) { btn.textContent = "正在移动…"; btn.disabled = true; }
      fetch("/api/dataset/apply", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ mode: DSB.mode, dir: DSB.dir,
                               names: DSB.hits.map(function (x) { return x.name; }) })
      }).then(function (r) { return r.json(); }).then(function (d) {
        DSB.stage = 2;
        DSB.last = d.items || [];
        if (btn) { btn.textContent = "撤销（放回 " + d.moved + " 张）"; btn.disabled = false; }
        dsPaint(d.moved, "张已移入 _trash · 可撤销", 100);
        toast("处理完成", d.moved + " 张移入 _trash，可撤销", "success");
      }).catch(function (e) {
        if (btn) { btn.textContent = "开始处理"; btn.disabled = false; }
        toast("移动失败", String(e.message || e), "error");
      });
      return;
    }

    /* stage 0：扫描 */
    if (btn) { btn.textContent = "正在扫描…"; btn.disabled = true; }
    dsPaint(null, "正在统计…", 0);
    var t0 = Date.now();
    fetch("/api/dataset/scan", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ mode: mode, dir: DSB.dir, threshold: mode === "dup" ? 6 : 10 })
    }).then(function (r) { return r.json(); }).then(function (d) {
      DSB.hits = d.hits || [];
      var sec = (Date.now() - t0) / 1000;
      if (!DSB.hits.length) {
        DSB.stage = 0;
        if (btn) { btn.textContent = "重新扫描"; btn.disabled = false; }
        dsPaint(0, "共 " + d.total + " 张，没有命中 · " + sec.toFixed(1) + " 秒", 0);
        toast("扫描完成", "没有需要处理的图片", "info");
        return;
      }
      DSB.stage = 1;
      if (btn) { btn.textContent = "确认移走 " + DSB.hits.length + " 张"; btn.disabled = false; }
      dsPaint(d.count, "张命中 · 共 " + d.total + " 张 · " + sec.toFixed(1) + " 秒",
              d.total ? d.count / d.total * 100 : 0);
      var hd0 = document.querySelector("#dsSteps .card .cardhd .hint");
      if (hd0) hd0.textContent = "共 " + d.total + " 张 · 命中 " + d.count + " 张";
    }).catch(function (e) {
      if (btn) { btn.textContent = "开始处理"; btn.disabled = false; }
      toast("扫描失败", String(e.message || e), "error");
    });
  }


  var DS_OPEN_AT = 0;
  function dsOpen() {
    var now = Date.now();
    if (now - DS_OPEN_AT < 800) return;      /* 防误触 / 重复触发 */
    DS_OPEN_AT = now;
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

  var STAGE_NAME = ["抽帧", "人脸对齐", "补遮罩", "完成"];

  function setExtractUI(s) {
    if ($("exBar")) $("exBar").style.width = (s.pct || 0) + "%";
    /* 四个数分开算，别再混着叫"抽帧速度"：
         帧速度   = 解码速度（含被跳过的帧）
         人脸速度 = 每秒产出多少张对齐脸 = faces / 已用秒数
         已对齐   = 累计张数
         已处理帧 = done / total */
    var frameRate = s.fps || 0;
    var faceRate = (s.elapsed > 0.5 && s.faces) ? (s.faces / s.elapsed) : 0;
    if ($("exFps")) $("exFps").textContent = frameRate.toFixed(1);
    if ($("exFaceRate")) $("exFaceRate").textContent = faceRate.toFixed(1);
    if ($("exFaces")) $("exFaces").textContent = Math.round(s.faces || 0);
    if ($("exFiles")) $("exFiles").textContent = s.done || 0;
    if ($("exFilesTotal")) $("exFilesTotal").textContent = "/ " + (s.total || 0);
    if ($("exCounter")) {
      var txt = s.total ? (s.done + " / " + s.total + " · " + s.pct + "%") : "就绪";
      if (s.running && s.eta > 0) txt += " · 剩 " + fmtEta(s.eta);
      if (s.running && s.elapsed > 0) txt += " · 已用 " + fmtEta(s.elapsed);
      $("exCounter").textContent = txt;
    }
    var st = document.getElementById("exStage");
    if (st) st.textContent = s.running ? (STAGE_NAME[s.stage] || "") : "";
    /* 后端那句 msg 是最准的状态：抽帧 → 对齐 → 完成 */
    var msgEl = document.getElementById("exMsg");
    if (msgEl && s.msg) msgEl.textContent = s.msg;
  }

  var EX_DST = "";   /* 本次提取的真实输出目录（后端解析后的绝对路径） */

  function subscribe() {
    if (es) es.close();
    es = new EventSource("/api/extract/progress");
    es.onmessage = function (ev) {
      var s = JSON.parse(ev.data);
      setExtractUI(s);
      if (!s.running && s.pct >= 100) {
        es.close(); es = null;
        toast("提取完成",
              (s.faces ? (s.faces + " 张人脸") : ((s.total || 0) + " 个文件")) +
              (EX_DST ? " → " + EX_DST : ""));

  if (btnEx) { btnEx.disabled = false; btnEx.textContent = "开始提取"; }
        setTimeout(function () { showProgress(false); }, 900);
      }
    };
    es.onerror = function () { if (es) { es.close(); es = null; } };
  }

  if (btnEx) {
    btnEx.addEventListener("click", function () {
      if (btnEx.disabled) return;
      btnEx.disabled = true;
      btnEx.textContent = "正在提取…";
      showProgress(true);
      var inEl = document.querySelector("#p-extract .pathval");
      var ins = document.querySelectorAll("#p-extract .pathval");
      var src = ins[0] ? String(ins[0].value || "").trim() : "";
      var dst = ins[1] ? String(ins[1].value || "").trim() : "";
      /* 帧跳跃 N 表示每 N 帧取一张 —— 源视频 24 fps 时，fps 参数就是 24/N */
      var skipInp = document.querySelector("#p-extract .num input, #p-extract input.num");
      var skip = skipInp ? Math.max(1, parseInt(skipInp.value, 10) || 1) : 1;
      /* 不再假设源是 24fps：把"每 N 帧取 1"直接交给后端，由它按源视频真实帧率换算 */

      /* 质量档 / 检测角度 / 帧跳跃：三个都是真参数，一起发给后端 */
      var qSeg = document.querySelector("#p-extract .prow.compact .pc:nth-child(1) .seg");
      var qOn = qSeg ? qSeg.querySelector("b.on") : null;
      /* 检测器：RetinaFace(det_10g) 准，TinyMog 快。质量档不再影响分辨率。 */
      var quality = qOn && /TinyMog/i.test(qOn.textContent) ? "fast" : "high";

      var aSeg = document.querySelector("#p-extract .prow.compact .pc:nth-child(2) .seg");
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

  function modelCard(m) {
    var d = document.createElement("div");
    d.className = "mcard";
    var cls = (m.state === "训练中") ? "ok" : "n";
    var meta = (m.arch || "—") + (m.iter ? " · iter " + fmt(m.iter) : "");
    d.innerHTML =
      '<div class="nm">' + m.name + ' <span class="tag ' + cls + '">' + m.state + '</span></div>' +
      '<div class="meta">' + meta + '</div>' +
      '<div class="kv"><span class="tag n">' + m.export + '</span></div>' +
      '<div class="acts"><button class="mini pri">训练</button>' +
      '<button class="mini">参数</button><button class="mini">删除</button></div>';
    return d;
  }

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
    if (m.has_dfm) kv.push('<span class="tag ok">DFM</span>');

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
        '<div class="acts"><button class="mini pri" data-act="run">训练</button>' +
        '<button class="mini" data-act="cfg">参数</button>' +
        '<button class="mini" data-act="export">导出</button>' +
        '<button class="mini" data-act="del">删除</button></div>' +
      "</div>";
    var runBtn = el.querySelector('[data-act="run"]');
    if (runBtn) {
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

  /* ---------- 合成工作室：真实视频预览（后端 /api/merge/*） ---------- */
  /* 所有状态都挂在 window 上：之前用模块内 var，出现过「传 300 却按 0 取帧」的诡异串台 */
  window.__mv = { path: "", frames: 0, fps: 30, opening: false, log: [] };

  window.mvStage = function (dataUrl) {
    var st = document.getElementById("mgStage");
    if (!st) return false;
    var img = st.querySelector("img");
    if (!img) {
      img = document.createElement("img");
      img.alt = "";
      st.appendChild(img);
    }
    img.onload = function () {
      /* 只看「检测」视图时才补画，别的视图不该出现框 */
      var onDet = document.querySelector('#mgViews b[data-v="det"].on');
      if (onDet) window.mvRedrawBoxes();
    };
    img.src = dataUrl;
    var hint = document.getElementById("mgStageHint");
    if (hint) hint.style.display = "none";
    return true;
  };

  window.mvMerged = function (n) {
    var st = window.__mv;
    n = parseInt(n, 10) || 0;
    fetch("/api/merge/merged/" + n + "?w=900").then(function (r) {
      return r.json().then(function (d) {
        if (!r.ok) throw new Error(d.detail || "取不到成片");
        return d;
      });
    }).then(function (d) {
      if (d.png) window.mvStage(d.png);
    }).catch(function (e) {
      toast("成片预览", String((e && e.message) || e).slice(0, 90), "info");
      var hint = document.getElementById("mgStageHint");
      if (hint) { hint.style.display = ""; hint.textContent = "还没有成片 —— 先导出一次"; }
    });
  };

  window.mvFrame = function (n) {
    var st = window.__mv;
    n = parseInt(n, 10) || 0;
    if (!st.path) { st.log.push("no-path " + n); return; }
    fetch("/api/merge/frame/" + n + "?w=900").then(function (r) { return r.json(); })
      .then(function (d) {
        st.log.push("req " + n + " -> " + (d.png ? d.png.length : "none") + " (idx " + d.idx + ")");
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
      if (window.MergeTimeline && window.MergeTimeline.init) {
        try { window.MergeTimeline.init(d.frames, d.fps || 30); } catch (e) {}
      }
      window.mvFrame(0);
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
      if (window.MergeTimeline && window.MergeTimeline.init) {
        try { window.MergeTimeline.init(d2.frames, 30); } catch (e) {}
      }
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
  window.__mergePrep = { running: false, project: "", timer: null };

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
  }
  window.mgShowPrepared = mgShowPrepared;

  window.mgOpenPrepared = function (project, frames, aligned) {
    var st = window.__mv;
    st.path = project || "project";
    st.frames = frames || 0;
    window.__mergeInputDir = project || "";
    fetch("/api/merge/info").then(function (r) { return r.json(); }).then(function (d) {
      st.frames = d.frames || frames || 0;
      st.fps = d.fps || 30;
      if (window.MergeTimeline && window.MergeTimeline.init) {
        try { window.MergeTimeline.init(st.frames, st.fps); } catch (e) {}
      }
      var tot = document.getElementById("mtlTotal");
      if (tot) tot.textContent = st.frames;
      mgShowPrepared(project, st.frames);
      window.mvFrame(0);
      if (window.mvAnalyze) window.mvAnalyze(24);
      toast("素材已就绪", st.frames + " 帧 · " + (aligned || 0) + " 张对齐脸", "success");
    }).catch(function () {
      window.mvFrame(0);
      toast("素材已就绪", (frames || 0) + " 帧", "success");
    });
  };

  window.mgPrepare = function (video) {
    if (!video) return;
    var st = window.__mergePrep;
    if (st.running) { toast("正在准备", "等这一段处理完再来", "info"); return; }
    st.running = true;
    st.project = "";
    window.__mergeInputDir = "";
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
          }).catch(function () {});
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
        if (empty) { empty.style.display = ""; }
        return;
      }
      if (empty) empty.style.display = "none";
      list.slice(0, 60).forEach(function (v) {
        var row = document.createElement("div");
        row.className = "mgitem";
        row.dataset.path = v.path;
        row.innerHTML = '<span class="nm">' + v.name + '</span><span class="tag">' + v.mb + " MB</span>";
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

  window.mvClearBoxes = function () {
    window.__mvBoxes = null;
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

    ctx.lineWidth = 2;
    ctx.strokeStyle = (getComputedStyle(document.documentElement).getPropertyValue("--ink-2") || "#17171a").trim();
    ctx.fillStyle = "rgba(128,128,140,.14)";
    boxes.forEach(function (b) {
      var x1 = b.box[0], y1 = b.box[1], x2 = b.box[2], y2 = b.box[3];
      var rx = ox + x1 * s, ry = oy + y1 * s;
      var rw = (x2 - x1) * s, rh = (y2 - y1) * s;
      ctx.fillRect(rx, ry, rw, rh);
      ctx.strokeRect(rx, ry, rw, rh);
      if (b.score != null) {
        var tag = Math.round(b.score * 100) + "%";
        ctx.font = "10px ui-monospace, monospace";
        var tw = ctx.measureText(tag).width + 6;
        ctx.fillStyle = "rgba(128,128,140,.92)";
        ctx.fillRect(rx, Math.max(0, ry - 14), tw, 13);
        ctx.fillStyle = "#fff";
        ctx.fillText(tag, rx + 3, Math.max(10, ry - 4));
        ctx.fillStyle = "rgba(128,128,140,.14)";
      }
    });
  };

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
    return (sel && sel.value) || "det_500m";
  }
  window.mgDetectorName = mgDetectorName;

  window.mvDetect = function (n) {
    var mv = window.__mv || {};
    if (!mv.path) return;
    fetch("/api/merge/detect/" + n + "?model=" + encodeURIComponent(mgDetectorName()) + "&thresh=0.5")
      .then(function (r) { return r.json(); })
      .then(function (d) {
        if (d.boxes) window.mvDrawBoxes(d.boxes, d.width, d.height);
        else if (d.detail) toast("检测失败", String(d.detail).slice(0, 90), "error");
      })
      .catch(function () {});
  };


  window.MergeVideo = { list: window.mvList, open: window.mvOpen, frame: window.mvFrame, state: window.__mv };

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

    /* 时间轴：每次进合成页都重绘一次（元素可能被重建），幂等 */
    setTimeout(function () {
      window.mvList();
      var mv = window.__mv || {};
      if (window.MergeTimeline && window.MergeTimeline.init) {
        try {
          window.MergeTimeline.init(mv.path ? mv.frames : 300, mv.fps || 30);
        } catch (e) { console.warn("timeline init", e); }
        /* 内容不足一屏时把栈拉满，刻度尺才不会只铺一半 */
        var st = document.querySelector(".mtl-stack");
        var sc = document.querySelector(".mtl-scroll");
        if (st && sc) st.style.width = Math.max(st.scrollWidth, sc.clientWidth) + "px";
      }
    }, 60);

    if (page.dataset.bound) return;
    page.dataset.bound = "1";
    page.querySelectorAll(".seg").forEach(function (seg) {
      seg.addEventListener("click", function (e) {
        var b = e.target.closest("b");
        if (!b || !seg.contains(b)) return;
        Array.prototype.forEach.call(seg.children, function (x) { x.classList.remove("on"); });
        b.classList.add("on");
      });
    });
    var views = $("mgViews");
    if (views) {
      views.addEventListener("click", function (e) {
        var b = e.target.closest("b[data-v]");
        if (!b) return;
        Array.prototype.forEach.call(views.children, function (x) { x.classList.remove("on"); });
        b.classList.add("on");
        var view = b.dataset.v;
        var f = document.getElementById("mtlFrame");
        var idx = parseInt(f ? f.value : 0, 10) || 0;
        if (view === "det") {
          window.mvDetect(idx);
        } else if (view === "swap") {
          window.mvMerged(idx);
        } else {
          window.mvClearBoxes();
          window.mvFrame(idx);
        }
      });
    }

    /* 导出流程：工作台（预览/剪辑）→ 人脸库（分组）→ 导出设置 → 导出中 */
    var MG_STEPS = ["main", "facedb", "settings"];
    function mgStep(name) {
      var step = document.getElementById("mgStep");
      var back = document.getElementById("mgBack");
      page.querySelectorAll(".mgview").forEach(function (v) {
        v.classList.toggle("in", v.dataset.view === name);
      });
      if (step) {
        var label = { facedb: "人脸库", settings: "导出设置" }[name] || "";
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
      CRUMBS.merge = trail;
      renderCrumbs("merge");
      window.__mergeStep = name;
      if (window.SUB_MEMORY) window.SUB_MEMORY["merge"] = null;   /* 合成器用 mgStep，不走 go */
      if (name !== "main" && window.MergePages) window.MergePages.show(name);
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
        var idx = MG_STEPS.indexOf(at);
        mgStep(idx <= 1 ? "main" : "facedb");
      });
    }

    var exBtn = $("mgExport");
    if (exBtn && !exBtn.dataset.bound) {
      exBtn.dataset.bound = "1";
      exBtn.addEventListener("click", function () {
        /* 人脸库（分组/去重）还没做，先不挡路：直接进导出设置。
           合成本身不需要人脸库 —— MergeStudio 靠 DFLJPG 的元数据定位人脸。 */
        if (!window.__mergeInputDir) {
          toast("还没有素材", "先导入一段视频（会自动抽帧 + 对齐）", "warn");
          return;
        }
        if (window.mgStep) window.mgStep("settings");
      });
    }

    /* 导出进度的演示：节奏参照真实 ffmpeg 抽帧 + ORT 推理，只为把界面跑起来看 */
    var MGP_TIMER = null;
    function mgDemoProgress(total) {
      /* 真跑合成的时候别让演示进度盖掉真实数据 */
      if (window.__mergeExporting) return;
      total = total || 486;
      var fill = $("mgpFill"), pct = $("mgpPct"), fr = $("mgpFrames"), eta = $("mgpEta");
      var steps = document.querySelectorAll("#p-merge .mgpstep");
      var cur = 0, stage = 0;
      var names = ["detect", "swap", "encode", "mux"];
      if (MGP_TIMER) clearInterval(MGP_TIMER);
      function paint() {
        var p = Math.min(100, Math.round((cur / total) * 100));
        if (fill) fill.style.width = p + "%";
        if (pct) pct.textContent = p + "%";
        if (fr) fr.textContent = cur + " / " + total + " 帧";
        var left = Math.max(0, total - cur);
        if (eta) eta.textContent = left ? ("剩余 " + Math.ceil(left / 24) + " 秒") : "完成";
        stage = p < 34 ? 0 : p < 78 ? 1 : p < 96 ? 2 : 3;
        for (var i = 0; i < steps.length; i++) {
          steps[i].classList.toggle("on", i === stage);
          steps[i].classList.toggle("done", i < stage);
          var s = steps[i].querySelector("span");
          if (s && i === 0) s.textContent = Math.round(cur * 0.62) + " 张";
          if (s && i === 1 && stage >= 1) s.textContent = "overlay · rct";
          if (s && i === 2 && stage >= 2) s.textContent = "h264_nvenc";
          if (s && i === 3 && stage >= 3) s.textContent = "mp4";
        }
      }
      paint();
      MGP_TIMER = setInterval(function () {
        cur = Math.min(total, cur + Math.round(total / 60));
        paint();
        if (cur >= total) { clearInterval(MGP_TIMER); MGP_TIMER = null; }
      }, 160);
    }
    window.mgDemoProgress = mgDemoProgress;

    var cancelBtn = $("mgpCancel");
    if (cancelBtn && !cancelBtn.dataset.bound) {
      cancelBtn.dataset.bound = "1";
      cancelBtn.addEventListener("click", function () {
        if (!window.__mergeExporting) { if (window.mgStep) window.mgStep("settings"); return; }
        cancelBtn.disabled = true;
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
    /* 真跑一次合成。工作台里的「导出」会先走到导出设置页，
       两个「开始导出」（#mgStartExport2 在工作台、#mgStartExport 在子页）共用这一份逻辑。 */
    function mgRunExport(payload) {
      payload = payload || {};
      document.dispatchEvent(new CustomEvent("merge:export-start", { detail: payload }));

      var inputDir = window.__mergeInputDir || "";
      if (!inputDir) {
        toast("没有素材", "先在工作台导入一段视频 —— 会自动抽帧 + 对齐", "warn");
        return;
      }
      if (window.__mergeExporting) {
        toast("正在合成", "上一次还没结束", "info");
        return;
      }
      window.__mergeExporting = true;

      var body = {
        input_dir: inputDir,
        project: inputDir,
        model: (document.querySelector("#mgModels .mgitem.on .nm") || {}).textContent || "",
        workers: parseInt(payload.num_workers, 10) || 0,
        encoder: payload.encoder || "h264_nvenc",
        hwaccel: payload.hwaccel === "" ? "off" : (payload.hwaccel || "auto"),
        settings: {
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
        }
      };
      if (payload.output_path && /^[A-Za-z]:[\\/]/.test(payload.output_path)) {
        body.output_dir = payload.output_path;
      }

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

    /* 导出设置子页里的「开始导出」：它以前只 dispatch 一个没人听的事件 */
    document.addEventListener("click", function (e) {
      var t = e.target && e.target.closest ? e.target.closest("#mgStartExport") : null;
      if (!t) return;
      e.preventDefault();
      mgRunExport((window.MergePages && window.MergePages.settings) ? window.MergePages.settings() : {});
    });

    var tlBox = $("mergeTracks");
    if (tlBox && !tlBox.dataset.frameBound) {
      tlBox.dataset.frameBound = "1";
      tlBox.addEventListener("mtl:frame", function (e) {
        var n = (e.detail && e.detail.frame) || 0;
        var onSwap = document.querySelector('#mgViews b[data-v="swap"].on');
        if (onSwap) window.mvMerged(n);
        else window.mvFrame(n);
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
              row.innerHTML = '<span class="nm">' + v.name + '</span><span class="tag">' + v.mb + " MB</span>";
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
    MERGE_POLL = setInterval(function () {
      fetch("/api/merge/export/status").then(function (r) { return r.json(); }).then(function (d) {
        var fill = document.getElementById("mgpFill");
        var pctEl = document.getElementById("mgpPct");
        var frEl = document.getElementById("mgpFrames");
        var msg = document.getElementById("mgpEta");
        var last = (d.log && d.log.length) ? d.log[d.log.length - 1] : "";
        /* 后端上报的是真实阶段进度（pct / msg / 帧数），进度页直接照它画 */
        var text = d.msg || last;
        var p = Math.max(0, Math.min(100, Number(d.pct) || 0));
        if (msg) msg.textContent = text ? String(text).slice(0, 60) : "合成中…";
        var fileEl = document.getElementById("mgpFile");
        if (fileEl && d.out && String(d.out) !== "—") {
          fileEl.textContent = String(d.out).split(/[\\/]/).pop();
        }
        if (fill) fill.style.width = (d.running ? Math.max(2, Math.min(99, p)) : 100) + "%";
        if (pctEl) pctEl.textContent = (d.running ? p : 100) + "%";

        /* 阶段直接用后端的 @@P stage：0 搭目录/载模型 · 3 换脸 · 5 合成 · 6 编码 */
        var srv = Number(d.stage);
        var stage;
        if (!isNaN(srv) && srv >= 0) {
          stage = srv >= 6 ? 2 : (srv >= 3 ? 1 : 0);
        } else {
          stage = p < 20 ? 0 : p < 92 ? 1 : p < 100 ? 2 : 3;
        }
        if (!d.running) stage = d.ok ? 3 : stage;
        var done = /Merging frames (\d+)\/(\d+)/.exec(text || "");
        if (frEl) {
          if (done) frEl.textContent = done[1] + " / " + done[2] + " 帧";
          else if (!d.running && d.ok) frEl.textContent = "完成";
        }
        var steps = document.querySelectorAll("#p-merge .mgpstep");
        for (var i = 0; i < steps.length; i++) {
          steps[i].classList.toggle("on", d.running && i === stage);
          steps[i].classList.toggle("done", (d.running && i < stage) || (!d.running && d.ok));
        }
        var stepInfo = [null, done ? (done[1] + " / " + done[2]) : null,
                        d.running && stage === 2 ? "h264" : null,
                        (!d.running && d.ok) ? "mp4" : null];
        for (var k = 0; k < steps.length; k++) {
          var sp = steps[k].querySelector("span");
          if (sp && stepInfo[k]) sp.textContent = stepInfo[k];
        }

        if (!d.running) {
          clearInterval(MERGE_POLL); MERGE_POLL = null;
          window.__mergeExporting = false;
          toast(d.ok ? "合成完成" : "合成失败",
                d.ok ? ("输出 " + d.out) : String(d.error || "").slice(0, 120),
                d.ok ? "success" : "error");
          if (d.ok) {
            /* 导出完自动切回工作台并显示成片那一帧，省得用户自己找 */
            try {
              if (window.mgStep) window.mgStep("main");
              var sw = document.querySelector('#mgViews b[data-v="swap"]');
              if (sw) sw.click();
            } catch (e) {}
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
    box.innerHTML = "";
    if (!list.length) {
      box.innerHTML = '<div class="empty-note">还没有可导出的模型</div>';
      return;
    }
    var cnt = el("mgModelCount");
    if (cnt) cnt.textContent = list.length;
    list.forEach(function (m, i) {
      var r = document.createElement("div");
      r.className = "mgitem" + (i === 0 ? " on" : "");
      r.innerHTML = '<span class="nm">' + m.name + '</span>' +
        (m.has_export
          ? '<span class="tag ok">DFM</span>'
          : '<button class="mini mgexp" title="导出为 .dfm">导出</button>');
      if (!m.has_export) {
        var expBtn = r.querySelector(".mgexp");
        if (expBtn) expBtn.addEventListener("click", function (e) {
          e.stopPropagation();
          exportDfm(m.name, expBtn);
        });
      }
      r.addEventListener("click", function () {
        Array.prototype.forEach.call(box.children, function (x) { x.classList.remove("on"); });
        r.classList.add("on");
      });
      box.appendChild(r);
    });
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
    var el = document.getElementById("dsPathCount");
    if (!el) return;
    var d = (STATE && STATE.datasets && STATE.datasets.src) || null;
    if (d && d.count != null) el.textContent = d.count + " 张";
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
      fetch("/api/state").then(function (r) { return r.json(); }).then(function (s) {
        var c = (s.datasets && s.datasets.src && s.datasets.src.count) || 0;
        if ($("dsPathCount")) $("dsPathCount").textContent = c + " 张";
      }).catch(function () {});
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

  function applyAuth(a) {
    AUTH = a || AUTH;
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
        $("uAvatar").textContent = nm.slice(0, 1).toUpperCase();
      }
    }
    var w = $("welcome");
    if (w) {
      var show = !AUTH.onboarded;
      w.style.display = show ? "grid" : "none";
      if (show && !BILI.key) setTimeout(biliQr, 120);
    }
    if ($("hello") && AUTH.logged_in && AUTH.name) {
      $("hello").textContent = greeting() + "，" + AUTH.name;
    }
  }

  /* ---------- B 站扫码登录 ---------- */
  var BILI = { timer: 0, key: "", busy: false };

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
      d.className = "job" + (j.state === "done" ? " done" : "");
      d.innerHTML =
        '<i class="jobdot"></i>' +
        '<span class="jobname">' + (j.title || j.bvid || "任务") + "</span>" +
        '<span class="jobkind">' + (j.kind || "B 站下载") + "</span>" +
        (j.state === "done"
          ? '<span class="jobstate done">已完成</span>'
          : (j.state === "queued"
            ? '<span class="jobstate">排队中</span>'
            : '<span class="jobbar"><i style="width:' + Math.round(j.pct || 0) + '%"></i></span>' +
              '<span class="jobpct">' + Math.round(j.pct || 0) + "%</span>" +
              '<span class="jobstate">' + (j.speed || "0.0 MB/s") + "</span>"));
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
        "<td>" + (v.title || "—") + "</td>" +
        '<td class="mono">' + (v.bvid || "—") + "</td>" +
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

  /* ---------- 模拟：分段拉取（B 站接口一次只给 20 条，且要限流） ---------- */
  var BILI_PAGE = 20;
  var BILI_TIMER = null;
  var BILI_FAKE = [
    "【4K】深夜书房的一次实验", "第一次尝试这种转场，效果意外地好",
    "帮我看看这段打光有没有问题", "练习：三分钟讲完一个故事",
    "棚拍花絮 · 那些没剪进正片的镜头", "用一盏灯拍完整支片子",
    "夏日限定 · 海边随手拍", "重制版 · 三年前的作品",
    "这期的调色我改了七遍", "街头采访第 12 期",
    "器材开箱 · 这次真的买贵了", "幕后 · 一场雨戏怎么拍",
    "五分钟看完整个项目", "关于构图，我想说的三件事",
    "第一次用这种机位，有点手抖", "夜景实拍 · 高感表现",
    "剪辑复盘 · 为什么删掉了这段", "合作投稿 · 来自观众的素材",
    "幕后花絮 · 拍到凌晨三点", "年度总结 · 这一年的 20 个片子"
  ];

  function biliMockFetch(uid, total) {
    if (BILI_TIMER) { clearInterval(BILI_TIMER); BILI_TIMER = null; }
    var bar = $("biliFetchBar"), fill = $("biliFetchFill"), txt = $("biliFetchText");
    var btn = $("biliFetch"), owner = $("biliOwner");
    total = total || (60 + Math.floor(Math.random() * 100));
    var got = 0, batch = 0, pages = Math.ceil(total / BILI_PAGE);
    BILI_VIDEOS.length = 0;
    renderBili([]);
    if (bar) bar.hidden = false;
    if (btn) { btn.disabled = true; btn.textContent = "获取中…"; }
    if (owner) owner.textContent = "UID " + uid;

    function step() {
      batch++;
      var take = Math.min(BILI_PAGE, total - got);
      for (var i = 0; i < take; i++) {
        var n = got + i + 1;
        BILI_VIDEOS.push({
          bvid: "BV1" + String(Math.abs((uid * 7919 + n * 104729) % 1e9)).padStart(9, "0").slice(0, 9),
          title: (BILI_FAKE[(n - 1) % BILI_FAKE.length] || ("视频 " + n)) + " · P" + n,
          coop: n % 7 === 0,
          checked: false
        });
      }
      got += take;
      renderBili();
      var pct = Math.round((got / total) * 100);
      if (fill) fill.style.width = pct + "%";
      if (txt) txt.textContent = "第 " + batch + " / " + pages + " 批 · 已取 " + got + " / " + total;
      if (got >= total) {
        clearInterval(BILI_TIMER); BILI_TIMER = null;
        if (bar) setTimeout(function () { bar.hidden = true; }, 900);
        if (btn) { btn.disabled = false; btn.textContent = "获取视频列表"; }
        if (fill) fill.style.width = "0%";
        toast("获取完成", "UID " + uid + " · 共 " + total + " 个视频 · " + pages + " 批", "success");
      }
    }
    step();
    BILI_TIMER = setInterval(step, 850);
  }
  window.biliMockFetch = biliMockFetch;

  /* ---------- 模拟：下载推进 ---------- */
  var DL_TIMER = null;
  var DL_PAUSED = false;
  var DL_THREADS = 4;
  function biliConcurrency() { return DL_THREADS; }
  function biliSetThreads(n) { DL_THREADS = n; }
  window.biliSetThreads = biliSetThreads;
  function biliStopAll() {
    if (DL_TIMER) { clearInterval(DL_TIMER); DL_TIMER = null; }
    DL_PAUSED = false;
    BILI_JOBS.length = 0;
    renderProcs();
  }
  window.biliStopAll = biliStopAll;
  function biliTogglePause() {
    DL_PAUSED = !DL_PAUSED;
    return DL_PAUSED;
  }
  window.biliTogglePause = biliTogglePause;

  var BILI_QUALITY = "best";
  function biliSetQuality(q) { BILI_QUALITY = q || "best"; }
  window.biliSetQuality = biliSetQuality;

  function biliMockDownload(picked) {
    /* 画质决定体积和速度：最高 40-300MB，1080P 25-120MB，720P 12-60MB */
    var qr = BILI_QUALITY === "best" ? [40, 300]
           : BILI_QUALITY === "1080" ? [25, 120] : [12, 60];
    var spd = BILI_QUALITY === "best" ? 1.0 : BILI_QUALITY === "1080" ? 1.3 : 1.7;
    picked.forEach(function (v, i) {
      /* 先落盘（后端用 ffmpeg 生成等价文件），再按返回的真实大小推进度 */
      fetch("/api/bili/download", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ bvid: v.bvid, title: v.title, quality: BILI_QUALITY })
      }).then(function (r) { return r.json(); }).then(function (d) {
        if (!d.ok) return;
        var job = BILI_JOBS.filter(function (j) { return j.bvid === v.bvid; })[0];
        if (job) {
          job.path = d.path;
          job.size = Math.round((d.size || 0) / 1048576 * 10) / 10;
        }
      }).catch(function () {});
      BILI_JOBS.push({
        bvid: v.bvid, title: v.title, kind: "B 站下载",
        state: i < DL_THREADS ? "downloading" : "queued",
        speed: "0.0 MB/s", pct: 0,
        size: qr[0] + Math.floor(Math.random() * (qr[1] - qr[0])),
        q: BILI_QUALITY, spd: spd
      });
    });
    renderProcs();
    if (DL_TIMER) return;
    DL_TIMER = setInterval(function () {
      if (DL_PAUSED) return;
      var active = BILI_JOBS.filter(function (j) { return j.state === "downloading"; });
      if (!active.length) {
        var queued = BILI_JOBS.filter(function (j) { return j.state === "queued"; });
        if (!queued.length) { clearInterval(DL_TIMER); DL_TIMER = null; return; }
        queued.slice(0, DL_THREADS).forEach(function (j) { j.state = "downloading"; });
        active = BILI_JOBS.filter(function (j) { return j.state === "downloading"; });
      }
      active.forEach(function (j) {
        j.pct = Math.min(100, (j.pct || 0) + (1.2 + Math.random() * 5) * (j.spd || 1));
        j.speed = ((2 + Math.random() * 7) * (j.spd || 1)).toFixed(1) + " MB/s";
        if (j.pct >= 100) { j.pct = 100; j.state = "done"; j.speed = ""; }
      });
      renderProcs();
      if (BILI_JOBS.every(function (j) { return j.state === "done"; })) {
        clearInterval(DL_TIMER); DL_TIMER = null;
        toast("下载完成", "全部 " + BILI_JOBS.length + " 个视频已下载", "success");
      }
    }, 400);
  }
  window.biliMockDownload = biliMockDownload;

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
        biliMockFetch(uid);
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
        toast("开始下载", "已选 " + picked.length + " 个视频", "info");
        if (window.biliMockDownload) window.biliMockDownload(picked);
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
      { k: "face_type", l: "人脸类型", t: "sel", o: ["whole_face", "full_face", "half_face", "midfull_face", "head"], v: "whole_face" }
    ]},
    { t: "训练参数", rows: [
      { k: "batch_size", l: "批次大小", t: "num", v: "8", d: "每次训练的样本数量" },
      { k: "target_iter", l: "目标迭代数", t: "num", v: "0", d: "0 表示无限训练" },
      { k: "autobackup_hour", l: "自动保存间隔", t: "num", v: "1", d: "4.0 无此项，仅为 UI 占位" },
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
      { k: "max_backups", l: "最大备份数", t: "num", v: "3" }
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

  function openCfg(m) {
    var box = $("cfgBox");
    if (!box) return;
    window.__cfgModel = m || window.__cfgModel || null;   /* 冻结层过滤要知道当前架构 */
    window.__cfgArchi = (m && m.archi) || window.__cfgArchi || "";

    /* 补充参数不再塞进 CFG_GROUPS：TrainExtra.mount() 会把它们并进同名卡片，
       这里只管渲染原有的分组，避免同一批参数出现两遍。 */

    box.innerHTML = "";

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
        row.className = "cfgrow" + (g.ro ? " ro" : "");

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
          cc.appendChild(inp);
        } else if (r.t === "seg") {
          var seg = document.createElement("div");
          seg.className = "seg";
          if (r.k) seg.dataset.cfg = r.k;          /* 之前漏了这行：seg 既不被回填也不进 patch */
          r.o.forEach(function (o, i) {
            var b = document.createElement("b");
            b.textContent = o;
            b.dataset.v = o;                       /* 业务值取这个，别取 textContent（会被翻译） */
            if (i === 0) b.className = "on";
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
        } else if (r.t === "sw") {
          var sw = document.createElement("div");
          sw.className = "sw" + (r.v ? " on" : "");
          if (r.k) sw.dataset.cfg = r.k;
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

    /* 拉真实的 *_data.dat options 回填 —— 页面上的值不再是写死的默认值 */
    if (m) {
      fetch("/api/models/" + encodeURIComponent(m.name) + "/config")
        .then(function (r) { return r.ok ? r.json() : null; })
        .then(function (d) {
          if (!d || !d.options) return;
          var opts = d.options;
          var filled = 0;
          box.querySelectorAll("[data-cfg]").forEach(function (el) {
            var k = el.dataset.cfg, v = opts[k];
            if (v === undefined) return;
            filled++;
            if (el.tagName === "INPUT") el.value = v;
            else if (el.tagName === "SELECT") {
              /* 4.0 的 *_data.dat 里 face_type 存的是短码（wf/hf/ff），
                 而下拉选项是全名（whole_face/…）。直接 el.value = "wf" 匹配不上，
                 浏览器把 selectedIndex 置成 -1 —— 下拉框变成空白，提交时传空串。
                 这里补一层短码→全名的映射，再兜底选第一项。 */
              var _ALIAS = { wf: "whole_face", ff: "full_face", hf: "half_face",
                             mf: "midfull_face", head: "head" };
              var sv = String(v);
              el.value = sv;
              if (el.selectedIndex < 0 && _ALIAS[sv.toLowerCase()]) el.value = _ALIAS[sv.toLowerCase()];
              if (el.selectedIndex < 0) el.selectedIndex = 0;
            }
            else if (el.classList.contains("sw")) el.classList.toggle("on", !!v);
            else if (el.classList.contains("seg")) {
              /* 按 data-v 比，不看 textContent —— 切到英文后文案变了，比文本永远不匹配 */
              var hit = false;
              Array.prototype.forEach.call(el.children, function (b) {
                var same = String(b.dataset.v !== undefined ? b.dataset.v : b.textContent) === String(v);
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
          var hint = document.getElementById("cfgHint");
          if (hint) {
            /* 先记下原始文案：openCfg 每重画一次都会拉一遍配置，直接往后追加会叠成
               "已读入 46 项真实参数 · 已读入 46 项真实参数"。 */
            if (!hint.dataset.base) {
              hint.dataset.base = (hint.textContent || "").replace(/\s*·\s*已读入 \d+ 项真实参数\s*$/, "");
            }
            hint.textContent = hint.dataset.base + " · 已读入 " + filled + " 项真实参数";
          }
        })
        .catch(function (e) { console.warn("config load failed", e); });
    }

    /* 补充参数并由 TrainExtra.mount() 并入同名卡片。
       以前是靠包装 window.openCfg 来实现的，但那份包装没生效（__trainExtra 一直是 false），
       于是每次重渲染都把补充项弄丢。直接在这里调，顺序上不可能错。 */
    if (window.TrainExtra && typeof window.TrainExtra.mount === "function") {
      try { window.TrainExtra.mount(); } catch (e) {}
    }
  }
  window.openCfg = openCfg;

  /* ---------- 训练页控制按钮：暂停 / 继续 · 立即保存 · 停止 ---------- */
  function trainCtl(path, okMsg) {
    return fetch(path, { method: "POST" })
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
  window.trainState = trainState;

  function pollTrain(m) {
    /* 调用点有时不给 m（例如「启动成功后 pollTrain()」），那样右侧统计就永远不刷新。
       记住最近一次训练界面用的模型，后续轮询都拿它。 */
    if (m && m.name) window.__runModel = m;
    var rm = (m && m.name) ? m : window.__runModel;
    if (TRAIN_POLL) clearInterval(TRAIN_POLL);
    function tick() {
      fetch("/api/train/status").then(function (r) { return r.json(); }).then(function (d) {
        window.__lastTrain = d;
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
        }
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
          $("runTarget").textContent = tg ? Number(tg).toLocaleString("en-US") : "不限";
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
    if (crop) {
      /* 没在训练、也没有实时图时，用模型里存的那张网格图裁一格当「上次的样子」 */
      var useCrop = !src && idx === 0 && st.static;
      crop.style.display = useCrop ? "" : "none";
      if (useCrop) {
        crop.style.backgroundImage = "url('" + st.static + "')";
      }
    }
    if (img) {
      if (src) {
        img.src = src;
        img.style.display = "";
        if (empty) empty.style.display = "none";
      } else {
        img.removeAttribute("src");
        img.style.display = "none";
        var hasCrop = !!(idx === 0 && st.static);
        if (empty) {
          empty.style.display = hasCrop ? "none" : "";
          empty.textContent = "这一类还没有图 —— 点「刷新」拉一张";
        }
      }
    }
    var hint = document.getElementById("prevHint");
    if (hint) {
      var bits = [];
      var cropped = !(st.imgs && st.imgs[idx]) && idx === 0 && st.static;
      if (cropped) {
        bits.push("上次保存的样本");
        if (st.modelIter) bits.push(_num(st.modelIter) + " iter");
      }
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
    if ($("runTarget")) $("runTarget").textContent = "不限";
    if ($("runSpeed")) $("runSpeed").textContent = "测速中…";
    if ($("runElapsed")) $("runElapsed").textContent = "—";
    if ($("runEta")) { $("runEta").textContent = ""; $("runEta").style.display = "none"; }
    if ($("runEtaDot")) $("runEtaDot").style.display = "none";
    if ($("runLossSrc")) $("runLossSrc").textContent = "—";
    if ($("runLossDst")) $("runLossDst").textContent = "—";
    window.__lastTrain = null;
  }
  window.trainStartingUi = trainStartingUi;

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
    window.__pv = { imgs: [], labels: [], iter: 0, at: 0, idx: 0,
                    static: (m && m.preview_url) || "",
                    bs: (m && m.batch_size) || 4,
                    modelIter: (m && m.iter) || 0 };
    /* 没在训练时，运行条读模型里保存的值 —— 显示 0 iter / 一堆破折号看着像没加载出来 */
    if (m && m.iter && !(window.__lastTrain && window.__lastTrain.running)) {
      if ($("runIter")) $("runIter").textContent = _num(m.iter);
      if ($("runTarget")) $("runTarget").textContent = m.target_iter ? _num(m.target_iter) : "不限";
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


  document.querySelectorAll("#wLang b").forEach(function (b) {
    b.addEventListener("click", function () {
      var lg = b.getAttribute("data-lang");
      document.querySelectorAll("#wLang b").forEach(function (x) { x.classList.toggle("on", x === b); });
      if (window.I18N) window.I18N.apply(lg);
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
    var STRUCT = ["resolution","archi","e_dims","ae_dims","d_dims","d_mask_dims",
                  "face_type","models_opt_on_gpu","gan_power","true_face_power",
                  "face_style_power","src","dst","name"];
    var patch = {};
    box.querySelectorAll("[data-cfg]").forEach(function (el) {
      var k = el.dataset.cfg;
      if (!k || STRUCT.indexOf(k) >= 0) return;
      if (el.tagName === "INPUT") {
        var v = el.value.trim();
        if (v === "") return;                 /* 空着 = 不改这项，别把 "" 写进 _data.dat */
        patch[k] = !isNaN(Number(v)) ? Number(v) : v;
      } else if (el.tagName === "SELECT") {
        patch[k] = el.value;
      } else if (el.classList.contains("sw")) {
        patch[k] = el.classList.contains("on");
      } else if (el.classList.contains("seg")) {
        /* 取 data-v（原始值），不能取 textContent —— 英文界面下会写进 "Fast" 这种 */
        var on = el.querySelector("b.on");
        if (on) {
          var sv = (on.dataset.v !== undefined ? on.dataset.v : on.textContent.trim());
          /* "1.0（关闭）" 这种带说明的选项值不能原样写进 .dat —— 4.0 会 float() 崩掉 */
          var mm = /^([0-9.]+)\s*[（(].*[）)]$/.exec(String(sv));
          patch[k] = mm ? mm[1] : sv;
        }
      }
    });

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
  var N_VIDEO = 40;       /* 视频轨缩略占位块 */
  var N_FACE = 25;        /* 人脸轨块数 */
  var N_CUT = 6;          /* 剪切区段数 */
  var N_ANGLE = 3;        /* 角度色带数 */

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
    var d = { video: [], faces: [], cuts: [], angles: [] };
    var i, span, s, e;

    /* 视频帧条：40 个缩略占位块，均匀铺满整条轴 */
    span = total / N_VIDEO;
    for (i = 0; i < N_VIDEO; i++) {
      d.video.push({ s: Math.round(i * span), e: Math.round((i + 1) * span) });
    }

    /* 剪切 / 角度：既然还没接后端，就不要造假数据 —— 空的轨道比假的红块诚实。
       有真实数据时由 MergeTimeline.setData({cuts, angles}) 填。 */
    return d;
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

  function placeHead() {
    var head = $('mtlHead');
    if (head) head.style.left = (S.frame * S.ppf) + 'px';
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

  function stopPlay() {
    if (S.timer) {
      clearInterval(S.timer);
      S.timer = null;
    }
    S.playing = false;
    setPlayIcon(false);
  }

  function startPlay() {
    if (S.total <= 0) return;
    if (S.frame >= S.total - 1) setFrame(0);
    S.playing = true;
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
    if (S.playing) stopPlay();
    else startPlay();
  }

  /* ---------- 交互 ---------- */
  function frameAt(clientX) {
    var stack = $('mtlStack');
    if (!stack) return -1;
    var rect = stack.getBoundingClientRect();
    return clampFrame((clientX - rect.left) / S.ppf);
  }

  function onDown(e) {
    var sc = $('mtlScroll');
    if (!sc || e.button !== 0) return;
    if (e.target === sc) return;          /* 点在滚动条本体上：交给原生滚动 */
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
      inp.addEventListener('change', function () {
        var v = parseInt(this.value, 10);
        if (!isNaN(v)) setFrame(v);
        else setFrame(S.frame);
      });
      inp.addEventListener('keydown', function (e) {
        if (e.key === 'Enter') {
          var v = parseInt(this.value, 10);
          if (!isNaN(v)) setFrame(v);
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
  }

  /* ---------- 初始化 ---------- */
  function init(totalFrames, fps) {
    var root = $('mergeTracks');
    if (!root) return null;
    S.total = Math.max(0, Math.floor(isFinite(totalFrames) ? totalFrames : 0));
    if (fps && fps > 0) S.fps = fps;
    if (!S.ready) {
      bindEvents();
      S.ready = true;
    }
    stopPlay();
    S.data = buildData(S.total);
    S.frame = 0;
    renderAll();
    return setFrame(0);
  }

  window.MergeTimeline = {
    /* 用真实数据覆盖演示数据（人脸轨、剪切段、角度带） */
    setData: function (partial) {
      if (!S.data || !partial) return;
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

  /* 占位自启动：宿主未显式调用 init 时，先按 300 帧渲染一条演示时间轴；
     宿主随后调用 MergeTimeline.init(state.totalFrames) 会整条重绘覆盖。 */
  document.addEventListener('DOMContentLoaded', function () {
    if (!S.ready) init(300);
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

  var PLACEHOLDER_COUNT = 24;          // 人脸网格占位数
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
      var target = null;
      all('.mgpage').forEach(function (el) {
        var on = el.getAttribute('data-page') === name;
        el.classList.toggle('on', on);
        if (on) {
          target = el;
          el.setAttribute('aria-hidden', 'false');
        } else {
          el.setAttribute('aria-hidden', 'true');
        }
      });

      // 主视图让位（CSS：#p-merge.mgp-on > .mgbody / .mgtl 隐藏）
      var merge = document.getElementById('p-merge');
      if (merge) merge.classList.toggle('mgp-on', !!target);

      this.current = target ? name : 'main';
      return this.current;
    },

    /* 生成 n 个棋盘格占位缩略图（默认 24） */
    buildFaceGrid: function (n) {
      var grid = document.getElementById('mgFaceGrid');
      if (!grid) return 0;
      var count = (typeof n === 'number' && n >= 0) ? n : PLACEHOLDER_COUNT;
      var html = '';
      for (var i = 1; i <= count; i++) {
        html += '<div class="mgp-face" data-idx="' + i + '" title="人脸 ' + i + '（占位）">' +
                '<i>' + i + '</i></div>';
      }
      grid.innerHTML = html;
      this.setFaceCount(count);
      return count;
    },

    /* 更新计数 + 空态（n = 0 时显示「还没有建立人脸库」） */
    setFaceCount: function (n) {
      var count = Math.max(0, parseInt(n, 10) || 0);
      var label = document.getElementById('mgFaceCount');
      var grid = document.getElementById('mgFaceGrid');
      var empty = document.getElementById('mgFaceEmpty');
      if (label) label.innerHTML = '已建库 <b>' + count + '</b> 张人脸';
      if (grid) grid.hidden = count === 0;
      if (empty) empty.hidden = count !== 0;
      return count;
    },

    /* 读取某个 seg 当前选中项的 data-v（没有 data-v 时退化成文字） */
    readSeg: function (key) {
      var seg = one('.mgpage .seg[data-k="' + key + '"]');
      if (!seg) return '';
      var on = seg.querySelector('b.on');
      if (!on) return '';
      var v = on.getAttribute('data-v');
      return v === null ? on.textContent.trim() : v;
    },

    /* 采集导出参数：字段名与 /api/export/start 一致 */
    settings: function () {
      var out = one('#mgOutPath');
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
        output_path: out ? out.value.trim() : '',
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
        erode_mask: tiny('遮罩腐蚀'),
        blur_mask: tiny('遮罩羽化'),
        model: (document.querySelector('#mgModels .mgitem.on .nm') || {}).textContent || ''
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

  /* ---------- 事件绑定 ---------- */

  // seg 点击：只处理 .mgpage 内的，避免和主页面其它 seg 抢事件
  function bindSeg() {
    document.addEventListener('click', function (e) {
      var b = e.target && e.target.closest ? e.target.closest('.mgpage .seg > b') : null;
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

    var pick = document.getElementById('mgOutPick');
    if (pick) {
      pick.addEventListener('click', function () {
        document.dispatchEvent(new CustomEvent('merge:pick-output-dir', { detail: {} }));
      });
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
    MergePages.buildFaceGrid(PLACEHOLDER_COUNT);
    MergePages.syncStatus();
    bindSeg();
    bindActions();
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
        { k: "prioritize_mouth_eyes", l: "嘴眼优先", t: "sw", v: 0, d: "优先训练嘴部和眼部区域" },
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
        { k: "uniform_yaw_distribution", l: "均匀偏航", t: "sw", v: 0, d: "侧脸少的 faceset 用，让 yaw 分布均匀" }
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
          k: "device", l: "设备", t: "sel", v: "0", d: "", titlelabel: true,
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
      inp.value = r.v == null ? "" : String(r.v);
      inp.setAttribute("inputmode", "decimal");
      inp.spellcheck = false;
      cc.appendChild(inp);
    } else if (r.t === "seg") {
      var seg = mk("div", "seg");
      (r.o || []).forEach(function (o, i) {
        var b = mk("b");
        b.textContent = o;
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
    } else if (r.t === "sw") {
      var sw = mk("div", "sw" + (r.v ? " on" : ""));
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

    groups.forEach(function (g) {
      /* "训练参数 · 补充" → "训练参数" */
      var base = String(g.t || "").split(" · ")[0].trim();
      base = MERGE_INTO[base] || base;
      /* 同一张卡只并一次：EXTRA_GROUPS 里"数据增强 · 补充"出现了两次，
         以前会并进去两遍，数据增强参数从 8 行涨到 17 行 */
      if (seen[base]) return;
      seen[base] = 1;
      var card = findCardByTitle(box, base);
      if (card) mergeInto(card, g);
      else leftover.push({ base: base, g: g });
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
      if (window.I18N && window.I18N.refresh) window.I18N.refresh();
    }, 80);

    if (window.I18N && window.I18N.refresh) window.I18N.refresh();
    return true;
  };

  /* 挂钩 openCfg：每次重渲染后把补齐块补回去（openCfg 会清空 cfgBox） */
  api.init = function () {
    if (typeof window.openCfg === "function" && !window.openCfg.__trainExtra) {
      var orig = window.openCfg;
      var wrapped = function () {
        var ret = orig.apply(this, arguments);
        try { api.mount(); } catch (e) { /* 静默：补齐块失败不影响原页面 */ }
        return ret;
      };
      wrapped.__trainExtra = true;
      window.openCfg = wrapped;
    }
    if (document.readyState === "loading") {
      document.addEventListener("DOMContentLoaded", function () { api.mount(); });
    } else {
      api.mount();
    }
    return api;
  };

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
   * 名字 / 架构 / 子分支 / 算子 / 分辨率 / 四个维度 / 脸型。
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
      var sbox = $("cfgSearch");
      var anchor = sbox ? sbox.parentNode : box;   /* 放在搜索框上面 */
      anchor.parentNode.insertBefore(host, anchor);
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
      row("脸型", "训练时按哪种脸对齐裁切，改等于重做数据集",
          '<select class="mgsel" id="nmFace">' + opt(S.face_types, "v", "label", "wf") + "</select>") +
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
      var sbox = $("cfgSearch");
      var anchor = sbox ? sbox.parentNode : box;
      anchor.parentNode.insertBefore(host, anchor);
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
      ro("脸型", m.face_type_label || m.face_type) +
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
                    "nmAe", "nmE", "nmD", "nmMask", "nmFace"];
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
      ["nmName", "nmArchi", "nmSub", "nmKernel", "nmFace"].forEach(function (k) {
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
    if (window.go) window.go("train-cfg");
    setTimeout(function () { nmEnter(true); nmBindCreate(); }, 80);
  }

  function nmSubmit() {
    var name = ($("nmName").value || "").trim();
    if (!name) { $("nmErr").textContent = "起个名字"; $("nmName").focus(); return; }
    var body = {
      name: name,
      archi_display: $("nmArchi").value,
      subarchi: $("nmSub").value,
      kernel_label: $("nmKernel") ? $("nmKernel").value : "",
      face_type: $("nmFace").value,
      resolution: parseInt($("nmRes").value, 10) || undefined,
      ae_dims: parseInt($("nmAe").value, 10) || undefined,
      e_dims: parseInt($("nmE").value, 10) || undefined,
      d_dims: parseInt($("nmD").value, 10) || undefined,
      d_mask_dims: parseInt($("nmMask").value, 10) || undefined,
      batch_size: (NEW.spec && NEW.spec.recommend && NEW.spec.recommend.batch_size) || 4
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
      nmEnter(false);
      btn.disabled = false;
      btn.textContent = "开始训练";
      if (window.toast) window.toast("已创建并开始训练", d.name + " · " + d.options.archi + " · " + d.options.resolution + " 分辨率", "success");
      if (window.go) window.go("run");
      if (window.pollTrain) window.pollTrain();
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
    function sync() { txt.textContent = cur(); btn.title = cur(); }
    function build() {
      menu.innerHTML = "";
      Array.prototype.forEach.call(sel.options, function (o, i) {
        var it = doll("div", "ddopt" + (i === sel.selectedIndex ? " on" : ""));
        it.textContent = o.textContent;
        it.title = o.textContent;
        it.addEventListener("click", function (ev) {
          ev.stopPropagation();
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

  var DEVOPTS = null;
  function applyDevices(d) {
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
