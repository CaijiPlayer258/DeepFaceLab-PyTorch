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

  /* ---------- 横幅调试（Alert / Banner） ---------- */
  /* 复用的是训练页那条常驻横幅的同款类名：.alert + 配色 + .loadbanner。
     这里只是把节点插进调试页的容器，行为（左侧强调条 / 圆点 i / 标题 / ×）与真实一致。 */
  function bnKind() {
    var el = document.querySelector("#bnKind > b.on");
    return el ? el.getAttribute("data-k") : "info";
  }

  function bnRender() {
    var host = $("bnHost");
    if (!host) return;
    var title = ($("bnTitle") || {}).value || "训练提示";
    var body = ($("bnBody") || {}).value || "";
    var kind = bnKind();
    host.innerHTML = "";
    var el = document.createElement("div");
    el.className = "alert " + kind + " loadbanner";
    el.setAttribute("aria-live", "polite");
    var head = document.createElement("div");
    head.className = "alhd";
    head.innerHTML = '<span class="lbico">' + (kind === "info" ? "i" : "!") + '</span>'
      + "<b></b>"
      + '<button class="alx" type="button" title="收起">×</button>';
    head.querySelector("b").textContent = title;
    head.querySelector(".alx").addEventListener("click", function () { host.innerHTML = ""; });
    var bd = document.createElement("div");
    bd.className = "albd";
    bd.textContent = body;
    el.appendChild(head);
    el.appendChild(bd);
    host.appendChild(el);
  }

  if ($("bnShow")) {
    $("bnShow").addEventListener("click", bnRender);
    $("bnHide").addEventListener("click", function () { var h = $("bnHost"); if (h) h.innerHTML = ""; });
    $("bnLong").addEventListener("click", function () {
      var el = $("bnBody");
      if (el) el.value = "预训练结束：inter 已重置为随机初始化，原权重已备份（New416_SAEHD_inter.pth.pretrain-backup、New416_SAEHD_inter_B.pth.pretrain-backup）。"
        + "这次训练的前 2000 步会明显比之前抖，属于正常现象；不想重置可以在参数里把 pretrain_reset_inter 关掉。";
      bnRender();
    });
    document.querySelectorAll("#bnKind > b").forEach(function (b) {
      b.addEventListener("click", function () {
        document.querySelectorAll("#bnKind > b").forEach(function (x) { x.classList.remove("on"); });
        b.classList.add("on");
        bnRender();
      });
    });
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
