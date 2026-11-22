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
    window.__loggedIn = !!(a && a.logged_in) || !!window.__preview;
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
      /* 预览模式（设计工作台画布）直接把欢迎页藏掉：那个会话没有登录态，
         停在扫码页就看不到任何真实界面。 */
      var show = window.__preview ? false : (!AUTH.logged_in || needLegal);
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
