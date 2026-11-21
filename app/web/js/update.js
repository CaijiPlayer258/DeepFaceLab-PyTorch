/* 自动更新：检查 GitHub Release，只拉代码包。
   只有一个按钮：没更新时是「检查更新」，发现新版本后同一个按钮变成「下载并安装」。 */
(function () {
  "use strict";
  var btn = document.getElementById("updCheck");
  var state = document.getElementById("updState");
  var notes = document.getElementById("updNotes");
  var local = document.getElementById("updLocal");
  if (!btn) return;

  var INFO = null;
  var MODE = "check";        // check | install

  function setState(t) { if (state) state.textContent = t; }
  function setBtn(t) { btn.textContent = t; }

  fetch("/api/update/version").then(function (r) { return r.json(); })
    .then(function (j) { if (local) local.textContent = "当前 v" + j.version; })
    .catch(function () {});

  btn.addEventListener("click", function () {
    if (MODE === "install") { doInstall(); return; }
    doCheck();
  });

  function doCheck() {
    btn.disabled = true;
    setState("正在检查…");
    fetch("/api/update/check").then(function (r) { return r.json(); }).then(function (j) {
      btn.disabled = false;
      INFO = j;
      if (!j.ok) { setState("检查失败：" + (j.error || "网络不可用")); return; }
      if (notes) {
        notes.style.display = "block";
        notes.textContent = (j.notes || "").slice(0, 4000) || "(该版本没有说明)";
      }
      if (j.behind) {
        MODE = "install";
        setBtn("下载并安装 " + j.tag);
        setState("发现新版本（" + j.package + "）");
      } else {
        MODE = "check";
        setBtn("检查更新");
        setState("已是最新（" + j.local + "）");
      }
    }).catch(function (e) {
      btn.disabled = false;
      setState("检查失败：" + e);
    });
  }

  function doInstall() {
    if (!INFO || !INFO.behind) return;
    btn.disabled = true;
    setState("正在下载并安装…");
    fetch("/api/update/apply", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ asset: INFO.asset ? INFO.asset.name : null })
    }).then(function (r) { return r.json(); }).then(function (j) {
      btn.disabled = false;
      if (!j.ok) { setState("安装失败：" + (j.error || "")); return; }
      if (!j.installed) { setState(j.message || "已是最新"); return; }
      MODE = "check";
      setBtn("检查更新");
      setState("已安装 " + j.files + " 个文件，重启后生效" + (j.backup ? "（备份：" + j.backup + "）" : ""));
    }).catch(function (e) {
      btn.disabled = false;
      setState("安装失败：" + e);
    });
  }
})();
