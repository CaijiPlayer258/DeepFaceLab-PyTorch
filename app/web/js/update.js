/* 自动更新：检查 GitHub Release，只拉代码包 */
(function () {
  "use strict";
  var btnCheck = document.getElementById("updCheck");
  var btnApply = document.getElementById("updApply");
  var state = document.getElementById("updState");
  var notes = document.getElementById("updNotes");
  var local = document.getElementById("updLocal");
  if (!btnCheck) return;

  function setState(t) { if (state) state.textContent = t; }

  function loadLocal() {
    fetch("/api/update/version").then(function (r) { return r.json(); })
      .then(function (j) { if (local) local.textContent = "当前 v" + j.version; })
      .catch(function () {});
  }

  var INFO = null;

  btnCheck.addEventListener("click", function () {
    setState("正在检查…");
    if (btnApply) btnApply.style.display = "none";
    fetch("/api/update/check").then(function (r) { return r.json(); }).then(function (j) {
      INFO = j;
      if (!j.ok) { setState("检查失败：" + (j.error || "网络不可用")); return; }
      if (notes) {
        notes.style.display = "block";
        notes.textContent = (j.notes || "").slice(0, 4000) || "(该版本没有说明)";
      }
      if (j.behind) {
        setState("发现新版本 " + j.tag + "（" + j.package + "）");
        if (btnApply) btnApply.style.display = "";
      } else {
        setState("已是最新（" + j.local + "）");
      }
    }).catch(function (e) { setState("检查失败：" + e); });
  });

  if (btnApply) btnApply.addEventListener("click", function () {
    if (!INFO || !INFO.behind) return;
    btnApply.disabled = true;
    setState("正在下载并安装…");
    fetch("/api/update/apply", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ asset: INFO.asset ? INFO.asset.name : null })
    }).then(function (r) { return r.json(); }).then(function (j) {
      btnApply.disabled = false;
      if (!j.ok) { setState("安装失败：" + (j.error || "")); return; }
      if (!j.installed) { setState(j.message || "已是最新"); return; }
      setState("已安装 " + j.files + " 个文件，重启后生效" + (j.backup ? "（备份：" + j.backup + "）" : ""));
    }).catch(function (e) { btnApply.disabled = false; setState("安装失败：" + e); });
  });

  loadLocal();
})();
