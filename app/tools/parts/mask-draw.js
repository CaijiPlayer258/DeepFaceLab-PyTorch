/* ═══════════════════════════════════════════════════════════════════════════
   遮罩绘制页（#p-tool-mask）交互补丁
   ───────────────────────────────────────────────────────────────────────────
   对照 4.0\ui\components\page_maskprocessor\page_maskprocessor.py：

     _on_launch        → python -m MaskProcessor
                         （端口写死在 MaskProcessor\__main__.py: port = 8000，
                           启动后 webbrowser.open("http://127.0.0.1:8000")）
     _on_apply_xseg    → python -m DataAugmenter -i "<人脸集>" -m <模型> -r <分辨率>
                         [--invert] [--model-file <文件>] [--engine 时追加 --trt]
     _on_export_onnx   → python models/Model_XSeg/export_XSeg.py
                         python models/Model_XSegLite/export_XSegLite.py

   本文件只做界面演示：不启动进程、不调后端，点击后用 toast 把真实命令亮出来
   （对应 4.0 的 show_command_notification）。
   暴露：window.MaskDraw = { init, buildApplyCmd, params }
   ═══════════════════════════════════════════════════════════════════════════ */

(function () {
  "use strict";

  var PAGE_ID = "p-tool-mask";
  var PY = "python";     // 4.0 用 sys.executable，前端只知道解释器叫 python
  var PORT = "8000";     // MaskProcessor/__main__.py 里写死

  function byId(id) { return document.getElementById(id); }
  function page() { return byId(PAGE_ID); }

  /* toast(title, sub, type, ms) 由 app.js 提供；单独打开片段时退化到 console */
  function notify(title, sub, type) {
    if (typeof window.toast === "function") window.toast(title, sub, type || "info");
    else console.log("[mask-draw] " + title + " — " + sub);
  }

  /* 按 .f 标签文字找行，再取这行里的 .seg —— 避开按顺序取第几个 seg 的脆弱写法 */
  function rowSeg(labelText) {
    var p = page();
    if (!p) return null;
    var rows = p.querySelectorAll(".dbgrow");
    for (var i = 0; i < rows.length; i++) {
      var lb = rows[i].querySelector(".f");
      if (lb && lb.textContent.trim() === labelText) return rows[i].querySelector(".seg");
    }
    return null;
  }

  function segText(seg) {
    if (!seg) return "";
    var on = seg.querySelector("b.on");
    return on ? on.textContent.trim() : "";
  }

  /* 人脸集路径：优先新加的 #mdPath，否则用现有页里那个 .pathval */
  function pathValue() {
    var p = page();
    if (!p) return "";
    var input = byId("mdPath") || p.querySelector(".pathinp .pathval");
    return input ? String(input.value || "").trim() : "";
  }

  /* 完全照 4.0 _on_apply_xseg 的拼接顺序与条件：
       - --invert          反转遮罩 == "是"
       - --model-file      只在模型类型为 XSegLite 且选中了真实文件时
       - --trt             选中的文件以 .engine 结尾
  */
  function buildApplyCmd() {
    var model = segText(rowSeg("遮罩模型")) || "XSegLite";
    var resEl = byId("mdRes");
    var res = (resEl && String(resEl.value || "").trim()) || "256";
    var invert = segText(rowSeg("反转遮罩")) === "是";

    var fileEl = byId("mdModelFile");
    var file = fileEl ? String(fileEl.value || "").trim() : "";
    if (file.charAt(0) === "（") file = "";   // “（未检测到模型文件）”当成没有选

    var cmd = PY + " -m DataAugmenter -i \"" + pathValue() + "\" -m " + model + " -r " + res;
    if (invert) cmd += " --invert";
    if (model === "XSegLite" && file) cmd += " --model-file " + file;
    if (/\.engine$/i.test(file)) cmd += " --trt";
    return cmd;
  }

  function params() {
    return {
      model: segText(rowSeg("遮罩模型")) || "XSegLite",
      res: (byId("mdRes") && byId("mdRes").value) || "256",
      invert: segText(rowSeg("反转遮罩")) === "是",
      modelFile: byId("mdModelFile") ? byId("mdModelFile").value : "",
      path: pathValue(),
      port: PORT
    };
  }

  function init() {
    var p = page();
    if (!p || p.dataset.mdReady) return;   // 幂等：来回切页不叠加 handler
    p.dataset.mdReady = "1";

    /* 1) 分段控件：现有页的 .seg 没绑事件，点了不切换 */
    var segs = p.querySelectorAll(".seg");
    Array.prototype.forEach.call(segs, function (seg) {
      seg.addEventListener("click", function (e) {
        var b = e.target.closest ? e.target.closest("b") : null;
        if (!b || !seg.contains(b)) return;
        Array.prototype.forEach.call(seg.querySelectorAll(":scope > b"), function (x) {
          x.classList.remove("on");
        });
        b.classList.add("on");
      });
    });

    /* 2) 启动 MaskProcessor（4.0 _on_launch）：只翻状态文字，不假装真的拉起进程 */
    var start = byId("mdStart") || p.querySelector(".dbgacts .btn.pri");
    var state = byId("mdState");
    if (start) {
      start.addEventListener("click", function () {
        var running = start.dataset.running === "1";
        start.dataset.running = running ? "0" : "1";
        if (state) state.textContent = running ? "未启动" : "运行中 · :" + PORT;
        notify(running ? "遮罩编辑器已关闭" : "启动 MaskProcessor",
               PY + " -m MaskProcessor（端口 " + PORT + "，起来后自动开浏览器）");
      });
    }

    /* 3) 在浏览器打开：MaskProcessor/__main__.py 里固定的地址 */
    var open = byId("mdOpen");
    if (open) open.addEventListener("click", function () {
      notify("浏览器地址", "http://127.0.0.1:" + PORT);
    });

    /* 4) 应用遮罩（4.0 _on_apply_xseg） */
    var apply = byId("mdApply");
    if (apply) apply.addEventListener("click", function () {
      if (!pathValue()) {
        notify("缺少人脸集路径", "先选已对齐的人脸集目录，再应用遮罩", "error");
        return;
      }
      notify("遮罩应用命令", buildApplyCmd());
    });

    /* 5) 导出 ONNX（4.0 _on_export_onnx 的两个脚本） */
    var exX = byId("mdExportXSeg");
    if (exX) exX.addEventListener("click", function () {
      notify("导出 XSeg → ONNX", PY + " models/Model_XSeg/export_XSeg.py");
    });
    var exL = byId("mdExportLite");
    if (exL) exL.addEventListener("click", function () {
      notify("导出 XSegLite → ONNX", PY + " models/Model_XSegLite/export_XSegLite.py");
    });
  }

  window.MaskDraw = { init: init, buildApplyCmd: buildApplyCmd, params: params };

  /* 片段独立预览时也能用；合进 index.html 后由 MaskDraw.init() 或下面的兜底触发 */
  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    setTimeout(init, 0);
  }

  /* app.js 的 go() 不会通知子模块，导航到本页时再补一次（init 自带幂等） */
  document.addEventListener("click", function (e) {
    var nav = e.target && e.target.closest ? e.target.closest('.nav[data-go="tool-mask"]') : null;
    if (nav) setTimeout(init, 0);
  });
})();
