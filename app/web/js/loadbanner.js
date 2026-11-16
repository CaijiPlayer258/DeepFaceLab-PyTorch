/* 训练页顶部「常驻横幅」：模块加载失败 → 权重与优化器一起被重置，这件事必须一直摆着。
 *
 * 和右下角 toast 的分工：
 *   toast    一次性反馈（"已保存" / "提取完成"），几秒后自己消失；
 *   横幅     状态。只要这次训练还在跑，它就在 —— 因为它描述的是"模型现在
 *            是从某个模块的初始状态开始训的"，不是"刚才发生了一件事"。
 *
 * 生命周期（用户要求的三种状态）：
 *   1) /api/train/status 里 warnings[] 非空且 running → 显示（文案含模块名与原因）
 *   2) 用户点 × → 本次挂载内不再显示
 *   3) 退出训练页 / 停止训练 → 收起；重新进训练页且训练仍在跑 → 重新显示
 *      （所以 dismissed 只记在内存里，不写 localStorage —— 刷新页面等于"重新进来"）
 *
 * 文案由后端组装（app/server/_load_events.py）：每条 warning 自带 text，
 * 这里只负责"多条合并成一条 + 把模块名列出来"。
 */
(function (global) {
  "use strict";

  var EL = "loadBanner";
  var state = {
    dismissed: false,   /* 只在本次挂载内有效 */
    lastStatus: null,
    lastWarnings: []
  };

  function fmtShape(shape) {
    if (!shape || !shape.length) return "?";
    var vals = [];
    for (var i = 0; i < shape.length; i++) {
      var v = Number(shape[i]);
      if (v > 1) vals.push(v);
    }
    if (shape.length === 1) return String(Number(shape[0]));
    if (!vals.length) return "[" + shape.join(",") + "]";
    return String(Math.max.apply(null, vals));
  }

  function shapeHint(w) {
    if (!w) return "";
    if (w.shape_hint) return w.shape_hint;
    if (w.reason !== "shape") return "";
    var exp = w.expected_shapes || [], fnd = w.found_shapes || [];
    if (exp.length === 1 && fnd.length === 1) return " " + fmtShape(fnd[0]) + "→" + fmtShape(exp[0]);
    if (exp.length === 1) return " 预期 " + fmtShape(exp[0]);
    return "";
  }

  function reasonText(w) {
    return (w && (w.reason_text || w.reason)) || "加载失败";
  }

  function sentence(w) {
    if (w.text) return w.text;   /* 后端已经组装好整句时直接用 */
    var head = w.module + " 未能加载（" + reasonText(w) + shapeHint(w) + "），"
             + "已重置该模块与优化器状态。";
    return head + "本次训练从该模块的初始状态开始。";
  }

  /* warnings[] -> 一条横幅 { title, body, modules, count }；无失败返回 null */
  function merge(warnings) {
    var list = [];
    (warnings || []).forEach(function (w) { if (w && w.module) list.push(w); });
    if (!list.length) return null;
    var names = list.map(function (w) { return String(w.module); });
    if (list.length === 1) {
      var w0 = list[0];
      return {
        title: names[0] + " 未能加载（" + reasonText(w0) + shapeHint(w0) + "）",
        body: sentence(w0),
        modules: names, count: 1
      };
    }
    return {
      title: names.length + " 个模块未能加载（" + names.join("、") + "）",
      body: list.map(sentence).join("\n"),
      modules: names, count: names.length
    };
  }

  function doc() {
    return global.document || null;
  }

  function byId(id) {
    var d = doc();
    if (!d || !d.getElementById) return null;
    return d.getElementById(id);
  }

  function setText(el, txt) {
    if (el) el.textContent = txt == null ? "" : String(txt);
  }

  function hide(el) {
    if (!el) return;
    el.classList.add("hide");
    el.hidden = true;
    el.setAttribute("aria-hidden", "1");
  }

  function show(el) {
    if (!el) return;
    el.classList.remove("hide");
    el.hidden = false;
    el.setAttribute("aria-hidden", "0");
  }

  /* 把合并结果画到 DOM 上；返回合并结果（无内容/已关闭 → 只返回合并结果或 null） */
  function render(el, warnings) {
    var merged = merge(warnings);
    state.lastWarnings = warnings || [];
    if (!el) return merged;
    if (!merged || state.dismissed) { hide(el); return merged; }
    setText(byId("loadBannerTitle"), merged.title);
    setText(byId("loadBannerBody"), merged.body);
    var list = byId("loadBannerList");
    if (list) {
      /* 多个模块时把名字再摆成一行小标签，扫一眼就知道是哪几个 */
      if (merged.count > 1) {
        list.innerHTML = merged.modules.map(function (n) {
          return '<span class="lbchip">' + escapeHtml(n) + "</span>";
        }).join("");
      } else {
        list.innerHTML = "";
      }
    }
    show(el);
    return merged;
  }

  function escapeHtml(s) {
    return String(s).replace(/[&<>"']/g, function (c) {
      return { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c];
    });
  }

  /* 每次 /api/train/status 轮询都调它（幂等） */
  function update(status) {
    if (status) state.lastStatus = status;
    var st = state.lastStatus;
    var el = byId(EL);
    if (!el) return null;
    /* 训练没在跑 = 横幅没有意义（"本次训练"已经结束了） */
    var running = !!(st && st.running);
    return render(el, running ? (st.warnings || []) : []);
  }

  function dismiss() {
    state.dismissed = true;
    hide(byId(EL));
  }

  /* 进训练页：清掉上次的关闭状态，立刻按已知状态重画一次
     —— 这样"重新进训练页"不用等下一次轮询，横幅就回来了 */
  function enter() {
    state.dismissed = false;
    return update(state.lastStatus);
  }

  /* 离开训练页：收起（下次进来由 enter() 重新出现） */
  function leave() {
    hide(byId(EL));
  }

  var API = {
    id: EL,
    state: state,
    merge: merge,
    sentence: sentence,
    shapeHint: shapeHint,
    render: render,
    update: update,
    dismiss: dismiss,
    enter: enter,
    leave: leave
  };

  global.LCSLoadBanner = API;

  /* 绑关闭按钮（DOM 就绪才绑；测试里没有 DOM 会跳过） */
  var d = doc();
  if (d && d.addEventListener) {
    d.addEventListener("DOMContentLoaded", function () {
      var x = byId("loadBannerX");
      if (x && x.addEventListener) x.addEventListener("click", dismiss);
    });
  }

  if (typeof module !== "undefined" && module.exports) module.exports = API;
})(typeof window !== "undefined" ? window : this);
