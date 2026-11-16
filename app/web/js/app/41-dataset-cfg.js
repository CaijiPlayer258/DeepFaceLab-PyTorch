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
