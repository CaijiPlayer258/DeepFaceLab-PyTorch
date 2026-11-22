  /* ---------- 面部 68 点：编辑器保存过就用保存的 ---------- */
  var FACE_PTS = null;
  function facePts() {
    if (FACE_PTS) return FACE_PTS;
    FACE_PTS = L68.map(function (p) { return [p[0], p[1], p[2]]; });
    return FACE_PTS;
  }
  /* 保存在工作区里（face68.json），换端口、换机器都还在。
     判据是 r.ok：后端没有这份数据时回 404 + detail（以前是 200 + points:null，
     只能靠 d.points == null 猜），这里把 detail 摆到界面上，别让它静默回内置坐标。 */
  /* loud=true 才弹提示：启动时这一发只是把保存过的 68 点读回来，没保存过是正常状态，
     不该在右下角冒一条吐司（用户报的「开窗就弹」）。真正该喊的是用户主动点角度图那类动作。 */
  function loadFace(loud) {
    return fetch("/api/face").then(function (r) {
      if (!r.ok) {
        return r.json().catch(function () { return {}; }).then(function (d) {
          if (loud) {
            toast("这一帧没有关键点",
                  String((d && (d.detail || d.error)) || ("HTTP " + r.status)).slice(0, 120),
                  "info");
          } else {
            console.info("[face] 没有保存过的 68 点（正常）：" +
                         String((d && (d.detail || d.error)) || ("HTTP " + r.status)).slice(0, 80));
          }
          return null;
        });
      }
      return r.json();
    }).then(function (d) {
      if (d && d.ok !== false && d.points && d.points.length === 68) {
        FACE_PTS = d.points;
        HEAD = null;
      }
    }).catch(function (e) { console.warn("face load failed", e); });
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
    /* 空字符串不要当结果用：contextBridge 会把 File 序列化，webUtils 拿到拷贝就返回空。
       这时退回 preload 在 drop 捕获阶段记下的那次路径。 */
    if (window.desktop && window.desktop.pathForFile) {
      var p = window.desktop.pathForFile(f);
      if (p) return p;
    }
    if (window.desktop && window.desktop.lastDropPath) {
      var q = window.desktop.lastDropPath();
      if (q) return q;
    }
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
    document.querySelectorAll(".pathinp").forEach(function (b) { if (window.bindClickToFocus) window.bindClickToFocus(b); });
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

  /* 整块路径框都能点：以前只有 <input> 的文本区接得住点击，
     点标签「输入」或者框内留白都没反应，得瞄准中间才点得进。 */
  function bindClickToFocus(box) {
    if (!box || box.dataset.focusBound) return;
    box.dataset.focusBound = "1";
    box.addEventListener("mousedown", function (e) {
      if (e.target.closest(".iconpill")) return;      /* 那颗图标是选文件/目录，别抢它的点击 */
      var inp = box.querySelector(".pathval");
      if (!inp || e.target === inp) return;
      e.preventDefault();
      inp.focus();
      try { inp.setSelectionRange(inp.value.length, inp.value.length); } catch (err) {}
    });
  }
  window.bindClickToFocus = bindClickToFocus;
  document.querySelectorAll(".pathinp").forEach(bindClickToFocus);

  /* 路径框统一去引号：从资源管理器「复制为路径」粘进来的值带双引号，
     带引号的字符串 Path 打不开 —— 后端也会清，但界面先改掉，
     用户才不会以为路径真的不存在。 */
  function cleanPathValue(el) {
    if (!el || el.tagName !== "INPUT" || !el.value) return;
    var t = el.value.trim();
    while (t.length && /[\s"\u201c\u201d]/.test(t.charAt(0))) t = t.slice(1);
    while (t.length && /[\s"\u201c\u201d]/.test(t.charAt(t.length - 1))) t = t.slice(0, -1);
    if (t !== el.value) el.value = t;
  }
  window.cleanPathValue = cleanPathValue;
  document.addEventListener("change", function (e) {
    var el = e.target;
    if (el && el.classList && el.classList.contains("pathval")) cleanPathValue(el);
  }, true);

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
