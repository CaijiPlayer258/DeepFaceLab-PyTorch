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
   * 名字 / 架构 / 子分支 / 算子 / 分辨率 / 四个维度。
   * 脸型不在这块卡里：「数据集参数」卡已经有一项（whole_face / full_face …），
   * 两处同时出现就是重复项。创建时 nmSubmit() 从那一项取，再转成 dat 认的短码。
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
      /* ★ 必须插进 #cfgBox 里面。
         以前是 "sbox.parentNode" 再 insertBefore —— 而搜索框外面那层 wrapper 是
         position:sticky(top:-2px)，结果整张「新建模型 · 结构」卡跟着一起吸顶，
         像 Excel 冻结首行那样压在参数列表上，页面看起来像"滚不动"。
         放进 cfgBox 后它跟普通卡片一样随页面滚动。 */
      if (box.firstChild) box.insertBefore(host, box.firstChild);
      else box.appendChild(host);
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
      /* 同上：不能插进吸顶的搜索框 wrapper，否则整卡跟着吸顶 */
      if (box.firstChild) box.insertBefore(host, box.firstChild);
      else box.appendChild(host);
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
                    "nmAe", "nmE", "nmD", "nmMask"];
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
      ["nmName", "nmArchi", "nmSub", "nmKernel"].forEach(function (k) {
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
    /* ★ 先确保参数行已经渲染出来。
       以前是直接 go("train-cfg")，而 go() 只在「#cfgBox 里一张卡都没有」时才调 openCfg；
       新建面板自己往 #cfgBox 插了一张「模型结构」卡，于是 go() 认为"已经渲染过"，
       参数行（批次大小 / 预训练 / 预训练集路径 / 优化器 …）**永远不出现** ——
       用户在新建流程里根本没法设这些，建出来的模型只能吃推荐值。 */
    var box = document.getElementById("cfgBox");
    var hasRows = box && box.querySelector(".card:not(#nmStruct):not(#nmInfo)");
    if (!hasRows) {
      var m0 = window.__cfgModel || (window.MODELS || [])[0] || null;
      if (m0 && window.openCfg) { try { window.openCfg(m0); } catch (e) { if (window.go) window.go("train-cfg"); } }
      else if (window.go) window.go("train-cfg");
    } else if (window.go) {
      window.go("train-cfg");
    }
    setTimeout(function () { nmEnter(true); nmBindCreate(); }, 80);
  }

  /* 脸型只有一个入口：「数据集参数」卡里的 face_type 下拉（whole_face / full_face …）。
     创建模型时要把它翻成 dat 认的短码 —— Model_SAEHD 是拿 'wf'/'f'/'h'/'mf'/'head'
     直接查 FaceType 表的，发全名过去，训练一启动就 KeyError。 */
  var FACE_SHORT = { whole_face: "wf", full_face: "f", half_face: "h",
                     midfull_face: "mf", head: "head" };

  function nmFaceType() {
    var sel = document.querySelector("#cfgBox [data-cfg=face_type]");
    var v = sel ? String(sel.value || "").trim() : "";
    return FACE_SHORT[v] || v || "wf";      /* 空着或认不出就给 4.1 的老默认 wf */
  }

  function nmSubmit() {
    var name = ($("nmName").value || "").trim();
    if (!name) { $("nmErr").textContent = "起个名字"; $("nmName").focus(); return; }
    /* 一次请求带全：spec（结构，创建后不可改）+ options（训练参数，页面上填的）。
       后端用同一套 _clean_options 解析，返回"实际生效的全部 options"。
       前端不再需要"创建成功后再发一次 config"这种补丁式做法。 */
    var body = {
      name: name,
      spec: {
        archi_display: $("nmArchi").value,
        subarchi: $("nmSub").value,
        kernel_label: $("nmKernel") ? $("nmKernel").value : "",
        face_type: nmFaceType(),
        resolution: parseInt($("nmRes").value, 10) || undefined,
        ae_dims: parseInt($("nmAe").value, 10) || undefined,
        e_dims: parseInt($("nmE").value, 10) || undefined,
        d_dims: parseInt($("nmD").value, 10) || undefined,
        d_mask_dims: parseInt($("nmMask").value, 10) || undefined
      },
      options: (window.collectCfgPatch ? window.collectCfgPatch(true) : {}),
      /* 批次大小：页面上的值已经在 options 里了；这里只在它缺失时用显卡推荐值兜底 */
      batch_size: (function () {
        var el = document.querySelector("#cfgBox [data-cfg=batch_size]");
        var v = el ? parseInt(el.value, 10) : 0;
        if (v > 0) return v;
        return (NEW.spec && NEW.spec.recommend && NEW.spec.recommend.batch_size) || 4;
      })()
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
      btn.disabled = false;
      btn.textContent = "开始训练";
      if (window.toast) window.toast("已创建并开始训练", d.name + " · " + d.options.archi + " · " + d.options.resolution + " 分辨率", "success");
      /* 后端会顺手把超范围的参数夹进合法区间（分辨率上限 640 等）。
         以前是静默的 —— 填 1024 建出来是 640，人还以为在训 1024。现在先说清楚。 */
      if (d.clamped && Object.keys(d.clamped).length) {
        var _nm = { resolution: "分辨率", ae_dims: "编码器宽度", e_dims: "编码器维度",
                    d_dims: "解码器维度", d_mask_dims: "遮罩维度" };
        var _txt = Object.keys(d.clamped).map(function (k) {
          var c = d.clamped[k];
          return (_nm[k] || k) + " " + c.requested + "→" + c.used;
        }).join("，");
        if (window.toast) window.toast("部分参数超出范围，已自动调整", _txt, "info");
      }
      /* ★ 关键：把「当前模型」切到刚建的这个再进运行页。
         以前这里是 go("run") —— 运行页不认模型，顶部标题还停在旧值（甚至
         是 HTML 里的占位 "384wf"），曲线/统计也按 "列表里第一个" 取，
         于是用户看到的是"点新建，却在训练第一个模型"。
         现在重拉一次模型列表，按名字找回这一条，openRun(它)。 */
      fetch("/api/models").then(function (r) { return r.json(); }).then(function (mm) {
        var list = (mm && mm.models) || [];
        var fresh = null;
        for (var i = 0; i < list.length; i++) {
          if (list[i] && list[i].name === d.name) { fresh = list[i]; break; }
        }
        if (list.length) window.MODELS = list;
        /* 立刻重画「模型选择」列表：以前新建完要切页/刷新才看得到新模型 */
        try { if (window.renderTrainer && window.MODELS) window.renderTrainer(window.MODELS); } catch (e) {}
        if (fresh) { window.__cfgModel = fresh; window.__runModel = fresh; }
        nmEnter(false);
        if (fresh && window.openRun) window.openRun(fresh);
        else if (window.go) window.go("run");
        if (window.pollTrain) window.pollTrain(fresh || undefined);
      }).catch(function () {
        nmEnter(false);
        if (window.go) window.go("run");
        if (window.pollTrain) window.pollTrain();
      });
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
    function off(i) { return (sel.__off && sel.__off[i]) || null; }
    function sync() {
      txt.textContent = cur();
      /* 选中项本身是「未安装」时，按钮的 tooltip 要能说明原因，不能只回一遍标签文字 */
      var o = sel.options[sel.selectedIndex], mm = o ? off(sel.selectedIndex) : null;
      btn.title = mm ? (mm.title || o.textContent) : cur();
      btn.classList.toggle("off", !!mm);
    }
    function build() {
      menu.innerHTML = "";
      Array.prototype.forEach.call(sel.options, function (o, i) {
        var mm = off(i);
        var it = doll("div", "ddopt" + (i === sel.selectedIndex ? " on" : "") + (mm ? " off" : ""));
        it.textContent = o.textContent;
        it.title = mm ? (mm.title || o.textContent) : o.textContent;
        if (mm) it.setAttribute("aria-disabled", "true");
        it.addEventListener("click", function (ev) {
          ev.stopPropagation();
          /* 未安装：不切换，只解释（清掉原生 select 被选中的可能） */
          if (mm) { if (mm.onBlock) mm.onBlock(); sync(); close(); return; }
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

  /* 「当前可用 GPU 列表」直接写在下拉框旁边。
     以前这些信息只藏在选项文字里：存的值一旦不在列表里（换机器、拔了显卡、
     从别的工程拷了 dat），下拉框就静默跳到「自动选择」，用户完全看不出
     自己的选择已经被换掉了 —— 和「选了卡却不生效」是同一类问题。 */
  function devAvailableText(d) {
    var parts = [];
    ((d && d.gpus) || []).forEach(function (g) {
      var mem = g.mem_mb ? " · " + Math.round(g.mem_mb / 1024) + " GB" : "";
      parts.push("GPU " + g.idx + " " + (g.name || "") + mem);
    });
    if (d && d.cpu) parts.push(String(d.cpu));
    if (!parts.length) return "本机没有可用 GPU";
    return "可用：" + parts.join(" ｜ ");
  }

  /* want 给了就用它（回填 dat 时浏览器已经把无效值丢掉，sel.value 读不回来） */
  function devGuard(sel, want) {
    if (!sel) return;
    var d = window.__DEVINFO || {};
    var stale = sel.querySelector('option[data-invalid="1"]');
    if (stale) stale.parentNode.removeChild(stale);
    var cur = (want === undefined || want === null) ? String(sel.value || "") : String(want);
    /* 只对 GPU 序号（"0" / "0,1"）做拦截：auto / cpu 永远合法 */
    var numeric = /^[0-9]+(,[0-9]+)*$/.test(cur);
    var known = false;
    Array.prototype.forEach.call(sel.options, function (o) { if (o.value === cur) known = true; });
    if (numeric && !known) {
      /* 禁用它比默默忽略好：这条 disabled 选项让「本机没有这张卡」变成看得见的事实 */
      var op = document.createElement("option");
      op.value = cur;
      op.textContent = "⚠ GPU " + cur + "（本机不存在）";
      op.disabled = true;
      op.setAttribute("data-invalid", "1");
      sel.insertBefore(op, sel.firstChild);
      sel.value = cur;
    }
    var row = sel.closest ? sel.closest(".cfgrow") : null;
    if (!row) return;
    var h = row.querySelector(".devhint");
    if (!h) {
      h = document.createElement("span");
      h.className = "devhint";
      (sel.parentNode || row).appendChild(h);
    }
    /* 只有一张卡时，下拉框里已经写着「GPU 0 · 型号 · 显存」，再挂一条“可用: …”就是重复。
       多卡才需要这行（帮用户确认另一张卡也在）。 */
    var _many = !!(d && d.devices && d.devices.length > 1);
    var txt = _many ? devAvailableText(d) : "";
    h.textContent = txt;
    h.title = txt;
    h.style.display = txt ? "" : "none";
  }
  window.__devGuard = devGuard;

  var DEVOPTS = null;
  function applyDevices(d) {
    window.__DEVINFO = d || null;
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
      /* 旧值不在新列表里：留一条 disabled 的说明，别让下拉框自己换成「自动选择」 */
      devGuard(sel, keep);
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
      /* 结构卡（新建 / 只读）不参与搜索过滤：它们是页面的上下文，
         过滤掉会让人以为"新建模型的面板不见了"。 */
      if (cards[i].id === "nmStruct" || cards[i].id === "nmInfo") continue;
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
