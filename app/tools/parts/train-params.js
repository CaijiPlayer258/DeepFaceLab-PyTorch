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
      t: "PIA · 补充",
      note: "仅 PIA 生效",
      rows: [
        { k: "pretrain_dataset_path", l: "预训练集", t: "path", v: "", d: "多身份人脸集目录，留空 = 用 dst 人脸集兜底" }
      ]
    }
  ];

  /* ------------------------------------------------------------
   * 二、必须用 select 的两行（候选项 > 4）
   * ------------------------------------------------------------ */
  var EXTRA_SELECT_GROUPS = [
    {
      t: "设备",
      note: "4.0 对应 --force-gpu-idxs / --cpu-only",
      rows: [
        {
          k: "device", l: "设备", t: "sel", v: "0",
          d: "auto = 自动选卡；cpu = 纯 CPU 训练；0,1 = 指定多张卡（逗号分隔）",
          o: ["auto", "cpu", "0", "1", "0,1"]
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

    var cl = mk("span", "cl");
    cl.textContent = r.l || "";
    row.appendChild(cl);

    var cc = mk("div", "cc");

    if (r.t === "path") {
      /* 路径行：标签已在 .cl，这里只放值 + 选目录按钮 */
      row.className += " cfgpathrow";
      var box = mk("div", "inp mono pathinp cfgpath");
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
      (r.o || []).forEach(function (o) {
        var op = document.createElement("option");
        op.textContent = o;
        if (String(o) === String(r.v)) op.selected = true;
        sel.appendChild(op);
      });
      cc.appendChild(sel);
    } else if (r.t === "sw") {
      var sw = mk("div", "sw" + (r.v ? " on" : ""));
      sw.setAttribute("role", "switch");
      sw.tabIndex = 0;
      cc.appendChild(sw);
    }

    row.appendChild(cc);
    row.appendChild(hint(r));
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

  /* 把补齐块挂到 #cfgBox 末尾（幂等） */
  api.mount = function () {
    var box = document.getElementById("cfgBox");
    if (!box) return false;
    if (box.querySelector('[data-part="train-extra"]')) return true;
    var wrap = mk("div", "train-extra");
    wrap.setAttribute("data-part", "train-extra");
    renderGroups(wrap, EXTRA_GROUPS);
    renderGroups(wrap, EXTRA_SELECT_GROUPS);
    box.appendChild(wrap);
    bind(wrap);
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
