/* ===== 遮罩绘制页（子智能体 C）===== */
/* ===== 训练参数补齐（子智能体 D）===== */
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
        { k: "eyes_mouth_prio", l: "嘴眼优先", t: "sw", v: 0, d: "优先训练嘴部和眼部区域" },
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
        { k: "uniform_yaw", l: "均匀偏航", t: "sw", v: 0, d: "侧脸少的 faceset 用，让 yaw 分布均匀" }
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
      /* 预训练集是路径，属于"数据集从哪来"，不是优化器的事 ——
         单独成组并进「数据集参数」，插在 dst 人脸集后面（after: "dst"）。 */
      t: "数据集参数 · 补充",
      rows: [
        { k: "pretrain_dataset_path", l: "预训练集", t: "path", v: "", after: "dst", full: true,
          d: "多身份人脸集目录，留空 = 用 dst 人脸集兜底" }
      ]
    }
  ];

  /* ------------------------------------------------------------
   * 二、必须用 select 的两行（候选项 > 4）
   * ------------------------------------------------------------ */
  var EXTRA_SELECT_GROUPS = [
    {
      t: "设备",
      pin: "top",                 /* 用哪个设备训练是最需要一眼看到的信息，别沉到页尾 */
      noheader: true,             /* 标题挪到行里，卡片不再单独占一行标题 */
      rows: [
        {
          /* 一行就够：左边"设备"，右边下拉框 */
          /* 默认 "自动选择"：dat 里没有 device 这个键时，引擎就是自动挑卡。
             以前这里写死 "0"，看起来像"用户已经选了 0 号卡"——是骗人的。 */
          k: "device", l: "设备", t: "sel", v: "auto", d: "", titlelabel: true,
          /* 选项在运行时由 /api/devices 填（GPU 型号 / CPU 线程数），
             这里的只是网还没回来时的兜底 */
          o: [{ v: "auto", l: "自动选择" }, { v: "cpu", l: "CPU" }, { v: "0", l: "GPU 0" }]
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

    /* nolabel：卡片标题已经说明了这是什么，行内再写一遍就是重复 */
    if (!r.nolabel) {
      var cl = mk("span", "cl");
      /* titlelabel：这一行没有卡片标题，标签本身充当标题 —— 用标题的字体和颜色 */
      if (r.titlelabel) cl.className = "cl ttl";
      cl.textContent = r.l || "";
      row.appendChild(cl);
    }

    var cc = mk("div", "cc");

    if (r.t === "path") {
      row.className += " cfgpathrow";
      var box = mk("div", "inp mono pathinp cfgpath");
      if (r.full) {
        /* full：和宿主「src 人脸集 / dst 人脸集」完全同款 —— 标签在框内、整行就是输入框。
           预训练集放在它们下面，三种路径长得一致才不会显得突兀。 */
        row.innerHTML = "";
        box.style.cssText = "flex:1 1 auto;width:auto;min-width:0";   /* 铺满整行，和宿主 src/dst 一致 */
        var lb = mk("span", "lb");
        lb.textContent = r.l || "";
        box.appendChild(lb);
        var val2 = document.createElement("input");
        val2.className = "pathval";
        val2.spellcheck = false;
        val2.autocomplete = "off";
        val2.value = r.v || "";
        if (r.k) val2.dataset.cfg = r.k;
        box.appendChild(val2);
        box.innerHTML += FOLDER_SVG;
        /* 直接挂在行上，不套 .cc —— .cc 是"控件靠右"的容器（宽 auto），
           把整行的路径框塞进去只会缩成一小截。 */
        row.appendChild(box);
        if (r.d) row.title = r.d;
        return row;
      }
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
      /* data-cfg 必须打上：没有它 openCfg 回填扫不到、collectCfgPatch 也不报，
         表现就是"这个值改了刷新就回默认"（补齐块的 num/seg/sw 以前全都漏了）。 */
      if (r.k) inp.dataset.cfg = r.k;
      inp.value = r.v == null ? "" : String(r.v);
      inp.setAttribute("inputmode", "decimal");
      inp.spellcheck = false;
      cc.appendChild(inp);
    } else if (r.t === "seg") {
      var seg = mk("div", "seg");
      if (r.k) seg.dataset.cfg = r.k;
      (r.o || []).forEach(function (o, i) {
        var b = mk("b");
        b.textContent = o;
        b.dataset.v = String(o);            /* 业务值取这个，别取 textContent（会被翻译） */
        if (String(o) === String(r.v) || (i === 0 && r.v == null)) b.className = "on";
        seg.appendChild(b);
      });
      cc.appendChild(seg);
    } else if (r.t === "sel") {
      var sel = mk("select", "mgsel");
      if (r.k) sel.dataset.cfg = r.k;
      var opts = (r.k === "device" && window.__DEVOPTS) ? window.__DEVOPTS : (r.o || []);
      opts.forEach(function (o) {
        var op = document.createElement("option");
        var ov = (o && typeof o === "object") ? o.v : o;
        var ol = (o && typeof o === "object") ? o.l : o;
        op.value = String(ov);
        op.textContent = ol;
        if (String(ov) === String(r.v)) op.selected = true;
        sel.appendChild(op);
      });
      if (sel.selectedIndex < 0) sel.selectedIndex = 0;
      cc.appendChild(sel);
      /* 设备行：控件旁边常驻「当前可用 GPU 列表」（值由 /api/devices 提供） */
      if (r.k === "device" && window.__devGuard) window.__devGuard(sel);
    } else if (r.t === "sw") {
      var sw = mk("div", "sw" + (r.v ? " on" : ""));
      if (r.k) sw.dataset.cfg = r.k;
      sw.dataset.v0 = sw.classList.contains("on") ? "1" : "0";
      sw.setAttribute("role", "switch");
      sw.tabIndex = 0;
      cc.appendChild(sw);
    }

    /* 顺序必须是 标签 → 说明 → 控件，和宿主页面（renderCfg）一致。
       以前是 标签 → 控件 → 说明，并进"训练参数"卡之后开关会跑到文字左边，
       整行看着像往左弯了一下。 */
    row.appendChild(hint(r));
    row.appendChild(cc);
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
    if (g.noheader) {          /* 标题挪进行里，卡片本身不要标题 */
      (g.rows || []).forEach(function (r) { card.appendChild(buildRow(r)); });
      return card;
    }
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

  /* 补齐参数不再单独成卡：优先并进已有的同名分组（训练参数 / 补丁参数 / 数据增强参数…）。
     只有找不到归属的组（比如 PIA）才退回独立卡片。 */
  /* 少数补充组没有同名卡片，显式指个归属 */
  var MERGE_INTO = { "PIA": "优化器参数" };

  function findCardByTitle(box, base) {
    var cards = box.querySelectorAll(".card");
    var want = [base, base + "参数"];
    var fallback = null;
    for (var i = 0; i < cards.length; i++) {
      var b = cards[i].querySelector(".cardhd b");
      if (!b) continue;
      var t = (b.textContent || "").trim();
      if (want.indexOf(t) >= 0) return cards[i];
      if (!fallback && t.indexOf(base) === 0) fallback = cards[i];
    }
    return fallback;
  }

  api.mount = function () {
    var box = document.getElementById("cfgBox");
    if (!box) return false;

    /* 可重复调用：先清掉上一次并入的行和残留的包装层。
       以前只在开头判断"有没有 train-extra 包装层"，结果 openCfg 重画后
       包装层消失、行还留着 → 每重画一次就多一份，页面越来越乱。 */
    Array.prototype.forEach.call(box.querySelectorAll('[data-extra="1"]'), function (r) {
      if (r.parentNode) r.parentNode.removeChild(r);
    });
    Array.prototype.forEach.call(box.querySelectorAll('[data-part="train-extra"]'), function (w) {
      if (w.parentNode) w.parentNode.removeChild(w);
    });

    var groups = (EXTRA_GROUPS || []).concat(EXTRA_SELECT_GROUPS || []);
    var seen = {};
    var leftover = [];

    function mergeInto(card, g) {
      /* 宿主卡里已经有的键不再补：Host 的"设备"卡自带 device 行，
         补齐块里也有一个 device 行，不加这层判断就会出现两行一样的 */
      var have = {};
      Array.prototype.forEach.call(card.querySelectorAll(".cfgrow[data-k]"), function (r) {
        var k = r.getAttribute("data-k");
        if (k) have[k] = 1;
      });
      (g.rows || []).forEach(function (r) {
        if (r.k && have[r.k]) return;
        var row = buildRow(r);
        row.setAttribute("data-extra", "1");   /* 打标记，下次 mount 好清理 */
        /* after: "<data-k>" —— 插到那一行后面，而不是整卡末尾。
           例：预训练集要跟在 dst 人脸集下面，别排到人脸类型之后。 */
        var anchor = r.after ? card.querySelector('.cfgrow[data-k="' + r.after + '"]') : null;
        if (anchor && anchor.nextSibling) card.insertBefore(row, anchor.nextSibling);
        else if (anchor) card.appendChild(row);
        else card.appendChild(row);
      });
      bind(card);
    }

    /* 同名组先归并 rows 再并卡：EXTRA_GROUPS 和 EXTRA_SELECT_GROUPS 里各有一个
       "数据增强 · 补充"，以前第二组被整组跳过 —— preview_grade_alg 因此从没出现在页面上。
       归并后顺序不变（先出现的组在前），同一张卡仍然只并一次。 */
    var byBase = [], order = {};
    groups.forEach(function (g) {
      var base = String(g.t || "").split(" · ")[0].trim();
      base = MERGE_INTO[base] || base;
      if (order[base] === undefined) {
        order[base] = byBase.length;
        /* pin / noheader 要跟着走：丢了它们设备卡就不置顶、还会多出一个标题 */
        byBase.push({ base: base, g: { t: g.t, note: g.note, pin: g.pin, noheader: g.noheader, rows: [] } });
      }
      var slot = byBase[order[base]];
      (g.rows || []).forEach(function (r) { slot.g.rows.push(r); });
    });
    byBase.forEach(function (it) {
      var base = it.base;
      if (seen[base]) return;
      seen[base] = 1;
      var card = findCardByTitle(box, base);
      if (card) mergeInto(card, it.g);
      else leftover.push({ base: base, g: it.g });
    });

    if (!leftover.length) {
      if (window.I18N && window.I18N.refresh) window.I18N.refresh();
      return true;
    }

    /* 宿主还有卡是稍后才挂上来的（"设备"卡就是 openCfg 之后才出现的），
       等一拍再找一次；仍找不到才自己建卡，避免和宿主重复。 */
    setTimeout(function () {
      var still = [];
      leftover.forEach(function (it) {
        var card = findCardByTitle(box, it.base);
        if (card) mergeInto(card, it.g);
        else still.push(it.g);
      });
      if (!still.length) return;

      /* pin: "top" 的组（设备）插到「训练参数」前面 ——
         "用什么设备训练"是用户最需要一眼确认的，不该排在页尾。 */
      var pinned = still.filter(function (g) { return g.pin; });
      var normal = still.filter(function (g) { return !g.pin; });

      function makeWrap(list) {
        var w = mk("div", "train-extra");
        w.setAttribute("data-part", "train-extra");
        list.forEach(function (g) { w.appendChild(buildGroup(g)); });
        bind(w);
        try { if (window.__enhanceSelects) window.__enhanceSelects(w); } catch (e) {}
        return w;
      }

      if (pinned.length) {
        var wp = makeWrap(pinned);
        var anchor = findCardByTitle(box, "训练参数") || box.firstChild;
        box.insertBefore(wp, anchor);
      }
      if (normal.length) box.appendChild(makeWrap(normal));
      var archNow = (window.__cfgModel && window.__cfgModel.archi) || window.__cfgArchi || "";
      try { if (window.__filterFreezeRows) window.__filterFreezeRows(box, archNow); } catch (e) {}

      /* 关键：这批行是 openCfg 的 /config 回填**之后**才挂上来的（晚 80ms），
         第一次回填扫不到它们 —— 于是「设备」下拉永远停在写死的默认值，
         dat 里 device="auto" 页面上却显示 "0"。这里补一次，只填还没基线的行。 */
      try {
        if (window.__fillCfgRows && window.__cfgOpts) {
          var _n = window.__fillCfgRows(window.__cfgOpts, true);
          if (_n && window.syncCfgBaseline) window.syncCfgBaseline();
        }
      } catch (e) {}
      if (window.I18N && window.I18N.refresh) window.I18N.refresh();
    }, 80);

    if (window.I18N && window.I18N.refresh) window.I18N.refresh();
    return true;
  };

  /* 渲染完由 openCfg 直接调 TrainExtra.mount()（见 openCfg 末尾）。
     这里以前还包了一层 window.openCfg，但那份包装从来没被调用过
     —— 留着只会让人以为有两套挂载机制，已经删掉。 */

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
