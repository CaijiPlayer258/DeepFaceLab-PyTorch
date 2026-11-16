  /* ---------- 参数设置（照 4.0 training_config_page 的分组） ---------- */
  var CFG_GROUPS = [
    { t: "模型架构参数", note: "创建后不可修改", ro: true, rows: [
      ["模型名称", "name"], ["分辨率", "resolution"], ["archi", "archi"],
      ["e_dims", "e_dims"], ["ae_dims", "ae_dims"], ["d_dims", "d_dims"], ["d_mask_dims", "d_mask_dims"]
    ]},
    { t: "数据集参数", rows: [
      { k: "src", l: "src 人脸集", t: "path", v: "workspace\\data_src\\" },
      { k: "dst", l: "dst 人脸集", t: "path", v: "workspace\\data_dst\\" },
      { k: "face_type", l: "人脸类型", t: "sel", ro: true, o: ["whole_face", "full_face", "half_face", "midfull_face", "head"], v: "whole_face",
        d: "新建模型时按这一项确定，建完就固定在模型里" }
    ]},
    { t: "训练参数", rows: [
      { k: "batch_size", l: "批次大小", t: "num", v: "8", d: "每次训练的样本数量" },
      { k: "target_batch", l: "目标批次", t: "num", v: "0",
        d: "想要的等效批次；= 批次大小 × 累积次数。0 = 关闭累积" },
      { k: "target_iter", l: "目标迭代数", t: "num", v: "0", d: "0 表示无限训练" },
      { k: "use_bf16", l: "训练精度", t: "seg", o: ["bf16", "fp32"], v: "bf16" },
      { k: "gradient_checkpointing", l: "梯度检查点", t: "sw", v: 0, d: "省显存，略慢" },
      { k: "clipgrad", l: "梯度裁剪", t: "sw", v: 1, d: "防止梯度爆炸" },
      { k: "models_opt_on_gpu", l: "优化器放显存", t: "sw", v: 1, d: "关掉可省显存" },
      { k: "use_fast_generator", l: "快速生成器", t: "sw", v: 0 },
      { k: "write_preview_history", l: "写入预览历史", t: "sw", v: 0, d: "预览图存到 _history" },
      { k: "pretrain", l: "预训练", t: "sw", v: 0, d: "只练通用特征" },
      { k: "pretrain_single_decoder", l: "预训练单解码器", t: "sw", v: 0, d: "预训练时只用一个解码器" },
      { k: "crash_threshold", l: "崩溃阈值", t: "num", v: "0.0", d: "loss 超过它视为崩溃，不覆盖备份" },
      { k: "log_code_stats", l: "Inter code 分布日志", t: "num", v: "0", d: "0 = 关闭，每 N 迭代输出一次" },
      { k: "max_backups", l: "最大备份数", t: "num", v: "3" },
      /* 下面三项引擎一直在读、也一直写在 dat 里，但参数页以前没有对应控件 ——
         用户只能直接改 dat 才能动它们，改前的摘要也不显示，等于隐形参数。 */
      { k: "nan_guard", l: "NaN 梯度保护", t: "sw", v: 0,
        d: "每步扫一遍梯度，出现 NaN 就跳过这一步（约慢 4%）" },
      { k: "write_train_log", l: "训练日志 CSV", t: "sw", v: 0,
        d: "在模型目录写 <模型名>_train_log.csv" },
      { k: "eager_mode", l: "即时模式(XLA)", t: "sw", v: 0,
        d: "仅 SAEHD；需要 torch_xla，普通训练保持关闭" }
    ]},
    { t: "优化器参数", rows: [
      { k: "optimizer", l: "优化器", t: "seg", o: ["adam", "adabelief", "lion"], v: "adam" },
      { k: "lr", l: "学习率", t: "num", v: "0.00005" },
      { k: "lr_policy", l: "学习率策略", t: "seg", o: ["None", "CosineAnnealingLR"], v: "None" },
      { k: "lr_cos", l: "余弦周期", t: "num", v: "0", d: "0 = 关闭" },
      { k: "lr_total_steps", l: "学习率总步数", t: "num", v: "0", d: "0 = 关闭" },
      { k: "lr_plateau_factor", l: "平台衰减系数", t: "seg", o: ["1.0", "0.5", "0.7", "0.3"], v: "1.0", d: "1.0 = 关闭" },
      { k: "lr_plateau_patience", l: "平台耐心", t: "num", v: "0", d: "连续 N 轮不降才衰减" },
      { k: "lr_plateau_min_ratio", l: "平台最小比例", t: "num", v: "0.1" },
      { k: "lr_warmup_steps", l: "LR 预热步数", t: "num", v: "0", d: "0 = 关闭；前 N 步线性升温" },
      { k: "lr_dropout", l: "LR Dropout", t: "seg", o: ["n", "y", "cpu"], v: "n" },
      { k: "pia_cycle_power", l: "PIA 周期", t: "num", v: "1.0" }
    ]},
    { t: "GAN 训练参数", rows: [
      { k: "gan_power", l: "GAN 强度", t: "num", v: "0.0", d: "0 = 关闭 GAN" },
      { k: "gan_patch_size", l: "GAN 块大小", t: "num", v: "32" },
      { k: "gan_dims", l: "GAN 维度", t: "num", v: "16" }
    ]},
    { t: "补丁参数", note: "0 = 关闭", rows: [
      { k: "true_face_power", l: "真脸强度", t: "num", v: "0.0", d: "CodeDiscriminator 真脸损失权重" },
      { k: "face_style_power", l: "面部风格强度", t: "num", v: "0.0", d: "风格迁移中面部的权重" },
      { k: "bg_style_power", l: "背景风格强度", t: "num", v: "0.0", d: "风格迁移中背景的权重" }
    ]},
    { t: "数据增强参数", rows: [
      { k: "rotation_range", l: "旋转范围", t: "num", v: "10" },
      { k: "scale_range", l: "缩放范围", t: "num", v: "0.15" },
      { k: "t_range", l: "偏移范围", t: "num", v: "0.05" },
      { k: "random_warp", l: "随机扭曲", t: "sw", v: 1,
        d: "对输入图像做弹性形变、重建目标不变：让模型在五官被扭曲的情况下也能解码出正常形状的人脸，减少 dst 五官形状对 src 解码结果的影响" },
      { k: "random_hsv_power", l: "随机 HSV 强度", t: "num", v: "0.0",
        d: "0.0~0.3 · 随机调整输入的 HSV：让模型从输入就学习更丰富的色彩，解出的人脸颜色也更丰富" },
      { k: "ct_mode", l: "颜色迁移", t: "sel", o: ["none", "rct", "lct", "mkl", "mkl-m", "idt", "idt-m", "sot-m", "mix-m", "mt", "lut"], v: "none",
        d: "把 src 的颜色映射到 dst 上：通过改变输入数据，让模型学习调色后的 src 图像，生成更贴合 dst 色彩的人脸" },
      { k: "random_occlusion", l: "随机遮挡", t: "sw", v: 0,
        d: "随机遮挡输入图像的一部分、重建目标不变：让模型在信息缺失（被挡住）的情况下也能解码出人脸" },
      { k: "random_noise", l: "随机噪点", t: "sw", v: 0,
        d: "给输入图像随机加噪声、重建目标不变：让模型在噪点严重时也能解码出清晰的人脸" },
      { k: "random_color_alg", l: "偏色算法", t: "sel", o: ["none", "direct", "rct", "mt", "lct"], v: "none",
        d: "对输入图像随机偏色、重建目标不变：让模型无论输入什么颜色的人脸，都只解码出正常颜色的人脸" },
      { k: "random_color_power", l: "偏色强度", t: "num", v: "0.0", d: "0.0~1.0" },
      { k: "edge_enhance_power", l: "边缘增强", t: "num", v: "0.0", d: "0.0~20.0" },
      { k: "eyes_scale", l: "眼睛缩放", t: "num", v: "1.0", d: "0.25~2.0" }
    ]},
    { t: "冻结层参数", rows: [
      { k: "freeze_encoder", l: "冻结编码器", t: "sw", v: 0 },
      { k: "freeze_decoder_mask", l: "冻结遮罩解码器", t: "sw", v: 0 },
      { k: "freeze_inter_AB", l: "冻结 inter_AB", t: "sw", v: 0 },
      { k: "freeze_inter_B", l: "冻结 inter_B", t: "sw", v: 0 }
    ]}
  ];


  /* ------------------------------------------------------------
   * 冻结层：按架构只留相关的行
   *   df     → 冻结中间 / 冻结输出
   *   liae   → 冻结 inter_AB / 冻结 inter_B
   *   amp    → inter_AB + inter_B + 冻结源 + 冻结目标
   *   DFLarge / LIAELarge 在 4.1 已经移除，它们的参数（瓶颈 / A遮罩 / B遮罩 /
   *   共享 / 瓶颈B）一律不显示 —— 留着只会让人以为能开。
   * null = 所有架构都有。
   * ------------------------------------------------------------ */
  var FREEZE_ARCH = {
    freeze_encoder: null,
    freeze_decoder_mask: null,
    freeze_inter_AB: ["liae", "amp"],
    freeze_inter_B: ["liae", "amp"],
    freeze_inter: ["df"],
    freeze_decoder_dst: ["df"],
    freeze_inter_src: ["amp"],
    freeze_inter_dst: ["amp"],
    freeze_bottleneck: [],
    freeze_decoderA_mask: [],
    freeze_decoderB_mask: [],
    freeze_bottleneck_AB: [],
    freeze_bottleneck_B: []
  };

  function archOf(archi) {
    var a = String(archi || "").toLowerCase();      /* liae-ud / df-ud / amp */
    if (a.indexOf("liae") === 0) return "liae";
    if (a.indexOf("df") === 0) return "df";
    if (a.indexOf("amp") === 0) return "amp";
    return "";
  }

  function filterFreezeRows(root, archi) {
    if (!root) return;
    var arch = archOf(archi);
    if (!arch) return;                              /* 拿不到架构就别乱删 */
    Array.prototype.forEach.call(root.querySelectorAll(".cfgrow[data-k]"), function (row) {
      var k = row.getAttribute("data-k");
      if (!(k in FREEZE_ARCH)) return;
      var allow = FREEZE_ARCH[k];
      if (allow === null) return;
      if (allow.indexOf(arch) < 0) row.remove();
    });
    /* 清完如果那张卡空了，就连卡一起收掉 */
    Array.prototype.forEach.call(root.querySelectorAll(".card"), function (c) {
      if (!c.querySelector(".cfgrow")) c.remove();
    });
  }

  window.__filterFreezeRows = filterFreezeRows;   /* mount() 在另一段 IIFE 里 */

  /* 训练中：给能热改的行挂一个「热」标记（只在这个模型真的在训练时才显示） */
  function markHotRows() {
    var box = $("cfgBox");
    if (!box) return;
    var m = window.__cfgModel;
    var lt = window.__lastTrain;
    var live = !!(lt && lt.running && m && lt.name === m.name);
    /* 训练在跑时主按钮的语义变了：不是「开始训练」，而是「应用参数」 */
    var startBtn = $("cfgStart");
    if (startBtn) startBtn.textContent = live ? "应用参数" : "开始训练";
    box.querySelectorAll(".cfgrow").forEach(function (row) {
      var lb = row.querySelector(".cl");
      if (!lb) return;
      /* 每一行都留一个固定宽的槽：有没有「热」都占同样的位置，
         这样说明文字的左边缘在整页是一条竖线（不然只有带热字的行被顶出去）。 */
      var slot = row.querySelector(".hotslot");
      if (!slot) {
        slot = document.createElement("span");
        slot.className = "hotslot";
        lb.insertAdjacentElement("afterend", slot);
      }
      var k = row.getAttribute("data-k");
      var tag = slot.querySelector(".hottag");
      var want = !!(live && window.HOT_KEYS && k && window.HOT_KEYS[k]);
      if (!want) {
        if (tag) tag.remove();
        row.classList.remove("hot");
        return;
      }
      row.classList.add("hot");
      if (!tag) {
        var s = document.createElement("span");
        s.className = "hottag";
        s.textContent = "热";
        s.title = "训练中改这一项会立即生效";
        slot.appendChild(s);
      }
    });
  }
  window.markHotRows = markHotRows;

  function openCfg(m) {
    var box = $("cfgBox");
    if (!box) return;
    window.__cfgModel = m || window.__cfgModel || null;   /* 冻结层过滤要知道当前架构 */
    window.__cfgArchi = (m && m.archi) || window.__cfgArchi || "";

    /* 补充参数不再塞进 CFG_GROUPS：TrainExtra.mount() 会把它们并进同名卡片，
       这里只管渲染原有的分组，避免同一批参数出现两遍。 */

    box.innerHTML = "";

    /* 上一个模型的迁移提示不能留在页面上：先清掉，等 /config 回来再按这次的结果画。
       （新格式 dat 的 legacy.detected=false，一路清空，页面上什么都不多出来。） */
    try {
      if (window.LegacyNotice) window.LegacyNotice.clear(document.getElementById("cfgLegacy"));
    } catch (e) { /* 提示层失败不能挡住参数页 */ }

    CFG_GROUPS.forEach(function (g) {
      /* 标了 ro 的分组（模型架构参数）内容和顶部"模型结构"只读卡完全重复：
         名字/分辨率/archi/e_dims/ae_dims/d_dims/d_mask_dims 一模一样。
         以前两处都画，页面上连着两块一样的信息，看着就是"混乱"。 */
      if (g.ro) return;
      var card = document.createElement("div");
      card.className = "card";
      var hd = document.createElement("div");
      hd.className = "cardhd";
      hd.innerHTML = "<b>" + g.t + "</b>" + (g.note ? '<span class="hint">' + g.note + "</span>" : "");
      card.appendChild(hd);

      (g.rows || []).forEach(function (r) {
        var row = document.createElement("div");
        row.className = "cfgrow" + ((g.ro || (!Array.isArray(r) && r.ro)) ? " ro" : "");

        /* 每行都带上 data-k：既给搜索/定位用，也让补齐块能按锚点插入
           （之前行上没有 data-k，after:"dst" 找不到锚点，只能落到卡片末尾）。 */
        if (r.k) row.setAttribute("data-k", r.k);

        if (!Array.isArray(r) && r.t === "path") {
          row.className += " cfgpathrow";
          /* 值用 <input>：路径行本身就该能直接改字（以前是 <span>，
             点上去没反应，看起来像"编辑框但打不了字"）。 */
          row.innerHTML = '<div class="inp mono pathinp cfgpath" style="flex:1 1 auto;width:auto;min-width:0">' +
            '<span class="lb">' + r.l + "</span>" +
            '<input class="pathval" spellcheck="false" autocomplete="off">' +
            '<span class="pill iconpill" title="选择目录"><svg class="ic"><use href="#i-folder"/></svg></span>' +
            "</div>";
          var pv = row.querySelector(".pathval");
          pv.value = r.v || "";
          if (r.k) pv.dataset.cfg = r.k;
          row.querySelector(".iconpill").addEventListener("click", function () {
            toast("选择目录", "文件选择器还没接", "info");
          });
          card.appendChild(row);
          return;
        }

        if (Array.isArray(r)) {
          var val = m ? (m[r[1]] || "—") : "—";
          row.innerHTML = '<span class="cl">' + r[0] + '</span><span class="cv">' + val + "</span>";
          card.appendChild(row);
          return;
        }

        row.innerHTML = '<span class="cl">' + r.l + "</span>" +
          '<span class="cd">' + (r.d || "") + "</span>" +
          '<span class="cc"></span>';
        var cc = row.querySelector(".cc");

        if (r.t === "num") {
          var inp = document.createElement("input");
          inp.className = "inp";
          inp.value = r.v;
          if (r.k) inp.dataset.cfg = r.k;
          inp.dataset.v0 = String(inp.value);
          cc.appendChild(inp);
        } else if (r.t === "seg") {
          var seg = document.createElement("div");
          seg.className = "seg";
          if (r.k) seg.dataset.cfg = r.k;          /* 之前漏了这行：seg 既不被回填也不进 patch */
          r.o.forEach(function (o, i) {
            var b = document.createElement("b");
            b.textContent = o;
            b.dataset.v = o;                       /* 业务值取这个，别取 textContent（会被翻译） */
            if (i === 0) { b.className = "on"; seg.dataset.v0 = String(o); }
            b.addEventListener("click", function () {
              Array.prototype.forEach.call(seg.children, function (x) { x.classList.remove("on"); });
              b.classList.add("on");
            });
            seg.appendChild(b);
          });
          cc.appendChild(seg);
        } else if (r.t === "sel") {
          var sel = document.createElement("select");
          sel.className = "mgsel";
          if (r.k) sel.dataset.cfg = r.k;
          (r.o || []).forEach(function (o) {
            var op = document.createElement("option");
            /* o 可以是字符串，也可以是 { v, l }：显示名与存的值分开 */
            var ov = (o && typeof o === "object") ? o.v : o;
            var ol = (o && typeof o === "object") ? o.l : o;
            op.value = String(ov);
            op.textContent = ol;
            sel.appendChild(op);
          });
          cc.appendChild(sel);
          /* 选中值要在**挂到文档之后**设：先设 value 再 append，浏览器会把选中态重置掉，
             结果是下拉框空着、提交时 face_type 变空串。 */
          sel.value = String(r.v);
          if (sel.selectedIndex < 0) sel.selectedIndex = 0;
          sel.dataset.v0 = String(sel.value);
          if (r.ro) {
            /* 结构性参数（face_type）：值照样从 dat 回显，但不许改 ——
               以前这里是可点的下拉框，点了保存却永远存不进去（假象）。 */
            sel.disabled = true;
            sel.dataset.ro = "1";
            sel.title = "创建后不可改：要换人脸类型只能新建模型";
          }
        } else if (r.t === "sw") {
          var sw = document.createElement("div");
          sw.className = "sw" + (r.v ? " on" : "");
          if (r.k) sw.dataset.cfg = r.k;
          sw.dataset.v0 = sw.classList.contains("on") ? "1" : "0";
          /* data-bound 必须打上：TrainExtra.mount() 会给「所有没绑过的 .sw」再挂一个
             click → 两个监听各 toggle 一次，等于抵消 —— 表现就是"开关拨不动"。 */
          sw.setAttribute("data-bound", "1");
          sw.addEventListener("click", function () { sw.classList.toggle("on"); });
          cc.appendChild(sw);
        }
        card.appendChild(row);
      });

      box.appendChild(card);
    });

    /* 结构卡在这里补一次：以前只靠 window.openCfg 的外层包装，
       而那个包装是在脚本加载时就抓走的 —— 只要 openCfg 定义得比包装晚，
       或者别的模块又包了一层，结构卡就不会出现（页面上只剩搜索框 + 参数卡）。
       现在由 openCfg 自己负责，谁调用都有效。 */
    try { filterFreezeRows(box, (m && m.archi) || window.__cfgArchi || ""); } catch (e) { /* 过滤失败就全留着 */ }

    try { if (window.__enhanceSelects) window.__enhanceSelects(box); } catch (e) { /* 增强失败不影响页面 */ }

    try {
      if (typeof nmReadonlyPanel === "function") {
        nmReadonlyPanel(m || window.__cfgModel || (window.MODELS || [])[0] || null);
      }
    } catch (e) { /* 结构卡失败不影响参数页 */ }

    var sub = $("cfgSub");
    if (sub) sub.textContent = m ? ("正在配置 " + m.name + "，确认后开始训练。") : "训练前确认参数。";
    if ($("cfgHint") && m) $("cfgHint").textContent = "当前 iter " + fmt(m.iter) + " · " + (m.archi || "");
    if (window.I18N) window.I18N.refresh();
    go("train-cfg");

    /* 回填一批控件：opts = /config 返回的 dat options。
       onlyNew=true 时只填「还没有基线」的行 —— TrainExtra 的「设备」卡是
       openCfg 之后 80ms 才挂上来的（见 mount 里的 setTimeout），这批行在第一次
       回填时还不存在，不补一次的话它们永远停在写死的默认值上：
       dat 里 device="auto"，下拉框却一直显示 "0"，用户以为改完没生效。 */
    function fillCfgRows(opts, onlyNew) {
      if (!opts) return 0;
      var filled = 0;
      box.querySelectorAll("[data-cfg]").forEach(function (el) {
        if (onlyNew && el.dataset.v0 !== undefined) return;
        var k = el.dataset.cfg, v = opts[k];
        if (k === "target_batch" && (v === undefined || v === null || Number(v) === 0)) {
          var _b0 = Number(opts.batch_size) || 1, _a0 = Number(opts.accum_steps) || 1;
          v = _a0 > 1 ? _b0 * _a0 : 0;
        }
        if (v === undefined) {
          /* dat 里没有这个键（老模型没存过 / 新建时没写）：
             标记一下，别给它记基线 —— 否则用户改成界面默认值会被判成"没改"，
             这个值永远存不进模型（"最大备份数填 3 不生效"就是这么来的）。 */
          el.dataset.missing = "1";
          return;
        }
        delete el.dataset.missing;
        filled++;
        if (el.tagName === "INPUT") el.value = v;
        else if (el.tagName === "SELECT") {
          /* 短码必须照 4.0 引擎的原表（Model_SAEHD:195 就是 ["h","mf","f","wf","head"]）。
             以前这里写的是 wf/ff/hf：face_type="f" 的模型在框里显示成第一项
             whole_face —— 只读行显示错的值比空白更误导。ff/hf 留着容忍早期写过的值。 */
          var _ALIAS = { h: "half_face", mf: "midfull_face", f: "full_face",
                         wf: "whole_face", head: "head",
                         ff: "full_face", hf: "half_face" };
          var sv = String(v);
          el.value = sv;
          if (el.selectedIndex < 0 && _ALIAS[sv.toLowerCase()]) el.value = _ALIAS[sv.toLowerCase()];
          /* 设备：必须在下面那句兜底之前过一遍守卫 —— 一旦 selectedIndex<0 被改成 0，
             「存的是 GPU 1 但本机只有 GPU 0」就变成了「用户选了自动」，看不出来了。 */
          if (el.dataset.cfg === "device" && window.__devGuard) window.__devGuard(el, sv);
          if (el.selectedIndex < 0) el.selectedIndex = 0;
        }
        else if (el.classList.contains("sw")) el.classList.toggle("on", !!v);
        else if (el.classList.contains("seg")) {
          /* 按 data-v 比，不看 textContent —— 切到英文后文案变了，比文本永远不匹配 */
          var hit = false;
          Array.prototype.forEach.call(el.children, function (b) {
            var bv = String(b.dataset.v !== undefined ? b.dataset.v : b.textContent);
            /* 数字要按数值比：dat 里存的是 float 1.0，分段控件的 token 是字符串 "1.0"，
               直接比字符串会一个都不亮（表现是这一行看着"没选中"）。 */
            var same = bv === String(v) ||
              (bv !== "" && String(v).trim() !== "" &&
               isFinite(Number(bv)) && isFinite(Number(v)) && Number(bv) === Number(v));
            if (same) hit = true;
            b.classList.toggle("on", same);
          });
          /* 存的是布尔（比如 use_bf16 -> 1/0），按 开/关 兜一次 */
          if (!hit && el.children.length === 2) {
            var onIdx = (v === true || v === 1 || v === "1") ? 0 : 1;
            Array.prototype.forEach.call(el.children, function (b, i) {
              b.classList.toggle("on", i === onIdx);
            });
          }
        }
      });

      /* 「已读入 N 项」按当前页面上真实填上的行数算：补充块（TrainExtra）的行
         可能比这次回填晚到，晚到的那批由下面的 __fillCfgRows 再补一次。 */
      var hint = document.getElementById("cfgHint");
      if (hint) {
        /* 先记下原始文案：openCfg 每重画一次都会拉一遍配置，直接往后追加会叠成
           "已读入 46 项真实参数 · 已读入 46 项真实参数"。 */
        if (!hint.dataset.base) {
          hint.dataset.base = (hint.textContent || "").replace(/\s*·\s*已读入 \d+ 项真实参数\s*$/, "");
        }
        var total = 0;
        box.querySelectorAll("[data-cfg]").forEach(function (el) {
          if (el.dataset.missing !== "1" && el.dataset.v0 !== undefined) total++;
        });
        hint.textContent = hint.dataset.base + " · 已读入 " + Math.max(filled, total) + " 项真实参数";
      }
      return filled;
    }
    window.__fillCfgRows = fillCfgRows;   /* TrainExtra 补挂的行要用它再填一次 */
    window.__cfgOpts = null;

    /* 拉真实的 *_data.dat options 回填 —— 页面上的值不再是写死的默认值 */
    if (m) {
      fetch("/api/models/" + encodeURIComponent(m.name) + "/config")
        .then(function (r) { return r.ok ? r.json() : null; })
        .then(function (d) {
          if (!d || !d.options) return;
          var opts = d.options;
          window.__cfgOpts = opts;         /* 给晚挂的行（设备卡）留一份 */
          fillCfgRows(opts, false);
          /* 回填完真实参数，把「初始值」重记一遍：之后只有用户真改过的才算改过 */
          if (window.syncCfgBaseline) window.syncCfgBaseline();

          /* TF 时代（DFL 1.x/2.0）的 dat：后端（server/_legacy.py）已经把 options 翻成
             4.1 的键，这里只负责把「迁了几项、还要不要转权重」放在页面顶部，
             一行 + 可展开明细。新格式 dat 传进来的 legacy.detected=false，这行不会出现。 */
          try {
            if (window.LegacyNotice) {
              window.LegacyNotice.render(document.getElementById("cfgLegacy"), d.legacy);
            }
          } catch (e) { console.warn("legacy notice failed", e); }

          /* 设备下拉的选项是 /api/devices 回来后才填的，可能在这次回填之后：
             等它到货时把「还没有基线」的行再填一遍（否则设备行永远停在默认值）。 */
          setTimeout(function () {
            var n = fillCfgRows(window.__cfgOpts, true);
            if (n && window.syncCfgBaseline) window.syncCfgBaseline();
          }, 0);
        })
        .catch(function (e) { console.warn("config load failed", e); });
    }

    /* 补充参数并由 TrainExtra.mount() 并入同名卡片。
       以前是靠包装 window.openCfg 来实现的，但那份包装没生效（__trainExtra 一直是 false），
       于是每次重渲染都把补充项弄丢。直接在这里调，顺序上不可能错。 */
    if (window.TrainExtra && typeof window.TrainExtra.mount === "function") {
      try { window.TrainExtra.mount(); } catch (e) {}
    }
    /* 补充区的行是上面 mount() 才插进来的，第一次标记扫不到 —— 渲染完再补一次 */
    setTimeout(function () { if (window.markHotRows) window.markHotRows(); }, 0);
  }
  window.openCfg = openCfg;
