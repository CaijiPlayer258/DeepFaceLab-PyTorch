/* ===== 合成工作室 · 人脸库/导出（B）===== */
/* ============================================================
   合成工作室 · 页内二级页控制（人脸库 / 导出设置）
   暴露：window.MergePages = { show(name), settings(), ... }
   参数名全部取自 4.0：
     routes_export.py ExportStartRequest
       image_format 'jpg' | 'png' · encoder 'h264_nvenc' | 'libx264'
       hwaccel 'auto' | '' · output_path str · num_workers 0~16
   本文件只写 DOM，不发请求；点击「开始导出」派发事件交给主 app。
   ============================================================ */
(function () {
  'use strict';

  var WORKERS_MIN = 0;                 // num_workers 下限（0 = 自动）
  var WORKERS_MAX = 16;                // num_workers 上限

  function one(sel, root) { return (root || document).querySelector(sel); }
  function all(sel, root) {
    return Array.prototype.slice.call((root || document).querySelectorAll(sel));
  }

  var MergePages = {
    current: 'main',

    /* 切换页面：name ∈ 'facedb' | 'settings' | 'main'
       'main' 表示收起所有二级页，回到合成主视图 */
    show: function (name) {
      name = String(name || 'main');
      var target = null;
      all('.mgview').forEach(function (el) {
        /* 页容器在 index.html 里是 .mgview[data-view]（main / facedb / settings /
           progress / project），这里以前只认 data-page —— 于是 .on 和 .mgp-on
           从来没挂上过、current 恒为 'main'，全靠 mgStep 的 .in 兜着。两种都认。 */
        var view = el.getAttribute('data-view') || el.getAttribute('data-page') || '';
        var on = (view === name) && (name !== 'main');
        el.classList.toggle('on', on);
        if (on) {
          target = el;
          el.setAttribute('aria-hidden', 'false');
        } else {
          el.setAttribute('aria-hidden', 'true');
        }
      });

      /* 主视图让位（CSS：#p-merge.mgp-on > .mgbody / .mgtl 隐藏）。
         name==='main' 一律清掉：主视图自己也带 data-view='main'，留着 .mgp-on
         就会命中 [data-view='main']{display:none}，回主视图直接一片空白。 */
      var merge = document.getElementById('p-merge');
      if (merge) merge.classList.toggle('mgp-on', name !== 'main' && !!target);

      this.current = (name === 'main' || !target) ? 'main' : name;
      return this.current;
    },

    /* 这里以前有 buildFaceGrid(n) / setFaceCount(n)：往 #mgFaceGrid 里塞 n 个
       「序号 + 棋盘格」占位缩略图，再 grid.hidden = (n === 0) 把它显示出来。
       那是 4.0 人脸库的**旧占位网格**的渲染器，页面重做成 .fdb（face_db.js 画真数据）之后
       只剩它自己还在跑 —— 而 #mgFaceGrid 这个 id 现在属于**舞台里的单脸六宫格**
       （index.html 里 .mgfgrid#mgFaceGrid），于是 getElementById 抓到的正是舞台那一块：
       初始化时 24 个 .mgp-face 被写进舞台，hidden 又被抹成 false，
       .mgfgrid 是 position:absolute/inset:0 的 flex column —— 合成器一打开，
       舞台正中就顶上来一列 1..24 的编号空方块，把画面和空态全盖住。
       人脸库现在只有一侧真源（face_db.js + /api/faces），占位网格连同渲染器一起删掉：
       不生成，不是隐藏。别再往 #mgFaceGrid 里写任何人脸库的东西。 */

    /* 读取某个 seg 当前选中项的 data-v（没有 data-v 时退化成文字） */
    readSeg: function (key) {
      var seg = one('#p-merge .mgview .seg[data-k="' + key + '"]');
      if (!seg) return '';
      var on = seg.querySelector('b.on');
      if (!on) return '';
      var v = on.getAttribute('data-v');
      return v === null ? on.textContent.trim() : v;
    },

    /* 采集导出参数：字段名与 /api/export/start 一致 */
    settings: function () {
      var out = one('#mgOutPath');
      /* 「输出目录」这一行现在是可编辑的路径框（原来是 readonly 回显）。
         默认值仍由 mgFillExportSettings 填（= 后端本来就会用的位置），
         所以只有用户真改过（__mgDefOutEdited）且填的是 Windows 绝对路径，
         才把它当 output_path 发出去 —— 不动它就和以前完全一致。 */
      var dirOut = one('#mgDefOut');
      var dirVal = dirOut ? String(dirOut.value || '').trim() : '';
      var dirUsed = !!window.__mgDefOutEdited && /^[A-Za-z]:[\\/]|^\\\\/.test(dirVal);
      var workers = one('#mgWorkers');
      var nw = parseInt(workers ? workers.value : '0', 10);
      if (isNaN(nw)) nw = 0;
      nw = Math.min(WORKERS_MAX, Math.max(WORKERS_MIN, nw));
      var byId = function (id) {
        var el = document.getElementById(id);
        return el ? String(el.value || '').trim() : '';
      };
      var segOf = function (dk) {
        var seg = document.querySelector('#p-merge .seg[data-k="' + dk + '"] b.on');
        return seg ? seg.textContent.trim() : '';
      };
      var tiny = function (label) {
        var rows = document.querySelectorAll('#p-merge .mgright .mgrow');
        for (var i = 0; i < rows.length; i++) {
          var f = rows[i].querySelector('.f');
          if (f && f.textContent.indexOf(label) >= 0) {
            var inp = rows[i].querySelector('input');
            return inp ? (parseFloat(inp.value) || 0) : 0;
          }
        }
        return 0;
      };
      return {
        image_format: this.readSeg('image_format') || 'jpg',
        encoder: this.readSeg('encoder') || 'h264_nvenc',
        hwaccel: this.readSeg('hwaccel'),
        output_path: (out ? out.value.trim() : '') || (dirUsed ? dirVal : ''),
        num_workers: nw,
        /* 检测 */
        detector: byId('mgDetector'),
        face_type: byId('mgFaceType'),
        redetect: segOf('detect'),
        /* 合成 */
        mode: byId('mgMode'),
        mask_source: segOf('seg'),
        mask_mode: segOf('mask'),
        color_transfer: byId('mgColor'),
        /* 融合方式的真源 = 「模式」下拉（4 个）；分段只是快捷方式 */
        blend: (function () {
          var m = String(byId('mgMode') || '');
          if (m === 'overlay' || m === 'hist-match' || m === 'seamless'
              || m === 'seamless-hist-match') return m;
          return segOf('blend') || 'seamless';
        })(),
        /* 新契约：ct_mode / blend / mask_mode(int) / mask_erode / mask_blur。
           导出和预览共用这一份，别让两条路各说一套。
           （blend 在上面只写一次 —— 同一个对象里写两次，后面的会把前面的盖掉。） */
        ct_mode: byId('mgColor') || 'mt',
        mask_mode_num: ({ 'prd': 1, 'dst': 2, 'prd×dst': 3 })[segOf('mask')] || 3,
        mask_erode: tiny('遮罩腐蚀'),
        mask_blur: tiny('遮罩羽化'),
        erode_mask: tiny('遮罩腐蚀'),
        blur_mask: tiny('遮罩羽化'),
        model: (document.querySelector('#mgModels .mgitem.on .nm') || {}).textContent || '',
        /* 多选契约：models=[{name, faces}]，导出事件 detail 里也带上。
           合成视图里按框分配过的，以那个为准（按脸分配优先于左栏多选）。
           导出真正发请求时用的是 mgExportModelsPayload()（全部选中项），
           这里留的是预览那一路的口径。 */
        models: (window.mgFaceModelsPayload ? window.mgFaceModelsPayload() : mgModelsPayload()),
        /* 导出设置页新增的两项：
           keep_cache —— 默认 false（成功导出后后端会删掉 <名字>_cache/，勾「保留」才留）；
           frame_from / frame_to —— 默认 0 / 空 = 全片，填了才随阶段请求带出去。 */
        keep_cache: String(this.readSeg('keep_cache') || '') === '1',
        /* 快速导出（融合流水线，后端 fused=1）：**默认开**。
           关掉 = 回退六阶段分段链（每步独立产物、可续跑），留给排查/对照。
           老页面没有这一项时按「开」处理 —— 默认行为不能因为少了个控件就变。 */
        fast_export: (function () {
          var seg = document.querySelector('#p-merge .mgview .seg[data-k="fast_export"]');
          if (!seg) return true;
          var on = seg.querySelector('b.on');
          var v = on ? on.getAttribute('data-v') : '';
          return String(v === null ? '1' : v) !== '';
        })(),
        frame_from: Math.max(0, parseInt(byId('mgFrameFrom'), 10) || 0),
        frame_to: Math.max(0, parseInt(byId('mgFrameTo'), 10) || 0),
        /* 性能那两个下拉（data-cfg 记忆的同一份值）：
             dfm_dtype      auto / fp32 / bf16 / bf16w    → body.dfm_dtype（auto 时不带）
             compose_backend auto / cpu / gpu             → body.compose_gpu（1 / 0，auto 时不带）
           组装请求体在 mgPerfBody（app.js 里那一份），这里只报「用户选了什么」。 */
        dfm_dtype: (function () {
          var el = document.getElementById('mgDfmDtype');
          return el ? String(el.value || 'auto') : 'auto';
        })(),
        compose_backend: (function () {
          var el = document.getElementById('mgCmpBackend');
          return el ? String(el.value || 'auto') : 'auto';
        })()
      };
    },

    /* 底部状态文字跟着当前设置走 */
    syncStatus: function (text) {
      var el = document.getElementById('mgExportStatus');
      if (!el) return;
      if (typeof text === 'string') {
        el.textContent = text;
        return;
      }
      var s = this.settings();
      var fmt = s.image_format === 'png' ? 'PNG' : 'JPEG';
      var dec = s.hwaccel ? '硬解' : '软解';
      var wk = s.num_workers > 0 ? s.num_workers + ' 进程' : '自动进程';
      el.textContent = '就绪 · ' + fmt + ' · ' + s.encoder + ' · ' + dec + ' · ' + wk;
    }
  };

  /* ---------- 合成器参数记忆 ---------- */
  var MG_PREF_KEY = "dfl-merge-prefs";
  function mgSavePrefs() {
    try {
      var p = (window.MergePages && window.MergePages.settings) ? window.MergePages.settings() : null;
      if (p) localStorage.setItem(MG_PREF_KEY, JSON.stringify(p));
    } catch (e) {}
  }
  function mgLoadPrefs() {
    var raw = null;
    try { raw = localStorage.getItem(MG_PREF_KEY); } catch (e) {}
    if (!raw) return;
    var p;
    try { p = JSON.parse(raw); } catch (e) { return; }
    var byId = { detector: "mgDetector", face_type: "mgFaceType", mode: "mgMode", color_transfer: "mgColor" };
    Object.keys(byId).forEach(function (k) {
      var el = document.getElementById(byId[k]);
      if (el && p[k] != null) el.value = p[k];
    });
    /* 存着老的模式值（debug / original / raw-*）时下拉会变成空值 → 回落到 seamless；
       再把「融合方式」两档分段同步成同一个值（否则两边亮的不一致）。 */
    var modeEl = document.getElementById("mgMode");
    if (modeEl) {
      var known = ["overlay", "hist-match", "seamless", "seamless-hist-match"];
      if (known.indexOf(String(modeEl.value || "")) < 0) modeEl.value = "seamless";
      if (window.mgSyncBlendSeg) window.mgSyncBlendSeg(modeEl.value);
    }
    /* seg 类按文字匹配 */
    var segMap = { detect: p.redetect, seg: p.mask_source, mask: p.mask_mode };
    Object.keys(segMap).forEach(function (dk) {
      var v = segMap[dk];
      if (!v) return;
      var seg = document.querySelector('#p-merge .seg[data-k="' + dk + '"]');
      if (!seg) return;
      Array.prototype.forEach.call(seg.children, function (b) {
        b.classList.toggle("on", b.textContent.trim() === String(v).trim());
      });
    });
    /* 数字行 */
    var numMap = { "遮罩腐蚀": p.erode_mask, "遮罩羽化": p.blur_mask };
    Object.keys(numMap).forEach(function (label) {
      if (numMap[label] == null) return;
      var rows = document.querySelectorAll('#p-merge .mgright .mgrow');
      for (var i = 0; i < rows.length; i++) {
        var f = rows[i].querySelector('.f');
        if (f && f.textContent.indexOf(label) >= 0) {
          var inp = rows[i].querySelector('input');
          if (inp) inp.value = numMap[label];
          break;
        }
      }
    });
  }
  window.mgSavePrefs = mgSavePrefs;
  window.mgLoadPrefs = mgLoadPrefs;

  /* ---------- 导出设置页的下拉记忆（data-cfg）----------
     和训练器参数页同一套键名约定：控件上打 data-cfg="dfm_dtype"，值按这个键存取。
     训练页那份记忆是模型 .dat 里的（后端给回来的）；导出设置这两颗是**用户偏好**，
     和主题 / 侧栏一样存本机 localStorage —— 刷新、关掉再开，还是上次选的那一档。
     为什么必须有：这两个值会改变画质 / 显存 / 速度。刷新一下就悄悄回到「自动」，
     用户会以为「设了没用」，然后去翻日志找原因。 */
  /* ---------- 合成后端 × 融合方式：界面上先互斥，别让用户选了再被后台降级 ----------
     GPU 合成目前只实现了 overlay（seamless 那条在 8GB 卡上实测更慢，默认不进 GPU）。
     以前的做法是「随便选、后台退回 CPU 并弹横幅」——用户看到的是「界面 cpu / 顶部说选了 gpu」，
     两边像在互相打架。所以融合方式是 seamless 时，把 gpu 这一档**禁用**并在 title 里说清。
     ★ 但「这一单用不上」≠「用户改主意了」：早先这里顺手把记忆（localStorage）也改成 auto，
       于是用户为了跑一单 seamless 而调好的偏好被**静默丢弃**，把融合方式改回 overlay 之后
       还得自己再挑一次 gpu（验收 D6「刷新后还是 gpu」就是照这个口径写的）。
       现在的分工：
         · _cmpPref      = 用户真正的偏好（初值来自记忆，只在用户动这个下拉时更新）；
         · 下拉的 value  = **这次**实际会用哪一档（seamless 时先显示「自动」）；
         · localStorage  = 只跟着用户的主动选择走，界面门控绝不写它。
       融合方式改回 overlay/hist-match 时，若偏好是 gpu 就直接恢复成 gpu —— 不用再挑。 */
  var _cmpPref = null;                 /* 'auto' | 'cpu' | 'gpu'；null = 还没读过记忆 */
  function mgCmpPrefRead() {
    var v = '';
    try { v = String((mgCfgMemRead() || {}).compose_backend || ''); } catch (e) {}
    return (v === 'gpu' || v === 'cpu' || v === 'auto') ? v : 'auto';
  }
  function mgCmpPref() {
    if (_cmpPref === null) _cmpPref = mgCmpPrefRead();
    return _cmpPref;
  }
  function mgCmpPrefSet(v) {
    v = String(v == null ? '' : v);
    if (v !== 'gpu' && v !== 'cpu' && v !== 'auto') return mgCmpPref();
    _cmpPref = v;
    return _cmpPref;
  }
  window.__mgCmpPref = mgCmpPref;       /* 验收读快照用（只读） */
  function mgGateCmpBackend() {
    var sel = document.getElementById('mgCmpBackend');
    if (!sel) return;
    var gpuOpt = null, i;
    for (i = 0; i < sel.options.length; i++) {
      if (sel.options[i].value === 'gpu') { gpuOpt = sel.options[i]; break; }
    }
    if (!gpuOpt) return;
    if (!sel.dataset.cmpTitle0) sel.dataset.cmpTitle0 = String(sel.title || '');
    /* seamless 现在**默认也走 GPU**（compose_gpu.seamless_on 默认 1），所以不再因为融合方式
       禁用 gpu 档。代价写进 option 的 title 里：Poisson 解会和 detect 抢卡，1080p 单脸下整条线
       可能比 CPU 慢；想要旧行为设 DFL_COMPOSE_GPU_SEAMLESS=0。 */
    gpuOpt.disabled = false;
    gpuOpt.title = 'compose_frame_gpu（torch CUDA）。seamless 也走 GPU（Poisson / DST）；'
      + '实测 1080p 单脸下 detect 会被挤、整条线可能比 CPU 慢，起不来会自动回退并告警';
    sel.title = sel.dataset.cmpTitle0 || sel.title;
    /* 用户在界面上挑过的值不该被程序改掉：偏好是 gpu 就回到 gpu */
    var pref = mgCmpPref();
    if (pref === 'gpu') sel.value = 'gpu';
    else if (String(sel.value) === 'gpu') sel.value = pref;
  }
  window.mgGateCmpBackend = mgGateCmpBackend;
  /* 只有用户**在界面上主动改**这个下拉，才算「他的选择」：更新偏好 + 写记忆。
     门控里那些 sel.value = 'auto' 是程序赋值，不派发 change，不会走到这里。 */
  (function bindCmpPref() {
    var sel = document.getElementById('mgCmpBackend');
    if (!sel || sel.dataset.cmpPrefBound) return;
    sel.dataset.cmpPrefBound = '1';
    sel.addEventListener('change', function () {
      mgCmpPrefSet(sel.value);
      try { if (typeof mgCfgMemSave === 'function') mgCfgMemSave('compose_backend', _cmpPref); } catch (e) {}
    });
  })();
  var MG_CFG_MEM_KEY = "dfl-mg-cfg-memory";
  function mgCfgMemRead() {
    try {
      var raw = localStorage.getItem(MG_CFG_MEM_KEY);
      var obj = raw ? JSON.parse(raw) : null;
      return (obj && typeof obj === "object") ? obj : {};
    } catch (e) { return {}; }
  }
  function mgCfgMemSave(key, val) {
    key = String(key || "");
    if (!key) return;
    var mem = mgCfgMemRead();
    mem[key] = String(val == null ? "" : val);
    try { localStorage.setItem(MG_CFG_MEM_KEY, JSON.stringify(mem)); } catch (e) {}
  }
  function mgCfgMemApply() {
    var mem = mgCfgMemRead();
    var els = document.querySelectorAll('#p-merge .mgview [data-cfg]');
    Array.prototype.forEach.call(els, function (el) {
      var k = el.getAttribute('data-cfg');
      if (!k || !Object.prototype.hasOwnProperty.call(mem, k)) return;
      var v = String(mem[k]);
      if (el.tagName === 'SELECT') {
        var hit = false;
        for (var i = 0; i < el.options.length; i++) {
          if (String(el.options[i].value) === v) { hit = true; break; }
        }
        /* 存的值不在选项里（老版本存的）就保留当前默认 —— 别把控件置成空值 */
        if (hit) el.value = v;
      } else if (String(el.value) !== v) {
        el.value = v;
      }
    });
  }
  window.mgCfgMemApply = mgCfgMemApply;
  window.mgCfgMemSave = mgCfgMemSave;
  window.__mgCfgMem = mgCfgMemRead;          /* 验收读快照用（只读） */
  (function bindCfgMem() {
    var view = document.querySelector('#p-merge .mgview[data-view="settings"]');
    if (!view || view.dataset.cfgMemBound) return;
    view.dataset.cfgMemBound = '1';
    view.addEventListener('change', function (e) {
      var el = e.target && e.target.closest ? e.target.closest('[data-cfg]') : null;
      if (!el) return;
      mgCfgMemSave(el.getAttribute('data-cfg'), el.value);
    });
  })();
  mgCfgMemApply();

  /* ---------- 事件绑定 ---------- */

  // seg 点击：只处理 .mgview 内的，避免和主页面其它 seg 抢事件
  function bindSeg() {
    document.addEventListener('click', function (e) {
      var b = e.target && e.target.closest ? e.target.closest('.mgview .seg > b') : null;
      if (!b) return;
      var seg = b.parentElement;
      all('b', seg).forEach(function (item) { item.classList.toggle('on', item === b); });
      MergePages.syncStatus();
    });
  }

  function bindActions() {
    var go = document.getElementById('mgStartExport');
    if (go) {
      go.addEventListener('click', function () {
        var payload = MergePages.settings();
        // 交由主 app 调 POST /api/export/start
        document.dispatchEvent(new CustomEvent('merge:export-start', { detail: payload }));
        MergePages.syncStatus('已提交导出 · ' + payload.image_format + ' · ' + payload.encoder);
      });
    }

    var compute = document.getElementById('mgComputeEmb');
    if (compute) {
      compute.addEventListener('click', function () {
        var btn = compute;
        var old = btn.textContent;
        btn.textContent = '计算中…';
        btn.disabled = true;
        // 交由主 app 调 POST /api/face/compute-embeddings
        document.dispatchEvent(new CustomEvent('merge:compute-embeddings', { detail: {} }));
        setTimeout(function () { btn.textContent = old; btn.disabled = false; }, 1200);
      });
    }

    /* 导出设置页两个路径框（输出目录 / 输出路径）右边那颗文件夹图标：
       走全局同一套 pickFolderFor（系统目录选择器 → 写回 .pathval 并派发 change）。
       以前 #mgOutPick 只派发 'merge:pick-output-dir'，而那条事件**没有任何监听者**，
       点了等于没点；现在两行图标绑同一份逻辑，行为一致。
       用 onclick 属性（不是 addEventListener）：bindPickPaths 也写这个属性，
       谁后绑定谁生效，不会一次点击弹两个对话框。 */
    function mgBindOutPickers() {
      var icons = document.querySelectorAll(
        '#p-merge .mgview[data-view="settings"] .pathinp .pill.iconpill');
      Array.prototype.forEach.call(icons, function (icon) {
        icon.onclick = function (e) {
          e.preventDefault();
          e.stopPropagation();
          var box = icon.closest ? icon.closest('.pathinp') : null;
          if (window.pickFolderFor) { window.pickFolderFor(box); return; }
          document.dispatchEvent(new CustomEvent('merge:pick-output-dir', { detail: {} }));
        };
      });
    }
    mgBindOutPickers();
    window.mgBindOutPickers = mgBindOutPickers;

    /* 「输出目录」被手改过才参与导出（默认值只是回显，不动它 = 与以前一致） */
    var defOut = document.getElementById('mgDefOut');
    if (defOut && !defOut.dataset.dirtyBound) {
      defOut.dataset.dirtyBound = '1';
      var markDirty = function () {
        window.__mgDefOutEdited = String(defOut.value || '').trim() !== '';
      };
      defOut.addEventListener('input', markDirty);
      defOut.addEventListener('change', markDirty);   /* 目录选择器写回来也走这条 */
    }

    var workers = document.getElementById('mgWorkers');
    if (workers) {
      workers.addEventListener('input', function () {
        var v = parseInt(workers.value, 10);
        if (isNaN(v)) { MergePages.syncStatus(); return; }
        if (v < WORKERS_MIN) { workers.value = String(WORKERS_MIN); }
        if (v > WORKERS_MAX) { workers.value = String(WORKERS_MAX); }
        MergePages.syncStatus();
      });
    }

    var out = document.getElementById('mgOutPath');
    if (out) out.addEventListener('input', function () { MergePages.syncStatus(); });
  }

  function init() {
    MergePages.show('main');          // 默认全部隐藏，停在主视图
    MergePages.syncStatus();
    bindSeg();
    bindActions();
    /* 记忆里的两档要**先**回填：syncStatus 只读格式/编码/进程，但用户在设置页
       第一眼就得看到上次选的精度/后端，而不是等某次渲染才补上。 */
    try { if (window.mgCfgMemApply) window.mgCfgMemApply(); } catch (e) {}
    /* 合成后端 × 融合方式的互斥要在「记忆 + 偏好都回填完」之后再判：
       data-cfg 记忆在这里回填，#mgMode 由 go('merge') 的 mgLoadPrefs 回填，两者都可能
       晚于 DOM（也晚于这一行）。所以判一次 + 三个时间点再兜底 —— 记忆回填是异步的，
       只判一次会漏掉「先渲染、后回填」的那条路。 */
    if (window.mgGateCmpBackend) {
      try { window.mgGateCmpBackend(); } catch (e) {}
      [0, 60, 400].forEach(function (ms) {
        setTimeout(function () { try { window.mgGateCmpBackend(); } catch (e) {} }, ms);
      });
    }
  }

  window.MergePages = MergePages;

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();
