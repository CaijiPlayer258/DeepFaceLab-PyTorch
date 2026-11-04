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

  var PLACEHOLDER_COUNT = 24;          // 人脸网格占位数
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
      var target = null;
      all('.mgpage').forEach(function (el) {
        var on = el.getAttribute('data-page') === name;
        el.classList.toggle('on', on);
        if (on) {
          target = el;
          el.setAttribute('aria-hidden', 'false');
        } else {
          el.setAttribute('aria-hidden', 'true');
        }
      });

      // 主视图让位（CSS：#p-merge.mgp-on > .mgbody / .mgtl 隐藏）
      var merge = document.getElementById('p-merge');
      if (merge) merge.classList.toggle('mgp-on', !!target);

      this.current = target ? name : 'main';
      return this.current;
    },

    /* 生成 n 个棋盘格占位缩略图（默认 24） */
    buildFaceGrid: function (n) {
      var grid = document.getElementById('mgFaceGrid');
      if (!grid) return 0;
      var count = (typeof n === 'number' && n >= 0) ? n : PLACEHOLDER_COUNT;
      var html = '';
      for (var i = 1; i <= count; i++) {
        html += '<div class="mgp-face" data-idx="' + i + '" title="人脸 ' + i + '（占位）">' +
                '<i>' + i + '</i></div>';
      }
      grid.innerHTML = html;
      this.setFaceCount(count);
      return count;
    },

    /* 更新计数 + 空态（n = 0 时显示「还没有建立人脸库」） */
    setFaceCount: function (n) {
      var count = Math.max(0, parseInt(n, 10) || 0);
      var label = document.getElementById('mgFaceCount');
      var grid = document.getElementById('mgFaceGrid');
      var empty = document.getElementById('mgFaceEmpty');
      if (label) label.innerHTML = '已建库 <b>' + count + '</b> 张人脸';
      if (grid) grid.hidden = count === 0;
      if (empty) empty.hidden = count !== 0;
      return count;
    },

    /* 读取某个 seg 当前选中项的 data-v（没有 data-v 时退化成文字） */
    readSeg: function (key) {
      var seg = one('.mgpage .seg[data-k="' + key + '"]');
      if (!seg) return '';
      var on = seg.querySelector('b.on');
      if (!on) return '';
      var v = on.getAttribute('data-v');
      return v === null ? on.textContent.trim() : v;
    },

    /* 采集导出参数：字段名与 /api/export/start 一致 */
    settings: function () {
      var out = one('#mgOutPath');
      var workers = one('#mgWorkers');
      var nw = parseInt(workers ? workers.value : '0', 10);
      if (isNaN(nw)) nw = 0;
      nw = Math.min(WORKERS_MAX, Math.max(WORKERS_MIN, nw));
      return {
        image_format: this.readSeg('image_format') || 'jpg',
        encoder: this.readSeg('encoder') || 'h264_nvenc',
        hwaccel: this.readSeg('hwaccel'),            // 'auto' 开 / '' 关
        output_path: out ? out.value.trim() : '',
        num_workers: nw
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

  /* ---------- 事件绑定 ---------- */

  // seg 点击：只处理 .mgpage 内的，避免和主页面其它 seg 抢事件
  function bindSeg() {
    document.addEventListener('click', function (e) {
      var b = e.target && e.target.closest ? e.target.closest('.mgpage .seg > b') : null;
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

    var pick = document.getElementById('mgOutPick');
    if (pick) {
      pick.addEventListener('click', function () {
        document.dispatchEvent(new CustomEvent('merge:pick-output-dir', { detail: {} }));
      });
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
    MergePages.buildFaceGrid(PLACEHOLDER_COUNT);
    MergePages.syncStatus();
    bindSeg();
    bindActions();
  }

  window.MergePages = MergePages;

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();
