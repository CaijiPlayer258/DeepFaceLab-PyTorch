/**
 * Point Tool — SAM 点击分割（正 / 负样本点）
 *
 * 交互：
 *   左键            → 正样本点（label 1，绿点）。默认「单点」：点一下就清空旧点集，
 *                     只留这一个点重新 predict
 *   Ctrl/Cmd + 左键 → 多点参考：把点追加到现有点集（多点时点上带编号）
 *   Alt + 左键/右键 → 负样本点（label 0，红点），仍然是追加，不触发单点清空
 *   每加一个点防抖 120 ms 打一次 /api/mask/sam/predict，把得分最高的 mask
 *   以「替换式」铺到遮罩层（不叠加），点集一直留着直到切图或 Esc。
 *
 * 键盘：
 *   Esc    → 清空点集，遮罩回到加点前的样子
 *   Enter  → 接受当前 mask，按画笔的 +/-0.5 口径合并进遮罩（合并前进撤销栈）
 *   Ctrl+Z → 不在这里绑：app.js 的全局 keydown 已经吃掉了，路由到 undoPoint()
 *            （先撤点、点集空了再撤历史）—— 本文件再绑一次就是一次按键撤两步
 *
 * 降级：
 *   /api/mask/sam/load 回 404 / 501（后端没装 sam 包）→ 整个工具下线，
 *   按钮 title 改「需要额外模型（约 3.4 GB）」；其它错误只 toast 一次。
 *
 * 依赖：window.API、window.MaskCanvas、以及 app 的
 *       { canvas, currentIndex, files, currentImagePath, pushHistory, setStatus,
 *         toast, onSamUnsupported, _updateCounts }
 * 导出：window.PointTool
 */
(function () {
  'use strict';

  var DEBOUNCE_MS      = 120;
  var DEFAULT_VARIANT  = 'base_plus';
  var BUSY_ID          = 'sam-busy-badge';
  var UNSUPPORTED_HINT = '需要额外模型（约 3.4 GB）';
  var MAX_POINTS       = 64;
  var HINT_ID          = 'sam-point-hint';
  var HINT_TEXT        = '点击=单点 · Ctrl+点击=多点 · Alt+点击=负样本';

  // ==========================================================================
  // 小工具：data URL 归一化 / 遮罩深拷贝 / PNG 解码
  // ==========================================================================

  /** 后端可能回裸 base64，也可能回 data URL，统一成 data URL。 */
  function toDataUrl (s) {
    if (!s) return '';
    return s.indexOf('data:') === 0 ? s : 'data:image/png;base64,' + s;
  }

  /** 遮罩深拷贝 —— 预览前的「底」和合并结果都要留副本。 */
  function copyMask (mask) {
    if (!mask || !mask.data) return null;
    var data = new Float32Array(mask.data.length);
    data.set(mask.data);
    return { width: mask.width, height: mask.height, data: data };
  }

  /**
   * 把一张 PNG 遮罩解成 size×size 的 Float32Array。
   * 没有 DOM（Node 单测）或解码失败时回调 null，调用方保留叠加显示即可。
   *
   * @param {string}   url  – data URL 或裸 base64
   * @param {number}   size – 目标边长（跟 canvas.maskResolution 对齐）
   * @param {Function} cb   – function ({ width, height, data } | null)
   */
  function decodeMask (url, size, cb) {
    if (!url || typeof Image === 'undefined' || typeof document === 'undefined') {
      cb(null);
      return;
    }
    var img = new Image();
    img.onload = function () {
      try {
        var c   = document.createElement('canvas');
        c.width  = size;
        c.height = size;
        var ctx = c.getContext('2d');
        ctx.drawImage(img, 0, 0, size, size);
        var px   = ctx.getImageData(0, 0, size, size).data;
        var data = new Float32Array(size * size);
        for (var i = 0; i < data.length; i++) data[i] = px[i * 4] / 255;
        cb({ width: size, height: size, data: data });
      } catch (err) {
        cb(null);
      }
    };
    img.onerror = function () { cb(null); };
    img.src = toDataUrl(url);
  }

  // ==========================================================================
  // PointTool
  // ==========================================================================

  class PointTool {

    /**
     * @param {MaskCanvas} canvas
     * @param {object}     app – 编辑器主对象（见文件头依赖列表）
     */
    constructor (canvas, app) {
      this.canvas = canvas;
      this.app    = app;

      // ---- 会话 -----------------------------------------------------------
      this.points       = [];      // [{ x, y, label }]，图片像素坐标
      this.unsupported  = false;   // 后端没装 sam 包
      this._session     = null;    // /sam/load 给的会话 id
      this._loadedImage = '';      // 会话对应的图片路径
      this._variant     = DEFAULT_VARIANT;
      this._loading     = null;    // 正在进行的 load Promise
      this._loadSeq     = 0;       // 丢弃切图后回来的旧 load
      this._loadedOnce  = false;

      // ---- 推理 -----------------------------------------------------------
      this._pendingBox = null;     // GROUNDED SAM2 的框提示
      this._timer      = null;     // 防抖句柄
      this._inflight   = false;    // 已有请求在飞
      this._dirty      = false;    // 飞行期间又加了点
      this._seq        = 0;        // 丢掉过期响应
      this._running    = null;

      // ---- 预览 -----------------------------------------------------------
      this._baseMask     = null;   // 加点前的遮罩（Esc 回滚 / Enter 合并的底）
      this._previewing   = false;
      this._previewUrl   = null;
      this._previewScore = 0;

      this._busy = false;

      // 画布角落 + 工具按钮 title 的操作提示（启动时默认工具就是 SAM）
      this._setHint(true);
    }

    // ========================================================================
    // 生命周期
    // ========================================================================

    /** 切到本工具：清空点集 + 重新 load（同一张图后端有缓存，很快）。 */
    activate () {
      if (this.canvas) this.canvas.setToolOverlay(null);
      this._setHint(true);
      this.clearPoints();
      if (this.unsupported) return;
      this._ensureLoaded(false);
    }

    /** 切走本工具：点集与预览一起收掉。 */
    deactivate () {
      this._setHint(false);
      this.clearPoints();
    }

    /**
     * 换图：会话作废，点集清空。
     * 图已经换好了（遮罩是新图自己的），所以这里不回滚遮罩。
     * @param {number} [_index]
     */
    onImageChanged (_index) {
      this._hardReset();
      if (this.unsupported) return;
      /* 本工具正在用 → 立刻建会话；之前用过 → 也顺手建好。
         从没用过又不在本工具 → 留给 activate()，省一次请求。 */
      if (this.app && (this.app.currentTool === 'sam' || this._loadedOnce)) {
        this._ensureLoaded(false);
      }
    }

    /** 清点集 + 会话，且不碰遮罩（换图专用）。 */
    _hardReset () {
      this.points = [];
      this._pendingBox = null;
      this._seq++;
      this._loadSeq++;                // 旧图的 load 响应一律不认
      this._dirty = false;
      if (this._timer) { clearTimeout(this._timer); this._timer = null; }

      this._baseMask     = null;
      this._previewing   = false;
      this._previewUrl   = null;
      this._previewScore = 0;
      this._session      = null;
      this._loadedImage  = '';
      this._loading      = null;

      if (this.canvas) this.canvas.setPoints([]);
      this.setBusy(false);
    }

    /**
     * 撤销恢复过遮罩：预览的「底」作废，点集留着（点还有用）。
     * 由 App.undo() 回调。
     */
    onHistoryRestored () {
      this._seq++;
      this._baseMask     = null;
      this._previewing   = false;
      this._previewUrl   = null;
      this._previewScore = 0;
      this.setBusy(false);
    }

    /**
     * 清空点集，并把遮罩恢复到加点之前的样子。
     * Esc、切工具、切图都会走这里。
     * @returns {boolean} 是否真的清掉了东西
     */
    clearPoints () {
      var had = this.points.length > 0 || this._previewing;

      this.points = [];
      this._pendingBox = null;
      this._seq++;
      this._dirty = false;
      if (this._timer) { clearTimeout(this._timer); this._timer = null; }

      if (this._previewing) {
        var base = this._baseMask;
        this._baseMask     = null;
        this._previewing   = false;
        this._previewUrl   = null;
        this._previewScore = 0;

        this.canvas.setMask(base ? copyMask(base) : null);
        if (this.app && this.app._updateCounts) this.app._updateCounts();
        this.canvas.renderNow();
      }

      if (this.canvas) this.canvas.setPoints([]);
      this.setBusy(false);
      return had;
    }

    /**
     * Ctrl+Z 的第一步：撤掉最后一个点，用剩下的点重新 predict。
     *
     * 预览态的 mask 从来不进历史栈（只有 Enter 合并才进），所以「撤点」必须
     * 先于「撤历史」：由 App._handleUndoRedo() 按这个顺序调用。
     * 点集被撤空时退回加点前的遮罩（复用 clearPoints 的回滚逻辑）。
     *
     * @returns {boolean} 是否真的撤了一个点（false → 交给历史栈）
     */
    undoPoint () {
      if (!this.points.length) return false;

      this.points.pop();
      // 已经发出去的那一枪是「更多点」的预测，作废掉，别让它盖回来
      this._seq++;
      this._dirty = false;

      if (!this.points.length) {
        this.clearPoints();                    // 点集空了：预览作废，遮罩回到加点前
        this._setStatus('已撤销最后一个点');
        return true;
      }

      this._renderPoints();
      this._schedulePredict();
      this._setStatus('已撤销最后一个点 · 剩 ' + this.points.length + ' 个');
      return true;
    }

    // ========================================================================
    // 鼠标
    // ========================================================================

    onMouseDown (e) {
      if (this.unsupported) return;
      if (!this.canvas || !this.canvas.image) return;

      var negative = (e.button === 2) || !!e.altKey;
      var positive = !negative && (e.button === undefined || e.button === 0);
      if (!negative && !positive) return;   // 中键/其它键忽略

      // Ctrl（Mac 上 Cmd）按住 = 多点参考；否则正样本点击就是「单点」
      var multi = !!(e.ctrlKey || e.metaKey);

      if (e.preventDefault) e.preventDefault();

      var img = this.canvas.screenToImage(e.offsetX, e.offsetY);
      var iw  = this.canvas.imgW || 0;
      var ih  = this.canvas.imgH || 0;
      if (iw && ih && (img.x < 0 || img.y < 0 || img.x > iw || img.y > ih)) return;   // 点在空白处

      /* 默认单点：不带修饰键的正样本点击 = 新的一次点选，旧点集作废。
         负样本（Alt / 右键）不参与，仍然往现有点集上追加。
         清点在越界检查之后做 —— 点到空白处不该把已有结果弄丢。 */
      if (positive && !multi) {
        this.points = [];
        this._pendingBox = null;     // 单点 = 新提示，旧框不能粘着
        this._seq++;                 // 旧点集的预测还在飞？作废
      }

      if (this.points.length >= MAX_POINTS) {
        this._toast('点太多了，先按 Esc 清一下', 'info');
        return;
      }

      this.points.push({
        x: Math.round(img.x),
        y: Math.round(img.y),
        label: negative ? 0 : 1
      });

      this._renderPoints();
      this._schedulePredict();
    }

    onMouseMove (e) {
      if (!this.canvas) return;
      var img = this.canvas.screenToImage(e.offsetX, e.offsetY);
      this.canvas.setToolOverlay({ type: 'sam-cursor', x: img.x, y: img.y });
    }

    onMouseUp (_e) { /* 无 */ }

    // ========================================================================
    // 键盘
    // ========================================================================

    onKeyDown (e) {
      if (e.key === 'Escape') {
        if (e.preventDefault) e.preventDefault();
        this.clearPoints();
        this._setStatus('');
        return;
      }
      if (e.key === 'Enter') {
        if (this.unsupported) return;
        if (e.preventDefault) e.preventDefault();
        this.acceptPreview();
      }
    }

    onKeyUp (_e) { /* 无 */ }

    // ========================================================================
    // 界面：点集、busy 角标
    // ========================================================================

    /**
     * 把点集丢给 canvas 画（绿=正样本，红=负样本）。
     * 只有一个点 = 单点模式，不写编号，画面上就一个小圆点；
     * 多点才编号 1..n（canvas 用空字符串 id 表示「不画字」）。
     */
    _renderPoints () {
      var single = this.points.length === 1;
      var out = [];
      for (var i = 0; i < this.points.length; i++) {
        out.push({
          x: this.points[i].x,
          y: this.points[i].y,
          label: this.points[i].label,
          id: single ? '' : i + 1
        });
      }
      this.canvas.setPoints(out);
    }

    /**
     * 画布左上角的一行操作提示 + 工具按钮的 title。
     * 默认工具就是 SAM，所以构造时也开一次；切走时收掉。
     * @param {boolean} on
     */
    _setHint (on) {
      if (typeof document === 'undefined') return;

      var btn = document.querySelector('.tool-btn[data-tool="sam"]');
      if (btn && on && !this.unsupported) btn.setAttribute('title', HINT_TEXT);

      var host = document.querySelector('.canvas-container');
      if (!host) return;

      var el = document.getElementById(HINT_ID);
      if (!on) {
        if (el) el.style.display = 'none';
        return;
      }
      if (!el) {
        el = document.createElement('div');
        el.id = HINT_ID;
        el.textContent = HINT_TEXT;
        el.style.cssText =
          'position:absolute;top:10px;left:12px;z-index:50;pointer-events:none;' +
          'padding:3px 8px;border-radius:99px;' +
          'font:10px/1.5 Inter,-apple-system,sans-serif;letter-spacing:.02em;' +
          'background:rgba(18,18,22,.72);color:#c9c9e8;' +
          'border:1px solid rgba(255,255,255,.10);';
        host.appendChild(el);
      }
      el.style.display = 'block';
    }

    /**
     * 画布角落的小角标（默认「SAM 计算中」）。
     * 不走 canvas.setLoading —— 那是全屏压暗，会挡眼睛。
     * @param {boolean} on
     * @param {string}  [label]
     */
    setBusy (on, label) {
      this._busy = !!on;
      if (typeof document === 'undefined') return;

      var host = document.querySelector('.canvas-container');
      if (!host) return;

      var el = document.getElementById(BUSY_ID);
      if (!on) {
        if (el) el.style.display = 'none';
        return;
      }
      if (!el) {
        el = document.createElement('div');
        el.id = BUSY_ID;
        el.style.cssText =
          'position:absolute;top:10px;right:12px;z-index:50;pointer-events:none;' +
          'display:flex;align-items:center;gap:5px;padding:3px 8px;border-radius:99px;' +
          'font:10px/1.5 Inter,-apple-system,sans-serif;letter-spacing:.02em;' +
          'background:rgba(18,18,22,.86);color:#c9c9e8;border:1px solid rgba(255,255,255,.14);';
        var dot = document.createElement('span');
        dot.style.cssText = 'width:5px;height:5px;border-radius:50%;background:#8b8be6;';
        if (dot.animate) {
          try {
            dot.animate([{ opacity: 1 }, { opacity: 0.2 }],
                        { duration: 700, iterations: Infinity, direction: 'alternate' });
          } catch (_) { /* 老浏览器没有 WAAPI，静态点也行 */ }
        }
        var txt = document.createElement('span');
        txt.textContent = label || 'SAM 计算中';
        el.__label = txt;

        el.appendChild(dot);
        el.appendChild(txt);
        host.appendChild(el);
      }
      if (label && el.__label) el.__label.textContent = label;
      el.style.display = 'flex';
    }

    // ========================================================================
    // 会话：/api/mask/sam/load
    // ========================================================================

    /**
     * 当前图片的路径（给 load / gdino 用）。
     * @returns {string}
     */
    _imagePath () {
      if (this.app && typeof this.app.currentImagePath === 'function') {
        var p = this.app.currentImagePath(this.app.currentIndex);
        if (p) return p;
      }
      // 兜底：文件列表 + 标题栏目录
      var list = (this.app && this.app.files) || [];
      var item = list[this.app ? this.app.currentIndex : 0];
      var name = item && (item.name || item.filename);
      if (!name) return '';
      var el  = (typeof document !== 'undefined') ? document.getElementById('workspace-path') : null;
      var dir = el && el.value ? String(el.value).trim() : '';
      if (!dir) return name;
      var sep = (dir.indexOf('/') >= 0 && dir.indexOf('\\') < 0) ? '/' : '\\';
      return dir.replace(/[\\/]+$/, '') + sep + name;
    }

    /**
     * 保证当前图有 SAM 会话。
     * @param {boolean} quiet – 不闪 busy 角标（推理链路内部调用时用）
     * @returns {Promise<boolean>}
     */
    _ensureLoaded (quiet) {
      var self = this;
      if (this.unsupported) return Promise.resolve(false);

      var image = this._imagePath();
      if (!image) return Promise.resolve(false);
      if (this._session && this._loadedImage === image) return Promise.resolve(true);
      if (this._loading) return this._loading;

      var gen = this._loadSeq;
      if (!quiet) this.setBusy(true, 'SAM 加载中');
      this._loading = API.samLoad(image, this._variant)
        .then(function (res) {
          if (gen === self._loadSeq) self._loading = null;   // 别把新一次 load 的句柄清掉
          if (!quiet) self.setBusy(false);
          if (gen !== self._loadSeq) return false;           // 已经换图了，这个会话作废
          if (!res || res.ok === false) {
            self._toast('SAM 载入失败', 'error');
            return false;
          }
          self._session     = res.session || '';
          self._loadedImage = image;
          self._loadedOnce  = true;
          return true;
        })
        .catch(function (err) {
          if (gen === self._loadSeq) self._loading = null;
          if (!quiet) self.setBusy(false);
          if (API.isUnsupported(err)) { self._degrade(err); return false; }
          if (gen === self._loadSeq) {
            self._toast('SAM 载入失败: ' + (err && err.message), 'error');
          }
          return false;
        });
      return this._loading;
    }

    // ========================================================================
    // 推理：/api/mask/sam/predict
    // ========================================================================

    /** 防抖 120 ms 再打，连点只发最后一枪。 */
    _schedulePredict () {
      var self = this;
      if (this._timer) clearTimeout(this._timer);
      this._timer = setTimeout(function () {
        self._timer = null;
        self._predict();
      }, DEBOUNCE_MS);
    }

    /**
     * 打一次 predict，返回 Promise<boolean>。
     * @returns {Promise<boolean>}
     */
    _predict () {
      var self = this;
      if (this.unsupported) return Promise.resolve(false);
      if (this.points.length === 0 && !this._pendingBox) return Promise.resolve(false);

      if (this._inflight) {          // 上一个还没回来，标记一下，回来再补
        this._dirty = true;
        return this._running || Promise.resolve(false);
      }

      this._inflight = true;
      this.setBusy(true);

      var run = this._ensureLoaded(true)
        .then(function (ok) {
          if (!ok || self.unsupported) return false;

          var seq = ++self._seq;
          var payload = [];
          for (var i = 0; i < self.points.length; i++) {
            payload.push({
              x: self.points[i].x,
              y: self.points[i].y,
              label: self.points[i].label
            });
          }

          var box = self._pendingBox;   // 这一次推理用哪只框，重试也保持一致

          function attempt (retried) {
            return API.samPredict(self._session, payload, box)
              .then(function (res) {
                if (seq !== self._seq) return false;    // 过期响应，丢掉
                return self._applyPrediction(res, seq);
              })
              .catch(function (err) {
                if (seq !== self._seq) return false;
                if (API.isUnsupported(err)) { self._degrade(err); return false; }

                /* 会话被 LRU 淘汰：清掉缓存，重建一次再重试；只重试一次 */
                if (!retried && API.isSessionExpired(err)) {
                  self._session     = null;
                  self._loadedImage = '';
                  self._loading     = null;
                  return self._ensureLoaded(true).then(function (loaded) {
                    if (seq !== self._seq) return false;
                    if (!loaded || self.unsupported) return false;
                    return attempt(true);
                  });
                }

                self._toast('SAM 推理失败: ' + (err && err.message), 'error');
                return false;
              });
          }

          return attempt(false);
        })
        .then(function (ok) {
          self._inflight = false;
          self.setBusy(false);
          if (self._dirty) { self._dirty = false; self._schedulePredict(); }
          return ok;
        });

      this._running = run;
      return run;
    }

    /**
     * 取得分最高的那张 mask，替换式铺到遮罩层。
     * @param {{ masks?: string[], scores?: number[], ms?: number }} result
     * @param {number} [seq] – 发起 predict 时的 _seq，解码回来再复查一次
     * @returns {boolean}
     */
    _applyPrediction (result, seq) {
      if (seq === undefined) seq = this._seq;   // 兼容直接调用：至少锁住进入时的代次
      var self   = this;
      var masks  = (result && result.masks) || [];
      var scores = (result && result.scores) || [];

      if (!masks.length) {
        this._toast('SAM 没有返回遮罩', 'error');
        return false;
      }

      var best = 0;
      if (scores.length === masks.length) {
        for (var i = 1; i < scores.length; i++) {
          if (scores[i] > scores[best]) best = i;
        }
      }
      var url = masks[best];
      if (!url) return false;

      if (!this._previewing) {         // 第一次预览：记住加点前的遮罩
        this._baseMask   = copyMask(this.canvas.mask);
        this._previewing = true;
      }
      this._previewUrl   = url;
      this._previewScore = scores.length ? scores[best] : 0;

      var size = this.canvas.maskResolution || 256;
      decodeMask(url, size, function (mask) {
        if (seq !== self._seq) return;   // 迟到的解码：点集/图片已经变了
        if (!mask) {
          self._toast('SAM 遮罩解码失败', 'error');
          return;
        }
        self.canvas._maskFadeStart = Date.now();
        self.canvas.setMask(mask);     // 替换式：整层换掉，不叠加
        self.canvas.renderNow();
        if (self.app && self.app._updateCounts) self.app._updateCounts();

        var ms    = (result && result.ms) || 0;
        var score = self._previewScore;
        self._setStatus(
          'SAM 点 ' + self.points.length + ' · 分数 ' + score.toFixed(3) +
          (ms ? ' · ' + Math.round(ms) + 'ms' : '')
        );
      });
      return true;
    }

    // ========================================================================
    // Enter：接受当前 mask
    // ========================================================================

    /**
     * 把预览里的 mask 合并进遮罩，口径跟画笔一致：
     * 绘制模式 +0.5（封顶 1），排除模式 -0.5（地板 -1）。
     * 拿不到预览就什么都不做，叠加显示原样留着。
     * @returns {boolean}
     */
    acceptPreview () {
      var self   = this;
      var canvas = this.canvas;
      if (!this._previewUrl) {
        this._setStatus('没有可接受的 SAM 遮罩');
        return false;
      }

      var isDraw = (canvas.mode || 'draw') === 'draw';
      var target = copyMask(this._baseMask);
      if (!target) {
        var size = canvas.maskResolution || 256;
        target = { width: size, height: size, data: new Float32Array(size * size) };
      }
      var size2      = target.width || target.height || 256;
      var loadSeq    = this._loadSeq;
      var previewUrl = this._previewUrl;
      decodeMask(previewUrl, size2, function (mask) {
        /* 解码期间换图/换预览了：这帧已经过期，别 setMask，也别 pushHistory */
        if (loadSeq !== self._loadSeq || previewUrl !== self._previewUrl) return;
        if (!mask) return;      // 解码不了：保留叠加显示，不硬造遮罩

        /* 合并前的那一帧必须进撤销栈，而且只在这里进（解码成功才进，
           解码失败不该平白多一条历史）。pushHistory 记的是 canvas.mask，
           此刻它还是「预览」，先把底临时换回去再 push —— Ctrl+Z 才能精确
           回到合并前。 */
        if (self.app && self.app.pushHistory) {
          var shown = canvas.mask;
          canvas.mask = self._baseMask ? copyMask(self._baseMask) : null;
          self.app.pushHistory();
          canvas.mask = shown;
        }

        var dst = target.data;
        var src = mask.data;
        var n   = Math.min(dst.length, src.length);
        for (var i = 0; i < n; i++) {
          if (src[i] > 0.5) {
            dst[i] = isDraw ? Math.min(1, dst[i] + 0.5) : Math.max(-1, dst[i] - 0.5);
          }
        }
        canvas._maskFadeStart = Date.now();
        canvas.setMask(target);
        canvas.renderNow();
        if (self.app && self.app._updateCounts) self.app._updateCounts();

        self._baseMask     = copyMask(target);   // 合并结果成为新的底
        self._previewing   = false;
        self._previewUrl   = null;
        self._previewScore = 0;
        /* 点集用完即弃：留着的话下一次 Ctrl+Z 会先「撤点」、用剩下的点重新
           predict 把刚合并的结果顶掉；可这一笔已经进历史了，Ctrl+Z 应该
           直接回到合并前。 */
        self.clearPoints();
        self._setStatus('已合并到遮罩 · Ctrl+Z 撤销');
      });
      return true;
    }

    // ========================================================================
    // GROUNDED SAM2：框当提示，效果等同框选
    // ========================================================================

    /**
     * @param {number[]} box – [x1, y1, x2, y2]，图片像素坐标
     * @returns {Promise<boolean>}
     */
    applyBoxPrompt (box) {
      var self = this;
      if (this.unsupported) return Promise.resolve(false);
      if (!box || box.length < 4) return Promise.resolve(false);

      this._pendingBox = [box[0], box[1], box[2], box[3]];
      this.points = [];                       // 框提示本身就是提示，点集清掉
      this.canvas.setPoints([]);
      if (this._timer) { clearTimeout(this._timer); this._timer = null; }

      return this._predict().then(function (ok) {
        if (!ok && self._pendingBox) self._setStatus('SAM 框提示没有出遮罩');
        return ok;
      });
    }

    // ========================================================================
    // 降级 / 提示
    // ========================================================================

    /**
     * 后端没装 sam 包：整个工具下线，按钮上留提示。
     * @param {Error} [_err]
     */
    _degrade (_err) {
      if (this.unsupported) return;
      this.unsupported  = true;
      this._session     = null;
      this._loadedImage = '';
      this._loading     = null;
      if (this._timer) { clearTimeout(this._timer); this._timer = null; }
      this._setHint(false);
      this.clearPoints();

      if (this.app && typeof this.app.onSamUnsupported === 'function') {
        this.app.onSamUnsupported(UNSUPPORTED_HINT, _err);
      }
    }

    /** 同一个错误只弹一次，交给 app 去重。 */
    _toast (msg, kind) {
      if (this.app && typeof this.app.toast === 'function') this.app.toast(msg, kind || 'error');
      else this._setStatus(msg);
    }

    /** 状态栏文案。 */
    _setStatus (msg) {
      if (this.app && this.app.setStatus) this.app.setStatus(msg);
    }
  }

  // ==========================================================================
  // 导出
  // ==========================================================================

  PointTool.UNSUPPORTED_HINT = UNSUPPORTED_HINT;
  PointTool.DEBOUNCE_MS      = DEBOUNCE_MS;

  window.PointTool = PointTool;

})();
