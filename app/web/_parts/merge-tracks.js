/* ============================================================
   合成工作室 · PR 风格多轨时间轴（片段）
   合并位置：app/web/js/ 独立引入（放在 app.js 之前或之后均可，勿改 app.js）
   暴露：window.MergeTimeline = { init(totalFrames), setFrame(n), getFrame() }

   参数命名全部来自 4.0 MergeStudio，未编造：
     state.totalFrames / state.currentFrame / state.fps
     state.cutSegments[{ start, end }]              （timeline.js 233 行）
     state.angleSegments[{ start, end, angles }]    （timeline.js 269 行）
     角度默认值 '0,90,180,270'                       （timeline.js 254 行）
     时间码格式 mm:ss.cs = 00:00.00                  （app.js updateTimecode）
     缩放档位 1 / 0.5 / 0.25 / 0.125                （index.html #res-select）
       → pxPerFrame 2 / 1 / 0.5 / 0.25（本片段映射，见 PPF）
     4.0 的 zoom-in / zoom-out 每次 ×2 / ÷2，范围 0.25 ~ 8
   老式写法：只用 var + function，不用 ES6 class / 箭头函数。
   ============================================================ */
(function () {
  'use strict';

  /* 4.0 四档缩放（值本身即 #res-select 的 option value） */
  var ZOOM_ORDER = [1, 0.5, 0.25, 0.125];
  var PPF = { '1': 2, '0.5': 1, '0.25': 0.5, '0.125': 0.25 };

  var RULER_STEP = 500;   /* 刻度尺每 500px 一个数字 */
  var N_VIDEO = 40;       /* 视频轨缩略占位块 */
  var N_FACE = 25;        /* 人脸轨块数 */
  var N_CUT = 6;          /* 剪切区段数 */
  var N_ANGLE = 3;        /* 角度色带数 */

  var PLAY_SVG = '<svg viewBox="0 0 16 16" width="14" height="14" aria-hidden="true"><path d="M5.4 3.6 12.4 8l-7 4.4V3.6Z" fill="currentColor"/></svg>';
  var PAUSE_SVG = '<svg viewBox="0 0 16 16" width="14" height="14" aria-hidden="true"><rect x="4.6" y="3.8" width="2.6" height="8.4" rx="0.8" fill="currentColor"/><rect x="8.8" y="3.8" width="2.6" height="8.4" rx="0.8" fill="currentColor"/></svg>';

  var S = {
    ready: false,
    total: 0,
    frame: 0,
    fps: 30,          /* 4.0 state.fps；init 第二参数可覆盖 */
    zoom: 1,          /* 当前缩放档位，对应 PPF */
    ppf: 2,           /* pxPerFrame */
    dragging: false,
    playing: false,
    timer: null,
    data: null
  };

  function $(id) {
    return document.getElementById(id);
  }

  function clampFrame(n) {
    n = Math.round(n);
    if (!isFinite(n)) n = 0;
    if (n < 0) n = 0;
    if (S.total > 0 && n > S.total - 1) n = S.total - 1;
    return n;
  }

  function contentWidth() {
    return Math.max(1, Math.ceil(S.total * S.ppf));
  }

  /* ---------- 占位数据：形状与 4.0 的 cutSegments / angleSegments 一致 ---------- */
  function buildData(total) {
    var d = { video: [], faces: [], cuts: [], angles: [] };
    var i, span, s, e;

    /* 视频帧条：40 个缩略占位块，均匀铺满整条轴 */
    span = total / N_VIDEO;
    for (i = 0; i < N_VIDEO; i++) {
      d.video.push({ s: Math.round(i * span), e: Math.round((i + 1) * span) });
    }

    /* 人脸标记：25 个宽窄不一的小块（真实数据接 state 里的人脸统计） */
    span = total / N_FACE;
    for (i = 0; i < N_FACE; i++) {
      s = Math.round(i * span + span * 0.10);
      e = Math.round(i * span + span * (0.55 + 0.035 * ((i * 7) % 10)));
      d.faces.push({ s: s, e: e > s ? e : s + 1 });
    }

    /* 剪切区段：6 段，长短交错（真实数据接 state.cutSegments） */
    var cutPlan = [
      [0.02, 0.09], [0.15, 0.21], [0.28, 0.375],
      [0.42, 0.50], [0.61, 0.70], [0.82, 0.94]
    ];
    for (i = 0; i < N_CUT; i++) {
      d.cuts.push({
        s: Math.round(cutPlan[i][0] * total),
        e: Math.round(cutPlan[i][1] * total)
      });
    }

    /* 角度色带：3 段（真实数据接 state.angleSegments，angles 同 4.0 默认值） */
    var anglePlan = [[0.0, 0.30], [0.30, 0.62], [0.62, 1.0]];
    var angleVal = ['0,90,180,270', '0,90,180,270', '0,45,90,135,180,225,270,315'];
    for (i = 0; i < N_ANGLE; i++) {
      d.angles.push({
        s: Math.round(anglePlan[i][0] * total),
        e: Math.round(anglePlan[i][1] * total),
        angles: angleVal[i]
      });
    }
    return d;
  }

  /* ---------- 渲染 ---------- */
  function addBlock(host, cls, s, e, title) {
    if (!host) return;
    var left = s * S.ppf;
    var w = (e - s) * S.ppf;
    var el = document.createElement('div');
    el.className = 'mtl-blk ' + cls;
    el.style.left = left + 'px';
    el.style.width = Math.max(2, w) + 'px';
    if (title) el.title = title;
    host.appendChild(el);
  }

  function clear(host) {
    if (host) host.innerHTML = '';
  }

  function renderRuler(w) {
    var host = $('mtlRuler');
    if (!host) return;
    clear(host);
    var px, num, tk, label;
    for (px = 0; px <= w; px += RULER_STEP) {
      num = Math.round(px / S.ppf);
      tk = document.createElement('div');
      tk.className = 'mtl-tk';
      tk.style.left = px + 'px';
      label = document.createElement('b');
      label.textContent = num;
      tk.appendChild(label);
      host.appendChild(tk);
    }
  }

  function renderVideo() {
    var host = $('mtlVideo');
    var i, d = S.data;
    if (!host || !d) return;
    clear(host);
    for (i = 0; i < d.video.length; i++) {
      addBlock(host, 'mtl-blk--video', d.video[i].s, d.video[i].e, '帧 ' + d.video[i].s + ' ~ ' + d.video[i].e);
    }
  }

  function renderFaces() {
    var host = $('mtlFaces');
    var i, d = S.data;
    if (!host || !d) return;
    clear(host);
    for (i = 0; i < d.faces.length; i++) {
      addBlock(host, 'mtl-blk--face', d.faces[i].s, d.faces[i].e, '人脸 帧 ' + d.faces[i].s + ' ~ ' + d.faces[i].e);
    }
  }

  function renderCut() {
    var host = $('mtlCut');
    var i, d = S.data;
    if (!host || !d) return;
    clear(host);
    for (i = 0; i < d.cuts.length; i++) {
      addBlock(host, 'mtl-blk--cut', d.cuts[i].s, d.cuts[i].e, '剪切 帧 ' + d.cuts[i].s + ' ~ ' + d.cuts[i].e);
    }
  }

  function renderAngle() {
    var host = $('mtlAngle');
    var i, d = S.data;
    if (!host || !d) return;
    clear(host);
    for (i = 0; i < d.angles.length; i++) {
      addBlock(host, 'mtl-blk--angle', d.angles[i].s, d.angles[i].e, '角度 ' + d.angles[i].angles);
    }
  }

  function renderAll() {
    var stack = $('mtlStack');
    if (!stack) return;
    var w = contentWidth();
    stack.style.width = w + 'px';
    renderRuler(w);
    renderVideo();
    renderFaces();
    renderCut();
    renderAngle();
    placeHead();
    var tot = $('mtlTotal');
    if (tot) tot.textContent = String(S.total);
  }

  function placeHead() {
    var head = $('mtlHead');
    if (head) head.style.left = (S.frame * S.ppf) + 'px';
  }

  /* 时间码：算法与 4.0 app.js updateTimecode 一致 → mm:ss.cs */
  function updateTimecode(idx) {
    var el = $('mtlTime');
    if (!el || !(S.fps > 0)) return;
    var s = idx / S.fps;
    var m = Math.floor(s / 60);
    s = Math.floor(s % 60);
    var cs = Math.floor((idx / S.fps % 1) * 100);
    el.textContent = (m < 10 ? '0' : '') + m + ':' + (s < 10 ? '0' : '') + s + '.' + (cs < 10 ? '0' : '') + cs;
  }

  function syncZoomSeg() {
    var seg = $('mtlZoomSeg');
    if (!seg) return;
    var items = seg.getElementsByTagName('b');
    var i, z;
    for (i = 0; i < items.length; i++) {
      z = parseFloat(items[i].getAttribute('data-zoom'));
      if (z === S.zoom) items[i].className = 'on';
      else items[i].className = '';
    }
  }

  function ensureVisible(px) {
    var sc = $('mtlScroll');
    if (!sc || !sc.clientWidth) return;
    var pad = 24;
    if (px < sc.scrollLeft) sc.scrollLeft = Math.max(0, px - pad);
    else if (px > sc.scrollLeft + sc.clientWidth - pad) sc.scrollLeft = px - sc.clientWidth + pad;
  }

  /* 通知宿主（如 app.js）：帧变化 */
  function fireFrame(n) {
    var root = $('mergeTracks');
    if (!root || typeof window.CustomEvent !== 'function') return;
    try {
      root.dispatchEvent(new CustomEvent('mtl:frame', { bubbles: true, detail: { frame: n, total: S.total } }));
    } catch (err) { /* 忽略：宿主未监听 */ }
  }

  /* ---------- 对外 API ---------- */
  function setFrame(n) {
    if (!S.ready) return 0;
    S.frame = clampFrame(n);
    placeHead();
    var inp = $('mtlFrame');
    if (inp) inp.value = String(S.frame);
    var tot = $('mtlTotal');
    if (tot) tot.textContent = String(S.total);
    updateTimecode(S.frame);
    ensureVisible(S.frame * S.ppf);
    fireFrame(S.frame);
    return S.frame;
  }

  function getFrame() {
    return S.frame;
  }

  function applyZoom(z) {
    z = parseFloat(z);
    if (!isFinite(z) || !PPF[String(z)]) z = 1;
    var sc = $('mtlScroll');
    var centerPx = 0;
    if (sc && sc.clientWidth) centerPx = sc.scrollLeft + sc.clientWidth / 2;
    var centerFrame = Math.round(centerPx / S.ppf);
    S.zoom = z;
    S.ppf = PPF[String(z)];
    syncZoomSeg();
    renderAll();
    if (sc && sc.clientWidth) {
      sc.scrollLeft = Math.max(0, Math.round(centerFrame * S.ppf - sc.clientWidth / 2));
    }
    setFrame(S.frame);
  }

  function zoomStep(dir) {
    /* dir -1 = 放大（走向 1:1），+1 = 缩小（走向 1/8） */
    var i, idx = 0;
    for (i = 0; i < ZOOM_ORDER.length; i++) {
      if (ZOOM_ORDER[i] === S.zoom) idx = i;
    }
    idx = idx + dir;
    if (idx < 0) idx = 0;
    if (idx > ZOOM_ORDER.length - 1) idx = ZOOM_ORDER.length - 1;
    applyZoom(ZOOM_ORDER[idx]);
  }

  /* ---------- 播放（占位驱动；4.0 由 video 元素驱动） ---------- */
  function setPlayIcon(on) {
    var btn = $('mtlPlay');
    if (btn) btn.innerHTML = on ? PAUSE_SVG : PLAY_SVG;
  }

  function stopPlay() {
    if (S.timer) {
      clearInterval(S.timer);
      S.timer = null;
    }
    S.playing = false;
    setPlayIcon(false);
  }

  function startPlay() {
    if (S.total <= 0) return;
    if (S.frame >= S.total - 1) setFrame(0);
    S.playing = true;
    setPlayIcon(true);
    var step = Math.max(30, Math.round(1000 / Math.max(1, S.fps)));
    S.timer = setInterval(function () {
      if (S.frame >= S.total - 1) {
        stopPlay();
        return;
      }
      setFrame(S.frame + 1);
    }, step);
  }

  function togglePlay() {
    if (S.playing) stopPlay();
    else startPlay();
  }

  /* ---------- 交互 ---------- */
  function frameAt(clientX) {
    var stack = $('mtlStack');
    if (!stack) return -1;
    var rect = stack.getBoundingClientRect();
    return clampFrame((clientX - rect.left) / S.ppf);
  }

  function onDown(e) {
    var sc = $('mtlScroll');
    if (!sc || e.button !== 0) return;
    if (e.target === sc) return;          /* 点在滚动条本体上：交给原生滚动 */
    S.dragging = true;
    stopPlay();
    var root = $('mergeTracks');
    if (root) root.className = root.className.indexOf('is-dragging') < 0 ? root.className + ' is-dragging' : root.className;
    setFrame(frameAt(e.clientX));
    e.preventDefault();
  }

  function onMove(e) {
    if (!S.dragging) return;
    setFrame(frameAt(e.clientX));
    e.preventDefault();
  }

  function onUp() {
    if (!S.dragging) return;
    S.dragging = false;
    var root = $('mergeTracks');
    if (root) root.className = root.className.replace(/\s*is-dragging/g, '');
  }

  function bindEvents() {
    var sc = $('mtlScroll');
    if (sc) {
      sc.addEventListener('mousedown', onDown);
      /* 4.0 行为：滚轮纵向滚动 → 时间轴横向滚动 */
      sc.addEventListener('wheel', function (e) {
        sc.scrollLeft += e.deltaY;
        e.preventDefault();
      }, { passive: false });
    }
    document.addEventListener('mousemove', onMove);
    document.addEventListener('mouseup', onUp);

    var inp = $('mtlFrame');
    if (inp) {
      inp.addEventListener('change', function () {
        var v = parseInt(this.value, 10);
        if (!isNaN(v)) setFrame(v);
        else setFrame(S.frame);
      });
      inp.addEventListener('keydown', function (e) {
        if (e.key === 'Enter') {
          var v = parseInt(this.value, 10);
          if (!isNaN(v)) setFrame(v);
          this.blur();
        }
      });
    }

    var prev = $('mtlPrev');
    if (prev) prev.addEventListener('click', function () { stopPlay(); setFrame(S.frame - 1); });

    var next = $('mtlNext');
    if (next) next.addEventListener('click', function () { stopPlay(); setFrame(S.frame + 1); });

    var play = $('mtlPlay');
    if (play) play.addEventListener('click', togglePlay);

    var zo = $('mtlZoomOut');
    if (zo) zo.addEventListener('click', function () { zoomStep(1); });

    var zi = $('mtlZoomIn');
    if (zi) zi.addEventListener('click', function () { zoomStep(-1); });

    var seg = $('mtlZoomSeg');
    if (seg) {
      seg.addEventListener('click', function (e) {
        var b = e.target;
        if (!b || b.tagName !== 'B') return;
        applyZoom(b.getAttribute('data-zoom'));
      });
    }
  }

  /* ---------- 初始化 ---------- */
  function init(totalFrames, fps) {
    var root = $('mergeTracks');
    if (!root) return null;
    S.total = Math.max(0, Math.floor(isFinite(totalFrames) ? totalFrames : 0));
    if (fps && fps > 0) S.fps = fps;
    if (!S.ready) {
      bindEvents();
      S.ready = true;
    }
    stopPlay();
    S.data = buildData(S.total);
    S.frame = 0;
    renderAll();
    return setFrame(0);
  }

  window.MergeTimeline = {
    init: init,
    setFrame: setFrame,
    getFrame: getFrame
  };

  /* 占位自启动：宿主未显式调用 init 时，先按 300 帧渲染一条演示时间轴；
     宿主随后调用 MergeTimeline.init(state.totalFrames) 会整条重绘覆盖。 */
  document.addEventListener('DOMContentLoaded', function () {
    if (!S.ready) init(300);
  });
})();
