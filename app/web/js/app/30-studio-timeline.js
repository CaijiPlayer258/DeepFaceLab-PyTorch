/* ===== 合成工作室 · 多轨时间轴（A）===== */
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
  /* 4.0 原来四档（1 / 0.5 / 0.25 / 0.125，对应 2 / 1 / 0.5 / 0.25 px/帧）。
     几万帧的素材在最小档下仍要 8899px —— 一屏看不全，还是得一直滚。
     补两档更小的（0.0625 / 0.03125 → 0.125 / 0.0625 px/帧）：
     35597 帧 × 0.0625 = 2225px，主流窗口一屏能看完整部。 */
  var ZOOM_ORDER = [1, 0.5, 0.25, 0.125, 0.0625, 0.03125];
  var PPF = { '1': 2, '0.5': 1, '0.25': 0.5, '0.125': 0.25,
              '0.0625': 0.125, '0.03125': 0.0625 };

  var RULER_STEP = 500;   /* 刻度尺每 500px 一个数字 */
  /* 视频轨以前铺 40 个均匀「缩略占位块」，纯装饰、和真实帧无关 ——
     用户明确要求不画（见 buildData）。这里不再保留占位块常量。 */

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
    /* 四条轨一律从「没有块」开始：
         · 视频轨 —— 以前那 40 个均匀占位块（缩略图条）是纯装饰，和真实帧一点关系
           都没有，而且一眼看上去像是「已经切好了片段」。用户要求直接不画，
           所以 video 恒为 []，只留刻度尺（renderRuler）和播放头。
         · 人脸 / 剪切 / 角度 —— 还没接后端就不造假数据，空轨道比假的红块诚实。
       有真实数据时由 MergeTimeline.setData({faces, cuts, angles}) 填。 */
    return { video: [], faces: [], cuts: [], angles: [] };
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
    /* ★ 刻度间隔自适应。以前固定每 500px 一个数字：ppf=2 时是每 250 帧，
       看着正常；缩到 0.0625（0.0625 px/帧）时 500px = 8000 帧，整条只标出三四个数，
       完全没法定位。现在先按「希望每 ~90px 一个刻度」算出帧间隔，
       再取整到 1/2/5×10^n 这种好读的数。 */
    var wantFrames = (w > 0) ? Math.max(1, Math.round(90 / Math.max(S.ppf, 1e-6))) : 1;
    var mag = Math.pow(10, Math.floor(Math.log10(wantFrames)));
    var norm = wantFrames / mag;
    var stepFrames = (norm <= 1 ? 1 : norm <= 2 ? 2 : norm <= 5 ? 5 : 10) * mag;
    var stepPx = Math.max(1, stepFrames * S.ppf);
    for (px = 0; px <= w; px += stepPx) {
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
    /* 没有块就整行收起（左边的「视频」标签一起收）—— 和其他三条轨同一套规矩。
       video 恒为 []，所以这一轨平时不出现；将来 setData({video:[…]}) 给了真数据
       会自己长回来。 */
    hideRow(host, !d.video.length);
    for (i = 0; i < d.video.length; i++) {
      addBlock(host, 'mtl-blk--video', d.video[i].s, d.video[i].e, '帧 ' + d.video[i].s + ' ~ ' + d.video[i].e);
    }
  }

  function renderFaces() {
    var host = $('mtlFaces');
    var i, d = S.data;
    if (!host || !d) return;
    clear(host);
    hideRow(host, !d.faces.length);
    for (i = 0; i < d.faces.length; i++) {
      addBlock(host, 'mtl-blk--face', d.faces[i].s, d.faces[i].e, '人脸 帧 ' + d.faces[i].s + ' ~ ' + d.faces[i].e);
    }
  }

  /* 轨道和它左边的标签名在两个不同的容器里，用序号对齐 */
  function hideRow(host, empty) {
    var stack = $('mtlStack');
    if (!host || !stack) return;
    var idx = Array.prototype.indexOf.call(stack.children, host);
    var names = document.querySelectorAll('#p-merge .mtl-name');
    if (idx >= 0 && names[idx]) names[idx].style.display = empty ? 'none' : '';
    host.style.display = empty ? 'none' : '';
  }

  function renderCut() {
    var host = $('mtlCut');
    var i, d = S.data;
    if (!host || !d) return;
    clear(host);
    hideRow(host, !d.cuts.length);
    for (i = 0; i < d.cuts.length; i++) {
      addBlock(host, 'mtl-blk--cut', d.cuts[i].s, d.cuts[i].e, '剪切 帧 ' + d.cuts[i].s + ' ~ ' + d.cuts[i].e);
    }
  }

  function renderAngle() {
    var host = $('mtlAngle');
    var i, d = S.data;
    if (!host || !d) return;
    clear(host);
    hideRow(host, !d.angles.length);
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

  /* 播放头定位：只写 transform，不动 left。
     播放时一秒钟 30 帧，改 left 会让浏览器每帧重排整条栈；translateX 只重合成。
     x = 帧 × 每帧像素（ppf），和刻度、轨道块、缩放共用同一个 S.ppf。
     CSS 里 .mtl-head 有 margin-left:-1px：2px 线的**中心**正好落在 x 上。 */
  function placeHead() {
    var head = $('mtlHead');
    if (!head) return;
    var x = S.frame * S.ppf;
    head.style.transform = 'translate3d(' + x + 'px,0,0)';
    head.setAttribute('data-x', String(x));
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
    /* 拖动中绝不能自动滚：滚动改 mtlStack 的 rect.left，下一帧 frameAt
       算出的帧号更大 → 再滚 → 正反馈，播放头会自己"飞"到末尾 */
    if (S && S.dragging) return;
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

  /* 播放状态变化也往外说一声：宿主拿它来「播放瞬间清框」——
     帧目录模式由时间轴自己 setInterval 步进，第一拍要 33ms 之后才到，
     光等 mtl:frame 会让旧框多挂一帧。 */
  function firePlay(on) {
    var root = $('mergeTracks');
    if (!root || typeof window.CustomEvent !== 'function') return;
    try {
      root.dispatchEvent(new CustomEvent('mtl:play', { bubbles: true, detail: { playing: !!on, frame: S.frame } }));
    } catch (err) { /* 忽略：宿主未监听 */ }
  }

  /* announce=1 才通知宿主「停了」：拖动/逐帧/换素材这些调用点后面紧跟一次
     setFrame（自己会开一条新链路），再补一发「暂停」只会白跑一趟检测 + 合成。 */
  function stopPlay(announce) {
    var was = S.playing;
    if (S.timer) {
      clearInterval(S.timer);
      S.timer = null;
    }
    S.playing = false;
    setPlayIcon(false);
    if (was && announce) firePlay(false);
  }

  function startPlay() {
    if (S.total <= 0) return;
    firePlay(true);                 /* 先通知：播放开始 = 帧马上要变 */
    S.playing = true;               /* 先立状态：下面 setFrame(0) 也要按「播放中」处理 */
    if (S.frame >= S.total - 1) setFrame(0);
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
    if (S.playing) stopPlay(true);
    else startPlay();
  }

  /* ---------- 交互 ---------- */
  /* 抓住播放头拖动时，指针和播放头之间会差几个像素；不补偿的话，
     1/4 缩放下（ppf=0.5）一按下就跳 8 帧。 */
  var HEAD_GRAB = 0;

  function headHit(t) {
    var h = $('mtlHead');
    return !!(h && t && (t === h || (h.contains && h.contains(t))));
  }

  function frameAt(clientX) {
    var stack = $('mtlStack');
    if (!stack) return -1;
    var rect = stack.getBoundingClientRect();
    return clampFrame((clientX - HEAD_GRAB - rect.left) / S.ppf);
  }

  function onDown(e) {
    var sc = $('mtlScroll');
    if (!sc || e.button !== 0) return;
    if (e.target === sc) return;          /* 点在滚动条本体上：交给原生滚动 */
    HEAD_GRAB = 0;
    if (headHit(e.target)) {
      /* 抓住把手：记下指针相对播放头的偏移，拖动时保持这个相对位置 */
      var stack0 = $('mtlStack');
      var hx = stack0 ? (stack0.getBoundingClientRect().left + S.frame * S.ppf) : e.clientX;
      HEAD_GRAB = e.clientX - hx;
    }
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
    HEAD_GRAB = 0;
    var root = $('mergeTracks');
    if (root) root.className = root.className.replace(/\s*is-dragging/g, '');
  }

  function bindEvents() {
    var sc = $('mtlScroll');
    if (sc) {
      sc.addEventListener('mousedown', onDown);
      /* 4.0 行为：滚轮纵向滚动 → 时间轴横向滚动 */
      sc.addEventListener('wheel', function (e) {
        /* ★ Ctrl + 滚轮 = 缩放时间轴（工业软件通行的快捷键）。
           不按 Ctrl 还是老行为：纵向滚 → 横向滚。
           几万帧的素材在 ppf=2 下要 7 万像素宽，一直滚轮翻不动 ——
           缩到 0.125（每帧 0.25px）一屏能看半部片。 */
        if (e.ctrlKey || e.metaKey) {
          e.preventDefault();
          var dir = (e.deltaY < 0) ? -1 : 1;      /* 上滚 = 放大 */
          var idx = ZOOM_ORDER.indexOf(S.zoom);
          if (idx < 0) idx = 0;
          var next = Math.max(0, Math.min(ZOOM_ORDER.length - 1, idx + dir));
          if (next !== idx) applyZoom(ZOOM_ORDER[next]);
          return;
        }
        sc.scrollLeft += e.deltaY;
        e.preventDefault();
      }, { passive: false });
    }
    document.addEventListener('mousemove', onMove);
    document.addEventListener('mouseup', onUp);

    var inp = $('mtlFrame');
    if (inp) {
      /* 边打边跳：只有「是数字并且不越界」才真的 setFrame，
         半截输入（''、'-'）、非数字（abc）、超尾帧都不动画面 */
      inp.addEventListener('input', function () {
        var v = parseInt(this.value, 10);
        if (isNaN(v) || v < 0 || v > S.total - 1) return;
        setFrame(v);
      });
      inp.addEventListener('change', function () {
        var v = parseInt(this.value, 10);
        if (!isNaN(v)) setFrame(v);
        /* 非法输入以前会原样留在框里：change 末尾一律拨回当前帧 */
        this.value = String(S.frame);
      });
      inp.addEventListener('keydown', function (e) {
        if (e.key === 'Enter') {
          var v = parseInt(this.value, 10);
          if (!isNaN(v)) setFrame(v);
          this.value = String(S.frame);
          this.blur();
        } else if (e.key === 'ArrowLeft') {
          e.preventDefault();
          stopPlay();
          setFrame(S.frame - (e.shiftKey ? 10 : 1));
          this.value = String(S.frame);
        } else if (e.key === 'ArrowRight') {
          e.preventDefault();
          stopPlay();
          setFrame(S.frame + (e.shiftKey ? 10 : 1));
          this.value = String(S.frame);
        } else if (e.key === 'Escape') {
          e.preventDefault();
          this.value = String(S.frame);
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

    /* 页面级快捷键：只在合成页激活（#p-merge 带 .on）时生效，且不抢输入控件/可编辑区的键。
       ←/→ 逐帧（Shift 十帧）；空格播放/暂停。 */
    document.addEventListener('keydown', function (e) {
      var page = document.getElementById('p-merge');
      if (!page || !page.classList || !page.classList.contains('on')) return;
      if (e.ctrlKey || e.metaKey || e.altKey) return;
      var t = e.target || {};
      var tag = String(t.tagName || '').toUpperCase();
      if (tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT' || t.isContentEditable) return;
      if (e.key === 'ArrowLeft') {
        e.preventDefault();
        stopPlay();
        setFrame(S.frame - (e.shiftKey ? 10 : 1));
      } else if (e.key === 'ArrowRight') {
        e.preventDefault();
        stopPlay();
        setFrame(S.frame + (e.shiftKey ? 10 : 1));
      } else if (e.key === ' ' || e.key === 'Spacebar') {
        e.preventDefault();
        togglePlay();
      }
    });
  }

  /* ---------- 初始化 ----------
     opts.keep = true：重进合成页的「续上」——保住播放头 S.frame 和已经算好的
     人脸轨 S.data.faces，只按新的 total 重排视频轨和刻度；事件不重复绑定。
     换素材时不要传 keep：人脸轨属于上一个素材，必须清掉，播放头归零。 */
  function init(totalFrames, fps, opts) {
    var root = $('mergeTracks');
    if (!root) return null;
    var keep = !!(opts && opts.keep);
    var oldFrame = S.frame;
    var oldFaces = (S.data && S.data.faces) ? S.data.faces : null;
    S.total = Math.max(0, Math.floor(isFinite(totalFrames) ? totalFrames : 0));
    if (fps && fps > 0) S.fps = fps;
    if (!S.ready) {
      bindEvents();
      S.ready = true;
    }
    stopPlay();
    S.data = buildData(S.total);
    if (keep && oldFaces) S.data.faces = oldFaces;
    /* keep 但没素材（total 0）时没有帧可停，回 0 */
    S.frame = (keep && S.total > 0) ? clampFrame(oldFrame) : 0;
    renderAll();
    return setFrame(S.frame);
  }

  window.MergeTimeline = {
    /* 验收/排查用：当前每帧像素、播放头的逻辑 x（= 帧 × ppf）、缩放档位 */
    getPpf: function () { return S.ppf; },
    /* 排查「帧数卡在某个值」用：一眼看出时间轴自己认为的总帧数、每帧像素、
       内容宽度、以及内部状态是否 ready。在 Console 里敲 MergeTimeline.diag() */
    diag: function () {
      var sc = document.getElementById('mtlScroll');
      return {
        total: S.total,
        frame: S.frame,
        zoom: S.zoom,
        ppf: S.ppf,
        contentWidth: contentWidth(),
        scrollLeft: sc ? Math.round(sc.scrollLeft) : null,
        clientWidth: sc ? Math.round(sc.clientWidth) : null,
        ready: S.ready,
        hostFrames: (window.__mv && window.__mv.frames) || 0,
        hostPath: (window.__mv && window.__mv.path) || ''
      };
    },
    getZoom: function () { return S.zoom; },
    getHeadX: function () { return S.frame * S.ppf; },
    isPlaying: function () { return !!S.playing; },
    seek: function (n) { stopPlay(); return setFrame(n); },
    /* 用真实数据覆盖演示数据（人脸轨、剪切段、角度带） */
    setData: function (partial) {
      if (!S.data || !partial) return;
      if (partial.video) S.data.video = partial.video;
      if (partial.faces) S.data.faces = partial.faces;
      if (partial.cuts) S.data.cuts = partial.cuts;
      if (partial.angles) S.data.angles = partial.angles;
      renderAll();
    },
    getData: function () { return S.data; },
    init: init,
    setFrame: setFrame,
    getFrame: getFrame
  };

  /* 占位自启动：宿主未显式调用 init 时先把空轴渲染出来（total 0）。
     以前这里按 300 帧渲染演示数据 —— 素材还没打开就显示 300 帧，全是假的；
     宿主随后调用 MergeTimeline.init(state.totalFrames) 会整条重绘覆盖。 */
  document.addEventListener('DOMContentLoaded', function () {
    if (!S.ready) init(0);
  });
})();
