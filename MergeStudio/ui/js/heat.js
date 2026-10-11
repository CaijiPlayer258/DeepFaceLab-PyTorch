// 热区 UI + 上下文播放路由 + 慢放引擎（融入主 UI，无新按钮）
// 依赖：App.loadFrame / Preview / cache-status regions
var Heat = {
  regions: [],            // [{start,end,cached,total,pct}]
  pollTimer: null,
  slowTimer: null,
  playing: false,

  // ── 热区数据轮询（600ms：预取推进可见） ──
  startPolling: function () {
    var self = this;
    this.stopPolling();
    this.pollTimer = setInterval(function () {
      fetch('/api/preview/cache-status').then(function (r) { return r.json(); }).then(function (s) {
        self.regions = s.regions || [];
        self._render();
      }).catch(function () {});
    }, 600);
  },
  stopPolling: function () {
    if (this.pollTimer) { clearInterval(this.pollTimer); this.pollTimer = null; }
    this._render();
  },

  // ── Video 轨道热区渲染（轨道底色叠加，非新轨道） ──
  _render: function () {
    var track = document.getElementById('track-video');
    if (!track) return;
    var old = document.getElementById('heat-overlay');
    if (old) old.remove();
    if (!this.regions.length || !App.state.totalFrames) return;
    var total = App.state.totalFrames;
    var layer = document.createElement('div');
    layer.id = 'heat-overlay';
    layer.style.cssText = 'position:absolute;inset:0;pointer-events:none;z-index:1;';
    this.regions.forEach(function (rg) {
      var el = document.createElement('div');
      var l = (rg.start / total) * 100, w = Math.max(0.2, ((rg.end - rg.start + 1) / total) * 100);
      var alpha = 0.18 + 0.30 * (rg.pct / 100);           // 预热越满越亮
      var inProgress = rg.pct < 99.5;
      el.style.cssText = 'position:absolute;left:' + l + '%;width:' + w + '%;top:0;bottom:0;' +
        'background:rgba(80,200,255,' + alpha + ');' +
        (inProgress ? 'border-right:2px solid rgba(80,200,255,0.8);' : '') +   // 推进边
        'transition:background .3s;';
      layer.appendChild(el);
    });
    var anchor = track.querySelector('.playhead');
    if (anchor && anchor.parentNode) {
      anchor.parentNode.insertBefore(layer, anchor);
    } else {
      track.appendChild(layer);
    }
  },

  // ── 当前帧是否在某热区内（且基本预热完） ──
  inHeat: function (idx, minPct) {
    minPct = (minPct === undefined) ? 90 : minPct;
    for (var i = 0; i < this.regions.length; i++) {
      var rg = this.regions[i];
      if (idx >= rg.start && idx <= rg.end && rg.pct >= minPct) return true;
    }
    return false;
  },

  // ── 慢放引擎：链式驱动（上一帧渲染完成才进下一帧，杜绝 _frameGen 互相作废） ──
  startSlow: function () {
    var self = this;
    if (this.slowTimer) return;
    this.playing = true;
    App._setPlayButtonIcon(true);
    this._advance();
  },
  _advance: function () {
    var self = this;
    if (!this.playing) return;
    var next = App.state.currentFrame + 1;
    if (next >= App.state.totalFrames) { this.stopSlow(); return; }

    // 热区维护：接近边缘/出区 → 预声明后续段（预取追着播放跑）
    var rg = this._regionOf(next);
    if ((rg && next > rg.end - 30) || !rg) {
      var s = rg ? rg.end + 1 : next;
      fetch('/api/preview/declare-heat', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ start: s, end: Math.min(s + 200, App.state.totalFrames - 1) })
      }).catch(function () {});
    }

    // 时间线先行（便宜），帧渲染等完成再续
    try { App.seekFrame(next, true); } catch (e) {}

    App.loadFrame(next);
    // 轮询等这一帧渲染完成（_frameGen 不再变化且 loading 全灭），
    // 再按节奏推进——链式，永不互相作废
    var t0 = Date.now();
    var wait = function () {
      if (!self.playing) return;
      // 判定：Swapped 面板 loading 消失（链路末端，或 opacity 过渡中）=整帧完成
      var ls = document.getElementById('loading-swapped');
      var done = !ls || ls.style.display === 'none' || ls.style.opacity === '0';
      if (done || Date.now() - t0 > 4000) {        // 完成或超时兜底
        self.slowTimer = setTimeout(function () { self._advance(); },
                                    self._adaptiveDelay());
      } else {
        self.slowTimer = setTimeout(wait, 60);
      }
    };
    this.slowTimer = setTimeout(wait, 80);
  },
  stopSlow: function () {
    this.playing = false;
    if (this.slowTimer) { clearTimeout(this.slowTimer); this.slowTimer = null; }
    App._setPlayButtonIcon(false);
  },

  _regionOf: function (idx) {
    for (var i = 0; i < this.regions.length; i++) {
      var rg = this.regions[i];
      if (idx >= rg.start && idx <= rg.end) return rg;
    }
    return null;
  },
  _adaptiveDelay: function () {
    // 慢放步进间隔：热且满=220ms(~4.5fps)，预热中=500ms，热区外=700ms
    var idx = App.state.currentFrame;
    var rg = this._regionOf(idx);
    if (rg && rg.pct >= 95) return 220;
    if (rg) return 500;
    return 700;
  }
};
