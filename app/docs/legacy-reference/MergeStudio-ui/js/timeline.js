(function () {
  'use strict';

  window.Timeline = {
    _pxPerFrame: 1,
    visibleStart: 0,
    visibleEnd: 0,

    _log: function (msg) {
      fetch('/api/preview/log?msg=' + encodeURIComponent(msg)).catch(function(){});
    },

    _ctxMenu: null,  // right-click context menu element

    _showContextMenu: function (x, y, label, callback) {
      var self = this;
      this._hideContextMenu();
      var menu = document.createElement('div');
      menu.className = 'timeline-ctx-menu';
      menu.style.cssText = 'position:fixed;left:'+x+'px;top:'+y+'px;z-index:9999;background:#1a1a1e;border:1px solid #2a2a2e;border-radius:6px;padding:4px 0;min-width:100px;box-shadow:0 4px 12px rgba(0,0,0,0.4);';
      var item = document.createElement('div');
      item.textContent = label;
      item.style.cssText = 'padding:6px 16px;cursor:pointer;font-size:12px;color:#e0e0e0;';
      item.addEventListener('mouseenter', function () { item.style.background = '#5b5bd6'; });
      item.addEventListener('mouseleave', function () { item.style.background = ''; });
      item.addEventListener('click', function () { self._hideContextMenu(); if (callback) callback(); });
      menu.appendChild(item);
      document.body.appendChild(menu);
      this._ctxMenu = menu;
      // Dismiss on click outside
      setTimeout(function () {
        document.addEventListener('click', self._hideContextMenu.bind(self), {once: true});
      }, 0);
    },

    _hideContextMenu: function () {
      if (this._ctxMenu) { this._ctxMenu.remove(); this._ctxMenu = null; }
    },

    init: function () {
      this._ensureTrackInner();
      this._initScrollSync();
      this._initTrackDrag();  // unified click + drag
      this.initCutButton();
      this.initAngleButton();
      this.renderLegend();
      this._initKeyboardDelete();
    },

    _initKeyboardDelete: function () {
      var self = this;
      document.addEventListener('keydown', function (e) {
        if (e.key === 'Delete' || e.key === 'Del') {
          // Angle segment delete
          var aidx = self._selectedAngleIdx;
          if (aidx !== undefined && aidx !== null) {
            var as = window.App.state.angleSegments;
            if (as && aidx < as.length) {
              as.splice(aidx, 1);
              window.App.state.angleSegments = as;
              self._selectedAngleIdx = null;
              self.updateAngleSegments();
              window.App._saveState && window.App._saveState();
              self._log('Angle segment deleted via Del key, re-detecting');
              var a2 = window.App;
              if (a2) { a2._detectGen = (a2._detectGen || 0) + 1; a2.loadFrame(a2.state.currentFrame); }
            }
            return;
          }
          // Cut segment delete
          var cidx = self._selectedCutIdx;
          if (cidx !== undefined && cidx !== null) {
            var cs = window.App.state.cutSegments;
            if (cs && cidx < cs.length) {
              cs.splice(cidx, 1);
              window.App.state.cutSegments = cs;
              self._selectedCutIdx = null;
              self.updateCutSegments();
            }
          }
        }
      });
    },

    _ensureTrackInner: function () {
      ['track-video', 'track-faces', 'track-cut', 'track-angle', 'track-scrollbar'].forEach(function (id) {
        var tc = document.getElementById(id);
        if (!tc) return;
        if (tc.querySelector('.track-inner')) return;
        var inner = document.createElement('div');
        inner.className = 'track-inner';
        while (tc.firstChild) inner.appendChild(tc.firstChild);
        tc.appendChild(inner);
      });
      var rn = document.getElementById('ruler-numbers');
      if (rn && !rn.querySelector('.track-inner')) {
        var inner = document.createElement('div');
        inner.className = 'track-inner';
        while (rn.firstChild) inner.appendChild(rn.firstChild);
        rn.appendChild(inner);
      }
    },

    _initScrollSync: function () {
      var self = this;
      var sb = document.getElementById('track-scrollbar');
      if (!sb) return;
      sb.addEventListener('scroll', function () {
        var scrollL = sb.scrollLeft;
        ['track-video', 'track-faces', 'track-cut', 'track-angle', 'ruler-numbers'].forEach(function (id) {
          var el = document.getElementById(id);
          if (el) el.scrollLeft = scrollL;
        });
        self._updateVisibleRange();
        if (window.App) {
          self.updatePlayhead(window.App.state.currentFrame, window.App.state.totalFrames);
        }
      });
    },

    // Unified drag handler: mousedown starts, mousemove updates, mouseup finalizes
    _initTrackDrag: function () {
      var self = this;
      var dragging = false;
      var lastFrame = -1;
      var mmTarget = null;

      var getFrame = function (clientX) {
        var inner = mmTarget ? mmTarget.querySelector('.track-inner') : null;
        if (!inner) return -1;
        var iRect = inner.getBoundingClientRect();
        var fullW = inner.offsetWidth || 1;
        // Use scrollbar's scrollLeft (always accurate, even before first scroll sync)
        var sb = document.getElementById('track-scrollbar');
        var scrollL = sb ? sb.scrollLeft || 0 : 0;
        var pct = Math.max(0, Math.min(1, (clientX - iRect.left + scrollL) / fullW));
        var total = window.App ? window.App.state.totalFrames : 0;
        if (total <= 0) return -1;
        return Math.max(0, Math.min(total - 1, Math.round(pct * total)));
      };

      var updateFrame = function (f) {
        if (f < 0) return;
        if (f === lastFrame) return;
        lastFrame = f;
        window.App.state.currentFrame = f;
        document.getElementById('frame-input').value = f;
        // Update playhead position
        var ph = document.getElementById('playhead');
        if (ph) {
          var inner = mmTarget ? mmTarget.querySelector('.track-inner') : null;
          var fullW = inner ? inner.offsetWidth : 1;
          var total = window.App.state.totalFrames || 1;
          ph.style.left = (f / total * fullW) + 'px';
          ph.style.position = 'absolute';
        }
        window.App.updateTimecode(f);
      };

      // mousedown on any track content
      ['track-video', 'track-faces', 'track-cut', 'track-angle', 'track-scrollbar'].forEach(function (id) {
        var el = document.getElementById(id);
        if (!el) return;
        el.addEventListener('mousedown', function (e) {
          if (e.target.closest('.cut-segment') || e.target.closest('.angle-segment') || e.target.closest('button')) return;
          dragging = true;
          mmTarget = ['track-video', 'track-faces', 'track-cut', 'track-angle'].indexOf(id) >= 0 ? document.getElementById(id) : null;
          lastFrame = -1;
          updateFrame(getFrame(e.clientX));
          e.preventDefault();
        });
        el.addEventListener('wheel', function (e) {
          var sb = document.getElementById('track-scrollbar');
          if (sb) sb.scrollLeft += e.deltaY;
          e.preventDefault();
        }, { passive: false });
      });

      // mousemove on document for smooth drag
      document.addEventListener('mousemove', function (e) {
        if (!dragging || !mmTarget) return;
        updateFrame(getFrame(e.clientX));
      });

      // mouseup on document
      document.addEventListener('mouseup', function (e) {
        if (!dragging) return;
        dragging = false;
        var finalFrame = getFrame(e.clientX);
        if (finalFrame >= 0 && window.App) {
          window.App.seekFrame(finalFrame);
        }
        mmTarget = null;
        lastFrame = -1;
      });
    },

    _updateVisibleRange: function () {
      var sb = document.getElementById('track-scrollbar');
      if (!sb) return;
      var containerWidth = sb.clientWidth;
      var scrollL = sb.scrollLeft;
      var startFrame = Math.round(scrollL / this._pxPerFrame);
      var endFrame = Math.round((scrollL + containerWidth) / this._pxPerFrame);
      var total = window.App ? window.App.state.totalFrames : 0;
      this.visibleStart = Math.max(0, startFrame);
      this.visibleEnd = Math.min(total, endFrame);
    },

    initCutButton: function () {
      var self = this;
      var btn = document.getElementById('btn-add-cut');
      if (!btn) return;
      btn.addEventListener('click', function () {
        var app = window.App;
        if (!app) return;
        var total = app.state.totalFrames;
        if (total <= 0) return;
        var cur = app.state.currentFrame;
        var end = Math.min(total - 1, cur + Math.max(30, Math.round(total * 0.01)));
        console.log('[Cut] currentFrame:', cur, 'end:', end, 'total:', total);
        var segs = app.state.cutSegments || [];
        var merged = false;
        for (var i = 0; i < segs.length; i++) {
          var s = segs[i];
          if (end >= s.start - 1 && cur <= s.end + 1) {
            s.start = Math.min(s.start, cur);
            s.end = Math.max(s.end, end);
            merged = true;
            break;
          }
        }
        if (!merged) segs.push({start: cur, end: end});
        app.state.cutSegments = segs;
        self.updateCutSegments();
      });
    },

    initAngleButton: function () {
      var self = this;
      var btn = document.getElementById('btn-add-angle');
      if (!btn) return;
      if (btn._angleInit) return;  // prevent double init
      btn._angleInit = true;
      btn.addEventListener('click', function () {
        self._log('Angle + clicked');
        var app = window.App;
        if (!app) return;
        var total = app.state.totalFrames;
        if (total <= 0) return;
        var segs = app.state.angleSegments || [];
        var cur = app.state.currentFrame;
        var end = Math.min(total - 1, cur + Math.max(30, Math.round(total * 0.01)));
        var anglesStr = prompt('输入检测角度（逗号分隔，如 0,90,180,270）：', '0,90,180,270');
        self._log('Angle prompt returned: ' + anglesStr);
        if (!anglesStr) return;
        // Merge with existing overlapping segments
        var merged = false;
        for (var i = 0; i < segs.length; i++) {
          var s = segs[i];
          if (end >= s.start - 1 && cur <= s.end + 1) {
            s.start = Math.min(s.start, cur);
            s.end = Math.max(s.end, end);
            s.angles = anglesStr;
            merged = true;
            break;
          }
        }
        if (!merged) segs.push({start: cur, end: end, angles: anglesStr});
        app.state.angleSegments = segs;
        self.updateAngleSegments();
        app._saveState && app._saveState();
        self._log('Angle segment added: ' + JSON.stringify(segs));
        // Force re-analyze with new angle settings
        self._log('Angle forcing loadFrame for re-detect');
        app._detectGen = (app._detectGen || 0) + 1;
        app.loadFrame(app.state.currentFrame);
      });
    },

    updateAngleSegments: function () {
      this._selectedAngleIdx = null;
      var app = window.App;
      if (!app) return;
      var segs = app.state.angleSegments || [];
      var ppf = this._pxPerFrame;
      var pxSegs = segs.map(function (s) {
        return {startPx: s.start * ppf, endPx: s.end * ppf, angles: s.angles || '0'};
      });
      this.renderAngleSegments(pxSegs);
    },

    renderAngleSegments: function (pxSegs) {
      var self = this;
      var container = document.getElementById('track-angle');
      var inner = container ? container.querySelector('.track-inner') : null;
      if (!inner) return;
      inner.querySelectorAll('.angle-segment').forEach(function (s) { s.remove(); });

      pxSegs.forEach(function (ps, idx) {
        var width = Math.max(2, ps.endPx - ps.startPx);
        var seg = document.createElement('div');
        seg.className = 'angle-segment';
        seg.style.left = ps.startPx + 'px';
        seg.style.width = width + 'px';
        seg.innerHTML = '<span class="angle-label">' + ps.angles + '</span>';

        // Handles
        var hl = document.createElement('div');
        hl.className = 'angle-handle';
        var hr = document.createElement('div');
        hr.className = 'angle-handle angle-handle--right';

        // Right-click context menu → delete
        seg.addEventListener('contextmenu', function (e) {
          e.preventDefault();
          e.stopPropagation();
          self._showContextMenu(e.clientX, e.clientY, '删除', function () {
          var segs = window.App.state.angleSegments;
          if (!segs) return;
          segs.splice(idx, 1);
          window.App.state.angleSegments = segs;
          self.updateAngleSegments();
          window.App._saveState && window.App._saveState();
          self._log('Angle segment deleted, re-detecting');
          var a2 = window.App;
          if (a2) { a2._detectGen = (a2._detectGen || 0) + 1; a2.loadFrame(a2.state.currentFrame); }
        });
        });
        // Left-click selects segment for Delete key
        seg.addEventListener('click', function (e) {
          self._selectedAngleIdx = idx;
        });

        // Drag reposition
        var startX = 0, origStart = 0, origEnd = 0;
        var body = seg;
        seg.addEventListener('mousedown', function (e) {
          if (e.target.classList.contains('angle-handle')) return;
          e.stopPropagation();
          startX = e.clientX;
          origStart = parseFloat(body.style.left) || 0;
          origEnd = origStart + (parseFloat(body.style.width) || 0);
          var onMove = function (ev) {
            var dx = ev.clientX - startX;
            body.style.left = Math.max(0, origStart + dx) + 'px';
          };
          var onUp = function () {
            document.removeEventListener('mousemove', onMove);
            document.removeEventListener('mouseup', onUp);
            var segs = window.App.state.angleSegments;
            if (segs && segs[idx]) {
              var newLeft = parseFloat(body.style.left) || 0;
              var newW = parseFloat(body.style.width) || 0;
              var newStart = Math.max(0, Math.round(newLeft / self._pxPerFrame));
              var newEnd = Math.min(window.App.state.totalFrames, Math.round((newLeft + newW) / self._pxPerFrame));
              self._log('Angle drag: left=' + newLeft + ' w=' + newW + ' ppf=' + self._pxPerFrame + ' start=' + newStart + ' end=' + newEnd);
              segs[idx].start = newStart;
              segs[idx].end = newEnd;
              window.App.state.angleSegments = segs;
              window.App._saveState && window.App._saveState();
              self._log('Angle segment resized, re-detecting');
              var a2 = window.App;
              if (a2 && a2.loadFrame) { a2._detectGen = (a2._detectGen || 0) + 1; a2.loadFrame(a2.state.currentFrame); }
            }
          };
          document.addEventListener('mousemove', onMove);
          document.addEventListener('mouseup', onUp);
        });

        // Handle resize
        hl.addEventListener('mousedown', function (e) {
          e.stopPropagation();
          startX = e.clientX;
          origStart = parseFloat(body.style.left) || 0;
          origEnd = origStart + (parseFloat(body.style.width) || 0);
          var onMove = function (ev) {
            var dx = ev.clientX - startX;
            var nl = Math.min(origEnd - 2, origStart + dx);
            body.style.left = nl + 'px';
            body.style.width = (origEnd - nl) + 'px';
          };
          var onUp = function () {
            document.removeEventListener('mousemove', onMove);
            document.removeEventListener('mouseup', onUp);
            var s = window.App.state.angleSegments;
            if (s && s[idx]) {
              var l2 = parseFloat(body.style.left) || 0;
              var w2 = parseFloat(body.style.width) || 0;
              s[idx].start = Math.max(0, Math.round(l2 / self._pxPerFrame));
              s[idx].end = Math.min(window.App.state.totalFrames, Math.round((l2 + w2) / self._pxPerFrame));
              window.App.state.angleSegments = s;
              window.App._saveState && window.App._saveState();
              self._log('Angle handle done: start=' + s[idx].start + ' end=' + s[idx].end);
              self.updateAngleSegments();
              var a3 = window.App;
              if (a3 && a3.loadFrame) { a3._detectGen = (a3._detectGen || 0) + 1; a3.loadFrame(a3.state.currentFrame); }
            }
          };
          document.addEventListener('mousemove', onMove);
          document.addEventListener('mouseup', onUp);
        });
        hr.addEventListener('mousedown', function (e) {
          e.stopPropagation();
          startX = e.clientX;
          var ow = parseFloat(body.style.width) || 0;
          var onMove = function (ev) {
            var dx = ev.clientX - startX;
            body.style.width = Math.max(2, ow + dx) + 'px';
          };
          var onUp = function () {
            document.removeEventListener('mousemove', onMove);
            document.removeEventListener('mouseup', onUp);
            var s = window.App.state.angleSegments;
            if (s && s[idx]) {
              var l2 = parseFloat(body.style.left) || 0;
              var w2 = parseFloat(body.style.width) || 0;
              s[idx].end = Math.min(window.App.state.totalFrames, Math.round((l2 + w2) / self._pxPerFrame));
              window.App.state.angleSegments = s;
              window.App._saveState && window.App._saveState();
              self._log('Angle handle done: start=' + s[idx].start + ' end=' + s[idx].end);
              self.updateAngleSegments();
              var a3 = window.App;
              if (a3 && a3.loadFrame) { a3._detectGen = (a3._detectGen || 0) + 1; a3.loadFrame(a3.state.currentFrame); }
            }
          };
          document.addEventListener('mousemove', onMove);
          document.addEventListener('mouseup', onUp);
        });

        seg.appendChild(hl);
        seg.appendChild(hr);
        inner.appendChild(seg);
      });
    },

    renderLegend: function () {
      var el = document.getElementById('timeline-legend');
      if (!el) return;
      el.innerHTML =
        '<span><span class="legend-swatch" style="background:#2a0c4a"></span> 多人脸</span>' +
        '<span><span class="legend-swatch" style="background:#8a6caa"></span> 少人脸</span>' +
        '<span><span class="legend-swatch" style="background:#4c1010"></span> 无人脸</span>' +
        '<span><span class="legend-swatch" style="background:#5b5bd6"></span> 选中人脸</span>';
    },

    updateZoom: function (zoom, totalFrames, currentFrame) {
      if (totalFrames <= 0) return;
      var sb = document.getElementById('track-scrollbar');
      var containerWidth = sb ? sb.clientWidth : 800;
      var basePx = Math.max(0.01, containerWidth / Math.max(30, totalFrames));
      this._pxPerFrame = Math.max(0.001, basePx * zoom);
      var innerWidth = Math.ceil(totalFrames * this._pxPerFrame);

      ['ruler-numbers', 'track-video', 'track-faces', 'track-cut', 'track-angle', 'track-scrollbar'].forEach(function (id) {
        var inner = document.querySelector('#' + id + ' .track-inner');
        if (inner) inner.style.width = innerWidth + 'px';
      });

      // Update ruler numbers
      var rn = document.querySelector('#ruler-numbers .track-inner');
      if (rn) {
        rn.innerHTML = '';
        var step = Math.max(1, Math.floor(containerWidth / this._pxPerFrame / 20));
        for (var i = 0; i <= totalFrames; i += step) {
          var num = document.createElement('span');
          num.className = 'ruler-number';
          num.style.left = (i * this._pxPerFrame) + 'px';
          num.textContent = i;
          rn.appendChild(num);
        }
      }

      this.updatePlayhead(currentFrame, totalFrames);
      this.updateCutSegments();
      this.updateAngleSegments();
      this._syncScroll(currentFrame);
    },

    _syncScroll: function (currentFrame) {
      var total = window.App ? window.App.state.totalFrames : 0;
      if (total <= 0) return;
      var targetX = currentFrame * this._pxPerFrame;
      var sb = document.getElementById('track-scrollbar');
      if (sb) {
        var halfW = sb.clientWidth / 2;
        sb.scrollLeft = Math.max(0, targetX - halfW);
      }
    },

    updatePlayhead: function (frameIdx, totalFrames) {
      var ph = document.getElementById('playhead');
      if (!ph) return;
      var ppf = this._pxPerFrame;
      ph.style.left = (frameIdx * ppf) + 'px';
      ph.style.position = 'absolute';
    },

    renderFaceDensityLine: function (faceData) {
      var container = document.getElementById('track-faces');
      if (!container) return;
      var inner = container.querySelector('.track-inner');
      if (!inner) return;
      inner.innerHTML = '<div class="face-density-line"></div>';
      var line = inner.querySelector('.face-density-line');
      if (!line) return;

      if (!faceData || faceData.length === 0) {
        var seg = document.createElement('div');
        seg.className = 'face-density-line__seg';
        seg.style.flex = '1';
        seg.style.background = '#0d0d0f';
        line.appendChild(seg);
        return;
      }

      var total = window.App ? window.App.state.totalFrames : faceData.length;
      var ppf = this._pxPerFrame;
      line.style.display = 'flex';
      line.style.width = (total * ppf) + 'px';

      faceData.forEach(function (d) {
        var seg = document.createElement('div');
        seg.className = 'face-density-line__seg';
        var count = d.face_count || 0;
        if (count === 0) seg.style.background = '#2a0c4a';
        else if (count < 3) seg.style.background = '#8a6caa';
        else seg.style.background = '#4c1010';
        seg.style.flex = d.count || 1;
        line.appendChild(seg);
      });
    },

    updateCutSegments: function () {
      this._selectedCutIdx = null;
      var app = window.App;
      if (!app) return;
      var segs = app.state.cutSegments || [];
      var ppf = this._pxPerFrame;
      var pxSegs = segs.map(function (s) {
        return {startPx: s.start * ppf, endPx: s.end * ppf};
      });
      this.renderCutSegments(pxSegs);
    },

    renderCutSegments: function (pxSegs) {
      var self = this;
      var container = document.getElementById('track-cut');
      var inner = container ? container.querySelector('.track-inner') : null;
      if (!inner) return;
      // Remove existing segments but keep the inner element
      inner.querySelectorAll('.cut-segment').forEach(function (s) { s.remove(); });

      pxSegs.forEach(function (ps, idx) {
        var width = Math.max(2, ps.endPx - ps.startPx);
        var seg = document.createElement('div');
        seg.className = 'cut-segment';
        seg.style.left = ps.startPx + 'px';
        seg.style.width = width + 'px';

        // Handles
        var hl = document.createElement('div');
        hl.className = 'cut-handle';
        var hr = document.createElement('div');
        hr.className = 'cut-handle cut-handle--right';

        // Right-click → delete
        seg.addEventListener('contextmenu', function (e) {
          e.preventDefault();
          self._showContextMenu(e.clientX, e.clientY, '删除', function () {
            var segs = window.App.state.cutSegments;
            if (!segs) return;
            segs.splice(idx, 1);
            window.App.state.cutSegments = segs;
            self._selectedCutIdx = null;
            self.updateCutSegments();
          });
        });
        // Click to select for Delete key
        seg.addEventListener('click', function (e) {
          e.stopPropagation();
          self._selectedCutIdx = idx;
        });

        // Drag reposition
        var startX = 0, origStart = 0, origEnd = 0;
        var body = seg;
        seg.addEventListener('mousedown', function (e) {
          if (e.target.classList.contains('cut-handle')) return;
          e.stopPropagation();
          startX = e.clientX;
          origStart = parseFloat(body.style.left) || 0;
          origEnd = origStart + (parseFloat(body.style.width) || 0);
          var onMove = function (ev) {
            var dx = ev.clientX - startX;
            body.style.left = Math.max(0, origStart + dx) + 'px';
          };
          var onUp = function () {
            document.removeEventListener('mousemove', onMove);
            document.removeEventListener('mouseup', onUp);
            var segs = window.App.state.cutSegments;
            if (segs && segs[idx]) {
              segs[idx].start = Math.max(0, Math.round(parseFloat(body.style.left) / self._pxPerFrame));
              segs[idx].end = Math.min(window.App.state.totalFrames, Math.round((parseFloat(body.style.left) + parseFloat(body.style.width)) / self._pxPerFrame));
              window.App.state.cutSegments = segs;
            }
          };
          document.addEventListener('mousemove', onMove);
          document.addEventListener('mouseup', onUp);
        });

        // Handle resize
        hl.addEventListener('mousedown', function (e) {
          e.stopPropagation();
          startX = e.clientX;
          origStart = parseFloat(body.style.left) || 0;
          origEnd = origStart + (parseFloat(body.style.width) || 0);
          var onMove = function (ev) {
            var dx = ev.clientX - startX;
            var nl = Math.min(origEnd - 2, origStart + dx);
            body.style.left = nl + 'px';
            body.style.width = (origEnd - nl) + 'px';
          };
          var onUp = function () { document.removeEventListener('mousemove', onMove); document.removeEventListener('mouseup', onUp); };
          document.addEventListener('mousemove', onMove);
          document.addEventListener('mouseup', onUp);
        });
        hr.addEventListener('mousedown', function (e) {
          e.stopPropagation();
          startX = e.clientX;
          var ow = parseFloat(body.style.width) || 0;
          var onMove = function (ev) {
            var dx = ev.clientX - startX;
            body.style.width = Math.max(2, ow + dx) + 'px';
          };
          var onUp = function () { document.removeEventListener('mousemove', onMove); document.removeEventListener('mouseup', onUp); };
          document.addEventListener('mousemove', onMove);
          document.addEventListener('mouseup', onUp);
        });

        seg.appendChild(hl);
        seg.appendChild(hr);
        inner.appendChild(seg);
      });
    },
  };

  document.addEventListener('DOMContentLoaded', function () { window.Timeline.init(); });
})();
