/**
 * MaskProcessor — Main Application
 *
 * Orchestrates MaskCanvas, tools, API, and UI event binding.
 * Global class attached to window.  No ES modules.
 *
 * Depends on (in order):
 *   window.API        – from api.js
 *   window.MaskCanvas – from canvas.js
 *   window.PointTool  – from tools/point.js
 *   window.BoxTool    – from tools/box.js
 *   window.BrushTool  – from tools/brush.js
 *   window.PenTool    – from tools/pen.js
 */
(function () {
  'use strict';

  // ==========================================================================
  // App
  // ==========================================================================

  class App {

    constructor () {
      this.canvas = new MaskCanvas(document.getElementById('mask-canvas'));
      this.tools        = {};
      this.currentTool  = 'sam';
      this.currentIndex = 0;
      this.files        = [];
      this.history      = [];
      this._navCooldown  = false;   // [{mask, points}] undo stack

      this._setupUI();
      this._registerTools();
      this._setupShortcuts();
      this._setupEventListeners();
    }

    // ========================================================================
    // Initialization
    // ========================================================================

    /** Bind UI elements — tool buttons, mode toggle, brush slider. */
    _setupUI () {
      var self = this;

      try {

      // ---- Tool buttons ----------------------------------------------------
      document.querySelectorAll('.tool-btn').forEach(function (btn) {
        btn.addEventListener('click', function () {
          self.setTool(btn.dataset.tool);
        });
      });

      // ---- Mode toggle -----------------------------------------------------
      document.querySelectorAll('.mode-toggle__btn').forEach(function (btn) {
        btn.addEventListener('click', function () {
          var parent = btn.parentElement;
          var active = parent.querySelector('.active');
          if (active) active.classList.remove('active');
          btn.classList.add('active');
          self.canvas.setMode(btn.dataset.mode);
        });
      });

      // ---- Brush slider ----------------------------------------------------
      var slider = document.getElementById('brush-size');
      if (slider) {
        var label = document.getElementById('brush-size-label');

        slider.addEventListener('input', function () {
          var val = parseInt(slider.value, 10);
          if (label) label.textContent = val;
          self.canvas.setBrushSize(val);
        });
      }

      // ---- Workspace path auto-resize + Enter key -------------------------
      var pathInput = document.getElementById('workspace-path');
      if (pathInput) {
        pathInput.addEventListener('input', function () {
          var len = pathInput.value.length || 1;
          pathInput.size = Math.max(15, len);
        });
        pathInput.addEventListener('keydown', function (e) {
          if (e.key === 'Enter') {
            e.preventDefault();
            self.changeWorkspace();
          }
        });
      }

      } catch(e) { console.error('_setupUI error:', e); }
    }

    /** Register all built-in tools. */
    _registerTools () {
      this.tools = {
        sam:   new PointTool(this.canvas, this),
        brush: new BrushTool(this.canvas, this),
        pen:   new PenTool(this.canvas, this)
      };
    }

    // ========================================================================
    // Tool Management
    // ========================================================================

    /**
     * Activate a named tool and update the UI.
     * @param {string} name – 'point', 'box', 'brush', or 'pen'
     */
    setTool (name) {
      if (!this.tools[name]) return;

      // Activate the tool (resets its internal state, clears overlays)
      this.tools[name].activate();
      this.currentTool = name;

      // Update toolbar button state
      document.querySelectorAll('.tool-btn').forEach(function (btn) {
        btn.classList.toggle('active', btn.dataset.tool === name);
      });

      // Clear previous tool overlay
      this.canvas.setToolOverlay(null);
    }

    /** Update cursor overlay for current tool at image coordinates. */
    _updateCursor (imgX, imgY) {
      if (this.currentTool === 'brush') {
        this.canvas.setToolOverlay({ type: 'brush-cursor', x: imgX, y: imgY });
      } else if (this.currentTool === 'sam' || this.currentTool === 'pen') {
        this.canvas.setToolOverlay({ type: 'sam-cursor', x: imgX, y: imgY });
      }
    }

    // ========================================================================
    // Keyboard Shortcuts
    // ========================================================================

    /** Register global keyboard shortcuts and forward keys to tools. */
    _setupShortcuts () {
      var self = this;

      document.addEventListener('keydown', function (e) {
        // Never intercept when the user is typing in an input
        if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') {
          return;
        }

        var key = e.key;

        // ---- Ctrl / Meta shortcuts -----------------------------------------
        if (e.ctrlKey || e.metaKey) {
          switch (key.toLowerCase()) {
            case 'z':
              e.preventDefault();
              self.undo();
              return;
            case 's':
              e.preventDefault();
              self.save();
              return;
          }
          // For other Ctrl combos (e.g. Ctrl held alone) fall through so the
          // active tool can observe the Ctrl state.
        }

        // ---- Image navigation (PageUp / PageDown) -------------------------
        if ((key === 'PageUp' || key === 'PageDown') && Date.now() > self._navCooldown) {
          e.preventDefault();
          self._navCooldown = Date.now() + 1000;
          var nextIdx = key === 'PageUp' ? self.currentIndex - 1 : self.currentIndex + 1;
          if (nextIdx >= 0 && nextIdx < self.files.length) self.navigateTo(nextIdx);
          return;
        }

        // ---- Tool switching (case-insensitive) -----------------------------
        switch (key.toLowerCase()) {
          case 's': e.preventDefault(); self.setTool('sam');    return;
          case 'r': e.preventDefault(); self.setTool('brush');  return;
          case 'n': e.preventDefault(); self.setTool('pen');    return;
          case 'd':
            e.preventDefault();
            self.canvas.setMode('draw');
            document.querySelectorAll('.mode-toggle__btn').forEach(function (btn) {
              btn.classList.toggle('active', btn.dataset.mode === 'draw');
            });
            return;
          case 'e':
            e.preventDefault();
            self.canvas.setMode('exclude');
            document.querySelectorAll('.mode-toggle__btn').forEach(function (btn) {
              btn.classList.toggle('active', btn.dataset.mode === 'exclude');
            });
            return;
        }

        // Forward unconsumed keys to the active tool (Escape, Enter, Ctrl…)
        if (self.tools[self.currentTool] && self.tools[self.currentTool].onKeyDown) {
          self.tools[self.currentTool].onKeyDown(e);
        }
      });

      document.addEventListener('keyup', function (e) {
        if (self.tools[self.currentTool] && self.tools[self.currentTool].onKeyUp) {
          self.tools[self.currentTool].onKeyUp(e);
        }
      });
    }

    // ========================================================================
    // Canvas Event Forwarding
    // ========================================================================

    /** Forward mouse / wheel / contextmenu events to the active tool. */
    _setupEventListeners () {
      var self   = this;
      var canvas = this.canvas.canvas;

      canvas.addEventListener('mousedown', function (e) {
        if (self.tools[self.currentTool]) {
          self.tools[self.currentTool].onMouseDown(e);
        }
      });

      canvas.addEventListener('mousemove', function (e) {
        // Update cursor coordinates in the status bar
        var img = self.canvas.screenToImage(e.offsetX, e.offsetY);
        var coordEl = document.getElementById('coord-display');
        if (coordEl) {
          coordEl.textContent = 'x: ' + Math.round(img.x) + ' y: ' + Math.round(img.y);
        }

        // Show tool cursor overlay
        self._updateCursor(img.x, img.y);

        if (self.tools[self.currentTool]) {
          self.tools[self.currentTool].onMouseMove(e);
        }
      });

      canvas.addEventListener('mouseup', function (e) {
        if (self.tools[self.currentTool]) {
          self.tools[self.currentTool].onMouseUp(e);
        }
      });

      // ---- Touch events (mobile) -------------------------------------------
      function touchToMouse(e) {
        e.preventDefault();
        var rect = canvas.getBoundingClientRect();
        var t = e.touches ? e.touches[0] : e.changedTouches[0];
        return { offsetX: t.clientX - rect.left, offsetY: t.clientY - rect.top };
      }

      // Pinch state for brush size
      self._pinchDist = 0;

      canvas.addEventListener('touchstart', function (e) {
        if (e.touches.length === 2) {
          // Two-finger gesture
          if (self.currentTool === 'pen' && self.tools.pen && self.tools.pen.points && self.tools.pen.points.length >= 2) {
            // Remove the last point added by the first finger's touchstart
            self.tools.pen.points.pop();
            self.tools.pen._closePath();
          }
          if (self.currentTool === 'brush') {
            self.undo();  // remove the first finger's accidental stroke
            var dx = e.touches[0].clientX - e.touches[1].clientX;
            var dy = e.touches[0].clientY - e.touches[1].clientY;
            self._pinchDist = Math.sqrt(dx * dx + dy * dy);
          }
          return;
        }
        // Single finger only — two-finger gestures skip onMouseDown
        if (e.touches.length === 1) {
          var me = touchToMouse(e);
          if (self.tools[self.currentTool]) self.tools[self.currentTool].onMouseDown(me);
        }
      }, { passive: false });

      canvas.addEventListener('touchmove', function (e) {
        if (e.touches.length === 2 && self.currentTool === 'brush' && self._pinchDist > 0) {
          e.preventDefault();
          var dx = e.touches[0].clientX - e.touches[1].clientX;
          var dy = e.touches[0].clientY - e.touches[1].clientY;
          var newDist = Math.sqrt(dx * dx + dy * dy);
          var ratio = newDist / self._pinchDist;
          self._pinchDist = newDist;
          if (Math.abs(1 - ratio) > 0.03) {
            var slider = document.getElementById('brush-size');
            if (slider) {
              var cur = parseInt(slider.value, 10);
              var delta = ratio > 1 ? 4 : -4;
              var newVal = Math.max(4, Math.min(100, cur + delta));
              slider.value = newVal;
              self.canvas.setBrushSize(newVal);
              var label = document.getElementById('brush-size-label');
              if (label) label.textContent = newVal;
            }
          }
          return;
        }
        var me = touchToMouse(e);
        var img = self.canvas.screenToImage(me.offsetX, me.offsetY);
        var coordEl = document.getElementById('coord-display');
        if (coordEl) coordEl.textContent = 'x: ' + Math.round(img.x) + ' y: ' + Math.round(img.y);
        self._updateCursor(img.x, img.y);
        if (self.tools[self.currentTool]) self.tools[self.currentTool].onMouseMove(me);
      }, { passive: false });

      canvas.addEventListener('touchend', function (e) {
        self._pinchDist = 0;
        var me = touchToMouse(e);
        if (self.tools[self.currentTool]) self.tools[self.currentTool].onMouseUp(me);
      }, { passive: false });

      // Scroll → adjust brush size (in brush mode) or zoom (other modes)
      canvas.addEventListener('wheel', function (e) {
        if (self.currentTool === 'brush') {
          e.preventDefault();
          var slider = document.getElementById('brush-size');
          if (!slider) return;

          var delta  = e.deltaY > 0 ? -2 : 2;
          var min    = parseInt(slider.min, 10) || 4;
          var max    = parseInt(slider.max, 10) || 100;
          var cur    = parseInt(slider.value, 10);
          var newVal = Math.max(min, Math.min(max, cur + delta));

          slider.value = newVal;
          self.canvas.setBrushSize(newVal);

          var label = document.querySelector('.size-label') ||
                      document.getElementById('brush-size-label');
          if (label) label.textContent = newVal;
        }
      });

      // Show cursor when mouse enters canvas
      canvas.addEventListener('mouseenter', function (e) {
        var img = self.canvas.screenToImage(e.offsetX, e.offsetY);
        self._updateCursor(img.x, img.y);
      });

      // Disable the browser context menu on the canvas
      canvas.addEventListener('contextmenu', function (e) {
        e.preventDefault();
      });
    }

    // ========================================================================
    // Project / Image Navigation
    // ========================================================================

    /**
     * Open a project directory.
     * @param {string} path – Absolute filesystem path.
     */
    async openProject (path) {
      try {
        var data = await API.openProject(path);
        // API returns { project, images }
        this.files = data.files || [];
        this._renderFileList();
        if (this.files.length > 0) {
          // Restore last viewed position
          var saved = 0;
          try {
            var prog = await API.getProgress();
            if (prog && prog.success && prog.index > 0 && prog.index < this.files.length) {
              saved = prog.index;
            }
          } catch (_) {}
          await this.navigateTo(saved);
        }

        var pathEl = document.getElementById('workspace-path');
        if (pathEl) {
          pathEl.value = path;
          pathEl.style.width = Math.max(130, path.length * 7.5 + 16) + 'px';
        }
      } catch (err) {
        console.error('Failed to open project:', err);
        this.setStatus('Error: ' + err.message);
      }
    }

    /**
     * Navigate to a specific image index.
     * Loads the image, resets the mask and points.
     * @param {number} index
     */
    async navigateTo (index) {
      if (index < 0 || index >= this.files.length) return;

      // Auto-save current mask before switching
      if (this.canvas.mask && this.currentIndex !== index && this.currentIndex >= 0) {
        var saveData = this.canvas.mask.data;
        // Flatten: remove negative values
        for (var ci = 0; ci < saveData.length; ci++) {
          if (saveData[ci] < 0) saveData[ci] = 0;
        }
        var curB64 = this._maskToB64();
        if (curB64) {
          try { await API.saveMask(this.currentIndex, curB64); } catch (_) {}
        }
      }

      this.currentIndex = index;

      // Persist progress so it's remembered across sessions
      try { API.setProgress(index); } catch (_) {}

      try {
        this.canvas.setLoading(30, 'Loading image...');
        var data = await API.loadImage(index);
        this.canvas.setLoading(60, 'Decoding...');

        var img  = new Image();
        var self = this;
        var loadTimeout;

        function onImageReady() {
          clearTimeout(loadTimeout);
          self.canvas.setImage(img);
          self.canvas.setPoints([]);
          self.history = [];

          // Load existing mask from DFLJPG (only if user hasn't started drawing)
          API.loadExistingMask(index).then(function (maskResult) {
            if (maskResult && maskResult.mask) {
              self._applyMaskB64(maskResult.mask, function () {
                self.canvas.clearLoading();
                self._updateFileListActive();
                self._updateCounts();
                self.setStatus('');
              });
            } else {
              if (!self.canvas.mask) self.canvas.setMask(null);
              self.canvas.clearLoading();
              self._updateFileListActive();
              self._updateCounts();
              self.setStatus('');
            }
          }).catch(function () {
            if (!self.canvas.mask) self.canvas.setMask(null);
            self.canvas.clearLoading();
            self._updateFileListActive();
            self._updateCounts();
            self.setStatus('');
          });
        }

        img.onload  = onImageReady;
        img.onerror = function () {
          clearTimeout(loadTimeout);
          self.canvas.clearLoading();
          self.setStatus('Failed to decode image');
        };

        // Load image
        img.src = 'data:image/jpeg;base64,' + data.image;

        // Fallback timeout
        loadTimeout = setTimeout(function () {
          if (!self.canvas.image) {
            self.canvas.clearLoading();
            self.setStatus('Image load timed out');
          }
        }, 20000);

      } catch (err) {
        this.canvas.clearLoading();
        console.error('Failed to load image:', err);
        this.setStatus('Error: ' + err.message);
      }
    }

    /**
     * Change workspace directory to the value in the path input.
     */
    changeWorkspace () {
      var input = document.getElementById('workspace-path');
      if (!input) return;
      var path = input.value.trim();
      if (!path) {
        this.setStatus('Please enter a workspace path');
        input.focus();
        return;
      }
      this.openProject(path);
    }

    // ========================================================================
    // Mask Helpers
    // ========================================================================

    /**
     * Decode a base64 PNG mask and set it on the canvas.
     * @param {string} b64 – Base64-encoded PNG (without data: prefix).
     */
    _applyMaskB64 (b64, cb) {
      var img  = new Image();
      var self = this;

      img.onload = function () {
        var res = self.canvas.maskResolution || 256;
        var c   = document.createElement('canvas');
        c.width  = res;
        c.height = res;
        var ctx = c.getContext('2d');
        ctx.drawImage(img, 0, 0, res, res);

        var imageData = ctx.getImageData(0, 0, res, res);
        var pixels    = imageData.data;
        var data      = new Float32Array(res * res);

        for (var i = 0; i < res * res; i++) {
          data[i] = pixels[i * 4] / 255;
        }

        self.canvas._maskFadeStart = Date.now();
        self.canvas.setMask({ width: res, height: res, data: data });
        self._updateCounts();
        if (cb) cb();
      };

      img.src = 'data:image/png;base64,' + b64;
    }

    /**
     * Convert the current canvas mask to a base64 PNG string (no prefix).
     * @returns {string}
     */
    _maskToB64 () {
      if (!this.canvas.mask) return '';

      var mask  = this.canvas.mask;
      var mw    = mask.width;
      var mh    = mask.height;
      var c     = document.createElement('canvas');
      c.width   = mw;
      c.height  = mh;
      var ctx   = c.getContext('2d');
      var imageData = ctx.createImageData(mw, mh);
      var pixels   = imageData.data;

      for (var i = 0; i < mw * mh; i++) {
        var val = Math.round(Math.min(mask.data[i], 1) * 255);
        pixels[i * 4]     = val;
        pixels[i * 4 + 1] = val;
        pixels[i * 4 + 2] = val;
        pixels[i * 4 + 3] = 255;
      }

      ctx.putImageData(imageData, 0, 0);
      return c.toDataURL('image/png').split(',')[1];
    }

    // ========================================================================
    // Undo
    // ========================================================================

    /** Save the current mask state to the undo stack. */
    pushHistory () {
      var maskCopy = null;
      if (this.canvas.mask && this.canvas.mask.data) {
        var data = this.canvas.mask.data;
        var copy = new Float32Array(data.length);
        copy.set(data);
        maskCopy = { width: this.canvas.mask.width, height: this.canvas.mask.height, data: copy };
      }

      this.history.push({
        mask:   maskCopy,
        points: this.canvas.points.map(function (p) {
          return { x: p.x, y: p.y, label: p.label, id: p.id };
        })
      });

      // Keep the undo stack bounded
      if (this.history.length > 50) {
        this.history.shift();
      }
    }

    /** Restore the most recent saved state. */
    undo () {
      if (this.history.length === 0) return;

      var state = this.history.pop();
      this.canvas.setMask(state.mask);
      this.canvas.setPoints([]);
      this._updateCounts();
    }

    // ========================================================================
    // Save
    // ========================================================================

    /** Persist the current mask via the API. */
    async save () {
      if (!this.canvas.mask) return;

      try {
        var b64  = this._maskToB64();
        var data = await API.saveMask(this.currentIndex, b64);
        if (data.success) {
          this.setStatus('Mask saved');
        } else {
          this.setStatus('Save failed');
        }
      } catch (err) {
        console.error('Save error:', err);
        this.setStatus('Save error: ' + err.message);
      }
    }

    // ========================================================================
    // File List Rendering
    // ========================================================================

    /** Populate the file list in the right panel. */
    _renderFileList () {
      var container = document.querySelector('.file-list');
      if (!container) return;

      container.innerHTML = '';

      var self = this;

      this.files.forEach(function (file, idx) {
        var el = document.createElement('div');
        el.className   = 'file-item';
        el.dataset.index = idx;

        var nameSpan = document.createElement('span');
        nameSpan.className = 'file-item__name';
        nameSpan.textContent = file.name || file.filename || 'Image ' + (idx + 1);
        el.appendChild(nameSpan);

        if (file.has_mask) {
          var check = document.createElement('span');
          check.className = 'file-item__check';
          check.textContent = '✓';
          check.style.cssText = 'margin-left:auto;color:#5bd696;font-size:9px;';
          el.appendChild(check);
        }

        el.addEventListener('click', function () {
          self.navigateTo(idx);
        });

        container.appendChild(el);
      });

      // Update file count badge
      var countEl = document.getElementById('file-count');
      if (countEl) countEl.textContent = String(this.files.length);
    }

    /** Toggle the .active class on file items to match the current index. */
    _updateFileListActive () {
      var items = document.querySelectorAll('.file-item');
      var idx   = this.currentIndex;
      for (var i = 0; i < items.length; i++) {
        items[i].classList.toggle('active',
          parseInt(items[i].dataset.index, 10) === idx
        );
      }
    }

    // ========================================================================
    // Status / Counts
    // ========================================================================

    /** Update FG / BG pixel counts in the status bar. */
    _updateCounts () {
      var fgEl = document.getElementById('fg-count');
      var bgEl = document.getElementById('bg-count');

      if (!this.canvas.mask || !this.canvas.mask.data) {
        if (fgEl) fgEl.textContent = 'FG: 0';
        if (bgEl) bgEl.textContent = 'BG: 0';
        return;
      }

      var data = this.canvas.mask.data;
      var fg = 0;
      for (var i = 0; i < data.length; i++) {
        if (data[i] > 0) fg++;
      }

      if (fgEl) fgEl.textContent = 'FG: ' + fg;
      if (bgEl) bgEl.textContent = 'BG: ' + (data.length - fg);
    }

    /**
     * Set a status / hint message in the status bar.
     * Called by tools via `this.app.setStatus(msg)`.
     * @param {string} msg
     */
    setStatus (msg) {
      var el = document.getElementById('status-message');
      if (el) el.textContent = msg;
    }
  }

  // ==========================================================================
  // Export
  // ==========================================================================

  window.App = App;

})();
