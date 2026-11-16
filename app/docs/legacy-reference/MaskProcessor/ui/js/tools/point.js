/**
 * Point Tool — Single-click and Ctrl multi-point group mode
 *
 * Two modes:
 *   1) Single click (no Ctrl): left click = foreground point, predict immediately
 *   2) Ctrl held (multi-point group): Ctrl+Left = FG, Ctrl+Right = BG, release to finalize
 *
 * Depends on: window.API, window.MaskCanvas
 * Exports:    window.PointTool
 */
(function () {
  'use strict';

  // ==========================================================================
  // Shared: decode a base64 PNG mask returned by the API and apply it
  // ==========================================================================

  /**
   * @param {MaskCanvas} canvas
   * @param {string}     b64   – Base64-encoded PNG (without data: prefix)
   */
  function decodeAndSetMask (canvas, b64, clickMode) {
    var img = new Image();
    img.onload = function () {
      var c      = document.createElement('canvas');
      c.width    = img.width;
      c.height   = img.height;
      var ctx    = c.getContext('2d');
      ctx.drawImage(img, 0, 0);
      var imageData = ctx.getImageData(0, 0, img.width, img.height);
      var pixels   = imageData.data;
      var len      = img.width * img.height;

      // Ensure canvas has a mask to merge into
      canvas.ensureMask();

      if (canvas.mask && canvas.mask.data) {
        var existing = canvas.mask.data;
        var mw = canvas.mask.width;
        var mh = canvas.mask.height;
        var scaleX = mw / img.width;
        var scaleY = mh / img.height;

        // Merge new mask into existing — positive for Draw, negative for Exclude
        var isDraw = (clickMode || canvas.mode) === 'draw';
        for (var py = 0; py < img.height; py++) {
          for (var px = 0; px < img.width; px++) {
            var srcIdx = (py * img.width + px) * 4;
            if (pixels[srcIdx] > 128) {
              var x0 = Math.floor(px * scaleX);
              var y0 = Math.floor(py * scaleY);
              var x1 = Math.ceil((px + 1) * scaleX);
              var y1 = Math.ceil((py + 1) * scaleY);
              for (var dy = y0; dy < y1 && dy < mh; dy++) {
                for (var dx = x0; dx < x1 && dx < mw; dx++) {
                  if (isDraw) {
                    existing[dy * mw + dx] = 1.0;
                  } else {
                    existing[dy * mw + dx] = -1.0;
                  }
                }
              }
            }
          }
        }
        canvas.renderNow();
      } else {
        // No existing mask — create new one
        var isDraw = (clickMode || canvas.mode) === 'draw';
        var data = new Float32Array(len);
        for (var i = 0; i < len; i++) {
          data[i] = pixels[i * 4] / 255;
          if (!isDraw) data[i] = -data[i];  // negative for exclude mode
        }
        canvas._maskFadeStart = Date.now();
        canvas.setMask({ width: img.width, height: img.height, data: data });
      }
    };
    img.src = 'data:image/png;base64,' + b64;
  }

  // ==========================================================================
  // PointTool
  // ==========================================================================

  class PointTool {

    /**
     * @param {MaskCanvas} canvas
     * @param {object}     app – { currentIndex, pushHistory, setStatus? }
     */
    constructor (canvas, app) {
      this.canvas      = canvas;
      this.app         = app;
      this.ctrlPressed = false;
      this.group       = null;    // [{ x, y, label }] when Ctrl is held
    }

    // ---- Lifecycle ----------------------------------------------------------

    /** Called when this tool becomes active. */
    activate () {
      this.ctrlPressed = false;
      this.group       = null;
      this.canvas.setToolOverlay(null);
    }

    // ---- Mouse events -------------------------------------------------------

    onMouseDown (e) {
      var img = this.canvas.screenToImage(e.offsetX, e.offsetY);

      if (this.ctrlPressed) {
        // ---- Multi-point group mode ----------------------------------------
        e.preventDefault();

        if (!this.group) this.group = [];

        if (e.button === undefined || e.button === 0) {
          // Ctrl + Left click (or touch) → foreground
          this.group.push({ x: Math.round(img.x), y: Math.round(img.y), label: 1 });
        } else if (e.button === 2) {
          // Ctrl + Right click → background
          this.group.push({ x: Math.round(img.x), y: Math.round(img.y), label: 0 });
        } else {
          return; // Middle button / other — ignore
        }

        this._updateGroupPreview();
        // Prediction happens on Ctrl release, not per click

      } else {
        // ---- Single click mode (no Ctrl) -----------------------------------
        if (e.button === undefined || e.button === 0) {
          // Foreground point → immediate prediction
          var pt = { x: Math.round(img.x), y: Math.round(img.y), label: 1 };
          var clicks = [[pt.x, pt.y, 1]];

          this.canvas.setPoints([
            { x: pt.x, y: pt.y, label: 1, id: 1 }
          ]);

          var self = this;
          var clickMode = this.canvas.mode;
          self.canvas.setLoading(50, 'Generating mask...');
          API.predictMask(this.app.currentIndex, clicks)
            .then(function (result) {
              self.canvas.clearLoading();
              if (result.success && result.mask) {
                self.app.pushHistory();
                decodeAndSetMask(self.canvas, result.mask, clickMode);
              }
              self.canvas.setPoints([]);
            })
            .catch(function (err) {
              self.canvas.clearLoading();
              self.canvas.setPoints([]);
              console.error('SAM predict error:', err);
            });
        }
        // Right click is ignored without Ctrl
      }
    }

    onMouseMove (e) {
      var img = this.canvas.screenToImage(e.offsetX, e.offsetY);
      this.canvas.setToolOverlay({
        type: 'sam-cursor',
        x: img.x,
        y: img.y
      });
    }

    onMouseUp (_e) {
      // No special action on release
    }

    // ---- Keyboard events ----------------------------------------------------

    onKeyDown (e) {
      if (e.key === 'Control' || e.key === 'Meta') {
        if (this.ctrlPressed) return;  // ignore keyboard repeat
        this.ctrlPressed = true;
        this.group       = [];
        this.canvas.setPoints([]);
        this._setStatus('Ctrl held — click to add points');
      }
    }

    onKeyUp (e) {
      if (e.key === 'Control' || e.key === 'Meta') {
        this.ctrlPressed = false;

        if (this.group && this.group.length > 0) {
          // Finalize the multi-point group (pushHistory is inside _predictGroup)
          this._predictGroup();
        }

        this.group = null;
        this._setStatus('');
      }
    }

    // ---- Internals ----------------------------------------------------------

    /** Update the canvas point display and status bar for the current group. */
    _updateGroupPreview () {
      if (!this.group || this.group.length === 0) return;

      var fgCount = 0;
      var bgCount = 0;
      var displayPts = [];

      for (var i = 0; i < this.group.length; i++) {
        var p = this.group[i];
        if (p.label === 1) fgCount++; else bgCount++;
        displayPts.push({ x: p.x, y: p.y, label: p.label, id: i + 1 });
      }

      this.canvas.setPoints(displayPts);
      this._setStatus(
        'Ctrl Group: ' + this.group.length + ' pts (' +
        fgCount + 'FG/' + bgCount + 'BG) — release to finalize'
      );
    }

    /** Call API.predictMask for the current group (preview or final). */
    _predictGroup () {
      if (!this.group || this.group.length === 0) return;

      var clicks = [];
      for (var i = 0; i < this.group.length; i++) {
        clicks.push([this.group[i].x, this.group[i].y, this.group[i].label]);
      }

      var self = this;
      var releaseMode = this.canvas.mode;
      self.canvas.setLoading(50, 'Generating mask...');
      API.predictMask(this.app.currentIndex, clicks)
        .then(function (result) {
          self.canvas.clearLoading();
          self.canvas.setPoints([]);
          if (result.success && result.mask) {
            self.app.pushHistory();
            decodeAndSetMask(self.canvas, result.mask, releaseMode);
          }
        })
        .catch(function (err) {
          self.canvas.clearLoading();
          self.canvas.setPoints([]);
          console.error('SAM group predict error:', err);
        });
    }

    /** Safe status update helper. */
    _setStatus (msg) {
      if (this.app.setStatus) this.app.setStatus(msg);
    }
  }

  // ==========================================================================
  // Export
  // ==========================================================================

  window.PointTool = PointTool;

})();
