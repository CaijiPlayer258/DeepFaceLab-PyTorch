/**
 * MaskCanvas — 4-Layer HTML5 Canvas Rendering Engine
 *
 * Layer 0: Original Image    – drawn fitted to canvas center
 * Layer 1: Mask Overlay      – Float32Array mask as green (draw) or red (exclude) at 40% opacity
 * Layer 2: Reference Points  – green/red circles with numeric labels
 * Layer 3: Tool Overlay      – pen preview, box selection, brush cursor
 *
 * No ES modules.  Attached globally as window.MaskCanvas.
 */

(function () {
  'use strict';

  // ==========================================================================
  // MaskCanvas
  // ==========================================================================

  class MaskCanvas {

    /**
     * @param {HTMLCanvasElement} canvasEl
     */
    constructor (canvasEl) {
      this.canvas = canvasEl;
      this.ctx    = canvasEl.getContext('2d');

      // ---- State -----------------------------------------------------------
      this.image       = null;   // HTMLImageElement
      this.mask        = null;   // { width, height, data: Float32Array }
      this.points      = [];     // [{ x, y, label, id }]
      this.toolOverlay = null;   // { type, ... }
      this.mode        = 'draw'; // 'draw' | 'exclude'
      this.brushSize   = 20;
      this.loading        = null;   // null | { percent: 0-100, label: 'string' }
      this._opacity       = 0.35;   // mask overlay opacity (0-1)
      this.maskResolution = 1024;   // mask canvas resolution (256/512/1024)

      // ---- Image display geometry ------------------------------------------
      this.scale    = 1;
      this.offsetX  = 0;
      this.offsetY  = 0;
      this.imgW     = 0;
      this.imgH     = 0;

      // ---- Offscreen canvas for mask rendering ----------------------------
      this._maskCanvas = document.createElement('canvas');
      this._maskCtx    = this._maskCanvas.getContext('2d');

      // ---- Throttle / Animation --------------------------------------------
      this._rafPending = false;
      this._loadingRaf = null;
      this._shimmerAngle = 0;
      this._maskFadeStart = 0;
      this._flash = null;
      this._excludeFade = null;  // { backup: Float32Array, start: ms }

      // ---- Resize ----------------------------------------------------------
      this._onResizeBound = this._onResize.bind(this);

      if (window.ResizeObserver) {
        this._resizeObserver = new ResizeObserver(this._onResizeBound);
        this._resizeObserver.observe(canvasEl.parentElement);
      } else {
        window.addEventListener('resize', this._onResizeBound);
      }

      // Initial fit
      this._updateCanvasSize();
      this._calculateFit();
    }

    // ========================================================================
    // Coordinate Conversion
    // ========================================================================

    /**
     * Convert canvas-space coordinates to image-space coordinates.
     * @param {number} sx – Canvas X (e.g. e.offsetX)
     * @param {number} sy – Canvas Y (e.g. e.offsetY)
     * @returns {{ x: number, y: number }}
     */
    screenToImage (sx, sy) {
      return {
        x: (sx - this.offsetX) / this.scale,
        y: (sy - this.offsetY) / this.scale
      };
    }

    /**
     * Convert image-space coordinates to canvas-space coordinates.
     * @param {number} ix – Image X
     * @param {number} iy – Image Y
     * @returns {{ x: number, y: number }}
     */
    imageToScreen (ix, iy) {
      return {
        x: ix * this.scale + this.offsetX,
        y: iy * this.scale + this.offsetY
      };
    }

    // ========================================================================
    // State Setters (auto-re-render)
    // ========================================================================

    /** @param {HTMLImageElement} img */
    setImage (img) {
      this.image = img;
      this._calculateFit();
      this.render();
    }

    /**
     * @param {{ width: number, height: number, data: Float32Array }} maskData
     */
    setMask (maskData) {
      this.mask = maskData;
      this.renderNow();
      if (this._maskFadeStart) this._startAnimLoop();
    }

    /** @param {Array<{ x: number, y: number, label: number, id?: number }>} pts */
    setPoints (pts) {
      this.points = Array.isArray(pts) ? pts : [];
      this.renderNow();
    }

    /** @param {'draw'|'exclude'} m */
    setMode (m) {
      this.mode = m;
      this.render();
    }

    /** Create an empty mask at maskResolution x maskResolution if none exists. */
    ensureMask () {
      if (this.mask && this.mask.data) return;
      var res = this.maskResolution || 256;
      this.mask = { width: res, height: res, data: new Float32Array(res * res) };
    }

    /** @param {number} s */
    setBrushSize (s) {
      this.brushSize = s;
      this.render();
    }

    /** Set mask overlay opacity (0-1). */
    setOpacity (val) {
      this._opacity = Math.max(0.05, Math.min(1, val));
      this.render();
    }

    /**
     * @param {null|{ type: string, ... }} overlay
     */
    setToolOverlay (overlay) {
      this.toolOverlay = overlay;
      this.render();
    }

    /**
     * Set overlay data without triggering render (for batch operations).
     * Caller must call render() manually after all strokes.
     */
    _setOverlayData (overlay) {
      this.toolOverlay = overlay;
    }

    /** Shared animation loop — keeps rendering while loading is active. */
    _startAnimLoop () {
      if (this._loadingRaf) return;
      this._shimmerAngle = 0;
      var lastT = Date.now();
      var self = this;
      function animate() {
        if (!self.loading && !self._flash && !self._maskFadeStart && !self._excludeFade) {
          self._loadingRaf = null; return;
        }
        var now = Date.now();
        self._shimmerAngle += (now - lastT) / 800 * Math.PI * 2;
        lastT = now;
        self._doRender();
        self._loadingRaf = requestAnimationFrame(animate);
      }
      self._loadingRaf = requestAnimationFrame(animate);
    }

    /** Show loading overlay with fade-in animation. */
    setLoading (percent, label) {
      this.loading = {
        label: label || '',
        fadeStart: Date.now(),
        fadeDuration: 200,
      };
      this.render();
      this._startAnimLoop();
    }

    /** Start fade-out — overlay gradually disappears revealing the result underneath. */
    clearLoading () {
      if (!this.loading) return;
      if (!this.loading.fadeOutStart) {
        this.loading.fadeOutStart = Date.now();
        this.loading.fadeDuration = 200;
      }
    }

    /** Brief dark pulse over image area (e.g. after commit). */
    flash () {
      this._flash = { start: Date.now(), duration: 400 };
      this._startAnimLoop();
    }

    /** Start fade-out of exclude (red) areas over 300ms. */
    fadeExclude () {
      if (!this.mask || !this.mask.data) return;
      var backup = new Float32Array(this.mask.data.length);
      backup.set(this.mask.data);
      this._excludeFade = { backup: backup, start: Date.now(), duration: 300 };
      this._startAnimLoop();
    }

    // ========================================================================
    // Resize
    // ========================================================================

    /** @private */
    _onResize () {
      this._updateCanvasSize();
      this._calculateFit();
      this.render();
    }

    /** @private */
    _updateCanvasSize () {
      var parent = this.canvas.parentElement;
      if (!parent) return;
      var rect = parent.getBoundingClientRect();
      this.canvas.width  = rect.width;
      this.canvas.height = rect.height;
    }

    /** @private */
    _calculateFit () {
      if (!this.image) return;

      var cw = this.canvas.width;
      var ch = this.canvas.height;

      // Use whichever dimension is available
      this.imgW = this.image.naturalWidth || this.image.width  || 0;
      this.imgH = this.image.naturalHeight || this.image.height || 0;

      if (this.imgW === 0 || this.imgH === 0) return;

      // Scale to fit inside canvas using 95 % of the smaller dimension
      var scaleX = (cw * 0.95) / this.imgW;
      var scaleY = (ch * 0.95) / this.imgH;
      this.scale = Math.min(scaleX, scaleY);

      // Centre the image
      this.offsetX = (cw - this.imgW * this.scale) / 2;
      this.offsetY = (ch - this.imgH * this.scale) / 2;
    }


    // ========================================================================
    // Main Render — draws all four layers
    // ========================================================================

    /** Throttled render using requestAnimationFrame. */
    render () {
      if (this._rafPending) return;
      this._rafPending = true;
      var self = this;
      requestAnimationFrame(function () {
        self._rafPending = false;
        self._doRender();
      });
    }

    /** Immediate render — bypasses RAF throttle. Use for critical updates. */
    renderNow () {
      this._rafPending = false;
      this._doRender();
    }

    /** Actual render — called once per animation frame. */
    _doRender () {
      var ctx = this.ctx;
      var cw  = this.canvas.width;
      var ch  = this.canvas.height;

      // Clear canvas
      ctx.clearRect(0, 0, cw, ch);

      if (!this.image) return;

      // ---- Layer 0 : Original Image ---------------------------------------
      ctx.drawImage(
        this.image,
        this.offsetX, this.offsetY,
        this.imgW * this.scale, this.imgH * this.scale
      );

      // ---- Layer 1 : Mask Overlay -----------------------------------------
      if (this.mask) {
        this._renderMask(ctx);
      }

      // ---- Layer 2 : Reference Points -------------------------------------
      this._renderPoints(ctx);

      // ---- Layer 3 : Tool Overlay -----------------------------------------
      this._renderToolOverlay(ctx);

      // ---- Layer 4 : Loading Overlay --------------------------------------
      this._renderLoading(ctx);

      // ---- Flash overlay (image area only, like a quick loading pulse) ----
      if (this._flash) {
        var elapsed = Date.now() - this._flash.start;
        var t = Math.min(1, elapsed / this._flash.duration);
        var alpha = 0.35 * (1 - t + 0.05);
        ctx.fillStyle = 'rgba(0, 0, 0, ' + alpha.toFixed(3) + ')';
        ctx.fillRect(this.offsetX, this.offsetY, this.imgW * this.scale, this.imgH * this.scale);
        if (t >= 1) this._flash = null;
      }
    }

    // ========================================================================
    // Layer 1 — Mask Overlay
    // ========================================================================

    /** @private */
    _renderMask (ctx) {
      if (!this.mask || !this.mask.data) return;

      var mw = this.mask.width;
      var mh = this.mask.height;
      var data = this.mask.data;
      var len = mw * mh;

      if (this._maskCanvas.width !== mw || this._maskCanvas.height !== mh) {
        this._maskCanvas.width = mw;
        this._maskCanvas.height = mh;
      }

      // Mask fade-in (for AI-generated masks)
      var fadeAlpha = 1;
      if (this._maskFadeStart > 0) {
        var elapsed = Date.now() - this._maskFadeStart;
        var t = Math.min(1, elapsed / 200);
        fadeAlpha = t;
        if (t >= 1) this._maskFadeStart = 0;
      }

      var imageData = this._maskCtx.createImageData(mw, mh);
      var px = imageData.data;
      var a  = Math.round(this._opacity * 255 * fadeAlpha);

      // Exclude fade (commit animation): gradually reduce negative values
      var excludeFadeMul = 1;
      if (this._excludeFade) {
        var elapsed = Date.now() - this._excludeFade.start;
        var t = Math.min(1, elapsed / this._excludeFade.duration);
        excludeFadeMul = 1 - t;
        if (t >= 1) { this._excludeFade = null; excludeFadeMul = 0; }
      }

      for (var i = 0; i < len; i++) {
        var v = data[i];
        if (v > 0) {
          var idx = i * 4;
          px[idx] = 0; px[idx + 1] = 255; px[idx + 2] = 0; px[idx + 3] = a;
        } else if (v < 0) {
          var idx = i * 4;
          var na = Math.round(a * excludeFadeMul);
          if (na > 0) {
            px[idx] = 255; px[idx + 1] = 0; px[idx + 2] = 0; px[idx + 3] = na;
          }
        }
      }

      this._maskCtx.putImageData(imageData, 0, 0);

      ctx.drawImage(
        this._maskCanvas,
        this.offsetX, this.offsetY,
        this.imgW * this.scale, this.imgH * this.scale
      );
    }

    // ========================================================================
    // Layer 2 — Reference Points
    // ========================================================================

    /** @private */
    _renderPoints (ctx) {
      if (!this.points || this.points.length === 0) return;

      var pts   = this.points;
      var len   = pts.length;
      var pt, sx, sy, isFG, color, labelStr, i;

      for (i = 0; i < len; i++) {
        pt = pts[i];

        sx = pt.x * this.scale + this.offsetX;
        sy = pt.y * this.scale + this.offsetY;

        isFG  = pt.label === 1;
        color = isFG ? '#22c55e' : '#ef4444';

        // Circle
        ctx.beginPath();
        ctx.arc(sx, sy, 6, 0, Math.PI * 2);
        ctx.fillStyle = color;
        ctx.fill();
        ctx.strokeStyle = 'rgba(255,255,255,0.8)';
        ctx.lineWidth   = 1.5;
        ctx.stroke();

        // Number label (1-based index unless an explicit id is provided)
        labelStr = String(pt.id !== undefined ? pt.id : (i + 1));

        ctx.fillStyle   = 'rgba(255,255,255,0.9)';
        ctx.font        = 'bold 11px Inter, -apple-system, sans-serif';
        ctx.textAlign   = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText(labelStr, sx, sy);
      }
    }

    // ========================================================================
    // Layer 3 — Tool Overlay
    // ========================================================================

    /** @private */
    _renderToolOverlay (ctx) {
      if (!this.toolOverlay) return;

      switch (this.toolOverlay.type) {
        case 'pen-preview':
          this._renderPenPreview(ctx);
          break;
        case 'box-select':
          this._renderBoxSelect(ctx);
          break;
        case 'brush-cursor':
          this._renderBrushCursor(ctx);
          break;
        case 'sam-cursor':
          this._renderSamCursor(ctx);
          break;
        default:
          break;
      }
    }

    /** @private */
    _renderPenPreview (ctx) {
      var overlay = this.toolOverlay;
      var anchors = overlay.anchorPoints;
      if (!anchors || anchors.length === 0) return;

      var sx, sy, i;

      ctx.save();
      ctx.strokeStyle = 'rgba(91, 91, 214, 0.7)';
      ctx.lineWidth   = 2;
      ctx.setLineDash([5, 4]);

      // Dashed line connecting anchor points
      ctx.beginPath();
      ctx.moveTo(
        anchors[0].x * this.scale + this.offsetX,
        anchors[0].y * this.scale + this.offsetY
      );

      for (i = 1; i < anchors.length; i++) {
        ctx.lineTo(
          anchors[i].x * this.scale + this.offsetX,
          anchors[i].y * this.scale + this.offsetY
        );
      }

      // Live line to mouse cursor
      if (overlay.mouseX !== undefined && overlay.mouseY !== undefined) {
        ctx.lineTo(
          overlay.mouseX * this.scale + this.offsetX,
          overlay.mouseY * this.scale + this.offsetY
        );
      }

      ctx.stroke();
      ctx.setLineDash([]);

      // Draw anchor point dots
      for (i = 0; i < anchors.length; i++) {
        sx = anchors[i].x * this.scale + this.offsetX;
        sy = anchors[i].y * this.scale + this.offsetY;

        ctx.beginPath();
        ctx.arc(sx, sy, 4, 0, Math.PI * 2);
        ctx.fillStyle = 'rgba(91, 91, 214, 0.9)';
        ctx.fill();
      }

      ctx.restore();
    }

    /** @private */
    _renderBoxSelect (ctx) {
      var overlay = this.toolOverlay;
      if (overlay.startX === undefined || overlay.startY === undefined) return;

      var sx = overlay.startX * this.scale + this.offsetX;
      var sy = overlay.startY * this.scale + this.offsetY;

      var endX = overlay.endX !== undefined ? overlay.endX : overlay.startX;
      var endY = overlay.endY !== undefined ? overlay.endY : overlay.startY;

      var ex = endX * this.scale + this.offsetX;
      var ey = endY * this.scale + this.offsetY;

      var x = Math.min(sx, ex);
      var y = Math.min(sy, ey);
      var w = Math.abs(ex - sx);
      var h = Math.abs(ey - sy);

      ctx.save();
      ctx.strokeStyle = 'rgba(91, 91, 214, 0.8)';
      ctx.lineWidth   = 2;
      ctx.setLineDash([4, 3]);
      ctx.strokeRect(x, y, w, h);
      ctx.setLineDash([]);

      ctx.fillStyle = 'rgba(91, 91, 214, 0.08)';
      ctx.fillRect(x, y, w, h);
      ctx.restore();
    }

    /** @private */
    _renderBrushCursor (ctx) {
      var overlay = this.toolOverlay;
      if (overlay.x === undefined || overlay.y === undefined) return;

      var sx     = overlay.x * this.scale + this.offsetX;
      var sy     = overlay.y * this.scale + this.offsetY;
      var radius = (this.brushSize / 2) * this.scale;

      ctx.save();
      ctx.beginPath();
      ctx.arc(sx, sy, radius, 0, Math.PI * 2);
      ctx.strokeStyle = 'rgba(255, 255, 255, 0.5)';
      ctx.lineWidth   = 1.5;
      ctx.stroke();

      ctx.beginPath();
      ctx.arc(sx, sy, 2, 0, Math.PI * 2);
      ctx.fillStyle = 'rgba(255, 255, 255, 0.7)';
      ctx.fill();
      ctx.restore();
    }

    /** @private */
    _renderSamCursor (ctx) {
      var overlay = this.toolOverlay;
      if (overlay.x === undefined || overlay.y === undefined) return;

      var sx = overlay.x * this.scale + this.offsetX;
      var sy = overlay.y * this.scale + this.offsetY;

      ctx.save();
      // Outer ring
      ctx.beginPath();
      ctx.arc(sx, sy, 8, 0, Math.PI * 2);
      ctx.strokeStyle = 'rgba(255, 255, 255, 0.35)';
      ctx.lineWidth = 1;
      ctx.stroke();

      // Crosshair lines
      ctx.beginPath();
      ctx.moveTo(sx - 12, sy); ctx.lineTo(sx + 12, sy);
      ctx.moveTo(sx, sy - 12); ctx.lineTo(sx, sy + 12);
      ctx.strokeStyle = 'rgba(255, 255, 255, 0.5)';
      ctx.lineWidth = 1;
      ctx.stroke();

      // Center dot
      ctx.beginPath();
      ctx.arc(sx, sy, 2, 0, Math.PI * 2);
      ctx.fillStyle = 'rgba(255, 255, 255, 0.8)';
      ctx.fill();
      ctx.restore();
    }

    /** @private */
    _renderLoading (ctx) {
      if (!this.loading) return;

      var cw = this.canvas.width;
      var ch = this.canvas.height;
      var now = Date.now();

      // Fade-in: 0 → 0.55, Fade-out: 0.55 → 0
      var maxAlpha = 0.55;
      var alpha;
      if (this.loading.fadeOutStart) {
        var elapsed = now - this.loading.fadeOutStart;
        var t = Math.min(1, elapsed / this.loading.fadeDuration);
        alpha = maxAlpha * (1 - t);
        if (t >= 1) { this.loading = null; return; }
      } else {
        var elapsed = now - this.loading.fadeStart;
        var t = Math.min(1, elapsed / this.loading.fadeDuration);
        alpha = maxAlpha * t;
      }

      if (alpha <= 0) return;

      // Dark overlay
      ctx.fillStyle = 'rgba(0, 0, 0, ' + alpha.toFixed(3) + ')';
      ctx.fillRect(0, 0, cw, ch);

      // Only show shimmer during fade-in, not during fade-out
      if (!this.loading.fadeOutStart) {
        var cx = cw / 2;
        var cy = ch / 2;
        var radius = 28;

        // Track circle (faint)
        ctx.beginPath();
        ctx.arc(cx, cy, radius, 0, Math.PI * 2);
        ctx.strokeStyle = 'rgba(255, 255, 255, 0.06)';
        ctx.lineWidth = 2;
        ctx.stroke();

        // Flowing shimmer arc
        var angle = this._shimmerAngle || 0;
        var arcLen = Math.PI * 1.2;

        // Trailing glow
        ctx.beginPath();
        ctx.arc(cx, cy, radius, angle - arcLen + 0.3, angle + 0.3);
        ctx.strokeStyle = 'rgba(139, 139, 230, 0.2)';
        ctx.lineWidth = 4;
        ctx.lineCap = 'round';
        ctx.stroke();

        // Main bright arc
        ctx.save();
        ctx.shadowColor = '#8b8be6';
        ctx.shadowBlur = 16;
        ctx.beginPath();
        ctx.arc(cx, cy, radius, angle - arcLen / 2, angle + arcLen / 2);
        ctx.strokeStyle = 'rgba(180, 180, 255, 0.9)';
        ctx.lineWidth = 3;
        ctx.lineCap = 'round';
        ctx.stroke();
        ctx.restore();

        // Label
        if (this.loading.label) {
          ctx.fillStyle = 'rgba(255, 255, 255, 0.5)';
          ctx.font = '12px Inter, sans-serif';
          ctx.textAlign = 'center';
          ctx.fillText(this.loading.label, cx, cy + radius + 22);
        }
      }
    }

    // ========================================================================
    // Tool Operations
    // ========================================================================

    /**
     * Apply a brush stroke at the given image-space position.
     * In draw mode, adds 0.3 to mask values (capped at 1.0).
     * In exclude mode, subtracts 0.3 (floored at 0).
     *
     * @param {number} x – Image X
     * @param {number} y – Image Y
     */
    applyBrushStroke (imgX, imgY) {
      this.ensureMask();
      if (!this.mask || !this.mask.data) return;
      if (!this.imgW || !this.imgH) return;

      var m = this.mask;
      // Scale from image coords to mask coords (mask is NxN, image is any size)
      var sx = m.width / this.imgW;
      var sy = m.height / this.imgH;
      var mx = imgX * sx;
      var my = imgY * sy;
      var r = Math.max(1, (this.brushSize / 2) * sx);
      var r2 = r * r;
      var isDraw = this.mode === 'draw';

      var minX = Math.max(0, Math.floor(mx - r));
      var maxX = Math.min(m.width - 1, Math.ceil(mx + r));
      var minY = Math.max(0, Math.floor(my - r));
      var maxY = Math.min(m.height - 1, Math.ceil(my + r));

      for (var py = minY; py <= maxY; py++) {
        for (var px = minX; px <= maxX; px++) {
          if ((px - mx) * (px - mx) + (py - my) * (py - my) <= r2) {
            if (isDraw) {
              m.data[py * m.width + px] = Math.min(1, m.data[py * m.width + px] + 0.5);
            } else {
              m.data[py * m.width + px] = Math.max(-1, m.data[py * m.width + px] - 0.5);
            }
          }
        }
      }
    }

    /**
     * Fill a polygon defined by the given array of points using a scanline
     * algorithm.  The same draw/exclude value logic as the brush stroke is
     * applied (0.3 increment / decrement).
     *
     * @param {Array<{ x: number, y: number }>} pts – Polygon vertices in image space.
     */
    fillPolygon (pts) {
      if (!this.mask || !this.mask.data) return;
      if (!pts || pts.length < 3) return;

      var mw     = this.mask.width;
      var mh     = this.mask.height;
      var data   = this.mask.data;
      var isDraw = this.mode === 'draw';
      if (!this.imgW || !this.imgH) return;

      // Scale image coords to mask coords
      var sx = mw / this.imgW;
      var sy = mh / this.imgH;
      var scaled = [];
      for (var si = 0; si < pts.length; si++) {
        scaled.push({ x: pts[si].x * sx, y: pts[si].y * sy });
      }
      pts = scaled;

      // Bounding box in mask space
      var minY = Infinity;
      var maxY = -Infinity;
      var i, p;

      for (i = 0; i < pts.length; i++) {
        p = pts[i];
        if (p.y < minY) minY = p.y;
        if (p.y > maxY) maxY = p.y;
      }

      minY = Math.max(0, Math.floor(minY));
      maxY = Math.min(mh - 1, Math.ceil(maxY));

      if (minY > maxY) return;

      // Build edge table
      var edges = [];
      var p1, p2, y1, y2, e;

      for (i = 0; i < pts.length; i++) {
        p1 = pts[i];
        p2 = pts[(i + 1) % pts.length];

        y1 = Math.floor(p1.y);
        y2 = Math.floor(p2.y);

        if (y1 === y2) continue; // Skip horizontal edges

        if (y1 < y2) {
          e = {
            yMin:      y1,
            yMax:      y2,
            xAtYMin:   p1.x,
            invSlope:  (p2.x - p1.x) / (p2.y - p1.y)
          };
        } else {
          e = {
            yMin:      y2,
            yMax:      y1,
            xAtYMin:   p2.x,
            invSlope:  (p1.x - p2.x) / (p1.y - p2.y)
          };
        }

        edges.push(e);
      }

      if (edges.length === 0) return;

      // Scanline fill
      var intersections, j, xStart, xEnd;
      var y, xIntersect, px, idx;

      for (y = minY; y <= maxY; y++) {
        intersections = [];

        for (j = 0; j < edges.length; j++) {
          e = edges[j];
          if (y >= e.yMin && y < e.yMax) {
            xIntersect = e.xAtYMin + e.invSlope * (y - e.yMin);
            intersections.push(xIntersect);
          }
        }

        if (intersections.length < 2) continue;

        // Sort intersections left to right
        intersections.sort(function (a, b) { return a - b; });

        // Fill between each pair
        for (j = 0; j + 1 < intersections.length; j += 2) {
          xStart = Math.max(0, Math.floor(intersections[j]));
          xEnd   = Math.min(mw - 1, Math.ceil(intersections[j + 1]));

          for (px = xStart; px <= xEnd; px++) {
            idx = y * mw + px;
            data[idx] = isDraw ? 0.8 : -0.8;
          }
        }
      }

      this.renderNow();
    }

    /**
     * Clean up listeners and resources.  Call when the canvas is removed from
     * the DOM.
     */
    destroy () {
      if (this._resizeObserver) {
        this._resizeObserver.disconnect();
        this._resizeObserver = null;
      } else {
        window.removeEventListener('resize', this._onResizeBound);
      }

      this.canvas = null;
      this.ctx    = null;
      this.image  = null;
      this.mask   = null;
      this.points = null;
    }
  }

  // ==========================================================================
  // Export
  // ==========================================================================

  window.MaskCanvas = MaskCanvas;

})();
