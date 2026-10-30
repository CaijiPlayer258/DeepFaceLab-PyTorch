/**
 * Brush Tool — Freehand paint on the mask (draw / exclude).
 *
 * Draw mode:  adds 0.3 to mask values (capped at 1.0)
 * Exclude mode: subtracts 0.3 (floored at 0)
 *
 * Interpolation between pointer samples ensures smooth strokes.
 * step = max(2, brushSize / 4)
 *
 * Depends on: window.API, window.MaskCanvas
 * Exports:    window.BrushTool
 */
(function () {
  'use strict';

  // ==========================================================================
  // BrushTool
  // ==========================================================================

  class BrushTool {

    /**
     * @param {MaskCanvas} canvas
     * @param {object}     app – { pushHistory, setStatus? }
     */
    constructor (canvas, app) {
      this.canvas  = canvas;
      this.app     = app;
      this.drawing = false;
      this.lastX   = 0;
      this.lastY   = 0;
    }

    // ---- Lifecycle ----------------------------------------------------------

    /** Called when this tool becomes active. */
    activate () {
      this.drawing = false;
      // Don't clear overlay here — the brush cursor should show on first move
    }

    // ---- Mouse events -------------------------------------------------------

    onMouseDown (e) {
      if (e.button !== undefined && e.button !== 0) return;
      this.drawing = true;

      var img = this.canvas.screenToImage(e.offsetX, e.offsetY);
      this.lastX = img.x;
      this.lastY = img.y;

      this.canvas.ensureMask();
      this.app.pushHistory();
      this.canvas.applyBrushStroke(img.x, img.y);
      this.canvas.renderNow();
    }

    onMouseMove (e) {
      var img = this.canvas.screenToImage(e.offsetX, e.offsetY);

      // Update cursor overlay
      this.canvas.setToolOverlay({
        type: 'brush-cursor',
        x:    img.x,
        y:    img.y
      });

      if (!this.drawing) return;

      // ---- Interpolated stroke between last and current position -----------
      var step  = Math.max(2, this.canvas.brushSize / 5);
      var dx    = img.x - this.lastX;
      var dy    = img.y - this.lastY;
      var dist  = Math.sqrt(dx * dx + dy * dy);

      if (dist > step) {
        var steps = Math.ceil(dist / step);
        var i, t, ix, iy;

        for (i = 0; i <= steps; i++) {
          t  = i / steps;
          ix = this.lastX + dx * t;
          iy = this.lastY + dy * t;
          this.canvas.applyBrushStroke(ix, iy);
        }
      } else {
        this.canvas.applyBrushStroke(img.x, img.y);
      }

      this.lastX = img.x;
      this.lastY = img.y;

      // Render immediately (bypass RAF throttle for responsive drawing)
      this.canvas.renderNow();
    }

    onMouseUp (_e) {
      this.drawing = false;
    }

    // ---- Keyboard events (none) ---------------------------------------------

    onKeyDown (_e) { /* no-op */ }
    onKeyUp   (_e) { /* no-op */ }
  }

  // ==========================================================================
  // Export
  // ==========================================================================

  window.BrushTool = BrushTool;

})();
