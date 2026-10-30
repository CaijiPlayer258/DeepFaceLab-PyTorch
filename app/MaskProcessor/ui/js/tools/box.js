/**
 * Box Tool — Draw a bounding box to prompt SAM-based segmentation.
 *
 * onMouseDown → start box
 * onMouseMove → update box-select overlay
 * onMouseUp   → finalise, call API.predictWithBox (minimum 5x5 px)
 *
 * Depends on: window.API, window.MaskCanvas
 * Exports:    window.BoxTool
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
  function decodeAndSetMask (canvas, b64) {
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
      var data     = new Float32Array(len);

      for (var i = 0; i < len; i++) {
        data[i] = pixels[i * 4] / 255;
      }

      canvas.setMask({ width: img.width, height: img.height, data: data });
    };
    img.src = 'data:image/png;base64,' + b64;
  }

  // ==========================================================================
  // BoxTool
  // ==========================================================================

  class BoxTool {

    /**
     * @param {MaskCanvas} canvas
     * @param {object}     app – { currentIndex, pushHistory, setStatus? }
     */
    constructor (canvas, app) {
      this.canvas   = canvas;
      this.app      = app;
      this.startX   = 0;
      this.startY   = 0;
      this.drawing  = false;
    }

    // ---- Lifecycle ----------------------------------------------------------

    /** Called when this tool becomes active. */
    activate () {
      this.drawing = false;
      this.canvas.setToolOverlay(null);
    }

    // ---- Mouse events -------------------------------------------------------

    onMouseDown (e) {
      if (e.button !== 0) return; // Left button only

      var img = this.canvas.screenToImage(e.offsetX, e.offsetY);
      this.startX  = img.x;
      this.startY  = img.y;
      this.drawing = true;
    }

    onMouseMove (e) {
      if (!this.drawing) {
        // Could show a crosshair cursor in the future
        return;
      }

      var img = this.canvas.screenToImage(e.offsetX, e.offsetY);

      this.canvas.setToolOverlay({
        type:   'box-select',
        startX: this.startX,
        startY: this.startY,
        endX:   img.x,
        endY:   img.y
      });
    }

    onMouseUp (e) {
      if (!this.drawing) return;
      this.drawing = false;
      this.canvas.setToolOverlay(null);

      // Determine box corners (normalise order)
      var img   = this.canvas.screenToImage(e.offsetX, e.offsetY);
      var x1    = Math.min(this.startX, img.x);
      var y1    = Math.min(this.startY, img.y);
      var x2    = Math.max(this.startX, img.x);
      var y2    = Math.max(this.startY, img.y);
      var w     = x2 - x1;
      var h     = y2 - y1;

      // Minimum box size: 5 x 5 pixels (too small = accidental click)
      if (w < 5 || h < 5) return;

      this.app.pushHistory();

      var self = this;
      API.predictWithBox(this.app.currentIndex, { x1: x1, y1: y1, x2: x2, y2: y2 })
        .then(function (result) {
          if (result.success && result.mask) {
            decodeAndSetMask(self.canvas, result.mask);
          }
        });
    }

    // ---- Keyboard events (none) ---------------------------------------------

    onKeyDown (_e) { /* no-op */ }
    onKeyUp   (_e) { /* no-op */ }
  }

  // ==========================================================================
  // Export
  // ==========================================================================

  window.BoxTool = BoxTool;

})();
