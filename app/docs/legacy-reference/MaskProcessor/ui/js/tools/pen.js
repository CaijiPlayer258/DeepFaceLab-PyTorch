/**
 * Pen / Polygon Tool — Define a polygonal region, then fill it on close.
 *
 * - Left click  → add anchor point; within 10 px of first → close path
 * - Right click → close path (if >= 3 points)
 * - Escape      → cancel current path
 * - Enter       → close path (if >= 3 points)
 *
 * On close: app.pushHistory() + canvas.fillPolygon(points)
 *
 * Depends on: window.API, window.MaskCanvas
 * Exports:    window.PenTool
 */
(function () {
  'use strict';

  // ==========================================================================
  // PenTool
  // ==========================================================================

  class PenTool {

    /**
     * @param {MaskCanvas} canvas
     * @param {object}     app – { pushHistory, setStatus? }
     */
    constructor (canvas, app) {
      this.canvas  = canvas;
      this.app     = app;
      this.points  = [];   // [{ x, y }] in image space
      this.closed  = false;
      this._closeThreshold = 10; // px image-space
    }

    // ---- Lifecycle ----------------------------------------------------------

    /** Called when this tool becomes active. */
    activate () {
      this.points = [];
      this.closed = false;
      this.canvas.setToolOverlay(null);
      this._setStatus('');
    }

    // ---- Mouse events -------------------------------------------------------

    onMouseDown (e) {
      var img = this.canvas.screenToImage(e.offsetX, e.offsetY);

      if (e.button === undefined || e.button === 0) {
        // ---- Left click / touch -------------------------------------------
        if (this.points.length === 0) {
          this.points.push({ x: img.x, y: img.y });
          this._updatePreview(null);
          return;
        }

        // Check if cursor is near the first anchor (close gesture)
        var first = this.points[0];
        var d = Math.sqrt(
          (img.x - first.x) * (img.x - first.x) +
          (img.y - first.y) * (img.y - first.y)
        );

        if (d < this._closeThreshold && this.points.length >= 3) {
          this._closePath();
        } else {
          this.points.push({ x: img.x, y: img.y });
          this._updatePreview(null);
        }

      } else if (e.button === 2) {
        // ---- Right click → close if enough points -------------------------
        e.preventDefault();
        if (this.points.length >= 3) {
          this._closePath();
        }
      }
    }

    onMouseMove (e) {
      if (this.closed) return;

      var img = this.canvas.screenToImage(e.offsetX, e.offsetY);
      this._updatePreview(img);
    }

    onMouseUp (_e) {
      // No special action on release
    }

    // ---- Keyboard events ----------------------------------------------------

    onKeyDown (e) {
      if (e.key === 'Escape') {
        // Cancel the current path
        this.points = [];
        this.closed = false;
        this.canvas.setToolOverlay(null);
        this._setStatus('');
      } else if (e.key === 'Enter') {
        // Close the path
        if (this.points.length >= 3) {
          this._closePath();
        }
      }
    }

    onKeyUp (_e) {
      // No special action on release
    }

    // ---- Internals ----------------------------------------------------------

    /**
     * Fill the polygon defined by `this.points`, push undo history, and reset
     * the tool state.
     */
    _closePath () {
      if (this.points.length < 3) return;

      this.closed = true;
      this.canvas.setToolOverlay(null);

      // Create mask if none exists
      this.canvas.ensureMask();

      this.app.pushHistory();
      this.canvas.fillPolygon(this.points);

      // Reset for the next polygon
      this.points = [];
      this.closed = false;
      this._setStatus('');
    }

    /**
     * Update the pen-preview overlay.
     * @param {null|{ x: number, y: number }} mousePos – current mouse pos in
     *        image space, or null to omit the live line to cursor.
     */
    _updatePreview (mousePos) {
      if (this.points.length === 0) return;  // let app.js cursor show

      var overlay = {
        type:          'pen-preview',
        anchorPoints:  this.points
      };

      if (mousePos) {
        overlay.mouseX = mousePos.x;
        overlay.mouseY = mousePos.y;
      }

      this.canvas.setToolOverlay(overlay);

      // ---- Status bar hints ------------------------------------------------
      if (this.points.length >= 1) {
        this._setStatus(this.points.length + ' pts');
      }
    }

    /** Safe status update helper. */
    _setStatus (msg) {
      if (this.app.setStatus) this.app.setStatus(msg);
    }
  }

  // ==========================================================================
  // Export
  // ==========================================================================

  window.PenTool = PenTool;

})();
