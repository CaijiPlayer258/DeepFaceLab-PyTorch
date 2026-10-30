(function () {
  'use strict';

  window.Preview = {
    init: function () {
      this.canvasOriginal = document.getElementById('canvas-original');
      this.canvasDetection = document.getElementById('canvas-detection');
      this.canvasSwapped = document.getElementById('canvas-swapped');
      this.videoOriginal = document.getElementById('video-original');
      this.debugGrid = document.getElementById('debug-grid');
    },

    showDebugGrid: function (urls) {
      this.canvasSwapped.style.display = 'none';
      this.debugGrid.style.display = 'flex';
      this.debugGrid.innerHTML = urls.map(function (u) {
        return '<img src="' + u + '?_=' + Date.now() + '" style="max-height:45vh;flex:1 1 300px;object-fit:contain;border-radius:4px;">';
      }).join('');
    },

    hideDebugGrid: function () {
      this.debugGrid.style.display = 'none';
      this.debugGrid.innerHTML = '';
      this.canvasSwapped.style.display = 'block';
    },

    showPlayback: function () {
      this.canvasOriginal.style.display = 'none';
      this.videoOriginal.style.display = 'block';
    },

    stopPlayback: function () {
      if (this.videoOriginal) {
        this.videoOriginal.pause();
        this.videoOriginal.style.display = 'none';
      }
      this.canvasOriginal.style.display = 'block';
    },

    setVideoSrc: function (url) {
      this.videoOriginal.src = url;
      this.videoOriginal.load();
    },

    getVideoCurrentFrame: function (fps) {
      if (this.videoOriginal && this.videoOriginal.currentTime > 0 && fps > 0) {
        return Math.min(Math.round(this.videoOriginal.currentTime * fps), 999999);
      }
      return -1;
    },

    bufferedAhead: function () {
      var v = this.videoOriginal;
      if (!v || !v.buffered || v.buffered.length === 0) return 0;
      return v.buffered.end(v.buffered.length - 1) - v.currentTime;
    },

    _drawImage: function (canvas, img, faces) {
      var ctx = canvas.getContext('2d');
      var parent = canvas.parentElement;
      var cw = parent.clientWidth;
      var ch = parent.clientHeight;
      canvas.width = cw;
      canvas.height = ch;

      var iw = img.naturalWidth || img.width;
      var ih = img.naturalHeight || img.height;
      var scale = Math.min(cw / iw, ch / ih, 1);
      var dw = iw * scale;
      var dh = ih * scale;
      var dx = (cw - dw) / 2;
      var dy = (ch - dh) / 2;

      ctx.clearRect(0, 0, cw, ch);
      ctx.drawImage(img, dx, dy, dw, dh);

      if (faces && faces.length > 0) {
        var sx = dw / iw;
        var sy = dh / ih;
        faces.forEach(function (face) {
          ctx.strokeStyle = '#5b5bd6';
          ctx.lineWidth = 1.5;
          ctx.strokeRect(dx + face.x * sx, dy + face.y * sy, face.w * sx, face.h * sy);

          if (face.landmarks) {
            ctx.fillStyle = '#00ff96';
            face.landmarks.forEach(function (lm) {
              ctx.beginPath();
              ctx.arc(dx + lm[0] * sx, dy + lm[1] * sy, 1.5, 0, Math.PI * 2);
              ctx.fill();
            });
          }
        });
      }
    },

    updateOriginal: function (imageUrl) {
      var img = new Image();
      var self = this;
      img.onload = function () { self._drawImage(self.canvasOriginal, img, null); };
      img.src = imageUrl;
    },

    updateDetection: function (imageUrl, faces) {
      var img = new Image();
      var self = this;
      img.onload = function () {
        self._drawImage(self.canvasDetection, img, faces || []);
        var countEl = document.getElementById('face-count');
        countEl.textContent = faces && faces.length > 0 ? faces.length + ' faces' : '';
      };
      img.src = imageUrl;
    },

    updateSwapped: function (imageUrl) {
      this._lastSwappedUrl = imageUrl;
      var img = new Image();
      var self = this;
      img.onload = function () { self._drawImage(self.canvasSwapped, img, null); };
      img.src = imageUrl;
    },

    clearAll: function () {
      ['canvas-original', 'canvas-detection', 'canvas-swapped'].forEach(function (id) {
        var c = document.getElementById(id);
        if (c) {
          var ctx = c.getContext('2d');
          if (ctx) ctx.clearRect(0, 0, c.width, c.height);
        }
      });
    },

    _downloadSwapped: function () {
      if (!this._lastSwappedUrl) return;
      var a = document.createElement('a');
      a.href = this._lastSwappedUrl;
      a.download = 'swap_' + Math.round(Date.now() / 1000) + '.jpg';
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
    },
  };

  document.addEventListener('DOMContentLoaded', function () {
    var btn = document.getElementById('btn-download-swapped');
    if (btn) btn.addEventListener('click', function () {
      window.Preview._downloadSwapped();
    });
  });
})();
