(function () {
  'use strict';

  window.ExportFlow = {
    state: { currentStep: 0, jobId: 'default' },
    _lastSpeed: { done: 0, time: 0 },

    init: function () {
      this._createLayers();
      this._bindEvents();
      this._populateFaceDB();
      this._populateSettings();
      // Set initial state: main page visible, all overlays hidden
      Object.values(this.layers).forEach(function (el) {
        if (el) el.classList.remove('step-0','step-1','step-2','step-3');
      });
      this.layers.main.classList.add('step-0');
    },

    _createLayers: function () {
      this.layers = {
        main: document.getElementById('layer-main'),
        facedb: document.getElementById('layer-facedb'),
        settings: document.getElementById('layer-settings'),
        progress: document.getElementById('layer-progress'),
      };
      this.nextBar = document.getElementById('fixed-next');
      this.exportBar = document.getElementById('fixed-export');
    },

    _bindEvents: function () {
      var self = this;
      var btn = document.getElementById('btn-export');
      if (btn) btn.addEventListener('click', function () { self.advance(1); });
      var next = document.getElementById('btn-next');
      if (next) next.addEventListener('click', function () { self.advance(2); });
      var start = document.getElementById('btn-start-export');
      if (start) start.addEventListener('click', function () { self.startExport(); });
      var cancel = document.getElementById('btn-cancel-export');
      if (cancel) cancel.addEventListener('click', function () {
        if (window.API && window.API.cancelExport) {
          window.API.cancelExport(self.state.jobId);
        }
        self.close();
      });
      var back1 = document.getElementById('btn-back-facedb');
      if (back1) back1.addEventListener('click', function () { self.advance(0); });
      var back2 = document.getElementById('btn-back-settings');
      if (back2) back2.addEventListener('click', function () { self.advance(1); });
      var computeBtn = document.getElementById('btn-compute-embeddings');
      if (computeBtn) computeBtn.addEventListener('click', function () { self.computeEmbeddings(); });
    },

    advance: function (targetStep) {
      var layers = this.layers;
      Object.keys(layers).forEach(function (key) {
        var el = layers[key];
        if (el) el.classList.remove('step-0', 'step-1', 'step-2', 'step-3');
      });
      if (targetStep >= 0) {
        layers.main.classList.add('step-' + targetStep);
      }
      if (targetStep >= 1) {
        layers.facedb.classList.add('step-' + targetStep);
        this._populateFaceDB();
      }
      if (targetStep >= 2) {
        layers.settings.classList.add('step-' + targetStep);
      }
      if (targetStep >= 3) {
        layers.progress.classList.add('step-' + targetStep);
      }
      if (this.nextBar) this.nextBar.classList.toggle('show', targetStep === 1);
      if (this.exportBar) this.exportBar.classList.toggle('show', targetStep === 2);
      this.state.currentStep = targetStep;
    },

    close: function () {
      this.advance(0);
    },

    _showToast: function (msg) {
      var el = document.createElement('div');
      el.textContent = msg;
      el.style.cssText = 'position:fixed;bottom:30px;left:50%;transform:translateX(-50%);background:#1a1a1e;border:1px solid rgba(91,91,214,0.3);color:#e0e0e0;padding:10px 24px;border-radius:8px;font:13px Inter,sans-serif;z-index:999;box-shadow:0 4px 24px rgba(0,0,0,0.5);opacity:0;transition:opacity 0.3s;';
      document.body.appendChild(el);
      requestAnimationFrame(function () { el.style.opacity = '1'; });
      setTimeout(function () { el.style.opacity = '0'; setTimeout(function () { el.remove(); }, 300); }, 3000);
    },

    _populateFaceDB: function () {
      var container = document.getElementById('facedb-list');
      if (!container) return;
      var app = window.App;
      if (!app) return;
      var db = app.state.faceDatabase || {};
      var keys = Object.keys(db);
      if (keys.length === 0) {
        container.innerHTML = '<div style="padding:20px;color:#555;text-align:center;">在帧中勾选人脸以建立数据库</div>';
        return;
      }
      // Group by model name using faceModelMap
      var byModel = {};
      var faceModelMap = app.state.faceModelMap || {};
      keys.forEach(function (key) {
        var parts = key.split('_');
        var faceIdx = parseInt(parts[1] || 0);
        var fd = db[key];
        var fdObj = typeof fd === 'object' ? fd : {};
        // Try full key first, fall back to integer key (backward compat)
        var modelName = faceModelMap[key] || faceModelMap[faceIdx];
        if (!modelName) {
          // Auto-assign if only one model loaded
          var singleModel = Object.keys(app.state.selectedModels || {});
          modelName = (singleModel.length === 1) ? singleModel[0] : '未分配';
        }
        if (!byModel[modelName]) byModel[modelName] = [];
        byModel[modelName].push({
          key: key,
          thumbUrl: fdObj.thumb_url || '/api/preview/face-thumb/' + parts[0] + '/' + (parts[1] || 0),
          label: fdObj.label || 'Face ' + faceIdx,
          source: 'Frame ' + parts[0],
        });
      });
      var self = this;
      var html = '';
      var modelNames = Object.keys(byModel);
      modelNames.forEach(function (mn) {
        html += '<div class="facedb-group-heading" data-model="' + mn + '" style="font-size:10px;color:#888;text-transform:uppercase;letter-spacing:0.5px;margin:12px 0 6px;padding:4px 8px;border-radius:4px;transition:background 0.15s;">' + mn + ' · ' + byModel[mn].length + ' 张人脸</div>';
        html += '<div class="facedb-group" data-model="' + mn + '" style="display:grid;grid-template-columns:repeat(auto-fill,minmax(110px,1fr));gap:8px;min-height:40px;padding:4px;border-radius:6px;transition:background 0.15s;">';
        byModel[mn].forEach(function (face) {
          html += '<div draggable="true" data-key="' + face.key + '" style="background:#121214;border:1px solid #2a2a2e;border-radius:6px;padding:8px;cursor:grab;">'
            + '<img src="' + face.thumbUrl + '" style="width:100%;aspect-ratio:1;border-radius:4px;object-fit:cover;display:block;">'
            + '<div style="font-size:9px;margin-top:4px;">' + face.label + '</div>'
            + '<div style="font-size:8px;color:#555;">' + face.source + '</div>'
            + '</div>';
        });
        html += '</div>';
      });
      container.innerHTML = html;

      // Drag-and-drop handlers
      self._dragSrcKey = null;

      container.querySelectorAll('[draggable="true"]').forEach(function (el) {
        el.addEventListener('dragstart', function (e) {
          self._dragSrcKey = this.getAttribute('data-key');
          this.style.opacity = '0.3';
          e.dataTransfer.effectAllowed = 'move';
        });
        el.addEventListener('dragend', function () {
          this.style.opacity = '1';
          container.querySelectorAll('.facedb-group').forEach(function (g) { g.style.background = ''; });
          container.querySelectorAll('.facedb-group-heading').forEach(function (g) { g.style.background = ''; });
        });
      });

      function handleDrop(e) {
        e.preventDefault();
        this.style.background = '';
        var targetModel = this.getAttribute('data-model');
        var srcKey = self._dragSrcKey;
        if (!srcKey || !targetModel) return;
        var fm = app.state.faceModelMap || {};
        var pts = srcKey.split('_');
        var fIdx = parseInt(pts[1] || 0);
        var cur = fm[srcKey] || fm[fIdx];
        if (!cur) {
          var m = Object.keys(app.state.selectedModels || {});
          cur = (m.length === 1) ? m[0] : '未分配';
        }
        if (cur === targetModel) return;
        fm[srcKey] = targetModel;
        app.state.faceModelMap = fm;
        self._populateFaceDB();
      }

      container.querySelectorAll('.facedb-group').forEach(function (g) {
        g.addEventListener('dragover', function (e) { e.preventDefault(); e.dataTransfer.dropEffect = 'move'; this.style.background = 'rgba(91,91,214,0.08)'; });
        g.addEventListener('dragleave', function () { this.style.background = ''; });
        g.addEventListener('drop', handleDrop);
      });
      container.querySelectorAll('.facedb-group-heading').forEach(function (h) {
        h.addEventListener('dragover', function (e) { e.preventDefault(); e.dataTransfer.dropEffect = 'move'; this.style.background = 'rgba(91,91,214,0.12)'; });
        h.addEventListener('dragleave', function () { this.style.background = ''; });
        h.addEventListener('drop', handleDrop);
      });
    },

    _populateSettings: function () {
      var encoderSelect = document.getElementById('layer-export-encoder');
      if (encoderSelect && encoderSelect.options.length === 0) {
        var encoders = ['h264_nvenc', 'h264_amf', 'h264_qsv', 'libx264', 'libx265', 'h264_videotoolbox'];
        for (var i = 0; i < encoders.length; i++) {
          var opt = document.createElement('option');
          opt.value = encoders[i];
          opt.textContent = encoders[i];
          if (encoders[i] === 'h264_nvenc') opt.selected = true;
          encoderSelect.appendChild(opt);
        }
      }
      var formatSelect = document.getElementById('layer-export-format');
      if (formatSelect && formatSelect.options.length === 0) {
        var formats = [['jpeg', 'JPEG'], ['png', 'PNG Sequence'], ['mp4', 'MP4 Video'], ['avi', 'AVI']];
        for (var j = 0; j < formats.length; j++) {
          var opt2 = document.createElement('option');
          opt2.value = formats[j][0];
          opt2.textContent = formats[j][1];
          if (formats[j][0] === 'mp4') opt2.selected = true;
          formatSelect.appendChild(opt2);
        }
      }
    },

    startExport: function () {
      var self = this;
      this.advance(3);
      this._lastSpeed = { done: 0, time: Date.now() };
      var app = window.App;
      var workers = parseInt((document.getElementById('export-workers') || {}).value) || 0;
      var rawFm = (app && app.state.faceModelMap) || {};
      var cleanFm = {};
      Object.keys(rawFm).forEach(function (k) { cleanFm[k] = rawFm[k]; });
      var settings = {
        image_format: (document.getElementById('layer-export-format') || {}).value || 'jpg',
        encoder: (document.getElementById('layer-export-encoder') || {}).value || 'h264_nvenc',
        hwaccel: (document.getElementById('layer-export-gpu') && document.getElementById('layer-export-gpu').checked) ? 'auto' : '',
        output_path: (document.getElementById('export-path') || {}).value || '',
        config: app ? app.state.config : {},
        video_path: app ? app.state.videoPath : '',
        face_db: app ? app.state.faceEmbeddings || {} : {},
        face_model_map: cleanFm,
        cut_segments: (app ? app.state.cutSegments || [] : []).filter(function (s) { return s && typeof s.start === 'number' && typeof s.end === 'number'; }),
        angle_segments: (app ? app.state.angleSegments || [] : []).filter(function (s) { return s && typeof s.start === 'number' && typeof s.end === 'number' && s.angles; }),
        detector: app ? app.state.detector : 'YOLOv8',
        landmarker: app ? app.state.landmarker : 'insightface-2d106det',
        res_scale: app ? app.state.resScale : 0.5,
        num_workers: workers,
      };
      if (window.API && window.API.startExport) {
        window.API.startExport(settings).then(function (data) {
          if (data) self.state.jobId = data.job_id || 'default';
          self._pollProgress();
        });
      }
    },

    computeEmbeddings: function () {
      if (window.API && window.API.computeEmbeddings) {
        var btn = document.getElementById('btn-compute-embeddings');
        if (btn) { btn.textContent = 'Computing...'; btn.disabled = true; }
        var app = window.App;
        window.API.computeEmbeddings({
          video_path: app ? app.state.videoPath : '',
          face_database: app ? app.state.faceDatabase : {},
        }).then(function (data) {
          if (btn) { btn.textContent = 'Compute Face Embeddings'; btn.disabled = false; }
          if (data && data.embedding_count > 0) {
            // Store embeddings and clusters in App state
            if (app) {
              app.state.faceEmbeddings = data.embeddings || {};
              app.state.faceClusters = data.clusters || {};
            }
            document.getElementById('facedb-count').textContent = data.embedding_count + ' embedded · ' + Object.keys(data.clusters || {}).length + ' clusters';
            // Refresh the face DB display to show clustering results
            self._populateFaceDB();
          } else if (data) {
            document.getElementById('facedb-count').textContent = '0 embedded (no faces found)';
          }
        });
      }
    },

    _pollProgress: function () {
      var self = this;
      var last = {done: 0, time: Date.now()};
      var interval = setInterval(function () {
        (window.API && window.API.getExportProgress(self.state.jobId).then(function (data) {
          if (!data) return;
          // Frontend speed: parse done/total from poll delta
          var m = data.message && data.message.match(/(\d+)\s*\/\s*(\d+)/);
          if (m) {
            var done = parseInt(m[1]);
            var now = Date.now();
            // Reset on stage transition or restart (done went backwards)
            if (done < self._lastSpeed.done) {
              self._lastSpeed.done = 0;
              self._lastSpeed.time = now;
            }
            if (done !== self._lastSpeed.done) {
              var dt = (now - self._lastSpeed.time) / 1000;
              if (dt > 0.3 && self._lastSpeed.done > 0 && done > self._lastSpeed.done) {
                var speed = (done - self._lastSpeed.done) / dt;
                data.message = data.message.replace(/\s*·\s*[\d.]+it\/s\s*$/, '') + ' · ' + speed.toFixed(1) + 'it/s';
              }
              self._lastSpeed.done = done;
              self._lastSpeed.time = now;
            }
          }
          if (data.tick > 0 && data.tick % 5 === 0) {
            console.log('Progress:', data.stage, Math.round(data.progress*100)+'%', data.message, 'tick='+data.tick);
          }
          self._updateProgressUI(data);
          if (!data.running) {
            clearInterval(interval);
            if (data.message && data.message !== 'Complete') {
              document.getElementById('progress-label-0').parentElement.textContent = '❌ ' + data.message;
            } else {
              self.advance(0);
              self._showToast('Export completed successfully');
            }
          }
        }));
      }, 100);
    },

    _updateProgressUI: function (data) {
      var stages = ['Extract', 'Detect', 'Match', 'Swap', 'Mask', 'Merge', 'Encode'];
      for (var i = 0; i < 7; i++) {
        var bar = document.getElementById('progress-bar-' + i);
        var label = document.getElementById('progress-label-' + i);
        if (!bar) continue;
        var pct = 0;
        if (data.stage > i) pct = 100;
        else if (data.stage === i) pct = Math.round((data.progress || 0) * 100);
        bar.style.width = pct + '%';
        if (label) {
          if (data.stage > i) label.textContent = stages[i] + ' ✓';
          else if (data.stage === i) {
            var extra = data.message || '';
            label.textContent = stages[i] + '... ' + pct + '%' + (extra ? '  ' + extra : '');
          }
          else label.textContent = stages[i];
        }
      }
    },

    _showToast: function (msg) {
      var el = document.createElement('div');
      el.textContent = msg;
      el.style.cssText = 'position:fixed;bottom:30px;left:50%;transform:translateX(-50%);background:#1a1a1e;border:1px solid rgba(91,91,214,0.3);color:#e0e0e0;padding:10px 24px;border-radius:8px;font:13px Inter,sans-serif;z-index:999;box-shadow:0 4px 24px rgba(0,0,0,0.5);opacity:0;transition:opacity 0.3s;';
      document.body.appendChild(el);
      requestAnimationFrame(function () { el.style.opacity = '1'; });
      setTimeout(function () { el.style.opacity = '0'; setTimeout(function () { el.remove(); }, 300); }, 3000);
    },
  };

  document.addEventListener('DOMContentLoaded', function () { window.ExportFlow.init(); });
})();
