(function () {
  'use strict';

  window.App = {
    state: {
      projectPath: null, videoPath: null, totalFrames: 0, currentFrame: 0, fps: 0,
      models: [], videos: [], videoDflMap: {},  // {videoName: true/false}
      selectedModels: {},  // {modelName: true} for multi-select
      faceModelMap: {},    // {faceIdx: modelName} per-face model assignment
      faceEmbeddings: {},  // {key: [512 floats]} ArcFace embeddings for face DB
      faceClusters: {},    // {main_key: [member_keys]} clustering groups
      resScale: 0.5,
      config: {
        face_type: 'whole_face', mode: 'overlay', mask_mode: 4, seg_mode: 'model',
        erode_mask_modifier: 0, blur_mask_modifier: 0, motion_blur_power: 0,
        output_face_scale: 0, super_resolution_power: 0, color_transfer_mode: 1,
        image_denoise_power: 0, bicubic_degrade_power: 0, color_degrade_power: 0,
        show_debug: false, detect_mode: 'skip_dfl', face_margin: 0.4,
      },
      detector: 'RetinaFace_10g', landmarker: 'insightface-2d106det',
      faceDatabase: {},
      loadedModel: null, cutSegments: [], angleSegments: [], zoom: 1.0,
    },
    _playInterval: null,
    _cachePollInterval: null,

    init: function () {
      this.initTransportSVG();
      this.initEventListeners();
      Params.init();
      Timeline.init();
      Preview.init();
      this.initDragDrop();
      this.initSliders();
      this._loadSavedConfig();
    },

    initTransportSVG: function () {
      var svgs = [
        '<svg width="16" height="16" viewBox="0 0 16 16" fill="none"><path d="M4 2V14" stroke="#9a9b9e" stroke-width="1.2"/><path d="M12 4L7 8L12 12Z" fill="#9a9b9e" stroke="#9a9b9e" stroke-width="0.5"/></svg>',
        '<svg width="16" height="16" viewBox="0 0 16 16" fill="none"><path d="M4 2V14" stroke="#ccc" stroke-width="1.2"/><path d="M12 5L8 8L12 11" stroke="#ccc" stroke-width="1.8" stroke-linecap="round"/></svg>',
        this._playSVG,
        '<svg width="16" height="16" viewBox="0 0 16 16" fill="none"><path d="M4 5L8 8L4 11" stroke="#ccc" stroke-width="1.8" stroke-linecap="round"/><path d="M12 2V14" stroke="#ccc" stroke-width="1.2"/></svg>',
        '<svg width="16" height="16" viewBox="0 0 16 16" fill="none"><path d="M4 4L9 8L4 12Z" fill="#9a9b9e" stroke="#9a9b9e" stroke-width="0.5"/><path d="M12 2V14" stroke="#9a9b9e" stroke-width="1.2"/></svg>',
      ];
      document.getElementById('transport-controls').innerHTML = svgs.join('');
    },

    _playSVG: '<svg width="18" height="18" viewBox="0 0 18 18" fill="none"><path d="M6 5L14 9L6 13V5Z" fill="#ccc"/></svg>',
    _pauseSVG: '<svg width="18" height="18" viewBox="0 0 18 18" fill="none"><rect x="5.5" y="4" width="3" height="10" rx="0.5" fill="#ccc"/><rect x="10.5" y="4" width="3" height="10" rx="0.5" fill="#ccc"/></svg>',

    initEventListeners: function () {
      var self = this;
      document.getElementById('frame-input').addEventListener('change', function (e) {
        var val = parseInt(e.target.value);
        if (!isNaN(val) && val >= 0 && val < self.state.totalFrames) self.seekFrame(val);
      });
      document.getElementById('transport-controls').addEventListener('click', function (e) {
        var target = e.target.closest('svg');
        if (!target) return;
        var idx = Array.from(e.currentTarget.children).indexOf(target);
        if (idx !== 2 && self._playInterval) {
          clearInterval(self._playInterval); self._playInterval = null;
          clearInterval(self._playUIInterval); self._playUIInterval = null;
          Preview.stopPlayback();
          self._setPlayButtonIcon(false);
        }
        if (idx === 0) self.seekFrame(Math.max(0, self.state.currentFrame - 30));
        else if (idx === 1) self.seekFrame(self.state.currentFrame - 1);
        else if (idx === 2) self.togglePlay();
        else if (idx === 3) self.seekFrame(self.state.currentFrame + 1);
        else if (idx === 4) self.seekFrame(Math.min(self.state.totalFrames - 1, self.state.currentFrame + 30));
      });
      document.getElementById('zoom-out').addEventListener('click', function () {
        self.state.zoom = Math.max(0.25, self.state.zoom / 2);
        Timeline.updateZoom(self.state.zoom, self.state.totalFrames, self.state.currentFrame);
      });
      document.getElementById('zoom-in').addEventListener('click', function () {
        self.state.zoom = Math.min(8, self.state.zoom * 2);
        Timeline.updateZoom(self.state.zoom, self.state.totalFrames, self.state.currentFrame);
      });
      // Resolution scale handler
      document.getElementById('res-select').addEventListener('change', function () {
        self.state.resScale = parseFloat(this.value);
        if (self.state.currentFrame > 0) self.loadFrame(self.state.currentFrame);
      });
      document.getElementById('btn-change-workspace').addEventListener('click', function () {
        var path = document.getElementById('workspace-path').value.trim();
        if (path) self.openProject(path);
      });
      // Auto-resize path input on input
      var pathInput = document.getElementById('workspace-path');
      pathInput.addEventListener('input', function () {
        this.style.width = Math.max(200, Math.min(600, this.value.length * 7.5 + 20)) + 'px';
      });
      // Export button handler
      var exportBtn = document.getElementById('btn-export');
      if (exportBtn) {
        exportBtn.addEventListener('click', function (e) {
          if (window.ExportFlow) {
            e.preventDefault();
            window.ExportFlow.advance(1);
          }
        });
      }

      // Collapsible sidebar sections
      document.querySelectorAll('#sidebar .section-header').forEach(function (header) {
        header.addEventListener('click', function () {
          this.classList.toggle('section-header--collapsed');
        });
      });
    },

    initDragDrop: function () {
      var dz = document.getElementById('video-list');
      if (!dz) return;
      dz.addEventListener('dragover', function (e) { e.preventDefault(); dz.classList.add('drop-zone--dragover'); });
      dz.addEventListener('dragleave', function () { dz.classList.remove('drop-zone--dragover'); });
      dz.addEventListener('drop', function (e) { e.preventDefault(); dz.classList.remove('drop-zone--dragover'); });
    },

    _resizeInput: function (inp) {
      if (!inp) return;
      inp.style.width = Math.max(28, inp.value.length * 8 + 8) + 'px';
    },

    initSliders: function () {
      var self = this;
      document.addEventListener('mousedown', function (e) {
        if (e.target.tagName === 'INPUT') return;
        var track = e.target.closest('.param-slider__track');
        if (!track) return;
        var slider = track.closest('.param-slider');
        if (!slider) return;
        var isBipolar = slider.dataset.bipolar === '1';

        var update = function (cx) {
          var r = track.getBoundingClientRect();
          var pct = Math.max(0, Math.min(1, (cx - r.left) / r.width));
          var inp = slider.querySelector('.param-slider__input');
          var fl = slider.querySelector('.param-slider__fill');
          var tb = slider.querySelector('.param-slider__thumb');

          if (isBipolar) {
            var min = parseInt(slider.dataset.min || '-100');
            var max = parseInt(slider.dataset.max || '100');
            var val = Math.round(min + pct * (max - min));
            var zeroPct = (0 - min) / (max - min);
            if (inp) { inp.value = val; self._resizeInput(inp); }
            if (tb) tb.style.left = (pct * 100) + '%';
            if (fl) {
              if (val > 0) {
                fl.className = 'param-slider__fill param-slider__fill--pos';
                fl.style.cssText = 'left:' + (zeroPct * 100) + '%;width:' + ((pct - zeroPct) * 100) + '%';
              } else {
                fl.className = 'param-slider__fill param-slider__fill--neg';
                fl.style.cssText = 'right:' + ((1 - zeroPct) * 100) + '%;width:' + ((zeroPct - pct) * 100) + '%';
              }
            }
          } else {
            var min = parseInt(slider.dataset.min || '0');
            var max = parseInt(slider.dataset.max || '100');
            var val = Math.round(min + pct * (max - min));
            if (inp) { inp.value = val; self._resizeInput(inp); }
            if (fl) fl.style.width = (pct * 100) + '%';
            if (tb) tb.style.left = (pct * 100) + '%';
          }
        };
        update(e.clientX);
        var onMove = function (ev) { update(ev.clientX); };
        var onUp = function () {
          document.removeEventListener('mousemove', onMove);
          document.removeEventListener('mouseup', onUp);
          var app = window.App;
          if (app) {
            var param = slider.dataset.param;
            var inp = slider.querySelector('.param-slider__input');
            if (param && inp) {
              var raw = parseInt(inp.value) || 0;
              app.state.config[param] = param === 'face_margin' ? raw / 100 : raw;
              if (param === 'face_margin') {
                app.loadFrame(app.state.currentFrame);
              } else {
                app.remergeFrame(app.state.currentFrame);
              }
            }
          }
        };
        document.addEventListener('mousemove', onMove);
        document.addEventListener('mouseup', onUp);
      });
      // Input resize on typing
      document.addEventListener('input', function (e) {
        var inp = e.target.closest('.param-slider__input');
        if (inp) self._resizeInput(inp);
      });
      // Input change handler for manual typing
      document.addEventListener('change', function (e) {
        var inp = e.target.closest('.param-slider__input');
        if (!inp) return;
        var slider = inp.closest('.param-slider');
        if (!slider) return;
        self._resizeInput(inp);
        var val = parseInt(inp.value);
        if (isNaN(val)) return;
        var min = parseInt(slider.dataset.min || '0');
        var max = parseInt(slider.dataset.max || '100');
        val = Math.max(min, Math.min(max, val));
        inp.value = val;
        var pct = max > min ? (val - min) / (max - min) : 0;
        var fl = slider.querySelector('.param-slider__fill');
        var tb = slider.querySelector('.param-slider__thumb');
        if (slider.dataset.bipolar === '1') {
          var zeroPct = (0 - min) / (max - min);
          if (tb) tb.style.left = (pct * 100) + '%';
          if (fl) {
            if (val > 0) {
              fl.className = 'param-slider__fill param-slider__fill--pos';
              fl.style.cssText = 'left:' + (zeroPct * 100) + '%;width:' + ((pct - zeroPct) * 100) + '%';
            } else {
              fl.className = 'param-slider__fill param-slider__fill--neg';
              fl.style.cssText = 'right:' + ((1 - zeroPct) * 100) + '%;width:' + ((zeroPct - pct) * 100) + '%';
            }
          }
        } else {
          if (fl) fl.style.width = (pct * 100) + '%';
          if (tb) tb.style.left = (pct * 100) + '%';
        }
        var app = window.App;
        if (app) {
          var p = slider.dataset.param;
          app.state.config[p] = p === 'face_margin' ? val / 100 : val;
          if (p === 'face_margin') {
            app.loadFrame(app.state.currentFrame);
          } else {
            app.remergeFrame(app.state.currentFrame);
          }
        }
      });
    },

    openProject: function (path) {
      var self = this;
      API.openProject(path).then(function (data) {
        if (data.status === 'ok') {
          self.state.projectPath = path; self.state.videoPath = null;
          self.state.totalFrames = data.total_frames; self.state.fps = data.fps;
          self.state.cutSegments = [];
          self.state.angleSegments = [];
          document.getElementById('workspace-path').value = path;
          document.getElementById('frame-total').textContent = data.total_frames;
          document.getElementById('status-indicator').textContent = '● loaded';
          self.state.models = data.models || [];
          self.state.videos = data.videos || [];
          self.state.videoDflMap = data.video_dfl_map || {};
          Timeline.updateZoom(self.state.zoom, data.total_frames, 0);
          self.renderModelList(self.state.models);
          self.renderVideoList(self.state.videos);
          // Don't auto-seek — user clicks a video to load it
          if (data.has_aligned && data.aligned_count > 0) {
            self.state.faceDensityData = [{ frame: 0, face_count: data.aligned_count }];
            Timeline.renderFaceDensityLine(self.state.faceDensityData);
          }
          self._startCachePoll();
        } else { alert('Failed: ' + (data.message || '')); }
      });
    },

    _startCachePoll: function () {
      var self = this;
      if (this._cachePollInterval) clearInterval(this._cachePollInterval);
      this._lastCachePct = -1;
      this._cachePollInterval = setInterval(function () {
        API.getCacheStatus().then(function (data) {
          if (data && data.total > 0 && data.pct !== self._lastCachePct) {
            self._lastCachePct = data.pct;
            var el = document.getElementById('status-indicator');
            if (data.pct < 100) {
              el.textContent = '● cache ' + data.pct + '% (' + data.cached + '/' + data.total + ')';
            } else {
              el.textContent = '● cached';
              clearInterval(self._cachePollInterval);
              self._cachePollInterval = null;
            }
          }
        });
      }, 5000);
    },

    selectVideo: function (name) {
      var self = this;
      if (!this.state.projectPath) return;
      var fullPath = this.state.projectPath.replace(/\\/g, '/') + '/' + name;
      self.setPanelLoading('loading-original', true, 'Loading video...');
      API.selectVideo(fullPath).then(function (data) {
        self.setPanelLoading('loading-original', false);
        if (data.status === 'ok') {
          self.state.videoPath = fullPath; self.state.totalFrames = data.total_frames; self.state.fps = data.fps;
          self.state.faceDensityData = [];
          self.state.faceDatabase = {};  // clear face DB on video switch
          // Reset video element for new video (lazy load on next play)
          var vid = document.getElementById('video-original');
          if (vid) { vid.removeAttribute('src'); vid.load(); }
          document.getElementById('frame-total').textContent = data.total_frames;
          document.getElementById('status-indicator').textContent = '● ' + name;
          self.renderVideoList(self.state.videos);
          self.renderFaceSection();
          // Auto-set detect_mode for DFL projects
          self.state.config.detect_mode = (data.is_dfl && data.aligned_count > 0) ? 'skip_dfl' : 'always';
          var dm = document.getElementById('detect-mode-select');
          if (dm) dm.value = self.state.config.detect_mode;
          Timeline.updateZoom(self.state.zoom, data.total_frames, 0);
          self.seekFrame(0);
          self._startCachePoll();
        }
      });
    },

    selectModel: function (modelName) {
      var self = this;
      if (!this.state.projectPath) return;
      var isActive = !!this.state.selectedModels[modelName];
      if (isActive) {
        // Deselect
        delete self.state.selectedModels[modelName];
        self.renderModelList(self.state.models);
        return;
      }
      // Load model
      var modelSection = document.getElementById('model-section');
      if (modelSection) modelSection.classList.add('model-section--loading');
      API.loadModel(modelName).then(function (data) {
        if (modelSection) modelSection.classList.remove('model-section--loading');
        if (data.status === 'loading' || data.status === 'loaded') {
          self.state.selectedModels[modelName] = true;
          // If single model, set as default loadedModel
          var keys = Object.keys(self.state.selectedModels);
          if (keys.length === 1) {
            self.state.loadedModel = modelName;
          }
          document.getElementById('status-indicator').textContent = '● models: ' + keys.join(', ');
          self.renderModelList(self.state.models);
          if (self.state.videoPath && self.state.totalFrames > 0) {
            self.loadFrame(self.state.currentFrame);
          }
        }
      });
    },

    seekFrame: function (idx, quick) {
      this.state.currentFrame = idx;
      document.getElementById('frame-input').value = idx;
      var vid = document.getElementById('video-original');
      // During playback: seek the <video> element directly (GPU fast seek)
      if (quick && this._playInterval && vid && vid.readyState >= 1) {
        vid.currentTime = idx / Math.max(1, this.state.fps || 30);
        if (Timeline._syncScroll) Timeline._syncScroll(idx);
        if (Timeline.updatePlayhead) Timeline.updatePlayhead(idx, this.state.totalFrames);
      } else if (quick) {
        if (Timeline._syncScroll) Timeline._syncScroll(idx);
        if (Timeline.updatePlayhead) Timeline.updatePlayhead(idx, this.state.totalFrames);
      } else {
        // Non-quick seek: stop playback first, then show frame
        if (this._playInterval) {
          this._cancelBuffering();
          clearInterval(this._playInterval); this._playInterval = null;
          clearInterval(this._playUIInterval); this._playUIInterval = null;
          Preview.stopPlayback();
          this._setPlayButtonIcon(false);
        }
        if (Timeline._syncScroll) Timeline._syncScroll(idx);
        this.loadFrame(idx);
      }
      this.updateTimecode(idx);
    },

    loadFrame: function (idx) {
      var self = this;
      if (!this.state.videoPath) return;
      // Generation counter: every frame switch increments it. All async
      // callbacks capture gen at creation time and bail out if it's stale.
      // This is race-proof: even if an old Image.onload fires before the
      // event-loop reaches loadFrame, its captured gen won't match.
      self._frameGen = (self._frameGen || 0) + 1;
      var gen = self._frameGen;
      self._currentFrameId = idx;
      Preview.stopPlayback();
      Preview.clearAll();
      self.setPanelLoading('loading-original', true, 'Loading frame...');
      self.setPanelLoading('loading-detection', true, 'Detecting faces...');
      self.setPanelLoading('loading-swapped', true, 'Merging...');
      API.getFrame(idx, 85, self.state.videoPath).then(function (r) { return r.blob(); }).then(function (blob) {
        if (gen !== self._frameGen) return;
        var url = URL.createObjectURL(blob);
        Preview.updateOriginal(url);
        self.setPanelLoading('loading-original', false);
        var selected = [];
        Object.keys(self.state.faceDatabase).forEach(function (key) {
          var parts = key.split('_');
          var fidx = parseInt(parts[0]);
          if (fidx === idx) {
            selected.push(parseInt(parts[1] || 0));
          }
        });
        var frameModelMap = {};
        Object.keys(self.state.faceModelMap).forEach(function (k) {
          var p = k.split('_');
          if (parseInt(p[0]) === idx) frameModelMap[parseInt(p[1] || 0)] = self.state.faceModelMap[k];
        });
        var analyzePromise = API.analyzeFrame({
          frame_idx: idx, config: self.state.config,
          detector: self.state.detector, landmarker: self.state.landmarker,
          res_scale: self.state.resScale,
          selected_faces: selected,
          face_model_map: frameModelMap,
          angle_segments: self.state.angleSegments || [],
        });

        // Poll: show detection image (already has boxes+landmarks from OpenCV)
        var detUrl = '/api/preview/detection/' + idx;
        (function pollDet() {
          if (gen !== self._frameGen) return;
          var img = new Image();
          img.onload = function () {
            if (gen !== self._frameGen) return;
            self.setPanelLoading('loading-detection', false);
            Preview._drawImage(Preview.canvasDetection, img, []);
          };
          img.onerror = function () { setTimeout(pollDet, 50); };
          img.src = detUrl + '?_=' + Date.now();
        })();

        self._saveState();
        analyzePromise.then(function (data) {
          if (gen !== self._frameGen) return;
          if (data) {
            // Detection is complete. If no faces, force-hide detection spinner
            // (pollDet may not have loaded the image yet).
            if (data.face_count === 0) {
              self.setPanelLoading('loading-detection', false);
              Preview.updateDetection(url, []);
            }
            var t = Date.now();
            if (data.debug_urls && data.debug_urls.length > 0) {
              Preview.showDebugGrid(data.debug_urls);
              var loaded = 0, total = data.debug_urls.length;
              data.debug_urls.forEach(function (u) {
                var im = new Image();
                im.onload = function () { if (++loaded >= total) self.setPanelLoading('loading-swapped', false); };
                im.onerror = function () { if (++loaded >= total) self.setPanelLoading('loading-swapped', false); };
                im.src = u + '?_=' + Date.now();
              });
            } else {
              Preview.hideDebugGrid();
              if (data.swapped_url) {
                (function pollSwap() {
                  if (gen !== self._frameGen) return;
                  var img = new Image();
                  img.onload = function () {
                    if (gen !== self._frameGen) return;
                    self.setPanelLoading('loading-swapped', false);
                    Preview.updateSwapped(data.swapped_url + '?_=' + Date.now());
                  };
                  img.onerror = function () { setTimeout(pollSwap, 100); };
                  img.src = data.swapped_url + '?_=' + Date.now();
                })();
              } else {
                Preview.updateSwapped(url);
                self.setPanelLoading('loading-swapped', false);
              }
            }
            document.getElementById('frame-info').textContent = 'Frame ' + idx + ' (' + data.face_count + ' faces)';
            self.state.currentFaces = (data.faces || []).map(function (f, fi) {
              return { key: idx + '_' + fi, face: f, frameIdx: idx, faceIdx: fi };
            });
            self.renderFaceSection();
          } else {
            self.setPanelLoading('loading-swapped', false);
          }
        });
      });
    },

    remergeFrame: function (idx) {
      var self = this;
      if (!this.state.videoPath) return;
      self.setPanelLoading('loading-swapped', true, 'Merging...');
      var selected = [];
      Object.keys(self.state.faceDatabase).forEach(function (key) {
        var parts = key.split('_');
        var fidx = parseInt(parts[0]);
        if (fidx === idx) selected.push(parseInt(parts[1] || 0));
      });
      // Build per-frame face model map from full-key map
      var frameModelMap = {};
      Object.keys(self.state.faceModelMap).forEach(function (k) {
        var p = k.split('_');
        if (parseInt(p[0]) === idx) frameModelMap[parseInt(p[1] || 0)] = self.state.faceModelMap[k];
      });
      self._saveState();
      API.remergeFrame({
        frame_idx: idx, config: self.state.config,
        detector: self.state.detector, landmarker: self.state.landmarker,
        res_scale: self.state.resScale,
        selected_faces: selected,
        face_model_map: frameModelMap,
        angle_segments: self.state.angleSegments || [],
      }).then(function (data) {
        if (data) {
          var t = Date.now();
          if (data.debug_urls && data.debug_urls.length > 0) {
            Preview.showDebugGrid(data.debug_urls);
            var loaded = 0, total = data.debug_urls.length;
            data.debug_urls.forEach(function (u) {
              var im = new Image();
              im.onload = function () { if (++loaded >= total) self.setPanelLoading('loading-swapped', false); };
              im.onerror = function () { if (++loaded >= total) self.setPanelLoading('loading-swapped', false); };
              im.src = u + '?_=' + Date.now();
            });
          } else {
            Preview.hideDebugGrid();
            if (data.swapped_url) {
              (function pollSwap() {
                if (self._currentFrameId !== idx) return;
                var img = new Image();
                img.onload = function () {
                  if (self._currentFrameId !== idx) return;
                  self.setPanelLoading('loading-swapped', false);
                  Preview.hideDebugGrid();
                  Preview.updateSwapped(data.swapped_url + '?_=' + Date.now());
                };
                img.onerror = function () { setTimeout(pollSwap, 100); };
                img.src = data.swapped_url + '?_=' + Date.now();
              })();
            } else {
              self.setPanelLoading('loading-swapped', false);
            }
          }
        } else {
          self.setPanelLoading('loading-swapped', false);
        }
      });
    },

    renderFaceSection: function () {
      var self = this;
      var container = document.getElementById('face-db-list');
      var countEl = document.getElementById('face-db-count');
      if (!container) return;

      container.innerHTML = '';
      var dbKeys = Object.keys(self.state.faceDatabase);
      var hasCurrent = false;

      // Section 1 (top): Current frame faces not yet in database
      (self.state.currentFaces || []).forEach(function (cf) {
        if (!self.state.faceDatabase[cf.key]) {
          var item = {
            key: cf.key, saved: false,
            thumbUrl: cf.face.thumb_url || null,
            label: 'Face ' + cf.faceIdx,
            source: 'Frame ' + cf.frameIdx,
            faceIdx: cf.faceIdx,
          };
          var div = self._renderFaceItemDiv(item);
          container.appendChild(div);
          hasCurrent = true;
        }
      });

      // Separator
      if (hasCurrent && dbKeys.length > 0) {
        var sep = document.createElement('div');
        sep.className = 'face-db-separator';
        container.appendChild(sep);
      }

      // Section 2 (bottom, scrollable): Saved faces from face database (cross-frame)
      if (dbKeys.length > 0) {
        var savedWrap = document.createElement('div');
        savedWrap.className = 'face-db-saved';
        dbKeys.forEach(function (key) {
          var parts = key.split('_');
          var fidx = parseInt(parts[0]), fi = parseInt(parts[1] || 0);
          var fd = self.state.faceDatabase[key];
          var fdObj = typeof fd === 'object' ? fd : {};
          var item = {
            key: key, saved: true,
            thumbUrl: fdObj.thumb_url || null,
            label: fdObj.label || 'Face ' + fi,
            source: 'Frame ' + fidx,
            faceIdx: fi,
          };
          var div = self._renderFaceItemDiv(item);
          savedWrap.appendChild(div);
        });
        container.appendChild(savedWrap);
      }

      if (!hasCurrent && dbKeys.length === 0) {
        container.innerHTML = '<div style="font-size:10px;color:#6b6c70;padding:4px 0;">No faces detected</div>';
      }
      if (countEl) countEl.textContent = dbKeys.length + ' in DB';
    },

    _renderFaceItemDiv: function (item) {
      var self = this;
      var isChecked = item.saved;
      var div = document.createElement('div');
      div.className = 'face-db-item ' + (isChecked ? 'face-db-item--checked' : 'face-db-item--unchecked');

      var cbHtml = isChecked
        ? '<svg width="16" height="16" viewBox="0 0 16 16" fill="none"><circle cx="8" cy="8" r="7" stroke="#6f8f90" stroke-width="1.5"/><circle cx="8" cy="8" r="4" fill="#6f8f90"/></svg>'
        : '<svg width="16" height="16" viewBox="0 0 16 16" fill="none"><circle cx="8" cy="8" r="7" stroke="#3a3a3e" stroke-width="1.5"/></svg>';

      var modelOpts = Object.keys(self.state.selectedModels);
      var currentModel = self.state.faceModelMap[item.key] || (modelOpts.length > 0 ? modelOpts[modelOpts.length - 1] : '');
      var modelSelectHtml = '';
      if (modelOpts.length > 1) {
        modelSelectHtml = '<select class="face-model-select" style="font-size:9px;background:#1a1a1e;border:1px solid #222329;border-radius:3px;padding:1px 4px;color:#ccc;margin-left:4px;" data-face="' + item.faceIdx + '">' +
          modelOpts.map(function (mn) {
            return '<option value="' + mn + '"' + (mn === currentModel ? ' selected' : '') + '>' + mn + '</option>';
          }).join('') + '</select>';
      }

      div.innerHTML =
        cbHtml +
        (item.thumbUrl ? '<img class="face-db-item__thumb" src="' + item.thumbUrl + '">' : '<div class="face-db-item__thumb" style="background:#1a1a1e"></div>') +
        '<div class="face-db-item__info">' +
        '  <div class="face-db-item__name">' + item.label + '</div>' +
        '  <div class="face-db-item__source">' + item.source + '</div>' +
        '</div>' +
        modelSelectHtml;

      if (modelOpts.length > 1) {
        var sel = div.querySelector('.face-model-select');
        if (sel) {
          sel.addEventListener('click', function (e) { e.stopPropagation(); });
          sel.addEventListener('change', function () {
            self.state.faceModelMap[item.key] = this.value;
            self.remergeFrame(self.state.currentFrame);
          });
        }
      }

      div.addEventListener('click', function () {
        var db = self.state.faceDatabase;
        if (db[item.key]) {
          delete db[item.key];
          delete self.state.faceModelMap[item.key];
        } else {
          db[item.key] = { thumb_url: item.thumbUrl, label: item.label };
          // Save current model assignment when adding to DB
          if (modelOpts.length > 0) {
            var sel = div.querySelector('.face-model-select');
            if (sel) {
              self.state.faceModelMap[item.key] = sel.value;
            } else if (modelOpts.length === 1) {
              self.state.faceModelMap[item.key] = modelOpts[0];
            }
          }
        }
        self.renderFaceSection();
        self.remergeFrame(self.state.currentFrame);
      });
      return div;
    },

    togglePlay: function () {
      var vid = document.getElementById('video-original');
      if (!vid) return;

      // ---- PAUSE: cancel everything ----
      if (this._playInterval) {
        this._cancelBuffering();
        this.setPanelLoading('loading-original', false);
        clearInterval(this._playInterval); this._playInterval = null;
        clearInterval(this._playUIInterval); this._playUIInterval = null;
        Preview.stopPlayback();
        if (vid.currentTime > 0 && this.state.fps > 0) {
          var f = Math.round(vid.currentTime * this.state.fps);
          this.state.currentFrame = Math.min(f, this.state.totalFrames - 1);
        }
        this.loadFrame(this.state.currentFrame);
        this._setPlayButtonIcon(false);
        return;
      }

      // ---- PLAY (with pre-buffering) ----
      var self = this;
      this._cancelBuffering();
      Preview.setVideoSrc('/api/preview/video-stream?v=' + Date.now());
      vid.currentTime = this.state.currentFrame / Math.max(1, this.state.fps || 30);
      document.getElementById('status-indicator').textContent = '● buffering...';
      self.setPanelLoading('loading-original', true, 'Buffering video...');
      self._buffering = { cancelled: false };

      var startPlay = function () {
        if (self._buffering && self._buffering.cancelled) return;
        self._cancelBuffering();
        self.setPanelLoading('loading-original', false);
        Preview.showPlayback();
        vid.play();
        self._setPlayButtonIcon(true);
        self._playInterval = setInterval(function () {}, 1000000);
        self._playUIInterval = setInterval(function () {
          if (!vid.paused && vid.currentTime > 0 && self.state.fps > 0) {
            var f = Math.round(vid.currentTime * self.state.fps);
            f = Math.min(f, self.state.totalFrames - 1);
            if (f !== self.state.currentFrame) {
              self.state.currentFrame = f;
              document.getElementById('frame-input').value = f;
              self.updateTimecode(f);
              Timeline._syncScroll(f);
              Timeline.updatePlayhead(f, self.state.totalFrames);
            }
          }
          var ba = Preview.bufferedAhead();
          if (ba > 0) {
            document.getElementById('status-indicator').textContent = '● ' + Math.round(ba) + 's buffered';
          }
        }, 200);
      };

      var bufferedEnough = function () {
        return vid.buffered.length > 0 &&
          vid.buffered.end(vid.buffered.length - 1) - vid.currentTime >= 10;
      };

      if (vid.readyState >= 3 || bufferedEnough()) {
        startPlay();
      } else {
        var checkBuffer = setInterval(function () {
          if (self._buffering && self._buffering.cancelled) {
            clearInterval(checkBuffer);
            return;
          }
          if (bufferedEnough() || vid.readyState >= 3) {
            clearInterval(checkBuffer);
            startPlay();
          }
        }, 200);

        var onCanPlay = function () {
          clearInterval(checkBuffer);
          startPlay();
        };
        vid.addEventListener('canplaythrough', onCanPlay, { once: true });

        // Fallback after 15s
        var fallbackTimer = setTimeout(function () {
          if (self._buffering && self._buffering.cancelled) return;
          clearInterval(checkBuffer);
          vid.removeEventListener('canplaythrough', onCanPlay);
          document.getElementById('status-indicator').textContent = '● playing';
          startPlay();
        }, 15000);

        // Save refs so _cancelBuffering can clean them up
        this._buffering = {
          cancelled: false,
          checkBuffer: checkBuffer,
          onCanPlay: onCanPlay,
          fallbackTimer: fallbackTimer,
        };
      }
    },

    setPanelLoading: function (panelId, show, label) {
      var el = document.getElementById(panelId);
      if (!el) return;
      if (show) {
        el.style.display = 'flex';
        var lbl = el.querySelector('.panel-loading-label');
        if (lbl && label) lbl.textContent = label;
      } else {
        el.style.opacity = '0';
        var self = this;
        setTimeout(function () { el.style.display = 'none'; el.style.opacity = '1'; }, 200);
      }
    },

    _cancelBuffering: function () {
      if (this._buffering) {
        this._buffering.cancelled = true;
        if (this._buffering.checkBuffer) clearInterval(this._buffering.checkBuffer);
        if (this._buffering.onCanPlay) {
          var vid = document.getElementById('video-original');
          if (vid) vid.removeEventListener('canplaythrough', this._buffering.onCanPlay);
        }
        if (this._buffering.fallbackTimer) clearTimeout(this._buffering.fallbackTimer);
        this._buffering = null;
      }
    },

    _setPlayButtonIcon: function (isPlaying) {
      var tc = document.getElementById('transport-controls');
      if (!tc) return;
      var child = tc.children[2];
      if (child) child.outerHTML = isPlaying ? this._pauseSVG : this._playSVG;
    },

    updateTimecode: function (idx) {
      if (this.state.fps > 0) {
        var s = idx / this.state.fps;
        var m = Math.floor(s / 60); s = Math.floor(s % 60);
        var cs = Math.floor((idx / this.state.fps % 1) * 100);
        document.getElementById('timecode').textContent =
          (m < 10 ? '0' : '') + m + ':' + (s < 10 ? '0' : '') + s + '.' + (cs < 10 ? '0' : '') + cs;
      }
    },

    renderModelList: function (models) {
      var self = this;
      var container = document.getElementById('model-list');
      if (!container) return;
      container.innerHTML = '';
      if (!models || models.length === 0) {
        container.innerHTML = '<div class="list-item list-item--disabled"><span class="list-item__name" style="color:#6b6c70">No models found</span></div>';
        return;
      }
      models.forEach(function (m) {
        var isDfm = m.format === 'dfm';
        var isSelected = !!self.state.selectedModels[m.name];
        var item = document.createElement('div');
        var cls = 'list-item';
        if (isSelected) cls += ' list-item--selected';
        else if (isDfm) cls += ' list-item--active';
        else cls += ' list-item--disabled';
        item.className = cls;
        var checkHtml = isDfm ? '<span class="list-item__check">' + (isSelected ? '✓' : '○') + '</span>' : '';
        item.innerHTML =
          checkHtml +
          '<span class="list-item__name">' + m.name + '</span>' +
          '<span class="list-item__meta">' + (isDfm ? 'ONNX' : '需导出') + '</span>';
        if (isDfm) {
          item.addEventListener('click', function () { self.selectModel(m.name); });
        }
        container.appendChild(item);
      });
    },

    renderVideoList: function (videos) {
      var self = this;
      var container = document.getElementById('video-list');
      if (!container) return;
      container.innerHTML = '';
      if (!videos || videos.length === 0) {
        container.innerHTML = '<div class="drop-zone">Drop video here</div>';
        return;
      }
      videos.forEach(function (v) {
        var isSelected = self.state.videoPath && self.state.videoPath.endsWith('/' + v);
        var cls = 'list-item' + (isSelected ? ' list-item--selected' : ' list-item--active');
        var item = document.createElement('div');
        item.className = cls;
        var dflBadge = self.state.videoDflMap && self.state.videoDflMap[v] ? '<span class="list-item__dfl">DFL</span>' : '';
        item.innerHTML = '<span class="list-item__name">' + v + '</span>' + dflBadge;
        item.addEventListener('click', function () { self.selectVideo(v); });
        container.appendChild(item);
      });
    },

    _saveState: function () {
      var s = this.state;
      var cfg = {};
      // Normalize: convert int color_transfer_mode to string for consistent save
      Object.keys(s.config).forEach(function (k) { cfg[k] = s.config[k]; });
      if (typeof cfg.color_transfer_mode === 'number') {
        cfg.color_transfer_mode = {0:'none',1:'rct',2:'lct',3:'mkl',4:'idt',5:'sot-m',6:'mix-m',9:'mt',10:'lut'}[cfg.color_transfer_mode] || cfg.color_transfer_mode;
      }
      API.saveConfig({
        config: cfg,
        detector: s.detector,
        landmarker: s.landmarker,
        resScale: s.resScale,
      });
    },

    _loadSavedConfig: function () {
      var self = this;
      API.loadConfig().then(function (saved) {
        if (!saved || !saved.config) return;
        if (saved.detector) self.state.detector = saved.detector;
        if (saved.landmarker) self.state.landmarker = saved.landmarker;
        if (saved.resScale) self.state.resScale = saved.resScale;
        if (saved.config) Object.assign(self.state.config, saved.config);
        // Refresh UI controls to reflect loaded values
        Params.refreshUI();
      });
    },
  };

  document.addEventListener('DOMContentLoaded', function () { window.App.init(); });
})();

// ── Layout toggle (ui-dark) ──
(function() {
  var layoutBtn = document.getElementById('layout-toggle');
  if (!layoutBtn) return;
  var saved = localStorage.getItem('ms_layout') || '2col';
  var pa = document.getElementById('preview-area');
  function applyLayout(mode) {
    pa.classList.remove('layout-2col', 'layout-3col-swap-big');
    if (mode === '2col') pa.classList.add('layout-2col');
    else if (mode === '3col-big') pa.classList.add('layout-3col-swap-big');
    layoutBtn.textContent = mode === '2col' ? '2栏' : '3栏';
    localStorage.setItem('ms_layout', mode);
  }
  applyLayout(saved);
  layoutBtn.addEventListener('click', function() {
    var cur = localStorage.getItem('ms_layout') || '2col';
    var next = cur === '2col' ? '3col' : (cur === '3col' ? '3col-big' : '2col');
    applyLayout(next);
  });
})();


// ── Draggable preview layout (ui-dark v2) ──
(function() {
  var pa = document.getElementById('preview-area');
  if (!pa) return;
  var splitter = document.getElementById('preview-splitter');
  var detP = document.getElementById('preview-detection');
  var swpP = document.getElementById('preview-swapped');
  var btnH = document.getElementById('btn-layout-h');
  var btnV = document.getElementById('btn-layout-v');
  if (!splitter || !detP || !swpP) return;

  var isDragging = false, startPos = 0, startPct = 50;
  var mode = localStorage.getItem('ms_layout_mode') || 'h';
  var ratio = parseFloat(localStorage.getItem('ms_layout_ratio')) || 50;

  function applyLayout() {
    pa.classList.toggle('layout-vertical', mode === 'v');
    detP.classList.add('drag-set');
    swpP.classList.add('drag-set');
    if (mode === 'h') {
      detP.style.width = ratio + '%'; detP.style.height = '';
      swpP.style.width = (100 - ratio) + '%'; swpP.style.height = '';
    } else {
      detP.style.height = ratio + '%'; detP.style.width = '';
      swpP.style.height = (100 - ratio) + '%'; swpP.style.width = '';
    }
    btnH.classList.toggle('active', mode === 'h');
    btnV.classList.toggle('active', mode === 'v');
    localStorage.setItem('ms_layout_mode', mode);
    localStorage.setItem('ms_layout_ratio', ratio);
  }

  splitter.addEventListener('mousedown', function(e) {
    e.preventDefault(); isDragging = true; startPos = mode === 'h' ? e.clientX : e.clientY;
    startPct = ratio; splitter.classList.add('dragging');
    document.body.style.cursor = mode === 'h' ? 'col-resize' : 'row-resize';
    document.body.style.userSelect = 'none';
  });
  document.addEventListener('mousemove', function(e) {
    if (!isDragging) return;
    var rect = pa.getBoundingClientRect();
    var pct;
    if (mode === 'h') pct = ((e.clientX - rect.left) / rect.width) * 100;
    else pct = ((e.clientY - rect.top) / rect.height) * 100;
    ratio = Math.max(15, Math.min(85, pct));
    applyLayout();
  });
  document.addEventListener('mouseup', function() {
    if (!isDragging) return;
    isDragging = false; splitter.classList.remove('dragging');
    document.body.style.cursor = ''; document.body.style.userSelect = '';
    localStorage.setItem('ms_layout_ratio', ratio);
  });

  if (btnH) btnH.addEventListener('click', function() { mode = 'h'; applyLayout(); });
  if (btnV) btnV.addEventListener('click', function() { mode = 'v'; applyLayout(); });
  applyLayout();
})();
