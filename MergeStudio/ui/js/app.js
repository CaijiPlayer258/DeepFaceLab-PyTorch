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
      this._syncFromBackend();
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
      var tlZoomOut = document.getElementById('zoom-out');
      if (tlZoomOut) tlZoomOut.addEventListener('click', function () {
        self.state.zoom = Math.max(0.25, self.state.zoom / 2);
        Timeline.updateZoom(self.state.zoom, self.state.totalFrames, self.state.currentFrame);
      });
      var tlZoomIn = document.getElementById('zoom-in');
      if (tlZoomIn) tlZoomIn.addEventListener('click', function () {
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
            if (window.MSFaceFocus) window.MSFaceFocus.setFrameSize([img.naturalWidth, img.naturalHeight]);
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
            } else {
              // pollDet draws the image but never sets the badge — do it here,
              // and tell the user which geometry source was used
              var fc = document.getElementById('face-count');
              if (fc) fc.textContent = data.face_count + ' faces' +
                (data.detect_source === 'dfl' ? ' · DFL' : '');
            }
            // 聚焦人脸：按本帧的人脸数据自动放大居中
            if (window.MSFaceFocus) window.MSFaceFocus.setFaces(data.faces || []);
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

    // ── 自动加载：页面打开时同步后端已就绪的 workspace/模型/视频 (autoload-sync) ──
    _syncFromBackend: function () {
      var self = this;
      var si = document.getElementById('status-indicator');
      var attempts = 0, MAX = 150; // ~3min @1.2s（阶段推进会重置；超时不终止，降频续轮询）
      var lastStage = null;
      var syncedOnce = false;                     // 已同步成功一次（用于检测后端重启）
      var handoverTimer = null;
      function poll() {
        // 已同步过：低频监听后端健康。后端重启（video/model 变空）→ 自动重新同步，
        // 旧标签页不再永久失联（2026-10-09 修复：重启服务后旧页面一直空白）。
        if (self.state.projectPath && syncedOnce) {
          if (handoverTimer) return;
          handoverTimer = setTimeout(function () {
            handoverTimer = null;
            if (!self.state.projectPath) return;
            API.getStatus().then(function (st) {
              if (st && st.video && st.model && st.model.loaded &&
                  st.video.path === self.state.videoPath) return;  // 同一视频仍在：后端正常
              syncedOnce = false;                                  // 后端重启/换片：重新接管
              self.state.projectPath = '';
              if (si) si.textContent = '\u25cf 后端重启，重新同步…';
              poll();
            }).catch(function () {});           // 暂时性错误：下轮再测
          }, 5000);
          return;
        }
        if (self.state.projectPath) return; // 用户已手动接管（未同步过 = 真手动，不干扰）
        API.getStatus().then(function (st) {
          if (self.state.projectPath && !syncedOnce) return;
          var al = st.autoload || {};
          var video = st.video, model = st.model;
          // 自动加载流程：隐藏左侧文件栏（顶栏「文件栏」按钮可再展开）
          if (((al && al.configured) || video || model) && window.MSSidebar &&
              localStorage.getItem('ms_sidebar') !== 'hidden') {
            window.MSSidebar('hidden');
          }
          if (video && model && model.loaded) {
            // 就绪：拉取列表，完整填充页面（等价于手动 open→选视频→选模型的最终态）
            API.getStatus(true).then(function (st2) {
              if (self.state.projectPath) return;
              if (!st.workspace) { setTimeout(poll, 1500); return; } // 状态不完整：等下一轮，不半初始化
              self.state.projectPath = st.workspace;
              document.getElementById('workspace-path').value = st.workspace || '';
              self.state.models = st2.models || [];
              self.state.videos = st2.videos || [];
              self.state.videoDflMap = {};
              self.state.cutSegments = [];
              self.state.angleSegments = [];
              self.state.videoPath = video.path;
              self.state.totalFrames = video.total_frames;
              self.state.fps = video.fps || 0;
              self.state.faceDensityData = [];
              self.state.faceDatabase = {};
              self.state.selectedModels = {};
              self.state.selectedModels[model.name] = true;
              self.state.loadedModel = model.name;
              self.state.config.detect_mode = video.has_aligned_dir ? 'skip_dfl' : 'always';
              var dm = document.getElementById('detect-mode-select');
              if (dm) dm.value = self.state.config.detect_mode;
              var vid = document.getElementById('video-original');
              if (vid) { vid.removeAttribute('src'); vid.load(); }
              document.getElementById('frame-total').textContent = video.total_frames;
              self.renderModelList(self.state.models);
              self.renderVideoList(self.state.videos);
              self.renderFaceSection();
              Timeline.updateZoom(self.state.zoom, video.total_frames, 0);
              if (si) si.textContent = '\u2713 ' + model.name + ' + ' + video.name +
                ' (' + Number(video.total_frames).toLocaleString() + '\u5e27)';
              syncedOnce = true;
              self.seekFrame(0);
              self._seekFirstVisibleFrame();       // 片头黑帧 → 跳到首个有画面的帧
              self._startCachePoll();
            }).catch(function () { setTimeout(poll, 1500); }); // 就绪瞬间的瞬时错误：重试，不让轮询静默死亡
            return;
          }
          // 未就绪：显示进度并继续轮询
          if (!al.configured && !video && !model) return; // 无自动加载配置，不打扰手动流程
          if (al.stage && al.stage !== lastStage) { lastStage = al.stage; attempts = 0; } // 阶段有推进：重置预算，慢加载不误判超时
          var slow = ++attempts > MAX; // 超预算不终止：降频续轮询，后端晚就绪仍能接上
          var stageTxt = {
            'idle': '\u7b49\u5f85\u542f\u52a8\u2026',
            'opening-workspace': '\u6253\u5f00 workspace\u2026',
            'loading-model': '\u52a0\u8f7d\u6a21\u578b ' + (al.detail || '') + '\u2026',
            'selecting-video': '\u52a0\u8f7d\u89c6\u9891 ' + (al.detail || '') + '\u2026',
            'error': '\u5931\u8d25: ' + (al.detail || '')
          }[al.stage] || '\u81ea\u52a8\u52a0\u8f7d\u4e2d\u2026';
          if (si) si.textContent = '\u25cf \u81ea\u52a8\u52a0\u8f7d' + (slow ? '\u8d85\u65f6\u2026' : ': ' + stageTxt);
          setTimeout(poll, slow ? 5000 : 1200);
        }).catch(function () {
          var slow = ++attempts > MAX;
          if (si && slow) si.textContent = '\u25cf \u670d\u52a1\u672a\u54cd\u5e94';
          setTimeout(poll, slow ? 5000 : 1500);
        });
      }
      poll();
    },

    _seekFirstVisibleFrame: function () {
      // 片头黑帧检测：autoload 定位到 frame 0 后，若开头是黑屏（JAV 片头带黑场很常见），
      // 逐步探测后续帧直到找到第一个非黑帧再定位过去，避免"预览区一片黑"的观感。
      var self = this;
      var candidates = [30, 60, 120, 240, 480, 960];   // 探测点递增
      var probe = 0;
      function next() {
        if (probe >= candidates.length) return;        // 全黑：停在 frame 0（真全黑片）
        var idx = Math.min(candidates[probe], Math.max(0, self.state.totalFrames - 1));
        probe++;
        var img = new Image();
        img.onload = function () {
          try {
            var c = document.createElement('canvas');
            c.width = 32; c.height = 18;
            var ctx = c.getContext('2d');
            ctx.drawImage(img, 0, 0, 32, 18);
            var d = ctx.getImageData(0, 0, 32, 18).data;
            var sum = 0;
            for (var i = 0; i < d.length; i += 4) sum += (d[i] + d[i+1] + d[i+2]) / 3;
            var mean = sum / (d.length / 4);
            if (mean > 20) {                           // 有画面：定位过去
              self.seekFrame(idx);
              return;
            }
            next();                                   // 还是黑：继续探测
          } catch (e) { /* 异常：放弃，停在当前帧 */ }
        };
        img.onerror = function () { /* 读不到：放弃 */ };
        img.src = '/api/preview/frame/' + idx + '?q=85&_=' + Date.now();
      }
      next();
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

// ── Top-bar layout switches: 双栏/三栏 + 左右/上下 (ui-dark v7) ──
(function() {
  var pa = document.getElementById('preview-area');
  var swCols = document.getElementById('sw-cols');
  var swOr = document.getElementById('sw-orient');
  if (!pa || !swCols || !swOr) return;
  function settle() {
    // v2 owns panel sizing: re-run it so the orientation/column change really applies
    if (window.MSLayoutApply) window.MSLayoutApply();
    setTimeout(function() {
      if (window.MSZoomApply) window.MSZoomApply();
    }, 150);
  }
  function applyCols(on3) {
    pa.classList.remove('layout-3col-swap-big');
    pa.classList.toggle('layout-2col', !on3);
    swCols.classList.toggle('on', on3);
    var lb = swCols.querySelector('.top-switch__label');
    if (lb) lb.textContent = on3 ? '三栏' : '双栏';
    localStorage.setItem('ms_layout', on3 ? '3col' : '2col');
    settle();
  }
  function applyOr(vert) {
    pa.classList.toggle('layout-vertical', vert);
    swOr.classList.toggle('on', vert);
    var lb = swOr.querySelector('.top-switch__label');
    if (lb) lb.textContent = vert ? '上下' : '左右';
    localStorage.setItem('ms_layout_mode', vert ? 'v' : 'h');
    settle();
  }
  applyCols((localStorage.getItem('ms_layout') || '2col') !== '2col');
  applyOr(localStorage.getItem('ms_layout_mode') === 'v');
  swCols.addEventListener('click', function() { applyCols(pa.classList.contains('layout-2col')); });
  swOr.addEventListener('click', function() { applyOr(!pa.classList.contains('layout-vertical')); });
})();


// ── Draggable preview layout (ui-dark v2) ──
(function() {
  var pa = document.getElementById('preview-area');
  if (!pa) return;
  var splitter = document.getElementById('preview-splitter');
  var detP = document.getElementById('preview-detection');
  var swpP = document.getElementById('preview-swapped');
  if (!splitter || !detP || !swpP) return;

  var isDragging = false, startPos = 0, startPct = 50;
  var origP = document.getElementById('preview-original');
  var mode = localStorage.getItem('ms_layout_mode') || 'h';
  var ratio = parseFloat(localStorage.getItem('ms_layout_ratio')) || 50;

  // Single source of truth for panel sizing — also driven by the top-bar switches.
  // Sizes every visible panel along the current axis and leaves room for the 8px splitter.
  function applyLayout() {
    var vert = pa.classList.contains('layout-vertical');
    var twoCol = pa.classList.contains('layout-2col');
    // space NOT available to panels: the 8px splitter plus the 2px flex gaps between
    // neighbouring items (each gap is charged to one side, so the sums fit exactly).
    var GAP_EDGE = 2;                             // 3-col: gap between Original and Detection
    var GAP_SPLIT = 6;                            // half the splitter (4) + one flex gap (2)
    var wOrig = twoCol ? 0 : 100 / 3;             // 3-col: Original keeps one third
    var rest = 100 - wOrig;
    var wDet = rest * ratio / 100, wSwp = rest * (100 - ratio) / 100;

    function size(p, pct, gap) {
      if (!p || !pct) return;
      var v = 'calc(' + pct.toFixed(3) + '% - ' + gap + 'px)';
      if (vert) { p.style.height = v; p.style.width = ''; }
      else { p.style.width = v; p.style.height = ''; }
    }
    [origP, detP, swpP].forEach(function (p) {
      if (!p) return;
      p.classList.add('drag-set');
      p.style.width = '';
      p.style.height = '';
    });
    size(origP, wOrig, GAP_EDGE);
    size(detP, wDet, GAP_SPLIT);
    size(swpP, wSwp, GAP_SPLIT);
    localStorage.setItem('ms_layout_mode', vert ? 'v' : 'h');
    localStorage.setItem('ms_layout_ratio', ratio);
  }
  window.MSLayoutApply = applyLayout;

  splitter.addEventListener('mousedown', function(e) {
    var _v = pa.classList.contains('layout-vertical');
    e.preventDefault(); e.stopPropagation();
    isDragging = true; startPos = _v ? e.clientY : e.clientX;
    startPct = ratio; splitter.classList.add('dragging');
    document.body.style.cursor = _v ? 'row-resize' : 'col-resize';
    document.body.style.userSelect = 'none';
  });
  document.addEventListener('mousemove', function(e) {
    if (!isDragging) return;
    var rect = pa.getBoundingClientRect();
    var _vert = pa.classList.contains('layout-vertical');
    var pct;
    if (!_vert) pct = ((e.clientX - rect.left) / rect.width) * 100;
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

  applyLayout();
})();


// ── Sidebar collapse (ui-dark v3) ──
(function() {
  var btn = document.getElementById('sidebar-toggle');
  var mc = document.getElementById('main-content');
  if (!btn || !mc) return;
  var saved = localStorage.getItem('ms_sidebar') || 'shown';
  var showBtn = document.getElementById('sidebar-show');
  function apply(v) {
    mc.classList.toggle('sidebar-hidden', v === 'hidden');
    localStorage.setItem('ms_sidebar', v);
  }
  window.MSSidebar = apply;
  apply(saved);
  btn.addEventListener('click', function() {
    var cur = localStorage.getItem('ms_sidebar') || 'shown';
    apply(cur === 'shown' ? 'hidden' : 'shown');
  });
  if (showBtn) showBtn.addEventListener('click', function() { apply('shown'); });
})();

// ── Preview zoom (ui-dark v3) ──
(function() {
  var slider = document.getElementById('zoom-slider');
  var label = document.getElementById('zoom-label');
  var canvas = document.getElementById('canvas-swapped');
  var zoomIn = document.getElementById('zoom-in');
  var zoomOut = document.getElementById('zoom-out');
  var zoomReset = document.getElementById('zoom-reset');
  var controls = document.getElementById('zoom-controls');
  if (!slider || !canvas) return;

  var zoom = 100;
  function applyZoom(z) {
    zoom = Math.max(100, Math.min(500, z));
    canvas.style.transform = 'scale(' + (zoom / 100) + ')';
    canvas.style.transformOrigin = 'center center';
    if (label) label.textContent = zoom + '%';
    if (slider) slider.value = zoom;
    localStorage.setItem('ms_zoom', zoom);
  }
  // 交换面板有图时显示缩放控制
  var swpPanel = document.getElementById('preview-swapped');
  if (swpPanel) {
    var obs = new MutationObserver(function() {
      var hasImg = canvas.width > 0;
      controls.style.display = hasImg ? 'flex' : 'none';
    });
    obs.observe(canvas, { attributes: true, attributeFilter: ['width'] });
  }

  if (slider) slider.addEventListener('input', function() { applyZoom(parseInt(this.value)); });
  if (zoomIn) zoomIn.addEventListener('click', function() { applyZoom(zoom + 25); });
  if (zoomOut) zoomOut.addEventListener('click', function() { applyZoom(zoom - 25); });
  if (zoomReset) zoomReset.addEventListener('click', function() { applyZoom(100); });
  applyZoom(parseInt(localStorage.getItem('ms_zoom')) || 100);
})();

// ── Synchronized Preview Zoom/Pan (ui-dark v4) ──
(function() {
  var pa = document.getElementById('preview-area');
  if (!pa) return;
  var panels = [].slice.call(pa.querySelectorAll('.preview-panel')).map(function(el) {
    return { el: el, canvas: el.querySelector('canvas') };
  }).filter(function(p) { return p.canvas; });
  if (panels.length === 0) return;

  var pz = { scale: 1, ox: 0, oy: 0 };
  var isPanning = false, panStart = { mx: 0, my: 0, ox: 0, oy: 0 };
  var pctEl = document.getElementById('pz-pct');
  var mm = document.getElementById('pz-minimap');
  var mmThumb = document.getElementById('pz-mini-thumb');
  var mmBox = document.getElementById('pz-mini-box');
  var swp = null;
  for (var k = 0; k < panels.length; k++) if (panels[k].el.id === 'preview-swapped') swp = panels[k];
  if (!swp) swp = panels[panels.length - 1];

  // 小地图 = 卡片（面板）的等比缩略：画面按它在卡片中的真实位置/大小画进去，
  // 画面外的空白也照实呈现；蓝色小框 = 画面此刻在卡片中占据的区域（自由拖动它）
  var MM_MAX = 120;                     // 缩略图最长边
  var mmThumbKey = null;
  function updateRegion() {
    if (!mm || !swp || !mmBox || !mmThumb) return;
    var fit = swp.canvas._fit;
    if (!swp.canvas.width || !fit || fit.w < 4 || fit.h < 4) {
      mm.style.display = 'none';
      return;
    }
    var rng = panRanges();
    if (!rng.show) { mm.style.display = 'none'; return; }   // 卡片与画面完全重合时无需小地图

    var pw = swp.el.clientWidth, ph = swp.el.clientHeight;
    var ex = swp.canvas.offsetLeft, ey = swp.canvas.offsetTop;
    var s = pz.scale;

    // 卡片比例 = 面板比例（严格对应，绝不拉伸）
    var sc = MM_MAX / Math.max(pw, ph, 1);
    var mmW = Math.max(14, Math.round(pw * sc)), mmH = Math.max(14, Math.round(ph * sc));
    if (mm.style.display !== 'block') mm.style.display = 'block';
    mm.style.width = mmW + 'px';
    mm.style.height = mmH + 'px';

    // 缩略图 = 整块画布（画面 + 空白），仅在帧变化/画布重建时重画
    var key = (swp.canvas._lastImg && swp.canvas._lastImg.src || '') + '|' + swp.canvas.width + 'x' + swp.canvas.height;
    if (key !== mmThumbKey) {
      mmThumbKey = key;
      try {
        mmThumb.width = mmW; mmThumb.height = mmH;
        var tctx = mmThumb.getContext('2d');
        tctx.clearRect(0, 0, mmW, mmH);
        tctx.drawImage(swp.canvas, 0, 0, swp.canvas.width, swp.canvas.height, 0, 0, mmW, mmH);
      } catch (e) { /* 跨域或未初始化时静默 */ }
    }

    // 画面在卡片里的屏幕范围 ∩ 卡片 → 小框（超出卡片的部分=被裁掉，框就贴边）
    var fsx = ex + s * (fit.x + pz.ox), fsy = ey + s * (fit.y + pz.oy);
    var fsw = s * fit.w, fsh = s * fit.h;
    var l = Math.max(0, fsx), t = Math.max(0, fsy);
    var rr = Math.min(pw, fsx + fsw), bb = Math.min(ph, fsy + fsh);
    if (rr - l < 2 || bb - t < 2) { mm.style.display = 'none'; return; }
    mmBox.style.left = (l / pw * mmW) + 'px';
    mmBox.style.top = (t / ph * mmH) + 'px';
    mmBox.style.width = ((rr - l) / pw * mmW) + 'px';
    mmBox.style.height = ((bb - t) / ph * mmH) + 'px';
  }

  function apply() {
    var t = 'scale(' + pz.scale + ') translate(' + pz.ox + 'px,' + pz.oy + 'px)';
    panels.forEach(function(p) {
      // 聚焦人脸模式下每个面板有自己的变换（各自几何独立居中）；
      // 其他情况共享手动缩放/平移状态。
      p.canvas.style.transform = (faceMode && p._faceT) ? p._faceT : t;
      p.canvas.style.transformOrigin = '0 0';
    });
    pa.classList.toggle('pz-panning', isPanning && pz.scale > 1);
    if (pctEl) pctEl.textContent = Math.round(pz.scale * 100) + '%';
    updateRegion();
  }

  function zoomAt(factor, cx, cy) {
    var oldS = pz.scale;
    var newS = Math.max(1, Math.min(8, oldS * factor));
    if (newS === oldS) return;
    clearFaceT();                          // 手动缩放接管，退出本帧的人脸聚焦构图
    var ratio = newS / oldS;
    pz.ox = cx - (cx - pz.ox) * ratio;
    pz.oy = cy - (cy - pz.oy) * ratio;
    pz.scale = newS;
    if (newS === 1 && window.Preview && typeof Preview.refit === 'function') Preview.refit();
    clampPan();
    apply();
  }

  // Auto-fit (re-drawing the canvases at the panel's current size) is allowed only while
  // the view is NOT zoomed in. Once zoomed the framed view is kept until the user hits FIT.
  function refitIfFit() {
    if (pz.scale > 1 && !faceMode) return;      // zoomed manual view is kept as-is
    if (window.Preview && typeof Preview.refit === 'function') Preview.refit();
    if (faceMode) maybeApplyFaceFocus();        // face mode re-centres for the new panel size
  }

  // Per-axis pan limits of the swapped panel.
  //   fit  : the picture's rect inside the panel (element px, unscaled)
  //   mode : 'slide'  — picture smaller than the panel: it may be moved anywhere inside,
  //                     from flush-top/left to flush-bottom/right (free positioning)
  //          'window' — picture larger than the panel: the view window stays over the
  //                     picture (no drifting into empty space)
  // Both modes share the same two bounds, only their order differs, so one formula each.
  function panRanges() {
    var out = { x: { min: 0, max: 0, slide: false }, y: { min: 0, max: 0, slide: false }, show: false };
    if (!swp || !swp.canvas._fit) return out;
    var fit = swp.canvas._fit;
    var s = pz.scale, pw = swp.el.clientWidth, ph = swp.el.clientHeight;
    var ex = swp.canvas.offsetLeft, ey = swp.canvas.offsetTop;
    if (faceMode) {
      // 人脸聚焦模式：以"人脸居中"为先，允许画面外留白，
      // 但至少保证 20% 的窗口仍被画面覆盖（不会把画面推出视野）
      var W = pw / s, H = ph / s;
      var loX = fit.x - 0.8 * W, hiX = fit.x + fit.w - 0.2 * W;
      var loY = fit.y - 0.8 * H, hiY = fit.y + fit.h - 0.2 * H;
      out.x = { min: -(hiX + ex / s), max: -(loX + ex / s), slide: true };
      out.y = { min: -(hiY + ey / s), max: -(loY + ey / s), slide: true };
      out.show = (out.x.max - out.x.min > 1) || (out.y.max - out.y.min > 1);
      return out;
    }
    var aX = -(fit.x + ex / s), bX = (pw - ex) / s - fit.x - fit.w;
    var aY = -(fit.y + ey / s), bY = (ph - ey) / s - fit.y - fit.h;
    if (fit.w * s <= pw) { out.x = { min: aX, max: bX, slide: true }; }
    else { out.x = { min: bX, max: aX, slide: false }; }
    if (fit.h * s <= ph) { out.y = { min: aY, max: bY, slide: true }; }
    else { out.y = { min: bY, max: aY, slide: false }; }
    out.show = (out.x.max - out.x.min > 1) || (out.y.max - out.y.min > 1);
    return out;
  }

  function clampPan() {
    if (!swp || !swp.canvas._fit) return;
    var r = panRanges();
    pz.ox = Math.min(r.x.max, Math.max(r.x.min, pz.ox));
    pz.oy = Math.min(r.y.max, Math.max(r.y.min, pz.oy));
  }

  function reset() {
    // FIT = manual re-enable of auto-fit (the panel may have changed size while zoomed)
    // and it leaves 聚焦人脸 mode (the two are alternative display modes)
    if (window.Preview && typeof Preview.refit === 'function') Preview.refit();
    pz.scale = 1; pz.ox = 0; pz.oy = 0;
    setFaceMode(false);
    apply();
  }

  panels.forEach(function(p) {
    p.el.addEventListener('wheel', function(e) {
      e.preventDefault();
      var rect = p.el.getBoundingClientRect();
      var cx = (e.clientX - rect.left) / pz.scale;
      var cy = (e.clientY - rect.top) / pz.scale;
      zoomAt(e.deltaY < 0 ? 1.15 : 1 / 1.15, cx, cy);
    }, { passive: false });
    p.el.addEventListener('mousedown', function(e) {
      if (pz.scale <= 1) return;
      e.preventDefault(); isPanning = true;
      clearFaceT();                        // 手动拖动接管
      panStart = { mx: e.clientX, my: e.clientY, ox: pz.ox, oy: pz.oy };
      pa.classList.add('pz-panning');
    });
    p.el.addEventListener('dblclick', function(e) { e.preventDefault(); reset(); });
    p.canvas.addEventListener('contextmenu', function(e) { e.preventDefault(); });
  });

  document.addEventListener('mousemove', function(e) {
    if (!isPanning) return;
    pz.ox = panStart.ox + (e.clientX - panStart.mx) / pz.scale;
    pz.oy = panStart.oy + (e.clientY - panStart.my) / pz.scale;
    clampPan(); apply();
  });
  document.addEventListener('mouseup', function() { isPanning = false; });

  // 小地图交互（卡片缩略语义）：
  //   画面比卡片小 → 小框就是画面本身，拖动它=在卡片里自由摆放画面（框随指针走）
  //   画面比卡片大 → 小框是卡片可视窗口，拖动它=平移视野（相机式，框贴边不动）
  // 点小框外任意处 = 把画面该处移到卡片中心
  if (mm) {
    var mmDrag = null, mmBase = null;
    function mmPoint(e) {
      var r = mm.getBoundingClientRect();
      return { px: e.clientX - r.left, py: e.clientY - r.top, w: r.width, h: r.height };
    }
    function viewTopLeft() {
      var s = pz.scale;
      return { x: -pz.ox - swp.canvas.offsetLeft / s, y: -pz.oy - swp.canvas.offsetTop / s };
    }
    function applyFromDrag(p) {
      var s = pz.scale;
      var dxEl = (p.px - mmBase.px) / p.w * swp.el.clientWidth / s;    // 卡片位移 → 元素位移
      var dyEl = (p.py - mmBase.py) / p.h * swp.el.clientHeight / s;
      var rng = panRanges();
      // slide 模式：画面跟着指针走（直接摆放）；window 模式：相机式平移
      // dxEl/dyEl 已是元素单位（= 屏幕位移 / s），ox/oy 与元素位移 1:1，
      // 之前这里又除了一次 s，导致放大后拖动位移只有应有的 1/s
      pz.ox = rng.x.slide ? (mmBase.ox + dxEl) : (mmBase.ox - dxEl);
      pz.oy = rng.y.slide ? (mmBase.oy + dyEl) : (mmBase.oy - dyEl);
      clampPan(); apply();
    }
    mm.addEventListener('mousedown', function(e) {
      if (!swp || !swp.canvas._fit) return;
      e.preventDefault(); e.stopPropagation();
      clearFaceT();                        // 小地图拖动接管
      var p = mmPoint(e), s = pz.scale, cur = viewTopLeft();
      var bl = parseFloat(mmBox.style.left) || 0, bt = parseFloat(mmBox.style.top) || 0;
      var bw = parseFloat(mmBox.style.width) || 0, bh = parseFloat(mmBox.style.height) || 0;
      var insideBox = (p.px >= bl && p.px <= bl + bw && p.py >= bt && p.py <= bt + bh);
      if (!insideBox) {                    // 点小框外：先把该处移到卡片中心
        var dx = (p.px / p.w - 0.5) * swp.el.clientWidth / s;
        var dy = (p.py / p.h - 0.5) * swp.el.clientHeight / s;
        pz.ox = -(cur.x + dx + swp.canvas.offsetLeft / s);
        pz.oy = -(cur.y + dy + swp.canvas.offsetTop / s);
        clampPan(); apply();
        cur = viewTopLeft();
      }
      mmDrag = true;
      mmBase = { px: p.px, py: p.py, x0: cur.x, y0: cur.y, ox: pz.ox, oy: pz.oy };
    });
    document.addEventListener('mousemove', function(e) {
      if (!mmDrag) return;
      applyFromDrag(mmPoint(e));
    });
    document.addEventListener('mouseup', function() { mmDrag = null; });
    var rzT = null;
    if (window.ResizeObserver) {
      var onPanelResize = function() {
        updateRegion();
        if (rzT) clearTimeout(rzT);
        rzT = setTimeout(function() { refitIfFit(); clampPan(); apply(); }, 120);
      };
      panels.forEach(function(p) { new ResizeObserver(onPanelResize).observe(p.el); });
    }
    if (window.MutationObserver && swp) {
      new MutationObserver(function() { updateRegion(); })
        .observe(swp.canvas, { attributes: true, attributeFilter: ['width', 'height', 'style'] });
    }
    // 画布重绘（新帧图/_drawImage）后 _fit 会变：聚焦需重算。
    // 用签名对比防止与 apply() 的 style 写入形成回环。
    var _lastFitSig = null;
    function fitSig() {
      return panels.map(function (p) {
        var f = p.canvas._fit;
        return f ? (p.canvas.width + 'x' + p.canvas.height + ':' +
                    [f.x, f.y, f.w, f.h].join(',')) : '0';
      }).join('|');
    }
    window.MSFaceFocusNoteFit = function () {
      var sig = fitSig();
      if (sig === _lastFitSig) return;
      _lastFitSig = sig;
      if (faceMode && faceFaces && faceFrameSize) focusWhenCanvasReady();
    };
    if (window.MutationObserver) {
      panels.forEach(function (p) {
        new MutationObserver(function () {
          if (window.MSFaceFocusNoteFit) window.MSFaceFocusNoteFit();
        }).observe(p.canvas, { attributes: true, attributeFilter: ['width', 'height'] });
      });
    }
  }

  // ── 聚焦人脸模式：主脸的换脸区域（对齐框）放大到约占卡片 80% 并居中，逐帧自动跟随 ──
  // 后端从 2026-10-09 起在每张脸里附带 align_rect（whole-face 对齐框，帧像素坐标，
  // = 换脸实际覆盖的区域）。实测 200 个样本：对齐框是检测框的 ~2.05x 宽 / ~1.42x 高。
  // 旧逻辑按检测框取 80% 缩放，实际换脸区域被放大到约 1.6x 卡片 → "脸大到显示不下"。
  // 现在以对齐框为基准：它完整可见，且观感尺寸依旧饱满。
  // 旧代码只有一套 ox/oy 被三个尺寸不同的面板共享（swap 面板居中 → detection/original
  // 面板中心被甩出 300px+）。现在每个面板按自身几何独立居中，存在 p._faceT；
  // 手动缩放/拖动/小地图接管时清除，下一帧 analyze 自动恢复聚焦。
  var FACE_FILL = 0.8;                                   // 对齐框占卡片受限边的比例
  // 默认强制开启（用户指定 2026-10-09）：不记忆“关闭”状态，每次打开页面都自动聚焦。
  // FACE 按钮仍可临时关闭看全幅，但刷新/重开页面后恢复默认开启。
  var faceMode = true;
  var faceFaces = null, faceFrameSize = null;
  var faceBtn = document.getElementById('pz-face');

  function clearFaceT() {
    panels.forEach(function (p) { p._faceT = null; });
  }

  // 找主脸（面积最大）的聚焦目标框：优先 align_rect，否则回退检测框
  function faceFocusBox(f) {
    if (f && f.align_rect && f.align_rect.length === 4) {
      var a = f.align_rect;
      var aw = a[2] - a[0], ah = a[3] - a[1];
      if (aw > 8 && ah > 8) return { x: a[0], y: a[1], w: aw, h: ah };
    }
    return f && f.w > 4 && f.h > 4 ? { x: f.x, y: f.y, w: f.w, h: f.h } : null;
  }

  function maybeApplyFaceFocus() {
    if (!faceMode) return false;
    var list = (faceFaces || []).map(faceFocusBox).filter(function (b) { return b; });
    if (!list.length || !faceFrameSize) {       // 无人脸或缺帧尺寸 → 回到整幅
      pz.scale = 1; pz.ox = 0; pz.oy = 0; clearFaceT(); clampPan(); apply();
      return false;
    }
    var fw = faceFrameSize[0], fh = faceFrameSize[1];
    if (!fw || !fh) return false;
    var best = list[0];                       // 面积最大的脸 = 主脸
    for (var i = 1; i < list.length; i++) {
      if (list[i].w * list[i].h > best.w * best.h) best = list[i];
    }
    // 每个面板独立计算：缩放倍数一致（都基于同一帧、同一目标框），
    // 但平移量按各自画布的 fit 与面板尺寸求出，人脸在每个卡片里都居中。
    // 平移同样做旧版的"至少 20% 窗口被画面覆盖"保护（脸在画面边缘时不把卡片整片推出画面）。
    var any = false;
    panels.forEach(function (p) {
      var fit = p.canvas._fit;
      var pw = p.el.clientWidth, ph = p.el.clientHeight;
      if (!fit || !pw || !ph) return;
      var k = fit.w / fw;                     // 帧像素 → 元素像素
      var fdw = best.w * k, fdh = best.h * k;
      // 按卡片的"受限边"取 80%：竖长卡片以宽度为准，横条卡片以高度为准
      var byW = pw / Math.max(1, fdw), byH = ph / Math.max(1, fdh);
      var target = Math.min(8, Math.max(1, FACE_FILL * Math.min(byW, byH)));
      var fcx = fit.x + (best.x + best.w / 2) * k;
      var fcy = fit.y + (best.y + best.h / 2) * k;
      var tx = pw / (2 * target) - fcx, ty = ph / (2 * target) - fcy;
      // 20% 覆盖保护（与 panRanges 的 faceMode 分支同公式）
      var ex = p.canvas.offsetLeft / target, ey = p.canvas.offsetTop / target;
      var W = pw / target, H = ph / target;
      var loX = fit.x - 0.8 * W, hiX = fit.x + fit.w - 0.2 * W;
      var loY = fit.y - 0.8 * H, hiY = fit.y + fit.h - 0.2 * H;
      tx = Math.min(-(loX + ex), Math.max(-(hiX + ex), tx));
      ty = Math.min(-(loY + ey), Math.max(-(hiY + ey), ty));
      p._faceT = 'scale(' + target + ') translate(' + tx + 'px,' + ty + 'px)';
      if (p === swp) {                        // 主面板状态同步给小地图/HUD/手动缩放
        pz.scale = target; pz.ox = tx; pz.oy = ty;
      }
      any = true;
    });
    if (!any) { clearFaceT(); return false; }
    clampPan();                               // 已按同公式预算过，此处通常为空操作
    apply();
    return true;
  }

  function setFaceMode(on) {
    faceMode = !!on;
    try { localStorage.setItem('ms_face_focus', faceMode ? 'on' : 'off'); } catch (e) {}  // 仅诊断用，加载时不再读取
    if (faceBtn) faceBtn.classList.toggle('active', faceMode);
    if (!faceMode) {                         // 关闭聚焦：回到整幅适配
      clearFaceT();
      pz.scale = 1; pz.ox = 0; pz.oy = 0;
      if (window.Preview && typeof Preview.refit === 'function') Preview.refit();
      clampPan(); apply();
      return;
    }
    focusWhenCanvasReady();
  }

  window.MSFaceFocus = {
    setFaces: function (f) { faceFaces = f || []; if (faceMode) focusWhenCanvasReady(); },
    setFrameSize: function (s) { faceFrameSize = s || null; if (faceMode) focusWhenCanvasReady(); },
    enabled: function () { return faceMode; },
    apply: maybeApplyFaceFocus,
    applyAsync: focusWhenCanvasReady,
    info: function () { return { faces: faceFaces, frameSize: faceFrameSize, mode: faceMode }; }
  };
  if (faceBtn) faceBtn.addEventListener('click', function () { setFaceMode(!faceMode); });

  var fitBtn = document.getElementById('pz-fit');
  var zin = document.getElementById('pz-zin');
  var zout = document.getElementById('pz-zout');
  if (fitBtn) fitBtn.addEventListener('click', reset);
  if (zin) zin.addEventListener('click', function() { zoomAt(1.25, pa.clientWidth / 2, pa.clientHeight / 2); });
  if (zout) zout.addEventListener('click', function() { zoomAt(1 / 1.25, pa.clientWidth / 2, pa.clientHeight / 2); });
  setFaceMode(faceMode);                     // 默认开启
  // 首帧场景：analyze 返回时 swapped 画布可能还未绘制（无 _fit），此时聚焦不生效；
  // 画布就绪后由 MutationObserver 触发 updateRegion，但没人重算聚焦。
  // 轮询到 _fit 就绪即补算一次聚焦（也会等 faces 数据到位）。
  var ffTick = 0;
  function focusWhenCanvasReady() {
    if (!faceMode) return;
    if (!faceFaces || !faceFrameSize) return;   // 数据未到，等下个回调
    ffTick = 0;                                // 外部入口重置放弃计数
    pollCanvasReady();
  }
  function pollCanvasReady() {
    if (!faceMode) return;
    // 只等可见面板（双栏布局下 Original 隐藏，clientWidth=0，永远不会有 _fit）
    var vis = panels.filter(function (p) { return p.el.clientWidth > 0 && p.el.clientHeight > 0; });
    if (!vis.length) return;
    var ready = vis.every(function (p) { return p.canvas._fit; });
    if (ready) { ffTick = 0; maybeApplyFaceFocus(); return; }
    if (++ffTick > 200) return;               // ~10s 后放弃
    setTimeout(pollCanvasReady, 50);
  }
  window.MSFaceFocus.applyAsync = focusWhenCanvasReady;
  if (faceMode && faceFaces && faceFrameSize) focusWhenCanvasReady();
  window.MSZoomApply = function() { refitIfFit(); clampPan(); apply(); };
})();

// ── Timeline track collapse toggle (ui-dark v5) ──
(function() {
  var btn = document.getElementById('tl-collapse');
  if (!btn) return;
  var tracks = ['track--faces', 'track--cut', 'track--angle'];
  var saved = localStorage.getItem('ms_tracks') || 'collapsed';
  function apply(v) {
    tracks.forEach(function(t) {
      var el = document.querySelector('.' + t);
      if (el) el.classList.toggle('tl-collapsed', v === 'collapsed');
    });
    btn.textContent = v === 'collapsed' ? '▸ 轨道' : '▾ 轨道';
    localStorage.setItem('ms_tracks', v);
  }
  apply(saved);
  btn.addEventListener('click', function() {
    var cur = localStorage.getItem('ms_tracks') || 'collapsed';
    apply(cur === 'collapsed' ? 'expanded' : 'collapsed');
  });
})();
