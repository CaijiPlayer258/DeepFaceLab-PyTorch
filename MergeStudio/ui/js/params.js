(function () {
  'use strict';

  window.Params = {
    init: function () {
      this.initDetectorDropdowns();
      this.initMergeParams();
      this.initChangeListeners();
    },

    initDetectorDropdowns: function () {
      var detectors = ['BlazeFace', 'CenterFace', 'DamoFD', 'LightweightFD', 'MogFace', 'MTCNN', 'RetinaFace_10g', 'RetinaFace_500m', 'S3FD', 'TinyMog', 'ULFD', 'YOLOv8', 'YoloV5Face', 'YoloV11nFace'];
      var landmarkers = ['insightface-2d106det'];
      var faceTypes = ['whole_face', 'full_face', 'half_face', 'midfull_face', 'head'];
      this.fillSelect('detector-select', detectors);
      this.fillSelect('landmarker-select', landmarkers);
      this.fillSelect('face-type-select', faceTypes);
    },

    fillSelect: function (id, options) {
      var sel = document.getElementById(id);
      if (!sel) return;
      sel.innerHTML = options.map(function (o) {
        return '<option value="' + o + '">' + o + '</option>';
      }).join('');
    },

    initChangeListeners: function () {
      ['detector-select', 'landmarker-select', 'face-type-select'].forEach(function (id) {
        var el = document.getElementById(id);
        if (!el) return;
        el.addEventListener('change', function () {
          var app = window.App;
          if (!app) return;
          app.state.detector = document.getElementById('detector-select').value;
          app.state.landmarker = document.getElementById('landmarker-select').value;
          app.state.config.face_type = document.getElementById('face-type-select').value;
          app.loadFrame(app.state.currentFrame);
        });
      });

      var mm = document.getElementById('merge-mode-btns');
      if (mm) {
        mm.addEventListener('click', function (e) {
          var btn = e.target.closest('.param-btn');
          if (!btn || !btn.dataset.mode) return;
          mm.querySelectorAll('[data-mode]').forEach(function (b) { b.classList.remove('param-btn--active'); });
          btn.classList.add('param-btn--active');
          var app = window.App;
          if (app) { app.state.config.mode = btn.dataset.mode; app.remergeFrame(app.state.currentFrame); }
        });
      }

      // Seg mode buttons
      var segBtns = document.getElementById('seg-mode-btns');
      if (segBtns) {
        segBtns.addEventListener('click', function (e) {
          var btn = e.target.closest('.param-btn');
          if (!btn || !btn.dataset.seg) return;
          segBtns.querySelectorAll('.param-btn').forEach(function (b) { b.classList.remove('param-btn--active'); });
          btn.classList.add('param-btn--active');
          var app = window.App;
          if (app) { app.state.config.seg_mode = btn.dataset.seg; app.remergeFrame(app.state.currentFrame); }
        });
      }
      // Mask mode buttons
      var mmBtns = document.getElementById('mask-mode-btns');
      if (mmBtns) {
        mmBtns.addEventListener('click', function (e) {
          var btn = e.target.closest('.param-btn');
          if (!btn || !btn.dataset.mask) return;
          mmBtns.querySelectorAll('.param-btn').forEach(function (b) { b.classList.remove('param-btn--active'); });
          btn.classList.add('param-btn--active');
          var app = window.App;
          if (app) { app.state.config.mask_mode = parseInt(btn.dataset.mask); app.remergeFrame(app.state.currentFrame); }
        });
      }

      var ct = document.getElementById('color-transfer-btns');
      if (ct) {
        ct.addEventListener('click', function (e) {
          var btn = e.target.closest('.param-btn');
          if (!btn || !btn.dataset.ct) return;
          ct.querySelectorAll('.param-btn').forEach(function (b) { b.classList.remove('param-btn--active'); });
          btn.classList.add('param-btn--active');
          var app = window.App;
          if (app) { app.state.config.color_transfer_mode = btn.dataset.ct; app.remergeFrame(app.state.currentFrame); }
        });
      }

      // Detection mode dropdown
      var dm = document.getElementById('detect-mode-select');
      if (dm) {
        // Sync initial value from state
        if (window.App) dm.value = window.App.state.config.detect_mode || 'skip_dfl';
        dm.addEventListener('change', function () {
          var app = window.App;
          if (app) { app.state.config.detect_mode = this.value; app.loadFrame(app.state.currentFrame); }
        });
      }

      // Face margin slider: sync initial value from config (0.0-2.0 → 0-200)
      var fmSlider = document.querySelector('.param-slider[data-param="face_margin"]');
      if (fmSlider) {
        var fmInp = fmSlider.querySelector('.param-slider__input');
        if (fmInp && window.App) {
          fmInp.value = Math.round(window.App.state.config.face_margin * 100);
        }
      }

      // Debug toggle
      setTimeout(function () {
        var dbgBtn = document.getElementById('btn-toggle-debug');
        if (dbgBtn) {
          dbgBtn.addEventListener('click', function () {
            var isOn = this.dataset.debug === '1';
            this.dataset.debug = isOn ? '0' : '1';
            this.classList.toggle('param-btn--active', !isOn);
            var app = window.App;
            if (app) {
              app.state.config.show_debug = !isOn;
              app.remergeFrame(app.state.currentFrame);
            }
          });
        }
      }, 0);
    },

    initMergeParams: function () {
      var modes = ['overlay', 'hist-match', 'seamless', 'raw-rgb'];
      var btnsContainer = document.getElementById('merge-mode-btns');
      if (btnsContainer) {
        btnsContainer.innerHTML = modes.map(function (m) {
          return '<button class="param-btn ' + (m === 'overlay' ? 'param-btn--active' : '') + '" data-mode="' + m + '">' + m + '</button>';
        }).join('') + '<button class="param-btn" id="btn-toggle-debug" data-debug="0">Debug</button>';
      }

      var container = document.getElementById('merge-params');
      if (!container) return;
      container.innerHTML =
        '<div style="margin-bottom:8px;">' +
        '  <div style="font-size:10px;color:#999;margin-bottom:4px;">Seg mode</div>' +
        '  <div class="param-btn-group" id="seg-mode-btns">' +
        '    <button class="param-btn param-btn--active" data-seg="model">DFM</button>' +
        '    <button class="param-btn" data-seg="xseg">XSeg</button>' +
        '    <button class="param-btn" data-seg="xseglite">XSegLite</button>' +
        '  </div>' +
        '</div>' +
        '<div style="margin-bottom:8px;">' +
        '  <div style="font-size:10px;color:#999;margin-bottom:4px;">Mask mode</div>' +
        '  <div class="param-btn-group" id="mask-mode-btns">' +
        '    <button class="param-btn param-btn--active" data-mask="4">prd×dst</button>' +
        '    <button class="param-btn" data-mask="2">prd</button>' +
        '    <button class="param-btn" data-mask="1">dst</button>' +
        '  </div>' +
        '</div>' +
        this.sliderHTML('erode_mask_modifier', 'Erode mask', -100, 100, 0, true) +
        this.sliderHTML('blur_mask_modifier', 'Blur mask', -100, 100, 4, true) +
        '<div style="margin-top:4px;">' +
        '  <div style="font-size:10px;color:#999;margin-bottom:3px;">Color transfer</div>' +
        '  <div class="param-btn-group" id="color-transfer-btns">' +
        ['rct', 'mt', 'lut', 'lct', 'mkl', 'idt', 'sot-m', 'mix-m', 'none'].map(function (c) {
          return '<button class="param-btn ' + (c === 'rct' ? 'param-btn--active' : '') + '" data-ct="' + c + '">' + c + '</button>';
        }).join('') +
        '  </div>' +
        '</div>' +
        this.sliderHTML('output_face_scale', 'Face scale', -50, 50, 0, true) +
        this.sliderHTML('super_resolution_power', 'Super res', 0, 100, 0) +
        '<div style="margin-top:4px;">' +
        '  <div style="display:flex;gap:4px;margin-bottom:2px;">' +
        ['off', 'box', 'gaussian'].map(function (s) {
          return '<button class="param-btn ' + (s === 'off' ? 'param-btn--active' : '') + '" data-sharpen="' + s + '">' + s + '</button>';
        }).join('') +
        '  </div>' +
        '</div>' +
        this.sliderHTML('motion_blur_power', 'Motion blur', 0, 100, 0) +
        this.sliderHTML('image_denoise_power', 'Denoise', 0, 500, 0) +
        this.sliderHTML('bicubic_degrade_power', 'Bicubic deg', 0, 100, 0);
    },

    sliderHTML: function (param, label, min, max, val, bipolar) {
      var pct = max > min ? ((val - min) / (max - min)) * 100 : 0;
      if (bipolar) {
        // Bipolar: 0 = center 50%, negative goes left, positive goes right
        var range = max - min;
        var zeroAt = (0 - min) / range * 100;  // percentage where value 0 sits
        var thumbPct = (val - min) / range * 100;
        return '<div class="param-slider" data-param="' + param + '" data-bipolar="1" data-min="' + min + '" data-max="' + max + '">' +
          '  <div class="param-slider__header">' +
          '    <span class="param-slider__label">' + label + '</span>' +
          '    <input class="param-slider__input" type="text" value="' + val + '" style="width:' + Math.max(28, String(val).length * 8 + 8) + 'px;min-width:24px;">' +
          '  </div>' +
          '  <div class="param-slider__track">' +
          '    <div class="param-slider__fill ' + (val > 0 ? 'param-slider__fill--pos' : 'param-slider__fill--neg') + '" style="' + (val > 0 ? 'left:' + zeroAt + '%;width:' + (thumbPct - zeroAt) + '%' : 'right:' + (100 - zeroAt) + '%;width:' + (zeroAt - thumbPct) + '%') + '"></div>' +
          '    <div class="param-slider__thumb" style="left:' + thumbPct + '%"></div>' +
          '  </div>' +
          '</div>';
      }
      return '<div class="param-slider" data-param="' + param + '">' +
        '  <div class="param-slider__header">' +
        '    <span class="param-slider__label">' + label + '</span>' +
        '    <input class="param-slider__input" type="text" value="' + val + '" style="width:' + Math.max(28, String(val).length * 8 + 8) + 'px;min-width:24px;">' +
        '  </div>' +
        '  <div class="param-slider__track">' +
        '    <div class="param-slider__fill" style="width:' + pct + '%"></div>' +
        '    <div class="param-slider__thumb" style="left:' + pct + '%"></div>' +
        '  </div>' +
        '</div>';
    },

    refreshUI: function () {
      var app = window.App;
      if (!app) return;
      var cfg = app.state.config;
      // Dropdowns
      var detSel = document.getElementById('detector-select');
      if (detSel) detSel.value = app.state.detector;
      var lmSel = document.getElementById('landmarker-select');
      if (lmSel) lmSel.value = app.state.landmarker;
      var ftSel = document.getElementById('face-type-select');
      if (ftSel) ftSel.value = cfg.face_type || 'whole_face';
      var dmSel = document.getElementById('detect-mode-select');
      if (dmSel) dmSel.value = cfg.detect_mode || 'skip_dfl';
      // Sliders
      document.querySelectorAll('.param-slider').forEach(function (sl) {
        var param = sl.dataset.param;
        var inp = sl.querySelector('.param-slider__input');
        if (!param || !inp) return;
        var raw = cfg[param];
        if (raw === undefined) return;
        inp.value = param === 'face_margin' ? Math.round(raw * 100) : raw;
      });
      // Param buttons (mode, seg, mask, color transfer)
      document.querySelectorAll('[data-mode]').forEach(function (b) {
        b.classList.toggle('param-btn--active', b.dataset.mode === cfg.mode);
      });
      document.querySelectorAll('[data-seg]').forEach(function (b) {
        b.classList.toggle('param-btn--active', b.dataset.seg === cfg.seg_mode);
      });
      document.querySelectorAll('[data-mask]').forEach(function (b) {
        b.classList.toggle('param-btn--active', parseInt(b.dataset.mask) === cfg.mask_mode);
      });
      document.querySelectorAll('[data-ct]').forEach(function (b) {
        var ct = cfg.color_transfer_mode;
        if (typeof ct === 'number') ct = {0:'none',1:'rct',2:'lct',3:'mkl',4:'idt',5:'sot-m',6:'mix-m',9:'mt',10:'lut'}[ct] || ct;
        b.classList.toggle('param-btn--active', b.dataset.ct === ct);
      });
      // Debug button
      var dbgBtn = document.getElementById('btn-toggle-debug');
      if (dbgBtn) {
        var isDebug = !!cfg.show_debug;
        dbgBtn.dataset.debug = isDebug ? '1' : '0';
        dbgBtn.classList.toggle('param-btn--active', isDebug);
      }
    },
  };
})();
