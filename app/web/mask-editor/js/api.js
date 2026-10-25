/**
 * MaskProcessor — API Client
 * Matches the actual backend routes in MaskProcessor/api/
 */
(function () {
  'use strict';

  function getBase() {
    if (window.__API_BASE) return window.__API_BASE;
    var meta = document.querySelector('meta[name="api-base"]');
    return meta ? meta.getAttribute('content') : '';
  }

  var BASE = getBase();

  function request(method, path, body) {
    var url = BASE + '/api' + path;
    var opts = {
      method: method,
      headers: { 'Accept': 'application/json' },
    };
    if (body !== undefined && body !== null) {
      opts.headers['Content-Type'] = 'application/json';
      opts.body = JSON.stringify(body);
    }
    return fetch(url, opts).then(function (res) {
      return res.json().then(function (data) {
        if (!res.ok) {
          var msg = (data && data.detail) || (data && data.error) || res.statusText || 'Request failed';
          var err = new Error(msg);
          err.status = res.status;
          err.data = data;
          throw err;
        }
        return data;
      });
    });
  }

  function get(path) { return request('GET', path); }
  function post(path, body) { return request('POST', path, body); }

  window.API = {

    // ---- Preload -----------------------------------------------------------
    preloadModels: function () {
      return post('/model/preload', {});
    },

    // ---- Project -----------------------------------------------------------
    openProject: function (path) {
      return post('/project/open', { path: path });
    },

    // ---- Image navigation --------------------------------------------------
    loadImage: function (index) {
      return get('/image/' + index);
    },

    // ---- SAM mask prediction -----------------------------------------------
    predictMask: function (index, clicks) {
      return post('/mask/predict', { image_index: index, clicks: clicks });
    },

    predictWithBox: function (index, box) {
      return post('/mask/predict', { image_index: index, box: box });
    },

    // ---- Text-to-mask (GroundedSAM2 / OWL-ViT) -----------------------------
    textToMask: function (index, text, backend) {
      if (backend === undefined) backend = 'grounded_sam2';
      return post('/mask/text', { image_index: index, text: text, backend: backend });
    },

    // ---- Face parsing (BiSeNet) --------------------------------------------
    parseFace: function (index, parts) {
      return post('/mask/bisenet', { image_index: index, parts: parts });
    },

    // ---- XSeg / XSegLite ---------------------------------------------------
    xsegPredict: function (index) {
      return post('/xseg/predict', { image_index: index });
    },
    xseglitePredict: function (index) {
      return post('/xseglite/predict', { image_index: index });
    },

    // ---- Load existing mask from DFLJPG ------------------------------------
    loadExistingMask: function (index) {
      return get('/image/' + index + '/mask');
    },

    // ---- Mask commit / save / undo -----------------------------------------
    commitMask: function (index, fgB64, bgB64) {
      return post('/mask/commit', { image_index: index, foreground: fgB64, background: bgB64 || '' });
    },
    saveMask: function (index, maskB64) {
      return post('/mask/save', { image_index: index, mask: maskB64 });
    },

    undoMask: function (index) {
      return post('/mask/undo', { image_index: index });
    },

    // ---- Progress persistence ------------------------------------------------
    getProgress: function () {
      return get('/progress/load');
    },
    setProgress: function (index) {
      return post('/progress/save', { index: index });
    },
  };

})();
