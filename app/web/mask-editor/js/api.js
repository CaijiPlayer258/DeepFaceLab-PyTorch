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
      /* 出错时后端不一定回 JSON（404 空体 / 501 HTML），先读文本再试解析，
         保证 err.status 一定带得出来 —— 前端要靠 404/501 判断 Sam 包装没装 */
      return res.text().then(function (text) {
        var data = null;
        if (text) {
          try { data = JSON.parse(text); } catch (_) { data = null; }
        }
        if (!res.ok) {
          var raw = data && (data.detail || data.error || data.message);
          if (typeof raw !== 'string') raw = res.statusText || ('HTTP ' + res.status);
          var err = new Error(raw);
          err.status = res.status;
          err.data = data;
          err.code = data && data.code;
          err.detail = data && data.detail;
          throw err;
        }
        return data === null ? {} : data;
      });
    });
  }

  function errorCode (err) {
    return (err && (err.code || (err.data && err.data.code))) || "";
  }

  /** 判断「这个能力不存在」，用于把 SAM 工具降级。
      优先看后端 code；没有 code 才回落到旧文案，并把盘符路径片段剔除后再匹配。 */
  function isUnsupported (err) {
    if (!err) return false;
    var code = errorCode(err);
    if (code) return code === "missing_weights";
    if (err.status === 501) return true;
    if (err.status !== 404) return false;
    var d = String(err.detail || err.message || "");
    // D:\weights\foo.png 这类路径只是图片名，不能把路径里的 weights 当缺权重
    d = d.replace(/[A-Za-z]:[\\/][^\s"']*/g, " ");
    return /模型|权重|weights|sam 包|未安装|not installed/i.test(d);
  }

  /** 409，或 code=session_expired / 文案明确说会话过期：重新 load 后再试一次。 */
  function isSessionExpired (err) {
    if (!err) return false;
    var code = errorCode(err);
    if (code) return code === "session_expired";
    if (err.status === 409) return true;
    if (err.status !== 404) return false;
    var d = String(err.detail || err.message || "");
    return /(session|会话)/i.test(d) && /(expired|过期|不存在|not found)/i.test(d);
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

    // ---- SAM 点击分割（POST /api/mask/sam/load | /sam/predict）-------------
    /** 为一张图建会话：{ ok, session, w, h }；同一张图后端有缓存 */
    samLoad: function (image, variant) {
      return post('/mask/sam/load', {
        image: image,
        variant: variant || 'base_plus'
      });
    },

    /**
     * 点集推理：正样本 label=1，负样本 label=0。
     * box 可选（GROUNDED SAM2 检出的框当提示，等同框选）。
     */
    samPredict: function (session, points, box) {
      var body = { session: session, points: points || [] };
      if (box && box.length >= 4) body.box = box;
      return post('/mask/sam/predict', body);
    },

    /** 文本检测：{ ok, boxes: [[x1,y1,x2,y2]], scores } */
    gdino: function (image, text) {
      return post('/mask/gdino', { image: image, text: text || 'face.' });
    },

    /** true = 后端没有这个能力（404/501），调用方该走降级 */
    isUnsupported: isUnsupported,
    /** true = 会话被 LRU 淘汰 / 过期，调用方清会话后重新 load 再试一次 */
    isSessionExpired: isSessionExpired,

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
      return post('/mask/save', { index: index, image_index: index, mask: maskB64 });
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
