(function () {
  'use strict';

  const BASE = '/api';

  window.API = {
    async openProject(path) {
      return (await fetch(BASE + '/project/open', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ path }),
      })).json();
    },

    async selectVideo(path) {
      return (await fetch(BASE + '/select-video', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ path }),
      })).json();
    },

    async getFrame(idx, quality, path) {
      var url = BASE + '/preview/frame/' + idx;
      var params = [];
      if (quality) params.push('q=' + quality);
      if (path) params.push('path=' + encodeURIComponent(path));
      if (params.length) url += '?' + params.join('&');
      return await fetch(url);
    },

    async analyzeFrame(data, opts) {
      var fetchOpts = {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(data),
      };
      if (opts && opts.signal) fetchOpts.signal = opts.signal;
      return (await fetch(BASE + '/preview/analyze', fetchOpts)).json();
    },

    async remergeFrame(data) {
      return (await fetch(BASE + '/preview/remerge', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(data),
      })).json();
    },

    async recomposite(data) {
      return (await fetch(BASE + '/preview/recomposite', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(data),
      })).json();
    },

    async getTimelineFaces() {
      return (await fetch(BASE + '/timeline/faces')).json();
    },

    async scanFaces() {
      return (await fetch(BASE + '/timeline/scan-faces')).json();
    },

    async updateCut(data) {
      return (await fetch(BASE + '/timeline/cut', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(data),
      })).json();
    },

    async startExport(data) {
      return (await fetch(BASE + '/export/start', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(data),
      })).json();
    },

    async getExportProgress(id) {
      return (await fetch(BASE + '/export/progress/' + id + '?_=' + Date.now())).json();
    },

    async cancelExport(id) {
      return (await fetch(BASE + '/export/cancel/' + id, { method: 'POST' })).json();
    },

    async getModels() {
      return (await fetch(BASE + '/models')).json();
    },

    async loadModel(name) {
      return (await fetch(BASE + '/models/load', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ name }),
      })).json();
    },

    async getCacheStatus() {
      return (await fetch(BASE + '/preview/cache-status')).json();
    },

    async computeEmbeddings(data) {
      return (await fetch(BASE + '/face/compute-embeddings', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(data),
      })).json();
    },

    async loadConfig() {
      try {
        return (await fetch(BASE + '/preview/load-config')).json();
      } catch (e) {
        return {};
      }
    },

    async saveConfig(data) {
      await fetch(BASE + '/preview/save-config', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(data),
      });
    },
  };
})();
