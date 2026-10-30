(function () {
  'use strict';

  window.FaceEmbedder = {
    computing: false,

    init: function () {
      var btn = document.getElementById('btn-compute-embeddings');
      if (btn) {
        btn.addEventListener('click', function () {
          window.FaceEmbedder.computeAll();
        });
      }
    },

    computeAll: function () {
      if (this.computing) return;
      this.computing = true;
      var btn = document.getElementById('btn-compute-embeddings');
      if (btn) { btn.disabled = true; btn.textContent = 'Computing...'; }

      var self = this;
      var faces = [];
      Object.keys(window.App.state.faceDatabase).forEach(function (key) {
        var parts = key.split('_');
        faces.push({
          key: key,
          frame_idx: parseInt(parts[0]),
          face_idx: parseInt(parts[1] || 0),
          thumb_url: '/api/preview/face-thumb/' + parts[0] + '/' + (parts[1] || 0),
        });
      });

      API.computeEmbeddings({faces: faces}).then(function (data) {
        self.computing = false;
        if (btn) { btn.disabled = false; btn.textContent = 'Recompute Embeddings'; }
        if (data && data.embeddings) {
          window.App.state.faceEmbeddings = data.embeddings;
          window.App.state.faceClusters = data.clusters;
          if (window.ExportFlow && window.ExportFlow.renderFaceDB) {
            window.ExportFlow.renderFaceDB();
          }
        }
      }).catch(function () {
        self.computing = false;
        if (btn) { btn.disabled = false; btn.textContent = 'Compute Failed'; }
      });
    },
  };

  document.addEventListener('DOMContentLoaded', function () {
    window.FaceEmbedder.init();
  });
})();
