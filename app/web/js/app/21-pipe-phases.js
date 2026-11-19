'use strict';
/* 分趟导出只需要一件事：把后端自报的 mode / mode_why 写进副标题。
   图本身由 16-pipe-draw 的经典渲染器画（六列 + t0..t7 + e0..e2 + 成片杯），
   这里绝不另画一套 —— 之前几轮反复「换个样子」已经被否掉了。 */
(function () {
  function syncSub(d) {
    var sub = document.getElementById('mgPipeSub');
    if (!sub || !d || !d.mode) return;
    if (d.mode === 'passes') sub.textContent = '分趟 · ' + (d.mode_why || '') + ' · 闸门后才合成';
    else sub.textContent = '快速导出 · ' + (d.mode_why || '');
  }
  window.mgPipeSubSync = syncSub;
  var last = '';
  function tick() {
    var svg = document.getElementById('mgPipeSvg');
    if (!svg || document.visibilityState === 'hidden') return;
    fetch('/api/merge/export/status').then(function (r) { return r.json(); }).then(function (d) {
      var sig = [d.mode, d.mode_why].join('|');
      if (sig === last) return;
      last = sig;
      syncSub(d);
    }).catch(function () {});
  }
  setInterval(tick, 1000);
  tick();
})();
