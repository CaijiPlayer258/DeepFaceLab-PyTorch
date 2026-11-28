
# -*- coding: utf-8 -*-
import sys, os, json, time
sys.path.insert(0, r"D:\MyProject\DeepFaceLab-PyTorch-4.1\app\tools\accept")
import merge_accept as MA

OUT = r"D:\MyProject\DeepFaceLab-PyTorch-4.1\app\tools\_tl_probe_out3.json"
VIDEO = os.environ.get("TL_VIDEO", r"D:\MyProject\DeepFaceLab-PyTorch-4.1\workspace\9月29日 (5).mp4")

INSTR = r"""
(function(){
  window.__log = [];
  window.__t0 = Math.round(performance.now());
  window.__L = function(tag, extra){
    try { window.__log.push(Object.assign({tp: Math.round(performance.now()) - window.__t0, tag: tag}, extra || {})); } catch(e){}
    if (window.__log.length > 800) window.__log.splice(0, 300);
  };
  var done = [];
  function wrap(name){
    var f = window[name];
    if (typeof f !== 'function') return;
    if (f.__wrapped) { done.push(name + ':already'); return; }
    var w = function(){
      var a = Array.prototype.slice.call(arguments);
      var keep = a.filter(function(x){ return !(x && typeof x === 'object'); }).map(String);
      var o = a.find(function(x){ return x && typeof x === 'object'; });
      __L(name, {a: keep, why: (o && o.why) || '', f: (o && o.frame !== undefined) ? o.frame : ''});
      return f.apply(this, arguments);
    };
    w.__wrapped = true;
    window[name] = w;
    done.push(name + ':ok');
  }
  ['mgFrameChange','mgVideoSeek','mgPreviewSoon','mgPreviewNow','mgCmpSoon','mgCmpNow','mvDetect','mgSyncTimeline','mgComposeFrame'].forEach(wrap);
  var of = window.fetch;
  if (of && !of.__wrapped) {
    var nf = function(u, o){
      var url = (typeof u === 'string') ? u : (u && u.url);
      var body = (o && o.body) ? String(o.body) : '';
      var mf = body.match(/"frame":(\d+)/);
      var mt = body.match(/"t":([0-9.]+)/);
      __L('fetch', {u: String(url).replace('http://127.0.0.1:8760','').slice(0,48), f: mf ? mf[1] : '', t: mt ? mt[1] : ''});
      var p = of.apply(this, arguments);
      p.then(function(r){ __L('done', {u: String(url).replace('http://127.0.0.1:8760','').slice(0,32), st: r.status, f: mf ? mf[1] : '', t: mt ? mt[1] : ''}); },
             function(e){ __L('err', {u: String(url).replace('http://127.0.0.1:8760','').slice(0,32), e: String((e && e.name) || e), f: mf ? mf[1] : ''}); });
      return p;
    };
    nf.__wrapped = true;
    window.fetch = nf;
    done.push('fetch:ok');
  }
  var v = document.getElementById('mgVideo');
  if (v && !v.__probed) {
    v.__probed = 1;
    ['loadedmetadata','durationchange','seeking','seeked','play','pause','ended','error','stalled','waiting','abort','emptied','suspend'].forEach(function(ev){
      v.addEventListener(ev, function(){ __L('v:' + ev, {ct: Math.round(v.currentTime*1000)/1000, dur: Math.round((v.duration||0)*1000)/1000, rs: v.readyState, ns: v.networkState, paused: v.paused}); });
    });
  }
  var tr = document.getElementById('mergeTracks');
  if (tr && !tr.__probed2) { tr.__probed2 = 1; tr.addEventListener('mtl:frame', function(e){ __L('mtl:frame', {n: e.detail && e.detail.frame}); }); }
  window.__wrapDone = done;
  return JSON.stringify(done);
})()
"""

def snap(b):
    return json.loads(b.eval(r"""
    (function(){
      var v = document.getElementById('mgVideo');
      var p = document.getElementById('mgPreview');
      var tl = window.MergeTimeline;
      return JSON.stringify({
        tlFrame: tl ? tl.getFrame() : null,
        ct: v ? Math.round(v.currentTime*1000)/1000 : null,
        ctFrame: v ? Math.round(v.currentTime * ((window.__mv||{}).fps||30)) : null,
        dur: v ? v.duration : null,
        rs: v ? v.readyState : null, ns: v ? v.networkState : null, paused: v ? v.paused : null,
        err: v && v.error ? {code: v.error.code, msg: String(v.error.message).slice(0,80)} : null,
        srcLen: v ? String(v.getAttribute('data-src')||'').length : 0,
        mv: {frames: (window.__mv||{}).frames, fps: (window.__mv||{}).fps, t: Math.round(((window.__mv||{}).t||0)*1000)/1000},
        ovl: p ? {disp: p.style.display, len: (p.getAttribute('src')||'').length} : null,
        gv: {streamDown: !!((window.__mgVideo||{}).streamDown), playing: !!((window.__mgVideo||{}).playing), pngLen: ((window.__mgVideo||{}).png||'').length, wait: !!((window.__mgVideo||{}).wait)},
        gate: (typeof MG_CHAIN !== 'undefined') ? null : null,
        bound: (document.getElementById('mergeTracks')||{}).dataset ? document.getElementById('mergeTracks').dataset.frameBound : null,
        hint: (document.getElementById('mgStageHint')||{}).textContent,
        note: (function(){ var e = document.getElementById('mgCmpNote'); return e ? (!!e.hidden ? '' : e.textContent) : null; })()
      });
    })()
    """))

def main():
    edge = MA.EDGE_CANDIDATES[1] if os.path.exists(MA.EDGE_CANDIDATES[1]) else MA.EDGE_CANDIDATES[2]
    b = MA.Browser(edge)
    res = {"steps": []}
    try:
        res["goto"] = b.goto(os.environ.get("TL_URL", "http://127.0.0.1:8760/"))
        for _ in range(160):
            try:
                d = json.loads(b.eval("JSON.stringify({mv: typeof window.mvOpen, tl: !!window.MergeTimeline, v: !!document.getElementById('mgVideo'), bu: typeof window.bindMergeUI, ready: document.readyState})"))
                if d["mv"] == "function" and d["tl"] and d["v"] and d["bu"] == "function":
                    break
            except Exception:
                pass
            time.sleep(0.25)
        res["app"] = d
        # 让合成页真的「在」：加 on 类 + 绑 UI（真实路径是 00-core.js:515 切页时调的）
        b.eval("(function(){var p=document.getElementById('p-merge'); if(p&&p.classList) p.classList.add('on'); window.bindMergeUI(); return 1;})()")
        time.sleep(0.6)
        res["instr"] = b.eval(INSTR)
        b.eval("window.mvOpen(%s)" % json.dumps(VIDEO.replace("\\", "/")))
        st = None
        for _ in range(240):
            st = snap(b)
            if st["mv"]["frames"] and st["dur"]:
                break
            time.sleep(0.25)
        res["afterOpen"] = st
        time.sleep(2.0)
        res["afterOpen2"] = snap(b)
        frames = [int(x) for x in os.environ.get("TL_FRAMES", "0,120,300,480,500,505,600,900,1200,1300").split(",")]
        for n in frames:
            b.eval("window.__log.length = 0; window.__t0 = Math.round(performance.now()); 1")
            b.eval("window.MergeTimeline.setFrame(%d)" % n)
            time.sleep(4.0)
            st = snap(b)
            log = json.loads(b.eval("JSON.stringify(window.__log)"))
            tags = {}
            for e in log:
                tags[e["tag"]] = tags.get(e["tag"], 0) + 1
            res["steps"].append({"req": n, "snap": st, "tags": tags, "log": log[:150]})
    finally:
        try: res["console"] = (b.console or [])[-12:]
        except Exception: res["console"] = []
        b.close()
    open(OUT, "w", encoding="utf-8").write(json.dumps(res, ensure_ascii=False, indent=1))
    print("WROTE", OUT)

main()