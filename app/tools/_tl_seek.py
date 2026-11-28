
# -*- coding: utf-8 -*-
import sys, os, json, time
sys.path.insert(0, r"D:\MyProject\DeepFaceLab-PyTorch-4.1\app\tools\accept")
import merge_accept as MA
OUT = r"D:\MyProject\DeepFaceLab-PyTorch-4.1\app\tools\_tl_seek_probe.json"
VIDEO = r"D:\MyProject\DeepFaceLab-PyTorch-4.1\workspace\9月29日 (5).mp4"

SETUP = r"""
(function(){
  var old = document.getElementById('tv'); if (old) old.remove();
  var v = document.createElement('video');
  v.id = 'tv'; v.muted = true; v.preload = 'auto';
  v.style.cssText = 'position:fixed;left:0;top:0;width:160px;height:90px;z-index:9999';
  document.body.appendChild(v);
  window.__ev = [];
  ['loadstart','loadedmetadata','durationchange','seeking','seeked','timeupdate','waiting','stalled','suspend','error','canplay','progress','emptied'].forEach(function(k){
    v.addEventListener(k, function(){ window.__ev.push({k:k, ct: v.currentTime, rs: v.readyState}); });
  });
  v.src = '/api/merge/video?path=' + encodeURIComponent(VIDEO);
  return 1;
})()
""".replace("VIDEO", json.dumps(VIDEO.replace("\\","/")))

def main():
    edge = MA.EDGE_CANDIDATES[1] if os.path.exists(MA.EDGE_CANDIDATES[1]) else MA.EDGE_CANDIDATES[2]
    b = MA.Browser(edge)
    res = {}
    try:
        b.goto("http://127.0.0.1:8760/")
        time.sleep(1.0)
        b.eval(SETUP)
        for _ in range(60):
            rs = b.eval("document.getElementById('tv').readyState")
            if rs and rs >= 1: break
            time.sleep(0.3)
        res["dur"] = b.eval("document.getElementById('tv').duration")
        res["frames_over_fps"] = 1439/30.0
        res["dur_x_fps"] = b.eval("document.getElementById('tv').duration * 30")
        tests = []
        def step(label, t):
            b.eval("window.__ev.length = 0; 1")
            b.eval("(function(){var v=document.getElementById('tv'); v.currentTime = %r; return v.currentTime;})()" % t)
            time.sleep(1.6)
            ev = json.loads(b.eval("JSON.stringify(window.__ev)"))
            ct = b.eval("document.getElementById('tv').currentTime")
            rs = b.eval("document.getElementById('tv').readyState")
            tests.append({"label": label, "set": t, "ct": ct, "x30": ct*30, "round30": round(ct*30),
                          "readyState": rs,
                          "events": [e["k"] for e in ev],
                          "seeking": sum(1 for e in ev if e["k"]=="seeking"),
                          "seeked": sum(1 for e in ev if e["k"]=="seeked")})
        step("t=17.3333 (frame 520)", 520/30.0)
        step("same t again", 520/30.0)
        step("same t again #2", 520/30.0)
        step("t=40 (frame 1200)", 40.0)
        step("same 40 again", 40.0)
        step("t=333.3333 (frame 10000, beyond dur)", 10000/30.0)
        step("clamped exactly to ct", None) if False else None
        ct_now = b.eval("document.getElementById('tv').currentTime")
        step("same ct again (really identical)", ct_now)
        res["tests"] = tests
    finally:
        b.close()
    open(OUT, "w", encoding="utf-8").write(json.dumps(res, ensure_ascii=False, indent=1))
    print("WROTE")
main()
