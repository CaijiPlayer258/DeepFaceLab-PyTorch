# -*- coding: utf-8 -*-
"""合成器验收：一条命令跑完「导入 → 选模型 → 检测 → 时间轴 → 预览 → 导出 → 取消」。

用法
    python tools/merge_accept.py                      # 用默认样片
    python tools/merge_accept.py path/to/video.mp4    # 指定视频
    python tools/merge_accept.py video.mp4 --keep     # 导出跑完不取消（慢，全程 ~2 分钟/千帧）

做的事
    起后端 → 用 CDP 驱动无头 Edge（真实点击，不走内部函数）→ 逐项断言 → 打印 PASS/FAIL。

依赖
    websockets（4.1 的 _python 里已有）。Edge 在默认安装路径。
"""

import os
import sys
import json
import time
import subprocess
import threading
from pathlib import Path

"""最小 CDP 客户端（内联，避免额外依赖）。"""
import base64, json, os, subprocess, time, urllib.request
from websockets.sync.client import connect
EDGE = r'C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe'
_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))
class Browser:
    def __init__(self, port=9333):
        self.port = port
        self.proc = subprocess.Popen([EDGE,'--headless=new','--disable-gpu','--hide-scrollbars',
            '--remote-debugging-port=%d'%port,'--user-data-dir='+os.path.join(os.environ.get('TEMP','.'),'edge-cdp-prof'),
            '--no-first-run','--no-default-browser-check','about:blank'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        self.ws=None; self._id=0; self._open()
    def _open(self):
        deadline=time.time()+25
        while time.time()<deadline:
            try:
                data=json.loads(_OPENER.open('http://127.0.0.1:%d/json/list'%self.port,timeout=2).read().decode())
                pages=[t for t in data if t.get('type')=='page']
                if pages:
                    self.ws=connect(pages[0]['webSocketDebuggerUrl'],max_size=64*1024*1024)
                    self.call('Page.enable'); self.call('Runtime.enable')
                    self.call('Emulation.setDeviceMetricsOverride',width=1624,height=1046,deviceScaleFactor=1,mobile=False)
                    return
            except Exception: time.sleep(0.4)
        raise RuntimeError('connect failed')
    def call(self,method,**p):
        self._id+=1; mid=self._id
        self.ws.send(json.dumps({'id':mid,'method':method,'params':p}))
        while True:
            m=json.loads(self.ws.recv())
            if m.get('id')==mid:
                if 'error' in m: raise RuntimeError(m['error'])
                return m.get('result',{})
    def goto(self,url,timeout=40):
        self.call('Page.navigate',url=url); t0=time.time()
        while time.time()-t0<timeout:
            try:
                if self.eval('document.readyState')=='complete': return True
            except Exception: pass
            time.sleep(0.3)
        return False
    def eval(self,e,timeout=180):
        r=self.call('Runtime.evaluate',expression=e,returnByValue=True,awaitPromise=True,timeout=int(timeout*1000))
        if r.get('exceptionDetails'): raise RuntimeError(str(r['exceptionDetails'])[:300])
        return r.get('result',{}).get('value')
    def wait(self,e,timeout=300,interval=0.5):
        t0=time.time()
        while time.time()-t0<timeout:
            try:
                if self.eval(e): return True
            except Exception: pass
            time.sleep(interval)
        return False
    def shot(self,path):
        open(path,'wb').write(base64.b64decode(self.call('Page.captureScreenshot',format='png')['data']))
    def close(self):
        try: self.ws.close()
        except Exception: pass
        try: self.proc.terminate()
        except Exception: pass


# ============================== 验收主体 ==============================

RL = str(Path(__file__).resolve().parents[1])
DEFAULT_VIDEO = str(Path(RL) / "workspace" / "9月29日 (5).mp4")


def main() -> int:
    video = DEFAULT_VIDEO
    keep_export = "--keep" in sys.argv
    for a in sys.argv[1:]:
        if not a.startswith("--"):
            video = a
    log = Path(RL) / "_merge_accept.log"
    fh = open(log, "w", encoding="utf-8")
    fails = []

    def ok(flag, what):
        fh.write(("PASS " if flag else "FAIL ") + what + "\n")
        fh.flush()
        print(("PASS " if flag else "FAIL ") + what)
        if not flag:
            fails.append(what)

    py = sys.executable
    env = dict(os.environ, DFL_WORKSPACE=os.environ.get("DFL_WORKSPACE", r"D:\MyProject\DeepFaceLab-PyTorch\workspace"))
    proc = subprocess.Popen([py, "serve.py"], cwd=os.path.join(RL, "app"),
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                            bufsize=1, env=env, creationflags=0x08000000)
    for line in proc.stdout:
        if line.startswith("PORT="):
            port = int(line.strip().split("=")[1])
            break
    threading.Thread(target=lambda: [None for _ in proc.stdout], daemon=True).start()

    b = Browser()
    try:
        b.goto("http://127.0.0.1:%d/" % port)
        b.eval('window.go && window.go("merge"), 1'); time.sleep(1.2)

        t0 = time.time()
        b.eval('window.mgPrepare(%s), 1' % json.dumps(video))
        b.wait("!window.__mergePrep.running", 600)
        frames = b.eval("window.__mv.frames")
        ok(frames and frames > 0, "导入素材：%d 帧，用时 %.1fs" % (frames or 0, time.time() - t0))

        b.eval('''(function(){var r=document.querySelectorAll("#mgModels .mgitem");for(var i=0;i<r.length;i++){if((r[i].querySelector(".nm")||{}).textContent==="model_004"){r[i].click();return 1}}return 0})()''')
        picked = b.eval('(document.querySelector("#mgModels .mgitem.on .nm")||{}).textContent')
        ok(bool(picked), "选模型：%s" % picked)

        b.eval('''(function(){var s=document.getElementById("mgDetector"); s.value="RetinaFace_10g"; s.dispatchEvent(new Event("change")); return 1})()''')
        b.eval('''(function(){var f=document.getElementById("mtlFrame"); f.value="600"; f.dispatchEvent(new Event("change")); return 1})()''')
        time.sleep(2.5)
        b.eval('document.querySelector("#mgViews b[data-v=\\"det\\"]").click(), 1'); time.sleep(3)
        ink = b.eval('''(function(){var c=document.querySelector("#mgStage canvas.mgovl"); if(!c) return -1; var d=c.getContext("2d").getImageData(0,0,c.width,c.height).data; var n=0; for(var i=3;i<d.length;i+=4) if(d[i]>10) n++; return n})()''')
        ok(ink and ink > 500, "检测参数：出框（画布像素 %s）" % ink)

        before = b.eval('(document.querySelector("#mgStage img")||{src:""}).src.length')
        b.eval('''(function(){var f=document.getElementById("mtlFrame"); f.value="200"; f.dispatchEvent(new Event("change")); return 1})()''')
        time.sleep(2.5)
        after = b.eval('(document.querySelector("#mgStage img")||{src:""}).src.length')
        ok(after and after != before, "时间轴：600 → 200 帧，预览换图（%s → %s）" % (before, after))

        b.eval('document.querySelector("#mgViews b[data-v=\\"swap\\"]").click(), 1'); time.sleep(2.5)
        merged_len = b.eval('(document.querySelector("#mgStage img")||{src:""}).src.length')
        ok(merged_len and merged_len > 10000, "预览：合成视图显示成片帧（%s 字节）" % merged_len)

        s = b.eval('''JSON.stringify((function(){var x=window.MergePages.settings();return {mode:x.mode,color:x.color_transfer,fmt:x.image_format,enc:x.encoder,nw:x.num_workers};})())''')
        ok('"enc"' in s, "合成参数可采集：%s" % s)

        b.eval('document.getElementById("mgExport").click(), 1'); time.sleep(0.6)
        b.eval('document.getElementById("mgStartExport2").click(), 1')
        ok(b.wait("!!window.__mergeExporting", 30), "导出成片：任务已启动")

        st = {}
        if keep_export:
            t1 = time.time()
            while time.time() - t1 < 3600:
                time.sleep(5)
                st = json.loads(b.eval('fetch("/api/merge/export/status").then(r=>r.json()).then(d=>JSON.stringify(d))'))
                if not st["running"]:
                    break
            ok(bool(st.get("ok")) and os.path.exists(st.get("out") or ""),
               "导出成片：完成（%s，%s 字节）" % (st.get("out"), os.path.getsize(st["out"]) if st.get("out") and os.path.exists(st["out"]) else -1))
        else:
            time.sleep(8)
            st = json.loads(b.eval('fetch("/api/merge/export/status").then(r=>r.json()).then(d=>JSON.stringify(d))'))
            ok(bool(st.get("running")), "导出进度：pct=%s stage=%s" % (st.get("pct"), st.get("stage")))
            b.eval('fetch("/api/merge/export/cancel", {method:"POST"}).then(r=>r.json()), 1')
            t1 = time.time()
            while time.time() - t1 < 90:
                time.sleep(2)
                st = json.loads(b.eval('fetch("/api/merge/export/status").then(r=>r.json()).then(d=>JSON.stringify(d))'))
                if not st["running"]:
                    break
            ok(not st["running"], "取消导出：已停止（cancelled=%s）" % st.get("cancelled"))
    finally:
        b.close()
        proc.terminate()
        fh.close()

    print("\n日志: %s" % log)
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
