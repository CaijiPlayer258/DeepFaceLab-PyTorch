# -*- coding: utf-8 -*-
"""探针 3：把三条出图路径（/frame、/merged、/face-debug）的超分行为一次照清楚。"""
import base64, json, os, time, urllib.request, urllib.error
import numpy as np, cv2

BASE = "http://127.0.0.1:8760"
HERE = os.path.dirname(os.path.abspath(__file__))
LOG = os.path.join(os.path.dirname(HERE), "_server_output.log")
OP = urllib.request.build_opener(urllib.request.ProxyHandler({}))
VID = r"D:\MyProject\DeepFaceLab-PyTorch-4.1\workspace\5_6172576231574219093.mp4"
MODEL = "害羞7-416直播模型"

def post(path, body, timeout=300):
    req = urllib.request.Request(BASE + path, data=json.dumps(body).encode("utf-8"),
                                 headers={"Content-Type": "application/json"})
    try:
        with OP.open(req, timeout=timeout) as r:
            return r.status, json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        raw = e.read().decode("utf-8", "replace")
        try: return e.code, json.loads(raw)
        except Exception: return e.code, raw or "<空>"

def loglines():
    with open(LOG, "r", encoding="utf-8", errors="replace") as f:
        return f.read().splitlines()

def decode(u):
    if not isinstance(u, str) or "," not in u: return None
    return cv2.imdecode(np.frombuffer(base64.b64decode(u.split(",", 1)[1]), np.uint8), cv2.IMREAD_COLOR)

def settings(sr):
    return {"mode": "seamless", "detector": "RetinaFace_10g", "face_type": "whole_face",
            "mask_source": "XSegLite", "mask_mode": 8, "color_transfer": "mt",
            "erode_mask": 10, "blur_mask": 10, "sr": sr}

def body(route_sr, frame=201, w=900, sr_set=None):
    if sr_set is None: sr_set = route_sr
    b = {"frame": frame, "idx": frame, "n": frame, "video": VID,
         "t": round(frame / 30.0, 3), "w": w, "model": MODEL,
         "models": [{"name": MODEL, "faces": []}],
         "no_swap_faces": [], "fmt": "jpg", "settings": settings(sr_set)}
    if route_sr is not None:
        b["sr"] = route_sr
    return b

def run(tag, path, b):
    before = len(loglines()); t0 = time.time()
    st, r = post(path, b); dt = time.time() - t0
    new = loglines()[before:]
    sr = [L for L in new if "[sr]" in L or "[srdbg]" in L]
    rr = r if isinstance(r, dict) else {}
    im = decode(rr.get("png") or rr.get("jpg") or rr.get("img"))
    print("  %-3s %-26s settings.sr=%-4r 顶层 sr=%-5r → HTTP %s %6.2fs  图=%s  sr日志=%d"
          % (tag, path, b["settings"]["sr"], b.get("sr"), st, dt,
             None if im is None else im.shape, len(sr)))
    for L in sr: print("        log|", L)
    if st != 200: print("        响应：", str(r)[:200])
    return im, rr, dt

def diff(a, b, label):
    if a is None or b is None: print("   %-40s n/a" % label); return
    if a.shape != b.shape: print("   %-40s shape 不同 %r/%r" % (label, a.shape, b.shape)); return
    d = np.abs(a.astype(np.int16) - b.astype(np.int16))
    ch = int((d.max(axis=2) > 0).sum())
    print("   %-40s max=%-4d mean=%7.4f 变化 %d/%d (%.2f%%)"
          % (label, d.max(), d.mean(), ch, a.shape[0]*a.shape[1], 100.0*ch/(a.shape[0]*a.shape[1])))

# ---------- 0. 打开素材 ----------
print("open:", post("/api/merge/open", {"path": VID})[0])

# ---------- 1. /api/merge/frame 第 202 帧：缓存键到底认不认 settings.sr ----------
print("\n[1] /api/merge/frame  frame=202（换一帧，避开上一轮缓存）")
q0, _, t_q0 = run("Q0", "/api/merge/frame", body(None, 202))
q1, _, t_q1 = run("Q1", "/api/merge/frame", body(None, 202, sr_set=100))
q2, _, t_q2 = run("Q2", "/api/merge/frame", body(100, 202, sr_set=0))
diff(q0, q1, "Q0(sr=0) vs Q1(settings.sr=100)")
diff(q0, q2, "Q0(sr=0) vs Q2(顶层 sr=100)")
print("   耗时：Q0=%.2fs Q1=%.2fs Q2=%.2fs（Q1 若 ≈0.03s 就是命中缓存，压根没重算）" % (t_q0, t_q1, t_q2))

# ---------- 2. /api/merge/merged（拖 SR 滑块时 mvMerged 发的就是这条，w=1920）----------
print("\n[2] /api/merge/merged  frame=201 w=1920（SR 滑块 change → mvMerged 的真实形状）")
m0, _, _ = run("M0", "/api/merge/merged", body(None, 201, w=1920))
m1, _, _ = run("M1", "/api/merge/merged", body(None, 201, w=1920, sr_set=100))
m2, _, _ = run("M2", "/api/merge/merged", body(100, 201, w=1920, sr_set=100))
diff(m0, m1, "M0(sr=0) vs M1(settings.sr=100)")
diff(m0, m2, "M0(sr=0) vs M2(顶层+settings sr=100)")

# ---------- 3. 六宫格 ----------
print("\n[3] /api/merge/face-debug  frame=201 face=0 panel=256")
def fdbg(sr_set, sr_top):
    b = body(sr_top, 201, sr_set=sr_set); b["face"] = 0; b["panel_size"] = 256; b.pop("idx", None)
    return b
def panel(rr):
    for pn in (rr.get("panels") or []):
        if str(pn.get("label") or "").strip() == "merged":
            return decode(pn.get("png"))
    return None
d0r = run("D0", "/api/merge/face-debug", fdbg(0, None))[1]
d1r = run("D1", "/api/merge/face-debug", fdbg(100, None))[1]
d2r = run("D2", "/api/merge/face-debug", fdbg(0, 100))[1]
p0, p1, p2 = panel(d0r), panel(d1r), panel(d2r)
print("   panels: D0=%d D1=%d D2=%d 格" % (len(d0r.get("panels") or []), len(d1r.get("panels") or []), len(d2r.get("panels") or [])))
diff(p0, p1, "merged 格：D0 vs D1(settings.sr=100)")
diff(p0, p2, "merged 格：D0 vs D2(顶层 sr=100)")
