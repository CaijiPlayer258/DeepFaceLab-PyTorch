# -*- coding: utf-8 -*-
"""探针：证明 /api/merge/preview 这条路上超分到底跑没跑。"""
import base64, io, json, os, sys, time, urllib.request, urllib.error
import numpy as np, cv2

BASE = "http://127.0.0.1:8760"
VID = "workspace/5_6172576231574219093.mp4"
LOG = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "_server_output.log")

def post(path, body, timeout=180):
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(BASE + path, data=data,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode("utf-8", "replace")

def logtail(n=40):
    try:
        with open(LOG, "r", encoding="utf-8", errors="replace") as f:
            return f.read().splitlines()[-n:]
    except Exception as e:
        return ["<log 读不到: %s>" % e]

def decode(u):
    if not isinstance(u, str) or "," not in u:
        return None
    b = base64.b64decode(u.split(",", 1)[1])
    return cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_COLOR)

st, r = post("/api/merge/open", {"path": VID})
print("== /api/merge/open ->", st)
print(json.dumps(r, ensure_ascii=False)[:600] if isinstance(r, dict) else str(r)[:600])
print("日志尾:")
for L in logtail(8): print("   |", L)

DET = "RetinaFace_10g"
def settings(sr):
    return {"ct_mode": "mt", "blend": "seamless", "mask_mode": 4,
            "mask_source": "XSegLite", "mask_erode": 0, "mask_blur": 4,
            "mode": "seamless", "color_transfer": "mt", "mask_style": "prd×dst",
            "erode_mask": 0, "blur_mask": 4, "sr": sr, "detector": DET,
            "face_type": "whole_face"}

def body(sr_set, sr_top=None, frame=201):
    b = {"frame": frame, "idx": frame, "n": frame,
         "video": VID, "t": round(frame / 30.0, 3), "w": 900,
         "project": "", "no_swap_faces": [], "fmt": "jpg",
         "model": "DF-UDT416",
         "models": [{"name": "DF-UDT416", "faces": []}],
         "settings": settings(sr_set)}
    if sr_top is not None:
        b["sr"] = sr_top
    return b

cases = [("A: settings.sr=0  (基线)", body(0)),
         ("B: settings.sr=100（前端真实形状：只有 settings 里有 sr）", body(100)),
         ("C: 顶层 sr=100（六宫格形状：settings.sr=0 + 顶层 100）", body(0, 100))]

imgs = {}
for name, b in cases:
    print("\n=== %s ===" % name)
    n0 = len(logtail(100000))
    t0 = time.time()
    st, r = post("/api/merge/preview", b)
    dt = time.time() - t0
    if st != 200 or not isinstance(r, dict):
        print("  HTTP", st, str(r)[:400]); imgs[name] = None; continue
    u = r.get("png") or r.get("jpg") or r.get("img")
    im = decode(u)
    imgs[name] = im
    print("  HTTP 200 · %.2fs · box=%r · has_face=%r · swapped=%r · faces=%d · timings=%s"
          % (dt, r.get("box"), r.get("has_face"), r.get("swapped"),
             len(r.get("faces") or []), json.dumps(r.get("timings") or {}, ensure_ascii=False)))
    print("  faces[0]=%s" % json.dumps((r.get("faces") or [None])[0], ensure_ascii=False)[:300])
    print("  img=%s bytes=%s" % (None if im is None else im.shape, len(u or "")))
    for L in logtail(100000)[n0:]:
        print("   log|", L)

def diff(a, b):
    if a is None or b is None: return "n/a"
    if a.shape != b.shape: return "shape 不同 %r vs %r" % (a.shape, b.shape)
    d = np.abs(a.astype(np.int16) - b.astype(np.int16))
    return "max=%d mean=%.3f 变化像素=%d/%d (%.2f%%)" % (
        d.max(), d.mean(), int((d.max(axis=2) > 0).sum()), a.shape[0]*a.shape[1],
        100.0 * (d.max(axis=2) > 0).sum() / (a.shape[0]*a.shape[1]))

names = [n for n, _ in cases]
print("\n===== 像素对比 =====")
print("A(基线) vs B(settings.sr=100):", diff(imgs.get(names[0]), imgs.get(names[1])))
print("A(基线) vs C(顶层 sr=100)  :", diff(imgs.get(names[0]), imgs.get(names[2])))
print("B vs C                     :", diff(imgs.get(names[1]), imgs.get(names[2])))
for k, v in imgs.items():
    if v is not None:
        cv2.imwrite(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 "out_%s.png" % (k[0])), v)
