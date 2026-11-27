# -*- coding: utf-8 -*-
"""探针 7：修完之后，/api/merge/frame 认不认 settings.sr（跑在 8761 这个新实例上）。"""
import base64, json, os, time, urllib.request, urllib.error
import numpy as np, cv2

BASE = os.environ.get("PROBE_BASE", "http://127.0.0.1:8761")
HERE = os.path.dirname(os.path.abspath(__file__))
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

def decode(u):
    if not isinstance(u, str) or "," not in u: return None
    return cv2.imdecode(np.frombuffer(base64.b64decode(u.split(",", 1)[1]), np.uint8), cv2.IMREAD_COLOR)

def settings(sr):
    return {"mode": "seamless", "detector": "RetinaFace_10g", "face_type": "whole_face",
            "mask_source": "XSegLite", "mask_mode": 8, "color_transfer": "mt",
            "erode_mask": 10, "blur_mask": 10, "sr": sr}

def body(sr_top, sr_set, frame=301, w=900):
    b = {"frame": frame, "idx": frame, "n": frame, "video": VID,
         "t": round(frame/30.0, 3), "w": w, "model": MODEL,
         "models": [{"name": MODEL, "faces": []}], "no_swap_faces": [],
         "fmt": "jpg", "settings": settings(sr_set)}
    if sr_top is not None: b["sr"] = sr_top
    return b

st, r = post("/api/merge/open", {"path": VID})
print("open ->", st, (r or {}).get("frames") if isinstance(r, dict) else r)

cases = [("基线  ", body(None, 0)),      # 完全不带 sr
         ("settings", body(None, 100)),  # 旧前端形状（只 settings）
         ("顶层   ", body(100, 0)),      # 新前端形状 + 六宫格形状
         ("两处   ", body(100, 100))]
imgs = {}
for tag, b in cases:
    t0 = time.time(); st, r = post("/api/merge/frame", b); dt = time.time()-t0
    rr = r if isinstance(r, dict) else {}
    im = decode(rr.get("png"))
    imgs[tag] = im
    print("  %-8s settings.sr=%-4r 顶层=%-5r → HTTP %s %6.2fs box=%r 图=%s"
          % (tag, b["settings"]["sr"], b.get("sr"), st, dt, rr.get("box"),
             None if im is None else im.shape))

def d(a, b, label):
    if a is None or b is None: print("   %-34s n/a" % label); return
    x = np.abs(a.astype(np.int16)-b.astype(np.int16))
    ch = int((x.max(axis=2) > 0).sum())
    print("   %-34s max=%-4d mean=%7.4f 变化 %d/%d (%.2f%%)"
          % (label, x.max(), x.mean(), ch, a.shape[0]*a.shape[1], 100.0*ch/(a.shape[0]*a.shape[1])))

print("\n像素对比：")
d(imgs["基线  "], imgs["settings"], "基线 vs settings.sr=100")
d(imgs["基线  "], imgs["顶层   "], "基线 vs 顶层 sr=100")
d(imgs["基线  "], imgs["两处   "], "基线 vs 两处都给 100")

# 六宫格：merged 格
print("\n六宫格 merged 格：")
pan = {}
for tag, (top, st_) in (("D0", (None, 0)), ("D1", (None, 100)), ("D2", (100, 0))):
    b = body(top, st_, frame=301); b["face"] = 0; b["panel_size"] = 256; b.pop("idx", None)
    st, r = post("/api/merge/face-debug", b)
    for pn in ((r or {}).get("panels") or []):
        if str(pn.get("label") or "").strip() == "merged":
            pan[tag] = decode(pn.get("png"))
    print("  %s HTTP %s 格数=%d" % (tag, st, len((r or {}).get("panels") or [])))
d(pan.get("D0"), pan.get("D1"), "D0 vs D1(settings.sr=100)")
d(pan.get("D0"), pan.get("D2"), "D0 vs D2(顶层 sr=100)")
