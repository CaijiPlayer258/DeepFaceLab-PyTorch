# -*- coding: utf-8 -*-
"""探针 2（no-proxy）：/api/merge/frame（合成工作台预览的主路）到底跑不跑超分。"""
import base64, json, os, time, urllib.request, urllib.error
import numpy as np, cv2

BASE = "http://127.0.0.1:8760"
HERE = os.path.dirname(os.path.abspath(__file__))
LOG = os.path.join(os.path.dirname(HERE), "_server_output.log")
OP = urllib.request.build_opener(urllib.request.ProxyHandler({}))
VID = r"D:\MyProject\DeepFaceLab-PyTorch-4.1\workspace\5_6172576231574219093.mp4"

def post(path, body, timeout=300):
    req = urllib.request.Request(BASE + path, data=json.dumps(body).encode("utf-8"),
                                 headers={"Content-Type": "application/json"})
    try:
        with OP.open(req, timeout=timeout) as r:
            return r.status, json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        raw = e.read().decode("utf-8", "replace")
        try:
            return e.code, json.loads(raw)
        except Exception:
            return e.code, raw or "<空响应体>"

def loglines():
    with open(LOG, "r", encoding="utf-8", errors="replace") as f:
        return f.read().splitlines()

def decode(u):
    if not isinstance(u, str) or "," not in u:
        return None
    return cv2.imdecode(np.frombuffer(base64.b64decode(u.split(",", 1)[1]), np.uint8),
                        cv2.IMREAD_COLOR)

st, r = post("/api/merge/open", {"path": VID})
print("open ->", st, json.dumps(r, ensure_ascii=False)[:220] if isinstance(r, dict) else str(r)[:220])

DET = "RetinaFace_10g"
def settings(sr):
    return {"mode": "seamless", "detector": DET, "face_type": "whole_face",
            "mask_source": "XSegLite", "mask_mode": 8, "color_transfer": "mt",
            "erode_mask": 10, "blur_mask": 10, "sr": sr}

def body(sr_set, sr_top=None, frame=201):
    b = {"frame": frame, "idx": frame, "n": frame, "video": VID,
         "t": round(frame / 30.0, 3), "w": 900, "model": "害羞7-416直播模型",
         "models": [{"name": "害羞7-416直播模型", "faces": []}],
         "settings": settings(sr_set)}
    if sr_top is not None:
        b["sr"] = sr_top
    return b

cases = [("F0", "/api/merge/frame",   body(0)),
         ("F1", "/api/merge/frame",   body(100)),
         ("F2", "/api/merge/frame",   body(0, 100)),
         ("P0", "/api/merge/preview", body(0)),
         ("P1", "/api/merge/preview", body(100)),
         ("P2", "/api/merge/preview", body(0, 100))]

imgs = {}
for tag, route, b in cases:
    before = len(loglines())
    t0 = time.time()
    st, r = post(route, b)
    dt = time.time() - t0
    new = loglines()[before:]
    sr_logs = [L for L in new if ("[sr]" in L or "[srdbg]" in L)]
    rr = r if isinstance(r, dict) else {}
    u = rr.get("png") or rr.get("jpg") or rr.get("img")
    im = decode(u)
    imgs[tag] = im
    print("\n=== %s  %s   settings.sr=%r  顶层 sr=%r" % (tag, route, b["settings"]["sr"], b.get("sr")))
    print("    HTTP %s · %.2fs · box=%r · has_face=%r · swapped=%r · img=%s"
          % (st, dt, rr.get("box"), rr.get("has_face"), rr.get("swapped"),
             None if im is None else im.shape))
    if st != 200:
        print("    响应：", str(r)[:300])
    print("    本条请求期间与超分有关的日志（%d 条）：" % len(sr_logs))
    if sr_logs:
        for L in sr_logs:
            print("      log|", L)
    else:
        print("      （零条 —— 超分那段代码一行都没执行）")
    if im is not None:
        cv2.imwrite(os.path.join(HERE, "img_%s.png" % tag), im)

def diff(a, b):
    if a is None or b is None:
        return "n/a（有一边没图）"
    if a.shape != b.shape:
        return "shape 不同 %r vs %r" % (a.shape, b.shape)
    d = np.abs(a.astype(np.int16) - b.astype(np.int16))
    ch = int((d.max(axis=2) > 0).sum())
    return "max=%d mean=%.4f 变化像素=%d/%d (%.2f%%)" % (
        d.max(), d.mean(), ch, a.shape[0]*a.shape[1], 100.0*ch/(a.shape[0]*a.shape[1]))

print("\n" + "="*72)
print("主路 /api/merge/frame（合成工作台预览真正走的接口）：")
print("   F0(sr=0) vs F1(settings.sr=100) :", diff(imgs["F0"], imgs["F1"]))
print("   F0(sr=0) vs F2(顶层 sr=100)     :", diff(imgs["F0"], imgs["F2"]))
print("/api/merge/preview（b54 修的接口）：")
print("   P0(sr=0) vs P1(settings.sr=100) :", diff(imgs["P0"], imgs["P1"]))
print("   P0(sr=0) vs P2(顶层 sr=100)     :", diff(imgs["P0"], imgs["P2"]))
