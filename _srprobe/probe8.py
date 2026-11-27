# -*- coding: utf-8 -*-
"""探针 8：旧路 /api/merge/preview 有没有被这次改动带坏（回归检查）。"""
import base64, json, os, time, urllib.request, urllib.error
import numpy as np, cv2
BASE = "http://127.0.0.1:8761"
OP = urllib.request.build_opener(urllib.request.ProxyHandler({}))
VID = r"D:\MyProject\DeepFaceLab-PyTorch-4.1\workspace\5_6172576231574219093.mp4"
MODEL = "害羞7-416直播模型"
def post(p, b, timeout=300):
    req = urllib.request.Request(BASE+p, data=json.dumps(b).encode("utf-8"),
                                 headers={"Content-Type": "application/json"})
    with OP.open(req, timeout=timeout) as r:
        return r.status, json.loads(r.read().decode("utf-8"))
def dec(u):
    return cv2.imdecode(np.frombuffer(base64.b64decode(u.split(",",1)[1]), np.uint8), cv2.IMREAD_COLOR)
def body(sr, frame=301):
    return {"frame": frame, "idx": frame, "n": frame, "video": VID,
            "t": round(frame/30.0,3), "w": 900, "model": MODEL,
            "models": [{"name": MODEL, "faces": []}], "no_swap_faces": [], "fmt": "jpg",
            "settings": {"mode":"seamless","detector":"RetinaFace_10g","face_type":"whole_face",
                         "mask_source":"XSegLite","mask_mode":8,"color_transfer":"mt",
                         "erode_mask":10,"blur_mask":10,"sr":sr}}
im = {}
for tag, sr in (("sr=0", 0), ("sr=100", 100)):
    st, r = post("/api/merge/preview", body(sr))
    im[tag] = dec(r["png"]); print("/api/merge/preview %-7s HTTP %s 图=%s" % (tag, st, im[tag].shape))
x = np.abs(im["sr=0"].astype(np.int16)-im["sr=100"].astype(np.int16))
print("  差异 max=%d mean=%.3f 变化=%.2f%%" % (x.max(), x.mean(),
      100.0*(x.max(axis=2)>0).sum()/x[:,:,0].size))
