# -*- coding: utf-8 -*-
"""探针 6：把 _sr_frame_bytes 单独拿出来按分支喂一遍 —— 证明
   ① 它的跳过分支确实会打日志（所以日志里"没有"= 真的没进去）
   ② 给了有效 box 它确实会把像素改掉（不是 alpha≈0）
"""
import os, sys, hashlib
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "app")); os.chdir(os.path.join(ROOT, "app"))
p = os.path.join(ROOT, "app/server/backend_merge/preview.py")
print("preview.py sha1 =", hashlib.sha1(open(p, "rb").read()).hexdigest()[:12])

import numpy as np, cv2
from server.backend_merge.preview import _sr_frame_bytes
from server.backend_merge import facesr as _sr

src = cv2.imread(os.path.join(ROOT, "_srprobe", "img_F0.png"), cv2.IMREAD_COLOR)
print("底图：", None if src is None else src.shape)

def enc(im, q=95):
    return cv2.imencode(".jpg", im, [int(cv2.IMWRITE_JPEG_QUALITY), q])[1].tobytes()
def dec(b):
    return cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_COLOR)
def d(a, b):
    x = np.abs(a.astype(np.int16) - b.astype(np.int16))
    return "max=%d mean=%.3f 变化=%d(%.2f%%)" % (x.max(), x.mean(),
            int((x.max(axis=2) > 0).sum()), 100.0*(x.max(axis=2) > 0).sum()/x[:,:,0].size)

blob = enc(src)

print("\n--- ① env DFL_SR 没设（= 预览请求里没有顶层 sr 时 compose 的后果）---")
os.environ.pop("DFL_SR", None)
out = _sr_frame_bytes(blob, [111, 189, 695, 354], np, cv2)
print("   strength()=%r 返回长度 %d（原 %d）  字节完全一致=%r"
      % (_sr.strength(), len(out), len(blob), out == blob))

print("\n--- ② env DFL_SR=100，但 box=None（老 ValueError 那条路的等价物）---")
os.environ["DFL_SR"] = "100"
out = _sr_frame_bytes(blob, None, np, cv2)
print("   strength()=%r size()=%r 返回长度 %d（原 %d）  完全一致=%r"
      % (_sr.strength(), _sr.size(), len(out), len(blob), out == blob))

print("\n--- ③ env DFL_SR=100 + 有效 box=[111,189,695,354]（真正干活）---")
out = _sr_frame_bytes(blob, [111, 189, 695, 354], np, cv2)
print("   返回长度 %d（原 %d）  完全一致=%r" % (len(out), len(blob), out == blob))
a, b = dec(blob), dec(out)
print("   解回图后整帧差异：", d(a, b))
box = (111, 189, 695, 354)
x, y, w, h = box
print("   脸框内 差异：", d(a[y:y+h, x:x+w], b[y:y+h, x:x+w]))
print("   脸框外 差异：", d(a[:y], b[:y]))
