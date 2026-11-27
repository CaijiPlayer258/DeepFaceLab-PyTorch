# -*- coding: utf-8 -*-
"""探针 5：当前工作树里 compose._frame_payload 的 SR 入口能不能跑通（不依赖 GPU）。"""
import os, sys, hashlib
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "app"))
os.chdir(os.path.join(ROOT, "app"))

for f in ("app/server/backend_merge/compose.py", "app/server/backend_merge/routes.py",
          "app/server/backend_merge/preview.py"):
    p = os.path.join(ROOT, f)
    print("%-45s sha1=%s" % (f, hashlib.sha1(open(p, "rb").read()).hexdigest()[:12]))

from server.backend_merge import compose as C
print("\n'_sr_lv' 在 compose 的模块全局里吗：", "_sr_lv" in vars(C))
print("'_sr_lv' 在 compose 的源码里出现几次：",
      open(os.path.join(ROOT, "app/server/backend_merge/compose.py"), encoding="utf-8")
      .read().count("_sr_lv("))

# 直接模拟 _frame_payload 开头那三行
payload = {"sr": None, "settings": {"sr": 100}}
try:
    from server.backend_merge.knobs import apply_sr as _apply_sr
    r = eval("_apply_sr(_sr_lv(payload) if isinstance(payload, dict) else None)",
             {"_sr_lv": getattr(C, "_sr_lv", None), "__builtins__": {}},
             {"_apply_sr": _apply_sr, "payload": payload}) if hasattr(C, "_sr_lv") else "跳过"
    print("模拟 _frame_payload:1508 →", r)
except Exception as e:
    print("模拟 _frame_payload:1508 →", type(e).__name__, e)

# 用正确的函数名试一次（这就是修法）
_lv = C._sr_level_of(payload)
print("\n_sr_level_of({'settings': {'sr': 100}}) =", _lv, "  ← 现在的实现已经认 settings 了")
print("apply_sr(_lv) →", _apply_sr(_lv), " → env DFL_SR =", os.environ.get("DFL_SR"))
from server.backend_merge import facesr
print("facesr.enabled() =", facesr.enabled(), " strength() =", facesr.strength(), " size() =", facesr.size())
