# -*- coding: utf-8 -*-
"""探针 4：不走 HTTP，直接喂两个请求体给这几个函数，看它们各自读到了什么。
   body_preview = mgCmpBody() 的真实形状（sr 只在 settings 里）
   body_fdbg    = mgFdbgBody() 的形状（settings.sr + 顶层 sr）
"""
import json, sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app"))
os.chdir(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app"))

from server.backend_merge.compose import (_sr_level_of, _compose_sig,
                                          _merge_settings_to_config)
from server.backend_merge.knobs import _sr_level

DET = "RetinaFace_10g"
st = {"mode": "seamless", "detector": DET, "face_type": "whole_face",
      "mask_source": "XSegLite", "mask_mode": 8, "color_transfer": "mt",
      "erode_mask": 10, "blur_mask": 10, "sr": 100}

mgCmpBody   = {"frame": 201, "idx": 201, "n": 201, "video": "v.mp4", "t": 6.7, "w": 900,
               "project": "", "no_swap_faces": [], "fmt": "jpg",
               "model": "M", "models": [{"name": "M", "faces": []}], "settings": st}
mgFdbgBody  = dict(mgCmpBody); mgFdbgBody.pop("idx"); mgFdbgBody["face"] = 0
mgFdbgBody["panel_size"] = 256; mgFdbgBody["sr"] = 100        # ← 10-merge-settings.js:365

for name, b in (("mgCmpBody（合成预览发出去的）", mgCmpBody),
                ("mgFdbgBody（六宫格发出去的）", mgFdbgBody)):
    print("\n%s 顶层 sr=%r  settings.sr=%r" % (name, b.get("sr"), b["settings"]["sr"]))
    print("   compose._sr_level_of(payload)      =", _sr_level_of(b),
          "   ← compose.py:1715 用它决定跑不跑 GPEN")
    print("   compose._apply_sr(payload['sr'])   → knobs._sr_level(%r) = %r"
          % (b.get("sr"), _sr_level(b.get("sr"))))

print("\ncfg（_merge_settings_to_config 的产物，进缓存键的那个）里有 sr 吗：",
      [k for k in _merge_settings_to_config(st) if "sr" in k.lower()] or "没有")
print("   cfg 的全部键：", sorted(_merge_settings_to_config(st)))

# 缓存身份：settings.sr=0 与 settings.sr=100 的同一个 body，签名一样吗
b0 = json.loads(json.dumps(mgCmpBody)); b0["settings"]["sr"] = 0
b1 = json.loads(json.dumps(mgCmpBody)); b1["settings"]["sr"] = 100
snap = {"path": r"D:\MyProject\DeepFaceLab-PyTorch-4.1\workspace\5_6172576231574219093.mp4"}
s0 = _compose_sig(snap, "video", 202, "x.jpg", _merge_settings_to_config(b0["settings"]),
                  "M.dfm", [{"name": "M", "dfm": "M.dfm", "faces": []}], "", DET, 0.5,
                  "jpg", 88, 900, [], False, _sr_level_of(b0))
s1 = _compose_sig(snap, "video", 202, "x.jpg", _merge_settings_to_config(b1["settings"]),
                  "M.dfm", [{"name": "M", "dfm": "M.dfm", "faces": []}], "", DET, 0.5,
                  "jpg", 88, 900, [], False, _sr_level_of(b1))
print("\n_compose_sig(settings.sr=0)  == _compose_sig(settings.sr=100) ?", s0 == s1)
print("   （相等 = 改了滑块，合成缓存仍然命中，不会重算）")
