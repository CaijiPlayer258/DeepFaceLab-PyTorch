
# -*- coding: utf-8 -*-
"""流水线冒烟：直接在进程里跑 _proj_pipe.run（不走 HTTP），可指定帧数。"""
import sys, os, json, shutil, time
sys.path.insert(0, r"D:/MyProject/DeepFaceLab-PyTorch-4.1/app")
import server.app as A
from server import routes_merge as rm
from pathlib import Path

NF = int(sys.argv[1]) if len(sys.argv) > 1 else 12
TAG = sys.argv[2] if len(sys.argv) > 2 else "smoke"
TMP = Path(os.environ["TEMP"])
SRC = TMP / "dfl_pool_accept" / "proj"
BOX = TMP / ("_pipe_" + TAG)
KEEP = bool(os.environ.get("KEEP_BOX"))
if not KEEP:
    shutil.rmtree(BOX, ignore_errors=True)
PROJ = BOX / "proj"; (PROJ / "aligned").mkdir(parents=True, exist_ok=True)
frames = sorted(p for p in SRC.iterdir() if p.suffix.lower() in (".png", ".jpg"))[:NF]
keep = set()
for f in frames:
    shutil.copy2(str(f), str(PROJ / f.name)); keep.add(f.stem)
for a in sorted((SRC / "aligned").glob("*.jpg")):
    if a.stem.split("_")[0] in keep:
        shutil.copy2(str(a), str(PROJ / "aligned" / a.name))
print("proj:", PROJ, len(frames), "frames,", len(list((PROJ/'aligned').glob('*.jpg'))), "aligned")

SETTINGS = {"mask_mode": 8, "color_transfer": "mt", "blend": "seamless",
            "mask_erode": 0, "mask_blur": 4}
_W = TMP / "dfl_pool_accept" / "ws" / "model"
_CAND = sorted(_W.glob("*.dfm"), key=lambda x: x.stat().st_size)
_WANT = sys.argv[3] if len(sys.argv) > 3 else "416"
DFM = str(_CAND[-1] if _WANT == "384" else _CAND[0])   # 384 = 最大的那份(张元英) / 416 = 最小的
XSEG = str(rm._xseg_of({"xseg": str(Path(A.WORKSPACE) / "model" / "XSegLite" / "xseglite.onnx")}))
cfg = {"project": str(PROJ), "frames_dir": str(PROJ), "aligned": str(PROJ / "aligned"),
       "merged": str(BOX / "merged"), "video": str(TMP / "dfl_pool_accept" / "media" / "clip.mp4"),
       "out_video": str(BOX / "out.mp4"), "fps": 30.0, "video_frames": NF,
       "models": [], "dfm": DFM, "model": Path(DFM).stem,
       "config": A._merge_settings_to_config(SETTINGS), "settings": SETTINGS,
       "xseg": XSEG, "detector": "tinymog", "thresh": 0.5, "quality": 92,
       "force": not KEEP, "workers": 1, "batch": 4, "adaptive": {}, "vram_plan": {}}
CANCEL_AFTER = int(sys.argv[5]) if len(sys.argv) > 5 else 0
if len(sys.argv) > 4 and "x" in sys.argv[4]:
    _w, _b = sys.argv[4].split("x")
    cfg["workers"], cfg["batch"] = int(_w), int(_b)
last = {"n": 0}
def watch():
    while True:
        s = rm._proj_snap()
        if s.get("done") != last["n"]:
            last["n"] = s.get("done")
        time.sleep(0.5)
import threading
if CANCEL_AFTER:
    def _cancel():
        while True:
            s = rm._proj_snap()
            if int(s.get("done") or 0) >= CANCEL_AFTER and s.get("running", True):
                with rm._PROJ_LOCK:
                    rm._PROJ_JOB["want_cancel"] = True
                print(">>> cancel requested at done=%s" % s.get("done"))
                return
            time.sleep(0.2)
    threading.Thread(target=_cancel, daemon=True).start()
t0 = time.time()
try:
    rm._proj_run(cfg)
except Exception as e:
    import traceback; traceback.print_exc()
    print("FAILED:", e)
s = rm._proj_snap()
print("--- status ---")
for k in ("done","swapped","copied","skipped","failed","frame_fps","swap_p50","workers","batch",
          "compose","pipeline","elapsed","swap_elapsed","merged_frames","encode"):
    v = s.get(k)
    if k == "encode" and isinstance(v, dict):
        v = {kk: v.get(kk) for kk in ("code","seconds")}
    print("  %-12s %s" % (k, v))
print("log:")
for ln in (s.get("log") or []):
    print("   ", ln)
print("warn:")
for ln in (s.get("warnings") or []):
    print("   ", ln)
print("wall %.1fs  merged files %d" % (time.time()-t0, len(list((BOX/'merged').glob('*')))))
(BOX / "status.json").write_text(json.dumps(s, ensure_ascii=False, indent=1), encoding="utf-8")
print("status ->", BOX / "status.json")
