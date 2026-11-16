
# -*- coding: utf-8 -*-
"""默认档（nvenc-h264 CQ21 / hevc CQ21）在 4K 上的干净计时：3 轮，机器空着。"""
import importlib.util, sys, json
from pathlib import Path
spec = importlib.util.spec_from_file_location("b", str(Path(__file__).with_name("bench4k.py")))
B = importlib.util.module_from_spec(spec); sys.argv=["x"]; spec.loader.exec_module(B)
D = B.D
fr = B.load("f_4k.raw", 2160, 3840, 80)
EX = {
 "B3 h264_nvenc CQ21": ["-c:v","h264_nvenc","-preset","p5","-tune","hq","-rc","vbr","-cq","21","-b:v","0"],
 "C3 hevc_nvenc CQ21": ["-c:v","hevc_nvenc","-preset","p5","-tune","hq","-rc","vbr","-cq","21","-b:v","0"],
}
out={}
for rnd in (1,2,3):
    for name,opts in EX.items():
        o = D/("out_4k_q21_r%d_%s.mp4"%(rnd,name.split()[0]))
        r = B.measure(name, opts, 2160, 3840, fr, str(o), do_gpu=False)
        out.setdefault(name,[]).append(r)
import statistics
print("\n=== 中位 ===")
for k,v in out.items():
    print("%-20s fps %.1f  ms/帧 %.1f  核 %.2f  %.2f MB  %.1f Mbps  md5稳定=%s" % (
        k, statistics.median(x["fps"] for x in v), statistics.median(x["ms_frame"] for x in v),
        statistics.median(x["cores_ffmpeg"] for x in v), statistics.median(x["mb"] for x in v),
        statistics.median(x["mbps"] for x in v), len({x["md5"] for x in v})==1))
json.dump(out, open(D/"results_cq21_clean.json","w"), indent=1)
