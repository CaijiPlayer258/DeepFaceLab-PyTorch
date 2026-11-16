
# -*- coding: utf-8 -*-
"""造 4K 素材：真实 DFL 成片 1080x1920 → 原生 1080p raw + lanczos 2160x3840 raw，各 80 帧。
另外给一条 '容易内容' 的对照：yellow-river 动画 → 3840x2160 raw 40 帧。
产出：f_hd.raw / f_4k.raw / f_easy4k.raw / audio.m4a / meta.json"""
import subprocess, time, os, json
from pathlib import Path
import imageio_ffmpeg, numpy as np

FF = imageio_ffmpeg.get_ffmpeg_exe()
D = Path(__file__).resolve().parents[4] / "_enc4k_bench"   # 仓库根的 scratch 素材/产物目录（不进仓）
D.mkdir(parents=True, exist_ok=True)
SRC = str(Path(__file__).resolve().parents[4] / "workspace" / "9月29日 (5)_merged.mp4")
# 外部对照素材（不在仓里）：换机器就设 ENC4K_EASY_SRC，或跳过 easy4k 这一条臂
EASY = os.environ.get("ENC4K_EASY_SRC", r"D:\MyProject\视频演示文稿\yellow-river\out\video_final.mp4")
N = 80
SS = 12

def dump_raw(src, w, h, n, out, ss=SS):
    cmd = [FF,"-y","-hide_banner","-loglevel","error","-ss",str(ss),"-i",src,
           "-frames:v",str(n),"-vf","scale=%d:%d:flags=lanczos"%(w,h),
           "-f","rawvideo","-pix_fmt","bgr24","pipe:1"]
    t0=time.perf_counter()
    p=subprocess.Popen(cmd,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    fr=w*h*3; buf=bytearray()
    with open(out,"wb") as fh:
        while True:
            b=p.stdout.read(fr)
            if len(b)<fr: break
            fh.write(b); buf+=b
    p.wait(); err=p.stderr.read()
    nf=len(buf)//fr
    print("%-14s %dx%d  %d 帧  %.2fs (%.1f ms/帧)  %.0f MB  rc=%s %s"%(
        out.name,w,h,nf,time.perf_counter()-t0,1000*(time.perf_counter()-t0)/max(1,nf),
        len(buf)/2**20, p.returncode, err[:200].decode("utf-8","replace")))
    return nf

meta={}
meta["hd"]=dict(w=1080,h=1920,n=dump_raw(SRC,1080,1920,N,D/"f_hd.raw"))
meta["4k"]=dict(w=2160,h=3840,n=dump_raw(SRC,2160,3840,N,D/"f_4k.raw"))
meta["easy4k"]=dict(w=3840,h=2160,n=dump_raw(EASY,3840,2160,40,D/"f_easy4k.raw",ss=30))
# 音轨（production -c:a copy 的那条）：切片 2.667s
subprocess.run([FF,"-y","-hide_banner","-loglevel","error","-ss",str(SS),"-t","2.666667",
                "-i",SRC,"-vn","-c:a","copy",str(D/"audio.m4a")],check=True)
print("audio.m4a", os.path.getsize(D/"audio.m4a"))
# 1080p 首帧看一眼（找脸的位置）
subprocess.run([FF,"-y","-hide_banner","-loglevel","error","-ss",str(SS),"-i",SRC,"-frames:v","1",
                "-vf","scale=540:960",str(D/"peek_hd.png")],check=True)
json.dump(meta, open(D/"meta.json","w"), indent=1)
print(json.dumps(meta))
