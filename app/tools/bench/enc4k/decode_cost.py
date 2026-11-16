
# -*- coding: utf-8 -*-
"""成片解码成本 + 容器/流参数：h264(CQ19) vs hevc(CQ19) vs x264，4K 与 1080p。"""
import subprocess, time, re
from pathlib import Path
import psutil, imageio_ffmpeg
FF = imageio_ffmpeg.get_ffmpeg_exe()
D = Path(__file__).resolve().parents[4] / "_enc4k_bench"   # 仓库根的 scratch 素材/产物目录（不进仓）
for tag in ("4k","hd"):
    print("=== %s ===" % tag)
    for a in ("A","B1","C1"):
        p = D/("out_%s_r1_%s.mp4"%(tag,a))
        info = subprocess.run([FF,"-hide_banner","-i",str(p)],capture_output=True).stderr.decode("utf-8","replace")
        vs = [l.strip() for l in info.splitlines() if "Video:" in l or "Audio:" in l]
        best=None
        for k in range(2):
            t0=time.perf_counter()
            subprocess.run([FF,"-v","error","-i",str(p),"-f","null","NUL"],check=True)
            dt=time.perf_counter()-t0
            best = dt if best is None else min(best,dt)
        print("  %-3s decode(软解) 80 帧 %.3fs = %6.1f fps  | %s" % (a, best, 80/best, " ; ".join(x[:70] for x in vs)))
