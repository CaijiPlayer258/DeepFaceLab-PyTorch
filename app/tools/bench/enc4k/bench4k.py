
# -*- coding: utf-8 -*-
"""4K 导出编码器三臂对照：libx264 crf16 / h264_nvenc CQ / hevc_nvenc CQ。

口径 = 生产（app/server/backend_merge/pipeline_fused.py::_fused_pipe_cmd）逐字对齐：
  -f rawvideo -pix_fmt bgr24 -s WxH -framerate 30 -i pipe:0 -i audio -map 0:v:0 -map 1:a?
  -vf pad=ceil(iw/2)*2:ceil(ih/2)*2,format=rgb24,scale=out_range=tv
  -pix_fmt yuv420p -c:a copy -movflags +faststart -frames:v N
唯一变量 = -c:v 那一段。同进程、按轮交替。
"""
import subprocess, threading, time, json, os, sys, hashlib, statistics
from pathlib import Path
import numpy as np
import imageio_ffmpeg

FF = imageio_ffmpeg.get_ffmpeg_exe()
D = Path(__file__).resolve().parents[4] / "_enc4k_bench"   # 仓库根的 scratch 素材/产物目录（不进仓）
AUDIO = str(D / "audio.m4a")
FPS = 30.0

ARMS = {
    "A  x264 crf16":        ["-c:v", "libx264", "-crf", "16"],
    "B1 h264_nvenc CQ19":   ["-c:v", "h264_nvenc", "-preset", "p5", "-tune", "hq",
                             "-rc", "vbr", "-cq", "19", "-b:v", "0"],
    "B2 h264_nvenc CQ23":   ["-c:v", "h264_nvenc", "-preset", "p5", "-tune", "hq",
                             "-rc", "vbr", "-cq", "23", "-b:v", "0"],
    "C1 hevc_nvenc CQ19":   ["-c:v", "hevc_nvenc", "-preset", "p5", "-tune", "hq",
                             "-rc", "vbr", "-cq", "19", "-b:v", "0"],
    "C2 hevc_nvenc CQ23":   ["-c:v", "hevc_nvenc", "-preset", "p5", "-tune", "hq",
                             "-rc", "vbr", "-cq", "23", "-b:v", "0"],
}
# 诊断臂：不编码，只走管道 + 滤镜（量「写管道+swscale 的地板」）
DIAG = {"D  pipe-only(no enc)": None}

VFLT = "pad=ceil(iw/2)*2:ceil(ih/2)*2,format=rgb24,scale=out_range=tv"


def cmd_for(opts, w, h, out, n):
    base = [FF, "-y", "-hide_banner", "-loglevel", "error", "-nostats", "-nostdin",
            "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", "%dx%d" % (w, h),
            "-framerate", "%g" % FPS, "-i", "pipe:0"]
    if opts is None:                                   # 诊断臂
        return base + ["-an", "-vf", VFLT, "-c:v", "rawvideo", "-f", "null", "NUL"]
    return base + ["-i", AUDIO, "-map", "0:v:0", "-map", "1:a?", "-vf", VFLT] + opts + \
           ["-pix_fmt", "yuv420p", "-c:a", "copy", "-movflags", "+faststart",
            "-frames:v", str(int(n)), str(out)]


class GpuSampler(threading.Thread):
    """一条常驻 nvidia-smi --query-gpu -lms 200，攒 (sm%, enc%)。"""
    def __init__(self):
        super().__init__(daemon=True)
        self.stop = threading.Event(); self.s = []; self.ok = False
    def run(self):
        try:
            p = subprocess.Popen(["nvidia-smi", "--query-gpu=utilization.gpu,utilization.encoder",
                                  "--format=csv,noheader,nounits", "-lms", "200"],
                                 stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                                 text=True, bufsize=1)
        except Exception:
            return
        self.ok = True
        for line in p.stdout:
            if self.stop.is_set(): break
            q = [x.strip() for x in line.strip().split(",")]
            try:
                self.s.append((float(q[0]), float(q[1])))
            except Exception:
                pass
        p.kill()
    def result(self):
        self.stop.set(); self.join(2.0)
        if not self.s: return (float("nan"), float("nan"), 0)
        sm = statistics.mean(x[0] for x in self.s)
        en = statistics.mean(x[1] for x in self.s)
        return (sm, en, len(self.s))


class CpuSampler(threading.Thread):
    """每 20ms 采一次 ffmpeg 子进程 CPU 时间（user+sys）。"""
    def __init__(self, pid):
        super().__init__(daemon=True)
        self.pid = pid; self.stop = threading.Event(); self.last = 0.0
    def run(self):
        try: pr = psutil.Process(self.pid)
        except Exception: return
        while not self.stop.is_set():
            try:
                t = pr.cpu_times()
                self.last = t.user + t.system
            except Exception:
                break
            time.sleep(0.02)
    def total(self):
        self.stop.set(); self.join(1.0)
        return self.last

import psutil


def measure(name, opts, w, h, frames, out, do_gpu=True):
    p = subprocess.Popen(cmd_for(opts, w, h, out, len(frames)),
                         stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
                         stderr=subprocess.PIPE)
    ck = CpuSampler(p.pid); ck.start()
    gs = GpuSampler()
    if do_gpu: gs.start()
    py0 = psutil.Process().cpu_times(); py0 = py0.user + py0.system
    t0 = time.perf_counter()
    wt = 0.0; wmax = 0.0; wn = 0
    try:
        for f in frames:
            tw = time.perf_counter()
            try:
                p.stdin.write(f)
                p.stdin.flush()
            except (BrokenPipeError, ValueError, OSError):
                break
            d = time.perf_counter() - tw
            wt += d; wmax = max(wmax, d); wn += 1
    finally:
        try: p.stdin.close()
        except Exception: pass
    tail = p.stderr.read(); p.wait()
    wall = time.perf_counter() - t0
    cores_ff = ck.total() / wall if wall > 0 else 0.0
    py1 = psutil.Process().cpu_times(); py1 = py1.user + py1.system
    cores_py = (py1 - py0) / wall if wall > 0 else 0.0
    sm, en, ns = gs.result() if do_gpu else (float("nan"), float("nan"), 0)
    n = len(frames)
    mb = os.path.getsize(out) / 2**20 if opts is not None and os.path.exists(out) else 0.0
    md5 = ""
    if mb:
        h5 = hashlib.md5()
        with open(out, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 22), b""):
                h5.update(chunk)
        md5 = h5.hexdigest()
    r = dict(arm=name, w=w, h=h, n=n, wall=round(wall, 3),
             fps=round(n / wall, 2), ms_frame=round(1000 * wall / n, 1),
             write_ms=round(1000 * wt / max(1, wn), 1), write_max_ms=round(1000 * wmax, 1),
             cores_ffmpeg=round(cores_ff, 2), cores_writer=round(cores_py, 2),
             cores_total=round(cores_ff + cores_py, 2),
             gpu_sm=round(sm, 1) if sm == sm else None,
             gpu_enc=round(en, 1) if en == en else None, gpu_samples=ns,
             mb=round(mb, 2), mbps=round(mb * 8 / (n / FPS), 2) if mb else 0.0,
             md5=md5, rc=p.returncode, err=tail[:300].decode("utf-8", "replace").strip())
    print("  %-22s %6.2fs %6.2f fps  %6.1f ms/帧  写 %5.1f ms  cores ff %.2f+wr %.2f  "
          "gpu sm %s/enc %s  %7.2f MB %s" % (name, r["wall"], r["fps"], r["ms_frame"],
          r["write_ms"], r["cores_ffmpeg"], r["cores_writer"], r["gpu_sm"], r["gpu_enc"],
          r["mb"], ("md5 " + r["md5"][:8]) if r["md5"] else ""), flush=True)
    return r


def load(fn, w, h, n):
    fr = w * h * 3
    b = Path(D / fn).read_bytes()
    return [b[i * fr:(i + 1) * fr] for i in range(n)]


def run_matrix(tag, w, h, fn, n, frames_mem, rounds=2, do_gpu=True):
    print("\n### %s  %dx%d (%.2f MP)  %d 帧" % (tag, w, h, w * h / 1e6, n), flush=True)
    res = []
    for rnd in range(1, rounds + 1):
        print("- round %d" % rnd, flush=True)
        for name, opts in list(ARMS.items()) + list(DIAG.items()):
            out = D / ("out_%s_r%d_%s.mp4" % (tag, rnd, name.split()[0]))
            res.append(measure(name, opts, w, h, frames_mem, str(out), do_gpu))
            if opts is None and os.path.exists(out): os.remove(out)
    return res


if __name__ == "__main__":
    allres = {}
    # ---- 4K（2160x3840：与 3840x2160 同 8.29 MP，真实实拍素材）----
    fr = load("f_4k.raw", 2160, 3840, 80)
    allres["4k"] = run_matrix("4k", 2160, 3840, "f_4k.raw", 80, fr)
    del fr
    # ---- 1080p 原生（同一条片子，1/4 像素）----
    fr = load("f_hd.raw", 1080, 1920, 80)
    allres["hd"] = run_matrix("hd", 1080, 1920, "f_hd.raw", 80, fr)
    del fr
    # ---- 对照：容易内容 3840x2160（动画）----
    fr = load("f_easy4k.raw", 3840, 2160, 40)
    allres["easy4k"] = run_matrix("easy4k", 3840, 2160, "f_easy4k.raw", 40, fr, do_gpu=False)
    json.dump(allres, open(D / "results.json", "w"), indent=1, ensure_ascii=False)
    print("\nwritten", D / "results.json")
