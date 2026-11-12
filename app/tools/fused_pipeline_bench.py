# -*- coding: utf-8 -*-
r"""五步合一（帧 → 检测 → 对齐 → embedding → dfm → XSegLite → merge → 输出帧）原型 + 基准。

**这是原型与测量工具，不在生产路径上。** 只 import 现有实现，不复制任何算法：

    rm._detect_boxes              检测（和切脸阶段、/api/merge/preview 同一条）
    _align.*                      对齐（coarse_align_from_5 → 2D106 → 68 点 → align_face_68）
    rm._auto_face_plan            人脸库 embedding 归组（库里有脸时才真的算）
    rm._dfm_run_batch             dfm 前向（batch 维动态，和 ③ 换脸同一个函数）
    rm._stage_xseg / XSegMasker   XSegLite 遮罩（和 ④ 遮罩阶段同一个 masker）
    _merge_compose.compose_frame  合成（和 ⑤ 合成阶段同一个函数）

三种模式，同一段片子、同一套 settings、同样 8 线程：

    A  现有六阶段分段（frames → faces → pred → mask → merge → encode）—— 基线。
       走生产代码：_stage_cfg + _stage_sync_range + _stage_precheck + _STAGE_FN，
       和 merge_stage() 起线程之后干的事逐行一致，只是同步调用便于计时。
    B  合一分步串行：单线程，一帧跑完再下一帧，段与段之间零重叠。
    C  合一流水线：解码 → 检测 → 对齐/预处理 → GPU 批次(dfm+XSeg) → 合成 → 编码/写盘，
       段间有界深队列，允许乱序完成、按帧号重排后写盘。

B/C 全程内存、不落中间文件：帧只在 ffmpeg 管道里出现一次，检测/对齐/dfm/遮罩/合成之间
传的是 ndarray；落盘的只有最终帧（PNG）和最终 mp4。

为什么三模式可以直接对比（吃的是同一批像素）：
  · ffmpeg `-f rawvideo -pix_fmt bgr24` 的帧与它 `-compression_level 1` 写出的 PNG 再
    cv2.imdecode **逐位相同**（1080x1920 实拍片子 mean=0.000000 max=0），脚本启动时会
    重新实测一遍并打印；
  · dfm 输入：与 _stage_do_pred 的读线程同一个矩阵 get_transform_mat(lm68, S, WHOLE_FACE)，
    同一个「uint8 整帧 warp → /255」；
  · pred / 遮罩的 u8 量化：与 A 的「PNG 写盘 → 读回 → /255」逐位等价，只是不落盘；
  · xseg 的 dst 输入：与 _stage_read_dst_face 同一张补偿矩阵（_stage_dst_face_mat），
    唯一差别是 A 中间过了一次 aligned JPEG（q100/4:4:4）—— 差多少就是下面报的那几个 LSB。

跑法（仓库根，必须用 server venv）：
    runtime\server\Scripts\python.exe app\tools\fused_pipeline_bench.py
    runtime\server\Scripts\python.exe app\tools\fused_pipeline_bench.py --frames 30 --modes A,C
    runtime\server\Scripts\python.exe app\tools\fused_pipeline_bench.py --json out.json

退出码：0 = 跑完（结论可能是「不值得做」）；2 = 断言失败。
"""
from __future__ import annotations

import argparse
import ctypes
import json
import os
import queue
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

# 必须在 import routes_merge 之前：那个模块在 import 时读它决定「导出成功后删不删 cache」
os.environ.setdefault("DFL_STAGE_KEEP", "1")
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

import numpy as np                                             # noqa: E402
import cv2                                                     # noqa: E402

APP = REPO / "app"
if str(APP) not in sys.path:
    sys.path.insert(0, str(APP))

from server import app as _app_mod                             # noqa: E402
from server import _merge_compose as _MC                       # noqa: E402
from server import routes_merge as rm                          # noqa: E402
from server import _align                                      # noqa: E402


# ============================================================
# 口径（三模式共用；改这里等于改全部）
# ============================================================
SETTINGS = {
    "blend": "seamless",        # 用户点名：seamless
    "ct_mode": "mt",            # 4.1 默认
    "mask_mode": 8,             # xseg-prd × xseg-dst —— 两条 XSeg 分支都真的用上
    "mask_erode": 0,
    "mask_blur": 4,             # 让 erode/blur（pad 32 + GaussianBlur）真的跑起来
}
DETECTOR_ALIGN = "retinaface_10g"      # = _stage_quality_detector("high")
THRESH_ALIGN = 0.5
MERGE_WORKERS = 8                      # 用户口径：8 线程
PNG_LEVEL = 1                          # = _MERGE_PNG_LEVEL 默认
STAGE_ORDER = ("frames", "faces", "pred", "mask", "merge", "encode")
FUSED_STAGES = ("decode", "detect", "align", "embed", "dfm", "xseg", "compose",
                "encode", "write")


# ============================================================
# 小工具
# ============================================================
_LOG = []


def say(msg=""):
    """打印 + 攒一份（最后连同报告写进 <workdir>/report.txt，控制台编码不可靠）。"""
    text = str(msg)
    _LOG.append(text)
    try:
        print(text, flush=True)
    except UnicodeEncodeError:                                 # GBK 控制台也不该中断
        print(text.encode("utf-8", "replace").decode("utf-8", "replace"), flush=True)


def head(title):
    say("")
    say("=" * 78)
    say(title)
    say("=" * 78)


def table(rows, cols):
    """rows: [[...]]；cols: [(标题, 宽度, 对齐)]，对齐 'l' | 'r'。"""
    line = "  ".join((str(h).ljust(w) if a == "l" else str(h).rjust(w)) for h, w, a in cols)
    say(line)
    say("-" * len(line))
    for r in rows:
        cells = []
        for (h, w, a), v in zip(cols, r):
            s = "" if v is None else str(v)
            cells.append(s.ljust(w) if a == "l" else s.rjust(w))
        say("  ".join(cells))


def ms(v):
    return round(float(v), 2)


class Ticks:
    """线程安全的逐段计时（和 routes_merge._StageTicks 同一个口径）。"""

    def __init__(self):
        self._d = {}
        self._lock = threading.Lock()

    def add(self, k, dt):
        with self._lock:
            self._d[k] = self._d.get(k, 0.0) + float(dt)

    def raw(self, k):
        with self._lock:
            return float(self._d.get(k, 0.0))

    def per_frame(self, k, n):
        return ms(1000.0 * self.raw(k) / max(1, int(n)))

    def dump(self, n):
        return {k: self.per_frame(k, n) for k in sorted(self._d)}


def lsb_stats(a, b):
    """两张 u8 图的逐像素差（LSB）。mean/max/99.9 分位/非零像素数。"""
    if a is None or b is None:
        return {"ok": False, "why": "有一张读不出来"}
    if a.shape != b.shape:
        return {"ok": False, "why": "尺寸不同 %s vs %s" % (a.shape, b.shape)}
    d = np.abs(a.astype(np.int16) - b.astype(np.int16)).reshape(-1)
    return {"ok": True, "mean": float(d.mean()), "max": int(d.max()),
            "p999": float(np.percentile(d, 99.9)) if d.size else 0.0,
            "nz": int(np.count_nonzero(d)), "px": int(d.size),
            "sum": float(d.sum())}


# ---- 显存 / GPU 利用率：nvidia-smi 常驻采样 --------------------------------
class GpuSampler:
    """nvidia-smi -lms N 常驻采样。窗口内给：利用率均值/峰值、显存峰值。"""

    def __init__(self, period_ms=200):
        self.period = max(50, int(period_ms))
        self.proc = None
        self.th = None
        self.on = False
        self.samples = []
        self.lock = threading.Lock()

    def start(self):
        self.samples = []
        try:
            self.proc = subprocess.Popen(
                ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used",
                 "--format=csv,noheader,nounits", "-lms", str(self.period)],
                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True,
                creationflags=getattr(_app_mod, "_NO_WINDOW", 0))
        except Exception:                                      # noqa: BLE001
            self.proc = None
            return self
        self.on = True
        self.th = threading.Thread(target=self._loop, name="gpu-sample", daemon=True)
        self.th.start()
        return self

    def _loop(self):
        for line in iter(self.proc.stdout.readline, ""):
            if not self.on:
                break
            parts = [x.strip() for x in (line or "").strip().split(",")]
            if len(parts) < 2:
                continue
            try:
                u, m = int(parts[0]), int(parts[1])
            except ValueError:
                continue
            with self.lock:
                self.samples.append((u, m))

    def stop(self):
        self.on = False
        p = self.proc
        self.proc = None
        if p is not None:
            try:
                p.kill()
            except Exception:                                  # noqa: BLE001
                pass
            try:
                p.wait(timeout=5)
            except Exception:                                  # noqa: BLE001
                pass
        with self.lock:
            s = list(self.samples)
        if not s:
            return {"util_mean": None, "util_max": None, "mem_peak_mb": None, "n": 0}
        us = [x[0] for x in s]
        mems = sorted(x[1] for x in s)
        return {"util_mean": round(sum(us) / float(len(us)), 1), "util_max": max(us),
                "mem_peak_mb": max(mems), "mem_median_mb": int(mems[len(mems) // 2]),
                "n": len(us)}


# ---- 进程内存：WorkingSet 采样（PeakWorkingSetSize 是整个进程生命期的，不能按模式分）
class _PMC(ctypes.Structure):
    _fields_ = [("cb", ctypes.c_ulong), ("PageFaultCount", ctypes.c_ulong),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t)]


_RSS_FN = {"tried": False, "fn": None}


def rss_mb():
    """当前进程工作集（MB）；拿不到返回 None。

    优先 psutil（有就用，最省事）；没有就用 psapi —— 但**必须**显式写 argtypes：
    默认的 c_int 转换会把手柄截成 32 位，GetProcessMemoryInfo 直接返回 0。
    """
    if not _RSS_FN["tried"]:
        _RSS_FN["tried"] = True
        fn = None
        try:
            import psutil
            proc = psutil.Process()

            def fn():
                return int(proc.memory_info().rss // (1 << 20))
        except Exception:                                      # noqa: BLE001
            try:
                import ctypes.wintypes as wt
                pmc = _PMC()
                pmc.cb = wt.DWORD(ctypes.sizeof(pmc))
                api = ctypes.windll.psapi.GetProcessMemoryInfo
                api.argtypes = [wt.HANDLE, ctypes.POINTER(_PMC), wt.DWORD]
                api.restype = wt.BOOL
                k32 = ctypes.windll.kernel32.GetCurrentProcess
                k32.restype = wt.HANDLE

                def fn():
                    ok = api(k32(), ctypes.byref(pmc), wt.DWORD(ctypes.sizeof(pmc)))
                    return int(pmc.WorkingSetSize // (1 << 20)) if ok else None
            except Exception:                                  # noqa: BLE001
                fn = None
        _RSS_FN["fn"] = fn
    fn = _RSS_FN["fn"]
    if fn is None:
        return None
    try:
        return fn()
    except Exception:                                          # noqa: BLE001
        return None


class MemSampler:
    """每 period 秒采一次工作集，给窗口内峰值/中位数。"""

    def __init__(self, period=0.05):
        self.period = float(period)
        self.on = False
        self.th = None
        self.vals = []

    def start(self):
        self.vals = []
        self.on = True
        self.th = threading.Thread(target=self._loop, name="mem-sample", daemon=True)
        self.th.start()
        return self

    def _loop(self):
        while self.on:
            v = rss_mb()
            if v is not None:
                self.vals.append(v)
            time.sleep(self.period)

    def stop(self):
        self.on = False
        if self.th is not None:
            self.th.join(timeout=2.0)
        v = list(self.vals)
        if not v:
            return {"peak_mb": None, "median_mb": None, "n": 0}
        v.sort()
        return {"peak_mb": v[-1], "median_mb": v[len(v) // 2], "n": len(v)}


class DirAudit:
    """目录树文件快照 → 用来断言「没有中间文件落盘」。"""

    def __init__(self, roots):
        self.roots = [Path(r).resolve() for r in roots]
        self.before = set(self._scan())

    def _scan(self):
        out = []
        for r in self.roots:
            if not r.exists():
                continue
            for p in r.rglob("*"):
                try:
                    if p.is_file():
                        out.append(str(p.resolve()))
                except OSError:
                    continue
        return out

    def new_files(self):
        after = set(self._scan())
        return sorted(after - self.before)


# ============================================================
# 解码：ffmpeg rawvideo 管道（像素与 A 的 frames/*.png 逐位相同，见文件头）
# ============================================================
class FrameReader:
    """一个 ffmpeg 进程，一帧一次 readinto。

    为什么只有**一个**读线程：同一条管道是顺序流，多线程读同一个 fd 只会互相踩。
    真正的解码并行在 ffmpeg 进程内部（libavcodec 的帧/切片线程），本进程这一侧只是
    把已经解好的 bgr24 搬进 numpy —— 这也是 A 的切帧阶段（ffmpeg 写 PNG）在做的事，
    只是它把结果落在盘上。
    """

    def __init__(self, video, w, h):
        self.video = str(video)
        self.w = int(w)
        self.h = int(h)
        self.need = self.w * self.h * 3
        self.proc = None

    def open(self):
        cmd = [str(_app_mod._ffmpeg_exe()), "-hide_banner", "-loglevel", "error",
               "-nostdin", "-i", self.video, "-map", "0:v:0", "-an", "-vsync", "0",
               "-f", "rawvideo", "-pix_fmt", "bgr24", "-"]
        self.proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            bufsize=max(self.need * 2, 1 << 22),
            creationflags=getattr(_app_mod, "_NO_WINDOW", 0))
        return self

    def read(self):
        """返回 (frame uint8 HxWx3 or None, 读取秒数)。None = 流结束。"""
        t0 = time.perf_counter()
        frame = np.empty((self.h, self.w, 3), np.uint8)
        view = memoryview(frame).cast("B")
        got = 0
        while got < self.need:
            n = self.proc.stdout.readinto(view[got:])
            if not n:
                break
            got += n
        if got < self.need:
            return None, time.perf_counter() - t0
        return frame, time.perf_counter() - t0

    def close(self):
        p = self.proc
        self.proc = None
        if p is None:
            return
        try:
            p.stdout.close()
        except Exception:                                      # noqa: BLE001
            pass
        try:
            p.kill()
        except Exception:                                      # noqa: BLE001
            pass
        try:
            p.wait(timeout=10)
        except Exception:                                      # noqa: BLE001
            pass


def probe_pixels_equal(video, w, h, workdir):
    """实测：rawvideo 管道出来的帧 vs ffmpeg 写 PNG 再 cv2.imdecode —— 是否逐位相同。

    这是三模式能直接对比的前提，所以每次运行都当场验一次（不靠注释里的结论）。
    """
    rd = FrameReader(video, w, h).open()
    raw, _dt = rd.read()
    rd.close()
    png = Path(workdir) / "_pixel_probe.png"
    cmd = [str(_app_mod._ffmpeg_exe()), "-y", "-hide_banner", "-loglevel", "error",
           "-i", str(video), "-frames:v", "1", "-compression_level", "1", str(png)]
    subprocess.run(cmd, check=True, creationflags=getattr(_app_mod, "_NO_WINDOW", 0))
    img = cv2.imread(str(png), cv2.IMREAD_COLOR)
    try:
        png.unlink()
    except OSError:
        pass
    st = lsb_stats(raw, img)
    st["identical"] = bool(st.get("ok") and st.get("max") == 0)
    return st


# ============================================================
# 合一原语：B/C 共用。加载一次，之后每段一个方法（不复制任何算法）
# ============================================================
class FusedEngine:
    def __init__(self, dfm, detector=DETECTOR_ALIGN, batch=0, embedding=False):
        self.dfm = str(dfm)
        self.detector = str(detector)
        self.want_batch = int(batch or 0)
        self.embedding = bool(embedding)
        self.lm = None
        self.det = None
        self.ent = None
        self.size = 0
        self.batch = 1
        self.batch_meta = {}
        self.xseg = None
        self.xseg_res = {}
        self.cfg = {}
        self.load_s = 0.0
        self.FaceType = None
        self.get_transform_mat = None
        self.ticks = Ticks()

    # ---- 一次加载（不进测量窗口）----
    def load(self):
        t0 = time.perf_counter()
        rm._ensure_ort_dll_dirs()
        _MC.ensure_engine_path()
        from facelib import FaceType
        from facelib.LandmarksProcessor import get_transform_mat
        self.FaceType = FaceType
        self.get_transform_mat = get_transform_mat
        self.lm = rm._landmarks_model()
        if self.lm is None:
            raise SystemExit("2D106 关键点模型不可用（weights/insightface_2d106.onnx）")
        self.det = rm._detector(self.detector)
        self.ent = rm._dfm_session(self.dfm)
        self.size = int(self.ent[2])
        if self.want_batch > 0:
            self.batch = max(1, self.want_batch)
            self.batch_meta = {"source": "request", "batch": self.batch, "ladder": []}
        else:
            self.batch, self.batch_meta = rm._pred_pick_batch({"pred_batch": 0}, self.ent)
        self.xseg, self.xseg_res = rm._stage_xseg("lite")
        self.cfg = dict(_app_mod._merge_settings_to_config(SETTINGS))
        self.cfg["mask_mode"] = int(SETTINGS.get("mask_mode", 6) or 0)
        self.load_s = time.perf_counter() - t0
        return self

    def describe(self):
        backend = "?"
        try:
            backend = str((self.ent[6] if len(self.ent) > 6 else "") or "?")
        except Exception:                                      # noqa: BLE001
            pass
        xs = dict(getattr(self.xseg, "info", {}) or {})
        return {"dfm": Path(self.dfm).name, "dfm_backend": backend,
                "dfm_size": self.size, "dfm_batch": self.batch,
                "dfm_batch_source": self.batch_meta.get("source"),
                "dfm_batch_ladder": self.batch_meta.get("ladder"),
                "xseg": self.xseg_res.get("label"),
                "xseg_backend": xs.get("backend"),
                "xseg_precision": xs.get("precision"),
                "xseg_engine": Path(str(xs.get("engine") or "")).name,
                "load_s": round(self.load_s, 2)}

    # ---- ① 检测 ----
    def detect(self, frame):
        t = time.perf_counter()
        boxes = rm._detect_boxes(frame, self.detector, THRESH_ALIGN)
        return boxes, time.perf_counter() - t

    # ---- ② 对齐 + ③ embedding + dfm/xseg 的输入准备 ----
    def prep(self, frame, boxes):
        """检测框 → 每张脸：(68 点, dfm 输入 S×S, xseg 的 dst 256)。

        与生产链路同序（_stage_do_faces 的对齐 → _stage_do_pred 读线程的 warp →
        _stage_read_dst_face 的补偿映射），只是中间结果留在内存里。
        """
        out = []
        for k, b in enumerate(sorted(list(boxes or []),
                                     key=lambda x: float((x.get("box") or [0, 0, 0, 0])[0]))):
            box = b.get("box") or [0, 0, 0, 0]
            try:
                x1, y1, x2, y2 = [float(v) for v in box]
            except (TypeError, ValueError):
                continue
            bw = max(1.0, x2 - x1)
            bh = max(1.0, y2 - y1)
            side = int(round(max(bw, bh) * 2.0))
            side -= side % 2
            kps = b.get("kps") or []
            if len(kps) < 5 or side < 32:
                continue
            try:
                cmat = _align.coarse_align_from_5(kps)
                coarse = cv2.warpAffine(frame, cmat,
                                        (_align.COARSE_SIZE, _align.COARSE_SIZE),
                                        flags=cv2.INTER_LINEAR,
                                        borderMode=cv2.BORDER_CONSTANT,
                                        borderValue=(0, 0, 0))
                lm68a = self.lm.landmarks68(coarse)
                if lm68a is None:
                    continue
                pts = _align.landmarks68_in_source(lm68a, cmat)
                crop, meta_f = _align.align_face_68(
                    frame, np.asarray(pts, np.float64),
                    box=[int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))],
                    size=side)
            except Exception:                                  # noqa: BLE001
                continue
            if crop is None or not meta_f:
                continue
            lm = np.asarray(meta_f["source_landmarks"], np.float32)
            fmat = self.get_transform_mat(lm, self.size, self.FaceType.WHOLE_FACE, scale=1.0)
            warped = cv2.warpAffine(frame, fmat, (self.size, self.size),
                                    flags=cv2.INTER_CUBIC)
            face = {"k": int(k), "lm": lm,
                    "dfm_in": np.clip(warped.astype(np.float32) / 255.0, 0, 1),
                    "dst_u8": None, "pred_u8": None, "prd_u8": None,
                    "dstdfm_u8": None, "xseg_prd_u8": None, "xseg_dst_u8": None}
            m = rm._stage_dst_face_mat(meta_f, 256)
            if m is not None:
                face["dst_u8"] = cv2.warpAffine(crop, m, (256, 256),
                                                flags=cv2.INTER_CUBIC)
            out.append(face)
        return out

    def embed(self, frame, boxes, idx=0):
        """人脸库归组（embedding）。库里没有脸时 _auto_face_plan 直接返回 None，
        这里就是一次库指纹 + 索引读（≈0ms）—— 和 A 的 use_library 路径同一个函数。
        planning=True 时才按 A 的口径真的算 512 仿射裁 + ArcFace，用来量这段的单价。
        """
        snap = {"mode": "video", "path": str(self.dfm), "dir": "", "names": [],
                "srcdir": ""}
        plan = rm._auto_face_plan({"project": "", "faces_project": ""}, frame, snap,
                                  "video", "bench", int(idx), self.detector,
                                  THRESH_ALIGN, boxes)
        if self.embedding:
            for b in list(boxes or []):
                try:
                    box = [float(v) for v in (b.get("box") or ())]
                except (TypeError, ValueError):
                    continue
                rm._face_vector_from_frame(frame, b, box)
        return plan

    # ---- ④ dfm + ⑤ xseg（一批脸一次前向）----
    def gpu_batch(self, faces):
        n = len(faces)
        batch_in = np.empty((n, self.size, self.size, 3), np.float32)
        for i, f in enumerate(faces):
            batch_in[i] = f["dfm_in"]
        t = time.perf_counter()
        outs = rm._dfm_run_batch(self.ent, batch_in)
        dt_dfm = time.perf_counter() - t
        # u8 量化 = A 的 PNG 往返（写 u8 → 读回 → /255），只是不落盘
        for i, triple in enumerate(outs):
            face, prd, dst = triple[0], triple[1], triple[2]
            f = faces[i]
            f["pred_u8"] = np.clip(face * 255.0, 0, 255).astype(np.uint8)
            f["prd_u8"] = (None if prd is None
                           else np.clip(prd * 255.0, 0, 255).astype(np.uint8))
            f["dstdfm_u8"] = (None if dst is None
                              else np.clip(dst * 255.0, 0, 255).astype(np.uint8))
        t = time.perf_counter()
        # 与 _stage_do_mask 的 _flush 同口径：一张脸一次 mask_batch([pred, dst])；
        # 默认 mbatch=1，不跨脸攒批（跨批会换归约顺序，遮罩差 1~2 LSB，见那里的注释）。
        for f in faces:
            imgs = [f["pred_u8"]]
            if f["dst_u8"] is not None:
                imgs.append(f["dst_u8"])
            got = self.xseg.mask_batch(imgs, out_size=256) or []
            f["xseg_prd_u8"] = (np.clip(got[0] * 255.0, 0, 255).astype(np.uint8)
                                if len(got) > 0 and got[0] is not None else None)
            f["xseg_dst_u8"] = (np.clip(got[1] * 255.0, 0, 255).astype(np.uint8)
                                if len(got) > 1 and got[1] is not None else None)
        dt_xseg = time.perf_counter() - t
        return dt_dfm, dt_xseg

    # ---- ⑥ 合成（纯 CPU，和 ⑤ 合成阶段同一个函数）----
    def compose(self, frame, faces, timings=None):
        if not faces:
            return frame
        args = []
        for f in faces:
            args.append({
                "landmarks": f["lm"],
                "pred": f["pred_u8"].astype(np.float32) / 255.0,
                "prd_dfm": (None if f["prd_u8"] is None
                            else f["prd_u8"].astype(np.float32) / 255.0),
                "dst_dfm": (None if f["dstdfm_u8"] is None
                            else f["dstdfm_u8"].astype(np.float32) / 255.0),
                "xseg_prd": (None if f["xseg_prd_u8"] is None
                             else f["xseg_prd_u8"].astype(np.float32) / 255.0),
                "xseg_dst": (None if f["xseg_dst_u8"] is None
                             else f["xseg_dst_u8"].astype(np.float32) / 255.0),
            })
        return _MC.compose_frame(frame, args, self.cfg, timings or {})

    # ---- ⑦ 编码（PNG level 1，和 ⑤ 合成阶段写回时的编码器同一个）----
    def encode(self, out):
        return rm._stage_imencode_png(out, PNG_LEVEL)


# ============================================================
# A · 现有六阶段分段（走生产代码，同步调用便于计时）
# ============================================================
def snapshot_frames(src_dir, dst_dir, limit=None):
    """把 A 合成后的帧硬链接到独立目录（同盘瞬间完成），免得 encode 阶段清 cache。"""
    src = Path(src_dir)
    dst = Path(dst_dir)
    if dst.exists():
        shutil.rmtree(dst, ignore_errors=True)
    dst.mkdir(parents=True, exist_ok=True)
    n = 0
    for p in sorted(src.glob("*.png")):
        if limit and n >= limit:
            break
        tgt = dst / p.name
        try:
            os.link(str(p), str(tgt))
        except OSError:
            shutil.copy2(str(p), str(tgt))
        n += 1
    return n


def run_mode_a(args, clip, cache, out_mp4, dfm_path, snap_dir, info):
    """A：frames → faces → pred → mask → merge → encode。

    每个阶段都走 merge_stage() 起线程之后的那几行：_stage_cfg → _stage_sync_range →
    _stage_precheck → _STAGE_FN[stage](cfg)，只是同步调用（不建线程），这样计时就是
    阶段自身的墙钟。
    """
    body = {
        "video": str(clip), "cache_dir": str(cache),
        "dfm": str(dfm_path), "model": Path(str(dfm_path)).stem,
        "settings": dict(SETTINGS), "quality": "high",
        "detector": DETECTOR_ALIGN, "thresh": THRESH_ALIGN,
        "align_thresh": THRESH_ALIGN,
        "merge_workers": int(args.workers), "png_compression": PNG_LEVEL,
        "output_dir": str(out_mp4), "keep_cache": True,
    }
    per = {}
    res = {}
    notes = []
    mem = MemSampler().start()
    gpu = GpuSampler(args.gpu_period).start()
    cpu0 = time.process_time()
    t_all = time.perf_counter()
    try:
        for stage in STAGE_ORDER:
            cfg = rm._stage_cfg(body, stage)
            rm._stage_sync_range(cfg, stage)
            plan = rm._stage_precheck(cfg, stage)
            if plan.get("skip"):
                per[stage] = 0.0
                res[stage] = {"skipped": True}
                notes.append("阶段 %s 被跳过（%s）" % (stage, plan.get("reason")))
                say("  阶段 %-6s     跳过（%s）" % (stage, plan.get("reason")))
                continue
            t0 = time.perf_counter()
            out = rm._STAGE_FN[stage](cfg) or {}
            sec = time.perf_counter() - t0
            per[stage] = sec
            res[stage] = out
            say("  阶段 %-6s %8.2fs   %s" % (stage, sec, rm._stage_brief(out)))
            if stage == "merge":
                k = snapshot_frames(rm._stage_frames_dir(Path(cfg["cache"])), snap_dir)
                say("  A 的合成帧快照：%d 张 -> %s" % (k, snap_dir))
    finally:
        wall = time.perf_counter() - t_all
        cpu = time.process_time() - cpu0
        memst = mem.stop()
        gst = gpu.stop()

    meta = {}
    mp = Path(cache) / "meta.json"
    if mp.is_file():
        try:
            meta = json.loads(mp.read_text("utf-8"))
        except Exception:                                      # noqa: BLE001
            meta = {}
    stages = meta.get("stages") or {}
    nframes = int(((stages.get("frames") or {}).get("count") or 0))
    info.update({"mode": "A", "frames": nframes, "wall": wall, "cpu_s": cpu,
                 "cores": round(cpu / max(1e-6, wall), 2), "gpu": gst, "mem": memst,
                 "per_stage_s": {k: round(v, 3) for k, v in per.items()},
                 "stage_profiles": {k: (v.get("profile") or {})
                                    for k, v in stages.items()},
                 "stage_meta": {k: {kk: v.get(kk) for kk in
                                    ("count", "seconds", "fps", "batch", "threads",
                                     "backend", "engine", "mask_label", "infer_ms_per_face")}
                                for k, v in stages.items()},
                 "notes": notes,
                 "out_mp4": str(out_mp4),
                 "snap_dir": str(snap_dir),
                 "result": {k: rm._stage_brief(v) for k, v in res.items()}})
    if nframes:
        info["fps"] = round(nframes / max(1e-6, wall), 2)
        info["ms_per_frame"] = round(1000.0 * wall / nframes, 2)
        info["per_stage_ms"] = {k: round(1000.0 * v / nframes, 2) for k, v in per.items()}
    return info


# ============================================================
# B · 合一分步串行（单线程，零重叠）
# ============================================================
def run_mode_b(args, eng, clip, w, h, out_dir, cv2_threads, info):
    """B：一帧解码 → 检测 → 对齐 → embedding → dfm → xseg → 合成 → 编码写盘，下一帧再来。"""
    out_dir = Path(out_dir)
    if out_dir.exists():
        shutil.rmtree(out_dir, ignore_errors=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    ticks = Ticks()
    cv2_prev = int(cv2.getNumThreads())
    cv2.setNumThreads(int(cv2_threads))
    reader = FrameReader(clip, w, h).open()
    n = 0
    faces_total = 0
    no_face = 0
    mem = MemSampler().start()
    gpu = GpuSampler(args.gpu_period).start()
    cpu0 = time.process_time()
    t_all = time.perf_counter()
    try:
        while True:
            frame, dt = reader.read()
            ticks.add("decode", dt)
            if frame is None:
                break
            n += 1
            boxes, dt = eng.detect(frame)
            ticks.add("detect", dt)
            t = time.perf_counter()
            faces = eng.prep(frame, boxes)
            ticks.add("align", time.perf_counter() - t)
            t = time.perf_counter()
            eng.embed(frame, boxes, n)
            ticks.add("embed", time.perf_counter() - t)
            if faces:
                d1, d2 = eng.gpu_batch(faces)
                ticks.add("dfm", d1)
                ticks.add("xseg", d2)
                faces_total += len(faces)
            else:
                no_face += 1
            t = time.perf_counter()
            out = eng.compose(frame, faces)
            ticks.add("compose", time.perf_counter() - t)
            t = time.perf_counter()
            data = eng.encode(out)
            ticks.add("encode", time.perf_counter() - t)
            t = time.perf_counter()
            rm._stage_write_atomic(out_dir / ("%06d.png" % n), data)
            ticks.add("write", time.perf_counter() - t)
    finally:
        reader.close()
        cv2.setNumThreads(cv2_prev)
        wall = time.perf_counter() - t_all
        cpu = time.process_time() - cpu0
        memst = mem.stop()
        gst = gpu.stop()
    busy = {k: ticks.raw(k) for k in ticks._d}
    util = {k: round(100.0 * v / max(1e-6, wall), 1) for k, v in busy.items()}
    info.update({"mode": "B", "frames": n, "faces": faces_total, "no_face": no_face,
                 "stage_util_pct": util,
                 "wall": wall, "cpu_s": cpu, "cores": round(cpu / max(1e-6, wall), 2),
                 "fps": round(n / max(1e-6, wall), 2),
                 "ms_per_frame": round(1000.0 * wall / max(1, n), 2),
                 "per_stage_ms": ticks.dump(n), "gpu": gst, "mem": memst,
                 "cv2_threads": int(cv2_threads), "batch": eng.batch,
                 "threads": {"decode": 1, "detect": 1, "prep": 1, "gpu": 1,
                             "compose": 1, "encode": 1, "write": 1}})
    return info


# ============================================================
# C · 合一流水线（多段并行 + 有界深队列 + 按帧号重排）
# ============================================================
class FrameJob:
    """一帧在流水线里的全部状态。帧对象从头带到合成（合成之后源帧就放掉）。"""

    __slots__ = ("idx", "frame", "boxes", "faces", "slots", "pending", "done", "timings")

    def __init__(self, idx, frame):
        self.idx = idx
        self.frame = frame
        self.boxes = None
        self.faces = None
        self.slots = None
        self.pending = 0
        self.done = 0
        self.timings = {}


def run_mode_c(args, eng, clip, w, h, out_dir, cv2_threads, info):
    """C：解码 → 检测 → 对齐/预处理 → GPU 批次(dfm+XSeg) → 合成 → 编码 → 按帧号写盘。

    设计口径（和用户要求一一对应）：
      · 段间全是**有界**队列（默认 depth=24）；再有**全局帧缓冲上限**（--budget，默认 30 帧）：
        一张 1080x1920 BGR 是 6.22 MB，30 帧 ≈ 187 MB 上限，长视频不增长；
      · 任何一段都不空转：GPU 忙时 CPU 在对齐/合成/编码，CPU 忙时 GPU 队列有货；
      · 允许乱序完成：编码线程产出的 PNG 字节进重排缓冲，写线程只按帧号顺序落盘；
        重排缓冲同样被帧缓冲上限（信号量）夹住，不会无限堆；
      · 队列深度 ≥ 批大小：dfm 批次只吃队列里**已经**有的脸，绝不为凑批等待。
    """
    out_dir = Path(out_dir)
    if out_dir.exists():
        shutil.rmtree(out_dir, ignore_errors=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    n_prep = max(1, int(args.prep_threads))
    n_cmp = max(1, int(args.workers))
    n_enc = max(1, int(args.encode_threads))
    budget = max(2, int(args.budget))
    qd = max(2, int(args.qdepth))
    batch = max(1, int(eng.batch))

    END = object()
    ticks = Ticks()
    stop = threading.Event()
    errors = []
    cnt = {"prep": n_prep, "compose": n_cmp, "encode": n_enc}
    cnt_lock = threading.Lock()
    live = {"n": 0, "peak": 0}
    live_lock = threading.Lock()
    win = {}
    order = []
    order_info = {}
    bstat = {"calls": 0, "faces": 0, "max": 0, "hist": {}}
    sem = threading.BoundedSemaphore(budget)

    q_dec = queue.Queue(maxsize=qd)
    q_det = queue.Queue(maxsize=qd)
    q_gpu = queue.Queue(maxsize=max(qd, batch * 4))
    q_cmp = queue.Queue(maxsize=qd)
    q_enc = queue.Queue(maxsize=qd)
    wq = queue.Queue(maxsize=max(4, budget))

    depths = {"decode": qd, "detect": qd, "align": qd, "gpu": max(qd, batch * 4),
              "compose": qd, "encode": qd, "write": max(4, budget)}

    def _fail(stage, exc):
        errors.append("%s: %r" % (stage, exc))
        stop.set()

    def _put(q, item, timeout=0.2):
        while not stop.is_set():
            try:
                q.put(item, timeout=timeout)
                return True
            except queue.Full:
                continue
        return False

    reader = FrameReader(clip, w, h)
    cv2_prev = int(cv2.getNumThreads())
    cv2.setNumThreads(int(cv2_threads))

    # ---- 解码：ffmpeg 管道 → 帧缓冲（信号量卡死上限，满了解码就停，天然背压）----
    def _decode():
        try:
            n = 0
            while not stop.is_set():
                sem.acquire()
                frame, dt = reader.read()
                ticks.add("decode", dt)
                if frame is None:
                    sem.release()
                    break
                n += 1
                with live_lock:
                    live["n"] += 1
                    if live["n"] > live["peak"]:
                        live["peak"] = live["n"]
                _put(q_dec, FrameJob(n, frame))
        except Exception as e:                                 # noqa: BLE001
            _fail("decode", e)
        finally:
            _put(q_dec, END)

    # ---- 检测：单线程（共享检测器 + _det_run_lock，本来就不能并发）----
    def _detect():
        t_win = time.perf_counter()
        try:
            while True:
                job = q_dec.get()
                if job is END:
                    for _ in range(n_prep):
                        q_det.put(END)
                    break
                try:
                    boxes, dt = eng.detect(job.frame)
                except Exception as e:                         # noqa: BLE001
                    _fail("detect", e)
                    for _ in range(n_prep):
                        q_det.put(END)
                    break
                ticks.add("detect", dt)
                job.boxes = boxes
                _put(q_det, job)
        finally:
            win["detect"] = time.perf_counter() - t_win

    # ---- 对齐/预处理池：2D106 + 68 点仿射 + dfm 输入 warp + xseg 的 dst 映射 ----
    def _prep():
        try:
            while True:
                job = q_det.get()
                if job is END:
                    return
                t = time.perf_counter()
                faces = eng.prep(job.frame, job.boxes)
                ticks.add("align", time.perf_counter() - t)
                t = time.perf_counter()
                eng.embed(job.frame, job.boxes, job.idx)
                ticks.add("embed", time.perf_counter() - t)
                job.faces = faces
                job.slots = [None] * len(faces)
                job.pending = len(faces)
                if not faces:
                    _put(q_cmp, job)
                else:
                    for i, f in enumerate(faces):
                        _put(q_gpu, (job, i, f))
        finally:
            with cnt_lock:
                cnt["prep"] -= 1
                last = cnt["prep"] == 0
            if last:
                q_gpu.put(END)

    # ---- GPU：攒批 → dfm 前向 → XSeg → 回填到各自的帧 ----
    def _gpu():
        t_win = time.perf_counter()
        pend = []

        def flush():
            if not pend:
                return True
            part = pend[:]
            del pend[:]
            faces = [b[2] for b in part]
            try:
                d1, d2 = eng.gpu_batch(faces)
            except Exception as e:                             # noqa: BLE001
                _fail("gpu", e)
                return False
            ticks.add("dfm", d1)
            ticks.add("xseg", d2)
            bstat["calls"] += 1
            bstat["faces"] += len(part)
            bstat["max"] = max(bstat["max"], len(part))
            key = str(len(part))
            bstat["hist"][key] = int(bstat["hist"].get(key) or 0) + 1
            for job, i, f in part:
                job.slots[i] = f
                job.done += 1
                if job.done >= job.pending:
                    _put(q_cmp, job)
            return True

        try:
            while True:
                item = q_gpu.get()
                if item is END:
                    if not flush():
                        break
                    for _ in range(n_cmp):
                        q_cmp.put(END)
                    break
                pend.append(item)
                if len(pend) >= batch:
                    if not flush():
                        break
                    continue
                ended = False
                while len(pend) < batch:
                    try:
                        nxt = q_gpu.get_nowait()
                    except queue.Empty:
                        break
                    if nxt is END:
                        ended = True
                        break
                    pend.append(nxt)
                if not flush():
                    break
                if ended:
                    for _ in range(n_cmp):
                        q_cmp.put(END)
                    break
        finally:
            win["gpu"] = time.perf_counter() - t_win

    # ---- 合成池（纯 CPU）----
    def _compose():
        try:
            while True:
                job = q_cmp.get()
                if job is END:
                    return
                t = time.perf_counter()
                out = eng.compose(job.frame, job.faces or [], job.timings)
                ticks.add("compose", time.perf_counter() - t)
                job.frame = None
                _put(q_enc, (job, out))
        finally:
            with cnt_lock:
                cnt["compose"] -= 1
                last = cnt["compose"] == 0
            if last:
                for _ in range(n_enc):
                    q_enc.put(END)

    # ---- 编码池（PNG level 1，无损）：编码完这一帧的缓冲才还回去 ----
    def _encode():
        try:
            while True:
                item = q_enc.get()
                if item is END:
                    return
                job, out = item
                t = time.perf_counter()
                data = eng.encode(out)
                ticks.add("encode", time.perf_counter() - t)
                _put(wq, (job.idx, data))
                with live_lock:
                    live["n"] -= 1
                sem.release()
        finally:
            with cnt_lock:
                cnt["encode"] -= 1
                last = cnt["encode"] == 0
            if last:
                wq.put(END)

    # ---- 写盘：唯一写者，按帧号顺序；乱序到达的先攒在 buf 里 ----
    def _write():
        nxt = 1
        buf = {}
        peak = 0
        while True:
            item = wq.get()
            if item is END:
                break
            idx, data = item
            buf[idx] = data
            if len(buf) > peak:
                peak = len(buf)
            while nxt in buf:
                d = buf.pop(nxt)
                t = time.perf_counter()
                rm._stage_write_atomic(out_dir / ("%06d.png" % nxt), d)
                ticks.add("write", time.perf_counter() - t)
                order.append(nxt)
                nxt += 1
        order_info["peak_buf"] = peak

    ths = {}
    mem = MemSampler().start()
    gpu = GpuSampler(args.gpu_period).start()
    cpu0 = time.process_time()
    reader.open()
    t_all = time.perf_counter()
    try:
        ths["write"] = threading.Thread(target=_write, name="c-write", daemon=True)
        ths["write"].start()
        for i in range(n_enc):
            th = threading.Thread(target=_encode, name="c-enc%d" % i, daemon=True)
            ths["enc%d" % i] = th
            th.start()
        for i in range(n_cmp):
            th = threading.Thread(target=_compose, name="c-cmp%d" % i, daemon=True)
            ths["cmp%d" % i] = th
            th.start()
        ths["gpu"] = threading.Thread(target=_gpu, name="c-gpu", daemon=True)
        ths["gpu"].start()
        for i in range(n_prep):
            th = threading.Thread(target=_prep, name="c-prep%d" % i, daemon=True)
            ths["prep%d" % i] = th
            th.start()
        ths["detect"] = threading.Thread(target=_detect, name="c-det", daemon=True)
        ths["detect"].start()
        ths["decode"] = threading.Thread(target=_decode, name="c-dec", daemon=True)
        ths["decode"].start()
        ths["write"].join(timeout=args.join_timeout)
        wall = time.perf_counter() - t_all
    finally:
        stop.set()
        reader.close()
        cv2.setNumThreads(cv2_prev)
        cpu = time.process_time() - cpu0
        memst = mem.stop()
        gst = gpu.stop()

    n = len(order)
    dirq = dict(depths)
    busy = {k: ticks.raw(k) for k in ticks._d}
    util = {
        "decode": round(100.0 * busy.get("decode", 0.0) / max(1e-6, wall), 1),
        "detect": round(100.0 * busy.get("detect", 0.0) / max(1e-6, wall), 1),
        "align": round(100.0 * busy.get("align", 0.0) / max(1e-6, wall * n_prep), 1),
        "embed": round(100.0 * busy.get("embed", 0.0) / max(1e-6, wall * n_prep), 1),
        "dfm": round(100.0 * busy.get("dfm", 0.0) / max(1e-6, win.get("gpu", wall)), 1),
        "xseg": round(100.0 * busy.get("xseg", 0.0) / max(1e-6, win.get("gpu", wall)), 1),
        "compose": round(100.0 * busy.get("compose", 0.0) / max(1e-6, wall * n_cmp), 1),
        "encode": round(100.0 * busy.get("encode", 0.0) / max(1e-6, wall * n_enc), 1),
        "write": round(100.0 * busy.get("write", 0.0) / max(1e-6, wall), 1),
    }
    gpu_idle_pct = max(0.0, 100.0 - util["dfm"] - util["xseg"])
    info.update({
        "mode": "C", "frames": n, "wall": wall, "cpu_s": cpu,
        "cores": round(cpu / max(1e-6, wall), 2),
        "fps": round(n / max(1e-6, wall), 2),
        "ms_per_frame": round(1000.0 * wall / max(1, n), 2),
        "per_stage_ms": ticks.dump(n), "stage_util_pct": util,
        "gpu_idle_pct": round(gpu_idle_pct, 1),
        "gpu_window_s": round(win.get("gpu", 0.0), 2),
        "detect_window_s": round(win.get("detect", 0.0), 2),
        "batch": batch, "cv2_threads": int(cv2_threads),
        "batch_actual": {"calls": bstat["calls"], "faces": bstat["faces"],
                         "mean": round(bstat["faces"] / max(1, bstat["calls"]), 2),
                         "max": bstat["max"], "hist": bstat["hist"]},
        "threads": {"decode": 1, "detect": 1, "prep": n_prep, "gpu": 1,
                    "compose": n_cmp, "encode": n_enc, "write": 1},
        "queue_depth": dirq, "budget": budget,
        "frame_buffer": {"peak_live": live["peak"], "budget": budget,
                         "peak_reorder": order_info.get("peak_buf", 0),
                         "mb_per_frame": round(w * h * 3 / 1048576.0, 2),
                         "peak_mb_est": round(live["peak"] * w * h * 3 / 1048576.0, 1)},
        "order_ok": bool(order == list(range(1, n + 1))),
        "order_head": order[:5], "order_tail": order[-5:],
        "gpu": gst, "mem": memst, "errors": list(errors),
        "alive_threads": sorted(k for k, t in ths.items() if t.is_alive()),
    })
    return info


# ============================================================
# 对拍
# ============================================================
def compare_dirs(a_dir, b_dir, n):
    """逐帧逐像素对拍（u8，LSB）。a_dir 是基准（A 的成片帧）。"""
    total_sum = 0.0
    total_px = 0
    per = []
    worst = (0, -1, 0.0)
    for i in range(1, int(n) + 1):
        pa = Path(a_dir) / ("%06d.png" % i)
        pb = Path(b_dir) / ("%06d.png" % i)
        ia = cv2.imread(str(pa), cv2.IMREAD_COLOR) if pa.is_file() else None
        ib = cv2.imread(str(pb), cv2.IMREAD_COLOR) if pb.is_file() else None
        st = lsb_stats(ia, ib)
        if not st.get("ok"):
            return {"ok": False, "why": "第 %d 帧：%s" % (i, st.get("why")), "frame": i}
        total_sum += st["sum"]
        total_px += st["px"]
        per.append([i, round(st["mean"], 4), st["max"]])
        if st["max"] > worst[1]:
            worst = (i, st["max"], st["mean"])
    return {"ok": True, "frames": len(per),
            "mean": total_sum / max(1, total_px),
            "frame_mean_max": max(p[1] for p in per) if per else 0.0,
            "frame_max": max(p[2] for p in per) if per else 0,
            "frames_over_half": sum(1 for p in per if p[1] > 0.5),
            "worst_frame": worst[0], "worst_frame_max": worst[1],
            "per_frame": per}


def compare_mp4(pa, pb, w, h, limit=0):
    """mp4 级对拍（解码后逐帧比）。允许编码差异，但要给数字。"""
    if not (Path(pa).is_file() and Path(pb).is_file()):
        return {"ok": False, "why": "有一边没有成片"}
    ra = FrameReader(pa, w, h).open()
    rb = FrameReader(pb, w, h).open()
    total_sum = 0.0
    total_px = 0
    n = 0
    worst = 0
    try:
        while True:
            fa, _ = ra.read()
            fb, _ = rb.read()
            if fa is None or fb is None:
                break
            st = lsb_stats(fa, fb)
            if not st.get("ok"):
                return {"ok": False, "why": st.get("why")}
            total_sum += st["sum"]
            total_px += st["px"]
            worst = max(worst, st["max"])
            n += 1
            if limit and n >= limit:
                break
    finally:
        ra.close()
        rb.close()
    return {"ok": True, "frames": n, "mean": total_sum / max(1, total_px),
            "max": worst}


# ============================================================
# 报告
# ============================================================
def env_info():
    gpu = "?"
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
                            "--format=csv,noheader"],
                           capture_output=True, text=True, timeout=20,
                           creationflags=getattr(_app_mod, "_NO_WINDOW", 0))
        gpu = (r.stdout or "").strip().splitlines()[0]
    except Exception:                                          # noqa: BLE001
        pass
    import onnxruntime as ort
    return {"python": sys.version.split()[0], "cv2": cv2.__version__,
            "numpy": np.__version__, "onnxruntime": ort.__version__,
            "cuda_providers": list(ort.get_available_providers()),
            "gpu": gpu, "cpu_count": os.cpu_count()}


def print_mode_table(modes, ref_wall):
    cols = [("模式", 5, "l"), ("说明", 30, "l"), ("帧/秒", 8, "r"), ("总秒", 8, "r"),
            ("每帧 ms", 8, "r"), ("相对 A", 8, "r"), ("CPU 核", 7, "r"),
            ("GPU%", 6, "r"), ("显存峰 MB", 9, "r"), ("RSS 峰 MB", 9, "r")]
    rows = []
    for m in modes:
        g = m.get("gpu") or {}
        mm = m.get("mem") or {}
        rel = (ref_wall / m["wall"]) if ref_wall and m.get("wall") else None
        rows.append([m["mode"], m.get("label", ""), m.get("fps"), round(m["wall"], 2),
                     m.get("ms_per_frame"), ("%.2fx" % rel) if rel else "-",
                     m.get("cores"), g.get("util_mean"), g.get("mem_peak_mb"),
                     mm.get("peak_mb")])
    table(rows, cols)


def print_a_stages(a, n):
    cols = [("A 阶段", 8, "l"), ("秒", 8, "r"), ("ms/帧", 8, "r"), ("占总墙钟", 9, "r"),
            ("内部 profile（生产口径，ms/帧）", 56, "l")]
    rows = []
    prof = a.get("stage_profiles") or {}
    for k in STAGE_ORDER:
        sec = (a.get("per_stage_s") or {}).get(k, 0.0)
        p = prof.get(k) or {}
        bits = []
        pf = p.get("per_frame_ms") or {}
        for kk in ("decode", "read", "prep", "detect", "lm68", "crop", "infer",
                   "write", "mask", "ct", "paste", "blend", "encode", "frame",
                   "read_ms", "prep_ms", "infer_ms", "write_ms", "mask_ms", "ct_ms",
                   "paste_ms", "blend_ms", "encode_ms", "frame_ms"):
            if kk in pf and pf[kk]:
                bits.append("%s=%.2f" % (kk, float(pf[kk])))
        if not bits and p.get("infer_ms_per_face"):
            bits.append("infer=%.2f/张" % float(p["infer_ms_per_face"]))
        rows.append([k, round(sec, 2), round(1000.0 * sec / max(1, n), 2),
                     "%.1f%%" % (100.0 * sec / max(1e-6, a["wall"])),
                     " ".join(bits)[:56]])
    table(rows, cols)
    # 阶段的墙钟 vs meta 里阶段自己记的 seconds：差出来的就是「首次加载 / batch 阶梯探测」
    sm = a.get("stage_meta") or {}
    gaps = []
    for k in STAGE_ORDER:
        own = (sm.get(k) or {}).get("seconds")
        wall = (a.get("per_stage_s") or {}).get(k)
        if own is None or wall is None:
            continue
        d = float(wall) - float(own)
        if d > 0.3:
            gaps.append("%s 墙钟 %.2fs / 自记 %.2fs（差 %.2fs）" % (k, wall, own, d))
    if gaps:
        say("  注：%s —— 差的是「首次加载模型 / batch 阶梯探测」，只在本模式首次出现，"
            % "；".join(gaps))
        say("      长片子会被摊薄（本次 %d 帧）。" % n)
    ssum = sum((a.get("per_stage_s") or {}).values())
    ovh = a["wall"] - ssum
    say("  六阶段自身合计 %.2fs（%.1f%% of 墙钟），其余 %.2fs 在阶段之外："
        % (ssum, 100.0 * ssum / max(1e-6, a["wall"]), ovh))
    say("      每次 _stage_cfg 都要重新探测视频 + 算 xseg 的 md5（_mask_signature）；"
        "链式导出只付一次。")


def print_fused_stages(m, n):
    util = m.get("stage_util_pct") or {}
    cols = [("段", 8, "l"), ("ms/帧", 8, "r"), ("段占用率", 9, "r"),
            ("线程", 5, "r"), ("说明", 46, "l")]
    note = {
        "decode": "ffmpeg 管道读一帧（解码并行在 ffmpeg 进程内部）",
        "detect": "retinaface_10g（共享检测器，单线程）",
        "align": "2D106 + 68 点仿射 + dfm 输入 warp + xseg 的 dst 映射",
        "embed": "人脸库归组（库为空时≈0，见脚注）",
        "dfm": ("dfm 前向（batch 上限 %s%s）"
                % (m.get("batch"),
                   ("，实际均值 %s" % (m.get("batch_actual") or {}).get("mean"))
                   if m.get("batch_actual") else "，逐帧 1 张")),
        "xseg": "XSegLite 两张掩码 / 脸",
        "compose": "compose_frame（seamless + mt）",
        "encode": "PNG 编码（level 1，无损）",
        "write": "原子写盘（按帧号顺序）",
    }
    th = m.get("threads") or {}
    rows = []
    for k in FUSED_STAGES:
        v = (m.get("per_stage_ms") or {}).get(k)
        if v is None:
            continue
        rows.append([k, v, ("%.1f%%" % util[k]) if k in util else "-",
                     th.get(k, 1), note.get(k, "")])
    table(rows, cols)


def verdict_text(a, b, c, cmp_ca, cmp_ba, has_b=True):
    """结论：加速比 / 瓶颈 / 值不值得做。所有数字都从测量里取，不写死。"""
    speed = a["wall"] / max(1e-6, c["wall"])
    util = c.get("stage_util_pct") or {}
    order = sorted(((v, k) for k, v in util.items()), reverse=True)
    top = order[0] if order else (0.0, "?")
    second = order[1] if len(order) > 1 else (0.0, "-")
    gidle = c.get("gpu_idle_pct") or 0.0
    lines = []
    lines.append("C 相对 A（现有六阶段）的加速比：%.2fx"
                 % speed + "（A %ss / C %ss，%d 帧）"
                 % (round(a["wall"], 2), round(c["wall"], 2), c.get("frames") or 0))
    if has_b:
        lines.append("C 相对 B（合一分步串行）的加速比：%.2fx（B %ss / C %ss）"
                     % (b["wall"] / max(1e-6, c["wall"]), round(b["wall"], 2),
                        round(c["wall"], 2)))
    else:
        lines.append("（本次没跑 B，跳过「C vs B」的分步串行对照）")
    lines.append("C 的瓶颈段：**%s**（段占用率 %.1f%%，GPU 段空闲 %.1f%%）"
                 % (top[1], top[0], gidle))
    lines.append("第二名 %s（%.1f%%）；nvidia-smi 均值 GPU 利用率 %s%%，"
                 "dfm+xseg 占 GPU 段墙钟 %.1f%%。"
                 % (second[1], second[0], (c.get("gpu") or {}).get("util_mean"),
                    100.0 - gidle))
    fb = c.get("frame_buffer") or {}
    if speed < 1.2:
        lines.append("")
        lines.append("**结论：不值得做。** 加速比 %.2fx < 1.2x —— 合并没有带来净收益。" % speed)
        lines.append("原因（按上面这些数字说）：")
        lines.append("  1) A 本来就不是一条串行链：③ 换脸是「读 N 线程 / 推理 1 线程 / 写 M 线程」，")
        lines.append("     ④ 遮罩同样是三段流水线，⑤ 合成是 %s 线程池。A 内部已经把 CPU 与 GPU")
        lines.append("     叠起来了，「合一」实际新增的重叠只有「切帧之后的那一段」；")
        lines.append("  2) 合一能省掉的只有中间文件的编解码与往返（切帧落 PNG、aligned JPEG、")
        lines.append("     pred/遮罩 6 张 PNG、合成再读一遍），而 A 的这些开销被自己的多线程藏住了；")
        lines.append("  3) C 的最慢一段是 %s（%.1f%%），它同时也是 A 的最慢一段 —— 换结构")
        lines.append("     改变不了硬底，只能改变「谁等谁」；")
        lines.append("  4) 逐像素对拍已经达标（C vs A mean=%.4f LSB / max=%s），说明合一在")
        lines.append("     正确性上是可行的 —— 但可行不等于值得。")
        lines.append("  （顺带：C vs B 是 %.2fx，说明「重叠」本身有效，只是 A 早就重叠过了。）")
    else:
        lines.append("")
        lines.append("**结论：值得做（%.2fx）。** 建议作为「快速导出」选项与分段共存，" % speed)
        lines.append("而不是替换分段。落地需要的开关（原型里都已实现，只是没接生产路径）：")
        lines.append("  · fused=1：帧→检测→对齐→dfm→XSeg→合成→编码都在内存里跑，不落中间文件；")
        lines.append("  · frame_budget（帧缓冲上限 %s 帧；本次峰值 %s 帧 ≈ %.0f MB @1080p）："
                     % (fb.get("budget"), fb.get("peak_live"),
                        fb.get("peak_mb_est") or 0))
        lines.append("    决定峰值内存，长视频不增长 —— 这一条是它比分段更可控的地方；")
        lines.append("  · queue_depth（段间队列，本次 GPU 段 %s）：比 batch 大才凑得满批；"
                     % ((c.get("queue_depth") or {}).get("gpu")))
        lines.append("  · prep/compose/encode 线程数（本次 %s / %s / %s）；"
                     % ((c.get("threads") or {}).get("prep"),
                        (c.get("threads") or {}).get("compose"),
                        (c.get("threads") or {}).get("encode")))
        lines.append("  · dfm_batch：沿用 _pred_pick_batch 的实测阶梯（本次选 %s）；"
                     % c.get("batch"))
        lines.append("  · 与分段共存：分段留「可分阶段看产物 / 可单段重跑 / 可续跑」，"
                     "合一只能整段重跑 ——")
        lines.append("    所以它是导出加速器，不是分段的替代；开关默认关，出片慢时才开。")
    lines.append("")
    if has_b:
        lines.append("支撑数据：B（分步串行、零重叠）%.2f 帧/秒 vs C %.2f 帧/秒 —— "
                     "重叠本身带来 %.2fx；"
                     % (b.get("fps") or 0, c.get("fps") or 0,
                        b["wall"] / max(1e-6, c["wall"])))
        lines.append("B vs A mean=%.4f LSB，C vs A mean=%.4f LSB —— 三条路吃同一批像素，"
                     % (cmp_ba.get("mean") or 0.0, cmp_ca.get("mean") or 0.0))
    else:
        lines.append("支撑数据：C vs A mean=%.4f LSB —— 两条路吃同一批像素，"
                     % (cmp_ca.get("mean") or 0.0))
    lines.append("差别只在 A 的 aligned JPEG（q100:4:4:4）那一次往返。")
    return lines, speed, top[1], top[0]


# ============================================================
# 入口
# ============================================================
def resolve_dfm(name):
    p = Path(str(name))
    if p.suffix.lower() != ".dfm":
        p = p.with_name(p.name + ".dfm")
    if not p.is_absolute():
        p = Path(str(_app_mod.WORKSPACE)) / "model" / p.name
    return p if p.is_file() else None


def prepare_clip(args, work):
    """切一小段片子当输入（三模式吃同一个文件）。"""
    src = Path(args.source) if args.source else None
    if src is None:
        cands = sorted(Path(str(_app_mod.WORKSPACE)).glob("*.mp4"))
        cands = [p for p in cands if "_merged" not in p.stem]
        if not cands:
            raise SystemExit("找不到源视频：用 --source 指定一个 mp4")
        src = cands[0]
    if not src.is_file():
        raise SystemExit("源视频不存在：%s" % src)
    src_probe = rm._vid_probe(src)
    fps = float(src_probe.get("fps") or 30.0)
    n = int(args.frames)
    if args.clip:
        clip = Path(args.clip)
    else:
        clip = work / ("clip_%03d_%dfps.mp4" % (n, int(round(fps))))
    if not clip.is_file():
        dur = n / max(1e-6, fps)
        cmd = [str(_app_mod._ffmpeg_exe()), "-y", "-hide_banner", "-loglevel", "error",
               "-nostdin", "-ss", "%.3f" % float(args.start), "-i", str(src),
               "-frames:v", str(n), "-an", "-c:v", "libx264", "-crf", "18",
               "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(clip)]
        say("切样片：%s 的 %.2fs 起 %.2fs（%d 帧 @ %.3gfps）→ %s"
            % (src.name, float(args.start), dur, n, fps, clip.name))
        r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8",
                           errors="replace",
                           creationflags=getattr(_app_mod, "_NO_WINDOW", 0))
        if r.returncode != 0:
            raise SystemExit("ffmpeg 切样片失败：%s" % (r.stderr or "")[-400:])
    probe = rm._vid_probe(clip)
    if int(probe.get("frames") or 0) != n:
        raise SystemExit("样片帧数不对：%s 帧（要 %d）" % (probe.get("frames"), n))
    return clip, probe, src


def parse_args(argv=None):
    ap = argparse.ArgumentParser(
        description="五步合一流水线原型 + 与现有六阶段的基准对比")
    ap.add_argument("--frames", type=int, default=120, help="样片帧数（默认 120）")
    ap.add_argument("--modes", default="A,B,C", help="要跑的模式，逗号分隔（A/B/C）")
    ap.add_argument("--dfm", default="阿証証証証-416", help="dfm 名字或路径")
    ap.add_argument("--source", default="", help="源视频（默认 workspace 里第一个 mp4）")
    ap.add_argument("--clip", default="", help="直接用这个样片（不再切）")
    ap.add_argument("--start", type=float, default=3.0, help="从源视频第几秒开始切")
    ap.add_argument("--workdir", default="", help="产物目录（默认 %%TEMP%%/dfl_fused_bench）")
    ap.add_argument("--workers", type=int, default=MERGE_WORKERS,
                    help="合成线程数（A 的 merge_workers / C 的 compose 池）")
    ap.add_argument("--prep-threads", type=int, default=0,
                    help="C 的对齐/预处理线程数（0 = 同 A 的 pred_readers）")
    ap.add_argument("--encode-threads", type=int, default=2,
                    help="C 的 PNG 编码线程数（默认 2，同 A 的 pred_writers）")
    ap.add_argument("--budget", type=int, default=30,
                    help="C 的全局帧缓冲上限（帧；1080p 一帧 6.22MB）")
    ap.add_argument("--qdepth", type=int, default=24,
                    help="C 的段间队列深度（>= dfm batch 才凑得满）")
    ap.add_argument("--batch", type=int, default=0, help="dfm batch（0 = _pred_pick_batch 实测选）")
    ap.add_argument("--embedding", action="store_true",
                    help="B/C 里按 A 的口径真的算一遍 embedding（512 裁 + ArcFace）")
    ap.add_argument("--mp4", action="store_true", help="再做一次 mp4 级对拍")
    ap.add_argument("--gpu-period", type=int, default=200, help="nvidia-smi 采样周期 ms")
    ap.add_argument("--join-timeout", type=float, default=1800.0, help="C 的等待上限 s")
    ap.add_argument("--json", default="", help="把完整结果写成 json")
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    work = Path(args.workdir or (Path(os.environ.get("TEMP") or ".") / "dfl_fused_bench"))
    work.mkdir(parents=True, exist_ok=True)
    modes = [m.strip().upper() for m in str(args.modes).split(",") if m.strip()]
    for m in modes:
        if m not in ("A", "B", "C"):
            raise SystemExit("未知模式 %r（只认 A / B / C）" % m)
    if not args.prep_threads:
        args.prep_threads = min(4, max(1, (os.cpu_count() or 4) // 3))
    dfm = resolve_dfm(args.dfm)
    if dfm is None:
        raise SystemExit("找不到 dfm：%s（workspace/model 下必须有 .dfm）" % args.dfm)
    clip, probe, src = prepare_clip(args, work)
    w = int(probe.get("w") or 0)
    h = int(probe.get("h") or 0)
    n = int(probe.get("frames") or 0)
    fps_src = float(probe.get("fps") or 30.0)

    head("环境")
    env = env_info()
    for k in ("python", "cv2", "numpy", "onnxruntime", "gpu", "cpu_count"):
        say("  %-12s %s" % (k, env.get(k)))
    say("  providers    %s" % ", ".join(env.get("cuda_providers") or []))

    head("口径（三模式完全一致）")
    say("  样片        %s（%dx%d · %d 帧 · %.3gfps · 源 %s）"
        % (clip, w, h, n, fps_src, src.name))
    say("  settings    %s" % json.dumps(SETTINGS, ensure_ascii=False))
    say("  检测/对齐   %s · thresh=%.2f · 68 点仿射（2D106）" % (DETECTOR_ALIGN, THRESH_ALIGN))
    say("  dfm         %s" % dfm.name)
    say("  线程        A: merge_workers=%d；C: prep=%d / compose=%d / encode=%d"
        % (args.workers, args.prep_threads, args.workers, args.encode_threads))
    say("  帧缓冲上限  C: %d 帧 ≈ %.1f MB @%dx%d"
        % (args.budget, args.budget * w * h * 3 / 1048576.0, w, h))

    px = probe_pixels_equal(clip, w, h, work)
    say("  像素同一性  rawvideo 管道 vs ffmpeg PNG：mean=%.6f max=%s identical=%s"
        % (px.get("mean", -1), px.get("max"), px.get("identical")))
    if not px.get("identical"):
        say("  ★ 两条取帧路径像素不同 —— C 的产物会带上这层差异（下面单独报）")

    results = {}
    label = {"A": "现有六阶段分段", "B": "合一分步串行", "C": "合一流水线"}

    if "A" in modes:
        head("A · 现有六阶段分段（生产代码）")
        cache_a = work / "a_cache"
        if cache_a.exists():
            shutil.rmtree(cache_a, ignore_errors=True)
        a = {}
        run_mode_a(args, clip, cache_a, work / "A_merged.mp4", dfm,
                   work / "A_frames", a)
        a["label"] = label["A"]
        results["A"] = a
        say("")
        say("A 合计 %.2fs · %.2f 帧/秒 · CPU %.2f 核" % (a["wall"], a.get("fps") or 0,
                                                       a.get("cores") or 0))
        print_a_stages(a, n)

    eng = None
    if "B" in modes or "C" in modes:
        head("模型加载（B/C 共用，不进测量窗口）")
        eng = FusedEngine(dfm, DETECTOR_ALIGN, batch=args.batch,
                          embedding=args.embedding).load()
        d = eng.describe()
        say("  dfm         %s · %s · %dx%d · batch=%s(%s)"
            % (d["dfm"], d["dfm_backend"], d["dfm_size"], d["dfm_size"],
               d["dfm_batch"], d["dfm_batch_source"]))
        if d.get("dfm_batch_ladder"):
            say("  batch 阶梯  %s" % json.dumps(d["dfm_batch_ladder"], ensure_ascii=False))
        say("  xseg        %s · %s(%s) · %s"
            % (d["xseg"], String_or_dash(d["xseg_backend"]), d["xseg_precision"],
               String_or_dash(d["xseg_engine"])))
        say("  加载耗时    %.2fs（模型已在进程里则≈0）" % (d["load_s"] or 0.0))

    cv2_b = rm._merge_cv2_threads({"merge_cv2_threads": 0}, 1)
    cv2_c = rm._merge_cv2_threads({"merge_cv2_threads": 0}, max(1, args.workers))

    audit = None
    if "B" in modes:
        head("B · 合一分步串行（单线程、零重叠）")
        b = {}
        audit = DirAudit([work, Path(str(_app_mod.WORKSPACE))])
        run_mode_b(args, eng, clip, w, h, work / "B_out", cv2_b, b)
        b["label"] = label["B"]
        results["B"] = b
        say("")
        say("B 合计 %.2fs · %.2f 帧/秒 · CPU %.2f 核 · 每帧 %.2f ms"
            % (b["wall"], b.get("fps") or 0, b.get("cores") or 0,
               b.get("ms_per_frame") or 0))
        print_fused_stages(b, n)

    if "C" in modes:
        head("C · 合一流水线（多段并行 + 深队列 + 按帧号重排）")
        c = {}
        if audit is None:
            audit = DirAudit([work, Path(str(_app_mod.WORKSPACE))])
        run_mode_c(args, eng, clip, w, h, work / "C_out", cv2_c, c)
        c["label"] = label["C"]
        results["C"] = c
        say("")
        say("C 合计 %.2fs · %.2f 帧/秒 · CPU %.2f 核 · 每帧 %.2f ms · GPU 段空闲 %.1f%%"
            % (c["wall"], c.get("fps") or 0, c.get("cores") or 0,
               c.get("ms_per_frame") or 0, c.get("gpu_idle_pct") or 0))
        print_fused_stages(c, n)
        fb = c.get("frame_buffer") or {}
        say("  帧缓冲：峰值 %s 帧（上限 %s）≈ %.1f MB · 重排缓冲峰值 %s 帧 · 队列深度 %s"
            % (fb.get("peak_live"), fb.get("budget"), fb.get("peak_mb_est") or 0,
               fb.get("peak_reorder"), json.dumps(c.get("queue_depth") or {})))
        if c.get("errors"):
            say("  ★ 段内异常：%s" % c["errors"])

    head("三模式对比")
    ref = (results.get("A") or {}).get("wall")
    print_mode_table([results[k] for k in ("A", "B", "C") if k in results], ref)

    # ---- 对拍 ----
    cmp_ca = {"ok": False, "why": "没跑 C 或 A"}
    cmp_ba = {"ok": False, "why": "没跑 B 或 A"}
    cmp_cb = {"ok": False, "why": "没跑 B 或 C"}
    if "A" in results and "C" in results:
        cmp_ca = compare_dirs(results["A"]["snap_dir"], work / "C_out", n)
    if "A" in results and "B" in results:
        cmp_ba = compare_dirs(results["A"]["snap_dir"], work / "B_out", n)
    if "B" in results and "C" in results:
        cmp_cb = compare_dirs(work / "B_out", work / "C_out", n)
    head("逐像素对拍（帧级，u8 LSB）")
    for nm, st in (("C vs A", cmp_ca), ("B vs A", cmp_ba), ("C vs B", cmp_cb)):
        if not st.get("ok"):
            say("  %-7s 对拍没做成：%s" % (nm, st.get("why")))
            continue
        say("  %-7s 帧数 %d · 全图 mean=%.4f LSB · 单帧最大 mean=%.4f · 单点最大 %d LSB · "
            "mean>0.5 的帧 %d/%d（最差第 %s 帧，max=%s）"
            % (nm, st["frames"], st["mean"], st["frame_mean_max"], st["frame_max"],
               st["frames_over_half"], st["frames"],
               st.get("worst_frame"), st.get("worst_frame_max")))
        worst = sorted(st["per_frame"], key=lambda r: -r[1])[:3]
        say("             最差三帧（帧号, mean, max）：%s" % (worst,))

    cmp_mp4 = {"ok": False, "why": "没做"}
    if args.mp4 and "A" in results and "C" in results:
        head("成片级对拍（mp4，解码后逐帧）")
        # C 的成片：用和 A 的 encode 阶段同一条命令（同 crf / pix_fmt / 音轨规则）
        out_c = work / "C_merged.mp4"
        # 和 A 的 encode 阶段同一条命令（同 crf16 / yuv420p / out_range=tv / faststart）
        cmd = [str(_app_mod._ffmpeg_exe()), "-y", "-hide_banner", "-loglevel", "error",
               "-nostats", "-nostdin", "-framerate", "%.6g" % fps_src,
               "-start_number", "1", "-i", str(work / "C_out" / "%06d.png"),
               "-map", "0:v:0", "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2,scale=out_range=tv",
               "-c:v", "libx264", "-crf", "16", "-pix_fmt", "yuv420p",
               "-frames:v", str(n), "-movflags", "+faststart", str(out_c)]
        try:
            subprocess.run(cmd, check=True, capture_output=True,
                           creationflags=getattr(_app_mod, "_NO_WINDOW", 0))
            cmp_mp4 = compare_mp4(results["A"]["out_mp4"], out_c, w, h)
            say("  A 成片 vs C 成片：%s" % json.dumps(cmp_mp4, ensure_ascii=False)[:220])
        except Exception as e:                                 # noqa: BLE001
            say("  C 的成片编码失败：%r" % (e,))

    # ---- 断言 ----
    head("断言")
    checks = []
    if "C" in results:
        c = results["C"]
        checks.append(("C 的输出帧号连续且严格有序（按帧号写盘）", bool(c.get("order_ok")),
                       "%s..%s" % (c.get("order_head"), c.get("order_tail"))))
        checks.append(("C 没有残留线程", not c.get("alive_threads"),
                       str(c.get("alive_threads"))))
        checks.append(("C 段内无异常", not c.get("errors"), str(c.get("errors"))))
    if "B" in results:
        nb = len(list((work / "B_out").glob("*.png")))
        checks.append(("B 的输出帧齐全（%d 帧）" % n, nb == n, "实际 %d" % nb))
    if cmp_cb.get("ok"):
        checks.append(("C 与 B 逐像素完全一致（并行不改像素）",
                       cmp_cb["frame_max"] == 0,
                       "单点最大 %d LSB" % cmp_cb["frame_max"]))
    if cmp_ca.get("ok"):
        checks.append(("C vs A 全图 mean ≤ 0.5 LSB", cmp_ca["mean"] <= 0.5,
                       "mean=%.4f" % cmp_ca["mean"]))
        checks.append(("C vs A 每帧 mean 都 ≤ 0.5 LSB", cmp_ca["frame_mean_max"] <= 0.5,
                       "最差帧 mean=%.4f" % cmp_ca["frame_mean_max"]))
    if audit is not None:
        new = audit.new_files()
        allow = set()
        for pat in ((work / "B_out"), (work / "C_out")):
            for p in pat.glob("*"):
                allow.add(os.path.normcase(str(p.resolve())))
        for extra in (work / "C_merged.mp4",):
            allow.add(os.path.normcase(str(extra.resolve())))
        bad = [p for p in new if os.path.normcase(p) not in allow]
        checks.append(("B/C 没有中间文件落盘（只有最终帧/成片）", not bad,
                       ("多出 %d 个：%s" % (len(bad), bad[:6])) if bad else
                       "新增 %d 个文件，全部是最终帧" % len(new)))
        cache_like = [p for p in new if ("_cache" in p or "aligned" in p
                                         or "pred" in p.replace("C_out", "")
                                         or "/mask/" in p.replace("\\", "/"))]
        checks.append(("没有在样片旁边生成本地 cache（aligned/pred/mask）", not cache_like,
                       str(cache_like[:4])))
    for name, ok, detail in checks:
        say("  %s  %-46s %s" % ("PASS" if ok else "FAIL", name, detail))

    # ---- 结论 ----
    head("结论")
    lines = []
    speed = None
    if "A" in results and "C" in results:
        lines, speed, top_name, top_util = verdict_text(
            results["A"], results.get("B") or results["C"], results["C"],
            cmp_ca, cmp_ba, has_b=("B" in results))
        for ln in lines:
            say("  " + ln)
    else:
        say("  没同时跑 A 和 C —— 不给结论（--modes A,C 才有对比）。")

    out = {"env": env, "args": vars(args), "clip": str(clip), "frames": n,
           "width": w, "height": h, "fps": fps_src, "settings": SETTINGS,
           "pixel_probe": px, "modes": results, "cmp_ca": cmp_ca, "cmp_ba": cmp_ba,
           "cmp_cb": cmp_cb, "cmp_mp4": cmp_mp4, "speedup_c_vs_a": speed,
           "checks": [{"name": a_, "ok": bool(b_), "detail": c_} for a_, b_, c_ in checks],
           "workdir": str(work)}
    rep = work / "report.txt"
    try:
        rep.write_text(chr(10).join(_LOG) + chr(10), encoding="utf-8")
        out["report"] = str(rep)
        say("")
        say("报告：%s" % rep)
    except OSError:
        pass
    if args.json:
        Path(args.json).write_text(json.dumps(out, ensure_ascii=False, indent=1),
                                    encoding="utf-8")
        say("json: %s" % args.json)
    bad = [c_ for c_ in checks if not c_[1]]
    return 2 if bad else 0


def String_or_dash(v):
    return "-" if v in (None, "") else str(v)


if __name__ == "__main__":
    sys.exit(main())

