"""4.1 合成器 → MergeStudio 导出管线的桥。

4.1 以前调 4.0 的 `main.py merge`，那条路是死的：mainscripts/Merger.py 第一行
依赖 `from merger import ...`，而 4.0 根目录下没有 merger 这个包 —— 真正的引擎是
独立的 MergeStudio/core/merger.py（一个模块，不是带 FrameInfo/MergerConfig 的包），
所以 `import merger` 必然 ModuleNotFoundError，4.1 那边只能返回 501。

这里改成直接调 MergeStudio 自己的 run_export_pipeline()。它自带 DFL 项目模式：
项目目录里有 aligned/（DFLJPG）时跳过人脸检测，直接从元数据建 face_data，
合成结果写进 merged/，最后用 ffmpeg 编码成视频。

4.1 的落盘命名（frame_000000.jpg + <dir>_aligned）不符合 DFL 项目的约定
（<proj>/<纯数字>.jpg + <proj>/aligned/），而且 _stage2_read_dfl_aligned 会
`int(source_filename 的 stem)` —— 所以这里先建一个 staging 项目目录：
帧用硬链接挂成 00000000.jpg，DFLJPG 也挂过去但把 source_filename 改写成分级名。
硬链接同盘零拷贝，跑完整个 staging 目录删掉。

导出层的职责（这些都是 4.0 管线自己不管的事）：
  · 参数归一 + 生效核对：4.1 界面送来的键名五花八门（mask_erode / erode_mask /
    erode_mask_modifier…），引擎真正读的只有 export_pipeline._phase_merge_worker
    里那几个 cfg 键。这里统一归一，并把"收下了但引擎不用"的参数（face_type、
    original/raw-*/debug 模式）在 @@PARAMS 里明确标成不生效。
  · XSeg 注入：引擎里 XSeg 路径写死成 <engine>/workspace/model/XSegLite/xseglite.onnx，
    `--xseg` 根本没人读；4.1 的模型在 4.1/workspace 下，于是 xseg_path=None，
    XSeg 遮罩模式(6/7/8/9)静默退化成"整块矩形贴脸"。这里把写死路径重定向到
    staging 目录里的真模型。
  · 收尾编码：不用 4.0 的 _stage5_encode_video（它 ffprobe 取 fps/码率、-map 1:a:0
    从占位源视频取音轨，再加 -shortest 会把成片截成占位片那么长）。这里自己拼
    ffmpeg 命令：先 -encoders + 试编码确认编码器可用，-pix_fmt yuv420p、显式 -framerate/-r、
    crf/码率、+faststart、原音轨（copy，失败退 aac，没音轨安全跳过跳）。
  · 进度：把 6 个 stage 的分段进度折算成单调递增的全局百分比，节流后输出 @@P。
  · 进程与清理：Windows 上把本进程绑进 KILL_ON_JOB_CLOSE 的 Job，父进程
    TerminateProcess 时 ffmpeg / multiprocessing worker 会被内核一起收掉；
    收到信号则 taskkill /T 自清，临时目录在退出路径上一定删。

用法（必须用 4.0 的 python 跑，工作目录随意，脚本会自己 sys.path）：
  python _merge_runner.py --frames DIR --aligned DIR --dfm X.dfm \
      --out Y.mp4 [--config '{...}'] [--xseg X.onnx] [--workers N] \
      [--source-video Z.mp4] [--encoder libx264] [--crf 17]

多模型（settings.models 契约，细节见下面「多模型」一节）：
  python _merge_runner.py --frames DIR --aligned DIR --out Y.mp4 \
      --settings settings.json          # settings["models"] = [{name, dfm, faces}]
  python _merge_runner.py --frames DIR --aligned DIR --out Y.mp4 \
      --dfm A.dfm --dfm-faces 0 --dfm B.dfm --dfm-faces 1
  等价写法：--dfm-models '[{"name":"张元英","dfm":"A.dfm","faces":[0]}, ...]'
单模型（只给一个 --dfm、不给 faces）与改动前逐字一致。

进度按行输出，父进程解析：
  @@P <stage> <pct 0..1> <msg>
  @@PARAMS <json>          参数生效报告
  @@DONE <输出文件>
  @@ERR <原因>
"""
from __future__ import annotations

import argparse
import atexit
import hashlib
import json
import os
import pickle
import shutil
import signal
import struct
import subprocess
import sys
import tempfile
import threading
import time
import traceback
from pathlib import Path

IMG_EXT = (".jpg", ".jpeg", ".png")

# GUI 拉起时不能弹黑框（父进程没有控制台，子进程默认会新建一个）
_NO_WINDOW = 0x08000000 if os.name == "nt" else 0
# 给 ffmpeg 单独开一个进程组：取消时能整组收掉（POSIX 用 killpg，Windows 走 taskkill /T）
_NEW_GROUP = 0x00000200 if os.name == "nt" else 0


# ---------------------------------------------------------------------------
# ORT 的 CUDA provider 得先在 DLL 搜索路径里找到 cuBLAS / cuDNN
# ---------------------------------------------------------------------------
# onnxruntime_providers_cuda.dll 依赖 cublasLt64_12.dll。找不到时 ORT **不抛异常**，
# 只在 warning 里留一句（实测原文）：
#   Error loading "...onnxruntime/capi/onnxruntime_providers_cuda.dll" which
#   depends on "cublasLt64_12.dll" which is missing. (Error 126: "找不到指定的模块。")
# 然后所有 DFM 会话静默跑 CPU —— 单张 140 ms(CPU) vs 11 ms(GPU)，差 12 倍。
# 这个 DLL 就在 <sys.prefix>/Lib/site-packages/torch/lib 里（torch 自带 cuBLAS + cuDNN 9），
# 只是不在 PATH 上。所以这里按 sys.prefix 拼路径，**不 import torch**：那个 import 要
# 2 秒多，而且它自己也可能因为缺 DLL 直接抛出来，反而把这步拖垮。
_ORT_DLL_DIR = None
_ORT_DLL_DONE = False
_ORT_DLL_COOKIES = []            # add_dll_directory 的句柄：对象一被回收，目录就被摘掉


def _setup_ort_dll_paths():
    """把 torch/lib 塞进本进程的 DLL 搜索路径（只做一次，任何异常都吞掉）。"""
    global _ORT_DLL_DIR, _ORT_DLL_DONE
    if _ORT_DLL_DONE:
        return _ORT_DLL_DIR
    _ORT_DLL_DONE = True
    try:
        seen = set()
        for prefix in (sys.prefix, getattr(sys, "base_prefix", "")):
            if not prefix or prefix in seen:
                continue
            seen.add(prefix)
            lib = Path(prefix) / "Lib" / "site-packages" / "torch" / "lib"
            try:
                if not lib.is_dir():
                    continue
            except OSError:
                continue
            try:
                _ORT_DLL_COOKIES.append(os.add_dll_directory(str(lib)))
            except Exception:
                pass
            # PATH 也要前置：add_dll_directory 只对带 LOAD_LIBRARY_SEARCH_* 的
            # LoadLibraryEx 生效，ORT 加载 provider DLL 走的是普通搜索顺序。
            try:
                os.environ["PATH"] = str(lib) + os.pathsep + os.environ.get("PATH", "")
            except Exception:
                pass
            _ORT_DLL_DIR = str(lib)
            break
    except Exception:
        pass
    return _ORT_DLL_DIR


# 模块一导入就做：必须早于任何 ORT 会话（model_loader / 引擎 worker 里的 session）
_setup_ort_dll_paths()

# 4.0 的合成管线自己 CreateProcess ffmpeg，不认我们的 creationflags；
# 装个全局补丁把整棵进程树都盖住。
try:
    from ._nowin import install as _install_nowin
except ImportError:
    from _nowin import install as _install_nowin
_install_nowin()


def _say(msg):
    try:
        print(msg, flush=True)
    except Exception:
        pass


def _prog_raw(stage, pct, msg=""):
    print("@@P %d %.4f %s" % (int(stage), float(pct), msg), flush=True)


# ---------- 进度 ----------
# 引擎的 stage：0 抽帧 1 读对齐 2 匹配身份 3 换脸 4 遮罩 5 合并 6 编码。
# 它每次给的是"本段百分比"，直接透传会让进度条在每个 stage 归零、并且中途就
# 撞到 100%。这里按权重折算成全局百分比，并保证单调不回退。
_STAGE_WEIGHT = ((0, 2.0), (1, 3.0), (2, 2.0), (3, 33.0), (4, 20.0), (5, 30.0), (6, 10.0))


class _Progress:
    def __init__(self):
        self._lock = threading.Lock()
        self.stage = 0
        self.overall = 0.0
        self._t = 0.0
        self._done = False

    @staticmethod
    def _span(stage):
        base = 0.0
        for s, w in _STAGE_WEIGHT:
            if s == stage:
                return base / 100.0, w / 100.0
            base += w
        return 1.0, 0.0

    def set(self, stage, pct, msg="", force=False):
        try:
            stage = max(0, min(int(stage), 6))
            pct = max(0.0, min(1.0, float(pct or 0.0)))
        except (TypeError, ValueError):
            return
        base, span = self._span(stage)
        overall = max(self.overall, min(1.0, base + span * pct))
        with self._lock:
            if self._done:
                return
            now = time.time()
            if not force and overall < 1.0 and (overall - self.overall) < 0.004 and (now - self._t) < 0.5:
                return
            self.stage = stage
            self.overall = overall
            self._t = now
        _prog_raw(stage, overall, str(msg or "")[:160])

    def done(self, msg="完成"):
        with self._lock:
            if self._done:
                return
            self._done = True
            self.stage = 6
            self.overall = 1.0
        _prog_raw(6, 1.0, msg)

    def fail(self, msg=""):
        with self._lock:
            self._done = True
            st, ov = self.stage, self.overall
        _prog_raw(st, ov, ("失败: " + str(msg or ""))[:180])


PROG = _Progress()


# ---------- 子进程登记 / 取消 / 清理 ----------
_CHILDREN = []
_CHILDREN_LOCK = threading.Lock()
_CANCEL = threading.Event()
_JOB = None                      # Job 句柄：进程活着就攥住，退出时内核自动收树
_STATE = {"work": None, "done": False, "out": None}


def _register(proc):
    with _CHILDREN_LOCK:
        _CHILDREN.append(proc)


def _unregister(proc):
    with _CHILDREN_LOCK:
        try:
            _CHILDREN.remove(proc)
        except ValueError:
            pass


def _kill_pid_tree(pid):
    """Windows: taskkill /F /T 连孙子一起收；POSIX: killpg。"""
    if not pid:
        return
    if os.name == "nt":
        try:
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(int(pid))],
                           capture_output=True, creationflags=_NO_WINDOW, timeout=15)
            return
        except Exception:
            pass
    else:
        try:
            os.killpg(os.getpgid(int(pid)), signal.SIGKILL)
            return
        except Exception:
            pass
    try:
        os.kill(int(pid), signal.SIGTERM)
    except Exception:
        pass


def _kill_children():
    with _CHILDREN_LOCK:
        procs = list(_CHILDREN)
        _CHILDREN.clear()
    for p in procs:
        try:
            if p.poll() is None:
                _kill_pid_tree(p.pid)
        except Exception:
            pass
        try:
            p.kill()
        except Exception:
            pass


def _install_job_object():
    """把本进程绑进一个 KILL_ON_JOB_CLOSE 的 Job Object（仅 Windows）。

    取消合成时 4.1 走的是 proc.terminate()（TerminateProcess）：没有 atexit、
    没有信号处理器，ffmpeg 和 multiprocessing 的 worker 会变成孤儿继续吃
    CPU / 显存 / 帧文件句柄，临时目录也删不掉。绑进 Job 后，本进程一死 Job 句柄
    随之关闭，内核顺手把整棵树杀掉。
    """
    global _JOB
    if os.name != "nt":
        return False
    try:
        import ctypes
        from ctypes import wintypes as _wt

        class _IO_COUNTERS(ctypes.Structure):
            _fields_ = [("ReadOperationCount", ctypes.c_uint64),
                        ("WriteOperationCount", ctypes.c_uint64),
                        ("OtherOperationCount", ctypes.c_uint64),
                        ("ReadTransferCount", ctypes.c_uint64),
                        ("WriteTransferCount", ctypes.c_uint64),
                        ("OtherTransferCount", ctypes.c_uint64)]

        class _BASIC_LIMIT(ctypes.Structure):
            _fields_ = [("PerProcessUserTimeLimit", ctypes.c_int64),
                        ("PerJobUserTimeLimit", ctypes.c_int64),
                        ("LimitFlags", _wt.DWORD),
                        ("MinimumWorkingSetSize", ctypes.c_size_t),
                        ("MaximumWorkingSetSize", ctypes.c_size_t),
                        ("ActiveProcessLimit", _wt.DWORD),
                        ("Affinity", ctypes.c_size_t),
                        ("PriorityClass", _wt.DWORD),
                        ("SchedulingClass", _wt.DWORD)]

        class _EXT_LIMIT(ctypes.Structure):
            _fields_ = [("BasicLimitInformation", _BASIC_LIMIT),
                        ("IoInfo", _IO_COUNTERS),
                        ("ProcessMemoryLimit", ctypes.c_size_t),
                        ("JobMemoryLimit", ctypes.c_size_t),
                        ("PeakProcessMemoryUsed", ctypes.c_size_t),
                        ("PeakJobMemoryUsed", ctypes.c_size_t)]

        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        k32.CreateJobObjectW.restype = _wt.HANDLE
        k32.CreateJobObjectW.argtypes = (_wt.LPVOID, _wt.LPCWSTR)
        k32.SetInformationJobObject.restype = _wt.BOOL
        k32.SetInformationJobObject.argtypes = (_wt.HANDLE, ctypes.c_int, _wt.LPVOID, _wt.DWORD)
        k32.AssignProcessToJobObject.restype = _wt.BOOL
        k32.AssignProcessToJobObject.argtypes = (_wt.HANDLE, _wt.HANDLE)
        k32.GetCurrentProcess.restype = _wt.HANDLE

        job = k32.CreateJobObjectW(None, None)
        if not job:
            return False
        info = _EXT_LIMIT()
        info.BasicLimitInformation.LimitFlags = 0x2000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        if not k32.SetInformationJobObject(job, 9, ctypes.byref(info), ctypes.sizeof(info)):
            return False
        if not k32.AssignProcessToJobObject(job, k32.GetCurrentProcess()):
            # 父进程已经在别的 Job 里且不允许嵌套（Win7）—— 忽略，退回信号/taskkill
            return False
        _JOB = job
        return True
    except Exception:
        return False


def _cleanup_work():
    work = _STATE.get("work")
    _STATE["work"] = None
    if not work:
        return
    p = Path(str(work))
    if not p.is_dir():
        return
    for _ in range(3):
        shutil.rmtree(str(p), ignore_errors=True)
        if not p.exists():
            return
        time.sleep(0.3)
    if p.exists():
        _say("[merge] 临时目录没删掉（有句柄占用）: %s" % p)


def _install_cleanup():
    def _on_signal(signum, _frame):
        _CANCEL.set()
        _say("[merge] 收到信号 %s：杀子进程 + 清临时目录" % signum)
        _kill_children()
        _cleanup_work()
        try:
            print("@@ERR 已取消（signal %s）" % signum, flush=True)
        except Exception:
            pass
        os._exit(130)

    for _name in ("SIGINT", "SIGTERM", "SIGBREAK", "SIGHUP"):
        _sig = getattr(signal, _name, None)
        if _sig is None:
            continue
        try:
            signal.signal(_sig, _on_signal)
        except Exception:
            pass

    def _atexit():
        _kill_children()
        if not _STATE.get("done"):
            _cleanup_work()

    atexit.register(_atexit)


# ---------- DFLJPG 元数据重写（图像字节原样搬运，不重新编码） ----------
def _split_app15(data: bytes):
    """返回 (去掉 APP15 后的 JPEG, APP15 的 pickle 负载或 None)。"""
    if not data.startswith(b"\xff\xd8"):
        return data, None
    out = bytearray(data[:2])
    payload = None
    i = 2
    n = len(data)
    while i + 4 <= n:
        if data[i] != 0xFF:
            break
        marker = data[i + 1]
        if marker in (0xD8, 0xD9) or 0xD0 <= marker <= 0xD7:
            out += data[i:i + 2]
            i += 2
            continue
        seg_len = struct.unpack(">H", data[i + 2:i + 4])[0]
        seg = data[i + 4:i + 2 + seg_len]
        if marker == 0xEF:
            # APP15（DFL 元数据）整段摘掉 —— 连长度头一起，不能只摘身体
            payload = bytes(seg)
        else:
            out += data[i:i + 2]
            out += data[i + 2:i + 2 + seg_len]
        i += 2 + seg_len
        if marker == 0xDA:          # SOS 之后是熵编码数据，整段照抄
            out += data[i:]
            return bytes(out), payload
    return bytes(out) + data[i:], payload


def _retag_source_filename(src: Path, dst: Path, new_name: str):
    """把 DFLJPG 里的 source_filename 换成 staging 里的帧名。"""
    data = src.read_bytes()
    body, payload = _split_app15(data)
    if payload is None:
        shutil.copy2(src, dst)
        return False
    try:
        meta = pickle.loads(payload)
    except Exception:
        shutil.copy2(src, dst)
        return False
    meta["source_filename"] = new_name
    pkl = pickle.dumps(meta, protocol=2).replace(b"numpy._core", b"numpy.core")
    app15 = b"\xff\xef" + struct.pack(">H", len(pkl) + 2) + pkl
    sos = body.find(b"\xff\xda")
    if sos <= 0:
        blob = body + app15
    else:
        blob = body[:sos] + app15 + body[sos:]
    dst.write_bytes(blob)
    return True


def _link_or_copy(src: Path, dst: Path):
    try:
        os.link(str(src), str(dst))
    except Exception:
        shutil.copy2(str(src), str(dst))


def stage_project(frames_dir: Path, aligned_dir: Path, work: Path):
    """把 4.1 的命名搭成 DFL 项目布局，返回项目目录。"""
    proj = work / "proj"
    if proj.exists():
        shutil.rmtree(proj, ignore_errors=True)
    (proj / "aligned").mkdir(parents=True, exist_ok=True)

    frames = sorted(p for p in frames_dir.iterdir()
                    if p.is_file() and p.suffix.lower() in IMG_EXT)
    if not frames:
        raise RuntimeError("帧目录里没有图片: %s" % frames_dir)

    name_to_idx = {}
    total = len(frames)
    step = max(1, total // 20)          # 每 N 帧报一次，别把管道刷爆
    for idx, fp in enumerate(frames):
        staged = "%08d%s" % (idx, fp.suffix.lower().replace(".jpeg", ".jpg"))
        _link_or_copy(fp, proj / staged)
        name_to_idx[fp.name] = idx
        if (idx + 1) % step == 0:
            PROG.set(0, 0.05 + 0.10 * (idx + 1) / total, "挂载帧 %d/%d" % (idx + 1, total))
    _say("[stage] 帧 %d 张 → %s" % (len(frames), proj))

    jpgs = sorted(aligned_dir.glob("*.jpg")) if aligned_dir.is_dir() else []
    if not jpgs:
        raise RuntimeError("对齐目录里没有 DFLJPG: %s" % aligned_dir)

    kept = skipped = 0
    for fp in jpgs:
        data = fp.read_bytes()
        _, payload = _split_app15(data)
        if payload is None:
            skipped += 1
            continue
        try:
            meta = pickle.loads(payload)
        except Exception:
            skipped += 1
            continue
        src_name = meta.get("source_filename") or ""
        idx = name_to_idx.get(Path(src_name).name)
        if idx is None:
            # 兜底：000004_0.jpg → frame_000004.jpg
            stem = fp.stem
            if "_" in stem:
                stem = stem.rsplit("_", 1)[0]
            cand = [v for k, v in name_to_idx.items() if Path(k).stem.endswith(stem)]
            idx = cand[0] if cand else None
        if idx is None:
            skipped += 1
            continue
        face_idx = 0
        if "_" in fp.stem:
            try:
                face_idx = int(fp.stem.rsplit("_", 1)[1])
            except ValueError:
                face_idx = 0
        new_name = "%08d.jpg" % idx
        _retag_source_filename(fp, proj / "aligned" / ("%08d_%d.jpg" % (idx, face_idx)), new_name)
        kept += 1
        if kept % 200 == 0:
            PROG.set(0, 0.15 + 0.05 * kept / max(1, len(jpgs)), "对齐脸 %d 张" % kept)
    _say("[stage] 对齐脸 %d 张（跳过 %d）" % (kept, skipped))
    if not kept:
        raise RuntimeError("没有任何对齐脸能对上帧（source_filename 与帧名不匹配）")
    return proj


# ---------- ffmpeg / 编码器 ----------
def _ascii_ok(p) -> bool:
    try:
        str(p).encode("ascii")
        return True
    except Exception:
        return False


def _resolve_ffmpeg(explicit_ffmpeg: str, explicit_ffprobe: str, dfl_root: Path):
    """找出可用的 ffmpeg / ffprobe：显式参数 > 引擎自带 > 4.1 的 imageio-ffmpeg > PATH。

    注意引擎 venv 里没装 imageio_ffmpeg（只有 server venv 有），所以光靠
    `import imageio_ffmpeg` 会找不到；这里把 4.1 的 binaries 目录也扫一遍。
    """
    cands = []
    if explicit_ffmpeg:
        cands.append(Path(explicit_ffmpeg))
    cands.append(dfl_root / "ffmpeg" / "ffmpeg.exe")
    here = Path(__file__).resolve()
    for root in (here.parents[2] if len(here.parents) > 2 else here.parent,
                 here.parents[1] if len(here.parents) > 1 else here.parent,
                 here.parent):
        for sub in ("server", "engine"):
            d = root / "runtime" / sub / "Lib" / "site-packages" / "imageio_ffmpeg" / "binaries"
            if d.is_dir():
                cands.extend(sorted(d.glob("ffmpeg*.exe"), reverse=True))
    try:
        import imageio_ffmpeg
        cands.append(Path(imageio_ffmpeg.get_ffmpeg_exe()))
    except Exception:
        pass
    which = shutil.which("ffmpeg")
    if which:
        cands.append(Path(which))

    ffmpeg = next((str(c) for c in cands if c and Path(c).is_file()), "")
    if not ffmpeg:
        raise RuntimeError("找不到 ffmpeg（4.1 的 imageio-ffmpeg 也没有？）")

    probes = []
    if explicit_ffprobe:
        probes.append(Path(explicit_ffprobe))
    probes.append(Path(ffmpeg).with_name("ffprobe.exe"))
    probes.append(Path(ffmpeg).parent / "ffprobe.exe")
    which_p = shutil.which("ffprobe")
    if which_p:
        probes.append(Path(which_p))
    ffprobe = next((str(c) for c in probes if c and Path(c).is_file()), "")
    return ffmpeg, ffprobe


def _list_encoders(ffmpeg):
    """ffmpeg -encoders 里有名字的编码器集合（空集 = 读不到）。"""
    try:
        r = subprocess.run([ffmpeg, "-hide_banner", "-encoders"],
                           capture_output=True, text=True, errors="replace",
                           creationflags=_NO_WINDOW, timeout=30)
    except Exception:
        return set()
    names = set()
    for line in (r.stdout or "").splitlines():
        parts = line.split()
        if len(parts) >= 2 and len(parts[0]) == 6 and parts[0][0] in "VASD":
            names.add(parts[1])
    return names


def _probe_encoder(ffmpeg, enc):
    """真编一帧：`-encoders` 里列着但驱动/显卡不匹配时，这里会立刻失败。"""
    cmd = [ffmpeg, "-hide_banner", "-v", "error",
           "-f", "lavfi", "-i", "color=c=black:s=256x256:r=25:d=0.2",
           "-frames:v", "1", "-c:v", enc, "-pix_fmt", "yuv420p", "-f", "null", "-"]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, errors="replace",
                           creationflags=_NO_WINDOW, timeout=40)
        return r.returncode == 0
    except Exception:
        return False


def _pick_encoder(ffmpeg, want):
    """返回 (实际用的编码器, 说明列表)。先查 -encoders，再真编一帧确认。"""
    notes = []
    want = (want or "").strip() or "libx264"
    avail = _list_encoders(ffmpeg)
    if not avail:
        notes.append("读不到 ffmpeg -encoders 输出，先按 %s 试（失败会自动退回 libx264）" % want)
        return want, notes
    if want in avail:
        if _probe_encoder(ffmpeg, want):
            notes.append("%s 可用（-encoders 列出 + 试编码通过）" % want)
            return want, notes
        notes.append("%s 在 -encoders 里但试编码失败（没有对应 GPU / 驱动不匹配）→ 退回软编" % want)
    else:
        notes.append("%s 不在 ffmpeg -encoders 里 → 退回软编" % want)
    for cand in ("libx264", "libx265", "mpeg4"):
        if cand in avail:
            notes.append("实际使用 %s" % cand)
            return cand, notes
    notes.append("没有可用的软编器，仍按 libx264 试")
    return "libx264", notes


def _quality_args(enc, crf, bitrate):
    e = (enc or "").lower()
    if "nvenc" in e:
        preset = ["-preset", "p5"]
    elif e.startswith(("libx26", "libsvt", "libaom", "libvpx")):
        preset = ["-preset", "veryfast"]
    else:
        preset = []
    if bitrate:
        return preset + ["-b:v", str(bitrate)]
    if "nvenc" in e:
        return preset + ["-rc", "vbr", "-cq", str(int(crf)), "-b:v", "0"]
    if "qsv" in e:
        return ["-global_quality", str(int(crf))]
    if e.startswith(("libx26", "libsvt", "libaom", "libvpx")):
        return preset + ["-crf", str(int(crf))]
    if "amf" in e:
        return preset + ["-quality", "balanced"]
    if "videotoolbox" in e:
        return ["-q:v", "65"]
    return preset


def _has_audio(ffmpeg, video, ffprobe=""):
    if not video:
        return False
    p = Path(str(video))
    if not p.is_file():
        return False
    if ffprobe:
        try:
            r = subprocess.run([ffprobe, "-v", "error", "-select_streams", "a",
                                "-show_entries", "stream=index", "-of", "csv=p=0", str(p)],
                               capture_output=True, text=True, errors="replace",
                               creationflags=_NO_WINDOW, timeout=30)
            return bool((r.stdout or "").strip())
        except Exception:
            pass
    try:
        r = subprocess.run([ffmpeg, "-hide_banner", "-i", str(p)],
                           capture_output=True, text=True, errors="replace",
                           creationflags=_NO_WINDOW, timeout=30)
        return "Audio:" in ((r.stderr or "") + (r.stdout or ""))
    except Exception:
        return False


def _locate_sequence(frames, scratch_root):
    """返回 (image2 pattern, 起始编号, 总帧数)。

    编号有洞时 ffmpeg 的 image2 会在第一个洞停住（成片被截短），所以有洞就先
    顺序重编到 scratch 目录里 —— 用硬链接，不复制像素。
    """
    ext = frames[0].suffix.lower()
    stems = [f.stem for f in frames]
    if all(s.isdigit() for s in stems):
        nums = sorted(int(s) for s in stems)
        pad = len(stems[0])
        if nums[-1] - nums[0] + 1 == len(nums):
            pattern = str(Path(frames[0].parent) / ("%0" + str(pad) + "d" + ext))
            return pattern, nums[0], len(nums)
    seq = Path(scratch_root) / "encode_seq"
    if seq.exists():
        shutil.rmtree(str(seq), ignore_errors=True)
    seq.mkdir(parents=True, exist_ok=True)
    for i, f in enumerate(frames):
        _link_or_copy(f, seq / ("%08d%s" % (i + 1, ext)))
    _say("[encode] 帧编号不连续，重编 %d 张 → %s" % (len(frames), seq))
    return str(seq / ("%08d" + ext)), 1, len(frames)


def _build_encode_cmd(ffmpeg, enc, audio_mode, audio_src, fps, pattern, first, out,
                       crf, bitrate, total=0):
    cmd = [ffmpeg, "-y", "-hide_banner", "-loglevel", "error", "-nostats",
           "-framerate", "%.6g" % fps, "-start_number", str(int(first)), "-i", str(pattern)]
    if audio_mode and audio_src:
        # 音轨只读到画面结束的位置（-t 放在 -i 前 = 输入选项，只作用于音轨输入）。
        # 这样既不会像 -shortest 那样把成片截短，也不会留下几十毫秒的音轨尾巴。
        if total > 0 and audio_mode == "copy":
            cmd += ["-t", "%.3f" % (total / float(max(1e-6, fps)) + 0.02)]
        cmd += ["-i", str(audio_src)]
    cmd += ["-map", "0:v:0"]
    if audio_mode and audio_src:
        cmd += ["-map", "1:a:0"]
    # 奇数宽高在 yuv420p + libx264 下会被直接拒绝（width not divisible by 2）
    cmd += ["-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2"]
    cmd += ["-c:v", enc] + _quality_args(enc, crf, bitrate)
    cmd += ["-pix_fmt", "yuv420p", "-r", "%.6g" % fps]
    if audio_mode == "copy":
        # 音轨原样搬（截断点已经在输入侧用 -t 定好了）。
        # 绝不能用 -shortest：原片音轨比帧数短时会把成片截短
        # （4.0 那边就是这么把成片变成 2 秒的）。
        cmd += ["-c:a", "copy"]
    elif audio_mode == "aac":
        # 重编 + apad 补静音：音轨被拉成无限长，-shortest 就只会按画面长度收尾
        cmd += ["-c:a", "aac", "-b:a", "192k", "-af", "apad", "-shortest"]
    if out.suffix.lower() in (".mp4", ".mov", ".m4v"):
        cmd += ["-movflags", "+faststart"]
    cmd += ["-progress", "pipe:1", str(out)]
    return cmd


def _run_encode(cmd, total, tick=None):
    """跑一次 ffmpeg，按 frame= 报进度，取消时连同子进程组一起杀。"""
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, encoding="utf-8", errors="replace", bufsize=1,
                            creationflags=_NO_WINDOW | _NEW_GROUP,
                            start_new_session=(os.name != "nt"))
    _register(proc)
    tail = []
    try:
        for line in (proc.stdout or []):
            s = (line or "").strip()
            if _CANCEL.is_set():
                _kill_pid_tree(proc.pid)
                break
            if s.startswith("frame="):
                try:
                    f = int(s.split("=", 1)[1])
                except (ValueError, IndexError):
                    continue
                if tick is not None and total > 0:
                    try:
                        tick(f, total)
                    except BaseException:
                        _kill_pid_tree(proc.pid)
                        raise
            elif s and not s.startswith(("progress=", "out_time", "bitrate=", "total_size=",
                                         "speed=", "dup_frames=", "drop_frames=", "fps=")):
                tail.append(s)
                if len(tail) > 30:
                    tail = tail[-30:]
    finally:
        try:
            proc.wait(timeout=15)
        except Exception:
            _kill_pid_tree(proc.pid)
            try:
                proc.wait(timeout=5)
            except Exception:
                pass
        _unregister(proc)
    return proc.returncode, "\n".join(tail)


def _make_encoder(ffmpeg, ffprobe, fps, encoder, audio_src, crf, bitrate, scratch_root):
    """替换 export_pipeline._stage5_encode_video：不再依赖 ffprobe / 占位源的音轨。

    4.0 那版收尾编码的问题是致命的：
      · fps / 码率靠 ffprobe 从"源视频"读，而我们的源视频是现造的占位片；
      · -map 1:a:0 把占位片的**静音轨**灌进成片，再叠 -shortest → 成片被截成 2 秒；
      · 编码器没有回退，nvenc 不可用就整段导出失败；
      · 没音轨时命令直接报错，也没有 +faststart。
    这里按参数自己拼命令，并按 [硬件 → 软编] [原音轨 copy → aac → 无音轨] 依次重试。
    """
    def _encode(frames_dir, output_path, source_video, encoder_, fmt,
                progress, stop_event=None, cut_segments=None):
        ext = "png" if str(fmt).lower() == "png" else "jpg"
        src_dir = Path(str(frames_dir))
        frames = sorted(src_dir.glob("*." + ext))
        if not frames:
            raise RuntimeError("没有可编码的帧: %s（合成阶段一张都没写出来）" % src_dir)
        pattern, first, total = _locate_sequence(frames, scratch_root)
        out = Path(str(output_path))
        out.parent.mkdir(parents=True, exist_ok=True)

        audio = ""
        if audio_src and _has_audio(ffmpeg, audio_src, ffprobe):
            audio = str(audio_src)
        enc = (encoder_ or encoder or "libx264").strip() or "libx264"

        combos = []
        for e in ([enc, "libx264"] if enc != "libx264" else [enc]):
            if audio:
                combos.append((e, "copy"))
                combos.append((e, "aac"))
            combos.append((e, None))
        plan = []
        for c in combos:
            if c not in plan:
                plan.append(c)

        _say("[encode] %d 帧 · %g fps · 编码器 %s · 音轨 %s"
             % (total, fps, enc, ("沿用原片 %s" % Path(audio).name) if audio else "无（输出静音）"))

        def _tick(f, total_frames):
            if stop_event is not None and stop_event.is_set():
                raise RuntimeError("Export cancelled")
            progress(6, min(1.0, float(f) / float(max(1, total_frames))),
                     "编码 %d/%d 帧" % (f, total_frames))

        last = ""
        for i, (e2, a2) in enumerate(plan):
            cmd = _build_encode_cmd(ffmpeg, e2, a2, audio, fps, pattern, first, out,
                                    crf, bitrate, total)
            _say("[encode] 第 %d/%d 次: %s · %s" % (i + 1, len(plan), e2,
                                                  ("音频 " + a2) if a2 else "无音轨"))
            try:
                rc, tail = _run_encode(cmd, total, _tick)
            except BaseException:
                if out.exists():
                    try:
                        out.unlink()
                    except OSError:
                        pass
                raise
            if rc == 0 and out.is_file() and out.stat().st_size > 0:
                _say("[encode] 完成: %s (%.1f MB)" % (out, out.stat().st_size / 1048576.0))
                progress(6, 1.0, "编码完成")
                return
            last = tail or ("ffmpeg 退出码 %s" % rc)
            if out.exists():
                try:
                    out.unlink()
                except OSError:
                    pass
        raise RuntimeError("ffmpeg 编码失败（试了 %d 种组合）: %s"
                           % (len(plan), str(last).replace("\n", " ")[:300]))
    return _encode


# ---------- XSeg 注入 ----------
class _PathRedirect:
    """临时替换模块级的 Path：只把"引擎自己的 export_pipeline.py 路径"映射到 staging 副本。"""

    def __init__(self, ep_module, real_file, fake_file):
        self.ep = ep_module
        try:
            self.real = os.path.normcase(str(Path(str(real_file)).resolve()))
        except Exception:
            self.real = os.path.normcase(str(real_file))
        self.fake = Path(str(fake_file))
        self._orig = None

    def __enter__(self):
        self._orig = self.ep.Path
        orig, real, fake = self._orig, self.real, self.fake

        def _Path(*a, **k):
            try:
                if len(a) == 1 and not k and isinstance(a[0], str):
                    if os.path.normcase(str(Path(a[0]).resolve())) == real:
                        return fake
            except Exception:
                pass
            return orig(*a, **k)

        self.ep.Path = _Path
        return self

    def __exit__(self, *_exc):
        try:
            self.ep.Path = self._orig
        except Exception:
            pass
        return False


def _prepare_xseg_hint(work: Path, onnx: str):
    """把 --xseg 的 onnx 摆到引擎写死的位置（staging 里的同名坑位）。返回 (fake_module_path, 真实 onnx)。"""
    if not onnx:
        return None, ""
    p = Path(str(onnx))
    if not p.is_file():
        return None, ""
    hint_root = work / "xsegroot"
    dst = hint_root / "workspace" / "model" / "XSegLite" / "xseglite.onnx"
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not dst.exists():
        _link_or_copy(p, dst)
    return hint_root / "MergeStudio" / "core" / "export_pipeline.py", str(dst)


# ---------- 参数归一 ----------
_CTM_ALIASES = {
    "none": "none", "": "none", "0": "none", "off": "none", "no": "none",
    "rct": "rct", "reinhard": "rct", "1": "rct",
    "lct": "lct", "linear": "lct", "2": "lct",
    "mkl": "mkl", "3": "mkl",
    "mkl-m": "mkl-m", "4": "mkl-m",
    "idt": "idt", "5": "idt",
    "idt-m": "idt-m", "6": "idt-m",
    "sot-m": "sot-m", "sot": "sot-m", "7": "sot-m",
    "mix-m": "mix-m", "mix": "mix-m", "8": "mix-m",
    "mt": "mt", "9": "mt",
    "lut": "lut", "10": "lut",
}

_MASK_MODE_ALIASES = {
    "full": 0, "dst": 1, "learned-dst": 3,
    "prd": 2, "learned-prd": 2,
    "prd*dst": 4, "learned-prd*learned-dst": 4,
    "prd+dst": 5, "learned-prd+learned-dst": 5,
    "xseg-prd": 6, "xseg-dst": 7, "xseg-prd*dst": 8, "xseg-all": 9,
}

_MASK_MODE_LABEL = {
    0: "full", 1: "dst", 2: "learned-prd", 3: "learned-dst",
    4: "learned-prd*learned-dst", 5: "learned-prd+learned-dst",
    6: "XSeg-prd", 7: "XSeg-dst", 8: "XSeg-prd*XSeg-dst",
    9: "learned-prd*learned-dst*XSeg-prd*XSeg-dst",
}

# 引擎的 _phase_merge_worker 只认这四个：其余（original / raw-* / debug）它一样按
# overlay 合成 —— 也就是说"收下了但不生效"。
_MODE_EFFECTIVE = {"overlay", "hist-match", "seamless", "seamless-hist-match"}
_MODE_KNOWN = _MODE_EFFECTIVE | {"original", "raw-rgb", "raw-predict", "debug"}


def _as_int(v, default=0):
    if isinstance(v, bool):
        return default
    try:
        return int(round(float(v)))
    except (TypeError, ValueError):
        return default


def _as_mask_mode(v):
    if isinstance(v, bool):
        return 6, "bool 值 → 默认 6"
    if isinstance(v, int):
        return max(0, min(9, v)), ""
    s = str(v if v is not None else "").strip()
    if s.isdigit():
        return max(0, min(9, int(s))), ""
    key = s.lower().replace("×", "*").replace(" ", "")
    hit = _MASK_MODE_ALIASES.get(key)
    if hit is None:
        return 6, "无法识别的 mask_mode %r → 默认 6" % (v,)
    return hit, ""


def _as_ctm(v):
    s = str(v if v is not None else "none").strip().lower()
    hit = _CTM_ALIASES.get(s)
    if hit is None:
        return "none", "无法识别的 color_transfer %r → 关掉色域迁移" % (v,)
    return hit, ""


def _normalise_config(cfg):
    """把 4.1（以及任何调用方）送来的别名归一成引擎认的键，并给出生效报告。"""
    cfg = dict(cfg or {})
    notes, eff = [], {}

    raw_mask = cfg.get("mask_mode", cfg.get("mask", 6))
    mask_mode, n1 = _as_mask_mode(raw_mask)
    if n1:
        notes.append(n1)

    raw_ct = cfg.get("color_transfer_mode", cfg.get("color_transfer", cfg.get("ct_mode", "none")))
    ct, n2 = _as_ctm(raw_ct)
    if n2:
        notes.append(n2)

    erode = _as_int(cfg.get("erode_mask_modifier",
                            cfg.get("erode_mask", cfg.get("mask_erode", cfg.get("erode", 0)))), 0)
    blur = _as_int(cfg.get("blur_mask_modifier",
                           cfg.get("blur_mask", cfg.get("mask_feather",
                                                        cfg.get("mask_blur", cfg.get("blur", 0))))), 0)
    erode = max(-100, min(100, erode))
    blur = max(-100, min(100, blur))

    mode = str(cfg.get("mode", "overlay") or "overlay").strip().lower()
    if mode not in _MODE_KNOWN:
        notes.append("无法识别的 mode %r → 按 overlay 合成" % (cfg.get("mode"),))
        mode = "overlay"

    config = {
        "mask_mode": mask_mode,
        "mode": mode,
        "color_transfer_mode": ct,
        "erode_mask_modifier": erode,
        "blur_mask_modifier": blur,
    }
    hmt = cfg.get("hist_match_threshold")
    if hmt is not None and str(hmt).strip() != "":
        config["hist_match_threshold"] = max(0, min(255, _as_int(hmt, 255)))

    eff["mask_mode"] = True
    eff["erode_mask_modifier"] = True
    eff["blur_mask_modifier"] = True
    eff["color_transfer_mode"] = ct != "none"
    eff["mode"] = mode in _MODE_EFFECTIVE
    if not eff["mode"]:
        notes.append("mode=%s 不生效：引擎的 phase 合并只实现 overlay / hist-match / seamless / "
                     "seamless-hist-match，其余一律按 overlay 出片" % mode)

    face_type = cfg.get("face_type")
    if face_type is not None and str(face_type).strip():
        eff["face_type"] = False
        notes.append("face_type=%s 不生效：引擎的换脸/合并阶段写死 FaceType.WHOLE_FACE"
                     "（_phase_swap_worker / _phase_merge_worker），这个值传不进算法" % (face_type,))
    for k in ("face_margin", "super_resolution_power", "output_face_scale",
              "motion_blur_power", "image_denoise_power", "bicubic_degrade_power",
              "color_degrade_power", "sharpen_mode", "blursharpen_amount"):
        if cfg.get(k) is not None:
            eff[k] = False
            notes.append("%s 不生效：引擎的 phase 合并路径不读这个键（只有预览侧的 MergeMaskedFace 用）" % k)

    for k in ("mask_erode", "mask_feather", "erode_mask", "blur_mask", "erode", "blur",
              "color_transfer", "ct_mode"):
        if k in cfg:
            notes.append("%s 已并入上面归一后的键" % k)

    return config, {"effective": eff, "notes": notes}


def _report_params(config, report, extra):
    label = _MASK_MODE_LABEL.get(int(config.get("mask_mode", 6)), "?")
    _say("[merge] 合成参数（生效值）: %s" % json.dumps(config, ensure_ascii=False))
    _say("[merge] 参数生效情况:")
    for k in ("mask_mode", "erode_mask_modifier", "blur_mask_modifier",
              "color_transfer_mode", "mode"):
        ok = report["effective"].get(k, True)
        _say("         %-20s = %-14s %s" % (k, config.get(k), "生效" if ok else "★ 不生效"))
    _say("         mask_mode 含义        = %s" % label)
    if report["notes"]:
        _say("[merge] 参数提示:")
        for n in report["notes"]:
            _say("         - %s" % n)
    payload = {"config": config, "effective": report["effective"], "notes": report["notes"]}
    payload.update(extra or {})
    print("@@PARAMS " + json.dumps(payload, ensure_ascii=False), flush=True)


# ---------- 多模型（settings["models"]） ----------
# 4.1 侧的契约：settings["models"] = [
#     {"name": "张元英", "dfm": "<abs .dfm>", "faces": [0]},
#     {"name": "韩素希", "dfm": "<abs .dfm>", "faces": [1]},
# ]
#   · faces = 这个模型负责的人脸序号，按检测顺序从 0 起（= 对齐文件 <帧>_<脸>.jpg 的 <脸>）
#   · 单模型时 faces 省略 / [] 都等价于「全部脸」，行为与改动前逐字一致
#   · 旧字段 model（单模型名）仍然认：路径按名字在模型目录里精确找
#
# 引擎侧事实（先把 mainscripts/Merger.py 和 MergeStudio/core/export_pipeline.py 读完才敢接）：
#   · Merger.py 只收一个 dfm_onnx_path，没有 argparse，也不认 faces 映射 —— 它本来就不是
#     这条桥的入口（这条桥直接调 MergeStudio 的 run_export_pipeline）。
#   · 真正支持多模型的是 export_pipeline：model_loader.load_model() 把 N 个 dfm 注册进
#     _sessions（键 = 绝对路径），face_model_map = {脸键: 模型键} 写进 face_data.model，
#     _stage4_swap_faces_mp 再按模型键分组，一个模型跑一轮换脸。
#   · 模型键是引擎自己从文件名推的：Path(p).stem.replace('_model','').split('_')[0]。
#     所以 face_model_map 的**值**必须用同一套推导，否则 Stage4 分组对不上，那些脸会被
#     静默跳过 —— 正好是本任务禁止的行为。
#   · face_model_map 的**键**用「脸序号」（"0" / "1"）：_worker_stage3 的兜底分支认
#     str(face_idx)。别改成 "帧_脸" 形式：那会让 Stage3 给每一条参考都算一遍 ArcFace
#     嵌入（N 帧 = N 次 embedding），导出会慢到不可用。

_FACES_ALL = None                       # faces = 全部脸


def _engine_model_key(dfm) -> str:
    """和 _stage4_swap_faces_mp 里的 _mkey 推导保持一致（唯一真相在引擎那边）。"""
    return Path(str(dfm)).stem.replace("_model", "").split("_")[0]


def _faces_label_list(faces):
    return "[" + ",".join(str(int(f)) for f in faces) + "]"


def _faces_list(v, who):
    """faces 字段 → 升序去重的 int 列表，或 None（= 全部脸）。

    "0" / "0,1" / [0, 1] / 0 都认；空值、空列表 = 全部脸（单模型时的老语义）。
    非整数、负数直接报错 —— 宁可炸在参数上，也不要悄悄少换一张脸。
    """
    if v is None:
        return _FACES_ALL
    if isinstance(v, bool):
        raise RuntimeError("%s 的 faces 是布尔值，看不懂: %r" % (who, v))
    if isinstance(v, (int, float)):
        v = [v]
    elif isinstance(v, str):
        s = v.strip().strip("[](){}")
        v = [t for t in s.replace(";", ",").replace(" ", ",").split(",") if t.strip()]
    elif isinstance(v, (list, tuple, set)):
        v = list(v)
    else:
        raise RuntimeError("%s 的 faces 类型不支持: %s" % (who, type(v).__name__))
    out = set()
    for t in v:
        try:
            f = int(str(t).strip())
        except (TypeError, ValueError):
            raise RuntimeError("%s 的 faces 里有非整数: %r" % (who, t))
        if f < 0:
            raise RuntimeError("%s 的 faces 里有负数（脸序号从 0 起）: %d" % (who, f))
        out.add(f)
    if not out:
        return _FACES_ALL
    return sorted(out)


def _merge_faces(a, b):
    if a is None or b is None:
        return None
    return sorted(set(a) | set(b))


class _Model:
    """一个 DFM + 它负责的脸。

    named=False 表示它来自 --dfm（没在 settings 里起名字）：这种走改动前的老文案
    （"DFM 不存在: …"），保持单模型时的输出逐字不变。
    """

    __slots__ = ("name", "path", "faces", "named", "source")

    def __init__(self, path, name=None, faces=None, named=True, source="settings"):
        self.path = Path(str(path))
        self.name = str(name or self.path.stem)
        self.faces = faces
        self.named = bool(named)
        self.source = source

    @property
    def key(self):
        return _engine_model_key(self.path)

    def faces_label(self):
        return "全部" if self.faces is None else _faces_label_list(self.faces)

    def missing_error(self):
        if self.named:
            return "模型「%s」的 DFM 不存在: %s" % (self.name, self.path)
        return "DFM 不存在: %s" % self.path

    def load_error(self, detail):
        if self.named:
            return "加载 DFM 失败（模型「%s」%s）: %s" % (self.name, self.path, detail)
        return "加载 DFM 失败（不是有效的 onnx/.dfm？）: %s" % detail

    def as_dict(self):
        return {"name": self.name,
                "dfm": str(self.path),
                "faces": (list(self.faces) if self.faces is not None else None),
                "key": self.key}


def _read_json_arg(raw, where):
    """--dfm-models / --settings 的值 → Python 对象。内联 JSON 或 JSON 文件路径都行。"""
    s = str(raw or "").strip()
    if not s:
        return None
    try:
        p = Path(s)
        is_file = p.is_file()
    except (OSError, ValueError):
        p, is_file = None, False
    if is_file:
        try:
            s = p.read_text(encoding="utf-8-sig")
        except Exception as e:
            raise RuntimeError("读不了模型清单 %s: %s" % (p, e))
        where = str(p)
    try:
        return json.loads(s)
    except Exception as e:
        raise RuntimeError("%s 既不是存在的文件也不是合法 JSON: %s" % (where, str(e)[:200]))


def _coerce_entries(payload, where):
    """settings / list / 单条 dict → [entry dict]。"""
    if payload is None:
        return []
    if isinstance(payload, dict):
        v = payload.get("models")
        if isinstance(v, (list, dict)):
            payload = v
        elif isinstance(payload.get("model"), str) and payload.get("model").strip():
            # 旧字段：单模型名（可能配 dfm / dfm_path）
            one = {"name": payload["model"].strip()}
            for k in ("dfm", "dfm_path", "dfmPath", "path"):
                if payload.get(k):
                    one["dfm"] = payload[k]
                    break
            return [one]
        elif any(payload.get(k) for k in ("dfm", "dfm_path", "dfmPath", "path", "name")):
            payload = [payload]
        else:
            return []
    if isinstance(payload, dict):
        payload = [payload]
    if not isinstance(payload, (list, tuple)):
        raise RuntimeError("%s 的形状不认识（要 list，或带 models 的对象）: %s"
                           % (where, type(payload).__name__))
    entries = []
    for i, e in enumerate(payload):
        if isinstance(e, str):
            e = {"dfm": e}
        if not isinstance(e, dict):
            raise RuntimeError("%s 第 %d 个模型不是对象: %r" % (where, i + 1, e))
        entries.append(dict(e))
    return entries


def _find_dfm(raw_dfm, name, base_dirs):
    """→ (Path | None, 说明)。

    给了路径就只认这条路径（相对路径依次在 base_dirs 里补全，绝不猜别的名字）；
    只给了名字才在模型目录里按 <名字>.dfm / <名字>-<迭代>.dfm 精确找。
    """
    if raw_dfm:
        cand = Path(str(raw_dfm))
        if cand.is_file():
            return cand, ""
        if not cand.is_absolute():
            for d in base_dirs:
                c = Path(str(d)) / cand
                if c.is_file():
                    return c, ""
        return cand, ""          # 交给调用方按「不存在」报错，报的是用户给的原样路径
    if not name:
        return None, ""
    for d in base_dirs:
        for c in (Path(str(d)) / (name + ".dfm"), Path(str(d)) / (name + ".DFM")):
            if c.is_file():
                return c, ""
    hits = []
    for d in base_dirs:
        try:
            hits.extend(p for p in Path(str(d)).glob(name + "-*.dfm") if p.is_file())
        except OSError:
            continue
    if not hits:
        return None, ""
    hits.sort(key=lambda p: _as_int(p.stem.rsplit("-", 1)[-1], 0))
    best = hits[-1]
    if len(hits) > 1:
        note = ("模型「%s」只给了名字：按 <名字>-<迭代>.dfm 找到 %d 个，取迭代最高的 %s"
                % (name, len(hits), best.name))
    else:
        note = "模型「%s」只给了名字：按模型目录找到 %s" % (name, best.name)
    return best, note


def _model_dirs(dfl_root):
    """按名字找 dfm 的搜索目录：4.1 的 workspace/model 优先，其次引擎的，最后 cwd。"""
    dirs = []
    here = Path(__file__).resolve()
    cands = []
    if len(here.parents) > 2:
        cands.append(here.parents[2] / "workspace" / "model")
    if dfl_root:
        cands.append(Path(str(dfl_root)) / "workspace" / "model")
    cands.append(Path.cwd())
    for c in cands:
        try:
            c = c.resolve()
        except OSError:
            pass
        if c.is_dir() and c not in dirs:
            dirs.append(c)
    return dirs


def _resolve_models(payload, dfm_args=(), dfm_faces=(), base_dirs=(), where="--dfm-models"):
    """settings["models"] + --dfm[*] → ([_Model], 说明列表)。文件缺失当场报错。

    payload 可以是：命令行原样字符串（内联 JSON / JSON 文件路径）、已解析的
    dict/list，或 None（只给 --dfm 的老写法）。
    """
    if isinstance(payload, str):
        payload = _read_json_arg(payload, where) if payload.strip() else None
    entries = _coerce_entries(payload, where)
    dfm_args = [str(x) for x in (dfm_args or ())]
    dfm_faces = list(dfm_faces or ())
    if dfm_faces and len(dfm_faces) != len(dfm_args):
        raise RuntimeError("--dfm 给了 %d 个、--dfm-faces 给了 %d 个，一一对应不上"
                           % (len(dfm_args), len(dfm_faces)))
    for i, p in enumerate(dfm_args):
        e = {"dfm": p, "_from_cli": True}
        if dfm_faces:
            e["faces"] = dfm_faces[i]
        entries.append(e)
    if not entries:
        raise RuntimeError("没有模型：--dfm / --dfm-models / --settings 至少给一个")

    out, notes, by_path = [], [], {}
    for i, e in enumerate(entries):
        raw_name = str(e.get("name") or "").strip()
        raw_dfm = ""
        for k in ("dfm", "dfm_path", "dfmPath", "path", "file", "dfm_file"):
            if e.get(k):
                raw_dfm = str(e[k]).strip()
                break
        from_cli = bool(e.get("_from_cli"))
        who = ("模型「%s」" % raw_name) if (raw_name and not from_cli) else ("第 %d 个模型" % (i + 1))
        if not raw_dfm and not raw_name:
            raise RuntimeError("%s 既没有 dfm 路径也没有名字" % who)
        path, note = _find_dfm(raw_dfm, raw_name, base_dirs)
        if note:
            notes.append(note)
        if path is None:
            raise RuntimeError("模型「%s」只给了名字，但 %s 里找不到 %s.dfm"
                               % (raw_name,
                                  " / ".join(str(d) for d in base_dirs) or "（没有可用搜索目录）",
                                  raw_name))
        faces = _faces_list(e.get("faces", e.get("face", e.get("indices"))), who)
        # settings 里的条目一律按「有名字」报错（哪个名字 + 哪个路径）；--dfm 保持老文案
        m = _Model(path, raw_name or None, faces, named=not from_cli,
                   source=("cli" if from_cli else "settings"))
        try:
            sig = os.path.normcase(str(m.path.resolve()))
        except OSError:
            sig = os.path.normcase(str(m.path))
        hit = by_path.get(sig)
        if hit is not None:
            hit.faces = _merge_faces(hit.faces, m.faces)
            notes.append("%s 和 %s 指向同一个 DFM：faces 合并成 %s"
                         % (hit.name, m.name, hit.faces_label()))
            continue
        by_path[sig] = m
        out.append(m)

    missing = [m for m in out if not m.path.is_file()]
    if missing:
        raise RuntimeError("；".join(m.missing_error() for m in missing))

    seen = {}
    for m in out:
        prev = seen.get(m.key)
        if prev is not None and prev.path != m.path:
            raise RuntimeError(
                "两个 DFM 推出同一个引擎模型键 %r：「%s」(%s) 与「%s」(%s)。"
                "Stage4 按这个键把脸分组，重名会拿错模型 —— 请改名（别共用第一个下划线前缀）"
                % (m.key, prev.name, prev.path, m.name, m.path))
        seen[m.key] = m
    return out, notes


def _staged_face_indices(aligned_dir):
    """staging 的 aligned/%08d_%d.jpg → 现场到底有哪些脸序号。"""
    out = set()
    try:
        for p in Path(str(aligned_dir)).glob("*.jpg"):
            stem = p.stem
            if "_" in stem:
                try:
                    out.add(int(stem.rsplit("_", 1)[1]))
                except ValueError:
                    continue
            else:
                out.add(0)
    except OSError:
        pass
    return out


def _build_face_model_map(models, face_indices):
    """[_Model] + 现场脸序号 → (face_model_map, 说明列表)。

    单模型且没点名 faces → {}：和改动前完全一样，引擎按「全部脸用默认模型」跑，
    Stage3 都不会被拉起来。
    多模型 → {"0": keyA, "1": keyB}；faces 省略/空的那个模型兜底吃下没被点名的脸。
    """
    notes, face_map = [], {}
    if not models:
        return face_map, notes
    if len(models) == 1 and models[0].faces is None:
        return face_map, notes

    catch_all = [m for m in models if m.faces is None]
    if len(catch_all) > 1:
        raise RuntimeError("有 %d 个模型都要「全部脸」：%s —— 分不出该用哪个，请在 faces 里点名"
                           % (len(catch_all), "、".join(m.name for m in catch_all)))

    known = set(int(f) for f in (face_indices or ()))
    claimed = {}
    for m in models:
        for f in (m.faces or ()):
            prev = claimed.get(f)
            if prev is not None and prev is not m:
                raise RuntimeError("脸 %d 同时被「%s」和「%s」认领（faces 有重叠）"
                                   % (f, prev.name, m.name))
            claimed[f] = m
    if catch_all:
        rest = sorted(known - set(claimed))
        if rest:
            for f in rest:
                claimed[f] = catch_all[0]
            notes.append("「%s」没点 faces → 兜底负责剩下的脸 %s"
                         % (catch_all[0].name, _faces_label_list(rest)))
        else:
            notes.append("「%s」没点 faces，但现场的脸都被别的模型点名了 → 它没有脸可换"
                         % catch_all[0].name)
    for f in sorted(claimed):
        face_map[str(f)] = claimed[f].key
    if known:
        for m in models:
            if m.faces is None:
                continue
            ghost = [f for f in m.faces if f not in known]
            if ghost:
                notes.append("模型「%s」的 faces 里 %s 在现场不存在（现场只有 %s）"
                             % (m.name, _faces_label_list(ghost),
                                _faces_label_list(sorted(known))))
    # 映射本身由 main() 统一打一行「face_model_map: …」，这里不重复输出
    return face_map, notes


def build_engine_cmd(models):
    """引擎侧的等价命令行 —— 日志里贴它，也是「命令拼装」的可测入口。

    单模型（没点名 faces）→ ["--dfm", path]，与改动前逐字相同；
    多模型（或点了 faces）→ --dfm 重复 + 一段 --dfm-models JSON（name/dfm/faces/引擎键）。
    引擎实际是 Python API（run_export_pipeline），真正传过去的是 face_model_map。
    """
    if not models:
        return []
    if len(models) == 1 and models[0].faces is None:
        return ["--dfm", str(models[0].path)]
    cmd = []
    for m in models:
        cmd += ["--dfm", str(m.path)]
    cmd += ["--dfm-models",
            json.dumps([m.as_dict() for m in models], ensure_ascii=False,
                       separators=(",", ":"))]
    return cmd


# ---------- 主流程 ----------
def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", required=True)
    ap.add_argument("--aligned", required=True)
    # --dfm 可重复（append）：单模型时命令行与改动前逐字相同；多模型配 --dfm-faces，
    # 或整份 settings 交给 --dfm-models / --settings（settings["models"] 契约）
    ap.add_argument("--dfm", action="append", default=[])
    ap.add_argument("--dfm-faces", action="append", default=[])
    ap.add_argument("--dfm-models", default="")
    ap.add_argument("--settings", default="")
    ap.add_argument("--out", required=True)
    ap.add_argument("--config", default="{}")
    ap.add_argument("--xseg", default="")
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--encoder", default="h264_nvenc")
    ap.add_argument("--hwaccel", default="auto")
    ap.add_argument("--image-format", default="jpg")
    ap.add_argument("--fps", type=float, default=25.0)
    ap.add_argument("--crf", type=int, default=17)
    ap.add_argument("--bitrate", default="")
    ap.add_argument("--source-video", default="")
    ap.add_argument("--ffmpeg", default="")
    ap.add_argument("--ffprobe", default="")
    ap.add_argument("--dfl-root", default="")
    args = ap.parse_args(argv)

    t_start = time.time()
    _install_job_object()
    _install_cleanup()

    dfl_root = Path(args.dfl_root) if args.dfl_root else \
        Path(__file__).resolve().parents[2].parent / "DeepFaceLab-PyTorch"
    if not (dfl_root / "MergeStudio").is_dir():
        raise RuntimeError("找不到 4.0 的 MergeStudio: %s" % dfl_root)
    sys.path.insert(0, str(dfl_root))

    frames_dir = Path(args.frames)
    aligned_dir = Path(args.aligned)
    if not frames_dir.is_dir():
        raise RuntimeError("帧目录不存在: %s" % frames_dir)
    if not aligned_dir.is_dir():
        raise RuntimeError("对齐目录不存在: %s" % aligned_dir)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # 多模型：settings["models"]（旧字段 model 仍认）+ 可重复的 --dfm。
    # 模型文件就在这一步校验 —— 少一个 dfm 必须当场炸，不能跑到合成阶段才发现有脸没换。
    models, model_notes = _resolve_models(
        (args.dfm_models or args.settings),
        args.dfm, args.dfm_faces, _model_dirs(dfl_root),
        where=("--settings" if (args.settings and not args.dfm_models) else "--dfm-models"))
    for _n in model_notes:
        _say("[merge] %s" % _n)

    try:
        fps = float(args.fps or 25.0)
    except (TypeError, ValueError):
        fps = 25.0
    if not (1.0 <= fps <= 240.0):
        _say("[merge] fps=%r 不合理，按 25 处理" % (args.fps,))
        fps = 25.0

    # staging 目录名必须是纯 ASCII：MergeStudio 内部用 cv2.imread/imwrite 读写帧和
    # merged/，OpenCV 在 Windows 上走 ANSI 路径，遇到中文目录会**静默失败**
    # （merged/ 变空，最后编码阶段报"没有可编码的帧"）。所以路径不全 ASCII 就退到系统 Temp。
    _sweep_parent = out_path.parent
    slug = hashlib.sha1(str(out_path.stem).encode("utf-8")).hexdigest()[:8]
    work_name = ".merge_" + slug
    work = None
    for base in (_sweep_parent, Path(tempfile.gettempdir())):
        if not _ascii_ok(base):
            _say("[merge] 跳过含非 ASCII 字符的目录（OpenCV 会静默失败）: %s" % base)
            continue
        try:
            cand = Path(base) / work_name
            cand.mkdir(parents=True, exist_ok=True)
            probe = cand / ".probe"
            probe.write_bytes(b"x")
            probe.unlink()
            work = cand
            break
        except Exception as e:
            _say("[merge] 临时目录 %s 不可用: %s" % (base, e))
    if work is None:
        raise RuntimeError("找不到可写的 ASCII 临时目录（输出目录含中文且系统 Temp 也不可用）")
    _STATE["work"] = str(work)
    _STATE["out"] = str(out_path)
    _sweep_stale(_sweep_parent, work)

    # 输出目录不是纯 ASCII 时，管线内部那些 cv2 写的中间件（swap_pred/、masks/、
    # merged/）同样会静默失败 —— 因为 work_dir 是 <输出目录>/.export_<stem>。
    # 先把成片编码到 staging 里，最后由 Python 搬到用户要的位置（搬文件不经过 cv2）。
    real_out = out_path
    if not _ascii_ok(out_path.parent):
        out_path = work / ("final" + (real_out.suffix or ".mp4"))
        _say("[merge] 输出目录含非 ASCII 字符：先在 staging 编码，最后搬到 %s" % real_out)

    PROG.set(0, 0.0, "搭 staging 项目目录", force=True)
    proj = stage_project(frames_dir, aligned_dir, work)

    n_frames = sum(1 for p in proj.glob("[0-9]*.[jp][pn]g"))
    n_aligned = sum(1 for p in (proj / "aligned").glob("*.jpg"))
    _say("[stage] 项目就绪: %d 帧 / %d 张对齐脸" % (n_frames, n_aligned))
    if n_frames < 10:
        # 引擎的 DFL 项目模式判定条件是 len(_frames) >= 10：不足 10 帧它会退回
        # "从源视频抽帧 + 重跑人脸检测"，而源视频只是我们造的占位片 —— 会去导出一条黑片。
        raise RuntimeError("帧数 %d < 10 —— MergeStudio 的 DFL 项目模式要求 ≥10 帧，"
                           "否则它会退回去跑人脸检测并输出一条黑片" % n_frames)

    # 脸序号来自 staging 的 aligned/%08d_%d.jpg —— 引擎的 face_idx 就是从这儿读的，
    # 所以「现场有哪些脸」在进管线之前就已经确定了，兜底模型能补得准。
    face_map, map_notes = _build_face_model_map(models, _staged_face_indices(proj / "aligned"))
    for _n in map_notes:
        _say("[merge] %s" % _n)
    _eng_cmd = build_engine_cmd(models)
    _say("[merge] 引擎参数: %s" % subprocess.list2cmdline(_eng_cmd))
    _say("[merge] face_model_map: %s" % json.dumps(face_map, ensure_ascii=False))

    PROG.set(0, 0.3, "加载模型")

    # 先把 CUDA/cuDNN 的 DLL 目录塞进搜索路径，再建 session。
    # 不加这一步，CUDAExecutionProvider 会**静默**退回 CPU（日志里只有一行 warning），
    # 实测 DFM 单张 140ms(CPU) vs 11ms(GPU)，差 12 倍。
    try:
        from MergeStudio.core.export_pipeline import _setup_dll_paths as _dll
        _dll()
    except Exception:
        pass
    try:
        import onnxruntime as _ort
        _say("[merge] onnxruntime providers: %s" % (", ".join(_ort.get_available_providers())))
    except Exception:
        pass

    from MergeStudio.core.model_loader import model_loader
    if len(models) > 1:
        _say("[merge] 多模型导出：共 %d 个 DFM" % len(models))
    for _m in models:
        try:
            _sess = model_loader.load_model(str(_m.path))
        except Exception as e:
            raise RuntimeError(_m.load_error(str(e)[:300]))
        try:
            _prov = _sess.get_providers() if hasattr(_sess, "get_providers") else []
            _say("[merge] DFM 运行在: %s" % (", ".join(_prov) or "?"))
        except Exception:
            pass
        _say("[merge] 模型已加载: %s" % _m.path.name)
    if len(models) > 1:
        for _m in models:
            _say("[merge]   %s → 脸 %s（引擎键 %s）"
                 % (_m.path.name, _m.faces_label(), _m.key))

    # 事后确认用的那一行：CUDA 不可用时 ORT 不抛异常、静默退回 CPU，只有会话自己知道
    # 实际在跑哪个 provider。多模型时它们共用同一个 provider 决定，取最后建的那个会话即可。
    try:
        _active = _sess.get_providers() if hasattr(_sess, "get_providers") else []
    except Exception:
        _active = []
    _say("[merge] ORT providers: %s" % (_active[0] if _active else "?"))
    if _active[:1] == ["CPUExecutionProvider"]:
        _say("[merge] CUDA 没生效，DFM 全跑 CPU（torch/lib=%s，查 cublasLt64_12.dll / cudnn64_9.dll / 驱动）"
             % (_ORT_DLL_DIR or "未找到"))

    try:
        cfg = json.loads(args.config or "{}")
        if not isinstance(cfg, dict):
            raise ValueError("config 不是对象")
    except Exception as e:
        _say("[merge] --config 解析失败（按默认值跑）: %s" % e)
        cfg = {}
    config, report = _normalise_config(cfg)

    import multiprocessing
    workers = args.workers if args.workers > 0 else max(1, multiprocessing.cpu_count() // 2)
    workers = max(1, min(64, int(workers)))

    import MergeStudio.core.export_pipeline as ep
    ffmpeg, ffprobe = _resolve_ffmpeg(args.ffmpeg, args.ffprobe, dfl_root)
    ep._FFMPEG = ffmpeg
    if ffprobe:
        ep._FFPROBE = ffprobe
    _say("[merge] ffmpeg: %s%s" % (ffmpeg, (" · ffprobe: " + ffprobe) if ffprobe else " · 无 ffprobe"))

    enc, enc_notes = _pick_encoder(ffmpeg, args.encoder)
    for n in enc_notes:
        _say("[merge] 编码器: %s" % n)
    _hw = "" if str(args.hwaccel).strip().lower() in ("off", "0", "false", "") else "auto"

    # 音轨来源：显式 --source-video > 输出目录旁的 <stem>.mp4 > 帧目录旁的 <dir>.mp4
    audio_src = _find_source_video(args.source_video, out_path, frames_dir)
    crf = max(0, min(51, args.crf if args.crf is not None else 17))
    bitrate = str(args.bitrate or "").strip()

    # XSeg：引擎里那条路径是写死的，这里重定向到我们指定的 onnx（否则 mask_mode 6/7/8/9 静默退化）
    fake_mod, xseg_used = _prepare_xseg_hint(work, args.xseg)
    engine_xseg = (Path(ep.__file__).resolve().parents[2] / "workspace" / "model" / "XSegLite" / "xseglite.onnx")
    xseg_effective = bool(xseg_used) or engine_xseg.is_file()
    xseg_note = ""
    if xseg_used:
        xseg_note = "XSeg 已注入引擎: %s" % xseg_used
    elif engine_xseg.is_file():
        xseg_note = "使用引擎自带 XSeg: %s" % engine_xseg
    else:
        xseg_note = "没有可用的 XSeg onnx（--xseg 空 且 %s 不存在）" % engine_xseg
        if int(config["mask_mode"]) >= 6:
            report["notes"].append(
                "mask_mode=%s 需要 XSeg 模型，但找不到 —— 引擎会把遮罩退化成整块矩形贴脸"
                % config["mask_mode"])
    report["effective"]["mask_mode_xseg"] = xseg_effective
    _say("[merge] %s" % xseg_note)

    fake_video = proj.parent / (proj.name + ".mp4")
    r = subprocess.run([ffmpeg, "-y", "-v", "error",
                        "-f", "lavfi", "-i", "color=c=black:s=64x64:r=%.6g:d=1" % fps,
                        "-c:v", "libx264", "-pix_fmt", "yuv420p", str(fake_video)],
                       capture_output=True, creationflags=_NO_WINDOW)
    if not fake_video.is_file() or fake_video.stat().st_size == 0:
        raise RuntimeError("造占位源视频失败（ffmpeg）: %s" % ffmpeg)
    # 占位片只用来让 Stage5 能认出"项目目录"（video stem → proj），音轨一律从原片取
    _say("[merge] 占位源视频: %s (%.6g fps)" % (fake_video.name, fps))

    _extra = {"encoder_requested": args.encoder, "encoder_used": enc,
              "ffmpeg": ffmpeg, "ffprobe": ffprobe, "fps": fps,
              "workers": workers, "audio": audio_src or "",
              "xseg": xseg_used or (str(engine_xseg) if engine_xseg.is_file() else ""),
              "out": str(out_path)}
    if face_map or len(models) > 1:
        # 多模型才加这两项：单模型时 @@PARAMS 的 payload 与改动前逐字相同
        _extra["models"] = [m.as_dict() for m in models]
        _extra["face_model_map"] = dict(face_map)
    _report_params(config, report, _extra)

    PROG.set(0, 0.4, "开始合成（%d workers）" % workers, force=True)

    def progress_callback(stage, stage_name, pct, msg=""):
        PROG.set(stage, pct, "%s %s" % (stage_name, (msg or "")))

    _orig_stage5 = ep._stage5_encode_video
    ep._stage5_encode_video = _make_encoder(ffmpeg, ffprobe, fps, enc, audio_src,
                                            crf, bitrate, work)

    from MergeStudio.core.export_pipeline import run_export_pipeline
    stop_event = _CANCEL

    _pipe_kwargs = dict(
        video_path=str(fake_video),
        output_path=str(out_path),
        image_format=("png" if str(args.image_format).lower() == "png" else "jpg"),
        encoder=enc,
        hwaccel=_hw,
        config=config,
        face_db={},
        # 单模型（没点名 faces）就是 {}，和改动前一致：引擎按默认模型换全部脸
        face_model_map=face_map,
        cut_segments=None,
        angle_segments=None,
        progress_callback=progress_callback,
        stop_event=stop_event,
        num_workers=workers,
    )
    # 引擎将来要是认 models=（契约里的 name/dfm/faces），就顺手一并给它；
    # 现在的 export_pipeline 只吃 face_model_map，别硬塞不认识的键。
    try:
        import inspect as _inspect
        if "models" in _inspect.signature(run_export_pipeline).parameters:
            _pipe_kwargs["models"] = [m.as_dict() for m in models]
    except Exception:
        pass

    try:
        if fake_mod is not None:
            with _PathRedirect(ep, ep.__file__, fake_mod):
                run_export_pipeline(**_pipe_kwargs)
        else:
            run_export_pipeline(**_pipe_kwargs)
    finally:
        try:
            ep._stage5_encode_video = _orig_stage5
        except Exception:
            pass

    if not out_path.is_file() or out_path.stat().st_size == 0:
        raise RuntimeError("管线跑完了但没有成片: %s" % out_path)

    if out_path != real_out:
        PROG.set(6, 0.99, "搬成片到输出目录")
        try:
            real_out.parent.mkdir(parents=True, exist_ok=True)
            if real_out.exists():
                real_out.unlink()
            shutil.move(str(out_path), str(real_out))
        except Exception as e:
            raise RuntimeError("搬成片到 %s 失败: %s" % (real_out, e))
        out_path = real_out

    PROG.done("合成完成，%.1f MB · %.0fs" % (out_path.stat().st_size / 1048576.0,
                                            time.time() - t_start))
    _STATE["done"] = True
    _cleanup_work()
    print("@@DONE %s" % out_path.resolve(), flush=True)
    return 0


def _find_source_video(explicit, out_path: Path, frames_dir: Path):
    """找原视频（取音轨用）。找不到就返回 ""，成片静音。"""
    if explicit:
        p = Path(str(explicit))
        if p.is_file():
            return str(p)
    exts = (".mp4", ".mkv", ".mov", ".avi", ".webm", ".flv")
    stems = []
    for s in (out_path.stem, Path(frames_dir).name):
        s = str(s or "").strip()
        if s.endswith("_merged"):              # <name>_merged.mp4 是我们自己的产物
            s = s[: -len("_merged")]
        if s and s not in stems:
            stems.append(s)
    # 4.1 的项目目录是 <workspace>/merge/<视频名>/，原视频在 <workspace>/<视频名>.mp4，
    # 所以帧目录往上两级也要找一遍（同名精确匹配，不做模糊猜测）。
    fd = Path(frames_dir)
    bases = [Path(out_path).parent, fd, fd.parent, fd.parent.parent]
    for base in bases:
        if not base.is_dir():
            continue
        for stem in stems:
            for ext in exts:
                c = base / (stem + ext)
                try:
                    if c.is_file() and c.stat().st_size > 0:
                        return str(c)
                except OSError:
                    continue
    return ""


def _sweep_stale(parent: Path, keep: Path):
    """清掉上次被强杀留下的 staging 目录（只碰 .merge_ 前缀、且超过 12 小时的）。"""
    try:
        now = time.time()
        for d in Path(parent).glob(".merge_*"):
            try:
                if not d.is_dir() or os.path.normcase(str(d)) == os.path.normcase(str(keep)):
                    continue
                if now - d.stat().st_mtime < 12 * 3600:
                    continue
            except OSError:
                continue
            shutil.rmtree(str(d), ignore_errors=True)
    except Exception:
        pass


if __name__ == "__main__":
    try:
        sys.exit(main())
    except SystemExit:
        raise
    except BaseException as e:
        try:
            PROG.fail(str(e))
        except Exception:
            pass
        print("@@ERR %s" % str(e).replace("\n", " ")[:400], flush=True)
        traceback.print_exc()
        _kill_children()
        _cleanup_work()
        sys.exit(2)
