"""本地 HTTP 服务：静态界面 + JSON API。

界面是纯静态文件（app/web），后端是 FastAPI。两边只通过 HTTP 说话，
所以这层壳既可以装进 pywebview，也可以原样装进 Electron / Tauri，
甚至直接在浏览器里打开调试。
"""
from __future__ import annotations

import asyncio
import json
import os
import threading
import time
from pathlib import Path

# 4.1 是 GUI 应用：任何子进程只要弹一下黑色控制台窗口，用户都会以为出事了。
# Windows 上统一用 CREATE_NO_WINDOW（POSIX 下是 0，无副作用）。
_NO_WINDOW = 0x08000000 if os.name == "nt" else 0

from fastapi import FastAPI, HTTPException, Response
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

WEB = Path(__file__).resolve().parent.parent / "web"
# 本仓库根目录（DeepFaceLab-PyTorch-4.1）：界面上不带盘符的路径都相对它解析
_APP_ROOT = Path(__file__).resolve().parents[2]
ROOT = Path(__file__).resolve().parents[2]
# 工作区可以用环境变量覆盖（迁移旧工程 / 多工程切换）
WORKSPACE = Path(os.environ.get("DFL_WORKSPACE") or (ROOT / "workspace"))

# 工作区只有一个来源：环境变量 DFL_WORKSPACE，否则就是本仓库的 workspace。
# （项目根的 workspace.txt 间接覆盖机制已取消——路径就是路径，不做隐藏跳转。）
SETTINGS_FILE = WORKSPACE / "settings.json"

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

app = FastAPI(title="DeepFaceLab 4.1", docs_url=None, redoc_url=None)


# ============================================================
# 合成工作室 · 视频预览（只读帧，不做 AI 推理）
# 依赖 imageio[ffmpeg]：_python 里已装 imageio 2.38 + ffmpeg 7.1
# ============================================================
import base64 as _b64
import io as _io

_merge_cache = {"path": None, "frames": 0, "fps": 30.0, "w": 0, "h": 0}


def _video_probe(path):
    import imageio_ffmpeg
    import imageio.v3 as iio
    frames, _secs = imageio_ffmpeg.count_frames_and_secs(str(path))
    meta = iio.immeta(str(path), plugin="FFMPEG")
    fps = float(meta.get("fps") or 30.0)
    size = meta.get("size") or (0, 0)
    return int(frames), fps, int(size[0]), int(size[1])


def _video_frame_jpeg(path, idx, maxw=900):
    """用 ffmpeg 直接抽帧并缩放成 JPEG。
    比 imageio 读全尺寸 4K 帧再 PIL 缩放快一个数量级（4K 竖屏实测 2.5s → 0.3s）。"""
    import subprocess
    import imageio_ffmpeg
    fps = float(_merge_cache.get("fps") or 30.0)
    t = max(0.0, float(idx) / fps) if fps else 0.0
    exe = imageio_ffmpeg.get_ffmpeg_exe()
    cmd = [
        exe, "-v", "error", "-ss", "%.3f" % t, "-i", str(path),
        "-frames:v", "1",
        "-vf", "scale=%d:-2" % int(maxw),
        "-f", "image2pipe", "-vcodec", "mjpeg", "-q:v", "4", "-"
    ]
    out = subprocess.run(cmd, capture_output=True, timeout=30, creationflags=_NO_WINDOW)
    if out.returncode != 0 or not out.stdout:
        raise RuntimeError((out.stderr or b"").decode("utf-8", "ignore")[:200])
    return out.stdout


def _sweep_orphan_trainers(why: str = "") -> int:
    """杀掉不属于当前对话的训练进程（命令行里带 _dfl_train.py 的那些）。"""
    global _train_proc
    try:
        import psutil
    except Exception:
        return 0
    mine = _train_proc.pid if _train_proc is not None else 0
    killed = 0
    try:
        for p in psutil.process_iter(["pid", "cmdline"]):
            try:
                cmd = " ".join(p.info.get("cmdline") or [])
                if "_dfl_train.py" not in cmd or p.pid == mine:
                    continue
                # 别误杀另一个 4.1 实例正在用的训练进程：
                # 只有当它的父进程已经没了（或者父进程不是 serve.py）才算孤儿。
                managed = False
                try:
                    par = p.parent()
                    if par is not None and par.is_running():
                        pcmd = " ".join(par.cmdline() or [])
                        managed = ("serve.py" in pcmd) or ("app" in pcmd and "python" in pcmd.lower())
                except Exception:
                    managed = False
                if managed:
                    continue
                p.kill()
                killed += 1
            except Exception:
                continue
    except Exception:
        return killed
    if killed:
        with _train_job_lock:
            _train_job["running"] = False
            _train_job["paused"] = False
        print("[trainer] 清理遗留训练进程 %d 个 %s" % (killed, why), flush=True)
    return killed


@app.on_event("startup")
def _startup_sweep():
    """上一次 4.1 没正常退出时，训练进程会留在这儿。开服先收掉。"""
    _sweep_orphan_trainers("(启动清扫)")


@app.on_event("shutdown")
def _shutdown_cleanup():
    """4.1 退出时把训练进程一起收掉 —— 否则它会变成孤儿继续占显存。”"""
    with _train_job_lock:
        running = _train_job["running"]
    if not running:
        return
    try:
        _webui_post("/request-quit", timeout=2.0)
    except Exception:
        pass
    for _ in range(20):
        if _train_proc is None or _train_proc.poll() is not None:
            return
        time.sleep(0.5)
    _kill_train_tree()




# ============================================================
# DFL 项目格式：DFLJPG = JPEG + 末尾 APP15 段（pickle 的 dict）
# 不依赖 4.0 的 DFLIMG（那要 scipy），自己扫段解析
# ============================================================
class _SafeUnpickler(__import__("pickle").Unpickler):
    """只放行 numpy 数组和基本容器。

    DFLJPG 的 APP15 段是 pickle，/api/merge/open 又能接受任意路径，
    裸 pickle.loads 等于把一个反序列化入口挂到网上。
    4.0 的 core/pathex.safe_pickle_load 就是干这个的，这里补回来。
    """
    _OK = {
        ("numpy.core.multiarray", "_reconstruct"),
        ("numpy._core.multiarray", "_reconstruct"),
        ("numpy", "ndarray"),
        ("numpy", "dtype"),
        ("collections", "OrderedDict"),
        ("builtins", "dict"), ("builtins", "list"), ("builtins", "tuple"),
        ("builtins", "set"), ("builtins", "frozenset"),
        ("builtins", "int"), ("builtins", "float"), ("builtins", "str"),
        ("builtins", "bytes"), ("builtins", "bool"),
    }

    def find_class(self, module, name):
        if (module, name) in self._OK:
            return super().find_class(module, name)
        raise __import__("pickle").UnpicklingError("不允许的类型: %s.%s" % (module, name))


def _safe_loads(raw):
    import io
    try:
        return _SafeUnpickler(io.BytesIO(raw)).load()
    except Exception:
        return {}

def _dfljpg_dict(path):
    """读出 DFLJPG 内嵌的元数据：landmarks / face_type / source_rect / xseg_mask ..."""
    import pickle
    try:
        data = Path(path).read_bytes()
    except Exception:
        return {}
    i, n = 0, len(data)
    while i < n - 1:
        if data[i] != 0xFF:
            break
        marker = data[i + 1]
        if marker in (0xD8, 0xD9) or 0xD0 <= marker <= 0xD7:
            i += 2
            continue
        if i + 4 > n:
            break
        seg_len = int.from_bytes(data[i + 2:i + 4], "big")
        if seg_len < 2:
            break
        if marker == 0xEF:                      # APP15
            payload = data[i + 4:i + 2 + seg_len]
            try:
                d = _safe_loads(payload)
                return d if isinstance(d, dict) else {}
            except Exception:
                return {}
        i += 2 + seg_len
    return {}


def _dfl_project_scan(dirp: Path):
    """扫一个 DFL 项目目录，返回可以拿去合成的帧列表。

    结构约定（和 4.0 一致）：
      data_dst/*.jpg              原帧
      data_dst/aligned/*.jpg      对齐人脸（DFLJPG，带 landmarks）

    有原帧就**用原帧**当预览源 —— 合成器要看到整张画面上的贴脸效果，
    不是一张张裁好的脸。只有 aligned 时才退回用对齐图。
    """
    aligned = dirp / "aligned"
    import os as _os

    def _list(d):
        try:
            return sorted(e.name for e in _os.scandir(d)
                          if e.name.lower().endswith(".jpg"))
        except Exception:
            return []

    # 只取文件名：四万张图如果每个都建 dict + str(Path)，光对象构造就要十几秒
    frames = _list(dirp)
    if frames:
        return frames, str(dirp), ("project" if aligned.is_dir() else "raw")
    faces = _list(aligned) if aligned.is_dir() else []
    return faces, str(aligned), ("aligned" if faces else "raw")


# ============================================================
# DFL 项目格式：DFLJPG = JPEG + 末尾 APP15 段（pickle 的 dict）
# 不依赖 4.0 的 DFLIMG（那要 scipy），自己扫段解析
# ============================================================
def _dfljpg_dict(path):
    """读出 DFLJPG 内嵌的元数据：landmarks / face_type / source_rect / xseg_mask ..."""
    import pickle
    try:
        data = Path(path).read_bytes()
    except Exception:
        return {}
    i, n = 0, len(data)
    while i < n - 1:
        if data[i] != 0xFF:
            break
        marker = data[i + 1]
        if marker in (0xD8, 0xD9) or 0xD0 <= marker <= 0xD7:
            i += 2
            continue
        if i + 4 > n:
            break
        seg_len = int.from_bytes(data[i + 2:i + 4], "big")
        if seg_len < 2:
            break
        if marker == 0xEF:                      # APP15
            payload = data[i + 4:i + 2 + seg_len]
            try:
                d = pickle.loads(payload)
                return d if isinstance(d, dict) else {}
            except Exception:
                return {}
        i += 2 + seg_len
    return {}



# ============================================================
# 合成器：把一段视频变成「能合成」的项目（抽帧 + 对齐）
# 4.1 的提取器只落对齐脸，不落原帧；而合成必须要有原帧才能把脸贴回去。
# 所以这里补一条 视频 → <项目>/000001.jpg + <项目>/aligned/000001_0.jpg 的路。
# ============================================================
_merge_prep = {"running": False, "ok": None, "error": "", "pct": 0, "msg": "",
               "stage": "", "project": "", "frames": 0, "aligned": 0, "video": ""}
_merge_prep_lock = threading.Lock()


def _safe_dirname(name: str) -> str:
    bad = '\\/:*?"<>|'
    out = "".join("_" if c in bad else c for c in (name or "video")).strip()
    return (out or "video")[:60]


def _imread_any(path):
    """OpenCV 的 imread 在 Windows 上按 ANSI 解析路径，中文目录一律返回 None。

    先用 imdecode + np.fromfile 绕过去（读文件是 Python 做的，路径随便中文）。
    """
    import cv2
    import numpy as np
    try:
        buf = np.frombuffer(Path(path).read_bytes(), dtype=np.uint8)
        return cv2.imdecode(buf, cv2.IMREAD_COLOR)
    except Exception:
        return None


def _ffmpeg_exe():
    try:
        import imageio_ffmpeg as _iff
        return _iff.get_ffmpeg_exe()
    except Exception:
        return "ffmpeg"


def _extract_frames(video: Path, out_dir: Path, total_hint: int = 0,
                    on_frame=None) -> int:
    """把视频逐帧抽成 000001.jpg…（6 位，和 aligned 的 source_filename 对齐）。"""
    import subprocess
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("*.jpg"):
        try:
            old.unlink()
        except OSError:
            pass
    cmd = [_ffmpeg_exe(), "-y", "-hide_banner", "-loglevel", "error", "-nostats",
           "-i", str(video), "-q:v", "2", "-start_number", "1",
           "-progress", "pipe:1", str(out_dir / "%06d.jpg")]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                            text=True, encoding="utf-8", errors="replace",
                            creationflags=_NO_WINDOW)
    made = 0
    try:
        for line in proc.stdout:
            line = line.strip()
            if line.startswith("frame="):
                try:
                    made = int(line.split("=", 1)[1])
                except ValueError:
                    continue
                if on_frame:
                    on_frame(made)
    finally:
        proc.wait()
    if proc.returncode != 0:
        raise RuntimeError("ffmpeg 抽帧失败（退出码 %s）" % proc.returncode)
    return len(list(out_dir.glob("*.jpg"))) or made


def _merge_prepare_worker(video: Path, project: Path):
    import threading as _th

    def _set(**kw):
        with _merge_prep_lock:
            _merge_prep.update(kw)

    try:
        project.mkdir(parents=True, exist_ok=True)
        aligned = project / "aligned"
        aligned.mkdir(parents=True, exist_ok=True)

        _set(stage="抽帧", pct=0, msg="分析视频…")
        try:
            import imageio_ffmpeg as _iff
            n, secs = _iff.count_frames_and_secs(str(video))
            total = int(n) if n else 0
            src_fps = (n / secs) if secs else 30.0
        except Exception:
            total, src_fps = 0, 30.0

        # 快路径：同一个视频已经准备过（帧数对得上、对齐脸也在），直接复用
        have = len(list(project.glob("*.jpg")))
        have_aligned = len(list(aligned.glob("*.jpg")))
        if total and have == total and have_aligned > 0:
            _set(frames=have, aligned=have_aligned, pct=100,
                 msg="复用已准备好的项目：%d 帧 · %d 张对齐脸" % (have, have_aligned))
            try:
                names, srcdir, kind = _dfl_project_scan(project)
                _merge_cache.update({"mode": "project", "dir": str(project), "srcdir": srcdir,
                                     "names": names, "path": "", "frames": len(names),
                                     "fps": src_fps, "w": 0, "h": 0})
            except Exception:
                pass
            _set(running=False, ok=True, stage="完成")
            return

        def _on_frame(made):
            pct = int(35 * made / total) if total else 0
            _set(pct=min(35, pct), msg="抽帧 %d/%s" % (made, total or "?"))

        # 只有确认要重跑才清对齐脸（上面快路径会复用它们）
        for old in aligned.glob("*.jpg"):
            try:
                old.unlink()
            except OSError:
                pass
        got = _extract_frames(video, project, total, _on_frame)
        _set(frames=got, pct=35, msg="抽帧完成：%d 帧" % got)

        # 对齐：复用提取器的解码+检测+对齐流水线（它会自己开 JOB）
        _set(stage="对齐", msg="加载检测器…")
        holder = {}

        def _run_align():
            # _align_stream 自己不置 running —— 提取器那边是调用方置的。
            # 不置的话它内部那句 "if not JOB.running: break" 会立刻退出，产出 0 张脸。
            try:
                JOB.running = True
                holder["n"] = _align_stream(str(video), aligned, fps=src_fps,
                                            rotate=0, quality="high", skip=1,
                                            src_fps=src_fps)
            except Exception as e:                     # noqa: BLE001
                holder["err"] = e
            finally:
                JOB.running = False

        th = _th.Thread(target=_run_align, daemon=True)
        th.start()
        while th.is_alive():
            try:
                done = int(getattr(JOB, "done", 0) or 0)
                tot = int(getattr(JOB, "total", 0) or 0) or got or 1
                pct = 35 + int(63 * min(1.0, done / float(tot)))
                _set(pct=min(98, pct), msg="对齐 %d/%d" % (done, tot))
            except Exception:
                pass
            th.join(0.4)
        if "err" in holder:
            raise holder["err"]
        made = int(holder.get("n") or 0)
        _set(aligned=made, pct=100, msg="完成：%d 帧 · %d 张对齐脸" % (got, made))
        if not made:
            _set(ok=False, error="对齐没有产出（视频里没检出人脸？）",
                 msg="对齐失败：0 张脸")

        # 直接把准备好的项目打开，前端不用再点一次
        try:
            names, srcdir, kind = _dfl_project_scan(project)
            _merge_cache.update({"mode": "project", "dir": str(project), "srcdir": srcdir,
                                 "names": names, "path": "", "frames": len(names),
                                 "fps": src_fps, "w": 0, "h": 0})
        except Exception:
            pass
        _set(running=False, ok=True, stage="完成")
    except Exception as e:                             # noqa: BLE001
        _set(running=False, ok=False, error=str(e)[:300], msg="失败", stage="出错")










_det_cache = {}


# 界面下拉里写的是模型名（RetinaFace_10g / TinyMog …），接口里用的是文件名。
# 两张表都认，免得前端还得记住磁盘上的命名。
_DET_ALIAS = {
    "retinaface_10g": "det_10g", "retinaface10g": "det_10g", "det_10g": "det_10g",
    "retinaface_500m": "det_500m", "retinaface500m": "det_500m", "det_500m": "det_500m",
    "retinaface": "det_10g",
}


def _get_detector(name="det_500m"):
    """按需加载检测器（模型在 4.0 的 modelhub 里，只读引用）。

    支持文件名（det_10g / det_500m）和界面名（RetinaFace_10g / TinyMog）。
    """
    key = str(name or "det_500m").strip()
    low = key.lower().replace("-", "_")
    if low.startswith("tinymog"):
        if "tinymog" in _det_cache:
            return _det_cache["tinymog"]
        from ._detect import TinyMogDet
        from ._paths import modelhub_dir
        for p in (modelhub_dir() / "onnx" / "TinyMog" / "TinyMog.onnx",
                  ROOT / "modelhub" / "onnx" / "TinyMog" / "TinyMog.onnx"):
            if p.is_file():
                det = TinyMogDet(str(p), workspace=WORKSPACE, engine_name="tinymog")
                _det_cache["tinymog"] = det
                return det
        raise HTTPException(404, "找不到 TinyMog 模型")

    file_name = _DET_ALIAS.get(low, key)
    if file_name in _det_cache:
        return _det_cache[file_name]
    from ._detect import RetinaFaceDet
    from ._paths import modelhub_dir
    hub = modelhub_dir() / "onnx" / "RetinaFace"
    cand = [hub / (file_name + ".onnx"),
            ROOT / "modelhub" / "onnx" / "RetinaFace" / (file_name + ".onnx")]
    for p in cand:
        if p.is_file():
            det = RetinaFaceDet(p, 640)
            _det_cache[file_name] = det
            return det
    raise HTTPException(404, "找不到检测模型 " + key)


# ============================================================
# DFM 导出（内置在合成器里）
# 用 4.0 的 python（有 torch 2.7 + onnx）跑 tools/export_dfm.py
# ============================================================
_export_job = {"running": False, "name": "", "log": [], "ok": None, "error": None}
from ._paths import engine_python as _engine_python, engine_root as _engine_root

_DFL_ROOT = _engine_root()
_DFL_PY = _engine_python()
_DFL_TOOL = _DFL_ROOT / "tools" / "export_dfm.py"
# 4.0 训练进程自带的监控服务端口（4.1 只从它拉预览，不展示那个界面）
WEBUI_PORT = 6790
# 4.0 的 WebUI 默认密码是 caiji；4.1 自己指定一个，控制接口（保存/退出）要用它
WEBUI_PASSWORD = "dfl41-webui"


def _arch_of(name: str) -> str:
    """从 *_<ARCH>_data.dat 反推架构名。"""
    md = WORKSPACE / "model"
    for dat in md.glob(name + "_*_data.dat"):
        stem = dat.name[: -len("_data.dat")]
        _, _, arch = stem.rpartition("_")
        return arch or "SAEHD"
    return "SAEHD"


def _snapshot_dats(name: str):
    """记下这个模型的 _data.dat 原始字节，导出后要还原。"""
    md = WORKSPACE / "model"
    snap = {}
    for dat in md.glob(name + "_*_data.dat"):
        try:
            snap[dat] = dat.read_bytes()
        except Exception:
            pass
    return snap


def _restore_dats(snap: dict):
    """导出脚本会改 _data.dat（把 iter 归 1 / 重写文件），这里原样还原。"""
    import shutil
    restored = []
    for dat, raw in snap.items():
        try:
            if dat.exists() and dat.read_bytes() == raw:
                continue
            shutil.copy2(dat, str(dat) + ".before-export")
            dat.write_bytes(raw)
            restored.append(dat.name)
        except Exception:
            pass
    return restored


def _export_worker(name: str, arch: str):
    import subprocess
    snap = _snapshot_dats(name)
    cmd = [str(_DFL_PY), str(_DFL_TOOL),
           "--model", arch,
           "--model-dir", str(WORKSPACE / "model"),
           "--force-name", name]
    with _export_job_lock:
        _export_job.update({"running": True, "name": name, "log": [], "ok": None, "error": None})
    try:
        proc = subprocess.Popen(cmd, cwd=str(_DFL_ROOT), creationflags=_NO_WINDOW,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, encoding="utf-8", errors="replace", bufsize=1)
        # 4.0 里那几行 WebUI 提示已经没意义了（服务器被摘掉了），别往日志里塞
        _TRAIN_LOG_SKIP = ("等待模型加载完成后启动 WebUI", "模型已加载，正在启动 WebUI", "[WebUI]")

        for line in proc.stdout:
            line = line.rstrip()
            if not line:
                continue
            if any(k in line for k in _TRAIN_LOG_SKIP):
                continue
            with _export_job_lock:
                _export_job["log"].append(line)
                if len(_export_job["log"]) > 400:
                    _export_job["log"] = _export_job["log"][-400:]
        proc.wait()
        ok = proc.returncode == 0
        restored = _restore_dats(snap)
        size_note = ""
        dfm = WORKSPACE / "model" / (name + ".dfm")
        if dfm.is_file():
            mb = dfm.stat().st_size / 1048576
            if mb < 1:
                ok = False
                size_note = ("导出产物只有 %.2f MB，明显不完整（正常几百 MB）。"
                             "4.0 的 export_dfm.py 对 liae 架构可能只写了结构，"
                             "权重没进去 —— 这是导出脚本的问题，不是界面。") % mb
        with _export_job_lock:
            _export_job["running"] = False
            _export_job["ok"] = ok
            if size_note:
                _export_job["error"] = size_note
                _export_job["log"].append("[!] " + size_note)
            if restored:
                _export_job["log"].append("[!] 导出脚本改动了 _data.dat，已还原: " + ", ".join(restored))
            if not ok:
                _export_job["error"] = "导出进程退出码 %d" % proc.returncode
    except Exception as e:
        with _export_job_lock:
            _export_job["running"] = False
            _export_job["ok"] = False
            _export_job["error"] = str(e)


import threading as _threading
_export_job_lock = _threading.Lock()


# ============================================================
# 模型训练参数：读写 *_data.dat 里的 options
# 结构性字段（分辨率/维度/架构）创建后不可改，白名单挡掉
# ============================================================
_CFG_STRUCTURAL = {"resolution", "archi", "e_dims", "ae_dims", "d_dims",
                   "d_mask_dims", "face_type", "models_opt_on_gpu",
                   "gan_power", "true_face_power", "face_style_power"}


def _model_dat(name: str):
    md = WORKSPACE / "model"
    cands = sorted(md.glob(name + "_*_data.dat"))
    return cands[0] if cands else None




# 与 4.0 Trainer.py 的 _HOT_KEYS 一一对应：这些参数训练中改会立即生效，
# 其余（分辨率/架构/维度/batch/优化器/预训练…）必须重启训练。
_HOT_OPTIONS = {
    "crash_threshold", "backup_interval", "max_backups",
    "face_style_power", "bg_style_power",
    "masked_training", "blur_out_mask", "eyes_mouth_prio", "uniform_yaw",
    "lr", "lr_cos", "lr_dropout",
    "random_src_flip", "random_dst_flip", "random_hsv_power",
    "random_warp", "random_occlusion", "random_noise",
    "random_color_power", "random_color_alg",
    "ct_mode",
    # 刚性形变范围
    "rotation_range", "scale_range", "t_range",
    # 眼睛 / 嘴巴缩放、边缘增强
    "eyes_scale", "mouth_scale", "edge_enhance_power",
    # Inter code 分布日志间隔
    "log_code_stats",
}










# ============================================================
# 合成导出：调 4.0 的 main.py merge --pipeline（ONNX DFM，不需要 torch 权重）
# ============================================================
_merge_job = {"running": False, "log": [], "ok": None, "error": None, "out": "",
              "pct": 0, "msg": ""}


# 界面上的「遮罩来源 × 遮罩模式」→ MergeStudio 的 mask_mode 整数
#   0 full · 1 dst · 2 learned-prd · 3 learned-dst · 4 learned-prd×learned-dst
#   6 XSeg-prd · 7 XSeg-dst · 8 XSeg-prd×XSeg-dst
_MASK_MODE_LABEL = {
    "prd×dst": 4, "prd": 2, "dst": 1, "full": 0,
    "xseg-prd×dst": 8, "xseg-prd": 6, "xseg-dst": 7,
}


def _mask_mode_int(v, default=6):
    if isinstance(v, bool):
        return default
    if isinstance(v, int):
        return v
    try:
        return int(str(v).strip())
    except (TypeError, ValueError):
        pass
    key = str(v or "").strip().lower().replace("*", "×").replace("x", "x")
    return _MASK_MODE_LABEL.get(key, default)


def _merge_settings_to_config(settings: dict) -> dict:
    """把 4.1 界面的合成参数翻成 MergeStudio 的键名。"""
    settings = settings or {}
    cfg = {
        "mode": settings.get("mode", "overlay"),
        "mask_mode": _mask_mode_int(settings.get("mask_mode"), 6),
        "color_transfer_mode": settings.get("color_transfer_mode",
                                           settings.get("color_transfer", "none")),
        "erode_mask_modifier": settings.get("erode_mask_modifier",
                                            settings.get("erode_mask", 0)),
        "blur_mask_modifier": settings.get("blur_mask_modifier",
                                           settings.get("blur_mask", 0)),
    }
    if settings.get("face_margin") is not None:
        cfg["face_margin"] = settings["face_margin"]
    return cfg


def _merge_worker(input_dir: str, out_mp4: str, aligned_dir: str, dfm: str,
                  settings: dict, mask_onnx: str, fps: float = 25.0, workers: int = 0,
                  encoder: str = "h264_nvenc", hwaccel: str = "auto"):
    """跑 _merge_runner.py（4.0 的 python + MergeStudio 导出管线）。

    以前这里是 `main.py merge`，而 mainscripts/Merger.py 依赖一个不存在的
    merger 包，跑到一半必崩；现在换成直接调 run_export_pipeline。
    """
    import subprocess
    py = _engine_python()
    runner = Path(__file__).with_name("_merge_runner.py")
    if not py.is_file():
        raise RuntimeError("找不到 4.0 的 python: %s" % py)
    if not runner.is_file():
        raise RuntimeError("找不到合成桥脚本: %s" % runner)
    if not (_DFL_ROOT / "MergeStudio").is_dir():
        raise RuntimeError("4.0 根目录下没有 MergeStudio: %s" % _DFL_ROOT)

    cmd = [str(py), str(runner),
           "--frames", input_dir,
           "--aligned", aligned_dir,
           "--dfm", dfm,
           "--out", out_mp4,
           "--config", json.dumps(_merge_settings_to_config(settings), ensure_ascii=False),
           "--dfl-root", str(_DFL_ROOT),
           "--fps", str(float(fps or 25.0))]
    if mask_onnx:
        cmd += ["--xseg", mask_onnx]
    if workers and int(workers) > 0:
        cmd += ["--workers", str(int(workers))]
    cmd += ["--encoder", str(encoder or "h264_nvenc")]
    cmd += ["--image-format", str((settings or {}).get("image_format") or "jpg")]
    cmd += ["--hwaccel", str(hwaccel if hwaccel is not None else "auto")]
    # 4.0 根下没有 ffmpeg/，用 4.1 自带的 imageio-ffmpeg（只读引用）
    try:
        import imageio_ffmpeg as _iff
        _ff = _iff.get_ffmpeg_exe()
        if _ff and Path(_ff).is_file():
            cmd += ["--ffmpeg", _ff]
    except Exception:
        pass

    with _merge_job_lock:
        _merge_job.update({"running": True, "log": [], "ok": None, "error": None,
                           "out": out_mp4, "pct": 0, "msg": "启动合成…"})
    try:
        env_note = "PYTHONIOENCODING=utf-8"
        import os as _os
        env = dict(_os.environ, PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1")
        global _merge_proc
        proc = subprocess.Popen(cmd, cwd=str(_DFL_ROOT), env=env, creationflags=_NO_WINDOW,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, encoding="utf-8", errors="replace", bufsize=1)
        _merge_proc = proc
        for line in proc.stdout:
            line = line.rstrip()
            if not line:
                continue
            if line.startswith("@@P "):
                parts = line.split(" ", 3)
                try:
                    pct = int(float(parts[2]) * 100)
                except Exception:
                    pct = None
                try:
                    stage = int(parts[1])
                except Exception:
                    stage = -1
                with _merge_job_lock:
                    if pct is not None:
                        _merge_job["pct"] = max(0, min(100, pct))
                    if stage >= 0:
                        _merge_job["stage"] = stage
                    if len(parts) > 3:
                        _merge_job["msg"] = parts[3][:120]
                continue
            if line.startswith("@@DONE "):
                with _merge_job_lock:
                    _merge_job["out"] = line[7:].strip()
                continue
            if line.startswith("@@ERR "):
                with _merge_job_lock:
                    _merge_job["error"] = line[6:].strip()
                continue
            with _merge_job_lock:
                _merge_job["log"].append(line)
                if len(_merge_job["log"]) > 400:
                    _merge_job["log"] = _merge_job["log"][-400:]
        proc.wait()
        ok = proc.returncode == 0
        with _merge_job_lock:
            _merge_job["running"] = False
            _merge_job["ok"] = ok
            if ok:
                _merge_job["pct"] = 100
                _merge_job["msg"] = "合成完成"
            elif not _merge_job.get("error"):
                _merge_job["error"] = "合成进程退出码 %d" % proc.returncode
    except Exception as e:
        with _merge_job_lock:
            _merge_job["running"] = False
            _merge_job["ok"] = False
            _merge_job["error"] = str(e)


import threading as _th2
_merge_job_lock = _th2.Lock()
_merge_proc = None          # 当前合成子进程，取消时用






def _merge_frame_image(idx: int):
    """按当前打开的东西取一帧：视频模式解码，项目模式直接读图。

    返回 (图像, 总帧数)。合成器两种模式都要能检测/分析，所以抽出来共用。
    """
    import cv2
    total = int(_merge_cache.get("frames") or 0)
    if _merge_cache.get("mode") == "project":
        names = _merge_cache.get("names") or []
        srcdir = _merge_cache.get("srcdir") or ""
        if not names:
            raise HTTPException(409, "还没有打开项目")
        idx = max(0, min(int(idx), len(names) - 1))
        img = _imread_any(Path(srcdir) / names[idx])
        if img is None:
            raise HTTPException(500, "读不了这张图: " + names[idx])
        return img, len(names)

    path = _merge_cache.get("path")
    if not path:
        raise HTTPException(409, "还没有打开视频")
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise HTTPException(500, "打不开视频")
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, int(idx)))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise HTTPException(500, "取不到第 %d 帧" % idx)
    return frame, total










# ============================================================
# 遮罩编辑器（照搬 MaskProcessor/ui 的 WebUI）
# 只实现不依赖模型的三个：开项目 / 读图 / 存遮罩
# 需要 AI 的功能（SAM / BiSeNet / XSeg）返回 501，界面上会提示
# ============================================================
_mask_state = {"dir": None, "files": [], "index": 0}






def _mask_ai_stub():
    raise HTTPException(501, "这个功能要加载 AI 模型（SAM / BiSeNet / XSeg），当前只接了手绘与保存")

























@app.get("/")
def index():
    """index.html 里的 ?v=N 按**文件修改时间**自动重写。

    之前是写死的 ?v=3，Chromium 会一直复用缓存 —— 改了 app.js 也看不到效果，
    表现出来就是"页面还是老样子"。这里每次请求都按 mtime 生成新版本号，
    再配上 no-store，改完刷新就能生效。
    """
    import re as _re
    html = (WEB / "index.html").read_text(encoding="utf-8")

    def _bust(m):
        sub, name = m.group(1), m.group(2)
        try:
            v = int((WEB / sub / name).stat().st_mtime)
        except OSError:
            v = 0
        return "/static/%s/%s?v=%d" % (sub, name, v)

    html = _re.sub(r"/static/(js|css)/([\w.\-]+)\?v=\d+", _bust, html)
    return Response(html, media_type="text/html", headers={"Cache-Control": "no-store"})


app.mount("/static", StaticFiles(directory=WEB), name="static")
if (WEB / "mask-editor").is_dir():
    app.mount("/mask-editor", StaticFiles(directory=WEB / "mask-editor", html=True), name="mask-editor")


@app.middleware("http")
async def _no_cache(request, call_next):
    """开发期不让 Chromium 缓存界面资源，否则改完 CSS 看不到效果。"""
    resp = await call_next(request)
    resp.headers["Cache-Control"] = "no-store, must-revalidate"
    return resp


# ---------------- B 站扫码登录（已拆到 routes_bili.py） ----------------
from .routes_bili import router as _bili_router
app.include_router(_bili_router)


# ---------------- 数据处理：真实扫描 ----------------
import threading

JOBS: dict = {}


def _list_images(folder: Path):
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    return sorted([p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in exts])


def _lap_var(img) -> float:
    """拉普拉斯方差：越小越糊。"""
    import numpy as np
    g = img.convert("L").resize((256, 256))
    a = np.asarray(g, dtype=np.float32)
    k = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32)
    h, w = a.shape
    out = np.zeros((h - 2, w - 2), dtype=np.float32)
    for dy in range(3):
        for dx in range(3):
            out += k[dy, dx] * a[dy:dy + h - 2, dx:dx + w - 2]
    return float(out.var())


def _dhash(img, size=8) -> int:
    g = img.convert("L").resize((size + 1, size))
    import numpy as np
    a = np.asarray(g, dtype=np.int16)
    bits = (a[:, 1:] > a[:, :-1]).flatten()
    v = 0
    for b in bits:
        v = (v << 1) | int(b)
    return v


# ---------------- 数据集体检（One Shot）（已拆到 routes_oneshot.py） ----------------

# ---------------- 面部 68 点（已拆到 routes_face.py） ----------------
from .routes_face import router as _face_router
app.include_router(_face_router)

# ---------------- 账号（已拆到 routes_account.py） ----------------

# ---------------- 模型库 ----------------
MODEL_DIR = WORKSPACE / "model"
CACHE_DIR = MODEL_DIR / ".cache"


def _read_dat(dat_path: Path) -> dict:
    """读 *_data.dat：iter / options / loss_history / sample_for_preview。"""
    import pickle
    with open(dat_path, "rb") as fh:
        return pickle.load(fh)


def _spark(loss_history, n=2500):
    """取最后 n 次迭代，拆成 src / dst 两条曲线。"""
    tail = loss_history[-n:] if len(loss_history) > n else loss_history
    src, dst = [], []
    for e in tail:
        if isinstance(e, (list, tuple)) and len(e) >= 2:
            try:
                src.append(round(float(e[0]), 5))
                dst.append(round(float(e[1]), 5))
            except Exception:
                pass
    return {"src": src, "dst": dst}


def _preview_png(dat: dict, out: Path) -> bool:
    """sample_for_preview 里是一条 (n,3,H,W) 的 float32 序列，通道是 BGR。
       拼成一条横图，前端再按 index 切片。"""
    try:
        import numpy as np
        from PIL import Image

        sp = dat.get("sample_for_preview")
        if not sp:
            return False
        arr = np.asarray(sp[0][0], dtype=np.float32)
        if arr.ndim != 4 or arr.shape[1] != 3:
            return False
        n, _, h, w = arr.shape
        strip = Image.new("RGB", (w * n, h), (0, 0, 0))
        for i in range(n):
            img = np.clip(arr[i], 0.0, 1.0)
            rgb = img[::-1].transpose(1, 2, 0)          # BGR -> RGB
            strip.paste(Image.fromarray((rgb * 255.0).astype(np.uint8), "RGB"), (i * w, 0))
        out.parent.mkdir(parents=True, exist_ok=True)
        strip.save(out, "PNG", optimize=True)
        return True
    except Exception:
        return False


def _read_params_cache(dat_path: Path):
    """读 4.0 写的 <model>_data.dat.params（1.7KB）。

    指纹（size + mtime_ns）必须和 data.dat 对得上 —— 4.0 那边改了文件就会作废，
    这里也照同样规则判，避免拿过期参数去做模型卡片。
    """
    import pickle
    try:
        st = dat_path.stat()
        data = pickle.loads(Path(str(dat_path) + ".params").read_bytes())
        if data.get("version") != 1:
            return None
        if data.get("src_size") != st.st_size or data.get("src_mtime_ns") != st.st_mtime_ns:
            return None
        return data
    except Exception:
        return None


def _read_preview_json(dat_path: Path):
    """读 4.0 训练时写的 <model>_preview.json（loss 尾巴 + iter）。"""
    try:
        return json.loads(Path(str(dat_path)[: -len("_data.dat")] + "_preview.json").read_text(encoding="utf-8"))
    except Exception:
        return None


def _preview_jpg(dat_path: Path) -> Path:
    return Path(str(dat_path)[: -len("_data.dat")] + "_preview.jpg")


def _fast_model_entry(dat_path: Path) -> dict:
    """两个小文件就够：参数缓存 + 预览快照。都不在就返回 None 走老路。

    老路要 pickle 整个 data.dat（122MB 的模型要 0.85s，每个模型每次保存后都要重来
    一遍）；新路读 1.7KB + 40KB，毫秒级。
    """
    params = _read_params_cache(dat_path)
    if params is None:
        return None
    name = dat_path.name[: -len("_data.dat")]
    model_name, _, archi_from_name = name.rpartition("_")
    if not model_name:
        model_name, archi_from_name = name, ""

    opt = params.get("options") or {}
    pj = _read_preview_json(dat_path)
    spark = {"src": [], "dst": []}
    if pj:
        tail = pj.get("loss_tail") or []
        spark = {"src": [round(float(e[0]), 5) for e in tail if len(e) >= 2],
                 "dst": [round(float(e[1]), 5) for e in tail if len(e) >= 2]}

    has_prev = _preview_jpg(dat_path).is_file()
    dfm = dat_path.parent / (model_name + ".dfm")
    _archi = str(opt.get("archi", archi_from_name)).lower()
    return {
        "name": model_name,
        "archi": _archi,
        "archi_parsed": _archi_parts(_archi),
        "face_type_label": _FACE_LABELS.get(str(opt.get("face_type", "")), str(opt.get("face_type", ""))),
        "resolution": int(opt.get("resolution", 0) or 0),
        "face_type": str(opt.get("face_type", "")),
        "ae_dims": int(opt.get("ae_dims", 0) or 0),
        "e_dims": int(opt.get("e_dims", 0) or 0),
        "d_dims": int(opt.get("d_dims", 0) or 0),
        "mask_dims": int(opt.get("d_mask_dims", 0) or 0),
        "batch_size": int(opt.get("batch_size", 0) or 0),
        "bf16": str(opt.get("use_bf16", "False")) == "True",
        "optimizer": str(opt.get("optimizer", "")),
        "iter": int(params.get("iter", 0) or 0),
        "spark": spark,
        "has_dfm": dfm.exists(),
        "weights": "npy" if any(dat_path.parent.glob(model_name + "_*decoder*.npy")) else "pth",
        "trained_at": int(dat_path.stat().st_mtime),
        "has_preview": has_prev,
        "preview_url": ("/api/models/" + model_name + "/preview.png") if has_prev else None,
    }


def _model_entry(dat_path: Path) -> dict:
    fast = _fast_model_entry(dat_path)
    if fast is not None:
        return fast

    name = dat_path.name[: -len("_data.dat")]
    model_name, _, archi_from_name = name.rpartition("_")
    if not model_name:
        model_name, archi_from_name = name, ""

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_json = CACHE_DIR / (model_name + ".json")
    preview = CACHE_DIR / (model_name + ".png")
    mtime = dat_path.stat().st_mtime

    cached = None
    try:
        cached = json.loads(cache_json.read_text(encoding="utf-8"))
    except Exception:
        cached = None
    if cached and abs(float(cached.get("mtime", 0)) - mtime) < 1 and preview.exists():
        entry = dict(cached)
        entry.pop("mtime", None)
        entry["preview_url"] = "/api/models/" + model_name + "/preview.png"
        return entry

    try:
        dat = _read_dat(dat_path)
    except Exception:
        dat = {}

    opt = dat.get("options") or {}
    loss = dat.get("loss_history") or []
    spark = _spark(loss)

    ok_prev = _preview_png(dat, preview)

    weights = "npy" if any(dat_path.parent.glob(model_name + "_*decoder*.npy")) else "pth"
    dfm = dat_path.parent / (model_name + ".dfm")

    _archi2 = str(opt.get("archi", archi_from_name)).lower()
    entry = {
        "name": model_name,
        "archi": _archi2,
        "archi_parsed": _archi_parts(_archi2),
        "face_type_label": _FACE_LABELS.get(str(opt.get("face_type", "")), str(opt.get("face_type", ""))),
        "resolution": int(opt.get("resolution", 0) or 0),
        "face_type": str(opt.get("face_type", "")),
        "ae_dims": int(opt.get("ae_dims", 0) or 0),
        "e_dims": int(opt.get("e_dims", 0) or 0),
        "d_dims": int(opt.get("d_dims", 0) or 0),
        "mask_dims": int(opt.get("d_mask_dims", 0) or 0),
        "batch_size": int(opt.get("batch_size", 0) or 0),
        "bf16": str(opt.get("use_bf16", "False")) == "True",
        "optimizer": str(opt.get("optimizer", "")),
        "iter": int(len(loss)),
        "spark": spark,
        "has_dfm": dfm.exists(),
        "weights": weights,
        "trained_at": int(dat_path.stat().st_mtime),
        "has_preview": bool(ok_prev),
    }

    try:
        payload = dict(entry)
        payload["mtime"] = mtime
        cache_json.write_text(json.dumps(payload), encoding="utf-8")
    except Exception:
        pass

    entry["preview_url"] = "/api/models/" + model_name + "/preview.png" if ok_prev else None
    return entry


def _training_pid():
    lock = MODEL_DIR / ".train.lock"
    try:
        return int(lock.read_text(encoding="utf-8").strip())
    except Exception:
        return 0


# ============================================================
# GPU 探测 + 新建模型（"别让用户为任何选项焦虑"）
# ============================================================
# 4.1 自己的 python 没有 torch，所以显卡信息走 nvidia-smi（一次约 80ms）。
def _gpu_list():
    import shutil as _shutil
    import subprocess as _sp
    exe = _shutil.which("nvidia-smi") or r"C:\Windows\System32\nvidia-smi.exe"
    if not Path(exe).is_file():
        return []
    try:
        out = _sp.run([exe, "--query-gpu=index,name,memory.total,memory.free",
                       "--format=csv,noheader,nounits"],
                      capture_output=True, text=True, timeout=8,
                      creationflags=_NO_WINDOW).stdout
    except Exception:
        return []
    gpus = []
    for line in out.strip().splitlines():
        parts = [x.strip() for x in line.split(",")]
        if len(parts) < 4:
            continue
        try:
            gpus.append({"index": int(parts[0]), "name": parts[1],
                         "total_gb": round(int(parts[2]) / 1024.0, 1),
                         "free_gb": round(int(parts[3]) / 1024.0, 1)})
        except Exception:
            continue
    return gpus


# 档位表：数值不是拍的，是拿本机真实训练过的模型反推的
#   256 / df-ud   权重约 0.49GB（DFLitetest、model_004 都是这套配置跑起来的）
#   384 / liae-ud 权重约 2.04GB（384wf，300 万次迭代）
# 训练显存 ≈ 权重 × 4（参数 1 + 优化器状态 ms/vs 2 + 梯度 1）+ 激活
_PRESETS = {
    "fast":     {"label": "快",   "resolution": 256, "archi": "df-ud",
                 "ae_dims": 256, "e_dims": 64, "d_dims": 128, "d_mask_dims": 42,
                 "weights_gb": 0.49,
                 "note": "最快出结果，细节一般，适合先跑通"},
    "balanced": {"label": "均衡", "resolution": 384, "archi": "liae-ud",
                 "ae_dims": 352, "e_dims": 102, "d_dims": 102, "d_mask_dims": 32,
                 "weights_gb": 2.04,
                 "note": "画质和速度都能接受，长期训练推荐"},
    "quality":  {"label": "极致", "resolution": 512, "archi": "liae-ud",
                 "ae_dims": 352, "e_dims": 102, "d_dims": 102, "d_mask_dims": 32,
                 "weights_gb": 2.04,
                 "note": "最高分辨率，需要大显存，速度慢"},
}


def _recommend_for(free_gb: float):
    """按可用显存给一档配置 + batch。

    实测：8GB 的卡跑 256/df-ud/batch4 没问题；384/liae-ud 权重 2.04GB×4 ≈ 8.2GB，
    8GB 上放不下 —— 所以 384 只在 14GB 以上才推荐。
    """
    table = (("quality", 24.0, 8), ("quality", 20.0, 4),
             ("balanced", 14.0, 4), ("fast", 10.0, 8), ("fast", 6.0, 4))
    for key, min_gb, batch in table:
        if free_gb >= min_gb:
            p = dict(_PRESETS[key])
            p.update({"preset": key, "batch_size": batch,
                      "need_gb": round(p["weights_gb"] * 4 + 1.5, 1)})
            return p
    p = dict(_PRESETS["fast"])
    p.update({"preset": "fast", "batch_size": 2, "need_gb": round(0.49 * 4 + 1.5, 1)})
    return p


def _archi_spec():
    """借用 4.0 的 core/archi_names.py —— 架构/子分支/算子的合法性只有一份出处，
    4.1 不复制那几张表。core/__init__.py 是空的，import 很轻。"""
    import importlib
    import sys as _sys
    root = str(_DFL_ROOT)
    added = root not in _sys.path
    if added:
        _sys.path.insert(0, root)
    try:
        return importlib.import_module("core.archi_names")
    finally:
        if added:
            try:
                _sys.path.remove(root)
            except ValueError:
                pass


# 脸型：值 ↔ 中文。取值来自 4.0 Model_SAEHD 的 face_type 映射表
_FACE_TYPES = [
    {"v": "h", "label": "半脸（只要脸，不要额头下巴）"},
    {"v": "mf", "label": "中全脸"},
    {"v": "f", "label": "全脸（默认）"},
    {"v": "wf", "label": "整头（含额头和下巴，推荐）"},
    {"v": "head", "label": "头肩（要脖子和肩膀）"},
]


_FACE_LABELS = {f["v"]: f["label"] for f in _FACE_TYPES}


def _archi_parts(archi):
    """把 'df-udt-l' 拆成给界面看的字段（显示名 / 子分支 / 算子）。"""
    try:
        spec = _archi_spec()
        parsed = spec.parse_archi_string(archi)
        if parsed is None:
            return {"display": str(archi), "sub": "", "kernel": "", "lite": False}
        base, opts, mod = parsed
        lite = spec.archi_is_lite(mod)
        kernel = ""
        for lab, m, _d in spec.KERNEL_CHOICES:
            if m == (mod or ""):
                kernel = lab
                break
        return {"display": ("DF" if base == "df" else "LIAE") + (" Lite" if lite else ""),
                "sub": ("-" + opts) if opts else "",
                "kernel": kernel, "lite": bool(lite)}
    except Exception:
        return {"display": str(archi), "sub": "", "kernel": "", "lite": False}


# 规格表是静态的（来自 4.0 源码），探到一次就留着；显卡信息缓存 5 秒。
# 不缓存的话 /api/model_spec 第一次要 600ms —— 面板会"迟一拍"才出现。
_SPEC_CACHE = {"tables": None, "gpu": [], "gpu_at": 0.0}


def _spec_tables():
    if _SPEC_CACHE["tables"] is None:
        spec = _archi_spec()
        _SPEC_CACHE["tables"] = {
            "archi": [{"name": n, "base": b, "lite": bool(l),
                       "desc": spec.ARCHI_DESCRIPTIONS.get(n, "")}
                      for n, b, l in spec.ARCHI_CHOICES],
            "subarchi": [{"v": s, "desc": spec.SUBARCHI_DESCRIPTIONS.get(s, "")}
                         for s in spec.SUBARCHI_CHOICES],
            "kernel": [{"label": lab, "mod": mod, "desc": d} for lab, mod, d in spec.KERNEL_CHOICES],
            "default_kernel": spec.default_kernel_label(),
            "face_types": _FACE_TYPES,
            "dims": {"ae_dims": 256, "e_dims": 64, "d_dims": 128, "d_mask_dims": 42},
            "dims_liae": {"ae_dims": 352, "e_dims": 102, "d_dims": 102, "d_mask_dims": 32},
            "limits": {"resolution": [64, 640], "ae_dims": [64, 1024],
                       "e_dims": [16, 256], "d_dims": [16, 256], "d_mask_dims": [8, 128],
                       "batch_size": [1, 64]},
        }
    return _SPEC_CACHE["tables"]


def _gpu_cached(ttl=5.0):
    now = time.time()
    if not _SPEC_CACHE["gpu"] or (now - _SPEC_CACHE["gpu_at"]) > ttl:
        _SPEC_CACHE["gpu"] = _gpu_list()
        _SPEC_CACHE["gpu_at"] = now
    return _SPEC_CACHE["gpu"]






_BAD_NAME_CHARS = set('/\\:*?"<>|') | {chr(i) for i in range(32)}


def _safe_model_name(name: str) -> str:
    name = (name or "").strip()
    if not name or name in (".", ".."):
        raise HTTPException(400, "模型名不能为空")
    if len(name) > 64:
        raise HTTPException(400, "模型名太长（最多 64 个字）")
    if any(ch in _BAD_NAME_CHARS for ch in name) or ".." in name:
        raise HTTPException(400, '模型名不能包含 / \\ : * ? " < > | 或 ..')
    return name




# ============================================================
# 训练：调 4.0 的 main.py train（真起进程，日志回传）
# ============================================================
_train_job = {"running": False, "name": "", "log": [], "ok": None, "error": None,
              "iter": 0, "it_s": 0.0, "iter_at": 0.0, "started_at": 0.0, "target": 0,
              "paused": False, "saving": False, "lr": 0.0,
              "loss_src": None, "loss_dst": None,
              "history": [], "webui_ready": False, "stopping": False}
_train_proc = None        # 4.0 训练主进程（控制/清理都要它）
# 最后一次拿到的实时预览（4.0 的 /preview 是破坏性 pop，得自己留一份给前端反复读）
_last_preview_state = {"body": None, "iter": 0, "at": 0.0}


def _train_worker(name: str, cpu_only: bool):
    import re
    import subprocess
    py = _engine_python()
    main_py = _DFL_ROOT / "main.py"
    env = dict(**__import__("os").environ)
    env["DFL_SILENT_INPUT"] = "1"
    # 不设这个，4.0 按系统代码页（GBK）往管道里写，我们按 UTF-8 解出来全是乱码：
    # 「正在创建备份」匹配不到、日志尾巴也没法看。
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUNBUFFERED"] = "1"

    # src/dst 是必填：优先用模型 options 里记的路径，否则用工作区默认的 aligned 目录
    import pickle as _pk
    src_dir = dst_dir = ""
    try:
        dat = _model_dat(name)
        if dat:
            opts = (_pk.loads(dat.read_bytes()).get("options") or {})
            src_dir = str(opts.get("src") or "")
            dst_dir = str(opts.get("dst") or "")
    except Exception:
        pass
    if not src_dir:
        src_dir = str(WORKSPACE / "data_src" / "aligned") if (WORKSPACE / "data_src" / "aligned").is_dir() else str(WORKSPACE / "data_src")
    if not dst_dir:
        dst_dir = str(WORKSPACE / "data_dst" / "aligned") if (WORKSPACE / "data_dst" / "aligned").is_dir() else str(WORKSPACE / "data_dst")

    # 走包装脚本：把 4.0 的 WebUI 和自动开浏览器摘掉，其余照旧
    boot = Path(__file__).resolve().parent / "_dfl_train.py"
    entry = str(boot) if boot.is_file() else str(main_py)
    cmd = [str(py), entry, "train",
           "--training-data-src-dir", src_dir,
           "--training-data-dst-dir", dst_dir,
           "--model-dir", str(WORKSPACE / "model"),
           "--model", "SAEHD",
           "--force-model-name", name,
           "--no-preview",
           "--silent-start",
           "--webui-port", str(WEBUI_PORT),
           "--webui-password", WEBUI_PASSWORD]
    if cpu_only:
        cmd.append("--cpu-only")

    # 目标迭代次数（模型 options 里记着，用来算 ETA）
    target = 0
    try:
        dat = _model_dat(name)
        if dat:
            target = int((_pk.loads(dat.read_bytes()).get("options") or {}).get("target_iter", 0) or 0)
    except Exception:
        target = 0

    with _train_job_lock:
        # 上一次的 iter / loss / history 必须清掉：否则第二次开训时界面立刻显示
        # "训练中 + 上次的迭代数"，而模型其实还在加载 —— 用户看到的就是假的进度。
        _train_job.update({"running": True, "name": name, "log": [], "ok": None,
                           "error": None, "webui_port": WEBUI_PORT,
                           "it_s": 0.0, "iter_at": 0.0, "started_at": time.time(),
                           "target": target,
                           "iter": 0, "iter_seen": False, "lr": 0.0,
                           "loss_src": None, "loss_dst": None, "history": [],
                           "paused": False, "saving": False, "stopping": False,
                           "last_backup_iter": 0, "pid": None})
    global _train_proc
    # 4.0 训练进度行的真实格式：
    #   [08:42:26][#002012][0061ms][lr:5.00e-05][1.0574][1.8756]
    #   时间       迭代     每轮耗时   学习率        src_loss dst_loss
    # 以前用 r"(\d+)\s*(?:it|iter)" 抓，抓到的是 tqdm 进度条里的 "73.25it/s" —— 迭代数
    # 和速度全是假的。这里按行首的 [#数字] 精确匹配。
    ITER_RE = re.compile(r"^\[[\d:]+\]\[#(\d+)\]\[(\d+)ms\]\[lr:([0-9.eE+-]+)\]"
                         r"\[([0-9.]+)\]\[([0-9.]+)\]")
    try:
        proc = subprocess.Popen(cmd, cwd=str(_DFL_ROOT), env=env, creationflags=_NO_WINDOW,
                                stdin=subprocess.DEVNULL,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, encoding="utf-8", errors="replace", bufsize=1)
        _train_proc = proc
        with _train_job_lock:
            _train_job["pid"] = proc.pid
        for line in proc.stdout:
            line = line.rstrip()
            if not line:
                continue
            m = ITER_RE.match(line)
            with _train_job_lock:
                _train_job["log"].append(line[:400])
                if len(_train_job["log"]) > 300:
                    _train_job["log"] = _train_job["log"][-300:]
                if "正在创建备份" in line:
                    # 4.0 打这行时正在备份，当前 iter 就是备份点
                    _train_job["last_backup_iter"] = _train_job["iter"]
                    _train_job["last_backup_at"] = time.time()
                if m:
                    it = int(m.group(1))
                    ms = max(1, int(m.group(2)))
                    _train_job["iter_seen"] = True
                    _train_job["iter"] = it
                    _train_job["iter_at"] = time.time()
                    _train_job["lr"] = float(m.group(3))
                    _train_job["loss_src"] = float(m.group(4))
                    _train_job["loss_dst"] = float(m.group(5))
                    inst = 1000.0 / ms
                    old_rate = _train_job["it_s"] or inst
                    _train_job["it_s"] = round(old_rate * 0.85 + inst * 0.15, 2)
                    hist = _train_job["history"]
                    hist.append([it, round(float(m.group(4)), 4), round(float(m.group(5)), 4)])
                    if len(hist) > 400:
                        del hist[:-400]
        proc.wait()
        ok = proc.returncode == 0
        with _train_job_lock:
            manual = bool(_train_job.get("stopping"))
            _train_job["running"] = False
            _train_job["paused"] = False
            _train_job["stopping"] = False
            # 用户点「停止」时进程可能是被强杀的（退出码非 0），不该显示成"训练失败"
            _train_job["ok"] = ok or manual
            if not ok and not manual:
                rc = proc.returncode
                # Windows 上被外部杀掉会报 4294967295（其实就是 -1），直接印出来没人看得懂
                if rc is not None and rc >= 0x80000000:
                    rc -= 0x100000000
                if rc is None or rc < 0:
                    _train_job["error"] = "训练进程被外部结束（退出码 %s）—— 是不是任务管理器里结束了它？" % rc
                else:
                    _train_job["error"] = "训练进程退出码 %d" % rc
            elif manual:
                _train_job["error"] = None
    except Exception as e:
        with _train_job_lock:
            _train_job["running"] = False
            _train_job["ok"] = False
            _train_job["error"] = str(e)


import threading as _th3
_train_job_lock = _th3.Lock()


def _repair_option_strings(name: str) -> list:
    """把 options 里 "1.0（关闭）" 这类被当数值写进去的字符串还原成数字。

    4.0 读 options 时直接 float()，带着中文说明就会崩：
        ValueError: could not convert string to float: '1.0（关闭）'
    这是 4.1 早先的参数页 bug 留下的坑，这里在启动前兜一次。
    """
    import pickle as _pk
    import re as _re
    dat = _model_dat(name)
    if dat is None:
        return []
    try:
        data = _pk.loads(dat.read_bytes())
    except Exception:
        return []
    opts = data.get("options") or {}
    fixed = []
    pat = _re.compile(r"^\s*([0-9]+(?:\.[0-9]+)?)\s*[（(].*[）)]\s*$")
    for k, v in list(opts.items()):
        if not isinstance(v, str):
            continue
        m = pat.match(v)
        if not m:
            continue
        num = float(m.group(1))
        opts[k] = int(num) if num.is_integer() and "." not in m.group(1) else num
        fixed.append(k)
    if not fixed:
        return []
    data["options"] = opts
    try:
        dat.with_suffix(".dat.bak").write_bytes(dat.read_bytes())
        tmp = dat.with_suffix(".dat.tmp")
        tmp.write_bytes(_pk.dumps(data))
        os.replace(str(tmp), str(dat))
    except Exception:
        return []
    return fixed




def _webui_post(path: str, payload: dict | None = None, timeout: float = 3.0):
    """给 4.0 训练进程里的 WebUI 发控制指令。没起来就抛异常。"""
    import json as _json
    import urllib.request
    body = dict(payload or {})
    body.setdefault("password", WEBUI_PASSWORD)
    req = urllib.request.Request(
        "http://127.0.0.1:%d%s" % (WEBUI_PORT, path),
        data=_json.dumps(body).encode(), method="POST",
        headers={"Content-Type": "application/json", "User-Agent": "dfl41"})
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(req, timeout=timeout) as r:
        return r.read().decode("utf-8", "replace")


def _kill_train_tree(force_after: float = 0.0):
    """把训练进程连同子进程收掉（psutil 可用时按树杀）。"""
    global _train_proc
    proc = _train_proc
    if proc is None:
        return False
    try:
        if force_after:
            time.sleep(force_after)
        if proc.poll() is None:
            try:
                import psutil
                parent = psutil.Process(proc.pid)
                for child in parent.children(recursive=True):
                    try:
                        child.kill()
                    except Exception:
                        pass
                parent.kill()
            except Exception:
                proc.kill()
    except Exception:
        pass
    try:
        proc.wait(timeout=10)
    except Exception:
        pass
    return True


















def _find_model_preview(name: str):
    """按名字在模型目录里"查表"找预览图 —— 用户传的 name 只用来做等值比较，
    绝不允许参与拼路径。

    返回 (Path, media_type)；找不到返回 (None, None)。
    两道保险：名字必须和某个 *_data.dat 推导出的模型名完全相等；选中的文件
    resolve() 之后必须仍在模型目录（或缓存目录）之内，且不能是符号链接。
    """
    try:
        root = MODEL_DIR.resolve()
    except Exception:
        return None, None
    cache_root = CACHE_DIR.resolve() if CACHE_DIR.exists() else None
    try:
        dats = sorted(MODEL_DIR.glob("*_data.dat"))
    except Exception:
        return None, None
    for dat in dats:
        stem = dat.name[: -len("_data.dat")]
        model_name = stem.rpartition("_")[0] or stem
        if model_name != name:
            continue
        for cand, mime in ((MODEL_DIR / (stem + "_preview.jpg"), "image/jpeg"),
                           (CACHE_DIR / (model_name + ".png"), "image/png")):
            try:
                if not cand.is_file() or cand.is_symlink():
                    continue
                rp = cand.resolve()
            except Exception:
                continue
            if rp.parent != root and (cache_root is None or rp.parent != cache_root):
                continue
            return rp, mime
    return None, None




# ---------------- 数据访问 ----------------
def _version() -> str:
    try:
        return (ROOT / "app" / "docs" / "..").resolve().joinpath("VERSION").read_text().strip()
    except Exception:
        pass
    try:
        pkg = json.loads((ROOT / "desktop" / "package.json").read_text(encoding="utf-8"))
        return str(pkg.get("version", "4.1.0"))
    except Exception:
        return "4.1.0"


def _count_images(folder: Path) -> int:
    if not folder.is_dir():
        return 0
    n = 0
    for p in folder.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMAGE_EXT:
            n += 1
    return n


def _dataset(name: str) -> dict:
    d = WORKSPACE / name
    return {
        "name": name,
        "path": str(d),
        "exists": d.is_dir(),
        "count": _count_images(d) if d.is_dir() else 0,
    }


def _models() -> list:
    md = WORKSPACE / "model"
    out = []
    if not md.is_dir():
        return out
    for dat in sorted(md.glob("*_data.dat")):
        stem = dat.name[: -len("_data.dat")]
        model_name, _, arch = stem.rpartition("_")
        if not model_name:
            model_name, arch = stem, ""
        dfm = md / (model_name + ".dfm")
        out.append({
            "name": model_name,
            "arch": arch.lower(),
            "iter": 0,
            "state": "已暂停",
            "export": "已导出 DFM" if dfm.exists() else "未导出",
            "has_export": dfm.exists(),
        })
    return out


def _ports() -> dict:
    defaults = {"mask": 8000, "merge": 8001, "angle": 8002}
    try:
        data = json.loads(SETTINGS_FILE.read_text(encoding="utf-8"))
        for k in list(defaults):
            if k in data:
                defaults[k] = int(data[k])
    except Exception:
        pass
    return defaults


@app.post("/api/settings")
async def save_settings(payload: dict):
    WORKSPACE.mkdir(parents=True, exist_ok=True)
    cur = _ports()
    for k in ("mask", "merge", "angle"):
        if k in payload:
            try:
                cur[k] = int(payload[k])
            except Exception:
                pass
    SETTINGS_FILE.write_text(json.dumps(cur, ensure_ascii=False, indent=2), encoding="utf-8")
    return cur


# ---------------- 演示用任务状态（M0 占位，M1 换成真实提取器） ----------------
class Job:
    def __init__(self):
        self.lock = threading.Lock()
        self.reset()

    def reset(self):
        with self.lock:
            self.running = False
            self.pct = 0
            self.fps = 0.0
            self.faces = 0.0
            self.done = 0
            self.total = 0
            self.msg = "就绪"
            self.stage = 0          # 0 抽帧 / 1 对齐 / 2 应用遮罩 / 3 完成
            self.t0 = 0.0

    def begin(self, stage: int, total: int, msg: str = ""):
        """进入一个新阶段：重置计时与计数，前端据此算速度和剩余时间。"""
        with self.lock:
            self.stage = stage
            self.t0 = time.time()
            self.done = 0
            self.total = int(total)
            self.pct = 0
            if msg:
                self.msg = msg

    def snapshot(self) -> dict:
        with self.lock:
            el = (time.time() - self.t0) if self.t0 else 0.0
            rate = (self.done / el) if (el > 0.3 and self.done) else 0.0
            left = max(0, self.total - self.done)
            eta = (left / rate) if (rate > 0 and left > 0) else 0.0
            return {
                "running": self.running, "pct": self.pct, "fps": self.fps,
                "faces": self.faces, "done": self.done, "total": self.total,
                "msg": self.msg, "stage": self.stage,
                "rate": round(rate, 2), "eta": round(eta, 1), "elapsed": round(el, 1),
            }


JOB = Job()


def _frames_worker(src: str, dst: str, fps: float = 2.0, rotate: int = 0,
                   align: bool = True, quality: str = "high", size: int = 512,
                   skip: int = 0):
    """用 ffmpeg 抽帧到 dst，然后（可选）对齐成人脸。

    这一版修了三处：
      1) **先清空输出目录**。ffmpeg 的 frame_%06d 从 1 开始编号，不清的话上一个
         视频的残留帧会被当成本次素材 —— 换短一点的视频就会混进旧数据。
      2) fps 由**源视频真实帧率**换算（skip 帧抽 1 帧），而不是假设源是 24fps。
         以前 UI 传的是 24/skip，30fps 的源会被多抽、60fps 的源会被砍掉一半。
      3) 进度改用 ffmpeg -progress 的 frame= 计数，不再每 0.25 s 扫一遍目录
         （几万个文件时那次 glob 本身就很贵）。
    """
    import subprocess
    import imageio_ffmpeg

    out_dir = Path(dst)                                        # 人脸输出目录（用户填的那个）
    out_dir.mkdir(parents=True, exist_ok=True)

    # 对齐模式：原帧不落盘，ffmpeg 出帧 → 内存里检测 → 直接写对齐图
    if align:
        try:
            src_fps = 30.0
            n, secs = imageio_ffmpeg.count_frames_and_secs(str(src))
            if secs > 0:
                src_fps = n / secs
            if skip and skip > 0:
                fps = max(0.05, src_fps / float(skip))
            fps = min(float(fps), src_fps)
        except Exception:
            pass
        return _align_stream(str(src), out_dir, fps, rotate, quality, skip, src_fps)

    # 不对齐：保留原来的"抽成 jpg"路径（给只想拆帧的场景用）
    exe = imageio_ffmpeg.get_ffmpeg_exe()
    frames_dir = out_dir.parent / (out_dir.name + "_frames")
    frames_dir.mkdir(parents=True, exist_ok=True)

    # ---- 1) 增量抽帧：接着已有编号往后写，绝不覆盖、也不删除 ----
    # 用户的用法是"往同一个目录里一点点加素材"，清空会误删别人的数据。
    # 但 ffmpeg 默认从 1 开始编号，直接再抽一次会**覆盖**前 N 张 —— 所以要接着最大序号写。
    existing = sorted(frames_dir.glob("frame_*.jpg"))
    start_no = 1
    if existing:
        try:
            start_no = max(int(p.stem.split("_")[-1]) for p in existing
                           if p.stem.split("_")[-1].isdigit()) + 1
        except Exception:
            start_no = len(existing) + 1

    # ---- 2) 源帧率 ----
    src_fps = 30.0
    total = 1
    try:
        n, secs = imageio_ffmpeg.count_frames_and_secs(str(src))
        if secs > 0:
            src_fps = n / secs
        if skip and skip > 0:
            fps = max(0.05, src_fps / float(skip))
        fps = min(float(fps), src_fps)
        total = max(1, int(secs * fps)) if secs else 1
    except Exception:
        pass

    with JOB.lock:
        JOB.running = True
        JOB.fps = 0.0
    # 阶段 0：ffmpeg 抽帧（进度用 -progress 的 frame= 计数驱动）
    JOB.begin(0, total,
              "正在抽帧（源 %.2f fps → 每 %.0f 帧取 1，输出 %.3f fps · 已有 %d 帧，从 %06d 接着写）"
              % (src_fps, skip, fps, len(existing), start_no))

    vf = []
    if rotate in (90, 180, 270):
        vf.append({90: "transpose=1", 180: "transpose=1,transpose=1", 270: "transpose=2"}[rotate])
    vf.append("fps=%g" % fps)
    cmd = [exe, "-v", "error", "-i", str(src),
           "-vf", ",".join(vf),
           "-q:v", "2", "-progress", "pipe:1", "-nostats",
           "-start_number", str(start_no),
           str(frames_dir / "frame_%06d.jpg")]

    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                            creationflags=_NO_WINDOW, text=True, bufsize=1)
    t0 = time.time()
    made = 0
    try:
        for line in proc.stdout:
            if not JOB.running:
                proc.terminate()
                break
            line = line.strip()
            if line.startswith("frame="):
                try:
                    made = int(line.split("=", 1)[1])
                except Exception:
                    continue
                with JOB.lock:
                    JOB.done = made
                    JOB.pct = min(95, int(made * 100 / max(1, total))) if total else 0
                    el = max(0.01, time.time() - t0)
                    JOB.fps = made / el
                    JOB.msg = "本次新增 %d / 约 %d 帧 · %.1f 帧/秒" % (made, total, JOB.fps)
    finally:
        proc.wait()

    made = len(list(frames_dir.glob("frame_*.jpg")))

    with JOB.lock:
        JOB.running = False
        JOB.done = made
        JOB.total = max(total, made)
        JOB.pct = 100
        JOB.faces = 0
        JOB.msg = "完成：%d 帧 → %s" % (made, str(frames_dir))
    return made


def _align_stream(src: str, face_dir: Path, fps: float = 2.0, rotate: int = 0,
                  quality: str = "high", skip: int = 0, src_fps: float = 0.0) -> int:
    """ffmpeg 边解码边对齐：原帧只在内存里走一趟，不落盘。

    三级流水线（每级一个线程，实测瓶颈在 H.264 解码 + 管道传输）：
        reader   解码 bgr24 rawvideo，扔进 q_frames
        detector 检测（GPU），扔进 q_detect
        writer   对齐 + JPEG 编码 + 写 DFLJPG
    稳态吞吐取三者最大值，而不是三者之和。

    尺寸：对齐边长 = 检测框长边 × 2，无上下限（越界部分填纯黑）。
    质量档只决定检测器：high=RetinaFace(det_10g)，fast=TinyMog。
    谁的档位谁负责 —— 不做跨检测器兜底（那会让 0 脸的帧跑两遍推理，
    既慢又让人误判"两个检测器一样快"）。
    """
    import queue
    import threading
    import imageio_ffmpeg
    import numpy as np
    import cv2
    from ._align import align_face, write_dfljpg, pose_from_kps

    # 谁的档位谁负责：high 只用 RetinaFace，fast 只用 TinyMog。
    # 以前会"0 脸就换另一个检测器兜底"，结果是同一批帧跑两次推理，
    # 速度被拖慢、还让人误以为两个检测器一样快。
    fast = _get_fast_detector() if quality != "high" else None
    heavy = _get_heavy_detector() if quality == "high" else None
    primary = heavy if quality == "high" else fast
    if primary is None:
        with JOB.lock:
            JOB.running = False
            JOB.msg = "检测器不可用：%s 的模型没找到" % ("RetinaFace(det_10g)" if quality == "high" else "TinyMog")
        return 0
    # TinyMog 走批量接口，需要提前把帧 letterbox 到 640；RetinaFace 自己会做
    need_canvas = bool(getattr(primary, "needs_canvas", False))

    lm106 = _get_landmarks106()
    face_dir.mkdir(parents=True, exist_ok=True)

    base_faces = len(list(face_dir.glob("*_*.jpg")))      # 目录里已有的对齐脸（续跑时不为 0）
    done_idx = set()
    for f in face_dir.glob("*_*.jpg"):
        head = f.stem.split("_")[0]
        if head.isdigit():
            done_idx.add(int(head))

    vf = []
    if rotate in (90, 180, 270):
        vf.append({90: "transpose=1", 180: "transpose=1,transpose=1", 270: "transpose=2"}[rotate])
    # 必须始终带一个滤镜：imageio_ffmpeg 在 output_params 为空或只有 null 时会拼出
    # 坏命令并杀掉 ffmpeg（实测稳定复现 0 帧）。即便 fps == 源帧率，也显式写出来。
    vf.append("fps=%g" % max(0.01, float(fps)))

    # vf 为空时不能传 -vf ""（ffmpeg 会报错），所以整段参数一起省掉
    # 不能传空列表：imageio_ffmpeg 在 output_params=[] 时会拼出坏命令并杀掉 ffmpeg（实测 0 帧）。
    # 没有滤镜就显式给个 null，等价于不处理。
    out_params = ["-vf", ",".join(vf)]
    gen = imageio_ffmpeg.read_frames(
        str(src),
        pix_fmt="bgr24",                       # 直接用 BGR：省掉一次通道反转拷贝（~3ms/帧）
        input_params=[],
        output_params=out_params,
    )
    meta = next(gen)
    W, H = meta.get("size") or (0, 0)
    if not W or not H:
        try: gen.close()
        except Exception: pass
        return 0
    dur = float(meta.get("duration") or 0)
    total_est = max(1, int(dur * max(0.01, float(fps)))) if dur else 1

    S = 640
    JOB.begin(0, total_est, "边解码边对齐（%dx%d · %.2f fps · 不落盘）" % (W, H, fps))
    # JOB.begin 不会把 running 置 True，而 reader 线程用 JOB.running 判断是否继续。
    # 少了这一行，reader 第一次 next 前就 break → 永远 0 帧（实测踩过）。
    with JOB.lock:
        JOB.running = True

    PROF = bool(os.environ.get("DFL_EXTRACT_PROFILE"))
    # 每级"真正干活"与"等队列/等管道"的累计秒数，用来判断是不是伪并行
    cost = {"read": 0.0, "put1": 0.0, "get1": 0.0, "det": 0.0, "put2": 0.0,
            "get2": 0.0, "emit": 0.0}
    qmax = {"q1": 0, "q2": 0}

    def _t():
        return time.perf_counter()

    q_frames: "queue.Queue" = queue.Queue(maxsize=6)
    q_detect: "queue.Queue" = queue.Queue(maxsize=6)
    stop = threading.Event()
    stats = {"seen": 0, "made": 0, "fallback": 0}
    t0 = time.time()
    lock = threading.Lock()

    def _canvas(img):
        h, w = img.shape[:2]
        sc = min(S / float(w), S / float(h))
        nw, nh = max(1, int(round(w * sc))), max(1, int(round(h * sc)))
        cv = np.full((S, S, 3), 128, np.uint8)
        cv[:nh, :nw] = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
        return cv, sc

    def _emit(idx, img, faces):
        n = 0
        for k, fc in enumerate(faces):
            if not JOB.running:
                break
            box = fc.get("box") or [0, 0, 0, 0]
            if min(box[2] - box[0], box[3] - box[1]) < 32:
                continue
            bw = max(1.0, float(box[2]) - float(box[0]))
            bh = max(1.0, float(box[3]) - float(box[1]))
            side = int(round(max(bw, bh) * 2.0))   # 2 倍框宽，不设上下限
            side -= side % 2
            crop, meta_f = align_face(img, fc["kps"], box=box, size=side)
            if crop is None:
                continue
            meta_f["source_filename"] = "%06d" % idx
            try:
                _y, _p, _r = pose_from_kps(fc["kps"], box)
                if _y is not None:
                    meta_f["yaw"], meta_f["pitch"], meta_f["roll"] = _y, _p, _r
            except Exception:
                pass
            if lm106 is not None:
                try:
                    lm68 = lm106.landmarks68(crop)
                    if lm68 is not None:
                        from ._align import transform_points as _tp
                        meta_f["landmarks"] = np.round(lm68, 4).tolist()
                        meta_f["source_landmarks"] = np.round(
                            _tp(lm68, np.array(meta_f["image_to_face_mat"]), invert=True), 4).tolist()
                except Exception:
                    pass
            try:
                write_dfljpg(face_dir / ("%06d_%d.jpg" % (idx, k)), crop, meta_f)
                n += 1
            except Exception:
                continue
        return n

    def _reader():
        idx = 0
        it = iter(gen)
        try:
            while True:
                t_a = _t()
                try:
                    raw = next(it)          # ← 真正的管道读取（ffmpeg 解码 + 传输）
                except StopIteration:
                    break
                cost["read"] += _t() - t_a
                if stop.is_set() or not JOB.running:
                    break
                if len(raw) != W * H * 3:
                    continue
                t_b = _t()
                frame = np.frombuffer(raw, np.uint8).reshape(H, W, 3)
                q_frames.put((idx, frame))
                cost["put1"] += _t() - t_b
                if PROF:
                    qmax["q1"] = max(qmax["q1"], q_frames.qsize())
                idx += 1
                with lock:
                    stats["seen"] = idx
                if idx % 10 == 0:
                    with JOB.lock:
                        JOB.done = idx
                        JOB.pct = min(99, int(idx * 100 / max(1, total_est)))
                        el = max(0.01, time.time() - t0)
                        JOB.fps = idx / el
                        JOB.faces = base_faces + stats["made"]
                        JOB.msg = "解码 %d 帧 · 已出 %d 张 · %.1f 帧/秒" % (
                            idx, stats["made"], JOB.fps)
        finally:
            q_frames.put(None)

    # TinyMog 的 batch 维是动态的：攒一批一起推理，能把每帧的固定开销摊薄。
    # RetinaFace 的引擎是单图 profile，老老实实一张一张来。
    # TinyMog 引擎的优化 profile 是 batch 1..4，超过 4 会被 TRT 拒绝（实测 batch=8 直接报错）
    BATCH = min(4, int(os.environ.get("DFL_EXTRACT_BATCH") or (4 if need_canvas else 1)))

    def _detect_batch(batch):
        """batch: list[(idx, img)] → list[list[face]]"""
        if not batch:
            return []
        if need_canvas:
            items = [(b[1], *_canvas(b[1])) for b in batch]
            try:
                res = primary.detect_batch(items)
            except Exception:
                res = []
            if len(res) != len(batch):
                res = [[] for _ in batch]
        else:
            res = []
            for idx, img in batch:
                try:
                    res.append(primary.detect(img))
                except Exception:
                    res.append([])
        return res

    def _detector():
        pending = []

        def flush():
            if not pending:
                return
            batch = list(pending)
            pending.clear()
            t_d = _t()
            faces_list = _detect_batch(batch)
            cost["det"] += _t() - t_d
            t_p = _t()
            for item, faces in zip(batch, faces_list):
                # item 可能是 (idx, img) 或 (idx, img, canvas, scale)
                q_detect.put((item[0], item[1], faces))
            cost["put2"] += _t() - t_p
            if PROF:
                qmax["q2"] = max(qmax["q2"], q_detect.qsize())

        while True:
            t_a = _t()
            item = q_frames.get()
            cost["get1"] += _t() - t_a
            if item is None:
                flush()
                q_detect.put(None)
                return
            if item[0] in done_idx:
                continue
            pending.append(item)
            if len(pending) < BATCH:
                continue
            # 批次没满时看看队列里还有没有，有就一起吃掉（减少一次同步）
            while len(pending) < BATCH:
                try:
                    nxt = q_frames.get_nowait()
                except Exception:
                    break
                if nxt is None:
                    q_frames.put(None)
                    break
                if nxt[0] not in done_idx:
                    pending.append(nxt)
            flush()

    def _writer():
        while True:
            t_a = _t()
            item = q_detect.get()
            cost["get2"] += _t() - t_a
            if item is None:
                return
            idx, img, faces = item
            t_e = _t()
            n = _emit(idx, img, faces)
            cost["emit"] += _t() - t_e
            if n:
                with lock:
                    stats["made"] += n

    threads = [threading.Thread(target=_reader, name="ext-read", daemon=True),
               threading.Thread(target=_detector, name="ext-detect", daemon=True),
               threading.Thread(target=_writer, name="ext-write", daemon=True)]
    for th in threads:
        th.start()
    try:
        for th in threads:
            th.join()
    finally:
        stop.set()
        try: gen.close()
        except Exception: pass

    if PROF:
        wall = max(1e-6, time.time() - t0)
        try:
            print("[extract-profile] wall=%.2fs  frames=%d  faces=%d" % (wall, stats["seen"], stats["made"]))
            for k in ("read", "put1", "get1", "det", "put2", "get2", "emit"):
                print("  %-5s %7.3fs  %5.1f%% of wall" % (k, cost[k], cost[k] * 100.0 / wall))
            print("  queue max depth: q1=%d q2=%d" % (qmax["q1"], qmax["q2"]))
        except Exception:
            pass

    with JOB.lock:
        JOB.running = False
        JOB.done = stats["seen"]
        JOB.total = max(stats["seen"], 1)
        JOB.pct = 100
        JOB.faces = base_faces + stats["made"]
        JOB.msg = "完成：解码 %d 帧 · %d 张人脸 → %s" % (stats["seen"], stats["made"], str(face_dir))
    return stats["made"]


def _five_from_box(box):
    """没有关键点时的兜底 5 点：按框比例摆出双眼/鼻尖/嘴角。

    RetinaFace 正常情况下会给出 5 点，这条只在模型没有 kps 输出时兜底，
    对齐质量差一些，但产出的仍然是合法 DFLJPG。
    """
    x1, y1, x2, y2 = [float(v) for v in box]
    w, h = max(1.0, x2 - x1), max(1.0, y2 - y1)
    return [
        [x1 + 0.32 * w, y1 + 0.38 * h],   # 左眼
        [x1 + 0.68 * w, y1 + 0.38 * h],   # 右眼
        [x1 + 0.50 * w, y1 + 0.62 * h],   # 鼻尖
        [x1 + 0.35 * w, y1 + 0.80 * h],   # 左嘴角
        [x1 + 0.65 * w, y1 + 0.80 * h],   # 右嘴角
    ]


def _get_landmarks106():
    """加载 insightface 2D106（模型在引擎仓的 modelhub 里）。找不到就返回 None。

    **默认关闭**：特征点只干两件事 —— 对齐（已由 5 点完成）、嘴/眼遮罩；
    而 5 点对齐已经把眼睛和嘴巴摆到规范位置，模板 68 点就够用。
    需要真 68 点（例如让 4.0 的 sort_by_yaw 也工作在 4.1 数据上）：DFL_LANDMARKS106=1
    """
    import os as _os
    if _os.environ.get("DFL_LANDMARKS106") != "1":
        return None
    if _lm106_cache["tried"]:
        return _lm106_cache["m"]
    _lm106_cache["tried"] = True
    try:
        from ._landmarks106 import InsightFace106
        from ._ortcuda import _candidate_roots
        for root in _candidate_roots():
            for rel in ("modelhub/onnx/InsightFace2d106/InsightFace2D106.onnx",
                        "DeepFaceLab-PyTorch/modelhub/onnx/InsightFace2d106/InsightFace2D106.onnx"):
                cand = Path(root) / rel
                if cand.is_file():
                    _lm106_cache["m"] = InsightFace106(str(cand), workspace=WORKSPACE,
                                                       engine_name="insightface_2d106")
                    return _lm106_cache["m"]
    except Exception:
        pass
    _lm106_cache["m"] = None
    return None


_xseg_cache = {"m": None, "tried": False}


def _get_xseg(model: str | None = None):
    """加载 XSeg 遮罩模型。model: 'lite' / 'xseg' / None(自动，Lite 优先)。

    按 model 分别缓存——切换遮罩类型不会重复加载同一个模型。
    """
    key = (model or "auto").lower()
    cache = _xseg_cache.setdefault("by_model", {})
    if key in cache:
        return cache[key]
    try:
        from ._xseg import XSegMasker
        # 模型可能在两个地方：本仓库 workspace，或者 4.0 引擎仓库的 workspace
        # （4.1 只是壳，模型历来放在引擎那边）
        roots = [ROOT / "weights" / "xseg", WORKSPACE]
        try:
            from ._paths import engine_root
            _er = engine_root()
            if _er and Path(_er) / "workspace" not in roots:
                roots.append(Path(_er) / "workspace")
        except Exception:
            pass
        paths = {}
        for r in roots:
            paths.setdefault("lite", r / "model" / "XSegLite" / "xseglite.onnx")
            paths.setdefault("xseg", r / "model" / "XSeg" / "XSeg.onnx")
        # 按 roots 顺序挑第一个真实存在的
        for k in ("lite", "xseg"):
            for r in roots:
                cand = r / "model" / ("XSegLite/xseglite.onnx" if k == "lite" else "XSeg/XSeg.onnx")
                if cand.is_file():
                    paths[k] = cand
                    break
        order = (["lite", "xseg"] if key == "auto" else [key]) +                 [k for k in ("lite", "xseg") if k != key and key != "auto"]
        for k in order:
            c = paths.get(k)
            if c and c.is_file():
                cache[key] = XSegMasker(str(c), workspace=WORKSPACE,
                                        engine_name=("xseglite" if k == "lite" else "xseg"))
                return cache[key]
    except Exception:
        pass
    _xseg_cache["m"] = None
    return None


_detector_cache = {"m": None}
_fast_detector_cache = {"m": None, "tried": False}
_heavy_detector_cache = {"m": None, "tried": False}


def _detector_path(name):
    hub = WORKSPACE / "modelhub" / "onnx"
    cands = [hub, Path(__file__).resolve().parents[2] / "modelhub" / "onnx",
             Path(__file__).resolve().parents[3] / "DeepFaceLab-PyTorch" / "modelhub" / "onnx"]
    for base in cands:
        p = base / name
        if p.is_file():
            return p
    return None


def _get_fast_detector():
    """快检测器：TinyMog（2.7 MB，SCRFD+MobileNetV1，batch 维是动态的）。"""
    if _fast_detector_cache["tried"]:
        return _fast_detector_cache["m"]
    _fast_detector_cache["tried"] = True
    try:
        from ._detect import TinyMogDet
        p = _detector_path("TinyMog/TinyMog.onnx")
        if p is not None:
            _fast_detector_cache["m"] = TinyMogDet(str(p), workspace=WORKSPACE, engine_name="tinymog")
    except Exception:
        pass
    return _fast_detector_cache["m"]


def _get_heavy_detector():
    """重检测器：det_10g（召回更好），没有就退到 det_500m。"""
    if _heavy_detector_cache["tried"]:
        return _heavy_detector_cache["m"]
    _heavy_detector_cache["tried"] = True
    try:
        from ._detect import RetinaFaceDet
        for nm in ("RetinaFace/det_10g.onnx", "RetinaFace/det_500m.onnx"):
            p = _detector_path(nm)
            if p is not None:
                _heavy_detector_cache["m"] = RetinaFaceDet(
                    str(p), workspace=WORKSPACE,
                    engine_name=("retinaface_10g" if "10g" in nm else "retinaface_500m"))
                break
    except Exception:
        pass
    return _heavy_detector_cache["m"]


def _demo_worker():
    total = 342
    with JOB.lock:
        JOB.running = True
        JOB.total = total
    for i in range(total + 1):
        if not JOB.running:
            break
        with JOB.lock:
            JOB.done = i
            JOB.pct = int(i * 100 / total)
            JOB.fps = 34 + (i % 7) * 0.9
            JOB.faces = 9 + (i % 5) * 0.2
            JOB.msg = "处理中 %d / %d" % (i, total)
        time.sleep(0.02)
    with JOB.lock:
        JOB.running = False
        JOB.pct = 100
        JOB.msg = "完成"


def _port_free(port: int) -> bool:
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            s.bind(("127.0.0.1", int(port)))
            return True
        except OSError:
            return False


def _runtime() -> dict:
    import platform
    return {
        "python": platform.python_version(),
        "platform": platform.system() + " " + platform.release(),
        "port": int(os.environ.get("DFL_PORT", 0)),
        "workspace": str(WORKSPACE),
    }


def _ports_status() -> dict:
    p = _ports()
    cur = int(os.environ.get("DFL_PORT", 0))
    return {
        "mask": True if p["mask"] == cur else _port_free(p["mask"]),
        "merge": True if p["merge"] == cur else _port_free(p["merge"]),
        "angle": True if p["angle"] == cur else _port_free(p["angle"]),
    }


# ============================================================
# 远程访问：把整个 WebGUI 在 0.0.0.0:PORT 上再开一份
# 给「部署在服务器上、用别的设备连」的场景
# ============================================================
_remote = {"server": None, "thread": None, "port": 0}


def _lan_ip():
    """猜一下本机在局域网里的地址，用来在界面上显示可访问的 URL。"""
    import socket as _s
    try:
        sk = _s.socket(_s.AF_INET, _s.SOCK_DGRAM)
        sk.connect(("8.8.8.8", 80))
        ip = sk.getsockname()[0]
        sk.close()
        return ip
    except Exception:
        return "127.0.0.1"


def _remote_stop():
    global _remote
    srv = _remote.get("server")
    if srv is not None:
        srv.should_exit = True
    _remote.update({"server": None, "thread": None, "port": 0})


def _remote_start(port: int):
    import threading
    import uvicorn
    _remote_stop()
    cfg = uvicorn.Config(app, host="0.0.0.0", port=port, log_level="warning")
    srv = uvicorn.Server(cfg)
    t = threading.Thread(target=srv.run, daemon=True)
    t.start()
    _remote.update({"server": srv, "thread": t, "port": port})


def _remote_boot():
    """上次开着远程访问的话，这次启动自动恢复。失败就静默跳过（端口被占之类）。"""
    try:
        if not SETTINGS_FILE.is_file():
            return
        cfg = json.loads(SETTINGS_FILE.read_text(encoding="utf-8"))
        if cfg.get("remote_enabled"):
            _remote_start(int(cfg.get("remote_port") or 8002))
    except Exception:
        pass


@app.get("/api/remote")
def remote_state():
    return {
        "enabled": _remote.get("server") is not None,
        "port": _remote.get("port") or 0,
        "ip": _lan_ip(),
        "url": ("http://%s:%d" % (_lan_ip(), _remote["port"])) if _remote.get("port") else "",
    }


@app.post("/api/remote")
def remote_set(body: dict):
    body = body or {}
    enabled = bool(body.get("enabled"))
    try:
        port = int(body.get("port") or 8002)
    except Exception:
        raise HTTPException(400, "端口必须是数字")
    if not (1024 <= port <= 65535):
        raise HTTPException(400, "端口要在 1024–65535 之间")

    try:
        if enabled:
            _remote_start(port)
        else:
            _remote_stop()
    except OSError as e:
        raise HTTPException(500, "端口 %d 起不来: %s" % (port, e))
    except Exception as e:
        raise HTTPException(500, str(e))

    # 记到设置里，下次启动可以按需恢复
    try:
        cfg = {}
        if SETTINGS_FILE.is_file():
            cfg = json.loads(SETTINGS_FILE.read_text(encoding="utf-8"))
        cfg["remote_enabled"] = enabled
        cfg["remote_port"] = port
        SETTINGS_FILE.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        pass

    return remote_state()


@app.get("/api/state")
def state():
    return {
        "version": _version(),
        "runtime": _runtime(),
        "ports_status": _ports_status(),
        "datasets": {"src": _dataset("data_src"), "dst": _dataset("data_dst")},
        "models": _models(),
        "ports": _ports(),
    }


def _mask_worker(folder: str, workers: int = 8, model: str | None = None, out_size: int = 256):
    """给一个 aligned 目录里的所有 DFLJPG 补 XSeg 遮罩（提取之后的单独一步）。

    提取本身不跑 XSeg：DFL 的流程也是 extract → mask 两步，
    而且遮罩模型是 3.5 ms/张，放在提取热路径上纯属浪费。
    """
    import cv2
    from concurrent.futures import ThreadPoolExecutor
    from ._align import read_dfljpg_meta, rewrite_dfljpg_meta, encode_xseg_mask

    xseg = _get_xseg(model)
    root = Path(folder)
    files = sorted(root.glob("*.jpg")) + sorted(root.glob("*.png"))
    total = len(files)
    with JOB.lock:
        JOB.running = True
        JOB.msg = "应用遮罩：共 %d 张" % total
    if total == 0:
        with JOB.lock:
            JOB.running = False
            JOB.pct = 100
            JOB.msg = "目录里没有 DFLJPG：%s" % str(root)
        return 0
    JOB.begin(2, total)
    if xseg is None:
        with JOB.lock:
            JOB.running = False
            JOB.msg = "找不到 XSeg 模型（workspace/model/XSegLite）"
        return 0

    done = {"n": 0, "ok": 0}

    def _one(fp):
        try:
            if not JOB.running:
                return
            meta = read_dfljpg_meta(fp)
            if meta is None:
                return
            if meta.get("xseg_mask") is not None:
                pass
            else:
                img = _imread_any(fp)
                if img is None:
                    return
                enc = encode_xseg_mask(xseg.mask(img, out_size=int(out_size)))
                if enc:
                    meta["xseg_mask"] = enc
                    rewrite_dfljpg_meta(fp, meta)
                    with JOB.lock:
                        done["ok"] += 1
        except Exception:
            pass
        finally:
            with JOB.lock:
                done["n"] += 1
                JOB.done = done["n"]
                JOB.pct = int(done["n"] * 100 / max(1, total))
                if done["n"] % 20 == 0 or done["n"] == total:
                    JOB.msg = "应用遮罩 %d / %d · 新写入 %d" % (done["n"], total, done["ok"])

    with ThreadPoolExecutor(max_workers=max(1, int(workers))) as pool:
        list(pool.map(_one, files))

    with JOB.lock:
        JOB.running = False
        JOB.pct = 100
        JOB.msg = "应用遮罩完成：%d 张，新增 %d" % (total, done["ok"])
    return done["ok"]



# ---------------- 拆出去的路由模块 ----------------
from .routes_bili import router as _bili_router
app.include_router(_bili_router)
from .routes_face import router as _face_router
app.include_router(_face_router)
from .routes_trt import router as _trt_router
app.include_router(_trt_router)

# ---------------- 拆出去的路由模块 ----------------
from .routes_account import router as _account_router
app.include_router(_account_router)
from .routes_bili import router as _bili_router
app.include_router(_bili_router)
from .routes_face import router as _face_router
app.include_router(_face_router)
from .routes_trt import router as _trt_router
app.include_router(_trt_router)

# ---------------- 拆出去的路由模块 ----------------
from .routes_account import router as _account_router
app.include_router(_account_router)
from .routes_bili import router as _bili_router
app.include_router(_bili_router)
from .routes_face import router as _face_router
app.include_router(_face_router)
from .routes_oneshot import router as _oneshot_router
app.include_router(_oneshot_router)
from .routes_trt import router as _trt_router
app.include_router(_trt_router)

# ---------------- 拆出去的路由模块 ----------------
from .routes_account import router as _account_router
app.include_router(_account_router)
from .routes_bili import router as _bili_router
app.include_router(_bili_router)
from .routes_face import router as _face_router
app.include_router(_face_router)
from .routes_merge import router as _merge_router
app.include_router(_merge_router)
from .routes_oneshot import router as _oneshot_router
app.include_router(_oneshot_router)
from .routes_trt import router as _trt_router
app.include_router(_trt_router)

# ---------------- 拆出去的路由模块 ----------------
from .routes_account import router as _account_router
app.include_router(_account_router)
from .routes_bili import router as _bili_router
app.include_router(_bili_router)
from .routes_face import router as _face_router
app.include_router(_face_router)
from .routes_mask import router as _mask_router
app.include_router(_mask_router)
from .routes_merge import router as _merge_router
app.include_router(_merge_router)
from .routes_oneshot import router as _oneshot_router
app.include_router(_oneshot_router)
from .routes_trt import router as _trt_router
app.include_router(_trt_router)

# ---------------- 拆出去的路由模块 ----------------
from .routes_account import router as _account_router
app.include_router(_account_router)
from .routes_bili import router as _bili_router
app.include_router(_bili_router)
from .routes_face import router as _face_router
app.include_router(_face_router)
from .routes_mask import router as _mask_router
app.include_router(_mask_router)
from .routes_merge import router as _merge_router
app.include_router(_merge_router)
from .routes_models import router as _models_router
app.include_router(_models_router)
from .routes_oneshot import router as _oneshot_router
app.include_router(_oneshot_router)
from .routes_trt import router as _trt_router
app.include_router(_trt_router)

# ---------------- 拆出去的路由模块 ----------------
from .routes_account import router as _account_router
app.include_router(_account_router)
from .routes_bili import router as _bili_router
app.include_router(_bili_router)
from .routes_face import router as _face_router
app.include_router(_face_router)
from .routes_mask import router as _mask_router
app.include_router(_mask_router)
from .routes_merge import router as _merge_router
app.include_router(_merge_router)
from .routes_models import router as _models_router
app.include_router(_models_router)
from .routes_oneshot import router as _oneshot_router
app.include_router(_oneshot_router)
from .routes_train import router as _train_router
app.include_router(_train_router)
from .routes_trt import router as _trt_router
app.include_router(_trt_router)
