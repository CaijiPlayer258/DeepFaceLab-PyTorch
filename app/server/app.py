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

# 也可以在项目根放一个 workspace.txt，里面写一行工作区路径（迁移旧工程时用）
_ws_txt = ROOT / "workspace.txt"
if not os.environ.get("DFL_WORKSPACE") and _ws_txt.exists():
    try:
        _p = Path(_ws_txt.read_text(encoding="utf-8").strip())
        if _p.is_dir():
            WORKSPACE = _p
    except Exception:
        pass
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


@app.get("/api/merge/videos")
def merge_videos(dir: str = ""):
    """列出视频文件。dir 给了就扫它，否则扫工作区。"""
    exts = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv"}
    roots = [Path(dir)] if dir else []
    roots.append(WORKSPACE)
    out = []
    for root in roots:
        if not root.is_dir():
            continue
        for f in sorted(root.rglob("*")):
            if not f.is_file() or f.suffix.lower() not in exts:
                continue
            # 合成产物 / 缓存目录不作为素材列出来（否则列表里全是自己的输出）
            if f.name.endswith("_merged.mp4") or ".merge_" in str(f.parent):
                continue
            out.append({"path": str(f), "name": f.name,
                        "mb": round(f.stat().st_size / 1048576, 1)})
            if len(out) >= 200:
                break
    return {"dir": str(roots[0]), "videos": out}


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


@app.post("/api/merge/prepare")
def merge_prepare(body: dict):
    """视频 → 可合成项目（抽帧 + 对齐）。跑在后台，进度用 /status 拉。"""
    body = body or {}
    raw = str(body.get("video") or "").strip()
    if not raw:
        raise HTTPException(400, "缺少 video")
    video = Path(raw)
    if not video.is_file():
        raise HTTPException(404, "视频不存在: " + raw)

    with _merge_prep_lock:
        if _merge_prep["running"]:
            raise HTTPException(409, "正在准备另一个视频")
        _merge_prep.update({"running": True, "ok": None, "error": "", "pct": 0,
                            "msg": "启动…", "stage": "准备", "frames": 0,
                            "aligned": 0, "video": str(video)})

    target = str(body.get("dir") or "").strip()
    project = Path(target) if target else (WORKSPACE / "merge" / _safe_dirname(video.stem))
    with _merge_prep_lock:
        _merge_prep["project"] = str(project)

    threading.Thread(target=_merge_prepare_worker, args=(video, project),
                     daemon=True).start()
    return {"ok": True, "project": str(project), "video": str(video)}


@app.get("/api/merge/prepare/status")
def merge_prepare_status():
    with _merge_prep_lock:
        return dict(_merge_prep)


@app.post("/api/merge/open")
def merge_open(body: dict):
    raw = (body or {}).get("path") or ""
    p = Path(raw)

    # 目录 = DFL 项目（data_dst / aligned / 裸帧目录）
    if p.is_dir():
        names, srcdir, kind = _dfl_project_scan(p)
        if not names:
            raise HTTPException(404, "目录里没有可用的图: " + raw)
        _merge_cache.update({
            "mode": "project", "dir": str(p), "srcdir": srcdir, "names": names,
            "path": "", "frames": len(names), "fps": 0, "w": 0, "h": 0,
        })
        sample = names[:24]
        lm = sum(1 for n in sample
                 if _dfljpg_dict(Path(srcdir) / n).get("landmarks") is not None)
        return {"path": str(p), "name": p.name, "mode": "project", "kind": kind,
                "frames": len(names), "fps": 0, "width": 0, "height": 0,
                "with_landmarks": lm, "sampled": len(sample)}

    if not p.is_file():
        raise HTTPException(404, "路径不存在: " + raw)
    try:
        frames, fps, w, h = _video_probe(p)
    except Exception as e:
        raise HTTPException(500, "无法解析视频: " + str(e))
    _merge_cache.update({"mode": "video", "path": str(p), "dir": "", "list": [],
                         "frames": frames, "fps": fps, "w": w, "h": h})
    return {"path": str(p), "name": p.name, "mode": "video", "frames": frames,
            "fps": fps, "width": w, "height": h}


@app.get("/api/merge/frame/{idx}")
def merge_frame(idx: int, w: int = 900):
    # 项目模式：直接从列表里取那张图
    if _merge_cache.get("mode") == "project":
        names = _merge_cache.get("names") or []
        srcdir = _merge_cache.get("srcdir") or ""
        if not names:
            raise HTTPException(409, "还没有打开项目")
        idx = max(0, min(idx, len(names) - 1))
        fp = Path(srcdir) / names[idx]
        if not fp.is_file():
            raise HTTPException(404, "图片不见了: " + str(fp))
        try:
            import cv2
            img = _imread_any(fp)
            if img is None:
                raise RuntimeError("解不开")
            h0, w0 = img.shape[:2]
            maxw = max(200, min(w, 1600))
            if w0 > maxw:
                img = cv2.resize(img, (maxw, int(h0 * maxw / w0)), interpolation=cv2.INTER_AREA)
            ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
            if not ok:
                raise RuntimeError("编码失败")
        except Exception as e:
            raise HTTPException(500, "取图失败: " + str(e))
        return {"idx": idx, "total": len(names), "name": names[idx],
                "png": "data:image/jpeg;base64," + _b64.b64encode(buf.tobytes()).decode()}

    path = _merge_cache.get("path")
    if not path:
        raise HTTPException(409, "还没有打开视频")
    total = _merge_cache.get("frames") or 0
    if total:
        idx = max(0, min(idx, total - 1))
    try:
        png = _video_frame_jpeg(path, idx, maxw=max(200, min(w, 1600)))
    except Exception as e:
        raise HTTPException(500, "取帧失败: " + str(e))
    return {"idx": idx, "png": "data:image/jpeg;base64," + _b64.b64encode(png).decode()}


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


@app.get("/api/models/{name}/config")
def model_config_get(name: str):
    """读一个模型当前保存的训练参数。"""
    import pickle
    dat = _model_dat(name)
    if dat is None:
        raise HTTPException(404, "找不到模型 " + name)
    try:
        data = pickle.loads(dat.read_bytes())
    except Exception as e:
        raise HTTPException(500, "读取 _data.dat 失败: " + str(e))
    opts = data.get("options") or {}
    return {"name": name, "iter": data.get("iter", 0),
            "options": {k: v for k, v in opts.items() if isinstance(v, (int, float, str, bool)) or v is None},
            "readonly": sorted([k for k in opts if k in _CFG_STRUCTURAL])}


@app.post("/api/models/{name}/config")
def model_config_set(name: str, body: dict):
    """写回训练参数。结构性字段拒绝，其余原样合并。"""
    import pickle
    dat = _model_dat(name)
    if dat is None:
        raise HTTPException(404, "找不到模型 " + name)
    patch = (body or {}).get("options") or {}
    if not patch:
        raise HTTPException(400, "没有要写入的参数")

    blocked = [k for k in patch if k in _CFG_STRUCTURAL]
    if blocked:
        raise HTTPException(400, "这些是创建后不可改的结构参数: " + ", ".join(blocked))

    try:
        data = pickle.loads(dat.read_bytes())
    except Exception as e:
        raise HTTPException(500, "读取 _data.dat 失败: " + str(e))

    opts = data.get("options") or {}

    # 按原字段的类型收敛：界面送上来的都是字符串，"" 表示「这项没填，别动」
    bad = []
    clean = {}
    for k, v in list(patch.items()):
        cur = opts.get(k)
        if isinstance(v, str):
            v = v.strip()
            if v == "":
                continue                      # 空着 = 不改这一项
        try:
            if isinstance(cur, bool):
                if isinstance(v, str):
                    v = v.lower() in ("1", "true", "yes", "on", "开")
                else:
                    v = bool(v)
            elif isinstance(cur, int) and not isinstance(cur, bool):
                v = int(round(float(v)))
            elif isinstance(cur, float):
                v = float(v)
            elif isinstance(cur, str):
                v = str(v)
        except (TypeError, ValueError):
            bad.append(k)
            continue
        clean[k] = v
    if bad:
        raise HTTPException(400, "这些参数不是有效数字: " + ", ".join(bad))
    patch = clean
    if not patch:
        return {"ok": True, "name": name, "changed": {}, "count": 0}

    backup = dat.with_suffix(".dat.bak")
    try:
        backup.write_bytes(dat.read_bytes())
    except Exception:
        pass

    changed = {}
    for k, v in patch.items():
        if opts.get(k) != v:
            changed[k] = {"from": opts.get(k), "to": v}
        opts[k] = v
    data["options"] = opts
    # 非原子写会把 _data.dat 写坏：先落临时文件再换名，中途崩也只是丢弃临时文件
    try:
        blob = pickle.dumps(data)
        tmp = dat.with_suffix(".dat.tmp")
        tmp.write_bytes(blob)
        os.replace(str(tmp), str(dat))
    except Exception as e:
        raise HTTPException(500, "写入失败: " + str(e))
    return {"ok": True, "name": name, "changed": changed, "count": len(changed)}


@app.post("/api/models/{name}/delete")
def model_delete(name: str):
    """删掉一个模型的所有文件（权重 / data.dat / 备份目录）。训练中的模型拒绝删。"""
    import shutil as _sh
    name = _safe_model_name(name)
    with _train_job_lock:
        if _train_job["running"] and _train_job.get("name") == name:
            raise HTTPException(409, "这个模型正在训练，先停止再删")
    md = WORKSPACE / "model"
    paths = sorted(md.glob(name + "_*"))
    if not paths:
        raise HTTPException(404, "找不到模型 " + name)
    removed = []
    for p in paths:
        try:
            if p.is_dir():
                _sh.rmtree(p, ignore_errors=True)
            else:
                p.unlink()
            removed.append(p.name)
        except Exception as e:
            raise HTTPException(500, "删除失败 %s: %s" % (p.name, str(e)[:120]))
    return {"ok": True, "name": name, "removed": removed, "count": len(removed)}


@app.post("/api/models/{name}/export")
def model_export(name: str):
    """把一个模型导出成 .dfm（后台跑，前端轮询 /api/models/export/status）。"""
    md = WORKSPACE / "model"
    if not (md / (name + "_" + _arch_of(name) + "_data.dat")).exists():
        # 兜底：直接看有没有 *_{name}_*_data.dat
        cands = list(md.glob(name + "_*_data.dat"))
        if not cands:
            raise HTTPException(404, "找不到模型 " + name)
    if not _DFL_PY.exists() or not _DFL_TOOL.exists():
        raise HTTPException(500, "找不到 4.0 的 python 或导出脚本")
    with _export_job_lock:
        if _export_job["running"]:
            raise HTTPException(409, "已经在导出 " + _export_job["name"])
    arch = _arch_of(name)
    _threading.Thread(target=_export_worker, args=(name, arch), daemon=True).start()
    return {"ok": True, "name": name, "arch": arch, "running": True}


@app.get("/api/models/export/status")
def model_export_status():
    with _export_job_lock:
        return {"running": _export_job["running"], "name": _export_job["name"],
                "ok": _export_job["ok"], "error": _export_job["error"],
                "log": _export_job["log"][-40:]}


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
    py = _DFL_ROOT / "python" / "python.exe"
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


@app.post("/api/merge/export")
def merge_export(body: dict):
    """把一段帧目录合成成换脸结果。真要跑，不是占位。"""
    body = body or {}
    # project 和 input_dir 是同一样东西：合成器里叫「项目」，接口里叫「输入帧目录」
    input_dir = str(body.get("input_dir") or body.get("project") or "").strip()
    if not input_dir:
        raise HTTPException(400, "缺少 input_dir（或 project）")
    ip = Path(input_dir)
    if not ip.is_dir():
        raise HTTPException(404, "输入目录不存在: " + input_dir)

    dfm = str(body.get("dfm") or "").strip()
    model_name = str(body.get("model") or "").strip()
    if not dfm and model_name:
        # 界面传的是模型名（384wf / DFLitetest …），这里落成 .dfm 路径
        cand = WORKSPACE / "model" / (model_name + ".dfm")
        if cand.is_file():
            dfm = str(cand)
    if not dfm:
        # 自动挑第一个 .dfm
        cands = sorted((WORKSPACE / "model").glob("*.dfm"))
        dfm = str(cands[0]) if cands else ""
    if not dfm or not Path(dfm).is_file():
        who = ("模型 %s 还没有导出 .dfm —— 去训练页点「导出」" % model_name) if model_name else "没有可用的 .dfm 模型"
        raise HTTPException(400, who)

    out_raw = str(body.get("output_dir") or "")
    aligned_dir = str(body.get("aligned_dir") or "")

    # 对齐集：显式传入 > 输入目录下的 aligned/ > 同级的 <name>_aligned/
    # 合成靠 DFLJPG 里的元数据定位人脸，没有对齐集就只能瞎猜，所以这里必须定死
    if not aligned_dir:
        cands = [ip / "aligned", ip.parent / (ip.name + "_aligned")]
        for c in cands:
            if c.is_dir():
                aligned_dir = str(c)
                break
    if not aligned_dir or not Path(aligned_dir).is_dir():
        raise HTTPException(400,
            "找不到对齐集（DFLJPG）：期望 %s/aligned 或 %s_aligned。"
            "先跑一次「抽帧 + 人脸对齐」。" % (ip, ip))
    if not list(Path(aligned_dir).glob("*.jpg")):
        raise HTTPException(400, "对齐集是空的: " + aligned_dir)

    if out_raw.lower().endswith(".mp4"):
        out_mp4 = out_raw
    elif out_raw:
        out_mp4 = str(Path(out_raw) / "merged.mp4")
    else:
        out_mp4 = str(ip.parent / (ip.name + "_merged.mp4"))

    mask_onnx = str(body.get("xseg_onnx") or "")
    if not mask_onnx:
        for c in (WORKSPACE / "model" / "XSegLite" / "xseglite.onnx",
                  WORKSPACE / "model" / "XSeg" / "XSeg.onnx"):
            if c.is_file():
                mask_onnx = str(c)
                break
    settings = body.get("settings") or {}

    with _merge_job_lock:
        if _merge_job["running"]:
            raise HTTPException(409, "已经在合成")
        # 先把状态置成 running 再起线程：否则前端刚拿到 ok=True 就来轮询，
        # 会读到线程启动前的旧状态（running=False / ok=None），当成"已完成"。
        _merge_job.update({"running": True, "log": [], "ok": None, "error": None,
                           "out": out_mp4, "pct": 0, "msg": "启动合成…"})

    try:
        fps = float(body.get("fps") or _merge_cache.get("fps") or 30.0)
    except Exception:
        fps = 30.0
    try:
        workers = int(body.get("workers") or 0)
    except Exception:
        workers = 0
    _th2.Thread(target=_merge_worker,
                args=(input_dir, out_mp4, aligned_dir, dfm, settings, mask_onnx, fps, workers,
                      str(body.get("encoder") or "h264_nvenc"),
                      str(body.get("hwaccel") if body.get("hwaccel") is not None else "auto")),
                daemon=True).start()
    return {"ok": True, "input_dir": input_dir, "aligned_dir": aligned_dir,
            "output": out_mp4, "output_dir": out_mp4, "dfm": dfm, "settings": settings}


@app.get("/api/merge/export/status")
def merge_export_status():
    with _merge_job_lock:
        return {"running": _merge_job["running"], "ok": _merge_job["ok"],
                "error": _merge_job["error"], "out": _merge_job["out"],
                "pct": _merge_job.get("pct", 0), "msg": _merge_job.get("msg", ""),
                "stage": _merge_job.get("stage", -1),
                "cancelled": _merge_job.get("cancelled", False),
                "log": _merge_job["log"][-50:]}


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


@app.get("/api/merge/merged/{idx}")
def merge_merged_frame(idx: int, w: int = 900):
    """从「最近一次导出的成片」里取一帧，给工作台的「合成」标签用。"""
    import cv2
    out = (_merge_job.get("out") or "").strip()
    if not out or not Path(out).is_file():
        # 重启之后 _merge_job 是空的 —— 按当前项目的默认命名找一遍
        for cand in (str(_merge_cache.get("dir") or "") + "_merged.mp4",
                     str(_merge_cache.get("dir") or "") + os.sep + "merged.mp4"):
            if cand and Path(cand).is_file():
                out = cand
                break
    if not out or not Path(out).is_file():
        raise HTTPException(409, "还没有导出的成片")
    cap = cv2.VideoCapture(out)
    if not cap.isOpened():
        raise HTTPException(500, "打不开成片: " + out)
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, int(idx)))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise HTTPException(404, "成片里没有第 %d 帧" % idx)
    h0, w0 = frame.shape[:2]
    maxw = max(200, min(int(w), 1600))
    if w0 > maxw:
        frame = cv2.resize(frame, (maxw, int(h0 * maxw / w0)), interpolation=cv2.INTER_AREA)
    ok2, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
    if not ok2:
        raise HTTPException(500, "编码失败")
    return {"idx": int(idx), "out": out,
            "png": "data:image/jpeg;base64," + _b64.b64encode(buf.tobytes()).decode()}


@app.post("/api/merge/export/cancel")
def merge_export_cancel():
    """终止正在跑的合成子进程。"""
    global _merge_proc
    with _merge_job_lock:
        running = _merge_job["running"]
    if not running or _merge_proc is None:
        return {"ok": False, "reason": "no running export"}
    try:
        _merge_proc.terminate()
    except Exception as e:
        return {"ok": False, "reason": str(e)[:120]}
    with _merge_job_lock:
        _merge_job["cancelled"] = True
        _merge_job["msg"] = "cancelling"
    return {"ok": True}


@app.get("/api/merge/detect/{idx}")
def merge_detect(idx: int, model: str = "det_500m", thresh: float = 0.5):
    """对当前帧做人脸检测，返回原图坐标的框。视频 / 项目两种模式都支持。"""
    frame, _total = _merge_frame_image(idx)
    det = _get_detector(model)
    det.thresh = float(thresh)
    boxes = det.detect(frame)
    h, w = frame.shape[:2]
    return {"idx": int(idx), "width": int(w), "height": int(h),
            "model": model, "boxes": boxes}


@app.get("/api/merge/analyze")
def merge_analyze(step: int = 12, model: str = "det_500m", thresh: float = 0.5):
    """按步长扫全片，统计每帧的人脸数，给时间轴的人脸轨用。
    486 帧 / step=12 → 41 次推理，约 20 秒。"""
    total = int(_merge_cache.get("frames") or 0)
    step = max(1, min(int(step), 200))
    det = _get_detector(model)
    det.thresh = float(thresh)

    marks = []
    idx = 0
    scanned = 0
    while idx < total:
        try:
            frame, _n = _merge_frame_image(idx)
        except HTTPException:
            break
        boxes = det.detect(frame)
        marks.append({"idx": idx, "count": len(boxes),
                      "box": boxes[0]["box"] if boxes else None})
        scanned += 1
        idx += step
    return {"total": total, "step": step, "scanned": scanned, "marks": marks}


# ============================================================
# 遮罩编辑器（照搬 MaskProcessor/ui 的 WebUI）
# 只实现不依赖模型的三个：开项目 / 读图 / 存遮罩
# 需要 AI 的功能（SAM / BiSeNet / XSeg）返回 501，界面上会提示
# ============================================================
_mask_state = {"dir": None, "files": [], "index": 0}


@app.post("/api/project/open")
def mask_project_open(body: dict):
    """打开一个已对齐的人脸集目录，列出图片供遮罩编辑器使用。"""
    raw = (body or {}).get("path") or ""
    rel = raw
    low = rel.lower().replace("/", "\\")
    for pre in ("workspace\\", "workspace"):
        if low.startswith(pre):
            rel = rel[len(pre):].lstrip("\\/")
            break
    d = Path(raw) if (raw and Path(raw).is_absolute()) else (WORKSPACE / rel if rel else WORKSPACE / "data_src" / "aligned")
    if not d.is_dir():
        raise HTTPException(404, "目录不存在: " + str(d))
    files = sorted([p for p in d.iterdir()
                    if p.is_file() and p.suffix.lower() in IMAGE_EXT])
    resume = 0
    fp = WORKSPACE / "_mask_progress.json"
    if fp.is_file():
        try:
            memo = json.loads(fp.read_text(encoding="utf-8"))
            if memo.get("dir") and Path(memo["dir"]) == d:
                resume = int(memo.get("index") or 0)
        except Exception:
            pass
    _mask_state.update({"dir": str(d), "files": [p.name for p in files], "index": resume})
    return {"path": str(d), "count": len(files), "resume": resume,
            "files": [{"name": p.name, "size": p.stat().st_size} for p in files[:500]]}


@app.get("/api/image/{index}")
def mask_image(index: int):
    """返回当前项目的第 index 张图（base64），以及是否已有遮罩。"""
    import base64
    files = _mask_state.get("files") or []
    if not files:
        raise HTTPException(409, "还没有打开项目")
    index = max(0, min(int(index), len(files) - 1))
    d = Path(_mask_state["dir"])
    p = d / files[index]
    if not p.is_file():
        raise HTTPException(404, "文件不存在: " + files[index])
    raw = p.read_bytes()
    ext = p.suffix.lower().lstrip(".")
    mime = "jpeg" if ext in ("jpg", "jpeg") else ext
    _mask_state["index"] = index
    return {"index": index, "name": files[index], "total": len(files),
            "image": base64.b64encode(raw).decode(),   # 纯 base64，前端自己拼 data URL
            "mime": mime, "has_mask": True}


def _mask_ai_stub():
    raise HTTPException(501, "这个功能要加载 AI 模型（SAM / BiSeNet / XSeg），当前只接了手绘与保存")


@app.post("/api/mask/predict")
def mask_predict(body: dict):
    _mask_ai_stub()


@app.post("/api/mask/text")
def mask_text(body: dict):
    _mask_ai_stub()


@app.post("/api/mask/bisenet")
def mask_bisenet(body: dict):
    _mask_ai_stub()


@app.post("/api/model/preload")
def mask_preload(body: dict | None = None):
    return {"ok": True, "note": "模型预加载未接"}


@app.get("/api/image/{index}/mask")
def mask_existing(index: int):
    """读这张图已有的 XSeg 遮罩。DFLJPG 里的遮罩段暂时不解析，返回空当新图处理。"""
    return {"mask": None}


@app.post("/api/progress/save")
def mask_progress_save(body: dict | None = None):
    """记住看到第几张，下次进来接着干。落盘，重启也还在。"""
    body = body or {}
    idx = int(body.get("index") or 0)
    _mask_state["index"] = idx
    try:
        (WORKSPACE / "_mask_progress.json").write_text(
            json.dumps({"index": idx, "dir": _mask_state.get("dir") or ""}, ensure_ascii=False),
            encoding="utf-8")
    except Exception:
        pass
    return {"ok": True, "index": idx}


@app.get("/api/progress/load")
def mask_progress_load():
    """读上次的进度；文件里还有上次打开的目录，前端可以据此恢复。"""
    idx = _mask_state.get("index") or 0
    d = ""
    fp = WORKSPACE / "_mask_progress.json"
    if fp.is_file():
        try:
            data = json.loads(fp.read_text(encoding="utf-8"))
            idx = int(data.get("index") or idx)
            d = data.get("dir") or ""
        except Exception:
            pass
    return {"index": idx, "dir": d}


@app.post("/api/mask/save")
def mask_save(body: dict):
    """把画好的遮罩存成 PNG，写到 _masks/<同名>.png。"""
    import base64
    files = _mask_state.get("files") or []
    if not files:
        raise HTTPException(409, "还没有打开项目")
    idx = int((body or {}).get("index", _mask_state.get("index") or 0))
    data = (body or {}).get("mask") or ""
    if "," in data:
        data = data.split(",", 1)[1]
    try:
        raw = base64.b64decode(data)
    except Exception:
        raise HTTPException(400, "遮罩数据不是合法 base64")
    d = Path(_mask_state["dir"]) / "_masks"
    d.mkdir(exist_ok=True)
    out = d / (Path(files[idx]).stem + ".png")
    out.write_bytes(raw)
    return {"ok": True, "path": str(out)}


@app.get("/api/mask/models")
def mask_models(kind: str = "XSegLite"):
    """扫描 workspace/model/<kind>/ 下的 .onnx / .engine，供遮罩绘制页的模型下拉用。
    4.0 的 _refresh_xseg_file_list() 就是这个逻辑：优先 .engine（TRT 最快）。"""
    base = WORKSPACE / "model" / (kind if kind in ("XSeg", "XSegLite") else "XSegLite")
    files = []
    if base.is_dir():
        for f in sorted(base.iterdir()):
            if f.is_file() and f.suffix.lower() in (".onnx", ".engine"):
                files.append({"name": f.name, "path": str(f),
                              "mb": round(f.stat().st_size / 1048576, 1),
                              "trt": f.suffix.lower() == ".engine"})
    files.sort(key=lambda x: (not x["trt"], x["name"]))
    return {"kind": kind, "dir": str(base), "files": files}


@app.post("/api/merge/project")
def merge_project(body: dict):
    """看一眼目录是不是能用的 DFL 项目：有几张帧、有没有 aligned、带不带 landmarks。"""
    raw = str((body or {}).get("path") or "").strip()
    p = Path(raw)
    if not p.is_dir():
        raise HTTPException(404, "不是目录: " + raw)
    names, srcdir, kind = _dfl_project_scan(p)
    aligned = p / "aligned"
    sample = names[:24]
    lm = sum(1 for n in sample
             if _dfljpg_dict(Path(srcdir) / n).get("landmarks") is not None)
    return {
        "path": str(p), "kind": kind, "total": len(names),
        "has_aligned": aligned.is_dir(),
        "with_landmarks": lm,          # 采样 24 张的结果，不是全量
        "sampled": len(sample),
        "sample": names[:6],
    }


@app.get("/api/merge/info")
def merge_info():
    return dict(_merge_cache)



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


# ---------------- B 站扫码登录 ----------------
from . import biliauth

BILI = biliauth.CookieStore(WORKSPACE)
_bili_session = {"key": "", "created": 0}


@app.post("/api/bili/qr")
def bili_qr():
    try:
        d = biliauth.generate()
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"生成二维码失败：{e}")
    _bili_session["key"] = d["key"]
    _bili_session["created"] = int(__import__("time").time())
    return {"ok": True, "key": d["key"], "qr": d["qr_png"]}


@app.post("/api/bili/poll")
def bili_poll(payload: dict = None):
    key = (payload or {}).get("key") or _bili_session.get("key")
    if not key:
        raise HTTPException(status_code=400, detail="还没有生成二维码")
    try:
        r = biliauth.poll(key)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"轮询失败：{e}")

    if r.get("status") == "ok" and r.get("cookie"):
        cookie = r["cookie"]
        BILI.save(cookie)
        info = biliauth.fetch_user_info(cookie)
        if info:
            BILI.save_info(info)
        _bili_session["key"] = ""
        return {"status": "ok", "name": info.get("name", ""), "uid": info.get("uid", 0),
                "avatar": info.get("avatar", "")}
    return {"status": r.get("status", "pending"), "message": r.get("message", "")}


# ============================================================
# B 站下载：真的往 workspace/raw 写文件（用 ffmpeg 生成等价小片）
# 现在是「可用的占位」：文件名、时长、体积按画质档变化，落盘后提取器能直接用
# ============================================================
@app.post("/api/bili/download")
def bili_download(body: dict):
    """把一个视频「下载」到 workspace/raw/。返回真实落盘路径和大小。"""
    import subprocess
    import imageio_ffmpeg
    body = body or {}
    bvid = str(body.get("bvid") or "").strip()
    title = str(body.get("title") or "").strip() or bvid or "untitled"
    quality = str(body.get("quality") or "best")

    raw = WORKSPACE / "raw"
    raw.mkdir(parents=True, exist_ok=True)

    safe = "".join(c for c in title if c not in r'\/:*?"<>|').strip()[:60] or "video"
    name = (bvid + "_" if bvid else "") + safe + ".mp4"
    out = raw / name
    if out.exists():
        return {"ok": True, "path": str(out), "size": out.stat().st_size, "skipped": True}

    # 画质决定时长（30/20/12 秒），顺便决定文件体积
    secs = {"best": 30, "1080": 20, "720": 12}.get(quality, 20)
    size = {"best": "1280x720", "1080": "1280x720", "720": "960x540"}.get(quality, "1280x720")

    exe = imageio_ffmpeg.get_ffmpeg_exe()
    cmd = [exe, "-v", "error", "-y",
           "-f", "lavfi", "-i", "color=c=0x202024:s=%s:d=%d:r=24" % (size, secs),
           "-f", "lavfi", "-i", "sine=frequency=440:duration=%d" % secs,
           "-c:v", "libx264", "-preset", "ultrafast", "-crf", "32",
           "-c:a", "aac", "-b:a", "64k", "-shortest", str(out)]
    try:
        proc = subprocess.run(cmd, capture_output=True, timeout=120, creationflags=_NO_WINDOW)
    except Exception as e:
        raise HTTPException(500, "生成失败: " + str(e))
    if proc.returncode != 0 or not out.exists():
        raise HTTPException(500, "ffmpeg 退出码 %d: %s" % (
            proc.returncode, (proc.stderr or b"").decode("utf-8", "replace")[:200]))
    return {"ok": True, "path": str(out), "size": out.stat().st_size,
            "seconds": secs, "quality": quality}


@app.get("/api/bili/me")
def bili_me():
    cookie = BILI.load()
    if not cookie or "SESSDATA" not in cookie:
        return {"logged_in": False}
    info = BILI.load_info()
    if not info:
        info = biliauth.fetch_user_info(cookie)
        if info:
            BILI.save_info(info)
    return {"logged_in": bool(info), "name": info.get("name", ""),
            "uid": info.get("uid", 0), "avatar": info.get("avatar", "")}


@app.post("/api/bili/logout")
def bili_logout():
    BILI.clear()
    _bili_session["key"] = ""
    return {"ok": True}


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


# ---------------- 数据集体检（One Shot） ----------------
# 扫描在 **4.0 的 python** 里跑（_scan_runner.py）：
#   ① 那边带着完整的 numpy/cv2/numba，4.1 只负责起进程、转进度
#   ② 以后人脸 embedding / 聚类也能接在同一条流水线上
# 协议：@@P <done> <total> <stage> 进度、@@R <json> 结果、@@ERR <msg> 错误。
_oneshot_job = {"running": False, "done": 0, "total": 0, "stage": "", "result": None,
                "error": None, "dir": "", "started": 0.0}
_oneshot_lock = threading.Lock()


@app.post("/api/dataset/oneshot")
def dataset_oneshot(payload: dict):
    payload = payload or {}
    target = str(payload.get("dir") or "data_src")
    if os.path.isabs(target) or ".." in target.replace("\\", "/").split("/") or ":" in target:
        raise HTTPException(400, "非法的数据集目录: " + target)
    root = WORKSPACE / target
    if (root / "aligned").is_dir():
        root = root / "aligned"
    if not root.is_dir():
        raise HTTPException(404, "目录不存在：%s" % root)

    with _oneshot_lock:
        if _oneshot_job["running"]:
            raise HTTPException(409, "已经在扫了")

    level = str(payload.get("level") or "strict")
    score_pct = float(payload.get("score_pct", 5.0) if payload.get("score_pct") is not None else 5.0)
    bucket_floor = int(payload.get("bucket_floor", 0) or 0)
    total_files = len(_list_images(root))

    with _oneshot_lock:
        _oneshot_job.update({"running": True, "done": 0, "total": total_files,
                             "stage": "scan", "result": None, "error": None,
                             "dir": target, "started": time.time()})

    def _work():
        import subprocess
        py = _DFL_ROOT / "python" / "python.exe"
        runner = Path(__file__).resolve().parent / "_scan_runner.py"
        if not py.is_file() or not runner.is_file():
            with _oneshot_lock:
                _oneshot_job["running"] = False
                _oneshot_job["error"] = "找不到引擎 python 或 _scan_runner.py"
            return
        env = dict(os.environ)
        env["PYTHONIOENCODING"] = "utf-8"
        cmd = [str(py), str(runner), "--dir", str(root),
               "--level", level,
               "--score-pct", str(score_pct), "--bucket-floor", str(bucket_floor)]
        try:
            proc = subprocess.Popen(cmd, cwd=str(_DFL_ROOT), env=env,
                                    creationflags=_NO_WINDOW,
                                    stdin=subprocess.DEVNULL,
                                    stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                                    text=True, encoding="utf-8", errors="replace", bufsize=1)
            for line in proc.stdout:
                line = line.rstrip("\n")
                if line.startswith("@@P "):
                    parts = line.split()
                    if len(parts) >= 3:
                        with _oneshot_lock:
                            _oneshot_job["done"] = int(parts[1])
                            _oneshot_job["total"] = int(parts[2])
                            if len(parts) >= 4:
                                _oneshot_job["stage"] = parts[3]
                elif line.startswith("@@ERR "):
                    with _oneshot_lock:
                        _oneshot_job["error"] = line[6:]
                elif line.startswith("@@R "):
                    try:
                        data = json.loads(line[4:])
                    except Exception as e:
                        with _oneshot_lock:
                            _oneshot_job["error"] = "结果解析失败: %s" % e
                        continue
                    with _oneshot_lock:
                        _oneshot_job["result"] = data
            proc.wait()
            with _oneshot_lock:
                _oneshot_job["running"] = False
                if _oneshot_job["result"] is None and not _oneshot_job["error"]:
                    _oneshot_job["error"] = "扫描进程退出码 %s" % proc.returncode
        except Exception as e:
            with _oneshot_lock:
                _oneshot_job["running"] = False
                _oneshot_job["error"] = "%s: %s" % (type(e).__name__, e)

    threading.Thread(target=_work, daemon=True).start()
    return {"ok": True, "total": total_files}


@app.get("/api/dataset/oneshot/status")
def dataset_oneshot_status():
    with _oneshot_lock:
        j = dict(_oneshot_job)
    j["elapsed"] = round(time.time() - j["started"], 1) if j.get("started") else 0
    return j


@app.post("/api/dataset/scan")
def dataset_scan(payload: dict):
    """只统计不落盘：返回会被处理掉的文件清单。"""
    mode = payload.get("mode", "blur")
    target = payload.get("dir") or "data_src"
    root = WORKSPACE / target
    if (root / "aligned").is_dir():
        root = root / "aligned"          # DFL 惯例：对齐结果放在 aligned 子目录
    if not root.exists():
        raise HTTPException(status_code=404, detail=f"目录不存在：{root}")

    from PIL import Image
    files = _list_images(root)
    hits, bad = [], []

    if mode == "blur":
        thr = float(payload.get("threshold", 10.0))
        for p in files:
            try:
                v = _lap_var(Image.open(p))
                if v < thr:
                    hits.append({"name": p.name, "score": round(v, 2)})
            except Exception:
                bad.append(p.name)
    elif mode == "bad":
        for p in files:
            try:
                with Image.open(p) as im:
                    im.verify()
            except Exception:
                hits.append({"name": p.name, "score": 0})
    elif mode == "dup":
        thr = int(payload.get("threshold", 6))
        seen = []
        for p in files:
            try:
                h = _dhash(Image.open(p))
            except Exception:
                continue
            dup = None
            for (h2, name) in seen:
                if bin(h ^ h2).count("1") <= thr:
                    dup = name
                    break
            if dup:
                hits.append({"name": p.name, "score": 0, "dup_of": dup})
            else:
                seen.append((h, p.name))
    else:
        raise HTTPException(status_code=400, detail="未知模式")

    return {"total": len(files), "hits": hits[:2000], "count": len(hits), "unreadable": bad[:200]}


def _within(base: Path, p: Path) -> bool:
    """p 是否真的在 base 里面（挡住 ../ 和绝对路径替换基路径）。"""
    try:
        base_r = base.resolve()
        p_r = p.resolve()
    except Exception:
        return False
    try:
        p_r.relative_to(base_r)
        return True
    except ValueError:
        return False


@app.post("/api/dataset/apply")
def dataset_apply(payload: dict):
    """把命中的文件移到 _trash（可恢复）。

    注意：这个装饰器以前被错挂在 _within(base, p) 上面，FastAPI 于是把两个 Path
    参数当成路径参数 —— 任何调用都返回 422，清理功能从来没通过。
    """
    mode = payload.get("mode", "blur")
    target = str(payload.get("dir") or "data_src")
    # 只允许 workspace 下的相对目录名：绝对路径会把基路径整个替换掉
    if os.path.isabs(target) or ".." in target.replace("\\", "/").split("/") or ":" in target:
        raise HTTPException(400, "非法的数据集目录: " + target)
    names = payload.get("names") or []
    root = WORKSPACE / target
    if (root / "aligned").is_dir():
        root = root / "aligned"
    trash = root / "_trash"
    trash.mkdir(exist_ok=True)
    items = []
    for n in names:
        n = str(n)
        if not n or "/" in n or "\\" in n or n in (".", "..") or ":" in n:
            continue
        src = root / n
        if not src.exists() or not _within(root, src):
            continue
        dst = trash / n
        if not _within(trash, dst):
            continue
        i = 1
        while dst.exists():
            dst = trash / (src.stem + f"_{i}" + src.suffix)
            i += 1
        src.rename(dst)
        items.append({"name": n, "trash": dst.name})
    return {"ok": True, "moved": len(items), "items": items, "trash": str(trash)}


@app.post("/api/dataset/restore")
def dataset_restore(payload: dict):
    """把刚移走的文件放回去（撤销）。"""
    target = str(payload.get("dir") or "data_src")
    if os.path.isabs(target) or ".." in target.replace("\\", "/").split("/") or ":" in target:
        raise HTTPException(400, "非法的数据集目录: " + target)
    items = payload.get("items") or []
    root = WORKSPACE / target
    if (root / "aligned").is_dir():
        root = root / "aligned"
    trash = root / "_trash"
    back = 0
    for it in items:
        tn = str(it.get("trash", ""))
        nm = str(it.get("name", ""))
        if not tn or not nm:
            continue
        if "/" in tn or "\\" in tn or ":" in tn or "/" in nm or "\\" in nm or ":" in nm:
            continue
        src = trash / tn
        dst = root / nm
        if not src.exists() or not _within(trash, src) or not _within(root, dst):
            continue
        if dst.exists():
            continue
        src.rename(dst)
        back += 1
    return {"ok": True, "restored": back}


# ---------------- 面部 68 点 ----------------
def _face_path() -> Path:
    return WORKSPACE / "face68.json"


@app.get("/api/face")
def face_get():
    try:
        return {"points": json.loads(_face_path().read_text(encoding="utf-8"))}
    except Exception:
        return {"points": None}


@app.post("/api/face")
def face_save(payload: dict):
    pts = payload.get("points")
    if not isinstance(pts, list) or len(pts) != 68:
        raise HTTPException(status_code=400, detail="需要 68 个点")
    WORKSPACE.mkdir(parents=True, exist_ok=True)
    _face_path().write_text(json.dumps(pts, ensure_ascii=False), encoding="utf-8")
    return {"ok": True}


@app.delete("/api/face")
def face_clear():
    try:
        _face_path().unlink()
    except Exception:
        pass
    return {"ok": True}


# ---------------- 账号（B 站） ----------------
AUTH_FILE = None      # 运行时按工作区解析


def _auth_path() -> Path:
    global AUTH_FILE
    if AUTH_FILE is None:
        AUTH_FILE = WORKSPACE / "auth.json"
    return AUTH_FILE


def _auth_read() -> dict:
    try:
        return json.loads(_auth_path().read_text(encoding="utf-8"))
    except Exception:
        return {}


@app.get("/api/auth")
def auth_state():
    a = _auth_read()
    return {
        "logged_in": bool(a.get("uid")),
        "uid": a.get("uid", 0),
        "name": a.get("name", ""),
        "avatar": a.get("avatar", ""),
        "since": a.get("since", 0),
        "onboarded": bool(a.get("onboarded")),
    }


@app.post("/api/auth/login")
def auth_login(payload: dict):
    """真正的扫码流程接上之前，先用它写入一次登录态。"""
    WORKSPACE.mkdir(parents=True, exist_ok=True)
    data = {
        "uid": int(payload.get("uid") or 0),
        "name": payload.get("name") or "",
        "avatar": payload.get("avatar") or "",
        "since": int(__import__("time").time()),
        "onboarded": True,
    }
    _auth_path().write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"ok": True, **data}


@app.post("/api/auth/skip")
def auth_skip():
    WORKSPACE.mkdir(parents=True, exist_ok=True)
    a = _auth_read()
    a["onboarded"] = True
    _auth_path().write_text(json.dumps(a, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"ok": True}


@app.post("/api/auth/logout")
def auth_logout():
    try:
        _auth_path().unlink()
    except Exception:
        pass
    return {"ok": True}


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


@app.get("/api/model_spec")
def model_spec():
    """新建模型页要用的全部下拉/范围/推荐值，一次拿全。"""
    tables = _spec_tables()
    gpus = _gpu_cached()
    rec = _recommend_for(max([g["free_gb"] for g in gpus], default=0.0))
    out = dict(tables)
    out["recommend"] = rec
    out["gpu"] = max(gpus, key=lambda g: g["free_gb"]) if gpus else None
    return out


@app.get("/api/gpu")
def gpu_info():
    gpus = _gpu_cached()
    best = max(gpus, key=lambda g: g["free_gb"]) if gpus else None
    rec = _recommend_for(best["free_gb"] if best else 0.0)
    presets = []
    for k, v in _PRESETS.items():
        presets.append({kk: vv for kk, vv in v.items() if kk != "weights_gb"} | {"key": k})
    return {"gpus": gpus, "best": best, "recommend": rec, "presets": presets}


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


@app.post("/api/models/create")
def model_create(body: dict):
    """按档位建一个模型（写一份 iter=1 的 data.dat，4.0 见到就当作已有模型）。"""
    import pickle as _pk
    body = body or {}
    name = _safe_model_name(body.get("name"))
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    dat_path = MODEL_DIR / (name + "_SAEHD_data.dat")
    if dat_path.exists():
        raise HTTPException(409, "已经有同名模型：" + name + "（换个名字，或直接去训练页继续练）")

    gpus = _gpu_list()
    free = max([g["free_gb"] for g in gpus], default=0.0)
    rec = _recommend_for(free)

    preset_key = str(body.get("preset") or "").strip()
    if preset_key not in _PRESETS:
        preset_key = rec["preset"]
    p = dict(_PRESETS[preset_key])

    def _pick(key, cast, fallback):
        v = body.get(key)
        if v is None or v == "":
            return fallback
        try:
            return cast(v)
        except Exception:
            return fallback

    # 架构字符串：优先用「架构 + 子分支 + 算子」三段式（和 4.0 的 GUI 同一套规则），
    # 也允许直接传 archi。合法性交给 4.0 的 parse_archi_string 判，不在 4.1 里另写一份。
    spec = _archi_spec()
    archi_display = str(body.get("archi_display") or "").strip()
    if archi_display:
        archi = spec.build_archi_string(archi_display,
                                        str(body.get("subarchi") or ""),
                                        str(body.get("kernel_label") or spec.default_kernel_label()))
        if spec.parse_archi_string(archi) is None:
            raise HTTPException(400, "架构组合不合法：" + archi)
    else:
        archi = str(body.get("archi") or p["archi"])
        if spec.parse_archi_string(archi) is None:
            raise HTTPException(400, "架构写法不合法：" + archi)

    def _clamp(v, lo, hi, step):
        v = max(lo, min(hi, int(v)))
        return int(round(v / float(step)) * step)

    options = {
        "resolution": _clamp(_pick("resolution", int, p["resolution"]), 64, 640, 32),
        "face_type": str(body.get("face_type") or "wf"),
        "archi": archi,
        "ae_dims": _clamp(_pick("ae_dims", int, p["ae_dims"]), 64, 1024, 8),
        "e_dims": _clamp(_pick("e_dims", int, p["e_dims"]), 16, 256, 8),
        "d_dims": _clamp(_pick("d_dims", int, p["d_dims"]), 16, 256, 8),
        "d_mask_dims": _clamp(_pick("d_mask_dims", int, p["d_mask_dims"]), 8, 128, 8),
        "batch_size": _pick("batch_size", int, rec["batch_size"]),
        "target_iter": max(0, _pick("target_iter", int, 0)),
        "optimizer": str(body.get("optimizer") or "adabelief"),
        "lr": float(body.get("lr") or 5e-05),
        "lr_dropout": "n",
        "masked_training": True,
        "eyes_mouth_prio": False,
        "uniform_yaw": False,
        "ct_mode": "none",
        "random_warp": True,
        "gan_power": 0.0,
        "true_face_power": 0.0,
        "pretrain": False,
        "use_fast_generator": True,
        "models_opt_on_gpu": True,
    }
    for k in ("src", "dst"):
        v = str(body.get(k) or "").strip()
        if v:
            options[k] = v

    data = {"iter": 1, "options": options, "loss_history": [],
            "sample_for_preview": None, "choosed_gpu_indexes": None}
    tmp = dat_path.with_name(dat_path.name + ".tmp")
    tmp.write_bytes(_pk.dumps(data, protocol=4))
    os.replace(str(tmp), str(dat_path))

    # 顺手写 4.0 的参数缓存：模型库和下次启动都不用再解析 data.dat
    try:
        st = dat_path.stat()
        cache = {"version": 1, "iter": 1, "options": options,
                 "choosed_gpu_indexes": None, "loss_count": 0,
                 "src_size": st.st_size, "src_mtime_ns": st.st_mtime_ns}
        Path(str(dat_path) + ".params").write_bytes(_pk.dumps(cache, protocol=4))
    except Exception:
        pass

    return {"ok": True, "name": name, "preset": preset_key,
            "dat": str(dat_path),
            "options": {k: options[k] for k in
                        ("resolution", "batch_size", "archi", "face_type",
                         "ae_dims", "e_dims", "d_dims", "d_mask_dims")},
            "note": p["note"]}


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
    py = _DFL_ROOT / "python" / "python.exe"
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


@app.post("/api/train/start")
def train_start(body: dict):
    body = body or {}
    name = str(body.get("name") or "").strip()
    if not name:
        raise HTTPException(400, "缺少模型名")
    dat = _model_dat(name)
    if dat is None:
        raise HTTPException(404, "找不到模型 %s —— 先用「新建模型」建一个，或检查 workspace/model 里有没有 %s_SAEHD_data.dat" % (name, name))
    if not (_DFL_ROOT / "python" / "python.exe").is_file():
        raise HTTPException(500, "找不到 4.0 的引擎（%s）—— 见 app/server/_paths.py：可以写 engine.txt 或设 DFL_ENGINE_ROOT" % _DFL_ROOT)
    with _train_job_lock:
        if _train_job["running"]:
            raise HTTPException(409, "已经在训练 " + _train_job["name"])
    fixed = _repair_option_strings(name)
    if fixed:
        print("[trainer] 修好了参数里的说明文字: " + ", ".join(fixed), flush=True)
    _th3.Thread(target=_train_worker, args=(name, bool(body.get("cpu_only"))), daemon=True).start()
    return {"ok": True, "name": name, "running": True}


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


@app.post("/api/train/save")
def train_save():
    """立即保存：让 4.0 在下一个迭代结束后落盘（不打断训练）。"""
    with _train_job_lock:
        if not _train_job["running"]:
            raise HTTPException(409, "现在没有在训练")
        _train_job["saving"] = True
        _train_job["saving_until"] = time.time() + 20
    try:
        _webui_post("/request-save")
    except Exception as e:
        with _train_job_lock:
            _train_job["saving"] = False
        raise HTTPException(503, "训练还没加载完（WebUI 未就绪）：%s" % str(e)[:120])
    return {"ok": True, "msg": "已请求保存，下一个迭代后落盘"}


@app.post("/api/train/pause")
def train_pause():
    """暂停：挂起训练进程（psutil suspend），迭代停在原地，显存不释放。"""
    global _train_proc
    with _train_job_lock:
        if not _train_job["running"]:
            raise HTTPException(409, "现在没有在训练")
        if _train_job["paused"]:
            return {"ok": True, "paused": True}
    proc = _train_proc
    if proc is None or proc.poll() is not None:
        raise HTTPException(409, "训练进程不在")
    try:
        import psutil
        p = psutil.Process(proc.pid)
        p.suspend()
        for c in p.children(recursive=True):
            try:
                c.suspend()
            except Exception:
                pass
    except Exception as e:
        raise HTTPException(500, "挂起失败：%s" % str(e)[:120])
    with _train_job_lock:
        _train_job["paused"] = True
    return {"ok": True, "paused": True}


@app.post("/api/train/resume")
def train_resume():
    global _train_proc
    with _train_job_lock:
        if not _train_job["running"]:
            raise HTTPException(409, "现在没有在训练")
        if not _train_job["paused"]:
            return {"ok": True, "paused": False}
    proc = _train_proc
    if proc is None or proc.poll() is not None:
        raise HTTPException(409, "训练进程不在")
    try:
        import psutil
        p = psutil.Process(proc.pid)
        for c in p.children(recursive=True):
            try:
                c.resume()
            except Exception:
                pass
        p.resume()
    except Exception as e:
        raise HTTPException(500, "恢复失败：%s" % str(e)[:120])
    with _train_job_lock:
        _train_job["paused"] = False
    return {"ok": True, "paused": False}


@app.post("/api/train/stop")
def train_stop():
    """停止：先请 4.0 保存并退出（/request-quit），超时再强杀整棵树。"""
    with _train_job_lock:
        running = _train_job["running"]
        _train_job["stopping"] = True
    if not running:
        n = _sweep_orphan_trainers("(停止时发现遗留)")
        return {"ok": True, "msg": ("已清理 %d 个遗留训练进程" % n) if n else "没有在训练"}
    # 挂起状态下整个进程（包括 WebUI 线程）都冻着，先恢复再发指令 ——
    # 否则每一次 /request-quit 都会超时，白等 90 秒最后强杀。
    with _train_job_lock:
        paused = _train_job["paused"]
    if paused:
        try:
            import psutil
            if _train_proc:
                p_ = psutil.Process(_train_proc.pid)
                for c_ in p_.children(recursive=True):
                    try:
                        c_.resume()
                    except Exception:
                        pass
                p_.resume()
        except Exception:
            pass
        with _train_job_lock:
            _train_job["paused"] = False
        time.sleep(0.5)

    graceful = False
    for attempt in range(6):
        try:
            _webui_post("/request-quit", timeout=4.0)
            graceful = True
            break
        except Exception:
            time.sleep(1.5)

    # 等它自己收尾（保存 + 退出）；最多 90 秒，然后再强杀
    for _ in range(90):
        if _train_proc is None or _train_proc.poll() is not None:
            break
        time.sleep(1)
    if _train_proc is not None and _train_proc.poll() is None:
        _kill_train_tree()
        return {"ok": True, "graceful": False, "msg": "已强制结束（没等到它自己退出）"}
    return {"ok": True, "graceful": graceful,
            "msg": "已保存并退出" if graceful else "已结束"}


@app.get("/api/train/preview")
def train_preview(wait: float = 0.0):
    """把 4.0 WebUI 的实时预览转出来。

    训练进程会在 127.0.0.1:6790 起一个监控服务：
      GET /request-preview  → 让它下一轮吐一张
      GET /preview          → pop 语义，取走就没了

    wait=N 会**一直等到有新图**（最多 N 秒）再返回。前端点「刷新」时用它，
    否则拿到的是上一次的缓存 —— 看起来就是「点了没反应 / 有滞后」。

    返回体在最外层补三个字段（4.0 原样透传 previews/labels）：
      fresh  这次是不是真的拿到了新图
      iter   这张图对应的迭代数（拿到新图时记录）
      age_s  这张图距今多少秒
    """
    import json as _json
    import time as _time
    import urllib.request

    port = WEBUI_PORT
    base = "http://127.0.0.1:%d" % port
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def _get(path, timeout=1.2):
        req = urllib.request.Request(base + path, headers={"User-Agent": "dfl41"})
        with opener.open(req, timeout=timeout) as r:
            return r.read()

    def _pack(raw, fresh):
        try:
            d = _json.loads(raw)
        except Exception:
            d = {"previews": [], "labels": {}}
        d["fresh"] = bool(fresh)
        d["iter"] = int(_last_preview_state.get("iter") or 0)
        at = float(_last_preview_state.get("at") or 0)
        d["age_s"] = round(time.time() - at, 1) if at else None
        return Response(content=_json.dumps(d).encode("utf-8"),
                        media_type="application/json",
                        headers={"Cache-Control": "no-store"})

    deadline = time.time() + max(0.0, min(float(wait or 0.0), 25.0))
    requested = False
    raw = None

    while True:
        if not requested:
            try:
                _get("/request-preview", timeout=0.8)
                requested = True
            except Exception:
                # 训练没在跑（或还没起 WebUI）：别让前端干等
                cached = _last_preview_state.get("body")
                if cached:
                    return _pack(cached, False)
                return _pack(b'{"previews":[],"labels":{}}', False)
        try:
            d = _json.loads(_get("/preview"))
            if d.get("previews"):
                raw = _json.dumps(d).encode("utf-8")
                _last_preview_state.update({"body": raw,
                                            "iter": _train_job.get("iter") or 0,
                                            "at": time.time()})
                return _pack(raw, True)
        except Exception:
            pass
        if time.time() >= deadline:
            break
        _time.sleep(0.2)

    cached = _last_preview_state.get("body")
    if cached:
        return _pack(cached, False)
    return _pack(b'{"previews":[],"labels":{}}', False)


@app.get("/api/train/status")
def train_status():
    with _train_job_lock:
        j = _train_job
        it = int(j.get("iter") or 0)
        rate = float(j.get("it_s") or 0.0)
        started = float(j.get("started_at") or 0.0)
        target = int(j.get("target") or 0)
        elapsed = (time.time() - started) if (started and j["running"]) else 0.0
        eta = None
        if target and rate > 0 and it < target:
            eta = int((target - it) / rate)
        return {"running": j["running"], "name": j["name"],
                "iter": it, "ok": j["ok"],
                "error": j["error"], "log": j["log"][-30:],
                "it_s": rate, "elapsed_s": int(elapsed),
                "target_iter": target, "eta_s": eta,
                "iter_seen": bool(j.get("iter_seen")),
                "paused": bool(j.get("paused")),
                "saving": bool(j.get("saving")) and time.time() < float(j.get("saving_until") or 0),
                "stopping": bool(j.get("stopping")),
                "last_backup_iter": j.get("last_backup_iter") or 0,
                "lr": j.get("lr") or 0.0,
                "loss_src": j.get("loss_src"),
                "loss_dst": j.get("loss_dst"),
                "history": j.get("history") or [],
                "has_preview": bool(_last_preview_state.get("body")),
                "preview_iter": _last_preview_state.get("iter") or 0}


@app.get("/api/models/{name}/stats")
def model_stats(name: str):
    """训练页右侧那几条：显存、利用率、磁盘、上次保存、下次自动、备份数。

    以前这些数字是写死在 HTML 里的（"9.2 / 12 GB"、"98%"、"3 / 5 份"），
    页面看起来一直很健康，其实和真相无关。这里按模型现场算。
    """
    import pickle as _pk
    name = _safe_model_name(name)
    dat = _model_dat(name)
    if dat is None:
        raise HTTPException(404, "找不到模型 " + name)

    opts = {}
    try:
        opts = (_pk.loads(dat.read_bytes()).get("options") or {})
    except Exception:
        opts = {}

    # GPU：训练时显存几乎满，用 nvidia-smi 的实时值
    gpu = None
    try:
        import shutil as _sh
        import subprocess as _sp
        exe = _sh.which("nvidia-smi") or r"C:\Windows\System32\nvidia-smi.exe"
        if Path(exe).is_file():
            out = _sp.run([exe, "--query-gpu=index,name,memory.total,memory.free,utilization.gpu",
                           "--format=csv,noheader,nounits"],
                          capture_output=True, text=True, timeout=8,
                          creationflags=_NO_WINDOW).stdout
            rows = [x.strip() for x in out.strip().splitlines() if x.strip()]
            if rows:
                p = [x.strip() for x in rows[0].split(",")]
                gpu = {"index": int(p[0]), "name": p[1],
                       "total_gb": round(int(p[2]) / 1024.0, 1),
                       "free_gb": round(int(p[3]) / 1024.0, 1),
                       "used_gb": round((int(p[2]) - int(p[3])) / 1024.0, 1),
                       "util": int(p[4]) if p[4].isdigit() else None}
    except Exception:
        gpu = None

    # 磁盘：模型目录所在盘还剩多少 + 这个模型的体积
    disk_free_gb = None
    model_gb = None
    try:
        import shutil as _sh
        disk_free_gb = round(_sh.disk_usage(str(MODEL_DIR)).free / (1024 ** 3), 1)
        total = sum(f.stat().st_size for f in MODEL_DIR.glob(name + "_*") if f.is_file())
        model_gb = round(total / (1024 ** 3), 2)
    except Exception:
        pass

    # 备份：4.0 会写到 <模型>_<arch>_autobackups/ 下
    backup_count = 0
    backup_dir = None
    try:
        cands = [p for p in MODEL_DIR.glob(name + "_*_autobackups") if p.is_dir()]
        if cands:
            backup_dir = max(cands, key=lambda p: p.stat().st_mtime)
            # 4.0 的备份是 autobackups/01、02、03 … 每个编号一个目录，里面才是权重文件。
            # 以前只数顶级文件，所以永远显示 0。
            dirs = [e for e in backup_dir.iterdir() if e.is_dir() and e.name.isdigit()]
            if dirs:
                backup_count = len(dirs)
            else:
                backup_count = len([f for f in backup_dir.iterdir() if f.is_file()])
    except Exception:
        pass

    # CPU / 内存：psutil.cpu_percent(None) 返回「上次调用到现在」的平均占用，
    # 页面每 2 秒轮询一次，正好是这个区间的真实值（第一次调用返回 0）。
    cpu_percent = None
    ram_used_gb = ram_total_gb = ram_percent = None
    try:
        import psutil
        cpu_percent = round(float(psutil.cpu_percent(None)), 1)
        vm = psutil.virtual_memory()
        ram_used_gb = round(vm.used / (1024 ** 3), 1)
        ram_total_gb = round(vm.total / (1024 ** 3), 1)
        ram_percent = round(float(vm.percent), 1)
    except Exception:
        pass

    last_save_ts = None
    try:
        last_save_ts = int(dat.stat().st_mtime)
    except Exception:
        pass

    return {"name": name, "gpu": gpu, "disk_free_gb": disk_free_gb, "model_gb": model_gb,
            "cpu_percent": cpu_percent,
            "ram_used_gb": ram_used_gb, "ram_total_gb": ram_total_gb, "ram_percent": ram_percent,
            "backup_count": backup_count, "has_backup_dir": backup_dir is not None,
            "backup_interval": int(opts.get("backup_interval") or 0),
            "max_backups": int(opts.get("max_backups") or 0)}


@app.get("/api/models")
def models():
    out = []
    if MODEL_DIR.is_dir():
        for dat in sorted(MODEL_DIR.glob("*_data.dat")):
            try:
                out.append(_model_entry(dat))
            except Exception:
                continue
    out.sort(key=lambda m: m.get("trained_at", 0), reverse=True)
    # 每个模型的状态：正在训练 / 已暂停 / 已停止。前端卡片上的标签直接用它
    with _train_job_lock:
        cur = _train_job.get("name") if _train_job.get("running") else ""
        paused = bool(_train_job.get("paused"))
    for m in out:
        if cur and m.get("name") == cur:
            m["state"] = "已暂停" if paused else "训练中"
        else:
            m["state"] = "已停止"
    pid = _training_pid()
    alive = False
    if pid:
        try:
            import psutil  # noqa: F401
            alive = psutil.pid_exists(pid)
        except Exception:
            import os as _os
            try:
                _os.kill(pid, 0)
                alive = True
            except Exception:
                alive = False
    return {"models": out, "training_pid": pid if alive else 0}


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


@app.get("/api/models/{name}/preview.png")
def model_preview(name: str):
    from fastapi.responses import FileResponse
    path, mime = _find_model_preview(name)
    if path is None:
        raise HTTPException(status_code=404)
    return FileResponse(str(path), media_type=mime)


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
            self.stage = 0          # 0 抽帧 / 1 对齐 / 2 补遮罩 / 3 完成
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

    done_idx = set()
    for f in face_dir.glob("*_*.jpg"):
        head = f.stem.split("_")[0]
        if head.isdigit():
            done_idx.add(int(head))

    vf = []
    if rotate in (90, 180, 270):
        vf.append({90: "transpose=1", 180: "transpose=1,transpose=1", 270: "transpose=2"}[rotate])
    # 帧跳跃=1 时 fps 就等于源帧率：这时候再加 fps 滤镜没有任何意义，
    # 反而会因为时间戳抖动丢帧/补帧。只有真要降帧率才加。
    if not src_fps or float(fps) < src_fps - 1e-3:
        vf.append("fps=%g" % max(0.01, float(fps)))

    # vf 为空时不能传 -vf ""（ffmpeg 会报错），所以整段参数一起省掉
    out_params = ["-vf", ",".join(vf)] if vf else []
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
                        JOB.faces = stats["made"]
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
        JOB.faces = stats["made"]
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


def _get_xseg():
    """加载 XSeg 遮罩模型（XSegLite 优先）。找不到就返回 None。"""
    if _xseg_cache["tried"]:
        return _xseg_cache["m"]
    _xseg_cache["tried"] = True
    try:
        from ._xseg import XSegMasker
        cands = [WORKSPACE / "model" / "XSegLite" / "xseglite.onnx",
                 WORKSPACE / "model" / "XSeg" / "XSeg.onnx"]
        for c in cands:
            if c.is_file():
                _xseg_cache["m"] = XSegMasker(str(c), workspace=WORKSPACE,
                                              engine_name=("xseglite" if "Lite" in str(c) else "xseg"))
                return _xseg_cache["m"]
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


def _mask_worker(folder: str, workers: int = 4):
    """给一个 aligned 目录里的所有 DFLJPG 补 XSeg 遮罩（提取之后的单独一步）。

    提取本身不跑 XSeg：DFL 的流程也是 extract → mask 两步，
    而且遮罩模型是 3.5 ms/张，放在提取热路径上纯属浪费。
    """
    import cv2
    from concurrent.futures import ThreadPoolExecutor
    from ._align import read_dfljpg_meta, rewrite_dfljpg_meta, encode_xseg_mask

    xseg = _get_xseg()
    root = Path(folder)
    files = sorted(root.glob("*.jpg")) + sorted(root.glob("*.png"))
    total = len(files)
    with JOB.lock:
        JOB.running = True
        JOB.msg = "补遮罩：共 %d 张" % total
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
                enc = encode_xseg_mask(xseg.mask(img, out_size=256))
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
                    JOB.msg = "补遮罩 %d / %d · 新写入 %d" % (done["n"], total, done["ok"])

    with ThreadPoolExecutor(max_workers=max(1, int(workers))) as pool:
        list(pool.map(_one, files))

    with JOB.lock:
        JOB.running = False
        JOB.pct = 100
        JOB.msg = "补遮罩完成：%d 张，新增 %d" % (total, done["ok"])
    return done["ok"]


# ---------------- TensorRT 引擎（编译/状态） ----------------
_engine_job = {"running": False, "name": "", "log": [], "error": "", "done": False}


def _engine_roots():
    """能找 onnx 的根目录：工作区优先，其次引擎仓。"""
    return [WORKSPACE, Path(__file__).resolve().parents[2],
            Path(__file__).resolve().parents[3] / "DeepFaceLab-PyTorch"]


def _engines_verify_all():
    """启动时在后台把引擎状态核一遍（md5 + SM），不阻塞启动。"""
    try:
        from ._engines import status as _st
        _st(WORKSPACE, _engine_roots())
    except Exception:
        pass


@app.get("/api/devices")
def api_devices():
    """可用训练设备：GPU 的真实名字 + 显存。

    界面上的"设备"下拉以前只有 0 / 1 / 0,1 这种序号 —— 用户得自己开命令行
    跑 nvidia-smi 才知道 0 是哪张卡。这里直接把型号返给前端。
    """
    gpus = []
    try:
        import subprocess as _sp
        r = _sp.run(["nvidia-smi", "--query-gpu=index,name,memory.total",
                     "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=6,
                    creationflags=getattr(_sp, "CREATE_NO_WINDOW", 0))
        for line in (r.stdout or "").strip().splitlines():
            parts = [x.strip() for x in line.split(",")]
            if len(parts) >= 3 and parts[0].isdigit():
                gpus.append({"idx": int(parts[0]), "name": parts[1],
                             "mem_mb": int(float(parts[2])) if parts[2].replace(".", "").isdigit() else 0})
    except Exception:
        pass
    cpu = os.environ.get("PROCESSOR_IDENTIFIER") or "CPU"
    try:
        cpu = (os.cpu_count() or 0) and ("CPU (%d 线程)" % (os.cpu_count() or 0)) or "CPU"
    except Exception:
        pass
    return {"gpus": gpus, "cpu": cpu}


@app.get("/api/build")
def api_build():
    """界面构建时间：拿 app.js / app.css / index.html 的最新修改时间。

    用途是把"你看到的到底是哪一版"变成可核对的事实 —— 之前排查样式问题时
    反复在猜客户端有没有重载。页面右下角会显示这个时间。
    """
    files = [WEB / "index.html", WEB / "js" / "app.js", WEB / "css" / "app.css"]
    best = 0.0
    names = {}
    for f in files:
        try:
            m = f.stat().st_mtime
        except Exception:
            continue
        names[f.name] = int(m)
        best = max(best, m)
    return {"ts": int(best),
            "iso": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(best)) if best else "",
            "files": names}


@app.get("/api/engines")
def engines_status():
    from ._engines import status as _st
    st = _st(WORKSPACE, _engine_roots())
    st["job"] = dict(_engine_job)
    return st


def _engine_build_worker(name: str):
    import subprocess
    import sys
    from ._engines import status as _st, engine_path as _ep
    from ._engines import MODEL_SPECS

    st = _st(WORKSPACE, _engine_roots())
    row = next((m for m in st["models"] if m["name"] == name), None)
    if not row or not row.get("onnx"):
        with _engine_job_lock:
            _engine_job.update({"running": False, "error": "找不到模型 " + name, "done": True})
        return
    eng = _ep(WORKSPACE, name, row["onnx"], st["sm"])
    if eng is None:
        with _engine_job_lock:
            _engine_job.update({"running": False, "error": "没有可用的 GPU", "done": True})
        return

    from ._engines import PROFILES
    cmd = [sys.executable, str(Path(__file__).parent / "_trt_build.py"),
           "--onnx", row["onnx"], "--out", str(eng), "--name", name, "--bf16"]
    prof = PROFILES.get(name)
    if prof:
        cmd += ["--profile", ";".join(",".join(str(v) for v in p) for p in prof)]
    with _engine_job_lock:
        _engine_job.update({"running": True, "name": name, "log": ["开始编译 " + name],
                            "error": "", "done": False, "pct": 0})
    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, bufsize=1, creationflags=_NO_WINDOW)
        for line in proc.stdout:
            line = line.strip()
            if line.startswith("@@P "):
                with _engine_job_lock:
                    try:
                        pct = int(line.split()[1])
                        _engine_job["pct"] = pct
                    except Exception:
                        pass
                    _engine_job["log"].append(line[4:])
                    _engine_job["log"] = _engine_job["log"][-40:]
            elif line.startswith("@@OK "):
                with _engine_job_lock:
                    _engine_job["log"].append("完成: " + Path(line[5:]).name)
            elif line.startswith("@@ERR "):
                with _engine_job_lock:
                    _engine_job["error"] = line[6:]
                    _engine_job["log"].append("失败: " + line[6:])
        proc.wait()
    except Exception as e:
        with _engine_job_lock:
            _engine_job["error"] = "%s: %s" % (type(e).__name__, e)
    with _engine_job_lock:
        _engine_job["running"] = False
        _engine_job["done"] = True


_engine_job_lock = threading.Lock()


@app.post("/api/engines/build")
def engines_build(payload: dict | None = None):
    payload = payload or {}
    name = str(payload.get("name") or "").strip()
    if not name:
        raise HTTPException(400, "缺少 name")
    with _engine_job_lock:
        if _engine_job.get("running"):
            raise HTTPException(409, "已有编译任务在跑：" + str(_engine_job.get("name")))
        _engine_job.update({"running": True, "name": name, "log": [], "error": "", "done": False, "pct": 0})
    threading.Thread(target=_engine_build_worker, args=(name,), daemon=True).start()
    return {"ok": True, "name": name}


@app.post("/api/dataset/mask")
def dataset_mask(payload: dict | None = None):
    """POST {dir: 'data_src/aligned', workers: 4} —— 给整个目录补 XSeg 遮罩。"""
    payload = payload or {}
    rel = str(payload.get("dir") or "data_src/aligned")
    root = (WORKSPACE / rel).resolve()
    if not str(root).startswith(str(WORKSPACE.resolve())):
        raise HTTPException(400, "路径不合法")
    if not root.is_dir():
        raise HTTPException(404, "目录不存在: %s" % rel)
    workers = int(payload.get("workers") or 4)
    with JOB.lock:
        JOB.running = True
        JOB.msg = "启动补遮罩…"
    threading.Thread(target=_mask_worker, args=(str(root), workers), daemon=True).start()
    return {"ok": True, "dir": str(root), "workers": workers}


def _resolve_dir(dst: str) -> Path:
    r"""把界面里填的目录解析成绝对路径。

    界面上显示的是「workspace\data_src\aligned」这种带前缀的相对路径。
    以前后端直接 Path(dst) —— 相对的是 serve.py 的当前目录（app/），
    于是帧被写到 app\workspace\...，用户在自己的 workspace 里什么都看不到。
    现在：绝对路径原样用；相对路径去掉开头的 workspace 段后挂到 WORKSPACE 下。
    """
    raw = str(dst or "").strip().strip('"')
    if not raw:
        return WORKSPACE
    q = Path(raw)
    if q.is_absolute():
        return q
    parts = [x for x in raw.replace("/", "\\").split("\\") if x and x != "."]
    if not parts:
        return WORKSPACE
    # workspace\... 仍然指向真实工作区（4.0 那边），保持老行为不改
    if parts[0].lower() == "workspace":
        rest = parts[1:]
        return WORKSPACE.joinpath(*rest) if rest else WORKSPACE
    # 其余相对路径一律相对**本仓库根目录**：
    #   abc\data_src\aligned → D:\MyProject\DeepFaceLab-PyTorch-4.1\abc\data_src\aligned
    return _APP_ROOT.joinpath(*parts)


@app.post("/api/extract/start")
def extract_start(payload: dict | None = None):
    """真抽帧：src = 视频文件，dst = 输出目录（默认 workspace/_frames），fps = 抽帧频率。"""
    payload = payload or {}
    src = (payload.get("src") or "").strip()
    dst = (payload.get("dst") or "").strip()
    fps = float(payload.get("fps") or 2)
    skip = int(payload.get("skip") or 0)      # 每 N 帧取 1；给了就以源帧率为准换算 fps
    rotate = int(payload.get("rotate") or 0)
    quality = str(payload.get("quality") or "high")
    align = bool(payload.get("align", True))
    size = int(payload.get("size") or 0)   # 0 = 不限制；分辨率跟着源图和检测框走

    if not src:
        raise HTTPException(400, "缺少输入视频路径")
    p = Path(src)
    if not p.is_file():
        raise HTTPException(404, "视频不存在: " + src)
    face_dir = _resolve_dir(dst) if dst else (WORKSPACE / "_frames")
    face_dir.mkdir(parents=True, exist_ok=True)
    dst = str(face_dir)

    JOB.reset()
    # 起线程前就把 running 置上：前端拿到 ok 立刻订阅 SSE，
    # 若这时还是 idle，收流逻辑会以为任务没起来（见 extract_progress 里的 idle 判定）。
    with JOB.lock:
        JOB.running = True
        JOB.msg = "启动抽帧…"
    threading.Thread(target=_frames_worker,
                     args=(str(p), dst, fps, rotate, align, quality, size, skip),
                     daemon=True).start()
    return {"ok": True, "src": str(p), "dst": dst,
            "frames": str(face_dir.parent / (face_dir.name + "_frames")),
            "fps": fps, "skip": skip,
            "rotate": rotate, "align": align, "quality": quality, "size": size}


@app.post("/api/extract/cancel")
def extract_cancel():
    JOB.running = False
    return {"ok": True}


@app.get("/api/extract/progress")
async def extract_progress():
    async def gen():
        last = None
        idle = 0.0
        while True:
            snap = JOB.snapshot()
            if snap != last:
                yield "data: " + json.dumps(snap, ensure_ascii=False) + "\n\n"
                last = snap
                idle = 0.0
            if not snap["running"] and snap["pct"] >= 100:
                return
            if not snap["running"]:
                # 任务没在跑、进度又没到 100：多半是"点了开始但工作线程还没起来"，
                # 给 1 秒宽限；再没有动静就收流。否则这条 SSE 会一直挂着不返回
                # （实测客户端等 60s 也拿不到响应）。
                idle += 0.08
                if idle > 1.0:
                    return
            await asyncio.sleep(0.08)

    return StreamingResponse(gen(), media_type="text/event-stream")




# 启动时后台核对一次引擎状态（md5 + SM 对不上会被标成过期/未编译）
threading.Thread(target=_engines_verify_all, daemon=True).start()
