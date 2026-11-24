# -*- coding: utf-8 -*-
"""媒体基础件：ffmpeg 抽帧 / rawvideo 解码管道 / DFLJPG 元数据 / 图像读取（从 app/server/app.py 切出）。

搬运时只有一处改动：包内相对 import 的深度（app/server/app.py 里的 from . 换成 from ..），
函数体 / 注释 / 日志文案逐字节照抄。
"""
from __future__ import annotations

from pathlib import Path

from .._common.bootstrap import _NO_WINDOW


# ============================================================
# 合成工作室 · 视频预览（只读帧，不做 AI 推理）
# 依赖 imageio[ffmpeg]：_python 里已装 imageio 2.38 + ffmpeg 7.1
# ============================================================

def _video_probe(path):
    import imageio_ffmpeg
    import imageio.v3 as iio
    frames, _secs = imageio_ffmpeg.count_frames_and_secs(str(path))
    meta = iio.immeta(str(path), plugin="FFMPEG")
    fps = float(meta.get("fps") or 30.0)
    size = meta.get("size") or (0, 0)
    return int(frames), fps, int(size[0]), int(size[1])

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
        ("numpy.core.multiarray", "scalar"),
        ("numpy._core.multiarray", "scalar"),
        ("numpy", "ndarray"),
        ("numpy", "dtype"),
        ("collections", "OrderedDict"),
        ("builtins", "dict"), ("builtins", "list"), ("builtins", "tuple"),
        ("builtins", "set"), ("builtins", "frozenset"),
        ("builtins", "int"), ("builtins", "float"), ("builtins", "str"),
        ("builtins", "bytes"), ("builtins", "bool"),
        # protocol≤2 没有 BINBYTES：bytes（xseg_mask 就是 bytes）一律存成
        # _codecs.encode(str,'latin1')。少了这条，带遮罩的 DFLJPG 在这里
        # 一律解不开 → 读出来是个空 dict，看着像"这张图没有元数据"。
        # routes_mask._APP15_ALLOWED / _align._ALLOWED 都补过同一条。
        ("_codecs", "encode"),
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

def _open_video(src, out_params, first=False, in_params=None, pix_fmt="bgr24"):
    """… pix_fmt="nv12" 时管道里传的是 NV12（1080p 3.1MB，比 bgr24 的 6.2MB 少一半），
    代价是下游要一次 cv2.cvtColor(NV12→BGR)（实测 2~3ms，比多传 3MB 便宜）。
    """
    """打开 rawvideo 解码管道：先按 _hwaccel 的结论上硬解，失败静默换软解。

    硬解在探测阶段过了、但这条片子解不开（编码 profile 不支持 / 显存不够）时也不
    抛给调用方 —— 关掉重开一条软解管道就是了，绝不让一次抽取因为硬解而挂掉。
    first=True 时顺带把第一帧读出来：硬解「给了头但吐不出帧」这种假成功只有真读
    一帧才发现得了。
    """
    import imageio_ffmpeg
    from .. import _hwaccel

    ip = list(in_params) if in_params is not None else _hwaccel.input_params(src)
    mode = _hwaccel.current()
    attempts = [(mode, ip), ("none", [])] if ip else [("none", [])]
    last = None
    for name, params in attempts:
        gen = None
        try:
            gen = imageio_ffmpeg.read_frames(
                str(src),
                pix_fmt=pix_fmt,               # bgr24（默认）/ nv12（提取主路：带宽减半）
                bits_per_pixel=(12 if pix_fmt == "nv12" else 24),
                input_params=list(params),
                output_params=list(out_params),
            )
            meta = next(gen)
            W, H = meta.get("size") or (0, 0)
            if not W or not H:
                raise RuntimeError("ffmpeg 没给出尺寸")
            if not first:
                return gen, meta
            try:
                f0 = next(gen)
            except StopIteration:
                f0 = None
            _need = W * H * 3 // 2 if pix_fmt == "nv12" else W * H * 3
            if f0 is None or len(f0) != _need:
                raise RuntimeError("解不出帧（%d 字节，期望 %d）" % (len(f0) if f0 else -1, _need))
            return gen, meta, f0
        except Exception as e:                      # noqa: BLE001
            last = e
            if gen is not None:
                try:
                    gen.close()
                except Exception:
                    pass
            if name != "none":
                _hwaccel.note_failure(name, e)
                print("[decode] hwaccel=%s 实跑失败，本次改用 none：%s"
                      % (name, str(e)[:160]), flush=True)
    raise last


def _extract_frames(video: Path, out_dir: Path, total_hint: int = 0,
                    on_frame=None) -> int:
    """把视频逐帧抽成 000000.jpg…（6 位，和 aligned 的 source_filename 对齐）。

    从 0 开始编号：_align_stream 的帧号 idx 也是 0 起，两边同号才能让
    aligned/000000_0.jpg 的 source_filename="000000" 指回真实的
    000000.jpg —— 以前这条写的是 -start_number 1，整整差一位，于是
    stage_project 认不出任何一张对齐脸（"没有任何对齐脸能对上帧"）。
    """
    import subprocess
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("*.jpg"):
        try:
            old.unlink()
        except OSError:
            pass
    cmd = [_ffmpeg_exe(), "-y", "-hide_banner", "-loglevel", "error", "-nostats",
           "-i", str(video), "-q:v", "2", "-start_number", "0",
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
