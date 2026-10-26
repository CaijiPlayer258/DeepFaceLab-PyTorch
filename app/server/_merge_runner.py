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

用法（必须用 4.0 的 python 跑，工作目录随意，脚本会自己 sys.path）：
  python _merge_runner.py --frames DIR --aligned DIR --dfm X.dfm \
      --out Y.mp4 [--config '{...}'] [--xseg X.onnx] [--workers N]

进度按行输出，父进程解析：
  @@P <stage> <pct 0..1> <msg>
  @@DONE <输出文件>
  @@ERR <原因>
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import shutil
import struct
import sys
import traceback
from pathlib import Path

IMG_EXT = (".jpg", ".jpeg", ".png")

# GUI 拉起时不能弹黑框（父进程没有控制台，子进程默认会新建一个）
_NO_WINDOW = 0x08000000 if os.name == "nt" else 0

# 4.0 的合成管线自己 CreateProcess ffmpeg，不认我们的 creationflags；
# 装个全局补丁把整棵进程树都盖住。
try:
    from ._nowin import install as _install_nowin
except ImportError:
    from _nowin import install as _install_nowin
_install_nowin()


def _say(msg):
    print(msg, flush=True)


def _prog(stage, pct, msg=""):
    print("@@P %d %.4f %s" % (int(stage), float(pct), msg), flush=True)


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
    for idx, fp in enumerate(frames):
        staged = "%08d%s" % (idx, fp.suffix.lower().replace(".jpeg", ".jpg"))
        _link_or_copy(fp, proj / staged)
        name_to_idx[fp.name] = idx
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
    _say("[stage] 对齐脸 %d 张（跳过 %d）" % (kept, skipped))
    if not kept:
        raise RuntimeError("没有任何对齐脸能对上帧（source_filename 与帧名不匹配）")
    return proj


def _resolve_ffmpeg(explicit_ffmpeg: str, explicit_ffprobe: str, dfl_root: Path):
    """找出可用的 ffmpeg / ffprobe：显式参数 > 4.0 自带 > imageio-ffmpeg > PATH。"""
    cands = []
    if explicit_ffmpeg:
        cands.append(Path(explicit_ffmpeg))
    cands.append(dfl_root / "ffmpeg" / "ffmpeg.exe")
    try:
        import imageio_ffmpeg
        cands.append(Path(imageio_ffmpeg.get_ffmpeg_exe()))
    except Exception:
        pass
    cands.append(Path(shutil.which("ffmpeg") or ""))
    ffmpeg = next((str(c) for c in cands if c and Path(c).is_file()), "")
    if not ffmpeg:
        raise RuntimeError("找不到 ffmpeg（4.1 的 imageio-ffmpeg 也没有？）")

    probes = []
    if explicit_ffprobe:
        probes.append(Path(explicit_ffprobe))
    probes.append(Path(ffmpeg).with_name("ffprobe.exe"))
    probes.append(Path(ffmpeg).parent / "ffprobe.exe")
    probes.append(Path(shutil.which("ffprobe") or ""))
    ffprobe = next((str(c) for c in probes if c and Path(c).is_file()), "")
    return ffmpeg, ffprobe


def _make_encoder_no_probe(ffmpeg: str, fps: float):
    """替换 _stage5_encode_video：不查 ffprobe、不混音，直接按已知 fps 编码。"""
    def _encode(frames_dir, output_path, source_video, encoder, fmt,
                progress, stop_event=None, cut_segments=None):
        import subprocess
        ext = "png" if fmt == "png" else "jpg"
        frames = sorted(Path(frames_dir).glob("*." + ext))
        if not frames:
            _say("[encode] 没有可编码的帧: %s" % frames_dir)
            return
        pad = len(frames[0].stem) if frames[0].stem.isdigit() else 8
        pattern = str(Path(frames_dir) / ("%%0%dd.%s" % (pad, ext)))
        progress(6, 0.2, "FFmpeg 编码")
        cmd = [ffmpeg, "-y", "-v", "error", "-r", "%g" % fps, "-i", pattern,
               "-c:v", encoder or "libx264", "-pix_fmt", "yuv420p", str(output_path)]
        r = subprocess.run(cmd, capture_output=True, text=True, errors="replace",
                           creationflags=_NO_WINDOW)
        if r.returncode != 0 or not Path(output_path).is_file():
            # 硬件编码器不可用（没 NVENC / 驱动不匹配）时退回软编码
            cmd[cmd.index("-c:v") + 1] = "libx264"
            _say("[encode] 硬件编码失败，退回 libx264")
            r = subprocess.run(cmd, capture_output=True, text=True, errors="replace",
                               creationflags=_NO_WINDOW)
            if r.returncode != 0:
                raise RuntimeError("编码失败: %s" % (r.stderr or "").strip()[:200])
        progress(6, 1.0, "编码完成")
    return _encode


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", required=True)
    ap.add_argument("--aligned", required=True)
    ap.add_argument("--dfm", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--config", default="{}")
    ap.add_argument("--xseg", default="")
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--encoder", default="h264_nvenc")
    ap.add_argument("--hwaccel", default="auto")
    ap.add_argument("--image-format", default="jpg")
    ap.add_argument("--fps", type=float, default=25.0)
    ap.add_argument("--ffmpeg", default="")
    ap.add_argument("--ffprobe", default="")
    ap.add_argument("--dfl-root", default="")
    args = ap.parse_args(argv)

    dfl_root = Path(args.dfl_root) if args.dfl_root else Path(__file__).resolve()
    if not args.dfl_root:
        # app/server/_merge_runner.py → 4.1 根 → 隔壁 4.0
        dfl_root = Path(__file__).resolve().parents[2].parent / "DeepFaceLab-PyTorch"
    if not (dfl_root / "MergeStudio").is_dir():
        raise RuntimeError("找不到 4.0 的 MergeStudio: %s" % dfl_root)
    sys.path.insert(0, str(dfl_root))

    frames_dir = Path(args.frames)
    aligned_dir = Path(args.aligned)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    dfm = Path(args.dfm)
    if not dfm.is_file():
        raise RuntimeError("DFM 不存在: %s" % dfm)

    # staging 目录名必须是纯 ASCII：MergeStudio 内部用 cv2.imwrite 写合并帧，
    # OpenCV 在 Windows 上走 ANSI 路径，遇到中文目录会**静默失败**（merged/ 变空，
    # 最后编码阶段报"没有可编码的帧"，但整体还返回成功）。
    import hashlib as _hashlib
    _slug = _hashlib.sha1(str(out_path.stem).encode("utf-8")).hexdigest()[:8]
    work = out_path.parent / (".merge_" + _slug)
    work.mkdir(parents=True, exist_ok=True)

    _prog(0, 0.0, "搭 staging 项目目录")
    proj = stage_project(frames_dir, aligned_dir, work)

    _prog(0, 0.2, "加载模型")

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
    _sess = model_loader.load_model(str(dfm))
    try:
        _prov = _sess.get_providers() if hasattr(_sess, "get_providers") else []
        _say("[merge] DFM 运行在: %s" % (", ".join(_prov) or "?"))
    except Exception:
        pass
    _say("[merge] 模型已加载: %s" % dfm.name)

    cfg = json.loads(args.config or "{}")
    # MergeStudio 的键名：mask_mode / mode / color_transfer_mode / erode/blur_mask_modifier
    config = {
        "mask_mode": int(cfg.get("mask_mode", 6)),
        "mode": cfg.get("mode", "overlay"),
        "color_transfer_mode": cfg.get("color_transfer_mode", "none"),
        "erode_mask_modifier": int(cfg.get("erode_mask_modifier", 0)),
        "blur_mask_modifier": int(cfg.get("blur_mask_modifier", 0)),
    }
    if cfg.get("face_margin") is not None:
        config["face_margin"] = float(cfg["face_margin"])

    _say("[merge] 合成参数: %s" % json.dumps(config, ensure_ascii=False))

    import multiprocessing
    workers = args.workers if args.workers > 0 else max(1, multiprocessing.cpu_count() // 2)

    def progress_callback(stage, stage_name, pct, msg=""):
        _prog(stage, pct, "%s %s" % (stage_name, msg or ""))

    import MergeStudio.core.export_pipeline as ep
    ffmpeg, ffprobe = _resolve_ffmpeg(args.ffmpeg, args.ffprobe, dfl_root)
    ep._FFMPEG = ffmpeg
    _say("[merge] ffmpeg: %s" % ffmpeg)
    if ffprobe:
        ep._FFPROBE = ffprobe
    else:
        # 4.0 的管线用 ffprobe 读源视频的 fps/码率，但这套环境里只有 ffmpeg。
        # 我们本来就知道 fps（占位源视频是自己造的），所以直接换一个不依赖
        # ffprobe 的收尾编码，音轨也不需要（帧目录本来就没有音频）。
        _say("[merge] 没有 ffprobe，改用内置收尾编码")
        ep._stage5_encode_video = _make_encoder_no_probe(ep._FFMPEG, float(args.fps))

    from MergeStudio.core.export_pipeline import run_export_pipeline
    # DFL 模式靠 video_path 的 stem 定位项目目录：<proj>/<stem> 必须等于 proj。
    # 而且 Stage5 要从源视频里取 fps 和音轨（-map 1:a:0），空文件会直接把编码炸掉，
    # 所以这里现造一个 2 秒的静音占位片。
    fake_video = proj.parent / (proj.name + ".mp4")
    import subprocess as _sp
    _sp.run([ffmpeg, "-y", "-v", "error",
             "-f", "lavfi", "-i", "color=c=black:s=64x64:r=%g:d=2" % float(args.fps),
             "-f", "lavfi", "-i", "anullsrc=r=44100:cl=mono",
             "-shortest", "-c:v", "libx264", "-pix_fmt", "yuv420p", str(fake_video)],
            capture_output=True, creationflags=_NO_WINDOW)
    if not fake_video.is_file() or fake_video.stat().st_size == 0:
        raise RuntimeError("造占位源视频失败（ffmpeg）: %s" % ffmpeg)
    _say("[merge] 占位源视频: %s (%.0f fps, 静音)" % (fake_video.name, float(args.fps)))
    _prog(0, 0.3, "开始合成（%d workers）" % workers)
    _enc = (args.encoder or "h264_nvenc").strip() or "h264_nvenc"
    _hw = "" if str(args.hwaccel).strip().lower() in ("off", "0", "false", "") else "auto"
    _say("[merge] 编码器: %s · 解码加速: %s" % (_enc, _hw or "关"))
    run_export_pipeline(
        video_path=str(fake_video),
        output_path=str(out_path),
        image_format=("png" if str(args.image_format).lower() == "png" else "jpg"),
        encoder=_enc,
        hwaccel=_hw,
        config=config,
        face_db={},
        face_model_map={},
        cut_segments=None,
        angle_segments=None,
        progress_callback=progress_callback,
        num_workers=workers,
    )
    try:
        shutil.rmtree(work, ignore_errors=True)
    except Exception:
        pass
    print("@@DONE %s" % out_path, flush=True)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        print("@@ERR %s" % str(e).replace("\n", " ")[:400], flush=True)
        traceback.print_exc()
        sys.exit(2)
