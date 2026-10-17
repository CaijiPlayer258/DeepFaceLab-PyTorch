"""
分片滚动导出管线（chunked export）。

背景：4K50 视频全量切帧占用巨大（~3.5MB/帧 × 3000帧/分钟 ≈ 10GB/分钟）。
方案：把「切帧→换脸→合成→编码」按分片滚动执行，每片完成后删除帧图，
     峰值磁盘占用 ≈ 单个分片大小；分片 mp4 用 concat demuxer 无损拼接。
前提：项目已有 aligned（切脸+筛脸全局一次完成，帧号全局对齐）。

分片大小策略：按帧缓冲盘剩余空间的百分比（默认 25%），封顶用户指定上限。
"""
import subprocess
import time
from pathlib import Path

from MergeStudio.core.export_pipeline import (
    _FFMPEG, _FFPROBE, STAGE_NAMES, StopRequested,
    probe_frame_count,
)


def _disk_free_bytes(path: Path) -> int:
    import shutil
    try:
        return shutil.disk_usage(str(path)).free
    except Exception:
        return 0


def plan_chunks(total_frames: int, buffer_dir: Path,
                free_frac: float = 0.25, max_chunk_gb: float = 20.0,
                bytes_per_frame: float = 3.5 * 1024 * 1024,
                min_frames: int = 300, headroom_gb: float = 5.0) -> list:
    """按缓冲盘剩余空间规划分片。

    返回 [(start0, end0_inclusive), ...]（0-based 全局帧号闭区间）。
    - 预算 = min(剩余空间 × free_frac, max_chunk_gb) - 安全余量
    - 至少留 min_frames，避免极端小盘死循环
    """
    free = _disk_free_bytes(buffer_dir)
    budget_bytes = min(free * free_frac, max_chunk_gb * 1024**3)
    budget_bytes = max(0.0, budget_bytes - headroom_gb * 1024**3)
    per_chunk = int(budget_bytes // bytes_per_frame)
    per_chunk = max(per_chunk, min_frames)
    if per_chunk >= total_frames:
        return [(0, total_frames - 1)]
    chunks = []
    s = 0
    while s < total_frames:
        e = min(s + per_chunk - 1, total_frames - 1)
        chunks.append((s, e))
        s = e + 1
    return chunks


def extract_frame_range(video_path, out_dir: Path, start0: int, end0: int, fmt="jpg",
                        hwaccel=None):
    """切出全局帧号 [start0, end0]（0-based 闭区间）到 out_dir，
    文件名保持全局帧号（%08d 从 start0+1 开始，1-based 命名与现有管线一致）。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    ext = "png" if fmt == "png" else "jpg"
    n = end0 - start0 + 1
    cmd = [_FFMPEG, "-y"]
    if hwaccel:
        cmd += ["-hwaccel", hwaccel]
    cmd += [
        "-i", str(video_path),
        "-vf", f"select='between(n,{start0},{end0})'",
        "-vsync", "0",
        "-frame_pts", "0",
        "-start_number", str(start0 + 1),  # ffmpeg image2 序列起始编号
        "-pix_fmt", "rgb24",
    ]
    if ext == "jpg":
        cmd += ["-q:v", "2"]
    cmd += [str(out_dir / f"%08d.{ext}")]
    # 注意：image2 的 -start_number 只对输入有效；输出用 -frame_pts 或手动重命名。
    # 稳妥方案：先输出 1..n，再按全局帧号重命名。
    tmp_pattern = str(out_dir / f"tmp_%08d.{ext}")
    cmd[-1] = tmp_pattern
    r = subprocess.run(cmd, capture_output=True)
    if r.returncode != 0:
        raise RuntimeError(f"chunk extract failed: {r.stderr.decode(errors='replace')[:300]}")
    # 重命名 tmp_00000001 -> 全局帧号+1（与全量切帧的 1-based 编号一致）
    i = start0 + 1
    for f in sorted(out_dir.glob(f"tmp_*.{ext}")):
        target = out_dir / f"{i:08d}.{ext}"
        f.rename(target)
        i += 1
    return i - (start0 + 1)  # 实际切出帧数


def encode_chunk(frames_dir: Path, chunk_mp4: Path, source_video, encoder,
                 fps: float, src_bitrate: str = ""):
    """把合成后的帧目录编码成分片 mp4（无音频，音轨在 concat 阶段统一取）。"""
    files = sorted(frames_dir.glob("*.jpg")) + sorted(frames_dir.glob("*.png"))
    if not files:
        raise RuntimeError(f"no frames to encode in {frames_dir}")
    ext = files[0].suffix.lstrip(".")
    # 重编号为 1..N（gap-free）
    for idx, f in enumerate(files):
        t = frames_dir / f"cenc_{idx+1:08d}.{ext}"
        f.rename(t)
    for idx in range(len(files)):
        t = frames_dir / f"cenc_{idx+1:08d}.{ext}"
        g = frames_dir / f"{idx+1:08d}.{ext}"
        if t != g:
            t.rename(g)
    pattern = str(frames_dir / f"%08d.{ext}")
    cmd = [_FFMPEG, "-y", "-r", str(fps), "-i", pattern,
           "-c:v", encoder, "-pix_fmt", "yuv420p"]
    if src_bitrate:
        cmd += ["-b:v", src_bitrate]
    cmd += [str(chunk_mp4)]
    r = subprocess.run(cmd, capture_output=True)
    if r.returncode != 0:
        raise RuntimeError(f"chunk encode failed: {r.stderr.decode(errors='replace')[:300]}")


def concat_chunks(chunk_mp4s: list, output_path: Path, source_video,
                  encoder_params_note=""):
    """concat demuxer 无损拼接分片，音轨从原视频映射。
    若无分片（单段），直接走单文件音轨合成。"""
    if len(chunk_mp4s) == 1:
        cmd = [_FFMPEG, "-y", "-i", str(chunk_mp4s[0]), "-i", str(source_video),
               "-map", "0:v:0", "-map", "1:a:0?", "-c", "copy", "-shortest",
               str(output_path)]
    else:
        lst = output_path.parent / f".{output_path.stem}_concat.txt"
        lst.write_text("\n".join(f"file '{p.as_posix()}'" for p in chunk_mp4s),
                       encoding="utf-8")
        cmd = [_FFMPEG, "-y", "-f", "concat", "-safe", "0", "-i", str(lst),
               "-i", str(source_video),
               "-map", "0:v:0", "-map", "1:a:0?", "-c", "copy", "-shortest",
               str(output_path)]
    r = subprocess.run(cmd, capture_output=True)
    if r.returncode != 0:
        # 回退：重编码视频轨（不同参数分片时 concat copy 可能失败）
        raise RuntimeError(f"concat failed: {r.stderr.decode(errors='replace')[:300]}")
    return True
