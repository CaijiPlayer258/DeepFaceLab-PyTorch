"""
分片导出主编排：run_chunked_export。

前置：项目已有 aligned（DFL 模式判定同 run_export_pipeline）。
流程：读全局 aligned → DB → [分片循环: 切帧→换脸→合成→编码→删帧] → concat。
空间：峰值 = 单分片帧图 + swap/mask 中间产物（在 buffer_dir）。
"""
import json
import shutil
import sqlite3
import subprocess
import time
from pathlib import Path

from MergeStudio.core.export_pipeline import (
    _FFMPEG, _FFPROBE, STAGE_NAMES, StopRequested,
    _check_ffmpeg, probe_frame_count, _init_db,
    _stage2_read_dfl_aligned, _stage3_match_faces, _stage4_swap_faces_mp,
    _sanitize_config, kill_children,
)
from MergeStudio.core.chunked_export import (
    plan_chunks, extract_frame_range, encode_chunk, concat_chunks, _disk_free_bytes,
)


def _probe_fps_bitrate(source_video):
    r = subprocess.run(
        [_FFPROBE, "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=r_frame_rate,bit_rate", "-of", "csv=p=0", str(source_video)],
        capture_output=True, text=True)
    parts = r.stdout.strip().split(',')
    fps = 30.0
    try:
        a, b = parts[0].split('/')
        fps = float(a) / float(b) if float(b) else 30.0
    except Exception:
        pass
    br = parts[1].strip() if len(parts) > 1 and parts[1].strip() else ""
    return fps, br


def run_chunked_export(
    video_path: str, output_path: str,
    encoder: str = "h264_nvenc",
    config: dict = None,
    face_db: dict = None,
    face_model_map: dict = None,
    cut_segments: list = None,
    buffer_dir: str = "",
    free_frac: float = 0.25,
    max_chunk_gb: float = 20.0,
    hwaccel: str = "",
    progress_callback=None,
    stop_event=None,
    num_workers: int = 4,
):
    """分片滚动导出。aligned 必须已存在（切脸+筛脸完成）。"""
    _check_ffmpeg()
    video = Path(video_path)
    proj_dir = video.parent / video.stem
    aligned_dir = proj_dir / "aligned"
    if not aligned_dir.is_dir():
        raise RuntimeError(f"chunked export requires an existing DFL project (aligned/ not found in {proj_dir})")

    # 帧缓冲目录：默认输出目录同盘（F:），可指向 SSD
    buf = Path(buffer_dir) if buffer_dir else Path(output_path).parent
    buf.mkdir(parents=True, exist_ok=True)

    work_dir = buf / f".chunked_{video.stem}"
    if work_dir.exists():
        shutil.rmtree(str(work_dir), ignore_errors=True)
    work_dir.mkdir(parents=True)
    db_path = work_dir / "export.db"
    chunks_dir = work_dir / "chunks"
    chunks_dir.mkdir()

    def progress(stage, pct, msg=""):
        if stop_event and stop_event.is_set():
            raise StopRequested("Export cancelled")
        if progress_callback:
            progress_callback(stage, STAGE_NAMES[stage], pct, msg)

    total_frames = probe_frame_count(str(video)) or 0
    if total_frames <= 0:
        raise RuntimeError("cannot probe total frame count from source video")
    print(f"[Chunked] source: {total_frames} frames", flush=True)

    fps, src_bitrate = _probe_fps_bitrate(video)
    print(f"[Chunked] fps={fps} bitrate={src_bitrate or 'n/a'}", flush=True)

    # ---- Stage 2: 全局读 aligned（一次性，与全量导出一致） ----
    progress(1, 0.0, "Loading aligned faces...")
    _init_db(db_path)
    _stage2_read_dfl_aligned(aligned_dir, proj_dir, db_path, progress, stop_event)

    # ---- Stage 3: 身份匹配（一次性全局） ----
    if face_model_map:
        progress(2, 0.0, "Matching identities...")
        _stage3_match_faces(proj_dir, db_path, face_db or {}, face_model_map or {},
                            progress, stop_event, aligned_dir=str(aligned_dir))
    else:
        progress(2, 1.0, "No references — default model for all")

    # ---- 分片规划（按缓冲盘剩余空间） ----
    chunks = plan_chunks(total_frames, buf, free_frac=free_frac, max_chunk_gb=max_chunk_gb)
    free_gb = _disk_free_bytes(buf) / 1024**3
    print(f"[Chunked] buffer={buf} free={free_gb:.1f}GB → {len(chunks)} chunk(s), "
          f"first={chunks[0]}, last={chunks[-1]}", flush=True)
    progress(0, 0.05, f"分片规划：{len(chunks)} 片（按 {free_gb:.0f}GB 空余的 {int(free_frac*100)}%）")

    # cut_segments（0-based 帧号闭区间）过滤：分片落在选中段内
    _valid_segs = [s for s in (cut_segments or []) if s.get('start') is not None and s.get('end') is not None]

    chunk_files = []
    t0 = time.time()
    done_fr = 0
    for ci, (s0, e0) in enumerate(chunks):
        if stop_event and stop_event.is_set():
            raise StopRequested("Export cancelled")
        n = e0 - s0 + 1
        progress(0, min(0.95, done_fr / total_frames),
                 f"分片 {ci+1}/{len(chunks)}：切帧 {s0}-{e0}（{n} 帧）")
        frames_dir = work_dir / f"frames_{ci:03d}"
        extract_frame_range(str(video), frames_dir, s0, e0, fmt="jpg", hwaccel=hwaccel or None)

        # 换脸+合成（复用 phase worker 池；merged 输出到独立目录）
        merged_dir = work_dir / f"merged_{ci:03d}"
        merged_dir.mkdir(parents=True, exist_ok=True)
        import MergeStudio.core.export_pipeline as _ep
        _ep._phase_merge_out_dir = str(merged_dir)
        _ep._cut_keep_set = None  # 分片本身就是范围控制
        # 阶段进度通过外层回调换算：内层 stage 3/4/5 映射到全局 3/4/5
        def _sub(stage, pct, msg="", _ci=ci, _n=n):
            if stop_event and stop_event.is_set():
                raise StopRequested("Export cancelled")
            base = done_fr / total_frames
            span = _n / total_frames
            if progress_callback and stage in (3, 4, 5):
                progress_callback(stage, min(1.0, base + span * pct),
                                  f"[{_ci+1}/{len(chunks)}] {msg}")
        _stage4_swap_faces_mp(frames_dir, db_path, work_dir, _sanitize_config(config or {}),
                              face_model_map or {}, _sub, stop_event, num_workers,
                              cut_segments=None)
        # swap/mask 中间产物清理（swap_pred/masks 很大）
        for d in ("swap_pred", "masks"):
            p = work_dir / d
            if p.exists():
                shutil.rmtree(str(p), ignore_errors=True)

        # 编码本片（帧图用完即删）
        chunk_mp4 = chunks_dir / f"chunk_{ci:04d}.mp4"
        progress(6, min(0.98, (done_fr + n) / total_frames),
                 f"分片 {ci+1}/{len(chunks)}：编码")
        encode_chunk(merged_dir, chunk_mp4, str(video), encoder, fps, src_bitrate)
        chunk_files.append(chunk_mp4)

        shutil.rmtree(str(frames_dir), ignore_errors=True)
        shutil.rmtree(str(merged_dir), ignore_errors=True)
        done_fr += n
        el = time.time() - t0
        print(f"[Chunked] chunk {ci+1}/{len(chunks)} done "
              f"({done_fr}/{total_frames}f, {el:.0f}s, buffer free {_disk_free_bytes(buf)/1024**3:.1f}GB)", flush=True)

    # ---- 拼接 + 音轨 ----
    progress(6, 0.99, "拼接分片 + 音轨...")
    ok = concat_chunks(chunk_files, Path(output_path), str(video))
    if not ok:
        raise RuntimeError("concat failed")
    print(f"\n[Chunked] [OK] 导出完成: {output_path}", flush=True)
    progress(6, 1.0, "Complete")

    # 清理工作目录（保留 concat 失败时的调试线索由外层决定；正常路径直接删）
    shutil.rmtree(str(work_dir), ignore_errors=True)
    if not work_dir.exists():
        print(f"[Chunked] [OK] 工作目录已清理: {work_dir.name}", flush=True)
