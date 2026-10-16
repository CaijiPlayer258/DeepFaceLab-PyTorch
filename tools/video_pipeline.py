# -*- coding: utf-8 -*-
"""video_pipeline —— 视频流水线（深变版）：切帧 → 自适应切脸 → (查重) → faceid 分拣 → 人工筛 UI

2026-10-17 用户指定：
  - 切帧流程不变：沿用老工作流 dfl_woni_video_pipeline_v2 的 ffmpeg 参数
    （默认 1fps PNG，HDR/DV hable tonemap），ffmpeg 用 F:\\dfl\\_internal\\ffmpeg。
  - 切脸换深变 Extractor（RetinaFace_10g + 2d106det，动态尺寸直出，px<短边4位>__ 前缀命名，
    与 person_extract 同款；-d/-l/-t 可换算法）。
  - 多视频流水重叠：切视频 N 的脸时，预启动视频 N+1 的 ffmpeg 切帧
    （参考老脚本 pending_frame_jobs 模式；ffmpeg 解码占 CPU，Extractor 占 GPU，互不抢）。
  - 切完 → 可选 --dedup（Filter.py --mode repeated，phash 查重移 repeated\\；放聚类前，
    避免相邻秒的重复脸灌爆 faceid 簇）→ faceid 聚类 → 出组快照 → 自动打开 faceid_browser 人工筛。

用法：
  python tools\\video_pipeline.py --videos a.mp4 b.mp4 --out "C:\\SRC\\<人物>"
  python tools\\video_pipeline.py --video-dir "G:\\Idol\\Sui\\video" --out "C:\\SRC\\Sui" --dedup
      [--fps 1] [--tonemap hdr10|dv] [--eps 0.35] [--work <帧目录>] [--del-frames]
      [--no-cut] [--allow-nonempty] [-d RetinaFace_10g] [-l insightface-2d106det] [-t whole_face]

人工筛后想对幸存脸（merged\\）再查重一次：
  python FacesetProcessor\\Filter.py -i <out>\\merged --mode repeated --threshold 0.98

帧目录默认 %TEMP%\\video_pipeline_frames\\<序号>_<标签>\\（纯 ASCII，防中文路径坑；
已存在的帧自动跳过=断点续跑）。脸图名：px<短边>__<视频名>__<帧号>_N.png。
"""
import argparse
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = os.path.join(ROOT, "python", "python.exe")
EXTRACTOR = os.path.join(ROOT, "Extractor", "Extractor.py")
FILTER = os.path.join(ROOT, "FacesetProcessor", "Filter.py")
SNAPSHOTS = os.path.join(ROOT, "tools", "make_group_snapshots.py")
BROWSER = os.path.join(ROOT, "tools", "faceid_browser.py")

FFMPEG_CANDIDATES = (r"F:\dfl\_internal\ffmpeg\ffmpeg.exe",)
VIDEO_EXTS = (".mp4", ".mkv", ".mov", ".avi", ".webm", ".m4v", ".wmv", ".flv", ".ts")


def find_ffmpeg():
    for c in FFMPEG_CANDIDATES:
        if os.path.isfile(c):
            return c
    w = shutil.which("ffmpeg")
    if w:
        return w
    sys.exit("[video_pipeline] 找不到 ffmpeg（F:\\dfl\\_internal\\ffmpeg 与 PATH 都没有）")


def safe_label(stem):
    """产出脸图名用（可保留中文，cv2 tofile 无碍）。"""
    s = re.sub(r'[<>:"/\\|?*\x00-\x1f]+', "_", stem).replace("__", "_").strip(" ._")
    return s or "video"


def ascii_label(stem):
    """帧目录名用：纯 ASCII（Extractor 中文路径坑——目录/文件名都不能带中文）。"""
    s = re.sub(r'[^A-Za-z0-9_.-]+', "_", stem).replace("__", "_").strip(" ._")
    return s[:40] or "video"


def frame_command(ffmpeg, video, frames_dir, fps, tonemap_mode):
    """与老 dfl_woni_video_pipeline_v2.frame_command 相同（切帧流程不变）。"""
    if tonemap_mode == "dv":
        vf = (
            "libplacebo=tonemapping=hable:colorspace=bt709:color_trc=bt709:"
            "color_primaries=bt709:format=yuv420p:peak_detect=1,"
            f"fps={fps}"
        )
    else:
        vf = (
            "zscale=transfer=linear,tonemap=hable,"
            "zscale=transfer=bt709:matrix=bt709:primaries=bt709,"
            f"format=yuv420p,fps={fps}"
        )
    cmd = [
        ffmpeg, "-hide_banner", "-loglevel", "warning", "-y", "-i", str(video),
        "-vf", vf, "-start_number", "0", str(frames_dir / "%08d.png"),
    ]
    return cmd


def extract_frames(ffmpeg, video, frames, fps, tonemap):
    """先走 v2 同款 tonemap 链；零产出失败则回退纯 fps 切帧（无色彩空间标签的源）。"""
    r = subprocess.run([str(x) for x in frame_command(ffmpeg, video, frames, fps, tonemap)])
    if r.returncode == 0 or any(frames.glob("*.png")):
        return
    print(f"[video_pipeline] tonemap 链失败（源可能无色彩空间标签），回退纯 fps 切帧: {frames.name}", flush=True)
    fb = [str(ffmpeg), "-hide_banner", "-loglevel", "warning", "-y", "-i", str(video),
          "-vf", f"fps={fps}", "-start_number", "0", str(frames / "%08d.png")]
    r2 = subprocess.run(fb)
    if r2.returncode != 0 and not any(frames.glob("*.png")):
        sys.exit(f"[video_pipeline] 切帧失败（含回退）: {frames.name}")


def run_checked(cmd, cwd=None, fail_msg=""):
    r = subprocess.run([str(x) for x in cmd], cwd=str(cwd) if cwd else None)
    if r.returncode != 0:
        sys.exit(fail_msg + f"（exit={r.returncode}）")
    return r


def collect_faces(stage_out, out, label):
    """stage_out 的 <帧号>_N.ext → <out>/px<短边>__<label>__<帧号>_N.png"""
    import cv2
    import numpy as np

    n = 0
    for f in sorted(os.listdir(stage_out)):
        if not f.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        stem = os.path.splitext(f)[0]
        suffix = ""
        if "_" in stem and stem.rsplit("_", 1)[-1].isdigit():
            stem, suffix = stem.rsplit("_", 1)[0], "_" + stem.rsplit("_", 1)[-1]
        img = cv2.imdecode(np.fromfile(os.path.join(stage_out, f), dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            continue
        px = min(img.shape[0], img.shape[1])
        dst = os.path.join(out, f"px{px:04d}__{label}__{stem}{suffix}.png")
        cv2.imencode(".png", img)[1].tofile(dst)
        n += 1
    return n


def main():
    ap = argparse.ArgumentParser(description="视频流水线（深变版）：切帧→切脸→查重→faceid→人工筛 UI")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--videos", nargs="+", help="视频文件路径（多个）")
    g.add_argument("--video-dir", help="视频文件夹（枚举常见视频扩展名）")
    ap.add_argument("--out", required=True, help="脸图输出目录（如 C:\\SRC\\<人物>）")
    ap.add_argument("--fps", type=int, default=1)
    ap.add_argument("--tonemap", choices=("hdr10", "dv"), default="hdr10")
    ap.add_argument("--work", default=None, help="帧目录（默认 %%TEMP%%\\video_pipeline_frames）")
    ap.add_argument("--eps", type=float, default=0.35, help="faceid DBSCAN eps（默认0.35 人工可审）")
    ap.add_argument("--dedup", action="store_true", help="聚类前先 phash 查重（重复脸移 repeated\\）")
    ap.add_argument("--del-frames", action="store_true", help="单个视频切脸成功后删除其帧目录")
    ap.add_argument("--no-cut", action="store_true", help="只切帧不切脸（调试/预热）")
    ap.add_argument("--allow-nonempty", action="store_true", help="输出目录非空时允许继续")
    ap.add_argument("-d", "--detector", default="RetinaFace_10g")
    ap.add_argument("-l", "--landmarker", default="insightface-2d106det")
    ap.add_argument("-t", "--face-type", default="whole_face")
    args = ap.parse_args()

    if args.videos:
        videos = [Path(v) for v in args.videos]
    else:
        vd = Path(args.video_dir)
        if not vd.is_dir():
            sys.exit(f"[video_pipeline] 目录不存在: {vd}")
        videos = sorted(p for p in vd.iterdir() if p.is_file() and p.suffix.lower() in VIDEO_EXTS)
    if not videos:
        sys.exit("[video_pipeline] 没有视频")
    for v in videos:
        if not v.is_file():
            sys.exit(f"[video_pipeline] 视频不存在: {v}")

    out = os.path.abspath(args.out)
    if os.listdir(out) if os.path.isdir(out) else False:
        if not args.allow_nonempty:
            sys.exit("[video_pipeline] 输出目录非空（防混入上一轮）；确认追加请加 --allow-nonempty")
    os.makedirs(out, exist_ok=True)

    ffmpeg = find_ffmpeg()
    work = Path(args.work) if args.work else Path(os.environ["TEMP"]) / "video_pipeline_frames"
    work.mkdir(parents=True, exist_ok=True)
    print(f"[video_pipeline] {len(videos)} 个视频 | fps={args.fps} tonemap={args.tonemap} | 帧目录 {work}")
    print(f"[video_pipeline] 切脸 {args.detector} + {args.landmarker}（--no-cut={args.no_cut}）")

    pending = {}   # 视频序号(1-based) -> ffmpeg Popen（上一轮预启动）
    total_faces = 0

    for idx, video in enumerate(videos, start=1):
        label = safe_label(video.stem)            # 脸图名用（可中文）
        frames = work / f"{idx:02d}_{ascii_label(video.stem)}"   # 帧目录纯 ASCII

        print(f"[video_pipeline] [{idx}/{len(videos)}] 切帧: {video.name} → {frames.name}", flush=True)
        # 1) 等上一轮预启动的切帧；没有则现场切（首视频/断点续跑场景）
        job = pending.pop(idx, None)
        frames.mkdir(parents=True, exist_ok=True)

        if job is not None:
            print(f"[video_pipeline] [{idx}/{len(videos)}] 等待并行切帧完成: {frames.name}", flush=True)
            if job.wait() != 0:
                extract_frames(ffmpeg, video, frames, args.fps, args.tonemap)  # 回退重试
        elif not any(frames.glob("*.png")):
            print(f"[video_pipeline] [{idx}/{len(videos)}] 切帧: {video.name}", flush=True)
            extract_frames(ffmpeg, video, frames, args.fps, args.tonemap)
        n_frames = len(list(frames.glob("*.png")))

        # 2) 预启动下一个视频的切帧（本视频切脸期间后台跑）
        if idx < len(videos):
            nlabel = ascii_label(videos[idx].stem)   # videos[idx] = 下一个（0-based idx 是当前-1）
            nframes = work / f"{idx + 1:02d}_{nlabel}"
            nframes.mkdir(parents=True, exist_ok=True)
            if not any(nframes.glob("*.png")):
                pending[idx + 1] = subprocess.Popen(
                    [str(x) for x in frame_command(ffmpeg, videos[idx], nframes, args.fps, args.tonemap)]
                )
                print(f"[video_pipeline] [并行] 预启动切帧: {nframes.name}", flush=True)

        # 3) 切脸（Extractor 走 GPU；此时下一个视频的 ffmpeg 在后台解码）
        if args.no_cut:
            print(f"[video_pipeline] [{idx}/{len(videos)}] --no-cut 跳过切脸（{n_frames} 帧）")
            continue
        stage_out = Path(os.environ["TEMP"]) / f"video_pipeline_out_{idx:02d}"
        if stage_out.exists():
            shutil.rmtree(stage_out)
        stage_out.mkdir(parents=True)
        cmd = [PY, EXTRACTOR, "-i", str(frames), "-o", str(stage_out),
               "-d", args.detector, "-l", args.landmarker, "-t", args.face_type, "-m", "image"]
        print(f"[video_pipeline] [{idx}/{len(videos)}] 切脸: {n_frames} 帧 → {video.name}", flush=True)
        r = subprocess.run(cmd, cwd=ROOT)
        if r.returncode != 0:
            sys.exit(f"[video_pipeline] Extractor 失败: {video.name}（exit={r.returncode}，帧保留在 {frames}）")
        n = collect_faces(stage_out, out, label)
        total_faces += n
        print(f"[video_pipeline] [{idx}/{len(videos)}] 产出脸图 {n} 张（累计 {total_faces}）", flush=True)
        shutil.rmtree(stage_out, ignore_errors=True)
        if args.del_frames:
            shutil.rmtree(frames, ignore_errors=True)

    # ── 收尾：查重(可选) → faceid → 快照 → 人工筛 UI ──
    if args.dedup:
        print("[video_pipeline] 查重（phash，重复脸移 repeated\\）...", flush=True)
        run_checked([PY, FILTER, "-i", out, "--mode", "repeated"],
                    fail_msg="[video_pipeline] 查重失败")
    print(f"[video_pipeline] faceid 聚类 eps={args.eps} ...", flush=True)
    run_checked([PY, FILTER, "-i", out, "--mode", "faceid", "--eps", str(args.eps)],
                fail_msg="[video_pipeline] faceid 聚类失败")
    print("[video_pipeline] 出组快照 ...", flush=True)
    subprocess.run([PY, SNAPSHOTS, "--work", out, "--cell", "200", "--cols", "8"])
    print(f"[video_pipeline] 出组快照 ...", flush=True)
    subprocess.run([PY, SNAPSHOTS, "--work", out, "--cell", "200", "--cols", "8"])
    # faceid_browser 固定 8104 + 启动自动杀旧实例接管（全局单实例，用户约定）
    print(f"[video_pipeline] 打开人工筛 UI: http://127.0.0.1:8104  root={out}", flush=True)
    # 关键：stdio 全脱钩（DEVNULL+DETACHED），否则上层 shell 会等 browser 的 stdout 关闭→假死
    detach = getattr(subprocess, "DETACHED_PROCESS", 0) | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
    subprocess.Popen([PY, BROWSER, "--root", out], cwd=ROOT,
                     stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                     creationflags=detach)
    print(f"[video_pipeline] 完成：脸图 {total_faces} 张 @ {out}")
    print("[video_pipeline] 人工筛后如需对 merged\\ 再查重：")
    print(f"    python FacesetProcessor\\Filter.py -i {out}\\merged --mode repeated --threshold 0.98")
    return 0


if __name__ == "__main__":
    main()
