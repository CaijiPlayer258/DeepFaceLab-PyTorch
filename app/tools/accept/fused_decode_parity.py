# -*- coding: utf-8 -*-
r"""fused 解码 ↔ 切帧产物 的逐帧逐像素对拍回归（补「只比成片」那个洞）。

为什么要有这一条：fused_export_accept 的 B 段是拿 **成片** 对拍（fused 出的 mp4 vs 分段出的
mp4）。成片过了 libx264 编码，源头 1~4 LSB 的差异会被量化抹平 —— 解码路径换了、像素差了一
点点，那条断言照样绿。所以这里直接对着**切帧阶段的产物** frames/%06d.png 比，一帧不少、一
像素不落。

比的是同一段视频的前 N 帧（默认 30）：
    a) 切帧阶段的产物：frames/%06d.png —— 由**真**的 _stage_extract_frames_ffmpeg 生成
       （和 /api/merge/stage/frames 走的是同一个函数、同一组 ffmpeg 参数）；
    b) fused 的默认解码：_fused_open_frames(..., mode="raw") —— rawvideo 软解。

断言（逐帧）：mean ≤ 0.2 LSB 且 max ≤ 2。另外打印「100% 像素相同的帧数」。

边界素材（这类改动最容易在非常规源上翻车，全部现造）：
    normal   1080x1920 8bit 4:2:0（和线上素材同规格）
    p422     8bit 4:2:2
    p444     8bit 4:4:4
    hevc10   HEVC Main10 4:2:0 10bit      → 切帧 PNG 是 **16bit**（rgb48be）
    hevc12   HEVC Main12 4:2:0 12bit
    gray10   10bit 灰度
    odd      1081x1921 奇数宽高（只能 4:4:4：4:2:0 要求偶数）
    rot90    手机竖拍：960x540 编码 + 90° 显示矩阵（ffmpeg autorotate → 540x960）

跑法（仓库根）：
    runtime\server\Scripts\python.exe app\tools\fused_decode_parity.py
    runtime\server\Scripts\python.exe app\tools\fused_decode_parity.py --frames 60
    runtime\server\Scripts\python.exe app\tools\fused_decode_parity.py --only odd,rot90 --keep
    runtime\server\Scripts\python.exe app\tools\fused_decode_parity.py --only hevc10 --force-bits 8
        （负控：强制把 10bit 源按 8bit 解 → 应该 FAIL，用来证明这条断言抓得住 1 LSB 级的差）

退出码 0 = 全过。
"""
from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
APP = REPO / "app"
PY = REPO / "runtime" / "server" / "Scripts" / "python.exe"
TEMP = Path(os.environ.get("TEMP") or os.environ.get("TMP") or ".")
BOX = TEMP / "dfl_decode_parity"

TOL_MEAN = 0.2          # 逐帧 |diff| 平均（LSB）
TOL_MAX = 2             # 逐帧单像素最大差（LSB）
FRAMES = 30

# 每种素材：怎么造、期望什么
FIXTURES = [
    # key       文件名                               缩放/滤镜      编码器       pix_fmt       crf
    ("normal", "normal_1080x1920_420_8bit.mp4", "scale=1080:1920", "libx264", "yuv420p", "18"),
    ("p422", "h264_422_540x960_8bit.mp4", "scale=540:960", "libx264", "yuv422p", "12"),
    ("p444", "h264_444_540x960_8bit.mp4", "scale=540:960", "libx264", "yuv444p", "12"),
    ("hevc10", "hevc10_540x960_420.mp4", "scale=540:960", "libx265", "yuv420p10le", "16"),
    ("hevc12", "hevc12_540x960_420.mp4", "scale=540:960", "libx265", "yuv420p12le", "16"),
    ("gray10", "hevc10_540x960_gray.mp4", "scale=540:960,format=gray", "libx265", "gray10le", "16"),
    ("odd", "h264_444_1081x1921.mp4", "scale=1081:1921", "libx264", "yuv444p", "12"),
    ("rot90", "rot90_phone_960x540.mp4", None, "libx264", "yuv420p", "12"),
]

RESULT = []


def check(ok, name, detail=""):
    ok = bool(ok)
    RESULT.append((ok, name, str(detail)))
    print(("[PASS] " if ok else "[FAIL] ") + name + ("  · " + str(detail) if detail else ""),
          flush=True)
    return ok


def say(msg):
    print("       " + str(msg), flush=True)


def head(t):
    print("")
    print("=" * 78)
    print(t)
    print("=" * 78)


def ffmpeg_exe():
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


def run_ff(args, what=""):
    cmd = [ffmpeg_exe(), "-y", "-hide_banner", "-loglevel", "error"] + [str(a) for a in args]
    r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if r.returncode != 0:
        raise SystemExit("ffmpeg 失败（%s）：%s" % (what, (r.stderr or "")[-600:]))
    return r


def load_backend():
    """导入后端模块拿真函数（顺序要紧：routes_merge 反过来 import app，先导 server.app）。"""
    sys.path.insert(0, str(APP))
    import server.app as _app                                    # noqa: F401
    from server import routes_merge as M
    return M


def build_fixtures(src: Path, rebuild=False):
    """%TEMP% 里现造边界素材（幂等：文件在就复用，--rebuild 才重造）。"""
    out = {}
    for key, name, vf, codec, pix, crf in FIXTURES:
        dst = BOX / name
        if dst.exists() and not rebuild:
            out[key] = dst
            continue
        if key == "rot90":
            # 编码 960x540 + 90° 显示矩阵：ffmpeg autorotate 之后是 540x960（竖屏）
            coded = BOX / ("_coded_" + name)
            run_ff(["-t", "1.2", "-i", str(src), "-vf", "scale=960:540", "-c:v", codec,
                    "-crf", crf, "-pix_fmt", pix, "-an", str(coded)], "rot90 编码")
            run_ff(["-display_rotation", "90", "-i", str(coded), "-c", "copy", str(dst)],
                   "rot90 显示矩阵")
            coded.unlink(missing_ok=True)
        else:
            run_ff(["-t", "1.2", "-i", str(src), "-vf", vf, "-c:v", codec, "-crf", crf,
                    "-pix_fmt", pix, "-an", "-movflags", "+faststart", str(dst)], name)
        out[key] = dst
        say("造素材 %-46s %6.1f KB" % (name, dst.stat().st_size / 1024.0))
    return out


def png_bit_depth(path: Path) -> int:
    """PNG IHDR 的 bit depth（8 / 16）—— 16 就是 png 编码器替 >8bit 源挑的那条路。"""
    blob = path.read_bytes()[:26]
    return int(blob[24]) if len(blob) >= 25 else 0


def cmp_frame(ref, got):
    import numpy as np
    d = np.abs(ref.astype(np.int16) - got.astype(np.int16))
    return float(d.mean()), int(d.max()), bool(int(d.max()) == 0)


def decode_raw(M, video, size, frames, mode="raw", start_i=1, bits=None):
    """用 fused 的解码器取一帧序列（读完立刻把 ffmpeg 收掉）。"""
    pr, nxt = M._fused_open_frames(video, int(start_i), size=size, mode=mode, bits=bits)
    rows = []
    ms = []
    try:
        for _i in range(int(frames)):
            t = time.perf_counter()
            f = nxt()
            if f is None:
                break
            ms.append(1000.0 * (time.perf_counter() - t))
            rows.append(f)
    finally:
        try:
            pr.kill()
        except Exception:                                       # noqa: BLE001
            pass
        try:
            pr.wait(timeout=10)
        except Exception:                                       # noqa: BLE001
            pass
    return rows, ms


def compare(name, refs, gots, tag=""):
    """逐帧逐像素对拍 → 打印 + 断言。返回 (ok, dict 统计)。"""
    n = min(len(refs), len(gots))
    if n <= 0:
        check(False, "%s %s对拍：一帧都没拿到" % (name, tag), "ref=%d got=%d"
              % (len(refs), len(gots)))
        return False, {}
    shape_bad = [(i + 1, r.shape, g.shape) for i, (r, g) in enumerate(zip(refs, gots))
                 if r.shape != g.shape]
    if shape_bad:
        i, a, b = shape_bad[0]
        check(False, "%s %s对拍：形状不一致" % (name, tag),
              "第 %d 帧 png=%s raw=%s（共 %d 帧不一致）" % (i, a, b, len(shape_bad)))
        return False, {}
    means, maxs, same = [], [], 0
    tot = 0.0
    for r, g in zip(refs, gots):
        m, mx, eq = cmp_frame(r, g)
        means.append(m)
        maxs.append(mx)
        same += 1 if eq else 0
        tot += m * r.size
    mean_all = tot / (n * refs[0].size)
    worst = max(means)
    mx = max(maxs)
    pct = 100.0 * same / n
    ok = (worst <= TOL_MEAN) and (mx <= TOL_MAX)
    check(ok, "%s %s：mean ≤ %.1f 且 max ≤ %d LSB" % (name, tag, TOL_MEAN, TOL_MAX),
          "%d 帧 · 全片 mean=%.4f · 最差单帧 mean=%.4f · max=%d · 逐位相同 %d/%d（%.1f%%）"
          % (n, mean_all, worst, mx, same, n, pct))
    if len(refs) != len(gots):
        check(False, "%s %s帧数对不上" % (name, tag), "png %d vs 解码 %d"
              % (len(refs), len(gots)))
        ok = False
    return ok, {"frames": n, "mean": mean_all, "worst": worst, "max": mx,
                "same": same, "same_pct": pct}


def run_fixture(M, key, video, frames, keep, force_bits=None):
    """一种素材：造切帧产物 → raw 解码 → png 解码（对照）→ 逐帧对拍。"""
    import cv2

    probe = M._vid_probe(video)
    fdir = BOX / ("frames_" + key)
    shutil.rmtree(fdir, ignore_errors=True)
    fdir.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    got = M._stage_extract_frames_ffmpeg({"video": str(video), "total": 0}, fdir)
    t_stage = time.perf_counter() - t0
    pngs = sorted(fdir.glob("*.png"))
    head("%s · %s · probe %dx%d · %d 帧 · fps %.2f"
         % (key, video.name, probe["w"], probe["h"], probe["frames"], probe["fps"]))
    if not check(got is not None and len(pngs) >= min(frames, probe["frames"] or frames),
                 "%s 切帧产物齐了（真 _stage_extract_frames_ffmpeg）" % key,
                 "%s 帧 · %.2fs · %s" % (len(pngs), t_stage, pngs[0].name if pngs else "-")):
        return {}
    depth = png_bit_depth(pngs[0])
    say("切帧 PNG：%dx%d bit_depth=%d（>8 说明 png 编码器挑了 16bit，raw 那边得用 bgr48le）"
        % (cv2.imread(str(pngs[0])).shape[1], cv2.imread(str(pngs[0])).shape[0], depth))

    refs = [cv2.imread(str(p), cv2.IMREAD_COLOR) for p in pngs[:frames]]
    if force_bits:
        say("!! --force-bits=%d：强制把源当 %dbit 解（负控，复现旧行为）" % (force_bits, force_bits))
    raw_rows, raw_ms = decode_raw(M, video, (probe["w"], probe["h"]), len(refs),
                                  bits=force_bits)
    ok, st = compare(key, refs, raw_rows)
    st["png_depth"] = depth
    st["ok"] = ok

    # 对照 1：DFL_FUSED_DECODE=png 那条可选路径 —— 必须也是 100%（它才是"同一条编解码路径"）
    png_rows, png_ms = decode_raw(M, video, (probe["w"], probe["h"]), len(refs), mode="png")
    _ok_png, st_png = compare(key, refs, png_rows, "· png 模式")
    st["png_ok"] = _ok_png
    # 解码耗时：都去掉前 5 帧（ffmpeg 起进程 + 首个关键帧不具代表性）
    r_ms = sum(raw_ms[5:]) / max(1, len(raw_ms[5:]))
    p_ms = sum(png_ms[5:]) / max(1, len(png_ms[5:]))
    say("解码耗时：raw %.2f ms/帧 vs png %.2f ms/帧（去掉前 5 帧）· %.2f×"
        % (r_ms, p_ms, (p_ms / r_ms) if r_ms else 0.0))
    st["raw_ms"], st["png_ms"] = r_ms, p_ms

    # 对照 2：跳帧路径（start_i > 1 用 select 滤镜）—— 第 k 帧必须还是第 k 帧
    if key == "normal" and len(pngs) > 12:
        off = 11
        rows, _ms = decode_raw(M, video, (probe["w"], probe["h"]), 8, start_i=off)
        _ok_off, st_off = compare(key, [cv2.imread(str(p), cv2.IMREAD_COLOR)
                                        for p in pngs[off - 1:off - 1 + len(rows)]],
                                  rows, "· start_i=%d 跳帧" % off)
        st["offset_ok"] = _ok_off
    if not keep:
        shutil.rmtree(fdir, ignore_errors=True)
    return st


def main(argv=None):
    ap = argparse.ArgumentParser(description="fused 解码 ↔ 切帧产物 逐像素回归")
    ap.add_argument("--frames", type=int, default=FRAMES, help="对拍前 N 帧（默认 30）")
    ap.add_argument("--only", default="", help="只跑这些素材（逗号分隔，见 FIXTURES 的 key）")
    ap.add_argument("--src", default="", help="素材来源视频（默认 workspace 里第一个 mp4）")
    ap.add_argument("--keep", action="store_true", help="保留切帧产物（排障用）")
    ap.add_argument("--rebuild", action="store_true", help="重造边界素材")
    ap.add_argument("--force-bits", type=int, default=0,
                    help="负控用：强制源位深（8 = 让 >8bit 源也走 bgr24，应当 FAIL）")
    args = ap.parse_args(argv)
    want = [k.strip() for k in str(args.only).split(",") if k.strip()]
    t_all = time.time()

    BOX.mkdir(parents=True, exist_ok=True)
    src = Path(args.src) if args.src else sorted((REPO / "workspace").glob("*.mp4"))[0]
    if not src.is_file():
        raise SystemExit("找不到来源视频：%s" % src)
    print("来源：%s" % src)
    M = load_backend()
    print("后端模块：%s（_fused_open_frames / _stage_extract_frames_ffmpeg 都是真的那份）"
          % Path(M.__file__).name)

    fixtures = build_fixtures(src, rebuild=args.rebuild)
    stats = {}
    for key, _name, _vf, _codec, _pix, _crf in FIXTURES:
        if want and key not in want:
            continue
        stats[key] = run_fixture(M, key, fixtures[key], int(args.frames), bool(args.keep),
                                 force_bits=(int(args.force_bits) or None))

    head("汇总")
    print("%-8s %-8s %-6s %9s %5s %12s %9s %9s %s"
          % ("素材", "PNG位深", "帧数", "mean", "max", "逐位相同", "raw", "png", "结论"))
    for key, st in stats.items():
        if not st:
            print("%-8s %s" % (key, "没跑成"))
            continue
        print("%-8s %-8d %-6d %9.4f %5d %7d/%-4d %7.2fms %7.2fms %s"
              % (key, st["png_depth"], st["frames"], st["mean"], st["max"], st["same"],
                 st["frames"], st.get("raw_ms") or 0.0, st.get("png_ms") or 0.0,
                 "PASS" if st["ok"] else "FAIL"))
    fails = [r for r in RESULT if not r[0]]
    print("")
    print("PASS %d · FAIL %d · 用时 %.1fs · 沙箱 %s"
          % (len(RESULT) - len(fails), len(fails), time.time() - t_all, BOX))
    for _ok, name, detail in fails:
        print("  FAIL %s · %s" % (name, detail))
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
