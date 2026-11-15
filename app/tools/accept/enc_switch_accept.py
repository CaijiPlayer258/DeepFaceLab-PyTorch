# -*- coding: utf-8 -*-
r"""导出编码器开关（DFL_FUSED_ENCODER）验收：默认逐字节不变 + NVENC 两档 + 4K 告警 + 预检。

验的是「加了开关，但默认那条命令一个字节都没变」这件事，外加换过去之后该成立的四条性质：

  A  默认（不设 DFL_FUSED_ENCODER）：_fused_pipe_cmd / _vid_cmd 的 argv 与**改动前录制**的
     金样本逐项相同，且 x264 两轮 md5 相同。
  B  nvenc-h264：两轮 md5 相同（确定性），且 ≠ x264 —— 换编码器就是换文件，这不是 bug。
  C  两条出口（raw 管道 / PNG 落盘）拿到**同一个**编码器：x264 与 nvenc 各验一次 md5 相同。
     这条不变量以前靠"两边都硬编码 libx264"成立，现在靠"两边都调 encoder_sel"，所以必须测。
  D  4K（3840x2160）+ auto：**不自动切**（argv 里还是 libx264），但必须喊一条 [eff-warn]
     并进 status.eff_warnings[]（进度页横幅读它）；同一单只喊一条，1080p 不吵人。
  E  显式 nvenc：argv 带上 -preset p5 -tune hq -rc vbr -cq CQ -b:v 0；hevc 再加 -tag:v hvc1；
     DFL_FUSED_ENCODER_CQ 真的生效。
  F  没有 NVENC 的机器：预检失败 → **起 ffmpeg 之前**抛 _BadInput（<1s），不留半截 mp4。

素材自己造（lavfi，不依赖仓库里的样片），一次最多 80 帧 1080x1920 / 8 帧 3840x2160。

跑法（仓库根）：
    runtime\server\Scripts\python.exe app\tools\accept\enc_switch_accept.py
    ... --keep          留着临时目录
退出码 0 = 全过；2 = 有 FAIL。
"""
from __future__ import annotations

import argparse
import hashlib
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
APP = REPO / "app"
for _p in (str(APP), str(REPO)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

TEMP = Path(os.environ.get("TEMP") or os.environ.get("TMP") or ".")
BOX = TEMP / "dfl_enc_switch"

RESULTS = []


def check(ok, name, detail=""):
    ok = bool(ok)
    RESULTS.append((ok, name, str(detail)))
    print(("[PASS] " if ok else "[FAIL] ") + name + ("  · " + str(detail) if detail else ""),
          flush=True)
    return ok


# ============================================================
# 金样本：**改动前**（4.1 那条硬编码 libx264 crf16 的版本）录下来的 argv
# ------------------------------------------------------------
# 1080x1920 @30 · audio_ss=0 · audio_t=2.666667 · limit=80，路径抹成 <EXE>/<VIDEO>/<TMP>。
# 这几个列表就是"默认路径一个字节都不许变"的定义；改它们等于改基线，先问清楚再说。
# ============================================================
GOLDEN_FUSED = ["<EXE>", "-y", "-hide_banner", "-loglevel", "error", "-nostats", "-nostdin",
                "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", "1080x1920", "-framerate", "30",
                "-i", "pipe:0", "-ss", "0.000000", "-t", "2.666667", "-i", "<VIDEO>",
                "-map", "0:v:0", "-map", "1:a?",
                "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2,format=rgb24,scale=out_range=tv",
                "-c:v", "libx264", "-crf", "16", "-pix_fmt", "yuv420p", "-c:a", "copy",
                "-movflags", "+faststart", "-frames:v", "80", "<TMP>/o.mp4"]

GOLDEN_VID = ["<EXE>", "-y", "-hide_banner", "-loglevel", "error", "-nostats", "-nostdin",
              "-framerate", "30", "-start_number", "1", "-i", "<TMP>/%06d.png",
              "-ss", "0.000000", "-t", "2.666667", "-i", "<VIDEO>",
              "-map", "0:v:0", "-map", "1:a?",
              "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2,scale=out_range=tv",
              "-c:v", "libx264", "-crf", "16", "-pix_fmt", "yuv420p", "-c:a", "copy",
              "-frames:v", "80", "-movflags", "+faststart", "-progress", "pipe:1",
              "<TMP>/o.mp4"]


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for c in iter(lambda: f.read(1 << 22), b""):
            h.update(c)
    return h.hexdigest()


def ffmpeg():
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


def make_raw(dest, w, h, n):
    """lavfi 造 n 帧 bgr24 raw（确定性；不碰仓库里的样片）。"""
    if dest.is_file() and dest.stat().st_size == w * h * 3 * n:
        return
    cmd = [ffmpeg(), "-y", "-hide_banner", "-loglevel", "error",
           "-f", "lavfi", "-i", "testsrc2=size=%dx%d:rate=30" % (w, h),
           "-frames:v", str(n), "-f", "rawvideo", "-pix_fmt", "bgr24", str(dest)]
    subprocess.run(cmd, check=True)


def make_audio(dest):
    """-i 的那条"源视频"：只要有音轨就行（-map 1:a? 只看音频）。"""
    if dest.is_file() and dest.stat().st_size > 0:
        return
    cmd = [ffmpeg(), "-y", "-hide_banner", "-loglevel", "error",
           "-f", "lavfi", "-i", "sine=frequency=440:duration=4", "-c:a", "aac", str(dest)]
    subprocess.run(cmd, check=True)


def run_pipe(cmd, frames):
    """按生产口径：单写者顺序喂 raw 帧。返回 (rc, stderr 尾巴)。"""
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
                         stderr=subprocess.PIPE)
    try:
        for f in frames:
            p.stdin.write(f)
    finally:
        try:
            p.stdin.close()
        except Exception:                                     # noqa: BLE001
            pass
    err = p.stderr.read().decode("utf-8", "replace")[-400:]
    return p.wait(), err


def run_plain(cmd):
    q = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    return q.returncode, q.stderr.decode("utf-8", "replace")[-400:]


def norm(argv, exe, video, part):
    out = []
    for x in argv:
        x = str(x)
        if x == str(exe):
            x = "<EXE>"
        x = x.replace(str(video), "<VIDEO>")
        x = x.replace(str(part), "<TMP>/o.mp4")
        x = re.sub(r"%06d\.png$", "%06d.png", x)
        x = x.replace(str(BOX), "<TMP>")
        out.append(x.replace("\\", "/"))       # 分隔符不影响命令语义，金样本统一用 /
    return out


def diff(name, got, want):
    if got == want:
        return check(True, name, "%d 项逐项相同" % len(want))
    for i, (a, b) in enumerate(zip(got, want)):
        if a != b:
            return check(False, name, "第 %d 项：got %r / want %r" % (i, a, b))
    return check(False, name, "长度 %d vs %d" % (len(got), len(want)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--keep", action="store_true")
    args = ap.parse_args()

    BOX.mkdir(parents=True, exist_ok=True)
    print("临时目录：%s" % BOX, flush=True)

    # ---- 环境先在干净状态上验"默认路径"（用户机器上真设了这个变量，这里是唯一能看见差异的地方）
    for k in ("DFL_FUSED_ENCODER", "DFL_FUSED_ENCODER_CQ", "DFL_FUSED_ENCODER_AUTO4K"):
        os.environ.pop(k, None)

    from server.backend_merge import pipeline_fused as PF
    from server._media import encoder as ENC
    from server._common import encoder_sel as SEL

    EXE = ffmpeg()
    W, H, N, FPS = 1080, 1920, 80, 30.0
    raw = BOX / "f1080x1920.raw"
    VID = BOX / "audio.m4a"
    make_raw(raw, W, H, N)
    make_audio(VID)
    blob = raw.read_bytes()
    FR = [blob[i * W * H * 3:(i + 1) * W * H * 3] for i in range(N)]

    # ================= A · 默认路径一个字节都不许变 =================
    print("\n== A 默认路径（DFL_FUSED_ENCODER 不设）==", flush=True)
    part = BOX / "o.mp4"
    cmd_f = [str(x) for x in PF._fused_pipe_cmd(part, VID, FPS, W, H,
                                                audio_ss=0.0, audio_t=N / FPS, limit=N)]
    cmd_v = [str(x) for x in ENC._vid_cmd(str(BOX / "%06d.png"), part, VID, FPS, img_start=1,
                                          audio_ss=0.0, audio_t=N / FPS, limit=N)]
    diff("A1 raw 管道那条命令与改动前金样本逐项相同", norm(cmd_f, EXE, VID, part), GOLDEN_FUSED)
    diff("A2 PNG 落盘那条命令与改动前金样本逐项相同", norm(cmd_v, EXE, VID, part), GOLDEN_VID)

    m = {}
    for rnd in (1, 2):
        out = BOX / ("a_x264_r%d.mp4" % rnd)
        c = [str(x) for x in PF._fused_pipe_cmd(out, VID, FPS, W, H,
                                                audio_ss=0.0, audio_t=N / FPS, limit=N)]
        rc, err = run_pipe(c, FR)
        check(rc == 0, "A3.%d x264 出片 rc=0" % rnd, err[:200])
        m["x264_r%d" % rnd] = md5(out) if out.is_file() else "MISSING"
    check(m["x264_r1"] == m["x264_r2"], "A4 x264 两轮 md5 相同（编码确定性）",
          "%s / %s" % (m["x264_r1"][:12], m["x264_r2"][:12]))

    # ================= B · nvenc-h264 =================
    print("\n== B nvenc-h264 ==", flush=True)
    os.environ["DFL_FUSED_ENCODER"] = "nvenc-h264"
    SEL._PROBE.clear()
    for rnd in (1, 2):
        out = BOX / ("b_nv_r%d.mp4" % rnd)
        c = [str(x) for x in PF._fused_pipe_cmd(out, VID, FPS, W, H,
                                                audio_ss=0.0, audio_t=N / FPS, limit=N)]
        rc, err = run_pipe(c, FR)
        check(rc == 0, "B1.%d nvenc 出片 rc=0" % rnd, err[:200])
        m["nv_r%d" % rnd] = md5(out) if out.is_file() else "MISSING"
    check(m["nv_r1"] == m["nv_r2"], "B2 nvenc 两轮 md5 相同（确定性）",
          "%s / %s" % (m["nv_r1"][:12], m["nv_r2"][:12]))
    check(m["nv_r1"] not in (m["x264_r1"], "MISSING"),
          "B3 nvenc ≠ x264（换编码器 = 换文件，这是纪律不是 bug）",
          "%s vs %s" % (m["nv_r1"][:12], m["x264_r1"][:12]))

    # ================= C · 两条出口同一个编码器 =================
    print("\n== C 两条出口（raw 管道 / PNG 落盘）==", flush=True)
    png = BOX / "png"
    png.mkdir(exist_ok=True)
    import cv2
    import numpy as np
    for i, f in enumerate(FR):
        if not (png / ("%06d.png" % (i + 1))).is_file():
            cv2.imwrite(str(png / ("%06d.png" % (i + 1))), np.frombuffer(f, np.uint8).reshape(H, W, 3))
    for tag, env in (("x264", "x264"), ("nvenc", "nvenc-h264")):
        os.environ["DFL_FUSED_ENCODER"] = env
        SEL._PROBE.clear()
        o1 = BOX / ("c_%s_raw.mp4" % tag)
        o2 = BOX / ("c_%s_png.mp4" % tag)
        rc1, e1 = run_pipe([str(x) for x in PF._fused_pipe_cmd(o1, VID, FPS, W, H,
                                                               audio_ss=0.0, audio_t=N / FPS,
                                                               limit=N)], FR)
        rc2, e2 = run_plain([str(x) for x in ENC._vid_cmd(str(png / "%06d.png"), o2, VID, FPS,
                                                          img_start=1, audio_ss=0.0,
                                                          audio_t=N / FPS, limit=N)])
        a = md5(o1) if o1.is_file() else "MISSING"
        b = md5(o2) if o2.is_file() else "MISSING"
        check(a == b and a != "MISSING", "C.%s 两条路成片 md5 相同" % tag,
              "%s / %s%s" % (a[:12], b[:12], ("  " + (e1 or e2)[:120]) if (rc1 or rc2) else ""))

    # ================= D · 4K auto：只告警、不自动切 =================
    print("\n== D 4K + auto ==", flush=True)
    for k in ("DFL_FUSED_ENCODER", "DFL_FUSED_ENCODER_AUTO4K"):
        os.environ.pop(k, None)
    c4k = [str(x) for x in PF._fused_pipe_cmd(BOX / "o4k.mp4", VID, FPS, 3840, 2160,
                                              audio_ss=0.0, audio_t=8 / FPS, limit=8)]
    check(SEL.effective_mode(3840, 2160) == "x264", "D1 4K 下 auto 不自动切（还是 x264）",
          SEL.effective_mode(3840, 2160))
    check("-c:v" in c4k and c4k[c4k.index("-c:v") + 1] == "libx264",
          "D2 4K 的 argv 里就是 libx264", " ".join(c4k[c4k.index("-c:v"):][:3]))
    txt = SEL.warn_text(3840, 2160)
    check(bool(txt) and "DFL_FUSED_ENCODER=nvenc-h264" in txt, "D3 4K 有那条人话告警", txt[:110])
    check(SEL.warn_text(1920, 1080) == "", "D4 1080p 不告警（别吵人）", SEL.warn_text(1920, 1080))
    before = len(PF._STAGE_JOB.get("eff_warnings") or [])
    PF._warn_encoder_4k({"w": 3840, "h": 2160})
    PF._warn_encoder_4k({"w": 3840, "h": 2160})
    after = PF._STAGE_JOB.get("eff_warnings") or []
    check(len(after) == before + 1 and after[-1].get("code") == "enc-4k",
          "D5 告警进 status.eff_warnings[]（code=enc-4k），重复调用只记一条",
          "条数 %d → %d" % (before, len(after)))
    check(any("[eff-warn]" in str(x) or "4K 素材" in str(x)
              for x in (PF._STAGE_JOB.get("warnings") or [])),
          "D6 warnings[] 里也有纯文本（横幅关掉也还能翻日志）",
          str((PF._STAGE_JOB.get("warnings") or [])[-1:])[:120])

    # ================= E · 显式 nvenc 的参数 =================
    print("\n== E 显式 nvenc 的参数 ==", flush=True)
    want_h264 = ["-c:v", "h264_nvenc", "-preset", "p5", "-tune", "hq", "-rc", "vbr",
                 "-cq", "21", "-b:v", "0"]
    want_hevc = ["-c:v", "hevc_nvenc", "-preset", "p5", "-tune", "hq", "-rc", "vbr",
                 "-cq", "21", "-b:v", "0", "-tag:v", "hvc1"]
    os.environ["DFL_FUSED_ENCODER"] = "nvenc-h264"
    os.environ.pop("DFL_FUSED_ENCODER_CQ", None)
    SEL._PROBE.clear()
    got = SEL.encoder_args(1920, 1080)
    check(got == want_h264, "E1 nvenc-h264 的参数段（CQ 默认 21）", " ".join(got))
    os.environ["DFL_FUSED_ENCODER"] = "nvenc-hevc"
    SEL._PROBE.clear()
    got = SEL.encoder_args(1920, 1080)
    check(got == want_hevc, "E2 nvenc-hevc 多一个 -tag:v hvc1（Apple 只认 hvc1）", " ".join(got))
    os.environ["DFL_FUSED_ENCODER_CQ"] = "23"
    SEL._PROBE.clear()
    got = SEL.encoder_args(1920, 1080)
    check("-cq" in got and got[got.index("-cq") + 1] == "23",
          "E3 DFL_FUSED_ENCODER_CQ 真的生效", " ".join(got))
    os.environ.pop("DFL_FUSED_ENCODER_CQ", None)

    # ================= F · 没有 NVENC 的机器：起 ffmpeg 之前就报错 =================
    print("\n== F 无 NVENC（注入预检失败）==", flush=True)
    SEL._PROBE.clear()
    SEL._PROBE["h264_nvenc"] = (False, "simulated: no capable devices found")
    os.environ["DFL_FUSED_ENCODER"] = "nvenc-h264"
    half = BOX / "f_half.mp4"
    if half.is_file():
        half.unlink()
    t0 = time.perf_counter()
    raised = ""
    try:
        PF._fused_pipe_cmd(half, VID, FPS, W, H, audio_ss=0.0, audio_t=N / FPS, limit=N)
    except Exception as e:                                    # noqa: BLE001
        raised = "%s: %s" % (type(e).__name__, e)
    dt = time.perf_counter() - t0
    check(bool(raised) and dt < 1.0, "F1 预检失败 → 构造命令阶段就抛（<1s，不等人跑完）",
          "%.0f ms · %s" % (dt * 1000, raised[:140]))
    check(not half.is_file(), "F2 没留下半截 mp4", str(half))
    SEL._PROBE.clear()

    # ================= G · 4K 真跑一遍 nvenc（尺寸/参数在 4K 上也成立）=================
    print("\n== G 4K 真跑 ==", flush=True)
    raw4k = BOX / "f3840x2160.raw"
    make_raw(raw4k, 3840, 2160, 8)
    b4 = raw4k.read_bytes()
    FR4 = [b4[i * 3840 * 2160 * 3:(i + 1) * 3840 * 2160 * 3] for i in range(8)]
    os.environ["DFL_FUSED_ENCODER"] = "nvenc-h264"
    SEL._PROBE.clear()
    o4 = BOX / "g_4k_nvenc.mp4"
    rc, err = run_pipe([str(x) for x in PF._fused_pipe_cmd(o4, VID, FPS, 3840, 2160,
                                                           audio_ss=0.0, audio_t=8 / FPS,
                                                           limit=8)], FR4)
    check(rc == 0 and o4.is_file() and o4.stat().st_size > 0,
          "G1 4K（3840x2160）8 帧 nvenc 出片", "%s · %d B · %s"
          % (rc, o4.stat().st_size if o4.is_file() else -1, err[:140]))

    for k in ("DFL_FUSED_ENCODER", "DFL_FUSED_ENCODER_CQ", "DFL_FUSED_ENCODER_AUTO4K"):
        os.environ.pop(k, None)

    bad = [r for r in RESULTS if not r[0]]
    print("\n==== %d/%d 通过 ====" % (len(RESULTS) - len(bad), len(RESULTS)), flush=True)
    if bad:
        for _, n, d in bad:
            print("  FAIL %s  %s" % (n, d), flush=True)
    if not args.keep:
        shutil.rmtree(BOX, ignore_errors=True)
    return 0 if not bad else 2


if __name__ == "__main__":
    sys.exit(main())
