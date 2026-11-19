# -*- coding: utf-8 -*-
"""多模型导出走分趟（docs/37 第 3 步接进 API）。

    Pass0 扫描 → 每个模型一趟（显存里一次只有一份）→ 合流写 PNG → 生产 ffmpeg 编码

为什么要单独一条路：旧路径让两个 dfm 会话同时常驻，8GB 卡上 warm 阶段会卡死
（线程栈见 workspace/*.stacks.txt）。这里改成顺序装、顺序跑，显存与模型数解耦。
"""
from __future__ import annotations

import shutil
import tempfile
import time
from pathlib import Path


def run(cfg, log=print, progress=None, cancelled=None) -> dict:
    """跑完整一单。cfg 需要：video / out / models(name+dfm) / config / start / end。

    progress(pct, msg, frames_done=None, frames_total=None) 由调用方给（接到 _stage_progress）。
    返回 {out, frames, elapsed_s}；异常原样抛给调用方（它负责把 job 标失败）。
    """
    from .merge_final import merge
    from .pass0 import scan
    from .pass_swap import run as pass_run
    from .. import _merge_runner as MR

    t0 = time.time()
    video = Path(str(cfg["video"]))
    crf = int(cfg.get("crf") or 17)
    # 编码器：优先用请求里给的那个（导出设置里的 h264_nvenc）；拿不到就 libx264。
    # 之前写死 libx264 —— 用户的设置被静默忽略，等于换了个编码器还不知道。
    _enc = str(cfg.get("encoder") or "").strip()
    if "nvenc" in _enc:
        _preset = "p5"
    elif _enc.startswith("libx"):
        _preset = "medium"
    else:
        _enc, _preset = "libx264", "medium"
    fps = 0.0
    out = Path(str(cfg["out"]))
    models = [m for m in (cfg.get("single_models") or []) if str(m.get("dfm") or "").strip()]
    if not models:
        raise RuntimeError("分趟导出需要至少一个模型")

    def _p(pct, msg, done=None, total=None, meta=None):
        if progress:
            progress(int(pct), msg, done, total, meta or {})
        log(msg)

    # ---- ① 扫描：分母在这一趟里长出来 ----
    _p(2, "分趟导出：扫描人脸分配表 …", meta={"phase": "scan", "pass_i": 0,
                                          "pass_n": len(models), "pool": 0})
    idx = scan(video, models=models, plan_payload=cfg.get("plan_payload"),
               start=int(cfg.get("start") or 0),
               end=(int(cfg["end"]) if cfg.get("end") else None), log=log)
    if not idx.get("swap_faces"):
        raise RuntimeError("这一单没有任何待换的脸（检测 %d 张，一张都没分配到模型）"
                           % int(idx.get("detected") or 0))
    total_faces = int(idx["swap_faces"])

    # ---- ② 每个模型一趟 ----
    metas = []
    done_faces = 0
    for k, m in enumerate(models, start=1):
        if cancelled and cancelled():
            raise RuntimeError("已取消")
        nm = str(m.get("name") or "")
        _p(2 + int(40 * (k - 1) / max(1, len(models))),
           "分趟导出：第 %d/%d 趟 · %s" % (k, len(models), nm),
           meta={"phase": "pass", "pass_i": int(k), "pass_n": len(models),
                 "pass_label": nm, "faces_done": int(done_faces),
                 "faces_total": int(total_faces), "pool": int(done_faces)})
        pool_idx = video.parent / (video.stem + "_cache") / "pool" / "index.json"
        # 一律算 XSeg 遮罩：合成要用哪张遮罩由 mask_mode 决定，原料里两张都备着，
        # 合流时才能和导出的遮罩口径一致（只算 dfm 遮罩时会差 34 LSB）。
        r = pass_run(video, pool_idx, nm, str(m["dfm"]), log=log, want_xseg=True)
        done_faces += int(r.get("faces") or 0)
        if r.get("faces"):
            metas.append(r["path"])
    if not metas:
        raise RuntimeError("没有任何模型产出脸原料")

    # ---- ③ 合流：写 PNG，交给生产编码器 ----
    _p(45, "分趟导出：闸门打开，合流（每帧只混合一次）…",
       meta={"phase": "merge", "pass_i": len(models), "pass_n": len(models),
             "pool": int(done_faces), "faces_total": int(total_faces)})
    png_dir = Path(tempfile.mkdtemp(prefix="dfl_passes_"))
    try:
        # 和快速导出同一套"有效设置"：knobs 会把 mode/ct/erode/blur 归一化，
        # 并可能改写模块级开关（合成后端、色彩迁移实现）。少了这一步，两边合成的
        # 默认口径不同 —— 实测差 32 LSB，且差异全在脸框内。
        from . import knobs as _knobs
        _kn = _knobs.refine_compose(_knobs.resolve(dict(cfg.get("settings") or {})),
                                    str((cfg.get("config") or {}).get("mode") or ""),
                                    fused=True)
        try:
            for _t in _knobs.apply(_kn):
                log(str(_t))
        except Exception:                                  # noqa: BLE001
            pass
        res = merge(video, metas, cfg=dict(cfg.get("config") or {}), png_dir=str(png_dir),
                    on_frame=lambda n: _p(45 + int(45 * min(1.0, n / max(1, idx.get("frames") or n))),
                                          "分趟导出：合流 %d 帧" % n, n, int(idx.get("frames") or 0)),
                    log=log)
        _p(92, "分趟导出：编码 …")
        _ff = MR._resolve_ffmpeg("", "", Path(str(cfg.get("dfl_root") or ".")))
        ffmpeg = str(_ff[0] if isinstance(_ff, (tuple, list)) else _ff)
        part = Path(str(out)).with_suffix(".part.mp4")
        # 编码命令直接在这里拼（不依赖 _merge_runner 内部那几个私有函数：
        # 它们的入参口径不稳，接的时候已经被 % 格式化坑过两次）。
        _fps = float(res.get("fps") or fps or 25.0)
        cmd = [ffmpeg, "-y", "-hide_banner", "-loglevel", "error", "-nostats",
               "-framerate", "%.6g" % _fps, "-start_number", "1",
               "-i", str(png_dir / "%08d.png"),
               "-i", str(video),
               "-map", "0:v:0", "-map", "1:a:0?",
               "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
               "-c:v", _enc, "-crf", str(int(crf)), "-preset", _preset,
               "-pix_fmt", "yuv420p", "-c:a", "copy", str(part)]
        # 直接起 ffmpeg（不走 _merge_runner._run_encode：它的进度回调里有 % 格式化的老坑）
        import subprocess as _sp
        _pr = _sp.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
        if _pr.returncode:
            raise RuntimeError("编码失败（ffmpeg %s）：%s"
                               % (_pr.returncode, (_pr.stderr or "")[-400:]))
        shutil.move(str(part), str(out))
    finally:
        shutil.rmtree(str(png_dir), ignore_errors=True)

    _p(100, "分趟导出完成：%d 帧" % int(res.get("frames") or 0))
    return {"out": str(out), "frames": int(res.get("frames") or 0),
            "faces": int(done_faces), "elapsed_s": round(time.time() - t0, 1)}
