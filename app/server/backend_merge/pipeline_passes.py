# -*- coding: utf-8 -*-
"""多模型导出走分趟（docs/37）：Pass0 扫描 → 每个模型一趟 → 闸门 → 合流 → 编码。

为什么单独一条路：旧路径让每个 dfm 各持一份会话（每 lane 再乘一份），8GB 卡上 warm 会卡死。
分趟是「一趟只装一个模型」，显存与模型数解耦。

**跑哪些模型以分配表为准**（index.json 里实际出现的 model），不是 body.models：
库判定可能把脸分给 body 里没有的模型；按 body 循环会漏跑，报「没有任何模型产出脸原料」。
"""
from __future__ import annotations

import shutil
import tempfile
import time
from pathlib import Path


def _abs_dfm(x, log=print) -> str:
    """库里的 dfm 字段存的是**名字**（阿証…-416），不是路径 —— 直接丢给 _dfm_session
    会 NO_SUCHFILE。这里解析成 workspace/model/<name>[.dfm]，并返回 "" 表示找不到。"""
    p = Path(str(x or "").strip())
    if p.is_file():
        return str(p)
    try:
        from .._common.paths import WORKSPACE
        for c in (Path(str(WORKSPACE)) / "model" / p.name,
                  Path(str(WORKSPACE)) / "model" / (p.name + ".dfm")):
            if c.is_file():
                return str(c)
    except Exception as e:                                     # noqa: BLE001
        from .loud import note
        note("passes 解析 dfm 路径", e, "库里的模型名无法变成路径", log)
    return ""


def run(cfg, log=print, progress=None, cancelled=None) -> dict:
    """跑完整一单。cfg: video / out / models / config / start / end / crf / encoder。"""
    from .merge_final import merge
    from .pass0 import scan
    from .pass_swap import run as pass_run
    from .loud import note, need
    from .. import _merge_runner as MR

    t0 = time.time()
    video = Path(str(cfg["video"]))
    out = Path(str(cfg["out"]))
    crf = int(cfg.get("crf") or 17)
    need(cfg, ["video", "out", "config"], "passes.run")

    enc = str(cfg.get("encoder") or "").strip()
    if "nvenc" in enc:
        preset = "p5"
    elif enc.startswith("libx"):
        preset = "medium"
    else:
        enc, preset = "libx264", "medium"

    def _p(pct, msg, done=None, total=None, meta=None):
        if progress:
            progress(int(pct), msg, done, total, meta or {})
        log(msg)

    models = [m for m in (cfg.get("single_models") or []) if m.get("dfm") or m.get("name")]
    if not models:
        raise RuntimeError("passes: cfg.single_models 为空（这一单没有模型）")

    # ① 扫描
    _mm = int((cfg.get("config") or {}).get("mask_mode", 6) or 0)
    _mask_on = _mm >= 6                     # 6/7/8/9 才走遮罩网络；<6 用 dfm 遮罩
    _p(2, "分趟导出：扫描人脸分配表 …",
       meta={"phase": "scan", "pass_i": 0, "pass_n": len(models), "pool": 0,
             "mask_on": bool(_mask_on), "frozen": False})
    idx = scan(video, models=models, plan_payload=cfg.get("plan_payload"),
               start=int(cfg.get("start") or 0),
               end=(int(cfg["end"]) if cfg.get("end") else None), log=log)
    total_faces = int(idx.get("swap_faces") or 0)
    if not total_faces:
        raise RuntimeError("这一单没有任何待换的脸（检测 %d 张，一张都没分配到模型）"
                           % int(idx.get("detected") or 0))

    # ② 按分配表决定跑哪些模型
    by_name = {str(m.get("name") or ""): _abs_dfm(m.get("dfm"), log) for m in models}
    try:
        from .. import _faces as _store
        from .compose import _auto_library
        lib = _auto_library({"project": cfg.get("project")})
        if lib is not None:
            for g in (_store.load_index(lib).get("groups") or []):
                nm = str(g.get("name") or "")
                if nm and not by_name.get(nm):
                    by_name[nm] = _abs_dfm(g.get("dfm"), log)
    except Exception as e:                                     # noqa: BLE001
        note("passes 读人脸库", e, "跨组判定不可用，只能用 body.models 里的模型", log)

    need_names = []
    for f in (idx.get("faces") or []):
        nm = str(f.get("model") or "")
        if nm and nm not in need_names:
            need_names.append(nm)
    todo = [{"name": n, "dfm": by_name.get(n, ""), "faces": []} for n in need_names]
    missing = [m["name"] for m in todo if not m["dfm"]]
    if missing:
        note("passes 模型解析", RuntimeError("%s" % missing),
             "这些模型的名字在 workspace/model 下找不到 .dfm，会被跳过", log)
    todo = [m for m in todo if m["dfm"]]
    if not todo:
        raise RuntimeError("分配表里的模型都找不到对应的 .dfm（表里：%s）" % (need_names or "空"))

    # ③ 逐趟
    metas = []
    done_faces = 0
    pool_idx = video.parent / (video.stem + "_cache") / "pool" / "index.json"
    for k, m in enumerate(todo, start=1):
        if cancelled and cancelled():
            raise RuntimeError("已取消")
        nm = str(m.get("name") or "")
        _p(2 + int(40 * (k - 1) / max(1, len(todo))),
           "分趟导出：第 %d/%d 趟 · %s" % (k, len(todo), nm),
           meta={"phase": "pass", "pass_i": k, "pass_n": len(todo),
                 "pass_label": nm, "faces_done": done_faces,
                 "faces_total": total_faces, "pool": done_faces,
                 "mask_on": bool(_mask_on), "frozen": True})
        r = pass_run(video, pool_idx, nm, str(m["dfm"]), log=log, want_xseg=True)
        done_faces += int(r.get("faces") or 0)
        if r.get("faces"):
            metas.append(r["path"])
    if not metas:
        raise RuntimeError("没有任何模型产出脸原料")

    # ④ 闸门后的合流
    _p(45, "分趟导出：闸门打开，合流（每帧只混合一次）…",
       meta={"phase": "merge", "pass_i": len(todo), "pass_n": len(todo),
             "pool": done_faces, "faces_total": total_faces,
             "mask_on": bool(_mask_on), "frozen": True})
    png_dir = Path(tempfile.mkdtemp(prefix="dfl_passes_"))
    try:
        res = merge(video, metas, cfg=dict(cfg.get("config") or {}), png_dir=str(png_dir),
                    on_frame=lambda n: _p(45 + int(45 * min(1.0, n / max(1, idx.get("frames") or n))),
                                          "分趟导出：合流 %d 帧" % n, n, int(idx.get("frames") or 0)),
                    log=log)
        _p(92, "分趟导出：编码 …", meta={"phase": "encode"})
        try:
            _ff = MR._resolve_ffmpeg("", "", Path("."))
            ffmpeg = str(_ff[0] if isinstance(_ff, (tuple, list)) else _ff)
        except Exception as e:                                 # noqa: BLE001
            note("passes 找 ffmpeg", e, "无法编码，这一单会失败", log)
            raise
        part = Path(str(out)).with_suffix(".part.mp4")
        fps = float(res.get("fps") or 25.0)
        cmd = [ffmpeg, "-y", "-hide_banner", "-loglevel", "error", "-nostats",
               "-framerate", "%.6g" % fps, "-start_number", "1",
               "-i", str(png_dir / "%08d.png"), "-i", str(video),
               "-map", "0:v:0", "-map", "1:a:0?",
               "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
               "-c:v", enc, "-crf", str(int(crf)), "-preset", preset,
               "-pix_fmt", "yuv420p", "-c:a", "copy", str(part)]
        import subprocess as sp
        pr = sp.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
        if pr.returncode:
            raise RuntimeError("编码失败（ffmpeg %s）：%s" % (pr.returncode, (pr.stderr or "")[-400:]))
        shutil.move(str(part), str(out))
    finally:
        shutil.rmtree(str(png_dir), ignore_errors=True)

    _p(100, "分趟导出完成：%d 帧" % int(res.get("frames") or 0))
    return {"out": str(out), "frames": int(res.get("frames") or 0),
            "faces": int(done_faces), "elapsed_s": round(time.time() - t0, 1)}
