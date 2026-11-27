# -*- coding: utf-8 -*-
"""多模型导出走分趟（docs/37）：Pass0 扫描 → 每个模型一趟 → 闸门 → 合流 → 编码。

为什么单独一条路：旧路径让每个 dfm 各持一份会话（每 lane 再乘一份），8GB 卡上 warm 会卡死。
分趟是「一趟只装一个模型」，显存与模型数解耦。

**跑哪些模型以分配表为准**（index.json 里实际出现的 model），不是 body.models：
库判定可能把脸分给 body 里没有的模型；按 body 循环会漏跑，报「没有任何模型产出脸原料」。
"""
from __future__ import annotations

import os
import shutil
import subprocess
import threading
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
    from .meter import _STAGE_METER as _METER
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
        _hb['t'] = time.time()
        if progress:
            progress(int(pct), msg, done, total, meta or {})
        log(msg)

    models = [m for m in (cfg.get("single_models") or []) if m.get("dfm") or m.get("name")]
    if not models:
        raise RuntimeError("passes: cfg.single_models 为空（这一单没有模型）")

    # ---- 看门狗：45 s 没有任何进度就把所有线程栈写到 <out>.stacks.txt ----
    # 为什么要它：分趟走 HTTP worker 线程时，第 2 趟会停在「装载模型 …」不动，
    # 而同一份代码在主线程里 2 s 就跑完 —— 没有栈就只能猜。
    _hb = {'t': time.time(), 'done': False}
    _stack_path = Path(str(cfg.get('out') or 'passes_out')).with_suffix('.stacks.txt')
    def _watchdog():
        import faulthandler
        dumped = 0.0
        while not _hb['done']:
            time.sleep(2)
            idle = time.time() - _hb['t']
            if idle >= 45 and (time.time() - dumped) >= 30:
                dumped = time.time()
                try:
                    with open(str(_stack_path), 'a', encoding='utf-8') as fh:
                        fh.write('\n==== %s · 空闲 %.0fs ====\n' % (time.strftime('%H:%M:%S'), idle))
                        faulthandler.dump_traceback(file=fh)
                    log('[watchdog] %.0fs 没进度，线程栈已写入 %s' % (idle, _stack_path.name))
                except Exception:                              # noqa: BLE001
                    pass
    threading.Thread(target=_watchdog, daemon=True, name='passes-watchdog').start()
    # ① 扫描
    _mm = int((cfg.get("config") or {}).get("mask_mode", 6) or 0)
    _mask_on = _mm >= 6                     # 6/7/8/9 才走遮罩网络；<6 用 dfm 遮罩
    _n_lane = max(1, min(8, int((cfg.get('config') or {}).get('compose_lanes') or 8)))
    _e_lane = max(1, min(6, int((cfg.get('config') or {}).get('encode_lanes') or 3)))
    try:
        _METER.begin()
        _METER.reset(lanes={
            'frames': ['解码'],
            'det': ['TinyMog#0'],
            'dfm': [str((models[0] or {}).get('name') or 'dfm') + '#0'] if models else ['dfm#0'],
            'xseg': ['XSegLite#0'],
            'compose': ['t%d' % i for i in range(_n_lane)],
            'encode': ['e%d' % i for i in range(_e_lane)],
        })
    except Exception:                                   # noqa: BLE001
        pass
    _p(2, "分趟导出：扫描人脸分配表 …",
       meta={"phase": "scan", "pass_i": 0, "pass_n": len(models), "pool": 0,
             "mask_on": bool(_mask_on), "frozen": False})
    def _scan_frame(nb):
        try:
            _METER.tick("frames")
            if nb:
                _METER.tick("det", nb)
        except Exception:                                     # noqa: BLE001
            pass

    # ★ cfg["start"] 的契约是**1 起**（pipeline_stages.py 里明写「start_frame 从 1 数起」，
    #   前端也是 frameFrom > 0 才发），而 scan() 的 start 是 **0 起下标**。
    #   以前直接原样传，于是区间首帧（0 起 S-1）被跳过 —— 只丢 1 帧，不致命，
    #   但既然对账就一并对齐。end 不用动：0 起开区间 [S, E) 的末帧正是 1 起闭区间的 E。
    _start0 = max(0, int(cfg.get("start") or 0) - 1)
    # ★ with_kps=True：68 点每张脸约 350ms（pass0.py:147 自己写的），而分趟导出
    #   会让**每个模型各算一遍同一张脸**（3 个模型 = 算 3 遍）。在 Pass 0 算一次
    #   存进 index.json，后面每趟直接读，省掉 (N-1) 倍。这是「分趟导出慢一个量级」
    #   的主因之一，和正确性无关但用户感知最强。
    idx = scan(video, models=models, plan_payload=cfg.get("plan_payload"),
               start=_start0,
               end=(int(cfg["end"]) if cfg.get("end") else None), log=log,
               with_kps=True,
               on_frame=_scan_frame)
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
        try:
            _METER.force_lanes({"dfm": [str(nm) + "#0"]})   # 这一趟的模型名，图上要跟着换
        except Exception:                                     # noqa: BLE001
            pass
        _live = {"faces": 0}

        def _on_face(n, _nm=nm):
            try:
                _METER.tick("dfm", n)
                if _mask_on:
                    _METER.tick("xseg", n)
                _METER.note_batch({"dfm": n})
                _live["faces"] += int(n)
                if _live["faces"] % 8 == 0:          # 池子边跑边长，别等整趟结束
                    _p(2 + int(40 * (k - 1) / max(1, len(todo))),
                       "分趟导出：第 %d/%d 趟 · %s" % (k, len(todo), _nm),
                       meta={"phase": "pass", "pass_i": k, "pass_n": len(todo),
                             "pass_label": _nm, "faces_done": done_faces + _live["faces"],
                             "faces_total": total_faces,
                             "pool": done_faces + _live["faces"],
                             "mask_on": bool(_mask_on), "frozen": True})
            except Exception:                                 # noqa: BLE001
                pass

        def _on_stage(what, _k=k, _nm=nm):
            try:
                _p(2 + int(40 * (_k - 1) / max(1, len(todo))),
                   ("分趟导出：第 %d/%d 趟 · %s · %s"
                    % (_k, len(todo), _nm, "装载模型 …" if what == "load" else "推理中")),
                   meta={"phase": "load" if what == "load" else "pass",
                         "pass_i": _k, "pass_n": len(todo), "pass_label": _nm,
                         "faces_done": done_faces, "faces_total": total_faces,
                         "pool": done_faces, "mask_on": bool(_mask_on), "frozen": True})
            except Exception:                                 # noqa: BLE001
                pass

        r = pass_run(video, pool_idx, nm, str(m["dfm"]), log=log, on_stage=_on_stage,
                     want_xseg=True, on_face=_on_face,
                     face_type=str((cfg.get("config") or {}).get("face_type") or "whole_face"))
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
        def _on_merged(n):
            try:
                _METER.tick("compose")
            except Exception:                             # noqa: BLE001
                pass
            _p(45 + int(45 * min(1.0, n / max(1, idx.get("frames") or n))),
               "分趟导出：合流 %d 帧" % n, n, int(idx.get("frames") or 0))

        # 区间必须一起传下去：以前只给了 scan()，merge() 从第 0 帧写照整片 ——
        # 「选 100~200 帧导出」静默变成全长，音画不错位但范围选择完全失效。
        _end0 = (int(cfg["end"]) if cfg.get("end") else None)
        res = merge(video, metas, cfg=dict(cfg.get("config") or {}), png_dir=str(png_dir),
                    on_frame=_on_merged, log=log, start=_start0, end=_end0)
        _p(92, "分趟导出：编码 …", meta={"phase": "encode"})
        try:
            _ff = MR._resolve_ffmpeg("", "", Path("."))
            ffmpeg = str(_ff[0] if isinstance(_ff, (tuple, list)) else _ff)
        except Exception as e:                                 # noqa: BLE001
            note("passes 找 ffmpeg", e, "无法编码，这一单会失败", log)
            raise
        # ★ 成片以前**从来没落到 out 上**：ffmpeg 只写 .part.mp4，之后没有 os.replace，
        #   而 finally 又把 png_dir 删了 —— 于是 job 报 ok=True / pct=100 / out=<承诺路径>，
        #   磁盘上却只有一个 <名字>.part.mp4，而本仓所有其他路径都把 .part 当半截文件
        #   （pipeline_stages.py:1718-1726 还会主动删掉它）。
        try:
            out.parent.mkdir(parents=True, exist_ok=True)
        except Exception as _e:                                # noqa: BLE001
            note("passes 建输出目录", _e, "编码可能失败", log)
        part = Path(str(out)).with_suffix(".part.mp4")
        fps = float(res.get("fps") or 25.0)
        cmd = [ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
               "-framerate", "%.6g" % fps, "-start_number", "1",
               "-i", str(png_dir / "%08d.png"),
               # 音频要跟视频段对齐：段是从 _start0 开始的，而 -c:a copy 直接拷整条，
               # 不 seek 就是「视频从 100 帧起、音频从 0 起」—— 音画整段错开。
               # -ss 放在 -i 之前是输入 seek（关键帧对齐，够用；精确 seek 会重编码，
               # 这里必须 copy 所以只能这样）。
               ] + (["-ss", "%.6f" % (float(_start0) / fps)] if _start0 else []) + [
               "-i", str(video),
               "-map", "0:v:0", "-map", "1:a:0?",
               "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
               "-c:v", enc, "-crf", str(int(crf)), "-preset", preset,
               "-pix_fmt", "yuv420p", "-c:a", "copy",
               # ★ 音频必须跟着截断。只给 -ss 不给 -t 的话，声音会从 seek 点一直拷到
               #   源片结尾 —— 成片时长 = 音轨剩余长度 > 画面长度，片尾画面冻住、声音还在走。
               #   对照 _merge_runner.py:704-705 那边是有 `-t` 的（还多吃 0.02s 防切片）。
               "-t", "%.6f" % (max(1, int(res.get("frames") or 1)) / max(1e-6, fps)),
               "-progress", "pipe:1", "-nostats", str(part)]
        import subprocess as sp
        _n_all = max(1, int(res.get("frames") or 1))
        pr = sp.Popen(cmd, stdout=sp.PIPE, stderr=sp.PIPE, text=True,
                      encoding="utf-8", errors="replace")
        _enc_done = 0
        for _line in (pr.stdout or []):
            _line = (_line or "").strip()
            if _line.startswith("frame="):
                try:
                    _n = int(_line.split("=", 1)[1])
                except Exception:                             # noqa: BLE001
                    continue
                _step = _n - _enc_done
                if _step > 0:
                    _enc_done = _n
                    try:
                        _METER.tick("encode", _step)
                    except Exception:                         # noqa: BLE001
                        pass
                    _p(92 + int(7 * min(1.0, _n / _n_all)),
                       "分趟导出：编码 %d/%d 帧" % (_n, _n_all),
                       _n, _n_all, meta={"phase": "encode", "mask_on": bool(_mask_on)})
        _err = (pr.stderr.read() if pr.stderr else "") or ""
        pr.wait()
        if pr.returncode:
            raise RuntimeError("编码失败（ffmpeg %s）：%s" % (pr.returncode, _err[-400:]))
        # 编码成功才改名：半截的 .part 留给排障，不要覆盖掉上一次的好成片。
        try:
            os.replace(str(part), str(out))
        except Exception as _e:                                # noqa: BLE001
            note("passes 把 .part 改名成成片", _e,
                 "成片还在 %s 这个临时名上" % part, log)
            raise
    finally:
        shutil.rmtree(str(png_dir), ignore_errors=True)

    try:
        _METER.tick("encode", max(0, int(res.get("frames") or 1) - int(_enc_done or 0)))
        _METER.stop()
    except Exception:                                     # noqa: BLE001
        pass

    _hb['done'] = True
    _p(100, "分趟导出完成：%d 帧" % int(res.get("frames") or 0))
    return {"out": str(out), "frames": int(res.get("frames") or 0),
            "faces": int(done_faces), "elapsed_s": round(time.time() - t0, 1)}
