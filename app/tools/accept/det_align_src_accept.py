# -*- coding: utf-8 -*-
r"""检测器「四列一致」验收：#4 / #5 / #6 / #7 的那四条静默回落。

跑法（仓库根）：
  runtime\server\Scripts\python.exe app\tools\accept\det_align_src_accept.py
  ... --frames 3        # 每单真跑几帧（默认 3：只为拿 <成片>.log 里那一行）
  ... --no-run          # 只造 cfg / eff，不真跑流水线（快，但没有「日志行」那一列）

五种请求形状（#4 的验收口径）：
  ① detector=TinyMog      界面「检测器」下拉发的就是它 → 以它为准
  ② 只带 quality=high     老脚本 / 老前端 → 按档推 retinaface_10g
  ③ 只带 quality=fast     同上 → tinymog
  ④ 什么都不带            ★ 以前静默落 retinaface_10g，现在必须 = _DEFAULT_DETECTOR(tinymog)
  ⑤ detector=垃圾值       必须 400（_BadInput），不许「就近退一个能用的」

每一列都必须是**同一个名字**：
  cfg.detector · cfg.det_align · <成片>.log 里「检测器 X（…）」· [eff] 抽屉的 detector
另外把分段链的 cfg 也照一遍（#7 的同一条口径：两字段永不打架）。
"""
from __future__ import annotations
import argparse, json, os, re, sys, time
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "app"))
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:                                          # noqa: BLE001
        pass

SETTINGS = {"blend": "seamless", "ct_mode": "mt", "mask_mode": 6,
            "mask_erode": 0, "mask_blur": 4}

OK = []; BAD = []


def say(s=""):
    print(s, flush=True)


def check(cond, name, detail=""):
    (OK if cond else BAD).append(name)
    say("  [%s] %s%s" % ("PASS" if cond else "FAIL", name,
                         ("  · " + str(detail)) if detail else ""))
    return bool(cond)


def det_of_log(p: Path) -> str:
    """<成片>.log 里「快速导出：检测器 <名字>（<来源> …）」→ 规范名。"""
    try:
        text = p.read_text("utf-8", "ignore")
    except OSError:
        return ""
    m = re.search(r"检测器\s+([A-Za-z0-9_.]+)\s*（", text)
    return m.group(1) if m else ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=3)
    ap.add_argument("--start", type=int, default=200)
    ap.add_argument("--no-run", action="store_true")
    a = ap.parse_args()

    from server import app as app_mod                            # noqa: F401
    from server.backend_merge import pipeline_fused as pf
    from server.backend_merge import pipeline_stages as ps
    from server.backend_merge import pipeline_video as pv
    from server.backend_merge import eff as effmod
    from server.backend_merge._shared import _DEFAULT_DETECTOR, _DETECTORS
    from server._common.http import _BadInput

    WS = REPO / "workspace"
    dfm = WS / "model" / "阿証証証証-416.dfm"
    video = WS / "9月29日 (5).mp4"
    for p in (dfm, video):
        if not p.exists():
            raise SystemExit("素材/模型不在：%s" % p)
    outdir = Path(os.environ.get("TEMP", ".")) / "dfl_det_align_src"
    outdir.mkdir(parents=True, exist_ok=True)

    say("检测器四列一致验收 · 默认档 = %s · 白名单 = %s"
        % (_DEFAULT_DETECTOR, " / ".join(_DETECTORS)))

    CASES = [
        ("①", "detector=TinyMog", {"detector": "TinyMog"}, "tinymog"),
        ("②", "只带 quality=high", {"quality": "high"}, "retinaface_10g"),
        ("③", "只带 quality=fast", {"quality": "fast"}, "tinymog"),
        ("④", "都不带", {}, _DEFAULT_DETECTOR),
        ("⑤", "detector=垃圾", {"detector": "resnet-9"}, 400),
    ]

    rows = []
    for tag, label, extra, want in CASES:
        say("")
        say("%s %s" % (tag, label))
        body = {"video": str(video), "model": dfm.name, "dfm": str(dfm),
                "settings": dict(SETTINGS),
                "start_frame": a.start, "end_frame": a.start + a.frames - 1,
                "output_dir": str(outdir / ("c%s" % tag)), "frame_budget": 0}
        body.update(extra)
        out = Path(body["output_dir"]) / ("c%s.mp4" % tag)
        try:
            cfg = pf._fused_cfg(body)
        except _BadInput as e:
            check(int(want) == 400 if isinstance(want, int) else False,
                  "%s 垃圾值 → %d（%s）" % (tag, e.status, e.msg[:60]), "")
            rows.append((tag, "400", "400", "400", "400", "400"))
            continue
        if isinstance(want, int):
            check(False, "%s 垃圾值本该 400，却造出了 cfg（detector=%s）"
                  % (tag, cfg.get("detector")), "")
            continue

        eff = effmod._eff_publish("/api/merge/export", "fused", body, cfg)
        eff_det = str(eff.get("detector") or "")
        log_det = ""
        if not a.no_run:
            pv._vid_open_log(out.with_suffix(".log"))
            try:
                res = pf._fused_run(cfg)
                got = str((res or {}).get("detector") or "")
            except BaseException as e:                           # noqa: BLE001
                say("    流水线跑挂了：%s: %s" % (type(e).__name__, str(e)[:200]))
                got = ""
            finally:
                pv._vid_close_log()
            log_det = det_of_log(out.with_suffix(".log")) or got
        # 分段链（#7 的同一条口径）
        try:
            scfg = ps._stage_cfg(body, "faces", chain=True)
            s_det, s_align = str(scfg.get("detector") or ""), str(scfg.get("det_align") or "")
        except BaseException as e:                               # noqa: BLE001
            s_det = s_align = "(%s)" % type(e).__name__

        rows.append((tag, str(cfg.get("detector")), str(cfg.get("det_align")),
                     log_det or "(no-run)", eff_det))
        say("    cfg.detector   = %s" % cfg.get("detector"))
        say("    cfg.det_align  = %s" % cfg.get("det_align"))
        say("    det_align_src  = %s" % cfg.get("det_align_src"))
        say("    日志行         = %s" % (log_det or "(no-run)"))
        say("    [eff] drawer   = %s（src=%s）" % (eff_det, eff.get("detector_src")))
        say("    分段 cfg        = detector %s / det_align %s" % (s_det, s_align))
        say("    [eff-warn]     = %s"
            % " · ".join(str(w.get("text")) for w in (eff.get("warnings") or [])) or "-")
        cols = [str(cfg.get("detector")), str(cfg.get("det_align"))]
        if log_det:
            cols.append(log_det)
        cols.append(eff_det)
        check(len(set(cols)) == 1 and cols[0] == want,
              "%s 四列一致 = %s" % (tag, want), " / ".join(cols))
        check(s_det == s_align == want,
              "%s 分段链两字段一致 = %s" % (tag, want), "%s / %s" % (s_det, s_align))
        if tag == "②":
            check(any("quality=high" in str(w.get("text")) for w in (eff.get("warnings") or [])),
                  "② 「由 quality 推出来」照旧进 [eff-warn]（eff_accept B 依赖它）")

    say("")
    say("=== 汇总 ===")
    for r in rows:
        say("  %s | cfg.detector %s | cfg.det_align %s | 日志 %s | eff %s"
            % (r[0], r[1], r[2], r[3], r[4]))
    say("PASS %d / FAIL %d" % (len(OK), len(BAD)))
    for b in BAD:
        say("  FAIL: " + b)
    return 0 if not BAD else 1


if __name__ == "__main__":
    sys.exit(main())
