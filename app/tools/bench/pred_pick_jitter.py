# -*- coding: utf-8 -*-
r"""_pred_pick_batch 的抖动量测：同一进程、同一 dfm、连续 N 次决策。

台阶是实测的，但实测本身有噪声；旧规则是「每张耗时 <= 最快值 * 1.05 里最大的
batch」，当两档差距正好落在 5% 边缘（本机 B1/B8 每张差 0~8%），每次都可能翻边。

  A 噪声底   固定档位重复跑 _dfm_run_batch，给每张 ms 的 cv（同一档位自身的抖动）
  B 决策序列 连调 N 次 _pred_pick_batch({"pred_batch": 0}, ent)，记 B1/B4/B8 与选择
  C 抖动率   switches / (N-1)，另报「选 1 / 选 8」各几次

跑法（仓库根，server venv）：
    runtime\server\Scripts\python.exe app\tools\bench\pred_pick_jitter.py --runs 10
    DFL_PRED_BATCH_HYST=0 ...      # 看旧行为（=1 是默认）
退出码：0 = 跑完；2 = 前置条件不成立（没 GPU / 没模型 / 掉到 CPU）。
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
for p in (str(REPO), str(REPO / "app")):
    if p not in sys.path:
        sys.path.insert(0, p)
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
os.environ.setdefault("DFL_STAGE_KEEP", "1")

import numpy as np                                # noqa: E402
from server import app as _app_mod                # noqa: E402,F401  先 import app，否则 routes_merge 的循环导入会炸
from server import routes_merge as rm            # noqa: E402


def say(m=""):
    try:
        print(m, flush=True)
    except UnicodeEncodeError:
        print(str(m).encode("utf-8", "replace").decode("utf-8", "replace"), flush=True)


def head(t):
    say("")
    say("=" * 78)
    say(t)
    say("=" * 78)


def noise_floor(ent, batches, reps):
    """同一档位重复跑：它自己抖多少（cv / 极差）。"""
    size = int(ent[2])
    out = {}
    for b in batches:
        x = np.zeros((int(b), size, size, 3), np.float32)
        rm._dfm_run_batch(ent, x)
        rm._dfm_run_batch(ent, x)
        per = []
        for _ in range(int(reps)):
            t0 = time.perf_counter()
            rm._dfm_run_batch(ent, x)
            per.append((time.perf_counter() - t0) * 1000.0 / float(b))
        mu = statistics.fmean(per)
        out[int(b)] = {"per_face_ms": [round(v, 2) for v in per],
                       "mean": round(mu, 3), "min": round(min(per), 3),
                       "max": round(max(per), 3),
                       "sd": round(statistics.pstdev(per), 3),
                       "cv_pct": round(100.0 * statistics.pstdev(per) / mu, 2),
                       "spread_pct": round(100.0 * (max(per) - min(per)) / mu, 2)}
    return out


DFM_PATH = ""


def decisions(ent, runs):
    seq = []
    for i in range(int(runs)):
        b, meta = rm._pred_pick_batch({"pred_batch": 0, "dfm": DFM_PATH}, ent)
        lad = meta.get("ladder") or []
        seq.append({"i": i + 1, "pick": int(b), "source": meta.get("source"),
                    "hyst": meta.get("hyst"), "want": meta.get("want_batch"),
                    "per_face": {int(r["batch"]): float(r["per_face_ms"]) for r in lad},
                    "ms": {int(r["batch"]): float(r["ms"]) for r in lad}})
    return seq


def summarize(seq):
    picks = [r["pick"] for r in seq]
    sw = sum(1 for a, b in zip(picks, picks[1:]) if a != b)
    n = max(1, len(picks) - 1)
    counts = {}
    for p in picks:
        counts[str(p)] = counts.get(str(p), 0) + 1
    for r in seq:                        # 子进程回来的 json：键是字符串，统一成 int
        r["per_face"] = {int(k): float(v) for k, v in (r.get("per_face") or {}).items()}
    per_batch = {}
    for b in sorted({k for r in seq for k in r["per_face"]}):
        vals = [r["per_face"][b] for r in seq if b in r["per_face"]]
        if vals:
            mu = statistics.fmean(vals)
            per_batch[str(b)] = {"mean": round(mu, 2), "min": round(min(vals), 2),
                                 "max": round(max(vals), 2),
                                 "cv_pct": round(100.0 * statistics.pstdev(vals) / mu, 2)}
    tail = picks[1:]
    return {"runs": len(picks), "picks": picks, "counts": counts, "switches": sw,
            "jitter_pct": round(100.0 * sw / n, 1),
            "batch_mean": round(statistics.fmean(picks), 2),
            "batch_mean_ex_first": round(statistics.fmean(tail), 2) if tail else None,
            "per_batch_per_face": per_batch,
            "hyst_env": os.environ.get("DFL_PRED_BATCH_HYST")}


def mixed(ent, n, model):
    """交替「冷进程 / 热进程」各一次决策 —— 生产上就是「重启一次再跑一次」。

    冷：子进程只加载 dfm 就判（刚起来的 app）；热：本进程（台阶已经跑过一遍）。
    这两者的「B8 比 B1 慢多少」不是一个数，旧规则的分界线又正好卡在中间 ——
    抖动就是这么来的，所以量抖动必须按这个顺序量。
    """
    import subprocess as sp
    tmp = Path(os.environ.get("TEMP") or ".") / ("_pickjit_child_%d.json" % os.getpid())
    rows = []
    for k in range(int(n)):
        if k % 2 == 0:
            if tmp.exists():
                tmp.unlink()
            sp.run([sys.executable, str(Path(__file__).resolve()), "--runs", "1",
                    "--no-noise", "--model", model, "--json", str(tmp)],
                   capture_output=True, text=True, encoding="utf-8", errors="replace")
            r = dict(json.loads(tmp.read_text("utf-8"))["decisions"][0])
            r["warm"] = False
        else:
            r = dict(decisions(ent, 1)[0])
            r["warm"] = True
        r["i"] = k + 1
        rows.append(r)
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=10)
    ap.add_argument("--model", default="阿証証証証-416")
    ap.add_argument("--noise-reps", type=int, default=10)
    ap.add_argument("--no-noise", action="store_true")
    ap.add_argument("--mixed", type=int, default=0,
                    help=">0 = 交替冷/热进程各一次决策，做 N 次（跨启动抖动）")
    ap.add_argument("--json", default="")
    args = ap.parse_args(argv)

    ws = Path(os.environ.get("DFL_WORKSPACE") or (REPO / "workspace"))
    name = args.model if args.model.endswith(".dfm") else args.model + ".dfm"
    dfm = name if Path(name).is_absolute() else str(ws / "model" / name)
    head("_pred_pick_batch 抖动探针")
    say("workspace：%s" % ws)
    say("dfm      ：%s" % dfm)
    say("环境     ：HYST=%r TRT=%r CPU_FALLBACK=%r"
        % (os.environ.get("DFL_PRED_BATCH_HYST"), os.environ.get("DFL_DFM_TRT"),
           os.environ.get("DFL_ALLOW_CPU_FALLBACK")))
    if not Path(dfm).is_file():
        say("没有这个 dfm：%s" % dfm)
        return 2
    globals()["DFM_PATH"] = dfm
    t0 = time.time()
    ent = rm._dfm_session(dfm)
    backend = ent[6] if len(ent) > 6 else "?"
    cap = int(ent[5] or 1) if len(ent) > 5 else 8
    say("会话就绪：%dx%d · %s · 单次最多 %d 张 · 加载 %.1fs"
        % (int(ent[2]), int(ent[2]), backend, cap, time.time() - t0))
    if str(backend).startswith("ort-cpu"):
        say("dfm 在 CPU 上 —— 这里量的是 GPU 台阶，CPU 上没意义")
        return 2

    noise = {}
    if not args.no_noise:
        head("A · 噪声底（同一档位重复 %d 次）" % args.noise_reps)
        noise = noise_floor(ent, [b for b in (1, 4, 8) if b <= cap], args.noise_reps)
        for b, row in sorted(noise.items()):
            say("B%-2d 每张 %.2f ms · min %.2f / max %.2f · sd %.2f · cv %.2f%% · 极差 %.2f%%"
                % (b, row["mean"], row["min"], row["max"], row["sd"], row["cv_pct"],
                   row["spread_pct"]))

    if args.mixed:
        head("B · 交替冷/热进程，共 %d 次决策（跨启动的抖动）" % args.mixed)
        decisions(ent, 1)                    # 预热本进程（不进表）
        seq = mixed(ent, args.mixed, args.model)
    else:
        head("B · 连续 %d 次决策（同一进程 / 同一 dfm）" % args.runs)
        seq = decisions(ent, args.runs)
    for r in seq:                        # 子进程回来的 json：键是字符串，打印前统一成 int
        r["per_face"] = {int(k): float(v) for k, v in (r.get("per_face") or {}).items()}
    say("  次  冷热     B1      B4      B8   选  实测倾向  滞回")
    say("-" * 62)
    for r in seq:
        hy = r.get("hyst") or {}
        say("%4d  %-4s  %6s  %6s  %6s  %3s  %6s   %s"
            % (r["i"],
               "-" if r.get("warm") is None else ("热" if r["warm"] else "冷"),
               "%.2f" % r["per_face"][1] if 1 in r["per_face"] else "-",
               "%.2f" % r["per_face"][4] if 4 in r["per_face"] else "-",
               "%.2f" % r["per_face"][8] if 8 in r["per_face"] else "-",
               r["pick"], r.get("want"),
               ("%s(gain=%s%%, streak=%s)" % (hy.get("action"), hy.get("gain_pct"),
                                              hy.get("streak"))
                if hy.get("on") else "-")))

    summ = summarize(seq)
    head("C · 抖动率")
    say("选择序列：%s" % " / ".join(str(p) for p in summ["picks"]))
    say("选 1 共 %s 次 · 选 8 共 %s 次 · 换档 %s 次"
        % (summ["counts"].get("1", 0), summ["counts"].get("8", 0), summ["switches"]))
    say("抖动率（换档 / 相邻对）：%.1f%% · 平均批 %.2f（去掉第一次 %.2f）"
        % (summ["jitter_pct"], summ["batch_mean"], summ["batch_mean_ex_first"] or 0.0))
    for b, row in sorted(summ["per_batch_per_face"].items(), key=lambda kv: int(kv[0])):
        say("跨决策 B%s 每张：mean %.2f · min %.2f · max %.2f · cv %.2f%%"
            % (b, row["mean"], row["min"], row["max"], row["cv_pct"]))

    blob = {"model": Path(dfm).name, "backend": backend, "cap": cap, "noise": noise,
            "decisions": seq, "summary": summ, "at": time.strftime("%Y-%m-%d %H:%M:%S")}
    if args.json:
        Path(args.json).write_text(json.dumps(blob, ensure_ascii=False, indent=1), "utf-8")
        say("json：%s" % args.json)
    say("PICKJITTER_JSON " + json.dumps(
        {"hyst": summ["hyst_env"], "jitter_pct": summ["jitter_pct"],
         "counts": summ["counts"], "batch_mean": summ["batch_mean"],
         "picks": summ["picks"],
         "noise_cv": {str(k): v["cv_pct"] for k, v in noise.items()}}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
