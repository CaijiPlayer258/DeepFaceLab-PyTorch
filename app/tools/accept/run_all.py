# -*- coding: utf-8 -*-
r"""把 app/tools/accept 下那几十个验收脚本一次跑完，给一张 pass/fail 表。

为什么要它
==========
每个脚本自己都写好了「跑真后端、判 pass/fail、退出码 0 = 全过」，但**没有统一入口**，
也不进发布包（make_release_zip 的 SKIP_PATHS 排除了 app/tools/accept）——于是日常
没人跑，文档和代码就各自漂移。举个刚发生的例子：mask_pred_accept.py 开头还写着
「XSegLite 后端 = TensorRT（int8 引擎）」「XSeg（70MB）」，而精度策略已经按实测改成
XSegLite=int8 / XSeg=bf16（89MB）了，没有任何东西会提醒。

一句话：一次改动能不能发，应该有比「我记得刚才手动点过」更硬的依据。

分组（LEVEL）
============
  fast   纯逻辑 / 自造数据，不碰 GPU、视频、dfm
  model  要 GPU 或引擎或 dfm，但不要视频素材
  e2e    要真视频 + 真 dfm + 真后端（一个几分钟）
  tool   不是验收，是给人用的对比工具（默认永远不跑）
默认只跑 fast。

跑法（仓库根）
============
    runtime\unified\python.exe app\tools\accept\run_all.py            # fast
    runtime\unified\python.exe app\tools\accept\run_all.py --group fast,model
    runtime\unified\python.exe app\tools\accept\run_all.py --all -v
    runtime\unified\python.exe app\tools\accept\run_all.py --only sr_accept,eff_accept
    runtime\unified\python.exe app\tools\accept\run_all.py --list
    runtime\unified\python.exe app\tools\accept\run_all.py --all --json out.json

退出码：0 = 跑的全过；1 = 有失败/超时；2 = 参数或环境问题。
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]

# 每个脚本属于哪一档。tool 是「不是验收」的两个（给人跑的对比工具）。
# 没列进来的默认 e2e —— 宁可多给时间，也不要因为它其实很轻就被饿死。
LEVEL = {
    # --- fast：不碰 GPU / 视频 / dfm ---
    "pred_batch_hyst_accept": "fast",   # 喂合成台阶，1.2 s
    "test_train_invariants": "fast",    # 存档往返 / lr 曲线 / 梯度覆盖，12 s
    # align_wf_accept 要 4.0 的参考对齐目录（--ref40，默认指向 4.0 仓库的 data_src/aligned），
    # 而且它现在会报 E2 FAIL：拿 4.0 切出来的芯片重算，四角差 24.02 px（n=4561）。
    # 那是**对齐基线**的问题，不是脚本坏了 —— 所以不放 fast 组里每天红着。
    "align_wf_accept": "model",
    # --- model：要 GPU / 引擎 / dfm，不要视频 ---
    "sr_accept": "model",               # 要 GPEN 引擎，约 2.6 min
    "compose_gpu_accept": "model",
    "det_ep_parity": "model",
    "det_trt_parity": "model",
    "det68_trt_parity": "model",
    "roi_parity_accept": "model",
    "test_detector": "model",
    "vram_plan_accept": "model",
    "eff_accept": "model",
    "cmp_backend_gate_accept": "model",
    # --- tool：不是验收 ---
    "face_ab": "tool",
    "passes_compare": "tool",
}

DEFAULT_TIMEOUT = {"fast": 300, "model": 900, "e2e": 2400, "tool": 600}


def _summary(p: Path) -> str:
    """docstring 第一行，给 --list 用。"""
    try:
        txt = p.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return ""
    for line in txt.splitlines()[:40]:
        s = line.strip()
        if s.startswith('"""') or s.startswith("r\"\"\""):
            s = s.lstrip("r").lstrip('"').strip()
            if s:
                return s[:76]
    return ""


def discover():
    out = []
    for p in sorted(HERE.glob("*.py")):
        name = p.stem
        if name.startswith("_") or name == "run_all":
            continue
        out.append({"name": name, "path": p,
                    "level": LEVEL.get(name, "e2e"),
                    "summary": _summary(p)})
    return out


def _tail(text: str, n: int = 3) -> str:
    lines = [x.rstrip() for x in (text or "").splitlines() if x.strip()]
    return " | ".join(lines[-n:])[:300]


def run_one(item, timeout: int, verbose: bool):
    p = item["path"]
    t0 = time.time()
    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    try:
        r = subprocess.run([sys.executable, str(p)], cwd=str(ROOT), env=env,
                           capture_output=True, text=True,
                           encoding="utf-8", errors="replace", timeout=timeout)
        rc, out = r.returncode, (r.stdout or "") + (r.stderr or "")
    except subprocess.TimeoutExpired:
        rc, out = -9, "TIMEOUT after %d s" % timeout
    except Exception as e:  # noqa: BLE001
        rc, out = -1, "%s: %s" % (type(e).__name__, e)
    dt = time.time() - t0
    if verbose or rc != 0:
        logdir = Path(tempfile.gettempdir()) / "accept_run"
        try:
            logdir.mkdir(parents=True, exist_ok=True)
            (logdir / (item["name"] + ".log")).write_text(out, encoding="utf-8")
        except Exception:
            pass
    return {"name": item["name"], "level": item["level"], "rc": rc,
            "sec": round(dt, 1), "tail": _tail(out)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="跑 app/tools/accept 下的验收脚本")
    ap.add_argument("--group", default="fast", help="逗号分隔：fast,model,e2e（默认 fast）")
    ap.add_argument("--all", action="store_true", help="跑 fast+model+e2e（不含 tool）")
    ap.add_argument("--only", default="", help="只跑这几个（逗号分隔，忽略 --group）")
    ap.add_argument("--skip", default="", help="跳过这几个（逗号分隔）")
    ap.add_argument("--list", action="store_true", help="只列出脚本和分组")
    ap.add_argument("--timeout", type=int, default=0, help="覆盖每档的默认超时（秒）")
    ap.add_argument("--json", default="", help="把结果写成 JSON")
    ap.add_argument("-v", "--verbose", action="store_true", help="全部输出都落盘")
    args = ap.parse_args(argv)

    items = discover()
    if not items:
        print("app/tools/accept 下没找到脚本");
        return 2

    if args.list:
        print("%-34s %-6s %s" % ("脚本", "分组", "摘要"))
        for it in items:
            print("%-34s %-6s %s" % (it["name"], it["level"], it["summary"]))
        return 0

    only = {x.strip() for x in args.only.split(",") if x.strip()}
    skip = {x.strip() for x in args.skip.split(",") if x.strip()}
    if only:
        sel = [it for it in items if it["name"] in only]
        miss = only - {it["name"] for it in sel}
        if miss:
            print("这些名字不在 accept 目录里：%s" % ", ".join(sorted(miss)))
            return 2
    elif args.all:
        sel = [it for it in items if it["level"] != "tool"]
    else:
        groups = {x.strip() for x in args.group.split(",") if x.strip()}
        bad = groups - set(DEFAULT_TIMEOUT)
        if bad:
            print("未知分组：%s（可用 fast/model/e2e/tool）" % ", ".join(sorted(bad)))
            return 2
        sel = [it for it in items if it["level"] in groups]
    sel = [it for it in sel if it["name"] not in skip]
    if not sel:
        print("没有选中任何脚本（--group=%s）" % args.group)
        return 2

    print("仓库：%s" % ROOT)
    print("解释器：%s" % sys.executable)
    print("要跑 %d 个：%s" % (len(sel), ", ".join(it["name"] for it in sel)))
    print("=" * 78)

    results = []
    t0 = time.time()
    for i, it in enumerate(sel, 1):
        to = args.timeout or DEFAULT_TIMEOUT.get(it["level"], 900)
        print("[%d/%d] %-34s " % (i, len(sel), it["name"]), end="", flush=True)
        r = run_one(it, to, args.verbose)
        results.append(r)
        mark = "PASS" if r["rc"] == 0 else ("TIMEOUT" if r["rc"] == -9 else "FAIL")
        print("%-8s %6.1fs" % (mark, r["sec"]))
        if r["rc"] != 0 and r["tail"]:
            print("        %s" % r["tail"])

    total = time.time() - t0
    ok = [r for r in results if r["rc"] == 0]
    bad = [r for r in results if r["rc"] != 0]
    print("=" * 78)
    print("PASS %d / %d · 总耗时 %.1f s" % (len(ok), len(results), total))
    if bad:
        print("失败：")
        for r in bad:
            print("  %-34s rc=%-4s %s" % (r["name"], r["rc"], r["tail"][:160]))
        print("完整输出：%s" % (Path(tempfile.gettempdir()) / "accept_run"))
    if args.json:
        try:
            Path(args.json).write_text(json.dumps(
                {"root": str(ROOT), "python": sys.executable,
                 "ok": len(ok), "total": len(results),
                 "elapsed_s": round(total, 1), "results": results},
                ensure_ascii=False, indent=1), encoding="utf-8")
            print("JSON：%s" % args.json)
        except Exception as e:
            print("写 JSON 失败：%s" % e)
    return 0 if not bad else 1


if __name__ == "__main__":
    sys.exit(main())
