# -*- coding: utf-8 -*-
r"""滞回开关的像素对拍：DFL_PRED_BATCH_HYST=0 vs =1，用 DFL_FUSED_DUMP 逐文件比。

要证明的事：这个改动只改**调度**（dfm 一次喂几张），不改算法。
所以两次的 dump 目录（每帧的脸 / pred / 两路遮罩 / 目标脸 / 68 点 / 用的哪个 dfm）
必须逐文件相同 —— 判据是 max|Δ| = 0，不是「肉眼看不出」。

顺带量一件事（不是本改动的口径，是背景）：批组成换了之后 ORT 的卷积核选择会变，
同一套 pipeline 在 batch=1 与 batch=8 之间的差 —— 这条既有差异本来就在
（fused_export_accept.py 里对分段的路子是 mean 0.30~0.70 LSB），这里给出数字。

跑法（仓库根，server venv）：
    runtime\server\Scripts\python.exe app\tools\bench\pred_batch_hyst_parity.py
退出码：0 = 逐文件相同；2 = 有差异 / 有单没跑成。
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
for p in (str(REPO), str(REPO / "app"), str(Path(__file__).resolve().parent)):
    if p not in sys.path:
        sys.path.insert(0, p)
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

import numpy as np                                              # noqa: E402
import cv2                                                      # noqa: E402

import pred_batch_hyst_ab as AB                                 # noqa: E402

DUMP = "1230,1260,1290"
RANGE = (1200, 1300)
ACC = []


def check(ok, name, detail=""):
    ACC.append(bool(ok))
    print(("[PASS] " if ok else "[FAIL] ") + name + ("  · " + str(detail) if detail else ""),
          flush=True)
    return ok


def run_export(tag, hyst, batch=0, warm=False):
    """起后端 →（可选）预热一单 → 带 dump 跑一单；返回 (dump 目录, 帧/秒, 选的批)。"""
    print("\n--- %s（HYST=%s%s）---" % (tag, hyst, "，batch=%s" % batch if batch else ""),
          flush=True)
    env = {"DFL_PRED_BATCH_HYST": str(hyst), "DFL_FUSED_DUMP": DUMP}
    proc, logpath = AB.start_server(tag, env)
    try:
        for i, (a, b, dump) in enumerate(
                [(RANGE[0], RANGE[0] + 99, False)] * (1 if warm else 0)
                + [(RANGE[0], RANGE[1], True)]):
            out = AB.OUT / ("%s_%d.mp4" % (tag, i))
            if out.exists():
                out.unlink()
            body = {"video": str(AB.VIDEO), "model": AB.MODEL,
                    "settings": dict(AB.SETTINGS), "output_dir": str(out),
                    "start_frame": int(a), "end_frame": int(b)}
            if int(batch or 0) > 0:
                body["batch"] = int(batch)
            st, d = AB.post("/api/merge/export", body)
            if st != 200 or d.get("ok") is False:
                raise SystemExit("[%s] 导出没起来：HTTP %s %s" % (tag, st, d))
            # 真实成片名带帧范围后缀（xxx_1200-1300.mp4）—— dump 目录/日志都跟着它
            real = Path(str(d.get("output") or out))
            fin = AB.wait_export(1800.0)
            if fin.get("phase") != "done":
                raise SystemExit("[%s] 导出失败：%s %s" % (tag, fin.get("phase"),
                                                          (fin.get("error") or "")[:300]))
            stats = (fin.get("vid") or {}).get("stats") or {}
            if dump:
                pick, _ln = AB.vid_log_pick(real)
                dumpdir = Path(str(real).replace(".mp4", "") + ".dump")
                fps = float(stats.get("fps") or 0.0)
                print("    帧/秒 %.2f · 批 %s · dump %s" % (fps, pick, dumpdir.name), flush=True)
                return dumpdir, fps, pick
    finally:
        AB.stop_server(proc)


def cmp_dirs(a: Path, b: Path):
    """逐文件比：png 给 mean|Δ|/max|Δ|，其它（lm.json / dfm.txt）逐字节。"""
    fa = sorted(p.name for p in a.glob("*") if p.is_file())
    fb = sorted(p.name for p in b.glob("*") if p.is_file())
    if fa != fb:
        return {"same_set": False, "a": fa, "b": fb, "rows": []}
    rows, worst_mean, worst_max = [], 0.0, 0
    for name in fa:
        pa, pb = a / name, b / name
        if name.endswith(".png"):
            ia = cv2.imread(str(pa), cv2.IMREAD_UNCHANGED)
            ib = cv2.imread(str(pb), cv2.IMREAD_UNCHANGED)
            if ia is None or ib is None or ia.shape != ib.shape:
                rows.append({"name": name, "bad": "shape/read"})
                continue
            d = np.abs(ia.astype(np.int16) - ib.astype(np.int16))
            m, mx = float(d.mean()), int(d.max())
            worst_mean, worst_max = max(worst_mean, m), max(worst_max, mx)
            rows.append({"name": name, "mean": round(m, 6), "max": mx})
        else:
            same = pa.read_bytes() == pb.read_bytes()
            rows.append({"name": name, "bytes_same": same})
            if not same:
                worst_max = 255
    return {"same_set": True, "n": len(fa), "rows": rows,
            "worst_mean": worst_mean, "worst_max": worst_max}


def main():
    AB.prepare_sandbox()
    d0, fps0, pick0 = run_export("h0", 0, warm=True)          # 预热一单 → 热进程判 8
    d1, fps1, pick1 = run_export("h1", 1, warm=True)
    print("")
    r = cmp_dirs(d0, d1)
    check(r.get("same_set"), "两次 dump 的文件清单一致", "n=%s" % r.get("n"))
    if r.get("same_set"):
        check(r["worst_max"] == 0 and r["worst_mean"] == 0.0,
              "HYST=0 vs =1 逐文件相同（像素与 lm 都逐字节）",
              "%d 个文件 · worst mean %.6f · worst max %d · 批 %s vs %s"
              % (r["n"], r["worst_mean"], r["worst_max"], pick0, pick1))

    # 背景数据：批组成变了，ORT 选核会变 —— batch=1 与 batch=8 的差（不是本改动的锅）
    db1, _f, pb1 = run_export("b1", 1, batch=1, warm=True)
    r2 = cmp_dirs(d0, db1)
    if r2.get("same_set"):
        print("    背景：batch %s vs %s → worst mean %.4f LSB · worst max %d"
              % (pick0, pb1, r2["worst_mean"], r2["worst_max"]), flush=True)

    print("")
    print("=" * 78)
    print("%d 项检查 · 全过" % len(ACC) if all(ACC) else "有检查没通过")
    return 0 if all(ACC) else 2


if __name__ == "__main__":
    sys.exit(main())
