# -*- coding: utf-8 -*-
"""train_log_append —— 训练结束自动追加记录到 workspace\\train_log.jsonl

用法（训练结束后，agent 执行）:
  python tools/train_log_append.py --model Anna_SAEHD --src "anna(4702,PNG)" --dst "TENN(17561)" \
      [--engine torch] [--verdict plateau_quit] [--note "..."] [--change "at=420,lr=3e-5->1.5e-5"] ...

iter/final_loss 从训练现场自动挖：优先 model 目录 data.dat（iter/loss_history 末值），
参数快照一并入档。agent 只填现场挖不到的主观字段（verdict/note/changes）。
"""
import argparse, json, pickle, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LOG = ROOT / "workspace" / "train_log.jsonl"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--engine", default="torch")
    ap.add_argument("--verdict", default="")
    ap.add_argument("--note", default="")
    ap.add_argument("--change", action="append", default=[])
    a = ap.parse_args()

    row = {"date": time.strftime("%Y-%m-%d"), "model": a.model, "engine": a.engine,
           "src": a.src, "dst": a.dst}

    # 从 data.dat 自动采集
    dat = ROOT / "workspace" / "model" / f"{a.model}_data.dat"
    if not dat.exists():
        # junction 场景：直接路径
        cands = list((ROOT / "workspace" / "model").glob(f"{a.model}_data.dat"))
        dat = cands[0] if cands else None
    if dat and dat.exists():
        try:
            d = pickle.loads(dat.read_bytes())
            hist = d.get("loss_history") or []
            o = d.get("options", {})
            row["iters"] = d.get("iter")
            row["final_loss"] = ({"src": round(float(hist[-1][0]), 4),
                                  "dst": round(float(hist[-1][1]), 4)} if hist else None)
            row["params_note"] = ",".join(f"{k}={o.get(k)}" for k in
                                          ("lr", "random_warp", "random_src_flip", "ct_mode")
                                          if k in o)
        except Exception as e:
            row["params_note"] = f"data.dat读取失败:{e}"
    else:
        row["iters"] = None
        row["final_loss"] = None
        row["params_note"] = ""

    row["changes"] = [{"raw": c} for c in a.change]
    row["verdict"] = a.verdict
    row["note"] = a.note

    with LOG.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"[train_log] 已追加: {a.model} {time.strftime('%Y-%m-%d')} -> {LOG}")


if __name__ == "__main__":
    main()
