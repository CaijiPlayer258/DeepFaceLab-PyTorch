# -*- coding: utf-8 -*-
"""项目导出的**自适应档位**：实测缓存 + 选择规则。

为什么单独一个文件：这一步只有三件事 —— 「量过什么」（缓存）、「能不能选它」（显存 /
争用 / 相邻档）、「为什么选它」（那句日志）。塞进 routes_merge.py 那张 15000 行的
文件里，以后没人敢改；而这三件事一行 GPU 都不碰，纯逻辑，好测。

档位 = (workers, batch)：workers 是几个 worker 进程（每份一份 dfm session = 一份
显存），batch 是一条请求带几帧（worker 内几线程）。同一个「在飞帧数」摊在 1 份会话
还是 2 份上，对 8GB 卡是**生死之差**（多一份就多背一份权重和 arena，实测 384 那份
模型 2×2 顶到 95% 换页、1×4 停在 77%）—— 所以这个值必须实测，不能按权重写死。

三条硬规则（用户定的，见 app/docs/20-PROJ-ADAPTIVE.md）：
  ① 只在「实测峰值 ≤ 85% 整卡」的候选里选**帧速最高**的；
  ② 2 个 worker 的 swap p50 > 1.4 × 1 个 worker 的 → 判为争用/换页，**不用 2 worker**；
  ③ 某档明显慢于低一档（>10%）→ 跳过（那是噪音/换页，不是配置）。

没有实测（第一次跑这份模型）→ 保守默认（≥1GB 权重 1×4、≤600MB 2×2），并把
**这一单跑出来的实测**写进 json，下次直接用 —— 所以「第一单慢一点」只发生一次。
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

__all__ = ["BENCH_VER", "CAP", "CONTEND", "STEP_LOSS", "model_key", "labels",
           "candidates", "load", "get_row", "put_row", "choose", "conservative",
           "note", "enabled", "path_of"]

BENCH_VER = 1                  # 口径变了就 +1（旧行整批作废，不拿错的口径算）
CAP = 0.85                     # 候选的实测显存上限（占整卡，用户口径）；这是**目标**
HARD = 0.92                    # 换页硬线：85% 内没有可选的档时，允许到这里（95% 是实测换页）
CONTEND = 1.4                  # 2 worker 的 swap p50 超过 1 worker 的这个倍数 → 疑似争用
CONTEND_WORTH = 1.05           # ……但只有「帧速也没快过 5%」才算真争用（否掉换页档）。
                               # 1.4× 那条单独用会误杀：416 的 2×2 实测 swap 335ms vs 220ms
                               # = 1.52×，但它 11.1 帧/秒 vs 10.0 —— 更快的那档不能因为排队时间
                               # 长就被否掉；真换页的样子是「swap 涨、帧速还塌」（384: 4.5×、3.49）
STEP_LOSS = 0.10               # 比低一档慢 10% 以上 → 跳过（噪音/换页）
SAMPLE_FRAMES = 16             # 标定用几帧（用户口径 8~12；取 16 是因为 2 worker 那档
                               # 头两批还在热 arena，12 帧里被启动项吃掉的比例太大，
                               # 实测会把 2×2 的 13.4 量成 10.0 → 选错档）
ENV_BENCH = "DFL_PROJ_BENCH"   # 0 = 不起标定（只用缓存 / 保守默认；排障与回归用）


def enabled() -> bool:
    return str(os.environ.get(ENV_BENCH) or "1").strip().lower() not in (
        "0", "false", "no", "off")


def path_of(workspace) -> Path:
    return Path(str(workspace)) / "_proj_bench.json"


# ---------------------------------------------------------------- 模型身份 / 缓存
def model_key(paths) -> str:
    """模型身份 = 名字 + 体积 + mtime（+ 多模型时的顺序）。

    换权重 / 重训导出会让 mtime 变 → 旧实测自动作废。和 _vram_sess_key 同一个口径，
    但**多模型**时把每一份都拼进来：项目导出可以一次贴多个 dfm，会话开销是叠加的。
    """
    bits = []
    for p in (paths or []):
        s = str(p or "")
        if not s:
            continue
        try:
            st = os.stat(s)
            bits.append("%s|%d|%d" % (Path(s).name, int(st.st_size), int(st.st_mtime)))
        except OSError:
            bits.append("%s|0|0" % Path(s).name)
    return "+".join(bits)


def labels() -> list:
    return ["%dx%d" % (w, b) for w in (1, 2) for b in (2, 4)]


def candidates(max_procs: int = 2, max_batch: int = 4) -> list:
    """要量的组合（顺序 = 从省显存到吃显存），去重后按「在飞帧数」升序。"""
    out = []
    for w in (1, 2)[: max(1, min(2, int(max_procs or 1)))]:
        for b in (2, 4):
            if b > int(max_batch or 4):
                continue
            out.append((w, b))
    return out


def load(workspace) -> dict:
    try:
        p = path_of(workspace)
        if p.is_file():
            d = json.loads(p.read_text("utf-8", "ignore") or "{}")
            if isinstance(d, dict):
                return d
    except Exception:                                       # noqa: BLE001
        pass
    return {}


def get_row(workspace, key: str) -> dict:
    """读一条实测（key = 模型身份）。口径不对 / 版本不对 → 当没有。"""
    if not key:
        return {}
    row = load(workspace).get(str(key)) or {}
    if not isinstance(row, dict) or int(row.get("v") or 0) != BENCH_VER:
        return {}
    return dict(row)


def put_row(workspace, key: str, row: dict) -> dict:
    """写一条实测（只保留最新一份模型的，文件不会无限长）。"""
    if not key:
        return {}
    d = load(workspace)
    d[str(key)] = dict(row)
    if len(d) > 12:                                        # 老模型的记录只留最近 12 条
        for k in sorted(d, key=lambda k: float((d.get(k) or {}).get("ts") or 0))[:-12]:
            d.pop(k, None)
    try:
        path_of(workspace).write_text(json.dumps(d, ensure_ascii=False, indent=1), "utf-8")
    except Exception:                                       # noqa: BLE001
        pass
    return d


# ---------------------------------------------------------------- 选择
def _est_peak(row: dict, base_used_mb) -> float:
    """这一档**今天**的整卡峰值估算。

    标定时量到的是「整卡占用」= 基线（桌面/别的程序）+ 本程序；基线变了（用户开了
    个视频、浏览器吃了 800MB）峰值就得跟着变，所以优先用 delta（本程序那部分）+
    现在的基线；没有 delta 才退回标定那天的绝对值。
    """
    d = row.get("delta_mb")
    if d is not None and base_used_mb:
        return float(base_used_mb) + float(d)
    return float(row.get("peak_mb") or 0.0)


def choose(rows: dict, total_mb, base_used_mb=None, cap: float = None) -> dict:
    """从实测里挑一档。rows = {"1x4": {workers,batch,fps,swap_p50,peak_mb,delta_mb}, ...}

    返回 {"pick": {...}|None, "rows": [...], "rejected": [...], "line": "...", "why": [...]}
    """
    cap = float(CAP if cap is None else cap)
    tot = float(total_mb or 0.0)
    picks, why = [], []
    for label, r in (rows or {}).items():
        try:
            w, b = int(r.get("workers") or 1), int(r.get("batch") or 1)
            fps = float(r.get("fps") or 0.0)
        except (TypeError, ValueError):
            continue
        if fps <= 0:
            continue
        peak = _est_peak(r, base_used_mb)
        pct = (100.0 * peak / tot) if tot > 0 else 0.0
        picks.append({"label": label, "workers": w, "batch": b, "fps": round(fps, 2),
                      "swap_p50": (None if r.get("swap_p50") is None
                                   else round(float(r["swap_p50"]), 1)),
                      "peak_mb": round(peak), "pct": round(pct, 1),
                      "inflight": w * b, "ok": True, "why": ""})
    if not picks:
        return {"pick": None, "rows": [], "rejected": [],
                "line": "", "why": ["没有实测"]}
    picks.sort(key=lambda p: (p["inflight"], p["workers"], p["batch"]))
    # ① 显存上限：实测峰值 > 85% 的档一律不选（那是换页档，帧速本来也会塌）
    for p in picks:
        if tot > 0 and p["pct"] > 100.0 * HARD:
            p["ok"] = False
            p["why"] = "显存 %.0f%% > 换页线 %.0f%%" % (p["pct"], 100.0 * HARD)
            why.append("%s 出局：显存 %.0f%% > 换页线 %.0f%%"
                       % (p["label"], p["pct"], 100.0 * HARD))
    # ② 争用：2 个 worker 的 swap p50 > 1.4 × 1 个 worker 的**最好值**，**并且**它并没有
    # 更快（不到 1 个 worker 的 1.2 倍）→ 判为争用/换页，不用 2 worker。
    # 为什么两个条件都要：2 份会话共用一块 GPU，单帧 swap 时间本来就会涨（那是排队，
    # 不是换页）；416 实测 2×2 是 13.4 帧/秒、swap 294ms vs 1×2 215ms（1.37×）—— 只看
    # swap 会把**更快**的那档否掉。真换页的特征是「swap 涨了、帧速还塌了」（384 的 2×2：
    # swap 952ms = 4.5×，帧速 3.49）。
    one = [p for p in picks if p["workers"] == 1 and p["swap_p50"]]
    if one:
        best1 = min(float(p["swap_p50"]) for p in one)
        fast1 = max(float(p["fps"]) for p in one)
        for p in picks:
            if p["workers"] >= 2 and p["swap_p50"]:
                if (float(p["swap_p50"]) > CONTEND * best1
                        and float(p["fps"]) < CONTEND_WORTH * fast1):
                    p["ok"] = False
                    p["why"] = ("2 worker 的 swap p50 %.0fms > %.1f× 1 worker 的 %.0fms，"
                                "帧速也没更快（%.2f vs %.2f）"
                                % (float(p["swap_p50"]), CONTEND, best1,
                                   float(p["fps"]), fast1))
                    why.append("%s 出局：%s" % (p["label"], p["why"]))
    # ③ 相邻档：比低一档（在飞帧数更少的那档）慢 10% 以上 → 跳过（噪音/换页）
    prev = None
    for p in picks:
        if prev is not None and prev.get("ok") and prev["fps"] > 0:
            if p["fps"] < (1.0 - STEP_LOSS) * float(prev["fps"]):
                p["ok"] = False
                p["why"] = ("比 %s 的 %.2f 帧/秒还慢 %.0f%%"
                            % (prev["label"], float(prev["fps"]),
                               100.0 * (1.0 - p["fps"] / float(prev["fps"]))))
                why.append("%s 出局：%s" % (p["label"], p["why"]))
        if p.get("ok"):
            prev = p
    ok = [p for p in picks if p.get("ok")]
    # ④ 85% 是**目标**，不是唯一门槛：85% 内最快的档如果比「85%~92%（换页线以内）」
    # 那一档慢 10% 以上，选后者并说明 —— 否则「桌面占了 1.6GB」的时候会把 1×4 判死、
    # 退到 1×2，白丢 10% 帧速（那是把安全边际当成了目标本身）。
    soft = [p for p in ok if tot <= 0 or p["pct"] <= 100.0 * cap]
    hard = [p for p in ok if tot <= 0 or p["pct"] <= 100.0 * HARD]
    pick, why_pick = None, ""
    if soft:
        best = max(soft, key=lambda p: p["fps"])
        spill = [p for p in hard if p["label"] not in {q["label"] for q in soft}
                 and p["fps"] > (1.0 + STEP_LOSS) * float(best["fps"])]
        if spill:
            hi = max(spill, key=lambda p: p["fps"])
            pick = hi
            why_pick = ("%s 的显存 %.0f%% 超过 %d%%，但比 %d%% 内最快的 %s（%.2f 帧/秒）"
                        "还快 %.0f%%，且仍在换页线 %d%% 以内"
                        % (hi["label"], hi["pct"], int(round(100 * cap)), int(round(100 * cap)),
                           best["label"], float(best["fps"]),
                           100.0 * (float(hi["fps"]) / max(1e-6, float(best["fps"])) - 1.0),
                           int(round(100 * HARD))))
        else:
            pick = best
            why_pick = ("%s 在 %d%% 显存以内最快（%.2f 帧/秒，显存 %.0f%%）"
                        % (best["label"], int(round(100 * cap)), float(best["fps"]), best["pct"]))
            if spill:
                pass
    elif hard:
        pick = max(hard, key=lambda p: p["fps"])
        why_pick = ("没有一档落在 %d%% 以内 → 取换页线 %d%% 以内最快的 %s（%.2f 帧/秒，显存 %.0f%%）"
                    % (int(round(100 * cap)), int(round(100 * HARD)),
                       pick["label"], float(pick["fps"]), pick["pct"]))
        why.append(why_pick)
    return {"pick": pick, "rows": picks, "rejected": [p for p in picks if not p.get("ok")],
            "why": why, "why_pick": why_pick, "cap": cap, "hard": HARD,
            "total_mb": (tot or None),
            "base_used_mb": (None if base_used_mb is None else round(float(base_used_mb)))}


def note(rows: dict, res: dict, label: str = "[project]") -> str:
    """一句话说清这一单凭什么选这一档（形状按用户给的那行）。"""
    parts = []
    for p in (res.get("rows") or []):
        tag = "" if p.get("ok") else "✗"
        sw = (" %.0fms" % float(p["swap_p50"])) if p.get("swap_p50") else ""
        parts.append("%s%s=%.1f fps%s" % (p["label"], tag, float(p["fps"]), sw))
    if not parts:
        return "%s 基准：没有实测 → 保守默认" % label
    pick = res.get("pick") or {}
    tail = []
    if pick:
        one = [p for p in (res.get("rows") or [])
               if p["workers"] == 1 and p.get("swap_p50")]
        two = [p for p in (res.get("rows") or [])
               if p["workers"] >= 2 and p.get("swap_p50")]
        if one and two:
            tail.append("%s swap p50 %.0fms vs %s %.0fms"
                        % (two[0]["label"], float(two[0]["swap_p50"]),
                           one[0]["label"], float(one[0]["swap_p50"])))
        tail.append("取 %s，显存 %.0f%%" % (pick["label"], float(pick["pct"])))
    else:
        tail.append("全部出局（%s）" % ("；".join(res.get("why") or []) or "没有合格档"))
    return "%s 基准：%s（%s）" % (label, " / ".join(parts), "，".join(tail))


def conservative(model_mb) -> dict:
    """没有实测时的保守档（和改之前那套静态规则一致，不算新判据）。

    ≥1GB → 1×4（一份会话 + 4 线程）；≤600MB → 2×2（两份小会话）；中间档 → 2 worker
    （线程 4//2=2），装不下由显存预算砍。
    """
    mb = float(model_mb or 0.0)
    if mb >= 1024:
        return {"workers": 1, "batch": 4, "why": "保守默认：权重 ≥1GB → 1×4（还没量过）"}
    if 0.0 < mb <= 600:
        return {"workers": 2, "batch": 2, "why": "保守默认：权重 ≤600MB → 2×2（还没量过）"}
    return {"workers": 2, "batch": 2, "why": "保守默认：中间档 → 2×2（还没量过）"}
