# -*- coding: utf-8 -*-
r"""批目标滞回（DFL_PRED_BATCH_HYST）的状态机验收：不需要 GPU，喂合成台阶就行。

为什么要这么测：真机上「量测差多少」是随机的，靠它证明不了规则对 —— 规则本身
（死区 / 升降档方向 / 连续确认 / 换阶梯）要能用确定的输入钉住。

口径（和 vram._pred_pick_batch 里调用它的顺序一模一样）：
    want = 旧公式（每张耗时 <= 最快值 * 1.05 里最大的批）
    pick, note = _pred_pick_hyst(key, rows, want)   →   _pred_pick_put(key, ...)
    旧行为（HYST=0）就是「不看 state，直接用 want」。

跑法（仓库根，server venv）：
    runtime\server\Scripts\python.exe app\tools\accept\pred_batch_hyst_accept.py
退出码 0 = 全过。
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
for p in (str(REPO), str(REPO / "app")):
    if p not in sys.path:
        sys.path.insert(0, p)
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

from server import app as _app_mod                                # noqa: E402,F401
from server.backend_merge import vram as V                        # noqa: E402

ACC = []
TMP = Path(tempfile.mkdtemp(prefix="dfl_hyst_"))
CACHE = TMP / "_pred_batch_pick.json"


def check(ok, name, detail=""):
    ACC.append(bool(ok))
    print(("[PASS] " if ok else "[FAIL] ") + name + ("  · " + str(detail) if detail else ""),
          flush=True)
    return ok


def rows_of(*per):
    return [{"batch": b, "ms": round(p * b, 1), "per_face_ms": float(p)} for b, p in per]


def want_of(rows):
    best = min(r["per_face_ms"] for r in rows)
    ok = [r for r in rows if r["per_face_ms"] <= best * 1.05]
    return max(ok, key=lambda r: r["batch"])


def reset():
    V._PRED_PICK_MEM = None
    if CACHE.exists():
        CACHE.unlink()


def step(key, rows):
    """走一遍生产里的那两步：算 want → 过滞回 → 落盘。"""
    want = want_of(rows)
    pick, note = V._pred_pick_hyst(key, rows, want)
    V._pred_pick_put(key, {"batch": int(pick["batch"]),
                           "per_face_ms": round(float(pick["per_face_ms"]), 2),
                           "challenger": (int(want["batch"])
                                          if int(want["batch"]) != int(pick["batch"]) else 0),
                           "streak": int(note.get("streak") or 0)})
    return int(pick["batch"]), int(want["batch"]), note


def main():
    # 把缓存指到临时目录：验收不该碰用户的 workspace
    V._pred_pick_file = lambda: CACHE
    print("缓存（临时）：%s" % CACHE)
    print("环境：DFL_PRED_BATCH_HYST=%r · 死区 %.0f%% · 升档连续 %d 次"
          % (os.environ.get(V._PRED_HYST_ENV), 100.0 * V._PRED_HYST_BAND, V._PRED_HYST_UP_N))

    # ① 没有历史 → 用本次实测（init），并且落盘
    reset()
    pick, want, note = step("m|416|8", rows_of((1, 35.1), (4, 41.0), (8, 37.9)))
    check(pick == 1 and want == 1 and note["action"] == "init",
          "无历史：按本次实测（B8 慢 8% → 选 1）", "pick=%d want=%d %s" % (pick, want, note["action"]))
    check(CACHE.is_file() and json.loads(CACHE.read_text("utf-8")).get("m|416|8", {}).get("batch") == 1,
          "决定落了盘（跨启动要用的就是它）", str(CACHE.name))

    # ② 核心：历史选 8，这次量出 8 比 1 慢 7.6%（< 10% 死区）→ 沿用 8
    reset()
    V._pred_pick_put("m|416|8", {"batch": 8, "per_face_ms": 37.9, "challenger": 0, "streak": 0})
    pick, want, note = step("m|416|8", rows_of((1, 35.1), (8, 37.9)))
    check(pick == 8 and want == 1 and note["action"] == "band-keep",
          "死区内沿用上次（want 想换 1，但差 7.4% < 死区 → 保持 8）",
          "pick=%d want=%d gain=%.1f%%" % (pick, want, note["gain_pct"]))
    check(want == 1, "旧行为会选 1（说明这条门确实拦住了抖动）", "旧公式 want=%d" % want)

    # ③ 反向同理：历史选 1，这次量出 8 稍快（1.4%）→ 也是死区，保持 1
    reset()
    V._pred_pick_put("m|416|8", {"batch": 1, "per_face_ms": 36.8, "challenger": 0, "streak": 0})
    pick, want, note = step("m|416|8", rows_of((1, 36.8), (8, 36.3)))
    check(pick == 1 and note["action"] == "band-keep",
          "死区内沿用上次（另一头：差 1.4% → 保持 1）", "pick=%d gain=%.1f%%" % (pick, note["gain_pct"]))

    # ④ 降档方向：小档真的快 27% → 立刻切（不等连续）
    reset()
    V._pred_pick_put("m|416|8", {"batch": 8, "per_face_ms": 48.0, "challenger": 0, "streak": 0})
    pick, want, note = step("m|416|8", rows_of((1, 35.0), (8, 48.0)))
    check(pick == 1 and note["action"] == "down-switch",
          "小档明显更快（27%）→ 立刻降档", "pick=%d gain=%.1f%%" % (pick, note["gain_pct"]))

    # ⑤ 升档方向：大档快 20%，但要连续 3 次
    reset()
    V._pred_pick_put("m|416|8", {"batch": 1, "per_face_ms": 40.0, "challenger": 0, "streak": 0})
    up = rows_of((1, 40.0), (8, 32.0))
    p1, _w, n1 = step("m|416|8", up)
    p2, _w, n2 = step("m|416|8", up)
    p3, _w, n3 = step("m|416|8", up)
    check(p1 == 1 and p2 == 1 and n1["action"] == "up-wait" and n2["action"] == "up-wait",
          "升档要先等（第 1/2 次都还是 1）", "streak=%s/%s" % (n1.get("streak"), n2.get("streak")))
    check(p3 == 8 and n3["action"] == "up-switch",
          "第 %d 次连续明显更快 → 升到 8" % V._PRED_HYST_UP_N, "streak=%s" % n3.get("streak"))

    # ⑥ 换阶梯（显存挤了，只探到 1/4）：缓存里的 8 不在台阶上 → 重新判
    reset()
    V._pred_pick_put("m|416|8", {"batch": 8, "per_face_ms": 37.9, "challenger": 0, "streak": 0})
    pick, want, note = step("m|416|8", rows_of((1, 35.0), (4, 43.0)))
    check(pick == 1 and note["action"] == "reladder",
          "台阶里没有上次的批 → 重新按实测判（不会拿 8 去用）",
          "pick=%d action=%s" % (pick, note["action"]))

    # ⑦ 缓存坏了不影响决策（读失败当成没有历史）
    reset()
    CACHE.write_text("{ 这不是 json", "utf-8")
    V._PRED_PICK_MEM = None
    pick, want, note = step("m|416|8", rows_of((1, 35.1), (8, 37.9)))
    check(pick == 1 and note["action"] == "init",
          "缓存文件坏了 → 当没有历史，照常给结论", "pick=%d %s" % (pick, note["action"]))

    # ⑧ 开关关掉时：连 state 都不读（旧行为逐字节相同）
    old_on, old_put = V._pred_hyst_on, V._pred_pick_put
    called = {"put": 0}

    def _no_put(*a, **k):
        called["put"] += 1
    V._pred_hyst_on = lambda: False
    V._pred_pick_put = _no_put
    try:
        want = want_of(rows_of((1, 35.1), (8, 37.9)))
        pick = want                      # HYST=0 时 _pred_pick_batch 直接 return want
        check(pick["batch"] == 1 and called["put"] == 0,
              "HYST=0：不读不写，直接按本次实测", "pick=%d" % pick["batch"])
    finally:
        V._pred_hyst_on, V._pred_pick_put = old_on, old_put

    # ⑨ 落盘格式：一个 key 一行，带 at/ladder 之类的现场（人看得懂）
    reset()
    step("m|416|8", rows_of((1, 35.1), (8, 37.9)))
    blob = json.loads(CACHE.read_text("utf-8"))
    check(isinstance(blob, dict) and "batch" in list(blob.values())[0],
          "缓存是可读的 dict（按 key 分组）", json.dumps(list(blob)[:1], ensure_ascii=False))

    print("")
    print("=" * 78)
    print("%d 项检查 · 全过" % len(ACC) if all(ACC) else "有检查没通过")
    return 0 if all(ACC) else 2


if __name__ == "__main__":
    sys.exit(main())
