# -*- coding: utf-8 -*-
r"""版本号生成：4.1-<MMDD>-b<N>，N = **当天第几个版本，从 0 开始**。

规则
----
    4.1-1129-b0     11 月 29 日发的第 1 个版本
    4.1-1129-b1     同一天的第 2 个（b2、b3 依次类推）

* 只有日期，不带时分 —— 一天之内通常只发一版；真发多版用 -bN 区分。
* **b 号永远带、从 0 起**：以前「当天第一版不带后缀、第二个叫 b2」，
  于是 1126 出现 b0/b1/b2/b3 和 1127 从 b4 一路排到 b67 两种写法混在一起，
  历史 tag 全乱。现在统一 0 基，字符串序仍 = 时间序，
  解析时「不带 -bN」按 b0 读（兼容旧 tag）。
* MMDD 固定四位（补零），所以字符串序 = 时间序，更新检查直接比元组。
* 关于页显示成 v4.1 · 11-29（当天多版再加 B1）。

用法
----
    python app/tools/bump_version.py              # 今天第一版；当天已有就自动 +1
    python app/tools/bump_version.py --build 2    # 指定 b 号
    python app/tools/bump_version.py --at 2026-12-01 --dry-run
"""
from __future__ import annotations

import argparse
import datetime as _dt
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
VERSION_FILE = ROOT / "version.txt"
BASE = "4.1"

_FMT = re.compile(r"^(\d+\.\d+)-(\d{4})(?:-b(\d+))?$")


def make_version(when=None, base: str = BASE, build: int = 0) -> str:
    t = when or _dt.datetime.now()
    v = "%s-%02d%02d" % (base, t.month, t.day)
    return "%s-b%d" % (v, max(0, int(build)))


def parse_version(v: str) -> dict:
    m = _FMT.match((v or "").strip())
    if not m:
        return {"version": v, "base": v, "build": 0, "human": "v" + v}
    base, mmdd, b = m.groups()
    # 不带 -bN 的旧 tag 按 b0 读（0 基之后，无后缀就等于当天第一个版本）
    build = int(b or 0)
    human = "v%s · %s-%s" % (base, mmdd[:2], mmdd[2:])
    if b is not None:
        human += " · B%d" % build
    return {"version": v, "base": base, "build": build, "human": human}


def _count_today_releases(today: _dt.date) -> int:
    """数 git 里今天有几个 release 提交（标题形如 release: 4.1-1128-bN）。"""
    try:
        import subprocess
        out = subprocess.run(
            ["git", "log", "--since=%s 00:00:00" % today.isoformat(),
             "--until=%s 23:59:59" % today.isoformat(),
             "--pretty=%s", "--grep", "^release: 4\\.1-"],
            cwd=str(ROOT), capture_output=True, text=True, timeout=10)
        lines = [x for x in (out.stdout or "").splitlines() if x.strip()]
        return len(lines)
    except Exception:
        return 0


def next_build(today: _dt.date, current: str) -> int:
    """当天已经发过就用下一个 b 号。

    ★ 自愈：如果 version.txt 里的 b 号明显不合理（比如当天日期却写着 b70），
    按 git 里同一天的 release 提交数重新推。这样即使以前手工传歪了，下一次也会归位。
    """
    m = _FMT.match((current or "").strip())
    if not m:
        return 0
    base, mmdd, b = m.groups()
    if int(mmdd[:2]) != today.month or int(mmdd[2:]) != today.day:
        return 0                      # 换了一天 → 从 b0 重新数
    cur_b = int(b or 0)
    # ★ 自愈：手里的号比「当天真实发布数」还大 → 说明是被手工传歪的历史包袱，
    #   按当天真实数重排。（比如 11-28 当天只发了 3 版，version.txt 却写着 b70。）
    real = _count_today_releases(today)
    if real > 0 and cur_b > real:
        return real
    return cur_b + 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--at", default="", help="指定日期，如 2026-12-01")
    ap.add_argument("--base", default=BASE)
    # ★ 不提供 --build：b 号就是「当天第几个版本」，由 next_build 自己数。
    #   以前留了这个口子，结果被一路手工传成 b55/b56/…/b70 —— 11 月 28 日当天
    #   只有三个版本，号却写着 b70，看起来像发了 69 次。
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    when = _dt.datetime.fromisoformat(args.at) if args.at else _dt.datetime.now()
    cur = ""
    try:
        cur = VERSION_FILE.read_text(encoding="utf-8-sig").strip()
    except Exception:
        pass
    build = next_build(when.date(), cur)
    v = make_version(when, args.base, build)
    info = parse_version(v)
    print("version :", v)
    print("human   :", info["human"])
    if not args.dry_run:
        VERSION_FILE.write_text(v + "\n", encoding="utf-8")
        print("written :", VERSION_FILE)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())