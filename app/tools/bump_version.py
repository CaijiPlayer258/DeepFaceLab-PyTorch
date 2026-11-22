# -*- coding: utf-8 -*-
r"""版本号生成：4.1-<MMDD>，同一天第二个版本 4.1-<MMDD>-b2。

规则
----
    4.1-1122        11 月 22 日的第一个版本
    4.1-1122-b2     同一���的第二个版本（b3、b4 依次类推）

* 只有日期，不带时分 —— 一天之内通常只发一版；真发多版用 -bN 区分。
* MMDD 固定四位（补零），所以字符串序 = 时间序，更新检查直接比元组。
* 关于页显示成 v4.1 · 11-22（当天多版再加 B2）。

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


def make_version(when=None, base: str = BASE, build: int = 1) -> str:
    t = when or _dt.datetime.now()
    v = "%s-%02d%02d" % (base, t.month, t.day)
    return v if build <= 1 else "%s-b%d" % (v, build)


def parse_version(v: str) -> dict:
    m = _FMT.match((v or "").strip())
    if not m:
        return {"version": v, "base": v, "build": 1, "human": "v" + v}
    base, mmdd, b = m.groups()
    build = int(b or 1)
    human = "v%s · %s-%s" % (base, mmdd[:2], mmdd[2:])
    if build > 1:
        human += " · B%d" % build
    return {"version": v, "base": base, "build": build, "human": human}


def next_build(today: _dt.date, current: str) -> int:
    """当天已经发过就用下一个 b 号。"""
    m = _FMT.match((current or "").strip())
    if not m:
        return 1
    base, mmdd, b = m.groups()
    if int(mmdd[:2]) != today.month or int(mmdd[2:]) != today.day:
        return 1
    return int(b or 1) + 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--at", default="", help="指定日期，如 2026-12-01")
    ap.add_argument("--base", default=BASE)
    ap.add_argument("--build", type=int, default=0, help="b 号；不填则自动")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    when = _dt.datetime.fromisoformat(args.at) if args.at else _dt.datetime.now()
    cur = ""
    try:
        cur = VERSION_FILE.read_text(encoding="utf-8-sig").strip()
    except Exception:
        pass
    build = args.build or next_build(when.date(), cur)
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