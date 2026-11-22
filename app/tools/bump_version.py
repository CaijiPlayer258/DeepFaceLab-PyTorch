# -*- coding: utf-8 -*-
r"""版本号生成：4.1.<YY>.<MMDD>.<HHMM>

规则
----
    MAJOR . MINOR . YY . MMDD . HHMM
    4     . 1     . 26 . 1122 . 2304      = 2026-11-22 23:04

为什么这么切
------------
* 每段**固定宽度**（YY 两位、MMDD 四位、HHMM 四位），字典序就是时间序，
  更新检查用 tuple 比较即可，不需要额外解析日期。
  反例：不补零的 12 月 1 日 = 121 < 11 月 22 日 = 1122，顺序会乱。
* 保留 MAJOR.MINOR 作为大版本语义（4.1 这条线），后面纯粹是构建时间。
* 版本号能反推时间（parse_version），关于页显示成 v4.1 加日期时分。

用法
----
    python app/tools/bump_version.py            # 用当前时间
    python app/tools/bump_version.py --dry-run
    python app/tools/bump_version.py --at 2026-11-22T23:04
"""
from __future__ import annotations

import argparse
import datetime as _dt
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
VERSION_FILE = ROOT / "version.txt"
BASE = "4.1"

_FMT = re.compile(r"^(\d+)\.(\d+)\.(\d{2})\.(\d{4})\.(\d{4})$")


def make_version(when=None, base: str = BASE) -> str:
    t = when or _dt.datetime.now()
    return "%s.%02d.%02d%02d.%02d%02d" % (base, t.year % 100, t.month, t.day, t.hour, t.minute)


def parse_version(v: str) -> dict:
    """4.1.26.1122.2304 -> {base, when, human}；解析不了就原样返回。"""
    m = _FMT.match((v or "").strip())
    if not m:
        return {"base": v, "when": None, "human": v}
    major, minor, yy, mmdd, hhmm = m.groups()
    try:
        when = _dt.datetime(2000 + int(yy), int(mmdd[:2]), int(mmdd[2:]),
                            int(hhmm[:2]), int(hhmm[2:]))
    except ValueError:
        return {"base": v, "when": None, "human": v}
    return {"base": "%s.%s" % (major, minor), "when": when,
            "human": "v%s.%s  |  %s" % (major, minor, when.strftime("%Y-%m-%d %H:%M"))}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--at", default="", help="指定时间，如 2026-11-22T23:04")
    ap.add_argument("--base", default=BASE, help="大版本前缀，默认 4.1")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    when = _dt.datetime.fromisoformat(args.at) if args.at else None
    v = make_version(when, args.base)
    info = parse_version(v)
    print("version :", v)
    print("human   :", info["human"])
    if not args.dry_run:
        VERSION_FILE.write_text(v + "\n", encoding="utf-8")
        print("written :", VERSION_FILE)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())