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


def _today_release_commits(today: _dt.date) -> list:
    """今天所有 release 提交：[(短 sha, 标题)]。

    **不用 git 的 --grep 过滤**：'^release:' 这种锚定会被行首的 BOM 打断 ——
    发版时标题前如果混进一个 \ufeff，这一条就数不到、b 号退一格，而没人会发现
    （2026-11-28 就发生过一次）。这里把标题整行取回来，在 Python 侧 strip 再判。
    """
    import subprocess
    try:
        out = subprocess.run(
            ["git", "log", "--since=%s 00:00:00" % today.isoformat(),
             "--until=%s 23:59:59" % today.isoformat(),
             "--pretty=%h\t%s"],
            cwd=str(ROOT), capture_output=True, text=True,
            encoding="utf-8", errors="replace", timeout=15)
    except Exception:
        return []
    hits = []
    for line in (out.stdout or "").splitlines():
        if "\t" not in line:
            continue
        sha, subj = line.split("\t", 1)
        subj = subj.lstrip("\ufeff").strip()
        if subj.startswith("release: " + BASE + "-"):
            hits.append((sha.strip(), subj))
    return hits


def _count_today_releases(today: _dt.date) -> int:
    """数 git 里今天有几个 release 提交（标题形如 release: 4.1-1128-bN）。"""
    return len(_today_release_commits(today))


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


def _verify() -> int:
    """核对三件事：version.txt / 本地 tag / 当天 release 提交数。

    为什么要它：b 号是靠「当天有几次 release: 提交」数出来的。任何一次标题写歪
    （行首 BOM、手滑、amend 掉、或者直接把 tag 打在手写的字符串上），编号就退一格
    或跳一格 —— 而整条发布流程里没有任何地方会发现。发完版跑一次这个，比事后
    翻 100 个 release 快得多。
    """
    import subprocess

    cur = ""
    try:
        cur = VERSION_FILE.read_text(encoding="utf-8-sig").strip().splitlines()[0].strip()
    except Exception as e:
        print("version.txt  读不到：%s" % e)
        return 2

    m = _FMT.match(cur)
    if not m:
        print("version.txt  : %-20s  NG（不符合 4.1-MMDD[-bN]）" % cur)
        return 1
    base, mmdd, b = m.groups()
    build = int(b or 0)
    now = _dt.date.today()
    try:
        day = _dt.date(now.year, int(mmdd[:2]), int(mmdd[2:]))
    except Exception:
        print("version.txt  : %-20s  NG（日期段不合法）" % cur)
        return 1

    fails = []
    print("version.txt  : %-20s  ok" % cur)

    # 1) 本地 tag
    tag = "v" + cur
    try:
        out = subprocess.run(["git", "tag", "--list", tag], cwd=str(ROOT),
                             capture_output=True, text=True,
                             encoding="utf-8", errors="replace", timeout=15)
        has_tag = bool((out.stdout or "").strip())
    except Exception:
        has_tag = False
    if has_tag:
        print("本地 tag     : %-20s  ok" % tag)
    else:
        print("本地 tag     : %-20s  NG（没有这个 tag）" % tag)
        fails.append("tag")

    # 2) 同一 MMDD 下的 tag：b 号必须从 b0 起、连续、且最新那个就是 version.txt
    #    按 **tag** 数，不按 commit 数 —— commit 标题会被行首 BOM / amend 影响
    #    （11-28 就漏过一次），而且跨日发版时「今天」和版本号里的日期根本不是一天
    #    （1128 发 1129 的号，按今天去数必然是 0 条）。
    try:
        out = subprocess.run(["git", "tag", "--list", "v%s-%s-b*" % (base, mmdd)],
                             cwd=str(ROOT), capture_output=True, text=True,
                             encoding="utf-8", errors="replace", timeout=15)
        tags = [x.strip() for x in (out.stdout or "").splitlines() if x.strip()]
    except Exception:
        tags = []
    builds = []
    for t in tags:
        tm = _FMT.match(t[1:] if t.startswith("v") else t)
        if tm and tm.group(2) == mmdd:
            builds.append(int(tm.group(3) or 0))
    builds.sort()
    print("同日 tag     : %d 个（%s）" % (len(builds),
          ", ".join("b%d" % x for x in builds) or "—"))
    if not builds:
        print("              NG（v%s-%s-b* 一个都没有）" % (base, mmdd))
        fails.append("tags")
    elif builds != list(range(len(builds))):
        print("              NG（b 号不连续：%s）" % ", ".join("b%d" % x for x in builds))
        fails.append("seq")
    elif build != len(builds) - 1:
        print("              NG（共 %d 个 → 最新应是 b%d，version.txt 写的是 b%d）"
              % (len(builds), len(builds) - 1, build))
        fails.append("count")

    # 3) 参考信息：今天有几条 release 提交（只打印，不判错）
    try:
        print("今天提交     : %d 条 release 提交（仅参考）"
              % len(_today_release_commits(_dt.date.today())))
    except Exception:
        pass

    print("-" * 46)
    if fails:
        print("VERIFY FAILED：%s" % ", ".join(fails))
        print("  · tag 缺 → git tag v%s && git push origin v%s" % (cur, cur))
        print("  · 计数不对 → 检查当天 release 提交的标题（会不会混了 BOM 或被 amend 掉）")
        return 1
    print("VERIFY OK")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--at", default="", help="指定日期，如 2026-12-01")
    ap.add_argument("--base", default=BASE)
    # ★ 不提供 --build：b 号就是「当天第几个版本」，由 next_build 自己数。
    #   以前留了这个口子，结果被一路手工传成 b55/b56/…/b70 —— 11 月 28 日当天
    #   只有三个版本，号却写着 b70，看起来像发了 69 次。
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--verify", action="store_true",
                    help="核对 version.txt / 本地 tag / 当天 release 提交数 三者是否一致")
    args = ap.parse_args()

    if args.verify:
        return _verify()

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