# -*- coding: utf-8 -*-
"""把旧的 aligned/_by_id/g00|g01…. 拍平成 aligned/id001|id002…（一次性迁移）。

背景：人脸身份分组早期写到 _by_id/g00、g01，
现在的规则是直接放在 aligned 下、命名为 id001 起。

用法（在 4.1 根目录执行）：
    runtime\\server\\Scripts\\python.exe app\\tools\\migrate_by_id.py --dry-run
    runtime\\server\\Scripts\\python.exe app\\tools\\migrate_by_id.py

安全性：只做移动，不删除任何图片；目标重名时自动加 _1 后缀。
"""
from __future__ import annotations

import argparse
from pathlib import Path

EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="workspace/data_src/aligned", help="aligned 目录")
    ap.add_argument("--dry-run", action="store_true", help="只打印计划，不动文件")
    a = ap.parse_args(argv)

    root = Path(a.dir)
    src = root / "_by_id"
    if not src.is_dir():
        print("没有 _by_id，无需迁移:", src)
        return 0

    groups = sorted(d for d in src.iterdir() if d.is_dir())
    if not groups:
        print("_by_id 是空的")
        return 0

    # 组按张数从多到少编号：id001 永远张数最多
    groups.sort(key=lambda d: -len([p for p in d.iterdir() if p.is_file() and p.suffix.lower() in EXTS]))
    width = max(3, len(str(len(groups))))
    moved = 0
    for idx, g in enumerate(groups, 1):
        name = "noise" if g.name.lower() == "noise" else ("id%0" + str(width) + "d") % idx
        dst_dir = root / name
        files = [p for p in g.iterdir() if p.is_file() and p.suffix.lower() in EXTS]
        print("%-24s -> %-8s %d 张" % (g.name, name + "/", len(files)))
        if a.dry_run:
            continue
        dst_dir.mkdir(exist_ok=True)
        for p in files:
            dst = dst_dir / p.name
            k = 1
            while dst.exists():
                dst = dst_dir / ("%s_%d%s" % (p.stem, k, p.suffix))
                k += 1
            p.rename(dst)
            moved += 1
        try:
            g.rmdir()                       # 只有空了才成功
        except OSError:
            pass

    if a.dry_run:
        print("\n(dry-run) 以上是计划，加 --dry-run 去掉就会真的搬")
        return 0
    print("\n搬运完成: %d 张" % moved)
    rest = [p for p in src.iterdir() if p.is_file() and p.suffix.lower() in EXTS]
    if not rest:
        try:
            src.rmdir()
            print("_by_id 已清空并删除")
        except OSError:
            pass
    else:
        print("_by_id 里还剩 %d 张（可能是子目录里的东西），保留" % len(rest))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
