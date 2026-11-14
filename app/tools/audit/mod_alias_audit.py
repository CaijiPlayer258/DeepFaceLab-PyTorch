# -*- coding: utf-8 -*-
r"""拆分后遗症体检：本项目内部的「模块别名.名字」是否真的存在。

为什么需要它（2026-xx 的真实故障）：routes_merge.py 拆成 backend_merge/* 之后，
pipeline_project.py 里保留了拆分前的写法

    frame = rm._app_mod._imread_any(str(src_p))        # rm = routes_merge（shim）

而 shim 的再导出清单里没有 _app_mod。**import 检查抓不到这一条**：
  · compileall / import 只验证「模块能导入、函数能定义」，rm._app_mod 是运行到那一行
    才发生的属性查找，导入阶段一次都不会碰；
  · 拆分时做的 AST 对拍也没用：这一行在拆分前后**字符相同**（rm 原来就是 routes_merge
    自己），变的是「rm 到底还有没有这个属性」；
  · 它外面套着读帧线程的 except Exception → 12 帧全被记成「这一帧读不出来（损坏？）」，
    收尾只报「merged/ 里一帧都没有」，真实原因是 AttributeError，一点痕迹都没有。

本脚本就是补这一格：把所有 <别名>.<名字>（别名指向本项目模块）抽出来逐个 hasattr，
跑通 = 拆分没漏符号。只查本项目（server.*）之间的引用 —— torch/os 那种平台差异不管。

跑法（仓库根）：
    runtime\server\Scripts\python.exe app\tools\audit\mod_alias_audit.py
    （换目录：--root app/server；只报某一层：--only backend_merge）
退出码 0 = 没有缺失；1 = 有缺失（逐条列文件:行 + 别名指向）。
"""
from __future__ import annotations

import argparse
import ast
import importlib
import importlib.util
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
APP = REPO / "app"


def _mod_name(p: Path):
    rel = p.relative_to(APP).with_suffix("")
    parts = list(rel.parts)
    is_pkg = parts[-1] == "__init__"
    if is_pkg:
        parts.pop()
    return ".".join(parts), is_pkg


def _get_mod(dotted: str):
    m = sys.modules.get(dotted)
    if m is not None:
        return m
    try:
        return importlib.import_module(dotted)
    except Exception:                                   # noqa: BLE001
        return None


def _scan(root: Path):
    bad, files, pairs = [], 0, 0
    for p in sorted(root.rglob("*.py")):
        if "__pycache__" in p.parts:
            continue
        files += 1
        try:
            tree = ast.parse(p.read_text(encoding="utf-8"))
        except SyntaxError as e:
            bad.append((str(p.relative_to(APP)), 0, "SYNTAX", str(e)))
            continue
        pkg, is_pkg = _mod_name(p)
        pkg_dir = pkg if is_pkg else pkg.rsplit(".", 1)[0]
        aliases = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                base = "." * node.level + (node.module or "")
                try:
                    absmod = (importlib.util.resolve_name(base, pkg_dir)
                              if base else pkg_dir)
                except ValueError:
                    continue
                for a in node.names:
                    if a.name == "*":
                        continue
                    m = _get_mod(absmod + "." + a.name)
                    if m is not None:
                        aliases[a.asname or a.name] = (absmod + "." + a.name, m)
            elif isinstance(node, ast.Import):
                for a in node.names:
                    target = a.asname or a.name.split(".")[0]
                    m = _get_mod(a.name if a.asname else target)
                    if m is not None:
                        aliases[target] = (m.__name__, m)
        if not aliases:
            continue
        seen = set()
        for node in ast.walk(tree):
            if not (isinstance(node, ast.Attribute)
                    and isinstance(node.value, ast.Name)):
                continue
            alias, attr = node.value.id, node.attr
            if alias not in aliases or (alias, attr) in seen:
                continue
            modname, mobj = aliases[alias]
            if not modname.startswith("server"):
                continue                     # 第三方 / 引擎侧不查（平台差异一堆）
            seen.add((alias, attr))
            pairs += 1
            if not hasattr(mobj, attr):
                bad.append((str(p.relative_to(APP)), node.lineno,
                            alias + "." + attr, modname))
    return files, pairs, bad


def main(argv=None):
    ap = argparse.ArgumentParser(description="模块别名属性体检（拆分后遗症）")
    ap.add_argument("--root", default=str(APP / "server"),
                    help="要扫的目录（默认 app/server）")
    ap.add_argument("--only", default="", help="只扫路径里含这个片段文件")
    args = ap.parse_args(argv)
    root = Path(args.root)
    if not root.is_absolute():
        root = REPO / root

    if str(APP) not in sys.path:
        sys.path.insert(0, str(APP))
    import server.app                                 # noqa: F401  走真入口：环上的 import 顺序只有它成立

    files, pairs, bad = _scan(root)
    if args.only:
        bad = [b for b in bad if args.only in b[0]]
    print("扫描 %d 个文件 · 检查 %d 处 <模块别名>.<名字>" % (files, pairs))
    print("缺失：%d" % len(bad))
    for f, ln, attr, mod in bad:
        print("  %s:%s  %s   (别名指向 %s)" % (f, ln, attr, mod))
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
