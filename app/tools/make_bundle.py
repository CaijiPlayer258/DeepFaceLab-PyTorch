# -*- coding: utf-8 -*-
r"""把「代码 + 统一运行时 + 模型」打成一个可直接解压运行的 zip。

用法（在仓库根执行）：
    runtime\unified\python.exe app\tools\make_bundle.py --out D:\MyProject\dist\DeepFaceLab.zip

为什么用 Python 的 zipfile 而不是 tar/7z：
  · bsdtar 写 zip 时按本地代码页写文件名，中文模型名到别的机器会变乱码；
    Python 写非 ASCII 名会置 UTF-8 标志位（0x800），解压端能正确还原。
  · 权重/onnx 基本压不动（~93%），但 dll/pyd/exe 能压到 35~65%，整体仍划算：
  默认 deflate；要最快打包就加 --store。

默认排除：__pycache__/*.pyc/*.tmp、模型目录里的 *.zip 备份、autobackups、old/。
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

KEEP_TOP = ["version.txt", "app", "desktop", "tools", "runtime/unified", "weights"]

# workspace 只带 XSegLite 这一个模型（遮罩推理要用），其余模型/数据集/缓存一概不带。
WORKSPACE_KEEP = ["workspace/model/XSegLite"]
# 解压后要有的空目录骨架（zip 里用目录条目表示）
WORKSPACE_SKELETON = [
    "workspace",
    "workspace/model",
    "workspace/data_src",
    "workspace/data_dst",
    "workspace/merge",
]

EXCLUDE_DIR_NAMES = {"__pycache__", ".git", ".cache", "old", "auth"}
EXCLUDE_SUFFIX = {".pyc", ".pyo", ".tmp", ".log"}
# 凭证：B 站登录 cookie / 本机授权信息，绝对不能进分发包
EXCLUDE_FILES = {"auth.json", "bilibili_info.json", "settings.json",
                "bilibili_cookie.txt", "bilibili_cookie.txt.bak"}
EXCLUDE_MODEL_GLOB = ("*.zip",)          # 用户自己打的备份包
EXCLUDE_DIR_HINTS = ("autobackups",)


def wanted(rel: Path) -> bool:
    parts = rel.parts
    if any(p in EXCLUDE_DIR_NAMES for p in parts):
        return False
    if any(h in p for p in parts for h in EXCLUDE_DIR_HINTS):
        return False
    if rel.suffix.lower() in EXCLUDE_SUFFIX:
        return False
    if rel.name in EXCLUDE_FILES:
        return False
    if "cookie" in rel.name.lower() and rel.suffix.lower() in (".txt", ".bak", ".json"):
        return False
    if parts and parts[0] == "workspace" and len(parts) > 2 and parts[1] == "model":
        name = rel.name.lower()
        if any(name.endswith(s[1:]) for s in EXCLUDE_MODEL_GLOB):
            return False
    return True


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--store", action="store_true",
                    help="不压缩直接存（快，包体大近一倍）；默认 deflate")
    args = ap.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        out.unlink()

    mode = zipfile.ZIP_STORED if args.store else zipfile.ZIP_DEFLATED
    n = 0
    total = 0
    t0 = time.time()
    missing = []
    sources = []
    for top in KEEP_TOP + WORKSPACE_KEEP:
        p = ROOT / top
        if not p.exists():
            missing.append(top)
            continue
        sources.append(p)
    if missing:
        print("[warn] 跳过不存在的条目：%s" % ", ".join(missing), flush=True)

    with zipfile.ZipFile(out, "w", mode, allowZip64=True,
                         compresslevel=None if args.store else 6) as zf:
        for d in WORKSPACE_SKELETON:
            zf.writestr(zipfile.ZipInfo(d + "/"), b"")
        for src in sources:
            if src.is_file():
                rel = src.relative_to(ROOT)
                zf.write(src, rel.as_posix())
                n += 1
                total += src.stat().st_size
                continue
            for dirpath, dirnames, filenames in os.walk(src):
                dirnames[:] = [d for d in dirnames if wanted(Path(dirpath).relative_to(ROOT) / d)]
                for name in filenames:
                    f = Path(dirpath) / name
                    rel = f.relative_to(ROOT)
                    if not wanted(rel):
                        continue
                    try:
                        size = f.stat().st_size
                    except OSError:
                        continue
                    zf.write(f, rel.as_posix())
                    n += 1
                    total += size
                    if n % 5000 == 0:
                        print("  %d 个文件 / %.2f GB / %.1f 分钟"
                              % (n, total / 1073741824, (time.time() - t0) / 60), flush=True)

    size = out.stat().st_size
    print("完成：%s" % out, flush=True)
    print("  文件 %d 个，原始 %.2f GB，包体 %.2f GB，用时 %.1f 分钟"
          % (n, total / 1073741824, size / 1073741824, (time.time() - t0) / 60), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
