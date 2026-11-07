"""权重下载 / 校验（`python -m modelhub.fetch`）。

代码与模型分离：仓库只带代码，权重按 `weights/manifest.json` 下载。
清单里的 path 是**平铺**的（`tinymog.onnx` / `retinaface_10g.onnx` / `sam2.1_hiera_small.pt`），
落到 `<权重根>/<path>`；唯一带目录的一条是 BERT tokenizer（`bert-base-uncased/`，
AutoTokenizer 要求词表和权重同目录，平铺不了）。
支持三种来源，按优先级：

1. `--source DIR`   从本地目录复制（离线分发包、内网镜像）
2. `--url-base URL` 从 HTTP 镜像下载：`<url-base>/<manifest 里的 path>`
3. 清单里若写了 `url`，直接用

用法：

    python -m modelhub.fetch --list                  # 看缺哪些、有多大
    python -m modelhub.fetch --group core            # 下检测器（约 1 GB）
    python -m modelhub.fetch --group sam             # 下 SAM/GroundingDINO（约 3.4 GB）
    python -m modelhub.fetch --all --source DIR   # 从本地目录拷
    python -m modelhub.fetch --verify                # 只校验 sha256
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import urllib.request
from pathlib import Path

from . import weights as W


def _sha256(p: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _entries(groups) -> list:
    data = W.manifest().get("files", [])
    if groups:
        data = [e for e in data if e.get("group") in groups]
    return data


def _target(entry) -> Path:
    return W.root() / entry["path"]


def cmd_list(entries):
    missing = [e for e in entries if not _target(e).is_file()]
    total = sum(e["size"] for e in entries)
    miss = sum(e["size"] for e in missing)
    print("清单：%d 个文件，共 %.2f GB" % (len(entries), total / 2**30))
    print("缺失：%d 个，需下载 %.2f GB" % (len(missing), miss / 2**30))
    for e in missing:
        print("  - %-58s %6.1f MB  [%s]" % (e["path"], e["size"] / 2**20, e.get("group", "")))
    return 0


def cmd_verify(entries):
    bad = 0
    for e in entries:
        p = _target(e)
        if not p.is_file():
            print("缺失 %s" % e["path"])
            bad += 1
            continue
        if _sha256(p) != e["sha256"]:
            print("校验失败 %s" % e["path"])
            bad += 1
    print("校验完成：%d 个文件，%d 个有问题" % (len(entries), bad))
    return 1 if bad else 0


def _download(url: str, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".part")
    with urllib.request.urlopen(url, timeout=60) as r, open(tmp, "wb") as f:
        total = int(r.headers.get("Content-Length") or 0)
        done = 0
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
            done += len(chunk)
            if total:
                sys.stdout.write("\r    %5.1f%%  %s" % (done * 100.0 / total, dst.name))
                sys.stdout.flush()
    print()
    tmp.replace(dst)


def cmd_fetch(entries, source: Path | None, url_base: str | None, force: bool):
    failed = []
    for e in entries:
        dst = _target(e)
        if dst.is_file() and not force and _sha256(dst) == e["sha256"]:
            continue
        print("→ %s" % e["path"])
        try:
            if source is not None:
                src = source / e["path"]
                if not src.is_file():
                    raise FileNotFoundError("本地源里没有 %s" % src)
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)
            else:
                url = e.get("url") or (url_base.rstrip("/") + "/" + e["path"] if url_base else "")
                if not url:
                    raise RuntimeError("清单里没有 url，也没给 --source/--url-base")
                _download(url, dst)
        except Exception as ex:
            print("   失败：%s" % ex)
            failed.append((e["path"], str(ex)))
            continue
        got = _sha256(dst)
        if got != e["sha256"]:
            print("   sha256 不匹配（期望 %s，实际 %s），已删除" % (e["sha256"][:12], got[:12]))
            dst.unlink(missing_ok=True)
            failed.append((e["path"], "sha256 mismatch"))
    print("\n完成：成功 %d，失败 %d" % (len(entries) - len(failed), len(failed)))
    for p, why in failed:
        print("  ✗ %s — %s" % (p, why))
    return 1 if failed else 0


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m modelhub.fetch", description="模型权重下载/校验")
    ap.add_argument("--list", action="store_true", help="列出缺失的文件")
    ap.add_argument("--verify", action="store_true", help="校验已存在文件的 sha256")
    ap.add_argument("--all", action="store_true", help="处理清单里的全部文件")
    ap.add_argument("--group", action="append", help="只处理某组：core / sam")
    ap.add_argument("--file", action="append", help="只处理某个相对路径（可多次）")
    ap.add_argument("--source", help="本地源目录（离线拷贝）")
    ap.add_argument("--url-base", help="HTTP 镜像前缀")
    ap.add_argument("--force", action="store_true", help="已存在也重下")
    a = ap.parse_args(argv)

    groups = a.group or None
    entries = _entries(groups)
    if a.file:
        entries = [e for e in entries if e["path"] in set(a.file)]

    if a.list:
        return cmd_list(entries)
    if a.verify:
        return cmd_verify(entries)
    if not (a.all or a.group or a.file):
        ap.print_help()
        return 0
    src = Path(a.source).expanduser() if a.source else None
    return cmd_fetch(entries, src, a.url_base, a.force)


if __name__ == "__main__":
    raise SystemExit(main())
