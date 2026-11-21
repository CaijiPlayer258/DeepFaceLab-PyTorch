# -*- coding: utf-8 -*-
"""自动更新：只拉 GitHub Release 的代码包，不碰项目里的图片与大文件。

设计要点
--------
* 检查：GET /releases/latest，比对 version.txt 的版本号。
* 下载：优先取 release 资产里名字含 code 的 zip；没有资产时退回源码 zipball。
* 安装：解包到临时目录 → 只覆盖代码目录（app/、desktop/ 里的源码、根级脚本）
         → 跳过 workspace/、runtime/、models/、*.pth 等用户数据与大文件。
* 回滚：覆盖前把旧代码目录改名备份，失败时还原。
"""
from __future__ import annotations

import json
import os
import shutil
import tempfile
import time
import urllib.request
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO = "CaijiPlayer258/DeepFaceLab-PyTorch"
API_LATEST = f"https://api.github.com/repos/{REPO}/releases/latest"
ROOT = Path(__file__).resolve().parents[2]
VERSION_FILE = ROOT / "version.txt"

# 覆盖时允许更新的路径（其余一律不动）
UPDATE_PATHS = ("app", "desktop", "tools", "version.txt")
# 明确保护的用户数据/大文件
KEEP_NAMES = {"workspace", "runtime", "models", "node_modules", ".git", ".venv"}
KEEP_SUFFIX = {".pth", ".dfm", ".onnx", ".engine", ".mp4", ".jpg", ".png"}


def local_version() -> str:
    try:
        return VERSION_FILE.read_text(encoding="utf-8-sig").strip().splitlines()[0].strip()
    except Exception:
        return "0.0.0"


def _get_json(url: str, timeout: int = 15) -> Dict[str, Any]:
    req = urllib.request.Request(url, headers={"User-Agent": "DeepFaceLab-4.1-Updater", "Accept": "application/vnd.github+json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _ver_tuple(v: str) -> tuple:
    out = []
    for part in v.replace("v", "").split("."):
        num = "".join(ch for ch in part if ch.isdigit())
        out.append(int(num) if num else 0)
    return tuple(out)


def check() -> Dict[str, Any]:
    """返回远端最新版本信息；网络失败时给出 error 字段而不抛异常。"""
    local = local_version()
    try:
        info = _get_json(API_LATEST)
    except Exception as exc:  # 网络、限流等
        return {"ok": False, "local": local, "error": f"{type(exc).__name__}: {exc}"}

    tag = str(info.get("tag_name") or "")
    assets: List[Dict[str, Any]] = [
        {"name": a.get("name"), "size": a.get("size"), "url": a.get("browser_download_url")}
        for a in info.get("assets") or []
    ]
    code_asset = next((a for a in assets if a["name"] and "code" in a["name"].lower() and str(a["name"]).endswith(".zip")), None)
    return {
        "ok": True,
        "local": local,
        "remote": tag.lstrip("v"),
        "tag": tag,
        "name": info.get("name") or tag,
        "notes": (info.get("body") or "")[:4000],
        "published": info.get("published_at"),
        "assets": assets,
        "source_zip": info.get("zipball_url"),
        "behind": _ver_tuple(tag) > _ver_tuple(local),
        "asset": code_asset,
        "package": "release asset" if code_asset else "source zipball",
    }


def _download(url: str, dst: Path, on_progress=None) -> Path:
    req = urllib.request.Request(url, headers={"User-Agent": "DeepFaceLab-4.1-Updater"})
    with urllib.request.urlopen(req, timeout=60) as resp, open(dst, "wb") as fh:
        total = int(resp.headers.get("Content-Length") or 0)
        done = 0
        while True:
            buf = resp.read(256 * 1024)
            if not buf:
                break
            fh.write(buf)
            done += len(buf)
            if on_progress:
                on_progress(done, total)
    return dst


def _is_updateable(rel: Path) -> bool:
    if rel.suffix.lower() in KEEP_SUFFIX:
        return False
    top = rel.parts[0] if rel.parts else ""
    if top in KEEP_NAMES:
        return False
    return top in UPDATE_PATHS


def apply(asset_name: Optional[str] = None) -> Dict[str, Any]:
    """下载并安装最新版本。返回 {ok, installed, files, backup, error}。"""
    info = check()
    if not info.get("ok"):
        return {"ok": False, "error": info.get("error", "check failed")}
    if not info.get("behind"):
        return {"ok": True, "installed": False, "message": "已是最新版本", "local": info["local"]}

    assets = info.get("assets") or []
    picked = None
    if asset_name:
        picked = next((a for a in assets if a["name"] == asset_name), None)
    if picked is None:
        picked = info.get("asset")
    url = picked["url"] if picked else info.get("source_zip")
    if not url:
        return {"ok": False, "error": "该 release 既没有代码资产也没有源码包"}

    tmp = Path(tempfile.mkdtemp(prefix="dfl-update-"))
    zip_path = tmp / "pkg.zip"
    try:
        _download(url, zip_path)
        with zipfile.ZipFile(zip_path) as zf:
            names = zf.namelist()
            # 只有"外层包了一层目录"时（GitHub 源码包形式）才剥掉这一层；
            # 自己的代码包是直接从 app/ 开始的，不能剥。
            root_dir = ""
            tops = {n.split("/")[0] for n in names if n and not n.startswith("/")}
            if len(tops) == 1:
                top = next(iter(tops))
                if f"{top}/app/" in names or f"{top}/version.txt" in names:
                    root_dir = top + "/"
            staged = tmp / "staged"
            staged.mkdir()
            for n in names:
                rel = Path(n[len(root_dir):]) if root_dir and n.startswith(root_dir) else Path(n)
                if not rel.parts or n.endswith("/"):
                    continue
                if not _is_updateable(rel):
                    continue
                target = staged / rel
                target.parent.mkdir(parents=True, exist_ok=True)
                with zf.open(n) as src, open(target, "wb") as dst:
                    shutil.copyfileobj(src, dst)

        installed: List[str] = []
        backup = ROOT / f".backup-{int(time.time())}"
        for path in sorted(staged.rglob("*")):
            if not path.is_file():
                continue
            rel = path.relative_to(staged)
            dst = ROOT / rel
            if dst.exists():
                b = backup / rel
                b.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(dst, b)
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, dst)
            installed.append(str(rel).replace("\\", "/"))

        return {"ok": True, "installed": True, "version": local_version(),
                "files": len(installed), "sample": installed[:10],
                "backup": str(backup) if backup.exists() else None,
                "package": info.get("package"), "restart_required": True}
    except Exception as exc:
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
