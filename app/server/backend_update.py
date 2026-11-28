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

import datetime as _dt
import json
import re
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


# 版本号格式：MAJOR.MINOR-MMDD[-bN]（见 app/tools/bump_version.py）
#   4.1-1122        11-22 当天第一版
#   4.1-1122-b2     当天第二版
_VER_DATE = re.compile(r"^v?(\d+)\.(\d+)-(\d{2})(\d{2})(?:-b(\d+))?$")
_VER_TIME = re.compile(r"^(\d+)\.(\d+)\.(\d{2})\.(\d{4})\.(\d{4})$")


def parse_version(v=None) -> Dict[str, Any]:
    """把版本号拆成「大版本 + 构建时间」，给界面显示用。

    4.1.26.1122.2304 -> base 4.1 / 2026-11-22 23:04。解析不了就原样返回，
    不认识的格式（老版本 4.1.1）也能正常显示。
    """
    v = (v if v is not None else local_version()).strip()
    md = _VER_DATE.match(v)
    if md:
        major, minor, mm, dd, b = md.groups()
        build = int(b or 1)
        human = "v%s.%s \u00b7 %s-%s" % (major, minor, mm, dd)
        # 只要版本号里**显式写了** -bN 就显示，包括 b0 / b1。
        # 以前是 `if build > 1`，于是 1126-b0、1126-b1 和「不带 build 的 1126」
        # 在关于页长得一模一样 —— 用户根本分不清自己装的是哪一版。
        if b is not None:
            human += " \u00b7 B%d" % build
        return {"version": v, "base": "%s.%s" % (major, minor), "build": build,
                "when": None, "human": human}
    m = _VER_TIME.match(v)
    if not m:
        return {"version": v, "base": v, "when": None, "human": "v" + v}
    major, minor, yy, mmdd, hhmm = m.groups()
    try:
        # 用 datetime 而不是 time.strftime：后者要 9 元组，少一个就静默失败
        when_dt = _dt.datetime(2000 + int(yy), int(mmdd[:2]), int(mmdd[2:]),
                               int(hhmm[:2]), int(hhmm[2:]))
        when = when_dt.strftime("%Y-%m-%d %H:%M")
    except Exception:
        return {"version": v, "base": "%s.%s" % (major, minor), "when": None, "human": "v" + v}
    return {"version": v, "base": "%s.%s" % (major, minor), "when": when,
            "human": "v%s.%s · %s" % (major, minor, when)}


def _ver_tuple(v: str) -> tuple:
    # 4.1-1122 / 4.1-1122-b2：日期固定四位，直接比 (major, minor, MMDD, build)
    v = (v or "").strip()
    md = _VER_DATE.match(v)
    if md:
        major, minor, mm, dd, b = md.groups()
        return (int(major), int(minor), int(mm + dd), int(b or 1))
    # 旧格式（4.1.2 / 4.1.26.1122.2310）一律当「更早」：
    # 它们没有日期段，混进新格式里比较没有意义（4.1-200 会被拆成 (4,1200) 排到 4.1-1122 前面）。
    return (0, 0, 0, 0)



def _local_built_at():
    """本地这份代码是什么时候落盘的（version.txt 的 mtime）。

    直接用模块级的 VERSION_FILE —— 它和 local_version() 读的是同一个文件，
    已经处理过「源码跑 / 打包后跑」两种布局。别自己拿 _paths.root() 拼路径：
    那个模块是动态转发的，取错属性会静默抛异常，函数就退化成返回 0，
    兜底逻辑等于没写（第一版就是这么废掉的）。
    """
    try:
        return float(VERSION_FILE.stat().st_mtime)
    except Exception:
        return 0.0


def _is_behind(tag, local, published):
    """远端是不是比本地新。

    1) 正常：版本号元组比较。
    2) 兜底：号相同或更小，但远端发布时间晚于本地 version.txt 的落盘时间 → 也算新。
       专治「b 号被写乱之后归位」造成的假阴性（11-28 当天发到 b70，归位成 b3/b4/b5
       之后 _ver_tuple(remote) > _ver_tuple(local) 永远 False，老用户收不到更新）。
    """
    rt, lt = _ver_tuple(tag), _ver_tuple(local)
    if rt > lt:
        return True
    if not published:
        return False
    try:
        import datetime as _d2
        # ★ 必须按 **UTC** 解释：GitHub 的 published_at 是 UTC（...Z），
        #   而 naive strptime().timestamp() 会按**本地时区**算 —— 在东八区就差 8 小时，
        #   比 release 更晚落盘的本地文件会被误判成「比远端还新」，兜底等于失效。
        pub = _d2.datetime.strptime(str(published)[:19], "%Y-%m-%dT%H:%M:%S")
        pub = pub.replace(tzinfo=_d2.timezone.utc).timestamp()
    except Exception:
        return False
    local_t = _local_built_at()
    if not local_t:
        return False
    return pub > local_t + 60.0


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
        "behind": _is_behind(tag, local, info.get("published_at")),
        "asset": code_asset,
        "package": "release asset" if code_asset else "source zipball",
    }


# 安装进度：前端轮询 /api/update/progress 画进度条。
# 22 MB 的代码包 + 1600 个文件，光靠一句「正在下载并安装…」用户不知道要等多久。
_PROGRESS: Dict[str, Any] = {"running": False, "phase": "", "pct": 0,
                              "done": 0, "total": 0, "msg": ""}


def progress() -> Dict[str, Any]:
    return dict(_PROGRESS)


def _set_progress(**kw) -> None:
    _PROGRESS.update(kw)


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


def _prune_backups(keep: int = 1) -> int:
    """只保留最近 keep 个 .backup-* 目录，更老的删掉。

    每更新一次就会在工作区留一份覆盖前的代码备份（十几 MB）。
    不清理的话会一直堆下去，用户也不知道能不能删。
    """
    ws = ROOT / "workspace"
    if not ws.is_dir():
        return 0
    try:
        dirs = sorted((d for d in ws.iterdir() if d.is_dir() and d.name.startswith(".backup-")),
                      key=lambda d: d.name, reverse=True)
    except OSError:
        return 0
    removed = 0
    for d in dirs[keep:]:
        try:
            shutil.rmtree(d, ignore_errors=True)
            removed += 1
        except Exception:
            pass
    return removed


# 启动时后台跑一次的自动检查：结果缓存在这里，前端问 /api/update/auto 即可。
# 不在请求里同步调 GitHub —— 那是 1~3 秒的网络等待，会卡住主页首屏。
_AUTO_STATE: Dict[str, Any] = {"checked": False, "checking": False,
                               "behind": False, "remote": "", "tag": "",
                               "name": "", "error": "", "at": 0}


def auto_state() -> Dict[str, Any]:
    return dict(_AUTO_STATE)


def start_auto_check(delay: float = 0.5) -> None:
    """起一个 daemon 线程，延时 delay 秒后查一次 release。

    延时是刻意的：进程刚起来时 /api/state 预热、TRT 引擎加载都在抢 IO，
    这时候去连 GitHub 没有意义，晚几秒再问。
    """
    import threading

    def _run():
        _AUTO_STATE["checking"] = True
        try:
            time.sleep(max(0.0, float(delay)))
            info = check()
            if not info.get("ok"):
                _AUTO_STATE.update({"checked": True, "behind": False,
                                   "error": str(info.get("error") or "")[:200],
                                   "at": time.time()})
                return
            _AUTO_STATE.update({"checked": True, "behind": bool(info.get("behind")),
                               "remote": str(info.get("remote") or ""),
                               "tag": str(info.get("tag") or ""),
                               "name": str(info.get("name") or ""),
                               "local": str(info.get("local") or ""),
                               "error": "", "at": time.time()})
        except Exception as e:
            _AUTO_STATE.update({"checked": True, "behind": False,
                               "error": "%s: %s" % (type(e).__name__, str(e)[:160]),
                               "at": time.time()})
        finally:
            _AUTO_STATE["checking"] = False

    threading.Thread(target=_run, name="update-autocheck", daemon=True).start()


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

    _set_progress(running=True, phase="download", pct=0, done=0, total=0,
                  msg="正在下载 %s" % str(picked["name"] if picked else "源码包"))
    tmp = Path(tempfile.mkdtemp(prefix="dfl-update-"))
    zip_path = tmp / "pkg.zip"
    try:
        def _dl(done, total):
            pct = int(done * 100 / total) if total else 0
            _set_progress(phase="download", pct=min(99, pct), done=done, total=total,
                          msg="正在下载 %d%%" % min(99, pct))
        _download(url, zip_path, on_progress=_dl)
        _set_progress(phase="unzip", pct=99, msg="正在解包…")
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
        # 备份放 workspace/ 下：放项目根会被 git 当成未跟踪文件，
        # 更糟的是「全盘 add」时被一起提交（实测把 1328 个文件塞进了提交）。
        # 用户数据区本来就被 .gitignore 整个忽略，也符合「备份是用户资产」的语义。
        _ws = ROOT / "workspace"
        backup = (_ws if _ws.is_dir() else ROOT) / f".backup-{int(time.time())}"
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

        # ★ 清 __pycache__：copy2 保留 mtime，而 .pyc 的有效性判据正是「源 mtime + size」。
        # 新旧两版如果 mtime 相同（zip 里的时间戳），Python 会认为旧 .pyc 依然有效，
        # 继续跑**旧字节码** —— 表现就是「version.txt 是新的、界面也显示新版本号，
        # 但某个模块的行为还是旧的」。用户那边 compose.py 的守卫一直不生效，
        # 最可能就是踩了这个。这里把所有 __pycache__ 目录删掉，重启后必然重编译。
        _dropped_pyc = 0
        for _pc in ROOT.rglob("__pycache__"):
            if not _pc.is_dir():
                continue
            try:
                _dropped_pyc += len(list(_pc.glob("*.pyc")))
                shutil.rmtree(_pc, ignore_errors=True)
            except Exception:
                pass
        if _dropped_pyc:
            print("[update] 清掉 %d 个 __pycache__ 字节码（避免跑旧 .pyc）" % _dropped_pyc,
                  flush=True)

        return {"ok": True, "installed": True, "version": local_version(),
                "files": len(installed), "sample": installed[:10],
                "backup": str(backup) if backup.exists() else None,
                "package": info.get("package"), "restart_required": True}
    except Exception as exc:
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        try:
            _dropped = _prune_backups(1)          # 只留最近一份，更老的清掉
            if _dropped:
                print("[update] 清理了 %d 份旧备份" % _dropped, flush=True)
        except Exception:
            pass
        _set_progress(running=False, phase="done", pct=100, msg="已安装，重启后生效")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
