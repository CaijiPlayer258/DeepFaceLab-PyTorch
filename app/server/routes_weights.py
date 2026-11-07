# -*- coding: utf-8 -*-
"""权重包状态 / 按需下载（weights/manifest.json）。

用户不装 sam 包时，「遮罩绘制」里的 SAM / GroundedSAM2 / BiSeNet 必须置灰：
装了 core 才有检测器、关键点、识别、XSeg。前端要的就是一句实话 —— 哪个包装了、
缺几个文件。所以这里三件事：

  GET  /api/weights/status        每个包 installed / present / missing（只查存在 + 大小，很快）
  POST /api/weights/fetch         起一个后台下载任务，立刻返回 job id
  GET  /api/weights/fetch/status  任务进度（百分比 / 最后一行日志 / 错误）

真正的下载交给引擎的 `python -m modelhub.fetch`，这里只负责拉起进程、把
stdout + stderr 收进环形缓冲（最后 200 行）并解析进度。事件循环里不做阻塞调用：
Popen 和读管道都在工作线程里，接口立刻返回。同一时刻只允许一个下载任务。

清单（manifest.json）描述目录结构，权重本体在 weights_dir() 里 —— 两者可以分开：
DFL_WEIGHTS_DIR 指向一个空目录时，仍然能按清单算出缺几个文件。
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import threading
import time
import uuid
from collections import deque
from pathlib import Path

from fastapi import APIRouter, HTTPException

from ._paths import engine_python as _paths_engine_python, weights_dir

router = APIRouter()

ROOT = Path(__file__).resolve().parents[2]
# GUI 应用：任何子进程都不许弹黑框（Windows 上这个位有效，POSIX 下是 0）
_CREATE_NO_WINDOW = 0x08000000 if os.name == "nt" else 0

LOG_LINES = 200          # 环形缓冲：只留最后 200 行
_PCT_RE = re.compile(r"(\d+(?:\.\d+)?)\s*%")

_log: deque = deque(maxlen=LOG_LINES)
_lock = threading.Lock()
_state = {
    "running": False, "job": "", "group": "", "pct": 0, "line": "",
    "error": "", "started_at": 0.0, "finished_at": 0.0, "rc": None,
    "total_files": 0, "done_files": 0, "pid": 0, "cmd": "", "url_base": "",
}


# ---------------------------------------------------------------- 小工具
def _truthy(v) -> bool:
    return str(v or "").strip().lower() in ("1", "true", "yes", "on", "y")


def _abs(p) -> str:
    try:
        return str(Path(p).resolve())
    except OSError:
        return str(p)


def _num(v, default: float = 0.0) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def _sha256(p: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _safe_rel(base: Path, rel) -> Path:
    """把清单里的相对路径钉死在 base 下面；绝对路径 / .. 一律 400。"""
    raw = str(rel or "").strip().replace("\\", "/")
    parts = [s for s in raw.split("/") if s not in ("", ".")]
    bad = (not parts) or raw.startswith("/") or (len(raw) > 1 and raw[1] == ":") \
        or any(s == ".." for s in parts)
    if bad:
        raise HTTPException(400, "清单里的路径越界：%r" % rel)
    p = base.joinpath(*parts)
    try:
        p.resolve().relative_to(base.resolve())
    except (ValueError, OSError):
        raise HTTPException(400, "清单里的路径越界：%r" % rel)
    return p


# ---------------------------------------------------------------- 清单
def _manifest_candidates(base: Path):
    seen, out = set(), []
    for cand in (base / "manifest.json",
                 ROOT / "weights" / "manifest.json",
                 ROOT / "app" / "engine" / "modelhub" / "manifest.json"):
        k = str(cand)
        if k not in seen:
            seen.add(k)
            out.append(cand)
    return out


def _load_manifest(base: Path):
    """(清单路径, 清单 dict)；一份都没有就 (None, None)。

    先看权重目录里的，再退回仓库自带的那份 —— DFL_WEIGHTS_DIR 指向空目录时，
    仍然要知道「应该有哪些文件」才能算出 missing。
    """
    for cand in _manifest_candidates(base):
        try:
            if cand.is_file():
                return cand, json.loads(cand.read_text(encoding="utf-8"))
        except Exception as ex:
            raise HTTPException(500, "清单读取失败 %s：%s" % (cand, ex))
    return None, None


def _package_groups(manifest: dict) -> list:
    pkgs = (manifest or {}).get("packages") or {}
    if pkgs:
        return [str(k) for k in pkgs.keys()]
    groups = []
    for e in (manifest or {}).get("files") or []:
        g = str((e or {}).get("group") or "")
        if g and g not in groups:
            groups.append(g)
    return groups


def _manifest_files(manifest: dict) -> list:
    return [e for e in ((manifest or {}).get("files") or []) if isinstance(e, dict)]


# ---------------------------------------------------------------- 下载器
def _fetch_ready():
    """(engine_dir, python_exe)；不可用时 (None, None) 并给出原因。"""
    engine = None
    for cand in (ROOT / "app" / "engine", ROOT):
        if (cand / "modelhub" / "fetch.py").is_file():
            engine = cand
            break
    if engine is None:
        tried = "、".join(str(ROOT / "app" / "engine" / "modelhub" / "fetch.py"),)
        return None, None, "找不到下载器 modelhub/fetch.py（找过 %s）" % tried
    pys = [ROOT / "runtime" / "engine" / "python.exe",
           ROOT / "runtime" / "engine" / "Scripts" / "python.exe",
           ROOT / "runtime" / "server" / "Scripts" / "python.exe"]
    for py in pys:
        if py.is_file():
            return engine, py, ""
    py = _paths_engine_python()
    if py.is_file():
        return engine, py, ""
    return None, None, ("找不到解释器（找过 %s）"
                        % "、".join(str(p) for p in pys))


# ---------------------------------------------------------------- 扫描
def _scan_package(pkg: dict, entries, do_verify: bool) -> dict:
    present = missing = checked = 0
    missing_bytes = 0.0
    want_bytes = 0.0
    missing_paths, size_bad, sha_bad = [], [], []
    for e, p in entries:
        rel = str(e.get("path") or "")
        want = int(_num(e.get("size")))
        want_bytes += want
        try:
            st = p.stat()
            is_file = p.is_file()
        except OSError:
            st, is_file = None, False
        if not is_file:
            missing += 1
            missing_bytes += want
            if len(missing_paths) < 20:
                missing_paths.append(rel)
            continue
        present += 1
        if want and st is not None and st.st_size != want:
            size_bad.append(rel)
            continue
        if do_verify:
            want_sha = str(e.get("sha256") or "").strip().lower()
            if want_sha:
                checked += 1
                if _sha256(p).lower() != want_sha:
                    sha_bad.append(rel)

    total = len(entries)
    installed = bool(total) and missing == 0 and not size_bad and not sha_bad
    out = {
        "installed": installed,
        "required": bool(pkg.get("required")),
        "title": str(pkg.get("title") or ""),
        "size_mb": round(_num(pkg.get("size_mb")) or want_bytes / 2 ** 20, 1),
        "files": int(_num(pkg.get("files")) or total),
        "declared_files": int(_num(pkg.get("files")) or total),
        "present": present,
        "missing": missing,
        "missing_mb": round(missing_bytes / 2 ** 20, 1),
        "size_bad": len(size_bad),
        "sha_bad": len(sha_bad),
        "unlocks": [str(u) for u in (pkg.get("unlocks") or [])],
    }
    if missing_paths:
        out["missing_sample"] = missing_paths
    if size_bad:
        out["size_bad_sample"] = size_bad[:10]
    if sha_bad:
        out["sha_bad_sample"] = sha_bad[:10]
    if do_verify:
        out["sha_checked"] = checked
    note = str(pkg.get("note") or "")
    if note:
        out["note"] = note
    return out


# ---------------------------------------------------------------- 状态
@router.get("/api/weights/status")
def weights_status(verify: str = ""):
    """哪个包装了、缺几个文件。默认只查存在 + 大小；verify=1 才逐个算 sha256（慢）。"""
    base = weights_dir()
    man_path, manifest = _load_manifest(base)
    engine, py, why = _fetch_ready()
    if manifest is None:
        return {
            "ok": False,
            "dir": _abs(base),
            "error": "找不到 weights/manifest.json（清单随仓库分发）",
            "hint": "确认 DFL_WEIGHTS_DIR 指向的目录或仓库 weights/ 里有 manifest.json",
            "packages": {},
            "total_mb": 0.0,
            "fetch": {"ready": False, "engine": "", "python": "", "hint": why},
        }

    entries = [(e, _safe_rel(base, e.get("path"))) for e in _manifest_files(manifest)]
    do_verify = _truthy(verify)
    packages = {}
    for key in _package_groups(manifest):
        pkg = ((manifest.get("packages") or {}).get(key) or {})
        mine = [(e, p) for e, p in entries if str(e.get("group") or "") == key]
        packages[key] = _scan_package(pkg, mine, do_verify)

    total_bytes = sum(_num(e.get("size")) for e in _manifest_files(manifest))
    return {
        "ok": True,
        "dir": _abs(base),
        "manifest": _abs(man_path),
        "verify": do_verify,
        "packages": packages,
        "total_mb": round(total_bytes / 2 ** 20, 1),
        "fetch": {
            "ready": bool(engine and py),
            "engine": _abs(engine) if engine else "",
            "python": str(py) if py else "",
            "hint": why,
        },
    }


# ---------------------------------------------------------------- 下载任务
def _set(**kw) -> None:
    with _lock:
        _state.update(kw)


def _push(line: str) -> None:
    with _lock:
        _log.append(line)
        _state["line"] = line[:400]


def _looks_like_error(line: str) -> bool:
    low = line.lower()
    if "失败：" in line or "✗" in line:
        return True
    if "traceback (most recent call last)" in low:
        return True
    for pat in ("modulenotfounderror", "filenotfounderror", "permissionerror",
                "urlerror", "httperror", "connectionrefused", "oserror"):
        if pat in low:
            return True
    return False


def _worker(job: str, group: str, url_base: str, total: int, engine: Path, py: Path) -> None:
    """工作线程：跑 modelhub.fetch，收日志、算进度。不碰事件循环。"""
    proc = None
    rc = None
    err = ""
    done = 0
    frac = 0.0
    try:
        cmd = [str(py), "-u", "-m", "modelhub.fetch", "--group", group]
        if url_base:
            cmd += ["--url-base", url_base]
        env = dict(os.environ)
        env["PYTHONUNBUFFERED"] = "1"
        env["PYTHONIOENCODING"] = "utf-8"
        # 子进程也认 DFL_WEIGHTS_DIR：页面上看到的目录就是它写进去的目录
        env["DFL_WEIGHTS_DIR"] = str(weights_dir())
        _set(cmd=" ".join(cmd))
        proc = subprocess.Popen(
            cmd, cwd=str(engine), env=env,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", errors="replace", bufsize=1,
            creationflags=_CREATE_NO_WINDOW,
        )
        _set(pid=proc.pid)
        stream = proc.stdout
        if stream is not None:
            for raw in stream:
                line = str(raw or "").strip()
                if not line:
                    continue
                _push(line)
                if line.startswith("→"):
                    done += 1
                    frac = 0.0
                else:
                    m = _PCT_RE.search(line)
                    if m:
                        try:
                            frac = min(100.0, max(0.0, float(m.group(1)))) / 100.0
                        except ValueError:
                            frac = 0.0
                if not err and _looks_like_error(line):
                    err = line[:400]
                pct = 0
                if total > 0:
                    pct = int(min(99.0, ((max(0, done - 1) + frac) * 100.0) / total))
                _set(pct=pct, done_files=done)
        rc = proc.wait()
    except Exception as ex:
        err = err or ("下载进程没能起来：%s" % ex)
    finally:
        if proc is not None and proc.poll() is None:
            try:
                proc.kill()
            except Exception:
                pass
        if rc is None:
            try:
                rc = proc.wait(timeout=5) if proc is not None else -1
            except Exception:
                rc = -1
        ok = (rc == 0 and not err)
        if not ok and not err:
            err = "下载进程退出码 %s" % rc
        if not ok:
            _push("[下载失败] %s" % err[:300])
        with _lock:
            _state.update(running=False, finished_at=time.time(), rc=rc,
                          error="" if ok else err[:400],
                          pct=100 if ok else int(_state.get("pct") or 0),
                          group=_state.get("group") or group)


@router.post("/api/weights/fetch")
def weights_fetch(body: dict | None = None):
    """{group, url_base} → 立刻返回 job id，下载在后台线程里跑。"""
    body = body or {}
    group = str(body.get("group") or "").strip()
    raw_base = body.get("url_base")
    url_base = "" if raw_base is None else str(raw_base).strip()

    base = weights_dir()
    _, manifest = _load_manifest(base)
    groups = _package_groups(manifest or {})
    if not group:
        raise HTTPException(400, "缺少 group（可选：%s）" % (" / ".join(groups) or "无"))
    if group not in groups:
        raise HTTPException(400, "group 非法：%r（可选：%s）" % (group, " / ".join(groups) or "无"))
    if url_base:
        low = url_base.lower()
        if not (low.startswith("http://") or low.startswith("https://")) \
                or any(c in url_base for c in "\r\n\t "):
            raise HTTPException(400, "url_base 必须是 http(s):// 开头的镜像地址：%r" % url_base)

    engine, py, why = _fetch_ready()
    if not (engine and py):
        raise HTTPException(501, "下载器不可用：%s" % (why or "modelhub.fetch 不存在"))

    total = len([e for e in _manifest_files(manifest or {})
                 if str(e.get("group") or "") == group])
    job = uuid.uuid4().hex[:12]
    with _lock:
        if _state["running"]:
            raise HTTPException(409, "已有下载任务在跑（group=%s，job=%s）"
                                % (_state.get("group"), _state.get("job")))
        _log.clear()
        _state.update(running=True, job=job, group=group, pct=0,
                      line="准备启动 modelhub.fetch …", error="",
                      started_at=time.time(), finished_at=0.0, rc=None,
                      total_files=total, done_files=0, pid=0,
                      cmd="", url_base=url_base)
    threading.Thread(target=_worker, daemon=True,
                     args=(job, group, url_base, total, engine, py),
                     name="weights-fetch").start()
    return {"ok": True, "job": job, "group": group, "files": total}


@router.get("/api/weights/fetch/status")
def weights_fetch_status(tail: int = 0):
    """进度汇总；tail=N 时把最后 N 行日志一起带回来。"""
    with _lock:
        st = dict(_state)
        lines = list(_log)
    started = _num(st.get("started_at"))
    stopped = _num(st.get("finished_at")) or time.time()
    out = {
        "ok": True,
        "running": bool(st.get("running")),
        "group": st.get("group") or "",
        "pct": int(_num(st.get("pct"))),
        "line": st.get("line") or "",
        "error": st.get("error") or "",
        "started_at": started,
        "finished_at": _num(st.get("finished_at")),
        "elapsed_s": round(stopped - started, 1) if started else 0.0,
        "job": st.get("job") or "",
        "rc": st.get("rc"),
        "files_done": int(_num(st.get("done_files"))),
        "files_total": int(_num(st.get("total_files"))),
        "url_base": st.get("url_base") or "",
        "cmd": st.get("cmd") or "",
        "pid": int(_num(st.get("pid"))),
    }
    if tail and tail > 0:
        out["log"] = lines[-min(int(tail), LOG_LINES):]
    return out
