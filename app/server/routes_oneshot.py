# -*- coding: utf-8 -*-
"""数据集体检（One Shot）（从 app.py 拆出）。"""
from __future__ import annotations

import asyncio
import atexit
import base64
import io
import json
import os
import queue
import re
import subprocess as _sp
import threading
import time
from pathlib import Path

import numpy as np

from fastapi import APIRouter, HTTPException, Response
from fastapi.responses import FileResponse, StreamingResponse

from .app import (WORKSPACE, _DFL_ROOT, _NO_WINDOW, _dhash, _lap_var, _list_images, app)
from ._paths import engine_python as _engine_python
from . import _dataset_fast as FF
from . import _occlusion as OCC

router = APIRouter()

# ---------------- 数据集体检（One Shot） ----------------
# 扫描在 **4.0 的 python** 里跑（_scan_runner.py）：
#   ① 那边带着完整的 numpy/cv2/numba，4.1 只负责起进程、转进度
#   ② 以后人脸 embedding / 聚类也能接在同一条流水线上
# 协议：@@P <done> <total> <stage> 进度、@@R <json> 结果、@@ERR <msg> 错误。
_oneshot_job = {"running": False, "done": 0, "total": 0, "stage": "", "result": None,
                "error": None, "dir": "", "started": 0.0}
_oneshot_lock = threading.Lock()


# ---------------- 数据集目录解析 / 计数（唯一口径） ----------------
def _norm_target(target) -> str:
    """workspace 下的相对目录名。绝对路径会把基路径整个替换掉，一律拒绝。"""
    s = str(target or "data_src").replace("\\", "/").strip("/")
    if os.path.isabs(s) or ".." in s.split("/") or ":" in s:
        raise HTTPException(400, "非法的数据集目录: " + s)
    return s or "data_src"


def _resolve_root(target):
    """数据集目录名 → 真正要处理的目录，返回 (root, redirected)。

    **aligned 兜底只在用户选的就是项目根时生效。** 以前是无条件优先 aligned：
    用户在 data_src/1 里点「人脸身份」，程序转头去聚 data_src/aligned 的 1203 张，
    跟他自己数的 670 张对不上（和 app.py: _count_images 注释里那次 rglob
    报 1204 张是同一类「数的是别的目录」的坑）。
    判据只有一条：**这一层自己有图，就按它；自己没图，才下探 aligned。**
    """
    root = WORKSPACE / _norm_target(target)
    if not root.is_dir():
        raise HTTPException(404, "目录不存在：%s" % root)
    redirected = False
    aligned = root / "aligned"
    if aligned.is_dir() and not FF.list_images(root):
        root, redirected = aligned, True
    return root, redirected


def _count_payload(root, target, redirected):
    """一处算完，UI 和引擎共用：这一层有几张、哪些子目录被跳过。"""
    files = FF.list_images(root)
    skipped = FF.skipped_subdirs(root)
    return {"ok": True, "target": target, "dir": str(root),
            "redirected_to_aligned": bool(redirected),
            "count": len(files), "n_files": len(files),
            "recursive_would_be": len(files) + sum(s["images"] for s in skipped),
            "skipped_dirs": [s for s in skipped if s["images"]]}


@router.post("/api/dataset/count")
def dataset_count(payload: dict = None):
    """**只数这一层**：前端/进度条一律用这个数，别再自己数。

    以前 UI 用 /api/state 里的 datasets.src.count 预填 —— 那个数永远数的是
    data_src/aligned，选中 data_src/1 也照样显示 1203 张（实际聚类 670 张）。
    """
    payload = payload or {}
    target = _norm_target(payload.get("dir") or "data_src")
    root, redirected = _resolve_root(target)
    return _count_payload(root, target, redirected)


@router.post("/api/dataset/oneshot")
def dataset_oneshot(payload: dict):
    payload = payload or {}
    target = _norm_target(payload.get("dir") or "data_src")
    root, _redirected = _resolve_root(target)

    with _oneshot_lock:
        if _oneshot_job["running"]:
            raise HTTPException(409, "已经在扫了")

    level = str(payload.get("level") or "strict")
    score_pct = float(payload.get("score_pct", 5.0) if payload.get("score_pct") is not None else 5.0)
    bucket_floor = int(payload.get("bucket_floor", 0) or 0)
    total_files = len(_list_images(root))

    with _oneshot_lock:
        _oneshot_job.update({"running": True, "done": 0, "total": total_files,
                             "stage": "scan", "result": None, "error": None,
                             "dir": target, "started": time.time()})

    def _work():
        import subprocess
        py = _engine_python()
        runner = Path(__file__).resolve().parent / "_scan_runner.py"
        if not py.is_file() or not runner.is_file():
            with _oneshot_lock:
                _oneshot_job["running"] = False
                _oneshot_job["error"] = "找不到引擎 python 或 _scan_runner.py"
            return
        env = dict(os.environ)
        env["PYTHONIOENCODING"] = "utf-8"
        cmd = [str(py), str(runner), "--dir", str(root),
               "--level", level,
               "--score-pct", str(score_pct), "--bucket-floor", str(bucket_floor)]
        try:
            proc = subprocess.Popen(cmd, cwd=str(_DFL_ROOT), env=env,
                                    creationflags=_NO_WINDOW,
                                    stdin=subprocess.DEVNULL,
                                    stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                                    text=True, encoding="utf-8", errors="replace", bufsize=1)
            for line in proc.stdout:
                line = line.rstrip("\n")
                if line.startswith("@@P "):
                    parts = line.split()
                    if len(parts) >= 3:
                        with _oneshot_lock:
                            _oneshot_job["done"] = int(parts[1])
                            _oneshot_job["total"] = int(parts[2])
                            if len(parts) >= 4:
                                _oneshot_job["stage"] = parts[3]
                elif line.startswith("@@ERR "):
                    with _oneshot_lock:
                        _oneshot_job["error"] = line[6:]
                elif line.startswith("@@R "):
                    try:
                        data = json.loads(line[4:])
                    except Exception as e:
                        with _oneshot_lock:
                            _oneshot_job["error"] = "结果解析失败: %s" % e
                        continue
                    with _oneshot_lock:
                        _oneshot_job["result"] = data
            proc.wait()
            with _oneshot_lock:
                _oneshot_job["running"] = False
                if _oneshot_job["result"] is None and not _oneshot_job["error"]:
                    _oneshot_job["error"] = "扫描进程退出码 %s" % proc.returncode
        except Exception as e:
            with _oneshot_lock:
                _oneshot_job["running"] = False
                _oneshot_job["error"] = "%s: %s" % (type(e).__name__, e)

    threading.Thread(target=_work, daemon=True).start()
    return {"ok": True, "total": total_files}


@router.get("/api/dataset/oneshot/status")
def dataset_oneshot_status():
    with _oneshot_lock:
        j = dict(_oneshot_job)
    j["elapsed"] = round(time.time() - j["started"], 1) if j.get("started") else 0
    return j


_FACEID_PROGRESS = {"path": ""}


@router.get("/api/dataset/faceid/progress")
def faceid_progress():
    """读引擎进程写的进度文件：stage=embed/cluster/done, done/total。"""
    # 不依赖内存里的路径（路由模块可能被导入两次）：直接找最新的进度文件
    import glob
    cands = sorted(glob.glob(str(WORKSPACE / "_faceid_*.json")) +
                   glob.glob(str(WORKSPACE / "_dsprog_*.json")),
                   key=os.path.getmtime, reverse=True)
    if not cands:
        return {"ok": False}
    try:
        with open(cands[0], "r", encoding="utf-8") as f:
            d = json.load(f)
        # 超过 5 分钟没更新就当过期，别让 UI 一直读旧值
        if time.time() - os.path.getmtime(cands[0]) > 3600:
            return {"ok": False}
        d["ok"] = True
        return d
    except Exception:
        return {"ok": False}


def _ds_progress_writer(stage, total):
    """写一份扫描进度文件（和 faceid 那套同一个约定），返回 put(done)。"""
    import json as _j
    import os as _o
    import time as _t

    path = WORKSPACE / ("_dsprog_%d.json" % _o.getpid())
    try:
        if path.exists():
            path.unlink()          # 上一轮留下的 finished 状态会让 UI 一进来就显示 100%
    except Exception:
        pass
    last = [_t.time()]

    def put(done, total_n=None, stage_name=None):
        now = _t.time()
        # 每 150ms 写一次就够前端 400ms 轮询看了，别把磁盘写爆
        if done < (total_n or total) and now - last[0] < 0.15:
            return
        last[0] = now
        try:
            tmp = path.with_suffix(".tmp")
            tmp.write_text(_j.dumps({"stage": stage_name or stage, "done": int(done),
                                     "total": int(total_n or total)}, ensure_ascii=False),
                           encoding="utf-8")
            _o.replace(str(tmp), str(path))
        except Exception:
            pass

    put(0)
    return put


# ---------------- 人脸身份：常驻引擎进程（一次加载、多次复用） ----------------
# 以前每点一次「人脸身份」就 subprocess.run 一个全新的引擎 python：
#   冷启动 + import torch/tensorrt + 反序列化 ~9.6 MB 的 TRT 引擎 ≈ 3.5~4.5 s，
#   而真正干活（670 张 ArcFace 特征 + 聚类）只要 0.3~0.5 s。
# 现在起一个常驻进程，用 stdin/stdout 的 JSON 行协议喂请求，引擎只加载一次；
# 第一次也可以提前付（prewarm，和 /api/merge/preview/warm 一个套路）。
_faceid_proc = {"proc": None, "q": None, "lock": threading.Lock(), "boot_ms": None,
                "warm_ms": None, "provider": "", "requests": 0, "error": "",
                "ready": False, "warming": False, "engine_ms": None}
_faceid_spawn_lock = threading.Lock()


def _faceid_alive() -> bool:
    p = _faceid_proc.get("proc")
    try:
        return bool(p is not None and p.poll() is None)
    except Exception:
        return False


def _faceid_reader(proc, q):
    """把引擎进程的 stdout 一行行塞进队列 —— readline() 没有超时，会卡死工作线程。"""
    try:
        for line in proc.stdout:
            q.put(line)
    except Exception:
        pass
    finally:
        q.put(None)


def _faceid_reset(reason: str = ""):
    """丢掉当前进程（下次请求会重新起一个）。"""
    proc = _faceid_proc.get("proc")
    _faceid_proc.update({"proc": None, "q": None, "ready": False, "warming": False})
    if reason:
        _faceid_proc["error"] = reason
    if proc is not None:
        try:
            proc.kill()
        except Exception:
            pass


def _faceid_spawn():
    """起常驻引擎进程（已有活的就复用）。**必须**在 _faceid_proc["lock"] 里调。"""
    if _faceid_alive():
        return _faceid_proc["proc"]
    with _faceid_spawn_lock:
        if _faceid_alive():
            return _faceid_proc["proc"]
        py = _engine_python()
        if not py.is_file():
            raise RuntimeError("找不到引擎 python: %s" % py)
        env = dict(os.environ)
        env["PYTHONIOENCODING"] = "utf-8"
        env["PYTHONUNBUFFERED"] = "1"
        t0 = time.time()
        proc = _sp.Popen([str(py), "-m", "server._faceid_cli", "--serve"],
                         cwd=str(Path(__file__).resolve().parents[1]), env=env,
                         creationflags=_NO_WINDOW,
                         stdin=_sp.PIPE, stdout=_sp.PIPE, stderr=_sp.DEVNULL,
                         text=True, encoding="utf-8", errors="replace", bufsize=1)
        q = queue.Queue()
        threading.Thread(target=_faceid_reader, args=(proc, q), daemon=True,
                         name="faceid-out").start()
        _faceid_proc.update({"proc": proc, "q": q, "error": "", "ready": False,
                             "boot_ms": int((time.time() - t0) * 1000)})
        return proc


def _faceid_exchange(req: dict, timeout: float = 1800.0) -> dict:
    """发一条请求、收一条结果。协议：@@R <json> 结果 / @@W <json> 预热完成 / @@ERR <msg>。

    进程是单线程的，请求必须排队。但**预热里可能在建引擎（75 秒）**，
    这时候宁可让调用方退回一次性 subprocess（约 2 秒），也不要让点击等 75 秒。
    """
    wait = 1.0 if _faceid_proc.get("warming") else 300.0
    if not _faceid_proc["lock"].acquire(timeout=wait):
        raise RuntimeError("引擎进程正忙（预热/上一次聚类还没结束）")
    try:
        proc = _faceid_spawn()
        q = _faceid_proc["q"]
        try:
            proc.stdin.write(json.dumps(req, ensure_ascii=False) + "\n")
            proc.stdin.flush()
        except Exception as e:
            _faceid_reset("%s: %s" % (type(e).__name__, e))
            raise RuntimeError("引擎进程写入失败: %s" % e)
        while True:
            try:
                line = q.get(timeout=timeout)
            except queue.Empty:
                _faceid_reset("引擎进程 %d 秒没有响应" % int(timeout))
                raise RuntimeError("人脸身份引擎超时（%d 秒）" % int(timeout))
            if line is None:
                code = proc.poll()
                err = _faceid_proc.get("error") or ""
                _faceid_reset(err or "引擎进程退出（%s）" % code)
                raise RuntimeError(err or "人脸身份引擎进程意外退出（%s）" % code)
            line = line.strip()
            if line.startswith("@@R "):
                _faceid_proc["requests"] += 1
                data = json.loads(line[4:])
                # 跑成功就说明引擎已经在常驻进程里加载好了 —— 后面点击不再付初始化
                _faceid_proc.update({"ready": True,
                                     "provider": data.get("provider") or _faceid_proc.get("provider") or "",
                                     "engine_ms": data.get("ms_engine_load")})
                return data
            if line.startswith("@@W "):
                try:
                    info = json.loads(line[4:])
                except Exception:
                    info = {}
                _faceid_proc.update({"ready": True, "warming": False,
                                     "warm_ms": info.get("ms"),
                                     "engine_ms": info.get("ms"),
                                     "provider": info.get("provider") or ""})
                return info
            if line.startswith("@@ERR "):
                raise RuntimeError(line[6:])
            # 其它行是引擎自己的 print，忽略
    finally:
        try:
            _faceid_proc["lock"].release()
        except RuntimeError:
            pass


def _faceid_warm(timeout: float = 900.0) -> dict:
    """把引擎加载（反序列化 / 建会话）提前做掉，之后的点击几乎零等待。"""
    _faceid_proc["warming"] = True
    try:
        return _faceid_exchange({"cmd": "warm"}, timeout=timeout)
    finally:
        _faceid_proc["warming"] = False


def _faceid_prewarm_async() -> bool:
    """后台预热；已经在预热/已经热了就返回 False。"""
    if _faceid_proc.get("ready") and _faceid_alive():
        return False
    if _faceid_proc.get("warming"):
        return False

    def _work():
        try:
            _faceid_warm()
        except Exception as e:
            _faceid_proc["error"] = "%s: %s" % (type(e).__name__, e)

    threading.Thread(target=_work, daemon=True, name="faceid-warm").start()
    return True


def _faceid_one_shot(root, eps, min_samples, merge_sim, prog) -> dict:
    """兜底：进程起不来时还是走一次性的 subprocess（慢，但一定能跑）。"""
    py = _engine_python()
    cmd = [str(py), "-m", "server._faceid_cli", "--dir", str(root),
           "--eps", str(eps), "--min-samples", str(min_samples),
           "--merge", str(merge_sim)]
    if prog:
        cmd += ["--progress", str(prog)]
    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"
    proc = _sp.run(cmd, capture_output=True, text=True, timeout=3600,
                   cwd=str(Path(__file__).resolve().parents[1]), env=env,
                   creationflags=_NO_WINDOW)
    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout or "").strip().splitlines()
        raise HTTPException(500, "人脸身份推理失败: " + (detail[-1] if detail else "unknown"))
    return json.loads(proc.stdout.strip().splitlines()[-1])


@router.post("/api/dataset/faceid/prewarm")
def faceid_prewarm(payload: dict = None):
    """预热：起常驻进程并把 ArcFace 引擎加载好（不阻塞，进度看 /api/dataset/faceid/warm）。"""
    payload = payload or {}
    t = time.time()
    started = _faceid_prewarm_async()
    return {"ok": True, "started": bool(started), "ready": bool(_faceid_proc.get("ready")),
            "warming": bool(_faceid_proc.get("warming")),
            "ms": int((time.time() - t) * 1000),
            "alive": _faceid_alive()}


@router.get("/api/dataset/faceid/warm")
def faceid_warm_status():
    return {"ok": True, "ready": bool(_faceid_proc.get("ready")),
            "warming": bool(_faceid_proc.get("warming")),
            "alive": _faceid_alive(), "boot_ms": _faceid_proc.get("boot_ms"),
            "engine_ms": _faceid_proc.get("engine_ms"), "warm_ms": _faceid_proc.get("warm_ms"),
            "provider": _faceid_proc.get("provider") or "",
            "requests": _faceid_proc.get("requests"), "error": _faceid_proc.get("error") or ""}


atexit.register(_faceid_reset)


@router.post("/api/dataset/merge-subdirs")
def dataset_merge_subdirs(payload: dict):
    """把一级子目录里的图片搬回 aligned 根目录（照搬 4.0 的「合并子文件夹」）。

    4.0 的做法（FacesetProcessor/Filter.py: merge_subfolders_to_aligned）：
      * 只看**一级**子目录，不递归；
      * 冲突直接覆盖（move /y）；
      * 原本就空的目录保留，搬空的目录删掉。
    这里多一条：跳过 _trash（我们自己的回收站，合回来等于撤销过滤）。
    """
    import shutil as _sh

    target = _norm_target(payload.get("dir") or "data_src")
    root, _redirected = _resolve_root(target)

    exts = {".jpg", ".jpeg", ".png", ".bmp"}

    def _skip(d):
        """_trash 是回收站、点开头是缓存，两者都不动；其余（含 _by_id）都算待合并。"""
        n = d.name.lower()
        return n.startswith("_trash") or d.name.startswith(".")

    # 递归收集所有层级的子目录。4.0 只处理一层，但用户真正会遇到的正是
    # aligned/_by_id/g00 这种多套一层的情况 —— 只扫一层等于什么都没做。
    subdirs = []
    stack = [d for d in root.iterdir() if d.is_dir() and not _skip(d)]
    while stack:
        d = stack.pop()
        subdirs.append(d)
        try:
            stack.extend(x for x in d.iterdir() if x.is_dir() and not _skip(x))
        except OSError:
            pass
    if not subdirs:
        return {"ok": True, "merged": 0, "conflicts": 0, "renamed": 0, "errors": 0,
                "removed": [], "kept_empty": [], "dirs": 0}

    def imgs(d):
        try:
            return [f for f in d.iterdir() if f.is_file() and f.suffix.lower() in exts]
        except OSError:
            return []

    # 整棵子树里一张图都没有的目录视为「用户自己建的空壳」，保留不动
    def has_images_under(d):
        if imgs(d):
            return True
        try:
            return any(has_images_under(x) for x in d.iterdir() if x.is_dir() and not _skip(x))
        except OSError:
            return False

    # 「用户自己建的空壳」= 子树里没图、自己也一个文件都没有；这种保留。
    # 只剩 .cache/ 的那种是我们自己的扫描缓存留下的空壳，要清掉。
    def has_files_under(d):
        try:
            return any(x.is_file() for x in d.rglob("*"))
        except OSError:
            return True
    keep = set()
    for d in subdirs:
        if not has_images_under(d) and not has_files_under(d):
            keep.update([d] + [x for x in subdirs if d in x.parents])

    files = [f for d in subdirs for f in imgs(d)]

    merged = conflicts = renamed = errors = 0
    for f in files:
        dst = root / f.name
        if dst.exists():
            # 4.0 是 move /y 直接盖掉；这里改成加后缀保留，避免两个不同文件同名时静默丢数据
            k2 = 1
            while dst.exists():
                dst = root / ("%s_%d%s" % (Path(f.name).stem, k2, Path(f.name).suffix))
                k2 += 1
            renamed += 1
        if _move_file(f, dst):
            merged += 1
        else:
            errors += 1

    removed = []
    # 从最深的开始删，父目录才有机会也跟着空掉
    for d in sorted(subdirs, key=lambda x: len(x.parts), reverse=True):
        if d in keep:
            continue
        try:
            d.rmdir()                      # 非空会抛，正好跳过
        except OSError:
            # 只剩我们自己写的缓存目录（.cache / __pycache__）时也删掉：
            # 里面的特征缓存对应的是已被搬走的文件，留着没用，还会挡住空目录清理
            try:
                rest = [x for x in d.iterdir() if x.name not in ("__pycache__",)]
                if not rest:
                    for x in d.iterdir():
                        _sh.rmtree(str(x), ignore_errors=True)
                    d.rmdir()
                    removed.append(d.name)
            except OSError:
                pass
        except OSError:
            pass

    return {"ok": True, "merged": merged, "conflicts": conflicts, "renamed": renamed,
            "removed": removed, "kept_empty": sorted({d.name for d in keep}),
            "dirs": len(subdirs)}


def _unavailable(cap: str, st: dict, root) -> dict:
    """能力没装：回一份「状态」，别用 409 让界面以为出错。

    形状和正常扫描结果对齐（total/hits/count 都在），前端不用为它写一套分支：
    看到 unavailable=true 就提示「未安装 + 怎么装」，并且绝不进入移动文件那一步。
    """
    st = st or {}
    return {
        "ok": False, "unavailable": True, "capability": cap,
        "reason": st.get("reason") or "", "install": st.get("install") or "",
        "status": st,
        "total": 0, "count": 0, "hits": [],
        "unreadable": [], "info": {},
        "dir": str(root), "n_files": 0,
        "redirected_to_aligned": False, "skipped_dirs": [],
    }


@router.post("/api/dataset/scan")
def dataset_scan(payload: dict):
    """只统计不落盘：返回会被处理掉的文件清单。

    走 _dataset_fast：一次解码同时算 dhash / 清晰度 / 直方图，线程池并发，
    去重是向量化汉明距离（旧实现是 Python 双层循环，慢两个数量级）。
    """
    mode = payload.get("mode", "blur")
    target = _norm_target(payload.get("dir") or "data_src")
    root, redirected = _resolve_root(target)

    # 唯一口径：这一层的图（不递归、不进 _trash/_masks/分组目录）
    files = FF.list_images(root)
    total = len(files)
    if mode in ("blur", "repeated"):
        # blur / repeated 也走真进度：每算完一张写一次进度文件
        _put = _ds_progress_writer("embed", total)
        recs, bad = FF.extract_cached(root, files, workers=int(payload.get("workers") or 8),
                                      progress=_put)
    else:
        # 人脸身份 / 错脸在各自那条路上算，不用为了 dhash 再全量解码一遍
        # （670 张多花约 1 秒，而且和人脸聚类毫无关系）
        _put = (lambda *a, **k: None)
        recs, bad = [], []
    hits = [{"name": n, "score": 0, "bad": True} for n in bad]     # 坏图在任何模式下都要清掉

    if mode == "blur":
        # 4.0 语义：按本数据集分位淘汰最差的一批（payload.percent，默认 10%）
        pct = float(payload.get("percent", 10.0)) / 100.0
        vals = [r["sharp"] for r in recs]
        thr = FF.percentile_cut(vals, pct)
        for r in recs:
            if r["sharp"] < thr:
                hits.append({"name": r["name"], "score": round(r["sharp"], 4)})
        info = {"threshold": round(thr, 4), "percent": pct * 100}
    elif mode == "repeated":
        sim = float(payload.get("similarity", 0.98))
        dup = FF.find_duplicates([r["dhash"] for r in recs], sim)
        for i, d in enumerate(dup):
            if d >= 0:
                hits.append({"name": recs[i]["name"], "score": round(1.0 - 0.0, 4),
                             "dup_of": recs[d]["name"]})
        info = {"similarity": sim}
    elif mode == "faceid":
        # 需要 onnxruntime + TRT，只能在 engine 环境跑 —— 常驻进程（见 _faceid_exchange）
        eps = float(payload.get("eps", 0.40))
        min_samples = int(payload.get("min_samples", 2))
        merge_sim = float(payload.get("merge") or 0.45)
        _prog = WORKSPACE / ("_faceid_%d.json" % os.getpid())
        try:
            if _prog.exists():
                _prog.unlink()
        except Exception:
            pass
        _FACEID_PROGRESS["path"] = str(_prog)
        t_fid = time.time()
        warm_before = bool(_faceid_proc.get("ready"))
        try:
            data = _faceid_exchange({"dir": str(root), "eps": eps, "min_samples": min_samples,
                                     "merge": merge_sim, "progress": str(_prog)})
            worker = "resident"
        except Exception as e:
            # 常驻进程崩了就退回一次性 subprocess（慢，但结果一样）
            try:
                data = _faceid_one_shot(root, eps, min_samples, merge_sim, _prog)
                worker = "one-shot(%s)" % type(e).__name__
            except HTTPException:
                raise
            except Exception as e2:
                raise HTTPException(500, f"人脸身份推理启动失败: {e2}")
        # 引擎真正处理了几张：以引擎自己的 total 为准（和上面的 count 同源）
        n_engine = int(data.get("total") or data.get("n_files") or len(files))
        total = n_engine
        # 命中项由「按身份分文件夹」决定，这里不再另算
        info = {"clusters": data.get("clusters", {}), "noise": data.get("noise", 0),
                "main": data.get("main"), "per_cluster": data.get("per_cluster"),
                "eps": eps, "min_samples": min_samples,
                "ms_extract": data.get("ms_extract"), "ms_cluster": data.get("ms_cluster"),
                "provider": data.get("provider"),
                # 核对用的三件事：数的是哪个目录、引擎吃到几张、引擎是谁
                "dir": str(root), "n_files": n_engine,
                "n_clustered": len(data.get("assign") or []),
                "ms_engine_load": data.get("ms_engine_load"),
                "engine_rebuilt": data.get("engine_rebuilt"),
                "warm_before": warm_before, "worker": worker,
                "ms_total": int((time.time() - t_fid) * 1000)}

        # 「按身份分文件夹」：每个身份簇一个目录，全部移动（和老逻辑一致）
        _assign = data.get("assign", [])
        groups_n = len({int(it.get("cluster", -1)) for it in _assign if int(it.get("cluster", -1)) >= 0})
        if payload.get("split_by_identity"):
            base = root          # 扁平放在 aligned 下：id001 / id002 / … / noise
            base.mkdir(exist_ok=True)
            moved, by_dir = 0, {}
            for it in data.get("assign", []):
                nm = it.get("file")
                cid = int(it.get("cluster", -1))
                if not nm:
                    continue
                # 组号从 1 开始，宽度至少 3 位（id001），组数超过 999 时自动加宽
                width = max(3, len(str(max(1, groups_n))))
                name = (("id%0" + str(width) + "d") % (cid + 1)) if cid >= 0 else "noise"
                d = base / name
                d.mkdir(exist_ok=True)
                src = root / nm
                if not src.exists():
                    continue
                dst = d / nm
                k2 = 1
                while dst.exists():
                    dst = d / ("%s_%d%s" % (Path(nm).stem, k2, Path(nm).suffix))
                    k2 += 1
                if _move_file(src, dst):
                    moved += 1
                    by_dir[name] = by_dir.get(name, 0) + 1
            info["split"] = {"dir": str(base), "moved": moved, "by_dir": by_dir, "groups": len(by_dir)}

    elif mode == "errface":
        # 68 点几何分类（xgboost）。数据集的 68 点必须是 2D106 真点；
        # 模板点数据集方差≈0，直接拒绝，避免"全部合格"的假阴性。
        from . import _errface
        # 先看能力状态：模型/运行库没装是「没装」，不是「坏了」——
        # 以前一路走到 scan() 才抛 409「缺 errface_xgb.ubj」，点一下就像出故障。
        st = _errface.status()
        if not st.get("available"):
            return _unavailable("errface", st, root)
        res = _errface.scan(str(root), threshold=float(payload.get("threshold", 0.5)),
                            percent=float(payload.get("percent") or 0),
                            estimate=bool(payload.get("estimate", True)))
        if not res.get("ok") and res.get("unavailable"):
            return _unavailable("errface", res.get("status") or st, root)
        if not res.get("ok"):
            raise HTTPException(status_code=409, detail=res.get("error") or "错脸检测不可用")
        for h in res.get("hits", []):
            hits.append({"name": h["file"], "score": h["prob"], "prob": h["prob"]})
        info = {"counts": res.get("counts"), "stats": res.get("stats"),
                "variance": res.get("variance"), "threshold": res.get("threshold"),
                "skipped": res.get("skipped"), "skipped_n": res.get("skipped_n", 0),
                "estimated_n": res.get("estimated_n", 0)}
    else:
        raise HTTPException(status_code=400, detail="未知模式")

    if mode in ("blur", "repeated"):
        _put(total, total, "done")
    return {"total": total, "hits": hits[:5000], "count": len(hits),
            "unreadable": bad[:200], "info": info,
            # 数字来源一次性说清楚：数的是哪个目录、一层几张、跳过了哪些子目录
            "dir": str(root), "n_files": total,
            "redirected_to_aligned": bool(redirected),
            "skipped_dirs": [s for s in FF.skipped_subdirs(root) if s["images"]]}


@router.post("/api/dataset/sort")
def dataset_sort(payload: dict):
    """按指定特征排序并重命名（sorted_00001_原名.jpg），可逆。"""
    by = payload.get("by") or "blur"
    target = _norm_target(payload.get("dir") or "data_src")
    root, _redirected = _resolve_root(target)

    files = FF.list_images(root)
    if not files:
        return {"ok": True, "count": 0}

    def order_key():
        if by == "origname":
            # 去掉 sorted_XXXXX_ 前缀后按原名排（等于「恢复原始顺序」）
            import re
            def base(n):
                return re.sub(r"^sorted_\d+_", "", n)
            return sorted(files, key=lambda p: base(p.name))
        recs, _bad = FF.extract_cached(root, files, workers=8)
        by_name = {r["name"]: r for r in recs}
        if by == "blur":
            return sorted(files, key=lambda p: -by_name[p.name]["sharp"])
        if by == "phash":
            # 按「与位中位数参考哈希的汉明距离」排序：同类脸聚在一起，
            # 而不是按哈希数值大小排（那等于随机顺序）
            arr = np.array([by_name[p.name]["dhash"] for p in files], dtype=np.uint64)
            bits = np.unpackbits(arr.view(np.uint8).reshape(-1, 8), axis=1)
            ref = (bits.mean(axis=0) >= 0.5).astype(np.uint8)
            dist = (bits != ref).sum(axis=1)
            return [files[i] for i in np.argsort(dist, kind="stable")]
        if by == "hist":
            # sum(..., None) 会拿 None 去加 ndarray，直接 TypeError（历史遗留）
            mean = np.stack([by_name[p.name]["hist"] for p in files]).mean(axis=0)
            return sorted(files, key=lambda p: -float((by_name[p.name]["hist"] * mean).sum()))
        if by == "yaw":
            from ._align import read_dfljpg_meta
            def yaw(p):
                m = read_dfljpg_meta(p) or {}
                v = m.get("yaw")
                return float(v) if isinstance(v, (int, float)) else 0.0
            return sorted(files, key=yaw)
        return sorted(files)

    ordered = order_key()

    # 两段式重命名；已经就位的文件直接跳过（重复排序几乎零成本）
    import re
    plan = []
    for i, p in enumerate(ordered):
        base = re.sub(r"^sorted_\d+_", "", p.name)
        want = "sorted_%05d_%s" % (i, base)
        if p.name != want:
            plan.append((p, want))

    final = []
    tmp = []
    from concurrent.futures import ThreadPoolExecutor

    def _stage(item):
        i, (p, want) = item
        t = p.with_name("__tmp_%06d_%s" % (i, p.name))
        p.rename(t)
        return (t, want)

    if plan:
        with ThreadPoolExecutor(max_workers=8) as ex:
            tmp = list(ex.map(_stage, enumerate(plan)))
    for t, want in tmp:
        t.rename(t.with_name(want))
        final.append(want)

    return {"ok": True, "count": len(final), "by": by}


# ---------------- 遮挡率（排序器，不是过滤器） ----------------
# 用户口径：68 点凸包**不含眉毛**（只用眼睛及以下），算 XSeg 遮罩在凸包内的
# 软面积占比；全被遮住 = 100%，一点没挡 = 0%。纯 numpy+cv2，不加载模型、
# 不起引擎进程 —— 实测 673 张 0.47~0.53 s（冷盘 0.91 s）、3345 张 1.9 s。
@router.post("/api/dataset/occlusion")
def dataset_occlusion(payload: dict = None):
    """action=scan：返回按遮挡率降序的排序清单 + 统计；action=move：按阈值移进 _trash。"""
    payload = payload or {}
    action = str(payload.get("action") or "scan").strip().lower()
    if action not in ("scan", "move"):
        raise HTTPException(status_code=400, detail="未知 action：%s" % action)
    target = _norm_target(payload.get("dir") or "data_src")
    root, redirected = _resolve_root(target)

    if action == "move":
        # 阈值用 0..1 小数（前端显示百分比）。移动前**重新扫一遍**：
        # 目录可能在这一瞬间被别的操作动过，拿前端缓存的清单去搬会搬错文件。
        try:
            thr = float(payload.get("threshold", 0.5))
        except (TypeError, ValueError):
            raise HTTPException(status_code=400, detail="阈值必须是 0..1 的小数")
        thr = max(0.0, min(1.0, thr))
        res = OCC.scan(root)
        hits = [it for it in res["items"] if it["valid"] and float(it["occ"]) > thr]
        mv = OCC.move_to_trash(root, [it["name"] for it in hits],
                               payload.get("target") or "_trash")
        occ_of = {it["name"]: it["occ"] for it in hits}
        items = [{"name": m["name"], "occ": occ_of.get(m["name"]), "trash": m["trash"]}
                 for m in mv["items"]]
        manifest = OCC.write_manifest(mv["trash"], {
            "kind": "occlusion", "dir": str(root), "threshold": thr,
            "at": time.strftime("%Y-%m-%d %H:%M:%S"), "count": mv["moved"],
            "items": items,
        })
        return {"ok": True, "action": "move", "dir": str(root), "target": target,
                "threshold": thr, "scanned": res["total"], "count": len(hits),
                "moved": mv["moved"], "items": items, "failed": mv["failed"],
                "trash": mv["trash"], "manifest": manifest, "ms": res["ms"]}

    res = OCC.scan(root)
    return {"ok": True, "action": "scan", "dir": str(root), "target": target,
            "total": res["total"], "valid": res["stats"]["n"],
            "items": res["items"], "stats": res["stats"], "ms": res["ms"],
            "redirected_to_aligned": bool(redirected),
            "skipped_dirs": [s for s in FF.skipped_subdirs(root) if s["images"]]}


@router.get("/api/dataset/thumb")
def dataset_thumb(dir: str = "data_src", name: str = "", s: int = 96):
    """排序网格的缩略图（方形裁切 + JPEG）。只读一个文件，别把整层扫一遍。

    解码按原图尺寸挑降采样档：JPEG 的 1/2、1/4 解码由 libjpeg 直接跳过 DCT 系数，
    比整张解完再缩快 2~3 倍（96px 缩略图看不出差别），单张 0.3~1.1 ms。
    """
    import cv2
    target = _norm_target(dir or "data_src")
    root, _redirected = _resolve_root(target)
    n = str(name or "")
    if not n or "/" in n or "\\" in n or ":" in n or n in (".", ".."):
        raise HTTPException(status_code=400, detail="非法的文件名")
    p = root / n
    if not p.is_file() or not _within(root, p):
        raise HTTPException(status_code=404, detail="找不到图片")
    side = max(32, min(256, int(s or 96)))
    try:
        raw = p.read_bytes()
    except OSError:
        raise HTTPException(status_code=404, detail="找不到图片")
    wh = OCC.jpeg_size(raw) or (0, 0)
    mind = min(wh) if wh[0] > 0 else 0
    flag = cv2.IMREAD_COLOR
    if mind >= 640:
        flag = cv2.IMREAD_REDUCED_COLOR_4
    elif mind >= 320:
        flag = cv2.IMREAD_REDUCED_COLOR_2
    try:
        img = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), flag)
    except Exception:
        img = None
    if img is None:
        raise HTTPException(status_code=415, detail="图片解不开")
    h, w = img.shape[:2]
    k = min(h, w)
    y0, x0 = (h - k) // 2, (w - k) // 2
    small = cv2.resize(img[y0:y0 + k, x0:x0 + k], (side, side),
                       interpolation=cv2.INTER_AREA)
    ok, out = cv2.imencode(".jpg", small, [int(cv2.IMWRITE_JPEG_QUALITY), 82])
    if not ok:
        raise HTTPException(status_code=500, detail="缩略图编码失败")
    return Response(content=out.tobytes(), media_type="image/jpeg",
                    headers={"Cache-Control": "public, max-age=86400"})


def _within(base: Path, p: Path) -> bool:
    """p 是否真的在 base 里面（挡住 ../ 和绝对路径替换基路径）。"""
    try:
        base_r = base.resolve()
        p_r = p.resolve()
    except Exception:
        return False
    try:
        p_r.relative_to(base_r)
        return True
    except ValueError:
        return False


def _move_file(src: Path, dst: Path) -> bool:
    """只搬不删：先同盘 rename，跨盘再退到 shutil.move，永不 unlink 源文件。"""
    import shutil as _sh

    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.replace(str(src), str(dst))
        return True
    except OSError:
        pass
    try:
        _sh.move(str(src), str(dst))
        return True
    except Exception:
        return False


@router.post("/api/dataset/apply")
def dataset_apply(payload: dict):
    """把命中的文件移到 _trash（可恢复）。

    注意：这个装饰器以前被错挂在 _within(base, p) 上面，FastAPI 于是把两个 Path
    参数当成路径参数 —— 任何调用都返回 422，清理功能从来没通过。

    搬运逻辑只有一份：_occlusion.move_to_trash（遮挡率排序器的「移走」走同一个
    函数，所以重名改名 / 越界判断 / 只搬不删的规矩两处天然一致）。
    """
    mode = payload.get("mode", "blur")
    # 只允许 workspace 下的相对目录名：绝对路径会把基路径整个替换掉
    target = _norm_target(payload.get("dir") or "data_src")
    names = payload.get("names") or []
    root, _redirected = _resolve_root(target)
    res = OCC.move_to_trash(root, names, "_trash")
    return {"ok": True, "moved": res["moved"], "items": res["items"],
            "failed": res["failed"], "trash": res["trash"], "mode": mode}


@router.post("/api/dataset/restore")
def dataset_restore(payload: dict):
    """把刚移走的文件放回去（撤销）。"""
    target = _norm_target(payload.get("dir") or "data_src")
    items = payload.get("items") or []
    root, _redirected = _resolve_root(target)
    trash = root / "_trash"
    back = 0
    failed = []
    for it in items:
        tn = str(it.get("trash", ""))
        nm = str(it.get("name", ""))
        if not tn or not nm:
            continue
        if "/" in tn or "\\" in tn or ":" in tn or "/" in nm or "\\" in nm or ":" in nm:
            continue
        src = trash / tn
        dst = root / nm
        if not src.exists() or not _within(trash, src) or not _within(root, dst):
            continue
        if dst.exists():
            continue
        if not _move_file(src, dst):
            failed.append(nm)
            continue
        back += 1
    return {"ok": True, "restored": back, "failed": failed}
