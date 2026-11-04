# -*- coding: utf-8 -*-
"""数据集体检（One Shot）（从 app.py 拆出）。"""
from __future__ import annotations

import asyncio
import base64
import io
import json
import os
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

router = APIRouter()

# ---------------- 数据集体检（One Shot） ----------------
# 扫描在 **4.0 的 python** 里跑（_scan_runner.py）：
#   ① 那边带着完整的 numpy/cv2/numba，4.1 只负责起进程、转进度
#   ② 以后人脸 embedding / 聚类也能接在同一条流水线上
# 协议：@@P <done> <total> <stage> 进度、@@R <json> 结果、@@ERR <msg> 错误。
_oneshot_job = {"running": False, "done": 0, "total": 0, "stage": "", "result": None,
                "error": None, "dir": "", "started": 0.0}
_oneshot_lock = threading.Lock()


@router.post("/api/dataset/oneshot")
def dataset_oneshot(payload: dict):
    payload = payload or {}
    target = str(payload.get("dir") or "data_src")
    if os.path.isabs(target) or ".." in target.replace("\\", "/").split("/") or ":" in target:
        raise HTTPException(400, "非法的数据集目录: " + target)
    root = WORKSPACE / target
    if (root / "aligned").is_dir():
        root = root / "aligned"
    if not root.is_dir():
        raise HTTPException(404, "目录不存在：%s" % root)

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

    target = str(payload.get("dir") or "data_src")
    if os.path.isabs(target) or ".." in target.replace(chr(92), "/").split("/") or ":" in target:
        raise HTTPException(400, "非法的数据集目录: " + target)
    root = WORKSPACE / target
    if (root / "aligned").is_dir():
        root = root / "aligned"
    if not root.exists():
        raise HTTPException(404, "目录不存在: " + str(root))

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


@router.post("/api/dataset/scan")
def dataset_scan(payload: dict):
    """只统计不落盘：返回会被处理掉的文件清单。

    走 _dataset_fast：一次解码同时算 dhash / 清晰度 / 直方图，线程池并发，
    去重是向量化汉明距离（旧实现是 Python 双层循环，慢两个数量级）。
    """
    from . import _dataset_fast as FF

    mode = payload.get("mode", "blur")
    target = str(payload.get("dir") or "data_src")
    root = WORKSPACE / target
    if (root / "aligned").is_dir():
        root = root / "aligned"
    if not root.exists():
        raise HTTPException(status_code=404, detail=f"目录不存在：{root}")

    files = FF.list_images(root)
    # blur / repeated 也走真进度：每算完一张写一次进度文件
    _put = _ds_progress_writer("embed", len(files)) if mode in ("blur", "repeated") else (lambda *a, **k: None)
    total = len(files)
    recs, bad = FF.extract_cached(root, files, workers=int(payload.get("workers") or 8), progress=_put)
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
        # 需要 onnxruntime + TRT，只能在 engine 环境跑 —— 走子进程
        import json as _json
        import subprocess as _sp
        import sys as _sys
        from ._paths import engine_python

        eps = float(payload.get("eps", 0.40))
        min_samples = int(payload.get("min_samples", 2))
        # 用 -m 方式启动：_faceid_cli 用了相对导入（from . import ...）
        _prog = WORKSPACE / ("_faceid_%d.json" % os.getpid())
        try:
            if _prog.exists():
                _prog.unlink()
        except Exception:
            pass
        _FACEID_PROGRESS["path"] = str(_prog)
        cmd = [engine_python(), "-m", "server._faceid_cli", "--dir", str(root),
               "--eps", str(eps), "--min-samples", str(min_samples),
               "--merge", str(float(payload.get("merge") or 0.45)),
               "--progress", str(_prog)]
        try:
            proc = _sp.run(cmd, capture_output=True, text=True, timeout=3600,
                           cwd=str(Path(__file__).resolve().parents[1]))
        except Exception as e:
            raise HTTPException(500, f"人脸身份推理启动失败: {e}")
        if proc.returncode != 0:
            detail = (proc.stderr or proc.stdout or "").strip().splitlines()
            raise HTTPException(500, "人脸身份推理失败: " + (detail[-1] if detail else "unknown"))
        data = json.loads(proc.stdout.strip().splitlines()[-1])
        # 命中项由「按身份分文件夹」决定，这里不再另算
        info = {"clusters": data.get("clusters", {}), "noise": data.get("noise", 0),
                "main": data.get("main"), "per_cluster": data.get("per_cluster"),
                "eps": eps, "min_samples": min_samples,
                "ms_extract": data.get("ms_extract"), "ms_cluster": data.get("ms_cluster"),
                "provider": data.get("provider")}

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
        res = _errface.scan(str(root), threshold=float(payload.get("threshold", 0.5)),
                            percent=float(payload.get("percent") or 0),
                            estimate=bool(payload.get("estimate", True)))
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
            "unreadable": bad[:200], "info": info}


@router.post("/api/dataset/sort")
def dataset_sort(payload: dict):
    """按指定特征排序并重命名（sorted_00001_原名.jpg），可逆。"""
    from . import _dataset_fast as FF

    by = payload.get("by") or "blur"
    target = str(payload.get("dir") or "data_src")
    root = WORKSPACE / target
    if (root / "aligned").is_dir():
        root = root / "aligned"
    if not root.exists():
        raise HTTPException(status_code=404, detail=f"目录不存在：{root}")

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
    """
    mode = payload.get("mode", "blur")
    target = str(payload.get("dir") or "data_src")
    # 只允许 workspace 下的相对目录名：绝对路径会把基路径整个替换掉
    if os.path.isabs(target) or ".." in target.replace("\\", "/").split("/") or ":" in target:
        raise HTTPException(400, "非法的数据集目录: " + target)
    names = payload.get("names") or []
    root = WORKSPACE / target
    if (root / "aligned").is_dir():
        root = root / "aligned"
    trash = root / "_trash"
    trash.mkdir(exist_ok=True)
    items = []
    failed = []
    for n in names:
        n = str(n)
        if not n or "/" in n or "\\" in n or n in (".", "..") or ":" in n:
            continue
        src = root / n
        if not src.exists() or not _within(root, src):
            continue
        dst = trash / n
        if not _within(trash, dst):
            continue
        i = 1
        while dst.exists():
            dst = trash / (src.stem + f"_{i}" + src.suffix)
            i += 1
        if not _move_file(src, dst):
            failed.append(n)
            continue
        items.append({"name": n, "trash": dst.name})
    return {"ok": True, "moved": len(items), "items": items, "failed": failed, "trash": str(trash)}


@router.post("/api/dataset/restore")
def dataset_restore(payload: dict):
    """把刚移走的文件放回去（撤销）。"""
    target = str(payload.get("dir") or "data_src")
    if os.path.isabs(target) or ".." in target.replace("\\", "/").split("/") or ":" in target:
        raise HTTPException(400, "非法的数据集目录: " + target)
    items = payload.get("items") or []
    root = WORKSPACE / target
    if (root / "aligned").is_dir():
        root = root / "aligned"
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
