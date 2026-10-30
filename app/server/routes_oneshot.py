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
    total = len(files)
    recs, bad = FF.extract(files, workers=int(payload.get("workers") or 8))
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

        eps = float(payload.get("eps", 0.314))
        min_samples = int(payload.get("min_samples", 2))
        # 用 -m 方式启动：_faceid_cli 用了相对导入（from . import ...）
        cmd = [engine_python(), "-m", "server._faceid_cli", "--dir", str(root),
               "--eps", str(eps), "--min-samples", str(min_samples)]
        try:
            proc = _sp.run(cmd, capture_output=True, text=True, timeout=3600,
                           cwd=str(Path(__file__).resolve().parents[1]))
        except Exception as e:
            raise HTTPException(500, f"人脸身份推理启动失败: {e}")
        if proc.returncode != 0:
            detail = (proc.stderr or proc.stdout or "").strip().splitlines()
            raise HTTPException(500, "人脸身份推理失败: " + (detail[-1] if detail else "unknown"))
        data = _json.loads(proc.stdout.strip().splitlines()[-1])
        for n in data.get("rare", []):
            hits.append({"name": n, "score": 0, "rare_identity": True})
        info = {"clusters": data.get("clusters", {}), "noise": data.get("noise", 0),
                "eps": eps, "min_samples": min_samples,
                "ms_extract": data.get("ms_extract"), "ms_cluster": data.get("ms_cluster"),
                "provider": data.get("provider")}
    else:
        raise HTTPException(status_code=400, detail="未知模式")

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
        recs, _bad = FF.extract(files, workers=8)
        by_name = {r["name"]: r for r in recs}
        if by == "blur":
            return sorted(files, key=lambda p: -by_name[p.name]["sharp"])
        if by == "phash":
            return sorted(files, key=lambda p: by_name[p.name]["dhash"])
        if by == "hist":
            mean = sum((by_name[p.name]["hist"] for p in files), None) / max(1, len(files))
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

    # 两段式重命名，避免中间态撞名
    tmp, final = [], []
    for i, p in enumerate(ordered):
        name = p.name
        import re
        name = re.sub(r"^sorted_\d+_", "", name)
        t = p.with_name(f"__tmp_{i:06d}_{name}")
        p.rename(t)
        tmp.append((t, i, name))
    for t, i, name in tmp:
        dst = t.with_name(f"sorted_{i:05d}_{name}")
        t.rename(dst)
        final.append(dst.name)

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
        src.rename(dst)
        items.append({"name": n, "trash": dst.name})
    return {"ok": True, "moved": len(items), "items": items, "trash": str(trash)}


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
        src.rename(dst)
        back += 1
    return {"ok": True, "restored": back}
