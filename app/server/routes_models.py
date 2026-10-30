# -*- coding: utf-8 -*-
"""模型库 / 规格 / GPU（从 app.py 按路径前缀抽出）。"""
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

from .app import (MODEL_DIR, WORKSPACE, _CFG_STRUCTURAL, _DFL_PY, _DFL_TOOL, _HOT_OPTIONS, _NO_WINDOW, _PRESETS, _arch_of, _archi_spec, _export_job, _export_job_lock, _export_worker, _find_model_preview, _gpu_cached, _gpu_list, _model_dat, _model_entry, _recommend_for, _safe_model_name, _spec_tables, _train_job, _train_job_lock, _training_pid, _webui_post)

router = APIRouter()

@router.get("/api/models/{name}/config")
def model_config_get(name: str):
    """读一个模型当前保存的训练参数。"""
    import pickle
    dat = _model_dat(name)
    if dat is None:
        raise HTTPException(404, "找不到模型 " + name)
    try:
        data = pickle.loads(dat.read_bytes())
    except Exception as e:
        raise HTTPException(500, "读取 _data.dat 失败: " + str(e))
    opts = data.get("options") or {}
    return {"name": name, "iter": data.get("iter", 0),
            "options": {k: v for k, v in opts.items() if isinstance(v, (int, float, str, bool)) or v is None},
            "readonly": sorted([k for k in opts if k in _CFG_STRUCTURAL])}


@router.post("/api/models/{name}/config")
def model_config_set(name: str, body: dict):
    """写回训练参数。结构性字段拒绝，其余原样合并。"""
    import pickle
    dat = _model_dat(name)
    if dat is None:
        raise HTTPException(404, "找不到模型 " + name)
    patch = (body or {}).get("options") or {}
    if not patch:
        raise HTTPException(400, "没有要写入的参数")

    blocked = [k for k in patch if k in _CFG_STRUCTURAL]
    if blocked:
        raise HTTPException(400, "这些是创建后不可改的结构参数: " + ", ".join(blocked))

    try:
        data = pickle.loads(dat.read_bytes())
    except Exception as e:
        raise HTTPException(500, "读取 _data.dat 失败: " + str(e))

    opts = data.get("options") or {}

    # 按原字段的类型收敛：界面送上来的都是字符串，"" 表示「这项没填，别动」
    bad = []
    clean = {}
    for k, v in list(patch.items()):
        cur = opts.get(k)
        if isinstance(v, str):
            v = v.strip()
            if v == "":
                continue                      # 空着 = 不改这一项
        try:
            if isinstance(cur, bool):
                if isinstance(v, str):
                    v = v.lower() in ("1", "true", "yes", "on", "开")
                else:
                    v = bool(v)
            elif isinstance(cur, int) and not isinstance(cur, bool):
                v = int(round(float(v)))
            elif isinstance(cur, float):
                v = float(v)
            elif isinstance(cur, str):
                v = str(v)
        except (TypeError, ValueError):
            bad.append(k)
            continue
        clean[k] = v
    if bad:
        raise HTTPException(400, "这些参数不是有效数字: " + ", ".join(bad))
    patch = clean
    if not patch:
        return {"ok": True, "name": name, "changed": {}, "count": 0}

    backup = dat.with_suffix(".dat.bak")
    try:
        backup.write_bytes(dat.read_bytes())
    except Exception:
        pass

    had_keys = set(opts.keys())
    changed = {}
    for k, v in patch.items():
        if opts.get(k) != v:
            changed[k] = {"from": opts.get(k), "to": v}
        opts[k] = v
    data["options"] = opts
    # 非原子写会把 _data.dat 写坏：先落临时文件再换名，中途崩也只是丢弃临时文件
    try:
        blob = pickle.dumps(data)
        tmp = dat.with_suffix(".dat.tmp")
        tmp.write_bytes(blob)
        os.replace(str(tmp), str(dat))
    except Exception as e:
        raise HTTPException(500, "写入失败: " + str(e))
    # 训练中改的参数直接热推给 4.0：它每个迭代都会 poll 一次 update-model-options
    # 只有「模型本来就有这个键」或者「本来就该由 4.1 补上的增强项」才热推；
    # 否则界面默认值（crash_threshold=0 之类）会被当成用户修改写进训练进程。
    # 这三项缺省值（0/3）会悄悄改掉训练行为，模型里没有就不动；
    # 其余热改项（增强、学习率、损失权重）本来就该能补。
    _SKIP_NEW = {"crash_threshold", "max_backups", "backup_interval"}
    hot = [k for k in changed if k in _HOT_OPTIONS and (k in had_keys or k not in _SKIP_NEW)]
    cold = [k for k in changed if k not in _HOT_OPTIONS]
    hot_applied = False
    hot_error = None
    with _train_job_lock:
        live = bool(_train_job["running"]) and _train_job.get("name") == name
    if hot and live:
        try:
            _webui_post("/update-model-options", {k: patch[k] for k in hot})
            hot_applied = True
        except Exception as e:
            hot_error = str(e)[:120]

    return {"ok": True, "name": name, "changed": changed, "count": len(changed),
            "hot": hot, "need_restart": cold,
            "hot_applied": hot_applied, "hot_error": hot_error}


@router.post("/api/models/{name}/delete")
def model_delete(name: str):
    """删掉一个模型的所有文件（权重 / data.dat / 备份目录）。训练中的模型拒绝删。"""
    import shutil as _sh
    name = _safe_model_name(name)
    with _train_job_lock:
        if _train_job["running"] and _train_job.get("name") == name:
            raise HTTPException(409, "这个模型正在训练，先停止再删")
    md = WORKSPACE / "model"
    paths = sorted(md.glob(name + "_*"))
    if not paths:
        raise HTTPException(404, "找不到模型 " + name)
    removed = []
    for p in paths:
        try:
            if p.is_dir():
                _sh.rmtree(p, ignore_errors=True)
            else:
                p.unlink()
            removed.append(p.name)
        except Exception as e:
            raise HTTPException(500, "删除失败 %s: %s" % (p.name, str(e)[:120]))
    return {"ok": True, "name": name, "removed": removed, "count": len(removed)}


@router.post("/api/models/{name}/export")
def model_export(name: str):
    """把一个模型导出成 .dfm（后台跑，前端轮询 /api/models/export/status）。"""
    md = WORKSPACE / "model"
    if not (md / (name + "_" + _arch_of(name) + "_data.dat")).exists():
        # 兜底：直接看有没有 *_{name}_*_data.dat
        cands = list(md.glob(name + "_*_data.dat"))
        if not cands:
            raise HTTPException(404, "找不到模型 " + name)
    if not _DFL_PY.exists() or not _DFL_TOOL.exists():
        raise HTTPException(500, "找不到 4.0 的 python 或导出脚本")
    with _export_job_lock:
        if _export_job["running"]:
            raise HTTPException(409, "已经在导出 " + _export_job["name"])
    arch = _arch_of(name)
    threading.Thread(target=_export_worker, args=(name, arch), daemon=True).start()
    return {"ok": True, "name": name, "arch": arch, "running": True}


@router.get("/api/models/export/status")
def model_export_status():
    with _export_job_lock:
        return {"running": _export_job["running"], "name": _export_job["name"],
                "ok": _export_job["ok"], "error": _export_job["error"],
                "log": _export_job["log"][-40:]}


@router.get("/api/model_spec")
def model_spec():
    """新建模型页要用的全部下拉/范围/推荐值，一次拿全。"""
    tables = _spec_tables()
    gpus = _gpu_cached()
    rec = _recommend_for(max([g["free_gb"] for g in gpus], default=0.0))
    out = dict(tables)
    out["recommend"] = rec
    out["gpu"] = max(gpus, key=lambda g: g["free_gb"]) if gpus else None
    return out


@router.get("/api/gpu")
def gpu_info():
    gpus = _gpu_cached()
    best = max(gpus, key=lambda g: g["free_gb"]) if gpus else None
    rec = _recommend_for(best["free_gb"] if best else 0.0)
    presets = []
    for k, v in _PRESETS.items():
        presets.append({kk: vv for kk, vv in v.items() if kk != "weights_gb"} | {"key": k})
    return {"gpus": gpus, "best": best, "recommend": rec, "presets": presets}


@router.post("/api/models/create")
def model_create(body: dict):
    """按档位建一个模型（写一份 iter=1 的 data.dat，4.0 见到就当作已有模型）。"""
    import pickle as _pk
    body = body or {}
    name = _safe_model_name(body.get("name"))
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    dat_path = MODEL_DIR / (name + "_SAEHD_data.dat")
    if dat_path.exists():
        raise HTTPException(409, "已经有同名模型：" + name + "（换个名字，或直接去训练页继续练）")

    gpus = _gpu_list()
    free = max([g["free_gb"] for g in gpus], default=0.0)
    rec = _recommend_for(free)

    preset_key = str(body.get("preset") or "").strip()
    if preset_key not in _PRESETS:
        preset_key = rec["preset"]
    p = dict(_PRESETS[preset_key])

    def _pick(key, cast, fallback):
        v = body.get(key)
        if v is None or v == "":
            return fallback
        try:
            return cast(v)
        except Exception:
            return fallback

    # 架构字符串：优先用「架构 + 子分支 + 算子」三段式（和 4.0 的 GUI 同一套规则），
    # 也允许直接传 archi。合法性交给 4.0 的 parse_archi_string 判，不在 4.1 里另写一份。
    spec = _archi_spec()
    archi_display = str(body.get("archi_display") or "").strip()
    if archi_display:
        archi = spec.build_archi_string(archi_display,
                                        str(body.get("subarchi") or ""),
                                        str(body.get("kernel_label") or spec.default_kernel_label()))
        if spec.parse_archi_string(archi) is None:
            raise HTTPException(400, "架构组合不合法：" + archi)
    else:
        archi = str(body.get("archi") or p["archi"])
        if spec.parse_archi_string(archi) is None:
            raise HTTPException(400, "架构写法不合法：" + archi)

    def _clamp(v, lo, hi, step):
        v = max(lo, min(hi, int(v)))
        return int(round(v / float(step)) * step)

    options = {
        "resolution": _clamp(_pick("resolution", int, p["resolution"]), 64, 640, 32),
        "face_type": str(body.get("face_type") or "wf"),
        "archi": archi,
        "ae_dims": _clamp(_pick("ae_dims", int, p["ae_dims"]), 64, 1024, 8),
        "e_dims": _clamp(_pick("e_dims", int, p["e_dims"]), 16, 256, 8),
        "d_dims": _clamp(_pick("d_dims", int, p["d_dims"]), 16, 256, 8),
        "d_mask_dims": _clamp(_pick("d_mask_dims", int, p["d_mask_dims"]), 8, 128, 8),
        "batch_size": _pick("batch_size", int, rec["batch_size"]),
        "target_iter": max(0, _pick("target_iter", int, 0)),
        "optimizer": str(body.get("optimizer") or "adabelief"),
        "lr": float(body.get("lr") or 5e-05),
        "lr_dropout": "n",
        "masked_training": True,
        "eyes_mouth_prio": False,
        "uniform_yaw": False,
        "ct_mode": "none",
        "random_warp": True,
        "gan_power": 0.0,
        "true_face_power": 0.0,
        "pretrain": False,
        "use_fast_generator": True,
        "models_opt_on_gpu": True,
    }
    for k in ("src", "dst"):
        v = str(body.get(k) or "").strip()
        if v:
            options[k] = v

    data = {"iter": 1, "options": options, "loss_history": [],
            "sample_for_preview": None, "choosed_gpu_indexes": None}
    tmp = dat_path.with_name(dat_path.name + ".tmp")
    tmp.write_bytes(_pk.dumps(data, protocol=4))
    os.replace(str(tmp), str(dat_path))

    # 顺手写 4.0 的参数缓存：模型库和下次启动都不用再解析 data.dat
    try:
        st = dat_path.stat()
        cache = {"version": 1, "iter": 1, "options": options,
                 "choosed_gpu_indexes": None, "loss_count": 0,
                 "src_size": st.st_size, "src_mtime_ns": st.st_mtime_ns}
        Path(str(dat_path) + ".params").write_bytes(_pk.dumps(cache, protocol=4))
    except Exception:
        pass

    return {"ok": True, "name": name, "preset": preset_key,
            "dat": str(dat_path),
            "options": {k: options[k] for k in
                        ("resolution", "batch_size", "archi", "face_type",
                         "ae_dims", "e_dims", "d_dims", "d_mask_dims")},
            "note": p["note"]}


@router.get("/api/models/{name}/stats")
def model_stats(name: str):
    """训练页右侧那几条：显存、利用率、磁盘、上次保存、下次自动、备份数。

    以前这些数字是写死在 HTML 里的（"9.2 / 12 GB"、"98%"、"3 / 5 份"），
    页面看起来一直很健康，其实和真相无关。这里按模型现场算。
    """
    import pickle as _pk
    name = _safe_model_name(name)
    dat = _model_dat(name)
    if dat is None:
        raise HTTPException(404, "找不到模型 " + name)

    opts = {}
    try:
        opts = (_pk.loads(dat.read_bytes()).get("options") or {})
    except Exception:
        opts = {}

    # GPU：训练时显存几乎满，用 nvidia-smi 的实时值
    gpu = None
    try:
        import shutil as _sh
        import subprocess as _sp
        exe = _sh.which("nvidia-smi") or r"C:\Windows\System32\nvidia-smi.exe"
        if Path(exe).is_file():
            out = _sp.run([exe, "--query-gpu=index,name,memory.total,memory.free,utilization.gpu",
                           "--format=csv,noheader,nounits"],
                          capture_output=True, text=True, timeout=8,
                          creationflags=_NO_WINDOW).stdout
            rows = [x.strip() for x in out.strip().splitlines() if x.strip()]
            if rows:
                p = [x.strip() for x in rows[0].split(",")]
                gpu = {"index": int(p[0]), "name": p[1],
                       "total_gb": round(int(p[2]) / 1024.0, 1),
                       "free_gb": round(int(p[3]) / 1024.0, 1),
                       "used_gb": round((int(p[2]) - int(p[3])) / 1024.0, 1),
                       "util": int(p[4]) if p[4].isdigit() else None}
    except Exception:
        gpu = None

    # 磁盘：模型目录所在盘还剩多少 + 这个模型的体积
    disk_free_gb = None
    model_gb = None
    try:
        import shutil as _sh
        disk_free_gb = round(_sh.disk_usage(str(MODEL_DIR)).free / (1024 ** 3), 1)
        total = sum(f.stat().st_size for f in MODEL_DIR.glob(name + "_*") if f.is_file())
        model_gb = round(total / (1024 ** 3), 2)
    except Exception:
        pass

    # 备份：4.0 会写到 <模型>_<arch>_autobackups/ 下
    backup_count = 0
    backup_dir = None
    try:
        cands = [p for p in MODEL_DIR.glob(name + "_*_autobackups") if p.is_dir()]
        if cands:
            backup_dir = max(cands, key=lambda p: p.stat().st_mtime)
            # 4.0 的备份是 autobackups/01、02、03 … 每个编号一个目录，里面才是权重文件。
            # 以前只数顶级文件，所以永远显示 0。
            dirs = [e for e in backup_dir.iterdir() if e.is_dir() and e.name.isdigit()]
            if dirs:
                backup_count = len(dirs)
            else:
                backup_count = len([f for f in backup_dir.iterdir() if f.is_file()])
    except Exception:
        pass

    # CPU / 内存：psutil.cpu_percent(None) 返回「上次调用到现在」的平均占用，
    # 页面每 2 秒轮询一次，正好是这个区间的真实值（第一次调用返回 0）。
    cpu_percent = None
    ram_used_gb = ram_total_gb = ram_percent = None
    try:
        import psutil
        cpu_percent = round(float(psutil.cpu_percent(None)), 1)
        vm = psutil.virtual_memory()
        ram_used_gb = round(vm.used / (1024 ** 3), 1)
        ram_total_gb = round(vm.total / (1024 ** 3), 1)
        ram_percent = round(float(vm.percent), 1)
    except Exception:
        pass

    last_save_ts = None
    try:
        last_save_ts = int(dat.stat().st_mtime)
    except Exception:
        pass

    return {"name": name, "gpu": gpu, "disk_free_gb": disk_free_gb, "model_gb": model_gb,
            "cpu_percent": cpu_percent,
            "ram_used_gb": ram_used_gb, "ram_total_gb": ram_total_gb, "ram_percent": ram_percent,
            "backup_count": backup_count, "has_backup_dir": backup_dir is not None,
            "backup_interval": int(opts.get("backup_interval") or 0),
            "max_backups": int(opts.get("max_backups") or 0)}


@router.get("/api/models")
def models():
    out = []
    if MODEL_DIR.is_dir():
        for dat in sorted(MODEL_DIR.glob("*_data.dat")):
            try:
                out.append(_model_entry(dat))
            except Exception:
                continue
    out.sort(key=lambda m: m.get("trained_at", 0), reverse=True)
    # 每个模型的状态：正在训练 / 已暂停 / 已停止。前端卡片上的标签直接用它
    with _train_job_lock:
        cur = _train_job.get("name") if _train_job.get("running") else ""
        paused = bool(_train_job.get("paused"))
    for m in out:
        if cur and m.get("name") == cur:
            m["state"] = "已暂停" if paused else "训练中"
        else:
            m["state"] = "已停止"
    pid = _training_pid()
    alive = False
    if pid:
        try:
            import psutil  # noqa: F401
            alive = psutil.pid_exists(pid)
        except Exception:
            import os as _os
            try:
                _os.kill(pid, 0)
                alive = True
            except Exception:
                alive = False
    return {"models": out, "training_pid": pid if alive else 0}


@router.get("/api/models/{name}/preview.png")
def model_preview(name: str):
    from fastapi.responses import FileResponse
    path, mime = _find_model_preview(name)
    if path is None:
        raise HTTPException(status_code=404)
    return FileResponse(str(path), media_type=mime)
