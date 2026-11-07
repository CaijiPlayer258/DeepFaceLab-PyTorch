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

from .app import (MODEL_DIR, WORKSPACE, _CFG_STRUCTURAL, _DFL_PY, _DFL_TOOL, _HOT_OPTIONS, _NO_WINDOW, _PRESETS, _arch_of, _archi_parts, _archi_spec, _export_job, _export_job_lock, _export_worker, _find_model_preview, _gpu_cached, _gpu_list, _model_dat, _model_entry, _read_dat_info, _recommend_for, _safe_model_name, _spec_tables, _train_job, _train_job_lock, _training_pid, _webui_post)

router = APIRouter()

@router.get("/api/models/{name}/config")
def model_config_get(name: str):
    """读一个模型当前保存的训练参数。

    TF 时代（DFL 1.x / 2.0）的 dat 先由 server/_legacy.py 翻成 4.1 的 options 再返回：
    参数页读到的键必须和训练时引擎读到的一致，否则「改了一项没生效」。
    返回值里的 legacy 是那条非阻塞提示用的（迁了几项、忽略了什么、要不要转权重）。
    """
    dat = _model_dat(name)
    if dat is None:
        raise HTTPException(404, "找不到模型 " + name)
    from ._legacy import describe, notice
    try:
        # 参数页只要 options：走 describe（命中缓存就一次 stat），不读整份 dat ——
        # 122MB 的模型整份 pickle 要 0.85s，而这里要的是 50 个键。
        info = describe(dat)
        legacy = notice(dat)
    except Exception as e:
        raise HTTPException(500, "读取 _data.dat 失败: " + str(e))
    opts = info["options"] or {}
    return {"name": name, "iter": info["iter"],
            "options": {k: v for k, v in opts.items() if isinstance(v, (int, float, str, bool)) or v is None},
            "readonly": sorted([k for k in opts if k in _CFG_STRUCTURAL]),
            "legacy": legacy}


@router.get("/api/models/{name}")
def model_detail(name: str):
    """单个模型的现状：卡片字段 + 完整 options + legacy 迁移信息。

    就是 /api/models 里那一条再补上参数页要用的 options —— 界面不用为了
    「这个模型是不是 TF 时代的、迁了几项」再发两个请求。
    """
    name = _safe_model_name(name)
    dat = _model_dat(name)
    if dat is None:
        raise HTTPException(404, "找不到模型 " + name)
    try:
        entry = _model_entry(dat)          # 卡片字段 + 迁移信息（顺手把缓存填好）
    except Exception as e:
        raise HTTPException(500, "读取模型失败: " + str(e))
    from ._legacy import describe
    try:
        info = describe(dat)               # 只要 options：不为了 50 个键去解整份 dat
    except Exception:
        info = {"options": {}, "iter": entry.get("iter", 0)}
    opts = info.get("options") or {}
    return {"name": name, "dat": str(dat), "iter": info.get("iter", entry.get("iter", 0)),
            "options": {k: v for k, v in opts.items()
                        if isinstance(v, (int, float, str, bool)) or v is None},
            "readonly": sorted([k for k in opts if k in _CFG_STRUCTURAL]),
            "legacy": entry.get("legacy") or {"detected": False}, "model": entry}


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
        # 读进来时 options 已经是 4.1 的形状：老 dat 的 learn_mask/ed_ch_dims 在
        # 这里就被换成了 masked_training/e_dims+d_dims，写回去的也是一份干净的
        # 新格式 dat（原文件先落 .dat.bak）。
        data, legacy = _read_dat_info(dat)
    except Exception as e:
        raise HTTPException(500, "读取 _data.dat 失败: " + str(e))

    opts = data.get("options") or {}

    # 与 create 共用同一套解析：类型按现有 options 收敛，空串表示「不改这一项」
    patch, bad = _clean_options(patch, opts)
    if bad:
        raise HTTPException(400, "这些参数不是有效数字: " + ", ".join(bad))
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

    # 保存后 dat 已经是 4.1 格式（上面的迁移结果被写回去了）：迁移那部分提示可以撤掉。
    # 但「权重还是 TF 的 .npy」这件事没变，needs_convert 和命令照旧返回，
    # 界面据此只去掉「已迁移 N 项」那一行，转换提示留着。
    legacy_after = dict(legacy or {})
    legacy_after.update({"detected": False, "source": "saved", "report": [],
                         "counts": {}, "reasons": []})
    return {"ok": True, "name": name, "changed": changed, "count": len(changed),
            "hot": hot, "need_restart": cold,
            "hot_applied": hot_applied, "hot_error": hot_error,
            "legacy": legacy_after}


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


# ============================================================================
#  参数解析协议（create / config 共用同一套）
#
#  前端只做一件事：把页面上的值收集成一个 JSON 发过来。后端负责：
#    * 按 base（预设默认值 / 已有 dat）的类型做类型收敛（"1" -> 1、1 -> True …）
#    * 空串表示「这一项没填，别动」
#    * 结构性字段（分辨率 / 架构 / 各维度 / face_type）不许走 options：
#      创建后改不了，要么走 spec，要么直接拒绝
#  返回里带上「实际生效的 options」，界面按它渲染就不会和磁盘对不上。
# ============================================================================
_STRUCTURAL_KEYS = {
    "resolution", "archi", "e_dims", "ae_dims", "d_dims", "d_mask_dims", "face_type",
}


def _clean_options(raw, base):
    """把前端发来的 JSON options 收敛成能直接写进 dat 的字典。

    raw  : 前端 JSON（值可能是数字/布尔/字符串，也可能是 "1"/"true"）
    base : 默认值或已有 options，用来决定每一项该是什么类型
    返回 (clean, bad)，bad 是类型不对的键名列表。
    """
    clean, bad = {}, []
    if not isinstance(raw, dict):
        return clean, bad
    for k, v in raw.items():
        if k in _STRUCTURAL_KEYS:
            continue                      # 结构参数不在这里改
        if isinstance(v, str):
            v = v.strip()
            if v == "":
                continue                  # 空着 = 不改
        cur = base.get(k)
        try:
            if isinstance(cur, bool):
                # 界面上的开关/分段派出来的是字符串：'bf16' / 'on' / '1' / '开' 都算真，
                # 'fp32' / 'off' / '0' / 'n' 算假。只认 "1" 会把"训练精度=bf16"判成 False。
                _s = str(v).strip().lower()
                clean[k] = v if isinstance(v, bool) else _s in (
                    "1", "true", "yes", "on", "y", "开", "bf16")
            elif isinstance(cur, int) and not isinstance(cur, bool):
                clean[k] = int(round(float(v)))
            elif isinstance(cur, float):
                clean[k] = float(v)
            elif isinstance(cur, str):
                clean[k] = str(v)
            elif isinstance(v, bool):
                clean[k] = v
            elif isinstance(v, int):
                clean[k] = v
            elif isinstance(v, float):
                clean[k] = v
            else:
                clean[k] = v
        except (TypeError, ValueError):
            bad.append(k)
    return clean, bad

@router.post("/api/models/create")
def model_create(body: dict):
    """按档位建一个模型（写一份 iter=1 的 data.dat，4.0 见到就当作已有模型）。"""
    import pickle as _pk
    body = body or {}
    # 协议：{ name, spec:{结构}, options:{训练参数} }；老的平铺写法继续兼容
    spec_in = body.get("spec") if isinstance(body.get("spec"), dict) else {}

    def _s(key):
        v = spec_in.get(key)
        return body.get(key) if (v is None or v == "") else v

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
        v = _s(key)
        if v is None or v == "":
            return fallback
        try:
            return cast(v)
        except Exception:
            return fallback

    # 架构字符串：优先用「架构 + 子分支 + 算子」三段式（和 4.0 的 GUI 同一套规则），
    # 也允许直接传 archi。合法性交给 4.0 的 parse_archi_string 判，不在 4.1 里另写一份。
    spec = _archi_spec()
    archi_display = str(_s("archi_display") or "").strip()
    if archi_display:
        archi = spec.build_archi_string(archi_display,
                                        str(_s("subarchi") or ""),
                                        str(_s("kernel_label") or spec.default_kernel_label()))
        if spec.parse_archi_string(archi) is None:
            raise HTTPException(400, "架构组合不合法：" + archi)
    else:
        archi = str(_s("archi") or p["archi"])
        if spec.parse_archi_string(archi) is None:
            raise HTTPException(400, "架构写法不合法：" + archi)

    # 被夹过的参数都记下来：以前是**静默**改数字（填 1024 建成 640，界面上什么都不说），
    # 用户以为自己在训 1024 的模型，其实参数是 640 的。
    _clamped = {}

    def _clamp(key, v, lo, hi, step):
        iv = int(v)
        cv = max(lo, min(hi, iv))
        cv = int(round(cv / float(step)) * step)
        if cv != iv:
            _clamped[key] = {"requested": iv, "used": cv, "allowed": [lo, hi], "step": step}
        return cv

    options = {
        # 上限从 640 放到 1536：1024/1536 这类"挑战型"模型允许建，
        # 但显存装不下时界面会红字提示（估算已改为读面板里的分辨率）。
        "resolution": _clamp("resolution", _pick("resolution", int, p["resolution"]), 64, 1536, 32),
        "face_type": str(_s("face_type") or "wf"),
        "archi": archi,
        "ae_dims": _clamp("ae_dims", _pick("ae_dims", int, p["ae_dims"]), 64, 1024, 8),
        "e_dims": _clamp("e_dims", _pick("e_dims", int, p["e_dims"]), 16, 256, 8),
        "d_dims": _clamp("d_dims", _pick("d_dims", int, p["d_dims"]), 16, 256, 8),
        "d_mask_dims": _clamp("d_mask_dims", _pick("d_mask_dims", int, p["d_mask_dims"]), 8, 128, 8),
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
        # 备份/崩溃检测这三项 4.0 一直有，4.1 的 create 以前没写 →
        # 模型的 dat 里没有这几个键，引擎就只能用缺省值（max_backups=24、backup_interval=500），
        # 界面上填的「最大备份数」也存不进去（collectCfgPatch 对没有基线的行会跳过）。
        "max_backups": max(1, _pick("max_backups", int, 3)),
        "backup_interval": max(0, _pick("backup_interval", int, 1000)),
        "crash_threshold": float(body.get("crash_threshold") or 0.0),
        "pretrain": False,
        # 预训练时只跑 encoder/inter/decoder_src 一条支路，decoder_dst 不参与前向/反向，
        # 保存时权重自动同步过去。设计上预训练就是"训一个共享主干 + 一个解码器"，
        # 所以默认开；只在 pretrain=True 时生效，普通训练完全无视它。
        # 收益实测（1024/ae256）：峰值显存 24.43 -> 18.03 GiB，20 GB 卡不再溢出。
        "pretrain_single_decoder": bool(body.get("pretrain_single_decoder", True)),
        "use_fast_generator": True,
        "models_opt_on_gpu": True,
        # BF16 混合精度：4.0 新模型默认就是开的，4.1 的 create 以前漏了这一项 ——
        # 结果新模型一律 fp32 训练，640 这种分辨率光权重+梯度就要多占一倍显存。
        "use_bf16": bool(body.get("use_bf16", True)),
        # 梯度检查点：省激活显存（权重/梯度/优化器状态它省不了）
        "gradient_checkpointing": bool(body.get("gradient_checkpointing", False)),
    }
    # ★ 协议核心：body.options 里是训练参数（批次大小 / 预训练 / 优化器 / 学习率 …），
    #   按默认值的类型收敛后一次性合并。前端不需要再发第二个 config 请求。
    _opts_in = body.get("options")
    if _opts_in:
        _clean, _bad = _clean_options(_opts_in, options)
        if _bad:
            raise HTTPException(400, "这些参数类型不对: " + ", ".join(_bad))
        options.update(_clean)
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

    if _clamped:
        print("[models] create 时被夹过：" + ", ".join(
            "%s %s->%s" % (k, v["requested"], v["used"]) for k, v in _clamped.items()), flush=True)

    return {"ok": True, "name": name, "preset": preset_key, "clamped": _clamped,
            "dat": str(dat_path),
            # 返回**实际生效的全部 options**：界面按这个渲染就不会和磁盘对不上
            "options": options,
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


# ---------------------------------------------------------------- 纯 .dfm 模型
# workspace/model 里可能存在「只有 .dfm、没有训练目录」的模型（导出后删了权重，
# 或者从别人机器上拷来的 .dfm）。以前 /api/models 只扫 *_data.dat，这些文件在
# 列表里完全隐形 —— 合成器左栏「模型」因此选不到它们，明明有 .dfm 却合成不了。
# 这里补一遍 *.dfm 扫描：字段和 _model_entry 对齐，数值解析不出来就给 0 / 空，
# 绝不靠猜。
_DFM_ITER_RE = re.compile(r"^(?P<base>.+?)-(?P<iter>\d{1,9})$")


def _dfm_iter_of(stem: str):
    """「名字-416」→ (名字, 416)；结尾不是 -数字 就 (名字, 0)，不猜。"""
    m = _DFM_ITER_RE.match(stem)
    if m:
        return m.group("base"), int(m.group("iter"))
    return stem, 0


def _dfm_archi_of(stem: str) -> str:
    """从文件名里认架构（df / liae-udt / df-udt-l 之类）。

    只认能通过 core/archi_names.parse_archi_string 的组合 —— 那是架构合法性的
    唯一出处；认不出来就返回空字符串，界面上显示「—」，而不是编一个架构出来。
    """
    toks = stem.split("-")
    try:
        spec = _archi_spec()
    except Exception:
        return ""
    for i, t in enumerate(toks):
        if t.strip().lower() not in ("df", "liae", "amp"):
            continue
        for n in (1, 2, 3):
            cand = "-".join(toks[i:i + n]).strip().lower()
            if not cand:
                continue
            try:
                if spec.parse_archi_string(cand) is not None:
                    return cand
            except Exception:
                return ""
    return ""


def _dfm_only_entry(dfm: Path) -> dict:
    """只有 .dfm 的模型 → 一条和 _model_entry 同构的记录。"""
    name = dfm.stem   # 文件名里的数字不是迭代，别解析
    archi = _dfm_archi_of(dfm.stem)
    try:
        trained_at = int(dfm.stat().st_mtime)
    except Exception:
        trained_at = 0
    try:
        dfm_path = str(dfm.resolve())
    except Exception:
        dfm_path = str(dfm)
    return {
        "name": name,
        "archi": archi,
        "archi_parsed": _archi_parts(archi),
        "face_type_label": "",
        "resolution": 0,
        "face_type": "",
        "ae_dims": 0,
        "e_dims": 0,
        "d_dims": 0,
        "mask_dims": 0,
        "batch_size": 0,
        "bf16": False,
        "optimizer": "",
        "iter": 0,   # .dfm 里没有迭代信息，文件名里的数字只是命名的一部分，不要解析
        "spark": {"src": [], "dst": []},
        "has_dfm": True,
        # 合成器左栏按 has_export 决定显示「DFM」还是「导出」按钮：
        # 已经有 .dfm 的条目没有可导出的权重，必须给 True，否则会凭空多一个假按钮
        "has_export": True,
        "weights": "",
        "trained_at": trained_at,
        "has_preview": False,
        "preview_url": None,
        "dfm": dfm_path,
        "dfm_only": True,      # 没有训练目录：训练/参数/删除这些操作对它没意义
    }


@router.get("/api/models")
def models():
    out = []
    if MODEL_DIR.is_dir():
        for dat in sorted(MODEL_DIR.glob("*_data.dat")):
            try:
                out.append(_model_entry(dat))
            except Exception:
                continue
        # 按名字 / 按 dfm 绝对路径去重：训练目录已经算过的模型不要再追加一条
        known_names = set()
        known_paths = set()
        for m in out:
            nm = str(m.get("name") or "")
            if not nm:
                continue
            known_names.add(nm.lower())
            try:
                known_paths.add(os.path.normcase(str((MODEL_DIR / (nm + ".dfm")).resolve())))
            except Exception:
                pass
        for dfm in sorted(MODEL_DIR.glob("*.dfm")):
            try:
                key = os.path.normcase(str(dfm.resolve()))
                if key in known_paths:
                    continue
                entry = _dfm_only_entry(dfm)
                if entry["name"].lower() in known_names:
                    continue
                out.append(entry)
                known_names.add(entry["name"].lower())
                known_paths.add(key)
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
