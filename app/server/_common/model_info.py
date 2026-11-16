# -*- coding: utf-8 -*-
"""模型档案：读 *_data.dat / 训练预览 / 架构规格表 / GPU 探测（从 app/server/app.py 切出）。

只看不改：这一层把磁盘上的模型「说清楚」—— iter、loss 曲线、预览图、架构与规格表、
显存推荐档位、可改/不可改的参数白名单。真正起训练进程的是 _common/runctl.py。

函数体 / 注释逐字节照抄 app.py；只有一处包内相对 import 的深度（_legacy）。
"""
from __future__ import annotations

import json
from pathlib import Path

from .bootstrap import _NO_WINDOW
from .paths import ROOT, WORKSPACE, _DFL_ROOT


# ============================================================
# 模型训练参数：读写 *_data.dat 里的 options
# 结构性字段（分辨率/维度/架构）创建后不可改，白名单挡掉
# ============================================================
# 只有这 7 个是「创建后不可改」的结构参数：必须和 routes_models._STRUCTURAL_KEYS、
# 前端 collectCfgPatch 的 STRUCT 三处一致，否则就是「界面能改、发上去存不进去」。
# gan_power / true_face_power / face_style_power / models_opt_on_gpu 以前被误列在这里：
# 它们是每次训练都能改的损失权重 / 放置开关（前三个还在 _HOT_OPTIONS 里），
# 结果界面上能改、发上来却被 400 整单拒绝，同一次保存里的其它参数也跟着全失败。
_CFG_STRUCTURAL = {"resolution", "archi", "e_dims", "ae_dims", "d_dims",
                   "d_mask_dims", "face_type"}


def _model_dat(name: str):
    md = WORKSPACE / "model"
    cands = sorted(md.glob(name + "_*_data.dat"))
    return cands[0] if cands else None




# 与 4.0 Trainer.py 的 _HOT_KEYS 一一对应：这些参数训练中改会立即生效，
# 其余（分辨率/架构/维度/batch/优化器/预训练…）必须重启训练。
_HOT_OPTIONS = {
    "crash_threshold", "backup_interval", "max_backups", "accum_steps",
    "face_style_power", "bg_style_power",
    "masked_training", "blur_out_mask", "eyes_mouth_prio", "uniform_yaw",
    "lr", "lr_cos", "lr_dropout",
    "random_src_flip", "random_dst_flip", "random_hsv_power",
    "random_warp", "random_occlusion", "random_noise",
    "random_color_power", "random_color_alg",
    "ct_mode",
    # 刚性形变范围
    "rotation_range", "scale_range", "t_range",
    # 眼睛 / 嘴巴缩放、边缘增强
    "eyes_scale", "mouth_scale", "edge_enhance_power",
    # Inter code 分布日志间隔
    "log_code_stats",
}
# ---------------- 模型库 ----------------
MODEL_DIR = WORKSPACE / "model"
CACHE_DIR = MODEL_DIR / ".cache"


def _read_dat(dat_path: Path) -> dict:
    """读 *_data.dat：iter / options / loss_history / sample_for_preview。

    TF 时代（DFL 1.x / 2.0）的 dat 在这里就被翻成 4.1 的 options 形状：裸 pickle 读出来
    的 learn_mask / ed_ch_dims / face_type 全名，引擎不认、参数页也不认。键名与取值
    规则只有 app/engine/core/legacy_dat.py 一份出处，见 server/_legacy.py。
    """
    return _read_dat_info(dat_path)[0]


def _read_dat_info(dat_path: Path):
    """(完整 dat, 迁移信息)。

    迁移信息是给界面用的那一条：detected / counts / report / 权重现状 / 可复制的
    转权重命令。新格式 dat 返回 detected=False，一个字都不改。
    """
    from .._legacy import notice, read_dat
    data, _info = read_dat(dat_path)
    try:
        return data, notice(dat_path)      # 命中缓存，只多一次 glob
    except Exception:
        return data, _info


def _spark(loss_history, n=2500):
    """取最后 n 次迭代，拆成 src / dst 两条曲线。"""
    tail = loss_history[-n:] if len(loss_history) > n else loss_history
    src, dst = [], []
    for e in tail:
        if isinstance(e, (list, tuple)) and len(e) >= 2:
            try:
                src.append(round(float(e[0]), 5))
                dst.append(round(float(e[1]), 5))
            except Exception:
                pass
    return {"src": src, "dst": dst}


def _preview_png(dat: dict, out: Path) -> bool:
    """sample_for_preview 里是一条 (n,3,H,W) 的 float32 序列，通道是 BGR。
       拼成一条横图，前端再按 index 切片。"""
    try:
        import numpy as np
        from PIL import Image

        sp = dat.get("sample_for_preview")
        if not sp:
            return False
        arr = np.asarray(sp[0][0], dtype=np.float32)
        if arr.ndim != 4 or arr.shape[1] != 3:
            return False
        n, _, h, w = arr.shape
        strip = Image.new("RGB", (w * n, h), (0, 0, 0))
        for i in range(n):
            img = np.clip(arr[i], 0.0, 1.0)
            rgb = img[::-1].transpose(1, 2, 0)          # BGR -> RGB
            strip.paste(Image.fromarray((rgb * 255.0).astype(np.uint8), "RGB"), (i * w, 0))
        out.parent.mkdir(parents=True, exist_ok=True)
        strip.save(out, "PNG", optimize=True)
        return True
    except Exception:
        return False


def _read_params_cache(dat_path: Path):
    """读 4.0 写的 <model>_data.dat.params（1.7KB）。

    指纹（size + mtime_ns）必须和 data.dat 对得上 —— 4.0 那边改了文件就会作废，
    这里也照同样规则判，避免拿过期参数去做模型卡片。
    """
    import pickle
    try:
        st = dat_path.stat()
        data = pickle.loads(Path(str(dat_path) + ".params").read_bytes())
        if data.get("version") != 1:
            return None
        if data.get("src_size") != st.st_size or data.get("src_mtime_ns") != st.st_mtime_ns:
            return None
        return data
    except Exception:
        return None


def _read_preview_json(dat_path: Path):
    """读 4.0 训练时写的 <model>_preview.json（loss 尾巴 + iter）。"""
    try:
        return json.loads(Path(str(dat_path)[: -len("_data.dat")] + "_preview.json").read_text(encoding="utf-8"))
    except Exception:
        return None


def _preview_jpg(dat_path: Path) -> Path:
    return Path(str(dat_path)[: -len("_data.dat")] + "_preview.jpg")


# 新格式 dat / 快路上的模型：legacy 字段的占位，保证 /api/models 里每条都同构
_NO_LEGACY = {
    "detected": False, "source": "params-cache", "counts": {}, "report": [],
    "reasons": [], "weights": {"torch": 0, "tf": 0, "pth": [], "npy": []},
    "needs_convert": False, "hint": "", "note": "", "command": "",
}


def _fast_model_entry(dat_path: Path) -> dict:
    """两个小文件就够：参数缓存 + 预览快照。都不在就返回 None 走老路。

    老路要 pickle 整个 data.dat（122MB 的模型要 0.85s，每个模型每次保存后都要重来
    一遍）；新路读 1.7KB + 40KB，毫秒级。
    """
    params = _read_params_cache(dat_path)
    if params is None:
        return None
    name = dat_path.name[: -len("_data.dat")]
    model_name, _, archi_from_name = name.rpartition("_")
    if not model_name:
        model_name, archi_from_name = name, ""

    opt = params.get("options") or {}
    pj = _read_preview_json(dat_path)
    spark = {"src": [], "dst": []}
    if pj:
        tail = pj.get("loss_tail") or []
        spark = {"src": [round(float(e[0]), 5) for e in tail if len(e) >= 2],
                 "dst": [round(float(e[1]), 5) for e in tail if len(e) >= 2]}

    has_prev = _preview_jpg(dat_path).is_file()
    dfm = dat_path.parent / (model_name + ".dfm")
    _archi = str(opt.get("archi", archi_from_name)).lower()
    return {
        "name": model_name,
        "archi": _archi,
        "archi_parsed": _archi_parts(_archi),
        "face_type_label": _FACE_LABELS.get(str(opt.get("face_type", "")), str(opt.get("face_type", ""))),
        "resolution": int(opt.get("resolution", 0) or 0),
        "face_type": str(opt.get("face_type", "")),
        "ae_dims": int(opt.get("ae_dims", 0) or 0),
        "e_dims": int(opt.get("e_dims", 0) or 0),
        "d_dims": int(opt.get("d_dims", 0) or 0),
        "mask_dims": int(opt.get("d_mask_dims", 0) or 0),
        "batch_size": int(opt.get("batch_size", 0) or 0),
        "bf16": str(opt.get("use_bf16", "False")) == "True",
        "optimizer": str(opt.get("optimizer", "")),
        "iter": int(params.get("iter", 0) or 0),
        "spark": spark,
        "has_dfm": dfm.exists(),
        "weights": "npy" if any(dat_path.parent.glob(model_name + "_*decoder*.npy")) else "pth",
        "trained_at": int(dat_path.stat().st_mtime),
        "has_preview": has_prev,
        "preview_url": ("/api/models/" + model_name + "/preview.png") if has_prev else None,
        # 走 .params 快路的模型一定是 4.1 自己训练出来的：没有 legacy 可言。
        # 字段保留是为了让界面只判断 legacy.detected，不用管接口来自哪条路。
        "legacy": dict(_NO_LEGACY),
    }


def _model_entry(dat_path: Path) -> dict:
    fast = _fast_model_entry(dat_path)
    if fast is not None:
        return fast

    name = dat_path.name[: -len("_data.dat")]
    model_name, _, archi_from_name = name.rpartition("_")
    if not model_name:
        model_name, archi_from_name = name, ""

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_json = CACHE_DIR / (model_name + ".json")
    preview = CACHE_DIR / (model_name + ".png")
    mtime = dat_path.stat().st_mtime

    cached = None
    try:
        cached = json.loads(cache_json.read_text(encoding="utf-8"))
    except Exception:
        cached = None
    if cached and abs(float(cached.get("mtime", 0)) - mtime) < 1 and preview.exists():
        entry = dict(cached)
        entry.pop("mtime", None)
        entry["preview_url"] = "/api/models/" + model_name + "/preview.png"
        return entry

    try:
        dat, legacy = _read_dat_info(dat_path)
    except Exception:
        dat, legacy = {}, dict(_NO_LEGACY)

    opt = dat.get("options") or {}
    loss = dat.get("loss_history") or []
    spark = _spark(loss)

    ok_prev = _preview_png(dat, preview)

    weights = "npy" if any(dat_path.parent.glob(model_name + "_*decoder*.npy")) else "pth"
    dfm = dat_path.parent / (model_name + ".dfm")

    _archi2 = str(opt.get("archi", archi_from_name)).lower()
    entry = {
        "name": model_name,
        "archi": _archi2,
        "archi_parsed": _archi_parts(_archi2),
        "face_type_label": _FACE_LABELS.get(str(opt.get("face_type", "")), str(opt.get("face_type", ""))),
        "resolution": int(opt.get("resolution", 0) or 0),
        "face_type": str(opt.get("face_type", "")),
        "ae_dims": int(opt.get("ae_dims", 0) or 0),
        "e_dims": int(opt.get("e_dims", 0) or 0),
        "d_dims": int(opt.get("d_dims", 0) or 0),
        "mask_dims": int(opt.get("d_mask_dims", 0) or 0),
        "batch_size": int(opt.get("batch_size", 0) or 0),
        "bf16": str(opt.get("use_bf16", "False")) == "True",
        "optimizer": str(opt.get("optimizer", "")),
        # dat 里的 iter 才是真迭代数；loss_history 只是采样，早年被当成 iter 用，
        # 结果是 41k 次训练的模型在卡片上写着「3 次迭代」。
        "iter": int(dat.get("iter") or 0) or int(len(loss)),
        "spark": spark,
        "has_dfm": dfm.exists(),
        "weights": weights,
        "trained_at": int(dat_path.stat().st_mtime),
        "has_preview": bool(ok_prev),
        # TF 时代的 dat：有没有迁移、迁了几项、要不要转权重，界面据此在参数页顶部
        # 出一条非阻塞提示（新格式这里就是 detected=False，什么都不显示）
        "legacy": legacy,
    }

    try:
        payload = dict(entry)
        payload["mtime"] = mtime
        cache_json.write_text(json.dumps(payload), encoding="utf-8")
    except Exception:
        pass

    entry["preview_url"] = "/api/models/" + model_name + "/preview.png" if ok_prev else None
    return entry
def _training_pid():
    lock = MODEL_DIR / ".train.lock"
    try:
        return int(lock.read_text(encoding="utf-8").strip())
    except Exception:
        return 0
# ============================================================
# GPU 探测 + 新建模型（"别让用户为任何选项焦虑"）
# ============================================================
# 4.1 自己的 python 没有 torch，所以显卡信息走 nvidia-smi（一次约 80ms）。
def _gpu_list():
    import shutil as _shutil
    import subprocess as _sp
    exe = _shutil.which("nvidia-smi") or r"C:\Windows\System32\nvidia-smi.exe"
    if not Path(exe).is_file():
        return []
    try:
        out = _sp.run([exe, "--query-gpu=index,name,memory.total,memory.free",
                       "--format=csv,noheader,nounits"],
                      capture_output=True, text=True, timeout=8,
                      creationflags=_NO_WINDOW).stdout
    except Exception:
        return []
    gpus = []
    for line in out.strip().splitlines():
        parts = [x.strip() for x in line.split(",")]
        if len(parts) < 4:
            continue
        try:
            gpus.append({"index": int(parts[0]), "name": parts[1],
                         "total_gb": round(int(parts[2]) / 1024.0, 1),
                         "free_gb": round(int(parts[3]) / 1024.0, 1)})
        except Exception:
            continue
    return gpus


def _engine_has_flag(flag: str) -> bool:
    """引擎的 main.py 认不认某个命令行开关。

    不是每个引擎版本都有 --cpu-only / --force-gpu-idxs（4.0 的 train 解析器是后来
    才加的）。不认的话 argparse 直接 "unrecognized arguments" 退出 —— 那就得在
    启动时明确告警，而不是让用户看着进程闪退还不知道为什么。
    """
    try:
        src = (_DFL_ROOT / "main.py").read_text(encoding="utf-8", errors="replace")
    except Exception:
        return False
    return flag in src


def _engine_supports_cpu_only() -> bool:
    """引擎的 train 子命令认不认 --cpu-only。"""
    return _engine_has_flag("--cpu-only")


# 档位表：数值不是拍的，是拿本机真实训练过的模型反推的
#   256 / df-ud   权重约 0.49GB（DFLitetest、model_004 都是这套配置跑起来的）
#   384 / liae-ud 权重约 2.04GB（384wf，300 万次迭代）
# 训练显存 ≈ 权重 × 4（参数 1 + 优化器状态 ms/vs 2 + 梯度 1）+ 激活
_PRESETS = {
    "fast":     {"label": "快",   "resolution": 256, "archi": "df-ud",
                 "ae_dims": 256, "e_dims": 64, "d_dims": 128, "d_mask_dims": 42,
                 "weights_gb": 0.49,
                 "note": "最快出结果，细节一般，适合先跑通"},
    "balanced": {"label": "均衡", "resolution": 384, "archi": "liae-ud",
                 "ae_dims": 352, "e_dims": 102, "d_dims": 102, "d_mask_dims": 32,
                 "weights_gb": 2.04,
                 "note": "画质和速度都能接受，长期训练推荐"},
    "quality":  {"label": "极致", "resolution": 512, "archi": "liae-ud",
                 "ae_dims": 352, "e_dims": 102, "d_dims": 102, "d_mask_dims": 32,
                 "weights_gb": 2.04,
                 "note": "最高分辨率，需要大显存，速度慢"},
}


def _recommend_for(free_gb: float):
    """按可用显存给一档配置 + batch。

    实测：8GB 的卡跑 256/df-ud/batch4 没问题；384/liae-ud 权重 2.04GB×4 ≈ 8.2GB，
    8GB 上放不下 —— 所以 384 只在 14GB 以上才推荐。
    """
    table = (("quality", 24.0, 8), ("quality", 20.0, 4),
             ("balanced", 14.0, 4), ("fast", 10.0, 8), ("fast", 6.0, 4))
    for key, min_gb, batch in table:
        if free_gb >= min_gb:
            p = dict(_PRESETS[key])
            p.update({"preset": key, "batch_size": batch,
                      "need_gb": round(p["weights_gb"] * 4 + 1.5, 1)})
            return p
    p = dict(_PRESETS["fast"])
    p.update({"preset": "fast", "batch_size": 2, "need_gb": round(0.49 * 4 + 1.5, 1)})
    return p


def _archi_spec():
    """借用 4.0 的 core/archi_names.py —— 架构/子分支/算子的合法性只有一份出处，
    4.1 不复制那几张表。core/__init__.py 是空的，import 很轻。"""
    import importlib
    import sys as _sys
    root = str(_DFL_ROOT)
    added = root not in _sys.path
    if added:
        _sys.path.insert(0, root)
    try:
        return importlib.import_module("core.archi_names")
    finally:
        if added:
            try:
                _sys.path.remove(root)
            except ValueError:
                pass


# 脸型：值 ↔ 中文。取值来自 4.0 Model_SAEHD 的 face_type 映射表
_FACE_TYPES = [
    {"v": "h", "label": "半脸（只要脸，不要额头下巴）"},
    {"v": "mf", "label": "中全脸"},
    {"v": "f", "label": "全脸（默认）"},
    {"v": "wf", "label": "整头（含额头和下巴，推荐）"},
    {"v": "head", "label": "头肩（要脖子和肩膀）"},
]


_FACE_LABELS = {f["v"]: f["label"] for f in _FACE_TYPES}


def _archi_parts(archi):
    """把 'df-udt-l' 拆成给界面看的字段（显示名 / 子分支 / 算子）。"""
    try:
        spec = _archi_spec()
        parsed = spec.parse_archi_string(archi)
        if parsed is None:
            return {"display": str(archi), "sub": "", "kernel": "", "lite": False}
        base, opts, mod = parsed
        lite = spec.archi_is_lite(mod)
        kernel = ""
        for lab, m, _d in spec.KERNEL_CHOICES:
            if m == (mod or ""):
                kernel = lab
                break
        return {"display": ("DF" if base == "df" else "LIAE") + (" Lite" if lite else ""),
                "sub": ("-" + opts) if opts else "",
                "kernel": kernel, "lite": bool(lite)}
    except Exception:
        return {"display": str(archi), "sub": "", "kernel": "", "lite": False}


# 规格表是静态的（来自 4.0 源码），探到一次就留着；显卡信息缓存 5 秒。
# 不缓存的话 /api/model_spec 第一次要 600ms —— 面板会"迟一拍"才出现。
_SPEC_CACHE = {"tables": None, "gpu": [], "gpu_at": 0.0}


def _spec_tables():
    if _SPEC_CACHE["tables"] is None:
        spec = _archi_spec()
        _SPEC_CACHE["tables"] = {
            "archi": [{"name": n, "base": b, "lite": bool(l),
                       "desc": spec.ARCHI_DESCRIPTIONS.get(n, "")}
                      for n, b, l in spec.ARCHI_CHOICES],
            "subarchi": [{"v": s, "desc": spec.SUBARCHI_DESCRIPTIONS.get(s, "")}
                         for s in spec.SUBARCHI_CHOICES],
            "kernel": [{"label": lab, "mod": mod, "desc": d} for lab, mod, d in spec.KERNEL_CHOICES],
            "default_kernel": spec.default_kernel_label(),
            "face_types": _FACE_TYPES,
            "dims": {"ae_dims": 256, "e_dims": 64, "d_dims": 128, "d_mask_dims": 42},
            "dims_liae": {"ae_dims": 352, "e_dims": 102, "d_dims": 102, "d_mask_dims": 32},
            "limits": {"resolution": [64, 1536], "ae_dims": [64, 1024],
                       "e_dims": [16, 256], "d_dims": [16, 256], "d_mask_dims": [8, 128],
                       "batch_size": [1, 64]},
        }
    return _SPEC_CACHE["tables"]


def _gpu_cached(ttl=5.0):
    now = time.time()
    if not _SPEC_CACHE["gpu"] or (now - _SPEC_CACHE["gpu_at"]) > ttl:
        _SPEC_CACHE["gpu"] = _gpu_list()
        _SPEC_CACHE["gpu_at"] = now
    return _SPEC_CACHE["gpu"]






_BAD_NAME_CHARS = set('/\\:*?"<>|') | {chr(i) for i in range(32)}


def _safe_model_name(name: str) -> str:
    name = (name or "").strip()
    if not name or name in (".", ".."):
        raise HTTPException(400, "模型名不能为空")
    if len(name) > 64:
        raise HTTPException(400, "模型名太长（最多 64 个字）")
    if any(ch in _BAD_NAME_CHARS for ch in name) or ".." in name:
        raise HTTPException(400, '模型名不能包含 / \\ : * ? " < > | 或 ..')
    return name
def _find_model_preview(name: str):
    """按名字在模型目录里"查表"找预览图 —— 用户传的 name 只用来做等值比较，
    绝不允许参与拼路径。

    返回 (Path, media_type)；找不到返回 (None, None)。
    两道保险：名字必须和某个 *_data.dat 推导出的模型名完全相等；选中的文件
    resolve() 之后必须仍在模型目录（或缓存目录）之内，且不能是符号链接。
    """
    try:
        root = MODEL_DIR.resolve()
    except Exception:
        return None, None
    cache_root = CACHE_DIR.resolve() if CACHE_DIR.exists() else None
    try:
        dats = sorted(MODEL_DIR.glob("*_data.dat"))
    except Exception:
        return None, None
    for dat in dats:
        stem = dat.name[: -len("_data.dat")]
        model_name = stem.rpartition("_")[0] or stem
        if model_name != name:
            continue
        for cand, mime in ((MODEL_DIR / (stem + "_preview.jpg"), "image/jpeg"),
                           (CACHE_DIR / (model_name + ".png"), "image/png")):
            try:
                if not cand.is_file() or cand.is_symlink():
                    continue
                rp = cand.resolve()
            except Exception:
                continue
            if rp.parent != root and (cache_root is None or rp.parent != cache_root):
                continue
            return rp, mime
    return None, None
