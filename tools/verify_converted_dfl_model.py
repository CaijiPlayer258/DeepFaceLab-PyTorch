#!/usr/bin/env python3
"""Verify TF->Torch converted DeepFaceLab models can be loaded correctly.

This script does not train or run inference.
It builds model/optimizer skeletons on CPU and verifies:
- Source TF/Leras saveable dict covers expected tensors (by key+shape)
- Converted Torch Saveable files exist and have expected param_0..param_N keys
- Optimizer state coverage for continued training (best-effort)
- data.dat options/iter can be read

Typical usage:
  python3 tools/verify_converted_dfl_model.py \
    --src workspace/model \
    --dst workspace/model_converted \
    --all

"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np


def _load_converter_module():
    repo_root = Path(__file__).resolve().parents[1]
    converter_path = repo_root / "tools" / "convert_dfl_tf_to_torch.py"
    spec = importlib.util.spec_from_file_location("convert_dfl_tf_to_torch", converter_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import converter module from {converter_path}")
    mod = importlib.util.module_from_spec(spec)
    # Ensure repo root is importable (converter expects it).
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    # Python 3.14 dataclasses may query sys.modules[cls.__module__] during decoration.
    # Register the module before execution.
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


_conv = _load_converter_module()

# Reuse the converter's logic to avoid drift.
_assign_from_tf_saveable_dict = _conv._assign_from_tf_saveable_dict
_assign_optimizer_from_tf_dict = _conv._assign_optimizer_from_tf_dict
_build_specs_for_model = _conv._build_specs_for_model
_discover_models = _conv._discover_models
_ensure_leras_cpu = _conv._ensure_leras_cpu
_find_existing_weight_file = _conv._find_existing_weight_file
_read_pickle = _conv._read_pickle
_read_saveable_dict = _conv._read_saveable_dict


def _read_data_dat(path: Path) -> dict:
    data = _read_pickle(path)
    if not isinstance(data, dict):
        raise TypeError(f"{path} is not a dict")
    return data


def _check_param_keys(path: Path, expected_count: int) -> Tuple[bool, str]:
    """检查 .pth 里的 param_0..param_N 键。

    ⚠️ 本仓库的 Saveable .pth 是 **legacy torch zip**（zip 里是 param_0.npy、
    param_1.npy …），不是普通 pickle。直接 pickle.load 会抛
    "A load persistent id instruction was encountered" —— 那是**读法不对**，
    不代表文件坏了（同一份文件紧接着 load_weights() 是成功的）。
    所以先按 zip 读，读不到再退回 pickle。
    """
    keys = None
    try:
        import zipfile

        with zipfile.ZipFile(path) as z:
            names = [n for n in z.namelist() if n.endswith(".npy")]
        if names:
            keys = [n[:-4] for n in names]
    except Exception:
        keys = None

    if keys is None:
        try:
            d = _read_saveable_dict(path)
            keys = list(d.keys())
        except Exception as e:
            return False, f"cannot read: {e}"

    # keys should be param_0..param_{n-1}
    missing = []
    for i in range(expected_count):
        if f"param_{i}" not in keys:
            missing.append(f"param_{i}")
            if len(missing) >= 5:
                break

    if missing:
        return False, f"missing keys sample: {missing}"

    return True, f"ok ({len(keys)} keys)"


def _verify_one_model(*, src_dir: Path, dst_dir: Path, model_name: str, model_class: str,
                      deep: bool = False) -> int:
    errors = 0

    src_data = src_dir / f"{model_name}_data.dat"
    dst_data = dst_dir / f"{model_name}_data.dat"

    if not src_data.exists():
        print(f"[ERR] missing source data.dat: {src_data}")
        return 1
    if not dst_data.exists():
        print(f"[ERR] missing converted data.dat: {dst_data}")
        return 1

    src_model_data = _read_data_dat(src_data)
    dst_model_data = _read_data_dat(dst_data)

    src_iter = int(src_model_data.get("iter", 0) or 0)
    dst_iter = int(dst_model_data.get("iter", 0) or 0)
    print(f"\n=== {model_name} ({model_class}) ===")
    print(f"data.dat iter: src={src_iter} dst={dst_iter}")

    # Options should exist and be dict.
    src_opts = src_model_data.get("options", {})
    dst_opts = dst_model_data.get("options", {})
    if not isinstance(src_opts, dict) or not isinstance(dst_opts, dict):
        print("[ERR] options is not dict")
        return 1

    # Build skeletons from src options (same as conversion).
    saveables, optimizers = _build_specs_for_model(model_class, dict(src_opts))

    # Verify each saveable:
    for spec in saveables:
        src_file = src_dir / f"{model_name}_{spec.filename}"
        src_existing = _find_existing_weight_file(src_file)

        dst_file = dst_dir / f"{model_name}_{spec.filename}"
        dst_existing = _find_existing_weight_file(dst_file)

        # Check expected param count from skeleton.
        expected_param_count = len(list(spec.obj.get_weights()))

        if src_existing is None:
            print(f"[WARN] source weights missing: {spec.scope} ({spec.filename})")
        else:
            tf_dict = _read_saveable_dict(src_existing)
            try:
                loaded, total, missing, _fb, _amb = _assign_from_tf_saveable_dict(
                    tf_dict=tf_dict, saveable=spec.obj, scope=spec.scope)
            except Exception as e:
                print(f"[ERR] map/shape fail: {spec.scope}: {e}")
                errors += 1
                continue

            miss_n = len(missing)
            ok = (loaded == total) and (miss_n == 0)
            if ok:
                print(f"[OK]  source coverage: {spec.scope}: {loaded}/{total}")
            else:
                sample = missing[:5]
                print(f"[WARN] source coverage: {spec.scope}: loaded {loaded}/{total}, missing {miss_n} sample={sample}")
                # missing weights means conversion would keep init values => likely quality hit
                errors += 1

        if dst_existing is None:
            print(f"[ERR] converted weights missing: {spec.scope} ({spec.filename})")
            errors += 1
        else:
            # --deep：把【源 npy 读进来的值】先存一份，后面让它读【转换后的 pth】，
            # 再逐个张量比对。这是唯一能证明「转换没丢/没串权重」的检查 ——
            # 单看「状态: 已加载」是看不出来的：旧 bug 产生的错乱张量形状照样对，
            # 照样能加载、能训练，只是输出纯色块。
            _src_vals = None
            if deep and src_existing is not None:
                try:
                    _src_vals = [w.detach().clone() for w in spec.obj.get_weights()]
                except Exception:
                    _src_vals = None

            ok_keys, msg = _check_param_keys(dst_existing, expected_param_count)
            if ok_keys:
                print(f"[OK]  converted file keys: {spec.scope}: {msg}")
            else:
                print(f"[ERR] converted file keys: {spec.scope}: {msg}")
                errors += 1

            # Real load test into skeleton (ensures shapes/dtypes compatible with Saveable.load_weights).
            try:
                ok_load = bool(spec.obj.load_weights(dst_existing))
            except Exception as e:
                ok_load = False
                print(f"[ERR] converted load exception: {spec.scope}: {e}")

            if ok_load:
                print(f"[OK]  converted load_weights(): {spec.scope}")
            else:
                print(f"[ERR] converted load_weights() failed: {spec.scope}")
                errors += 1

            if _src_vals is not None:
                try:
                    _cur = [w.detach().cpu().float() for w in spec.obj.get_weights()]
                    _worst = 0.0
                    _worst_i = -1
                    for _i, (_w, _r) in enumerate(zip(_cur, _src_vals)):
                        _d = float((_w - _r.detach().cpu().float()).abs().max())
                        if _d > _worst:
                            _worst, _worst_i = _d, _i
                    if _worst == 0.0:
                        print(f"[OK]  深度比对（源npy vs 转换后pth）: {spec.scope}: "
                              f"{len(_cur)} 个张量逐元素完全一致 ✓")
                    else:
                        _sh = tuple(_cur[_worst_i].shape) if _worst_i >= 0 else '?'
                        print(f"[ERR] 深度比对失败: {spec.scope}: max|差| = {_worst:.3e}"
                              f"（第 {_worst_i} 个张量 {_sh}）—— 转换没保住权重")
                        errors += 1
                except Exception as e:
                    print(f"[WARN] 深度比对异常: {spec.scope}: {e}")

    # Verify each optimizer:
    for opt_spec in optimizers:
        src_file = src_dir / f"{model_name}_{opt_spec.filename}"
        src_existing = _find_existing_weight_file(src_file)

        dst_file = dst_dir / f"{model_name}_{opt_spec.filename}"
        dst_existing = _find_existing_weight_file(dst_file)

        expected_state_count = len(list(opt_spec.opt.get_weights()))

        if src_existing is None:
            print(f"[WARN] source optimizer missing: {opt_spec.scope} ({opt_spec.filename})")
            # This is not always fatal (can continue training, but optimizer will reset).
        else:
            tf_dict = _read_saveable_dict(src_existing)
            try:
                loaded, total, missing = _assign_optimizer_from_tf_dict(tf_dict=tf_dict, opt_spec=opt_spec)
            except Exception as e:
                print(f"[ERR] optimizer map/shape fail: {opt_spec.scope}: {e}")
                errors += 1
                continue

            miss_n = len(missing)
            # Optimizers are best-effort; allow missing but warn.
            if loaded == total and miss_n == 0:
                print(f"[OK]  optimizer coverage: {opt_spec.scope}: {loaded}/{total}")
            else:
                sample = missing[:5]
                print(f"[WARN] optimizer coverage: {opt_spec.scope}: loaded {loaded}/{total}, missing {miss_n} sample={sample}")

        if dst_existing is None:
            print(f"[ERR] converted optimizer missing: {opt_spec.scope} ({opt_spec.filename})")
            errors += 1
        else:
            ok_keys, msg = _check_param_keys(dst_existing, expected_state_count)
            if ok_keys:
                print(f"[OK]  converted optimizer keys: {opt_spec.scope}: {msg}")
            else:
                print(f"[ERR] converted optimizer keys: {opt_spec.scope}: {msg}")
                errors += 1

            # Real load test for optimizer state.
            try:
                ok_load = bool(opt_spec.opt.load_weights(dst_existing))
            except Exception as e:
                ok_load = False
                print(f"[ERR] converted optimizer load exception: {opt_spec.scope}: {e}")
            if ok_load:
                print(f"[OK]  converted optimizer load_weights(): {opt_spec.scope}")
            else:
                print(f"[WARN] converted optimizer load_weights() failed: {opt_spec.scope} (optimizer may reset)")

    return errors


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Verify TF->Torch converted DFL models can be loaded")
    p.add_argument("--src", required=True, type=Path, help="源模型目录（原版/TF 导出的 saved_models）")
    p.add_argument("--dst", required=True, type=Path, help="转换后模型目录（Torch saved_models）")

    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--all", action="store_true", help="验证 src 下所有 *_data.dat")
    g.add_argument("--name", type=str, help="只验证一个模型名（不含 _data.dat），例如 DF-UD384_SAEHD")

    p.add_argument("--model", type=str, default=None, help="可选：强制模型类型 SAEHD/AMP/Quick96/XSeg")
    p.add_argument("--deep", action="store_true",
                   help="深度比对：把源 npy 读进来的权重和转换后的 pth 逐元素比一遍。"
                        "这是唯一能证明「转换没串权重」的检查 —— 光看加载成功是看不出来的。")

    args = p.parse_args(argv)

    src_dir: Path = args.src
    dst_dir: Path = args.dst

    if not src_dir.exists():
        raise FileNotFoundError(src_dir)
    if not dst_dir.exists():
        raise FileNotFoundError(dst_dir)

    _ensure_leras_cpu()

    errors = 0

    if args.all:
        models = _discover_models(src_dir)
        for model_name, model_class in models:
            if model_class is None:
                continue
            errors += _verify_one_model(src_dir=src_dir, dst_dir=dst_dir, model_name=model_name,
                                         model_class=model_class, deep=bool(args.deep))
    else:
        model_name = str(args.name)
        if args.model is not None:
            model_class = str(args.model)
        else:
            # infer class from suffix
            if "_" in model_name:
                model_class = model_name.rsplit("_", 1)[1]
            else:
                model_class = model_name
        errors += _verify_one_model(src_dir=src_dir, dst_dir=dst_dir, model_name=model_name,
                                     model_class=model_class, deep=bool(args.deep))

    if errors == 0:
        print("\nRESULT: OK (no blocking issues detected)")
        return 0

    print(f"\nRESULT: WARN/ERR (issues={errors}). If coverage warnings appear, some weights/states were not found in source dict and would stay initialized.")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
