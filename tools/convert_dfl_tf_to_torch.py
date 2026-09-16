#!/usr/bin/env python3
"""Convert original DeepFaceLab (TF/Leras) model files to deepfacelab_Torch PyTorch Saveable format.

This repo uses a custom Saveable format:
- Torch side: pickle dict with keys param_0, param_1, ... (see core/leras/layers/Saveable.py)
- Original DFL TF side: pickle dict with keys like "conv1/weight:0" (see DeepFaceLab-master/core/leras/layers/Saveable.py)

Goal:
- Convert network weights AND optimizer states for continued training
- Batch convert all models in a saved_models directory
- Include XSeg conversion

Notes:
- This script does NOT run training or inference.
- It builds model/optimizer skeletons on CPU to obtain deterministic parameter order.

Usage examples:
  python tools/convert_dfl_tf_to_torch.py --src /path/to/DFL_saved_models --dst /path/to/torch_saved_models --all

  python tools/convert_dfl_tf_to_torch.py --src ... --dst ... --model SAEHD --name MyModel

"""

from __future__ import annotations

import argparse
import pickle
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _read_pickle(path: Path):
    return pickle.loads(path.read_bytes())


def _read_saveable_dict(path: Path) -> Dict[str, np.ndarray]:
    data = _read_pickle(path)
    if not isinstance(data, dict):
        raise TypeError(f"{path} is not a dict saveable")
    return data


def _find_existing_weight_file(base_path: Path) -> Optional[Path]:
    """Return an existing file, trying extension swaps (.pth <-> .npy) and the old/ archive.

    最后那条 old/ 查找是必要的：转换结束后源 .npy 会被归档到 <dir>/old/，
    否则 _convert_one_model 再跑一次、或者 verify_converted_dfl_model.py 事后校验，
    都会找不到源文件。
    """
    def _swaps(p: Path) -> List[Path]:
        out = [p]
        if p.suffix == ".pth":
            out.append(p.with_suffix(".npy"))
        elif p.suffix == ".npy":
            out.append(p.with_suffix(".pth"))
        return out

    for cand in _swaps(base_path):
        if cand.exists():
            return cand

    for cand in _swaps(base_path.parent / "old" / base_path.name):
        if cand.exists():
            return cand

    return None


def _np_to_np(arr) -> np.ndarray:
    if isinstance(arr, np.ndarray):
        return arr
    return np.asarray(arr)


def _tf_key_variants(key: str) -> List[str]:
    """Return reasonable TF key variants to try."""
    out = [key]
    if key.endswith(":0"):
        out.append(key[:-2])
    else:
        out.append(key + ":0")

    # Some TF checkpoints (rare) may store without trailing :0 in nested names.
    return list(dict.fromkeys(out))


def _convert_array_to_shape(src: np.ndarray, dst_shape: Sequence[int]) -> np.ndarray:
    src = _np_to_np(src)
    dst_shape = tuple(int(x) for x in dst_shape)

    if tuple(src.shape) == dst_shape:
        return src

    # Common Conv2D / Conv2DTranspose conversion.
    # Original DFL (TF leras) weight formats (confirmed from source):
    #   Conv2D:         weight = (kernel, kernel, in_ch, out_ch)
    #   Conv2DTranspose: weight = (kernel, kernel, out_ch, in_ch) — opposite channel order!
    #
    # PyTorch formats:
    #   Conv2D:         weight = (out_ch, in_ch, kernel, kernel)
    #   Conv2DTranspose: weight = (in_ch, out_ch, kernel, kernel)
    #
    # Both TF formats convert to their Torch equivalent via the SAME transpose:
    #   For Conv2D:         (H, W, in, out) —perm(3,2,0,1)—> (out, in, H, W)  ✓
    #   For Conv2DTranspose: (H, W, out, in) —perm(3,2,0,1)—> (in, out, H, W) ✓
    if src.ndim == 4 and len(dst_shape) == 4:
        if (src.shape[0], src.shape[1], src.shape[2], src.shape[3]) == (dst_shape[2], dst_shape[3], dst_shape[1], dst_shape[0]):
            return np.transpose(src, (3, 2, 0, 1))

    # Dense transpose is not expected in this repo (Dense stores (in, out)),
    # but keep a safe fallback.
    if src.ndim == 2 and len(dst_shape) == 2:
        if src.shape[::-1] == dst_shape:
            return src.T

    # ⚠️ 元素数相同就 reshape —— 这个兜底**极其危险**，已收窄。
    # 它会让形状根本对不上的权重静默通过：np.reshape 只是把内存顺序重新解释一遍，
    # 得到的是一个完全错乱的张量，而且上层完全不知道出过事。
    # 旧版转换器靠"名字匹配失败 → 扫到第一个元素数相同的键 → reshape"这套组合，
    # 实际上是在按字典顺序瞎猜参数（LIAE ae=384 转出纯色块就是这个原因）。
    # 现在只接受**不改变元素顺序**的等价形式：去掉/补上长度为 1 的维度，
    # 例如 (N,) <-> (1,N) <-> (N,1)。这种 reshape 是安全的。
    src_flat = [int(x) for x in src.shape if int(x) != 1]
    dst_flat = [int(x) for x in dst_shape if int(x) != 1]
    if src_flat == dst_flat:
        return np.reshape(src, dst_shape)

    raise ValueError(f"shape mismatch: {src.shape} -> {dst_shape}")


@dataclass
class SaveableSpec:
    scope: str
    obj: object
    filename: str


@dataclass
class OptimizerSpec:
    scope: str
    opt: object
    filename: str
    kind: str  # 'rmsprop' | 'adabelief'
    param_full_tf_names: List[str]


def _collect_tensor_name_maps(module) -> Tuple[Dict[int, str], Dict[int, str]]:
    """Return (id(tensor)->tf_subkey, id(buffer)->tf_subkey).

    优先级（顺序很重要）：

      1. **ModelBase._get_tf_weight_names()** —— DFL 自带的递归走树，给出**正确的
         层级 TF 名**，例如 down1/downs_0/conv1/weight:0。这也是
         core/leras/layers/Saveable._load_tf_weights() 和 facelib/XSegNet 用的
         同一套，是本仓库里权威的 TF 名来源。
      2. torch.nn.Module.named_parameters() —— 层级名（a/b/weight:0），兜底。
      3. leras get_layers() —— **只有扁平名**（conv1/weight:0），最后的兜底。

    ⚠️ 历史坑（v4.0.8.4 修复）：旧版转换器**跳过了第 1 条**，只用了 2 和 3。
    而 DFL 的 leras 把子层放在普通 list/dict 里，Torch 的 named_parameters()
    看不见它们，于是实际全部落到第 3 条扁平名上 —— 实测 encoder 的 8 个张量
    只映射出 2 个不同的名字（conv1/weight:0、conv1/bias:0），重名 6 个。
    名字匹配失败之后代码会落到「按形状贪心匹配」，再叠上元素数 reshape 的兜底，
    就变成了按字典顺序瞎猜参数 —— LIAE ae_dims=384 转换出纯色块、以及训练越训
    越炸，根因就在这里。
    """
    param_map: Dict[int, str] = {}
    buf_map: Dict[int, str] = {}

    # 1) 权威来源：递归走 DFL 层树，拿正确的层级 TF 名。
    #    注意：本模块顶层不 import torch（转换器允许在没装 torch 的环境里报错退出），
    #    所以这里局部导入。
    try:
        import torch as _torch

        get_tf = getattr(module, "_get_tf_weight_names", None)
        if callable(get_tf):
            for tf_name, t in get_tf():
                key = str(tf_name)
                if isinstance(t, _torch.nn.Parameter):
                    param_map.setdefault(id(t), key)
                else:
                    buf_map.setdefault(id(t), key)
    except Exception:
        pass

    # 2) Torch-registered modules —— 层级名，仅补第 1 条没覆盖到的。
    try:
        for name, p in module.named_parameters(recurse=True):
            # torch name: a.b.weight -> TF subkey: a/b/weight:0
            param_map.setdefault(id(p), name.replace(".", "/") + ":0")
    except Exception:
        pass

    try:
        for name, b in module.named_buffers(recurse=True):
            buf_map.setdefault(id(b), name.replace(".", "/") + ":0")
    except Exception:
        pass

    # 3) leras graph fallback —— 只有扁平名，仅当上面两条都没覆盖到时才用。
    try:
        get_layers = getattr(module, "get_layers", None)
        if callable(get_layers):
            layers = list(get_layers())

            for layer in layers:
                lname = getattr(layer, "name", None)
                if not lname:
                    continue

                for attr in ("weight", "bias", "running_mean", "running_var"):
                    try:
                        t = getattr(layer, attr)
                    except Exception:
                        t = None
                    if t is None:
                        continue

                    try:
                        is_param = hasattr(t, "requires_grad")
                    except Exception:
                        is_param = False

                    key = f"{lname}/{attr}:0"
                    if is_param:
                        param_map.setdefault(id(t), key)
                    else:
                        buf_map.setdefault(id(t), key)
    except Exception:
        pass

    return param_map, buf_map


def _iter_tf_named_weights_in_saveable(saveable, scope: str) -> Iterator[Tuple[object, str]]:
    """Yield (tensor, full_tf_name) in Saveable.get_weights() order."""
    # Trigger build first so named_parameters() returns hierarchical names.
    weights = list(saveable.get_weights())
    param_map, buf_map = _collect_tensor_name_maps(saveable)

    for w in weights:
        tf_sub = param_map.get(id(w)) or buf_map.get(id(w))
        if tf_sub is None:
            raise KeyError(
                f"Cannot map weight tensor to a TF key for scope={scope}. "
                "This likely means a buffer/param is not discoverable via named_parameters/buffers."
            )
        yield w, f"{scope}/{tf_sub}"


def _assign_from_tf_saveable_dict(*, tf_dict: Dict[str, np.ndarray], saveable, scope: str) -> Tuple[int, int, List[str]]:
    """Assign weights into saveable from a TF-style dict.

    Returns: (loaded_count, total_count, missing_keys)
    """
    loaded = 0
    missing: List[str] = []
    fallback: List[Tuple[str, str]] = []   # (torch_tf_sub, 实际用到的 tf key)
    ambiguous: List[str] = []              # 有多个同形状候选，拒绝瞎猜

    # For non-optimizer saveables, TF dict keys are typically WITHOUT the top-level scope.
    # But some forks may include the scope prefix; we try both.
    prefix = scope + "/"

    # Trigger build FIRST so named_parameters() returns hierarchical names.
    weights = list(saveable.get_weights())
    param_map, buf_map = _collect_tensor_name_maps(saveable)

    def _copy_np_to_tensor(dst_tensor, arr: np.ndarray) -> None:
        dst_shape = tuple(int(x) for x in dst_tensor.shape)
        try:
            import torch

            t = torch.from_numpy(arr).to(device=dst_tensor.device, dtype=getattr(dst_tensor, "dtype", None) or torch.float32)
            if tuple(t.shape) != dst_shape:
                t = t.reshape(dst_shape)
            dst_tensor.data.copy_(t)
        except Exception:
            dst_tensor.data.copy_(arr)

    remaining_keys = list(tf_dict.keys())
    remaining_set = set(remaining_keys)

    for w in weights:
        tf_sub = param_map.get(id(w)) or buf_map.get(id(w))
        if tf_sub is None:
            raise KeyError(f"Cannot map weight tensor to TF key in scope={scope}")

        dst_shape = tuple(int(x) for x in w.shape)

        def _try_key(key: str) -> Optional[np.ndarray]:
            if key not in remaining_set:
                return None
            src_arr = _np_to_np(tf_dict[key])
            try:
                return _convert_array_to_shape(src_arr, dst_shape)
            except Exception:
                return None

        chosen_key = None
        conv = None

        # 1) Prefer name-based mapping.
        for k in _tf_key_variants(tf_sub):
            conv = _try_key(k)
            if conv is not None:
                chosen_key = k
                break
        if chosen_key is None:
            for k in _tf_key_variants(prefix + tf_sub):
                conv = _try_key(k)
                if conv is not None:
                    chosen_key = k
                    break

        # 2) 形状兜底。
        #
        # ⚠️ 旧版是「扫到第一个形状能对上的键就用」—— 再配上 _convert_array_to_shape
        #    里"元素数相同就 reshape"的兜底，几乎总能扫到某个东西，于是实际语义
        #    变成「按字典顺序瞎猜参数」。这就是 LIAE ae=384 转出乱码的根因。
        #
        # 现在改成：先收集【形状兼容的候选】，然后
        #   - 恰好 1 个  -> 接受，并记进 fallback（说明名字匹配失败了，要报出来）
        #   - 多于 1 个  -> 记进 ambiguous，**不猜**，报 missing
        #   - 0 个       -> missing
        if chosen_key is None:
            cands: List[Tuple[str, np.ndarray]] = []
            for k in remaining_keys:
                if k not in remaining_set:
                    continue
                c = _try_key(k)
                if c is not None:
                    cands.append((k, c))
                    if len(cands) > 1:
                        break
            if len(cands) == 1:
                chosen_key, conv = cands[0]
                fallback.append((tf_sub, chosen_key))
            elif len(cands) > 1:
                ambiguous.append(tf_sub)

        if chosen_key is None or conv is None:
            missing.append(tf_sub)
            continue

        _copy_np_to_tensor(w, conv)
        remaining_set.remove(chosen_key)
        loaded += 1

    # 诊断输出：名字匹配本该命中全部张量；走到 fallback/ambiguous 就说明
    # TF 名对不上，需要人工看一眼，而不是让它静默过去。
    if fallback or ambiguous:
        print(f"    [诊断] scope={scope}: 张量 {len(weights)} 个，"
              f"名字命中 {loaded - len(fallback)} 个，形状兜底 {len(fallback)} 个，"
              f"候选歧义 {len(ambiguous)} 个")
        for a, b in fallback[:5]:
            print(f"      - 名字未命中，按唯一形状兜底: {a}  <->  {b}")
        for a in ambiguous[:5]:
            print(f"      - 有多个同形状候选，拒绝猜测: {a}")

    return loaded, len(weights), missing, fallback, ambiguous


def _assign_optimizer_from_tf_dict(*, tf_dict: Dict[str, np.ndarray], opt_spec: OptimizerSpec) -> Tuple[int, int, List[str]]:
    """Assign optimizer state tensors from TF dict.

    Primary method: name-based mapping using computed per-param TF names.
    Fallback: if mapping coverage is low and element count matches exactly, align
    by insertion order (iters + remaining states).
    """
    opt = opt_spec.opt

    weights = list(opt.get_weights())
    if len(weights) == 0:
        return 0, 0, []

    loaded = 0
    missing: List[str] = []

    def _get_arr_by_key(key: str) -> Optional[np.ndarray]:
        for k in _tf_key_variants(key):
            if k in tf_dict:
                return _np_to_np(tf_dict[k])
        return None

    # iterations
    it_arr = _get_arr_by_key("iters:0")
    if it_arr is None:
        it_arr = _get_arr_by_key("iters")
    if it_arr is not None:
        it_arr = np.asarray(it_arr).reshape(tuple(int(x) for x in weights[0].shape))
        try:
            import torch

            weights[0].data.copy_(torch.from_numpy(it_arr).to(device=weights[0].device, dtype=weights[0].dtype))
        except Exception:
            weights[0].data.copy_(it_arr)
        loaded += 1
    else:
        missing.append("iters:0")

    # remaining states (name-based)
    full_names = opt_spec.param_full_tf_names

    if opt_spec.kind == "rmsprop":
        # weights: [iters] + accus
        n = min(len(full_names), max(0, len(weights) - 1))
        for i in range(n):
            full_tf_name = full_names[i]
            base = full_tf_name.replace(":", "_")
            key = f"acc_{base}:0"
            arr = _get_arr_by_key(key)
            if arr is None:
                missing.append(key)
                continue
            dst = weights[1 + i]
            conv = _convert_array_to_shape(arr, tuple(int(x) for x in dst.shape))
            try:
                import torch

                dst.data.copy_(torch.from_numpy(conv).to(device=dst.device, dtype=dst.dtype))
            except Exception:
                dst.data.copy_(conv)
            loaded += 1

    elif opt_spec.kind == "adabelief":
        # weights: [iters] + ms + vs
        n = len(full_names)
        if len(weights) != 1 + 2 * n:
            n = min(n, max(0, (len(weights) - 1) // 2))

        for i in range(n):
            full_tf_name = full_names[i]
            base = full_tf_name.replace(":", "_")
            key = f"ms_{base}:0"
            arr = _get_arr_by_key(key)
            if arr is None:
                missing.append(key)
                continue
            dst = weights[1 + i]
            conv = _convert_array_to_shape(arr, tuple(int(x) for x in dst.shape))
            try:
                import torch

                dst.data.copy_(torch.from_numpy(conv).to(device=dst.device, dtype=dst.dtype))
            except Exception:
                dst.data.copy_(conv)
            loaded += 1

        for i in range(n):
            full_tf_name = full_names[i]
            base = full_tf_name.replace(":", "_")
            key = f"vs_{base}:0"
            arr = _get_arr_by_key(key)
            if arr is None:
                missing.append(key)
                continue
            dst = weights[1 + n + i]
            conv = _convert_array_to_shape(arr, tuple(int(x) for x in dst.shape))
            try:
                import torch

                dst.data.copy_(torch.from_numpy(conv).to(device=dst.device, dtype=dst.dtype))
            except Exception:
                dst.data.copy_(conv)
            loaded += 1

    else:
        raise ValueError(f"Unknown optimizer kind: {opt_spec.kind}")

    # Fallback by insertion order (iters + remaining) when dict size matches.
    # Validate shapes; if fails, use greedy shape matching.
    if loaded != len(weights) and len(tf_dict) == len(weights):
        tf_items = list(tf_dict.items())
        it_idx = None
        for idx, (k, _v) in enumerate(tf_items):
            if str(k).startswith("iters"):
                it_idx = idx
                break

        if it_idx is not None:
            state_items = tf_items[:it_idx] + tf_items[it_idx + 1 :]
            if len(state_items) == len(weights) - 1:
                # iters
                it_arr = _np_to_np(tf_items[it_idx][1])
                it_arr = np.asarray(it_arr).reshape(tuple(int(x) for x in weights[0].shape))
                try:
                    import torch

                    weights[0].data.copy_(torch.from_numpy(it_arr).to(device=weights[0].device, dtype=weights[0].dtype))
                except Exception:
                    weights[0].data.copy_(it_arr)

                ok = True
                # states by insertion order
                for dst, (_k, src_val) in zip(weights[1:], state_items):
                    arr = _np_to_np(src_val)
                    try:
                        conv = _convert_array_to_shape(arr, tuple(int(x) for x in dst.shape))
                    except Exception:
                        ok = False
                        break
                    try:
                        import torch

                        dst.data.copy_(torch.from_numpy(conv).to(device=dst.device, dtype=dst.dtype))
                    except Exception:
                        dst.data.copy_(conv)

                if ok:
                    return len(weights), len(weights), []

                # Greedy shape match among remaining state items.
                remaining = list(state_items)
                used = [False] * len(remaining)
                for dst in weights[1:]:
                    dst_shape = tuple(int(x) for x in dst.shape)
                    found = False
                    for j, (_k, src_val) in enumerate(remaining):
                        if used[j]:
                            continue
                        arr = _np_to_np(src_val)
                        try:
                            conv = _convert_array_to_shape(arr, dst_shape)
                        except Exception:
                            continue
                        try:
                            import torch

                            dst.data.copy_(torch.from_numpy(conv).to(device=dst.device, dtype=dst.dtype))
                        except Exception:
                            dst.data.copy_(conv)
                        used[j] = True
                        found = True
                        break
                    if not found:
                        # give up; keep partially loaded
                        break
                if all(used):
                    return len(weights), len(weights), []

    return loaded, len(weights), missing


def _ensure_leras_cpu():
    from core.leras import nn as lnn

    # Force CPU build for determinism and to avoid GPU requirements.
    lnn.initialize(lnn.DeviceConfig.CPU(), data_format="NCHW")


def _build_saehd_specs(options: dict) -> Tuple[List[SaveableSpec], List[OptimizerSpec]]:
    from core.leras import nn as lnn

    resolution = int(options.get("resolution", 128))
    archi = str(options.get("archi", "liae-ud"))
    archi_split = archi.split("-")
    archi_type = archi_split[0]
    archi_opts = archi_split[1] if len(archi_split) == 2 else ""

    ae_dims = int(options.get("ae_dims", 256))
    e_dims = int(options.get("e_dims", 64))
    d_dims = int(options.get("d_dims", 64))
    d_mask_dims = int(options.get("d_mask_dims", max(16, d_dims // 3)))

    gan_power = float(options.get("gan_power", 0.0) or 0.0)
    gan_patch_size = int(options.get("gan_patch_size", max(3, resolution // 8)))
    gan_dims = int(options.get("gan_dims", 16))

    true_face_power = float(options.get("true_face_power", 0.0) or 0.0)

    adabelief = bool(options.get("adabelief", True))
    clipgrad = bool(options.get("clipgrad", False))

    lr_dropout_opt = str(options.get("lr_dropout", "n"))
    lr_dropout = 0.3 if lr_dropout_opt in ("y", "cpu") else 1.0
    lr_cos = 500 if lr_dropout_opt in ("y", "cpu") else 0
    clipnorm = 1.0 if clipgrad else 0.0

    model_archi = lnn.DeepFakeArchi(resolution, opts=archi_opts)

    saveables: List[SaveableSpec] = []
    optimizers: List[OptimizerSpec] = []

    tracked_full_tf_names: List[str] = []
    tracked_tensors: List[object] = []

    def _append_tracked(scope: str, obj):
        nonlocal tracked_full_tf_names, tracked_tensors
        for w, full_name in _iter_tf_named_weights_in_saveable(obj, scope):
            tracked_tensors.append(w)
            tracked_full_tf_names.append(full_name)

    if archi_type == "df":
        encoder = model_archi.Encoder(in_ch=3, e_ch=e_dims, name="encoder")
        encoder_out_ch = encoder.get_out_ch() * encoder.get_out_res(resolution) ** 2

        inter = model_archi.Inter(in_ch=encoder_out_ch, ae_ch=ae_dims, ae_out_ch=ae_dims, name="inter")
        inter_out_ch = inter.get_out_ch()

        decoder_src = model_archi.Decoder(in_ch=inter_out_ch, d_ch=d_dims, d_mask_ch=d_mask_dims, name="decoder_src")
        decoder_dst = model_archi.Decoder(in_ch=inter_out_ch, d_ch=d_dims, d_mask_ch=d_mask_dims, name="decoder_dst")

        saveables += [
            SaveableSpec(scope="encoder", obj=encoder, filename="encoder.pth"),
            SaveableSpec(scope="inter", obj=inter, filename="inter.pth"),
            SaveableSpec(scope="decoder_src", obj=decoder_src, filename="decoder_src.pth"),
            SaveableSpec(scope="decoder_dst", obj=decoder_dst, filename="decoder_dst.pth"),
        ]

        _append_tracked("encoder", encoder)
        _append_tracked("inter", inter)
        _append_tracked("decoder_src", decoder_src)
        _append_tracked("decoder_dst", decoder_dst)

        # src_dst optimizer
        OptimizerClass = lnn.AdaBelief if adabelief else lnn.RMSprop
        src_dst_opt = OptimizerClass(tracked_tensors, lr=5e-5, lr_dropout=lr_dropout, lr_cos=lr_cos, clipnorm=clipnorm, name="src_dst_opt")
        kind = "adabelief" if adabelief else "rmsprop"
        optimizers += [
            OptimizerSpec(
                scope="src_dst_opt",
                opt=src_dst_opt,
                filename="src_dst_opt.pth",
                kind=kind,
                param_full_tf_names=tracked_full_tf_names,
            )
        ]

        if true_face_power != 0.0:
            code_discriminator = lnn.CodeDiscriminator(ae_dims, code_res=inter.get_out_res(), name="dis")
            saveables += [SaveableSpec(scope="dis", obj=code_discriminator, filename="code_discriminator.pth")]

            # D_code optimizer
            d_code_names: List[str] = []
            d_code_tensors: List[object] = []
            for w, full_name in _iter_tf_named_weights_in_saveable(code_discriminator, "dis"):
                d_code_tensors.append(w)
                d_code_names.append(full_name)

            d_code_opt = OptimizerClass(d_code_tensors, lr=5e-5, lr_dropout=lr_dropout, lr_cos=lr_cos, clipnorm=clipnorm, name="D_code_opt")
            optimizers += [
                OptimizerSpec(
                    scope="D_code_opt",
                    opt=d_code_opt,
                    filename="D_code_opt.pth",
                    kind=kind,
                    param_full_tf_names=d_code_names,
                )
            ]

        if gan_power != 0.0:
            d_src = lnn.UNetPatchDiscriminator(patch_size=gan_patch_size, in_ch=3, base_ch=gan_dims, name="D_src")
            saveables += [SaveableSpec(scope="D_src", obj=d_src, filename="GAN.pth")]

            gan_names: List[str] = []
            gan_tensors: List[object] = []
            for w, full_name in _iter_tf_named_weights_in_saveable(d_src, "D_src"):
                gan_tensors.append(w)
                gan_names.append(full_name)

            gan_opt = OptimizerClass(gan_tensors, lr=5e-5, lr_dropout=lr_dropout, lr_cos=lr_cos, clipnorm=clipnorm, name="GAN_opt")
            optimizers += [
                OptimizerSpec(
                    scope="GAN_opt",
                    opt=gan_opt,
                    filename="GAN_opt.pth",
                    kind=kind,
                    param_full_tf_names=gan_names,
                )
            ]

    elif archi_type == "liae":
        encoder = model_archi.Encoder(in_ch=3, e_ch=e_dims, name="encoder")
        encoder_out_ch = encoder.get_out_ch() * encoder.get_out_res(resolution) ** 2

        inter_ab = model_archi.Inter(in_ch=encoder_out_ch, ae_ch=ae_dims, ae_out_ch=ae_dims * 2, name="inter_AB")
        inter_b = model_archi.Inter(in_ch=encoder_out_ch, ae_ch=ae_dims, ae_out_ch=ae_dims * 2, name="inter_B")

        inter_out_ch = inter_ab.get_out_ch()
        decoder = model_archi.Decoder(in_ch=inter_out_ch * 2, d_ch=d_dims, d_mask_ch=d_mask_dims, name="decoder")

        saveables += [
            SaveableSpec(scope="encoder", obj=encoder, filename="encoder.pth"),
            SaveableSpec(scope="inter_AB", obj=inter_ab, filename="inter_AB.pth"),
            SaveableSpec(scope="inter_B", obj=inter_b, filename="inter_B.pth"),
            SaveableSpec(scope="decoder", obj=decoder, filename="decoder.pth"),
        ]

        _append_tracked("encoder", encoder)
        _append_tracked("inter_AB", inter_ab)
        _append_tracked("inter_B", inter_b)
        _append_tracked("decoder", decoder)

        OptimizerClass = lnn.AdaBelief if adabelief else lnn.RMSprop
        src_dst_opt = OptimizerClass(tracked_tensors, lr=5e-5, lr_dropout=lr_dropout, lr_cos=lr_cos, clipnorm=clipnorm, name="src_dst_opt")
        kind = "adabelief" if adabelief else "rmsprop"
        optimizers += [
            OptimizerSpec(
                scope="src_dst_opt",
                opt=src_dst_opt,
                filename="src_dst_opt.pth",
                kind=kind,
                param_full_tf_names=tracked_full_tf_names,
            )
        ]

        if gan_power != 0.0:
            d_src = lnn.UNetPatchDiscriminator(patch_size=gan_patch_size, in_ch=3, base_ch=gan_dims, name="D_src")
            saveables += [SaveableSpec(scope="D_src", obj=d_src, filename="GAN.pth")]

            gan_names: List[str] = []
            gan_tensors: List[object] = []
            for w, full_name in _iter_tf_named_weights_in_saveable(d_src, "D_src"):
                gan_tensors.append(w)
                gan_names.append(full_name)

            gan_opt = OptimizerClass(gan_tensors, lr=5e-5, lr_dropout=lr_dropout, lr_cos=lr_cos, clipnorm=clipnorm, name="GAN_opt")
            optimizers += [
                OptimizerSpec(
                    scope="GAN_opt",
                    opt=gan_opt,
                    filename="GAN_opt.pth",
                    kind=kind,
                    param_full_tf_names=gan_names,
                )
            ]

    else:
        raise ValueError(f"Unsupported SAEHD archi_type: {archi_type}")

    return saveables, optimizers


def _build_quick96_specs(options: dict) -> Tuple[List[SaveableSpec], List[OptimizerSpec]]:
    # Quick96 is fixed-size in this repo.
    from core.leras import nn as lnn

    resolution = 96
    input_ch = 3
    ae_dims = 128
    e_dims = 64
    d_dims = 64
    d_mask_dims = 16

    model_archi = lnn.DeepFakeArchi(resolution, opts="ud")

    encoder = model_archi.Encoder(in_ch=input_ch, e_ch=e_dims, name="encoder")
    encoder_out_res = encoder.get_out_res(resolution)
    encoder_out_ch = encoder.get_out_ch() * encoder_out_res * encoder_out_res

    inter = model_archi.Inter(in_ch=encoder_out_ch, ae_ch=ae_dims, ae_out_ch=ae_dims, name="inter")
    inter_out_ch = inter.get_out_ch()

    decoder_src = model_archi.Decoder(in_ch=inter_out_ch, d_ch=d_dims, d_mask_ch=d_mask_dims, name="decoder_src")
    decoder_dst = model_archi.Decoder(in_ch=inter_out_ch, d_ch=d_dims, d_mask_ch=d_mask_dims, name="decoder_dst")

    saveables = [
        SaveableSpec(scope="encoder", obj=encoder, filename="encoder.pth"),
        SaveableSpec(scope="inter", obj=inter, filename="inter.pth"),
        SaveableSpec(scope="decoder_src", obj=decoder_src, filename="decoder_src.pth"),
        SaveableSpec(scope="decoder_dst", obj=decoder_dst, filename="decoder_dst.pth"),
    ]

    full_names: List[str] = []
    tensors: List[object] = []
    for scope, obj in (("encoder", encoder), ("inter", inter), ("decoder_src", decoder_src), ("decoder_dst", decoder_dst)):
        for w, full in _iter_tf_named_weights_in_saveable(obj, scope):
            tensors.append(w)
            full_names.append(full)

    # Quick96 optimizer in this repo is RMSprop with lr=2e-4 and lr_dropout=0.3
    opt = lnn.RMSprop(tensors, lr=2e-4, lr_dropout=0.3, name="src_dst_opt")
    optimizers = [
        OptimizerSpec(
            scope="src_dst_opt",
            opt=opt,
            filename="src_dst_opt.pth",
            kind="rmsprop",
            param_full_tf_names=full_names,
        )
    ]

    return saveables, optimizers


def _build_xseg_specs(options: dict) -> Tuple[List[SaveableSpec], List[OptimizerSpec]]:
    from core.leras import nn as lnn

    resolution = 256
    name = "XSeg"

    model = lnn.XSeg(3, 32, 1, name=name)

    saveables = [SaveableSpec(scope=name, obj=model, filename=f"{name}_{resolution}.pth")]

    # Optimizer name in wrappers is 'opt'.
    full_names: List[str] = []
    tensors: List[object] = []
    for w, full in _iter_tf_named_weights_in_saveable(model, name):
        tensors.append(w)
        full_names.append(full)

    opt = lnn.RMSprop(tensors, lr=0.0001, lr_dropout=0.3, name="opt")
    optimizers = [
        OptimizerSpec(
            scope="opt",
            opt=opt,
            filename=f"{name}_{resolution}_opt.pth",
            kind="rmsprop",
            param_full_tf_names=full_names,
        )
    ]

    return saveables, optimizers


def _build_amp_specs(options: dict) -> Tuple[List[SaveableSpec], List[OptimizerSpec]]:
    """Build AMP modules by importing its PyTorch implementation and instantiating only the submodules.

    AMP defines its Encoder/Inter/Decoder classes inside on_initialize, so we instantiate AMPModel
    in a non-training mode using a temp-like options state is not required here; instead we rebuild
    its module graph by constructing the submodules directly via the model class.

    To keep this script robust and non-interactive, we avoid creating any sample generators.
    """

    # Instantiate AMPModel on CPU but with is_training=False to avoid generators.
    from core.leras import nn as lnn
    from models.Model_AMP.Model_pytorch import AMPModel

    # Create a minimal dummy saved_models_path holder; AMPModel's on_initialize_options reads options
    # from self.options which is populated from data.dat by ModelBase normally. Here we bypass that
    # by creating the object via __new__ and calling its build logic directly is risky.
    # Instead, we instantiate normally but provide a temp folder and inject options afterward.
    import tempfile

    # Write a minimal data.dat so ModelBase sees iter!=0 and doesn't prompt.
    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        # Model name must match '<something>_AMP' pattern in ModelBase storage naming.
        dummy_model_name = "__convtool__AMP"
        data_path = td_path / f"{dummy_model_name}_data.dat"
        data = {"iter": 1, "options": options, "loss_history": [], "sample_for_preview": None, "choosed_gpu_indexes": None}
        data_path.write_bytes(pickle.dumps(data))

        m = AMPModel(
            is_training=False,
            is_exporting=False,
            saved_models_path=td_path,
            training_data_src_path=td_path,
            training_data_dst_path=td_path,
            pretraining_data_path=None,
            pretrained_model_path=None,
            no_preview=True,
            force_model_name=None,
            force_gpu_idxs=None,
            cpu_only=True,
            debug=False,
            force_model_class_name=dummy_model_name,
            silent_start=True,
        )

        # Extract built submodules.
        encoder = m.encoder
        inter_src = m.inter_src
        inter_dst = m.inter_dst
        decoder = m.decoder

    saveables: List[SaveableSpec] = [
        SaveableSpec(scope="encoder", obj=encoder, filename="encoder.pth"),
        SaveableSpec(scope="inter_src", obj=inter_src, filename="inter_src.pth"),
        SaveableSpec(scope="inter_dst", obj=inter_dst, filename="inter_dst.pth"),
        SaveableSpec(scope="decoder", obj=decoder, filename="decoder.pth"),
    ]

    # Optimizer specs (AMP uses AdaBelief for G + optional GAN)
    clipgrad = bool(options.get("clipgrad", False))
    lr_dropout_opt = str(options.get("lr_dropout", "n"))
    lr_dropout = 0.3 if lr_dropout_opt in ("y", "cpu") else 1.0
    lr_cos = 500 if lr_dropout_opt in ("y", "cpu") else 0
    clipnorm = 1.0 if clipgrad else 0.0

    gan_power = float(options.get("gan_power", 0.0) or 0.0)
    gan_patch_size = int(options.get("gan_patch_size", int(options.get("resolution", 224)) // 8))
    gan_dims = int(options.get("gan_dims", 16))

    full_names: List[str] = []
    tensors: List[object] = []
    for scope, obj in (("encoder", encoder), ("decoder", decoder), ("inter_src", inter_src), ("inter_dst", inter_dst)):
        for w, full in _iter_tf_named_weights_in_saveable(obj, scope):
            tensors.append(w)
            full_names.append(full)

    src_dst_opt = lnn.AdaBelief(tensors, lr=5e-5, lr_dropout=lr_dropout, lr_cos=lr_cos, clipnorm=clipnorm, name="src_dst_opt")
    optimizers: List[OptimizerSpec] = [
        OptimizerSpec(
            scope="src_dst_opt",
            opt=src_dst_opt,
            filename="src_dst_opt.pth",
            kind="adabelief",
            param_full_tf_names=full_names,
        )
    ]

    if gan_power != 0.0:
        gan = lnn.UNetPatchDiscriminator(patch_size=gan_patch_size, in_ch=3, base_ch=gan_dims, name="GAN")
        saveables += [SaveableSpec(scope="GAN", obj=gan, filename="GAN.pth")]

        gan_names: List[str] = []
        gan_tensors: List[object] = []
        for w, full in _iter_tf_named_weights_in_saveable(gan, "GAN"):
            gan_tensors.append(w)
            gan_names.append(full)

        gan_opt = lnn.AdaBelief(gan_tensors, lr=5e-5, lr_dropout=lr_dropout, lr_cos=lr_cos, clipnorm=clipnorm, name="GAN_opt")
        optimizers += [
            OptimizerSpec(
                scope="GAN_opt",
                opt=gan_opt,
                filename="GAN_opt.pth",
                kind="adabelief",
                param_full_tf_names=gan_names,
            )
        ]

    return saveables, optimizers


def _build_specs_for_model(model_class: str, options: dict) -> Tuple[List[SaveableSpec], List[OptimizerSpec]]:
    model_class = str(model_class)
    if model_class == "SAEHD":
        return _build_saehd_specs(options)
    if model_class == "Quick96":
        return _build_quick96_specs(options)
    if model_class == "AMP":
        return _build_amp_specs(options)
    if model_class == "XSeg":
        return _build_xseg_specs(options)

    raise ValueError(f"Unsupported model class for conversion: {model_class}")


def _discover_models(src_dir: Path) -> List[Tuple[str, Optional[str]]]:
    """Return list of (model_name, model_class).

    model_name is the prefix used in file naming (e.g., 'MyModel_SAEHD' or 'XSeg').
    model_class is the class suffix (e.g., 'SAEHD' or 'XSeg').
    """
    out: List[Tuple[str, Optional[str]]] = []
    for fp in sorted(src_dir.glob("*_data.dat")):
        stem = fp.name[: -len("_data.dat")]
        if "_" in stem:
            # <base>_<class>
            _, cls = stem.rsplit("_", 1)
            out.append((stem, cls))
        else:
            # XSeg style
            out.append((stem, stem))
    return out


def _convert_one_model(*, src_dir: Path, dst_dir: Path, model_name: str, model_class: str,
                       strict: bool = False) -> None:
    src_data = src_dir / f"{model_name}_data.dat"
    if not src_data.exists():
        raise FileNotFoundError(src_data)

    model_data = _read_pickle(src_data)
    options = dict(model_data.get("options", {}))

    saveables, optimizers = _build_specs_for_model(model_class, options)

    dst_dir.mkdir(parents=True, exist_ok=True)

    # Copy data.dat as-is (options/iter/loss_history).
    (dst_dir / src_data.name).write_bytes(src_data.read_bytes())

    # Copy default options / summary if present.
    # DFL convention: <ModelClass>_default_options.dat (no model_name prefix).
    default_opt = src_dir / f"{model_class}_default_options.dat"
    if default_opt.exists():
        (dst_dir / default_opt.name).write_bytes(default_opt.read_bytes())

    # Summary is optional but handy.
    summary = src_dir / f"{model_name}_summary.txt"
    if summary.exists():
        (dst_dir / summary.name).write_bytes(summary.read_bytes())

    # 1) Convert model weights (before archiving, so _find_existing_weight_file can find .npy).
    _bad: List[str] = []
    for spec in saveables:
        src_file = src_dir / f"{model_name}_{spec.filename}"
        src_existing = _find_existing_weight_file(src_file)
        if src_existing is None:
            print(f"  WARN: no source file found for {spec.filename}, saving initialized weights")
            dst_file = dst_dir / f"{model_name}_{spec.filename}"
            spec.obj.save_weights(dst_file)
            continue

        tf_dict = _read_saveable_dict(src_existing)
        loaded, total, missing, fallback, ambiguous = _assign_from_tf_saveable_dict(
            tf_dict=tf_dict, saveable=spec.obj, scope=spec.scope)

        if missing or ambiguous:
            print(f"  ⚠️ {spec.filename}: 名字匹配未覆盖全部张量")
            print(f"       missing={len(missing)}（保持初始化值，训练会带着垃圾权重跑）")
            print(f"       ambiguous={len(ambiguous)}（有多个同形状候选，拒绝猜测，同上）")
            for _m in missing[:5]:
                print(f"       missing:   {_m}")
            for _m in ambiguous[:5]:
                print(f"       ambiguous: {_m}")
            _bad.append(spec.filename)
        if fallback:
            print(f"  ⚠️ {spec.filename}: {len(fallback)} 个张量是靠形状兜底硬塞的（名字没对上）")
            _bad.append(spec.filename)

        dst_file = dst_dir / f"{model_name}_{spec.filename}"
        spec.obj.save_weights(dst_file)
        print(f"  converted {spec.filename}  ({loaded}/{total} 名字命中)"
              + (f"  形状兜底 {len(fallback)}" if fallback else ""))

    # 2) Convert optimizer states (best-effort).
    for opt_spec in optimizers:
        src_file = src_dir / f"{model_name}_{opt_spec.filename}"
        src_existing = _find_existing_weight_file(src_file)

        if src_existing is not None:
            tf_dict = _read_saveable_dict(src_existing)
            _assign_optimizer_from_tf_dict(tf_dict=tf_dict, opt_spec=opt_spec)

        dst_file = dst_dir / f"{model_name}_{opt_spec.filename}"
        opt_spec.opt.save_weights(dst_file)

    # 收尾总结：本次转换里有没有「靠猜」的地方。有就明说，别让人以为转成功了。
    if _bad:
        print("")
        print("  " + "!" * 66)
        print(f"  ⚠️ 本次转换有 {len(_bad)} 个文件没能全部按名字对上权重：")
        for _f in dict.fromkeys(_bad):
            print(f"       - {_f}")
        print("     这说明转换器的 TF 名映射和这个模型对不上。转出来的模型可能带错权重，")
        print("     请先用它跑一次预览确认，不要直接接着训练。")
        print("  " + "!" * 66)
        print("")
        if strict:
            raise RuntimeError(
                f"strict 模式：{len(_bad)} 个文件没能全部按名字对上权重：" + ", ".join(dict.fromkeys(_bad)))

    # 3) Archive old .npy to old/ AFTER conversion so source files are still accessible.
    npy_files = list(src_dir.glob(f"{model_name}_*.npy"))
    if npy_files:
        old_dir = src_dir / "old"
        old_dir.mkdir(exist_ok=True)
        for npy in npy_files:
            dest = old_dir / npy.name
            if not dest.exists():
                shutil.move(str(npy), str(dest))
                print(f"  archived {npy.name} → old/")
        print(f"  {len(npy_files)} old .npy files archived to old/")


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Convert DFL TF/Leras models to deepfacelab_Torch Saveable .pth")
    p.add_argument("--src", required=True, type=Path, help="源 saved_models 目录（原版 DFL / DeepFaceLab-master 导出的模型目录）")
    p.add_argument("--dst", required=True, type=Path, help="目标 saved_models 目录（deepfacelab_Torch 使用的模型目录）")

    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--all", action="store_true", help="扫描并转换 src 下所有 *_data.dat 模型（包含 XSeg）")
    g.add_argument("--model", type=str, help="指定模型类型：SAEHD / AMP / Quick96 / XSeg")

    p.add_argument("--name", type=str, default=None, help="指定模型名（不含 _data.dat 后缀，例如 MyModel_SAEHD 或 XSeg）")

    p.add_argument("--strict", action="store_true",
                   help="权重名字没全部对上时直接报错退出（默认只打印警告并继续）。"
                        "怀疑转出来的模型有问题时加这个，能立刻定位到哪些键对不上。")

    args = p.parse_args(argv)

    src_dir: Path = args.src
    dst_dir: Path = args.dst

    if not src_dir.exists():
        raise FileNotFoundError(src_dir)

    _ensure_leras_cpu()

    if args.all:
        models = _discover_models(src_dir)
        for model_name, model_class in models:
            if model_class is None:
                continue
            _convert_one_model(src_dir=src_dir, dst_dir=dst_dir, model_name=model_name,
                               model_class=model_class, strict=bool(args.strict))
        return 0

    # single
    model_class = str(args.model)
    if args.name is None:
        raise ValueError("--name is required when using --model")
    model_name = str(args.name)
    _convert_one_model(src_dir=src_dir, dst_dir=dst_dir, model_name=model_name,
                       model_class=model_class, strict=bool(args.strict))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
