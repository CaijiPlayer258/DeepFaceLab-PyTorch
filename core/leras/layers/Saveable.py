import io
import pickle
import struct
from pathlib import Path
from core import pathex
import numpy as np
import torch
import os
import zipfile

from core.leras.nn import nn

class Saveable():
    def __init__(self, name=None):
        self.name = name

    #override
    def get_weights(self):
        #return torch parameters that should be initialized/loaded/saved
        return []

    #override
    def get_weights_np(self):
        weights = self.get_weights()
        if len(weights) == 0:
            return []
        return [w.detach().cpu().to(torch.float32).numpy() if w.dtype == torch.bfloat16 else w.detach().cpu().numpy() for w in weights]

    def set_weights(self, new_weights):
        weights = self.get_weights()
        if len(weights) != len(new_weights):
            raise ValueError ('len of lists mismatch')

        for w, new_w in zip(weights, new_weights):
            if isinstance(new_w, torch.nn.Parameter) or isinstance(new_w, torch.Tensor):
                src = new_w.data if hasattr(new_w, 'data') else new_w
                src = src.to(device=w.device, dtype=w.dtype)
                w.data.copy_(src)
            else:
                if not isinstance(new_w, np.ndarray):
                    new_w = np.array(new_w)
                src = torch.from_numpy(new_w).reshape(w.shape).to(device=w.device, dtype=w.dtype)
                w.data.copy_(src)

    def _weight_to_npy_bytes(self, w, force_dtype=None):
        """Convert a single weight tensor to .npy bytes (streaming-friendly)."""
        w_t = w.detach().cpu()
        if w_t.dtype == torch.bfloat16:
            w_t = w_t.to(torch.float32)
        w_val = w_t.numpy().copy()
        if force_dtype is not None:
            w_val = w_val.astype(force_dtype)
        buf = io.BytesIO()
        np.save(buf, w_val)
        return buf.getvalue()

    def save_weights(self, filename, force_dtype=None):
        if self.name is None:
            raise Exception("name must be defined.")

        p = Path(filename)
        p_tmp = p.parent / (p.name + '.tmp')
        weights = self.get_weights()

        # 逐权重写入 zip（不建大 dict），每个权重存为 param_{i}.npy
        with zipfile.ZipFile(p_tmp, 'w', zipfile.ZIP_STORED) as zf:
            for i, w in enumerate(weights):
                npy_bytes = self._weight_to_npy_bytes(w, force_dtype=force_dtype)
                zf.writestr(f"param_{i}.npy", npy_bytes)

        if p.exists():
            p.unlink()
        p_tmp.rename(p)

    @staticmethod
    def _detect_format(filepath):
        """Detect whether file is pickle (legacy) or zip+npy (current)."""
        with open(filepath, 'rb') as f:
            magic = f.read(4)
        # Pickle protocol 2+: \\x80\\x02-\\x05
        if magic and magic[0] == 0x80 and 2 <= magic[1] <= 5:
            return 'pickle'
        # Zip: PK\\x03\\x04
        if magic[:2] == b'PK':
            return 'zip_npy'
        return 'unknown'

    def load_weights(self, filename):
        """
        returns True if file exists
        """
        filepath = Path(filename)

        if not filepath.exists():
            alt = None
            if filepath.suffix == '.pth':
                alt = filepath.with_suffix('.npy')
            elif filepath.suffix == '.npy':
                alt = filepath.with_suffix('.pth')
            if alt is not None and alt.exists():
                filepath = alt

        if not filepath.exists():
            return False

        fmt = self._detect_format(filepath)
        if fmt == 'zip_npy':
            # New format: zip of .npy files (memory-efficient save)
            try:
                with zipfile.ZipFile(filepath, 'r') as zf:
                    d = {}
                    for name in zf.namelist():
                        if name.endswith('.npy'):
                            key = name[:-4]  # strip '.npy'
                            with zf.open(name) as entry:
                                buf = io.BytesIO(entry.read())
                                d[key] = np.load(buf)
                return self._load_pt_weights(d)
            except Exception as e:
                print(
                    f"[WARN] Failed to load zip+npy weights from '{os.fspath(filepath)}': {e}."
                )
                return False

        # Legacy pickle format
        try:
            with open(filepath, 'rb') as f:
                d = pickle.load(f)
        except (pickle.UnpicklingError, EOFError, ValueError, OSError) as e:
            try:
                with open(filepath, 'rb') as f:
                    head = f.read(32)
            except Exception:
                head = b''
            print(
                f"[WARN] Failed to load weights from '{os.fspath(filepath)}': {e}. "
                f"This file does not look like a pickle weights file. "
                f"Header bytes: {head!r}. "
                f"You may need to delete/replace the corrupted or wrong-format weights file."
            )
            return False

        # Detect original DFL (TensorFlow) format: keys contain TF scope names
        if any(isinstance(k, str) and ':0' in k for k in d.keys()):
            return self._load_tf_weights(d)

        return self._load_pt_weights(d)

    def _load_pt_weights(self, d):
        """Load current-format weights (param_0, param_1, ...)."""
        weights = self.get_weights()

        if self.name is None:
            raise Exception("name must be defined.")

        try:
            for i, w in enumerate(weights):
                w_name = f"param_{i}"
                w_val = d.get(w_name, None)

                if w_val is None:
                    pass
                else:
                    w_val = np.reshape(w_val, w.shape)
                    src = torch.from_numpy(w_val).to(device=w.device, dtype=w.dtype)
                    w.data.copy_(src)
        except:
            return False

        return True

    def _load_tf_weights(self, d):
        """Load original DeepFaceLab (TensorFlow) .npy weights.

        NOTE: this is an independent COPY of the matching logic in
        tools/convert_dfl_tf_to_torch.py (that CLI is not imported here, so the
        two must be kept in sync by hand). It is the code path actually used by
        training auto-convert (load_weights -> ':0' in key) and by the merger
        page (.npy -> DFM via tools/export_dfm.py convert_npy_to_pth).

          1. Build TF name -> param mapping from layer hierarchy
          2. Try exact name match first, then with/without the ':0' suffix
          3. Fall back to shape-based matching ONLY when the candidate is unique
          4. Two-pass: verify ALL params match before applying any
        """
        if not hasattr(self, '_get_tf_weight_names'):
            return False
        tf_weights = self._get_tf_weight_names()
        if not tf_weights:
            return False

        remaining_keys = set(d.keys())
        validated: List[object] = []

        def _convert_to_shape(src: np.ndarray, dst_shape) -> np.ndarray:
            dst_shape = tuple(int(x) for x in dst_shape)
            if tuple(src.shape) == dst_shape:
                return src
            # Conv2D TF NHWC (H,W,in,out) -> Torch NCHW (out,in,H,W)
            if src.ndim == 4 and len(dst_shape) == 4:
                if (src.shape[0], src.shape[1], src.shape[2], src.shape[3]) == (
                    dst_shape[2], dst_shape[3], dst_shape[1], dst_shape[0]):
                    return np.transpose(src, (3, 2, 0, 1))
            # Dense transpose fallback
            if src.ndim == 2 and len(dst_shape) == 2:
                if src.shape[::-1] == dst_shape:
                    return src.T
            # Degenerate-dimension reshape only (squeeze / unsqueeze).
            # A free element-count reshape is deliberately NOT allowed: whenever
            # two layers happen to hold the same number of elements it would
            # silently reinterpret one parameter as another. That is exactly what
            # turned large LIAE models (e.g. ae_dims=352 at res 384) into
            # solid-colour output.
            if int(np.prod(src.shape)) == int(np.prod(dst_shape)):
                _s = tuple(int(x) for x in src.shape if int(x) != 1)
                _d = tuple(int(x) for x in dst_shape if int(x) != 1)
                if _s == _d:
                    return np.reshape(src, dst_shape)
            raise ValueError(f"shape mismatch: src {src.shape} -> dst {dst_shape}")

        def _try_key(key: str, dst_shape) -> object:
            if key not in remaining_keys:
                return None
            try:
                return _convert_to_shape(d[key], dst_shape)
            except ValueError:
                return None

        n_exact = n_suffix = n_shape = 0

        # --- Pass 1: name-based + shape-fallback matching ---
        for tf_name, param in tf_weights:
            dst_shape = tuple(int(x) for x in param.shape)
            chosen = None
            chosen_key = None

            # 1a) exact name
            chosen = _try_key(tf_name, dst_shape)
            if chosen is not None:
                chosen_key = tf_name
                n_exact += 1

            # 1b) without / with :0 suffix
            if chosen is None:
                alt = tf_name[:-2] if tf_name.endswith(':0') else tf_name + ':0'
                chosen = _try_key(alt, dst_shape)
                if chosen is not None:
                    chosen_key = alt
                    n_suffix += 1

            # 1c) shape-based fallback among unused keys -- ONLY if unique.
            # Two same-shaped layers in one model (the decoder alone holds
            # (1024,1024,3,3) x2 and (512,512,3,3) x2) are not distinguishable
            # by shape, so guessing here corrupts the model silently.
            if chosen is None:
                cands = [c for c in sorted(remaining_keys)
                         if _try_key(c, dst_shape) is not None]
                if len(cands) > 1:
                    # Prefer candidates already in torch layout (no transpose)
                    _exact = [c for c in cands if tuple(d[c].shape) == dst_shape]
                    if len(_exact) == 1:
                        cands = _exact
                if len(cands) == 1:
                    chosen = _try_key(cands[0], dst_shape)
                    chosen_key = cands[0]
                    n_shape += 1
                elif len(cands) > 1:
                    name_hint = self.name or type(self).__name__
                    print(f"[WARN] TF weight '{tf_name}' (shape {param.shape}) 在 {name_hint} 中有 "
                          f"{len(cands)} 个同形候选，无法唯一确定，放弃转换以免写错权重：")
                    for c in cands[:8]:
                        print(f"         candidate: {c} {d[c].shape}")
                    return False

            if chosen is None:
                name_hint = self.name or type(self).__name__
                print(f"[WARN] TF weight '{tf_name}' (shape {param.shape}) 在 {name_hint} 中无匹配，跳过加载")
                return False  # refuse partial load

            remaining_keys.discard(chosen_key)
            validated.append((param, chosen))

        # --- Pass 2: apply ---
        for param, w_val in validated:
            w_val = np.reshape(w_val, param.shape)
            param.data.copy_(
                torch.from_numpy(w_val).to(device=param.device, dtype=param.dtype)
            )

        if n_shape:
            name_hint = self.name or type(self).__name__
            print(f"[INFO] {name_hint}: TF weights 名称命中 {n_exact} + 后缀命中 {n_suffix} "
                  f"+ 形状推断 {n_shape} / 共 {len(tf_weights)}")
        return True

    def init_weights(self):
        # PyTorch initializes weights automatically
        pass

nn.Saveable = Saveable
