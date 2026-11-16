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
from core.leras import load_events as _lev

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
        （失败原因/形状等明细见 load_weights_ex，也会写在 self.last_load_result 上）
        """
        return self.load_weights_ex(filename).ok

    def load_weights_ex(self, filename, model_prefix=None):
        """加载权重，返回带**失败原因**的 LoadResult（bool(result) == 是否成功）。

        和旧的 load_weights 相比补了三件事：

        1. **四种失败可分辨**（load_events.REASON_*）：
             missing  文件不存在
             renamed  目录里有像它的 .pth/.npy，但名字对不上
             shape    参数个数/形状与当前网络不符（含 param_N 缺失）
             corrupt  文件在，但内容不是可用的权重（magic/zip/pickle/npy 坏）
        2. **原子性**：先全量校验，全部通过才写参数。
           旧实现在循环里 copy_，第 k 个参数形状不符时前 k-1 个已经写进去了 ——
           模块停在"一半新一半旧"的状态，上层再把它当"加载失败→随机初始化"处理，
           于是权重和优化器（动量/二阶矩）彻底不同步。
        3. **不吞异常**：出来的是结构化的 reason + 出问题的参数层，而不是一行 stderr。

        参数 model_prefix：模型的文件名前缀（如 '384wf'），只用于把
        "目录里有别的 .pth" 提升成 renamed 判断；取不到就退回后缀启发式。
        """
        p = Path(filename)
        res = _lev.LoadResult(module=self.name or type(self).__name__,
                              path=os.fspath(p))
        self.last_load_result = res

        filepath = p
        if not filepath.exists():
            alt = None
            if filepath.suffix == '.pth':
                alt = filepath.with_suffix('.npy')
            elif filepath.suffix == '.npy':
                alt = filepath.with_suffix('.pth')
            if alt is not None and alt.exists():
                filepath = alt
                res.fallback_path = os.fspath(alt)

        if not filepath.exists():
            cands = self._find_renamed_candidates(p, model_prefix)
            if cands:
                res.reason = _lev.REASON_RENAMED
                res.candidates = cands
                res.detail = ("期望 %s，目录里有 %d 个相似文件：%s"
                              % (p.name, len(cands), ", ".join(cands[:4])))
            else:
                res.reason = _lev.REASON_MISSING
                res.detail = "文件不存在：%s" % os.fspath(p)
            return res

        res.path = os.fspath(filepath)

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
                                d[key] = np.load(buf, allow_pickle=False)
            except Exception as e:
                res.reason = _lev.REASON_CORRUPT
                res.detail = ("zip+npy 读取失败：%s: %s" % (type(e).__name__, e))
                print("[WARN] Failed to load zip+npy weights from '%s': %s."
                      % (os.fspath(filepath), e))
                return res
            return self._load_pt_weights(d, res)

        # Legacy pickle format。
        # 注意 'unknown' 也要走这里：DFL 4.0 的 TF 权重虽然叫 .npy，内容其实是
        # pickle 出来的 dict（键里带 ':0'）。旧实现也是"非 zip 就当 pickle 试"，
        # 这里保持不变 —— 只是失败时的原因不再是空白，而是 corrupt + 文件头。
        del fmt
        try:
            with open(filepath, 'rb') as f:
                d = pickle.load(f)
        except Exception as e:
            try:
                with open(filepath, 'rb') as f:
                    head = f.read(32)
            except Exception:
                head = b''
            res.reason = _lev.REASON_CORRUPT
            res.detail = "pickle 读取失败：%s: %s（文件头 %r）" % (type(e).__name__, e, head[:16])
            print(
                f"[WARN] Failed to load weights from '{os.fspath(filepath)}': {e}. "
                f"This file does not look like a pickle weights file. "
                f"Header bytes: {head!r}. "
                f"You may need to delete/replace the corrupted or wrong-format weights file."
            )
            return res

        if not isinstance(d, dict):
            res.reason = _lev.REASON_CORRUPT
            res.detail = "反序列化结果不是 dict，而是 %s" % type(d).__name__
            return res

        # Detect original DFL (TensorFlow) format: keys contain TF scope names
        if any(isinstance(k, str) and ':0' in k for k in d.keys()):
            if self._load_tf_weights(d):
                res.ok = True
                res.reason = _lev.REASON_OK
                return res
            res.reason = _lev.REASON_CORRUPT
            res.detail = "TF(.npy) 权重无法唯一映射到当前网络（见上面的 [WARN]）"
            return res

        return self._load_pt_weights(d, res)

    def _weight_owner_names(self, n):
        """返回与 get_weights() 等长的“这个参数属于哪一层”名字列表。"""
        cached = getattr(self, '_weight_owner_cache', None)
        if cached is not None and len(cached) == n:
            return cached
        out = []
        layers = getattr(self, 'layers', None)
        if isinstance(layers, list):
            for layer in layers:
                try:
                    cnt = len(layer.get_weights())
                except Exception:
                    cnt = 0
                nm = getattr(layer, 'name', None) or type(layer).__name__
                out.extend([nm] * cnt)
        fallback = self.name or type(self).__name__
        while len(out) < n:
            out.append(fallback)
        self._weight_owner_cache = out
        return out

    def _module_token(self, stem, model_prefix=None):
        """把文件名糖化成“模块令牌”：384wf_decoder_src -> decoder_src。"""
        s = str(stem)
        if model_prefix and s.startswith(model_prefix + '_'):
            s = s[len(model_prefix) + 1:]
        elif '_' in s:
            # 没有前缀时只能猜：剥掉第一段（模型名通常不带下划线）
            s = s.split('_', 1)[1]
        return s.lower()

    def _find_renamed_candidates(self, filepath, model_prefix=None):
        """文件不在预期位置时，找找目录里“像它”的 .pth/.npy（改名/换了模型名）。

        只做**诊断**用：命中就报 renamed（比一句“文件不存在”有用得多），
        不猜、不自动加载 —— 名字对不上时硬加载是以前踩过的坑。
        """
        parent = filepath.parent
        try:
            if not parent.is_dir():
                return []
        except Exception:
            return []
        want = self._module_token(filepath.stem, model_prefix)
        if not want:
            return []
        out = []
        try:
            names = sorted([q.name for q in parent.iterdir()
                            if q.is_file() and q.suffix in ('.pth', '.npy')])
        except Exception:
            return []
        for name in names:
            if name == filepath.name:
                continue
            tok = self._module_token(Path(name).stem, model_prefix)
            if not tok:
                continue
            if tok == want or tok.startswith(want) or want.startswith(tok):
                out.append(name)
        return out

    def _load_pt_weights(self, d, res=None):
        """Load current-format weights (param_0, param_1, ...).

        先全量校验（个数/形状），全部通过才写参数；任何一项不过就一个字节都不写。
        """
        weights = self.get_weights()

        if self.name is None:
            raise Exception("name must be defined.")

        if res is None:
            res = _lev.LoadResult(module=self.name)
        owners = self._weight_owner_names(len(weights))
        _MISSING = object()

        entries = []
        problems = []
        found_shapes = []
        for i, w in enumerate(weights):
            w_name = "param_%d" % i
            exp_shape = tuple(int(x) for x in w.shape)
            w_val = d.get(w_name, _MISSING)

            if w_val is _MISSING:
                problems.append((owners[i], i, exp_shape, None))
                found_shapes.append([])
                continue

            try:
                val = np.asarray(w_val)
            except Exception as e:
                problems.append((owners[i], i, exp_shape, None))
                found_shapes.append([])
                res.detail = "%s: ndarray 化失败 %s" % (w_name, e)
                continue

            fnd_shape = tuple(int(x) for x in val.shape)
            found_shapes.append(list(fnd_shape))

            # 元素个数必须一致。形状不同但元素数相同（(1,64) vs (64,)）允许 reshape，
            # 这是历史存档的常态；元素数不同就是真的对不上。
            if int(val.size) != int(w.numel()):
                problems.append((owners[i], i, exp_shape, fnd_shape))
                continue

            try:
                # ★ 不能用 ascontiguousarray 一把梭：它对 0 维数组会返回 (1,)，
                #   而 iterations / lr_cur 这类标量参数就是 0 维的 —— copy_ 会报
                #   "output with shape [] does not match the broadcast shape [1]"。
                arr = val.reshape(exp_shape)
                if arr.ndim > 0 and not arr.flags["C_CONTIGUOUS"]:
                    arr = np.ascontiguousarray(arr)
                src = torch.from_numpy(arr).to(device=w.device, dtype=w.dtype)
            except Exception as e:
                problems.append((owners[i], i, exp_shape, fnd_shape))
                res.detail = "%s: 无法转成张量 %s: %s" % (w_name, type(e).__name__, e)
                continue

            entries.append((w, src))

        res.expected_shapes = [list(tuple(int(x) for x in w.shape)) for w in weights]
        res.found_shapes = found_shapes

        if problems:
            res.ok = False
            res.reason = _lev.REASON_SHAPE
            res.affected = problems
            worst = problems[0]
            res.detail = ("%d/%d 个参数对不上，第一个：%s #%d 期望 %s 实际 %s"
                          % (len(problems), len(weights), worst[0], worst[1],
                             list(worst[2]), list(worst[3]) if worst[3] else "缺失"))
            return res

        # ---- 全部通过，才开始写 ----
        try:
            with torch.no_grad():
                for w, src in entries:
                    w.data.copy_(src)
        except Exception as e:
            res.ok = False
            res.reason = _lev.REASON_CORRUPT
            res.detail = "写入参数时失败：%s: %s" % (type(e).__name__, e)
            return res

        res.ok = True
        res.reason = _lev.REASON_OK
        return res

    def reset_weights(self):
        """把权重**重新初始化**（就地，保持 Parameter 对象身份不变）。

        为什么不能直接再调一次 build_weights()：那会新建 Parameter 对象，
        而优化器（以及各处的 trainable_weights 列表）拿的是旧对象的引用 ——
        换掉对象等于让优化器对着"已经不在网络里的张量"更新，梯度写不回模型。

        做法：先记下每个 (持有者, 属性名, 旧 Parameter)，跑一遍该层自己的
        build_weights()（于是拿到和构造时**完全同一套**初始化分布：
        Conv2D/Dense 的 xavier_uniform_ / wscale 的 normal_(0,1) /
        BatchNorm 的 ones-zeros / TLU 的 zeros …），再把新数值 copy 回旧对象，
        最后把属性重新指回旧对象。

        返回重新初始化的参数个数；没有可重建的东西时返回 0。
        """
        layers = getattr(self, 'layers', None)
        if isinstance(layers, list):
            # ModelBase：权重来自各子层，逐个下放
            total = 0
            for layer in layers:
                try:
                    total += layer.reset_weights()
                except Exception:
                    pass
            return total
        return self._reset_leaf_weights()

    def _reset_leaf_weights(self):
        build = getattr(self, 'build_weights', None)
        if not callable(build):
            return 0
        params = [(a, p) for a, p in list(getattr(self, '_parameters', {}).items())
                  if isinstance(p, torch.nn.Parameter)]
        buffers = [(a, b) for a, b in list(getattr(self, '_buffers', {}).items())
                   if isinstance(b, torch.Tensor)]
        if not params and not buffers:
            return 0
        try:
            build()
        except Exception:
            # 重建失败：保持旧数值不动（回归"什么都没发生"比留下半套新值安全）
            return 0
        n = 0
        with torch.no_grad():
            for attr, old in params:
                new = self._parameters.get(attr)
                if isinstance(new, torch.nn.Parameter) and new is not old:
                    if tuple(new.shape) == tuple(old.shape):
                        old.data.copy_(new.data)
                self._parameters[attr] = old
                n += 1
            for attr, old in buffers:
                new = self._buffers.get(attr)
                if isinstance(new, torch.Tensor) and new is not old:
                    if tuple(new.shape) == tuple(old.shape):
                        old.data.copy_(new.data)
                self._buffers[attr] = old
        return n

    def _load_tf_weights(self, d):
        """Load original DeepFaceLab (TensorFlow) .npy weights.

        NOTE: this is an independent COPY of the matching logic in
        tools/migrate/convert_dfl_tf_to_torch.py (that CLI is not imported here, so the
        two must be kept in sync by hand). It is the code path actually used by
        training auto-convert (load_weights -> ':0' in key) and by the merger
        page (.npy -> DFM via tools/migrate/export_dfm.py convert_npy_to_pth).

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
