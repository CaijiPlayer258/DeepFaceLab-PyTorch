from __future__ import annotations

import pickle
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from core.interact import interact as io
from core.leras import nn


class XSegNet(object):
    """PyTorch-only XSeg wrapper with DFL-compatible API.

    This replaces the original TF-based implementation. It keeps the public
    methods used by `mainscripts/XSegUtil.py` and the editor/util pipeline.
    """

    VERSION = 1

    def __init__(
        self,
        name,
        resolution=256,
        load_weights=True,
        weights_file_root=None,
        training=False,
        place_model_on_cpu=False,
        run_on_cpu=False,
        optimizer=None,
        data_format="NCHW",
        raise_on_no_model_files=False,
    ):

        self.resolution = int(resolution)
        self.weights_file_root = Path(weights_file_root) if weights_file_root is not None else Path(__file__).parent
        self.training = bool(training)

        # Ensure leras main env is initialized.
        nn.initialize_main_env()

        # XSeg PyTorch implementation assumes NCHW in places (cat/reshape). Enforce it.
        nn.set_data_format("NCHW")

        torch = nn.torch

        model_name = f"{name}_{self.resolution}"
        self.model_filename_list = []

        # Choose device.
        if run_on_cpu or place_model_on_cpu:
            model_device = torch.device("cpu")
        else:
            model_device = nn.device

        # Build model on the chosen device.
        prev_nn_device = nn.device
        try:
            nn.device = model_device
            self.model = nn.XSeg(3, 32, 1, name=name)
            self.model_weights = self.model.get_weights()

            if self.training:
                if optimizer is None:
                    raise ValueError("Optimizer should be provided for training mode.")
                self.opt = optimizer
                self.opt.initialize_variables(self.model_weights, vars_on_cpu=bool(place_model_on_cpu))
                self.model_filename_list += [[self.opt, f"{model_name}_opt.pth"]]
        finally:
            nn.device = prev_nn_device

            self.model_filename_list += [[self.model, f"{model_name}.pth"]]

        if not self.training:
            self.model.eval()

            def net_run(input_np: np.ndarray) -> np.ndarray:
                x = self._np_to_torch_nchw(input_np, device=model_device)
                with torch.no_grad():
                    _, pred = self.model(x)
                out = pred.detach().to("cpu").numpy()
                return out

            self.net_run = net_run

        # Load / init weights.
        self.initialized = True
        for model, filename in self.model_filename_list:
            do_init = not load_weights

            if not do_init:
                model_file_path = self.weights_file_root / filename

                if model is self.model:
                    ok = self._load_model_weights_compat(model_file_path)
                else:
                    ok = model.load_weights(model_file_path)

                do_init = not ok
                if do_init:
                    if raise_on_no_model_files:
                        raise Exception(f"{model_file_path} 不存在或加载失败。")
                    if not self.training:
                        self.initialized = False
                        break

            if do_init:
                model.init_weights()

    def get_resolution(self):
        return self.resolution

    def flow(self, x, pretrain=False):
        return self.model(x, pretrain=pretrain)

    def get_weights(self):
        return self.model_weights

    def save_weights(self):
        for model, filename in io.progress_bar_generator(self.model_filename_list, "Saving", leave=False):
            model.save_weights(self.weights_file_root / filename)

    def extract(self, input_image: np.ndarray):
        if not self.initialized:
            return 0.5 * np.ones((self.resolution, self.resolution, 1), dtype=np.float32)

        # 缩放输入图片到模型所需分辨率
        h, w = input_image.shape[:2]
        if h != self.resolution or w != self.resolution:
            input_image = cv2.resize(input_image, (self.resolution, self.resolution),
                                     interpolation=cv2.INTER_LINEAR)

        input_shape_len = len(input_image.shape)
        if input_shape_len == 3:
            input_image = input_image[None, ...]

        # net_run returns NCHW mask with C==1.
        result = np.clip(self.net_run(input_image), 0.0, 1.0)
        result[result < 0.1] = 0.0

        # Return HWC for single image (DFL convention for masks in util/editor).
        result = np.transpose(result, (0, 2, 3, 1))

        if input_shape_len == 3:
            result = result[0]

        return result

    def _np_to_torch_nchw(self, arr: np.ndarray, device):
        """Accept HWC or NHWC (float 0..1) and return torch NCHW."""
        torch = nn.torch
        if arr.ndim == 3:
            arr = arr[None, ...]

        # Heuristic: if last dim is 1/3 treat as NHWC, else assume already NCHW.
        if arr.ndim != 4:
            raise ValueError(f"expected 3D/4D image array, got shape {arr.shape}")

        if arr.shape[-1] in (1, 3):
            arr = np.transpose(arr, (0, 3, 1, 2))

        x = torch.from_numpy(arr).to(device=device, dtype=torch.float32)
        return x

    def _load_model_weights_compat(self, filename: Path) -> bool:
        """Load weights from .pth (PyTorch state_dict) or .npy (TF named keys).

        - .pth: load via torch.load + load_state_dict
        - .npy: uses _get_tf_weight_names() to map TF variable names (conv01/conv/weight:0)
          to model parameters, converting layouts (HWIO→OIHW, HWOI→IOHW) as needed.
        """
        filename = Path(filename)
        if not filename.exists():
            alt = None
            if filename.suffix == '.pth':
                alt = filename.with_suffix('.npy')
            elif filename.suffix == '.npy':
                alt = filename.with_suffix('.pth')
            if alt is not None and alt.exists():
                filename = alt
            else:
                return False

        torch = nn.torch

        # --- .pth: standard PyTorch state_dict ---
        if filename.suffix == '.pth':
            try:
                state = torch.load(str(filename), map_location='cpu', weights_only=False)
                if isinstance(state, dict) and 'model' in state:
                    state = state['model']
                missing, unexpected = self.model.load_state_dict(state, strict=False)
                if len(missing) > 0:
                    print(f'  [WARN] XSeg .pth: {len(missing)} missing keys, {len(unexpected)} unexpected')
                return True
            except Exception:
                return False

        # --- .npy: TF-style named keys ---
        try:
            d = pickle.loads(filename.read_bytes())
        except Exception:
            return False

        tf_map = self.model._get_tf_weight_names()
        if not tf_map:
            return False

        loaded_any = False
        with torch.no_grad():
            for tf_name, param_tensor in tf_map:
                w_val = d.get(tf_name, None)
                if w_val is None:
                    continue

                if not isinstance(w_val, np.ndarray):
                    w_val = np.array(w_val)

                target_shape = tuple(param_tensor.shape)
                src = self._convert_weight_to_shape(w_val, target_shape)
                if src is None:
                    # No element-count reshape fallback: two layers can hold the
                    # same number of elements and still be different parameters,
                    # so reinterpreting one as the other silently corrupts the
                    # model. Skip and report instead.
                    print(f'  [WARN] XSeg .npy: {tf_name} shape {w_val.shape} '
                          f'cannot be converted to {target_shape}, skipped')
                    continue

                t = torch.from_numpy(src).to(device=param_tensor.device, dtype=param_tensor.dtype)
                param_tensor.data.copy_(t)
                loaded_any = True

        return loaded_any

    @staticmethod
    def _degenerate_reshape(w: np.ndarray, target_shape: tuple[int, ...]) -> Optional[np.ndarray]:
        """Reshape ONLY across size-1 axes (squeeze / unsqueeze).

        A free element-count reshape is never allowed: two different layers can
        have the same element count, so reshaping would write one parameter into
        another's slot.
        """
        if int(np.prod(w.shape)) != int(np.prod(target_shape)):
            return None
        a = tuple(int(x) for x in w.shape if int(x) != 1)
        b = tuple(int(x) for x in target_shape if int(x) != 1)
        if a != b:
            return None
        return w.reshape(target_shape)

    def _convert_weight_to_shape(self, w: np.ndarray, target_shape: tuple[int, ...]) -> Optional[np.ndarray]:
        if tuple(w.shape) == tuple(target_shape):
            return w

        if w.ndim == 1:
            return self._degenerate_reshape(w, target_shape)

        if w.ndim == 2:
            if w.T.shape == tuple(target_shape):
                return w.T
            return self._degenerate_reshape(w, target_shape)

        if w.ndim == 4:
            # TF layout candidates for this codebase's .npy export:
            #  - HWIO -> OIHW: (k,k,in,out) -> (out,in,k,k)   [primary]
            #  - HWOI -> IOHW: (k,k,out,in) -> (in,out,k,k)
            candidates = [w, np.transpose(w, (3, 2, 0, 1)), np.transpose(w, (2, 3, 0, 1))]
            hits = [c for c in candidates if tuple(c.shape) == tuple(target_shape)]
            if len(hits) == 1:
                return hits[0]
            if len(hits) > 1:
                # Ambiguous permutation (can happen for square in==out kernels):
                # prefer the canonical TF HWIO -> torch OIHW transpose.
                return np.transpose(w, (3, 2, 0, 1)) if tuple(np.transpose(w, (3, 2, 0, 1)).shape) == tuple(target_shape) else None
            return self._degenerate_reshape(w, target_shape)

        return self._degenerate_reshape(w, target_shape)