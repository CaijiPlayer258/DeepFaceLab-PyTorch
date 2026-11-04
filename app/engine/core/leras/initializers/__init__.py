import numpy as np
import torch

from core.leras import nn

from .CA import CAInitializerSubprocessor


def _np_dtype_to_torch(dtype):
    if dtype is None:
        return None
    if isinstance(dtype, torch.dtype):
        return dtype
    try:
        dt = np.dtype(dtype)
    except Exception:
        return None

    if dt == np.float16:
        return torch.float16
    if dt == np.float32:
        return torch.float32
    if dt == np.float64:
        return torch.float64
    return None


class initializers():
    class ca:
        def __call__(self, shape, dtype=None):
            # Matches DFL behavior: initializer returns zeros; CA weights are generated via generate_batch.
            torch_dtype = _np_dtype_to_torch(dtype) or torch.float32
            return torch.zeros(shape, dtype=torch_dtype)

        @staticmethod
        def generate_batch(data_list, eps_std=0.05):
            # list of (shape, np.dtype)
            # eps_std kept for API parity; CAInitializerSubprocessor uses default eps_std internally.
            return CAInitializerSubprocessor(data_list).run()


nn.initializers = initializers
