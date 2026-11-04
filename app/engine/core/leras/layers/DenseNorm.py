import torch
from core.leras import nn

class DenseNorm(nn.LayerBase):
    """
    密集归一化层
    """
    def __init__(self, dense=False, eps=1e-06, dtype=None, name=None, **kwargs):
        self.dense = dense
        # Match TF: epsilon is a dtype-specific constant tensor.
        self.eps = torch.tensor(eps, dtype=dtype if dtype is not None else (nn.floatx if hasattr(nn, 'floatx') else torch.float32))
        if dtype is None:
            dtype = nn.floatx if hasattr(nn, 'floatx') else torch.float32
        self.dtype = dtype

        super().__init__(name=name, **kwargs)

    def __call__(self, x):
        eps = self.eps.to(device=x.device, dtype=x.dtype)
        return x * torch.rsqrt(torch.mean(torch.square(x), dim=-1, keepdim=True) + eps)
        
nn.DenseNorm = DenseNorm