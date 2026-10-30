import torch
import torch.nn as torch_nn

from core.leras.nn import nn

class LayerBase(torch_nn.Module):
    def __init__(self, name=None, **kwargs):
        super().__init__()
        self.name = name  # 层的名称，用于在ModelBase中注册
        self.build_weights()
    
    #override
    def build_weights(self):
        pass
    
    #override
    def forward(self, *args, **kwargs):
        pass
    
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
    
    def get_weights(self):
        return list(self.parameters())
    
    def get_weights_np(self):
        return [w.detach().cpu().numpy() for w in self.get_weights()]
    
    def set_weights(self, new_weights):
        weights = self.get_weights()
        if len(weights) != len(new_weights):
            raise ValueError('len of lists mismatch')
        
        for w, new_w in zip(weights, new_weights):
            if isinstance(new_w, (torch_nn.Parameter, torch.Tensor)):
                src = new_w.data if hasattr(new_w, 'data') else new_w
                w.data.copy_(src.to(device=w.device, dtype=w.dtype))
            else:
                import numpy as np
                if isinstance(new_w, np.ndarray):
                    src = torch.from_numpy(new_w).reshape(w.shape)
                else:
                    src = torch.tensor(new_w).reshape(w.shape)
                w.data.copy_(src.to(device=w.device, dtype=w.dtype))


nn.LayerBase = LayerBase
