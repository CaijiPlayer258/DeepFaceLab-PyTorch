import torch
from core.leras import nn
from core.leras.layers.Saveable import Saveable

class OptimizerBase(Saveable):
    def __init__(self, name=None):
        super().__init__(name=name)
        self.optimizer = None
    
    def torch_clip_norm(self, g, c, n):
        """Clip the gradient `g` if the L2 norm `n` exceeds `c`."""
        if c <= 0:
            return g
        
        if n >= c:
            return g * (c / n)
        return g

nn.OptimizerBase = OptimizerBase
