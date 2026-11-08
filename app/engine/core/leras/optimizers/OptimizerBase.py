import torch
from core.leras import nn
from core.leras.layers.Saveable import Saveable

class OptimizerBase(Saveable):
    def __init__(self, name=None):
        super().__init__(name=name)
        self.optimizer = None
    
    def reset_state(self):
        """把优化器状态清空到"刚构造"的样子。

        权重加载失败、被重新初始化时**必须**一起调用 —— 只重置权重、
        动量/二阶矩还是上一份存档的话，前几步的更新方向是"按旧参数尺度"算出来的，
        实际效果是拿新初始化的权重接着跑旧轨迹，比从头训练更糟。

        清空的内容：ms/vs（或 accus）、iterations、lr_cur（回到初始 lr）、
        平台期检测的历史与基准、best_loss。
        """
        params = list(getattr(self, 'params', []) or [])
        if params:
            # 各优化器的 initialize_variables 会按当前参数重建 ms/vs
            self.initialize_variables(params)
        it = getattr(self, 'iterations', None)
        if it is not None:
            try:
                it.fill_(0)
            except Exception:
                self.iterations = torch.tensor(0, dtype=torch.int64)
        lr_cur = getattr(self, 'lr_cur', None)
        if lr_cur is not None:
            try:
                lr_cur.fill_(float(getattr(self, 'lr', 0.0)))
            except Exception:
                pass
        self._loss_hist = []
        self._lr_last_check = 0
        self._lr_best_base = None
        self._best_loss = {}
        return {'params': len(params),
                'buffers': len(getattr(self, 'ms', []) or []) + len(getattr(self, 'vs', []) or [])}

    def torch_clip_norm(self, g, c, n):
        """Clip the gradient `g` if the L2 norm `n` exceeds `c`."""
        if c <= 0:
            return g
        
        if n >= c:
            return g * (c / n)
        return g

nn.OptimizerBase = OptimizerBase
