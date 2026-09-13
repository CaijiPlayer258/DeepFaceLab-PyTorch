import math
from typing import Iterable, List, Optional

import numpy as np
import torch

from core.leras import nn


def _torch_to_numpy_dtype(dtype: torch.dtype):
    if dtype == torch.float16:
        return np.float16
    if dtype == torch.float32:
        return np.float32
    if dtype == torch.float64:
        return np.float64
    return np.float32


class AdaBelief(nn.OptimizerBase):
    """DeepFaceLab AdaBelief implementation (TF-equivalent math)."""

    def __init__(
        self,
        trainable_weights: Optional[Iterable] = None,
        lr: float = 0.001,
        beta_1: float = 0.9,
        beta_2: float = 0.999,
        lr_dropout: float = 1.0,
        lr_cos: int = 0,
        lr_total_steps: int = 0,
        lr_plateau_factor: float = 1.0,
        lr_plateau_patience: int = 0,
        lr_plateau_delta: float = 0.0,
        lr_plateau_min_ratio: float = 0.1,
        lr_plateau_partitions: int = 1,
        clipnorm: float = 0.0,
        name: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(name=name)

        if name is None:
            self.name = self.__class__.__name__

        self.lr = float(lr)
        self.beta_1 = float(beta_1)
        self.beta_2 = float(beta_2)
        self.lr_dropout = float(lr_dropout)
        self.lr_cos = int(lr_cos)
        self.lr_total_steps = int(lr_total_steps)
        # ---- 平台期自动降 lr ----
        # factor >= 1.0 视为关闭。patience<=0 时按 lr_cos 自动推导。
        # lr_cur 用 0 维 Parameter 承载（requires_grad=False）：
        # 优化器存档走 get_weights() 的位置索引，普通 float 不会进存档，
        # 那样平台期降下来的 lr 在断点恢复后就丢了、退回初始值。
        # 用 Parameter 还有个好处：它单独占一个位置，不会影响到别的张量。
        self.lr_cur = torch.nn.Parameter(
            torch.tensor(float(lr), dtype=torch.float32), requires_grad=False)
        self._lr_factor = float(lr_plateau_factor)
        self._lr_patience = int(lr_plateau_patience or (self.lr_cos if self.lr_cos > 0 else 2000))
        self._lr_delta = float(lr_plateau_delta)
        self._lr_min_ratio = float(lr_plateau_min_ratio)
        self._lr_partitions = int(lr_plateau_partitions or 1)
        self._loss_hist = []
        self._lr_last_check = 0      # 上次平台判定的 iter（冷却用）
        self._lr_best_base = None    # 判定基准：只前移不回退
        self._best_loss = {}
        self.clipnorm = float(clipnorm)

        self.iterations = torch.tensor(0, dtype=torch.int64)

        self.params: List[torch.nn.Parameter] = []
        self.ms: List[torch.Tensor] = []
        self.vs: List[torch.Tensor] = []
        self.lr_masks: List[Optional[torch.Tensor]] = []

        if trainable_weights is not None:
            self.initialize_variables(trainable_weights)

    def get_weights(self):
        # 顺序必须是 [iterations, lr_cur, ms..., vs..., best_loss..., loss_hist...]
        # 且 get_weights() 全程稳定 —— initialize_variables 要跳过
        # lr_cur（第 1 位）之后新增的那几个标量。
        out = [self.iterations, self.lr_cur] + self.ms + self.vs
        bl = getattr(self, '_best_loss', None) or {}
        for k in sorted(bl.keys()):
            out.append(torch.nn.Parameter(
                torch.tensor(float(bl[k]), dtype=torch.float32), requires_grad=False))
        lh = getattr(self, '_loss_hist', None) or []
        if lh:
            out.append(torch.nn.Parameter(
                torch.tensor([float(v) for v in lh], dtype=torch.float32), requires_grad=False))
        return out

    def initialize_variables(self, trainable_weights, vars_on_cpu=True, lr_dropout_on_cpu=False):
        params: List[torch.nn.Parameter] = []
        # 注意：这里收到的是**真实参数列表**（构造时由 Model 传进来），
        # 不是 get_weights() 的输出。所以绝不能在这里切片跳过前两项 ——
        # 那会把前两个真实参数整个丢掉，导致它们静默不更新。
        # 恢复调度器状态由 _restore_sched_state() 单独负责。
        for item in trainable_weights:
            if isinstance(item, (list, tuple)):
                for p in item:
                    if isinstance(p, torch.nn.Parameter):
                        params.append(p)
            elif isinstance(item, torch.nn.Parameter):
                params.append(item)
            elif hasattr(item, 'parameters'):
                params.extend([p for p in item.parameters() if isinstance(p, torch.nn.Parameter)])

        self.params = params
        self.ms = [torch.zeros_like(p, device=p.device, dtype=p.dtype) for p in self.params]
        self.vs = [torch.zeros_like(p, device=p.device, dtype=p.dtype) for p in self.params]

        # lr_dropout 的掩码**不在这里生成**。
        # 原来只在初始化时抽一次、之后每步复用，等于「训练全程固定冻结同一批参数」，
        # 而不是「每步随机丢一部分」。已挪到 step() 里每步重采样。
        # 附带好处：掩码不再是有状态量，也就不需要进存档 ——
        # 之前它不在 get_weights() 里，断点恢复会重新抽样、静默换掉冻结集合。
        self.lr_masks = [None for _ in self.params]

    def _get_plateau_lr(self) -> float:
        """平台期自动降 lr 后的实际学习率。

        判定方式：每积累满 _lr_patience 步，检查这段窗口的**最小 loss**
        有没有比"上一次判定时的最小 loss"改善超过 _lr_delta。
        没有 -> 认为进入平台期，lr *= _lr_factor。

        两个必须有的保护（都是实测踩出来的）：
          1. **冷却期**：每次判定后要再等满一个 patience 窗口才允许下次判定。
             否则窗口每步滑动一点，segment 边界跟着变，比较基准不稳，
             会连锁触发 —— 实测出现过连续 4 步把 lr 一路降到下限。
          2. **基准只前移、不回退**：基准取 max(上次基准, 本窗口最优)。
             直接用窗口均值当基准的话，如果 loss 其实在下降，
             下一窗口反而"更优"，会误判成平台 —— 实测把一条持续下降的
             曲线错误地衰减到了下限。
        """
        factor = getattr(self, '_lr_factor', 1.0)
        lo = self.lr * getattr(self, '_lr_min_ratio', 0.0)
        if factor >= 1.0 or float(self.lr_cur) <= lo:
            return float(self.lr_cur)
        pat = int(getattr(self, '_lr_patience', 0) or 0)
        t = int(self.iterations.item())
        if pat <= 1:
            return float(self.lr_cur)

        # 冷却：距上次判定不足一个窗口就不重复判定
        if (t - getattr(self, '_lr_last_check', 0)) < pat:
            return float(self.lr_cur)
        recent = self._loss_hist[-pat:]
        if len(recent) < pat:
            return float(self.lr_cur)      # 历史还没攒够一个完整窗口，别下结论

        cur_best = min(recent)
        base = getattr(self, '_lr_best_base', None)
        self._lr_last_check = t
        if base is None:
            self._lr_best_base = cur_best
            return float(self.lr_cur)
        delta = getattr(self, '_lr_delta', 0.0)
        if cur_best < base - delta:
            # 还在改善：基准前移到更优值（只前进、不回退）
            self._lr_best_base = cur_best
            return float(self.lr_cur)

        # 平台期
        with torch.no_grad():
            self.lr_cur.fill_(max(lo, float(self.lr_cur) * factor))
        self._lr_best_base = cur_best
        return float(self.lr_cur)

    def observe_loss(self, loss):
        """训练每一步把 loss 喂进来，供平台期检测用。

        调用时机：**在 optimizer.step() 之前**。这样 step() 里的
        "刚好积累到 patience 步就用当前窗口判断" 是对齐的。
        """
        if self._lr_factor >= 1.0:
            return
        v = float(loss)
        if v != v or v == float('inf'):     # NaN / inf 不参与判断
            return
        self._loss_hist.append(v)
        cap = int(getattr(self, '_lr_patience', 0) or 0) + 1
        if cap > 1 and len(self._loss_hist) > cap:
            del self._loss_hist[:-cap]

    def get_lr(self) -> float:
        """当前实际生效的学习率（含余弦退火 / 平台期衰减 / lr_dropout 之外的调度）。

        训练日志打印用它 —— 因为调度器会让 lr 随时变化，
        只看起始的 lr 无法判断"现在到底是多少"。
        """
        try:
            return float(self._get_lr_value())
        except Exception:
            return float(getattr(self, 'lr', 0.0))

    def _get_lr_value(self) -> float:
        base = self._get_plateau_lr()
        t = float(int(self.iterations.item()))
        # 两种余弦，别混：
        #   余弦退火（单次，不重启）：相位 0 -> pi，lr 从 base 单调降到 ~0，只走一遍
        #   余弦退火·热重启（SGDR）：每 lr_cos 步把 lr 弹回峰值重开一轮
        # 前者优先。lr_total_steps 是"退火跑完整个下降过程"所用的总步数。
        # 注意它必须大致等于你打算跑的总步数：
        #   设准 -> 跑完时 lr 正好接近 0（没收敛就卡死）
        #   设太远 -> 余弦在 0 附近极平坦，等于几乎没退火（5 倍余量时跑完只降到 90%）
        if getattr(self, 'lr_total_steps', 0) > 0:
            prog = min(1.0, t / float(self.lr_total_steps))
            base *= (math.cos(math.pi * prog) + 1.0) / 2.0
        elif self.lr_cos != 0:
            base *= (math.cos(t * (2 * math.pi / float(self.lr_cos))) + 1.0) / 2.0
        return base

    def load_weights(self, filename):
        ok = super().load_weights(filename)
        if ok:
            self._restore_sched_state()
        return ok

    def _restore_sched_state(self):
        """load_weights 是按 param_i 位置存的，调度器状态就写在
        ms/vs 之后，这里按同样顺序读回来。"""
        try:
            w = self.get_weights()
        except Exception:
            return
        # get_weights 布局 = [iterations, lr_cur] + 动量缓冲区 + 调度器状态
        # 动量缓冲区个数按各优化器实现决定（AdaBelief/Adam 是 2 倍，
        # Lion/RMSprop 只有 1 倍），所以用 lr_cur 的位置往后推。
        n_buf = len(w) - 2
        extra = []
        for x in w[2:]:
            if x is self.lr_cur or x is self.iterations:
                continue
            extra.append(x)
        # 前 len(self.ms)+len(getattr(self,'vs',[])) 个是动量，剩下的才是调度器状态
        n_mom = len(self.ms) + len(getattr(self, 'vs', []) or [])
        extra = extra[n_mom:]
        if not extra:
            return
        bl = getattr(self, '_best_loss', None) or {}
        keys = sorted(bl.keys())
        for i, k in enumerate(keys):
            if i >= len(extra):
                break
            bl[k] = float(extra[i].detach().cpu())
        self._best_loss = bl
        rest = extra[len(keys):]
        if rest and rest[0].numel() > 1:
            self._loss_hist = [float(v) for v in rest[0].detach().cpu().tolist()]

    def zero_grad(self):
        for p in self.params:
            if p.grad is None:
                continue
            p.grad.detach_()
            p.grad.zero_()

    def step(self):
        if len(self.params) == 0:
            return

        self.iterations += 1
        lr = self._get_lr_value()

        scale = 1.0
        if self.clipnorm > 0.0:
            sq_sum = None
            for p in self.params:
                if p.grad is None:
                    continue
                g = p.grad
                v = (g.detach().float() ** 2).sum()
                sq_sum = v if sq_sum is None else (sq_sum + v)
            if sq_sum is not None:
                norm = torch.sqrt(sq_sum)
                n = float(norm.item())
                if n >= self.clipnorm and n > 0.0:
                    scale = float(self.clipnorm) / n

        b1 = self.beta_1
        b2 = self.beta_2

        with torch.no_grad():
            for i, p in enumerate(self.params):
                if p.grad is None:
                    continue

                g = p.grad
                if scale != 1.0:
                    g = g * scale

                m = self.ms[i]
                v = self.vs[i]

                m_t = b1 * m + (1.0 - b1) * g
                v_t = b2 * v + (1.0 - b2) * (g - m_t).pow(2)

                resolution = float(np.finfo(_torch_to_numpy_dtype(p.dtype)).resolution)
                step_delta = (-lr) * m_t / (torch.sqrt(v_t) + resolution)

                # lr_dropout：每步重新独立采样（原语义是「每次更新丢弃一部分参数」）。
                # 这是在**反传之后**对更新量做掩码 —— 前向和反向看到的都是完整模型，
                # 所以它不构成 dropout 那种正则化，更接近「随机减小有效学习率」。
                if self.lr_dropout != 1.0:
                    mask = torch.bernoulli(
                        torch.full_like(p, self.lr_dropout, dtype=p.dtype, device=p.device))
                    step_delta = step_delta * mask

                p.add_(step_delta)
                self.ms[i] = m_t
                self.vs[i] = v_t


nn.AdaBelief = AdaBelief

