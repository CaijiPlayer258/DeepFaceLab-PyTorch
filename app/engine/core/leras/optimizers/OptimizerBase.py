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


    # ------------------------------------------------------------------
    # 存档布局标记：优化器状态是按**位置**存的，一旦训练时的参数列表变了
    # （LIAE 的 random_warp 开关决定 inter_AB 在不在列表里），位置就会整体错位；
    # inter_AB / inter_B 形状完全相同，形状校验根本发现不了，于是动量被写进
    # 另一条支路 —— 表现就是「开关几次 warp 之后模型越来越糊」。
    # 这里把构造时的布局名字写进旁挂 json，加载时对不上就整份作废（不会串味）。
    # ------------------------------------------------------------------
    layout_tag = None

    def _layout_path(self, filename):
        from pathlib import Path
        return Path(str(filename) + ".layout.json")

    def save_weights(self, filename, force_dtype=None):
        super().save_weights(filename, force_dtype=force_dtype)
        try:
            import json
            Path = __import__("pathlib").Path
            self._layout_path(filename).write_text(
                json.dumps({"tag": self.layout_tag,
                            "n": len(self.get_weights())}, ensure_ascii=False),
                encoding="utf-8")
        except Exception:
            pass

    def layout_matches(self, filename):
        """旁挂缺失 -> True（老存档，没得比）；存在则必须一致。"""
        p = self._layout_path(filename)
        try:
            if not p.is_file():
                return True
            import json
            d = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return True
        if d.get("tag") is None or self.layout_tag is None:
            return True
        return str(d.get("tag")) == str(self.layout_tag)

    def load_weights(self, filename):
        """优化器状态按参数形状存档。分辨率变化后 Inter 的形状对不上，
        以前会整份丢弃（动量、二阶矩、iterations 全没），曲线出现台阶。
        这里改成部分加载：能对上的参数照搬，对不上的保持初始化值。"""
        if not self.layout_matches(filename):
            # 布局变了（典型：LIAE 开关 random_warp 让 inter_AB 进出参数列表）。
            # 位置索引已不可信，宁可清空状态，也不要把动量串到另一条支路。
            from core.leras import load_events as _lev
            res = _lev.LoadResult(module=getattr(self, "name", "optimizer"))
            res.ok = False
            res.reason = _lev.REASON_SHAPE
            res.detail = ("训练时的参数布局与本次不同（warp / freeze 开关变过），"
                          "优化器状态已作废并重置")
            self.last_load_result = res
            return False
        res = self.load_weights_ex(filename, partial=True)
        self.last_load_result = res
        return bool(res)

nn.OptimizerBase = OptimizerBase

