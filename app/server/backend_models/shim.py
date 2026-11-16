# -*- coding: utf-8 -*-
"""numexpr 的极简 numpy 替身（只覆盖 4.0 色彩迁移真正用到的那一点）。

为什么要这个：4.0 的 core/imagelib/color_transfer.py 顶上就 `import numexpr as ne`，
而 4.1 的 server python（runtime/server/Scripts/python.exe）里没装 numexpr —— 整个
模块 import 不进来，rct（reinhard）这条色彩迁移就直接没了。numexpr 在那边只出现在
reinhard_color_transfer 的三行：

    target_l = ne.evaluate('(target_l - target_l_mean) * source_l_std / target_l_std + target_l_mean')

都是单层四则运算，numpy 逐元素算出来是同一个结果（float32 输入，numpy 会按 float32
算；numexpr 内部用 double，末位可能差 1 LSB —— 远小于验收的 mean ≤ 2）。

只实现 evaluate()：其它属性访问一律 AttributeError，免得有人以为这里有全套 numexpr。
"""
from __future__ import annotations

import numpy as np  # noqa: F401  （表达式里会用到的名字由 local_dict 给）


def evaluate(expr, local_dict=None, global_dict=None, **kwargs):
    """把表达式按 numpy 求值。只支持纯算术（color_transfer 那三行的形状）。"""
    env = dict(local_dict or {})
    if global_dict:
        env.update(global_dict)
    return eval(str(expr), {"__builtins__": {}}, env)      # noqa: S307 - 表达式是内部写死的


def set_num_threads(n):                                     # 兼容调用方
    return None


def get_num_threads():
    return 1


def __getattr__(name):                                       # pragma: no cover
    raise AttributeError("numexpr shim 只实现 evaluate(): %r" % (name,))
