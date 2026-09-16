"""archi 字符串解析（后端侧）。

纯字符串逻辑已移到 **core/archi_names.py**（那里不 import torch）。
原因见该模块的 docstring：GUI 在 QApplication 之后 import torch 会触发
Windows `WinError 1114 / c10.dll`，所以 GUI 不许碰 `core.leras`。

这里只保留需要 `core.leras`（进而需要 torch）的 `build_archi_classes`。
老代码 `from core.leras.archis.archi_parse import parse_archi_string` 继续可用（转发）。
"""

from core.archi_names import (
    ARCHI_BASE_TYPES as _ARCHI_BASE_TYPES,
    ARCHI_OPTS_CHARS as _ARCHI_OPTS_CHARS,
    ARCHI_MODS as _ARCHI_MODS,
    parse_archi_string,
    archi_is_lite,
)

__all__ = ['parse_archi_string', 'archi_is_lite', 'build_archi_classes',
           '_ARCHI_BASE_TYPES', '_ARCHI_OPTS_CHARS', '_ARCHI_MODS']


def build_archi_classes(mod, mod_kwargs=None):
    """按修饰符返回 (ArchiClass, 额外 kwargs)。

    mod 为空 -> 原版 DeepFakeArchi（行为与改动前完全一致）。
    mod 含 'l' -> DFLiteArchi，后缀字母的含义：
        '3' -> kernel='3x3'    稠密 3x3 + RepBlock（可折，感受野 3x3）
        '5' -> kernel='repdw5' DW5x5+identity 可折叠加块（可折，感受野 5x5）
        无   -> kernel='dw5'   DW5x5+PW1x1（不可折，参数最少）
        'x' -> inter_stride=True，瓶颈用 stride 投影
    mod_kwargs 用于覆盖默认（如把 GUI 里的 ae_dims 传进去）。
    """
    from core.leras import nn as _nn
    # DFLiteArchi 用 nn.LayerBase / nn.ModelBase，这两个要在 leras.layers /
    # leras.models 被导入后才挂到 nn 命名空间上。这里自举一次，保证单独
    # import archi_parse 也能用（CLI 侧就是这种情况）。
    import core.leras.layers  # noqa: F401
    import core.leras.models.ModelBase  # noqa: F401
    if not mod:
        if not hasattr(_nn, 'DeepFakeArchi'):
            import core.leras.archis.DeepFakeArchi  # noqa: F401
        return _nn.DeepFakeArchi, {}
    from core.leras.archis.DFLiteArchi import DFLiteArchi
    mod = str(mod)
    if '3' in mod:
        kernel = '3x3'          # 稠密 3x3（可折）
    elif 'm' in mod:
        kernel = 'ms5'          # DW5x5 + DW3x3 多尺度可折块
    elif '5' in mod:
        kernel = 'repdw5'       # 稠密 3x3 下采样 + DW5x5 可折块
    else:
        kernel = 'dw5'          # DW5x5 + PW1x1（当前最优）
    kwargs = {
        'kernel': kernel,
        'inter_stride': ('x' in mod),
    }
    if mod_kwargs:
        kwargs.update(mod_kwargs)
    return DFLiteArchi, kwargs
