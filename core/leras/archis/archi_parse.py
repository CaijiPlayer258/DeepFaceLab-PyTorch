"""archi 字符串解析 —— DF / LIAE / DFLite 统一入口。

格式（第二段的 opts 与原本一致，`u`/`d`/`t`/`c`）：

    'df'           -> ('df',   None,  '')
    'liae-ud'      -> ('liae', 'ud',  '')
    'df-udt-l'     -> ('df',   'udt', 'l')      DFLite，默认 dw5（DW5x5+PW1x1，不可折）
    'df-udt-l5'    -> ('df',   'udt', 'l5')     DFLite，repdw5（DW5x5+identity 可折叠加块）
    'df-udt-l3'    -> ('df',   'udt', 'l3')     DFLite，稠密 3x3（RepBlock，可折）
    'df-udt-lx'    -> ('df',   'udt', 'lx')     DFLite，dw5 + 瓶颈 stride 投影
    'df-udt-l5x'   -> ('df',   'udt', 'l5x')    DFLite，repdw5 + stride
    'df-udt-l3x'   -> ('df',   'udt', 'l3x')    DFLite，3x3 + stride

第三段是可选后缀，**不放进 opts** —— 否则 lite 里的 `t` 会被当成「编码器多下一次采样」，
`l` 也不是合法单字符选项。这样旧模型文件（两段式）与旧 CLI 输入完全不受影响。
"""

_ARCHI_BASE_TYPES = ('df', 'liae')
_ARCHI_OPTS_CHARS = ('u', 'd', 't', 'c')
_ARCHI_MODS = ('', 'l', 'l3', 'l5', 'lm', 'lx', 'l3x', 'l5x', 'lmx', 'l35', 'l35x')


def parse_archi_string(archi):
    """返回 (archi_type, archi_opts, archi_mod)；无法解析时返回 None。"""
    if archi is None:
        return None
    parts = str(archi).lower().split('-')
    if len(parts) == 1:
        return (parts[0], None, '') if parts[0] in _ARCHI_BASE_TYPES else None
    if len(parts) == 2:
        base, opts = parts
        mod = ''
    elif len(parts) == 3:
        base, opts, mod = parts
    else:
        return None

    if base not in _ARCHI_BASE_TYPES:
        return None
    if mod not in _ARCHI_MODS:
        return None
    if opts == '':
        return None
    if opts is not None and any(ch not in _ARCHI_OPTS_CHARS for ch in opts):
        return None
    return (base, opts, mod)


def archi_is_lite(mod):
    """mod 是否表示 DFLite 架构。"""
    return bool(mod) and 'l' in str(mod)


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
    # import archi_parse 也能用（GUI 侧就是这种情况）。
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
