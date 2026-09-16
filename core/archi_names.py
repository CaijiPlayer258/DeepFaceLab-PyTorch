"""archi 字符串的纯逻辑 —— **不 import torch**。

为什么单独拆一个模块：

    GUI 启动时（ui/start.py）先构造 QApplication（第 102-106 行 import PyQt5），
    之后第 115 行才 import 页面模块。页面模块只要碰 `core.leras` 一次，
    `core/leras/__init__.py` 就会 `from .nn import nn`，而 nn.py 第 13 行是 `import torch`。
    在 Qt 已经加载过 MSVC runtime / OpenMP 之后再去 import torch，Windows 上会直接炸：

        OSError: [WinError 1114] 动态链接库(DLL)初始化例程失败。
        Error loading "...torch\\lib\\c10.dll" or one of its dependencies.

    所以 archi 的**字符串解析/拼装逻辑必须放在 core.leras 包之外**，
    GUI 才能在不碰 torch 的前提下拿到它。

    `core.leras.archis.archi_parse`（后端，含 build_archi_classes）与
    `core.leras.archis.archi_spec`（GUI 老路径）都保留为转发层，老代码不受影响。

格式（第二段 opts 与原本一致，`u`/`d`/`t`/`c`）：:

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

ARCHI_BASE_TYPES = ('df', 'liae')
ARCHI_OPTS_CHARS = ('u', 'd', 't', 'c')
ARCHI_MODS = ('', 'l', 'l3', 'l5', 'lm', 'lx', 'l3x', 'l5x', 'lmx', 'l35', 'l35x')


def parse_archi_string(archi):
    """返回 (archi_type, archi_opts, archi_mod)；无法解析时返回 None。"""
    if archi is None:
        return None
    parts = str(archi).lower().split('-')
    if len(parts) == 1:
        return (parts[0], None, '') if parts[0] in ARCHI_BASE_TYPES else None
    if len(parts) == 2:
        base, opts = parts
        mod = ''
    elif len(parts) == 3:
        base, opts, mod = parts
    else:
        return None

    if base not in ARCHI_BASE_TYPES:
        return None
    if mod not in ARCHI_MODS:
        return None
    if opts == '':
        return None
    if opts is not None and any(ch not in ARCHI_OPTS_CHARS for ch in opts):
        return None
    return (base, opts, mod)


def archi_is_lite(mod):
    """mod 是否表示 DFLite 架构。"""
    return bool(mod) and 'l' in str(mod)


# ============================================================================
# GUI 侧选项表（原先在 core/leras/archis/archi_spec.py）
# ============================================================================

# 架构下拉：显示名 -> 实际 archi_type
ARCHI_CHOICES = (
    ('DF', 'df', False),
    ('LIAE', 'liae', False),
    ('AMP', 'amp', False),
    ('DF Lite', 'df', True),
    ('LIAE Lite', 'liae', True),
)

ARCHI_DESCRIPTIONS = {
    'DF': '编码器-inter-双解码器架构',
    'LIAE': '编码器-双inter-解码器架构',
    'AMP': '编码器-双inter(身份分离)-解码器架构',
    'DF Lite': 'DF 接口不变，换掉 5×5 卷积的轻量版（见下方「算子」）',
    'LIAE Lite': 'LIAE 接口不变，换掉 5×5 卷积的轻量版（见下方「算子」）',
}

# 子分支下拉
SUBARCHI_CHOICES = ('-u', '-ud', '-ut', '-udt', '-d', '-dt', '-t')

SUBARCHI_DESCRIPTIONS = {
    '-u': '像素进行归一化处理',
    '-d': '提供一种可学习的上采样',
    '-t': '编码器增加一次下采样',
    '-ud': '像素归一化 + 可学习上采样',
    '-ut': '像素归一化 + 编码器增加下采样',
    '-udt': '像素归一化 + 可学习上采样 + 编码器增加下采样 ',
    '-dt': '可学习上采样 + 编码器增加@下采样',
}

# 算子下拉：显示名 -> (修饰符, 描述)
KERNEL_CHOICES = (
    ('original - 原版（5×5 稠密，与 DF 完全一致）', '', '原版 5×5 稠密卷积'),
    ('dw5 - DW5×5+PW1×1 / Rep3×3（默认，最快）', 'l',
     'DW5x5+PW1x1 保感受野 + Rep3x3 可折叠（推荐）'),
    ('dw5+stride - 额外压缩瓶颈空间投影（参数最少）', 'lx',
     '在 dw5 基础上再压缩瓶颈空间投影，参数量最少'),
    ('3x3 - 稠密 3×3 / Rep3×3（可折叠，感受野 3×3）', 'l3',
     '稠密 3x3 + Rep3x3：可折叠，但感受野掉到 3x3'),
    ('叠加 - DW5×5+identity 可折叠（感受野 5×5）', 'l5',
     'dw5 与 repvgg 叠加：DW5x5 + identity 折成单个 depthwise 5x5，'
     '同时拿到 5x5 感受野和可折叠性'),
    ('多尺度 - DW5×5+DW3×3 双分支可折叠', 'lm',
     '同 groups 的多尺度 depthwise（5x5 与 3x3）折成单个 5x5，'
     '同时覆盖粗细两种尺度'),
)

_KERNEL_BY_LABEL = {label: (mod, desc) for label, mod, desc in KERNEL_CHOICES}

# 每个算子的"关键词"。只走精确匹配的话，标签被改动/带空白/被截断时
# 会**静默**回退到默认 'l'（dw5）——那是个"参数最少但不可折"的选项，
# 用户以为选了 3x3 实际训出来的是 dw5，很难察觉。所以再加一层关键词匹配。
_KERNEL_KEYWORDS = (
    ('dw5+stride', 'lx'),
    ('stride',     'lx'),
    ('3x3',        'l3'),
    ('叠加',        'l5'),
    ('identity',   'l5'),
    ('多尺度',      'lm'),
    ('ms5',        'lm'),
    ('original',   ''),
    ('原版',        ''),
    ('dw5',        'l'),
)

# 默认算子。DFLite 下实测：dw5 训练快 2.25x、参数少 27%、折叠后推理快 1.60x，
# 且保住 5x5 感受野 —— 所以选它当默认；'original' 仍可手动选。
DEFAULT_KERNEL_LABEL = KERNEL_CHOICES[1][0]


def default_kernel_label():
    """新建模型时的默认算子显示名。"""
    return DEFAULT_KERNEL_LABEL


def kernel_mod_for(label):
    """算子下拉的显示名 -> 修饰符（'' / 'l' / 'lx' / 'l3' / 'l5' / 'lm'）。

    先精确匹配；不中再按关键词模糊匹配（顺序敏感，'dw5+stride' 要排在 'dw5' 前）；
    最后才回退到默认 'l'。
    """
    if label in _KERNEL_BY_LABEL:
        return _KERNEL_BY_LABEL[label][0]
    s = str(label or '').strip().lower()
    if not s:
        return 'l'
    for kw, mod in _KERNEL_KEYWORDS:
        if kw in s:
            return mod
    return 'l'


def kernel_desc_for(label):
    return _KERNEL_BY_LABEL.get(label, ('l', ''))[1]


def archi_type_for(display):
    """架构下拉的显示名 -> archi_type（'DF Lite' -> 'df'）。"""
    for name, base, _lite in ARCHI_CHOICES:
        if name == display:
            return base
    return str(display).lower()


def is_lite_display(display):
    for name, _base, lite in ARCHI_CHOICES:
        if name == display:
            return lite
    return False


def build_archi_string(archi_display, subarchi, kernel_label):
    """拼成后端可解析的三段式 archi 字符串，并做合法性断言。

    ('DF', '-udt', 'dw5 - ...')      -> 'df-udt-l'
    ('DF Lite', '-udt', '3x3 - ...') -> 'df-udt-l3'
    ('DF', '-udt', 'original - ...') -> 'df-udt'      （与改动前完全一致）
    """
    base = archi_type_for(archi_display)
    sub = (subarchi or '').strip()
    if sub.startswith('-'):
        sub = sub[1:]
    if not sub:
        return base
    if not is_lite_display(archi_display):
        return base + '-' + sub
    mod = kernel_mod_for(kernel_label)
    if not mod:
        return base + '-' + sub
    # 三段式：base-opts-mod（mod 必须独立成段，不能直接接在 opts 后面）
    out = base + '-' + sub + '-' + mod
    if parse_archi_string(out) is None:
        # 不该发生；回退到不带修饰符的形式，避免把非法 archi 写进 _data.dat
        return base + '-' + sub
    return out


# --- 下划线别名：老代码 import 的是这些名字 -----------------------------------
_ARCHI_BASE_TYPES = ARCHI_BASE_TYPES
_ARCHI_OPTS_CHARS = ARCHI_OPTS_CHARS
_ARCHI_MODS = ARCHI_MODS
