"""GUI 侧 archi 选项的唯一事实来源（纯函数，无 Qt 依赖，可直接单测）。"""

from core.leras.archis.archi_parse import parse_archi_string, archi_is_lite, _ARCHI_MODS

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
