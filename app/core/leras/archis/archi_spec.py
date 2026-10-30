"""向后兼容转发层。

真正的实现已移到 **core/archi_names.py**（不 import torch）。
本模块保留是为了让 `from core.leras.archis.archi_spec import ...` 继续可用。

⚠️ GUI 代码请直接 `from core.archi_names import ...`。
走本模块会拉起 `core/leras/__init__.py` -> `nn.py` -> `import torch`，
在 Qt 之后 import torch 会在 Windows 上炸 WinError 1114（c10.dll）。
"""

from core.archi_names import (  # noqa: F401
    ARCHI_CHOICES, ARCHI_DESCRIPTIONS,
    SUBARCHI_CHOICES, SUBARCHI_DESCRIPTIONS,
    KERNEL_CHOICES,
    DEFAULT_KERNEL_LABEL, default_kernel_label,
    kernel_mod_for, kernel_desc_for, archi_type_for, is_lite_display,
    build_archi_string,
    parse_archi_string, archi_is_lite,
    _ARCHI_MODS,
)
