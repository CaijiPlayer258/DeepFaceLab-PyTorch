"""Model_Quick96 package entry.

默认导出PyTorch实现（Model_pytorch.py）。
如果导入失败（例如开发中断），再回退到旧版Model.py。
"""

try:
	from .Model_pytorch import Model
except Exception:  # pragma: no cover
	from .Model import Model
