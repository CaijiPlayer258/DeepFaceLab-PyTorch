"""Model_XSeg package entry.

Prefer PyTorch implementation.
"""

try:
	from .Model_pytorch import Model
except Exception:  # pragma: no cover
	from .Model import Model
