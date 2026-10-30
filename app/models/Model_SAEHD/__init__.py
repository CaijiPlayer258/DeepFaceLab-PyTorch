"""Model_SAEHD package.

Prefer PyTorch training implementation, fallback to legacy TF-style one.
"""

try:
	from .Model_pytorch import Model
except Exception:
	from .Model import Model
