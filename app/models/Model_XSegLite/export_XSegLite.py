"""XSegLite: export .pth weights to ONNX."""
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import warnings
warnings.filterwarnings('ignore', category=UserWarning, message=".*dynamic_axes.*")

import torch
from core.xseglite_torch import XSegLiteTorch

model_dir = REPO / 'workspace' / 'model' / 'XSegLite'
out_path = model_dir / 'xseglite.onnx'

pth_files = list(model_dir.glob('*.pth'))
if not pth_files:
    print(f'[ERROR] No .pth weights found in {model_dir}')
    sys.exit(1)

model = XSegLiteTorch(3, 32).eval()
ckpt = torch.load(str(pth_files[0]), map_location='cpu')
if 'model' in ckpt:
    model.load_state_dict(ckpt['model'])
else:
    model.load_state_dict(ckpt)
print(f'Loaded: {pth_files[0].name}')

dummy = torch.randn(1, 3, 256, 256)
torch.onnx.export(model, dummy, str(out_path),
    input_names=['input'],
    output_names=['logits', 'pred'],
    dynamic_axes={'input': {0: 'batch'}},
    opset_version=18,
    dynamo=False,
)
if out_path.exists():
    size = out_path.stat().st_size / 1024 / 1024
    print(f'Exported: {out_path.name} ({size:.0f}MB)')
else:
    print('[ERROR] Export failed')
    sys.exit(1)
print('Done!')
print('提示: 已导出 ONNX，直接使用即可')
