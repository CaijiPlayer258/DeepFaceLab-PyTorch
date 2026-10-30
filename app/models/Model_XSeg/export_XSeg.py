"""XSeg: export legacy .npy / .pth weights to ONNX + PyTorch .pth."""
import sys
import shutil
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import numpy as np
import torch

from core.leras import nn
from facelib import XSegNet


def main():
    model_dir = REPO / 'workspace' / 'model' / 'XSeg'
    old_dir = REPO / 'workspace' / 'model' / 'old'
    resolution = 256

    # Step 1: find weight files (.pth or legacy .npy)
    pth_files = list(model_dir.glob('XSeg_*.pth'))
    npy_files = sorted(model_dir.glob('XSeg_*.npy'))

    if pth_files:
        print(f'Found .pth weights: {[f.name for f in pth_files]}')
        mode = 'pth'
        weight_file = pth_files[0]
    elif npy_files:
        print(f'Found legacy .npy weights: {[f.name for f in npy_files]}')
        mode = 'npy'
        weight_file = npy_files[0]
    else:
        print(f'[ERROR] No XSeg weights found in {model_dir}')
        print('Expected: XSeg_256.pth or XSeg_256.npy / XSeg_256_opt.npy')
        sys.exit(1)

    # Step 2: initialize leras and create model
    nn.initialize_main_env()
    nn.set_data_format('NCHW')

    if mode == 'npy':
        # Load from .npy — XSegNet loads the weights automatically
        print('Loading legacy .npy weights...')
        net = XSegNet(
            name='XSeg',
            resolution=resolution,
            load_weights=True,
            weights_file_root=model_dir,
            training=False,
        )
    else:
        # Load from .pth
        print(f'Loading .pth: {weight_file.name}')
        net = XSegNet(
            name='XSeg',
            resolution=resolution,
            load_weights=False,
            weights_file_root=model_dir,
            training=False,
        )
        state = torch.load(str(weight_file), map_location='cpu', weights_only=False)
        if isinstance(state, dict) and 'model' in state:
            net.model.load_state_dict(state['model'])
        else:
            net.model.load_state_dict(state)

    net.model.eval()
    print('Model loaded successfully.')

    # Step 3: export to ONNX
    onnx_path = model_dir / 'XSeg.onnx'
    print(f'\nExporting ONNX to {onnx_path}...')

    class _Wrapper(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model
        def forward(self, x):
            _, pred = self.model(x, pretrain=False)
            return pred

    wrapper = _Wrapper(net.model).cpu().eval()
    dummy = torch.randn(1, 3, resolution, resolution)

    torch.onnx.export(
        wrapper, dummy, str(onnx_path),
        input_names=['input'],
        output_names=['pred'],
        dynamic_axes={'input': {0: 'batch'}, 'pred': {0: 'batch'}},
        opset_version=18,
        dynamo=False,  # 单文件 ONNX，不产生 .data 碎片
    )
    onnx_size = onnx_path.stat().st_size / 1024 / 1024
    print(f'  ONNX exported: {onnx_path.name} ({onnx_size:.0f}MB)')

    # Step 4: save .pth weights
    pth_out = model_dir / f'XSeg_{resolution}.pth'
    torch.save({'model': net.model.state_dict()}, str(pth_out))
    print(f'  .pth saved: {pth_out.name}')

    # Step 6: move legacy .npy files to old/
    if mode == 'npy' and npy_files:
        old_dir.mkdir(parents=True, exist_ok=True)
        for f in npy_files:
            dst = old_dir / f.name
            shutil.move(str(f), str(dst))
            print(f'  Moved {f.name} → old/')

    print('\nDone!')


if __name__ == '__main__':
    main()
