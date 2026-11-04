#!/usr/bin/env python3
"""Inter 分辨率扩缩容工具。

检测已保存模型的实际分辨率，与目标 resolution 比较。
不一致时对 Inter 的 Dense 权重做空间插值填充，
Encoder/Decoder 权重直接复用（纯 Conv 分辨率无关）。

用法:
    python -m models.Model_SAEHD.resize_inter --model-dir /path/to/model --new-res 416
    python -m models.Model_SAEHD.resize_inter --model-dir /path/to/model --new-res 416 --dry-run
"""

import argparse, sys, pickle, zipfile, shutil
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from core.leras import nn


# ── 读写 Saveable 格式 ────────────────────────────────

def load_saveable(path: Path) -> dict:
    """读取 .pth 文件（ZIP of .npy 或 pickle 格式）。"""
    raw = path.read_bytes()
    if raw[:2] == b'PK':  # ZIP
        result = {}
        with zipfile.ZipFile(path, 'r') as z:
            for name in z.namelist():
                with z.open(name) as entry:
                    result[name] = np.load(entry)
        return result
    return pickle.loads(raw)


def save_saveable(path: Path, data: dict):
    """保存为 ZIP-of-npy 格式的 .pth 文件。"""
    import io
    tmp = path.parent / (path.name + '.tmp')
    with zipfile.ZipFile(tmp, 'w', zipfile.ZIP_DEFLATED) as z:
        for key, arr in data.items():
            buf = io.BytesIO()
            np.save(buf, np.asarray(arr))
            z.writestr(key, buf.getvalue())
    tmp.replace(path)


# ── 插值核心 ──────────────────────────────────────────

def interpolate_dense_weight(w: np.ndarray, old_hw: int, new_hw: int) -> np.ndarray:
    """Dense 权重 (C*H*W, out) 空间插值到 (C*H'*W', out)。"""
    t = torch.from_numpy(w)
    C = t.shape[0] // (old_hw * old_hw)
    out_ch = t.shape[1]
    w_4d = t.reshape(C, old_hw, old_hw, out_ch).permute(0, 3, 1, 2)
    w_4d = w_4d.reshape(1, C * out_ch, old_hw, old_hw)
    w_4d = F.interpolate(w_4d, size=(new_hw, new_hw), mode='bilinear', align_corners=False)
    return w_4d.reshape(C, out_ch, new_hw, new_hw).permute(0, 2, 3, 1).reshape(C * new_hw * new_hw, out_ch).numpy()


def interpolate_bias(b: np.ndarray, old_hw: int, new_hw: int) -> np.ndarray:
    """Bias (H*W*C,) 空间插值到 (H'*W'*C,)。"""
    t = torch.from_numpy(b)
    C = t.shape[0] // (old_hw * old_hw)
    b_3d = t.reshape(old_hw, old_hw, C).permute(2, 0, 1)[None]
    b_new = F.interpolate(b_3d, size=(new_hw, new_hw), mode='bilinear', align_corners=False)[0]
    return b_new.permute(1, 2, 0).reshape(new_hw * new_hw * C).numpy()


# ── 主流程 ────────────────────────────────────────────

def resize_model(model_dir: Path, new_res: int, dry_run: bool = False, verbose: bool = True) -> bool:
    """检测并缩放 Inter 权重到新分辨率。"""

    # 定位文件
    data_files = list(model_dir.glob('*_data.dat'))
    if not data_files:
        if verbose: print('[ERROR] No *_data.dat found')
        return False
    data_path = data_files[0]
    name_prefix = data_path.stem.replace('_data', '')

    inter_path = model_dir / f'{name_prefix}_inter.pth'
    if not inter_path.exists():
        if verbose: print(f'[ERROR] {inter_path} not found')
        return False

    # 读取配置
    with open(data_path, 'rb') as f:
        model_data = pickle.load(f)
    opts = model_data.get('options', {})
    ae_dims = int(opts.get('ae_dims', 256))
    e_dims = int(opts.get('e_dims', 64))
    archi = str(opts.get('archi', 'df-ud'))
    archi_opts = archi.split('-')[1] if '-' in archi else ''
    is_d = 'd' in archi_opts

    # 从权重反推原分辨率
    ckpt = load_saveable(inter_path)
    # 键可能有 .npy 后缀
    param_keys = sorted([k for k in ckpt if k.startswith('param_')])
    pk0 = param_keys[0]
    d1w = ckpt[pk0]
    C_enc = e_dims * 8
    saved_hw = int(np.sqrt(d1w.shape[0] / C_enc))
    saved_res = saved_hw * (32 if 't' in archi_opts else 16)

    if saved_res == new_res:
        if verbose: print(f'[OK] Inter already at {new_res}')
        return True

    old_lr = saved_res // (32 if is_d else 16)
    new_lr = new_res // (32 if is_d else 16)

    if verbose:
        print(f'[RESIZE] Inter: {saved_res}→{new_res}  low_res: {old_lr}→{new_lr}')

    # 插值 Inter 权重
    def _key(i):
        return param_keys[i] if i < len(param_keys) else f'param_{i}'

    new_inter = {}
    new_inter[_key(0)] = interpolate_dense_weight(ckpt[_key(0)], old_lr, new_lr)  # dense1.weight
    new_inter[_key(1)] = ckpt[_key(1)]  # dense1.bias
    new_inter[_key(2)] = interpolate_dense_weight(ckpt[_key(2)].T, old_lr, new_lr).T  # dense2.weight
    new_inter[_key(3)] = interpolate_bias(ckpt[_key(3)], old_lr, new_lr)  # dense2.bias
    # 后续参数 (upscale1 conv2d 等) 直接复制
    for i in range(4, len(param_keys)):
        new_inter[param_keys[i]] = ckpt[param_keys[i]]

    if dry_run:
        if verbose:
            print(f'[DRY RUN] Would save Inter @ {new_res}')
            print(f'  dense1: {ckpt[_key(0)].shape} → {new_inter[_key(0)].shape}')
            print(f'  dense2: {ckpt[_key(2)].shape} → {new_inter[_key(2)].shape}')
        return True

    # 备份原有 Inter
    backup = model_dir / f'{name_prefix}_inter_{saved_res}.pth.bak'
    if not backup.exists():
        shutil.copy2(inter_path, backup)
        if verbose: print(f'  Backup: {backup.name}')

    # 保存 Inter
    save_saveable(inter_path, new_inter)
    if verbose: print(f'  Saved: {inter_path.name}')

    # 更新 data.dat 中的 resolution
    opts['resolution'] = new_res
    model_data['options'] = opts
    with open(data_path, 'wb') as f:
        pickle.dump(model_data, f)
    if verbose: print(f'  Updated: {data_path.name} resolution → {new_res}')

    # Encoder/Decoder 权重不变（纯 Conv，分辨率无关）
    if verbose: print('[DONE] Encoder/Decoder weights unchanged (resolution-independent)')
    return True


def main():
    p = argparse.ArgumentParser(description='Resize Inter weights for new resolution')
    p.add_argument('--model-dir', required=True, type=Path)
    p.add_argument('--new-res', required=True, type=int)
    p.add_argument('--dry-run', action='store_true')
    args = p.parse_args()
    resize_model(args.model_dir.resolve(), args.new_res, dry_run=args.dry_run)


if __name__ == '__main__':
    main()
