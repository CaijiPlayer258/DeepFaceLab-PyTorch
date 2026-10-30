#!/usr/bin/env python3
"""从 SAEHD DF 模型迁移到 DFSingle。

迁移内容：
- Encoder / Inter / decoder_src 权重 → 直接复制（结构完全相同）
- decoder_src → 拆分为人脸分支 + 遮罩分支
decoder_dst → 提取遮罩分支
- data.dat → 更新 resolution 和模型名称

用法:
    python models/Model_DFSingle/migrate_from_saehd.py \\
        --src /path/to/SAEHD/model --name MyModel_SAEHD \\
        --dst /path/to/DFSingle/model --new-name MyModel
"""

import argparse, sys, shutil, pickle, zipfile, io as _io
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))


def load_saveable(path: Path) -> dict:
    """读取 .pth 文件（ZIP of .npy 或 pickle 格式）。"""
    raw = path.read_bytes()
    if raw[:2] == b'PK':
        result = {}
        with zipfile.ZipFile(path, 'r') as z:
            for name in z.namelist():
                with z.open(name) as entry:
                    result[name] = __import__('numpy').load(entry)
        return result
    return pickle.loads(raw)


def save_saveable(path: Path, data: dict):
    """保存为 ZIP-of-npy 格式的 .pth 文件。"""
    import numpy as np
    tmp = path.parent / (path.name + '.tmp')
    with zipfile.ZipFile(tmp, 'w', zipfile.ZIP_DEFLATED) as z:
        for key, arr in data.items():
            buf = _io.BytesIO()
            np.save(buf, np.asarray(arr))
            z.writestr(key, buf.getvalue())
    tmp.replace(path)


def migrate(args):
    src_dir = Path(args.src_dir)
    dst_dir = Path(args.dst_dir)
    name = args.name
    new_name = args.new_name or name.replace('_SAEHD', '')

    dst_dir.mkdir(parents=True, exist_ok=True)

    print(f'[迁移] 源: {src_dir}')
    print(f'[迁移] 目标: {dst_dir}')
    print(f'[迁移] 模型: {name} → {new_name}')

    # ── 1. 复制 Encoder / Inter + 拆分 decoder_src ──
    for comp in ['encoder', 'inter']:
        src = src_dir / f'{name}_{comp}.pth'
        dst = dst_dir / f'{new_name}_DFSingle_{comp}.pth'
        if src.exists():
            shutil.copy2(src, dst)
            print(f'  [OK] {comp}: {src.name} → {dst.name}')
        else:
            print(f'  [WARN] {comp}: {src} 不存在，跳过')

    # 拆分 decoder_src → 人脸分支 + 遮罩分支
    for prefix, save_as in [('decoder_src', 'decoder_src'), ('decoder_dst', 'decoder_dst_mask')]:
        pth = src_dir / f'{name}_{prefix}.pth'
        if not pth.exists():
            print(f'  [WARN] {pth.name} 不存在，跳过')
            continue
        ckpt = load_saveable(pth)
        total = len([k for k in ckpt if k.startswith('param_')])
        is_d = total >= 36
        mask_count = 10 if is_d else 8
        face_count = total - mask_count
        mask_keys = sorted([k for k in ckpt if k.startswith('param_')])

        if prefix == 'decoder_src':
            # 人脸分支：前 face_count 个参数
            face_params = {}
            for i, k in enumerate(mask_keys):
                if i < face_count:
                    face_params[k] = ckpt[k]
            face_dst = dst_dir / f'{new_name}_DFSingle_decoder_src.pth'
            save_saveable(face_dst, face_params)
            print(f'  [OK] decoder_src (face): {face_count} 参数 → {face_dst.name}')

            # src 遮罩分支：后 mask_count 个参数
            src_mask = {}
            for i, k in enumerate(mask_keys):
                if i >= face_count:
                    new_idx = i - face_count
                    src_mask[f'param_{new_idx}'] = ckpt[k]
            src_mask_dst = dst_dir / f'{new_name}_DFSingle_decoder_src_mask.pth'
            save_saveable(src_mask_dst, src_mask)
            print(f'  [OK] decoder_src_mask: {mask_count} 参数 → {src_mask_dst.name}')
        else:
            # decoder_dst_mask：遮罩分支
            dst_mask = {}
            for i, k in enumerate(mask_keys):
                if i >= face_count:
                    new_idx = i - face_count
                    dst_mask[f'param_{new_idx}'] = ckpt[k]
            dst_mask_dst = dst_dir / f'{new_name}_DFSingle_decoder_dst_mask.pth'
            save_saveable(dst_mask_dst, dst_mask)
            print(f'  [OK] decoder_dst_mask: {mask_count} 参数 → {dst_mask_dst.name}')

    # ── 3. 创建 data.dat ──
    dat_src = src_dir / f'{name}_data.dat'
    if dat_src.exists():
        with open(dat_src, 'rb') as f:
            data = pickle.load(f)
        data['options']['resolution'] = int(data['options'].get('resolution', 128))
        dat_dst = dst_dir / f'{new_name}_DFSingle_data.dat'
        with open(dat_dst, 'wb') as f:
            pickle.dump(data, f)
        print(f'  [OK] data.dat → {dat_dst.name}')
    else:
        print(f'  [WARN] {dat_src.name} 不存在')

    print(f'[迁移] 完成')


def main():
    p = argparse.ArgumentParser(description='从 SAEHD DF 迁移到 DFSingle')
    p.add_argument('--src-dir', required=True, type=str, help='SAEHD 模型目录')
    p.add_argument('--name', required=True, type=str, help='SAEHD 模型名（如 MyModel_SAEHD）')
    p.add_argument('--dst-dir', required=True, type=str, help='DFSingle 目标目录')
    p.add_argument('--new-name', type=str, default=None, help='新模型名（默认去掉 _SAEHD 后缀）')
    args = p.parse_args()
    migrate(args)


if __name__ == '__main__':
    main()
