#!/usr/bin/env python
"""
PTH / NPY → DFM (ONNX) 导出工具。
支持 SAEHD / DeepFakeLarge / LIAELarge / XSeg / XSegLite 架构。

用法:
    python tools/export_dfm.py --model SAEHD --model-dir workspace/model
    python tools/export_dfm.py --model DeepFakeLarge --model-dir workspace/model/DeepFakeLarge
"""
import os, sys, warnings, threading, time, pickle
from pathlib import Path

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
sys.path.insert(0, str(Path(__file__).parent.parent))

warnings.filterwarnings('ignore',
    message=r'Constant folding - Only steps=1 can be constant folded.*onnx::Slice.*',
    category=UserWarning,
)


def _silent(*args, **kwargs):
    """Suppress all io.log_info during model loading."""
    pass


def _spinner_until(flag_ref, text, out_file, sub_texts=None):
    """Show spinner while flag_ref[0] is False. Optionally cycle sub_texts."""
    chars = '/-\\|'
    i = 0
    while not flag_ref[0]:
        sub = sub_texts[min(i // 8, len(sub_texts) - 1)] if sub_texts else ''
        if sub:
            out_file.write(f'\r{text} {sub} {chars[i % 4]}  ')
        else:
            out_file.write(f'\r{text} {chars[i % 4]}  ')
        out_file.flush()
        i += 1
        time.sleep(0.12)


def export(model_class_name: str, saved_models_path: str, force_model_name: str = None):
    """Load model and export to ONNX (.dfm)."""
    from core.interact import interact as io

    os.environ['DFL_SILENT_INPUT'] = '1'

    # Auto-convert .npy → .pth if no .pth files exist yet
    if force_model_name:
        _pth_glob = list(Path(saved_models_path).glob(f'{force_model_name}_{model_class_name}_*.pth'))
        _npy_glob = list(Path(saved_models_path).glob(f'{force_model_name}_*.npy'))
        if not _pth_glob and _npy_glob:
            sys.__stdout__.write(f'[1] 转换 {force_model_name} (.npy → .pth)\n')
            sys.__stdout__.flush()
            convert_npy_to_pth(model_class_name, saved_models_path, force_model_name)

    # Ensure iter >= 1 to skip first-run setup
    if force_model_name:
        dat_path = Path(saved_models_path) / f'{force_model_name}_{model_class_name}_data.dat'
        if dat_path.exists():
            try:
                data = pickle.loads(dat_path.read_bytes())
                if data.get('iter', 0) == 0:
                    data['iter'] = 1
                    dat_path.write_bytes(pickle.dumps(data))
            except Exception:
                pass

    # Silence model output
    import models as _models
    _orig_log = io.log_info
    io.log_info = _silent
    _orig_stderr = sys.stderr
    sys.stderr = open(os.devnull, 'w')
    _orig_stdout = sys.stdout
    sys.stdout = open(os.devnull, 'w')

    # Phase 1: load model with spinner that cycles module names
    _spin = [True]
    _mods = ['encoder', 'inter', 'decoder', 'decoder_src', 'decoder_dst']
    _t = threading.Thread(target=_spinner_until,
                          args=(_spin, 'loading', _orig_stdout, _mods), daemon=True)
    _t.start()
    try:
        _load_kw = dict(
            is_exporting=True,
            saved_models_path=Path(saved_models_path),
            cpu_only=True,
            silent_start=True,
        )
        if force_model_name:
            _suffix = f"_{model_class_name}"
            if force_model_name.endswith(_suffix):
                _load_kw['force_model_name'] = force_model_name
            else:
                _load_kw['force_model_name'] = f"{force_model_name}{_suffix}"
        model = _models.import_model(model_class_name)(**_load_kw)
    finally:
        _spin[0] = False
        io.log_info = _orig_log
        sys.stderr = _orig_stderr
        sys.stdout = _orig_stdout
    _t.join(0.3)
    # Strip model class suffix from name for cleaner export filename
    for _suffix in ['_SAEHD', '_PIA', '_DeepFakeLarge', '_LIAELarge', '_XSeg', '_XSegLite']:
        if model.model_name.endswith(_suffix):
            model.model_name = model.model_name[:-len(_suffix)]
            break
    # Overwrite spinner line, then print modules in tree style
    _orig_stdout.write('\r                              \r')
    _mods_found = [m for m in ['encoder', 'inter', 'inter_B', 'inter_AB', 'decoder',
                               'decoder_src', 'decoder_dst', 'decoder_mask'] if hasattr(model, m)]
    for idx, _m in enumerate(_mods_found):
        _pfx = '├' if idx < len(_mods_found) - 1 else '└'
        _orig_stdout.write(f'{_pfx} {_m}: loaded\n')
        _orig_stdout.flush()
        time.sleep(0.12)
    _orig_stdout.write('└ model loaded\n')
    _orig_stdout.flush()

    # Phase 2: export ONNX with spinner
    done = [False]
    t2 = threading.Thread(target=_spinner_until, args=(done, 'exporting', _orig_stdout), daemon=True)
    t2.start()
    try:
        sys.stdout = open(os.devnull, 'w')
        model.export_dfm()
    finally:
        done[0] = True
        sys.stdout = _orig_stdout
    t2.join(0.3)

    # Rename exported .dfm: New416_model.dfm → New416.dfm
    out_path = ''
    try:
        base = Path(saved_models_path)
        dfm_files = list(base.glob('*.dfm'))
        if dfm_files:
            f = max(dfm_files, key=lambda x: x.stat().st_mtime)  # most recent
            if f.stem.endswith('_model'):
                new_name = f.stem[:-6] + '.dfm'
                f.rename(base / new_name)
                out_path = str(base / new_name)
            else:
                out_path = str(f)
        if not out_path:
            out_path = '(not found)'
    except Exception as e:
        out_path = f'(error: {e})'

    _orig_stdout.write(f'\r[OK] {out_path}\n')
    _orig_stdout.flush()


def convert_npy_to_pth(model_class_name: str, saved_models_path: str, force_model_name: str = None):
    """加载 .npy 权重并保存为 .pth（抑制模型摘要，仅显示模块树）"""
    from core.interact import interact as io
    import models

    _orig_log = io.log_info
    io.log_info = _silent
    # 重定向 OS 级 stdin 阻止交互输入回显
    _orig_stdin_fd = os.dup(0)
    _null_fd = os.open(os.devnull, os.O_RDONLY)
    os.dup2(_null_fd, 0)

    _kw = dict(
        is_exporting=True,
        saved_models_path=Path(saved_models_path),
        cpu_only=True,
        silent_start=True,
    )
    if force_model_name:
        # 避免双后缀：force_model_name 可能已包含 _SAEHD
        _suffix = f"_{model_class_name}"
        if force_model_name.endswith(_suffix):
            _full_name = force_model_name
        else:
            _full_name = f"{force_model_name}{_suffix}"
        _kw['force_model_name'] = _full_name
    model = models.import_model(model_class_name)(**_kw)

    # 检测是否加载了已有配置，如果没有则尝试从 old/ 恢复
    if model.iter == 0 or 'resolution' not in model.options:
        _old_dir = Path(saved_models_path) / "old"
        _dat_stem = f"{_kw['force_model_name']}_data.dat"
        _old_dat = _old_dir / _dat_stem
        _cur_dat = Path(saved_models_path) / _dat_stem
        if _old_dat.exists():
            import shutil
            shutil.copy2(str(_old_dat), str(_cur_dat))
            print(f"[restored] 已从 old/{_dat_stem} 恢复模型配置", flush=True)
            # 重新加载模型
            model = models.import_model(model_class_name)(**_kw)
        else:
            print(f"[WARNING] 未找到模型配置（{_dat_stem}），请手动从备份恢复", flush=True)

    io.log_info = _orig_log

    # 显示模块树
    _mods = [m for m in ['encoder', 'inter', 'inter_B', 'inter_AB', 'decoder',
                         'decoder_src', 'decoder_dst', 'decoder_mask'] if hasattr(model, m)]
    for i, _m in enumerate(_mods):
        _pfx = '├' if i < len(_mods) - 1 else '└'
        print(f'{_pfx} {_m}: .npy → .pth')
    print('└ model: .npy → .pth')

    model.save()
    print(f'[OK] {model_class_name} .npy → .pth 转换完成')
    os.dup2(_orig_stdin_fd, 0)
    os.close(_null_fd)
    os.close(_orig_stdin_fd)


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser(description='PTH / NPY → DFM (ONNX) 导出')
    p.add_argument('--model', required=True, dest='model_name', help='模型类型 (SAEHD, DeepFakeLarge, LIAELarge, XSeg, XSegLite)')
    p.add_argument('--model-dir', required=True, dest='model_dir', help='模型保存目录')
    p.add_argument('--force-name', default=None, dest='force_name', help='强制模型名称（可选）')
    args = p.parse_args()
    export(args.model_name, args.model_dir, args.force_name)
