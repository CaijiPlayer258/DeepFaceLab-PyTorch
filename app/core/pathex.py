import io
import pickle
import struct
import subprocess
from pathlib import Path
from os import scandir

import cv2
import numpy as np

image_extensions = [".jpg", ".jpeg", ".png", ".tif", ".tiff"]


# ── 批量文件移动 ─────────────────────────────────────────────────

def batch_move_files(move_list, work_dir=None):
    """
    批量移动文件：写 .bat → 一次性提交 cmd.exe 执行。

    Args:
        move_list: [(src_path, dst_path), ...] 或 Path 对象列表
        work_dir: batch 文件存放目录
    Returns:
        bool 是否全部成功
    """
    if not move_list:
        return True
    if work_dir is None:
        work_dir = Path(move_list[0][1]).parent
    else:
        work_dir = Path(work_dir)
    bat = work_dir / '_move_temp.bat'
    try:
        with open(bat, 'w', encoding='utf-8') as f:
            for src, dst in move_list:
                f.write(f'move /y "{src}" "{dst}"\n')
        r = subprocess.run([str(bat)], stdout=subprocess.DEVNULL,
                           stderr=subprocess.DEVNULL)
        return r.returncode == 0
    finally:
        if bat.exists():
            bat.unlink()


# ── 安全的 pickle 反序列化 ───────────────────────────────────────
# 只允许 DFL 元数据必需的类型，拒绝任意代码执行。

_SAFE_PICKLE_TYPES = {
    'builtins':           {'dict', 'list', 'tuple', 'str', 'int', 'float',
                           'bool', 'bytes', 'set', 'slice', 'range'},
    '__builtin__':        {'dict', 'list', 'tuple', 'str', 'int', 'float',
                           'bool', 'bytes', 'set', 'slice', 'range'},
    'numpy':              {'ndarray', 'float32', 'float64', 'float16',
                           'int32', 'int64', 'int16', 'int8', 'uint8',
                           'uint16', 'uint32', 'uint64', 'dtype',
                           'generic', 'scalar'},
    'numpy.core.multiarray': {'_reconstruct', 'scalar'},
    'numpy._core.multiarray': {'_reconstruct', 'scalar'},
    'numpy.core.numeric':    {'ones', 'zeros'},
    'numpy._core.numeric':   {'ones', 'zeros'},
    'collections':        {'OrderedDict'},
    '_codecs':            {'encode'},
}


def safe_pickle_load(data):
    """ pickle.loads 的安全版本：白名单类型检查 """
    from core.imagelib import SegIEPolys as _SegIEPolys
    class _RestrictedUnpickler(pickle.Unpickler):
        def find_class(self, module, name):
            allowed = _SAFE_PICKLE_TYPES.get(module)
            if allowed is not None and name in allowed:
                return super().find_class(module, name)
            if name == 'SegIEPolys' and module in ('core.imagelib.SegIEPolys',
                                                    'core.imagelib'):
                return _SegIEPolys
            raise pickle.UnpicklingError(
                f"unsafe pickle type: {module}.{name}")
    return _RestrictedUnpickler(io.BytesIO(data)).load()


# ── DFL 兼容 JPG 保存 ───────────────────────────────────────────

def save_dfljpg(filepath, img, meta_dict):
    """
    保存为 DFL 兼容 JPG（含 APP15 元数据 chunk）。
    零额外内存复制：分段写入避免拼接整个 JPEG 字节串。
    """
    ret, enc = cv2.imencode('.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
    if not ret:
        raise RuntimeError(f"JPEG 编码失败: {filepath}")
    data = enc.tobytes()

    dict_data = {k: v for k, v in meta_dict.items() if v is not None}
    has_numpy = any(isinstance(v, np.ndarray) for v in dict_data.values())
    if has_numpy:
        def _unumpy(obj):
            if isinstance(obj, np.ndarray):
                if obj.dtype.kind in ('u', 'b'):
                    return bytes(obj)
                return obj.tolist()
            if isinstance(obj, (np.floating, np.integer, np.bool_)):
                return obj.item()
            if isinstance(obj, dict):
                return {k: _unumpy(v) for k, v in obj.items()}
            if isinstance(obj, (list, tuple)):
                return [_unumpy(i) for i in obj]
            return obj
        dict_data = _unumpy(dict_data)

    pickle_data = pickle.dumps(dict_data, protocol=2)
    pickle_data = pickle_data.replace(b'numpy._core', b'numpy.core')

    app15 = struct.pack('BB', 0xFF, 0xEF)
    app15 += struct.pack('>H', len(pickle_data) + 2)
    app15 += pickle_data

    sos = data.find(b'\xff\xda')
    with open(filepath, 'wb') as f:
        if sos > 0:
            f.write(data[:sos])
            f.write(app15)
            f.write(data[sos:])
        else:
            f.write(data)
            f.write(app15)


# ── 旧版工具函数 ─────────────────────────────────────────────────

def write_bytes_safe(p, bytes_data):
    """原子写文件（先写.tmp再rename），Windows 上如果目标被其他进程
    打开会导致 PermissionError，此处重试最多 5 次（每次间隔 200ms）。"""
    import time as _time
    p_tmp = p.parent / (p.name + '.tmp')
    p_tmp.write_bytes(bytes_data)
    for _retry in range(5):
        try:
            if p.exists():
                p.unlink()
            p_tmp.rename(p)
            return
        except PermissionError:
            if _retry < 4:
                _time.sleep(0.2)
                continue
            # 最后一次尝试：用 os.replace（可能覆盖打开的文件）
            try:
                import os as _os
                _os.replace(str(p_tmp), str(p))
                return
            except Exception:
                raise
        except Exception:
            raise

def scantree(path):
    for entry in scandir(path):
        if entry.is_dir(follow_symlinks=False):
            yield from scantree(entry.path)
        else:
            yield entry

def get_image_paths(dir_path, image_extensions=image_extensions, subdirs=False, return_Path_class=False):
    dir_path = Path(dir_path)
    result = []
    if dir_path.exists():
        if subdirs:
            gen = scantree(str(dir_path))
        else:
            gen = scandir(str(dir_path))
        for x in list(gen):
            if any([x.name.lower().endswith(ext) for ext in image_extensions]):
                result.append(x.path if not return_Path_class else Path(x.path))
    return sorted(result)

def get_image_unique_filestem_paths(dir_path, verbose_print_func=None):
    result = get_image_paths(dir_path)
    result_dup = set()
    for f in result[:]:
        f_stem = Path(f).stem
        if f_stem in result_dup:
            result.remove(f)
            if verbose_print_func is not None:
                verbose_print_func("Duplicate filenames are not allowed, skipping: %s" % Path(f).name)
            continue
        result_dup.add(f_stem)
    return sorted(result)

def get_paths(dir_path):
    dir_path = Path(dir_path)
    if dir_path.exists():
        return [Path(x) for x in sorted([x.path for x in list(scandir(str(dir_path)))])]
    else:
        return []

def get_file_paths(dir_path):
    dir_path = Path(dir_path)
    if dir_path.exists():
        return [Path(x) for x in sorted([x.path for x in list(scandir(str(dir_path))) if x.is_file()])]
    else:
        return []

def get_all_dir_names(dir_path):
    dir_path = Path(dir_path)
    if dir_path.exists():
        return sorted([x.name for x in list(scandir(str(dir_path))) if x.is_dir()])
    else:
        return []

def get_all_dir_names_startswith(dir_path, startswith):
    dir_path = Path(dir_path)
    startswith = startswith.lower()
    result = []
    if dir_path.exists():
        for x in list(scandir(str(dir_path))):
            if x.name.lower().startswith(startswith):
                result.append(x.name[len(startswith):])
    return sorted(result)

def get_first_file_by_stem(dir_path, stem, exts=None):
    dir_path = Path(dir_path)
    stem = stem.lower()
    if dir_path.exists():
        for x in sorted(list(scandir(str(dir_path))), key=lambda x: x.name):
            if not x.is_file():
                continue
            xp = Path(x.path)
            if xp.stem.lower() == stem and (exts is None or xp.suffix.lower() in exts):
                return xp
    return None

def move_all_files(src_dir_path, dst_dir_path):
    paths = get_file_paths(src_dir_path)
    for p in paths:
        p = Path(p)
        p.rename(Path(dst_dir_path) / p.name)

def delete_all_files(dir_path):
    paths = get_file_paths(dir_path)
    for p in paths:
        p = Path(p)
        p.unlink()
