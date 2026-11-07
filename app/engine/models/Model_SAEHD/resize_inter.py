#!/usr/bin/env python3
"""Inter 分辨率扩缩容工具（A_fix 版：网格与轴序已修正，原子写 + 回滚）。

把一个已训练模型的 Inter 低分辨率网格权重做空间插值，使其匹配新的 resolution；
Encoder / Decoder 是纯 Conv，分辨率无关，权重原样复用。

支持的布局（baseline inter，见 core/leras/archis/DeepFakeArchi.py）
    liae-*   <name>_inter_AB.pth + <name>_inter_B.pth    两份都重采样
    df-*     <name>_inter.pth                            单份

网格必须按引擎真实的 flatten / reshape_4D 语义解析 —— 两者都是通道优先
（索引 = c*h*w + h*w + w）：

    enc = res // 16        （-t 时 res // 32）   encoder 输出网格 = dense1 的输入网格
    low = res // 32        （无 -d 时 res // 16） dense2 的输出网格 = Inter 自己的网格

    dense1.weight  (e_dims*8*enc^2, ae_dims)    行按 (c,h,w) 展平 -> 按 enc 网格插值
    dense2.weight  (ae_dims, ae_out*low^2)      列按 (c,h,w) 展平 -> 转置后按 low 网格插值
    dense2.bias    (ae_out*low^2,)              同样按 (c,h,w) 展平（4.0 曾错用 (h,w,c)）
    upscale1.*                                  与分辨率无关，原样复制

写盘规则：先备份 <file>.bak.<时间戳>，再写临时文件 -> 校验 -> os.replace；
任一步失败则整体回滚。dat 只改 options['resolution']（并追加只读的
options['_resize_history'] 审计记录），其它键原样保留。

用法:
    python -m models.Model_SAEHD.resize_inter --model-dir <dir> --new-res 416
    python -m models.Model_SAEHD.resize_inter --model-dir <dir> --new-res 416 --dry-run
    python -m models.Model_SAEHD.resize_inter --model-dir <dir> --new-res 256 --method bicubic
"""

from __future__ import annotations

import argparse
import io as _io
import math
import os
import pickle
import shutil
import sys
import time
import zipfile
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

try:                                        # 纯字符串解析器，不 import torch
    from core.archi_names import parse_archi_string
except Exception:                            # 模块缺失时退化为本文件内的解析
    parse_archi_string = None


__all__ = [
    'ResizeError',
    'resize_model',
    'plan_resize',
    'apply_plan',
    'interpolate_dense_weight',
    'interpolate_bias',
    'load_saveable',
    'save_saveable',
    'main',
]

_METHODS = ('bilinear', 'bicubic')
_BACKUP_TS_FMT = '%Y%m%d-%H%M%S'
_HISTORY_KEY = '_resize_history'
_HISTORY_MAX = 50


class ResizeError(Exception):
    """无法安全完成缩放（架构 / 分辨率 / 文件不符）时抛出，消息本身即原因。"""


# ────────────────────────────── 存档读写 ──────────────────────────────

def _param_index(key) -> Optional[int]:
    """'param_7' / 'param_7.npy' -> 7；其它键返回 None。"""
    s = str(key)
    if s.endswith('.npy'):
        s = s[:-4]
    if not s.startswith('param_'):
        return None
    tail = s[len('param_'):]
    return int(tail) if tail.isdigit() else None


def _entry_name(key) -> str:
    """写到 zip 里的条目名：与 core.leras.layers.Saveable 完全一致（param_<i>.npy）。"""
    idx = _param_index(key)
    return 'param_' + str(idx) + '.npy' if idx is not None else str(key)


def load_saveable(path) -> dict:
    """读取 .pth 权重（ZIP of .npy，或老式 pickle）。返回 原始键 -> ndarray。"""
    path = Path(path)
    if not path.exists():
        raise ResizeError('权重文件不存在: ' + str(path))
    with open(path, 'rb') as fh:
        head = fh.read(2)
    try:
        if head == b'PK':
            result = {}
            with zipfile.ZipFile(path, 'r') as z:
                for name in z.namelist():
                    if name.endswith('/'):
                        continue
                    with z.open(name) as entry:
                        result[name] = np.load(entry)
            return result
        return pickle.loads(path.read_bytes())
    except ResizeError:
        raise
    except Exception as e:
        raise ResizeError(path.name + ' 读取失败（' + type(e).__name__ + ': ' + str(e) + '）')


def _save_zip(tmp: Path, data: dict) -> None:
    """把 {键: ndarray} 原子写入 tmp（ZIP_STORED + param_<i>.npy，与引擎一致）。"""
    with zipfile.ZipFile(tmp, 'w', zipfile.ZIP_STORED) as z:
        for key, arr in data.items():
            buf = _io.BytesIO()
            np.save(buf, np.ascontiguousarray(np.asarray(arr)))
            z.writestr(_entry_name(key), buf.getvalue())
    with open(tmp, 'rb+') as fh:                 # 落盘再 rename，断电也不会留半个文件
        fh.flush()
        os.fsync(fh.fileno())


def save_saveable(path, data: dict) -> None:
    """保存为 ZIP-of-npy 的 .pth（先写临时文件再 os.replace）。"""
    path = Path(path)
    tmp = path.parent / (path.name + '.tmp' + str(os.getpid()))
    try:
        _save_zip(tmp, data)
        os.replace(tmp, path)
    finally:
        _unlink_quiet(tmp)


def _unlink_quiet(p) -> None:
    try:
        if p is not None and Path(p).exists():
            Path(p).unlink()
    except OSError:
        pass


def _peek_param_shapes(path) -> Dict[int, Tuple[int, ...]]:
    """只读 npy 头，拿到每个 param_<i> 的形状（不把几百 MB 权重读进内存）。"""
    path = Path(path)
    with open(path, 'rb') as fh:
        head = fh.read(2)
    if head != b'PK':                            # 老式 pickle：只能整体反序列化
        ckpt = load_saveable(path)
        out = {}
        for k, v in ckpt.items():
            i = _param_index(k)
            if i is not None:
                out[i] = tuple(int(s) for s in np.asarray(v).shape)
        return out
    out = {}
    with zipfile.ZipFile(path, 'r') as z:
        for name in z.namelist():
            i = _param_index(name)
            if i is None or name.endswith('/'):
                continue
            try:
                with z.open(name) as fh:
                    version = np.lib.format.read_magic(fh)
                    if version == (1, 0):
                        shape = np.lib.format.read_array_header_1_0(fh)[0]
                    else:
                        shape = np.lib.format.read_array_header_2_0(fh)[0]
                out[i] = tuple(int(s) for s in shape)
            except Exception:                    # 头解析失败就退回整读
                with z.open(name) as fh:
                    out[i] = tuple(int(s) for s in np.load(fh).shape)
    return out


def _load_inter_params(path):
    """-> (按 param_<i> 排序的 ndarray 列表, 其它非 param 条目)"""
    ckpt = load_saveable(path)
    params, extra = {}, {}
    for k, v in ckpt.items():
        i = _param_index(k)
        if i is None:
            extra[str(k)] = np.asarray(v)
        else:
            params[i] = np.asarray(v)
    idx = sorted(params)
    if idx != list(range(len(idx))):
        raise ResizeError(Path(path).name + ' 的 param 序号不连续: ' + str(idx[:12]))
    return [params[i] for i in range(len(idx))], extra


def _verify_ckpt(path, expect: Dict[int, Tuple[int, ...]], label: str) -> None:
    shapes = _peek_param_shapes(path)
    if set(shapes) != set(expect):
        raise ResizeError(label + ' 校验失败：param 个数 ' + str(len(shapes))
                          + '，期望 ' + str(len(expect)))
    for i, shape in expect.items():
        if tuple(shapes[i]) != tuple(shape):
            raise ResizeError(label + ' 校验失败：param_' + str(i) + ' 形状 '
                              + str(tuple(shapes[i])) + '，期望 ' + str(tuple(shape)))


# ────────────────────────────── 插值核心 ──────────────────────────────

def _interp_channel_major(mat: np.ndarray, C: int, old_hw: int, new_hw: int,
                          method: str = 'bilinear') -> np.ndarray:
    """(C*old_hw^2, N) 按通道优先展平的空间权重 -> (C*new_hw^2, N)。

    每一 (通道, 输出列) 组合当成一张 old_hw x old_hw 的图做插值；
    C*N 个图一次性塞进 batch 维度，一次 F.interpolate 搞定。
    """
    if old_hw <= 0 or new_hw <= 0:
        raise ResizeError('非法网格: ' + str(old_hw) + ' -> ' + str(new_hw))
    mat = np.ascontiguousarray(np.asarray(mat))
    if mat.ndim != 2:
        raise ResizeError('dense 权重必须是 2D，实际 ' + str(mat.shape))
    rows, out_ch = mat.shape
    if rows != C * old_hw * old_hw:
        raise ResizeError('权重行数 ' + str(rows) + ' 与 ' + str(C) + '*' + str(old_hw)
                          + '^2 不符（通道优先展平）')
    t = torch.from_numpy(mat.astype(np.float32, copy=False))
    x = t.reshape(C, old_hw, old_hw, out_ch).permute(0, 3, 1, 2)
    x = x.reshape(1, C * out_ch, old_hw, old_hw)
    x = F.interpolate(x, size=(new_hw, new_hw), mode=method, align_corners=False)
    x = x.reshape(C, out_ch, new_hw, new_hw).permute(0, 2, 3, 1)
    out = x.reshape(C * new_hw * new_hw, out_ch).numpy()
    if not np.isfinite(out).all():
        raise ResizeError('插值结果含 NaN/Inf，已中止（未写盘）')
    return out


def interpolate_dense_weight(w, old_hw: int, new_hw: int, method: str = 'bilinear') -> np.ndarray:
    """Dense 权重 (C*H*W, out) 空间插值到 (C*H'*W', out)；行索引 = c*H*W + h*W + w。"""
    w = np.asarray(w)
    if w.ndim != 2:
        raise ResizeError('dense 权重必须是 2D，实际 ' + str(w.shape))
    cells = old_hw * old_hw
    if w.shape[0] % cells != 0:
        raise ResizeError('权重行数 ' + str(w.shape[0]) + ' 不能被 ' + str(old_hw)
                          + '^2 整除，无法按通道优先解析')
    C = w.shape[0] // cells
    return _interp_channel_major(w, C, old_hw, new_hw, method)


def interpolate_bias(b, old_hw: int, new_hw: int, method: str = 'bilinear') -> np.ndarray:
    """Bias (C*H*W,) -> (C*H'*W',)。

    修正点：展平索引是 c*H*W + h*W + w，所以必须按 (C, h, w) reshape；
    4.0 用的是 (h, w, c)，等于把同一段内存按另一种形状重新解释（轴序错）。
    """
    b = np.ascontiguousarray(np.asarray(b).reshape(-1))
    cells = old_hw * old_hw
    if b.size % cells != 0:
        raise ResizeError('bias 长度 ' + str(b.size) + ' 不能被 ' + str(old_hw)
                          + '^2 整除，无法按通道优先解析')
    C = b.size // cells
    x = torch.from_numpy(b.astype(np.float32, copy=False)).reshape(1, C, old_hw, old_hw)
    x = F.interpolate(x, size=(new_hw, new_hw), mode=method, align_corners=False)
    out = x.reshape(-1).numpy()
    if not np.isfinite(out).all():
        raise ResizeError('bias 插值结果含 NaN/Inf，已中止（未写盘）')
    return out


# ────────────────────────────── 几何 / 校验 ──────────────────────────────

def _encoder_grid(res: int, archi_opts: str) -> int:
    """encoder 输出网格（= dense1 的输入网格）。与 Encoder.get_out_res 一致。"""
    return res // (32 if 't' in archi_opts else 16)


def _inter_grid(res: int, archi_opts: str) -> int:
    """Inter 自己张开的网格（= dense2 的输出网格）。与 lowest_dense_res 一致。"""
    return res // (32 if 'd' in archi_opts else 16)


def _requirement(archi_opts: str) -> Tuple[int, str]:
    """返回 (new_res 必须是的倍数, 原因)。"""
    if 'd' in archi_opts:
        return 32, '-d 让 Inter 网格再降一半（res//32 才可能是整数网格）'
    if 't' in archi_opts:
        return 32, '-t 让 encoder 多下一次采样（encoder 网格 = res//32）'
    return 16, 'encoder 4 次下采样，res//16 必须是整数网格'


def _check_new_res(new_res: int, archi_opts: str) -> None:
    if not isinstance(new_res, int) or isinstance(new_res, bool) or new_res <= 0:
        raise ResizeError('new-res 必须是正整数，收到 ' + repr(new_res))
    step, why = _requirement(archi_opts)
    if new_res % step != 0:
        lo = max(step, (new_res // step) * step)
        raise ResizeError('分辨率 ' + str(new_res) + ' 不是 ' + str(step) + ' 的倍数：' + why
                          + '。该架构（opts=' + (archi_opts or '-') + '）要求 new-res 是 '
                          + str(step) + ' 的倍数，最近的合法值是 ' + str(lo) + ' / ' + str(lo + step))
    g = _encoder_grid(new_res, archi_opts)
    if g < 2:
        raise ResizeError('分辨率 ' + str(new_res) + ' 太小：encoder 网格只有 ' + str(g) + 'x' + str(g)
                          + '，插值没有意义（至少需要 ' + str(16 * 2) + '）')


def _opt_int(opts: dict, key: str) -> Optional[int]:
    try:
        return int(opts[key])
    except (KeyError, TypeError, ValueError):
        return None


# ────────────────────────────── dat 读写 ──────────────────────────────

def _read_dat(path: Path):
    """-> (model_data, pickle 协议号)。协议号沿用原文件，避免下游老工具读不了。"""
    raw = path.read_bytes()
    proto = pickle.DEFAULT_PROTOCOL
    if len(raw) > 1 and raw[0] == 0x80:
        p = int(raw[1])
        if 2 <= p <= pickle.HIGHEST_PROTOCOL:
            proto = p
    try:
        data = pickle.loads(raw)
    except Exception as e:
        raise ResizeError(path.name + ' 反序列化失败（' + type(e).__name__ + ': ' + str(e) + '）')
    if not isinstance(data, dict):
        raise ResizeError(path.name + ' 内容不是 dict，不像 SAEHD 的 data.dat')
    return data, proto


def _write_dat_atomic(path: Path, model_data: dict, protocol: int) -> None:
    tmp = path.parent / (path.name + '.tmp' + str(os.getpid()))
    try:
        with open(tmp, 'wb') as fh:
            pickle.dump(model_data, fh, protocol=protocol)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    finally:
        _unlink_quiet(tmp)


# ────────────────────────────── 定位 / 架构判定 ──────────────────────────────

def _parse_archi(archi: str):
    if parse_archi_string is not None:
        return parse_archi_string(archi)
    parts = str(archi).lower().split('-')
    if len(parts) == 1:
        return (parts[0], None, '') if parts[0] in ('df', 'liae') else None
    if len(parts) in (2, 3):
        base, opts = parts[0], parts[1]
        mod = parts[2] if len(parts) == 3 else ''
        if base in ('df', 'liae') and opts and all(ch in 'udtc' for ch in opts):
            return (base, opts, mod)
    return None


def _locate_model(model_dir: Path) -> Tuple[str, Path]:
    """从 *_inter*.pth 反推模型名，再找同名 data.dat。多个/零个都直接报错。"""
    names = set()
    for p in sorted(model_dir.iterdir()):
        if not p.is_file() or p.suffix.lower() != '.pth':
            continue
        stem = p.stem
        for suffix in ('_inter_AB', '_inter_B', '_inter'):
            if stem.endswith(suffix):
                names.add(stem[:-len(suffix)])
                break
    if not names:
        raise ResizeError('目录里没有 inter 权重（期望 <name>_inter.pth，或 <name>_inter_AB.pth + '
                          '<name>_inter_B.pth）：' + str(model_dir))
    if len(names) > 1:
        raise ResizeError('目录里有多个模型权重 ' + str(sorted(names))
                          + '，无法判定要处理哪一个；请只保留一个模型的文件')
    name = names.pop()
    dat = model_dir / (name + '_data.dat')
    if not dat.exists():
        raise ResizeError('找到 ' + name + '_inter*.pth 但缺少 ' + dat.name
                          + '（拿不到 archi / ae_dims / resolution）')
    return name, dat


@dataclass
class _ArchiInfo:
    archi: str
    type: str
    opts: str
    inter_names: List[str]
    opts_known: bool
    notes: List[str] = field(default_factory=list)


def _detect_archi(model_dir: Path, model_name: str, archi_raw: str, opts: dict) -> _ArchiInfo:
    """优先用 dat 的 options.archi；缺失/无法解析时按存在的 inter 文件推断类型。"""
    has_df = (model_dir / (model_name + '_inter.pth')).is_file()
    has_ab = (model_dir / (model_name + '_inter_AB.pth')).is_file()
    has_b = (model_dir / (model_name + '_inter_B.pth')).is_file()

    parsed = _parse_archi(archi_raw) if archi_raw else None
    if parsed is not None:
        atype, aopts, amod = parsed
        aopts = aopts or ''
        if amod:
            raise ResizeError('archi=' + repr(archi_raw) + ' 是 DFLite 变体（修饰符 ' + repr(amod)
                              + '）：DFLite 的 Inter 布局与 baseline 不同（dense2 网格 / proj / '
                              '无 upscale1），本工具不处理；请用 baseline inter 或手工迁移')
        names = ([model_name + '_inter_AB.pth', model_name + '_inter_B.pth']
                 if atype == 'liae' else [model_name + '_inter.pth'])
        missing = [n for n in names if not (model_dir / n).is_file()]
        if missing:
            raise ResizeError('options.archi=' + repr(archi_raw) + ' 判定为 ' + atype
                              + '，但缺少 ' + ', '.join(missing)
                              + '。dat 与权重不匹配，先确认哪个是对的（不要靠猜）')
        return _ArchiInfo(archi_raw, atype, aopts, names, True)

    # ---- dat 里没有可用的 archi：按文件推断架构类型（不猜 opts） ----
    notes = ['dat 的 options.archi=' + repr(archi_raw) + ' 无法解析，改为按权重文件推断']
    if has_ab and has_b and not has_df:
        atype, names = 'liae', [model_name + '_inter_AB.pth', model_name + '_inter_B.pth']
    elif has_df and not (has_ab or has_b):
        atype, names = 'df', [model_name + '_inter.pth']
    elif not (has_df or has_ab or has_b):
        raise ResizeError('没有任何 inter 权重文件，这不是已训练的 SAEHD 模型目录: ' + str(model_dir))
    else:
        raise ResizeError('目录里同时存在 inter.pth 与 inter_AB/inter_B，且 dat 的 archi=' + repr(archi_raw)
                          + ' 无法解析，无法判定架构；请只保留一种权重')
    notes.append('按文件推断架构类型 = ' + atype)
    return _ArchiInfo(archi_raw, atype, '', names, False, notes)


def _infer_opts(enc: int, low: int, dat_res: Optional[int], ae_dims: int,
                e_dims: int, ae_out_ch: int, path: Path) -> Tuple[str, int, str]:
    """从 (encoder 网格, Inter 网格) 反推 -t / -d 组合；只有唯一解才接受。"""
    cands = []
    for has_t in (False, True):
        res = enc * (32 if has_t else 16)
        for has_d in (False, True):
            if res // (32 if has_d else 16) == low:
                cands.append((('t' if has_t else '') + ('d' if has_d else ''), res))
    if len(cands) > 1 and dat_res:
        filtered = [c for c in cands if c[1] == dat_res]
        if len(filtered) == 1:
            cands = filtered
    if len(cands) != 1:
        raise ResizeError('无法判定 -t/-d 组合：' + path.name + ' 的网格是 enc='
                          + str(enc) + ' / low=' + str(low) + '，可能的组合 '
                          + str([c[0] or '-' for c in cands]) + '（推导分辨率 '
                          + str([c[1] for c in cands]) + '，dat 记录 '
                          + str(dat_res) + '）。请在 dat 的 options.archi 里显式写上架构')
    opts, res = cands[0]
    return opts, res, ('按网格推断 archi opts = ' + (opts or '-') + '，分辨率 ' + str(res))


# ────────────────────────────── 计划 ──────────────────────────────

@dataclass
class InterPlan:
    path: Path
    old_res: int
    old_enc: int
    new_enc: int
    old_low: int
    new_low: int
    old_shapes: Dict[int, Tuple[int, ...]]
    new_shapes: Dict[int, Tuple[int, ...]]
    kinds: Dict[int, str]
    resample: bool

    def describe(self) -> str:
        def shp(i):
            a, b = self.old_shapes[i], self.new_shapes[i]
            return str(a) + ' -> ' + str(b) if a != b else str(a) + ' (不变)'
        extra = len(self.old_shapes) - 4
        return ('dense1 ' + shp(0) + '   dense2 ' + shp(2) + '   bias ' + shp(3)
                + '   其余 ' + str(extra) + ' 个参数原样复制')


@dataclass
class ResizePlan:
    model_dir: Path
    model_name: str
    dat_path: Path
    archi: str
    archi_type: str
    archi_opts: str
    ae_dims: int
    e_dims: int
    ae_out_ch: int
    dat_res: Optional[int]
    old_res: int
    new_res: int
    old_enc: int
    new_enc: int
    old_low: int
    new_low: int
    method: str
    inters: List[InterPlan]
    dat_data: dict
    dat_protocol: int
    need_resize: bool
    dat_needs_update: bool
    backups: List[Tuple[Path, Path]] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)


def _plan_inter(path: Path, archi_opts: str, ae_dims: int, e_dims: int,
                ae_out_ch: int, new_res: int, method: str) -> InterPlan:
    shapes = _peek_param_shapes(path)
    if not shapes:
        raise ResizeError(path.name + ' 里没有 param_* 条目，不是有效的 inter 权重')
    idx = sorted(shapes)
    if idx != list(range(len(idx))):
        raise ResizeError(path.name + ' 的 param 序号不连续: ' + str(idx[:12]))
    if len(idx) < 4:
        raise ResizeError(path.name + ' 只有 ' + str(len(idx)) + ' 个参数，Inter 至少 4 个'
                          '（dense1.w/b + dense2.w/b），布局不是 baseline Inter')

    c_enc = e_dims * 8
    rows1 = shapes[0][0]
    if rows1 % c_enc != 0:
        raise ResizeError(path.name + ' dense1 行数 ' + str(rows1) + ' 不是 e_dims*8='
                          + str(c_enc) + ' 的整数倍：不是 baseline encoder(通道优先) 的输出')
    enc_sq = rows1 // c_enc
    enc = math.isqrt(enc_sq)
    if enc * enc != enc_sq:
        raise ResizeError(path.name + ' dense1 行数 ' + str(rows1) + ' / ' + str(c_enc)
                          + ' = ' + str(enc_sq) + ' 不是完全平方数，解析不出 encoder 网格')
    old_res = enc * (32 if 't' in archi_opts else 16)
    low = _inter_grid(old_res, archi_opts)

    want = {
        0: (c_enc * enc * enc, ae_dims),
        1: (ae_dims,),
        2: (ae_dims, ae_out_ch * low * low),
        3: (ae_out_ch * low * low,),
    }
    bad = [i for i in want if tuple(shapes[i]) != tuple(want[i])]
    if bad:
        detail = ' / '.join('param_' + str(i) + ' 实际 ' + str(tuple(shapes[i]))
                            + ' 应为 ' + str(want[i]) for i in bad)
        raise ResizeError(path.name + ' 的形状与 archi（opts=' + (archi_opts or '-') + '）推导不符：'
                          + detail + '。可能 dat 的 resolution/archi 被外部改过，或不是 baseline Inter')
    new_enc = _encoder_grid(new_res, archi_opts)
    new_low = _inter_grid(new_res, archi_opts)
    new_shapes = dict(shapes)
    new_shapes[0] = (c_enc * new_enc * new_enc, ae_dims)
    new_shapes[2] = (ae_dims, ae_out_ch * new_low * new_low)
    new_shapes[3] = (ae_out_ch * new_low * new_low,)
    return InterPlan(path=path, old_res=old_res, old_enc=enc, new_enc=new_enc,
                     old_low=low, new_low=new_low, old_shapes=dict(shapes),
                     new_shapes=new_shapes, kinds={0: 'dense1', 2: 'dense2', 3: 'bias'},
                     resample=(old_res != new_res))


def plan_resize(model_dir, new_res, method: str = 'bilinear') -> ResizePlan:
    """只做检测与校验，不碰磁盘（抛 ResizeError 说明原因）。"""
    model_dir = Path(model_dir).expanduser().resolve()
    if not model_dir.is_dir():
        raise ResizeError('模型目录不存在: ' + str(model_dir))
    if method not in _METHODS:
        raise ResizeError('未知插值方法 ' + repr(method) + '，可选 ' + ' / '.join(_METHODS))
    new_res = int(new_res)

    model_name, dat_path = _locate_model(model_dir)
    model_data, dat_protocol = _read_dat(dat_path)
    opts = model_data.get('options')
    if not isinstance(opts, dict):
        raise ResizeError(dat_path.name + ' 里没有 options 字典，不是 SAEHD 的数据文件')

    archi_raw = str(opts.get('archi') or '')
    info = _detect_archi(model_dir, model_name, archi_raw, opts)

    ae_dims = _opt_int(opts, 'ae_dims')
    e_dims = _opt_int(opts, 'e_dims')
    if not ae_dims or not e_dims:
        raise ResizeError(dat_path.name + ' 的 options 缺少 ae_dims/e_dims（ae_dims='
                          + repr(opts.get('ae_dims')) + ', e_dims=' + repr(opts.get('e_dims'))
                          + '），解析不了 Inter 布局')
    ae_out_ch = ae_dims * 2 if info.type == 'liae' else ae_dims
    dat_res = _opt_int(opts, 'resolution')

    archi_opts = info.opts
    if not info.opts_known:
        shapes0 = _peek_param_shapes(model_dir / info.inter_names[0])
        c_enc = e_dims * 8
        rows1 = shapes0.get(0, (0,))[0]
        if rows1 <= 0 or rows1 % c_enc != 0:
            raise ResizeError(info.inter_names[0] + ' dense1 行数 ' + str(rows1)
                              + ' 解析不出 encoder 网格')
        enc_sq = rows1 // c_enc
        enc0 = math.isqrt(enc_sq)
        if enc0 * enc0 != enc_sq:
            raise ResizeError(info.inter_names[0] + ' dense1 网格不是完全平方数')
        cols2 = shapes0.get(2, (0, 0))[1]
        if cols2 <= 0 or cols2 % ae_out_ch != 0:
            raise ResizeError(info.inter_names[0] + ' dense2 列数 ' + str(cols2)
                              + ' 不是 ae_out_ch=' + str(ae_out_ch) + ' 的整数倍')
        low_sq = cols2 // ae_out_ch
        low0 = math.isqrt(low_sq)
        if low0 * low0 != low_sq:
            raise ResizeError(info.inter_names[0] + ' dense2 网格不是完全平方数')
        archi_opts, _, note = _infer_opts(enc0, low0, dat_res, ae_dims, e_dims, ae_out_ch,
                                          model_dir / info.inter_names[0])
        info.notes.append(note)

    _check_new_res(new_res, archi_opts)

    inters = [_plan_inter(model_dir / n, archi_opts, ae_dims, e_dims, ae_out_ch, new_res, method)
              for n in info.inter_names]
    old_res = inters[0].old_res
    for ip in inters[1:]:
        if ip.old_res != old_res:
            raise ResizeError(ip.path.name + ' 推导出的分辨率 ' + str(ip.old_res) + ' 与 '
                              + inters[0].path.name + ' 的 ' + str(old_res)
                              + ' 不一致：两份 inter 不是同一次保存的')
    if old_res != new_res and dat_res and dat_res != old_res:
        info.notes.append('dat 记录的 resolution=' + str(dat_res) + ' 与权重推导的 '
                          + str(old_res) + ' 不一致：以权重为准（上次可能中断过）')

    need_resize = (old_res != new_res)
    dat_needs_update = bool(need_resize or (dat_res != new_res))

    if dat_needs_update:
        opts['resolution'] = new_res
        hist = opts.get(_HISTORY_KEY)
        hist = list(hist) if isinstance(hist, list) else []
        hist.append({
            'at': datetime.now().astimezone().isoformat(timespec='seconds'),
            'tool': 'resize_inter',
            'from': int(old_res),
            'to': int(new_res),
            'method': method,
            'archi': archi_raw,
            'inter_files': [ip.path.name for ip in inters],
            'iter': int(model_data.get('iter', 0) or 0),
        })
        opts[_HISTORY_KEY] = hist[-_HISTORY_MAX:]
        model_data['options'] = opts

    return ResizePlan(model_dir=model_dir, model_name=model_name, dat_path=dat_path,
                      archi=archi_raw, archi_type=info.type, archi_opts=archi_opts,
                      ae_dims=ae_dims, e_dims=e_dims, ae_out_ch=ae_out_ch, dat_res=dat_res,
                      old_res=old_res, new_res=new_res,
                      old_enc=inters[0].old_enc, new_enc=inters[0].new_enc,
                      old_low=inters[0].old_low, new_low=inters[0].new_low,
                      method=method, inters=inters, dat_data=model_data,
                      dat_protocol=dat_protocol, need_resize=need_resize,
                      dat_needs_update=dat_needs_update, notes=info.notes)


def format_plan(plan: ResizePlan, indent: str = '         ') -> List[str]:
    lines = ['[PLAN] 模型 ' + plan.model_name + '  目录 ' + str(plan.model_dir),
             '[PLAN] archi=' + (plan.archi or '(未记录)') + ' -> type=' + plan.archi_type
             + ' opts=' + (plan.archi_opts or '-') + '  ae_dims=' + str(plan.ae_dims)
             + ' e_dims=' + str(plan.e_dims) + ' ae_out_ch=' + str(plan.ae_out_ch),
             '[PLAN] resolution ' + str(plan.old_res) + ' -> ' + str(plan.new_res)
             + '   method=' + plan.method + '   网格 enc ' + str(plan.old_enc) + '->'
             + str(plan.new_enc) + '  low ' + str(plan.old_low) + '->' + str(plan.new_low)]
    for ip in plan.inters:
        lines.append(indent + ip.path.name + ': ' + ip.describe())
    lines.append(indent + plan.dat_path.name + ': options.resolution '
                 + str(plan.dat_res) + ' -> ' + str(plan.new_res)
                 + ('' if plan.dat_needs_update else '（已是目标值）'))
    if not plan.need_resize:
        lines.append(indent + 'Inter 已经在新分辨率上，无需重采样')
    for n in plan.notes:
        lines.append('[NOTE] ' + n)
    return lines


# ────────────────────────────── 执行 / 回滚 ──────────────────────────────

def _make_backup(target: Path, stamp: str) -> Path:
    backup = target.with_name(target.name + '.bak.' + stamp)
    n = 2
    while backup.exists():
        backup = target.with_name(target.name + '.bak.' + stamp + '-' + str(n))
        n += 1
    shutil.copy2(target, backup)
    return backup


def _rollback(backups: Sequence[Tuple[Path, Path]], touched: Sequence[Path]) -> int:
    """把 touched 里的文件用备份原子还原；返回成功还原的个数。"""
    done = 0
    for target, backup in backups:
        if target not in touched:
            continue
        tmp = target.with_name(target.name + '.restore.tmp')
        try:
            shutil.copy2(backup, tmp)
            os.replace(tmp, target)
            done += 1
        except Exception:
            _unlink_quiet(tmp)
    return done


def _build_resized_data(plan: ResizePlan, ip: InterPlan):
    params, extra = _load_inter_params(ip.path)
    if len(params) != len(ip.old_shapes):
        raise ResizeError(ip.path.name + ' 参数个数 ' + str(len(params))
                          + ' 与头部信息 ' + str(len(ip.old_shapes)) + ' 不一致')
    out = []
    for i, p in enumerate(params):
        kind = ip.kinds.get(i)
        dtype = p.dtype
        if kind == 'dense1':
            new = interpolate_dense_weight(p, ip.old_enc, ip.new_enc, plan.method)
        elif kind == 'dense2':
            new = interpolate_dense_weight(p.T, ip.old_low, ip.new_low, plan.method).T
        elif kind == 'bias':
            new = interpolate_bias(p, ip.old_low, ip.new_low, plan.method)
        else:
            new = p
        new = np.ascontiguousarray(new.astype(dtype, copy=False))
        if tuple(new.shape) != tuple(ip.new_shapes[i]):
            raise ResizeError(ip.path.name + ' param_' + str(i) + ' 重采样后形状 '
                              + str(tuple(new.shape)) + '，期望 ' + str(tuple(ip.new_shapes[i])))
        out.append(new)
    if extra:
        raise ResizeError(ip.path.name + ' 里含非 param 条目 ' + str(sorted(extra))
                          + '，本工具不该动它，已中止')
    data = {'param_' + str(i): arr for i, arr in enumerate(out)}
    return data, dict(ip.new_shapes)


def _post_verify(plan: ResizePlan) -> None:
    data, _ = _read_dat(plan.dat_path)
    got = _opt_int(data.get('options') or {}, 'resolution')
    if got != plan.new_res:
        raise ResizeError('写回校验失败：dat 里的 resolution=' + str(got)
                          + '，期望 ' + str(plan.new_res))
    for ip in plan.inters:
        if not ip.resample:
            continue
        shapes = _peek_param_shapes(ip.path)
        if tuple(shapes.get(0, ())) != tuple(ip.new_shapes[0]):
            raise ResizeError('写回校验失败：' + ip.path.name + ' dense1 形状 '
                              + str(tuple(shapes.get(0, ()))) + '，期望 ' + str(ip.new_shapes[0]))
        if tuple(shapes.get(2, ())) != tuple(ip.new_shapes[2]):
            raise ResizeError('写回校验失败：' + ip.path.name + ' dense2 形状 '
                              + str(tuple(shapes.get(2, ()))) + '，期望 ' + str(ip.new_shapes[2]))


def apply_plan(plan: ResizePlan, verbose: bool = True) -> List[Path]:
    """备份 -> 逐文件（写临时 + 校验 + replace）-> 写 dat -> 整体校验；失败回滚。"""
    stamp = time.strftime(_BACKUP_TS_FMT)
    targets = [ip.path for ip in plan.inters if ip.resample] + [plan.dat_path]
    backups: List[Tuple[Path, Path]] = []
    touched: List[Path] = []
    try:
        for t in targets:
            b = _make_backup(t, stamp)
            backups.append((t, b))
            if verbose:
                print('  [BACKUP] ' + t.name + ' -> ' + b.name)
        plan.backups = backups

        if plan.need_resize:
            for ip in plan.inters:
                if not ip.resample:
                    continue
                data, expect = _build_resized_data(plan, ip)
                touched.append(ip.path)
                tmp = ip.path.parent / (ip.path.name + '.tmp' + str(os.getpid()))
                try:
                    _save_zip(tmp, data)
                    _verify_ckpt(tmp, expect, ip.path.name)
                    os.replace(tmp, ip.path)
                finally:
                    _unlink_quiet(tmp)
                if verbose:
                    print('  [WRITE ] ' + ip.path.name + ': ' + ip.describe())

        if plan.dat_needs_update:
            touched.append(plan.dat_path)
            _write_dat_atomic(plan.dat_path, plan.dat_data, plan.dat_protocol)
            if verbose:
                print('  [DAT   ] ' + plan.dat_path.name + ': options.resolution '
                      + str(plan.dat_res) + ' -> ' + str(plan.new_res)
                      + '（+ ' + _HISTORY_KEY + ' 记录）')
        _post_verify(plan)
    except Exception as e:
        restored = _rollback(backups, touched)
        msg = str(e) if isinstance(e, ResizeError) else (type(e).__name__ + ': ' + str(e))
        raise ResizeError(msg + '；已回滚 ' + str(restored) + '/' + str(len(touched))
                          + ' 个文件（备份保留为 *.bak.' + stamp + '）') from e
    return [b for _, b in backups]


# ────────────────────────────── 入口 ──────────────────────────────

def resize_model(model_dir, new_res, dry_run: bool = False, verbose: bool = True,
                 method: str = 'bilinear') -> bool:
    """检测并缩放 Inter 权重到新分辨率。成功返回 True，任何不安全的情况返回 False。"""
    try:
        plan = plan_resize(model_dir, new_res, method=method)
    except ResizeError as e:
        if verbose:
            print('[ERROR] ' + str(e))
        return False

    if verbose:
        for line in format_plan(plan):
            print(line)

    if not plan.need_resize and not plan.dat_needs_update:
        if verbose:
            print('[OK] Inter 与 dat 都已在 ' + str(plan.new_res) + '，无需改动')
        return True

    if dry_run:
        if verbose:
            print('[DRY RUN] 以上为计划，未写入任何文件')
        return True

    try:
        apply_plan(plan, verbose=verbose)
    except ResizeError as e:
        if verbose:
            print('[ERROR] ' + str(e))
        return False
    if verbose:
        print('[DONE] Inter -> ' + str(plan.new_res) + '；Encoder/Decoder 权重未动（分辨率无关）')
        if plan.backups:
            print('       备份: ' + ', '.join(b.name for _, b in plan.backups))
    return True


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description='Resize Inter weights for a new resolution')
    p.add_argument('--model-dir', required=True, type=Path)
    p.add_argument('--new-res', required=True, type=int)
    p.add_argument('--dry-run', action='store_true',
                   help='只打印将要改动的内容，不写盘')
    p.add_argument('--method', choices=_METHODS, default='bilinear',
                   help='dense 空间插值方法（默认 bilinear）')
    args = p.parse_args(argv)
    ok = resize_model(args.model_dir, args.new_res, dry_run=args.dry_run, method=args.method)
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
