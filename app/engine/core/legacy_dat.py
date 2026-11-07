"""(TF 时代) DFL 1.x / 2.0 的 ``*_data.dat`` 兼容读取层 —— 只取 options，不碰权重。

为什么需要它
------------
iperov 的 TF 版把 iter / options / loss_history / sample_for_preview 一起 pickle 进
``<name>_data.dat``。4.1 读同一个文件，但 options 的键名和取值在这几年里改过好几轮：
老 dat 里的 ``face_type='wf'``、``archi='dfhd'``、``learn_mask``、``ed_ch_dims``
到了 4.1 要么查表 KeyError（``{'h':...}[options['face_type']]``），要么被
``load_or_def_option`` 当成「没有这个键」而静默用缺省值 —— 后者更糟：用户看到的是
一个分辨率对、架构却悄悄变成 liae-ud 的模型。

本模块只做「参数翻译」。TF 的权重（``*_encoder.npy`` / ``.ckpt``）与 Torch 的张量布局
不同，只能走 ``tools/convert_dfl_tf_to_torch.py``；``.dfm`` 不在讨论范围。

键差异从哪来（不是猜的）
------------------------
iperov-dfl 的 git 历史（``models/Model_SAEHD/Model.py``）：

  2019-10-08 ac77251  learn_mask / optimizer_mode / ed_ch_dims / apply_random_ct /
                      true_face_training；face_type 只有 {h, mf, f}；archi 只有
                      {df, liae}（无后缀）；目标迭代存成 options['target_epoch']，
                      顶层是 epoch（ModelBase: iter = max(iter, epoch)）
  2020-02-21 f1d115b  face_type 增加 'wf'
  2020-02-27 74999ce  出现 dfhd / liaehd / liaech 三种 archi 与 models_opt_on_gpu
  2020-03-10 ba58621  learn_mask 删除 → masked_training；eyes_prio（后改名 eyes_mouth_prio）
  2020-06-19 0c2e1c3  archi 支持 -u/-d 后缀；archi_type 在那时只是**局部变量**，
                      不是 options 键（有些分支把它写成了键，这里按别名兜住）
  2021-08-29 6e094d8  archi 支持 -t

4.1 这边「有哪些键、缺省值是多少」的唯一事实来源是各
``models/Model_<X>/Model_pytorch.py`` 里的 ``self.load_or_def_option('x', 默认值)``。
本模块用 ast 直接从那几份源码里抽，不另存一份会过期的表。
"""

from __future__ import annotations

import ast
import os
import pickle
from pathlib import Path

ENGINE_ROOT = Path(__file__).resolve().parents[1]

# 4.1 引擎查表用的短码（Model_SAEHD/Model_pytorch.py 里 {'h':..., 'wf':...}）
MODERN_FACE_TYPES = ('h', 'mf', 'f', 'wf', 'head')

# 老值 → 4.1 短码。短码与全名都收：4.1 的界面下拉用的是全名，手改过的 dat 里
# 出现 'whole_face' 时不该炸在查表处；'hf'/'ff' 是界面别名表里的那套写法。
FACE_TYPE_ALIASES = {
    'h': 'h', 'half': 'h', 'hf': 'h', 'half_face': 'h',
    'mf': 'mf', 'mid': 'mf', 'mid_full': 'mf', 'midfull': 'mf',
    'midfull_face': 'mf', 'mid_full_face': 'mf',
    'f': 'f', 'full': 'f', 'ff': 'f', 'full_face': 'f',
    'wf': 'wf', 'whole': 'wf', 'wholeface': 'wf', 'whole_face': 'wf',
    'head': 'head',
}

# TF 时代才有的键（同时用于「这是不是老 dat」的判定）
LEGACY_ONLY_KEYS = {
    'learn_mask': 'TF 的 mask 训练开关，4.1 统一叫 masked_training',
    'optimizer_mode': 'TF 专有：把优化器放 CPU 的档位 1/2/3，4.1 用 models_opt_on_gpu',
    'ed_ch_dims': 'TF 的编码器/解码器通道维，4.1 拆成 e_dims + d_dims',
    'apply_random_ct': 'TF 的随机色彩迁移开关，4.1 是 ct_mode 字符串',
    'true_face_training': 'TF 的 true face 开关，4.1 是 true_face_power(0..1)',
    'eyes_prio': '旧名，4.1 叫 eyes_mouth_prio',
    'archi_type': '不是 iperov 原版的键（历史里只是局部变量），部分分支写成了键',
    'target_epoch': 'DFL 1.x 的目标迭代，4.1 叫 target_iter',
    'use_fp16': 'TF 的 fp16 开关，4.1 用 use_bf16',
    'dst_denoise': 'DFL 2.0 早期选项，后来被删掉了',
}

# 只在 dat 顶层出现、不该当 options 处理的老键
LEGACY_TOP_KEYS = ('epoch', 'iter', 'loss_history', 'sample_for_preview', 'choosed_gpu_indexes')

_MISSING = object()

_np_cache = {}
_DEFAULTS_CACHE = {}


# --------------------------------------------------------------------------
#  小工具
# --------------------------------------------------------------------------
def _numpy():
    if 'np' not in _np_cache:
        try:
            import numpy as np
            _np_cache['np'] = np
        except Exception:
            _np_cache['np'] = None
    return _np_cache['np']


def _decode(b):
    if not isinstance(b, (bytes, bytearray)):
        return str(b)
    try:
        return bytes(b).decode('utf-8')
    except UnicodeDecodeError:
        return bytes(b).decode('latin1')


def _pyify(v, depth=0):
    """numpy 标量/数组 → 原生 Python。返回 (值, 是否改过)。"""
    np = _numpy()
    if np is not None:
        if isinstance(v, np.generic):
            return v.item(), True
        if isinstance(v, np.ndarray):
            return v.tolist(), True
    if depth < 3 and isinstance(v, (list, tuple)):
        changed = False
        out = []
        for item in v:
            nv, ch = _pyify(item, depth + 1)
            changed = changed or ch
            out.append(nv)
        return (out if changed else v), changed
    if depth < 3 and isinstance(v, dict):
        changed = False
        out = {}
        for k, item in v.items():
            nv, ch = _pyify(item, depth + 1)
            changed = changed or ch
            out[k] = nv
        return (out if changed else v), changed
    return v, False


def _as_bool(v):
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return bool(v)
    s = str(v).strip().lower()
    if s in ('1', 'y', 'yes', 'true', 'on'):
        return True
    if s in ('0', 'n', 'no', 'false', 'off', ''):
        return False
    raise ValueError('不是布尔值: %r' % (v,))


def _short_list(items, limit=8):
    items = [str(x) for x in items]
    if len(items) <= limit:
        return ', '.join(items)
    return ', '.join(items[:limit]) + ' …（共 %d 项）' % len(items)


# --------------------------------------------------------------------------
#  4.1 的选项定义（从 Model_pytorch.py 的源码里抽，不复制）
# --------------------------------------------------------------------------
def _literal(node):
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)) \
            and isinstance(node.operand, ast.Constant):
        v = node.operand.value
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            return -v if isinstance(node.op, ast.USub) else v
    return _MISSING


def _parse_model_py(path):
    """返回 (defaults, assigned_keys)。

    defaults      : load_or_def_option('键', <字面量>) 里的字面量默认值
    assigned_keys : 源码里出现过 self.options['键'] 的键

    非字面量的默认值（例如 d_mask_dims = d_dims // 3）故意不抽：那种值由引擎自己
    在加载时算，硬填一个数字反而是引入差异。
    """
    p = Path(path)
    try:
        st = p.stat()
    except OSError:
        return {}, set()
    cache_key = str(p)
    hit = _DEFAULTS_CACHE.get(cache_key)
    if hit is not None and hit[0] == st.st_mtime_ns and hit[1] == st.st_size:
        return hit[2], hit[3]

    defaults, assigned = {}, set()
    try:
        tree = ast.parse(p.read_text(encoding='utf-8'))
    except Exception:
        _DEFAULTS_CACHE[cache_key] = (st.st_mtime_ns, st.st_size, defaults, assigned)
        return defaults, assigned

    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) \
                and node.func.attr == 'load_or_def_option' and len(node.args) >= 2:
            name = node.args[0]
            if isinstance(name, ast.Constant) and isinstance(name.value, str):
                val = _literal(node.args[1])
                if val is not _MISSING and name.value not in defaults:
                    defaults[name.value] = val
        elif isinstance(node, ast.Assign) and len(node.targets) == 1:
            tgt = node.targets[0]
            if isinstance(tgt, ast.Subscript) and isinstance(tgt.value, ast.Attribute) \
                    and tgt.value.attr == 'options':
                sl = tgt.slice
                if isinstance(sl, ast.Constant) and isinstance(sl.value, str):
                    assigned.add(sl.value)
                    val = _literal(node.value)
                    if val is not _MISSING and sl.value not in defaults:
                        defaults[sl.value] = val

    _DEFAULTS_CACHE[cache_key] = (st.st_mtime_ns, st.st_size, defaults, assigned)
    return defaults, assigned


def option_defaults_from_model_py(model_py):
    """某个 Model_pytorch.py 里能静态确定的选项默认值。"""
    if not model_py:
        return {}
    defaults, _ = _parse_model_py(model_py)
    return dict(defaults)


_KNOWN_CACHE = {'stamp': None, 'keys': frozenset()}


def known_option_keys(engine_root=None):
    """4.1 所有模型源码里出现过的选项键（用于判断「无法识别的键」）。"""
    root = Path(engine_root or ENGINE_ROOT)
    models = root / 'models'
    files = [models / 'ModelBase.py'] + sorted(models.glob('Model*/Model_pytorch.py'))
    stamp = tuple((str(p), p.stat().st_mtime_ns) for p in files if p.exists())
    if _KNOWN_CACHE['stamp'] == stamp:
        return _KNOWN_CACHE['keys']
    keys = set()
    for p in files:
        if not p.exists():
            continue
        d, a = _parse_model_py(p)
        keys |= set(d) | set(a)
    _KNOWN_CACHE['stamp'] = stamp
    _KNOWN_CACHE['keys'] = frozenset(keys)
    return _KNOWN_CACHE['keys']


# --------------------------------------------------------------------------
#  archi
# --------------------------------------------------------------------------
def parse_archi(archi):
    """4.1 的 archi 语法（core/archi_names.py，纯字符串、不碰 torch）。"""
    try:
        from core.archi_names import parse_archi_string
        return parse_archi_string(archi)
    except Exception:
        parts = str(archi).lower().split('-')
        if 1 <= len(parts) <= 3 and parts[0] in ('df', 'liae'):
            return (parts[0], parts[1] if len(parts) > 1 else None, '')
        return None


# 只在 TF 侧存在过的 archi 写法
_LEGACY_ARCHI = {'dfhd': 'df', 'liaehd': 'liae', 'liaech': 'liae'}


# --------------------------------------------------------------------------
#  判定
# --------------------------------------------------------------------------
def detect_legacy(options):
    """返回「像 TF 时代 dat」的理由列表；空列表 = 新格式，兼容层不该介入。"""
    reasons = []
    if not isinstance(options, dict):
        return ['options 不是 dict（%s）' % type(options).__name__]

    bad_keys = [k for k in options if not isinstance(k, str)]
    if bad_keys:
        reasons.append('键是 bytes/非 str：' + _short_list(
            [_decode(k) if isinstance(k, (bytes, bytearray)) else repr(k) for k in bad_keys]))

    markers = sorted(k for k in options if isinstance(k, str) and k in LEGACY_ONLY_KEYS)
    if markers:
        reasons.append('TF 时代专有键 ' + _short_list(markers))

    ft = options.get('face_type')
    if isinstance(ft, (bytes, bytearray)):
        reasons.append('face_type 是 bytes（%r）' % (ft,))
    elif isinstance(ft, str) and ft.strip().lower() not in MODERN_FACE_TYPES:
        reasons.append('face_type=%r 不是 4.1 的短码' % (ft,))

    archi = options.get('archi')
    if isinstance(archi, str):
        base = archi.strip().lower().split('-')[0]
        if base in ('df', 'liae') and parse_archi(archi.strip().lower()) is None:
            reasons.append('archi=%r 4.1 解析不了（TF 的 hd/ch 变体）' % (archi,))

    for k in options:
        if isinstance(k, str) and k in LEGACY_TOP_KEYS:
            reasons.append('options 里混进了 dat 顶层键 %r' % (k,))

    return reasons


# --------------------------------------------------------------------------
#  迁移
# --------------------------------------------------------------------------
def migrate_options(raw_options, model_py=None, model_class=None):
    """老 options → 4.1 的 options。返回 (options, report)。

    report 每行以 [迁移]/[改键]/[忽略]/[补默认]/[无法识别]/[说明] 开头，
    界面和 CLI 都能直接按前缀分段显示。
    """
    report = []
    options = {}

    if not isinstance(raw_options, dict):
        return {}, ['[说明] options 不是 dict（%s），按空处理' % type(raw_options).__name__]

    # 1) 键/值归一：bytes → str、numpy → 原生
    for k, v in raw_options.items():
        if isinstance(k, (bytes, bytearray)):
            nk = _decode(k)
            report.append('[迁移] 键 %r → %r（py2 的 str 在 py3 里是 bytes）' % (k, nk))
        elif isinstance(k, str):
            nk = k
        else:
            nk = str(k)
            report.append('[迁移] 键 %r → %r（不是字符串）' % (k, nk))

        nv, changed = _pyify(v)
        if isinstance(nv, (bytes, bytearray)):
            report.append('[迁移] %s: %r → %r（bytes → str）' % (nk, nv, _decode(nv)))
            nv = _decode(nv)
        elif changed:
            report.append('[迁移] %s: numpy %r → %r（转成原生类型）' % (nk, v, nv))

        if nk in options:
            report.append('[忽略] 键 %s 归一后撞名，保留先出现的那份' % nk)
            continue
        options[nk] = nv

    # 2) face_type：老短码 + 全名 → 4.1 短码
    ft = options.get('face_type')
    if isinstance(ft, str):
        short = FACE_TYPE_ALIASES.get(ft.strip().lower())
        if short is None:
            report.append('[无法识别] face_type=%r 不是已知脸型，留给引擎缺省' % (ft,))
        elif short != ft:
            options['face_type'] = short
            report.append("[迁移] face_type: %r → %r（4.1 引擎查表只用短码）" % (ft, short))

    # 3) archi：archi_type 别名 + hd/ch 变体 + 大小写
    if 'archi_type' in options:
        v = options.pop('archi_type')
        if 'archi' not in options:
            options['archi'] = v
            report.append('[改键] archi_type → archi（%r）' % (v,))
        else:
            report.append('[忽略] archi_type=%r：dat 里已有 archi=%r，以 archi 为准'
                          % (v, options['archi']))

    archi = options.get('archi')
    if isinstance(archi, str):
        low = archi.strip().lower()
        if low in _LEGACY_ARCHI:
            options['archi'] = _LEGACY_ARCHI[low]
            report.append('[迁移] archi: %r → %r（TF 的 hd/ch 变体 4.1 没有对应实现；'
                          '参数层面按基础架构走）' % (archi, options['archi']))
        elif low != archi:
            options['archi'] = low
            report.append('[迁移] archi 规范化: %r → %r' % (archi, low))
        if parse_archi(options['archi']) is None:
            report.append('[无法识别] archi=%r 4.1 解析不了，引擎会回退 df-ud；建议手工改'
                          % (options['archi'],))

    # 4) 值映射 / 改键
    #   learn_mask(布尔)          → masked_training
    #   eyes_prio                 → eyes_mouth_prio
    #   target_epoch              → target_iter
    #   ed_ch_dims                → e_dims + d_dims（TF 一个值管两头）
    #   apply_random_ct(布尔)     → ct_mode('rct'/'none')
    #   true_face_training(布尔)  → true_face_power(0.01/0.0)
    def _rename(old, new, mapper, note):
        if old not in options:
            return
        v = options.pop(old)
        if new in options:
            report.append('[忽略] %s=%r：已有 %s=%r，以新键为准'
                          % (old, v, new, options[new]))
            return
        try:
            nv = mapper(v)
        except Exception as e:
            report.append('[无法识别] %s=%r 转不成 %s（%s）' % (old, v, new, e))
            return
        options[new] = nv
        report.append('[改键] %s → %s: %r → %r（%s）' % (old, new, v, nv, note))

    _rename('learn_mask', 'masked_training', _as_bool, 'TF 的 mask 训练开关')
    _rename('eyes_prio', 'eyes_mouth_prio', _as_bool, '旧名 → 新名')
    _rename('target_epoch', 'target_iter', lambda v: int(v), 'DFL 1.x 的迭代目标')
    _rename('apply_random_ct', 'ct_mode',
            lambda v: 'rct' if _as_bool(v) else 'none',
            'True 记成 rct；之后可以按需改成 lct/mkl/sot')
    _rename('true_face_training', 'true_face_power',
            lambda v: 0.01 if _as_bool(v) else 0.0,
            '布尔 → 0.01（TF 文档里的典型值）；0 表示关闭')

    if 'ed_ch_dims' in options:
        v = options.pop('ed_ch_dims')
        try:
            iv = int(v)
        except Exception:
            report.append('[无法识别] ed_ch_dims=%r 不是整数' % (v,))
            iv = None
        if iv is not None:
            hits = []
            for key in ('e_dims', 'd_dims'):
                if key not in options:
                    options[key] = iv
                    hits.append(key)
            if hits:
                report.append('[改键] ed_ch_dims → %s = %d（TF 一个值管编码器+解码器，'
                              '4.1 拆成两项）' % (' + '.join(hits), iv))
            else:
                report.append('[忽略] ed_ch_dims=%r：e_dims/d_dims 都已存在' % (iv,))

    # 5) TF 专有键：留个记录再丢
    for k in sorted(LEGACY_ONLY_KEYS):
        if k in options:
            report.append('[忽略] %s=%r（%s）' % (k, options.pop(k), LEGACY_ONLY_KEYS[k]))

    # 6) 顶层键混进 options 的情况
    if 'epoch' in options:
        report.append('[忽略] options 里的 epoch=%r 是 dat 顶层键' % (options.pop('epoch'),))

    # 7) 补默认：只补源码里写死的字面量；引擎自己算出来的（如 d_mask_dims）不碰
    defaults = option_defaults_from_model_py(model_py)
    for key, dv in defaults.items():
        if key in options or dv is None:
            continue
        options[key] = dv
        report.append('[补默认] %s = %r（dat 里没存这一项）' % (key, dv))
    if not defaults:
        report.append('[说明] 没找到 Model_pytorch.py，未补默认值（缺的键由引擎按自身缺省处理）')

    # 8) 无法识别的键：既不在 4.1 任何模型源码里，也不是上面处理过的老键
    known = known_option_keys()
    for k in sorted(k for k in options if isinstance(k, str) and k not in known):
        report.append('[无法识别] %s=%r：4.1 的选项定义里没有这个键，原样保留但不会生效'
                      % (k, options[k]))

    return options, report


# --------------------------------------------------------------------------
#  报告汇总
# --------------------------------------------------------------------------
_REPORT_TAGS = ('迁移', '改键', '忽略', '补默认', '无法识别', '说明')


def report_counts(report):
    counts = {t: 0 for t in _REPORT_TAGS}
    for line in report or []:
        for t in _REPORT_TAGS:
            if line.startswith('[' + t + ']'):
                counts[t] += 1
                break
    return counts


def moved_count(report):
    c = report_counts(report)
    return c['迁移'] + c['改键']


def format_report(report, indent='  '):
    return '\n'.join(indent + line for line in (report or []))


# --------------------------------------------------------------------------
#  读 / 写
# --------------------------------------------------------------------------
def pickle_compat_loads(blob):
    """按 py3 读；py2 写的（str 里带非 ASCII）退回 bytes 模式再读。"""
    try:
        return pickle.loads(blob)
    except UnicodeDecodeError:
        pass
    except (pickle.UnpicklingError, ValueError):
        pass
    return pickle.loads(blob, encoding='bytes')


def read_dat(path):
    return pickle_compat_loads(Path(path).read_bytes())


def model_class_from_dat_name(dat_path):
    """MyModel_SAEHD_data.dat → SAEHD；models/Model_SAEHD 不存在就当认不出。"""
    name = Path(dat_path).name
    if name.endswith('_data.dat'):
        name = name[:-len('_data.dat')]
    if '_' not in name:
        return None
    cls = name.rsplit('_', 1)[1]
    if (ENGINE_ROOT / 'models' / ('Model_' + cls)).is_dir():
        return cls
    return None


def model_py_for(dat_path, model_class=None):
    cls = model_class or model_class_from_dat_name(dat_path)
    if not cls:
        return None
    p = ENGINE_ROOT / 'models' / ('Model_' + cls) / 'Model_pytorch.py'
    return p if p.exists() else None


def load_legacy_dat(path, model_py=None, model_class=None):
    """老 dat → (options, meta, report)。

    options : 4.1 形状的选项字典（已迁移 / 已补默认 / 未知键原样保留）
    meta    : 顶层信息 —— iter / loss_history / 有没有预览样本 等
    report  : 迁移报告（每行带 [迁移]/[忽略]/… 前缀）
    """
    p = Path(path)
    raw = read_dat(p)
    if not isinstance(raw, dict):
        raise ValueError('%s 顶层不是 dict（%s），不像 DFL 的 data.dat'
                         % (p, type(raw).__name__))

    # 顶层键也归一：py2 写的 pickle 在 py3 里可能连 'iter' 都变成 bytes
    top_notes = []
    raw_norm = {}
    for _k, _v in raw.items():
        if isinstance(_k, (bytes, bytearray)):
            _nk = _decode(_k)
            top_notes.append("[迁移] dat 顶层键 %r → %r（py2 的 str 在 py3 里是 bytes）"
                             % (_k, _nk))
        elif isinstance(_k, str):
            _nk = _k
        else:
            _nk = str(_k)
            top_notes.append("[迁移] dat 顶层键 %r → %r（不是字符串）" % (_k, _nk))
        raw_norm[_nk] = _v
    raw = raw_norm

    iter_v = raw.get('iter', None)
    epoch_v = raw.get('epoch', None)
    base_iter = iter_v if iter_v is not None else epoch_v
    try:
        base_iter = int(base_iter or 0)
    except Exception:
        base_iter = 0

    meta = {
        'iter': base_iter,
        'epoch': epoch_v,
        'loss_history': raw.get('loss_history') or [],
        'has_sample_for_preview': raw.get('sample_for_preview') is not None,
        'choosed_gpu_indexes': raw.get('choosed_gpu_indexes', None),
        'keys': sorted(str(k) for k in raw),
        'path': str(p),
    }

    mp = Path(model_py) if model_py else model_py_for(p, model_class)
    options_raw = raw.get('options') or {}
    if not isinstance(options_raw, dict):
        raise ValueError('%s 里的 options 不是 dict（%s）'
                         % (p, type(options_raw).__name__))
    reasons = detect_legacy(options_raw)
    meta['legacy_reasons'] = reasons
    if not reasons:
        # 新格式：一个字都不改（连键序都照旧），report 空 = 兼容层没介入
        return dict(options_raw), meta, []

    options, report = migrate_options(options_raw, model_py=mp, model_class=model_class)

    report = top_notes + report

    if epoch_v is not None and iter_v is None:
        report.insert(0, '[迁移] 顶层 epoch=%r → iter=%d（DFL 1.x 的写法）'
                      % (epoch_v, base_iter))
    if meta['has_sample_for_preview']:
        report.append('[说明] 老 dat 里有 sample_for_preview：TF 的排布（NHWC/BGR）4.1 '
                      '用不了，写新 dat 时会丢掉，训练时自动重取')

    return options, meta, report


def build_new_dat(options, meta):
    """组装 4.1 自己的 dat 结构。"""
    return {
        'iter': int(meta.get('iter') or 0),
        'options': dict(options),
        'loss_history': meta.get('loss_history') or [],
        'sample_for_preview': None,
        'choosed_gpu_indexes': meta.get('choosed_gpu_indexes', None),
    }


def write_dat(path, data):
    """先写 .tmp 再 os.replace：中途崩不会留下半个 dat。"""
    p = Path(path)
    if not p.parent.is_dir():
        raise FileNotFoundError('目标目录不存在：%s（本工具不会替你建模型目录）' % p.parent)
    tmp = p.with_name(p.name + '.tmp')
    tmp.write_bytes(pickle.dumps(data, protocol=4))
    os.replace(str(tmp), str(p))
    return p
