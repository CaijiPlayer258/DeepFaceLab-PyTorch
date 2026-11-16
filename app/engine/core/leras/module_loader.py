# -*- coding: utf-8 -*-
"""模块加载的统一入口：**加载失败 ⇒ 权重 + 优化器状态一起重置**，并留下结构化事件。

为什么需要这一层
----------------
历史上各调用点都是同一个模式：

    ok = model.load_weights(path)
    if not ok:
        model.init_weights()      # PyTorch 里是 pass —— 也就是"什么都没做"

于是出现两种半重置：
  A. 权重真的被随机初始化了，优化器状态却是从旧存档读进来的（形状还恰好能对上，
     因为文件是按 param_i 顺序存、ms/vs 跟着参数走的）；
  B. 权重文件形状不符（判定失败），优化器文件却加载成功 —— 动量按旧尺度继续推
     新初始化的参数，loss 直接起飞，没有任何提示。

这里把"三件事"绑成一个原子动作：
    1. 权重重置（Saveable.reset_weights，就地重初始化，Parameter 身份不变）
    2. 绑定到这些参数的优化器状态清空（OptimizerBase.reset_state）
    3. 一条结构化事件（load_events.make_event → stdout 前缀 JSON + 进程内缓冲）

并且**反向也管**：优化器自己加载失败（文件缺失/形状不符）照样清状态 + 出事件；
如果某个权重模块已经失败过，与它共享参数的优化器即使文件能读也不加载 ——
否则 A 场景又回来了。
"""
from __future__ import annotations

from core.leras import load_events as _lev
from core.leras import nn

# 解码器里“遮罩分支”的层名前缀（decoder 的 upsamplem* / out_convm）。
# 它们和主干同存一个 .pth，所以失败时要能单独把模块名标成 xxx_mask。
MASK_LAYER_PREFIXES = ("upscalem", "out_convm")

_MAX_SHAPES_IN_EVENT = 8


_OPT_CLASS = [None]


def _optimizer_class():
    """拿到 OptimizerBase 类。

    不能只靠 nn.OptimizerBase：那是 Saveable 的注册表，只有当
    core.leras.optimizers.* 被 import 过才存在。少一处 import 就会让
    is_optimizer() 一直返回 False —— 于是"优化器失败也重置状态"整条路静默失效，
    正是这个改动要修的那类 bug，所以这里显式兜住。
    """
    cls = _OPT_CLASS[0]
    if cls is not None:
        return cls
    cls = getattr(nn, 'OptimizerBase', None)
    if cls is None:
        try:
            from core.leras.optimizers.OptimizerBase import OptimizerBase as _OB
            cls = _OB
        except Exception:
            cls = None
    _OPT_CLASS[0] = cls
    return cls


def is_optimizer(obj):
    cls = _optimizer_class()
    if cls is None:
        return False
    try:
        return isinstance(obj, cls)
    except Exception:
        return False


def module_label(module, result=None):
    """事件里显示的模块名。

    普通模块就是自己的 name（encoder / inter_AB / decoder_src …）。
    如果出问题的参数**全部**属于解码器的遮罩分支，就标成 <name>_mask，
    和界面上模块列表的写法一致（decoder_src_mask / decoder_dst_mask）。
    """
    base = getattr(module, 'name', None) or type(module).__name__
    if result is None or not result.affected:
        return base
    owners = [a[0] for a in result.affected if a[0]]
    if owners and all(str(o).lower().startswith(MASK_LAYER_PREFIXES) for o in owners):
        return base + "_mask"
    return base


def optimizer_label(opt):
    return getattr(opt, 'name', None) or type(opt).__name__


def bound_optimizers(module, candidates):
    """返回与 module 的权重**共享参数对象**的优化器（按 Parameter 身份判断）。

    比手工维护 module→optimizer 映射可靠：src_dst_opt 同时管 encoder/inter/decoder，
    code_discriminator 归 D_code_opt，D_src 归 GAN_opt —— 一旦以后拆分/合并优化器，
    手工表就会静默过期（正是"只重置一半"的温床）。
    """
    out = []
    cands = [c for c in (candidates or []) if c is not None]
    if not cands:
        return out
    try:
        ids = set(id(w) for w in module.get_weights())
    except Exception:
        return out
    if not ids:
        return out
    for opt in cands:
        try:
            params = getattr(opt, 'params', None) or []
        except Exception:
            params = []
        if any(id(p) in ids for p in params):
            out.append(opt)
    return out


def _trim_shapes(res):
    """事件里只带少量形状：大模型的 expected_shapes 动辄几百项，整条 JSON 会爆。"""
    exp, fnd = res.expected_shapes, res.found_shapes
    total = max(len(exp), len(fnd))
    if total <= _MAX_SHAPES_IN_EVENT:
        return exp, fnd, total, False
    keep = [a[1] for a in res.affected[:_MAX_SHAPES_IN_EVENT]] or list(range(min(4, total)))
    sel = sorted(set(keep))
    return ([exp[i] for i in sel if i < len(exp)],
            [fnd[i] for i in sel if i < len(fnd)], total, True)


def _reset_module_weights(module):
    n = 0
    try:
        n = int(module.reset_weights() or 0)
    except Exception:
        n = 0
    if n == 0:
        # 兜底：至少把参数洗成随机的，别把"半新半旧"留在内存里
        try:
            import torch
            with torch.no_grad():
                for w in module.get_weights():
                    if w.is_floating_point():
                        w.normal_(0.0, 0.02)
                        n += 1
        except Exception:
            pass
    return n


def reset_optimizers(optimizers, emit=False):
    """清空一批优化器的状态，返回被清空的优化器名。"""
    names = []
    for opt in (optimizers or []):
        if opt is None:
            continue
        try:
            opt.reset_state()
            names.append(optimizer_label(opt))
        except Exception:
            pass
    return names


def load_or_init_phase(entries, path_of, do_init_of=None, optimizers=(),
                       iter_=0, model_prefix=None, emit=True, stream=None):
    """按顺序加载一组 (module, filename)：**失败即重置权重 + 清优化器状态 + 出事件**。

    这是 Model_SAEHD 加载循环里唯一有语义的一段，抽出来是为了能被测试直接驱动
    （测试跑的就是训练时跑的那份代码，不是复刻一份）。

    entries     [(module, filename), ...]，顺序即加载顺序（优化器通常排在最后）
    path_of     filename -> 绝对路径
    do_init_of  (module, filename) -> True 表示"这次不读存档，直接初始化"
                （首次运行 / 刚关掉预训练 / GAN 换模型 等模型侧判断）
    返回        [{'module','filename','ok','skipped','event','needs_init'}, ...]
    """
    results = []
    blocked_opt_ids = set()   # 参数被重置过的模块所绑定的优化器：存档不能再读
    for module, filename in entries:
        skip = False
        if do_init_of is not None:
            try:
                skip = bool(do_init_of(module, filename))
            except Exception:
                skip = False
        ok = False
        event = None
        if not skip:
            path = path_of(filename)
            if is_optimizer(module) and id(module) in blocked_opt_ids:
                # 它的参数刚被重新初始化：存档哪怕结构完好也不能读，
                # 否则动量按旧轨迹推新权重（就是"只重置一半"那个 bug）。
                try:
                    module.reset_state()
                except Exception:
                    pass
                ok = False
                # 不额外发事件：模块那一条已经写了 optimizers_reset
            else:
                opts = (() if is_optimizer(module)
                        else bound_optimizers(module, optimizers))
                ok, event = load_or_reset(module, path, optimizers=opts,
                                          model_prefix=model_prefix, iter_=iter_,
                                          emit=emit, stream=stream)
                if not ok and opts:
                    for o in opts:
                        blocked_opt_ids.add(id(o))
        results.append({'module': module, 'filename': filename, 'ok': ok,
                        'skipped': skip, 'event': event,
                        'needs_init': (not ok)})
    return results


def load_or_reset(module, filename, optimizers=(), model_prefix=None,
                  iter_=0, emit=True, stream=None):
    """加载一个模块；失败则**权重重置 + 优化器状态清空 + 结构化事件**。

    返回 (ok, event)。event 只在失败时非 None。
    """
    if is_optimizer(module):
        return _load_optimizer(module, filename, iter_=iter_, emit=emit, stream=stream)

    try:
        res = module.load_weights_ex(filename, model_prefix=model_prefix)
    except Exception as e:  # 加载器自身炸了也算 corrupt，不再向上抛
        res = _lev.LoadResult(module=getattr(module, 'name', None), path=str(filename),
                              ok=False, reason=_lev.REASON_CORRUPT,
                              detail="加载器异常 %s: %s" % (type(e).__name__, e))

    if res.ok:
        return True, None

    label = module_label(module, res)
    reset_n = _reset_module_weights(module)
    opt_names = reset_optimizers(optimizers)

    exp, fnd, total, truncated = _trim_shapes(res)
    res.expected_shapes, res.found_shapes = exp, fnd
    event = _lev.make_event(
        label, res, iter_=iter_, optimizers_reset=opt_names,
        extra={"weights_reset": reset_n,
               "params_total": total,
               "shapes_truncated": truncated,
               "file": res.path})
    if emit:
        _lev.emit_event(event, stream=stream)
        try:
            print(_lev.format_human(event), flush=True)
        except Exception:
            pass
    try:
        module.load_event = event
    except Exception:
        pass
    return False, event


def _load_optimizer(opt, filename, iter_=0, emit=True, stream=None, forced_reason=None,
                    forced_detail="", linked_module=None):
    """优化器自己的加载：失败 → 清状态 + 事件（原因同样分四类）。"""
    if forced_reason is not None:
        res = _lev.LoadResult(module=optimizer_label(opt), path=str(filename),
                              ok=False, reason=forced_reason, detail=forced_detail)
    else:
        try:
            res = opt.load_weights_ex(filename)
        except Exception as e:
            res = _lev.LoadResult(module=optimizer_label(opt), path=str(filename),
                                  ok=False, reason=_lev.REASON_CORRUPT,
                                  detail="加载器异常 %s: %s" % (type(e).__name__, e))
        if res.ok:
            return True, None

    reset = {}
    try:
        reset = opt.reset_state() or {}
    except Exception:
        pass
    exp, fnd, total, truncated = _trim_shapes(res)
    res.expected_shapes, res.found_shapes = exp, fnd
    event = _lev.make_event(
        optimizer_label(opt), res, iter_=iter_,
        optimizers_reset=[optimizer_label(opt)],
        extra={"weights_reset": 0,
               "params_total": total,
               "shapes_truncated": truncated,
               "file": res.path,
               "linked_module": linked_module,
               "state_cleared": bool(reset.get("params", 0) or reset.get("buffers", 0))})
    if emit:
        _lev.emit_event(event, stream=stream)
        try:
            print(_lev.format_human(event), flush=True)
        except Exception:
            pass
    return False, event


def optimizer_blocked_event(opt, filename, failed_module, iter_=0, emit=True, stream=None):
    """某个权重模块已重置 ⇒ 与它共享参数的优化器不再读存档（直接清状态）。"""
    return _load_optimizer(
        opt, filename, iter_=iter_, emit=emit, stream=stream,
        forced_reason=_lev.REASON_SHAPE,
        forced_detail=("参数所属模块 %s 已重置，优化器状态与权重不同步，"
                       "本次不加载存档" % failed_module),
        linked_module=failed_module)
