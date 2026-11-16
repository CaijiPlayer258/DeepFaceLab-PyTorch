# -*- coding: utf-8 -*-
"""模块加载失败的**结构化事件**。

为什么要有这个文件
------------------
原来是 ``except: return False``，上层只知道"没加载成功"：
  · 分不清是文件不在（missing）、名字对不上（renamed）、形状不符（shape）
    还是内容坏了（corrupt）—— 四种情况的处理建议完全不同；
  · 打出来的是一行 stderr 字符串，界面没法拿它渲染常驻横幅；
  · 最要命的是**只重置了一半**：权重被随机初始化，优化器里的
    Adam 动量/二阶矩还留着上一份存档的形状，两者不同步。

这里只做三件事，不依赖 torch / numpy，方便被服务端与测试单独引用：
  1. 定义原因常量 REASON_*；
  2. LoadResult —— 一次加载的完整结果（含形状、路径、明细）；
  3. make_event() / emit_event() —— 生成并输出一条结构化事件。

事件走 stdout 的**带前缀 JSON 行**（DFL_LOAD_EVENT_PREFIX），
服务端 app/server/_load_events.py 按同一个前缀解析，
再经 /api/train/status 的 warnings[] 交给界面。
选 stdout 而不是写文件：训练进程的标准输出本来就逐行回传给界面，
不需要为此新增落盘文件（也不碰只读的工作区）。
"""
from __future__ import annotations

import json
import os
import sys
import time

# 事件行前缀。引擎与服务端必须一致（服务端那份：app/server/_load_events.py）。
DFL_LOAD_EVENT_PREFIX = "[DFL-LOAD-EVENT]"

# ---- 失败原因 ------------------------------------------------------
REASON_OK = "ok"
REASON_MISSING = "missing"   # 文件不存在
REASON_RENAMED = "renamed"   # 目录里有像它的文件，但名字对不上
REASON_SHAPE = "shape"       # 参数个数/形状与当前网络不符
REASON_CORRUPT = "corrupt"   # 文件在，但内容不是可用的权重

REASONS = (REASON_OK, REASON_MISSING, REASON_RENAMED, REASON_SHAPE, REASON_CORRUPT)

# 给界面看的中文短语（服务端也用同一张表）
REASON_TEXT = {
    REASON_OK: "正常",
    REASON_MISSING: "文件不存在",
    REASON_RENAMED: "文件名对不上",
    REASON_SHAPE: "形状不匹配",
    REASON_CORRUPT: "文件内容损坏",
}

# 进程内保留的最近事件（同一次训练里多次加载/重载都能拿到）
_EVENTS = []
_EVENTS_CAP = 200


class LoadResult(object):
    """一次 load_weights 的完整结果。

    保留 truthiness（__bool__）是为了不破坏几十处 `if ok:` 的老调用点。
    """

    __slots__ = ("ok", "reason", "module", "path", "expected_shapes",
                 "found_shapes", "detail", "affected", "candidates",
                 "fallback_path", "skipped")

    def __init__(self, module=None, path=None, ok=False, reason=REASON_MISSING,
                 detail="", expected_shapes=None, found_shapes=None,
                 affected=None, candidates=None, fallback_path=None):
        self.ok = bool(ok)
        self.reason = reason
        self.module = module
        self.path = path
        self.expected_shapes = list(expected_shapes or [])
        self.found_shapes = list(found_shapes or [])
        self.detail = detail or ""
        # affected: [(owner_layer_name, index, expected_shape, found_shape), ...]
        self.affected = list(affected or [])
        self.candidates = list(candidates or [])
        # .pth/.npy 后缀互换命中时记一笔（这是兼容路径，不算失败）
        self.fallback_path = fallback_path
        self.skipped = False

    def __bool__(self):
        return self.ok

    __nonzero__ = __bool__

    def __repr__(self):
        return "<LoadResult %s ok=%s reason=%s>" % (self.module, self.ok, self.reason)

    def as_dict(self):
        return {
            "module": self.module,
            "ok": self.ok,
            "reason": self.reason,
            "reason_text": REASON_TEXT.get(self.reason, self.reason),
            "path": self.path,
            "expected_shapes": self.expected_shapes,
            "found_shapes": self.found_shapes,
            "detail": self.detail,
        }


def shape_list(v):
    """把 numpy / torch 形状统一成 [int, ...]；None -> []。"""
    if v is None:
        return []
    try:
        return [int(x) for x in tuple(v)]
    except Exception:
        return []


def make_event(module, result, iter_=0, optimizers_reset=None, extra=None):
    """把一次失败组装成结构化事件。

    字段（界面与测试都依赖这些键）：
      module            模块名（decoder_mask / decoder_src / src_dst_opt ...）
      module_file       对应的权重文件名（decoder_src.pth）
      reason            missing / shape / corrupt / renamed
      reason_text       原因的中文短语
      expected_shapes   当前网络期望的形状
      found_shapes      文件里实际的形状（没有则为空）
      affected          出问题的具体参数层 [{layer, index, expected, found}]
      at                事件时间（unix 秒）
      time              ISO 时间
      iter              事件发生时的迭代数
      optimizers_reset  被一起清空状态的优化器名
      path              文件路径
      detail            一句话明细
    """
    ev = {
        "module": module,
        "module_file": os.path.basename(result.path) if result.path else None,
        "reason": result.reason,
        "reason_text": REASON_TEXT.get(result.reason, result.reason),
        "expected_shapes": result.expected_shapes,
        "found_shapes": result.found_shapes,
        "affected": [
            {"layer": a[0], "index": a[1],
             "expected": shape_list(a[2]), "found": shape_list(a[3])}
            for a in result.affected
        ],
        "candidates": result.candidates,
        "detail": result.detail,
        "at": time.time(),
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "iter": int(iter_ or 0),
        "optimizers_reset": list(optimizers_reset or []),
        "path": result.path,
    }
    if extra:
        ev.update(extra)
    return ev


def emit_event(event, stream=None):
    """输出一条结构化事件（带前缀的 JSON 单行）。同时留在进程内环形缓冲里。"""
    line = DFL_LOAD_EVENT_PREFIX + " " + json.dumps(event, ensure_ascii=False, sort_keys=True)
    _EVENTS.append(event)
    if len(_EVENTS) > _EVENTS_CAP:
        del _EVENTS[:-_EVENTS_CAP]
    out = stream if stream is not None else sys.stdout
    try:
        out.write(line + chr(10))
        out.flush()
    except Exception:
        # 输出失败不影响训练：事件已经进缓冲，进程内仍可查
        pass
    return line


def recent_events():
    """本进程最近的加载事件（只读副本）。"""
    return list(_EVENTS)


def clear_events():
    del _EVENTS[:]


def fmt_shape(shape):
    """形状的短写法：[1, 1, 128, 128] -> 128×128，[1024, 1024] -> 1024×1024，[] -> ?"""
    if not shape:
        return "?"
    s = [int(x) for x in shape]
    while len(s) > 1 and s[0] == 1:
        s.pop(0)
    if not s:
        return "?"
    return "×".join(str(x) for x in s)


def shape_hint(event):
    """形状不匹配时给出 "96→128" 这种一眼能读的提示。

    优先用 affected 里那**一个**出问题的参数（最准确）；
    没有就用（已裁剪过的）expected/found 形状列表。
    """
    if event.get("reason") != REASON_SHAPE:
        return ""
    exp = fnd = None
    aff = event.get("affected") or []
    if aff:
        exp = aff[0].get("expected") or None
        fnd = aff[0].get("found") or None
    if not exp or not fnd:
        e = event.get("expected_shapes") or []
        f = event.get("found_shapes") or []
        if len(e) == 1 and len(f) == 1:
            exp, fnd = e[0], f[0]
    if not exp or not fnd:
        return ""
    return " %s→%s" % (fmt_shape(fnd), fmt_shape(exp))


def format_human(event):
    """一行人类可读说明（日志里与结构化事件并存，方便直接 grep）。"""
    opt = event.get("optimizers_reset") or []
    return ("[WARN] %s 未能加载（%s%s），已重置该模块权重与优化器状态%s。"
            % (event.get("module"),
               event.get("reason_text"),
               shape_hint(event),
               ("：" + ", ".join(opt)) if opt else ""))
