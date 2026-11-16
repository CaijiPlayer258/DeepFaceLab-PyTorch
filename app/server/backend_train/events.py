# -*- coding: utf-8 -*-
"""把训练进程 stdout 里的 **模块加载失败事件** 解析出来，交给界面。

引擎侧（app/engine/core/leras/load_events.py）在一次模块加载失败时会打一行：

    [DFL-LOAD-EVENT] {"module": "decoder_src_mask", "reason": "shape", ...}

这里负责：
  · parse_line()   单行 → dict（不认识的行返回 None，绝不会因为脏行炸掉）
  · collect()      多行 → 事件列表（按模块去重，保留最后一次）
  · compose_banner()  多个事件**合并成一条**横幅（标题 + 正文 + 模块名列表）

单独一个文件的原因：app.py 会拉起 fastapi/psutil，测试时不想连带加载那些，
而这段解析逻辑是要被单测直接 import 的。
"""
from __future__ import annotations

import json

# 必须与 app/engine/core/leras/load_events.py 的 DFL_LOAD_EVENT_PREFIX 一致
PREFIX = "[DFL-LOAD-EVENT]"

# 与引擎侧 REASON_TEXT 同步（服务端不 import 引擎的 core.leras：那是另一个进程的
# sys.path，跨进程复用同一份代码只会让两边部署耦合）
REASON_TEXT = {
    "ok": "正常",
    "missing": "文件不存在",
    "renamed": "文件名对不上",
    "shape": "形状不匹配",
    "corrupt": "文件内容损坏",
}


def parse_line(line):
    """一行 stdout → 事件 dict；不是事件行就返回 None。"""
    if not line:
        return None
    s = line.strip()
    if not s.startswith(PREFIX):
        return None
    raw = s[len(PREFIX):].strip()
    if not raw:
        return None
    try:
        ev = json.loads(raw)
    except Exception:
        return None
    if not isinstance(ev, dict):
        return None
    ev.setdefault("module", "?")
    ev.setdefault("reason", "corrupt")
    ev["reason_text"] = ev.get("reason_text") or REASON_TEXT.get(ev["reason"], ev["reason"])
    if not ev.get("optimizers_reset"):
        ev["optimizers_reset"] = []
    if not ev.get("affected"):
        ev["affected"] = []
    return ev


def collect(lines, out=None):
    """把多行事件解析并**按模块去重**（同一个模块反复失败只留最后一条）。"""
    out = [] if out is None else out
    for line in lines or []:
        ev = parse_line(line)
        if ev is None:
            continue
        out = [e for e in out if e.get("module") != ev.get("module")]
        out.append(ev)
    return out


def fmt_shape(shape):
    """[1, 1, 128, 128] -> 128×128；[1024, 1024] -> 1024×1024；[] -> ?"""
    if not shape:
        return "?"
    try:
        s = [int(x) for x in shape]
    except Exception:
        return "?"
    while len(s) > 1 and s[0] == 1:
        s.pop(0)
    if not s:
        return "?"
    return "×".join(str(x) for x in s)


def shape_hint(ev):
    """形状不匹配时给 "96→128"；其它原因返回空串。

    优先用 affected 里那一个出问题的参数 —— 大模型的 expected_shapes
    有几百项，取"第一对"往往是错的。
    """
    if ev.get("reason") != "shape":
        return ""
    exp = fnd = None
    aff = ev.get("affected") or []
    if aff:
        exp = aff[0].get("expected") or None
        fnd = aff[0].get("found") or None
    if not exp or not fnd:
        e = ev.get("expected_shapes") or []
        f = ev.get("found_shapes") or []
        if len(e) == 1 and len(f) == 1:
            exp, fnd = e[0], f[0]
    if not exp or not fnd:
        if ev.get("expected_shapes"):
            return " 预期 %s" % fmt_shape(ev["expected_shapes"][0])
        return ""
    return " %s→%s" % (fmt_shape(fnd), fmt_shape(exp))


def module_sentence(ev, tail=True):
    """单个模块的一句话（就是横幅正文里的那一行）。

    例：decoder_mask 未能加载（形状不匹配 128→96），已重置该模块与优化器状态。
        本次训练从该模块的初始状态开始。
    """
    name = ev.get("module") or "?"
    why = ev.get("reason_text") or REASON_TEXT.get(ev.get("reason"), ev.get("reason"))
    head = "%s 未能加载（%s%s）" % (name, why, shape_hint(ev))
    mid = "，已重置该模块与优化器状态"
    opt = ev.get("optimizers_reset") or []
    if opt:
        mid += "（%s）" % "、".join(str(o) for o in opt)
    mid += "。"
    if not tail:
        return head + mid
    return head + mid + "本次训练从该模块的初始状态开始。"


def compose_banner(events):
    """多个事件 → 一条横幅（界面只挂一条，标题说个数，正文逐条列模块名）。"""
    evs = [e for e in (events or []) if e]
    if not evs:
        return None
    names = [str(e.get("module") or "?") for e in evs]
    if len(evs) == 1:
        ev = evs[0]
        title = "%s 未能加载（%s%s）" % (names[0],
                                        ev.get("reason_text") or REASON_TEXT.get(ev.get("reason")),
                                        shape_hint(ev))
        body = module_sentence(ev)
    else:
        title = "%d 个模块未能加载（%s）" % (len(evs), "、".join(names))
        body = "\n".join(module_sentence(e) for e in evs)
    return {
        "title": title,
        "body": body,
        "modules": names,
        "count": len(evs),
        "reasons": [e.get("reason") for e in evs],
        "events": evs,
    }


def warnings_payload(events):
    """给 /api/train/status 的 warnings[]：每条附上可直接显示的一句话 + 横幅。"""
    evs = [dict(e) for e in (events or [])]
    for ev in evs:
        ev["text"] = module_sentence(ev)
        ev["shape_hint"] = shape_hint(ev)
        ev["module"] = ev.get("module") or "?"
        ev["reason_text"] = ev.get("reason_text") or REASON_TEXT.get(ev.get("reason"), ev.get("reason"))
    return evs
