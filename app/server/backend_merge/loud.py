# -*- coding: utf-8 -*-
"""静默失败计数器：这条导出链上「except 之后什么都不说」的地方一律改走这里。

为什么单独一个文件：这条链踩过的四次坑都是同一形态 ——
  导入错模块 / 读不存在的键 / 参数少传 / 挂错入口，
全部被 `except Exception: pass` 吞成「看起来正常」，表现出来是「前端没反应」「只换了一张脸」。
约定：**每一次捕获都必须说出「哪里、什么异常、后果是什么」**，日志里能直接看出降级。
"""
from __future__ import annotations

import os


def enabled() -> bool:
    """DFL_LOUD=0 可关（只影响噪声，不影响功能）。默认开。"""
    return str(os.environ.get("DFL_LOUD", "1")).strip() not in ("0", "false", "no")


def note(where: str, exc, effect: str, log=print) -> str:
    """把一次降级写成一行：`[loud] <where> · <类型>: <消息> · 后果：<effect>`。"""
    if not enabled():
        return ""
    line = "[loud] %s · %s: %s · 后果：%s" % (
        where, exc.__class__.__name__, str(exc)[:200], effect)
    try:
        log(line)
    except Exception:                                          # noqa: BLE001
        pass
    return line


def need(d: dict, keys, where: str, log=print) -> list:
    """必需键缺失时**说出来**（不抛异常）：返回值是缺的那些键。"""
    miss = [k for k in keys if d.get(k) in (None, "", [], {})]
    if miss and enabled():
        try:
            log("[loud] %s · 缺少键 %s · 后果：这一步会用默认值/跳过" % (where, miss))
        except Exception:                                      # noqa: BLE001
            pass
    return miss
