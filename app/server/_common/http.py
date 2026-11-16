# -*- coding: utf-8 -*-
"""统一响应 / 统一校验 / 错误类型：_ok / _fail / _api / _BadInput / _as_dict + 入参数值收敛。

从 app/server/routes_merge.py 按关注点原样切出来的（16741 行的单文件 → 按域分家），
函数体 / 注释 / 日志文案逐字节照抄，只补 import；同包之间的 import 放在文件末尾
（见文末「环上的依赖」），另有一处 __file__ 相对路径下标跟着目录层级改了。
旧行号 → 现文件 的对照表：app/server/backend_merge/MAPPING.md。
"""
from __future__ import annotations

import functools
import traceback
from fastapi import HTTPException
from fastapi.responses import JSONResponse


# ============================================================
# 统一响应 / 统一校验
# ============================================================
def _ok(**kw) -> dict:
    """成功体：{ok: true, ...}。"""
    body = {"ok": True}
    body.update(kw)
    return body


def _fail(status: int, msg: str, extra: dict = None) -> JSONResponse:
    """失败体：HTTP 4xx/5xx + {ok: false, error, detail}。

    detail 是给旧前端留的（界面里到处是 d.detail || "…"），error 是这次的约定。
    返回 JSONResponse 而不是抛 HTTPException，是为了让响应体长得一样：
    抛异常时 FastAPI 只会给 {"detail": ...}。

    extra：**结构化字段**（显存拒绝那一路用）。摊在顶层，不塞进一个子对象：
    前端要能直接读 d.need_mb / d.avail_mb / d.free_mb / d.suggest，
    不用先判断"这次错误是不是带明细的那种"。ok/error/detail 三个键不让被覆盖。
    """
    text = str(msg or "未知错误")
    body = {"ok": False, "error": text, "detail": text}
    if extra:
        for k, v in dict(extra).items():
            if k not in body:
                body[k] = v
    return JSONResponse(status_code=int(status), content=body)


class _BadInput(Exception):
    """校验失败 / 前置条件不满足。由 _api 统一转成 _fail。

    extra：要摊进响应体顶层的结构化字段（见 _fail）。显存拒绝靠它把
    need_mb / avail_mb / free_mb / suggest[] 一起带出去 —— 前端只弹一条提示、
    并且知道该建议用户干什么，全靠这几个数，不用去正则抠中文。
    """

    def __init__(self, msg, status=400, extra: dict = None):
        super().__init__(msg)
        self.status = int(status)
        self.msg = str(msg)
        self.extra = dict(extra or {})


def _api(fn):
    """把路由里抛出的东西统一成 {ok:false,error}（HTTP 状态码原样保留）。

    校验类错误抛 _BadInput，app.py 里那些 HTTPException 也顺路接住 —— 否则同一个
    /api/merge/* 下有的接口回 {"detail"}、有的回 {"ok":false}，前端得写两套。
    """

    @functools.wraps(fn)
    def _wrapped(*a, **kw):
        try:
            return fn(*a, **kw)
        except _BadInput as e:
            return _fail(e.status, e.msg, getattr(e, "extra", None))
        except HTTPException as e:
            return _fail(e.status_code, str(e.detail))
        except Exception as e:                       # noqa: BLE001
            traceback.print_exc()
            return _fail(500, "服务端内部错误：%s" % (str(e)[:200] or e.__class__.__name__))

    return _wrapped


def _as_dict(body) -> dict:
    if body is None:
        return {}
    if not isinstance(body, dict):
        raise _BadInput("请求体要是一个 JSON 对象，收到 %s" % type(body).__name__)
    return body


# ---- 数值 ----
_MISSING = object()


def _float_val(name, raw, lo, hi, default=_MISSING, unit=""):
    if raw is None or (isinstance(raw, str) and not raw.strip()):
        if default is _MISSING:
            raise _BadInput("缺少参数 %s" % name)
        return default
    try:
        v = float(str(raw).strip())
    except (TypeError, ValueError):
        raise _BadInput("「%s」要是数字，收到 %r" % (name, raw))
    if v != v or v in (float("inf"), float("-inf")):     # NaN / inf
        raise _BadInput("「%s」不是有效数字：%r" % (name, raw))
    if lo is not None and v < lo:
        raise _BadInput("「%s」不能小于 %s%s，收到 %s" % (name, lo, unit, v))
    if hi is not None and v > hi:
        raise _BadInput("「%s」不能大于 %s%s，收到 %s" % (name, hi, unit, v))
    return v


def _int_val(name, raw, lo, hi, default=_MISSING):
    if raw is None or (isinstance(raw, str) and not raw.strip()):
        if default is _MISSING:
            raise _BadInput("缺少参数 %s" % name)
        return default
    text = str(raw).strip()
    try:
        v = int(text, 10)
    except (TypeError, ValueError):
        try:                                  # 容忍 "4.0" / 4.0 这种前端塞过来的浮点串
            f = float(text)
        except (TypeError, ValueError):
            raise _BadInput("「%s」要是整数，收到 %r" % (name, raw))
        if not f.is_integer():
            raise _BadInput("「%s」要是整数，收到 %r" % (name, raw))
        v = int(f)
    if lo is not None and v < lo:
        raise _BadInput("「%s」不能小于 %d，收到 %d" % (name, lo, v))
    if hi is not None and v > hi:
        raise _BadInput("「%s」不能大于 %d，收到 %d" % (name, hi, v))
    return v


def _bool_val(raw, default=False) -> bool:
    if raw is None or (isinstance(raw, str) and not raw.strip()):
        return default
    if isinstance(raw, bool):
        return raw
    if isinstance(raw, (int, float)):
        return bool(raw)
    text = str(raw).strip().lower()
    if text in ("1", "true", "yes", "on", "y"):
        return True
    if text in ("0", "false", "no", "off", "n"):
        return False
    raise _BadInput("「%r」不是布尔值（认 1/0、true/false、on/off）" % (raw,))
