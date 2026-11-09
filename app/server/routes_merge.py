# -*- coding: utf-8 -*-
"""合成器路由（从 app.py 按路径前缀抽出）。

这一版把合成工作室（Merger / 合成工作室）里"参数进来了却没生效、坏了还不报"的地方
修掉，全部改动都在这一个文件里：

  1) 入参校验：fps / workers / encoder / hwaccel / input_dir / dfm / model / settings /
     thresh / step / xseg_onnx 一律**先校验再动后台线程**。非法值给 400 + 中文说明，
     不再掉进 500，也不再"静默退回默认值"（以前选中的模型没导出 .dfm 会被悄悄换成
     列表里第一个 .dfm，成片用的是别的脸，谁都不知道）。
  2) 缓存与并发：_merge_cache 的读写统一走 _cache_lock，每个请求先取一份快照再算，
     避免 mode / names / srcdir 分别取自两次 update 之间的混合状态；frame / merged /
     detect / analyze 对越界、缺文件、缓存过期分别给 404 / 410 / 422，不再 500。
  3) 导出状态：_merge_job["log"] 本来就是子进程 stdout+stderr 合流，失败时把尾巴挑出来
     放进 error / error_detail / log_tail；终态时 stage 归 -1、pct 归 0、msg 写"合成失败"，
     不卡在中间值。
  4) 取消幂等：重复取消照样 200；取消旗子先立（子进程还没起来也能取消），看门狗负责
     补刀和兜底收尾，终态落成 cancelled，不会永远 running。
  5) 路径统一：相对路径按仓库根解析，反斜杠/正斜杠/引号/结尾分隔符全部归一，Windows
     大小写用 os.path.normcase 比较（见 _norm_path / _same_path）。
     app/server/_paths.py 只做引擎根解析、没有通用的路径规范化函数（本次该文件只读），
     所以这里复用它的 ROOT 当相对路径基准，规范化在本地实现。
  6) 响应体统一：成功 {"ok": true, ...}；失败 {"ok": false, "error": "...", "detail": "..."}
     （detail 只是兼容旧前端 d.detail || "…" 的写法）。
     例外：/prepare/status 与 /export/status 的 ok 是"任务成功"标志 —— 前端拿 d.ok
     决定弹"合成完成"还是"合成失败"，改成恒 true 会把每次失败都报成成功；这两个接口
     多给一个 success 字段（语义同 ok），其余字段一律照旧。

  7) 视频播放与单帧实时预览（这两个是新增能力，重活都在这一个文件里）：
     · GET /api/merge/video —— FileResponse 不支持 Range（<video> 拖时间轴就废），
       所以自己解析 Range 头、自己按片段读文件：206 + Content-Range/Accept-Ranges，
       没有 Range 时 200 全量流式返回。文件不存在 404，路径越界 400。
     · POST /api/merge/preview —— 取 t 时刻一帧 → 检测 → 2D106 的 68 点 → 换脸
       → 遮罩 → 合成 → JPEG。换脸只能在 4.0 引擎的 python 里跑（MergeStudio /
       facelib 都在那边），所以这里养一个常驻 worker 子进程：JSON lines 走
       stdin/stdout，DFM 与 XSeg 只加载一次，空闲 10 分钟自动退出，崩了下次请求
       自动重启；单帧干活上限 15s，超时就杀掉 worker（ONNX 推理没法从外部打断）
       并回 504。

  8) 多模型预览：/api/merge/preview 的 body 多认一个 models: [{name, dfm, faces}]
     （faces = 该模型负责的人脸序号，按检测顺序从 0 起；单模型时等价于 faces=[]
     或缺省；老字段 model 照样能用）。worker 里最多常驻 3 个 DFM，超了先卸最久
     没用过的（LRU）再加载新的，卸载时 gc + 清 CUDA 缓存；一帧里检测到几张脸就
     按分配一张一张贴，没分配的用第 0 个。返回体多给 models_used / faces / swaps。
     /api/merge/preview/warm 同样认 models 数组，一次热一组（上限仍是 3）。

  9) 标准 DFL 项目模式（iperov 布局，本文件末尾那一大段）：
     POST /api/merge/export-project —— <项目>/<帧名> + aligned/ → 逐帧合成 →
     <项目>/merged/<帧名>（整帧）→ 按原 fps / 音轨编码成 <项目>.mp4。
     · 逐帧合成不另写一套：复用 _preview_job（检测 → 68 点 → 换脸/遮罩/合成），
       只把「从视频按时刻取帧」换成「直接读帧文件」；POST /api/merge/preview-frame
       是同一份逻辑的静态帧入口。
     · 帧序 = 源帧文件名字典序（= ffmpeg image2 的读取顺序）；aligned 的
       source_filename 只用来决定「这一帧有没有脸」和点名缺帧（认不出就报，不猜）。
     · 断点续跑：merged 里存在且非空的帧跳过（force=true 才重做）；落盘是
       .part → os.replace，不会把半张图当成已完成。
     · 编码前对账：merged 比源帧少就点名缺哪几张，并写明成片会短多少秒；
       编号有洞时先硬链接重排成 %06d 再编（image2 遇洞会停住，成片被悄悄截短）。
     · fps / 分辨率 / 音轨只认原视频（读不到就报错）；默认输出 <项目目录>.mp4，
       和原视频重名时改用 <项目目录>_merged.mp4，绝不覆盖素材。
     · 编码命令刻意不带 -shortest（音轨比画面长会把成片截短），并挂
       scale=out_range=tv，否则 JPEG 帧会让成片标成 yuvj420p。
     · POST /api/merge/reveal —— 资源管理器里定位产物（「打开所在目录」）。
"""
from __future__ import annotations

import atexit
import base64 as _b64
import collections
import functools
import hashlib
import json
import os
import queue
import re
import shutil
import subprocess
import tempfile
import threading
import time
import traceback
from email.utils import formatdate as _httpdate
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

from . import app as _app_mod
from ._paths import ROOT as _REPO_ROOT
from ._paths import engine_python as _engine_python, engine_root as _engine_root
from .app import (WORKSPACE, _dfl_project_scan, _dfljpg_dict, _get_detector,
                  _imread_any, _merge_cache, _merge_frame_image, _merge_job,
                  _merge_job_lock, _merge_prep, _merge_prep_lock,
                  _merge_prepare_worker, _merge_settings_to_config, _merge_worker,
                  _safe_dirname,
                  _video_frame_jpeg, _video_probe)

try:                                    # app.py 里那张表是唯一真源
    from .app import _MASK_MODE_LABEL as _MASK_LABELS
except Exception:                       # pragma: no cover - 表被改名时的退路
    _MASK_LABELS = {"full": 0, "dst": 1, "prd": 2, "prd×dst": 4,
                    "xseg-prd": 6, "xseg-dst": 7, "xseg-prd×dst": 8}

router = APIRouter()


# ============================================================
# onnxruntime 的 CUDA provider：先把 cublasLt / cuDNN 的目录找出来
# ------------------------------------------------------------
# ORT 的 CUDAExecutionProvider 要 cublasLt64_12.dll（还有 cuDNN 9）。这两份都在
# torch/lib 里，但那个目录默认不在进程的 DLL 搜索路径上 —— 找不到时 ORT 只留一行
# warning（Error 126）然后**静默**退回 CPU。实测差 7 倍：同一个 416 的 DFM、同样
# 1920x1706 的帧，单张合成 CUDA 78ms / CPU 648ms（前向 36ms vs 524ms）。
# 所以 import 这个模块时就把路径补上：检测器、2D106、预览 worker 都是这之后才建
# session 的。补不上也不影响可用性 —— provider 列表里本来就带着 CPU 兜底。
# ============================================================
_ORT_DLL_DIRS = []
_ORT_DLL_COOKIES = []      # add_dll_directory 的句柄：不持有的话目录会被回收掉


def _ort_dll_candidates() -> list:
    """可能要补进 DLL 搜索路径的目录（只返回真的存在的，去重、保序）。"""
    import sys as _sys
    out, seen = [], set()

    def _push(*parts):
        if not parts or not parts[0]:
            return
        try:
            p = os.path.join(*[str(x) for x in parts if str(x)])
        except Exception:
            return
        try:
            if not p or not os.path.isdir(p):
                return
            k = os.path.normcase(p)
        except OSError:
            return
        if k not in seen:
            seen.add(k)
            out.append(p)

    prefix = str(getattr(_sys, "prefix", "") or "")
    # 1) 本进程自己装的 torch（服务器 python 是 cu128 版，cublasLt64_12.dll 就在里面）
    _push(prefix, "Lib", "site-packages", "torch", "lib")
    # 2) 4.0 引擎 python 的 torch（预览 worker 跑在那个解释器里）
    try:
        _push(Path(str(_engine_root())), "python", "Lib", "site-packages", "torch", "lib")
    except Exception:
        pass
    # 3) pip 的 nvidia-* 布局（有的话，版本和 ORT 更贴）
    _push(prefix, "Lib", "site-packages", "nvidia", "cublas", "bin")
    _push(prefix, "Lib", "site-packages", "nvidia", "cudnn", "bin")
    return out


def _ensure_ort_dll_dirs() -> list:
    """把上面那些目录塞进 DLL 搜索路径；返回真加进去的目录（幂等，只做一次）。"""
    global _ORT_DLL_DIRS
    if _ORT_DLL_DIRS:
        return list(_ORT_DLL_DIRS)
    # 仓库里已经有一份现成的（_ortcuda 会挑带 cudnn64_9.dll 的那个 torch/lib）
    try:
        from ._ortcuda import add_cuda_dll_dirs as _add
        _add()
    except Exception:
        pass
    added = []
    for d in _ort_dll_candidates():
        try:
            if hasattr(os, "add_dll_directory"):
                _ORT_DLL_COOKIES.append(os.add_dll_directory(d))   # 句柄要留着
        except Exception:
            pass
        added.append(d)
    if added:
        try:
            os.environ["PATH"] = os.pathsep.join(added + [os.environ.get("PATH", "")])
        except Exception:
            pass
    _ORT_DLL_DIRS = list(added)
    if added:
        print("[merge-preview] ORT DLL 目录：%s" % "；".join(added), flush=True)
    return list(_ORT_DLL_DIRS)


try:                                    # 别为了补 DLL 把整个路由 import 弄挂
    _ensure_ort_dll_dirs()
except Exception:                       # pragma: no cover
    _ORT_DLL_DIRS = []


# ============================================================
# 统一响应 / 统一校验
# ============================================================
def _ok(**kw) -> dict:
    """成功体：{ok: true, ...}。"""
    body = {"ok": True}
    body.update(kw)
    return body


def _fail(status: int, msg: str) -> JSONResponse:
    """失败体：HTTP 4xx/5xx + {ok: false, error, detail}。

    detail 是给旧前端留的（界面里到处是 d.detail || "…"），error 是这次的约定。
    返回 JSONResponse 而不是抛 HTTPException，是为了让响应体长得一样：
    抛异常时 FastAPI 只会给 {"detail": ...}。
    """
    text = str(msg or "未知错误")
    return JSONResponse(status_code=int(status),
                        content={"ok": False, "error": text, "detail": text})


class _BadInput(Exception):
    """校验失败 / 前置条件不满足。由 _api 统一转成 _fail。"""

    def __init__(self, msg, status=400):
        super().__init__(msg)
        self.status = int(status)
        self.msg = str(msg)


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
            return _fail(e.status, e.msg)
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


# ---- 路径 ----
def _norm_path(raw):
    """把界面传进来的路径收敛成一个可用的 Path（做存在性判断前必须过这一步）。

    * 空串 / None / 只有空白 -> None
    * 去掉首尾空白和包裹的引号（从资源管理器拖进来常带）
    * 反斜杠、正斜杠都当分隔符（Windows 下 Path 两种都认）
    * 相对路径按仓库根解析（和 app.py 顶部"不带盘符的路径相对 4.1 根"的约定一致）
    * normpath 收掉 . / .. 和结尾多余的分隔符
    """
    if raw is None:
        return None
    s = str(raw).strip().strip('"').strip("'").strip()
    if not s:
        return None
    s = os.path.expandvars(os.path.expanduser(s))
    p = Path(s)
    if not p.anchor:                    # 没盘符也没根 = 相对路径
        p = _REPO_ROOT / p
    return Path(os.path.normpath(str(p)))


def _same_path(a, b) -> bool:
    """Windows 上同一个路径大小写可以不一样，比较前统一 normcase。"""
    if a is None or b is None:
        return False
    try:
        return os.path.normcase(str(a)) == os.path.normcase(str(b))
    except Exception:
        return False


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


# ---- 合成参数白名单 ----
_VIDEO_EXT = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv", ".wmv", ".m4v",
              ".mpg", ".mpeg", ".ts", ".m2ts", ".m2v", ".vob", ".ogv", ".3gp"}
_IMAGE_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
# MergeStudio 的 -c:v 只认这些（其余名字 ffmpeg 会直接报 Unknown encoder）
_ENCODERS = ("h264_nvenc", "hevc_nvenc", "h264_qsv", "hevc_qsv",
             "h264_amf", "hevc_amf", "libx264", "libx265")
# 4.0 的 hwaccel 只有 auto / ""（关）两种语义，_merge_runner 也是这么折的
_HWACCEL = {"auto": "auto", "on": "auto", "1": "auto", "true": "auto",
            "off": "off", "none": "off", "0": "off", "false": "off", "": "off"}
# MergeStudio.core.config.mode_dict —— 但**只收融合方式**。
# original / raw-rgb / raw-predict / debug 是「视图」（看原图 / 看原始预测 / 看调试
# 信息），不是融合方式：以前混在这张表里，于是下拉里既有 overlay 又有 debug，
# 传 mode=debug 后端还默默按 overlay 出片（_merge_runner 的 _MODE_EFFECTIVE 根本
# 不认它）—— 用户选什么都不会变，等于假选项。
# 现在：mode/blend 只认这 4 个，其余一律 400；视图走独立参数
#   raw=1（原图，不合成） / debug=1（合成 + 遮罩/边界调试叠加）。
_MERGE_MODES = ("overlay", "hist-match", "seamless", "seamless-hist-match")
# 老前端/老脚本可能还在往 mode 里塞这些「视图名」：明确 400 并说清楚该用哪个参数，
# 不许静默折成 overlay（那正是「选了没反应」的老 bug 的温床）。
_MERGE_VIEW_MODES = ("original", "raw-rgb", "raw-predict", "debug")
# MergeStudio.core.config.ctm_dict（+ "sot" 这个直白写法：worker 的 ct_map 把它
# 折成 ctm_dict 里那一档，见 _PREVIEW_WORKER_SRC 的 ct_map["sot"]）
_COLOR_TRANSFER = ("none", "rct", "lct", "mkl", "mkl-m", "idt", "idt-m",
                   "sot", "sot-m", "mix-m", "mt", "lut")
# 接口契约里的两条主参数（★ 默认值就是「不传 settings」时的结果，改这里 = 改默认）
_CT_MODES = ("mt", "rct", "lct", "mkl", "sot")
# 以前这里还有一份只写两个值的 _BLEND_MODES（seamless / overlay），谁都没用 ——
# 留着只会让人以为 hist-match / seamless-hist-match 不合法。真白名单就一个：_MERGE_MODES。
DEFAULT_CT_MODE = "mt"
DEFAULT_BLEND = "seamless"
_IMG_FORMATS = {"jpg": "jpg", "jpeg": "jpg", "png": "png"}

# ---- 检测器白名单 ----
# 4.1 只有这两个：tinymog（默认，快）/ retinaface_10g（重，召回好）。
# retinaface_500m 这个模型不存在，已经从所有列表里删掉；再点到它要明确 400，
# 不许"就近退一个能用的顶上"（静默回退曾让用户以为在跑 500m）。
_DETECTORS = ("tinymog", "retinaface_10g")
_DET_ALIAS_API = {
    "tinymog": "tinymog",
    "retinaface_10g": "retinaface_10g", "retinaface10g": "retinaface_10g",
    "det_10g": "retinaface_10g", "retinaface": "retinaface_10g",
}
_REMOVED_DETECTORS = ("retinaface_500m", "retinaface500m", "det_500m",
                      "retinaface_500m.onnx", "det_500m.onnx")
_DEFAULT_DETECTOR = "tinymog"      # ★ 默认检测器（所有 entrypoint 共用这一个常量）


def _encoder_val(raw) -> str:
    key = str(raw or "").strip()
    if not key:
        return "h264_nvenc"
    low = key.lower()
    if low not in _ENCODERS:
        raise _BadInput("「encoder」只认 %s，收到 %r" % (" / ".join(_ENCODERS), key))
    return low


def _hwaccel_val(raw) -> str:
    if raw is None:
        return "auto"
    key = str(raw).strip().lower()
    if key not in _HWACCEL:
        raise _BadInput("「hwaccel」只认 auto（开）/ off（关），收到 %r" % (raw,))
    return _HWACCEL[key]


def _mask_mode_val(raw, default=6) -> int:
    if raw is None or (isinstance(raw, str) and not raw.strip()):
        return default
    if isinstance(raw, bool):
        raise _BadInput("「mask_mode」要是 0~9 的整数，收到 %r" % (raw,))
    text = str(raw).strip()
    if text.isdigit():
        return _int_val("mask_mode", text, 0, 9, default)
    key = text.lower().replace("*", "×")
    if key in _MASK_LABELS:
        return int(_MASK_LABELS[key])
    raise _BadInput("「mask_mode」只认 0~9 或 %s，收到 %r"
                    % (" / ".join(sorted(_MASK_LABELS)), raw))


def _settings_val(raw) -> dict:
    """校验并规范化合成参数。

    以前 settings 原样丢给 _merge_settings_to_config()，非法值要等子进程跑到一半
    才炸（erode_mask 传 "abc" 会在 _merge_runner 里 int() 崩）。这里先翻译成
    MergeStudio 的键名并逐个校验，_merge_settings_to_config 直接取到就用 ——
    参数才真正生效。
    """
    if raw is None:
        raw = {}
    if isinstance(raw, str):                       # 有些前端会把 settings 二次 encode
        try:
            raw = json.loads(raw or "{}")
        except Exception:
            raise _BadInput("settings 不是合法 JSON：%s" % str(raw)[:80])
    if not isinstance(raw, dict):
        raise _BadInput("settings 要是一个 JSON 对象，收到 %s" % type(raw).__name__)

    out = dict(raw)                                # 保留用户原值，方便回显

    def _first(*names):
        """按顺序取第一个非空值（两代键名：新的在前）。"""
        for n in names:
            v = raw.get(n)
            if v is None or (isinstance(v, str) and not v.strip()):
                continue
            return v
        return None

    # ---- blend（= MergeStudio 的 mode）----
    # 契约字段叫 blend，老写法叫 mode；认一个就够，两个都给时以 blend 为准。
    # 缺省 = seamless。**只认 4 个融合方式**（白名单见 _MERGE_MODES）。
    mode = str(_first("blend", "mode") or DEFAULT_BLEND).strip().lower() or DEFAULT_BLEND
    if mode in _MERGE_VIEW_MODES:
        raise _BadInput(
            "settings.blend 只认 4 个融合方式：%s；收到 %r —— 这是「视图」，不是融合方式，"
            "视图请用 raw=1（原图）/ debug=1（调试）"
            % (" / ".join(_MERGE_MODES), mode))
    if mode not in _MERGE_MODES:
        raise _BadInput("settings.blend 只认 %s，收到 %r"
                        % (" / ".join(_MERGE_MODES), raw.get("blend", raw.get("mode"))))
    out["blend"] = mode
    out["mode"] = mode                             # 老键名照样回显，前端不用改
    out["mask_mode"] = _mask_mode_val(raw.get("mask_mode"), 6)

    # ---- ct_mode（= MergeStudio 的 color_transfer_mode）----
    ct = str(_first("ct_mode", "color_transfer_mode", "color_transfer")
             or DEFAULT_CT_MODE).strip().lower() or DEFAULT_CT_MODE
    if ct not in _COLOR_TRANSFER:
        raise _BadInput("settings.ct_mode 只认 %s（也认 %s），收到 %r"
                        % (" / ".join(_CT_MODES), " / ".join(_COLOR_TRANSFER),
                           raw.get("ct_mode", raw.get("color_transfer"))))
    out["ct_mode"] = ct
    out["color_transfer_mode"] = ct
    out["color_transfer"] = ct

    # ---- 遮罩腐蚀 / 羽化（mask_erode / mask_blur 是契约字段）----
    erode = _first("mask_erode", "erode_mask_modifier", "erode_mask")
    blur = _first("mask_blur", "blur_mask_modifier", "blur_mask")
    out["mask_erode"] = _int_val("settings.mask_erode", erode, -100, 100, 0)
    out["mask_blur"] = _int_val("settings.mask_blur", blur, -100, 100, 0)
    out["erode_mask_modifier"] = out["mask_erode"]
    out["blur_mask_modifier"] = out["mask_blur"]

    fmt = str(raw.get("image_format") or "jpg").strip().lower()
    if fmt not in _IMG_FORMATS:
        raise _BadInput("settings.image_format 只认 jpg / png，收到 %r"
                        % (raw.get("image_format"),))
    out["image_format"] = _IMG_FORMATS[fmt]

    if raw.get("face_margin") is not None:
        out["face_margin"] = _float_val("settings.face_margin", raw.get("face_margin"),
                                        -4.0, 4.0, None)
    return out


# ============================================================
# 缓存快照（_merge_cache 是 app.py 里的全局字典，多请求共用）
# ============================================================
_cache_lock = threading.RLock()
_det_load_lock = threading.Lock()      # 检测器懒加载：别让并发请求各建一份
_det_run_lock = threading.Lock()       # det.thresh + det.detect 是共享对象上的两步操作


def _cache_snap() -> dict:
    """取一份快照。

    dict.update() 本身在 C 层是一口气做完的，但"读 mode、再读 names、再读 srcdir"
    是三次独立读，中间被另一次 /open 插进来就会拼出不存在的东西。所以每请求只取
    一次快照，之后全部读快照。
    """
    with _cache_lock:
        snap = dict(_merge_cache)
    snap.setdefault("mode", "")
    snap.setdefault("names", [])
    snap.setdefault("path", "")
    snap.setdefault("srcdir", "")
    snap.setdefault("dir", "")
    snap.setdefault("frames", 0)
    snap.setdefault("fps", 0)
    return snap


def _cache_set(**kw):
    with _cache_lock:
        _merge_cache.update(kw)


def _cache_mode(snap: dict) -> str:
    mode = str(snap.get("mode") or "")
    if mode:
        return mode
    if snap.get("names"):
        return "project"                     # 兼容这次改动之前写进去的缓存
    if snap.get("path"):
        return "video"
    return ""


def _scan_first(dirp, exts, limit=1):
    """目录里前 limit 个符合扩展名的文件名（排序后）。"""
    out = []
    try:
        with os.scandir(dirp) as it:
            for e in it:
                try:
                    if e.is_file() and Path(e.name).suffix.lower() in exts:
                        out.append(e.name)
                        if len(out) >= limit * 4:
                            break
                except OSError:
                    continue
    except OSError:
        return []
    return sorted(out)[:limit]


def _has_dfl_meta(dirp, tries=8) -> bool:
    """抽样看这个目录里的图带不带 DFLJPG 元数据（landmarks）。

    合成完全靠 source_filename / landmarks 定位人脸，没有元数据的对齐集必然在
    子进程里以"没有任何对齐脸能对上帧"收场 —— 在这儿就说清楚。
    """
    for n in _scan_first(dirp, {".jpg", ".jpeg"}, tries):
        try:
            if _dfljpg_dict(Path(dirp) / n).get("landmarks") is not None:
                return True
        except Exception:
            continue
    return False


# ============================================================
# 合成子进程的收尾 / 失败信息（stderr 尾巴）
# ============================================================
_ERR_HINT = re.compile(r"(Traceback|Error|error\b|Exception|failed|失败|"
                       r"No such|not found|[Cc]annot|CUDA|out of memory|内存|"
                       r"ModuleNotFound|DLL)", re.I)


def _log_tail(log, n=20):
    lines = [str(x).rstrip() for x in (log or []) if str(x).strip()]
    return lines[-n:]


def _brief_failure(log, n=4) -> str:
    """从 stdout+stderr 合流日志里挑一条能当错误摘要的行。

    子进程硬崩（DLL 缺失 / CUDA OOM / python 直接 traceback 退出）时不会打
    "@@ERR"，父进程以前只能说一句"进程退出码 1"，用户完全不知道为什么。
    """
    tail = _log_tail(log, 60)
    if not tail:
        return ""
    hits = [ln for ln in tail if _ERR_HINT.search(ln)]
    pick = (hits or tail)[-n:]
    return " ｜ ".join(x.strip()[:200] for x in pick)[:600]


def _finalize():
    """合成收尾：把 running / ok / stage / pct / msg 收敛成终态。

    _merge_worker 正常结束时只写 running/ok（成功时补 pct=100/msg），失败时
    pct、stage、msg 全停在"跑到一半"的样子，前端就会把一次崩掉的任务显示成
    47% 还在跑。这里统一收尾。
    """
    with _merge_job_lock:
        _merge_job["running"] = False
        _merge_job["finished_at"] = time.time()
        cancelled = bool(_merge_job.get("cancelled"))
        ok = bool(_merge_job.get("ok"))
        log = list(_merge_job.get("log") or [])
        _merge_job["error_detail"] = "\n".join(_log_tail(log, 30))[:4000]
        if ok:
            _merge_job["pct"] = 100
            if _merge_job.get("stage") in (None, -1):
                _merge_job["stage"] = 6          # 4.0 的最后一步是编码
            _merge_job["msg"] = ("合成完成（取消命令来晚了，成片已生成）" if cancelled
                                 else "合成完成")
            return
        _merge_job["pct_at_failure"] = int(_merge_job.get("pct") or 0)
        _merge_job["pct"] = 0                    # 别让调用方把失败读成"还在 47%"
        _merge_job["stage"] = -1
        if cancelled:
            _merge_job["msg"] = "已取消"
            _merge_job["error"] = "已取消合成（用户手动停止）"
            return
        brief = _brief_failure(log)
        err = str(_merge_job.get("error") or "").strip()
        if not err:
            err = brief or "合成失败：子进程没有给出原因（见 error_detail）"
        elif brief and brief not in err:
            err = err + " ｜ " + brief
        _merge_job["msg"] = "合成失败"
        _merge_job["error"] = err[:1200]


def _run_export(input_dir, out_mp4, aligned_dir, dfm, settings, mask_onnx,
                fps, workers, encoder, hwaccel):
    """合成线程。

    _merge_worker 在起子进程之前就会抛（找不到引擎 python / MergeStudio / 桥脚本），
    以前线程里抛出去没人接：状态永远停在 running，前端一直转圈、取消也点不动。
    """
    try:
        _merge_worker(input_dir, out_mp4, aligned_dir, dfm, settings, mask_onnx,
                      fps, workers, encoder, hwaccel)
    except Exception as e:                       # noqa: BLE001
        traceback.print_exc()
        with _merge_job_lock:
            _merge_job["ok"] = False
            if not _merge_job.get("error"):
                _merge_job["error"] = "合成没起来：%s" % (str(e)[:300] or e.__class__.__name__)
    _finalize()


def _kill_proc_tree(proc) -> bool:
    """停掉合成子进程和它拉起的 ffmpeg（Windows 上 terminate 只杀父进程）。"""
    if proc is None:
        return False
    try:
        if proc.poll() is not None:
            return False
    except Exception:
        return False
    try:
        import psutil
        children = []
        try:
            children = psutil.Process(proc.pid).children(recursive=True)
        except Exception:
            children = []
        for c in children:
            try:
                c.terminate()
            except Exception:
                pass
        try:
            proc.terminate()
        except Exception:
            pass
        for c in children:
            try:
                c.wait(timeout=3)
            except Exception:
                try:
                    c.kill()
                except Exception:
                    pass
        return True
    except Exception:
        try:
            proc.terminate()
            return True
        except Exception:
            return False


_cancel_guard = threading.Lock()
_cancel_watching = False                        # 看门狗只留一个，重复取消不叠加线程


def _ensure_cancel_watch():
    global _cancel_watching
    with _cancel_guard:
        if _cancel_watching:
            return
        _cancel_watching = True
    threading.Thread(target=_cancel_watch, daemon=True, name="merge-cancel").start()


def _cancel_watch():
    """取消的看门狗：进程还没起来就等它起来补刀，worker 卡住就兜底收尾。

    取消请求落地时子进程可能还没 Popen 出来（线程刚起、还在 import MergeStudio），
    以前这种情况直接返回 "no running export"，用户点了取消什么也没发生。
    """
    global _cancel_watching
    try:
        deadline = time.time() + 60
        killed_pid = None
        while time.time() < deadline:
            if not _merge_job.get("running"):
                break
            proc = getattr(_app_mod, "_merge_proc", None)
            try:
                if (proc is not None and proc.poll() is None
                        and getattr(proc, "pid", None) != killed_pid):
                    killed_pid = getattr(proc, "pid", None)
                    _kill_proc_tree(proc)
            except Exception:
                pass
            time.sleep(0.3)
        with _merge_job_lock:
            if not _merge_job.get("running"):
                return                           # _finalize 已经写好终态
            if _merge_job.get("cancelled"):
                _merge_job.update({"running": False, "ok": False, "stage": -1,
                                   "pct_at_failure": int(_merge_job.get("pct") or 0),
                                   "pct": 0, "msg": "已取消",
                                   "error": (_merge_job.get("error")
                                             or "已取消合成（用户手动停止）"),
                                   "finished_at": time.time()})
    finally:
        with _cancel_guard:
            _cancel_watching = False


# ============================================================
# 路由
# ============================================================
@router.get("/api/merge/videos")
@_api
def merge_videos(dir: str = ""):
    """列出视频文件。dir 给了就只扫它，否则扫工作区。"""
    root = _norm_path(dir)
    if root is not None and not root.is_dir():
        raise _BadInput("目录不存在：%s" % root, 404)
    roots = [root] if root is not None else [Path(WORKSPACE)]

    out, seen = [], set()
    limit, budget, truncated = 200, 20000, False
    for r in roots:
        if not r.is_dir():
            continue
        try:
            entries = sorted(r.rglob("*"))
        except OSError:
            entries = []
        for f in entries:
            budget -= 1
            if budget <= 0 or len(out) >= limit:
                truncated = True
                break
            try:
                if not f.is_file() or f.suffix.lower() not in _VIDEO_EXT:
                    continue
                # 合成产物 / 缓存目录不作为素材列出来（否则列表里全是自己的输出）
                if f.name.endswith("_merged.mp4") or ".merge_" in str(f.parent):
                    continue
                key = os.path.normcase(str(f))
                if key in seen:
                    continue
                seen.add(key)
                out.append({"path": str(f), "name": f.name, "dir": str(r),
                            "mb": round(f.stat().st_size / 1048576, 1)})
            except OSError:
                continue
    return _ok(dir=str(roots[0]), videos=out, count=len(out), truncated=truncated)


@router.post("/api/merge/prepare")
@_api
def merge_prepare(body: dict):
    """视频 → 可合成项目（抽帧 + 对齐）。跑在后台，进度用 /status 拉。"""
    body = _as_dict(body)
    video = _norm_path(body.get("video"))
    if video is None:
        raise _BadInput("缺少 video：要一个视频文件路径")
    if not video.is_file():
        raise _BadInput("视频不存在：%s" % video, 404)
    if video.suffix.lower() not in _VIDEO_EXT:
        raise _BadInput("不是视频文件：%s（只认 %s）"
                        % (video.suffix or "无扩展名", " ".join(sorted(_VIDEO_EXT))))

    target = _norm_path(body.get("dir"))
    if target is None:
        target = Path(WORKSPACE) / "merge" / _safe_dirname(video.stem)
    if target.exists() and not target.is_dir():
        raise _BadInput("dir 是一个已存在的文件，要目录：%s" % target)
    _guard_prepare_target(target)

    with _merge_prep_lock:
        if _merge_prep.get("running"):
            raise _BadInput("正在准备另一个视频（%s）—— 等它跑完再试"
                            % (_merge_prep.get("video") or "?"), 409)
        # 置 running 和 project 放在同一把锁里：否则 /status 可能先看到 running=True
        # 却读到上一轮的 project。
        _merge_prep.update({"running": True, "ok": None, "error": "", "pct": 0,
                            "msg": "启动…", "stage": "准备", "frames": 0,
                            "aligned": 0, "video": str(video),
                            "project": str(target), "started_at": time.time()})

    threading.Thread(target=_merge_prepare_worker, args=(video, target),
                     daemon=True, name="merge-prepare").start()
    return _ok(project=str(target), video=str(video))


def _guard_prepare_target(target: Path):
    """准备过程会清空目标目录里的 *.jpg 和 aligned/*.jpg（app._merge_prepare_worker）。

    所以先确认这个目录要么空着，要么就是上次的产物（000001.jpg / aligned/），
    免得手滑指向 data_dst 把用户的帧清光。
    """
    if not target.is_dir():
        return
    try:
        entries = list(os.scandir(target))
    except OSError as e:
        raise _BadInput("目标目录读不了：%s（%s）" % (target, e))
    bad = []
    for e in entries:
        try:
            if e.is_dir():
                if e.name.lower() != "aligned":
                    bad.append(e.name + "/")
                continue
        except OSError:
            continue
        low = e.name.lower()
        if low.endswith(".mp4"):
            continue
        if low.endswith((".jpg", ".jpeg", ".png")):
            if not re.fullmatch(r"\d{4,8}(_\d+)?\.(jpg|jpeg|png)", low):
                bad.append(e.name)
    if bad:
        raise _BadInput("目标目录里已经有别的东西（%s）—— 准备过程会清掉它的 *.jpg，"
                        "换个空目录或先备份" % "、".join(sorted(bad)[:5]))


@router.get("/api/merge/prepare/status")
@_api
def merge_prepare_status():
    with _merge_prep_lock:
        snap = dict(_merge_prep)
    snap["running"] = bool(snap.get("running"))
    snap["success"] = snap.get("ok")          # 语义同 ok（ok 是任务标志，前端在用）
    return snap


@router.post("/api/merge/open")
@_api
def merge_open(body: dict):
    """打开一个 DFL 项目目录（data_dst / aligned / 裸帧目录）或一个视频。"""
    body = _as_dict(body)
    p = _norm_path(body.get("path"))
    if p is None:
        raise _BadInput("缺少 path：要一个 DFL 项目目录或视频文件")

    # 目录 = DFL 项目（data_dst / aligned / 裸帧目录）
    if p.is_dir():
        names, srcdir, kind = _dfl_project_scan(p)
        if not names:
            raise _BadInput("目录里没有可用的图（.jpg）：%s" % p, 404)
        sample = names[:24]
        lm = 0
        for n in sample:
            try:
                if _dfljpg_dict(Path(srcdir) / n).get("landmarks") is not None:
                    lm += 1
            except Exception:
                continue
        fps = _float_val("fps", body.get("fps"), 0.0, 240.0, 0.0)
        _cache_set(mode="project", dir=str(p), srcdir=srcdir, names=names,
                   path="", list=[], frames=len(names), fps=fps, w=0, h=0,
                   stale=False, opened_at=time.time())
        return _ok(path=str(p), name=p.name, mode="project", kind=kind,
                   srcdir=srcdir, frames=len(names), fps=fps,
                   width=0, height=0, with_landmarks=lm, sampled=len(sample))

    if not p.exists():
        raise _BadInput("路径不存在：%s" % p, 404)
    if not p.is_file():
        raise _BadInput("既不是目录也不是文件：%s" % p)
    if p.suffix.lower() not in _VIDEO_EXT:
        raise _BadInput("不是视频文件：%s（只认 %s）"
                        % (p.suffix or "无扩展名", " ".join(sorted(_VIDEO_EXT))))
    try:
        frames, fps, w, h = _video_probe(p)
    except Exception as e:
        raise _BadInput("这个视频解析不了（ffmpeg/imageio）：%s（%s）"
                        % (p.name, str(e)[:160]))
    if not frames or int(frames) <= 0 or not fps or float(fps) <= 0:
        raise _BadInput("视频里读不出帧（frames=%s, fps=%s）：%s"
                        % (frames, fps, p.name))
    _cache_set(mode="video", path=str(p), dir="", list=[], names=[],
               frames=int(frames), fps=float(fps), w=int(w), h=int(h),
               stale=False, opened_at=time.time())
    return _ok(path=str(p), name=p.name, mode="video", frames=int(frames),
               fps=float(fps), width=int(w), height=int(h))


def _check_cache_ready(snap: dict):
    """缓存还在不在：项目目录 / 视频文件被删了就说"过期"，别让下游 500。"""
    mode = _cache_mode(snap)
    if mode == "project":
        srcdir = str(snap.get("srcdir") or "")
        if not srcdir or not Path(srcdir).is_dir():
            _cache_set(stale=True)
            raise _BadInput("项目目录不在了：%s —— 重新打开一次" % (srcdir or "?"), 410)
        return mode
    if mode == "video":
        path = str(snap.get("path") or "")
        if not path or not Path(path).is_file():
            _cache_set(stale=True)
            raise _BadInput("视频文件不见了：%s —— 可能被移动/删除了，重新打开一次"
                            % (path or "?"), 410)
        return mode
    raise _BadInput("还没有打开素材（先 /api/merge/open 或 /api/merge/prepare）", 409)


def _check_idx(snap: dict, idx: int) -> int:
    if idx < 0:
        raise _BadInput("帧号不能是负数：%d" % idx)
    mode = _cache_mode(snap)
    if mode == "project":
        total = len(snap.get("names") or [])
    else:
        total = int(snap.get("frames") or 0)
    if total and idx >= total:
        raise _BadInput("帧号越界：%d，这里只有 %d 帧（0~%d）" % (idx, total, total - 1), 404)
    return idx


# ============================================================
# 带参数的整帧合成：GET/POST /api/merge/frame/{idx} 与 /api/merge/merged/{idx}
# ------------------------------------------------------------
# 用户报的 BUG：「右侧改参数，画面不动」。
# 根因不是 worker 不认参数 —— /api/merge/preview 一直是通的 —— 而是这两个接口
# **根本没把参数带进合成链路**：
#   · /frame  原样把源帧编成 JPEG 就返回（settings 一个字节都没用上）
#   · /merged 从「上一次导出的成片」里抠一帧（那是导出那一刻的参数，改滑块当然不动）
# 现在两条都走下面 _frame_payload()：帧 → 检测 → 68 点 → worker 换脸/遮罩/合成
# （和 /preview 完全同一份 _preview_job），settings / model / models 逐项透传，
# 且在**同一帧同一模型**下只改一项，像素就必须变。
#
# 兼容：
#   · 不带 settings = 用当前默认值（ct_mode=mt、blend=seamless、detector=tinymog），
#     和显式传同一组默认参数**逐字节一致**（同一份 _settings_val → 同一个 cfg）
#   · raw=1（或 compose=0）= 老行为：原样返回源帧（界面「原图 / 检测」两视图用）
#   · 没有可用 .dfm 时不报错，退化成原帧并在 note 里说明（老调用不会突然 4xx）
# ============================================================
_COMPOSE_CACHE = collections.OrderedDict()      # 整帧合成结果（含 data URL）
_COMPOSE_CACHE_LOCK = threading.Lock()
_COMPOSE_CACHE_MAX = 6                          # 拖滑块来回切参数时命中率最高的几个


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


def _source_frame(snap: dict, mode: str, idx: int):
    """按当前打开的东西取第 idx 帧的**原图**（BGR，不缩放）。返回 (图像, 显示名)。

    视频模式走 _grab_frame_at：它复用常开句柄 + 上一帧缓存，比每次重开
    cv2.VideoCapture / 起 ffmpeg 都快，而且和 /preview 用的是同一条取帧路径
    （同一帧走 /preview 和走 /frame，合成输入是同一份像素）。
    """
    if mode == "project":
        names = snap.get("names") or []
        if not names:
            raise _BadInput("还没有打开项目", 409)
        fp = Path(str(snap.get("srcdir"))) / names[idx]
        if not fp.is_file():
            raise _BadInput("图片不见了：%s" % fp, 404)
        img = _imread_any(fp)
        if img is None:
            raise _BadInput("这张图解不开（不是有效图片？）：%s" % fp.name, 422)
        return img, names[idx]

    path = str(snap.get("path") or "")
    if not path:
        raise _BadInput("还没有打开视频", 409)
    try:
        fps = float(snap.get("fps") or 0.0) or 30.0
    except (TypeError, ValueError):
        fps = 30.0
    try:
        img = _grab_frame_at(path, float(idx) / fps, None)
    except _BadInput:
        raise
    except Exception as e:                     # noqa: BLE001
        raise _BadInput("取帧失败（ffmpeg）：%s" % str(e)[:200], 500)
    if img is None:
        raise _BadInput("这一帧读不出来：%d" % idx, 422)
    return img, os.path.basename(path)


def _encode_frame(img, maxw: int, fmt: str, quality: int) -> bytes:
    """整帧 → 字节。先按 maxw 缩（接口一直只用缩略图给界面看），再编码。"""
    import cv2
    h0, w0 = img.shape[:2]
    if w0 > maxw:
        img = cv2.resize(img, (maxw, int(h0 * maxw / w0)), interpolation=cv2.INTER_AREA)
    if str(fmt or "jpg").strip().lower() == "png":
        ok, buf = cv2.imencode(".png", img, [int(cv2.IMWRITE_PNG_COMPRESSION), 3])
    else:
        ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        raise _BadInput("这一帧编码失败", 500)
    return buf.tobytes()


def _source_ident(snap: dict, mode: str, name: str) -> list:
    """「这一帧属于哪份素材」的身份段（合成缓存 / 自动分配缓存共用同一份口径）。"""
    if mode == "project":
        return ["project", str(snap.get("srcdir")), str(name)]
    return ["video"] + list(_grab_key(str(snap.get("path") or "")))


def _compose_sig(snap: dict, mode: str, idx: int, name: str, cfg: dict, dfm: str,
                 models, xseg: str, detector: str, thresh: float,
                 fmt: str, quality: int, maxw: int, no_swap=None,
                 debug: bool = False) -> str:
    """合成结果的身份：**参数一个都不能漏**，漏了就是"改了参数画面不动"的老 bug。

    debug 也是身份的一部分：同一帧同一组参数，「合成」和「调试」是两张不同的图，
    缓存键不分家的话切视图会拿到上一张（看着像"点了没反应"）。
    """
    ident = _source_ident(snap, mode, name)
    try:
        tail = json.dumps([int(idx), cfg, str(dfm), str(xseg), str(detector),
                           round(float(thresh), 4),
                           [[m["name"], m["dfm"], list(m["faces"])] for m in (models or [])],
                           str(fmt), int(quality), int(maxw),
                           sorted(int(x) for x in (no_swap or ())),
                           bool(debug)],
                          sort_keys=True, default=str)
    except Exception:                          # noqa: BLE001
        tail = repr([idx, cfg, dfm, xseg, detector, thresh, fmt, quality, maxw])
    return json.dumps(ident, default=str) + "|" + tail


def _compose_cached(sig: str):
    if not sig:
        return None
    with _COMPOSE_CACHE_LOCK:
        hit = _COMPOSE_CACHE.get(sig)
        if hit is not None:
            _COMPOSE_CACHE.move_to_end(sig)
    return dict(hit) if hit is not None else None


def _compose_store(sig: str, resp: dict):
    if not sig:
        return
    with _COMPOSE_CACHE_LOCK:
        _COMPOSE_CACHE[sig] = dict(resp)
        _COMPOSE_CACHE.move_to_end(sig)
        while len(_COMPOSE_CACHE) > _COMPOSE_CACHE_MAX:
            _COMPOSE_CACHE.popitem(last=False)


def _merge_query_into(payload: dict, **kw):
    """query 参数并进 payload：**body 优先**（GET 没 body 时全靠它）。"""
    for k, v in kw.items():
        if v is None:
            continue
        if isinstance(v, str) and not v.strip():
            continue
        if k in payload and payload[k] not in (None, ""):
            continue
        payload[k] = v


# ============================================================
# 自动分配：人脸 → 组 → dfm（请求没给 models 时）
# ------------------------------------------------------------
# 产品语义（用户定的）：组里的人脸 = 那个组的 dfm。所以合成时「哪张脸用哪个 dfm」
# 不该由左栏选中的那一个模型决定，而应该由**这张脸更像哪个组里的人脸**决定：
#   每张检测到的脸 → ArcFace embedding → 和每个组的均值向量比余弦 → 取最大且 >= 阈值。
# 匹配不上（< 阈值）的脸**不换**（保持原像素），绝不兜给"第 0 个模型"——
# 那等于把别人的人脸换上去，比不换更糟。
#
# 只读不写：这里绝不动 index.json，也不改 .npy（补算走 _face_group 的自己那条路）。
# ============================================================
_AUTO_PLAN_CACHE = collections.OrderedDict()   # 帧身份 → 自动分配结果（拖滑块时不用重算）
_AUTO_PLAN_MAX = 6
_AUTO_EMB_LIMIT = 40          # 一次自动分配最多临时补算多少张老脸的特征


def _auto_cache_get(key: str):
    with _COMPOSE_CACHE_LOCK:
        hit = _AUTO_PLAN_CACHE.get(key)
        if hit is not None:
            _AUTO_PLAN_CACHE.move_to_end(key)
    return dict(hit) if hit is not None else None


def _auto_cache_put(key: str, plan: dict):
    if not key:
        return
    with _COMPOSE_CACHE_LOCK:
        _AUTO_PLAN_CACHE[key] = dict(plan)
        _AUTO_PLAN_CACHE.move_to_end(key)
        while len(_AUTO_PLAN_CACHE) > _AUTO_PLAN_MAX:
            _AUTO_PLAN_CACHE.popitem(last=False)


def _auto_library(payload: dict):
    """请求 → 人脸库目录（project > faces_project > 默认库）。拿不到返回 None。

    自动分配是锦上添花：库路径非法/越界就当"没有库"，照旧走单模型老路径，
    绝不因为一个可选的 project 字段把整帧打成 400。
    """
    from . import _faces as _store
    raw = payload.get("project")
    if raw in (None, ""):
        raw = payload.get("faces_project")
    try:
        return _store.resolve_library(str(raw or "").strip())
    except Exception:                                     # noqa: BLE001
        return None


def _dfm_by_name(name: str):
    """模型名 → workspace/model/<name>.dfm；不存在返回 None（组可能已 orphan）。"""
    nm = str(name or "").strip()
    if nm.lower().endswith(".dfm"):
        nm = nm[:-4]
    if not nm or "/" in nm or "\\" in nm or ".." in nm:
        return None
    path = Path(WORKSPACE) / "model" / (nm + ".dfm")
    return path if path.is_file() else None


def _detect_boxes(frame, detector: str, thresh: float) -> list:
    """整帧检测：和 _preview_job 里那一步**完全同一条**（同一把锁、同一个 thresh）。"""
    det = _detector(detector)
    with _det_run_lock:
        det.thresh = float(thresh)
        try:
            return det.detect(frame) or []
        except Exception as e:                            # noqa: BLE001
            raise _BadInput("检测失败：%s" % (str(e)[:200],), 500)


def _face_vector_from_frame(frame, det_box, box_xyxy):
    """这一帧里那张脸的特征：68 点 → 512 仿射裁 → ArcFace。拿不到返回 None。"""
    from . import _face_group as _fg
    lm = _frame_landmarks(frame, det_box, box_xyxy)
    if lm is None:
        return None
    try:
        import numpy as np
        from ._align import align_face_68
        crop, _meta = align_face_68(frame, np.asarray(lm, dtype=np.float64),
                                    box_xyxy, 512)
        if crop is None:
            return None
        return _fg.embed_crop(crop)
    except Exception:                                     # noqa: BLE001
        return None


def _auto_face_plan(payload: dict, frame, snap: dict, mode: str, name: str,
                    idx: int, detector: str, thresh: float, boxes=None):
    """按人脸库自动决定每张脸用哪个 dfm。返回 None = 这次不走自动。

    boxes 可选：调用方**已经为这一帧检测过一次**时把结果递进来，这里就不再检测
    第二遍（那一份检测同时还要用来判「这一帧到底有没有脸」，重检一遍纯属浪费）。
    boxes=None 仍然自己检测，老调用方（/api/merge/face-debug）行为一个字都不变。

    None 的三种情形（都退回老的单模型路径，行为一个字都不变）：
      · 库不存在 / 库里的组都没有 dfm；
      · 组里一张有特征的脸都没有（模型缺失、老库还没补算成功）；
      · 这一帧一张脸都没检测到。
    返回体：
      models    [{name, dfm(路径), faces:[脸序号]}]  —— 直接喂给 _preview_multi
      faces     [{idx, group, dfm, similarity, matched, auto:true}]  —— 给前端显示
      unmatched [脸序号]  —— 谁都不像：调用方把它并进 no_swap，保持原像素
      boxes     这次检测的结果（_preview_job 复用它，不再检测第二遍）
    """
    from . import _face_group as _fg
    from . import _faces as _store

    library = _auto_library(payload)
    if library is None:
        return None
    try:
        index = _store.load_index(library)
    except _store.FacesError:
        return None
    if not any(str(g.get("dfm") or "").strip() for g in index["groups"]):
        return None

    threshold = _float_val("face_threshold",
                           payload.get("face_threshold",
                                       payload.get("threshold")),
                           0.0, 1.0, _fg.DEFAULT_THRESHOLD)
    try:
        sig = _fg.library_signature(library)
    except Exception:                                     # noqa: BLE001
        sig = str(library)
    key = json.dumps([_source_ident(snap, mode, name), int(idx),
                      round(float(threshold), 4), sig], sort_keys=True, default=str)
    hit = _auto_cache_get(key)
    if hit is not None:
        return hit

    profiles = _fg.group_profiles(library, index, compute_limit=_AUTO_EMB_LIMIT)
    if not profiles:
        return None

    if boxes is None:
        boxes = _detect_boxes(frame, detector, thresh)
    boxes = list(boxes or [])
    if not boxes:
        return None

    faces, by_dfm, unmatched = [], {}, []
    for i, b in enumerate(boxes[:_PREVIEW_MAX_SWAPS]):
        try:
            x1, y1, x2, y2 = [float(v) for v in (b.get("box") or ())]
        except Exception:                                 # noqa: BLE001
            unmatched.append(int(i))
            continue
        vec = _face_vector_from_frame(frame, b, [x1, y1, x2, y2])
        best = _fg.match_vector(vec, profiles, threshold) if vec is not None else None
        matched = bool(best and best.get("matched") and best.get("dfm"))
        path = _dfm_by_name(best["dfm"]) if matched else None
        if path is None:                     # 组还在但 .dfm 没了（orphan）→ 当没匹配
            matched = False
        if matched:
            by_dfm.setdefault(best["dfm"], {"path": path, "faces": []})["faces"].append(int(i))
        else:
            unmatched.append(int(i))
        faces.append({"idx": int(i), "face": int(i),
                      "group": (best.get("group") or "") if best else "",
                      "group_name": (best.get("name") or "") if best else "",
                      "dfm": best["dfm"] if matched else "",
                      "model": best["dfm"] if matched else "",
                      "similarity": (best.get("similarity") if best else None),
                      "matched": bool(matched), "auto": True})

    models = []
    for pos, (nm, info) in enumerate(by_dfm.items()):
        models.append({"name": str(nm), "dfm": str(info["path"]),
                       "faces": sorted(info["faces"]), "index": int(pos)})
    plan = {"models": models, "faces": faces, "unmatched": sorted(set(unmatched)),
            "boxes": boxes, "threshold": float(threshold), "library": str(library),
            "profiles": len(profiles), "detected": len(boxes),
            "matched": sum(1 for f in faces if f["matched"])}
    _auto_cache_put(key, plan)
    return plan


def _auto_faces_merged(worker_faces, plan: dict) -> list:
    """worker 真贴了哪几张 × 自动判定的目标 → 对外那一份 faces（顺序 = 脸序号）。"""
    by_face = {}
    for f in worker_faces or []:
        try:
            by_face[int(f.get("face"))] = f
        except (TypeError, ValueError):
            continue
    out = []
    for item in plan.get("faces") or []:
        rec = dict(item)
        hit = by_face.get(int(item["idx"])) or {}
        if hit.get("box"):
            rec["box"] = hit["box"]
        rec["applied"] = bool(hit)
        out.append(rec)
    return out


def _manual_faces(worker_faces, library=None) -> list:
    """手动指定（models 里点名）时的 faces：auto=false，group 由 dfm 反查。"""
    index = None
    if library is not None:
        try:
            from . import _faces as _store
            index = _store.load_index(library)
        except Exception:                                 # noqa: BLE001
            index = None
    from . import _face_group as _fg
    out = []
    for f in worker_faces or []:
        try:
            i = int(f.get("face"))
        except (TypeError, ValueError):
            continue
        # 和自动分配那条路对齐：dfm 记**模型名**（路径另放 dfm_path），
        # 前端/验收拿名字直接比，不用先 basename。
        dfm_path = str(f.get("dfm") or "")
        dfm_name = str(f.get("model") or "") or Path(dfm_path).stem
        out.append({"idx": i, "face": i,
                    "group": _fg.group_id_for_dfm(index, dfm_name) if index else "",
                    "dfm": dfm_name, "model": dfm_name, "dfm_path": dfm_path,
                    "similarity": None, "matched": True, "auto": False,
                    "box": f.get("box")})
    return out



# ---- 调试视图（视图级，独立于融合方式）------------------------------------
# 调试图 = 合成结果 + 4 类调试信息，一个像素都不改回合成链路：
#   1) 每张脸的「遮罩区域」：整块检测框淡绿覆盖（算法认为的脸在哪儿）
#   2) 68 点凸包边界：黄线，真正贴回去的多边形边界（不是矩形）
#   3) 检测框 + 序号：绿框 = 换过了；红框 + keep = 这张没换（no_swap / 点框切的原图）
#   4) 顶部 HUD：帧号 + 生效参数（blend / mask / ct / erode / blur）
# 以前 debug 是塞在 settings.mode 里的一个「假融合方式」（引擎 _MODE_EFFECTIVE 不认，
# 选了也按 overlay 出片）。现在它只由 payload.debug=1 触发，融合方式原样保留。
_DEBUG_TINT_BGR = (60, 200, 120)
_DEBUG_EDGE_BGR = (0, 210, 255)
_DEBUG_BOX_BGR = (90, 220, 120)
_DEBUG_NOSWAP_BGR = (70, 70, 235)
_DEBUG_HUD_BGR = (24, 26, 30)
_DEBUG_HUD_TXT = (120, 240, 150)


def _debug_view_image(out, frame, boxes, no_swap, settings, idx, total):
    """合成结果 → 调试图。返回 (图, 画了几张脸)。

    boxes 是**原帧坐标**里的检测框（_detect_boxes 的格式）；out 可能已经缩到
    maxw，这里按 frame 的尺寸算缩放比，框和边界才不会飘。
    """
    import cv2
    import numpy as np

    if out is None or getattr(out, "ndim", 0) != 3:
        return out, 0
    img = np.ascontiguousarray(out).copy()
    h, w = img.shape[:2]
    try:
        fh, fw = frame.shape[:2]
    except Exception:                                   # noqa: BLE001
        fh, fw = h, w
    sx = (w / float(fw)) if fw else 1.0
    sy = (h / float(fh)) if fh else 1.0

    def _rect_of(b):
        try:
            x1, y1, x2, y2 = [float(v) for v in (b.get("box") or ())]
        except Exception:                               # noqa: BLE001
            return None
        if x2 <= x1 or y2 <= y1:
            return None
        X1 = max(0, min(w - 1, int(round(x1 * sx))))
        Y1 = max(0, min(h - 1, int(round(y1 * sy))))
        X2 = max(X1 + 1, min(w - 1, int(round(x2 * sx))))
        Y2 = max(Y1 + 1, min(h - 1, int(round(y2 * sy))))
        return X1, Y1, X2, Y2

    no_swap = {int(x) for x in (no_swap or ())}
    rects = []
    for i, b in enumerate(boxes or []):
        r = _rect_of(b or {})
        if r is not None:
            rects.append((int(i), b, r))

    # ---- 1) 遮罩区域：整块框淡绿（先整体混一次，比逐框 addWeighted 快也干净）----
    if rects:
        region = img.copy()
        for _i, _b, (X1, Y1, X2, Y2) in rects:
            cv2.rectangle(region, (X1, Y1), (X2, Y2), _DEBUG_TINT_BGR, -1)
        img = cv2.addWeighted(img, 0.88, region, 0.12, 0)

    # ---- 2/3) 68 点凸包边界 + 检测框 ----
    for i, b, (X1, Y1, X2, Y2) in rects:
        try:
            lm = _frame_landmarks(frame, b, [float(v) for v in ((b or {}).get("box") or ())])
        except Exception:                               # noqa: BLE001
            lm = None
        if lm is not None and len(lm) >= 5:
            try:
                pts = np.asarray([[float(p[0]) * sx, float(p[1]) * sy] for p in lm],
                                 dtype=np.float32)
                hull = cv2.convexHull(pts.reshape(-1, 1, 2)).astype(np.int32)
                cv2.polylines(img, [hull], True, _DEBUG_EDGE_BGR, 2, cv2.LINE_AA)
            except Exception:                           # noqa: BLE001
                pass
        kept = i in no_swap
        col = _DEBUG_NOSWAP_BGR if kept else _DEBUG_BOX_BGR
        cv2.rectangle(img, (X1, Y1), (X2, Y2), col, 2, cv2.LINE_AA)
        tag = "#%d%s" % (i, " keep" if kept else "")
        ty = (Y1 - 6) if Y1 >= 18 else (Y1 + 16)
        cv2.putText(img, tag, (X1 + 3, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (12, 12, 12), 3, cv2.LINE_AA)
        cv2.putText(img, tag, (X1 + 3, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1,
                    cv2.LINE_AA)

    # ---- 4) HUD：帧号 + 生效参数（全是 ASCII，cv2 的 Hershey 字体只能画 ASCII）----
    def _v(key, dflt="-"):
        v = (settings or {}).get(key) if isinstance(settings, dict) else None
        return dflt if v is None or v == "" else str(v)

    hud = "DEBUG  idx=%s/%s  blend=%s  mask=%s  ct=%s  erode=%s  blur=%s  faces=%d" % (
        idx, (total if total not in (None, "") else "?"), _v("blend"), _v("mask_mode"),
        _v("ct_mode"), _v("mask_erode"), _v("mask_blur"), len(rects))
    scale = max(0.38, min(0.62, w / 1700.0))
    bar = int(round(24 * scale)) + 8
    cv2.rectangle(img, (0, 0), (w, bar), _DEBUG_HUD_BGR, -1)
    cv2.putText(img, hud, (8, bar - int(round(7 * scale))), cv2.FONT_HERSHEY_SIMPLEX,
                scale, _DEBUG_HUD_TXT, 1, cv2.LINE_AA)
    cv2.rectangle(img, (1, 1), (w - 2, h - 2), _DEBUG_EDGE_BGR, 2, cv2.LINE_AA)
    return img, len(rects)


def _frame_payload(idx: int, payload: dict, default_w: int = 900) -> dict:
    """第 idx 帧 → 按 payload 里的合成参数出图（data URL）。

    payload 认的键（全部可选）：
        w / fmt / quality / raw / compose / debug
        settings: {ct_mode, blend, mask_mode, mask_erode, mask_blur, …}
        model: "<dfm 名>"        models: [{"name","dfm","faces"}]
        detector / thresh / xseg_onnx
    raw=1（或 compose=0）= 原图视图，不合成；debug=1 = 调试视图（合成 + 调试叠加）。
    settings.blend 只认 4 个融合方式（overlay / hist-match / seamless /
    seamless-hist-match），其它值一律 400 —— 视图不进 settings。

    **两个短路（都是原帧，不碰 worker）**：
      · 没点名模型（models / model / dfm 一个都没有）→ no_model=true，不做
        「挑第一个 .dfm」那种代用户决定的事，响应 note="没有选择模型，返回原帧"；
      · 这一帧一张脸都没检出 → no_faces=true，跳过自动分配（它还要再检测一次）。
    三个布尔字段的语义固定为：
        有人脸 + 有模型 → has_face=true,  swapped=true,  no_faces=false, no_model=false
        有人脸 + 无模型 → has_face=true,  swapped=false, no_faces=false, no_model=true
        无人脸         → has_face=false, swapped=false, no_faces=true
    检测整条链路只做一次（调用方这份 boxes 直接交给 worker 和自动分配复用）。
    """
    t_start = time.time()
    payload = _as_dict(payload)
    maxw = max(200, min(int(payload.get("w") or default_w or 900), 1600))
    if idx < 0:                          # 参数错先说参数错，别被“没打开素材”盖住
        raise _BadInput("帧号不能是负数：%d" % idx)

    snap = _cache_snap()
    mode = _check_cache_ready(snap)
    idx = _check_idx(snap, idx)
    total = int(snap.get("frames") or 0) or len(snap.get("names") or [])

    fmt = "png" if str(payload.get("fmt") or "jpg").strip().lower() == "png" else "jpg"
    quality = _int_val("quality", payload.get("quality"), 1, 100,
                       88 if fmt == "jpg" else 92)
    raw_only = _bool_val(payload.get("raw"), False) or (
        "compose" in payload and not _bool_val(payload.get("compose"), True))
    # 视图级的「调试」：独立参数（不是融合方式）。raw 优先级更高 —— 原图就是原图，
    # 一个调试像素都不该有。
    debug_view = (not raw_only) and _bool_val(payload.get("debug"), False)

    t_grab = time.time()
    frame, name = _source_frame(snap, mode, idx)
    grab_ms = round((time.time() - t_grab) * 1000.0, 1)

    # ---- 1) raw：老行为，一个字节都不动（原图视图要它）----
    if raw_only:
        buf = _encode_frame(frame, maxw, fmt, quality)
        mime = "image/png" if fmt == "png" else "image/jpeg"
        return _ok(idx=idx, total=total, name=name, raw=True, composed=False,
                   fmt=fmt, bytes=len(buf), ms=int(round((time.time() - t_start) * 1000.0)),
                   png="data:%s;base64,%s" % (mime, _b64.b64encode(buf).decode("ascii")))

    # ---- 2) 参数（缺省 = 当前默认值，和显式给默认值走同一条路）----
    settings = _settings_val(payload.get("settings"))
    cfg = _merge_settings_to_config(settings)
    # 检测器：顶层 detector 优先，其次 settings.detector（界面把它塞在 settings 里）
    detector = _detector_val(payload.get("detector") or settings.get("detector"))
    thresh = _float_val("thresh", payload.get("thresh"), 0.0, 1.0, 0.5)
    models = _models_of(payload)
    # ---- 2a) 请求里到底有没有点名模型？三种写法都算：models 数组 / model 名 / dfm 路径 ----
    # 一个都没有 = **用户没选模型**。以前这里会掉进 _dfm_of 的老兜底：悄悄拿
    # workspace/model 里的第一个 .dfm 就合成，用户没选模型却看到一张换过的脸，
    # 而且谁也不知道用的是哪个模型。现在这种情况直接回原帧（见下面 2d / 2e）。
    asked_model = bool(models) or bool(str(payload.get("model") or "").strip()) \
        or bool(str(payload.get("dfm") or "").strip())
    # 自动分配只对「没点名模型」的请求有意义：点名了就以手动为准，一个字节都不参与。
    auto_on = (not models) and _bool_val(payload.get("auto_faces",
                                                     payload.get("auto")), True)

    # 点名"不换"的脸（前端「不换脸」勾选）。没给 = 老行为（未声明的脸兜给第 0 个模型）。
    no_swap = _face_ids_val("no_swap_faces",
                            payload.get("no_swap_faces",
                                        payload.get("skip_faces",
                                                    settings.get("no_swap_faces"))))
    dfm, model_name = None, ""

    def _short_circuit(img, note, *, has_face, no_model, no_faces, detect_ms=0.0,
                       dbg_boxes=None, extra=None):
        """不合成的那两条路（no_model / no_faces）→ **还是原帧**，只是换个编码。

        不碰 worker、不跑 dfm、不重检测：这里只剩「画调试信息（可选）+ 编码」，
        所以耗时和 raw=1 那一条是同一量级。三个字段按语义写死：
        has_face / no_model / no_faces，前端不用再靠 note 猜。
        """
        t_enc = time.time()
        out_img, dbg_n = img, 0
        if debug_view and dbg_boxes:
            # 没有模型也能看调试视图：框 / 遮罩区域 / HUD 全都在，只是没有换脸结果。
            out_img, dbg_n = _debug_view_image(out_img, frame, list(dbg_boxes), [],
                                               settings, idx, total)
        buf = _encode_frame(out_img, maxw, fmt, quality)
        enc_ms = round((time.time() - t_enc) * 1000.0, 1)
        mime = "image/png" if fmt == "png" else "image/jpeg"
        resp = _ok(idx=int(idx), total=int(total), name=name, raw=True,
                   composed=False, swapped=False, has_face=bool(has_face),
                   no_model=bool(no_model), no_faces=bool(no_faces), faces=[],
                   fmt=fmt, bytes=len(buf),
                   ms=int(round((time.time() - t_start) * 1000.0)),
                   timings={"grab_ms": grab_ms,
                            "detect_ms": round(float(detect_ms), 1),
                            "encode_ms": enc_ms, "swap_ms": None, "worker": 0},
                   detector=detector, thresh=float(thresh), settings=settings,
                   debug=bool(debug_view), debug_faces=int(dbg_n), note=note,
                   png="data:%s;base64,%s" % (mime, _b64.b64encode(buf).decode("ascii")))
        if model_name:
            resp["model"] = model_name
            resp["dfm"] = str(dfm)
        if extra:
            resp.update(extra)
        return resp

    def _cache_hit(hit):
        # 缓存里放的是共享 dict：拷一份再改 ms / cached，别把缓存本身写脏
        hit = dict(hit)
        hit["cached"] = True
        hit["ms"] = int(round((time.time() - t_start) * 1000.0))
        tm = dict(hit.get("timings") or {})
        tm["cache_hit"] = 1
        hit["timings"] = tm
        return hit

    # ---- 2b) 给了 models 数组：先解析 dfm（不存在就照实 404），先试一次缓存 ----
    # 只有显式 models 能在检测**之前**查缓存：别的写法下自动分配还要参与，
    # 结果和"单模型全帧换"不是同一张图（sig 分不开），必须先跑完自动分配。
    if models:
        dfm, model_name = Path(models[0]["dfm"]), models[0]["name"]
        xseg = _xseg_of(payload)
        sig = _compose_sig(snap, mode, idx, name, cfg, str(dfm), models, xseg,
                           detector, float(thresh), fmt, int(quality), int(maxw),
                           no_swap, debug=debug_view)
        hit = _compose_cached(sig)
        if hit is not None:
            return _cache_hit(hit)
    else:
        xseg, sig = "", ""

    # ---- 2c) 检测**一次**：自动分配 / 「这一帧有没有脸」/ worker 换脸 共用这一份 ----
    t_det = time.time()
    boxes = list(_detect_boxes(frame, detector, thresh))
    detect_ms = round((time.time() - t_det) * 1000.0, 1)

    # ---- 2d) 一张脸都没检出 → 原帧（no_faces）：不自动分配、不进 worker ----
    # 以前这里照样往下走：_auto_face_plan 内部再检测一遍，然后整帧编码回一张
    # 和原图一模一样的图。现在最贵的一步只剩「检测 + 编码」。
    if not boxes:
        return _short_circuit(frame, "这一帧没检出人脸：返回原帧",
                              has_face=False, no_model=not asked_model,
                              no_faces=True, detect_ms=detect_ms)

    # ---- 2e) 没给 models 数组 → 按人脸库自动分配（复用上面那一份检测结果）----
    # 和以前一样：**只有 models 数组为空才走自动**（左栏那个 model 只是"库判不出来
    # 时"的老兜底，不参与自动判定）。
    auto_plan = None
    if not models and auto_on:
        auto_plan = _auto_face_plan(payload, frame, snap, mode, name, int(idx),
                                    detector, thresh, boxes=boxes)
        if auto_plan is not None:
            models = auto_plan["models"] or None

    auto_extra = None
    if auto_plan is not None:
        # 前端要照这份清单显示每张脸的相似度（auto=true = 判定来自人脸库）
        auto_extra = {"auto": True, "faces": list(auto_plan["faces"]),
                      "face_threshold": float(auto_plan["threshold"]),
                      "face_library": str(auto_plan.get("library") or ""),
                      "auto_matched": int(auto_plan.get("matched") or 0),
                      "auto_detected": int(auto_plan.get("detected") or 0)}

    if not models:
        if auto_plan is not None:
            # 自动分配判定过、但一张脸都没匹配上：**不能退回"默认模型全帧换"**，
            # 那等于把不相干的人换上去。整帧保持原像素，判定结果照样带出去。
            return _short_circuit(
                frame, "没有可用的 .dfm 模型：只回原帧（自动分配：这一帧没有和任何人脸组"
                       "匹配上的脸）", has_face=True, no_model=not asked_model,
                no_faces=False, detect_ms=detect_ms, dbg_boxes=boxes, extra=auto_extra)
        if not asked_model:
            # 请求里没有任何模型信息 = 用户没选模型：**绝不替用户挑一个 .dfm** ——
            # 那等于用了一个他不知道的模型，还把别人的脸换上去。
            return _short_circuit(frame, "没有选择模型，返回原帧",
                                  has_face=True, no_model=True, no_faces=False,
                                  detect_ms=detect_ms, dbg_boxes=boxes)
        # 点名了 model / dfm：按老规矩解析（不存在就 404），然后正常合成。
        dfm, model_name = _dfm_of(payload)
        xseg = _xseg_of(payload)
    else:
        dfm, model_name = Path(models[0]["dfm"]), models[0]["name"]
        xseg = _xseg_of(payload)

    if auto_plan is not None:
        # 谁都不像的脸：并进 no_swap，保持原像素（不换 > 换错）
        no_swap = sorted(set(int(x) for x in no_swap)
                         | set(int(x) for x in (auto_plan.get("unmatched") or [])))

    # ---- 3) 命中缓存就直接回（同一帧 + 同一组参数，拖滑块回退时立刻出图）----
    if not sig:
        sig = _compose_sig(snap, mode, idx, name, cfg, str(dfm), models, xseg,
                           detector, float(thresh), fmt, int(quality), int(maxw),
                           no_swap, debug=debug_view)
        hit = _compose_cached(sig)
        if hit is not None:
            return _cache_hit(hit)

    # ---- 4) 真合成：和 /preview、/preview-frame 共用 _preview_job ----
    stop = threading.Event()
    state = {"stage": "start"}
    # fmt=png 让 worker 直接给无损的中间结果（逐参数对比要看真实像素差，
    # 不能被 JPEG 量化噪声盖住）；jpg 就用 worker 编好的那一份，不再二次压缩。
    want_png = (fmt == "png")
    res = _preview_job({"frame": frame, "models": models, "dfm": dfm, "xseg": xseg,
                        "config": cfg, "detector": detector, "thresh": float(thresh),
                        "stop": stop, "state": state,
                        "fmt": "png" if want_png else "jpg",
                        "quality": int(quality),
                        "no_swap": list(no_swap),
                        # 这一帧唯一的那次检测（2c）直接交给 worker：它不再检测第二遍，
                        # 自动分配也是拿这份 boxes 算的 embedding —— 全链路只检测一次。
                        "boxes": list(boxes),
                        # png 请求要的是"能逐像素对比"的结果 → 帧无损发给 worker
                        "send_lossless": bool(want_png)})
    img = res.get("img") or res.get("jpg") or b""
    if not img:
        raise _BadInput("这一帧没有产出图", 500)
    import cv2
    import numpy as np
    out = cv2.imdecode(np.frombuffer(img, np.uint8), cv2.IMREAD_COLOR)
    if out is None:
        raise _BadInput("合成结果解不开", 500)
    # 先缩到 maxw 再画调试信息：字号/线宽按最终展示尺寸给，图缩一半时框线才不会细成一条。
    resized = False
    if out.shape[1] > maxw:
        out = cv2.resize(out, (maxw, max(1, int(out.shape[0] * maxw / out.shape[1]))),
                         interpolation=cv2.INTER_AREA)
        resized = True
    dbg_faces = 0
    if debug_view:
        out, dbg_faces = _debug_view_image(out, frame, boxes, no_swap,
                                           settings, idx, total)
    # 什么时候要重编：png（无损）· 刚缩过图 · 画了调试信息（不重编等于白画）。
    # 老逻辑是「jpg 且没超出 maxw 就复用 worker 编好的那一份」—— 现在 out 可能
    # 已经被我们缩过/画过，必须显式重编，不能再用 worker 那张原图。
    if want_png or resized or debug_view:
        img = _encode_frame(out, maxw, fmt, quality)
    mime = "image/png" if fmt == "png" else "image/jpeg"

    # timings 只记一次检测：worker 那份是 reuse（它没再检测），detect_ms 换成 2c 实测的数字
    timings = dict(res.get("timings") or {})
    timings["grab_ms"] = grab_ms
    timings["detect_ms"] = detect_ms
    timings["detect_reused"] = 1             # worker 用的是 2c 那一份，不是它自己检的
    resp = _ok(idx=int(idx), total=int(total), name=name, composed=True,
               swapped=bool(res.get("swapped")), has_face=bool(res.get("has_face")),
               no_model=False, no_faces=bool(res.get("no_faces")),
               box=res.get("box"), fmt=fmt, bytes=len(img),
               ms=int(round((time.time() - t_start) * 1000.0)),
               model=model_name, dfm=str(dfm), detector=detector,
               thresh=float(thresh), settings=settings, effective=cfg,
               mask_mode=int(cfg.get("mask_mode") or 0),
               mode=cfg.get("mode"), ct_mode=cfg.get("color_transfer_mode"),
               erode_mask=cfg.get("erode_mask_modifier"),
               blur_mask=cfg.get("blur_mask_modifier"),
               xseg=xseg, timings=timings,
               no_swap_faces=list(no_swap), skipped=list(res.get("skipped") or []),
               cached=False, note=res.get("note") or "",
               debug=bool(debug_view), debug_faces=int(dbg_faces),
               png="data:%s;base64,%s" % (mime, _b64.b64encode(img).decode("ascii")))
    if models:
        resp["models"] = [{"name": m["name"], "dfm": m["dfm"],
                           "faces": list(m["faces"])} for m in models]
        resp["models_used"] = list(res.get("models_used") or [])
        if auto_plan is not None:
            # 自动分配：faces = 每张检测到的脸 + 它被判定给了哪个组/dfm/相似度
            resp["faces"] = _auto_faces_merged(res.get("faces"), auto_plan)
            resp["auto"] = True
            resp["face_threshold"] = float(auto_plan["threshold"])
            resp["face_library"] = str(auto_plan.get("library") or "")
            resp["auto_matched"] = int(auto_plan.get("matched") or 0)
            resp["auto_detected"] = int(auto_plan.get("detected") or 0)
        else:
            # 手动指定（chip 上有值）：auto=false，group 由 dfm 反查
            resp["faces"] = _manual_faces(res.get("faces"), _auto_library(payload))
            resp["auto"] = False
        resp["swaps"] = int(res.get("swaps") or 0)
    _compose_store(sig, resp)
    return resp


def _body_idx(payload: dict) -> int:
    """帧号从 body 里取：idx > frame > n（三种写法前端都在用）。"""
    for k in ("idx", "frame", "n", "index", "frame_idx"):
        if payload.get(k) not in (None, ""):
            return _int_val("idx", payload.get(k), 0, None, 0)
    raise _BadInput("缺少帧号：body 里给 idx（也认 frame / n），或者用 /api/merge/frame/{idx}")


def _frame_body(idx: int, body, **query) -> dict:
    """POST body + query 合成一个 payload（body 优先）。"""
    payload = dict(_as_dict(body))
    _merge_query_into(payload, **query)
    return payload


@router.get("/api/merge/frame/{idx}")
@router.post("/api/merge/frame/{idx}")
@_api
def merge_frame(idx: int, body: dict = None, w: int = 900, settings: str = "",
                model: str = "", models: str = "", detector: str = "",
                thresh: float = None, fmt: str = "", quality: int = None,
                raw: str = "", compose: str = "", xseg_onnx: str = "",
                no_swap_faces: str = "", debug: str = ""):
    """第 idx 帧的**合成结果**（data URL）。GET 走 query、POST 走 JSON body，两套都认。

    body（全部可选，缺省 = 当前 UI 默认值）：
        settings: {"ct_mode":"mt"|"rct"|"lct"|"mkl"|"sot",
                   "blend":"overlay"|"hist-match"|"seamless"|"seamless-hist-match",
                   "mask_mode":0~9, "mask_erode":-100~100, "mask_blur":-100~100}
        model: "<dfm 名>"  或  models: [{"name":"<dfm>","faces":[0,1]}]
        detector / thresh / w / fmt(png|jpg) / quality / xseg_onnx
    raw=1（或 compose=0）：不合成，原样返回源帧（视图「原图」用）。
    debug=1：合成 + 遮罩/边界/检测框/HUD 调试叠加（视图「调试」用），
             融合方式照旧由 settings.blend 决定。视图不是融合方式：
             往 settings.blend 里塞 original / raw-* / debug 一律 400。
    """
    payload = _frame_body(idx, body, w=w, settings=settings, model=model,
                          models=models, detector=detector, thresh=thresh,
                          fmt=fmt, quality=quality, raw=raw, compose=compose,
                          xseg_onnx=xseg_onnx, no_swap_faces=no_swap_faces, debug=debug)
    return _frame_payload(idx, payload, default_w=w)


@router.post("/api/merge/frame")
@_api
def merge_frame_post(body: dict = None, w: int = 900, settings: str = "",
                     model: str = "", models: str = "", detector: str = "",
                     thresh: float = None, fmt: str = "", quality: int = None,
                     raw: str = "", compose: str = "", xseg_onnx: str = "",
                     no_swap_faces: str = "", debug: str = ""):
    """POST /api/merge/frame —— 帧号放在 body 里（idx / frame / n），其余同 {idx} 版。

    前端在视频模式下拿的是「时间轴第几帧」而不是 URL 段，所以契约里这个不带路径参数
    的写法必须也认；两套走的是同一个 _frame_payload。
    """
    payload = dict(_as_dict(body))
    _merge_query_into(payload, w=w, settings=settings, model=model, models=models,
                      detector=detector, thresh=thresh, fmt=fmt, quality=quality,
                      raw=raw, compose=compose, xseg_onnx=xseg_onnx,
                      no_swap_faces=no_swap_faces, debug=debug)
    return _frame_payload(_body_idx(payload), payload, default_w=w)


# ============================================================
# 单脸调试：GET / POST /api/merge/face-debug
# ------------------------------------------------------------
# 一张脸的「六宫格」证据链：两张模型遮罩、它们的合并、预测脸、对齐后的原脸、
# 以及真的合成出来的那张脸。前端自己排版（3 列 × 2 行），所以这里返回**六张
# 独立图**而不是拼好的一整张 —— 点开放大、单格存图都靠这个。
#   第 1 行：(1) mask(dfm-pred)  (2) mask(dst)  (3) merged(mask(dfm-pred), mask(dst))
#   第 2 行：(4) dfm-pred        (5) dst-original (6) merged
# 图从哪来：**复用预览 worker**（同一份 MergeMaskedFace、同一批常驻 DFM），
# 只多跑一次单脸合成 + 把中间量截下来，不重新加载模型、不另起进程。
# 参数语义和 /api/merge/frame 完全一致（settings / models / detector …），
# 唯一的例外：**idx 是「第几张脸」**，帧号只认 frame / n（见 _debug_face_idx）。
# ============================================================
_FACE_DBG_CACHE = collections.OrderedDict()     # 同一帧同一张脸 + 同一组参数 → 六格图
_FACE_DBG_MAX = 8
_FACE_DBG_LOCK = threading.Lock()


def _debug_face_idx(payload: dict) -> int:
    """这张脸是这一帧里的第几张（检测顺序，从 0 起）。

    帧号只认 frame / n / frame_idx：本接口的 idx 按契约是**人脸序号**，
    和老接口的「idx = 帧号」正好相反，所以这里绝不走 _body_idx()。
    """
    for k in ("face", "face_idx", "face_index", "idx", "i"):
        if payload.get(k) not in (None, ""):
            return _int_val("face", payload.get(k), 0, 63)
    raise _BadInput("缺少人脸序号：给 face（也认 face_idx / idx）—— 这一帧里第几张脸，从 0 起")


def _debug_frame_idx(payload: dict) -> int:
    for k in ("frame", "n", "frame_idx"):
        if payload.get(k) not in (None, ""):
            return _int_val("frame", payload.get(k), 0, None)
    return 0


def _face_debug_payload(payload: dict) -> dict:
    """一张脸 → 六张图（data URL）。"""
    t_start = time.time()
    payload = _as_dict(payload)
    panel_size = max(96, min(int(payload.get("panel_size") or 256), 512))

    snap = _cache_snap()
    mode = _check_cache_ready(snap)
    idx = _check_idx(snap, _debug_frame_idx(payload))
    total = int(snap.get("frames") or 0) or len(snap.get("names") or [])
    face_i = _debug_face_idx(payload)
    frame, name = _source_frame(snap, mode, idx)

    settings = _settings_val(payload.get("settings"))
    cfg = _merge_settings_to_config(settings)
    detector = _detector_val(payload.get("detector") or settings.get("detector"))
    thresh = _float_val("thresh", payload.get("thresh"), 0.0, 1.0, 0.5)

    # ---- 这一帧的脸都在哪：有 models 就照 models 分配，没有就按人脸库自动判 ----
    models = _models_of(payload)
    auto_plan = None
    if not models:
        auto_plan = _auto_face_plan(payload, frame, snap, mode, name, int(idx),
                                    detector, thresh)
        if auto_plan is not None:
            models = auto_plan["models"] or None
    boxes = list((auto_plan or {}).get("boxes") or [])
    if not boxes:
        try:
            boxes = list(_detect_boxes(frame, detector, thresh))
        except Exception:                                 # noqa: BLE001
            boxes = []
    if not boxes:
        raise _BadInput("这一帧没检测到人脸 —— 换一帧再点调试", 404)
    if face_i >= len(boxes):
        raise _BadInput("这一帧只有 %d 张脸，要的是第 %d 张（从 0 起）"
                        % (len(boxes), face_i), 404)

    # ---- 这张脸用哪个 dfm：models 点名 → 自动判定 → 显式 dfm/model ----
    dfm, model_name = None, ""
    if models:
        mi = _model_for_face(face_i, models)
        m = models[mi]
        # 只在「这个模型确实管这张脸」时才认（faces 空 = 管所有脸；点名了但没这张 = 不管）
        named = (not (m.get("faces") or [])) or (int(face_i) in (m.get("faces") or []))
        if named:
            dfm, model_name = Path(m["dfm"]), str(m["name"])
    if dfm is None:
        try:
            dfm, model_name = _dfm_of(payload)
        except _BadInput:
            dfm, model_name = None, ""
    if dfm is None:
        raise _BadInput("第 %d 张脸没有可用的 .dfm（谁的组都不像）—— 先在框上的 chip 里"
                        "给它选一个模型，再看六宫格" % face_i, 409)

    b = boxes[face_i]
    try:
        x1, y1, x2, y2 = [float(v) for v in (b.get("box") or ())]
    except Exception:                                     # noqa: BLE001
        raise _BadInput("第 %d 张脸的检测框不合法" % face_i, 500)
    lm = _frame_landmarks(frame, b, [x1, y1, x2, y2])
    if lm is None:
        raise _BadInput("第 %d 张脸算不出 68 点 —— 换一帧再试" % face_i, 409)
    box_wh = [int(round(x1)), int(round(y1)),
              int(round(max(0.0, x2 - x1))), int(round(max(0.0, y2 - y1)))]

    xseg = _xseg_of(payload)
    key = json.dumps([_source_ident(snap, mode, name), int(idx), int(face_i),
                      str(dfm), cfg, xseg, int(panel_size)], sort_keys=True, default=str)
    with _FACE_DBG_LOCK:
        hit = _FACE_DBG_CACHE.get(key)
        if hit is not None:
            _FACE_DBG_CACHE.move_to_end(key)
            out = dict(hit)
            out["cached"] = True
            out["ms"] = int(round((time.time() - t_start) * 1000.0))
            return out

    import numpy as np
    resp = _WORKER.call({
        "cmd": "face_debug",
        "frame": _send_frame_b64(frame, {"send_lossless": True}),
        "dfm": str(dfm),
        "landmarks": np.asarray(lm, dtype=np.float64).round(2).tolist(),
        "box": list(box_wh),
        "config": dict(cfg),
        "xseg": xseg,
        "panel_size": int(panel_size),
    })
    if not resp.get("ok"):
        raise _BadInput("六宫格出图失败：%s" % (resp.get("error") or "预览 worker 没给原因"), 500)

    panels = []
    for item in resp.get("panels") or []:
        raw = item.get("png") or ""
        if not raw:
            continue
        panels.append({"key": str(item.get("key") or ""),
                       "label": str(item.get("label") or ""),
                       "w": int(item.get("w") or panel_size),
                       "h": int(item.get("h") or panel_size),
                       "png": "data:image/png;base64," + raw})
    if len(panels) != 6:
        raise _BadInput("六宫格只出了 %d 张（要 6 张）—— worker 版本对不上？" % len(panels), 500)

    out = _ok(frame=int(idx), total=int(total), name=name, size=panel_size,
              face={"idx": int(face_i), "box": box_wh,
                    "score": (b.get("score") if isinstance(b, dict) else None),
                    "model": model_name, "dfm": str(dfm)},
              panels=panels, stats=resp.get("stats") or {},
              settings=settings, detector=detector, xseg=xseg,
              model=model_name, dfm=str(dfm), auto=bool(auto_plan is not None),
              ms=int(round((time.time() - t_start) * 1000.0)), cached=False)
    with _FACE_DBG_LOCK:
        _FACE_DBG_CACHE[key] = out
        while len(_FACE_DBG_CACHE) > _FACE_DBG_MAX:
            _FACE_DBG_CACHE.popitem(last=False)
    return out


@router.get("/api/merge/face-debug")
@router.post("/api/merge/face-debug")
@_api
def merge_face_debug(body: dict = None, frame: int = None, face: int = None,
                     idx: int = None, settings: str = "", model: str = "",
                     models: str = "", detector: str = "", thresh: float = None,
                     xseg_onnx: str = "", panel_size: int = None):
    """一张脸的六宫格（3 列 × 2 行）。

    GET  /api/merge/face-debug?frame=0&idx=0            —— idx = 人脸序号
    POST /api/merge/face-debug  {"frame":0,"face":0, "settings":{…}, "models":[…]}

    body / query（除 frame、face 外全部可选）：
        frame      帧号（也认 n；**不要**用 idx 传帧号）
        face       这一帧里第几张脸，从 0 起（也认 face_idx / idx）
        settings   {ct_mode, blend, mask_mode, mask_erode, mask_blur, detector, face_type}
        models     [{"name","dfm","faces"}]（不给 = 按人脸库自动判每张脸用哪个 dfm）
        detector / thresh / xseg_onnx / panel_size(默认 256)

    返回 {ok, frame, face:{idx,box,model,dfm,score}, size, panels:[{key,label,w,h,png}], stats}
    panels 顺序固定（行优先）：
        mask-prd     mask(dfm-pred)
        mask-dst     mask(dst)
        mask-both    merged(mask(dfm-pred), mask(dst))
        dfm-pred     dfm-pred
        dst-original dst-original
        merged       merged
    """
    payload = dict(_as_dict(body))
    _merge_query_into(payload, frame=frame, face=face, idx=idx, settings=settings,
                      model=model, models=models, detector=detector, thresh=thresh,
                      xseg_onnx=xseg_onnx, panel_size=panel_size)
    return _face_debug_payload(payload)


@router.post("/api/merge/export")
@_api
def merge_export(body: dict):
    """把一段帧目录合成成换脸结果。真要跑，不是占位。"""
    body = _as_dict(body)

    # ---- 1) 输入帧目录：project 和 input_dir 是同一样东西 ----
    ip = _norm_path(body.get("input_dir") or body.get("project"))
    if ip is None:
        raise _BadInput("缺少 input_dir（或 project）：要一个放帧图片的目录")
    if not ip.exists():
        raise _BadInput("输入目录不存在：%s" % ip, 404)
    if not ip.is_dir():
        raise _BadInput("input_dir 不是目录：%s" % ip)
    if not _scan_first(ip, _IMAGE_EXT, 1):
        raise _BadInput("输入目录里没有帧图片（.jpg/.png）：%s —— 先跑一次「抽帧 + 人脸对齐」" % ip)

    # ---- 2) 模型：显式 dfm > 模型名 > 自动挑第一个 ----
    model_name = str(body.get("model") or "").strip()
    if model_name and (("/" in model_name) or ("\\" in model_name)
                       or model_name in (".", "..") or ".." in model_name):
        raise _BadInput("「model」只能是模型名，不能带路径：%r" % model_name)

    dfm_raw = body.get("dfm")
    dfm = None
    if str(dfm_raw or "").strip():
        dfm = _norm_path(dfm_raw)
        if dfm.suffix.lower() != ".dfm":
            raise _BadInput("dfm 要是一个 .dfm 文件：%s" % dfm)
        if not dfm.is_file():
            raise _BadInput("找不到 DFM：%s" % dfm, 404)
    elif model_name:
        name = model_name if model_name.lower().endswith(".dfm") else model_name + ".dfm"
        cand = Path(WORKSPACE) / "model" / name
        if not cand.is_file():
            # 这里以前会悄悄退回"第一个 .dfm"，结果用别的模型出片、没人发现
            raise _BadInput("模型 %s 还没有导出 .dfm（%s）—— 去训练页点「导出」"
                            % (model_name, cand))
        dfm = cand
    if dfm is None:
        cands = sorted((Path(WORKSPACE) / "model").glob("*.dfm"))
        if not cands:
            raise _BadInput("没有可用的 .dfm 模型：先在训练页把模型导出成 .dfm（%s）"
                            % (Path(WORKSPACE) / "model"))
        dfm = cands[0]

    # ---- 3) 对齐集（DFLJPG）：显式 > 输入目录下 aligned/ > 同级 <name>_aligned/ ----
    aligned = _norm_path(body.get("aligned_dir"))
    if aligned is None:
        for c in (ip / "aligned", ip.parent / (ip.name + "_aligned")):
            if c.is_dir() and _scan_first(c, {".jpg", ".jpeg"}, 1):
                aligned = c
                break
    if aligned is None and _has_dfl_meta(ip):
        aligned = ip                            # 输入目录自己就是对齐集
    if aligned is None or not aligned.is_dir():
        raise _BadInput("找不到对齐集（DFLJPG）：期望 %s 下的 aligned/ 或 %s_aligned。"
                        "先跑一次「抽帧 + 人脸对齐」。" % (ip, ip))
    if not _scan_first(aligned, {".jpg", ".jpeg"}, 1):
        raise _BadInput("对齐集是空的：%s" % aligned)
    if not _has_dfl_meta(aligned):
        raise _BadInput("对齐集里没有 DFLJPG 元数据（landmarks）：%s —— "
                        "合成靠它定位人脸，换一个带元数据的 aligned 目录" % aligned)

    # ---- 4) 输出 ----
    out_raw = str(body.get("output_dir") or body.get("output") or "").strip()
    if out_raw:
        op = _norm_path(out_raw)
        if op.suffix.lower() == ".mp4":
            out_mp4 = op
        else:
            if op.exists() and op.is_file():
                raise _BadInput("输出路径是一个已存在的文件：%s（要目录或 .mp4）" % op)
            out_mp4 = op / "merged.mp4"
    else:
        out_mp4 = ip.parent / (ip.name + "_merged.mp4")
    if out_mp4.is_dir():
        raise _BadInput("输出路径是个目录：%s（要 .mp4 文件）" % out_mp4)
    if _same_path(out_mp4, ip):
        raise _BadInput("输出路径不能是输入目录：%s" % out_mp4)
    try:
        out_mp4.parent.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        raise _BadInput("输出目录建不出来：%s（%s）" % (out_mp4.parent, e))

    # ---- 5) XSeg 遮罩模型 ----
    mask_onnx = ""
    xr = body.get("xseg_onnx")
    if str(xr or "").strip():
        xp = _norm_path(xr)
        if not xp.is_file():
            raise _BadInput("找不到 XSeg 模型：%s" % xp, 404)
        mask_onnx = str(xp)
    else:
        for c in (Path(WORKSPACE) / "model" / "XSegLite" / "xseglite.onnx",
                  Path(WORKSPACE) / "model" / "XSeg" / "XSeg.onnx"):
            if c.is_file():
                mask_onnx = str(c)
                break

    # ---- 6) 合成参数 / 编码参数 ----
    settings = _settings_val(body.get("settings"))
    raw_fps = body.get("fps")
    if raw_fps in (None, "", 0, "0", 0.0, "0.0"):
        raw_fps = None
    fps = _float_val("fps", raw_fps, 1.0, 240.0, None)
    if fps is None:
        cached = _float_val("fps", _cache_snap().get("fps"), 0.0, 240.0, 0.0)
        fps = cached if cached > 0 else 30.0
    workers = _int_val("workers", body.get("workers"), 0, 64, 0)
    encoder = _encoder_val(body.get("encoder"))
    hwaccel = _hwaccel_val(body.get("hwaccel"))

    # ---- 7) 起线程：先把状态置成 running 再起线程 ----
    with _merge_job_lock:
        if _merge_job.get("running"):
            raise _BadInput("已经在合成（%s）—— 等它结束或点取消"
                            % (_merge_job.get("out") or "上一个任务"), 409)
        # 否则前端刚拿到 ok=True 就来轮询，会读到线程启动前的旧状态
        # （running=False / ok=None、以及上一轮残留的 cancelled 和 stage）。
        _merge_job.update({"running": True, "log": [], "ok": None, "error": None,
                           "error_detail": "", "out": str(out_mp4), "pct": 0,
                           "msg": "启动合成…", "stage": -1, "cancelled": False,
                           "pct_at_failure": None, "started_at": time.time(),
                           "finished_at": None, "fps": fps, "workers": workers,
                           "encoder": encoder, "hwaccel": hwaccel,
                           "dfm": str(dfm)})
    threading.Thread(target=_run_export,
                     args=(str(ip), str(out_mp4), str(aligned), str(dfm), settings,
                           mask_onnx, fps, workers, encoder, hwaccel),
                     daemon=True, name="merge-export").start()
    return _ok(input_dir=str(ip), aligned_dir=str(aligned), output=str(out_mp4),
               output_dir=str(out_mp4), dfm=str(dfm), model=model_name,
               settings=settings, fps=fps, workers=workers, encoder=encoder,
               hwaccel=hwaccel, mask_onnx=mask_onnx)


@router.get("/api/merge/export/status")
@_api
def merge_export_status():
    """导出进度。ok = 任务是否成功（前端 d.ok 决定弹完成还是失败，所以不能恒 true）。"""
    with _merge_job_lock:
        snap = dict(_merge_job)
        log = list(_merge_job.get("log") or [])
    log_tail = log[-50:]

    cancelled = bool(snap.get("cancelled"))
    ok = snap.get("ok")
    # 取消了就对外立刻不是 running（进程正在被看门狗收掉），状态落 cancelled
    running = bool(snap.get("running")) and not cancelled
    if running:
        phase = "running"
    elif ok is True:
        phase = "done"
    elif cancelled:
        phase = "cancelled"
    elif ok is False:
        phase = "failed"
    else:
        phase = "idle"

    error = str(snap.get("error") or "")
    detail = str(snap.get("error_detail") or "")
    if not detail and error:                      # 兜底：至少把日志尾巴给出来
        detail = "\n".join(_log_tail(log, 20))

    stage = snap.get("stage", -1)
    stage = -1 if stage is None else int(stage)
    if phase in ("failed", "cancelled"):
        stage = -1
    pct = int(snap.get("pct") or 0)
    if phase in ("failed", "cancelled"):
        pct = 0
    started = snap.get("started_at")
    finished = snap.get("finished_at")
    try:
        elapsed = round(float((finished or time.time()) - started), 1) if started else None
    except Exception:
        elapsed = None

    return {"ok": ok, "success": ok, "phase": phase, "running": running,
            "cancelled": cancelled, "stage": stage, "pct": pct,
            "pct_at_failure": snap.get("pct_at_failure"),
            "msg": str(snap.get("msg") or ""),
            "error": error, "error_detail": detail,
            "out": str(snap.get("out") or ""),
            "log": log_tail, "log_tail": log_tail[-12:],
            "started_at": started, "finished_at": finished, "elapsed": elapsed,
            "fps": snap.get("fps"), "workers": snap.get("workers"),
            "encoder": snap.get("encoder"), "hwaccel": snap.get("hwaccel"),
            "dfm": snap.get("dfm")}


def _merged_from_export(idx: int, maxw: int, payload: dict = None) -> dict:
    """老行为：从「最近一次导出的成片」里抠一帧（没有打开的素材时才走这条）。"""
    with _merge_job_lock:
        last_out = str(_merge_job.get("out") or "").strip()
    cands = []
    if last_out:
        cands.append(Path(last_out))
    snap = _cache_snap()
    d = str(snap.get("dir") or "").strip()
    if d:                                    # 重启之后 _merge_job 是空的 → 按默认命名找
        cands.append(Path(d + "_merged.mp4"))
        cands.append(Path(d) / "merged.mp4")
    out = next((p for p in cands if p and p.is_file()), None)
    if out is None:
        raise _BadInput("还没有导出的成片 —— 先点一次导出（或者先打开素材再看合成）", 409)

    import cv2
    cap = cv2.VideoCapture(str(out))
    if not cap.isOpened():
        raise _BadInput("打不开成片：%s" % out, 500)
    total = 0
    try:
        try:
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        except Exception:
            total = 0
        if total and idx >= total:
            raise _BadInput("帧号越界：%d，成片只有 %d 帧（0~%d）"
                            % (idx, total, total - 1), 404)
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        good, frame = cap.read()
        if not good or frame is None:
            raise _BadInput("成片里没有第 %d 帧" % idx, 404)
    finally:
        cap.release()

    fmt = "png" if str((payload or {}).get("fmt") or "jpg").strip().lower() == "png" else "jpg"
    buf = _encode_frame(frame, maxw, fmt, 88)
    mime = "image/png" if fmt == "png" else "image/jpeg"
    return _ok(idx=int(idx), total=total, out=str(out), source="export",
               composed=False, fmt=fmt, bytes=len(buf),
               png="data:%s;base64,%s" % (mime, _b64.b64encode(buf).decode("ascii")))


@router.get("/api/merge/merged/{idx}")
@router.post("/api/merge/merged/{idx}")
@_api
def merge_merged_frame(idx: int, body: dict = None, w: int = 900, settings: str = "",
                       model: str = "", models: str = "", detector: str = "",
                       thresh: float = None, fmt: str = "", quality: int = None,
                       raw: str = "", compose: str = "", xseg_onnx: str = "",
                       no_swap_faces: str = ""):
    """第 idx 帧的成片图（data URL）——**现算**，参数改了就立刻变。

    以前这里只会去「最近一次导出的成片」里抠一帧：那是导出那一刻的参数，
    所以右侧改 ct_mode / blend / 遮罩，画面一动不动（用户报的就是这个）。
    现在：
      · 有打开的素材（视频 / 帧目录）→ 和第 idx 帧**现算**一遍（和 /frame 同一条
        合成链路、同一份参数），响应里 source="live"
      · 没有素材、只有历史成片 → 仍退回「最近一次导出的成片」里那一帧（source="export"）
    body 和 /api/merge/frame 完全一致（settings / model / models / detector / …）。
    """
    if idx < 0:
        raise _BadInput("帧号不能是负数：%d" % idx)
    payload = _frame_body(idx, body, w=w, settings=settings, model=model,
                          models=models, detector=detector, thresh=thresh,
                          fmt=fmt, quality=quality, raw=raw, compose=compose,
                          xseg_onnx=xseg_onnx, no_swap_faces=no_swap_faces)

    snap = _cache_snap()
    mode = _cache_mode(snap)
    if mode:
        try:
            _check_cache_ready(snap)
        except _BadInput:
            mode = ""                        # 素材没了：退到成片那条路
    if mode:
        resp = _frame_payload(idx, payload, default_w=w)
        resp["source"] = "live"
        return resp
    maxw = max(200, min(int(payload.get("w") or 900), 1600))
    return _merged_from_export(idx, maxw, payload)


@router.post("/api/merge/merged")
@_api
def merge_merged_post(body: dict = None, w: int = 900, settings: str = "",
                      model: str = "", models: str = "", detector: str = "",
                      thresh: float = None, fmt: str = "", quality: int = None,
                      raw: str = "", compose: str = "", xseg_onnx: str = "",
                      no_swap_faces: str = ""):
    """POST /api/merge/merged —— 帧号放在 body 里，其余同 {idx} 版。"""
    payload = dict(_as_dict(body))
    _merge_query_into(payload, w=w, settings=settings, model=model, models=models,
                      detector=detector, thresh=thresh, fmt=fmt, quality=quality,
                      raw=raw, compose=compose, xseg_onnx=xseg_onnx,
                      no_swap_faces=no_swap_faces)
    idx = _body_idx(payload)
    snap = _cache_snap()
    mode = _cache_mode(snap)
    if mode:
        try:
            _check_cache_ready(snap)
        except _BadInput:
            mode = ""
    if mode:
        resp = _frame_payload(idx, payload, default_w=w)
        resp["source"] = "live"
        return resp
    return _merged_from_export(idx, max(200, min(int(payload.get("w") or 900), 1600)),
                               payload)


@router.post("/api/merge/export/cancel")
@_api
def merge_export_cancel():
    """终止正在跑的合成。幂等：没在跑、重复点，都返回 ok=true。"""
    with _merge_job_lock:
        running = bool(_merge_job.get("running"))
        was_cancelled = bool(_merge_job.get("cancelled"))
        _merge_job["cancelled"] = True
        if running:
            _merge_job["msg"] = "正在取消…"

    proc = getattr(_app_mod, "_merge_proc", None)
    try:
        alive = bool(proc is not None and proc.poll() is None)
    except Exception:
        alive = False

    if not running and not alive:
        with _merge_job_lock:
            _merge_job["running"] = False
            if _merge_job.get("ok") is None:
                _merge_job["msg"] = "没有正在进行的合成"
        return _ok(cancelled=True, running=False, already=True, killed=False,
                   reason="没有正在进行的合成")
    if was_cancelled and not alive:                 # 重复点：已经请求过、进程也不在了
        return _ok(cancelled=True, running=False, already=True, killed=False,
                   reason="已经在取消中")

    killed = _kill_proc_tree(proc) if alive else False
    _ensure_cancel_watch()
    return _ok(cancelled=True, running=False, already=False, killed=killed,
               reason="正在终止合成进程" if killed else "已请求取消，等子进程退出")


def _frame_image(idx: int):
    """给检测 / 分析取一帧：先做形位检查，别让 app._merge_frame_image 的 500 冒出来。"""
    if idx < 0:
        raise _BadInput("帧号不能是负数：%d" % idx)
    snap = _cache_snap()
    _check_cache_ready(snap)
    idx = _check_idx(snap, idx)
    if _cache_mode(snap) == "project":
        fp = Path(str(snap.get("srcdir"))) / (snap.get("names") or [])[idx]
        if not fp.is_file():
            raise _BadInput("图片不见了：%s" % fp, 404)
    try:
        frame, total = _merge_frame_image(idx)
    except HTTPException as e:
        raise _BadInput(str(e.detail), e.status_code)
    except Exception as e:
        raise _BadInput("取帧失败：%s" % str(e)[:200], 500)
    if frame is None:
        raise _BadInput("这一帧读不出来：%d" % idx, 422)
    return frame, total


def _detector_val(raw) -> str:
    """检测器入参 → 规范名（tinymog / retinaface_10g）；缺省 = tinymog。

    只有白名单里的两个名字能过。retinaface_500m / det_500m **明确 400**：
    这个模型不存在（4.1 已移除），以前它靠别名的兜底也能"成功"，属于静默回退。
    """
    key = str(raw or "").strip()
    low = key.lower().replace("-", "_")
    if low.endswith(".onnx"):
        low = low[:-len(".onnx")]
    if not low:
        return _DEFAULT_DETECTOR
    if low in _REMOVED_DETECTORS:
        raise _BadInput("检测器 %s 不存在（4.1 已移除 RetinaFace_500m），可选：%s"
                        % (key, " / ".join(_DETECTORS)))
    canon = _DET_ALIAS_API.get(low)
    if canon is None:
        raise _BadInput("未知检测器 %r，只认：%s" % (key, " / ".join(_DETECTORS)))
    return canon


def _detector(name=None):
    key = _detector_val(name)
    try:
        with _det_load_lock:              # 加载要几百毫秒，别让并发请求各建一份
            return _get_detector(key)
    except HTTPException as e:
        raise _BadInput(str(e.detail), e.status_code)
    except Exception as e:
        raise _BadInput("检测器加载失败（%s）：%s" % (key, str(e)[:160]), 500)


def _with_kps68(frame, boxes):
    # 给每个检测框补一条 kps68（原图坐标，68 行 2 列，两位小数）。
    #
    # 为什么在这里补：人脸库要的是 68 点仿射（5 点抖动大，旋转/缩放不稳），
    # 而前端手上只有检测框。让检测回包把 68 点直接带出去，前端点「＋」时
    # 就是纯搬运 —— extract 和 add 拿到的是同一份数，像素差才是 0。
    import numpy as np
    out = []
    for b in (boxes or []):
        try:
            item = dict(b)
        except Exception:
            out.append(b)
            continue
        lm = None
        try:
            lm = _frame_landmarks(frame, b, b.get("box"))
        except Exception:
            lm = None
        if lm is not None:
            try:
                item["kps68"] = np.asarray(lm, dtype=np.float64).round(2).tolist()
            except Exception:
                pass
        out.append(item)
    return out


@router.get("/api/merge/detect/{idx}")
@_api
def merge_detect(idx: int, model: str = _DEFAULT_DETECTOR, thresh: float = 0.5):
    """对当前帧做人脸检测，返回原图坐标的框。视频 / 项目两种模式都支持。

    model 缺省 = tinymog。retinaface_500m / det_500m / 任何没见过的名字 → 400。
    """
    t = _float_val("thresh", thresh, 0.0, 1.0, 0.5)
    det_name = _detector_val(model)
    frame, total = _frame_image(idx)
    det = _detector(det_name)
    with _det_run_lock:                    # thresh 是共享检测器上的可变字段
        det.thresh = t
        try:
            boxes = det.detect(frame)
        except Exception as e:
            raise _BadInput("检测失败：%s" % str(e)[:200], 500)
    h, w = frame.shape[:2]
    # 每张脸顺手回归一次 68 点（2D106）：前端点「＋」时直接把它交给
    # /api/merge/faces/add 落库，和人脸库 extract 的取点链路是同一条。
    # 回归不出来就不给这个键（add 那边会明确报 no_landmarks68，绝不退回 5 点）。
    boxes = _with_kps68(frame, boxes)
    return _ok(idx=int(idx), total=int(total or 0), width=int(w), height=int(h),
               model=model, detector=det_name, thresh=t, boxes=boxes)


@router.get("/api/merge/analyze")
@_api
def merge_analyze(step: int = 12, model: str = _DEFAULT_DETECTOR, thresh: float = 0.5):
    """按步长扫全片，统计每帧的人脸数，给时间轴的人脸轨用。
    486 帧 / step=12 → 41 次推理，约 20 秒。model 缺省 = tinymog。"""
    step = _int_val("step", step, 1, 500, 12)
    det_name = _detector_val(model)
    t = _float_val("thresh", thresh, 0.0, 1.0, 0.5)
    snap = _cache_snap()
    total = int(snap.get("frames") or 0)
    _check_cache_ready(snap)
    if total <= 0:
        raise _BadInput("还没有打开素材（先 /api/merge/open 或 /api/merge/prepare）", 409)
    det = _detector(det_name)

    marks, idx, scanned, broken = [], 0, 0, ""
    while idx < total:
        try:
            frame, _n = _frame_image(idx)
        except _BadInput as e:              # 扫到一半文件没了：别装作扫完了
            broken = e.msg
            break
        with _det_run_lock:
            det.thresh = t
            try:
                boxes = det.detect(frame)
            except Exception as e:
                raise _BadInput("检测失败（第 %d 帧）：%s" % (idx, str(e)[:200]), 500)
        marks.append({"idx": idx, "count": len(boxes),
                      "box": boxes[0]["box"] if boxes else None})
        scanned += 1
        idx += step
    return _ok(total=total, step=step, scanned=scanned, model=model,
               detector=det_name, thresh=t, marks=marks, truncated=bool(broken),
               error=broken)


@router.post("/api/merge/project")
@_api
def merge_project(body: dict):
    """看一眼目录是不是能用的 DFL 项目：有几张帧、有没有 aligned、带不带 landmarks。"""
    body = _as_dict(body)
    p = _norm_path(body.get("path"))
    if p is None:
        raise _BadInput("缺少 path：要一个项目目录")
    if not p.exists():
        raise _BadInput("目录不存在：%s" % p, 404)
    if not p.is_dir():
        raise _BadInput("不是目录：%s" % p)
    names, srcdir, kind = _dfl_project_scan(p)
    aligned = p / "aligned"
    sample = names[:24]
    lm = 0
    for n in sample:
        try:
            if _dfljpg_dict(Path(srcdir) / n).get("landmarks") is not None:
                lm += 1
        except Exception:
            continue
    return _ok(path=str(p), kind=kind, total=len(names),
               has_aligned=aligned.is_dir(),
               aligned_dir=str(aligned) if aligned.is_dir() else "",
               frames_dir=srcdir,
               with_landmarks=lm,     # 采样 24 张的结果，不是全量
               sampled=len(sample), sample=names[:6])


@router.get("/api/merge/info")
@_api
def merge_info():
    """当前打开的东西。names 可能几万条，不回传（只给条数）。"""
    snap = _cache_snap()
    names = snap.get("names") or []
    payload = {k: v for k, v in snap.items() if k not in ("names", "list")}
    mode = _cache_mode(snap)
    stale = False
    if mode == "project":
        stale = not Path(str(snap.get("srcdir") or "")).is_dir()
    elif mode == "video":
        stale = not Path(str(snap.get("path") or "")).is_file()
    opened = snap.get("opened_at")
    payload.update({"mode": mode, "stale": bool(stale) or bool(snap.get("stale")),
                    "names_count": len(names),
                    "age": round(time.time() - float(opened), 1) if opened else None})
    return _ok(**payload)


# ============================================================
# 视频播放（HTTP Range）
# 前端 <video> 拖时间轴要 206 Partial Content，FileResponse 不支持 Range
# （它只会回 200 + 整个文件，浏览器拿不到 Content-Range 就不给拖），
# 所以这里自己解析 Range 头、自己读片段。
# ============================================================
_VIDEO_CTYPE = {
    ".mp4": "video/mp4", ".m4v": "video/mp4", ".mov": "video/quicktime",
    ".mkv": "video/x-matroska", ".webm": "video/webm", ".avi": "video/x-msvideo",
    ".flv": "video/x-flv", ".wmv": "video/x-ms-wmv", ".ogv": "video/ogg",
    ".ts": "video/mp2t", ".m2ts": "video/mp2t", ".mts": "video/mp2t",
    ".3gp": "video/3gpp", ".mpg": "video/mpeg", ".mpeg": "video/mpeg",
}
_VIDEO_CHUNK = 512 * 1024


def _video_ctype(p) -> str:
    return _VIDEO_CTYPE.get(Path(str(p)).suffix.lower(), "application/octet-stream")


def _inside_repo(p) -> bool:
    """路径是不是在仓库里（Windows 上大小写不敏感）。"""
    try:
        root = os.path.normcase(os.path.abspath(str(_REPO_ROOT)))
        full = os.path.normcase(os.path.abspath(str(p)))
    except Exception:
        return False
    return full == root or full.startswith(root + os.sep)


def _raw_is_absolute(raw) -> bool:
    """界面上传来的原始串本身是不是绝对路径（D:/x、\\\\server\\share、/x）。"""
    s = str(raw or "").strip().strip('"').strip("'").strip()
    if not s:
        return False
    try:
        if Path(s).is_absolute():
            return True
    except Exception:
        pass
    try:
        import ntpath
        drive, _rest = ntpath.splitdrive(s)
        return bool(drive)
    except Exception:
        return False


def _parse_range(raw, size: int):
    """Range: bytes=a-b -> (start, end) 闭区间。

    返回 (None, None) 表示「没这个头 / 认不了」→ 调用方回 200 全量；
    返回 ("bad", None) 表示区间不合法 / 越界 → 调用方回 416。
    """
    text = str(raw or "").strip().lower()
    if not text or not text.startswith("bytes="):
        return None, None                 # 别的单位（items=…）按 RFC 忽略
    spec = text[6:].split(",")[0].strip()  # 多段只认第一段
    if not spec or "-" not in spec:
        return None, None
    head, _, tail = spec.partition("-")
    head, tail = head.strip(), tail.strip()
    if not head and not tail:
        return None, None
    try:
        if not head:                      # "-500" = 最后 500 字节
            n = int(tail)
            if n <= 0:
                return "bad", None
            start, end = max(0, size - n), size - 1
        else:
            start = int(head)
            end = int(tail) if tail else size - 1
    except ValueError:
        return None, None
    if size <= 0 or start < 0 or start >= size:
        return "bad", None
    end = min(end, size - 1)
    if end < start:
        return "bad", None
    return start, end


def _file_chunks(path, start: int, length: int, chunk: int = _VIDEO_CHUNK):
    """从 start 起读 length 字节（生成器：客户端断开时 finally 会关文件）。"""
    fh = open(str(path), "rb")
    try:
        fh.seek(int(start))
        left = int(length)
        while left > 0:
            buf = fh.read(min(chunk, left))
            if not buf:
                break
            left -= len(buf)
            yield buf
    finally:
        try:
            fh.close()
        except Exception:
            pass


@router.get("/api/merge/video")
@_api
def merge_video(request: Request, path: str = ""):
    """把视频文件原样吐给 <video>。支持 Range（206），没有 Range 就 200 全量流式。"""
    raw = str(path or "")
    p = _norm_path(raw)
    if p is None:
        raise _BadInput("缺少 path：要一个视频文件路径")
    # 越界先判：「相对路径」一律锚在仓库根（_norm_path），所以只有显式的绝对路径
    # 才可能跑出仓库；出了仓库还要求它「原样就是绝对路径且真的存在」，否则 400。
    # （先判越界再看存在性：../../ 这种穿越不能因为文件恰好不存在就降级成 404。）
    if not _inside_repo(p):
        if not (_raw_is_absolute(raw) and p.is_file()):
            raise _BadInput("路径越界：%s 既不在仓库（%s）里，也不是一个存在的绝对路径"
                            % (p, _REPO_ROOT), 400)
    if not p.is_file():
        raise _BadInput("文件不存在：%s" % p, 404)
    try:
        size = int(p.stat().st_size)
    except OSError as e:
        raise _BadInput("读不了文件大小：%s（%s）" % (p, e), 500)

    headers = {"accept-ranges": "bytes",
               "content-type": _video_ctype(p),
               "cache-control": "no-store"}
    try:
        headers["last-modified"] = _httpdate(p.stat().st_mtime, usegmt=True)
    except Exception:
        pass

    start, end = _parse_range(request.headers.get("range"), size)
    if start == "bad":
        body = {"ok": False, "error": "Range 越界：bytes=%s，文件只有 %d 字节"
                                     % (str(request.headers.get("range"))[6:], size),
                "detail": "Range 越界"}
        return JSONResponse(status_code=416, content=body,
                            headers={"accept-ranges": "bytes",
                                     "content-range": "bytes */%d" % size})

    if start is None:
        status, start, end = 200, 0, max(0, size - 1)
    else:
        status = 206
        headers["content-range"] = "bytes %d-%d/%d" % (start, end, size)
    length = max(0, end - start + 1)
    headers["content-length"] = str(length)
    return StreamingResponse(_file_chunks(p, start, length),
                             status_code=status, headers=headers)


# ============================================================
# 单帧实时预览
#
# 换脸跑不了在这边：merge 引擎（MergeStudio / facelib / onnxruntime-gpu）装在
# 4.0 那份 python 里，只能在那个解释器里 import。每帧起一个子进程要 3~6 秒
# （import + 加载 DFM），拖动时间轴根本没法用 —— 所以这里养一个常驻 worker：
#
#   父进程（4.1 的 python，有 FastAPI）              子进程（4.0 的 python）
#   ─────────────────────────────────────           ─────────────────────────
#   取帧 grab / 检测 / 68 点（本地缓存复用）   ->    换脸 + 遮罩 + 合成（DFM 只加载一次）
#   JPEG 帧 ──── JSON line（stdin）────>            MergeMaskedFace(frame, lm68, cfg, pred)
#   <──── JSON line（stdout）──── JPEG
#
# 几个硬约定：
#   · 协议走 ASCII 转义的 JSON lines（模型名/路径可能是中文，不能让子进程的
#     stdin 按 cp936 解）；worker 里把 fd 1 换成 dup 出来的协议管道，引擎那些
#     非 UTF-8 的日志全部赶去 stderr，不会插进 JSON 中间。
#   · 一次只有一个请求在飞（_PreviewWorker._lock），所以不需要请求路由表。
#   · 单帧干活上限 15s；超时就杀掉整个 worker（ONNX 推理没法从外部打断，
#     杀进程是唯一可靠的取消），下次请求自动重启。冷启动/换模型的加载时间
#     单独算，不算进 15s（否则第一次点预览必然是 504）。
#   · worker 空闲 10 分钟自动退出；崩溃了下次请求自动重启。
# ============================================================
_PREVIEW_WORK_TIMEOUT = 15.0       # 单帧「干活」上限（秒）—— 超了回 504
_PREVIEW_LOAD_TIMEOUT = 120.0      # 冷启动 / 换模型加载的上限（单独计时）
_PREVIEW_BOOT_TIMEOUT = 90.0       # 进程 + 引擎 import 的上限
_PREVIEW_IDLE_EXIT = 600.0         # 空闲 10 分钟自动退出
_PREVIEW_SEND_QUALITY = 95         # 送给 worker 的帧 JPEG 质量
_PREVIEW_QUALITY = 92              # 回给界面的 JPEG 质量
_PREVIEW_HIT = {}                 # 上一帧的整份响应（只留 1 条：同一 t 连点两下时整份复用）
_PREVIEW_HIT_LOCK = threading.Lock()
_WORKER_MAX_MODELS = 3            # worker 里 DFM 的常驻上限（它自己报 max_models，这里是兜底）
_PREVIEW_MAX_MODELS = 8           # 一次请求最多声明几个模型（真正常驻的仍只有 3 个）
_PREVIEW_MAX_SWAPS = 8            # 单帧最多换几张脸（多模型模式下检测到的脸全换）

# 预览 worker 源码：写进系统 Temp（不进仓库），内容变了就换文件名重新落盘。
_PREVIEW_WORKER_SRC = r'''# -*- coding: utf-8 -*-
"""4.1 合成器 · 单帧预览 worker（常驻；父进程用 4.0 引擎的 python 起它）。

用法: python preview_worker.py <4.0引擎根>
协议: stdin/stdout 上的 JSON lines（ASCII 转义）。
"""
import base64
import io
import json
import os
import sys
import time
from collections import OrderedDict

# 协议通道：先 dup 一份 stdout，再把 fd 1 指到 stderr。
# 引擎里 torch / onnxruntime / facelib 会往 fd 1 打非 UTF-8 的中文日志，
# 混进 JSON lines 会让父进程整行解码直接炸。
_OUT = os.fdopen(os.dup(1), "w", encoding="utf-8", newline="\n")
os.dup2(2, 1)
sys.stdout = sys.stderr
_IN = io.TextIOWrapper(sys.stdin.buffer, encoding="utf-8",
                       errors="replace", newline="\n")


def _say(obj):
    try:
        _OUT.write(json.dumps(obj, ensure_ascii=True) + "\n")
        _OUT.flush()
    except Exception:
        pass


def _mask2d(t, size, np, cv2):
    m = np.squeeze(t)
    if m.ndim == 3:
        m = m[:, :, 0]
    m = m.astype(np.float32)
    if m.shape[:2] != (size, size):
        m = cv2.resize(m, (size, size), interpolation=cv2.INTER_CUBIC)
    return m


def _paste_face(frame, face, box, np, cv2):
    """非常规尺寸的合成结果（raw-* 模式回的是脸本身）按检测框羽化贴回原帧。"""
    h, w = frame.shape[:2]
    try:
        x, y, bw, bh = [int(v) for v in (box or [])]
    except Exception:
        x = y = 0
        bw, bh = w, h
    bw = max(8, min(int(bw), w))
    bh = max(8, min(int(bh), h))
    x = max(0, min(int(x), w - bw))
    y = max(0, min(int(y), h - bh))
    small = cv2.resize(face, (bw, bh), interpolation=cv2.INTER_LANCZOS4)
    mask = np.zeros((bh, bw), np.float32)
    cv2.ellipse(mask, (bw // 2, bh // 2),
                (max(1, bw // 2 - 2), max(1, bh // 2 - 2)), 0, 0, 360, 1.0, -1)
    mask = cv2.GaussianBlur(mask, (0, 0), max(1.0, min(bw, bh) * 0.03))[..., None]
    roi = frame[y:y + bh, x:x + bw].astype(np.float32) / 255.0
    blended = roi * (1.0 - mask) + (small.astype(np.float32) / 255.0) * mask
    out = frame.copy()
    out[y:y + bh, x:x + bw] = np.clip(blended * 255.0, 0, 255).astype(np.uint8)
    return out


def main():
    t_boot = time.time()
    engine = sys.argv[1] if len(sys.argv) > 1 else ""
    if engine and engine not in sys.path:
        sys.path.insert(0, engine)
    import numpy as np
    import cv2
    try:
        from MergeStudio.core.export_pipeline import _setup_dll_paths
        _setup_dll_paths()               # CUDA/cuDNN 的 DLL 先塞进搜索路径
    except Exception:
        pass
    # 再兜一道：cublasLt64_12.dll / cuDNN 9 都在 torch/lib 里，那个目录不在 DLL
    # 搜索路径上时 ORT 会报 Error 126 然后**静默**跑 CPU（11ms → 140ms 一张）。
    # worker 是独立进程，父进程补过的路径带不过来，必须在这里自己加一遍。
    _ort_dirs = []
    try:
        for _d in (os.path.join(sys.prefix, "Lib", "site-packages", "torch", "lib"),
                   os.path.join(os.path.dirname(sys.executable), "Lib",
                                "site-packages", "torch", "lib")):
            if _d and os.path.isdir(_d) and _d not in _ort_dirs:
                _ort_dirs.append(_d)
        _ort_cookies = []
        for _d in _ort_dirs:
            try:
                if hasattr(os, "add_dll_directory"):
                    _ort_cookies.append(os.add_dll_directory(_d))   # 句柄留着别被回收
            except Exception:
                pass
        if _ort_dirs:
            os.environ["PATH"] = os.pathsep.join(_ort_dirs + [os.environ.get("PATH", "")])
    except Exception:
        _ort_dirs = []
    import onnxruntime as ort
    import MergeStudio.core.merger as _merger
    from MergeStudio.core.merger import MergeMaskedFace
    from MergeStudio.core.config import MergerConfigMasked, ctm_dict
    from facelib import FaceType

    ct_map = {str(v).strip().lower(): int(k) for k, v in ctm_dict.items()}
    ct_map["none"] = 0
    # "sot" 是 ct_mode 契约里的直白写法；ctm_dict 里只有 "sot-m"（带遮罩的 SOT，
    # 见 merger.py 的 ct_functions[7]）。不补这一行的话，ct_mode="sot" 会落到
    # .get(...,0) = 0（none），改参数画面不动 —— 正是这次要修的那类透传洞。
    ct_map.setdefault("sot", ct_map.get("sot-m", 7))

    # 常驻的 DFM。显存保护：最多 _MAX_MODELS 个（4G 卡上一张脸一个 1.6GB 的 DFM，
    # 三个就是极限），超了先卸最久没用过的那个（LRU），再建新的。
    # OrderedDict 的顺序就是「最近用过」的顺序：尾巴是刚用的，头上是最久没用的。
    _MAX_MODELS = 3
    models = OrderedDict()   # dfm -> (sess, 输入名, 边长, 是否 NCHW)
    xsegs = {}       # xseg -> extract 函数

    def _vram_flush():
        """把刚放掉的那些显存真的还给驱动：先 gc 收引用，再清 CUDA 缓存池。

        torch 只在它已经加载进这个进程、且 CUDA 也初始化过的时候才用 ——
        为了 empty_cache() 去 import torch 并建一个 CUDA context，纯粹是净亏。
        （走 MergeStudio.core.merger 的进程其实已经被 torch 带进来了，所以这里
        那一步是真会执行到的那一条；没加载 torch 就只剩 gc。）
        """
        try:
            import gc
            gc.collect()
            gc.collect()               # 第二轮收掉第一轮析构出来的循环引用
        except Exception:
            pass
        try:
            _torch = sys.modules.get("torch")
            if _torch is not None and _torch.cuda.is_initialized():
                _torch.cuda.empty_cache()      # CUDA 的缓存分配器：不清就还挂在卡上
                try:
                    _torch.cuda.ipc_collect()
                except Exception:
                    pass
        except Exception:
            pass

    def _drop(path):
        """真的把一个 DFM 从显存放掉：掐断 session 引用 → gc → 清 CUDA 缓存。"""
        ent = models.pop(path, None)
        if ent is not None:
            try:
                ent = list(ent)
                ent[0] = None                  # 先把 InferenceSession 的那一栏掐掉
            except Exception:
                pass
        ent = None
        del ent
        _vram_flush()
        print("[merge-preview] unload %s" % os.path.basename(str(path)),
              file=sys.stderr, flush=True)
        return str(path)

    def _unload(keep=""):
        """只留 keep 这一个 DFM，其它当场卸掉；返回被卸掉的文件列表。

        「cmd=unload」走这条（显式要求只留一个）。平时换模型不走这里，走
        _evict_lru()：那是按上限淘汰，不是一刀切。
        顺序要紧：置 None / del（打断 session 引用）→ gc → 清缓存，最后才建新的。
        """
        keep = str(keep or "")
        return [_drop(path) for path in [p for p in list(models) if str(p) != keep]]

    def _evict_lru(keep=""):
        """常驻满了：卸掉最久没用过的那个（keep 里的不动）。卸不了就回 ""。"""
        keep = str(keep or "")
        for path in list(models):              # OrderedDict：排前面 = 最久没用过
            if str(path) == keep:
                continue
            return _drop(path)
        return ""

    def _load(dfm, rid):
        hit = models.get(dfm)
        if hit is not None:
            models.move_to_end(dfm)        # 用一次就算「最近用过」，别被 LRU 冤枉淘汰
            return hit                     # 幂等：已经在显存里，原样返回，不卸也不重载
        # 显存保护：先数一下常驻几个，到上限就先卸最旧的，再建新的
        while len(models) >= _MAX_MODELS:
            gone = _evict_lru()
            if not gone:
                break
            _say({"id": rid, "stage": "model_evicted", "file": os.path.basename(gone),
                  "models": sorted(models)})
        _say({"id": rid, "stage": "loading_model", "file": os.path.basename(dfm),
              "dfm": str(dfm), "slots": len(models)})
        sess = ort.InferenceSession(dfm, providers=["CUDAExecutionProvider",
                                                    "CPUExecutionProvider"])
        shape = list(sess.get_inputs()[0].shape)
        nchw = len(shape) >= 4 and shape[1] in (1, 3)
        try:
            size = int(shape[2] if nchw else shape[1])
        except Exception:
            size = 256
        models[dfm] = (sess, sess.get_inputs()[0].name, size, nchw)
        models.move_to_end(dfm)
        try:
            providers = list(sess.get_providers())
        except Exception:
            providers = []
        _say({"id": rid, "stage": "model_ready", "size": size, "dfm": str(dfm),
              "providers": providers,
              "models": sorted(models),      # 父进程照这个对齐 _models（现在最多 3 个）
              "models_lru": list(models), "max_models": _MAX_MODELS,
              "load_ms": round((time.time() - t_boot) * 1000.0, 1)})
        return models[dfm]

    def _xseg(path, rid):
        if not path:
            return None
        if path in xsegs:
            return xsegs[path]
        xs = ort.InferenceSession(path, providers=["CUDAExecutionProvider",
                                                   "CPUExecutionProvider"])
        xi = xs.get_inputs()[0].name
        xsh = list(xs.get_inputs()[0].shape)
        nchw = len(xsh) >= 4 and xsh[1] in (1, 3)

        def _extract(face_in, _xs=xs, _xi=xi, _nchw=nchw):
            oh, ow = face_in.shape[:2]
            fr = face_in if (oh, ow) == (256, 256) else cv2.resize(
                face_in, (256, 256), interpolation=cv2.INTER_CUBIC)
            fc = np.clip(fr, 0, 1) if float(fr.max()) < 2.0 else fr
            rgb = cv2.cvtColor(fc, cv2.COLOR_BGR2RGB).astype(np.float32)
            if float(rgb.max()) > 1.0:
                rgb = rgb / 255.0
            inp = np.transpose(rgb, (2, 0, 1))[None] if _nchw else rgb[None]
            out = np.squeeze(_xs.run(None, {_xi: inp})[0])
            if out.shape[:2] != (oh, ow):
                out = cv2.resize(out, (ow, oh), interpolation=cv2.INTER_CUBIC)
            return np.clip(out, 0, 1).astype(np.float32)
        xsegs[path] = _extract
        return _extract

    def _predictor(sess, inp_name, size, nchw):
        """DFM 前向：和 export_pipeline 里那个闭包逐行等价（脸 / prd 遮罩 / dst 遮罩）。"""
        def _pred(face_img):
            if face_img.shape[0] != size or face_img.shape[1] != size:
                face_img = cv2.resize(face_img, (size, size),
                                      interpolation=cv2.INTER_LANCZOS4)
            img = face_img.astype(np.float32)
            inp = np.transpose(img, (2, 0, 1))[None] if nchw else img[None]
            outs = sess.run(None, {inp_name: inp})
            face_out = None
            for o in outs:
                if o.ndim == 4 and o.shape[-1] == 3:
                    face_out = o
                    break
            if face_out is None:
                face_out = outs[0]
                if face_out.shape[-1] == 1:
                    face_out = np.repeat(face_out, 3, axis=-1)
            pm = dm = None
            if len(outs) >= 3:
                pm = _mask2d(outs[0], size, np, cv2)
                dm = _mask2d(outs[2], size, np, cv2)
            while face_out.ndim >= 5:
                face_out = face_out[0]
            if face_out.ndim == 4:
                face_out = (np.transpose(face_out[0], (1, 2, 0)) if nchw
                            else face_out[0])
            if face_out.ndim == 2:
                face_out = np.stack([face_out] * 3, axis=-1)
            if face_out.shape[-1] > 3:
                face_out = face_out[:, :, :3]
            elif face_out.shape[-1] == 1:
                face_out = np.repeat(face_out, 3, axis=-1)
            if face_out.shape[:2] != (size, size):
                face_out = cv2.resize(face_out, (size, size),
                                      interpolation=cv2.INTER_LANCZOS4)
            return (face_out, pm, dm)
        return _pred

    _say({"ready": True, "boot_ms": round((time.time() - t_boot) * 1000.0, 1),
          "providers": ort.get_available_providers(), "engine": engine,
          "max_models": _MAX_MODELS, "pid": os.getpid()})

    for line in _IN:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except Exception as e:
            _say({"ok": False, "error": "bad json: %s" % e})
            continue
        rid = req.get("id")
        cmd = str(req.get("cmd") or "")
        if cmd == "exit":
            break
        if cmd == "unload":
            # 手动放显存：{"cmd":"unload"} 全放；{"cmd":"unload","dfm":X} 只留 X
            keep = str(req.get("dfm") or "")
            if keep not in models:
                keep = ""                  # 要留的那个本来就没加载，那就全放
            gone = _unload(keep)
            _say({"id": rid, "ok": True, "unload": True, "unloaded": len(gone),
                  "keep": keep, "models": sorted(models),
                  "max_models": _MAX_MODELS})
            continue
        if cmd == "warm":
            # 只把 DFM / XSeg 装进显存、空跑一遍前向，不取帧、不合成 —— warm 用。
            # 多模型：dfms 是一次要热的整组（最多常驻 _MAX_MODELS 个，超了 LRU 淘汰），
            # dfm 是老的单个写法（向后兼容，排在最前面）。
            dfms = [str(x) for x in (req.get("dfms") or []) if str(x or "").strip()]
            one = str(req.get("dfm") or "")
            if one and one not in dfms:
                dfms.insert(0, one)
            if not dfms:
                _say({"id": rid, "ok": False, "error": "DFM 不存在: "})
                continue
            missing = [p for p in dfms if not os.path.isfile(p)]
            if missing:
                _say({"id": rid, "ok": False, "error": "DFM 不存在: %s" % missing[0]})
                continue
            t_w = time.time()
            per = []
            try:
                for p in dfms:
                    t_m = time.time()
                    _sess, _in, size, _nchw = _load(p, rid)
                    # 第一次前向要把 cuDNN/TRT 的句柄和 kernel 建起来（实测 ~1.1s）。
                    # 用一张 zeros 走一遍把这份开销留在预热里 —— 不取帧、不合成。
                    t_p = time.time()
                    _predictor(_sess, _in, size, _nchw)(
                        np.zeros((size, size, 3), np.float32))
                    per.append({"dfm": p, "size": size,
                                "load_ms": round((t_p - t_m) * 1000.0, 1),
                                "first_ms": round((time.time() - t_p) * 1000.0, 1)})
            except Exception as e:
                _say({"id": rid, "ok": False,
                      "error": "%s: %s" % (e.__class__.__name__, str(e)[:300])})
                continue
            xp = str(req.get("xseg") or "")
            xs_err = ""
            if xp:
                try:
                    _xs_fn = _xseg(xp, rid)
                    _xs_fn(np.zeros((256, 256, 3), np.float32))   # 同上，先付一次
                except Exception as e:
                    xs_err = "%s: %s" % (e.__class__.__name__, str(e)[:200])
            head = per[0] if per else {"size": 256, "first_ms": 0.0}
            _say({"id": rid, "ok": True, "warm": True, "size": head["size"],
                  "xseg": bool(xp) and not xs_err, "xseg_error": xs_err,
                  "first_ms": head["first_ms"],
                  "warm_ms": round((time.time() - t_w) * 1000.0, 1),
                  "per_model": per, "count": len(per),
                  "pid": os.getpid(), "models": sorted(models),
                  "models_lru": list(models), "max_models": _MAX_MODELS})
            continue
        if cmd == "face_debug":
            # 单脸六宫格：只跑**这一张脸**的一次合成，把中间量拆成六张 PNG 发回去。
            # 走的就是普通的 MergeMaskedFace，只是多带一个 capture dict ——
            # capture 只写不读，一个像素都不改，所以第 6 格「merged」和用户在画面上
            # 看到的那张脸是同一份像素（脸发蒙时它是最直接的证据）。
            try:
                raw = base64.b64decode(req.get("frame") or "")
                frame = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
                if frame is None:
                    raise RuntimeError("帧解码失败")
                dfm = str(req.get("dfm") or "")
                if not dfm or not os.path.isfile(dfm):
                    raise RuntimeError("DFM 不存在: %s" % dfm)
                c = req.get("config") or {}
                cfg = MergerConfigMasked(
                    face_type=str(c.get("face_type") or "whole_face"),
                    mode=str(c.get("mode") or "seamless"),
                    mask_mode=max(0, min(9, int(c.get("mask_mode", 6)))),
                    color_transfer_mode=ct_map.get(
                        str(c.get("color_transfer_mode") or "none").strip().lower(), 0),
                    erode_mask_modifier=max(-100, min(100, int(c.get("erode_mask_modifier", 0)))),
                    blur_mask_modifier=max(-100, min(100, int(c.get("blur_mask_modifier", 0)))),
                    show_debug=False,
                )
                lm = np.array(req.get("landmarks") or [], dtype=np.float32)
                if lm.ndim != 2 or lm.shape[1] < 2 or lm.shape[0] < 5:
                    raise RuntimeError("68 点不合法: %r" % (getattr(lm, "shape", None),))
                sess, inp_name, size, nchw = _load(dfm, rid)
                xs = _xseg(str(req.get("xseg") or ""), rid)
                _merger._model_input_size = size
                cap = {}
                MergeMaskedFace(frame, lm, cfg, _predictor(sess, inp_name, size, nchw),
                                xseg_256_extract_func=xs, capture=cap)
                panel_size = int(req.get("panel_size") or 256)
                panels = []
                for key, label, img in _merger.build_debug_panels(cap, panel_size):
                    ok_png, buf = cv2.imencode(".png", img,
                                               [int(cv2.IMWRITE_PNG_COMPRESSION), 3])
                    if not ok_png:
                        raise RuntimeError("PNG 编码失败: %s" % key)
                    panels.append({"key": key, "label": label,
                                   "w": int(img.shape[1]), "h": int(img.shape[0]),
                                   "png": base64.b64encode(buf.tobytes()).decode("ascii")})

                def _mean(k):
                    try:
                        return round(float(np.asarray(cap.get(k), np.float32).mean()), 4)
                    except Exception:                        # noqa: BLE001
                        return None
                _say({"id": rid, "ok": True, "cmd": "face_debug", "size": size,
                      "panels": panels,
                      "stats": {"mask_prd": _mean("prd_face_mask"),
                                "mask_dst": _mean("dst_face_mask"),
                                "mask_effective": _mean("effective_mask")}})
            except Exception as e:
                import traceback
                traceback.print_exc(file=sys.stderr)
                _say({"id": rid, "ok": False,
                      "error": "%s: %s" % (e.__class__.__name__, str(e)[:300])})
            continue
        t0 = time.time()
        try:
            raw = base64.b64decode(req.get("frame") or "")
            frame = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
            if frame is None:
                raise RuntimeError("帧解码失败")

            c = req.get("config") or {}
            # config 就是父进程 _merge_settings_to_config() 的产物，**这一帧的参数
            # 只从这里来**：不读任何全局/缓存值，也不回落到"默认参数"。
            # 逐项都在这儿落地，少一项 = 那一项改了画面不动。
            cfg = MergerConfigMasked(
                face_type=str(c.get("face_type") or "whole_face"),
                mode=str(c.get("mode") or "seamless"),
                mask_mode=max(0, min(9, int(c.get("mask_mode", 6)))),
                color_transfer_mode=ct_map.get(
                    str(c.get("color_transfer_mode") or "none").strip().lower(), 0),
                erode_mask_modifier=max(-100, min(100, int(c.get("erode_mask_modifier", 0)))),
                blur_mask_modifier=max(-100, min(100, int(c.get("blur_mask_modifier", 0)))),
                show_debug=False,
            )
            q = int(req.get("jpeg_quality") or 92)
            q = max(30, min(100, q))

            faces = req.get("faces") or None      # 多模型：[{dfm, landmarks, box}, …]
            if not faces:
                # ---- 老的单模型请求：一行行为都不改 ----
                dfm = str(req.get("dfm") or "")
                if not dfm or not os.path.isfile(dfm):
                    raise RuntimeError("DFM 不存在: %s" % dfm)
                sess, inp_name, size, nchw = _load(dfm, rid)
                xs = _xseg(str(req.get("xseg") or ""), rid)
                _say({"id": rid, "stage": "swap"})
                lm = np.array(req.get("landmarks") or [], dtype=np.float32)
                if lm.ndim != 2 or lm.shape[1] < 2 or lm.shape[0] < 5:
                    raise RuntimeError("68 点不合法: %r" % (getattr(lm, "shape", None),))
                _merger._model_input_size = size     # MergeMaskedFace 读这个模块级变量
                fo, _fmask = MergeMaskedFace(frame, lm, cfg,
                                             _predictor(sess, inp_name, size, nchw),
                                             xseg_256_extract_func=xs)
                if fo is None or not hasattr(fo, "shape"):
                    raise RuntimeError("MergeMaskedFace 没返回图")
                out = fo
                if out.ndim != 3 or out.shape[2] != 3:
                    raise RuntimeError("合成结果形状不对: %r" % (getattr(out, "shape", None),))
                if out.shape[:2] != frame.shape[:2]:
                    if str(cfg.mode) == "debug":
                        pass                          # 调试视图本身就是输出
                    else:
                        out = _paste_face(frame, out, req.get("box"), np, cv2)
                per = [{"dfm": dfm, "size": size,
                        "ms": round((time.time() - t0) * 1000.0, 1)}]
            else:
                # ---- 多模型：同一张帧上按顺序一张脸一张脸贴 ----
                # load 是这次「声明要用」的整组；先装进显存，免得到第 2 张脸才付加载时间。
                # 超过 _MAX_MODELS 的部分 _load 会自动 LRU 淘汰，这里不用管上限。
                for p in [str(x) for x in (req.get("load") or [])]:
                    if p and os.path.isfile(p) and p not in models:
                        _load(p, rid)
                xs = _xseg(str(req.get("xseg") or ""), rid)
                out = frame
                per = []
                for k, fc in enumerate(faces):
                    fc = fc or {}
                    _say({"id": rid, "stage": "swap", "face": k,
                          "faces": len(faces)})
                    f_dfm = str(fc.get("dfm") or "")
                    if not f_dfm or not os.path.isfile(f_dfm):
                        raise RuntimeError("第 %d 张脸的 DFM 不存在: %s" % (k + 1, f_dfm))
                    sess_k, in_k, size_k, nchw_k = _load(f_dfm, rid)
                    lm_k = np.array(fc.get("landmarks") or [], dtype=np.float32)
                    if lm_k.ndim != 2 or lm_k.shape[1] < 2 or lm_k.shape[0] < 5:
                        raise RuntimeError("第 %d 张脸的 68 点不合法: %r"
                                           % (k + 1, getattr(lm_k, "shape", None)))
                    t_k = time.time()
                    _merger._model_input_size = size_k
                    # 调试视图本身就是整帧输出，再叠第二张会把上一张盖掉 —— 每张都
                    # 从原帧出发；正常模式贴的就是上一次的结果，多张脸能叠在一起。
                    src = frame if str(cfg.mode) == "debug" else out
                    fo, _fmask = MergeMaskedFace(src, lm_k, cfg,
                                                 _predictor(sess_k, in_k, size_k, nchw_k),
                                                 xseg_256_extract_func=xs)
                    if fo is None or not hasattr(fo, "shape"):
                        raise RuntimeError("第 %d 张脸：MergeMaskedFace 没返回图" % (k + 1))
                    if fo.ndim != 3 or fo.shape[2] != 3:
                        raise RuntimeError("第 %d 张脸：合成结果形状不对: %r"
                                           % (k + 1, getattr(fo, "shape", None)))
                    if fo.shape[:2] != frame.shape[:2]:
                        if str(cfg.mode) == "debug":
                            out = fo                      # 调试视图本身就是输出
                        else:
                            out = _paste_face(out, fo, fc.get("box"), np, cv2)
                    else:
                        out = fo
                    per.append({"dfm": f_dfm, "size": size_k,
                                "ms": round((time.time() - t_k) * 1000.0, 1)})
            # 编码：默认 jpg（预览一直是这条，行为不变）；fmt=png 给「整帧落盘」用
            # —— 项目模式导出要把 merged 当中间素材再编码，JPEG 每帧掉一次画质没意义。
            fmt = str(req.get("fmt") or "jpg").strip().lower()
            if fmt == "png":
                ok, buf = cv2.imencode(".png", out, [int(cv2.IMWRITE_PNG_COMPRESSION), 3])
                if not ok:
                    raise RuntimeError("PNG 编码失败")
            else:
                fmt = "jpg"
                ok, buf = cv2.imencode(".jpg", out, [int(cv2.IMWRITE_JPEG_QUALITY), q])
                if not ok:
                    raise RuntimeError("JPEG 编码失败")
            _say({"id": rid, "ok": True,
                  "size": (per[0]["size"] if per else 256),
                  "fmt": fmt,
                  "jpeg": base64.b64encode(buf.tobytes()).decode("ascii"),
                  "swap_ms": round((time.time() - t0) * 1000.0, 1),
                  "swaps": len(per), "per_face": per,
                  "models": sorted(models), "models_lru": list(models),
                  "max_models": _MAX_MODELS})
        except Exception as e:
            import traceback
            traceback.print_exc(file=sys.stderr)
            _say({"id": rid, "ok": False,
                  "error": "%s: %s" % (e.__class__.__name__, str(e)[:300])})


main()
'''


class _PreviewWorker:
    """常驻预览子进程的管家：启动 / 一帧一请求 / 超时杀进程 / 空闲退出 / 崩了重启。"""

    def __init__(self):
        self._lock = threading.Lock()          # 同时只让一个请求在飞
        self._proc = None
        self._stdin = None
        self._q = None
        self._out_t = None
        self._err_t = None
        self._idle_t = None
        self._tail = collections.deque(maxlen=40)
        self._seq = 0
        self._spawns = 0
        self._boot_ms = None
        self._last_used = 0.0
        self._models = {}                      # dfm -> size（worker 已经加载过的）
        self._max_models = 3                   # worker 自报的常驻上限（同步自它的 ready）
        self._sess_providers = {}               # dfm -> worker 建 session 时真正拿到的 provider
        self._script = None
        self._providers = []

    # ---- 状态 ----
    def alive(self) -> bool:
        p = self._proc
        try:
            return p is not None and p.poll() is None
        except Exception:
            return False

    def status(self) -> dict:
        return {"running": self.alive(),
                "pid": getattr(self._proc, "pid", None) if self.alive() else None,
                "spawns": self._spawns, "boot_ms": self._boot_ms,
                "idle_s": round(time.time() - self._last_used, 1) if self._last_used else None,
                "models": sorted(self._models), "providers": list(self._providers),
                "max_models": int(self._max_models or 3),
                "session_providers": {os.path.basename(k): list(v)
                                      for k, v in self._sess_providers.items()},
                "script": str(self._script or "")}

    def _tail_text(self, n=6) -> str:
        lines = [x for x in list(self._tail) if str(x).strip()][-n:]
        return " ｜ ".join(x.strip()[:200] for x in lines)[:600]

    # ---- 落盘 / 启动 ----
    def _script_path(self) -> Path:
        root = Path(tempfile.gettempdir()) / "dfl41_merge_preview"
        root.mkdir(parents=True, exist_ok=True)
        digest = hashlib.sha1(_PREVIEW_WORKER_SRC.encode("utf-8")).hexdigest()[:10]
        sp = root / ("preview_worker_%s.py" % digest)
        if not sp.is_file() or sp.stat().st_size < 1024:
            tmp = root / ("preview_worker_%s.tmp" % digest)
            tmp.write_bytes(_PREVIEW_WORKER_SRC.encode("utf-8"))
            os.replace(str(tmp), str(sp))
        return sp

    def _spawn(self):
        py = Path(str(_engine_python()))
        if not py.is_file():
            raise _BadInput("找不到 4.0 引擎的 python：%s" % py, 500)
        root = Path(str(_engine_root()))
        if not (root / "MergeStudio").is_dir():
            raise _BadInput("找不到 4.0 的 MergeStudio：%s" % root, 500)
        try:
            script = self._script_path()
        except OSError as e:
            raise _BadInput("预览 worker 落不了盘（%s）：%s" % (tempfile.gettempdir(), e), 500)
        env = dict(os.environ)
        env.update({"PYTHONIOENCODING": "utf-8", "PYTHONUNBUFFERED": "1",
                    "PYTHONUTF8": "1"})
        try:
            proc = subprocess.Popen(
                [str(py), str(script), str(root)],
                cwd=str(root), env=env, stdin=subprocess.PIPE,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                creationflags=getattr(_app_mod, "_NO_WINDOW", 0))
        except Exception as e:
            raise _BadInput("预览 worker 起不来：%s" % (str(e)[:200],), 500)
        self._proc = proc
        self._stdin = proc.stdin
        self._q = queue.Queue()
        self._script = script
        self._spawns += 1
        self._boot_ms = None
        self._providers = []
        self._models = {}
        self._sess_providers = {}
        self._tail.clear()
        self._out_t = threading.Thread(target=self._read_stdout, args=(proc, self._q),
                                       daemon=True, name="merge-preview-out")
        self._out_t.start()
        self._err_t = threading.Thread(target=self._read_stderr, args=(proc,),
                                       daemon=True, name="merge-preview-err")
        self._err_t.start()
        if self._idle_t is None or not self._idle_t.is_alive():
            self._idle_t = threading.Thread(target=self._idle_loop, daemon=True,
                                            name="merge-preview-idle")
            self._idle_t.start()

    def _read_stdout(self, proc, q):
        try:
            for raw in proc.stdout:
                line = raw.decode("utf-8", "replace").strip()
                if not line:
                    continue
                try:
                    q.put(json.loads(line))
                except Exception:
                    continue
        except Exception:
            pass
        finally:
            q.put({"eof": True})

    def _read_stderr(self, proc):
        try:
            for raw in proc.stderr:
                line = raw.decode("utf-8", "replace").rstrip()
                if line.strip():
                    self._tail.append(line[:300])
        except Exception:
            pass

    def _idle_loop(self):
        """空闲 10 分钟收掉 worker（显存/句柄别一直占着）。"""
        proc = self._proc
        while True:
            # 20s 一轮（阈值调小的时候 tick 跟着变小，测试/调试不用干等 10 分钟）
            time.sleep(max(2.0, min(20.0, float(_PREVIEW_IDLE_EXIT) / 30.0)))
            if self._proc is not proc:
                return
            try:
                if proc.poll() is not None:
                    return
            except Exception:
                return
            if self._lock.locked():           # 有请求在飞，不算空闲
                continue
            if self._last_used and (time.time() - self._last_used) < _PREVIEW_IDLE_EXIT:
                continue
            self.close("空闲退出")
            return

    def close(self, why="") -> bool:
        """收掉 worker：先请它自己退，5 秒不走就整棵进程树杀掉。"""
        p = self._proc
        if p is None:
            return False
        self._proc = None
        self._stdin = None
        self._q = None
        self._models = {}
        self._sess_providers = {}
        self._providers = []
        self._boot_ms = None
        if why:
            print("[merge-preview] worker 退出：%s（pid %s）"
                  % (why, getattr(p, "pid", "?")), flush=True)
        try:
            if p.poll() is None:
                try:
                    p.stdin.write(b'{"cmd": "exit"}\n')
                    p.stdin.flush()
                except Exception:
                    pass
                try:
                    p.wait(timeout=5)
                except Exception:
                    pass
            if p.poll() is None:
                _kill_proc_tree(p)
        except Exception:
            pass
        try:
            for f in (p.stdin, p.stdout, p.stderr):
                if f is not None and not f.closed:
                    f.close()
        except Exception:
            pass
        return True

    # ---- 一帧一请求 ----
    def _await_ready(self):
        q = self._q
        deadline = time.time() + float(_PREVIEW_BOOT_TIMEOUT)
        while True:
            left = deadline - time.time()
            if left <= 0:
                tail = self._tail_text()
                self.close("启动超时")
                raise _BadInput("预览 worker 启动超时（%.0fs）%s"
                                % (_PREVIEW_BOOT_TIMEOUT, ("：" + tail) if tail else ""), 504)
            try:
                msg = q.get(timeout=min(left, 1.0))
            except queue.Empty:
                continue
            if msg.get("eof"):
                tail = self._tail_text()
                self.close("启动失败")
                raise _BadInput("预览 worker 起不来%s" % (("：" + tail) if tail else ""), 500)
            if msg.get("ready"):
                self._boot_ms = msg.get("boot_ms")
                self._providers = list(msg.get("providers") or [])
                if msg.get("max_models"):
                    try:
                        self._max_models = int(msg.get("max_models"))
                    except Exception:
                        pass
                self._last_used = time.time()
                return msg

    def ping(self) -> dict:
        return self.call({"cmd": "ping"}, work_timeout=10.0, load_timeout=10.0)

    def unload(self, dfm="") -> dict:
        """请 worker 放掉显存里的 DFM：dfm 空 = 全放，给了就只留它。

        和 warm / preview 共用同一把 _lock，所以不会和正在飞的那一帧抢显存。
        """
        return self.call({"cmd": "unload", "dfm": str(dfm or "")},
                         work_timeout=10.0, load_timeout=10.0)

    def call(self, payload: dict, work_timeout=None, load_timeout=None,
             on_stage=None) -> dict:
        """发一帧给 worker 并等结果。超时/失联会杀掉 worker 再抛 _BadInput。"""
        work_timeout = float(_PREVIEW_WORK_TIMEOUT if work_timeout is None else work_timeout)
        load_timeout = float(_PREVIEW_LOAD_TIMEOUT if load_timeout is None else load_timeout)
        with self._lock:
            if not self.alive():
                self._spawn()
                self._await_ready()
            self._seq += 1
            rid = self._seq
            msg_in = dict(payload)
            msg_in["id"] = rid
            line = (json.dumps(msg_in) + "\n").encode("utf-8")   # ASCII 转义，安全过管道
            try:
                self._stdin.write(line)
                self._stdin.flush()
            except Exception as e:
                tail = self._tail_text()
                self.close("写不进子进程")
                raise _BadInput("预览 worker 失联（%s）%s —— 已重启，请再试一次"
                                % (str(e)[:120], ("：" + tail) if tail else ""), 503)

            now = time.time()
            self._last_used = now
            soft = now + work_timeout
            ceiling = now + work_timeout + load_timeout
            loading = False
            q = self._q
            while True:
                now = time.time()
                left = min(soft, ceiling) - now
                if left <= 0:
                    tail = self._tail_text()
                    self.close("超时")
                    if loading:
                        raise _BadInput("预览加载超时（>%.0fs 还没把模型准备好）%s"
                                        % (load_timeout, ("：" + tail) if tail else ""), 504)
                    raise _BadInput("预览超时（单帧上限 %.0fs）—— 已终止这次任务%s"
                                    % (work_timeout, ("：" + tail) if tail else ""), 504)
                try:
                    msg = q.get(timeout=min(left, 0.5))
                except queue.Empty:
                    continue
                if msg.get("eof"):
                    tail = self._tail_text()
                    self.close("子进程退出")
                    raise _BadInput("预览 worker 退出了%s" % (("：" + tail) if tail else ""), 502)
                if msg.get("ready"):
                    continue
                mid = msg.get("id")
                if mid is not None and mid != rid:
                    continue                       # 上一轮超时留下的残响，丢掉
                stage = str(msg.get("stage") or "")
                if stage:
                    if on_stage is not None:
                        try:
                            on_stage(stage)
                        except Exception:
                            pass
                    if stage == "loading_model":
                        loading = True
                        soft = min(ceiling, time.time() + load_timeout)
                    elif stage == "model_ready":
                        loading = False
                        # worker 报回来什么就是什么，别在这边攒着旧名字
                        # （攒着的话 worker_models 会虚报、_warm_hot 也会误判成热）。
                        # 多模型时它的 models 是整份常驻清单（最多 3 个），照抄就对。
                        dfm_hot = str(msg.get("dfm") or payload.get("dfm") or "")
                        hot = {str(p): 0 for p in (msg.get("models") or [])}
                        if not hot and dfm_hot:
                            hot = {dfm_hot: 0}
                        if dfm_hot and msg.get("size"):
                            hot[dfm_hot] = msg.get("size")
                        if msg.get("max_models"):
                            try:
                                self._max_models = int(msg.get("max_models"))
                            except Exception:
                                pass
                        if dfm_hot and isinstance(msg.get("providers"), list):
                            # worker 建的 session 到底跑在哪：CUDA 起不来时它会静默给
                            # ['CPUExecutionProvider']，这里留档，排障不用靠猜
                            self._sess_providers[dfm_hot] = list(msg.get("providers"))
                        for p in [x for x in self._models if x not in hot]:
                            print("[merge-preview] unload %s" % os.path.basename(p),
                                  flush=True)
                        self._models = hot
                        soft = time.time() + work_timeout
                    continue
                self._last_used = time.time()
                if isinstance(msg.get("models"), list):
                    # warm / unload 的回应会带一份现场清单：以它为准对齐缓存
                    self._models = {str(p): self._models.get(str(p)) or 0
                                    for p in msg.get("models")}
                return msg


_WORKER = _PreviewWorker()


def _shutdown_preview_worker():
    try:
        _WORKER.close("进程退出")
    except Exception:
        pass


atexit.register(_shutdown_preview_worker)


# ---- 预览：取帧 / 关键点 / 模型 ----
_DUR_CACHE = {}
_DUR_LOCK = threading.Lock()


def _video_duration(p) -> float:
    """视频时长（秒）。按 大小+mtime 缓存：拖动时每帧都要问，不能每帧都去探。"""
    pp = Path(str(p))
    try:
        st = pp.stat()
    except OSError:
        return 0.0
    key = os.path.normcase(str(pp))
    sig = (int(st.st_size), int(st.st_mtime))
    with _DUR_LOCK:
        hit = _DUR_CACHE.get(key)
        if hit and hit[0] == sig:
            return hit[1]
    dur = 0.0
    try:
        import cv2
        cap = cv2.VideoCapture(str(pp))
        try:
            fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
            n = float(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0)
            if fps > 0 and n > 0:
                dur = n / fps
        finally:
            cap.release()
    except Exception:
        dur = 0.0
    if not dur:
        try:
            import imageio_ffmpeg
            _n, secs = imageio_ffmpeg.count_frames_and_secs(str(pp))
            dur = float(secs or 0.0)
        except Exception:
            dur = 0.0
    with _DUR_LOCK:
        _DUR_CACHE[key] = (sig, dur)
    return dur


# ---- 取帧：同帧复用 / 顺序解码缓存 / ffmpeg 快速 seek / cv2 回退 ----
#
# 先说清楚为什么不是「每次 ffmpeg -ss 一下」，下面是这台机器上量的（workspace 里那个
# 1080x1920 / 30fps / GOP=5s / 57MB 的 mp4）：
#   · ffmpeg 起一个进程本身就要 47ms（-version）、110ms（带 -i）——光启动就吃掉 50ms 的
#     预算，再叠 seek+解码，整次取帧实测 208~264ms（-ss 放 -i 前面已经是关键帧级定位了）；
#   · cv2 不换进程，复用常开句柄重新定位是 78~215ms，而且定位完句柄正好停在 t 上，
#     下一次「往后拖」的请求能接着顺序读（1.2ms/帧，完全不重新定位）。
# 所以冷定位默认走 cv2（_GRAB_FFMPEG_FIRST=False），ffmpeg 快速 seek 留着当回退；
# 想反过来「先 ffmpeg」把开关翻成 True 就行，两条路都实现在这里。
# 真正能进 50ms 的是前两条：同一帧直接复用（≈0ms）、前向 1.5s 内顺序解码（10~60ms）。
_GRAB_LOCK = threading.RLock()          # 常开句柄 + 缓存帧是共享可变状态（预览是串行的）
_GRAB_FWD_WINDOW = 1.5                  # 顺序解码窗口（秒）：超过就重新定位
_GRAB_FFMPEG_FIRST = False              # 冷定位先试 ffmpeg 快速 seek？（实测更慢，见上）
_GRAB_FFMPEG_TIMEOUT = 8.0              # 单次 ffmpeg 取帧上限（秒）
_GRAB_STATE = {"key": None, "cap": None, "fps": 0.0, "idx": -1,
               "frame": None, "pos": False}


def _grab_key(video):
    """缓存身份：路径 + 大小 + mtime。片子换了/改了，句柄和缓存帧一起作废。"""
    p = os.path.normcase(str(video))
    try:
        stt = os.stat(str(video))
        return (p, int(stt.st_size), int(stt.st_mtime))
    except OSError:
        return (p, 0, 0)


def _grab_close_locked():
    """丢掉常开句柄和缓存帧（换片时用；进程退出时句柄跟着回收）。"""
    cap = _GRAB_STATE.get("cap")
    _GRAB_STATE.update({"key": None, "cap": None, "fps": 0.0, "idx": -1,
                        "frame": None, "pos": False})
    if cap is not None:
        try:
            cap.release()
        except Exception:
            pass


def _grab_index(t, fps):
    """t → 帧号。口径跟 cv2 的精确 seek 对齐：实测 set(POS_MSEC) 落的是 round(t*fps) 那一帧
    （7.21→216、7.23→217、7.25→218、20.598→618），四舍五入用 floor(x+0.5)，因为 t*fps 在
    浮点里常常贴着整数（7.2*30 = 215.99999999999997），直接 floor/ceil 都会差一帧。
    """
    import math
    x = max(0.0, float(t)) * float(fps)
    return int(math.floor(x + 0.5))


def _grab_ensure_locked(video):
    """按 key 准备常开句柄（换了片就重建）。cv2 打不开也不炸，回 (None, 30.0)。"""
    import cv2
    key = _grab_key(video)
    st = _GRAB_STATE
    cap = st.get("cap")
    if st.get("key") != key or cap is None or not cap.isOpened():
        _grab_close_locked()
        try:
            cap = cv2.VideoCapture(str(video))
        except Exception:
            cap = None
        if cap is None or not cap.isOpened():
            if cap is not None:
                try:
                    cap.release()
                except Exception:
                    pass
            return None, 30.0
        try:
            fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        except Exception:
            fps = 0.0
        if not (1.0 <= fps <= 480.0):
            fps = 30.0
        st.update({"key": key, "cap": cap, "fps": fps, "idx": -1,
                   "frame": None, "pos": False})
    return st["cap"], float(st["fps"] or 30.0)


def _grab_via_ffmpeg(video, t):
    """ffmpeg 快速 seek 取一帧：-ss 放在 -i 前面是关键（关键帧级定位 + accurate_seek 到精确帧）。

    成功回 (frame, "")，失败回 (None, 原因) —— 原因要带进 _BadInput 里给前端看。
    """
    import cv2
    import numpy as np
    try:
        import imageio_ffmpeg
        exe = imageio_ffmpeg.get_ffmpeg_exe()
    except Exception as e:
        return None, "没有 ffmpeg：%s" % (str(e)[:100])
    cmd = [exe, "-v", "error", "-nostdin", "-ss", "%.3f" % float(t), "-i", str(video),
           "-frames:v", "1", "-f", "image2pipe", "-vcodec", "png", "-"]
    try:
        out = subprocess.run(cmd, capture_output=True, timeout=_GRAB_FFMPEG_TIMEOUT,
                             creationflags=getattr(_app_mod, "_NO_WINDOW", 0))
    except Exception as e:
        return None, "ffmpeg 跑不起来：%s" % (str(e)[:120])
    if out.returncode != 0 or not out.stdout:
        why = (out.stderr or b"").decode("utf-8", "ignore").strip()
        return None, "ffmpeg：%s" % (why[:160] or "没有输出")
    try:
        frame = cv2.imdecode(np.frombuffer(out.stdout, np.uint8), cv2.IMREAD_COLOR)
    except Exception as e:
        return None, "ffmpeg 给的图解不开：%s" % (str(e)[:100])
    if frame is None or not int(getattr(frame, "size", 0)):
        return None, "ffmpeg 给的图解不开"
    return frame, ""


def _grab_frame_at(video, t: float, stop=None):
    """取 t 秒那一帧（BGR）。

    三条路，从便宜到贵：
      1) 同一帧：t 跟上一帧落在同一帧号上（前端 120ms 防抖会把同一个 t 请求两遍）→ 直接给缓存帧；
      2) 顺序解码：t 在上次之后 _GRAB_FWD_WINDOW 秒内 → 常开句柄 grab() 往前读，不再定位；
      3) 重新定位：ffmpeg 快速 seek（_GRAB_FFMPEG_FIRST 为真时先试）→ cv2 的
         set(CAP_PROP_POS_MSEC)（原实现的路子，只是复用常开句柄，省掉每次开文件的 6~9ms）。
    错误语义不变：取不出来 500、取消 499；成功时顺手把「上一帧」记进 _GRAB_STATE。
    """
    import cv2

    t = max(0.0, float(t))
    err = ""
    with _GRAB_LOCK:
        cap, fps = _grab_ensure_locked(video)
        st = _GRAB_STATE
        target = _grab_index(t, fps)

        # 1) 同一帧复用（同一个 t，或者两个 t 落在同一帧号上）
        if st["frame"] is not None and int(st["idx"]) == target:
            return st["frame"]

        # 2) 顺序解码：句柄就位 + 目标在前向窗口内
        need = target - int(st["idx"]) if int(st["idx"]) >= 0 else -1
        window = max(1, int(round(_GRAB_FWD_WINDOW * fps)))
        if st["pos"] and cap is not None and 0 < need <= window:
            if stop is not None and stop.is_set():
                raise _BadInput("预览已取消", 499)
            ok = True
            for _ in range(need):
                if not cap.grab():
                    ok = False
                    break
                st["idx"] = int(st["idx"]) + 1
            if ok:
                try:
                    ok, frame = cap.retrieve()
                except Exception:
                    ok, frame = False, None
                if ok and frame is not None and int(getattr(frame, "size", 0)):
                    st["frame"] = frame
                    st["pos"] = True
                    return frame
            err = "顺序解码没读到这一帧"
            # 句柄位置和缓存帧的帧号一起作废（否则下次同一个 t 会拿到错一帧的缓存）
            st["idx"] = -1
            st["pos"] = False
        if stop is not None and stop.is_set():
            raise _BadInput("预览已取消", 499)

        # 3) 重新定位：ffmpeg 快速 seek 与 cv2 seek 都试，谁先看开关
        if _GRAB_FFMPEG_FIRST:
            frame, why = _grab_via_ffmpeg(video, t)
            if frame is not None:
                # 句柄没跟着走：位置标脏，下一次前向请求老老实实重新定位（不会串帧）
                st.update({"frame": frame, "idx": target, "pos": False})
                return frame
            err = (err + "；" if err else "") + (why or "ffmpeg 取帧失败")

        if cap is None:
            err = (err + "；" if err else "") + "cv2 打不开这个文件"
        else:
            try:
                cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
                ok, frame = cap.read()
            except Exception as e:
                ok, frame = False, None
                err = (err + "；" if err else "") + "cv2 取帧失败：%s" % (str(e)[:120])
            if ok and frame is not None and int(getattr(frame, "size", 0)):
                try:                            # 帧号以句柄实际位置为准（read 之后指向下一帧）
                    posf = int(round(float(cap.get(cv2.CAP_PROP_POS_FRAMES))))
                except Exception:
                    posf = -1
                st.update({"frame": frame,
                           "idx": (posf - 1) if posf > 0 else target,
                           "pos": True})
                return frame
            err = (err + "；" if err else "") + "cv2 没读到这一帧"

        if not _GRAB_FFMPEG_FIRST:
            frame, why = _grab_via_ffmpeg(video, t)
            if frame is not None:
                st.update({"frame": frame, "idx": target, "pos": False})
                return frame
            err = (err + "；" if err else "") + (why or "ffmpeg 取帧失败")

    raise _BadInput("这一帧取不出来（t=%.3fs）：%s" % (t, err[:300]), 500)


def _pick_face(boxes):
    """多脸时挑「主角」：先比框面积，再比置信度。"""
    best, best_key = None, None
    for b in boxes or []:
        try:
            x1, y1, x2, y2 = [float(v) for v in (b.get("box") or (0, 0, 0, 0))]
            key = (max(0.0, x2 - x1) * max(0.0, y2 - y1), float(b.get("score") or 0.0))
        except Exception:
            continue
        if best_key is None or key > best_key:
            best, best_key = b, key
    return best


_LM106 = {"m": None, "tried": False}


def _landmarks_model():
    """2D106 关键点模型（68 点由它回归出来），本进程只加载一次。

    **不传给 app._get_landmarks106()**：那个会把 workspace/engine_name 一起传下去，
    _engines.try_engine 会命中预编译的 TensorRT 引擎，_trt_session.run 走
    torch.cuda.CUDAGraph.replay —— CUDA 图是在「捕获它的那个线程」的流上建的，
    预览是每个请求一个新线程，跨线程 replay 直接把整个进程打崩（实测 0xC0000005
    access violation，第一次请求没事、第二次就死）。这里用纯 onnxruntime + CPU：
    CPU 实测 4~5ms/张，够快，也不会碰 CUDA 图。
    """
    if _LM106["tried"]:
        return _LM106["m"]
    _LM106["tried"] = True
    try:
        from ._landmarks106 import InsightFace106
        from ._paths import weights_dir
        # 平铺布局：<weights>/insightface_2d106.onnx；老的分目录布局作兜底
        for base in (weights_dir(), Path(__file__).resolve().parents[2] / "weights"):
            for rel in ("insightface_2d106.onnx",
                        "onnx/InsightFace2d106/InsightFace2D106.onnx"):
                cand = Path(base) / rel
                if cand.is_file():
                    _LM106["m"] = InsightFace106(str(cand),
                                                 providers=["CPUExecutionProvider"])
                    return _LM106["m"]
    except Exception:
        traceback.print_exc()
    _LM106["m"] = None
    return None


def _frame_landmarks(frame, det_box, box_xyxy):
    """检测框/5 点 → 原图坐标的 68 点。**只认 2D106 真的回归出来的点。**

    跟 _align_stream（抽帧+对齐那条路）同一个套路：5 点先粗对齐一次（只为了给
    模型一个像样的输入），2D106 回归 68 点，再用 image_to_face_mat 反变换回原图。

    模型缺失 / 跑挂时返回 None，由调用方报「68 点算不出来」。以前这里有个
    "2D106 不可用就退回 5 点插值"的分支（合成链路默认走它），已经删掉：那等于
    把插值估算当成检测结果，误差大 4 倍却完全看不出来（见 _align.py 文件头）。
    """
    import numpy as np

    kps = det_box.get("kps") or []
    lm68 = None
    if len(kps) >= 5:
        try:
            from ._align import align_face, transform_points
            aligned, meta = align_face(frame, np.array(kps, dtype=np.float64), box_xyxy)
            if aligned is not None and meta:
                lm = _landmarks_model()
                if lm is not None:
                    lm68a = lm.landmarks68(aligned)
                    if lm68a is not None:
                        lm68 = transform_points(
                            np.asarray(lm68a, dtype=np.float64),
                            np.array(meta["image_to_face_mat"], dtype=np.float64),
                            invert=True)
        except Exception:
            lm68 = None
    if lm68 is None:
        return None
    arr = np.asarray(lm68, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] < 2 or not np.all(np.isfinite(arr)):
        return None
    return arr


def _dfm_of(body: dict):
    """body 的 dfm / model → (dfm 路径, 模型名)。规矩和 /export 完全一致。"""
    model_name = str(body.get("model") or "").strip()
    if model_name and (("/" in model_name) or ("\\" in model_name)
                       or model_name in (".", "..") or ".." in model_name):
        raise _BadInput("「model」只能是模型名，不能带路径：%r" % model_name)
    dfm_raw = body.get("dfm")
    if str(dfm_raw or "").strip():
        dfm = _norm_path(dfm_raw)
        if dfm.suffix.lower() != ".dfm":
            raise _BadInput("dfm 要是一个 .dfm 文件：%s" % dfm)
        if not dfm.is_file():
            raise _BadInput("找不到 DFM：%s" % dfm, 404)
        return dfm, (model_name or dfm.stem)
    if model_name:
        name = model_name if model_name.lower().endswith(".dfm") else model_name + ".dfm"
        cand = Path(WORKSPACE) / "model" / name
        if not cand.is_file():
            raise _BadInput("模型 %s 还没有导出 .dfm（%s）—— 去训练页点「导出」"
                            % (model_name, cand), 404)
        return cand, model_name
    cands = sorted((Path(WORKSPACE) / "model").glob("*.dfm"))
    if not cands:
        raise _BadInput("没有可用的 .dfm 模型：先在训练页把模型导出成 .dfm（%s）"
                        % (Path(WORKSPACE) / "model"), 409)
    return cands[0], cands[0].stem


def _faces_val(pos, raw) -> list:
    """models[i].faces → 去重后的人脸序号列表。

    口径：人脸序号 = 检测顺序，从 0 起；[] / 缺省 = 这个模型负责全部脸（单模型就是
    这种）。允许 "0,1" 这种逗号串（前端有把数组 encode 成字符串的老毛病）。
    """
    if raw is None or (isinstance(raw, str) and not raw.strip()):
        return []
    if isinstance(raw, str):
        raw = [x for x in re.split(r"[,，\s]+", raw.strip()) if x]
    if not isinstance(raw, (list, tuple)):
        raise _BadInput("models[%d].faces 要是一个数组（人脸序号），收到 %r" % (pos, raw))
    out = []
    for f in raw:
        v = _int_val("models[%d].faces" % pos, f, 0, 63)
        if v not in out:
            out.append(int(v))
    return out


def _face_ids_val(name: str, raw) -> list:
    """脸索引列表 → 去重保序的 int 列表（no_swap_faces 这类"点名的脸"）。

    认 [1,3] / "1,3" / " 1 3 " / "[1,3]"（前端偶尔把数组 encode 成字符串）。
    序号就是检测顺序，从 0 起，和 /api/merge/detect 的 boxes 下标一致。
    """
    if raw is None or (isinstance(raw, str) and not raw.strip()):
        return []
    if isinstance(raw, str):
        text = raw.strip()
        if text[:1] in ("[", "("):
            try:
                raw = json.loads(text)
            except Exception:
                raise _BadInput("「%s」不是合法 JSON 数组：%s" % (name, text[:60]))
        else:
            # 分隔符是「逗号 / 全角逗号 / 空白」：原来写的是 r"[,，s]+"，
            # 那个 s 是**字母 s**，不是 \s —— "1 3" 这种空格分隔的写法会被当成
            # 一整串塞进 _int_val，直接 400。（docstring 一直说认 " 1 3 "。）
            raw = [x for x in re.split(r"[,，\s]+", text) if x]
    if not isinstance(raw, (list, tuple)):
        raise _BadInput("「%s」要是一个数组（人脸序号），收到 %s"
                        % (name, type(raw).__name__))
    out = []
    for v in raw:
        i = _int_val(name, v, 0, 63)
        if i not in out:
            out.append(int(i))
    return out


def _models_of(body: dict):
    """body 的 models 数组 → [{name, dfm, faces, index}, …]；没给 / 空数组 → None。

    None 表示「老的单模型路径」，调用方照旧走 _dfm_of()，行为一个字都不改。
    每一项的 dfm/name 解析规则和单模型完全一致（显式 dfm > 模型名 > 报错），
    只是这里不再"悄悄挑 workspace 里第一个 .dfm"——多模型下挑错了没人看得出来。
    """
    raw = body.get("models")
    if raw is None:
        return None
    if isinstance(raw, str):
        if not raw.strip():
            return None
        try:
            raw = json.loads(raw)
        except Exception:
            raise _BadInput("「models」不是合法 JSON：%s" % str(raw)[:80])
    if raw == [] or raw == {}:
        return None
    if not isinstance(raw, list):
        raise _BadInput("「models」要是一个数组（[{name, dfm, faces}]），收到 %s"
                        % type(raw).__name__)
    if len(raw) > _PREVIEW_MAX_MODELS:
        raise _BadInput("「models」一次最多 %d 个模型，收到 %d 个"
                        % (_PREVIEW_MAX_MODELS, len(raw)))
    out = []
    for pos, item in enumerate(raw):
        if not isinstance(item, dict):
            raise _BadInput("models[%d] 要是一个 JSON 对象，收到 %s"
                            % (pos, type(item).__name__))
        name_raw = str(item.get("name") or item.get("model") or "").strip()
        dfm_raw = item.get("dfm")
        if not name_raw and not str(dfm_raw or "").strip():
            raise _BadInput("models[%d] 要给 name（模型名）或 dfm（.dfm 路径）" % pos)
        dfm, name = _dfm_of({"model": name_raw, "dfm": dfm_raw})
        out.append({"name": str(name), "dfm": str(dfm),
                    "faces": _faces_val(pos, item.get("faces")), "index": int(pos)})
    return out


def _model_for_face(idx: int, models) -> int:
    """第 idx 张脸归哪个模型。

    三级：
      1) 有模型显式点名这张脸（faces 里含 idx）→ 按 models 顺序先声明先得；
      2) 没有点名的，交给 faces=[] 的那个（空 faces 就是「我管所有脸」，单模型就
         是这种写法，所以单模型下整帧都归它）；
      3) 连空 faces 的都没有 → 第 0 个。
    """
    for i, m in enumerate(models or []):
        if int(idx) in (m.get("faces") or []):
            return int(i)
    for i, m in enumerate(models or []):
        if not (m.get("faces") or []):
            return int(i)
    return 0


def _used_names(models, loaded, applied=None) -> list:
    """models 里哪些真的参与了这次请求（按 models 顺序、去重）。

    loaded = worker 回的那份常驻清单（路径）；applied = 这一帧真贴上脸的那几个。
    两个取并集：声明了但这一帧没脸可贴的模型也算「装了、参与了预览」（前端拿
    models_used 核对配置有没有被吃）；声明超过 worker 上限（3 个）时被 LRU 挤掉、
    又没真正贴过脸的，才算没参与。
    """
    have = {os.path.normcase(str(p)) for p in (loaded or []) if str(p or "").strip()}
    have |= {os.path.normcase(str(p)) for p in (applied or []) if str(p or "").strip()}
    out = []
    for m in models or []:
        if os.path.normcase(str(m["dfm"])) not in have:
            continue
        if m["name"] not in out:
            out.append(m["name"])
    return out


def _xseg_of(body: dict) -> str:
    """遮罩模型：body 显式给 > workspace 里的 XSegLite > XSeg。都没有就空串。"""
    xr = body.get("xseg_onnx") or body.get("mask_onnx") or body.get("xseg")
    if str(xr or "").strip():
        xp = _norm_path(xr)
        if not xp.is_file():
            raise _BadInput("找不到 XSeg 模型：%s" % xp, 404)
        return str(xp)
    for c in (Path(WORKSPACE) / "model" / "XSegLite" / "xseglite.onnx",
              Path(WORKSPACE) / "model" / "XSeg" / "XSeg.onnx"):
        if c.is_file():
            return str(c)
    return ""


def _jpeg_bytes(img, quality=_PREVIEW_QUALITY) -> bytes:
    import cv2
    ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        raise _BadInput("这一帧编不出 JPEG", 500)
    return buf.tobytes()


def _img_bytes(img, fmt="jpg", quality=_PREVIEW_QUALITY) -> bytes:
    """整帧编码。fmt=png 走无损（项目模式导出落盘用，避免每帧掉一次画质）；
    其余一律还是 jpg，和预览完全一致。"""
    if str(fmt or "jpg").strip().lower() != "png":
        return _jpeg_bytes(img, quality)
    import cv2
    ok, buf = cv2.imencode(".png", img, [int(cv2.IMWRITE_PNG_COMPRESSION), 3])
    if not ok:
        raise _BadInput("这一帧编不出 PNG", 500)
    return buf.tobytes()


def _send_frame_b64(frame, job: dict) -> str:
    """整帧发给 worker 的 base64。

    job["send_lossless"]（= 请求 fmt=png）时走 PNG：**只有无损传输，没被换的脸
    才可能和原帧逐像素相同**（no_swap_faces 的验收条件就是"脸区域差异为 0"）。
    jpg（界面预览那条路）仍旧是 q95 JPEG，速度优先，眼睛看不出来。
    """
    if job.get("send_lossless"):
        import cv2
        ok, buf = cv2.imencode(".png", frame, [int(cv2.IMWRITE_PNG_COMPRESSION), 1])
        if ok:
            return _b64.b64encode(buf.tobytes()).decode("ascii")
    return _b64.b64encode(_jpeg_bytes(frame, _PREVIEW_SEND_QUALITY)).decode("ascii")


def _plain_frame(job: dict, frame, box_wh, timings, note="", has_face=True,
                 no_faces=False) -> dict:
    """这一帧没换脸时的返回。

    默认还是「整帧编一张图回给预览」（质量 88，和以前完全一样）。项目模式导出
    会带 job["no_face_copy"]：没脸 = 原帧原样进 merged，重编码一遍只会白掉一档
    画质 —— 所以这里不编码，交一张 copy=True 的凭据，让调用方直接搬源文件。

    no_faces=True 只给「整帧一张脸都没检出」那一条用（和「检出了但没换」区分开：
    后者 has_face=True + no_faces=False + swapped=False）。
    """
    if job.get("no_face_copy"):
        return {"copy": True, "jpg": None, "img": None, "box": box_wh,
                "has_face": bool(has_face), "no_faces": bool(no_faces),
                "swapped": False, "timings": timings,
                "note": note, "faces": [], "models_used": [], "swaps": 0}
    fmt = str(job.get("fmt") or "jpg").strip().lower()
    img = _img_bytes(frame, fmt, 88)
    return {"jpg": img, "img": img, "box": box_wh, "has_face": bool(has_face),
            "no_faces": bool(no_faces), "swapped": False, "timings": timings,
            "note": note, "faces": [], "models_used": [], "swaps": 0}


def _preview_multi(job: dict, frame, boxes, box_wh, timings, stage_fn, state) -> dict:
    """多模型换脸：检测到的每张脸都按 models 里的分配换，没分配的用第 0 个。

    一次 worker 请求把「这次要用的整组模型」先装进显存（worker 上限 3 个，超了
    LRU 淘汰），然后在同一张帧上按检测顺序一张一张贴 —— 每张贴的都是上一次的
    结果，所以 3 张脸就是 3 次 MergeMaskedFace，最后只编一次 JPEG。

    人脸序号就是检测顺序（det.detect() 的返回顺序，从 0 起）；和 /api/merge/detect
    返回的 boxes 下标完全一致，前端拿它做脸-模型的勾选就对了。
    """
    import numpy as np

    models = job["models"]
    fmt = str(job.get("fmt") or "jpg").strip().lower()       # "jpg"（预览）/ "png"（落盘）
    q = int(job.get("quality") or _PREVIEW_QUALITY)
    stage_fn("faces")
    t_lm = time.time()
    plan, skipped = [], []
    # 「不换脸」的脸：直接不参与合成 —— 既不算进 plan，也不兜给第 0 个模型。
    # 这条必须在 _model_for_face 之前判掉，否则"未声明"会被兜底成第 0 个模型。
    no_swap = {int(x) for x in (job.get("no_swap") or ())}
    for i, b in enumerate((boxes or [])[:_PREVIEW_MAX_SWAPS]):
        if int(i) in no_swap:
            skipped.append({"face": int(i), "why": "no_swap_faces：按要求保持原像素"})
            continue
        try:
            x1, y1, x2, y2 = [float(v) for v in (b.get("box") or ())]
        except Exception:
            skipped.append({"face": int(i), "why": "检测框不合法"})
            continue
        mi = _model_for_face(i, models)
        m = models[mi]
        lm = _frame_landmarks(frame, b, [x1, y1, x2, y2])
        if lm is None:
            skipped.append({"face": int(i), "why": "68 点算不出来", "model": m["name"]})
            continue
        plan.append({"face": int(i), "model": int(mi), "name": m["name"],
                     "dfm": str(m["dfm"]),
                     "box": [int(round(x1)), int(round(y1)),
                             int(round(max(0.0, x2 - x1))),
                             int(round(max(0.0, y2 - y1)))],
                     "landmarks": np.asarray(lm, dtype=np.float64).round(2).tolist()})
    timings["landmarks_ms"] = round((time.time() - t_lm) * 1000.0, 1)
    timings["faces"] = len(plan)

    if not plan:
        if no_swap and all(s.get("why", "").startswith("no_swap_faces") for s in skipped):
            return _plain_frame(job, frame, box_wh, timings,
                                note="这一帧的脸都在 no_swap_faces 里 —— 整帧保持原像素")
        return _plain_frame(job, frame, box_wh, timings, note="检出人脸但 68 点算不出来 —— 只回原帧")

    # 声明要用的整组（去重、按 models 顺序）。worker 只常驻 3 个，多出来的它自己
    # LRU 换 —— 这里截到上限就行，免得让它在一次请求里反复横跳。
    load, seen = [], set()
    for m in models:
        key = os.path.normcase(str(m["dfm"]))
        if key in seen:
            continue
        seen.add(key)
        load.append(str(m["dfm"]))
        if len(load) >= _WORKER_MAX_MODELS:
            break

    stage_fn("swap")
    state["stage"] = "queued"          # worker 一次只跑一帧：排队时间不算进 15s
    t0 = time.time()
    payload = {
        "frame": _send_frame_b64(frame, job),
        "faces": [{"dfm": p["dfm"], "landmarks": p["landmarks"], "box": p["box"]}
                  for p in plan],
        "load": load,
        "xseg": str(job.get("xseg") or ""),
        "config": dict(job["config"]),
        "jpeg_quality": q,
        "fmt": fmt,
    }
    resp = _WORKER.call(payload, on_stage=lambda st: state.__setitem__("stage", st))
    if not resp.get("ok"):
        raise _BadInput("换脸失败：%s" % (resp.get("error") or "预览 worker 没给原因"), 500)
    try:
        jpg = _b64.b64decode(resp.get("jpeg") or "")
    except Exception:
        raise _BadInput("换脸结果是坏的（base64 解不开）", 500)
    if not jpg:
        raise _BadInput("换脸结果为空", 500)
    timings["swap_ms"] = resp.get("swap_ms")
    timings["swap_wall_ms"] = round((time.time() - t0) * 1000.0, 1)

    note = ""
    if skipped:
        note = "%d 张脸没换（%s）" % (len(skipped), "；".join(
            "#%d %s" % (s["face"], s["why"]) for s in skipped))
    if len(boxes or []) > _PREVIEW_MAX_SWAPS:
        note = (note + "；" if note else "") + ("只换前 %d 张脸（这一帧检出 %d 张）"
                                                % (_PREVIEW_MAX_SWAPS, len(boxes)))
    return {"jpg": jpg, "img": jpg, "box": box_wh, "has_face": True, "no_faces": False,
            "swapped": True,
            "timings": timings, "model_size": resp.get("size"), "note": note,
            "faces": [{"face": p["face"], "model": p["name"], "dfm": p["dfm"],
                       "box": p["box"]} for p in plan],
            "models_used": _used_names(models, resp.get("models"),
                                       [p["dfm"] for p in plan]),
            "swaps": len(plan), "skipped": skipped}


def _preview_job(job: dict) -> dict:
    """一次预览的重活：取帧 → 检测 → 68 点 → 换脸 + 遮罩 + 合成。

    job: {video, t, dfm, xseg, config, detector, thresh, stop, state[, models]}
    models 为空/缺省 = 老的单模型路径：只换「主角」那一张脸，行为一个字都不改；
    给了 models = 多模型（见 _preview_multi）。
    每个阶段之间都查 stop —— 超时/取消时调用方置位，这里尽快收手。
    """
    import cv2
    import numpy as np

    stop = job["stop"]
    state = job["state"]
    timings = {}
    fmt = str(job.get("fmt") or "jpg").strip().lower()       # "jpg"（预览）/ "png"（落盘）
    q = int(job.get("quality") or _PREVIEW_QUALITY)

    def _stage(name):
        state["stage"] = name
        if stop.is_set():
            raise _BadInput("预览已取消", 499)

    _stage("grab")
    t0 = time.time()
    src = job.get("frame")
    if src is not None:
        # 项目模式：帧已经是内存里的 BGR（调用方从 <项目>/000000.jpg 读的），
        # 不用再走视频定位 —— 检测/换脸/合成这些活一模一样。
        frame = src
        timings["grab_ms"] = 0.0
    else:
        frame = _grab_frame_at(job["video"], job["t"], stop)
        timings["grab_ms"] = round((time.time() - t0) * 1000.0, 1)

    _stage("detect")
    t0 = time.time()
    boxes = job.get("boxes")
    if boxes is not None:
        # 调用方（自动分配那条路）已经为了算 embedding 检测过一遍了，直接用
        boxes = list(boxes)
        timings["detect_ms"] = 0.0
        timings["detect_reused"] = 1
    else:
        boxes = _detect_boxes(frame, job["detector"], job["thresh"])
        timings["detect_ms"] = round((time.time() - t0) * 1000.0, 1)

    best = _pick_face(boxes)
    box_xyxy, box_wh = None, None
    if best is not None:
        try:
            x1, y1, x2, y2 = [float(v) for v in best["box"]]
            box_xyxy = [x1, y1, x2, y2]
            box_wh = [int(round(x1)), int(round(y1)),
                      int(round(max(0.0, x2 - x1))), int(round(max(0.0, y2 - y1)))]
        except Exception:
            box_xyxy, box_wh = None, None

    if box_wh is None:
        # 一张脸都没检出：原帧原样回去（导出那条路会 copy=True 直接搬源文件），
        # 后面「自动分配 / 68 点 / 换脸」一步都不该再跑。
        return _plain_frame(job, frame, None, timings, note="这一帧没检出人脸",
                            has_face=False, no_faces=True)

    if job.get("models"):                      # 多模型：每张脸各用各的模型
        return _preview_multi(job, frame, boxes, box_wh, timings, _stage, state)

    # 单模型路径：主角那张脸被点名"不换"就整帧原样回去（不能假装换过）
    no_swap = {int(x) for x in (job.get("no_swap") or ())}
    if no_swap:
        best_i = next((i for i, b in enumerate(boxes) if b is best), None)
        if best_i is not None and best_i in no_swap:
            return _plain_frame(job, frame, box_wh, timings,
                                note="第 %d 张脸在 no_swap_faces 里 —— 整帧保持原像素"
                                     % best_i)

    _stage("landmarks")
    t0 = time.time()
    lm68 = _frame_landmarks(frame, best, box_xyxy)
    timings["landmarks_ms"] = round((time.time() - t0) * 1000.0, 1)
    if lm68 is None:
        return _plain_frame(job, frame, box_wh, timings,
                            note="检出人脸但 68 点算不出来 —— 只回原帧")

    _stage("swap")
    state["stage"] = "queued"          # worker 一次只跑一帧：排队时间不算进 15s
    t0 = time.time()
    payload = {
        "frame": _send_frame_b64(frame, job),
        "landmarks": np.asarray(lm68, dtype=np.float64).round(2).tolist(),
        "dfm": str(job["dfm"]),
        "xseg": str(job.get("xseg") or ""),
        "box": list(box_wh),
        "config": dict(job["config"]),
        "jpeg_quality": q,
        "fmt": fmt,
    }
    resp = _WORKER.call(payload, on_stage=lambda st: state.__setitem__("stage", st))
    if not resp.get("ok"):
        raise _BadInput("换脸失败：%s" % (resp.get("error") or "预览 worker 没给原因"), 500)
    try:
        jpg = _b64.b64decode(resp.get("jpeg") or "")
    except Exception:
        raise _BadInput("换脸结果是坏的（base64 解不开）", 500)
    if not jpg:
        raise _BadInput("换脸结果为空", 500)
    timings["swap_ms"] = resp.get("swap_ms")
    timings["swap_wall_ms"] = round((time.time() - t0) * 1000.0, 1)
    return {"jpg": jpg, "img": jpg, "box": box_wh, "has_face": True, "no_faces": False,
            "swapped": True,
            "timings": timings, "model_size": resp.get("size"), "note": ""}


@router.post("/api/merge/preview")
@_api
def merge_preview(body: dict):
    """t 时刻那一帧 → 检测 → 换脸 → 遮罩 → 合成 → JPEG（data: URL）。

    body 多认一个可选的 models: [{name, dfm, faces}]（见 _models_of）：给了就是
    多模型，一帧里检测到的每张脸按分配换，没分配的用第 0 个；不给就是老的单模型
    路径，行为一个字都不改。
    """
    t_start = time.time()
    body = _as_dict(body)

    video = _norm_path(body.get("video") or body.get("path"))
    if video is None:
        raise _BadInput("缺少 video：要一个视频文件路径")
    if not video.is_file():
        raise _BadInput("视频不存在：%s" % video, 404)
    if video.suffix.lower() not in _VIDEO_EXT:
        raise _BadInput("不是视频文件：%s（只认 %s）"
                        % (video.suffix or "无扩展名", " ".join(sorted(_VIDEO_EXT))))

    t = _float_val("t", body.get("t"), 0.0, None, 0.0)
    dur = _video_duration(video)
    if dur > 0:
        if t > dur + 0.5:
            raise _BadInput("t=%.2fs 超出了视频长度（%.2fs）" % (t, dur))
        t = min(t, max(0.0, dur - 0.05))

    models = _models_of(body)                 # None = 老的单模型路径
    if models:
        dfm, model_name = Path(models[0]["dfm"]), models[0]["name"]
    else:
        dfm, model_name = _dfm_of(body)
    settings = _settings_val(body.get("settings"))
    cfg = _merge_settings_to_config(settings)
    detector = _detector_val(body.get("detector"))
    thresh = _float_val("thresh", body.get("thresh"), 0.0, 1.0, 0.5)
    xseg = _xseg_of(body)

    note = ""
    if int(cfg.get("mask_mode") or 0) >= 6 and not xseg:
        note = ("mask_mode=%s 要 XSeg 模型，但找不到 —— 引擎会退化成整块矩形贴脸"
                % cfg.get("mask_mode"))
    if settings.get("face_margin") not in (None, ""):
        note = (note + "；" if note else "") + "face_margin 只影响抽帧对齐，预览不读它"

    # 同一个 t 连着来（前端 120ms 防抖会把同一帧请求两遍）：整份响应直接复用 ——
    # 不取帧、不检测、不换脸。只留 1 条，键里带着文件身份 / t / 模型 / 遮罩 / 参数，
    # 任何一项变了都不会串味。
    try:
        hit_sig = json.dumps([list(_grab_key(video)), round(float(t), 3),
                              str(dfm), str(xseg), detector, float(thresh),
                              cfg, settings,
                              [[m["name"], m["dfm"], list(m["faces"])]
                               for m in (models or [])]], sort_keys=True, default=str)
    except Exception:
        hit_sig = ""
    if hit_sig:
        with _PREVIEW_HIT_LOCK:
            hit = _PREVIEW_HIT.get(hit_sig)
        if hit is not None:
            hit_res = dict(hit)
            tm = dict(hit_res.get("timings") or {})
            tm["grab_ms"] = 0.0            # 这一帧真没再取：grab 的活全省了
            tm["cache_hit"] = 1
            hit_res["timings"] = tm
            hit_res["ms"] = int(round((time.time() - t_start) * 1000.0))
            try:
                hit_res["worker"] = _WORKER.status()
            except Exception:
                pass
            return hit_res

    stop = threading.Event()
    state = {"stage": "start"}
    holder = {}
    done = threading.Event()

    def _run():
        try:
            holder["r"] = _preview_job({"video": video, "t": float(t), "dfm": dfm,
                                        "models": models,
                                        "xseg": xseg, "config": cfg,
                                        "detector": detector, "thresh": float(thresh),
                                        "stop": stop, "state": state,
                                        "no_swap": _face_ids_val(
                                            "no_swap_faces", body.get("no_swap_faces"))})
        except _BadInput as e:
            holder["e"] = e
        except Exception as e:                  # noqa: BLE001
            traceback.print_exc()
            holder["e"] = _BadInput("预览失败：%s"
                                    % (str(e)[:200] or e.__class__.__name__), 500)
        finally:
            done.set()

    threading.Thread(target=_run, daemon=True, name="merge-preview").start()

    # 15s 是「干活」的上限；worker 冷启动/加载模型的那一段单独给它时间，
    # 但总时长封在 work + load 以内，不会无限等下去。
    soft = t_start + _PREVIEW_WORK_TIMEOUT
    ceiling = t_start + _PREVIEW_WORK_TIMEOUT + _PREVIEW_LOAD_TIMEOUT + _PREVIEW_BOOT_TIMEOUT
    grace_used = False
    while not done.is_set():
        now = time.time()
        if now >= soft:
            stage = str(state.get("stage") or "")
            if stage in ("boot", "loading_model", "queued") and now < ceiling and not grace_used:
                grace_used = True
                soft = min(ceiling, now + _PREVIEW_LOAD_TIMEOUT)
                continue
            stop.set()
            if stage != "queued":
                # 杀进程才是真的取消（ONNX 推理从外部打断不了）。
                # queued 的时候 worker 正被另一个请求用着，不能连坐。
                _WORKER.close("预览超时")
            raise _BadInput("预览超时：单帧上限 %.0fs（t=%.2fs，阶段 %s）—— 已终止这次任务"
                            % (_PREVIEW_WORK_TIMEOUT, t, stage or "?"), 504)
        done.wait(0.1)

    if "e" in holder:
        raise holder["e"]
    res = holder.get("r") or {}
    jpg = res.get("jpg") or b""
    if not jpg:
        raise _BadInput("预览没有产出图", 500)

    resp = _ok(png="data:image/jpeg;base64," + _b64.b64encode(jpg).decode("ascii"),
                ms=int(round((time.time() - t_start) * 1000.0)),
                box=res.get("box"), has_face=bool(res.get("has_face")),
                no_faces=bool(res.get("no_faces")),
                swapped=bool(res.get("swapped")),
                video=str(video), t=round(float(t), 3),
                duration=round(float(dur), 3) if dur else None,
                model=model_name, dfm=str(dfm),
                model_size=res.get("model_size"),
                mask_mode=int(cfg.get("mask_mode") or 0),
                mode=cfg.get("mode"), color_transfer=cfg.get("color_transfer_mode"),
                erode_mask=cfg.get("erode_mask_modifier"),
                blur_mask=cfg.get("blur_mask_modifier"),
                xseg=xseg, detector=detector,
                settings=settings, timings=res.get("timings") or {},
                worker=_WORKER.status(), note=res.get("note") or note)
    if models:                          # 多模型：只增不改，把「谁被用了」摊开给前端核对
        resp["models"] = [{"name": m["name"], "dfm": m["dfm"],
                           "faces": list(m["faces"])} for m in models]
        resp["models_used"] = list(res.get("models_used") or [])
        resp["faces"] = list(res.get("faces") or [])
        resp["swaps"] = int(res.get("swaps") or 0)
        if res.get("skipped"):
            resp["skipped"] = list(res.get("skipped") or [])
    if hit_sig:                        # 只留最新一帧：前端再问同一个 t 就直接给这份
        with _PREVIEW_HIT_LOCK:
            _PREVIEW_HIT.clear()
            _PREVIEW_HIT[hit_sig] = resp
    return resp


@router.get("/api/merge/preview/status")
@_api
def merge_preview_status():
    """预览 worker 的现状（排障用：起没起、加载过哪些模型、跑在哪个 provider 上）。"""
    return _ok(**_WORKER.status())

@router.post("/api/merge/preview-frame")
@_api
def merge_preview_frame(body: dict):
    """一张**静态帧文件** -> 检测 -> 换脸 -> 遮罩 -> 合成 -> 图（data: URL）。

    和 /api/merge/preview 只差一点：帧不是按时刻从视频里取的，而是直接读一个
    图片文件（项目模式就是 <项目>/000000.jpg）。检测 / 68 点 / 换脸 / 合成
    仍然是 _preview_job 那一份 —— 同一帧走预览和走项目导出，输出是同一份字节。
    """
    t_start = time.time()
    body = _as_dict(body)
    fp = _norm_path(body.get("frame_file") or body.get("image") or body.get("path"))
    if fp is None:
        raise _BadInput("缺少 frame_file：要一张帧图（jpg/png）的路径")
    if not fp.is_file():
        raise _BadInput("帧图不存在：%s" % fp, 404)
    if fp.suffix.lower() not in _IMAGE_EXT:
        raise _BadInput("不是图片：%s（只认 %s）"
                        % (fp.suffix or "无扩展名", " ".join(sorted(_IMAGE_EXT))))

    models = _models_of(body)                 # None = 老的单模型路径
    if models:
        dfm, model_name = Path(models[0]["dfm"]), models[0]["name"]
    else:
        dfm, model_name = _dfm_of(body)
    settings = _settings_val(body.get("settings"))
    cfg = _merge_settings_to_config(settings)
    detector = _detector_val(body.get("detector"))
    thresh = _float_val("thresh", body.get("thresh"), 0.0, 1.0, 0.5)
    xseg = _xseg_of(body)
    fmt = "png" if str(body.get("fmt") or "jpg").strip().lower() == "png" else "jpg"
    quality = _int_val("quality", body.get("quality"), 1, 100, _PREVIEW_QUALITY)

    frame = _app_mod._imread_any(fp)
    if frame is None:
        raise _BadInput("这张帧图解不开：%s" % fp.name, 422)
    stop = threading.Event()
    state = {"stage": "start"}
    res = _preview_job({"frame": frame, "models": models, "dfm": dfm,
                        "xseg": xseg, "config": cfg, "detector": detector,
                        "thresh": float(thresh), "stop": stop, "state": state,
                        "fmt": fmt, "quality": int(quality),
                        "no_swap": _face_ids_val("no_swap_faces",
                                                 body.get("no_swap_faces")),
                        "send_lossless": (fmt == "png")})
    img = res.get("img") or res.get("jpg") or b""
    if not img:
        raise _BadInput("这一帧没有产出图", 500)
    mime = "image/png" if fmt == "png" else "image/jpeg"
    resp = _ok(png="data:%s;base64,%s" % (mime, _b64.b64encode(img).decode("ascii")),
               ms=int(round((time.time() - t_start) * 1000.0)),
               frame_file=str(fp), fmt=fmt, bytes=len(img),
               box=res.get("box"), has_face=bool(res.get("has_face")),
               swapped=bool(res.get("swapped")), note=res.get("note") or "",
               timings=res.get("timings") or {},
               model=model_name, dfm=str(dfm), xseg=xseg, detector=detector,
               settings=settings)
    if models:
        resp["models"] = [{"name": m["name"], "dfm": m["dfm"],
                           "faces": list(m["faces"])} for m in models]
        resp["faces"] = list(res.get("faces") or [])
        resp["swaps"] = int(res.get("swaps") or 0)
    return resp



# ============================================================
# 预览预热：POST /api/merge/preview/warm  +  GET .../warm/status
# ------------------------------------------------------------
# 冷启动一次 /preview 要 5.3s：worker 起进程 1.7s + 检测器 1.9s + DFM 1.4s。
# 用户点开视频总要先看几秒原片 —— 这段时间足够把三样东西在后台备齐。
#
# 语义：确保 worker 起来了、检测器加载了、指定 DFM 已经在 worker 里了；
#      不取帧、不推理、不合成。
# 返回：立刻回（前端 300ms 内拿到），活交给一根后台线程：
#      · 已经热了 -> {"ok":true,"warming":false,"ready":true,"ms":...,"cold":false}
#      · 要干活   -> {"ok":true,"warming":true,"ready":false,"cold":true/false}
# 并发：预热走 _WORKER.call()，和 /preview 抢同一把 _PreviewWorker._lock ——
#      有 preview 在飞就排队，不插队，也不会把 worker 挤成两份。
# 幂等：worker 活着 + 这个 dfm 已在 _WORKER._models 里 + 检测器已在
#      app._det_cache 里 -> 直接回 warming=false，连 worker 都不惊动（<20ms）。
# 多模型：body 也可以给 models: [{name, dfm, faces}]，一次热一组（worker 里最多
#      常驻 3 个，超了 LRU 淘汰）—— 幂等判定要求整组都在，缺一个就整组重热。
# 失败：模型不存在在接口里就 404 掉（不走线程）；线程里炸了写进 status 的
#      error 字段并打日志，不会静默死掉。
# ============================================================
_PREVIEW_WARM_TIMEOUT = 60.0       # 预热里「干活」的上限（加载模型另算）
_WARM_LOCK = threading.Lock()
_WARM_THREAD = None                # 一根线程串行干，不并发建模型
_WARM_PENDING = None               # 只留最新一个请求：(dfm, 模型名, 检测器, 进接口时刻)
_WARM_STATE = {
    "ready": False,                # worker / 检测器 / DFM 三样齐了
    "warming": False,              # 后台还有活在跑
    "stage": "",                   # 走到哪一步了（queued / boot / loading_model / ready）
    "model": "",                   # 界面上的模型名（多模型时是第 0 个）
    "dfm": "",                     # 真正加载的那个 .dfm（多模型时是第 0 个）
    "dfms": [],                    # 这一轮要热的整组 .dfm（单模型就一个）
    "models": [],                  # [{name, dfm, size?}] —— 和 dfms 一一对应
    "models_used": [],             # 真装进显存的那几个名字（≤3）
    "detector": "",
    "ms": None,                    # 这一轮从进接口到加载完的耗时
    "cold": None,                  # 这一轮是不是从「worker 没起」开始的
    "boot_ms": None,               # worker 自报的 import 耗时
    "load_ms": None,               # worker 自报的模型加载耗时
    "first_pass_ms": None,         # 第一次前向（建 kernel/句柄）的耗时
    "detector_ms": None,
    "worker_pid": None,
    "error": "",
    "xseg_error": "",
    "at": None,                    # 就绪时刻（epoch 秒）
}


def _det_cached(name) -> bool:
    """检测器是不是已经在父进程里了 —— 和 app._get_detector 看同一张缓存表。

    缓存键就是规范名（tinymog / retinaface_10g），所以这里先把入参归一。
    """
    cache = getattr(_app_mod, "_det_cache", None) or {}
    try:
        canon = _detector_val(name)
    except _BadInput:
        return False
    return canon in cache


def _worker_pid():
    """worker 的 pid；没起来就是 None（别拿跑过的旧 pid 忽悠前端）。"""
    try:
        p = _WORKER._proc
        if p is not None and p.poll() is None:
            return int(p.pid)
    except Exception:                              # noqa: BLE001
        pass
    return None


def _worker_models() -> dict:
    try:
        return dict(_WORKER._models)
    except Exception:                              # noqa: BLE001
        return {}


def _warm_hot(dfms, det_name):
    """三样都齐了吗？齐了就不用干活。返回 (是否就绪, 还缺什么)。

    dfms 可以是单个路径，也可以是一组（多模型：整组都在才算热）。
    """
    if not _WORKER.alive():
        return False, "worker 没起来"
    want = [str(p) for p in (dfms if isinstance(dfms, (list, tuple, set)) else [dfms])]
    have = _worker_models()
    missing = [p for p in want if p not in have]
    if missing:
        if len(want) > 1:
            return False, "DFM 还没在 worker 里加载：%s" % os.path.basename(missing[0])
        return False, "DFM 还没在 worker 里加载"
    if not _det_cached(det_name):
        return False, "检测器还没加载"
    return True, ""


def _warm_note_hot(dfms, names, det_name):
    """已经热了：顺手把状态对齐到这次请求的模型。

    status 里的 model 是「最近一次预热的目标」——不跟着请求走的话，先热 A 再热 B、
    又切回本来就还热着的 A 时，status 会一直粘在 B 上。
    """
    dfms = [str(p) for p in (dfms if isinstance(dfms, (list, tuple)) else [dfms])]
    names = [str(n or "") for n in (names if isinstance(names, (list, tuple)) else [names])]
    with _WARM_LOCK:
        hot, why = _warm_hot(dfms, det_name)
        if hot and not _WARM_STATE.get("warming"):
            _WARM_STATE.update({"ready": True, "stage": "ready", "error": "",
                                "model": (names[0] if names else ""),
                                "dfm": (dfms[0] if dfms else ""),
                                "dfms": list(dfms),
                                "models": [{"name": n, "dfm": d}
                                           for n, d in zip(names, dfms)],
                                "models_used": [n for n in names if n],
                                "detector": str(det_name)})
        return hot, why


def _warm_set(**kw):
    with _WARM_LOCK:
        _WARM_STATE.update(kw)


def _warm_fail(names, dfms, det_name, msg, t_req):
    """失败也要留痕：状态里带 error、日志里有一行 —— 不能悄无声息死在后台。"""
    names = [str(n or "") for n in (names if isinstance(names, (list, tuple)) else [names])]
    dfms = [str(p or "") for p in (dfms if isinstance(dfms, (list, tuple)) else [dfms])]
    _warm_set(ready=False, stage="error", model=(names[0] if names else ""),
              dfm=(dfms[0] if dfms else ""), dfms=list(dfms),
              detector=str(det_name or ""), at=None,
              ms=round((time.time() - t_req) * 1000.0, 1),
              error=str(msg or "预热失败")[:400], worker_pid=_worker_pid())
    print("[merge-preview] 预热失败（%s）：%s" % (dfms, str(msg)[:300]), flush=True)


def _warm_do(dfms, names, det_name, t_req):
    """真的干一次：检测器（父进程）+ worker + 整组 DFM。只在线程里跑。"""
    dfms = [str(p) for p in (dfms if isinstance(dfms, (list, tuple)) else [dfms])]
    names = [str(n or "") for n in (names if isinstance(names, (list, tuple)) else [names])]
    dfm = dfms[0] if dfms else ""
    model_name = names[0] if names else ""
    box = {}

    def _det_job():
        t = time.time()
        try:
            det = _detector(det_name)
            # 检测器的第一帧推理要建 session/kernel（实测 ~0.9s）：拿一张黑图先跑掉，
            # 不取视频帧也不合成，纯属把这份一次性开销留在预热里。thresh 用默认值，
            # 和 /preview 里那个 = _float_val(..., 0.5) 的默认一致。
            try:
                import numpy as _np
                with _det_run_lock:
                    det.thresh = 0.5
                    det.detect(_np.zeros((720, 1280, 3), _np.uint8))
            except Exception:                  # noqa: BLE001
                pass
            box["ms"] = round((time.time() - t) * 1000.0, 1)
        except _BadInput as e:
            box["err"] = e.msg
        except Exception as e:                     # noqa: BLE001
            box["err"] = "%s: %s" % (e.__class__.__name__, str(e)[:200])

    # 检测器在父进程、模型在子进程，两边互不相干 —— 并行跑，墙钟少一半
    det_t = threading.Thread(target=_det_job, daemon=True,
                             name="merge-preview-warm-det")
    det_t.start()
    cold = not _WORKER.alive()
    _warm_set(cold=cold, stage="boot")
    try:
        xseg = _xseg_of({})                        # 顺手把遮罩模型也装进去
    except _BadInput:
        xseg = ""
    try:
        resp = _WORKER.call({"cmd": "warm", "dfm": str(dfm), "dfms": list(dfms),
                             "xseg": xseg},
                            work_timeout=_PREVIEW_WARM_TIMEOUT,
                            load_timeout=_PREVIEW_LOAD_TIMEOUT,
                            on_stage=lambda st: _warm_set(stage=st))
    finally:
        det_t.join(timeout=60.0)
    if not resp.get("ok"):
        raise _BadInput("预热失败：%s" % (resp.get("error") or "worker 没给原因"), 500)
    det_err = str(box.get("err") or "")
    if det_err:
        raise _BadInput("检测器没加载起来：%s" % det_err, 500)

    at = time.time()
    pid = resp.get("pid") or _worker_pid()
    per = {os.path.normcase(str(p.get("dfm") or "")): p
           for p in (resp.get("per_model") or []) if isinstance(p, dict)}
    warmed = []
    for n, d in zip(names, dfms):
        hit = per.get(os.path.normcase(str(d))) or {}
        warmed.append({"name": n, "dfm": str(d), "size": hit.get("size"),
                       "load_ms": hit.get("load_ms"), "first_ms": hit.get("first_ms")})
    with _WARM_LOCK:
        # 期间又来了别的模型的请求：这一轮结果不写进状态（免得报假就绪）
        if _WARM_PENDING is not None:
            pending = [str(p) for p in (_WARM_PENDING[0] or [])]
            if pending != list(dfms):
                return
        _WARM_STATE.update({
            "ready": True, "stage": "ready", "model": str(model_name or ""),
            "dfm": str(dfm), "dfms": list(dfms), "models": warmed,
            "models_used": _used_names([{"name": n, "dfm": d} for n, d in zip(names, dfms)],
                                       resp.get("models")),
            "detector": str(det_name), "at": at,
            "ms": round((at - t_req) * 1000.0, 1), "cold": bool(cold),
            "boot_ms": _WORKER._boot_ms, "load_ms": resp.get("warm_ms"), "first_pass_ms": resp.get("first_ms"),
            "detector_ms": box.get("ms"), "worker_pid": pid,
            "error": "", "xseg_error": str(resp.get("xseg_error") or "")[:400],
        })


def _warm_loop():
    """一根线程串行干：排着的活一个一个来，没活了自己退场。"""
    global _WARM_PENDING, _WARM_THREAD
    while True:
        with _WARM_LOCK:
            job = _WARM_PENDING
            _WARM_PENDING = None
            if job is None:
                _WARM_STATE["warming"] = False
                _WARM_THREAD = None
                return
        dfms, names, det_name, t_req = job
        try:
            _warm_do(dfms, names, det_name, t_req)
        except _BadInput as e:
            _warm_fail(names, dfms, det_name, e.msg, t_req)
        except Exception as e:                     # noqa: BLE001
            traceback.print_exc()
            _warm_fail(names, dfms, det_name,
                       "预热失败：%s" % (str(e)[:200] or e.__class__.__name__), t_req)


def _warm_kick(dfms, names, det_name, t_req) -> bool:
    """把活排上（立刻返回）。返回 True = 这一下把线程拉起来了（之前没人干）。"""
    global _WARM_PENDING, _WARM_THREAD
    dfms = [str(p) for p in (dfms if isinstance(dfms, (list, tuple)) else [dfms])]
    names = [str(n or "") for n in (names if isinstance(names, (list, tuple)) else [names])]
    with _WARM_LOCK:
        _WARM_PENDING = (tuple(dfms), tuple(names), str(det_name), float(t_req))
        _WARM_STATE.update({"warming": True, "ready": False, "error": "",
                            "stage": "queued", "model": (names[0] if names else ""),
                            "dfm": (dfms[0] if dfms else ""), "dfms": list(dfms),
                            "models": [{"name": n, "dfm": d} for n, d in zip(names, dfms)],
                            "models_used": [], "detector": str(det_name),
                            "cold": not _WORKER.alive()})
        if _WARM_THREAD is not None and _WARM_THREAD.is_alive():
            return False
        _WARM_THREAD = threading.Thread(target=_warm_loop, daemon=True,
                                        name="merge-preview-warm")
        _WARM_THREAD.start()
        return True


def _warm_status() -> dict:
    """预热状态。ready 是现场算的：worker 挂了 / 模型被换掉，立刻变 false。"""
    with _WARM_LOCK:
        st = dict(_WARM_STATE)
    dfm = str(st.get("dfm") or "")
    dfms = [str(p) for p in (st.get("dfms") or ([dfm] if dfm else []))]
    models = _worker_models()
    hot = (bool(dfms) and _WORKER.alive() and all(p in models for p in dfms)
           and _det_cached(st.get("detector")))
    st["ready"] = bool(hot)
    st["worker_pid"] = _worker_pid()
    st["worker_models"] = sorted(models)
    st["max_models"] = int(getattr(_WORKER, "_max_models", _WORKER_MAX_MODELS) or 3)
    st["detector_loaded"] = _det_cached(st.get("detector")) if st.get("detector") else False
    if not hot:
        st["reason"] = (_warm_hot(dfms, st.get("detector"))[1] if dfms else "还没预热过")
    st["ts"] = round(time.time(), 3)
    return st


@router.post("/api/merge/preview/warm")
@_api
def merge_preview_warm(body: dict = None):
    """后台把 worker / 检测器 / DFM 备好（不取帧、不合成），立刻返回。

    body: {"model": "<模型名>"}（可省；省了用 workspace/model 下第一个 .dfm）
          也可以给 "models": [{name, dfm, faces}] 一次热一组（worker 里最多常驻 3 个，
          超了 LRU 淘汰；faces 在预热里不参与——预热只关心"哪几个模型"）。
    已经热了就 {"ok":true,"warming":false,"ms":…,"cold":false}；要干活就
    {"ok":true,"warming":true,...}，进度看 GET /api/merge/preview/warm/status。
    """
    t_req = time.time()
    body = _as_dict(body)
    models = _models_of(body)                  # None = 老的单模型路径
    if models:
        dfms = [m["dfm"] for m in models]
        names = [m["name"] for m in models]
    else:
        dfm, model_name = _dfm_of(body)        # 模型不存在：这里就 404，不进线程
        dfms, names = [str(dfm)], [str(model_name or "")]
    det_name = _detector_val(body.get("detector"))

    hot, why = _warm_note_hot(dfms, names, det_name)
    ms = int(round((time.time() - t_req) * 1000.0))
    if hot:
        return _ok(warming=False, ready=True, ms=ms, cold=False,
                   model=names[0], dfm=dfms[0], detector=det_name,
                   worker_pid=_worker_pid(),
                   models=[{"name": n, "dfm": d} for n, d in zip(names, dfms)],
                   models_used=list(names), max_models=_WORKER_MAX_MODELS)

    started = _warm_kick(dfms, names, det_name, t_req)
    return _ok(warming=True, ready=False, ms=int(round((time.time() - t_req) * 1000.0)),
               cold=(_worker_pid() is None), model=names[0], dfm=dfms[0],
               detector=det_name, worker_pid=_worker_pid(),
               models=[{"name": n, "dfm": d} for n, d in zip(names, dfms)],
               models_used=[], max_models=_WORKER_MAX_MODELS,
               started=bool(started), reason=why)


@router.get("/api/merge/preview/warm/status")
@_api
def merge_preview_warm_status():
    """预热状态：{ready, model, ms, warming, cold, error, worker_pid, …}。"""
    return _ok(**_warm_status())


# ============================================================
# DFL 项目模式（iperov 布局）· 逐帧合成落盘 + 按原 fps/音轨重新编码
# ------------------------------------------------------------
# 项目目录（和 iperov 的 data_dst 结构一致）：
#
#   <项目>/000000.jpg              原帧（6 位序号，按顺序）
#   <项目>/aligned/000000_0.jpg    对齐脸 DFLJPG；元数据 source_filename 指回
#                                  "000000"（也认 "000000.jpg"）
#   <项目>/merged/000000.jpg       产物：换脸后的**整帧**（本接口写的）
#
# 和 /api/merge/export 的分工：
#   /export          -> 4.0 MergeStudio 子进程一口气出 mp4，断了要从头再跑
#   /export-project  -> 逐帧合成落盘到 merged/，再单独编码；能断点续跑、
#                       能报失败帧、产物可以一张张复查
#
# 「逐帧合成」这一段不另写一套：直接复用单帧预览的 _preview_job
# （检测 -> 2D106 的 68 点 -> DFM 换脸 + 遮罩 + 合成），只把「从视频里取帧」
# 换成「从 <项目>/<帧名> 读图」。同一帧走预览和走导出，输出是同一份字节。
# ============================================================

_PROJ_LOCK = threading.RLock()
_PROJ_PCT_SWAP = 96          # 逐帧合成占 0~96%，剩下的给编码
_PROJ_MAX_LIST = 200         # 失败帧 / 缺帧清单最多回传这么多条
_PROJ_FRAME_EXT = {".jpg", ".jpeg", ".png"}
_PROJ_JOB = {
    "running": False, "ok": None, "phase": "idle", "cancelled": False,
    "error": "", "error_detail": "", "stage": "idle", "pct": 0, "msg": "",
    "project": "", "frames_dir": "", "aligned": "", "merged": "",
    "video": "", "out_video": "", "fps": 0.0, "width": 0, "height": 0,
    "video_frames": 0, "total": 0, "done": 0, "swapped": 0, "copied": 0,
    "skipped": 0, "failed": 0, "frame_fps": 0.0, "eta_s": None, "elapsed": 0.0,
    "started_at": None, "finished_at": None, "force": False,
    "dfm": "", "model": "", "models": [], "settings": {}, "detector": "",
    "warnings": [], "failed_frames": [], "missing": [], "missing_source": [],
    "aligned_faces": 0, "merged_frames": 0, "encode": {}, "log": [],
    "want_cancel": False,
}


class _ProjCancel(Exception):
    """用户取消（和失败分开记：取消不是错误）。"""


def _proj_snap() -> dict:
    with _PROJ_LOCK:
        snap = dict(_PROJ_JOB)
        for k in ("warnings", "failed_frames", "missing", "missing_source",
                  "models", "log"):
            snap[k] = list(snap.get(k) or [])
        snap["settings"] = dict(snap.get("settings") or {})
        snap["encode"] = dict(snap.get("encode") or {})
        return snap


def _proj_set(**kw):
    with _PROJ_LOCK:
        _PROJ_JOB.update(kw)


def _proj_log(msg):
    line = str(msg)[:300]
    with _PROJ_LOCK:
        log = _PROJ_JOB.get("log")
        if not isinstance(log, list):
            log = []
            _PROJ_JOB["log"] = log
        log.append(line)
        if len(log) > 300:
            del log[:-300]


def _proj_warn(msg):
    """记一条警告：进 warnings（给界面看）同时进 log（给排障用）。"""
    text = str(msg)[:400]
    with _PROJ_LOCK:
        warns = _PROJ_JOB.get("warnings")
        if not isinstance(warns, list):
            warns = []
            _PROJ_JOB["warnings"] = warns
        if text not in warns:
            warns.append(text)
            if len(warns) > 40:
                del warns[:-40]
    _proj_log("[warn] " + text)


def _proj_cancelled() -> bool:
    with _PROJ_LOCK:
        return bool(_PROJ_JOB.get("want_cancel"))


def _proj_frame_files(dirp) -> list:
    """目录里的帧图名（按文件名排序 = ffmpeg image2 的读取顺序）。"""
    out = []
    try:
        with os.scandir(str(dirp)) as it:
            for e in it:
                try:
                    if e.is_file() and Path(e.name).suffix.lower() in _PROJ_FRAME_EXT:
                        out.append(e.name)
                except OSError:
                    continue
    except OSError:
        return []
    return sorted(out)


def _proj_head_meta(path, limit=96 * 1024) -> dict:
    """只读文件头就把 DFLJPG 里的 APP15（pickle）解出来。

    app._dfljpg_dict() 是整份读进内存 —— 四万张对齐脸就是几 GB 的读放大。
    APP15 紧跟 SOI，96KB 足够；段长超过窗口时才回退到整份读。
    """
    try:
        with open(str(path), "rb") as fh:
            data = fh.read(int(limit))
    except OSError:
        return {}
    i, n = 0, len(data)
    while i < n - 1:
        if data[i] != 0xFF:
            return {}
        marker = data[i + 1]
        if marker in (0xD8, 0xD9) or 0xD0 <= marker <= 0xD7:
            i += 2
            continue
        if i + 4 > n:
            return {}
        seg_len = int.from_bytes(data[i + 2:i + 4], "big")
        if seg_len < 2:
            return {}
        if marker == 0xEF:                        # APP15
            end = i + 2 + seg_len
            if end > n:                           # 段比窗口还长：退回整份读
                try:
                    return _app_mod._dfljpg_dict(Path(path)) or {}
                except Exception:
                    return {}
            try:
                d = _app_mod._safe_loads(data[i + 4:end])
            except Exception:
                return {}
            return d if isinstance(d, dict) else {}
        i += 2 + seg_len
    return {}


def _proj_match_frame(src, frame_set, by_stem, shift=0):
    """DFLJPG 的 source_filename -> 真实存在的帧名；认不出来回 None（不猜）。"""
    s = str(src or "").strip().replace("\\", "/")
    if not s:
        return None
    base = s.rsplit("/", 1)[-1]
    if base in frame_set:
        return base
    stem = Path(base).stem
    if stem in by_stem:
        return by_stem[stem]
    if shift and stem.isdigit():
        alt = str(int(stem) + int(shift))
        if len(alt) < len(stem):
            alt = alt.zfill(len(stem))
        if alt in by_stem:
            return by_stem[alt]
    return None


def _proj_scan_aligned(aligned_dir, frames) -> dict:
    """aligned 目录 -> 每个源帧对应哪几张对齐脸。

    返回 {map: {帧名: [脸文件名…]}, missing: [认不出原帧的 source_filename…],
          no_meta: 几张没有元数据, shift: 用到的人数偏移, faces: 总脸数}

    认帧顺序：source_filename 原名 -> 去扩展名 -> 整体平移 ±1（老项目帧从
    000001 起、对齐从 000000 起，差一位）。三种都对不上就进 missing，绝不瞎配。
    """
    frame_set = set(frames)
    by_stem = {}
    for n in frames:
        by_stem.setdefault(Path(n).stem, n)
    try:
        names = sorted(n for n in os.listdir(str(aligned_dir))
                       if Path(n).suffix.lower() in (".jpg", ".jpeg"))
    except OSError:
        names = []
    pairs, no_meta = [], 0
    for n in names:
        meta = _proj_head_meta(Path(str(aligned_dir)) / n)
        if not meta:
            no_meta += 1
            continue
        src = str(meta.get("source_filename") or "").strip()
        if not src:                       # 没有这个字段：按文件名前缀认
            src = n.rsplit("_", 1)[0] if "_" in n else Path(n).stem
        pairs.append((n, src))
    def _hits(shift):
        return sum(1 for _n, s in pairs
                   if _proj_match_frame(s, frame_set, by_stem, shift))

    best_shift = 0
    if pairs and _hits(0) <= 0:
        # 一条都对不上才怀疑是「整体差一位」的老项目（帧 000001 起、对齐 000000 起）。
        # 只要原名能对上就别平移：总数上「多对上一张」也可能把 A 的脸贴到 B 上。
        scored = sorted(((_hits(s), s) for s in (1, -1)), reverse=True)
        if scored and scored[0][0] > 0:
            best_shift = scored[0][1]
    out, missing = {}, []
    for n, src in pairs:
        cand = _proj_match_frame(src, frame_set, by_stem, best_shift)
        if cand is None:
            if len(missing) < _PROJ_MAX_LIST:
                missing.append(str(src) or n)
            continue
        out.setdefault(cand, []).append(n)
    return {"map": out, "missing": missing, "no_meta": int(no_meta),
            "shift": int(best_shift), "faces": len(pairs)}


def _proj_seq_info(names, contiguous=True):
    """帧名列表 -> (prefix, 数字位宽, 后缀, 起始编号, 编号列表)；认不出就 None。

    认不出来（前缀/后缀/位宽不统一、编号不连号）的帧序列没法用 image2 的
    %0Nd 写法喂给 ffmpeg —— 有洞时它会在第一个洞停住，成片被悄悄截短。
    """
    if not names:
        return None
    rx = re.compile(r"^(.*?)(\d+)(\D*)$")
    parsed = []
    for n in names:
        m = rx.match(str(n))
        if not m:
            return None
        parsed.append((m.group(1), int(m.group(2)), len(m.group(2)), m.group(3)))
    prefix, suffix, width = parsed[0][0], parsed[0][3], parsed[0][2]
    for p, _num, w, suf in parsed:
        if p != prefix or suf != suffix or w != width:
            return None
    nums = [x[1] for x in parsed]
    if sorted(nums) != nums:
        return None
    if contiguous and nums[-1] - nums[0] + 1 != len(nums):
        return None                        # 编号有洞（image2 会在这儿停住）
    return prefix, int(width), suffix, int(nums[0]), nums


def _proj_pattern(names):
    """帧名列表 -> (image2 的 pattern, 起始编号)；认不出 / 编号有洞就 None。"""
    info = _proj_seq_info(names)
    if info is None:
        return None
    prefix, width, suffix, start, _nums = info
    return prefix + "%0" + str(width) + "d" + suffix, int(start)


def _proj_put(dst: Path, data=None, src=None):
    """原子落盘：先写 .part 再 os.replace。

    中途被杀 / 断电只会留下 .part，不会把「写了一半的图」当成已完成 ——
    断点续跑靠的就是「文件存在且非空」这一条判据。
    """
    tmp = Path(str(dst) + ".part")
    try:
        if data is not None:
            with open(str(tmp), "wb") as fh:
                fh.write(data)
        else:
            shutil.copyfile(str(src), str(tmp))
        os.replace(str(tmp), str(dst))
    finally:
        try:
            if tmp.exists():
                tmp.unlink()
        except OSError:
            pass


def _proj_encode(pattern, start, fps, video, out_video, total) -> dict:
    """按需求里那条命令编码：merged 帧 + 原视频音轨，libx264 / crf16 / faststart。

    刻意**不加 -shortest**：音轨比画面长时它会把成片截短（4.0 那边踩过）。
    """
    ffmpeg = str(_app_mod._ffmpeg_exe())
    ext = Path(str(out_video)).suffix.lower()
    cmd = [ffmpeg, "-y", "-hide_banner", "-loglevel", "error", "-nostats", "-nostdin",
           "-framerate", "%.6g" % float(fps),
           "-start_number", str(int(start)), "-i", str(pattern),
           "-i", str(video),
           "-map", "0:v:0", "-map", "1:a?",
           # 1) 奇数宽高在 yuv420p 下会被 libx264 直接拒绝，垫一像素最省事；
           # 2) merged 是 JPEG（full range），不显式转 limited 的话成片的 pix_fmt
           #    会被标成 yuvj420p(pc)，很多播放器按 tv 解释 → 整体发灰。
           "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2,scale=out_range=tv",
           "-c:v", "libx264", "-crf", "16", "-pix_fmt", "yuv420p",
           "-c:a", "copy"]
    if ext in (".mp4", ".mov", ".m4v"):
        cmd += ["-movflags", "+faststart"]
    cmd += ["-progress", "pipe:1", str(out_video)]

    err_path = None
    code = -1
    fh = tempfile.NamedTemporaryFile(delete=False, suffix=".ffmpeg.log")
    err_path = fh.name
    fh.close()
    try:
        with open(str(err_path), "wb") as errfp:
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=errfp,
                                    text=True, encoding="utf-8", errors="replace",
                                    creationflags=getattr(_app_mod, "_NO_WINDOW", 0))
            try:
                for line in proc.stdout:
                    if _proj_cancelled():
                        _kill_proc_tree(proc)
                        raise _ProjCancel()
                    line = line.strip()
                    if not line or "=" not in line:
                        continue
                    k, _eq, v = line.partition("=")
                    if k != "frame":
                        continue
                    try:
                        done = int(v)
                    except ValueError:
                        continue
                    pct = _PROJ_PCT_SWAP
                    if total > 0:
                        pct = _PROJ_PCT_SWAP + int((100 - _PROJ_PCT_SWAP) * done / total)
                    _proj_set(pct=min(99, pct), stage="编码",
                              msg="编码 %d/%d 帧" % (done, total))
            finally:
                code = proc.wait()
    finally:
        tail = ""
        try:
            with open(str(err_path), "rb") as rf:
                raw = rf.read()[-4000:]
            tail = raw.decode("utf-8", "ignore").strip()
        except OSError:
            tail = ""
        try:
            os.unlink(str(err_path))
        except OSError:
            pass
    return {"code": int(code), "tail": tail[-1200:],
            "cmd": " ".join('"%s"' % a if " " in str(a) else str(a) for a in cmd)}


def _proj_source_video(raw, project: Path, frames_dir: Path):
    """原视频（fps / 分辨率 / 音轨的唯一来源）：显式 > prepare 记下的 > 项目里唯一的片子。

    找不到就回 (None, 原因) 让调用方报错 —— 这里绝不猜一个 fps 出来。
    """
    cands = []
    raw = str(raw or "").strip()
    if raw:
        p = _norm_path(raw)
        if p is None or not p.is_file():
            raise _BadInput("指定的原视频不存在：%s" % (p or raw), 404)
        cands.append(p)
    for d in (str(frames_dir), str(project)):
        try:
            src = _app_mod._merge_source_video(d)
        except Exception:
            src = ""
        if src:
            cands.append(Path(src))
            break
    for d in (project, project.parent):
        try:
            vids = sorted(x for x in Path(str(d)).iterdir()
                          if x.is_file() and x.suffix.lower() in _VIDEO_EXT)
        except OSError:
            vids = []
        vids = [v for v in vids if ".merge_" not in v.name]
        same = [v for v in vids
                if v.stem in (Path(str(project)).name, Path(str(frames_dir)).name)]
        cands.extend(same or (vids if len(vids) == 1 else []))
    for p in cands:
        try:
            if p is not None and Path(str(p)).is_file():
                return Path(str(p)), ""
        except OSError:
            continue
    return None, "项目目录里找不到原视频（.mp4）—— 用 video 参数指定一个"


def _proj_run(cfg: dict):
    project = Path(cfg["project"])
    frames_dir = Path(cfg["frames_dir"])
    aligned_dir = Path(cfg["aligned"])
    merged_dir = Path(cfg["merged"])
    force = bool(cfg.get("force"))

    merged_dir.mkdir(parents=True, exist_ok=True)
    frames = _proj_frame_files(frames_dir)
    if not frames:
        raise _BadInput("项目里没有帧图（.jpg/.png）：%s" % frames_dir, 422)
    ext_count = {}
    for n in frames:
        e = Path(n).suffix.lower()
        ext_count[e] = ext_count.get(e, 0) + 1
    frame_ext = sorted(ext_count, key=lambda k: (-ext_count[k], k))[0]
    if len(ext_count) > 1:
        _proj_warn("帧目录里有多种扩展名（%s），只用 %s 参与编码"
                   % ("、".join(sorted(ext_count)), frame_ext))

    info = _proj_scan_aligned(aligned_dir, frames)
    face_map = info["map"]
    _proj_set(total=len(frames), aligned_faces=int(info["faces"]),
              missing_source=list(info["missing"]))
    _proj_log("项目 %s：%d 帧，%d 张对齐脸，能对上帧的 %d 帧"
              % (project.name, len(frames), info["faces"], len(face_map)))
    if info["no_meta"]:
        _proj_warn("%d 张对齐脸读不出 DFLJPG 元数据（不是 DFLJPG？），已跳过"
                   % int(info["no_meta"]))
    if info["missing"]:
        _proj_warn("aligned 里有 %d 条 source_filename 找不到对应原帧（帧被删了？）：%s"
                   % (len(info["missing"]),
                      "、".join(info["missing"][:6])))
    if info["shift"]:
        _proj_warn("对齐脸的 source_filename 与帧号整体差 %+d（老项目），已按偏移对上"
                   % int(info["shift"]))
    if not face_map:
        _proj_warn("没有任何一张对齐脸能对上帧 —— 这次只是把原帧原样搬进 merged/")

    stop = threading.Event()
    state = {"stage": ""}
    q = int(cfg.get("quality") or _PREVIEW_QUALITY)
    models = cfg.get("models") or None

    t0 = time.time()
    total = len(frames)
    done = swapped = copied = skipped = 0
    failed_frames, face_miss = [], []
    for name in frames:
        if _proj_cancelled():
            stop.set()
            raise _ProjCancel()
        src_p = frames_dir / name
        dst_p = merged_dir / name
        # ---- 断点续跑：merged 里已经有非空文件就跳过（--force 时才重做）----
        if not force and dst_p.is_file() and dst_p.stat().st_size > 0:
            skipped += 1
        else:
            frame = None
            try:
                frame = _app_mod._imread_any(src_p)
            except Exception:
                frame = None
            if frame is None:
                failed_frames.append({"frame": name, "error": "这一帧读不出来（损坏？）"})
            elif not (face_map.get(name) or []):
                # 有没有脸：aligned 说了算。没有就原样搬（无损，也不白跑一次检测）
                try:
                    _proj_put(dst_p, src=src_p)
                    copied += 1
                except Exception as e:              # noqa: BLE001
                    failed_frames.append({"frame": name,
                                          "error": "搬运失败：%s" % str(e)[:120]})
            else:
                fmt = "png" if Path(name).suffix.lower() == ".png" else "jpg"
                try:
                    res = _preview_job({"frame": frame, "models": models,
                                        "dfm": cfg["dfm"], "xseg": cfg["xseg"],
                                        "config": cfg["config"],
                                        "detector": cfg["detector"],
                                        "thresh": float(cfg["thresh"]),
                                        "stop": stop, "state": state,
                                        "fmt": fmt, "quality": q,
                                        "no_face_copy": True})
                except _BadInput as e:
                    res = None
                    failed_frames.append({"frame": name, "error": str(e.msg)[:200]})
                except Exception as e:              # noqa: BLE001
                    res = None
                    traceback.print_exc()
                    failed_frames.append({"frame": name,
                                          "error": "%s: %s" % (e.__class__.__name__,
                                                               str(e)[:160])})
                if res is not None:
                    try:
                        if res.get("copy"):
                            _proj_put(dst_p, src=src_p)
                            copied += 1
                            if not res.get("has_face"):
                                # 有对齐脸但检测不到脸：记下来，别让它悄悄过
                                if len(face_miss) < _PROJ_MAX_LIST:
                                    face_miss.append(name)
                        else:
                            _proj_put(dst_p, data=res.get("img") or res.get("jpg"))
                            swapped += 1
                    except Exception as e:          # noqa: BLE001
                        failed_frames.append({"frame": name,
                                              "error": "落盘失败：%s" % str(e)[:120]})
        done += 1
        el = max(0.001, time.time() - t0)
        # 帧速只按真正干过活的帧算：续跑跳过的不算，否则「全跳过」会报出 3 万帧/秒
        rate = (swapped + copied + len(failed_frames)) / el
        _proj_set(done=done, swapped=swapped, copied=copied, skipped=skipped,
                  failed=len(failed_frames), elapsed=round(el, 1),
                  frame_fps=round(rate, 2),
                  eta_s=(int((total - done) / rate) if rate > 0 and done < total else 0),
                  pct=min(_PROJ_PCT_SWAP, int(_PROJ_PCT_SWAP * done / float(total))),
                  msg="合成 %d/%d 帧 · %.1f 帧/秒%s"
                      % (done, total, rate, ("（跳过 %d）" % skipped) if skipped else ""))
        if len(failed_frames) > _PROJ_MAX_LIST:
            del failed_frames[_PROJ_MAX_LIST:]
        if done % 20 == 0 or done == total:
            _proj_log("合成 %d/%d 帧 · 换脸 %d · 原样 %d · 跳过 %d · 失败 %d"
                      % (done, total, swapped, copied, skipped, len(failed_frames)))

    if face_miss:
        _proj_warn("%d 帧有对齐脸但没检出人脸，已按原帧落盘：%s"
                   % (len(face_miss), "、".join(face_miss[:6])))
    # frame_fps 保留循环里最后一次的值（按真正处理过的帧算，跳过的不算）
    _proj_set(failed_frames=failed_frames)

    # ---- 编码前对账：缺帧必须说清楚缺哪几张，别让成片悄悄变短 ----
    merged_names = [n for n in _proj_frame_files(merged_dir)
                    if Path(n).suffix.lower() == frame_ext]
    merged_set = set(merged_names)
    frames_set = set(frames)
    missing = [n for n in frames if n not in merged_set]       # 有源帧、没产物
    v_frames = int(cfg.get("video_frames") or 0)
    fps = max(0.001, float(cfg["fps"]))
    # 项目和视频本该一一对应才比较帧数（抽帧按 2fps 之类的本来就对不上）
    near_full = bool(v_frames) and (len(frames) + 10 >= v_frames)
    if v_frames and near_full and len(frames) < v_frames:
        # 按命名规律点名缺了哪几张（编号本来就有洞，contiguous=False）
        info_seq = _proj_seq_info(frames, contiguous=False)
        if info_seq is not None:
            prefix, width, suffix, start, _nums = info_seq
            want = [prefix + str(i).zfill(width) + suffix
                    for i in range(int(start), int(start) + v_frames)]
            gone = [n for n in want if n not in frames_set]
            if gone:
                _proj_warn("源视频 %d 帧，项目里缺 %d 帧：%s（这些帧不会出现在成片里）"
                           % (v_frames, len(gone), "、".join(gone[:12])))
                missing = gone + missing
    _proj_set(missing=missing[: _PROJ_MAX_LIST], merged_frames=len(merged_names))
    if len(merged_names) < len(frames):
        lack = [n for n in missing if n in frames_set]
        _proj_warn("merged 只有 %d 帧 < 源帧 %d 帧 —— 缺 %d 帧%s"
                   % (len(merged_names), len(frames), len(frames) - len(merged_names),
                      ("：" + "、".join(lack[:12])) if lack else ""))
    if v_frames and near_full and len(merged_names) < v_frames:
        _proj_warn("成片只会有 %d 帧（源视频 %d 帧），比原片短 %.2f 秒"
                   % (len(merged_names), v_frames,
                      (v_frames - len(merged_names)) / fps))
    if not merged_names:
        raise _BadInput("merged/ 里一帧都没有，没法编码：%s" % merged_dir, 422)

    # ---- 编码 ----
    _proj_set(stage="编码", pct=_PROJ_PCT_SWAP,
              msg="编码 %d 帧 → %s" % (len(merged_names),
                                       Path(str(cfg["out_video"])).name))
    pat = _proj_pattern(merged_names)
    seq_dir = None
    try:
        if pat is None:
            # 编号有洞（缺帧 / 名字不规则）：按顺序硬链接成 %06d 再编码。
            # 不这么做的话 image2 会在第一个洞停住，成片被截短，而界面上写着「成功」。
            seq_dir = project / "_encode_seq"
            shutil.rmtree(str(seq_dir), ignore_errors=True)
            seq_dir.mkdir(parents=True, exist_ok=True)
            for i, n in enumerate(merged_names):
                d = seq_dir / ("%06d%s" % (i, frame_ext))
                try:
                    os.link(str(merged_dir / n), str(d))
                except OSError:
                    shutil.copyfile(str(merged_dir / n), str(d))
            # 注意别写成 "%06d%s" % (...)：那样 %06d 会被当场格式化掉，
            # image2 拿到的是一个具体文件名 → 只编出 1 帧
            pat = ("%06d" + frame_ext, 0)
            _proj_log("帧编号不连续，已按顺序重排 %d 帧到 %s" % (len(merged_names), seq_dir))
        pattern = str(Path(str(merged_dir if seq_dir is None else seq_dir)) / pat[0])
        t_enc = time.time()
        enc = _proj_encode(pattern, pat[1], float(cfg["fps"]), cfg["video"],
                           cfg["out_video"], len(merged_names))
        enc["seconds"] = round(time.time() - t_enc, 2)
        _proj_set(encode=enc)
        _proj_log("编码：%s" % enc.get("cmd"))
        if enc.get("code") != 0:
            raise _BadInput("编码失败（ffmpeg 退出码 %s）：%s"
                            % (enc.get("code"), (enc.get("tail") or "")[-500:]), 500)
    finally:
        if seq_dir is not None:
            shutil.rmtree(str(seq_dir), ignore_errors=True)

    el = time.time() - t0
    _proj_set(done=total, pct=100, stage="完成", msg="完成：%d 帧 → %s"
              % (len(merged_names), Path(str(cfg["out_video"])).name),
              elapsed=round(el, 1), eta_s=0)


def _proj_worker(cfg: dict):
    """线程入口：所有异常在这里落成终态，绝不让状态永远停在 running。"""
    try:
        _proj_run(cfg)
        with _PROJ_LOCK:
            _PROJ_JOB.update({"running": False, "ok": True, "phase": "done",
                              "cancelled": False, "finished_at": time.time(),
                              "stage": "完成",
                              "msg": "完成：%d 帧 → %s"
                                     % (int(_PROJ_JOB.get("merged_frames") or 0),
                                        Path(str(cfg["out_video"])).name)})
    except _ProjCancel:
        with _PROJ_LOCK:
            _PROJ_JOB.update({"running": False, "ok": False, "phase": "cancelled",
                              "cancelled": True, "stage": "已取消", "pct": 0,
                              "msg": "已取消", "finished_at": time.time(),
                              "error": "已取消（merged/ 里已写好的帧留着，下次接着跑）"})
    except _BadInput as e:
        with _PROJ_LOCK:
            _PROJ_JOB.update({"running": False, "ok": False, "phase": "failed",
                              "stage": "失败", "pct": 0, "msg": "导出失败",
                              "error": str(e.msg)[:1200],
                              "finished_at": time.time()})
    except Exception as e:                          # noqa: BLE001
        traceback.print_exc()
        with _PROJ_LOCK:
            log = list(_PROJ_JOB.get("log") or [])
            _PROJ_JOB.update({"running": False, "ok": False, "phase": "failed",
                              "stage": "失败", "pct": 0, "msg": "导出失败",
                              "error": "%s: %s" % (e.__class__.__name__, str(e)[:600]),
                              "error_detail": "\n".join(log[-20:])[:4000],
                              "finished_at": time.time()})


@router.post("/api/merge/export-project")
@_api
def merge_export_project(body: dict):
    """标准 DFL 项目 -> 换脸后的整帧图（merged/）+ 按原 fps/音轨编码成片。

    入参：
      project_dir   必填，<视频名>/ 目录（帧图 + aligned/）
      frames_dir    可选，帧目录（默认就是 project_dir）
      aligned_dir   可选，默认 <project_dir>/aligned
      merged_dir    可选，默认 <project_dir>/merged
      video         可选，原视频（fps / 分辨率 / 音轨都从它来）
      out_video     可选，默认 <项目目录>.mp4
      models / model / dfm / settings / xseg_onnx / detector / thresh
                    —— 和 /api/merge/export、/api/merge/preview 同一套协议
      force         可选，true = 忽略 merged 里已有的帧，全部重做
      quality       可选，jpg 帧的编码质量（默认 92，和单帧预览一致）
    """
    body = _as_dict(body)
    project = _norm_path(body.get("project_dir") or body.get("project"))
    if project is None:
        raise _BadInput("缺少 project_dir：要一个 DFL 项目目录（<视频名>/）")
    if not project.exists():
        raise _BadInput("项目目录不存在：%s" % project, 404)
    if not project.is_dir():
        raise _BadInput("project_dir 不是目录：%s" % project)

    frames_dir = _norm_path(body.get("frames_dir")) or project
    if not frames_dir.is_dir():
        raise _BadInput("帧目录不存在：%s" % frames_dir, 404)
    if not _scan_first(frames_dir, _PROJ_FRAME_EXT, 1):
        raise _BadInput("帧目录里没有帧图（.jpg/.png）：%s —— 先抽帧" % frames_dir)

    aligned = _norm_path(body.get("aligned_dir")) or (project / "aligned")
    if not aligned.is_dir():
        raise _BadInput("找不到对齐集 aligned/：%s —— 先跑一次人脸对齐" % aligned, 404)
    if not _scan_first(aligned, {".jpg", ".jpeg"}, 1):
        raise _BadInput("对齐集是空的：%s" % aligned)
    merged = _norm_path(body.get("merged_dir")) or (project / "merged")

    # ---- 模型：和 /export、/preview 完全同一套协议 ----
    models = _models_of(body)
    if models:
        dfm, model_name = Path(models[0]["dfm"]), models[0]["name"]
    else:
        dfm, model_name = _dfm_of(body)
    settings = _settings_val(body.get("settings"))
    cfg_main = _merge_settings_to_config(settings)
    detector = _detector_val(body.get("detector"))
    thresh = _float_val("thresh", body.get("thresh"), 0.0, 1.0, 0.5)
    xseg = _xseg_of(body)
    quality = _int_val("quality", body.get("quality"), 1, 100, _PREVIEW_QUALITY)
    force = bool(body.get("force"))

    # ---- 原视频 / fps / 分辨率：读不到就报错，不猜 ----
    video, why = _proj_source_video(body.get("video") or body.get("source_video"),
                                    project, frames_dir)
    if video is None:
        raise _BadInput("%s（项目：%s）" % (why, project), 422)
    try:
        v_frames, v_fps, v_w, v_h = _app_mod._video_probe(video)
    except Exception as e:                          # noqa: BLE001
        raise _BadInput("原视频解析不出来（ffmpeg/imageio）：%s（%s）"
                        % (video.name, str(e)[:160]), 422)
    fps_body = _float_val("fps", body.get("fps"), 1.0, 240.0, None)
    if fps_body is not None:
        fps = float(fps_body)
    elif v_fps and float(v_fps) > 0:
        fps = float(v_fps)
    else:
        raise _BadInput("读不出原视频的 fps：%s —— 不猜，先确认这个文件能正常解码"
                        % video, 422)

    # ---- 输出 ----
    out_raw = str(body.get("out_video") or body.get("out") or "").strip()
    out_video = _norm_path(out_raw) if out_raw else Path(str(project) + ".mp4")
    if out_video.suffix.lower() not in _VIDEO_EXT:
        out_video = out_video.with_suffix(".mp4")
    if out_video.is_dir():
        raise _BadInput("输出路径是个目录：%s（要 .mp4 文件）" % out_video)
    if _same_path(out_video, video):
        if out_raw:                     # 用户明确指到原片上：拒绝，别把素材覆盖了
            raise _BadInput("输出路径不能是原视频本身：%s" % out_video)
        # 默认名 <项目目录>.mp4 正好和原视频同名（把视频放在项目目录旁边时就是这样）：
        # 换个名字，绝不覆盖素材
        out_video = Path(str(project) + "_merged.mp4")
    try:
        out_video.parent.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        raise _BadInput("输出目录建不出来：%s（%s）" % (out_video.parent, e))

    cfg = {"project": str(project), "frames_dir": str(frames_dir),
           "aligned": str(aligned), "merged": str(merged),
           "video": str(video), "out_video": str(out_video),
           "fps": float(fps), "video_frames": int(v_frames or 0),
           "width": int(v_w or 0), "height": int(v_h or 0),
           "models": models, "dfm": str(dfm), "model": str(model_name),
           "config": cfg_main, "settings": settings, "xseg": xseg,
           "detector": detector, "thresh": float(thresh), "quality": int(quality),
           "force": force}

    with _PROJ_LOCK:
        if _PROJ_JOB.get("running"):
            raise _BadInput("已经在导出项目（%s）—— 等它结束或点取消"
                            % (_PROJ_JOB.get("project") or "?"), 409)
        _PROJ_JOB.update({
            "running": True, "ok": None, "phase": "running", "cancelled": False,
            "want_cancel": False, "error": "", "error_detail": "", "log": [],
            "stage": "准备", "pct": 0, "msg": "启动…", "elapsed": 0.0,
            "project": str(project), "frames_dir": str(frames_dir),
            "aligned": str(aligned), "merged": str(merged),
            "video": str(video), "out_video": str(out_video), "fps": float(fps),
            "width": int(v_w or 0), "height": int(v_h or 0),
            "video_frames": int(v_frames or 0), "total": 0, "done": 0,
            "swapped": 0, "copied": 0, "skipped": 0, "failed": 0,
            "frame_fps": 0.0, "eta_s": None, "started_at": time.time(),
            "finished_at": None, "force": force, "dfm": str(dfm),
            "model": str(model_name),
            "models": [{"name": m["name"], "dfm": m["dfm"],
                        "faces": list(m["faces"])} for m in (models or [])],
            "settings": settings, "detector": detector, "aligned_faces": 0,
            "merged_frames": 0, "warnings": [], "failed_frames": [], "missing": [],
            "missing_source": [], "encode": {},
        })
    threading.Thread(target=_proj_worker, args=(cfg,), daemon=True,
                     name="merge-export-project").start()

    snap = _proj_snap()
    snap.pop("want_cancel", None)
    snap.pop("log", None)
    return _ok(**snap)


@router.get("/api/merge/export-project/status")
@_api
def merge_export_project_status():
    """项目导出的进度。ok = 任务是否成功（和 /export/status 一个口径）。"""
    snap = _proj_snap()
    ok = snap.get("ok")
    cancelled = bool(snap.get("cancelled"))
    want = bool(snap.get("want_cancel"))
    running = bool(snap.get("running")) and not cancelled
    if running:
        phase = "running"
    elif ok is True:
        phase = "done"
    elif cancelled:
        phase = "cancelled"
    elif ok is False:
        phase = "failed"
    else:
        phase = "idle"
    if phase in ("failed", "cancelled"):
        snap["pct"] = 0
    started = snap.get("started_at")
    try:
        el = round(float((snap.get("finished_at") or time.time()) - started), 1) \
            if started else None
    except Exception:
        el = None
    snap.update({"phase": phase, "running": running, "success": ok,
                 "cancelling": bool(want and running), "elapsed": el,
                 "log": list(snap.get("log") or [])[-12:]})
    snap.pop("want_cancel", None)
    return snap


@router.post("/api/merge/export-project/cancel")
@_api
def merge_export_project_cancel():
    """取消项目导出。幂等：没在跑、重复点都回 ok=true。"""
    with _PROJ_LOCK:
        running = bool(_PROJ_JOB.get("running"))
        _PROJ_JOB["want_cancel"] = True
        if running:
            _PROJ_JOB["msg"] = "正在取消…"
    if not running:
        return _ok(cancelled=False, running=False, already=True,
                   reason="没有正在进行的项目导出")
    return _ok(cancelled=True, running=True, already=False, reason="已请求取消")


@router.post("/api/merge/reveal")
@_api
def merge_reveal(body: dict):
    """在资源管理器里定位一个文件/目录（「打开所在目录」）。"""
    body = _as_dict(body)
    p = _norm_path(body.get("path"))
    if p is None:
        raise _BadInput("缺少 path：要一个文件或目录")
    if not p.exists():
        raise _BadInput("路径不存在：%s" % p, 404)
    target = p if p.is_dir() else p.parent
    try:
        import sys as _sys
        flags = getattr(_app_mod, "_NO_WINDOW", 0)
        if _sys.platform.startswith("win"):
            if p.is_dir():
                subprocess.Popen(["explorer", "/e,", str(p)], creationflags=flags)
            else:
                subprocess.Popen(["explorer", "/select," + str(p)], creationflags=flags)
        elif _sys.platform == "darwin":
            subprocess.Popen(["open", "-R" if p.is_file() else "", str(p)])
        else:
            subprocess.Popen(["xdg-open", str(target)])
    except Exception as e:                            # noqa: BLE001
        raise _BadInput("打不开所在目录：%s" % str(e)[:160], 500)
    return _ok(path=str(p), dir=str(target))

