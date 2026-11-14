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
import sys
import tempfile
import threading
import time
import traceback
from email.utils import formatdate as _httpdate
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

from . import _eff as _eff
from . import _projplan as _projplan
from . import app as _app_mod
from ._face_group import clear_face_group_cache, register_cache_clearer
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
# 这份方案里存着「每张脸和各组均值的相似度」= embedding 派生的结果，所以它和组均值
# 属于同一类缓存：进程启动清一次、每次导出结束清一次（见 _face_group.clear_face_group_cache）。
try:
    register_cache_clearer(_AUTO_PLAN_CACHE.clear, size=_AUTO_PLAN_CACHE.__len__,
                           name="auto_plan")
except Exception:                                   # noqa: BLE001 - 登记失败不影响合成
    pass
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
    # 帧走裸字节块（PNG，无损）：六宫格要的是"脸发蒙时最直接的证据"，缩放/有损编码
    # 都会把证据毁掉，所以这里仍旧无损，只是不再 base64。
    _enc, _blob, _h, _w = _send_frame_bytes(frame, {"send_lossless": True})
    resp = _WORKER.call({
        "cmd": "face_debug",
        "dfm": str(dfm),
        "landmarks": np.asarray(lm, dtype=np.float64).round(2).tolist(),
        "box": list(box_wh),
        "config": dict(cfg),
        "xseg": xseg,
        "panel_size": int(panel_size),
    }, [_blob])
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
    """合成导出。两种输入，走两条链路：

      · 给了 video、又没有帧目录 -> **视频模式**：不要求先建 DFL 项目，
        从源视频逐帧解 -> 检测 -> 68 点 -> 判定模型 -> dfm -> 遮罩 -> 合成 ->
        写帧 -> ffmpeg（音轨 copy）出 mp4（见 _start_video_export）；
      · 给了帧目录 -> 老路：4.0 MergeStudio 子进程一口气出 mp4（下面的代码）。
    """
    body = _as_dict(body)
    # ---- 收到请求先回显一份（脱敏）：路径只留文件名、base64 换成长度 ----
    _eff.log_request("/api/merge/export", body)

    # ---- 0) 视频模式：video 有值 + 没有帧目录 -> 不需要项目 ----
    if (str(body.get("video") or body.get("source_video") or "").strip()
            and not str(body.get("input_dir") or body.get("project") or "").strip()):
        return _start_video_export(body)

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
                           "dfm": str(dfm),
                           # 上一轮可能是视频模式：视频专有的明细一起清掉，
                           # 否则 /export/status 会把上一轮的逐帧记录当成这一轮的
                           "mode": "project", "video": "", "total": 0,
                           "frames_done": 0, "frames": [], "stats": {},
                           "cache": {}, "encode": {},
                           "eff": None, "eff_warnings": [], "warnings": []})
    # ---- 生效配置（老项目模式：4.0 MergeStudio 子进程一口气出 mp4）----
    # 这条路的检测器由子进程选，但合成参数/模型/并发照样要对得上账。
    try:
        _pcfg = {"settings": settings, "config": _merge_settings_to_config(settings),
                 "detector": _DEFAULT_DETECTOR, "det_align": _DEFAULT_DETECTOR,
                 "mask_src": _mask_src(settings, settings.get("mask_mode")),
                 "dfm": str(dfm), "model": model_name, "single_models": [],
                 "library": "", "use_library": False, "chain": False, "fused": False,
                 "range": {}, "workers": workers, "batch": 0,
                 "jpeg_quality": _STAGE_JPEG_QUALITY}
        eff = _eff_publish("/api/merge/export", "project-modes", body, _pcfg)
        with _merge_job_lock:
            _merge_job.update(_eff_status_patch(eff))
    except Exception as e:                                  # noqa: BLE001
        print("[eff] 生成失败（不影响这一单）：%s: %s"
              % (e.__class__.__name__, str(e)[:200]), flush=True)
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

    mode = str(snap.get("mode") or "")
    resp = {"ok": ok, "success": ok, "phase": phase, "running": running,
            # active = worker 还活着（起单 → 收尾结束）。取消之后 running 立刻变 false
            # （看门狗靠这条），所以「还能不能再点一次导出」要看 active。
            # active = worker 线程还活着（从起单到收尾结束）。取消之后 running 立刻变
            # false（看门狗靠这条），所以「能不能再点一次导出」要看 active。
            "active": bool(snap.get("active")),
            "cancelled": cancelled, "stage": stage, "pct": pct,
            "pct_at_failure": snap.get("pct_at_failure"),
            "msg": str(snap.get("msg") or ""),
            "error": error, "error_detail": detail,
            "out": str(snap.get("out") or ""),
            "log": log_tail, "log_tail": log_tail[-12:],
            "started_at": started, "finished_at": finished, "elapsed": elapsed,
            "fps": snap.get("fps"), "workers": snap.get("workers"),
            "encoder": snap.get("encoder"), "hwaccel": snap.get("hwaccel"),
            "dfm": snap.get("dfm"),
            "mode": mode or "project",
            # 生效配置 + 一致性警告（进度页「当前生效」面板 / 顶部横幅读这两个键）
            "eff": snap.get("eff"), "eff_warnings": list(snap.get("eff_warnings") or []),
            "warnings": list(snap.get("warnings") or []),
            # 流水线实况（进度页那条 SVG 图的唯一数据源）：六个 key =
            # frames/det/dfm/xseg/compose/encode，每个给 ticks / inflight / lanes(+ms/batch)。
            # 没开过单时是空 dict —— 前端据此显示「等待流水线…」，不猜 lane 数。
            "stages": _STAGE_METER.snapshot()}
    if mode == "video":
        # 视频模式多给一段明细：逐帧记录（每张脸用了谁）+ 计数 + 结束时的缓存状态。
        # 老前端不认识这些字段，多给不会影响它（缺字段才是坑）。
        fused = bool(snap.get("fused"))
        stats = dict(snap.get("stats") or {})
        resp["vid"] = {"video": str(snap.get("video") or ""),
                       "total": int(snap.get("total") or 0),
                       "done": int(snap.get("frames_done") or 0),
                       "stats": stats,
                       "cache": dict(snap.get("cache") or {}),
                       "encode": dict(snap.get("encode") or {}),
                       "frames": list(snap.get("frames") or [])[-600:],
                       # 快速导出（融合流水线）专有：没有阶段概念，只有帧进度 + 帧预算
                       "fused": fused,
                       "resumable": (not fused),
                       "frame_budget": (snap.get("frame_budget")
                                        if snap.get("frame_budget") is not None
                                        else stats.get("frame_budget")),
                       "frame_fps": float(stats.get("fps") or 0.0),
                       "eta_s": stats.get("eta_s"),
                       "cache_dir": str(snap.get("cache_dir") or ""),
                       "per_stage_ms": dict(stats.get("per_stage_ms") or {}),
                       "stage_util_pct": dict(stats.get("stage_util_pct") or {}),
                       "pipeline": dict(stats.get("pipeline") or {})}
        # 顶层也给一份（前端不必再进 vid 里翻）
        resp["fused"] = fused
        resp["resumable"] = (not fused)
        resp["frame_budget"] = resp["vid"]["frame_budget"]
        resp["no_cache"] = bool(fused)
    # 显存计划（三条路共用一个形状）：训练/导出出问题时第一眼看它 ——
    # 里面有可用显存、需求拆项、选了哪一档、有没有降档、是不是在 CPU 上跑。
    resp["vram_plan"] = dict(snap.get("vram_plan") or _vram_last("快速导出") or {})
    resp["on_cpu"] = bool(resp["vram_plan"].get("cpu")) or bool(snap.get("on_cpu"))
    # 强制尝试撞上 OOM 的现场（有值 = 这一单是「确实不够、已回滚」，前端据此回到被拒态；
    # 里面带 summary/need/avail/reserve/free/own/other，和 400 的结构化字段同一份形状）
    resp["vram_oom"] = (dict(snap.get("vram_oom") or {}) or None)
    return resp


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
    had = _det_cached(key)            # 已经在父进程里 = 不再占新显存，也不用量
    _own_f0 = None if had else _vram_measure_begin()
    try:
        with _det_load_lock:              # 加载要几百毫秒，别让并发请求各建一份
            det = _get_detector(key)
    except HTTPException as e:
        raise _BadInput(str(e.detail), e.status_code)
    except Exception as e:
        raise _BadInput("检测器加载失败（%s）：%s" % (key, str(e)[:160]), 500)
    if _own_f0 is not None:
        _vram_measure_end("detector:" + key, _own_f0)
    return det


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
def _env_secs(name: str, default: float) -> float:
    """环境变量里的秒数（DFL_* 那套写法）：没给 / 不合法 / <=0 就用默认值。"""
    try:
        v = float(str(os.environ.get(name) or "").strip())
    except (TypeError, ValueError):
        return float(default)
    return v if v > 0 else float(default)


# 单帧「干活」上限（秒）—— 超了回 504。默认 15s；机器被别的活占满时可以调大
# （分段导出的合成阶段也走这条路，超时=这个阶段失败，cache 留着下次续跑）。
_PREVIEW_WORK_TIMEOUT = _env_secs("DFL_PREVIEW_WORK_TIMEOUT", 15.0)
_PREVIEW_LOAD_TIMEOUT = 120.0      # 冷启动 / 换模型加载的上限（单独计时）
_PREVIEW_BOOT_TIMEOUT = 90.0       # 进程 + 引擎 import 的上限
_PREVIEW_IDLE_EXIT = 600.0         # 空闲 10 分钟自动退出
_PREVIEW_SEND_QUALITY = 95         # 送给 worker 的帧 JPEG 质量
_PREVIEW_QUALITY = 92              # 回给界面的 JPEG 质量
_PREVIEW_HIT = {}                 # 上一帧的整份响应（只留 1 条：同一 t 连点两下时整份复用）
_PREVIEW_HIT_LOCK = threading.Lock()
# 上一次预览真正生效的配置：预览是高频入口（播放/拖动时每帧一次），[eff] 只在配置
# 变化时打一行；/api/merge/preview/status 读这里，界面也能对照。
_PREVIEW_EFF = {}
_PREVIEW_EFF_LOCK = threading.Lock()
_WORKER_MAX_MODELS = 3            # worker 里 DFM 的常驻上限（它自己报 max_models，这里是兜底）
_PREVIEW_MAX_MODELS = 8           # 一次请求最多声明几个模型（真正常驻的仍只有 3 个）
_PREVIEW_MAX_SWAPS = 8            # 单帧最多换几张脸（多模型模式下检测到的脸全换）
_PREVIEW_MAX_BATCH = 16           # 一条请求最多带几帧（worker 内部同样多个线程并行）

# 预览 worker 源码：写进系统 Temp（不进仓库），内容变了就换文件名重新落盘。
_PREVIEW_WORKER_SRC = r'''# -*- coding: utf-8 -*-
"""4.1 合成器 · 单帧预览 worker（常驻；父进程用 4.0 引擎的 python 起它）。

用法: python preview_worker.py <4.0引擎根>
协议: stdin/stdout 上的 JSON lines（ASCII 转义）。
"""
import base64
import json
import os
import sys
import threading
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor

# 协议通道：先 dup 一份 stdout，再把 fd 1 指到 stderr。
# 引擎里 torch / onnxruntime / facelib 会往 fd 1 打非 UTF-8 的中文日志，
# 混进 JSON lines 会让父进程整行解码直接炸。
#
# 通道是**二进制**的：一行 JSON 头（ASCII 转义）+ 紧随其后的裸字节块。
# 帧不再 base64 —— base64 把 3 字节抬成 4 字节，父进程还要 ~5ms/帧编解码，
# 而头和块都排在同一条管道上，不需要"可打印字符"。
# 头的 "blobs": [n1, n2, …] 是后面几块的字节数，两边都按它精确读满。
_OUT = os.fdopen(os.dup(1), "wb", buffering=1 << 20)
os.dup2(2, 1)
sys.stdout = sys.stderr
_IN = sys.stdin.buffer


def _say(obj, blobs=()):
    """一行 JSON 头 + 紧随其后的裸字节块（同一条管道、按顺序写、最后 flush 一次）。"""
    try:
        _OUT.write((json.dumps(obj, ensure_ascii=True) + "\n").encode("utf-8"))
        for b in blobs:
            if b:
                _OUT.write(b)
        _OUT.flush()
    except Exception:
        pass


def _read_exact(n):
    """从 stdin 精确读 n 字节 —— 管道短读是常态，必须循环读满。"""
    parts, got = [], 0
    while got < n:
        chunk = _IN.read(min(n - got, 1 << 20))
        if not chunk:
            raise EOFError("stdin 提前结束")
        parts.append(chunk)
        got += len(chunk)
    return parts[0] if len(parts) == 1 else b"".join(parts)


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
    # 父进程说「显存不够、用户开了 DFL_ALLOW_CPU_FALLBACK」时会带这个环境变量过来：
    # 本进程一律只建 CPU EP 的 session（不然 dfm 让出了显存、XSeg 又去抢）。
    _FORCE_CPU = str(os.environ.get("DFL_FORCE_CPU_EP") or "").strip().lower() in (
        "1", "true", "yes", "on")
    _PROVIDERS = (["CPUExecutionProvider"] if _FORCE_CPU
                  else ["CUDAExecutionProvider", "CPUExecutionProvider"])
    if _FORCE_CPU:
        print("[preview] ★ 当前在 CPU 上跑 ★（DFL_ALLOW_CPU_FALLBACK=1 + 显存不够）",
              file=sys.stderr, flush=True)
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
        sess = ort.InferenceSession(dfm, providers=list(_PROVIDERS))
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
              "providers": providers, "on_cpu": bool(_FORCE_CPU),
              "models": sorted(models),      # 父进程照这个对齐 _models（现在最多 3 个）
              "models_lru": list(models), "max_models": _MAX_MODELS,
              "load_ms": round((time.time() - t_boot) * 1000.0, 1)})
        return models[dfm]

    def _xseg(path, rid):
        if not path:
            return None
        if path in xsegs:
            return xsegs[path]
        xs = ort.InferenceSession(path, providers=list(_PROVIDERS))
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
          "max_models": _MAX_MODELS, "pid": os.getpid(), "on_cpu": bool(_FORCE_CPU),
          "ep": ("CPUExecutionProvider" if _FORCE_CPU else "auto")})

    # ------------------------------------------------------------
    # 一帧的活（批处理时每项都走这里，k 个线程并行跑 k 项）
    # ------------------------------------------------------------
    _MODELS_LOCK = threading.Lock()     # OrderedDict / xsegs 的增删只在锁里做

    def _load_locked(dfm, rid):
        with _MODELS_LOCK:
            return _load(dfm, rid)

    def _xseg_locked(path, rid):
        with _MODELS_LOCK:
            return _xseg(path, rid)

    def _sorted_models():
        with _MODELS_LOCK:
            return sorted(models)

    def _lru_snapshot():
        with _MODELS_LOCK:
            return list(models)

    def _decode_frame(it, blob):
        """帧字节 → BGR。enc=raw 是裸 BGR（父进程不再编 JPEG），jpg/png 走 imdecode。

        copy 一次：下面的合成要往这块内存里读很多次，frombuffer 给的是只读视图，
        留一份可写的省得某个分支（cv2 的 out 参数、.copy 之外的就地写）突然炸。
        """
        enc = str(it.get("enc") or "jpg").strip().lower()
        if enc == "raw":
            h = int(it.get("h") or 0)
            w = int(it.get("w") or 0)
            need = h * w * 3
            if h <= 0 or w <= 0:
                raise RuntimeError("raw 帧缺 h/w")
            if len(blob) < need:
                raise RuntimeError("raw 帧短了：%d < %d" % (len(blob), need))
            return np.frombuffer(blob, np.uint8, need).reshape(h, w, 3).copy()
        return cv2.imdecode(np.frombuffer(blob, np.uint8), cv2.IMREAD_COLOR)

    def _swap_item(it, blob, rid):
        """一项（一帧）：解码 → 换脸/遮罩/融合 → 出图。返回 (meta, 出图字节)。

        **像素路径和「一请求一帧」时逐行相同**：cfg 逐项落地、_load/_xseg 幂等、
        MergeMaskedFace 同一份调用、_paste_face 同一个分支、编码同一个函数。
        多线程只是让不同帧的这些步骤重叠，不改变任何一帧的计算。
        """
        t0 = time.time()
        frame = _decode_frame(it, blob)
        if frame is None or not int(getattr(frame, "size", 0)):
            raise RuntimeError("帧解码失败")

        c = it.get("config") or {}
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
        q = int(it.get("jpeg_quality") or 92)
        q = max(30, min(100, q))

        faces = it.get("faces") or None      # 多模型：[{dfm, landmarks, box}, …]
        if not faces:
            # ---- 老的单模型请求：一行行为都不改 ----
            dfm = str(it.get("dfm") or "")
            if not dfm or not os.path.isfile(dfm):
                raise RuntimeError("DFM 不存在: %s" % dfm)
            sess, inp_name, size, nchw = _load_locked(dfm, rid)
            xs = _xseg_locked(str(it.get("xseg") or ""), rid)
            lm = np.array(it.get("landmarks") or [], dtype=np.float32)
            if lm.ndim != 2 or lm.shape[1] < 2 or lm.shape[0] < 5:
                raise RuntimeError("68 点不合法: %r" % (getattr(lm, "shape", None),))
            # input_size 显式传进去，不靠模块级 _merger._model_input_size：
            # 批处理里两个线程同时写那个全局，另一帧就会按错的边长 warp
            # （多模型、输入尺寸不同时必现）。
            fo, _fmask = MergeMaskedFace(frame, lm, cfg,
                                         _predictor(sess, inp_name, size, nchw),
                                         xseg_256_extract_func=xs, input_size=size)
            if fo is None or not hasattr(fo, "shape"):
                raise RuntimeError("MergeMaskedFace 没返回图")
            out = fo
            if out.ndim != 3 or out.shape[2] != 3:
                raise RuntimeError("合成结果形状不对: %r" % (getattr(out, "shape", None),))
            if out.shape[:2] != frame.shape[:2]:
                if str(cfg.mode) == "debug":
                    pass                          # 调试视图本身就是输出
                else:
                    out = _paste_face(frame, out, it.get("box"), np, cv2)
            per = [{"dfm": dfm, "size": size}]
        else:
            # ---- 多模型：同一张帧上按顺序一张脸一张脸贴 ----
            # load 是这次「声明要用」的整组；先装进显存，免得到第 2 张脸才付加载时间。
            # 超过 _MAX_MODELS 的部分 _load 会自动 LRU 淘汰，这里不用管上限。
            for p in [str(x) for x in (it.get("load") or [])]:
                if p and os.path.isfile(p):
                    _load_locked(p, rid)          # 幂等：已经在显存里就原样返回
            xs = _xseg_locked(str(it.get("xseg") or ""), rid)
            out = frame
            per = []
            for k, fc in enumerate(faces):
                fc = fc or {}
                f_dfm = str(fc.get("dfm") or "")
                if not f_dfm or not os.path.isfile(f_dfm):
                    raise RuntimeError("第 %d 张脸的 DFM 不存在: %s" % (k + 1, f_dfm))
                sess_k, in_k, size_k, nchw_k = _load_locked(f_dfm, rid)
                lm_k = np.array(fc.get("landmarks") or [], dtype=np.float32)
                if lm_k.ndim != 2 or lm_k.shape[1] < 2 or lm_k.shape[0] < 5:
                    raise RuntimeError("第 %d 张脸的 68 点不合法: %r"
                                       % (k + 1, getattr(lm_k, "shape", None)))
                t_k = time.time()
                # 调试视图本身就是整帧输出，再叠第二张会把上一张盖掉 —— 每张都
                # 从原帧出发；正常模式贴的就是上一次的结果，多张脸能叠在一起。
                src = frame if str(cfg.mode) == "debug" else out
                fo, _fmask = MergeMaskedFace(src, lm_k, cfg,
                                             _predictor(sess_k, in_k, size_k, nchw_k),
                                             xseg_256_extract_func=xs, input_size=size_k)
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
        # 出图：默认 jpg（预览一直是这条）。fmt=raw 给「父进程自己编 PNG」用 ——
        # 1080x1920 的 PNG 在 worker 里要 132~145ms/帧，挪到父进程的落盘线程里
        # 就能和别的帧的换脸重叠，而且逐像素无损（PNG 是 0 损失，raw 也是）。
        fmt = str(it.get("fmt") or "jpg").strip().lower()
        if fmt == "raw":
            blob_out = np.ascontiguousarray(out).tobytes()
        elif fmt == "png":
            ok, buf = cv2.imencode(".png", out, [int(cv2.IMWRITE_PNG_COMPRESSION), 3])
            if not ok:
                raise RuntimeError("PNG 编码失败")
            blob_out = buf.tobytes()
        else:
            fmt = "jpg"
            ok, buf = cv2.imencode(".jpg", out, [int(cv2.IMWRITE_JPEG_QUALITY), q])
            if not ok:
                raise RuntimeError("JPEG 编码失败")
            blob_out = buf.tobytes()
        return ({"ok": True, "size": (per[0]["size"] if per else 256), "fmt": fmt,
                 "swap_ms": round((time.time() - t0) * 1000.0, 1),
                 "shape": [int(out.shape[0]), int(out.shape[1])],
                 "swaps": len(per), "per_face": per,
                 "models": _sorted_models(), "models_lru": _lru_snapshot(),
                 "max_models": _MAX_MODELS}, blob_out)

    def _safe_item(it, blob, rid):
        """一项炸了只影响它自己：错误折进 meta，其它项照跑。"""
        try:
            return _swap_item(it, blob, rid)
        except Exception as e:
            import traceback
            traceback.print_exc(file=sys.stderr)
            return ({"ok": False, "error": "%s: %s" % (e.__class__.__name__,
                                                       str(e)[:300])}, b"")

    _POOL = {"k": 0, "ex": None}

    def _pool_for(k):
        """按 k 建（或换）线程池：k 变了才重建，不然一直用同一组线程。"""
        k = max(1, min(16, int(k)))
        ex = _POOL.get("ex")
        if ex is not None and int(_POOL.get("k") or 0) == k:
            return ex
        if ex is not None:
            try:
                ex.shutdown(wait=True)
            except Exception:
                pass
        ex = ThreadPoolExecutor(max_workers=k, thread_name_prefix="preview-swap")
        _POOL.update({"k": k, "ex": ex})
        return ex

    def _run_items(items, blobs, k, rid):
        """k 项一起跑。返回 (metas, payload)，两边顺序都和 items 一致。

        失败的项 meta 里是 ok=False、payload 是 b""，父进程靠下标对回帧名，不会串。
        """
        if int(k) <= 1 or len(items) <= 1:
            out = [_safe_item(it, bl, rid) for it, bl in zip(items, blobs)]
        else:
            ex = _pool_for(k)
            futs = [ex.submit(_safe_item, it, bl, rid)
                    for it, bl in zip(items, blobs)]
            out = []
            for f in futs:
                try:
                    out.append(f.result())
                except Exception as e:              # 兜底：_safe_item 自己不抛
                    out.append(({"ok": False,
                                 "error": "%s: %s" % (e.__class__.__name__,
                                                      str(e)[:300])}, b""))
        return [o[0] for o in out], [o[1] for o in out]

    for line in _IN:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line.decode("utf-8"))
        except Exception as e:
            _say({"ok": False, "error": "bad json: %s" % e})
            continue
        rid = req.get("id")
        cmd = str(req.get("cmd") or "")
        if cmd == "exit":
            break
        # 请求可以带裸字节块：头的 "blobs": [n1, n2, …] 就是后面几块的字节数。
        # 先把块读满再干活 —— 读不满说明通道错位（父进程半路走了），直接收工。
        blobs = []
        try:
            for _n in (req.get("blobs") or []):
                _n = int(_n)
                blobs.append(_read_exact(_n) if _n > 0 else b"")
        except EOFError:
            break
        if cmd == "unload":
            # 手动放显存：{"cmd":"unload"} 全放；{"cmd":"unload","dfm":X} 只留 X
            keep = str(req.get("dfm") or "")
            if keep not in models:
                keep = ""                  # 要留的那个本来就没加载，那就全放
            gone = _unload(keep)
            _say({"id": rid, "ok": True, "unload": True, "unloaded": len(gone),
                  "keep": keep, "models": _sorted_models(),
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
                    _sess, _in, size, _nchw = _load_locked(p, rid)
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
                    _xs_fn = _xseg_locked(xp, rid)
                    _xs_fn(np.zeros((256, 256, 3), np.float32))   # 同上，先付一次
                except Exception as e:
                    xs_err = "%s: %s" % (e.__class__.__name__, str(e)[:200])
            head = per[0] if per else {"size": 256, "first_ms": 0.0}
            _say({"id": rid, "ok": True, "warm": True, "size": head["size"],
                  "xseg": bool(xp) and not xs_err, "xseg_error": xs_err,
                  "first_ms": head["first_ms"],
                  "warm_ms": round((time.time() - t_w) * 1000.0, 1),
                  "per_model": per, "count": len(per),
                  "pid": os.getpid(), "models": _sorted_models(),
                  "models_lru": _lru_snapshot(), "max_models": _MAX_MODELS})
            continue
        if cmd == "face_debug":
            # 单脸六宫格：只跑**这一张脸**的一次合成，把中间量拆成六张 PNG 发回去。
            # 走的就是普通的 MergeMaskedFace，只是多带一个 capture dict ——
            # capture 只写不读，一个像素都不改，所以第 6 格「merged」和用户在画面上
            # 看到的那张脸是同一份像素（脸发蒙时它是最直接的证据）。
            try:
                blob0 = blobs[0] if blobs else base64.b64decode(req.get("frame") or "")
                frame = cv2.imdecode(np.frombuffer(blob0, np.uint8), cv2.IMREAD_COLOR)
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
                sess, inp_name, size, nchw = _load_locked(dfm, rid)
                xs = _xseg_locked(str(req.get("xseg") or ""), rid)
                cap = {}
                MergeMaskedFace(frame, lm, cfg, _predictor(sess, inp_name, size, nchw),
                                xseg_256_extract_func=xs, capture=cap, input_size=size)
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

        # ---- 换脸（默认分支）：items 里可以是 1 帧，也可以是一次 k 帧 ----
        # 父进程把 k 帧合成**一条**请求发过来，这里用 k 个线程并行处理：GPU 前向、
        # 整帧 float32 的遮罩/合成、PNG 编码互相重叠，管道往返只付一次。
        items = req.get("items") or [req]
        if len(blobs) < len(items):
            blobs = list(blobs) + [b""] * (len(items) - len(blobs))
        k = max(1, min(16, int(req.get("threads") or len(items) or 1)))
        try:
            metas, payload = _run_items(items, blobs[:len(items)], k, rid)
            _say({"id": rid, "ok": True, "k": k,
                  "items": metas, "blobs": [len(b) for b in payload]}, payload)
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
        self._on_cpu = False           # worker 是不是被逼着跑 CPU（DFL_ALLOW_CPU_FALLBACK）
        self._vram_plan = {}           # 起 worker 时那份显存计划（status 里回显）
        # worker 现在占着多少显存（MB，**实测**：这一进程里加载模型前后空闲之差）。
        # 它是「本程序占用」的一半，也是回收能腾出来的那块（见 _vram_reclaim）。
        self._used_mb = 0.0

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
                "script": str(self._script or ""),
                "on_cpu": bool(self._on_cpu),
                "used_mb": round(float(self._used_mb or 0.0), 1),
                "cpu_note": ("★ 当前在 CPU 上跑 ★（显存不够 + DFL_ALLOW_CPU_FALLBACK=1，"
                             "预计慢约 17 倍）" if self._on_cpu else ""),
                "vram_plan": dict(self._vram_plan or {})}

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
        # 显存预算说「装不下但允许 CPU 回落」时，把这个决定带进子进程 ——
        # 子进程是独立解释器，父进程的开关传不过去。
        self._on_cpu = bool(_vram_cpu_on())
        if self._on_cpu:
            env["DFL_FORCE_CPU_EP"] = "1"
            print("[merge-preview] ★ 当前在 CPU 上跑 ★（显存不够 + %s=1）"
                  % _VRAM_ENV_CPU, flush=True)
        else:
            env.pop("DFL_FORCE_CPU_EP", None)
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
        self._used_mb = 0.0           # 新进程 = 显存从零开始，上一次的实测值作废
        self._vram_plan = dict(_vram_last("预览") or {})
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
        """读 worker 的 stdout：一行 JSON 头，后面可能跟若干裸字节块。

        头里的 "blobs": [n…] 是紧随其后的块长度（帧/出图不再 base64）。块读不满 =
        通道错位（worker 崩了/被杀）→ 塞 eof，让等着的人立刻醒，不会白等到超时。
        """
        f = proc.stdout
        try:
            while True:
                line = f.readline()
                if not line:
                    break
                s = line.decode("utf-8", "replace").strip()
                if not s:
                    continue
                try:
                    msg = json.loads(s)
                except Exception:
                    continue
                sizes = msg.get("blobs")
                if isinstance(sizes, list) and sizes:
                    blocks, bad = [], False
                    for n in sizes:
                        try:
                            n = int(n)
                        except (TypeError, ValueError):
                            bad = True
                            break
                        buf = bytearray()
                        while len(buf) < n:
                            chunk = f.read(n - len(buf))
                            if not chunk:
                                bad = True
                                break
                            buf += chunk
                        if bad:
                            break
                        blocks.append(bytes(buf))
                    if bad:
                        break
                    msg["_blobs"] = blocks
                q.put(msg)
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
        self._used_mb = 0.0           # 进程都收了，它占的那块显存跟着没了
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

    def call(self, payload: dict, blobs=(), work_timeout=None, load_timeout=None,
             on_stage=None) -> dict:
        """发一次请求给 worker 并等结果。超时/失联会杀掉 worker 再抛 _BadInput。

        payload 是 JSON 头，blobs 是紧跟其后的裸字节块（帧/出图）：头的 "blobs"
        写的是每块字节数，worker 按它精确读满 —— 帧不再 base64。
        """
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
            msg_in["blobs"] = [len(b) for b in (blobs or ())]
            # 这一发会不会**新建 session**（= 新占显存）？会的话量一下：worker 里的
            # 「模型 + 它自己的 CUDA context + ORT arena」是"本程序占用"的一半，
            # 也是回收能腾出来的那块。已经常驻的调用不量（每帧再跑两次 nvidia-smi 太亏）。
            _wanted = [str(p) for p in (payload.get("dfms") or []) if str(p or "")]
            if payload.get("dfm"):
                _wanted.append(str(payload.get("dfm")))
            for _it in (payload.get("items") or []):      # cmd=swap：这一轮要装哪些 dfm
                if isinstance(_it, dict):
                    if _it.get("dfm"):                    # 单模型预览：dfm 挂在 item 上
                        _wanted.append(str(_it.get("dfm")))
                    _wanted.extend(str(p) for p in (_it.get("load") or []) if str(p or ""))
            _will_load = any(p not in (self._models or {}) for p in _wanted)
            _own_f0 = _vram_measure_begin() if _will_load else None
            line = (json.dumps(msg_in) + "\n").encode("utf-8")   # ASCII 转义，安全过管道
            try:
                self._stdin.write(line)
                for b in (blobs or ()):              # 裸字节块：紧跟 JSON 头，一次 flush
                    if b:
                        self._stdin.write(b)
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
                if _own_f0 is not None:
                    # 实测：这一次加载让整卡空闲掉了多少（含 context / arena）。
                    # 累加成 max —— 同一份 worker 里换模型时前后两笔别互相抵消。
                    _f1, _t = _vram_free_mb(fresh=True)
                    if _f1 is not None:
                        _used = max(0.0, float(_own_f0) - float(_f1))
                        if _used > 0:
                            self._used_mb = round(max(float(self._used_mb or 0.0), _used), 1)
                            print("[merge-preview] 预览模型常驻实测 %.0f MB（空闲 %d → %d MB）"
                                  % (_used, int(_own_f0), int(_f1)), flush=True)
                if not self._models and msg.get("unload"):
                    self._used_mb = 0.0          # 全放掉了：账清零（下次加载重新量）
                return msg


_WORKER = _PreviewWorker()


# ------------------------------------------------------------
# 项目导出的 worker 进程池（对齐 iperov 的分片多进程）
#
# iperov 的导出 Stage4（export_pipeline._stage4_swap_faces_mp）是一个
# multiprocessing.Pool：主进程切片下发，每个子进程在 initializer 里建**自己的**
# dfm/xseg session（_phase_init_all），进程内直接读帧/换脸/融合/落盘，主进程只
# 顺着 imap_unordered 收回执。这里不重造那套代码 —— _PreviewWorker 本来就是
# 「一份 session + 一条 JSON 协议」的独立进程，把它复制成 N 份即可：像素路径
# 原样不动，多出来的只有并行度。
# ------------------------------------------------------------
# 默认 worker 数的两条线（**按模型体量**，不是按核数，也不是「一律 2」）
_PROJ_BIG_MB = 1024.0       # ≥1GB 权重：一份会话真跑起来 ≈ 3.4GB，2 个必撞墙
_PROJ_SMALL_MB = 600.0      # ≤600MB：两份会话（≈2.2GB/份）仍在 8GB 卡的预算里


def _proj_workers_default(model_mb=None) -> int:
    """项目模式默认并行度：**先按模型体量**（DFL_PROJ_WORKERS 可覆盖）。

    老口径是「一律 2」（按核数 // 4 封顶），而那正是用户这一单的病根：默认值**没有按
    模型体量收缩**，于是 1.53GB 的 384 模型也开 2 个 worker → 2×3470 ≈ 6.9GB + 检测/
    遮罩 → 整卡 7732/8151 = 94.9% → WDDM 换页 → swap p50 1338ms / **3.61 帧/秒**；
    同一台机器换成 0.5GB 的 416 模型就是 13.42 帧/秒（那组数字是这么来的，不是 worker
    数变了，是模型变了）。

    现在分三档（实测见 16-VRAM-BUDGET.md / 17-PROJ-BATCH.md 与本轮 bench）：
      · 大模型（≥1GB 权重）→ **1**：一份会话 ≈ 3.4GB，2 份就把 8GB 卡顶到 95%；
        单 worker × 4 线程实测 10.87 帧/秒 / 峰值 6160MB（75.6%），2×2 只有 4.79。
      · 小模型（≤600MB）→ **2**：一份 ≈ 2.2GB，2×2 实测 10.5~13.4 帧/秒 / 峰值 6430MB。
      · 中间档（600MB~1GB）→ 先要 **2**，由 _vram_plan 用「free + 实测会话峰值」
        （_vram_sess_peak_mb：标定值 max 上一单量到的值）砍到真装得下的个数 ——
        不在这里另造一套判据，砍完的顺序仍是「worker → 线程 → 拒绝」。
    """
    try:
        n = int(str(os.environ.get("DFL_PROJ_WORKERS") or "").strip() or 0)
    except (TypeError, ValueError):
        n = 0
    if n > 0:
        return max(1, min(16, n))
    mb = float(model_mb or 0.0)
    if mb >= _PROJ_BIG_MB:
        return 1
    if 0.0 < mb <= _PROJ_SMALL_MB:
        return 2
    cpu = os.cpu_count() or 8
    return max(1, min(2, cpu // 4))


def _proj_batch_default(workers=None) -> int:
    """一次请求带几帧 = worker 内部同时跑几帧：默认 **4 // workers**（DFL_PROJ_BATCH 可覆盖）。

    也就是说 1 个 worker → 4 线程、2 个 worker → 2 线程（老默认）、3+ 个 → 1 线程：
    同时在飞的帧数始终锁在 4 左右 —— 那是本机实测的甜点（8 帧在飞就换页了）。

    为什么单 worker 时必须是 4 而不是 2（384 模型、1080x1920 真项目、120 帧，本轮实测）：
        1×2  8.73 帧/秒  峰值 4612MB  swap p50 208ms   ← 线程不够，GPU 喂不饱
        1×4 10.87        峰值 6160MB  swap p50 362ms   ← 甜点
        1×8  4.44        峰值 7747MB  swap p50 1906ms  ← 换页，掉回 3.61 那一档
    见 %TEMP%/dfl_projplan/bench_prefix384.json。

    **2 个 worker 时仍取 2 线程**（= 老默认）：那个数是**跟着 ROI 化重定的**（见
    18-MERGE-ROI.md）—— 合成单帧从 288ms 砍到
    137ms、每帧的 CPU 活少了一半之后，「几个线程抢同一台机器的 CPU」开始亏。同一台
    机器、同一个真项目（1080x1920、416 的 dfm，proj_pool_accept 沙箱）重跑扫描表：

        1×1  8.21 帧/秒  2470MB        1×4 11.66  4626MB
        2×2 13.61        6881MB   ← 最优，显存还更低
        2×4 12.62        7701MB   （改造前的默认档）

    那一轮 2×2 最快（13.61 vs 2×4 的 12.62），后来的重跑则显示这两档**帧速同档**：
    2×2 稳在 13.5~14.5，2×4 在 13.2~16.5 之间大跳（4 个线程抢 CPU，时快时慢）；每次
    都成立的差别是**显存少 0.7~1.0GB**。默认取 2 买的就是这 1GB 余量（8GB 卡上 2×4
    已经贴到 7.5~7.7GB，再往上一档 2×8/3×4 直接 spill 到 2 帧/秒）。显存装不下时由
    _vram_plan 先砍**进程数**、再砍线程数（procs → batch，见 _vram_cands），这个值不兜底。
    """
    try:
        n = int(str(os.environ.get("DFL_PROJ_BATCH") or "").strip() or 0)
    except (TypeError, ValueError):
        n = 0
    if n > 0:
        return max(1, min(_PREVIEW_MAX_BATCH, n))
    w = max(1, int(workers or 2))
    return max(1, min(_PREVIEW_MAX_BATCH, 4 // w))


def _proj_plan_note(model_mb, plan: dict, label: str = "[project]") -> str:
    """一句话说清这一单的并行度决定（server.log 与 job.log 都写它）。

    形状（用户口径；括号里那截就是「为什么减」）：
        [project] 模型 1.53GB · 可用 6.49GB（空闲 6.83GB − 预留 0.34GB）
        · 一份会话 3.30GB · 计划 1 worker(s) × 4 线程
        （2 个要 7.0GB = 88% —— 95% 是换页线，已减）

    数字全部取这份计划自己的账（sess/aux/batch_mb/avail/total），不另算一套 —— 日志里
    那句话和拒绝消息 / status 里那几个数必须互相对得上。
    """
    plan = dict(plan or {})
    sess = float(plan.get("sess_mb") or 0.0)
    aux = float(plan.get("aux_mb") or 0.0)
    bmb = float(plan.get("batch_mb") or 0.0)
    procs = max(1, int(plan.get("procs") or 1))
    batch = max(1, int(plan.get("batch") or 1))
    req_p = max(1, int(plan.get("requested_procs") or procs))
    req_b = max(1, int(plan.get("requested_batch") or batch))
    total = plan.get("total_mb")
    avail = plan.get("avail_mb")
    params_m = plan.get("params_m")
    size_px = int(plan.get("size_px") or 0)
    head = "%s 参数量 %s" % (label, ("%.1fM" % float(params_m)) if params_m else "?")
    head += " · 权重 %s" % (("%.1fGB（%.0fMB）" % (float(model_mb or 0.0) / 1024.0,
                                                  float(model_mb or 0.0)))
                            if model_mb else "?")
    if size_px:
        head += " · 输入 %d" % size_px
    bits = [head]
    if avail is not None:
        avail_txt = "可用 %.2fGB" % (float(avail) / 1024.0)
        if plan.get("free_mb") is not None:
            avail_txt += ("（空闲 %.2fGB − 预留 %.2fGB）"
                          % (float(plan.get("free_mb") or 0.0) / 1024.0,
                             float(plan.get("reserve_mb") or 0.0) / 1024.0))
        bits.append(avail_txt)
    if sess > 0:
        bits.append("一份会话 %.2fGB" % (sess / 1024.0))
    bits.append("计划 %d worker(s) × %d 线程" % (procs, batch))
    why = []
    if req_p > procs:
        need = float(req_p) * (sess + bmb) + aux
        if total:
            why.append("%d 个要 %.2fGB = %d%%（95%% 是换页线，已减）"
                       % (req_p, need / 1024.0,
                          int(round(100.0 * need / float(total)))))
        else:
            why.append("%d 个要 %.2fGB > 可用 %.2fGB（已减）"
                       % (req_p, need / 1024.0, float(avail or 0.0) / 1024.0))
    if req_b > batch:
        why.append("线程 %d→%d（worker 数已经减到 1，仍不够）" % (req_b, batch))
    if not why:
        why.append("按预算装得下，没减")
    return " · ".join(bits) + "（" + "；".join(why) + "）"


def _proj_send_raw() -> bool:
    """传帧用裸 BGR（默认）；DFL_PROJ_SEND=jpg 退回 q95 JPEG。

    换的理由是**像素门过了**，不是"看着差不多"：同一帧、同参数对拍
    （%TEMP%/dfl_proj_pool，1080x1920 真项目帧）q95 JPEG 往返 vs 裸字节
    = mean 0.1506 LSB / 差异像素 12.60% / max 5 LSB，在 0.5 LSB 门内。
    代价只是管道上多走 6.2MB/帧（本机 ~25MB/s 的管道，8 帧/秒也就 50MB/s），
    省下的是父进程 20ms/帧的 JPEG 编码 + worker 6ms/帧的解码。
    """
    v = str(os.environ.get("DFL_PROJ_SEND") or "").strip().lower()
    return v not in ("jpg", "jpeg", "b64", "base64", "off", "0")


def _proj_out_raw() -> bool:
    """**只对 .png 帧生效**：worker 回传裸 BGR，PNG 改由父进程的落盘线程编。

    1080x1920 的 PNG（compression=3）在 worker 里要 132~145ms/帧（那一路单帧 475ms
    的三成）；它是**纯计算**，挪到父进程的落盘线程就能和别的帧的换脸重叠。像素无损
    （PNG 无损、raw 也是逐字节原样），实测「PNG 回传 vs 裸字节回传」逐像素最大差 0
    → 产物像素不变（文件字节会变：编码器换成了父进程那份 cv2）。

    **.jpg 帧不走这条**：那条路 worker 编的是 JPEG(q92)，合并产物就是 JPEG ——
    改写成无损 PNG 会**改变像素**（q95/q92 往返 mean ≈ 1 LSB），也违背「不改像素」。
    DFL_PROJ_OUT=png 可以让 .png 帧也交回 worker 编（产物字节和以前逐位相同）。
    """
    v = str(os.environ.get("DFL_PROJ_OUT") or "").strip().lower()
    return v not in ("png", "jpg", "jpeg", "worker", "off", "0")


class _PreviewWorkerPool:
    """N 个常驻 worker 进程，按「空/忙」借还；每个进程有自己的 dfm session。

    · 预热是**串行**的：同时起 N 个进程抢显存时谁先失败、失败几个都不确定；
      一个个来才能精确知道这台机器装得下几个（8GB 卡 + 1.6GB 的 384 模型通常只
      装得下 2 个），装不下的当场 close 掉，池子里剩下的才是真能用的。
    · 借还用 Queue：线程池里正好 N 个线程，所以不会有两个线程拿到同一个 worker。
    """

    def __init__(self, n: int):
        n = max(1, int(n))
        self._all = [_PreviewWorker() for _ in range(n)]
        self._free = queue.Queue()
        for w in self._all:
            self._free.put(w)
        self._closed = False

    def __len__(self) -> int:
        return len(self._all)

    def acquire(self):
        return self._free.get()

    def release(self, w):
        self._free.put(w)

    def prewarm(self, dfms, xseg) -> int:
        """逐个把 dfm/xseg 装进显存并空跑一次前向；返回真正起来的 worker 个数。"""
        ok = []
        for w in list(self._all):
            try:
                r = w.call({"cmd": "warm",
                            "dfms": [str(p) for p in (dfms or []) if str(p or "")],
                            "xseg": str(xseg or "")})
                if not r.get("ok"):
                    raise _BadInput(str(r.get("error") or "warm 失败"), 500)
                ok.append(w)
            except Exception as e:                      # noqa: BLE001
                print("[merge-proj] worker 预热失败（%s）：%s"
                      % (e.__class__.__name__, str(e)[:200]), flush=True)
                try:
                    w.close("预热失败")
                except Exception:                       # noqa: BLE001
                    pass
        self._all = ok
        self._free = queue.Queue()
        for w in ok:
            self._free.put(w)
        return len(ok)

    def add_one(self, dfms, xseg) -> bool:
        """再热一份会话（自适应标定要量 2 worker 那一档才加）。

        返回是否成功：装不下 / 加载失败就当场关掉，池子里保持原样 —— 标定本身不能因为
        「想量 2 worker」把这一单弄崩。
        """
        w = _PreviewWorker()
        try:
            r = w.call({"cmd": "warm",
                        "dfms": [str(p) for p in (dfms or []) if str(p or "")],
                        "xseg": str(xseg or "")})
            if not r.get("ok"):
                raise _BadInput(str(r.get("error") or "warm 失败"), 500)
        except Exception as e:                          # noqa: BLE001
            print("[merge-proj] 追加 worker 失败（%s）：%s"
                  % (e.__class__.__name__, str(e)[:200]), flush=True)
            try:
                w.close("追加失败")
            except Exception:                           # noqa: BLE001
                pass
            return False
        self._all.append(w)
        self._free.put(w)
        return True

    def retire(self, why="自适应降档") -> bool:
        """**关掉一个空闲 worker**（自适应降档 / 标定完不需要 2 份时用）。

        只在「手里没有它」的时候能调：从空闲队列 get_nowait 拿不到就返回 False（正忙），
        绝不打断在飞的那一批 —— 降档慢一帧无所谓，丢帧/失败帧才是事故。
        """
        if len(self._all) <= 1:
            return False
        try:
            w = self._free.get_nowait()
        except queue.Empty:
            return False
        self._all = [x for x in self._all if x is not w]
        try:
            w.close(why)
        except Exception:                               # noqa: BLE001
            pass
        return True

    def close(self, why="项目导出结束"):
        if self._closed:
            return
        self._closed = True
        for w in self._all:
            try:
                w.close(why)
            except Exception:                           # noqa: BLE001
                pass


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


def _send_frame_bytes(frame, job: dict):
    """整帧发给 worker 的裸字节 → (enc, blob, h, w)。**不再 base64。**

    三条路（默认全局共用一份开关，见 _proj_send_raw）：
      · job["send_raw"]     裸 BGR。项目模式导出走它（省 20ms 编码 + 6ms 解码，
                            像素差 mean 0.1506 LSB，见 _proj_send_raw 的注释）；
      · job["send_lossless"] PNG，无损（no_swap_faces 那条验收要的"脸区域差异为 0"
                            靠它；和以前完全同一条路，只是不再 base64）；
      · 其余（界面单帧预览） q95 JPEG —— **字节和以前一模一样**，只是不再 base64。
    """
    import cv2
    h, w = int(frame.shape[0]), int(frame.shape[1])
    if job.get("send_raw"):
        import numpy as np
        return "raw", np.ascontiguousarray(frame).tobytes(), h, w
    if job.get("send_lossless"):
        ok, buf = cv2.imencode(".png", frame, [int(cv2.IMWRITE_PNG_COMPRESSION), 1])
        if ok:
            return "png", buf.tobytes(), h, w
    ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY),
                                          _PREVIEW_SEND_QUALITY])
    if not ok:
        raise _BadInput("这一帧编不出 JPEG", 500)
    return "jpg", buf.tobytes(), h, w


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


# ------------------------------------------------------------
# 「一帧」的三个半边：准备（取帧/检测/68 点/组帧）、发车（一条协议）、收尾（结果）
#
# 拆成三块是为了批处理 —— 项目模式导出一次发 k 帧（worker 里 k 个线程并行），
# 主进程这边 k 帧的检测/68 点/组帧也能和别帧的换脸重叠。单帧预览
# （/api/merge/preview）走的还是同一条路，只是 k=1；准备/发车/收尾这三步本来就
# 是 _preview_job 里连着做的，所以预览的像素与响应字段一个字都没变。
# ------------------------------------------------------------
def _swap_desc(job: dict, frame, boxes, best, box_xyxy, box_wh, timings,
               stage_fn, state):
    """检测之后 → (desc, plain_note)。

    desc=None 表示这一帧不用去 worker（调用方用 _plain_frame 出结果，note 就是
    第二个返回值）。否则 desc 里带：
      item   要发给 worker 的那一份（帧字节另挂，见 _swap_blob）
      multi  True = 多模型（每张脸各用各的），False = 老的单模型
      plan   真贴上脸的那些（多模型用），skipped 没贴的 + 原因
      box    检测框 [x, y, w, h]，n_boxes 这一帧检出几张脸
      models 多模型的整组声明
    """
    import numpy as np

    models = job.get("models")
    fmt = str(job.get("fmt") or "jpg").strip().lower()
    q = int(job.get("quality") or _PREVIEW_QUALITY)
    common = {"xseg": str(job.get("xseg") or ""), "config": dict(job["config"]),
              "jpeg_quality": int(q), "fmt": fmt}
    no_swap = {int(x) for x in (job.get("no_swap") or ())}
    boxes = list(boxes or [])

    if not models:
        # ---- 单模型路径：主角那张脸被点名"不换"就整帧原样回去（不能假装换过）----
        best_i = next((i for i, b in enumerate(boxes) if b is best), None)
        if no_swap and best_i is not None and best_i in no_swap:
            return None, "第 %d 张脸在 no_swap_faces 里 —— 整帧保持原像素" % best_i
        stage_fn("landmarks")
        t_lm = time.time()
        lm68 = _frame_landmarks(frame, best, box_xyxy)
        timings["landmarks_ms"] = round((time.time() - t_lm) * 1000.0, 1)
        if lm68 is None:
            return None, "检出人脸但 68 点算不出来 —— 只回原帧"
        item = dict(common)
        item["dfm"] = str(job["dfm"])
        item["landmarks"] = np.asarray(lm68, dtype=np.float64).round(2).tolist()
        item["box"] = list(box_wh)
        return ({"item": item, "multi": False, "plan": [], "skipped": [],
                 "box": box_wh, "n_boxes": len(boxes), "models": None}, "")

    # ---- 多模型：检测到的每张脸都按 models 里的分配换，没分配的用第 0 个 ----
    # 一次 worker 请求把「这次要用的整组模型」先装进显存（worker 上限 3 个，超了
    # LRU 淘汰），然后在同一张帧上按检测顺序一张一张贴 —— 每张贴的都是上一次的
    # 结果，所以 3 张脸就是 3 次 MergeMaskedFace，最后只编一次图。
    # 人脸序号就是检测顺序（det.detect() 的返回顺序，从 0 起）；和 /api/merge/detect
    # 返回的 boxes 下标完全一致，前端拿它做脸-模型的勾选就对了。
    stage_fn("faces")
    t_lm = time.time()
    plan, skipped = [], []
    # 「不换脸」的脸：直接不参与合成 —— 既不算进 plan，也不兜给第 0 个模型。
    # 这条必须在 _model_for_face 之前判掉，否则"未声明"会被兜底成第 0 个模型。
    for i, b in enumerate(boxes[:_PREVIEW_MAX_SWAPS]):
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
            return None, "这一帧的脸都在 no_swap_faces 里 —— 整帧保持原像素"
        return None, "检出人脸但 68 点算不出来 —— 只回原帧"

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

    item = dict(common)
    item["faces"] = [{"dfm": p["dfm"], "landmarks": p["landmarks"], "box": p["box"]}
                     for p in plan]
    item["load"] = load
    return ({"item": item, "multi": True, "plan": plan, "skipped": skipped,
             "box": box_wh, "n_boxes": len(boxes), "models": models}, "")


def _swap_blob(job: dict, frame, desc: dict) -> dict:
    """把帧字节挂到 item 上（不进 JSON：几 MB 的裸字节走 blob 通道）。"""
    enc, blob, h, w = _send_frame_bytes(frame, job)
    item = dict(desc["item"])
    item["enc"] = enc
    if enc == "raw":
        item["h"], item["w"] = int(h), int(w)
    item["_blob"] = blob
    return item


def _worker_swap(worker, items, threads=1, state=None):
    """一次请求 k 帧（worker 内部用同样多个线程并行）。返回 (metas, imgs)。

    metas[i] / imgs[i] 对应 items[i]；某项失败只会让那一项 ok=False，其它项照旧
    —— 主进程按帧名单独记账。超时按「轮数」放大：k 帧分 ceil(k/线程) 轮跑完。
    """
    blobs = [it.pop("_blob", b"") or b"" for it in items]
    k = max(1, min(_PREVIEW_MAX_BATCH, int(threads or 1)))
    rounds = max(1, (len(items) + k - 1) // k)
    if state is not None:
        state["stage"] = "queued"      # worker 里排队的时间不算进单帧超时
    on_stage = ((lambda st: state.__setitem__("stage", st)) if state is not None else None)
    resp = worker.call({"cmd": "swap", "threads": k, "items": items}, blobs,
                       work_timeout=_PREVIEW_WORK_TIMEOUT * rounds,
                       on_stage=on_stage)
    if not resp.get("ok"):
        raise _BadInput("换脸失败：%s" % (resp.get("error") or "预览 worker 没给原因"), 500)
    metas = list(resp.get("items") or [])
    imgs = list(resp.get("_blobs") or [])
    while len(metas) < len(items):
        metas.append({"ok": False, "error": "worker 少了这一项的回执"})
    while len(imgs) < len(items):
        imgs.append(b"")
    return metas, imgs


def _preview_finish(desc: dict, resp: dict, img, timings, t0) -> dict:
    """worker 的回执 → 结果 dict（字段和改造前一致，前端一个字都不用改）。"""
    if not resp.get("ok"):
        raise _BadInput("换脸失败：%s" % (resp.get("error") or "预览 worker 没给原因"), 500)
    if not img:
        raise _BadInput("换脸结果为空", 500)
    timings["swap_ms"] = resp.get("swap_ms")
    timings["swap_wall_ms"] = round((time.time() - t0) * 1000.0, 1)

    out = {"jpg": img, "img": img, "box": desc.get("box"), "has_face": True,
           "no_faces": False, "swapped": True, "timings": timings,
           "model_size": resp.get("size"), "note": ""}
    if str(resp.get("fmt") or "").strip().lower() == "raw":
        # worker 回的是裸 BGR：PNG 由落盘那一侧编（见 _proj_out_raw）。
        shp = list(resp.get("shape") or [])
        if len(shp) >= 2:
            out["raw"] = img
            out["raw_hw"] = (int(shp[0]), int(shp[1]))
        out["jpg"] = out["img"] = None
    if not desc.get("multi"):
        return out

    skipped = desc.get("skipped") or []
    models = desc.get("models") or []
    plan = desc.get("plan") or []
    n_boxes = int(desc.get("n_boxes") or 0)
    note = ""
    if skipped:
        note = "%d 张脸没换（%s）" % (len(skipped), "；".join(
            "#%d %s" % (s["face"], s["why"]) for s in skipped))
    if n_boxes > _PREVIEW_MAX_SWAPS:
        note = (note + "；" if note else "") + ("只换前 %d 张脸（这一帧检出 %d 张）"
                                                % (_PREVIEW_MAX_SWAPS, n_boxes))
    out.update({"note": note,
                "faces": [{"face": p["face"], "model": p["name"], "dfm": p["dfm"],
                           "box": p["box"]} for p in plan],
                "models_used": _used_names(models, resp.get("models"),
                                           [p["dfm"] for p in plan]),
                "swaps": len(plan), "skipped": skipped})
    return out


def _preview_prepare(job: dict) -> dict:
    """前半段：取帧 → 检测 → 68 点 → 拼好要发给 worker 的那一份。

    返回 {"done": res}   = 这一帧不用去 worker（_plain_frame 已经出结果）；
    或 {"item", "desc", "timings"} = 发给 worker，回来走 _preview_finish。
    **不碰 worker**，所以它能和别的帧的换脸同时跑（项目导出的批处理靠这个）。
    """
    stop = job["stop"]
    state = job["state"]
    timings = {}

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
        return {"done": _plain_frame(job, frame, None, timings,
                                     note="这一帧没检出人脸",
                                     has_face=False, no_faces=True)}

    desc, plain_note = _swap_desc(job, frame, boxes, best, box_xyxy, box_wh,
                                  timings, _stage, state)
    if desc is None:
        return {"done": _plain_frame(job, frame, box_wh, timings, note=plain_note)}
    _stage("swap")
    return {"item": _swap_blob(job, frame, desc), "desc": desc, "timings": timings}


def _preview_job(job: dict) -> dict:
    """一次预览的重活：取帧 → 检测 → 68 点 → 换脸 + 遮罩 + 合成。

    job: {video, t, dfm, xseg, config, detector, thresh, stop, state[, models]}
    models 为空/缺省 = 老的单模型路径：只换「主角」那一张脸，行为一个字都不改；
    给了 models = 多模型（见 _swap_desc）。

    **预览只有一条路**：换脸/遮罩/融合全在 worker 里由 MergeMaskedFace 一次做完，
    返回的图直接在内存里，任何中间量都不落盘、不读 <视频>_cache/。导出走的是六阶段
    （pred/mask 落盘 + 纯 CPU compose_frame），两条路算法等价、代码路径不同，同一帧的
    像素差允许存在（无损对拍 mean 0.02~0.03；成片 vs 预览 mean ≈ 3.6~4 是编码往返）——
    详见上面那段说明。

    这里是「准备 + 发一帧 + 收尾」：和项目模式的批处理共用同一份 _swap_desc /
    _worker_swap / _preview_finish，只是 k=1。
    """
    prep = _preview_prepare(job)
    if prep.get("done") is not None:
        return prep["done"]
    t0 = time.time()
    metas, imgs = _worker_swap(job.get("worker") or _WORKER, [prep["item"]],
                               threads=1, state=job.get("state"))
    return _preview_finish(prep["desc"], metas[0], imgs[0], prep["timings"], t0)


# ------------------------------------------------------------# 单帧预览：一步式 MergeMaskedFace（不碰 <视频>_cache/，不调 stage 函数）
#
# 这里以前放过另一条路：预览也走 ⑤ 合成 —— 先读 <视频>_cache/pred|mask（有就复用），
# 没有就现场算 dfm + XSeg，再调 _merge_compose.compose_frame。它和导出的字节更接近，
# 但代价是：① 预览开始依赖 cache 存在与否（有 cache 和没 cache 出图不一样，还会读到
# 别人留下的半成品）；② 整帧 float32 的色彩迁移 + compose 比一步式慢一倍
# （1080×1920 实测 10 帧端到端：改前 497ms/帧 → 改后 230ms/帧，见
# app/tools/accept/preview_fast_accept.py）；③ 一次 UI 预览把六阶段的代码路径也拉进了请求
# 线程。整条已删掉，连同「从 <视频>_cache/frames 读帧」那一小段。
#
# 现在只有一条路：取帧 → worker → MergeMaskedFace（dfm 推理 + 遮罩 + 融合一次做完）
# → 内存里的 JPEG/PNG 原路返回。**不读 cache、不写 cache、不创建 cache 目录，
# 也不调用任何 _stage_* 函数**（本文件里 _stage_* 只服务六阶段导出）。
# worker 自己的临时目录（%TEMP%/dfl41_merge_preview，只放 worker 脚本）不在素材目录
# 或 cache 目录树里。
#
# 与导出的关系：算法等价、代码路径不同。
#   · 预览：merger.MergeMaskedFace —— 一张脸从头到尾一次算完；4.0 在 seamless 之后
#     还有一段「把结果 warp 回脸空间再做一次色彩迁移」，预览保留。
#   · 导出：六阶段（stage6_accept.py 验的那条）—— pred/mask 先落盘，合成是纯 CPU 的
#     _merge_compose.compose_frame，逐段对照 4.0 的 _phase_merge_worker。
# 所以同一帧「预览」和「导出」的像素差是**允许存在**的，实测（stage6_accept 的 d 段）：
#   · 预览 PNG vs 导出帧 PNG（同一帧、都无损）→ mean 0.02~0.03 / max 20~33：两条路的
#     融合只差边缘那几级灰度；
#   · 成片（crf16 yuv420p + 预览那侧 JPEG）vs 预览 → mean ≈ 3.6~4：这才是那 ~3.7 的来源，
#     是编码往返，不是算法。
# 不要为了「逐像素相同」再把预览接回 compose —— 那等于用一倍的耗时换一个用户看不出来
# 的差值，还顺手把预览绑在 cache 上。
# ------------------------------------------------------------



@router.post("/api/merge/preview")
@_api
def merge_preview(body: dict):
    """t 时刻那一帧 → 检测 → 换脸 → 遮罩 → 合成 → JPEG（data: URL）。

    body 多认一个可选的 models: [{name, dfm, faces}]（见 _models_of）：给了就是
    多模型，一帧里检测到的每张脸按分配换，没分配的用第 0 个；不给就是老的单模型
    路径，行为一个字都不改。

    链路固定为一步式：帧现场从视频取 → 检测 → 68 点 → worker 里 MergeMaskedFace
    （dfm 推理 + 遮罩 + 融合一次做完）→ 内存里的 JPEG。**不读也不写 <视频>_cache/**，
    不调用六阶段的任何 stage 函数；cache 在不在、里面有没有半成品都不影响这一枪。
    """
    t_start = time.time()
    body = _as_dict(body)
    # 预览是高频入口（播放/拖动时每帧一次）：内容没变就不重复打，免得淹掉别的日志
    _eff.log_request("/api/merge/preview", body, dedupe=True,
                     ignore=("t", "frame", "face", "idx", "n", "step"))

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

    # ---- 生效配置（预览：配置没变就不重复打，否则播放时每帧一行会淹掉日志）----
    _pcfg = {"settings": settings, "config": cfg,
             "detector": detector, "det_align": detector,
             "mask_src": _mask_src(settings, settings.get("mask_mode")),
             "dfm": str(dfm), "model": model_name, "single_models": models or [],
             "library": str(body.get("faces_project") or ""),
             "use_library": True, "chain": False, "fused": False,
             "range": {}, "workers": 0, "batch": 0,
             "jpeg_quality": _PREVIEW_QUALITY}
    eff = _eff_publish("/api/merge/preview", "preview", body, _pcfg, dedupe=True)
    with _PREVIEW_EFF_LOCK:
        _PREVIEW_EFF.clear()
        _PREVIEW_EFF.update(eff)

    # ---- 显存闸门：模型装不下就**明确拒绝**，别让 worker 半死不活 ----
    # 以前这里是「起 worker → 加载失败 → 报一句 worker 的 stderr 尾巴」，用户看到的
    # 是一条 CUDA/ORT 的英文异常；现在先算账：装不下就 400 + 「需要 X GB / 可用 Y GB /
    # 建议」，开了 DFL_ALLOW_CPU_FALLBACK 就走 CPU（worker 会收到 DFL_FORCE_CPU_EP=1）。
    plan = _preview_guard([m["dfm"] for m in (models or []) if m.get("dfm")] or [str(dfm)])

    note = ""
    if int(cfg.get("mask_mode") or 0) >= 6 and not xseg:
        note = ("mask_mode=%s 要 XSeg 模型，但找不到 —— 引擎会退化成整块矩形贴脸"
                % cfg.get("mask_mode"))
    if plan.get("cpu"):
        note = (note + "；" if note else "") + ("★ 当前在 CPU 上跑 ★（显存不够 + "
                                                "DFL_ALLOW_CPU_FALLBACK=1，预计慢约 17 倍）")
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
            hit_res["vram_plan"] = dict(plan)
            hit_res["on_cpu"] = bool(plan.get("cpu"))
            try:
                hit_res["worker"] = _WORKER.status()
            except Exception:
                pass
            return hit_res

    stop = threading.Event()
    state = {"stage": "start"}
    holder = {}
    done = threading.Event()

    # 帧一律现场从视频里取：这里**不再**看 <视频>_cache/frames 有没有 —— 预览不能
    # 因为「有没有 cache」出两张不同的图，更不能依赖别人留下的中间产物。
    def _run():
        try:
            job = {"video": video, "t": float(t), "dfm": dfm,
                   "models": models,
                   "xseg": xseg, "config": cfg,
                   "detector": detector, "thresh": float(thresh),
                   "stop": stop, "state": state,
                   "no_swap": _face_ids_val(
                       "no_swap_faces", body.get("no_swap_faces"))}
            holder["r"] = _preview_job(job)
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
                worker=_WORKER.status(), note=res.get("note") or note,
                vram_plan=dict(plan), on_cpu=bool(plan.get("cpu")),
                eff=eff, eff_warnings=list((eff or {}).get("warnings") or []))
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
    """预览 worker 的现状（排障用：起没起、加载过哪些模型、跑在哪个 provider 上）。

    另带一份 vram_plan：这次加载是按什么显存预算放行的、有没有降档 / 是不是在 CPU 上跑。
    """
    st = _WORKER.status()
    plan = dict(st.get("vram_plan") or _vram_last("预览") or {})
    with _PREVIEW_EFF_LOCK:
        eff = dict(_PREVIEW_EFF) if _PREVIEW_EFF else None
    return _ok(**dict(st, vram_plan=plan,
                      on_cpu=bool(st.get("on_cpu") or plan.get("cpu")),
                      eff=eff,
                      eff_warnings=list((eff or {}).get("warnings") or []),
                      warnings=[str(w.get("text") or w)
                                for w in ((eff or {}).get("warnings") or [])]))

@router.post("/api/merge/preview-frame")
@_api
def merge_preview_frame(body: dict):
    """一张**静态帧文件** -> 检测 -> 换脸 -> 遮罩 -> 合成 -> 图（data: URL）。

    和 /api/merge/preview 只差一点：帧不是按时刻从视频里取的，而是直接读一个
    图片文件（项目模式就是 <项目>/000000.jpg）。检测 / 68 点 / 换脸 / 遮罩 / 融合
    仍然是 _preview_job 那一份 —— 一步式 MergeMaskedFace，**不读 <视频>_cache/**。
    六阶段导出走的是另一条代码路径（pred/mask 落盘 + 纯 CPU compose_frame），
    算法等价但像素差允许存在（无损对拍 mean 0.02~0.03；成片 vs 预览 ≈ 3.6~4 是编码
    往返），别为了"同一份字节"把预览接回 compose。
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

    # 显存闸门（和 /preview 同一口径）：装不下就 400 + 数字，别去起 worker 试
    plan = _preview_guard([m["dfm"] for m in (models or []) if m.get("dfm")] or [str(dfm)])

    frame = _app_mod._imread_any(fp)
    if frame is None:
        raise _BadInput("这张帧图解不开：%s" % fp.name, 422)
    stop = threading.Event()
    state = {"stage": "start"}
    res = _preview_job({"frame": frame, "models": models, "dfm": dfm,
                        "xseg": xseg, "config": cfg, "detector": detector,
                        "thresh": float(thresh), "stop": stop, "state": state,
                        "fmt": fmt, "quality": int(quality),
                        # 帧就是这张图本身；不再认 frame_stem / cache_dir —— 预览不读
                        # <视频>_cache/ 里的 pred/mask/68 点（那是导出六阶段的中间产物）。
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
               settings=settings, vram_plan=dict(plan), on_cpu=bool(plan.get("cpu")))
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
    # 预热也要过显存闸门：装不下就在这里 400（带数字），别让后台线程去撞 CUDA
    plan = _preview_guard(dfms)

    hot, why = _warm_note_hot(dfms, names, det_name)
    ms = int(round((time.time() - t_req) * 1000.0))
    if hot:
        return _ok(warming=False, ready=True, ms=ms, cold=False,
                   model=names[0], dfm=dfms[0], detector=det_name,
                   worker_pid=_worker_pid(),
                   models=[{"name": n, "dfm": d} for n, d in zip(names, dfms)],
                   models_used=list(names), max_models=_WORKER_MAX_MODELS,
                   vram_plan=dict(plan), on_cpu=bool(plan.get("cpu")))

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
        for k in ("warnings", "eff_warnings", "failed_frames", "missing",
                  "missing_source", "models", "log"):
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


def _proj_finish(cfg: dict, project, frames, frame_ext, merged_dir, t0, total):
    """项目导出的收尾：对账（缺帧要说清楚）+ ffmpeg 编码成片。

    老路（_proj_run）和流水线（_proj_pipe.run）共用这一段 —— 成片的 ffmpeg
    命令、缺帧清单、进度口径只有一个实现，不会两条路各写一套再慢慢漂。
    """
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

def _proj_run(cfg: dict):
    # 默认走流水线（app/server/_proj_pipe.py · docs/21-PROJ-PIPELINE.md）：
    # 读 DFLJPG 的 68 点不检测 + dfm 攒批 + compose_frame（ROI/多线程）。
    # DFL_PROJ_PIPELINE=legacy 回到下面这条老路（worker 子进程 + MergeMaskedFace）。
    from . import _proj_pipe
    if _proj_pipe.enabled():
        return _proj_pipe.run(cfg)
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

    # ---- 并行度：N 个 worker 进程，每个进程一份自己的 dfm session ----
    # iperov 的 stage4 是 Pool(N) 按帧分片；这里对齐同一个语义。区别只在实现：
    # 那套在每个子进程的 initializer 里建 session（_phase_init_all），我们借还的
    # 是一个已经预热好的常驻 worker 进程（它本来就是「一份 session + 一条协议」）。
    # 只有 N=1 时走老路径（就在本进程外那一个 _WORKER 上），逐像素基准就是它。
    vplan = dict(cfg.get("vram_plan") or {})
    if vplan:
        for _ln in _vram_log_lines(vplan, "项目导出"):
            _proj_log(_ln)
        # 并行度那一行（模型多大 / 可用多少 / 计划几个 worker × 几线程 / 减了没有）
        if vplan.get("worker_note"):
            _proj_log(str(vplan["worker_note"]))
    want_workers = int(cfg.get("workers") or _proj_workers_default())
    # 显存计划给的是**上限**：真起不来的仍有 prewarm 兜底（那才是这台机器的实测），
    # 但没理由在计划已经说「只装得下 1 个」的时候还去起 4 个进程试运气。
    if vplan.get("procs"):
        want_workers = min(int(want_workers), max(1, int(vplan["procs"])))
        if int(vplan["procs"]) < int(vplan.get("requested_procs") or 0):
            _proj_warn("显存预算只装得下 %d 个 worker（请求 %d）→ 实际按 %d 开"
                       % (int(vplan["procs"]), int(vplan.get("requested_procs") or 0),
                          int(vplan["procs"])))
    want_workers = max(1, min(int(want_workers), int(total) or 1, 16))
    # 线程数（= 一条请求 k 帧）：默认 4 // workers（见 _proj_batch_default），
    # DFL_PROJ_BATCH / 请求体 batch 都能覆盖。显存不够时**先砍进程数、再砍它**
    # （vplan.procs → vplan.batch，_vram_cands 的阶梯），两样都到最小才拒绝。
    batch = int(cfg.get("batch") or _proj_batch_default(want_workers))
    batch = max(1, min(int(batch), _PREVIEW_MAX_BATCH, int(total) or 1))
    # 实际在跑的档位：自适应会在运行中改它（降线程不用重建 worker；降 worker 走
    # pool.retire —— 只关空闲的那个，绝不打断在飞的批）
    cur = {"workers": 0, "batch": int(batch)}

    # 一个 worker 也走池子：批量协议（一次请求 k 帧，worker 里 k 个线程）在 N=1 时
    # 照样有效；workers=1 + batch=1 就是"一帧一请求"的老语义（逐像素基准那一档）。
    pool = _PreviewWorkerPool(want_workers)
    _dfms = [m["dfm"] for m in (models or []) if m.get("dfm")] or [str(cfg["dfm"])]
    # 开跑前的整卡占用（含预热时的加载）—— 这一单结束要拿「峰值 − 这个数」反推一份会话多大
    base_used = None
    try:
        _bf, _bt, _bu = _vram_free_smi(fresh=True)
        base_used = _bu
    except Exception:                                   # noqa: BLE001
        base_used = None
    t_warm = time.time()
    n_ok = 0
    try:
        n_ok = pool.prewarm(_dfms, cfg.get("xseg"))
    except Exception as e:                          # noqa: BLE001
        _proj_warn("worker 预热整体失败：%s: %s"
                   % (e.__class__.__name__, str(e)[:160]))
    _proj_log("并行：请求 %d 个 worker 进程，显存装下 %d 个（预热 %.1fs）；每进程 %d 帧并行"
              % (want_workers, n_ok, time.time() - t_warm, batch))
    if n_ok <= 0:
        pool.close("预热全失败")
        raise _BadInput("项目导出：worker 进程一个都没起来 —— 见上一条日志", 500)
    if n_ok < want_workers:
        _proj_warn("%d 个 worker 没起来（显存不够 / 加载失败），实际并行 %d"
                   % (want_workers - n_ok, n_ok))
    n_workers = int(n_ok)
    cur["workers"] = int(n_workers)
    _proj_set(workers=int(n_workers), batch=int(batch))
    # ---- 显存采样：这一单整卡峰到多少（喂 _proj_record_sess_peak）----
    # 每 2 秒一次 nvidia-smi（80ms，无窗口），开销可以忽略；峰值只在正常跑完时才采纳。
    samp = {"peak": 0, "total": 0, "util": 0.0, "now": 0}
    samp_stop = threading.Event()

    def _vram_sampler():
        while not samp_stop.is_set():
            try:
                _f2, _t2, _u2 = _vram_free_smi(fresh=True)
                if _u2 is not None:
                    samp["peak"] = max(int(samp["peak"]), int(_u2))
                    samp["now"] = int(_u2)      # 降档判定看**当前**，不看历史最高
                    samp["total"] = int(_t2 or 0)
            except Exception:                           # noqa: BLE001
                pass
            # GPU 利用率：自适应「升档」要看它（显存宽松 + GPU 吃不满 = 线程不够）
            try:
                _r = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu",
                                     "--format=csv,noheader,nounits"],
                                    capture_output=True, text=True, timeout=6,
                                    creationflags=0x08000000)
                _line = (_r.stdout or "").strip().splitlines()[0]
                samp["util"] = float(_line)
            except Exception:                           # noqa: BLE001
                pass
            samp_stop.wait(2.0)

    threading.Thread(target=_vram_sampler, daemon=True,
                     name="proj-vram-sampler").start()
    # 帧速从**预热之后**开始算：预热是建 session / 加载模型，不是干活
    t_swap = time.time()
    swap_ms = []            # 每帧 worker 内部的换脸耗时（p50 进 status，扫描表要用）
    send_raw = _proj_send_raw()
    out_raw = _proj_out_raw()

    def _prep(name, bench=False):
        """一帧的「准备」：读帧 → 检测 → 68 点 → 拼好要发 worker 的那一份。

        bench=True：标定用 —— **不认断点续跑**（merged 里已经有也要重算），否则续跑时
        标定会量到一串 skip，帧速变成无穷大。

        返回 (kind, payload)：
          skip  续跑：merged 里已经有这一帧（payload=None）
          copy  没有对齐脸 → 源文件原样搬（payload=None）
          item  要换脸：payload={"item","desc","timings"}（item 里挂着裸帧字节）
          res   _plain_frame 的结果（有脸但没换，整帧原样回去）
          err   这一帧失败（payload={"frame","error"}）
        这一步**不碰 worker**，所以它能和别的帧的换脸同时跑（见 prep_pool）。
        """
        dst_p = merged_dir / name
        src_p = frames_dir / name
        # ---- 断点续跑：merged 里已经有非空文件就跳过（--force 时才重做）----
        if not bench and not force and dst_p.is_file() and dst_p.stat().st_size > 0:
            return "skip", None
        try:
            frame = _app_mod._imread_any(src_p)
        except Exception:
            frame = None
        if frame is None:
            return "err", {"frame": name, "error": "这一帧读不出来（损坏？）"}
        if not (face_map.get(name) or []):
            # 有没有脸：aligned 说了算。没有就原样搬（无损，也不白跑一次检测）
            return "copy", None
        # 产物编码格式和以前**完全一致**（.png 帧编 PNG、.jpg 帧编 JPEG q92）；
        # 只有 .png 帧在 out_raw 时改由父进程编（像素不变，见 _proj_out_raw）。
        fmt = "png" if Path(name).suffix.lower() == ".png" else "jpg"
        if out_raw and fmt == "png":
            fmt = "raw"
        try:
            prep = _preview_prepare({"frame": frame, "models": models,
                                     "dfm": cfg["dfm"], "xseg": cfg["xseg"],
                                     "config": cfg["config"],
                                     "detector": cfg["detector"],
                                     "thresh": float(cfg["thresh"]),
                                     "stop": stop, "state": state,
                                     "fmt": fmt, "quality": q,
                                     "no_face_copy": True, "send_raw": send_raw})
        except _BadInput as e:
            return "err", {"frame": name, "error": str(e.msg)[:200]}
        except Exception as e:                      # noqa: BLE001
            traceback.print_exc()
            return "err", {"frame": name,
                           "error": "%s: %s" % (e.__class__.__name__, str(e)[:160])}
        if prep.get("done") is not None:
            return "res", prep["done"]
        return "item", prep

    def _swap_batch(names_b, preps, worker, k=None):
        """一个批：等这批的准备结果 → 换脸合成**一次**请求（k 帧并行）→ 摊平。

        返回 [(name, kind, payload)]，顺序和 names_b 一致。
        k 不传就是当前档位的线程数（cur["batch"]）—— 标定要跑别的档位，所以它是参数。
        """
        prepared = []
        for i, fut in enumerate(preps):
            try:
                prepared.append(fut.result())
            except Exception as e:                  # noqa: BLE001
                prepared.append(("err", {"frame": names_b[i],
                                         "error": "%s: %s"
                                                  % (e.__class__.__name__,
                                                     str(e)[:160])}))
        out = [None] * len(names_b)
        items, slots = [], []
        for i, (kind, payload) in enumerate(prepared):
            if kind == "item":
                items.append(payload)
                slots.append(i)
            else:
                out[i] = (kind, payload)
        if items:
            t_send = time.time()
            metas, imgs = _worker_swap(worker, [p["item"] for p in items],
                                       threads=min(int(k or cur["batch"]), len(items)),
                                       state=state)
            for j, slot in enumerate(slots):
                ctx = items[j]
                meta = metas[j]
                nm = names_b[slot]
                if not meta.get("ok"):
                    out[slot] = ("err", {"frame": nm,
                                         "error": str(meta.get("error")
                                                      or "换脸失败")[:200]})
                    continue
                try:
                    res = _preview_finish(ctx["desc"], meta, imgs[j],
                                          ctx["timings"], t_send)
                except _BadInput as e:
                    out[slot] = ("err", {"frame": nm, "error": str(e.msg)[:200]})
                    continue
                except Exception as e:              # noqa: BLE001
                    traceback.print_exc()
                    out[slot] = ("err", {"frame": nm,
                                         "error": "%s: %s"
                                                  % (e.__class__.__name__,
                                                     str(e)[:160])})
                    continue
                out[slot] = ("res", res)
        return [(names_b[i], out[i][0], out[i][1]) for i in range(len(names_b))]

    def _write_frame(name, kind, payload):
        """落盘一帧（在落盘线程里跑 —— PNG 编码 130ms/帧，压在换脸线程里就白等了）。

        原子落盘、跳过/搬运、失败记账的口径和以前**一模一样**（_proj_put 没动）。
        """
        dst_p = merged_dir / name
        src_p = frames_dir / name
        try:
            if kind == "copy":
                _proj_put(dst_p, src=src_p)
                return "copy", None
            res = payload or {}
            if res.get("copy"):
                _proj_put(dst_p, src=src_p)
                return ("copy" if res.get("has_face") else "copy_noface"), None
            raw = res.get("raw")
            if raw is not None:
                # 只有 .png 帧会走到这儿（见 _proj_out_raw）：编成 PNG 落盘，
                # 像素和 worker 编的完全一样，省下的是 worker 里那 130ms。
                import numpy as np
                hh, ww = res.get("raw_hw") or (0, 0)
                img = np.frombuffer(raw, np.uint8).reshape(int(hh), int(ww), 3)
                _proj_put(dst_p, data=_img_bytes(img, "png", q))
            else:
                _proj_put(dst_p, data=res.get("img") or res.get("jpg"))
            return "res", None
        except Exception as e:                      # noqa: BLE001
            return "err", {"frame": name, "error": "落盘失败：%s" % str(e)[:120]}

    def _count(name, kind, err):
        """记账（主线程、按帧顺序调 —— 进度和续跑游标才和串行时一模一样）。"""
        nonlocal skipped, copied, swapped
        if kind == "skip":
            skipped += 1
        elif kind == "err":
            failed_frames.append(err or {"frame": name, "error": "未知错误"})
        elif kind == "copy":
            copied += 1
        elif kind == "copy_noface":
            copied += 1
            if len(face_miss) < _PROJ_MAX_LIST:
                face_miss.append(name)      # 有对齐脸但检测不到脸：记下来，别悄悄过
        else:
            swapped += 1

    # ============================================================
    # 自适应档位：标定 → 选档 → 运行中升降档（用户口径「项目模式要有智能调度」）
    # ------------------------------------------------------------
    # 静态规则（_proj_workers_default / _proj_batch_default）只是**没有实测时**的兜底：
    # 同一张 8GB 卡，「4 帧摊在 1 份会话还是 2 份上」是 77% vs 95%（换页）—— 这件事跟
    # 权重只是相关，不是因果（还看 arena 涨多少、帧多大、桌面占了几个 GB）。所以这里
    # 现场用 SAMPLE_FRAMES 帧把候选 (workers × batch) 真跑一遍，量帧速 / 显存峰 /
    # swap p50，再按 _projplan.choose 的三条规则选；跑起来之后每 ADAPT_EVERY 帧复查，
    # 显存或 swap 恶化就降档（降线程 → 降 worker），长期空闲且显存宽松再升回来。
    # ============================================================
    adapt = dict(cfg.get("adaptive") or {})
    ad_state = {"events": [], "base_p50": None, "at": 0, "downs": 0, "ups": 0}
    adapt_every = max(20, int(os.environ.get("DFL_PROJ_ADAPT_EVERY") or 200))
    # 降档的显存阈值（默认 88%）；验收/排障可以调低，让「运行中自适应」这条路径真被触发
    adapt_pct = float(os.environ.get("DFL_PROJ_ADAPT_PCT") or 88.0)
    best_batch = max(1, int(cur["batch"]))

    def _adapt_snap() -> dict:
        return {"src": str(adapt.get("src") or "off"), "pick": adapt.get("pick"),
                "rows": adapt.get("rows"), "note": str(adapt.get("note") or ""),
                "cap": adapt.get("cap"), "events": list(ad_state.get("events") or []),
                "cur": {"workers": len(pool), "batch": int(cur["batch"])}}

    def _measure(names, nw, nb):
        """量一档（nw 个 worker、每批 nb 帧）→ {fps, swap_p50, peak_mb, ...}。"""
        if not names or len(pool) < int(nw):
            return None
        stop_s = threading.Event()
        peak = {"mb": 0, "total": 0}

        def _samp():
            while not stop_s.is_set():
                try:
                    _f, _t, _u = _vram_free_smi(fresh=True)
                    if _u is not None:
                        peak["mb"] = max(int(peak["mb"]), int(_u))
                        peak["total"] = int(_t or peak["total"])
                except Exception:                       # noqa: BLE001
                    pass
                stop_s.wait(0.15)

        threading.Thread(target=_samp, daemon=True, name="proj-bench-vram").start()

        rounds = []

        def _slot(names_b, preps):
            w = pool.acquire()
            t_b = time.time()
            try:
                return _swap_batch(names_b, preps, w, k=int(nb))
            finally:
                rounds.append((t_b, time.time(), len(names_b)))
                pool.release(w)

        chunks = [list(names[i:i + int(nb)]) for i in range(0, len(names), int(nb))]
        got_ms, bad = [], 0
        from concurrent.futures import ThreadPoolExecutor as _TPE
        t0 = time.time()
        with _TPE(max_workers=max(2, int(nb) * 2),
                  thread_name_prefix="proj-bench-prep") as prep_ex, \
                _TPE(max_workers=max(1, int(nw)), thread_name_prefix="proj-bench") as ex:
            futs = []
            for ch in chunks:
                preps = [prep_ex.submit(_prep, nm, True) for nm in ch]
                futs.append((ch, ex.submit(_slot, ch, preps)))
            for ch, f in futs:
                try:
                    landed = f.result()
                except Exception as e:                  # noqa: BLE001
                    bad += len(ch)
                    _proj_warn("标定 %d×%d 有一批失败：%s: %s"
                               % (nw, nb, e.__class__.__name__, str(e)[:120]))
                    continue
                for (_nm, kind, payload) in landed:
                    if kind == "err":
                        bad += 1
                    elif kind == "res" and payload:
                        _ms = (payload.get("timings") or {}).get("swap_ms")
                        if _ms:
                            got_ms.append(float(_ms))
        stop_s.set()
        # 帧速按**稳态**算：头 nw 批还在热 arena / 填流水线（2 worker 那档尤其明显，
        # 12 帧里它能把 13.4 量成 10.0），所以丢掉每路的第一批，只数后面的。
        rounds.sort()
        skip = int(nw) if len(rounds) > int(nw) + 1 else 0
        used = rounds[skip:] or rounds
        try:
            span = max(0.001, max(r[1] for r in used) - min(r[0] for r in used))
            cnt = sum(int(r[2]) for r in used)
        except ValueError:                              # 一轮都没跑成（全失败）
            span, cnt = max(0.001, time.time() - t0), 0
        vals = sorted(got_ms)
        return {"workers": int(nw), "batch": int(nb),
                "fps": round(cnt / span, 2) if cnt else 0.0,
                "swap_p50": (round(vals[len(vals) // 2], 1) if vals else None),
                "peak_mb": int(peak["mb"] or 0), "total_mb": int(peak["total"] or 0),
                "frames": int(cnt) or len(names), "bad": int(bad), "sec": round(span, 2)}

    def _calibrate():
        """没有实测时：8~12 帧 × 候选档位 → 选一档，并把结果写进 _proj_bench.json。"""
        nonlocal batch, best_batch
        # 取帧不能只看「aligned 里有脸」：本片就有 16 帧有对齐脸但检测器没检出（按原帧
        # 落盘），拿它们标定会量到「原样搬运」的 110 帧/秒 —— 那不是换脸的帧速。所以先
        # 用 _prep 探一遍，只收真能换脸（kind=item）的帧。
        want = int(_projplan.SAMPLE_FRAMES)
        names = []
        for _nm in [n for n in frames if (face_map.get(n) or [])]:
            if len(names) >= want:
                break
            try:
                _kind, _pl = _prep(_nm, True)
            except Exception:                       # noqa: BLE001
                continue
            if _kind == "item":
                names.append(_nm)
        if len(names) < 4:
            _proj_log("标定：能换脸的帧只找到 %d 张（要 4 张以上）→ 不标定"
                      % len(names))
            return None
        size_px = int(vplan.get("size_px") or _dfm_size_hint(_dfms[0]) or 0)
        mmb = float(vplan.get("model_mb") or 0.0)
        sess_mb = float(vplan.get("sess_mb") or 0.0)
        cands = []
        for (nw, nb) in _projplan.candidates(2, _PREVIEW_MAX_BATCH):
            fit, est_mb = True, None
            try:
                cand = _vram_plan(mmb, {"mode": "procs", "max_procs": int(nw),
                                        "batch": int(nb), "aux": True,
                                        "sess_mb": sess_mb, "size_px": size_px,
                                        "sess_src": "标定：用本单的会话峰值",
                                        "label": "标定 %d×%d" % (nw, nb)})
                for st in (cand.get("steps") or []):
                    if int(st.get("procs") or 0) == int(nw) and int(st.get("batch") or 0) == int(nb):
                        est_mb = float(st.get("need_mb") or 0.0)
                        break
                plan_ok = (bool(cand.get("ok"))
                           and int(cand.get("procs") or 1) >= int(nw)
                           and int(cand.get("batch") or 1) >= int(nb))
                # 预算是**估算**（会话峰值 + 批台阶），实测常常比它低几百 MB ——
                # 416 就是被估到 5781 而真跑只有 4906（6503 整卡）。所以这里不用
                # 「预算说不行就不量」，而是「估算的整卡占用没超过换页线就量一量，
                # 量完按**实测峰值**判」。真装不下由 add_one 失败兜住（当场 close）。
                tot_mb = float(vplan.get("total_mb") or 0.0)
                base_mb = float(base_used or 0.0)
                fits_hard = (est_mb is not None and tot_mb > 0
                             and (est_mb + base_mb) <= 100.0 * _projplan.HARD * tot_mb / 100.0)
                fit = bool(plan_ok or fits_hard)
                if not fit:
                    _proj_log("标定跳过 %d×%d：估算整卡 %.0fMB > 换页线 %.0f%%（%s）"
                              % (nw, nb, float(est_mb or 0) + base_mb,
                                 100.0 * _projplan.HARD,
                                 str(cand.get("reason") or "")[:80]))
                elif not plan_ok:
                    _proj_log("标定 %d×%d：预算说装不下，但估算整卡 %.0fMB 在换页线内 → "
                              "按实测判（真起不来会当场关掉）"
                              % (nw, nb, float(est_mb or 0) + base_mb))
            except Exception:                           # noqa: BLE001
                fit = True
            if fit:
                cands.append((nw, nb))
        rows = {}
        _proj_log("标定：用 %d 帧量 %s 档（每档跑完就记帧速 / 显存峰 / swap p50）"
                  % (len(names), "、".join("%d×%d" % c for c in cands) or "无"))
        for (nw, nb) in cands:
            while len(pool) > int(nw):
                if not pool.retire("标定：先量 worker 少的档"):
                    break
            while len(pool) < int(nw):
                if not pool.add_one(_dfms, cfg.get("xseg")):
                    break
            cur["workers"] = len(pool)
            if len(pool) < int(nw):
                _proj_log("标定跳过 %d×%d：第 %d 个 worker 起不来（显存不够）"
                          % (nw, nb, int(nw)))
                continue
            r = _measure(names, nw, nb)
            if not r or r.get("bad") or float(r.get("fps") or 0) <= 0:
                _proj_log("标定 %d×%d 没量成（失败 %s / %.2f 帧/秒）"
                          % (nw, nb, (r or {}).get("bad"), float((r or {}).get("fps") or 0)))
                continue
            rows["%dx%d" % (nw, nb)] = r
            _proj_log("标定 %d×%d：%.2f 帧/秒 · swap p50 %sms · 显存峰 %dMB"
                      % (nw, nb, r["fps"], r["swap_p50"], r["peak_mb"]))
        if not rows:
            return None
        tot = int(rows[sorted(rows)[0]].get("total_mb") or 0)
        res = _projplan.choose(rows, tot, base_used)
        line = _projplan.note(rows, res)
        _proj_log(line)
        pick = res.get("pick") or None
        if pick:
            cur["batch"] = int(pick["batch"])
            batch = int(pick["batch"])
            best_batch = max(1, int(cur["batch"]))
            while len(pool) > int(pick["workers"]):
                if not pool.retire("标定：选中的档位只要 %d 个 worker" % int(pick["workers"])):
                    break
            cur["workers"] = len(pool)
            _proj_set(workers=int(cur["workers"]), batch=int(cur["batch"]))
        adapt.update({"src": "bench", "rows": res.get("rows"), "pick": pick,
                      "rejected": res.get("rejected"), "note": line,
                      "cap": res.get("cap")})
        cfg["adaptive"] = dict(adapt)
        _proj_set(adaptive=_adapt_snap())
        # 标定本身不干活：帧速和 swap p50 都要从**正式开跑**那一拍重新起算，
        # 否则 8~12 帧的标定时间会被摊进这一单的帧速里（实测差 8.3 vs 10.1）。
        nonlocal t_swap
        t_swap = time.time()
        del swap_ms[:]
        ad_state["at"] = 0
        ad_state["base_p50"] = None
        if adapt.get("key"):
            # 每一档同时记 delta（本程序那一块）：缓存命中时基线可能已经变了
            # （用户多开了个浏览器就 +800MB），用 delta + 当时的基线才算得准。
            rows_out = {}
            for _k, _r in rows.items():
                _rr = dict(_r)
                if base_used:
                    _rr["delta_mb"] = max(0, int(_rr.get("peak_mb") or 0) - int(base_used))
                rows_out[_k] = _rr
            _projplan.put_row(WORKSPACE, adapt["key"],
                              {"v": _projplan.BENCH_VER, "ts": time.time(),
                               "model": str(cfg.get("model") or ""),
                               "size_px": size_px, "base_mb": base_used,
                               "rows": rows_out, "sample_frames": len(names),
                               "token": "bench"})
            _proj_log("标定结果已存 %s（下次同一份模型直接用，不再花这 %d 帧）"
                      % (_projplan.path_of(WORKSPACE).name, len(names)))
        return pick

    def _adapt_check():
        """运行中复查（每 adapt_every 帧）：显存 >88% 或 swap p50 恶化 >50% → 降一档。"""
        if done - int(ad_state.get("at") or 0) < adapt_every:
            return
        ad_state["at"] = done
        tot = int(samp.get("total") or 0)
        if not tot:
            return
        # 用「当前占用」判降档：peak 是单调的，一旦摸到过 89% 就永远 89% → 会一路
        # 从 4 线程掉到 1（棘轮 bug）。peak 只用来记账和写报告。
        pct = 100.0 * float(samp.get("now") or samp.get("peak") or 0) / tot
        recent = sorted(x for x in swap_ms[-adapt_every:] if x)
        p50 = recent[len(recent) // 2] if recent else None
        if ad_state.get("base_p50") is None and p50:
            ad_state["base_p50"] = p50
        base = ad_state.get("base_p50") or p50
        util = float(samp.get("util") or 0.0)
        why = ""
        if pct > adapt_pct:
            why = "显存 %.0f%% > %.0f%%" % (pct, adapt_pct)
        elif base and p50 and p50 > 1.5 * float(base):
            why = ("swap p50 %.0fms 恶化到 %.1f×（基线 %.0fms）"
                   % (p50, p50 / float(base), float(base)))
        if why:
            msg = ""
            if int(cur["batch"]) > 1:
                cur["batch"] = int(cur["batch"]) - 1
                _proj_set(batch=int(cur["batch"]))
                msg = "[project] 自适应降档：%s → 线程 %d" % (why, int(cur["batch"]))
            elif len(pool) > 1 and pool.retire("自适应降档"):
                cur["workers"] = len(pool)
                _proj_set(workers=int(cur["workers"]))
                msg = ("[project] 自适应降档：%s → worker %d（线程已经是 1）"
                       % (why, int(cur["workers"])))
            if msg:
                ad_state["downs"] = int(ad_state.get("downs") or 0) + 1
                ad_state["events"].append({"frame": int(done), "kind": "down", "why": why,
                                           "workers": len(pool),
                                           "batch": int(cur["batch"])})
                ad_state["base_p50"] = p50 or base
                _proj_log(msg)
                _proj_set(adaptive=_adapt_snap())
            return
        # 升档：GPU 长期吃不满（<50%）且显存宽松（<70%）→ 线程 +1（最多回到实测最优）
        if (pct < 70.0 and util and util < 50.0 and int(cur["batch"]) < int(best_batch)
                and int(ad_state.get("ups") or 0) < 2):
            cur["batch"] = int(cur["batch"]) + 1
            ad_state["ups"] = int(ad_state.get("ups") or 0) + 1
            ad_state["events"].append({"frame": int(done), "kind": "up",
                                       "why": "GPU %.0f%% / 显存 %.0f%%" % (util, pct),
                                       "workers": len(pool), "batch": int(cur["batch"])})
            _proj_log("[project] 自适应升档：GPU %.0f%% 没吃饱 / 显存 %.0f%% → 线程 %d"
                      % (util, pct, int(cur["batch"])))
            _proj_set(batch=int(cur["batch"]), adaptive=_adapt_snap())

    def _tick():
        """进度（只在主线程跑，和以前串行时的口径一模一样）。

        帧速用**预热之后**的时钟算（预热是加载模型，不是干活）；顺带把每帧 worker
        内部的换脸耗时收成 p50 —— 扫描「进程数 × 线程数」那张表要的就是它。
        """
        nonlocal done
        done += 1
        el = max(0.001, time.time() - t0)
        el_swap = max(0.001, time.time() - t_swap)
        # 帧速只按真正干过活的帧算：续跑跳过的不算，否则「全跳过」会报出 3 万帧/秒
        rate = (swapped + copied + len(failed_frames)) / el_swap
        vals = sorted(x for x in swap_ms if x)
        p50 = round(vals[len(vals) // 2], 1) if vals else None
        _proj_set(done=done, swapped=swapped, copied=copied, skipped=skipped,
                  failed=len(failed_frames), elapsed=round(el, 1),
                  swap_elapsed=round(el_swap, 1), swap_frames=len(swap_ms),
                  swap_p50=p50,
                  frame_fps=round(rate, 2),
                  eta_s=(int((total - done) / rate) if rate > 0 and done < total else 0),
                  pct=min(_PROJ_PCT_SWAP, int(_PROJ_PCT_SWAP * done / float(total))),
                  msg="合成 %d/%d 帧 · %.1f 帧/秒%s"
                      % (done, total, rate, ("（跳过 %d）" % skipped) if skipped else ""))
        if len(failed_frames) > _PROJ_MAX_LIST:
            del failed_frames[_PROJ_MAX_LIST:]
        _adapt_check()
        if done % 20 == 0 or done == total:
            _proj_log("合成 %d/%d 帧 · 换脸 %d · 原样 %d · 跳过 %d · 失败 %d · "
                      "worker 单帧 p50 %sms"
                      % (done, total, swapped, copied, skipped, len(failed_frames),
                         p50 if p50 is not None else "-"))

    cancelled = False               # finally 里要读它（记账那一拍：取消的单不采纳峰值）
    try:
        # ---- 自适应：没有实测就现场标定（有缓存的话 merge_export_project 已经选好了）----
        if str(adapt.get("src") or "") == "bench" and _projplan.enabled():
            try:
                _calibrate()
            except Exception as e:                      # noqa: BLE001
                traceback.print_exc()
                _proj_warn("标定失败（不影响这一单，按保守档 %d×%d 跑）：%s: %s"
                           % (len(pool), int(cur["batch"]), e.__class__.__name__,
                              str(e)[:160]))
        _proj_set(adaptive=_adapt_snap())
        # ---- 三段流水：准备（读帧/检测/68 点）→ 换脸（worker 里 k 帧并行）→ 落盘 ----
        # 主线程只干三件事：切批、按**提交顺序**收结果、记账 —— 落盘顺序因此和串行时
        # 一样（断点续跑的游标不变），PNG 编码则挪到落盘线程里，不占换脸的时间。
        from concurrent.futures import ThreadPoolExecutor
        prep_n = max(2, min(8, int(batch) * 2))
        enc_n = max(2, min(6, int(n_workers) * 2))
        window_b = max(1, int(n_workers)) + 1

        def _slot_batch(names_b, preps):
            w = pool.acquire()
            try:
                return _swap_batch(names_b, preps, w)
            finally:
                pool.release(w)

        batch_q = collections.deque()
        writes = collections.deque()
        it = iter(frames)

        with ThreadPoolExecutor(max_workers=prep_n,
                                thread_name_prefix="proj-prep") as prep_pool, \
                ThreadPoolExecutor(max_workers=max(1, int(n_workers)),
                                   thread_name_prefix="proj-swap") as swap_pool, \
                ThreadPoolExecutor(max_workers=enc_n,
                                   thread_name_prefix="proj-write") as write_pool:

            def _pump(limit):
                """按顺序收落盘结果（顺序 = 帧顺序 → 续跑游标才对）。"""
                while len(writes) > limit:
                    nm, fut = writes.popleft()
                    try:
                        kind, err = fut.result()
                    except Exception as e:          # noqa: BLE001
                        kind, err = "err", {"frame": nm,
                                            "error": "%s: %s"
                                                     % (e.__class__.__name__,
                                                        str(e)[:160])}
                    _count(nm, kind, err)
                    _tick()

            while True:
                while len(batch_q) < window_b:
                    names_b = []
                    for _k in range(int(cur["batch"])):
                        try:
                            names_b.append(next(it))
                        except StopIteration:
                            break
                    if not names_b:
                        break
                    if _proj_cancelled():
                        cancelled = True
                        break
                    preps = [prep_pool.submit(_prep, nm) for nm in names_b]
                    batch_q.append((names_b,
                                    swap_pool.submit(_slot_batch, names_b, preps)))
                if cancelled or not batch_q:
                    break
                names_b, fut = batch_q.popleft()
                try:
                    landed = fut.result()
                except Exception as e:              # noqa: BLE001
                    landed = [(nm, "err", {"frame": nm, "error": "%s: %s"
                                           % (e.__class__.__name__, str(e)[:160])})
                              for nm in names_b]
                for (nm, kind, payload) in landed:
                    if kind == "res" and payload:
                        _ms = (payload.get("timings") or {}).get("swap_ms")
                        if _ms:
                            swap_ms.append(float(_ms))
                    writes.append((nm, write_pool.submit(_write_frame,
                                                         nm, kind, payload)))
                _pump(enc_n * 2)
            if cancelled:
                # 取消：先杀掉 worker（ONNX 推理从外部打断不了），在飞的那几个
                # 立刻带着"失联"错误返回，线程池才收得掉；已经写好的帧留着。
                stop.set()
                pool.close("项目导出已取消")
            _pump(0)                  # 已经算出来的帧照样落盘（续跑会跳过它们）
            if cancelled:
                raise _ProjCancel()
    finally:
        samp_stop.set()          # 采样线程：这一单的峰值到这里为止
        if pool is not None:
            pool.close()
    # 把「一份 worker 会话真跑起来多大」记下来（下一单的预算用它；取消/掉帧/贴墙的不记）
    _proj_record_sess_peak(_dfms, n_workers, batch, base_used, samp.get("peak"),
                           samp.get("total"), size_px=_dfm_size_hint(_dfms[0]),
                           ok=(not cancelled) and not failed_frames)
    if face_miss:
        _proj_warn("%d 帧有对齐脸但没检出人脸，已按原帧落盘：%s"
                   % (len(face_miss), "、".join(face_miss[:6])))
    # frame_fps 保留循环里最后一次的值（按真正处理过的帧算，跳过的不算）
    _proj_set(failed_frames=failed_frames)

    return _proj_finish(cfg, project, frames, frame_ext, merged_dir, t0, total)


def _proj_worker(cfg: dict):
    """线程入口：所有异常在这里落成终态，绝不让状态永远停在 running。"""
    try:
        _proj_run(cfg)
        if cfg.get("downgrade_warn"):
            _proj_warn(str(cfg["downgrade_warn"]))     # 收尾再提一次：status.log 只看最后 12 行
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
      workers       可选，并行 worker 进程数。**默认按模型体量**（不是一律 2）：
                    大模型（≥1GB 权重，例如 384 那份 1.53GB）→ 1；小模型（≤600MB，
                    例如 416 那份 0.5GB）→ 2；中间档先要 2，再由显存预算按「free +
                    一份会话的实测峰值」砍（见 _proj_workers_default）。每个 worker
                    是一份独立的 dfm session，而那份会话**真跑起来**的峰值是 3.4GB
                    （384）/ 2.2GB（416）—— 不是文件体积那 1.5GB / 0.5GB。
                    实际值回在 status.workers 里。
      batch         可选，一条请求里带几帧 = worker 内部同时跑几帧。默认 **4 // workers**
                    （1 worker → 4 线程、2 worker → 2 线程），同时在飞的帧数因此锁在
                    4 左右 —— 那是实测的甜点（8 帧在飞就换页）。384 模型实测：1×2 8.73
                    帧/秒、1×4 10.87、1×8 4.44，所以单 worker 时线程必须给到 4。
                    workers×batch = 同时在飞的帧数，实际值回在 status.batch 里。

      显存不够时的退档顺序（用户定的）：**先减 worker → 再降线程（batch）→ 最后才拒绝**
      （拒绝是 HTTP 400 + need_mb/avail_mb/suggest 等结构化字段）。
    """
    body = _as_dict(body)
    _eff.log_request("/api/merge/export-project", body)
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
    # 并行度：N 个 worker 进程，各持自己的 dfm session（iperov stage4 的同一个语义）；
    # batch = 一条请求里带几帧（= worker 内部同时跑几帧）。
    # 默认值**按模型体量**：大模型（≥1GB 权重）1 个、小模型（≤600MB）2 个（见
    # _proj_workers_default）；线程数 = 4 // workers（见 _proj_batch_default）。
    plan_paths = [m["dfm"] for m in (models or []) if m.get("dfm")] or [str(dfm)]
    model_mb = _vram_models_mb(plan_paths)
    workers = int(_int_val("workers", body.get("workers"), 1, 16,
                           _proj_workers_default(model_mb)))
    batch = int(_int_val("batch", body.get("batch"), 1, _PREVIEW_MAX_BATCH,
                         _proj_batch_default(workers)))
    # ---- 自适应档位：**有实测就按实测选，没有就保守起、跑完把实测写回去** ----
    # 静态规则（按权重 ≥1GB → 1 worker）只是「还没量过」时的兜底。同一张 8GB 卡，
    # 「4 帧摊在 1 份会话还是 2 份上」实测是 77% vs 95%（换页）—— 这件事跟权重只是
    # 相关、不是因果（还看 arena 怎么涨、帧尺寸、桌面/浏览器占了几个 GB）。所以：
    #   · 本机量过这份模型 → 用 <workspace>/_proj_bench.json 里的实测挑（_projplan.choose）
    #   · 没量过 → 按保守档起，_proj_run 开跑前用 8~12 帧现场标定，选完再正式跑
    #   · 请求体 / 环境变量显式给了 workers/batch → 一句话都不改（尊重手动）
    adapt = {"src": "off", "explicit": False, "note": ""}
    explicit = bool(str(body.get("workers") or "").strip()
                    or str(body.get("batch") or "").strip()
                    or str(os.environ.get("DFL_PROJ_WORKERS") or "").strip()
                    or str(os.environ.get("DFL_PROJ_BATCH") or "").strip())
    adapt["explicit"] = bool(explicit)
    adapt_key = ""
    if not explicit and _projplan.enabled():
        adapt_key = _projplan.model_key(plan_paths)
        _row = _projplan.get_row(WORKSPACE, adapt_key)
        if _row.get("rows"):
            try:
                _f0, _t0, _u0 = _vram_free_smi(fresh=True)
            except Exception:                       # noqa: BLE001
                _t0 = _u0 = None
            _pick = _projplan.choose(_row["rows"], _t0, _u0)
            if _pick.get("pick"):
                workers = int(_pick["pick"]["workers"])
                batch = int(_pick["pick"]["batch"])
                adapt = {"src": "cache", "key": adapt_key, "rows": _pick["rows"],
                         "pick": _pick["pick"], "rejected": _pick.get("rejected"),
                         "cap": _pick.get("cap"),
                         "note": _projplan.note(_row["rows"], _pick)}
            else:
                _cons = _projplan.conservative(model_mb)
                workers, batch = int(_cons["workers"]), int(_cons["batch"])
                adapt = {"src": "cache-none", "key": adapt_key, "rows": _pick.get("rows"),
                         "pick": None,
                         "note": _projplan.note(_row["rows"], _pick) + " → " + _cons["why"]}
        else:
            _cons = _projplan.conservative(model_mb)
            workers, batch = int(_cons["workers"]), int(_cons["batch"])
            adapt = {"src": "bench", "key": adapt_key, "rows": None, "pick": None,
                     "note": "%s（本次导出前现场标定 8~12 帧）" % _cons["why"]}
    elif explicit:
        adapt["note"] = "手动指定 workers/batch → 不改（只按显存预算削减）"
    # ---- 显存预算：能开几个 worker 就开几个，装不下就按 worker → 线程 → 拒绝 的顺序退 ----
    # 每多一个 worker 就多一份「会话 + CUDA context」，而且那份会话**真跑起来还要再长一块**
    # （384 实测 ≈3.4GB/份，不是「权重 + context」那 2.08GB）—— 所以这里喂进去的是
    # _vram_sess_peak_mb 的实测峰值（没量过就按标定值），不是文件体积推算。
    # 以前是「先起 4 个进程、看谁崩」，崩了再退 —— 那一步本身就会把卡顶到 OOM。
    ses = _vram_sess_peak_mb(plan_paths)
    minfo = _dfm_info(plan_paths[0]) if plan_paths else {}
    # 预算削减前用户/自适应想要的 worker 数：砍了就一定要在 job.log 里说一句
    # （「降级并提示」是这条链的契约；只降不提示 = 用户以为自己在跑 2 份会话）
    workers_req = int(workers)
    plan = _vram_guard(plan_paths, {"mode": "procs", "max_procs": int(workers),
                                    "batch": int(batch), "aux": True,
                                    "sess_mb": float(ses["sess_mb"]),
                                    "sess_src": str(ses["src"]),
                                    "model_info": minfo,
                                    "reclaim_first": True}, label="项目导出")
    if int(plan.get("procs") or 1) < int(workers):
        workers = max(1, int(plan.get("procs") or 1))
    if int(plan.get("batch") or 1) < int(batch):
        batch = max(1, int(plan.get("batch") or 1))
    # ---- 实测优先：缓存的实测档被预算砍掉时，以**实测**为准（不写死、也不拿估算压实测）----
    # 预算那套是「一份会话的峰值 + 批台阶」的估算，会高估：416 今天估 5781MB / 实测整卡
    # 6503MB（程序 4906MB），于是 2×2 被砍成 1 个 worker —— 而同一台机器真的量到过
    # 79.8% / 14.8 帧/秒。既然实测在换页线以内，就没理由为了估算退档。
    _pick = adapt.get("pick") or None
    if (_pick and str(adapt.get("src")) == "cache"
            and int(_pick.get("workers") or 1) > int(plan.get("procs") or 1)
            and float(_pick.get("pct") or 0.0) <= 100.0 * _projplan.HARD):
        print("[project] 实测档 %s 整卡 %.0f%% ≤ 换页线 %.0f%% → 覆盖预算削减（%d → %d worker）"
              % (_pick["label"], float(_pick["pct"]), 100.0 * _projplan.HARD,
                 int(plan.get("procs") or 1), int(_pick["workers"])), flush=True)
        workers, batch = int(_pick["workers"]), int(_pick["batch"])
        plan.update({"procs": int(workers), "batch": int(batch),
                     "requested_procs": int(workers), "requested_batch": int(batch),
                     "adaptive_override": True, "downgraded": False})
    # 这一行是排障第一现场（server.log + job.log 各一份）：模型多大 / 可用多少 /
    # 计划几个 worker × 几线程 / 减了没有、为什么减。
    plan["worker_note"] = _proj_plan_note(model_mb, plan)
    print(plan["worker_note"], flush=True)
    # 档位是**怎么来的**：缓存的实测 / 现场标定 / 保守默认 / 手动指定 —— 和上面那行
    # 预算说明并列写进日志与 status（"为什么不是另一个档"要能当场回答）。
    plan["adaptive"] = dict(adapt)
    if adapt.get("note"):
        print("[project] " + str(adapt["note"]), flush=True)
    # ---- 起 worker **之前**先把本进程的死重放掉 ----
    # 项目导出的 worker 是独立进程（各持自己的 dfm session），所以本进程里缓存的
    # dfm/XSeg 会话、以及常驻预览 worker 里那份模型，在这一单里都是**一点用没有**的
    # 死重 —— 而它们正好吃掉「峰值 ≤85%」的那 15% 余量：实测同一单（384 · 1×4）
    # 卡干净的峰值是 6140MB（75.3%），先跑过阶段链、父进程还留着 dfm/XSeg 时会顶到
    # 7488MB（91.9%）。没有别的任务在跑时才动（有任务在跑时放掉会话等于锯它的腿）。
    plan["released"] = {}
    if not _merge_job.get("running") and not _STAGE_JOB.get("running"):
        try:
            rel = _vram_release_inproc("项目导出开始前")
            rec = _vram_reclaim("项目导出开始前")
            plan["released"] = {"inproc": list((rel or {}).get("released") or []),
                                "preview_mb": round(float((rec or {}).get("freed_mb") or 0.0), 1)}
            if plan["released"]["inproc"] or plan["released"]["preview_mb"]:
                print("[project] 起 worker 前先放掉死重：本进程会话 %s · 预览模型 %.0f MB"
                      % ("、".join(plan["released"]["inproc"]) or "无",
                         plan["released"]["preview_mb"]), flush=True)
        except Exception as e:                          # noqa: BLE001
            print("[project] 起 worker 前放死重失败（不影响这一单）：%s: %s"
                  % (e.__class__.__name__, str(e)[:160]), flush=True)

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
           "force": force, "workers": int(workers), "batch": int(batch),
           "adaptive": dict(adapt), "vram_plan": dict(plan)}

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
            "workers": int(workers), "vram_plan": dict(plan),
            "on_cpu": bool(plan.get("cpu")),
            "model": str(model_name),
            "models": [{"name": m["name"], "dfm": m["dfm"],
                        "faces": list(m["faces"])} for m in (models or [])],
            "settings": settings, "detector": detector, "aligned_faces": 0,
            "merged_frames": 0, "warnings": [], "failed_frames": [], "missing": [],
            "missing_source": [], "encode": {},
            "eff": None, "eff_warnings": [],
        })
    # ---- 生效配置：一行 [eff] + 一致性自检（warnings[] 里同时放纯文本）----
    try:
        eff = _eff_publish("/api/merge/export-project", "project", body, cfg)
        _proj_set(**_eff_status_patch(eff))
    except Exception as e:                                  # noqa: BLE001
        print("[eff] 生成失败（不影响这一单）：%s: %s"
              % (e.__class__.__name__, str(e)[:200]), flush=True)
    # ---- 降级要留痕：预算把 worker 数砍下来时，job.log 里必须有一句 [warn] ----
    # 顺序：先 eff（它会把 warnings 置成生效配置的那几条），再追加降级提示。
    if int(workers) < int(workers_req):
        try:
            from . import _proj_pipe as _pp
            _pipeline = bool(_pp.enabled())
        except Exception:                                   # noqa: BLE001
            _pipeline = False
        _budget = ""
        if plan.get("avail_mb"):
            _budget = ("（可用 %.2fGB / 需要 %.2fGB）"
                       % (float(plan.get("avail_mb") or 0.0) / 1024.0,
                          float(plan.get("need_mb") or 0.0) / 1024.0))
        if _pipeline:
            _downgrade_msg = ("只有一份 dfm 会话（融合流水线）：请求 %d 个 worker，实际 1 份；"
                              "worker 数只用于预算档位%s" % (int(workers_req), _budget))
        else:
            _downgrade_msg = ("只装得下 %d 个 worker（请求 %d）：显存预算不够，已减到 %d 份会话%s"
                              % (int(workers), int(workers_req), int(workers), _budget))
        _proj_warn(_downgrade_msg)
        # /status 的 log 只回最后 12 行：开头这一句会被几十帧的进度冲掉。收尾再提一次，
        # 因为「用户会看的那一屏」就是最后 12 行（这也是这个任务要治的病：写了但看不见）。
        cfg["downgrade_warn"] = _downgrade_msg
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
    snap["vram_plan"] = dict(snap.get("vram_plan") or _vram_last("项目导出") or {})
    snap["on_cpu"] = bool(snap["vram_plan"].get("cpu")) or bool(snap.get("on_cpu"))
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


# ============================================================
# 视频模式导出（不需要先建 DFL 项目）
# ------------------------------------------------------------
# 用户定的链路，一步一步照它实现（别加前置条件）：
#   读源视频 -> 逐帧解码 -> 检测 -> 68 点 -> 决定模型 -> dfm 推理 -> 遮罩推理
#   -> 合成 -> 写出帧 -> ffmpeg 合成视频（音轨 copy）
#
# 「哪张脸用哪个 dfm」只有一处判定，见 _vid_models_for_frame()：
#   * 没有人脸库（库不存在 / 库里没有任何组 / 组里一张有特征的脸都没有）
#     -> 不做 embedding 判定，**所有脸都用左栏当前选中的那个 dfm**
#        （models=[{faces: []}]，空 faces 就是「我管全部脸」）；
#   * 有人脸库 -> _auto_face_plan()：每张脸算 ArcFace embedding，和每个组的均值
#     比余弦，最大且 >= 0.35 归该组、用该组的 dfm；**不匹配的脸进 no_swap
#     （保持原像素）**，绝不兜给任何一个模型。
# 无人脸的帧：检测之后一步都不跑（不算 68 点、不跑 dfm），原帧直出。
#
# 进度 / 取消沿用 /api/merge/export 那一套（_merge_job + /export/status +
# /export/cancel），前端拿到的进度字段一个都没变，只是多了 vid 那一段明细。
# ============================================================
_VID_PCT_SWAP = 92                 # 逐帧合成占 0~92%，剩下 8% 给编码
_VID_MAX_LOG = 4000                # 逐帧日志保留上限（超了丢最老的）
_VID_MAX_RECORDS = 20000           # 逐帧记录（给验收 / 排障看）


class _VidCancel(Exception):
    """用户取消（和失败分开记：取消不是错误）。"""


def _vid_set(**kw):
    with _merge_job_lock:
        _merge_job.update(kw)


# 逐帧日志同时落一份到 <成片>.log：状态接口只回最后 50 行，整片每帧都走了 dfm
# 这件事得有个能 grep 的地方（用户要的是「日志显示每帧都走了 dfm」）。
_VID_LOG_LOCK = threading.Lock()
_VID_LOG_FH = {"fh": None, "path": ""}


def _vid_open_log(path: Path) -> str:
    """开一份逐帧日志（写不进去就算了，不影响导出）。"""
    try:
        fh = open(str(path), "w", encoding="utf-8")
    except OSError:
        return ""
    with _VID_LOG_LOCK:
        _VID_LOG_FH["fh"] = fh
        _VID_LOG_FH["path"] = str(path)
    return str(path)


def _vid_close_log() -> str:
    with _VID_LOG_LOCK:
        fh = _VID_LOG_FH.get("fh")
        _VID_LOG_FH["fh"] = None
        path = str(_VID_LOG_FH.get("path") or "")
        _VID_LOG_FH["path"] = ""
    if fh is not None:
        try:
            fh.close()
        except OSError:
            pass
    return path


def _vid_log(msg):
    line = str(msg)[:300]
    with _VID_LOG_LOCK:
        fh = _VID_LOG_FH.get("fh")
        if fh is not None:
            try:
                fh.write(line + "\n")
                fh.flush()
            except OSError:
                pass
    with _merge_job_lock:
        log = _merge_job.get("log")
        if not isinstance(log, list):
            log = []
            _merge_job["log"] = log
        log.append(line)
        if len(log) > _VID_MAX_LOG:
            del log[:-_VID_MAX_LOG]


def _vid_stats(**kw):
    with _merge_job_lock:
        st = dict(_merge_job.get("stats") or {})
        st.update(kw)
        _merge_job["stats"] = st


def _vid_record(rec: dict):
    with _merge_job_lock:
        rows = _merge_job.get("frames")
        if not isinstance(rows, list):
            rows = []
            _merge_job["frames"] = rows
        rows.append(dict(rec))
        if len(rows) > _VID_MAX_RECORDS:
            del rows[:-_VID_MAX_RECORDS]


def _vid_cancelled() -> bool:
    with _merge_job_lock:
        return bool(_merge_job.get("cancelled"))


def _vid_probe(video: Path) -> dict:
    """源视频的 帧数 / fps / 宽高（导出只认源视频，不猜）。

    ffmpeg 的 metadata 探测偶尔会抽风（文件正被别的进程摸一下、刚写入还没 flush 完），
    一次失败就 422 会让整条分段链莫名其妙断掉 —— 这里重试两次再说。
    """
    last = None
    frames = fps = 0
    w = h = 0
    for attempt in range(3):
        try:
            frames, fps, w, h = _app_mod._video_probe(video)
            last = None
            break
        except Exception as e:                              # noqa: BLE001
            last = e
            time.sleep(0.25 * (attempt + 1))
    if last is not None:
        raise _BadInput("读不出视频信息（%s）：%s" % (video.name, str(last)[:160]), 422)
    if not int(w or 0) or not int(h or 0):
        raise _BadInput("视频尺寸读不出来（%s）—— 换一个能解的片子" % video.name, 422)
    return {"frames": int(frames or 0), "fps": float(fps or 30.0),
            "w": int(w), "h": int(h)}


def _vid_models_for_frame(cfg: dict, frame, boxes, idx: int):
    """（判定点）这一帧每张脸用哪个 dfm。**只有这一处**决定换不换、用谁换。

    返回 (models, no_swap, plan)：
      plan is None —— 没有人脸库：models = [左栏选中的那个 dfm，faces=[] 管全部脸]，
                     no_swap 空，所有脸都用它换；
      plan 有值    —— 有库：models = 命中的那些组（每条的 faces 是按检测顺序的脸序号），
                     no_swap = 谁都不像的脸（保持原像素，不兜给任何模型）；
                     plan["models"] 为空 = 这一帧没一张脸匹配上 -> 整帧不换。
    """
    plan = _auto_face_plan(cfg["plan_payload"], frame, cfg["snap"], "video", "",
                           int(idx), cfg["detector"], cfg["thresh"], boxes=boxes)
    if plan is None:
        return list(cfg["single_models"]), [], None
    return list(plan.get("models") or []), list(plan.get("unmatched") or []), plan


def _vid_box_wh(boxes, i):
    """第 i 个检测框 -> [x, y, w, h]（原帧坐标）；算不出来回 None。"""
    try:
        x1, y1, x2, y2 = [float(v) for v in (boxes[int(i)].get("box") or ())]
    except Exception:                                       # noqa: BLE001
        return None
    return [int(round(x1)), int(round(y1)),
            int(round(max(0.0, x2 - x1))), int(round(max(0.0, y2 - y1)))]


def _vid_face_rows(cfg: dict, plan, res: dict, boxes) -> list:
    """这一帧每张脸最终用了谁 —— 给验收/排障看的按脸明细（下标 = 检测顺序）。

    每行：idx 脸序号 / dfm 用的模型名（空 = 这张脸没换）/ matched / similarity /
    box 原帧坐标的 w-h 框 / applied 这张脸真的贴上去了吗。
    """
    applied = {}
    for f in (res.get("faces") or []):
        try:
            applied[int(f.get("face"))] = f
        except (TypeError, ValueError):
            continue
    rows = []
    if plan is not None:
        by_idx = {}
        for f in (plan.get("faces") or []):
            try:
                by_idx[int(f.get("idx"))] = f
            except (TypeError, ValueError):
                continue
        for i in range(len(boxes or [])):
            src = by_idx.get(i) or {}
            dfm = str(src.get("dfm") or "")
            rows.append({"idx": int(i), "dfm": dfm,
                         "matched": bool(src.get("matched")) and bool(dfm),
                         "similarity": src.get("similarity"),
                         "group": str(src.get("group") or ""),
                         "auto": True, "box": _vid_box_wh(boxes, i),
                         "applied": int(i) in applied})
        return rows
    name = str(cfg["model"])
    for i in range(len(boxes or [])):
        rows.append({"idx": int(i), "dfm": name, "matched": True, "similarity": None,
                     "group": "", "auto": False, "box": _vid_box_wh(boxes, i),
                     "applied": int(i) in applied})
    return rows


def _vid_one(cfg: dict, frame, idx: int, scratch: Path, stop, state):
    """一帧的活：检测 ->（无人脸：原帧直出）-> 68 点 -> dfm -> 遮罩 -> 合成 -> PNG。

    返回 (逐帧记录, PNG 字节)。整帧只有一次检测（检测结果同时喂给自动分配），
    和 /api/merge/preview 那条路是同一份 _preview_job —— 同一帧走预览和走导出，
    输出是同一份像素。
    """
    t0 = time.time()
    boxes = _detect_boxes(frame, cfg["detector"], cfg["thresh"])
    rec = {"frame": int(idx), "detected": len(boxes or []), "faces": [],
           "dfm": "", "branch": "", "no_face": False, "swapped": 0, "ms": 0.0}
    if not boxes:
        # 无人脸的帧：不跑 dfm，原帧直出（PNG 无损，免得白掉一档画质）
        data = _img_bytes(frame, "png")
        rec.update(no_face=True, branch="copy",
                   ms=round((time.time() - t0) * 1000.0, 1))
        return rec, data

    models, no_swap, plan = _vid_models_for_frame(cfg, frame, boxes, idx)
    rec["branch"] = "library" if plan is not None else "single"
    if plan is not None and not models:
        # 有库，但这一帧一张脸都不像任何组 —— 整帧保持原像素（没有兜底模型）
        data = _img_bytes(frame, "png")
        rec.update(faces=_vid_face_rows(cfg, plan, {}, boxes), branch="library-none",
                   ms=round((time.time() - t0) * 1000.0, 1))
        return rec, data

    res = _preview_job({"frame": frame, "models": models or None, "dfm": cfg["dfm"],
                        "xseg": cfg["xseg"], "config": cfg["config"],
                        "detector": cfg["detector"], "thresh": float(cfg["thresh"]),
                        "stop": stop, "state": state, "fmt": "png",
                        "quality": _PREVIEW_QUALITY, "boxes": boxes,
                        "no_swap": list(no_swap or ()), "send_lossless": True})
    data = res.get("img") or res.get("jpg") or b""
    if not data:
        raise _BadInput("第 %d 帧没合成出图" % int(idx), 500)
    rows = _vid_face_rows(cfg, plan, res, boxes)
    rec.update(faces=rows, swapped=int(res.get("swaps") or 0),
               dfm=str((models[0] or {}).get("name") if models else cfg["model"]),
               ms=round((time.time() - t0) * 1000.0, 1))
    return rec, data



# ============================================================
# 分段式视频导出：<视频名>_cache/ + 五个能单独跑的阶段
# ------------------------------------------------------------
# 用户定的口径（推翻原来「一个循环 decode→detect→swap→encode」）：
#
#   D:/x/9月29日 (5).mp4
#   D:/x/9月29日 (5)_cache/           <- 同目录，名字 + _cache
#     frames/   000001.png …          <- 切帧；合成阶段**同名就地覆盖**成合成结果
#     aligned/  000001_00.jpg …       <- 切脸：DFLJPG（original_name + 68 点 + face_mat）
#                                       遮罩阶段把 256 的 XSeg 遮罩写回它的 APP15
#     meta.json                       <- 源信息 / 阶段标记 / settings 快照
#
# 五个阶段各自一个 API（都能单独跑，也从任意已完成的阶段续跑）：
#   POST /api/merge/stage/frames   视频 -> frames/
#   POST /api/merge/stage/faces    frames/ -> aligned/
#   POST /api/merge/stage/mask     aligned/ -> 写回 APP15['xseg_mask']（只有遮罩来源是
#                                  XSeg / XSegLite 时才需要；来源=DFM 时本阶段自动跳过）
#   POST /api/merge/stage/merge    frames/ + aligned/ + dfm -> 就地覆盖 frames/
#   POST /api/merge/stage/encode   frames/ + 源音轨 -> <视频名>_merged.mp4
#   状态 GET /api/merge/stage/status · 取消 POST /api/merge/stage/cancel
#
# 三条硬口径：
#   1) cache 只是缓存：**导出成功就把整个 <名字>_cache/ 删掉**；失败/取消保留，续跑用。
#   2) 合成结果直接覆盖 frames/ 里的同名帧，不另开 merged/（不凭空多一倍空间）。
#      覆盖过的 frames 由 meta.frame_swapped=true 标记 —— 这时再跑切帧会被拒
#      （否则会把原帧盖回来、却留下已经合成过的 meta）。
#   3) 帧图一律最高质量：默认 PNG 无损（IMWRITE_PNG_COMPRESSION=3）；要 jpg 就是
#      quality=100 + 4:4:4，绝不用默认的 95/85/75。
# ============================================================
_STAGE_ORDER = ("frames", "faces", "pred", "mask", "merge", "encode")
_STAGE_LABEL = {"frames": "切帧", "faces": "切脸", "pred": "换脸", "mask": "遮罩",
                "merge": "合成", "encode": "导出", "fused": "快速导出"}
# 老 _merge_job["stage"] 的 7 段进度条（0..6）里没有「遮罩」，给它 2 就是和切脸同一格：
# 老前端照样能看出「还在前面几步」，不会因为多插一步跳到 3 把 31% 显示成合成中。
_STAGE_NUM = {"frames": 1, "faces": 2, "pred": 2, "mask": 2, "merge": 3, "encode": 6}
_STAGE_PCT = {"frames": (0, 10), "faces": (10, 20), "pred": (20, 35), "mask": (35, 48),
              "merge": (48, 92), "encode": (92, 100)}
_STAGE_ALIAS = {
    "frames": "frames", "frame": "frames", "extract-frames": "frames",
    "extract_frames": "frames", "extractframes": "frames", "decode": "frames",
    "切帧": "frames",
    "faces": "faces", "face": "faces", "extract-faces": "faces",
    "extract_faces": "faces", "extractfaces": "faces", "align": "faces",
    "aligned": "faces", "切脸": "faces",
    "pred": "pred", "preds": "pred", "dfm": "pred", "infer": "pred", "face": "faces",
    "swap": "pred", "swapped": "pred", "换脸": "pred", "推理": "pred", "dfm推理": "pred",
    "mask": "mask", "masks": "mask", "xseg": "mask", "xseg-mask": "mask",
    "xseg_mask": "mask", "xsegs": "mask", "seg": "mask", "apply-mask": "mask",
    "segmask": "mask", "遮罩": "mask", "遮罩推理": "mask",
    "merge": "merge", "compose": "merge", "merged": "merge",
    "swap-merge": "merge", "merge-frames": "merge", "合成": "merge",
    "encode": "encode", "mux": "encode", "export": "encode", "video": "encode",
    "导出": "encode",
}
_STAGE_META_NAME = "meta.json"
_STAGE_IMG_EXTS = (".png", ".jpg", ".jpeg")
_STAGE_LOG_MAX = 3000
_STAGE_LOCK = threading.RLock()
_STAGE_JOB = {
    "running": False, "ok": None, "cancelled": False, "error": "", "error_detail": "",
    "phase": "idle", "stage": "", "stage_index": -1, "stage_pct": 0, "pct": 0,
    "msg": "", "video": "", "cache_dir": "", "out_video": "", "frames_total": 0,
    "frames_done": 0, "frame_fps": 0.0, "eta_s": None, "counts": {}, "stages_done": [],
    "log": [], "meta": {}, "seconds": {}, "resumed": {}, "swapped": 0, "copied": 0,
    "no_face": 0, "cache_removed": False, "started_at": None, "finished_at": None,
    "chain": False, "keep_cache": False, "force": False,
    # 生效配置 + 一致性警告（[eff] 行同款；进度页顶部横幅读它）
    "eff": None, "eff_warnings": [], "warnings": [],
}


def _stage_name(raw):
    """阶段名 -> 规范名（frames/faces/mask/merge/encode）。认不出来返回 ""。"""
    return _STAGE_ALIAS.get(str(raw or "").strip().lower(), "")


# ---------------- 遮罩来源（决定 mask 阶段要不要跑） ----------------
def _mask_src(settings, mask_mode=None) -> dict:
    """settings（或 meta.settings）-> {"source": xseg|dfm, "model": lite|xseg, "label", "model_from"}。

    两代口径都认，**显式 mask_source 优先**：
      · 界面 seg 给的原文：DFM / XSeg / XSegLite；
      · mask_mode 整数：6/7/8/9 = XSeg 系列（xseg-prd / xseg-dst / prd×dst / 全叠），
        0/1/2/3/4/5 = 模型自带遮罩（DFM）。

    **为什么不返回 "auto" 了（用户报的 bug 的根因之一）**：
      老前端（app.js 的 mgMergeSettings）把「遮罩来源」和「遮罩模式」折成一个 mask_mode
      整数发出来，XSeg 和 XSegLite 都变成 6/7/8 —— 模型身份在传输里就丢了。这里以前
      回 "auto"（谁在磁盘上就用谁，Lite 优先）并把 label 写死成 "XSeg"，于是：
        · 界面显示 XSegLite、status 显示 XSeg、实际跑的是哪个看磁盘 —— 三者随时能不一致；
        · _mask_signature 对 XSeg / XSegLite 都算出 "xseg:auto"，切模型**不会**让 mask/
          失效，换模型后静默复用另一个模型算出来的遮罩。
      现在：给了 mask_source 就只认它（且**不**退回另一个模型）；只给 mask_mode 时按
      界面默认值 XSegLite 解析，并在 model_from="mask_mode" 里说明「这是推断的」，
      label 与实际加载的模型永远一致。
    """
    settings = settings if isinstance(settings, dict) else {}
    raw = ""
    for k in ("mask_source", "mask_seg", "seg_source", "seg", "mask_model", "xseg_model"):
        v = settings.get(k)
        if v is not None and str(v).strip():
            raw = str(v).strip()
            break
    key = (raw.lower().replace("_", "").replace("-", "").replace(" ", "")
           .replace("×", ""))
    mm = mask_mode if mask_mode is not None else settings.get("mask_mode")
    try:
        mm = int(mm) if str(mm if mm is not None else "").strip() != "" else None
    except (TypeError, ValueError):
        mm = None

    # ① 显式选择：只有它说了算（认不出就往下按 mask_mode 推，并标注 model_from）
    if key in ("dfm", "model", "learned", "learnedmask", "none", "off", "0"):
        return {"source": "dfm", "model": "", "label": "DFM", "raw": raw,
                "model_from": "settings", "model_note": ""}
    if key in ("xseg", "xsegprddst", "xsegdst", "xsegprd", "1"):
        return {"source": "xseg", "model": "xseg", "label": "XSeg", "raw": raw,
                "model_from": "settings", "model_note": ""}
    if key in ("xseglite", "lite", "xseglite2", "2"):
        return {"source": "xseg", "model": "lite", "label": "XSegLite", "raw": raw,
                "model_from": "settings", "model_note": ""}
    if raw:
        # 认不出的写法：不猜成 dfm，按 mask_mode（没有就按界面默认的 XSegLite）推断
        _note = ("遮罩来源写了 %r，认不出来 —— 按 mask_mode 推断" % raw)
    else:
        _note = "请求里没有 mask_source（只有 mask_mode），模型按界面默认推断"

    # ② 只有 mask_mode：XSeg 系列一律按界面默认的 XSegLite 解析，并标注「这是推断的」
    if mm is not None and mm >= 6:
        return {"source": "xseg", "model": "lite", "label": "XSegLite", "raw": raw,
                "model_from": "mask_mode", "model_note": _note}
    if mm is not None:
        return {"source": "dfm", "model": "", "label": "DFM", "raw": raw,
                "model_from": "mask_mode", "model_note": ""}
    # ③ settings 里什么都没有：和界面默认一致（遮罩来源默认 XSegLite）
    return {"source": "xseg", "model": "lite", "label": "XSegLite", "raw": raw,
            "model_from": "default", "model_note": _note}


# ============================================================
# 生效配置（[eff] / [req] / 一致性自检）
# ------------------------------------------------------------
# 用户的原话：「我们的控制台、日志显示，做得是不是和没有一样」。事实是检测器那个
# bug 里日志一直写着 retinaface_10g，可没人能从流水账里看出「这跟我设的不一样」。
# 这里把真正生效的配置摊成一行 JSON（每个字段带来源 ui/default/inferred），把
# 「请求体 vs 界面默认」对不上账的地方打成 [eff-warn] 并塞进 status.warnings[]。
# ============================================================
def _eff_model_names(cfg) -> list:
    names = []
    for m in (cfg.get("single_models") or cfg.get("models") or []):
        try:
            n = str(m.get("name") or "")
        except Exception:                                   # noqa: BLE001
            n = str(m or "")
        if n and n not in names:
            names.append(n)
    if not names and cfg.get("model"):
        names = [str(cfg.get("model"))]
    return names


def _eff_basename(value) -> str:
    """只留名字：日志和界面都不需要整条用户路径。"""
    text = str(value or "").strip()
    if not text:
        return ""
    return os.path.basename(text.rstrip("\\/")) or text


def _eff_values_of(cfg: dict, kind: str) -> dict:
    """cfg -> [eff] 的字段值。**只从 cfg 取**，不重新算一遍（否则就不是「生效」了）。"""
    cfg = cfg or {}
    conf = cfg.get("config") or {}
    settings = cfg.get("settings") or {}
    mask = cfg.get("mask_src")
    if not isinstance(mask, dict):
        mask = _mask_src(settings, settings.get("mask_mode"))
    if kind == "project":
        workers, batch, fast = cfg.get("workers"), cfg.get("batch"), False
    elif kind == "fused":
        workers = cfg.get("merge_workers")
        batch = cfg.get("fused_batch") or cfg.get("pred_batch")
        fast = True
    else:
        workers = cfg.get("merge_workers")
        batch = cfg.get("pred_batch")
        fast = bool(cfg.get("chain"))
    q = cfg.get("jpeg_quality")
    if q is None:
        try:
            q = int(cfg.get("quality"))
        except (TypeError, ValueError):
            q = _STAGE_JPEG_QUALITY
    return {
        "detector": str(cfg.get("det_align") or cfg.get("detector") or _DEFAULT_DETECTOR),
        "mask_source": str(mask.get("label") or ""),
        "ct_mode": str(conf.get("color_transfer_mode") or settings.get("ct_mode") or DEFAULT_CT_MODE),
        "blend": str(conf.get("mode") or settings.get("blend") or DEFAULT_BLEND),
        "erode": int(conf.get("erode_mask_modifier") or 0),
        "blur": int(conf.get("blur_mask_modifier") or 0),
        "models": _eff_model_names(cfg),
        # 人脸库只留名字：日志/界面都不需要整条用户路径
        "faces_project": _eff_basename(cfg.get("library") or ""),
        "library_enabled": bool(cfg.get("use_library", True)),
        "fast_export": bool(fast),
        "frame_range": dict(cfg.get("range") or {}),
        "workers": int(workers or 0),
        "batch": int(batch or 0),
        "jpeg_quality": int(q or 0),
    }


def _eff_inferred_of(cfg: dict, body: dict) -> dict:
    """哪些字段是后端从别的字段推出来的（写进 src_note，必要时打 [eff-warn]）。"""
    inf = {}
    body = body if isinstance(body, dict) else {}
    if not str(body.get("detector") or "").strip():
        q = str(body.get("quality") or body.get("align_quality") or "").strip().lower()
        if q:
            inf["detector"] = "quality=%s" % q
    mask = cfg.get("mask_src") if isinstance(cfg.get("mask_src"), dict) else {}
    if str(mask.get("model_from") or "") == "mask_mode":
        inf["mask_source"] = "mask_mode=%s" % (cfg.get("mask_mode"),)
    return inf


def _eff_requested_of(body: dict, kind: str) -> dict:
    """请求体里写的原值（只取和 [eff] 同一个旋钮的那几个键，用来发现「被后端改写」）。"""
    body = body if isinstance(body, dict) else {}
    out = {}

    def _num(*keys):
        for k in keys:
            v = body.get(k)
            if v is None or (isinstance(v, str) and not v.strip()):
                continue
            try:
                return int(float(v))
            except (TypeError, ValueError):
                continue
        return None

    w = _num("workers") if kind == "project" else _num("merge_workers", "threads", "merge_threads")
    if w is not None:
        out["workers"] = w
    if kind == "fused":
        # eff 的 batch 就是 fused_batch；body 的 pred_batch/batch 是另一个旋钮，别拿来对账
        b = _num("fused_batch")
    elif kind == "project":
        b = _num("batch")
    else:
        b = _num("batch", "pred_batch")
    if b is not None:
        out["batch"] = b
    q = _num("jpeg_quality")
    if q is not None:
        out["jpeg_quality"] = q
    return out


def _eff_publish(route: str, kind: str, body: dict, cfg: dict,
                 dedupe: bool = False) -> dict:
    """打 [eff] + [eff-warn]，返回生效配置（调用方把它塞进 status）。

    请求体那一行 [req] 在路由入口打（每次 HTTP 请求一行，dry_run 预检也算），
    这里只负责「真正生效的那份配置」。
    """
    body = body if isinstance(body, dict) else {}
    quiet = {"mask_source"}          # mask_source ← mask_mode 是契约，不算对不上
    if str(kind).startswith("stage:"):
        # 单个阶段不做「快速导出」这个选择（fused 是 /api/merge/export 的旋钮）
        quiet.add("fast_export")
        # 六阶段里只有 faces 真正用检测器（其余阶段拿的是上游产物）
        if kind != "stage:faces":
            quiet.add("detector")
    if kind == "project":
        # DFL 项目导出：没有「快速导出」开关；jpeg 质量是它自己的帧编码旋钮（界面没这控件）
        quiet.update({"fast_export", "jpeg_quality"})
    if kind == "project-modes":
        # 4.0 MergeStudio 老项目模式：不选快速导出、不读人脸库、帧图质量是子进程的事
        quiet.update({"fast_export", "jpeg_quality", "library_enabled"})
    if kind == "preview":
        # 预览不是导出：没有「快速导出」这个选择
        quiet.add("fast_export")
    eff = _eff.build(route, kind, body, _eff_values_of(cfg, kind),
                     inferred=_eff_inferred_of(cfg, body),
                     quiet=quiet,
                     requested=_eff_requested_of(body, kind))
    eff["req_unused"] = _eff.unused_keys(body)
    _eff.log_eff(eff, dedupe=dedupe)
    return eff


def _eff_status_patch(eff: dict) -> dict:
    """生效配置 -> 塞进 status 的几个键（warnings[] 里也放一份纯文本）。"""
    warns = list((eff or {}).get("warnings") or [])
    return {"eff": eff, "eff_warnings": warns,
            "warnings": [str(w.get("text") or w) for w in warns]}


def _stage_files(dirp, exts=_STAGE_IMG_EXTS) -> list:
    """目录里的图，按文件名排序（000001.png … 定长补零，字典序 = 帧序）。"""
    d = Path(dirp)
    if not d.is_dir():
        return []
    out = []
    try:
        for p in d.iterdir():
            try:
                if p.is_file() and p.suffix.lower() in exts:
                    out.append(p)
            except OSError:
                continue
    except OSError:
        return []
    out.sort(key=lambda p: p.name)
    return out


def _stage_frames_dir(cache) -> Path:
    return Path(cache) / "frames"


def _stage_aligned_dir(cache) -> Path:
    return Path(cache) / "aligned"


def _stage_frame_name(i: int) -> str:
    """第 i 帧（0 起）-> 文件名：000001.png（**1 起**，和界面/用户口径一致）。"""
    return "%06d.png" % (int(i) + 1)


def _stage_face_name(i: int, k: int) -> str:
    return "%06d_%02d.jpg" % (int(i) + 1, int(k))


def _stage_meta_path(cache) -> Path:
    return Path(cache) / _STAGE_META_NAME


def _stage_meta_load(cache) -> dict:
    p = _stage_meta_path(cache)
    try:
        if p.is_file():
            with open(str(p), "r", encoding="utf-8") as fh:
                d = json.load(fh)
            if isinstance(d, dict):
                return d
    except (OSError, ValueError):
        pass
    return {}


def _stage_meta_save(cache, meta: dict):
    """meta 落盘：先写 .tmp 再 os.replace —— 断电也不会留下半个 JSON。"""
    if not isinstance(meta, dict):
        return
    p = _stage_meta_path(cache)
    meta["updated_at"] = time.time()
    tmp = Path(str(p) + ".tmp")
    try:
        with open(str(tmp), "w", encoding="utf-8") as fh:
            json.dump(meta, fh, ensure_ascii=False, indent=1)
        os.replace(str(tmp), str(p))
    except (OSError, TypeError, ValueError):
        try:
            if tmp.exists():
                tmp.unlink()
        except OSError:
            pass


def _stage_meta_blank(video: Path, probe: dict, settings: dict) -> dict:
    return {
        "version": 1,
        "source": {"video": str(video), "name": video.name,
                   "fps": float(probe.get("fps") or 0.0),
                   "frames": int(probe.get("frames") or 0),
                   "width": int(probe.get("w") or 0), "height": int(probe.get("h") or 0)},
        "settings": dict(settings or {}),
        "stages": {k: {} for k in _STAGE_ORDER},
        "frame_swapped": False,
        "created_at": time.time(),
    }


def _stage_meta_ready(meta: dict, video: Path, probe: dict) -> dict:
    """老 meta 补齐字段（换了源视频 / 老版本 meta 都能直接接着用）。"""
    meta.setdefault("version", 1)
    src = meta.get("source")
    if not isinstance(src, dict):
        src = {}
    src["video"] = str(video)
    src["name"] = video.name
    src["fps"] = float(probe.get("fps") or src.get("fps") or 0.0)
    src["frames"] = int(probe.get("frames") or src.get("frames") or 0)
    src["width"] = int(probe.get("w") or src.get("width") or 0)
    src["height"] = int(probe.get("h") or src.get("height") or 0)
    meta["source"] = src
    st = meta.get("stages")
    if not isinstance(st, dict):
        st = {}
    for k in _STAGE_ORDER:
        if not isinstance(st.get(k), dict):
            st[k] = {}
    meta["stages"] = st
    meta["frame_swapped"] = bool(meta.get("frame_swapped"))
    return meta


# ---------------- 帧图读写（最高质量）----------------
def _stage_imencode(img, ext=".png", quality=None) -> bytes:
    """帧图编码：png 走无损；jpg 走 **quality=100 + 4:4:4**（禁止默认 95/85）。

    采样因子不再在这里手写：cv2 5.0 只认 ImwriteJPEGSamplingFactorParams 枚举，
    裸数字会被拒并**静默**落成 4:2:0（见 _align.py 的 jpeg444_probe 注释）。
    这里跟对齐脸共用同一份探测结果，一次探测、全进程一个策略。

    quality：帧图 JPEG 质量，默认 100（= _STAGE_JPEG_QUALITY；界面已经没这个旋钮，
    只有老客户端/验收脚本会传）。**PNG 分支不看它**：无损，只认 png_compression。
    """
    import cv2
    ext = str(ext or ".png").lower()
    if ext == ".png":
        ok, buf = cv2.imencode(".png", img, [int(cv2.IMWRITE_PNG_COMPRESSION), 3])
    else:
        from ._align import jpeg_encode_params
        q = _STAGE_JPEG_QUALITY if quality is None else max(1, min(100, int(quality)))
        ok, buf = cv2.imencode(".jpg", img, jpeg_encode_params(q, 0))
    if not ok:
        raise _BadInput("这一帧编不出图（%s）" % ext, 500)
    return buf.tobytes()


def _stage_imencode_png(img, level: int = 1) -> bytes:
    """PNG 编码，压缩级别可指定。合成写回默认 level=1（体积换速度）。

    压缩策略用 **Z_RLE**：PNG 是无损格式，策略只影响压缩后的字节流，解码出来的
    像素逐位不变（实测 np.array_equal ✔）。1080×1920 实测（单线程）：
        level 1 + DEFAULT：编码 68.95ms · 解码 21.21ms · 1.48MB
        level 1 + RLE    ：编码 60.52ms · 解码 22.56ms · 1.35MB
    —— 每帧省 7ms、文件还小 9%（写盘和后面 ffmpeg 读盘都跟着省）。level 0 虽然编得
    更快（35ms），但体积翻 4 倍（6.2MB），实测整体反而更慢（12.32 vs 13.56 帧/秒）。
    """
    import cv2
    params = [int(cv2.IMWRITE_PNG_COMPRESSION), int(level)]
    strat = getattr(cv2, "IMWRITE_PNG_STRATEGY", None)
    rle = getattr(cv2, "IMWRITE_PNG_STRATEGY_RLE", None)
    if strat is not None and rle is not None:
        params += [int(strat), int(rle)]
    ok, buf = cv2.imencode(".png", img, params)
    if not ok:
        raise _BadInput("这一帧编不出 PNG", 500)
    return buf.tobytes()


def _stage_write_atomic(path, data: bytes):
    """先写 .part 再 os.replace：半张图永远不会被当成「这一帧已经好了」。"""
    p = Path(path)
    tmp = Path(str(p) + ".part")
    with open(str(tmp), "wb") as fh:
        fh.write(data)
    os.replace(str(tmp), str(p))


def _stage_read_image(path):
    return _app_mod._imread_any(path)


def _stage_read_frame(path):
    """整帧 PNG 的快速读法：读字节 + imdecode（实测 20ms，_imread_any 要 40ms）。

    _imread_any 里有一堆兜底分支（非 UTF-8 路径、EXIF、多解码器），合成阶段每帧都
    走一遍是净亏。open() 用的是 Python 的 Unicode API，中文路径没问题。
    """
    import cv2
    import numpy as np
    try:
        with open(str(path), "rb") as fh:
            data = fh.read()
    except OSError:
        return None
    if not data:
        return None
    return cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)


def _stage_dst_face_mat(meta: dict, size: int = 256):
    """aligned/*.jpg → 合成画布(size) 的仿射矩阵（2×3）；元数据不全返回 None。

    **为什么不是「直接 resize 到 256」**：这条链的「合成画布」是 facelib 那一套
    （compose_frame 用 get_transform_mat 把 pred 反贴回原帧），所以 dst 遮罩必须落在
    同一张画布上 —— 否则 mask_mode 8/7 的遮罩会整体错开，贴回原帧就是边缘多一条
    挖进脸里的缝。

    **2026-12 更正**：_align.align_mat_from_68 现在只拟合 33 点、与 facelib 逐位等价
    （见 app/docs/24-ALIGN-WHOLEFACE.md），重切之后的 aligned 芯片本来就是 facelib
    画布，所以这张补偿矩阵对它们约等于恒等。历史数据（重切前切的）仍然需要它：那时
    两张画布确实不是同一张（1203 张实测中位缩放差 0.36%、旋转 2.5°，尾部到 13%；
    早期在 256 画布上量到的是「残差 6.1~7.2 px、缩放差 5.5%」）。

    这里把 aligned 先按 image_to_face_mat 的逆变换放回原帧坐标，再用法线（facelib）
    变换压到 size —— 一张矩阵、一次重采样，像素仍然来自 aligned 本身，不需要 1080p 原帧。
    """
    import numpy as np

    if not isinstance(meta, dict):
        return None
    lm = meta.get("source_landmarks") or meta.get("landmarks")
    mat_a = meta.get("image_to_face_mat")
    if not lm or len(lm) != 68 or not mat_a:
        return None
    try:
        from ._merge_compose import ensure_engine_path
        ensure_engine_path()                    # facelib 在 4.0 引擎根里（_paths.engine_root）
        from facelib import FaceType
        from facelib.LandmarksProcessor import get_transform_mat

        a = np.asarray(get_transform_mat(np.asarray(lm, np.float32), int(size),
                                         FaceType.WHOLE_FACE, scale=1.0), np.float64)
        a = np.vstack([a, [0.0, 0.0, 1.0]]) if a.shape == (2, 3) else a
        b = np.asarray(mat_a, np.float64)
        b = np.vstack([b, [0.0, 0.0, 1.0]]) if b.shape == (2, 3) else b
        m = a @ np.linalg.inv(b)                # aligned -> 原帧 -> 画布
        m = np.asarray(m, np.float64)[:2]
        if m.shape != (2, 3) or not np.all(np.isfinite(m)):
            return None
        return m
    except Exception:                                       # noqa: BLE001
        return None


def _stage_read_dst_face(path, size: int = 256):
    """aligned/*.jpg（DFLJPG）→ (合成画布上的 (size,size) BGR uint8, 失败原因)。

    成功时返回 (img, "")；失败时 (None, 一句话原因) —— 调用方按「跳过并计数」处理，
    **绝不**退回读 1080p 原帧（那正是这一版删掉的东西）。

    读图是裸读字节 + imdecode（DFLJPG 的 APP15 元数据段不影响 JPEG 解码，和
    _stage_read_frame 同一条快路径，绕开 _imread_any 的兜底分支）。
    """
    import cv2
    import numpy as np
    from ._align import read_dfljpg_meta

    try:
        with open(str(path), "rb") as fh:
            data = fh.read()
    except OSError:
        return None, "文件读不出来"
    if not data:
        return None, "文件是空的"
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None or getattr(img, "ndim", 0) != 3 or int(img.shape[2]) != 3:
        return None, "JPEG 解不出来/坏图"
    if int(img.shape[0]) < 32 or int(img.shape[1]) < 32:
        return None, "图太小（%dx%d）" % (img.shape[1], img.shape[0])
    m = _stage_dst_face_mat(read_dfljpg_meta(path) or {}, size)
    if m is None:
        return None, "DFLJPG 元数据缺 68 点 / image_to_face_mat"
    out = cv2.warpAffine(img, m, (int(size), int(size)), flags=cv2.INTER_CUBIC)
    if out is None or int(out.shape[0]) != size or int(out.shape[1]) != size:
        return None, "映射到画布失败"
    return out, ""


def _stage_clear_dir(dirp, exts=_STAGE_IMG_EXTS, prefix="") -> int:
    """清掉目录里的阶段产物（只删指定后缀，别误伤 meta.json 之类）。"""
    n = 0
    for p in _stage_files(dirp, exts):
        if prefix and not p.name.startswith(prefix):
            continue
        try:
            p.unlink()
            n += 1
        except OSError:
            pass
    return n


# ---------------- 进度 / 日志 ----------------
def _stage_log(msg):
    line = str(msg)[:300]
    with _STAGE_LOCK:
        log = _STAGE_JOB.get("log")
        if not isinstance(log, list):
            log = []
            _STAGE_JOB["log"] = log
        log.append(line)
        if len(log) > _STAGE_LOG_MAX:
            del log[:-_STAGE_LOG_MAX]
        chain = bool(_STAGE_JOB.get("chain"))
    if chain:
        _vid_log(line)
    try:
        print("[stage] " + line, flush=True)
    except UnicodeEncodeError:
        pass


def _stage_set(**kw):
    with _STAGE_LOCK:
        _STAGE_JOB.update(kw)


def _stage_progress(stage, pct, msg=None, **kw):
    pct = max(0, min(100, int(round(float(pct or 0)))))
    lo, hi = _STAGE_PCT.get(stage, (0, 100))
    with _STAGE_LOCK:
        _STAGE_JOB["stage"] = stage
        try:
            _STAGE_JOB["stage_index"] = _STAGE_ORDER.index(stage)
        except ValueError:
            _STAGE_JOB["stage_index"] = -1
        _STAGE_JOB["stage_pct"] = pct
        _STAGE_JOB["pct"] = int(round(lo + (hi - lo) * pct / 100.0))
        if msg is not None:
            _STAGE_JOB["msg"] = str(msg)[:300]
        for k, v in kw.items():
            _STAGE_JOB[k] = v
    _stage_mirror()


def _stage_mirror():
    """chain 模式（/api/merge/export 视频分支）把进度同步到老的 _merge_job。"""
    with _STAGE_LOCK:
        if not _STAGE_JOB.get("chain"):
            return
        snap = dict(_STAGE_JOB)
    stage = str(snap.get("stage") or "")
    with _merge_job_lock:
        _merge_job["pct"] = int(snap.get("pct") or 0)
        _merge_job["stage"] = int(_STAGE_NUM.get(stage, 0))
        _merge_job["msg"] = str(snap.get("msg") or "")
        _merge_job["frames_done"] = int(snap.get("frames_done") or 0)
        _merge_job["total"] = int(snap.get("frames_total") or 0)
        st = dict(_merge_job.get("stats") or {})
        st.update({"done": int(snap.get("frames_done") or 0),
                   "swapped": int(snap.get("swapped") or 0),
                   "copied": int(snap.get("copied") or 0),
                   "no_face": int(snap.get("no_face") or 0),
                   "frame_fps": float(snap.get("frame_fps") or 0.0)})
        _merge_job["stats"] = st


def _stage_cancelled() -> bool:
    with _STAGE_LOCK:
        if _STAGE_JOB.get("cancelled"):
            return True
        chain = bool(_STAGE_JOB.get("chain"))
    if chain:
        return _vid_cancelled()
    return False


class _StageCancel(Exception):
    """用户点了取消。和失败分开记 —— 取消不是错误。"""


def _stage_check_cancel():
    if _stage_cancelled():
        raise _StageCancel()


# ---------------- 阶段实现 ----------------
def _stage_quality_detector(quality) -> str:
    """切脸用的检测档：high = RetinaFace(det_10g)，fast = TinyMog（和抽取器同一套档位）。"""
    q = str(quality or "high").strip().lower()
    return "tinymog" if q in ("fast", "tinymog") else "retinaface_10g"


def _align_detector(body: dict):
    """切脸/对齐用哪个检测器 → (规范名, 来源说明)。**界面选的 detector 优先。**

    以前这里只有 quality 一个入口：cfg["det_align"] = _stage_quality_detector(quality)，
    而界面根本不发 quality（app.js 里是 payload.align_quality || "high"），于是
    「检测器」下拉无论选什么都落成 retinaface_10g —— 界面上写着 TinyMog、导出日志里
    却是 retinaface_10g，detect 段 49ms/帧（retina 前向本来就 ~12ms，还要跟 dfm 抢 GPU）。
    这跟 mask_source 那次是同一个形状：**界面选了，但没传**。

    现在：请求体里有 detector（界面「检测器」下拉发的就是它）就以它为准；
    没有才按 quality 档推 —— 老脚本 / 验收工具（quality=fast）一个字都不用改。
    """
    raw = str((body or {}).get("detector") or "").strip()
    if raw:
        return _detector_val(raw), "界面/请求体 detector=%s" % raw
    q = str((body or {}).get("quality") or (body or {}).get("align_quality") or "high")
    return (_stage_quality_detector(q),
            "请求体没给 detector → 按 quality=%s 推" % q.strip().lower())


def _stage_brief(res: dict) -> str:
    bits = []
    for k in ("count", "frames", "swapped", "copied", "no_face", "out", "seconds"):
        v = res.get(k)
        if v not in (None, "", 0):
            bits.append("%s=%s" % (k, v))
    return " · ".join(bits) or "ok"


def _stage_extract_frames_ffmpeg(cfg: dict, frames_dir):
    """快路径：ffmpeg 把视频直接写成 000001.png…（逐帧进度、可取消）。

    为什么不让 python 逐帧 imencode：1080x1920 的 PNG 编码 ~0.3s/帧，一段 3 分钟的
    片子光切帧就要 20 分钟。ffmpeg 的 png 编码器快一个数量级，输出同样是无损 PNG
    （-compression_level 1：快、略大；这是缓存，导出成功后整个目录都会删）。
    失败（退出码非 0）返回 None，调用方退回进程内解码那条路。
    """
    video = Path(cfg["video"])
    ffmpeg = str(_app_mod._ffmpeg_exe())
    pattern = str(Path(frames_dir) / "%06d.png")
    cmd = [ffmpeg, "-y", "-hide_banner", "-loglevel", "error", "-nostats", "-nostdin",
           "-i", str(video), "-map", "0:v:0", "-an",
           # -vsync 0：原样搬每一帧（不按时间戳丢/补帧），帧数 = 解码出来的帧数
           "-vsync", "0", "-start_number", "1", "-compression_level", "1",
           "-progress", "pipe:1", pattern]
    t0 = time.time()
    made = 0
    fh = tempfile.NamedTemporaryFile(delete=False, suffix=".ffmpeg.log")
    err_path = fh.name
    fh.close()
    code = -1
    try:
        with open(str(err_path), "wb") as errfp:
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=errfp,
                                    text=True, encoding="utf-8", errors="replace",
                                    creationflags=getattr(_app_mod, "_NO_WINDOW", 0))
            try:
                for line in proc.stdout:
                    if _stage_cancelled():
                        _kill_proc_tree(proc)
                        raise _StageCancel()
                    line = line.strip()
                    if not line.startswith("frame="):
                        continue
                    try:
                        made = int(line.split("=", 1)[1])
                    except ValueError:
                        continue
                    el = max(0.001, time.time() - t0)
                    total = max(int(cfg.get("total") or 0), made)
                    _stage_progress("frames", 100.0 * made / total if total else 0,
                                    "切帧 %d/%s 帧 · %.1f 帧/秒"
                                    % (made, cfg.get("total") or "?", made / el),
                                    frames_done=made, frames_total=total,
                                    frame_fps=round(made / el, 2))
            finally:
                code = proc.wait()
    finally:
        tail = ""
        try:
            with open(str(err_path), "rb") as rf:
                tail = rf.read()[-2000:].decode("utf-8", "ignore").strip()
        except OSError:
            pass
        try:
            os.unlink(str(err_path))
        except OSError:
            pass
    if code != 0:
        _stage_log("ffmpeg 抽帧失败（退出码 %s）：%s —— 退回进程内解码"
                   % (code, (tail or "")[-300:]))
        return None
    cfg["_frames_via"] = "ffmpeg"
    return len(_stage_files(frames_dir))


def _stage_do_frames(cfg: dict) -> dict:
    """阶段 1 · 切帧：视频 -> frames/000001.png…（保留原 fps，PNG 无损）。"""
    # 流水线实况：分段链里这一段只有一路 ffmpeg 解码（一条 lane，名字就叫「解码」）
    _STAGE_METER.set_lanes({"frames": ["解码"]})
    cache = Path(cfg["cache"])
    video = Path(cfg["video"])
    meta = cfg["meta"]
    st = meta["stages"].setdefault("frames", {})
    frames_dir = _stage_frames_dir(cache)
    aligned_dir = _stage_aligned_dir(cache)
    total_hint = int(cfg.get("total") or 0)
    have = len(_stage_files(frames_dir))

    if (not cfg["force"]) and st.get("done") and have and (not total_hint or have == total_hint):
        _stage_log("切帧：复用已有 %d 帧（不重跑；要重切传 force=true）" % have)
        cfg["total"] = have
        return {"skipped": True, "count": have, "reason": "already-done"}

    if meta.get("frame_swapped") and not cfg["force"]:
        raise _BadInput("frames/ 已经被合成结果覆盖（meta.frame_swapped=true）—— 重切会把原帧"
                        "盖回来，和 meta 里「已合成」的标记打架。要重来请传 force=true："
                        "它会清掉 frames/ + aligned/ 和后面三个阶段的标记。", 409)

    _stage_log("切帧：清空 frames/ 与 aligned/（帧变了，按帧配对的对齐脸和它们的遮罩一起作废）")
    n_old = _stage_clear_dir(frames_dir)
    n_old += _stage_clear_dir(aligned_dir)
    frames_dir.mkdir(parents=True, exist_ok=True)
    aligned_dir.mkdir(parents=True, exist_ok=True)
    # 帧变了 -> 按帧配对的脸、换脸结果、遮罩一起作废
    for k in ("faces", "pred", "mask", "merge", "encode"):
        meta["stages"][k] = {}
    meta["frame_swapped"] = False
    meta["stages"]["frames"] = {"done": False, "msg": "切帧中"}
    _stage_meta_save(cache, meta)
    if n_old:
        _stage_log("清掉旧产物 %d 个文件" % n_old)

    t0 = time.time()
    idx = 0
    w = int(meta["source"].get("width") or 0)
    h = int(meta["source"].get("height") or 0)

    # 快路径：ffmpeg 直接写 PNG 序列；失败才退回进程内解码 + cv2.imencode
    got = _stage_extract_frames_ffmpeg(cfg, frames_dir)
    if got is not None:
        idx = int(got)
    else:
        idx = _stage_frames_inproc(cfg, frames_dir, meta, w, h)

    if idx == 0:
        raise _BadInput("一帧都没解出来：%s" % video.name, 422)
    sec = round(time.time() - t0, 2)
    meta["source"]["frames"] = idx
    meta["source"]["width"] = w
    meta["source"]["height"] = h
    rng_f = _stage_range(cfg)
    meta["stages"]["frames"] = {
        "done": True, "count": idx, "seconds": sec, "finished_at": time.time(),
        # 切帧**永远整段**（方案 B：帧索引必须完整，成本也极低）；这里记的 range 是
        # 「frames 覆盖了 1..idx」，后五个阶段各自记自己真正处理的那一段。
        "range": {"start": 1, "end": int(idx)},
        "range_key": "1-%d" % int(idx),
        "range_requested": {"start": int(rng_f["start"]), "end": int(rng_f["end"]),
                            "total": int(rng_f["total"])},
        "msg": "切帧完成：%d 帧" % idx,
        # 帧图的真实编码参数（验收要能一眼看到，不再靠"默认值"猜）
        # png 是无损：ffmpeg 走 -compression_level 1（快、略大）；兜底的进程内那条
        # 走 cv2.imencode PNG level 3。两条都是无损，差别只在文件大小/耗时。
        "format": "png", "extractor": str(cfg.get("_frames_via") or "opencv"),
        "png_compression": (1 if cfg.get("_frames_via") == "ffmpeg" else 3),
        "jpeg_quality": 100,
        # jpg 帧走 4:4:4（cv2 5.0 起必须用枚举，见 _stage_imencode）；写不出 4:4:4 的
        # 机器由 _align.jpeg444_probe 启动时 WARN 一次，落成库默认的 4:2:0。
        "jpeg_subsampling": "4:4:4（仅当 image_format=jpg 时才是 jpg）",
        "width": w, "height": h,
    }
    meta["frame_swapped"] = False
    _stage_meta_save(cache, meta)
    cfg["total"] = idx
    _stage_log("切帧完成：%d 帧 · %dx%d · %.2f s" % (idx, w, h, sec))
    return {"count": idx, "seconds": sec, "frames": idx}


def _stage_frames_inproc(cfg: dict, frames_dir, meta, w, h) -> int:
    """兜底：进程内解码 + 逐帧 PNG（ffmpeg 抽帧不可用时走这条）。"""
    import numpy as np

    video = Path(cfg["video"])
    total_hint = int(cfg.get("total") or 0)
    t0 = time.time()
    idx = 0
    try:
        gen, mi = _app_mod._open_video(str(video), [], first=False)
    except Exception as e:                                  # noqa: BLE001
        raise _BadInput("这个视频解不开：%s（%s）" % (video.name, str(e)[:160]), 422)
    try:
        size = (mi or {}).get("size") or (w, h)
        w, h = int(size[0]), int(size[1])
        while True:
            _stage_check_cancel()
            try:
                raw = next(gen)
            except StopIteration:
                break
            except Exception as e:                          # noqa: BLE001
                raise _BadInput("解码到第 %d 帧失败：%s" % (idx + 1, str(e)[:160]), 500)
            frame = np.frombuffer(raw, np.uint8).reshape((h, w, 3)).copy()
            _stage_tick("frames")        # 交接点：解码 → frames/*.png（下一段从这里读）
            _stage_write_atomic(frames_dir / _stage_frame_name(idx),
                                _stage_imencode(frame, ".png"))
            idx += 1
            el = max(0.001, time.time() - t0)
            total = max(total_hint, idx)
            _stage_progress("frames", 100.0 * idx / total if total else 0,
                            "切帧 %d/%s 帧 · %.1f 帧/秒" % (idx, total_hint or "?", idx / el),
                            frames_done=idx, frames_total=total,
                            frame_fps=round(idx / el, 2))
    finally:
        try:
            gen.close()
        except Exception:                                   # noqa: BLE001
            pass
    return idx



# ---------------- 切脸流水线：读帧 ‖ 检测 ‖ 后处理 ----------------
# 为什么要有这一段：切脸原来是「读一帧 → 检测 → 2D106 → 裁剪 → 编码写盘」一条直线。
# 1080x1920 的 PNG 实测（40 帧平均，retinaface_10g）：
#     读帧 19.0ms（47%）· 检测 17.8ms（44%）· 2D106 1.2ms · 裁剪 0.3ms · 编码写盘 0.6ms
# 串起来 40ms/帧 = 25 帧/秒；而这三段用的是完全不同的资源（磁盘+解码 / GPU / CPU），
# 串行等于每一段都在等另外两段。拆成三段重叠之后，单帧成本 ≈ 三段里最慢的那一段。
_FACES_END = object()                       # 流水线哨兵：这一条队列到头了
_FACES_READ_ENV = "DFL_FACES_READ_THREADS"  # 读帧线程数（1..12）
_FACES_QUEUE_ENV = "DFL_FACES_QUEUE"        # 读帧队列深度（在飞的帧数）
_FACES_BATCH_ENV = "DFL_FACES_BATCH"        # 检测批次（0 = 自动）


class _StageTicks:
    """阶段内计时（多线程安全）。profile 表就是从这里出去的。"""

    def __init__(self):
        self._d = {}
        self._lk = threading.Lock()

    def add(self, k: str, dt: float):
        with self._lk:
            self._d[k] = self._d.get(k, 0.0) + float(dt)

    def total(self, k: str) -> float:
        with self._lk:
            return float(self._d.get(k, 0.0))

    def ms(self, k: str, n: int) -> float:
        with self._lk:
            v = self._d.get(k, 0.0)
        return round(1000.0 * v / max(1, int(n)), 2)


def _stage_faces_int(env: str, default: int, lo: int, hi: int) -> int:
    try:
        raw = os.environ.get(env)
        v = int(str(raw).strip()) if (raw is not None and str(raw).strip()) else int(default)
    except (TypeError, ValueError):
        v = int(default)
    return max(int(lo), min(int(hi), v))


def _stage_faces_read_threads() -> int:
    """读帧线程数。PNG 解码在 cv2 里会放开 GIL，所以多线程是真的并行。

    实测（1080x1920 PNG，24 帧平均）：单线程 20.2ms / 2 线程 10.8ms / 6 线程 4.96ms /
    8 线程 4.34ms。默认取 6：再往上每加一个线程只省零点几毫秒，却要跟 2D106 抢 CPU。
    """
    cpu = int(os.cpu_count() or 4)
    return _stage_faces_int(_FACES_READ_ENV, min(6, max(2, cpu // 4)), 1, 12)


def _stage_faces_queue(read_threads: int) -> int:
    """读帧队列深度（同时最多几张帧在飞）。一张 1080x1920 BGR 是 6.2MB。"""
    return _stage_faces_int(_FACES_QUEUE_ENV, max(4, read_threads * 2), 2, 32)


def _stage_faces_batchable(det) -> bool:
    """这个检测器能不能真批量。只看它自己声明的能力，不看配置。"""
    return bool(hasattr(det, "detect_batch") and getattr(det, "needs_canvas", False))


def _stage_faces_batch(det) -> int:
    """一次送几张进检测器（1 = 逐帧）。

    能不能批量由**模型的 onnx 输入签名**决定，这里不猜：
      · TinyMog        input [batch_size, 3, 640, 640]        → 动态 batch，可以；
      · RetinaFace_10g input [1, 3, ?, ?]                     → batch 维写死 1
        （只有 H/W 是动态的），硬塞 batch 只会在 ORT 里报 shape 错，所以恒定逐帧。
    引擎侧还有一层上界（TrtSession.max_batch）：超过它 batch 反而不走 CUDA Graph。
    """
    if not _stage_faces_batchable(det):
        return 1
    # 默认逐帧：**批量在这台机器上是负收益**，实测（整条片子 1439 帧，tinymog）——
    #   逐帧   detect 5.06 ms/帧 → 75.9 帧/秒（单帧走 CUDA Graph replay）
    #   批次 4 detect 9.04 ms/帧 → 69.7 帧/秒（batch>1 走 numpy 路径，图重放用不上）
    # 所以批量做成开关（DFL_FACES_BATCH=4 之类），不默认开。
    want = _stage_faces_int(_FACES_BATCH_ENV, 1, 1, 16)
    if want <= 1:
        return 1
    cap = None
    for sess in (getattr(det, "sess_batch", None), getattr(det, "sess", None)):
        if sess is None:
            continue
        try:
            cap = int(sess.max_batch())
        except Exception:                                      # noqa: BLE001
            cap = None
        if cap:
            break
    if cap is not None:
        if cap <= 1:
            return 1
        want = min(want, cap)
    return max(1, want)


def _stage_faces_canvas(frame, det):
    """按检测器自己的 letterbox 口径把整帧缩进 S×S 画布（批量检测用）。

    这与 TinyMogDet.detect() 里那三行逐字一致（128 填充 + INTER_LINEAR），
    所以批量与逐帧的输入像素相同。
    """
    import cv2
    import numpy as np

    S = int(getattr(det, "input_size", 640) or 640)
    H, W = frame.shape[:2]
    scale = min(S / float(W), S / float(H))
    nw = max(1, int(round(W * scale)))
    nh = max(1, int(round(H * scale)))
    canvas = np.full((S, S, 3), 128, np.uint8)
    canvas[:nh, :nw] = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_LINEAR)
    return canvas, scale


def _stage_faces_detect(det, imgs, thresh):
    """一批帧 → 一批框（与输入等长、同序）。

    batch = 1 或检测器不支持动态 batch 时，逐帧走 det.detect() —— 和优化前**同一条
    调用路径**，结果逐位一致；只有真支持批量时才走 detect_batch，跑挂了就当场退回逐帧。
    """
    if len(imgs) <= 1 or not _stage_faces_batchable(det):
        return [det.detect(im) for im in imgs]
    items = []
    for im in imgs:
        canvas, scale = _stage_faces_canvas(im, det)
        items.append((im, canvas, scale))
    try:
        return det.detect_batch(items)
    except Exception as e:                                     # noqa: BLE001
        _stage_log("切脸：批量检测失败（%s）→ 这一批退回逐帧" % (str(e)[:120],))
        return [det.detect(im) for im in imgs]


def _stage_do_faces(cfg: dict) -> dict:
    """阶段 2 · 切脸：frames/ -> aligned/000001_00.jpg（DFLJPG：original_name + 68 点 + face_mat）。"""
    import cv2
    import numpy as np
    from ._align import (COARSE_SIZE, align_face_68, coarse_align_from_5,
                         landmarks68_in_source, pose_from_kps, write_dfljpg)

    cache = Path(cfg["cache"])
    meta = cfg["meta"]
    # 流水线实况：切脸 = 1 个检测 session（读线程不算 lane，它们不产 bit）
    _STAGE_METER.set_lanes({"det": ["det#0"]})
    st = meta["stages"].setdefault("faces", {})
    frames = _stage_files(_stage_frames_dir(cache))
    if not frames:
        raise _BadInput("frames/ 里没有帧图 —— 先跑切帧", 409)
    aligned_dir = _stage_aligned_dir(cache)
    # 帧范围：只处理 [start, end] 里的帧（帧索引仍是全局的 —— aligned/ 的命名不许挪位）
    rng = _stage_range(cfg)
    rkey = str(cfg.get("range_key") or rng.get("key") or "")
    rows = [i for i, p in enumerate(frames) if _stage_in_range(rng, p.stem)]
    total = len(rows)                     # 范围内的帧数（游标从这里数到 0）
    have = len(_stage_files(aligned_dir, (".jpg", ".jpeg")))

    if ((not cfg["force"]) and not cfg.get("range_changed") and st.get("done") and have
            and int(st.get("frames") or 0) == len(frames)):
        _stage_log("切脸：复用已有 %d 张对齐脸（第 %d..%d 帧，不重跑）"
                   % (have, int(rng["start"]), int(rng["end"])))
        return {"skipped": True, "count": have, "reason": "already-done"}

    if cfg["force"]:
        n = _stage_clear_dir(aligned_dir, (".jpg", ".jpeg"))
        _stage_log("切脸：force=true，清掉旧的 %d 张对齐脸" % n)
    aligned_dir.mkdir(parents=True, exist_ok=True)

    det_name = str(cfg.get("det_align") or "retinaface_10g")
    thresh = float(cfg.get("align_thresh") or 0.5)
    lm = _landmarks_model()          # 和预览/合成同款：纯 CPU onnxruntime 的 2D106
    if lm is None:
        raise _BadInput("2D106 关键点模型不可用（weights/insightface_2d106.onnx）—— "
                        "没有 68 点就不写对齐脸", 500)

    # 游标 **相对范围**：0..total（total = end-start+1），换范围时旧游标（相对旧范围）作废
    cursor = 0 if (cfg["force"] or cfg.get("range_changed")) else int(st.get("done_frames") or 0)
    cursor = max(0, min(cursor, total))
    faces_total = 0 if (cfg["force"] or cfg.get("range_changed")) else int(st.get("count") or 0)
    with_faces = 0 if (cfg["force"] or cfg.get("range_changed")) else int(st.get("frames_with_faces") or 0)
    skipped = int(st.get("skipped") or 0)
    if cursor:
        _stage_log("切脸：从第 %d 帧续跑（前 %d 帧的对齐脸已经在）"
                   % (int(rng["start"]) + cursor, cursor))
    _stage_log("切脸：检测器 %s（%s）· thresh=%.2f · 68 点仿射 · 帧范围 %d..%d（%d 帧）"
               % (det_name, str(cfg.get("det_align_src") or "cfg.det_align"), thresh,
                  int(rng["start"]), int(rng["end"]), total))
    # 对齐脸是这条链上唯一的 JPEG 产物：质量回显出来（验收/排障要能一眼看到，
    # 别再靠"默认值"猜）。默认 100 + 4:4:4；PNG 帧完全不受这个数影响。
    _stage_log("切脸：对齐脸 JPEG quality=%d · 4:4:4（界面无旋钮，固定质量）"
               % int(cfg.get("jpeg_quality") or _STAGE_JPEG_QUALITY))
    # 检测器先建好（加载要几百毫秒）：流水线里只有一个线程用它，不再每帧走 _detector()
    det = _detector(det_name)

    t0 = time.time()

    def _save_progress(up_to: int, done: bool = False):
        meta["stages"]["faces"] = {
            "done": bool(done), "done_frames": int(up_to), "frames": len(frames),
            "total": int(total), "range": {"start": int(rng["start"]), "end": int(rng["end"])},
            "range_key": rkey, "range_total": int(total),
            "count": faces_total, "frames_with_faces": with_faces, "skipped": skipped,
            "detector": det_name, "align": "68", "thumb": "2x box",
            "msg": ("切脸完成：%d 张脸 / %d 帧（第 %d..%d 帧）"
                    % (faces_total, total, int(rng["start"]), int(rng["end"]))) if done
                   else ("切脸 %d/%d 帧（第 %d..%d 帧）"
                         % (up_to, total, int(rng["start"]), int(rng["end"]))),
        }
        _stage_meta_save(cache, meta)

    # ---- 流水线：读帧线程池 → 检测线程 → 本线程做 2D106 / 裁剪 / 编码写盘 ----
    # 三段用的资源互不相干（磁盘+解码 / GPU / CPU），串行做等于每段都在等另外两段。
    from concurrent.futures import ThreadPoolExecutor

    nread = _stage_faces_read_threads()
    depth = _stage_faces_queue(nread)
    batch_n = _stage_faces_batch(det)
    ticks = _StageTicks()
    count0 = faces_total
    _stage_log("切脸流水线：读帧 %d 线程 · 队列 %d · 检测批次 %d%s"
               % (nread, depth, batch_n,
                  "" if batch_n > 1 else "（%s 的 onnx 输入 batch 维固定=1 → 逐帧）" % det_name))

    read_q = queue.Queue(maxsize=depth)     # (pos, frame, err)；满了读帧线程就等，内存可控
    det_q = queue.Queue()                   # (pos, frame, boxes) / ("err", None, 异常)
    stop = threading.Event()
    pool = ThreadPoolExecutor(max_workers=nread, thread_name_prefix="faces-read")

    def _read_job(path):
        t = time.perf_counter()
        try:
            return _stage_read_image(path), None
        except Exception as e:                                 # noqa: BLE001
            return None, e
        finally:
            ticks.add("read", time.perf_counter() - t)

    def _q_put(q, item):
        while not stop.is_set():
            try:
                q.put(item, timeout=0.2)
                return True
            except queue.Full:
                continue
        return False

    def _feeder():
        """按帧序读图：最多 depth 张在飞，读出来的帧立即进 read_q。

        注意「读完了」和「在飞的都取完了」是两件事：帧迭代器耗尽后 pending 里还压着
        最多 depth 张，必须继续取完再收尾（曾经在这里直接 break，末尾 12 帧被静默丢掉，
        阶段照样报 40/40 完成 —— 所以主循环末尾还有一道 consumed 对账）。
        """
        pending = collections.deque()
        it = iter(range(cursor, total))
        exhausted = False
        try:
            while True:
                while (not exhausted) and (not stop.is_set()) and len(pending) < depth:
                    try:
                        pos = next(it)
                    except StopIteration:
                        exhausted = True
                        break
                    try:
                        pending.append((pos, pool.submit(_read_job, frames[rows[pos]])))
                    except RuntimeError:            # 取消时主线程已经 shutdown 了线程池
                        exhausted = True
                        break
                if not pending:
                    break
                pos, fut = pending.popleft()
                try:
                    img, err = fut.result()
                except Exception as e:                         # noqa: BLE001
                    img, err = None, e
                if not _q_put(read_q, (pos, img, err)):
                    return
        except Exception:                       # 收尾阶段的竞态不该把线程栈打到控制台
            pass
        finally:
            _q_put(read_q, _FACES_END)

    def _detect_worker():
        """从 read_q 取帧、按 batch_n 攒一批送检测器，结果按帧序进 det_q。"""
        ended = False
        try:
            while True:
                item = read_q.get()
                if item is _FACES_END:
                    ended = True
                    item = None
                group = []
                if item is not None:
                    if item[2] is not None:
                        det_q.put(item)
                        continue
                    group.append(item)
                while (not ended) and len(group) < batch_n:
                    try:
                        nxt = read_q.get_nowait()
                    except queue.Empty:
                        break
                    if nxt is _FACES_END:
                        ended = True
                        break
                    if nxt[2] is not None:
                        det_q.put(nxt)
                        continue
                    group.append(nxt)
                if group:
                    _stage_check_cancel()
                    imgs = [g[1] for g in group]
                    t = time.perf_counter()
                    try:
                        with _det_run_lock:                     # thresh 是共享检测器上的字段
                            det.thresh = float(thresh)
                            got = _stage_faces_detect(det, imgs, thresh)
                    except _StageCancel:
                        raise
                    except Exception as e:                      # noqa: BLE001
                        raise _BadInput("检测失败：%s" % (str(e)[:200],), 500)
                    finally:
                        ticks.add("detect", time.perf_counter() - t)
                    for g, boxes in zip(group, got):
                        det_q.put((g[0], g[1], boxes))
                if ended:
                    return
        except BaseException as e:                              # noqa: BLE001
            det_q.put(("err", None, e))
        finally:
            det_q.put(_FACES_END)

    feeder_th = threading.Thread(target=_feeder, name="faces-feed", daemon=True)
    det_th = threading.Thread(target=_detect_worker, name="faces-detect", daemon=True)
    feeder_th.start()
    det_th.start()

    consumed = 0
    try:
        while True:
            item = det_q.get()
            if item is _FACES_END:
                break
            pos, frame, extra = item
            if pos == "err":
                raise extra
            if frame is None:
                raise _BadInput("这一帧读不出来：%s" % frames[rows[pos]], 500)
            if isinstance(extra, BaseException):
                raise extra
            boxes = sorted(list(extra or []),
                           key=lambda b: float((b.get("box") or [0, 0, 0, 0])[0]))
            n = 0
            for k, b in enumerate(boxes):
                _stage_check_cancel()
                box = b.get("box") or [0, 0, 0, 0]
                x1, y1, x2, y2 = [float(v) for v in box]
                bw = max(1.0, x2 - x1)
                bh = max(1.0, y2 - y1)
                side = int(round(max(bw, bh) * 2.0))
                side -= side % 2
                kps = b.get("kps") or []
                if len(kps) < 5 or side < 32:
                    skipped += 1
                    continue
                try:
                    cmat = coarse_align_from_5(kps)
                    coarse = cv2.warpAffine(frame, cmat, (COARSE_SIZE, COARSE_SIZE),
                                            flags=cv2.INTER_LINEAR,
                                            borderMode=cv2.BORDER_CONSTANT,
                                            borderValue=(0, 0, 0))
                    t_lm = time.perf_counter()
                    lm68a = lm.landmarks68(coarse)
                    ticks.add("lm68", time.perf_counter() - t_lm)
                    if lm68a is None:
                        raise ValueError("2D106 没有输出")
                    pts = landmarks68_in_source(lm68a, cmat)
                    t_cr = time.perf_counter()
                    crop, meta_f = align_face_68(frame, np.asarray(pts, np.float64),
                                                 box=[int(round(x1)), int(round(y1)),
                                                      int(round(x2)), int(round(y2))],
                                                 size=side)
                    ticks.add("crop", time.perf_counter() - t_cr)
                except Exception as e:                          # noqa: BLE001
                    skipped += 1
                    _stage_log("切脸跳过 %s 第 %d 张：%s" % (frames[rows[pos]].name, k,
                                                          str(e)[:120]))
                    continue
                if crop is None or not meta_f:
                    skipped += 1
                    _why = (meta_f or {}).get("reject_reason") if isinstance(meta_f, dict) else ""
                    if _why:
                        _stage_log("切脸拒收 %s 第 %d 张：%s"
                                   % (frames[rows[pos]].name, k, _why))
                    continue
                # 配对键 / 68 点 / face_mat 三样都写进 DFLJPG 的 APP15
                meta_f.update({
                    "original_name": frames[rows[pos]].name,
                    "source_filename": frames[rows[pos]].stem,
                    "face_mat": meta_f.get("image_to_face_mat"),
                    "landmarks_68": meta_f.get("landmarks"),
                    "source_landmarks_68": meta_f.get("source_landmarks"),
                    "frame_index": int(rows[pos]),
                    "frame_name": frames[rows[pos]].name, "face_index": int(k),
                    "detector": det_name, "side": int(side),
                })
                try:
                    yaw, pitch, roll = pose_from_kps(kps, box)
                    if yaw is not None:
                        meta_f["yaw"], meta_f["pitch"], meta_f["roll"] = yaw, pitch, roll
                except Exception:                               # noqa: BLE001
                    pass
                _stage_tick("det")       # 交接点：切脸 → aligned/*.jpg（下一段从这里读）
                t_en = time.perf_counter()
                write_dfljpg(aligned_dir / _stage_face_name(rows[pos], k), crop, meta_f,
                             quality=int(cfg.get("jpeg_quality") or _STAGE_JPEG_QUALITY),
                             subsampling=0)
                ticks.add("encode", time.perf_counter() - t_en)
                n += 1
            faces_total += n
            if n:
                with_faces += 1
            el = max(0.001, time.time() - t0)
            _stage_progress("faces", 100.0 * (pos + 1) / total,
                            "切脸 %d/%d 帧（第 %d..%d 帧）· %d 张脸 · %.2f 帧/秒"
                            % (pos + 1, total, int(rng["start"]), int(rng["end"]),
                               faces_total, (pos + 1 - cursor) / el),
                            frames_done=pos + 1, frames_total=total,
                            frame_fps=round((pos + 1 - cursor) / el, 2))
            consumed += 1
            if (pos + 1) % 20 == 0:
                _save_progress(pos + 1)
    finally:
        stop.set()
        pool.shutdown(wait=False)
        det_th.join(timeout=1.5)
        feeder_th.join(timeout=1.5)

    sec = round(time.time() - t0, 2)
    done_n = max(0, total - cursor)
    # 对账：流水线少喂了帧就等于少切了脸，绝不能把「跑到 30/40」报成「40/40 完成」
    if consumed != done_n:
        raise _BadInput("切脸流水线提前结束：只有 %d/%d 帧走完了检测（产物不完整，"
                        "重跑 force=true）" % (consumed, done_n), 500)
    fps = float(done_n) / max(1e-6, sec)
    # 三段是**并行**的，所以「每帧在各段上花的时间」加起来会大于每帧墙钟：
    #   read     = 读图总耗时 / 读帧线程数（摊到单帧、按并行度折算）
    #   others   = 每帧真的花在那一段上的时间
    #   stage_util = 这一段占用了整段墙钟的百分之多少（>100% = 多线程并行）
    post = ticks.total("lm68") + ticks.total("crop") + ticks.total("encode")
    prof = {"frames": int(done_n), "faces": int(faces_total - count0), "seconds": sec,
            "fps": round(fps, 2),
            "per_frame_ms": {"read": round(1000.0 * ticks.total("read") / max(1, nread)
                                           / max(1, done_n), 2),
                             "detect": ticks.ms("detect", done_n),
                             "lm68": ticks.ms("lm68", done_n),
                             "crop": ticks.ms("crop", done_n),
                             "encode": ticks.ms("encode", done_n)},
            "stage_util": {"read": round(100.0 * ticks.total("read") / max(1e-6, sec), 1),
                           "detect": round(100.0 * ticks.total("detect") / max(1e-6, sec), 1),
                           "post": round(100.0 * post / max(1e-6, sec), 1)},
            "lm68_ms_per_face": ticks.ms("lm68", max(1, faces_total - count0)),
            "pipeline": {"read_threads": int(nread), "queue": int(depth),
                         "detect_batch": int(batch_n),
                         "batchable": bool(_stage_faces_batchable(det)),
                         "overlap": "read || detect || post"}}
    # 谁在拖后腿：按「占用墙钟的比例」比，读帧那一路要除以线程数
    # （read 的 stage_util 是 6 个线程之和，直接比会把多线程误判成瓶颈）
    eff = {"read": prof["stage_util"]["read"] / max(1, int(nread)),
           "detect": prof["stage_util"]["detect"],
           "post": prof["stage_util"]["post"]}
    prof["stage_share"] = {k: round(v, 1) for k, v in eff.items()}
    prof["bottleneck"] = max(eff.items(), key=lambda kv: kv[1])[0] if eff else ""
    prof["per_frame_total_ms"] = round(1000.0 * sec / max(1, done_n), 2)
    prof["per_frame_ms"]["other"] = round(max(
        0.0, prof["per_frame_total_ms"] - sum(prof["per_frame_ms"].values())), 2)
    _save_progress(total, done=True)
    meta["stages"]["faces"]["seconds"] = sec
    meta["stages"]["faces"]["fps"] = round(fps, 2)
    meta["stages"]["faces"]["profile"] = prof
    _stage_meta_save(cache, meta)
    _stage_log("切脸完成：第 %d..%d 帧（%d 帧）· %d 张脸（%d 帧有脸）· 跳过 %d · %.2f s · "
               "%.2f 帧/秒（读 %s / 检测 %s / 2D106 %s / 裁剪 %s / 写盘 %s ms，"
               "读帧 %d 线程 · 批次 %d）"
               % (int(rng["start"]), int(rng["end"]), total, faces_total,
                  with_faces, skipped, sec, fps,
                  prof["per_frame_ms"]["read"], prof["per_frame_ms"]["detect"],
                  prof["per_frame_ms"]["lm68"], prof["per_frame_ms"]["crop"],
                  prof["per_frame_ms"]["encode"], nread, batch_n))
    return {"count": faces_total, "frames": total, "skipped": skipped, "seconds": sec,
            "fps": round(fps, 2), "profile": prof,
            "range": {"start": int(rng["start"]), "end": int(rng["end"])}}


# ---------------- ③ 换脸（dfm 推理）----------------
# 产物（和 4.0 的 work 目录一一对应，export_pipeline.py:769/846/856 那三个名字）：
#   pred/000001_00.png        换脸后的正脸（无损 PNG，S×S，S = dfm 输入边长）
#   pred/000001_00_prdm.png   dfm 自带的 prd 遮罩（灰色 PNG，0..255）
#   pred/000001_00_dstdm.png  dfm 自带的 dst 遮罩
# 这一阶段**只做推理**：不碰遮罩组合、不碰合成、不写 frames/。
_PRED_LOCK = threading.RLock()
_DFM_SESS = {}          # dfm -> (sess, input_name, size, nchw, providers)
_INFER_CALLS = collections.Counter()   # 各阶段真正跑了几次推理（验收要断言合成阶段 = 0）


def _stage_pred_dir(cache) -> Path:
    return Path(cache) / "pred"


def _stage_mask_dir(cache) -> Path:
    return Path(cache) / "mask"


def _dfm_engine_name(dfm) -> str:
    r"""dfm → TRT 引擎名（稳定、纯 ASCII）。

    为什么必须 ASCII：TensorRT 的 C++ 侧用 stat() 打开 onnx，非 ASCII 路径直接
    INVALID_VALUE（实测 "阿証証証証-416.dfm" 就过不去），所以引擎名里不能留中文。

    编引擎的办法（没有引擎就自动回落 ort-CUDA，功能不受影响，只是慢 16%）：
        :: 中文模型名先把 onnx 拷成 ASCII 再喂给 builder
        copy "workspace\model\<模型>.dfm" "%TEMP%\dfm.onnx"
        runtime\server\Scripts\python.exe app\server\_trt_build.py ^
            --onnx "%TEMP%\dfm.onnx" --out "workspace\_engines\<本函数的名字>_<md5>_sm<SM>_trt<TRT>.bf16.engine" ^
            --name <本函数的名字> --bf16 --profile "1,416,416,3;4,416,416,3;8,416,416,3"
    引擎文件名里的 md5/SM/TRT 三段必须是这台机器的真实值（[TRT] 那行日志会直接
    打印它在找哪个文件名，照抄即可）。
    """
    stem = Path(str(dfm)).stem
    key = re.sub(r"[^0-9A-Za-z_.-]+", "_", stem).strip("._-") or "dfm"
    return "dfm_" + key[:40]


def _sess_batch_cap(sess) -> int:
    """这个 session 一次前向最多能塞几张（1 = 只能逐张）。"""
    try:
        mb = getattr(sess, "max_batch", None)
        if callable(mb):
            return max(1, int(mb() or 1))            # 原生 TRT：按优化档位
    except Exception:                                   # noqa: BLE001
        return 1
    try:
        shape = list(sess.get_inputs()[0].shape)
        if shape and isinstance(shape[0], int) and shape[0] > 0:
            return int(shape[0])
    except Exception:                                   # noqa: BLE001
        pass
    return 8                                            # 动态 batch（ort）


# ---------------------------------------------------------------
# dfm 的推理后端开关：**默认只用 onnxruntime（CUDA EP / fp32）**
# ---------------------------------------------------------------
# 用户硬规则：**dfm 只走 onnxruntime，不编 TRT 引擎、不降精度。**
# 所以 .engine 对 dfm 是「默认关」的一档加速：磁盘上就算躺着 dfm 的引擎，
# 也要按开关忽略，并且**明确喊一行日志**（禁止静默换后端）。
# 想回到旧行为（原生 TRT 优先）做对照实验，就显式设 DFL_DFM_TRT=1。
# 注意范围：这个开关**只管 dfm**。XSeg / 检测 / 2D106 的 TRT 不受影响
# （它们的入口是 _stage_xseg / _xseg.py，压根不读这两个变量）。
_DFM_TRT_ENV = "DFL_DFM_TRT"          # 0（默认）= 禁；1 = 允许原生 TRT 引擎
_DFM_ENGINE_ENV = "DFL_DFM_ENGINE"    # 老变量，语义同上（只为兼容显式写 1 的人）
_DFM_TRT_ON = ("1", "true", "yes", "on", "trt", "engine")


def _dfm_trt_enabled() -> bool:
    """dfm 能不能用 .engine —— **默认不能**（用户要求 ONNX only）。"""
    for k in (_DFM_TRT_ENV, _DFM_ENGINE_ENV):
        v = str(os.environ.get(k) or "").strip().lower()
        if v in _DFM_TRT_ON:
            return True                          # 显式要求，尊重
        if v:
            return False                         # 显式关（"0"）压过后面那个变量
    return False                                 # 谁都没设 → 关（ONNX only）


def _dfm_trt_why() -> str:
    """一句话说明「现在是按哪个变量关的」，给日志用。"""
    for k in (_DFM_TRT_ENV, _DFM_ENGINE_ENV):
        v = str(os.environ.get(k) or "").strip()
        if v:
            return "%s=%s" % (k, v)
    return "%s=0（默认）" % _DFM_TRT_ENV


def _dfm_mb(n) -> str:
    try:
        return "%.1fMB" % (float(n) / (1024.0 * 1024.0))
    except Exception:                                   # noqa: BLE001
        return "?MB"


def _dfm_engines_on_disk(ws, name: str) -> list:
    """磁盘上属于这个 dfm 的 .engine（不校验 md5/SM，只为「发现了但忽略」那行日志）。

    按体积从大到小：bf16 引擎几百 MB，int8 的几十 MB，用户看到的最大那个才是重点。
    """
    out = []
    try:
        d = Path(ws) / "_engines"
        if not d.is_dir():
            return out
        out = list(d.glob("%s_*.engine" % name))
        out.sort(key=lambda p: p.stat().st_size, reverse=True)
    except Exception:                                   # noqa: BLE001
        return []
    return out


def _dfm_session(dfm: str):
    """常驻的 dfm session：**默认 ort-CUDA（fp32）**；CPU 只在真装不下时（且喊出来）。

    用户硬规则：**dfm 只走 onnxruntime，不编 TRT 引擎、不降精度。**
    所以判定顺序是：

        DFL_DFM_TRT=1（显式要求，仅供对照实验）→ 原生 TRT 引擎 → ort-CUDA
        DFL_DFM_TRT=0（**默认**）              → ort-CUDA（磁盘上的 .engine 忽略 + 喊一行）
        显存装不下 + DFL_ALLOW_CPU_FALLBACK=1  → CPU EP（慢 ~17 倍，日志写死）

    历史背景（留给想看数的人）：416 dfm 上 ort-CUDA 35.9 ms/张（27.9 张/秒）、
    原生 TRT(bf16) 30.9 ms/张（32.3 张/秒）—— 那 ~16% 就是换掉 ONNX 的代价，
    按用户口径**不换**：这条链上 dfm 永远是 onnxruntime，日志里 provider 列表为准。

    唯一的硬要求是**不能静默退到 CPU**：CPU 单张 527 ms（1.9 张/秒），而 ORT 在
    CUDA provider 加载失败时只会打一行 warning 就继续跑，所以这里自己把
    providers 拿出来核对、打醒目日志。

    session 按路径常驻，切模型才重建。
    """
    key = str(dfm)
    if _vram_oom_inject_on():
        # 验收专用（DFL_ALLOW_TEST_HOOK=1 + DFL_VRAM_OOM_INJECT=1）：**在真的加载之前**
        # 抛一次「CUDA out of memory」，把「强制尝试撞 OOM → 回滚 → 还能再试」这条路
        # 变成可复现的。为什么需要它：WDDM 会把别的进程的显存换页出去，本机实测
        # 「空闲 600MB + force=1」竟然还能跑完 12 帧（慢，但不 OOM）——真 OOM 只能
        # 靠把卡压到连提交都失败，那在 32GB 内存的机器上做不到。
        # 生产路径上这两个开关都没人设，一个字节都不走。
        raise RuntimeError("CUDA out of memory. Tried to allocate 2079 MiB"
                           " (DFL_VRAM_OOM_INJECT=1 注入的 OOM，验收用)")
    cpu = _vram_cpu_on()            # 显存不够 + DFL_ALLOW_CPU_FALLBACK=1 → 整条链走 CPU EP
    ckey = (key + "|cpu") if cpu else key
    with _PRED_LOCK:
        hit = _DFM_SESS.get(ckey)
        if hit is not None:
            return hit
        import onnxruntime as ort
        try:
            _ensure_ort_dll_dirs()
        except Exception:                                   # noqa: BLE001
            pass
        # 量这一份会话到底吃掉多少显存（权重 + 这个进程的 CUDA context + ORT arena）。
        # 为什么要量：arena 会长（384 模型：建会话 +2.3GB，之后每批 8 张的前向再涨 ~1GB），
        # 估出来的数偏小 —— 而「显存去向」那一栏必须是真数，不能是拍的。
        _own_f0 = _vram_measure_begin()
        ws = Path(WORKSPACE)
        info = {}
        sess = None
        backend = "ort-cuda"
        if cpu:
            info = {"reason": "DFL_ALLOW_CPU_FALLBACK=1（显存预算装不下，主动回落 CPU）"}
        elif not _dfm_trt_enabled():
            # 用户硬规则：**dfm 只用 onnxruntime**。引擎在磁盘上也不加载 —— 但必须喊出来，
            # 否则「A/B 报告里 416 其实是 TRT 的数字」这种事下次还会悄悄发生。
            engs = _dfm_engines_on_disk(ws, _dfm_engine_name(key))
            if engs:
                more = ("（另有 %d 份同类引擎一起忽略）" % (len(engs) - 1)) if len(engs) > 1 else ""
                _stage_log("[dfm] 发现 TRT 引擎 %s（%s）但按 %s忽略，使用 onnxruntime CUDA%s"
                           % (engs[0].name, _dfm_mb(engs[0].stat().st_size),
                              _dfm_trt_why(), more))
            else:
                _stage_log("[dfm] 按 %s走 ONNX only（磁盘上没有 dfm 的 .engine）"
                           % _dfm_trt_why())
            info = {"reason": "%s —— 按用户要求 dfm 只走 onnxruntime，"
                              "磁盘上有 .engine 也不加载" % _dfm_trt_why()}
            sess = None
        else:
            try:
                from . import _engines as _E
                sess = _E.try_engine(ws, _dfm_engine_name(key), key,
                                     precision=("bf16",), use_graph=False, info=info)
            except Exception as e:                      # noqa: BLE001
                info = {"reason": "try_engine 抛异常：%r" % (e,)}
                sess = None
        if cpu:
            backend = "ort-cpu-forced"
            so = ort.SessionOptions()
            so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            sess = ort.InferenceSession(key, sess_options=so,
                                        providers=["CPUExecutionProvider"])
            shape = list(sess.get_inputs()[0].shape)
            name = sess.get_inputs()[0].name
            provs = list(sess.get_providers())
            _stage_log("换脸：★ 当前在 CPU 上跑 ★ dfm=%s 走 CPU EP —— 显存不够且开了 "
                       "DFL_ALLOW_CPU_FALLBACK=1（416 实测 527 ms/张，GPU 约 31 ms/张，"
                       "慢 %.0f 倍）。关掉这个开关就改成明确拒绝。"
                       % (Path(key).name, _VRAM_CPU_SLOW))
        elif sess is not None:
            backend = "trt"
            sess.precision = info.get("precision") or ""
            shape = list(sess.get_inputs()[0].shape)
            name = sess.get_inputs()[0].name
            provs = ["TensorrtExecutionProvider"]
            _stage_log("换脸：dfm 走原生 TRT(%s) 引擎 %s"
                       % (info.get("precision") or "?", Path(str(info.get("engine") or "")).name))
        else:
            so = ort.SessionOptions()
            so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            sess = ort.InferenceSession(key, sess_options=so,
                                        providers=["CUDAExecutionProvider",
                                                   "CPUExecutionProvider"])
            shape = list(sess.get_inputs()[0].shape)
            name = sess.get_inputs()[0].name
            provs = list(sess.get_providers())
            if provs[:1] == ["CPUExecutionProvider"]:
                backend = "ort-cpu"
                _stage_log("换脸：★ 警告 ★ dfm 掉到 CPU 了（%s）—— 这张卡上会慢 ~15 倍"
                           "（416 实测 527 ms/张）。查 cublasLt64_12.dll / cudnn64_9.dll "
                           "是否在 PATH 上（torch/lib）；dfm 的 TRT 只能靠"
                           " DFL_DFM_TRT=1 显式打开（默认 ONNX only）。"
                           % (info.get("reason") or "CUDA provider 没加载上"))
            else:
                # 验收要的证据就在这一行：provider 列表 = 真后端，不是"听说用了 GPU"
                _stage_log("换脸：dfm 走 onnxruntime(%s) · providers=%s · 后端=ort-cuda（%s）"
                           % (provs[0], list(provs),
                              str(info.get("reason") or "未查询")[:140]))
        nchw = len(shape) >= 4 and shape[1] in (1, 3)
        try:
            size = int(shape[2] if nchw else shape[1])
        except Exception:                                   # noqa: BLE001
            size = 256
        cap = _sess_batch_cap(sess)
        ent = (sess, name, size, nchw, provs, cap, backend)
        _DFM_SESS[ckey] = ent
        _vram_measure_end("dfm:" + Path(key).name, _own_f0)
        _stage_log("换脸：dfm 已加载 %s · %d×%d · %s · 单次最多 %d 张"
                   % (Path(key).name, size, size, backend, cap))
        return ent


def _mask2d(t, size):
    """dfm 的遮罩输出 -> (size, size) float32 0..1（和 preview worker 里同名函数一致）。"""
    import cv2
    import numpy as np
    m = np.squeeze(t)
    if m.ndim == 3:
        m = m[:, :, 0]
    m = np.asarray(m, np.float32)
    if m.shape[:2] != (int(size), int(size)):
        m = cv2.resize(m, (int(size), int(size)), interpolation=cv2.INTER_CUBIC)
    return m


def _dfm_run_batch(ent, batch_in):
    """一次前向跑 N 张（N = batch_in.shape[0]），返回 [(face, prd_mask, dst_mask)]。

    batch_in: (N,S,S,3) float32 0..1（NHWC；NCHW 的 dfm 在调用方转置）。
    dfm 的 batch 维是动态的（['batch',416,416,3]），所以批处理不用重新导出模型；
    CUDA graph 那种「按 batch 捕获」的坑在 ORT 这边不存在（我们不开 cuda_graph），
    真要开 TRT 引擎就得按 batch 分桶 —— 见 _engines.try_engine 的 graph_max_batch。
    """
    import cv2
    import numpy as np
    sess, name, size, nchw = ent[0], ent[1], ent[2], ent[3]
    cap = int(ent[5] or 1) if len(ent) > 5 else 1
    n_in = int(batch_in.shape[0])
    if cap > 1 and n_in > cap:
        # 一次塞不下（原生 TRT 的优化档位可能只有 1）—— 按上限切开，结果顺序不变
        out = []
        for _i in range(0, n_in, cap):
            out.extend(_dfm_run_batch(ent, batch_in[_i:_i + cap]))
        return out
    x = batch_in
    if nchw:
        x = np.transpose(x, (0, 3, 1, 2))
    _INFER_CALLS["dfm"] += int(x.shape[0])
    outs = sess.run(None, {name: np.ascontiguousarray(x, np.float32)})
    n = int(x.shape[0])
    faces, prds, dsts = [], [], []
    for i in range(n):
        face = None
        for o in outs:
            a = np.asarray(o)
            if a.ndim >= 4 and a.shape[-1] == 3:
                face = a[i] if a.ndim == 4 else a[i][0]
                break
        if face is None:
            a = np.asarray(outs[0])
            face = a[i]
            if face.ndim == 3 and face.shape[-1] == 1:
                face = np.repeat(face, 3, axis=-1)
        face = np.asarray(face, np.float32)
        if face.ndim > 3:
            face = face[0]
        if nchw and face.shape[0] == 3:
            face = np.transpose(face, (1, 2, 0))
        if face.shape[:2] != (size, size):
            face = cv2.resize(face, (size, size), interpolation=cv2.INTER_LANCZOS4)
        prd = dst = None
        if len(outs) >= 3:
            prd = _mask2d(np.asarray(outs[0])[i], size)
            dst = _mask2d(np.asarray(outs[2])[i], size)
        faces.append(np.clip(face, 0, 1))
        prds.append(prd)
        dsts.append(dst)
    return list(zip(faces, prds, dsts))


def _stage_face_rows(aligned) -> list:
    """aligned/ 里的对齐脸按帧归组：[("000001", [路径…]), …]（帧序 = 文件名字典序）。"""
    out, order = {}, []
    for p in aligned:
        key = p.stem.split("_")[0]
        if key not in out:
            out[key] = []
            order.append(key)
        out[key].append(p)
    return [(k, out[k]) for k in order]


def _stage_frame_path(frames_dir, key: str):
    d = Path(frames_dir)
    for ext in _STAGE_IMG_EXTS:
        cand = d / (key + ext)
        if cand.is_file():
            return cand
    return None


def _stage_pred_assign(cfg: dict, frame, metas: list) -> list:
    """这一帧每张脸用哪个 dfm（多模型 = 人脸库 embedding 归组；没有库就全用同一个）。

    和 13-MERGE-SEMANTICS 的口径一致：
      · 没有人脸库 -> 左栏那个 dfm 换全部脸；
      · 有人脸库   -> 按 embedding 归组，**不匹配的脸不给 pred**（⑤ 保持原像素，
        绝不兜给任何模型）。
    返回 [dfm 路径 or ""]（下标 = aligned 里的脸序号）。
    """
    single = str(cfg.get("dfm") or "")
    if not cfg.get("use_library") or not cfg.get("library"):
        return [single] * len(metas)
    from . import _face_group as _fg
    from . import _faces as _store
    from ._align import align_face_68

    cache = cfg.setdefault("_pred_plan_cache", {})
    if "profiles" not in cache:
        library = _auto_library(cfg.get("plan_payload") or {})
        index = None
        profiles = []
        if library is not None:
            try:
                index = _store.load_index(library)
                if any(str(g.get("dfm") or "").strip() for g in index["groups"]):
                    profiles = _fg.group_profiles(library, index,
                                                  compute_limit=_AUTO_EMB_LIMIT)
            except Exception:                               # noqa: BLE001
                profiles = []
        cache["profiles"] = profiles or []
        if not cache["profiles"]:
            cache["off"] = True                             # 库不可用 -> 退回单模型
    if cache.get("off"):
        return [single] * len(metas)
    threshold = cfg.get("face_threshold")
    threshold = float(_fg.DEFAULT_THRESHOLD if threshold is None else threshold)
    out = []
    for m in metas:
        dfm = ""
        try:
            lm = np.asarray(m.get("lm"), np.float32)
            box = m.get("box")
            crop, _mf = align_face_68(frame, lm, box, 512)
            vec = _fg.embed_crop(crop) if crop is not None else None
            best = _fg.match_vector(vec, cache["profiles"], threshold) if vec is not None else None
            if best and best.get("matched") and best.get("dfm"):
                dfm = _dfm_by_name(best["dfm"]) or ""
        except Exception:                                   # noqa: BLE001
            dfm = ""
        out.append(dfm or "")
    return out


# ---------------- ③ 换脸（dfm 推理）：batch / 显存 / 流水线 ----------------
def _vram_free_torch():
    """torch 看到的 (free, total) MB —— 和推理同一次驱动查询，但**偏乐观**（见下）。"""
    try:
        import torch
        if torch.cuda.is_available():
            free, total = torch.cuda.mem_get_info(0)
            return int(free // (1 << 20)), int(total // (1 << 20))
    except Exception:                                       # noqa: BLE001
        pass
    return None, None


_SMI_MEM = {"at": 0.0, "free": None, "total": None, "used": None}


def _vram_free_smi(fresh: bool = False):
    """nvidia-smi 的 (free, total, used) MB。

    '--query-gpu' 一行一卡，**只取第 0 行**：torch 的 'cuda:0' 就是 nvidia-smi 的
    GPU 0（这台机器上另一块是 AMD，nvidia-smi 根本看不见它，所以不存在「把 AMD 的
    显存算进来」这回事）。一次约 80ms，2 秒小缓存 —— 预览每拖一帧都要问。
    """
    now = time.time()
    if not fresh and _SMI_MEM["at"] and (now - _SMI_MEM["at"]) < 2.0:
        return _SMI_MEM["free"], _SMI_MEM["total"], _SMI_MEM["used"]
    free = total = used = None
    try:
        out = subprocess.run(["nvidia-smi",
                              "--query-gpu=memory.free,memory.total,memory.used",
                              "--format=csv,noheader,nounits"], capture_output=True,
                             text=True, timeout=10,
                             creationflags=getattr(_app_mod, "_NO_WINDOW", 0))
        row = [x for x in (out.stdout or "").strip().splitlines() if x.strip()]
        f, t, u = [int(float(x)) for x in row[0].split(",")[:3]]
        free, total, used = f, t, u
    except Exception:                                       # noqa: BLE001
        pass
    if free is not None:
        _SMI_MEM.update({"at": now, "free": free, "total": total, "used": used})
    return free, total, used


def _vram_free_mb(fresh: bool = False):
    """(free, total) MB —— torch 与 nvidia-smi 取**更小的那个空闲**。

    **为什么两个都问**（这一条是量的，不是拍的；本机 RTX 5070 Laptop 8GB / WDDM）：
    空载时 torch.cuda.mem_get_info 说空闲 6999 MB，同一刻 nvidia-smi 说 5693 MB。
    差的 1.3GB 不是谁算错了 —— WDDM 下 mem_get_info 报的是「驱动还愿意答应多少」
    （不够就换页/回落共享内存），真分配下去立刻变成 nvidia-smi 的 used。**拿它当
    预算就是乐观到 OOM**；反过来 nvidia-smi 的 free 是「这张卡上所有进程之外还剩
    多少」，那才是敢用的数。所以两个都探、取小。

    两个都探不到的机器（没有 torch、也没有 nvidia-smi）返回 (None, None) ——
    调用方按「不猜」处理（不降档、不拒绝）。
    """
    tf, tt = _vram_free_torch()
    sf, st, _su = _vram_free_smi(fresh=fresh)
    cands = [(v, t) for v, t in ((tf, tt), (sf, st)) if v is not None]
    if not cands:
        return None, None
    free = min(v for v, _t in cands)
    total = next((t for _v, t in cands if t), None)
    return int(free), (None if total is None else int(total))


# ============================================================
# 显存预算（三条路共用）：模型体量 → (计划 batch, 帧预算, 预计峰值, 可行?)
# ============================================================
# 为什么要有这一块：以前只有 _fused_budget（只按 30/16/8 降帧）和 _pred_pick_batch
# （OOM 之后才降 batch）—— 两条都是「先跑，崩了再退」，**没有把模型体量算进总需求**，
# 也没在「连 batch=1 都装不下」时停下来。结果是 3GB 空闲的机器上点导出会一路 OOM 到
# 崩进程（fused 那条尤其难看：线程全炸、日志只剩 traceback）。
#
# 现在的口径是「按预算跑，跑不了就明说」：
#   ① 探测可用显存（torch → nvidia-smi；DFL_VRAM_FREE_MB 可伪造，给验收/排障用）；
#   ② 预算 = 空闲 × (1 - 20%) - 512MB（20% 是驱动/碎片余量，512MB 给桌面合成器）；
#   ③ 需求 = 会话（权重 + ORT arena）+ CUDA context + 检测/遮罩 + 帧缓冲 + dfm 批；
#   ④ 沿阶往下试，第一个装得下的就用（降档要记账）；连最小档都装不下：
#        · 默认 → 明确拒绝（HTTP 400，消息里给「需要 X GB / 可用 Y GB / 建议」）；
#        · DFL_ALLOW_CPU_FALLBACK=1 → 回落 CPU EP 照跑（慢 ~17 倍），日志和 status
#          都写「当前在 CPU 上跑」。
#
# 这里的常数都是**量的**，不是拍的（RTX 5070 Laptop 8GB + 416 dfm + ort-CUDA）：
#   会话建好（含这个进程的 CUDA context）          1161 MB   ← 权重 514 + context 512 + arena 135
#   batch 1 / 2 / 4 / 8 / 16 的整卡增量             1173 / 1240 / 2199 / 2256 / 6274 MB
#     → 减去会话本身：+12 / +79 / +1036 / +1091 / +5117（batch 4 起 ORT 的 arena
#       换到另一个池子，所以是台阶不是直线；16 会把 8GB 卡顶到墙 —— 与 _pred_pick_batch
#       里「16 撑爆 arena 之后整个进程变慢」的实测一致）；
#   384 模型（文件 1567MB）会话 ≈ 2.07GB（用户口径 + 本仓库实测）→ 权重 + context 就够，
#   arena 是小头，所以 _VRAM_ARENA_MB 取 140（416 实测 135，上取整）。
_VRAM_ENV_FREE = "DFL_VRAM_FREE_MB"        # 验收/排障：假装可用显存只有这么多（三条路共用）
_VRAM_ENV_CPU = "DFL_ALLOW_CPU_FALLBACK"   # 1 = 显存不够时回落 CPU EP（默认关：宁可拒绝）
# ---- 预留（2026-05 改小：老的「空闲 20% + 512MB」没有任何实测依据，在 8GB 卡上
#      等于给用户凭空造了 1.37GB 的死重 —— 见 _vram_reserve_mb 的注释）----
_VRAM_SAFETY = 0.05                        # 预留 = 空闲的 5%（随空闲缩放）
_VRAM_RESERVE_MB = 256                     # 底线：探测抖动（torch 与 smi 差 ≤256MB 的容差）
_VRAM_RESERVE_MAX_MB = 768                 # 上限：大卡上不许把预留变成 1GB+
_VRAM_CTX_MB = 512                         # 每进程一份 CUDA context
_VRAM_ARENA_MB = 140                       # ORT CUDA arena：会话建好后比权重多出来的那点
_VRAM_AUX_MB = 300                         # 检测器 + 2D106 关键点 + XSeg（用户口径 0.3GB）
_VRAM_CPU_SLOW = 17.0                       # 回落 CPU 的代价（416 实测 527ms vs 31ms/张）
_VRAM_BASE_PX = 416                         # 下面两张表都是按 416 量的，其它尺寸按 (s/416)² 缩放
# dfm 批的成本**按模型各量一张表**（用户口径：同一个 batch，大网络和小网络多占的不是
# 一个数）。点集 = ((batch, 相对同一份会话 batch=1 时多占的 MB), ...)，台阶不是直线（ORT 的
# arena 在 batch 4 换池子），所以查表 + 分段线性。
# 实测（本机 RTX 5070 Laptop 8GB · 1080x1920 真项目 120 帧 · 1 个 worker · 整卡 nvidia-smi
# 峰值，空载基线 450MB；见 %TEMP%/dfl_projplan/bench_steps.json）：
#   384（权重 1567MB · 输入 384）：b1 3184 / b2 4858（+1674）/ b4 6160（+2976）/ b8 7736（+4552）
#   416（权重  514MB · 输入 416）：b1 1990 / b2 3469（+1479）/ b4 4040（+2050）/ b8 7322（+5332）
# 注意两件事：
#   · 增量主要由**帧尺寸**（合成那一路的整帧张量）决定，不是由 dfm 大小 —— 所以两张表
#     差不多，但 384 那份在 batch 4 明显更贵（+2976 vs +2050）；
#   · batch 8 两张表都超过 +4500MB，这时候已经在换页（实测 swap p50 1906~2266ms）。
_VRAM_BATCH_PTS_416 = ((1, 0.0), (2, 1479.0), (4, 2050.0), (8, 5332.0))
_VRAM_BATCH_PTS_384 = ((1, 0.0), (2, 1674.0), (4, 2976.0), (8, 4552.0))
# (权重上限 MB, 量的那个输入边长, 点集)：权重 ≤900MB 用 416 那张，更大的用 384 那张
_VRAM_BATCH_TABLES = ((900.0, 416, _VRAM_BATCH_PTS_416),
                      (1e9, 384, _VRAM_BATCH_PTS_384))
# fused（帧预算那条路）的 dfm 批台阶是**另一个量**：那条路一次只让 dfm 前向 k 张，
# 不背「整帧合成 + 落盘」的 k 份缓冲，所以它比项目导出便宜得多。老表就是在那条路上量的
# （416 实测：+80/+160/+1100/+1150/+5200），保留给 frames / none 两条路；项目导出用上面
# 那两张按模型各量的表。混用会把 fused 的降档判据整体抬高一倍（实测：30 帧/批 1 从
# 1644MB 变成 3232MB，fused_export_accept 的 C 段当场从「降到 16 帧」变成「照跑 30 帧」）。
_VRAM_BATCH_PTS = ((1, 80.0), (2, 160.0), (4, 1100.0), (8, 1150.0), (16, 5200.0))
# 项目导出 worker 的**运行时工作集**：会话建好、再真跑起来之后 ORT/CuDNN 还会多占一块，
# 而这一块**不在**「权重 + context」里。实测（本机 8GB，1080x1920 真项目 120 帧，
# 1 个 worker、整卡 nvidia-smi 峰值、空载基线 450MB，见 %TEMP%/dfl_projplan/bench_steps.json）：
#   · 384（权重 1567MB · 输入 384）batch=1 整卡 3184MB → 一份会话
#       = 3184 − 450(桌面) − 300(检测/遮罩) − 0(batch 1) = **2434MB**
#       = 1567 权重 + 512 context + **355 运行时**；
#   · 416（权重  514MB · 输入 416）batch=1 整卡 1990MB → 一份会话 = **1240MB**
#       = 514 + 512 + **214 运行时**。
# 取 320MB（两次实测的中间偏上）。**batch 的成本不在这里**：它按模型各有一张表
# （_VRAM_BATCH_TABLES），因为 384 那份 batch 4 要多 2976MB、416 只要 2050MB。
# 最早那版把这个常数写成 1300，是把 dfm 批那一项重复算进了「会话」里，已修正。
_VRAM_SESS_RUNTIME_MB = 320
_VRAM_PROBE_TTL = 4.0                       # 预览每拖一次都要问：nvidia-smi 80ms 不能每次都跑
_VRAM_PROBE = {"at": 0.0, "free": None, "total": None, "src": ""}
_VRAM_MODEL_CACHE = {}
_VRAM_SIZE_CACHE = {}
_VRAM_LAST = {}                             # label -> 最近一次的计划（status 端点要能回显）
_VRAM_LOGGED = {}                           # label -> 上一次打过的签名（同样的决定只打一次）
_VRAM_LOCK = threading.RLock()
_CPU_EP = {"on": False, "why": ""}          # 全局「强制 CPU EP」：_dfm_session / _stage_xseg 读它
_CPU_EP_LOCK = threading.Lock()


# ============================================================
# 显存归属：这块卡现在被谁占着（本程序 / 其它程序 / 空闲）
# ------------------------------------------------------------
# 用户那一版截图最糟的地方不是被拒，而是**被拒了不知道是谁占的**：界面写「空闲
# 1387MB」，而他自己什么都没跑 —— 因为占着的就是我们（预览 worker 里常驻的 dfm
# 会话、本进程里加载的 dfm / 检测器，以及 ORT 那个会长大的 CUDA arena）。
#
# 这里只记**我们自己**的账，而且尽量是**实测**：
#   · 本进程：每个 session 加载前后各探一次空闲，差值就是它真的吃掉的那一块
#     （含 CUDA context / ORT arena —— 估不出来，只能量）；
#   · 预览 worker：每次 call 前后探一次，差值记在 worker 上（它可能是 1 个也可能是
#     3 个模型，还带着自己的 arena）；
#   · 剩下的 = 总显存 - 空闲 - 本程序 → 归「其它程序」（浏览器/游戏/另一个导出）。
# ============================================================
_VRAM_OWN = {}          # tag -> MB（本进程自己加载的，实测增量）
_VRAM_OWN_LOCK = threading.RLock()
_VRAM_OWN_PENDING = {}  # tag -> 起始空闲（量到一半的，防并发重复计）


def _vram_own_note(tag: str, mb: float):
    """记一笔「本进程里这个东西吃掉了多少显存」（MB，实测增量）。"""
    tag = str(tag or "")
    mb = max(0.0, float(mb or 0.0))
    if not tag or mb <= 0:
        return
    with _VRAM_OWN_LOCK:
        _VRAM_OWN[tag] = round(max(mb, float(_VRAM_OWN.get(tag) or 0.0)), 1)


def _vram_own_drop(tag: str):
    """这个东西被放掉了（session 关了/切了 CPU）—— 从本进程的账上划掉。"""
    with _VRAM_OWN_LOCK:
        _VRAM_OWN.pop(str(tag or ""), None)


def _vram_own_detail() -> dict:
    with _VRAM_OWN_LOCK:
        return {str(k): round(float(v), 1) for k, v in sorted(_VRAM_OWN.items())}


def _vram_own_inproc_mb() -> float:
    """本进程自己占着多少（MB）。"""
    with _VRAM_OWN_LOCK:
        return round(float(sum(_VRAM_OWN.values())), 1)


def _vram_measure_begin():
    """开始量：返回起始空闲（MB，探不到就是 None）。"""
    f0, _t = _vram_free_mb(fresh=True)
    return f0


def _vram_measure_end(tag: str, f0):
    """量完：把「空闲掉了多少」记到 tag 上（探不到就不记，绝不用 0 冒充）。"""
    if f0 is None:
        return None
    f1, _t = _vram_free_mb(fresh=True)
    if f1 is None:
        return None
    used = max(0.0, float(f0) - float(f1))
    _vram_own_note(tag, used)
    return round(used, 1)


def _vram_preview_info() -> dict:
    """常驻预览 worker 现在占着多少、都常驻了什么、能不能回收。

    优先用**实测**（worker 每次调用前后探到的空闲差，见 _PreviewWorker.call）；
    没量过才退回会话口径估（模型体量 + 一份 CUDA context + ORT arena）。实测优先
    是因为 ORT 的 CUDA arena 会长：384 模型实测「建会话 +2.3GB」，之后每一批 8 张
    的前向还会再涨 ~1GB（本机 3 次之后稳定在 6.4GB/8151MB）—— 估出来的数会偏小。
    """
    try:
        w = _WORKER
        alive = bool(w.alive())
        models = list((w._models or {}).keys()) if alive else []
        measured = float(getattr(w, "_used_mb", 0.0) or 0.0)
    except Exception:                                       # noqa: BLE001
        alive, models, measured = False, [], 0.0
    est = 0.0
    if alive and models:
        est = (float(_vram_models_mb(models)) + float(_VRAM_CTX_MB)
               + float(_VRAM_ARENA_MB) * max(1, len(models)))
    # **没有常驻模型 = 那块显存已经还回去了**（unload / LRU 淘汰都走 worker 里的
    # _vram_flush：掐引用 → gc → empty_cache）。这时 `_used_mb` 是上一次加载的旧账，
    # 不能再算进"本程序占用"—— 否则账上永远挂着 1.6GB，谁会去回收一个不存在的东西。
    mb = max(measured, est) if models else 0.0
    return {"alive": bool(alive), "models": [os.path.basename(str(p)) for p in models],
            "paths": [str(p) for p in models], "measured_mb": round(measured, 1),
            "estimated_mb": round(est, 1), "mb": round(mb, 1),
            "measured": bool(measured > 0)}


def _vram_owners(free_mb=None, total_mb=None, source: str = "") -> dict:
    """谁占着显存：本程序（分项 + 可回收多少）/ 其它程序 / 空闲。

    拒绝消息、status、日志都读这一份，所以「空闲 1387MB」不再是一个孤零零的数字 ——
    用户能看见「其中本程序 2.0GB 是可回收的预览模型，其它程序 3.4GB」。
    """
    if free_mb is None or total_mb is None:
        pfree, ptotal, _p = _vram_probe()
        free_mb = pfree if free_mb is None else free_mb
        total_mb = ptotal if total_mb is None else total_mb
    pv = _vram_preview_info()
    inproc = _vram_own_inproc_mb()
    own = float(inproc) + float(pv.get("mb") or 0.0)
    other = None
    src = str(source or "")
    real = not (src.startswith("env:") or src.startswith("file:") or src == "override")
    if free_mb is not None and total_mb is not None and real:
        # 只有真探到的 free 才能反推「其它程序」；env/file/override 是伪造值，
        # 拿它减出来的"其它程序"是假账（宁可显示 ?，也不给用户一个错数）。
        other = round(max(0.0, float(total_mb) - float(free_mb) - own), 1)
    torch_res = None
    try:
        import torch
        if torch.cuda.is_available():
            torch_res = round(torch.cuda.memory_reserved() / 1048576.0, 1)
    except Exception:                                       # noqa: BLE001
        pass
    return {"free_mb": (None if free_mb is None else int(free_mb)),
            "total_mb": (None if total_mb is None else int(total_mb)),
            "own_mb": round(own, 1), "own_inproc_mb": round(float(inproc), 1),
            "own_preview_mb": round(float(pv.get("mb") or 0.0), 1),
            "own_parts": _vram_own_detail(),
            "preview_models": list(pv.get("models") or []),
            "preview_measured": bool(pv.get("measured")),
            "reclaimable_mb": round(float(pv.get("mb") or 0.0), 1),
            "other_mb": other, "torch_reserved_mb": torch_res,
            "free_torch_mb": _vram_free_torch()[0],
            "free_smi_mb": _vram_free_smi()[0]}


def _vram_reclaim(why: str = "导出") -> dict:
    """把**我们自己**占的显存收回来：请常驻预览 worker 卸掉它的 DFM 会话。

    为什么导出前必须做一遍：用户先预览过一帧（worker 里 1.6GB 的 dfm + CUDA context
    常驻），再点导出 —— 计划看到的是「空闲 1.4GB」，于是把一单本来完全跑得动的活拒
    了，而占着的正是我们自己的子进程。这里先让它放掉、等驱动真回收（轮询空闲涨上来，
    最多 4s），再拿新的空闲去算预算。

    导出结束后**不重载**：下次预览再懒加载（实测一次重载 ~4.1s，比来回抖一晚上强）。
    """
    t0 = time.time()
    try:
        w = _WORKER
        alive = bool(w.alive())
        models = list((w._models or {}).keys()) if alive else []
        used = float(getattr(w, "_used_mb", 0.0) or 0.0)
    except Exception:                                       # noqa: BLE001
        alive, models, used = False, [], 0.0
    if not alive or not models:
        return {"ok": True, "did": False, "freed_mb": 0.0, "ms": 0.0,
                "models": [], "before_mb": None, "after_mb": None,
                "why": "预览 worker 里没有常驻模型，不用回收"}
    before, _t = _vram_free_mb(fresh=True)
    want = max(float(used),
               float(_vram_models_mb(models)) + float(_VRAM_CTX_MB)
               + float(_VRAM_ARENA_MB) * max(1, len(models)))
    names = [os.path.basename(str(p)) for p in models]
    reply = {}
    try:
        reply = dict(w.unload("") or {})
    except Exception as e:                                  # noqa: BLE001
        print("[vram] 回收预览模型失败（%s）：unload 抛异常：%s"
              % (why, str(e)[:200]), flush=True)
        return {"ok": False, "did": False, "freed_mb": 0.0,
                "ms": round((time.time() - t0) * 1000.0, 1), "models": names,
                "before_mb": before, "after_mb": before,
                "why": "回收失败（unload 抛异常）：%s" % str(e)[:160]}
    # ★ 「回应说没放掉」同样算失败。worker 的 unload 是在**子进程**里执行的，它自己
    #   失败时（模型被那一帧占着 / _vram_flush 抛错 / 不认这个命令）不会抛异常，只回
    #   ok=false 或**原样带着 models 清单**回来。以前这里只 catch 异常 —— 于是失败被
    #   记成「成功、腾出 0MB」，日志上只剩一个 0，原因永远看不见（用户口径：失败要写
    #   清原因，不能只打 0）。
    try:
        left = [str(p) for p in (w._models or {}).keys()]
    except Exception:                                       # noqa: BLE001
        left = []
    if left or reply.get("ok") is False:
        _r_why = str(reply.get("error") or reply.get("why") or reply.get("msg")
                     or reply.get("stage") or "").strip()
        if not _r_why:
            _r_why = ("worker 回了 ok=false" if not left else
                      "worker 说放掉了，但 %d 份模型还在：%s"
                      % (len(left), "/".join(os.path.basename(p) for p in left[:3])))
        print("[vram] 回收预览模型失败（%s）：%s（worker 里仍常驻 %d 份：%s）"
              % (why, _r_why[:200], len(left),
                 "/".join(os.path.basename(p) for p in left[:3]) or "-"), flush=True)
        return {"ok": False, "did": False, "freed_mb": 0.0,
                "ms": round((time.time() - t0) * 1000.0, 1), "models": names,
                "left_models": [os.path.basename(p) for p in left],
                "before_mb": before, "after_mb": before,
                "why": "回收失败：%s" % _r_why[:160]}
    after = before
    deadline = time.time() + 4.0
    while time.time() < deadline:
        after, _t = _vram_free_mb(fresh=True)
        if before is None or after is None:
            break
        if (float(after) - float(before)) >= 0.85 * max(1.0, want):
            break
        time.sleep(0.15)
    freed = (0.0 if (before is None or after is None)
             else max(0.0, float(after) - float(before)))
    # ---- 卸了模型、显存却没回来？把 worker **进程**整个回收掉 ----
    # WDDM 上释放不等于还给驱动：本机实测预览 worker 卸掉一份 384 模型之后，进程仍握着
    # 2347MB（Windows「GPU Process Memory · Dedicated」实测），而 nvidia-smi 的 free 只
    # 涨了 135MB。这块残留就是用户那句「我什么都没跑，却被自己的后端堵死」的另一半 ——
    # 只要进程不退，那 2.3GB 就一直在账上，谁也算不出「其实是自己」。
    # 只在「卸完几乎没腾出东西」时动手（正常情况这里是 2GB+，一行都不用走），
    # 代价是下次预览要重开进程（实测 boot 1.6s + 冷加载 3.5~4.2s），比被拒强得多。
    _rc_switch = str(os.environ.get("DFL_VRAM_RECYCLE_WORKER") or "").strip().lower()
    recycled = False
    if _rc_switch not in ("0", "false", "no", "off") and freed < 0.5 * max(1.0, want):
        print("[vram] 卸掉预览模型只腾出 %.0f MB（应约 %.0f MB）—— 进程仍握着这些显存，"
              "直接回收 worker（下次预览懒加载）" % (freed, want), flush=True)
        try:
            recycled = bool(w.close("回收显存：卸模型后只腾出 %.0f MB" % freed))
        except Exception as e:                              # noqa: BLE001
            print("[vram] 回收预览 worker 失败：%s" % str(e)[:160], flush=True)
        if recycled:
            deadline = time.time() + 8.0
            while time.time() < deadline:
                a2, _t = _vram_free_mb(fresh=True)
                if a2 is not None:
                    after = a2
                if (before is None or after is None
                        or (float(after) - float(before)) >= 0.85 * max(1.0, want)):
                    break
                time.sleep(0.2)
            freed = (0.0 if (before is None or after is None)
                     else max(0.0, float(after) - float(before)))
    try:
        w._used_mb = 0.0            # 账已经清了：下一次 call 重新量
    except Exception:                                       # noqa: BLE001
        pass
    ms = (time.time() - t0) * 1000.0
    # 卸掉了模型、空闲却没涨够（WDDM 上「释放」不等于还给驱动）：这同样是「回收没成」，
    # 必须有个原因挂在 why 上 —— 否则预算里那一栏是 0，日志上也只有一个 0。
    # 口径（按「腾出来多少」分两档，别把 84% 的火气也写成失败）：
    #   < 50%  → 真失败（ok=False，why 进日志/summary/拒绝消息）
    #   50~85% → 只是比「模型体量 + context」估的少（note，不判失败）
    _short = ""
    _note = ""
    if before is None or after is None:
        _short = "显存探不到（torch / nvidia-smi 都没答），只能按模型体量估 %.0f MB" % want
    elif freed < 0.5 * max(1.0, want):
        _short = ("卸了模型但空闲只涨了 %.0f MB（应约 %.0f MB）%s"
                  % (freed, want,
                     "；worker 进程已回收" if recycled else
                     "；进程没回收（DFL_VRAM_RECYCLE_WORKER=0 或回收失败）"))
    elif freed < 0.85 * max(1.0, want):
        _note = ("比按模型体量估的 %.0f MB 少（进程%s）"
                 % (want, "已回收" if recycled else "仍常驻，WDDM 上释放不全正常"))
    print("[vram] 回收预览模型 %.0f MB（%s · %s · %.0fms%s）：空闲 %s → %s MB%s%s"
          % (freed, why, "/".join(names[:3]) or "-", ms,
             " · 进程已回收" if recycled else "",
             "?" if before is None else int(before),
             "?" if after is None else int(after),
             (" · " + _short) if _short else "",
             (" · " + _note) if _note else ""), flush=True)
    return {"ok": not bool(_short), "did": True, "freed_mb": round(freed, 1),
            "ms": round(ms, 1),
            "models": names, "before_mb": before, "after_mb": after,
            "want_mb": round(want, 1), "recycled": bool(recycled),
            "why": _short, "note": _note}


def _vram_src_real(src) -> bool:
    """这个 free 是不是**真探到的**。env / file / override 都是伪造或单测注入的值，
    拿它们再加「可回收」就是假账（一个文本文件就能把预算抬高 2.7GB）。"""
    s = str(src or "")
    return not (s.startswith("env:") or s.startswith("file:") or s in ("override", ""))


_VRAM_RECLAIM_MARK = {"at": 0.0, "rec": {}}


def _vram_reclaim_once(why: str = "导出", gap: float = 6.0) -> dict:
    """回收预览 worker 的模型，但**同一单里只真收一次**。

    为什么要节流：导出起手有两拍预算 —— 路由闸门（HTTP 400 那一拍）和 worker 线程
    里 _fused_run 的第一拍。两拍都开 reclaim_first，就会连着卸两次；第二拍其实什么
    都没得卸（worker 里已经空了），却会再跑 2~3 次 nvidia-smi（WDDM 上每次 ~80ms）。
    gap 内的重复调用直接返回上一次的结果（带 throttled=True 留痕）。
    """
    # 回滚 / A-B 对照开关：1 = 完全不卸预览模型（日志里会写明是它，不装成「没得卸」）
    if str(os.environ.get("DFL_VRAM_NO_RECLAIM") or "").strip().lower() in ("1", "true", "yes", "on"):
        return {"ok": False, "did": False, "freed_mb": 0.0, "ms": 0.0, "models": [],
                "before_mb": None, "after_mb": None,
                "why": "DFL_VRAM_NO_RECLAIM=1（回滚/对照用：这一轮不卸预览模型）"}
    now = time.time()
    with _VRAM_LOCK:
        at = float(_VRAM_RECLAIM_MARK.get("at") or 0.0)
        last = dict(_VRAM_RECLAIM_MARK.get("rec") or {})
    if at and (now - at) < float(gap):
        out = dict(last or {})
        out.update({"throttled": True, "age_ms": round((now - at) * 1000.0, 1)})
        return out
    try:
        rec = _vram_reclaim(why)
    except Exception as e:                                  # noqa: BLE001
        rec = {"ok": False, "did": False, "freed_mb": 0.0, "ms": 0.0, "models": [],
               "before_mb": None, "after_mb": None,
               "why": "回收抛异常：%s: %s" % (e.__class__.__name__, str(e)[:140])}
        print("[vram] 回收预览模型失败（%s）：%s" % (why, str(rec["why"])[:200]), flush=True)
    with _VRAM_LOCK:
        _VRAM_RECLAIM_MARK.update({"at": time.time(), "rec": dict(rec or {})})
    return dict(rec or {})


# ---------------- 进程内存（工作集 / 提交）：导出前后差，用户看的就是这两个数 ----
def _fmt_mb(v):
    return "?" if v is None else "%.1f" % float(v)


def _proc_mem_mb() -> dict:
    """本进程内存（MB）：rss = 工作集（任务管理器「内存」那一列）·
    private = 提交（「已提交」那一列）· peak_* = 峰值。

    为什么不用 psutil：后端不是到处都装了它，而 psapi 就在 kernel32 旁边 ——
    和任务管理器同源，打进日志的数字能和用户截图对上（不然「日志 800MB、任务管理器
    1.3GB」又是一轮谁也说不清的对话）。

    两个数说的不是一回事：**提交**包含压到页面文件里的、以及 WDDM 上由显存兜着的
    那部分；**工作集**才是当下真占着物理内存的量。用户看到的「内存 13/15.2」是
    工作集口径，「已提交 37/129」是提交口径 —— 拿提交去喊内存泄漏是错的。
    """
    try:
        import ctypes
        if sys.platform.startswith("win"):
            class _PMC(ctypes.Structure):
                _fields_ = [("cb", ctypes.c_ulong),
                            ("PageFaultCount", ctypes.c_ulong),
                            ("PeakWorkingSetSize", ctypes.c_size_t),
                            ("WorkingSetSize", ctypes.c_size_t),
                            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                            ("QuotaPagedPoolUsage", ctypes.c_size_t),
                            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                            ("PagefileUsage", ctypes.c_size_t),
                            ("PeakPagefileUsage", ctypes.c_size_t)]
            pmc = _PMC()
            pmc.cb = ctypes.sizeof(pmc)
            k32 = ctypes.WinDLL("kernel32.dll")
            k32.GetCurrentProcess.restype = ctypes.c_void_p
            psapi = ctypes.WinDLL("psapi.dll")
            psapi.GetProcessMemoryInfo.argtypes = [ctypes.c_void_p, ctypes.c_void_p,
                                                   ctypes.c_ulong]
            if psapi.GetProcessMemoryInfo(ctypes.c_void_p(k32.GetCurrentProcess()),
                                          ctypes.byref(pmc), ctypes.sizeof(pmc)):
                return {"rss": round(pmc.WorkingSetSize / 1048576.0, 1),
                        "private": round(pmc.PagefileUsage / 1048576.0, 1),
                        "peak_rss": round(pmc.PeakWorkingSetSize / 1048576.0, 1),
                        "peak_private": round(pmc.PeakPagefileUsage / 1048576.0, 1)}
    except Exception:                                       # noqa: BLE001
        pass
    try:                                          # 非 Windows：只有 rss 这一列
        with open("/proc/self/statm", "r", encoding="ascii") as f:
            pages = int((f.read() or "0 0").split()[1])
        return {"rss": round(pages * 4096 / 1048576.0, 1), "private": None,
                "peak_rss": None, "peak_private": None}
    except Exception:                                       # noqa: BLE001
        return {}


def _mem_delta(m0: dict, m1: dict) -> str:
    """「导出前后」那一行：RSS 差 / 提交差 + 峰值 RSS。取不到的项写 ?，不抛。"""
    def _n(d, k):
        v = (d or {}).get(k)
        return None if v is None else float(v)

    a, b = _n(m0, "rss"), _n(m1, "rss")
    c, d = _n(m0, "private"), _n(m1, "private")
    out = "RSS %s → %s MB" % (_fmt_mb(a), _fmt_mb(b))
    if a is not None and b is not None:
        out += "（差 %+.1f）" % (b - a)
    out += " · 提交 %s → %s MB" % (_fmt_mb(c), _fmt_mb(d))
    if c is not None and d is not None:
        out += "（差 %+.1f）" % (d - c)
    pk = _n(m1, "peak_rss") or _n(m0, "peak_rss")
    if pk is not None:
        out += " · 峰值 RSS %.1f MB" % pk
    return out


def _trim_ws(why: str = "") -> dict:
    """把本进程的**工作集**还给系统（EmptyWorkingSet），返回前后读数。

    解决的是「谁都没在跑，任务管理器还显示 1.x GB」：Python/NT 堆把刚释放的页留在
    进程里（已提交 ≠ 驻留），系统压力不大就不主动回收。**它只降工作集，不降提交**
    —— 提交要等对象真的被释放才降，所以日志里两个数都打，不许拿工作集下降冒充
    「泄漏修好了」。

    代价：下次访问这些页要重新缺页（预览第一帧慢几十毫秒量级）。默认开，
    DFL_FUSED_TRIM_WS=0 可关。
    """
    m0 = _proc_mem_mb()
    if str(os.environ.get("DFL_FUSED_TRIM_WS") or "1").strip().lower() in (
            "0", "false", "no", "off"):
        return {"did": False, "before": m0, "after": m0, "why": "DFL_FUSED_TRIM_WS=0"}
    try:
        import ctypes
        if not sys.platform.startswith("win"):
            return {"did": False, "before": m0, "after": m0, "why": "非 Windows"}
        k32 = ctypes.WinDLL("kernel32.dll")
        k32.GetCurrentProcess.restype = ctypes.c_void_p
        k32.SetProcessWorkingSetSize.argtypes = [ctypes.c_void_p, ctypes.c_size_t,
                                                 ctypes.c_size_t]
        ok = k32.SetProcessWorkingSetSize(ctypes.c_void_p(k32.GetCurrentProcess()),
                                          ctypes.c_size_t(-1), ctypes.c_size_t(-1))
        return {"did": bool(ok), "before": m0, "after": _proc_mem_mb(), "why": why}
    except Exception as e:                                  # noqa: BLE001
        return {"did": False, "before": m0, "after": m0, "why": "%r" % (e,)}


def _free_preview_sessions(why: str = "") -> dict:
    """让**预览 worker** 把它常驻的 dfm / XSeg 卸掉（进程留着，下次预览懒加载）。

    为什么导出收尾必须做这一下：导出用的是**后端进程自己**的会话，预览 worker
    那份是另开的一份 —— 用户实测「导出完了还占着」，占的就是它（一张 384 dfm：
    worker 侧实测 2.3 GB 显存 / 1.3 GB 工作集 / 5.7 GB 提交）。
    _vram_release_inproc 只放本进程，管不到子进程；这里补上。

    worker 没起过（alive()=False）就什么都不做 —— **绝不能**为了「卸一下」把一个
    已经退出的 worker 叫醒（call() 会 _spawn）。
    """
    try:
        if not _WORKER.alive():
            return {"ok": True, "did": False, "why": "worker 没起（不需要卸）"}
    except Exception:                                       # noqa: BLE001
        return {"ok": True, "did": False, "why": "worker 状态读不到"}
    t0 = time.time()
    try:
        r = _WORKER.unload("")
    except Exception as e:                                  # noqa: BLE001
        return {"ok": False, "did": False,
                "why": "%s: %s" % (e.__class__.__name__, str(e)[:160])}
    return {"ok": bool((r or {}).get("ok", True)), "did": True,
            "unloaded": int((r or {}).get("unloaded") or 0),
            "ms": round((time.time() - t0) * 1000.0, 1), "why": why}


def _vram_release_inproc(reason: str = "任务收尾") -> dict:
    """把**本进程**常驻的 dfm / XSeg 会话放掉（导出收尾时用，成功/失败/取消都放）。

    为什么不能在导出结束时留着：一份 384 会话建好 2.3GB，跑完几十帧之后 ORT 的
    CUDA arena 还会继续涨（本机实测：12 帧之后整卡 used 7049/8151，空闲只剩 843MB）。
    留着它，用户下一次预览（要新建一份 2.4GB 会话）就会被**自己的后端**堵死 ——
    那正是这一轮用户报的「我什么都没跑，却说我显存不够」。放掉之后：下一次预览/导出
    各自懒加载（实测冷加载 3.5~4.2s），中间这段时间卡是干净的。

    只动本进程的缓存，**不动预览 worker**（那一路有自己的懒加载节奏）。
    """
    freed = []
    try:
        with _PRED_LOCK:
            for k in list(_DFM_SESS):
                _DFM_SESS.pop(k, None)
                freed.append("dfm:" + Path(str(k).split("|")[0]).name)
    except Exception:                                       # noqa: BLE001
        pass
    try:
        with _XSEG_SESS_LOCK:
            for _k in list(_XSEG_SESS):
                _XSEG_SESS.pop(_k, None)
                freed.append("xseg")
    except Exception:                                       # noqa: BLE001
        pass
    for tag in freed:
        _vram_own_drop(tag)
    if not freed:
        return {"ok": True, "did": False, "released": [], "reason": reason}
    # 引用断了之后真的把缓存还给驱动（ORT 每个 session 的 arena 跟着 session 一起没）
    try:
        import gc
        gc.collect()
        gc.collect()
        torch = sys.modules.get("torch")
        if torch is not None and torch.cuda.is_initialized():
            torch.cuda.empty_cache()
    except Exception:                                       # noqa: BLE001
        pass
    f1, _t = _vram_free_mb(fresh=True)
    print("[vram] 收尾放掉本进程的会话（%s）：%s · 空闲现在 %s MB"
          % (reason, "、".join(freed[:4]), "?" if f1 is None else int(f1)), flush=True)
    return {"ok": True, "did": True, "released": list(freed), "free_mb": f1,
            "reason": reason}


def _vram_probe(ttl: float = None) -> tuple:
    """(free_mb, total_mb, source)。

    source 用来在日志里说清「这个数是哪来的」：env / file / torch / nvidia-smi / none。
    env 伪造的值不走缓存（验收脚本会在同一个进程里换档，缓存会骗人）。

    还有一个**只在验收里用**的伪造口：<WORKSPACE>/_vram_free_mb.txt（一行整数，MB）。
    比 env 强的地方是它能在**同一个后端进程**里改 —— 用户那条「关掉占显存的程序后
    再点一次导出，不重启后端也能出片」就是靠它复现的（重启后端来换 env 等于把
    "缓存有没有坏"这件事测没了）。文件不存在就当没有，生产路径上零成本。
    """
    fake = str(os.environ.get(_VRAM_ENV_FREE) or "").strip()
    if fake:
        try:
            return int(float(fake)), None, "env:" + _VRAM_ENV_FREE
        except (TypeError, ValueError):
            pass
    _hook_on = str(os.environ.get("DFL_ALLOW_TEST_HOOK") or "").strip() in ("1", "true", "yes", "on")
    ff = Path(WORKSPACE) / "_vram_free_mb.txt"
    if not _hook_on:                                     # 生产环境不读这个文件（否则工作区里一个文本就能改显存判断）
        ff = None
    try:
        if ff is not None and ff.is_file():
            txt = ff.read_text("utf-8", "ignore").strip().splitlines()[0].strip()
            return int(float(txt)), None, "file:" + ff.name
    except Exception:                                       # noqa: BLE001
        # 文件在、但这一拍读不出来（正好被改写/内容不是数字）：**不许退回旧缓存**
        # —— 那会让"刚刚改成 1400"读成上一次的 8000。这里改成强制重新探测一次。
        if ff is not None and ff.is_file():
            ttl = 0.0
    old = str(os.environ.get(_FUSED_ENV_VRAM) or "").strip()   # 老开关继续认（既有验收脚本在用）
    if old:
        try:
            return int(float(old)), None, "env:" + _FUSED_ENV_VRAM
        except (TypeError, ValueError):
            pass
    ttl = _VRAM_PROBE_TTL if ttl is None else float(ttl)
    now = time.time()
    with _VRAM_LOCK:
        if _VRAM_PROBE["at"] and (now - _VRAM_PROBE["at"]) < ttl:
            return _VRAM_PROBE["free"], _VRAM_PROBE["total"], _VRAM_PROBE["src"]
    free, total = _vram_free_mb(fresh=(ttl <= 0.0))
    # 来源写清楚：torch 与 nvidia-smi 取小（WDDM 上 torch 偏乐观 1.3GB，见 _vram_free_mb）
    tf, sf = _vram_free_torch()[0], _vram_free_smi()[0]
    src = ("none" if free is None else
           "min(torch %s / smi %s)" % ("?" if tf is None else int(tf),
                                       "?" if sf is None else int(sf)))
    with _VRAM_LOCK:
        _VRAM_PROBE.update({"at": now, "free": free, "total": total, "src": src,
                            "free_torch": tf, "free_smi": sf})
    return free, total, src


# ------------------------------------------------------------
# .dfm 到底是什么：它就是一份 **ONNX ModelProto**（_dfm_session 直接
# ort.InferenceSession(dfm)）。所以「模型体量」要问图里要，**别看文件名、也别看分辨率**：
# 用户那一单里「416」那份权重只有 0.5GB、「384」那份 1.5GB —— 名字里的数字只是输入边长，
# 参数量由网络宽度决定，两者没有可比性。
#
# 这里按 protobuf 线格式扫一遍（mmap + 只读头，raw_data 只跳不读）：
#   ModelProto.graph = 7 · GraphProto.initializer = 5 (TensorProto)
#   TensorProto.dims = 1 / data_type = 2 / raw_data = 9
#   GraphProto.input = 11 → ValueInfoProto.type(2).tensor_type(1).shape(2).dim(1).dim_value(1)
# 不依赖 onnx 包（这台机器没装），也不建 session、不加载权重 —— 实测两份模型合计 < 0.2s。
# ------------------------------------------------------------
_DFM_INFO = {}
_DFM_INFO_LOCK = threading.RLock()
_PB_DT_SIZE = {1: 4, 2: 1, 3: 1, 4: 2, 5: 2, 6: 4, 7: 8, 8: 8, 9: 1, 10: 2, 11: 8, 12: 4,
               13: 8, 14: 4, 15: 8, 16: 2}


def _pb_varint(buf, i, end):
    shift = val = 0
    while True:
        b = buf[i]
        i += 1
        val |= (b & 0x7F) << shift
        if not (b & 0x80):
            return val, i
        shift += 7


def _pb_fields(buf, i, end):
    """遍历 protobuf 字段：(fno, wire, value)；wire=0 → int、wire=2 → (start, n)。"""
    while i < end:
        key, i = _pb_varint(buf, i, end)
        fno, wt = key >> 3, key & 7
        if wt == 0:
            v, i = _pb_varint(buf, i, end)
            yield fno, wt, v
        elif wt == 2:
            n, i = _pb_varint(buf, i, end)
            yield fno, wt, (i, n)
            i += n
        elif wt == 5:
            yield fno, wt, int.from_bytes(buf[i:i + 4], "little")
            i += 4
        elif wt == 1:
            yield fno, wt, int.from_bytes(buf[i:i + 8], "little")
            i += 8
        else:
            raise ValueError("wire type %d" % wt)


def _pb_tensor(buf, s, e):
    """一个 TensorProto → (字节数, 元素数, data_type)。"""
    dims, dt, raw = [], None, None
    for fno, wt, v in _pb_fields(buf, s, e):
        if fno == 1:
            if wt == 0:
                dims.append(int(v))
            elif wt == 2:
                i, end = v[0], v[0] + v[1]
                while i < end:
                    d, i = _pb_varint(buf, i, end)
                    dims.append(int(d))
        elif fno == 2 and wt == 0:
            dt = int(v)
        elif fno == 9 and wt == 2:
            raw = int(v[1])                    # 只记长度，不读内容（1.5GB 权重在这儿）
    n = 1
    for d in dims:
        n *= int(d)
    nbytes = raw if raw is not None else n * int(_PB_DT_SIZE.get(dt or 0, 0))
    return int(nbytes), int(n), dt


def _pb_input_px(buf, s, e):
    """ValueInfoProto → 输入边长（取 shape 里最大的那个空间维，动态维跳过）。"""
    for fno, wt, v in _pb_fields(buf, s, e):
        if fno != 2 or wt != 2:                # type
            continue
        for f2, w2, v2 in _pb_fields(buf, v[0], v[0] + v[1]):
            if f2 != 1 or w2 != 2:             # tensor_type
                continue
            for f3, w3, v3 in _pb_fields(buf, v2[0], v2[0] + v2[1]):
                if f3 != 2 or w3 != 2:         # shape
                    continue
                dims = []
                for f4, w4, v4 in _pb_fields(buf, v3[0], v3[0] + v3[1]):
                    if f4 != 1 or w4 != 2:     # dim
                        continue
                    for f5, w5, v5 in _pb_fields(buf, v4[0], v4[0] + v4[1]):
                        if f5 == 1 and w5 == 0:
                            dims.append(int(v5))
                if dims:
                    return int(max(dims[1:])) if len(dims) > 1 else int(max(dims))
    return 0


def _dfm_info(path) -> dict:
    """一份 .dfm 的**真实**体量：权重字节（initializer 之和）/ 参数量 / 输入边长 / 非权重字节。

    返回 {weights_mb, params, params_m, size_px, file_mb, non_weight_mb, init, src}；
    读不出来（不是 ONNX / 用了 external data）→ weights_mb=0 + src 写明原因，调用方回退文件体积。
    """
    key = str(path or "")
    if not key:
        return {"weights_mb": 0.0, "params": 0, "params_m": 0.0, "size_px": 0,
                "file_mb": 0.0, "non_weight_mb": 0.0, "init": 0, "src": "空路径"}
    try:
        st = os.stat(key)
        sig = (int(st.st_size), int(st.st_mtime))
    except OSError:
        return {"weights_mb": 0.0, "params": 0, "params_m": 0.0, "size_px": 0,
                "file_mb": 0.0, "non_weight_mb": 0.0, "init": 0, "src": "文件读不到"}
    with _DFM_INFO_LOCK:
        hit = _DFM_INFO.get(key)
    if hit and hit.get("sig") == sig:
        return dict(hit)
    info = {"sig": sig, "weights_mb": 0.0, "params": 0, "params_m": 0.0, "size_px": 0,
            "file_mb": round(st.st_size / 1048576.0, 1), "non_weight_mb": 0.0,
            "init": 0, "src": ""}
    try:
        import mmap
        with open(key, "rb") as fh:
            buf = mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ)
            try:
                graph = None
                for fno, wt, v in _pb_fields(buf, 0, len(buf)):
                    if fno == 7 and wt == 2:          # ModelProto.graph
                        graph = v
                        break
                if graph is None:
                    info["src"] = "不是 ONNX（找不到 graph）→ 退回文件体积"
                else:
                    nbytes = params = init = 0
                    px = 0
                    for fno, wt, v in _pb_fields(buf, graph[0], graph[0] + graph[1]):
                        if fno == 5 and wt == 2:      # initializer
                            b, n, _dt = _pb_tensor(buf, v[0], v[0] + v[1])
                            nbytes += b
                            params += n
                            init += 1
                        elif fno == 11 and wt == 2 and not px:
                            px = _pb_input_px(buf, v[0], v[0] + v[1])
                    if nbytes <= 0:
                        info["src"] = "initializer 是 external data → 退回文件体积"
                    else:
                        info.update({"weights_mb": round(nbytes / 1048576.0, 1),
                                     "params": int(params),
                                     "params_m": round(params / 1e6, 2),
                                     "size_px": int(px), "init": int(init),
                                     "non_weight_mb":
                                         round((st.st_size - nbytes) / 1048576.0, 1),
                                     "src": "ONNX initializer 实测"})
            finally:
                buf.close()
    except Exception as e:                        # noqa: BLE001
        info["src"] = "ONNX 解析失败（%s）→ 退回文件体积" % str(e)[:80]
    with _DFM_INFO_LOCK:
        _DFM_INFO[key] = dict(info)
    return dict(info)


def _vram_model_mb(path) -> float:
    """一份 .dfm 的**权重**体量（MB）—— 取 ONNX initializer 之和，不是文件体积、更不是名字。

    实测（本机两份真模型，_dfm_info 直接扫图）：
        张元英.dfm        输入 384 · 权重 **1566.9MB**（410.76M 参数 · 94 个 initializer）
        阿証証証証-416.dfm 输入 416 · 权重 **514.2MB**（134.79M 参数 · 78 个 initializer）
    两份的「文件体积 − 权重」分别是 0.0 / 0.1MB（dfm 里没有别的数据），所以文件体积在这两
    份上是准的 —— 但**口径**必须是权重：外部权重、带训练残留、或将来换成 fp16 时，体积就
    不再等于权重了。读不出图（不是 ONNX / external data）才退回文件体积，并在日志里写明。
    """
    key = str(path or "")
    if not key:
        return 0.0
    info = _dfm_info(key)
    mb = float(info.get("weights_mb") or 0.0)
    if mb > 0:
        return mb
    try:                                        # 兜底：文件体积（src 里已写明原因）
        return max(0.0, os.stat(key).st_size / 1048576.0)
    except OSError:
        return 0.0


def _vram_models_mb(paths) -> float:
    """一组 dfm 里最大的那份。

    多模型导出时它们常驻在同一个进程里（上限 _MAX_MODELS=3），但同一帧只会用到其中
    一路、而且 A 卸 B 载是按 LRU 走的 —— 按**最大那份**算比按和算更接近峰值，
    又不至于把 3 份 0.5GB 的模型误判成装不下。
    """
    best = 0.0
    for p in (paths or []):
        best = max(best, _vram_model_mb(p))
    return best


# ------------------------------------------------------------
# 项目导出：**一份 worker 会话**真跑起来占多少（实测优先，标定值兜底）
# ------------------------------------------------------------
# 为什么要单独记这一笔：同一份 .dfm，「会话建好」和「跑了几十帧」不是一个数 ——
# 前者是权重 + context（384：2079MB），后者还要加 ORT/CuDNN 那截会长大的工作集
# （实测 384 一份 ≈ 3470MB、416 一份 ≈ 2220MB，见 _VRAM_SESS_RUNTIME_MB 的注释）。
# 预算按前者算就会放行装不下的 worker 数 → WDDM 换页 → 帧速崩到 3.61。
#
# 口径：跑完一单就把「整卡峰值 − 开跑前的整卡占用」摊到每个 worker 上，记进
# <WORKSPACE>/_vram_sess_peak.json（和 _sess_bench.json 一个套路）；下一次算账就用
# max(标定值, 实测值) —— 实测只能把预估**抬高**（更保守），不会把已经放行的单子
# 悄悄放宽。峰值贴到 88% 以上的那一单（已经在换页了）不记：那种数是被墙压扁的。
_VRAM_SESS_PEAK = {}
_VRAM_SESS_LOCK = threading.RLock()
_VRAM_SESS_PAGE_PCT = 0.88
# 记账口径的版本号：口径一变（比如 dfm 批改成「按模型查表 + 乘进程数」），旧文件里那些
# 按旧口径反推出来的会话体量就是错的（它把批成本吞进了会话里），必须整批判掉。
_VRAM_SESS_VER = 2


def _vram_sess_key(dfm_path) -> str:
    """模型身份 = 名字 + 体积 + mtime：换模型 / 重训导出之后实测值自动失效。"""
    key = str(dfm_path or "")
    if not key:
        return ""
    try:
        st = os.stat(key)
        return "%s|%d|%d" % (Path(key).name, int(st.st_size), int(st.st_mtime))
    except OSError:
        return "%s|0|0" % Path(key).name if key else ""


def _vram_sess_path() -> Path:
    return Path(WORKSPACE) / "_vram_sess_peak.json"


def _vram_sess_load() -> dict:
    try:
        p = _vram_sess_path()
        if p.is_file():
            d = json.loads(p.read_text("utf-8", "ignore") or "{}")
            if isinstance(d, dict):
                return d
    except Exception:                                       # noqa: BLE001
        pass
    return {}


def _vram_sess_put(key: str, row: dict):
    """记一笔实测（内存 + 工作区 json；写不进去只影响下一次的预估，不影响这一单）。"""
    if not key:
        return
    with _VRAM_SESS_LOCK:
        _VRAM_SESS_PEAK[str(key)] = dict(row)
    try:
        d = _vram_sess_load()
        d[str(key)] = dict(row)
        _vram_sess_path().write_text(json.dumps(d, ensure_ascii=False, indent=1), "utf-8")
    except Exception:                                       # noqa: BLE001
        pass


def _vram_sess_get(key: str) -> dict:
    if not key:
        return {}
    with _VRAM_SESS_LOCK:
        hit = _VRAM_SESS_PEAK.get(str(key))
    if hit is None:
        hit = _vram_sess_load().get(str(key))
    if not isinstance(hit, dict):
        return {}
    if int(hit.get("v") or 0) != int(_VRAM_SESS_VER):   # 旧口径记的：整批判掉
        return {}
    return dict(hit)


def _vram_sess_peak_mb(paths) -> dict:
    """一份项目导出 worker 会话的**峰值**（MB）—— 多模型时取最大的那一份。

    返回 {"sess_mb", "src", "model_mb", "measured_mb", "estimate_mb", "key", "row"}：
      · estimate = 权重 + context + _VRAM_SESS_RUNTIME_MB（实测标定的运行时工作集）；
      · measured = 上一单真跑出来的「整卡峰值增量 / worker 数」（没有就空）；
      · 采用 max(estimate, measured) —— 实测只抬高、不放松（见上面那一段）。
    """
    paths = [str(p) for p in (paths or []) if str(p or "")]
    model_mb = _vram_models_mb(paths)
    est = float(model_mb) + float(_VRAM_CTX_MB) + float(_VRAM_SESS_RUNTIME_MB)
    best_mb, best_row = 0.0, {}
    for p in paths:
        row = _vram_sess_get(_vram_sess_key(p))
        try:
            mb = float(row.get("sess_mb") or 0.0)
        except (TypeError, ValueError):
            mb = 0.0
        if mb > best_mb:
            best_mb, best_row = mb, row
    sess = max(est, best_mb) if best_mb > 0 else est
    # 兜底上限：实测值再离谱也不许超过整卡的 60%（那是「读数坏了」而不是这台机器）
    try:
        _free, total, _src = _vram_probe()
        if total:
            sess = min(sess, float(total) * 0.6)
    except Exception:                                       # noqa: BLE001
        pass
    return {"sess_mb": round(float(sess), 1),
            "src": ("实测 max(标定 %.0f, 量到 %.0f)" % (est, best_mb)) if best_mb > 0
                   else ("标定 %.0f = 权重 %.0f + context %d + 运行时 %d"
                         % (est, model_mb, int(_VRAM_CTX_MB), int(_VRAM_SESS_RUNTIME_MB))),
            "model_mb": round(float(model_mb), 1),
            "measured_mb": round(float(best_mb), 1), "estimate_mb": round(est, 1),
            "key": _vram_sess_key(paths[0]) if paths else "", "row": dict(best_row)}

def _proj_record_sess_peak(dfms, n_workers: int, batch: int, base_used_mb,
                           peak_used_mb, total_mb, size_px=None,
                           ok: bool = True) -> dict:
    """把这一单**实测**的「一份 worker 会话峰值」记下来（下一单的预算直接用它）。

    口径：ours = 整卡峰值 − 开跑前的整卡占用（含预热那几秒）；一份 = ours / worker 数。
    不采纳的四种情形（宁可不记，也不许把预算带偏）：
      · 这一单不是正常跑完的（取消/有失败帧）—— 峰值没跑到；
      · 峰值贴到整卡 88% 以上 —— 已经在换页，读数是墙压出来的（正是要修的病症）；
      · 量出来的比权重还小 —— 读数坏了（探测到的是别人的进程在退）；
      · 多模型项目（LRU 换进换出，摊不到单份上）。
    """
    model_mb = _vram_models_mb(dfms)
    try:
        base = float(base_used_mb or 0.0)
        peak = float(peak_used_mb or 0.0)
        total = float(total_mb or 0.0)
    except (TypeError, ValueError):
        return {}
    if not ok or len([p for p in (dfms or []) if str(p or "")]) != 1:
        return {}
    if peak <= 0 or base <= 0 or peak <= base:
        return {}
    if total and peak >= total * float(_VRAM_SESS_PAGE_PCT):
        print("[project] 峰值 %.0f/%.0f MB = %.0f%% 已经贴墙（换页），这一次的会话体量不记账"
              % (peak, total, 100.0 * peak / total), flush=True)
        return {}
    per = (peak - base) / float(max(1, int(n_workers)))
    # 把预算里**单列的那两项**扣回去，否则同一笔钱记两遍：
    #   · dfm 批（这一档线程数）—— 下一单按它自己的 batch 再加；
    #   · 检测/遮罩（_VRAM_AUX_MB）—— 多 worker 时它只算一份，摊到每份头上就是白记。
    # 剩下的才是「一份会话」：权重 + context + 运行时工作集 + 本进程那点 torch context。
    try:
        per -= float(_vram_batch_mb(int(size_px or _VRAM_BASE_PX), int(batch),
                                    model_mb, "procs"))
        per -= float(_VRAM_AUX_MB)
    except Exception:                                   # noqa: BLE001
        pass
    # 实测只用来**纠偏**：抬到标定值的 1.5 倍以上就不像话了（多半是本进程那份 torch
    # context / 别人的进程在那一拍抖了一下），按 1.5 倍记。
    est = float(model_mb or 0.0) + float(_VRAM_CTX_MB) + float(_VRAM_SESS_RUNTIME_MB)
    per = min(float(per), est * 1.5)
    if per < float(model_mb or 0.0):
        return {}
    row = {"v": int(_VRAM_SESS_VER), "sess_mb": round(float(per), 1),
           "workers": int(n_workers), "batch": int(batch), "base_used_mb": round(base, 1),
           "peak_used_mb": round(peak, 1), "total_mb": round(total, 1),
           "at": time.strftime("%Y-%m-%d %H:%M:%S")}
    key = _vram_sess_key(dfms[0])
    _vram_sess_put(key, row)
    print("[project] 实测一份 worker 会话 %.2fGB（整卡峰值 %.0f − 开跑前 %.0f，%d 个 worker）"
          "→ 记进 _vram_sess_peak.json，下一单的预算按它算"
          % (per / 1024.0, peak, base, int(n_workers)), flush=True)
    return row


def _dfm_size_hint(dfm) -> int:
    """这个 .dfm 的输入边长（384/416/…）——**不加载模型**就要知道，因为预算要在加载前算。

    口径：**问图要**（ONNX 的 graph.input 形状，_dfm_info），不问文件名 —— 文件名里的数字
    会骗人（本仓库两份真模型：名字里带 416 的那份输入确实是 416，不带数字的 张元英.dfm
    其实是 **384**，老代码按名字取不到就退回 416，预算里的帧/批两项因此一直算偏）。
    名字只作为最后的兜底，再不行才 416。
    真正喂给 session 的尺寸仍由 session 自己决定：_fused_run 加载后会拿实测尺寸
    把这个预算**再算一遍**（不一样就按新尺寸重新判一次）。
    """
    key = str(dfm or "")
    with _VRAM_LOCK:
        hit = _VRAM_SIZE_CACHE.get(key)
    if hit:
        return int(hit)
    size = int((_dfm_info(key) or {}).get("size_px") or 0)
    if size:
        with _VRAM_LOCK:
            _VRAM_SIZE_CACHE[key] = int(size)
        return int(size)
    try:
        stem = Path(key).stem
        cand = [int(m) for m in re.findall(
            r"(?<![0-9])(256|288|320|384|416|448|480|512|640)(?![0-9])", stem)]
        if cand:
            size = max(cand)
    except Exception:                                       # noqa: BLE001
        size = 0
    if not size:
        size = _VRAM_BASE_PX
    with _VRAM_LOCK:
        _VRAM_SIZE_CACHE[key] = int(size)
    return int(size)


def _vram_frame_mb(size_px: int) -> float:
    """一帧在飞时压在 GPU 上大约多少 MB：dfm 输入 + 输出脸 + prd + dst 四份 float32。

    416 → 7.92MB（用户口径 30×7.9 ≈ 237MB）；512 → 12.0MB。
    """
    size_px = max(64, int(size_px or _VRAM_BASE_PX))
    return max(4.0, (size_px * size_px * 3 * 4 * 4) / 1048576.0)


def _vram_batch_table(model_mb=None, mode: str = "frames"):
    """挑那张批台阶表 → (量的输入边长, 点集)。

    · `mode == "procs"`（项目导出）：按**权重体量**挑 —— ≤900MB 用 416 那张
      （阿証…-416.dfm，514MB），更大的用 384 那张（张元英.dfm，1567MB）；
    · 其它（fused 帧预算 / 预览）：老表 `_VRAM_BATCH_PTS`（量的是「dfm 前向 k 张」，
      不是项目那条路的「k 帧合成 + 落盘」）；基准边长 416。
    """
    if str(mode) != "procs":
        return int(_VRAM_BASE_PX), _VRAM_BATCH_PTS
    mb = float(model_mb or 0.0)
    for lim, base_px, pts in _VRAM_BATCH_TABLES:
        if mb <= float(lim):
            return int(base_px), pts
    return int(_VRAM_BATCH_TABLES[-1][1]), _VRAM_BATCH_TABLES[-1][2]


def _vram_batch_mb(size_px: int, batch, model_mb=None, mode: str = "frames") -> float:
    """dfm 一次前向喂 batch 张时**多占**的显存（相对同一份会话 batch=1）。

    点集按模型体量挑（_vram_batch_table），再按 (size/基准边长)² 缩放：基准边长就是
    那张表量的时候那个模型的输入（384 表在 384 上量的，416 表在 416 上量的）。
    """
    b = int(batch or 0)
    if b <= 0:
        return 0.0
    base_px, pts = _vram_batch_table(model_mb, mode)
    k = (max(64, int(size_px or base_px)) / float(base_px)) ** 2
    if b <= pts[0][0]:
        return pts[0][1] * k
    for i in range(1, len(pts)):
        b0, c0 = pts[i - 1]
        b1, c1 = pts[i]
        if b <= b1:
            return (c0 + (c1 - c0) * (b - b0) / float(b1 - b0)) * k
    b_last, c_last = pts[-1]
    b_prev, c_prev = pts[-2]
    slope = (c_last - c_prev) / float(b_last - b_prev)
    return (c_last + slope * (b - b_last)) * k


def _vram_reserve_mb(free_mb) -> float:
    """这次该给驱动/碎片/别人插队留多少 MB —— **可解释，而且不给 8GB 卡留死重**。

    口径：reserve = min(768, max(256, free × 5%))。

    为什么不是老的「free × 20% + 512MB」（那在 8GB 卡空闲 3.4GB 时是 1.19GB，
    再叠上「一份会话 / 一帧缓冲」之后正好变成用户看到的「需要 2.46GB / 可用 2.03GB」）：

      · **20% 是拍的，不是量的。** 它出在 ceaf4f4 第一版预算里（提交信息只有
        「avail=free×0.8-512MB」，没有任何测量支撑）。而真正会被预留吸收的那两项
        早就有实测，而且都已经落在 need 里：
          - CUDA context：每份会话 512MB，need 里明着算（_VRAM_CTX_MB）；
          - ORT 的 CUDA arena：416 实测 +135MB、384 实测 +221MB（会话建好那一刻），
            而「batch 4 起 arena 换池子」那一步的 +1036MB **已经包含在
            _VRAM_BATCH_PTS 的实测台阶里**（那张表量的是「整卡增量 − 会话本身」），
            再拿一个 20% 去覆盖它 = 同一笔钱记两遍。
      · 真正需要预留的是**探测滞后**和**别人插队**，量级都在几百 MB：
          - P0 校准：我们的 free 与 nvidia-smi 原始行差 ≤256MB（两条路取小之后）；
          - 探测有 2~4s 缓存（_VRAM_PROBE_TTL / _SMI_MEM），这期间浏览器开个视频
            硬件加速就能抢走几百 MB；
          - 用户改一次参数会连打几次 dry_run，中间那点抖动不该被判成「不够」。
        所以底线取 256MB 正好覆盖第一条，5% 让大空闲时自动放宽一点，
        上限 768MB 防止 24GB 卡上又长出一个 1.2GB 的死重。

    实测佐证（本机 RTX 5070 Laptop 8GB）：按新口径算，「空闲 3.17GB / 需要 2.46GB」
    的单子直接放行，真跑 12 帧 0 OOM（app/tools/accept/vram_force_accept.py 的 A 段）；
    而按老口径它必被拒 —— 那 1.37GB 里没有一个字节是实测需要的。
    """
    free = max(0.0, float(free_mb or 0.0))
    # 对照/回滚用：DFL_VRAM_RESERVE_MODE=legacy20 复原老口径（空闲 20% + 512MB）。
    # （下面的 _vram_reserve_rule_text 把同一件事说成一句人话，文案里照抄它。）
    # 只给验收脚本做「新旧两套数字并排」用 —— 生产默认 auto（上面的新口径）。
    if str(os.environ.get("DFL_VRAM_RESERVE_MODE") or "").strip().lower() in ("legacy", "legacy20", "old"):
        return free * 0.20 + 512.0
    return min(float(_VRAM_RESERVE_MAX_MB),
               max(float(_VRAM_RESERVE_MB), free * float(_VRAM_SAFETY)))

def _vram_cpu_fallback_env() -> bool:
    return str(os.environ.get(_VRAM_ENV_CPU) or "").strip().lower() in ("1", "true", "yes", "on")


def _vram_cpu_on() -> bool:
    """当前是不是「强制 CPU EP」（显存不够 + 用户开了 DFL_ALLOW_CPU_FALLBACK）。

    _dfm_session / _stage_xseg 都读它 —— 一处开关，全链路一致，不会出现
    「dfm 在 CPU 上、XSeg 还在抢显存」这种半吊子状态。
    """
    with _CPU_EP_LOCK:
        return bool(_CPU_EP["on"])


def _vram_set_cpu(on: bool, why: str = "") -> bool:
    on = bool(on)
    with _CPU_EP_LOCK:
        was = bool(_CPU_EP["on"])
        _CPU_EP["on"] = on
        _CPU_EP["why"] = str(why or "") if on else ""
    if on != was:
        print("[vram] %s CPU EP（%s）"
              % ("★ 切到" if on else "解除 CPU（回到 GPU）", why or "显存预算恢复"), flush=True)
    return on


def _vram_cands(mode: str, want_frames: int, want_batch: int,
                want_procs: int, procs: int) -> list:
    """候选计划：从「最想要」到「最省」。

    · frames —— fused 视频导出：先保帧预算（流水线的重叠靠它），再降 dfm 批
      （416 上 B1/B4/B8 每张耗时几乎一样，所以降批几乎不花钱）；
    · procs  —— 项目导出：先降 worker 进程数（每份 session + 一个 CUDA context），
      减到 1 个之后再降 worker 内的线程数（batch=几帧并行）；两样都到最小才拒绝；
    · none   —— 预览：只问「一份会话装不装得下」。
    """
    if mode == "none":
        return [{"frames": 0, "batch": 1, "procs": max(1, int(procs))}]
    if mode == "procs":
        # 降档顺序是用户定的：**先减 worker（每份 = 一整套会话 + 一个 CUDA context），
        # 减到 1 个之后才降 worker 内的线程数（batch），两样都到最小还装不下才拒绝**。
        # 以前只列 procs、batch 永远按 1 算 —— 于是「1 个 worker 也装不下」时
        # 直接 400，而其实把 2 线程降到 1 线程就够（dfm 批那一项 160 → 80MB 也是钱）。
        procs_ladder = [{"frames": 0, "batch": int(want_batch), "procs": n}
                        for n in range(max(1, int(want_procs)), 0, -1)]
        batches = sorted({int(b) for b in (want_batch, 4, 2, 1)
                          if 1 <= int(b) < int(want_batch)}, reverse=True)
        return procs_ladder + [{"frames": 0, "batch": int(b), "procs": 1} for b in batches]
    frames_ladder = sorted({int(f) for f in (want_frames, int(_FUSED_BUDGETS[1]),
                                             int(_FUSED_BUDGETS[2]), int(_FUSED_BUDGETS[-1]))
                            if 1 <= int(f) <= int(want_frames)}, reverse=True) \
        or [int(want_frames)]
    batch_ladder = sorted({int(b) for b in (want_batch, 4, 2, 1)
                           if 1 <= int(b) <= int(want_batch)}, reverse=True) \
        or [int(want_batch)]
    out = []
    for n in [max(1, int(procs))]:               # fused：单进程，只有帧/批两个旋钮
        for f in frames_ladder:
            for b in batch_ladder:
                out.append({"frames": int(f), "batch": int(b), "procs": int(n)})
    return out


def _vram_need_mb(cand: dict) -> float:
    """一个候选/计划的总需求（MB）：会话×进程 + 检测遮罩 + 帧缓冲 + dfm 批。

    resident = 这个进程**已经**持有几份会话（fused 加载完 dfm 后重算那一拍 = 1）：
    那几份的「权重 + context」已经在卡上了，不再算进需求 —— 再算一遍就是重复计数，
    会把一个本来跑得动的单子判死（用户截图那一单就是这么被拒的）。
    """
    procs = max(1, int(cand.get("procs") or 1))
    new_sess = max(0, procs - max(0, int(cand.get("resident") or 0)))
    # dfm 批是**每个进程各自**的（worker 里 k 个线程同时前向），所以要乘进程数 ——
    # 只在外面加一次会漏掉「2 个 worker × 2 线程」的那份（实测差 1.4GB）。
    return (float(cand.get("sess_mb") or 0.0) * new_sess
            + float(cand.get("batch_mb") or 0.0) * procs
            + float(cand.get("aux_mb") or 0.0)
            + int(cand.get("frames") or 0) * float(cand.get("frame_mb") or 0.0))


def _vram_plan(model_mb, opts: dict = None) -> dict:
    """统一的显存预算：模型体量 + 选项 → 计划 batch / 帧预算 / 预计峰值 / 可行?

    opts:
      mode       frames（fused 视频导出，默认）| procs（项目导出 worker）| none（预览）
      size_px    dfm 输入边长（默认 416；用 _dfm_size_hint 拿，别加载模型）
      frames     想要的帧缓冲（默认 30 = _FUSED_BUDGETS[0]）
      batch      想要的 dfm 批（默认 _FUSED_BATCH_TARGET）
      procs      固定要几份会话（fused/preview = 1）
      max_procs  mode=procs 时的上限（默认 = procs）
      aux        要不要算检测器/关键点/XSeg（默认 True）
      label      用途名（进日志与 status）
      free_mb / total_mb  覆盖探测量（单测用；不传就走 _vram_probe）
      reclaim    起算之前先**回收**我们自己占的那一块（常驻预览 worker），默认关；
                 导出闸门开它 —— 用户「预览过一帧再点导出」被拒的病根就在这里
      resident   本进程**已经**持有这份会话了（fused 加载后重算那一拍）：会话那一项
                 不再计入需求，否则等于把一个已经吃掉的 2GB 又算一遍 → 必被拒
      sess_mb / sess_src  **一份会话的峰值**直接给（项目导出用）：给了就不再用
                 「权重 + context」推 —— 那条路一份会话真跑起来还要多 1.3GB，
                 见 _vram_sess_peak_mb / _VRAM_SESS_RUNTIME_MB 的实测注释

    返回 dict（关键字见下面 plan = {...}）。**ok=False 的调用方必须拒绝**：
    _vram_plan 只算账，不抛异常（拒绝的样子由 _vram_guard / 调用点决定）。
    开了 DFL_ALLOW_CPU_FALLBACK 且装不下时，返回 ok=True + cpu=True。
    """
    opts = dict(opts or {})
    mode = str(opts.get("mode") or "frames")
    label = str(opts.get("label") or "显存预算")
    size_px = int(opts.get("size_px") or _VRAM_BASE_PX)
    procs = max(1, int(opts.get("procs") or 1))
    want_frames = int(opts.get("frames") or 0) or int(_FUSED_BUDGETS[0])
    want_frames = max(int(_FUSED_BUDGETS[-1]), min(64, int(want_frames)))
    want_batch = int(opts.get("batch") or 0) or int(_FUSED_BATCH_TARGET)
    want_batch = max(1, min(int(_PRED_BATCH_CAP), int(want_batch)))
    max_procs = max(1, min(16, int(opts.get("max_procs") or procs)))
    aux = float(_VRAM_AUX_MB) if opts.get("aux", True) else 0.0
    # resident：这份会话（含它自己的 CUDA context）**已经在当前进程里**了。
    # 只对「同一进程、同一份会话」成立，而且只在重算那一拍用（见 _fused_run）。
    resident = 1 if opts.get("resident") else 0
    model_mb = max(0.0, float(model_mb or 0.0))
    # 模型身份（参数量 / 权重 / 输入边长）：由调用方从 .dfm 的图里读出来（_dfm_info），
    # 这里只负责挂到 plan 上，让 status/日志/拒绝消息都能说清「凭什么按这一档」。
    minfo = dict(opts.get("model_info") or {})
    # 一份会话 = 权重 + 它自己的 CUDA context。ORT 的 arena（416 建好时 +135MB）不单列
    # —— 它由「空闲的 5% + 256MB」那份预留吸收。但**项目导出**那条路上会话跑起来还要
    # 再长一块（实测 +1190 / +1390MB，见 _VRAM_SESS_RUNTIME_MB）：那一块必须明着算，
    # 否则 384 模型 2 个 worker 会被算成 4538MB（≤ 可用）放行，真跑起来顶到 94.9%
    # → WDDM 换页 → 3.61 帧/秒。调用方给了 sess_mb（实测值）就照给的算。
    sess_override = opts.get("sess_mb")
    if sess_override is not None:
        sess_mb = max(float(sess_override or 0.0), model_mb)
        sess_src = str(opts.get("sess_src") or "调用方给的实测值")
    else:
        sess_mb = model_mb + float(_VRAM_CTX_MB)
        sess_src = "权重 + context"
        if mode == "procs":
            sess_mb += float(_VRAM_SESS_RUNTIME_MB)
            sess_src = ("权重 + context + 运行时工作集 %dMB（标定值）"
                        % int(_VRAM_SESS_RUNTIME_MB))
    frame_mb = _vram_frame_mb(size_px)
    # ---- ★ 顺序（用户口径，别改回去）：先回收我们自己占的那块 → 再探测 → 再算预算 ----
    # 以前是「先按现状算，**装不下才**回收」，而阶梯里永远有 8 帧 / 批 1 这一档 ——
    # 于是「装得下（降档）」成立、回收那一步一次都没走到：日志里「可回收 2733 MB」进了
    # 「显存去向」那一栏，公式里却是 0（reclaim_left 初值写死 0），帧预算 30 → 8、
    # dfm 批 8 → 1，用户实测 22 → 15.8 帧/秒。
    # reclaim_first=True = 导出真起手那一拍（路由闸门 + worker 线程里的那一次）：
    #   卸掉常驻预览 worker 的模型，**用回收后的空闲**算账；dry_run 预检不传它
    #   （预检不该为了算账去卸模型：拖一次滑杆就重载 4 秒），预检靠下面
    #   「可回收计入可用」这一条拿到同样的结论，真起手时再真回收。
    rec = {}
    if opts.get("free_mb") is not None:
        free, total, src = int(opts["free_mb"]), opts.get("total_mb"), "override"
    else:
        if opts.get("reclaim") and opts.get("reclaim_first"):
            rec = _vram_reclaim_once(label)
        # 开过回收就**重新探**（ttl=0）：拿回收前的旧数算账等于白回收一遍
        free, total, src = _vram_probe(0.0 if rec else None)

    def _mk(frames, batch, nproc):
        # resident=1 时，第一份会话的「权重 + context」已经在这块卡上了（本进程持有），
        # 需求里只剩**增量**：检测/遮罩 + 帧缓冲 + dfm 批。
        cand = {"sess_mb": round(sess_mb, 1), "aux_mb": round(aux, 1),
                "frame_mb": round(frame_mb, 2), "frames": int(frames),
                "batch": int(batch),
                "batch_mb": round(_vram_batch_mb(size_px, batch, model_mb, mode), 1),
                "procs": int(nproc), "resident": int(resident),
                "resident_mb": round(float(sess_mb) * min(int(nproc), int(resident)), 1)}
        cand["need_mb"] = round(_vram_need_mb(cand), 1)
        return cand

    ladder = [_mk(c["frames"], c["batch"], c["procs"])
              for c in _vram_cands(mode, want_frames, want_batch, max_procs, procs)]
    floor = ladder[-1]
    plan = {
        "ok": False, "cpu": False, "mode": mode, "label": label, "size_px": int(size_px),
        "model_mb": round(model_mb, 1), "sess_mb": round(sess_mb, 1),
        # 模型体量的来路：参数量 / 权重 MB / 输入边长（全部来自 ONNX 图，不看文件名）
        "params_m": (round(float(minfo.get("params_m") or 0.0), 2)
                     if minfo.get("params_m") else None),
        "params": (int(minfo["params"]) if minfo.get("params") else None),
        "model_mb_src": str(minfo.get("src") or "文件体积（没读到图）"),
        "file_mb": float(minfo.get("file_mb") or 0.0) or None,
        "non_weight_mb": round(float(minfo.get("non_weight_mb") or 0.0), 1),
        "size_px_src": ("ONNX 图输入" if minfo.get("size_px") else "文件名/默认"),
        # 一份会话里「权重 + context」之外的那截（项目导出的运行时工作集 / 实测值）
        "sess_src": sess_src,
        "sess_extra_mb": round(max(0.0, float(sess_mb) - model_mb - float(_VRAM_CTX_MB)), 1),
        "ctx_mb": int(_VRAM_CTX_MB), "arena_mb": int(_VRAM_ARENA_MB),   # arena 只在日志里提一句
        "aux_mb": round(aux, 1), "frame_mb": round(frame_mb, 2),
        # 预留：随空闲缩放（free×5%，底线 256MB、上限 768MB），见 _vram_reserve_mb
        "safety": float(_VRAM_SAFETY),
        "reserve_mb": round(_vram_reserve_mb(free), 1),
        "reserve_min_mb": int(_VRAM_RESERVE_MB), "reserve_cap_mb": int(_VRAM_RESERVE_MAX_MB),
        "reserve_pct_mb": round(float(free or 0.0) * float(_VRAM_SAFETY), 1),
        "forced": False,
        "free_mb": (None if free is None else int(free)),
        "total_mb": (None if total is None else int(total)),
        "source": src, "avail_mb": None, "need_mb": round(floor["need_mb"], 1),
        "peak_mb": None, "procs": int(max_procs if mode == "procs" else procs),
        # 谁占着显存 + 我们回收掉了多少（拒绝消息/status/日志都读这几个键）
        "reclaim": dict(rec or {}),
        "reclaimed_mb": round(float((rec or {}).get("freed_mb") or 0.0), 1),
        "reclaim_ms": round(float((rec or {}).get("ms") or 0.0), 1),
        "reclaim_note": str((rec or {}).get("why") or ""),
        "reclaim_left_mb": 0.0,
        "own_mb": None, "own_inproc_mb": None, "own_preview_mb": None,
        "reclaimable_mb": None, "other_mb": None, "preview_models": [],
        "free_torch_mb": None, "free_smi_mb": None, "resident_mb": round(floor["resident_mb"], 1),
        "frames": int(want_frames if mode != "procs" else 0),
        # procs 模式下 batch = worker 内的线程数（几帧并行），已经是预算的一部分了
        "batch": int(want_batch),
        "batch_mb": round(_vram_batch_mb(size_px, want_batch, model_mb, mode), 1),
        "requested_frames": int(want_frames), "requested_batch": int(want_batch),
        "requested_procs": int(max_procs if mode == "procs" else procs),
        "downgraded": False, "steps": [], "reason": "", "msg": "", "suggest": [],
        # 老字段（_fused_budget 的调用方/验收脚本还在读）
        "frame_budget": int(want_frames if mode != "procs" else 0),
        "requested": int(want_frames), "per_frame_mb": round(frame_mb, 2),
    }
    if free is None:
        plan.update({"ok": True,
                     "reason": "拿不到显存信息（torch / nvidia-smi 都没答）→ 不猜、不降档",
                     "msg": "%s：拿不到显存信息 → 按请求值跑（帧预算 %d / 批 %d）"
                            % (label, plan["frames"], plan["batch"])})
        return plan
    # ---- 谁占着这块卡（本程序 / 其它程序 / 空闲）+ 可回收多少 ----
    def _apply_own(o):
        plan.update({"own_mb": o.get("own_mb"),
                     "own_inproc_mb": o.get("own_inproc_mb"),
                     "own_preview_mb": o.get("own_preview_mb"),
                     "own_parts": dict(o.get("own_parts") or {}),
                     "reclaimable_mb": o.get("reclaimable_mb"),
                     "other_mb": o.get("other_mb"),
                     "preview_models": list(o.get("preview_models") or []),
                     "free_torch_mb": o.get("free_torch_mb"),
                     "free_smi_mb": o.get("free_smi_mb")})

    def _avail_for(extra_mb):
        # 口径：可用 = 空闲 − 预留(free) + 可回收。
        #   · 「+ 可回收」：我们自己占着 2GB 但随时能收回来，不该算成「这台机器不够跑」；
        #   · 「− 预留」：预留**随空闲缩放**（5%，底线 256 / 上限 768），
        #     不是老的 20%+512MB —— 见 _vram_reserve_mb 那一大段实测依据。
        reserve = float(_vram_reserve_mb(free))
        plan["reserve_mb"] = round(reserve, 1)
        plan["reserve_pct_mb"] = round(float(free) * float(_VRAM_SAFETY), 1)
        return max(0.0, float(free) - reserve + float(extra_mb))

    def _pick(avail_value):
        plan["avail_mb"] = round(avail_value, 1)
        plan["steps"] = []
        first = None
        for cand in ladder:
            fit = bool(cand["need_mb"] <= avail_value)
            plan["steps"].append({"frames": int(cand["frames"]),
                                  "batch": int(cand["batch"]),
                                  "procs": int(cand["procs"]),
                                  "need_mb": cand["need_mb"],
                                  "batch_mb": cand["batch_mb"],
                                  "avail_mb": round(avail_value, 1), "fit": fit})
            if fit and first is None:
                first = cand
        return first

    own = _vram_owners(free, total, src)
    _apply_own(own)
    # ---- 可回收**真的加进公式** ----
    # 这一条以前是写死的 0：日志第二行把 reclaim_left_mb 当「+ 可回收」打出来，
    # 而它只在「装不下才回收」那个分支里赋值 —— 那个分支永远走不到，于是
    # 「可回收 2733 MB」只是个装饰数字，预算里加的是 0。
    # 口径：回收**没成**（这回没卸 / 卸了没腾出来）时，那 N MB 仍按「可回收」计入
    # 可用 —— 因为导出起手时还会再收一次；回收**成了**就不再计（那 N MB 已经算在
    # 新的 free 里，重复加等于骗自己）。
    # 只对真探测的 free 生效：env/file/override 是伪造/单测值，拿它加可回收就是假账。
    reclaim_left = 0.0
    if opts.get("reclaim") and not (rec or {}).get("did") and _vram_src_real(src):
        reclaim_left = float(own.get("reclaimable_mb") or 0.0)
    plan["reclaim_left_mb"] = round(reclaim_left, 1)
    avail = _avail_for(reclaim_left)
    hit = _pick(avail)

    # ---- 装不下才回收我们自己占的那块（预览 worker 里常驻的 dfm），然后算第二遍 ----
    # 回收成功 → 空闲里已经含那 2GB，**不再**把它加进 avail（重复算等于骗自己）；
    # 没收成（worker 正忙/超时）→ 那 2GB 仍按「可回收」计入 avail，消息里注明
    # 「导出前会再收一次」（_vram_reclaim 在起 job 前还会被调一次）。
    # 三档模式（frames / procs / none）都适用：阶段链的闸门也是 mode=none。
    # 预览自己不会走到这里 —— _preview_guard 明确传 reclaim=False（它要用的正是
    # worker 里那份常驻模型）。
    # 补一次回收：只在「这一拍还没收过」且真的降到底才走（dry_run 预检的第一拍就会
    # 命中这里 —— 预检不该先卸模型，但真降到底时宁可收一次也别把一个能跑的单子拒了）。
    if hit is None and opts.get("reclaim") and opts.get("free_mb") is None and not rec:
        rec = _vram_reclaim_once(label)
        plan.update({"reclaim": dict(rec or {}),
                     "reclaimed_mb": round(float((rec or {}).get("freed_mb") or 0.0), 1),
                     "reclaim_ms": round(float((rec or {}).get("ms") or 0.0), 1),
                     "reclaim_note": str((rec or {}).get("why") or "")})
        free2, total2, src2 = _vram_probe(0.0)
        if free2 is not None and (rec.get("did") or int(free2) > int(free)):
            free = free2
            if total2 is not None:
                total = total2
            src = src2
            plan["free_mb"] = int(free)
            plan["total_mb"] = (None if total is None else int(total))
            plan["source"] = src
        own = _vram_owners(free, total, src)
        _apply_own(own)
        # 没收成 → 那 N MB 仍按「可回收」计入（起 job 前 _fused_run/_stage_worker 会再收
        # 一次；这一拍先别把它判死）。收成了 → 已经在 free 里，计 0，绝不重复加。
        reclaim_left = (float(own.get("reclaimable_mb") or 0.0)
                        if (not rec.get("did") and _vram_src_real(src)) else 0.0)
        plan["reclaim_left_mb"] = round(reclaim_left, 1)
        avail = _avail_for(reclaim_left)
        hit = _pick(avail)

    # ---- 还装不下？那就放掉**本进程**常驻的会话（预览/下一次导出再懒加载）----
    # 只在「没有任何任务在跑」时做：有任务在跑时放掉 session 等于把它的腿锯了。
    if (hit is None and opts.get("reclaim_inproc") and opts.get("free_mb") is None
            and not _merge_job.get("running") and not _STAGE_JOB.get("running")):
        rel = _vram_release_inproc(label)
        if rel.get("did"):
            plan["reclaim_inproc"] = dict(rel)
            free2, total2, src2 = _vram_probe(0.0)
            if free2 is not None:
                free = free2
                if total2 is not None:
                    total = total2
                src = src2
                plan["free_mb"] = int(free)
                plan["total_mb"] = (None if total is None else int(total))
                plan["source"] = src
                own = _vram_owners(free, total, src)
                _apply_own(own)
                avail = _avail_for(0.0)
                hit = _pick(avail)
    if hit is not None:
        # 降档只看**这条路真的有旋钮**的那几维：procs 模式帧预算恒为 0（requested_frames
        # 却有个默认 30），照老写法比一下永远「降了档」，status 里那句「已降档」就成了噪音。
        _down = bool(hit["batch"] < plan["requested_batch"]
                     or hit["procs"] < plan["requested_procs"])
        if mode != "procs":
            _down = _down or bool(hit["frames"] < plan["requested_frames"])
        plan.update({"ok": True, "frames": int(hit["frames"]), "batch": int(hit["batch"]),
                     "procs": int(hit["procs"]), "batch_mb": hit["batch_mb"],
                     "peak_mb": hit["need_mb"], "downgraded": _down})
        plan["frame_budget"] = int(plan["frames"])
        plan["reason"] = ("装得下：请求 (帧 %d / 批 %d / 进程 %d) → 实际 (帧 %d / 批 %d / 进程 %d)"
                          % (plan["requested_frames"], plan["requested_batch"],
                             plan["requested_procs"], plan["frames"], plan["batch"],
                             plan["procs"])
                          if plan["downgraded"] else
                          "装得下：帧 %d / 批 %d / 进程 %d（不用降档）"
                          % (plan["frames"], plan["batch"], plan["procs"]))
        if rec.get("did") and float(plan.get("reclaimed_mb") or 0.0) > 0:
            plan["reason"] += ("（回收预览模型 %.1f GB 之后才装得下）"
                               % (float(plan["reclaimed_mb"]) / 1024.0))
        elif reclaim_left > 0:
            plan["reason"] += ("（其中 %.1f GB 要靠回收预览 worker 才腾得出来%s）"
                               % (reclaim_left / 1024.0,
                                  ("；这一拍没回收成：" + str(rec.get("why") or "没有原因")[:80])
                                  if rec and not rec.get("did") else ""))
        plan["msg"] = "%s：%s · 预计峰值 %.0f MB / 可用 %.0f MB" % (
            label, plan["reason"], hit["need_mb"], avail)
        # 放行的这一份也带 summary：status 页 / 排障日志里的「本程序 / 其它程序 /
        # 空闲」和拒绝态是同一句话，用户不用在两个地方看两套口径。
        plan["summary"] = _vram_summary_line(plan)
        return plan

    # 连最小档都装不下：把「最小档」的数字落到 plan 上（拒绝消息里的拆项必须和
    # need_mb 对得上，否则用户照着算会发现少了一截）
    plan.update({"peak_mb": None, "frames": int(floor["frames"]),
                 "batch": int(floor["batch"]), "batch_mb": floor["batch_mb"],
                 "procs": int(floor["procs"])})
    plan["frame_budget"] = int(plan["frames"])
    plan["msg"] = _vram_refuse_msg(plan, label)
    plan["summary"] = _vram_summary_line(plan)
    # 注意：这段文字会进后端日志，而验收脚本靠「日志里没有 out of memory / OOM」
    # 证明「拒绝发生在加载模型之前」—— 所以这里不写 OOM 字样（按钮标签里有，够了）。
    plan["suggest"] = ["关掉占 GPU 的程序后重试（界面里有「本程序/其它程序/空闲」三栏）",
                       "点「仍然尝试（可能爆显存）」强制放行一次，真跑不动会自动回滚",
                       "换更小的 .dfm / 调低帧预算",
                       "设 %s=1 用 CPU 跑" % _VRAM_ENV_CPU]
    if _vram_cpu_fallback_env():
        plan.update({"ok": True, "cpu": True, "procs": 1,
                     "frames": int(want_frames if mode != "procs" else 0), "batch": 1,
                     "downgraded": True,
                     "reason": "显存不够但开了 %s → 回落 CPU EP" % _VRAM_ENV_CPU})
        plan["frame_budget"] = int(plan["frames"])
        plan["msg"] = ("%s：显存不够（需要 %.2f GB / 可用 %.2f GB）→ ★ 当前在 CPU 上跑 ★"
                       "（慢约 %.0f 倍；不设 %s 时这里会明确拒绝）"
                       % (label, plan["need_mb"] / 1024.0, avail / 1024.0,
                          _VRAM_CPU_SLOW, _VRAM_ENV_CPU))
        return plan
    if opts.get("force"):
        # 用户明确要试（拒绝面板上的「仍然尝试（可能 OOM）」）：按**最小档**放行。
        # 这不是把闸门拆了 —— 只是把「我替你决定」换成「你自己决定」：
        # 该记的账全都留在 plan 上（forced=True、差多少 MB），真 OOM 了由
        # _vram_oom_rollback 把现场收干净并回到拒绝态（见 routes 里的 OOM 分支）。
        plan.update({"ok": True, "cpu": False, "forced": True, "downgraded": True,
                     "peak_mb": round(float(floor["need_mb"]), 1),
                     "reason": "用户选择强制尝试：按最小档（帧 %d / 批 %d / 进程 %d）放行，"
                               "可能跑不动" % (int(floor["frames"]), int(floor["batch"]),
                                               int(floor["procs"]))})
        plan["msg"] = ("%s：强制尝试 —— 需要 %.2f GB / 可用 %.2f GB（差 %.0f MB），"
                       "按最小档放行（帧 %d / 批 %d）；真跑不动会自动回滚到拒绝态"
                       % (label, float(plan["need_mb"]) / 1024.0, avail / 1024.0,
                          max(0.0, float(plan["need_mb"]) - float(avail)),
                          int(plan["frames"]), int(plan["batch"])))
        plan["summary"] = _vram_summary_line(plan)
        print("[vram] 强制尝试 %s：需要 %.0f MB / 可用 %.0f MB（预留 %.0f MB）→ 放行最小档"
              % (label, float(plan["need_mb"]), float(avail),
                 float(plan.get("reserve_mb") or 0.0)), flush=True)
        return plan
    plan["reason"] = (("连 1 个 worker 都要 %.0f MB > 可用 %.0f MB"
                       % (floor["need_mb"], avail)) if mode == "procs" else
                      ("连最小档（帧 %d / 批 %d）都要 %.0f MB > 可用 %.0f MB"
                       % (floor["frames"], floor["batch"], floor["need_mb"], avail)))
    return plan


def _vram_refuse_extra(plan: dict, what: str = "") -> dict:
    """拒绝时的**结构化字段**（HTTP 响应体顶层）。

    前端只认这几个数就能写出「显存不足：需要 2.44 GB，可用 0.60 GB」这一行，
    不用去正则抠中文；vram 里那份完整计划留给排障和 status 回显。

    口径：
      need_mb   最小档真跑起来的预计峰值（和拒绝消息里的「需要 X GB」同一个数）
      avail_mb  预算（= 空闲 × (1-安全系数) - 预留），也就是「可用 Y GB」
      free_mb   探测到的空闲显存（伪造时是 DFL_VRAM_FREE_MB 那个值）
      suggest   建议列表（第一条就是「关掉占 GPU 的程序后重试」）
    """
    label = str(what or plan.get("label") or "")
    suggest = [str(s) for s in (plan.get("suggest") or [])
               if str(s or "").strip()]
    if not suggest:
        suggest = ["关掉占 GPU 的程序后重试"]
    def _num(key):
        v = plan.get(key)
        return None if v is None else round(float(v or 0.0), 1)
    return {
        "kind": "vram",
        "label": label,
        "need_mb": round(float(plan.get("need_mb") or 0.0), 1),
        "avail_mb": (None if plan.get("avail_mb") is None
                     else round(float(plan.get("avail_mb") or 0.0), 1)),
        "free_mb": (None if plan.get("free_mb") is None else int(plan.get("free_mb") or 0)),
        "total_mb": (None if plan.get("total_mb") is None else int(plan.get("total_mb") or 0)),
        # 显存去向（前端可以直接把「谁占的」写进提示；本程序那部分是可回收的）
        "own_mb": _num("own_mb"), "own_inproc_mb": _num("own_inproc_mb"),
        "own_preview_mb": _num("own_preview_mb"),
        "reclaimable_mb": _num("reclaimable_mb"), "reclaimed_mb": _num("reclaimed_mb"),
        "other_mb": _num("other_mb"), "preview_models": list(plan.get("preview_models") or []),
        # 预留是**显式**的一个数（不再是「20% + 512MB」那种让用户猜的口径）
        "reserve_mb": _num("reserve_mb"), "reserve_min_mb": int(plan.get("reserve_min_mb") or 0),
        "reserve_pct_mb": _num("reserve_pct_mb"),
        "safety": _num("safety"),
        # 一行话（前端原样显示：需要 / 空闲 − 预留 = 可用 / 本程序·其它程序·空闲）
        "summary": _vram_summary_line(plan),
        # 拒绝面板上那颗「仍然尝试（可能 OOM）」：前端看到它才亮按钮
        "can_force": True, "forced": bool(plan.get("forced")),
        "procs": int(plan.get("procs") or 1),
        "frames": int(plan.get("frames") or 0),
        "batch": int(plan.get("batch") or 0),
        "suggest": suggest,
        "vram": dict(plan),
    }


def _vram_summary_line(plan: dict) -> str:
    """**一行**说清这笔账（前端 toast、拒绝面板、日志都读它，不再自己拼中文）。

    形状（用户口径，这一轮定的）：
        需要 2.46 GB · 空闲 3.40 GB 里预留 0.25 GB 给驱动/碎片（max(256MB, 空闲×5%)）
        = 实际可用 3.15 GB · 本程序 0.01 GB / 其它程序 4.78 GB / 空闲 3.40 GB

    为什么非要挤在一行：老的文案把「显存去向」另起了一句，而前端 mgVramLine 只认
    need_mb/avail_mb 两个数 —— 结构化字段**根本没被显示出来**，用户看到的永远只有
    「需要 X，可用 Y」，当然不知道那 4.6GB 是谁占的、也不知道 1.37GB 是预留。
    """
    plan = dict(plan or {})
    free = plan.get("free_mb")
    avail = float(plan.get("avail_mb") or 0.0) / 1024.0
    need = float(plan.get("need_mb") or 0.0) / 1024.0
    reserve = float(plan.get("reserve_mb") or 0.0) / 1024.0
    pct = int(round(float(plan.get("safety") or 0.0) * 100))
    floor_mb = int(plan.get("reserve_min_mb") or _VRAM_RESERVE_MB)
    bits = ["需要 %.2f GB" % need]
    if free is not None:
        bits.append("空闲 %.2f GB 里预留 %.2f GB 给驱动/碎片（规则 %s）"
                    " = 实际可用 %.2f GB" % (float(free) / 1024.0, reserve,
                                             _vram_reserve_rule_text(), avail))
    else:
        bits.append("实际可用 %.2f GB（显存探不到）" % avail)
    own = plan.get("own_mb")
    other = plan.get("other_mb")
    if own is not None or other is not None:
        bits.append("本程序 %.2f GB / 其它程序 %s / 空闲 %s"
                    % (float(own or 0.0) / 1024.0,
                       ("?" if other is None else "%.2f GB" % (float(other or 0.0) / 1024.0)),
                       ("?" if free is None else "%.2f GB" % (float(free) / 1024.0))))
    # 回收预览模型的结果也进这一行：用户看 toast 就得知道「那 2.7GB 是我们自己的」，
    # 以及**为什么没腾出来**（只说 0 = 什么也没说）。
    _rec = dict(plan.get("reclaim") or {})
    _got = float(plan.get("reclaimed_mb") or 0.0)
    _why2 = str((_rec or {}).get("why") or "").strip()
    if _rec and not _rec.get("did") and float(plan.get("reclaimable_mb") or 0.0) <= 0.0:
        _rec = {}                       # 没有常驻模型可收 = 正常，不进 summary
    if _got > 0 and not _why2:
        bits.append("（已回收预览模型 %.2f GB）" % (_got / 1024.0))
    elif _rec and not _rec.get("did"):
        bits.append("（预览模型 %.2f GB 回收失败：%s）"
                    % (float(plan.get("reclaimable_mb") or 0.0) / 1024.0,
                       (_why2 or "没有原因")[:80]))
    elif _rec and _why2:
        bits.append("（预览模型只腾出 %.2f GB：%s）" % (_got / 1024.0, _why2[:80]))
    if plan.get("forced"):
        bits.append("（强制尝试）")
    return " · ".join(bits)


def _vram_refuse_msg(plan: dict, what: str = "这个任务") -> str:
    """拒绝消息：**必须带数字**（需要多少 / 空闲多少 / 预留多少 / 可用多少 / 怎么改）。

    第一句就是 _vram_summary_line 那一行（前端原样显示），后面才是拆项与建议 ——
    用户口径：「把预留写出来，别让我猜」。
    """
    what = str(what or plan.get("label") or "这个任务")
    need = float(plan.get("need_mb") or 0.0) / 1024.0
    avail = float(plan.get("avail_mb") or 0.0) / 1024.0
    free = plan.get("free_mb")
    nproc = int(plan.get("requested_procs") or 1)
    parts = ["模型 %.2f GB × %d 份" % (float(plan.get("model_mb") or 0.0) / 1024.0,
                                      max(1, min(nproc, int(plan.get("procs") or 1)))),
             "CUDA context %.2f GB × %d"
             % (int(plan.get("ctx_mb") or _VRAM_CTX_MB) / 1024.0,
                max(1, min(nproc, int(plan.get("procs") or 1)))),
             "检测/遮罩 %.2f GB" % (float(plan.get("aux_mb") or 0.0) / 1024.0)]
    if float(plan.get("sess_extra_mb") or 0.0) > 0:
        parts.append("运行时工作集 %.2f GB × %d（%s）"
                     % (float(plan.get("sess_extra_mb") or 0.0) / 1024.0,
                        max(1, min(nproc, int(plan.get("procs") or 1))),
                        plan.get("sess_src") or "标定值"))
    if int(plan.get("frames") or 0):
        parts.append("帧缓冲 %d×%.2f MB"
                     % (int(plan.get("frames") or 0),
                        float(plan.get("per_frame_mb") or 0.0)))
    parts.append("dfm 批（最小档）%.2f GB" % (float(plan.get("batch_mb") or 0.0) / 1024.0))
    # ---- 显存去向的补注（归属那一行在 _vram_summary_line 里，前端/toast 都显示它）----
    rec_left = float(plan.get("reclaimable_mb") or 0.0) / 1024.0
    got_back = float(plan.get("reclaimed_mb") or 0.0) / 1024.0
    src = ""
    if got_back > 0:
        src = "（本轮已经回收预览模型 %.2f GB）" % got_back
    elif rec_left > 0:
        src = "（预览 worker 忙，这 %.2f GB 还没回收掉 —— 导出前会再收一次）" % rec_left
    return ("%s：显存不够 —— %s。"
            "拆项：%s。%s"
            "建议：① 关掉占 GPU 的程序（浏览器硬件加速/游戏/另一个导出）后重试；"
            "② 点「仍然尝试（可能爆显存）」强制放行一次（真跑不动会自动回滚，不会把后端搞坏）；"
            "③ 换更小的 .dfm，或把「帧预算」调低（请求的是 %d 帧）；"
            "④ 设 %s=1 改用 CPU 跑（慢约 %.0f 倍，但能出片）。"
            % (what, _vram_summary_line(plan), " + ".join(parts), src,
               int(plan.get("requested_frames") or 0),
               _VRAM_ENV_CPU, _VRAM_CPU_SLOW))


def _vram_log_lines(plan: dict, tag: str = "") -> list:
    """把一份计划摊成几行日志（降档逐档记账，拒绝能看出差多少）。"""
    tag = str(tag or plan.get("label") or "显存")
    src = str(plan.get("source") or "?")
    if plan.get("free_mb") is None:
        return ["%s：显存探不到（%s）→ %s" % (tag, src, plan.get("reason") or "不降档")]
    lines = ["%s：模型 参数量 %s · 权重 %.0f MB（%s）· 输入 %s px（%s）· 非权重 %.1f MB"
             % (tag, ("%.1fM" % float(plan.get("params_m")))
                if plan.get("params_m") else "?",
                float(plan.get("model_mb") or 0), str(plan.get("model_mb_src") or "?"),
                plan.get("size_px") or "?", str(plan.get("size_px_src") or "?"),
                float(plan.get("non_weight_mb") or 0.0)),
             "%s：显存 空闲 %s / 共 %s MB（%s）· 预算 %.0f MB = 空闲 − 预留 %.0f MB"
             "（max(%d MB, 空闲×%d%%)）+ 可回收 %.0f MB"
             % (tag, plan.get("free_mb"), plan.get("total_mb") or "?", src,
                float(plan.get("avail_mb") or 0.0),
                float(plan.get("reserve_mb") or 0.0),
                int(plan.get("reserve_min_mb") or _VRAM_RESERVE_MB),
                int(round(float(plan.get("safety") or 0.0) * 100)),
                float(plan.get("reclaim_left_mb") or 0.0)),
             # 显存去向：谁占的（本程序/其它程序）。用户那句「我什么都没跑」就是靠这一行回答
             "%s：显存去向 本程序 %.0f MB（本进程 %.0f + 预览 worker %.0f）· 其它程序 %s MB"
             " · 可回收 %.0f MB%s"
             % (tag, float(plan.get("own_mb") or 0.0),
                float(plan.get("own_inproc_mb") or 0.0),
                float(plan.get("own_preview_mb") or 0.0),
                ("?" if plan.get("other_mb") is None else int(plan.get("other_mb"))),
                float(plan.get("reclaimable_mb") or 0.0),
                ("" if not plan.get("reclaimed_mb")
                 else "（已回收 %.0f MB / %.0fms）"
                      % (float(plan.get("reclaimed_mb") or 0.0),
                         float(plan.get("reclaim_ms") or 0.0)))),
             "%s：需求拆项 模型 %.0f + context %d%s%s + 检测/遮罩 %.0f MB/份"
             " + 帧 %d×%.2f + dfm 批 %s MB（ORT arena %d MB 含在预留里）"
             % (tag, float(plan.get("model_mb") or 0), int(plan.get("ctx_mb") or 0),
                ("（本进程已持有，不计入）" if plan.get("resident_mb") else ""),
                ((" + 运行时工作集 %.0f（%s）" % (float(plan.get("sess_extra_mb") or 0),
                                                 plan.get("sess_src") or ""))
                 if float(plan.get("sess_extra_mb") or 0) > 0 else ""),
                float(plan.get("aux_mb") or 0),
                int(plan.get("frames") or 0), float(plan.get("frame_mb") or 0.0),
                plan.get("batch_mb") if plan.get("batch_mb") is not None else "?",
                int(plan.get("arena_mb") or 0))]
    # ---- 回收结果单独一行：成功要能看见「卸了多少 / 空闲涨到哪 / 花了多久」，
    #      失败**必须写出原因**（用户口径：不能只打一个 0）----
    _rec = dict(plan.get("reclaim") or {})
    # 「worker 里本来就没有常驻模型」不是故障：那一栏空着，别写成「没回收掉」吓人。
    if _rec and not _rec.get("did") and float(plan.get("reclaimable_mb") or 0.0) <= 0.0 \
            and not _rec.get("freed_mb"):
        _rec = {}
    if _rec:
        _freed = float(_rec.get("freed_mb") or 0.0)
        _why = str(_rec.get("why") or "").strip()
        _rb = _rec.get("before_mb")
        _ra = _rec.get("after_mb")
        # 形状只有一个：「回收预览模型 <freed> MB（<ms>ms · …）：空闲 A → B MB · 附注」
        # —— 人眼和验收脚本的正则都只认这一种，别按分支换写法。
        if _rec.get("did"):
            _tail = (" · " + _why) if _why else (
                (" · " + str(_rec.get("note") or "")) if _rec.get("note") else "")
            lines.insert(2, "%s：回收预览模型 %.0f MB（%.0fms%s）：空闲 %s → %s MB%s"
                         % (tag, _freed, float(_rec.get("ms") or 0.0),
                            " · 进程已回收" if _rec.get("recycled") else "",
                            "?" if _rb is None else int(_rb),
                            "?" if _ra is None else int(_ra), _tail))
        else:
            lines.insert(2, "%s：预览 worker 的模型没回收掉 —— %s"
                         "（可回收 %.0f MB 已按「计入可用」算，导出起手时会再收一次）"
                         % (tag, _why or "没有原因（worker 没回 why）",
                            float(plan.get("reclaim_left_mb")
                                  if plan.get("reclaim_left_mb") is not None
                                  else plan.get("reclaimable_mb") or 0.0)))
    if plan.get("mode") == "procs":
        lines.append("%s：计划 %d 个 worker × %d 线程（请求 %d × %d）· 预计峰值 %.0f MB"
                     % (tag, int(plan.get("procs") or 1), int(plan.get("batch") or 1),
                        int(plan.get("requested_procs") or 1),
                        int(plan.get("requested_batch") or 1),
                        float(plan.get("peak_mb") or plan.get("need_mb") or 0)))
    else:
        lines.append("%s：计划 帧预算 %d / dfm 批 %d%s · 预计峰值 %s（可用 %.0f MB）%s"
                     % (tag, int(plan.get("frames") or 0), int(plan.get("batch") or 0),
                        "（已降档）" if plan.get("downgraded") else "",
                        (("%.0f MB" % plan["peak_mb"]) if plan.get("peak_mb") is not None
                         else "装不下（没有一档塞得进预算）"),
                        float(plan.get("avail_mb") or 0.0),
                        ("（会话已在本进程里，%.0f MB 不计入需求）"
                         % float(plan.get("resident_mb") or 0.0))
                        if plan.get("resident_mb") else ""))
    # 逐档记账：拿到「采用」那一档就停（后面的档位这次用不上，刷日志只会淹掉重点）；
    # 全都装不下时把 12 档全列出来 —— 拒绝的理由要看得出「差多少」。
    _stop = False
    for st in (plan.get("steps") or []):
        if _stop:
            break
        lines.append("%s：%s 帧 %s / 批 %s / 进程 %s → 要 %.0f MB %s %.0f MB"
                     % (tag, "采用" if st.get("fit") else "否决",
                        st.get("frames"), st.get("batch"), st.get("procs"),
                        float(st.get("need_mb") or 0),
                        "≤" if st.get("fit") else ">", float(st.get("avail_mb") or 0)))
        _stop = bool(st.get("fit"))
    if plan.get("forced"):
        lines.append("%s：⚠ 强制尝试（用户点了「仍然尝试」）—— 需要 %.0f MB / 可用 %.0f MB，"
                     "差 %.0f MB；真跑不动会自动回滚到拒绝态"
                     % (tag, float(plan.get("need_mb") or 0.0),
                        float(plan.get("avail_mb") or 0.0),
                        max(0.0, float(plan.get("need_mb") or 0.0)
                            - float(plan.get("avail_mb") or 0.0))))
    if plan.get("cpu"):
        lines.append("%s：★ 当前在 CPU 上跑 ★（%s；不设 %s 时这里会明确拒绝）"
                     % (tag, plan.get("reason") or "", _VRAM_ENV_CPU))
    return lines


def _vram_apply(plan: dict, label: str = "", sticky: bool = False) -> dict:
    """落实一份计划：切 CPU/GPU 开关、记进 _VRAM_LAST（status 端点读它）。

    sticky=True 表示「只许开、不许关」：预览每拖一帧都会算一次计划，它没有权力替一个
    正在跑的导出把 CPU 开关关掉（那会让半个任务在 CPU、半个在 GPU）。关这个开关的
    只有「新任务起手时自己算的那一份计划」。
    """
    label = str(label or plan.get("label") or "vram")
    plan["label"] = label
    plan["at"] = round(time.time(), 3)
    if plan.get("cpu") or not sticky:
        _vram_set_cpu(bool(plan.get("cpu")), plan.get("reason") or "")
    with _VRAM_LOCK:
        _VRAM_LAST[label] = dict(plan)
        sig = (label, bool(plan.get("ok")), plan.get("frames"), plan.get("batch"),
               plan.get("procs"), bool(plan.get("cpu")),
               None if plan.get("avail_mb") is None else int(float(plan["avail_mb"]) // 256))
        fresh = _VRAM_LOGGED.get(label) != sig
        if fresh:
            _VRAM_LOGGED[label] = sig
    # 服务端日志留一份账（预览这种每秒都可能问一次的，只在「决定变了」时打）
    if fresh:
        for _ln in _vram_log_lines(plan, label):
            print("[vram] " + _ln, flush=True)
    return plan


def _vram_last(label: str = None) -> dict:
    with _VRAM_LOCK:
        if label:
            return dict(_VRAM_LAST.get(str(label)) or {})
        return {k: dict(v) for k, v in _VRAM_LAST.items()}


def _preview_guard(dfms, label: str = "预览") -> dict:
    """预览的显存闸门（sticky：只许开 CPU，不许关别人的）。

    特例 —— **模型已经常驻在预览 worker 里**时不算增量：那一帧只跑一次前向，不新建
    session，显存本来就已经占着了（这是唯一一种「空闲 0MB 但预览照样能出图」的情形）。
    不认这个特例会误杀：stage6 验收里 4.0 对拍把卡占满的那几分钟，预览是被拒的，
    而它连一个字节都不用再分配。**只有在要新加载模型时才真的需要预算**。
    """
    paths = [str(p) for p in (dfms or []) if str(p or "")]
    try:
        live = _WORKER.alive()
        loaded = {os.path.normcase(str(k)) for k in (_WORKER._models or {})} if live else set()
    except Exception:                                       # noqa: BLE001
        live, loaded = False, set()
    if live and paths and all(os.path.normcase(p) in loaded for p in paths):
        plan = _vram_plan(_vram_models_mb(paths),
                          {"mode": "none", "procs": 1, "label": label})
        plan.update({"ok": True, "cpu": False, "resident": True,
                     "reason": "模型已在预览 worker 里常驻（这一帧不再新占显存）",
                     "msg": "%s：模型已常驻，不再额外占显存（空闲 %s MB / 增量 0 MB）"
                            % (label, plan.get("free_mb"))})
        return _vram_apply(plan, label, sticky=True)
    # 预览从不回收**自己**（worker 里的模型正是它要用的），但可以放掉父进程里那些
    # 「导出留下的」dfm/XSeg 会话 —— 没有任务在跑时那是纯占用（见 _vram_release_inproc）。
    return _vram_guard(paths, {"mode": "none", "procs": 1, "reclaim_inproc": True},
                       label=label, sticky=True, reclaim=False)


def _vram_guard(dfms, opts: dict = None, label: str = "任务",
                sticky: bool = False, reclaim: bool = None, force: bool = None) -> dict:
    """加载模型**之前**的一次预算检查：装得下返回计划，装不下抛 400（消息带数字）。

    sticky=True（预览用）：只在「要在 CPU 上跑」时短路全链路，不允许把别人的 CPU 开关关掉。

    reclaim：算账之前先回收**我们自己**占的显存（常驻预览 worker 里那份 dfm 会话）。
    默认 = 非 sticky，也就是所有导出/阶段闸门都回收，预览自己不回收（它正是要靠那份
    常驻模型才快，而且 _preview_guard 有「已常驻 = 增量 0」的特例）。

    force：用户点了拒绝面板上的「仍然尝试（可能 OOM）」—— 装不下也放行最小档
    （plan.forced=True）。**只放行、不改账**：need/avail/差多少 MB 全留在 plan 上，
    真 OOM 时 _vram_oom_rollback 收现场并回到拒绝态。
    """
    if isinstance(dfms, (str, Path)):
        paths = [str(dfms)]
    else:
        paths = [str(p) for p in (dfms or []) if str(p or "")]
    opts = dict(opts or {})
    if not opts.get("size_px"):
        opts["size_px"] = _dfm_size_hint(paths[0] if paths else "")
    if reclaim is None:
        reclaim = (not sticky)
    if force is None:
        force = bool(opts.get("force"))
    plan = _vram_plan(_vram_models_mb(paths),
                      dict(opts, label=label, reclaim=bool(reclaim), force=bool(force)))
    _vram_apply(plan, label, sticky=bool(sticky))
    if not plan.get("ok"):
        msg = plan.get("msg") or _vram_refuse_msg(plan, label)
        # 拒绝必须有日志（排障第一现场）：这一行进后端 stdout（→ server.log）
        print("[vram] ★ 拒绝 %s ★ %s" % (label, msg), flush=True)
        # 400 带上 need_mb / avail_mb / free_mb / suggest[]（前端只弹一条提示、
        # 不再去正则抠中文），并且**在起 job / 加载模型之前**抛出去。
        raise _BadInput(msg, 400, extra=_vram_refuse_extra(plan, label))
    return plan

def _vram_forced_rollback(label: str, model_mb: float, opts: dict, err) -> dict:
    """强制尝试失败之后：该不该按「确实不够 / 已回滚」收口？要的话返回回滚现场。

    判据两条，满足其一就算**真的不够**：
      ① 报错文本是显存类（ORT / torch / TRT 的说法）；
      ② 把现场收干净、重探一次之后，按当下的空闲**仍然装不下**。

    ② 是必要的兜底：ORT 建 session 失败时给的异常五花八门（有的只有 "CUDA failure"），
    只看文本会漏判，用户就只看到一句「导出失败」，而真相是「这块卡确实不够」。
    反过来说，如果重探之后其实装得下（内存被别的程序还回来了 / 是我们自己的 bug），
    这里**不**改写成「显存不够」—— 那种错该老实报错。
    """
    rb = _vram_oom_rollback(label, model_mb, opts, err)
    if _vram_oom_is_text(err):
        return rb
    if not bool((rb.get("plan") or {}).get("ok")):
        return rb
    return {}


def _vram_oom_inject_on() -> bool:
    """验收专用：DFL_ALLOW_TEST_HOOK=1 + DFL_VRAM_OOM_INJECT=1 时，_dfm_session 立刻抛 OOM。

    为什么需要这个钩子：WDDM 下「显存不够」通常**不会** OOM —— 驱动会把别人的分配换页
    出去，本机实测把空闲压到 600MB 再 force=1 竟然还能出片（慢，但不失败）。要让
    「强制尝试 → OOM → 回滚 → 还能再试」这条路可复现，只能注入一次 OOM。
    双开关是照 _vram_free_mb.txt 那条规矩来的：生产环境不读任何测试钩子。
    """
    if str(os.environ.get("DFL_ALLOW_TEST_HOOK") or "").strip() not in ("1", "true", "yes", "on"):
        return False
    return str(os.environ.get("DFL_VRAM_OOM_INJECT") or "").strip() in ("1", "true", "yes", "on")


def _vram_reserve_rule_text() -> str:
    """预留这条规则的一句话（文案/日志共用，避免口径和数字对不上）。"""
    if str(os.environ.get("DFL_VRAM_RESERVE_MODE") or "").strip().lower() in ("legacy",
                                                                             "legacy20", "old"):
        return "老口径 空闲×20% + 512MB"
    return "max(256MB, 空闲×5%)"


def _vram_oom_is_text(text) -> bool:
    """这段文字/异常是不是「真的没显存了」（ORT / torch / TRT 的说法都认）。"""
    s = str(text or "").lower()
    return any(k in s for k in ("out of memory", "cuda_error_out_of_memory",
                               "cudaerrormemoryallocation", "failed to allocate",
                               "insufficient memory", "resource_exhausted",
                               "cudnn_status_alloc_failed"))


def _vram_oom_rollback(label: str = "任务", model_mb: float = 0.0, opts: dict = None,
                       err="") -> dict:
    """强制尝试撞上 OOM 之后：**干净地**回到拒绝态（用户那条「别把后端搞坏」）。

    四件事，顺序不能反：
      ① 放掉本进程的 dfm / XSeg 会话 + gc + empty_cache —— OOM 之后 ORT 的 arena
         可能还挂在那个建到一半的 session 上，不清掉下一次连预览都抢不到显存；
      ② 重新探一次空闲（fresh，不吃 2~4s 缓存），按**当下的数**重算一份拒绝计划
         （不是拿 OOM 之前那份旧账忽悠用户）；
      ③ 一句话说清「确实不够 / 已回滚 / 差多少 / 可以再试」；
      ④ 别人的东西一概不动：预览 worker 不重载、后台进程不杀 —— 所以下次还能再试。

    返回的 dict 会挂到 /export/status 与 /stage/status 的 vram_oom 键上（前端据此
    把界面从「进度页」搬回「被拒态」并把 current 的数字重画一遍）。
    """
    rel = {}
    try:
        rel = _vram_release_inproc("强制尝试 OOM 回滚")
    except Exception as e:                                  # noqa: BLE001
        print("[vram] OOM 回滚：放掉本进程会话失败：%s" % str(e)[:160], flush=True)
    try:
        import gc
        gc.collect()
        gc.collect()
        torch = sys.modules.get("torch")
        if torch is not None and torch.cuda.is_initialized():
            torch.cuda.empty_cache()
    except Exception:                                       # noqa: BLE001
        pass
    o2 = dict(opts or {})
    for k in ("force", "free_mb", "total_mb", "reclaim", "reclaim_inproc", "resident"):
        o2.pop(k, None)          # 回滚这一拍：不许再强制，也不许拿一份旧 free 充数
    o2.update({"label": label, "reclaim": False, "force": False})
    try:
        plan = _vram_plan(float(model_mb or 0.0), o2)
    except Exception as e:                                  # noqa: BLE001
        plan = {"ok": False, "need_mb": None, "avail_mb": None,
                "reason": "重算预算失败：%s" % str(e)[:160], "msg": ""}
    try:
        free = plan.get("free_mb")
        summary = "强制尝试失败：%s" % _vram_summary_line(plan)
        gap = max(0.0, float(plan.get("need_mb") or 0.0) - float(plan.get("avail_mb") or 0.0))
        msg = ("%s：强制尝试失败了 —— 确实不够（运行中 OOM：%s）。"
               "已回滚：放掉本进程会话 %s，空闲现在 %s MB（%s）。差 %.0f MB，"
               "关掉占 GPU 的程序后可以再点一次。"
               % (label, str(err or "").strip()[:200] or "显存分配失败",
                  "、".join(list((rel or {}).get("released") or [])[:3]) or "无（本来就没有）",
                  ("?" if free is None else int(free)), summary, gap))
    except Exception:                                       # noqa: BLE001
        summary, msg = "强制尝试失败：显存确实不够，已回滚", "强制尝试失败：显存确实不够，已回滚"
    print("[vram] ★ %s ★" % msg, flush=True)
    return {"ok": False, "kind": "vram_oom", "label": label,
            "released": list((rel or {}).get("released") or []),
            "need_mb": plan.get("need_mb"), "avail_mb": plan.get("avail_mb"),
            "free_mb": plan.get("free_mb"), "total_mb": plan.get("total_mb"),
            "reserve_mb": plan.get("reserve_mb"), "own_mb": plan.get("own_mb"),
            "other_mb": plan.get("other_mb"), "can_force": False,
            "summary": summary, "msg": msg, "error": msg,
            "suggest": ["关掉占 GPU 的程序后重试", "换更小的 .dfm / 调低帧预算",
                        "设 %s=1 用 CPU 跑" % _VRAM_ENV_CPU],
            "plan": dict(plan)}

_PRED_VRAM_SAFETY = 0.20      # 留 20% 余量：探测量打八折才是敢用的预算
_PRED_BATCH_CAP = 32          # 上限（再大对单卡没意义，batch 16 起反而开始抖）


def _pred_pick_batch(cfg: dict, ent) -> tuple:
    """挑 batch：请求里写了就照用，没写就**把阶梯真跑一遍再决定**（不写死、也不猜）。

    为什么不能写死 8，也不能只按显存估算：
      · 8GB 卡上 8 是个好值，但同一张卡只剩 1.2GB 空闲时 8 就是一路 OOM 退到 1；
      · 反过来，实测这台机上 batch 16/32 的每张耗时是 batch 1 的 5 倍以上
        （ORT 的 arena 撑爆后整个进程都变慢，之后不再恢复）—— 光看「可用显存够不够」
        会一路选到 32，然后整条链路崩掉。所以**耗时也要量**。

    阶梯规则（全部实测，约 1s 的预热，同时也是真正的 warm-up）：
      1. 依次跑 batch = 1 / 4 / 8 的空张量，记下「每张耗时」和「可用显存」；
      2. 选「每张耗时 ≤ 最快值 × 1.05」里最大的那个 batch（本机：1/4/8 → 选 8）；
      3. 只有在「8 比 1 快 2 倍以上（说明是小模型 + 大卡，批量真的有用）+ 还剩 ≥6GB
         + 下一档塞得进 80% 预算」三条同时成立时，才继续试 16 / 32。
         本机上 8 只比 1 快 1.3×，而 16 会把 8GB 顶到 7806/8151 MB 且并不更快，
         还会让 ORT 的 arena 撑爆、之后整个进程掉到 15 张/秒 —— 所以老实停在 8；
      4. 每一步都按实测单张开销核对 80% 显存预算，不够就停。
    运行中还有两道保险：OOM → 8→4→2→1 退档；比历史最好慢 60% → 主动退半（见 _flush）。

    返回 (batch, {"source","ladder","free_mb","total_mb","per_item_mb","best_per_face_ms"})
    """
    want = int(cfg.get("pred_batch") or 0)
    if want > 0:
        return max(1, min(64, want)), {"source": "request", "batch": int(want),
                                       "ladder": [], "free_mb": None, "total_mb": None,
                                       "per_item_mb": None, "best_per_face_ms": None}
    if _vram_cpu_on():
        # CPU 上跑还探阶梯没意义（每档都慢 17 倍，而且没有显存可省）：直接 1 张。
        return 1, {"source": "cpu", "batch": 1, "ladder": [], "free_mb": None,
                   "total_mb": None, "per_item_mb": None, "best_per_face_ms": None}
    import numpy as np
    _sess, _name, size, _nchw = ent[0], ent[1], ent[2], ent[3]
    _cap = int(ent[5] or 1) if len(ent) > 5 else 8
    size = int(size)
    free0, total0 = _vram_free_mb()
    # 阶梯不能超过 session 真正能吃的批（原生 TRT 引擎的优化档位可能就是 batch 1）
    ladder = [1]
    for _b in (4, 8):
        if _b <= _cap and _b not in ladder:
            ladder.append(_b)
    rows = []
    best_per = None
    try:
        i = 0
        while i < len(ladder):
            b = min(int(ladder[i]), _PRED_BATCH_CAP, _cap)
            x = np.zeros((b, size, size, 3), np.float32)
            # 每个档位先跑一次不计时：第一次跑某个形状要编 kernel / 选算法，实测能把
            # batch 1 的量测放大到 1000 ms（真值是 36 ms）—— 以前就是拿这个被污染的
            # 数字去选 batch，结果选了更慢的那一档。取两次里快的那次才是稳态。
            _dfm_run_batch(ent, x)
            t0 = time.perf_counter()
            _dfm_run_batch(ent, x)
            dt = (time.perf_counter() - t0) * 1000.0
            t0 = time.perf_counter()
            _dfm_run_batch(ent, x)
            dt = min(dt, (time.perf_counter() - t0) * 1000.0)
            per = dt / b
            free_now, _t = _vram_free_mb()
            rows.append({"batch": b, "ms": round(dt, 1), "per_face_ms": round(per, 2),
                         "free_mb": free_now})
            best_per = per if best_per is None else min(best_per, per)
            # 显存：按实测单张开销核对 80% 预算
            fits = None
            if free_now is not None and total0:
                per_item = max(size * size * 3 * 4 * 6 / 1048576.0,
                               max(0.0, (free0 - free_now)) / float(max(1, b)))
                budget = free_now * (1.0 - _PRED_VRAM_SAFETY)
                fits = int(budget / per_item) if per_item > 0 else b
                rows[-1]["per_item_mb"] = round(per_item, 1)
                rows[-1]["budget_batch"] = fits
                if fits <= b or free_now < 1024:
                    break
            # 大卡才继续往上探（两个条件都要满足）：
            #   ① 8 张/批确实比单张快（GPU 跑满的机器不会踏进 16/32 那个坑）；
            #   ② 下一档还塞得进 80% 显存预算 —— 否则试一次就把显存顶到墙，
            #      整条流水线都会被拖慢（实测这台机 batch 16 每张 40ms → 峰值 7.3GB）。
            if i == len(ladder) - 1 and b >= 8 and best_per is not None:
                nxt = 16
                first = float(rows[0]["per_face_ms"] or 0.0)
                gain = (first / float(rows[-1]["per_face_ms"] or 1.0)) if first else 1.0
                # 越过 8 要三个条件同时成立：批量确实带来 2× 以上的单张收益（说明是
                # 小模型 + 大卡，批量有用）、显存至少还剩 6GB、下一档塞得进 80% 预算。
                # 本机（416 dfm、8GB、批量收益 1.3×）三个都不满足 → 老实停在 8：
                # 实测 batch 16 会把显存顶到墙，之后整个进程每张 40ms → 60ms，越试越慢。
                if gain >= 2.0 and (free_now is None or free_now >= 6144)                         and (fits is None or fits >= nxt):
                    ladder.extend([16, 32])
                else:
                    break
            i += 1
    except Exception as e:                                  # noqa: BLE001
        _stage_log("换脸：batch 探测中断（%s）→ 用已测到的档位" % str(e)[:100])
        if not rows:
            return 4, {"source": "probe-failed", "batch": 4, "ladder": [],
                       "free_mb": free0, "total_mb": total0,
                       "per_item_mb": None, "best_per_face_ms": None}
    if not rows:
        return 4, {"source": "no-vram-info", "batch": 4, "ladder": [],
                   "free_mb": free0, "total_mb": total0,
                   "per_item_mb": None, "best_per_face_ms": None}
    ok = [r for r in rows if r["per_face_ms"] <= best_per * 1.05]
    pick = max(ok or rows[:1], key=lambda r: r["batch"])
    meta = {"source": "auto", "batch": int(pick["batch"]), "ladder": rows,
            "free_mb": free0, "total_mb": total0,
            "per_item_mb": pick.get("per_item_mb"),
            "best_per_face_ms": round(min(r["per_face_ms"] for r in rows), 2),
            "safety": _PRED_VRAM_SAFETY}
    _stage_log("换脸：batch 阶梯 %s → 选 %d"
               % (" / ".join("B%d=%.1fms(%.2fms/张)" % (r["batch"], r["ms"], r["per_face_ms"])
                             for r in rows), int(pick["batch"])))
    return int(pick["batch"]), meta


def _is_oom(e) -> bool:
    s = str(e).lower()
    return ("allocate" in s or "out of memory" in s or "memory" in s and "insufficient" in s)


def _stage_do_pred(cfg: dict) -> dict:
    """阶段 3 · 换脸：frames/ + aligned/ -> pred/（换脸正脸 + dfm 自带两张遮罩）。

    语义照 4.0 export_pipeline.py:769（fo = warpAffine(frame, get_transform_mat(lm,
    swap_sz, WHOLE_FACE)) → dfm 前向）。

    **吞吐**（本机实测，1080×1920、每帧 1 张脸、app/tools/bench/stage_perf_profile.py）：
        read 28.5ms/张（4 线程摊薄后，整帧 PNG 解码 17.7ms 是底）· prep 9ms/张 ·
        infer 37.6ms(ort-CUDA) / 30.4ms(原生 TRT bf16) ← 硬底 · write 26ms/张（2 线程）
    墙钟 ≈ max(三段)：25.6 张/秒(ort-CUDA) / 31.8 张/秒(原生 TRT)。
    所以这一段**已经贴着推理跑了**，再优化 CPU 侧不会让吞吐变高，只会让它更抗噪。
    老实现是一条串行链（读帧 → warp → 攒 batch → 前向 → 写 3 张 PNG），
    一张脸 = 22(读) + 8(float 转换) + 36(前向) + 21(三张 PNG 编码) ≈ 100ms，实测 9.6 张/秒，
    而且 batch 从 1 加到 16 吞吐几乎不动 —— 说明瓶颈根本不在推理。
    现在拆成三段，用队列连起来（谁慢谁是瓶颈，其它两段被完全藏住）：
        读线程 ×N    PNG 解码 + 整帧 → float32 + 按 68 点 warp（1080p ≈ 30ms/帧，
                      4 线程实测摊到 7.3ms/帧）
        推理线程     攒够 batch 一次前向（416 dfm ≈ 36ms/张，GPU 99% 占用，是硬底）
        写线程 ×M    PNG 编码 + 原子落盘（416×416 ×3 ≈ 21ms/张）
    墙钟 ≈ max(三段)，而不是三段之和。

    已经踩过的两个坑（别再踩）：
      · 读线程以前先把 1080p 整帧 astype(float32)/255 再 warp（实测多花 25ms/帧），
        而 4.0 是「uint8 整帧 warp → 再 /255」，两边只差舍入 —— 现在和 4.0 同序；
      · 进度里的「帧/秒」以前拿续跑游标 k 算，而 k 为了不留空洞会滞后 readers 倍：
        25.4 张/秒被显示成 6.3 帧/秒，用户报的「5.8~6.2 帧/秒」就是它，不是真变慢。

    其它三处通用优化（都不是针对某台机器）：
      · batch 按可用显存实测自动选（_pred_pick_batch），OOM / 明显变慢自动退档 8→4→2→1；
      · 输入缓冲按 batch 预分配，每批只做一次 memcpy，不再是 np.stack(...).astype(float32)
        的「分配 + 转换 + 再拷一遍」；
      · 人脸库归组（embedding）仍然在主线程做 —— 它要用整帧，且非默认路径。

    多模型：有人脸库时按 embedding 归组，每组用各自的 dfm；不匹配的脸**不产出 pred**
    （⑤ 保持原像素）。游标 = meta.stages.pred.done_frames（帧数），取消可续。
    """
    import queue
    import sys as _sys
    import cv2
    import numpy as np
    from ._merge_compose import ensure_engine_path
    ensure_engine_path()
    from facelib import FaceType
    from facelib.LandmarksProcessor import get_transform_mat
    from ._align import read_dfljpg_meta

    # 流水线实况：换脸 = 1 个 dfm session（一条 lane；写盘线程不产 bit，不算 lane）
    _STAGE_METER.set_lanes({"dfm": ["dfm#0"]})
    cache = Path(cfg["cache"])
    meta = cfg["meta"]
    st = meta["stages"].setdefault("pred", {})
    aligned = _stage_files(_stage_aligned_dir(cache), (".jpg", ".jpeg"))
    if not aligned:
        raise _BadInput("aligned/ 里没有对齐脸 —— 先跑切脸（POST /api/merge/stage/faces）", 409)
    if not (meta["stages"].get("faces") or {}).get("done"):
        raise _BadInput("切脸（faces）还没完成 —— 先跑切脸，再换脸", 409)
    frames_dir = _stage_frames_dir(cache)
    # 帧范围：只对 [start, end] 里的帧做推理（groups 的下标/游标都按过滤后的列表数）
    rng = _stage_range(cfg)
    rkey = str(cfg.get("range_key") or rng.get("key") or "")
    groups_all = _stage_face_rows(aligned)
    groups = [(k, ps) for k, ps in groups_all if _stage_in_range(rng, k)]
    total = len(groups)
    faces_in_range = sum(len(ps) for _k, ps in groups)
    pred_dir = _stage_pred_dir(cache)
    pred_dir.mkdir(parents=True, exist_ok=True)

    dfm = str(cfg.get("dfm") or "")
    if not dfm or not Path(dfm).is_file():
        raise _BadInput("没有 .dfm 模型（换脸阶段要它）：%s" % (dfm or "空"), 409)

    ent0 = _dfm_session(dfm)                       # 先加载：batch 要按它的显存开销算
    batch, bmeta = _pred_pick_batch(cfg, ent0)
    size0 = int(ent0[2])
    # 并行度：读线程按 CPU 核数给（PNG 解码在 OpenCV 里是放 GIL 的），写线程固定 2 个
    readers = max(1, int(cfg.get("pred_readers")
                         or min(4, max(1, (os.cpu_count() or 4) // 3))))
    writers = max(1, int(cfg.get("pred_writers") or 2))
    use_lib = bool(cfg.get("use_library") and cfg.get("library"))

    # 游标 **相对范围**：0..total（total = 范围内有脸的帧数）
    rchg = bool(cfg.get("range_changed"))
    cursor = 0 if (cfg["force"] or rchg) else max(0, min(int(st.get("done_frames") or 0), total))
    if (not cfg["force"]) and not rchg and st.get("done") and cursor >= total and total:
        _stage_log("换脸：复用已有 %d 帧的 pred/（第 %d..%d 帧，不重跑）"
                   % (total, int(rng["start"]), int(rng["end"])))
        return {"skipped": True, "count": int(st.get("count") or 0), "reason": "already-done"}
    if cursor:
        _stage_log("换脸：从范围内第 %d 帧续跑（前 %d 帧的 pred/ 已经在）" % (cursor + 1, cursor))
    _stage_log("换脸：dfm=%s · batch=%d(%s) · 帧范围 %d..%d（%d 帧 / %d 张脸）· "
               "读 %d 线程 / 写 %d 线程%s"
               % (Path(dfm).name, batch, bmeta.get("source"),
                  int(rng["start"]), int(rng["end"]), total, faces_in_range,
                  readers, writers, " · 人脸库归组" if use_lib else ""))
    if bmeta.get("source") == "auto":
        _stage_log("换脸：batch 自动选择 %d（可用显存 %s / 共 %s MB · 单张约 %s MB · "
                   "留 %d%% 余量 · 最快 %s ms/张）"
                   % (batch, bmeta.get("free_mb"), bmeta.get("total_mb"),
                      bmeta.get("per_item_mb"), int(_PRED_VRAM_SAFETY * 100),
                      bmeta.get("best_per_face_ms")))

    t0 = time.time()
    # force 重跑时从 0 数起：以前会把上一轮的 count 累加（90 张脸跑完显示 180），
    # 于是 faces_per_sec 一直偏大 —— 吞吐对比全是错的数
    done_faces = 0 if (cfg.get("force") or rchg) else int(st.get("count") or 0)
    if cfg.get("force") or rchg:
        st["count"] = 0
    unmatched = int(st.get("unmatched") or 0)
    failed = int(st.get("failed") or 0)
    sizes = {}
    prof = {"read_ms": 0.0, "prep_ms": 0.0, "infer_ms": 0.0, "write_ms": 0.0,
            "faces": 0, "frames": 0}
    plock = threading.Lock()

    def _bump(ms_name: str, sec: float, n: int = 0):
        with plock:
            prof[ms_name] += sec * 1000.0
            if n:
                prof["faces"] += n

    def _save(up_to: int, done: bool = False, extra: dict = None):
        node = {
            "done": bool(done), "done_frames": int(up_to), "frames": total,
            "total": int(total), "range": {"start": int(rng["start"]), "end": int(rng["end"])},
            "range_key": rkey, "range_total": int(total),
            "count": done_faces, "unmatched": unmatched, "failed": failed,
            "faces_total": int(faces_in_range), "dfm": str(dfm), "model": Path(dfm).name,
            "batch": int(batch), "batch_source": str(bmeta.get("source") or ""),
            "batch_meta": dict(bmeta), "sizes": dict(sizes), "pred_dir": str(pred_dir),
            "readers": int(readers), "writers": int(writers),
            "resumed_from": int(cursor),
            "msg": ("换脸完成：%d 张脸 / %d 帧（第 %d..%d 帧 · 未匹配 %d）"
                    % (done_faces, total, int(rng["start"]), int(rng["end"]), unmatched))
                   if done else ("换脸 %d/%d 帧（第 %d..%d 帧）"
                                 % (up_to, total, int(rng["start"]), int(rng["end"]))),
        }
        if extra:
            node.update(extra)
        meta["stages"]["pred"] = node
        _stage_meta_save(cache, meta)

    # ---- 队列与共享状态 ----
    q_frames = queue.Queue(maxsize=max(2, readers * 2))
    q_write = queue.Queue(maxsize=max(4, batch * 2))
    stop = threading.Event()
    boss = {"exc": None}
    pre = np.empty((batch, size0, size0, 3), np.float32)      # 预分配输入缓冲（NHWC）
    pend, completed, plock2 = {}, set(), threading.Lock()
    commit = {"n": cursor}

    def _commit_upto():
        with plock2:
            n = commit["n"]
            while n in completed:
                completed.discard(n)
                n += 1
            commit["n"] = n
            return n

    def _pend_add(gi: int):
        with plock2:
            pend[gi] = int(pend.get(gi) or 0) + 1

    def _pend_done(gi: int):
        nonlocal failed
        with plock2:
            left = int(pend.get(gi) or 0) - 1
            if left <= 0:
                pend.pop(gi, None)
                completed.add(gi)
            else:
                pend[gi] = left

    # ---- 写线程：PNG 编码 + 原子落盘 ----
    def _writer():
        nonlocal failed
        while True:
            item = q_write.get()
            if item is None:
                return
            gi, key, face, prd, dst = item
            _stage_tick("dfm")       # 交接点：dfm 换脸 → pred/*.png
            tw = time.perf_counter()
            try:
                _stage_write_atomic(
                    pred_dir / (key + ".png"),
                    _stage_imencode_png(np.clip(face * 255.0, 0, 255).astype(np.uint8), 1))
                if prd is not None:
                    _stage_write_atomic(
                        pred_dir / (key + "_prdm.png"),
                        _stage_imencode_png(np.clip(prd * 255.0, 0, 255).astype(np.uint8), 1))
                if dst is not None:
                    _stage_write_atomic(
                        pred_dir / (key + "_dstdm.png"),
                        _stage_imencode_png(np.clip(dst * 255.0, 0, 255).astype(np.uint8), 1))
            except Exception as e:                          # noqa: BLE001
                failed += 1
                _stage_log("换脸写盘失败 %s：%s" % (key, str(e)[:140]))
            finally:
                _bump("write_ms", time.perf_counter() - tw)
                _pend_done(gi)

    def _put_frame(item):
        """带超时的 put：主线程没了（取消/异常）也别把读线程吊死在队列上。"""
        while not stop.is_set():
            try:
                q_frames.put(item, timeout=0.2)
                return
            except queue.Full:
                continue

    # ---- 读线程：解码 + warp（不碰推理，也不碰写盘）----
    def _reader(lo: int, hi: int):
        try:
            for gi in range(lo, hi):
                if stop.is_set():
                    break
                key, paths = groups[gi]
                fp = _stage_frame_path(frames_dir, key)
                if fp is None:
                    _put_frame((gi, key, False, None, None, None))
                    continue
                tr = time.perf_counter()
                frame = _stage_read_frame(fp)
                _bump("read_ms", time.perf_counter() - tr)
                if frame is None:
                    _put_frame((gi, key, False, None, None, None))
                    continue
                crops = None
                if not use_lib:
                    tp = time.perf_counter()
                    crops = []
                    for ap in paths:
                        m = read_dfljpg_meta(ap) or {}
                        lm = m.get("source_landmarks") or m.get("landmarks")
                        if not lm or len(lm) != 68:
                            crops.append((ap.stem, None, None))
                            continue
                        try:
                            sz = size0
                            fmat = get_transform_mat(np.asarray(lm, np.float32), sz,
                                                     FaceType.WHOLE_FACE, scale=1.0)
                            # 顺序和 4.0 export_pipeline:720-721 完全一致：**uint8 整帧
                            # warp → 再 /255**。以前是先把 1080p 整帧转 float32 再 warp，
                            # 实测这多花的 25ms/帧（astype 11.8 + 浮点 warp 13.4 vs
                            # uint8 warp 0.18）只是把同样的双线性插值算在 float 上，
                            # 结果只差舍入 —— 却把 4 个读线程全拖成瓶颈。
                            crop = cv2.warpAffine(frame, fmat, (sz, sz), flags=cv2.INTER_CUBIC)
                            crops.append((ap.stem,
                                          np.clip(crop.astype(np.float32) / 255.0, 0, 1), None))
                        except Exception as e:              # noqa: BLE001
                            crops.append((ap.stem, None, str(e)[:100]))
                    _bump("prep_ms", time.perf_counter() - tp)
                # ok=True 才是「这一帧能处理」；frame 只在人脸库归组时才需要带回主线程
                _put_frame((gi, key, True, frame if use_lib else None, None, crops))
        except Exception as e:                              # noqa: BLE001
            boss["exc"] = boss["exc"] or e
            stop.set()
            try:
                q_frames.put((None, None, False, None, None, None))
            except Exception:                               # noqa: BLE001
                pass

    # ---- 一批：拷进预分配缓冲 → 一次前向 → 交给写线程 ----
    jobs = {}          # dfm -> ([gi], [keys], [crops])
    # best_ms 用阶梯探测的最好成绩起手：否则第一批（可能已经偏大）会被当成基准，
    # 后面的退档守卫就永远不会触发
    stat = {"best_ms": bmeta.get("best_per_face_ms") or None, "shrunk": 0}

    def _flush(use_dfm: str, force: bool = False):
        nonlocal batch, done_faces, failed
        bucket = jobs.get(use_dfm)
        if not bucket:
            return
        gis, keys, crops = bucket
        while keys and (len(keys) >= batch or (force and keys)):
            n = min(len(keys), batch)
            ent = _dfm_session(use_dfm)
            size = int(ent[2])
            sizes[Path(use_dfm).name] = size
            if size0 == size and pre.shape[0] >= n:
                arr = pre[:n]
            else:
                arr = np.empty((n, size, size, 3), np.float32)
            for i in range(n):
                arr[i] = crops[i]
            ti = time.perf_counter()
            try:
                outs = _dfm_run_batch(ent, arr)
            except Exception as e:                          # noqa: BLE001
                if not (_is_oom(e) and n > 1):
                    raise
                step = max(1, n // 2)
                _stage_log("换脸：batch=%d 显存不够（%s）→ 退到 %d 再试"
                           % (n, str(e)[:80], step))
                batch = max(1, min(batch, step))
                stat["shrunk"] += 1
                outs = []
                for i in range(0, n, step):
                    outs.extend(_dfm_run_batch(ent, arr[i:i + step]))
            dt = time.perf_counter() - ti
            per_face = dt * 1000.0 / max(1, n)
            _bump("infer_ms", dt, n)
            # 变慢守卫：比历史最好成绩慢 60% 说明 batch 太大（显存抖动 / 换出），主动退档
            if stat["best_ms"] is None or per_face < stat["best_ms"]:
                stat["best_ms"] = per_face
            elif n > 1 and per_face > stat["best_ms"] * 1.6 and stat["shrunk"] < 4:
                half = max(1, n // 2)
                _stage_log("换脸：batch=%d 时每张 %.1f ms（最好 %.1f ms）→ 退到 %d"
                           % (n, per_face, stat["best_ms"], min(batch, half)))
                batch = max(1, min(batch, half))
                stat["shrunk"] += 1
            for i in range(n):
                q_write.put((gis[i], keys[i], outs[i][0], outs[i][1], outs[i][2]))
                done_faces += 1
            del gis[:n], keys[:n], crops[:n]

    def _submit(gi: int, stem: str, use_dfm: str, crop):
        _pend_add(gi)
        bucket = jobs.setdefault(use_dfm, ([], [], []))
        bucket[0].append(gi)
        bucket[1].append(stem)
        bucket[2].append(crop)
        _flush(use_dfm)

    _save(cursor)
    threads = []
    for i in range(writers):
        t = threading.Thread(target=_writer, daemon=True, name="pred-w%d" % i)
        t.start()
        threads.append(t)
    hi_all = total
    for r in range(readers):
        lo = cursor + (hi_all - cursor) * r // readers
        hi = cursor + (hi_all - cursor) * (r + 1) // readers
        if lo >= hi:
            continue
        t = threading.Thread(target=_reader, args=(lo, hi), daemon=True, name="pred-r%d" % r)
        t.start()
        threads.append(t)

    k = cursor
    cancelled = {"flag": False}
    got = 0            # 已经处理完的帧数（显示用）。k 是**续跑游标**，它会为了
                       # 「不留空洞」而滞后（4 个读线程各跑一段，游标只认连续前缀），
                       # 以前拿 k 算帧/秒 → 正好少报 readers 倍（实测 6.3 帧/秒 vs
                       # 真实 25.4 张/秒，差 4 倍，就是被当成「慢」的那个数）。
    try:
        for gi in range(cursor, total):
            _stage_check_cancel()
            item = q_frames.get()
            if item[0] is None:
                break
            got += 1
            gi2, key, ok, frame, fc, crops = item
            if not ok:
                _stage_log("换脸跳过 %s：frames/ 里没有这一帧（或读不出来）" % key)
                with plock2:
                    completed.add(gi2)
            elif use_lib:
                # 人脸库归组：要整帧 + embedding，只能在主线程做
                metas = []
                for ap in groups[gi2][1]:
                    m = read_dfljpg_meta(ap) or {}
                    metas.append({"ap": ap,
                                  "lm": m.get("source_landmarks") or m.get("landmarks"),
                                  "box": m.get("source_rect") or None})
                plans = _stage_pred_assign(cfg, frame, metas)
                for meta_f, use_dfm in zip(metas, plans):
                    lm = meta_f["lm"]
                    if not lm or len(lm) != 68:
                        failed += 1
                        _stage_log("换脸跳过 %s：没有 68 点" % meta_f["ap"].name)
                        continue
                    if not use_dfm:
                        unmatched += 1
                        continue
                    size = int(_dfm_session(use_dfm)[2])
                    fmat = get_transform_mat(np.asarray(lm, np.float32), size,
                                             FaceType.WHOLE_FACE, scale=1.0)
                    crop = cv2.warpAffine(frame, fmat, (size, size), flags=cv2.INTER_CUBIC)
                    _submit(gi2, meta_f["ap"].stem, use_dfm,
                            np.clip(crop.astype(np.float32) / 255.0, 0, 1))
            else:
                for stem, crop, err in (crops or []):
                    if crop is None:
                        failed += 1
                        _stage_log("换脸跳过 %s%s" % (stem, ("：" + err) if err else "：没有 68 点"))
                        continue
                    _submit(gi2, stem, dfm, crop)
            _commit_upto()
            k = commit["n"]
            el = max(0.001, time.time() - t0)
            _stage_progress("pred", 100.0 * got / total,
                            "换脸 %d/%d 帧（第 %d..%d 帧）· %d 张脸 · %.1f 帧/秒 · "
                            "%.1f 张/秒 · batch %d"
                            % (got, total, int(rng["start"]), int(rng["end"]), done_faces,
                               got / el, done_faces / el, batch),
                            frames_done=got, frames_total=total,
                            frame_fps=round(got / el, 2))
            if (gi - cursor) % 20 == 0:
                _save(k)
        else:
            k = total
    except _StageCancel:
        # 取消也要把游标落盘 —— 但**必须等排空之后**再落：以前在 except 里立刻
        # _save(commit)，那一刻队里还有没写完的脸，游标会停在 0（实测 60 张脸
        # 取消时 done_frames=0，续跑等于从头再来）。现在置标志位，排空在 finally 里做。
        cancelled["flag"] = True
    finally:
        stop.set()
        exc_t = _sys.exc_info()[0]
        broken = boss.get("exc") is not None or exc_t not in (None, _StageCancel)
        if not broken:
            for use_dfm in list(jobs):
                _flush(use_dfm, force=True)
        for _i in range(writers):
            q_write.put(None)
        if not broken:
            # 等写盘排空：游标必须对得上磁盘上真的有的文件（否则续跑会跳过没写的帧）
            t_wait = time.time()
            while time.time() - t_wait < 600:
                with plock2:
                    left = sum(int(v) for v in pend.values())
                if left <= 0:
                    break
                time.sleep(0.05)
        k = _commit_upto()
        for t in threads:
            t.join(timeout=5.0)
    if cancelled["flag"]:
        _save(k)
        _stage_log("换脸取消：停在 %d/%d 帧（游标已存；已跑完 %d 张脸）"
                   % (k, total, done_faces))
        raise _StageCancel()
    if boss.get("exc") is not None:
        raise boss["exc"]
    sec = round(time.time() - t0, 2)
    prof_out = {"read_ms": round(prof["read_ms"], 1), "prep_ms": round(prof["prep_ms"], 1),
                "infer_ms": round(prof["infer_ms"], 1), "write_ms": round(prof["write_ms"], 1),
                "readers": int(readers), "writers": int(writers),
                "per_face_ms": {
                    "read": round(prof["read_ms"] / max(1, total - cursor), 2),
                    "prep": round(prof["prep_ms"] / max(1, total - cursor), 2),
                    "infer": round(prof["infer_ms"] / max(1, prof["faces"]), 2),
                    "write": round(prof["write_ms"] / max(1, prof["faces"]), 2),
                }}
    _save(total, done=True, extra={
        "seconds": sec, "finished_at": time.time(),
        "faces_per_sec": round(done_faces / sec, 2) if sec else None,
        "frame_fps": round((total - cursor) / sec, 2) if sec else None,
        "profile": prof_out, "batch_final": int(batch)})
    _stage_log("换脸完成：第 %d..%d 帧（%d 帧）· %d 张脸 · 未匹配 %d · 失败 %d · %.2f s · "
               "%.1f 张/秒（batch %d · 读 %d/写 %d 线程 · 每张 读 %.1f + 预处理 %.1f + "
               "推理 %.1f + 写 %.1f ms）"
               % (int(rng["start"]), int(rng["end"]), total, done_faces, unmatched,
                  failed, sec, done_faces / sec if sec > 0 else 0.0, batch, readers,
                  writers, prof_out["per_face_ms"]["read"], prof_out["per_face_ms"]["prep"],
                  prof_out["per_face_ms"]["infer"], prof_out["per_face_ms"]["write"]))
    return {"count": done_faces, "frames": total, "unmatched": unmatched, "failed": failed,
            "seconds": sec, "faces_per_sec": round(done_faces / sec, 2) if sec else None,
            "batch": int(batch), "lane": "pipeline",
            "range": {"start": int(rng["start"]), "end": int(rng["end"])}}


# ---------------- ④ 遮罩推理 ----------------
_XSEG_SESS = {}          # (model_key, onnx 路径) -> (_xseg.XSegMasker)
_XSEG_SESS_LOCK = threading.RLock()


def _engine_workspace():
    """4.0 引擎仓的 workspace（模型历来也放那边）。拿不到就 None。"""
    try:
        from ._paths import engine_root
        r = engine_root()
        return r if r else None
    except Exception:                                       # noqa: BLE001
        return None


def _xseg_resolve(model):
    """按界面选择解析 XSeg / XSegLite 的 onnx（不加载，只给路径 + md5）。"""
    from . import _xseg as _xseglib
    return _xseglib.resolve(model, WORKSPACE, _engine_workspace())


def _mask_signature(src: dict, cfg: dict = None) -> str:
    """遮罩阶段的「来源指纹」：来源/模型/模型文件变了，mask/ 就失效（不许静默复用）。

    以前是 "xseg:auto" —— 对 XSeg / XSegLite 是同一个值，于是「界面从 XSegLite 切到
    XSeg 再点合成」会拿到上一个模型算的遮罩，而且没有任何提示。现在把**实际会用到的
    那个 onnx 的 md5 前 8 位**放进指纹：换模型、换文件、换目录都会让它失效。
    """
    if src.get("source") == "dfm":
        return "dfm"
    key = str(src.get("model") or "")
    try:
        res = _xseg_resolve(key)
    except Exception:                                       # noqa: BLE001
        return "xseg:%s" % (key or "auto")
    if not res.get("key"):
        return "xseg:%s" % (key or "auto")
    return "xseg:%s:%s" % (res["key"], (res.get("md5") or "")[:8] or "nomd5")


def _xseg_sess_alive(m) -> tuple:
    """缓存的 masker 还能直接用吗：TRT 的引擎文件还在不在？onnx 还是原来那份？

    为什么要查：用户重编引擎、删引擎、或者换了一个同名 onnx 之后，**不重启软件**再跑
    遮罩阶段，必须如实重新加载并如实回显新的后端 —— 不能因为「对象还挂在缓存里」就
    继续假装自己在用 TRT（这正是「引擎明明没了还在报 trt」的那类谎话）。
    返回 (alive, 原因)。
    """
    info = getattr(m, "info", {}) or {}
    if str(info.get("backend") or "") != "trt":
        # 反向也要查一遍：上次是回落到 onnxruntime，后来用户把引擎编出来了 ——
        # 下一次遮罩阶段就该用上 TRT，不能因为对象还挂在缓存里就一直跑 onnxruntime
        try:
            from . import _engines as _E
            sm, tv = _E.sm_tag(), _E.trt_version()
            md5 = info.get("md5") or _E.md5_fresh(str(info.get("name") or ""),
                                                  str(info.get("onnx") or ""))
            d = _E.engine_dir(WORKSPACE)
            for prec in ("int8", "bf16"):
                p = d / ("%s_%s_sm%s_trt%s.%s.engine"
                         % (info.get("name"), md5, sm, tv, prec))
                if p.is_file():
                    return False, "TRT 引擎已经编出来了（%s）" % p.name
        except Exception:                                   # noqa: BLE001
            pass
        return True, ""
    ep = str(info.get("engine_path") or "")
    if ep and not Path(ep).is_file():
        return False, "引擎文件不在了（%s）" % Path(ep).name
    try:
        from ._engines import md5_fresh
        cur = md5_fresh(str(info.get("name") or ""), str(info.get("onnx") or ""))
        if cur and info.get("md5") and cur != str(info["md5"]):
            return False, "onnx 变了（md5 %s → %s）" % (info["md5"], cur)
    except Exception:                                       # noqa: BLE001
        pass
    return True, ""


def _stage_xseg(model):
    """④ 用的 XSeg 遮罩器：**按界面选择**加载，后端走 _engines.try_engine。

    返回 (masker, res)；masker.info 里有 backend / engine / precision / reason。
    这里不走 app._get_xseg 是有意的：那份缓存按「请求的 key」存，但内部会静默退到
    另一个模型（Lite 丢了就跑 XSeg），遮罩阶段必须知道「选的和跑的到底是不是同一个」。
    缓存按 (模型, onnx 路径) 存 —— 切模型不会重复加载同一个文件；每次复用前还要过一遍
    _xseg_sess_alive（引擎被删/onnx 被换 → 重新加载，后端如实变化）。
    """
    from . import _xseg as _xseglib
    res = _xseglib.resolve(model, WORKSPACE, _engine_workspace())
    if not res.get("path"):
        raise _BadInput("找不到 XSeg 模型（workspace/model/XSegLite/xseglite.onnx 或 "
                        "workspace/model/XSeg/XSeg.onnx）—— 遮罩来源选了 %s 就必须有它；"
                        "找过：%s" % (res.get("label") or model or "XSeg",
                                     "、".join(res.get("candidates") or [])[:300]), 500)
    # 显存不够 + DFL_ALLOW_CPU_FALLBACK=1 时，遮罩也必须在 CPU 上：
    # 不然 dfm 让出了显存、XSeg 又去抢 TRT 引擎 —— 半吊子状态比全 CPU 更难查。
    cpu = _vram_cpu_on()
    key = (res["key"], str(res["path"]), bool(cpu))
    with _XSEG_SESS_LOCK:
        hit = _XSEG_SESS.get(key)
    if hit is not None:
        alive, why = _xseg_sess_alive(hit)
        if alive:
            return hit, res
        _stage_log("遮罩：缓存的遮罩器不能再用（%s）→ 重新加载并重新判定后端" % why)
        with _XSEG_SESS_LOCK:
            _XSEG_SESS.pop(key, None)
    _own_f0 = _vram_measure_begin()
    m = _xseglib.XSegMasker(str(res["path"]),
                            workspace=(None if cpu else WORKSPACE),
                            engine_name=res["engine"],
                            providers=(["CPUExecutionProvider"] if cpu else None))
    _vram_measure_end("xseg:" + Path(str(res["path"])).name, _own_f0)
    if cpu:
        _stage_log("遮罩：★ 当前在 CPU 上跑 ★ %s 走 CPU EP（DFL_ALLOW_CPU_FALLBACK=1）"
                   % Path(str(res["path"])).name)
    with _XSEG_SESS_LOCK:
        _XSEG_SESS[key] = m
    return m, res


# dst 遮罩的像素来源（写进 meta.stages.mask.dst_input）：aligned = aligned/*.jpg 里的
# 对齐脸（再经一张补偿矩阵映射到 facelib 的合成画布）。老版本没有这个字段
# （那时每次都要解码 1080p 原帧再对齐）。
_MASK_DST_INPUT = "aligned"


def _stage_do_mask(cfg: dict) -> dict:
    """阶段 4 · 遮罩推理：aligned/ + pred/ -> mask/（每个对齐脸两张灰图）。

    语义照 4.0 export_pipeline.py:758-799（_phase_mask_worker）：
      · XSeg / XSegLite 来源：mask/<key>_xseg_prd.png = XSeg(pred)，
        mask/<key>_xseg_dst.png = XSeg(dst 对齐脸)
      · DFM 来源：遮罩就是 dfm 自带的那两条分支（③ 已经落盘 prdm/dstdm），这一步
        只把它们按 mask/ 的命名**落盘留档**（便于检查），不跑任何推理。
    两种来源都会把 dfm 的两张遮罩一起放进 mask/（mask_mode=9 全乘要用）。
    游标 = meta.stages.mask.done_frames（对齐脸张数）。

    **dst 输入 = aligned/<frame>_<fidx>.jpg：不读 frames/，只补一张几何补偿矩阵**
      以前为了拿「dst 对齐脸」先解码 1080p 原帧（≈24 ms/张脸，同一帧多张脸还各解码一次），
      再用 facelib 的 WHOLE_FACE 变换压到 256 —— 解码原帧这一步是纯浪费：同一张脸就在
      aligned/*.jpg 里（切脸阶段 68 点仿射 _align.align_mat_from_68 的直接产物）。
      现在读 aligned，再用一张补偿矩阵（image_to_face_mat⁻¹ × facelib WHOLE_FACE，见
      _stage_dst_face_mat）映射到合成画布。**为什么要补偿**：这条链的合成画布是 facelib 的
      （③ pred、⑤ compose_frame 都用它），而 aligned 芯片在 2026-12 之前是另一张画布
      （68 点拟合，中位 0.36% 缩放 + 2.5° 旋转，尾部 13%）—— 直接 resize 会让 dst 遮罩
      整体错开，mask_mode 8/7 贴回原帧就是边缘一条挖进脸里的缝。重切之后两边已经是同一张
      矩阵，这张补偿对它们约等于恒等，留着是为了兼容没重切过的老项目。
      dst 输入整条读下来实测 5.6 ms/张（解码 3.4 + 元数据 0.6 + 矩阵 0.5 + 映射 1.1），
      省掉的是 1080p 解码（≈24 ms）和每张脸重算一遍的 68 点变换。
      读不出来 / 损坏的那一张**跳过并计数**（failed + 日志点名 + meta 的 dst_bad_aligned），
      既不拿全帧兜底、也不算进成功数；一张都没成就报错。

    **吞吐**：这条链以前是纯串行（读 pred → 元数据 → 读原帧 → XSeg(prd) → 写盘 →
    XSeg(dst) → 写盘 → 复制 dfm 遮罩），一台机器上 2059 张脸 ≈ 26 张/秒，而原生 TRT
    的 XSegLite 纯推理只要 1.44 ms/图（实测），也就是 92% 的时间不在推理上。现在拆成
    读 N 线程 / 推理 1 线程 / 写 M 线程三段（和 ③ 换脸同一套结构），墙钟从「各段相加」
    变成「取最长的那段」；dst 输入换成 aligned 之后读线程不再是瓶颈（口径见
    app/tools/bench/stage_perf_profile.py 与 %TEMP% 里的 before/after 基准）。
    """
    import sys as _sys
    import numpy as np

    # 流水线实况：遮罩 = 1 个 XSeg session（DFM 来源时这块不推理，仍然是 1 条 lane）
    _STAGE_METER.set_lanes({"xseg": ["xse#0"]})
    cache = Path(cfg["cache"])
    meta = cfg["meta"]
    st = meta["stages"].setdefault("mask", {})
    src = _mask_src(cfg.get("settings"), cfg.get("mask_mode"))
    cfg["mask_src"] = src
    sig = _mask_signature(src, cfg)
    aligned_all = _stage_files(_stage_aligned_dir(cache), (".jpg", ".jpeg"))
    if not aligned_all:
        raise _BadInput("aligned/ 里没有对齐脸 —— 先跑切脸（POST /api/merge/stage/faces）", 409)
    if not (meta["stages"].get("faces") or {}).get("done"):
        raise _BadInput("切脸（faces）还没完成 —— 先跑切脸，再补遮罩", 409)
    if not (meta["stages"].get("pred") or {}).get("done"):
        raise _BadInput("换脸（pred）还没完成 —— 遮罩阶段要用换脸后的正脸，先跑换脸", 409)
    # 帧范围：只对 [start, end] 里的对齐脸推理（total/游标都按这一段的张数数）
    rng = _stage_range(cfg)
    rkey = str(cfg.get("range_key") or rng.get("key") or "")
    rchg = bool(cfg.get("range_changed"))
    aligned = _stage_filter_range(rng, aligned_all, lambda p: p.stem.split("_")[0])
    total = len(aligned)
    pred_dir = _stage_pred_dir(cache)
    mask_dir = _stage_mask_dir(cache)
    mask_dir.mkdir(parents=True, exist_ok=True)

    if (st.get("signature") and st.get("signature") != sig and not cfg["force"]
            and int(st.get("done_frames") or 0) > 0):
        raise _BadInput("遮罩来源变了（%s → %s）：mask/ 里的东西已经失效，"
                        "不能静默复用。传 force=true 重跑遮罩阶段"
                        % (st.get("signature"), sig), 409)
    # 老版本（dst 输入 = 原帧重新对齐）算出来的 mask/ 不能静默复用：来源没变，但几何变了。
    # 这里不报 409（挡的是「换了模型还复用旧遮罩」），直接当 force 整段重算并说清原因。
    stale_dst = (bool(st.get("done")) and bool(st.get("inference"))
                 and str(st.get("dst_input") or "") != _MASK_DST_INPUT)
    if stale_dst:
        _stage_log("遮罩：现有 mask/ 是「dst 输入 = 原帧重新对齐」那一版算的（dst_input=%s）"
                   "→ 口径已改成 aligned/，几何变了，整段重算"
                   % (st.get("dst_input") or "-",))
    redo = bool(cfg["force"] or rchg or stale_dst)
    if (not redo) and st.get("done") and st.get("signature") == sig \
            and int(st.get("done_frames") or 0) >= total:
        _stage_log("遮罩：复用已有 %d 张遮罩（第 %d..%d 帧，不重跑；要重做传 force=true）"
                   % (total, int(rng["start"]), int(rng["end"])))
        return {"skipped": True, "count": int(st.get("count") or 0), "reason": "already-done"}

    # ---- 按界面选择加载模型；后端 = TRT 有就用、没有就 onnxruntime（结论全部回显）----
    xseg = None
    backend = "none"
    engine_file = ""
    engine_prec = ""
    fallback = ""
    model_onnx = ""
    model_label = str(src.get("label") or "")
    if src["source"] == "xseg":
        xseg, res = _stage_xseg(src.get("model"))
        info = dict(getattr(xseg, "info", {}) or {})
        backend = str(info.get("backend") or "ort")
        engine_file = str(info.get("engine") or "")
        engine_prec = str(info.get("precision") or "")
        fallback = str(info.get("reason") or "")
        model_onnx = str(info.get("onnx") or res.get("path") or "")
        model_label = res.get("label") or model_label
        if backend == "trt":
            _stage_log("遮罩：模型=%s（%s）· 后端=TRT(%s) · 引擎=%s"
                       % (model_label, Path(model_onnx).name, engine_prec or "?", engine_file))
        else:
            _stage_log("遮罩：模型=%s（%s）· 后端=onnxruntime · 没有可用的 TRT 引擎 → 回落"
                       "（原因：%s）" % (model_label, Path(model_onnx).name,
                                        fallback or "未知"))
    engine = backend          # 老字段（导出/验收还在用它）—— 语义就是 trt|ort|none
    cursor = 0 if redo else max(0, min(int(st.get("done_frames") or 0), total))
    if cursor:
        _stage_log("遮罩：从范围内第 %d 张续跑（前 %d 张已经在）" % (cursor + 1, cursor))
    _stage_log("遮罩：来源=%s · 模型=%s · 引擎=%s%s · 第 %d..%d 帧（%d 张对齐脸）%s"
               % (src.get("label") or src.get("source"), model_label or "-", engine,
                  ("(%s)" % engine_file) if engine_file else "",
                  int(rng["start"]), int(rng["end"]), total,
                  ("" if not src.get("model_note") else
                   " · 注意：" + str(src.get("model_note")))))

    t0 = time.time()
    # force / 换范围 / dst 输入口径变了 都从 0 数（游标和计数都相对范围）
    wrote = 0 if redo else int(st.get("count") or 0)
    if redo:
        st["count"] = 0
    failed = int(st.get("failed") or 0)
    means = []
    t_infer = 0.0          # 纯推理累计（两张 256 的 XSeg 前向，不含读图/写盘）
    # 逐段打点（和 ③ 换脸同一套口径：读图 / 预处理 / 推理 / 写盘，各自累计 ms）。
    # dst 输入改成 aligned 之后，几何映射并在读图段里（一次 558→256 的重采样）→ 预处理段
    # prep_ms 恒为 0（保留这个打点名，是为了和上一版的 profile 直接对比）；
    # 读图 = pred PNG 解码 + aligned JPEG 解码 + 补偿矩阵 + 映射到 256。
    prof = {"read_ms": 0.0, "prep_ms": 0.0, "infer_ms": 0.0, "write_ms": 0.0,
            "faces": 0, "images": 0, "bad_aligned": 0}
    bad_aligned = []       # 读不出来的 aligned 文件名（最多留 20 个，进 meta / 日志）
    plock = threading.Lock()

    def _bump(ms_name: str, sec: float, n: int = 0, images: int = 0):
        with plock:
            prof[ms_name] += sec * 1000.0
            if n:
                prof["faces"] += n
            if images:
                prof["images"] += images

    def _bad_aligned(name: str):
        """记一张读不出来的 aligned 文件（计数 + 最多留 20 个名字）。"""
        with plock:
            prof["bad_aligned"] = int(prof.get("bad_aligned") or 0) + 1
            if len(bad_aligned) < 20:
                bad_aligned.append(str(name))

    def _save(up_to: int, done: bool = False, extra: dict = None):
        node = {
            "done": bool(done), "done_frames": int(up_to), "total": total,
            "range": {"start": int(rng["start"]), "end": int(rng["end"])},
            "range_key": rkey, "range_total": int(total),
            "count": wrote, "failed": failed, "signature": sig,
            "mask_source": src.get("label") or src.get("source"),
            "mask_model": str(src.get("model") or ""),
            "mask_model_from": str(src.get("model_from") or ""),
            "mask_onnx": model_onnx, "mask_label": model_label,
            "engine": engine,                 # 老字段，语义 = backend
            "backend": backend,               # trt | ort | none
            "engine_file": engine_file,       # 实际加载的引擎文件名（trt 才有）
            "engine_precision": engine_prec,  # int8 / bf16
            "fallback_reason": fallback,      # 没上 TRT 的一句话原因（有引擎时为空）
            "size": 256, "inference": (src["source"] == "xseg"),
            # dst 遮罩的像素来源：aligned（aligned/*.jpg 本身）/ dfm（dfm 自带，不推理）。
            # 老版本没有这个字段 —— _stage_do_mask 靠它认出「从原帧重建」那一版并重算。
            "dst_input": (_MASK_DST_INPUT if src["source"] == "xseg" else "dfm"),
            "dst_bad_aligned": int(prof.get("bad_aligned") or 0),
            "dir": str(mask_dir), "resumed_from": int(cursor),
            "msg": ("遮罩完成：%d 张脸（第 %d..%d 帧 · 来源 %s）"
                    % (wrote, int(rng["start"]), int(rng["end"]),
                       src.get("label") or src.get("source"))) if done
                   else ("遮罩 %d/%d 张（第 %d..%d 帧）"
                         % (up_to, total, int(rng["start"]), int(rng["end"]))),
        }
        if extra:
            node.update(extra)
        meta["stages"]["mask"] = node
        _stage_meta_save(cache, meta)

    def _copy_dfm_masks(key: str):
        """把 ③ 的 dfm 遮罩按 mask/ 的命名落盘（来源=DFM 时就是这一步的产物）。"""
        n = 0
        for src_name, dst_name in (("_prdm", "_dfm_prd"), ("_dstdm", "_dfm_dst")):
            a = pred_dir / (key + src_name + ".png")
            if not a.is_file():
                continue
            try:
                _stage_write_atomic(mask_dir / (key + dst_name + ".png"), a.read_bytes())
                n += 1
            except OSError:
                pass
        return n

    # ---- 串行链 → 三段流水线 ----
    # 以前这里是一条串行链：读 pred PNG → 读 DFLJPG 元数据 → 读 1080p 原帧 → 对齐到
    # 256 → XSeg(prd) → 编码写盘 → XSeg(dst) → 编码写盘 → 复制 dfm 的两张遮罩。
    # 纯推理（原生 TRT int8，见 stage_perf_profile.py 的实测）只有 1.44 ms/张，
    # 两张也就 2.9 ms；2059 张脸却只有 26 张/秒（≈38 ms/张）—— 也就是 92% 的时间
    # 不在推理上，而且串行意味着「各段相加」。现在拆成三段用队列连起来：
    #     读线程 ×N   pred PNG 解码 + aligned JPEG 解码并映射到 256（_stage_read_dst_face）
    #     推理线程    攒够 mask_batch 张脸，一次前向（ORT 动态 batch 或 TRT 逐张）
    #     写线程 ×M   PNG 编码 + 原子落盘 + dfm 两张遮罩的文件复制
    # 墙钟 ≈ max(三段) 而不是三段之和。
    # dst 输入换成 aligned 之后，读线程从「每张脸解码一张 1080p PNG（≈24 ms/张）」降到
    # 「解码一张 558 的 JPEG 再映射到 256（≈4 ms/张）」，同一帧的多张脸也不再共用帧缓存
    # （每张脸读自己那张 aligned）；写线程仍是「每张脸 4 个 PNG + 2 次文件复制」（~24 ms/张）。
    readers = max(1, int(cfg.get("mask_readers")
                         or min(6, max(2, (os.cpu_count() or 4) // 4))))
    writers = max(1, int(cfg.get("mask_writers")
                         or min(6, max(2, (os.cpu_count() or 4) // 8))))
    # 默认**不批量**（1）：本机实测 256² 逐张 4.79 ms/图 比 batch4 的 5.24 ms/图 更快
    # （ORT 在 batch>1 时换 kernel，流水线里还被队列拉长），而批量会换规约顺序、遮罩
    # 逐像素差 1~2 LSB。要试就传 mask_batch=N（引擎真吃批时的收益见 _xseg.mask_batch）。
    mbatch = max(1, int(cfg.get("mask_batch") or 1))
    if src["source"] == "xseg" and xseg is not None:
        try:
            # TRT 的 xseglite 引擎是静态 batch=1（_engines.PROFILES 单档位）→ 老实逐张；
            # ort-CUDA 的输入是动态 batch → 一次多张（256² 实测 4.02 → 2.68 ms/张）
            mbatch = max(1, min(mbatch, int(xseg.batch_limit())))
        except Exception:                                   # noqa: BLE001
            mbatch = 1
    _stage_log("遮罩：流水线 读 %d 线程 / 推理 batch %d / 写 %d 线程 · 后端 %s"
               % (readers, mbatch, writers, backend))

    if src["source"] != "xseg":
        # 来源=DFM：遮罩就是 ③ 落盘的那两张，这里只按 mask/ 的命名留档，不推理。
        k = cursor
        try:
            for k in range(cursor, total):
                _stage_check_cancel()
                key = aligned[k].stem
                try:
                    _copy_dfm_masks(key)
                    wrote += 1
                except Exception as e:                      # noqa: BLE001
                    failed += 1
                    _stage_log("遮罩跳过 %s：%s" % (key, str(e)[:140]))
                el = max(0.001, time.time() - t0)
                _stage_progress("mask", 100.0 * (k + 1) / total,
                                "遮罩 %d/%d 张（第 %d..%d 帧）· %.1f 张/秒"
                                % (k + 1, total, int(rng["start"]), int(rng["end"]),
                                   (k + 1 - cursor) / el),
                                frames_done=k + 1, frames_total=total,
                                frame_fps=round((k + 1 - cursor) / el, 2))
        except _StageCancel:
            _save(k)                # 取消也要把游标落盘：下次从这里接着跑
            _stage_log("遮罩取消：停在 %d/%d 张（游标已存）" % (k, total))
            raise
        sec = round(time.time() - t0, 2)
        done_n = max(1, total - cursor)
        _save(total, done=True, extra={
            "seconds": sec, "finished_at": time.time(),
            "fps": round((total - cursor) / sec, 2) if sec else None,
            "mask_mean": None, "infer_seconds": 0.0, "infer_ms_per_face": 0.0,
        })
        _stage_log("遮罩完成：第 %d..%d 帧 · %d 张 · 失败 %d · %.2f s · %.1f 张/秒 · "
                   "来源 %s（不推理，只落 dfm 自带遮罩）"
                   % (int(rng["start"]), int(rng["end"]), total, failed, sec,
                      (total - cursor) / sec if sec > 0 else 0.0,
                      src.get("label") or src.get("source")))
        return {"count": wrote, "failed": failed, "total": total, "seconds": sec,
                "mask_source": src.get("label") or src.get("source"),
                "signature": sig, "engine": engine, "backend": backend,
                "engine_file": engine_file,
                "range": {"start": int(rng["start"]), "end": int(rng["end"])},
                "infer_ms_per_face": 0.0}

    # ---------------- XSeg：读 / 推理 / 写 三段流水线 ----------------
    q_prep = queue.Queue(maxsize=max(2, readers * 2))
    q_write = queue.Queue(maxsize=max(4, writers * mbatch * 2))
    stop = threading.Event()
    boss = {"exc": None}
    pend, completed, plock2 = {}, set(), threading.Lock()
    commit = {"n": cursor}

    def _commit_upto():
        with plock2:
            n = commit["n"]
            while n in completed:
                completed.discard(n)
                n += 1
            commit["n"] = n
            return n

    def _pend_add(gi: int):
        with plock2:
            pend[gi] = int(pend.get(gi) or 0) + 1

    def _pend_done(gi: int):
        with plock2:
            left = int(pend.get(gi) or 0) - 1
            if left <= 0:
                pend.pop(gi, None)
                completed.add(gi)
            else:
                pend[gi] = left

    def _put_prep(item):
        while not stop.is_set():
            try:
                q_prep.put(item, timeout=0.2)
                return
            except queue.Full:
                continue

    def _u8(m):
        return None if m is None else np.clip(m * 255.0, 0, 255).astype(np.uint8)

    def _writer():
        nonlocal failed
        while True:
            item = q_write.get()
            if item is None:
                return
            gi, key, prd_u8, dst_u8 = item
            _stage_tick("xseg")      # 交接点：XSeg 遮罩 → mask/*.png
            tw = time.perf_counter()
            try:
                _copy_dfm_masks(key)
                if prd_u8 is not None:
                    # PNG level=1（无损，只是少压一层）：掩码是灰度，level 3 每张只快一点点
                    _stage_write_atomic(mask_dir / (key + "_xseg_prd.png"),
                                        _stage_imencode_png(prd_u8, 1))
                if dst_u8 is not None:
                    _stage_write_atomic(mask_dir / (key + "_xseg_dst.png"),
                                        _stage_imencode_png(dst_u8, 1))
            except Exception as e:                          # noqa: BLE001
                failed += 1
                _stage_log("遮罩写盘失败 %s：%s" % (key, str(e)[:140]))
            finally:
                _bump("write_ms", time.perf_counter() - tw)
                _pend_done(gi)

    def _reader(lo: int, hi: int):
        try:
            for gi in range(lo, hi):
                if stop.is_set():
                    break
                ap = aligned[gi]
                key = ap.stem
                tr = time.perf_counter()
                pm = pred_dir / (key + ".png")
                pred = _stage_read_image(pm) if pm.is_file() else None
                # dst 输入 = aligned/*.jpg（切脸阶段 68 点仿射的直接产物），只做一次「映射到
                # 合成画布」的重采样：不读 1080p 原帧（见 _stage_read_dst_face）。
                dst, why_dst = _stage_read_dst_face(ap)
                _bump("read_ms", time.perf_counter() - tr)
                why = ""
                if pred is None:
                    why = "pred/ 里没有换脸结果（先跑换脸）"
                elif dst is None:
                    why = "aligned 用不了 %s：%s" % (ap.name, why_dst)
                    _bad_aligned(ap.name)
                if why:
                    _put_prep((gi, key, None, None, why))
                    continue
                _put_prep((gi, key, pred, dst, ""))
        except Exception as e:                              # noqa: BLE001
            boss["exc"] = boss["exc"] or e
            stop.set()
            try:
                q_prep.put((None, None, None, None, None))
            except Exception:                               # noqa: BLE001
                pass

    jobs = []

    def _flush(force: bool = False):
        nonlocal t_infer, wrote
        while jobs and (len(jobs) >= mbatch or (force and jobs)):
            n = min(len(jobs), mbatch)
            chunk = jobs[:n]
            imgs, slots = [], []
            for i, (_gi, _k, pred, dst, _why) in enumerate(chunk):
                imgs.append(pred)
                slots.append((i, 0))
                if dst is not None:
                    imgs.append(dst)
                    slots.append((i, 1))
            ti = time.perf_counter()
            outs = xseg.mask_batch(imgs, out_size=256)
            dt = time.perf_counter() - ti
            t_infer += dt
            _bump("infer_ms", dt, n, len(imgs))
            _INFER_CALLS["xseg"] += len(imgs)
            prd_m = [None] * n
            dst_m = [None] * n
            for (i, slot), mk in zip(slots, outs):
                if slot:
                    dst_m[i] = mk
                else:
                    prd_m[i] = mk
                    means.append(float(np.asarray(mk).mean()))
            for i, (gi, key, _p, _d, _w) in enumerate(chunk):
                q_write.put((gi, key, _u8(prd_m[i]), _u8(dst_m[i])))
                wrote += 1
            del jobs[:n]

    _save(cursor)
    threads = []
    for i in range(writers):
        t = threading.Thread(target=_writer, daemon=True, name="mask-w%d" % i)
        t.start()
        threads.append(t)
    for r in range(readers):
        lo = cursor + (total - cursor) * r // readers
        hi = cursor + (total - cursor) * (r + 1) // readers
        if lo >= hi:
            continue
        t = threading.Thread(target=_reader, args=(lo, hi), daemon=True, name="mask-r%d" % r)
        t.start()
        threads.append(t)

    got = 0                     # 已经处理完的张数（显示用；游标 commit 会为了不留空洞而滞后）
    k = cursor
    cancelled = {"flag": False}
    try:
        for gi in range(cursor, total):
            _stage_check_cancel()
            item = q_prep.get()
            if item[0] is None:
                break
            got += 1
            gi2, key, pred, dst, why = item
            if pred is None:
                failed += 1
                _stage_log("遮罩跳过 %s：%s" % (key, why or "pred/ 里没有换脸结果（先跑换脸）"))
                with plock2:
                    completed.add(gi2)
            else:
                _pend_add(gi2)
                jobs.append((gi2, key, pred, dst, ""))
                _flush()
            _commit_upto()
            k = commit["n"]
            el = max(0.001, time.time() - t0)
            _stage_progress("mask", 100.0 * got / total,
                            "遮罩 %d/%d 张（第 %d..%d 帧）· %.1f 张/秒 · batch %d"
                            % (got, total, int(rng["start"]), int(rng["end"]),
                               got / el, mbatch),
                            frames_done=got, frames_total=total,
                            frame_fps=round(got / el, 2))
    except _StageCancel:
        cancelled["flag"] = True
    finally:
        stop.set()
        exc_t = _sys.exc_info()[0]
        broken = boss.get("exc") is not None or exc_t not in (None, _StageCancel)
        if not broken:
            _flush(force=True)
        for _i in range(writers):
            q_write.put(None)
        if not broken:
            t_wait = time.time()
            while time.time() - t_wait < 600:
                with plock2:
                    left = sum(int(v) for v in pend.values())
                if left <= 0:
                    break
                time.sleep(0.05)
        k = _commit_upto()
        for t in threads:
            t.join(timeout=5.0)
    if cancelled["flag"]:
        # 取消也要把游标落盘（排空之后再落，才和磁盘上真的有的文件对得上）
        _save(k)
        _stage_log("遮罩取消：停在 %d/%d 张（游标已存）" % (k, total))
        raise _StageCancel()
    if boss.get("exc") is not None:
        raise boss["exc"]
    if total > 0 and int(prof["faces"]) == 0:
        # 一张都没推理成功 → 别把空 mask/ 当「已完成」（下游合成会拿不到遮罩却以为跑过了）
        raise _BadInput("遮罩：%d 张对齐脸一张都没推理成功（aligned 读不出来 %d 张 / pred 缺失？）"
                        "—— 空 mask/ 不算成功。先看 aligned/ 和 pred/：%s"
                        % (total, int(prof.get("bad_aligned") or 0),
                           "、".join(bad_aligned[:5]) or "-"), 500)
    sec = round(time.time() - t0, 2)
    done_n = max(1, total - cursor)
    n_faces = max(1, prof["faces"])
    prof_out = {
        "read_ms": round(prof["read_ms"], 1), "prep_ms": round(prof["prep_ms"], 1),
        "infer_ms": round(prof["infer_ms"], 1), "write_ms": round(prof["write_ms"], 1),
        "readers": int(readers), "writers": int(writers), "mask_batch": int(mbatch),
        "backend": backend, "engine_file": engine_file,
        "images": int(prof["images"]),
        "dst_input": _MASK_DST_INPUT, "bad_aligned": int(prof.get("bad_aligned") or 0),
        "dst_mat": "image_to_face_mat⁻¹ · facelib WHOLE_FACE",
        # 每张脸各段各花多少 ms（读图 / 预处理（这一版恒 0）/ 2 次推理 / 编码+落盘）
        "per_face_ms": {
            "read": round(prof["read_ms"] / max(1, total - cursor), 2),
            "prep": round(prof["prep_ms"] / max(1, total - cursor), 2),
            "infer": round(prof["infer_ms"] / n_faces, 2),
            "write": round(prof["write_ms"] / n_faces, 2),
        },
        # 单张纯推理 = 总推理时间 / 推理张数（一张脸两张图）
        "per_image_ms": round(prof["infer_ms"] / max(1, prof["images"]), 3),
    }
    _save(total, done=True, extra={
        "seconds": sec, "finished_at": time.time(),
        "fps": round((total - cursor) / sec, 2) if sec else None,
        "mask_mean": (round(sum(means) / len(means), 4) if means else None),
        # TRT vs ORT 的对比就靠这两个数：同一个视频、同一份模型、同一个预处理
        "infer_seconds": round(t_infer, 3),
        "infer_ms_per_face": (round(t_infer * 1000.0 / done_n, 3)
                              if src["source"] == "xseg" else 0.0),
        "profile": prof_out,
    })
    _stage_log("遮罩完成：第 %d..%d 帧 · %d 张 · 失败 %d%s · %.2f s · %.1f 张/秒 · "
               "后端 %s%s · 来源 %s · dst 输入 %s · 每张 读 %.1f + 预处理 %.1f + 推理 %.1f"
               "（%.2f ms/图 × %d 图）+ 写 %.1f ms（读 %d / 写 %d 线程 · batch %d）"
               % (int(rng["start"]), int(rng["end"]), total, failed,
                  ("（其中 aligned 读不出来 %d）" % int(prof.get("bad_aligned") or 0))
                  if int(prof.get("bad_aligned") or 0) else "", sec,
                  (total - cursor) / sec if sec > 0 else 0.0,
                  backend, ("(%s)" % engine_file) if engine_file else "",
                  src.get("label") or src.get("source"), _MASK_DST_INPUT,
                  prof_out["per_face_ms"]["read"], prof_out["per_face_ms"]["prep"],
                  prof_out["per_face_ms"]["infer"], prof_out["per_image_ms"],
                  prof["images"], prof_out["per_face_ms"]["write"],
                  readers, writers, mbatch))
    return {"count": wrote, "failed": failed, "total": total, "seconds": sec,
            "mask_source": src.get("label") or src.get("source"),
            "signature": sig, "engine": engine, "backend": backend,
            "engine_file": engine_file, "dst_input": _MASK_DST_INPUT,
            "bad_aligned": int(prof.get("bad_aligned") or 0),
            "range": {"start": int(rng["start"]), "end": int(rng["end"])},
            "mask_batch": int(mbatch), "profile": prof_out,
            "infer_ms_per_face": (round(t_infer * 1000.0 / done_n, 3)
                                  if src["source"] == "xseg" else 0.0)}





def _stage_profile_blank() -> dict:
    """合成阶段的逐帧打点容器（五段 + 帧计数）。"""
    return {"frames": 0, "faces": 0, "no_face": 0, "swapped": 0,
            "read_ms": 0.0, "prep_ms": 0.0, "mask_ms": 0.0, "ct_ms": 0.0, "paste_ms": 0.0,
            "blend_ms": 0.0, "encode_ms": 0.0, "write_ms": 0.0, "frame_ms": 0.0}


def _gpu_mem_used_mb():
    """当前显存占用（MB）。拿不到就 None —— 只在验收/报告里用，不影响主流程。"""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
            creationflags=getattr(_app_mod, "_NO_WINDOW", 0))
        vals = [int(x) for x in (out.stdout or "").split() if x.strip().isdigit()]
        return max(vals) if vals else None
    except Exception:                                       # noqa: BLE001
        return None


def _stage_merge_face_inputs(cache, ap, cfg) -> dict:
    """一张对齐脸的合成输入：pred PNG + 三组遮罩（全从磁盘读，不跑推理）。"""
    import numpy as np
    from ._align import read_dfljpg_meta
    from ._merge_compose import _read_gray

    key = ap.stem
    pred_dir = _stage_pred_dir(cache)
    mask_dir = _stage_mask_dir(cache)
    m = read_dfljpg_meta(ap) or {}
    lm = m.get("source_landmarks") or m.get("landmarks")
    if not lm or len(lm) != 68:
        return None
    pred = _stage_read_image(pred_dir / (key + ".png"))
    if pred is None:
        return None
    return {
        "key": key,
        "landmarks": np.asarray(lm, np.float32),
        "pred": pred.astype(np.float32) / 255.0,
        "prd_dfm": _read_gray(pred_dir / (key + "_prdm.png")),
        "dst_dfm": _read_gray(pred_dir / (key + "_dstdm.png")),
        "xseg_prd": _read_gray(mask_dir / (key + "_xseg_prd.png")),
        "xseg_dst": _read_gray(mask_dir / (key + "_xseg_dst.png")),
    }


def _merge_worker_count(cfg: dict) -> int:
    """这次合成开几个 worker（默认 8；请求体 / 环境变量可覆盖，1..32）。"""
    for key in ("merge_workers", "merge_threads", "threads"):
        raw = cfg.get(key)
        if raw is None or (isinstance(raw, str) and not str(raw).strip()):
            continue
        try:
            return max(1, min(32, int(raw)))
        except (TypeError, ValueError):
            continue
    return max(1, min(32, int(_MERGE_WORKERS_DEFAULT or 8)))


def _merge_cv2_threads(cfg: dict, workers: int) -> int:
    """每个 worker 里该给 OpenCV 留几条内部线程（0 = 这一版不去动 cv2 的设置）。

    OpenCV 的 parallel_for 线程池是**进程级**的：不显式设，每个 worker 里的
    warpAffine / seamlessClone / GaussianBlur 都会各自摊到全部 32 条线程上
    （cv2.getNumThreads() 默认 = cpu 数），8 个 worker 一起跑就是 256 条线程抢 16 核，
    每帧的 cv2 调用从 6ms 涨到 20-40ms —— 这正是「8 线程只有 31% CPU」的主因之一。
    核少、活多的时候（workers <= 1）反过来：让 cv2 自己把**一帧**切到多核才划算
    （实测 1 worker：cv2=32 时 4.27 帧/秒，cv2=1 时 2.86 帧/秒）。
    """
    raw = cfg.get("merge_cv2_threads")
    if raw is None:
        raw = _MERGE_CV2_THREADS_DEFAULT
    try:
        want = int(raw or 0)
    except (TypeError, ValueError):
        want = 0
    if want > 0:
        return min(64, want)
    cpu = os.cpu_count() or 4
    if int(workers) <= 1:
        return max(1, min(cpu, 8))
    # worker 一多就**别**再让 OpenCV 自己摊线程：1080×1920 · seamless · 8 worker 实测
    #   cv2=1 → 13.56 帧/秒 · 7.4 核      cv2=8 → 12.89 帧/秒 · 8.7 核
    # 同样的吞吐，cv2=1 少烧 1.3 个核（用户要留给训练/预览）。只有「核相对 worker
    # 还很富余」时才给 cv2 留内部线程。
    return max(1, min(4, cpu // (4 * max(1, int(workers)))))


def _stage_do_merge(cfg: dict) -> dict:
    """阶段 5 · 合成：**纯 CPU**。frames/ + aligned/ + pred/ + mask/ -> 就地覆盖 frames/。

    语义基准 = 4.0 MergeStudio/core/export_pipeline.py 的 _phase_merge_worker()
    （见 app/server/_merge_compose.py 顶部的逐段对照表）：遮罩准备（mask_mode 组合 +
    erode/blur）→ 色彩迁移（ct_mode）→ 融合（overlay / hist-match / seamless /
    seamless-hist-match）→ 反贴回原帧（face_mat 逆变换）→ 写盘。

    这一阶段**不许碰推理**：dfm / XSeg 都已经在 ③④ 落盘，这里只读文件。
    验收会断言「推理调用计数 = 0」且显存不增长（profile 里也有 gpu_mem 前后值）。
    续跑游标 = meta.stages.merge.done_frames（就地覆盖只往前推）。
    """
    import cv2
    import numpy as np
    from ._merge_compose import compose_frame

    cache = Path(cfg["cache"])
    meta = cfg["meta"]
    st = meta["stages"].setdefault("merge", {})
    frames_all = _stage_files(_stage_frames_dir(cache))
    if not frames_all:
        raise _BadInput("frames/ 里没有帧图 —— 先跑切帧", 409)
    # 帧范围：只覆盖 [start, end] 里的帧（就地覆盖 frames/，别的帧一个都不碰）
    rng = _stage_range(cfg)
    rkey = str(cfg.get("range_key") or rng.get("key") or "")
    rchg = bool(cfg.get("range_changed"))
    frames = _stage_filter_range(rng, frames_all)
    total = len(frames)
    if ((not cfg["force"]) and not rchg and st.get("done")
            and int(st.get("done_frames") or 0) >= total and total):
        _stage_log("合成：复用已完成的结果（第 %d..%d 帧，不重跑）"
                   % (int(rng["start"]), int(rng["end"])))
        return {"skipped": True, "count": total, "reason": "already-done"}
    for need, label in (("faces", "切脸"), ("pred", "换脸"), ("mask", "遮罩")):
        if not (meta["stages"].get(need) or {}).get("done"):
            raise _BadInput("合成要用 %s 的产物，但「%s」阶段还没完成 —— 先跑%s"
                            "（POST /api/merge/stage/%s）" % (label, label, label, need), 409)
    src = _mask_src(cfg.get("settings"), cfg.get("mask_mode"))
    sig = _mask_signature(src, cfg)
    mst = meta["stages"].get("mask") or {}
    if mst.get("signature") and mst.get("signature") != sig:
        raise _BadInput("遮罩来源变了（mask/ 是 %s 产出的，现在是 %s）—— 先重跑遮罩阶段"
                        "（force=true），别拿旧遮罩合成" % (mst.get("signature"), sig), 409)

    aligned = _stage_files(_stage_aligned_dir(cache), (".jpg", ".jpeg"))
    by_frame = {}
    for ap in aligned:
        by_frame.setdefault(ap.stem.split("_")[0], []).append(ap)

    # 游标 **相对范围**：0..total
    cursor = 0 if (rchg or (cfg["force"] and not meta.get("frame_swapped"))) \
        else int(st.get("done_frames") or 0)
    cursor = max(0, min(cursor, total))
    if meta.get("frame_swapped") and cursor == 0 and int(st.get("done_frames") or 0) == 0:
        raise _BadInput("frames/ 已经是合成结果（frame_swapped=true），但 meta 里没有续跑位置 —— "
                        "只能重跑切帧 + 切脸 + 换脸 + 遮罩", 409)
    if meta.get("frame_swapped") and rchg:
        _stage_log("合成：frames/ 已经就地合成过，这次换到第 %d..%d 帧 —— 这一段会在已合成的帧上"
                   "再合成一次；要绝对干净请先 force 重切帧"
                   % (int(rng["start"]), int(rng["end"])))
    if cursor:
        _stage_log("合成：从范围内第 %d 帧续跑（前 %d 帧已经是合成结果）" % (cursor + 1, cursor))

    meta["frame_swapped"] = True
    comp_cfg = dict(cfg.get("config") or {})
    comp_cfg["mask_mode"] = int((cfg.get("settings") or {}).get("mask_mode", 6) or 0)
    threads = _merge_worker_count(cfg)
    cv2_threads = _merge_cv2_threads(cfg, threads)
    png_level = int(cfg.get("png_level") if cfg.get("png_level") is not None else 1)
    _stage_log("合成（纯 CPU）：第 %d..%d 帧（%d 帧）· 遮罩 %s/%s · ct=%s · blend=%s · "
               "%d 线程（每个 cv2 %d 线程）· PNG level %d"
               % (int(rng["start"]), int(rng["end"]), total,
                  src.get("label") or src.get("source"),
                  comp_cfg.get("mask_mode"), comp_cfg.get("color_transfer_mode"),
                  comp_cfg.get("mode"), threads, cv2_threads, png_level))

    prof = dict(_stage_profile_blank())
    for k in ("read_ms", "prep_ms", "mask_ms", "ct_ms", "paste_ms", "blend_ms", "encode_ms",
              "write_ms", "frame_ms"):
        prof[k] = float(st.get("profile", {}).get(k) or 0.0)
    prof["frames"] = cursor
    swapped = int(st.get("swapped") or 0)
    copied = int(st.get("copied") or 0)
    no_face = int(st.get("no_face") or 0)
    failed = int(st.get("failed") or 0)
    t0 = time.time()
    start_cursor = cursor
    infer0 = int(sum(_INFER_CALLS.values()))
    sess0 = len(_DFM_SESS)
    gpu0 = _gpu_mem_used_mb()
    stop = threading.Event()
    lock = threading.Lock()

    def _save(up_to: int, done: bool = False, extra: dict = None):
        node = {
            "done": bool(done), "done_frames": int(up_to), "frames": total,
            "frames_all": len(frames_all),
            "range": {"start": int(rng["start"]), "end": int(rng["end"])},
            "range_key": rkey, "range_total": int(total),
            "swapped": swapped, "copied": copied, "no_face": no_face, "failed": failed,
            "resumed_from": int(start_cursor), "in_place": True,
            "aligned_faces": len(aligned), "pure_cpu": True,
            "threads": threads, "png_level": png_level,
            "mask_mode": comp_cfg.get("mask_mode"),
            "ct_mode": comp_cfg.get("color_transfer_mode"),
            "blend": comp_cfg.get("mode"),
            "profile": {k: round(float(v), 2) for k, v in prof.items()},
            "msg": ("合成完成：第 %d..%d 帧（%d 帧 · 换脸 %d · 原样 %d）"
                    % (int(rng["start"]), int(rng["end"]), total, swapped, copied)) if done
                   else ("合成 %d/%d 帧（第 %d..%d 帧）"
                         % (up_to, total, int(rng["start"]), int(rng["end"]))),
        }
        if extra:
            node.update(extra)
        meta["stages"]["merge"] = node
        _stage_meta_save(cache, meta)

    def _one(fp):
        """一帧：读图 → 逐脸合成 → PN G编码 → 写盘。返回 (rec, 各段耗时)。"""
        if stop.is_set():
            # 取消后不再开新帧（排队里的那些也直接放掉，别让 shutdown 干等）。
            return {"key": fp.stem, "ok": True, "no_face": True, "skipped": True}, {}
        t = time.time()
        tim = {}
        frame = _stage_read_frame(fp)
        if frame is None:
            return {"key": fp.stem, "ok": False, "why": "读不出来"}, tim
        tim["read_ms"] = (time.time() - t) * 1000.0
        aps = by_frame.get(fp.stem) or []
        faces = []
        for ap in aps:
            f = _stage_merge_face_inputs(cache, ap, cfg)
            if f is not None:
                faces.append(f)
        if not faces:
            return {"key": fp.stem, "ok": True, "no_face": True, "faces": len(aps)}, tim
        t_c = time.time()
        out = compose_frame(frame, faces, comp_cfg, tim)
        tim["frame_ms"] = (time.time() - t_c) * 1000.0
        t_e = time.time()
        data = _stage_imencode_png(out, png_level)
        tim["encode_ms"] = (time.time() - t_e) * 1000.0
        _stage_tick("compose")   # 交接点：合成 → merged/*.png
        t_w = time.time()
        _stage_write_atomic(fp, data)
        tim["write_ms"] = (time.time() - t_w) * 1000.0
        return {"key": fp.stem, "ok": True, "swapped": 1, "faces": len(faces)}, tim

    from concurrent.futures import ThreadPoolExecutor
    chunk = max(threads * 2, 8)
    # 流水线实况：合成 lane 数 = 这个池**实际**的线程数（不分段链写死 8）
    _STAGE_METER.set_lanes({"compose": ["t%d" % _i for _i in range(int(threads))]})
    prog = {"done": cursor}

    def _report():
        """进度 + meta（每 chunk 一次；和以前同一个口径）。"""
        el = max(0.001, time.time() - t0)
        rate = (prog["done"] - start_cursor) / el
        eta = int((total - prog["done"]) / rate) if rate > 0 and prog["done"] < total else None
        n = max(1, prof["frames"] - cursor)
        _stage_progress(
            "merge", 100.0 * prog["done"] / total,
            "合成 %d/%d 帧（第 %d..%d 帧）· %.1f 帧/秒（换脸 %d · 原样 %d）"
            % (prog["done"], total, int(rng["start"]), int(rng["end"]),
               rate, swapped, copied),
            frames_done=prog["done"], frames_total=total,
            frame_fps=round(rate, 2), eta_s=eta,
            swapped=swapped, copied=copied, no_face=no_face,
            profile={"per_frame_ms":
                     {k: round(float(prof.get(k) or 0.0) / n, 2)
                      for k in ("read_ms", "prep_ms", "mask_ms", "ct_ms", "paste_ms",
                                "blend_ms", "encode_ms", "write_ms", "frame_ms")},
                     "fps": round(rate, 2)})
        _save(prog["done"])

    # cv2 内部线程数：跟 worker 数一起调（见 _merge_cv2_threads 的注释）。
    cv2_prev = 0
    cv2_set = 0
    if cv2_threads > 0:
        try:
            cv2_prev = int(cv2.getNumThreads())
            if cv2_prev != int(cv2_threads):
                cv2.setNumThreads(int(cv2_threads))
                cv2_set = int(cv2_threads)
        except Exception:                                         # noqa: BLE001
            cv2_prev, cv2_set = 0, 0
    pool = None
    try:
        # **整段一个池**：以前是每个 chunk 重建一次 ThreadPoolExecutor，
        # 每个 chunk 末尾都有一道隐式栅栏（shutdown(wait=True)）—— 最后一批 worker
        # 空等，下一批才开工。8 worker / chunk=16 时白丢 6~12% 吞吐。
        pool = ThreadPoolExecutor(max_workers=threads, thread_name_prefix="merge")
        done_in_chunk = 0
        # 一次性把所有帧提交进池：worker 干完一帧立刻拿下一帧，没有 chunk 边界。
        for rec, tim in pool.map(_one, frames[cursor:]):
            if _stage_cancelled():
                stop.set()
                raise _StageCancel()
            if rec.get("skipped"):
                continue                  # 取消尾巴：不算帧、不动游标
            if not rec.get("ok"):
                failed += 1
                _stage_log("合成跳过 %s：%s" % (rec.get("key"), rec.get("why")))
            elif rec.get("no_face"):
                copied += 1
                no_face += 1
            else:
                swapped += 1
            prof["frames"] += 1
            prof["faces"] += int(rec.get("faces") or 0)
            for kk, vv in tim.items():
                prof[kk] = float(prof.get(kk) or 0.0) + float(vv or 0.0)
            prog["done"] += 1
            done_in_chunk += 1
            if done_in_chunk >= chunk:
                done_in_chunk = 0
                _report()
        _report()                     # 收尾也报一次（进度到 100%），和以前每 chunk 末尾一致
    except _StageCancel:
        _save(prog["done"])
        _stage_log("合成取消：停在 %d/%d 帧（游标已存）" % (prog["done"], total))
        raise
    finally:
        if pool is not None:
            # 取消时别等已经排队的帧：置 stop + cancel_futures，worker 下一帧开头就退出。
            pool.shutdown(wait=not stop.is_set(), cancel_futures=True)
        if cv2_set:
            try:
                cv2.setNumThreads(cv2_prev)
            except Exception:                                     # noqa: BLE001
                pass

    sec = round(time.time() - t0, 2)
    n = max(1, prof["frames"] - cursor)
    per = {k: round(float(prof.get(k) or 0.0) / n, 2)
           for k in ("read_ms", "prep_ms", "mask_ms", "ct_ms", "paste_ms", "blend_ms",
                     "encode_ms", "write_ms", "frame_ms")}
    infer_n = int(sum(_INFER_CALLS.values())) - infer0
    gpu1 = _gpu_mem_used_mb()
    sess_after = len(_DFM_SESS)
    fps = round((total - start_cursor) / sec, 2) if sec > 0 else 0.0
    prof_out = {
        "per_frame_ms": per,
        "frames": int(prof["frames"]), "faces": int(prof["faces"]),
        "threads": threads, "png_level": png_level,
        "fps": fps, "pure_cpu": True,
        "inference_calls": infer_n,
        "dfm_sessions_before": sess0, "dfm_sessions_after": sess_after,
        "gpu_mem_mb_before": gpu0, "gpu_mem_mb_after": gpu1,
        "wall_s": sec,
    }
    _save(total, done=True, extra={"seconds": sec, "profile": prof_out,
                                   "fps": fps, "inference_calls": infer_n})
    _stage_log("合成完成：第 %d..%d 帧（%d 帧 · 换脸 %d · 原样 %d，无人脸 %d · 失败 %d）· "
               "%.2f s · %.2f 帧/秒"
               % (int(rng["start"]), int(rng["end"]), total, swapped, copied,
                  no_face, failed, sec, fps))
    _stage_log("合成 profile（每帧 ms）：%s"
               % " · ".join("%s=%.2f" % (k, v) for k, v in per.items()))
    _stage_log("合成阶段推理调用：%d（必须为 0）· 显存 %s → %s MB"
               % (infer_n, gpu0, gpu1))
    if infer_n:
        _stage_log("警告：合成阶段发生了 %d 次推理 —— 违反「纯 CPU」约定" % infer_n)
    return {"count": total, "swapped": swapped, "copied": copied, "no_face": no_face,
            "failed": failed, "seconds": sec, "fps": fps,
            "range": {"start": int(rng["start"]), "end": int(rng["end"])},
            "inference_calls": infer_n, "profile": per}



def _vid_cmd(pattern: str, dest: Path, video: Path, fps: float, img_start: int = 1,
             audio_ss: float = None, audio_t: float = None, limit: int = None) -> list:
    """一帧不差地构造导出用的 ffmpeg 命令（分段链和快速导出**共用这一份**）。

    ```
    ffmpeg -y ... -framerate FPS -start_number IMG_START -i <帧序列>
           [-ss AUDIO_SS -t AUDIO_T] -i <源视频>
           -map 0:v:0 -map 1:a?
           -vf pad=ceil(iw/2)*2:ceil(ih/2)*2,scale=out_range=tv
           -c:v libx264 -crf 16 -pix_fmt yuv420p -c:a copy [-frames:v LIMIT]
           [-movflags +faststart] -progress pipe:1 <dest>
    ```

    三条不能动的口径（改了就成片不一致）：
      · crf16 + yuv420p + out_range=tv：PNG 帧是 full range，不显式转 limited 会被标成 yuvj420p；
      · 音轨 -c:a copy：不重编码、**不带 -shortest**（带上去会按视频/音轨较短的一边截断，
        帧范围的片子就少一截音）；
      · 奇数宽高先 pad 一像素，否则 x264 直接拒。
    """
    pre_img = ["-start_number", str(int(img_start))]
    pre_aud = []
    if audio_ss is not None:
        pre_aud += ["-ss", "%.6f" % float(audio_ss)]
    if audio_t is not None:
        pre_aud += ["-t", "%.6f" % float(audio_t)]
    lim = ["-frames:v", str(int(limit))] if limit else []
    cmd = [str(_app_mod._ffmpeg_exe()), "-y", "-hide_banner", "-loglevel", "error",
           "-nostats", "-nostdin", "-framerate", "%.6g" % fps] + pre_img + \
          ["-i", str(pattern)] + pre_aud + ["-i", str(video),
           "-map", "0:v:0", "-map", "1:a?",
           "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2,scale=out_range=tv",
           "-c:v", "libx264", "-crf", "16", "-pix_fmt", "yuv420p",
           "-c:a", "copy"] + lim
    if Path(dest).suffix.lower() in (".mp4", ".mov", ".m4v"):
        cmd += ["-movflags", "+faststart"]
    cmd += ["-progress", "pipe:1", str(dest)]
    return cmd


def _vid_encode(cmd: list, part: Path, total: int, on_progress=None,
                cancelled=None) -> dict:
    """跑一条 ffmpeg 编码命令（cmd 末尾已经是 part 路径）→ {code, tail, seconds}。

    取消时收掉进程树再抛 _StageCancel（分段链和快速导出都按「取消不是失败」处理）。
    失败时**不动 dest**：.part 由调用方删 —— 半截文件绝不能看起来像成片。
    """
    cancelled = cancelled or _stage_cancelled
    t0 = time.time()
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
                    if cancelled():
                        _kill_proc_tree(proc)
                        raise _StageCancel()
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
                    if on_progress is not None:
                        on_progress(done)
            finally:
                code = proc.wait()
    finally:
        tail = ""
        try:
            with open(str(err_path), "rb") as rf:
                tail = rf.read()[-4000:].decode("utf-8", "ignore").strip()
        except OSError:
            tail = ""
        try:
            os.unlink(str(err_path))
        except OSError:
            pass
    return {"code": int(code), "tail": tail,
            "seconds": round(time.time() - t0, 2), "part": str(part), "cmd": list(cmd)}


def _stage_do_encode(cfg: dict) -> dict:
    """阶段 4 · 导出：frames/（已是合成结果）+ 源音轨 -> <视频名>_merged.mp4。

    帧是 PNG（无损），编码这一层不需要任何 qmin/qmax 兜底：crf16 / yuv420p /
    out_range=tv / 音轨 copy，和原来的成片参数一模一样。写 .part.mp4 再改名，
    中途挂掉不会留下"看起来像成片"的半截文件。
    """
    cache = Path(cfg["cache"])
    video = Path(cfg["video"])
    meta = cfg["meta"]
    st = meta["stages"].setdefault("encode", {})
    frames_all = _stage_files(_stage_frames_dir(cache))
    if not frames_all:
        raise _BadInput("frames/ 里没有帧图 —— 先跑合成", 409)
    # 帧范围：只编码 [start, end] 里的帧 -> 成片时长 = N / fps；音轨从源视频同一位置切
    rng = _stage_range(cfg)
    rkey = str(cfg.get("range_key") or rng.get("key") or "")
    rchg = bool(cfg.get("range_changed"))
    frames = _stage_filter_range(rng, frames_all)
    total = len(frames)
    out = Path(cfg["out"])
    if _same_path(out, video):
        raise _BadInput("输出不能是源视频本身：%s" % out)
    if ((not cfg["force"]) and not rchg and st.get("done")
            and out.is_file() and out.stat().st_size > 0):
        _stage_log("导出：复用已有成片 %s（不重跑）" % out)
        return {"skipped": True, "count": total, "out": str(out), "reason": "already-done"}
    if not (meta["stages"].get("merge") or {}).get("done"):
        raise _BadInput("合成（merge）还没完成——现在导出会得到一版没换脸的成片。先跑合成", 409)
    if not frames:
        raise _BadInput("frames/ 里没有第 %d..%d 帧 —— 这个范围超出了已切好的帧"
                        % (int(rng["start"]), int(rng["end"])), 409)

    fps = float(cfg.get("fps") or (meta.get("source") or {}).get("fps") or 30.0)
    try:
        out.parent.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        raise _BadInput("输出目录建不出来：%s（%s）" % (out.parent, e))
    part = out.with_name(out.stem + ".part" + out.suffix)

    pattern = str(_stage_frames_dir(cache) / "%06d.png")
    # 范围内的第一帧帧号（frames/ 是 000001.png 起的定长序号）
    start_no = (_stage_frame_index(frames[0].stem) or int(rng["start"])) if frames else int(rng["start"])
    dur = (total / fps) if fps > 0 else 0.0
    t0_audio = ((start_no - 1) / fps) if fps > 0 else 0.0
    # 全片：命令和以前**一模一样**（回归靠它）；有范围才加 -start_number / -frames:v / -ss / -t。
    # 命令本身收敛在 _vid_cmd()：快速导出（fused）调的是同一个函数，参数不许各写一套。
    cmd = _vid_cmd(pattern, part, video, fps,
                   img_start=(1 if rng["full"] else int(start_no)),
                   audio_ss=(None if rng["full"] else t0_audio),
                   audio_t=(None if rng["full"] else dur),
                   limit=(None if rng["full"] else int(total)))

    _stage_log("导出：第 %d..%d 帧（%d 帧 @ %.6g fps · 时长 %.4f s）-> %s"
               % (int(start_no), int(start_no) + total - 1, total, fps, dur, out.name))
    if not rng["full"]:
        _stage_log("导出：音轨从源视频 %.4f s 起 copy、长 %.4f s（帧边界误差 ≤ 1 帧）"
                   % (t0_audio, dur))

    _STAGE_METER.set_lanes({"encode": ["e0"]})   # ffmpeg 一个进程 = 一条 lane
    _enc_seen = {"n": 0}

    def _on_prog(done):
        pct = (100.0 * done / total) if total else 0
        _stage_progress("encode", pct, "编码 %d/%d 帧" % (done, total),
                        frames_done=min(done, total), frames_total=total)
        # 交接点：编码 → 成片。一次回调可能跨好几帧（ffmpeg 的 -progress 有节流），
        # 所以按增量补 tick —— 图上的光点才和「真的编了几帧」对得上。
        _n = int(done) - int(_enc_seen["n"])
        if _n > 0:
            _enc_seen["n"] = int(done)
            _stage_tick("encode", _n)

    enc = _vid_encode(cmd, part, total, on_progress=_on_prog)
    code, tail, sec = enc["code"], enc["tail"], enc["seconds"]

    if code != 0:
        try:
            if part.is_file():
                part.unlink()
        except OSError:
            pass
        raise _BadInput("编码失败（ffmpeg 退出码 %s）：%s" % (code, (tail or "")[-600:]), 500)
    if not part.is_file() or part.stat().st_size <= 0:
        raise _BadInput("编码跑完了却没有产物：%s" % part, 500)
    os.replace(str(part), str(out))
    # sec 来自 _vid_encode（它自己量的墙钟）—— 这里不再重算，免得两处口径打架
    mb = round(out.stat().st_size / 1048576.0, 2)
    meta["stages"]["encode"] = {
        "done": True, "count": total, "seconds": sec, "out": str(out), "mb": mb,
        "fps": fps, "crf": 16, "pix_fmt": "yuv420p",
        # 老字段 range = **色彩范围**（tv），别和帧范围混：帧范围在 frame_range/range_key
        "range": "tv", "color_range": "tv",
        "frame_range": {"start": int(start_no), "end": int(start_no) + total - 1},
        "range_key": rkey, "range_total": int(total),
        "duration": round(dur, 6), "audio_ss": round(t0_audio, 6), "audio": "copy",
        "finished_at": time.time(),
        "msg": "导出完成：%s（%.2f MB · 第 %d..%d 帧 · %.3f s）" % (
            out.name, mb, int(start_no), int(start_no) + total - 1, dur),
    }
    _stage_meta_save(cache, meta)
    _stage_log("导出完成：%s（%.2f MB · %.2f s · %d 帧 · 时长 %.4f s）"
               % (out, mb, sec, total, dur))
    _stage_log("编码命令：%s" % " ".join('"%s"' % a if " " in str(a) else str(a) for a in cmd))

    # ---- cache 的一生：成功导出 = 这份缓存没用了，删干净 ----
    removed = _stage_drop_cache(cfg)
    _stage_set(cache_removed=removed)
    return {"count": total, "out": str(out), "seconds": sec, "mb": mb,
            "duration": round(dur, 6), "start_frame": int(start_no),
            "end_frame": int(start_no) + total - 1,
            "range": {"start": int(rng["start"]), "end": int(rng["end"])},
            "cache_removed": removed}


def _stage_drop_cache(cfg: dict) -> bool:
    """导出成功后删掉整个 <名字>_cache/；失败/取消**不删**（留着续跑）。"""
    cache = Path(cfg["cache"])
    if cfg.get("keep_cache"):
        _stage_log("DFL_STAGE_KEEP=1：这次不删 cache（测试用）：%s" % cache)
        return False
    shutil.rmtree(str(cache), ignore_errors=True)
    gone = not cache.exists()
    _stage_log("cache 清理：%s（%s）" % ("已删除" if gone else "没删干净", cache))
    return gone


_STAGE_FN = {
    "frames": _stage_do_frames,
    "faces": _stage_do_faces,
    "pred": _stage_do_pred,
    "mask": _stage_do_mask,
    "merge": _stage_do_merge,
    "encode": _stage_do_encode,
}


# ---------------- 请求 -> cfg ----------------
_STAGE_KEEP = str(os.environ.get("DFL_STAGE_KEEP") or "").strip().lower() in ("1", "true", "yes", "on")
# ③ 换脸一批几张 / ⑤ 合成几个线程 / ⑤ 写 PNG 的压缩级别（1 = 体积换速度）
# ③ 换脸一批几张：**0 = 按可用显存自动选**（_pred_pick_batch）；env 或请求体给了就照用。
# 以前默认写死 8 —— 8GB 卡上 batch 16 会抖到 5 张/秒，12GB 空闲时 8 又浪费一半。
_PRED_BATCH_ENV = str(os.environ.get("DFL_PRED_BATCH") or "").strip()
_PRED_BATCH_DEFAULT = max(0, int(_PRED_BATCH_ENV)) if _PRED_BATCH_ENV.isdigit() else 0
def _env_int(name, default=0):
    """环境变量里的整数；没设/写坏了就用 default（不抛、不猜）。"""
    raw = str(os.environ.get(name) or "").strip()
    try:
        return int(raw, 10)
    except (TypeError, ValueError):
        return int(default)


# ⑤ 合成的并发度：**默认 8**（用户口径：8 个 worker，其余核留给训练/预览）。
#
# 以前这里写的是 max(4, min(8, cpu//4)) —— 在 16 核 32 线程上恰好也是 8，但在 8 核机
# 上是 4、在 64 线程服务器上还是 8：和核数无关的常数。现在**默认明确钉在 8**，只是想
# 换就能换（环境变量 / 请求体），别再去改代码。
#
# 线程数只是故事的一半：OpenCV 每个进程默认开 32 条内部线程，8 个 worker × 32 =
# 256 条可运行线程抢 16 个核。实测（1080×1920 · seamless · 120 帧，见 ⑤ 阶段报告）
# process_cpu 8.5 核、吞吐 11 帧/秒，而同样这堆活用 8 个独立进程只花 ~4.5 核就跑到
# 11.4 帧/秒 —— 一半的 CPU 在 OpenCV 的 spin-wait 里空转。所以两个旋钮必须一起给：
#   DFL_MERGE_WORKERS / body.merge_workers        worker 数（默认 8）
#   DFL_MERGE_CV2_THREADS / body.merge_cv2_threads
#                                                每个 worker 里 OpenCV 的线程数
#                                                （不设 = 自动：cpu//workers，上限 4）
_MERGE_WORKERS_DEFAULT = max(1, _env_int("DFL_MERGE_WORKERS",
                                         _env_int("DFL_MERGE_THREADS", 8)))
_MERGE_WORKERS_DEFAULT = min(32, _MERGE_WORKERS_DEFAULT)
# 兼容老名字（别的地方/老脚本还在用 _MERGE_THREADS_DEFAULT）
_MERGE_THREADS_DEFAULT = _MERGE_WORKERS_DEFAULT
_MERGE_CV2_THREADS_DEFAULT = max(0, _env_int("DFL_MERGE_CV2_THREADS", 0))
_MERGE_PNG_LEVEL = max(0, min(9, int(str(os.environ.get("DFL_MERGE_PNG_LEVEL") or "1").strip() or 1)))
# 帧图 JPEG 质量：**固定 100**（4:4:4），界面不再有这个旋钮。
# 理由（用户定的）：默认就是无损优先，低画质没有使用场景，留个能调的旋钮只会误导
# —— 上一版设置页那颗「质量 95」既不可编辑、后端也从来没读过它。
# 老客户端/验收脚本仍然可以传 jpeg_quality 覆盖（1..100）；不传就是 100。
# 注意：**PNG 路径完全不受影响**（无损，只认 png_compression）。
_STAGE_JPEG_QUALITY = 100


# ---------------- 帧范围（1 起、闭区间） ----------------
# 方案 B：**切帧仍然整段**（索引完整、成本极低），后面五个阶段只处理 [start, end]。
# 口径：都不传 = 全片；只给 start = start..结尾；只给 end = 1..end。
# 为什么 frames 也记 range：界面/验收/续跑都要能一眼看出「这一段是哪一段」，
# 而且下游阶段的指纹要带上范围 —— 改了范围，已经跑完的产物必须重跑。
_RANGE_START_KEYS = ("start_frame", "frame_start")
_RANGE_END_KEYS = ("end_frame", "frame_end")


def _range_int(name: str, raw):
    """帧范围里的一个数：None/空 -> None；不是整数 -> 422（不静默兜底）。"""
    if raw is None:
        return None
    if isinstance(raw, str) and not raw.strip():
        return None
    text = str(raw).strip()
    try:
        return int(text, 10)
    except (TypeError, ValueError):
        pass
    try:
        f = float(text)
    except (TypeError, ValueError):
        raise _BadInput("帧范围的 %s 要是整数，收到 %r" % (name, raw), 422)
    if not f.is_integer():
        raise _BadInput("帧范围的 %s 要是整数，收到 %r" % (name, raw), 422)
    return int(f)


def _stage_range_raw(body: dict) -> tuple:
    """请求体 -> (start_raw, end_raw)。认 start_frame/end_frame、frame_start/frame_end、range{}。"""
    body = _as_dict(body)
    s = e = None
    nested = body.get("range")
    if isinstance(nested, dict):
        s = nested.get("start", nested.get("start_frame"))
        e = nested.get("end", nested.get("end_frame"))
    if s is None:
        for k in _RANGE_START_KEYS:
            if body.get(k) is not None:
                s = body.get(k)
                break
    if e is None:
        for k in _RANGE_END_KEYS:
            if body.get(k) is not None:
                e = body.get(k)
                break
    return s, e


def _stage_range_norm(body: dict, total: int) -> dict:
    """请求体 + 源视频总帧数 -> {"start","end","total","all","full","key"}（1 起、闭区间）。

    非法输入一律 422，并说清楚是哪一条不对（调用方只负责把它原样回给界面）：
      start < 1 / end < 1 / start > end / end > 源总帧数。
    """
    total = int(total or 0)
    raw_s, raw_e = _stage_range_raw(body)
    s = _range_int("start_frame", raw_s)
    e = _range_int("end_frame", raw_e)
    if s is None and e is None:
        s, e = 1, (total if total > 0 else 0)
    else:
        if total <= 0:
            raise _BadInput("拿不到源视频的总帧数，没法按帧范围校验（先确认这个视频能解码）", 422)
        s = 1 if s is None else int(s)
        e = total if e is None else int(e)
        if s < 1:
            raise _BadInput("start_frame 从 1 数起，收到 %d" % s, 422)
        if e < 1:
            raise _BadInput("end_frame 从 1 数起，收到 %d" % e, 422)
        if s > e:
            raise _BadInput("帧范围反了：start_frame=%d > end_frame=%d" % (s, e), 422)
        if e > total:
            raise _BadInput("end_frame=%d 超过源视频总帧数 %d" % (e, total), 422)
    n = (int(e) - int(s) + 1) if (int(s) >= 1 and int(e) >= int(s)) else total
    full = bool(total > 0 and int(s) <= 1 and int(e) >= total) or \
        bool(total <= 0 and int(s) <= 1 and int(e) <= 0)
    return {"start": int(s), "end": int(e), "total": int(n), "all": total,
            "full": bool(full),
            "key": ("%d-%d" % (int(s), int(e))) if int(e) >= int(s) else "all"}


def _stage_range(cfg: dict) -> dict:
    """cfg -> 这次的帧范围（老 cfg / 没配就是全片）。"""
    r = cfg.get("range")
    if isinstance(r, dict) and r.get("start") is not None:
        return r
    total = int(cfg.get("total") or 0)
    return {"start": 1, "end": total, "total": total, "all": total,
            "full": True, "key": "all"}


def _stage_sync_range(cfg: dict, stage: str) -> bool:
    """把「这个阶段上一次跑的范围 vs 这次要的范围」算清楚，写回 cfg["range_changed"]。

    改了范围 = 阶段指纹变了：旧游标（相对旧范围数的）作废，这一段要重跑。
    幂等：precheck（单阶段 API）和 worker（链式导出）都调用它，两边口径永远一致。
    切帧永远整段，不存在「范围变了」。
    """
    rng = _stage_range(cfg)
    rkey = str(cfg.get("range_key") or rng.get("key") or "")
    st = ((cfg.get("meta") or {}).get("stages") or {}).get(stage) or {}
    old = str(st.get("range_key") or "")
    chg = bool(old) and old != rkey and stage != "frames"
    cfg["range_changed"] = bool(chg)
    return chg


def _stage_frame_index(key) -> int:
    """帧键（000123 / 000123_00 / Path）-> 1 起的帧号；认不出来返回 0。"""
    name = key.stem if isinstance(key, Path) else str(key or "")
    head = name.split("_")[0].strip()
    try:
        return int(head, 10)
    except (TypeError, ValueError):
        return 0


def _stage_in_range(rng: dict, key) -> bool:
    """这一帧在范围内吗。帧号认不出来的一律**保留**（宁可多跑一次，别把产物漏掉）。"""
    i = _stage_frame_index(key)
    if i <= 0:
        return True
    return int(rng["start"]) <= i <= int(rng["end"])


def _stage_filter_range(rng: dict, paths, keyfn=None) -> list:
    """按帧范围过滤一列文件（默认用 stem，对齐脸传 keyfn 取 000123_00 的前半段）。"""
    out = []
    for p in paths:
        k = keyfn(p) if keyfn else p
        if _stage_in_range(rng, k):
            out.append(p)
    return out


def _stage_dfm_pair(body: dict, need: bool):
    """dfm/模型名：合成阶段必须有；切帧/切脸/导出只是顺带记进 meta（没有也能跑）。"""
    explicit = bool(str(body.get("dfm") or "").strip()) or bool(str(body.get("model") or "").strip())
    if not explicit:
        explicit = bool(list((Path(WORKSPACE) / "model").glob("*.dfm")))
    if need or explicit:
        return _dfm_of(body)
    return "", ""


def _stage_out_path(video: Path, rng: dict, body: dict) -> Path:
    """输出文件名：带范围后缀（xxx_merged_0100-0200.mp4），免得把全片结果盖掉；全片不带。

    分段链（_stage_cfg）和快速导出（_fused_cfg）**同一条规则**：用户点的是同一个
    「输出路径」，两条路不该给出不一样的文件名。
    """
    tag = "" if rng["full"] else "_%04d-%04d" % (int(rng["start"]), int(rng["end"]))
    out_raw = str(body.get("output_dir") or body.get("output") or "").strip()
    if out_raw:
        op = _norm_path(out_raw)
        if op is None:
            raise _BadInput("output_dir 不合法：%s" % out_raw)
        if op.suffix.lower() == ".mp4":
            out = op if not tag else op.with_name(op.stem + tag + op.suffix)
        else:
            if op.exists() and op.is_file():
                raise _BadInput("输出路径是一个已存在的文件：%s（要目录或 .mp4）" % op)
            out = op / (video.stem + "_merged" + tag + ".mp4")
    else:
        out = video.parent / (video.stem + "_merged" + tag + ".mp4")
    if out.is_dir():
        raise _BadInput("输出路径是个目录：%s（要 .mp4 文件）" % out)
    return out


def _stage_cfg(body: dict, stage: str, chain: bool = False) -> dict:
    """请求体 -> 这一次阶段运行的 cfg（也会把 meta.json 读出来、缺字段补齐）。"""
    body = _as_dict(body)
    video = _norm_path(body.get("video") or body.get("source_video"))
    cache = _norm_path(body.get("cache_dir"))
    if video is None and cache is not None:
        base = cache.name
        if base.lower().endswith("_cache"):
            base = base[:-6]
        for ext in sorted(_VIDEO_EXT):
            cand = cache.parent / (base + ext)
            if cand.is_file():
                video = cand
                break
        if video is None:
            raise _BadInput("从 cache_dir 反推不到源视频：%s（同目录没有 %s*）" % (cache, base), 404)
    if video is None:
        raise _BadInput("缺少 video：要一个视频文件路径（或 cache_dir）")
    if not video.is_file():
        raise _BadInput("视频不存在：%s" % video, 404)
    if video.suffix.lower() not in _VIDEO_EXT:
        raise _BadInput("不是视频文件：%s（只认 %s）"
                        % (video.suffix or "无扩展名", " ".join(sorted(_VIDEO_EXT))))
    if cache is None:
        cache = Path(str(video.parent / (video.stem + "_cache")))
    try:
        cache.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        raise _BadInput("cache 目录建不出来：%s（%s）" % (cache, e))

    probe = _vid_probe(video)
    # 帧范围：1 起、闭区间；都不传 = 全片。非法（start>end / start<1 / end>总帧数）一律 422。
    rng = _stage_range_norm(body, int(probe.get("frames") or 0))
    rkey = str(rng["key"])
    settings = _settings_val(body.get("settings"))
    meta = _stage_meta_load(cache)
    if not meta:
        meta = _stage_meta_blank(video, probe, settings)
    meta = _stage_meta_ready(meta, video, probe)
    if settings:
        meta["settings"] = dict(settings)
    # 范围进 meta（界面/验收/续跑都读它）；只有真的变了才写盘，别每次点都改 meta。
    prev_rng = meta.get("range") if isinstance(meta.get("range"), dict) else {}
    meta["range"] = {"start": int(rng["start"]), "end": int(rng["end"]),
                     "total": int(rng["total"]), "frames": int(rng["all"]),
                     "full": bool(rng["full"]), "key": rkey}
    if (int(prev_rng.get("start") or 0) != int(rng["start"])
            or int(prev_rng.get("end") or 0) != int(rng["end"])
            or int(prev_rng.get("frames") or 0) != int(rng["all"])):
        _stage_meta_save(cache, meta)

    fps = _float_val("fps", body.get("fps"), 1.0, 240.0, None)
    dfm, model_name = _stage_dfm_pair(body, need=(stage == "merge"))
    xseg = _xseg_of(body)

    out = _stage_out_path(video, rng, body)

    snap = dict(_cache_snap())
    snap.update({"mode": "video", "path": str(video), "dir": "", "names": [], "srcdir": ""})
    library_raw = str(body.get("faces_project") or body.get("face_library")
                      or body.get("library") or "").strip()
    plan_payload = {"project": library_raw, "faces_project": library_raw}
    face_thr = _float_val("face_threshold", body.get("face_threshold"), 0.0, 1.0, None)
    if face_thr is not None:
        plan_payload["face_threshold"] = float(face_thr)

    models = []
    if body.get("models"):
        models = _models_of(body) or []
    if not models and dfm:
        models = [{"name": str(model_name), "dfm": str(dfm), "faces": [], "index": 0}]
    if not models:
        raise _BadInput("没有可用的 .dfm 模型（合成阶段要它）：先在训练页把模型导出成 .dfm（%s）"
                        % (Path(WORKSPACE) / "model"), 409)
    if not model_name:
        model_name = str(models[0].get("name") or "")

    return {
        "video": str(video), "cache": str(cache), "out": str(out),
        "total": int(probe.get("frames") or 0),
        # 帧范围（1 起闭区间）：range_total = 后五个阶段真正要处理的帧数
        "range": rng, "range_key": rkey, "range_total": int(rng["total"]),
        "range_full": bool(rng["full"]), "range_changed": False,
        "fps": float(fps if fps else (probe.get("fps") or 0.0)),
        "probe": probe, "settings": dict(settings), "meta": meta,
        "force": bool(_bool_val(body.get("force"), False)),
        "keep_cache": bool(_STAGE_KEEP or _bool_val(body.get("keep_cache"), False)),
        "detector": _detector_val(body.get("detector")),
        "det_align": _align_detector(body)[0],
        "det_align_src": _align_detector(body)[1],
        "thresh": _float_val("thresh", body.get("thresh"), 0.0, 1.0, 0.5),
        "align_thresh": _float_val("align_thresh", body.get("align_thresh"), 0.0, 1.0, 0.5),
        "dfm": str(dfm), "model": str(model_name), "xseg": str(xseg),
        "mask_mode": int((settings or {}).get("mask_mode", 6) or 0),
        "mask_src": _mask_src(settings, (settings or {}).get("mask_mode")),
        "config": dict(_merge_settings_to_config(settings)),
        "snap": snap, "plan_payload": plan_payload, "single_models": models,
        "use_library": True, "chain": bool(chain), "stages": [stage],
        "library": library_raw, "face_threshold": face_thr,
        # ③ 的批大小 / ⑤ 的线程数与 PNG 压缩级别（都能从请求体或环境变量调）
        "pred_batch": _int_val("batch", body.get("batch") or body.get("pred_batch"), 0, 64,
                               _PRED_BATCH_DEFAULT),
        # ⑤ 的并发度：默认 8（用户口径），请求体 merge_workers / 老名字 threads、
        # merge_threads 都认；环境变量 DFL_MERGE_WORKERS / DFL_MERGE_THREADS 兜底。
        "merge_workers": _int_val(
            "merge_workers",
            body.get("merge_workers") if body.get("merge_workers") is not None
            else (body.get("threads") if body.get("threads") is not None
                  else body.get("merge_threads")),
            1, 32, _MERGE_WORKERS_DEFAULT),
        # 每个 worker 里 OpenCV 的线程数（0 = 按核数/worker 数自动算）
        "merge_cv2_threads": _int_val("merge_cv2_threads", body.get("merge_cv2_threads"),
                                      0, 64, _MERGE_CV2_THREADS_DEFAULT),
        # 兼容：老代码/老日志还读 merge_threads
        "merge_threads": _int_val(
            "merge_threads",
            body.get("merge_workers") if body.get("merge_workers") is not None
            else (body.get("threads") if body.get("threads") is not None
                  else body.get("merge_threads")),
            1, 32, _MERGE_WORKERS_DEFAULT),
        "png_level": _int_val("png_compression", body.get("png_compression"), 0, 9,
                              _MERGE_PNG_LEVEL),
        # 帧图 JPEG 质量：固定 100（界面没有旋钮）；老客户端传了就认，但**默认 100**
        "jpeg_quality": _int_val("jpeg_quality", body.get("jpeg_quality"), 1, 100,
                                 _STAGE_JPEG_QUALITY),
        # ③④ 两条流水线的并行度（0 = 自动；验收/测速脚本可以点名覆盖）
        "pred_readers": _int_val("pred_readers", body.get("pred_readers"), 0, 16, 0),
        "pred_writers": _int_val("pred_writers", body.get("pred_writers"), 0, 16, 0),
        "mask_readers": _int_val("mask_readers", body.get("mask_readers"), 0, 16, 0),
        "mask_writers": _int_val("mask_writers", body.get("mask_writers"), 0, 16, 0),
        "mask_batch": _int_val("mask_batch", body.get("mask_batch"), 0, 32, 0),
    }


def _stage_precheck(cfg: dict, stage: str) -> dict:
    """阶段前的闸门：缺前置产物就 409；已完成就 skip（幂等）；frames 被覆盖过就拒。

    六步链：frames → faces → pred → mask → merge → encode，每一步只读它前面那几步的
    产物。少跑一步就 409 点名前一步（不再"偷偷在合成里现算"）。
    """
    cache = Path(cfg["cache"])
    meta = cfg["meta"]
    frames = _stage_files(_stage_frames_dir(cache))
    aligned_all = _stage_files(_stage_aligned_dir(cache), (".jpg", ".jpeg"))
    # 帧范围：后五个阶段只认 [start, end] 里的帧；下面的 skip / 覆盖判定全按范围内的数算
    rng = _stage_range(cfg)
    frames_in = _stage_filter_range(rng, frames)
    aligned = _stage_filter_range(rng, aligned_all, lambda p: p.stem.split("_")[0])
    st = meta["stages"].get(stage) or {}
    force = bool(cfg.get("force"))
    # 上一次这个阶段跑的是另一段范围 -> 旧游标（相对旧范围数）作废：自动重跑，不是 409。
    # 产物是按帧号落盘的，换了范围就把这段重算一遍 —— 日志里会说清楚（见 merge_stage）。
    chg = _stage_sync_range(cfg, stage)

    def _need(stage_name: str, why: str = ""):
        if not (meta["stages"].get(stage_name) or {}).get("done"):
            raise _BadInput("%s还没完成 —— 先跑%s（POST /api/merge/stage/%s）%s"
                            % (_STAGE_LABEL.get(stage_name, stage_name),
                               _STAGE_LABEL.get(stage_name, stage_name), stage_name,
                               ("：" + why) if why else ""), 409)

    def _gap(up: str, label: str):
        """上游阶段只跑过某一段：盖不住这次要的范围就 409，并点名要重跑哪一个。"""
        ust = meta["stages"].get(up) or {}
        u = ust.get("range") if isinstance(ust.get("range"), dict) else {}
        if not u or up == "frames":
            return                       # frames 永远整段切；老 meta 没写范围 = 整段
        us = int(u.get("start") or 1)
        ue = int(u.get("end") or 0)
        if us <= int(rng["start"]) and (ue <= 0 or ue >= int(rng["end"])):
            return
        raise _BadInput("%s只处理了第 %d..%d 帧，这次要第 %d..%d 帧 —— 先用同样的范围重跑%s"
                        "（force=true），别拿别的范围的结果凑"
                        % (label, us, ue, int(rng["start"]), int(rng["end"]), label), 409)

    if stage == "frames":
        # 「已经切完了」先判：续跑时 frames 往往已经被合成结果覆盖过，
        # 这一步本来就无事可做（跳过），不该拿 frame_swapped 去拦它。
        if st.get("done") and frames and not force:
            hint = int((meta.get("source") or {}).get("frames") or 0)
            if not hint or len(frames) == hint:
                return {"skip": True, "reason": "already-done", "count": len(frames)}
        if meta.get("frame_swapped") and not force:
            raise _BadInput("frames/ 已经被合成结果覆盖（meta.frame_swapped=true）—— 重切会把原帧"
                            "盖回来、却留下「已合成」的标记。要重来请传 force=true："
                            "会清掉 frames/ + aligned/ + pred/ + mask/ 和后面几个阶段的标记。", 409)
        return {"skip": False}

    if stage == "faces":
        if not frames:
            raise _BadInput("frames/ 是空的 —— 先跑切帧（POST /api/merge/stage/frames）", 409)
        if (st.get("done") and not force and not chg
                and int(st.get("frames") or 0) == len(frames) and aligned_all):
            return {"skip": True, "reason": "already-done", "count": len(aligned)}
        return {"skip": False}

    if stage == "pred":
        if not frames:
            raise _BadInput("frames/ 是空的 —— 先跑切帧（POST /api/merge/stage/frames）", 409)
        _need("faces")
        _gap("faces", "切脸")
        if not aligned:
            raise _BadInput("aligned/ 里没有第 %d..%d 帧的对齐脸 —— 先用同样的范围跑切脸"
                            % (int(rng["start"]), int(rng["end"])), 409)
        groups = len(_stage_face_rows(aligned))
        if (st.get("done") and not force and not chg and int(st.get("done_frames") or 0) >= groups):
            return {"skip": True, "reason": "already-done", "count": int(st.get("count") or 0)}
        return {"skip": False}

    if stage == "mask":
        _need("faces")
        _need("pred", "遮罩阶段要拿换脸结果去跑 XSeg")
        _gap("faces", "切脸")
        _gap("pred", "换脸")
        if not aligned:
            raise _BadInput("aligned/ 里没有第 %d..%d 帧的对齐脸 —— 先用同样的范围跑切脸"
                            % (int(rng["start"]), int(rng["end"])), 409)
        src = _mask_src(cfg.get("settings"), cfg.get("mask_mode"))
        sig = _mask_signature(src, cfg)
        cfg["mask_src"] = src
        if (st.get("signature") and st.get("signature") != sig and not force
                and int(st.get("done_frames") or 0) > 0):
            raise _BadInput("遮罩来源变了（%s → %s）：mask/ 里的东西已经失效，不能静默复用。"
                            "传 force=true 重跑遮罩阶段" % (st.get("signature"), sig), 409)
        if (st.get("done") and not force and not chg and st.get("signature") == sig
                and int(st.get("done_frames") or 0) >= len(aligned)):
            return {"skip": True, "reason": "already-done", "count": int(st.get("count") or 0)}
        return {"skip": False}

    if stage == "merge":
        if not frames:
            raise _BadInput("frames/ 是空的 —— 先跑切帧", 409)
        if not frames_in:
            raise _BadInput("frames/ 里没有第 %d..%d 帧 —— 这个范围超出了已切好的帧"
                            % (int(rng["start"]), int(rng["end"])), 409)
        _need("faces")
        _need("pred")
        _need("mask")
        _gap("faces", "切脸")
        _gap("pred", "换脸")
        _gap("mask", "遮罩")
        src = _mask_src(cfg.get("settings"), cfg.get("mask_mode"))
        sig = _mask_signature(src, cfg)
        mst = meta["stages"].get("mask") or {}
        if mst.get("signature") and mst.get("signature") != sig:
            raise _BadInput("遮罩来源变了（mask/ 是 %s 产出的，现在是 %s）—— 先重跑遮罩阶段"
                            "（force=true），别拿旧遮罩合成" % (mst.get("signature"), sig), 409)
        if st.get("done") and not force and not chg:
            return {"skip": True, "reason": "already-done", "count": len(frames_in)}
        if meta.get("frame_swapped") and int(st.get("done_frames") or 0) == 0:
            raise _BadInput("frames/ 已经是合成结果（frame_swapped=true）但没有续跑位置 —— "
                            "只能重跑切帧 + 切脸 + 换脸 + 遮罩", 409)
        return {"skip": False}

    if stage == "encode":
        if not frames:
            raise _BadInput("frames/ 是空的 —— 先跑切帧 + 合成", 409)
        if not frames_in:
            raise _BadInput("frames/ 里没有第 %d..%d 帧 —— 这个范围超出了已切好的帧"
                            % (int(rng["start"]), int(rng["end"])), 409)
        _need("merge")
        _gap("merge", "合成")
        out = Path(cfg["out"])
        if st.get("done") and out.is_file() and out.stat().st_size > 0 and not force and not chg:
            return {"skip": True, "reason": "already-done", "count": len(frames_in)}
        return {"skip": False}

    return {"skip": False}


def _stage_counts(cache) -> dict:
    """各阶段产物数量（status 里给界面/验收看）。"""
    pred_dir = _stage_pred_dir(cache)
    mask_dir = _stage_mask_dir(cache)
    return {"frames": len(_stage_files(_stage_frames_dir(cache))),
            "faces": len(_stage_files(_stage_aligned_dir(cache), (".jpg", ".jpeg"))),
            "pred": len([x for x in _stage_files(pred_dir) if "_prdm" not in x.name
                         and "_dstdm" not in x.name]),
            "masks": len(_stage_files(mask_dir))}


def _stage_stages_meta(meta: dict) -> dict:
    st = meta.get("stages") if isinstance(meta, dict) else {}
    if not isinstance(st, dict):
        st = {}
    return {k: (dict(st.get(k)) if isinstance(st.get(k), dict) else {}) for k in _STAGE_ORDER}


def _stage_done_list(meta: dict) -> list:
    st = _stage_stages_meta(meta)
    return [k for k in _STAGE_ORDER if st.get(k, {}).get("done")]


def _stage_next(meta: dict) -> str:
    st = _stage_stages_meta(meta)
    for k in _STAGE_ORDER:
        if not st.get(k, {}).get("done"):
            return k
    return ""


def _stage_brief_meta(cfg: dict, extra: dict = None) -> dict:
    cache = Path(cfg["cache"])
    meta = cfg["meta"]
    out = {
        "stage": str(cfg["stages"][0]) if cfg.get("stages") else "",
        "video": str(cfg.get("video") or ""),
        "cache_dir": str(cache),
        "out_video": str(cfg.get("out") or ""),
        "counts": _stage_counts(cache),
        "stages": _stage_stages_meta(meta),
        # 流水线实况（进度页那张 SVG 图的数据源）。这里不能叫 stages —— 那个键从 4.0
        # 起就是「每段的元数据」（done/msg/…），两个形状混一个键名是最容易出幻觉的坑。
        "stages_live": _STAGE_METER.snapshot(),
        "stages_done": _stage_done_list(meta),
        "next_stage": _stage_next(meta),
        "frame_swapped": bool(meta.get("frame_swapped")),
        "meta": meta,
        "dfm": str(cfg.get("dfm") or ""),
        "model": str(cfg.get("model") or ""),
        "detector": str(cfg.get("detector") or ""),
        "align_detector": str(cfg.get("det_align") or ""),
        "fps": float(cfg.get("fps") or 0.0),
        "total": int(cfg.get("total") or 0),
        # 帧范围：range 是这次要处理的一段，range_total = 这一段多少帧（相对进度的分母）
        "range": dict(cfg.get("range") or {}),
        "range_total": int(cfg.get("range_total") or 0),
        "range_changed": bool(cfg.get("range_changed")),
        "settings": dict(cfg.get("settings") or {}),
        # 帧图编码口径（JSON 回显）：PNG 无损，JPEG 固定 100 —— 界面没有这个旋钮，
        # 但验收/排障要能从响应里读到真实值，不用去翻 meta.json。
        "jpeg_quality": int(cfg.get("jpeg_quality") or _STAGE_JPEG_QUALITY),
        "png_level": int(cfg.get("png_level") if cfg.get("png_level") is not None else 1),
        "image_format": str((cfg.get("settings") or {}).get("image_format") or "jpg"),
    }
    if extra:
        out.update(extra)
    return out


# ---------------- 线程入口 ----------------
def _stage_worker(cfg: dict):
    """跑一个或一串阶段；异常一律落成终态，cache/缓存按口径处理。"""
    stages = list(cfg.get("stages") or ["frames"])
    ok, err, cancelled = True, "", False
    try:
        _STAGE_METER.begin()          # 流水线实况：这一单开始记（串行六段累同一份）
        for stage in stages:
            _stage_check_cancel()
            # 链式导出不走 precheck：范围指纹/游标失效必须在这里也算一遍（和单阶段口径一致）
            _stage_sync_range(cfg, stage)
            _stage_log("===== 阶段 %s（%s）开始 =====" % (stage, _STAGE_LABEL.get(stage, stage)))
            t0 = time.time()
            res = _STAGE_FN[stage](cfg) or {}
            sec = round(time.time() - t0, 2)
            if res.get("skipped"):
                _stage_log("阶段 %s：已完成，跳过（%s）" % (stage, res.get("reason") or ""))
            else:
                _stage_log("阶段 %s 完成：%s · %.2f s" % (stage, _stage_brief(res), sec))
            if cfg.get("chain") and stage == "encode" and res.get("out"):
                with _merge_job_lock:            # 老 /export/status 的 encode 段
                    _merge_job["encode"] = {"seconds": sec, "code": 0,
                                            "out": str(res.get("out")),
                                            "mb": res.get("mb")}
            with _STAGE_LOCK:
                secs = dict(_STAGE_JOB.get("seconds") or {})
                if not res.get("skipped"):
                    secs[stage] = sec
                _STAGE_JOB["seconds"] = secs
                done = list(_STAGE_JOB.get("stages_done") or [])
                if stage not in done:
                    done.append(stage)
                _STAGE_JOB["stages_done"] = done
                _STAGE_JOB["skipped"] = (_STAGE_JOB.get("skipped") or []) + \
                    ([stage] if res.get("skipped") else [])
                if res.get("out"):
                    _STAGE_JOB["out_video"] = str(res["out"])
                if res.get("cache_removed"):
                    _STAGE_JOB["cache_removed"] = True
    except _StageCancel:
        ok, cancelled = False, True
        err = "已取消（已完成的阶段和产物都保留，下次从这里接着跑）"
    except _BadInput as e:
        ok, err = False, str(e.msg)[:1200]
    except Exception as e:                                  # noqa: BLE001
        traceback.print_exc()
        ok, err = False, "%s: %s" % (e.__class__.__name__, str(e)[:600])
    finally:
        # 强制尝试失败：先回滚现场（放掉本进程会话 + 重算当下的账 + 改写 err），
        # 再往 meta/status 落字 —— 顺序反了用户就会看到「导出失败」而不是「确实不够」。
        if (not ok) and (not cancelled) and cfg.get("vram_force"):
            _rb = _vram_forced_rollback(
                str(_STAGE_LABEL.get(str(_STAGE_JOB.get("stage") or cfg.get("stage") or ""),
                                     "阶段")),
                _vram_models_mb([str(cfg.get("dfm") or "")]),
                {"mode": "none", "procs": 1}, err)
            if _rb:
                cfg["vram_oom"] = _rb
                err = str(_rb.get("msg") or err)
        cache = Path(cfg["cache"])
        meta = cfg.get("meta") or {}
        # 失败/取消：把状态和错误写进 meta（断电续跑时一眼能看出停在哪），cache 保留
        if not ok:
            try:
                if not isinstance(meta, dict):
                    meta = {}
                meta["last_error"] = err
                meta["last_stage"] = str(_STAGE_JOB.get("stage") or "")
                _stage_meta_save(cache, meta)
            except Exception:                               # noqa: BLE001
                pass
        else:
            try:
                _stage_meta_save(cache, meta)
            except Exception:                               # noqa: BLE001
                pass
        # ---- 缓存语义：整条链结束（成功/失败/取消）都清空组均值 + auto_plan ----
        cache_state = clear_face_group_cache("分段导出完成" if ok
                                             else ("分段导出取消" if cancelled else "分段导出失败"))
        _stage_log("缓存已清空：组均值 %s 条 · 其它 %s · 磁盘缓存文件 %s 个"
                   % ((cache_state or {}).get("profiles"),
                      (cache_state or {}).get("extra") or "无",
                      len((cache_state or {}).get("files") or [])))
        chain = bool(cfg.get("chain"))
        _oom = dict(cfg.get("vram_oom") or {})
        with _STAGE_LOCK:
            pct = 100 if ok else int(_STAGE_JOB.get("pct") or 0)
            if _oom:
                vp = dict(_oom.get("plan") or _STAGE_JOB.get("vram_plan") or {})
                vp["forced_failed"] = True
                _STAGE_JOB["vram_plan"] = vp
            _STAGE_JOB["vram_oom"] = (_oom or None)
            _STAGE_JOB.update({
                "running": False, "ok": bool(ok), "cancelled": bool(cancelled),
                "phase": ("done" if ok else ("cancelled" if cancelled else "failed")),
                "finished_at": time.time(), "error": "" if ok else err,
                "error_detail": "" if ok else "\n".join(list(_STAGE_JOB.get("log") or [])[-30:]),
                "pct": pct, "stage_pct": (100 if ok else int(_STAGE_JOB.get("stage_pct") or 0)),
                "cache_removed": (not cache.exists()),
                "cache": cache_state,
                "msg": (("%s 完成" % _STAGE_LABEL.get(str(_STAGE_JOB.get("stage") or ""), "导出"))
                        if ok else ("已取消" if cancelled else "失败：" + err[:80])),
                "meta": meta,
            })
        if chain:
            with _merge_job_lock:
                _merge_job.update({
                    "running": False, "ok": bool(ok) and not cancelled,
                    "cancelled": bool(cancelled), "finished_at": time.time(),
                    "pct": (100 if ok else 0),
                    "pct_at_failure": (None if ok else int(_merge_job.get("pct") or 0)),
                    "stage": (6 if ok else -1),
                    "msg": ("合成完成" if ok else ("已取消" if cancelled else "合成失败")),
                    "error": ("" if ok else err),
                    "error_detail": ("" if ok else "\n".join(_log_tail(
                        list(_merge_job.get("log") or []), 30))[:4000]),
                    "cache": cache_state,
                    # active=false = 这一单真的收尾完了（分段/快速导出共用一个位，
                    # 前端靠它决定「能不能再点一次导出」）
                    "active": False,
                })
            # 同一条理由：分段链收尾也把本进程会话放掉（下一次预览/导出懒加载）
            try:
                _vram_release_inproc("分段导出结束")
            except Exception:                               # noqa: BLE001
                pass
            _vid_close_log()


# ---------------- API ----------------
@router.get("/api/merge/stage/status")
@_api
def merge_stage_status(cache_dir: str = "", video: str = ""):
    """阶段状态：哪个阶段在跑 / 跑到哪 / 每阶段完成没 / cache 还在不在。

    没在跑的时候可以带 ?video= 或 ?cache_dir=，从磁盘上的 meta.json 把阶段标记读回来
    （刷新页面、重启进程之后照样能看出「停在哪一步」）。
    """
    with _STAGE_LOCK:
        snap = dict(_STAGE_JOB)
        log = list(snap.get("log") or [])
    running = bool(snap.get("running"))
    ok = snap.get("ok")
    cancelled = bool(snap.get("cancelled"))
    if running:
        phase = "running"
    elif cancelled:
        phase = "cancelled"
    elif ok is True:
        phase = "done"
    elif ok is False:
        phase = "failed"
    else:
        phase = "idle"

    cache = str(snap.get("cache_dir") or "")
    meta = snap.get("meta") if isinstance(snap.get("meta"), dict) else {}
    if not running:
        cand = _norm_path(cache_dir) if str(cache_dir or "").strip() else None
        if cand is None and str(video or "").strip():
            v = _norm_path(video)
            if v is not None:
                cand = Path(str(v.parent / (v.stem + "_cache")))
        if cand is None and cache:
            cand = Path(cache)
        if cand is not None:
            disk = _stage_meta_load(cand)
            if disk:
                meta = disk
                cache = str(cand)
    counts = _stage_counts(cache) if cache else {"frames": 0, "faces": 0}
    stages = _stage_stages_meta(meta)
    rng_m = meta.get("range") if isinstance(meta.get("range"), dict) else {}
    out = {
        "ok": (True if running else ok), "success": (None if running else ok),
        "phase": phase, "running": running, "cancelled": cancelled,
        "stage": str(snap.get("stage") or ""),
        "stage_index": int(snap.get("stage_index") if snap.get("stage_index") is not None else -1),
        "stage_pct": int(snap.get("stage_pct") or 0),
        "pct": int(snap.get("pct") or 0),
        "msg": str(snap.get("msg") or ""),
        "error": str(snap.get("error") or ""),
        "error_detail": str(snap.get("error_detail") or ""),
        "video": str(snap.get("video") or video or ""),
        "cache_dir": cache,
        "cache_exists": bool(cache) and Path(cache).exists(),
        "cache_removed": bool(snap.get("cache_removed")),
        "out_video": str(snap.get("out_video") or ""),
        # 这次阶段是按什么显存预算放行的（和被拒时的结构化字段同一份形状）
        "vram_plan": dict(snap.get("vram_plan") or {}),
        # 强制尝试撞上 OOM 的现场（非空 = 确实不够、已回滚；前端回到被拒态）
        "vram_oom": (dict(snap.get("vram_oom") or {}) or None),
        # 帧范围 + **相对进度**：frames_total/frames_done 都是「这一段」里的数
        "range": dict(rng_m),
        "range_key": str(rng_m.get("key") or ""),
        "range_total": int(rng_m.get("total") or 0),
        # 每个阶段各自真正处理过的那一段（encode 的 range 是色彩范围，另给 frame_range）
        "stage_ranges": {k: dict((v or {}).get("range") or {})
                         for k, v in stages.items()
                         if isinstance((v or {}).get("range"), dict)},
        "frames_total": int(snap.get("frames_total") or rng_m.get("total")
                            or counts["frames"] or 0),
        "frames_done": int(snap.get("frames_done") or 0),
        "frame_fps": float(snap.get("frame_fps") or 0.0),
        "eta_s": snap.get("eta_s"),
        "swapped": int(snap.get("swapped") or 0),
        "copied": int(snap.get("copied") or 0),
        "no_face": int(snap.get("no_face") or 0),
        "stages": stages,
        "stages_done": _stage_done_list(meta),
        "next_stage": _stage_next(meta),
        "frame_swapped": bool(meta.get("frame_swapped")),
        "counts": counts,
        "seconds": dict(snap.get("seconds") or {}),
        "stage_seconds": {k: float((v or {}).get("seconds") or 0.0)
                          for k, v in stages.items()},
        "stage_fps": {k: float((v or {}).get("fps") or 0.0) for k, v in stages.items()},
        "profile": snap.get("profile") if isinstance(snap.get("profile"), dict) else None,
        "mask_source": str((stages.get("mask") or {}).get("mask_source") or
                           (_mask_src(meta.get("settings")).get("label") or "")),
        "mask_signature": str((stages.get("mask") or {}).get("signature") or ""),
        # ④ 到底跑在哪个后端上（用户报的「选了 XSegLite 却在跑 onnx」就是靠这几个字段对的）
        "mask_backend": str((stages.get("mask") or {}).get("backend")
                            or (stages.get("mask") or {}).get("engine") or ""),
        "mask_engine": str((stages.get("mask") or {}).get("engine_file") or ""),
        "mask_precision": str((stages.get("mask") or {}).get("engine_precision") or ""),
        "mask_model": str((stages.get("mask") or {}).get("mask_model") or ""),
        "mask_onnx": str((stages.get("mask") or {}).get("mask_onnx") or ""),
        "mask_fallback": str((stages.get("mask") or {}).get("fallback_reason") or ""),
        # ③ 的吞吐与 batch（batch_source = auto/request，ladder 是实测表）
        "pred_batch": int((stages.get("pred") or {}).get("batch") or 0),
        "pred_batch_source": str((stages.get("pred") or {}).get("batch_source") or ""),
        "pred_ladder": list(((stages.get("pred") or {}).get("batch_meta") or {})
                            .get("ladder") or []),
        "pred_profile": dict((stages.get("pred") or {}).get("profile") or {}),
        "pred_faces_per_sec": float((stages.get("pred") or {}).get("faces_per_sec") or 0.0),
        "pred": dict(stages.get("pred") or {}),
        "skipped_why": str((stages.get(str(snap.get("stage") or "")) or {})
                           .get("skipped_why") or ""),
        "skipped": list(snap.get("skipped") or []),
        "started_at": snap.get("started_at"), "finished_at": snap.get("finished_at"),
        "log": log[-80:],
        "meta": meta,
        # 快速导出（融合流水线）：没有阶段概念，只有帧进度 + 帧预算；**不支持续跑**
        "fused": bool(snap.get("fused")),
        "resumable": (not bool(snap.get("fused"))),
        "frame_budget": snap.get("frame_budget"),
        "resumed": bool(snap.get("resumed")),
        # 生效配置 + 一致性警告（[eff] 行同款；进度页「当前生效」面板与顶部横幅）
        "eff": snap.get("eff"), "eff_warnings": list(snap.get("eff_warnings") or []),
        "warnings": list(snap.get("warnings") or []),
    }
    if running and out["frames_total"] and out["frames_done"]:
        rate = float(out["frame_fps"] or 0.0)
        if rate > 0:
            out["eta_s"] = int((out["frames_total"] - out["frames_done"]) / rate)
    return out


@router.post("/api/merge/stage/cancel")
@_api
def merge_stage_cancel(body: dict = None):
    """取消当前阶段（幂等）。已完成的阶段与产物都保留 —— 下次从停下的地方续跑。"""
    with _STAGE_LOCK:
        running = bool(_STAGE_JOB.get("running"))
        chain = bool(_STAGE_JOB.get("chain"))
        _STAGE_JOB["cancelled"] = True
        stage = str(_STAGE_JOB.get("stage") or "")
    if chain:
        with _merge_job_lock:
            _merge_job["cancelled"] = True
    return _ok(running=running, cancelled=True, stage=stage,
               note=("正在取消 %s" % _STAGE_LABEL.get(stage, stage)) if running else "当前没有阶段在跑")


def _stage_gpu_paths(cfg: dict, stage: str) -> list:
    """这个阶段**真要加载**的 GPU 模型（没有就不算显存账）。

    按阶段点名，不按"导出"一把梭：
      · pred / merge —— 每个选中模型一份 dfm session（_dfm_session 常驻）
      · mask         —— XSeg 遮罩 onnx（来源是 DFM 时本阶段本来就跳过）
      · faces        —— 检测器 + 2D106 关键点（onnx，走 CUDA EP）
      · frames/encode —— 一个 GPU 模型都不加载（ffmpeg 干活），不列
    """
    if stage in ("pred", "merge"):
        return [str(m.get("dfm") or "") for m in (cfg.get("single_models") or [])
                if str(m.get("dfm") or "")]
    if stage == "mask":
        src = cfg.get("mask_src") or {}
        if str(src.get("source") or "").lower() == "xseg" and str(cfg.get("xseg") or ""):
            return [str(cfg["xseg"])]
        return []
    if stage == "faces":
        # 2D106 是纯 CPU（见 _landmarks_model 的注释），不算显存账；只算检测器。
        try:
            flat = _app_mod._DET_ONNX[_detector_val(cfg.get("det_align"))]
            p = _app_mod._detector_path(flat)
        except Exception:                                   # noqa: BLE001
            p = None
        return [str(p)] if p else []
    return []


def _stage_model_resident(stage: str, paths: list, cfg: dict) -> bool:
    """这个阶段要的那份模型是不是**已经在当前进程里**了（缓存命中 = 不再新占显存）。

    阶段链是同一个后端进程里连着跑的：pred 建的 dfm session、faces 加载的检测器、
    mask 建的 XSeg 都常驻在缓存里。闸门如果还把「一份新会话」算进需求，等于把已经
    吃掉的那 2GB 再算一遍 —— 于是第 3 个阶段开始必被拒（而它其实一个字节都不用再分配）。
    """
    paths = [str(p) for p in (paths or []) if str(p or "")]
    if not paths:
        return False
    if stage in ("pred", "merge"):
        live = {str(k).split("|")[0] for k in list(_DFM_SESS)}
        return all(p in live for p in paths)
    if stage == "faces":
        return _det_cached(cfg.get("det_align"))
    if stage == "mask":
        try:
            with _XSEG_SESS_LOCK:
                live = {str(k[1]) for k in list(_XSEG_SESS) if len(k) > 1}
            return all(p in live for p in paths)
        except Exception:                                   # noqa: BLE001
            return False
    return False


def _stage_vram_gate(cfg: dict, stage: str, force: bool = False,
                     dry: bool = False) -> dict:
    """阶段起手前的显存闸门 —— **在起 job / 加载模型 / 开线程之前**跑。

    用户口径（截图 2）：显存不够时点导出，不许先起一个 job 再把六个阶段全打失败。
    所以这里只算账（_vram_plan，不加载任何模型），装不下就抛 400 + 结构化字段。

    frames / encode 不加载 GPU 模型，但它们同样要先过一遍账：
    连「一份 CUDA context + 一帧的解码缓冲」都放不下时，后面五个阶段一个也跑不成，
    与其让用户在第二步才吃到 400，不如第一步就说清楚。
    """
    label = _STAGE_LABEL.get(stage, stage)
    paths = _stage_gpu_paths(cfg, stage)
    opts = {"mode": "none", "procs": 1, "label": label,
            "aux": bool(stage not in ("frames", "encode"))}
    # 这一份会话是不是**已经**在本进程里了（pred → mask → merge 是同一条链，上一个
    # 阶段加载的 dfm 还在 _DFM_SESS 里）。在的话就不再算一份新会话 —— 那 2GB 已经在
    # 卡上、也已经在「空闲」里扣过了，再算一遍就是重复计数（阶段 3 之后必被拒）。
    try:
        if paths and _stage_model_resident(stage, paths, cfg):
            opts["resident"] = True
    except Exception:                                       # noqa: BLE001
        pass
    # 真装不下时允许放掉本进程里**别人的**会话（没有任务在跑才敢，见 _vram_plan）
    opts["reclaim_inproc"] = True
    # 真起手（非 dry_run）：先卸常驻预览 worker 的模型 → 再探 → 再算账（顺序见 _vram_plan）。
    # 分段链每个阶段都过这道闸，但 worker 里第一次之后就空了，第二次起是零成本的空转。
    opts["reclaim_first"] = (not dry)
    # force：「仍然尝试（可能 OOM）」—— 只放行最小档，真 OOM 由 _stage_worker 回滚
    opts["force"] = bool(force)
    return _vram_guard(paths, opts, label=label, force=bool(force))


@router.post("/api/merge/stage/{name}")
@_api
def merge_stage(name: str, body: dict = None):
    """跑一个阶段（frames / faces / merge / encode，也认 extract-frames / swap / export 这些别名）。

    幂等：已完成的阶段再点一次不会重跑（返回 skipped=true），force=true 才重做。
    并发：一次只允许一个阶段在跑（409）。
    显存：起 job 之前先过 _stage_vram_gate，装不下 → HTTP 400（结构化字段，
    不建 job、不加载模型、不开 worker）。dry_run=1 只算账不干活（前端预检用）。
    """
    stage = _stage_name(name)
    if not stage:
        raise _BadInput("未知阶段 %r（只认 %s）"
                        % (name, " / ".join(_STAGE_ORDER)), 404)
    body = _as_dict(body)
    _eff.log_request("/api/merge/stage/%s" % stage, body)
    cfg = _stage_cfg(body, stage)
    with _STAGE_LOCK:
        if _STAGE_JOB.get("running"):
            raise _BadInput("已经有阶段在跑（%s）—— 等它结束或点取消"
                            % _STAGE_LABEL.get(_STAGE_JOB.get("stage"), "?"), 409)
    plan = _stage_precheck(cfg, stage)
    if plan.get("skip"):
        _stage_set(running=False, ok=True, cancelled=False, phase="done",
                   stage=stage, stage_index=_STAGE_ORDER.index(stage), stage_pct=100,
                   pct=100, msg="%s 已完成，跳过" % _STAGE_LABEL.get(stage, stage),
                   video=str(cfg["video"]), cache_dir=str(cfg["cache"]),
                   out_video=str(cfg["out"]), meta=cfg["meta"],
                   range=dict(cfg.get("range") or {}),
                   frames_total=int(cfg.get("range_total") or 0),
                   stages_done=_stage_done_list(cfg["meta"]),
                   counts=_stage_counts(cfg["cache"]),
                   started_at=None, finished_at=time.time())
        return _ok(started=False, skipped=True, reason=plan.get("reason") or "already-done",
                   skipped_why=str(plan.get("skipped_why") or ""),
                   **_stage_brief_meta(cfg))

    # ---- 显存闸门：**skip 之后、建 job 之前** ----
    # 顺序有讲究：先跑前置检查（缺上游 → 409 点名要跑哪一步），再算显存账。
    # 反过来的话，一个"还没切帧"的用户会拿到「显存不够」，而真原因是那个 409。
    # 闸门只算账：不加载模型、不建 job、不开线程 —— 拒绝 = 这一单什么都没发生。
    vplan = _stage_vram_gate(cfg, stage, force=_bool_val(body.get("force"), False),
                             dry=_bool_val(body.get("dry_run"), False))
    cfg["vram_plan"] = vplan
    cfg["vram_force"] = bool(vplan.get("forced"))
    if _bool_val(body.get("dry_run"), False):
        return _ok(dry_run=True, stage=stage, started=False, skipped=False,
                   vram_plan=dict(vplan), need_mb=vplan.get("need_mb"),
                   avail_mb=vplan.get("avail_mb"), free_mb=vplan.get("free_mb"),
                   suggest=list(vplan.get("suggest") or []),
                   frames=int(vplan.get("frames") or 0),
                   batch=int(vplan.get("batch") or 0),
                   on_cpu=bool(_vram_cpu_on()))

    # 线程起来之后 meta 会被 worker 改：范围日志要用的旧指纹先在这里取好
    old_key = str(((cfg.get("meta") or {}).get("stages", {}).get(stage) or {})
                  .get("range_key") or "all")
    with _STAGE_LOCK:
        _STAGE_JOB.update({
            "running": True, "ok": None, "cancelled": False, "error": "",
            "error_detail": "", "phase": "running", "stage": stage,
            "stage_index": _STAGE_ORDER.index(stage), "stage_pct": 0, "pct": 0,
            "msg": "%s 启动…" % _STAGE_LABEL.get(stage, stage),
            "video": str(cfg["video"]), "cache_dir": str(cfg["cache"]),
            "out_video": str(cfg["out"]),
            # 进度一律**相对范围**：frames_total = 这一段要处理的帧数（frames 阶段才是全片）
            "frames_total": int(cfg.get("range_total") or cfg.get("total") or 0),
            "range": dict(cfg.get("range") or {}),
            "frames_done": 0, "frame_fps": 0.0, "eta_s": None, "swapped": 0,
            "copied": 0, "no_face": 0, "stages_done": _stage_done_list(cfg["meta"]),
            "log": [], "meta": cfg["meta"], "seconds": {}, "skipped": [],
            "cache_removed": False, "counts": _stage_counts(cfg["cache"]),
            "started_at": time.time(), "finished_at": None, "chain": False,
            "keep_cache": bool(cfg.get("keep_cache")), "force": bool(cfg.get("force")),
            "frames": [], "vram_plan": dict(cfg.get("vram_plan") or {}),
        })
    # ---- 生效配置：一行 [eff] + 一致性自检（进 status.warnings[] / 进度页横幅）----
    try:
        eff = _eff_publish("/api/merge/stage/%s" % stage, "stage:%s" % stage, body, cfg)
        with _STAGE_LOCK:
            _STAGE_JOB.update(_eff_status_patch(eff))
    except Exception as e:                                  # noqa: BLE001
        print("[eff] 生成失败（不影响这一单）：%s: %s"
              % (e.__class__.__name__, str(e)[:200]), flush=True)
    threading.Thread(target=_stage_worker, args=(cfg,), daemon=True,
                     name="merge-stage-%s" % stage).start()
    rng = cfg.get("range") or {}
    _stage_log("阶段 %s 启动：%s -> %s" % (stage, cfg["video"], cfg["cache"]))
    _stage_log("帧图编码：PNG 无损（compression level %s）/ JPEG quality=%d + 4:4:4"
               "（界面没有该旋钮，固定质量；PNG 不受影响）"
               % (cfg.get("png_level"), int(cfg.get("jpeg_quality")
                                            or _STAGE_JPEG_QUALITY)))
    _stage_log("帧范围：第 %s..%s 帧（共 %s 帧）%s"
               % (rng.get("start"), rng.get("end"), rng.get("total"),
                  "· 全片" if rng.get("full") else "· 只处理这一段"))
    if cfg.get("range_changed"):
        _stage_log("范围变了：%s 上一次跑的是另一段范围（指纹 %s → %s），旧游标作废，"
                   "本次从范围内第 1 帧重跑"
                   % (_STAGE_LABEL.get(stage, stage), old_key, cfg.get("range_key")))
    return _ok(started=True, skipped=False, **_stage_brief_meta(cfg))

# ============================================================
# 视频模式 · 快速导出（融合流水线 fused）
# ------------------------------------------------------------
# 口径（用户定的，实测见 app/tools/bench/fused_pipeline_bench.py，同段 1080p 120 帧）：
#   分段六阶段 7.20 帧/秒 · 六阶段合一**串行** 6.10（0.85x）· 合一**流水线** 20.55（2.85x）。
# 这份生产实现的实测（app/tools/accept/fused_export_accept.py，同一个样片 1080x1920 120 帧）：
#   17.4~18.2 帧/秒（分段 7.20 → ≈2.5x）· quality=fast 20.9 帧/秒。
#   差的那一点全在检测器：retinaface_10g 单帧 47~50ms = 约 20 帧/秒的硬底（它跑在
#   onnxruntime 上，workspace/_engines 里的 retinaface TRT 引擎 app 层没接上）。
# 所以视频模式默认走流水线（fused=1）；分段链保留成 fused=0 的对照/排障路径。
# DFL 项目导出（/api/merge/export-project，iperov 语义、单模型、按 original_name 对帧）
# 一个字都没动。
#
# 这里**不是第二套算法**，每一段调的都是现成的那一个函数：
#   检测 _detect_boxes · 检测框→68 点→对齐 _align.*（和 ② 切脸同一串）·
#   归组 _auto_face_plan（经 _vid_models_for_frame）· dfm _dfm_run_batch（session 走 _dfm_session、
#   批大小走 _pred_pick_batch）· XSeg _stage_xseg().mask_batch · 合成 _merge_compose.compose_frame ·
#   编码 _vid_cmd/_vid_encode（和 ⑥ 导出同一条命令）。
# 差别只在**数据怎么走**：分段是「写出 PNG → 下一段读回来」，fused 是一帧的内存对象
# 从检测一路带到合成，中间不落一个文件 —— 所以 <视频>_cache/ 一个都不建（_fused_run 里断言）。
#
# 像素为什么能和分段对得上（每一条都是"照抄"，不是"差不多"）：
#   · 68 点/对齐：coarse_align_from_5 → 2D106 → landmarks68_in_source → align_face_68，
#     和 ② 切脸同一串；compose 用的 source_landmarks 就是 align_face_68 里那份 round(4)
#     的结果 —— DFLJPG 里存的也是它，所以 fused 和分段手里的 landmarks 逐位相同。
#   · dfm 输入：get_transform_mat(lm, S, WHOLE_FACE) + INTER_CUBIC warp → /255，
#     和 ③ 换脸同序（**uint8 整帧 warp 再 /255**，不是先转 float）。
#   · dfm 输出：np.clip(face*255).astype(uint8) 之后**才** /255 —— 就是 PNG 往返那一步，
#     只是不落盘；XSeg 的输入/输出同理（先 u8 再喂，遮罩也是 u8 量化后 /255）。
#   · dst 遮罩输入：分段的输入是 aligned/*.jpg（q100 4:4:4，仍然是有损的），所以这里也做
#     一次同样的 JPEG 往返再映射到 256 —— 省掉它 dst 遮罩就会差一点点。
#   · 合成：compose_frame(原帧, faces, 同一份 cfg)，连 mask_mode 都取自同一处。
#
# 不支持续跑：帧是一次性的内存对象，没有游标可存。取消 = 整段作废（临时帧当场删掉），
# 进度页会说清楚「快速导出不支持续跑，需整段重跑」；要续跑就切回分段（fused=0）。
# ============================================================
_FUSED_BUDGETS = (30, 16, 8)        # 帧缓冲预算阶梯（用户口径：30/16/8 自动降档）
_FUSED_QDEPTH = 24                  # 段间有界队列深度（原型同值；>= dfm batch 才凑得满）
_FUSED_RESERVE_MB = 1024            # 模型 + CUDA context 的固定占用（416 dfm 实测 ≈1GB）
_FUSED_VRAM_SAFETY = 0.80           # 探测量打八折才是敢用的预算
_FUSED_ENV_VRAM = "DFL_FUSED_VRAM_FREE_MB"     # 验收/排障：假装可用显存只有这么多
_FUSED_ENV_BUDGET = "DFL_FUSED_BUDGET"         # 验收/排障：钉住帧预算（不走显存探测）
_FUSED_ENV_DUMP = "DFL_FUSED_DUMP"             # 验收/排障：把这几帧的逐脸输入 dump 出来
# 为什么要这个开关：「fused 和分段像素一致」这件事只能靠**同样的输入**证明，光看成片
# 分不清差在 pred / 遮罩 / 几何 / 编码哪一层。DFL_FUSED_DUMP=41,80 时把那几帧的原帧、
# 每张脸的 pred / prd / dst / xseg_prd / xseg_dst / 68 点全写成文件（<成片>.dump/），
# 直接和分段 cache/ 里的同名产物逐位比。默认关（一个文件都不写）。

# ---- 攒批与线程数：B 配置的口径（桌面实测见 _attr_bench/REPORT.md 第 8 节）----
# 攒批目标：dfm 一次前向喂几张。这条链原来只是「pend 攒满 _pred_pick_batch 就送」，
# 而那个阶梯只看**空张量单发**的每张耗时（416：B1=30.0 / B4=33.8 / B8=31.5），
# 于是 416 被选到 4、更小的模型会被选到 1 —— 判断依据错了：进流水线之后真正的对手是
# 「GPU 空转等批」，不是单张能跑多快。这里另给一个显式目标（默认 8 = B 配置），
# 实际取 max(阶梯值, 目标)，再按 session 原生上限 与 _PRED_BATCH_CAP 截。
# 像素不受它影响：同一个 session、同一份权重、同一批算子，只是一次喂进去的**张数**不同
# （对拍证据见 _attr_bench/fused_ab_*.json 与 dump 对拍）。
_FUSED_BATCH_TARGET = max(1, min(int(_PRED_BATCH_CAP), _env_int("DFL_FUSED_BATCH", 8)))
# 攒批超时（ms）：从**这一批第一张脸进来**的时刻起算；超时就把手里这点先送出去。
# 为什么必须有：片子尾巴、以及多模型时「某一路的脸一直凑不满目标」这两种情况下，
# 已经进来的脸会一直压着不动 —— 那一帧也就一直进不了合成。0 = 关（一直攒到目标或 END）。
# 默认 20ms —— **这是量出来的，不是拍的**（_attr_bench/fused_ab_wait*.json，90 帧 ×3 次中位）：
#   超时 0（=一直攒满 8）15.62 帧/秒 · 250ms 15.82 · **20ms 16.79**（384 模型，同段同参）。
# 为什么「超时越短反而越快」：384 dfm 单张 32~38ms，而这条线的硬底是检测（retinaface_10g
# 50~55ms/帧），dfm 早就不在关键路径上了；batch 8 的代价是 GPU 段每 ~400ms 才动一次，
# 一次憋出 8 帧，把下游 compose/encode 从「连续小批」变成「一阵一阵」，实测 compose 的
# 墙钟从 108~121ms 拉到 126~161ms —— 吞吐反而掉。改成 20ms 之后平均批次 1~2 张，
# dfm 每张贵 3~4ms，但整条线多 10%。**所以「攒到 8」是上限，不是目标**：
# 上游真能喂满（>400 张脸/秒）时它照样攒满 8；平时它就是一条不憋气的流水线。
_FUSED_BATCH_WAIT_MS = max(0, _env_int("DFL_FUSED_BATCH_WAIT_MS", 20))
# 对齐/预处理线程 与 合成线程（B 配置：4 / 8）。以前对齐是 min(4, cpu//3) —— 核少的机器
# 会自己掉到 1~2，同一个产品在两台机器上跑出两种口径。现在钉成常量，要改走环境变量。
_FUSED_PREP_DEFAULT = max(1, min(16, _env_int("DFL_FUSED_PREP", 4)))
_FUSED_CMP_DEFAULT = max(1, min(32, _env_int("DFL_FUSED_CMP", 8)))


# ---------------- 异构推理池：n×GPU + m×CPU（探测 / 微基准 / 配比 / 调度）----------------
# 设计在 server/_sess.py（探测、基准、derive、策略、Pool）；这里只做三件事：
#   · 把「这台机器的事实」喂给它（显存那本账已经算好了，别自己再猜一次）
#   · 把微基准的结果**缓存到磁盘**（同一份模型只量一次，跨进程复用）
#   · 把推出来的配比写进日志/status（用户问「为什么是这个配比」要能答）
#
# 为什么默认只给**检测**段开池：检测是这条链的硬底（实测 48.9 ms/帧、96~98% 占用），
# 只有瓶颈段多开一路才有净收益；dfm 70% / xseg 13% / 合成 21% 加一路只会抢 CPU
# （数据见 docs/sess_pool_bench.md）。要开也只是一个环境变量的事。
#
# ★ 2026-11 修正：那 48.9ms 的构成**不是**「resize/NMS 占大头」。分三段实测（1080p 零图）：
# letterbox 0.7~0.9ms · 前向 12.0ms（retinaface_10g）/ 1.8ms（tinymog）· NMS+解码 0.1ms；
# 剩下的三十几毫秒是**前向在等 GPU**（dfm+xseg 同时在用同一块卡）。这也是「CPU session」
# 净增 −2.4%~−8% 的原因：它缩短不了 GPU 上的那一段，只是把同一帧的 CPU 预处理再做一遍。
# 真要提速只有两条：换更轻的检测器（tinymog，实测整条线 +20~24%）或让检测器少排队。
# 另有 DFL_FUSED_DET_PREP=0 可把 letterbox 退回检测线程（默认在解码线程做）。
from . import _sess        # 异构池（探测 / 微基准 / 配比 / 调度），见 server/_sess.py

_SESS_LOCK = threading.Lock()
_SESS_BUSY = {}          # kind -> 上一轮实测的段占用（0..1）：derive 的输入
_SESS_BENCH = {}         # key -> {"gpu_ms","cpu_ms",...}（进程内缓存）
_SESS_PLAN_LAST = {}     # kind -> plan（status 回显）
_DET_EP = {}             # (name, ep) -> 检测器实例（GPU 那份走 app 的缓存）


def _sess_bench_path() -> Path:
    return Path(WORKSPACE) / "_sess_bench.json"


def _sess_bench_load() -> dict:
    try:
        p = _sess_bench_path()
        if p.is_file():
            d = json.loads(p.read_text("utf-8", "ignore") or "{}")
            if isinstance(d, dict):
                return d
    except Exception:                                       # noqa: BLE001
        pass
    return {}


def _sess_bench_put(key: str, row: dict):
    with _SESS_LOCK:
        _SESS_BENCH[str(key)] = dict(row)
    try:
        d = _sess_bench_load()
        d[str(key)] = dict(row)
        _sess_bench_path().write_text(json.dumps(d, ensure_ascii=False, indent=1), "utf-8")
    except Exception:                                       # noqa: BLE001
        pass


def _sess_bench_get(key: str) -> dict:
    with _SESS_LOCK:
        hit = _SESS_BENCH.get(str(key))
    if hit:
        return dict(hit)
    row = _sess_bench_load().get(str(key)) or {}
    return dict(row) if isinstance(row, dict) else {}


def _sess_model_key(kind: str, path) -> str:
    """模型身份 = 类型 + 文件名 + 体积 + mtime。换权重 / 重编引擎都会让它失效。"""
    try:
        st = Path(str(path)).stat()
        return "%s:%s:%d:%d" % (kind, Path(str(path)).name, int(st.st_size),
                                int(st.st_mtime))
    except Exception:                                       # noqa: BLE001
        return "%s:%s" % (kind, Path(str(path)).name)


def _sess_ensure(key: str) -> dict:
    """读一遍磁盘缓存（跨进程复用，同一份模型只量一次）。"""
    row = _sess_bench_get(key)
    if row.get("gpu_ms") is not None or row.get("cpu_ms") is not None:
        with _SESS_LOCK:
            _SESS_BENCH[key] = row
    return row


def _detect_for(name, ep="gpu"):
    """按 EP 取一个检测器。GPU 那份复用 app 的缓存（别建第二份显存）；CPU 那份单独建。"""
    key = (str(name), str(ep))
    with _SESS_LOCK:
        hit = _DET_EP.get(key)
    if hit is not None:
        return hit
    det = None
    if ep == "gpu":
        det = _detector(name)            # 复用 app 的缓存（别为同一份权重占第二块显存）
    elif ep == "gpu2":
        try:                             # 第二份 CUDA session：显存够时才能多开一路
            from .app import _DET_ONNX, _detector_path
            from ._ortcuda import providers as _prov
            canon = _detector_val(name)
            p = _detector_path(_DET_ONNX[canon])
            if p is None:
                return None
            from ._detect import RetinaFaceDet, TinyMogDet
            if canon == "tinymog":
                det = TinyMogDet(str(p), workspace=WORKSPACE, providers=list(_prov()))
            else:
                det = RetinaFaceDet(p, 640)          # 自带 CUDA providers
        except Exception as e:                              # noqa: BLE001
            _stage_log("异构池：第二份 GPU 检测器起不来（%s）" % str(e)[:160])
            return None
    else:
        # CPU 那一路（DFL_DET_EP=cpu|auto 选 CPU 的那档，见 _det_ep_resolve）。
        # 现在的做法是**直接按 CPU EP 造**，两个类都开了会话通道（providers / sess_options）：
        # 以前是「先照原样造好（会顺手加载 TRT 引擎、建一块 CUDA context）、再把 sess 换掉」，
        # 给 CPU 用的检测器白占 5MB 引擎 + 一份 context，而且日志里会冒出
        # 「[TRT] tinymog: 使用 bf16 引擎」这种把人带偏的行。
        # intra_op 线程数：DFL_DET_EP_THREADS（默认 3，夹在 2..4）；老的 DFL_SESS_THREADS
        # 仍然优先（那是池级口径，显式设了就听它的）。
        try:
            from .app import _DET_ONNX, _detector_path
            from . import _detect as _dk
            canon = _detector_val(name)
            p = _detector_path(_DET_ONNX[canon])
            if p is None:
                return None
            nthr = _sess.int_env(_sess.ENV_THREADS, 0) or _dk.threads()
            so = _dk.cpu_session_options(nthr)
            from ._detect import RetinaFaceDet, TinyMogDet
            if canon == "tinymog":
                det = TinyMogDet(str(p), workspace=None, providers=["CPUExecutionProvider"],
                                 sess_options=so)
            else:
                det = RetinaFaceDet(p, 640, providers=["CPUExecutionProvider"],
                                    sess_options=so)
            _stage_log("检测：CPU EP · intra_op=%d · inter_op=1 · %s"
                       % (int(nthr), Path(str(p)).name))
        except Exception as e:                              # noqa: BLE001
            _stage_log("异构池：CPU 检测器起不来（%s）" % str(e)[:160])
            return None
    with _SESS_LOCK:
        _DET_EP[key] = det
    return det


def _detect_on(det, frame, thresh: float) -> list:
    """用**指定**的检测器跑一帧（不碰共享实例上的 thresh 字段，可并发）。"""
    det.thresh = float(thresh)
    return det.detect(frame) or []


def _det_prep(det, frame):
    """把检测器的 letterbox 先算出来（decode 线程干，检测线程只留前向 + NMS）。

    检测器没有 prep（老/第三方实现）就返回 None，调用方退回「帧直接丢给 detect」。
    prep 只读 self.input_size、不碰 session，所以可以和安全跑着的 detect 并发。
    """
    fn = getattr(det, "prep", None)
    if fn is None:
        return None
    return fn(frame)


def _detect_on_ctx(det, ctx, thresh: float) -> list:
    """已经 prep 好的上下文 → 人脸列表（自己的 session 用，不加锁）。"""
    det.thresh = float(thresh)
    return det.detect_ctx(ctx) or []


def _detect_boxes_ctx(ctx, detector: str, thresh: float) -> list:
    """共享实例版：和 _detect_boxes 同一把锁、同一个 thresh 口径，只是跳过 resize。"""
    det = _detector(detector)
    with _det_run_lock:
        det.thresh = float(thresh)
        try:
            return det.detect_ctx(ctx) or []
        except Exception as e:                            # noqa: BLE001
            raise _BadInput("检测失败：%s" % (str(e)[:200],), 500)


def _det_quick_profile(det, w: int, h: int, n: int = 3) -> dict:
    """单帧检测的三段实测（1080p 真尺寸零图，n 次取中位）—— 只用来打日志。

    为什么要在**起流水线之前**量这一段：用户报的是「界面选了 TinyMog，日志却 48ms/帧」。
    有这张表就能一眼分开三件事：模型本身多贵（前向）、resize 多贵（预处理）、NMS 多贵；
    剩下的差额就是「在等 GPU/CPU」——那属于整条流水线的账，不属于检测器。
    """
    import time as _t

    import numpy as np

    frame = np.zeros((max(2, int(h or 1080)), max(2, int(w or 1920)), 3), np.uint8)
    out = {"w": int(frame.shape[1]), "h": int(frame.shape[0]), "n": int(max(1, n)),
           "sess": type(getattr(det, "sess", None)).__name__}
    full, prep, fwd, post = [], [], [], []
    try:
        # 先空跑一张：第一次前向可能包含 CUDA Graph 捕获 / torch 张量路径初始化
        # （实测 tinymog 第一张 76ms、之后 2ms）—— 把那次算进「每帧耗时」就是纯误导。
        det.detect(frame)
        for _ in range(max(1, int(n))):
            t0 = _t.perf_counter()
            det.detect(frame)
            t1 = _t.perf_counter()
            full.append((t1 - t0) * 1000.0)
        prep_fn = getattr(det, "prep", None)
        ctx_fn = getattr(det, "detect_ctx", None)
        if prep_fn is not None and ctx_fn is not None:
            for _ in range(max(1, int(n))):
                t0 = _t.perf_counter()
                ctx = prep_fn(frame)
                t1 = _t.perf_counter()
                fw = det._forward(ctx["canvas"])
                t2 = _t.perf_counter()
                ctx_fn(ctx)
                t3 = _t.perf_counter()
                prep.append((t1 - t0) * 1000.0)
                fwd.append((t2 - t1) * 1000.0)
                # 注意：detect_ctx 内部**又跑了一次**前向，所以 post 这一栏 = 前向 + NMS/解码；
                # 报出去时减掉前向的中位数，剩下的才是 NMS/解码本身。
                post.append((t3 - t2) * 1000.0)
        med = lambda v: (sorted(v)[len(v) // 2] if v else None)          # noqa: E731
        out["full_ms"] = None if not full else round(med(full), 2)
        out["prep_ms"] = None if not prep else round(med(prep), 2)
        out["fwd_ms"] = None if not fwd else round(med(fwd), 2)
        out["post_ms"] = (None if not post or not fwd
                          else round(max(0.0, med(post) - med(fwd)), 2))
    except Exception as e:                                            # noqa: BLE001
        out["err"] = str(e)[:160]
    return out


def _detect_bench(name, w: int, h: int) -> dict:
    """检测段的微基准：GPU / CPU 各 2 张取中位（1080p 真尺寸的零图）。

    为什么用 1080p 而不是 640 的方块：这一步里有一大块是 **cv2 的 letterbox resize**
    （1080p → 640），那部分不管在哪个 EP 上都吃 CPU —— 只量模型会把 CPU 那一路
    算得偏快，配比就偏激进。
    """
    import numpy as np
    key = "detect:%s:%dx%d:%dt" % (_detector_val(name), int(w or 1080), int(h or 1920),
                                   _dv_threads())
    row = _sess_ensure(key)
    if row.get("gpu_ms") and row.get("cpu_ms") and row.get("gpu_fwd_ms") is not None:
        return row
    frame = np.zeros((int(h or 1920), int(w or 1080), 3), np.uint8)
    out = {"key": key, "at": round(time.time(), 1)}
    for ep in ("gpu", "cpu"):
        det = None
        try:
            det = _detect_for(name, ep)
        except Exception as e:                              # noqa: BLE001
            out["%s_err" % ep] = str(e)[:160]
        if det is None:
            out["%s_ms" % ep] = None
            out["%s_fwd_ms" % ep] = None
            continue
        try:
            ms = _sess.bench(lambda: _detect_on(det, frame, 0.5), n=2, warmup=1)
        except Exception as e:                              # noqa: BLE001
            out["%s_err" % ep] = str(e)[:160]
            ms = None
        out["%s_ms" % ep] = ms
        # 前向单独量：DFL_DET_EP=auto 的判据就是这个数（见 server/_detect.py 的 decide）。
        # 为什么不用 gpu_ms（整帧）：那里面有 letterbox(0.4~0.9ms)+NMS/解码(0.1ms)，
        # 这部分两个 EP 完全一样，混进去只会把「模型本身多贵」稀释掉。
        # warmup=1 必须留着：TRT 的第一张要捕获 CUDA Graph（16ms），算进去就是假数。
        try:
            ctx = det.prep(frame) if hasattr(det, "prep") else None
            out["%s_fwd_ms" % ep] = (None if ctx is None else
                                     _sess.bench(lambda: det._forward(ctx["canvas"]),
                                                 n=3, warmup=1))
        except Exception as e:                              # noqa: BLE001
            out["%s_err" % ep] = str(e)[:160]
            out["%s_fwd_ms" % ep] = None
    _sess_bench_put(key, out)
    return out


def _dv_threads() -> int:
    """CPU 检测会话的 intra_op 线程数（缓存键带上它：线程数一改，cpu_ms 就不是同一个数）。"""
    try:
        from . import _detect as _dk
        nthr = _sess.int_env(_sess.ENV_THREADS, 0) or _dk.threads()
        return int(nthr)
    except Exception:                                       # noqa: BLE001
        return 0


def _det_ep_resolve(det_name, w=None, h=None) -> dict:
    """这一单检测走哪个 EP → {"ep","why","gpu_fwd_ms","cpu_fwd_ms","thr","src","mode","bench"}。

    判据与阈值在 server/_detect.py 的 decide()（DFL_DET_EP=auto 时用实测的前向 ms，
    量不到就查已知表）。这里只负责「把实测喂进去 + 把结论摆成字典」，不自己拍数。
    """
    b = _detect_bench(det_name, w or 1920, h or 1080)
    from . import _detect as _dk
    pol = _dk.decide(det_name, b.get("gpu_fwd_ms"),
                     gain_ok=_sess.cpu_ep_ok("detect", WORKSPACE),
                     gain_pct=_sess.cpu_ep_gain("detect", WORKSPACE))
    pol["bench"] = b
    pol["cpu_fwd_ms"] = b.get("cpu_fwd_ms")
    pol["name"] = _detector_val(det_name)
    return pol


def _det_ep_line(pol) -> str:
    """一行日志：选了哪个 EP + 为什么 + 两边的实测数字。"""
    if not pol:
        return "-"
    return ("检测 EP：%s · %s · GPU 前向 %s ms / CPU 前向 %s ms（整帧 GPU %s / CPU %s ms）"
            % (str(pol.get("ep") or "?").upper(), "；".join(pol.get("why") or []),
               (("%.2f" % pol["gpu_fwd_ms"]) if pol.get("gpu_fwd_ms") is not None else "未量到"),
               (("%.2f" % pol["cpu_fwd_ms"]) if pol.get("cpu_fwd_ms") is not None else "未量到"),
               (pol.get("bench") or {}).get("gpu_ms"),
               (pol.get("bench") or {}).get("cpu_ms")))


def _det_ep_apply(plan: dict, pol: dict, over=None) -> dict:
    """把「检测走哪个 EP」落进配比。

    · ep=cpu：这个检测器**整段**挪出 GPU 流 → 池里就只留 CPU 那一路（gpu=0, cpu=1）。
      为什么不是「两边各一份、谁快谁上」：检测是逐帧串行的（一张脸一帧），
      两路只会让同一帧的 letterbox/NMS 做两遍，而 GPU 那一路正是要躲开的东西。
    · ep=gpu：不动 GPU 份数（derive 已经按显存/边际算过了），但**清零 CPU 份数** ——
      「政策说要 GPU」时再挂一路 CPU 检测器就是把同一帧分回 GPU 队列之外，
      端到端反而多一份排队（这就是这次实验 A 档的口径）。
      请求体显式给了 sess_cpu（比如做实验）就尊重它，不清零。
    """
    why = plan.setdefault("why", [])
    ep = str((pol or {}).get("ep") or "gpu")
    plan["det_ep"] = dict(pol or {})
    if ep == "cpu":
        if int(plan.get("gpu") or 0) != 0 or int(plan.get("cpu") or 0) != 1:
            why.append("检测 EP=CPU → 池改成 0×GPU + 1×CPU（原 %d×GPU + %d×CPU）"
                       % (int(plan.get("gpu") or 0), int(plan.get("cpu") or 0)))
        plan["gpu"] = 0
        plan["cpu"] = 1
    else:
        if int(plan.get("gpu") or 0) < 1:
            why.append("检测 EP=GPU → 至少 1 份 GPU 检测会话")
            plan["gpu"] = 1
        exp_cpu = (over or {}).get("cpu")
        if not (isinstance(exp_cpu, str) and exp_cpu.strip()) and exp_cpu not in (0,) \
                and int(plan.get("cpu") or 0) > 0:
            why.append("检测 EP=GPU → CPU 那几路（%d 份）清零，别让同一帧又排到 CPU"
                       % int(plan.get("cpu") or 0))
            plan["cpu"] = 0
    return plan


def _fused_det_plan(det_name, w, h, over=None) -> dict:
    """检测池的配比：探测 → 微基准 → derive → EP 政策。over = 请求体覆盖（sess_gpu/sess_cpu/…）。"""
    b = _detect_bench(det_name, w, h)
    pol = _det_ep_resolve(det_name, w, h)
    gp = _sess.probe(*(_vram_free_mb()[:2]))
    # 验收/复现用：把「机器档位」换成低配/中配/高配的**事实**（不改真探测，
    # 只影响这一次推导 —— 生产路径上没人设 DFL_SESS_LEVEL，一行都不走）。
    lv = str((over or {}).get("level") or os.environ.get("DFL_SESS_LEVEL")
             or "").strip().lower()
    if lv:
        gp.update(_sess.level_cfg(lv))
        gp["level"] = lv
    # 检测器很小（retinaface 22 MB / tinymog 5 MB），一份 session 的显存 ≈ 权重 + 300MB
    # workspace/arena（日志里那个「检测/遮罩 300 MB/份」就是它）。
    mem_gpu = float(_vram_model_mb(det_name if str(det_name).endswith(".onnx")
                                   else "") or 0.0) + 300.0
    plan = _sess.derive("detect", gp, b.get("gpu_ms"), b.get("cpu_ms"),
                        mem_per_gpu_mb=mem_gpu, mem_per_cpu_mb=260.0,
                        stage_busy=_SESS_BUSY.get("detect"), over=over)
    plan["bench"] = b
    _det_ep_apply(plan, pol, over)
    return plan


def _fused_on(body) -> bool:
    """这一单走不走融合流水线。**默认走**（fused=1）；fused=0 回退六阶段分段链。"""
    if not isinstance(body, dict):
        return True
    return _bool_val(body.get("fused"), True)


def _fused_vram_free():
    """(free, total) MB。DFL_FUSED_VRAM_FREE_MB 可覆盖 free —— 只给验收/排障用。"""
    fake = str(os.environ.get(_FUSED_ENV_VRAM) or "").strip()
    if fake:
        try:
            return int(float(fake)), None
        except (TypeError, ValueError):
            pass
    return _vram_free_mb()


def _fused_frame_mb(size_px: int) -> float:
    """一帧在飞时压在 GPU 上大约多少 MB：dfm 输入 + 输出脸 + prd + dst 四份 float32。

    计算在 _vram_frame_mb（三条路共用），这里只留名字给历史调用点。
    """
    return _vram_frame_mb(size_px)


def _fused_budget(want: int, free_mb, total_mb, size_px: int, model_mb: float = 0.0,
                  batch: int = 0, label: str = "快速导出",
                  resident: bool = False, force: bool = False,
                  reclaim: bool = False, reclaim_first: bool = False) -> dict:
    """fused 的帧预算 —— 现在就是 _vram_plan 的一层壳（老调用方读的键一个都不改）。

    和以前那一版（只按 30/16/8 降帧）的区别只有两点，但都是这次要治的病：
      · 把**模型体量**和 **dfm 批的开销**算进总需求（以前只算帧，模型 2GB 也当没看见）；
      · 连「最小档（8 帧 / 批 1）」都装不下时返回 ok=False —— 以前是压到 8 帧照跑，
        让 OOM 在流水线里炸；现在由调用方明确拒绝（或按 DFL_ALLOW_CPU_FALLBACK 走 CPU）。

    free_mb 传 None = 「按 _vram_probe 自己探」（env 伪造值优先）；调用方已经探过了就传进来。

    resident=True：这份会话**已经在当前进程里加载好了**（_fused_run 加载完 dfm 之后
    拿实测尺寸再判一次那一拍）。那一拍里会话的「权重 + context」已经在卡上，再算一遍
    就是重复计数 —— 384 模型实测「文件 1567MB / 建会话后整卡 +2.3GB」，重复计数会让
    一个本来跑得动的单子被判死（用户截图那一单就是这么被拒的）。
    """
    opts = {"mode": "frames", "size_px": int(size_px or _VRAM_BASE_PX),
            "frames": int(want or 0), "batch": int(batch or 0), "procs": 1,
            "label": label, "resident": bool(resident), "force": bool(force),
            # reclaim_first：先卸掉常驻预览 worker 的模型，再用**回收后**的空闲算账
            # —— 这就是「帧预算不该因为自己的预览 worker 占着而降到 8」那一条。
            "reclaim": bool(reclaim), "reclaim_first": bool(reclaim_first)}
    if free_mb is not None:
        opts["free_mb"] = int(free_mb)
        if total_mb is not None:
            opts["total_mb"] = int(total_mb)
    return _vram_plan(float(model_mb or 0.0), opts)


def _fused_cfg(body: dict) -> dict:
    """请求体 → 快速导出的 cfg。

    和 _stage_cfg 的差别只有一处：**不碰 <视频>_cache/** —— 不建目录、不读不写 meta.json。
    快速导出没有中间产物，也就没有阶段标记/续跑游标，meta 只在内存里当个空壳。
    """
    body = _as_dict(body)
    video = _norm_path(body.get("video") or body.get("source_video"))
    if video is None:
        raise _BadInput("缺少 video：要一个视频文件路径")
    if not video.is_file():
        raise _BadInput("视频不存在：%s" % video, 404)
    if video.suffix.lower() not in _VIDEO_EXT:
        raise _BadInput("不是视频文件：%s（只认 %s）"
                        % (video.suffix or "无扩展名", " ".join(sorted(_VIDEO_EXT))))

    probe = _vid_probe(video)
    rng = _stage_range_norm(body, int(probe.get("frames") or 0))
    settings = _settings_val(body.get("settings"))
    fps = _float_val("fps", body.get("fps"), 1.0, 240.0, None)
    dfm, model_name = _stage_dfm_pair(body, need=True)
    xseg = _xseg_of(body)
    out = _stage_out_path(video, rng, body)
    if _same_path(out, video):
        raise _BadInput("输出不能是源视频本身：%s" % out)

    snap = dict(_cache_snap())
    snap.update({"mode": "video", "path": str(video), "dir": "", "names": [], "srcdir": ""})
    library_raw = str(body.get("faces_project") or body.get("face_library")
                      or body.get("library") or "").strip()
    plan_payload = {"project": library_raw, "faces_project": library_raw}
    face_thr = _float_val("face_threshold", body.get("face_threshold"), 0.0, 1.0, None)
    if face_thr is not None:
        plan_payload["face_threshold"] = float(face_thr)

    models = []
    if body.get("models"):
        models = _models_of(body) or []
    if not models and dfm:
        models = [{"name": str(model_name), "dfm": str(dfm), "faces": [], "index": 0}]
    if not models:
        raise _BadInput("没有可用的 .dfm 模型：先在训练页把模型导出成 .dfm（%s）"
                        % (Path(WORKSPACE) / "model"), 409)
    if not model_name:
        model_name = str(models[0].get("name") or "")

    want = body.get("frame_budget")
    if want is None:
        want = _env_int(_FUSED_ENV_BUDGET, 0)
    cfg = {
        "video": str(video), "out": str(out), "fps": float(fps or probe.get("fps") or 0.0),
        "probe": probe, "total": int(probe.get("frames") or 0),
        "range": rng, "range_key": str(rng["key"]), "range_total": int(rng["total"]),
        "range_full": bool(rng["full"]), "settings": dict(settings),
        "detector": _detector_val(body.get("detector")),
        "det_align": _align_detector(body)[0],
        "det_align_src": _align_detector(body)[1],
        "thresh": _float_val("thresh", body.get("thresh"), 0.0, 1.0, 0.5),
        "align_thresh": _float_val("align_thresh", body.get("align_thresh"), 0.0, 1.0, 0.5),
        "dfm": str(dfm), "model": str(model_name), "xseg": str(xseg),
        "mask_mode": int((settings or {}).get("mask_mode", 6) or 0),
        "mask_src": _mask_src(settings, (settings or {}).get("mask_mode")),
        "config": dict(_merge_settings_to_config(settings)),
        "snap": snap, "plan_payload": plan_payload, "single_models": models,
        "use_library": True, "library": library_raw, "face_threshold": face_thr,
        "pred_batch": _int_val("batch", body.get("batch") or body.get("pred_batch"), 0, 64,
                               _PRED_BATCH_DEFAULT),
        # 合成线程默认 8（B 配置）；fused 有自己的默认，不再跟着 DFL_MERGE_WORKERS 走
        "merge_workers": _int_val(
            "merge_workers",
            body.get("merge_workers") if body.get("merge_workers") is not None
            else (body.get("threads") if body.get("threads") is not None
                  else body.get("merge_threads")),
            1, 32, _FUSED_CMP_DEFAULT),
        "merge_cv2_threads": _int_val("merge_cv2_threads", body.get("merge_cv2_threads"),
                                      0, 64, _MERGE_CV2_THREADS_DEFAULT),
        "png_level": _int_val("png_compression", body.get("png_compression"), 0, 9,
                              _MERGE_PNG_LEVEL),
        # 帧图 JPEG 质量：固定 100（和分段链同一条口径，见 _STAGE_JPEG_QUALITY）
        "jpeg_quality": _int_val("jpeg_quality", body.get("jpeg_quality"), 1, 100,
                                 _STAGE_JPEG_QUALITY),
        # 对齐/预处理线程默认 4（B 配置），0 也让它是 4 —— 以前 0 落到 cpu//3，核少就掉档
        "pred_readers": _int_val("pred_readers", body.get("pred_readers"), 0, 16,
                                 _FUSED_PREP_DEFAULT),
        "pred_writers": _int_val("pred_writers", body.get("pred_writers"), 0, 16, 0),
        # 攒批：目标张数 + 超时 ms（都可以从请求体/环境变量压，排障用）
        "fused_batch": _int_val("fused_batch", body.get("fused_batch"), 1,
                                int(_PRED_BATCH_CAP), _FUSED_BATCH_TARGET),
        "fused_batch_wait_ms": _int_val("fused_batch_wait_ms",
                                        body.get("fused_batch_wait_ms"), 0, 60000,
                                        _FUSED_BATCH_WAIT_MS),
        "frame_budget_want": int(want or 0),
        # 异构 session 池的覆盖（0/空 = 自动推导，见 _fused_det_plan / server/_sess.py）
        "sess_gpu": (None if body.get("sess_gpu") in (None, "")
                     else _int_val("sess_gpu", body.get("sess_gpu"), 0, 8, 0)),
        "sess_cpu": (None if body.get("sess_cpu") in (None, "")
                     else _int_val("sess_cpu", body.get("sess_cpu"), 0, 16, 0)),
        "sess_strategy": (str(body.get("sess_strategy") or "").strip().lower() or None),
        # 验收/复现用：把「机器档位」按 low / mid / high 伪造（只影响配比推导）
        "sess_level": (str(body.get("sess_level") or "").strip().lower() or None),
        # fused 专有：没有 cache / meta / 阶段游标
        "fused": True, "cache": "", "meta": {}, "stages": ["fused"],
        "chain": True, "force": True, "keep_cache": False, "range_changed": False,
    }
    return cfg


_PNG_SIG = b"\x89PNG\r\n\x1a\n"


# 解码路径：**默认 raw**（-f rawvideo，软解）；DFL_FUSED_DECODE=png 退回 PNG 管道。
_FUSED_ENV_DECODE = "DFL_FUSED_DECODE"
# 检测输入（letterbox canvas）在 decode 线程先算：DFL_FUSED_DET_PREP=0 关（回老口径）
_FUSED_ENV_DET_PREP = "DFL_FUSED_DET_PREP"
_FUSED_RAW_MODES = ("raw", "rawvideo", "bgr24", "bgr48le")
# -i 的流信息里挑「pix_fmt, 宽x高」这一对；位深只从 pix_fmt 名字看
_FUSED_PIX_TOKEN = re.compile(r"Video:\s[^\n]*?,\s*([a-z][a-z0-9]+)(?:\([^)\n]*\))?,\s*\d+x\d+")
_FUSED_WIDE_TOKEN = re.compile(r"(10|12|16|48|64)(?:le|be)?$")


def _fused_src_bits(video: Path) -> int:
    """源视频**解码后**的位深（8 / 10 / 12 / 16）—— 决定 raw 管道进 bgr24 还是 bgr48le。

    为什么要问这一句：切帧阶段落盘的是 PNG，而 ffmpeg 的 png 编码器对 >8bit 的源会挑
    16bit（rgb48be）。所以「切帧产物」那一路是 16bit PNG 被 cv2 砍成 8bit（libpng 的
    chop = 取高字节，等价 >>8），这和 swscale 的 10→8 正确舍入**不是同一个数**：
    HEVC 10bit 实测 bgr24 vs frames/*.png mean 0.6038 / max 4 / 只有 5.16% 像素相同。
    改成 bgr48le 进来再 >>8 之后逐位相同（30 帧 mean 0.0000 / max 0 / 100%）。
    8bit 源仍然走 bgr24（多要一倍管道带宽没意义）。

    判不出来就当 8bit（= 原来的 bgr24 行为，一行都不变）。解析一次 -i 的流信息约 50ms，
    一次导出只调一次。
    """
    try:
        r = subprocess.run([str(_app_mod._ffmpeg_exe()), "-hide_banner", "-i", str(video)],
                           capture_output=True,
                           creationflags=getattr(_app_mod, "_NO_WINDOW", 0))
        txt = (r.stderr or b"").decode("utf-8", "ignore")
    except Exception:                                       # noqa: BLE001
        return 8
    m = _FUSED_PIX_TOKEN.search(txt)
    if not m:
        return 8
    # yuv420p10le / gbrp12le / gray16le → 10 / 12 / 16；yuv420p / rgb24 / rgba → 8
    d = _FUSED_WIDE_TOKEN.search(m.group(1) or "")
    if not d:
        return 8
    return 16 if d.group(1) in ("48", "64") else int(d.group(1))


def _fused_open_frames(video: Path, start_i: int, size=None, mode="raw", bits=None):
    """开一路「ffmpeg → rawvideo 帧流」（默认）或「ffmpeg → PNG 帧流」，返回 (proc, 读一帧的函数)。

    默认 raw：`-f rawvideo -pix_fmt bgr24`（8bit 源）/`bgr48le -> >>8`（>8bit 源），
    软解。产物和切帧阶段落盘的 frames/%06d.png **逐位相同**（30 帧 mean 0.0000 / max 0 /
    100% 像素相同），每帧 25.2 → 8.9 ms。回归见 app/tools/accept/fused_decode_parity.py。

    DFL_FUSED_DECODE=png 保留 PNG 管道（ffmpeg 解 → 无损 PNG → 管道 → cv2.imdecode）：
    排障时用它，或者要「同一条 PNG 编解码路径」做对照时用它（约 25ms/帧）。

    **注意别用 -hwaccel cuda**：NVDEC 那条路把帧下载成 nv12 再转 BGR，色度是另一套
    插值，实测 mean 0.8895 / max 12（软解/NVDEC 的 YUV 平面本身逐位相同，差全在
    nv12→RGB 这一步）。**这条 0.89 与本改动无关**，容易误判成 raw 的锅。所以 raw 一律软解。

    >8bit 为什么是 bgr48le：见 _fused_src_bits（PNG 那一路是 16bit 截高字节，不是舍入）。

    旋转元数据（手机竖拍）：两条路都走 ffmpeg 的 autorotate，且 probe 给的是**显示方向**
    的宽高（imageio 读的就是 540x960 这种），所以 reshape 对得上 —— 90° 素材实测 100%
    相同。**-noautorotate 别加**：那会让 raw 变成未旋转的原始帧，和 PNG 产物对不上。

    start_i > 1 时用 ffmpeg 的 select 滤镜跳帧（0 基：保留 n >= start-1），跳过的帧不
    编码/不转换，比"读出来再丢"省得多；帧号仍然是绝对的（N 帧 = frames/NNNNNN.png）。
    """
    import cv2
    import numpy as np

    # ---- 三条解码路径 ----
    #   raw  · 8bit ：ffmpeg 解 → rawvideo bgr24  → 管道 → 直接 reshape。9 ms/帧。
    #   raw  · >8bit：ffmpeg 解 → rawvideo bgr48le → 管道 → <u2 reshape >>8。约 12 ms/帧。
    #   png  · 任意 ：ffmpeg 解 → PNG(无损) → 管道 → cv2.imdecode。25 ms/帧（可选）。
    # 为什么 raw 不改像素：两条路都是「同一个软解 → 同一个 swscale → BGR」，PNG 只是无损
    # 容器（8bit 源 rgb24 / >8bit 源 rgb48be，后者 cv2 取高字节 = 这里的 >>8）。
    # 这条口径由 app/tools/accept/fused_decode_parity.py 对着 frames/%06d.png 逐帧逐像素盯着。
    raw = str(mode or "raw").strip().lower() in _FUSED_RAW_MODES
    rf = (0, 0)
    deep = False
    if raw:
        if not size or int(size[0] or 0) <= 0 or int(size[1] or 0) <= 0:
            raise _BadInput("raw 解码要源视频的宽高，但 probe 里没有", 422)
        rf = (int(size[0]), int(size[1]))
        if bits is None:
            bits = _fused_src_bits(video)
        deep = int(bits) > 8

    ffmpeg = str(_app_mod._ffmpeg_exe())
    cmd = [ffmpeg, "-y", "-hide_banner", "-loglevel", "error", "-nostats", "-nostdin",
           "-i", str(video), "-map", "0:v:0", "-an"]
    if int(start_i) > 1:
        cmd += ["-vf", "select=gte(n\\,%d)" % (int(start_i) - 1)]
    # -vsync 0：原样搬每一帧（不按时间戳丢/补），帧数 = 选中的帧数
    cmd += ["-vsync", "0"]
    if raw:
        cmd += ["-f", "rawvideo", "-pix_fmt", ("bgr48le" if deep else "bgr24"), "pipe:1"]
    else:
        cmd += ["-f", "image2pipe", "-vcodec", "png",
                "-compression_level", "1", "pipe:1"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                            creationflags=getattr(_app_mod, "_NO_WINDOW", 0))

    def _read_png(fh):
        """从管道里读一整张 PNG（按 chunk 读，8 字节签名 + 长度/类型/数据/CRC 到 IEND）。"""
        head = fh.read(8)
        if len(head) < 8 or head != _PNG_SIG:
            return None
        parts = [head]
        while True:
            ln = fh.read(4)
            if len(ln) < 4:
                return None
            typ = fh.read(4)
            if len(typ) < 4:
                return None
            size = int.from_bytes(ln, "big")
            body = fh.read(size) if size else b""
            crc = fh.read(4)
            if (size and len(body) < size) or len(crc) < 4:
                return None
            parts.append(ln)
            parts.append(typ)
            parts.append(body)
            parts.append(crc)
            if typ == b"IEND":
                return b"".join(parts)

    def _next_frame_raw():
        """rawvideo：定长读一帧 → (h,w,3) BGR uint8；流结束返回 None。

        >8bit 的源进的是 bgr48le（w*h*3*2 字节），这里 `>> 8` 取高字节 —— 和 16bit PNG
        被 libpng chop（= 取高字节）的结果逐位相同；用 swscale 的 10→8 舍入就对不上了。
        """
        bpp = 6 if deep else 3
        nb = rf[0] * rf[1] * bpp
        buf = proc.stdout.read(nb)
        if not buf or len(buf) < nb:
            return None
        if deep:
            return (np.frombuffer(buf, "<u2").reshape(rf[1], rf[0], 3) >> 8).astype(np.uint8)
        return np.frombuffer(buf, np.uint8).reshape(rf[1], rf[0], 3).copy()

    def _next_frame():
        """读下一帧 → BGR uint8；流结束返回 None。"""
        if raw:
            return _next_frame_raw()
        blob = _read_png(proc.stdout)
        if not blob:
            return None
        img = cv2.imdecode(np.frombuffer(blob, np.uint8), cv2.IMREAD_COLOR)
        return img

    return proc, _next_frame


class _FusedFrame:
    """一帧在流水线里的全部状态（帧对象从头带到合成，合成之后源帧就放掉）。"""

    __slots__ = ("idx", "n", "frame", "boxes", "faces", "timings", "plan", "rec",
                 "det_ctx")

    def __init__(self, idx, n, frame):
        self.idx = int(idx)          # 源视频里的绝对帧号（1 起）
        self.n = int(n)              # 本段里的序号（1..total，PNG 文件名用它）
        self.frame = frame
        # 检测输入（decode 线程预先算好的 letterbox canvas + scale）；None = 检测线程自己做
        self.det_ctx = None
        self.boxes = None
        self.faces = None
        self.timings = {}
        self.plan = None
        self.rec = {"frame": int(idx), "detected": 0, "faces": 0, "swapped": 0,
                    "no_face": False, "dfm": "", "branch": ""}


# ============================================================
# 流水线实况：交接点计数（导出进度页那条 SVG 图的**唯一**数据源）
# ------------------------------------------------------------
# 进度页从「六个圆点」换成「6 列 lane + 贝塞尔曲线 + 光点」之后，前端要三样东西，
# 少一样图就是假的：
#   ticks     这个交接点一共过了多少帧/张脸（只增）—— 光点按**增量**放，列标题显示「N 帧」
#   inflight  这个交接点**下游那个队列**的当前长度（q.qsize()）—— 列间那句「在飞 N」
#   lanes     这个阶段**真的有**几条 lane（线程数）：合成池 8 线程就是 t0..t7，
#             编码 6 线程就是 e0..e5，dfm/xseg/det 各一个 session 就是 1 条（带 #0）
# 开销：一次 dict 自增 + 一次 time.monotonic()，一把锁（临界区就两行）。tick 里不拼字符串、
# 不进 JSON、不碰队列；只有 status 那一拍（500ms 一次）才 snapshot，qsize() 是 O(1)。
# 为什么不用「每帧发一次请求」：那样 20 帧/秒就是 20 个请求，界面比片还重。
# ============================================================
_STAGE_KEYS = ("frames", "det", "dfm", "xseg", "compose", "encode")
# 每段的耗时取自 _FusedTicks 的同名 key（per-frame ms，和 <成片>.log 里那份同一个口径）
_STAGE_MS_SRC = {"frames": "decode", "det": "detect", "dfm": "dfm",
                 "xseg": "xseg", "compose": "compose", "encode": "encode"}


class _StageMeter:
    """交接点计数器：{key: ticks} + 每段的 lane 名 + 下游队列。

    一个进程里同时只会跑一单导出（起单处有互斥），所以这里是**单例**：
    reset() 开一单，stop() 收一单（收单后保留最后一份快照，界面上那张图不会突然塌掉）。
    """

    def __init__(self):
        self.lock = threading.Lock()
        self.ticks = {k: 0 for k in _STAGE_KEYS}
        self.lanes = {k: [] for k in _STAGE_KEYS}
        self.queues = {}
        self.ms = {}
        self.batch = {}
        self.armed = False          # 开过单没（没开过 → status 里干脆不出现 stages）
        self.on = False             # 现在还在跑没（收单后不再涨）

    def reset(self, lanes=None) -> None:
        with self.lock:
            self.ticks = {k: 0 for k in _STAGE_KEYS}
            self.lanes = {k: list((lanes or {}).get(k) or []) for k in _STAGE_KEYS}
            self.queues = {}
            self.ms = {}
            self.batch = {}
            self.armed = True
            self.on = True

    def bind(self, queues: dict) -> None:
        """key -> 下游那个 queue（inflight 就是它的 qsize()）。"""
        with self.lock:
            self.queues = dict(queues or {})

    def stop(self) -> None:
        """收单：不再涨，但保留 lanes/ticks 给界面画最后一帧（inflight 归零）。"""
        with self.lock:
            self.on = False
            self.queues = {}

    def begin(self) -> None:
        """一单开始（分段链 / 项目流水是**串行**跑的）：上一单收过单就清空重来。"""
        with self.lock:
            if not self.on:
                self.ticks = {k: 0 for k in _STAGE_KEYS}
                self.lanes = {k: [] for k in _STAGE_KEYS}
                self.queues = {}
                self.ms = {}
                self.batch = {}
            self.armed = True
            self.on = True

    def set_lanes(self, lanes: dict) -> None:
        """只补空的 lane：分段链一次只跑一段，lane 数每段才定，但不能把上一段的清掉
        （六段是**同一单**里串行跑的，tick 要一路累上去 —— 图上的「N 帧」才对得上）。"""
        with self.lock:
            for k, v in (lanes or {}).items():
                if v and not self.lanes.get(k):
                    self.lanes[k] = list(v)

    def tick(self, key: str, n: int = 1) -> None:
        """交接点 +n（n>1 只给「一次回调跨了好几帧」那种地方用，比如 ffmpeg 编码进度）。"""
        with self.lock:
            if not self.on:
                return
            self.ticks[key] = int(self.ticks.get(key) or 0) + max(1, int(n))

    def note_ms(self, ms: dict) -> None:
        with self.lock:
            for k, v in (ms or {}).items():
                if v is not None:
                    self.ms[k] = float(v)

    def note_batch(self, batch: dict) -> None:
        with self.lock:
            for k, v in (batch or {}).items():
                if v is not None:
                    self.batch[k] = int(v)

    def snapshot(self) -> dict:
        """status.stages 的那一块（没开过单就是空 dict，前端据此显示「等待流水线…」）。"""
        with self.lock:
            if not self.armed:
                return {}
            out = {}
            for k in _STAGE_KEYS:
                qq = self.queues.get(k)
                inflight = 0
                if qq is not None:
                    try:
                        inflight = int(qq.qsize())
                    except Exception:                          # noqa: BLE001
                        inflight = 0
                row = {"ticks": int(self.ticks.get(k) or 0),
                       "inflight": inflight,
                       "lanes": list(self.lanes.get(k) or [])}
                if k in self.ms:
                    row["ms"] = round(float(self.ms[k]), 2)
                if k in self.batch:
                    row["batch"] = int(self.batch[k])
                out[k] = row
            return out


_STAGE_METER = _StageMeter()


def _stage_tick(key: str, n: int = 1) -> None:
    """交接点 +1 —— 在**入队/交给下一阶段之前**调用（六个 key 见 _STAGE_KEYS）。

    故意做成模块级函数：埋点处只写一行，读的人一眼知道「这里是交接点」。
    """
    _STAGE_METER.tick(key, n)


class _FusedTicks:
    """逐段累计耗时（和原型同一套口径：每段的总忙时间 / 帧数 = 每帧 ms）。"""

    def __init__(self):
        self.d = {}
        self.lock = threading.Lock()

    def add(self, k, dt):
        with self.lock:
            self.d[k] = float(self.d.get(k) or 0.0) + float(dt)

    def raw(self, k):
        with self.lock:
            return float(self.d.get(k) or 0.0)

    def dump(self, n):
        with self.lock:
            return {k: round(1000.0 * v / max(1, int(n)), 2) for k, v in self.d.items()}


def _fused_cancelled() -> bool:
    """取消：/export/cancel 和 /stage/cancel 都算（两条老取消路都能停掉快速导出）。"""
    if _vid_cancelled():
        return True
    with _STAGE_LOCK:
        return bool(_STAGE_JOB.get("cancelled"))


def _fused_dump(cfg: dict, job, composed) -> None:
    """DFL_FUSED_DUMP 点名的帧：把「合成前的每一份输入」落盘，和分段 cache/ 逐位比。

    只给排障/验收用（默认一个文件都不写）。
    """
    import cv2
    import numpy as np

    try:
        root = Path(str(cfg["out"])).with_suffix("")
        root = Path(str(root) + ".dump")
        root.mkdir(parents=True, exist_ok=True)
        tag = "%06d" % int(job.n)
        if composed is not None:
            cv2.imwrite(str(root / (tag + "_composed.png")), composed)
        for f in (job.faces or []):
            sub = tag + "_%02d" % int(f.get("k"))
            if f.get("pred_u8") is not None:
                cv2.imwrite(str(root / (sub + "_pred.png")), f["pred_u8"])
            for key, name in (("prd_u8", "prdm"), ("dstdfm_u8", "dstdm"),
                              ("xseg_prd_u8", "xseg_prd"), ("xseg_dst_u8", "xseg_dst")):
                arr = f.get(key)
                if arr is not None:
                    cv2.imwrite(str(root / (sub + "_" + name + ".png")), arr)
            if f.get("dst_u8") is not None:
                cv2.imwrite(str(root / (sub + "_dstin.png")), f["dst_u8"])
            (root / (sub + "_lm.json")).write_text(
                json.dumps(np.asarray(f["lm"]).tolist()), "utf-8")
            (root / (sub + "_dfm.txt")).write_text(
                "%s\n%s\n" % (Path(str(f.get("dfm") or "")).name, f.get("name") or ""),
                "utf-8")
    except Exception as e:                                     # noqa: BLE001
        _vid_log("快速导出：dump 失败（%s）" % str(e)[:160])


def _src_of(cfg: dict, keys, default, env_name: str) -> str:
    """这个数是从哪来的：环境变量 / 请求体 / 默认。**只用来打日志**（不改任何取值）。

    线程数这种东西一旦「日志写 8、实际跑 4」，后面所有排障都是错的 —— 所以每次导出都
    把「实际生效值 + 来源」写进 <成片>.log，改过 env 的人自己能看到。
    """
    env = str(os.environ.get(env_name) or "").strip()
    val = None
    for k in keys:
        v = cfg.get(k)
        if v is not None and not (isinstance(v, str) and not str(v).strip()):
            val = v
            break
    cur = default if val is None else val
    if env and str(env) == str(cur):
        return "env %s=%s" % (env_name, cur)
    if val is not None and str(val) != str(default):
        return "请求体 %s=%s" % (keys[0], cur)
    return "默认 %s" % cur


def _fused_run(cfg: dict) -> dict:
    """融合流水线：解码 → 检测 → 对齐/归组 → dfm → XSeg → 合成 → 编码。

    段间全是**有界**队列（depth=24）+ 一个**全局帧缓冲上限**（frame_budget 帧，
    1080p 一帧 6.22MB）；谁忙都不空转，谁也堆不起来。允许乱序完成：编码好的 PNG 字节
    进重排缓冲，写线程只按帧号顺序落盘；到写盘那一步才真正出现文件（问题 8 的断言：
    这些文件在 %TEMP% 里，**不是** <视频>_cache/）。
    """
    import queue
    import cv2
    import numpy as np
    from ._merge_compose import compose_frame, ensure_engine_path
    ensure_engine_path()
    from facelib import FaceType
    from facelib.LandmarksProcessor import get_transform_mat
    from ._align import (COARSE_SIZE, align_face_68, coarse_align_from_5,
                         landmarks68_in_source)

    video = Path(cfg["video"])
    out = Path(cfg["out"])
    mem0 = _proc_mem_mb()          # 流水线起点的本进程内存（收尾要和它比）
    rng = _stage_range(cfg)
    start_i, end_i, total = int(rng["start"]), int(rng["end"]), int(rng["total"])
    probe = cfg.get("probe") or {}
    fps = float(cfg.get("fps") or probe.get("fps") or 30.0)
    full = bool(rng["full"])
    mask_src = _mask_src(cfg.get("settings"), cfg.get("mask_mode"))
    comp_cfg = dict(cfg.get("config") or {})
    comp_cfg["mask_mode"] = int((cfg.get("settings") or {}).get("mask_mode", 6) or 0)
    need_xseg = str(mask_src.get("source") or "").lower() == "xseg"
    need_dst = bool(need_xseg and int(comp_cfg["mask_mode"]) in (7, 8, 9))
    det_name = str(cfg.get("det_align") or "retinaface_10g")
    det_src = str(cfg.get("det_align_src") or "cfg.det_align")
    det_thresh = float(cfg.get("align_thresh") or 0.5)
    # 检测的 letterbox 在 decode 线程先做（检测线程只留前向 + NMS）。关掉 = 回到老口径，
    # 两条路逐位一致（同一个 prep()，见 app/tools/accept/det_prep_parity.py），留这个开关
    # 只是为了 A/B 与排障时能量到两边的账。
    det_prep_on = str(os.environ.get(_FUSED_ENV_DET_PREP) or "1").strip().lower() \
        not in ("0", "off", "false", "no")
    png_level = int(cfg.get("png_level") if cfg.get("png_level") is not None else 1)
    n_cmp = max(1, int(_merge_worker_count(cfg)))
    # PNG 编码是这一条链上最贵的纯 CPU 活（1080p 实测 94ms/帧，cv2.imencode 会放 GIL，
    # 多线程是真并行：2 线程 ≈2 倍、4 线程 ≈3.6 倍）。默认 3：detect（52.8ms/帧）才是硬底，
    # 再往上加只会跟检测抢 CPU。
    n_enc = max(1, int(cfg.get("pred_writers") or 3))
    # 对齐/预处理线程：**默认 4**（B 配置）。以前是 min(4, cpu//3) —— 16 核机是 4、
    # 8 核机是 2，同一份代码在两台机器上跑出两种口径；现在钉成 _FUSED_PREP_DEFAULT。
    n_prep = max(1, int(cfg.get("pred_readers") or _FUSED_PREP_DEFAULT))
    # 攒批：目标张数 + 超时（秒）。实际目标 = max(阶梯值, 这个值)，但阶梯实测如果证明
    # 目标档每张更贵就退回阶梯那档 —— 见 _target_for 与 _FUSED_BATCH_TARGET 的注释。
    batch_target = max(1, min(int(_PRED_BATCH_CAP),
                              int(cfg.get("fused_batch") or _FUSED_BATCH_TARGET)))
    _wait_ms = cfg.get("fused_batch_wait_ms")
    if _wait_ms is None:
        _wait_ms = _FUSED_BATCH_WAIT_MS
    batch_wait = max(0.0, float(_wait_ms) / 1000.0)
    qd = int(_FUSED_QDEPTH)
    # cv2 的内部线程数（进程级）：合成阶段的口径是「worker 多了就别让 cv2 自己摊线程」，
    # 但这条链的**硬底是检测**（retinaface_10g 整段 52.8ms/帧）。合成那 8 个 worker 实测
    # 有 76% 是空等（见 profile 的 cmp_wait），所以这里反过来：给 cv2 留几条线程，让检测的
    # 预处理能并行起来。实测把它钉成 1 时 detect 从 52.8 涨到 65.8 ms/帧、整条线 12.1 帧/秒。
    # ★ 这 52.8ms 里 letterbox 只占 0.7~0.9ms（见 _det_quick_profile 的实测），大头是前向
    #   在等 GPU。cv2 线程数仍然值得留，但别指望靠它把检测段砍下来。
    cv2_threads = max(2, min(8, int(os.cpu_count() or 8) // max(1, n_cmp)))

    cache_dir = Path(str(video.parent / (video.stem + "_cache")))
    cache_pre = cache_dir.is_dir()
    dump_frames = set()
    for _tok in str(os.environ.get(_FUSED_ENV_DUMP) or "").replace(";", ",").split(","):
        _tok = _tok.strip()
        if _tok.isdigit():
            dump_frames.add(int(_tok))

    # ---- 模型 / 显存 / 帧预算（起流水线之前，出问题要在这里就喊出来）----
    # 顺序要紧：**先算预算，再加载模型**。加载本身就是最可能 OOM 的那一步（一份 384
    # 会话 2.07GB，只有 2GB 空闲的卡上「加载」两个字就已经崩了），所以拒绝要在它之前。
    # 尺寸不加载也能知道（_dfm_size_hint 看文件名），加载后如果实测尺寸不一样再判一次。
    model_mb = _vram_model_mb(cfg["dfm"])
    size_hint = _dfm_size_hint(cfg["dfm"])
    want_frames = int(cfg.get("frame_budget_want") or 0)
    # ★ reclaim_first：**导出起手第一件事**是把常驻预览 worker 的模型卸掉，然后
    #   重新探显存、再算这一份预算。顺序反过来的代价是实测的 22 → 15.8 帧/秒：
    #   预览 worker 常驻 2733MB 时「空闲」只有 2766MB，帧预算被压到 8、dfm 批压到 1，
    #   而占着那 2733MB 的正是我们自己。
    bud = _fused_budget(want_frames, None, None, size_hint,
                        model_mb=model_mb, batch=int(batch_target),
                        force=bool(cfg.get("vram_force")),
                        reclaim=True, reclaim_first=True)
    _vram_apply(bud, "快速导出")
    for _ln in _vram_log_lines(bud, "快速导出"):
        _vid_log(_ln)
    lm = _landmarks_model()
    if lm is None:
        raise _BadInput("2D106 关键点模型不可用（weights/insightface_2d106.onnx）—— "
                        "没有 68 点就跑不了快速导出", 500)
    # 检测走哪个 EP（DFL_DET_EP=auto|gpu|cpu）—— 必须在**拿检测器**之前定，
    # 否则日志里那行「单帧 ms」量的是 GPU 那份、跑的是 CPU 那份，数字对不上账。
    det_pol = _det_ep_resolve(det_name, probe.get("w"), probe.get("h"))
    _vid_log("快速导出 %s" % _det_ep_line(det_pol))
    det = _detect_for(det_name, det_pol.get("ep") or "gpu")
    if det is None:                       # CPU 那一路起不来（极少数机器）→ 老实退回 GPU 并喊出来
        _vid_log("快速导出：检测器 %s EP 起不来 → 退回 GPU（%s）"
                 % (str(det_pol.get("ep")).upper(), det_name))
        det = _detector(det_name)
        det_pol["ep"] = "gpu"
        det_pol["fallback"] = True
    # 「用了哪个检测器 + 它到底多贵」必须在起流水线之前就落进 <成片>.log：
    # 界面上选的是 TinyMog、日志里出现 retinaface_10g 这种对不上账的问题，只有这里能复盘。
    _dp = _det_quick_profile(det, probe.get("w"), probe.get("h"))
    _vid_log("快速导出：检测器 %s（%s · %s）· 1080p 单帧 %s ms = letterbox %s + 前向 %s + "
             "NMS/解码 %s（零图 %d 次中位）"
             % (det_name, det_src, _dp.get("sess") or "?",
                _dp.get("full_ms"), _dp.get("prep_ms"), _dp.get("fwd_ms"),
                _dp.get("post_ms"), int(_dp.get("n") or 0)))
    if _dp.get("err"):
        _vid_log("快速导出：检测器基准没量成（%s）" % _dp["err"])
    elif det_name == "tinymog" and (_dp.get("full_ms") or 0.0) > 20.0:
        # 这条 warning 是给「以为在用 TinyMog」的人：tinymog 正常 2~6ms，20ms 以上只有三种可能
        _vid_log("快速导出：★ 检测器是 tinymog，单帧却 %.1f ms（正常 2~6ms）—— 查这三件事："
                 "有没有 TRT 引擎、是不是退回 onnxruntime、是不是在 CPU EP 上"
                 % float(_dp.get("full_ms") or 0.0))
    # 预算说不可行时**不加载** dfm —— 加载是要占显存的，先拒绝再别动显存
    if not bud.get("ok"):
        raise _BadInput(bud.get("msg") or _vram_refuse_msg(bud, "快速导出"), 400)
    ent0 = _dfm_session(str(cfg["dfm"]))
    size_px = int(ent0[2])
    if size_px != int(size_hint):
        _vid_log("快速导出：文件名给的尺寸 %d 与实测 %d 不一致 → 按实测重算预算"
                 % (int(size_hint), size_px))
        # **resident=True**：这一拍 dfm 已经在本进程里加载好了（size_px 是它自己报的），
        # 会话那 2GB 已经在卡上了 —— 再算一遍就是重复计数，会把可跑的单子判死。
        # 只判增量：检测/遮罩（检测器上面已经加载了）+ 帧缓冲 + dfm 批。
        bud = _fused_budget(want_frames, None, None, size_px,
                            model_mb=model_mb, batch=int(batch_target),
                            resident=True, force=bool(cfg.get("vram_force")),
                            reclaim=True, reclaim_first=False)
        _vram_apply(bud, "快速导出")
        for _ln in _vram_log_lines(bud, "快速导出"):
            _vid_log(_ln)
        if not bud.get("ok"):
            raise _BadInput(bud.get("msg") or _vram_refuse_msg(bud, "快速导出"), 400)
    budget = int(bud.get("frames") or bud.get("frame_budget") or _FUSED_BUDGETS[0])
    # 预算批 ≤ 攒批目标：显存只够批 2 时就别再按 8 攒（_target_for 还会按实测再降一次）
    _plan_batch = max(1, int(bud.get("batch") or batch_target))
    if _plan_batch < batch_target:
        _vid_log("快速导出：dfm 攒批 目标 %d → %d（显存预算）" % (batch_target, _plan_batch))
        batch_target = _plan_batch
    cfg["frame_budget"] = budget
    cfg["budget"] = bud
    cfg["vram_plan"] = bud
    xseg = None
    xs_label = ""
    if need_xseg:
        xseg, xs_res = _stage_xseg(mask_src.get("model"))
        xs_label = str(xs_res.get("label") or "")
        info = dict(getattr(xseg, "info", {}) or {})
        _vid_log("快速导出：遮罩 %s（%s）· 后端=%s%s"
                 % (xs_label or "-", Path(str(info.get("onnx") or "")).name or "-",
                    info.get("backend") or "?", ("(%s)" % info.get("precision"))
                    if info.get("precision") else ""))
    _vid_log("快速导出：dfm %s · %dx%d · %s · 检测器 %s thresh=%.2f（%s）"
             % (Path(str(cfg["dfm"])).name, size_px, size_px,
                (ent0[6] if len(ent0) > 6 else "?"), det_name, det_thresh, det_src))
    _vid_log("快速导出：显存预算 → 帧缓冲 %d / dfm 批 %d（请求 %d 帧 / 批 %d%s）· "
             "预计峰值 %s MB / 可用 %.0f MB"
             % (budget, int(bud.get("batch") or 0), int(bud.get("requested") or 0),
                int(bud.get("requested_batch") or 0),
                "，已降档" if bud.get("downgraded") else "，不用降档",
                ("%.0f" % bud["peak_mb"]) if bud.get("peak_mb") is not None else "装不下",
                float(bud.get("avail_mb") or 0.0)))
    if bud.get("cpu"):
        _vid_log("快速导出：★ 当前在 CPU 上跑 ★（显存装不下 + %s=1）—— 预计比 GPU 慢约 %.0f 倍"
                 % (_VRAM_ENV_CPU, _VRAM_CPU_SLOW))

    frames_dir = Path(tempfile.mkdtemp(prefix="dfl_fused_"))
    cfg["frames_dir"] = str(frames_dir)
    # 线程数一律把**实际生效的值**和它是从哪来的打出来：用户改过环境变量/请求体时，
    # 「日志里明明是 8，为什么只有 4」这种事不该再发生（这条日志是唯一能复盘的地方）。
    _vid_log("快速导出：线程 检测1/解码1/对齐%d/GPU1/合成%d/编码%d · 队列 %d · cv2 %d 线程 · "
             "检测预处理在解码线程 %s"
             % (n_prep, n_cmp, n_enc, qd, cv2_threads, "开" if det_prep_on else "关"))
    _vid_log("快速导出：线程来源 对齐=%s 合成=%s（B 配置默认 4/8；DFL_FUSED_PREP / "
             "DFL_FUSED_CMP / body.pred_readers / body.merge_workers 可覆盖 · cpu %d 线程）"
             % (_src_of(cfg, ("pred_readers",), _FUSED_PREP_DEFAULT, "DFL_FUSED_PREP"),
                _src_of(cfg, ("merge_workers", "merge_threads", "threads"),
                        _FUSED_CMP_DEFAULT, "DFL_FUSED_CMP"),
                int(os.cpu_count() or 0)))
    _vid_log("快速导出：攒批 目标 %d 张 / 超时 %dms（0=不超时，攒满或 END 才送）"
             % (batch_target, int(round(batch_wait * 1000.0))))
    _vid_log("快速导出：临时帧目录 %s（成功/取消都删；<视频>_cache/ 一个都不建）" % frames_dir)

    END = object()
    ticks = _FusedTicks()
    stop = threading.Event()
    errors = []
    sem = threading.BoundedSemaphore(budget)
    q_dec = queue.Queue(maxsize=max(2, qd))
    q_det = queue.Queue(maxsize=max(2, qd))
    q_gpu = queue.Queue(maxsize=max(qd, 64))
    q_cmp = queue.Queue(maxsize=max(2, qd))
    q_enc = queue.Queue(maxsize=max(2, qd))
    cnt = {"prep": n_prep, "compose": n_cmp, "encode": n_enc}
    cnt_lock = threading.Lock()
    live = {"n": 0, "peak": 0}
    live_lock = threading.Lock()
    stat_lock = threading.Lock()
    bstat = {"calls": 0, "faces": 0, "max": 0, "hist": {}, "why": {}}
    rep = {"at": 0.0}
    got = {"frames": 0, "swapped": 0, "copied": 0, "no_face": 0, "faces": 0, "skipped": 0}

    def _fail(stage, exc):
        errors.append("%s: %r" % (stage, exc))
        stop.set()

    def _put(q, item, timeout=0.2):
        while not stop.is_set():
            try:
                q.put(item, timeout=timeout)
                return True
            except queue.Full:
                continue
        return False

    def _get(q, timeout=0.2):
        while not stop.is_set():
            try:
                return q.get(timeout=timeout)
            except queue.Empty:
                continue
        return END

    def _poll(q, timeout=0.005):
        """非阻塞取一个：这一拍没活就返回 None，停止时返回 END。

        为什么 GPU 段需要它：攒批超时只能在「等下一张脸」的空档里去检查 —— 用 _get
        的无限等待，尾部那几张脸就会一直压在 pend 里等着凑满目标（见 _FUSED_BATCH_WAIT_MS）。
        """
        try:
            return q.get(timeout=timeout)
        except queue.Empty:
            return END if stop.is_set() else None

    def _report(done, force=False):
        now = time.time()
        if not force and (now - rep["at"]) < 0.25:
            return
        rep["at"] = now
        el = max(0.001, now - t_all)
        rate = float(done) / el
        eta = int((total - done) / rate) if rate > 0 and 0 < done < total else None
        pct = int(round(_VID_PCT_SWAP * float(done) / total)) if total else 0
        _stage_progress(
            "fused", min(100, pct),
            "快速导出 %d/%d 帧（第 %d..%d 帧）· %.1f 帧/秒"
            % (done, total, start_i, end_i, rate),
            frames_done=int(done), frames_total=int(total),
            frame_fps=round(rate, 2), eta_s=eta,
            swapped=int(got["swapped"]), copied=int(got["copied"]),
            no_face=int(got["no_face"]))
        _vid_stats(done=int(done), total=int(total), fps=round(rate, 2), eta_s=eta,
                   frame_budget=budget, fused=True, peak_live=int(live["peak"]))
        # 每段的「每帧 ms」+ dfm 当前的攒批（界面 lane 旁边那个小字）；_report 自己
        # 就把频率压到 0.25s 一拍，所以这里不用再节流。
        try:
            _pm = ticks.dump(int(done))
            _STAGE_METER.note_ms({k: _pm.get(v) for k, v in _STAGE_MS_SRC.items()})
            _STAGE_METER.note_batch(
                {"dfm": int(batch_of.get(str(cfg["dfm"])) or batch_target)})
        except Exception:                                  # noqa: BLE001
            pass

    # ---- 显存保护：dfm 批大小按 session 实测阶梯选（每个 dfm 一次）----
    batch_of = {}
    batch_meta = {}
    _batch_note = {}
    batch_lock = threading.Lock()

    def _batch_for(dfm_path):
        with batch_lock:
            hit = batch_of.get(dfm_path)
        if hit is not None:
            return hit
        ent = _dfm_session(dfm_path)
        b, bmeta = _pred_pick_batch({**cfg, "pred_batch": int(cfg.get("pred_batch") or 0)}, ent)
        with batch_lock:
            batch_of[dfm_path] = int(b)
            batch_meta[dfm_path] = dict(bmeta or {})
        _vid_log("快速导出：dfm %s 批大小 %d（%s）· %dx%d"
                 % (Path(dfm_path).name, int(b), bmeta.get("source"),
                    int(ent[2]), int(ent[2])))
        return int(b)

    # ---- ① 解码：ffmpeg 的 rawvideo 帧流 → 帧缓冲（信号量卡死上限，满了解码就停，天然背压）----
    # 默认 rawvideo（软解）：和切帧产物 frames/%06d.png 逐位相同，25.2→8.9 ms/帧
    # （见 _fused_open_frames 的注释 + app/tools/accept/fused_decode_parity.py 的对拍回归）。
    # DFL_FUSED_DECODE=png 可退回 PNG 管道（排障 / 想走「同一条 PNG 编解码路径」时用）。
    gen = {}
    dec_mode = str(os.environ.get(_FUSED_ENV_DECODE) or "raw").strip().lower()
    if dec_mode not in _FUSED_RAW_MODES:
        dec_mode = "png"                      # 只认显式非 raw 值（png）= 回落 PNG 管道
    src_bits = _fused_src_bits(video)        # 8 / 10 / 12 / 16：日志 + raw 选 bgr24 / bgr48le
    _vid_log("快速导出：解码 %s（%s）· 源 %sx%s / %sbit"
             % ("rawvideo %s（软解）" % ("bgr48le→>>8" if src_bits > 8 else "bgr24")
                if dec_mode != "png" else "PNG 管道",
                "默认；和切帧阶段落盘的 frames/%06d.png 逐位相同、25→9 ms/帧"
                if dec_mode != "png"
                else "DFL_FUSED_DECODE=png，可选；同一条 PNG 编解码路径，约 25 ms/帧",
                probe.get("w"), probe.get("h"), src_bits))

    def _decode():
        proc = None
        try:
            proc, next_frame = _fused_open_frames(video, start_i,
                                                 size=(probe.get("w"), probe.get("h")),
                                                 mode=dec_mode, bits=(src_bits or None))
            gen["proc"] = proc
            n = start_i - 1
            while not stop.is_set():
                if n >= end_i:
                    break
                t_rd = time.perf_counter()
                frame = next_frame()
                ticks.add("decode", time.perf_counter() - t_rd)
                if frame is None:
                    break
                n += 1
                t_wait = time.perf_counter()
                while not stop.is_set():
                    if sem.acquire(timeout=0.2):
                        break
                ticks.add("dec_wait", time.perf_counter() - t_wait)
                if stop.is_set():
                    break
                with live_lock:
                    live["n"] += 1
                    if live["n"] > live["peak"]:
                        live["peak"] = live["n"]
                job = _FusedFrame(n, n - start_i + 1, frame)
                if det_prep_on:
                    # 检测的 letterbox 在这条线程做：它只忙 15ms/帧（30%），有富余；
                    # 检测线程是硬底，把 0.7ms 的 resize + 填边搬走就是净赚。
                    # 逐位一致由「同一个 prep()」保证（见 _detect.py 的 _prep_ctx）。
                    t_prep = time.perf_counter()
                    job.det_ctx = _det_prep(det, frame)
                    if job.det_ctx is not None:
                        ticks.add("det_prep", time.perf_counter() - t_prep)
                _stage_tick("frames")            # 交接点：解码 → 检测队列
                if not _put(q_dec, job):
                    break
        except BaseException as e:                             # noqa: BLE001
            _fail("decode", e)
        finally:
            _put(q_dec, END)

    # ---- ② 检测：默认 1 路（检测器是共享对象，本来就不能并发）；异构池时 N 路 ----
    # 多路的由来：检测是这条链的硬底（实测 48.9 ms/帧 = 20.4 帧/秒，占用 96~98%），
    # 只靠一路 GPU 就把整条线钉死了。多开的那几路各拿**自己的** session（CPU EP 或
    # 第二份 CUDA），帧从 q_dec 里抢着取 —— 谁空谁拿，天然就是「最短队列优先」。
    # 这一路只做「前向 + NMS」：letterbox 已经在 decode 线程做完（见 _det_prep）。
    # 搬家只在「检测是瓶颈」时才值钱（retinaface：detect 占用 88%）；tinymog 下瓶颈
    # 已经移到 dfm/编码，端到端 0 收益 —— 别把它当提速手段。
    def _detect(k=0, det_obj=None, n_workers=1):
        try:
            while True:
                t_wait = time.perf_counter()
                job = _get(q_dec)
                ticks.add("det_wait", time.perf_counter() - t_wait)
                if job is END:
                    # 毒丸只放了一颗：谁吃到谁负责传下去（n_workers-1 颗），
                    # 最后各自退出 —— 不然其余几路会在 _get 里等到天荒地老。
                    for _i in range(max(0, int(n_workers) - 1)):
                        _put(q_dec, END)
                    for _i in range(n_prep):
                        _put(q_det, END)
                    break
                t = time.perf_counter()
                try:
                    if job.det_ctx is not None:
                        # 主路径：canvas 已经在 decode 线程做好，这里只剩前向 + NMS
                        boxes = (_detect_on_ctx(det_obj, job.det_ctx, det_thresh)
                                 if det_obj is not None
                                 else _detect_boxes_ctx(job.det_ctx, det_name, det_thresh))
                        job.det_ctx = None       # 1.2MB/帧：用完就放，别带到合成再扔
                    else:
                        boxes = (_detect_on(det_obj, job.frame, det_thresh)
                                 if det_obj is not None
                                 else _detect_boxes(job.frame, det_name, det_thresh))
                except Exception as e:                         # noqa: BLE001
                    _fail("detect", e)
                    for _i in range(n_prep):
                        _put(q_det, END)
                    break
                ticks.add("detect", time.perf_counter() - t)
                # 和 ② 切脸同一个次序（按 x1 排）：脸序号 = 这个顺序，compose 也按它贴
                job.boxes = sorted(list(boxes or []),
                                   key=lambda b: float((b.get("box") or [0, 0, 0, 0])[0]))
                job.rec["detected"] = len(job.boxes)
                _stage_tick("det")               # 交接点：检测 → 对齐/预处理
                _put(q_det, job)
        except BaseException as e:                             # noqa: BLE001
            _fail("detect", e)

    # ---- ③ 对齐/预处理 + 归组：68 点、dfm 输入、xseg 的 dst 画布、这一帧用哪个 dfm ----
    def _prep():
        try:
            while True:
                job = _get(q_det)
                if job is END:
                    return
                t = time.perf_counter()
                faces = []
                for k, b in enumerate(job.boxes or []):
                    box = b.get("box") or [0, 0, 0, 0]
                    try:
                        x1, y1, x2, y2 = [float(v) for v in box]
                    except (TypeError, ValueError):
                        continue
                    bw = max(1.0, x2 - x1)
                    bh = max(1.0, y2 - y1)
                    side = int(round(max(bw, bh) * 2.0))
                    side -= side % 2
                    kps = b.get("kps") or []
                    if len(kps) < 5 or side < 32:
                        continue
                    try:
                        cmat = coarse_align_from_5(kps)
                        coarse = cv2.warpAffine(job.frame, cmat,
                                                (COARSE_SIZE, COARSE_SIZE),
                                                flags=cv2.INTER_LINEAR,
                                                borderMode=cv2.BORDER_CONSTANT,
                                                borderValue=(0, 0, 0))
                        lm68a = lm.landmarks68(coarse)
                        if lm68a is None:
                            continue
                        pts = landmarks68_in_source(lm68a, cmat)
                        crop, meta_f = align_face_68(
                            job.frame, np.asarray(pts, np.float64),
                            box=[int(round(x1)), int(round(y1)),
                                 int(round(x2)), int(round(y2))], size=side)
                    except Exception as e:                     # noqa: BLE001
                        got["skipped"] += 1
                        _vid_log("快速导出：第 %d 帧第 %d 张脸跳过：%s"
                                 % (job.idx, k, str(e)[:100]))
                        continue
                    if crop is None or not meta_f:
                        got["skipped"] += 1
                        _why = ((meta_f or {}).get("reject_reason")
                                if isinstance(meta_f, dict) else "")
                        if _why:
                            _vid_log("快速导出：第 %d 帧第 %d 张脸拒收：%s"
                                     % (job.idx, k, _why))
                        continue
                    # compose 用的 landmarks = DFLJPG 里那份（align_face_68 已 round(4)）
                    lm_f = np.asarray(meta_f["source_landmarks"], np.float32)
                    xseg_dst = None
                    if need_dst:
                        # 分段的 dst 输入是 aligned/*.jpg（q100 4:4:4，仍然有损）→ 照抄这一步
                        # 质量取同一个 cfg["jpeg_quality"]（默认 100）：两条路必须逐位一致，
                        # 所以这里不写死 100，免得以后改一处忘另一处。
                        from ._align import jpeg_encode_params
                        okj, buf = cv2.imencode(
                            ".jpg", crop,
                            jpeg_encode_params(int(cfg.get("jpeg_quality")
                                                   or _STAGE_JPEG_QUALITY), 0))
                        if okj:
                            dec = cv2.imdecode(np.frombuffer(buf.tobytes(), np.uint8),
                                               cv2.IMREAD_COLOR)
                            m = _stage_dst_face_mat(meta_f, 256)
                            if dec is not None and m is not None:
                                xseg_dst = cv2.warpAffine(dec, m, (256, 256),
                                                          flags=cv2.INTER_CUBIC)
                    faces.append({"k": int(k), "lm": lm_f, "meta": meta_f,
                                  "crop": crop, "dst_u8": xseg_dst,
                                  "dfm": "", "dfm_in": None, "pred_u8": None,
                                  "prd_u8": None, "dstdfm_u8": None,
                                  "xseg_prd_u8": None, "xseg_dst_u8": None})
                ticks.add("align", time.perf_counter() - t)
                # 归组（有人脸库才真的算 embedding；没库就是一次索引读）
                t = time.perf_counter()
                models, no_swap, plan = _vid_models_for_frame(cfg, job.frame, job.boxes,
                                                              job.idx)
                ticks.add("embed", time.perf_counter() - t)
                job.plan = plan
                if plan is not None and not models:
                    job.faces = []          # 有库但这帧一张都不像：整帧保持原像素
                    job.rec["branch"] = "library-none"
                elif not faces:
                    job.faces = []
                    job.rec["branch"] = "copy"
                else:
                    job.rec["branch"] = "library" if plan is not None else "single"
                    skip = set(int(i) for i in (no_swap or []))
                    use = []
                    for f in faces:
                        i = int(f["k"])
                        if i in skip:
                            continue        # 谁都不像的脸：保持原像素，不兜给任何模型
                        midx = _model_for_face(i, models)
                        m = (models[midx] if models else {}) or {}
                        f["dfm"] = str(m.get("dfm") or cfg["dfm"])
                        f["name"] = str(m.get("name") or cfg["model"])
                        # ③ 换脸那一刀的输入：同一条 get_transform_mat(lm, S, WHOLE_FACE)
                        # + INTER_CUBIC warp + **uint8 整帧 warp 再 /255**（和 ③ 逐字同序）。
                        # 模型尺寸按「这张脸归谁的 dfm」定（多模型时各不相同）。
                        try:
                            sz = int(_dfm_session(f["dfm"])[2])
                            fmat = get_transform_mat(np.asarray(f["lm"], np.float32), sz,
                                                     FaceType.WHOLE_FACE, scale=1.0)
                            warped = cv2.warpAffine(job.frame, fmat, (sz, sz),
                                                    flags=cv2.INTER_CUBIC)
                            f["dfm_in"] = np.clip(warped.astype(np.float32) / 255.0, 0, 1)
                        except Exception as e:                 # noqa: BLE001
                            got["skipped"] += 1
                            _vid_log("快速导出：第 %d 帧第 %d 张脸 warp 失败：%s"
                                     % (job.idx, i, str(e)[:100]))
                            continue
                        use.append(f)
                    job.faces = use
                job.faces = job.faces or []
                if not job.faces:
                    _put(q_cmp, job)
                else:
                    # 每张脸都带上「这一帧一共几张」：GPU 段按 dfm 攒批，**可能先收到第 2 张、
                    # 后收到第 1 张**（不同 dfm 分开 flush）。只有生产者知道总数，收到的张数
                    # 凑齐才把这一帧交给合成 —— 否则 2 张脸的第一张一到就提前合成，
                    # 第二张的脸就丢了（这是原型里没暴露、但生产上必须堵的洞）。
                    nf = len(job.faces)
                    for f in job.faces:
                        _stage_tick("dfm")           # 交接点：对齐 → dfm/xseg（每张脸一次）
                        if not _put(q_gpu, (job, f, nf)):
                            break
        except BaseException as e:                             # noqa: BLE001
            _fail("prep", e)
        finally:
            with cnt_lock:
                cnt["prep"] -= 1
                last = cnt["prep"] == 0
            if last:
                _put(q_gpu, END)

    # ---- ④ dfm + ⑤ XSeg：按 dfm 攒批，一次前向；回填到各自的帧 ----
    oom_cap = {}                    # dfm_path -> OOM 退档后的硬上限（优先级最高）

    def _ladder_per_face(dfm_path) -> dict:
        """阶梯实测的 {batch: 每张 ms}（_pred_pick_batch 留下的那几行）。"""
        meta = batch_meta.get(dfm_path) or {}
        out = {}
        for row in (meta.get("ladder") or []):
            try:
                out[int(row.get("batch"))] = float(row.get("per_face_ms"))
            except (TypeError, ValueError, KeyError):
                continue
        return out

    def _target_for(dfm_path) -> int:
        """这一路 dfm 一次前向喂几张：攒批目标（默认 8），但**实测说不行就让开**。

        为什么不是硬写 8：攒批只在「GPU 是瓶颈」时值钱。416 的阶梯实测
        B1=29.1 / B4=28.7 / **B8=31.0~31.5**（每张），8 张比 4 张每张贵 5~11%；
        硬攒到 8 的端到端实测是 **-11%**（fused_ab_after_416：15.85 vs 17.82）。
        规则（三行）：
          · 阶梯没量到目标档            → 用目标（没数据就别猜）
          · 目标档每张 ≤ 最优 × 1.02    → 用目标（384：B8 就是最优 → 8）
          · 目标档每张 > 最优 × 1.02    → 用阶梯选的那档（416：用 4），日志说一次原因
        一句话：**目标 8 是「想攒到多少」的意图，不是「必须塞 8 张」的命令**。
        OOM 退档（oom_cap）优先级最高，压在所有规则之上。
        """
        b = int(_batch_for(dfm_path) or 1)
        cap_now = int(oom_cap.get(dfm_path) or _PRED_BATCH_CAP)
        tgt = max(b, int(batch_target))
        if tgt > b:
            per = _ladder_per_face(dfm_path)
            best = min(per.values()) if per else None
            at = per.get(tgt)
            if best and at is not None and at > best * 1.02:
                with batch_lock:
                    noted = bool(_batch_note.get(dfm_path))
                    _batch_note[dfm_path] = True
                if not noted:
                    _vid_log("快速导出：攒批目标 %d 张，但阶梯实测 B%d=%.1fms/张 比最优 "
                             "%.1fms 贵 %.0f%% → 按实测用 %d 张"
                             % (tgt, tgt, at, best, 100.0 * (at / best - 1.0), b))
                tgt = b
        return max(1, min(int(_PRED_BATCH_CAP), cap_now, tgt))

    def _gpu():
        pend = {}
        pend_frames = {}
        born = {}                       # dfm_path -> 「这一批第一张脸」进来的时刻

        def flush(only=None, why="full"):
            for dfm_path in (list(pend) if only is None else [only]):
                part = pend.get(dfm_path) or []
                if not part:
                    continue
                del pend[dfm_path]
                born.pop(dfm_path, None)
                bstat["why"][why] = int(bstat["why"].get(why) or 0) + 1
                # 攒批目标同时决定「什么时候送」和「一次塞几张」：攒到 8 就一次前向 8 张
                # （这正是 B 配置要的东西）。显存不够时的退档走 oom_cap，优先级最高。
                step = _target_for(dfm_path)
                i = 0
                while i < len(part):
                    chunk = part[i:i + step]
                    i += step
                    faces = [c[1] for c in chunk]
                    sz = size_px_of(dfm_path)
                    arr = np.empty((len(faces), sz, sz, 3), np.float32)
                    for k, f in enumerate(faces):
                        if f.get("dfm_in") is None:
                            # 兜底：绝不让 None/NaN 进网络（早先漏算 warp 时就是靠这个
                            # 变成一片糊的——静默穿过 TRT 再从合成里冒出来，极难查）
                            raise _BadInput("第 %d 张脸没有 dfm 输入（warp 没做）"
                                            % int(f.get("k") or 0), 500)
                        arr[k] = f["dfm_in"]
                    t = time.perf_counter()
                    try:
                        outs = _dfm_run_batch(_dfm_session(dfm_path), arr)
                    except Exception as e:                     # noqa: BLE001
                        if not (_is_oom(e) and len(faces) > 1):
                            raise
                        h = max(1, len(faces) // 2)
                        _vid_log("快速导出：dfm batch=%d 显存不够（%s）→ 退到 %d 再试"
                                 % (len(faces), str(e)[:80], h))
                        with batch_lock:
                            batch_of[dfm_path] = max(1, min(int(batch_of.get(dfm_path) or h), h))
                        # 攒批目标也要跟着退，否则下一次照样攒到 8 张再撞 OOM
                        oom_cap[dfm_path] = max(1, min(int(oom_cap.get(dfm_path) or _PRED_BATCH_CAP), h))
                        outs = []
                        for j in range(0, len(faces), h):
                            outs.extend(_dfm_run_batch(_dfm_session(dfm_path),
                                                       arr[j:j + h]))
                    ticks.add("dfm", time.perf_counter() - t)
                    bstat["calls"] += 1
                    bstat["faces"] += len(faces)
                    bstat["max"] = max(bstat["max"], len(faces))
                    key = str(len(faces))
                    bstat["hist"][key] = int(bstat["hist"].get(key) or 0) + 1
                    for k, f in enumerate(faces):
                        prd, dst = outs[k][1], outs[k][2]
                        f["pred_u8"] = np.clip(outs[k][0] * 255.0, 0, 255).astype(np.uint8)
                        f["prd_u8"] = (None if prd is None
                                       else np.clip(prd * 255.0, 0, 255).astype(np.uint8))
                        f["dstdfm_u8"] = (None if dst is None
                                          else np.clip(dst * 255.0, 0, 255).astype(np.uint8))
                    t = time.perf_counter()
                    if need_xseg:
                        # 和 ④ 遮罩阶段同一个口径：一张脸一次 mask_batch([pred, dst])
                        for f in faces:
                            imgs = [f["pred_u8"]]
                            if f.get("dst_u8") is not None:
                                imgs.append(f["dst_u8"])
                            try:
                                masks = xseg.mask_batch(imgs, out_size=256) or []
                            except Exception as e:             # noqa: BLE001
                                raise _BadInput("遮罩推理失败：%s" % (str(e)[:160]), 500)
                            f["xseg_prd_u8"] = (
                                np.clip(masks[0] * 255.0, 0, 255).astype(np.uint8)
                                if len(masks) > 0 and masks[0] is not None else None)
                            f["xseg_dst_u8"] = (
                                np.clip(masks[1] * 255.0, 0, 255).astype(np.uint8)
                                if len(masks) > 1 and masks[1] is not None else None)
                    ticks.add("xseg", time.perf_counter() - t)
                    for job, _f in chunk:
                        rec = pend_frames.get(id(job))
                        if rec is not None:
                            rec[0] -= 1
                            if rec[0] <= 0:
                                pend_frames.pop(id(job), None)
                                _stage_tick("xseg")      # 交接点：dfm/xseg → 合成
                                _put(q_cmp, job)

        def size_px_of(dfm_path):
            return int(_dfm_session(dfm_path)[2])

        try:
            while True:
                t_wait = time.perf_counter()
                item = _poll(q_gpu, 0.005)
                ticks.add("gpu_wait", time.perf_counter() - t_wait)
                if item is END:
                    flush(why="end")
                    for _i in range(n_cmp):
                        _put(q_cmp, END)
                    break
                if item is None:
                    # 这一拍没有新脸进来：看看哪一路已经攒到超时了，先把它送出去。
                    # 这是「尾部/断流」的保险 —— 正常出片时到脸间隔 30~80ms，
                    # 超时 20ms 是量出来的最优（默认值见 _FUSED_BATCH_WAIT_MS 的实测表）。
                    if batch_wait > 0.0 and pend:
                        now = time.perf_counter()
                        for _p in list(pend):
                            if now - float(born.get(_p, now)) >= batch_wait:
                                flush(_p, why="timeout")
                    continue
                job, f, nf = item
                if id(job) not in pend_frames:
                    # 生产者给的「这一帧一共几张脸」：收到的回填进度按它算（见 _prep 的说明）
                    pend_frames[id(job)] = [int(nf)]
                if not pend.get(f["dfm"]):
                    born[f["dfm"]] = time.perf_counter()
                pend.setdefault(f["dfm"], []).append((job, f))
                if len(pend[f["dfm"]]) >= _target_for(f["dfm"]):
                    flush(f["dfm"], why="full")
        except BaseException as e:                             # noqa: BLE001
            _fail("gpu", e)

    # ---- ⑤ 合成（纯 CPU，就是 ⑤ 合成阶段那个 compose_frame）----
    def _compose():
        try:
            while True:
                t_wait = time.perf_counter()
                job = _get(q_cmp)
                ticks.add("cmp_wait", time.perf_counter() - t_wait)
                if job is END:
                    return
                t = time.perf_counter()
                args = []
                for f in (job.faces or []):
                    prd = f.get("prd_u8")
                    dst = f.get("dstdfm_u8")
                    xp = f.get("xseg_prd_u8")
                    xd = f.get("xseg_dst_u8")
                    args.append({
                        "landmarks": f["lm"],
                        "pred": f["pred_u8"].astype(np.float32) / 255.0,
                        "prd_dfm": (None if prd is None
                                    else prd.astype(np.float32) / 255.0),
                        "dst_dfm": (None if dst is None
                                    else dst.astype(np.float32) / 255.0),
                        "xseg_prd": (None if xp is None
                                     else xp.astype(np.float32) / 255.0),
                        "xseg_dst": (None if xd is None
                                     else xd.astype(np.float32) / 255.0),
                    })
                try:
                    out_frame = compose_frame(job.frame, args, comp_cfg, job.timings)
                except Exception as e:                         # noqa: BLE001
                    raise _BadInput("第 %d 帧合成失败：%s" % (job.idx, str(e)[:160]), 500)
                ticks.add("compose", time.perf_counter() - t)
                if dump_frames and int(job.idx) in dump_frames:
                    _fused_dump(cfg, job, out_frame)
                job.rec["faces"] = len(args)
                job.rec["swapped"] = len(args)
                job.rec["no_face"] = not args
                job.frame = None
                _stage_tick("compose")           # 交接点：合成 → PNG 编码
                _put(q_enc, (job, out_frame))
        except BaseException as e:                             # noqa: BLE001
            _fail("compose", e)
        finally:
            with cnt_lock:
                cnt["compose"] -= 1
                last = cnt["compose"] == 0
            if last:
                for _i in range(n_enc):
                    _put(q_enc, END)

    # ---- ⑥ PNG 编码 + 落盘（无损，level 同 ⑤ 合成阶段）----
    # 为什么**不需要**重排/单写者：帧图是 `%06d.png` 定长序列，文件名自带帧号，
    # 谁先写完都不影响 ffmpeg 读到的是第几帧（分段链的 merge 阶段也是多 worker 乱序写盘）。
    # 之前那版照抄原型留了个"唯一写者按帧号顺序"的环节，代价是：写线程死等最慢的那一帧，
    # 缓冲一满就把 encode 线程堵住、sem 不释放、解码停摆 —— 实测 detect 只有 61% 占用、
    # 整条线 10.75 帧/秒（detect 单独跑是 52.8ms/帧 = 18.9 帧/秒的硬底）。
    def _encode():
        try:
            while True:
                t_wait = time.perf_counter()
                item = _get(q_enc)
                ticks.add("enc_wait", time.perf_counter() - t_wait)
                if item is END:
                    return
                job, img = item
                t = time.perf_counter()
                data = _stage_imencode_png(img, png_level)
                ticks.add("encode", time.perf_counter() - t)
                _stage_tick("encode")            # 交接点：PNG 编码 → 写盘
                t = time.perf_counter()
                _stage_write_atomic(frames_dir / ("%06d.png" % job.n), data)
                ticks.add("write", time.perf_counter() - t)
                with live_lock:
                    live["n"] -= 1
                try:
                    sem.release()
                except ValueError:
                    pass
                with stat_lock:
                    got["frames"] += 1
                    if job.rec.get("no_face"):
                        got["copied"] += 1
                        got["no_face"] += 1
                    else:
                        got["swapped"] += 1
                    got["faces"] += int(job.rec.get("faces") or 0)
                    done = got["frames"]
                _vid_record(job.rec)
                _report(done)
        except BaseException as e:                             # noqa: BLE001
            _fail("encode", e)

    # dfm 批大小阶梯（_pred_pick_batch）会真的跑几次空张量前向来量耗时：留在流水线里会让
    # 开头 1~2 秒的 GPU 被它占着（120 帧的片子那是 15% 的开销）。单模型这条最常见的路
    # 提前在这里跑掉 —— 既预热了 session，也让测速窗口干净。
    try:
        _batch_for(str(cfg["dfm"]))
    except Exception as e:                                     # noqa: BLE001
        _vid_log("快速导出：dfm 批大小探测失败（%s）→ 流水线里再试" % str(e)[:120])

    cv2_prev = 0
    cv2_set = 0
    if cv2_threads > 0:
        try:
            cv2_prev = int(cv2.getNumThreads())
            if cv2_prev != int(cv2_threads):
                cv2.setNumThreads(int(cv2_threads))
                cv2_set = int(cv2_threads)
        except Exception:                                      # noqa: BLE001
            cv2_prev, cv2_set = 0, 0

    # ---- 异构 session 池：探测 → 微基准 → 配比（见 server/_sess.py）----
    # 默认走池（n_det = plan.gpu + plan.cpu）；DFL_SESS_PLAN=0 退回单路老行为。
    det_sess = []
    det_plan = {}
    if _sess.enabled():
        try:
            det_plan = _fused_det_plan(det_name, probe.get("w"), probe.get("h"), {
                "gpu": cfg.get("sess_gpu"), "cpu": cfg.get("sess_cpu"),
                "strategy": cfg.get("sess_strategy"), "level": cfg.get("sess_level")})
            for _i in range(max(0, int(det_plan.get("gpu") or 0))):
                _o = _detect_for(det_name, "gpu" if _i == 0 else "gpu2")
                if _o is not None:
                    det_sess.append(("gpu", _o))
            for _j in range(max(0, int(det_plan.get("cpu") or 0))):
                _o = _detect_for(det_name, "cpu")
                if _o is not None:
                    det_sess.append(("cpu", _o))
            _vid_log("快速导出 检测池：%s" % _sess.plan_line(det_plan))
            _vid_log("快速导出 检测池 EP：%s"
                     % _det_ep_line(det_plan.get("det_ep") or det_pol))
            _vid_log("快速导出 检测池 基准：GPU %s ms / CPU %s ms（1080p 零图，2 张取中位）"
                     "· 实际起来 %d 路（GPU %d / CPU %d）"
                     % ((det_plan.get("bench") or {}).get("gpu_ms"),
                        (det_plan.get("bench") or {}).get("cpu_ms"), len(det_sess),
                        sum(1 for _e, _o in det_sess if _e.startswith("gpu")),
                        sum(1 for _e, _o in det_sess if _e == "cpu")))
        except Exception as e:                                 # noqa: BLE001
            det_sess = []
            _vid_log("快速导出 检测池：推导失败（%s）→ 退回单路" % str(e)[:160])
    n_det = len(det_sess) or 1

    # ---- 流水线实况：lane 数 = **实际**线程数（不是写死的）----
    # 合成池 n_cmp 条（t0..t7）、编码 n_enc 条（e0..e5）、dfm/xseg 各一个 session 就 1 条、
    # det 有几路异构 session 就有几条。xseg 不用（遮罩不来自模型）时给空列表 ——
    # 界面画成「不用」，不假装有一条在跑的 lane。
    _STAGE_METER.reset(lanes={
        "frames": ["解码"],
        "det": ["det#%d" % _i for _i in range(n_det)],
        "dfm": ["dfm#0"],
        "xseg": (["xse#0"] if need_xseg else []),
        "compose": ["t%d" % _i for _i in range(n_cmp)],
        "encode": ["e%d" % _i for _i in range(n_enc)],
    })
    # 每个阶段的「下游队列」= 列间那句「在飞 N」（inflight 直接读 qsize()）
    _STAGE_METER.bind({"frames": q_dec, "det": q_det, "dfm": q_gpu,
                       "xseg": q_cmp, "compose": q_enc})

    ths = {}
    t_all = time.time()
    try:
        ths["decode"] = threading.Thread(target=_decode, name="fused-dec", daemon=True)
        if n_det <= 1:
            ths["detect"] = threading.Thread(target=_detect, name="fused-det", daemon=True)
        else:
            for _i, (_ep, _o) in enumerate(det_sess):
                ths["detect%d" % _i] = threading.Thread(
                    target=_detect, args=(_i, _o, n_det), name="fused-det%d-%s" % (_i, _ep),
                    daemon=True)
        for i in range(n_prep):
            ths["prep%d" % i] = threading.Thread(target=_prep, name="fused-prep%d" % i,
                                                 daemon=True)
        ths["gpu"] = threading.Thread(target=_gpu, name="fused-gpu", daemon=True)
        for i in range(n_cmp):
            ths["cmp%d" % i] = threading.Thread(target=_compose, name="fused-cmp%d" % i,
                                                daemon=True)
        for i in range(n_enc):
            ths["enc%d" % i] = threading.Thread(target=_encode, name="fused-enc%d" % i,
                                                daemon=True)
        # 从下游往上游起：先有消费者，生产者才不会一上来就撞满队列
        _det_keys = (["detect"] if n_det <= 1
                     else ["detect%d" % i for i in range(n_det)])
        for k in (["enc%d" % i for i in range(n_enc)]
                  + ["cmp%d" % i for i in range(n_cmp)] + ["gpu"]
                  + ["prep%d" % i for i in range(n_prep)] + _det_keys + ["decode"]):
            ths[k].start()
        # 全部线程退出 = 全片跑完；中途取消就置 stop 让所有段退出
        while True:
            if _fused_cancelled():
                stop.set()
                break
            if not any(t.is_alive() for t in ths.values()):
                break
            time.sleep(0.1)
    finally:
        wall = time.time() - t_all
        stop.set()
        for k, th in ths.items():
            try:
                th.join(timeout=10.0)
            except Exception:                                  # noqa: BLE001
                pass
        alive = [k for k, th in ths.items() if th.is_alive()]
        try:
            proc = gen.get("proc")
            if proc is not None:
                _kill_proc_tree(proc)
        except Exception:                                      # noqa: BLE001
            pass
        if cv2_set:
            try:
                cv2.setNumThreads(cv2_prev)
            except Exception:                                  # noqa: BLE001
                pass

    written = int(got["frames"] or 0)
    if _fused_cancelled():
        raise _StageCancel()
    if errors:
        raise _BadInput("快速导出失败（%s）" % (errors[0][:400],), 500)
    if alive:
        raise _BadInput("快速导出异常收尾：还有线程没退出（%s）" % "、".join(sorted(alive)), 500)
    if written <= 0:
        raise _BadInput("一帧都没解出来：%s" % video.name, 422)

    sec = round(time.time() - t_all, 2)
    rate = written / max(1e-6, sec)
    per_stage = ticks.dump(written)
    busy = {k: ticks.raw(k) for k in per_stage}
    util = {
        # 等待（背压）也算一列：某一段 100% 却整条线慢，看 *_wait 就知道谁在等谁
        "dec_wait": round(100.0 * busy.get("dec_wait", 0.0) / max(1e-6, wall), 1),
        "det_wait": round(100.0 * busy.get("det_wait", 0.0) / max(1e-6, wall), 1),
        "gpu_wait": round(100.0 * busy.get("gpu_wait", 0.0) / max(1e-6, wall), 1),
        "cmp_wait": round(100.0 * busy.get("cmp_wait", 0.0) / max(1e-6, wall * n_cmp), 1),
        "enc_wait": round(100.0 * busy.get("enc_wait", 0.0) / max(1e-6, wall * n_enc), 1),
        "decode": round(100.0 * busy.get("decode", 0.0) / max(1e-6, wall), 1),
        "detect": round(100.0 * busy.get("detect", 0.0) / max(1e-6, wall), 1),
        "det_prep": round(100.0 * busy.get("det_prep", 0.0) / max(1e-6, wall), 1),
        "align": round(100.0 * busy.get("align", 0.0) / max(1e-6, wall * n_prep), 1),
        "embed": round(100.0 * busy.get("embed", 0.0) / max(1e-6, wall * n_prep), 1),
        "dfm": round(100.0 * busy.get("dfm", 0.0) / max(1e-6, wall), 1),
        "xseg": round(100.0 * busy.get("xseg", 0.0) / max(1e-6, wall), 1),
        "compose": round(100.0 * busy.get("compose", 0.0) / max(1e-6, wall * n_cmp), 1),
        "encode": round(100.0 * busy.get("encode", 0.0) / max(1e-6, wall * n_enc), 1),
        "write": round(100.0 * busy.get("write", 0.0) / max(1e-6, wall), 1),
    }
    _vid_log("快速导出：流水线跑完 %d 帧 · %.2f s · %.2f 帧/秒 · 帧缓冲 %d（峰值在飞 %d 帧）"
             % (written, sec, rate, budget, int(live["peak"])))
    _vid_log("快速导出 profile（每帧 ms）：%s"
             % " · ".join("%s=%.2f" % (k, v) for k, v in per_stage.items()))
    _vid_log("快速导出 各段占用：%s"
             % " · ".join("%s=%.0f%%" % (k, v) for k, v in util.items()))
    _vid_log("快速导出 dfm 批次：调用 %d 次 / %d 张脸 / 平均 %.1f 张 · 最大 %d · 分布 %s"
             % (bstat["calls"], bstat["faces"],
                bstat["faces"] / max(1, bstat["calls"]), bstat["max"],
                dict(sorted(bstat["hist"].items(), key=lambda kv: int(kv[0])))))
    _vid_log("快速导出 dfm 攒批：目标 %d 张 / 超时 %dms · 送出原因 %s（full=攒满 · "
             "timeout=超时 · end=收尾）"
             % (batch_target, int(round(batch_wait * 1000.0)),
                dict(sorted((bstat.get("why") or {}).items()))))
    mem1 = _proc_mem_mb()
    _vid_log("快速导出 内存（本进程）：流水线前 RSS %s / 提交 %s MB → 流水线后 RSS %s / 提交 %s MB"
             " · 峰值 RSS %s MB（帧缓冲 %d 帧 × %.2f MB = %.0f MB 是设计内的常驻，"
             "ORT 建会话时那 1.5GB 权重要过一遍内存，是峰值的主要来源）"
             % (_fmt_mb(mem0.get("rss")), _fmt_mb(mem0.get("private")),
                _fmt_mb(mem1.get("rss")), _fmt_mb(mem1.get("private")),
                _fmt_mb(mem1.get("peak_rss")), budget, _fused_frame_mb(size_px),
                budget * _fused_frame_mb(size_px)))
    if written != total:
        _vid_log("快速导出：实解出 %d 帧（源信息说 %d 帧）→ 按实解帧数编码"
                 % (written, total))
    # 把这一轮实测的「段占用」留给下一轮的 derive()（这就是「按实测调配比」的回路）
    try:
        _SESS_BUSY["detect"] = round(float(util.get("detect") or 0.0) / 100.0, 3)
        _SESS_PLAN_LAST["detect"] = dict(det_plan or {})
        # 顺带学两件事（见 _sess.record_gain 的注释）：
        #   ① 1x1 / 1x2（多挂一路 CPU 检测 session）值不值；
        #   ② 0x1（检测整段走 CPU EP）值不值 —— 这是 DFL_DET_EP=auto 的门②。
        # 以前这行的参数少传了一个（把 rate 当 cpu_n、把 WORKSPACE 当 fps），
        # float(Path) 每次都抛 TypeError 被下面 except 吞掉 —— 于是 gain:detect
        # 永远写不进磁盘，「按实测学」这条回路其实一直没通电。
        _g = _sess.record_gain("detect", int((det_plan or {}).get("gpu") or 0),
                               int((det_plan or {}).get("cpu") or 0), rate, WORKSPACE)
        if _g.get("gain_pct") is not None:
            _vid_log("快速导出 检测池 实测增益：%s%%（基线 1×GPU %.2f → 本次 %s %.2f 帧/秒）→ "
                     "自动档%s"
                     % (_g.get("gain_pct"), float(_g.get("base") or 0),
                        _g.get("now_tag") or "-", float(rate),
                        ("会开 CPU session" if _g.get("auto_cpu_ok")
                         else "仍然只开 GPU（净增 <5%）")))
        if _g.get("cpu_ep_gain_pct") is not None:
            _vid_log("快速导出 检测 EP 实测：CPU %.2f 帧/秒 对基线 1×GPU %.2f = %+.2f%% → "
                     "DFL_DET_EP=auto 的门②%s（工作区 _sess_bench.json 的 gain:detect）"
                     % (float(_g.get("cpu_ep_fps") or 0), float(_g.get("base") or 0),
                        float(_g.get("cpu_ep_gain_pct") or 0),
                        ("已通过（≥+5%%），以后 auto 会选 CPU EP"
                         if _g.get("auto_cpu_ep_ok")
                         else "未通过（<+5%），auto 固定留在 GPU")))
    except Exception:                                       # noqa: BLE001
        pass
    _vid_stats(done=written, total=written, fps=round(rate, 2), eta_s=0,
               frame_budget=budget, fused=True, per_stage_ms=per_stage,
               stage_util_pct=util, batch=batch_of, batch_actual=dict(bstat),
               sess_plan={"detect": dict(det_plan or {}), "n_detect": int(n_det),
                          "sessions": [{"ep": _e} for _e, _o in det_sess]},
               pipeline={
                   "threads": {"decode": 1, "detect": 1, "prep": n_prep, "gpu": 1,
                               "compose": n_cmp, "encode": n_enc},
                   "threads_src": {
                       "prep": _src_of(cfg, ("pred_readers",), _FUSED_PREP_DEFAULT,
                                       "DFL_FUSED_PREP"),
                       "compose": _src_of(cfg, ("merge_workers", "merge_threads", "threads"),
                                          _FUSED_CMP_DEFAULT, "DFL_FUSED_CMP")},
                   "batch_target": int(batch_target),
                   "batch_wait_ms": int(round(batch_wait * 1000.0)),
                   "decode_mode": dec_mode,
                   "queue_depth": qd, "budget": budget, "cv2_threads": cv2_threads})
    _stage_progress("fused", _VID_PCT_SWAP,
                    "快速导出：%d 帧跑完，开始编码（%d 帧 @ %.6g fps）"
                    % (written, written, fps),
                    frames_done=written, frames_total=written,
                    frame_fps=round(rate, 2))

    # ---- ⑦ 编码：和 ⑥ 导出**同一个函数**（crf16 / yuv420p / tv-range / 音轨 copy）----
    out.parent.mkdir(parents=True, exist_ok=True)
    part = out.with_name(out.stem + ".part" + out.suffix)
    dur = (written / fps) if fps > 0 else 0.0
    t_audio = ((start_i - 1) / fps) if fps > 0 else 0.0
    cmd = _vid_cmd(str(frames_dir / "%06d.png"), part, video, fps, img_start=1,
                   audio_ss=(None if full else t_audio),
                   audio_t=(None if full else dur),
                   limit=(None if full else int(written)))

    def _on_prog(done):
        pct = _VID_PCT_SWAP + (100 - _VID_PCT_SWAP) * min(1.0, done / max(1, written))
        _stage_progress("encode", pct, "编码 %d/%d 帧" % (done, written),
                        frames_done=min(done, written), frames_total=written)

    enc = _vid_encode(cmd, part, written, on_progress=_on_prog,
                      cancelled=_fused_cancelled)
    if enc["code"] != 0:
        try:
            if part.is_file():
                part.unlink()
        except OSError:
            pass
        raise _BadInput("编码失败（ffmpeg 退出码 %s）：%s"
                        % (enc["code"], (enc["tail"] or "")[-600:]), 500)
    if not part.is_file() or part.stat().st_size <= 0:
        raise _BadInput("编码跑完了却没有产物：%s" % part, 500)
    os.replace(str(part), str(out))
    mb = round(out.stat().st_size / 1048576.0, 2)
    _vid_log("快速导出：编码完成 %s（%.2f MB · %d 帧 · %.3f s · %.4f s 墙钟）"
             % (out.name, mb, written, dur, enc["seconds"]))
    _vid_log("快速导出 编码命令：%s"
             % " ".join('"%s"' % a if " " in str(a) else str(a) for a in cmd))

    # ---- 断言：全程没碰 <视频>_cache/（这是 fused 和分段的关键区别）----
    if (not cache_pre) and cache_dir.exists():
        raise _BadInput("快速导出不该创建 %s，但它出现了 —— 说明有代码偷偷写了 cache"
                        % cache_dir, 500)
    _vid_log("快速导出：<视频>_cache/ %s" % ("本来就在（没动它）" if cache_pre else "全程没建 ✔"))

    return {"frames": written, "seconds": sec, "fps": round(rate, 2),
            "encode_seconds": enc["seconds"], "mb": mb, "duration": round(dur, 6),
            "out": str(out), "frame_budget": budget, "budget": bud,
            "per_stage_ms": per_stage, "stage_util_pct": util,
            "batch": dict(batch_of), "batch_actual": dict(bstat),
            "peak_live": int(live["peak"]),
            "cache_dir": str(cache_dir), "cache_created": False,
            "audio_ss": (None if full else round(t_audio, 6)),
            "range": {"start": start_i, "end": end_i, "total": total, "full": full},
            "frames_dir": str(frames_dir),
            "threads": {"prep": n_prep, "compose": n_cmp, "encode": n_enc},
            "threads_src": {
                "prep": _src_of(cfg, ("pred_readers",), _FUSED_PREP_DEFAULT,
                                "DFL_FUSED_PREP"),
                "compose": _src_of(cfg, ("merge_workers", "merge_threads", "threads"),
                                   _FUSED_CMP_DEFAULT, "DFL_FUSED_CMP")},
            "batch_target": int(batch_target),
            "batch_wait_ms": int(round(batch_wait * 1000.0)),
            "decode_mode": dec_mode,
            "detector": str(det_name), "detector_src": str(det_src),
            "det_prep": bool(det_prep_on), "det_probe": dict(_dp or {})}


def _fused_worker(cfg: dict):
    """跑一次快速导出，并把终态落到两个 status（老 /export/status + /stage/status）。"""
    ok, err, cancelled = True, "", False
    t0 = time.time()
    mem0 = _proc_mem_mb()          # 导出前（起线程之前那一刻）：收尾要拿它做差
    frames_dir = ""
    try:
        _stage_progress("fused", 0, "快速导出启动…",
                        frames_done=0, frames_total=int(cfg.get("range_total") or 0))
        res = _fused_run(cfg) or {}
        frames_dir = str(res.get("frames_dir") or cfg.get("frames_dir") or "")
        with _merge_job_lock:
            _merge_job["encode"] = {"seconds": res.get("encode_seconds"), "code": 0,
                                    "out": str(res.get("out")), "mb": res.get("mb")}
        cfg["_fused_result"] = res
        _vid_log("快速导出完成：%s（%d 帧 · %.2f 帧/秒 · 编码 %.2fs）"
                 % (res.get("out"), res.get("frames"), res.get("fps"),
                    res.get("encode_seconds") or 0))
    except _StageCancel:
        ok, cancelled = False, True
        err = "已取消（快速导出不支持续跑，需整段重跑）"
    except _BadInput as e:
        ok, err = False, str(e.msg)[:1200]
    except Exception as e:                                     # noqa: BLE001
        traceback.print_exc()
        ok, err = False, "%s: %s" % (e.__class__.__name__, str(e)[:600])
    finally:
        # 流水线实况收单：不再涨，但**保留** lanes/ticks —— 进度页那张图在「完成」
        # 那一刻要还是满的（收单就清空的话，界面会当场塌成一排占位格）。
        _STAGE_METER.stop()
        # 强制尝试失败：先回滚现场（放掉本进程会话 + 重算当下的账 + 改写 err）再收尾 ——
        # 见 _vram_oom_rollback / _vram_forced_rollback。顺序反了用户看到的就是
        # 「导出失败」，而不是「确实不够，已回滚」。
        if (not ok) and (not cancelled) and cfg.get("vram_force"):
            _rb = _vram_forced_rollback("快速导出", _vram_model_mb(cfg.get("dfm")),
                                        {"mode": "frames",
                                         "frames": int(cfg.get("frame_budget_want") or 0),
                                         "batch": int(cfg.get("fused_batch") or 0),
                                         "procs": 1}, err)
            if _rb:
                cfg["vram_oom"] = _rb
                err = str(_rb.get("msg") or err)
        # active 保持 true 直到下面的收尾全部做完（删临时帧 / 清半截成片 / 写终态）。
        # 取消之后 /export/status 的 running 立刻变 false（既有语义，看门狗靠它），
        # 所以「能不能再点一次导出」要看 active —— 临时帧还没删完就再起一单，
        # 会在 %TEMP% 里留下一个没人管的 dfl_fused_* 目录。
        pass
        # 临时帧目录：成功、失败、取消**都删**（不存在续跑这回事）
        tmp = str(cfg.get("frames_dir") or frames_dir or "")
        removed = False
        if tmp:
            try:
                shutil.rmtree(tmp, ignore_errors=True)
                removed = not Path(tmp).exists()
            except OSError:
                removed = False
            _vid_log("快速导出：临时帧目录 %s（%s）" % (tmp, "已删除" if removed else "没删干净"))
        # 半截成片（.part）绝不能留着被当成结果
        try:
            op = Path(str(cfg.get("out") or ""))
            if op.name:
                p = op.with_name(op.stem + ".part" + op.suffix)
                if p.is_file():
                    p.unlink()
                    _vid_log("快速导出：删掉半截成片 %s" % p.name)
        except OSError:
            pass
        res = cfg.get("_fused_result") or {}
        cache_state = clear_face_group_cache(
            "快速导出完成" if ok else ("快速导出取消" if cancelled else "快速导出失败"))
        wall = round(time.time() - t0, 2)
        bud = (res.get("frame_budget") if res else None) or cfg.get("frame_budget")
        # 显示层用的是 res["budget"]（_fused_run 落实过的计划）；res 没有（失败得太早）就
        # 用 _start_video_export 闸门算的那一份 —— status 里始终有 vram_plan 可查
        vplan = dict((res.get("budget") if res else None) or cfg.get("vram_plan") or {})
        _oom = dict(cfg.get("vram_oom") or {})
        if _oom:
            # 强制尝试 OOM 之后：status 里的 vram_plan 换成**回滚后当下**的这一份，
            # 前端拿到它就能把「被拒态」的数字重画一遍（不是 OOM 之前那份旧账）。
            vplan = dict(_oom.get("plan") or vplan)
            vplan["forced_failed"] = True
        if vplan:
            vplan["cpu"] = bool(_vram_cpu_on())
        with _STAGE_LOCK:
            _STAGE_JOB.update({
                "running": False, "ok": bool(ok), "cancelled": bool(cancelled),
                "phase": ("done" if ok else ("cancelled" if cancelled else "failed")),
                "finished_at": time.time(), "error": "" if ok else err,
                "vram_oom": (_oom or None),
                "error_detail": "" if ok else "\n".join(
                    list(_STAGE_JOB.get("log") or [])[-30:]),
                "pct": (100 if ok else int(_STAGE_JOB.get("pct") or 0)),
                "stage_pct": (100 if ok else int(_STAGE_JOB.get("stage_pct") or 0)),
                "stage": "fused", "stage_index": -1,
                "msg": ("快速导出完成：%d 帧 · %.2f 帧/秒"
                        % (int(res.get("frames") or 0), float(res.get("fps") or 0.0))
                        if ok else ("已取消（快速导出不支持续跑）" if cancelled
                                    else "失败：" + err[:80])),
                "out_video": str(res.get("out") or cfg.get("out") or ""),
                "cache_dir": "", "cache_removed": removed, "cache": cache_state,
                "frame_budget": bud, "resumed": False, "seconds": {"fused": wall},
                "counts": {"frames": int(res.get("frames") or 0), "faces": 0,
                           "pred": 0, "masks": 0},
                "meta": {}, "vram_plan": vplan, "on_cpu": bool(_vram_cpu_on()),
            })
        with _merge_job_lock:
            _merge_job.update({
                "running": False, "ok": bool(ok) and not cancelled,
                "cancelled": bool(cancelled), "finished_at": time.time(),
                "pct": (100 if ok else int(_merge_job.get("pct") or 0)),
                "pct_at_failure": (None if ok else int(_merge_job.get("pct") or 0)),
                "stage": (6 if ok else -1),
                "msg": ("合成完成" if ok else
                        ("已取消（快速导出不支持续跑，需整段重跑）" if cancelled else "合成失败")),
                "error": ("" if ok else err),
                "error_detail": ("" if ok else "\n".join(_log_tail(
                    list(_merge_job.get("log") or []), 30))[:4000]),
                "cache": cache_state, "frame_budget": bud, "active": False,
                "vram_plan": vplan, "on_cpu": bool(_vram_cpu_on()),
                "vram_oom": (_oom or None),
            })
        _vid_log("快速导出结束：%s · %.2f s%s"
                 % ("成功" if ok else ("已取消" if cancelled else "失败"), wall,
                    ("（不支持续跑）" if cancelled else "")))
        # 收尾把本进程的 dfm/XSeg 会话放掉（成功/失败/取消都放）：留着它下一次预览
        # 会被自己的后端堵死 —— 见 _vram_release_inproc 的注释。预览 worker 不重载，
        # 下次预览再懒加载。
        try:
            _vram_release_inproc("快速导出结束")
        except Exception:                                   # noqa: BLE001
            pass
        # ---- 内存收尾（三步，顺序有讲究）----
        # ① 预览 worker 那份会话：导出结束后它就是纯占着的东西（显存 2.1~2.4GB /
        #    内存 1.3GB），而导出用的是本进程那份 —— 两份互不相干，必须分开还。
        #    DFL_FUSED_FREE_PREVIEW=0 可保留（下一次预览更快，代价是显存一直占着）。
        try:
            _rel = ({"did": False, "why": "DFL_FUSED_FREE_PREVIEW=0"}
                    if str(os.environ.get("DFL_FUSED_FREE_PREVIEW") or "1").strip().lower()
                    in ("0", "false", "no", "off") else _free_preview_sessions("快速导出结束"))
        except Exception as e:                              # noqa: BLE001
            _rel = {"did": False, "why": repr(e)}
        # ② 断完引用先收一遍垃圾：循环引用（闭包/异常栈）只有 gc 能收，光 drop 不够
        try:
            import gc as _gc
            _gc.collect()
            _gc.collect()
        except Exception:                                   # noqa: BLE001
            pass
        # ③ 先量「收尾后、回收前」—— 这才是**真的还剩下的**（对象都释放了，但堆把页
        #    攥在进程里）；再把工作集交给系统。两笔分开打，谁也不许拿回收冒充释放。
        mem1 = _proc_mem_mb()
        _vid_log("快速导出 内存（收尾后 · 回收前）：%s" % _mem_delta(mem0, mem1))
        try:
            _trim = _trim_ws("快速导出收尾")
        except Exception as e:                              # noqa: BLE001
            _trim = {"did": False, "why": repr(e)}
        mem2 = _proc_mem_mb()
        _vid_log("快速导出 内存（导出前后 · 最终）：%s" % _mem_delta(mem0, mem2))
        if (_trim or {}).get("did"):
            _vid_log("快速导出 内存：工作集回收 RSS %s → %s MB（提交 %s → %s MB）—— 回收只还"
                     "工作集，提交只随对象释放而降；两者不是一回事"
                     % (_fmt_mb((_trim.get("before") or {}).get("rss")),
                        _fmt_mb((_trim.get("after") or {}).get("rss")),
                        _fmt_mb((_trim.get("before") or {}).get("private")),
                        _fmt_mb((_trim.get("after") or {}).get("private"))))
        if (_rel or {}).get("did"):
            _vid_log("快速导出 内存：预览 worker 已卸下常驻 dfm（%d 个 · %.0f ms）—— 那是另开的"
                     "一份会话，导出用不到它"
                     % (int(_rel.get("unloaded") or 0), float(_rel.get("ms") or 0)))
        else:
            _vid_log("快速导出 内存：预览 worker 未卸（%s）" % str((_rel or {}).get("why") or "-")[:120])
        _vid_close_log()


def _start_video_export(body: dict):
    """POST /api/merge/export 的视频分支。**两条路，按 fused 选**：

      · fused=1（默认，快速导出）：_fused_run 的全内存流水线 —— 解码 → 检测 → 对齐/归组
        → dfm → XSeg → 合成 → 编码 同时跑，段间只有有界队列，**不落中间产物**
        （<视频名>_cache/ 一个都不建）。原型（app/tools/bench/fused_pipeline_bench.py）同段
        1080p 120 帧量到 20.55 帧/秒（分段 7.20）；这份生产实现的验收见
        app/tools/accept/fused_export_accept.py：同一段片 18.21 帧/秒（分段 7.20 → 2.5x），
        换 fast 档检测器 20.87 —— 硬底是 retinaface_10g 单帧 47~50ms（它跑在
        onnxruntime 上，workspace/_engines 里那个 retinaface TRT 引擎 app 层没接）。
        代价：不支持续跑（取消就整段重跑）。
      · fused=0（分段链）：切帧 → 切脸 → 换脸 → 遮罩 → 合成 → 导出，每步独立产物、
        可单跑可续跑（<视频名>_cache/ 成功后删除）。留给排查和对照。

    两条路的进度都镜像到 /api/merge/export/status（老前端和既有验收脚本继续能用），
    fused 还额外回显 frame_budget / 逐段耗时。
    """
    body = _as_dict(body)
    fused = _fused_on(body)
    cfg = _fused_cfg(body) if fused else _stage_cfg(body, "frames", chain=True)
    if not fused:
        cfg["stages"] = list(_STAGE_ORDER)
    # ---- 显存闸门：**在这里**判（HTTP 400），不能在 worker 线程里判 ----
    # 线程里抛 _BadInput 只会变成「任务失败」，前端拿不到 400，也拿不到那句
    # 「需要 X GB / 可用 Y GB / 建议」。真正的加载在 _fused_run / _stage_worker 里，
    # 这里只算账（_vram_plan 不加载任何模型）—— 拒绝时**不建 job、不开线程**。
    #
    # 两条路都要判（用户口径，截图 2）：
    #   · fused  —— 帧预算 + dfm 批是它的两个旋钮，按 mode=frames 算；
    #   · 分段链 —— 一次一帧一张脸，没有帧缓冲，按 mode=none（一份会话）算。
    _dfms = (cfg.get("dfm")
             or [m.get("dfm") for m in cfg.get("single_models") or []])
    # force = 用户在拒绝面板上点了「仍然尝试（可能 OOM）」：装不下也放行最小档。
    # 只放行、不改账（plan.forced=True 且 need/avail 照旧），真 OOM 由 worker 里的
    # _vram_oom_rollback 收现场并回到拒绝态 —— 见 _fused_worker / _stage_worker。
    _force = _bool_val(body.get("force"), False)
    # dry_run 是预检（前端拖一次滑杆就打一次）：**不卸**预览模型，靠「可回收计入可用」
    # 拿到同样的结论；真起手这一拍才卸（reclaim_first），因为那时模型才真的要腾地方。
    _dry = _bool_val(body.get("dry_run"), False)
    if fused:
        plan = _vram_guard(_dfms,
                           {"mode": "frames", "frames": int(cfg.get("frame_budget_want") or 0),
                            "batch": int(cfg.get("fused_batch") or 0), "procs": 1,
                            "reclaim_first": (not _dry)},
                           label="快速导出", force=_force)
    else:
        plan = _vram_guard(_dfms, {"mode": "none", "procs": 1,
                                   "reclaim_first": (not _dry)},
                           label="分段导出", force=_force)
    cfg["vram_plan"] = plan
    cfg["vram_force"] = bool(plan.get("forced"))
    if _dry:
        # 预检（前端在起 job 之前问一次）：只回计划，不建 job、不开线程、不碰 status。
        return _ok(dry_run=True, fused=bool(fused), stages=(["fused"] if fused
                                                           else list(_STAGE_ORDER)),
                   video=str(cfg["video"]), output=str(cfg["out"]),
                   vram_plan=dict(plan), need_mb=plan.get("need_mb"),
                   avail_mb=plan.get("avail_mb"), free_mb=plan.get("free_mb"),
                   suggest=list(plan.get("suggest") or []),
                   frames=int(plan.get("frames") or 0),
                   batch=int(plan.get("batch") or 0),
                   on_cpu=bool(_vram_cpu_on()))
    with _STAGE_LOCK:
        if _STAGE_JOB.get("running"):
            raise _BadInput("已经有阶段在跑（%s）—— 等它结束或点取消"
                            % _STAGE_LABEL.get(_STAGE_JOB.get("stage"), "?"), 409)
    started = time.time()
    with _merge_job_lock:
        if _merge_job.get("running"):
            raise _BadInput("已经在导出（%s）—— 等它结束或点取消"
                            % (_merge_job.get("out") or "上一个任务"), 409)
        _merge_job.update({
            "running": True, "active": True,
            "log": [], "ok": None, "error": None, "error_detail": "",
            "out": str(cfg["out"]), "pct": 0,
            "msg": ("启动快速导出（融合流水线）…" if fused else "启动导出…"),
            "stage": 0,
            "cancelled": False, "pct_at_failure": None, "started_at": started,
            "finished_at": None, "fps": float(cfg.get("fps") or 0.0), "workers": 1,
            "encoder": "libx264", "hwaccel": "", "dfm": str(cfg["dfm"]),
            "mode": "video", "video": str(cfg["video"]), "total": int(cfg.get("total") or 0),
            "frames_done": 0, "frames": [], "stats": {}, "cache": {}, "encode": {},
            "fused": bool(fused), "frame_budget": None, "cache_dir": str(cfg.get("cache") or ""),
            "vram_plan": dict(cfg.get("vram_plan") or {}),
        })
    with _STAGE_LOCK:
        _STAGE_JOB.update({
            "running": True, "ok": None, "cancelled": False, "error": "", "error_detail": "",
            "phase": "running",
            "stage": ("fused" if fused else "frames"),
            "stage_index": (-1 if fused else 0), "stage_pct": 0,
            "pct": 0,
            "msg": ("启动快速导出（融合流水线）…" if fused else "启动导出（分段）…"),
            "video": str(cfg["video"]),
            "cache_dir": str(cfg.get("cache") or ""), "out_video": str(cfg["out"]),
            "frames_total": int(cfg.get("range_total") or cfg.get("total") or 0),
            "range": dict(cfg.get("range") or {}), "frames_done": 0,
            "frame_fps": 0.0, "eta_s": None, "swapped": 0, "copied": 0, "no_face": 0,
            "stages_done": ([] if fused else _stage_done_list(cfg["meta"])),
            "log": [], "meta": ({} if fused else cfg["meta"]),
            "seconds": {}, "skipped": [], "cache_removed": False,
            "counts": ({"frames": 0, "faces": 0, "pred": 0, "masks": 0} if fused
                       else _stage_counts(cfg["cache"])),
            "started_at": started,
            "finished_at": None, "chain": True,
            "keep_cache": bool(cfg.get("keep_cache")), "force": bool(cfg.get("force")),
            "frames": [], "fused": bool(fused), "frame_budget": None, "resumed": False,
            "vram_plan": dict(cfg.get("vram_plan") or {}),
            "eff": None, "eff_warnings": [], "warnings": [],
        })
    # ---- 生效配置：一行 [eff] + 一致性自检 ----
    # 用户要的就是这一行：把「我设的」和「它在跑的」摆在一起，对不上就 [eff-warn]。
    eff = None
    try:
        eff = _eff_publish("/api/merge/export", ("fused" if fused else "staged"), body, cfg)
        patch = _eff_status_patch(eff)
        with _merge_job_lock:
            _merge_job.update(patch)
        with _STAGE_LOCK:
            _STAGE_JOB.update(patch)
    except Exception as e:                                  # noqa: BLE001
        print("[eff] 生成失败（不影响这一单）：%s: %s"
              % (e.__class__.__name__, str(e)[:200]), flush=True)
    cfg["log_file"] = _vid_open_log(Path(cfg["out"]).with_suffix(".log"))
    # 生效配置也贴进产物日志（<out>.log）：这是用户报 bug 时最常附的那份文件
    if eff:
        try:
            _vid_log("[eff] " + _eff.line_of(eff))
            for _w in (eff.get("warnings") or []):
                _vid_log("[eff-warn] " + str(_w.get("text") or _w))
        except Exception:                                   # noqa: BLE001
            pass
    # 帧图编码口径（两条路同一句）：PNG 无损 / JPEG 固定 quality=100 + 4:4:4。
    # 设置页那颗「质量」旋钮已经删掉 —— 后端不再从界面读它，日志里照旧回显真实值。
    _vid_log("帧图编码：PNG 无损（compression level %s）/ JPEG quality=%d + 4:4:4"
             "（界面没有该旋钮，固定质量；PNG 不受影响）"
             % (cfg.get("png_level"), int(cfg.get("jpeg_quality")
                                          or _STAGE_JPEG_QUALITY)))
    if fused:
        threading.Thread(target=_fused_worker, args=(cfg,), daemon=True,
                         name="merge-export-fused").start()
        _vid_log("导出链路（快速导出 · 融合流水线）：%s -> %s"
                 % (cfg["video"], cfg["out"]))
        _vid_log("中间产物：不落 <视频>_cache/（帧只在 %%TEMP%% 里经过，成功/取消都删）")
        _vid_log("续跑：不支持 —— 取消就是整段重跑；要续跑请把「快速导出」关掉走分段链")
        _vid_log("帧范围：第 %d..%d 帧（%d 帧）%s"
                 % (int((cfg.get("range") or {}).get("start") or 1),
                    int((cfg.get("range") or {}).get("end") or 0),
                    int(cfg.get("range_total") or 0),
                    "· 全片" if cfg.get("range_full") else "· 只处理这一段"))
        return _ok(mode="video", fused=True, staged=False, stages=["fused"],
                   video=str(cfg["video"]), output=str(cfg["out"]),
                   output_dir=str(cfg["out"]), cache_dir="",
                   no_cache=True, resumable=False,
                   frame_budget=None, frame_budget_want=int(cfg.get("frame_budget_want") or 0),
                   vram_plan=dict(cfg.get("vram_plan") or {}),
                   on_cpu=bool(_vram_cpu_on()),
                   dfm=str(cfg["dfm"]), model=str(cfg["model"]),
                   settings=dict(cfg.get("settings") or {}),
                   detector=str(cfg["detector"]), thresh=float(cfg.get("thresh") or 0.5),
                   xseg=str(cfg["xseg"]), fps=float(cfg.get("fps") or 0.0),
                   range=dict(cfg.get("range") or {}),
                   range_key=str(cfg.get("range_key") or ""),
                   face_threshold=cfg.get("face_threshold"),
                   library=cfg.get("library") or "",
                   models=[{"name": m.get("name"), "dfm": m.get("dfm"),
                            "faces": list(m.get("faces") or [])}
                           for m in cfg["single_models"]])
    threading.Thread(target=_stage_worker, args=(cfg,), daemon=True,
                     name="merge-export-video").start()
    _vid_log("导出链路（分段）：%s -> 切帧 -> 切脸 -> 合成 -> 导出 -> %s"
             % (cfg["video"], cfg["out"]))
    _vid_log("cache：%s（导出成功后整目录删除）" % cfg["cache"])
    return _ok(mode="video", fused=False, staged=True, stages=list(_STAGE_ORDER),
               video=str(cfg["video"]), output=str(cfg["out"]),
               output_dir=str(cfg["out"]), cache_dir=str(cfg["cache"]),
               dfm=str(cfg["dfm"]), model=str(cfg["model"]),
               settings=dict(cfg.get("settings") or {}),
               detector=str(cfg["detector"]), thresh=float(cfg.get("thresh") or 0.5),
               xseg=str(cfg["xseg"]), fps=float(cfg.get("fps") or 0.0),
               face_threshold=cfg.get("face_threshold"), library=cfg.get("library") or "",
               models=[{"name": m.get("name"), "dfm": m.get("dfm"),
                        "faces": list(m.get("faces") or [])}
                       for m in cfg["single_models"]])


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

