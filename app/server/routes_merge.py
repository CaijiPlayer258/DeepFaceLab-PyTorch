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
# MergeStudio.core.config.mode_dict
_MERGE_MODES = ("overlay", "original", "hist-match", "seamless",
                "seamless-hist-match", "raw-rgb", "raw-predict", "debug")
# MergeStudio.core.config.ctm_dict
_COLOR_TRANSFER = ("none", "rct", "lct", "mkl", "mkl-m", "idt", "idt-m",
                   "sot-m", "mix-m", "mt", "lut")
_IMG_FORMATS = {"jpg": "jpg", "jpeg": "jpg", "png": "png"}


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
    mode = str(raw.get("mode") or "overlay").strip().lower()
    if mode not in _MERGE_MODES:
        raise _BadInput("settings.mode 只认 %s，收到 %r"
                        % (" / ".join(_MERGE_MODES), raw.get("mode")))
    out["mode"] = mode
    out["mask_mode"] = _mask_mode_val(raw.get("mask_mode"), 6)

    ct = raw.get("color_transfer_mode", raw.get("color_transfer", "none"))
    ct = str(ct if ct is not None else "none").strip().lower() or "none"
    if ct not in _COLOR_TRANSFER:
        raise _BadInput("settings.color_transfer 只认 %s，收到 %r"
                        % (" / ".join(_COLOR_TRANSFER), raw.get("color_transfer")))
    out["color_transfer_mode"] = ct
    out["color_transfer"] = ct

    erode = raw.get("erode_mask_modifier", raw.get("erode_mask"))
    blur = raw.get("blur_mask_modifier", raw.get("blur_mask"))
    out["erode_mask_modifier"] = _int_val("settings.erode_mask", erode, -100, 100, 0)
    out["blur_mask_modifier"] = _int_val("settings.blur_mask", blur, -100, 100, 0)

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


@router.get("/api/merge/frame/{idx}")
@_api
def merge_frame(idx: int, w: int = 900):
    maxw = max(200, min(int(w), 1600))
    if idx < 0:                          # 参数错先说参数错，别被“没打开素材”盖住
        raise _BadInput("帧号不能是负数：%d" % idx)
    snap = _cache_snap()
    mode = _check_cache_ready(snap)
    idx = _check_idx(snap, idx)

    if mode == "project":
        names = snap.get("names") or []
        if not names:
            raise _BadInput("还没有打开项目", 409)
        fp = Path(str(snap.get("srcdir"))) / names[idx]
        if not fp.is_file():
            raise _BadInput("图片不见了：%s" % fp, 404)
        try:
            import cv2
            img = _imread_any(fp)
            if img is None:
                raise RuntimeError("解不开")
            h0, w0 = img.shape[:2]
            if w0 > maxw:
                img = cv2.resize(img, (maxw, int(h0 * maxw / w0)),
                                 interpolation=cv2.INTER_AREA)
            good, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
            if not good:
                raise RuntimeError("编码失败")
        except Exception as e:                   # 文件在、但解不出来（半张/损坏）
            raise _BadInput("这张图解不开（不是有效图片？）：%s（%s）"
                            % (fp.name, str(e)[:120]), 422)
        return _ok(idx=idx, total=len(names), name=names[idx],
                   png="data:image/jpeg;base64," + _b64.b64encode(buf.tobytes()).decode())

    path = str(snap.get("path"))
    try:
        png = _video_frame_jpeg(path, idx, maxw=maxw)
    except Exception as e:
        raise _BadInput("取帧失败（ffmpeg）：%s" % str(e)[:200], 500)
    return _ok(idx=idx, total=int(snap.get("frames") or 0),
               png="data:image/jpeg;base64," + _b64.b64encode(png).decode())


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


@router.get("/api/merge/merged/{idx}")
@_api
def merge_merged_frame(idx: int, w: int = 900):
    """从「最近一次导出的成片」里取一帧，给工作台的「合成」标签用。"""
    if idx < 0:
        raise _BadInput("帧号不能是负数：%d" % idx)
    maxw = max(200, min(int(w), 1600))

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
        raise _BadInput("还没有导出的成片 —— 先点一次导出", 409)

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

    h0, w0 = frame.shape[:2]
    if w0 > maxw:
        frame = cv2.resize(frame, (maxw, int(h0 * maxw / w0)), interpolation=cv2.INTER_AREA)
    good2, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
    if not good2:
        raise _BadInput("成片这一帧编码失败", 500)
    return _ok(idx=int(idx), total=total, out=str(out),
               png="data:image/jpeg;base64," + _b64.b64encode(buf.tobytes()).decode())


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


def _detector(name):
    key = str(name or "det_500m").strip() or "det_500m"
    try:
        with _det_load_lock:              # 加载要几百毫秒，别让并发请求各建一份
            return _get_detector(key)
    except HTTPException as e:
        raise _BadInput(str(e.detail), e.status_code)
    except Exception as e:
        raise _BadInput("检测器加载失败（%s）：%s" % (key, str(e)[:160]), 500)


@router.get("/api/merge/detect/{idx}")
@_api
def merge_detect(idx: int, model: str = "det_500m", thresh: float = 0.5):
    """对当前帧做人脸检测，返回原图坐标的框。视频 / 项目两种模式都支持。"""
    t = _float_val("thresh", thresh, 0.0, 1.0, 0.5)
    frame, total = _frame_image(idx)
    det = _detector(model)
    with _det_run_lock:                    # thresh 是共享检测器上的可变字段
        det.thresh = t
        try:
            boxes = det.detect(frame)
        except Exception as e:
            raise _BadInput("检测失败：%s" % str(e)[:200], 500)
    h, w = frame.shape[:2]
    return _ok(idx=int(idx), total=int(total or 0), width=int(w), height=int(h),
               model=model, thresh=t, boxes=boxes)


@router.get("/api/merge/analyze")
@_api
def merge_analyze(step: int = 12, model: str = "det_500m", thresh: float = 0.5):
    """按步长扫全片，统计每帧的人脸数，给时间轴的人脸轨用。
    486 帧 / step=12 → 41 次推理，约 20 秒。"""
    step = _int_val("step", step, 1, 500, 12)
    t = _float_val("thresh", thresh, 0.0, 1.0, 0.5)
    snap = _cache_snap()
    total = int(snap.get("frames") or 0)
    _check_cache_ready(snap)
    if total <= 0:
        raise _BadInput("还没有打开素材（先 /api/merge/open 或 /api/merge/prepare）", 409)
    det = _detector(model)

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
    return _ok(total=total, step=step, scanned=scanned, model=model, thresh=t,
               marks=marks, truncated=bool(broken), error=broken)


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
        t0 = time.time()
        try:
            raw = base64.b64decode(req.get("frame") or "")
            frame = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
            if frame is None:
                raise RuntimeError("帧解码失败")

            c = req.get("config") or {}
            cfg = MergerConfigMasked(
                face_type=FaceType.WHOLE_FACE,
                mode=str(c.get("mode") or "overlay"),
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
            ok, buf = cv2.imencode(".jpg", out, [int(cv2.IMWRITE_JPEG_QUALITY), q])
            if not ok:
                raise RuntimeError("JPEG 编码失败")
            _say({"id": rid, "ok": True,
                  "size": (per[0]["size"] if per else 256),
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
        from ._ortcuda import _candidate_roots
        for root in _candidate_roots():
            for rel in ("modelhub/onnx/InsightFace2d106/InsightFace2D106.onnx",
                        "DeepFaceLab-PyTorch/modelhub/onnx/InsightFace2d106/InsightFace2D106.onnx"):
                cand = Path(root) / rel
                if cand.is_file():
                    _LM106["m"] = InsightFace106(str(cand),
                                                 providers=["CPUExecutionProvider"])
                    return _LM106["m"]
    except Exception:
        traceback.print_exc()
    _LM106["m"] = None
    return None


def _frame_landmarks(frame, det_box, box_xyxy):
    """检测框/5 点 → 原图坐标的 68 点。

    跟 _align_stream（抽帧+对齐那条路）同一个套路：5 点先求 512 对齐，再让
    2D106 回归出 68 点，最后用 image_to_face_mat 反变换回原图。2D106 不可用时
    退回「5 点 + 模板」，只是精度差一档。
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
            try:
                from ._align import landmarks68_from_5
                lm68 = landmarks68_from_5(np.array(kps, dtype=np.float64))
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
    stage_fn("faces")
    t_lm = time.time()
    plan, skipped = [], []
    for i, b in enumerate((boxes or [])[:_PREVIEW_MAX_SWAPS]):
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
        return {"jpg": _jpeg_bytes(frame, 88), "box": box_wh, "has_face": True,
                "swapped": False, "timings": timings, "faces": [], "models_used": [],
                "note": "检出人脸但 68 点算不出来 —— 只回原帧"}

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
        "frame": _b64.b64encode(_jpeg_bytes(frame, _PREVIEW_SEND_QUALITY)).decode("ascii"),
        "faces": [{"dfm": p["dfm"], "landmarks": p["landmarks"], "box": p["box"]}
                  for p in plan],
        "load": load,
        "xseg": str(job.get("xseg") or ""),
        "config": dict(job["config"]),
        "jpeg_quality": _PREVIEW_QUALITY,
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
    return {"jpg": jpg, "box": box_wh, "has_face": True, "swapped": True,
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

    def _stage(name):
        state["stage"] = name
        if stop.is_set():
            raise _BadInput("预览已取消", 499)

    _stage("grab")
    t0 = time.time()
    frame = _grab_frame_at(job["video"], job["t"], stop)
    timings["grab_ms"] = round((time.time() - t0) * 1000.0, 1)

    _stage("detect")
    t0 = time.time()
    det = _detector(job["detector"])
    with _det_run_lock:                        # thresh 是共享检测器上的可变字段
        det.thresh = float(job["thresh"])
        try:
            boxes = det.detect(frame)
        except Exception as e:
            raise _BadInput("检测失败：%s" % (str(e)[:200],), 500)
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
        return {"jpg": _jpeg_bytes(frame, 88), "box": None, "has_face": False,
                "swapped": False, "timings": timings, "note": "这一帧没检出人脸"}

    if job.get("models"):                      # 多模型：每张脸各用各的模型
        return _preview_multi(job, frame, boxes, box_wh, timings, _stage, state)

    _stage("landmarks")
    t0 = time.time()
    lm68 = _frame_landmarks(frame, best, box_xyxy)
    timings["landmarks_ms"] = round((time.time() - t0) * 1000.0, 1)
    if lm68 is None:
        return {"jpg": _jpeg_bytes(frame, 88), "box": box_wh, "has_face": True,
                "swapped": False, "timings": timings,
                "note": "检出人脸但 68 点算不出来 —— 只回原帧"}

    _stage("swap")
    state["stage"] = "queued"          # worker 一次只跑一帧：排队时间不算进 15s
    t0 = time.time()
    payload = {
        "frame": _b64.b64encode(_jpeg_bytes(frame, _PREVIEW_SEND_QUALITY)).decode("ascii"),
        "landmarks": np.asarray(lm68, dtype=np.float64).round(2).tolist(),
        "dfm": str(job["dfm"]),
        "xseg": str(job.get("xseg") or ""),
        "box": list(box_wh),
        "config": dict(job["config"]),
        "jpeg_quality": _PREVIEW_QUALITY,
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
    return {"jpg": jpg, "box": box_wh, "has_face": True, "swapped": True,
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
    detector = str(body.get("detector") or "det_500m").strip() or "det_500m"
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
                                        "stop": stop, "state": state})
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
    """检测器是不是已经在父进程里了 —— 和 app._get_detector 看同一张缓存表。"""
    cache = getattr(_app_mod, "_det_cache", None) or {}
    key = str(name or "det_500m").strip()
    low = key.lower().replace("-", "_")
    if low.startswith("tinymog"):
        return "tinymog" in cache
    alias = getattr(_app_mod, "_DET_ALIAS", {}).get(low, key)
    return alias in cache


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
    det_name = str(body.get("detector") or "det_500m").strip() or "det_500m"

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
