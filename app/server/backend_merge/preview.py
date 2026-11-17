# -*- coding: utf-8 -*-
"""常驻预览 worker（子进程）、worker 池、热模型、序列帧预览与挑脸。

从 app/server/routes_merge.py 按关注点原样切出来的（16741 行的单文件 → 按域分家），
函数体 / 注释 / 日志文案逐字节照抄，只补 import；同包之间的 import 放在文件末尾
（见文末「环上的依赖」），另有一处 __file__ 相对路径下标跟着目录层级改了。
旧行号 → 现文件 的对照表：app/server/backend_merge/MAPPING.md。
"""
from __future__ import annotations
import numpy as np

import atexit
import collections
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
from pathlib import Path
from .._common.env import (
    _env_secs,
)
from .._common.http import (
    _BadInput, _int_val,
)
from .._common.proc import (
    _kill_proc_tree,
)
from ._shared import (
    _norm_path,
)


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
        # （预览 worker 只建 ORT 会话，没有第二个缓存要放；见 _load 上面的注释。）
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

    # 预览 worker 跑在**4.0 引擎的 python**（独立解释器 + MergeStudio + engine 的 torch 2.7），
    # 不 import 4.1 的 backend_models —— 所以这里仍旧只建 ORT 会话，不跟 DFL_DFM_BACKEND 走。
    # 代价：导出（torch+bf16）与预览（ORT fp32）同一帧差 ≤5 LSB（真脸 + 成片实测，见
    # app/tools/bench/DFM_TORCH_BF16_REPORT.md §5）；stage6 的 g 段（成片 vs 预览 ≤8）实测
    # 仍远在容差内。等 4.0 引擎那边的 torch 版本与这边对齐了再谈统一。
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
        # 人脸超分滑块随请求走：worker 是**独立解释器**，spawn 之后父进程再改 env 它也看不见，
        # 所以每一发都按这一发带的 sr 重设一次（预览才看得见 GPEN 的效果）。
        try:
            from .knobs import apply_sr as _apply_sr
            _apply_sr(req.get("sr"))
        except Exception:                                  # noqa: BLE001
            pass
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


# ============================================================
# 环上的依赖：同包 import 放在文件末尾
# ------------------------------------------------------------
# 拆分前这些都是同一个模块里的全局名，可以随手互相引用；拆开后 preview 与 _app_mod、_engine_python、_engine_root、app、compose、media_frames、vram
# 互相要用对方的东西（合成 / 流水线 / 显存账本本来就咬在一起）。放在文件末尾
# import 是唯一静态成立的做法：任何模块被 import 时自己的 def 和赋值都已跑完，
# 对方再来取名字一定拿得到，先 import 谁都一样。
# ============================================================
from .. import app as _app_mod

from .._paths import engine_python as _engine_python

from .._paths import engine_root as _engine_root

from ..app import (
    WORKSPACE,
)

from .compose import (
    _det_run_lock, _detect_boxes, _detector, _detector_val, _frame_landmarks,
)

from .._media.frames import (
    _grab_frame_at,
)

from .vram import (
    _VRAM_ENV_CPU, _vram_cpu_on, _vram_free_mb, _vram_last, _vram_measure_begin,
)
