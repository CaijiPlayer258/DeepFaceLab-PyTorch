# -*- coding: utf-8 -*-
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
        q = int(it.get("jpeg_quality") or 100)
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
            # ★ box 一起回：父进程的超分（_sr_apply_resp）要用人脸框才知道
            #   该把哪一块送去 GPEN。以前这里只有 dfm/size，父进程 resp[box] 恒空，
            #   _sr_frame_bytes 解包失败 → 静默跳过 —— 表现是「开了超分没效果」。
            # ★ 遮罩一起回：父进程的超分要「只作用在遮罩内」（不然头发/背景
            #   一起被锐化）。这份 _fmask 是 MergeMaskedFace 顺手算出来的，
            #   父进程再用它一次 —— 比在那边重跑 XSeg 省 1.5~4 ms/张。
            _mk_b64 = ""
            try:
                if _fmask is not None and getattr(_fmask, "size", 0):
                    _mk8 = np.clip(np.asarray(_fmask, np.float32) * 255.0, 0, 255).astype(np.uint8)
                    if _mk8.shape[:2] != (256, 256):
                        _mk8 = cv2.resize(_mk8, (256, 256), interpolation=cv2.INTER_LINEAR)
                    _ok8, _buf8 = cv2.imencode(".png", _mk8, [int(cv2.IMWRITE_PNG_COMPRESSION), 3])
                    if _ok8:
                        import base64 as _b64
                        _mk_b64 = _b64.b64encode(_buf8.tobytes()).decode("ascii")
            except Exception:                          # noqa: BLE001
                _mk_b64 = ""
            per = [{"dfm": dfm, "size": size, "box": list(it.get("box") or []),
                    "mask": _mk_b64}]
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
                # ★ 遮罩随 per 回传：父进程的超分要「只作用在遮罩内」（不然头发/背景
                #   一起被锐化），而这份 _fmask 是 MergeMaskedFace 顺手算出来的 ——
                #   父进程再用它一次，比在那边重跑 XSeg 省 1.5~4 ms/张。
                _mk_b64 = ""
                try:
                    if _fmask is not None and getattr(_fmask, "size", 0):
                        _mk8 = np.clip(np.asarray(_fmask, np.float32) * 255.0, 0, 255).astype(np.uint8)
                        if _mk8.shape[:2] != (256, 256):
                            _mk8 = cv2.resize(_mk8, (256, 256), interpolation=cv2.INTER_LINEAR)
                        _ok8, _buf8 = cv2.imencode(".png", _mk8, [int(cv2.IMWRITE_PNG_COMPRESSION), 3])
                        if _ok8:
                            import base64 as _b64
                            _mk_b64 = _b64.b64encode(_buf8.tobytes()).decode("ascii")
                except Exception:                          # noqa: BLE001
                    _mk_b64 = ""
                per.append({"dfm": f_dfm, "size": size_k,
                            "ms": round((time.time() - t_k) * 1000.0, 1),
                            "box": list(fc.get("box") or []),
                            "mask": _mk_b64})
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
                 "box": (per[0].get("box") if per else []),
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
        # 超分滑块随每一发请求走（env 在 spawn 时就定死了，改不动）
        # 人脸超分滑块随请求走：worker 是**独立解释器**，spawn 之后父进程再改 env 它也看不见，
        # 所以每一发都按这一发带的 sr 重设一次（预览才看得见 GPEN 的效果）。
        # ★ 这个 worker 是拷到 %TEMP%/.../preview_worker_<sha>.py 当**脚本**跑的 ——
        #   相对 import（from .knobs）在脚本模式下必然 ImportError，而它整段包在
        #   except 里，失败得一声不响。于是「超分滑块随请求走」这件事从来没生效过。
        #   先按绝对名试（脚本模式的 sys.path 里父进程已经塞了 app/），再退相对。
        _apply_sr = None
        for _imp in (lambda: __import__("server.backend_merge.knobs", fromlist=["apply_sr"]).apply_sr,
                     lambda: __import__("backend_merge.knobs", fromlist=["apply_sr"]).apply_sr):
            try:
                _apply_sr = _imp()
                break
            except Exception:                              # noqa: BLE001
                continue
        if _apply_sr is None:
            try:
                from .knobs import apply_sr as _apply_sr   # 正常包内 import 时的兜底
            except Exception:                              # noqa: BLE001
                _apply_sr = None
        if _apply_sr is not None:
            try:
                _apply_sr(req.get("sr"))
            except Exception:                              # noqa: BLE001
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
