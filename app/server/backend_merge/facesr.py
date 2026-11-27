# -*- coding: utf-8 -*-
"""人脸超分（GPEN-BFR-512）—— 合成器的可选后处理。

放在**帧空间**做，不在芯片上做
--------------------------------
芯片是 224~416，GPEN 要 512：先放大到 512、再缩回去，等于「先糊一遍再加锐」。
所以这里拿的是**合成之后那帧里的脸区域**（原生分辨率），只重采样一次：
裁 → 512 → 推理 → 缩回 → 按遮罩 alpha 混回。

强度是**连续滑块**，不是开关（GPEN 自身没有强度参数）
------------------------------------------------
GPEN 的输出是确定的 —— 模型里没有"变阻器"。真正的连续旋钮是**混回比例**：
      out = 原图 × (1 - α) + SR × α
α = 0 就是关；α = 1 是整脸替换（会把目标人特征洗掉，实测 ArcFace 相似度明显下降）。

滑块 0..100 分两段（两档模型，颜色也分两段）：
      0        关
      1..50    用 GPEN-256，α = lv/50
      51..100  用 GPEN-512，α = (lv-50)/50
所以 60 不是"512 的 60%"，而是"512 满力的 20%" —— 滑块是连续的，模型在 50 处换档。

后端优先级：TRT 引擎 → onnxruntime(CUDA) → 都不行就**明确告警并跳过**（不静默）
----------------------------------------------------------------------------
  DFL_SR=0..100                 滑块值（默认 0 = 关；兼容 off/low/mid/high = 0/35/60/100）
  DFL_SR_BACKEND=auto|trt|onnx   后端（auto = 有 engine 用 engine）
  DFL_SR_SIZE=256|512            强制尺寸（排障用；正常由滑块决定）
  DFL_SR_MODEL=<onnx 路径>        覆盖模型路径（不给就按 size 选）
  engine 放 weights/_engine/gpen_bfr_512.<sm>.<prec>.engine（用 app/tools/models/build_sr_engine.py 编）

为什么不做 torch 后端：GPEN 官方权重是 ONNX 更省事（torch 版要 vendor 它的网络定义 + StyleGAN2 的
自定义 op），而 bf16/CUDA Graph 那套优化 TRT 引擎天然就有（--bf16）。
"""
from __future__ import annotations

import os
import threading
import time
from pathlib import Path

# 档位→百分比（旧写法继续可用）；真身是 **0..100 的滑块值**，见 level()
# 兼容旧写法（以前是离散档位）；数值就是滑块位置。mid 保持"256 满力的六成"这层含义不变，
# 免得历史实测数据（mid=0.6、256）对不上。
_PREF = {"off": 0, "low": 18, "mid": 30, "high": 100}
_FULL = 50.0      # 滑块 0..100 的中点：<=50 用 256 模型，>50 用 512
_LOCK = threading.Lock()
# 会话缓存按 size 分家：256 与 512 是两个不同引擎，共用一个就会出现
# 「引擎不接受输入形状 (1,3,512,512)（实际生效 (1,3,256,256)）」→ 静默跳过。
_SESS = {}          # size -> {"obj", "backend", "name", "why"}
_SIZE_DEFAULT = 256          # 256 版单次约 1/4 成本；512 版质量更好但本机吃不下


def _log(msg: str) -> None:
    print("[sr] " + msg, flush=True)


def level() -> int:
    """滑块值 0..100。0 = 关；<=50 走 256 模型，>50 走 512 模型。

    兼容旧写法：DFL_SR=off/low/mid/high 分别等价于 0/35/60/100。
    """
    raw = str(os.environ.get("DFL_SR") or "0").strip().lower()
    if raw in _PREF:
        return int(_PREF[raw])
    try:
        v = float(raw)
    except (TypeError, ValueError):
        return 0
    return max(0, min(100, int(round(v))))


def mode() -> str:
    """给日志/回显用的短标签（不参与计算）。"""
    lv = level()
    return "off" if lv <= 0 else ("%d" % lv)


def strength() -> float:
    """混合强度 0..1 —— 这是 GPEN 之外的**连续旋钮**（模型本身没有强度参数）。

    左半段（1..50）用 256，强度 = lv/50；右半段（51..100）用 512，强度 = (lv-50)/50。
    所以右半段的起点是「512 刚开、强度很小」，不是「512 满力」——滑块是连续的。
    """
    lv = level()
    if lv <= 0:
        return 0.0
    if lv <= _FULL:
        return lv / _FULL
    return (lv - _FULL) / _FULL


def enabled() -> bool:
    return level() > 0


def _weights_dir() -> Path:
    from .._paths import weights_dir
    return Path(weights_dir())


def size() -> int:
    """工作分辨率：滑块 <=50 用 256，>50 用 512。DFL_SR_SIZE 可显式覆盖（排障用）。"""
    forced = str(os.environ.get("DFL_SR_SIZE") or "").strip()
    if forced:
        try:
            return 512 if int(float(forced)) >= 384 else 256
        except (TypeError, ValueError):
            pass
    return 256 if level() <= _FULL else 512


def _model_path() -> Path:
    p = str(os.environ.get("DFL_SR_MODEL") or "").strip()
    if p:
        return Path(p)
    return _weights_dir() / ("gpen_bfr_%d.onnx" % size())


def _engine_path() -> Path | None:
    """weights/_engine/ 下与本机 SM 匹配的 GPEN 引擎（没有就 None）。"""
    try:
        from ..backend_models.engines import sm_of
    except Exception:                                       # noqa: BLE001
        sm_of = None
    roots = [_weights_dir() / "_engine", _weights_dir() / "engine"]
    try:
        sm = int(sm_of()) if sm_of else 0
    except Exception:                                       # noqa: BLE001
        sm = 0
    for root in roots:
        if not root.is_dir():
            continue
        for suf in (".bf16.engine", ".fp32.engine", ".engine"):
            cand = root / ("gpen_bfr_%d.%d%s" % (size(), sm, suf))
            if cand.is_file():
                return cand
        hits = sorted(root.glob("gpen_bfr_%d*.engine" % size()))
        if hits:
            return hits[0]
    return None


def _build_session():
    """惰性建会话；失败把原因记在 _SESS["why"] 里（调用方只读不猜）。"""
    want = str(os.environ.get("DFL_SR_BACKEND") or "auto").strip().lower()
    eng = _engine_path()
    if want in ("auto", "trt") and eng is not None:
        try:
            from ..backend_models.trt_session import TrtSession
            # use_graph=False：CUDA Graph 在多线程下每条线程各捕一次（实测一条 40~80ms，
            # 8 条合成线程会反复捕 + 抢同一张图，跑流水线时直接卡住）。SR 的开销在算力上
            # （57ms），图重放省不到 2ms，不值得冒这个风险。own_stream=True：别和 torch 抢默认流。
            s = TrtSession(str(eng), use_graph=False, own_stream=True)
            return s, "trt", eng.name
        except Exception as e:                              # noqa: BLE001
            _SESS["_why"] = "TRT 引擎加载失败（%s）：%s" % (eng.name, str(e)[:160])
            if want == "trt":
                return None, "trt", eng.name
    onnx = _model_path()
    if not onnx.is_file():
        _SESS["_why"] = "找不到模型 %s（DFL_SR_MODEL 可指定）" % onnx
        return None, "", ""
    try:
        import onnxruntime as ort
        so = ort.SessionOptions()
        so.log_severity_level = 4
        provs = [p for p in ("CUDAExecutionProvider", "CPUExecutionProvider")
                 if p in ort.get_available_providers()]
        s = ort.InferenceSession(str(onnx), so, providers=provs)
        return s, ("onnx-cuda" if provs and provs[0].startswith("CUDA") else "onnx-cpu"), onnx.name
    except Exception as e:                                  # noqa: BLE001
        _SESS["_why"] = "onnxruntime 加载失败：%s" % str(e)[:200]
        return None, "", ""


def session(sz=None):
    """返回 (会话, 原因)。按 sz 取自己的那一份（不传就现读 env，见 _SESS 的注释）。"""
    if sz is None:
        sz = size()
    with _LOCK:
        rec = _SESS.get(sz)
        if rec is None:
            obj, backend, name = _build_session()
            rec = {"obj": obj, "backend": backend, "name": name, "why": _SESS.pop("_why", "")}
            _SESS[sz] = rec
            if obj is None:
                _log("不可用（%dpx）：%s" % (sz, rec["why"] or "未知原因"))
            else:
                _log("后端 %s · %s" % (backend, name))
        return rec["obj"], (rec.get("why") or "")


def snapshot() -> dict:
    """**一次**把 (level, strength, size) 读出来 —— 三者必须来自同一刻。

    为什么要它：这三个都读 env DFL_SR，而 env 会被并发的另一个请求
    （/api/merge/frame 的 apply_sr(0)）清掉。分开读就会出现日志里那种
    自相矛盾的字典：{"level":100, "strength":0.0, "size":256} ——
    紧接着 apply_roi 按 256 的输入喂 512 的模型，GPEN 输出崩成「彩色铅笔描的」。
    """
    lv = level()
    if lv <= 0:
        return {"level": 0, "strength": 0.0, "size": 256}
    a = lv / _FULL if lv <= _FULL else (lv - _FULL) / _FULL
    return {"level": int(lv), "strength": float(a),
            "size": 256 if lv <= _FULL else 512}


def info() -> dict:
    """给 [eff] / status 用的回显（报当前 size 那一份会话）。"""
    rec = _SESS.get(size()) or {}
    return {"level": level(), "mode": mode(), "strength": round(strength(), 2), "size": size(),
            "backend": rec.get("backend") or "", "model": rec.get("name") or "",
            "why": rec.get("why") or ""}


_RUN_LOCK = threading.Lock()


def _run(nchw, sz=None):
    """串行推理：合成车道是 8 条线程，同一个会话**不能**并发 run（TRT/ORT 都会崩）。

    ★ sz 必须由调用方**固定一次**传进来。以前这里自己 session() → size() 再读一遍
    env，而 env 会被并发的另一个请求（/api/merge/frame 的 apply_sr(0)）清掉：
    入口读到 512（按 512 resize 输入）、这里读到 256（用 256 的会话）→
    **512 的输入喂 256 的模型**，GPEN 输出直接崩成「彩色铅笔描的」。
    info() 里出现 level=100 / strength=0.0 / size=256 这种自相矛盾的字典，也是同一个
    竞态的证据（三次读 env 读到了不同的值）。
    """
    obj, why = (session(sz) if sz is not None else session())
    if obj is None:
        return None
    with _RUN_LOCK:
        return _infer(obj, nchw)


def _infer(obj, nchw):
    out = obj.run(None, {"input": nchw})
    # TrtSession 与 ort.InferenceSession 都是 run(outputs, feed) 签名
    return out[0] if isinstance(out, (list, tuple)) else out


def apply_roi(roi_u8, mask_f, alpha: float, timings: dict | None = None, sz: int = None):
    """在**帧空间**的脸区域上跑一次超分并按 mask*alpha 混回。

    roi_u8 : 合成之后这块 ROI 的 uint8 BGR
    mask_f : 同一块 ROI 的遮罩（0..1，float32，HxW）
    返回新的 roi_u8（失败时返回原图，不抛 —— 合成不该因为超分挂掉，但要打日志）
    """
    import cv2
    import numpy as np

    if alpha <= 0 or roi_u8 is None or roi_u8.size == 0:
        return roi_u8
    timings = timings if isinstance(timings, dict) else {}
    m = np.asarray(mask_f, np.float32)
    if m.ndim == 3:
        m = m[..., 0]
    if m.shape[:2] != roi_u8.shape[:2]:
        m = cv2.resize(m, (roi_u8.shape[1], roi_u8.shape[0]), interpolation=cv2.INTER_LINEAR)
    # ★ 推理用**整块 ROI**，混合才用遮罩 —— 这两个必须分开。
    #   以前按 np.nonzero(m > 0.02) 的 bbox 裁一块送去推理：XSeg 遮罩只盖皮肤，
    #   不含额头/下巴/耳朵，GPEN 拿到的是一张「残缺的脸」，输出的五官位置就错，
    #   混回去表现为眼睛变黑线、嘴唇变色块。
    #   （实测同一张画布：mask 全 1 → 输出正常；用 XSeg 遮罩 → 崩。）
    #   这也正是「调试六宫格看着正常」的原因：那条路的掩膜是盖满整块 panel 的椭圆，
    #   bbox 就是整块，等于整块推理。
    t0 = time.time()
    # ★ 尺寸由调用方钉死一次传进来最好（sz 参数）。不传才现读 env ——
    #   现读有竞态：env 会被并发的另一个请求（/api/merge/frame 的 apply_sr(0)）清掉。
    if sz is None:
        sz = size()
    sz = int(sz)
    _ch, _cw = roi_u8.shape[:2]
    if _ch < 8 or _cw < 8:
        return roi_u8
    # 长边缩到 sz、短边居中 pad（复制边缘），推理完裁回 —— 保持宽高比，
    # 否则 400×248 会被横向拉 1.6×、纵向压 0.5×，出来的脸是变形的。
    _sc = float(sz) / float(max(_ch, _cw))
    _nh, _nw = max(1, int(round(_ch * _sc))), max(1, int(round(_cw * _sc)))
    big = cv2.resize(roi_u8, (_nw, _nh), interpolation=cv2.INTER_LINEAR)
    _pt, _pb = (sz - _nh) // 2, sz - _nh - (sz - _nh) // 2
    _pl, _pr = (sz - _nw) // 2, sz - _nw - (sz - _nw) // 2
    big = cv2.copyMakeBorder(big, _pt, _pb, _pl, _pr, cv2.BORDER_REPLICATE)
    x = big.astype(np.float32) / 255.0
    x = (x - 0.5) / 0.5
    nchw = np.transpose(x, (2, 0, 1))[None, ...].astype(np.float32)
    out = _run(nchw, sz)          # sz 固定一次传下去，见 _run 的说明
    if out is None:
        timings["sr_ms"] = timings.get("sr_ms", 0.0) + (time.time() - t0) * 1000.0
        return roi_u8
    y = np.asarray(out, np.float32)[0]
    y = np.transpose(y, (1, 2, 0))
    y = (y * 0.5 + 0.5) * 255.0
    sr = np.clip(y, 0, 255).astype(np.uint8)
    sr = sr[_pt:_pt + _nh, _pl:_pl + _nw]          # 裁掉 pad
    sr = cv2.resize(sr, (_cw, _ch), interpolation=cv2.INTER_LINEAR)
    # 混合：这里**才**用遮罩。整块结果都在，遮罩只决定贴哪一部分。
    w = np.clip(m[..., None] * float(alpha), 0.0, 1.0)
    mixed = (sr.astype(np.float32) * w + roi_u8.astype(np.float32) * (1.0 - w))
    timings["sr_ms"] = timings.get("sr_ms", 0.0) + (time.time() - t0) * 1000.0
    return np.clip(mixed, 0, 255).astype(np.uint8)
