"""RetinaFace 检测器（精简版：整图检测 + 5 点关键点，不含 TRT / 滑窗）。

从 4.0 的 modelhub/onnx/RetinaFace/RetinaFace.py 摘出核心解码逻辑，
去掉对 xlib / det_preprocess 的依赖，只保留 onnxruntime + numpy + cv2。

关键点是必须的：DFL 的对齐靠 5 点相似变换，只有框的话裁出来的脸对不上
DFL 的对齐空间（眼睛/嘴巴位置会飘），所以 _forward 把 kps 分支一并解出来。
"""
from __future__ import annotations

import contextlib
import threading
from pathlib import Path

import os
import numpy as np

# 检测器的"回退/换引擎"只讲一次，别每帧刷屏；但一定要讲 —— 静默回退是上一次
# "以为在用 int8 / 以为开了 CUDA Graph" 的根因。
_DET_LOGGED = set()


def _det_log(key, msg):
    if key in _DET_LOGGED:
        return
    _DET_LOGGED.add(key)
    print(msg, flush=True)


# ============================================================
# 检测走哪个 EP：DFL_DET_EP = auto | gpu | cpu（默认 auto）
# ============================================================
# 为什么要这个开关：融合流水线里 dfm / XSeg / 检测**共用同一个 GPU 流**。
# dfm 一次前向 30~38ms（416），而 tinymog 单张前向只要 1.4ms —— 检测插进去的代价
# 不是那 1.4ms，而是「排在一批 dfm 后面 + 一次上下文切换」。把这种**小到可以忽略**
# 的模型挪到 CPU，GPU 流就少一个插队者；反过来说，retinaface_10g 前向 9.9ms、CPU 上
# 49.9ms（慢 5 倍），挪过去只会把整条线拖死。
#
# 判据是**实测**，不是型号名。auto 是两道门（细节见 decide()）：
#   门① 单张前向（640 canvas，不含 letterbox/NMS）GPU ≤ DFL_DET_EP_GPU_MS（默认 1.5ms）
#        → 它在 GPU 上占的时间已小到没收益，收益只能来自「不再排队」→ **候选**；
#   门② 这台机器/这个模型上量到过「检测走 CPU 净增 ≥5%」的端到端记录（工作区
#        _sess_bench.json 的 gain:detect）→ 才真的切。两门都过才走 CPU，否则留在 GPU。
# 为什么非要门②：门①只证明「值得试」。本机 1080p 120 帧 · 384 dfm · 6 轮交错 A/B 的
# 实测是 **19.16 → 18.42 帧/秒（配对中位 −3.86%，6 轮全负）**：省下 1.4ms 的 GPU，
# 赔上 26~29ms 的 CPU 忙时间，还跟 8 条合成线程抢核（合成 129.8 → 133.8 ms/帧）。
# 拿实测去选，而不是拿「前向很小」去选 —— 这就是门②存在的理由。
# 数字都是 2026-11 本机（RTX 5070 Laptop 8GB / 32 逻辑核）实测中位：
#   tinymog        GPU 1.4ms（TRT bf16 + CUDA Graph） / CPU 8.9~12.5ms（intra_op=3）
#   retinaface_10g GPU 9.9ms（ORT-CUDA fp32）         / CPU 49.9~150ms
#   insightface_2d106 / xseglite 不在这里：2d106 本来就在 CPU；xseg 留 GPU（CPU 慢 11 倍）
# 拿不到实测时退回 KNOWN_GPU_FWD_MS 这张已知表（同一台机器量的），表里也没有
# 就**留在 GPU**（不认识的东西不动它 —— 赌错一次就是整条线慢一倍）。
ENV_EP = "DFL_DET_EP"                    # auto | gpu | cpu
ENV_THREADS = "DFL_DET_EP_THREADS"       # CPU EP 的 intra_op 线程数（2..4）
ENV_GPU_MS = "DFL_DET_EP_GPU_MS"         # auto 的判据阈值（单张 GPU 前向 ms）

AUTO_GPU_MS = 1.5
CPU_THREADS_MIN, CPU_THREADS_MAX, CPU_THREADS_DEFAULT = 2, 4, 3

KNOWN_GPU_FWD_MS = {
    "tinymog": 1.4,             # TRT bf16 + CUDA Graph，前向（不含 letterbox）
    "retinaface_10g": 9.9,      # ORT-CUDA fp32
}


def _env_str(name):
    import os
    return str(os.environ.get(name) or "").strip()


def _env_flag(name, default: bool) -> bool:
    v = _env_str(name)
    if not v:
        return bool(default)
    return v.lower() not in ("0", "off", "false", "no")


def mode() -> str:
    """auto | gpu | cpu（写错/为空 = auto，不抛异常：一个错字不该让导出起不来）。"""
    v = _env_str(ENV_EP).lower()
    return v if v in ("auto", "gpu", "cpu") else "auto"


def threads() -> int:
    """CPU EP 用几个 intra_op 线程。2~4：合成那 8 条线程 + 编码 3 条 + 对齐 4 条
    已经把核占得差不多了，检测再要 8 条就是互相抢（实测 T8 只比 T4 快 20%，
    代价是合成段变慢）。"""
    try:
        v = int(float(_env_str(ENV_THREADS) or CPU_THREADS_DEFAULT))
    except (TypeError, ValueError):
        v = CPU_THREADS_DEFAULT
    return max(CPU_THREADS_MIN, min(CPU_THREADS_MAX, v))


def auto_gpu_ms() -> float:
    try:
        v = float(_env_str(ENV_GPU_MS) or AUTO_GPU_MS)
    except (TypeError, ValueError):
        v = AUTO_GPU_MS
    return v if v > 0 else AUTO_GPU_MS


def norm_name(name) -> str:
    """'tinymog.onnx' / 'weights\\tinymog.onnx' → 'tinymog'（表是按规范名写的）。"""
    s = str(name or "").strip().replace("\\", "/").rsplit("/", 1)[-1]
    return s[:-len(".onnx")] if s.lower().endswith(".onnx") else s


def decide(name, gpu_fwd_ms=None, gain_ok=None, gain_pct=None) -> dict:
    """→ {"ep": "gpu"|"cpu", "why": [...], "gpu_fwd_ms", "thr", "src", "mode"}

    why 是给人看的一句话（日志直接打）；src = 这个数是实测的还是查表的。

    **auto 是两道门，缺一不可**：
      门① 物理：单张前向 GPU ≤ thr —— 只有这种模型「挪走能省的 GPU 时间」小到可以忽略，
           收益才只可能来自「不再排队」。
      门② 端到端：工作区里得有「检测走 CPU 那档净增 ≥5%」的实测（_sess_bench.json 的
           gain:detect → auto_cpu_ep_ok，见 routes_merge 收尾处的 record_gain）。
           为什么非要有这道门：门①只证明「值得试」，不证明「真的赚」。本机实测
           （1080p 120 帧 · 384 dfm · 6 轮交错 A/B）tinymog 前向 1.42ms 过门①，
           端到端却是 **19.16 → 18.42 帧/秒（配对中位 −3.86%，6 轮全负）**：
           省下 1.4ms 的 GPU，赔上 29ms 的 CPU 忙时间，还跟 8 条合成线程抢核
           （compose 129.8 → 133.8ms）。**没到 +5% 就不动默认**。
      所以全新工作区上 auto == 现状（GPU）；跑过一次 DFL_DET_EP=cpu 并记下 fps 之后，
      这台机器/这个模型上的 auto 才开始按实测走。
    """
    m = mode()
    key = norm_name(name)
    thr = auto_gpu_ms()
    why = []
    if m == "gpu":
        why.append("%s=gpu（显式）→ 全部留在 GPU 流" % ENV_EP)
        return {"ep": "gpu", "why": why, "gpu_fwd_ms": gpu_fwd_ms,
                "thr": thr, "src": "explicit", "mode": m}
    if m == "cpu":
        why.append("%s=cpu（显式）→ 检测走 CPU EP（intra_op=%d）"
                   % (ENV_EP, threads()))
        return {"ep": "cpu", "why": why, "gpu_fwd_ms": gpu_fwd_ms,
                "thr": thr, "src": "explicit", "mode": m}
    # ---- auto ----
    src = "measured"
    ms = gpu_fwd_ms
    if ms is None:
        ms = KNOWN_GPU_FWD_MS.get(key)
        src = "known-table" if ms is not None else "unknown"
    if ms is None:
        why.append("%s=auto 但 %s 既没量到前向、已知表里也没有 → 留在 GPU（不拿没量过的模型赌）"
                   % (ENV_EP, key))
        return {"ep": "gpu", "why": why, "gpu_fwd_ms": None,
                "thr": thr, "src": src, "mode": m}
    if float(ms) <= thr:
        head = ("%s=auto · %s 单张前向 GPU %.2f ms ≤ 阈值 %.2f ms（%s）→ 门①过"
                % (ENV_EP, key, float(ms), thr, src))
        if not gain_ok:
            if gain_pct is None:
                tail = ("；门②没过：工作区里还没有『检测走 CPU 净增 ≥5%』的实测"
                        "（_sess_bench.json 的 gain:detect）→ 留在 GPU。要试就显式 "
                        "DFL_DET_EP=cpu 跑一遍（本机实测 −3.86%，不值得默认开）")
            else:
                tail = ("；门②没过：实测记录里 CPU 那档是 %+.2f%%（要 ≥+5%%）→ 留在 GPU"
                        % float(gain_pct))
            why.append(head + tail)
            return {"ep": "gpu", "why": why, "gpu_fwd_ms": float(ms),
                    "thr": thr, "src": src, "mode": m, "gate2": False,
                    "gain_pct": gain_pct}
        why.append(head + "；门②也过（实测 CPU 那档 %+.2f%%）→ 走 CPU EP（intra_op=%d）"
                   % (float(gain_pct if gain_pct is not None else 0.0), threads()))
        return {"ep": "cpu", "why": why, "gpu_fwd_ms": float(ms),
                "thr": thr, "src": src, "mode": m, "gate2": True,
                "gain_pct": gain_pct}
    why.append("%s=auto · %s 单张前向 GPU %.2f ms > 阈值 %.2f ms（%s）→ 留在 GPU"
               "（CPU 上只会更慢，挪走就是拖死流水线）"
               % (ENV_EP, key, float(ms), thr, src))
    return {"ep": "gpu", "why": why, "gpu_fwd_ms": float(ms),
            "thr": thr, "src": src, "mode": m}


def cpu_session_options(nthr=None):
    """CPU EP 的 SessionOptions：图优化全开 + 固定 intra_op（inter_op=1）。

    inter_op=1 是有意的：检测是「一次一张」的串行负载，ORT 的 inter_op 线程池
    在这儿只会多一次唤醒，实测反而慢 ~0.3ms。
    """
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    so.intra_op_num_threads = int(nthr or threads())
    so.inter_op_num_threads = 1
    return so


_PREP_CANVAS = threading.local()


def _prep_ctx(img_bgr, input_size, pad, reuse=False):
    """整帧 → {'canvas','scale','w','h'}：letterbox + 填边，**只此一份**。

    RetinaFace 用 0 填边、TinyMog 用 128（和 4.0 的 modelgraph 一致），插值都是
    INTER_LINEAR —— 这里的每一个字都决定框的位置，改一处就是另一套坐标。

    reuse=True 时画布**按线程复用**（填边只做第一次）：调用方必须保证「同一线程里上一张
    canvas 已经用完、并且不会再被别处引用」—— 融合流水线的 detect 线程就是这样（prep →
    前向 → 立刻 drop），省掉每帧一次 640²×3 = 1.2MB 的 np.full（实测 0.72 → 0.33 ms/帧，
    见 app/docs/notes/experiments/dec_lane.md）。默认 False：老调用方（切脸等）照旧拿独立画布。

    另外记一笔**没采用**的路子：cv2 的 INTER_LINEAR 在奇数整数倍下采样（1080x1920→360x640，
    k=3）上恰好等于步长采样 frame[(k-1)//2::k, (k-1)//2::k]（0 像素不同，control.py 量的），
    但步长采样本身比 cv2.resize 更慢（1.15 vs 0.72 ms/帧）—— 它是 6.2MB 的跨步 gather，
    cv2 是流式 SIMD。所以这里不做这个"优化"。
    """
    import cv2

    H, W = img_bgr.shape[:2]
    S = int(input_size)
    scale = min(S / float(W), S / float(H))
    nw, nh = max(1, int(round(W * scale))), max(1, int(round(H * scale)))
    canvas = None
    if reuse:
        c = getattr(_PREP_CANVAS, "buf", None)
        if c is not None and c.shape == (S, S, 3) and c.dtype == np.uint8:
            canvas = c
        else:
            canvas = (np.full((S, S, 3), int(pad), np.uint8) if pad
                      else np.zeros((S, S, 3), np.uint8))
            _PREP_CANVAS.buf = canvas
    if canvas is None:
        canvas = (np.full((S, S, 3), int(pad), np.uint8) if pad
                  else np.zeros((S, S, 3), np.uint8))
    canvas[:nh, :nw] = cv2.resize(img_bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)
    return {"canvas": canvas, "scale": scale, "w": int(W), "h": int(H)}


_NULL_CTX = contextlib.nullcontext()


def _stream_ctx(det):
    """检测器该在哪条 CUDA 流上跑（自己那条；没有这个概念就是空上下文）。

    为什么检测要独占一条流（2026-11 实测）：dfm 走 torch、xseg 走 torch，两者都在
    **默认流**上。检测以前也在默认流，于是它 1.27ms 的前向只能插在 21ms 的 dfm 两批
    之间；更糟的是 TrtSession 每次 replay 还调了整块卡同步（见 trt_session.py 开头的
    「同步口径」），detect 那一笔因此量到 19~46ms。独占流 + 帧级事件同步之后，检测的
    1.27ms 真跟 dfm 的 21ms 重叠，detect 那一笔回到 2~3ms。
    """
    fn = getattr(getattr(det, "sess", None), "own_stream_ctx", None)
    return fn() if fn is not None else _NULL_CTX


_DET_MISS = {"frames": 0, "best": 0.0, "threshold": 0.0, "last_max": 0.0}
_DET_DROP = {"frames": 0, "merged": 0, "small": 0}


def _faces_from_fwd(det, fwd, ctx, min_face=16):
    """前向输出（canvas 坐标）→ 原图坐标的人脸列表。

    解码 → 按分数排 → NMS → 除以 scale 回原图 → 丢掉太小的。这一步是**纯 CPU**，
    所以「检测段 48ms 里 resize/NMS 占多少」只能靠它和 _prep_ctx 分别计时来回答。
    """
    if len(fwd) == 3:
        scores, boxes, kpss = fwd
    else:
        scores, boxes = fwd
        kpss = np.zeros((scores.shape[0], 5, 2), np.float32)
    if scores.size == 0:
        return []

    scale = float(ctx["scale"])
    W, H = int(ctx["w"]), int(ctx["h"])
    order = scores.argsort()[::-1]
    dets = np.hstack([boxes[order], scores[order, None]]).astype(np.float32)
    kpss = kpss[order]
    keep = det._nms(dets, det.nms_thresh)
    # 三个「脸悄悄消失」的环节各记一笔：用户在成片里看到「明明有脸却没换」时，
    # 这三条是唯一能告诉他丢在哪儿的线索。以前全在函数里静默蒸发。
    #   命中 N 个候选 → NMS 留 K → min_face 再砍 M。
    _n_cand, _n_nms = int(dets.shape[0]), int(len(keep))
    # ★ 一帧都没过阈值时，把「最高分是多少」记一笔 —— 用户报「检测不到」时，
    #   这一条能立刻区分「模型没加载」和「分数不够（阈值太高）」。
    #   TinyMog 的分数分布本来就偏低（实测中位 0.69），0.5 的绝对门比 S3FD 严得多。
    if _n_nms == 0:
        # 每**帧**记一次（_faces_from_fwd 是一帧一次；_forward 可能一帧多次）
        try:
            _DET_MISS["frames"] = int(_DET_MISS.get("frames") or 0) + 1
            _top = float(_DET_MISS.get("last_max") or 0.0)
            _DET_MISS["best"] = max(float(_DET_MISS.get("best") or 0.0), _top)
            _DET_MISS["threshold"] = float(det.thresh)
        except Exception:                                  # noqa: BLE001
            pass

    out = []
    for i in keep:
        d = dets[i]
        x1 = max(0.0, d[0] / scale)
        y1 = max(0.0, d[1] / scale)
        x2 = min(float(W), d[2] / scale)
        y2 = min(float(H), d[3] / scale)
        if min(x2 - x1, y2 - y1) >= min_face:
            kp = kpss[i] / scale
            out.append({"box": [round(float(x1), 1), round(float(y1), 1),
                                round(float(x2), 1), round(float(y2), 1)],
                        "kps": [[round(float(p[0]), 1), round(float(p[1]), 1)] for p in kp],
                        "score": round(float(d[4]), 4)})
    # 只在真的丢了东西时打一行（正常情况下绝大多数帧不会命中，不会刷屏）。
    _n_out = len(out)
    # ★ 这是**统计**，不是错误：一张脸被检成 4 个候选、NMS 合成 1 个，是检测器的
    #   正常工作方式。b28 我把它写成每帧都打 note()，结果用户一次导出 1439 帧
    #   刷了 1254 行 —— 整份日志 87% 是这个，真正的错误全被淹了。
    #   现在：只统计不逐帧打；收尾由 pipeline_fused 报一行汇总（见那里的 z_* 计数）。
    global _DET_DROP
    if _n_cand > _n_out:
        try:
            if _n_cand - _n_nms > 0 or _n_nms - _n_out > 0:
                _DET_DROP["frames"] = int(_DET_DROP.get("frames") or 0) + 1
                _DET_DROP["merged"] = int(_DET_DROP.get("merged") or 0) + max(0, _n_cand - _n_nms)
                _DET_DROP["small"] = int(_DET_DROP.get("small") or 0) + max(0, _n_nms - _n_out)
                _k = 0

        except Exception:                                  # noqa: BLE001
            pass
    out.sort(key=lambda x: -x["score"])
    return out


class RetinaFaceDet:
    needs_canvas = False      # detect(img) 内部自己做 letterbox
    """det_10g / det_500m 通用。输入 BGR ndarray，输出人头框列表。"""

    STEPS = (8, 16, 32)

    def __init__(self, model_path, input_size=640, thresh=0.5, nms_thresh=0.4, graph_max_batch=None,
                 workspace=None, engine_name=None, providers=None, sess_options=None):
        import onnxruntime as ort
        if providers is not None:
            # 显式给了 providers 就用它（DFL_DET_EP=cpu 那条路要的：不建 CUDA session，
            # 也不碰 TRT 引擎 —— 给 CPU 用的检测器不该先占一块显存再把会话扔掉）
            _providers = list(providers)
        else:
            try:
                from .._ortcuda import providers as _prov
                _providers = _prov()          # 装了 onnxruntime-gpu 就用 CUDA
            except Exception:
                _providers = ["CPUExecutionProvider"]
        # 优先用编译好的 TRT 引擎（引擎名按 onnx 的 md5 + SM 命名）
        self.sess = None
        if workspace is not None:
            try:
                from ..backend_models.engines import try_engine
                self.sess = try_engine(workspace, engine_name or Path(model_path).stem, model_path)
            except Exception as e:                             # noqa: BLE001
                _det_log(("retina", "try", str(model_path)),
                         "[RetinaFaceDet] TRT 引擎查找出错（%r）→ 回退 onnxruntime" % (e,))
                self.sess = None
        if self.sess is None:
            _det_log(("retina", "ort", str(model_path)),
                     "[RetinaFaceDet] 使用 onnxruntime（providers=%s）" % (list(_providers),))
            self.sess = ort.InferenceSession(str(model_path), sess_options=sess_options,
                                             providers=_providers)
        # 检测独占一条 CUDA 流（DFL_DET_OWN_STREAM=0 可关；见 _stream_ctx 的注释）
        if hasattr(self.sess, "own_stream"):
            self.sess.own_stream = _env_flag("DFL_DET_OWN_STREAM", True)
        self.input_name = self.sess.get_inputs()[0].name
        self.input_size = int(input_size)
        self.thresh = float(thresh)
        self.nms_thresh = float(nms_thresh)
        self.fmc = 3
        self.num_anchors = 2

    # ---------- 解码 ----------
    @staticmethod
    def _distance2bbox(points, distance):
        return np.stack([
            points[:, 0] - distance[:, 0],
            points[:, 1] - distance[:, 1],
            points[:, 0] + distance[:, 2],
            points[:, 1] + distance[:, 3],
        ], axis=-1)

    def _forward(self, img):
        """img: 已经 letterbox 到 (S,S) 的 BGR 图。"""
        S = self.input_size
        # 一次拷贝：直接写进 float32 的 NCHW 缓冲，再原地减均值/缩放。
        # 原来的写法 (astype -> -127.5 -> /128 -> 通道翻转 -> transpose+ascontiguousarray)
        # 一共 5 遍全图遍历，实测 3.48 ms；这里 1.05 ms，结果逐位一致。
        blob = np.empty((1, 3, S, S), np.float32)
        blob[0] = img[:, :, ::-1].transpose(2, 0, 1)
        blob -= 127.5
        blob *= (1.0 / 128.0)

        outs = self.sess.run(None, {self.input_name: blob})

        scores_list, bboxes_list, kps_list = [], [], []
        for idx, stride in enumerate(self.STEPS):
            scores = outs[idx].reshape(-1)
            bbox_preds = outs[idx + self.fmc].reshape(-1, 4) * stride
            kps_preds = outs[idx + 2 * self.fmc].reshape(-1, 10) * stride

            h = S // stride
            w = S // stride
            centers = np.stack(np.mgrid[:h, :w][::-1], axis=-1).astype(np.float32)
            centers = (centers * stride).reshape(-1, 2)
            if self.num_anchors > 1:
                centers = np.stack([centers] * self.num_anchors, axis=1).reshape(-1, 2)

            bboxes = self._distance2bbox(centers, bbox_preds)
            kps = self._distance2kps(centers, kps_preds)
            keep = np.where(scores >= self.thresh)[0]
            scores_list.append(scores[keep])
            bboxes_list.append(bboxes[keep])
            kps_list.append(kps[keep])

        if not scores_list or sum(s.size for s in scores_list) == 0:
            return np.empty((0,)), np.empty((0, 4)), np.empty((0, 10))
        return (np.concatenate(scores_list),
                np.concatenate(bboxes_list, axis=0),
                np.concatenate(kps_list, axis=0))

    @staticmethod
    def _distance2kps(points, distance):
        """5 个关键点（10 个数）从中心点 + 偏移还原。"""
        out = []
        for i in range(0, distance.shape[1], 2):
            px = points[:, 0] + distance[:, i]
            py = points[:, 1] + distance[:, i + 1]
            out.append(np.stack([px, py], axis=-1))
        return np.stack(out, axis=1)      # (N, 5, 2)

    @staticmethod
    def _nms(dets, thresh):
        if dets.shape[0] == 0:
            return []
        x1, y1, x2, y2 = dets[:, 0], dets[:, 1], dets[:, 2], dets[:, 3]
        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = dets[:, 4].argsort()[::-1]
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h
            ovr = inter / (areas[i] + areas[order[1:]] - inter)
            order = order[np.where(ovr <= thresh)[0] + 1]
        return keep

    # ---------- 对外 ----------
    def prep(self, img_bgr, reuse=False):
        """整帧 → 检测输入上下文 {'canvas','scale','w','h'}。**不改像素，只做 letterbox**。

        单独拆出来是为了让调用方在**别的线程**先把这一步做掉（decode 线程有富余，
        检测线程是硬底）：detect() = prep() + detect_ctx()，两步的分界线就是
        「CPU 的 resize / 填边」与「前向 + NMS」，拆开前后逐位一致（同一个函数、
        同一个插值、同一张 canvas），见 app/tools/accept/det_prep_parity.py。
        """
        return _prep_ctx(img_bgr, self.input_size, 0, reuse=reuse)

    def detect_ctx(self, ctx, min_face=16):
        """上下文 → [{'box','kps','score'}]（原图坐标）。和 detect() 完全同一条口径。"""
        # 整个调用（GPU 预处理 + 前向 + NMS）都在检测自己的流上：进流前输入还在 host，
        # 设备上没有要接的依赖，所以不 wait 调用方的流（那正是要躲掉的串行化）。
        with _stream_ctx(self):
            return _faces_from_fwd(self, self._forward(ctx["canvas"]), ctx, min_face)

    def detect(self, img_bgr, min_face=16):
        """返回 [{'box': [x1,y1,x2,y2], 'score': float}, ...]（原图坐标）。"""
        return self.detect_ctx(self.prep(img_bgr), min_face=min_face)


class TinyMogDet:
    needs_canvas = True       # detect_batch 吃的是预先 letterbox 好的 640 canvas
    """TinyMog（SCRFD + MobileNetV1）—— 快检测器。

    与 RetinaFaceDet 同一套对外接口（detect(img) -> [{box, kps, score}]），
    区别是模型只有 2.7 MB、输入 640、batch 维是**动态**的（可以批量推理）。

    预处理与 4.0 的 modelhub/onnx/TinyMog 一致：letterbox 到 640（padding=128）、
    blobFromImage(1/128, mean=127.5, swapRB=True)。输出 9 个张量：每个 stride
    一组 (cls, bbox, kps)，anchor 顺序与 RetinaFace 相同（2 anchors，8/16/32）。
    """

    STEPS = (8, 16, 32)
    INPUT_SIZE = 640

    # 精度优先级：int8 有就先用，没有再退 bf16，两个都没有才回 onnxruntime。
    # int8 引擎必须是"用当前 onnx 的 md5 + 当前 TRT 版本"编出来的规范命名，
    # 目录里那些 <name>_int8.engine 是旧命名，永远命中不到（见 _engines.try_engine 日志）。
    PRECISION = ("int8", "bf16")

    def __init__(self, model_path, input_size=640, thresh=0.5, nms_thresh=0.4, graph_max_batch=None,
                 providers=None, workspace=None, engine_name=None, sess_options=None):
        import onnxruntime as ort

        if providers is None:
            try:
                from .._ortcuda import providers as _prov
                providers = _prov()
            except Exception:
                providers = ["CPUExecutionProvider"]
        # 优先用编译好的 TRT 引擎（引擎名按 onnx 的 md5 + SM + TRT 版本命名）
        self.sess = None
        self.sess_batch = None          # 仅当首选引擎做不了 batch（例如单张 int8）时才单独一个
        name = engine_name or Path(model_path).stem
        if workspace is not None:
            try:
                from ..backend_models.engines import try_engine
                self.sess = try_engine(workspace, name, model_path, precision=self.PRECISION, graph_max_batch=graph_max_batch)
            except Exception as e:                             # noqa: BLE001
                _det_log(("tinymog", "try", str(model_path)),
                         "[TinyMogDet] TRT 引擎查找出错（%r）→ 回退 onnxruntime" % (e,))
                self.sess = None
        if self.sess is None:
            _det_log(("tinymog", "ort", str(model_path)),
                     "[TinyMogDet] 没有可用的 TRT 引擎，使用 onnxruntime（providers=%s）"
                     % (list(providers),))
            self.sess = ort.InferenceSession(str(model_path), sess_options=sess_options,
                                             providers=list(providers))
        else:
            prec = getattr(self.sess, "precision", "?")
            mb = self.sess.max_batch() if hasattr(self.sess, "max_batch") else None
            print("[TinyMogDet] 单张走 %s 引擎（precision=%s, max_batch=%s）"
                  % (type(self.sess).__name__, prec, mb), flush=True)
            if prec == "int8" and mb is not None and mb < 2:
                # 老的 int8 引擎是静态 batch=1：批量检测没法用同一份引擎
                try:
                    self.sess_batch = try_engine(workspace, name, model_path, precision=("bf16",), graph_max_batch=graph_max_batch)
                except Exception as e:                         # noqa: BLE001
                    _det_log(("tinymog", "batch_try", str(model_path)),
                             "[TinyMogDet] int8 只支持单张，找 bf16 批量引擎时出错（%r）" % (e,))
                if self.sess_batch is not None:
                    print("[TinyMogDet] int8 引擎 max_batch=1，批量检测改走 bf16 引擎", flush=True)
                else:
                    _det_log(("tinymog", "batch_ort", str(model_path)),
                             "[TinyMogDet] int8 引擎只支持单张、又没有 bf16 引擎 → "
                             "批量检测回退 onnxruntime")
                    self.sess_batch = ort.InferenceSession(str(model_path),
                                                           sess_options=sess_options,
                                                           providers=list(providers))
        if hasattr(self.sess, "own_stream"):
            self.sess.own_stream = _env_flag("DFL_DET_OWN_STREAM", True)
        if hasattr(self.sess_batch, "own_stream") and self.sess_batch is not self.sess:
            self.sess_batch.own_stream = _env_flag("DFL_DET_OWN_STREAM", True)
        self.input_name = self.sess.get_inputs()[0].name
        self.batch_input_name = (self.sess_batch.get_inputs()[0].name
                                 if self.sess_batch is not None else self.input_name)
        self.input_size = int(input_size)
        self.thresh = float(thresh)
        self.nms_thresh = float(nms_thresh)
        self._centers_cache = {}

    # ---- 解码 ----
    @staticmethod
    def _centers(stride, S, num_anchors=2):
        h = w = S // stride
        c = np.stack(np.mgrid[:h, :w][::-1], axis=-1).astype(np.float32)
        c = (c * stride).reshape(-1, 2)
        if num_anchors > 1:
            c = np.stack([c] * num_anchors, axis=1).reshape(-1, 2)
        return c

    def _forward(self, canvas):
        S = self.input_size
        outs = None
        # ★ 单张和批量必须是**同一份权重**。int8 引擎 max_batch=1 时会另建一个 bf16
        #   的 sess_batch（:511），于是「同一帧」走 detect() 用 int8、走 detect_batch()
        #   用 bf16 —— 两份量化不同，小脸上的分数和框会差一点。而 faces:[k] 的契约
        #   就是 det.detect() 的返回顺序（fused_lanes.py:123-128）：同一份 cache 的
        #   前一半帧（批量）和后一半帧（批量失败退回逐帧）里，「第 k 张脸」可能不是
        #   同一张 —— 多模型点名时就是换错人。
        #   有 sess_batch 就单张也用它（bf16 跑 batch=1 没问题），代价是 int8 那点
        #   速度；一致性优先于这 1ms。DFL_DET_UNIFY=0 可关。
        _unify = str(os.environ.get("DFL_DET_UNIFY") or "1").strip().lower() not in (
            "0", "false", "no", "off")
        _s1 = (self.sess_batch if (_unify and self.sess_batch is not None) else self.sess)
        _in1 = (self.batch_input_name if (_s1 is self.sess_batch and
                getattr(self, "batch_input_name", None)) else self.input_name)
        # 优先走 GPU 预处理：640² 的 BGR→RGB + CHW + 归一化在 numpy 里要 1ms 以上
        if getattr(_s1, "run_tensor", None) is not None:
            try:
                import torch
                t = torch.from_numpy(np.ascontiguousarray(canvas)).cuda()
                t = t.permute(2, 0, 1).flip(0).float()
                t = (t - 127.5) * (1.0 / 128.0)
                outs = _s1.run_tensor({_in1: t})
            except Exception as e:                             # noqa: BLE001
                # GPU 预处理这条路炸了就退回 numpy blob（结果一样），但必须留痕
                _det_log("run_tensor_fail",
                         "[TinyMogDet] GPU 预处理/推理失败（%r）→ 本次改用 numpy blob 预处理"
                         % (e,))
                outs = None
        if outs is None:
            blob = np.empty((1, 3, S, S), np.float32)
            blob[0] = canvas[:, :, ::-1].transpose(2, 0, 1)
            blob -= 127.5
            blob *= (1.0 / 128.0)
            outs = _s1.run(None, {_in1: blob})

        scores_list, boxes_list, kps_list = [], [], []
        for idx, stride in enumerate(self.STEPS):
            scores = outs[idx * 3].reshape(-1)
            key = stride
            if key not in self._centers_cache:
                self._centers_cache[key] = self._centers(stride, S)
            centers = self._centers_cache[key]

            # ★ 记这一帧的原始最高分（**阈值之前**），存成 last_max 给上层判。
            #   不在这一层累加帧数：_forward 可能被同一帧调多次（多尺度/重试），
            #   累加会把「31 帧」记成「134 帧」。计数交给 _faces_from_fwd（每帧一次）。
            #   TinyMog 的分数分布本来就偏低（实测中位 0.69），0.5 的绝对门比 S3FD
            #   严得多 —— 这是「benchmark 好、实际差」的一半原因。
            try:
                _mx = float(scores.max()) if scores.size else 0.0
                _DET_MISS["last_max"] = _mx
                _DET_MISS["threshold"] = float(self.thresh)
            except Exception:                              # noqa: BLE001
                pass
            # 只在命中阈值的索引上取值：不再整表 stack / where 两遍
            keep = np.flatnonzero(scores >= self.thresh)
            if keep.size == 0:
                continue
            n = min(len(scores), len(centers))
            keep = keep[keep < n]
            if keep.size == 0:
                continue
            bboxes = outs[idx * 3 + 1].reshape(-1, 4)[keep] * stride
            kps = outs[idx * 3 + 2].reshape(-1, 10)[keep] * stride
            c = centers[keep]

            b = np.empty((keep.size, 4), np.float32)
            b[:, 0] = c[:, 0] - bboxes[:, 0]
            b[:, 1] = c[:, 1] - bboxes[:, 1]
            b[:, 2] = c[:, 0] + bboxes[:, 2]
            b[:, 3] = c[:, 1] + bboxes[:, 3]

            k = kps.reshape(-1, 5, 2)
            k[:, :, 0] += c[:, 0:1]
            k[:, :, 1] += c[:, 1:2]

            scores_list.append(scores[keep])
            boxes_list.append(b)
            kps_list.append(k)
        if not scores_list:
            return np.empty((0,)), np.empty((0, 4)), np.empty((0, 5, 2))
        return (np.concatenate(scores_list),
                np.concatenate(boxes_list, axis=0),
                np.concatenate(kps_list, axis=0))

    _nms = staticmethod(RetinaFaceDet._nms)

    # ---------- 批量 ----------
    def forward_batch(self, canvases):
        """canvases: list of (640,640,3) BGR uint8，返回每张的 (scores, boxes, kps)。"""
        n = len(canvases)
        S = self.input_size
        blob = np.empty((n, 3, S, S), np.float32)
        for i, c in enumerate(canvases):
            blob[i] = c[:, :, ::-1].transpose(2, 0, 1)
        blob -= 127.5
        blob *= (1.0 / 128.0)
        sess = self.sess_batch or self.sess                 # int8 单张引擎不能做 batch
        outs = sess.run(None, {self.batch_input_name: blob})

        results = []
        for b in range(n):
            scores_list, boxes_list, kps_list = [], [], []
            for idx, stride in enumerate(self.STEPS):
                scores = outs[idx * 3][b].reshape(-1)
                bboxes = outs[idx * 3 + 1][b].reshape(-1, 4) * stride
                kps = outs[idx * 3 + 2][b].reshape(-1, 10) * stride
                if stride not in self._centers_cache:
                    self._centers_cache[stride] = self._centers(stride, S)
                centers = self._centers_cache[stride]
                m = min(len(scores), len(centers))
                scores, bboxes, kps, centers = scores[:m], bboxes[:m], kps[:m], centers[:m]
                keep = np.where(scores >= self.thresh)[0]
                if keep.size == 0:
                    continue
                c = centers[keep]
                bb = bboxes[keep]
                kk = kps[keep].reshape(-1, 5, 2).copy()
                kk[:, :, 0] += c[:, 0][:, None]
                kk[:, :, 1] += c[:, 1][:, None]
                scores_list.append(scores[keep])
                boxes_list.append(np.stack([c[:, 0] - bb[:, 0], c[:, 1] - bb[:, 1],
                                            c[:, 0] + bb[:, 2], c[:, 1] + bb[:, 3]], axis=-1))
                kps_list.append(kk)
            if not scores_list:
                results.append((np.empty(0), np.empty((0, 4)), np.empty((0, 5, 2))))
            else:
                results.append((np.concatenate(scores_list),
                                np.concatenate(boxes_list, 0),
                                np.concatenate(kps_list, 0)))
        return results

    def detect_batch(self, items, min_face=16):
        """items: [(img_bgr, canvas, scale)] → 每张的 [{box,kps,score}]（原图坐标）。"""
        canvases = [it[1] for it in items]
        res = self.forward_batch(canvases)
        out = []
        for (img, canvas, scale), (scores, boxes, kpss) in zip(items, res):
            H, W = img.shape[:2]
            if scores.size == 0:
                out.append([])
                continue
            order = scores.argsort()[::-1]
            dets = np.hstack([boxes[order], scores[order, None]]).astype(np.float32)
            kpss = kpss[order]
            keep = self._nms(dets, self.nms_thresh)
            faces = []
            for i in keep:
                d = dets[i]
                x1 = max(0.0, d[0] / scale); y1 = max(0.0, d[1] / scale)
                x2 = min(float(W), d[2] / scale); y2 = min(float(H), d[3] / scale)
                if min(x2 - x1, y2 - y1) < min_face:
                    continue
                kp = kpss[i] / scale
                faces.append({"box": [round(float(x1), 1), round(float(y1), 1),
                                      round(float(x2), 1), round(float(y2), 1)],
                              "kps": [[round(float(p[0]), 1), round(float(p[1]), 1)] for p in kp],
                              "score": round(float(d[4]), 4)})
            faces.sort(key=lambda x: -x["score"])
            out.append(faces)
        return out

    def prep(self, img_bgr, reuse=False):
        """整帧 → {'canvas','scale','w','h'}（letterbox 填 128，和 4.0 的预处理一致）。

        reuse=True：本线程复用自己的画布（调用方保证上一张已用完，见 _prep_ctx）。
        """
        return _prep_ctx(img_bgr, self.input_size, 128, reuse=reuse)

    def detect_ctx(self, ctx, min_face=16):
        # 同 RetinaFaceDet.detect_ctx：预处理 + 前向都在检测自己的流上（理由见 _stream_ctx）
        with _stream_ctx(self):
            return _faces_from_fwd(self, self._forward(ctx["canvas"]), ctx, min_face)

    def detect(self, img_bgr, min_face=16):
        return self.detect_ctx(self.prep(img_bgr), min_face=min_face)
