"""XSeg 遮罩推理（4.1 侧：TensorRT 引擎优先，没有才回落 onnxruntime）。

对齐时顺手生成 xseg_mask 写进 DFLJPG —— 真 DFL 提取的每张脸都带 (256,256,1) 遮罩，
而 4.1 之前写出去的 DFLJPG 没有，训练端要么退化到用 landmarks 算 hull（而 4.1 的
68 点是从 5 点刚体投影出来的，形状是死的），要么根本没有遮罩。

预处理与 MergeStudio 的 _phase_mask_worker 完全一致：
  BGR → 256×256 → RGB → /255 → CHW → [1,3,256,256]；输出 0..1，按需缩放回原尺寸。

**后端选择只有一条路**：_engines.try_engine(workspace, <引擎名>, <onnx>)。
有匹配的引擎（md5 + SM + TRT 版本都对）就用 TRT，没有就回落 onnxruntime CUDA。
以前这里（以及 routes_merge 里那份预览 worker）各写各的 ort.InferenceSession，
「引擎躺在目录里却没被用」只能靠 stdout 的一行 [TRT] 猜 —— 现在加载结果全记在
self.info 里，遮罩阶段直接把它回显到 status / 日志。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

# 模型名（= 界面上的选择）→ onnx 的相对路径候选。顺序 = 目录里先找谁。
MODEL_REL = {
    "lite": ("model/XSegLite/xseglite.onnx", "model/XSegLite/XSegLite.onnx"),
    "xseg": ("model/XSeg/XSeg.onnx",),
}
MODEL_LABEL = {"lite": "XSegLite", "xseg": "XSeg"}
MODEL_ENGINE = {"lite": "xseglite", "xseg": "xseg"}   # 对应 _engines.MODEL_SPECS 的名字


def norm_model(model) -> str:
    """界面/请求里的写法 → "lite" / "xseg"（认不出返回空串）。"""
    k = str(model or "").strip().lower().replace("_", "").replace("-", "").replace(" ", "")
    if k in ("lite", "xseglite", "xseglite2", "2"):
        return "lite"
    if k in ("xseg", "1"):
        return "xseg"
    return ""


def model_roots(workspace, engine_workspace=None) -> list:
    """找 onnx 的目录，按优先级。和 app._get_xseg 用的是同一套顺序。"""
    from ._paths import ROOT, weights_dir
    out = []
    for r in (Path(weights_dir()) / "xseg", Path(workspace)):
        if r not in out:
            out.append(r)
    if engine_workspace:
        ws = Path(engine_workspace) / "workspace"
        if ws not in out:
            out.append(ws)
    if (ROOT / "weights" / "xseg") not in out:
        out.append(ROOT / "weights" / "xseg")
    return out


def resolve(model, workspace, engine_workspace=None) -> dict:
    """模型选择 → 实际会用哪个 onnx 文件（**不加载**，只查路径 + 算 md5）。

    返回 {"request": 规范化后的请求, "key": lite|xseg, "path": Path, "label": "XSegLite",
          "fallback": bool（请求的和实际找到的不是同一个）, "md5": 前 12 位, "engine": 引擎名,
          "candidates": [找过的路径…]}
    请求的模型文件不存在时**退到另一个**（老行为），但 fallback=True 让调用方必须回显出来。
    """
    want = norm_model(model)
    roots = model_roots(workspace, engine_workspace)
    order = ([want] if want else ["lite", "xseg"]) + \
            [k for k in ("lite", "xseg") if k != want]
    tried = []
    for key in order:
        for rel in MODEL_REL[key]:
            for root in roots:
                cand = Path(root) / rel
                tried.append(str(cand))
                if cand.is_file():
                    try:
                        from ._engines import md5_fresh
                        md5 = md5_fresh(MODEL_ENGINE[key], str(cand))
                    except Exception:                    # noqa: BLE001
                        md5 = ""
                    return {"request": want or "auto", "key": key, "path": cand,
                            "label": MODEL_LABEL[key], "engine": MODEL_ENGINE[key],
                            "fallback": bool(want and want != key), "md5": md5,
                            "candidates": tried}
    return {"request": want or "auto", "key": "", "path": None, "label": "",
            "engine": "", "fallback": False, "md5": "", "candidates": tried}


class XSegMasker:
    def __init__(self, model_path, providers=None, threads: int = 2, workspace=None,
                 engine_name: str | None = None):
        self.onnx = str(model_path)
        self.engine_name = engine_name or Path(model_path).stem
        self.backend = "ort"
        self.engine_file = ""
        self.precision = ""
        # 加载全过程记在这儿：backend / engine / precision / md5 / reason / providers
        self.info = {"backend": "ort", "engine": "", "engine_path": "", "precision": "",
                     "md5": None, "reason": "", "onnx": self.onnx,
                     "name": self.engine_name, "providers": []}
        # 先看有没有编译好的 TensorRT 引擎（MD5 + SM 都对才算数）
        self.sess = None
        if workspace is not None:
            try:
                from ._engines import try_engine
                # 遮罩优先 INT8（实测 2.0ms → 0.75ms），没有 INT8 引擎就退 bf16
                info = {}
                self.sess = try_engine(workspace, self.engine_name, model_path,
                                       precision=("int8", "bf16"), use_graph=False, info=info)
                if info:
                    self.info.update(info)
                    self.info["onnx"] = self.onnx
            except Exception as e:                       # noqa: BLE001
                self.sess = None
                self.info["reason"] = "try_engine 抛异常：%r" % (e,)
        else:
            self.info["reason"] = "没有 workspace，未查询 TRT 引擎"
        if self.sess is not None:
            self.backend = "trt"
            self.engine_file = str(self.info.get("engine") or "")
            self.precision = str(self.info.get("precision") or "")
            self.info.update(backend="trt", engine=self.engine_file,
                             precision=self.precision,
                             providers=["TensorrtExecutionProvider"])
            self.input_name = self.sess.get_inputs()[0].name
            return
        import onnxruntime as ort

        if providers is None:
            from ._ortcuda import providers as _prov
            providers = _prov()          # 有 CUDA 走 CUDA，否则 CPU
        self.sess = ort.InferenceSession(str(model_path), providers=list(providers))
        self.input_name = self.sess.get_inputs()[0].name
        try:
            self.sess.set_providers(list(providers), [{}] * len(providers))
        except Exception:
            pass
        self.backend = "ort"
        self.engine_file = ""
        self.precision = ""
        self.info.update(backend="ort", engine="", precision="",
                         providers=list(self.sess.get_providers()))
        if not self.info.get("reason"):
            self.info["reason"] = "没有匹配的 TRT 引擎"

    # ---- 回显用 ----
    def describe(self) -> str:
        """一行「这次用的是哪个后端/引擎」（日志与 status 共用）。"""
        if self.backend == "trt":
            return "TRT(%s) · 引擎 %s" % (self.precision or "?", self.engine_file or "?")
        prov = ",".join(self.info.get("providers") or []) or "?"
        return "onnxruntime(%s)" % prov

    def batch_limit(self) -> int:
        """一次前向最多塞几张（1 = 只能逐张）。首次调用时探测一次并缓存。"""
        cap = getattr(self, "_batch_cap", None)
        if cap is not None:
            return cap
        cap = 1
        try:
            mb = getattr(self.sess, "max_batch", None)
            if callable(mb):
                cap = int(mb() or 1)                 # 原生 TRT：按优化档位
            else:
                shape = list(self.sess.get_inputs()[0].shape)
                if shape and isinstance(shape[0], int) and shape[0] > 0:
                    cap = int(shape[0])              # 静态 batch 的 ort 模型
                else:
                    cap = 8                          # 动态 batch（ort）
        except Exception:                             # noqa: BLE001
            cap = 1
        self._batch_cap = max(1, int(cap))
        return self._batch_cap

    def mask_batch(self, faces, out_size: int = 256):
        """N 张 BGR 对齐脸 → N 张 (out_size,out_size) float32 0..1。

        能批量就一次前向（本机 256² 实测：ort-CUDA 逐张 4.02 ms/张 → batch2 2.68 ms/张），
        引擎只吃 batch 1（TRT 的 xseglite 档位是单档）就老实逐张 —— 逐张也不慢，
        1.44 ms/张，瓶颈根本不在推理（见 app/tools/bench/stage_perf_profile.py 的实测表）。
        批量失败时把上限降回 1 并逐张重跑，行为静默但对调用方仍然正确。
        """
        import cv2
        import numpy as np

        faces = list(faces or [])
        if not faces:
            return []
        cap = self.batch_limit()
        if any(f is None for f in faces):
            # 保住「下标一一对应」：批量里混进 None 时逐张走，绝不让返回值的个数错位
            return [None if f is None else self.mask(f, out_size=out_size) for f in faces]
        if len(faces) == 1 or cap <= 1:
            return [self.mask(f, out_size=out_size) for f in faces]
        outs = []
        for i in range(0, len(faces), cap):
            part = [f for f in faces[i:i + cap] if f is not None]
            if not part:
                continue
            blobs = [cv2.dnn.blobFromImage(
                (f if f.shape[:2] == (256, 256)
                 else cv2.resize(f, (256, 256), interpolation=cv2.INTER_LINEAR)),
                1.0 / 255.0, (256, 256), swapRB=True, crop=False) for f in part]
            try:
                y = self.sess.run(None, {self.input_name: np.concatenate(blobs, axis=0)})[0]
            except Exception:                         # noqa: BLE001
                self._batch_cap = 1                   # 这张引擎其实吃不了批量
                return [self.mask(f, out_size=out_size) for f in faces]
            for b in np.asarray(y)[:len(part)]:
                m = np.clip(np.squeeze(b).astype(np.float32), 0.0, 1.0)
                if m.shape[0] != out_size or m.shape[1] != out_size:
                    m = cv2.resize(m, (out_size, out_size), interpolation=cv2.INTER_LINEAR)
                outs.append(m)
        return outs

    def mask(self, face_bgr, out_size: int = 256):
        """ face_bgr: 任意尺寸的 BGR 对齐脸；返回 (out_size, out_size) float32，0..1。"""
        import cv2

        if face_bgr is None:
            return None
        x = face_bgr
        if x.shape[0] != 256 or x.shape[1] != 256:
            x = cv2.resize(x, (256, 256), interpolation=cv2.INTER_LINEAR)
        # 用 OpenCV 的 blobFromImage：resize + BGR→RGB + /255 + HWC→CHW 全在 C 里完成，
        # 比 numpy 的切片转置快 ~0.4ms/张（实测 300 张的平均值）
        inp = cv2.dnn.blobFromImage(x, 1.0 / 255.0, (256, 256), swapRB=True, crop=False)
        out = self.sess.run(None, {self.input_name: inp})[0]
        m = np.squeeze(out).astype(np.float32)
        m = np.clip(m, 0.0, 1.0)
        if m.shape[0] != out_size or m.shape[1] != out_size:
            m = cv2.resize(m, (out_size, out_size), interpolation=cv2.INTER_LINEAR)
        return m
