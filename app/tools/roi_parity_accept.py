# -*- coding: utf-8 -*-
r"""ROI 版 MergeMaskedFace 的验收：与整帧版**逐位对拍**，覆盖参数矩阵与边界。

为什么要有这一条：合成器把「整帧 float32 的遮罩/合成」压进脸的 ROI 之后，
单帧从 ~285ms 掉到 ~100ms（1080p、416 的 dfm）。这类改动最容易出的问题是
「某几个像素悄悄变了」——所以这里不是抽检，是**逐位**对拍：

  A) 真项目 120 帧 × 项目参数（seamless + mt + XSeg 遮罩 + blur4）：mean=0 max=0
  B) 参数矩阵：mode × mask_mode × erode/blur × 色彩迁移 × 运动模糊/锐化 × 超分
  C) 六宫格 capture：两侧都把中间量交出来，六张面板逐位相同
  D) 边界：小图、奇数尺寸、脸一半在画面外（ROI 被裁）、脸整个在画面外、
     遮罩全零（走「遮罩支撑太小」那条路）
  E) 计时：项目参数下 ROI=0 / ROI=1 的单帧墙钟（同一批帧、同一份模型输出）
  F) ROI 面积：遮罩包围盒 / 工作区占整帧的比例

跑法（仓库根）：
    runtime\server\Scripts\python.exe app\tools\roi_parity_accept.py
    runtime\server\Scripts\python.exe app\tools\roi_parity_accept.py --frames 120 --cases A,B,C,D,E,F
    --cv2-threads 1   固定 cv2 内部线程数（扫吞吐时用得上）

前提：
  · workspace\model 里有 .dfm 与 XSeg/XSegLite（没有就先训练/导出）；
  · 有一个 DFL 项目（<proj>/frames + <proj>/aligned 带 DFLJPG 的 landmarks）。
    没有的话先用 app\tools\proj_pool_accept.py --cases a 组一个（它落在
    %TEMP%\dfl_pool_accept\proj），本脚本会自动找 %TEMP% 下这几个候选：
      dfl_proj_pool\A · dfl_pool_accept\proj · dfl_roi_parity\proj
退出码 0 = 全过。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import types
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
ENGINE = REPO / "app" / "engine"
TEMP = Path(os.environ.get("TEMP") or os.environ.get("TMP") or ".")
BOX = TEMP / "dfl_roi_parity"
LOGDIR = BOX
RESULTS = []

PROJ_CANDS = [TEMP / "dfl_proj_pool" / "A", TEMP / "dfl_pool_accept" / "proj",
              BOX / "proj"]


# ------------------------------------------------------------
# 引擎侧 import 前置（跟 preview worker 同一套：DLL 路径 + numexpr/tqdm 替身）
# ------------------------------------------------------------
def engine_boot():
    sys.path.insert(0, str(ENGINE))
    for name in ("numexpr", "tqdm"):
        try:
            __import__(name)
        except Exception:                                   # noqa: BLE001
            m = types.ModuleType(name)
            if name == "numexpr":
                m.evaluate = lambda e, local_dict=None, global_dict=None, **k: eval(
                    str(e), {"__builtins__": {}}, dict(local_dict or {}))
                m.set_num_threads = lambda n: None
                m.get_num_threads = lambda: 1
            else:
                class _T:
                    def __init__(self, *a, **k):
                        pass

                    def __iter__(self):
                        return iter([])

                    def update(self, *a):
                        pass

                    def close(self):
                        pass
                m.tqdm = _T
            sys.modules[name] = m
    try:
        from MergeStudio.core.export_pipeline import _setup_dll_paths
        _setup_dll_paths()
    except Exception:                                       # noqa: BLE001
        pass
    d = os.path.join(sys.prefix, "Lib", "site-packages", "torch", "lib")
    if os.path.isdir(d):
        try:
            os.add_dll_directory(d)
        except Exception:                                   # noqa: BLE001
            pass
        os.environ["PATH"] = d + os.pathsep + os.environ.get("PATH", "")


# ------------------------------------------------------------
# 小工具
# ------------------------------------------------------------
def check(ok, name, detail=""):
    ok = bool(ok)
    RESULTS.append((ok, name, str(detail)))
    print(("[PASS] " if ok else "[FAIL] ") + name + ("  · " + str(detail) if detail else ""),
          flush=True)
    return ok


def say(msg):
    print("       " + str(msg), flush=True)


def head(t):
    print("")
    print("=" * 78)
    print(t)
    print("=" * 78)


def pick_dfm():
    p = REPO / "workspace" / "model" / "阿証証証証-416.dfm"
    if p.is_file():
        return p
    cands = sorted((REPO / "workspace" / "model").glob("*.dfm"),
                   key=lambda x: x.stat().st_size)
    return cands[0] if cands else None


def pick_xseg():
    for c in (REPO / "workspace" / "model" / "XSegLite" / "xseglite.onnx",
              REPO / "workspace" / "model" / "XSeg" / "XSeg.onnx"):
        if c.is_file():
            return c
    return None


def pick_project(explicit=""):
    if explicit:
        return Path(explicit)
    for c in PROJ_CANDS:
        if (c / "frames").is_dir() or any(c.glob("*.png")) or any(c.glob("*.jpg")):
            return c
    return None


def load_items(proj: Path, limit: int):
    """项目 → [(name, frame_u8, landmarks68)]。landmarks 取 DFLJPG 的 source_landmarks。"""
    import numpy as np
    import cv2
    import DFLIMG
    frames_dir = proj / "frames" if (proj / "frames").is_dir() else proj
    aligned = proj / "aligned"
    names = sorted(p.name for p in frames_dir.iterdir()
                   if p.suffix.lower() in (".jpg", ".png"))[:limit]
    out = []
    for n in names:
        # 对齐脸的文件名各版本不同（_0.jpg / _00.jpg …），按 <stem>_* 找第一张
        cands = sorted(aligned.glob(Path(n).stem + "_*"))
        ap = cands[0] if cands else None
        if ap is None or not ap.is_file():
            continue
        try:
            d = DFLIMG.DFLJPG.load(str(ap))
        except Exception:                                   # noqa: BLE001
            continue
        lm = d.get_source_landmarks()
        if lm is None:
            lm = d.get_landmarks()
        img = cv2.imread(str(frames_dir / n), cv2.IMREAD_COLOR)
        if img is None or lm is None:
            continue
        out.append((n, img, np.asarray(lm, np.float32)))
    return out


# ------------------------------------------------------------
# 模型 / 帧的确定性包装：两侧必须看到**同一份**前向输出
# ------------------------------------------------------------
class Stub:
    """把 dfm / xseg 的输出按 (帧号, 第几次调用) 缓存，ROI 开/关两次跑吃的是同一份。"""

    def __init__(self, sess, xsess, cache):
        import numpy as np
        import cv2
        self.np, self.cv2 = np, cv2
        self.cache = cache
        sh = list(sess.get_inputs()[0].shape)
        self.nchw = len(sh) >= 4 and sh[1] in (1, 3)
        self.size = int(sh[2] if self.nchw else sh[1])
        self.sess = sess
        self.inp = sess.get_inputs()[0].name
        self.xs = xsess
        xsh = list(xsess.get_inputs()[0].shape)
        self.x_nchw = len(xsh) >= 4 and xsh[1] in (1, 3)
        self.x_inp = xsess.get_inputs()[0].name

    def _mask2d(self, t, size):
        np, cv2 = self.np, self.cv2
        m = np.squeeze(t)
        if m.ndim == 3:
            m = m[:, :, 0]
        m = m.astype(np.float32)
        if m.shape[:2] != (size, size):
            m = cv2.resize(m, (size, size), interpolation=cv2.INTER_CUBIC)
        return m

    def pred(self, face_img):
        np, cv2 = self.np, self.cv2
        k = ("pred", self.cache["i"])
        if k in self.cache:
            return self.cache[k]
        size, nchw = self.size, self.nchw
        if face_img.shape[0] != size or face_img.shape[1] != size:
            face_img = cv2.resize(face_img, (size, size), interpolation=cv2.INTER_LANCZOS4)
        img = face_img.astype(np.float32)
        inp = np.transpose(img, (2, 0, 1))[None] if nchw else img[None]
        outs = self.sess.run(None, {self.inp: inp})
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
            pm = self._mask2d(outs[0], size)
            dm = self._mask2d(outs[2], size)
        while face_out.ndim >= 5:
            face_out = face_out[0]
        if face_out.ndim == 4:
            face_out = (np.transpose(face_out[0], (1, 2, 0)) if nchw else face_out[0])
        if face_out.ndim == 2:
            face_out = np.stack([face_out] * 3, axis=-1)
        if face_out.shape[-1] > 3:
            face_out = face_out[:, :, :3]
        elif face_out.shape[-1] == 1:
            face_out = np.repeat(face_out, 3, axis=-1)
        if face_out.shape[:2] != (size, size):
            face_out = cv2.resize(face_out, (size, size), interpolation=cv2.INTER_LANCZOS4)
        self.cache[k] = (face_out, pm, dm)
        return self.cache[k]

    def xseg(self, face_in):
        np, cv2 = self.np, self.cv2
        n = self.cache.get("xn", 0)
        self.cache["xn"] = n + 1
        k = ("xseg", self.cache["i"], n)
        if k in self.cache:
            return self.cache[k]
        oh, ow = face_in.shape[:2]
        fr = face_in if (oh, ow) == (256, 256) else cv2.resize(
            face_in, (256, 256), interpolation=cv2.INTER_CUBIC)
        fc = np.clip(fr, 0, 1) if float(fr.max()) < 2.0 else fr
        rgb = cv2.cvtColor(fc, cv2.COLOR_BGR2RGB).astype(np.float32)
        if float(rgb.max()) > 1.0:
            rgb = rgb / 255.0
        inp = np.transpose(rgb, (2, 0, 1))[None] if self.x_nchw else rgb[None]
        out = np.squeeze(self.xs.run(None, {self.x_inp: inp})[0])
        if out.shape[:2] != (oh, ow):
            out = cv2.resize(out, (ow, oh), interpolation=cv2.INTER_CUBIC)
        r = np.clip(out, 0, 1).astype(np.float32)
        self.cache[k] = r
        return r


# ------------------------------------------------------------
# 一次对拍：同一帧、同一参数，ROI=0 与 ROI=1 各跑一次
# ------------------------------------------------------------
def run_once(M, MergeMaskedFace, stub, idx, frame, lm, cfg, capture_of=None):
    # color_transfer_idt / mix 里有一句 np.random.normal（没播种），两侧必须同一个
    # 随机状态，否则比的是随机方向而不是 ROI —— 这是对拍口径，不是算法的一部分。
    import numpy as _np
    _np.random.seed(20260101)
    stub.cache["i"] = idx
    stub.cache["xn"] = 0
    cap = None
    if capture_of is not None:
        cap = {}
        capture_of.append(cap)
    out = MergeMaskedFace(frame, lm, cfg, stub.pred, xseg_256_extract_func=stub.xseg,
                          capture=cap, input_size=stub.size)
    return out


def cmp_pair(M, stub, idx, frame, lm, cfg, label, panels=None):
    """返回 (ok, detail)。两侧都用同一份前向输出。"""
    import numpy as np
    caps = {"0": [], "1": []}
    res = {}
    for mode, key in ((0, "0"), (1, "1")):
        os.environ["DFL_MERGE_ROI"] = str(mode)
        t0 = time.perf_counter()
        res[key] = run_once(M, M.MergeMaskedFace, stub, idx, frame, lm, cfg,
                            caps[key] if panels is not None else None)
        res[key + "_t"] = (time.perf_counter() - t0) * 1000.0
    fa, ma = res["0"]
    fb, mb = res["1"]
    if getattr(fa, "shape", None) != getattr(fb, "shape", None):
        return False, "形状不同 %r vs %r" % (getattr(fa, "shape", None),
                                          getattr(fb, "shape", None))
    if fa.dtype != fb.dtype:
        return False, "dtype 不同 %r vs %r" % (fa.dtype, fb.dtype)
    d = np.abs(fa.astype(np.int32) - fb.astype(np.int32)) if fa.ndim else \
        np.abs(fa.astype(np.int32) - fb.astype(np.int32))
    nz = int((d > 0).sum())
    mx = int(d.max()) if d.size else 0
    dmx = 0
    dnz = 0
    if getattr(ma, "shape", None) == getattr(mb, "shape", None):
        dm = np.abs(ma.astype(np.int32) - mb.astype(np.int32))
        dmx = int(dm.max()) if dm.size else 0
        dnz = int((dm > 0).sum())
    else:
        return False, "遮罩形状不同"
    detail = "%s：图 diff 像素 %d/%d max %d · 遮罩 diff 像素 %d max %d" % (
        label, nz, d.size, mx, dnz, dmx)
    if panels is not None and caps["0"] and caps["1"]:
        from MergeStudio.core.merger import build_debug_panels
        p0 = build_debug_panels(caps["0"][0], 128)
        p1 = build_debug_panels(caps["1"][0], 128)
        bad = []
        for (k0, _l0, i0), (k1, _l1, i1) in zip(p0, p1):
            if i0.shape != i1.shape or int(np.abs(i0.astype(np.int32) - i1.astype(np.int32)).max() or 0) != 0:
                bad.append(k0)
        if bad:
            return False, detail + " · 六宫格面板不同: %s" % ",".join(bad)
        detail += " · 六宫格 6/6 面板逐位相同"
    return (nz == 0 and mx == 0 and dnz == 0 and dmx == 0), detail


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=120)
    ap.add_argument("--cases", default="A,B,C,D,E,F")
    ap.add_argument("--project", default="")
    ap.add_argument("--dfm", default="")
    ap.add_argument("--xseg", default="")
    ap.add_argument("--cv2-threads", type=int, default=0)
    opt = ap.parse_args()

    engine_boot()
    import numpy as np
    import cv2
    import onnxruntime as ort
    import MergeStudio.core.merger as M
    from MergeStudio.core.config import MergerConfigMasked, ctm_dict

    if opt.cv2_threads:
        cv2.setNumThreads(int(opt.cv2_threads))
    dfm = Path(opt.dfm) if opt.dfm else pick_dfm()
    xs = Path(opt.xseg) if opt.xseg else pick_xseg()
    proj = pick_project(opt.project)
    if dfm is None or not dfm.is_file():
        raise SystemExit(r"找不到 .dfm（workspacemodel*.dfm）")
    if xs is None:
        raise SystemExit(r"找不到 XSeg onnx（workspacemodelXSeg**.onnx）")
    if proj is None:
        raise SystemExit("找不到 DFL 项目；先跑 app/tools/proj_pool_accept.py --cases a")
    cases = [c.strip().upper() for c in opt.cases.split(",") if c.strip()]
    print("dfm=%s\nxseg=%s\nproj=%s" % (dfm.name, xs.name, proj), flush=True)

    sess = ort.InferenceSession(str(dfm),
                               providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    xsess = ort.InferenceSession(str(xs),
                                 providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    cache = {"i": -1, "xn": 0}
    stub = Stub(sess, xsess, cache)
    ct_map = {str(v).strip().lower(): int(k) for k, v in ctm_dict.items()}
    ct_map["none"] = 0
    ct_map.setdefault("sot", ct_map.get("sot-m", 7))

    def mkcfg(**kw):
        base = dict(face_type="whole_face", mode="seamless", mask_mode=8,
                    color_transfer_mode=ct_map.get("mt", 9),
                    erode_mask_modifier=0, blur_mask_modifier=4, show_debug=False)
        base.update(kw)
        return MergerConfigMasked(**base)

    items = load_items(proj, opt.frames)
    if not items:
        raise SystemExit("项目里读不出帧+landmarks：%s" % proj)
    H, W = items[0][1].shape[:2]
    say("素材：%d 帧 %dx%d（%s）" % (len(items), W, H, items[0][0]))
    proj_cfg = mkcfg()

    # ---------- A · 真项目全量逐位对拍 ----------
    if "A" in cases:
        head("A · 真项目 %d 帧 × 项目参数：ROI 版 vs 整帧版逐位对拍" % len(items))
        M.ROI_STATS.update({k: 0 for k in M.ROI_STATS})
        worst = None
        bad = 0
        t0 = time.time()
        for i, (name, frame, lm) in enumerate(items):
            ok, detail = cmp_pair(M, stub, i, frame, lm, proj_cfg, name)
            if not ok:
                bad += 1
                if worst is None:
                    worst = detail
            if (i + 1) % 30 == 0:
                say("%d/%d 帧（%.1fs）" % (i + 1, len(items), time.time() - t0))
        check(bad == 0, "A %d 帧逐位一致（mean=0 / max=0 / 遮罩=0）" % len(items),
              "不一致 %d 帧%s" % (bad, ("；首个：" + worst) if worst else ""))
        check(M.ROI_STATS.get("roi_taken", 0) == len(items),
              "A 全部帧都走了 ROI 快路", "roi_taken=%s / 帧 %s" % (
                  M.ROI_STATS.get("roi_taken"), len(items)))

    # ---------- B · 参数矩阵 ----------
    if "B" in cases:
        head("B · 参数矩阵（每个组合 2 帧，全部要求逐位一致）")
        combos = []
        for mode in ("seamless", "overlay", "hist-match", "seamless-hist-match"):
            combos.append(("mode=%s" % mode, dict(mode=mode)))
        for mm in (0, 1, 2, 3, 4, 5, 6, 7, 8, 9):
            combos.append(("mask_mode=%d" % mm, dict(mask_mode=mm)))
        for ct in (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10):
            combos.append(("ct=%d" % ct, dict(color_transfer_mode=ct)))
        for er, bl in ((0, 0), (5, 0), (0, 10), (-6, 0), (0, 33), (-20, 60)):
            combos.append(("erode=%d blur=%d" % (er, bl),
                           dict(erode_mask_modifier=er, blur_mask_modifier=bl)))
        for extra in (dict(motion_blur_power=20), dict(blursharpen_amount=40),
                      dict(super_resolution_power=10), dict(output_face_scale=15),
                      dict(output_face_scale=-20), dict(mode="raw-rgb"),
                      dict(mode="raw-predict"), dict(mode="debug"),
                      dict(mode="seamless", color_transfer_mode=0),
                      dict(face_type="full_face"), dict(face_type="head")):
            combos.append((" ".join("%s=%s" % (k, v) for k, v in extra.items()), dict(**extra)))
        bad = []
        for label, over in combos:
            cfg = mkcfg(**over)
            for i in (0, len(items) // 2):
                name, frame, lm = items[i]
                ok, detail = cmp_pair(M, stub, i, frame, lm, cfg, "%s @%s" % (label, name))
                if not ok:
                    bad.append(detail)
        check(not bad, "B %d 组参数 × 2 帧逐位一致" % len(combos),
              ("%d 组不一致：" % len(bad) + " | ".join(x[:120] for x in bad[:6])) if bad else "")

    # ---------- C · 六宫格 capture ----------
    if "C" in cases:
        head("C · 六宫格 capture（预览那条路：capture 只写不读，两侧面板都得相同）")
        ok, detail = cmp_pair(M, stub, 0, items[0][1], items[0][2], proj_cfg,
                              "capture", panels=[])
        check(ok, "C capture 路径逐位一致", detail)

    # ---------- D · 边界 ----------
    if "D" in cases:
        head("D · 边界：尺寸 / 脸在画面外 / 遮罩全零")
        name0, frame0, lm0 = items[0]
        # 1) 小图
        import cv2 as _cv
        for (nw, nh) in ((320, 240), (1081, 1921)):
            f = _cv.resize(frame0, (nw, nh), interpolation=_cv.INTER_AREA)
            lms = lm0 * np.array([nw / W, nh / H], np.float32)
            ok, detail = cmp_pair(M, stub, 0, f, lms, proj_cfg, "resize %dx%d" % (nw, nh))
            check(ok, "D 尺寸 %dx%d 逐位一致" % (nw, nh), detail)
        # 2) 脸一半在画面外（ROI 被帧边裁掉）
        lms = lm0 - np.array([W * 0.42, 0.0], np.float32)
        ok, detail = cmp_pair(M, stub, 0, frame0, lms, proj_cfg, "face-left-edge")
        check(ok, "D 脸一半在画面外（ROI 被裁）逐位一致", detail)
        # 3) 脸整个在画面外
        lms = lm0 + np.array([W * 3.0, 0.0], np.float32)
        ok, detail = cmp_pair(M, stub, 0, frame0, lms, proj_cfg, "face-outside")
        check(ok, "D 脸整个在画面外（work=None → 整帧版）逐位一致", detail)
        # 4) 遮罩全零：xseg 返回 0 → 遮罩支撑为空
        real_xseg = stub.xseg

        def zero_xseg(face_in):
            return np.zeros((256, 256), np.float32)

        stub.xseg = zero_xseg
        try:
            ok, detail = cmp_pair(M, stub, 0, frame0, lm0, proj_cfg, "mask-empty")
            check(ok, "D 遮罩支撑为空（roi_false 分支）与原版一致", detail)
        finally:
            stub.xseg = real_xseg

    # ---------- E · 计时 ----------
    if "E" in cases:
        head("E · 单帧墙钟：ROI=0 vs ROI=1（同帧同参数，前 3 帧预热）")
        n = min(len(items), 48)
        rows = {}
        for mode in (0, 1):
            os.environ["DFL_MERGE_ROI"] = str(mode)
            # 清掉前向缓存：两档都必须真的各跑一次 dfm/xseg，否则第二档白赚
            # 20ms 的「缓存命中」，帧率对比就成了假的。
            for k in [k for k in cache if isinstance(k, tuple)]:
                cache.pop(k, None)
            ts = []
            for i, (name, frame, lm) in enumerate(items[:n]):
                cache["i"] = i
                cache["xn"] = 0
                t0 = time.perf_counter()
                M.MergeMaskedFace(frame, lm, proj_cfg, stub.pred,
                                  xseg_256_extract_func=stub.xseg, input_size=stub.size)
                dt = (time.perf_counter() - t0) * 1000.0
                if i >= 3:
                    ts.append(dt)
            a = np.array(ts)
            rows[mode] = a
            say("ROI=%d：%d 帧 mean %.1fms · p50 %.1f · min %.1f · max %.1f"
                % (mode, len(a), a.mean(), np.percentile(a, 50), a.min(), a.max()))
        sp = rows[0].mean() / max(1e-9, rows[1].mean())
        check(rows[1].mean() < rows[0].mean(),
              "E ROI 更快", "%.1fms → %.1fms（%.2fx，省 %.0f%%，cv2 线程 %d）"
              % (rows[0].mean(), rows[1].mean(), sp,
                 100.0 * (1 - rows[1].mean() / rows[0].mean()), cv2.getNumThreads()))

    # ---------- F · ROI 面积 ----------
    if "F" in cases:
        head("F · ROI 面积（遮罩支撑 / 包围盒 / 工作区 占整帧）")
        os.environ["DFL_MERGE_ROI"] = "1"
        px_ratio, bbox_ratio, work_ratio = [], [], []
        H0, W0 = items[0][1].shape[:2]
        area = float(H0 * W0)
        for i, (name, frame, lm) in enumerate(items[:min(len(items), 24)]):
            cap = {}
            for k in [k for k in cache if isinstance(k, tuple)]:
                cache.pop(k, None)
            cache["i"] = i
            cache["xn"] = 0
            fo, fm = M.MergeMaskedFace(frame, lm, proj_cfg, stub.pred,
                                       xseg_256_extract_func=stub.xseg, capture=cap,
                                       input_size=stub.size)
            mk = np.asarray(fm, np.float32)
            gt0 = mk > 0
            if gt0.any():
                r = np.flatnonzero(gt0.any(axis=1))
                c = np.flatnonzero(gt0.any(axis=0))
                bbox_ratio.append(((r[-1] - r[0] + 1) * (c[-1] - c[0] + 1)) / area)
            px_ratio.append(float(gt0.mean()))
            from facelib import FaceType, LandmarksProcessor
            fmat = LandmarksProcessor.get_transform_mat(lm, stub.size, face_type=FaceType.WHOLE_FACE)
            q = M._quad_box(fmat, stub.size, (H0, W0))
            if q:
                work_ratio.append(((q[2] - q[0]) * (q[3] - q[1])) / area)
        if px_ratio:
            say("mask>0 像素占比 %.3f%% · 遮罩包围盒 %.2f%% · 工作区(画布+8px) %.2f%%"
                % (100 * np.mean(px_ratio), 100 * np.mean(bbox_ratio),
                   100 * np.mean(work_ratio)))
            check(np.mean(bbox_ratio) < 0.25, "F 包围盒远小于整帧（有得省）",
                  "包围盒 %.2f%% / 工作区 %.2f%%" % (100 * np.mean(bbox_ratio),
                                                  100 * np.mean(work_ratio)))

    bad = [x for x in RESULTS if not x[0]]
    print("")
    print("=" * 78)
    print("结果：%d 项通过，%d 项失败" % (len(RESULTS) - len(bad), len(bad)))
    for _ok, name, detail in bad:
        print("  [FAIL] %s  · %s" % (name, detail))
    print("=" * 78)
    sys.exit(1 if bad else 0)


main()
