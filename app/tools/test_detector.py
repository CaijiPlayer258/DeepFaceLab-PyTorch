# -*- coding: utf-8 -*-
"""检测器冒烟测试：不依赖后端服务，直接用 runtime/server 的 python 跑。

用法（仓库根目录）：

    runtime\\server\\Scripts\\python.exe app\\tools\\test_detector.py

跑什么
    1) RetinaFaceDet 构造                          → 不抛异常
    2) TinyMogDet 构造                             → 不抛异常
    3) RetinaFaceDet.detect(一张真实对齐图)         → list，每个 box 是 4 个数
    4) TinyMogDet.detect(同一张图)                 → 同上
    5) TinyMogDet(graph_max_batch=8) 再构造一次     → 不许 TypeError（回归点）

为什么有第 5 步
    TinyMogDet 的 __init__ 曾经漏掉 graph_max_batch 形参，而 app.py 的
    _get_fast_detector() 已经按新签名传了它 —— TypeError 抛在抽取线程里被
    except 吞掉，界面显示「提取完成 0 帧」。这里用**和抽取器完全一样的调用方式**
    （含 workspace / engine_name / graph_max_batch）构造一遍，把签名漂移挡在提交前。

约定
    · 样图只读：只从 workspace/data_src/aligned 里读 jpg，绝不写 workspace
    · 挑图优先挑「RetinaFace 真能检出人脸」的那张 —— 空图只能验证返回类型，
      验证不了框；挑不到就退回第一张，把 faces=0 当合法结果
    · 没有权重 / 没有样图 → [SKIP]，退出码 0
    · 任何一步失败 → 打印 [FAIL] 哪一步、哪个断言，退出码 1
"""
from __future__ import annotations

import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]          # 仓库根（DeepFaceLab-PyTorch-4.1）
WORKSPACE = ROOT / "workspace"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

RESULTS = []        # [(状态, 步骤名, 说明)]


# ----------------------------------------------------------------- 权重定位
def _weight_roots():
    """按 app 的查找顺序列出权重根目录（去重，只读）。

    权重是**平铺**的：weights/tinymog.onnx，没有 onnx/<模型>/ 这一层。
    """
    from app.server._paths import weights_dir

    cands = [weights_dir(), ROOT / "weights",
             ROOT / "app" / "engine" / "modelhub"]
    out, seen = [], set()
    for p in cands:
        try:
            key = str(Path(p).resolve())
        except OSError:
            key = str(p)
        if key not in seen:
            seen.add(key)
            out.append(Path(p))
    return out


_FLAT = {
    "TinyMog/TinyMog.onnx": "tinymog.onnx",
    "RetinaFace/det_10g.onnx": "retinaface_10g.onnx",
    "RetinaFace/det_500m.onnx": "retinaface_500m.onnx",
}


def _find_onnx(rel):
    """rel 可以写平铺名（tinymog.onnx），也可以写老布局的 "<模型>/<文件>"。"""
    name = _FLAT.get(rel, rel)
    for base in _weight_roots():
        for cand in (base / name, base / "onnx" / name, base / rel,
                     base / "onnx" / rel):
            if cand.is_file():
                return cand
    return None


# ----------------------------------------------------------------- 样图
def _candidates(cv2, limit=40):
    """workspace/data_src/aligned 里能打开的 jpg（只读，按文件名序，最多 limit 张）。"""
    out = []
    d = WORKSPACE / "data_src" / "aligned"
    if not d.is_dir():
        return out
    for p in sorted(d.glob("*.jpg")):
        if not p.is_file():
            continue
        img = cv2.imread(str(p))
        if img is None or not img.size:
            continue
        out.append((p, img))
        if len(out) >= limit:
            break
    return out


# ----------------------------------------------------------------- 断言
def _check_faces(tag, faces):
    """detect() 的返回值契约：list，每项 dict，box 是 4 个数。"""
    assert isinstance(faces, list), \
        "%s: detect() 应返回 list，实际是 %s" % (tag, type(faces).__name__)
    for i, fc in enumerate(faces):
        assert isinstance(fc, dict), \
            "%s: 第 %d 项不是 dict，是 %s" % (tag, i, type(fc).__name__)
        box = fc.get("box")
        assert isinstance(box, (list, tuple)), \
            "%s: 第 %d 项没有 box（keys=%s）" % (tag, i, sorted(fc))
        assert len(box) == 4, \
            "%s: 第 %d 项的 box 不是 4 个数：%r" % (tag, i, box)
        assert all(isinstance(v, (int, float)) for v in box), \
            "%s: 第 %d 项的 box 里有非数字：%r" % (tag, i, box)
    if not faces:
        return "faces=0（这张图上没检出人脸 —— 合法，但值得看一眼）"
    return "faces=%d · box 全为 4 元组" % len(faces)


def _run(name, fn):
    print("[ .. ] " + name, flush=True)
    try:
        detail = fn()
    except AssertionError as e:
        print("[FAIL] %s → %s" % (name, e), flush=True)
        RESULTS.append(("FAIL", name, str(e)))
        return None
    except Exception:                                   # noqa: BLE001
        print("[FAIL] %s → 抛异常：" % name, flush=True)
        traceback.print_exc()
        last = traceback.format_exc().strip().splitlines()[-1]
        RESULTS.append(("FAIL", name, last))
        return None
    print("[ OK ] %s · %s" % (name, detail or ""), flush=True)
    RESULTS.append(("OK", name, detail or ""))
    return detail


# ----------------------------------------------------------------- main
def main() -> int:
    print("=" * 68)
    print("检测器冒烟测试 · " + str(ROOT))
    print("python " + sys.version.split()[0])
    tiny = _find_onnx("TinyMog/TinyMog.onnx")
    retina = _find_onnx("RetinaFace/det_10g.onnx") or _find_onnx("RetinaFace/det_500m.onnx")
    print("TinyMog    : " + (str(tiny) if tiny else "（没有）"))
    print("RetinaFace : " + (str(retina) if retina else "（没有）"))
    print("=" * 68, flush=True)

    if tiny is None and retina is None:
        print("[SKIP] 没有检测器权重（weights/ 根下没有 tinymog.onnx / retinaface_10g.onnx）")
        print("[SKIP] 退出码 0 —— 这不是失败，是环境没装模型")
        return 0

    try:
        import cv2
    except Exception as e:                              # noqa: BLE001
        print("[FAIL] 导入 cv2 失败：%s: %s" % (type(e).__name__, e))
        return 1

    cands = _candidates(cv2)
    if not cands:
        print("[SKIP] workspace/data_src/aligned 里没有能打开的 jpg，跳过")
        print("[SKIP] 退出码 0")
        return 0

    from app.server._detect import RetinaFaceDet, TinyMogDet

    built = {}

    def step_build_retina():
        built["retina"] = RetinaFaceDet(str(retina), 640)
        return "构造成功（%s）" % type(built["retina"]).__name__

    def step_build_tinymog():
        built["tinymog"] = TinyMogDet(str(tiny))
        return "构造成功（%s）" % type(built["tinymog"]).__name__

    if retina is not None:
        _run("RetinaFaceDet 构造", step_build_retina)
    else:
        print("[SKIP] 没有 det_10g/det_500m.onnx，跳过 RetinaFaceDet")

    if tiny is not None:
        _run("TinyMogDet 构造", step_build_tinymog)
    else:
        print("[SKIP] 没有 TinyMog.onnx，跳过 TinyMogDet")

    # 挑图：优先拿 RetinaFace 真能检出人脸的那张（空图只看得出返回类型）
    img_path, img, note = cands[0][0], cands[0][1], "第一张能打开的图"
    rd = built.get("retina")
    if rd is not None:
        for p, im in cands:
            try:
                if rd.detect(im):
                    img_path, img, note = p, im, "RetinaFace 在这张上检出了人脸"
                    break
            except Exception as e:                      # noqa: BLE001
                print("[ .. ] 挑图时 %s 失败：%s" % (p.name, e), flush=True)
    print("样图       : %s  (%dx%d · %s)"
          % (img_path, img.shape[1], img.shape[0], note), flush=True)

    def step_retina_detect():
        return _check_faces("RetinaFaceDet", built["retina"].detect(img))

    def step_tinymog_detect():
        return _check_faces("TinyMogDet", built["tinymog"].detect(img))

    def step_signature():
        # 完全是 app.py:_get_fast_detector() 的调用形状
        try:
            det = TinyMogDet(str(tiny), workspace=WORKSPACE, engine_name="tinymog",
                             graph_max_batch=8)
        except TypeError as e:
            raise AssertionError(
                "TinyMogDet 不接受 graph_max_batch —— app.py 的 _get_fast_detector() "
                "就是这么调的，抽取会直接 TypeError：%s" % e)
        return "构造成功（%s），graph_max_batch=8 被接受" % type(det).__name__

    if "retina" in built:
        _run("RetinaFaceDet.detect", step_retina_detect)
    if "tinymog" in built:
        _run("TinyMogDet.detect", step_tinymog_detect)
    if tiny is not None:
        _run("TinyMogDet(graph_max_batch=8) 签名回归", step_signature)

    bad = [r for r in RESULTS if r[0] == "FAIL"]
    print("=" * 68)
    print("通过 %d · 失败 %d" % (len(RESULTS) - len(bad), len(bad)))
    for _st, name, why in bad:
        print("  FAIL %s → %s" % (name, why))
    if not bad:
        print("全部通过")
    return 0 if not bad else 1


if __name__ == "__main__":
    raise SystemExit(main())
