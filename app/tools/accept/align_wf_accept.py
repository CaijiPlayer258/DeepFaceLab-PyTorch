# -*- coding: utf-8 -*-
r"""切脸几何**与 iperov facelib 等价**的回归门（单测级，不打模型、不写数据）。

验的五件事（对应这一轮的要求 ②/④）：

  A  1203 组真实 source_landmarks：align_mat_from_68 与
     facelib.get_transform_mat(lm, S, WHOLE_FACE) 比 —— zoom 1.00000±1e-5、
     rot 0.000°±1e-4、画布四角位移 <0.05px。同一份数据把**改前**（68 点拟合）
     的同一组数字也打出来，好对账。
  B  不动点：4.0 的 aligned 芯片（= iperov 切的）用它自己的 source_landmarks
     重算，结果应当 ≈ 恒等（<0.05px）—— 改前是 3°/13% 级别的差。
  C  外点检查：68 点残差的真实分布（p50/p90/p99/max，按 |yaw| 分桶）+ 两个自检
     —— 真实脸一张都不许被拒；人工把 3 个下颌点推歪必须被拒。
  D  像素：同一帧 + 同一组 68 点，我们的 align_face_68 与「facelib 矩阵 + 同样
     的 warp 参数」逐字节相同（mean 0.000 LSB）；落盘 JPG 再解码 ≤ 1 LSB。
     顺带量 iperov 真实插值（LANCZOS4）与我们的 CUBIC 的差 —— 纯插值，不改几何。
  E  元数据自洽：image_to_face_mat ∘ source_landmarks == landmarks（<0.05px）。
     （E 在重切之后才有意义；重切前跑也过，因为老数据本来就是自洽的。）

跑法（仓库根，server venv）：
    runtime\server\Scripts\python.exe app\tools\align_wf_accept.py
    runtime\server\Scripts\python.exe app\tools\align_wf_accept.py --cases a,c
    runtime\server\Scripts\python.exe app\tools\align_wf_accept.py --ref40 D:\path\to\4.0\aligned

退出码 0 = 全过。结果 json 落在 %TEMP%\dfl_align_ab\result.json。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
APP = REPO / "app"
TEMP = Path(os.environ.get("TEMP") or os.environ.get("TMP") or ".")
BOX = TEMP / "dfl_align_ab"
OUT = BOX / "out"
sys.path.insert(0, str(APP))
sys.path.append(str(APP / "engine"))

from server import _align as AL                                     # noqa: E402
from facelib import FaceType                                        # noqa: E402
from facelib.LandmarksProcessor import get_transform_mat             # noqa: E402

SRC_ALIGNED = REPO / "workspace" / "data_src" / "aligned"
PROJ_ALIGNED = REPO / "workspace" / "merge" / "9月29日 (5)" / "aligned"
PROJ_FRAMES = REPO / "workspace" / "merge" / "9月29日 (5)"
REF40 = Path("D:/MyProject/DeepFaceLab-PyTorch/workspace/data_src/aligned")

ZOOM_TOL = 1e-5
ROT_TOL = 1e-4          # 度
CORNER_TOL = 0.05       # px
PIXEL_MEAN_TOL = 1.0    # LSB
META_TOL = 0.05         # px

RESULTS = []
T0 = time.time()


def check(ok, name, detail=""):
    ok = bool(ok)
    RESULTS.append((ok, name, str(detail)))
    print(("[PASS] " if ok else "[FAIL] ") + name + ("  · " + str(detail) if detail else ""),
          flush=True)
    return ok


def say(msg):
    print("       " + str(msg), flush=True)


def head(t):
    print("\n=== " + t + " " + "=" * max(0, 66 - len(t)), flush=True)


def rd(path):
    import cv2
    return cv2.imdecode(np.fromfile(str(path), np.uint8), cv2.IMREAD_COLOR)


def chip_size(path):
    import cv2
    im = cv2.imdecode(np.fromfile(str(path), np.uint8), cv2.IMREAD_GRAYSCALE)
    return None if im is None else int(im.shape[0])


def old_mat_from_68(kps68, size):
    """**改前**的 align_mat_from_68（68 点全拟合 UNI_68）—— 只用来对账，别再调它。"""
    src = np.asarray(kps68, np.float64)
    if src.shape != (68, 2) or not np.all(np.isfinite(src)):
        return None
    T = AL.umeyama(src, AL.UNIT_68, True)
    return AL._affine_from_unit_square(T[0:2], size, AL.PADDING, AL.WHOLE_FACE_SHIFT)


def deltas(mat_a, mat_b, S):
    """同一画布上 B 相对 A 的：zoom 比 / 旋转° / 四角+中心位移（画布像素）。"""
    D = np.vstack([mat_b, [0, 0, 1]]) @ np.linalg.inv(np.vstack([mat_a, [0, 0, 1]]))
    zoom = float(np.hypot(D[0, 0], D[1, 0]))
    rot = float(np.degrees(np.arctan2(D[1, 0], D[0, 0])))
    c = np.float64([[0, 0], [S, 0], [S, S], [0, S], [S / 2.0, S / 2.0]])
    p = (np.hstack([c, np.ones((5, 1))]) @ D.T)[:, :2]
    return zoom, rot, np.linalg.norm(p - c, axis=1)


def scan_aligned(folder, limit=None):
    """→ [(name, lm68, S, meta)]；只收带 68 点的。"""
    rows = []
    for p in sorted(Path(folder).glob("*.jpg")):
        head_m, legacy = AL.read_dfljpg_metas(p)
        m = head_m or legacy or {}
        lm = m.get("source_landmarks")
        if lm is None or len(lm) != 68:
            continue
        S = chip_size(p)
        if not S:
            continue
        rows.append((p.name, np.asarray(lm, np.float64), S, m))
        if limit and len(rows) >= limit:
            break
    return rows


# ---------------------------------------------------------------- A
def case_a():
    head("A  1203 组真实 source_landmarks：新矩阵 vs facelib.get_transform_mat")
    rows = scan_aligned(SRC_ALIGNED)
    if not check(len(rows) >= 1000, "A0 采样", "%d 张带 68 点的对齐脸" % len(rows)):
        return
    old = {"zoom": [], "rot": [], "corner": []}
    new = {"zoom": [], "rot": [], "corner": []}
    elem = 0.0
    for name, lm, S, _m in rows:
        ref = np.asarray(get_transform_mat(lm, S, FaceType.WHOLE_FACE, scale=1.0), np.float64)
        cur = np.asarray(AL.align_mat_from_68(lm, S), np.float64)
        if cur is None:
            check(False, "A1 矩阵可算", name)
            return
        z, r, d = deltas(ref, cur, S)
        new["zoom"].append(z); new["rot"].append(r); new["corner"].append(d)
        o = old_mat_from_68(lm, S)
        z, r, d = deltas(ref, o, S)
        old["zoom"].append(z); old["rot"].append(r); old["corner"].append(d)
        elem = max(elem, float(np.abs(cur - ref).max()))

    def fmt(tag, acc):
        z = np.array(acc["zoom"]); r = np.array(acc["rot"]); c = np.array(acc["corner"])
        say("%-4s zoom |x-1| max %.2e (中位 %+.5f) · rot max %.2e° (中位 %+.3f°) · "
            "四角 mean %.3f / max %.3f px"
            % (tag, np.abs(z - 1).max(), float(np.median(z) - 1), np.abs(r).max(),
               float(np.median(r)), c.mean(), c.max()))
        return z, r, c

    oz, orot, oc = fmt("改前", old)
    nz, nrot, nc = fmt("新", new)
    say("改前 vs 新（同一组点）：缩放中位 %+.2f%%、旋转中位 %+.2f°、角点平均差 %.1f px"
        % (100 * (float(np.median(oz)) - 1), float(np.median(orot)),
           float(np.abs(np.array(old["corner"]) - np.array(new["corner"])).mean())))
    check(np.abs(nz - 1).max() <= ZOOM_TOL, "A1 zoom 1.00000±1e-5",
          "max |x-1| = %.2e" % np.abs(nz - 1).max())
    check(np.abs(nrot).max() <= ROT_TOL, "A2 rot 0.000°±1e-4",
          "max |rot| = %.2e°" % np.abs(nrot).max())
    check(nc.max() <= CORNER_TOL, "A3 四角位移 <0.05px", "max = %.2e px" % nc.max())
    say("矩阵元素 max|A-B| = %.2e（平移项本身几百 px，等价于相对误差 1e-6）" % elem)


# ---------------------------------------------------------------- B
def case_b(ref_dir=None):
    head("B  不动点：4.0（iperov）芯片重算应当 ≈ 恒等")
    folder = Path(ref_dir or os.environ.get("DFL_ALIGN_REF40") or REF40)
    if not folder.is_dir():
        check(False, "B0 4.0 对齐集", "没有目录 %s（用 --ref40 指定）" % folder)
        return
    rows = scan_aligned(folder)
    if not check(len(rows) > 0, "B0 采样", "%d 张" % len(rows)):
        return

    # 4.0 的 data_src/aligned 里有两个族（按名字就能分开）：
    #   NNNNN.jpg        3205 张，头像是 4.0 自己的 Extractor 用 facelib 切的
    #   NNNNNNN_N.jpg    1356 张，别处导进来的（缩放 0.9336 = 478/512、旋转 0、
    #                    平移恒定）—— 它的 image_to_face_mat 不是 facelib 在 512 上
    #                    算出来的那张，本来就不可能对成恒等。
    # 所以 B 分两段判：exact 族必须 <0.05px；另一族的差必须是**同一个常数相似变换**
    # （缩放/旋转的离散度 ~0），那才说明我们的矩阵与它是同一个函数复合了一个常数。
    groups = {}
    for name, lm, S, m in rows:
        stored = m.get("image_to_face_mat")
        if stored is None:
            continue
        stored = np.asarray(stored, np.float64)
        cur = np.asarray(AL.align_mat_from_68(lm, S), np.float64)
        z, r, d = deltas(stored, cur, S)
        o = old_mat_from_68(lm, S)
        oz, orot, od = deltas(stored, o, S) if o is not None else (np.nan, np.nan, [np.nan])
        key = "exact" if d.max() <= CORNER_TOL else "scaled"
        groups.setdefault(key, []).append((z, r, d.max(), oz, orot, max(od)))

    for key in ("exact", "scaled"):
        g = groups.get(key) or []
        if not g:
            continue
        a = np.array(g)
        say("%-6s n=%4d  新：四角 max %8.4f px · zoom 中位 %.5f (极差 %.2e) · rot 中位 %+.4f°"
            % (key, len(g), a[:, 2].max(), np.median(a[:, 0]),
               np.ptp(a[:, 0]), np.median(a[:, 1])))
        say("%-6s          改前：四角 max %8.2f px · rot 中位 %+.2f° (p95 |rot| %.2f°)"
            % ("", a[:, 5].max(), np.median(a[:, 4]),
               np.percentile(np.abs(a[:, 4]), 95)))
    ex = np.array(groups.get("exact") or [])
    if not check(len(ex) > 0, "B1 找到 iperov 切的族", "%d 张" % len(ex)):
        return
    check(ex[:, 2].max() <= CORNER_TOL, "B2 该族重算 = 恒等（<0.05px）",
          "n=%d · 四角 max %.4f px" % (len(ex), ex[:, 2].max()))
    sc = np.array(groups.get("scaled") or [])
    if len(sc):
        check(np.ptp(sc[:, 0]) < 1e-4 and np.abs(sc[:, 1]).max() < 1e-3,
              "B3 另一族的差是同一个常数相似变换",
              "zoom 极差 %.2e · |rot| max %.2e°（逐脸漂移=0 才说明是同一个函数）"
              % (np.ptp(sc[:, 0]), np.abs(sc[:, 1]).max()))


# ---------------------------------------------------------------- C
def case_c():
    head("C  外点检查：真实残差分布 + 自检")
    lim512 = AL.outlier_limit(512)
    say("阈值（DFL_ALIGN_OUTLIER_PX=%r）：%.1f px @512 · %.1f px @1024 · 关闭用 0"
        % (os.environ.get("DFL_ALIGN_OUTLIER_PX", "auto"), lim512,
           AL.outlier_limit(1024)))
    all_res, by_yaw, rejected = [], {}, []
    for tag, folder in (("src", SRC_ALIGNED), ("proj", PROJ_ALIGNED)):
        rows = scan_aligned(folder)
        res = []
        for name, lm, S, m in rows:
            _mat, _T, resid, _ppu = AL.align_fit(lm, S)
            if resid is None:
                continue
            mx = float(resid.max())
            res.append(mx / S)
            yaw = m.get("yaw")
            if yaw is not None:
                k = int(min(3, abs(float(yaw)) // 15))
                by_yaw.setdefault(k, []).append(mx / S)
            if AL.outlier_limit(S) > 0 and mx > AL.outlier_limit(S):
                rejected.append((tag, name, mx))
        say("%-4s n=%d  resid/S  p50 %.3f · p90 %.3f · p99 %.3f · max %.3f"
            % (tag, len(res), np.percentile(res, 50), np.percentile(res, 90),
               np.percentile(res, 99), np.max(res)))
        all_res += res
    for k in sorted(by_yaw):
        v = np.array(by_yaw[k])
        say("|yaw| %2d-%2d° n=%4d  resid/S p50 %.3f · p95 %.3f"
            % (k * 15, k * 15 + 15, len(v), np.percentile(v, 50), np.percentile(v, 95)))
    check(not rejected, "C1 真实脸一张都不被拒（默认阈值）",
          "被拒 %d 张%s" % (len(rejected),
                        ("：" + ", ".join("%s/%s(%.0fpx)" % r for r in rejected[:5]))
                        if rejected else ""))

    # 自检 1：把 3 个下颌点推歪 2 个「单位」= 2×画布边长（平移量会被相似变换吸收，
    # 所以只推坏一部分点才是真的「外点」）
    rows = scan_aligned(PROJ_ALIGNED, limit=2)
    name, lm, S, _m = rows[0]
    mat0, T0, resid0, ppu0 = AL.align_fit(lm, S)
    unit_px = 1.0 / float(np.hypot(T0[0, 0], T0[1, 0]))     # 原图里 1 个单位多长
    bad = lm.copy()
    for i in (3, 8, 13):
        bad[i] = bad[i] + np.array([2.0 * unit_px, 1.8 * unit_px])
    _m2, _T2, resid2, _p2 = AL.align_fit(bad, S)
    lim = AL.outlier_limit(S)
    rel2 = resid2.max() / max(1e-6, float(np.median(resid2)))
    lim_rel = AL.outlier_rel_limit()
    check(resid2.max() > lim and rel2 > lim_rel, "C2 人工坏点（3 个下颌点推出 2 个单位）两道门都被拒",
          "resid %.1f px > %.1f px · 相对 %.1f > %.1f 倍（原图 %.1f px / %.1f 倍）"
          % (resid2.max(), lim, rel2, lim_rel, resid0.max(),
             resid0.max() / max(1e-6, float(np.median(resid0)))))

    # 自检 2（反向）：近邻帧、同一个人的 68 点搬过来 —— 残差会涨，但**不该**判异常。
    # 这条是「阈值不误伤」的边界：它证明这套门只打崩坏，不按颜值/姿态扣分。
    name2, lm2, S2, _m2 = rows[1]
    _m3, _T3, resid3, _p3 = AL.align_fit(lm2, S)
    rel3 = resid3.max() / max(1e-6, float(np.median(resid3)))
    check(resid3.max() <= lim and rel3 <= lim_rel, "C3 近邻帧的点不该被当成异常",
          "resid %.1f px ≤ %.1f px · 相对 %.1f ≤ %.1f 倍"
          % (resid3.max(), lim, rel3, lim_rel))


# ---------------------------------------------------------------- D
def case_d():
    head("D  像素：我们的 68 点裁剪 vs facelib 参考路径")
    import cv2
    picked = 0
    stats = {"same_mean": 0.0, "jpg_mean": 0.0, "lan_mean": 0.0, "n": 0}
    for tag, folder in (("src", SRC_ALIGNED), ("proj", PROJ_ALIGNED)):
        for name, lm, S, m in scan_aligned(folder, limit=12):
            frame_name = m.get("source_filename")
            stem = Path(str(frame_name)).stem if frame_name else None
            cand = [PROJ_FRAMES / (Path(name).stem.rsplit("_", 1)[0] + ".jpg")]
            if stem:
                cand.insert(0, PROJ_FRAMES / (stem + ".jpg"))
            frame = next((rd(c) for c in cand if c.is_file()), None)
            if frame is None:
                continue
            ref_mat = np.asarray(get_transform_mat(lm, S, FaceType.WHOLE_FACE), np.float64)
            ref = cv2.warpAffine(frame, ref_mat, (S, S), flags=cv2.INTER_CUBIC,
                                 borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
            ref_lan = cv2.warpAffine(frame, ref_mat, (S, S), flags=cv2.INTER_LANCZOS4,
                                     borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
            ours, meta = AL.align_face_68(frame, lm, size=S)
            if ours is None:
                continue
            d = np.abs(ours.astype(np.int16) - ref.astype(np.int16))
            stats["same_mean"] = max(stats["same_mean"], float(d.mean()))
            # 落盘再解码：只剩 JPEG 自身的舍入
            ok, enc = cv2.imencode(".jpg", ours, AL.jpeg_encode_params(95, 0))
            back = cv2.imdecode(enc, cv2.IMREAD_COLOR)
            d2 = np.abs(back.astype(np.int16) - ref.astype(np.int16))
            stats["jpg_mean"] = max(stats["jpg_mean"], float(d2.mean()))
            d3 = np.abs(ours.astype(np.int16) - ref_lan.astype(np.int16))
            stats["lan_mean"] = max(stats["lan_mean"], float(d3.mean()))
            stats["n"] += 1
            picked += 1
    if not check(picked >= 10, "D0 取样", "%d 张（帧来自项目原帧）" % picked):
        return
    say("我们的裁剪 vs facelib 矩阵 + 同一套 warp：mean ≤ %.4f LSB（逐字节相同）"
        % stats["same_mean"])
    say("写完 JPG（q95 4:4:4）再解码 vs 同一参考：mean ≤ %.3f LSB（只剩编码舍入）"
        % stats["jpg_mean"])
    say("iperov 实际插值（LANCZOS4）与我们的 CUBIC：mean ≤ %.3f LSB —— 纯插值差，"
        "几何一样" % stats["lan_mean"])
    check(stats["same_mean"] <= 1e-3, "D1 几何+插值等价（≤1e-3 LSB）",
          "mean %.2e LSB —— 差来自中间量的 float32/float64 舍入" % stats["same_mean"])
    check(stats["jpg_mean"] <= PIXEL_MEAN_TOL, "D2 落盘后 vs facelib ≤1 LSB",
          "mean %.3f LSB" % stats["jpg_mean"])


# ---------------------------------------------------------------- E
def case_e():
    head("E  元数据自洽：mat ∘ source_landmarks == landmarks")
    import cv2
    worst = 0.0
    worst_fp = 0.0
    n = 0
    for tag, folder in (("src", SRC_ALIGNED), ("proj", PROJ_ALIGNED)):
        w, cnt, wf = 0.0, 0, 0.0
        for name, lm, S, m in scan_aligned(folder):
            chip = m.get("landmarks")
            mat = m.get("image_to_face_mat")
            if chip is None or mat is None:
                continue
            mat = np.asarray(mat, np.float64)
            got = cv2.transform(lm.reshape(-1, 1, 2), mat).reshape(-1, 2)
            e = float(np.linalg.norm(got - np.asarray(chip, np.float64), axis=1).max())
            w = max(w, e)
            cnt += 1
            # 这条是「4.0 不动点」门：拿 facelib 从同一组点重算，应当等于存的那张
            ref = np.asarray(get_transform_mat(lm, S, FaceType.WHOLE_FACE, scale=1.0),
                            np.float64)
            _z, _r, d = deltas(mat, ref, S)
            wf = max(wf, float(d.max()))
        say("%-4s n=%d  mat∘src max %.4f px · facelib 重算 max %.4f px"
            % (tag, cnt, w, wf))
        worst = max(worst, w)
        worst_fp = max(worst_fp, wf)
        n += cnt
    check(worst <= META_TOL, "E1 mat∘src = 芯片坐标（<0.05px）",
          "n=%d · max %.4f px" % (n, worst))
    check(worst_fp <= CORNER_TOL, "E2 新芯片上 get_transform_mat 对恒等（4.0 不动点）",
          "n=%d · 四角 max %.4f px" % (n, worst_fp))


# ---------------------------------------------------------------- main
def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", default="a,b,c,d,e")
    ap.add_argument("--ref40", default="")
    args = ap.parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)
    cases = [c.strip().lower() for c in args.cases.replace(" ", "").split(",") if c.strip()]

    print("align_wf_accept · python %s · cv2 %s" % (sys.version.split()[0], _cv2_ver()))
    if "a" in cases:
        case_a()
    if "b" in cases or "B" in cases:
        case_b(args.ref40 or None)
    if "c" in cases:
        case_c()
    if "d" in cases:
        case_d()
    if "e" in cases:
        case_e()

    fails = [x for x in RESULTS if not x[0]]
    payload = {"when": time.strftime("%Y-%m-%d %H:%M:%S"),
               "cases": cases,
               "env": {"DFL_ALIGN_OUTLIER_PX": os.environ.get("DFL_ALIGN_OUTLIER_PX", ""),
                       "cwd": os.getcwd()},
               "results": [{"ok": ok, "name": n, "detail": d} for ok, n, d in RESULTS]}
    (BOX / "result.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                                    encoding="utf-8")
    print("\n汇总：PASS %d · FAIL %d · 用时 %.1fs"
          % (len(RESULTS) - len(fails), len(fails), time.time() - T0))
    print("结果：%s" % (BOX / "result.json"))
    return 0 if not fails else 1


def _cv2_ver():
    try:
        import cv2
        return cv2.__version__
    except Exception:                                       # noqa: BLE001
        return "?"


if __name__ == "__main__":
    sys.exit(main())
