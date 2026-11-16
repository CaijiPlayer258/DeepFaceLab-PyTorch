# -*- coding: utf-8 -*-
r"""按 68 点重切对齐集（几何 = facelib/iperov，不重新检测）。

背景：2026-12 起 align_mat_from_68 只拟合 33 点，和 facelib.get_transform_mat
逐位等价。这一版之前切出来的芯片是「68 点拟合」的产物：缩放差 0.36% 中位、
旋转中位 2.5°、画布四角平均差 34 px。这里拿 **已经存在 DFLJPG 里的 68 点**
（source_landmarks）+ 原帧重切一遍，尺寸 S 保持不变，不改检测、不引入新点。

每个芯片：
  1) 读出 source_landmarks / image_to_face_mat / source_filename / xseg_mask；
  2) 找原帧（source_filename → 帧名），解码（同一个帧只解一次）；
  3) 新矩阵 = _align.align_fit(lm, S)（33 点，= facelib）；外点检查不过就**拒收**，
     不写文件、记进报告； 
  4) 写新芯片（默认 INTER_CUBIC / q95 / 4:4:4，与当前切脸默认一致）；
  5) 元数据跟着更新：landmarks（芯片坐标）= mat∘source_landmarks、
     image_to_face_mat = 新矩阵；xseg_mask 用 delta = 新 ∘ 旧⁻¹ 一起搬过去，
     不然遮罩会和脸错位（3° 在外圈能有 10 px）；pitch/yaw/roll、source_rect、
     source_filename 原样保留（source_rect 是**检测框**，重切不重新检测，所以不改）；
  6) 旧芯片先复制到 aligned.bak_YYYYMMDD/，一个都不覆盖丢失。

跑法（仓库根，server venv）：
    runtime\server\Scripts\python.exe app\tools\align_recut.py --dry-run
    runtime\server\Scripts\python.exe app\tools\align_recut.py --sets src,proj
    （--limit 20 试跑；--no-backup 不备份；--interp lanczos4 换成 iperov 实际用的插值）

产物：%TEMP%\dfl_align_ab\recut\  （report.json + 每批 40 张对照图）
"""
from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import sys
import time
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[2]
APP = REPO / "app"
TEMP = Path(os.environ.get("TEMP") or os.environ.get("TMP") or ".")
OUT = TEMP / "dfl_align_ab" / "recut"
sys.path.insert(0, str(APP))
sys.path.append(str(APP / "engine"))

from server import _align as AL                                     # noqa: E402
from facelib import FaceType                                        # noqa: E402
from facelib.LandmarksProcessor import get_transform_mat              # noqa: E402

PROJ_DIR = REPO / "workspace" / "merge" / "9月29日 (5)"
SETS = {
    "src": {"aligned": REPO / "workspace/data_src/aligned", "frames": PROJ_DIR},
    "proj": {"aligned": PROJ_DIR / "aligned", "frames": PROJ_DIR},
}
INTERP = {"cubic": cv2.INTER_CUBIC, "lanczos4": cv2.INTER_LANCZOS4,
          "linear": cv2.INTER_LINEAR}


def say(msg=""):
    print(str(msg), flush=True)


def head(t):
    say("\n=== " + t + " " + "=" * max(0, 66 - len(t)))


def rd(path):
    return cv2.imdecode(np.fromfile(str(path), np.uint8), cv2.IMREAD_COLOR)


def _scale(k):
    return np.array([[float(k), 0.0, 0.0], [0.0, float(k), 0.0], [0.0, 0.0, 1.0]])


def warp_mask(mask_png, m_from, m_to, new_mat, old_mat):
    """把 xseg_mask 按 delta = 新 ∘ 旧⁻¹ 搬到新画布（分辨率保持 m_from→m_to）。"""
    if not mask_png:
        return None
    m = cv2.imdecode(np.frombuffer(mask_png, np.uint8), cv2.IMREAD_GRAYSCALE)
    if m is None:
        return None
    h, w = m.shape[:2]
    delta = np.vstack([new_mat, [0, 0, 1]]) @ np.linalg.inv(np.vstack([old_mat, [0, 0, 1]]))
    # 画布坐标 ↔ 遮罩像素坐标（同一个归一化画布，只是分辨率不同）
    fx = float(w) / float(m_from or w)
    fy = float(h) / float(m_from or h)
    M = (_scale(fx) @ delta @ np.diag([1.0 / fx if fx else 1.0, 1.0 / fy if fy else 1.0, 1.0]))
    out = cv2.warpAffine(m, M[0:2], (w, h), flags=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return AL.encode_xseg_mask(out.astype(np.float32) / 255.0)


def recut_set(tag, cfg, args, rng):
    head("%s · %s" % (tag, cfg["aligned"]))
    aligned, frames = Path(cfg["aligned"]), Path(cfg["frames"])
    files = sorted(aligned.glob("*.jpg"))
    if args.limit:
        files = files[:args.limit]
    stamp = time.strftime("%Y%m%d")
    bak = aligned.parent / ("aligned.bak_%s" % stamp)
    if not args.no_backup and not args.dry_run:
        bak.mkdir(parents=True, exist_ok=True)

    cache = {"path": None, "img": None}
    st = {"n": 0, "ok": 0, "rejected": 0, "no_frame": 0, "no_meta": 0, "worst": 0.0,
          "px_ref_mean": 0.0, "px_ref_max": 0.0, "px_old_mean": 0.0,
          "px_lan_mean": 0.0, "meta_max": 0.0, "t_warp": 0.0, "t_jpeg": 0.0,
          "size_changed": 0}
    rejected, missing = [], []
    plates = []
    t0 = time.time()
    for p in files:
        st["n"] += 1
        head_m, legacy = AL.read_dfljpg_metas(p)
        # 头部那份是引擎读的；EOI 前那份（老布局）只在缺键时补 —— 两份都带上，
        # 免得把某个只存在于 legacy 里的键（历史上是 xseg_mask）悄悄丢掉。
        meta = dict(legacy or {})
        meta.update(head_m or {})
        if not meta:
            st["no_meta"] += 1
            continue
        lm = meta.get("source_landmarks")
        old_mat = meta.get("image_to_face_mat")
        if lm is None or len(lm) != 68 or old_mat is None:
            st["no_meta"] += 1
            continue
        lm = np.asarray(lm, np.float64)
        old_mat = np.asarray(old_mat, np.float64)

        chip_old = rd(p)
        if chip_old is None:
            st["no_meta"] += 1
            continue
        S = int(chip_old.shape[0])

        # ---- 原帧 ----
        src = str(meta.get("source_filename") or "").strip()
        fname = Path(src).stem if src else p.stem.rsplit("_", 1)[0]
        fpath = frames / (fname + ".jpg")
        if not fpath.is_file():
            alt = frames / (p.stem.rsplit("_", 1)[0] + ".jpg")
            fpath = alt if alt.is_file() else fpath
        if not fpath.is_file():
            st["no_frame"] += 1
            missing.append(p.name)
            continue
        if cache["path"] != str(fpath):
            cache["path"] = str(fpath)
            cache["img"] = rd(fpath)
        frame = cache["img"]
        if frame is None:
            st["no_frame"] += 1
            missing.append(p.name)
            continue

        # ---- 新矩阵 + 外点检查 ----
        t_a = time.perf_counter()
        mat, T, resid, ppu = AL.align_fit(lm, S)
        if mat is None:
            st["rejected"] += 1
            rejected.append((p.name, "矩阵算不出来（点数/数值退化）"))
            continue
        rmax = float(resid.max())
        rrel = rmax / max(1e-6, float(np.median(resid)))
        lim, lim_rel = AL.outlier_limit(S), AL.outlier_rel_limit()
        if (lim > 0 and rmax > lim) or (lim_rel > 0 and rrel > lim_rel):
            st["rejected"] += 1
            rejected.append((p.name, "残差 %.1f px / %.1f 倍 > %.1f / %.1f"
                             % (rmax, rrel, lim, lim_rel)))
            continue

        chip_new = cv2.warpAffine(frame, mat, (S, S), flags=INTERP[args.interp],
                                  borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
        st["t_warp"] += time.perf_counter() - t_a

        # ---- 参考：iperov 的矩阵（facelib）+ 同一套 warp ----
        ref_mat = np.asarray(get_transform_mat(lm, S, FaceType.WHOLE_FACE, scale=1.0), np.float64)
        ref = cv2.warpAffine(frame, ref_mat, (S, S), flags=INTERP[args.interp],
                             borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
        ref_lan = cv2.warpAffine(frame, ref_mat, (S, S), flags=cv2.INTER_LANCZOS4,
                                 borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
        d_ref = np.abs(chip_new.astype(np.int16) - ref.astype(np.int16))
        st["px_ref_mean"] = max(st["px_ref_mean"], float(d_ref.mean()))
        st["px_ref_max"] = max(st["px_ref_max"], float(d_ref.max()))
        st["px_lan_mean"] = max(st["px_lan_mean"],
                                float(np.abs(chip_new.astype(np.int16)
                                             - ref_lan.astype(np.int16)).mean()))
        d_old = np.abs(chip_new.astype(np.int16) - chip_old.astype(np.int16))
        st["px_old_mean"] = max(st["px_old_mean"], float(d_old.mean()))

        # ---- 元数据 ----
        new_meta = dict(meta)
        lm_chip = cv2.transform(lm.reshape(-1, 1, 2), mat).reshape(-1, 2)
        new_meta["source_landmarks"] = np.round(lm, 4).tolist()
        new_meta["landmarks"] = np.round(lm_chip, 4).tolist()
        new_meta["image_to_face_mat"] = np.asarray(mat, np.float64).tolist()
        new_meta["align"] = "68"
        new_meta["align_kernel"] = "facelib33"
        new_meta["landmarks_source"] = meta.get("landmarks_source") or "model68"
        new_meta["align_resid_px"] = round(rmax, 3)
        new_meta["align_resid_rel"] = round(rrel, 3)
        mk = meta.get("xseg_mask")
        if mk:
            nm = warp_mask(mk, S, S, mat, old_mat)
            if nm:
                new_meta["xseg_mask"] = nm
            else:
                new_meta.pop("xseg_mask", None)
        for k in ("face_mat", "landmarks_68", "source_landmarks_68"):
            if k in new_meta:                       # 老项目里的别名键一起更新
                new_meta[k] = (new_meta["image_to_face_mat"] if k == "face_mat"
                               else new_meta["landmarks"] if k == "landmarks_68"
                               else new_meta["source_landmarks"])
        new_meta.pop("rejected", None)      # 重切成功就不该再有这个键

        if args.dry_run:
            st["ok"] += 1
            continue

        if not args.no_backup:
            dst = bak / p.name
            if not dst.exists():
                shutil.copy2(p, dst)
        t_j = time.perf_counter()
        AL.write_dfljpg(p, chip_new, new_meta)
        st["t_jpeg"] += time.perf_counter() - t_j
        st["ok"] += 1

        # 元数据自洽：mat∘src 必须落在刚写的 landmarks 上
        back = cv2.transform(lm.reshape(-1, 1, 2), mat).reshape(-1, 2)
        e = float(np.linalg.norm(back - np.asarray(new_meta["landmarks"]), axis=1).max())
        st["meta_max"] = max(st["meta_max"], e)

        if len(plates) < args.plates and rng.random() < 0.5:
            plates.append((p.name, chip_old, chip_new, ref))

    dt = time.time() - t0
    res = {"set": tag, "aligned": str(aligned), "backup": "" if args.no_backup else str(bak),
           "n": st["n"], "ok": st["ok"], "rejected": st["rejected"],
           "no_frame": st["no_frame"], "no_meta": st["no_meta"],
           "seconds": round(dt, 2),
           "ms_per_face": round(1000.0 * dt / max(1, st["n"]), 1),
           "warp_ms": round(1000.0 * st["t_warp"] / max(1, st["n"]), 3),
           "jpeg_ms": round(1000.0 * st["t_jpeg"] / max(1, st["ok"]), 3),
           "px_vs_iperov_mean": round(st["px_ref_mean"], 4),
           "px_vs_iperov_max": round(st["px_ref_max"], 2),
           "px_vs_iperov_lanczos_mean": round(st["px_lan_mean"], 4),
           "px_vs_old_chip_mean": round(st["px_old_mean"], 3),
           "meta_roundtrip_max_px": round(st["meta_max"], 5),
           "rejected_list": rejected[:20], "missing_list": missing[:20]}
    for k, v in res.items():
        if k not in ("rejected_list", "missing_list"):
            say("%-24s %s" % (k, v))
    if rejected[:5]:
        say("拒收样本：" + "; ".join("%s (%s)" % r for r in rejected[:5]))
    if missing[:5]:
        say("找不到原帧：" + ", ".join(missing[:5]))
    return res, plates


def plate_image(rows, tag):
    """每行 3 格：旧芯片 | 新芯片 | iperov（facelib 矩阵 + 同一插值）。"""
    tiles = []
    for name, a, b, c in rows:
        w = 240
        row = np.hstack([cv2.resize(x, (w, w), interpolation=cv2.INTER_AREA) for x in (a, b, c)])
        cv2.putText(row, name, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        for i, t in enumerate(("old", "new", "iperov")):
            cv2.putText(row, t, (6 + i * w, w - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 255, 255), 1)
        tiles.append(row)
    if not tiles:
        return None
    per = 8
    sheets = []
    for i in range(0, len(tiles), per):
        block = tiles[i:i + per]
        while len(block) < per:
            block.append(np.zeros_like(block[0]))
        sheets.append(np.vstack(block))
    return sheets


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", default="src,proj")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--plates", type=int, default=40)
    ap.add_argument("--interp", default="cubic", choices=sorted(INTERP))
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--no-backup", action="store_true")
    ap.add_argument("--seed", type=int, default=20261205)
    args = ap.parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)

    say("align_recut · python %s · cv2 %s · interp=%s%s"
        % (sys.version.split()[0], cv2.__version__, args.interp,
           " · DRY-RUN" if args.dry_run else ""))
    say("阈值：DFL_ALIGN_OUTLIER_PX=%r → %.0f%% 画布 · DFL_ALIGN_OUTLIER_REL=%r"
        % (os.environ.get("DFL_ALIGN_OUTLIER_PX", "auto"), 100 * AL.OUTLIER_FRACTION,
           os.environ.get("DFL_ALIGN_OUTLIER_REL", "auto")))

    report = {"when": time.strftime("%Y-%m-%d %H:%M:%S"), "interp": args.interp,
              "dry_run": bool(args.dry_run), "sets": []}
    for tag in [s.strip() for s in args.sets.split(",") if s.strip()]:
        cfg = SETS.get(tag)
        if not cfg:
            say("未知的集合：%s" % tag)
            continue
        res, plates = recut_set(tag, cfg, args, rng)
        report["sets"].append(res)
        if plates and not args.dry_run:
            sheets = plate_image(plates, tag)
            paths = []
            for i, sh in enumerate(sheets or []):
                p = OUT / ("plates_%s_%d.png" % (tag, i + 1))
                cv2.imwrite(str(p), sh)
                paths.append(str(p))
            res["plates"] = paths
            say("对照图：%s" % "; ".join(paths))
    (OUT / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2),
                                    encoding="utf-8")
    say("\n报告：%s" % (OUT / "report.json"))
    fails = [s for s in report["sets"] if s["ok"] == 0]
    return 0 if not fails else 1


if __name__ == "__main__":
    sys.exit(main())
