# -*- coding: utf-8 -*-
r"""检测器 EP 对拍（**第二层**）：5 点 → 68 点 → 对齐脸。det_trt_parity.py 只量了第一层。

为什么要多这一层（2026-11 补的）
────────────────────────────────────────────────────────────────────────
det_trt_parity.py 只比 boxes + 5 点（--max-px 0.3）。5 点在下游只干一件事：算
coarse_align_from_5 给 2D106 准备 192 画布。**几何是 68 点定的**：

    检测器 5 点 → coarse_align_from_5（粗裁，只影响 106 看得清不清楚）
    → 2D106 → landmark106to68（按下标挑，不插值）→ landmarks68_in_source
    → align_face_68（33 点相似变换）→ 对齐脸 → dfm 输入 → 成片

所以「5 点 ≤0.3px 就算过」这条判据不成立。本机 40 帧真实素材（retinaface_10g，
ORT-CUDA fp32 vs workspace 里的 TRT bf16，同帧同权重）实测：

    bf16 的**真实**数值差          ：5 点 max 0.055 px / 中位 0.015 px（未 round）
    _faces_from_fwd 把 kps round(1) ：0.015px 的差被**量化**成 0.0 或 0.1 px
                                      40 帧里 17 帧翻到 0.1，23 帧正好同格 → 逐位相同
    0.1px 的 5 点差 → 粗裁 192      ：7~21% 的像素变了（max 78 LSB）
    → 2D106 → 68 点                 ：max 1.54 px（脸宽 117px 的 1.3%），中位帧 0
    → 终裁矩阵                      ：四角 max 15.7 px、zoom 0.79%、rot 0.63°
    → 对齐脸（376px 画布）          ：max|Δ| 123 LSB，26% 的通道样本 >2LSB
    → 合成后整帧                    ：max 14 LSB、0.02% 的像素 >2LSB（dump 实测）

结论写成一句话：**0.1px 的量子级差被 106 放大约 6 倍，落到亚像素几何上；但因为
对齐与反变换用的是同一组 68 点，几何自洽，成片只差 0.02% 的像素。**

它查四件事（一一对应上面的流水线）：
  ① 5 点：round 后的位移（和 det_trt_parity 同一口径）+ **未 round 的原始差**
  ② 68 点：max / p95 / mean + 按组（轮廓 / 眉 / 鼻 / 眼 / 嘴）+ 归一化到脸宽
  ③ 对齐脸：max / mean / >2LSB 通道占比（尺寸按生产的 2×框高，不写死 416）
  ④ 链路自证：同一份 5 点跑两遍必须逐位相同；106 单张 vs 批 1 必须逐位相同
     —— 这两条不成立，②③ 的差就归不了因（现在实测都是 0.000000）

成片 md5 不在这个脚本里（要真 dfm + 真导出，是另一层），但口径必须一起看：
**同一配置复跑两次成片 md5 就不同**（本机 13 次导出 11 个不同 md5；同臂复跑的
帧均值 0.94~0.96 LSB，和跨臂 0.96 完全同量级）—— 所以 md5 只能当「有没有动过」
的哨兵，不能当「差多少」的尺子，更不能当「换 EP 有没有换脸」的判据。

跑法（仓库根）：
    runtime\server\Scripts\python.exe app\tools\accept\det68_trt_parity.py --frames 40
    ... --crops %TEMP%\det68 --json out.json     # 把最差帧的裁图/热图落下来看
退出码：0 = 68 点 max ≤ --max-68-px 且 对齐脸 >--max-lsb 占比 = 0；
        2 = 超了 / 量不出来。超了不等于「不许换 EP」，等于「要不要接受这个基线变更
        由人定，而且成片也得重新对拍」。
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
APP = REPO / "app"
if str(APP) not in sys.path:
    sys.path.insert(0, str(APP))
os.environ.setdefault("DFL_WORKSPACE", str(REPO / "workspace"))
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

import numpy as np                                             # noqa: E402

GROUPS = (("轮廓 0-16", list(range(0, 17))), ("眉 17-26", list(range(17, 27))),
          ("鼻 27-35", list(range(27, 36))), ("眼 36-47", list(range(36, 48))),
          ("嘴 48-67", list(range(48, 68))))


def _q(v, p):
    s = sorted(v)
    return s[min(len(s) - 1, max(0, int(round(p * (len(s) - 1)))))] if s else float("nan")


def _frames(video, det, want, stride=1):
    """顺序解码（和抽取同一条 ffmpeg rawvideo 口径），攒 want 帧「检测器认为有脸」的。"""
    import subprocess
    import imageio_ffmpeg
    ff = imageio_ffmpeg.get_ffmpeg_exe()
    proc = subprocess.Popen([ff, "-v", "error", "-i", str(video), "-vsync", "0",
                             "-f", "rawvideo", "-pix_fmt", "bgr24", "-"],
                            stdout=subprocess.PIPE)
    out, i, fsz = [], 0, None
    while len(out) < want:
        if fsz is None:
            import cv2
            cap = cv2.VideoCapture(str(video))
            W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()
            fsz, shape = W * H * 3, (H, W, 3)
            print("[取帧] 源 %dx%d" % (W, H), flush=True)
        b = proc.stdout.read(fsz)
        if len(b) < fsz:
            break
        f = np.frombuffer(b, np.uint8).reshape(shape).copy()
        det.thresh = 0.5
        if i % stride == 0 and det.detect(f):
            out.append((i, f))
        i += 1
    try:
        proc.stdout.close()
        proc.kill()
    except Exception:                                          # noqa: BLE001
        pass
    return out, i


def raw5_delta(ort, trt, img):
    """**未 round** 的 5 点差（canvas px）→ 原图 px。bf16 的真实数值差只在这里看得见。

    _faces_from_fwd 把 kps round(1)，所以 detect() 的输出已经被量化到 0.1px 网格上：
    「两臂不同」要么是 0，要么是 0.1（对角 0.1414）。真实差多少，只有比原始输出才知道。
    """
    ctx = ort.prep(img)
    blob = np.empty((1, 3, 640, 640), np.float32)
    blob[0] = ctx["canvas"][:, :, ::-1].transpose(2, 0, 1)
    blob -= 127.5
    blob *= (1.0 / 128.0)
    outs = [s.run(None, {ort.input_name: blob}) for s in (ort.sess, trt.sess)]
    picked = []
    for o in outs:
        best = (0.0, None)
        for idx, stride in enumerate((8, 16, 32)):
            S = 640
            h = w = S // stride
            c = np.stack(np.mgrid[:h, :w][::-1], axis=-1).astype(np.float32)
            c = (c * stride).reshape(-1, 2)
            c = np.stack([c] * 2, axis=1).reshape(-1, 2)
            sc = np.asarray(o[idx]).reshape(-1)
            kk = np.asarray(o[idx + 6]).reshape(-1, 10) * stride
            n = min(len(sc), len(c), len(kk))
            if n == 0:
                continue
            j = int(np.argmax(sc[:n]))
            if sc[j] > best[0]:
                kps = kk.reshape(-1, 5, 2)[j] + c[j]
                best = (float(sc[j]), kps)
        picked.append(best[1])
    if picked[0] is None or picked[1] is None:
        return float("nan"), float("nan")
    d = float(np.abs(picked[0] - picked[1]).max())
    return d, d / float(ctx["scale"])


def main(argv=None):
    ap = argparse.ArgumentParser(description="检测器 EP 对拍（第二层：68 点 + 对齐脸）")
    ap.add_argument("--det", default="retinaface_10g", choices=("retinaface_10g",))
    ap.add_argument("--frames", type=int, default=40)
    ap.add_argument("--video", default="")
    ap.add_argument("--stride", type=int, default=1, help="每隔几帧取一帧（默认 1）")
    ap.add_argument("--max-68-px", type=float, default=0.1)
    ap.add_argument("--max-lsb", type=float, default=2.0,
                    help="允许的对齐脸最大逐像素差（LSB）；默认 2 = 只容忍取整噪声")
    ap.add_argument("--crops", default="", help="把最差帧的裁图/热图写到这个目录")
    ap.add_argument("--json", default="")
    a = ap.parse_args(argv)

    from server._common.paths import WORKSPACE
    from server._vision import align as AL
    from server._vision import detect as dk
    from server._landmarks106 import InsightFace106

    W = REPO / "weights"
    p = W / "retinaface_10g.onnx"
    if not p.is_file():
        print("退出码 2：找不到 weights/retinaface_10g.onnx")
        return 2

    # ---- 1) 接线（和 det_trt_parity 同一套断言：引擎不接上，后面全是量错东西）----
    bad = 0
    for flag, want in (("1", "TrtSession"), ("0", "InferenceSession")):
        os.environ["DFL_DET_TRT"] = flag
        from server.backend_extract import detectors as det_mod
        det_mod._det_cache.clear()
        got = type(det_mod._get_detector("retinaface_10g").sess).__name__
        ok = (got == want)
        print("[接线] DFL_DET_TRT=%s → %-16s（要 %s）%s"
              % (flag, got, want, "OK" if ok else "**不一致**"))
        bad += 0 if ok else 1
    det_mod._det_cache.clear()
    if bad:
        print("退出码 2：检测器 EP 接线不对")
        return 2

    ort = dk.RetinaFaceDet(p, 640)
    trt = dk.RetinaFaceDet(p, 640, workspace=str(WORKSPACE), engine_name="retinaface_10g")
    lm = InsightFace106(str(W / "insightface_2d106.onnx"), workspace=str(WORKSPACE),
                        engine_name="insightface_2d106")
    print("[对拍] ORT=%s / TRT=%s(%s) / 106=%s"
          % (type(ort.sess).__name__, type(trt.sess).__name__,
             getattr(trt.sess, "precision", "-"), type(lm.sess).__name__), flush=True)
    ort.detect(np.zeros((64, 64, 3), np.uint8))
    trt.detect(np.zeros((64, 64, 3), np.uint8))
    lm.landmarks68(np.zeros((192, 192, 3), np.uint8))

    video = Path(a.video) if a.video else (WORKSPACE / "9月29日 (5).mp4")
    if not video.is_file():
        cand = sorted(WORKSPACE.glob("*.mp4"))
        if not cand:
            print("退出码 2：workspace 里没有 mp4，给 --video")
            return 2
        video = cand[0]
    t0 = time.time()
    items, scanned = _frames(video, ort, a.frames, max(1, a.stride))
    print("[取帧] %s：扫 %d 帧拿到 %d 帧有脸（%.1fs）"
          % (video.name, scanned, len(items), time.time() - t0), flush=True)
    if not items:
        print("退出码 2：一帧有脸的都没取到")
        return 2

    crops = Path(a.crops) if a.crops else None
    if crops:
        crops.mkdir(parents=True, exist_ok=True)

    per = []
    d5, d5raw, d68, rel, chain_self, batch_diff = [], [], [], [], [], []
    grp = {k: [] for k, _ in GROUPS}
    face_max, face_mean, face_gt2, corner, zoom, rot = [], [], [], [], [], []
    worst = (0.0, None)

    for n, (idx, img) in enumerate(items):
        ctx = ort.prep(img)                       # prep 与 EP 无关，两臂共用一张画布
        A = ort.detect_ctx(ctx)
        B = trt.detect_ctx(ctx)
        if not A or not B:
            print("  第 %d 帧：一臂没检出，跳过" % idx)
            continue
        ka = np.asarray(A[0]["kps"], np.float64)
        kb = np.asarray(B[0]["kps"], np.float64)
        ba = np.asarray(A[0]["box"], np.float64)
        bb = np.asarray(B[0]["box"], np.float64)
        # 生产口径：side = round(max(bw,bh)*2)，偶数化（frames.py / pipeline_fused.py 同一行）
        side = int(round(max(ba[2] - ba[0], ba[3] - ba[1]) * 2.0))
        side -= side % 2

        def chain(kps, box, size):
            cmat = AL.coarse_align_from_5(kps)
            coarse = _warp(img, cmat)
            l68 = lm.landmarks68(coarse)
            pts = AL.landmarks68_in_source(l68, cmat)
            crop, meta = AL.align_face_68(img, pts, box=[float(v) for v in box], size=size)
            return cmat, coarse, l68, pts, crop, meta

        cma, coarse_a, l68a, pts_a, cr_a, ma = chain(ka, ba, side)
        cmb, coarse_b, l68b, pts_b, cr_b, mb = chain(kb, bb, side)
        # ④ 链路自证：同一份输入再跑一次 / 106 走批 1
        _, _, _, pts_a2, _, _ = chain(ka, ba, side)
        chain_self.append(float(np.linalg.norm(pts_a - pts_a2, axis=1).max()))
        batch_diff.append(float(np.abs(np.asarray(
            lm.landmarks68_batch([coarse_a])[0]) - np.asarray(l68a)).max()))

        dd5 = float(np.linalg.norm(ka - kb, axis=1).max())
        d5.append(dd5)
        r_raw, r_raw_src = raw5_delta(ort, trt, img)
        d5raw.append(r_raw_src)
        d68v = np.linalg.norm(pts_a - pts_b, axis=1)
        bw = max(1e-6, float(ba[2] - ba[0]))
        d68.append(float(d68v.max()))
        rel.append(float(d68v.max() / bw))
        for k, ids in GROUPS:
            grp[k].append(float(d68v[ids].max()))

        row = {"i": int(idx), "side": side, "box": [round(v, 1) for v in ba],
               "kps_same": bool(np.array_equal(ka, kb)),
               "d5": round(dd5, 4), "d5_raw_src": round(r_raw_src, 4),
               "d68_max": round(float(d68v.max()), 4),
               "d68_mean": round(float(d68v.mean()), 4),
               "d68_p95": round(_q(list(d68v), .95), 4),
               "d68_rel": round(float(d68v.max() / bw), 6),
               "coarse_diff_pct": round(float(np.count_nonzero(coarse_a != coarse_b))
                                        / coarse_a.size * 100, 3)}
        if cr_a is not None and cr_b is not None:
            Ma, Mb = np.asarray(ma["image_to_face_mat"]), np.asarray(mb["image_to_face_mat"])
            sa_ = float(np.hypot(Ma[0, 0], Ma[1, 0]))
            sb_ = float(np.hypot(Mb[0, 0], Mb[1, 0]))
            ra_ = float(np.degrees(np.arctan2(Ma[1, 0], Ma[0, 0])))
            rb_ = float(np.degrees(np.arctan2(Mb[1, 0], Mb[0, 0])))
            cor = np.float32([[0, 0], [side, 0], [side, side], [0, side]]).reshape(-1, 1, 2)
            import cv2
            pa = cv2.transform(cor, Ma).reshape(-1, 2)
            pb = cv2.transform(cor, Mb).reshape(-1, 2)
            corner.append(float(np.linalg.norm(pa - pb, axis=1).max()))
            zoom.append(abs(sb_ / sa_ - 1) * 100)
            rot.append(abs(rb_ - ra_))
            d = np.abs(cr_a.astype(np.int16) - cr_b.astype(np.int16))
            face_max.append(int(d.max()))
            face_mean.append(float(d.mean()))
            face_gt2.append(float(np.count_nonzero(d > a.max_lsb)) / d.size * 100)
            row.update({"face_max": int(d.max()), "face_mean": round(float(d.mean()), 5),
                        "face_gt2_pct": round(face_gt2[-1], 4),
                        "corner_px": round(corner[-1], 4),
                        "zoom_pct": round(zoom[-1], 6), "rot_deg": round(rot[-1], 6)})
            if crops is not None and d.max() >= worst[0]:
                worst = (int(d.max()), int(idx))
                _dump(crops, idx, cr_a, cr_b, d)
        per.append(row)
        if (n + 1) % 10 == 0:
            print("  %d/%d 帧 %d：5点 %.3f（原始 %.3f）· 粗裁差 %.1f%% · 68点 %.3f · 脸 max %s"
                  % (n + 1, len(items), idx, dd5, r_raw_src, row["coarse_diff_pct"],
                     row["d68_max"], row.get("face_max", "-")), flush=True)

    # ---------------- 报告 ----------------
    print("")
    print("=" * 78)
    _sides = sorted({r["side"] for r in per})
    print("检测器 EP 对拍（第二层）：%s · %d 帧 · 对齐脸画布 %d~%d px（%d 种）"
          % (a.det, len(per), _sides[0], _sides[-1], len(_sides)))
    print("① 5 点位移  round 后 max %.4f · p95 %.4f（%.0f%% 的帧逐位相同）"
          % (max(d5), _q(d5, .95), 100.0 * sum(1 for r in per if r["kps_same"]) / len(per)))
    print("   未 round 的原始差 max %.4f px / 中位 %.4f px —— **这才是 bf16 的真实数值差**"
          % (max(d5raw), statistics.median(d5raw)))
    print("② 68 点位移 max %.4f · p95 %.4f · 中位 %.4f px ；归一化(px/脸宽) max %.6f"
          % (max(d68), _q(d68, .95), statistics.median(d68), max(rel)))
    for k, _ in GROUPS:
        v = grp[k]
        print("   %-10s max %.4f · p95 %.4f · 中位 %.4f px"
              % (k, max(v), _q(v, .95), statistics.median(v)))
    if face_max:
        print("③ 对齐脸 max|Δ| %d LSB · mean 中位 %.5f · >%.0fLSB 通道占比 max %.2f%%（中位 %.2f%%）"
              % (max(face_max), statistics.median(face_mean), a.max_lsb,
                 max(face_gt2), statistics.median(face_gt2)))
        print("   终裁矩阵：四角 max %.3f px · 中位 %.4f ；zoom max %.4f%% ；rot max %.4f°"
              % (max(corner), statistics.median(corner), max(zoom), max(rot)))
    print("④ 链路自证：同输入复跑 68 点 max %.6f px ；106 单张 vs 批1 max %.6f（必须是 0）"
          % (max(chain_self), max(batch_diff)))
    ok_lm = (max(d68) <= a.max_68_px) and (max(chain_self) < 1e-9)
    ok_face = (not face_gt2) or (max(face_gt2) == 0.0 and max(face_max) <= a.max_lsb)
    if not face_max:
        ok_face = False
        print("   ** 有帧判 rejected，对齐脸没得比 —— 这一轮不算过")
    print("结论：68 点 %s；对齐脸 %s"
          % ("在 %.2fpx 内" % a.max_68_px if ok_lm else "**超出 %.2fpx**" % a.max_68_px,
             "逐像素一致" if ok_face else "**变了**"))
    if not (ok_lm and ok_face):
        print("       → 换 EP 会改几何。要不要接受这个基线变更由人定；接受的话成片 md5")
        print("         金样本必须重新录，而且以后任何 EP/精度切换都得走这一层 + 成片对拍。")
    if crops is not None and worst[1] is not None:
        print("  最差帧 %d 的裁图/热图：%s" % (worst[1], crops))
    if a.json:
        Path(a.json).write_text(json.dumps({"per": per, "frames": len(per), "scanned": scanned,
                                            "d5": d5, "d5_raw": d5raw, "d68": d68,
                                            "rel": rel, "group": grp,
                                            "chain_self": chain_self,
                                            "aligned_face": {"max": face_max, "mean": face_mean,
                                                             "gt2_pct": face_gt2},
                                            "corner_px": corner, "zoom_pct": zoom,
                                            "rot_deg": rot},
                                           ensure_ascii=False, indent=1, default=float),
                               "utf-8")
        print("  JSON：%s" % a.json)
    return 0 if (ok_lm and ok_face) else 2


def _warp(img, cmat):
    import cv2
    from server._vision.align import COARSE_SIZE
    return cv2.warpAffine(img, cmat, (COARSE_SIZE, COARSE_SIZE), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))


def _dump(root, idx, cr_a, cr_b, d):
    """最差帧落盘：两张裁图 + |Δ|x8 热图（看「肉眼能不能辨」就靠这三张）。"""
    import cv2
    cv2.imwrite(str(root / ("face_%06d_ort.png" % idx)), cr_a)
    cv2.imwrite(str(root / ("face_%06d_trt.png" % idx)), cr_b)
    cv2.imwrite(str(root / ("diff_%06d_x8.png" % idx)),
                cv2.applyColorMap(np.clip(d * 8, 0, 255).astype(np.uint8), cv2.COLORMAP_INFERNO))


if __name__ == "__main__":
    sys.exit(main())
