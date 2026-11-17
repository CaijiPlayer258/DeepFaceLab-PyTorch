# -*- coding: utf-8 -*-
r"""人脸超分的两条硬约束：**真的改了像素** + **不引入帧间抖动**。

跑法（仓库根）：
    runtime\server\Scripts\python.exe app\tools\accept\sr_accept.py

为什么要它：这一块踩过一次「静默 no-op」—— 推理照跑、结果被丢，
日志/计时/帧数全正常，只有像素没变（A/B 图左右两半 md5 相同）。人眼很难发现，脚本可以。

判据：
  A 单张正脸：apply_roi 前后平均绝对差 > 1 LSB（256 约 3.6、512 约 4.7）
  B 时序：SR 开/关各渲 6 帧，相邻帧差的中位数不高于关闭时的 1.3 倍
     （GAN 会给脸加"风格"，若按帧/按脸有条件地开，就会帧间跳 —— 这条守住它）
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
PY = REPO / "runtime" / "server" / "Scripts" / "python.exe"
TMP = Path(os.environ.get("TEMP") or ".")
PASS, FAIL = [], []


def check(ok, name, detail=""):
    (PASS if ok else FAIL).append(name)
    print(("  PASS " if ok else "  FAIL ") + name + ("  · " + detail if detail else ""))


def one_face():
    import numpy as np
    import cv2
    sys.path.insert(0, str(REPO / "app"))
    from server.backend_extract.media import _imread_any
    from server._vision.align import read_dfljpg_meta
    from server.backend_merge import facesr

    root = REPO / "workspace" / "data_src" / "aligned"
    pick, best = None, None
    for p in sorted(root.glob("*.jpg"))[:400]:
        try:
            m = read_dfljpg_meta(str(p)) or {}
            if not m.get("landmarks"):
                continue
            s = abs(float(m.get("yaw") or 0)) + 0.5 * abs(float(m.get("pitch") or 0))
            if best is None or s < best:
                pick, best = p, s
        except Exception:                                   # noqa: BLE001
            continue
    if pick is None:
        check(False, "A 找到一张可用的脸", "workspace/data_src/aligned 里没有 DFLJPG")
        return
    img = cv2.resize(_imread_any(pick), (512, 512), interpolation=cv2.INTER_LINEAR)
    mask = np.ones(img.shape[:2], np.float32)
    out = facesr.apply_roi(img, mask, 0.6, {})
    d = float(np.mean(np.abs(out.astype(np.int16) - img.astype(np.int16))))
    check(d > 1.0, "A 超分确实改了像素（不是静默 no-op）",
          "平均差 %.2f LSB（%s）" % (d, facesr.info().get("model") or "-"))


def render(tag, sr):
    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"
    env["DFL_SR"] = sr
    wd = TMP / ("sr_accept_" + tag)
    subprocess.run([str(PY), str(REPO / "app" / "tools" / "bench" / "fused_pipeline_bench.py"),
                    "--frames", "6", "--modes", "B", "--workdir", str(wd)],
                   cwd=str(REPO), env=env, stdout=subprocess.DEVNULL,
                   stderr=subprocess.DEVNULL)
    return sorted((wd / "B_out").glob("*.png"))


def temporal():
    import numpy as np
    import cv2
    a = render("off", "off")
    b = render("on", "mid")
    if len(a) < 4 or len(b) < 4:
        check(False, "B 渲染出足够的帧", "off=%d on=%d" % (len(a), len(b)))
        return

    def jitter(paths):
        fs = [cv2.imread(str(p)) for p in paths]
        ds = [float(np.mean(np.abs(x.astype(np.int16) - y.astype(np.int16))))
              for x, y in zip(fs, fs[1:])]
        return float(np.median(ds))

    j0, j1 = jitter(a), jitter(b)
    check(j1 <= max(0.5, j0 * 1.3), "B 开关超分不放大帧间抖动",
          "关 %.3f → 开 %.3f LSB/帧（阈值 1.3×）" % (j0, j1))


def main():
    os.environ.setdefault("DFL_SR", "mid")
    one_face()
    temporal()
    print("\n汇总：PASS %d · FAIL %d" % (len(PASS), len(FAIL)))
    return 0 if not FAIL else 1


if __name__ == "__main__":
    sys.exit(main())
