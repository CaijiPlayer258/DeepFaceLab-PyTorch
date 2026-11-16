# -*- coding: utf-8 -*-
r"""5 点退化闸门：对**已有** aligned/ 目录做只读体检，默认写一张 _bad_frames.txt。

它量的不是「点抖不抖」，而是「脸在芯片里有没有塌成一团」：
    5 点把粗裁视野拖偏 → 2D106 在错视野里出点 → 33 点拟合出错的尺度
    → 芯片里那张脸只占画布四成，68 个点却自洽地糊在肩膀/背景上。
这种帧的 resid 反而更小（已知坏 3.0~4.4 < 已知好 4.4~8.1），现有两道外点检查
一条都拦不住，图照样落盘。判据、实测与误杀分析见 docs/30-align-gate.md。

只读保证：**不重切、不改 JPEG、不移动、不删除**；唯一的写动作是
<目录>/_bad_frames.txt（--no-write 可关）。

跑法（仓库根，CPU 即可）：
    python app\tools\align_gate.py --workspace                 # 体检整个 workspace
    python app\tools\align_gate.py --dir workspace\data_dst\aligned
    python app\tools\align_gate.py --assert-labeled            # 素材 29..134 断言
    python app\tools\align_gate.py --workspace --strict-align   # 预演真拒（只报数）

退出码：0 = 量完了；2 = 断言没过 / 依赖缺失。
"""
from __future__ import annotations

import argparse
import io
import json
import os
import pickle
import struct
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
APP = REPO / "app"
for _p in (str(APP), str(APP / "engine")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
os.environ.setdefault("DFL_WORKSPACE", str(REPO / "workspace"))
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

import numpy as np                                             # noqa: E402
from server._vision import align as AL                          # noqa: E402

_SOF = (0xC0, 0xC1, 0xC2)
_SAFE = {
    "builtins": {"dict", "list", "tuple", "str", "int", "float", "bool", "bytes",
                 "set", "slice", "range"},
    "__builtin__": {"dict", "list", "tuple", "str", "int", "float", "bool", "bytes",
                    "set", "slice", "range"},
    "numpy": {"ndarray", "float32", "float64", "float16", "int32", "int64", "int16",
              "int8", "uint8", "uint16", "uint32", "uint64", "dtype", "generic", "scalar"},
    "numpy.core.multiarray": {"_reconstruct", "scalar"},
    "numpy._core.multiarray": {"_reconstruct", "scalar"},
    "numpy.core.numeric": {"ones", "zeros"},
    "numpy._core.numeric": {"ones", "zeros"},
    "collections": {"OrderedDict"},
    "_codecs": {"encode"},
}


class _Restricted(pickle.Unpickler):
    def find_class(self, module, name):
        if name in _SAFE.get(module, ()):
            return super().find_class(module, name)
        raise pickle.UnpicklingError("unsafe pickle type: %s.%s" % (module, name))


def _say(msg):
    """打印永不让体检挂掉：控制台是 GBK 时，个别字符编不出来就降级。"""
    try:
        sys.stdout.write(msg + "\n")
        sys.stdout.flush()
    except UnicodeEncodeError:
        enc = (getattr(sys.stdout, "encoding", "") or "utf-8")
        sys.stdout.write(msg.encode(enc, "replace").decode(enc, "replace") + "\n")
        sys.stdout.flush()


def read_meta(path):
    """mmap 走 JPEG 段：APP15 → 元数据 dict，SOF0/2 → (h, w)。不读扫描数据。"""
    import mmap
    meta, hw = None, None
    with open(path, "rb") as f:
        try:
            mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        except ValueError:                                    # 空文件
            return None, None
        try:
            n = len(mm)
            i = 2
            while i + 4 <= n:
                if mm[i] != 0xFF:
                    break
                m = mm[i + 1]
                if m == 0xDA:                                 # SOS：后面是压缩数据
                    break
                ln = struct.unpack(">H", mm[i + 2:i + 4])[0]
                body = i + 4
                if m == 0xEF and meta is None:
                    try:
                        meta = _Restricted(io.BytesIO(mm[body:body + ln - 2])).load()
                    except Exception:                          # noqa: BLE001
                        meta = None
                elif m in _SOF and hw is None and ln >= 7:
                    hw = struct.unpack(">HH", mm[body + 1:body + 5])
                i = body + ln - 2
        finally:
            mm.close()
    return meta, hw


def scan_dir(d: Path, write_list=True, quiet=False):
    """一个 aligned/ 目录：逐张算闸门，返回统计。只写 _bad_frames.txt。"""
    files = sorted(d.glob("*.jpg"))
    t0 = time.time()
    n = bad = no_lm = 0
    spans, iods, hits = [], [], []
    for f in files:
        n += 1
        dd, hw = read_meta(f)
        if dd is None or dd.get("landmarks") is None:
            no_lm += 1
            continue
        lm = np.asarray(dd["landmarks"], np.float64)
        side = hw[1] if hw else 512
        geo = AL.chip_geometry(lm, side)
        if geo is None:
            no_lm += 1
            continue
        spans.append(geo[0])
        iods.append(geo[1])
        g = AL.align_gate(lm, side)
        if g and g["bad"]:
            bad += 1
            hits.append({"name": f.name, "span": g["span"], "iod": g["iod"],
                         "reason": g["reason"]})
    if write_list and (hits or (d / "_bad_frames.txt").exists()):
        # 整份重写：体检跑第二遍不应该出现重复行（抽取那条路才是追加）。
        # 零命中的目录不新建文件；但**已有**清单要跟着改成「没有」（否则旧名单会一直骗人）。
        body = AL.bad_frame_header() + "".join(
            AL.bad_frame_line(h["name"], h) for h in hits)
        if not hits:
            body += "# （本目录没有检出退化帧）\n"
        (d / "_bad_frames.txt").write_text(body, encoding="utf-8")
    dt = time.time() - t0
    if not quiet:
        _say("%-46s 张=%-6d 退化=%-4d (%.2f%%) 无点=%-5d %.1fs"
              % (str(d).replace(str(REPO) + os.sep, ""), n, bad,
                 100.0 * bad / max(1, n - no_lm), no_lm, dt))
    return {"dir": str(d), "n": n, "bad": bad, "no_landmarks": no_lm,
            "pct": (100.0 * bad / max(1, n - no_lm)) if n - no_lm else 0.0,
            "span_p50": (float(np.median(spans)) if spans else None),
            "iod_p50": (float(np.median(iods)) if iods else None),
            "hits": hits}


def find_dirs(ws=None):
    """workspace 下所有 aligned/（含一层项目目录），外加 merge/*/aligned。"""
    ws = Path(ws) if ws else (REPO / "workspace")
    out = []
    for pat in ("*/aligned", "*/*/aligned"):
        out += [p for p in sorted(ws.glob(pat)) if p.is_dir()]
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", action="append", default=[], help="aligned/ 目录（可重复）")
    ap.add_argument("--workspace", action="store_true", help="扫 workspace 下所有 aligned/")
    ap.add_argument("--workspace-root", default=None,
                    help="--workspace 的根（默认本仓库的 workspace；体检另一份素材时指过去）")
    ap.add_argument("--strict-align", action="store_true",
                    help="预演 --strict-align：按「真拒」口径报告（本工具从不移动/删除）")
    ap.add_argument("--no-write", action="store_true", help="不写 _bad_frames.txt")
    ap.add_argument("--assert-labeled", action="store_true",
                    help="断言 4.1 素材 29..80 全拒、81..134 全放（就是已知集）")
    ap.add_argument("--json", default=None)
    a = ap.parse_args(argv)

    gs, gi = AL.gate_limits()
    gm = "strict" if a.strict_align else AL.gate_mode()
    _say("[gate] 判据 face_span<%.2f 且 iod_chip<%.2f · 模式=%s%s"
          % (gs, gi, gm,
             "" if a.strict_align else "（warn：只记录，不拒）"))
    _say("[gate] 环境：DFL_ALIGN_GATE=%r DFL_ALIGN_GATE_SPAN=%r DFL_ALIGN_GATE_IOD=%r"
          % (os.environ.get("DFL_ALIGN_GATE"), os.environ.get("DFL_ALIGN_GATE_SPAN"),
             os.environ.get("DFL_ALIGN_GATE_IOD")))

    dirs = [Path(p) for p in a.dir]
    if a.workspace:
        dirs += [p for p in find_dirs(a.workspace_root) if p not in dirs]
    if a.assert_labeled:
        d = REPO / "workspace" / "data_src" / "aligned"
        if d not in dirs:
            dirs.append(d)
    if not dirs:
        _say("退出码 2：没给 --dir / --workspace")
        return 2

    res = []
    for d in dirs:
        if not d.is_dir():
            _say("  跳过（不是目录）：%s" % d)
            continue
        res.append(scan_dir(d, write_list=not a.no_write))

    tot = sum(r["n"] for r in res)
    tot_bad = sum(r["bad"] for r in res)
    tot_nolm = sum(r["no_landmarks"] for r in res)
    _say("[gate] 合计 %d 张，退化 %d（%.2f%%），无 landmarks %d"
          % (tot, tot_bad, 100.0 * tot_bad / max(1, tot - tot_nolm), tot_nolm))
    if not a.no_write:
        _say("[gate] 清单已写到各目录的 _bad_frames.txt（不拒、不删、不改 JPEG）")

    rc = 0
    if a.assert_labeled:
        d = REPO / "workspace" / "data_src" / "aligned"
        bad_ids, good_ids = set(), set()
        for r in res:
            if Path(r["dir"]) != d:
                continue
            for h in r["hits"]:
                bad_ids.add(h["name"].split("_")[0])
        files = sorted(p.name for p in d.glob("*.jpg"))
        good_ids = {f.split("_")[0] for f in files} - bad_ids
        n_bad = sum(1 for f in files if 29 <= int(f.split("_")[0]) <= 80)
        n_good = sum(1 for f in files if 81 <= int(f.split("_")[0]) <= 134)
        r_bad = sum(1 for f in files
                    if 29 <= int(f.split("_")[0]) <= 80 and f.split("_")[0] in bad_ids)
        r_good = sum(1 for f in files
                     if 81 <= int(f.split("_")[0]) <= 134 and f.split("_")[0] in bad_ids)
        ok = (r_bad == n_bad) and (r_good == 0) and n_good == 54
        _say("[assert] 素材 29..80 现存 %d 帧（原始 52 帧里 15 帧从未落盘）→ 拒 %d；"
              "81..134 %d 帧 → 误拒 %d  ⇒ %s"
              % (n_bad, r_bad, n_good, r_good, "PASS" if ok else "FAIL"))
        rc = 0 if ok else 2

    if a.json:
        Path(a.json).write_text(json.dumps({"gate": [gs, gi], "mode": gm,
                                            "results": res}, ensure_ascii=False, indent=1),
                                encoding="utf-8")
        _say("[gate] -> %s" % a.json)
    return rc


if __name__ == "__main__":
    sys.exit(main())