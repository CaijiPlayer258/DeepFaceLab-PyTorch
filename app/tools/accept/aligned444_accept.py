# -*- coding: utf-8 -*-
r"""对齐脸 JPEG 4:4:4 的验收：cv2 5.0 下 subsampling=0 一直被静默降级成 4:2:0。

背景（改前实测，本脚本 §a 每次都会重新算一遍）：
  cv2 5.0 起 IMWRITE_JPEG_SAMPLING_FACTOR 只认 ImwriteJPEGSamplingFactorParams 枚举，
  老写法 [ID, 0] 被拒并**静默**落成 4:2:0 —— imencode 照样返回 True，字节与「不传
  这个参数」一模一样。write_dfljpg 里那句 except AttributeError 兜不住它。

验的东西（全部在 %TEMP% 里跑）：
  a) 探测策略：cv2 版本 + 常量在不在 + 走哪条路（enum444 / int0 / none）；
     老写法复现：subsampling=0 与不传参数**字节相同**、文件头是 (2,2)；
     新写法：文件头三个分量都 (1,1)。
  b) 单张对比（同一张真实对齐脸）：旧 4:2:0 vs 新 4:4:4 的
     文件大小 / 体积增幅% / 整体 PSNR / 色度 PSNR（写盘后解码回来对比原图）。
  c) 30 帧小样真提取（真后端 + 真视频，切成正好 30 帧）：
     全部 aligned 的文件头都是 4:4:4；读取端（引擎 DFLJPG.load / get_landmarks）无异常；
     后端日志里「探测」只出现一次（不再逐张刷），cv2 的 fallback warning 出现 0 次。
  d) 老数据（4:2:0）与新数据（4:4:4）并存：同一个目录里两种都读得回来，
     元数据、landmarks、shape 都对 —— 读取端不区分采样因子。

跑法（仓库根）：
    runtime\server\Scripts\python.exe app\tools\aligned444_accept.py
    # 只跑某一组：--only a,b

退出码 0 = 全过。
"""
from __future__ import annotations

import argparse
import io
import json
import math
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
APP = REPO / "app"
PY = REPO / "runtime" / "server" / "Scripts" / "python.exe"
ENGINE_PY = REPO / "runtime" / "engine" / "python.exe"
TEMP = Path(os.environ.get("TEMP") or os.environ.get("TMP") or ".")
BOX = TEMP / "dfl_aligned444_accept"
WS = BOX / "ws"
MEDIA = BOX / "media"
OUT = BOX / "img"
LOGDIR = BOX / "log"
SERVER_LOG = LOGDIR / "server.log"
SRC_FACES = REPO / "workspace" / "data_src" / "aligned"
ALIGNED = WS / "data_src" / "aligned"
NO_WINDOW = 0x08000000 if os.name == "nt" else 0

if str(APP) not in sys.path:
    sys.path.insert(0, str(APP))

import cv2        # noqa: E402  —— 被测模块用的就是它
import numpy as np  # noqa: E402
from server import _align  # noqa: E402

BASE = ""
RESULTS = []
OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))
PSNR_INF = float("inf")


def check(ok, name, detail=""):
    ok = bool(ok)
    RESULTS.append((ok, name, str(detail)))
    print(("[PASS] " if ok else "[FAIL] ") + name + ("  · " + str(detail) if detail else ""),
          flush=True)
    return ok


def say(msg):
    print("       " + str(msg), flush=True)


# ============================================================
# HTTP
# ============================================================
def http(method, path, body=None, timeout=120.0):
    data = None
    headers = {}
    if body is not None:
        data = json.dumps(body, ensure_ascii=False).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(BASE + path, data=data, headers=headers, method=method)
    try:
        with OPENER.open(req, timeout=timeout) as r:
            return r.status, json.loads(r.read().decode("utf-8") or "{}")
    except urllib.error.HTTPError as e:
        blob = e.read()
        try:
            return e.code, json.loads(blob.decode("utf-8") or "{}")
        except Exception:                                   # noqa: BLE001
            return e.code, {"error": blob[:300].decode("utf-8", "replace")}


def get(path, **kw):
    return http("GET", path, **kw)


def post(path, body, **kw):
    return http("POST", path, body, **kw)


# ============================================================
# 后端进程
# ============================================================
def _kill_tree(pid):
    try:
        subprocess.run(["taskkill", "/F", "/T", "/PID", str(pid)],
                       capture_output=True, creationflags=NO_WINDOW)
    except Exception:                                       # noqa: BLE001
        pass


def start_server(tag=""):
    global BASE
    env = dict(os.environ)
    env["DFL_WORKSPACE"] = str(WS)
    env["PYTHONIOENCODING"] = "utf-8"
    env["NO_PROXY"] = env["no_proxy"] = "127.0.0.1,localhost"
    env.pop("DFL_PORT", None)
    env.pop("DFL_STAGE_KEEP", None)
    logf = open(str(SERVER_LOG), "ab")
    logf.write(("\n===== start %s =====\n" % tag).encode("utf-8"))
    logf.flush()
    proc = subprocess.Popen([str(PY), "serve.py"], cwd=str(APP), env=env,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    port = {"v": None}

    def reader():
        for line in iter(proc.stdout.readline, b""):
            try:
                logf.write(line)
                logf.flush()
            except Exception:                               # noqa: BLE001
                pass
            text = line.decode("utf-8", "replace")
            if port["v"] is None and text.startswith("PORT="):
                try:
                    port["v"] = int(text.strip().split("=", 1)[1])
                except Exception:                           # noqa: BLE001
                    pass

    threading.Thread(target=reader, daemon=True).start()
    t0 = time.time()
    while port["v"] is None and time.time() - t0 < 180:
        if proc.poll() is not None:
            raise SystemExit("后端启动失败，见 %s" % SERVER_LOG)
        time.sleep(0.2)
    if port["v"] is None:
        raise SystemExit("等不到 PORT= 行，见 %s" % SERVER_LOG)
    BASE = "http://127.0.0.1:%d" % port["v"]
    say("后端%s：%s（pid=%s）" % (tag, BASE, proc.pid))
    while time.time() - t0 < 240:
        try:
            st, _d = get("/api/extract/status", timeout=20)
            if st == 200:
                return proc
        except Exception:                                   # noqa: BLE001
            pass
        time.sleep(0.4)
    raise SystemExit("后端起来了但 /api/extract/status 不应答，见 %s" % SERVER_LOG)


def stop_server(proc):
    if proc is None:
        return
    _kill_tree(proc.pid)
    try:
        proc.kill()
    except Exception:                                       # noqa: BLE001
        pass
    time.sleep(1.2)


# ============================================================
# 引擎侧读取（DFLJPG.load + get_landmarks）—— 必须用引擎那套 python
#   server 的 venv 里没有 tqdm/torch，引擎的 core.* 一 import 就炸
# ============================================================
ENGINE_HELPER = r'''
import sys, io, json, contextlib
from pathlib import Path
REPO = Path(sys.argv[1])
sys.path.insert(0, str(REPO / "app" / "engine"))
import numpy as np
from DFLIMG import DFLJPG

paths = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
out = []
for p in paths:
    rec = {"path": p, "load": False, "has_data": False, "shape": None,
           "landmarks": None, "err": ""}
    sink = io.StringIO()
    try:
        with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
            d = DFLJPG.load(p)
        rec["load"] = d is not None
        if d is not None:
            rec["has_data"] = bool(d.has_data())
            rec["shape"] = list(d.shape) if d.shape else None
            if rec["has_data"]:
                rec["landmarks"] = list(np.asarray(d.get_landmarks()).shape)
    except Exception as e:
        rec["err"] = "%s: %s" % (type(e).__name__, str(e)[:160])
    out.append(rec)
print("@@RESULT@@" + json.dumps(out, ensure_ascii=False))
'''


def engine_read(paths):
    """→ [{path, load, has_data, shape, landmarks, err}]；引擎 python 跑子进程。"""
    helper = BOX / "_engine_check.py"
    helper.write_text(ENGINE_HELPER, encoding="utf-8")
    lst = BOX / "_engine_paths.json"
    lst.write_text(json.dumps([str(p) for p in paths], ensure_ascii=False), encoding="utf-8")
    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"
    r = subprocess.run([str(ENGINE_PY), str(helper), str(REPO), str(lst)],
                       capture_output=True, text=True, encoding="utf-8",
                       errors="replace", env=env, creationflags=NO_WINDOW)
    for line in (r.stdout or "").splitlines():
        if line.startswith("@@RESULT@@"):
            try:
                return json.loads(line[len("@@RESULT@@"):])
            except Exception:                               # noqa: BLE001
                continue
    raise SystemExit("引擎侧读取失败：%s\n%s" % ((r.stdout or "")[-800:], (r.stderr or "")[-800:]))


# ============================================================
# 指标
# ============================================================
def psnr(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    mse = float(np.mean((a - b) ** 2))
    if mse <= 1e-12:
        return PSNR_INF
    return 10.0 * math.log10(255.0 ** 2 / mse)


def chroma_psnr(a, b):
    """只在 Cb/Cr 上算 PSNR —— 4:2:0 抽稀的地方。"""
    ya = cv2.cvtColor(a, cv2.COLOR_BGR2YCrCb)[:, :, 1:]
    yb = cv2.cvtColor(b, cv2.COLOR_BGR2YCrCb)[:, :, 1:]
    return psnr(ya, yb)


def fmt(x):
    return "inf" if x == PSNR_INF else "%.2f" % x


def sf_444(blob):
    return _align.jpeg_sof_sampling(blob) == [(1, 1), (1, 1), (1, 1)]


def sf_420(blob):
    return _align.jpeg_sof_sampling(blob) == [(2, 2), (1, 1), (1, 1)]


def tile(seed=7):
    """探测/复现用的样板图：色度越脏越好（采样因子的差别才看得出来）。"""
    rng = np.random.default_rng(seed)
    img = np.zeros((256, 256, 3), np.uint8)
    yy, xx = np.mgrid[0:256, 0:256]
    img[:, :, 0] = np.clip(yy * 0.4 + 60, 0, 255)
    img[:, :, 1] = np.clip(xx * 0.3 + 70, 0, 255)
    img[:, :, 2] = np.clip((xx + yy) * 0.2 + 80, 0, 255)
    img[60:140, 70:170] = (40, 60, 200)
    img[::7, ::5] = (255, 30, 30)
    return np.clip(img.astype(np.int16) + rng.integers(-12, 12, img.shape),
                   0, 255).astype(np.uint8)


# ============================================================
# a) 探测策略 + 老写法复现
# ============================================================
def case_a():
    print("\n== a) cv2 常量与探测策略 ==", flush=True)
    probe = _align.jpeg444_probe()
    say("cv2 = %s" % probe["cv2"])
    for name in ("IMWRITE_JPEG_QUALITY", "IMWRITE_JPEG_SAMPLING_FACTOR",
                 "IMWRITE_JPEG_SAMPLING_FACTOR_444", "IMWRITE_JPEG_SAMPLING_FACTOR_422",
                 "IMWRITE_JPEG_SAMPLING_FACTOR_420", "IMWRITE_JPEG_SAMPLING_FACTOR_440"):
        v = getattr(cv2, name, None)
        say("  %-34s = %s" % (name, "（不存在）" if v is None else v))
    say("探测结论：path=%s sf=%s（%s）" % (probe["path"], probe["sf"], probe["detail"]))
    check(probe["path"] in ("enum444", "int0") and probe["ok"],
          "a 探测出可写 4:4:4 的策略", "path=%s · %s" % (probe["path"], probe["detail"]))

    img = tile()
    q100 = [int(cv2.IMWRITE_JPEG_QUALITY), 100]
    nop = cv2.imencode(".jpg", img, q100)[1].tobytes()
    fid = getattr(cv2, "IMWRITE_JPEG_SAMPLING_FACTOR", None)
    if fid is not None:
        say("（下面这次 encode 会打 cv2 自己的 fallback warning —— 这就是被修掉的那个静默降级）")
        old = cv2.imencode(".jpg", img, q100 + [int(fid), 0])[1].tobytes()   # 老写法
        says = _align.jpeg_sof_sampling(old)
        check(len(old) == len(nop) and says == _align.jpeg_sof_sampling(nop),
              "a 老写法 [ID,0] 的字节与「不传参数」完全相同 = 静默 4:2:0",
              "old=%d B %s / noparam=%d B %s" % (len(old), says, len(nop),
                                                 _align.jpeg_sof_sampling(nop)))
        check(says is not None and says[0] == (2, 2),
              "a 老写法落盘的文件头确实是 4:2:0（Y=(2,2)）", str(says))
    else:
        say("本机 cv2 没有 SAMPLING_FACTOR 常量，老写法本来就用不上，跳过复现")

    new = cv2.imencode(".jpg", img, _align.jpeg_encode_params(100, 0))[1].tobytes()
    check(sf_444(new), "a 新写法（jpeg_encode_params(100,0)）落盘是 4:4:4",
          "%d B · %s" % (len(new), _align.jpeg_sof_sampling(new)))
    noparams = cv2.imencode(".jpg", img, q100)[1].tobytes()
    check(sf_420(noparams) and len(noparams) < len(new),
          "a 对照：不带参数仍是 4:2:0 且明显更小",
          "%d B vs %d B（+%.1f%%）" % (len(noparams), len(new),
                                       100.0 * (len(new) - len(noparams)) / len(noparams)))
    case_a_legacy()


LEGACY_PROBE = r'''
import sys, json
sys.path.insert(0, sys.argv[1])
import cv2, numpy as np
# 模拟「没有 4:4:4 枚举常量」的老 cv2：只剩 SAMPLING_FACTOR 那个 id，且它只认裸整数
delattr(cv2, "IMWRITE_JPEG_SAMPLING_FACTOR_444")
from server import _align
probe = _align.jpeg444_probe()
img = np.zeros((32, 32, 3), np.uint8)
img[:, :16] = (255, 0, 0)
img[:, 16:] = (0, 0, 255)
soft = []
for i in range(3):
    p = sys.argv[2] + "/legacy_%d.jpg" % i
    _align.write_dfljpg(p, img, {"landmarks_source": "none"})
    soft.append(_align.jpeg_sof_sampling(p and open(p, "rb").read()))
print("@@LEGACY@@" + json.dumps({"path": probe["path"], "ok": probe["ok"], "sf": soft},
                               ensure_ascii=False))
'''


def case_a_legacy():
    """模拟「本机写不出 4:4:4」：删掉枚举常量 → 必须走 int0 候选 → 失败 → WARN 一次。

    cv2 5.0 里 int0 会被拒，所以这里肯定落到 none；这条正是「老版本回退 + 不许静默」的
    验证：一次 WARN 说清后果，之后每张图都安静地写 4:2:0（不再刷屏）。
    """
    d = BOX / "legacy_probe"
    shutil.rmtree(str(d), ignore_errors=True)
    d.mkdir(parents=True, exist_ok=True)
    helper = BOX / "_legacy_probe.py"
    helper.write_text(LEGACY_PROBE, encoding="utf-8")
    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"
    r = subprocess.run([str(PY), str(helper), str(APP), str(d)],
                       capture_output=True, text=True, encoding="utf-8",
                       errors="replace", env=env, creationflags=NO_WINDOW)
    payload = None
    for line in (r.stdout or "").splitlines():
        if line.startswith("@@LEGACY@@"):
            payload = json.loads(line[len("@@LEGACY@@"):])
    if payload is None:
        return check(False, "a2 老 cv2 回退探测能跑起来",
                     (r.stdout or "")[-300:] + (r.stderr or "")[-300:])
    say("模拟「没有 _444 常量」：path=%s ok=%s 三张文件头=%s"
        % (payload["path"], payload["ok"], payload["sf"]))
    check(payload["path"] == "none" and payload["ok"] is False,
          "a2 枚举常量缺失 → 明确判定「写不出 4:4:4」（不谎报成功）",
          "path=%s ok=%s" % (payload["path"], payload["ok"]))
    sfs = [[tuple(c) for c in s] for s in payload["sf"]]
    check(all(s == [(2, 2), (1, 1), (1, 1)] for s in sfs),
          "a2 此时确实落成 4:2:0（库默认），没有假装是 4:4:4", str(sfs))
    n_warn = (r.stdout or "").count("写不出 4:4:4")
    n_cv2 = (r.stderr or "").count("fallbacked to IMWRITE_JPEG_SAMPLING_FACTOR_420")
    check(n_warn == 1, "a2 告警只打一条（3 张图不刷屏）", "%d 条" % n_warn)
    check(n_cv2 == 1, "a2 cv2 自己的 fallback warning 只在探测那一次出现",
          "%d 次" % n_cv2)


# ============================================================
# b) 单张对比：size / PSNR
# ============================================================
FRAME_CACHE = REPO / "workspace" / "9月29日 (5)_cache" / "frames"


def fresh_crop(p):
    """老对齐脸 → 用无损 PNG 帧 + 它自己的 image_to_face_mat 重新 warp 一张「新鲜」对齐脸。

    为什么要多这一步：现有 aligned 本身就是 4:2:0 解码结果，把**它**当基准去量两个
    编码器，4:2:0 是这套滤波链的不动点（下采样再上采样几乎还原），会得出「444 更差」
    的假结论。提取器真正交给 write_dfljpg 的是从源帧 warp 出来的新鲜裁剪 —— 那才是
    要比的输入。拿不到 PNG 帧就返回 None（表 2 会退化成合成图）。
    """
    meta = _align.read_dfljpg_meta(p) or {}
    mat = np.asarray(meta.get("image_to_face_mat") or [], np.float64)
    jpg = cv2.imdecode(np.frombuffer(p.read_bytes(), np.uint8), cv2.IMREAD_COLOR)
    if mat.shape != (2, 3) or jpg is None:
        return None, jpg
    head = p.stem.split("_")[0]
    png = FRAME_CACHE / ("%06d.png" % int(head)) if head.isdigit() else None
    if png is None or not png.is_file():
        return None, jpg
    frame = cv2.imdecode(np.frombuffer(png.read_bytes(), np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        return None, jpg
    h, w = jpg.shape[:2]
    return cv2.warpAffine(frame, mat, (w, h), flags=cv2.INTER_CUBIC,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0)), jpg


def _roundtrip(tag, src, meta):
    """src → 旧(4:2:0) / 新(4:4:4) 各写一次 DFLJPG，读回来对比。返回一行数字。"""
    po = OUT / ("%s_old.jpg" % tag)
    pn = OUT / ("%s_new.jpg" % tag)
    _align.write_dfljpg(po, src, meta, quality=95, subsampling=None)   # 旧行为
    _align.write_dfljpg(pn, src, meta, quality=95)                     # 新默认 4:4:4
    bo, bn = po.read_bytes(), pn.read_bytes()
    do = cv2.imdecode(np.frombuffer(bo, np.uint8), cv2.IMREAD_COLOR)
    dn = cv2.imdecode(np.frombuffer(bn, np.uint8), cv2.IMREAD_COLOR)
    # 第二代：tonemap_aligned_faces 这类工具会重写一遍文件
    _align.write_dfljpg(po, do, meta, quality=95, subsampling=None)
    _align.write_dfljpg(pn, dn, meta, quality=95)
    do2 = cv2.imdecode(np.frombuffer(po.read_bytes(), np.uint8), cv2.IMREAD_COLOR)
    dn2 = cv2.imdecode(np.frombuffer(pn.read_bytes(), np.uint8), cv2.IMREAD_COLOR)
    return {
        "tag": tag, "src": tag, "old": len(bo), "new": len(bn),
        "grow": 100.0 * (len(bn) - len(bo)) / max(1, len(bo)),
        "psnr_old": psnr(src, do), "psnr_new": psnr(src, dn),
        "cpsnr_old": chroma_psnr(src, do), "cpsnr_new": chroma_psnr(src, dn),
        "c2_old": chroma_psnr(src, do2), "c2_new": chroma_psnr(src, dn2),
        "sf_old": _align.jpeg_sof_sampling(bo), "sf_new": _align.jpeg_sof_sampling(bn),
        "shape": src.shape, "wire": (set(_align.read_dfljpg_meta(po) or {}) ==
                                     set(_align.read_dfljpg_meta(pn) or {})),
    }


def _print_table(rows, title):
    print("       %s" % title, flush=True)
    for r in rows:
        print("       %-18s %7d %7d %7.1f%%  %6s/%6s  %7s/%7s  %7s/%7s"
              % (r["tag"], r["old"], r["new"], r["grow"],
                 fmt(r["psnr_old"]), fmt(r["psnr_new"]),
                 fmt(r["cpsnr_old"]), fmt(r["cpsnr_new"]),
                 fmt(r["c2_old"]), fmt(r["c2_new"])), flush=True)


def case_b():
    print("\n== b) 同一张对齐脸：旧 4:2:0 vs 新 4:4:4（文件大小 / 体积增幅 / PSNR）==",
          flush=True)
    files = sorted(SRC_FACES.glob("*.jpg"))[:5]
    if not files:
        return check(False, "b 找不到真实对齐脸样本", str(SRC_FACES))
    OUT.mkdir(parents=True, exist_ok=True)
    say("列：文件 旧B 新B 增幅 · PSNR旧/新（对输入） · 色度PSNR旧/新 · 第二代色度PSNR旧/新")

    # ---- 表 1：写现有 aligned 数据（它自己就是 4:2:0 解码结果）----
    t1 = []
    for p in files:
        jpg = cv2.imdecode(np.frombuffer(p.read_bytes(), np.uint8), cv2.IMREAD_COLOR)
        meta = {"face_type": "whole_face",
                "source_rect": [0, 0, int(jpg.shape[1]), int(jpg.shape[0])],
                "image_to_face_mat": np.eye(2, 3).tolist(),
                "landmarks_source": "none", "origin": p.name}
        t1.append(_roundtrip("老数据-" + p.stem, jpg, meta))
    _print_table(t1, "表1｜输入 = 现有 aligned（本身已是 4:2:0）")
    check(all(r["sf_old"] == [(2, 2), (1, 1), (1, 1)] for r in t1),
          "b 表1 旧写法的文件头全是 4:2:0", str(t1[0]["sf_old"]))
    check(all(r["sf_new"] == [(1, 1), (1, 1), (1, 1)] for r in t1),
          "b 表1 新写法的文件头全是 4:4:4", str(t1[0]["sf_new"]))
    check(all(r["new"] > r["old"] for r in t1), "b 表1 每张 4:4:4 都更大",
          "平均 +%.1f%%" % (sum(r["grow"] for r in t1) / len(t1)))
    check(all(r["wire"] for r in t1), "b 表1 APP15 元数据两边一致（只换编码）")
    check(all(r["psnr_new"] > 50.0 for r in t1),
          "b 表1 4:4:4 仍是视觉无损（PSNR > 50 dB）",
          "%s dB" % "/".join(fmt(r["psnr_new"]) for r in t1))
    say("表1 的 PSNR 不能用来判优劣：输入是 4:2:0 解码出来的，"
        "再走一遍 4:2:0 是同一套滤波的不动点（下采样→上采样几乎还原），"
        "所以旧写法在这类输入上 PSNR 反而更高。真正的对比看表2。")

    # ---- 表 2：写「新鲜」对齐脸（提取器实际交给写盘函数的输入）----
    t2, fresh_n = [], 0
    for p in files:
        fresh, jpg = fresh_crop(p)
        if fresh is None:
            continue
        fresh_n += 1
        meta = {"face_type": "whole_face",
                "source_rect": [0, 0, int(fresh.shape[1]), int(fresh.shape[0])],
                "image_to_face_mat": np.eye(2, 3).tolist(),
                "landmarks_source": "none", "origin": p.name}
        t2.append(_roundtrip("新鲜裁剪-" + p.stem, fresh, meta))
    if t2:
        _print_table(t2, "表2｜输入 = 从无损 PNG 帧按 image_to_face_mat 重新 warp（%d 张）"
                     % fresh_n)
        d_all = sum(r["psnr_new"] - r["psnr_old"] for r in t2) / len(t2)
        d_ch = sum(r["cpsnr_new"] - r["cpsnr_old"] for r in t2) / len(t2)
        d_c2 = sum(r["c2_new"] - r["c2_old"] for r in t2) / len(t2)
        say("表2 平均：整体 PSNR %+.2f dB · 色度 PSNR %+.2f dB · 第二代色度 %+.2f dB"
            " · 体积 +%.1f%%" % (d_all, d_ch, d_c2, sum(r["grow"] for r in t2) / len(t2)))
        check(d_all > 0.8, "b 表2 4:4:4 整体 PSNR 更好（>0.8 dB）", "%+.2f dB" % d_all)
        check(d_ch > 1.5, "b 表2 4:4:4 色度 PSNR 明显更好（>1.5 dB —— 抽稀补回来）",
              "%+.2f dB" % d_ch)
        check(d_c2 > 1.0, "b 表2 重写一遍（第二代）色度仍更好（>1 dB）", "%+.2f dB" % d_c2)
    else:
        # 没有 PNG 帧缓存时用合成的高色度细节图（机制一样，只是不是真脸）
        say("没有可用的无损 PNG 帧，表2 退化用合成图（高色度细节）")
        src = tile()
        meta = {"face_type": "whole_face", "source_rect": [0, 0, 256, 256],
                "image_to_face_mat": np.eye(2, 3).tolist(), "landmarks_source": "none"}
        t2 = [_roundtrip("合成色度块", src, meta)]
        _print_table(t2, "表2｜输入 = 合成高色度细节图")
        check(t2[0]["cpsnr_new"] > t2[0]["cpsnr_old"] + 1.5,
              "b 表2 色度 PSNR 更好（>1.5 dB）",
              "%+.2f dB" % (t2[0]["cpsnr_new"] - t2[0]["cpsnr_old"]))
    return t1, t2


# ============================================================
# c) 30 帧小样真提取
# ============================================================
def pick_video():
    p = REPO / "workspace" / "9月29日 (5).mp4"
    if p.is_file():
        return p
    for c in sorted((REPO / "workspace").glob("*.mp4")):
        if not c.stem.endswith("_merged"):
            return c
    return None


def build_clip(exe, clip):
    """做一段正好 30 帧、**每帧都有脸**的小样。

    别赌「视频片头有人脸」：实测源视频前 1 秒只检出 1 张脸（片头几乎是空的），
    那样验不了「30 帧全是 4:4:4」。这里用已存在的无损 PNG 帧 + 现有 aligned 的名字
    反查出「连续 30 帧且每帧有脸」的窗口，数据驱动，不靠猜时间点。
    """
    have_png = {int(p.stem) for p in FRAME_CACHE.glob("*.png") if p.stem.isdigit()} \
        if FRAME_CACHE.is_dir() else set()
    have_face = {int(p.stem.split("_")[0]) for p in SRC_FACES.glob("*.jpg")
                 if p.stem.split("_")[0].isdigit()}
    for n in sorted(have_png & have_face):
        if all((n + k) in have_png and (n + k) in have_face for k in range(30)):
            subprocess.run([exe, "-v", "error", "-y", "-framerate", "30",
                            "-start_number", str(n), "-i", str(FRAME_CACHE / "%06d.png"),
                            "-frames:v", "30", "-an", "-c:v", "libx264",
                            "-crf", "18", "-pix_fmt", "yuv420p", str(clip)],
                           check=True, creationflags=NO_WINDOW)
            return "无损 PNG 帧 %06d..%06d（每帧都有脸）" % (n, n + 29)
    src = pick_video()
    if src is None:
        return None
    subprocess.run([exe, "-v", "error", "-y", "-ss", "5", "-i", str(src),
                    "-frames:v", "30", "-an", "-c:v", "libx264", "-crf", "18",
                    "-pix_fmt", "yuv420p", str(clip)],
                   check=True, creationflags=NO_WINDOW)
    return "源视频第 5 秒起的 30 帧（%s）" % src.name


def case_c():
    print("\n== c) 30 帧小样真提取（真后端 + 真 30 帧）==", flush=True)
    import imageio_ffmpeg
    MEDIA.mkdir(parents=True, exist_ok=True)
    clip = MEDIA / "clip30.mp4"
    what = build_clip(imageio_ffmpeg.get_ffmpeg_exe(), clip)
    if what is None:
        return check(False, "c 造不出 30 帧小样", "既没有 PNG 帧也没有源视频")
    n, secs = imageio_ffmpeg.count_frames_and_secs(str(clip))
    say("小样：%s · %s" % (clip.name, what))
    check(n == 30, "c 小样视频正好 30 帧", "%d 帧 / %.3f s · %.2f fps"
          % (n, secs, (n / secs) if secs else 0))

    shutil.rmtree(ALIGNED, ignore_errors=True)
    # 日志是 append 的：先把上次的日志清掉，否则「探测只打一条」会数到上一轮那条
    try:
        SERVER_LOG.unlink()
    except OSError:
        pass
    proc = start_server("444")
    try:
        st, d = post("/api/extract/start", {"src": str(clip), "dst": str(ALIGNED),
                                            "align": True, "quality": "high", "skip": 1})
        check(st == 200, "c POST /api/extract/start 200", "HTTP %s · %s"
              % (st, d.get("error") or d.get("dst")))
        t0 = time.time()
        final = {}
        while time.time() - t0 < 900:
            _s, s = get("/api/extract/status", timeout=30)
            if not s.get("running"):
                final = s
                break
            time.sleep(0.5)
        check(bool(final) and final.get("ok") is True,
              "c 提取跑完（ok=true）", "err=%s · %s" % (final.get("error"),
                                                        final.get("msg")))
    finally:
        stop_server(proc)

    aligned = sorted(ALIGNED.glob("*.jpg"))
    check(len(aligned) == 30, "c aligned 张数 = 帧数 = 30", "%d 张" % len(aligned))
    bad = [p.name for p in aligned if not sf_444(p.read_bytes())]
    hist = {}
    for p in aligned:
        k = str(_align.jpeg_sof_sampling(p.read_bytes()))
        hist[k] = hist.get(k, 0) + 1
    check(not bad, "c 全部 aligned 文件头 = 4:4:4", "分布 %s%s"
          % (hist, "" if not bad else " · 不合规 %s" % bad[:3]))
    check(all(re.fullmatch(r"\d{6}_\d+\.jpg", p.name) for p in aligned),
          "c 命名仍是 %06d_%d.jpg", aligned[0].name if aligned else "无")

    # ---- 外部解码器的独立意见：不信自己那个 parser ----
    from PIL import Image as PILImage

    def pil_sampling(p):
        im = PILImage.open(str(p))
        im.load()
        return [(int(t[1]), int(t[2])) for t in getattr(im, "layer", [])]

    sample = aligned[:3]
    pil = [(p.name, pil_sampling(p)) for p in sample]
    for n, hv in pil:
        say("PIL %s → sampling factor %s" % (n, hv))
    check(pil and all(hv == [(1, 1), (1, 1), (1, 1)] for _n, hv in pil),
          "c PIL 独立读出的 sampling factor 全是 (1,1)（=4:4:4 / subsampling=0）",
          str(pil[0][1]) if pil else "无")

    exe = imageio_ffmpeg.get_ffmpeg_exe()

    def ff_pixfmt(p):
        r = subprocess.run([exe, "-hide_banner", "-i", str(p), "-f", "null", "-"],
                           capture_output=True, text=True, encoding="utf-8",
                           errors="replace", creationflags=NO_WINDOW)
        for line in (r.stderr or "").splitlines():
            if "Video: mjpeg" in line:
                return line.strip()
        return ""

    fmts = [ff_pixfmt(p) for p in sample]
    for f in fmts:
        say("ffmpeg %s" % f)
    check(fmts and all("yuvj444p" in f for f in fmts),
          "c ffmpeg 解码器认的是 yuvj444p（4:4:4）", (fmts[0][:90] if fmts else ""))
    old_p = sorted(SRC_FACES.glob("*.jpg"))
    if old_p:
        oldf = ff_pixfmt(old_p[0])
        say("ffmpeg（老数据）%s" % oldf)
        check("yuvj420p" in oldf, "c 对照：老 aligned 在 ffmpeg 眼里是 yuvj420p",
              oldf[:90])
    sizes = [p.stat().st_size for p in aligned]
    say("体积：合计 %.2f MB · 单张平均 %.1f KB"
        % (sum(sizes) / 1048576.0, sum(sizes) / max(1, len(sizes)) / 1024.0))

    # 读取端 1：引擎 DFLJPG.load + get_landmarks
    recs = engine_read(aligned)
    check(all(r["load"] and r["has_data"] for r in recs),
          "c 引擎 DFLJPG.load / has_data 全部成功", "%d/%d" % (
              sum(1 for r in recs if r["load"] and r["has_data"]), len(recs)))
    check(all(r["landmarks"] == [68, 2] for r in recs),
          "c 引擎 get_landmarks 全是 (68, 2)", "%d/%d" % (
              sum(1 for r in recs if r["landmarks"] == [68, 2]), len(recs)))
    shapes = {tuple(r["shape"] or ()) for r in recs}
    # 对齐边长 = 检测框长边 × 2（动态分辨率），所以尺寸本来就不统一 —— 只要求全是正方形
    check(shapes and all(len(s) == 3 and s[0] == s[1] and s[2] == 3 for s in shapes),
          "c 引擎读到的 shape 全部是正方形 RGB 对齐图",
          "%d 种尺寸 %d..%d" % (len(shapes), min(s[0] for s in shapes),
                                max(s[0] for s in shapes)))

    # 读取端 2：服务端自己的 read_dfljpg_meta
    metas = [_align.read_dfljpg_meta(p) for p in aligned]
    check(all(isinstance(m, dict) and m for m in metas),
          "c 服务端 read_dfljpg_meta 全部解出 dict",
          "%d/%d" % (sum(1 for m in metas if m), len(metas)))

    # 「探测一次」的证据：后端日志里那条 INFO 只出现一次，且没有任何 fallback warning
    log = SERVER_LOG.read_text(encoding="utf-8", errors="replace") if SERVER_LOG.is_file() else ""
    n_info = log.count("[align] JPEG 4:4:4 探测")
    n_fb = log.count("fallbacked to IMWRITE_JPEG_SAMPLING_FACTOR_420")
    n_warn = log.count("写不出 4:4:4")
    check(n_info == 1, "c 探测 INFO 只打一条（30 张脸不再逐张刷）", "%d 条" % n_info)
    check(n_fb == 0, "c 后端日志里 cv2 的静默降级 warning = 0 次", "%d 次" % n_fb)
    check(n_warn == 0, "c 没有「写不出 4:4:4」的告警（本机能写）", "%d 次" % n_warn)
    return aligned


# ============================================================
# d) 老数据（4:2:0）与新数据（4:4:4）并存
# ============================================================
def case_d(aligned):
    print("\n== d) 老 4:2:0 与新 4:4:4 并存时的读取端 ==", flush=True)
    old = sorted(SRC_FACES.glob("*.jpg"))[:3]
    if not aligned or not old:
        return check(False, "d 缺样本", "aligned=%d old=%d" % (len(aligned or []), len(old)))
    names = {p.name for p in aligned}
    copied = []
    for i, p in enumerate(old):
        dst = ALIGNED / ("9%05d_0.jpg" % i)      # 故意换个不会被覆盖的名字
        if dst.name in names:
            continue
        shutil.copy2(str(p), str(dst))
        copied.append(dst)
    mixed = sorted(ALIGNED.glob("*.jpg"))
    hist = {}
    for p in mixed:
        k = str(_align.jpeg_sof_sampling(p.read_bytes()))
        hist[k] = hist.get(k, 0) + 1
    say("混合目录：%d 张（新 %d + 老 %d）· 采样因子分布 %s"
        % (len(mixed), len(aligned), len(copied), hist))
    check(len(hist) == 2, "d 目录里确实两种采样因子并存", str(sorted(hist)))
    recs = engine_read(mixed)
    check(all(r["load"] and r["has_data"] for r in recs),
          "d 混合目录 引擎 DFLJPG.load 全部成功（4:2:0 与 4:4:4 都认）",
          "%d/%d" % (sum(1 for r in recs if r["load"] and r["has_data"]), len(recs)))
    check(all(r["landmarks"] == [68, 2] for r in recs),
          "d 混合目录 get_landmarks 全部 (68, 2)", "%d/%d" % (
              sum(1 for r in recs if r["landmarks"] == [68, 2]), len(recs)))
    metas = [_align.read_dfljpg_meta(p) for p in mixed]
    check(all(isinstance(m, dict) and m for m in metas),
          "d 混合目录 read_dfljpg_meta 全部非空", "%d/%d" % (
              sum(1 for m in metas if m), len(metas)))
    imgs = [cv2.imread(str(p), cv2.IMREAD_COLOR) for p in mixed]
    check(all(i is not None and i.ndim == 3 and i.shape[2] == 3 for i in imgs),
          "d 混合目录 cv2.imread 全部成功",
          "shape 集合 %s" % {tuple(i.shape) for i in imgs if i is not None})


# ============================================================
# 主流程
# ============================================================
def main():
    ap = argparse.ArgumentParser(description="对齐脸 JPEG 4:4:4 验收")
    ap.add_argument("--only", default="", help="只跑某几组，逗号分隔（a,b,c,d）")
    args = ap.parse_args()
    only = {x.strip().lower() for x in (args.only or "").split(",") if x.strip()}

    BOX.mkdir(parents=True, exist_ok=True)
    LOGDIR.mkdir(parents=True, exist_ok=True)
    MEDIA.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    print("repo : %s" % REPO, flush=True)
    print("box  : %s" % BOX, flush=True)
    print("python: %s" % sys.executable, flush=True)

    aligned = []
    if not only or "a" in only:
        case_a()
    if not only or "b" in only:
        case_b()
    if not only or "c" in only:
        aligned = case_c() or []
    if not only or "d" in only:
        if not aligned:
            aligned = sorted(ALIGNED.glob("*.jpg"))
        case_d(aligned)

    npass = sum(1 for ok, _n, _d in RESULTS if ok)
    nfail = len(RESULTS) - npass
    print("\n===== 结果：%d 项 · PASS %d · FAIL %d =====" % (len(RESULTS), npass, nfail),
          flush=True)
    for ok, name, detail in RESULTS:
        if not ok:
            print("[FAIL] " + name + "  · " + detail, flush=True)
    print("证据目录：%s" % BOX, flush=True)
    return 0 if nfail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
