# -*- coding: utf-8 -*-
r"""人脸库「组 = dfm」验收：真后端 + %TEMP% 沙箱（不碰仓库里的 workspace）。

验的是用户定的产品语义：

    放在某个组里的人脸 = 那个组对应的 dfm 模型。新脸归哪一组由**人脸 embedding
    对比**决定（这张脸更像哪个组里的脸）。

跑法（仓库根）：

    runtime\server\Scripts\python.exe app\tools\face_group_accept.py

它自己会：
  1. 在 %TEMP%\dfl_fg_accept\ 搭沙箱：ws\（DFL_WORKSPACE）+ ws\lib\（人脸库）
     + frames\（合成用的两脸帧）。ws\model 是指向仓库 workspace\model 的目录联接，
     所以 .dfm 是真的、但一个字节都不会被写。
  2. 起真后端（app\serve.py，自己的端口），等 /api/merge/faces 能应答。
  3. 用**真人素材**建两个组：杨幂(阿証証証証-416.dfm) / 张元英(张元英.dfm)，
     各放 3 张同一个人的脸。
  4. 抽这两个人的新帧点＋ → 断言自动落到正确的组并打印真实相似度。
  5. 第三个人 → matched:false（不落库）；带 chip 的 dfm → matched:false 但落进该组。
  6. 合成一帧两张脸（一张 A、一张 B，不给 models）→ 断言两张脸各用各的 dfm、auto=true。
  7. 手动指定 models → auto=false 且用了手动值。
  8. 删掉全部 .npy 伪装成老库 → GET 惰性补算（断言文件被创建、第二次 GET 不再算）。
  9. 前端：页面里不能再有「提取到」下拉。

退出码 0 = 全过。
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
APP = REPO / "app"
PY = REPO / "runtime" / "server" / "Scripts" / "python.exe"
TEMP = Path(os.environ.get("TEMP") or os.environ.get("TMP") or ".")
SANDBOX = TEMP / "dfl_fg_accept"
WS = SANDBOX / "ws"
LIB = WS / "lib"
FRAMES = SANDBOX / "frames"
LOG = SANDBOX / "server.log"

# 两个组：显示名 / dfm（dfm 必须是 workspace\model 下真实存在的模型名）
GROUP_A = {"name": "张元英", "dfm": "张元英"}
GROUP_B = {"name": "杨幂", "dfm": "阿証証証証-416"}
THRESHOLD = 0.35

BASE = ""
RESULTS = []

# Windows 上的系统代理（注册表里的 Internet Settings）会被 urllib 用上：
# 本机代理对 127.0.0.1 一律回 404，读起来像"路由不存在"。本机端口不要走代理。
OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))
LAST_ERROR = ""


# ============================================================
# 小工具
# ============================================================
def check(ok, name, detail=""):
    ok = bool(ok)
    RESULTS.append((ok, name, str(detail)))
    print(("[PASS] " if ok else "[FAIL] ") + name + ("  · " + str(detail) if detail else ""),
          flush=True)
    return ok


def say(msg):
    print("       " + str(msg), flush=True)


def http(method, path, body=None, timeout=900.0, raw=False):
    url = BASE + path
    data = None
    headers = {}
    if body is not None:
        data = json.dumps(body, ensure_ascii=False).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with OPENER.open(req, timeout=timeout) as r:
            blob = r.read()
            if raw:
                return r.status, blob
            return r.status, json.loads(blob.decode("utf-8") or "{}")
    except urllib.error.HTTPError as e:
        blob = e.read()
        if raw:
            return e.code, blob
        try:
            return e.code, json.loads(blob.decode("utf-8") or "{}")
        except Exception:                                  # noqa: BLE001
            return e.code, {"error": blob[:300].decode("utf-8", "replace")}


def get(path, **kw):
    return http("GET", path, **kw)


def post(path, body, **kw):
    return http("POST", path, body, **kw)


def b64_img(img, quality=92):
    import base64
    import cv2
    ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        raise RuntimeError("JPEG 编码失败")
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode("ascii")


def read_img(path):
    """Unicode 路径也能读（cv2.imread 在 Windows 上读不了中文路径）。"""
    import cv2
    import numpy as np
    buf = np.fromfile(str(path), dtype=np.uint8)
    return cv2.imdecode(buf, cv2.IMREAD_COLOR) if buf.size else None


def write_img(path, img):
    import cv2
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    if not ok:
        raise RuntimeError("JPEG 编码失败：%s" % path)
    buf.tofile(str(path))


def face_count(data):
    return int(((data or {}).get("stats") or {}).get("active") or 0)


def fetch_library():
    st, d = get("/api/merge/faces?project=" + urllib.parse.quote(str(LIB), safe=""))
    return d if st == 200 else {}


# ============================================================
# 沙箱 / 后端
# ============================================================
def _drop_model_link():
    r"""摘掉 ws\model 这个目录联接（rmdir 只删链接，绝不碰仓库里的 .dfm）。

    顺序很重要：先把链接摘掉再 rmtree。要是先 rmtree，万一实现对"目录联接"
    当普通目录跟进去删，仓库的 workspace\model（1.6 GB 模型）就没了。
    """
    model = WS / "model"
    try:
        if os.path.isdir(str(model)) or os.path.islink(str(model)):
            os.rmdir(str(model))
    except OSError:
        pass


def _child_pids(pid):
    try:
        out = subprocess.run(
            ["powershell", "-NoProfile", "-Command",
             "Get-CimInstance Win32_Process -Filter \"ParentProcessId=%d\" | "
             "Select-Object -ExpandProperty ProcessId" % int(pid)],
            capture_output=True, text=True, timeout=30).stdout
    except Exception:                                      # noqa: BLE001
        return []
    return [int(x) for x in out.split() if x.strip().isdigit()]


def _kill_tree(pid, depth=0):
    if depth > 4:
        return
    for child in _child_pids(pid):
        _kill_tree(child, depth + 1)
    try:
        subprocess.run(["taskkill", "/F", "/PID", str(int(pid))],
                       capture_output=True, timeout=30)
    except Exception:                                      # noqa: BLE001
        pass


def prepare_sandbox(reuse=False):
    if SANDBOX.exists() and not reuse:
        _drop_model_link()
        shutil.rmtree(SANDBOX, ignore_errors=True)
    SANDBOX.mkdir(parents=True, exist_ok=True)
    WS.mkdir(parents=True, exist_ok=True)
    LIB.mkdir(parents=True, exist_ok=True)
    FRAMES.mkdir(parents=True, exist_ok=True)
    model = WS / "model"
    src = REPO / "workspace" / "model"
    if not model.exists():
        # 目录联接（不需要管理员）：读得到真 .dfm，写不进仓库
        r = subprocess.run(["cmd", "/c", "mklink", "/J", str(model), str(src)],
                           capture_output=True, text=True)
        if r.returncode != 0:
            raise SystemExit("建目录联接失败：%s %s" % (r.stdout, r.stderr))
    say("沙箱：%s" % SANDBOX)
    say("工作区：%s（model → %s）" % (WS, src))


def start_server():
    global BASE
    env = dict(os.environ)
    env["DFL_WORKSPACE"] = str(WS)
    env["PYTHONIOENCODING"] = "utf-8"
    env["NO_PROXY"] = env["no_proxy"] = "127.0.0.1,localhost"
    env.pop("DFL_PORT", None)
    logf = open(str(LOG), "wb")
    proc = subprocess.Popen([str(PY), "serve.py"], cwd=str(APP), env=env,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    port = {"v": None}

    def reader():
        for line in iter(proc.stdout.readline, b""):
            try:
                logf.write(line)
                logf.flush()
            except Exception:                              # noqa: BLE001
                pass
            text = line.decode("utf-8", "replace")
            if port["v"] is None and text.startswith("PORT="):
                try:
                    port["v"] = int(text.strip().split("=", 1)[1])
                except Exception:                          # noqa: BLE001
                    pass

    threading.Thread(target=reader, daemon=True).start()
    t0 = time.time()
    while port["v"] is None and time.time() - t0 < 120:
        if proc.poll() is not None:
            raise SystemExit("后端启动失败，见 %s" % LOG)
        time.sleep(0.2)
    if port["v"] is None:
        raise SystemExit("等不到 PORT= 行，见 %s" % LOG)
    BASE = "http://127.0.0.1:%d" % port["v"]
    say("后端：%s（pid=%s）" % (BASE, proc.pid))
    global LAST_ERROR
    while time.time() - t0 < 180:
        try:
            st, d = get("/api/merge/faces?project=", timeout=10)
            if st == 200 and d.get("ok"):
                return proc, port["v"]
            LAST_ERROR = "HTTP %s %s" % (st, str(d)[:200])
        except Exception as e:                             # noqa: BLE001
            LAST_ERROR = "%s: %s" % (type(e).__name__, e)
        time.sleep(0.4)
    raise SystemExit("后端起来了但 /api/merge/faces 不应答（%s），见 %s" % (LAST_ERROR, LOG))


def stop_server(proc):
    """连 preview worker 一起收掉（后端 python.exe 是个转发的 shim，
    真正的解释器是它的子进程 —— 只按 pid 杀会留一个占着端口的孤儿）。"""
    if proc is None:
        return
    _kill_tree(proc.pid)
    try:
        proc.kill()
    except Exception:                                      # noqa: BLE001
        pass


# ============================================================
# 素材
# ============================================================
def video_frames():
    """视频 → {帧号: BGR}（A 的素材：真人视频帧）。"""
    import cv2
    vids = sorted((REPO / "workspace").glob("*.mp4"))
    if not vids:
        raise SystemExit("仓库 workspace 里没有 .mp4 素材")
    cap = cv2.VideoCapture(str(vids[0]))
    out = {}
    try:
        for n in (100, 221, 379, 500, 856, 900):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(n))
            ok, img = cap.read()
            if ok and img is not None:
                out[int(n)] = img
    finally:
        cap.release()
    return vids[0], out


def composite(frame_a, face_b, face_c=None):
    """一张帧里多个不同的人：A 的原帧 + B 贴左上（+ C 贴右下）。

    B/C 都是"已经裁好的人脸图"，贴进真实场景里才检测得出来（紧贴的裁图
    TinyMog 认不出，所以这类素材要配 RetinaFace_10g —— 见调用处）。
    """
    import cv2
    canvas = frame_a.copy()
    h, w = canvas.shape[:2]
    size = max(200, int(min(h, w) * 0.22))

    def _paste(src, x, y):
        patch = cv2.resize(src, (size, size), interpolation=cv2.INTER_AREA)
        canvas[y:y + size, x:x + size] = patch

    _paste(face_b, int(w * 0.06), int(h * 0.08))
    if face_c is not None:
        _paste(face_c, max(0, w - size - int(w * 0.06)), max(0, h - size - int(h * 0.06)))
    return canvas


# ============================================================
# 步骤
# ============================================================
def step_groups():
    names = [GROUP_A["dfm"], GROUP_B["dfm"]]
    st, d = post("/api/merge/faces/groups", {"project": str(LIB), "dfms": names})
    ok = st == 200 and d.get("ok")
    check(ok, "建两个组（%s / %s）" % (GROUP_A["name"], GROUP_B["name"]),
          "dfms=%s" % names if not ok else str(d.get("project") or LIB))
    idx_path = LIB / "_faces" / "index.json"
    if idx_path.is_file():
        raw = json.loads(idx_path.read_text(encoding="utf-8"))
        rename = {GROUP_A["dfm"]: GROUP_A["name"], GROUP_B["dfm"]: GROUP_B["name"]}
        for g in raw.get("groups") or []:
            if g.get("dfm") in rename:
                g["name"] = rename[g["dfm"]]
        idx_path.write_text(json.dumps(raw, ensure_ascii=False, indent=2), encoding="utf-8")
    d = fetch_library()
    groups = d.get("groups") or []
    by_name = {g["name"]: g for g in groups}
    ga = by_name.get(GROUP_A["name"]) or {}
    gb = by_name.get(GROUP_B["name"]) or {}
    check(bool(ga.get("id")) and bool(gb.get("id")),
          "组的显示名 = 张元英 / 杨幂", "g1=%s g2=%s" % (ga.get("id"), gb.get("id")))
    return ga.get("id") or "", gb.get("id") or ""


def add_face(body):
    st, d = post("/api/merge/faces/add", body, timeout=180)
    return st, d


def step_setup_faces(gid_a, gid_b, vid, frames):
    # ---- A（张元英组）：视频里那 3 帧 ----
    for n in (221, 379, 856):
        img = frames.get(n)
        if img is None:
            check(False, "素材帧 %d 读不出来" % n)
            continue
        st, d = add_face({"project": str(LIB), "frame": n, "idx": 0,
                          "source": {"video": "accept_video", "frame": n},
                          "frame_b64": b64_img(img), "group": gid_a})
        ok = st == 200 and d.get("ok") and d.get("group") == gid_a
        check(ok, "A 的脸入库（帧 %d）" % n,
              "group=%s sim=%s matched=%s" % (d.get("group"), d.get("similarity"),
                                              d.get("matched")))
    # ---- B（杨幂组）：data_src 的对齐脸（TinyMog 认不出紧贴的裁图，用 RetinaFace_10g）----
    bsrc = sorted((REPO / "workspace" / "data_src" / "aligned").glob("*.jpg"))
    for k, p in enumerate(bsrc[:3]):
        st, d = add_face({"project": str(LIB), "frame": 2000 + k, "idx": 0,
                          "source": {"video": "accept_b", "frame": 2000 + k},
                          "frame_path": str(p), "detector": "RetinaFace_10g",
                          "group": gid_b})
        ok = st == 200 and d.get("ok") and d.get("group") == gid_b
        check(ok, "B 的脸入库（%s）" % p.name,
              "group=%s sim=%s" % (d.get("group"), d.get("similarity")))


def step_auto_add(gid_a, frames):
    """这两个人的新帧 → 不给 group/dfm → 自动归组（相似度要 > 阈值）。"""
    sims = []
    for n in (100, 500, 900):
        img = frames.get(n)
        if img is None:
            check(False, "新素材帧 %d 读不出来" % n)
            continue
        st, d = add_face({"project": str(LIB), "frame": n, "idx": 0,
                          "source": {"video": "accept_video", "frame": n},
                          "frame_b64": b64_img(img)})
        sim = d.get("similarity")
        if sim is not None:
            sims.append(float(sim))
        ok = (st == 200 and d.get("ok") and d.get("matched") is True
              and d.get("group") == gid_a and d.get("auto") is True
              and sim is not None and float(sim) >= THRESHOLD)
        check(ok, "新帧 %d 自动落到「%s」" % (n, GROUP_A["name"]),
              "similarity=%s ≥ %.2f · group=%s(%s) · source=%s"
              % (sim, THRESHOLD, d.get("group"), d.get("group_name"), d.get("source")))
    return sims


def step_threshold(gid_a):
    csrc = sorted(Path("D:/MyProject/示例图").glob("*.jpg"))
    if not csrc:
        check(False, "找不到第三个人的素材（D:/MyProject/示例图）")
        return
    cpath = csrc[0]
    before = face_count(fetch_library())
    st, d = add_face({"project": str(LIB), "frame": 3000, "idx": 0,
                      "source": {"video": "accept_c", "frame": 3000},
                      "frame_path": str(cpath), "detector": "RetinaFace_10g"})
    after = face_count(fetch_library())
    ok = (st == 200 and d.get("ok") is False and d.get("reason") == "no_group_match"
          and d.get("matched") is False and after == before)
    check(ok, "第三个人（%s）→ matched:false 且不落库" % cpath.name[:26],
          "similarity=%s best=%s 脸数 %d→%d reason=%s"
          % (d.get("similarity"), d.get("best_name"), before, after, d.get("reason")))

    st, d = add_face({"project": str(LIB), "frame": 3001, "idx": 0,
                      "source": {"video": "accept_c", "frame": 3001},
                      "frame_path": str(cpath), "detector": "RetinaFace_10g",
                      "dfm": GROUP_A["dfm"]})
    ok = (st == 200 and d.get("ok") and d.get("matched") is False
          and d.get("source") == "chip" and d.get("group") == gid_a)
    check(ok, "第三个人 + chip 的 dfm → 落进 chip 那一组（matched:false）",
          "group=%s(%s) source=%s similarity=%s"
          % (d.get("group"), d.get("group_name"), d.get("source"), d.get("similarity")))


def step_compose(gid_a, gid_b, frame_a, face_b, face_c=None):
    # 帧 0 = A + B；帧 1 = A + B + C（C 谁的组都不像 → 这张脸必须保持原像素）
    write_img(FRAMES / "000000.jpg", composite(frame_a, face_b))
    write_img(FRAMES / "000001.jpg", composite(frame_a, face_b, face_c))
    st, d = post("/api/merge/open", {"path": str(FRAMES)})
    check(st == 200 and d.get("ok") and int(d.get("frames") or 0) == 2,
          "打开合成素材（2 帧：两脸 / 三脸）", "%s frames=%s"
          % (d.get("mode"), d.get("frames")))

    t0 = time.time()
    st, d = post("/api/merge/frame", {"frame": 0, "project": str(LIB), "w": 900,
                                      "detector": "RetinaFace_10g",
                                      "fmt": "jpg"}, timeout=1200)
    dt = time.time() - t0
    faces = d.get("faces") or []
    ok = (st == 200 and d.get("ok") and d.get("auto") is True and len(faces) >= 2)
    check(ok, "合成：不给 models → 两张脸各自自动分到两个 dfm",
          "%.1fs · swapped=%s faces=%d" % (dt, d.get("swapped"), len(faces)))
    if not ok:
        say(json.dumps(d, ensure_ascii=False)[:800])
        return
    by_group = {f.get("group"): f for f in faces}
    fa, fb = by_group.get(gid_a), by_group.get(gid_b)
    ok = (fa is not None and fb is not None
          and fa.get("dfm") == GROUP_A["dfm"] and fb.get("dfm") == GROUP_B["dfm"]
          and fa.get("auto") is True and fb.get("auto") is True)
    check(ok, "两张脸分别用「%s.dfm」和「%s.dfm」（auto=true）"
          % (GROUP_A["dfm"], GROUP_B["dfm"]),
          "A: idx=%s sim=%s | B: idx=%s sim=%s"
          % (fa and fa.get("idx"), fa and fa.get("similarity"),
             fb and fb.get("idx"), fb and fb.get("similarity")))
    models = d.get("models") or []
    ok = (len(models) == 2
          and all(len(m.get("faces") or []) == 1 for m in models))
    check(ok, "models 里两个 dfm 各负责一张脸",
          json.dumps([{"name": m.get("name"), "faces": m.get("faces")} for m in models],
                     ensure_ascii=False))

    st, d2 = post("/api/merge/frame", {"frame": 0, "project": str(LIB), "w": 900,
                                       "detector": "RetinaFace_10g", "fmt": "jpg",
                                       "models": [{"name": GROUP_A["dfm"], "faces": []}]},
                  timeout=1200)
    faces2 = d2.get("faces") or []
    ok = (st == 200 and d2.get("ok") and d2.get("auto") is False
          and len(faces2) >= 2
          and all(f.get("auto") is False and f.get("dfm") == GROUP_A["dfm"]
                  for f in faces2))
    check(ok, "手动指定 chip（models）→ auto:false 且都用指定的 dfm",
          "faces=%s" % json.dumps([{"idx": f.get("idx"), "dfm": f.get("dfm"),
                                    "auto": f.get("auto")} for f in faces2],
                                  ensure_ascii=False)[:300])

    if face_c is None:
        return
    st, d3 = post("/api/merge/frame", {"frame": 1, "project": str(LIB), "w": 900,
                                       "detector": "RetinaFace_10g", "fmt": "jpg"},
                  timeout=1200)
    faces3 = d3.get("faces") or []
    lonely = [f for f in faces3 if not f.get("dfm")]
    noswap = set(int(x) for x in (d3.get("no_swap_faces") or []))
    ok = (st == 200 and d3.get("ok") and len(faces3) == 3 and len(lonely) == 1
          and lonely[0].get("matched") is False
          and (lonely[0].get("similarity") is None
               or float(lonely[0]["similarity"]) < THRESHOLD)
          and lonely[0].get("applied") is False
          and int(lonely[0].get("idx")) in noswap
          and sum(1 for f in faces3 if f.get("dfm")) == 2)
    check(ok, "画面里第三个人 → 这张脸不换（dfm 空 / matched:false / no_swap）",
          "faces=%s · no_swap=%s"
          % (json.dumps([{"idx": f.get("idx"), "dfm": f.get("dfm"),
                          "sim": f.get("similarity"), "matched": f.get("matched")}
                         for f in faces3], ensure_ascii=False)[:300], sorted(noswap)))


def step_backfill():
    emb = LIB / "_faces" / "emb"
    total = face_count(fetch_library())          # 先记总数（这一次 GET 保证 .npy 齐）
    npy = sorted(emb.glob("*.npy")) if emb.is_dir() else []
    for p in npy:
        p.unlink()
    st, d = get("/api/merge/faces?project=" + urllib.parse.quote(str(LIB), safe=""))
    created = sorted(emb.glob("*.npy")) if emb.is_dir() else []
    ok = (st == 200 and int(d.get("embedded") or 0) == total
          and len(created) == total and total > 0)
    check(ok, "老库（删掉 .npy）→ GET 惰性补算",
          "算 %s / 共 %d 张，落盘 %d 个 .npy"
          % (d.get("embedded"), total, len(created)))
    st, d2 = get("/api/merge/faces?project=" + urllib.parse.quote(str(LIB), safe=""))
    ok = (st == 200 and int(d2.get("embedded") or 0) == 0
          and int(d2.get("emb_have") or 0) == total)
    check(ok, "第二次 GET 不再计算", "embedded=%s emb_have=%s"
          % (d2.get("embedded"), d2.get("emb_have")))


def step_frontend():
    """页面上不能再有那个下拉：按**控件标识**断言（注释里提到"提取到"是正常的）。"""
    st, blob = get("/", raw=True)
    html = blob.decode("utf-8", "replace") if st == 200 else ""
    gone = ("fdbDfm", "fdb-dfm", "fdb-extract", "fdbExtractWrap")
    bad = [k for k in gone if k in html]
    check(st == 200 and not bad,
          "页面里没有「提取到 <dfm>」下拉", "HTTP %s · %d 字节 · 命中 %s"
          % (st, len(html), bad or "无"))
    st, blob = get("/static/js/face_db.js", raw=True)
    js = blob.decode("utf-8", "replace") if st == 200 else ""
    bad = [k for k in ("addTarget", "selectedDfm:", "fdbDfm", "fdb-extract") if k in js]
    check(st == 200 and not bad,
          "face_db.js 里没有 addTarget / selectedDfm / fdbDfm",
          "HTTP %s · %d 字节 · 命中 %s" % (st, len(js), bad or "无"))


# ============================================================
# main
# ============================================================
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reuse", action="store_true", help="复用已存在的沙箱（不删库）")
    ap.add_argument("--keep-server", action="store_true", help="跑完不杀后端")
    ap.add_argument("--skip-compose", action="store_true", help="跳过合成（调试用）")
    args = ap.parse_args()

    prepare_sandbox(reuse=args.reuse)
    vid, frames = video_frames()
    say("A 的素材：%s（帧 %s）" % (vid.name, sorted(frames)))
    bsrc = sorted((REPO / "workspace" / "data_src" / "aligned").glob("*.jpg"))
    say("B 的素材：%s（%d 张对齐脸）" % (bsrc[0].parent, len(bsrc)))

    proc = None
    try:
        proc, _port = start_server()
        gid_a, gid_b = step_groups()
        if not (gid_a and gid_b):
            return 2
        step_setup_faces(gid_a, gid_b, vid, frames)
        sims = step_auto_add(gid_a, frames)
        step_threshold(gid_a)
        if not args.skip_compose:
            face_b = read_img(bsrc[10])
            # 第三个"谁的组都不像"的人：用**和阈值边界那一步不同的**一张
            # （那张已经被"chip 兜底"塞进组里了，再用它就不是"不像"了 ——
            #  组均值会被它自己拉过去，相似度从 0.16 涨到 0.38）。
            csrc = sorted(Path("D:/MyProject/示例图").glob("*.jpg"))
            face_c = read_img(csrc[2] if len(csrc) > 2 else csrc[0]) if csrc else None
            step_compose(gid_a, gid_b, frames.get(500), face_b, face_c)
        step_backfill()
        step_frontend()
    finally:
        if proc is not None and not args.keep_server:
            stop_server(proc)

    bad = [r for r in RESULTS if not r[0]]
    print()
    print("=" * 68)
    print("阈值 %.2f · 自动归组的真实相似度：%s"
          % (THRESHOLD, ", ".join("%.4f" % s for s in sims) if sims else "（无）"))
    print("PASS %d / %d" % (len(RESULTS) - len(bad), len(RESULTS)))
    for ok, name, detail in RESULTS:
        if not ok:
            print("  FAIL " + name + ("  · " + detail if detail else ""))
    print("沙箱：%s" % SANDBOX)
    print("=" * 68)
    return 0 if not bad else 1


if __name__ == "__main__":
    raise SystemExit(main())
