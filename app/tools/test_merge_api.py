# -*- coding: utf-8 -*-
"""合成器（Merger / 合成工作室）HTTP 层冒烟测试 —— 一条命令跑完，不需要 uvicorn。

用法（必须用仓库自带的 server 运行时）：
    runtime/server/Scripts/python.exe app/tools/test_merge_api.py
    runtime/server/Scripts/python.exe app/tools/test_merge_api.py -v      # 连响应体一起打

做的事
    1. 在系统临时目录里造一个最小合成项目：
         3 张 256x256 的原帧（纯 JPEG）+ aligned/ 里 3 张 DFLJPG
         （JPEG + APP15 段里 pickle 的 landmarks，和真实项目同一种封装）；
       另外造一个「只有 aligned/」的变体项目，用来验证 DFLJPG 元数据那条路。
    2. **先** 把 DFL_WORKSPACE 指到临时目录里，**再** import server.app ——
       /api/merge/videos 扫工作区、/api/merge/export 找 .dfm 都会落在临时目录，
       仓库下 workspace/ 里的真实数据一个字节都不碰。
    3. 用 FastAPI TestClient（进程内 ASGI，不监听端口、不起 uvicorn）依次打：
         /api/merge/info → /videos → /open → /frame/0
         → /frame/9999（越界必须 4xx，不能是 200 也不能是 500）
         → /detect/0 → /export（参数非法必须 4xx，且不能真起导出）
         → /export/status → /export/cancel
    4. 每条断言失败都打印实际状态码与响应体，定位不用猜。
    5. 结束时删掉整个临时目录（assert 它真的没了）。

退出码
    0 = 全部通过；1 = 有断言失败；2 = 环境/准备阶段就挂了。

依赖
    fastapi + starlette。这份 runtime 的 starlette.testclient 要 httpx2/httpx，
    server 环境里没装 —— 脚本会临时把 runtime/engine 的 site-packages 加进 sys.path
    （只读借用，同为 CPython 3.12）；借不到就退回内置的最小 ASGI 客户端，
    同样进程内、同样不监听端口。
"""

from __future__ import annotations

import base64
import json
import math
import os
import pickle
import re
import shutil
import sys
import tempfile
import traceback
import warnings
from pathlib import Path

# Windows 控制台默认 GBK，中文会乱码；先把 stdout 钉到 utf-8
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

HERE = Path(__file__).resolve().parent          # app/tools
APP_DIR = HERE.parent                           # app
ROOT = APP_DIR.parent                           # 仓库根
REAL_WORKSPACE = ROOT / "workspace"             # 绝对不许被写的地方

VERBOSE = any(a in ("-v", "--verbose") for a in sys.argv[1:])

# ============================================================
# 临时目录 + 环境隔离（必须在 import server.app 之前）
# ============================================================
TMP = Path(tempfile.mkdtemp(prefix="dfl_merge_api_"))
WORK_DIR = TMP / "ws"                    # 假工作区（DFL_WORKSPACE 指这儿）
PROJ = TMP / "proj"                      # 主项目：原帧 + aligned/
PROJ_ALIGNED = TMP / "faces"             # 变体项目：只有 aligned/
ASSETS = TMP / "assets"                  # /api/merge/videos 的扫描目录

os.environ["DFL_WORKSPACE"] = str(WORK_DIR)


# ============================================================
# 断言与响应
# ============================================================
class _Resp(object):
    """统一的响应壳：TestClient 和内置客户端都折成这个。"""

    def __init__(self, status_code: int, headers=None, content: bytes = b""):
        self.status_code = int(status_code or 0)
        self.headers = headers or {}
        self.content = content or b""

    @property
    def text(self) -> str:
        return self.content.decode("utf-8", "replace")

    def json(self):
        return json.loads(self.text)


def _brief(resp) -> str:
    body = resp.text
    # data:image/...;base64,xxxx 有几千字节，会把真正的错误信息淹掉
    body = re.sub(r"(data:image/[a-z]+;base64,)[A-Za-z0-9+/=]{64,}",
                  lambda m: m.group(1) + "<…省略…>", body)
    body = body.replace("\r", " ").replace("\n", " ")
    if len(body) > 1000:
        # 掐中间、留两头：末尾那几行通常才是有用的那句 traceback
        body = body[:520] + "  …(略)…  " + body[-460:]
    return "HTTP %d · %s（共 %d 字节）" % (resp.status_code, body or "<空响应体>", len(resp.content))


class Report(object):
    def __init__(self):
        self.passed = 0
        self.fails = []

    def ok(self, cond, what, resp=None, note=""):
        if cond:
            self.passed += 1
            print("PASS " + what)
            if VERBOSE and resp is not None:
                print("     " + _brief(resp))
            return True
        print("FAIL " + what)
        if resp is not None:
            print("     实际响应: " + _brief(resp))
        if note:
            print("     为什么要管: " + note)
        self.fails.append(what)
        return False

    def equal(self, got, want, what, resp=None, note=""):
        return self.ok(got == want, "%s（期望 %r，实际 %r）" % (what, want, got), resp, note)


def _body_dict(rep: Report, resp: _Resp, what: str) -> dict:
    try:
        d = resp.json()
    except Exception:
        rep.ok(False, "%s：响应体是合法 JSON" % what, resp)
        return {}
    if not isinstance(d, dict):
        rep.ok(False, "%s：响应体是 JSON 对象（拿到 %s）" % (what, type(d).__name__), resp)
        return {}
    return d


def _request(client, method: str, path: str, params: dict = None, body: dict = None) -> _Resp:
    """发一个请求；服务端抛异常时折成 500，好让「必须是 4xx，不能是 500」这类断言有东西可看。"""
    try:
        if method.upper() == "GET":
            r = client.get(path, params=params)
        else:
            r = client.post(path, json=body)
        return _Resp(getattr(r, "status_code", 0), dict(getattr(r, "headers", {}) or {}),
                     getattr(r, "content", b""))
    except Exception as exc:                       # noqa: BLE001
        tb = traceback.format_exc()
        return _Resp(500, {}, ("<服务端抛异常> %s\n%s" % (exc, tb)).encode("utf-8", "replace"))


class _MiniClient(object):
    """没有 httpx 时的兜底：直接把 ASGI app 当函数调，不监听端口、不起 uvicorn。"""

    def __init__(self, asgi_app):
        self.app = asgi_app

    def _call(self, method: str, path: str, params: dict = None, body: dict = None) -> _Resp:
        import asyncio
        from urllib.parse import urlencode

        raw = b"" if body is None else json.dumps(body).encode("utf-8")
        scope = {
            "type": "http", "asgi": {"version": "3.0", "spec_version": "2.3"},
            "http_version": "1.1", "method": method.upper(), "scheme": "http",
            "path": path, "raw_path": path.encode("utf-8"), "root_path": "",
            "query_string": urlencode(params or {}).encode("utf-8"),
            "headers": [(b"host", b"testserver"),
                        (b"content-type", b"application/json"),
                        (b"content-length", str(len(raw)).encode())],
            "client": ("testclient", 50000), "server": ("testserver", 80),
            "state": {},
        }

        async def run():
            box = {"status": 500, "headers": []}
            chunks = bytearray()
            pending = {"body": raw}

            async def receive():
                if pending["body"] is not None:
                    chunk, pending["body"] = pending["body"], None
                    return {"type": "http.request", "body": chunk, "more_body": False}
                return {"type": "http.disconnect"}

            async def send(message):
                if message["type"] == "http.response.start":
                    box["status"] = int(message["status"])
                    box["headers"] = list(message.get("headers") or [])
                elif message["type"] == "http.response.body":
                    chunks.extend(message.get("body") or b"")

            await self.app(scope, receive, send)
            return box["status"], box["headers"], bytes(chunks)

        try:
            status, headers, content = asyncio.run(run())
        except Exception as exc:                   # noqa: BLE001
            return _Resp(500, {}, ("<ASGI 调用抛异常> %s\n%s"
                                   % (exc, traceback.format_exc())).encode("utf-8", "replace"))
        hd = {}
        for k, v in headers:
            try:
                hd[k.decode("latin-1").lower()] = v.decode("latin-1")
            except Exception:
                continue
        return _Resp(status, hd, content)

    def get(self, path: str, params: dict = None) -> _Resp:
        return self._call("GET", path, params=params)

    def post(self, path: str, json: dict = None) -> _Resp:
        return self._call("POST", path, body=json)

    def close(self):
        pass


def _build_client(fastapi_app):
    """优先 FastAPI 自带的 TestClient；这份 runtime 缺 httpx 时按需兜底。"""
    warnings.filterwarnings("ignore")
    err = None
    try:
        from fastapi.testclient import TestClient
        return TestClient(fastapi_app), "fastapi.testclient.TestClient"
    except Exception as exc:                       # noqa: BLE001
        err = exc
    borrowed = ROOT / "runtime" / "engine" / "Lib" / "site-packages"
    if borrowed.is_dir():
        sys.path.append(str(borrowed))             # 只读借用；engine 与 server 同为 CPython 3.12
        try:
            from fastapi.testclient import TestClient
            return (TestClient(fastapi_app),
                    "fastapi.testclient.TestClient（借用 runtime/engine 的 httpx）")
        except Exception as exc:                   # noqa: BLE001
            err = exc
    print("! 拿不到 TestClient（%s）" % str(err)[:160])
    print("  改用内置 ASGI 客户端：同样进程内、同样不监听端口")
    return _MiniClient(fastapi_app), "内置 _MiniClient（进程内 ASGI）"


# ============================================================
# 造最小项目
# ============================================================
def _jpeg_bytes(seed: int = 0) -> bytes:
    """一张 256x256 的“有张脸”的 JPEG，当合成后的原帧用。"""
    import cv2
    import numpy as np

    w = h = 256
    img = np.zeros((h, w, 3), np.uint8)
    img[:, :, 0] = np.linspace(20, 90, w, dtype=np.uint8)[None, :]
    img[:, :, 1] = np.linspace(30, 120, w, dtype=np.uint8)[None, :]
    img[:, :, 2] = 60 + seed * 10
    cv2.ellipse(img, (128, 132), (62, 80), 0, 0, 360, (170, 190, 225), -1, cv2.LINE_AA)
    cv2.circle(img, (104, 112), 9, (40, 40, 45), -1, cv2.LINE_AA)
    cv2.circle(img, (152, 112), 9, (40, 40, 45), -1, cv2.LINE_AA)
    cv2.ellipse(img, (128, 165), (26, 12), 0, 0, 180, (60, 60, 120), 2, cv2.LINE_AA)
    ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    if not ok:
        raise RuntimeError("cv2.imencode 失败")
    return buf.tobytes()


def _landmarks() -> list:
    """68 个点，排成一张脸的椭圆（形状不重要，能被读出来才重要）。"""
    pts = []
    for k in range(68):
        a = 2.0 * math.pi * k / 68.0
        pts.append([round(128 + 60 * math.cos(a), 2), round(128 + 78 * math.sin(a), 2)])
    return pts


def _as_dfljpg(jpeg: bytes, meta: dict) -> bytes:
    """JPEG + APP15 段（FFEF + 长度 + pickle(dict)）—— 和 DFL 的 DFLJPG 同一套路。"""
    payload = pickle.dumps(meta, protocol=4)
    if len(payload) + 2 > 0xFFFF:
        raise RuntimeError("APP15 段太长")
    if jpeg[:2] != b"\xff\xd8":
        raise RuntimeError("不是 JPEG")
    seg = b"\xff\xef" + (len(payload) + 2).to_bytes(2, "big") + payload
    return jpeg[:2] + seg + jpeg[2:]


def _build_fixtures():
    for d in (WORK_DIR, PROJ, PROJ / "aligned", PROJ_ALIGNED / "aligned", ASSETS):
        d.mkdir(parents=True, exist_ok=True)

    names = ["000001.jpg", "000002.jpg", "000003.jpg"]
    for i, name in enumerate(names):
        jpg = _jpeg_bytes(i)
        (PROJ / name).write_bytes(jpg)                     # 原帧：纯 JPEG
        meta = {
            "face_type": 0,
            "shape": (256, 256, 3),
            "landmarks": _landmarks(),
            "source_filename": name,
            "source_rect": [64, 52, 128, 160],
            "image_to_face_mat": None,
        }
        face = name.replace(".jpg", "_0.jpg")
        (PROJ / "aligned" / face).write_bytes(_as_dfljpg(jpg, meta))
        (PROJ_ALIGNED / "aligned" / face).write_bytes(_as_dfljpg(jpg, meta))

    # /api/merge/videos 的素材：一个正常文件 + 一个合成产物（后者应被过滤掉）+ 一个非视频
    (ASSETS / "clip_a.mp4").write_bytes(b"\x00" * 2048)
    (ASSETS / "clip_a_merged.mp4").write_bytes(b"\x00" * 4096)
    (ASSETS / "notes.txt").write_text("not a video", encoding="utf-8")

    # 假的 .dfm：只用来把 /api/merge/export 的校验往后推一层（永远不会被执行）
    (WORK_DIR / "model").mkdir(parents=True, exist_ok=True)
    (WORK_DIR / "model" / "fake.dfm").write_bytes(b"DFM" + b"\x00" * 32)


# ============================================================
# 主体
# ============================================================
def _run(rep: Report) -> int:
    _build_fixtures()

    print("=" * 78)
    print("合成器 HTTP 冒烟测试（FastAPI TestClient，进程内 ASGI，不起 uvicorn）")
    print("临时目录 : %s" % TMP)
    print("假工作区 : %s" % WORK_DIR)
    print("项目目录 : %s" % PROJ)
    print("=" * 78)

    # ---- 导入后端：必须在 DFL_WORKSPACE 设好之后 ----
    if str(APP_DIR) not in sys.path:
        sys.path.insert(0, str(APP_DIR))
    eng = APP_DIR / "engine"
    if eng.is_dir() and str(eng) not in sys.path:
        sys.path.append(str(eng))          # 和 serve.py 一致：引擎代码要惰性 import
    import server.app as S                 # noqa: E402

    ws = Path(S.WORKSPACE).resolve()
    if str(ws).lower() == str(REAL_WORKSPACE.resolve()).lower():
        print("!! WORKSPACE 指到了仓库里的 workspace/（%s），中止，绝不动真实数据" % ws)
        return 2
    rep.ok(str(TMP.resolve()).lower() in str(ws).lower(),
           "环境隔离：DFL_WORKSPACE = %s（在临时目录里）" % ws)
    rep.ok(ws.is_dir() or not ws.exists(), "假工作区可写（未预先创建 workspace/）")

    client, how = _build_client(S.app)
    print("客户端   : %s\n" % how)
    try:
        return _checks(rep, client)
    finally:
        try:
            client.close()
        except Exception:
            pass


def _checks(rep: Report, client) -> int:
    # ================= 1. /api/merge/info（还没打开任何东西） =================
    r = _request(client, "GET", "/api/merge/info")
    rep.equal(r.status_code, 200, "GET /api/merge/info → 200", r)
    d = _body_dict(rep, r, "merge/info")
    for k in ("path", "frames", "fps", "w", "h"):
        rep.ok(k in d, "merge/info 带字段 %r" % k, None if k in d else r)
    rep.equal(d.get("frames", -1), 0, "merge/info 初始 frames = 0", r)

    # ================= 2. /api/merge/videos =================
    r = _request(client, "GET", "/api/merge/videos", params={"dir": str(ASSETS)})
    rep.equal(r.status_code, 200, "GET /api/merge/videos → 200", r)
    d = _body_dict(rep, r, "merge/videos")
    vids = d.get("videos")
    rep.ok(isinstance(vids, list), "merge/videos.videos 是数组", r)
    vids = vids if isinstance(vids, list) else []
    paths = [str(v.get("path") or "") for v in vids if isinstance(v, dict)]
    rep.ok(any(p.endswith("clip_a.mp4") for p in paths),
           "merge/videos 列得出 clip_a.mp4", r, "实际: %r" % paths[:8])
    rep.ok(not any(p.endswith("clip_a_merged.mp4") for p in paths),
           "merge/videos 过滤掉自己的产物 *_merged.mp4", r, "实际: %r" % paths[:8])
    rep.ok(all(str(ASSETS) in p for p in paths),
           "merge/videos 只扫了临时目录（没走真实 workspace）", r, "实际: %r" % paths[:8])
    rep.ok(all(isinstance(v, dict) and {"path", "name", "mb"} <= set(v) for v in vids),
           "merge/videos 条目含 path/name/mb", r)

    # ================= 3. /api/merge/open（目录 = DFL 项目） =================
    r = _request(client, "POST", "/api/merge/open", body={"path": str(PROJ)})
    rep.equal(r.status_code, 200, "POST /api/merge/open(项目目录) → 200", r)
    d = _body_dict(rep, r, "merge/open")
    for k in ("path", "name", "mode", "kind", "frames", "fps",
              "width", "height", "with_landmarks", "sampled"):
        rep.ok(k in d, "merge/open 含字段 %r" % k, None if k in d else r)
    rep.equal(d.get("mode"), "project", "merge/open.mode = project", r)
    rep.equal(d.get("kind"), "project", "merge/open.kind = project（原帧 + aligned/ 都在）", r)
    rep.equal(d.get("frames"), 3, "merge/open.frames = 3", r)
    rep.equal(d.get("sampled"), 3, "merge/open.sampled = 3", r)
    try:
        same = str(Path(str(d.get("path") or "")).resolve()).lower() == str(PROJ.resolve()).lower()
    except Exception:
        same = False
    rep.ok(same, "merge/open.path 回显被打开的项目路径", r)

    # open 之后缓存要真的更新（前端全靠 /info 取状态）
    r = _request(client, "GET", "/api/merge/info")
    d = _body_dict(rep, r, "merge/info(open 之后)")
    rep.equal(d.get("mode"), "project", "open 之后 merge/info.mode = project", r)
    rep.equal(d.get("frames"), 3, "open 之后 merge/info.frames = 3", r)
    rep.equal(len(d.get("names") or []), 3, "open 之后 merge/info.names 有 3 项", r)

    # ================= 4. /api/merge/frame/0 =================
    r = _request(client, "GET", "/api/merge/frame/0", params={"w": 900})
    rep.equal(r.status_code, 200, "GET /api/merge/frame/0 → 200", r)
    d = _body_dict(rep, r, "merge/frame/0")
    rep.equal(d.get("idx"), 0, "merge/frame/0.idx = 0", r)
    rep.equal(d.get("total"), 3, "merge/frame/0.total = 3", r)
    rep.equal(d.get("name"), "000001.jpg", "merge/frame/0.name = 000001.jpg", r)
    png = str(d.get("png") or "")
    rep.ok(png.startswith("data:image/jpeg;base64,"),
           "merge/frame/0.png 是 data:image/jpeg;base64,… 的 URL", r)
    raw = b""
    if "," in png:
        try:
            raw = base64.b64decode(png.split(",", 1)[1])
        except Exception:
            raw = b""
    rep.ok(raw[:2] == b"\xff\xd8", "merge/frame/0.png 解出来是 JPEG（SOI = FF D8）", r)
    shape = None
    try:
        import cv2
        import numpy as np
        img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
        shape = None if img is None else tuple(img.shape[:2])
    except Exception:
        shape = None
    rep.equal(shape, (256, 256), "merge/frame/0 的图是 256x256（不超缩略宽度就不缩放）", r)

    # ================= 5. /api/merge/frame/9999（越界） =================
    r = _request(client, "GET", "/api/merge/frame/9999")
    rep.ok(400 <= r.status_code < 500,
           "GET /api/merge/frame/9999（越界）→ 4xx", r,
           "把越界夹到最后一帧再回 200，前端分不清“越界”和“就这一帧”，"
           "索引算错会被静默吞掉")
    r = _request(client, "GET", "/api/merge/frame/2")
    rep.equal(r.status_code, 200, "越界请求没把缓存搞坏：frame/2 仍 200", r)

    # ================= 6. /api/merge/detect/0 =================
    r = _request(client, "GET", "/api/merge/detect/0",
                 params={"model": "det_500m", "thresh": 0.2})
    lo = 0
    if rep.equal(r.status_code, 200, "GET /api/merge/detect/0 → 200", r,
                 "检测器加载不了，时间轴上的人脸框、按人脸位对齐全都跟着废"):
        d = _body_dict(rep, r, "merge/detect/0")
        for k in ("idx", "width", "height", "model", "boxes"):
            rep.ok(k in d, "merge/detect/0 含字段 %r" % k, None if k in d else r)
        rep.equal(d.get("idx"), 0, "merge/detect/0.idx = 0", r)
        rep.equal((d.get("width"), d.get("height")), (256, 256),
                  "merge/detect/0 报的原图尺寸 = 256x256", r)
        rep.equal(d.get("model"), "det_500m", "merge/detect/0.model 回显请求的检测器", r)
        boxes = d.get("boxes")
        rep.ok(isinstance(boxes, list), "merge/detect/0.boxes 是数组", r)
        if isinstance(boxes, list):
            lo = len(boxes)
            if lo:
                rep.ok(isinstance(boxes[0], dict) and "box" in boxes[0],
                       "merge/detect/0.boxes[0] 带 box 字段", r, "实际: %r" % (boxes[0],))
    else:
        print("     ↑ /detect/0 没通，跳过它后面的字段断言")
    r_hi = _request(client, "GET", "/api/merge/detect/0",
                    params={"model": "det_500m", "thresh": 0.95})
    hi = None
    if r_hi.status_code == 200:
        try:
            hi = len(r_hi.json().get("boxes") or [])
        except Exception:
            hi = None
    rep.ok(hi is not None and 0 <= hi <= lo,
           "merge/detect 的 thresh 真生效（0.95 的框数 %s ≤ 0.2 的框数 %d）" % (hi, lo), r_hi,
           "thresh 没进检测器的话两个阈值会给出同样多的框")

    # ================= 7. /api/merge/export（参数非法 → 4xx，且不许真起任务） =================
    fake_dfm = WORK_DIR / "model" / "fake.dfm"
    empty_aligned = TMP / "empty_aligned"
    empty_aligned.mkdir(parents=True, exist_ok=True)
    no_aligned = TMP / "proj_no_aligned"
    no_aligned.mkdir(parents=True, exist_ok=True)
    (no_aligned / "000001.jpg").write_bytes(_jpeg_bytes(7))

    bad = [
        ("缺 input_dir", {}, "缺输入目录要 400"),
        ("目录不存在", {"input_dir": str(TMP / "nope")}, "不存在的输入目录要 404"),
        ("找不到对齐集", {"input_dir": str(no_aligned), "dfm": str(fake_dfm)},
         "没有 aligned/ 要 400 —— 合成靠 DFLJPG 定位人脸，不能瞎猜"),
        ("对齐集是空的",
         {"input_dir": str(PROJ), "dfm": str(fake_dfm), "aligned_dir": str(empty_aligned)},
         "空对齐集要 400"),
    ]
    for label, payload, why in bad:
        r = _request(client, "POST", "/api/merge/export", body=payload)
        rep.ok(400 <= r.status_code < 500,
               "POST /api/merge/export（%s）→ 4xx" % label, r, why)

    # ================= 8. /api/merge/export/status =================
    r = _request(client, "GET", "/api/merge/export/status")
    rep.equal(r.status_code, 200, "GET /api/merge/export/status → 200", r)
    d = _body_dict(rep, r, "merge/export/status")
    for k in ("running", "ok", "error", "out", "pct", "msg", "log"):
        rep.ok(k in d, "merge/export/status 含字段 %r" % k, None if k in d else r)
    rep.ok(d.get("running") is False,
           "上面 4 次非法导出没把任务跑起来（running = false）", r,
           "参数校验必须在起线程 / 起子进程之前")
    rep.ok(isinstance(d.get("log"), list), "merge/export/status.log 是数组", r)

    # ================= 9. /api/merge/export/cancel =================
    r = _request(client, "POST", "/api/merge/export/cancel")
    rep.ok(r.status_code == 200 or 400 <= r.status_code < 500,
           "POST /api/merge/export/cancel（没在跑）→ 200 或 4xx", r)
    if r.status_code == 200:
        d = _body_dict(rep, r, "merge/export/cancel")
        rep.ok("ok" in d, "merge/export/cancel 返回带 ok 的 JSON", r)
        rep.ok(d.get("ok") is False,
               "没在导出时 cancel 明确回 ok = false", r,
               "静默回 ok = true 会让前端以为取消成功，掩盖状态不同步")
    r = _request(client, "POST", "/api/merge/export/cancel")
    rep.ok(r.status_code == 200 or 400 <= r.status_code < 500,
           "merge/export/cancel 幂等（连点两次不炸）", r)

    # ================= 10. 只有 aligned/ 的项目：DFLJPG 元数据这条路 =================
    r = _request(client, "POST", "/api/merge/open", body={"path": str(PROJ_ALIGNED)})
    rep.equal(r.status_code, 200, "POST /api/merge/open(只有 aligned/) → 200", r)
    d = _body_dict(rep, r, "merge/open(aligned)")
    rep.equal(d.get("kind"), "aligned", "只有 aligned/ 时 kind = aligned", r)
    rep.equal(d.get("frames"), 3, "只有 aligned/ 时 frames = 3", r)
    rep.equal(d.get("with_landmarks"), 3,
              "DFLJPG 的 APP15 landmarks 读得出来（3/3）", r,
              "读不出来 → 合成时定位不到人脸，贴脸位置全错")

    return 0


def main() -> int:
    rep = Report()
    code = 2
    try:
        code = _run(rep)
    except Exception:                              # noqa: BLE001
        print("\n!! 脚本自己挂了：")
        traceback.print_exc()
        code = 2
    finally:
        shutil.rmtree(TMP, ignore_errors=True)
        if TMP.exists():
            print("\n临时目录没删掉: %s" % TMP)
            code = code or 1
        else:
            print("\n临时目录已清理: %s" % TMP)

    print("\n" + "-" * 78)
    print("通过 %d 条，失败 %d 条" % (rep.passed, len(rep.fails)))
    for f in rep.fails:
        print("  FAIL  " + f)
    if code == 2:
        print("结果: 环境/准备阶段失败（见上面 traceback）")
        return 2
    print("结果: %s" % ("全部通过" if not rep.fails else "有失败"))
    return 0 if not rep.fails else 1


if __name__ == "__main__":
    raise SystemExit(main())
