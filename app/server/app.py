"""本地 HTTP 服务：静态界面 + JSON API。

界面是纯静态文件（app/web），后端是 FastAPI。两边只通过 HTTP 说话，
所以这层壳既可以装进 pywebview，也可以原样装进 Electron / Tauri，
甚至直接在浏览器里打开调试。

**这个文件现在只剩装配**：按域拆开的实现在下面这些包里，一张表看完

  _common/bootstrap.py    进程引导（子进程不弹控制台）
  _common/paths.py        WEB / ROOT / WORKSPACE / SETTINGS_FILE / 4.0 引擎路径
  _common/jobs.py         任务注册表（Job / JOB / JOBS / LAST_EXTRACT / 共享运行时状态）
  _common/maskcfg.py      遮罩模式表
  _common/model_info.py   模型档案（dat / 预览 / 架构规格 / GPU）
  _common/runctl.py       运行控制（训练 / DFM 导出 / 启动退出钩子）
  _common/merge_run.py    合成任务编排（起 _merge_runner.py 子进程）
  backend_extract/*       提取器（切帧 / 切脸 / 检测器 / 进度路由）
  backend_dataset/*       数据处理（排序去重 / 应用遮罩 / 统计 / /api/state）

旧名字一个都没丢：文末的 __getattr__（PEP 562）把这些名字原样转给真身模块，所以
from .app import _train_worker 这类老写法、以及 app_mod._merge_proc 这种「按属性取
模块级可变状态」的写法，都照旧可用 —— 拿回来的是真身模块里的当前值，不是快照。
"""
from __future__ import annotations

import os

import json

# 导入即打补丁：Windows 上这套补丁必须在任何子进程被拉起之前生效（原 app.py 第 50 行）
from ._common.bootstrap import _NO_WINDOW, _install_no_window
from fastapi import FastAPI, HTTPException, Response
from fastapi.staticfiles import StaticFiles

from ._paths import engine_python as _engine_python, engine_root as _engine_root
from ._common.paths import (IMAGE_EXT, ROOT, SETTINGS_FILE, WEB, WORKSPACE,
                            _APP_ROOT, _DFL_PY, _DFL_ROOT, _DFL_TOOL)
from ._common.runctl import install_events
from .backend_dataset.stats import _ports
# backend_mask/routes.py 与 painter.py 走 from ..app import (...)：这四个名字必须留在 app 上
from .backend_dataset.maskops import _mask_ai_stub, _mask_state
from .backend_extract.routes import router as _extract_router
from .backend_dataset.routes import router as _dataset_router

# 转发目标：一个个模块对象，属性访问时按顺序找第一个真实存在的
from ._common import jobs as _mod_jobs
from ._common import maskcfg as _mod_maskcfg
from ._common import merge_run as _mod_merge_run
from ._common import model_info as _mod_model_info
from ._common import paths as _mod_paths
from ._common import runctl as _mod_runctl
from .backend_dataset import imageops as _mod_imageops
from .backend_dataset import maskops as _mod_maskops
from .backend_dataset import routes as _mod_dataset_routes
from .backend_dataset import stats as _mod_stats
from .backend_extract import detectors as _mod_detectors
from .backend_extract import frames as _mod_frames
from .backend_extract import media as _mod_media
from .backend_extract import routes as _mod_extract_routes

# 合成用的取帧/建项目（从 backend_extract/frames.py 搬出来的，见该模块头）
from .backend_merge import frames_prep as _mod_frames_prep


app = FastAPI(title="DeepFaceLab 4.1", docs_url=None, redoc_url=None)


_FORWARD_MODULES = (
    _mod_paths, _mod_jobs, _mod_maskcfg, _mod_model_info, _mod_runctl, _mod_merge_run,
    _mod_media, _mod_detectors, _mod_frames, _mod_extract_routes, _mod_frames_prep,
    _mod_imageops, _mod_maskops, _mod_stats, _mod_dataset_routes,
)


def __getattr__(name):
    """PEP 562：搬走的名字在这里转发给真身模块（一次 import，之后就是普通属性）。

    为什么不用 from .x import * 把它们抄进本模块的命名空间：_train_proc / _merge_proc /
    _LAST_EXTRACT 是会被**重新赋值**的模块全局，抄进来就是 import 那一刻的快照，
    训练窗口的 pid、合成进程的句柄会永远停在 None。转发保证拿到的是当前值。
    """
    if name.startswith("__"):
        raise AttributeError("module %r has no attribute %r" % (__name__, name))
    for _m in _FORWARD_MODULES:
        try:
            return getattr(_m, name)
        except AttributeError:
            continue
    raise AttributeError("module %r has no attribute %r" % (__name__, name))


# 启动/退出钩子：清扫遗留训练进程（等价于原来的 @app.on_event 两行装饰器）
install_events(app)


def _prewarm_state() -> None:
    """后台预热 /api/state 的冷启动开销。

    实测进程起来后的**第一个** /api/state 要 ~2.2s：errface 的能力探测（惰性导入
    xgboost）约 1.1s，runtime/datasets 各 0.1s。界面登录完立刻拉这个接口，
    那 2 秒就是「进主页很卡」。这些值在整段进程生命周期里基本不变，
    所以在后台线程先算一遍，首个请求直接命中缓存。
    """
    try:
        from .backend_dataset import stats as _st
        _st._version()
        _st._runtime()
        _st._dataset("data_src")
        _st._dataset("data_dst")
        _st._models()
        _st._capabilities()
        # 目录清单也预热：/api/dataset/count 首次扫 data_dst（4 万张）要 1.8s，
        # 提前在后台跑一遍，用户点开数据集时直接命中缓存。
        from .backend_oneshot import routes as _or
        for _name in ("data_src", "data_dst"):
            _dir = _or.WORKSPACE / _name
            # 真正被数的是 aligned（有它就下探）；两个都预热，key 才对得上
            for _d in (_dir / "aligned", _dir):
                if _d.is_dir():
                    _or._list_cached(_d)
    except Exception:
        pass


try:
    import threading as _threading
    _threading.Thread(target=_prewarm_state, name="state-prewarm", daemon=True).start()
except Exception:
    pass


@app.get("/")
def index():
    """index.html 里的 ?v=N 按**文件修改时间**自动重写。

    之前是写死的 ?v=3，Chromium 会一直复用缓存 —— 改了 app.js 也看不到效果，
    表现出来就是"页面还是老样子"。这里每次请求都按 mtime 生成新版本号，
    再配上 no-store，改完刷新就能生效。
    """
    import re as _re
    html = (WEB / "index.html").read_text(encoding="utf-8")

    # 预览模式（serve.py --preview）：在任何脚本之前把标志打进去。
    # 放到前端脚本里判断已经太晚了 —— 协议门在 41-*.js 里同步执行，
    # 那时 /api/state 还没回来，__preview 仍是 undefined，门照样弹。
    if os.environ.get("DFL_PREVIEW") == "1":
        html = html.replace("<head>",
                            "<head>" + chr(10) +
                            "<script>window.__preview=true;window.__loggedIn=true;</script>",
                            1)

    def _bust(m):
        sub, name = m.group(1), m.group(2)
        try:
            v = int((WEB / sub / name).stat().st_mtime)
        except OSError:
            # app.js 已经没有实体文件了（由 js/app/*.js 拼接返回）：取这些分片的**最大** mtime，
            # 改了任何一个分片，index.html 里的 ?v= 都会变，Chromium 不会复用旧脚本。
            v = 0
            if sub == "js" and name == "app.js":
                try:
                    v = int(max(q.stat().st_mtime for q in _APP_JS_DIR.glob("*.js")))
                except Exception:                       # noqa: BLE001
                    v = 0
        return "/static/%s/%s?v=%d" % (sub, name, v)

    html = _re.sub(r"/static/(js|css)/([\w.\-]+)\?v=\d+", _bust, html)
    return Response(html, media_type="text/html", headers={"Cache-Control": "no-store"})


# ============================================================
# /static/js/app.js —— 由 js/app/*.js 按文件名顺序**拼接**后返回（前端源码已按页拆成 15 个文件）。
#
# 为什么要拼接而不是 <script> 引 15 个文件：整个 app.js 外面是一层 IIFE，
# 594 个函数靠同一个作用域互相调用；拆成多个 <script> 就得先做命名空间重写。
# 拼接保持了「一个作用域、一个文件」的运行语义（逐字节等价，拆分时校验过 roundtrip），
# 同时让改代码的人只在几百行的文件里找。
#
# 这条路由必须注册在 app.mount("/static", ...) **之前**，否则 StaticFiles 先接管路径。
# ============================================================
_APP_JS_DIR = WEB / "js" / "app"


@app.get("/static/js/app.js", include_in_schema=False)
def _app_js_bundle():
    parts = sorted(_APP_JS_DIR.glob("*.js"))
    body = "\n".join(p.read_text(encoding="utf-8") for p in parts)
    return Response(body, media_type="application/javascript",
                    headers={"Cache-Control": "no-store"})


app.mount("/static", StaticFiles(directory=WEB), name="static")
if (WEB / "mask-editor").is_dir():
    app.mount("/mask-editor", StaticFiles(directory=WEB / "mask-editor", html=True), name="mask-editor")

@app.middleware("http")
async def _no_cache(request, call_next):
    """开发期不让 Chromium 缓存界面资源，否则改完 CSS 看不到效果。

    例外：人脸库缩略图（/api/merge/faces/thumb）是带 ETag 的静态缓存，
    必须保留 public/max-age，否则每划一次列表就重新解码一张原图。
    """
    resp = await call_next(request)
    if not request.url.path.startswith("/api/merge/faces/thumb"):
        resp.headers["Cache-Control"] = "no-store, must-revalidate"
    return resp

# ---------------- B 站扫码登录（已拆到 routes_bili.py） ----------------
from .routes_bili import router as _bili_router
app.include_router(_bili_router)

# ---------------- 数据集体检（One Shot）（已拆到 routes_oneshot.py） ----------------

# ---------------- 面部 68 点（已拆到 routes_face.py） ----------------
from .routes_face import router as _face_router
app.include_router(_face_router)

@app.post("/api/settings")
async def save_settings(payload: dict):
    WORKSPACE.mkdir(parents=True, exist_ok=True)
    cur = _ports()
    for k in ("mask", "merge", "angle"):
        if k in payload:
            try:
                cur[k] = int(payload[k])
            except Exception:
                pass
    SETTINGS_FILE.write_text(json.dumps(cur, ensure_ascii=False, indent=2), encoding="utf-8")
    return cur

# 提取器：进度轮询 /api/extract/status（实现见 backend_extract/routes.py）
app.include_router(_extract_router)

# ============================================================
# 远程访问：把整个 WebGUI 在 0.0.0.0:PORT 上再开一份
# 给「部署在服务器上、用别的设备连」的场景
# ============================================================
_remote = {"server": None, "thread": None, "port": 0}


def _lan_ip():
    """猜一下本机在局域网里的地址，用来在界面上显示可访问的 URL。"""
    import socket as _s
    try:
        sk = _s.socket(_s.AF_INET, _s.SOCK_DGRAM)
        sk.connect(("8.8.8.8", 80))
        ip = sk.getsockname()[0]
        sk.close()
        return ip
    except Exception:
        return "127.0.0.1"


def _remote_stop():
    global _remote
    srv = _remote.get("server")
    if srv is not None:
        srv.should_exit = True
    _remote.update({"server": None, "thread": None, "port": 0})


def _remote_start(port: int):
    import threading
    import uvicorn
    _remote_stop()
    cfg = uvicorn.Config(app, host="0.0.0.0", port=port, log_level="warning")
    srv = uvicorn.Server(cfg)
    t = threading.Thread(target=srv.run, daemon=True)
    t.start()
    _remote.update({"server": srv, "thread": t, "port": port})


def _remote_boot():
    """上次开着远程访问的话，这次启动自动恢复。失败就静默跳过（端口被占之类）。"""
    try:
        if not SETTINGS_FILE.is_file():
            return
        cfg = json.loads(SETTINGS_FILE.read_text(encoding="utf-8"))
        if cfg.get("remote_enabled"):
            _remote_start(int(cfg.get("remote_port") or 8002))
    except Exception:
        pass


@app.get("/api/remote")
def remote_state():
    return {
        "enabled": _remote.get("server") is not None,
        "port": _remote.get("port") or 0,
        "ip": _lan_ip(),
        "url": ("http://%s:%d" % (_lan_ip(), _remote["port"])) if _remote.get("port") else "",
    }


@app.post("/api/remote")
def remote_set(body: dict):
    body = body or {}
    enabled = bool(body.get("enabled"))
    try:
        port = int(body.get("port") or 8002)
    except Exception:
        raise HTTPException(400, "端口必须是数字")
    if not (1024 <= port <= 65535):
        raise HTTPException(400, "端口要在 1024–65535 之间")

    try:
        if enabled:
            _remote_start(port)
        else:
            _remote_stop()
    except OSError as e:
        raise HTTPException(500, "端口 %d 起不来: %s" % (port, e))
    except Exception as e:
        raise HTTPException(500, str(e))

    # 记到设置里，下次启动可以按需恢复
    try:
        cfg = {}
        if SETTINGS_FILE.is_file():
            cfg = json.loads(SETTINGS_FILE.read_text(encoding="utf-8"))
        cfg["remote_enabled"] = enabled
        cfg["remote_port"] = port
        SETTINGS_FILE.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        pass

    return remote_state()

# 数据处理：GET /api/state（实现见 backend_dataset/routes.py）
app.include_router(_dataset_router)

# ---------------- 拆出去的路由模块 ----------------
# 引擎编译页（独立包）：GET /api/engines · POST /api/engines/build
from .backend_engines.routes import router as _engines_router
# 训练/设备、测速、遮罩、提取、引擎各有自己的包：原来那个 routes_trt 壳已随 backend_models/trt.py 一起删除
from .backend_bench.routes import router as _bench_router
app.include_router(_bench_router)
app.include_router(_engines_router)

from .routes_account import router as _account_router
app.include_router(_account_router)

from .routes_oneshot import router as _oneshot_router
app.include_router(_oneshot_router)

from .routes_merge import router as _merge_router
app.include_router(_merge_router)

from .routes_mask import router as _mask_router
app.include_router(_mask_router)

from .routes_models import router as _models_router
app.include_router(_models_router)
try:
    from . import routes_update as _update_mod
    app.include_router(_update_mod.router)
except Exception:
    pass

from .routes_train import router as _train_router
app.include_router(_train_router)

from .routes_convert import router as _convert_router
app.include_router(_convert_router)

from .routes_errface import router as _errface_router
app.include_router(_errface_router)
from .routes_weights import router as _weights_router
app.include_router(_weights_router)

# 人脸数据库（一个合成项目一个库：workspace/merge/<视频名>/_faces）
from .routes_faces import router as _faces_router
app.include_router(_faces_router)
