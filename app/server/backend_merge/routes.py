# -*- coding: utf-8 -*-
"""纯路由层：@router 装饰的入口（参数校验 + 调用）+ /api/merge/video 的 Range 解析。

从 app/server/routes_merge.py 按关注点原样切出来的（16741 行的单文件 → 按域分家），
函数体 / 注释 / 日志文案逐字节照抄，只补 import；同包之间的 import 放在文件末尾
（见文末「环上的依赖」），另有一处 __file__ 相对路径下标跟着目录层级改了。
旧行号 → 现文件 的对照表：app/server/backend_merge/MAPPING.md。
"""
from __future__ import annotations

import base64 as _b64
import json
import os
import re
import shutil
import subprocess
import threading
import time
import traceback
from pathlib import Path
from email.utils import formatdate as _httpdate
from fastapi import APIRouter
from fastapi import Request
from fastapi.responses import JSONResponse
from fastapi.responses import StreamingResponse
from .._common.http import (
    _BadInput, _api, _as_dict, _bool_val, _float_val, _int_val, _ok,
)
from .._common import det_names as _det_names
from .._common.proc import (
    _kill_proc_tree,
)
from ._shared import (
    _DEFAULT_DETECTOR, _norm_path, _same_path, _settings_val,
)


router = APIRouter()


# ============================================================
# 路由
# ============================================================
# ============================================================
# DFL 项目模式：可用性判定 + 单一入口
# ============================================================
def _dfl_mode_state(video: str = "", path: str = "", project: str = "") -> dict:
    """「能不能走 DFL 项目模式」的**唯一**判定。

    规则（用户定死的，别再往里加检查）：
      同名文件夹存在
      且 该文件夹下有 aligned/
      且 aligned/ 里至少一张图
      且 该文件夹里至少一张帧图
      → 视为可以开启 DFL 项目模式。

    **不检查人脸与帧图一一对应**，不检查元数据完整性，不抽样比对 —— 那些是导出时
    真出问题再报错的事，不该挡在门口。
    """
    cands = []
    for raw in (project, path):
        raw = str(raw or "").strip()
        if raw:
            cands.append(Path(raw))
    vraw = str(video or "").strip()
    if vraw:
        v = Path(vraw)
        stem = v.stem                      # 同名文件夹：<视频名>/（不含扩展名）
        cands.append(Path(WORKSPACE) / stem)
        cands.append(v.parent / stem)
    seen = set()
    tried = []
    for d in cands:
        key = str(d).lower()
        if key in seen:
            continue
        seen.add(key)
        tried.append(str(d))
        if not d.is_dir():
            continue
        aligned = d / "aligned"
        frames = _scan_first(d, _PROJ_FRAME_EXT, 1)
        faces = _scan_first(aligned, {".jpg", ".jpeg"}, 1) if aligned.is_dir() else []
        if frames and faces:
            return {"ok": True, "dir": str(d), "aligned": str(aligned), "why": ""}
        if not frames:
            return {"ok": False, "dir": str(d), "aligned": str(aligned),
                    "why": "同名文件夹里没有帧图：%s（先抽帧）" % d}
        return {"ok": False, "dir": str(d), "aligned": str(aligned),
                "why": "同名文件夹里没有 aligned/ 或它是空的：%s（先跑人脸对齐）" % aligned}
    return {"ok": False, "dir": "", "aligned": "",
            "why": "找不到同名文件夹（找过：%s）" % ("、".join(tried) if tried else "无输入")}


@router.get("/api/merge/dfl-mode")
@_api
def merge_dfl_mode(video: str = "", path: str = "", project: str = ""):
    """给界面用：当前素材能不能开 DFL 项目模式（判据见 _dfl_mode_state）。"""
    return _ok(**_dfl_mode_state(video=video, path=path, project=project))

def _sr_of_payload(payload: dict) -> int:
    """这一单的超分档位：**顶层 sr 优先，其次 settings.sr**。

    前端把滑块放在合成参数里（settings.sr，和 ct_mode / blend 同级），而导出协议在顶层。
    两个口径都认，免得「界面上滑了、后端没收到」。
    """
    from .knobs import _sr_level as _lv
    payload = payload or {}
    if payload.get("sr") not in (None, ""):
        return int(_lv(payload.get("sr")))
    st = payload.get("settings")
    if isinstance(st, dict):
        return int(_lv(st.get("sr")))
    return 0


def _sr_apply_resp(resp: dict, payload: dict) -> dict:
    """预览响应的**统一出口**：按滑块对这张脸做一次超分，再写回 resp。

    为什么放在路由出口而不是 worker/合成里：
      · worker 是 4.0 引擎的独立解释器，stdio 是 JSON 协议，ORT 的日志会把协议打花；
      · 合成函数有缓存（_COMPOSE_CACHE / _PREVIEW_HIT），缓存命中的响应不会再走合成，
        放在里面就会出现「第一次有效、第二次没效果」。
    在出口处理，所有路径（现算 / 缓存命中 / 单帧 / 批量）表现一致。
    """
    try:
        from .knobs import apply_sr as _apply_sr
        # 先把这一单的档位落到 env：facesr 的 strength()/size() 都读 env，
        # 不设的话这里会用上一单的档（表现是"滑了没反应 / 反应慢半拍"）。
        if int(_apply_sr(_sr_of_payload(payload))) <= 0:
            return resp
        import base64 as _b64s, numpy as _np, cv2 as _cv2
        from .preview import _sr_frame_bytes
        box = resp.get("box")
        for key in ("jpg", "img", "png"):
            u = resp.get(key)
            if not isinstance(u, str) or not u:
                continue
            head, _, b64 = u.partition(",")
            blob = _b64s.b64decode(b64 or u)
            blob = _sr_frame_bytes(blob, box, _np, _cv2)
            resp[key] = (head + "," if head else "data:image/jpeg;base64,") + _b64s.b64encode(blob).decode("ascii")
            if key == "jpg":
                resp["img"] = resp[key]
        return resp
    except Exception as e:                                  # noqa: BLE001
        print("[sr] 预览跳过（%s: %s）" % (e.__class__.__name__, str(e)[:120]), flush=True)
        return resp


_SKIP_SCAN_DIRS = {"model", "data_src", "data_dst", "merged", "_engines", ".git",
                   "node_modules", "__pycache__", "runtime"}


def _walk_videos(root: Path):
    """逐层走目录，跳过模型 / 数据 / 缓存等大目录；顺序稳定，便于前端展示。"""
    stack = [root]
    while stack:
        d = stack.pop()
        try:
            children = sorted(d.iterdir())
        except OSError:
            continue
        subdirs = []
        for p in children:
            try:
                if p.is_dir():
                    name = p.name
                    if name in _SKIP_SCAN_DIRS or name.endswith("_cache") or ".merge_" in name:
                        continue
                    subdirs.append(p)
                elif p.suffix.lower() in _VIDEO_EXT:
                    yield p
            except OSError:
                continue
        stack.extend(reversed(subdirs))


@router.get("/api/merge/videos")
@_api
def merge_videos(dir: str = ""):
    """列出视频文件。dir 给了就只扫它，否则扫工作区。"""
    root = _norm_path(dir)
    if root is not None and not root.is_dir():
        raise _BadInput("目录不存在：%s" % root, 404)
    roots = [root] if root is not None else [Path(WORKSPACE)]
    # 默认只看工作区顶层：导入的视频就复制在这一层，子目录留给项目自己的中间产物
    # （merge/ 里的人脸库、_cache/、data_src/ 之类不该混进素材列表）。
    # 显式给了 dir 时仍递归，方便按目录整理素材的人。

    out, seen = [], set()
    limit, truncated = 200, False
    # 只按“视频文件个数”限流，不按“扫过多少条目”限流：
    # workspace 里有几千个 json / 日志 / 中间帧，按条目计数会在大目录里提前截断，
    # 表现就是排在后面的素材（尤其名字以非 ASCII 开头的）整段消失。
    for r in roots:
        if not r.is_dir():
            continue
        # 默认（没给 dir）只列顶层文件；给了 dir 才递归
        _files = (list(_walk_videos(r)) if root is not None
                  else [p for p in sorted(r.iterdir())
                        if p.is_file() and p.suffix.lower() in _VIDEO_EXT])
        for f in _files:
            if len(out) >= limit:
                truncated = True
                break
            try:
                # 合成产物 / 缓存目录不作为素材列出来（否则列表里全是自己的输出）
                if "_merged" in f.stem or ".merge_" in str(f.parent):
                    continue
                # 导入视频时 prepare 会把源文件复制到 workspace/merge/<视频名>/，
                # 那份副本再被扫出来，列表里就会出现两条一模一样的名字。
                # 副本只是工作区里的中间态，不作为可选素材。
                try:
                    _rp = f.relative_to(Path(WORKSPACE)).parts
                except ValueError:
                    _rp = ()
                if _rp and _rp[0].lower() == "merge":
                    continue
                key = os.path.normcase(str(f))
                if key in seen:
                    continue
                # 同名同体积的也算同一个（同一文件被软链/复制到别处时去重）
                _dup = (f.name.lower(), f.stat().st_size)
                if _dup in seen:
                    continue
                seen.add(key)
                seen.add(_dup)
                out.append({"path": str(f), "name": f.name, "dir": str(r),
                            "mb": round(f.stat().st_size / 1048576, 1)})
            except OSError:
                continue
    return _ok(dir=str(roots[0]), videos=out, count=len(out), truncated=truncated)


@router.post("/api/merge/prepare")
@_api
def merge_prepare(body: dict):
    """视频 → 可合成项目（抽帧 + 对齐）。跑在后台，进度用 /status 拉。"""
    body = _as_dict(body)
    video = _norm_path(body.get("video"))
    if video is None:
        raise _BadInput("缺少 video：要一个视频文件路径")
    if not video.is_file():
        raise _BadInput("视频不存在：%s" % video, 404)
    if video.suffix.lower() not in _VIDEO_EXT:
        raise _BadInput("不是视频文件：%s（只认 %s）"
                        % (video.suffix or "无扩展名", " ".join(sorted(_VIDEO_EXT))))

    target = _norm_path(body.get("dir"))
    if target is None:
        # 副本直接放工作区根：素材列表只扫这一层，用户一眼就能看到自己导入的视频。
        # 以前放在 workspace/merge/<视频名>/，列表递归扫上去就是两条同名的视频。
        target = Path(WORKSPACE)
    if target.exists() and not target.is_dir():
        raise _BadInput("dir 是一个已存在的文件，要目录：%s" % target)
    # ---- 快速导入：只把视频复制进工作区，不抽帧、不对齐 ----
    # 合成本身是「视频模式」：_start_video_export 会自己逐帧解 → 检测 → 合成（见 merge_export），
    # 这里再抽一遍帧 + 对齐纯属重复劳动，还要求目标目录一尘不染。
    # 需要在准备阶段就落盘对齐结果的，传 align=true 走老路径。
    _align = bool(body.get("align"))
    if not _align:
        target.mkdir(parents=True, exist_ok=True)
        dest = target / video.name
        # 同名但内容不同（体积不同）时**不覆盖**——换个带序号的名字，
        # 免得导入 B 视频把 workspace 里同名的 A 视频冲掉。
        try:
            if (dest.exists() and video.resolve() != dest.resolve()
                    and dest.stat().st_size != video.stat().st_size):
                _n = 2
                while True:
                    _cand = target / ("%s (%d)%s" % (video.stem, _n, video.suffix))
                    if not _cand.exists():
                        dest = _cand
                        break
                    _n += 1
        except OSError:
            pass
        try:
            if not dest.exists() or dest.stat().st_size != video.stat().st_size:
                if video.resolve() != dest.resolve():
                    shutil.copy2(str(video), str(dest))
        except OSError as e:
            raise _BadInput("复制视频失败：%s（%s）" % (dest, e))
        with _merge_prep_lock:
            _merge_prep.update({"running": False, "ok": True, "error": "", "pct": 100,
                                "msg": "已复制到 %s" % dest, "stage": "copied",
                                "frames": 0, "aligned": 0, "video": str(dest),
                                "project": str(target), "started_at": time.time(),
                                "finished_at": time.time(), "success": True})
        return _ok(project=str(target), video=str(dest), copied=True)

    _guard_prepare_target(target)

    with _merge_prep_lock:
        if _merge_prep.get("running"):
            raise _BadInput("正在准备另一个视频（%s）—— 等它跑完再试"
                            % (_merge_prep.get("video") or "?"), 409)
        # 置 running 和 project 放在同一把锁里：否则 /status 可能先看到 running=True
        # 却读到上一轮的 project。
        _merge_prep.update({"running": True, "ok": None, "error": "", "pct": 0,
                            "msg": "启动…", "stage": "准备", "frames": 0,
                            "aligned": 0, "video": str(video),
                            "project": str(target), "started_at": time.time()})

    threading.Thread(target=_merge_prepare_worker, args=(video, target),
                     daemon=True, name="merge-prepare").start()
    return _ok(project=str(target), video=str(video))


def _guard_prepare_target(target: Path):
    """准备过程会清空目标目录里的 *.jpg 和 aligned/*.jpg（app._merge_prepare_worker）。

    所以先确认这个目录要么空着，要么就是上次的产物（000001.jpg / aligned/），
    免得手滑指向 data_dst 把用户的帧清光。
    """
    if not target.is_dir():
        return
    try:
        entries = list(os.scandir(target))
    except OSError as e:
        raise _BadInput("目标目录读不了：%s（%s）" % (target, e))
    bad = []
    for e in entries:
        try:
            if e.is_dir():
                # aligned 是产物；aligned.bak_* / aligned_* 是备份 —— 一律不算杂物，
                # 否则备份过一次就再也准备不了这个目录（报「里面已经有别的东西」）。
                if not e.name.lower().startswith("aligned"):
                    bad.append(e.name + "/")
                continue
        except OSError:
            continue
        low = e.name.lower()
        if low.endswith(".mp4"):
            continue
        if low.endswith((".jpg", ".jpeg", ".png")):
            if not re.fullmatch(r"\d{4,8}(_\d+)?\.(jpg|jpeg|png)", low):
                bad.append(e.name)
    if bad:
        raise _BadInput("目标目录里已经有别的东西（%s）—— 准备过程会清掉它的 *.jpg，"
                        "换个空目录或先备份" % "、".join(sorted(bad)[:5]))


@router.get("/api/merge/prepare/status")
@_api
def merge_prepare_status():
    with _merge_prep_lock:
        snap = dict(_merge_prep)
    snap["running"] = bool(snap.get("running"))
    snap["success"] = snap.get("ok")          # 语义同 ok（ok 是任务标志，前端在用）
    return snap


@router.post("/api/merge/open")
@_api
def merge_open(body: dict):
    """打开一个 DFL 项目目录（data_dst / aligned / 裸帧目录）或一个视频。"""
    body = _as_dict(body)
    p = _norm_path(body.get("path"))
    if p is None:
        raise _BadInput("缺少 path：要一个 DFL 项目目录或视频文件")

    # 目录 = DFL 项目（data_dst / aligned / 裸帧目录）
    if p.is_dir():
        names, srcdir, kind = _dfl_project_scan(p)
        if not names:
            raise _BadInput("目录里没有可用的图（.jpg）：%s" % p, 404)
        sample = names[:24]
        lm = 0
        for n in sample:
            try:
                if _dfljpg_dict(Path(srcdir) / n).get("landmarks") is not None:
                    lm += 1
            except Exception:
                continue
        fps = _float_val("fps", body.get("fps"), 0.0, 240.0, 0.0)
        try:
            from ..backend_faces import store as _fstore
            _fstore.set_active_source(p.name)
        except Exception:
            pass
        _cache_set(mode="project", dir=str(p), srcdir=srcdir, names=names,
                       path="", list=[], frames=len(names), fps=fps, w=0, h=0,
                       stale=False, opened_at=time.time())
        return _ok(path=str(p), name=p.name, mode="project", kind=kind,
                   srcdir=srcdir, frames=len(names), fps=fps,
                   width=0, height=0, with_landmarks=lm, sampled=len(sample))

    if not p.exists():
        raise _BadInput("路径不存在：%s" % p, 404)
    if not p.is_file():
        raise _BadInput("既不是目录也不是文件：%s" % p)
    if p.suffix.lower() not in _VIDEO_EXT:
        raise _BadInput("不是视频文件：%s（只认 %s）"
                        % (p.suffix or "无扩展名", " ".join(sorted(_VIDEO_EXT))))
    try:
        frames, fps, w, h = _video_probe(p)
    except Exception as e:
        raise _BadInput("这个视频解析不了（ffmpeg/imageio）：%s（%s）"
                        % (p.name, str(e)[:160]))
    if not frames or int(frames) <= 0 or not fps or float(fps) <= 0:
        raise _BadInput("视频里读不出帧（frames=%s, fps=%s）：%s"
                        % (frames, fps, p.name))
    # 人脸库按素材隔离：告诉 store 现在打开的是哪个视频
    try:
        from ..backend_faces import store as _fstore
        _fstore.set_active_source(p.stem)
    except Exception:
        pass
    _cache_set(mode="video", path=str(p), dir="", list=[], names=[],
               frames=int(frames), fps=float(fps), w=int(w), h=int(h),
               stale=False, opened_at=time.time())
    return _ok(path=str(p), name=p.name, mode="video", frames=int(frames),
               fps=float(fps), width=int(w), height=int(h))


@router.get("/api/merge/frame/{idx}")
@router.post("/api/merge/frame/{idx}")
@_api
def merge_frame(idx: int, body: dict = None, w: int = 900, settings: str = "",
                model: str = "", models: str = "", detector: str = "",
                thresh: float = None, fmt: str = "", quality: int = None,
                raw: str = "", compose: str = "", xseg_onnx: str = "",
                no_swap_faces: str = "", debug: str = ""):
    """第 idx 帧的**合成结果**（data URL）。GET 走 query、POST 走 JSON body，两套都认。

    body（全部可选，缺省 = 当前 UI 默认值）：
        settings: {"ct_mode":"mt"|"rct"|"lct"|"mkl"|"sot",
                   "blend":"overlay"|"hist-match"|"seamless"|"seamless-hist-match",
                   "mask_mode":0~9, "mask_erode":-100~100, "mask_blur":-100~100}
        model: "<dfm 名>"  或  models: [{"name":"<dfm>","faces":[0,1]}]
        detector / thresh / w / fmt(png|jpg) / quality / xseg_onnx
    raw=1（或 compose=0）：不合成，原样返回源帧（视图「原图」用）。
    debug=1：合成 + 遮罩/边界/检测框/HUD 调试叠加（视图「调试」用），
             融合方式照旧由 settings.blend 决定。视图不是融合方式：
             往 settings.blend 里塞 original / raw-* / debug 一律 400。
    """
    payload = _frame_body(idx, body, w=w, settings=settings, model=model,
                          models=models, detector=detector, thresh=thresh,
                          fmt=fmt, quality=quality, raw=raw, compose=compose,
                          xseg_onnx=xseg_onnx, no_swap_faces=no_swap_faces, debug=debug)
    return _frame_payload(idx, payload, default_w=w)


@router.post("/api/merge/frame")
@_api
def merge_frame_post(body: dict = None, w: int = 900, settings: str = "",
                     model: str = "", models: str = "", detector: str = "",
                     thresh: float = None, fmt: str = "", quality: int = None,
                     raw: str = "", compose: str = "", xseg_onnx: str = "",
                     no_swap_faces: str = "", debug: str = ""):
    """POST /api/merge/frame —— 帧号放在 body 里（idx / frame / n），其余同 {idx} 版。

    前端在视频模式下拿的是「时间轴第几帧」而不是 URL 段，所以契约里这个不带路径参数
    的写法必须也认；两套走的是同一个 _frame_payload。
    """
    payload = dict(_as_dict(body))
    _merge_query_into(payload, w=w, settings=settings, model=model, models=models,
                      detector=detector, thresh=thresh, fmt=fmt, quality=quality,
                      raw=raw, compose=compose, xseg_onnx=xseg_onnx,
                      no_swap_faces=no_swap_faces, debug=debug)
    return _frame_payload(_body_idx(payload), payload, default_w=w)


@router.get("/api/merge/face-debug")
@router.post("/api/merge/face-debug")
@_api
def merge_face_debug(body: dict = None, frame: int = None, face: int = None,
                     idx: int = None, settings: str = "", model: str = "",
                     models: str = "", detector: str = "", thresh: float = None,
                     xseg_onnx: str = "", panel_size: int = None):
    """一张脸的六宫格（3 列 × 2 行）。

    GET  /api/merge/face-debug?frame=0&idx=0            —— idx = 人脸序号
    POST /api/merge/face-debug  {"frame":0,"face":0, "settings":{…}, "models":[…]}

    body / query（除 frame、face 外全部可选）：
        frame      帧号（也认 n；**不要**用 idx 传帧号）
        face       这一帧里第几张脸，从 0 起（也认 face_idx / idx）
        settings   {ct_mode, blend, mask_mode, mask_erode, mask_blur, detector, face_type}
        models     [{"name","dfm","faces"}]（不给 = 按人脸库自动判每张脸用哪个 dfm）
        detector / thresh / xseg_onnx / panel_size(默认 256)

    返回 {ok, frame, face:{idx,box,model,dfm,score}, size, panels:[{key,label,w,h,png}], stats}
    panels 顺序固定（行优先）：
        mask-prd     mask(dfm-pred)
        mask-dst     mask(dst)
        mask-both    merged(mask(dfm-pred), mask(dst))
        dfm-pred     dfm-pred
        dst-original dst-original
        merged       merged
    """
    payload = dict(_as_dict(body))
    _merge_query_into(payload, frame=frame, face=face, idx=idx, settings=settings,
                      model=model, models=models, detector=detector, thresh=thresh,
                      xseg_onnx=xseg_onnx, panel_size=panel_size)
    return _face_debug_payload(payload)


@router.post("/api/merge/export")
@_api
def merge_export(body: dict):
    """合成导出。两种输入，走两条链路：

      · 给了 video、又没有帧目录 -> **视频模式**：不要求先建 DFL 项目，
        从源视频逐帧解 -> 检测 -> 68 点 -> 判定模型 -> dfm -> 遮罩 -> 合成 ->
        写帧 -> ffmpeg（音轨 copy）出 mp4（见 _start_video_export）；
      · 给了帧目录 -> 老路：4.0 MergeStudio 子进程一口气出 mp4（下面的代码）。
    """
    body = _as_dict(body)
    # ---- 检测策略：沿用 DFL = 走项目模式（用现成 aligned，不重新检测）----
    # 这是**两个按钮合成一个按钮之后唯一的区分依据**（界面上那格「总是重检 / 沿用 DFL」）。
    # 判定规则见 _dfl_mode_state()：同名文件夹 + aligned/ + 至少一张帧图，仅此而已。
    _policy = str(body.get("detect_policy") or "").strip().lower().replace("-", "_")
    if _policy in ("reuse", "reuse_dfl", "dfl", "project"):
        _st = _dfl_mode_state(video=str(body.get("video") or body.get("source_video") or ""),
                              path=str(body.get("input_dir") or ""),
                              project=str(body.get("project_dir") or body.get("project") or ""))
        if not _st.get("ok"):
            raise _BadInput("DFL 项目模式不可用：%s" % (_st.get("why") or "未知原因"))
        _sub = dict(body)
        _sub["project_dir"] = _st["dir"]
        _sub.setdefault("frames_dir", _st["dir"])
        _sub.setdefault("aligned_dir", _st["aligned"])
        return merge_export_project(_sub)
    # ---- 收到请求先回显一份（脱敏）：路径只留文件名、base64 换成长度 ----
    _eff.log_request("/api/merge/export", body)

    # ---- 0) 视频模式：video 有值 + 没有帧目录 -> 不需要项目 ----
    if (str(body.get("video") or body.get("source_video") or "").strip()
            and not str(body.get("input_dir") or body.get("project") or "").strip()):
        return _start_video_export(body)

    # ---- 1) 输入帧目录：project 和 input_dir 是同一样东西 ----
    ip = _norm_path(body.get("input_dir") or body.get("project"))
    if ip is None:
        raise _BadInput("缺少 input_dir（或 project）：要一个放帧图片的目录")
    if not ip.exists():
        raise _BadInput("输入目录不存在：%s" % ip, 404)
    if not ip.is_dir():
        raise _BadInput("input_dir 不是目录：%s" % ip)
    if not _scan_first(ip, _IMAGE_EXT, 1):
        raise _BadInput("输入目录里没有帧图片（.jpg/.png）：%s —— 先跑一次「抽帧 + 人脸对齐」" % ip)

    # ---- 2) 模型：显式 dfm > 模型名 > 自动挑第一个 ----
    model_name = str(body.get("model") or "").strip()
    if model_name and (("/" in model_name) or ("\\" in model_name)
                       or model_name in (".", "..") or ".." in model_name):
        raise _BadInput("「model」只能是模型名，不能带路径：%r" % model_name)

    dfm_raw = body.get("dfm")
    dfm = None
    if str(dfm_raw or "").strip():
        dfm = _norm_path(dfm_raw)
        if dfm.suffix.lower() != ".dfm":
            raise _BadInput("dfm 要是一个 .dfm 文件：%s" % dfm)
        if not dfm.is_file():
            raise _BadInput("找不到 DFM：%s" % dfm, 404)
    elif model_name:
        name = model_name if model_name.lower().endswith(".dfm") else model_name + ".dfm"
        cand = Path(WORKSPACE) / "model" / name
        if not cand.is_file():
            # 这里以前会悄悄退回"第一个 .dfm"，结果用别的模型出片、没人发现
            raise _BadInput("模型 %s 还没有导出 .dfm（%s）—— 去训练页点「导出」"
                            % (model_name, cand))
        dfm = cand
    if dfm is None:
        cands = sorted((Path(WORKSPACE) / "model").glob("*.dfm"))
        if not cands:
            raise _BadInput("没有可用的 .dfm 模型：先在训练页把模型导出成 .dfm（%s）"
                            % (Path(WORKSPACE) / "model"))
        dfm = cands[0]

    # ---- 3) 对齐集（DFLJPG）：显式 > 输入目录下 aligned/ > 同级 <name>_aligned/ ----
    aligned = _norm_path(body.get("aligned_dir"))
    if aligned is None:
        for c in (ip / "aligned", ip.parent / (ip.name + "_aligned")):
            if c.is_dir() and _scan_first(c, {".jpg", ".jpeg"}, 1):
                aligned = c
                break
    if aligned is None and _has_dfl_meta(ip):
        aligned = ip                            # 输入目录自己就是对齐集
    if aligned is None or not aligned.is_dir():
        raise _BadInput("找不到对齐集（DFLJPG）：期望 %s 下的 aligned/ 或 %s_aligned。"
                        "先跑一次「抽帧 + 人脸对齐」。" % (ip, ip))
    if not _scan_first(aligned, {".jpg", ".jpeg"}, 1):
        raise _BadInput("对齐集是空的：%s" % aligned)
    if not _has_dfl_meta(aligned):
        raise _BadInput("对齐集里没有 DFLJPG 元数据（landmarks）：%s —— "
                        "合成靠它定位人脸，换一个带元数据的 aligned 目录" % aligned)

    # ---- 4) 输出 ----
    out_raw = str(body.get("output_dir") or body.get("output") or "").strip()
    if out_raw:
        op = _norm_path(out_raw)
        if op.suffix.lower() == ".mp4":
            out_mp4 = op
        else:
            if op.exists() and op.is_file():
                raise _BadInput("输出路径是一个已存在的文件：%s（要目录或 .mp4）" % op)
            out_mp4 = op / "merged.mp4"
    else:
        out_mp4 = ip.parent / (ip.name + "_merged.mp4")
    if out_mp4.is_dir():
        raise _BadInput("输出路径是个目录：%s（要 .mp4 文件）" % out_mp4)
    if _same_path(out_mp4, ip):
        raise _BadInput("输出路径不能是输入目录：%s" % out_mp4)
    try:
        out_mp4.parent.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        raise _BadInput("输出目录建不出来：%s（%s）" % (out_mp4.parent, e))

    # ---- 5) XSeg 遮罩模型 ----
    mask_onnx = ""
    xr = body.get("xseg_onnx")
    if str(xr or "").strip():
        xp = _norm_path(xr)
        if not xp.is_file():
            raise _BadInput("找不到 XSeg 模型：%s" % xp, 404)
        mask_onnx = str(xp)
    else:
        for c in (Path(WORKSPACE) / "model" / "XSegLite" / "xseglite.onnx",
                  Path(WORKSPACE) / "model" / "XSeg" / "XSeg.onnx"):
            if c.is_file():
                mask_onnx = str(c)
                break

    # ---- 6) 合成参数 / 编码参数 ----
    settings = _settings_val(body.get("settings"))
    raw_fps = body.get("fps")
    if raw_fps in (None, "", 0, "0", 0.0, "0.0"):
        raw_fps = None
    fps = _float_val("fps", raw_fps, 1.0, 240.0, None)
    if fps is None:
        cached = _float_val("fps", _cache_snap().get("fps"), 0.0, 240.0, 0.0)
        fps = cached if cached > 0 else 30.0
    workers = _int_val("workers", body.get("workers"), 0, 64, 0)
    encoder = _encoder_val(body.get("encoder"))
    hwaccel = _hwaccel_val(body.get("hwaccel"))

    # ---- 6b) 检测器：这条链**以前完全不读它** ----
    # 4.0 子进程（_merge_runner → run_export_pipeline）的 detector 默认是 "YOLOv8"
    # （最重那个），而下面的生效配置硬写 tinymog —— 抽屉说 tinymog、实跑 YOLOv8，
    # 用户体感就是「选了 tinymog 还是慢」。现在：
    #   · 读了就以它为准（未知名字 400，不替你挑一个）；
    #   · 没给 = 4.1 的默认档（tinymog），**显式传下去**，两侧从此同一个值。
    # 转名只在 app/server/_common/det_names.py 一张表里做。
    try:
        det_name = _det_names.canon(body.get("detector"), _det_names.DEFAULT)
    except ValueError as e:
        raise _BadInput(str(e), 400)

    # ---- 7) 起线程：先把状态置成 running 再起线程 ----
    with _merge_job_lock:
        if _merge_job.get("running"):
            raise _BadInput("已经在合成（%s）—— 等它结束或点取消"
                            % (_merge_job.get("out") or "上一个任务"), 409)
        # 否则前端刚拿到 ok=True 就来轮询，会读到线程启动前的旧状态
        # （running=False / ok=None、以及上一轮残留的 cancelled 和 stage）。
        _merge_job.update({"running": True, "log": [], "ok": None, "error": None,
                           "error_detail": "", "out": str(out_mp4), "pct": 0,
                           "msg": "启动合成…", "stage": -1, "cancelled": False,
                           "pct_at_failure": None, "started_at": time.time(),
                           "finished_at": None, "fps": fps, "workers": workers,
                           "encoder": encoder, "hwaccel": hwaccel,
                           "dfm": str(dfm),
                           # 上一轮可能是视频模式：视频专有的明细一起清掉，
                           # 否则 /export/status 会把上一轮的逐帧记录当成这一轮的
                           "mode": "project", "video": "", "total": 0,
                           "frames_done": 0, "frames": [], "stats": {},
                           "cache": {}, "encode": {},
                           "eff": None, "eff_warnings": [], "warnings": []})
    # ---- 生效配置（老项目模式：4.0 MergeStudio 子进程一口气出 mp4）----
    # 这条路的检测器由子进程选，但合成参数/模型/并发照样要对得上账。
    try:
        _pcfg = {"settings": settings, "config": _merge_settings_to_config(settings),
                 # 以前这两个都是 _DEFAULT_DETECTOR（写死 tinymog，只为「对上账」）：
                 # 子进程按 4.0 的默认 YOLOv8 跑，抽屉却报 tinymog。现在报的是真值。
                 "detector": det_name, "det_align": det_name,
                 "mask_src": _mask_src(settings, settings.get("mask_mode")),
                 "dfm": str(dfm), "model": model_name, "single_models": [],
                 "library": "", "use_library": False, "chain": False, "fused": False,
                 "range": {}, "workers": workers, "batch": 0,
                 "jpeg_quality": _STAGE_JPEG_QUALITY}
        eff = _eff_publish("/api/merge/export", "project-modes", body, _pcfg)
        with _merge_job_lock:
            _merge_job.update(_eff_status_patch(eff))
    except Exception as e:                                  # noqa: BLE001
        print("[eff] 生成失败（不影响这一单）：%s: %s"
              % (e.__class__.__name__, str(e)[:200]), flush=True)
    threading.Thread(target=_run_export,
                     args=(str(ip), str(out_mp4), str(aligned), str(dfm), settings,
                           mask_onnx, fps, workers, encoder, hwaccel),
                     kwargs={"detector": det_name},
                     daemon=True, name="merge-export").start()
    # output 是**文件**（mp4 全路径）。以前这里还多给一个 output_dir=同一个值，
    # 键名撒谎（它指向文件、不是目录），全仓没有一处读它 —— 前端读的是 d.output
    # （app.js:8528）。删掉，别让下一个照键名猜语义。
    return _ok(input_dir=str(ip), aligned_dir=str(aligned), output=str(out_mp4),
               dfm=str(dfm), model=model_name,
               settings=settings, fps=fps, workers=workers, encoder=encoder,
               hwaccel=hwaccel, mask_onnx=mask_onnx,
               detector=det_name, detector_engine=_det_names.engine(det_name))


@router.get("/api/merge/export/status")
@_api
def merge_export_status():
    """导出进度。ok = 任务是否成功（前端 d.ok 决定弹完成还是失败，所以不能恒 true）。"""
    with _merge_job_lock:
        snap = dict(_merge_job)
        log = list(_merge_job.get("log") or [])
    log_tail = log[-50:]

    cancelled = bool(snap.get("cancelled"))
    ok = snap.get("ok")
    # 取消了就对外立刻不是 running（进程正在被看门狗收掉），状态落 cancelled
    running = bool(snap.get("running")) and not cancelled
    if running:
        phase = "running"
    elif ok is True:
        phase = "done"
    elif cancelled:
        phase = "cancelled"
    elif ok is False:
        phase = "failed"
    else:
        phase = "idle"

    error = str(snap.get("error") or "")
    detail = str(snap.get("error_detail") or "")
    if not detail and error:                      # 兜底：至少把日志尾巴给出来
        detail = "\n".join(_log_tail(log, 20))

    stage = snap.get("stage", -1)
    stage = -1 if stage is None else int(stage)
    if phase in ("failed", "cancelled"):
        stage = -1
    pct = int(snap.get("pct") or 0)
    if phase in ("failed", "cancelled"):
        pct = 0
    started = snap.get("started_at")
    finished = snap.get("finished_at")
    try:
        elapsed = round(float((finished or time.time()) - started), 1) if started else None
    except Exception:
        elapsed = None

    mode = str(snap.get("mode") or "")
    resp = {"ok": ok, "success": ok, "phase": phase, "running": running,
            # active = worker 还活着（起单 → 收尾结束）。取消之后 running 立刻变 false
            # （看门狗靠这条），所以「还能不能再点一次导出」要看 active。
            # active = worker 线程还活着（从起单到收尾结束）。取消之后 running 立刻变
            # false（看门狗靠这条），所以「能不能再点一次导出」要看 active。
            "active": bool(snap.get("active")),
            "cancelled": cancelled, "stage": stage, "pct": pct,
            # 分相位流水线（多模型）：后端声明，前端只画。
            # 这两个键以前被这里"整个丢掉"—— 前端拿不到 shape/phase，只能照六列画，
            # 于是「图上六个阶段并发、实际一趟一趟串行」，看起来当然对不上。
            "mode": snap.get("mode"),
            "mode_why": snap.get("mode_why"),
            "pipeline": snap.get("pipeline"),
            "mask_on": snap.get("mask_on"),
            "pool_frozen": snap.get("pool_frozen"),
            "pool": snap.get("pool"),
            "pass_info": snap.get("pass"),
            "work_phase": snap.get("phase"),
            "pct_at_failure": snap.get("pct_at_failure"),
            "msg": str(snap.get("msg") or ""),
            "error": error, "error_detail": detail,
            "out": str(snap.get("out") or ""),
            "log": log_tail, "log_tail": log_tail[-12:],
            "started_at": started, "finished_at": finished, "elapsed": elapsed,
            "fps": snap.get("fps"), "workers": snap.get("workers"),
            "encoder": snap.get("encoder"), "hwaccel": snap.get("hwaccel"),
            "dfm": snap.get("dfm"),
            "mode": mode or "project",
            # 生效配置 + 一致性警告（进度页「当前生效」面板 / 顶部横幅读这两个键）
            "eff": snap.get("eff"), "eff_warnings": list(snap.get("eff_warnings") or []),
            "warnings": list(snap.get("warnings") or []),
            # 流水线实况（进度页那条 SVG 图的唯一数据源）：六个 key =
            # frames/det/dfm/xseg/compose/encode，每个给 ticks / inflight / lanes(+ms/batch)。
            # 没开过单时是空 dict —— 前端据此显示「等待流水线…」，不猜 lane 数。
            "stages": _STAGE_METER.snapshot()}
    if mode == "video":
        # 视频模式多给一段明细：逐帧记录（每张脸用了谁）+ 计数 + 结束时的缓存状态。
        # 老前端不认识这些字段，多给不会影响它（缺字段才是坑）。
        fused = bool(snap.get("fused"))
        stats = dict(snap.get("stats") or {})
        resp["vid"] = {"video": str(snap.get("video") or ""),
                       "total": int(snap.get("total") or 0),
                       "done": int(snap.get("frames_done") or 0),
                       "stats": stats,
                       "cache": dict(snap.get("cache") or {}),
                       "encode": dict(snap.get("encode") or {}),
                       "frames": list(snap.get("frames") or [])[-600:],
                       # 快速导出（融合流水线）专有：没有阶段概念，只有帧进度 + 帧预算
                       "fused": fused,
                       "resumable": (not fused),
                       "frame_budget": (snap.get("frame_budget")
                                        if snap.get("frame_budget") is not None
                                        else stats.get("frame_budget")),
                       "frame_fps": float(stats.get("fps") or 0.0),
                       "eta_s": stats.get("eta_s"),
                       "cache_dir": str(snap.get("cache_dir") or ""),
                       "per_stage_ms": dict(stats.get("per_stage_ms") or {}),
                       "stage_util_pct": dict(stats.get("stage_util_pct") or {}),
                       "pipeline": dict(stats.get("pipeline") or {}),
                       # 坏帧降级（DFL_FUSED_SKIP_BAD=1，默认关）：成了几帧坏帧、
                       # 帧号是哪些。默认关时这里恒为 0/[]（键在，值空）。
                       "skip_bad": bool(snap.get("skip_bad") or stats.get("skip_bad")),
                       "skipped_count": int(snap.get("skipped_count")
                                            or stats.get("skipped_count") or 0),
                       "skipped_frames": list(snap.get("skipped_frames")
                                              or stats.get("skipped_frames") or [])}
        # 顶层也给一份（前端不必再进 vid 里翻）
        resp["fused"] = fused
        resp["resumable"] = (not fused)
        resp["frame_budget"] = resp["vid"]["frame_budget"]
        resp["no_cache"] = bool(fused)
        resp["skip_bad"] = resp["vid"]["skip_bad"]
        resp["skipped_count"] = resp["vid"]["skipped_count"]
        resp["skipped_frames"] = list(resp["vid"]["skipped_frames"])
    # 显存计划（三条路共用一个形状）：训练/导出出问题时第一眼看它 ——
    # 里面有可用显存、需求拆项、选了哪一档、有没有降档、是不是在 CPU 上跑。
    resp["vram_plan"] = dict(snap.get("vram_plan") or _vram_last("快速导出") or {})
    resp["on_cpu"] = bool(resp["vram_plan"].get("cpu")) or bool(snap.get("on_cpu"))
    # 强制尝试撞上 OOM 的现场（有值 = 这一单是「确实不够、已回滚」，前端据此回到被拒态；
    # 里面带 summary/need/avail/reserve/free/own/other，和 400 的结构化字段同一份形状）
    resp["vram_oom"] = (dict(snap.get("vram_oom") or {}) or None)
    return resp


@router.get("/api/merge/merged/{idx}")
@router.post("/api/merge/merged/{idx}")
@_api
def merge_merged_frame(idx: int, body: dict = None, w: int = 900, settings: str = "",
                       model: str = "", models: str = "", detector: str = "",
                       thresh: float = None, fmt: str = "", quality: int = None,
                       raw: str = "", compose: str = "", xseg_onnx: str = "",
                       no_swap_faces: str = ""):
    """第 idx 帧的成片图（data URL）——**现算**，参数改了就立刻变。

    以前这里只会去「最近一次导出的成片」里抠一帧：那是导出那一刻的参数，
    所以右侧改 ct_mode / blend / 遮罩，画面一动不动（用户报的就是这个）。
    现在：
      · 有打开的素材（视频 / 帧目录）→ 和第 idx 帧**现算**一遍（和 /frame 同一条
        合成链路、同一份参数），响应里 source="live"
      · 没有素材、只有历史成片 → 仍退回「最近一次导出的成片」里那一帧（source="export"）
    body 和 /api/merge/frame 完全一致（settings / model / models / detector / …）。
    """
    if idx < 0:
        raise _BadInput("帧号不能是负数：%d" % idx)
    payload = _frame_body(idx, body, w=w, settings=settings, model=model,
                          models=models, detector=detector, thresh=thresh,
                          fmt=fmt, quality=quality, raw=raw, compose=compose,
                          xseg_onnx=xseg_onnx, no_swap_faces=no_swap_faces)

    snap = _cache_snap()
    mode = _cache_mode(snap)
    if mode:
        try:
            _check_cache_ready(snap)
        except _BadInput:
            mode = ""                        # 素材没了：退到成片那条路
    if mode:
        resp = _frame_payload(idx, payload, default_w=w)
        resp["source"] = "live"
        return resp
        return resp
    maxw = max(200, min(int(payload.get("w") or 900), 1600))
    return _merged_from_export(idx, maxw, payload)


@router.post("/api/merge/merged")
@_api
def merge_merged_post(body: dict = None, w: int = 900, settings: str = "",
                      model: str = "", models: str = "", detector: str = "",
                      thresh: float = None, fmt: str = "", quality: int = None,
                      raw: str = "", compose: str = "", xseg_onnx: str = "",
                      no_swap_faces: str = ""):
    """POST /api/merge/merged —— 帧号放在 body 里，其余同 {idx} 版。"""
    payload = dict(_as_dict(body))
    _merge_query_into(payload, w=w, settings=settings, model=model, models=models,
                      detector=detector, thresh=thresh, fmt=fmt, quality=quality,
                      raw=raw, compose=compose, xseg_onnx=xseg_onnx,
                      no_swap_faces=no_swap_faces)
    idx = _body_idx(payload)
    snap = _cache_snap()
    mode = _cache_mode(snap)
    if mode:
        try:
            _check_cache_ready(snap)
        except _BadInput:
            mode = ""
    if mode:
        resp = _frame_payload(idx, payload, default_w=w)
        resp["source"] = "live"
        return resp
        return resp
    return _merged_from_export(idx, max(200, min(int(payload.get("w") or 900), 1600)),
                               payload)


@router.post("/api/merge/export/cancel")
@_api
def merge_export_cancel():
    """终止正在跑的合成。幂等：没在跑、重复点，都返回 ok=true。"""
    with _merge_job_lock:
        running = bool(_merge_job.get("running"))
        was_cancelled = bool(_merge_job.get("cancelled"))
        _merge_job["cancelled"] = True
        if running:
            _merge_job["msg"] = "正在取消…"

    proc = getattr(_app_mod, "_merge_proc", None)
    try:
        alive = bool(proc is not None and proc.poll() is None)
    except Exception:
        alive = False

    if not running and not alive:
        with _merge_job_lock:
            _merge_job["running"] = False
            if _merge_job.get("ok") is None:
                _merge_job["msg"] = "没有正在进行的合成"
        return _ok(cancelled=True, running=False, already=True, killed=False,
                   reason="没有正在进行的合成")
    if was_cancelled and not alive:                 # 重复点：已经请求过、进程也不在了
        return _ok(cancelled=True, running=False, already=True, killed=False,
                   reason="已经在取消中")

    killed = _kill_proc_tree(proc) if alive else False
    _ensure_cancel_watch()
    return _ok(cancelled=True, running=False, already=False, killed=killed,
               reason="正在终止合成进程" if killed else "已请求取消，等子进程退出")


@router.get("/api/merge/detect/{idx}")
@_api
def merge_detect(idx: int, model: str = _DEFAULT_DETECTOR, thresh: float = 0.5):
    """对当前帧做人脸检测，返回原图坐标的框。视频 / 项目两种模式都支持。

    model 缺省 = tinymog。retinaface_500m / det_500m / 任何没见过的名字 → 400。
    """
    t = _float_val("thresh", thresh, 0.0, 1.0, 0.5)
    det_name = _detector_val(model)
    frame, total = _frame_image(idx)
    det = _detector(det_name)
    with _det_run_lock:                    # thresh 是共享检测器上的可变字段
        det.thresh = t
        try:
            boxes = det.detect(frame)
        except Exception as e:
            raise _BadInput("检测失败：%s" % str(e)[:200], 500)
    h, w = frame.shape[:2]
    # 每张脸顺手回归一次 68 点（2D106）：前端点「＋」时直接把它交给
    # /api/merge/faces/add 落库，和人脸库 extract 的取点链路是同一条。
    # 回归不出来就不给这个键（add 那边会明确报 no_landmarks68，绝不退回 5 点）。
    boxes = _with_kps68(frame, boxes)
    return _ok(idx=int(idx), total=int(total or 0), width=int(w), height=int(h),
               model=model, detector=det_name, thresh=t, boxes=boxes)


@router.get("/api/merge/analyze")
@_api
def merge_analyze(step: int = 12, model: str = _DEFAULT_DETECTOR, thresh: float = 0.5):
    """按步长扫全片，统计每帧的人脸数，给时间轴的人脸轨用。
    486 帧 / step=12 → 41 次推理，约 20 秒。model 缺省 = tinymog。"""
    step = _int_val("step", step, 1, 500, 12)
    det_name = _detector_val(model)
    t = _float_val("thresh", thresh, 0.0, 1.0, 0.5)
    snap = _cache_snap()
    total = int(snap.get("frames") or 0)
    _check_cache_ready(snap)
    if total <= 0:
        raise _BadInput("还没有打开素材（先 /api/merge/open 或 /api/merge/prepare）", 409)
    det = _detector(det_name)

    marks, idx, scanned, broken = [], 0, 0, ""
    # 单帧检测失败不该让整次分析挂掉：那只是一帧没读到脸，时间轴上少一个采样点而已。
    # 以前这里是 raise，于是第 43608 帧一次 ORT 抖动（CUDA failure 900 那种）就把整条
    # 分析废掉，界面上弹一句「分析失败」—— 而此时导出页正显示「12535 帧 完成」，
    # 两件事没关系，用户却只能看到一条红字。
    # 连续 _DET_FAIL_MAX 次才真放弃：那是系统性问题（context 坏了 / 模型没加载），
    # 继续扫只会把同一句话重复上万次。
    _fail_streak = 0
    _fail_first = ""
    _DET_FAIL_MAX = 8
    while idx < total:
        try:
            frame, _n = _frame_image(idx)
        except _BadInput as e:              # 扫到一半文件没了：别装作扫完了
            broken = e.msg
            break
        with _det_run_lock:
            det.thresh = t
            try:
                boxes = det.detect(frame)
                _fail_streak = 0
            except Exception as e:
                _fail_streak += 1
                if not _fail_first:
                    _fail_first = "第 %d 帧：%s" % (idx, str(e)[:200])
                if _fail_streak >= _DET_FAIL_MAX:
                    raise _BadInput("检测连续失败 %d 次（%s）—— 不是单帧问题，先看输出页有没有 CUDA / 显存报错" % (_fail_streak, _fail_first), 500)
                boxes = []            # 这一帧按「没脸」算，继续往下扫
        marks.append({"idx": idx, "count": len(boxes),
                      "box": boxes[0]["box"] if boxes else None})
        scanned += 1
        idx += step
    return _ok(total=total, step=step, scanned=scanned, model=model,
               detector=det_name, thresh=t, marks=marks, truncated=bool(broken),
               error=broken)


@router.post("/api/merge/project")
@_api
def merge_project(body: dict):
    """看一眼目录是不是能用的 DFL 项目：有几张帧、有没有 aligned、带不带 landmarks。"""
    body = _as_dict(body)
    p = _norm_path(body.get("path"))
    if p is None:
        raise _BadInput("缺少 path：要一个项目目录")
    if not p.exists():
        raise _BadInput("目录不存在：%s" % p, 404)
    if not p.is_dir():
        raise _BadInput("不是目录：%s" % p)
    names, srcdir, kind = _dfl_project_scan(p)
    aligned = p / "aligned"
    sample = names[:24]
    lm = 0
    for n in sample:
        try:
            if _dfljpg_dict(Path(srcdir) / n).get("landmarks") is not None:
                lm += 1
        except Exception:
            continue
    return _ok(path=str(p), kind=kind, total=len(names),
               has_aligned=aligned.is_dir(),
               aligned_dir=str(aligned) if aligned.is_dir() else "",
               frames_dir=srcdir,
               with_landmarks=lm,     # 采样 24 张的结果，不是全量
               sampled=len(sample), sample=names[:6])


@router.get("/api/merge/info")
@_api
def merge_info():
    """当前打开的东西。names 可能几万条，不回传（只给条数）。"""
    snap = _cache_snap()
    names = snap.get("names") or []
    payload = {k: v for k, v in snap.items() if k not in ("names", "list")}
    mode = _cache_mode(snap)
    stale = False
    if mode == "project":
        stale = not Path(str(snap.get("srcdir") or "")).is_dir()
    elif mode == "video":
        stale = not Path(str(snap.get("path") or "")).is_file()
    opened = snap.get("opened_at")
    payload.update({"mode": mode, "stale": bool(stale) or bool(snap.get("stale")),
                    "names_count": len(names),
                    "age": round(time.time() - float(opened), 1) if opened else None})
    return _ok(**payload)


# ============================================================
# 视频播放（HTTP Range）
# 前端 <video> 拖时间轴要 206 Partial Content，FileResponse 不支持 Range
# （它只会回 200 + 整个文件，浏览器拿不到 Content-Range 就不给拖），
# 所以这里自己解析 Range 头、自己读片段。
# ============================================================
_VIDEO_CTYPE = {
    ".mp4": "video/mp4", ".m4v": "video/mp4", ".mov": "video/quicktime",
    ".mkv": "video/x-matroska", ".webm": "video/webm", ".avi": "video/x-msvideo",
    ".flv": "video/x-flv", ".wmv": "video/x-ms-wmv", ".ogv": "video/ogg",
    ".ts": "video/mp2t", ".m2ts": "video/mp2t", ".mts": "video/mp2t",
    ".3gp": "video/3gpp", ".mpg": "video/mpeg", ".mpeg": "video/mpeg",
}
_VIDEO_CHUNK = 512 * 1024


def _video_ctype(p) -> str:
    return _VIDEO_CTYPE.get(Path(str(p)).suffix.lower(), "application/octet-stream")


def _inside_repo(p) -> bool:
    """路径是不是在仓库里（Windows 上大小写不敏感）。"""
    try:
        root = os.path.normcase(os.path.abspath(str(_REPO_ROOT)))
        full = os.path.normcase(os.path.abspath(str(p)))
    except Exception:
        return False
    return full == root or full.startswith(root + os.sep)


def _raw_is_absolute(raw) -> bool:
    """界面上传来的原始串本身是不是绝对路径（D:/x、\\\\server\\share、/x）。"""
    s = str(raw or "").strip().strip('"').strip("'").strip()
    if not s:
        return False
    try:
        if Path(s).is_absolute():
            return True
    except Exception:
        pass
    try:
        import ntpath
        drive, _rest = ntpath.splitdrive(s)
        return bool(drive)
    except Exception:
        return False


def _parse_range(raw, size: int):
    """Range: bytes=a-b -> (start, end) 闭区间。

    返回 (None, None) 表示「没这个头 / 认不了」→ 调用方回 200 全量；
    返回 ("bad", None) 表示区间不合法 / 越界 → 调用方回 416。
    """
    text = str(raw or "").strip().lower()
    if not text or not text.startswith("bytes="):
        return None, None                 # 别的单位（items=…）按 RFC 忽略
    spec = text[6:].split(",")[0].strip()  # 多段只认第一段
    if not spec or "-" not in spec:
        return None, None
    head, _, tail = spec.partition("-")
    head, tail = head.strip(), tail.strip()
    if not head and not tail:
        return None, None
    try:
        if not head:                      # "-500" = 最后 500 字节
            n = int(tail)
            if n <= 0:
                return "bad", None
            start, end = max(0, size - n), size - 1
        else:
            start = int(head)
            end = int(tail) if tail else size - 1
    except ValueError:
        return None, None
    if size <= 0 or start < 0 or start >= size:
        return "bad", None
    end = min(end, size - 1)
    if end < start:
        return "bad", None
    return start, end


def _file_chunks(path, start: int, length: int, chunk: int = _VIDEO_CHUNK):
    """从 start 起读 length 字节（生成器：客户端断开时 finally 会关文件）。"""
    fh = open(str(path), "rb")
    try:
        fh.seek(int(start))
        left = int(length)
        while left > 0:
            buf = fh.read(min(chunk, left))
            if not buf:
                break
            left -= len(buf)
            yield buf
    finally:
        try:
            fh.close()
        except Exception:
            pass


@router.get("/api/merge/video")
@_api
def merge_video(request: Request, path: str = ""):
    """把视频文件原样吐给 <video>。支持 Range（206），没有 Range 就 200 全量流式。"""
    raw = str(path or "")
    p = _norm_path(raw)
    if p is None:
        raise _BadInput("缺少 path：要一个视频文件路径")
    # 越界先判：「相对路径」一律锚在仓库根（_norm_path），所以只有显式的绝对路径
    # 才可能跑出仓库；出了仓库还要求它「原样就是绝对路径且真的存在」，否则 400。
    # （先判越界再看存在性：../../ 这种穿越不能因为文件恰好不存在就降级成 404。）
    if not _inside_repo(p):
        if not (_raw_is_absolute(raw) and p.is_file()):
            raise _BadInput("路径越界：%s 既不在仓库（%s）里，也不是一个存在的绝对路径"
                            % (p, _REPO_ROOT), 400)
    if not p.is_file():
        raise _BadInput("文件不存在：%s" % p, 404)
    try:
        size = int(p.stat().st_size)
    except OSError as e:
        raise _BadInput("读不了文件大小：%s（%s）" % (p, e), 500)

    headers = {"accept-ranges": "bytes",
               "content-type": _video_ctype(p),
               "cache-control": "no-store"}
    try:
        headers["last-modified"] = _httpdate(p.stat().st_mtime, usegmt=True)
    except Exception:
        pass

    start, end = _parse_range(request.headers.get("range"), size)
    if start == "bad":
        body = {"ok": False, "error": "Range 越界：bytes=%s，文件只有 %d 字节"
                                     % (str(request.headers.get("range"))[6:], size),
                "detail": "Range 越界"}
        return JSONResponse(status_code=416, content=body,
                            headers={"accept-ranges": "bytes",
                                     "content-range": "bytes */%d" % size})

    if start is None:
        status, start, end = 200, 0, max(0, size - 1)
    else:
        status = 206
        headers["content-range"] = "bytes %d-%d/%d" % (start, end, size)
    length = max(0, end - start + 1)
    headers["content-length"] = str(length)
    return StreamingResponse(_file_chunks(p, start, length),
                             status_code=status, headers=headers)


# ------------------------------------------------------------# 单帧预览：一步式 MergeMaskedFace（不碰 <视频>_cache/，不调 stage 函数）
#
# 这里以前放过另一条路：预览也走 ⑤ 合成 —— 先读 <视频>_cache/pred|mask（有就复用），
# 没有就现场算 dfm + XSeg，再调 _merge_compose.compose_frame。它和导出的字节更接近，
# 但代价是：① 预览开始依赖 cache 存在与否（有 cache 和没 cache 出图不一样，还会读到
# 别人留下的半成品）；② 整帧 float32 的色彩迁移 + compose 比一步式慢一倍
# （1080×1920 实测 10 帧端到端：改前 497ms/帧 → 改后 230ms/帧，见
# app/tools/accept/preview_fast_accept.py）；③ 一次 UI 预览把六阶段的代码路径也拉进了请求
# 线程。整条已删掉，连同「从 <视频>_cache/frames 读帧」那一小段。
#
# 现在只有一条路：取帧 → worker → MergeMaskedFace（dfm 推理 + 遮罩 + 融合一次做完）
# → 内存里的 JPEG/PNG 原路返回。**不读 cache、不写 cache、不创建 cache 目录，
# 也不调用任何 _stage_* 函数**（本文件里 _stage_* 只服务六阶段导出）。
# worker 自己的临时目录（%TEMP%/dfl41_merge_preview，只放 worker 脚本）不在素材目录
# 或 cache 目录树里。
#
# 与导出的关系：算法等价、代码路径不同。
#   · 预览：merger.MergeMaskedFace —— 一张脸从头到尾一次算完；4.0 在 seamless 之后
#     还有一段「把结果 warp 回脸空间再做一次色彩迁移」，预览保留。
#   · 导出：六阶段（stage6_accept.py 验的那条）—— pred/mask 先落盘，合成是纯 CPU 的
#     _merge_compose.compose_frame，逐段对照 4.0 的 _phase_merge_worker。
# 所以同一帧「预览」和「导出」的像素差是**允许存在**的，实测（stage6_accept 的 d 段）：
#   · 预览 PNG vs 导出帧 PNG（同一帧、都无损）→ mean 0.02~0.03 / max 20~33：两条路的
#     融合只差边缘那几级灰度；
#   · 成片（crf16 yuv420p + 预览那侧 JPEG）vs 预览 → mean ≈ 3.6~4：这才是那 ~3.7 的来源，
#     是编码往返，不是算法。
# 不要为了「逐像素相同」再把预览接回 compose —— 那等于用一倍的耗时换一个用户看不出来
# 的差值，还顺手把预览绑在 cache 上。
# ------------------------------------------------------------



@router.post("/api/merge/preview")
@_api
def merge_preview(body: dict):
    """t 时刻那一帧 → 检测 → 换脸 → 遮罩 → 合成 → JPEG（data: URL）。

    body 多认一个可选的 models: [{name, dfm, faces}]（见 _models_of）：给了就是
    多模型，一帧里检测到的每张脸按分配换，没分配的用第 0 个；不给就是老的单模型
    路径，行为一个字都不改。

    链路固定为一步式：帧现场从视频取 → 检测 → 68 点 → worker 里 MergeMaskedFace
    （dfm 推理 + 遮罩 + 融合一次做完）→ 内存里的 JPEG。**不读也不写 <视频>_cache/**，
    不调用六阶段的任何 stage 函数；cache 在不在、里面有没有半成品都不影响这一枪。
    """
    t_start = time.time()
    body = _as_dict(body)
    # 预览是高频入口（播放/拖动时每帧一次）：内容没变就不重复打，免得淹掉别的日志
    _eff.log_request("/api/merge/preview", body, dedupe=True,
                     ignore=("t", "frame", "face", "idx", "n", "step"))

    video = _norm_path(body.get("video") or body.get("path"))
    if video is None:
        raise _BadInput("缺少 video：要一个视频文件路径")
    if not video.is_file():
        raise _BadInput("视频不存在：%s" % video, 404)
    if video.suffix.lower() not in _VIDEO_EXT:
        raise _BadInput("不是视频文件：%s（只认 %s）"
                        % (video.suffix or "无扩展名", " ".join(sorted(_VIDEO_EXT))))

    t = _float_val("t", body.get("t"), 0.0, None, 0.0)
    dur = _video_duration(video)
    if dur > 0:
        if t > dur + 0.5:
            raise _BadInput("t=%.2fs 超出了视频长度（%.2fs）" % (t, dur))
        t = min(t, max(0.0, dur - 0.05))

    models = _models_of(body)                 # None = 老的单模型路径
    if models:
        dfm, model_name = Path(models[0]["dfm"]), models[0]["name"]
    else:
        dfm, model_name = _dfm_of(body)
    settings = _settings_val(body.get("settings"))
    cfg = _merge_settings_to_config(settings)
    detector = _detector_val(body.get("detector"))
    thresh = _float_val("thresh", body.get("thresh"), 0.0, 1.0, 0.5)
    xseg = _xseg_of(body)

    # ---- 生效配置（预览：配置没变就不重复打，否则播放时每帧一行会淹掉日志）----
    _pcfg = {"settings": settings, "config": cfg,
             "detector": detector, "det_align": detector,
             "mask_src": _mask_src(settings, settings.get("mask_mode")),
             "dfm": str(dfm), "model": model_name, "single_models": models or [],
             "library": str(body.get("faces_project") or ""),
             "use_library": True, "chain": False, "fused": False,
             "range": {}, "workers": 0, "batch": 0,
             "jpeg_quality": _PREVIEW_QUALITY}
    eff = _eff_publish("/api/merge/preview", "preview", body, _pcfg, dedupe=True)
    with _PREVIEW_EFF_LOCK:
        _PREVIEW_EFF.clear()
        _PREVIEW_EFF.update(eff)

    # ---- 显存闸门：模型装不下就**明确拒绝**，别让 worker 半死不活 ----
    # 以前这里是「起 worker → 加载失败 → 报一句 worker 的 stderr 尾巴」，用户看到的
    # 是一条 CUDA/ORT 的英文异常；现在先算账：装不下就 400 + 「需要 X GB / 可用 Y GB /
    # 建议」，开了 DFL_ALLOW_CPU_FALLBACK 就走 CPU（worker 会收到 DFL_FORCE_CPU_EP=1）。
    plan = _preview_guard([m["dfm"] for m in (models or []) if m.get("dfm")] or [str(dfm)])

    note = ""
    if int(cfg.get("mask_mode") or 0) >= 6 and not xseg:
        note = ("mask_mode=%s 要 XSeg 模型，但找不到 —— 引擎会退化成整块矩形贴脸"
                % cfg.get("mask_mode"))
    if plan.get("cpu"):
        note = (note + "；" if note else "") + ("★ 当前在 CPU 上跑 ★（显存不够 + "
                                                "DFL_ALLOW_CPU_FALLBACK=1，预计慢约 17 倍）")
    if settings.get("face_margin") not in (None, ""):
        note = (note + "；" if note else "") + "face_margin 只影响抽帧对齐，预览不读它"

    # 同一个 t 连着来（前端 120ms 防抖会把同一帧请求两遍）：整份响应直接复用 ——
    # 不取帧、不检测、不换脸。只留 1 条，键里带着文件身份 / t / 模型 / 遮罩 / 参数，
    # 任何一项变了都不会串味。
    try:
        # 人脸超分滑块也算「参数」：只改它、别的都不动时，缓存必须判为不同的一单，
        # 否则滑到 512 预览还是上一张（用户报的「看不出增强」有一半是这条）。
        try:
            from .knobs import _sr_level as _lv
            _sr_lv = _lv((body or {}).get("sr"))
        except Exception:                                   # noqa: BLE001
            _sr_lv = 0
        hit_sig = json.dumps([list(_grab_key(video)), round(float(t), 3),
                              str(dfm), str(xseg), detector, float(thresh),
                              cfg, settings, int(_sr_lv),
                              [[m["name"], m["dfm"], list(m["faces"])]
                               for m in (models or [])]], sort_keys=True, default=str)
    except Exception:
        hit_sig = ""
    if hit_sig:
        with _PREVIEW_HIT_LOCK:
            hit = _PREVIEW_HIT.get(hit_sig)
        if hit is not None:
            hit_res = dict(hit)
            tm = dict(hit_res.get("timings") or {})
            tm["grab_ms"] = 0.0            # 这一帧真没再取：grab 的活全省了
            tm["cache_hit"] = 1
            hit_res["timings"] = tm
            hit_res["ms"] = int(round((time.time() - t_start) * 1000.0))
            hit_res["vram_plan"] = dict(plan)
            hit_res["on_cpu"] = bool(plan.get("cpu"))
            try:
                hit_res["worker"] = _WORKER.status()
            except Exception:
                pass
            return hit_res

    stop = threading.Event()
    state = {"stage": "start"}
    holder = {}
    done = threading.Event()

    # 帧一律现场从视频里取：这里**不再**看 <视频>_cache/frames 有没有 —— 预览不能
    # 因为「有没有 cache」出两张不同的图，更不能依赖别人留下的中间产物。
    def _run():
        try:
            job = {"video": video, "t": float(t), "dfm": dfm,
                   "models": models,
                   "xseg": xseg, "config": cfg,
                   "detector": detector, "thresh": float(thresh),
                   "stop": stop, "state": state,
                   "no_swap": _face_ids_val(
                       "no_swap_faces", body.get("no_swap_faces"))}
            holder["r"] = _preview_job(job)
        except _BadInput as e:
            holder["e"] = e
        except Exception as e:                  # noqa: BLE001
            traceback.print_exc()
            holder["e"] = _BadInput("预览失败：%s"
                                    % (str(e)[:200] or e.__class__.__name__), 500)
        finally:
            done.set()

    threading.Thread(target=_run, daemon=True, name="merge-preview").start()

    # 15s 是「干活」的上限；worker 冷启动/加载模型的那一段单独给它时间，
    # 但总时长封在 work + load 以内，不会无限等下去。
    soft = t_start + _PREVIEW_WORK_TIMEOUT
    ceiling = t_start + _PREVIEW_WORK_TIMEOUT + _PREVIEW_LOAD_TIMEOUT + _PREVIEW_BOOT_TIMEOUT
    grace_used = False
    while not done.is_set():
        now = time.time()
        if now >= soft:
            stage = str(state.get("stage") or "")
            if stage in ("boot", "loading_model", "queued") and now < ceiling and not grace_used:
                grace_used = True
                soft = min(ceiling, now + _PREVIEW_LOAD_TIMEOUT)
                continue
            stop.set()
            if stage != "queued":
                # 杀进程才是真的取消（ONNX 推理从外部打断不了）。
                # queued 的时候 worker 正被另一个请求用着，不能连坐。
                _WORKER.close("预览超时")
            raise _BadInput("预览超时：单帧上限 %.0fs（t=%.2fs，阶段 %s）—— 已终止这次任务"
                            % (_PREVIEW_WORK_TIMEOUT, t, stage or "?"), 504)
        done.wait(0.1)

    if "e" in holder:
        raise holder["e"]
    res = holder.get("r") or {}
    jpg = res.get("jpg") or b""
    if not jpg:
        raise _BadInput("预览没有产出图", 500)

    resp = _ok(png="data:image/jpeg;base64," + _b64.b64encode(jpg).decode("ascii"),
                ms=int(round((time.time() - t_start) * 1000.0)),
                box=res.get("box"), has_face=bool(res.get("has_face")),
                no_faces=bool(res.get("no_faces")),
                swapped=bool(res.get("swapped")),
                video=str(video), t=round(float(t), 3),
                duration=round(float(dur), 3) if dur else None,
                model=model_name, dfm=str(dfm),
                model_size=res.get("model_size"),
                mask_mode=int(cfg.get("mask_mode") or 0),
                mode=cfg.get("mode"), color_transfer=cfg.get("color_transfer_mode"),
                erode_mask=cfg.get("erode_mask_modifier"),
                blur_mask=cfg.get("blur_mask_modifier"),
                xseg=xseg, detector=detector,
                settings=settings, timings=res.get("timings") or {},
                worker=_WORKER.status(), note=res.get("note") or note,
                vram_plan=dict(plan), on_cpu=bool(plan.get("cpu")),
                eff=eff, eff_warnings=list((eff or {}).get("warnings") or []))
    if models:                          # 多模型：只增不改，把「谁被用了」摊开给前端核对
        resp["models"] = [{"name": m["name"], "dfm": m["dfm"],
                           "faces": list(m["faces"])} for m in models]
        resp["models_used"] = list(res.get("models_used") or [])
        resp["faces"] = list(res.get("faces") or [])
        resp["swaps"] = int(res.get("swaps") or 0)
        if res.get("skipped"):
            resp["skipped"] = list(res.get("skipped") or [])
    if hit_sig:                        # 只留最新一帧：前端再问同一个 t 就直接给这份
        with _PREVIEW_HIT_LOCK:
            _PREVIEW_HIT.clear()
            _PREVIEW_HIT[hit_sig] = resp
    return resp


@router.get("/api/merge/preview/status")
@_api
def merge_preview_status():
    """预览 worker 的现状（排障用：起没起、加载过哪些模型、跑在哪个 provider 上）。

    另带一份 vram_plan：这次加载是按什么显存预算放行的、有没有降档 / 是不是在 CPU 上跑。
    """
    st = _WORKER.status()
    plan = dict(st.get("vram_plan") or _vram_last("预览") or {})
    with _PREVIEW_EFF_LOCK:
        eff = dict(_PREVIEW_EFF) if _PREVIEW_EFF else None
    return _ok(**dict(st, vram_plan=plan,
                      on_cpu=bool(st.get("on_cpu") or plan.get("cpu")),
                      eff=eff,
                      eff_warnings=list((eff or {}).get("warnings") or []),
                      warnings=[str(w.get("text") or w)
                                for w in ((eff or {}).get("warnings") or [])]))

@router.post("/api/merge/preview-frame")
@_api
def merge_preview_frame(body: dict):
    """一张**静态帧文件** -> 检测 -> 换脸 -> 遮罩 -> 合成 -> 图（data: URL）。

    和 /api/merge/preview 只差一点：帧不是按时刻从视频里取的，而是直接读一个
    图片文件（项目模式就是 <项目>/000000.jpg）。检测 / 68 点 / 换脸 / 遮罩 / 融合
    仍然是 _preview_job 那一份 —— 一步式 MergeMaskedFace，**不读 <视频>_cache/**。
    六阶段导出走的是另一条代码路径（pred/mask 落盘 + 纯 CPU compose_frame），
    算法等价但像素差允许存在（无损对拍 mean 0.02~0.03；成片 vs 预览 ≈ 3.6~4 是编码
    往返），别为了"同一份字节"把预览接回 compose。
    """
    t_start = time.time()
    body = _as_dict(body)
    fp = _norm_path(body.get("frame_file") or body.get("image") or body.get("path"))
    if fp is None:
        raise _BadInput("缺少 frame_file：要一张帧图（jpg/png）的路径")
    if not fp.is_file():
        raise _BadInput("帧图不存在：%s" % fp, 404)
    if fp.suffix.lower() not in _IMAGE_EXT:
        raise _BadInput("不是图片：%s（只认 %s）"
                        % (fp.suffix or "无扩展名", " ".join(sorted(_IMAGE_EXT))))

    models = _models_of(body)                 # None = 老的单模型路径
    if models:
        dfm, model_name = Path(models[0]["dfm"]), models[0]["name"]
    else:
        dfm, model_name = _dfm_of(body)
    settings = _settings_val(body.get("settings"))
    cfg = _merge_settings_to_config(settings)
    detector = _detector_val(body.get("detector"))
    thresh = _float_val("thresh", body.get("thresh"), 0.0, 1.0, 0.5)
    xseg = _xseg_of(body)
    fmt = "png" if str(body.get("fmt") or "jpg").strip().lower() == "png" else "jpg"
    quality = _int_val("quality", body.get("quality"), 1, 100, _PREVIEW_QUALITY)

    # 显存闸门（和 /preview 同一口径）：装不下就 400 + 数字，别去起 worker 试
    plan = _preview_guard([m["dfm"] for m in (models or []) if m.get("dfm")] or [str(dfm)])

    frame = _app_mod._imread_any(fp)
    if frame is None:
        raise _BadInput("这张帧图解不开：%s" % fp.name, 422)
    stop = threading.Event()
    state = {"stage": "start"}
    res = _preview_job({"frame": frame, "models": models, "dfm": dfm,
                        "xseg": xseg, "config": cfg, "detector": detector,
                        "thresh": float(thresh), "stop": stop, "state": state,
                        "fmt": fmt, "quality": int(quality),
                        # 帧就是这张图本身；不再认 frame_stem / cache_dir —— 预览不读
                        # <视频>_cache/ 里的 pred/mask/68 点（那是导出六阶段的中间产物）。
                        "no_swap": _face_ids_val("no_swap_faces",
                                                 body.get("no_swap_faces")),
                        "send_lossless": (fmt == "png")})
    img = res.get("img") or res.get("jpg") or b""
    if not img:
        raise _BadInput("这一帧没有产出图", 500)
    mime = "image/png" if fmt == "png" else "image/jpeg"
    resp = _ok(png="data:%s;base64,%s" % (mime, _b64.b64encode(img).decode("ascii")),
               ms=int(round((time.time() - t_start) * 1000.0)),
               frame_file=str(fp), fmt=fmt, bytes=len(img),
               box=res.get("box"), has_face=bool(res.get("has_face")),
               swapped=bool(res.get("swapped")), note=res.get("note") or "",
               timings=res.get("timings") or {},
               model=model_name, dfm=str(dfm), xseg=xseg, detector=detector,
               settings=settings, vram_plan=dict(plan), on_cpu=bool(plan.get("cpu")))
    if models:
        resp["models"] = [{"name": m["name"], "dfm": m["dfm"],
                           "faces": list(m["faces"])} for m in models]
        resp["faces"] = list(res.get("faces") or [])
        resp["swaps"] = int(res.get("swaps") or 0)
    return resp


@router.post("/api/merge/preview/warm")
@_api
def merge_preview_warm(body: dict = None):
    """后台把 worker / 检测器 / DFM 备好（不取帧、不合成），立刻返回。

    body: {"model": "<模型名>"}（可省；省了用 workspace/model 下第一个 .dfm）
          也可以给 "models": [{name, dfm, faces}] 一次热一组（worker 里最多常驻 3 个，
          超了 LRU 淘汰；faces 在预热里不参与——预热只关心"哪几个模型"）。
    已经热了就 {"ok":true,"warming":false,"ms":…,"cold":false}；要干活就
    {"ok":true,"warming":true,...}，进度看 GET /api/merge/preview/warm/status。
    """
    t_req = time.time()
    body = _as_dict(body)
    models = _models_of(body)                  # None = 老的单模型路径
    if models:
        dfms = [m["dfm"] for m in models]
        names = [m["name"] for m in models]
    else:
        dfm, model_name = _dfm_of(body)        # 模型不存在：这里就 404，不进线程
        dfms, names = [str(dfm)], [str(model_name or "")]
    det_name = _detector_val(body.get("detector"))
    # 预热也要过显存闸门：装不下就在这里 400（带数字），别让后台线程去撞 CUDA
    plan = _preview_guard(dfms)

    hot, why = _warm_note_hot(dfms, names, det_name)
    ms = int(round((time.time() - t_req) * 1000.0))
    if hot:
        return _ok(warming=False, ready=True, ms=ms, cold=False,
                   model=names[0], dfm=dfms[0], detector=det_name,
                   worker_pid=_worker_pid(),
                   models=[{"name": n, "dfm": d} for n, d in zip(names, dfms)],
                   models_used=list(names), max_models=_WORKER_MAX_MODELS,
                   vram_plan=dict(plan), on_cpu=bool(plan.get("cpu")))

    started = _warm_kick(dfms, names, det_name, t_req)
    return _ok(warming=True, ready=False, ms=int(round((time.time() - t_req) * 1000.0)),
               cold=(_worker_pid() is None), model=names[0], dfm=dfms[0],
               detector=det_name, worker_pid=_worker_pid(),
               models=[{"name": n, "dfm": d} for n, d in zip(names, dfms)],
               models_used=[], max_models=_WORKER_MAX_MODELS,
               started=bool(started), reason=why)


@router.get("/api/merge/preview/warm/status")
@_api
def merge_preview_warm_status():
    """预热状态：{ready, model, ms, warming, cold, error, worker_pid, …}。"""
    return _ok(**_warm_status())


@router.post("/api/merge/export-project")
@_api
def merge_export_project(body: dict):
    """标准 DFL 项目 -> 换脸后的整帧图（merged/）+ 按原 fps/音轨编码成片。

    入参：
      project_dir   必填，<视频名>/ 目录（帧图 + aligned/）
      frames_dir    可选，帧目录（默认就是 project_dir）
      aligned_dir   可选，默认 <project_dir>/aligned
      merged_dir    可选，默认 <project_dir>/merged
      video         可选，原视频（fps / 分辨率 / 音轨都从它来）
      out_video     可选，默认 <项目目录>.mp4
      models / model / dfm / settings / xseg_onnx / detector / thresh
                    —— 和 /api/merge/export、/api/merge/preview 同一套协议
      force         可选，true = 忽略 merged 里已有的帧，全部重做
      quality       可选，jpg 帧的编码质量（默认 92，和单帧预览一致）
      workers       可选，并行 worker 进程数。**默认按模型体量**（不是一律 2）：
                    大模型（≥1GB 权重，例如 384 那份 1.53GB）→ 1；小模型（≤600MB，
                    例如 416 那份 0.5GB）→ 2；中间档先要 2，再由显存预算按「free +
                    一份会话的实测峰值」砍（见 _proj_workers_default）。每个 worker
                    是一份独立的 dfm session，而那份会话**真跑起来**的峰值是 3.4GB
                    （384）/ 2.2GB（416）—— 不是文件体积那 1.5GB / 0.5GB。
                    实际值回在 status.workers 里。
      batch         可选，一条请求里带几帧 = worker 内部同时跑几帧。默认 **4 // workers**
                    （1 worker → 4 线程、2 worker → 2 线程），同时在飞的帧数因此锁在
                    4 左右 —— 那是实测的甜点（8 帧在飞就换页）。384 模型实测：1×2 8.73
                    帧/秒、1×4 10.87、1×8 4.44，所以单 worker 时线程必须给到 4。
                    workers×batch = 同时在飞的帧数，实际值回在 status.batch 里。

      显存不够时的退档顺序（用户定的）：**先减 worker → 再降线程（batch）→ 最后才拒绝**
      （拒绝是 HTTP 400 + need_mb/avail_mb/suggest 等结构化字段）。
    """
    body = _as_dict(body)
    _eff.log_request("/api/merge/export-project", body)
    project = _norm_path(body.get("project_dir") or body.get("project"))
    if project is None:
        raise _BadInput("缺少 project_dir：要一个 DFL 项目目录（<视频名>/）")
    if not project.exists():
        raise _BadInput("项目目录不存在：%s" % project, 404)
    if not project.is_dir():
        raise _BadInput("project_dir 不是目录：%s" % project)

    frames_dir = _norm_path(body.get("frames_dir")) or project
    if not frames_dir.is_dir():
        raise _BadInput("帧目录不存在：%s" % frames_dir, 404)
    if not _scan_first(frames_dir, _PROJ_FRAME_EXT, 1):
        raise _BadInput("帧目录里没有帧图（.jpg/.png）：%s —— 先抽帧" % frames_dir)

    aligned = _norm_path(body.get("aligned_dir")) or (project / "aligned")
    if not aligned.is_dir():
        raise _BadInput("找不到对齐集 aligned/：%s —— 先跑一次人脸对齐" % aligned, 404)
    if not _scan_first(aligned, {".jpg", ".jpeg"}, 1):
        raise _BadInput("对齐集是空的：%s" % aligned)
    merged = _norm_path(body.get("merged_dir")) or (project / "merged")

    # ---- 模型：和 /export、/preview 完全同一套协议 ----
    models = _models_of(body)
    if models:
        dfm, model_name = Path(models[0]["dfm"]), models[0]["name"]
    else:
        dfm, model_name = _dfm_of(body)
    settings = _settings_val(body.get("settings"))
    cfg_main = _merge_settings_to_config(settings)
    detector = _detector_val(body.get("detector"))
    thresh = _float_val("thresh", body.get("thresh"), 0.0, 1.0, 0.5)
    xseg = _xseg_of(body)
    quality = _int_val("quality", body.get("quality"), 1, 100, _PREVIEW_QUALITY)
    force = bool(body.get("force"))
    # 并行度：N 个 worker 进程，各持自己的 dfm session（原版 stage4 的同一个语义）；
    # batch = 一条请求里带几帧（= worker 内部同时跑几帧）。
    # 默认值**按模型体量**：大模型（≥1GB 权重）1 个、小模型（≤600MB）2 个（见
    # _proj_workers_default）；线程数 = 4 // workers（见 _proj_batch_default）。
    plan_paths = [m["dfm"] for m in (models or []) if m.get("dfm")] or [str(dfm)]
    model_mb = _vram_models_mb(plan_paths)
    workers = int(_int_val("workers", body.get("workers"), 1, 16,
                           _proj_workers_default(model_mb)))
    batch = int(_int_val("batch", body.get("batch"), 1, _PREVIEW_MAX_BATCH,
                         _proj_batch_default(workers)))
    # ---- 自适应档位：**有实测就按实测选，没有就保守起、跑完把实测写回去** ----
    # 静态规则（按权重 ≥1GB → 1 worker）只是「还没量过」时的兜底。同一张 8GB 卡，
    # 「4 帧摊在 1 份会话还是 2 份上」实测是 77% vs 95%（换页）—— 这件事跟权重只是
    # 相关、不是因果（还看 arena 怎么涨、帧尺寸、桌面/浏览器占了几个 GB）。所以：
    #   · 本机量过这份模型 → 用 <workspace>/_proj_bench.json 里的实测挑（_projplan.choose）
    #   · 没量过 → 按保守档起，_proj_run 开跑前用 8~12 帧现场标定，选完再正式跑
    #   · 请求体 / 环境变量显式给了 workers/batch → 一句话都不改（尊重手动）
    adapt = {"src": "off", "explicit": False, "note": ""}
    explicit = bool(str(body.get("workers") or "").strip()
                    or str(body.get("batch") or "").strip()
                    or str(os.environ.get("DFL_PROJ_WORKERS") or "").strip()
                    or str(os.environ.get("DFL_PROJ_BATCH") or "").strip())
    adapt["explicit"] = bool(explicit)
    adapt_key = ""
    if not explicit and _projplan.enabled():
        adapt_key = _projplan.model_key(plan_paths)
        _row = _projplan.get_row(WORKSPACE, adapt_key)
        if _row.get("rows"):
            try:
                _f0, _t0, _u0 = _vram_free_smi(fresh=True)
            except Exception:                       # noqa: BLE001
                _t0 = _u0 = None
            _pick = _projplan.choose(_row["rows"], _t0, _u0)
            if _pick.get("pick"):
                workers = int(_pick["pick"]["workers"])
                batch = int(_pick["pick"]["batch"])
                adapt = {"src": "cache", "key": adapt_key, "rows": _pick["rows"],
                         "pick": _pick["pick"], "rejected": _pick.get("rejected"),
                         "cap": _pick.get("cap"),
                         "note": _projplan.note(_row["rows"], _pick)}
            else:
                _cons = _projplan.conservative(model_mb)
                workers, batch = int(_cons["workers"]), int(_cons["batch"])
                adapt = {"src": "cache-none", "key": adapt_key, "rows": _pick.get("rows"),
                         "pick": None,
                         "note": _projplan.note(_row["rows"], _pick) + " → " + _cons["why"]}
        else:
            _cons = _projplan.conservative(model_mb)
            workers, batch = int(_cons["workers"]), int(_cons["batch"])
            adapt = {"src": "bench", "key": adapt_key, "rows": None, "pick": None,
                     "note": "%s（本次导出前现场标定 8~12 帧）" % _cons["why"]}
    elif explicit:
        adapt["note"] = "手动指定 workers/batch → 不改（只按显存预算削减）"
    # ---- 显存预算：能开几个 worker 就开几个，装不下就按 worker → 线程 → 拒绝 的顺序退 ----
    # 每多一个 worker 就多一份「会话 + CUDA context」，而且那份会话**真跑起来还要再长一块**
    # （384 实测 ≈3.4GB/份，不是「权重 + context」那 2.08GB）—— 所以这里喂进去的是
    # _vram_sess_peak_mb 的实测峰值（没量过就按标定值），不是文件体积推算。
    # 以前是「先起 4 个进程、看谁崩」，崩了再退 —— 那一步本身就会把卡顶到 OOM。
    ses = _vram_sess_peak_mb(plan_paths)
    minfo = _dfm_info(plan_paths[0]) if plan_paths else {}
    # 预算削减前用户/自适应想要的 worker 数：砍了就一定要在 job.log 里说一句
    # （「降级并提示」是这条链的契约；只降不提示 = 用户以为自己在跑 2 份会话）
    workers_req = int(workers)
    plan = _vram_guard(plan_paths, {"mode": "procs", "max_procs": int(workers),
                                    "batch": int(batch), "aux": True,
                                    "sess_mb": float(ses["sess_mb"]),
                                    "sess_src": str(ses["src"]),
                                    "model_info": minfo,
                                    "reclaim_first": True}, label="项目导出")
    if int(plan.get("procs") or 1) < int(workers):
        workers = max(1, int(plan.get("procs") or 1))
    if int(plan.get("batch") or 1) < int(batch):
        batch = max(1, int(plan.get("batch") or 1))
    # ---- 实测优先：缓存的实测档被预算砍掉时，以**实测**为准（不写死、也不拿估算压实测）----
    # 预算那套是「一份会话的峰值 + 批台阶」的估算，会高估：416 今天估 5781MB / 实测整卡
    # 6503MB（程序 4906MB），于是 2×2 被砍成 1 个 worker —— 而同一台机器真的量到过
    # 79.8% / 14.8 帧/秒。既然实测在换页线以内，就没理由为了估算退档。
    _pick = adapt.get("pick") or None
    if (_pick and str(adapt.get("src")) == "cache"
            and int(_pick.get("workers") or 1) > int(plan.get("procs") or 1)
            and float(_pick.get("pct") or 0.0) <= 100.0 * _projplan.HARD):
        print("[project] 实测档 %s 整卡 %.0f%% ≤ 换页线 %.0f%% → 覆盖预算削减（%d → %d worker）"
              % (_pick["label"], float(_pick["pct"]), 100.0 * _projplan.HARD,
                 int(plan.get("procs") or 1), int(_pick["workers"])), flush=True)
        workers, batch = int(_pick["workers"]), int(_pick["batch"])
        plan.update({"procs": int(workers), "batch": int(batch),
                     "requested_procs": int(workers), "requested_batch": int(batch),
                     "adaptive_override": True, "downgraded": False})
    # 这一行是排障第一现场（server.log + job.log 各一份）：模型多大 / 可用多少 /
    # 计划几个 worker × 几线程 / 减了没有、为什么减。
    plan["worker_note"] = _proj_plan_note(model_mb, plan)
    print(plan["worker_note"], flush=True)
    # 档位是**怎么来的**：缓存的实测 / 现场标定 / 保守默认 / 手动指定 —— 和上面那行
    # 预算说明并列写进日志与 status（"为什么不是另一个档"要能当场回答）。
    plan["adaptive"] = dict(adapt)
    if adapt.get("note"):
        print("[project] " + str(adapt["note"]), flush=True)
    # ---- 起 worker **之前**先把本进程的死重放掉 ----
    # 项目导出的 worker 是独立进程（各持自己的 dfm session），所以本进程里缓存的
    # dfm/XSeg 会话、以及常驻预览 worker 里那份模型，在这一单里都是**一点用没有**的
    # 死重 —— 而它们正好吃掉「峰值 ≤85%」的那 15% 余量：实测同一单（384 · 1×4）
    # 卡干净的峰值是 6140MB（75.3%），先跑过阶段链、父进程还留着 dfm/XSeg 时会顶到
    # 7488MB（91.9%）。没有别的任务在跑时才动（有任务在跑时放掉会话等于锯它的腿）。
    plan["released"] = {}
    if not _merge_job.get("running") and not _STAGE_JOB.get("running"):
        try:
            rel = _vram_release_inproc("项目导出开始前")
            rec = _vram_reclaim("项目导出开始前")
            plan["released"] = {"inproc": list((rel or {}).get("released") or []),
                                "preview_mb": round(float((rec or {}).get("freed_mb") or 0.0), 1)}
            if plan["released"]["inproc"] or plan["released"]["preview_mb"]:
                print("[project] 起 worker 前先放掉死重：本进程会话 %s · 预览模型 %.0f MB"
                      % ("、".join(plan["released"]["inproc"]) or "无",
                         plan["released"]["preview_mb"]), flush=True)
        except Exception as e:                          # noqa: BLE001
            print("[project] 起 worker 前放死重失败（不影响这一单）：%s: %s"
                  % (e.__class__.__name__, str(e)[:160]), flush=True)

    # ---- 原视频 / fps / 分辨率：读不到就报错，不猜 ----
    video, why = _proj_source_video(body.get("video") or body.get("source_video"),
                                    project, frames_dir)
    if video is None:
        raise _BadInput("%s（项目：%s）" % (why, project), 422)
    try:
        v_frames, v_fps, v_w, v_h = _app_mod._video_probe(video)
    except Exception as e:                          # noqa: BLE001
        raise _BadInput("原视频解析不出来（ffmpeg/imageio）：%s（%s）"
                        % (video.name, str(e)[:160]), 422)
    fps_body = _float_val("fps", body.get("fps"), 1.0, 240.0, None)
    if fps_body is not None:
        fps = float(fps_body)
    elif v_fps and float(v_fps) > 0:
        fps = float(v_fps)
    else:
        raise _BadInput("读不出原视频的 fps：%s —— 不猜，先确认这个文件能正常解码"
                        % video, 422)

    # ---- 输出 ----
    out_raw = str(body.get("out_video") or body.get("out") or "").strip()
    out_video = _norm_path(out_raw) if out_raw else Path(str(project) + ".mp4")
    if out_video.suffix.lower() not in _VIDEO_EXT:
        out_video = out_video.with_suffix(".mp4")
    if out_video.is_dir():
        raise _BadInput("输出路径是个目录：%s（要 .mp4 文件）" % out_video)
    if _same_path(out_video, video):
        if out_raw:                     # 用户明确指到原片上：拒绝，别把素材覆盖了
            raise _BadInput("输出路径不能是原视频本身：%s" % out_video)
        # 默认名 <项目目录>.mp4 正好和原视频同名（把视频放在项目目录旁边时就是这样）：
        # 换个名字，绝不覆盖素材
        out_video = Path(str(project) + "_merged.mp4")
    try:
        out_video.parent.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        raise _BadInput("输出目录建不出来：%s（%s）" % (out_video.parent, e))

    cfg = {"project": str(project), "frames_dir": str(frames_dir),
           "aligned": str(aligned), "merged": str(merged),
           "video": str(video), "out_video": str(out_video),
           "fps": float(fps), "video_frames": int(v_frames or 0),
           "width": int(v_w or 0), "height": int(v_h or 0),
           "models": models, "dfm": str(dfm), "model": str(model_name),
           "config": cfg_main, "settings": settings, "xseg": xseg,
           "detector": detector, "thresh": float(thresh), "quality": int(quality),
           "force": force, "workers": int(workers), "batch": int(batch),
           "adaptive": dict(adapt), "vram_plan": dict(plan)}

    with _PROJ_LOCK:
        if _PROJ_JOB.get("running"):
            raise _BadInput("已经在导出项目（%s）—— 等它结束或点取消"
                            % (_PROJ_JOB.get("project") or "?"), 409)
        _PROJ_JOB.update({
            "running": True, "ok": None, "phase": "running", "cancelled": False,
            "want_cancel": False, "error": "", "error_detail": "", "log": [],
            "stage": "准备", "pct": 0, "msg": "启动…", "elapsed": 0.0,
            "project": str(project), "frames_dir": str(frames_dir),
            "aligned": str(aligned), "merged": str(merged),
            "video": str(video), "out_video": str(out_video), "fps": float(fps),
            "width": int(v_w or 0), "height": int(v_h or 0),
            "video_frames": int(v_frames or 0), "total": 0, "done": 0,
            "swapped": 0, "copied": 0, "skipped": 0, "failed": 0,
            "frame_fps": 0.0, "eta_s": None, "started_at": time.time(),
            "finished_at": None, "force": force, "dfm": str(dfm),
            "workers": int(workers), "vram_plan": dict(plan),
            "on_cpu": bool(plan.get("cpu")),
            "model": str(model_name),
            "models": [{"name": m["name"], "dfm": m["dfm"],
                        "faces": list(m["faces"])} for m in (models or [])],
            "settings": settings, "detector": detector, "aligned_faces": 0,
            "merged_frames": 0, "warnings": [], "failed_frames": [], "missing": [],
            "missing_source": [], "encode": {},
            "eff": None, "eff_warnings": [],
        })
    # ---- 生效配置：一行 [eff] + 一致性自检（warnings[] 里同时放纯文本）----
    try:
        eff = _eff_publish("/api/merge/export-project", "project", body, cfg)
        _proj_set(**_eff_status_patch(eff))
    except Exception as e:                                  # noqa: BLE001
        print("[eff] 生成失败（不影响这一单）：%s: %s"
              % (e.__class__.__name__, str(e)[:200]), flush=True)
    # ---- 降级要留痕：预算把 worker 数砍下来时，job.log 里必须有一句 [warn] ----
    # 顺序：先 eff（它会把 warnings 置成生效配置的那几条），再追加降级提示。
    if int(workers) < int(workers_req):
        # 老路（多 worker 子进程）已删除，只剩流水线实现 → 这句降级文案只有一种说法。
        _budget = ""
        if plan.get("avail_mb"):
            _budget = ("（可用 %.2fGB / 需要 %.2fGB）"
                       % (float(plan.get("avail_mb") or 0.0) / 1024.0,
                          float(plan.get("need_mb") or 0.0) / 1024.0))
        _downgrade_msg = ("只有一份 dfm 会话（融合流水线）：请求 %d 个 worker，实际 1 份；"
                          "worker 数只用于预算档位%s" % (int(workers_req), _budget))
        _proj_warn(_downgrade_msg)
        # /status 的 log 只回最后 12 行：开头这一句会被几十帧的进度冲掉。收尾再提一次，
        # 因为「用户会看的那一屏」就是最后 12 行（这也是这个任务要治的病：写了但看不见）。
        cfg["downgrade_warn"] = _downgrade_msg
    threading.Thread(target=_proj_worker, args=(cfg,), daemon=True,
                     name="merge-export-project").start()

    snap = _proj_snap()
    snap.pop("want_cancel", None)
    snap.pop("log", None)
    return _ok(**snap)


@router.get("/api/merge/export-project/status")
@_api
def merge_export_project_status():
    """项目导出的进度。ok = 任务是否成功（和 /export/status 一个口径）。"""
    snap = _proj_snap()
    snap["vram_plan"] = dict(snap.get("vram_plan") or _vram_last("项目导出") or {})
    snap["on_cpu"] = bool(snap["vram_plan"].get("cpu")) or bool(snap.get("on_cpu"))
    ok = snap.get("ok")
    cancelled = bool(snap.get("cancelled"))
    want = bool(snap.get("want_cancel"))
    running = bool(snap.get("running")) and not cancelled
    if running:
        phase = "running"
    elif ok is True:
        phase = "done"
    elif cancelled:
        phase = "cancelled"
    elif ok is False:
        phase = "failed"
    else:
        phase = "idle"
    if phase in ("failed", "cancelled"):
        snap["pct"] = 0
    started = snap.get("started_at")
    try:
        el = round(float((snap.get("finished_at") or time.time()) - started), 1) \
            if started else None
    except Exception:
        el = None
    snap.update({"phase": phase, "running": running, "success": ok,
                 "cancelling": bool(want and running), "elapsed": el,
                 "log": list(snap.get("log") or [])[-12:]})
    snap.pop("want_cancel", None)
    return snap


@router.post("/api/merge/export-project/cancel")
@_api
def merge_export_project_cancel():
    """取消项目导出。幂等：没在跑、重复点都回 ok=true。"""
    with _PROJ_LOCK:
        running = bool(_PROJ_JOB.get("running"))
        _PROJ_JOB["want_cancel"] = True
        if running:
            _PROJ_JOB["msg"] = "正在取消…"
    if not running:
        return _ok(cancelled=False, running=False, already=True,
                   reason="没有正在进行的项目导出")
    return _ok(cancelled=True, running=True, already=False, reason="已请求取消")


# ---------------- API ----------------
@router.get("/api/merge/stage/status")
@_api
def merge_stage_status(cache_dir: str = "", video: str = ""):
    """阶段状态：哪个阶段在跑 / 跑到哪 / 每阶段完成没 / cache 还在不在。

    没在跑的时候可以带 ?video= 或 ?cache_dir=，从磁盘上的 meta.json 把阶段标记读回来
    （刷新页面、重启进程之后照样能看出「停在哪一步」）。
    """
    with _STAGE_LOCK:
        snap = dict(_STAGE_JOB)
        log = list(snap.get("log") or [])
    running = bool(snap.get("running"))
    ok = snap.get("ok")
    cancelled = bool(snap.get("cancelled"))
    if running:
        phase = "running"
    elif cancelled:
        phase = "cancelled"
    elif ok is True:
        phase = "done"
    elif ok is False:
        phase = "failed"
    else:
        phase = "idle"

    cache = str(snap.get("cache_dir") or "")
    meta = snap.get("meta") if isinstance(snap.get("meta"), dict) else {}
    if not running:
        cand = _norm_path(cache_dir) if str(cache_dir or "").strip() else None
        if cand is None and str(video or "").strip():
            v = _norm_path(video)
            if v is not None:
                cand = Path(str(v.parent / (v.stem + "_cache")))
        if cand is None and cache:
            cand = Path(cache)
        if cand is not None:
            disk = _stage_meta_load(cand)
            if disk:
                meta = disk
                cache = str(cand)
    counts = _stage_counts(cache) if cache else {"frames": 0, "faces": 0}
    stages = _stage_stages_meta(meta)
    rng_m = meta.get("range") if isinstance(meta.get("range"), dict) else {}
    out = {
        "ok": (True if running else ok), "success": (None if running else ok),
        "phase": phase, "running": running, "cancelled": cancelled,
        "stage": str(snap.get("stage") or ""),
        "stage_index": int(snap.get("stage_index") if snap.get("stage_index") is not None else -1),
        "stage_pct": int(snap.get("stage_pct") or 0),
        "pct": int(snap.get("pct") or 0),
        "msg": str(snap.get("msg") or ""),
        "error": str(snap.get("error") or ""),
        "error_detail": str(snap.get("error_detail") or ""),
        "video": str(snap.get("video") or video or ""),
        "cache_dir": cache,
        "cache_exists": bool(cache) and Path(cache).exists(),
        "cache_removed": bool(snap.get("cache_removed")),
        "out_video": str(snap.get("out_video") or ""),
        # 这次阶段是按什么显存预算放行的（和被拒时的结构化字段同一份形状）
        "vram_plan": dict(snap.get("vram_plan") or {}),
        # 强制尝试撞上 OOM 的现场（非空 = 确实不够、已回滚；前端回到被拒态）
        "vram_oom": (dict(snap.get("vram_oom") or {}) or None),
        # 帧范围 + **相对进度**：frames_total/frames_done 都是「这一段」里的数
        "range": dict(rng_m),
        "range_key": str(rng_m.get("key") or ""),
        "range_total": int(rng_m.get("total") or 0),
        # 每个阶段各自真正处理过的那一段（encode 的 range 是色彩范围，另给 frame_range）
        "stage_ranges": {k: dict((v or {}).get("range") or {})
                         for k, v in stages.items()
                         if isinstance((v or {}).get("range"), dict)},
        "frames_total": int(snap.get("frames_total") or rng_m.get("total")
                            or counts["frames"] or 0),
        "frames_done": int(snap.get("frames_done") or 0),
        "frame_fps": float(snap.get("frame_fps") or 0.0),
        "eta_s": snap.get("eta_s"),
        "swapped": int(snap.get("swapped") or 0),
        "copied": int(snap.get("copied") or 0),
        "no_face": int(snap.get("no_face") or 0),
        "stages": stages,
        "stages_done": _stage_done_list(meta),
        "next_stage": _stage_next(meta),
        "frame_swapped": bool(meta.get("frame_swapped")),
        "counts": counts,
        "seconds": dict(snap.get("seconds") or {}),
        "stage_seconds": {k: float((v or {}).get("seconds") or 0.0)
                          for k, v in stages.items()},
        "stage_fps": {k: float((v or {}).get("fps") or 0.0) for k, v in stages.items()},
        "profile": snap.get("profile") if isinstance(snap.get("profile"), dict) else None,
        "mask_source": str((stages.get("mask") or {}).get("mask_source") or
                           (_mask_src(meta.get("settings")).get("label") or "")),
        "mask_signature": str((stages.get("mask") or {}).get("signature") or ""),
        # ④ 到底跑在哪个后端上（用户报的「选了 XSegLite 却在跑 onnx」就是靠这几个字段对的）
        "mask_backend": str((stages.get("mask") or {}).get("backend")
                            or (stages.get("mask") or {}).get("engine") or ""),
        "mask_engine": str((stages.get("mask") or {}).get("engine_file") or ""),
        "mask_precision": str((stages.get("mask") or {}).get("engine_precision") or ""),
        "mask_model": str((stages.get("mask") or {}).get("mask_model") or ""),
        "mask_onnx": str((stages.get("mask") or {}).get("mask_onnx") or ""),
        "mask_fallback": str((stages.get("mask") or {}).get("fallback_reason") or ""),
        # ③ 的吞吐与 batch（batch_source = auto/request，ladder 是实测表）
        "pred_batch": int((stages.get("pred") or {}).get("batch") or 0),
        "pred_batch_source": str((stages.get("pred") or {}).get("batch_source") or ""),
        "pred_ladder": list(((stages.get("pred") or {}).get("batch_meta") or {})
                            .get("ladder") or []),
        "pred_profile": dict((stages.get("pred") or {}).get("profile") or {}),
        "pred_faces_per_sec": float((stages.get("pred") or {}).get("faces_per_sec") or 0.0),
        "pred": dict(stages.get("pred") or {}),
        "skipped_why": str((stages.get(str(snap.get("stage") or "")) or {})
                           .get("skipped_why") or ""),
        "skipped": list(snap.get("skipped") or []),
        # ★ 上面那个 skipped 是「跳过的**阶段**」（同名不同物，历史包袱），别混。
        # 坏帧降级（DFL_FUSED_SKIP_BAD=1）用的是下面这三个键：帧数 / 帧号 / 进度页那行提示。
        "skip_bad": bool(snap.get("skip_bad")),
        "skipped_count": int(snap.get("skipped_count") or 0),
        "skipped_frames": list(snap.get("skipped_frames") or []),
        "skip_hint": str(snap.get("skip_hint") or ""),
        "started_at": snap.get("started_at"), "finished_at": snap.get("finished_at"),
        "log": log[-80:],
        "meta": meta,
        # 快速导出（融合流水线）：没有阶段概念，只有帧进度 + 帧预算；**不支持续跑**
        "fused": bool(snap.get("fused")),
        "resumable": (not bool(snap.get("fused"))),
        "frame_budget": snap.get("frame_budget"),
        "resumed": bool(snap.get("resumed")),
        # 生效配置 + 一致性警告（[eff] 行同款；进度页「当前生效」面板与顶部横幅）
        "eff": snap.get("eff"), "eff_warnings": list(snap.get("eff_warnings") or []),
        "warnings": list(snap.get("warnings") or []),
    }
    if running and out["frames_total"] and out["frames_done"]:
        rate = float(out["frame_fps"] or 0.0)
        if rate > 0:
            out["eta_s"] = int((out["frames_total"] - out["frames_done"]) / rate)
    return out


@router.post("/api/merge/stage/cancel")
@_api
def merge_stage_cancel(body: dict = None):
    """取消当前阶段（幂等）。已完成的阶段与产物都保留 —— 下次从停下的地方续跑。"""
    with _STAGE_LOCK:
        running = bool(_STAGE_JOB.get("running"))
        chain = bool(_STAGE_JOB.get("chain"))
        _STAGE_JOB["cancelled"] = True
        stage = str(_STAGE_JOB.get("stage") or "")
    if chain:
        with _merge_job_lock:
            _merge_job["cancelled"] = True
    return _ok(running=running, cancelled=True, stage=stage,
               note=("正在取消 %s" % _STAGE_LABEL.get(stage, stage)) if running else "当前没有阶段在跑")


@router.post("/api/merge/stage/{name}")
@_api
def merge_stage(name: str, body: dict = None):
    """跑一个阶段（frames / faces / merge / encode，也认 extract-frames / swap / export 这些别名）。

    幂等：已完成的阶段再点一次不会重跑（返回 skipped=true），force=true 才重做。
    并发：一次只允许一个阶段在跑（409）。
    显存：起 job 之前先过 _stage_vram_gate，装不下 → HTTP 400（结构化字段，
    不建 job、不加载模型、不开 worker）。dry_run=1 只算账不干活（前端预检用）。
    """
    stage = _stage_name(name)
    if not stage:
        raise _BadInput("未知阶段 %r（只认 %s）"
                        % (name, " / ".join(_STAGE_ORDER)), 404)
    body = _as_dict(body)
    _eff.log_request("/api/merge/stage/%s" % stage, body)
    cfg = _stage_cfg(body, stage)
    with _STAGE_LOCK:
        if _STAGE_JOB.get("running"):
            raise _BadInput("已经有阶段在跑（%s）—— 等它结束或点取消"
                            % _STAGE_LABEL.get(_STAGE_JOB.get("stage"), "?"), 409)
    plan = _stage_precheck(cfg, stage)
    if plan.get("skip"):
        _stage_set(running=False, ok=True, cancelled=False, phase="done",
                   stage=stage, stage_index=_STAGE_ORDER.index(stage), stage_pct=100,
                   pct=100, msg="%s 已完成，跳过" % _STAGE_LABEL.get(stage, stage),
                   video=str(cfg["video"]), cache_dir=str(cfg["cache"]),
                   out_video=str(cfg["out"]), meta=cfg["meta"],
                   range=dict(cfg.get("range") or {}),
                   frames_total=int(cfg.get("range_total") or 0),
                   stages_done=_stage_done_list(cfg["meta"]),
                   counts=_stage_counts(cfg["cache"]),
                   started_at=None, finished_at=time.time())
        return _ok(started=False, skipped=True, reason=plan.get("reason") or "already-done",
                   skipped_why=str(plan.get("skipped_why") or ""),
                   **_stage_brief_meta(cfg))

    # ---- 显存闸门：**skip 之后、建 job 之前** ----
    # 顺序有讲究：先跑前置检查（缺上游 → 409 点名要跑哪一步），再算显存账。
    # 反过来的话，一个"还没切帧"的用户会拿到「显存不够」，而真原因是那个 409。
    # 闸门只算账：不加载模型、不建 job、不开线程 —— 拒绝 = 这一单什么都没发生。
    vplan = _stage_vram_gate(cfg, stage, force=_bool_val(body.get("force"), False),
                             dry=_bool_val(body.get("dry_run"), False))
    cfg["vram_plan"] = vplan
    cfg["vram_force"] = bool(vplan.get("forced"))
    if _bool_val(body.get("dry_run"), False):
        return _ok(dry_run=True, stage=stage, started=False, skipped=False,
                   vram_plan=dict(vplan), need_mb=vplan.get("need_mb"),
                   avail_mb=vplan.get("avail_mb"), free_mb=vplan.get("free_mb"),
                   suggest=list(vplan.get("suggest") or []),
                   frames=int(vplan.get("frames") or 0),
                   batch=int(vplan.get("batch") or 0),
                   on_cpu=bool(_vram_cpu_on()))

    # 线程起来之后 meta 会被 worker 改：范围日志要用的旧指纹先在这里取好
    old_key = str(((cfg.get("meta") or {}).get("stages", {}).get(stage) or {})
                  .get("range_key") or "all")
    with _STAGE_LOCK:
        _STAGE_JOB.update({
            "running": True, "ok": None, "cancelled": False, "error": "",
            "error_detail": "", "phase": "running", "stage": stage,
            "stage_index": _STAGE_ORDER.index(stage), "stage_pct": 0, "pct": 0,
            "msg": "%s 启动…" % _STAGE_LABEL.get(stage, stage),
            "video": str(cfg["video"]), "cache_dir": str(cfg["cache"]),
            "out_video": str(cfg["out"]),
            # 进度一律**相对范围**：frames_total = 这一段要处理的帧数（frames 阶段才是全片）
            "frames_total": int(cfg.get("range_total") or cfg.get("total") or 0),
            "range": dict(cfg.get("range") or {}),
            "frames_done": 0, "frame_fps": 0.0, "eta_s": None, "swapped": 0,
            "copied": 0, "no_face": 0, "stages_done": _stage_done_list(cfg["meta"]),
            "log": [], "meta": cfg["meta"], "seconds": {}, "skipped": [],
            "cache_removed": False, "counts": _stage_counts(cfg["cache"]),
            "started_at": time.time(), "finished_at": None, "chain": False,
            "keep_cache": bool(cfg.get("keep_cache")), "force": bool(cfg.get("force")),
            "frames": [], "vram_plan": dict(cfg.get("vram_plan") or {}),
        })
    # ---- 生效配置：一行 [eff] + 一致性自检（进 status.warnings[] / 进度页横幅）----
    try:
        eff = _eff_publish("/api/merge/stage/%s" % stage, "stage:%s" % stage, body, cfg)
        with _STAGE_LOCK:
            _STAGE_JOB.update(_eff_status_patch(eff))
    except Exception as e:                                  # noqa: BLE001
        print("[eff] 生成失败（不影响这一单）：%s: %s"
              % (e.__class__.__name__, str(e)[:200]), flush=True)
    threading.Thread(target=_stage_worker, args=(cfg,), daemon=True,
                     name="merge-stage-%s" % stage).start()
    rng = cfg.get("range") or {}
    _stage_log("阶段 %s 启动：%s -> %s" % (stage, cfg["video"], cfg["cache"]))
    _stage_log("帧图编码：PNG 无损（compression level %s）/ JPEG quality=%d + 4:4:4"
               "（界面没有该旋钮，固定质量；PNG 不受影响）"
               % (cfg.get("png_level"), int(cfg.get("jpeg_quality")
                                            or _STAGE_JPEG_QUALITY)))
    _stage_log("帧范围：第 %s..%s 帧（共 %s 帧）%s"
               % (rng.get("start"), rng.get("end"), rng.get("total"),
                  "· 全片" if rng.get("full") else "· 只处理这一段"))
    if cfg.get("range_changed"):
        _stage_log("范围变了：%s 上一次跑的是另一段范围（指纹 %s → %s），旧游标作废，"
                   "本次从范围内第 1 帧重跑"
                   % (_STAGE_LABEL.get(stage, stage), old_key, cfg.get("range_key")))
    return _ok(started=True, skipped=False, **_stage_brief_meta(cfg))


@router.post("/api/merge/reveal")
@_api
def merge_reveal(body: dict):
    """在资源管理器里定位一个文件/目录（「打开所在目录」）。"""
    body = _as_dict(body)
    p = _norm_path(body.get("path"))
    if p is None:
        raise _BadInput("缺少 path：要一个文件或目录")
    if not p.exists():
        raise _BadInput("路径不存在：%s" % p, 404)
    target = p if p.is_dir() else p.parent
    try:
        import sys as _sys
        flags = getattr(_app_mod, "_NO_WINDOW", 0)
        if _sys.platform.startswith("win"):
            if p.is_dir():
                subprocess.Popen(["explorer", "/e,", str(p)], creationflags=flags)
            else:
                subprocess.Popen(["explorer", "/select," + str(p)], creationflags=flags)
        elif _sys.platform == "darwin":
            subprocess.Popen(["open", "-R" if p.is_file() else "", str(p)])
        else:
            subprocess.Popen(["xdg-open", str(target)])
    except Exception as e:                            # noqa: BLE001
        raise _BadInput("打不开所在目录：%s" % str(e)[:160], 500)
    return _ok(path=str(p), dir=str(target))


# ============================================================
# 环上的依赖：同包 import 放在文件末尾
# ------------------------------------------------------------
# 拆分前这些都是同一个模块里的全局名，可以随手互相引用；拆开后 routes 与 _REPO_ROOT、_app_mod、_eff、_projplan、app、compose、eff、media_encoder、media_frames、meter、pipeline_video、preview、proj、stages、vram
# 互相要用对方的东西（合成 / 流水线 / 显存账本本来就咬在一起）。放在文件末尾
# import 是唯一静态成立的做法：任何模块被 import 时自己的 def 和赋值都已跑完，
# 对方再来取名字一定拿得到，先 import 谁都一样。
# ============================================================
from .._paths import ROOT as _REPO_ROOT

from .. import app as _app_mod

from .. import _eff as _eff

from .. import _projplan as _projplan

from ..app import (
    WORKSPACE, _dfl_project_scan, _dfljpg_dict, _merge_job, _merge_job_lock, _merge_prep,
    _merge_prep_lock, _merge_prepare_worker, _merge_settings_to_config, _safe_dirname,
    _video_probe,
)

from .compose import (
    _body_idx, _cache_mode, _cache_set, _cache_snap, _check_cache_ready, _det_run_lock,
    _detector, _detector_val, _face_debug_payload, _frame_body, _frame_image, _frame_payload,
    _has_dfl_meta, _merge_query_into, _merged_from_export, _scan_first, _with_kps68,
)

from .eff import (
    _PREVIEW_EFF, _PREVIEW_EFF_LOCK, _eff_publish, _eff_status_patch,
)

from .._media.encoder import (
    _encoder_val, _hwaccel_val,
)

from .._media.frames import (
    _IMAGE_EXT, _VIDEO_EXT, _grab_key, _video_duration,
)

from .meter import (
    _STAGE_METER,
)

from .pipeline_video import (
    _ensure_cancel_watch, _log_tail, _run_export, _start_video_export,
)

from .preview import (
    _PREVIEW_BOOT_TIMEOUT, _PREVIEW_HIT, _PREVIEW_HIT_LOCK, _PREVIEW_LOAD_TIMEOUT,
    _PREVIEW_MAX_BATCH, _PREVIEW_QUALITY, _PREVIEW_WORK_TIMEOUT, _WORKER, _WORKER_MAX_MODELS,
    _dfm_of, _face_ids_val, _models_of, _preview_job, _proj_batch_default, _proj_plan_note,
    _proj_workers_default, _warm_kick, _warm_note_hot, _warm_status, _worker_pid, _xseg_of,
)

from .pipeline_project import (
    _PROJ_FRAME_EXT, _PROJ_JOB, _PROJ_LOCK, _proj_set, _proj_snap, _proj_source_video,
    _proj_warn, _proj_worker,
)

from .pipeline_stages import (
    _STAGE_JOB, _STAGE_JPEG_QUALITY, _STAGE_LABEL, _STAGE_LOCK, _STAGE_ORDER, _mask_src,
    _stage_brief_meta, _stage_cfg, _stage_counts, _stage_done_list, _stage_log, _stage_meta_load,
    _stage_name, _stage_next, _stage_precheck, _stage_set, _stage_stages_meta, _stage_vram_gate,
    _stage_worker,
)

from .vram import (
    _dfm_info, _preview_guard, _vram_cpu_on, _vram_free_smi, _vram_guard, _vram_last,
    _vram_models_mb, _vram_reclaim, _vram_release_inproc, _vram_sess_peak_mb,
)
