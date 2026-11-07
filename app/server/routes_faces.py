# -*- coding: utf-8 -*-
"""人脸数据库 API：/api/merge/faces/*（挂在合成工作室的同一个 app 上）。

存储细节全在 `_faces.py`；这一层只做三件事：

1. 解请求 / 校验参数（project、frame、ids、group、dfm）；
2. 复用现有的检测与对齐内核：`app._get_detector()` + `_align.align_face()`，
   不另写一套检测/裁脸逻辑；
3. 把 `_faces.FacesError` 统一成 `{ok:false, error, detail}`，
   并把「这一帧没有脸」这种正常结果做成 200 + `{ok:false, reason:"no_face"}`。

接口清单见 app/docs/12-FACE-DB.md。
"""
from __future__ import annotations

import base64
import functools
import traceback
import urllib.parse
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse, Response

from . import _faces as store
from ._align import align_face

router = APIRouter()

_VIDEO_EXT = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv", ".wmv", ".m4v",
              ".mpg", ".mpeg", ".ts", ".m2ts", ".m2v", ".vob", ".ogv", ".3gp"}


# ============================================================
# 统一响应
# ============================================================
def _err(status: int, msg: str) -> JSONResponse:
    text = str(msg or "未知错误")
    return JSONResponse(status_code=int(status),
                        content={"ok": False, "error": text, "detail": text})


def _api(fn):
    """把 FacesError / HTTPException / 未捕获异常统一成同一种错误体。"""

    @functools.wraps(fn)
    def _wrapped(*a, **kw):
        try:
            return fn(*a, **kw)
        except store.FacesError as e:
            return _err(e.status, e.msg)
        except HTTPException as e:
            return _err(e.status_code, str(e.detail))
        except Exception as e:                       # noqa: BLE001
            traceback.print_exc()
            return _err(500, "服务端内部错误：%s" % (str(e)[:200] or e.__class__.__name__))

    return _wrapped


def _as_dict(body) -> dict:
    if body is None:
        return {}
    if not isinstance(body, dict):
        raise store.FacesError("请求体要是一个 JSON 对象，收到 %s" % type(body).__name__, 400)
    return body


def _int_val(raw, name: str, lo: int = 0, hi: int = 10 ** 8) -> int:
    if raw is None or (isinstance(raw, str) and not raw.strip()):
        raise store.FacesError("缺少参数 %s" % name, 400)
    if isinstance(raw, bool):
        raise store.FacesError("「%s」要是整数，收到布尔值" % name, 400)
    try:
        value = int(str(raw).strip(), 10)
    except (TypeError, ValueError):
        raise store.FacesError("「%s」要是整数，收到 %r" % (name, raw), 400)
    if value < lo or value > hi:
        raise store.FacesError("「%s」要在 %d~%d 之间，收到 %d" % (name, lo, hi, value), 400)
    return value


def _thumb_url(project: Path, face_id: str) -> str:
    return "/api/merge/faces/thumb?project=%s&id=%s" % (
        urllib.parse.quote(str(project), safe=""),
        urllib.parse.quote(str(face_id), safe=""),
    )


def _decorate(project: Path, faces: list) -> list:
    out = []
    for f in faces:
        item = dict(f)
        item["thumb_url"] = _thumb_url(project, f["id"])
        out.append(item)
    return out


def _decorate_groups(groups: list) -> list:
    """给每个组补上 DFM 输入尺寸（正方形边长）。

    前端组头要显示「384×384 / 416×416」。尺寸的真相在 .dfm 的 ONNX 输入张量里，
    而 GET /api/merge/faces 本来只回 id/dfm/name/orphan —— 不在这一层补，前端只能瞎猜。
    找不到 .dfm 时沿用后端提取时的同一规则：512。
    """
    out = []
    for g in groups:
        item = dict(g)
        try:
            _name, size = _resolve_dfm(g.get("dfm"))
        except Exception:
            size = 512
        item["size"] = int(size)
        out.append(item)
    return out


# ============================================================
# 取当前帧 / 解析 DFM
# ============================================================
def _source_video(project: Path):
    """项目对应的原视频：prepare 记下的 > 项目目录里唯一/同名的 .mp4。"""
    from . import app as app_mod
    try:
        src = app_mod._merge_source_video(str(project))
    except Exception:
        src = ""
    if src:
        p = Path(src)
        if p.is_file():
            return p
    try:
        vids = sorted(v for v in project.iterdir()
                      if v.is_file() and v.suffix.lower() in _VIDEO_EXT
                      and ".merge_" not in v.name)
    except OSError:
        vids = []
    same = [v for v in vids if v.stem == project.name]
    if same:
        return same[0]
    if len(vids) == 1:
        return vids[0]
    return None


def _load_frame(project: Path, frame: int):
    """当前帧 → (BGR 图, source dict)。

    优先读项目里的帧图（prepare 抽出来的 000000.jpg…，和视频帧号一一对应），
    没有帧图才回退到项目里的源视频按帧号 seek。两条路都走不通就 404/409。
    """
    from . import app as app_mod
    names, srcdir, _kind = app_mod._dfl_project_scan(project)
    video = _source_video(project)
    video_name = video.name if video is not None else project.name

    if names and frame < len(names):
        img = app_mod._imread_any(Path(srcdir) / names[frame])
        if img is not None:
            return img, {"video": video_name, "frame": int(frame)}

    if video is not None:
        import cv2
        cap = cv2.VideoCapture(str(video))
        try:
            if cap.isOpened():
                total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
                if total and frame >= total:
                    raise store.FacesError("帧号越界：%d，视频只有 %d 帧（0~%d）"
                                           % (frame, total, total - 1), 404)
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame))
                ok, img = cap.read()
                if ok and img is not None:
                    return img, {"video": video.name, "frame": int(frame)}
        finally:
            cap.release()

    if names:
        raise store.FacesError("帧号越界：%d，项目里只有 %d 帧（0~%d）"
                               % (frame, len(names), len(names) - 1), 404)
    raise store.FacesError("项目里既没有帧图也没有源视频：%s" % project, 409)


def _resolve_dfm(raw):
    """dfm 参数 → (模型名, 输入尺寸)。

    dfm 只认**模型名**（和 /api/merge/export 的 model 一个规矩，不能带路径）。
    在 workspace/model/<name>.dfm 找到就读 ONNX 输入尺寸；找不到就用 512 ——
    这样前端即便只传一个组名，提取也能继续。
    """
    name = str(raw or "").strip()
    if not name:
        return "", 512
    if "/" in name or "\\" in name or name in (".", "..") or ".." in name:
        raise store.FacesError("dfm 只能是模型名，不能带路径：%r" % raw, 400)
    if Path(name).drive:
        raise store.FacesError("dfm 只能是模型名，不能带盘符：%r" % raw, 400)
    if name.lower().endswith(".dfm"):
        name = name[:-4]
    from . import app as app_mod
    path = Path(app_mod.WORKSPACE) / "model" / (name + ".dfm")
    size = store.dfm_input_size(path, 512) if path.is_file() else 512
    return name, int(size)


# ============================================================
# API
# ============================================================
@router.get("/api/merge/faces")
@_api
def list_faces(project: str = ""):
    """整个库：groups + faces + stats。"""
    p = store.resolve_project(project)
    data = store.snapshot(p)
    data["groups"] = _decorate_groups(data["groups"])
    data["faces"] = _decorate(p, data["faces"])
    return {"ok": True, "project": str(p), **data}


@router.post("/api/merge/faces/extract")
@_api
def extract_faces(body: dict):
    """从当前帧检测并裁出**所有**人脸，一张脸一条记录。

    一帧多脸不合并、不取最大——检测器返回几张就存几张，前端再让用户勾选。
    无人脸是正常结果：200 + {ok:false, reason:"no_face"}，不是 500。
    """
    body = _as_dict(body)
    p = store.resolve_project(body.get("project"))
    frame = _int_val(body.get("frame"), "frame", 0, 10 ** 7)
    dfm_name, size = _resolve_dfm(body.get("dfm"))

    img, source = _load_frame(p, frame)
    detections = store.detect_faces_in_image(img)
    crops, boxes, scores = [], [], []
    for det in detections or []:
        try:
            kps = det.get("kps")
            box = det.get("box")
            if not kps or not box:
                continue
            crop, _meta = align_face(img, kps, box, size)
        except Exception:
            continue
        if crop is None:
            continue
        crops.append(crop)
        boxes.append([round(float(v), 1) for v in box])
        scores.append(round(float(det.get("score") or 0), 4))

    if not crops:
        return {"ok": False, "reason": "no_face", "project": str(p),
                "frame": frame, "detected": len(detections or []),
                "size": int(size)}

    records, group = store.add_faces(p, crops, source, dfm=dfm_name)
    for rec, box, score in zip(records, boxes, scores):
        rec["thumb_url"] = _thumb_url(p, rec["id"])
        path, _etag = store.ensure_thumb(p, rec["id"])
        rec["thumb"] = "data:image/jpeg;base64," + base64.b64encode(
            path.read_bytes()).decode("ascii")
        rec["box"] = box
        rec["score"] = score
    return {"ok": True, "project": str(p), "frame": frame, "size": int(size),
            "dfm": dfm_name, "group": group, "count": len(records),
            "detected": len(detections or []), "faces": records}


@router.post("/api/merge/faces/move")
@_api
def move_faces(body: dict):
    """跨组投放：只改 face.group，文件一个字节都不动。"""
    body = _as_dict(body)
    p = store.resolve_project(body.get("project"))
    group = body.get("group")
    if group is None:
        group = ""
    if not isinstance(group, str):
        raise store.FacesError("group 要是组 id 字符串（未分组给空串）", 400)
    faces = store.move_faces(p, body.get("ids"), group)
    return {"ok": True, "project": str(p), "group": group,
            "updated": len(faces), "faces": _decorate(p, faces)}


@router.post("/api/merge/faces/trash")
@_api
def trash_faces(body: dict):
    body = _as_dict(body)
    p = store.resolve_project(body.get("project"))
    faces, missing = store.trash_faces(p, body.get("ids"))
    return {"ok": True, "project": str(p), "updated": len(faces),
            "faces": _decorate(p, faces), "missing_files": missing}


@router.post("/api/merge/faces/restore")
@_api
def restore_faces(body: dict):
    body = _as_dict(body)
    p = store.resolve_project(body.get("project"))
    faces, missing = store.restore_faces(p, body.get("ids"))
    return {"ok": True, "project": str(p), "updated": len(faces),
            "faces": _decorate(p, faces), "missing_files": missing}


@router.delete("/api/merge/faces")
@_api
def delete_faces(body: dict):
    body = _as_dict(body)
    p = store.resolve_project(body.get("project"))
    removed, missing = store.delete_faces(p, body.get("ids"))
    return {"ok": True, "project": str(p), "deleted": removed,
            "count": len(removed), "missing_files": missing}


@router.post("/api/merge/faces/groups")
@_api
def sync_groups(body: dict):
    """同步 .dfm 组：幂等；不再选的组有脸 → orphan=true，不删脸。"""
    body = _as_dict(body)
    p = store.resolve_project(body.get("project"))
    data = store.sync_groups(p, body.get("dfms"))
    data["groups"] = _decorate_groups(data["groups"])
    data["faces"] = _decorate(p, data["faces"])
    return {"ok": True, "project": str(p), **data}


@router.get("/api/merge/faces/thumb")
def face_thumb(request: Request, project: str = "", id: str = ""):
    """128×128 JPEG 缩略图；_thumb/<id>.jpg 缓存 + ETag/Cache-Control。"""
    try:
        p = store.resolve_project(project)
        path, etag = store.ensure_thumb(p, id)
    except store.FacesError as e:
        return _err(e.status, e.msg)
    headers = {
        "Cache-Control": "public, max-age=604800, immutable",
        "ETag": etag,
    }
    inm = request.headers.get("if-none-match") or ""
    if inm:
        tags = {t.strip() for t in inm.split(",") if t.strip()}
        if etag in tags or inm.strip() == "*":
            return Response(status_code=304, headers=headers)
    try:
        blob = path.read_bytes()
    except OSError as e:
        return _err(404, "缩略图读不出来：%s" % e)
    return Response(content=blob, media_type="image/jpeg", headers=headers)
