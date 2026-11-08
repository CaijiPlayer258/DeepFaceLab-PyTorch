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
from ._align import align_face, align_face_68

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


def _box_val(raw):
    """可选检测框 → [x1,y1,x2,y2]（只当元数据用，不参与裁剪）。"""
    if raw is None:
        return None
    if not isinstance(raw, (list, tuple)) or len(raw) < 4:
        raise store.FacesError("box 要是 [x1, y1, x2, y2]", 400)
    try:
        return [float(v) for v in raw[:4]]
    except (TypeError, ValueError):
        raise store.FacesError("box 里有非数字：%r" % (raw,), 400)


def _kps68_val(raw):
    """68 点关键点 → (68,2) float64（两位小数）；不是 68 点就 None。

    只认 68：5 点对齐的旋转/缩放抖动会让同一个人的两张入库图对不齐，
    而库里的图是后面分组/导出的输入 —— 所以这里宁可报错也不退回 5 点。
    """
    import numpy as np
    if raw is None:
        return None
    try:
        arr = np.asarray(raw, dtype=np.float64)
    except (TypeError, ValueError):
        return None
    if arr.ndim != 2 or arr.shape != (68, 2) or not np.all(np.isfinite(arr)):
        return None
    return arr.round(2)


def _landmarks68(img, det, box):
    """检测结果 → 原图坐标的 68 点（2D106 真回归出来的）。

    和合成链路 _frame_landmarks() 是同一条（只认模型输出，没有模板兜底）：
    5 点先求 512 对齐 → 2D106 回归 68 点 → 用 image_to_face_mat 反变换回原图。
    拿不到就是 None，调用方必须报 no_landmarks68。
    """
    from .routes_merge import _frame_landmarks
    try:
        lm = _frame_landmarks(img, det, box)
    except Exception:
        lm = None
    if lm is None:
        return None
    import numpy as np
    arr = np.asarray(lm, dtype=np.float64)
    if arr.ndim != 2 or arr.shape != (68, 2) or not np.all(np.isfinite(arr)):
        return None
    # 两位小数：/api/merge/detect 回给前端的是同一份数，add 传回来才对得上
    return arr.round(2)


def _face_crop68(img, det, box, size):
    """检测结果 → 68 点 → 仿射裁剪。返回 (crop, kps68, why)；why 非空 = 失败原因。"""
    lm = _landmarks68(img, det, box)
    if lm is None:
        return None, None, "no_landmarks68"
    try:
        crop, _meta = align_face_68(img, lm, box, size)
    except Exception:
        crop = None
    if crop is None:
        return None, lm, "align_failed"
    return crop, lm, ""


def _read_frame(project: Path, frame: int, created: bool = False):
    """取整帧原图：项目帧图/项目里的源视频 > merge 缓存里打开着的素材（视频模式）。

    两条路都是 cv2 解码同一份字节，所以 extract 和 add 拿到的像素完全一样。
    """
    if not created:
        try:
            return _load_frame(project, frame)
        except store.FacesError:
            pass
    from .routes_merge import _frame_image
    img, _total = _frame_image(int(frame))
    src = _source_video(project)
    return img, {"video": (src.name if src is not None else project.name),
                 "frame": int(frame)}


def _detector_name(raw) -> str:
    """detector 入参 → 规范名；空 = 服务器默认。白名单和 /api/merge/detect 同一套。"""
    if raw is None or (isinstance(raw, str) and not raw.strip()):
        return ""
    try:
        from .routes_merge import _detector_val
        return _detector_val(raw)
    except store.FacesError:
        raise
    except Exception:
        raise store.FacesError("未知检测器：%r（只认 TinyMog / RetinaFace_10g）" % (raw,), 400)


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


def _attach_extras(p: Path, records, boxes, scores, kps68):
    """给新记录补前端要的三样：缩略图 URL / 内联缩略图 / 框与分数。"""
    for rec, box, score, lm in zip(records, boxes, scores, kps68):
        rec["thumb_url"] = _thumb_url(p, rec["id"])
        try:
            path, _etag = store.ensure_thumb(p, rec["id"])
            rec["thumb"] = "data:image/jpeg;base64," + base64.b64encode(
                path.read_bytes()).decode("ascii")
        except Exception:
            rec["thumb"] = ""
        rec["box"] = box
        rec["score"] = score
        rec["kps68"] = lm
    return records


@router.post("/api/merge/faces/extract")
@_api
def extract_faces(body: dict):
    """从当前帧检测并裁出**所有**人脸，一张脸一条记录。

    一帧多脸不合并、不取最大——检测器返回几张就存几张，前端再让用户勾选。
    裁剪统一走 68 点仿射（align_face_68）：2D106 回归 68 点，拿不到就**跳过**
    并报 no_landmarks68，绝不退回 5 点。

    结果是 200 + {ok:false, reason:...}（no_face / no_landmarks68），不是 500。
    project 目录不存在时自动创建（只在工作区内），响应里 created=true。
    """
    body = _as_dict(body)
    frame = _int_val(body.get("frame"), "frame", 0, 10 ** 7)
    dfm_name, size = _resolve_dfm(body.get("dfm"))
    det_name = _detector_name(body.get("detector"))
    p, created = store.resolve_project_for_write(body.get("project"),
                                                default=store.default_project())
    img, source = _read_frame(p, frame, created=created)
    detections = store.detect_faces_in_image(img, det_name)
    crops, boxes, scores, kps_all, skipped = [], [], [], [], []
    for i, det in enumerate(detections or []):
        box = det.get("box")
        if not box or not det.get("kps"):
            skipped.append({"face": i, "why": "检测框少了 5 点"})
            continue
        crop, lm, why = _face_crop68(img, det, box, size)
        if crop is None:
            skipped.append({"face": i, "why": why})
            continue
        crops.append(crop)
        boxes.append([round(float(v), 1) for v in box])
        scores.append(round(float(det.get("score") or 0), 4))
        kps_all.append(lm.round(2).tolist())

    if not crops:
        reason = "no_face"
        if detections and all(s["why"] == "no_landmarks68" for s in skipped):
            reason = "no_landmarks68"
        return {"ok": False, "reason": reason, "project": str(p),
                "created": bool(created), "frame": frame,
                "detected": len(detections or []), "skipped": skipped,
                "size": int(size)}

    if created:
        store.ensure_index(p)
    records, group = store.add_faces(p, crops, source, dfm=dfm_name, align="68")
    _attach_extras(p, records, boxes, scores, kps_all)
    return {"ok": True, "project": str(p), "created": bool(created),
            "frame": frame, "size": int(size), "align": "68", "detector": det_name,
            "dfm": dfm_name, "group": group, "count": len(records),
            "detected": len(detections or []), "skipped": skipped,
            "faces": records}


# ============================================================
# 直接入库（不再整帧重检测）
# ============================================================
# 一帧 4K PNG 的 base64 也就几 MB；给个上限，别让人把服务器撑爆。
_MAX_IMAGE_BYTES = 32 * 1024 * 1024


def _decode_b64_image(raw):
    """dataURL / 裸 base64 → BGR 图；解不开返回 None。"""
    import cv2
    import numpy as np
    text = str(raw or "").strip()
    if not text:
        return None
    if text.startswith("data:"):
        text = text.split(",", 1)[-1]
    if not text:
        return None
    try:
        blob = base64.b64decode(text, validate=False)
    except Exception:
        return None
    if not blob or len(blob) > _MAX_IMAGE_BYTES:
        return None
    return cv2.imdecode(np.frombuffer(blob, np.uint8), cv2.IMREAD_COLOR)


def _read_image_path(raw):
    """frame_path → BGR 图（必须是一个存在的文件）；读不到返回 None。"""
    text = str(raw or "").strip()
    if not text:
        return None
    try:
        fp = Path(text)
    except (OSError, ValueError):
        return None
    if not fp.is_file():
        return None
    from . import app as app_mod
    try:
        return app_mod._imread_any(fp)
    except Exception:
        return None


@router.post("/api/merge/faces/add")
@_api
def add_face(body: dict):
    """把已经知道是哪张脸的那一块直接入库（不重跑检测）。

    请求：{ project?, kps68|kps, box?, dfm?, frame, source?:{video,frame},
            frame_b64?, frame_path? }

    * kps 必须 68x2（/api/merge/detect 的每张脸都带 kps68）。只给 5 点 →
      200 + {ok:false, reason:"no_landmarks68"}，一个文件都不落（用户明确禁止
      5 点对齐：5 点的旋转/缩放抖动会让同一个人的两张入库图对不齐）。
    * 裁剪和 extract 完全同一条 68 点仿射（align_face_68），所以同一帧同一张脸
      两条路落下来的 PNG 逐像素相同。
    * project 可有可无：没有就用默认库 <工作区>/merge（后端自动建目录与
      index.json，越界到工作区外面一律 400）。
    """
    body = _as_dict(body)
    dfm_name, size = _resolve_dfm(body.get("dfm"))
    src = body.get("source") if isinstance(body.get("source"), dict) else {}
    frame = _int_val(body.get("frame", src.get("frame", 0)), "frame", 0, 10 ** 7)

    raw_kps = body.get("kps68")
    if raw_kps is None:
        raw_kps = body.get("kps")
    kps = _kps68_val(raw_kps)
    if kps is None:
        try:
            got = len(raw_kps or [])
        except Exception:
            got = 0
        return {"ok": False, "reason": "no_landmarks68",
                "detail": "add 只认 68 点仿射：kps 要是 68x2，收到 %d 个点" % got,
                "expect": "68x2", "size": int(size), "dfm": dfm_name}

    box = _box_val(body.get("box"))
    proj = str(body.get("project") or "").strip()

    # 取整帧原图（只读）：项目帧图 > frame_path > frame_b64 > 打开着的素材
    existing = None
    if proj:
        try:
            existing = store.resolve_project(proj)
        except store.FacesError:
            existing = None
    img, source = None, None
    if existing is not None:
        img, source = _read_frame(existing, frame)
    if img is None and body.get("frame_path"):
        img = _read_image_path(body.get("frame_path"))
    if img is None and body.get("frame_b64"):
        img = _decode_b64_image(body.get("frame_b64"))
    if img is None:
        img, source = _read_frame(store.default_project(), frame, created=True)
    if img is None:
        raise store.FacesError("取不到这一帧的图（frame=%d）：先在合成器里打开素材" % frame, 409)
    if not source:
        video = str(src.get("video") or body.get("video") or "").strip()
        source = {"video": (Path(video).name if video else ""), "frame": int(frame)}

    crop, _meta = align_face_68(img, kps, box, size)
    if crop is None:
        raise store.FacesError("68 点退化（共线/面积为 0），这张脸对齐不出来", 400)

    p, created = store.resolve_project_for_write(proj, default=store.default_project())
    if created:
        store.ensure_index(p)
    records, group = store.add_faces(p, [crop], source, dfm=dfm_name, align="68")
    _attach_extras(p, records, [box], [None], [kps.round(2).tolist()])
    rec = records[0]
    return {"ok": True, "project": str(p), "created": bool(created),
            "dfm": dfm_name, "group": group, "size": int(size), "align": "68",
            "frame": int(frame), "count": 1, "face": rec, "faces": [rec]}


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
