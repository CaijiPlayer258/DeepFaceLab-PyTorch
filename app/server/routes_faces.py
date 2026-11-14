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

from . import _face_group as facegrp
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


def _threshold_val(raw, default=None) -> float:
    """相似度阈值入参 → [0,1] 的浮点；缺省用 _face_group.DEFAULT_THRESHOLD。"""
    if default is None:
        default = facegrp.DEFAULT_THRESHOLD
    if raw is None or (isinstance(raw, str) and not raw.strip()):
        return float(default)
    if isinstance(raw, bool):
        raise store.FacesError("「threshold」要是 0~1 的小数，收到布尔值", 400)
    try:
        value = float(str(raw).strip())
    except (TypeError, ValueError):
        raise store.FacesError("「threshold」要是 0~1 的小数，收到 %r" % (raw,), 400)
    if not (0.0 <= value <= 1.0):
        raise store.FacesError("「threshold」要在 0~1 之间，收到 %s" % value, 400)
    return value


def _store_embeddings(p: Path, records, crops):
    """刚入库的脸顺手把特征写掉：GET 的老库补算就不会再碰这几张。

    失败一律吞掉（模型缺失 / 图坏了）：入库是主流程，特征只是给分组判定用的，
    不能因为算不出向量就让「＋」失败 —— 下次 GET 还会再试一次。
    """
    for rec, crop in zip(records or [], crops or []):
        try:
            vec = facegrp.embed_crop(crop)
            if vec is not None:
                facegrp.write_vector(p, rec["id"], vec)
        except Exception:                                  # noqa: BLE001
            continue


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


def _thumb_url(project: Path, face_id: str, version=None) -> str:
    """缩略图 URL；version 非空时带 v=。

    同一帧那张脸被「替换」时 id 不变、PNG 被覆盖、_thumb/<id>.jpg 被删掉重建，
    浏览器不能继续吃 immutable 缓存里的旧图 —— 所以带上 added_at 当版本号。
    """
    url = "/api/merge/faces/thumb?project=%s&id=%s" % (
        urllib.parse.quote(str(project), safe=""),
        urllib.parse.quote(str(face_id), safe=""),
    )
    if version:
        url += "&v=%s" % urllib.parse.quote(str(version), safe="")
    return url


def _decorate(project: Path, faces: list) -> list:
    out = []
    for f in faces:
        item = dict(f)
        item["thumb_url"] = _thumb_url(project, f["id"], f.get("added_at") or None)
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
        crop, meta = align_face_68(img, lm, box, size)
    except Exception:
        crop, meta = None, None
    if crop is None:
        # 外点检查判异常时 meta 里带着原因（resid / 阈值），原样上抛给调用方记账
        why = (meta or {}).get("reject_reason") if isinstance(meta, dict) else ""
        return None, lm, ("align_rejected: " + why) if why else "align_failed"
    return crop, lm, ""


def _material_snap():
    """合成器当前打开着的素材 → (snap dict, 模式 "video"/"project"/"")。"""
    try:
        from .routes_merge import _cache_mode, _cache_snap
        snap = _cache_snap() or {}
        return snap, str(_cache_mode(snap) or "")
    except Exception:                                     # noqa: BLE001
        return {}, ""


def _frame_probe(proj_raw, frame, body=None):
    """「现在到底能不能拿到这一帧的图」—— add 和 faces/source 唯一的口径。

    取图顺序和「点＋」实际走的那条完全一致：
      1. 项目帧图 / 项目里的源视频（项目模式）；
      2. body 里的 frame_path / frame_b64（前端自己截的那一张）；
      3. 合成器打开着的素材（视频模式，或打开着的帧目录）。

    返回 dict：img / source / project / mode / why。
      img      BGR 图；拿不到就是 None
      mode     "project" / "video" / "upload" / "none"；拿不到图时是**当前模式**
      why      拿不到时的原因。**必须**说清楚「当前是视频模式/项目模式、缺的是
               哪一样」—— 前端原样弹这一句，不再自己猜。
    """
    proj_text = str(proj_raw or "").strip()
    snap, cache_mode = _material_snap()
    material_path = str(snap.get("path") or "")
    project, why_project = None, ""

    if proj_text:
        try:
            project = store.resolve_project(proj_text)
        except store.FacesError as e:
            why_project = "项目目录打不开（%s）" % e.msg
        else:
            try:
                img, src = _load_frame(project, frame)
                if img is not None:
                    return {"img": img, "source": src, "project": project,
                            "mode": "project", "why": ""}
            except store.FacesError as e:
                why_project = "项目里的帧读不出来（%s）" % e.msg
            except Exception as e:                        # noqa: BLE001
                why_project = "项目里的帧读不出来（%s）" % str(e)[:160]

    if body:
        for key, reader in (("frame_path", _read_image_path),
                            ("frame_b64", _decode_b64_image)):
            if not body.get(key):
                continue
            img = reader(body.get(key))
            if img is not None:
                return {"img": img, "source": None, "project": project,
                        "mode": "upload", "why": ""}
            why_project = why_project or ("%s 里的图解不开" % key)

    why_material = ""
    if cache_mode:
        try:
            from .routes_merge import _frame_image
            img, _total = _frame_image(int(frame))
            if img is not None:
                name = Path(material_path).name if material_path else (
                    project.name if project is not None else "")
                return {"img": img, "source": {"video": name, "frame": int(frame)},
                        "project": project, "mode": cache_mode, "why": ""}
            why_material = "这一帧读不出来"
        except Exception as e:                            # noqa: BLE001
            why_material = str(getattr(e, "msg", "") or e)[:200]

    # 传了 project 就算「项目模式」（哪怕这个目录根本打不开）：报错才不至于说成
    # 「既没有打开视频、也没有项目目录」—— 用户明明给了一个项目路径。
    mode = cache_mode or ("project" if (project is not None or proj_text) else "none")
    if mode == "project":
        head = "当前是项目模式"
        lack = why_project or why_material or "项目里这一帧既没有帧图也没有源视频"
    elif mode == "video":
        head = "当前是视频模式"
        lack = why_material or "还没打开素材（先导入一段视频）"
    else:
        head = "当前既没有打开视频、也没有项目目录"
        lack = why_project or "先在合成器里导入一段视频或打开一个 DFL 项目"
    return {"img": None, "source": None, "project": project, "mode": mode,
            "why": "%s：%s（frame=%d）" % (head, lack, int(frame))}


def _read_frame(project: Path, frame: int, created: bool = False):
    """取整帧原图：项目帧图/项目里的源视频 > merge 缓存里打开着的素材（视频模式）。

    两条路都是 cv2 解码同一份字节，所以 extract 和 add 拿到的像素完全一样。
    created=True 表示项目是这次新建的（肯定还没有帧图），直接读打开着的素材。
    """
    probe = _frame_probe("" if created else project, frame)
    if probe["img"] is None:
        raise store.FacesError(probe["why"]
                               or ("取不到这一帧的图（frame=%d）" % frame), 409)
    source = probe["source"]
    if not source:
        src = _source_video(project)
        source = {"video": (src.name if src is not None else project.name),
                  "frame": int(frame)}
    return probe["img"], source


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
    """整个库：groups + faces + stats。

    project 为空（视频模式 / 默认库）→ 读 <工作区>/merge；默认库还不存在就建，
    第一次进来看到的是一份空库而不是「缺少 project」的 400。
    """
    p = store.resolve_library(project)
    # 同一帧同一张脸只留一张：老库里堆出来的重复在这里挪进回收站（可还原，不删）
    deduped = store.dedup_frames(p)
    data = store.snapshot(p)
    # 老库的脸没有特征文件：**在这里惰性补算**（有 PNG 就能算）。
    # 算不出来的跳过，列表照常返回；一次最多补 BACKFILL_PER_GET 张，大库不卡首屏。
    emb = facegrp.backfill(p)
    data["groups"] = _decorate_groups(data["groups"])
    data["faces"] = _decorate(p, data["faces"])
    return {"ok": True, "project": str(p), "deduped": int(deduped),
            "embedded": int(emb["computed"]), "emb_have": int(emb["have"]),
            "emb_total": int(emb["total"]), "emb_failed": int(emb["failed"]),
            "emb_pending": int(emb["pending"]), **data}


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
    crops, boxes, scores, kps_all, skipped, idx_all = [], [], [], [], [], []
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
        idx_all.append(int(i))          # 同一帧里的第几张：多脸帧各存各的

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
    records, group = store.add_faces(p, crops, source, dfm=dfm_name, align="68",
                                     idxs=idx_all)
    _store_embeddings(p, records, crops)
    _attach_extras(p, records, boxes, scores, kps_all)
    return {"ok": True, "project": str(p), "created": bool(created),
            "frame": frame, "size": int(size), "align": "68", "detector": det_name,
            "dfm": dfm_name, "group": group, "count": len(records),
            "replaced": sum(1 for r in records if r.get("replaced")),
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


def _nearest_det(detections, box):
    """在检测结果里挑「离传入 box 最近」的那张脸。

    box 是前端画面上那个框（原图坐标）。有重叠就按 IoU 取最大；一张都没重叠
    （用户框和这次检测差了一帧 / 抖了一下）就按框中心距离取最近。没传 box 时
    退化成「面积最大的那张脸」—— 一帧多脸时那通常就是用户正在看的那张。
    """
    best, best_key = None, None
    for det in (detections or []):
        try:
            b = [float(v) for v in (det.get("box") or [])[:4]]
        except (TypeError, ValueError):
            continue
        if len(b) < 4:
            continue
        bw, bh = max(0.0, b[2] - b[0]), max(0.0, b[3] - b[1])
        if bw <= 0 or bh <= 0:
            continue
        if box is None:
            key = (-bw * bh, -float(det.get("score") or 0))
        else:
            iw = max(0.0, min(b[2], box[2]) - max(b[0], box[0]))
            ih = max(0.0, min(b[3], box[3]) - max(b[1], box[1]))
            inter = iw * ih
            union = bw * bh + max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1]) - inter
            iou = (inter / union) if union > 0 else 0.0
            dx = (b[0] + b[2]) / 2 - (box[0] + box[2]) / 2
            dy = (b[1] + b[3]) / 2 - (box[1] + box[3]) / 2
            key = (-round(iou, 6), (dx * dx + dy * dy) ** 0.5)
        if best_key is None or key < best_key:
            best, best_key = det, key
    return best


def _server_landmarks68(img, box, detector):
    """服务端兜底取点：这一帧 → 检测框（离传入 box 最近的那张脸）→ 2D106 → 68 点。

    返回 ((kps68, box, score), None) 或 (None, {"reason", "detail"})。
    和 /api/merge/detect 同一条链路（_frame_landmarks）：拿不到就明确失败，
    绝不退回 5 点插值。
    """
    try:
        detections = store.detect_faces_in_image(img, detector)
    except Exception as e:                                # noqa: BLE001
        msg = str(getattr(e, "detail", "") or e)[:160]
        return None, {"reason": "detect_failed",
                      "detail": "服务端跑检测时失败：%s" % (msg or "未知错误")}
    det = _nearest_det(detections, box)
    if det is None:
        return None, {"reason": "no_face",
                      "detail": "服务端在这一帧里没检测到人脸（detector=%s）"
                                % (detector or "默认")}
    dbox = det.get("box")
    lm = _landmarks68(img, det, dbox)
    if lm is None:
        return None, {"reason": "no_landmarks68",
                      "detail": "服务端检测到了框，但 2D106 回归 68 点失败"
                                "（权重缺失或这一帧太糊）"}
    return (lm, [round(float(v), 1) for v in dbox[:4]],
            round(float(det.get("score") or 0), 4) or None), None


@router.post("/api/merge/faces/add")
@_api
def add_face(body: dict):
    """把已经知道是哪张脸的那一块直接入库（不整帧落多张）。

    请求：{ project?, kps68?, box?, dfm?, frame, detector?, source?:{video,frame},
            frame_path?, frame_b64? }

    * **kps68 可以不给**：缺了、只有 5 点、或者形状不是 68x2，后端自己从这一帧
      的图上重跑「检测（挑离传入 box 最近的那张脸）→ 2D106 回归 68 点 → 68 点
      仿射裁剪」，返回体里写 landmarks_from:"server"。
      以前这里只认前端手里的 kps68：老后端（/api/merge/detect 不回 kps68）时
      前端只能弹「缺少 68 点」，＋等于什么都没干。
    * 真的拿不到图才失败，错误里写明「当前是视频模式 / 项目模式、缺的是哪一个」。
    * 裁剪和 extract 完全同一条 68 点仿射（align_face_68），所以同一帧同一张脸
      两条路落下来的 PNG 逐像素相同；5 点插值的假点一个都不落盘。
    * project 可有可无：没有就用默认库 <工作区>/merge（后端自动建目录与
      index.json，越界到工作区外面一律 400）。
    """
    body = _as_dict(body)
    dfm_name, size = _resolve_dfm(body.get("dfm"))
    det_name = _detector_name(body.get("detector"))
    src = body.get("source") if isinstance(body.get("source"), dict) else {}
    frame = _int_val(body.get("frame", src.get("frame", 0)), "frame", 0, 10 ** 7)
    # 同一帧只留一张脸：idx = 这张脸在这一帧里是第几张（多脸帧各存各的）
    raw_idx = body.get("idx")
    if raw_idx is None:
        raw_idx = body.get("face_index")
    try:
        face_idx = int(raw_idx or 0)
    except (TypeError, ValueError):
        raise store.FacesError("「idx」要是整数（同一帧里的第几张脸，0 起），收到 %r"
                               % (raw_idx,), 400)
    if face_idx < 0:
        raise store.FacesError("「idx」不能是负数，收到 %d" % face_idx, 400)
    proj = str(body.get("project") or "").strip()
    box = _box_val(body.get("box"))

    raw_kps = body.get("kps68")
    if raw_kps is None:
        raw_kps = body.get("kps")
    kps = _kps68_val(raw_kps)
    landmarks_from = "client" if kps is not None else ""

    probe = _frame_probe(proj, frame, body)
    img = probe["img"]
    if img is None:
        raise store.FacesError(probe["why"]
                               or ("取不到这一帧的图（frame=%d）" % frame), 409)
    source = probe["source"]
    if not source:
        video = str(src.get("video") or body.get("video") or "").strip()
        source = {"video": (Path(video).name if video else ""), "frame": int(frame)}

    score = None
    if kps is None:
        got, why = _server_landmarks68(img, box, det_name)
        if got is None:
            return {"ok": False, "reason": why["reason"], "detail": why["detail"],
                    "project": str(probe["project"] or ""), "frame": int(frame),
                    "mode": str(probe["mode"] or ""), "landmarks_from": "",
                    "size": int(size), "dfm": dfm_name, "align": "68",
                    "expect": "68x2"}
        kps, box, score = got
        landmarks_from = "server"

    crop, _meta = align_face_68(img, kps, box, size)
    if crop is None:
        _why = (_meta or {}).get("reject_reason") if isinstance(_meta, dict) else ""
        raise store.FacesError(_why or "68 点退化（共线/面积为 0），这张脸对齐不出来", 400)

    p, created = store.resolve_project_for_write(proj, default=store.default_project())
    if created:
        store.ensure_index(p)

    # ---- 组归属：embedding 自动判定 ----
    # 产品语义：组里的人脸 = 那个组的 dfm，新脸该进哪一组由「它更像哪一组的脸」决定，
    # 不再由界面上的下拉框决定。顺序（见 _face_group.decide_for_add）：
    #   显式 group > 和组均值最像且 >= 阈值 > 合成器 chip 上的 dfm > 不归组
    threshold = _threshold_val(
        body.get("threshold") if body.get("threshold") not in (None, "")
        else body.get("face_threshold"))
    explicit_group = str(body.get("group") or "").strip()
    vec = facegrp.embed_crop(crop)
    decision = facegrp.decide_for_add(p, vec, explicit_group=explicit_group,
                                      chip_dfm=dfm_name, threshold=threshold)
    if not decision["group"] and not decision["dfm"]:
        # 既不像任何组、合成器那边也没给它分 dfm → **不落库**（落进去也只会归错人）。
        # 前端按 reason 提示「请手动拖到某个组」。
        return {"ok": False, "reason": "no_group_match", "matched": False,
                "detail": "这张脸和现有分组都不像，请手动拖到某个组",
                "project": str(p), "created": bool(created), "frame": int(frame),
                "threshold": float(threshold),
                "similarity": decision.get("similarity"),
                "best_group": decision.get("best_group") or "",
                "best_dfm": decision.get("best_dfm") or "",
                "best_similarity": decision.get("best_similarity"),
                "dfm": dfm_name, "group": "", "count": 0, "faces": [],
                "size": int(size), "align": "68",
                "landmarks_from": landmarks_from, "detector": det_name,
                "mode": str(probe["mode"] or ""), "box": box, "idx": int(face_idx)}

    records, group = store.add_faces(p, [crop], source, dfm=decision["dfm"],
                                     group=decision["group"], align="68",
                                     idxs=[face_idx])
    if vec is not None:
        facegrp.write_vector(p, records[0]["id"], vec)      # 新脸特征随 PNG 一起落盘
    _attach_extras(p, records, [box], [score], [kps.round(2).tolist()])
    rec = records[0]
    group_name, group_dfm = "", ""
    try:
        for g in store.load_index(p)["groups"]:
            if g["id"] == group:
                group_name = str(g.get("name") or g.get("dfm") or g["id"])
                group_dfm = str(g.get("dfm") or "")
                break
    except store.FacesError:
        pass
    # replaced=true 表示这一帧这张脸本来就在库里：覆盖了它，id / group 都没变
    return {"ok": True, "project": str(p), "created": bool(created),
            "dfm": dfm_name, "group": group, "group_name": group_name,
            "group_dfm": group_dfm,
            "matched": bool(decision["matched"]),
            "similarity": decision.get("similarity"),
            "threshold": float(threshold),
            "auto": decision["source"] == "auto",
            "source": decision["source"],
            "size": int(size), "align": "68",
            "landmarks_from": landmarks_from, "detector": det_name,
            "mode": str(probe["mode"] or ""), "box": box, "idx": int(face_idx),
            "replaced": bool(rec.get("replaced")), "id": rec["id"],
            "frame": int(frame), "count": 1, "face": rec, "faces": [rec]}


@router.get("/api/merge/faces/source")
@_api
def face_source(project: str = "", frame: int = 0):
    """「现在到底能不能拿到这一帧的图」—— 点＋ / 开人脸库之前问这一个。

    以前前端只看 window.__mergeInputDir：视频模式下它永远是空串，于是画面里
    明明有视频，点＋也会弹「还没有素材」。判定权现在在后端：项目帧（项目模式）
    / 打开着的素材（视频模式）/ 都没有 —— why 里写清楚缺的是哪一个。

    回：{ok, available, mode, project, default_project, frame, why, material, frames}
    """
    frame = _int_val(frame, "frame", 0, 10 ** 7)
    probe = _frame_probe(project, frame)
    snap, cache_mode = _material_snap()
    return {"ok": True,
            "available": probe["img"] is not None,
            "mode": str(probe["mode"] or ""),
            "frame": int(frame),
            "why": str(probe["why"] or ""),
            "project": str(probe["project"] or ""),
            "default_project": str(store.default_project()),
            "material": str(snap.get("path") or ""),
            "material_mode": str(cache_mode or ""),
            "frames": int(snap.get("frames") or 0)}


@router.post("/api/merge/faces/move")
@_api
def move_faces(body: dict):
    """跨组投放：只改 face.group，文件一个字节都不动。"""
    body = _as_dict(body)
    p = store.resolve_library(body.get("project"))
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
    p = store.resolve_library(body.get("project"))
    faces, missing = store.trash_faces(p, body.get("ids"))
    return {"ok": True, "project": str(p), "updated": len(faces),
            "faces": _decorate(p, faces), "missing_files": missing}


@router.post("/api/merge/faces/restore")
@_api
def restore_faces(body: dict):
    body = _as_dict(body)
    p = store.resolve_library(body.get("project"))
    faces, missing = store.restore_faces(p, body.get("ids"))
    return {"ok": True, "project": str(p), "updated": len(faces),
            "faces": _decorate(p, faces), "missing_files": missing}


@router.delete("/api/merge/faces")
@_api
def delete_faces(body: dict):
    body = _as_dict(body)
    p = store.resolve_library(body.get("project"))
    removed, missing = store.delete_faces(p, body.get("ids"))
    return {"ok": True, "project": str(p), "deleted": removed,
            "count": len(removed), "missing_files": missing}


@router.post("/api/merge/faces/groups")
@_api
def sync_groups(body: dict):
    """同步 .dfm 组：幂等；不再选的组有脸 → orphan=true，不删脸。"""
    body = _as_dict(body)
    p = store.resolve_library(body.get("project"))
    data = store.sync_groups(p, body.get("dfms"))
    data["groups"] = _decorate_groups(data["groups"])
    data["faces"] = _decorate(p, data["faces"])
    return {"ok": True, "project": str(p), **data}


@router.get("/api/merge/faces/thumb")
def face_thumb(request: Request, project: str = "", id: str = ""):
    """128×128 JPEG 缩略图；_thumb/<id>.jpg 缓存 + ETag/Cache-Control。"""
    try:
        p = store.resolve_library(project)
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


@router.get("/api/merge/faces/cache")
@_api
def faces_cache_state(fresh: str = ""):
    """进程内人脸缓存的真实规模（embedding / 组均值 / 逐帧自动分配方案）。

    给验收和排障断言用 —— 全 0 且 files 为空才叫「干净」：

        {"ok":true,"profiles":0,"extra":{"auto_plan":0},"files":[],"threshold":0.35}

    fresh=1：**先清一次再报**（等价于「导出结束 / 进程重启」之后该看到的样子）。
    两个正式调用点在别处：serve 启动（app.py 的 startup 钩子）和每一次导出 job
    结束（routes_merge._vid_worker 的 finally）。这里只报数，不落盘、不动库文件。
    """
    if str(fresh or "").strip().lower() in ("1", "true", "yes", "on"):
        data = facegrp.clear_face_group_cache("接口 fresh=1")
    else:
        data = facegrp.cache_state()
    data["ok"] = True
    return data

