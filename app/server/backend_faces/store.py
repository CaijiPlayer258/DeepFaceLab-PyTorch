# -*- coding: utf-8 -*-
"""人脸数据库（Face DB）的存储层：<项目>/_faces/ 的读写，不依赖 FastAPI。

一个合成项目（= 一个视频）一个库，目录长这样：

    _faces/
        index.json          唯一清单：version / next_face / groups / faces
        index.json.bak      上一次成功写入的清单（只留一份）
        f_0001.png          统一尺寸的 PNG 人脸（512 或该 DFM 的输入尺寸）
        _trash/             回收站：文件搬进来，index 里 trashed=true
        _thumb/             128×128 JPEG 缩略图缓存（首次访问才生成）
        emb/                每张脸的 ArcFace 特征 <id>.npy（512 维 float32，L2 归一化）
        .lock               跨进程文件锁（不参与业务）

几个刻意的约定：

* `face["file"]` 只存**基名**（`f_0001.png`）。`trashed=false` 时文件在
  `_faces/`，`trashed=true` 时在 `_faces/_trash/` —— 回收站/还原只翻转
  trashed 并搬文件，id 和 file 都不变，前端不用重新对 id。
* `face["group"]` 可以是空串（只为了兼容老数据）：界面上已经没有「未分组」区，
  新脸一定落进某个组；空串的老脸在页面上算进第一个组。
* `face["idx"]` = 这张脸在同一帧里是第几张（0 起）。同一帧同一张脸只留一条：
  再点一次「＋」是**替换**（覆盖 PNG、刷新 added_at、id 与 group 都不动）。
* 组 id（g1 / g2 …）一旦分配永不回收；同步组时按 dfm 名字复用，名字大小写变了
  也只更新显示名，id 不动。
* 所有从请求或 index 里拿到的相对路径都过 :func:`safe_join`：绝对路径、盘符、
  `..` 一律 400 —— 库里的文件永远只能在 `_faces/` 里面。
* 写入一律 `.part` → `os.replace`；index 每次替换前先把旧内容存成 `.bak`。
* 一个项目一把锁：进程内 RLock + 跨进程 `.lock`（Windows msvcrt / POSIX flock）。
  同一时刻只有一个请求能改 index，10 个 move 并发也不会写出半个 JSON。
"""
from __future__ import annotations
import numpy as np

import contextlib
import json
import mmap
import os
import re
import threading
import time
from pathlib import Path

# 4.1 仓库根：界面上不带盘符的相对路径都相对它解析（和 app.py / routes_merge 一致）
ROOT = Path(__file__).resolve().parents[3]

FACES_DIRNAME = "_faces"
TRASH_DIRNAME = "_trash"
THUMB_DIRNAME = "_thumb"
EMB_DIRNAME = "emb"          # 每张脸一个 <id>.npy：分组判定用的 512 维特征
INDEX_NAME = "index.json"
BACKUP_NAME = "index.json.bak"
# 同帧去重只做一次：老库里堆出来的重复清完就记一笔，之后用户从回收站还原的
# 那张不会再被「latest 优先」重新塞回回收站（还原要能站得住）。
DEDUPE_VERSION = 1
LOCK_NAME = ".lock"

FACE_ID_RE = re.compile(r"^f_(\d{4,})$")
GROUP_ID_RE = re.compile(r"^g(\d+)$")

THUMB_SIZE = 128
THUMB_QUALITY = 90
MAX_IDS = 5000
LOCK_TIMEOUT = 15.0


class FacesError(Exception):
    """业务错误：带 HTTP 状态码，由路由层统一转成 {ok:false, error}。"""

    def __init__(self, msg, status=400):
        super().__init__(str(msg))
        self.msg = str(msg)
        self.status = int(status)


# ============================================================
# 小工具
# ============================================================
def _text(raw) -> str:
    if raw is None:
        return ""
    return str(raw).strip().strip('"').strip("'").strip()


def _parts(raw: str) -> list:
    r"""按 / 和 \ 切路径段（Windows 两种都认）。"""
    return [p for p in re.split(r"[\\/]+", str(raw)) if p]


def _has_parent(raw: str) -> bool:
    return ".." in _parts(raw)


def workspace_root() -> Path:
    """工作区根（和 app.WORKSPACE 同一条规则：DFL_WORKSPACE > <仓库>/workspace）。

    这里不 import app —— 存储层要能脱离 FastAPI 单独跑（测试/脚本直接 import）。
    """
    raw = str(os.environ.get("DFL_WORKSPACE") or "").strip()
    base = Path(raw) if raw else (ROOT / "workspace")
    try:
        return base.resolve()
    except OSError:
        return base


def default_project() -> Path:
    """没给 project 时的默认库：<工作区>/merge（它的 _faces/ 就是人脸库）。"""
    return workspace_root() / "merge"


def _resolve_project(raw, create: bool = False):
    """(项目目录, 是否这次新建)。

    * 空 / 只有空白 → 400
    * 任何一段是 `..` → 400（不做「先规范化再看落点」的宽容处理）
    * 相对路径按 4.1 仓库根解析（和 /api/merge/open 的 _norm_path 同规矩）
    * 目录不存在：create=False → 404；create=True → **只允许在工作区内新建**
      （越界一律 400，绝不 mkdir 到工作区外面去）
    """
    text = _text(raw)
    if not text:
        raise FacesError("缺少 project：要一个合成项目目录", 400)
    if "\x00" in text:
        raise FacesError("project 里有非法字符", 400)
    if _has_parent(text):
        raise FacesError("project 不能包含 ..：%r" % text, 400)
    try:
        p = Path(os.path.expandvars(os.path.expanduser(text)))
        if not p.is_absolute():
            p = ROOT / p
        p = Path(os.path.normpath(str(p)))
    except (OSError, ValueError) as e:
        raise FacesError("project 路径不合法：%r（%s）" % (text, e), 400)
    if not p.exists():
        if not create:
            raise FacesError("项目目录不存在：%s" % p, 404)
        root = workspace_root()
        try:
            target = p.resolve()
            target.relative_to(root)
        except (OSError, ValueError):
            raise FacesError("project 不在工作区里，不能自动创建：%s（工作区 %s）"
                             % (p, root), 400)
        try:
            target.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            raise FacesError("项目目录建不出来：%s（%s）" % (p, e), 500)
        return target, True
    if not p.is_dir():
        raise FacesError("project 不是目录：%s" % p, 400)
    return p, False


def resolve_project(raw) -> Path:
    """请求里的 project → 已存在的项目目录（不存在就 404）。"""
    return _resolve_project(raw, create=False)[0]


def resolve_library(raw) -> Path:
    """人脸库读写用的项目目录：**project 为空就落到默认库 <工作区>/merge**。

    视频模式下前端没有 DFL 项目，合成器点「人脸数据库」时传的就是空串；
    默认库还不存在也要能读（第一次进库看到的就是一份空库），所以这条路径
    带 create —— 但它只对默认库生效：显式给了 project 却不存在，仍然是 404，
    不替用户凭空造项目。
    """
    text = _text(raw)
    if not text:
        return _resolve_project(str(default_project()), create=True)[0]
    return _resolve_project(text, create=False)[0]


def resolve_project_for_write(raw, default=None):
    """写库用的 project → (目录, 是否新建)；缺省/空 → default（默认库）。"""
    text = _text(raw)
    if not text:
        if default is None:
            raise FacesError("缺少 project：要一个合成项目目录", 400)
        text = str(default)
    return _resolve_project(text, create=True)


def safe_join(base: Path, rel, what="路径") -> Path:
    """base/rel，且保证结果仍在 base 里面；绝对路径 / 盘符 / .. → 400。"""
    text = _text(rel)
    if not text:
        raise FacesError("%s 不能为空" % what, 400)
    if "\x00" in text:
        raise FacesError("%s 里有非法字符" % what, 400)
    if text.startswith(("/", "\\")):
        raise FacesError("%s 必须是相对路径（拒绝绝对路径）：%r" % (what, text), 400)
    rp = Path(text)
    if rp.is_absolute() or rp.drive:
        raise FacesError("%s 必须是相对路径（拒绝绝对路径）：%r" % (what, text), 400)
    if _has_parent(text):
        raise FacesError("%s 不能包含 ..：%r" % (what, text), 400)
    root = Path(base).resolve()
    target = (root / rp).resolve()
    try:
        target.relative_to(root)
    except ValueError:
        raise FacesError("%s 跑出项目目录了：%r" % (what, text), 400)
    return target


def faces_dir(project: Path) -> Path:
    return Path(project) / FACES_DIRNAME


def ensure_layout(project: Path) -> Path:
    d = faces_dir(project)
    (d / TRASH_DIRNAME).mkdir(parents=True, exist_ok=True)
    (d / THUMB_DIRNAME).mkdir(parents=True, exist_ok=True)
    return d


# ============================================================
# 原子写 / 文件锁
# ============================================================
def atomic_write(path: Path, data: bytes, tmp_name: str = "") -> Path:
    """先写同目录的 .part，再 os.replace；半截文件永远不会出现在正式名字上。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(tmp_name or (path.name + ".part"))
    try:
        with open(tmp, "wb") as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(str(tmp), str(path))
    finally:
        try:
            if tmp.exists():
                tmp.unlink()
        except OSError:
            pass
    return path


_LOCKS = {}
_LOCKS_GUARD = threading.Lock()


def _thread_lock(key: str) -> threading.RLock:
    with _LOCKS_GUARD:
        lock = _LOCKS.get(key)
        if lock is None:
            lock = threading.RLock()
            _LOCKS[key] = lock
        return lock


def _lock_fh(fh, timeout: float):
    end = time.monotonic() + max(0.1, float(timeout))
    if os.name == "nt":
        import msvcrt
        while True:
            try:
                fh.seek(0)
                msvcrt.locking(fh.fileno(), msvcrt.LK_NBLCK, 1)
                return
            except OSError:
                if time.monotonic() >= end:
                    raise FacesError("人脸库正忙（等锁超时），稍后重试", 503)
                time.sleep(0.02)
    else:                                     # pragma: no cover - Linux/macOS
        import fcntl
        while True:
            try:
                fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                return
            except OSError:
                if time.monotonic() >= end:
                    raise FacesError("人脸库正忙（等锁超时），稍后重试", 503)
                time.sleep(0.02)


def _unlock_fh(fh):
    try:
        if os.name == "nt":
            import msvcrt
            fh.seek(0)
            msvcrt.locking(fh.fileno(), msvcrt.LK_UNLCK, 1)
        else:                                 # pragma: no cover
            import fcntl
            fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
    except OSError:
        pass


@contextlib.contextmanager
def project_lock(project: Path, timeout: float = LOCK_TIMEOUT):
    """同一项目串行化所有 index 读写（进程内 RLock + 跨进程 .lock）。"""
    d = ensure_layout(project)
    key = os.path.normcase(str(Path(project).resolve()))
    tlock = _thread_lock(key)
    with tlock:
        fh = open(d / LOCK_NAME, "a+b")
        try:
            fh.seek(0, os.SEEK_END)
            if fh.tell() == 0:
                fh.write(b"\0")
                fh.flush()
            _lock_fh(fh, timeout)
            try:
                yield
            finally:
                _unlock_fh(fh)
        finally:
            fh.close()


# ============================================================
# index.json
# ============================================================
def empty_index() -> dict:
    return {"version": 1, "next_face": 1, "next_group": 1,
            "dedupe_version": 0, "groups": [], "faces": []}


def _face_num(fid: str) -> int:
    m = FACE_ID_RE.match(str(fid))
    return int(m.group(1)) if m else 0


def _group_num(gid: str) -> int:
    m = GROUP_ID_RE.match(str(gid))
    return int(m.group(1)) if m else 0


def normalize_index(raw) -> dict:
    """把磁盘上的 JSON 收敛成内部结构；坏数据报 500，绝不带着病继续写。"""
    if not isinstance(raw, dict):
        raise FacesError("index.json 不是 JSON 对象", 500)
    try:
        version = int(raw.get("version") or 1)
    except (TypeError, ValueError):
        version = 1

    groups_in = raw.get("groups")
    faces_in = raw.get("faces")
    groups_in = [] if groups_in is None else groups_in
    faces_in = [] if faces_in is None else faces_in
    if not isinstance(groups_in, list) or not isinstance(faces_in, list):
        raise FacesError("index.json 的 groups/faces 必须是数组", 500)

    groups, gids = [], set()
    for g in groups_in:
        if not isinstance(g, dict):
            raise FacesError("index.json 里有非对象的 group", 500)
        gid = _text(g.get("id"))
        if not GROUP_ID_RE.match(gid):
            raise FacesError("index.json 里有非法组 id：%r" % gid, 500)
        if gid in gids:
            raise FacesError("index.json 里组 id 重复：%s" % gid, 500)
        dfm = _text(g.get("dfm"))
        name = _text(g.get("name")) or dfm or gid
        item = {"id": gid, "dfm": dfm, "name": name}
        if g.get("orphan"):
            item["orphan"] = True
        groups.append(item)
        gids.add(gid)

    faces, fids = [], set()
    for f in faces_in:
        if not isinstance(f, dict):
            raise FacesError("index.json 里有非对象的 face", 500)
        fid = _text(f.get("id"))
        if not FACE_ID_RE.match(fid):
            raise FacesError("index.json 里有非法人脸 id：%r" % fid, 500)
        if fid in fids:
            raise FacesError("index.json 里人脸 id 重复：%s" % fid, 500)
        file_name = _text(f.get("file")) or (fid + ".png")
        # file 只允许基名：index 被手改成 "../../x.png" 时不能带着这个路径继续跑
        if ("/" in file_name or "\\" in file_name
                or Path(file_name).is_absolute() or Path(file_name).drive
                or _has_parent(file_name)):
            raise FacesError("人脸 %s 的 file 必须是 _faces 下的基名：%r"
                             % (fid, file_name), 500)
        group = _text(f.get("group"))
        if group and not GROUP_ID_RE.match(group):
            raise FacesError("人脸 %s 的 group 非法：%r" % (fid, group), 500)
        src = f.get("source")
        if not isinstance(src, dict):
            src = {}
        try:
            frame = int(src.get("frame") or 0)
        except (TypeError, ValueError):
            frame = 0
        try:
            added = float(f.get("added_at") or 0)
        except (TypeError, ValueError):
            added = 0.0
        try:
            idx = int(f.get("idx") or 0)
        except (TypeError, ValueError):
            idx = 0
        if idx < 0:
            idx = 0
        faces.append({
            "id": fid,
            "file": file_name,
            "group": group,
            "source": {"video": _text(src.get("video")), "frame": frame},
            "added_at": added,
            "trashed": bool(f.get("trashed")),
            # 这张脸是用几个点对齐的："68" = 68 点仿射；老库没这个键 = 早期的
            # 5 点对齐（尺寸一样、对齐略有差异），原样留着不改写。
            "align": (_text(f.get("align")) or ""),
            # 同一帧里的第几张脸（老库没这个键 = 0）：同帧去重键的一部分
            "idx": idx,
        })
        fids.add(fid)

    next_face = raw.get("next_face")
    try:
        next_face = int(next_face)
    except (TypeError, ValueError):
        next_face = 1
    next_face = max(1, next_face, max([_face_num(x) for x in fids] or [0]) + 1)

    next_group = raw.get("next_group")
    try:
        next_group = int(next_group)
    except (TypeError, ValueError):
        next_group = 1
    next_group = max(1, next_group, max([_group_num(x) for x in gids] or [0]) + 1)

    try:
        dedupe_version = int(raw.get("dedupe_version") or 0)
    except (TypeError, ValueError):
        dedupe_version = 0

    return {"version": version, "next_face": next_face, "next_group": next_group,
            "dedupe_version": max(0, dedupe_version),
            "groups": groups, "faces": faces}


def load_index(project: Path) -> dict:
    path = faces_dir(project) / INDEX_NAME
    if not path.is_file():
        return empty_index()
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        bak = faces_dir(project) / BACKUP_NAME
        try:
            backup = bak.read_bytes()
            raw = json.loads(backup.decode("utf-8"))
        except Exception:
            raise FacesError("index.json 读不出来，且没有可用的 index.json.bak", 500)
        # 主文件坏了、备份是好的：先把主文件修回来，别让下一次 save 把坏内容盖进 .bak
        try:
            atomic_write(path, backup)
        except OSError:
            pass
    return normalize_index(raw)


def ensure_index(project: Path) -> dict:
    """目录结构就位，且 _faces/index.json 一定存在（新库里写一份空清单）。

    自动建出来的库要能立刻被 GET /api/merge/faces 读到，否则第一个请求回来
    只有 "文件落盘了"、前端刷新还是「库不存在」。
    """
    d = ensure_layout(project)
    if not (d / INDEX_NAME).is_file():
        save_index(project, empty_index())
    return load_index(project)


def save_index(project: Path, index: dict):
    """旧 index 先落成 .bak，再把新内容 .part→replace。"""
    d = ensure_layout(project)
    path = d / INDEX_NAME
    if path.is_file():
        try:
            old = path.read_bytes()
            atomic_write(d / BACKUP_NAME, old)
        except OSError:
            pass
    blob = json.dumps(index, ensure_ascii=False, indent=2).encode("utf-8")
    atomic_write(path, blob)


# ============================================================
# 组 / 脸
# ============================================================
def _next_group_id(index: dict) -> str:
    used = [_group_num(g["id"]) for g in index["groups"]]
    n = max(int(index.get("next_group") or 1), max(used or [0]) + 1)
    index["next_group"] = n + 1
    return "g%d" % n


def ensure_group(index: dict, dfm: str) -> str:
    """按 dfm 名找组；没有就新建。大小写变体复用同一个 id（只更新显示名）。"""
    name = _text(dfm)
    if name.lower().endswith(".dfm"):
        name = name[:-4]
    if not name:
        return ""
    for g in index["groups"]:
        if g.get("dfm") == name:
            if not _text(g.get("name")):
                g["name"] = name
            g.pop("orphan", None)
            return g["id"]
    low = name.casefold()
    for g in index["groups"]:
        old = _text(g.get("dfm"))
        if old.casefold() == low:
            # 大小写变体：dfm 更新成新的写法；显示名只在它本来跟着 dfm 时才跟着改，
            # 用户自定义过的名字（name != 旧 dfm）保留。
            if _text(g.get("name")).casefold() == old.casefold():
                g["name"] = name
            g["dfm"] = name
            g.pop("orphan", None)
            return g["id"]
    gid = _next_group_id(index)
    index["groups"].append({"id": gid, "dfm": name, "name": name})
    return gid


def _face_path(project: Path, face: dict) -> Path:
    d = faces_dir(project)
    base = safe_join(d, face.get("file") or (face["id"] + ".png"), "人脸文件")
    if face.get("trashed"):
        return safe_join(d / TRASH_DIRNAME, base.name, "回收站文件")
    return base


def thumb_path(project: Path, face_id: str) -> Path:
    if not FACE_ID_RE.match(str(face_id or "")):
        raise FacesError("非法的人脸 id：%r" % face_id, 400)
    return safe_join(faces_dir(project) / THUMB_DIRNAME, face_id + ".jpg", "缩略图")


def emb_dir(project: Path) -> Path:
    """特征向量目录（_faces/emb/）。写的时候才建，读的时候不建。"""
    return faces_dir(project) / EMB_DIRNAME


def emb_path(project: Path, face_id: str) -> Path:
    """一张脸的特征文件：_faces/emb/<id>.npy（独立文件，不进 index）。

    为什么不塞进 index.json：512 维 float32 = 2 KB，1000 张脸就是 2 MB 的
    JSON 数字文本，每次 move / trash 都要重写一遍；独立 .npy 还能让
    「还没算过」= 文件不存在，判起来零成本。
    """
    if not FACE_ID_RE.match(str(face_id or "")):
        raise FacesError("非法的人脸 id：%r" % face_id, 400)
    return safe_join(emb_dir(project), face_id + ".npy", "特征向量")


def _imread(path: Path):
    import cv2
    import numpy as np
    try:
        buf = np.frombuffer(Path(path).read_bytes(), dtype=np.uint8)
        return cv2.imdecode(buf, cv2.IMREAD_COLOR)
    except Exception:
        return None


def _encode_png(img) -> bytes:
    import cv2
    ok, buf = cv2.imencode(".png", img, [int(cv2.IMWRITE_PNG_COMPRESSION), 3])
    if not ok:
        raise FacesError("PNG 编码失败", 500)
    return buf.tobytes()


def _new_face_id(index: dict) -> str:
    n = int(index.get("next_face") or 1)
    used = {_face_num(f["id"]) for f in index["faces"]}
    while n in used:
        n += 1
    index["next_face"] = n + 1
    return "f_%04d" % n


def _idx_val(raw) -> int:
    """「同一帧里的第几张脸」→ 0 起的整数；坏值一律当 0。"""
    try:
        n = int(raw)
    except (TypeError, ValueError):
        return 0
    return n if n >= 0 else 0


def _frame_key(face: dict):
    """同帧去重键：(视频名, 帧号, 同一帧里的第几张)。"""
    src = face.get("source") or {}
    try:
        frame = int(src.get("frame") or 0)
    except (TypeError, ValueError):
        frame = 0
    return (_text(src.get("video")), frame, _idx_val(face.get("idx")))


def _drop_thumb(project: Path, face_id: str):
    """丢掉缩略图缓存：同一个 id 的 PNG 被覆盖后必须重新生成。"""
    try:
        tp = thumb_path(project, face_id)
        if tp.is_file():
            tp.unlink()
    except (OSError, FacesError):
        pass


def add_faces(project: Path, crops: list, source: dict,
              dfm: str = "", group: str = "", align: str = "68", idxs=None):
    """把已经对齐好的人脸图写进库，返回 (记录列表, 实际 group id)。

    同一帧同一张脸只留一条：库里已有 (video, frame, idx) 相同、没进回收站的记录，
    就**覆盖**它 —— PNG 重写、added_at 刷新、id 与 group 都不动，记录带
    replaced=True；不再新增。旧版每点一次「＋」就多一条，一帧能堆 9 张。
    idxs 是每张脸在同一帧里的序号（0 起），缺省全 0（单脸帧 / 老调用方）。

    先落 PNG 再写 index：index 写失败会把这次新写的图删掉，不留孤儿文件。
    align 记的是「这张脸用几个点对齐的」：extract / add 现在都是 68 点仿射，
    老库里没有这个键的按 5 点看待（读回来是空串）。
    """
    if not crops:
        return [], group
    src = {
        "video": _text((source or {}).get("video")),
        "frame": int((source or {}).get("frame") or 0),
    }
    seq = list(idxs or [])
    d = ensure_layout(project)
    written = []
    with project_lock(project):
        index = load_index(project)
        if _text(dfm):
            group = ensure_group(index, dfm)
        elif group and not any(g["id"] == group for g in index["groups"]):
            raise FacesError("组不存在：%s" % group, 400)

        # 已有脸按同帧键建索引（历史重复留 added_at 最大的那条当「这一张」）
        by_key = {}
        for f in index["faces"]:
            if f.get("trashed"):
                continue
            k = _frame_key(f)
            cur = by_key.get(k)
            if cur is None or float(f.get("added_at") or 0) >= float(cur.get("added_at") or 0):
                by_key[k] = f

        records = []
        try:
            for n, crop in enumerate(crops):
                idx = _idx_val(seq[n] if n < len(seq) else 0)
                blob = _encode_png(crop)
                key = (src["video"], src["frame"], idx)
                old = by_key.get(key)
                if old is not None:
                    path = safe_join(d, old.get("file") or (old["id"] + ".png"), "人脸文件")
                    atomic_write(path, blob)
                    old["added_at"] = round(time.time(), 3)
                    old["align"] = _text(align)
                    old["idx"] = idx
                    _drop_thumb(project, old["id"])
                    rec = dict(old)
                    rec["replaced"] = True
                    records.append(rec)
                    continue
                fid = _new_face_id(index)
                file_name = fid + ".png"
                path = safe_join(d, file_name, "人脸文件")
                atomic_write(path, blob)
                written.append(path)
                rec = {
                    "id": fid,
                    "file": file_name,
                    "group": group or "",
                    "source": dict(src),
                    "added_at": round(time.time(), 3),
                    "trashed": False,
                    "align": _text(align),
                    "idx": idx,
                }
                index["faces"].append(rec)
                by_key[key] = rec
                records.append(dict(rec))
            save_index(project, index)
        except Exception:
            for p in written:
                try:
                    p.unlink()
                except OSError:
                    pass
            raise
    return records, (group or "")


def dedup_frames(project: Path) -> int:
    """同一帧同一张脸只留最新的一张，其余的搬进回收站（可还原）。

    旧版前端把「＋」当成「再加一张」，同一帧堆 9 条很常见。GET 列表时收敛**一次**
    就够（index 里 dedupe_version 记一笔，之后不再动）—— 否则用户从回收站还原的
    那张会在下一次 GET 又被「latest 优先」塞回去，还原等于没用。
    只改 trashed 标记 + 搬文件，绝不 unlink（数据安全铁律：删除权永远在用户手里）。
    """
    d = ensure_layout(project)
    moved = 0
    with project_lock(project):
        index = load_index(project)
        # 一次性清理：标记已经写过就再不动 index（否则用户还原的那张会被再塞回去）
        if int(index.get("dedupe_version") or 0) >= DEDUPE_VERSION:
            return 0
        buckets = {}
        for f in index["faces"]:
            if f.get("trashed"):
                continue
            buckets.setdefault(_frame_key(f), []).append(f)
        victims = []
        for lst in buckets.values():
            if len(lst) < 2:
                continue
            lst.sort(key=lambda f: (float(f.get("added_at") or 0), _face_num(f["id"])))
            victims.extend(lst[:-1])
        if not victims:
            index["dedupe_version"] = DEDUPE_VERSION
            save_index(project, index)
            return 0
        tdir = d / TRASH_DIRNAME
        tdir.mkdir(parents=True, exist_ok=True)
        for f in victims:
            src = _face_path(project, f)
            dst = safe_join(tdir, Path(f["file"]).name, "回收站文件")
            try:
                if src.is_file():
                    os.replace(str(src), str(dst))
            except OSError as e:
                raise FacesError("去重时移进回收站失败：%s（%s）" % (f["id"], e), 500)
            f["trashed"] = True
            moved += 1
        index["dedupe_version"] = DEDUPE_VERSION
        save_index(project, index)
    return moved


def _require_faces(index: dict, ids) -> list:
    wanted = []
    if not isinstance(ids, (list, tuple)):
        raise FacesError("ids 要是一个数组", 400)
    for raw in ids:
        fid = _text(raw)
        if not fid:
            continue
        if not FACE_ID_RE.match(fid):
            raise FacesError("非法的人脸 id：%r（只认 f_0001 这种，不能带路径）" % raw, 400)
        if fid not in wanted:
            wanted.append(fid)
    if not wanted:
        raise FacesError("ids 不能为空", 400)
    if len(wanted) > MAX_IDS:
        raise FacesError("一次最多操作 %d 张脸" % MAX_IDS, 400)
    by_id = {f["id"]: f for f in index["faces"]}
    missing = [fid for fid in wanted if fid not in by_id]
    if missing:
        raise FacesError("找不到这些人脸：%s" % "、".join(missing[:10])
                         + ("…" if len(missing) > 10 else ""), 404)
    return [by_id[fid] for fid in wanted]


def move_faces(project: Path, ids, group: str):
    target = _text(group)
    with project_lock(project):
        index = load_index(project)
        if target and not any(g["id"] == target for g in index["groups"]):
            raise FacesError("组不存在：%s" % target, 400)
        faces = _require_faces(index, ids)
        for f in faces:
            f["group"] = target
        save_index(project, index)
        return [dict(f) for f in faces]


def trash_faces(project: Path, ids):
    d = ensure_layout(project)
    with project_lock(project):
        index = load_index(project)
        faces = _require_faces(index, ids)
        trashed_dir = d / TRASH_DIRNAME
        trashed_dir.mkdir(parents=True, exist_ok=True)
        missing = []
        for f in faces:
            if f.get("trashed"):
                continue
            src = _face_path(project, f)
            dst = safe_join(trashed_dir, Path(f["file"]).name, "回收站文件")
            try:
                if src.is_file():
                    os.replace(str(src), str(dst))
                elif not dst.is_file():
                    missing.append(f["id"])
            except OSError as e:
                raise FacesError("移进回收站失败：%s（%s）" % (f["id"], e), 500)
            f["trashed"] = True
        save_index(project, index)
        return [dict(f) for f in faces], missing


def restore_faces(project: Path, ids):
    d = ensure_layout(project)
    with project_lock(project):
        index = load_index(project)
        faces = _require_faces(index, ids)
        missing = []
        # 先把所有目标检查完再动文件：否则第二张撞名时，第一张已经搬回去了但 index 没保存
        plans = []
        for f in faces:
            if not f.get("trashed"):
                continue
            src = _face_path(project, f)
            dst = safe_join(d, Path(f["file"]).name, "人脸文件")
            if src.is_file() and dst.exists():
                raise FacesError("还原目标已存在：%s" % dst.name, 409)
            plans.append((f, src, dst))
        for f, src, dst in plans:
            try:
                if src.is_file():
                    os.replace(str(src), str(dst))
                elif not dst.is_file():
                    missing.append(f["id"])
            except OSError as e:
                raise FacesError("从回收站还原失败：%s（%s）" % (f["id"], e), 500)
            f["trashed"] = False
        save_index(project, index)
        return [dict(f) for f in faces], missing


def delete_faces(project: Path, ids):
    d = ensure_layout(project)
    removed, missing_files = [], []
    with project_lock(project):
        index = load_index(project)
        faces = _require_faces(index, ids)
        drop = {f["id"] for f in faces}
        for f in faces:
            paths = (_face_path(project, dict(f, trashed=False)),
                     _face_path(project, dict(f, trashed=True)))
            for p in paths:
                try:
                    if p.is_file():
                        p.unlink()
                except OSError:
                    missing_files.append(f["id"])
            try:
                tp = thumb_path(project, f["id"])
                if tp.is_file():
                    tp.unlink()
            except (OSError, FacesError):
                pass
            try:
                ep = emb_path(project, f["id"])
                if ep.is_file():
                    ep.unlink()
            except (OSError, FacesError):
                pass
            removed.append(f["id"])
        index["faces"] = [f for f in index["faces"] if f["id"] not in drop]
        save_index(project, index)
    return removed, missing_files


def sync_groups(project: Path, dfms):
    """同步 .dfm 组：幂等；不再选的组有脸就 orphan=true 保留，没脸才删组。"""
    if not isinstance(dfms, (list, tuple)):
        raise FacesError("dfms 要是一个数组", 400)
    names = []
    for raw in dfms:
        name = _text(raw)
        if not name:
            continue
        if "/" in name or "\\" in name:
            raise FacesError("dfm 只能是模型名，不能带路径：%r" % raw, 400)
        if name.lower().endswith(".dfm"):
            name = name[:-4]
        if name not in names:
            names.append(name)
    with project_lock(project):
        index = load_index(project)
        selected = set()
        for name in names:
            selected.add(ensure_group(index, name))
        kept = []
        for g in index["groups"]:
            if g["id"] in selected:
                kept.append(g)
                continue
            has_faces = any(f.get("group") == g["id"] for f in index["faces"])
            if has_faces:
                g["orphan"] = True
                kept.append(g)
        index["groups"] = kept
        save_index(project, index)
        return snapshot_locked(project, index)


# ============================================================
# 读视图 / 缩略图
# ============================================================
def _public_face(face: dict) -> dict:
    return {
        "id": face["id"],
        "file": face["file"],
        "group": face["group"],
        "source": dict(face.get("source") or {"video": "", "frame": 0}),
        "added_at": face.get("added_at", 0),
        "trashed": bool(face.get("trashed")),
        "align": str(face.get("align") or ""),
        "idx": int(face.get("idx") or 0),
    }


def _stats(index: dict) -> dict:
    groups = index["groups"]
    gids = {g["id"] for g in groups}
    orphan_ids = {g["id"] for g in groups if g.get("orphan")}
    faces = index["faces"]
    active = [f for f in faces if not f.get("trashed")]
    trashed = [f for f in faces if f.get("trashed")]
    # 组 id 在 groups 里找不到的（手改 index 的脏数据）也算未分组，保证 grouped+ungrouped=active
    ungrouped = [f for f in active
                 if not f.get("group") or f.get("group") not in gids]
    grouped = [f for f in active if f.get("group") and f.get("group") in gids]
    return {
        "total": len(faces),
        "active": len(active),
        "trashed": len(trashed),
        "groups": len(groups),
        "grouped": len(grouped),
        "ungrouped": len(ungrouped),
        "orphan_groups": len(orphan_ids),
        "orphan_faces": sum(1 for f in active if f.get("group") in orphan_ids),
        "next_face": int(index.get("next_face") or 1),
    }


def _public_groups(index: dict) -> list:
    return [{"id": g["id"], "dfm": g.get("dfm", ""), "name": g.get("name", ""),
             "orphan": bool(g.get("orphan"))} for g in index["groups"]]


def snapshot_locked(project: Path, index: dict) -> dict:
    return {
        "groups": _public_groups(index),
        "faces": [_public_face(f) for f in index["faces"]],
        "stats": _stats(index),
    }


def snapshot(project: Path) -> dict:
    with project_lock(project):
        return snapshot_locked(project, load_index(project))


def _thumb_ready(tp: Path, face_id: str):
    """缓存已就绪 → etag；文件不在/是空的 → None（并发删除也不抛）。"""
    try:
        st = tp.stat()
    except OSError:
        return None
    if st.st_size <= 0:
        return None
    return '"%s-%d-%d"' % (face_id, st.st_size, st.st_mtime_ns)


def ensure_thumb(project: Path, face_id: str):
    """返回 (缩略图路径, etag)。首次才解码原图；之后直接命中 _thumb 缓存。"""
    tp = thumb_path(project, face_id)
    etag = _thumb_ready(tp, face_id)
    if etag:
        return tp, etag
    with project_lock(project):
        index = load_index(project)
        by_id = {f["id"]: f for f in index["faces"]}
        face = by_id.get(face_id)
        if face is None:
            raise FacesError("找不到这张脸：%s" % face_id, 404)
        etag = _thumb_ready(tp, face_id)
        if etag:
            return tp, etag
        src = _face_path(project, face)
        if not src.is_file():
            raise FacesError("人脸文件不见了：%s" % src.name, 404)
        img = _imread(src)
        if img is None:
            raise FacesError("人脸文件解码失败：%s" % src.name, 500)
        import cv2
        thumb = cv2.resize(img, (THUMB_SIZE, THUMB_SIZE), interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".jpg", thumb, [int(cv2.IMWRITE_JPEG_QUALITY), THUMB_QUALITY])
        if not ok:
            raise FacesError("缩略图编码失败", 500)
        tmp = "%s.%d.%d.part" % (tp.name, os.getpid(), threading.get_ident())
        atomic_write(tp, buf.tobytes(), tmp_name=tmp)
        st = tp.stat()
        return tp, '"%s-%d-%d"' % (face_id, st.st_size, st.st_mtime_ns)


# ============================================================
# DFM 输入尺寸（ONNX 输入张量的 H/W）
# ============================================================
_DFM_SIZE_CACHE = {}


def _pb_varint(buf, i: int, end: int):
    shift = 0
    val = 0
    while i < end:
        b = buf[i]
        i += 1
        val |= (b & 0x7F) << shift
        if not (b & 0x80):
            return val, i
        shift += 7
        if shift > 63:
            break
    raise ValueError("bad varint")


def _pb_fields(buf, start: int, end: int):
    """最小 protobuf 字段扫描：只 yield (字段号, wire, 值起点, 终点)。"""
    i = start
    while i < end:
        key, i = _pb_varint(buf, i, end)
        fn, wire = key >> 3, key & 7
        if wire == 0:
            val, i = _pb_varint(buf, i, end)
            yield fn, wire, val, i
        elif wire == 1:
            yield fn, wire, i, i + 8
            i += 8
        elif wire == 2:
            length, i = _pb_varint(buf, i, end)
            yield fn, wire, i, i + length
            i += length
        elif wire == 5:
            yield fn, wire, i, i + 4
            i += 4
        else:
            raise ValueError("unsupported wire type %d" % wire)


def _pb_dims(buf, start: int, end: int) -> list:
    dims = []
    for fn, wire, a, b in _pb_fields(buf, start, end):
        if fn != 1 or wire != 2:
            continue
        value = -1
        for f2, w2, a2, _b2 in _pb_fields(buf, a, b):
            if f2 == 1 and w2 == 0:
                value = int(a2)
        dims.append(value)
    return dims


def _pb_tensor_shape(buf, start: int, end: int):
    # 传进来的是 TensorShapeProto 本身：dim 是字段 1，直接取维度
    return _pb_dims(buf, start, end)


def _pb_type_shape(buf, start: int, end: int):
    for fn, wire, a, b in _pb_fields(buf, start, end):
        if fn == 1 and wire == 2:                 # TypeProto.tensor_type
            for f2, w2, a2, b2 in _pb_fields(buf, a, b):
                if f2 == 2 and w2 == 2:           # Tensor.shape
                    return _pb_tensor_shape(buf, a2, b2)
    return []


def _pb_value_info_shape(buf, start: int, end: int):
    for fn, wire, a, b in _pb_fields(buf, start, end):
        if fn == 2 and wire == 2:                 # ValueInfo.type
            return _pb_type_shape(buf, a, b)
    return []


def read_onnx_input_shape(path: Path):
    """读 .dfm（ONNX）第一个输入的维度；不加载权重，只扫 protobuf 头部。"""
    with open(path, "rb") as fh:
        if os.fstat(fh.fileno()).st_size < 16:
            return []
        mm = mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ)
        try:
            mv = memoryview(mm)
            try:
                graph = None
                for fn, wire, a, b in _pb_fields(mv, 0, len(mv)):
                    if fn == 7 and wire == 2:     # ModelProto.graph
                        graph = (a, b)
                        break
                if graph is None:
                    return []
                shapes = []
                for fn, wire, a, b in _pb_fields(mv, graph[0], graph[1]):
                    if fn == 11 and wire == 2:    # GraphProto.input
                        shapes.append(_pb_value_info_shape(mv, a, b))
                return shapes
            finally:
                mv.release()
        finally:
            mm.close()


def dfm_input_size(path, default: int = 512) -> int:
    """DFM 的输入尺寸：能读就用，读不到（或不是 ONNX）回 default。"""
    try:
        p = Path(path)
        st = p.stat()
        key = (os.path.normcase(str(p.resolve())), st.st_size, st.st_mtime_ns)
        if key in _DFM_SIZE_CACHE:
            return _DFM_SIZE_CACHE[key]
        shapes = read_onnx_input_shape(p)
        size = 0
        if shapes:
            dims = shapes[0] or []
            # NHWC: [N,H,W,C]；NCHW: [N,C,H,W]；无 batch: [C,H,W]
            cand = [int(d) for d in dims[1:] if isinstance(d, int) and int(d) > 16]
            if cand:
                size = max(cand)
        if 32 <= size <= 4096:
            _DFM_SIZE_CACHE[key] = int(size)
            return int(size)
    except Exception:
        pass
    return int(default)


# ============================================================
# 检测（复用服务器现有的 RetinaFace 检测器）
# ============================================================
def detect_faces_in_image(img, detector: str = "") -> list:
    """整帧检测，返回 [{box, kps, score}, …]；一帧多脸会全部返回。

    detector 为空 = 服务器默认（Tinymog）；传界面名（TinyMog / RetinaFace_10g）
    就用那一颗 —— 和 /api/merge/detect 同一个缓存实例，结果逐位相同。
    """
    from ..app import _get_detector
    det = _get_detector(detector or None)
    return det.detect(img) or []
