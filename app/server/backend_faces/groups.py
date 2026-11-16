# -*- coding: utf-8 -*-
"""人脸库的「组 = dfm 模型」判定：特征存取 + 组均值 + 余弦匹配。

产品语义（用户定的，别再自己发明）：

    放在某个组里的人脸 = 那个组对应的 dfm 模型。
    杨幂组放人物 A 的几张脸、张元英组放人物 B 的几张脸 →
    A 的换脸结果是杨幂，B 的是张元英。**判定方式 = 人脸 embedding 对比**，
    看这张脸更像哪个组里的脸。

所以这里是三件事：

1. **存**：<库>/_faces/emb/<id>.npy，512 维 float32，L2 归一化（独立文件，
   路径规则见 _faces.emb_path）。老库没有文件 = 还没算，backfill() 会在
   GET /api/merge/faces 时惰性补算，算不出来的跳过，绝不挡住列表。
2. **组的代表向量**：组内所有脸的 embedding 的**均值**（归一化后）。均值比
   「离某一张脸最近」稳：3 张脸里有一张侧面糊掉的，均值只会被拉偏一点，
   而单张匹配会被那一张带跑。
3. **判定**：新脸向量 vs 每个组的均值 → 取余弦相似度最大的那个；
   >= 阈值（默认 0.35）才算命中，否则 matched=False —— 归错组比不归组更糟。

阈值 0.35 的依据见 DEFAULT_THRESHOLD 的注释。
"""
from __future__ import annotations

import os
import threading
from pathlib import Path

import numpy as np

from . import emb as emb_mod
from . import store

# ArcFace(arcface_mbf / w600k_mbf) 的余弦相似度实测分布：
#   同一人不同帧 0.45 ~ 0.85；不同人 0.00 ~ 0.30（同性别同年龄段更难分，能到 0.3）。
# 取 0.35 = 「不同人的上沿 + 一点余量」，宁可让用户手动拖一次，也不把 A 的脸塞进
# B 的组 —— 组错了就等于换错人，是产品语义里最贵的错误。
DEFAULT_THRESHOLD = 0.35
BACKFILL_PER_GET = 120          # 一次 GET 最多补算多少张脸（大库不卡住列表）
PROFILE_COMPUTE_LIMIT = 40      # 合成时临时补算的上限（预览要快）

_PROFILES = {}
_PROFILES_LOCK = threading.Lock()
_PROFILES_MAX = 8

# 「组均值」之外也会攒 embedding 派生的结果（帧级自动分配方案、一次导出里逐帧算过的
# 相似度）。它们不属于这个模块，但**清理时机和 _PROFILES 完全一样**，所以这里开一个
# 极小的登记口：别的模块 import 时把自己那一份 (clear, size) 挂上来 —— _face_group
# 不用反过来 import 它们，也就没有循环 import。
_EXTRA_CLEARERS = []            # [(名字, clear(), size() 或 None)]
_EXTRA_LOCK = threading.Lock()


def register_cache_clearer(fn, size=None, name: str = "") -> bool:
    """登记一份「embedding 派生」的进程内缓存（幂等）。

    fn    不带参数，清空自己那张表；
    size  可选，不带参数回当前条数（cache_state() 用它断言"确实是 0"）；
    name  只用于展示。
    """
    if not callable(fn):
        return False
    key = str(name or getattr(fn, "__name__", "extra"))
    with _EXTRA_LOCK:
        if any(n == key for n, _f, _s in _EXTRA_CLEARERS):
            return True
        _EXTRA_CLEARERS.append((key, fn, size if callable(size) else None))
    return True


def _cache_files(project=None) -> list:
    """磁盘上有没有「缓存文件」—— 按设计必须**一个都没有**。

    进程内的表（_PROFILES / 帧级方案）重启就空；落盘的只有两样东西，都**不是缓存**，
    这里也刻意不删、不算进来：
      · <库>/_faces/index.json    —— 人脸库清单（用户数据）
      · <库>/_faces/emb/<id>.npy  —— 库里每张脸的特征（用户数据；删了要重算）
    这里只扫"名字里带 cache"的 json：以前没有、以后也不该多出
    <库>/_faces/profiles_cache.json 这种跨进程复用的快照。
    """
    import glob as _glob
    out = []
    pats = []
    try:
        # 人脸库都在 <项目>/_faces/ 下面，所以只扫这一层 —— 工作区里别的
        # *cache*.json（比如训练用的 data.meta_cache.json）跟人脸库无关。
        pats.append(str(store.workspace_root() / "**" / FACES_DIRNAME / "**" / "*cache*.json"))
    except Exception:                                       # noqa: BLE001
        pass
    if project is not None:
        pats.append(str(Path(project) / "**" / "*cache*.json"))
    for pat in pats:
        try:
            for p in _glob.glob(pat, recursive=True):
                out.append(str(p))
        except Exception:                                   # noqa: BLE001
            continue
    return sorted(set(out))


def cache_state() -> dict:
    """当前进程内缓存的规模（给断言 / 内部接口用）。全 0 才算干净。"""
    with _PROFILES_LOCK:
        profiles = len(_PROFILES)
    extra = {}
    with _EXTRA_LOCK:
        entries = list(_EXTRA_CLEARERS)
    for name, _fn, size_fn in entries:
        try:
            extra[name] = int(size_fn()) if size_fn is not None else None
        except Exception:                                   # noqa: BLE001
            extra[name] = -1
    return {"profiles": int(profiles), "extra": extra, "files": _cache_files(),
            "threshold": float(DEFAULT_THRESHOLD)}


def clear_face_group_cache(reason: str = "") -> dict:
    """清空**进程内**的人脸 embedding / 组均值缓存，返回清理后的状态。

    只用两个时机（用户定的语义，别加别减）：
      1) serve 启动时一次（app.py 的 startup 钩子）—— 进程重启缓存本来就该是空的；
      2) 每一次导出 job 结束时一次（视频模式导出线程的 finally，成功/失败/取消
         都走这一条）—— 下一次导出重新算，不拿上一轮的结果。
    只清进程内的表，**绝不删** <库>/_faces 下的 index.json / emb/*.npy（那是库本身）。
    """
    with _PROFILES_LOCK:
        n_profiles = len(_PROFILES)
        _PROFILES.clear()
    with _EXTRA_LOCK:
        entries = list(_EXTRA_CLEARERS)
    cleared = {}
    for name, fn, _size in entries:
        try:
            fn()
            cleared[name] = 0
        except Exception as e:                              # noqa: BLE001
            cleared[name] = -1
            print("[face-cache] 清理 %s 失败：%s" % (name, str(e)[:160]), flush=True)
    state = cache_state()
    state["cleared_profiles"] = int(n_profiles)
    state["cleared_extra"] = cleared
    state["reason"] = str(reason or "")
    print("[face-cache] 清空（%s）：组均值 %d 条 · 其它 %s · 磁盘缓存文件 %d 个"
          % (state["reason"] or "手动", n_profiles,
             cleared or "无", len(state.get("files") or [])), flush=True)
    return state


# ============================================================
# 特征文件
# ============================================================
def _normalize(vec):
    v = np.asarray(vec, dtype=np.float32).reshape(-1)
    n = float(np.linalg.norm(v))
    if n > 1e-8:
        v = v / n
    return v.astype(np.float32, copy=False)


def read_vector(project: Path, face_id: str):
    """读一张脸的特征；没有/坏了都返回 None。"""
    try:
        path = store.emb_path(project, face_id)
    except store.FacesError:
        return None
    if not path.is_file():
        return None
    try:
        v = np.load(str(path), allow_pickle=False)
    except Exception:                                       # noqa: BLE001
        return None
    arr = np.asarray(v, dtype=np.float32).reshape(-1)
    if arr.size != emb_mod.EMB_DIM or not np.all(np.isfinite(arr)):
        return None
    return _normalize(arr)


def write_vector(project: Path, face_id: str, vec) -> bool:
    """原子写 <id>.npy（.part → os.replace）。失败返回 False，不抛。"""
    if vec is None:
        return False
    arr = np.asarray(vec, dtype=np.float32).reshape(-1)
    if arr.size != emb_mod.EMB_DIM or not np.all(np.isfinite(arr)):
        return False
    try:
        path = store.emb_path(project, face_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".part")
        with open(str(tmp), "wb") as fh:
            np.save(fh, _normalize(arr))
        os.replace(str(tmp), str(path))
        return True
    except Exception:                                       # noqa: BLE001
        return False


def drop_vector(project: Path, face_id: str) -> bool:
    try:
        path = store.emb_path(project, face_id)
    except store.FacesError:
        return False
    try:
        if path.is_file():
            path.unlink()
            return True
    except OSError:
        pass
    return False


def embed_crop(crop):
    """裁好的人脸图（BGR）→ 512 维向量；模型缺失/失败返回 None。"""
    return emb_mod.embed(crop)


def face_image(project: Path, face: dict):
    """人脸记录 → BGR 图（回收站里的也能读）。"""
    try:
        path = store._face_path(project, face)
    except store.FacesError:
        return None
    if not path.is_file():
        return None
    return store._imread(path)


def face_vector(project: Path, face: dict, compute: bool = True, save: bool = True):
    """一张脸的特征：先看文件，没有就（可选）从 PNG 现算并落盘。"""
    fid = str(face.get("id") or "")
    if not fid:
        return None
    vec = read_vector(project, fid)
    if vec is not None:
        return vec
    if not compute:
        return None
    img = face_image(project, face)
    if img is None:
        return None
    vec = emb_mod.embed(img)
    if vec is None:
        return None
    if save:
        write_vector(project, fid, vec)
    return vec


def backfill(project: Path, limit: int = BACKFILL_PER_GET) -> dict:
    """老库补算：给还缺特征的脸补上 .npy。算不出来的跳过，不阻塞、不报错。

    返回 {"total", "have", "computed", "failed", "pending"}：
      total     活跃脸总数（回收站里的不算）
      have      现在有特征文件的脸数
      computed  这一次真的算了几张
      failed    这一次尝试失败了几张（PNG 缺失/解码失败/模型缺失）
      pending   还差多少张没算（超过 limit 的那部分）
    """
    out = {"total": 0, "have": 0, "computed": 0, "failed": 0, "pending": 0}
    try:
        index = store.load_index(project)
    except store.FacesError:
        return out
    missing = []
    for f in index["faces"]:
        if f.get("trashed"):
            continue
        out["total"] += 1
        try:
            exists = store.emb_path(project, f["id"]).is_file()
        except store.FacesError:
            exists = False
        if exists:
            out["have"] += 1
        else:
            missing.append(f)
    if not missing:
        return out
    limit = max(0, int(limit))
    for f in missing[:limit]:
        vec = face_vector(project, f)
        if vec is None:
            out["failed"] += 1
        else:
            out["computed"] += 1
            out["have"] += 1
    out["pending"] = max(0, len(missing) - limit)
    return out


# ============================================================
# 组的代表向量
# ============================================================
def library_signature(project: Path) -> str:
    """库的「身份」：index.json + emb/ 目录的 mtime/size。

    用来判缓存（组均值）与新不新。新增一张脸一定动 index.json；
    补算一张老脸一定动 emb/ 的目录 mtime —— 两者都不动时，均值也不会变。
    """
    parts = []
    for p in (store.faces_dir(project) / store.INDEX_NAME, store.emb_dir(project)):
        try:
            st = p.stat()
            parts.append("%d:%d" % (int(st.st_mtime_ns), int(st.st_size)))
        except OSError:
            parts.append("-")
    return "%s|%s" % (str(project), "/".join(parts))


def _profiles_uncached(project: Path, index: dict, compute_limit: int) -> list:
    groups = {}
    for g in index["groups"]:
        if not str(g.get("dfm") or "").strip():
            continue
        groups[g["id"]] = g
    buckets = {}
    computed = 0
    for f in index["faces"]:
        if f.get("trashed"):
            continue
        gid = str(f.get("group") or "")
        if gid not in groups:
            continue
        vec = face_vector(project, f, compute=(computed < compute_limit))
        if vec is None:
            continue
        buckets.setdefault(gid, []).append(vec)
        computed += 1
    out = []
    for gid in sorted(buckets, key=store._group_num):
        mats = np.stack(buckets[gid])
        mean = _normalize(mats.mean(axis=0))
        g = groups[gid]
        out.append({"group": gid, "dfm": str(g.get("dfm") or ""),
                    "name": str(g.get("name") or g.get("dfm") or gid),
                    "n": int(mats.shape[0]), "vec": mean,
                    "orphan": bool(g.get("orphan"))})
    return out


def group_profiles(project: Path, index: dict = None,
                   compute_limit: int = PROFILE_COMPUTE_LIMIT) -> list:
    """每个组一个均值向量；组顺序按 g1/g2/… 排。带进程内缓存（按库签名失效）。"""
    if index is None:
        try:
            index = store.load_index(project)
        except store.FacesError:
            return []
    sig = library_signature(project)
    with _PROFILES_LOCK:
        hit = _PROFILES.get(sig)
    if hit is not None:
        return hit
    out = _profiles_uncached(project, index, compute_limit)
    with _PROFILES_LOCK:
        _PROFILES[sig] = out
        while len(_PROFILES) > _PROFILES_MAX:
            _PROFILES.pop(next(iter(_PROFILES)))
    return out


def group_id_for_dfm(index: dict, dfm: str) -> str:
    """dfm 名 → 组 id（大小写/带不带 .dfm 都认）。找不到返回空串。"""
    name = str(dfm or "").strip()
    if name.lower().endswith(".dfm"):
        name = name[:-4]
    if not name:
        return ""
    low = name.casefold()
    for g in index["groups"]:
        if str(g.get("dfm") or "").casefold() == low:
            return g["id"]
    return ""


# ============================================================
# 匹配 / 落库决策
# ============================================================
def match_vector(vec, profiles: list, threshold: float) -> dict:
    """向量 vs 一组组均值 → 最像的那个组 + 是否过阈值。"""
    best = None
    for prof in profiles or []:
        try:
            sim = emb_mod.cosine(vec, prof["vec"])
        except Exception:                                   # noqa: BLE001
            continue
        if best is None or sim > best["similarity"]:
            best = {"group": prof["group"], "dfm": prof["dfm"],
                    "name": prof["name"], "similarity": round(float(sim), 4),
                    "n": int(prof.get("n") or 0)}
    if best is None:
        return {"group": "", "dfm": "", "name": "", "similarity": None,
                "matched": False, "threshold": float(threshold)}
    best["matched"] = bool(best["similarity"] >= float(threshold))
    best["threshold"] = float(threshold)
    return best


def decide_for_add(project: Path, vec, explicit_group: str = "",
                   chip_dfm: str = "",
                   threshold: float = DEFAULT_THRESHOLD) -> dict:
    """新脸该落进哪个组。返回 add 接口要回给前端的那一份判定。

    顺序（用户的语义，按这个实现）：
      1. 显式给了 group → 就用它（脚本/前端明确指定，不再猜）；
      2. 库里有脸 → 和每个组的均值比余弦，最大且 >= 阈值 → 归该组（matched=True）；
      3. 不到阈值 → 若这张脸在合成器上分了 dfm（chip 有值）→ 交给 chip 那个组；
      4. 都没有 → 不归组（"", matched=False），前端提示手动拖。

    source 说明这一条是谁定的：explicit / auto / chip / none。
    """
    thr = float(threshold)
    gid = str(explicit_group or "").strip()
    result = {"group": "", "group_name": "", "dfm": "", "similarity": None,
              "matched": False, "threshold": thr, "source": "none",
              "best_group": "", "best_dfm": "", "best_name": "",
              "best_similarity": None}
    try:
        index = store.load_index(project)
    except store.FacesError:
        index = store.empty_index()

    profiles = group_profiles(project, index)
    best = match_vector(vec, profiles, thr) if vec is not None else None
    if best is not None:
        result["similarity"] = best.get("similarity")
        result["best_group"] = best.get("group") or ""
        result["best_dfm"] = best.get("dfm") or ""
        result["best_name"] = best.get("name") or ""
        result["best_similarity"] = best.get("similarity")
        result["matched"] = bool(best.get("matched"))

    if gid:
        g = next((x for x in index["groups"] if x["id"] == gid), None)
        if g is None:
            raise store.FacesError("组不存在：%s" % gid, 400)
        result.update({"group": gid, "dfm": "", "source": "explicit",
                       "group_name": str(g.get("name") or g.get("dfm") or gid)})
        return result

    if best is not None and best.get("matched") and best.get("group"):
        result.update({"group": best["group"], "dfm": "", "source": "auto",
                       "group_name": best.get("name") or "",
                       "similarity": best.get("similarity")})
        return result

    chip = str(chip_dfm or "").strip()
    if chip:
        result.update({"group": "", "dfm": chip, "source": "chip",
                       "group_name": chip})
        return result

    return result
