"""数据集体检：**一次解码算出全部关键指标**。

设计动机：4.0 的 Sorter 有 12 种排序，每种都要把整个数据集重新解一遍 —— 用户为了
把糊片、坏图、重复图都挑出来，得跑三遍（每遍 40 分钟）。这里改成 One Shot：
每张图只解码一次，同时算出清晰度 / 梯度能量 / 曝光 / 纯色 / 重复，再由一次统一决策
给出"该删哪些"。4 万张在 8 核上约 1~2 分钟。

指标都是绝对的（不依赖数据集分布），只有清晰度阈值走分位数 —— 因为"糊"是相对的。
"""
from __future__ import annotations

import math
import os
import pickle
import multiprocessing as mp
from pathlib import Path

import cv2
import numpy as np

try:                                    # 4.0 的 python 装了 numba；4.1 的没有，退回 numpy
    from numba import njit as _njit
    HAVE_NUMBA = True
except Exception:                       # pragma: no cover
    HAVE_NUMBA = False

    def _njit(*a, **k):
        def deco(f):
            return f
        return deco if not a or callable(a[0]) else deco


_ANGLE_CACHE = {"loaded": False, "funcs": None}
_ANGLE_LOCK = __import__("threading").Lock()


def _angle_funcs():
    """一次性把 DFLIMG + 角度估计函数取回来。

    原来写成"每张图 import 一次 + 每次 Path(__file__).resolve()（0.27 ms）+
    每次 sys.path.insert（列表越插越长）"，纯属浪费。
    """
    # 注意：扫描是 8 线程并发的，惰性加载必须上锁 ——
    # 之前先置 loaded=True 再填 funcs，抢跑的线程会拿到 None 并**永久**认定没有角度功能，
    # 表现是"同一批图，多线程扫描时部分图没有姿态、单线程却全有"。
    with _ANGLE_LOCK:
        if _ANGLE_CACHE["loaded"]:
            return _ANGLE_CACHE["funcs"]
        _ANGLE_CACHE["loaded"] = True
        try:
            import sys as _sys
            _root = str(Path(__file__).resolve().parents[3])
            if _root not in _sys.path:
                _sys.path.append(_root)
            from DFLIMG import DFLIMG as _DFLIMG
            from facelib.LandmarksProcessor import estimate_pitch_yaw_roll as _epyr
            _ANGLE_CACHE["funcs"] = (_DFLIMG, _epyr)
        except Exception:
            _ANGLE_CACHE["funcs"] = None
        return _ANGLE_CACHE["funcs"]


def _read_bytes(path: str) -> bytes:
    with open(path, "rb") as f:
        return f.read()


def _decode_buf(buf: bytes):
    """从已经读进来的 bytes 解码，避免二次读盘。"""
    return cv2.imdecode(np.frombuffer(buf, dtype=np.uint8), cv2.IMREAD_COLOR)


@_njit(cache=True, fastmath=False, nogil=True)
def _stats_fused(gray, lap, gx, gy, flat_thr, clip_lo, clip_hi):
    """把 sharp / edge / noise / clip 四个指标的归约合成**一趟**遍历。

    原来这几个都是独立 numpy 表达式：abs(lap) 产生一份 1MB 临时数组、
    flat_mask 又一份 bool、abs_lap[flat_mask] 再做一次 gather —— 512² 上
    光这些临时数组就要 1.8 ms，比解码还贵。编译版只遍历一次、零分配。
    """
    h, w = gray.shape
    n = 0.0
    s1 = 0.0
    s2 = 0.0
    edge = 0.0
    gsum = 0.0
    gsum2 = 0.0
    abs_lap_flat = 0.0
    n_flat = 0.0
    n_clip = 0.0
    for r in range(h):
        for c in range(w):
            lv = lap[r, c]
            s1 += lv
            s2 += lv * lv
            g = gx[r, c]
            if g < 0.0:
                g = -g
            g2 = gy[r, c]
            if g2 < 0.0:
                g2 = -g2
            edge += g + g2
            if (g + g2) < flat_thr:
                av = lv if lv >= 0.0 else -lv
                abs_lap_flat += av
                n_flat += 1.0
            gv = gray[r, c]
            gsum += gv
            gsum2 += gv * gv
            if gv < clip_lo or gv > clip_hi:
                n_clip += 1.0
            n += 1.0
    mean = s1 / n
    var = s2 / n - mean * mean
    if var < 0.0:
        var = 0.0
    mean_abs = (abs_lap_flat / n_flat) if n_flat > n * 0.05 else (abs_lap_flat / n) * (n / max(1.0, n_flat))
    gmean = gsum / n
    gvar = gsum2 / n - gmean * gmean
    if gvar < 0.0:
        gvar = 0.0
    return var, edge / n, mean_abs, n_clip / n, gmean, gvar ** 0.5


def occlusion_of(dfl_dict, size: int, erode: int = 21):
    """遮挡程度 = 1 - (hull 内被 XSegLite 认作人脸的比例)。

    思路（用户提的，实测有效）：
      · landmarks 的凸包代表"这张脸应该在哪"——哪怕被手/话筒挡着，5 点也能推出来
      · XSegLite 的遮罩只覆盖**看得见的皮肤**，遮挡物不在里面
      · 两者相减 ≈ 被挡住的比例
    细节：
      · 只取眉线以下（landmarks 17~26 的平均 y）。眉线以上是额头/头发，
        XSeg 本来就不标，不砍掉会把它算成"遮挡"
      · hull 内缩 21px（256 尺度上约 8%），避免边缘一圈把下巴轮廓算成遮挡
    返回 0~1；缺 mask 或缺 landmarks 返回 None。
    """
    raw = (dfl_dict or {}).get("xseg_mask")
    lm = (dfl_dict or {}).get("landmarks")
    if raw is None or lm is None or len(lm) < 68:
        return None
    try:
        # 遮罩存法有几种：PNG 字节（bytes）、PNG 字节被包成 N×1 的 uint8 数组、
        # 或者直接就是 2D 遮罩数组。先按 PNG 试，失败再当数组用。
        if isinstance(raw, np.ndarray):
            blob = np.ascontiguousarray(raw).tobytes()
            m = cv2.imdecode(np.frombuffer(blob, np.uint8), cv2.IMREAD_UNCHANGED)
            if m is None:
                m = raw
        else:
            m = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_UNCHANGED)
        if m is None:
            return None
        if m.ndim == 3:
            m = m[..., 0]
        H, W = m.shape[:2]

        # landmarks 也有两种坐标：DFL 的 DFLJPG 存 0~1 归一化，4.1 提取器写的是像素。
        # 用"最大值是否 <= 1.5"来判断，比按来源分支可靠。
        pts_raw = np.array([[float(q[0]), float(q[1])] for q in lm[:68]], np.float32)
        if float(np.abs(pts_raw).max()) <= 1.5:
            pts_raw = pts_raw * float(size)
        sc = float(W) / float(size)
        pts = pts_raw * sc
        hull = cv2.convexHull(pts.astype(np.int32))
        hm = np.zeros((H, W), np.uint8)
        cv2.fillConvexPoly(hm, hull, 255)
        brow_y = float(pts[17:27, 1].mean())
        if brow_y > 0:
            hm[:int(brow_y), :] = 0
        hm = cv2.erode(hm, np.ones((erode, erode), np.uint8))
        area = int((hm > 0).sum())
        if area < 64:
            return None
        cov = int(((m > 127) & (hm > 0)).sum()) / float(area)
        return max(0.0, min(1.0, 1.0 - cov))
    except Exception:
        return None


def measure(path: str, on_small=None) -> dict:
    """单张图的全部指标。约 2~3 ms（512x512）。"""
    name = Path(path).name
    # 一次读盘：imdecode 和 DFLJPG 的 APP15 解析共用同一份 bytes
    buf = None
    try:
        buf = _read_bytes(path)
        img = _decode_buf(buf)
    except Exception:
        img = None
    if img is None:
        return {"name": name, "unreadable": True}

    # 顺手交一份 224 的缩略图给 IQA 阶段 —— 那边就不用再读盘+解码一遍（省 1.6 ms/张）
    if on_small is not None:
        try:
            on_small(name, cv2.resize(img, (224, 224), interpolation=cv2.INTER_LINEAR))
        except Exception:
            pass

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    # 清晰度 / 梯度能量 / 噪声 / 裁切：四段归约合成一趟（numba），没装 numba 走 numpy
    lap = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)

    if HAVE_NUMBA:
        sharp, edge, mean_abs_lap, clip, mean, std = _stats_fused(gray, lap, gx, gy, 12.0, 4, 251)
        noise = float(mean_abs_lap * np.sqrt(np.pi / 2.0) / 6.0)
    else:
        sharp = float(lap.var())
        edge = float(np.mean(np.abs(gx)) + np.mean(np.abs(gy)))
        flat_mask = (np.abs(gx) + np.abs(gy)) < 12.0
        abs_lap = np.abs(lap)
        if np.count_nonzero(flat_mask) > lap.size * 0.05:
            mean_abs_lap = float(abs_lap[flat_mask].mean())
        else:
            mean_abs_lap = float(abs_lap.mean())
        noise = float(mean_abs_lap * np.sqrt(np.pi / 2.0) / 6.0)
        clip = float(np.count_nonzero((gray < 4) | (gray > 251)) / gray.size)
        mean = float(gray.mean())
        std = float(gray.std())

    # 姿态角：DFLJPG 里已经存了 68 个对齐点，直接算，不用重新检测（约 0.25 ms）
    yaw = pitch = roll = None

    # 遮挡评分：自己解 APP15，不依赖 4.0 的 DFL 加载器（那个在引擎 python 里可能导不进来）
    occ = None
    try:
        _i = buf.find(b"\xff\xef")
        if _i >= 0:
            _ln = int.from_bytes(buf[_i + 2:_i + 4], "big")
            _meta = pickle.loads(buf[_i + 4:_i + 2 + _ln])
            occ = occlusion_of(_meta, int(w))
    except Exception:
        occ = None

    _a = _angle_funcs()
    if _a is not None:
        try:
            d, fn = _a[0].load(Path(path), loader_func=(lambda _fn: buf)), _a[1]
            if d is not None and d.has_data():
                # 优先用提取时写好的姿态（由原帧 5 点解出，已校到 DFL 的约定）。
                # 4.1 提取的 68 点是模板投影，再拿它算姿态只会得到常数 —— 所以有这个分支。
                raw = getattr(d, "dfl_dict", None) or {}
                if raw.get("yaw") is not None and raw.get("pitch") is not None:
                    # 存的就是角度，直接用（下面不再做 radians→degrees）
                    yaw = float(raw["yaw"]); pitch = float(raw["pitch"])
                    roll = raw.get("roll")
                    roll = float(roll) if roll is not None else None
                else:
                    _y, _p, _r = fn(d.get_landmarks(), size=int(w))
                    yaw, pitch, roll = (float(np.degrees(_y)), float(np.degrees(_p)),
                                        float(np.degrees(_r)))
        except Exception:
            yaw = pitch = roll = None

    # 64 位 pHash：32x32 -> DCT -> 左上 8x8 -> 与中位数比。比 dHash 更抗整体亮度变化
    small = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA)
    # 16x16 缩略图：哈希只用来找候选，真正判"重复"要看像素差
    thumb = cv2.resize(small, (16, 16), interpolation=cv2.INTER_AREA)
    dct = cv2.dct(small.astype(np.float32))[:8, :8].flatten()
    med = float(np.median(dct[1:]))
    hsh = 0
    for v in dct:
        hsh = (hsh << 1) | (1 if v > med else 0)

    return {
        "name": name, "unreadable": False,
        "sharp": round(sharp, 1), "edge": round(edge, 2),
        "mean": round(mean, 1), "std": round(std, 1), "clip": round(clip, 3),
        "noise": round(noise, 2),
        "w": int(w), "h": int(h), "hash": hsh,
        "yaw": None if yaw is None else round(float(yaw), 1),
        "pitch": None if pitch is None else round(float(pitch), 1),
        "roll": None if roll is None else round(float(roll), 1),
        "occ": None if occ is None else round(float(occ), 4),
        "thumb": thumb.tobytes(),
    }


# 姿态分桶：yaw 5 档 × pitch 3 档 = 最多 15 个桶。
# "稀缺角度保护"要的就是这份分布 —— 先看清楚数据缺什么，才谈得上保护。
YAW_BINS = ((-90, -30, "左大"), (-30, -15, "左小"), (-15, 15, "正面"),
            (15, 30, "右小"), (30, 90, "右大"))
PITCH_BINS = ((-90, -10, "低头"), (-10, 10, "平视"), (10, 90, "抬头"))


def bucket_of(yaw, pitch):
    """返回 (yaw_label, pitch_label)，角度缺失时给 (None, None)。"""
    if yaw is None or pitch is None:
        return None, None
    yl = pl = None
    for lo, hi, name in YAW_BINS:
        if lo <= yaw < hi:
            yl = name
            break
    for lo, hi, name in PITCH_BINS:
        if lo <= pitch < hi:
            pl = name
            break
    return yl, pl


# 三档强度。实测相邻帧 16x16 平均像素差 p10=0.8 / p25=1.3 / p50=2.0 / p90=13.3，
# pHash 汉明 p50=2 —— 所以阈值必须让用户选，不能替他决定：
#   严格  MAE<=1.5 & ham<=2  → 只删几乎一模一样的（默认）
#   标准  MAE<=2.5 & ham<=4  → 连拍 / 同姿态近似帧也删
#   宽松  MAE<=4.0 & ham<=6  → 保守训练集会砍掉一大半
LEVELS = {
    "strict": {"dup_mae": 1.5, "dup_hamming": 2, "blur_pct": 5.0},
    "normal": {"dup_mae": 2.5, "dup_hamming": 4, "blur_pct": 10.0},
    "loose":  {"dup_mae": 4.0, "dup_hamming": 6, "blur_pct": 20.0},
}


_POOL = None


def _worker_init():
    try:
        cv2.setNumThreads(1)     # 多进程下再开线程只会互相拖慢
    except Exception:
        pass


def scan(files, workers=0, progress=None, on_small=None):
    """并行跑完整个目录，返回逐张指标。progress(done, total) 会被频繁调用。"""
    global _POOL
    files = [str(f) for f in files]
    total = len(files)
    out = []
    if total == 0:
        return out

    n = workers or max(1, min(16, (mp.cpu_count() or 4)))
    n = max(1, min(n, total))

    def _one(f):
        return measure(f, on_small=on_small)

    if n == 1:
        for i, f in enumerate(files, 1):
            out.append(_one(f))
            if progress and i % 25 == 0:
                progress(i, total)
        if progress:
            progress(total, total)
        return out

    # 用线程而不是进程：①服务里 spawn 会重新 import 主模块（uvicorn 会被再拉一次）
    # ②imdecode / Laplacian / Sobel 都是 C++ 实现，会释放 GIL，实测 4 线程 ≈ 3 倍
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=n) as pool:
        for i, r in enumerate(pool.map(_one, files), 1):
            out.append(r)
            if progress and i % 25 == 0:
                progress(i, total)
    if progress:
        progress(total, total)
    return out


def plan(rows, blur_pct=10.0, dup_hamming=4, dup_mae=1.5,
         score_pct=5.0, bucket_floor=0):
    """一次决策：给每张图判定 keep / trash，附原因。"""
    bad = {"unreadable": [], "blur": [], "flat": [], "exposure": [], "occ": [], "dup": []}

    # 遮挡阈值：实测 600 张的分布是 P50=0、P90=0.017、max=0.27 —— 0.15 只切尾部。
    occ_thr = float(os.environ.get("DFL_OCC_THR") or 0.15)

    for r in rows:
        if r.get("unreadable"):
            bad["unreadable"].append(r["name"])

    ok = [r for r in rows if not r.get("unreadable")]

    # 清晰度阈值：分位数（"糊"是相对的，绝对阈值在不同来源上完全不通用）
    sharp_vals = np.array([r["sharp"] for r in ok], dtype=np.float32) if ok else np.array([0.0])
    thr_sharp = float(np.percentile(sharp_vals, blur_pct)) if ok else 0.0

    # ---- 三个维度各自打成 0~100（绝对刻度，跨数据集可比；分位数只用于删哪些）----
    # 清晰度：log10(sharp) 从 1.7（≈50，很糊）线性映射到 4.3（≈20000，很锐）
    def q_blur_of(sharp):
        v = (math.log10(max(1.0, sharp)) - 1.7) / (4.3 - 1.7)
        return max(0.0, min(100.0, 100.0 * v))

    # 噪点：噪声标准差 0 → 100，18 → 0（JPEG 压过的脸通常 1~6）
    def q_noise_of(noise):
        return max(0.0, min(100.0, 100.0 - 5.5 * noise))

    for r in ok:
        r["q_blur"] = round(q_blur_of(r["sharp"]), 1)
        r["q_noise"] = round(q_noise_of(r["noise"]), 1)
        # 曝光：偏离中灰越多、裁切越多，扣得越狠
        dev = abs(r["mean"] - 118.0) / 118.0
        r["q_exp"] = round(max(0.0, 100.0 * (1.0 - 1.6 * dev) - 250.0 * r["clip"]), 1)
        if r.get("occ") is not None:
            r["q_occ"] = round(100.0 * (1.0 - float(r["occ"])), 1)
        # 综合分：清晰度为主，噪声次之，曝光兜底。
        # 噪点会给 Laplacian 方差"加分"（高频能量上去了），所以 q_noise 低时要把
        # q_blur 的权重打折 —— 否则 σ=30 的噪图反而比 σ=16 分高。
        gate = min(1.0, r["q_noise"] / 30.0)
        # 只用确定性指标：清晰度为主、噪点次之、曝光兜底。
        # （曾经把 LIQE 的百分位按 35% 权重混进来，实测它惩罚的是光照而不是可用性，
        #   会把"暗光但清晰"的好脸判成最差，已移除。）
        r["q_total"] = round(0.55 * r["q_blur"] * gate + 0.30 * r["q_noise"] + 0.15 * r["q_exp"], 1)

    decisions = {}
    for r in ok:
        reasons = []
        if r["sharp"] <= thr_sharp:
            reasons.append("blur")
        if r["std"] < 6.0:
            reasons.append("flat")
        if r.get("occ") is not None and float(r["occ"]) >= occ_thr:
            reasons.append("occ")
        if r["mean"] < 20.0 or r["mean"] > 240.0 or r["clip"] > 0.4:
            reasons.append("exposure")
        decisions[r["name"]] = reasons

    # 去重：pHash 分桶检索 + 16x16 像素确认，**组内保留 q_total 最高的那张**。
    # 原来"先到先得"是错的：同一组近重复帧里第一张可能恰好是最糊的。
    # 4 万张两两比是 8 亿次，所以按 16 位分 4 段建索引：汉明距离 <= 3 时至少有一段相同。
    groups = []          # [{rep_hash, rep_thumb, members:[row, ...]}]
    bands = [dict() for _ in range(4)]

    for r in ok:
        if decisions[r["name"]]:      # 已经被模糊/纯色/曝光判掉的，不参与去重
            continue
        h = r["hash"]
        thumb = np.frombuffer(r["thumb"], dtype=np.uint8).astype(np.int16)
        cands = []
        if dup_hamming <= 3:
            for i in range(4):
                cands.extend(bands[i].get((h >> (i * 16)) & 0xFFFF, ()))
        else:
            cands.extend(bands[0].get("__all__", ()))

        hit = None
        for gi in cands:
            g = groups[gi]
            if bin(h ^ g["hash"]).count("1") > dup_hamming:
                continue
            if float(np.abs(thumb - g["thumb"]).mean()) <= dup_mae:
                hit = gi
                break

        if hit is None:
            gi = len(groups)
            groups.append({"hash": h, "thumb": thumb, "members": [r]})
            if dup_hamming <= 3:
                for i in range(4):
                    bands[i].setdefault((h >> (i * 16)) & 0xFFFF, []).append(gi)
            else:
                bands[0].setdefault("__all__", []).append(gi)
        else:
            groups[hit]["members"].append(r)

    for g in groups:
        if len(g["members"]) < 2:
            continue
        g["members"].sort(key=lambda x: x["q_total"], reverse=True)
        keep = g["members"][0]
        for r in g["members"][1:]:
            decisions[r["name"]].append("dup")
            bad["dup"].append(r["name"])
            r["dup_of"] = keep["name"]

    for r in ok:
        for why in decisions[r["name"]]:
            if why == "dup":
                continue
            bad[why].append(r["name"])

    # ---- 模型轨：规则没抓到、但综合分垫底的那一撮 ----
    # 规则轨（模糊/纯色/曝光/坏图/重复）是可解释的；模型轨补它看不见的
    # （锐但脏、压缩块、色带……）。只在"规则的幸存者"里挑，避免重复计数。
    bad.setdefault("score", [])
    survivors = [r for r in ok if not decisions[r["name"]]]
    if score_pct and survivors:
        k = int(round(len(survivors) * float(score_pct) / 100.0))
        for r in sorted(survivors, key=lambda x: x.get("q_total", 0))[:max(0, k)]:
            decisions[r["name"]].append("score")
            bad["score"].append(r["name"])

    # ---- 稀缺角度保护：某个人脸角度桶被砍得太狠时，把分数最高的几张捞回来 ----
    rescued = []
    if bucket_floor and bucket_floor > 0:
        by_bucket = {}
        for r in ok:
            key = bucket_of(r.get("yaw"), r.get("pitch"))
            by_bucket.setdefault(key, []).append(r)
        for key, rs in by_bucket.items():
            kept = [r for r in rs if not decisions[r["name"]]]
            need = int(bucket_floor) - len(kept)
            if need <= 0:
                continue
            # 只救"功能上还能用"的：模糊/曝光/重复不救，只救被模型轨否掉的
            cand = sorted([r for r in rs if decisions[r["name"]] == ["score"]],
                          key=lambda x: x.get("q_total", 0), reverse=True)
            for r in cand[:need]:
                decisions[r["name"]] = []
                bad["score"].remove(r["name"])
                rescued.append(r["name"])

    trash_set = set(n for lst in bad.values() for n in lst)

    # ---- 姿态分桶统计：这是"该不该做稀缺角度保护"的唯一依据 ----
    buckets = {}
    for r in ok:
        yl, pl = bucket_of(r.get("yaw"), r.get("pitch"))
        if yl is None:
            buckets.setdefault(("未知", "未知"), []).append(r)
        else:
            buckets.setdefault((yl, pl), []).append(r)
    bucket_stats = []
    for (yl, pl), rs in sorted(buckets.items(), key=lambda kv: -len(kv[1])):
        vals = [x["q_total"] for x in rs if "q_total" in x]
        kept = [x for x in rs if x["name"] not in trash_set]
        bucket_stats.append({
            "yaw": yl, "pitch": pl,
            "count": len(rs),
            "keep": len(kept),
            "trash": len(rs) - len(kept),
            "q_mean": round(float(np.mean(vals)), 1) if vals else 0.0,
            "q_max": round(float(np.max(vals)), 1) if vals else 0.0,
        })
    bucket_stats.sort(key=lambda b: -b["count"])

    trash = sorted(trash_set)
    avg = {}
    for key in ("q_total", "q_blur", "q_noise", "q_exp", "q_occ"):
        vals = [r.get(key) for r in ok if r.get(key) is not None]
        avg[key] = round(float(np.mean(vals)), 1) if vals else 0.0
    worst = sorted(ok, key=lambda r: r.get("q_total", 0))[:8]

    return {
        "total": len(rows),
        "avg": avg,
        "occ_thr": occ_thr,
        "occ_count": len(bad["occ"]),
        "occ_pct": round(100.0 * len(bad["occ"]) / max(1, len(ok)), 1),
        "worst": [{"name": r["name"], "q": r.get("q_total"),
                   "occ": None if r.get("occ") is None else round(float(r["occ"]), 3),
                   "blur": r.get("q_blur"), "noise": r.get("q_noise"),
                   "exp": r.get("q_exp")} for r in worst],
        "thr_sharp": round(thr_sharp, 1),
        "blur_pct": blur_pct,
        "dup_hamming": dup_hamming,
        "dup_mae": dup_mae,
        "counts": {k: len(v) for k, v in bad.items()},
        "trash": trash,
        "buckets": bucket_stats,
        "score_pct": score_pct,
        "bucket_floor": bucket_floor,
        "rescued": rescued,
        "trash_count": len(trash),
    }
