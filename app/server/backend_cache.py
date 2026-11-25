"""缓存清理：字节码 / workspace 运行期产物。

为什么要有这个：更新器只覆盖文件，**不清 __pycache__** —— 而 .pyc 的有效性判据是
「源文件 mtime + size」，copy2 保留 mtime，新旧两版撞上同一个时间戳时 Python 会继续跑
旧字节码。表现就是「关于页显示新版本号，但某个模块的行为还是旧的」，用户和开发者都
查不出原因（实测：compose.py 的尺寸守卫一直不生效，日志里连那行 print 都没有）。

同时把 workspace 下那些「删了会自动重建」的东西一起收进来 —— 它们散在根目录上，
用户看到一堆 _sess_bench.json / *merged.log 也不知道能不能删。

**绝不碰**用户资产：模型 .dfm / .pth、数据集、成片 mp4、.backup 之外的一切目录。
"""
from __future__ import annotations

import shutil
from pathlib import Path

from fastapi import APIRouter

from ._common import paths as _paths

router = APIRouter()

ROOT = Path(__file__).resolve().parents[2]      # 项目根
WORKSPACE = _paths.WORKSPACE


def _size(p: Path) -> int:
    """目录递归求和 / 文件直接取大小。读不到就当 0。"""
    try:
        if p.is_file():
            return p.stat().st_size
        total = 0
        for f in p.rglob("*"):
            try:
                if f.is_file():
                    total += f.stat().st_size
            except OSError:
                pass
        return total
    except OSError:
        return 0


def _targets() -> list:
    """[(key, 标签, 路径, 说明)] —— 每一项都是「删了会自动重建或本来就不该留」。"""
    out = [
        ("pycache", "字节码缓存", None,
         "所有 __pycache__/*.pyc。更新后必须清 —— 否则可能继续跑旧字节码"),
        ("ws_cache", "预览与实测缓存", WORKSPACE / ".cache",
         "预览 png、会话池实测、批大小记忆。删了下次会重新生成"),
        ("ws_log", "合成日志", None,
         "workspace 下的 *_merged.log（每次导出都会新写一个）"),
        ("ws_backup", "旧版备份", None,
         "更新时留下的 .backup-<时间戳> 目录（十几 MB 一份，只留最近一份就够）"),
        ("tmp_dfl", "流水线临时文件", None,
         "%TEMP% 下 dfl_ 开头的中间产物（正常结束会自己删，崩溃时会残留）"),
    ]
    return out


def _expand(key: str) -> list:
    """把一个 key 展开成具体的路径列表。"""
    if key == "pycache":
        try:
            return [p for p in ROOT.rglob("__pycache__") if p.is_dir()]
        except OSError:
            return []
    if key == "ws_cache":
        p = WORKSPACE / ".cache"
        return [p] if p.exists() else []
    if key == "ws_log":
        try:
            return [p for p in WORKSPACE.glob("*_merged.log") if p.is_file()]
        except OSError:
            return []
    if key == "ws_backup":
        try:
            return [p for p in WORKSPACE.glob(".backup-*") if p.is_dir()]
        except OSError:
            return []
    if key == "tmp_dfl":
        import tempfile
        tmp = Path(tempfile.gettempdir())
        try:
            return [p for p in tmp.glob("dfl_*")]
        except OSError:
            return []
    return []


def _scan() -> list:
    items = []
    for key, label, _p, note in _targets():
        paths = _expand(key)
        # 日志/备份/临时这几类是「多个小文件」：报数量比报路径有用
        n = len(paths)
        size = sum(_size(p) for p in paths)
        items.append({"key": key, "label": label, "note": note,
                      "count": n, "bytes": size,
                      "sample": [str(p.relative_to(ROOT)) if str(p).startswith(str(ROOT))
                                 else str(p) for p in paths[:3]]})
    return items


def _clean(keys) -> dict:
    """删掉选中的项，返回每项删了多少。"""
    want = list(keys) if keys else [k for k, _l, _p, _n in _targets()]
    dropped = []
    freed = 0
    per = []
    for key in want:
        paths = _expand(key)
        n = len(paths)
        size = sum(_size(p) for p in paths)
        ok = 0
        for p in paths:
            try:
                if p.is_dir():
                    shutil.rmtree(p, ignore_errors=True)
                else:
                    p.unlink(missing_ok=True)
                ok += 1
            except Exception:                                   # noqa: BLE001
                pass
        per.append({"key": key, "removed": ok, "total": n, "bytes": size})
        freed += size
        dropped.append(key)
    return {"ok": True, "keys": dropped, "items": per,
            "bytes": freed, "mb": round(freed / 1048576.0, 1)}


@router.get("/api/cache/scan")
def cache_scan():
    """看一眼有哪些缓存、各占多大。只读，不删任何东西。"""
    return {"ok": True, "items": _scan(),
            "root": str(ROOT), "workspace": str(WORKSPACE)}


@router.post("/api/cache/clean")
def cache_clean(body: dict = None):
    """清理缓存。body.keys 留空 = 全清。

    **不重启**：清 __pycache__ 只影响**下一次**启动的导入；当前进程已经在内存里的模块
    不会因为删了 .pyc 就变回旧的 —— 所以清完提示用户重启一次，才真正加载新代码。
    """
    body = body if isinstance(body, dict) else {}
    keys = body.get("keys")
    keys = [str(k) for k in keys] if isinstance(keys, list) else []
    r = _clean(keys)
    r["restart_required"] = "pycache" in r["keys"]
    print("[cache] 清理：%s（%.1f MB）" % (", ".join(r["keys"]) or "无", r["mb"]), flush=True)
    return r
