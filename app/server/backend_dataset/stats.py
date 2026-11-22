# -*- coding: utf-8 -*-
"""统计与状态：工作区计数、模型清单、端口、运行时、能力清单（从 app/server/app.py 切出）。

/api/state 回的就是这里的几份数据。函数体 / 注释逐字节照抄；只有包内 import 深度变了。
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from .._common.paths import ROOT, SETTINGS_FILE, WORKSPACE


# ---------------- 数据访问 ----------------
def _version() -> str:
    # 唯一来源是项目根的 version.txt（4.1-1122 这种）。
    # 以前这里读 app/docs/VERSION（不存在）再退回 desktop/package.json 的 4.1.0 ——
    # 于是关于页永远显示 4.1.0，和实际版本对不上。
    try:
        return (ROOT / "version.txt").read_text(encoding="utf-8-sig").strip().splitlines()[0].strip()
    except Exception:
        pass
    try:
        return (ROOT / "app" / "docs" / "..").resolve().joinpath("VERSION").read_text().strip()
    except Exception:
        pass
    try:
        pkg = json.loads((ROOT / "desktop" / "package.json").read_text(encoding="utf-8"))
        return str(pkg.get("version", "4.1.0"))
    except Exception:
        return "4.1.0"


# 目录计数缓存：key = 目录路径, value = (mtime_ns, 文件数)。
# data_dst/aligned 这种 4 万份对齐图的目录，扫一层就要 ~600ms；
# 而 /api/state 每次刷新都会问一遍 —— 登录后「进主页很卡」主要就是它。
# 用目录 mtime 做失效依据：直接子项增删一定会改父目录的 mtime，数一层正好对得上。
_COUNT_CACHE: dict = {}


def _count_images(folder: Path) -> int:
    """只数**这一层**的图片，不递归，也不数 _/. 开头的缓存文件。

    训/合成用的都是 aligned 根目录下的那批（DFL 只扫一层），
    以前这里用 rglob，分组之后根目录空了还照样报 1204 张，
    和训练器实际看到的 0 张对不上。
    现在口径统一到 _dataset_fast.list_images（唯一实现），这里只做转发。
    """
    if not folder.is_dir():
        return 0
    from .._dataset_fast import count_images
    key = str(folder)
    try:
        stamp = folder.stat().st_mtime_ns
    except OSError:
        return count_images(folder)
    hit = _COUNT_CACHE.get(key)
    if hit and hit[0] == stamp:
        return hit[1]
    n = count_images(folder)
    _COUNT_CACHE[key] = (stamp, n)
    return n


def _dataset(name: str) -> dict:
    root = WORKSPACE / name
    d = root / "aligned" if (root / "aligned").is_dir() else root
    return {
        "name": name,
        "path": str(root),
        "aligned": str(d),
        "exists": root.is_dir(),
        "count": _count_images(d) if d.is_dir() else 0,
        # 这个 count 数的是**哪个目录**：以前 UI 把它当"当前选中的目录"用，
        # 于是选了 data_src/1（670 张）也照样显示 data_src/aligned 的 1203 张。
        "count_dir": str(d) if d.is_dir() else "",
    }


def _models() -> list:
    md = WORKSPACE / "model"
    out = []
    if not md.is_dir():
        return out
    for dat in sorted(md.glob("*_data.dat")):
        stem = dat.name[: -len("_data.dat")]
        model_name, _, arch = stem.rpartition("_")
        if not model_name:
            model_name, arch = stem, ""
        dfm = md / (model_name + ".dfm")
        out.append({
            "name": model_name,
            "arch": arch.lower(),
            "iter": 0,
            "state": "已暂停",
            "export": "已导出 DFM" if dfm.exists() else "未导出",
            "has_export": dfm.exists(),
        })
    return out


def _ports() -> dict:
    defaults = {"mask": 8000, "merge": 8001, "angle": 8002}
    try:
        data = json.loads(SETTINGS_FILE.read_text(encoding="utf-8"))
        for k in list(defaults):
            if k in data:
                defaults[k] = int(data[k])
    except Exception:
        pass
    return defaults

def _port_free(port: int) -> bool:
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            s.bind(("127.0.0.1", int(port)))
            return True
        except OSError:
            return False


def _runtime() -> dict:
    import platform
    return {
        "python": platform.python_version(),
        "platform": platform.system() + " " + platform.release(),
        "port": int(os.environ.get("DFL_PORT", 0)),
        "workspace": str(WORKSPACE),
    }


def _ports_status() -> dict:
    p = _ports()
    cur = int(os.environ.get("DFL_PORT", 0))
    return {
        "mask": True if p["mask"] == cur else _port_free(p["mask"]),
        "merge": True if p["merge"] == cur else _port_free(p["merge"]),
        "angle": True if p["angle"] == cur else _port_free(p["angle"]),
    }

def _capabilities() -> dict:
    """可选能力的状态清单 —— 界面先问清楚再决定按钮灰不灰。

    缺模型不是故障：点了才报错，用户看到的是「坏了」；进页就知道「没装」，
    看到的是「装一下就好」。加新能力时在这里登记一条即可。
    """
    out = {}
    try:
        from .. import _errface
        out["errface"] = _errface.status()
    except Exception as e:                      # 探测自己炸了也要给出状态，不能拖垮 /api/state
        out["errface"] = {"capability": "errface", "available": False,
                          "reason": "能力探测失败：%s" % str(e)[:200], "reason_en": "",
                          "install": "", "install_en": "", "deps": {}}
    return out
