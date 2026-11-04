"""权重文件解析 —— 代码与模型分离。

这个仓库只分发**代码**：`modelhub/onnx/<模型>/*.py` 进 git。
**权重**（.onnx / .pth）由用户自己下载，放在仓库根的 `weights/` 下，
不进版本库（见 .gitignore 与 weights/manifest.json）。

查找顺序（第一个存在者胜出）：

1. 环境变量 `DFL_WEIGHTS_DIR` 指向的目录
2. `<仓库根>/weights`
3. 代码同目录（老布局 / 开发时把 .onnx 直接放在 .py 旁边）

用法：

    from modelhub import weights
    p = weights.path("onnx/RetinaFace/det_10g.onnx")   # 缺失时抛 WeightsMissing
    d = weights.dir("onnx/RetinaFace")                 # 目录，不检查存在

配套命令行：`python -m modelhub.fetch --list`（见 fetch.py）。
"""
from __future__ import annotations

import json
import os
from pathlib import Path

__all__ = ["WeightsMissing", "root", "path", "dir", "try_path", "manifest", "repo_root"]

_HERE = Path(__file__).resolve().parent          # app/engine/modelhub
_ENV = "DFL_WEIGHTS_DIR"


class WeightsMissing(FileNotFoundError):
    """权重文件缺失。消息里带下载/放置指引，不要用裸 FileNotFoundError。"""

    def __init__(self, rel: str, tried, hint: str = ""):
        self.rel = rel
        self.tried = list(tried)
        lines = ["缺少权重文件：%s" % rel, "已查找："]
        lines += ["  - %s" % t for t in self.tried]
        if hint:
            lines.append(hint)
        lines.append("下载：python -m modelhub.fetch --file %s" % rel)
        lines.append("或手动放到：%s" % (Path(tried[0]) / rel if tried else rel))
        super().__init__("\n".join(lines))


def repo_root() -> Path:
    """仓库根：app/engine/modelhub → 上溯三级。"""
    return _HERE.parents[2]


def root() -> Path:
    """权重根目录。"""
    env = os.environ.get(_ENV)
    if env:
        return Path(env).expanduser()
    return repo_root() / "weights"


def _candidates(rel: str):
    yield root() / rel
    yield _HERE / rel                       # 老布局：权重就在代码旁边


def try_path(rel: str):
    for c in _candidates(rel):
        if c.is_file():
            return c
    return None


def path(rel: str, hint: str = "") -> Path:
    p = try_path(rel)
    if p is None:
        raise WeightsMissing(rel, [str(c) for c in _candidates(rel)], hint)
    return p


def dir(rel: str) -> Path:
    """数据目录（不检查是否存在，用于输出/批量扫描）。"""
    d = root() / rel
    if not d.is_dir():
        legacy = _HERE / rel
        if legacy.is_dir() and any(legacy.iterdir()):
            return legacy
    return d


def manifest() -> dict:
    mf = root() / "manifest.json"
    if mf.is_file():
        return json.loads(mf.read_text(encoding="utf-8"))
    return {"version": 1, "files": []}
