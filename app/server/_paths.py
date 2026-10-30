"""4.0 引擎根目录的解析（训练 / 合成 / DFM 导出 / 检测模型都还靠它）。

4.1 只是个壳：真正的 python 环境、modelhub、MergeStudio、main.py 都在 4.0 里。
这些路径以前写死在 app.py 和 _dfl_train.py 里（`D:/MyProject/DeepFaceLab-PyTorch`），
换台机器、换个目录名，第一件事就是 404 / FileNotFoundError。现在统一从这里取：

  1. 环境变量 `DFL_ENGINE_ROOT`
  2. 项目根的 `engine.txt`（一行路径，和 workspace.txt 一个套路）
  3. 同级目录 `../DeepFaceLab-PyTorch`
  4. 旧硬编码路径（只为兼容现在的开发机，别在别处再抄它）

解析结果缓存在模块里，第一次用到时算一次。
"""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

_LEGACY = Path("D:/MyProject/DeepFaceLab-PyTorch")
_cache: dict = {"root": None}


def _looks_like_engine(p: Path) -> bool:
    """判定标准：有 main.py（训练/合成入口），或者有 Python 环境。"""
    if not p or not p.is_dir():
        return False
    return (p / "main.py").is_file() or (p / "python" / "python.exe").is_file()


def candidates():
    out = []
    # 引擎代码现在收在 app/ 里（core/samplelib/python/main.py 都在那儿）
    app_dir = ROOT / "app"
    if _looks_like_engine(app_dir):
        out.append(app_dir)
    if _looks_like_engine(ROOT):
        out.append(ROOT)
    env = os.environ.get("DFL_ENGINE_ROOT")
    if env:
        out.append(Path(env))
    txt = ROOT / "engine.txt"
    if txt.exists():
        try:
            line = txt.read_text(encoding="utf-8").strip().splitlines()[0].strip()
            if line:
                out.append(Path(line))
        except Exception:
            pass
    out.append(ROOT.parent / "DeepFaceLab-PyTorch")
    out.append(_LEGACY)
    return out


def engine_root() -> Path:
    """返回 4.0 根目录。找不到就返回最后那个候选（调用方自己判存在性）。"""
    if _cache["root"] is not None:
        return _cache["root"]
    cands = candidates()
    for p in cands:
        if _looks_like_engine(p):
            _cache["root"] = p
            return p
    _cache["root"] = cands[-1] if cands else _LEGACY
    return _cache["root"]


def engine_python() -> Path:
    """引擎运行时：runtime/engine/python.exe（旧的 app/python 也认）。"""
    for cand in (ROOT / "runtime" / "engine" / "python.exe",
                 engine_root() / "python" / "python.exe"):
        if cand.is_file():
            return cand
    return engine_root() / "python" / "python.exe"


def engine_missing_hint() -> str:
    """给用户看的说明（找不到引擎时拼在错误信息后面）。"""
    tried = "\n".join("  - %s" % c for c in candidates())
    return ("找不到 4.0 引擎目录。按顺序试过：\n%s\n"
            "可以在 4.1 根目录写一个 engine.txt（一行路径），或设环境变量 DFL_ENGINE_ROOT。"
            % tried)


def modelhub_dir() -> Path:
    """ONNX 权重目录：根 weights/（含 onnx/），其次 app/modelhub，最后引擎的。"""
    for cand in (ROOT / "weights", ROOT / "app" / "modelhub", engine_root() / "modelhub"):
        if (cand / "onnx").is_dir():
            return cand
    return engine_root() / "modelhub"
