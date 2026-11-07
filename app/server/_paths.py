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
    # 引擎代码收在 app/engine/（core / samplelib / main.py 都在那儿）
    engine_dir = ROOT / "app" / "engine"
    if _looks_like_engine(engine_dir):
        out.append(engine_dir)
    app_dir = ROOT / "app"          # 老布局：main.py 直接放在 app/ 下
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


def weights_dir() -> Path:
    """权重根目录：$DFL_WEIGHTS_DIR > <仓库根>/weights。

    代码与模型分离后，.onnx/.pth 一律放这里，不进版本库。
    **布局是平铺的**：weights/tinymog.onnx、weights/retinaface_10g.onnx……
    没有 onnx/<模型>/ 这种子目录；唯一保留的子目录是 bert-base-uncased/
    （AutoTokenizer.from_pretrained(dir) 要求词表和权重同目录）。

    $DFL_WEIGHTS_DIR 仍可指向别处；指向老的分目录布局也能用
    （modelhub/weights.py 的 LEGACY_LAYOUT 负责兜底），但默认路径是平铺。
    """
    env = os.environ.get("DFL_WEIGHTS_DIR")
    if env:
        return Path(env)
    return ROOT / "weights"


def modelhub_dir() -> Path:
    """modelhub **代码**目录（不是权重目录）。

    权重已经平铺到 <仓库根>/weights/ 根下，不再有 weights/onnx/ 这一层，
    所以这里只回答「modelhub 这个 python 包在哪」—— 4.1 自己那份优先。
    """
    for cand in (ROOT / "app" / "engine" / "modelhub", engine_root() / "modelhub"):
        if (cand / "onnx").is_dir():
            return cand
    return ROOT / "app" / "engine" / "modelhub"
