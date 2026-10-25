"""onnxruntime-gpu 的 CUDA 依赖准备。

4.1 自己的 python 里没有 torch，也就没有 cuDNN 9 / cuBLAS —— onnxruntime-gpu
建 CUDA session 时会因为找不到 DLL 而**静默退回 CPU**（只留一行 warning）。
引擎仓（DeepFaceLab-PyTorch）的 python 里有 torch，torch/lib 里带着 cuDNN 9，
所以这里按路径借用它的 DLL 目录。
"""
from __future__ import annotations

import os
from pathlib import Path

_CACHE = {"dir": None, "done": False}


def _candidate_roots():
    env = os.environ.get("DFL_ENGINE_ROOT")
    if env:
        yield Path(env)
    here = Path(__file__).resolve()
    # _ortcuda.py 在 <4.1>/app/server/ 下：parents[2] = 4.1 根，parents[3] = 它们的父目录
    for base in (here.parents[2], here.parents[3] if len(here.parents) > 3 else None):
        if base is None:
            continue
        yield base
        yield base / "DeepFaceLab-PyTorch"
    yield Path(r"D:\MyProject\DeepFaceLab-PyTorch")


def _local_torch_lib():
    """4.1 自己装了 torch 的话，直接用它自带的 cuDNN（版本自洽）。

    注意：**不要 import torch** 来找路径 —— 那个 import 要 2 秒多（加载 CUDA 运行时），
    而我们只是要一个目录字符串。按 sys.prefix 拼出来就够了。
    """
    import sys
    cand = Path(sys.prefix) / "Lib" / "site-packages" / "torch" / "lib"
    return cand if cand.is_dir() else None


def add_cuda_dll_dirs():
    """把引擎 torch/lib 加进 DLL 搜索路径；返回目录或 None（只做一次）。"""
    if _CACHE["done"]:
        return _CACHE["dir"]
    _CACHE["done"] = True

    # 本仓自己的 torch 优先（版本和 onnxruntime-gpu 更可能对得上），
    # 但只有它真的带 cuDNN 才算数 —— PyPI 上的 Windows torch 是 +cpu 构建，没有 cuDNN。
    added = []

    def _add(d):
        try:
            os.add_dll_directory(str(d))
        except Exception:
            pass
        os.environ["PATH"] = str(d) + os.pathsep + os.environ.get("PATH", "")
        added.append(str(d))

    _own = _local_torch_lib()
    if _own is not None and (_own / "cudnn64_9.dll").is_file():
        _add(_own)
        _CACHE["dir"] = str(_own)
        return _CACHE["dir"]

    for root in _candidate_roots():
        lib = root / "python" / "Lib" / "site-packages" / "torch" / "lib"
        try:
            if not (lib.is_dir() and (lib / "cudnn64_9.dll").is_file()):
                continue
        except OSError:
            continue
        _add(lib)
        _CACHE["dir"] = str(lib)
        return _CACHE["dir"]

    # 本仓 torch 就算没 cuDNN 也加进去：它只是没 GPU 支持，不影响别的
    if _own is not None and _own.is_dir() and not added:
        _add(_own)
    return added[0] if added else None


def providers(prefer_cuda: bool = True):
    """按可用性给出 provider 列表：有 CUDA 就用，否则 CPU。"""
    import onnxruntime as ort

    avail = ort.get_available_providers()
    if prefer_cuda and "CUDAExecutionProvider" in avail:
        add_cuda_dll_dirs()
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]
