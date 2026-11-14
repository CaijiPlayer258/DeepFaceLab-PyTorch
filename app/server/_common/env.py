# -*- coding: utf-8 -*-
"""环境变量解析：_env_int / _env_secs（没设或写坏了就用默认值，不抛、不猜）。

从 app/server/routes_merge.py 按关注点原样切出来的（16741 行的单文件 → 按域分家），
函数体 / 注释 / 日志文案逐字节照抄，只补 import；同包之间的 import 放在文件末尾
（见文末「环上的依赖」），另有一处 __file__ 相对路径下标跟着目录层级改了。
旧行号 → 现文件 的对照表：app/server/backend_merge/MAPPING.md。
"""
from __future__ import annotations

import os


# ============================================================
# 单帧实时预览
#
# 换脸跑不了在这边：merge 引擎（MergeStudio / facelib / onnxruntime-gpu）装在
# 4.0 那份 python 里，只能在那个解释器里 import。每帧起一个子进程要 3~6 秒
# （import + 加载 DFM），拖动时间轴根本没法用 —— 所以这里养一个常驻 worker：
#
#   父进程（4.1 的 python，有 FastAPI）              子进程（4.0 的 python）
#   ─────────────────────────────────────           ─────────────────────────
#   取帧 grab / 检测 / 68 点（本地缓存复用）   ->    换脸 + 遮罩 + 合成（DFM 只加载一次）
#   JPEG 帧 ──── JSON line（stdin）────>            MergeMaskedFace(frame, lm68, cfg, pred)
#   <──── JSON line（stdout）──── JPEG
#
# 几个硬约定：
#   · 协议走 ASCII 转义的 JSON lines（模型名/路径可能是中文，不能让子进程的
#     stdin 按 cp936 解）；worker 里把 fd 1 换成 dup 出来的协议管道，引擎那些
#     非 UTF-8 的日志全部赶去 stderr，不会插进 JSON 中间。
#   · 一次只有一个请求在飞（_PreviewWorker._lock），所以不需要请求路由表。
#   · 单帧干活上限 15s；超时就杀掉整个 worker（ONNX 推理没法从外部打断，
#     杀进程是唯一可靠的取消），下次请求自动重启。冷启动/换模型的加载时间
#     单独算，不算进 15s（否则第一次点预览必然是 504）。
#   · worker 空闲 10 分钟自动退出；崩溃了下次请求自动重启。
# ============================================================
def _env_secs(name: str, default: float) -> float:
    """环境变量里的秒数（DFL_* 那套写法）：没给 / 不合法 / <=0 就用默认值。"""
    try:
        v = float(str(os.environ.get(name) or "").strip())
    except (TypeError, ValueError):
        return float(default)
    return v if v > 0 else float(default)
def _env_int(name, default=0):
    """环境变量里的整数；没设/写坏了就用 default（不抛、不猜）。"""
    raw = str(os.environ.get(name) or "").strip()
    try:
        return int(raw, 10)
    except (TypeError, ValueError):
        return int(default)
