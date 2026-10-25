"""训练入口包装：跑 4.0 的 main.py train，摘掉它自动弹浏览器的行为。

4.0 的 Trainer 在模型加载完之后无条件做两件事：
  1. 起一个自己的 HTTP WebUI（默认 6789）
  2. 用 webbrowser 打开它
4.0 的 Trainer 在模型加载完之后会：
  1. 起一个 HTTP WebUI（127.0.0.1:6789）—— 【保留】，4.1 的预览图从这里拉
  2. 用 webbrowser 打开它 —— 【禁掉】，界面已经有 4.1 自己了

4.0 目录是只读的，改不了它的源码，所以在这里做猴补丁。
"""
from __future__ import annotations

# 4.0 的 main.py 第一件事就是 set_start_method("spawn")。
# 而我们为了打补丁要先 import torch（经 WebUI.page_trainer），
# torch 会抢先设成别的 context，之后 4.0 那句就抛 "context has already been set"。
# 所以这里在最开头先替它设好。
import multiprocessing as _mp

# 先装上"子进程一律不新建控制台窗口"的补丁：4.0 的 Trainer 会 spawn 预取进程、
# 合成管线会 spawn ffmpeg，没有这个补丁用户会看到黑框一闪一闪。
try:
    from ._nowin import install as _install_nowin
except ImportError:
    from _nowin import install as _install_nowin
_install_nowin()

try:
    _mp.set_start_method("spawn", force=True)
except Exception:
    pass
# 4.0 的 main.py 还会再调一次（不带 force），那时会抛 "context has already been set"。
# 把函数换成空操作，让它的那句变成无害的重复调用。
_mp.set_start_method = lambda *a, **k: None

import runpy
import sys
import webbrowser
from pathlib import Path

# 4.0 根目录统一从 _paths 解析（env DFL_ENGINE_ROOT > engine.txt > 同级目录 > 旧路径）。
# 这个文件既可能被当模块跑（server._dfl_train），也可能被当脚本跑，所以两种导入都留。
try:
    from ._paths import engine_root, engine_missing_hint
except ImportError:  # 直接 python _dfl_train.py 时没有包上下文
    from _paths import engine_root, engine_missing_hint

DFL_ROOT = engine_root()


def main() -> int:
    if not (DFL_ROOT / "main.py").is_file():
        print(engine_missing_hint(), flush=True)
        return 2
    if str(DFL_ROOT) not in sys.path:
        sys.path.insert(0, str(DFL_ROOT))

    # 1. 别弹浏览器
    for name in ("open", "open_new", "open_new_tab"):
        setattr(webbrowser, name, lambda *a, **k: False)
    try:
        import webbrowser as _wb
        _wb.get = lambda *a, **k: None
    except Exception:
        pass

    # 2. WebUI 服务【保留】——4.1 的训练页靠它拿实时预览图。
    #    它只监听 127.0.0.1，用户看不到那个界面；被禁掉的只有"自动打开浏览器"。
    #    （见上面 webbrowser 的补丁）
    # 3. 照常跑 4.0 的 main.py
    sys.argv = ["main.py"] + sys.argv[1:]
    try:
        runpy.run_path(str(DFL_ROOT / "main.py"), run_name="__main__")
    except SystemExit as e:
        return int(e.code or 0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
