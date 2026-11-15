# -*- coding: utf-8 -*-
"""路径常量：界面静态目录 / 仓库根 / 工作区 / settings.json / 4.0 引擎路径。

原来全在 app/server/app.py 顶部。搬到 app/server/_common/ 之后目录深了一层，
所以 __file__.parents[N] 的下标跟着 +1（app.py 的 parents[2] → 这里的 parents[3]）。

**这里是唯一来源**：app.py 只 import 再转出去，别的模块一律从这儿取。
"""
from __future__ import annotations

import os
from pathlib import Path

from .engine_paths import engine_python, engine_root

# 老名字（原 app.py 里叫 _engine_python / _engine_root，别的模块按这个名取）
_engine_python = engine_python
_engine_root = engine_root


WEB = Path(__file__).resolve().parents[2] / "web"
# 本仓库根目录（DeepFaceLab-PyTorch-4.1）：界面上不带盘符的路径都相对它解析
_APP_ROOT = Path(__file__).resolve().parents[3]
ROOT = Path(__file__).resolve().parents[3]
# 工作区可以用环境变量覆盖（迁移旧工程 / 多工程切换）
WORKSPACE = Path(os.environ.get("DFL_WORKSPACE") or (ROOT / "workspace"))

# 工作区只有一个来源：环境变量 DFL_WORKSPACE，否则就是本仓库的 workspace。
# （项目根的 workspace.txt 间接覆盖机制已取消——路径就是路径，不做隐藏跳转。）
SETTINGS_FILE = WORKSPACE / "settings.json"

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

# 4.0 引擎（训练 / 合成 / DFM 导出 / 检测模型都还靠它）：解析结果在 _paths 里缓存
_DFL_ROOT = engine_root()
_DFL_PY = engine_python()
_DFL_TOOL = _DFL_ROOT / "tools" / "export_dfm.py"
