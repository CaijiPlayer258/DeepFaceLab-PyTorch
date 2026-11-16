# backend_mask 行号对照表（结构重构，未改行为）

旧文件 `app/server/routes_mask.py`（1708 行）和 `app/server/_occlusion.py`（351 行）现在只剩
PEP 562 惰性代理；真身在这个包里。下表左边是**拆分前**的行号，右边是新位置。

函数体逐字搬过去，只改了两类机械依赖（其余一字未动，diff 里只有这些行）：

1. 相对 import 深度 +1：`.app` → `..app`、`._paths` → `.._paths`、
   `._align` → `.._vision.align`（`_align` 现在只是 `_vision/align` 的代理，指真身）
2. 文件下沉一层 → `Path(__file__).parents[N]` 的 N +1：
   仓库根 `parents[2]`→`[3]`、`app/engine` 的 `parents[1]`→`[2]`（返回值不变）

## routes_mask.py → backend_mask/{routes,painter}.py

| 旧 routes_mask.py | 新位置 | 内容 |
| --- | --- | --- |
| 1–29 | painter.py 1–39 / routes.py 1–33 | 头部 docstring 与 import 重新分配；`router = APIRouter()` 留在 routes.py |
| 31–96 | painter.py 40–105 | 错误码契约：`_MASK_STATUS` / `_HTTP_CODE` / `MaskErr` / `_mask_error_body` / `_mask_api` |
| 99–382 | painter.py 108–391 | 遮罩落盘 / 读回：`_MASK_SUBDIR` … `_load_mask_b64`（`_masks` PNG + DFLJPG APP15） |
| 385–591 | routes.py 35–241 | 9 个路由：project/open、image/{index}、mask/predict·text·bisenet、model/preload、image/{index}/mask、progress/save·load、mask/save、mask/models |
| 594–1596 | painter.py 394–1396 | SAM2 / GroundingDINO：常量、`_root_dir`/`_engine_dir`/`_weights_root`/`_engine_python`、`_WORKER_SRC`、`_MaskInfer`、`_raise_infer`、`_call_or_504` |
| 1599–1708 | routes.py 245–353 | 4 个路由：mask/sam/load、mask/sam/predict、mask/gdino、mask/sam/status |

搬走时**没有**跟着过去的名字（拆分前也是 0 引用，属死 import）：
`re`、`Response`、`FileResponse`、`StreamingResponse`。

## _occlusion.py → backend_mask/occlusion.py

| 旧 _occlusion.py | 新位置 | 说明 |
| --- | --- | --- |
| 1–35 | occlusion.py 1–41 | docstring 多了 6 行搬迁说明（第 4–7 行那块），口径正文一字未动 |
| 36–351 | occlusion.py 42–357 | 算法原样：`HULL_IDX` … `write_manifest` |

唯一代码改动：旧 61 行 `from .app import _safe_loads as _f` → 新 66 行 `from ..app import ...`。

## 旧路径 shim

| 旧路径 | 现在 |
| --- | --- |
| `app/server/routes_mask.py` | PEP 562 惰性代理 → `backend_mask.routes`（找不到再问 `backend_mask.painter`） |
| `app/server/_occlusion.py` | PEP 562 惰性代理 → `backend_mask.occlusion` |

`app.py` 里 `from .routes_mask import router as _mask_router`（3 处）；
`routes_oneshot.py` 里 `from . import _occlusion as OCC` + `OCC.scan(...)`：都不用改。
