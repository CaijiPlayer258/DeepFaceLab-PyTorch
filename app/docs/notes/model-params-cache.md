# 模型参数缓存约定（写 dat 必同步 JSON）

模型的训练参数存在 `<name>_data.dat` 里（pickle，含 loss_history，实测 122 MB）。
解析一次要 **2.8 s**，而参数页只要 70 来个键 —— 所以读路径统一走 JSON 边车。

## 文件

```
<name>_data.dat              权威数据（引擎读写，含 loss_history）
<name>_data.opt_cache.json   参数镜像：{ver, mtime, size, iter, options}   ← 4.1 读这个
<name>_data.meta_cache.json  4.0 Qt WebUI 的旧格式 {iter, options}      ← 只为本兼容而保留
```

## 硬约定

> **任何写 `_data.dat` 的地方，写完必须立刻调用 `_write_opt_cache(dat, iter, options)`。**

否则 `(mtime, size)` 指纹失配 → 下次读参数页退回到解 122 MB 的 dat（2.8 s），
更糟的是如果哪天有人放宽了指纹校验，界面就会拿着旧参数去覆盖新 dat。

### 现有的三个写入点（新增写入点时照抄）

| 位置 | 何时 | 做法 |
|---|---|---|
| `engine/models/ModelBase.py::_write_params_cache` | 训练保存权重 | 写 opt_cache + meta_cache |
| `server/backend_models/routes.py::model_config_set` | 参数页保存 | 写 dat 后调 `_write_opt_cache` |
| `server/backend_train/routes.py::train_start` | 开训前落盘参数 | 同上（失败不影响启动） |

## 读路径（不要自己 pickle dat）

```python
from server.backend_models.routes import _read_options_cached
info = _read_options_cached(dat)      # {"iter": int, "options": dict}
```

顺序：`opt_cache.json`（校验 ver/mtime/size）→ `meta_cache.json`（按时间比）→ 解析 dat 并写回。

实测：

```
冷启动首次（无任何缓存，旧模型）  2.832 s   ← 只发生一次，之后自动转为 JSON
命中 opt_cache                    0.0001 s
命中 meta_cache                   0.001 s
```

## 兼容

- 旧模型只有 dat：首次打开参数页慢一次，随后写出 opt_cache，不再慢。
- 旧模型有 meta_cache：直接复用，连那次都省了。
- 4.0 的 Qt WebUI 仍读 meta_cache，所以它继续写，不要删。
- JSON 只镜像 options + iter；`loss_history` 留在 dat 里（体积大、变更频繁）。

## 相关前端约定（同一条链上的另外两个坑）

```
1) 控件必须有 data-cfg="键名" —— 没有它，openCfg 回填扫不到、collectCfgPatch 也不报，
   表现是「改了刷新就回默认」。
2) 程序化赋值（回填 / 恢复默认 / 换模型）必须调 el.__ddSync() 同步主题化下拉的标签，
   否则值是 mt、界面显示 same。
3) 晚挂的行（openCfg 之后 80ms 才挂上来的补充块）靠 el.dataset.filled === "1" 判断
   是否已被 dat 填过；不要再用 v0 当判据，那样整批行会被跳过。
```