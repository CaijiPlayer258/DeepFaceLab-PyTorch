# TF 权重兼容（iperov 原版 → torch）

## 版本归属，别再说错

| 版本 | 内核 | 权重格式 |
|---|---|---|
| **iperov/DeepFaceLab 原版（2.0）** | TensorFlow | `.npy` |
| **本项目 4.0 ~ 4.1** | **torch（作者自研）** | `.pth` |

所以「兼容 TF 权重」= 把 iperov 原版那套 `.npy` 读进 torch 的 `Saveable`。
实现在 `app/tools/migrate/convert_dfl_tf_to_torch.py`，入口是模型卡片上的「转换」。

## 两条命名，都是 iperov 的

```
<stem>_weights.npy                       单文件版
<stem>_encoder.npy / _inter.npy /       分模块版
  _decoder_src.npy / _decoder_dst.npy
```

两者都是 TF 2.0 的权重，不要写成「4.0 后期」——4.0 已经是 torch 了。

## 转换必须按顺序，不要按名字猜

TF 的 `Saveable.load_weights` 是照 `get_weights()` 的顺序读的；
存盘时 pickle dict 的插入顺序 = 当时 `get_weights()` 的顺序。

```python
_ordered = list(tf_dict.values())
if len(_ordered) == len(weights):          # 数量一致才试
    for _w, _arr in zip(weights, _ordered):
        _copy_np_to_tensor(_w, _convert_array_to_shape(_arr, _w.shape))
    return len(weights), len(weights), [], [], []
```

**血泪教训**：4.1 早期版本严格按名字匹配（`encoder/conv1/weight:0` 这种），
名字对不上就退回形状兜底，结果 44 个张量只命中 16 个、12 个留随机值 ——
转出来的模型能加载、能训练，但换脸是花的。

改成顺序优先之后（`877eb46`）：`encoder 18/18`、`inter 4/4`、`decoder_src 44/44`、`decoder_dst 44/44`，
一条警告都没有。

## 读 npy 的三个坑

1. **`UnpicklingError: no persistent_load function`** —— TF 用 `np.save` 存 object 数组，
   直接 `pickle.loads` 读不了。要走 `np.load(path, allow_pickle=True)`。
2. **`'NpzFile' object has no attribute 'item'`** —— TF 的 `save_weights` 写的是 npz，
   要按 `arr.files` 拆成 dict。
3. **优化器动量形状对不上**（`(5,5,3,80) -> ()`）—— best-effort，跳过即可，
   权重已经写好了；缺动量只是继续训练时头几百 iter 收敛慢一点。

## 边界：转换 ≠ 导出

```
转换   TF .npy  →  torch .pth     convert_dfl_tf_to_torch.py
导出   torch .pth → .dfm          export_dfm.py
```

两个独立按钮。转换绝不能顺带生成 `.dfm`（那是几百 MB 的重复产物）。
