# 分辨率课程：迁移工具就位

## 架构层面的事实（已在代码里确认）

```
DeepFakeArchi.py:113   down1 = DownscaleBlock(..., n_downscales=4 if 't' not in opts else 5)
DeepFakeArchi.py:146   lowest_dense_res = resolution // (32 if 'd' in opts else 16)
```

encoder 的下采样层数是**常数**（4 或 5），与分辨率无关；分辨率只进入
`lowest_dense_res`，进而只影响 Inter 的三个张量（dense1 / dense2 / bias）。

所以跨档迁移的真实规模是：**卷积权重与它们全部的 Adam 矩原样带走，只有 3 个张量需要重采样。**

## 工具

`app/tools/bench/curriculum_transfer.py`

```bat
python app\tools\bench\curriculum_transfer.py ^
    --src workspace/_curriculum/model_128 ^
    --dst workspace/_curriculum/model_192 ^
    --src-res 128 --dst-res 192 --archi-opts udt --method blend --moment resample
```

做三件事：

| 对象 | 处理 |
|---|---|
| encoder / decoder 的 .pth | 逐字节复制（形状与分辨率无关） |
| inter 的 .pth | 调 `resize_inter`（dense1 伪逆、dense2/bias lanczos） |
| `*_opt.pth` | 按索引对齐：形状相同 → 复制；形状变化 → 用同一算子重采样（`--moment zero` 可改为置零） |

## 关于「优化器重置」这个头疼点

结论比预想的好：**不需要重置整个优化器**。

```
encoder / decoder 的 Adam 矩      形状不变 → 直接搬，热身为 0
Inter 的 6 个数组（3 张量 × 2 矩）  形状变化 → 用 pinv / lanczos 重采样
```

也就是说每档切换时，只有 Inter 的矩是“新”的；这部分恰好也是唯一需要换分辨率的部分，
两者在语义上是一致的。

## 仍未验证的部分

```
1) 迁移后的优化器文件能否被训练器直接读入而不报形状错误（需要一次真实迁移 + 起训练）
2) 矩重采样（resample）相对置零（zero）是否有实际收益
3) 迁移后第一段 loss 是否平滑（若出现尖峰，说明矩的量纲需要校正）
```

这三条都要跑真实训练才能回答，本轮没有跑到。