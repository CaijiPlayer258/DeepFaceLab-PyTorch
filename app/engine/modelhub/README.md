# ModelHub

模型代码与权重分离：仓库只带**代码**，权重放在仓库根的 `weights/`，不进版本库。

## 布局

```
app/engine/modelhub/
├── onnx/            # 各模型的推理封装（*.py，随代码入库）
├── weights.py       # 权重路径解析（平铺布局 + manifest）
├── fetch.py         # 按 weights/manifest.json 下载 / 校验
└── __init__.py

weights/             # 仓库根，权重集中平铺
├── manifest.json             清单（路径 / 大小 / sha256 / 分组）
├── tinymog.onnx              检测
├── retinaface_10g.onnx       检测
├── insightface_2d106.onnx    特征点
├── arcface_mbf.onnx          识别
├── sam2.1_hiera_small.pt     分割
├── groundingdino_swint_ogc.pth  文本引导检测
└── bert-base-uncased/        唯一保留的子目录（tokenizer 要求词表与权重同目录）
```

## 用法

```bash
python -m modelhub.fetch --list          # 看缺哪些、各多大
python -m modelhub.fetch                 # 按清单下载
python -m modelhub.fetch --source DIR    # 从本地目录复制（离线包 / 内网镜像）
python -m modelhub.fetch --url-base URL  # 从 HTTP 镜像取
```

清单在 `weights/manifest.json`，解析规则在 `weights.py`。缺哪个权重，运行时会直接报出
对应的 manifest 条目，不会静默降级。
