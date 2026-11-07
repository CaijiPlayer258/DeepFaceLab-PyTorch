"""权重文件解析 —— 代码与模型分离 + **平铺布局**。

这个仓库只分发**代码**：`modelhub/onnx/<模型>/*.py` 进 git。
**权重**（.onnx / .pth / .pt / .ubj）由用户自己下载，**全部平铺**放在仓库根的
`weights/` 下，不进版本库（见 .gitignore 与 weights/manifest.json）。

    布局（默认，唯一目录例外是 BERT tokenizer）：

        weights/
        ├── manifest.json                  清单（路径/大小/sha256/分组，入库）
        ├── README.md                      说明（入库）
        ├── tinymog.onnx                   ← 模型名 + 原文件名
        ├── retinaface_10g.onnx
        ├── insightface_2d106.onnx
        ├── arcface_mbf.onnx
        ├── sam2.1_hiera_small.pt
        ├── groundingdino_swint_ogc.pth
        └── bert-base-uncased/             ← 唯一保留的子目录
            ├── vocab.txt                  AutoTokenizer.from_pretrained(dir)
            ├── tokenizer.json             要求词表/权重/配置**同目录**，
            ├── tokenizer_config.json      所以它不能平铺
            ├── config.json
            └── pytorch_model.bin

查找顺序（第一个存在者胜出）：

1. `$DFL_WEIGHTS_DIR/<名字>`                     ← 新布局（默认）
2. `<仓库根>/weights/<名字>`                     ← 新布局（默认）
3. `$DFL_WEIGHTS_DIR/onnx/<模型>/<原文件名>`      ← 老的分目录布局，仍然认
4. `<仓库根>/weights/onnx/<模型>/<原文件名>`      ← 同上
5. 代码同目录（开发时把 .onnx 直接放在 .py 旁边）

第 3/4 条是**向后兼容**：老用户把权重按 `weights/onnx/RetinaFace/det_10g.onnx`
放过，或者 `$DFL_WEIGHTS_DIR` 指向一个老布局的树，代码照样找得到 —— 默认（新装）
只认平铺。

用法：

    from modelhub import weights
    p = weights.path("retinaface_10g.onnx")     # 缺失时抛 WeightsMissing
    p = weights.path("onnx/RetinaFace/det_10g.onnx")   # 老名字也认，自动折算成平铺名
    d = weights.root()                          # 权重根

配套命令行：`python -m modelhub.fetch --list`（见 fetch.py）。
"""
from __future__ import annotations

import json
import os
from pathlib import Path

__all__ = ["WeightsMissing", "root", "path", "dir", "try_path", "manifest",
           "repo_root", "flat_name", "LEGACY_LAYOUT"]

_HERE = Path(__file__).resolve().parent          # app/engine/modelhub
_ENV = "DFL_WEIGHTS_DIR"

# 平铺文件名 → 老布局（weights/ 下的相对路径）。
# 只用于「读老布局」的兜底，不是默认路径；新装只看 weights/<平铺名>。
LEGACY_LAYOUT = {
    "tinymog.onnx": "onnx/TinyMog/TinyMog.onnx",
    "retinaface_10g.onnx": "onnx/RetinaFace/det_10g.onnx",
    "retinaface_500m.onnx": "onnx/RetinaFace/det_500m.onnx",
    "insightface_2d106.onnx": "onnx/InsightFace2d106/InsightFace2D106.onnx",
    "insightface_3d68.onnx": "onnx/InsightFace3D68/1k3d68.onnx",
    "arcface_mbf.onnx": "onnx/ArcFace/w600k_mbf.onnx",
    "arcface_r50.onnx": "onnx/ArcFace/w600k_r50.onnx",
    "blazeface.onnx": "onnx/BlazeFace/blaze.onnx",
    "centerface.onnx": "onnx/CenterFace/CenterFace.onnx",
    "damofd.onnx": "onnx/DamoFD/DamoFD.onnx",
    "faceenhancer.onnx": "onnx/FaceEnhancer/FaceEnhancer.onnx",
    "facemesh.onnx": "onnx/FaceMesh/FaceMesh.onnx",
    "faceparser.onnx": "onnx/FaceParser/faceparser.onnx",
    "fan_2d.onnx": "onnx/FAN/2DFAN.onnx",
    "fan_3d.onnx": "onnx/FAN/3DFAN.onnx",
    "fastfacealign.onnx": "onnx/FastFaceAlign/FastFaceAlign.onnx",
    "genderage.onnx": "onnx/GenderAge/genderage.onnx",
    "lia_generator.onnx": "onnx/LIA/generator.onnx",
    "lightweightfd.onnx": "onnx/LightweightFD/face_det_lite.onnx",
    "mobilefacenet.onnx": "onnx/MobileFaceNet/landmark_detection_56_se_external.onnx",
    "mogface.onnx": "onnx/MogFace/MogFace.onnx",
    "mtcnn_pnet.onnx": "onnx/MTCNN/pnet.onnx",
    "mtcnn_rnet.onnx": "onnx/MTCNN/rnet.onnx",
    "mtcnn_onet.onnx": "onnx/MTCNN/onet.onnx",
    "openseeface_lm0.onnx": "onnx/OpenSeeFace/lm_model0_opt.onnx",
    "openseeface_lm1.onnx": "onnx/OpenSeeFace/lm_model1_opt.onnx",
    "openseeface_retinaface.onnx": "onnx/OpenSeeFace/retinaface_640x640_opt.onnx",
    "pfld.onnx": "onnx/PFLD/pfld.onnx",
    "s3fd.onnx": "onnx/S3FD/S3FD.onnx",
    "ulfd.onnx": "onnx/ULFD/ULFD.onnx",
    "yolov5face.onnx": "onnx/YoloV5Face/YoloV5Face.onnx",
    "yolov8n_face.onnx": "onnx/YoloV8Face/yolov8n-face.onnx",
    "yolov11n_face.onnx": "onnx/YoloV11nFace/yolov11n-face.onnx",
    "bisenet_model_final.pth": "bisenet/model_final.pth",
    "errface_xgb.ubj": "errface/errface_xgb.ubj",
    "sam_vit_b_01ec64.pth": "sam/sam_vit_b_01ec64.pth",
    "groundingdino_swint_ogc.pth": "grounding_dino/gdino_checkpoints/groundingdino_swint_ogc.pth",
    "groundingdino_swinb_cogcoor.pth": "grounding_dino/gdino_checkpoints/groundingdino_swinb_cogcoor.pth",
    "sam2.1_hiera_tiny.pt": "sam2/checkpoints/sam2.1_hiera_tiny.pt",
    "sam2.1_hiera_small.pt": "sam2/checkpoints/sam2.1_hiera_small.pt",
    "sam2.1_hiera_base_plus.pt": "sam2/checkpoints/sam2.1_hiera_base_plus.pt",
    "sam2.1_hiera_large.pt": "sam2/checkpoints/sam2.1_hiera_large.pt",
}

# 老路径 → 平铺名（调用方还在用老写法时折算）
_ALIAS = {legacy: flat for flat, legacy in LEGACY_LAYOUT.items()}


class WeightsMissing(FileNotFoundError):
    """权重文件缺失。消息里带下载/放置指引，不要用裸 FileNotFoundError。"""

    def __init__(self, rel: str, tried, hint: str = ""):
        self.rel = rel
        self.tried = list(tried)
        lines = ["缺少权重文件：%s" % rel, "已查找："]
        lines += ["  - %s" % t for t in self.tried]
        if hint:
            lines.append(hint)
        lines.append("下载：python -m modelhub.fetch --file %s" % rel)
        lines.append("或手动放到：%s" % (Path(tried[0]) if tried else rel))
        super().__init__("\n".join(lines))


def repo_root() -> Path:
    """仓库根：app/engine/modelhub → 上溯三级。"""
    return _HERE.parents[2]


def root() -> Path:
    """权重根目录（默认 <仓库根>/weights，可用 $DFL_WEIGHTS_DIR 覆盖）。"""
    env = os.environ.get(_ENV)
    if env:
        return Path(env).expanduser()
    return repo_root() / "weights"


def flat_name(rel: str) -> str:
    """把老布局的名字（onnx/RetinaFace/det_10g.onnx）折算成平铺名（retinaface_10g.onnx）。"""
    rel = str(rel).replace("\\", "/").lstrip("./")
    return _ALIAS.get(rel, rel)


def _candidates(rel: str):
    name = flat_name(rel)
    yield root() / name                     # 新布局（默认）
    legacy = LEGACY_LAYOUT.get(name)
    if legacy:
        yield root() / legacy               # 老布局：weights/onnx/<模型>/<原文件名>
    yield _HERE / name                      # 开发时权重就在代码旁边
    if legacy:
        yield _HERE / legacy


def try_path(rel: str):
    for c in _candidates(rel):
        if c.is_file():
            return c
    return None


def path(rel: str, hint: str = "") -> Path:
    p = try_path(rel)
    if p is None:
        raise WeightsMissing(rel, [str(c) for c in _candidates(rel)], hint)
    return p


def dir(rel: str = "") -> Path:
    """数据目录（不检查是否存在，用于输出/批量扫描）。

    平铺布局下所有权重都在 root()，所以 dir("") 就是权重根；只有传入老目录名时
    才可能落到老布局。
    """
    name = flat_name(rel)
    if not name:
        return root()
    d = root() / name
    if not d.is_dir():
        legacy = LEGACY_LAYOUT.get(name, name)
        if (root() / legacy).is_dir() and any((root() / legacy).iterdir()):
            return root() / legacy
        if (_HERE / legacy).is_dir() and any((_HERE / legacy).iterdir()):
            return _HERE / legacy
    return d


def manifest() -> dict:
    mf = root() / "manifest.json"
    if mf.is_file():
        return json.loads(mf.read_text(encoding="utf-8"))
    return {"version": 1, "files": []}
