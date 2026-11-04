---
license: cc-by-nc-nd-4.0
library_name: transformers
pipeline_tag: object-detection
tags:
- face-detection
- object-detection
- ssdlite
- mobilenetv3
- torchvision
- custom-code
- research
datasets:
- CUHK-CSE/wider_face
base_model: []
model-index:
- name: BananaMind FaceDetect V1
  results: []
---

# BananaMind FaceDetect V1

BananaMind FaceDetect V1 is an early face detection model from Banaxi Tech. It uses a custom Hugging Face Transformers-compatible wrapper around `torchvision.models.detection.ssdlite320_mobilenet_v3_large` and is intended for research and experimentation.

This model is licensed under **CC-BY-NC-ND-4.0** and inherits non-commercial/no-derivatives usage restrictions from the WIDER FACE dataset license listed on Hugging Face. Do not treat this model as Apache, MIT, or commercial-use safe.

## Model Details

- **Repository:** `Banaxi-Tech/BananaMind-FaceDetect-V1`
- **Model name:** BananaMind FaceDetect V1
- **Task:** Face detection / object detection
- **Architecture:** SSDLite320 MobileNetV3 Large via torchvision
- **Framework wrapper:** Custom Hugging Face Transformers remote code
- **Training:** Trained from scratch
- **Pretrained backbone weights:** None
- **Ultralytics/YOLO:** Not used
- **Number of classes:** 2
- **Class 0:** background
- **Class 1:** face

The repository includes:

- `model.safetensors`: main safetensors checkpoint
- `original_model.pt`: original PyTorch checkpoint
- `config.json`: custom Transformers config
- `configuration_banaxi_face_detector.py`: custom config class
- `modeling_banaxi_face_detector.py`: custom model wrapper
- `README.md`: model card

## Dataset And Attribution

This model was trained on WIDER FACE using the Hugging Face dataset repository [`CUHK-CSE/wider_face`](https://huggingface.co/datasets/CUHK-CSE/wider_face).

WIDER FACE is a face detection benchmark containing face bounding boxes across images with large variation in face scale, pose, occlusion, and scene type. The Hugging Face dataset page lists the dataset license as **CC-BY-NC-ND-4.0**.

Please also cite and follow the terms for WIDER FACE when using this model:

```bibtex
@inproceedings{yang2016wider,
  title={WIDER FACE: A Face Detection Benchmark},
  author={Yang, Shuo and Luo, Ping and Loy, Chen Change and Tang, Xiaoou},
  booktitle={IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2016}
}
```

## Training Notes

- Dataset source: [`CUHK-CSE/wider_face`](https://huggingface.co/datasets/CUHK-CSE/wider_face)
- Dataset task: face detection benchmark
- Training hardware: NVIDIA A100 80GB GPU
- Training used converted WIDER FACE bounding boxes
- Very tiny faces were filtered out during training because the detector input size is 320px and tiny WIDER FACE boxes destabilized early training
- Base detector: `ssdlite320_mobilenet_v3_large`
- Input format: RGB image tensor normalized to `0..1`

## Usage

This repository uses custom remote code:

- `configuration_banaxi_face_detector.py`
- `modeling_banaxi_face_detector.py`

Load the model with `trust_remote_code=True`.

```python
import torch
from transformers import AutoModel

model = AutoModel.from_pretrained(
    "Banaxi-Tech/BananaMind-FaceDetect-V1",
    trust_remote_code=True,
)
model.eval()

# pixel_values must be a tensor shaped B x 3 x H x W.
# Values should be RGB floats in the range 0..1.
pixel_values = torch.rand(1, 3, 320, 320)

with torch.no_grad():
    outputs = model(pixel_values=pixel_values)

prediction = outputs[0]
boxes = prediction["boxes"]    # xyxy pixel coordinates
scores = prediction["scores"]  # confidence scores
labels = prediction["labels"]  # label 1 means face

keep = scores >= 0.5
face_boxes = boxes[keep]
face_scores = scores[keep]
```

Outputs follow the torchvision detection format:

- `boxes`: predicted bounding boxes in `xyxy` pixel coordinates
- `scores`: confidence scores
- `labels`: predicted class labels

Label `1` means `face`. A confidence threshold around `0.5` to `0.7` is recommended, depending on the use case.

## Limitations

- This is an early V1 model.
- It was trained from scratch and may be less accurate than large pretrained face detectors.
- It may miss very small faces.
- It may produce duplicate or partial boxes in difficult cases.
- It is intended mainly for experimentation and research.
- The model inherits usage restrictions from the WIDER FACE dataset license.

## License

This model is released under **CC-BY-NC-ND-4.0**.

The training dataset, WIDER FACE as hosted at [`CUHK-CSE/wider_face`](https://huggingface.co/datasets/CUHK-CSE/wider_face), is also listed on Hugging Face as **CC-BY-NC-ND-4.0**. Because of these terms, this model should be used for research/non-commercial purposes and should not be presented as commercially usable or permissively licensed.
