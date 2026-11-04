# BlazeFace ONNX

A super-fast and accurate face detector model exported to ONNX format, ready to run with [onnxruntime](https://onnxruntime.ai/).

- **Input:** RGB image, resized to 128x128.
- **Output:** List of detected faces with bounding boxes and facial landmarks.
- **Use case:** Face detection in images or video frames, ideal for real-time apps.

---

## 📥 Download Model

Download the ONNX model using:

```bash
wget https://huggingface.co/garavv/blazeface-onnx/resolve/main/blaze.onnx?download=true -O blaze.onnx
```

## 🚀 Quick Start

```python
import numpy as np
from PIL import Image, ImageDraw
import onnxruntime as ort
import pathlib

# -------- CONFIG --------
ONNX_PATH   = "blaze.onnx"                      # Path to your ONNX model
IMAGE_PATH  = "your_image.jpg"                  # Path to your input image

CONF_THRESH = 0.5
IOU_THRESH  = 0.3
MAX_DET     = 25

# -------- LOAD & PREP --------
orig = Image.open(IMAGE_PATH).convert("RGB")
W, H = orig.size

# Resize to 128×128 for model input
img128 = orig.resize((128,128))
img_np  = np.asarray(img128, dtype=np.float32) / 255.0  # HWC, [0..1]
img_np  = np.transpose(img_np, (2,0,1))[None,...]       # (1,3,128,128)

# -------- RUN ONNX --------
sess = ort.InferenceSession(str(pathlib.Path(ONNX_PATH)))
outs = sess.run(None, {
    "image": img_np.astype(np.float32),
    "conf_threshold": np.array([CONF_THRESH], dtype=np.float32),
    "max_detections": np.array([MAX_DET],     dtype=np.int64),
    "iou_threshold":  np.array([IOU_THRESH],  dtype=np.float32),
})

# -------- PARSE OUTPUT --------
boxes = outs[0][0]
if boxes.ndim==1: boxes = boxes.reshape(1,16)
scores = outs[1][0] if len(outs)>1 else np.ones(len(boxes), dtype=np.float32)

draw = ImageDraw.Draw(orig)
for det, score in zip(boxes, scores):
    (top_y, top_x, bot_y, bot_x,
     ley_x, ley_y, rey_x, rey_y,
     nose_x, nose_y, mou_x, mou_y,
     lea_x, lea_y, rea_x, rea_y) = det
    x1 = int(top_x * W)
    y1 = int(top_y * H)
    x2 = int(bot_x * W)
    y2 = int(bot_y * H)
    if x2-x1 < 5 or y2-y1 < 5: continue
    draw.rectangle([x1, y1, x2, y2], outline="lime", width=5)
    draw.text((x1, y1-12), f"{score:.2f}", fill="lime")
    # Draw landmarks
    lm = [
      ("red",     ley_x, ley_y),
      ("red",     rey_x, rey_y),
      ("blue",    nose_x, nose_y),
      ("magenta", mou_x, mou_y),
      ("orange",  lea_x, lea_y),
      ("orange",  rea_x, rea_y),
    ]
    for color, nx, ny in lm:
        cx = int(nx * W)
        cy = int(ny * H)
        draw.ellipse([cx-6, cy-6, cx+6, cy+6], fill=color)

# SAVE RESULT
out_path = "result_blazeface.jpg"
orig.save(out_path)
orig.show()
print(f"Saved → {out_path}")
```

---

## 📦 Dependencies

- Python 3.7+
- onnxruntime
- Pillow
- numpy

**Install with:**

```bash
pip install onnxruntime pillow numpy
```

---

## 📝 Model Details

- **Architecture:** BlazeFace (ONNX, opset 16)
- **Input shape:** (1, 3, 128, 128) (batch, channels, height, width)
- **Outputs:**
    - [N, 16] face detections (bounding box + 6 facial landmarks per face)
    - [N] scores (confidence)

**Landmarks are:**

Left Eye, Right Eye, Nose, Mouth, Left Cheek, Right Cheek

---

