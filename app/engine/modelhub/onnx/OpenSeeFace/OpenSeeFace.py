"""OpenSeeFace face landmark detector (66 points) using ONNX models.
CPU only (model uses unsupported CUDA fused conv ops).
"""
from pathlib import Path
from modelhub import weights
import cv2
import numpy as np
import onnxruntime


class OpenSeeFace:
    @staticmethod
    def get_available_devices():
        return ['CPU']

    def __init__(self, device_info=None, model_name='lm_model0_opt.onnx'):
        path = weights.path('onnx/OpenSeeFace/' + model_name)
        if not path.exists():
            raise FileNotFoundError(f'Model not found: {path}')
        self._sess = onnxruntime.InferenceSession(str(path), providers=['CPUExecutionProvider'])
        self._input_name = self._sess.get_inputs()[0].name

    def extract(self, img: np.ndarray) -> np.ndarray:
        """Detect 66 landmarks from a face crop.

        Args:
            img: BGR face image (H, W, 3).

        Returns:
            (66, 2) array of landmark x,y coordinates.
        """
        H, W = img.shape[:2]

        # Preprocess: resize to 224x224, normalize (same as MobileNetV3 training)
        resized = cv2.resize(img, (224, 224))
        rgb = resized[:, :, ::-1].astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        normalized = (rgb - mean) / std
        blob = np.transpose(normalized, (2, 0, 1))[np.newaxis, ...].astype(np.float32)

        out = self._sess.run(None, {self._input_name: blob})[0]  # (1, 198, 28, 28)
        x = out[0]  # (198, 28, 28)

        # OpenSeeFace decoding:
        # 0:66   = main classification heatmaps
        # 66:132 = x-offset heatmaps
        # 132:198 = y-offset heatmaps
        num_pts = 66
        HW = 28 * 28  # 784

        t_main = x[:num_pts].reshape(num_pts, HW)           # (66, 784)
        t_off_x = x[num_pts:num_pts*2].reshape(num_pts, HW)  # (66, 784)
        t_off_y = x[num_pts*2:num_pts*3].reshape(num_pts, HW)  # (66, 784)

        # argmax for each landmark
        t_m = np.argmax(t_main, axis=1)  # (66,) index in [0, 784]

        # Get confidence and check mean
        t_conf = t_main[np.arange(num_pts), t_m]  # (66,)
        mean_conf = t_conf.mean()
        if mean_conf < 5.0:
            # Low confidence - landmarks likely invalid
            pass
        off_x = t_off_x[np.arange(num_pts), t_m]  # (66,)
        off_y = t_off_y[np.arange(num_pts), t_m]

        # logit_arr(x) = log(sigmoid(x)/(1-sigmoid(x))) / 16 = x / 16
        # So: decoded_offset = 223 * logit_arr(x) + 0.5 = 223 * x / 16 + 0.5
        off_x = np.floor(223.0 * off_x / 16.0 + 0.5).astype(np.int32)
        off_y = np.floor(223.0 * off_y / 16.0 + 0.5).astype(np.int32)

        # Row (y) and column (x) in 28x28 grid
        row = t_m // 28  # y index in heatmap
        col = t_m % 28   # x index in heatmap

        # Scale to 224x224 space: multiply by 223/27
        factor = 223.0 / 27.0
        lm_x = factor * col + off_x  # (66,)
        lm_y = factor * row + off_y

        # Clip to 224x224 range and pack
        lm_x = np.clip(lm_x, 0, 223)
        lm_y = np.clip(lm_y, 0, 223)
        landmarks = np.stack([lm_x, lm_y], axis=1)

        # Scale from 224x224 to original image size
        landmarks[:, 0] = landmarks[:, 0] / 224.0 * W
        landmarks[:, 1] = landmarks[:, 1] / 224.0 * H

        # Pad 66→68 points for DFL compatibility
        # OpenSeeFace inner mouth has 6 pts (idx 60-65), DFL needs 8 (idx 60-67)
        # Interpolate the missing 2 pts from adjacent inner mouth points
        full = np.zeros((68, 2), dtype=np.float32)
        full[:60] = landmarks[:60]
        full[60:64] = landmarks[60:64]  # first 4 inner mouth pts: idx 60-63
        full[64] = (landmarks[63] + landmarks[64]) / 2  # interpolate idx 64
        full[65] = landmarks[64]                         # copy idx 65
        full[66] = (landmarks[64] + landmarks[65]) / 2   # interpolate idx 66
        full[67] = landmarks[65]                         # copy idx 67
        # Return as list of arrays (Extractor expects list, one per face)
        return [full]

# TRT: skipped — CPU-only model (unsupported fused conv ops)
