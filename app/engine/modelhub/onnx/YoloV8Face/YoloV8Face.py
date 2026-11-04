from pathlib import Path
from modelhub import weights
from typing import List
import numpy as np
from xlib import math as lib_math
from xlib.image import ImageProcessor
from xlib.onnxruntime import (InferenceSession_with_device, ORTDeviceInfo,
                              get_available_devices_info)


class YoloV8Face:
    """
    YOLOv8-Face face detection model.
    
    Anchor-free architecture with improved speed and accuracy over YOLOv5-Face.
    
    arguments
     device_info    ORTDeviceInfo
        use YoloV8Face.get_available_devices()
        to determine a list of available devices accepted by model
    
    raises
     Exception
    """

    @staticmethod
    def get_available_devices() -> List[ORTDeviceInfo]:
        return get_available_devices_info()

    def __init__(self, device_info: ORTDeviceInfo):
        if device_info not in YoloV8Face.get_available_devices():
            raise Exception(f'device_info {device_info} is not in available devices for YoloV8Face')

        path = weights.path('onnx/YoloV8Face/yolov8n-face.onnx')
        self._sess = sess = InferenceSession_with_device(str(path), device_info)
        self._input_name = sess.get_inputs()[0].name
        
        # Get input size from model
        input_shape = sess.get_inputs()[0].shape
        self.input_size = input_shape[-1]  # Assuming square input (e.g., 640)

        # ── TRT BF16 加速 ──────────────────────
        _trt_path = None
        try:
            from xlib.trt import find_trt_engine
            _trt_path = find_trt_engine(str(path), 'yolov8n-face')
        except Exception:
            pass
        if _trt_path:
            try:
                from xlib.trt import TRTInferenceSession
                self._sess = TRTInferenceSession(_trt_path)
            except Exception as e:
                import warnings as _w
                _w.warn(f'TRT fallback: {e}')

    def extract(self, img, threshold: float = 0.3, fixed_window=0, min_face_size=8, augment=False,
                input_mode='one_stage', resize_mode='letterbox', input_size=None):
        """
        arguments
         img    np.ndarray      ndim 2,3,4
         fixed_window(0)    int  size
                                 0 mean don't use
                                 fit image in fixed window
                                 downscale if bigger than window
                                 pad if smaller than window
                                 increases performance, but decreases accuracy
         min_face_size(8)
         augment(False)     bool    augment image to increase accuracy
                                    decreases performance
        
        returns a list of [l,t,r,b] for every batch dimension of img
        """
        ip = ImageProcessor(img)
        _, H, W, _ = ip.get_dims()
        
        if H > 2048 or W > 2048:
            fixed_window = 2048

        # YOLOv8 requires fixed input size (640x640)
        # Always resize to 640x640 for consistent inference
        target_size = 640
        
        if fixed_window != 0:
            # Use user-specified fixed_window, but must be multiple of 32
            target_size = max(32, max(1, fixed_window // 32) * 32)
        
        # Resize and pad to target_size x target_size
        img_scale = ip.fit_in(target_size, target_size, pad_to_target=True, allow_upscale=False)

        ip.ch(3).to_ufloat32()
        _, H, W, _ = ip.get_dims()
        
        preds = self._get_preds(ip.get_image('NCHW'))

        if augment:
            rl_preds = self._get_preds(ip.flip_horizontal().get_image('NCHW'))
            rl_preds[:, :, 0] = W - rl_preds[:, :, 0]
            preds = np.concatenate([preds, rl_preds], 1)

        faces_per_batch = []
        for pred in preds:
            # Filter by confidence threshold
            pred = pred[pred[..., 4] >= threshold]

            # Extract bounding box coordinates and scores
            # YOLOv8 output format: [cx, cy, w, h, score, ...landmarks...]
            cx, cy, w, h, score = pred.T

            # Convert to corner coordinates
            l, t, r, b = cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2
            
            # Apply NMS
            keep = lib_math.nms(l, t, r, b, score, 0.5)
            l, t, r, b = l[keep], t[keep], r[keep], b[keep]

            faces = []
            for l_val, t_val, r_val, b_val in zip(l, t, r, b):
                # Scale back to original image coordinates
                if img_scale != 1.0:
                    l_val, t_val, r_val, b_val = l_val / img_scale, t_val / img_scale, r_val / img_scale, b_val / img_scale

                # Filter by minimum face size
                if min(r_val - l_val, b_val - t_val) < min_face_size:
                    continue
                    
                faces.append((l_val, t_val, r_val, b_val))

            faces_per_batch.append(faces)

        return faces_per_batch

    def _get_preds(self, img):
        """
        Run inference and process predictions.
        
        YOLOv8 output shape: [1, 5, 8400] for face detection
        Where 5 = [cx, cy, w, h, score]
        And 8400 = number of predictions (for 640x640 input)
        """
        N, C, H, W = img.shape
        preds = self._sess.run(None, {self._input_name: img})
        
        # YOLOv8 returns [N, 5+num_landmarks, num_predictions]
        # Transpose to [N, num_predictions, 5+num_landmarks]
        pred = preds[0].transpose(0, 2, 1)
        
        return pred[..., :5]  # Return only bbox + score (first 5 channels)
