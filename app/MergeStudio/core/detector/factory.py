"""
Face detector and landmarker factory.
Reference: Extractor/Extractor.py
Each import is individually guarded so missing modules don't block others.
"""
import sys
from pathlib import Path

_proj_root = str(Path(__file__).resolve().parent.parent.parent.parent)
if _proj_root not in sys.path:
    sys.path.insert(0, _proj_root)


def _safe_import(module_path, attr=None):
    """Try to import a module; return the class or None on failure."""
    try:
        if attr:
            import importlib
            mod = importlib.import_module(module_path)
            return getattr(mod, attr, None)
        return __import__(module_path, fromlist=[''])
    except Exception:
        return None


class DetectorFactory:
    DETECTORS = {}
    _loaded = False

    @classmethod
    def _ensure_loaded(cls):
        if cls._loaded:
            return
        cls._loaded = True
        print("[MergeStudio] Loading detectors...")

        # Try each detector independently
        _models = [
            ('BlazeFace',       'modelhub.onnx',             'BlazeFace'),
            ('CenterFace',      'modelhub.onnx',             'CenterFace'),
            ('S3FD',            'modelhub.onnx',             'S3FD'),
            ('YoloV5Face',      'modelhub.onnx',             'YoloV5Face'),
            ('YoloV8Face',      'modelhub.onnx.YoloV8Face',  'YoloV8Face'),
            ('RetinaFace',      'modelhub.onnx',             'RetinaFace'),
            ('DamoFD',          'modelhub.onnx',             'DamoFD'),
            ('TinyMog',         'modelhub.onnx',             'TinyMog'),
            ('ULFD',            'modelhub.onnx',             'ULFD'),
            ('MogFace',         'modelhub.onnx',             'MogFace'),
            ('MTCNN',           'modelhub.onnx',             'MTCNN'),
            ('LightweightFD',   'modelhub.onnx.LightweightFD','LightweightFD'),
            ('YoloV11nFace',    'modelhub.onnx.YoloV11nFace', 'YoloV11nFace'),
        ]
        for name, mod_path, cls_name in _models:
            cls_ = _safe_import(mod_path, cls_name)
            if cls_ is not None:
                if name == 'YoloV8Face':
                    cls.DETECTORS['YoloV8Face'] = cls_
                    cls.DETECTORS['YOLOv8'] = cls_
                elif name == 'RetinaFace':
                    cls.DETECTORS['RetinaFace'] = cls_
                    cls.DETECTORS['RetinaFace_10g'] = cls_
                    cls.DETECTORS['RetinaFace_500m'] = cls_
                else:
                    cls.DETECTORS[name] = cls_

        print(f"[MergeStudio] Detectors loaded: {list(cls.DETECTORS.keys())}")

    _RETINAFACE_MODELS = {
        'RetinaFace': 'det_10g',
        'RetinaFace_10g': 'det_10g',
        'RetinaFace_500m': 'det_500m',
    }

    @classmethod
    def create(cls, name: str, device_info):
        cls._ensure_loaded()
        detector_class = cls.DETECTORS.get(name)
        if detector_class is None:
            raise ValueError(f"Unsupported detector: {name}. modelhub may not be installed.")
        if name in cls._RETINAFACE_MODELS:
            return detector_class(device_info, model_name=cls._RETINAFACE_MODELS[name])
        return detector_class(device_info)


class LandmarkFactory:
    LANDMARKS = {}
    _loaded = False

    @classmethod
    def _ensure_loaded(cls):
        if cls._loaded:
            return
        cls._loaded = True
        print("[MergeStudio] Loading landmarkers...")

        _models = [
            ('insightface-2d106det', 'modelhub.onnx',                'InsightFace2D106'),
            ('2DFAN-4',              'modelhub.onnx.FAN',            'FAN'),
            ('3DFAN-4',              'modelhub.onnx.FAN',            'FAN'),
            ('insightface-3d68',     'modelhub.onnx',                'InsightFace3D68'),
            ('Google-mediapipe',     'modelhub.onnx',                 'FaceMesh'),
            ('OpenSeeFace',          'modelhub.onnx.OpenSeeFace',    'OpenSeeFace'),
            ('PFLD',                 'modelhub.onnx.PFLD',           'PFLD'),
            ('MobileFaceNet',        'modelhub.onnx.MobileFaceNet',  'MobileFaceNet'),
        ]
        for name, mod_path, cls_name in _models:
            cls_ = _safe_import(mod_path, cls_name)
            if cls_ is not None:
                cls.LANDMARKS[name] = cls_

        print(f"[MergeStudio] Landmarkers loaded: {list(cls.LANDMARKS.keys())}")

    @classmethod
    def create(cls, name: str, device_info):
        cls._ensure_loaded()
        landmark_class = cls.LANDMARKS.get(name)
        if landmark_class is None:
            raise ValueError(f"Unsupported landmarker: {name}. modelhub may not be installed.")
        try:
            if name == '3DFAN-4':
                landmarker = landmark_class(device_info, landmarks_3D=True)
            elif name == '2DFAN-4':
                landmarker = landmark_class(device_info, landmarks_3D=False)
            else:
                landmarker = landmark_class(device_info)
            print(f"[MergeStudio] Landmarker loaded: {name}")
            return landmarker
        except Exception as e:
            import traceback
            print(f"[MergeStudio] Landmarker {name} create failed: {e}")
            traceback.print_exc()
            raise


def get_device_info():
    """Get the first available device for modelhub detectors."""
    from xlib.onnxruntime import get_available_devices_info
    devices = get_available_devices_info()
    if devices:
        return devices[0]
    raise RuntimeError("No available devices found for ONNX Runtime")
