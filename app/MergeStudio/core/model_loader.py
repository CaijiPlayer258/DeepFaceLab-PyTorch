"""
Thread-safe ONNX model singleton loader.
Transplanted from WebUI/studio_pipeline.py, inspired by MaskProcessor/core/model_loader.py.
"""
from pathlib import Path
import threading
import onnxruntime


class ModelLoader:
    """Thread-safe singleton ONNX model loader."""

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        with self._lock:
            if self._initialized:
                return
            self._sessions = {}
            self._device_info = None
            self._initialized = True

    def get_device_info(self):
        if self._device_info is not None:
            return self._device_info
        providers = onnxruntime.get_available_providers()
        if 'CUDAExecutionProvider' in providers:
            self._device_info = ('CUDAExecutionProvider', {})
        elif 'DmlExecutionProvider' in providers:
            self._device_info = ('DmlExecutionProvider', {})
        else:
            self._device_info = ('CPUExecutionProvider', {})
        return self._device_info

    def load_model(self, model_path: str) -> onnxruntime.InferenceSession:
        model_path = str(Path(model_path).resolve())
        if model_path in self._sessions:
            return self._sessions[model_path]
        provider, provider_options = self.get_device_info()
        session = onnxruntime.InferenceSession(model_path, providers=[provider])
        self._sessions[model_path] = session
        return session

    def unload_model(self, model_path: str):
        self._sessions.pop(str(Path(model_path).resolve()), None)

    def unload_all(self):
        self._sessions.clear()


model_loader = ModelLoader()
