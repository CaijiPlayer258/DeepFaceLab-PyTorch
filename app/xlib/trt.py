"""
TensorRT InferenceSession -- ONNX Runtime API 兼容封装。
提供 .run(None, feed_dict) -> [ndarray, ...] 接口，模型层零改动。
"""
import os, hashlib, warnings, threading
from pathlib import Path
from typing import List

import numpy as np
import torch

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

TRT_AVAILABLE = False
try:
    import tensorrt as trt
    TRT_AVAILABLE = True
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
except ImportError:
    trt = None
    TRT_LOGGER = None


class TensorInfo:
    """兼容 ONNX Runtime NodeArg 的 .name/.shape 访问。"""
    def __init__(self, name: str, shape, dtype):
        self.name = name
        self.shape = shape
        self.type = dtype


class TRTInferenceSession:
    """TensorRT inference session with ONNX-compatible .run() API.
    线程安全：每个线程独立 context + stream + buffer，无需锁。
    Usage:
        sess = TRTInferenceSession('model.engine')
        out = sess.run(None, {'input': numpy_array})[0]
    """

    def __init__(self, engine_path: str, device_id: int = 0):
        if not TRT_AVAILABLE:
            raise ImportError('tensorrt not installed')
        self.device_id = device_id
        self.device = f'cuda:{device_id}'

        # 反序列化 engine（只读，线程安全）
        if isinstance(engine_path, (str, Path)):
            with open(engine_path, 'rb') as f:
                engine_bytes = f.read()
        else:
            engine_bytes = engine_path

        runtime = trt.Runtime(TRT_LOGGER)
        self.engine = runtime.deserialize_cuda_engine(engine_bytes)
        if self.engine is None:
            raise RuntimeError(f'Failed to deserialize engine: {engine_path}')
        engine_name = Path(engine_path).name if isinstance(engine_path, (str, Path)) else 'engine'
        print(f'⚡ [TRT] {engine_name} loaded (BF16)')

        # 遍历 I/O bindings（只读元数据）
        self.input_names: List[str] = []
        self.output_names: List[str] = []
        self._input_infos: List[TensorInfo] = []
        self._output_infos: List[TensorInfo] = []

        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            dtype = self.engine.get_tensor_dtype(name)
            shape = self.engine.get_tensor_shape(name)
            mode = self.engine.get_tensor_mode(name)
            info = TensorInfo(name, shape, dtype)

            if mode == trt.TensorIOMode.INPUT:
                self.input_names.append(name)
                self._input_infos.append(info)
            else:
                self.output_names.append(name)
                self._output_infos.append(info)

        # 每线程独立状态（懒创建）
        self._thread_local = threading.local()

    def _get_thread_state(self):
        """获取当前线程的 context + stream + buffers（懒创建）。"""
        try:
            ctx = self._thread_local.context
        except AttributeError:
            ctx = self.engine.create_execution_context()
            self._thread_local.context = ctx
            self._thread_local.stream = torch.cuda.Stream(device=self.device_id)
            self._thread_local.buffers = {}
            for info in self._input_infos + self._output_infos:
                alloc_shape = [s if s > 0 else 1 for s in info.shape]
                torch_dtype = self._trt_to_torch(info.type)
                buf = torch.zeros(alloc_shape, dtype=torch_dtype, device=self.device)
                self._thread_local.buffers[info.name] = buf
                ctx.set_tensor_address(info.name, buf.data_ptr())
        return ctx, self._thread_local.stream, self._thread_local.buffers

    @staticmethod
    def _trt_to_torch(trt_dtype):
        mapping = {
            trt.float32: torch.float32,
            trt.float16: torch.float16,
            trt.int32: torch.int32,
            trt.int64: torch.int64,
            trt.int8: torch.int8,
            trt.uint8: torch.uint8,
            trt.bool: torch.bool,
        }
        return mapping.get(trt_dtype, torch.float32)

    def get_inputs(self):
        return self._input_infos

    def get_outputs(self):
        return self._output_infos

    def run(self, output_names, feed_dict):
        """Run inference. Thread-safe — per-thread context/stream.
        Args:
            output_names: list of output names to fetch, or None for all.
            feed_dict: {input_name: numpy_array}
        Returns:
            list of numpy arrays matching output_names order.
        """
        ctx, stream, buffers = self._get_thread_state()

        # 1. Validate required inputs
        missing = [n for n in self.input_names if n not in feed_dict]
        if missing:
            raise ValueError(f'Missing required inputs: {missing}')

        # 2. Set input shapes before querying output shapes
        for name in self.input_names:
            ctx.set_input_shape(name, feed_dict[name].shape)

        # 3. Allocate/reallocate output buffers based on inferred shapes
        for name in self.output_names:
            inferred = tuple(ctx.get_tensor_shape(name))
            if -1 in inferred:
                raise RuntimeError(
                    f'Output "{name}" has unresolved dynamic dims {inferred} '
                    f'after set_input_shape. Check optimization profile.')
            torch_dtype = self._trt_to_torch(self.engine.get_tensor_dtype(name))
            if name not in buffers or list(inferred) != list(buffers[name].shape):
                buffers[name] = torch.zeros(inferred, dtype=torch_dtype,
                                             device=self.device)
                ctx.set_tensor_address(name, buffers[name].data_ptr())

        # 4. H2D on the same stream as TRT execution
        with torch.cuda.stream(stream):
            for name, arr in feed_dict.items():
                if name not in buffers:
                    warnings.warn(f'Unknown input: {name}, skipping')
                    continue
                t = torch.from_numpy(arr).to(device=self.device, non_blocking=True)
                buf = buffers[name]
                if t.shape == buf.shape:
                    buf.copy_(t)
                else:
                    buffers[name] = t.contiguous()
                    ctx.set_tensor_address(name, buffers[name].data_ptr())

        # 5. Execute
        ctx.execute_async_v3(stream.cuda_stream)

        # 6. Sync
        stream.synchronize()

        # 7. D2H
        if output_names is None:
            output_names = self.output_names
        results = []
        for name in output_names:
            buf = buffers[name]
            results.append(buf.cpu().numpy())
        return results


# 兼容模式标记文件路径（存在时禁用 TRT）
_TRT_COMPAT_FLAG = os.environ.get('DFL_TRT_COMPAT_FLAG',
    str(Path(__file__).parent.parent / '.trt_compat'))

def _trt_compat_mode() -> bool:
    """检查是否启用兼容模式（禁用 TRT，强制 ONNX fallback）。"""
    return os.path.exists(_TRT_COMPAT_FLAG)

def set_trt_compat_mode(enabled: bool):
    """设置/取消兼容模式。"""
    flag = Path(_TRT_COMPAT_FLAG)
    if enabled:
        flag.write_text('')
    else:
        flag.unlink(missing_ok=True)

def find_trt_engine(onnx_path, model_name, batch=1, device_id=0):
    """查找匹配当前 GPU 和 ONNX 内容的 TRT engine 文件。

    Args:
        onnx_path: ONNX 文件路径
        model_name: 引擎名（不包含 _bsN_sm... 后缀）
        batch: 批大小
        device_id: 查询 SM 的 GPU 设备号

    Returns:
        str (engine 路径) 或 None
    """
    if batch < 1:
        return None
    if _trt_compat_mode():
        return None
    if not TRT_AVAILABLE or not torch.cuda.is_available():
        return None
    onnx_path = Path(onnx_path)
    if not onnx_path.exists():
        return None

    with open(onnx_path, 'rb') as f:
        md5 = hashlib.md5(f.read()).hexdigest()[:16]

    props = torch.cuda.get_device_properties(device_id)
    sm_str = f'sm{props.major}{props.minor}0'
    pattern = f'{model_name}_bs{batch}_{sm_str}_trt*_{md5}.engine'
    matches = list(onnx_path.parent.glob(pattern))
    return str(matches[0]) if matches else None
