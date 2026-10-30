"""
XSegAugmenter — XSeg / XSegLite ONNX 批量遮罩生成器

并行策略自动选择：
- CUDA 可用 → 线程池（共享一个 CUDA session，零 GPU 竞争）
- 仅 CPU    → 进程池（每进程独立 session）
- workers=1 → 串行
"""

from __future__ import annotations

import concurrent.futures
import pickle
import struct
import sys
import time
from multiprocessing import cpu_count
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
import tqdm

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 注册 CUDA/cuDNN DLL 搜索路径（项目 Python 可能不继承系统 PATH）
# 尝试注册 CUDA DLL 搜索路径（部分项目 Python 可能不继承系统 PATH）
import os as _os
for _try_path in (
    [p for p in _os.environ.get('PATH', '').split(';') if p] +
    [r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4\bin',
     r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1\bin',
     r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.4\bin',
     r'C:\Program Files\NVIDIA\CUDNN\v9.17\bin\12.9',
    ]):
    if _os.path.exists(_try_path):
        try:
            _os.add_dll_directory(_try_path)
        except Exception:
            pass

# 惰性导入 onnxruntime（避免 DLL 加载失败直接崩溃）
_ort = None
def _get_ort():
    global _ort
    if _ort is None:
        try:
            import onnxruntime as _m
            _m.set_default_logger_severity(3)
            import warnings
            warnings.filterwarnings('ignore', module='onnxruntime')
            _ort = _m
        except Exception as e:
            print("\n" + "="*60)
            print("[错误] ONNX Runtime 加载失败!")
            print(f"        {e}")
            print("="*60)
            print("[诊断] 系统信息:")
            # onnxruntime 包版本（即使 import 失败，pip 元数据可能还在）
            try:
                import importlib.metadata as _ilm
                try:
                    _ver = _ilm.version('onnxruntime')
                    print(f"  onnxruntime 版本 (pip): {_ver}")
                except:
                    _ver = _ilm.version('onnxruntime-gpu')
                    print(f"  onnxruntime 版本 (pip): {_ver} (gpu)")
            except:
                print(f"  onnxruntime 版本 (pip): 未知")
            # CUDA 版本
            try:
                _r = __import__('subprocess').run(
                    ['nvidia-smi', '--query-gpu=name,driver_version,compute_cap',
                     '--format=csv,noheader'],
                    capture_output=True, text=True, timeout=5)
                if _r.returncode == 0:
                    print(f"  GPU: {_r.stdout.strip()}")
            except:
                pass
            try:
                _r = __import__('subprocess').run(
                    ['nvidia-smi'],
                    capture_output=True, text=True, timeout=5)
                for _l in _r.stdout.splitlines():
                    if 'CUDA Version' in _l:
                        print(f"  {_l.strip()}")
                        break
            except:
                pass
            # PATH 中的 CUDA 相关 DLL 路径
            print(f"  PATH 中包含的 CUDA/cuDNN 路径:")
            _found = 0
            for _p in __import__('os').environ.get('PATH', '').split(';'):
                _p = _p.strip()
                if not _p:
                    continue
                _pl = _p.lower()
                if any(x in _pl for x in ('cuda', 'cudnn', 'tensorrt', 'nvidia', 'deepfacelab')):
                    _exists = "✓" if __import__('os').path.isdir(_p) else "✗"
                    print(f"    {_exists} {_p}")
                    _found += 1
            if _found == 0:
                print(f"    未找到 CUDA/cuDNN/TensorRT 路径")
            # PyTorch CUDA
            try:
                _torch = __import__('torch')
                print(f"  PyTorch: {_torch.__version__}  CUDA: {_torch.version.cuda or '无'}")
            except:
                print(f"  PyTorch: 未安装")
            print("="*60)
            print("[建议] 解决方法:")
            print("  1. 安装 CUDA 12.x + cuDNN 9.x（onnxruntime 自带的 CUDA 12 DLL 需要系统匹配）")
            print("  2. 或安装 CPU 版 onnxruntime: pip install onnxruntime")
            print("  3. 或设置 CUDA_PYTHON 环境变量指向可用 CUDA 的 Python")
            print("="*60)
            raise RuntimeError(f"ONNX Runtime 加载失败: {e}") from e
    return _ort


# ═══════════════════════════════════════════════════════════════
# 共享函数
# ═══════════════════════════════════════════════════════════════

def _encode_mask(mask: np.ndarray) -> np.ndarray:
    """将 float32 遮罩压缩为 PNG/JPEG bytes（同 DFLJPG.set_xseg_mask 逻辑）。"""
    img_data = np.clip(mask * 255, 0, 255).astype(np.uint8)
    if img_data.ndim == 2:
        img_data = img_data[..., None]
    data_max_len = 50000
    ret, buf = cv2.imencode('.png', img_data)
    if not ret or len(buf) > data_max_len:
        for q in range(100, -1, -1):
            ret, buf = cv2.imencode('.jpg', img_data,
                                    [int(cv2.IMWRITE_JPEG_QUALITY), q])
            if ret and len(buf) <= data_max_len:
                break
    if not ret or len(buf) > data_max_len:
        raise RuntimeError(f"遮罩编码失败: 无法压缩到 {data_max_len} 字节以内 (最终 {len(buf)} 字节)")
    return buf


def _save_mask_fast(img_path: str, mask: np.ndarray, pre_read_data: bytes = None):
    """快速写入 xseg_mask：直接定位 APP15 chunk 替换，跳过 JPG 完整解析。

    约定文件为标准 DFLJPG 格式（包含 APP15 标记）。
    如果已有 pre_read_data（文件原始 bytes），跳过再次读取。
    """
    buf = _encode_mask(mask)

    if pre_read_data is not None:
        data = pre_read_data
    else:
        with open(img_path, 'rb') as f:
            data = f.read()

    marker = b'\xff\xef'
    pos = data.find(marker)
    if pos == -1:
        raise RuntimeError(f"非标准 DFLJPG（无 APP15）: {img_path}")

    old_len = struct.unpack('>H', data[pos + 2:pos + 4])[0]
    old_pickle = data[pos + 4:pos + 2 + old_len]
    dict_data = pickle.loads(old_pickle)

    dict_data['xseg_mask'] = buf
    for k in list(dict_data.keys()):
        if dict_data[k] is None:
            del dict_data[k]

    xseg_val = dict_data.get('xseg_mask')

    def _unumpy(obj):
        if isinstance(obj, np.ndarray):
            if obj is xseg_val:
                return obj
            if obj.dtype.kind in ('u', 'b'):
                return bytes(obj)
            return obj.tolist()
        if isinstance(obj, (np.floating, np.integer, np.bool_)):
            return obj.item()
        if isinstance(obj, dict):
            return {k: _unumpy(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_unumpy(i) for i in obj]
        return obj

    dict_data = _unumpy(dict_data)
    new_pickle = pickle.dumps(dict_data, protocol=2)
    new_pickle = new_pickle.replace(b'numpy._core', b'numpy.core')

    new_chunk_data = struct.pack('>H', len(new_pickle) + 2) + new_pickle
    old_total = 2 + old_len

    if len(new_chunk_data) == old_len:
        out = bytearray(data)
        out[pos + 2:pos + 2 + old_len] = new_chunk_data
    else:
        out = data[:pos] + marker + new_chunk_data + data[pos + old_total:]

    with open(img_path, 'wb') as f:
        f.write(out)


# ═══════════════════════════════════════════════════════════════
# XSegAugmenter 类
# ═══════════════════════════════════════════════════════════════

class XSegAugmenter:
    """XSeg / XSegLite ONNX 批量遮罩生成器。

    Args:
        input_path: 人脸集目录路径（必须是目录，不能是文件）。
        model_type: 模型类型 ``'XSeg'`` 或 ``'XSegLite'``。
        resolution: ONNX 推理分辨率（用户自行保证正确）。
        invert: 是否反转遮罩（0 ↔ 1）。
        workers: 线程/进程数，默认 CPU 核心数。
    """

    MODEL_PATHS = {
        'XSeg':     ('workspace', 'model', 'XSeg'),
        'XSegLite': ('workspace', 'model', 'XSegLite'),
    }

    MODEL_FILES = {
        'XSeg':     ['XSeg.onnx'],
        'XSegLite': ['xseglite.onnx'],
    }

    OUTPUT_RESOLUTION = 512

    def __init__(
        self,
        input_path: str | Path,
        model_type: str = 'XSeg',
        resolution: int = 256,
        invert: bool = False,
        workers: Optional[int] = None,
        trt: bool = False,
        trt_batch: int = 1,
        model_file: Optional[str] = None,
    ):
        self.input_path = Path(input_path)
        self.model_type = model_type
        self.resolution = resolution
        self.invert = invert
        self.workers = workers or cpu_count()
        self.trt = trt
        self.trt_batch = trt_batch
        self.model_file = model_file

        if not self.input_path.exists():
            raise FileNotFoundError(f"路径不存在: {self.input_path}")
        if not self.input_path.is_dir():
            raise NotADirectoryError(f"路径必须是目录: {self.input_path}")
        if model_type not in self.MODEL_PATHS:
            raise ValueError(
                f"不支持的模型类型: '{model_type}'。可选: {', '.join(self.MODEL_PATHS.keys())}"
            )
        self._input_nchw = False  # 将在 _create_session 中根据模型自动检测

    # ── 公开方法 ─────────────────────────────────────────────

    def run(self) -> Dict[str, Any]:
        """主入口：遍历目录，对每张图片执行推理并写入遮罩。"""
        images = self._scan_images()
        total = len(images)

        if total == 0:
            print(f"[!] 目录中没有 .jpg 文件: {self.input_path}")
            return {'total': 0, 'success': 0, 'failed': 0, 'failed_files': []}

        model_path = self._find_model_path()

        # TRT 模式：跳过 ONNX，直接使用 TensorRT BF16 batch 推理
        if self.trt and self.model_type == 'XSegLite':
            print(f"XSegLite TRT 编译不兼容，改走 ONNX Runtime CUDA")
            self.trt = False

        try:
            has_cuda = 'CUDAExecutionProvider' in _get_ort().get_available_providers()
        except Exception:
            has_cuda = False

        print(f"发现 {total} 张图片，模型: {self.model_type}，分辨率: {self.resolution}")
        print(f"工作数: {self.workers}  |  模式: {'串行' if self.workers <= 1 else '线程池'}")
        if self.invert:
            print("  遮罩反转: 开启")
        print(f"加载模型: {model_path}")
        print(f"  ONNX Runtime 可用提供者: {_get_ort().get_available_providers()}")

        # 如果没有 CUDA，尝试用系统 Python 重新执行（项目 Python 的 onnxruntime 可能无法加载 CUDA）
        if not has_cuda and self.workers > 1:
            # 搜索顺序: sys.executable → PATH 上的 python/python3 → 常见安装路径
            _candidates = [
                _os.environ.get('CUDA_PYTHON', ''),
                sys.executable,
            ]
            # 从 PATH 搜索 Python
            for _p in _os.environ.get('PATH', '').split(';'):
                for _name in ('python.exe', 'python3.exe', 'py.exe'):
                    _fp = _os.path.join(_p, _name)
                    if _os.path.isfile(_fp) and _fp not in _candidates:
                        _candidates.append(_fp)
            # 常见系统 Python 安装路径（兜底）
            for _root in (r'C:\Python312', r'C:\Python311', r'C:\Program Files\Python312',
                          r'C:\Program Files\Python311', r'C:\Users\%s\anaconda3' % _os.getlogin()):
                _candidates.append(_root + r'\python.exe')
            # 去重，保留第一个存在的
            _system_python = None
            for _p in _candidates:
                if _p and _os.path.isfile(_p) and _p != sys.executable:
                    _system_python = _p
                    break
            if _system_python:
                print(f"\n  [!] 当前 Python 无 CUDA, 尝试调用: {_system_python}")
                import subprocess as _sp
                _cmd = [
                    _system_python, '-m', 'DataAugmenter',
                    '-i', str(self.input_path),
                    '-m', self.model_type,
                    '-r', str(self.resolution),
                    '-w', str(self.workers),
                ]
                if self.invert:
                    _cmd.append('--invert')
                sys.exit(_sp.run(_cmd).returncode)
            else:
                print(f"\n  [!] 未找到系统 Python，将继续使用当前 Python（无 CUDA）")
                print(f"  [!] 可设置环境变量 CUDA_PYTHON 指定 CUDA 版 Python 路径")

        if self.workers <= 1:
            stats = self._run_serial(images, model_path)
        else:
            stats = self._run_threaded(images, model_path)

        if stats['failed'] > 0:
            self._move_failed(self.input_path, stats['failed_files'])
            print(f"  已移动 {stats['failed']} 个失败文件到 {self.input_path / '_failed_xseg'}")

        return stats

    def _run_trt(self, images, model_path=None):
        import tensorrt as _trt, torch
        import concurrent.futures, threading, time, tqdm
        from pathlib import Path

        engine_dir = Path(__file__).parent.parent / 'workspace' / 'model' / 'XSegLite'
        if self.model_file:
            engine_path = engine_dir / self.model_file
        else:
            engines = sorted(engine_dir.glob('*.engine'))
            engine_path = engines[-1] if engines else None
        if not engine_path or not engine_path.exists():
            print('[ERROR] .engine not found')
            return {'total': len(images), 'success': 0, 'failed': len(images), 'failed_files': []}

        R = self.resolution
        total = len(images)
        n_workers = min(self.workers, total)

        print(f'使用 TensorRT BF16 引擎')
        runtime = _trt.Runtime(_trt.Logger(_trt.Logger.WARNING))
        with open(engine_path, 'rb') as f:
            engine = runtime.deserialize_cuda_engine(f.read())
        if engine is None:
            print(f'  [ERROR] 引擎加载失败（版本不兼容），回退 ONNX Runtime')
            return (self._run_serial(images, model_path) if self.workers <= 1
                    else self._run_threaded(images, model_path))
        print(f'  模型: {engine_path.name}')
        print(f'  工作数: {n_workers}')
        if self.invert:
            print('  遮罩反转: 开启')

        path_strs = [str(p) for p in images]
        completed = [0]
        errors: list = []

        # Rolling timing samples for progress breakdown (pre/gpu/post)
        import collections as _col
        _timing = _col.deque(maxlen=max(n_workers * 10, 100))

        # ── 每个线程自己做 I/O，NVMe SSD 高队列深度下并发更快 ──

        # Per-thread TRT context + stream (lazy init)
        # Per-thread TRT context + stream (lazy init)
        _ctx_lock = threading.Lock()
        _ctx_local = threading.local()

        def _get_ctx():
            try:
                return (_ctx_local.ctx, _ctx_local.stream,
                        _ctx_local.gpu_pred, _ctx_local.gpu_logits)
            except AttributeError:
                with _ctx_lock:
                    _ctx_local.ctx = engine.create_execution_context()
                    _ctx_local.stream = torch.cuda.Stream()
                    _ctx_local.gpu_pred = torch.zeros((1, 1, R, R), device='cuda')
                    _ctx_local.gpu_logits = torch.zeros((1, 1, R, R), device='cuda')
                    _ctx_local.ctx.set_tensor_address('logits', _ctx_local.gpu_logits.data_ptr())
                    _ctx_local.ctx.set_tensor_address('pred', _ctx_local.gpu_pred.data_ptr())
                    _ctx_local.ctx.set_input_shape('input', (1, 3, R, R))
                return (_ctx_local.ctx, _ctx_local.stream,
                        _ctx_local.gpu_pred, _ctx_local.gpu_logits)

        def _worker(img_path):
            t0 = time.perf_counter()
            try:
                # 一次读取，供 decode 和 APP15 修改共用
                with open(img_path, 'rb') as f:
                    jpeg_bytes = f.read()
                bgr = cv2.imdecode(np.frombuffer(jpeg_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
                if bgr is None:
                    errors.append((Path(img_path).name, "无法读取"))
                    return

                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                resized = cv2.resize(rgb, (R, R),
                                     interpolation=cv2.INTER_LINEAR).astype(np.float32)
                inp = np.transpose(resized, (2, 0, 1))[None] / 255.0
                t1 = time.perf_counter()

                ctx, stream, gpu_pred, gpu_logits = _get_ctx()
                # 全部 GPU 操作在一个 stream，消除跨 stream 数据依赖
                with torch.cuda.stream(stream):
                    gpu_in = torch.from_numpy(inp).cuda()
                    ctx.set_tensor_address('input', gpu_in.data_ptr())
                    ctx.execute_async_v3(stream.cuda_stream)
                stream.synchronize()
                t2 = time.perf_counter()

                # 取 logits 过 _postprocess（512 二值化），与 ONNX 路径一致
                raw_out = gpu_logits.cpu().numpy()
                mask = self._postprocess(raw_out)
                _save_mask_fast(img_path, mask, pre_read_data=jpeg_bytes)
                t3 = time.perf_counter()

                _timing.append(((t1 - t0) * 1000,
                                (t2 - t1) * 1000,
                                (t3 - t2) * 1000))

            except Exception as e:
                errors.append((Path(img_path).name, str(e)))
            finally:
                completed[0] += 1

        pbar = tqdm.tqdm(total=total, desc="", unit="img", ascii=True)
        with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as ex:
            for p in path_strs:
                ex.submit(_worker, p)
            while completed[0] < total:
                pbar.n = completed[0]
                if _timing:
                    n_t = len(_timing)
                    pbar.set_postfix(
                        pre=f"{sum(t[0] for t in _timing)/n_t:.1f}ms",
                        gpu=f"{sum(t[1] for t in _timing)/n_t:.2f}ms",
                        post=f"{sum(t[2] for t in _timing)/n_t:.1f}ms",
                    )
                pbar.refresh()
                sys.stdout.flush()
                time.sleep(0.1)
            pbar.n = total
        pbar.close()

        success = total - len(errors)
        failed_files = [name for name, _ in errors]
        if errors:
            for name, err in errors:
                print(f"✗ 失败: {name} — {err}")

        print(f"完成: {success}/{total} 成功")
        resp = {'total': total, 'success': success,
                'failed': len(errors), 'failed_files': failed_files}
        return resp
    def _create_session(self, model_path: Path):
        """Create ONNX session (prefer TRT EP for subgraph-based acceleration)."""
        # 添加 nvinfer_10.dll 路径到 PATH（使 TRT EP 可用）
        _trt_paths = [
            Path(__file__).parent.parent / 'python',  # 项目自带的 TRT DLL
            Path(r'C:\MySoftware\CrackEverything\DeepFaceLab-trt'),
            Path(r'C:\MySoftware\CrackEverything\OEMcongyun'),
        ]
        for _p in _trt_paths:
            if _p.exists() and str(_p) not in _os.environ.get('PATH', ''):
                _os.environ['PATH'] = str(_p) + ';' + _os.environ.get('PATH', '')

        so = _get_ort().SessionOptions()
        so.intra_op_num_threads = 1
        so.inter_op_num_threads = 1
        available = _get_ort().get_available_providers()
        providers = ['CPUExecutionProvider']
        for p in ['CUDAExecutionProvider', 'TensorrtExecutionProvider',
                  'DmlExecutionProvider']:
            if p in available:
                providers.insert(0, p)
        try:
            session = _get_ort().InferenceSession(
                str(model_path), sess_options=so, providers=providers,
            )
        except Exception as e:
            print(f"  CUDA 提供者加载失败: {e}")
            # 输出诊断信息
            try:
                import subprocess, sys
                import ctypes
                # CUDA 版本
                try:
                    cudart = ctypes.cdll.LoadLibrary('cudart64_12.dll') if hasattr(ctypes, 'cdll') else None
                    if cudart:
                        ver = ctypes.c_int()
                        cudart.cudaRuntimeGetVersion(ctypes.byref(ver))
                        print(f"  CUDA Runtime: {ver.value // 1000}.{(ver.value % 1000) // 10}")
                except:
                    print(f"  CUDA Runtime: 未检测到")
                # cuDNN 版本
                try:
                    cudnn = ctypes.cdll.LoadLibrary('cudnn64_9.dll') if hasattr(ctypes, 'cdll') else None
                    if cudnn:
                        ver = ctypes.c_size_t()
                        cudnn.cudnnGetVersion(ctypes.byref(ver))
                        print(f"  cuDNN: {ver.value}")
                except:
                    print(f"  cuDNN: 未检测到")
                print(f"  ONNX Runtime: {_get_ort().__version__ if _get_ort() else 'N/A'}")
            except:
                pass
            print(f"  回退到 CPU 推理（如需 GPU 加速请安装匹配的 CUDA 版本）")
            session = _get_ort().InferenceSession(
                str(model_path), sess_options=so, providers=['CPUExecutionProvider'],
            )
        inp = session.get_inputs()[0]
        shape = inp.shape
        self._input_nchw = len(shape) == 4 and shape[1] in (1, 3)
        print(f"  推理提供者: {session.get_providers()[0]}")
        print(f"  输入: {inp.name} shape={shape} type={inp.type}")
        return session

    # ── 串行 ─────────────────────────────────────────────────

    def _run_serial(self, images: List[Path], model_path: Path) -> Dict[str, Any]:
        session = self._create_session(model_path)
        input_name = session.get_inputs()[0].name
        total = len(images)
        success = failed = 0
        failed_files: List[str] = []

        pbar = tqdm.tqdm(total=total, desc="", unit="img", ascii=True)
        for img_path in images:
            try:
                if img_path.stat().st_size == 0:
                    raise RuntimeError("空文件 (0 byte)")
                bgr = cv2.imread(str(img_path))
                if bgr is None:
                    raise RuntimeError("无法读取图片")
                inp = self._preprocess(bgr)
                raw = session.run(None, {input_name: inp})[0]
                mask = self._postprocess(raw)
                _save_mask_fast(str(img_path), mask)
                success += 1
            except Exception as e:
                msg = str(e).split('\n')[0]
                pbar.write(f"[FAIL] {img_path.name} - {msg}")
                failed += 1
                failed_files.append(img_path.name)
            pbar.update(1)
            sys.stdout.flush()
        pbar.close()

        print(f"完成: {success}/{total} 成功")
        return {'total': total, 'success': success,
                'failed': failed, 'failed_files': failed_files}

    # ── 线程池（CUDA 模式）────────────────────────────────────

    def _run_threaded(self, images: List[Path], model_path: Path) -> Dict[str, Any]:
        """线程池推理：共享一个 CUDA session，零 GPU 竞争。

        用一个全局计数器 completed 跟踪进度，主线程轮询刷新进度条，
        避免 as_completed 循环中的竞争。
        """
        session = self._create_session(model_path)
        input_name = session.get_inputs()[0].name

        total = len(images)
        n_workers = min(self.workers, total)
        path_strs = [str(p) for p in images]

        # 共享计数器（list 可变容器，CPython GIL 下自增原子安全）
        completed = [0]
        errors: List[tuple] = []

        def _worker(img_path: str):
            try:
                # 预检：跳过空文件（避免 OpenCV imdecode 抛异常）
                fsize = Path(img_path).stat().st_size
                if fsize == 0:
                    errors.append((Path(img_path).name, "空文件 (0 byte)"))
                    return

                bgr = cv2.imread(img_path)
                if bgr is None:
                    errors.append((Path(img_path).name, "无法读取图片"))
                    return

                inp = self._preprocess(bgr)
                raw = session.run(None, {input_name: inp})[0]
                mask = self._postprocess(raw)
                _save_mask_fast(img_path, mask)
            except Exception as e:
                # 清理异常信息：去掉 OpenCV 文件路径等噪音
                msg = str(e).split('\n')[0].split('error:')[0].strip()
                errors.append((Path(img_path).name, msg or str(e)[:120]))
            finally:
                completed[0] += 1

        pbar = tqdm.tqdm(total=total, desc="", unit="img", ascii=True)
        with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as ex:
            for p in path_strs:
                ex.submit(_worker, p)

            # 主线程轮询计数器刷新进度条（避免 as_completed 竞争）
            while completed[0] < total:
                if completed[0] != pbar.n:
                    pbar.n = completed[0]
                    pbar.refresh()
                    sys.stdout.flush()
                time.sleep(0.1)

            pbar.n = total
        pbar.close()

        success = total - len(errors)
        failed_files = [name for name, _ in errors]
        if errors:
            for name, err in errors:
                print(f"✗ 失败: {name} — {err}")

        print(f"完成: {success}/{total} 成功")
        return {'total': total, 'success': success,
                'failed': len(errors), 'failed_files': failed_files}

    @staticmethod
    def _move_failed(input_path: Path, failed_files: List[str], tag: str = "xseg"):
        """将失败的文件移动到 _failed_<tag>/ 目录。"""
        if not failed_files:
            return
        failed_dir = input_path / f"_failed_{tag}"
        failed_dir.mkdir(exist_ok=True)
        for name in failed_files:
            src = input_path / name
            if src.exists():
                src.rename(failed_dir / name)

    # ── 工具方法 ─────────────────────────────────────────────

    def _scan_images(self) -> List[Path]:
        return sorted([
            f for f in self.input_path.iterdir()
            if f.is_file() and f.suffix.lower() in ('.jpg', '.jpeg')
            and not f.stem.endswith('_mask')
        ])

    def _find_model_path(self) -> Path:
        rel_parts = self.MODEL_PATHS[self.model_type]
        model_dir = project_root.joinpath(*rel_parts)
        if not model_dir.exists():
            raise FileNotFoundError(
                f"模型目录不存在: {model_dir}\n"
                f"请将 {self.model_type}.onnx 放入 {model_dir}")
        candidates = self.MODEL_FILES[self.model_type]
        for name in candidates:
            p = model_dir / name
            if p.exists():
                return p
        raise FileNotFoundError(
            f"在 {model_dir} 中未找到 {self.model_type} ONNX 模型文件。\n"
            f"期望的文件: {', '.join(candidates)}\n"
            f"目录内容: {[f.name for f in model_dir.iterdir()]}")

    def _preprocess(self, bgr: np.ndarray) -> np.ndarray:
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (self.resolution, self.resolution),
                             interpolation=cv2.INTER_LINEAR)
        normalized = resized.astype(np.float32) / 255.0
        if self._input_nchw:
            return np.transpose(normalized, (2, 0, 1))[None, ...]
        return normalized[None, ...]

    def _postprocess(self, raw_output: np.ndarray) -> np.ndarray:
        mask = 1.0 / (1.0 + np.exp(-raw_output))
        mask = np.squeeze(mask)
        while mask.ndim > 2:
            mask = mask[0]
        mask = cv2.resize(mask, (self.OUTPUT_RESOLUTION, self.OUTPUT_RESOLUTION),
                          interpolation=cv2.INTER_LINEAR)
        mask = (mask > 0.5).astype(np.float32)
        if self.invert:
            mask = 1.0 - mask
        return mask