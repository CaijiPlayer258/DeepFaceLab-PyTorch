#!/usr/bin/env python
"""
通用 ONNX → TensorRT BF16 引擎编译工具。

用法:
    python tools/export_onnx_to_trt.py model.onnx --name model_name [options]

引擎命名:
    {name}_bs{batch}_sm{SM}_trt{VER}_{MD5[:16]}.engine

示例:
    python tools/export_onnx_to_trt.py modelhub/onnx/ArcFace/w600k_mbf.onnx --name w600k_mbf
"""
import os, sys, time, hashlib, argparse, warnings
from pathlib import Path

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

try:
    import tensorrt as trt
except ImportError:
    print('[ERROR] tensorrt not installed. Run: pip install tensorrt')
    sys.exit(1)

import torch
import onnx

warnings.filterwarnings('ignore')
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)


def get_sm(device_id=0):
    """获取当前 GPU 的 SM 版本字符串（如 sm860）。"""
    props = torch.cuda.get_device_properties(device_id)
    return f'sm{props.major}{props.minor}0'


def get_trt_version():
    """获取 TRT 版本字符串（如 trt111010）。"""
    return trt.__version__.replace('.', '')[:6]


def compute_md5(path):
    """计算 ONNX 文件的 MD5（前 16 位）。"""
    return hashlib.md5(open(path, 'rb').read()).hexdigest()[:16]


def print_io_info(onnx_path):
    """打印 ONNX 模型的 I/O 信息。"""
    model = onnx.load(onnx_path)
    g = model.graph
    print(f'\nONNX: {Path(onnx_path).name}')
    print(f'├ Graph: {g.name}')
    print(f'├ Opset: {model.opset_import[0].version if model.opset_import else "?"}')
    print(f'├ Inputs:')
    for inp in g.input:
        shape = [d.dim_value if d.dim_value > 0 else -1 for d in inp.type.tensor_type.shape.dim]
        print(f'│   {inp.name}: {shape}')
    print(f'├ Outputs:')
    for out in g.output:
        shape = [d.dim_value if d.dim_value > 0 else -1 for d in out.type.tensor_type.shape.dim]
        print(f'│   {out.name}: {shape}')
    print(f'└ Initializers: {len(g.initializer)}')
    return model


def build_engine(onnx_path, batch=1, workspace_gb=6, input_shapes=None, fp16=False):
    """ONNX → TRT engine，返回 engine bytes。

    Args:
        input_shapes: dict {tensor_name: [h, w]} 或 None（未知维度默认 640）
        fp16: 使用 FP16 精度（默认 BF16，TRT 11+）
    """
    # Build network
    flag = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH) \
        if hasattr(trt.NetworkDefinitionCreationFlag, 'EXPLICIT_BATCH') else 0
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(flag)
    parser = trt.OnnxParser(network, TRT_LOGGER)

    print('\nParsing ONNX...')
    t0 = time.time()
    # 使用 onnx.load() 加载（支持外部 .data 权重文件）
    model_proto = onnx.load(str(onnx_path))
    model_bytes = model_proto.SerializeToString()
    if not parser.parse(model_bytes):
        for i in range(min(parser.num_errors, 10)):
            print(f'  ERR {i}: {parser.get_error(i)}')
        sys.exit(1)
    t1 = time.time()
    print(f'├ Parsed: {t1-t0:.1f}s, {network.num_layers} layers')
    for i in range(network.num_inputs):
        t = network.get_input(i)
        print(f'├ Input: {t.name} {t.shape}')
    for i in range(network.num_outputs):
        t = network.get_output(i)
        print(f'├ Output: {t.name} {t.shape}')

    # Config
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_gb << 30)

    # Precision: FP16（--fp16）或 BF16（默认，TRT 11+）
    if fp16:
        config.set_flag(trt.BuilderFlag.FP16)
        print('├ FP16: enabled')
    elif hasattr(trt.BuilderFlag, 'BF16'):
        config.set_flag(trt.BuilderFlag.BF16)
        print('├ BF16: enabled')
    else:
        print('├ BF16: auto (TRT 11+)')

    # Optimization profile
    profile = builder.create_optimization_profile()
    for i in range(network.num_inputs):
        t = network.get_input(i)
        shape = list(t.shape)
        # 替换未知维度（-1 或 0）为默认值
        default_hw = 640  # 检测模型常用
        if input_shapes and t.name in input_shapes:
            override = input_shapes[t.name]
            for i_dim in range(len(shape)):
                if shape[i_dim] <= 0 and i_dim >= 2:  # 非 batch 维度的未知 dim
                    shape[i_dim] = override[i_dim - 2] if len(override) > i_dim - 2 else default_hw
        else:
            for i_dim in range(len(shape)):
                if i_dim == 0 and shape[i_dim] <= 0:
                    shape[i_dim] = 1 if shape[i_dim] == 0 else -1  # 0→1(explicit), -1→-1(dynamic batch)
                elif i_dim >= 2 and shape[i_dim] <= 0:
                    shape[i_dim] = default_hw

        if shape[0] == -1:
            min_shape = [1] + shape[1:]
            opt_shape = [batch] + shape[1:]
            max_shape = [batch] + shape[1:]
            profile.set_shape(t.name, min_shape, opt_shape, max_shape)
            print(f'├ Profile[{t.name}]: dynamic batch, opt={opt_shape}')
        else:
            profile.set_shape(t.name, shape, shape, shape)
    config.add_optimization_profile(profile)

    # Build
    print('Building engine (this may take a while)...')
    t0 = time.time()
    engine = builder.build_serialized_network(network, config)
    if not engine:
        print('[ERROR] Build failed')
        sys.exit(1)
    engine_bytes = bytes(engine)
    t1 = time.time()
    print(f'├ Build: {t1-t0:.1f}s, {len(engine_bytes)/1024/1024:.1f} MB')
    return engine_bytes


def save_engine(engine_bytes, onnx_path, name, batch=1, output_dir=None):
    """保存 engine 并命名。"""
    md5 = compute_md5(onnx_path)
    sm = get_sm()
    trt_ver = get_trt_version()
    filename = f'{name}_bs{batch}_{sm}_trt{trt_ver}_{md5}.engine'
    if output_dir is None:
        output_dir = Path(onnx_path).parent
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    dst = output_dir / filename
    with open(dst, 'wb') as f:
        f.write(engine_bytes)
    print(f'├ Saved: {dst.name}')
    print(f'├ Size: {len(engine_bytes)/1024/1024:.1f} MB')
    return str(dst)


def verify_engine(engine_path, input_shapes=None):
    """加载 engine 并跑一次推理验证（抄自 FastFaceAlign export_trt.py --verify）。"""
    from xlib.trt import TRTInferenceSession
    sess = TRTInferenceSession(engine_path)
    inp = sess.get_inputs()[0]
    shape = [s if s > 0 else 1 for s in inp.shape]
    if input_shapes and inp.name in input_shapes:
        ov = input_shapes[inp.name]
        for i in range(2, len(shape)):
            if i - 2 < len(ov):
                shape[i] = ov[i - 2]
    x = np.random.randn(*shape).astype(np.float32)
    outs = sess.run(None, {inp.name: x})
    print(f'[verify] engine OK: {inp.name}{tuple(shape)} -> ' +
          ', '.join(f'{o.shape}' for o in outs))


def main():
    p = argparse.ArgumentParser(description='ONNX → TensorRT Engine (BF16/FP16)')
    p.add_argument('onnx', help='ONNX 模型路径')
    p.add_argument('--name', required=True, help='模型名（用于引擎文件名）')
    p.add_argument('--batch', type=int, default=1, help='最大 batch size（默认 1）')
    p.add_argument('--workspace', type=int, default=6, help='Workspace in GB（默认 6）')
    p.add_argument('--fp16', action='store_true', help='使用 FP16 精度（默认 BF16）')
    p.add_argument('--verify', action='store_true', help='编译后加载引擎做一次推理验证')
    p.add_argument('--output-dir', default=None, help='引擎输出目录（默认 ONNX 同目录）')
    p.add_argument('--input-shape', default=None, help='指定输入形状覆盖未知维度，格式: input_name:h,w。多个用;分割')
    p.add_argument('--info', action='store_true', help='仅打印 I/O 信息，不编译')
    args = p.parse_args()

    onnx_path = Path(args.onnx)
    if not onnx_path.exists():
        print(f'[ERROR] ONNX not found: {onnx_path}')
        sys.exit(1)

    if args.info:
        print_io_info(onnx_path)
        return

    print(f'Model: {args.name}')
    print(f'Batch: {args.batch}')
    print(f'SM: {get_sm()}  TRT: {get_trt_version()}')

    # 解析 input-shape 参数
    input_shapes = None
    if args.input_shape:
        input_shapes = {}
        for part in args.input_shape.split(';'):
            name_dims = part.split(':')
            if len(name_dims) == 2:
                name, dims_str = name_dims
                dims = [int(x) for x in dims_str.split(',')]
                input_shapes[name] = dims

    engine_bytes = build_engine(str(onnx_path), args.batch, args.workspace, input_shapes, fp16=args.fp16)
    dst = save_engine(engine_bytes, str(onnx_path), args.name, args.batch, args.output_dir)
    if args.verify:
        verify_engine(dst, input_shapes)
    print('Done.')


if __name__ == '__main__':
    main()
