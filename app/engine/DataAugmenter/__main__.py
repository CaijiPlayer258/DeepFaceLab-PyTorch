
"""
DataAugmenter CLI — XSeg/XSegLite 遮罩批量应用入口

Usage:
    python -m DataAugmenter -i <faceset_dir> -m XSeg -r 256
    python -m DataAugmenter -i <faceset_dir> -m XSegLite -r 256 --invert
"""

import argparse
import sys
import traceback
from pathlib import Path


def main():
    # 所有输出同时写入日志文件（GUI 新窗口可能吞掉 stdout）
    _log = open(Path(__file__).parent / "_last_run.log", "w", encoding="utf-8")
    def _log_print(*a, **kw):
        print(*a, file=_log, **kw)
        print(*a, **kw)
    _log_print("[DataAugmenter] 启动...")
    sys.stdout.flush()
    try:
        _log_print("[DataAugmenter] 导入模块...")
        from DataAugmenter import XSegAugmenter
        _log_print("[DataAugmenter] 模块导入成功")
    except Exception as e:
        import traceback as _tb
        _log_print(f"[FATAL] 导入失败: {e}")
        _tb.print_exc(file=_log)
        _log.close()
        sys.exit(1)
    parser = argparse.ArgumentParser(
        description="DataAugmenter — XSeg/XSegLite 遮罩批量应用工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python -m DataAugmenter -i "workspace/data_dst/aligned" -m XSeg -r 256
  python -m DataAugmenter -i "workspace/data_src/aligned" -m XSegLite -r 256 --invert
  python -m DataAugmenter -i "workspace/data_dst/aligned" -m XSeg -r 512
        """,
    )

    parser.add_argument(
        '-i', '--input',
        type=str,
        required=True,
        help='人脸集目录路径（必须为目录，不能是文件）',
    )
    parser.add_argument(
        '-m', '--model-type',
        type=str,
        choices=['XSeg', 'XSegLite'],
        default='XSeg',
        help='模型类型: XSeg (默认) 或 XSegLite',
    )
    parser.add_argument(
        '-r', '--resolution',
        type=int,
        default=256,
        help='ONNX 推理分辨率 (默认: 256)，用户自行保证与模型匹配',
    )
    parser.add_argument(
        '--invert',
        action='store_true',
        default=False,
        help='遮罩反转: 将 0/1 互换',
    )
    parser.add_argument(
        '-w', '--workers',
        type=int,
        default=None,
        help='工作进程数 (默认: CPU 核心数)，设为 1 使用单进程',
    )
    parser.add_argument(
        '--trt',
        action='store_true',
        default=False,
        help='启用 TensorRT BF16 推理（仅 XSegLite，需 .engine 文件）',
    )
    parser.add_argument(
        '--trt-batch',
        type=int,
        default=1,
        help='TRT 推理 batch 尺寸（自动从文件名解析，也可手动指定）',
    )
    parser.add_argument(
        '--model-file',
        type=str,
        default=None,
        help='模型文件名（如 XSegLite__bf16_sm860_bs8_trt111010_hash.engine），在 workspace/model/XSegLite/ 下',
    )

    args = parser.parse_args()

    # 兼容 cmd.exe 传过来的引号
    raw_input = args.input.strip('"\'')
    input_path = Path(raw_input)
    if not input_path.exists():
        print(f"错误: 路径不存在 — {input_path}")
        sys.exit(1)
    if not input_path.is_dir():
        print(f"错误: 路径必须是目录 — {input_path}")
        sys.exit(1)

    try:
        augmenter = XSegAugmenter(
            input_path=input_path,
            model_type=args.model_type,
            resolution=args.resolution,
            invert=args.invert,
            workers=args.workers,
            trt=args.trt,
            trt_batch=args.trt_batch,
            model_file=args.model_file,
        )
        stats = augmenter.run()
        _log_print(f"\n统计: 共 {stats['total']} 张, "
              f"成功 {stats['success']} 张, "
              f"失败 {stats['failed']} 张")
    except Exception as e:
        _log_print(f"错误: {e}")
        traceback.print_exc(file=_log)
        sys.exit(1)
    finally:
        _log_print("[DataAugmenter] 结束")
        _log.close()
        input("\n按 Enter 键退出...")


if __name__ == '__main__':
    main()

