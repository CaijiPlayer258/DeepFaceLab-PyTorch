#!/usr/bin/env python3
"""XSegLite ONNX 导出 + 批量推理 Benchmark

用法:
    python tools/benchmark_xseglite.py
    python tools/benchmark_xseglite.py --model-dir workspace/model/XSegLite
    python tools/benchmark_xseglite.py --src-dir workspace/data_src/aligned

按键:  ESC=退出  Space=暂停/继续
"""
import sys, os, time, argparse, glob
from pathlib import Path
import numpy as np
import cv2

# ── 路径 ──────────────────────────────────────────────
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model-dir', default=str(REPO / 'workspace' / 'model' / 'XSegLite'),
                   help='模型目录（包含 .pth 权重）')
    p.add_argument('--src-dir', default=str(REPO / 'workspace' / 'data_src' / 'aligned'),
                   help='测试图片目录')
    p.add_argument('--resolution', type=int, default=256, help='推理分辨率')
    args = p.parse_args()

    model_dir = Path(args.model_dir)
    src_dir = Path(args.src_dir)

    if not src_dir.exists():
        print(f'[ERROR] 图片目录不存在: {src_dir}')
        return

    # 查找图片
    exts = ('*.jpg', '*.jpeg', '*.png', '*.bmp')
    img_paths = []
    for ext in exts:
        img_paths.extend(sorted(src_dir.glob(ext)))
        img_paths.extend(sorted(src_dir.glob(ext.upper())))
    if not img_paths:
        print(f'[ERROR] {src_dir} 下没有图片')
        return
    print(f'找到 {len(img_paths)} 张图片')

    # ── 加载 PyTorch 模型 ──────────────────────────
    import torch
    from core.xseglite_torch import XSegLiteTorch

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'设备: {device}')

    model = XSegLiteTorch(3, 32).eval().to(device)

    # 加载权重
    pth_files = list(model_dir.glob('*.pth'))
    if pth_files:
        ckpt = torch.load(str(pth_files[0]), map_location=device)
        if 'model' in ckpt:
            model.load_state_dict(ckpt['model'])
        else:
            model.load_state_dict(ckpt)
        print(f'加载权重: {pth_files[0].name}')
    else:
        print('[WARN] 未找到 .pth 权重，使用随机初始化的模型')

    # ── 导出 ONNX ──────────────────────────────────
    onnx_path = REPO / 'workspace' / 'model' / 'xseglite_bench.onnx'
    dummy = torch.randn(1, 3, args.resolution, args.resolution).to(device)
    torch.onnx.export(model, dummy, str(onnx_path),
                      input_names=['input'],
                      output_names=['logits', 'pred'],
                      dynamic_axes={'input': {0: 'batch'}},
                      opset_version=18,
                      dynamo=False)
    onnx_size = os.path.getsize(onnx_path) / 1024 / 1024
    print(f'ONNX 导出: {onnx_path.name} ({onnx_size:.0f}MB)')

    # ── ONNX Runtime ───────────────────────────────
    import onnxruntime as ort
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    sess = ort.InferenceSession(str(onnx_path), providers=providers)
    input_name = sess.get_inputs()[0].name
    print(f'ONNX Runtime 就绪 (provider={sess.get_providers()[0]})')

    # ── Benchmark ──────────────────────────────────
    R = args.resolution
    times_ms = []
    paused = False

    print(f'\n按 ESC 退出，Space 暂停/继续')
    print(f'总共 {len(img_paths)} 张，循环测试中...\n')

    try:
        while True:
            for idx, img_path in enumerate(img_paths):
                if paused:
                    key = cv2.waitKey(100) & 0xFF
                    if key == 27: break
                    if key == ord(' '): paused = False
                    continue

                # 读取并预处理
                img_bgr = cv2.imread(str(img_path))
                if img_bgr is None:
                    continue
                h0, w0 = img_bgr.shape[:2]
                img_resized = cv2.resize(img_bgr, (R, R))
                inp = img_resized.astype(np.float32).transpose(2, 0, 1)[None, ...] / 255.0

                # 推理
                t0 = time.perf_counter()
                logits, pred = sess.run(None, {input_name: inp})
                t1 = time.perf_counter()
                ms = (t1 - t0) * 1000
                times_ms.append(ms)

                # 后处理
                mask = pred[0, 0]  # (R, R) float32 [0,1]
                mask_resized = cv2.resize(mask, (w0, h0))
                mask_bgr = (np.clip(mask, 0, 1) * 255).astype(np.uint8)
                mask_bgr = cv2.cvtColor(mask_bgr, cv2.COLOR_GRAY2BGR)

                # 遮罩下的人脸
                mask_3 = np.stack([mask] * 3, axis=-1)
                face_masked = img_resized.astype(np.float32) / 255.0 * mask_3
                face_masked = (np.clip(face_masked, 0, 1) * 255).astype(np.uint8)

                # 拼图显示 (原图 | 遮罩 | 遮罩下的人脸)
                gap = 4
                vis = np.zeros((R, R * 3 + gap * 2, 3), dtype=np.uint8)
                vis[:, :R] = img_resized
                vis[:, R + gap:2 * R + gap] = mask_bgr
                vis[:, 2 * R + 2 * gap:] = face_masked

                # 信息
                fps = 1000 / ms if ms > 0 else 0
                avg_ms = np.mean(times_ms[-100:]) if times_ms else 0
                avg_fps = 1000 / avg_ms if avg_ms > 0 else 0
                info = [
                    f'{ms:.1f}ms ({fps:.0f}fps)',
                    f'avg({len(times_ms)}): {avg_ms:.1f}ms ({avg_fps:.0f}fps)',
                    f'{idx + 1}/{len(img_paths)}',
                    'PAUSED' if paused else '',
                ]
                for li, text in enumerate(info):
                    if text:
                        cv2.putText(vis, text, (4, R - 8 - li * 16),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

                # 分栏线
                for col in [0, R + gap, 2 * R + 2 * gap]:
                    cv2.line(vis, (col + R // 2, 0), (col + R // 2, R), (0, 255, 0), 1)
                    cv2.line(vis, (col, R // 2), (col + R, R // 2), (0, 255, 0), 1)

                cv2.imshow('XSegLite Benchmark - ESC退出 Space暂停', vis)
                key = cv2.waitKey(1) & 0xFF
                if key == 27:
                    raise KeyboardInterrupt
                if key == ord(' '):
                    paused = True

    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()

    # ── 输出跑分 ──────────────────────────────────
    print('\n' + '=' * 60)
    print('          XSegLite Benchmark 结果')
    print('=' * 60)
    if times_ms:
        times = np.array(times_ms)
        print(f'  总推理次数:  {len(times)}')
        print(f'  平均耗时:    {times.mean():.2f}ms')
        print(f'  平均 FPS:    {1000 / times.mean():.1f}')
        print(f'  中位数:      {np.median(times):.2f}ms')
        print(f'  标准差:      {times.std():.2f}ms')
        print(f'  最小值:      {times.min():.2f}ms')
        print(f'  最大值:      {times.max():.2f}ms')
        print(f'  P50:         {np.percentile(times, 50):.2f}ms')
        print(f'  P95:         {np.percentile(times, 95):.2f}ms')
        print(f'  P99:         {np.percentile(times, 99):.2f}ms')
    else:
        print('  无数据')
    print(f'  ONNX 文件:   {onnx_size:.0f}MB')
    print(f'  分辨率:      {R}x{R}')
    print(f'  设备:        {device}')
    print(f'  Provider:    {sess.get_providers()[0]}')
    print('=' * 60)


if __name__ == '__main__':
    main()
