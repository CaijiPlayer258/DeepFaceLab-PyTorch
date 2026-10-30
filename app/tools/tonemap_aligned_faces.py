"""
为已提取的人脸（aligned 目录）批量做 HDR→SDR 色调映射。
用于杜比视界/HDR 视频切脸后的颜色纠正。

用法：
    python tools/tonemap_aligned_faces.py -i workspace/data_dst/aligned -o workspace/data_dst/aligned_tonemap
    python tools/tonemap_aligned_faces.py -i workspace/data_dst/aligned --inplace
"""
import argparse, subprocess, sys, os, time
from pathlib import Path
from multiprocessing import cpu_count

FFMPEG = Path(__file__).parent.parent / "ffmpeg" / "ffmpeg.exe"
# bat709 参数（已验证和杜比视界转码 bat 一致）
FILTER = (
    "libplacebo=tonemapping=hable:"
    "colorspace=1:color_trc=2:color_primaries=1:"
    "range=tv:dithering=blue:format=yuv420p"
)

def main():
    parser = argparse.ArgumentParser(description="批量色调映射已提取人脸")
    parser.add_argument("-i", "--input", required=True, help="aligned 目录")
    parser.add_argument("-o", "--output", default=None, help="输出目录（默认覆盖原图）")
    parser.add_argument("--inplace", action="store_true", help="直接覆盖原图")
    parser.add_argument("-j", "--jobs", type=int, default=cpu_count(), help="并行数")
    args = parser.parse_args()

    input_dir = Path(args.input)
    if not input_dir.is_dir():
        print(f"错误: 目录不存在: {input_dir}")
        return 1

    if args.inplace:
        output_dir = input_dir
    elif args.output:
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)
    else:
        output_dir = input_dir.parent / (input_dir.name + "_tonemap")
        output_dir.mkdir(parents=True, exist_ok=True)

    images = sorted(input_dir.glob("*.jpg")) + sorted(input_dir.glob("*.png"))
    if not images:
        print(f"警告: 未找到图片文件")
        return 0

    print(f"找到 {len(images)} 张人脸")
    print(f"输入: {input_dir}")
    print(f"输出: {output_dir}")
    print(f"滤镜: {FILTER}")
    print()

    success = 0
    failed = 0
    start = time.time()

    for i, img_path in enumerate(images, 1):
        out_path = output_dir / img_path.name
        cmd = [
            str(FFMPEG), "-y",
            "-i", str(img_path),
            "-vf", FILTER,
            "-q:v", "2",
            str(out_path),
        ]
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=30)

        if r.returncode == 0 and out_path.exists() and out_path.stat().st_size > 100:
            success += 1
        else:
            failed += 1
            print(f"  [{i}/{len(images)}] 失败: {img_path.name}")

        # 每 100 张或完成时显示进度
        if i % 100 == 0 or i == len(images):
            elapsed = time.time() - start
            rate = i / elapsed
            print(f"  [{i}/{len(images)}] 成功={success} 失败={failed}  "
                  f"耗时={elapsed:.0f}s 速度={rate:.0f}张/s")

    elapsed = time.time() - start
    print(f"\n完成! 处理 {len(images)} 张, 成功 {success}, 失败 {failed}")
    print(f"总耗时: {elapsed:.1f}s ({elapsed/60:.1f}min)")
    if success > 0:
        rate = success / elapsed
        print(f"平均速度: {rate:.1f} 张/s")

if __name__ == "__main__":
    sys.exit(main())
