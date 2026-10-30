#!/usr/bin/env python
"""
PAK 文件解包工具 — 将 faceset.pak 解包为原始图片目录。

用法:
    python tools/unpack_pak.py <faceset_dir>
    python tools/unpack_pak.py workspace/data_src/aligned

解包后的文件输出到 faceset_dir/ 下（与 .pak 同目录）。
支持按人物子目录（person_name）分目录存放。
"""
import sys, struct, pickle
from pathlib import Path


def unpack_pak(pak_path: Path, output_dir: Path = None):
    """
    解包 faceset.pak 到 output_dir。

    Args:
        pak_path: .pak 文件路径
        output_dir: 输出目录（默认与 pak 同目录）
    """
    pak_path = Path(pak_path)
    if not pak_path.exists():
        print(f"[!] 文件不存在: {pak_path}")
        return False

    if output_dir is None:
        output_dir = pak_path.parent
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    # ── 解析 pak 头 ──
    with open(pak_path, "rb") as f:
        version = struct.unpack("Q", f.read(8))[0]
        print(f"[PAK] 版本: {version}")

        sizeof_samples = struct.unpack("Q", f.read(8))[0]
        print(f"[PAK] 样本数: ? (从 pickle 解析)")

        samples_configs = pickle.loads(f.read(sizeof_samples))
        print(f"[PAK] 样本数: {len(samples_configs)}")

        samples = []
        for sc in samples_configs:
            # 重新 pickle 再解包，兼容旧版 numpy 类型
            sc2 = pickle.loads(pickle.dumps(sc))
            samples.append(sc2)

        # 读取偏移表
        offsets = [
            struct.unpack("Q", f.read(8))[0]
            for _ in range(len(samples) + 1)
        ]
        data_start = f.tell()

        # ── 提取每个文件 ──
        ok = 0
        for i, sc in enumerate(samples):
            start = data_start + offsets[i]
            end = data_start + offsets[i + 1]
            size = end - start
            f.seek(start)

            filename = sc.get("filename", f"face_{i:05d}.jpg")
            person = sc.get("person_name", None)

            if person:
                out_dir = output_dir / person
            else:
                out_dir = output_dir
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / filename

            raw = f.read(size)
            out_path.write_bytes(raw)
            ok += 1

    print(f"[PAK] 完成: {ok} 个文件 → {output_dir}")
    return True


def main():
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        print(__doc__)
        return

    target = Path(sys.argv[1])
    if target.suffix.lower() == ".pak":
        pak_path = target
    else:
        pak_path = target / "faceset.pak"

    if not pak_path.exists():
        print(f"[!] 找不到 .pak 文件: {pak_path}")
        return

    output = Path(sys.argv[2]) if len(sys.argv) > 2 else None
    unpack_pak(pak_path, output)


if __name__ == "__main__":
    main()
