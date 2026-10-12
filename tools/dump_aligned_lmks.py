#!/usr/bin/env python3
"""dump_aligned_lmks.py — 从 aligned 脸集导出 dfm_test_merge 用的 npy landmarks。

用法: python tools\\dump_aligned_lmks.py <aligned目录> <输出npy目录>

规则：
- 只取每源图的第一张脸（*_0.*）
- npy 命名 = <source_filename去扩展名>_0.npy（dfm_test_merge 的查找约定）
- 内容 = get_source_landmarks()（帧坐标系 68x2 float32；异常时回退 get_landmarks()）
- 需要 source_filename 元数据（Extractor 图片模式 / 老工作流 extract 均有）

2026-10-13 建：测试集切脸更新管线用（Extractor 输出是序号命名，靠本工具映射回源图名）。
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from DFLIMG import DFLIMG


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(2)
    src = Path(sys.argv[1])
    dst = Path(sys.argv[2])
    dst.mkdir(parents=True, exist_ok=True)
    n = skip = 0
    for f in sorted(src.glob('*_0.*')):
        if f.suffix.lower() not in ('.png', '.jpg', '.jpeg', '.webp'):
            continue
        dfl = DFLIMG.load(f)
        if dfl is None or not dfl.has_data():
            print(f'[skip] {f.name}: 无元数据')
            skip += 1
            continue
        try:
            lm = dfl.get_source_landmarks()
        except Exception:
            lm = dfl.get_landmarks()
        if lm is None or len(lm) == 0:
            print(f'[skip] {f.name}: 无landmark')
            skip += 1
            continue
        src_name = dfl.get_source_filename()
        if not src_name:
            print(f'[skip] {f.name}: 无source_filename')
            skip += 1
            continue
        stem = Path(src_name).stem
        np.save(dst / f'{stem}_0.npy', np.asarray(lm, dtype=np.float32))
        n += 1
    print(f'[done] 导出 {n} 个 npy（skip {skip}）-> {dst}')
    sys.exit(0 if n > 0 else 1)


if __name__ == '__main__':
    main()
