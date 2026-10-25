"""数据集体检的运行器 —— 在 **4.0 的 python** 里跑。

放在 4.0 的 python 里跑：那边带着完整的 numpy/cv2/numba，4.1 这一侧只负责
起子进程、转进度。流程：解码 + 指标（cv2，多线程）→ 一次决策。

协议（和 _merge_runner 一致，4.1 逐行解析）：
    @@P <done> <total> [stage]   进度
    @@R <json>                   结果
    @@ERR <msg>                  失败

用法：
    engine_python _scan_runner.py --dir <目录> --level strict
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# 4.1 的 app/server 目录（_oneshot.py 在这里）
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import _oneshot  # noqa: E402


def _say(line: str) -> None:
    sys.stdout.write(line + "\n")
    sys.stdout.flush()



def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--level", default="strict")
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--score-pct", type=float, default=5.0, help="模型轨：规则幸存者里再砍百分之几")
    ap.add_argument("--bucket-floor", type=int, default=0, help="每个姿态桶至少保留几张（0=关闭）")
    args = ap.parse_args(argv)

    root = Path(args.dir)
    if not root.is_dir():
        _say("@@ERR 目录不存在: %s" % root)
        return 2

    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    files = sorted([p for p in root.iterdir() if p.is_file() and p.suffix.lower() in exts])
    total = len(files)

    def prog(done, tot):
        _say("@@P %d %d scan" % (done, tot))

    rows = _oneshot.scan(files, workers=args.workers, progress=prog)

    preset = _oneshot.LEVELS.get(args.level, _oneshot.LEVELS["strict"])
    plan = _oneshot.plan(rows, blur_pct=preset["blur_pct"],
                         dup_hamming=preset["dup_hamming"],
                         dup_mae=preset["dup_mae"],
                         score_pct=args.score_pct, bucket_floor=args.bucket_floor)
    plan["level"] = args.level
    plan["rows"] = [{k: v for k, v in r.items() if k != "thumb"} for r in rows[:500]]
    _say("@@R " + json.dumps(plan, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
