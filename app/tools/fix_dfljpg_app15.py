# -*- coding: utf-8 -*-
"""把 DFLJPG 里「挂在 EOI 之前的那段 APP15」搬到头部 —— 引擎只读得到头部那一份。

背景（已实测，别再论证）：
  DFLJPG.load_raw 从头一段段走到 SOS，然后直接跳到 EOI 划出压缩数据段 ——
  **APP15 只有落在 SOS 之前的头部才读得到**。4.1 的 _align.rewrite_dfljpg_meta
  以前把 APP15 追加在 EOI 之前，于是整个 workspace/data_src/aligned（1203 张，
  实测 1203/1203 都是「头部 9 键无遮罩 + EOI 前 10 键带 xseg_mask」）的遮罩在
  引擎侧一直是死的：has_xseg_mask() 永远 False。

这个脚本只做一件事：把 EOI 前那份 legacy APP15 合并进头部（**已有的头部优先**），
再把 EOI 前那份摘掉。遮罩字节原样搬，不重新编码；JPEG 扫描数据一个字节都不动。

安全设计：
  * 默认 --dry-run：只在内存里算、不落盘，但照样报告「将会变成什么样」；
  * 落盘前先比对解码后的像素（cv2.imdecode 逐字节比），不一致就拒绝写；
  * --apply 默认先备份到 <扫描目录> 的同级 <名字>_app15_backup/<时间戳>/<相对路径>；
  * 幂等：搬完再跑一次全部落到 clean，不再产生任何写入。

用法：
    cd app
    python -m tools.fix_dfljpg_app15 ../workspace/data_src/aligned                  # dry-run
    python -m tools.fix_dfljpg_app15 ../workspace/data_src/aligned --limit 3        # 只扫 3 张
    python -m tools.fix_dfljpg_app15 ../workspace/data_src/aligned --apply --backup
    python -m tools.fix_dfljpg_app15 单张.jpg --apply --report out.json

引擎侧校验（DFLJPG.get_xseg_mask）要引擎那套 python（runtime/engine/python.exe），
server 环境缺 tqdm 之类，探针会自动降级成「未校验」而不是报错。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tempfile
import time
from pathlib import Path

APP_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = APP_DIR.parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from server._align import (           # noqa: E402  (先插 sys.path 才能 import)
    find_app15_header,
    find_app15_legacy,
    write_app15_head,
)

IMAGE_EXTS = (".jpg", ".jpeg")
SOI = b"\xff\xd8"
JPEG_EOI = b"\xff\xd9"


# ── 引擎侧探针：DFLJPG.get_xseg_mask ────────────────────────────────────
_ENGINE = {"tried": False, "mod": None, "err": None}


def engine_mod():
    """懒加载引擎的 DFLJPG；拿不到就返回 None（降级，不抛）。"""
    if not _ENGINE["tried"]:
        _ENGINE["tried"] = True
        try:
            eng = REPO_ROOT / "app" / "engine"
            if str(eng) not in sys.path:
                sys.path.insert(0, str(eng))
            from DFLIMG import DFLJPG
            _ENGINE["mod"] = DFLJPG
        except Exception as ex:              # 缺 tqdm/torch：降级
            _ENGINE["err"] = "%s: %s" % (type(ex).__name__, ex)
    return _ENGINE["mod"]


def _mask_of(mod, path_like):
    """引擎读一次 → (能不能 load, (shape, mean) | None)。

    两个 None 不一样，别混：
      (False, None) —— 引擎连这张图都 load 不了（load_raw 报 No Valid JPG info），
                      训练时整张被丢掉；
      (True,  None) —— 图能读，只是头部 APP15 里没有 xseg_mask。
    引擎自己的 DFLJPG.load 碰到坏文件会往 stderr 打一大段 traceback（io.log_err），
    批量扫的时候能把报告淹掉，这里收进 StringIO 里丢掉。
    """
    import contextlib
    import io as _io
    import numpy as np
    sink = _io.StringIO()
    with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
        d = mod.load(str(path_like))
    if d is None or not d.has_data():
        return (False, None)
    try:
        m = d.get_xseg_mask()
    except Exception:
        m = None
    if m is None:
        return (True, None)
    return (True, (tuple(m.shape), float(np.mean(m))))


def engine_mask(path):
    """磁盘上的文件 → (loadable, (shape, mean) | None)。走引擎自己的 DFLJPG.get_xseg_mask。"""
    mod = engine_mod()
    if mod is None:
        return None
    try:
        return _mask_of(mod, path)
    except Exception:
        return None


def engine_mask_bytes(data: bytes):
    """内存里的字节 → (loadable, (shape, mean) | None)。落盘前先问引擎「搬完能不能读到」。"""
    mod = engine_mod()
    if mod is None:
        return None
    tmp = Path(tempfile.gettempdir()) / ("dfljpg_probe_%d_%d.jpg"
                                         % (time.time_ns(), id(data) % 1000))
    try:
        tmp.write_bytes(data)
        return _mask_of(mod, tmp)
    except Exception:
        return None
    finally:
        try:
            if tmp.exists():
                tmp.unlink()
        except Exception:
            pass


# ── 单张规划（纯内存，不落盘） ─────────────────────────────────────────
def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def pixels_of(data: bytes):
    try:
        import cv2
        import numpy as np
        return cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    except Exception:
        return None


def same_pixels(a, b) -> bool:
    if a is None or b is None:
        return False
    try:
        import numpy as np
        return a.shape == b.shape and bool(np.array_equal(a, b))
    except Exception:
        return False


def plan_file(path: Path, do_engine: bool = True) -> dict:
    row = {
        "file": str(path),
        "name": path.name,
        "action": "",
        "detail": "",
        "bytes_before": 0,
        "bytes_after": 0,
        "sha_before": "",
        "sha_after": "",
        "head_keys": 0,
        "legacy_keys": 0,
        "mask_bytes": 0,
        "mask_conflict": False,
        "pixels_equal": None,
        "head_ok": None,
        "legacy_gone": None,
        "engine_before": None,
        "engine_after": None,
        "engine_repair": False,
        "_out": None,
    }
    try:
        data = path.read_bytes()
    except OSError as ex:
        row["action"], row["detail"] = "error", "读不了：%s" % ex
        return row

    row["bytes_before"] = len(data)
    row["sha_before"] = sha256(data)
    if not data.startswith(SOI):
        row["action"], row["detail"] = "skip", "不是 JPEG"
        return row

    head = find_app15_header(data)
    legacy = find_app15_legacy(data)
    row["head_keys"] = len(head or {})
    row["legacy_keys"] = len(legacy or {})

    if head is None and legacy is None:
        row["action"], row["detail"] = "skip", "没有 APP15（不是 DFLJPG）"
        return row

    if legacy is None:
        row["action"] = "clean"
        row["detail"] = "没有 EOI 前的 legacy APP15，无需搬"
        row["sha_after"], row["bytes_after"] = row["sha_before"], row["bytes_before"]
        if do_engine:
            row["engine_before"] = row["engine_after"] = engine_mask(path)
        return row

    merged = dict(legacy)                 # legacy 打底
    merged.update(head or {})             # 头部优先（任务定的规则）
    lm, hm = legacy.get("xseg_mask"), (head or {}).get("xseg_mask")
    if lm is not None and hm is not None and bytes(lm) != bytes(hm):
        row["mask_conflict"] = True       # 两份遮罩不一样：保留头部，旧的那份丢
    row["mask_bytes"] = len(merged.get("xseg_mask") or b"")

    try:
        out = write_app15_head(data, merged)
    except Exception as ex:
        row["action"], row["detail"] = "error", "重写失败：%s: %s" % (type(ex).__name__, ex)
        return row

    row["pixels_equal"] = same_pixels(pixels_of(data), pixels_of(out))
    new_head = find_app15_header(out)
    row["head_ok"] = (new_head is not None
                      and (new_head.get("xseg_mask") is not None
                           or merged.get("xseg_mask") is None))
    row["legacy_gone"] = find_app15_legacy(out) is None

    if not row["pixels_equal"]:
        row["action"], row["detail"] = "error", "像素校验失败（解码结果不一致），拒绝写"
        return row
    if not (row["head_ok"] and row["legacy_gone"]):
        row["action"] = "error"
        row["detail"] = "结构校验失败（head_ok=%s legacy_gone=%s）" % (
            row["head_ok"], row["legacy_gone"])
        return row

    row["action"] = "migrate"
    row["bytes_after"], row["sha_after"] = len(out), sha256(out)
    row["_out"] = out
    if do_engine:
        row["engine_before"] = engine_mask(path)
        row["engine_after"] = engine_mask_bytes(out)
        # 顺带修复：legacy APP15 的 pickle 里正好有 FF D9 时，DFLJPG.load_raw 从 SOS
        # 往后 find(FFD9) 会撞进这段假数据，把整张图判成 No Valid JPG info —— 这种
        # 文件引擎是完全看不见的（不是没遮罩，是整个被丢掉）。搬到头部之后那段字节
        # 跑到 SOS 之前，按段长解析，不再影响扫描数据，图就回来了。
        row["engine_repair"] = (row["engine_before"] is not None
                                and not row["engine_before"][0]
                                and row["engine_after"] is not None
                                and row["engine_after"][1] is not None)
    return row


# ── 目录扫描 / 落盘 ────────────────────────────────────────────────────
def collect(target: Path, limit: int = 0, underscore: bool = False):
    """扫 jpg。默认跳过名字以 _ 开头的子目录（_trash / _masks / 自己的备份目录）。"""
    if target.is_file():
        return [target]
    files = []
    for ext in IMAGE_EXTS:
        for p in target.rglob("*" + ext):
            if not underscore:
                rel_dirs = p.relative_to(target).parts[:-1]
                if any(d.startswith("_") for d in rel_dirs):
                    continue
            files.append(p)
    files = sorted(files)
    return files[:limit] if limit else files


def is_live_dataset(target: Path) -> bool:
    """是不是仓库 workspace 里的数据集（--apply 时要求先备份）。"""
    try:
        target.resolve().relative_to((REPO_ROOT / "workspace").resolve())
        return True
    except Exception:
        return False


def backup_dest(target: Path, f: Path, backup_root: Path) -> Path:
    base = target if target.is_dir() else target.parent
    try:
        rel = f.resolve().relative_to(base.resolve())
    except Exception:
        rel = Path(f.name)
    return backup_root / rel


def fmt_engine(v) -> str:
    """探针结果 → 一列短文本：- 未探 / 读不出 / 无遮罩 / 256x256x1 m=0.084。"""
    if v is None:
        return "-"
    loadable, mask = v
    if not loadable:
        return "读不出"
    if mask is None:
        return "无遮罩"
    shape, mean = mask
    return "%s m=%.3f" % ("x".join(str(s) for s in shape), mean)


def main() -> int:
    ap = argparse.ArgumentParser(
        description="把 DFLJPG 的 EOI 前 APP15 搬到头部（默认 dry-run，不落盘）")
    ap.add_argument("target", help="目录或单张 jpg")
    ap.add_argument("--apply", action="store_true", help="真的落盘（默认只 dry-run）")
    ap.add_argument("--dry-run", action="store_true", help="显式 dry-run（本来就是默认）")
    ap.add_argument("--backup", nargs="?", const="", default=None, metavar="DIR",
                    help="落盘前备份原文件；不给 DIR 就用 <扫描目录>_app15_backup/<时间戳>")
    ap.add_argument("--no-backup", action="store_true", help="不备份")
    ap.add_argument("--force", action="store_true",
                    help="workspace 真实数据集上不备份也允许写（慎用）")
    ap.add_argument("--limit", type=int, default=0, help="只处理前 N 张")
    ap.add_argument("--include-underscore", action="store_true",
                    help="连 _trash / _masks 这类下划线目录一起扫（默认跳过）")
    ap.add_argument("--no-engine", action="store_true", help="不调 DFLJPG.get_xseg_mask")
    ap.add_argument("--report", default="", help="每张图的结果写成 JSON")
    ap.add_argument("--quiet", action="store_true", help="只打汇总")
    args = ap.parse_args()

    target = Path(args.target).resolve()
    if not target.exists():
        print("目标不存在：%s" % target)
        return 2
    apply = bool(args.apply) and not args.dry_run
    do_engine = not args.no_engine

    files = collect(target, args.limit, args.include_underscore)
    if not files:
        print("没找到 jpg：%s" % target)
        return 2

    do_backup = (args.backup is not None) and not args.no_backup
    live = is_live_dataset(target)
    if apply and live and not do_backup and not args.force:
        print("拒绝执行：%s 在仓库 workspace 里（真实数据集）。" % target)
        print("  加 --backup 先备份，或显式 --force --no-backup。")
        return 3

    stamp = time.strftime("%Y%m%d-%H%M%S")
    backup_root = None
    if do_backup:
        if args.backup:
            backup_root = Path(args.backup).resolve() / stamp
        else:
            base = target if target.is_dir() else target.parent
            backup_root = base.parent / (base.name + "_app15_backup") / stamp

    print("扫描：%s" % target)
    print("模式：%s    文件：%d 张    引擎探针：%s"
          % ("APPLY" if apply else "DRY-RUN（不落盘）", len(files),
             "开" if do_engine else "关"))
    if do_backup:
        print("备份：只在真要搬（migrate）时写 → %s" % backup_root)
    if do_engine and engine_mod() is None:
        print("  引擎探针降级（%s）" % (_ENGINE["err"] or "?"))
        print("  换 runtime/engine/python.exe 跑才有 get_xseg_mask 那一列。")

    rows, counts = [], {}
    t0 = time.time()
    for i, f in enumerate(files, 1):
        row = plan_file(f, do_engine=do_engine)
        out = row.pop("_out", None)
        counts[row["action"]] = counts.get(row["action"], 0) + 1

        if apply and row["action"] == "migrate" and out is not None:
            if do_backup:
                dst = backup_dest(target, f, backup_root)
                dst.parent.mkdir(parents=True, exist_ok=True)
                dst.write_bytes(f.read_bytes())
            tmp = f.with_name(f.name + ".app15tmp")
            tmp.write_bytes(out)
            try:
                tmp.replace(f)                 # 同目录 rename：整文件原子替换
            finally:
                if tmp.exists():
                    tmp.unlink()
            row["apply_engine_after"] = engine_mask(f) if do_engine else None
        rows.append(row)

        if not args.quiet:
            if row["action"] == "clean":
                print("[%4d/%4d] %-26s %-7s 头部已带遮罩、没有 legacy 段，跳过  engine %s"
                      % (i, len(files), row["name"], row["action"],
                         fmt_engine(row["engine_before"])))
            elif row["action"] in ("migrate", "error"):
                print("[%4d/%4d] %-26s %-7s %d→%d B  keys %d+%d  mask=%d  px=%s  "
                      "head=%s  legacy_gone=%s  engine %s → %s%s"
                      % (i, len(files), row["name"], row["action"],
                         row["bytes_before"], row["bytes_after"],
                         row["head_keys"], row["legacy_keys"], row["mask_bytes"],
                         row["pixels_equal"], row["head_ok"], row["legacy_gone"],
                         fmt_engine(row["engine_before"]),
                         fmt_engine(row.get("apply_engine_after") or row["engine_after"]),
                         "  ← 修复：迁移前引擎整张读不出来" if row["engine_repair"] else ""))
            else:
                print("[%4d/%4d] %-26s %-7s %s"
                      % (i, len(files), row["name"], row["action"], row["detail"]))

    dt = time.time() - t0
    print("-" * 90)
    print("汇总：%s    用时 %.1fs" % (
        "  ".join("%s=%d" % (k, v) for k, v in sorted(counts.items())), dt))
    repairs = sum(1 for r in rows if r.get("engine_repair"))
    if repairs:
        print("顺带修复：%d 张原本引擎整张读不出来的图"
              "（legacy 段里正好有 FF D9，把扫描数据截断了）" % repairs)
    if not apply:
        print("dry-run：一个字节都没写。要真搬就加 --apply --backup。")
    elif do_backup:
        print("备份根：%s" % backup_root)

    if args.report:
        rp = Path(args.report)
        rp.parent.mkdir(parents=True, exist_ok=True)
        rp.write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
        print("报告：%s" % rp)
    return 0 if counts.get("error", 0) == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
