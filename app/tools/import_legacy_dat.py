#!/usr/bin/env python3
"""把 TF 时代（iperov DFL 1.x / 2.0）的 *_data.dat 迁移成 4.1 的新格式。

只搬参数（options / iter），不碰权重：TF 的 .npy 与 torch 的 .pth 张量布局不同，
只能走 tools/convert_dfl_tf_to_torch.py。默认只读，打印迁移报告；--apply 才写新文件
（必须给 --out，绝不覆盖原 dat）；再给 --convert 就顺手把权重转换跑完。

顺序很重要：转换脚本是**按 dat 里的 options 建网络骨架**的，2020 年以前的 dat
（archi=dfhd/liaech、face_type 全名）它直接 ValueError: Unsupported SAEHD archi_type。
所以这里先把 migrated dat 落盘，再在同一目录里跑转换。

用法：
  # 1) 只看报告（不动任何文件）
  python tools/import_legacy_dat.py --dat "D:/models/Old_SAEHD_data.dat"

  # 2) 只写新格式 dat（原文件保持不变）
  python tools/import_legacy_dat.py --dat ".../Old_SAEHD_data.dat" --apply \
      --out ".../Old_migrated/Old_SAEHD_data.dat"

  # 3) 一步做完：迁 dat + 复制 TF 权重 + 转成 .pth
  python tools/import_legacy_dat.py --dat ".../Old/Old_SAEHD_data.dat" --apply \
      --out ".../New/Old_SAEHD_data.dat" --convert

退出码：0 成功 / 2 不是老格式 / 3 参数不对 / 4 转换脚本失败 / 5 读不了 dat
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

_HERE = Path(__file__).resolve()
_APP = _HERE.parents[1]                  # <repo>/app
_ENGINE = _APP / 'engine'                # <repo>/app/engine（core/ 在这里）
for _p in (str(_ENGINE), str(_APP)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from core.legacy_dat import (            # noqa: E402
    build_new_dat, format_report, load_legacy_dat, model_class_from_dat_name,
    model_py_for, report_counts, write_dat,
)

# 被后端/管道调用时统一 UTF-8（后端 Popen 用的是 encoding='utf-8'）；
# 交互式控制台保持原样，免得在 GBK 窗口里反过来花屏。
try:
    if not sys.stdout.isatty():
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

TOOL_DIR = _APP / 'tools'
CONVERTER = TOOL_DIR / 'convert_dfl_tf_to_torch.py'


def _dat_stem(dat_path: Path) -> str:
    """Old_SAEHD_data.dat → Old_SAEHD（转换脚本的 --name 要的就是这个）。"""
    name = Path(dat_path).name
    return name[:-len('_data.dat')] if name.endswith('_data.dat') else Path(dat_path).stem


def _scan_weights(folder: Path, stem: str):
    """(torch 的 .pth, TF 的 .npy)。.ckpt 不在其中：转换脚本只认 DFL 的 .npy 存档。"""
    try:
        pth = sorted(Path(folder).glob(stem + '_*.pth'))
        npy = sorted(Path(folder).glob(stem + '_*.npy'))
    except Exception:
        return [], []
    return pth, npy


def _convert_cmd(python_exe, src_dir: Path, dst_dir: Path, model_class: str, stem: str):
    return [str(python_exe), str(CONVERTER),
            '--src', str(src_dir), '--dst', str(dst_dir),
            '--model', str(model_class), '--name', str(stem), '--strict']


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="迁移 TF 时代（DFL 1.x/2.0）的 *_data.dat：只取 options，不碰权重。",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="默认只读；--apply 才写新 dat（需要 --out，且不会覆盖原文件）。")
    ap.add_argument('--dat', type=Path, required=True,
                    help='老 dat 路径，例如 workspace/model/Old_SAEHD_data.dat')
    ap.add_argument('--out', type=Path, default=None,
                    help='迁移后的新 dat 路径（--apply 必填，不允许等于 --dat）')
    ap.add_argument('--apply', action='store_true', help='把迁移结果写成新格式 dat（默认只打印报告）')
    ap.add_argument('--model', type=str, default=None,
                    help='模型类名（SAEHD/AMP/XSeg…）；默认从文件名 <name>_<Class>_data.dat 推')
    ap.add_argument('--weights-dir', type=Path, default=None,
                    help='TF 权重目录（含 <name>_*.npy）；默认用 --dat 所在目录')
    ap.add_argument('--convert', action='store_true',
                    help='顺手调用 convert_dfl_tf_to_torch.py 转权重（需要 --apply/--out）')
    ap.add_argument('--python', type=Path, default=Path(sys.executable),
                    help='调用转换脚本用的解释器（默认当前解释器）')
    ap.add_argument('--json', action='store_true', help='输出机器可读 JSON（给界面用）')
    ap.add_argument('--quiet', action='store_true', help='不打印逐条报告，只给汇总')
    args = ap.parse_args(argv)

    def say(*a):
        if not args.json:
            print(*a)

    dat_path = args.dat
    if not dat_path.is_file():
        print('找不到 dat：%s' % dat_path, file=sys.stderr)
        return 5

    model_class = args.model or model_class_from_dat_name(dat_path)
    model_py = model_py_for(dat_path, model_class)

    try:
        options, meta, report = load_legacy_dat(dat_path, model_py=model_py, model_class=model_class)
    except Exception as e:
        print('读不了 %s：%s' % (dat_path, e), file=sys.stderr)
        return 5

    counts = report_counts(report)
    reasons = meta.get('legacy_reasons') or []
    stem = _dat_stem(dat_path)
    weights_dir = (args.weights_dir or dat_path.parent).resolve()
    out_dir = (args.out.parent if args.out else dat_path.parent).resolve()
    cmd = (_convert_cmd(args.python, out_dir, out_dir, model_class, stem)
           if model_class else None)
    pth, npy = _scan_weights(weights_dir, stem)

    payload = {
        'dat': str(dat_path),
        'model_class': model_class,
        'model_py': str(model_py) if model_py else None,
        'iter': meta.get('iter'),
        'options': {k: v for k, v in options.items()
                    if isinstance(v, (int, float, str, bool)) or v is None},
        'report': report,
        'counts': counts,
        'is_legacy': bool(reasons),
        'legacy_reasons': reasons,
        'weights': {'pth': [str(p) for p in pth], 'npy': [str(p) for p in npy]},
        'convert_command': cmd,
    }

    def finish(code, applied=None, converted=None):
        payload['applied'] = applied
        payload['converted'] = converted
        if args.json:
            print(json.dumps(payload, ensure_ascii=False, indent=2))
        return code

    if not reasons:
        say('这份 dat 已经是 4.1 的格式（没有 TF 时代的键），兼容层不介入：%s' % dat_path)
        return finish(2)

    if not args.quiet:
        say('=== TF 时代 dat 迁移报告 ===')
        say('文件    : %s（%.1f KB）' % (dat_path, dat_path.stat().st_size / 1024.0))
        say('迭代    : iter = %s，loss_history = %d 条'
            % (meta.get('iter'), len(meta.get('loss_history') or [])))
        say('模型类  : %s（默认值取自 %s）'
            % (model_class or '认不出', model_py.name if model_py else '（没找到 Model_pytorch.py）'))
        say('判定    : %s' % '；'.join(reasons))
        say('-' * 72)
        say(format_report(report, indent=''))
        say('-' * 72)

    say('汇总    : 迁移 %d · 改键 %d · 忽略 %d · 补默认 %d · 无法识别 %d'
        % (counts['迁移'], counts['改键'], counts['忽略'], counts['补默认'], counts['无法识别']))
    if pth:
        say('权重    : %s 里已有 %d 个 .pth（torch 权重）' % (weights_dir.name, len(pth)))
    elif npy:
        say('权重    : %s 里有 %d 个 TF 权重（.npy），还没转' % (weights_dir.name, len(npy)))
    else:
        say('权重    : %s 里没有任何权重文件 —— 这份 dat 只是参数模板，训练会从零开始'
            % weights_dir.name)
    if cmd:
        say('转权重  : ' + subprocess.list2cmdline(cmd))
    else:
        say('转权重  : 认不出模型类（文件名不像 <name>_<Class>_data.dat），用 --model 指定后重跑')

    if not args.apply:
        say('未写入任何文件（默认只读）。要写成新格式：--apply --out <新 dat 路径>')
        return finish(0)

    # ---------------- 写盘 ----------------
    if args.out is None:
        print('--apply 需要 --out：不会替你决定新文件名，更不会覆盖原 dat。', file=sys.stderr)
        return finish(3)
    out_path = args.out
    if out_path.resolve() == dat_path.resolve():
        print('--out 和 --dat 是同一个文件；本工具不覆盖原始 dat。', file=sys.stderr)
        return finish(3)
    if not out_path.parent.is_dir():
        print('目标目录不存在：%s —— 本工具不会替你建模型目录，先自己 mkdir。' % out_path.parent,
              file=sys.stderr)
        return finish(3)

    if _dat_stem(out_path) != stem:
        msg = ('--out 的文件名要跟原 dat 一致（%s_data.dat）：引擎靠文件名认模型类，'
               '转换脚本也靠 --name 找权重。' % stem)
        if args.convert:
            print(msg, file=sys.stderr)
            return finish(3)
        say('提醒    : ' + msg)

    converted = None
    if args.convert:
        if not model_class:
            print('--convert 需要模型类名：用 --model 指定。', file=sys.stderr)
            return finish(3)
        if out_path.parent.resolve() != weights_dir:
            copied = []
            for f in npy:
                try:
                    shutil.copy2(f, out_path.parent / f.name)
                    copied.append(f.name)
                except Exception as e:
                    print('复制 %s 失败：%s' % (f, e), file=sys.stderr)
                    return finish(3)
            if copied:
                say('已复制  : %d 个 TF 权重（.npy）→ %s（源目录不动）'
                    % (len(copied), out_path.parent))
        # 先落 migrated dat：转换脚本是拿它读 options 的
        try:
            write_dat(out_path, build_new_dat(options, meta))
        except Exception as e:
            print('写 %s 失败：%s' % (out_path, e), file=sys.stderr)
            return finish(5)
        say('已写入  : %s（新格式）' % out_path)
        cmd2 = _convert_cmd(args.python, out_path.parent, out_path.parent, model_class, stem)
        say('调用转换脚本：' + subprocess.list2cmdline(cmd2))
        rc = subprocess.call(cmd2, cwd=str(_APP))
        if rc != 0:
            print('转换脚本返回 %d：migrated dat 已写好，权重没转成，先看上面的转换报告。' % rc,
                  file=sys.stderr)
            return finish(4, applied=str(out_path))
        converted = str(out_path.parent)
        say('转换完成: 权重已按 migrated dat 的架构写进 %s' % out_path.parent)
        return finish(0, applied=str(out_path), converted=converted)

    try:
        write_dat(out_path, build_new_dat(options, meta))
    except Exception as e:
        print('写 %s 失败：%s' % (out_path, e), file=sys.stderr)
        return finish(5)
    say('已写入  : %s（新格式；原 dat 未改）' % out_path)
    if cmd:
        say('下一步  : 想接着以前的训练先转权重 → ' + subprocess.list2cmdline(cmd))
    else:
        say('下一步  : 这份 dat 只有参数，没有权重；训练会从零开始。')
    return finish(0, applied=str(out_path))


if __name__ == '__main__':
    raise SystemExit(main())
