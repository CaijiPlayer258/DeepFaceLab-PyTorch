# -*- coding: utf-8 -*-
"""trainctl —— 深变训练一行命令控制（agent 侧工具，零改动深变本体）

查询：
  trainctl status                       # 一屏状态：iter/loss趋势/模型/耗时/结论
  trainctl options                      # 当前全部模型参数
  trainctl preview --out p.jpg          # 拉取预览图（手机两列版，直接发 IM）
  trainctl preview --cols 2 --rows 4    # 自定义行列
  trainctl loss --hours 6 --chart l.jpg # loss 文字摘要 +（可选）matplotlib 图

控制：
  trainctl set lr=0.00005               # 热改参（可多个：lr=.. random_occlusion=y）
  trainctl set lr_dropout=y eyes_mouth_prio=n
  trainctl save                         # 请求保存（不退出）
  trainctl quit                         # 保存并结束训练（优雅退出）

通用：--port 6789 --password caiji --host 127.0.0.1
语义：set 直接 POST /update-model-options（架构类参数不可热更，训练日志会打印
     「跳过不可热更新参数」；lr/增强/损失权重类立即生效）。
"""
import argparse
import base64
import io
import json
import os
import sys
import time
import urllib.request
import urllib.error

try:
    import numpy as np
except ImportError:
    np = None

try:
    from PIL import Image
except ImportError:
    Image = None


def _die(msg, code=1):
    print(f'[trainctl] {msg}')
    sys.exit(code)


def _http(base, method, path, payload=None, timeout=15):
    url = f'{base}{path}'
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(url, data=data, method=method,
                                 headers={'Content-Type': 'application/json'})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            body = r.read()
            ctype = r.headers.get('Content-Type', '')
            if 'json' in ctype:
                return json.loads(body)
            return body
    except urllib.error.HTTPError as e:
        try:
            return json.loads(e.read())
        except Exception:
            _die(f'HTTP {e.code} @ {path}')
    except urllib.error.URLError as e:
        _die(f'连不上 {url}：{e.reason}\n训练 WebUI 没起来？先 Start-Process 启动训练（main.py train --webui-port <port>）')


def _ok(res):
    if isinstance(res, dict) and res.get('ok') is False:
        _die(f"服务端拒绝: {res.get('error', res)}")


def _fetch_loss(base):
    return _http(base, 'GET', '/loss-history')


def _loss_vec_format(v):
    return ' '.join(f'{x:.4f}' for x in v) if isinstance(v, (list, tuple)) else f'{v:.4f}'


def _trend_summary(history, hours=None, sides=('src', 'dst')):
    """history: list of [src_loss, dst_loss, ...]（Trainer 每 iter push，无 iter 字段）。
    返回 (文字摘要, 最近分量tuple)。"""
    if not history:
        return '无 loss 记录', None
    import statistics
    n = len(history)
    last = [float(x) for x in history[-1]]

    def seg_mean(a, b, idx):
        return statistics.mean(float(h[idx]) for h in history[a:b])

    out = []
    seg = 24 if n < 400 else 48
    for i, name in enumerate(sides):
        if i >= len(last):
            break
        cur = last[i]
        r = seg_mean(max(0, n - seg), n, i)
        p = seg_mean(max(0, n - 2 * seg), max(0, n - seg), i)
        delta = (r - p) / p * 100 if p else 0
        trend = '↓健康' if delta < -1 else ('↑注意' if delta > 3 else '→走平')
        out.append(f'{name} {cur:.4f}（近段均 {r:.4f} vs 前段 {p:.4f}，{delta:+.1f}% {trend}）')
    summ = f'记录 {n} 条 | ' + ' | '.join(out)
    return summ, tuple(last)
    span = min(n - 1, 200)
    seg = 8 if n < 80 else 24 if n < 400 else 48
    if n >= seg * 2:
        recent = [scalar(h) for h in history[-seg:]]
        prev = [scalar(h) for h in history[-2 * seg:-seg]]
        import statistics
        r, p = statistics.mean(recent), statistics.mean(prev)
        delta = (r - p) / p * 100 if p else 0
        trend = '下降(健康)' if delta < -0.5 else ('上升(注意)' if delta > 1 else '走平')
        summ = f'iter {last_it} | loss {last_v:.4f} | 近段均值 {r:.4f}（前段 {p:.4f}，{delta:+.1f}%）→ {trend}'
    else:
        summ = f'iter {last_it} | loss {last_v:.4f}（样本还少，暂判趋势）'
    return summ, last_it, last_v


def cmd_status(args):
    base = args.base
    loss = _fetch_loss(base)
    opts = _http(base, 'GET', '/current-model-options')
    summ, last = _trend_summary(loss)
    lines = ['[trainctl] 训练状态', f'  {summ}']
    if opts.get('model_name'):
        lines[0] += f"  [{opts['model_name']}]"
    if opts.get('lr') is not None:
        lines[0] += f"  lr={opts['lr']:.2e}"
    if opts:
        keys = ['model_name', 'iteration', 'lr', 'lr_dropout', 'random_occlusion',
                'random_noise', 'random_color_power', 'edge_enhance_power', 'face_type', 'resolution']
        kv = {k: opts[k] for k in keys if k in opts}
        if kv:
            lines.append('  参数: ' + ' '.join(f'{k}={v_}' for k, v_ in kv.items()))
        else:
            k0 = list(opts)[:6]
            lines.append('  参数(前6): ' + ' '.join(f'{k}={opts[k]}' for k in k0))
    # preview 有没有（不取走，只提示）
    print('\n'.join(lines))
    print("  提示: trainctl preview --out <file> 可拉两列手机预览图")


def cmd_options(args):
    opts = _http(args.base, 'GET', '/current-model-options')
    print(json.dumps(opts, ensure_ascii=False, indent=2))


def cmd_set(args):
    base = args.base
    payload = {'password': args.password}
    pairs = []
    for kv in args.kv:
        if '=' not in kv:
            _die(f'参数格式应为 k=v，收到 {kv}')
        k, v = kv.split('=', 1)
        v = v.strip()
        low = v.lower()
        if low in ('y', 'yes', 'true'):
            v = True
        elif low in ('n', 'no', 'false'):
            v = False
        elif low.replace('.', '', 1).replace('-', '', 1).isdigit():
            v = float(v) if '.' in v or 'e' in low else int(v)
        payload[k] = v
        pairs.append(f'{k}={v}')
    print(f'[trainctl] 下发热更新: {", ".join(pairs)}')
    res = _http(base, 'POST', '/update-model-options', payload)
    _ok(res)
    print('[trainctl] 已受理。立即生效于下一个 save 间隔；架构类参数（分辨率/ae_dims 等）不可热更，'
          '训练日志会出现「跳过不可热更新参数」。可用 trainctl options 复核。')


def cmd_save(args):
    res = _http(args.base, 'POST', '/request-save', {'password': args.password})
    _ok(res)
    print('[trainctl] 已请求保存（训练继续）')


def cmd_quit(args):
    res = _http(args.base, 'POST', '/request-quit', {'password': args.password})
    _ok(res)
    print('[trainctl] 已请求 保存+结束（优雅退出，等效点页面 Quit）')


def cmd_loss(args):
    base = args.base
    history = _fetch_loss(base)
    summ, last = _trend_summary(history)
    print(f'[trainctl] {summ}')
    if history:
        print(f'  最新分量: {_loss_vec_format(history[-1])}')
        print(f'  样本数: {len(history)}（/loss-history 最多保留 1000 条）')
    if args.chart:
        if np is None:
            _die('画图需要 numpy（深变自带 python 里有）')
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
        except ImportError:
            _die('未装 matplotlib，先 pip install matplotlib（装进深变 python）')
        v0 = history[-1]
        if isinstance(v0, (list, tuple)):
            names = ['src', 'dst'] + [f'L{i}' for i in range(2, len(v0))]
            for i in range(len(v0)):
                plt.plot([float(h[i]) for h in history], lw=2, label=names[i])
            plt.legend()
        else:
            plt.plot([float(h) for h in history], lw=2)
        plt.xlabel('iter'); plt.ylabel('loss'); plt.title('training loss')
        plt.tight_layout()
        plt.savefig(args.chart, dpi=150)
        print(f'[trainctl] 图已存 {args.chart}')


def cmd_preview(args):
    if Image is None:
        _die('需要 Pillow（深变自带 python 里有）')
    base = args.base
    _http(base, 'POST', '/request-preview')          # 先请求刷新
    deadline = time.time() + 300
    data = None
    printed = False
    while time.time() < deadline:
        time.sleep(5)
        res = _http(base, 'GET', '/preview')
        if res.get('previews'):
            data = res
            break
        if not printed:
            print('[trainctl] 等待下一个迭代边界生成预览（迭代长时需 1-3 分钟）...')
            printed = True
    if not data:
        _die('300 秒没等到新预览（训练可能在保存/繁忙），稍后再试')
    labels = data.get('labels') or {}
    outs = []
    for i, p in enumerate(data['previews']):
        raw = base64.b64decode(p['data'])
        img = Image.open(io.BytesIO(raw))
        w, h = img.size
        # 预览天然是 n_cols 列 × n_samples 行的小脸格（SAEHD 5 列）。
        # 手机两列版 = 按自然格切开后重排成 2 列大图。
        n_cols = int(labels.get('n_cols') or 5)
        n_rows = int(labels.get('n_samples') or 4)
        # 自然格尺寸：宽 w//n_cols；行高用总高均分（保护除零）
        cell_w = max(1, w // n_cols)
        cell_h = max(1, h // max(1, n_rows))
        cells = []
        for r in range(n_rows):
            for c in range(n_cols):
                box = (c * cell_w, r * cell_h,
                       min((c + 1) * cell_w, w), min((r + 1) * cell_h, h))
                cells.append(img.crop(box))
        # 目标：两列、单格宽 540（1080 手机宽），按格宽等比放大
        cols = 2
        target_w = 1080 // cols
        scale = target_w / cell_w
        target_h = max(1, int(cell_h * scale))
        rows_n = (len(cells) + cols - 1) // cols
        sheet = Image.new('RGB', (target_w * cols, rows_n * target_h), (16, 16, 16))
        for idx, cell in enumerate(cells):
            cell = cell.resize((target_w, target_h), Image.LANCZOS)
            r, c = divmod(idx, cols)
            sheet.paste(cell, (c * target_w, r * target_h))
        base_out = args.out or f'preview_{int(time.time())}'
        out = base_out if len(data['previews']) == 1 else base_out.replace('.jpg', '') + f'_{i}.jpg'
        if not out.lower().endswith('.jpg'):
            out += '.jpg'
        sheet.save(out, quality=88)
        outs.append((out, sheet.size, p['name']))
    for out, size, name in outs:
        print(f'[trainctl] 预览已存 {out} ({size[0]}x{size[1]}) [{name}]')
    print('[trainctl] 手机两列版：每格一张大脸，直接发 IM')


def main():
    ap = argparse.ArgumentParser(description='深变训练一行命令控制（查询/热改参/保存/退出/预览）')
    ap.add_argument('--host', default=os.environ.get('TRAINCTL_HOST', '127.0.0.1'))
    ap.add_argument('--port', type=int, default=int(os.environ.get('TRAINCTL_PORT', '6789')))
    ap.add_argument('--password', default=os.environ.get('TRAINCTL_PASSWORD', 'caiji'))
    sub = ap.add_subparsers(dest='cmd', required=True)

    s = sub.add_parser('status', help='一屏状态')
    s.set_defaults(func=cmd_status)
    s = sub.add_parser('options', help='当前模型参数(JSON)')
    s.set_defaults(func=cmd_options)

    s = sub.add_parser('set', help='热改参: set lr=0.00005 random_occlusion=y')
    s.add_argument('kv', nargs='+')
    s.set_defaults(func=cmd_set)

    s = sub.add_parser('save', help='请求保存(不退出)')
    s.set_defaults(func=cmd_save)
    s = sub.add_parser('quit', help='保存并结束训练')
    s.set_defaults(func=cmd_quit)

    s = sub.add_parser('loss', help='loss 摘要(可选画图)')
    s.add_argument('--hours', type=float, default=None)
    s.add_argument('--chart', default=None)
    s.set_defaults(func=cmd_loss)

    s = sub.add_parser('preview', help='拉取预览图(手机两列版)')
    s.add_argument('--out', default=None)
    s.add_argument('--cols', type=int, default=2)
    s.add_argument('--rows', type=int, default=2)
    s.set_defaults(func=cmd_preview)

    args = ap.parse_args()
    args.base = f'http://{args.host}:{args.port}'
    args.func(args)


if __name__ == '__main__':
    main()
