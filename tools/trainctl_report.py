# -*- coding: utf-8 -*-
"""trainctl_report —— 训练图文报告（一张手机宽图：结论段 + loss 折线 + 两列预览）

设计（对着 IM/手机看）：
  1080 宽一张图，从上到下：标题 → 文字结论段（loss 报告 + 开关型参数）→
  dashboard 风格 loss 折线（src/dst 双线，标注末值与趋势）→ 两列大脸预览。
参数显示原则（用户指定 2026-10-11）：只显示「开关型/值得看的」——
  开着的增强与训练开关、lr 类、batch；不显示分辨率/face_type/archi 等
  固定架构参数，也不显示值为 0/False/None 的没开参数。

用法：
  真训练： python tools/trainctl_report.py --port 6789 --out report.jpg
  演示：   python tools/trainctl_demo.py --port 6790  另开终端
           python tools/trainctl_report.py --port 6790 --out demo_report.jpg
依赖： matplotlib + Pillow（深变自带 python 均有）
"""
import argparse
import base64
import io
import json
import os
import statistics
import time
import urllib.request

W = 1080

# ---------- 参数显示白名单：开关型 + 值得看的（参考 new-params-guide） ----------
SWITCH_KEYS = [
    ('lr', 'lr', '{:.2e}'),
    ('lr_dropout', 'lr_dropout', None),
    ('lr_cos', 'lr_cos', None),
    ('random_occlusion', '随机遮挡', None),
    ('random_noise', '随机噪点', None),
    ('random_color_power', '随机偏色', '{:.2f}'),
    ('edge_enhance_power', '边缘强化', '{:.1f}'),
    ('eyes_mouth_prio', '眼嘴优先', None),
    ('uniform_yaw', '均匀yaw', None),
    ('random_src_flip', 'src翻转', None),
    ('random_dst_flip', 'dst翻转', None),
    ('random_warp', '随机扭曲', None),
    ('masked_training', '遮罩训练', None),
    ('gan_power', 'GAN', '{:.2f}'),
    ('face_style_power', '脸风格', '{:.2f}'),
    ('bg_style_power', '背景风格', '{:.2f}'),
    ('vgg_perceptual_power', 'VGG感知', '{:.1f}'),
    ('use_bf16', 'BF16', None),
    ('pretrain', '预训练', None),
]


def _fmt(v, fmt):
    if fmt:
        try:
            return fmt.format(v)
        except Exception:
            return str(v)
    if isinstance(v, bool):
        return '开' if v else '关'
    return str(v)


def pick_switches(opts):
    """只留 开着的/非零 的开关型参数。"""
    out = []
    for key, label, fmt in SWITCH_KEYS:
        v = opts.get(key)
        if v is None:
            continue
        if v is False or v == 0 or v == 'n' or v == '' or v == 'none':
            continue  # 没开的不显示
        out.append(f'{label}={_fmt(v, fmt)}')
    return out


def _http(base, method, path, payload=None, timeout=15):
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(f'{base}{path}', data=data, method=method,
                                 headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


# ---------- loss 分析 ----------
def analyze(history):
    n = len(history)
    if n < 4:
        return None
    seg = min(48, n // 4)

    def m(a, b, i):
        return statistics.mean(float(h[i]) for h in history[a:b])

    r = {}
    for i, name in enumerate(('src', 'dst')):
        if i >= len(history[-1]):
            break
        cur = float(history[-1][i])
        near, prev = m(n - seg, n, i), m(max(0, n - 2 * seg), max(0, n - seg), i)
        delta = (near - prev) / prev * 100 if prev else 0
        trend = '下降' if delta < -1 else ('上升' if delta > 3 else '走平')
        r[name] = dict(cur=cur, near=near, delta=delta, trend=trend)
    return r


def verdict(a, opts):
    """结论段：一段人话。"""
    if not a:
        return 'loss 记录不足，暂无法判断。'
    s, d = a.get('src'), a.get('dst')
    lines = []
    both_flat = s and d and s['trend'] == '走平' and d['trend'] == '走平'
    any_up = (s and s['trend'] == '上升') or (d and d['trend'] == '上升')
    if any_up:
        lines.append('⚠ loss 在上升：检查是否 lr 过大/崩溃前兆，建议 preview 目检。')
    elif both_flat:
        lines.append('结论：src/dst 均已走平（平台期）。可选：降 lr 再压、收工导出、或继续挂。')
    else:
        lines.append('结论：训练正常收敛中，继续挂即可。')
    if s:
        lines.append(f"src {s['cur']:.4f}（近段 {s['delta']:+.1f}% {s['trend']}）")
    if d:
        lines.append(f"dst {d['cur']:.4f}（近段 {d['delta']:+.1f}% {d['trend']}）")
    return '\n'.join(lines)


# ---------- 绘制 ----------
def draw_loss_chart(history, title='loss（近 1000 记录）'):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    fig, ax = plt.subplots(figsize=(W / 100, 3.2), dpi=100)
    names = ['src', 'dst'] + [f'L{i}' for i in range(2, len(history[-1]))]
    xs = np.arange(len(history))
    for i in range(len(history[-1])):
        ys = np.array([float(h[i]) for h in history])
        # 轻度平滑（窗口9 均值），和 dashboard 观感一致
        if len(ys) > 12:
            k = 9
            ys_s = np.convolve(ys, np.ones(k) / k, mode='valid')
            xs_s = xs[k - 1:]
        else:
            ys_s, xs_s = ys, xs
        ax.plot(xs_s, ys_s, lw=2.2, label=f"{names[i]} {float(history[-1][i]):.4f}")
    ax.legend(loc='upper right', fontsize=11, framealpha=0.5)
    ax.set_title(title, fontsize=13, loc='left')
    ax.grid(alpha=0.25)
    ax.margins(x=0)
    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format='png', facecolor='white')
    plt.close(fig)
    buf.seek(0)
    from PIL import Image
    return Image.open(buf).convert('RGB')


def fetch_preview_b64(base, wait_sec=300):
    try:
        _http(base, 'GET', '/request-preview')
    except Exception:
        pass
    deadline = time.time() + wait_sec
    while time.time() < deadline:
        time.sleep(4)
        r = _http(base, 'GET', '/preview')
        if r.get('previews'):
            return r
    return None


def build_preview_grid(pv_data, labels, cols=2, max_rows=4):
    """预览重排成两列大脸；只取第一路预览（合并预览最有信息量，在最后）。"""
    from PIL import Image
    imgs = []
    for p in pv_data['previews']:
        img = Image.open(io.BytesIO(base64.b64decode(p['data'])))
        imgs.append((p['name'], img))
    if not imgs:
        return None, ''
    name, img = imgs[-1]  # 最后一路通常是「合并预览」
    w, h = img.size
    n_cols = int((labels or {}).get('n_cols') or 5)
    n_rows = int((labels or {}).get('n_samples') or 4)
    n_rows = min(n_rows, max_rows)
    cw, ch = max(1, w // n_cols), max(1, h // max(1, int((labels or {}).get('n_samples') or 4)))
    cells = []
    for r in range(n_rows):
        for c in range(n_cols):
            cells.append(img.crop((c * cw, r * ch, min((c + 1) * cw, w), min((r + 1) * ch, h))))
    tw = W // cols
    th = int(ch * (tw / cw))
    rows_n = (len(cells) + cols - 1) // cols
    sheet = Image.new('RGB', (tw * cols, rows_n * th), (16, 16, 16))
    for idx, cell in enumerate(cells):
        sheet.paste(cell.resize((tw, th), Image.LANCZOS), ((idx % cols) * tw, (idx // cols) * tw * 0 + (idx // cols) * th))
    return sheet, name


def render_text_block(title, body_lines, switches, w=W):
    from PIL import Image, ImageDraw, ImageFont

    def font(sz, bold=False):
        p = r'C:\Windows\Fonts\msyhbd.ttc' if bold else r'C:\Windows\Fonts\msyh.ttc'
        try:
            return ImageFont.truetype(p, sz)
        except Exception:
            return ImageFont.truetype(r'C:\Windows\Fonts\msyh.ttc', sz)

    f_title, f_body, f_tag = font(34, True), font(26), font(24)
    pad = 28
    tmp = ImageDraw.Draw(Image.new('RGB', (8, 8)))
    # 预量高度
    h = pad * 2 + 46
    for ln in body_lines:
        h += 36
    h += 16
    if switches:
        h += 44 + 34 + 32 * ((len(switches) + 1) // 2)
    im = Image.new('RGB', (w, h), (250, 250, 250))
    d = ImageDraw.Draw(im)
    y = pad
    d.text((pad, y), title, font=f_title, fill=(20, 20, 20))
    y += 46
    for ln in body_lines:
        color = (180, 40, 40) if ln.startswith('⚠') else (60, 60, 60)
        d.text((pad, y), ln, font=f_body, fill=color)
        y += 36
    if switches:
        y += 10
        d.text((pad, y), '启用中的开关', font=f_tag, fill=(120, 120, 120))
        y += 34
        col_w = (w - pad * 2 - 24) // 2   # 两列，中间留 24 间距
        half = (len(switches) + 1) // 2
        left, right = switches[:half], switches[half:]
        for i in range(half):
            d.text((pad, y), left[i], font=f_tag, fill=(40, 70, 120))
            if i < len(right):
                d.text((pad + col_w + 24, y), right[i], font=f_tag, fill=(40, 70, 120))
            y += 32
    return im


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--host', default=os.environ.get('TRAINCTL_HOST', '127.0.0.1'))
    ap.add_argument('--port', type=int, default=int(os.environ.get('TRAINCTL_PORT', '6789')))
    ap.add_argument('--password', default=os.environ.get('TRAINCTL_PASSWORD', 'caiji'))
    ap.add_argument('--out', default=None, help='输出 jpg（默认 report_<时间戳>.jpg）')
    ap.add_argument('--no-preview', action='store_true', help='不要预览（纯 loss 报告）')
    ap.add_argument('--preview-wait', type=int, default=300)
    args = ap.parse_args()
    base = f'http://{args.host}:{args.port}'

    from PIL import Image
    history = _http(base, 'GET', '/loss-history')
    opts = _http(base, 'GET', '/current-model-options')
    a = analyze(history)
    body = verdict(a, opts).split('\n')
    body.append(f"记录 {len(history)} 条 | 模型 {opts.get('model_name', '?')} | lr {opts.get('lr', 0):.2e}")
    switches = pick_switches(opts)

    ts = time.strftime('%m-%d %H:%M')
    stem = args.out or f'report_{time.strftime("%Y%m%d_%H%M")}'
    if stem.lower().endswith('.jpg'):
        stem = stem[:-4]
    outs = {}

    # ① 文字段：纯文本直接输出（IM 原生文本，参数两列表格）
    print()
    print(f'📊 训练报告 {ts} · {opts.get("model_name", "?")}')
    for ln in body:
        print(ln)
    if switches:
        print()
        print('启用中的开关：')
        half = (len(switches) + 1) // 2
        left, right = switches[:half], switches[half:]
        lw = max(len(s) for s in left) + 2
        for i in range(half):
            r = right[i] if i < len(right) else ''
            print(f'  {left[i]:<{lw}}{r}')

    # ② loss 折线（图 1/2）
    if history:
        outs['loss'] = stem + '_loss.png'
        draw_loss_chart(history).save(outs['loss'])

    # ③ 预览（图 2/2）
    if not args.no_preview:
        print('[report] 请求预览（等待迭代边界，最长 {}s）...'.format(args.preview_wait))
        pv = fetch_preview_b64(base, args.preview_wait)
        if pv:
            grid, name = build_preview_grid(pv, pv.get('labels'))
            if grid:
                outs['preview'] = stem + '_preview.jpg'
                grid.save(outs['preview'], quality=88)
        else:
            print('[report] 预览未及到达（训练繁忙/迭代长），跳过预览')

    if outs:
        print('[report] 图片已生成:')
        for k, v in outs.items():
            print(f'  {k}: {v}')
    return outs


if __name__ == '__main__':
    main()
