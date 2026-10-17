# -*- coding: utf-8 -*-
"""trainctl_demo —— mock 训练 WebUI：不跑真训练，演示/调试 trainctl 各命令

用法：
  python tools/trainctl_demo.py --port 6790        # 起 mock 服务（Ctrl+C 停）
  另开终端：
  python tools/trainctl.py --port 6790 status      # 看 status 输出效果
  python tools/trainctl.py --port 6790 loss
  python tools/trainctl.py --port 6790 options

模拟数据特征（真实感）：
  - 300 次迭代记录：src 0.42→0.21 指数收敛，最后 60 条进平台期（降不动）
  - dst 0.55→0.28 同理；随机噪声 ±0.004
  - 参数取自 Anna_SAEHD 真实形态（df-ud / 416 / wf / lr 3.1e-05）
"""
import argparse
import base64
import io
import json
import math
import random
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

random.seed(42)


def make_loss_history():
    h = []
    for i in range(300):
        if i < 240:
            src = 0.42 * math.exp(-i / 80) + 0.18 + random.uniform(-0.004, 0.004)
            dst = 0.55 * math.exp(-i / 100) + 0.24 + random.uniform(-0.005, 0.005)
        else:  # 平台期：在收敛值附近抖动
            src = 0.211 + random.uniform(-0.004, 0.004)
            dst = 0.283 + random.uniform(-0.005, 0.005)
        h.append([round(src, 4), round(dst, 4)])
    return h


LOSS = make_loss_history()
OPTIONS = {
    'model_name': 'Anna_SAEHD',
    'model_dir': r'F:\DFL-PyTorch\workspace\model',
    'resolution': 416, 'face_type': 'wf', 'archi': 'df-ud',
    'ae_dims': 400, 'e_dims': 80, 'd_dims': 80, 'd_mask_dims': 26,
    'lr': 3.1e-05, 'lr_dropout': True, 'lr_d': 'n', 'lr_cos': 0,
    'random_occlusion': True, 'random_noise': False,
    'random_color_power': 0.0, 'edge_enhance_power': 0.0,
    'eyes_mouth_prio': True, 'uniform_yaw': True,
    'random_src_flip': False, 'random_dst_flip': True,
    'use_bf16': True, 'gradient_checkpointing': True,
    'batch_size': 8, 'target_iter': 0,
    'gan_power': 0.0, 'face_style_power': 0.0, 'bg_style_power': 0.0,
}


# ---------- 假预览图：程序化生成 5 列 x 4 行的人脸风格网格 ----------
def make_fake_preview():
    """用 PIL 画一个 5x4 网格：第 1 列干净脸（渐变椭圆脸+五官），第 2 列掩码，
    第 3 列遮挡，第 4 列红色遮罩，第 5 列融合——模拟 SAEHD 预览布局。"""
    from PIL import Image, ImageDraw
    import random as _r
    _r.seed(7)
    cols, rows, cell = 5, 4, 160
    W_, H_ = cols * cell, rows * cell
    im = Image.new('RGB', (W_, H_), (24, 24, 24))
    d = ImageDraw.Draw(im)
    for r in range(rows):
        hue = _r.randint(200, 255)
        for c in range(cols):
            x0, y0 = c * cell, r * cell
            cx, cy = x0 + cell // 2, y0 + cell // 2
            if c == 1:  # mask
                d.ellipse((x0+30, y0+25, x0+cell-30, y0+cell-20), fill=(255, 220, 140))
            elif c == 2:  # occluded
                d.ellipse((x0+30, y0+25, x0+cell-30, y0+cell-20), fill=(hue, 200, 180))
                d.rectangle((x0, y0+cell//2, x0+cell, y0+cell//2+18), fill=(90, 60, 30))
            elif c == 3:  # red overlay
                d.ellipse((x0+30, y0+25, x0+cell-30, y0+cell-20), fill=(hue, 200, 180))
                d.ellipse((x0+30, y0+25, x0+cell-30, y0+cell-20), fill=(200, 60, 60))
            elif c == 4:  # merged
                d.ellipse((x0+30, y0+25, x0+cell-30, y0+cell-20), fill=(hue, 195, 175))
            else:         # clean face
                d.ellipse((x0+30, y0+25, x0+cell-30, y0+cell-20), fill=(hue, 205, 185), outline=(180,160,150))
            # 五官（第1/5列画）
            if c in (0, 4):
                d.ellipse((cx-26, cy-14, cx-8, cy+2), fill=(40, 40, 50))
                d.ellipse((cx+8, cy-14, cx+26, cy+2), fill=(40, 40, 50))
                d.ellipse((cx-8, cy+22, cx+8, cy+34), fill=(150, 90, 90))
                d.arc((cx-20, cy-2, cx+20, cy+24), 20, 160, fill=(120, 70, 70), width=2)
    buf = io.BytesIO()
    im.save(buf, format='JPEG', quality=90)
    return base64.b64encode(buf.getvalue()).decode()


FAKE_PREVIEW = None  # 懒生成


_STAGE = {'n': 0}  # 演进计数：每次请求 loss-history 推进一段（模拟训练在走）


def stage_history(total_reqs):
    """按已请求次数返回当前 loss 曲线：前段快速下降，中段慢降，末段平台。"""
    n = min(300 + total_reqs * 40, 900)
    h = []
    for i in range(n):
        if i < 200:   # 快降段：0.42 -> 0.215（与慢降段起点衔接）
            src = 0.42 * math.exp(-i / 60) + 0.20
            dst = 0.55 * math.exp(-i / 75) + 0.245
        elif i < n - 80:   # 慢降段：0.215 -> 0.2095 平滑过渡
            k = i - 200
            src = 0.2095 + 0.0055 * math.exp(-k / 150)
            dst = 0.2835 + 0.0075 * math.exp(-k / 150)
        else:              # 平台期
            src = 0.2095
            dst = 0.2835
        h.append([round(src + random.uniform(-0.004, 0.004), 4),
                  round(dst + random.uniform(-0.005, 0.005), 4)])
    return h


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _j(self, obj, code=200):
        b = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.send_header('Content-Length', str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def do_GET(self):
        from urllib.parse import urlparse, parse_qs
        p = self.path.split('?')[0]
        q = parse_qs(urlparse(self.path).query)
        if p == '/loss-history':
            if q.get('stage'):
                _STAGE['n'] = int(q['stage'][0])
                self._j(stage_history(_STAGE['n']))
            elif q.get('evolve'):
                _STAGE['n'] += 1
                self._j(stage_history(_STAGE['n']))
            else:
                self._j(LOSS)
        elif p == '/current-model-options':
            self._j(OPTIONS)
        elif p == '/current-settings':
            self._j({})
        elif p == '/request-preview':
            self._j({'ok': True})
        elif p == '/preview':
            global FAKE_PREVIEW
            if FAKE_PREVIEW is None:
                FAKE_PREVIEW = make_fake_preview()
            self._j({'previews': [{'name': '合并预览', 'data': FAKE_PREVIEW}],
                     'labels': {'n_cols': 5, 'n_samples': 4, 'src_fnames': ['a.png','b.png','c.png','d.png']}})
        else:
            self._j({'error': 'not found', 'hint': '这是 demo mock 服务'}, 404)

    def do_POST(self):
        cl = int(self.headers.get('Content-Length', 0))
        try:
            data = json.loads(self.rfile.read(cl)) if cl else {}
        except Exception:
            data = {}
        p = self.path.split('?')[0]
        if p == '/update-model-options':
            for k, v in data.items():
                if k != 'password':
                    OPTIONS[k] = v
            print(f'[demo] options updated: { {k: v for k, v in data.items() if k != "password"} }')
            self._j({'ok': True})
        else:
            self._j({'ok': True, 'demo': True})


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--port', type=int, default=6790)
    a = ap.parse_args()
    print(f'[trainctl-demo] mock WebUI on http://127.0.0.1:{a.port}（Ctrl+C 停止）')
    ThreadingHTTPServer(('127.0.0.1', a.port), H).serve_forever()
