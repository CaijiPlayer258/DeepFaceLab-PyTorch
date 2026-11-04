# -*- coding: utf-8 -*-
"""假冒 iperov DFL 的 "Training preview" 窗口 —— 左：原图+特征点，右：原图+loss。

- 顶部：iperov 原文三行按键提示
- 100px loss 面板：iperov 的 get_loss_history_preview（每列 min~max 填条 + Iter 文本）
    · 颜色照 iperov 的 hsv(p/loss_count)，但原版没做 RGB→BGR，所以显示成 蓝 / 黄
    · 历史点 = 迭代数（400 多次就 400 多个点，曲线是一根细线，不是密色带）
- 预览：左右各 768px 正方形，左图带 68 点特征点，右图左下角写 loss

空格切换 SRC → DST（每按一次迭代 +1，曲线上多一个点）。
"""
from __future__ import annotations

import colorsys
import math
import random
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import cv2

ROOT = Path(__file__).resolve().parents[3]
IMG_DIR = ROOT.parent / "示例图"
SRC = [IMG_DIR / "0.5352_2218_女 (9948).jpg", IMG_DIR / "0.5410_1821_033_女 (5021).jpg"]
DST = [IMG_DIR / "0.5469_1720_024_女 (24935).jpg", IMG_DIR / "0.5469_1727_033_女 (6426).jpg"]

PANEL = 768                       # 单张预览边长（2 张并排 = 1536，2K 屏放得下）
W = PANEL * 2
HEAD_LINES = ["[s]:save [b]:backup [enter]:exit",
              "[p]:update [space]:next preview [l]:change history range"]
HEAD_LINE_H = 15
LH_H = 100
H = HEAD_LINE_H * 3 + LH_H + PANEL

FONT = "C:/Windows/Fonts/arial.ttf"
FONT_BIG = "C:/Windows/Fonts/consolab.ttf"

_LM = None


def text_strip(w, h, text, color=0.8):
    canvas = Image.new("RGB", (w, h), (0, 0, 0))
    d = ImageDraw.Draw(canvas)
    try:
        f = ImageFont.truetype(FONT, size=max(8, h - 2))
    except Exception:
        f = ImageFont.load_default()
    d.text((1, 0), text, font=f, fill=tuple(int(c * 255) for c in (color,) * 3))
    return np.asarray(canvas, dtype=np.float32) / 255.0


def loss_panel(history, iters, w):
    """iperov 的 ModelBase.get_loss_history_preview 原样实现。"""
    lh = np.ones((LH_H, w, 3), np.float32) * 0.1
    if history:
        arr = np.asarray(history, dtype=np.float32)
        n, ch = arr.shape
        # 纯粹一个点一个点地画（不连线）：每个迭代占 2px 宽
        for i in range(n):
            col0 = int(round(i * (w - 1) / max(1, n - 1)))
            for p_ in range(ch):
                col_rgb = colorsys.hsv_to_rgb(p_ * (1.0 / ch), 1.0, 1.0)
                col_rgb = (col_rgb[2], col_rgb[1], col_rgb[0])    # 原版缺的 RGB→BGR
                ph = int(np.clip(float(arr[i, p_]) / max(1e-6, ABS_MAX[0]) * (LH_H - 1),
                                 0, LH_H - 1))
                y = LH_H - ph - 1
                lh[y, col0] = col_rgb          # 1 个像素一个点：681 个点 / 1536 列 = 断断续续的点线
    lines = 5
    step = (LH_H - 1) / lines
    for i in range(lines + 1):
        lh[int(i * step), :] = 0.8
    t, b = int((lines - 1) * step), int(lines * step)
    lh[t:b, 0:w] += text_strip(w, b - t, "Iter: %d" % iters)
    return np.clip(lh, 0, 1)


ABS_MAX = [1.0]


def landmarks68(path: Path):
    """2D106 在这张对齐脸上取 68 点（点位置准，只画点 + 下颌线）。"""
    global _LM
    sys.path.insert(0, str(ROOT / "app"))
    try:
        from server._landmarks106 import InsightFace106
        if _LM is None:
            _LM = InsightFace106(str(ROOT / "app" / "modelhub" / "onnx" /
                                     "InsightFace2d106" / "InsightFace2D106.onnx"))
    except Exception as e:
        print("[lm] init fail:", e)
        return None
    img = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return None
    pts = _LM.landmarks68(img)
    if pts is None:
        return None
    s = PANEL / float(img.shape[1])
    return np.asarray(pts, dtype=np.float64) * s


def load_square(path: Path) -> Image.Image:
    im = Image.open(path).convert("RGB")
    s = min(im.size)
    ox, oy = (im.width - s) // 2, (im.height - s) // 2
    return im.crop((ox, oy, ox + s, oy + s)).resize((PANEL, PANEL), Image.LANCZOS)


def hold_vram(gb=2.0):
    try:
        import torch
        if not torch.cuda.is_available():
            print("[vram] CUDA 不可用")
            return None
        n = int(gb * (1024 ** 3) / 4)
        t = torch.empty(n, dtype=torch.float32, device="cuda")
        t.fill_(0.5)
        torch.cuda.synchronize()
        print("[vram] 占用 %.2f GB" % (torch.cuda.memory_allocated() / 1024 ** 3))
        return t
    except Exception as e:
        print("[vram] 失败:", e)
        return None


class Mock:
    def __init__(self):
        self.iters = 681
        self.state = 0                  # 0 = src, 1 = dst
        self.pair = 0
        # 一个迭代一个点：414 次就 414 个点，曲线是细线
        self.hist = []
        # 在 0.10~0.30 之间来回跳（带均值回归的随机游走），看不出下降趋势
        # 两条线起点明显不同；抖动幅度自己随机游走（不是分段开关）
        def series(start):
            out = []
            level = start
            amp = random.uniform(0.006, 0.02)
            for i in range(self.iters):
                # 幅度只是小幅漂移（0.005~0.028），不会出现"一段死平、一段炸开"
                amp = min(0.038, max(0.018, amp * math.exp(random.gauss(0, 0.08))))
                level = max(0.115, level - random.uniform(0.000008, 0.000055))   # 下移很慢，永远碰不到底
                v = level + random.gauss(0, amp)
                # 碰到边界做镜像折返，而不是硬夹住（硬夹会堆出一条平底）
                if v < 0.105:
                    v = 0.105 + (0.105 - v)
                elif v > 0.295:
                    v = 0.295 - (v - 0.295)
                out.append(min(0.30, max(0.10, v)))
            return out

        a_ = series(0.212)      # src 从高位起步
        b_ = series(0.176)      # dst 从低位起步
        self.hist = [[x, y] for x, y in zip(a_, b_)]
        ABS_MAX[0] = 0.34        # 标度略放宽，两条线落在面板中部而不是贴着顶
        self.loss = self.hist[-1][0]
        self._cache = {}

    def press(self):
        self.state = 1 - self.state
        if self.state == 0:
            self.pair = 1 - self.pair
        self.iters += 1
        self.loss = random.uniform(0.10, 0.30)
        self.hist.append([self.loss, random.uniform(0.10, 0.30)])

    def panels(self):
        key = (self.state, self.pair)
        if key not in self._cache:
            p = (SRC if self.state == 0 else DST)[self.pair]
            self._cache[key] = (load_square(p), landmarks68(p))
        return self._cache[key]

    def render_image(self):
        """返回 PIL 图（给 Tk 窗口用）。"""
        frame = self.render()
        return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    def render(self) -> np.ndarray:
        name = ["SAEHD src", "SAEHD dst"][self.state]
        head = np.ones((HEAD_LINE_H * 3, W, 3), np.float32) * 0.1
        lines = HEAD_LINES + ['Preview: "%s" [%d/2]' % (name, self.state + 1)]
        for i, line in enumerate(lines):
            t = i * HEAD_LINE_H
            head[t:t + HEAD_LINE_H, 0:W] += text_strip(W, HEAD_LINE_H, line)

        lh = loss_panel(self.hist[-self.iters:], self.iters, W)

        left_img, pts = self.panels()
        canvas = Image.new("RGB", (W, PANEL), (0, 0, 0))
        canvas.paste(left_img, (0, 0))
        canvas.paste(left_img, (PANEL, 0))          # 右边是原图（模型输出画面）

        d = ImageDraw.Draw(canvas)
        if pts is not None:
            for a, b in zip(range(0, 16), range(1, 17)):     # 下颌线
                d.line([(float(pts[a][0]), float(pts[a][1])),
                        (float(pts[b][0]), float(pts[b][1]))], fill=(0, 255, 0), width=2)
            for px, py in pts:
                d.ellipse((px - 2.5, py - 2.5, px + 2.5, py + 2.5), fill=(0, 255, 0))
        # 右图左下角 loss
        try:
            fbig = ImageFont.truetype(FONT_BIG, 34)
        except Exception:
            fbig = ImageFont.load_default()
        d.text((PANEL + 14, PANEL - 50), "%.4f" % self.loss, font=fbig, fill=(0, 255, 0))

        body = np.asarray(canvas, dtype=np.float32) / 255.0
        final = np.concatenate([head, lh, body], axis=0)
        return cv2.cvtColor((np.clip(final, 0, 1) * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)


def main():
    if "--no-vram" not in sys.argv:
        hold_vram(2.0)
    m = Mock()
    out = Path(__file__).resolve().parent
    if "--shot" in sys.argv:
        cv2.imwrite(str(out / "mock_preview_shot.png"), m.render())
        m.press()
        cv2.imwrite(str(out / "mock_preview_shot_dst.png"), m.render())
        print("saved:", out / "mock_preview_shot.png")
        return

    gui = "GUI:                           NONE" not in cv2.getBuildInformation()
    if gui:
        win = "Training preview"
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win, W, H)
        while True:
            cv2.imshow(win, m.render())
            k = cv2.waitKey(30) & 0xFF
            if k in (27, 13, ord("q")):
                break
            if k == 32:
                m.press()
        cv2.destroyAllWindows()
        return

    # 这套 Python 里的 cv2 是 headless 版（GUI: NONE），用 Tk 开一个原生窗口显示同样的画面
    import tkinter as tk
    from PIL import ImageTk

    root = tk.Tk()
    root.title("Training preview")
    root.configure(bg="black")
    root.resizable(False, False)
    label = tk.Label(root, bd=0, highlightthickness=0)
    label.pack()
    state = {"img": None}

    def repaint(_=None):
        pil = m.render_image()
        state["img"] = ImageTk.PhotoImage(pil)
        label.configure(image=state["img"])

    def on_key(e):
        if e.keysym in ("space",):
            m.press()
            repaint()
        elif e.keysym in ("Escape", "Return", "q"):
            root.destroy()

    root.bind("<Key>", on_key)
    repaint()
    root.focus_force()
    root.mainloop()


if __name__ == "__main__":
    main()
