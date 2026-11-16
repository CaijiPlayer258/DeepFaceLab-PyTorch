
# -*- coding: utf-8 -*-
"""三臂成片的逐像素对拍：x264 为参照，另给「离原始合成帧多远」，并抠对比图看肉眼差别。"""
import subprocess, json, os
from pathlib import Path
import numpy as np, cv2, imageio_ffmpeg

FF = imageio_ffmpeg.get_ffmpeg_exe()
D = Path(__file__).resolve().parents[4] / "_enc4k_bench"   # 仓库根的 scratch 素材/产物目录（不进仓）
TAGS = {"4k": (2160, 3840, 80, "f_4k.raw"), "hd": (1080, 1920, 80, "f_hd.raw")}
ARMS = ["A", "B1", "B2", "C1", "C2"]


def dec(path, w, h):
    p = subprocess.Popen([FF, "-v", "error", "-i", str(path), "-map", "0:v:0",
                          "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                         stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    fr = w * h * 3
    while True:
        b = p.stdout.read(fr)
        if len(b) < fr: break
        yield np.frombuffer(b, np.uint8).reshape(h, w, 3)
    p.wait()


def ssim(x, y):
    g1 = cv2.cvtColor(x, cv2.COLOR_RGB2GRAY).astype(np.float32)
    g2 = cv2.cvtColor(y, cv2.COLOR_RGB2GRAY).astype(np.float32)
    C1, C2 = (0.01*255)**2, (0.03*255)**2
    k = (11, 11); s = 1.5
    m1 = cv2.GaussianBlur(g1, k, s); m2 = cv2.GaussianBlur(g2, k, s)
    s11 = cv2.GaussianBlur(g1*g1, k, s) - m1*m1
    s22 = cv2.GaussianBlur(g2*g2, k, s) - m2*m2
    s12 = cv2.GaussianBlur(g1*g2, k, s) - m1*m2
    v = ((2*m1*m2+C1)*(2*s12+C2))/((m1*m1+m2*m2+C1)*(s11+s22+C2))
    return float(v.mean())


def crop_meta(w, h):
    # 以 1080x1920 为基准的坐标，按比例放大
    sx, sy = w/1080.0, h/1920.0
    face = (int(590*sx), int(600*sy), int(910*sx), int(1000*sy))
    grad = (int(100*sx), int(200*sy), int(420*sx), int(400*sy))
    return face, grad


out = {}
for tag, (w, h, n, fn) in TAGS.items():
    ref = [np.ascontiguousarray(f) for f in dec(D/("out_%s_r1_A.mp4"%tag), w, h)]
    src = np.memmap(D/fn, dtype=np.uint8, mode="r", shape=(n, h, w, 3)).copy()  # 源=bgr24
    face, grad = crop_meta(w, h)
    out[tag] = {}
    for a in ARMS:
        p = D/("out_%s_r1_%s.mp4"%(tag, a))
        if not p.exists(): continue
        diffs=[]; ps=[]; pss=[]; ssis=[]; gt8=[]; mx=[]
        for i, f in enumerate(dec(p, w, h)):
            r = ref[i] if i < len(ref) else ref[-1]
            d = np.abs(f.astype(np.int16) - r.astype(np.int16))
            diffs.append(d.mean()); mx.append(d.max()); gt8.append((d > 8).mean()*100)
            mse = ((f.astype(np.float64)-r)**2).mean()
            ps.append(99.0 if mse == 0 else 10*np.log10(255.0**2/mse))
            s = src[i][:, :, ::-1]
            mse2 = ((f.astype(np.float64)-s)**2).mean()
            pss.append(10*np.log10(255.0**2/mse2))
            ssis.append(ssim(f, r))
        out[tag][a] = dict(mean_diff=round(float(np.mean(diffs)),3), max=round(float(np.max(mx)),1),
                           p99=round(float(np.percentile(diffs,99)),3),
                           gt8_pct=round(float(np.mean(gt8)),2),
                           psnr_vs_x264=round(float(np.mean(ps)),2),
                           psnr_vs_src=round(float(np.mean(pss)),2),
                           ssim_vs_x264=round(float(np.mean(ssis)),5))
        # 抠图（第 40 帧）
        for i, f in enumerate(dec(p, w, h)):
            if i != 39: continue
            cv2.imwrite(str(D/("crop_%s_%s_face.png"%(tag,a))), cv2.cvtColor(f[face[1]:face[3], face[0]:face[2]], cv2.COLOR_RGB2BGR))
            cv2.imwrite(str(D/("crop_%s_%s_grad.png"%(tag,a))), cv2.cvtColor(f[grad[1]:grad[3], grad[0]:grad[2]], cv2.COLOR_RGB2BGR))
            break
    del ref, src
    print(tag, json.dumps(out[tag], ensure_ascii=False, indent=1))
json.dump(out, open(D/"quality.json","w"), indent=1, ensure_ascii=False)
