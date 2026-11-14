"""严格对照 iperov 原版（克隆自 GitHub）的 TF2 推理脚本 — 512 liae-udt"""
import os, sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import tensorflow as tf
import numpy as np, cv2, pickle

def load(p):
    with open(p, 'rb') as f:
        return pickle.load(f)

def act(x, alpha=0.1):
    return tf.nn.leaky_relu(x, alpha)

def conv2d_sym(x, w, strides=1):
    """原版 Conv2D: 对称 pad ((k-1)*d+1)//2 + VALID conv"""
    p = ((w.shape[0] - 1) * 1 + 1) // 2
    x = tf.pad(x, [[0, 0], [p, p], [p, p], [0, 0]], mode='CONSTANT')
    return tf.nn.conv2d(x, w, strides, 'VALID')

def flatten(x):
    """原版 flatten: NHWC -> NCHW -> reshape（通道优先）"""
    x = tf.transpose(x, (0, 3, 1, 2))
    return tf.reshape(x, (-1, np.prod(x.shape[1:])))

def reshape_4D(x, w, h, c):
    """原版 reshape_4D: 展平按 (c,h,w) 解释 -> 转 NHWC"""
    x = tf.reshape(x, (-1, c, h, w))
    return tf.transpose(x, (0, 2, 3, 1))

def depth_to_space(x, size):
    """原版 depth_to_space (NHWC): reshape+transpose"""
    b, h, w, c = x.shape.as_list()
    oh, ow = h * size, w * size
    oc = c // (size * size)
    x = tf.reshape(x, (-1, h, w, size, size, oc))
    x = tf.transpose(x, (0, 1, 3, 2, 4, 5))
    return tf.reshape(x, (-1, oh, ow, oc))

def pixel_norm(x, axes):
    return x * tf.math.rsqrt(tf.reduce_mean(tf.square(x), axis=axes, keepdims=True) + 1e-06)

# ---------------- Encoder ('t' 分支) ----------------
def encoder(w, x):
    x = act(conv2d_sym(x, w['down1/conv1/weight:0'], 2))
    r = act(conv2d_sym(x, w['res1/conv1/weight:0']), 0.2)
    r = act(conv2d_sym(r, w['res1/conv2/weight:0']), 0.2)
    x = act(x + r, 0.2)
    x = act(conv2d_sym(x, w['down2/conv1/weight:0'], 2))
    x = act(conv2d_sym(x, w['down3/conv1/weight:0'], 2))
    x = act(conv2d_sym(x, w['down4/conv1/weight:0'], 2))
    x = act(conv2d_sym(x, w['down5/conv1/weight:0'], 2))
    r = act(conv2d_sym(x, w['res5/conv1/weight:0']), 0.2)
    r = act(conv2d_sym(r, w['res5/conv2/weight:0']), 0.2)
    x = act(x + r, 0.2)
    x = flatten(x)
    x = pixel_norm(x, axes=-1)
    return x

# ---------------- Inter ('t' 无 upscale) ----------------
def inter(w, x):
    x = tf.matmul(x, w['dense1/weight:0']) + w['dense1/bias:0']
    x = tf.matmul(x, w['dense2/weight:0']) + w['dense2/bias:0']
    return reshape_4D(x, 16, 16, 640)

# ---------------- Decoder ('t' 分支) ----------------
def decoder(w, z):
    def up(x, k):
        x = act(conv2d_sym(x, w[k + '/conv1/weight:0']))
        return depth_to_space(x, 2)
    def res(x, k):
        r = act(conv2d_sym(x, w[k + '/conv1/weight:0']), 0.2)
        r = act(conv2d_sym(r, w[k + '/conv2/weight:0']), 0.2)
        return act(x + r, 0.2)
    x = up(z, 'upscale0'); x = res(x, 'res0')
    x = up(x, 'upscale1'); x = res(x, 'res1')
    x = up(x, 'upscale2'); x = res(x, 'res2')
    x = up(x, 'upscale3'); x = res(x, 'res3')
    o = tf.concat([
        conv2d_sym(x, w['out_conv/weight:0']),
        conv2d_sym(x, w['out_conv1/weight:0']),
        conv2d_sym(x, w['out_conv2/weight:0']),
        conv2d_sym(x, w['out_conv3/weight:0'])], -1)
    o = depth_to_space(o, 2)
    return tf.nn.sigmoid(o)

def main():
    base = 'workspace/model/LiaeUDT512_SAEHD'
    enc_w = load(base + '_encoder.npy')
    ia_w = load(base + '_inter_AB.npy')
    dec_w = load(base + '_decoder.npy')

    img = cv2.imread('workspace/data_dst/aligned/0 (1)_0.jpg')
    x = cv2.resize(img, (512, 512)).astype(np.float32) / 127.5 - 1.0

    e = encoder(enc_w, x[None])
    ab = inter(ia_w, e)
    cat = tf.concat([ab, ab], -1)  # 原版 AE_merge: inter_AB 双份
    out = decoder(dec_w, cat).numpy()

    out_img = (out[0] * 255).clip(0, 255).astype(np.uint8)
    cv2.imwrite('workspace/tf_strict_512.png', out_img)
    g = cv2.cvtColor(out_img, cv2.COLOR_BGR2GRAY)
    print('严格原版 TF 512: 输出均值 %.0f 平滑度 %.3f' % (
        g.mean(), np.corrcoef(g[:, :-1].ravel(), g[:, 1:].ravel())[0, 1]))
    print('encoder out std=%.3f mean=%.4f' % (float(e.numpy().std()), float(e.numpy().mean())))
    print('inter out std=%.3f mean=%.3f max=%.1f' % (
        float(ab.numpy().std()), float(ab.numpy().mean()), float(ab.numpy().max())))

if __name__ == '__main__':
    main()
