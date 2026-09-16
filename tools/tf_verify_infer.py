"""TF 原版推理验证：用 iperov 原版 DFL 结构 + npy 权重直接前向（TF 原生语义）"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import tensorflow as tf
import numpy as np, pickle, cv2

def load_npy(p):
    with open(p, 'rb') as f:
        return pickle.load(f)

def act(x, a=0.1):
    return tf.nn.leaky_relu(x, a)

# ---------- encoder ('ud', 4x DownscaleBlock) ----------
def build_encoder(w, x):
    x = act(tf.nn.conv2d(x, w['down1/downs_0/conv1/weight:0'], 2, 'SAME') + w['down1/downs_0/conv1/bias:0'])
    x = act(tf.nn.conv2d(x, w['down1/downs_1/conv1/weight:0'], 2, 'SAME') + w['down1/downs_1/conv1/bias:0'])
    x = act(tf.nn.conv2d(x, w['down1/downs_2/conv1/weight:0'], 2, 'SAME') + w['down1/downs_2/conv1/bias:0'])
    x = act(tf.nn.conv2d(x, w['down1/downs_3/conv1/weight:0'], 2, 'SAME') + w['down1/downs_3/conv1/bias:0'])
    x = tf.reshape(x, (-1, 816 * 24 * 24))   # NHWC 展平（空间优先，TF 原生）
    x = x / tf.sqrt(tf.reduce_mean(x ** 2, -1, keepdims=True) + 1e-8)  # pixel_norm 'u'
    return x

# ---------- inter_AB / inter_B (liae, ae_out_ch=ae*2, 含 upscale1) ----------
def build_inter(w, x):
    x = tf.matmul(x, w['dense1/weight:0']) + w['dense1/bias:0']
    x = tf.matmul(x, w['dense2/weight:0']) + w['dense2/bias:0']
    x = tf.reshape(x, (-1, 12, 12, 704))     # NHWC reshape
    x = act(tf.nn.conv2d(x, w['upscale1/conv1/weight:0'], 1, 'SAME') + w['upscale1/conv1/bias:0'])
    x = tf.nn.depth_to_space(x, 2)           # TF 原生 d2s
    return x                                 # (B,24,24,704)

# ---------- decoder ('ud' + 'd') ----------
def build_decoder(w, z):
    x = act(tf.nn.conv2d(z, w['upscale0/conv1/weight:0'], 1, 'SAME') + w['upscale0/conv1/bias:0'])
    x = tf.nn.depth_to_space(x, 2)
    # res0
    r = act(tf.nn.conv2d(x, w['res0/conv1/weight:0'], 1, 'SAME') + w['res0/conv1/bias:0'], 0.2)
    r = act(tf.nn.conv2d(r, w['res0/conv2/weight:0'], 1, 'SAME') + w['res0/conv2/bias:0'], 0.2)
    x = act(x + r, 0.2)
    # upscale1
    x = act(tf.nn.conv2d(x, w['upscale1/conv1/weight:0'], 1, 'SAME') + w['upscale1/conv1/bias:0'])
    x = tf.nn.depth_to_space(x, 2)
    # res1
    r = act(tf.nn.conv2d(x, w['res1/conv1/weight:0'], 1, 'SAME') + w['res1/conv1/bias:0'], 0.2)
    r = act(tf.nn.conv2d(r, w['res1/conv2/weight:0'], 1, 'SAME') + w['res1/conv2/bias:0'], 0.2)
    x = act(x + r, 0.2)
    # upscale2
    x = act(tf.nn.conv2d(x, w['upscale2/conv1/weight:0'], 1, 'SAME') + w['upscale2/conv1/bias:0'])
    x = tf.nn.depth_to_space(x, 2)
    # res2
    r = act(tf.nn.conv2d(x, w['res2/conv1/weight:0'], 1, 'SAME') + w['res2/conv1/bias:0'], 0.2)
    r = act(tf.nn.conv2d(r, w['res2/conv2/weight:0'], 1, 'SAME') + w['res2/conv2/bias:0'], 0.2)
    x = act(x + r, 0.2)
    # out ('d': 4 conv + depth_to_space + sigmoid)
    o = tf.concat([
        tf.nn.conv2d(x, w['out_conv/weight:0'], 1, 'SAME') + w['out_conv/bias:0'],
        tf.nn.conv2d(x, w['out_conv1/weight:0'], 1, 'SAME') + w['out_conv1/bias:0'],
        tf.nn.conv2d(x, w['out_conv2/weight:0'], 1, 'SAME') + w['out_conv2/bias:0'],
        tf.nn.conv2d(x, w['out_conv3/weight:0'], 1, 'SAME') + w['out_conv3/bias:0'],
    ], -1)
    o = tf.nn.depth_to_space(o, 2)
    return tf.nn.sigmoid(o)                   # (B,384,384,3)

def main():
    enc_w = load_npy('workspace/model/384wf_SAEHD_encoder.npy')
    ia_w = load_npy('workspace/model/384wf_SAEHD_inter_AB.npy')
    ib_w = load_npy('workspace/model/384wf_SAEHD_inter_B.npy')
    dec_w = load_npy('workspace/model/384wf_SAEHD_decoder.npy')

    img = cv2.imread('workspace/data_dst/aligned/0 (1)_0.jpg')  # BGR
    x = cv2.resize(img, (384, 384)).astype(np.float32) / 127.5 - 1.0
    x = x[None]  # (1,384,384,3) BGR [-1,1]

    e = build_encoder(enc_w, x)
    ia = build_inter(ia_w, e)
    ib = build_inter(ib_w, e)
    cat = tf.concat([ib, ia], -1)             # inter_B 在前（liae）
    out = build_decoder(dec_w, cat)
    out_np = out[0].numpy()
    print('TF 输出: shape', out_np.shape, 'range', float(out_np.min()), '-', float(out_np.max()))
    out_bgr = (out_np * 255).clip(0, 255).astype(np.uint8)
    cv2.imwrite('workspace/tf_out.png', out_bgr)
    cv2.imwrite('workspace/tf_in.png', cv2.resize(img, (384, 384)))
    print('输出 BGR 均值: (%.0f, %.0f, %.0f)' % (out_bgr[:,:,0].mean(), out_bgr[:,:,1].mean(), out_bgr[:,:,2].mean()))
    print('输入 BGR 均值: (%.0f, %.0f, %.0f)' % (cv2.resize(img,(384,384))[:,:,0].mean(), cv2.resize(img,(384,384))[:,:,1].mean(), cv2.resize(img,(384,384))[:,:,2].mean()))
    print('已保存: workspace/tf_out.png')

if __name__ == '__main__':
    main()
