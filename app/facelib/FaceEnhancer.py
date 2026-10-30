import operator
from pathlib import Path

import cv2
import numpy as np

from core.leras import nn


def _to_tensor_nchw(img_bgr: np.ndarray):
    """把 HWC numpy 转成 NCHW torch Tensor。

    说明：本仓库的 leras(Pytorch) 默认使用 NCHW。
    旧版 DFL(TF) 的 FaceEnhancer 是 NHWC + tf.*，在 Torch 版会直接崩溃。
    """
    t = nn.torch
    if img_bgr.ndim != 3 or img_bgr.shape[2] != 3:
        raise ValueError(f"期望输入为 HWC 且 3 通道，实际 {img_bgr.shape}")

    x = t.from_numpy(np.ascontiguousarray(img_bgr))
    # HWC -> NCHW
    x = x.permute(2, 0, 1).unsqueeze(0).contiguous()
    return x


def _to_numpy_hwc(x):
    """把 NCHW torch Tensor 转回 HWC numpy。"""
    t = nn.torch
    if not isinstance(x, t.Tensor):
        raise TypeError("x 必须是 torch.Tensor")
    if x.ndim != 4:
        raise ValueError(f"期望 NCHW 4D tensor，实际 ndim={x.ndim}")
    x = x.detach().cpu()
    x = x[0].permute(1, 2, 0).contiguous().numpy()
    return x

class FaceEnhancer(object):
    """
    x4 face enhancer
    """
    def __init__(self, place_model_on_cpu=False, run_on_cpu=False):
        # Torch 版：使用 PyTorch 后端，不再依赖 TensorFlow 的 nn.tf。
        # 为了与仓库其他网络保持一致，这里强制使用 NCHW。
        # 有些子进程/独立脚本场景下，Devices 可能尚未初始化。
        # 原版报错提示也指出需要 initialize_main_env()。
        nn.initialize_main_env()

        # 如果用户明确要求 CPU，就用 CPU-only DeviceConfig，避免探测 GPU。
        if run_on_cpu or place_model_on_cpu:
            nn.initialize(device_config=nn.DeviceConfig([]), data_format="NCHW")
        else:
            nn.initialize(data_format="NCHW")

        import torch
        import torch.nn.functional as F

        class FaceEnhancerNet(nn.ModelBase):
            def __init__(self, name='FaceEnhancer'):
                super().__init__(name=name)

            def on_build(self):
                self.conv1 = nn.Conv2D (3, 64, kernel_size=3, strides=1, padding='SAME')

                self.dense1 = nn.Dense (1, 64, use_bias=False)
                self.dense2 = nn.Dense (1, 64, use_bias=False)

                self.e0_conv0 = nn.Conv2D (64, 64, kernel_size=3, strides=1, padding='SAME')
                self.e0_conv1 = nn.Conv2D (64, 64, kernel_size=3, strides=1, padding='SAME')

                self.e1_conv0 = nn.Conv2D (64, 112, kernel_size=3, strides=1, padding='SAME')
                self.e1_conv1 = nn.Conv2D (112, 112, kernel_size=3, strides=1, padding='SAME')

                self.e2_conv0 = nn.Conv2D (112, 192, kernel_size=3, strides=1, padding='SAME')
                self.e2_conv1 = nn.Conv2D (192, 192, kernel_size=3, strides=1, padding='SAME')

                self.e3_conv0 = nn.Conv2D (192, 336, kernel_size=3, strides=1, padding='SAME')
                self.e3_conv1 = nn.Conv2D (336, 336, kernel_size=3, strides=1, padding='SAME')

                self.e4_conv0 = nn.Conv2D (336, 512, kernel_size=3, strides=1, padding='SAME')
                self.e4_conv1 = nn.Conv2D (512, 512, kernel_size=3, strides=1, padding='SAME')

                self.center_conv0 = nn.Conv2D (512, 512, kernel_size=3, strides=1, padding='SAME')
                self.center_conv1 = nn.Conv2D (512, 512, kernel_size=3, strides=1, padding='SAME')
                self.center_conv2 = nn.Conv2D (512, 512, kernel_size=3, strides=1, padding='SAME')
                self.center_conv3 = nn.Conv2D (512, 512, kernel_size=3, strides=1, padding='SAME')

                self.d4_conv0 = nn.Conv2D (1024, 512, kernel_size=3, strides=1, padding='SAME')
                self.d4_conv1 = nn.Conv2D (512, 512, kernel_size=3, strides=1, padding='SAME')

                self.d3_conv0 = nn.Conv2D (848, 512, kernel_size=3, strides=1, padding='SAME')
                self.d3_conv1 = nn.Conv2D (512, 512, kernel_size=3, strides=1, padding='SAME')

                self.d2_conv0 = nn.Conv2D (704, 288, kernel_size=3, strides=1, padding='SAME')
                self.d2_conv1 = nn.Conv2D (288, 288, kernel_size=3, strides=1, padding='SAME')

                self.d1_conv0 = nn.Conv2D (400, 160, kernel_size=3, strides=1, padding='SAME')
                self.d1_conv1 = nn.Conv2D (160, 160, kernel_size=3, strides=1, padding='SAME')

                self.d0_conv0 = nn.Conv2D (224, 96, kernel_size=3, strides=1, padding='SAME')
                self.d0_conv1 = nn.Conv2D (96, 96, kernel_size=3, strides=1, padding='SAME')

                self.out1x_conv0 = nn.Conv2D (96, 48, kernel_size=3, strides=1, padding='SAME')
                self.out1x_conv1 = nn.Conv2D (48, 3, kernel_size=3, strides=1, padding='SAME')

                self.dec2x_conv0 = nn.Conv2D (96, 96, kernel_size=3, strides=1, padding='SAME')
                self.dec2x_conv1 = nn.Conv2D (96, 96, kernel_size=3, strides=1, padding='SAME')

                self.out2x_conv0 = nn.Conv2D (96, 48, kernel_size=3, strides=1, padding='SAME')
                self.out2x_conv1 = nn.Conv2D (48, 3, kernel_size=3, strides=1, padding='SAME')

                self.dec4x_conv0 = nn.Conv2D (96, 72, kernel_size=3, strides=1, padding='SAME')
                self.dec4x_conv1 = nn.Conv2D (72, 72, kernel_size=3, strides=1, padding='SAME')

                self.out4x_conv0 = nn.Conv2D (72, 36, kernel_size=3, strides=1, padding='SAME')
                self.out4x_conv1 = nn.Conv2D (36, 3 , kernel_size=3, strides=1, padding='SAME')

            def forward(self, bgr, param, param1):
                # bgr: NCHW, param/param1: (N,1)
                x = self.conv1(bgr)

                a = self.dense1(param).view(-1, 64, 1, 1)
                b = self.dense2(param1).view(-1, 64, 1, 1)

                x = F.leaky_relu(x + a + b, 0.1)

                x = F.leaky_relu(self.e0_conv0(x), 0.1)
                x = e0 = F.leaky_relu(self.e0_conv1(x), 0.1)

                x = F.avg_pool2d(x, kernel_size=2, stride=2)
                x = F.leaky_relu(self.e1_conv0(x), 0.1)
                x = e1 = F.leaky_relu(self.e1_conv1(x), 0.1)

                x = F.avg_pool2d(x, kernel_size=2, stride=2)
                x = F.leaky_relu(self.e2_conv0(x), 0.1)
                x = e2 = F.leaky_relu(self.e2_conv1(x), 0.1)

                x = F.avg_pool2d(x, kernel_size=2, stride=2)
                x = F.leaky_relu(self.e3_conv0(x), 0.1)
                x = e3 = F.leaky_relu(self.e3_conv1(x), 0.1)

                x = F.avg_pool2d(x, kernel_size=2, stride=2)
                x = F.leaky_relu(self.e4_conv0(x), 0.1)
                x = e4 = F.leaky_relu(self.e4_conv1(x), 0.1)

                x = F.avg_pool2d(x, kernel_size=2, stride=2)
                x = F.leaky_relu(self.center_conv0(x), 0.1)
                x = F.leaky_relu(self.center_conv1(x), 0.1)
                x = F.leaky_relu(self.center_conv2(x), 0.1)
                x = F.leaky_relu(self.center_conv3(x), 0.1)

                x = torch.cat([nn.resize2d_bilinear(x), e4], dim=1)
                x = F.leaky_relu(self.d4_conv0(x), 0.1)
                x = F.leaky_relu(self.d4_conv1(x), 0.1)

                x = torch.cat([nn.resize2d_bilinear(x), e3], dim=1)
                x = F.leaky_relu(self.d3_conv0(x), 0.1)
                x = F.leaky_relu(self.d3_conv1(x), 0.1)

                x = torch.cat([nn.resize2d_bilinear(x), e2], dim=1)
                x = F.leaky_relu(self.d2_conv0(x), 0.1)
                x = F.leaky_relu(self.d2_conv1(x), 0.1)

                x = torch.cat([nn.resize2d_bilinear(x), e1], dim=1)
                x = F.leaky_relu(self.d1_conv0(x), 0.1)
                x = F.leaky_relu(self.d1_conv1(x), 0.1)

                x = torch.cat([nn.resize2d_bilinear(x), e0], dim=1)
                x = F.leaky_relu(self.d0_conv0(x), 0.1)
                x = d0 = F.leaky_relu(self.d0_conv1(x), 0.1)

                x = F.leaky_relu(self.out1x_conv0(x), 0.1)
                x = self.out1x_conv1(x)
                out1x = bgr + torch.tanh(x)

                x = d0
                x = F.leaky_relu(self.dec2x_conv0(x), 0.1)
                x = F.leaky_relu(self.dec2x_conv1(x), 0.1)
                x = d2x = nn.resize2d_bilinear(x)

                x = F.leaky_relu(self.out2x_conv0(x), 0.1)
                x = self.out2x_conv1(x)
                out2x = nn.resize2d_bilinear(out1x) + torch.tanh(x)

                x = d2x
                x = F.leaky_relu(self.dec4x_conv0(x), 0.1)
                x = F.leaky_relu(self.dec4x_conv1(x), 0.1)
                x = nn.resize2d_bilinear(x)

                x = F.leaky_relu(self.out4x_conv0(x), 0.1)
                x = self.out4x_conv1(x)
                out4x = nn.resize2d_bilinear(out2x) + torch.tanh(x)

                return out4x

        model_path = Path(__file__).parent / "FaceEnhancer.npy"
        if not model_path.exists():
            raise Exception("Unable to load FaceEnhancer.npy")

        # 设备策略：Torch 版没有 TF 那种“CPU 放权重、GPU 运行”的 session 机制。
        # 这里按最简单/可预测的方式处理：run_on_cpu 或 place_model_on_cpu 都强制 CPU。
        if run_on_cpu or place_model_on_cpu:
            self._device = torch.device('cpu')
        else:
            self._device = getattr(nn, 'device', torch.device('cpu'))

        self.model = FaceEnhancerNet()
        self.model.build()
        self.model.to(self._device)
        ok = self.model.load_weights(model_path)
        if not ok:
            raise Exception("FaceEnhancer.npy 加载失败（权重文件格式或结构不匹配）")
        self.model.eval()

    def enhance (self, inp_img, is_tanh=False, preserve_size=True):
        if not is_tanh:
            inp_img = np.clip( inp_img * 2 -1, -1, 1 )

        param = np.array([0.2])
        param1 = np.array([1.0])
        up_res = 4
        patch_size = 192
        patch_size_half = patch_size // 2

        ih,iw,ic = inp_img.shape
        h,w,c = ih,iw,ic

        th,tw = h*up_res, w*up_res

        t_padding = 0
        b_padding = 0
        l_padding = 0
        r_padding = 0

        if h < patch_size:
            t_padding = (patch_size-h)//2
            b_padding = (patch_size-h) - t_padding

        if w < patch_size:
            l_padding = (patch_size-w)//2
            r_padding = (patch_size-w) - l_padding

        if t_padding != 0:
            inp_img = np.concatenate ([ np.zeros ( (t_padding,w,c), dtype=np.float32 ), inp_img ], axis=0 )
            h,w,c = inp_img.shape

        if b_padding != 0:
            inp_img = np.concatenate ([ inp_img, np.zeros ( (b_padding,w,c), dtype=np.float32 ) ], axis=0 )
            h,w,c = inp_img.shape

        if l_padding != 0:
            inp_img = np.concatenate ([ np.zeros ( (h,l_padding,c), dtype=np.float32 ), inp_img ], axis=1 )
            h,w,c = inp_img.shape

        if r_padding != 0:
            inp_img = np.concatenate ([ inp_img, np.zeros ( (h,r_padding,c), dtype=np.float32 ) ], axis=1 )
            h,w,c = inp_img.shape


        i_max = w-patch_size+1
        j_max = h-patch_size+1

        final_img = np.zeros ( (h*up_res,w*up_res,c), dtype=np.float32 )
        final_img_div = np.zeros ( (h*up_res,w*up_res,1), dtype=np.float32 )

        x = np.concatenate ( [ np.linspace (0,1,patch_size_half*up_res), np.linspace (1,0,patch_size_half*up_res) ] )
        x,y = np.meshgrid(x,x)
        patch_mask = (x*y)[...,None]

        j=0
        while j < j_max:
            i = 0
            while i < i_max:
                patch_img = inp_img[j:j+patch_size, i:i+patch_size,:]
                # PyTorch 推理（NCHW），输出再转回 HWC
                import torch
                with torch.no_grad():
                    bgr_t = _to_tensor_nchw(patch_img).to(device=self._device, dtype=nn.floatx)
                    p0 = torch.tensor(param.reshape(1, 1), device=self._device, dtype=nn.floatx)
                    p1 = torch.tensor(param1.reshape(1, 1), device=self._device, dtype=nn.floatx)
                    out_t = self.model(bgr_t, p0, p1)
                x = _to_numpy_hwc(out_t)

                final_img    [j*up_res:(j+patch_size)*up_res, i*up_res:(i+patch_size)*up_res,:] += x*patch_mask
                final_img_div[j*up_res:(j+patch_size)*up_res, i*up_res:(i+patch_size)*up_res,:] += patch_mask
                if i == i_max-1:
                    break
                i = min( i+patch_size_half, i_max-1)
            if j == j_max-1:
                break
            j = min( j+patch_size_half, j_max-1)

        final_img_div[final_img_div==0] = 1.0
        final_img /= final_img_div

        if t_padding+b_padding+l_padding+r_padding != 0:
            final_img = final_img [t_padding*up_res:(h-b_padding)*up_res, l_padding*up_res:(w-r_padding)*up_res,:]

        if preserve_size:
            final_img = cv2.resize (final_img, (iw,ih), interpolation=cv2.INTER_LANCZOS4)

        if not is_tanh:
            final_img = np.clip( final_img/2+0.5, 0, 1 )

        return final_img


"""

    def enhance (self, inp_img, is_tanh=False, preserve_size=True):
        if not is_tanh:
            inp_img = np.clip( inp_img * 2 -1, -1, 1 )

        param = np.array([0.2])
        param1 = np.array([1.0])
        up_res = 4
        patch_size = 192
        patch_size_half = patch_size // 2

        h,w,c = inp_img.shape

        th,tw = h*up_res, w*up_res

        preupscale_rate = 1.0

        if h < patch_size or w < patch_size:
            preupscale_rate = 1.0 / ( max(h,w) / patch_size )

        if preupscale_rate != 1.0:
            inp_img = cv2.resize (inp_img, ( int(w*preupscale_rate), int(h*preupscale_rate) ), interpolation=cv2.INTER_LANCZOS4)
            h,w,c = inp_img.shape

        i_max = w-patch_size+1
        j_max = h-patch_size+1

        final_img = np.zeros ( (h*up_res,w*up_res,c), dtype=np.float32 )
        final_img_div = np.zeros ( (h*up_res,w*up_res,1), dtype=np.float32 )

        x = np.concatenate ( [ np.linspace (0,1,patch_size_half*up_res), np.linspace (1,0,patch_size_half*up_res) ] )
        x,y = np.meshgrid(x,x)
        patch_mask = (x*y)[...,None]

        j=0
        while j < j_max:
            i = 0
            while i < i_max:
                patch_img = inp_img[j:j+patch_size, i:i+patch_size,:]
                x = self.model.run( [ patch_img[None,...], [param], [param1] ] )[0]
                final_img    [j*up_res:(j+patch_size)*up_res, i*up_res:(i+patch_size)*up_res,:] += x*patch_mask
                final_img_div[j*up_res:(j+patch_size)*up_res, i*up_res:(i+patch_size)*up_res,:] += patch_mask
                if i == i_max-1:
                    break
                i = min( i+patch_size_half, i_max-1)
            if j == j_max-1:
                break
            j = min( j+patch_size_half, j_max-1)

        final_img_div[final_img_div==0] = 1.0
        final_img /= final_img_div

        if preserve_size:
            final_img = cv2.resize (final_img, (w,h), interpolation=cv2.INTER_LANCZOS4)
        else:
            if preupscale_rate != 1.0:
                final_img = cv2.resize (final_img, (tw,th), interpolation=cv2.INTER_LANCZOS4)

        if not is_tanh:
            final_img = np.clip( final_img/2+0.5, 0, 1 )

        return final_img
"""