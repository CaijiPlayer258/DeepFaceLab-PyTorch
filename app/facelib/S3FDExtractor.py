import operator
from pathlib import Path

import cv2
import numpy as np
import torch

from core.leras import nn


class _L2Norm(torch.nn.Module):
    def __init__(self, n_channels: int):
        super().__init__()
        self.n_channels = int(n_channels)
        self.weight = torch.nn.Parameter(torch.ones(self.n_channels, dtype=torch.float32))

    def forward(self, x):
        # x: (N,C,H,W)
        denom = torch.sqrt(torch.sum(x * x, dim=1, keepdim=True) + 1e-10)
        x = x / denom
        return x * self.weight.view(1, -1, 1, 1)


class _S3FD(torch.nn.Module):
    def __init__(self):
        super().__init__()
        tnn = torch.nn
        self.minus = torch.tensor([104.0, 117.0, 123.0], dtype=torch.float32).view(1, 3, 1, 1)

        def conv(in_ch, out_ch, k=3, s=1, p=1):
            return tnn.Conv2d(in_ch, out_ch, kernel_size=k, stride=s, padding=p, bias=True)

        self.conv1_1 = conv(3, 64)
        self.conv1_2 = conv(64, 64)

        self.conv2_1 = conv(64, 128)
        self.conv2_2 = conv(128, 128)

        self.conv3_1 = conv(128, 256)
        self.conv3_2 = conv(256, 256)
        self.conv3_3 = conv(256, 256)

        self.conv4_1 = conv(256, 512)
        self.conv4_2 = conv(512, 512)
        self.conv4_3 = conv(512, 512)

        self.conv5_1 = conv(512, 512)
        self.conv5_2 = conv(512, 512)
        self.conv5_3 = conv(512, 512)

        # Note: original DFL uses padding=3 here.
        self.fc6 = conv(512, 1024, k=3, s=1, p=3)
        self.fc7 = tnn.Conv2d(1024, 1024, kernel_size=1, stride=1, padding=0, bias=True)

        self.conv6_1 = tnn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0, bias=True)
        self.conv6_2 = conv(256, 512, k=3, s=2, p=1)

        self.conv7_1 = tnn.Conv2d(512, 128, kernel_size=1, stride=1, padding=0, bias=True)
        self.conv7_2 = conv(128, 256, k=3, s=2, p=1)

        self.conv3_3_norm = _L2Norm(256)
        self.conv4_3_norm = _L2Norm(512)
        self.conv5_3_norm = _L2Norm(512)

        self.conv3_3_norm_mbox_conf = conv(256, 4)
        self.conv3_3_norm_mbox_loc = conv(256, 4)

        self.conv4_3_norm_mbox_conf = conv(512, 2)
        self.conv4_3_norm_mbox_loc = conv(512, 4)

        self.conv5_3_norm_mbox_conf = conv(512, 2)
        self.conv5_3_norm_mbox_loc = conv(512, 4)

        self.fc7_mbox_conf = conv(1024, 2)
        self.fc7_mbox_loc = conv(1024, 4)

        self.conv6_2_mbox_conf = conv(512, 2)
        self.conv6_2_mbox_loc = conv(512, 4)

        self.conv7_2_mbox_conf = conv(256, 2)
        self.conv7_2_mbox_loc = conv(256, 4)

        self.relu = torch.nn.ReLU(inplace=True)

    def forward(self, x_rgb_u8):
        # x: (N,3,H,W) float32, range 0..255, RGB
        x = x_rgb_u8.to(dtype=torch.float32)
        x = x - self.minus.to(device=x.device)

        x = self.relu(self.conv1_1(x))
        x = self.relu(self.conv1_2(x))
        x = torch.nn.functional.max_pool2d(x, kernel_size=2, stride=2, padding=0)

        x = self.relu(self.conv2_1(x))
        x = self.relu(self.conv2_2(x))
        x = torch.nn.functional.max_pool2d(x, kernel_size=2, stride=2, padding=0)

        x = self.relu(self.conv3_1(x))
        x = self.relu(self.conv3_2(x))
        x = self.relu(self.conv3_3(x))
        f3_3 = x
        x = torch.nn.functional.max_pool2d(x, kernel_size=2, stride=2, padding=0)

        x = self.relu(self.conv4_1(x))
        x = self.relu(self.conv4_2(x))
        x = self.relu(self.conv4_3(x))
        f4_3 = x
        x = torch.nn.functional.max_pool2d(x, kernel_size=2, stride=2, padding=0)

        x = self.relu(self.conv5_1(x))
        x = self.relu(self.conv5_2(x))
        x = self.relu(self.conv5_3(x))
        f5_3 = x
        x = torch.nn.functional.max_pool2d(x, kernel_size=2, stride=2, padding=0)

        x = self.relu(self.fc6(x))
        x = self.relu(self.fc7(x))
        ffc7 = x

        x = self.relu(self.conv6_1(x))
        x = self.relu(self.conv6_2(x))
        f6_2 = x

        x = self.relu(self.conv7_1(x))
        x = self.relu(self.conv7_2(x))
        f7_2 = x

        f3_3 = self.conv3_3_norm(f3_3)
        f4_3 = self.conv4_3_norm(f4_3)
        f5_3 = self.conv5_3_norm(f5_3)

        cls1_raw = self.conv3_3_norm_mbox_conf(f3_3)
        reg1 = self.conv3_3_norm_mbox_loc(f3_3)

        # max-out background label
        bmax = torch.max(torch.max(cls1_raw[:, 0:1], cls1_raw[:, 1:2]), cls1_raw[:, 2:3])
        cls1 = torch.cat([bmax, cls1_raw[:, 3:4]], dim=1)
        cls1 = torch.nn.functional.softmax(cls1, dim=1)

        cls2 = torch.nn.functional.softmax(self.conv4_3_norm_mbox_conf(f4_3), dim=1)
        reg2 = self.conv4_3_norm_mbox_loc(f4_3)

        cls3 = torch.nn.functional.softmax(self.conv5_3_norm_mbox_conf(f5_3), dim=1)
        reg3 = self.conv5_3_norm_mbox_loc(f5_3)

        cls4 = torch.nn.functional.softmax(self.fc7_mbox_conf(ffc7), dim=1)
        reg4 = self.fc7_mbox_loc(ffc7)

        cls5 = torch.nn.functional.softmax(self.conv6_2_mbox_conf(f6_2), dim=1)
        reg5 = self.conv6_2_mbox_loc(f6_2)

        cls6 = torch.nn.functional.softmax(self.conv7_2_mbox_conf(f7_2), dim=1)
        reg6 = self.conv7_2_mbox_loc(f7_2)

        return [cls1, reg1, cls2, reg2, cls3, reg3, cls4, reg4, cls5, reg5, cls6, reg6]


def _assign_conv(module, weight_hwio: np.ndarray, bias_hwio: np.ndarray | None):
    w = torch.from_numpy(weight_hwio).to(dtype=torch.float32)
    w = w.permute(3, 2, 0, 1).contiguous()
    if module.weight.shape != w.shape:
        raise ValueError(f"Conv weight shape mismatch for {module}: expected {tuple(module.weight.shape)} got {tuple(w.shape)}")
    module.weight.data.copy_(w)
    if bias_hwio is not None:
        b = torch.from_numpy(bias_hwio).to(dtype=torch.float32).reshape(-1)
        if module.bias is None:
            raise ValueError("bias provided but module.bias is None")
        if module.bias.shape != b.shape:
            raise ValueError(f"Conv bias shape mismatch: expected {tuple(module.bias.shape)} got {tuple(b.shape)}")
        module.bias.data.copy_(b)


def _assign_l2norm(module: _L2Norm, weight_hwio: np.ndarray):
    w = torch.from_numpy(weight_hwio).to(dtype=torch.float32).reshape(-1)
    if module.weight.shape != w.shape:
        raise ValueError(f"L2Norm weight shape mismatch: expected {tuple(module.weight.shape)} got {tuple(w.shape)}")
    module.weight.data.copy_(w)


def _load_s3fd_npy(model: _S3FD, npy_path: Path):
    d = np.load(str(npy_path), allow_pickle=True)
    if not isinstance(d, dict):
        raise ValueError(f"Unexpected weights format: {type(d)}")
    for key, arr in d.items():
        parts = key.split('/')
        if len(parts) < 2:
            continue
        layer_name = parts[0]
        tensor_name = parts[1]
        if tensor_name.endswith(':0'):
            tensor_name = tensor_name[:-2]
        if not hasattr(model, layer_name):
            continue
        layer = getattr(model, layer_name)
        if isinstance(layer, torch.nn.Conv2d):
            if tensor_name == 'weight':
                bias_key = f"{layer_name}/bias:0"
                bias = d.get(bias_key, None)
                _assign_conv(layer, arr, bias)
        elif isinstance(layer, _L2Norm):
            if tensor_name == 'weight':
                _assign_l2norm(layer, arr)


class S3FDExtractor(object):
    def __init__(self, place_model_on_cpu: bool = False):
        nn.initialize_main_env()
        nn.initialize(data_format='NCHW')

        self._device = torch.device('cpu') if place_model_on_cpu else nn.device

        model_path = Path(__file__).resolve().parents[1] / 'DeepFaceLab-master' / 'facelib' / 'S3FD.npy'
        if not model_path.exists():
            # fallback: allow running if user copied weights elsewhere
            model_path = Path(__file__).parent / 'S3FD.npy'
        if not model_path.exists():
            raise Exception('Unable to load S3FD.npy (DeepFaceLab-master/facelib/S3FD.npy not found)')

        self.model = _S3FD().to(self._device)
        _load_s3fd_npy(self.model, model_path)
        self.model.eval()

    def __enter__(self):
        return self

    def __exit__(self, exc_type=None, exc_value=None, traceback=None):
        return False

    def extract(self, input_image, is_bgr=True, is_remove_intersects=False):
        if is_bgr:
            input_image = input_image[:, :, ::-1]
            is_bgr = False

        (h, w, ch) = input_image.shape

        d = max(w, h)
        scale_to = 640 if d >= 1280 else d / 2
        scale_to = max(64, scale_to)

        input_scale = d / scale_to
        input_image = cv2.resize(input_image, (int(w / input_scale), int(h / input_scale)), interpolation=cv2.INTER_LINEAR)

        # NCHW, RGB, float32
        img = input_image.astype(np.float32)
        x = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(self._device)

        with torch.no_grad():
            outs = self.model(x)

        # Convert to NHWC numpy arrays without batch dim, matching original refine()
        olist = []
        for t in outs:
            y = t.detach().to('cpu')
            y = y.squeeze(0).permute(1, 2, 0).numpy()
            olist.append(y)

        detected_faces = []
        for ltrb in self.refine(olist):
            l, t, r, b = [x * input_scale for x in ltrb]
            bt = b - t
            if min(r - l, bt) < 40:
                continue
            b += bt * 0.1
            detected_faces.append([int(x) for x in (l, t, r, b)])

        detected_faces = [[(l, t, r, b), (r - l) * (b - t)] for (l, t, r, b) in detected_faces]
        detected_faces = sorted(detected_faces, key=operator.itemgetter(1), reverse=True)
        detected_faces = [x[0] for x in detected_faces]

        if is_remove_intersects:
            for i in range(len(detected_faces) - 1, 0, -1):
                l1, t1, r1, b1 = detected_faces[i]
                l0, t0, r0, b0 = detected_faces[i - 1]

                dx = min(r0, r1) - max(l0, l1)
                dy = min(b0, b1) - max(t0, t1)
                if (dx >= 0) and (dy >= 0):
                    detected_faces.pop(i)

        return detected_faces

    def refine(self, olist):
        bboxlist = []
        for i, (ocls, oreg) in enumerate(zip(olist[::2], olist[1::2])):
            stride = 2 ** (i + 2)  # 4,8,16,32,64,128
            s_d2 = stride / 2
            s_m4 = stride * 4

            for hindex, windex in zip(*np.where(ocls[..., 1] > 0.05)):
                score = ocls[hindex, windex, 1]
                loc = oreg[hindex, windex, :]
                priors = np.array([windex * stride + s_d2, hindex * stride + s_d2, s_m4, s_m4])
                priors_2p = priors[2:]
                box = np.concatenate(
                    (
                        priors[:2] + loc[:2] * 0.1 * priors_2p,
                        priors_2p * np.exp(loc[2:] * 0.2),
                    )
                )
                box[:2] -= box[2:] / 2
                box[2:] += box[:2]

                bboxlist.append([*box, score])

        bboxlist = np.array(bboxlist)
        if len(bboxlist) == 0:
            bboxlist = np.zeros((1, 5))

        bboxlist = bboxlist[self.refine_nms(bboxlist, 0.3), :]
        bboxlist = [x[:-1].astype(np.int32) for x in bboxlist if x[-1] >= 0.5]
        return bboxlist

    def refine_nms(self, dets, thresh):
        keep = list()
        if len(dets) == 0:
            return keep

        x_1, y_1, x_2, y_2, scores = dets[:, 0], dets[:, 1], dets[:, 2], dets[:, 3], dets[:, 4]
        areas = (x_2 - x_1 + 1) * (y_2 - y_1 + 1)
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx_1 = np.maximum(x_1[i], x_1[order[1:]])
            yy_1 = np.maximum(y_1[i], y_1[order[1:]])
            xx_2 = np.minimum(x_2[i], x_2[order[1:]])
            yy_2 = np.minimum(y_2[i], y_2[order[1:]])

            w = np.maximum(0.0, xx_2 - xx_1 + 1)
            h = np.maximum(0.0, yy_2 - yy_1 + 1)
            inter = w * h
            ovr = inter / (areas[i] + areas[order[1:]] - inter)

            inds = np.where(ovr <= thresh)[0]
            order = order[inds + 1]
        return keep
