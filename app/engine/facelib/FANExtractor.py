import traceback
from pathlib import Path

import cv2
import numpy as np
import torch

from facelib import FaceType, LandmarksProcessor
from core.leras import nn


class _TFBatchNorm2D(torch.nn.Module):
    """Inference-only BN matching DeepFaceLab-master BatchNorm2D."""

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.dim = int(dim)
        self.eps = float(eps)
        self.weight = torch.nn.Parameter(torch.ones(self.dim, dtype=torch.float32))
        self.bias = torch.nn.Parameter(torch.zeros(self.dim, dtype=torch.float32))
        self.register_buffer('running_mean', torch.zeros(self.dim, dtype=torch.float32))
        self.register_buffer('running_var', torch.zeros(self.dim, dtype=torch.float32))

    def forward(self, x):
        # x: (N,C,H,W)
        mean = self.running_mean.view(1, -1, 1, 1)
        var = self.running_var.view(1, -1, 1, 1)
        w = self.weight.view(1, -1, 1, 1)
        b = self.bias.view(1, -1, 1, 1)
        x = (x - mean) / torch.sqrt(var + self.eps)
        return x * w + b


class _ConvBlock(torch.nn.Module):
    def __init__(self, in_planes: int, out_planes: int):
        super().__init__()
        tnn = torch.nn
        self.in_planes = int(in_planes)
        self.out_planes = int(out_planes)

        self.bn1 = _TFBatchNorm2D(in_planes)
        self.conv1 = tnn.Conv2d(in_planes, out_planes // 2, kernel_size=3, stride=1, padding=1, bias=False)

        self.bn2 = _TFBatchNorm2D(out_planes // 2)
        self.conv2 = tnn.Conv2d(out_planes // 2, out_planes // 4, kernel_size=3, stride=1, padding=1, bias=False)

        self.bn3 = _TFBatchNorm2D(out_planes // 4)
        self.conv3 = tnn.Conv2d(out_planes // 4, out_planes // 4, kernel_size=3, stride=1, padding=1, bias=False)

        if self.in_planes != self.out_planes:
            self.down_bn1 = _TFBatchNorm2D(in_planes)
            self.down_conv1 = tnn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False)
        else:
            self.down_bn1 = None
            self.down_conv1 = None

        self.relu = tnn.ReLU(inplace=True)

    def forward(self, x):
        inp = x
        x = self.bn1(x)
        x = self.relu(x)
        out1 = self.conv1(x)

        x = self.bn2(out1)
        x = self.relu(x)
        out2 = self.conv2(x)

        x = self.bn3(out2)
        x = self.relu(x)
        out3 = self.conv3(x)

        x = torch.cat([out1, out2, out3], dim=1)

        if self.in_planes != self.out_planes:
            down = self.down_bn1(inp)
            down = self.relu(down)
            down = self.down_conv1(down)
            x = x + down
        else:
            x = x + inp
        return x


class _HourGlass(torch.nn.Module):
    def __init__(self, in_planes: int, depth: int):
        super().__init__()
        self.b1 = _ConvBlock(in_planes, 256)
        self.b2 = _ConvBlock(in_planes, 256)

        if depth > 1:
            self.b2_plus = _HourGlass(256, depth - 1)
        else:
            self.b2_plus = _ConvBlock(256, 256)

        self.b3 = _ConvBlock(256, 256)

    def forward(self, x):
        up1 = self.b1(x)
        low1 = torch.nn.functional.avg_pool2d(x, kernel_size=2, stride=2)
        low1 = self.b2(low1)
        low2 = self.b2_plus(low1)
        low3 = self.b3(low2)
        up2 = torch.nn.functional.interpolate(low3, scale_factor=2, mode='nearest')
        return up1 + up2


class _FAN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        tnn = torch.nn

        self.conv1 = tnn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=True)
        self.bn1 = _TFBatchNorm2D(64)

        self.conv2 = _ConvBlock(64, 128)
        self.conv3 = _ConvBlock(128, 128)
        self.conv4 = _ConvBlock(128, 256)

        self.relu = tnn.ReLU(inplace=True)

        # 4 stacked hourglasses. Use explicit attribute names to match .npy keys.
        for i in range(4):
            setattr(self, f"m_{i}", _HourGlass(256, 4))
            setattr(self, f"top_m_{i}", _ConvBlock(256, 256))
            setattr(self, f"conv_last_{i}", tnn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0, bias=True))
            setattr(self, f"bn_end_{i}", _TFBatchNorm2D(256))
            setattr(self, f"l_{i}", tnn.Conv2d(256, 68, kernel_size=1, stride=1, padding=0, bias=True))
            if i < 3:
                setattr(self, f"bl_{i}", tnn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0, bias=True))
                setattr(self, f"al_{i}", tnn.Conv2d(68, 256, kernel_size=1, stride=1, padding=0, bias=True))

    def forward(self, x):
        # x: (N,3,256,256) RGB float32 0..1
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = torch.nn.functional.avg_pool2d(x, kernel_size=2, stride=2)
        x = self.conv3(x)
        x = self.conv4(x)

        previous = x
        outputs = []
        for i in range(4):
            ll = getattr(self, f"m_{i}")(previous)
            ll = getattr(self, f"top_m_{i}")(ll)
            ll = getattr(self, f"conv_last_{i}")(ll)
            ll = getattr(self, f"bn_end_{i}")(ll)
            ll = self.relu(ll)
            tmp_out = getattr(self, f"l_{i}")(ll)
            outputs.append(tmp_out)
            if i < 3:
                ll2 = getattr(self, f"bl_{i}")(ll)
                previous = previous + ll2 + getattr(self, f"al_{i}")(tmp_out)

        return outputs[-1]


def _set_by_path(root, path_parts):
    obj = root
    for p in path_parts:
        obj = getattr(obj, p)
    return obj


def _load_fan_npy(model: _FAN, npy_path: Path):
    d = np.load(str(npy_path), allow_pickle=True)
    if not isinstance(d, dict):
        raise ValueError(f"Unexpected weights format: {type(d)}")

    for key, arr in d.items():
        parts = key.split('/')
        if len(parts) < 2:
            continue
        tensor_name = parts[-1]
        if tensor_name.endswith(':0'):
            tensor_name = tensor_name[:-2]
        obj_path = parts[:-1]

        try:
            layer = _set_by_path(model, obj_path)
        except Exception:
            continue

        if isinstance(layer, torch.nn.Conv2d):
            if tensor_name == 'weight':
                w = torch.from_numpy(arr).to(dtype=torch.float32).permute(3, 2, 0, 1).contiguous()
                if layer.weight.shape != w.shape:
                    raise ValueError(f"Conv weight mismatch for {key}: expected {tuple(layer.weight.shape)} got {tuple(w.shape)}")
                layer.weight.data.copy_(w)
            elif tensor_name == 'bias':
                b = torch.from_numpy(arr).to(dtype=torch.float32).reshape(-1)
                if layer.bias is None:
                    raise ValueError(f"Bias provided but layer has no bias: {key}")
                if layer.bias.shape != b.shape:
                    raise ValueError(f"Conv bias mismatch for {key}: expected {tuple(layer.bias.shape)} got {tuple(b.shape)}")
                layer.bias.data.copy_(b)

        elif isinstance(layer, _TFBatchNorm2D):
            v = torch.from_numpy(arr).to(dtype=torch.float32).reshape(-1)
            if tensor_name == 'weight':
                layer.weight.data.copy_(v)
            elif tensor_name == 'bias':
                layer.bias.data.copy_(v)
            elif tensor_name == 'running_mean':
                layer.running_mean.data.copy_(v)
            elif tensor_name == 'running_var':
                layer.running_var.data.copy_(v)


class FANExtractor(object):
    """DeepFaceLab FAN landmark extractor (Torch port, weights from DeepFaceLab-master)."""

    def __init__(self, landmarks_3D: bool = False, place_model_on_cpu: bool = False):
        nn.initialize_main_env()
        nn.initialize(data_format='NCHW')

        model_path = Path(__file__).resolve().parents[1] / 'DeepFaceLab-master' / 'facelib' / (
            '3DFAN.npy' if landmarks_3D else '2DFAN.npy'
        )
        if not model_path.exists():
            # fallback for users copying weights locally
            model_path = Path(__file__).parent / ('3DFAN.npy' if landmarks_3D else '2DFAN.npy')
        if not model_path.exists():
            raise Exception('Unable to load FANExtractor model .npy')

        self._device = torch.device('cpu') if place_model_on_cpu else nn.device
        self.model = _FAN().to(self._device)
        _load_fan_npy(self.model, model_path)
        self.model.eval()

    def extract(self, input_image, rects, second_pass_extractor=None, is_bgr=True, multi_sample=False):
        if rects is None or len(rects) == 0:
            return []

        if is_bgr:
            input_image = input_image[:, :, ::-1]
            is_bgr = False

        landmarks = []
        for (left, top, right, bottom) in rects:
            scale = (right - left + bottom - top) / 195.0

            center = np.array([(left + right) / 2.0, (top + bottom) / 2.0], dtype=np.float32)
            centers = [center]
            if multi_sample:
                centers += [
                    center + [-1, -1],
                    center + [1, -1],
                    center + [1, 1],
                    center + [-1, 1],
                ]

            images = []
            ptss = []

            try:
                for c in centers:
                    images.append(self.crop(input_image, c, scale))

                images = np.stack(images).astype(np.float32) / 255.0

                with torch.no_grad():
                    x = torch.from_numpy(images).permute(0, 3, 1, 2).to(self._device, dtype=torch.float32)
                    pred = self.model(x).detach().to('cpu').numpy()

                for i in range(pred.shape[0]):
                    ptss.append(self.get_pts_from_predict(pred[i], centers[i], scale))

                pts_img = np.mean(np.array(ptss), 0)
                landmarks.append(pts_img)
            except Exception:
                landmarks.append(None)

        if second_pass_extractor is not None:
            for i, lmrks in enumerate(landmarks):
                try:
                    if lmrks is not None:
                        image_to_face_mat = LandmarksProcessor.get_transform_mat(lmrks, 256, FaceType.FULL)
                        face_image = cv2.warpAffine(input_image, image_to_face_mat, (256, 256), cv2.INTER_CUBIC)

                        rects2 = second_pass_extractor.extract(face_image, is_bgr=is_bgr)
                        if rects2 is not None and len(rects2) == 1:
                            lmrks2 = self.extract(face_image, [rects2[0]], is_bgr=is_bgr, multi_sample=True)[0]
                            landmarks[i] = LandmarksProcessor.transform_points(lmrks2, image_to_face_mat, True)
                except Exception:
                    pass

        return landmarks

    def transform(self, point, center, scale, resolution):
        pt = np.array([point[0], point[1], 1.0], dtype=np.float32)
        h = 200.0 * scale
        m = np.eye(3, dtype=np.float32)
        m[0, 0] = resolution / h
        m[1, 1] = resolution / h
        m[0, 2] = resolution * (-center[0] / h + 0.5)
        m[1, 2] = resolution * (-center[1] / h + 0.5)
        m = np.linalg.inv(m)
        return np.matmul(m, pt)[0:2]

    def crop(self, image, center, scale, resolution=256.0):
        ul = self.transform([1, 1], center, scale, resolution).astype(np.int32)
        br = self.transform([resolution, resolution], center, scale, resolution).astype(np.int32)

        if image.ndim > 2:
            newDim = np.array([br[1] - ul[1], br[0] - ul[0], image.shape[2]], dtype=np.int32)
            newImg = np.zeros(newDim, dtype=np.uint8)
        else:
            newDim = np.array([br[1] - ul[1], br[0] - ul[0]], dtype=np.int32)
            newImg = np.zeros(newDim, dtype=np.uint8)

        ht = image.shape[0]
        wd = image.shape[1]
        newX = np.array([max(1, -ul[0] + 1), min(br[0], wd) - ul[0]], dtype=np.int32)
        newY = np.array([max(1, -ul[1] + 1), min(br[1], ht) - ul[1]], dtype=np.int32)
        oldX = np.array([max(1, ul[0] + 1), min(br[0], wd)], dtype=np.int32)
        oldY = np.array([max(1, ul[1] + 1), min(br[1], ht)], dtype=np.int32)

        newImg[newY[0] - 1 : newY[1], newX[0] - 1 : newX[1]] = image[oldY[0] - 1 : oldY[1], oldX[0] - 1 : oldX[1]]

        newImg = cv2.resize(newImg, dsize=(int(resolution), int(resolution)), interpolation=cv2.INTER_LINEAR)
        return newImg

    def get_pts_from_predict(self, a, center, scale):
        a_ch, a_h, a_w = a.shape

        b = a.reshape((a_ch, a_h * a_w))
        c = b.argmax(1).reshape((a_ch, 1)).repeat(2, axis=1).astype(np.float32)
        c[:, 0] %= a_w
        c[:, 1] = np.apply_along_axis(lambda x: np.floor(x / a_w), 0, c[:, 1])

        for i in range(a_ch):
            pX, pY = int(c[i, 0]), int(c[i, 1])
            if pX > 0 and pX < 63 and pY > 0 and pY < 63:
                diff = np.array([a[i, pY, pX + 1] - a[i, pY, pX - 1], a[i, pY + 1, pX] - a[i, pY - 1, pX]])
                c[i] += np.sign(diff) * 0.25

        c += 0.5

        return np.array([self.transform(c[i], center, scale, a_w) for i in range(a_ch)])
