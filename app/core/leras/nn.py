"""Leras (PyTorch port).

DFL's lightweight NN layer/model helpers.

NCHW is the default data format in PyTorch.
"""

import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

import numpy as np
import torch

import numpy as np

from core.interact import interact as io
from .device import Devices


class nn:
    current_DeviceConfig = None

    torch = None
    device = None

    data_format = None
    conv2d_ch_axis = None
    conv2d_spatial_axes = None

    floatx = None  # torch dtype

    @staticmethod
    def initialize(device_config=None, floatx="float32", data_format="NCHW"):
        if device_config is None:
            device_config = nn.DeviceConfig.BestGPU()

        nn.setCurrentDeviceConfig(device_config)

        if nn.torch is None:
            import torch

            nn.torch = torch

            # Load registries (layers/ops/etc). Must be re-entrant.
            import core.leras.ops  # noqa: F401
            import core.leras.layers  # noqa: F401
            import core.leras.initializers  # noqa: F401
            import core.leras.optimizers  # noqa: F401
            import core.leras.models  # noqa: F401
            import core.leras.archis  # noqa: F401

        torch = nn.torch

        if len(device_config.devices) == 0:
            nn.device = torch.device('cpu')
        else:
            dev0 = device_config.devices[0]
            dev_type = getattr(dev0, 'device_type', 'GPU')
            if dev_type == 'MPS':
                nn.device = torch.device('mps')
            else:
                nn.device = torch.device(f'cuda:{dev0.index}')

        if floatx == "float32":
            nn.set_floatx(torch.float32)
        elif floatx == "float16":
            nn.set_floatx(torch.float16)
        else:
            raise ValueError(f"unsupported floatx {floatx}")

        nn.set_data_format(data_format)

    @staticmethod
    def initialize_main_env():
        Devices.initialize_main_env()
        if nn.torch is None:
            nn.initialize(nn.DeviceConfig.BestGPU())

    @staticmethod
    def set_floatx(torch_dtype):
        nn.floatx = torch_dtype

    @staticmethod
    def set_data_format(data_format):
        if data_format not in ("NHWC", "NCHW"):
            raise ValueError(f"unsupported data_format {data_format}")
        nn.data_format = data_format

        if data_format == "NHWC":
            nn.conv2d_ch_axis = 3
            nn.conv2d_spatial_axes = [1, 2]
        else:
            nn.conv2d_ch_axis = 1
            nn.conv2d_spatial_axes = [2, 3]

    @staticmethod
    def get4Dshape(w, h, c):
        if nn.data_format == "NHWC":
            return (None, h, w, c)
        return (None, c, h, w)

    @staticmethod
    def to_data_format(x, to_data_format, from_data_format):
        if to_data_format == from_data_format:
            return x

        if to_data_format == "NHWC":
            return np.transpose(x, (0, 2, 3, 1))
        if to_data_format == "NCHW":
            return np.transpose(x, (0, 3, 1, 2))
        raise ValueError(f"unsupported to_data_format {to_data_format}")

    @staticmethod
    def getCurrentDeviceConfig():
        if nn.current_DeviceConfig is None:
            nn.current_DeviceConfig = nn.DeviceConfig.BestGPU()
        return nn.current_DeviceConfig

    @staticmethod
    def setCurrentDeviceConfig(device_config):
        nn.current_DeviceConfig = device_config

    @staticmethod
    def reset_session():
        pass

    @staticmethod
    def close_session():
        pass

    @staticmethod
    def ask_choose_device_idxs(choose_only_one=False, allow_cpu=True, suggest_best_multi_gpu=False, suggest_all_gpu=False):
        devices = Devices.getDevices()
        if len(devices) == 0:
            return []

        all_devices_indexes = [device.index for device in devices]

        if choose_only_one:
            suggest_best_multi_gpu = False
            suggest_all_gpu = False

        if suggest_all_gpu:
            best_device_indexes = all_devices_indexes
        elif suggest_best_multi_gpu:
            best_device_indexes = [device.index for device in devices.get_equal_devices(devices.get_best_device())]
        else:
            best_device_indexes = [devices.get_best_device().index]

        best_device_indexes = ",".join([str(x) for x in best_device_indexes])

        io.log_info("")
        io.log_info("选择一个 GPU idx。" if choose_only_one else "选择一个或多个 GPU idx（用逗号分隔）。")
        io.log_info("")

        if allow_cpu:
            io.log_info("[CPU] : CPU")
        for device in devices:
            io.log_info(f"  [{device.index}] : {device.name}")

        io.log_info("")

        while True:
            try:
                prompt = "选择哪个 GPU index?" if choose_only_one else "选择哪些 GPU indexes?"
                choosed_idxs = io.input_str(prompt, best_device_indexes)

                if allow_cpu and choosed_idxs.lower() == "cpu":
                    choosed_idxs = []
                    break

                choosed_idxs = [int(x) for x in choosed_idxs.split(',')]

                if choose_only_one:
                    if len(choosed_idxs) == 1:
                        break
                else:
                    if all([idx in all_devices_indexes for idx in choosed_idxs]):
                        break
            except Exception:
                pass

        io.log_info("")
        return choosed_idxs

    @staticmethod
    def depth_to_space(x, block_size):
        import torch
        import torch.nn.functional as F

        # TF depth_to_space (NCHW):
        #   output[n, c, h*r+i, w*r+j] = input[n, (i*r+j)*C_out + c, h, w]
        # PT pixel_shuffle:
        #   output[n, c, h*r+i, w*r+j] = input[n, c*r² + i*r + j, h, w]
        # Permute input channels to match TF ordering before pixel_shuffle
        # so that converted TF Conv2D weights produce correct spatial layout.
        C_in = x.shape[1]
        r = block_size
        C_out = C_in // (r * r)

        idx = torch.arange(C_in, device=x.device)
        c = idx // (r * r)       # output channel index
        ij = idx % (r * r)       # sub-pixel position (i*r+j) packed
        i = ij // r
        j = ij % r
        perm = (i * r + j) * C_out + c

        return F.pixel_shuffle(x[:, perm, :, :], r)

    @staticmethod
    def flatten(x):
        # 原版 iperov：通道优先展平（NCHW reshape），与 ops 版本一致
        return x.view(x.size(0), -1)

    @staticmethod
    def pixel_norm(x, epsilon=1e-6, axes=-1):
        import torch

        return x / torch.sqrt(torch.mean(x**2, dim=axes, keepdim=True) + epsilon)

    @staticmethod
    def reshape_4D(x, h, w, c):
        batch_size = x.size(0)
        # 原版 iperov：通道优先（先按 (c,h,w) 解释），与 ops 版本一致
        return x.view(batch_size, c, h, w)

    class DeviceConfig:
        @staticmethod
        def ask_choose_device(*args, **kwargs):
            return nn.DeviceConfig.GPUIndexes(nn.ask_choose_device_idxs(*args, **kwargs))

        def __init__(self, devices=None):
            devices = devices or []
            if not isinstance(devices, Devices):
                devices = Devices(devices)
            self.devices = devices
            self.cpu_only = len(devices) == 0

        @staticmethod
        def BestGPU():
            devices = Devices.getDevices()
            if len(devices) == 0:
                return nn.DeviceConfig.CPU()
            return nn.DeviceConfig([devices.get_best_device()])

        @staticmethod
        def WorstGPU():
            devices = Devices.getDevices()
            if len(devices) == 0:
                return nn.DeviceConfig.CPU()
            return nn.DeviceConfig([devices.get_worst_device()])

        @staticmethod
        def GPUIndexes(indexes):
            if len(indexes) != 0:
                devices = Devices.getDevices().get_devices_from_index_list(indexes)
            else:
                devices = []
            return nn.DeviceConfig(devices)

        @staticmethod
        def CPU():
            return nn.DeviceConfig([])


from . import ops  # noqa: F401


from core.leras.archis.ArchiBase import ArchiBase
from core.leras.archis.DeepFakeArchi import DeepFakeArchi

nn.ArchiBase = ArchiBase
nn.DeepFakeArchi = DeepFakeArchi