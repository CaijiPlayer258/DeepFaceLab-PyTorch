import os
import sys
import traceback
import queue
import threading
import time
import warnings
import numpy as np
import itertools
from pathlib import Path
from core import pathex
from core import imagelib
import cv2
import models
from core.interact import interact as io


def main(model_class_name, saved_models_path):
    silent = str(os.environ.get('DFL_SILENT_INPUT', '')).strip().lower() in ('1', 'y', 'yes', 'true', 'on')

    model = models.import_model(model_class_name)(
                        is_exporting=True,
                        saved_models_path=saved_models_path,
                        cpu_only=True,
                        silent_start=silent,
                        force_model_name='export' if silent else None)
    # torch.onnx 在导出时会打印一些“常量折叠未应用”的告警（例如 Slice steps!=1）。
    # 这类提示通常不影响导出结果，但在控制台里很像“报错”。这里做定向过滤：
    # 只屏蔽该条已知无害 warning，不影响其他重要告警/异常。
    warnings.filterwarnings(
        'ignore',
        message=r'Constant folding - Only steps=1 can be constant folded.*onnx::Slice.*',
        category=UserWarning,
    )

    model.export_dfm()
