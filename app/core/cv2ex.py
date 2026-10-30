import cv2
import numpy as np
from pathlib import Path
from core.interact import interact as io
import traceback
# 注意：core.imagelib 会拖进 scipy（+0.4s / +42MB）。本模块只有 normalize_channels
# 一个函数用到它，而 cv2ex 被 DFLIMG 在模块顶部 import —— 预取子进程、Sorter、
# Filter 全都要付这笔钱。所以改为函数内导入。

def cv2_imread(filename, flags=cv2.IMREAD_UNCHANGED, loader_func=None, verbose=True):
    """
    allows to open non-english characters path
    """
    try:
        if loader_func is not None:
            bytes = bytearray(loader_func(filename))
        else:
            with open(filename, "rb") as stream:
                bytes = bytearray(stream.read())
        numpyarray = np.asarray(bytes, dtype=np.uint8)
        return cv2.imdecode(numpyarray, flags)
    except:
        if verbose:
            io.log_err(f"Exception occured in cv2_imread : {traceback.format_exc()}")
        return None

def cv2_imwrite(filename, img, *args):
    ret, buf = cv2.imencode( Path(filename).suffix, img, *args)
    if ret == True:
        try:
            with open(filename, "wb") as stream:
                stream.write( buf )
        except:
            pass

def cv2_resize(x, *args, **kwargs):
    h,w,c = x.shape
    x = cv2.resize(x, *args, **kwargs)
    
    from core import imagelib                  # 惰性：scipy
    x = imagelib.normalize_channels(x, c)
    return x
    