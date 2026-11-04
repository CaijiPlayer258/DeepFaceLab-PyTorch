"""
人脸检测器共享输入预处理。

两种检测模式：
- one_stage     : 整图一次缩放到检测器输入尺寸，再检测（快，适合小图/常见场景）
- sliding_window: 用固定窗口在图上滑动扫描，逐窗口检测后合并（适合大图中找小脸）

两种缩放方式（resize_mode，作用于 one_stage 整图缩放 与 sliding_window 边缘窗口）：
- letterbox: 等比缩放 + 左上角补边到 size×size（保纵横比、不变形）
- warp     : 直接拉伸填满 size×size（不保纵横比、不补边，速度最快）

所有路径返回 meta 字典，配合 map_back() 把检测框/关键点从处理空间映射回原图空间。
仅依赖 cv2 / numpy，不引入项目内其他模块。
"""
import cv2
import numpy as np
from typing import Tuple, Dict, Callable, List


def letterbox(img: np.ndarray, size: int = 640, pad_value: int = 0,
              allow_upscale: bool = True) -> Tuple[np.ndarray, float, int, int]:
    """等比缩放 + 左上角补边到 size×size。

    Args:
        img: BGR 图像
        size: 目标正方形边长
        pad_value: 补边像素值 (0=黑, 128=灰)
        allow_upscale: 是否允许放大 (小图放大到 size)

    Returns:
        (out, scale, pad_x, pad_y)
        - out: size×size 图像
        - scale: 统一缩放比
        - pad_x / pad_y: 补边偏移 (左上角粘贴时为 0)
    """
    H, W = img.shape[:2]
    scale = min(size / H, size / W)
    if scale > 1.0 and not allow_upscale:
        scale = 1.0
    new_h = max(1, int(round(H * scale)))
    new_w = max(1, int(round(W * scale)))
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    out = np.full((size, size, 3), pad_value, dtype=np.uint8)
    out[:new_h, :new_w, :] = resized
    return out, scale, 0, 0


def warp_affine(img: np.ndarray, size: int = 640) -> Tuple[np.ndarray, float, float]:
    """直接拉伸到 size×size（非均匀缩放，不保纵横比、不补边）。

    Returns:
        (out, scale_x, scale_y)  —— x/y 各自独立缩放比
    """
    H, W = img.shape[:2]
    scale_x = size / W
    scale_y = size / H
    out = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    return out, scale_x, scale_y


def preprocess(img: np.ndarray, resize_mode: str = 'letterbox', size: int = 640,
               pad_value: int = 0) -> Tuple[np.ndarray, dict]:
    """统一调度入口（one_stage 整图缩放）。

    Args:
        img: BGR 图像
        resize_mode: 'letterbox' | 'warp'
        size: 目标正方形尺寸
        pad_value: 补边像素值

    Returns:
        (out, meta) —— out 为 size×size 处理图，meta 供 map_back 使用
    """
    # 规范化为 8 的倍数（SCRFD anchor 网格 stride 8/16/32 对齐）
    size = max(8, int(size) // 8 * 8)
    if resize_mode == 'warp':
        out, sx, sy = warp_affine(img, size)
        return out, {'mode': 'warp', 'scale_x': sx, 'scale_y': sy}
    # letterbox (default)
    out, scale, px, py = letterbox(img, size, pad_value=pad_value)
    return out, {'mode': 'letterbox', 'scale': scale, 'pad_x': px, 'pad_y': py}


def map_back(coords: np.ndarray, meta: dict) -> np.ndarray:
    """把检测框/关键点从处理空间映射回原图空间。

    Args:
        coords: 任意数组，末维为 2 (点 x,y) 或 4 (框 x1,y1,x2,y2)。
                如 SCRFD dets 形状 (N,5) 传 det[:, :4]，kps (N,5,2) 直接传。
        meta: preprocess 返回的元数据

    Returns:
        映射后的新数组（不改动原数组）
    """
    c = np.asarray(coords).copy()
    last = c.shape[-1]
    if last == 2:
        x, y = c[..., 0], c[..., 1]
    elif last == 4:
        x, y = c[..., [0, 2]], c[..., [1, 3]]
    else:
        raise ValueError(f'last dim must be 2 or 4, got {last}')

    if meta['mode'] == 'warp':
        x = x / meta['scale_x']
        y = y / meta['scale_y']
    else:  # letterbox
        x = (x - meta['pad_x']) / meta['scale']
        y = (y - meta['pad_y']) / meta['scale']

    if last == 2:
        c[..., 0], c[..., 1] = x, y
    else:
        c[..., 0], c[..., 2] = x[..., 0], x[..., 1]
        c[..., 1], c[..., 3] = y[..., 0], y[..., 1]
    return c


# ── 滑窗检测 ─────────────────────────────────────────────

def _nms_boxes(boxes: np.ndarray, nms_thresh: float = 0.4) -> np.ndarray:
    """对 (N,5) [x1,y1,x2,y2,score] 做 NMS，返回保留索引。"""
    if len(boxes) == 0:
        return boxes
    x1, y1, x2, y2, scores = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3], boxes[:, 4]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        order = order[np.where(ovr <= nms_thresh)[0] + 1]
    return boxes[keep]


def sliding_window_detect(img: np.ndarray, window_size: int, resize_mode: str,
                          detect_fn: Callable[[np.ndarray], np.ndarray],
                          pad_value: int = 0,
                          overlap_ratio: float = 0.2) -> np.ndarray:
    """滑窗扫描检测。

    用固定窗口在图上滑动，逐窗口调用 detect_fn 检测，把检测框映射回原图坐标后合并。

    Args:
        img: BGR 原图
        window_size: 扫描窗口边长（检测器输入尺寸）
        resize_mode: 边缘不完整窗口的处理方式 'letterbox' | 'warp'
        detect_fn: 接收单个窗口图像 (size×size)，返回 boxes 数组 (N,5)
                   [x1,y1,x2,y2,score]，坐标在窗口内。
        pad_value: letterbox 补边像素值
        overlap_ratio: 相邻窗口重叠比例 (0~1)，越大越慢但漏检越少

    Returns:
        boxes: (N,5) 合并后原图坐标的检测框
    """
    h, w = img.shape[:2]
    ws = window_size
    stride = max(1, int(ws * (1 - overlap_ratio)))
    all_boxes = []

    for y in range(0, h, stride):
        for x in range(0, w, stride):
            ch = min(ws, h - y)
            cw = min(ws, w - x)
            crop = img[y:y + ch, x:x + cw]

            if ch == ws and cw == ws:
                # 完整窗口，1:1 坐标映射
                boxes = detect_fn(crop)
                if len(boxes) == 0:
                    continue
                boxes[:, [0, 2]] += x
                boxes[:, [1, 3]] += y
            else:
                # 边缘不完整窗口：按 resize_mode 填充或拉伸
                if resize_mode == 'warp':
                    sx, sy = ws / cw, ws / ch
                    win = cv2.resize(crop, (ws, ws), interpolation=cv2.INTER_AREA)
                    boxes = detect_fn(win)
                    if len(boxes) == 0:
                        continue
                    boxes[:, [0, 2]] = boxes[:, [0, 2]] / sx + x
                    boxes[:, [1, 3]] = boxes[:, [1, 3]] / sy + y
                else:  # letterbox 填充（内容 1:1 贴左上，其余补边）
                    win = np.full((ws, ws, 3), pad_value, dtype=np.uint8)
                    win[:ch, :cw, :] = crop
                    boxes = detect_fn(win)
                    if len(boxes) == 0:
                        continue
                    boxes[:, [0, 2]] += x
                    boxes[:, [1, 3]] += y
            all_boxes.append(boxes)

    if not all_boxes:
        return np.empty((0, 5))
    merged = np.concatenate(all_boxes, axis=0)
    return _nms_boxes(merged)
