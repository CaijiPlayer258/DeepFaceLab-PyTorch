"""
DeepFaceLab Torch - Extractor Module
人脸提取器模块：从视频或图片中提取对齐后的人脸
"""

import sys
import os
from pathlib import Path
import cv2
import numpy as np
import traceback
import argparse
from typing import List, Tuple, Optional, Dict
import tqdm
from multiprocessing import cpu_count
import concurrent.futures
import time
import math

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pickle
import struct
from core.pathex import save_dfljpg

# 减少ONNX Runtime警告
import onnxruntime
onnxruntime.set_default_logger_severity(3)
import warnings
import subprocess  # FFmpeg pipe
warnings.filterwarnings('ignore', module='onnxruntime')

# 导入modelhub模块（安全导入：单个模型缺失/导入失败不影响启动，缺失的模型不可选）
def _safe_import(module_path: str, cls_name: str):
    """安全导入 modelhub 模型类，失败返回 None（该模型不可用）。"""
    try:
        import importlib
        return getattr(importlib.import_module(module_path), cls_name)
    except Exception as e:
        print(f'[Extractor] 模型 {module_path}.{cls_name} 导入失败，已跳过: {e}')
        return None

BlazeFace = _safe_import('modelhub.onnx.BlazeFace.BlazeFace', 'BlazeFace')
CenterFace = _safe_import('modelhub.onnx.CenterFace.CenterFace', 'CenterFace')
S3FD = _safe_import('modelhub.onnx.S3FD.S3FD', 'S3FD')
YoloV5Face = _safe_import('modelhub.onnx.YoloV5Face.YoloV5Face', 'YoloV5Face')
FastFaceAlign = _safe_import('modelhub.onnx.FastFaceAlign.FastFaceAlign', 'FastFaceAlign')
InsightFace2D106 = _safe_import('modelhub.onnx.InsightFace2d106.InsightFace2D106', 'InsightFace2D106')
FaceMesh = _safe_import('modelhub.onnx.FaceMesh.FaceMesh', 'FaceMesh')
YoloV8Face = _safe_import('modelhub.onnx.YoloV8Face.YoloV8Face', 'YoloV8Face')
RetinaFace = _safe_import('modelhub.onnx.RetinaFace.RetinaFace', 'RetinaFace')
DamoFD = _safe_import('modelhub.onnx.DamoFD.DamoFD', 'DamoFD')
TinyMog = _safe_import('modelhub.onnx.TinyMog.TinyMog', 'TinyMog')
ULFD = _safe_import('modelhub.onnx.ULFD.ULFD', 'ULFD')
MogFace = _safe_import('modelhub.onnx.MogFace.MogFace', 'MogFace')
MTCNN = _safe_import('modelhub.onnx.MTCNN.MTCNN', 'MTCNN')
LightweightFD = _safe_import('modelhub.onnx.LightweightFD.LightweightFD', 'LightweightFD')
FAN = _safe_import('modelhub.onnx.FAN.FAN', 'FAN')
InsightFace3D68 = _safe_import('modelhub.onnx.InsightFace3D68.InsightFace3D68', 'InsightFace3D68')
OpenSeeFaceLandmark = _safe_import('modelhub.onnx.OpenSeeFace.OpenSeeFace', 'OpenSeeFace')
PFLDLandmark = _safe_import('modelhub.onnx.PFLD.PFLD', 'PFLD')
MobileFaceNetLandmark = _safe_import('modelhub.onnx.MobileFaceNet.MobileFaceNet', 'MobileFaceNet')
YoloV11nFace = _safe_import('modelhub.onnx.YoloV11nFace.YoloV11nFace', 'YoloV11nFace')
from xlib.onnxruntime import get_cpu_device_info, get_available_devices_info
from facelib.LandmarksProcessor import get_transform_mat, get_canonical_68
import facelib

# 导入多语言支持
from strings import S


def save_dfljpg(filepath, img, meta_dict):
    """
    保存为 DFL 兼容 JPG（含 APP15 元数据 chunk）。
    零额外内存复制：分段写入避免拼接整个 JPEG 字节串。
    """
    # 编码为 JPEG（纯内存，无磁盘 I/O）
    ret, enc = cv2.imencode('.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
    if not ret:
        raise RuntimeError(f"JPEG 编码失败: {filepath}")
    data = enc.tobytes()

    # 准备 APP15 数据（pickle 序列化）
    # 提取时 metadata 已经是纯 Python 类型（caller 调用了 .tolist()），
    # 但兜底转换 numpy 类型以防万一
    dict_data = {k: v for k, v in meta_dict.items() if v is not None}
    # 快速路径：只有值包含 ndarray 才需要深入转换
    has_numpy = any(isinstance(v, np.ndarray) for v in dict_data.values())
    if has_numpy:
        def _unumpy(obj):
            if isinstance(obj, np.ndarray):
                if obj.dtype.kind in ('u', 'b'):
                    return bytes(obj)
                return obj.tolist()
            if isinstance(obj, (np.floating, np.integer, np.bool_)):
                return obj.item()
            if isinstance(obj, dict):
                return {k: _unumpy(v) for k, v in obj.items()}
            if isinstance(obj, (list, tuple)):
                return [_unumpy(i) for i in obj]
            return obj
        dict_data = _unumpy(dict_data)

    pickle_data = pickle.dumps(dict_data, protocol=2)
    pickle_data = pickle_data.replace(b'numpy._core', b'numpy.core')

    # 构建 APP15 chunk
    app15 = struct.pack('BB', 0xFF, 0xEF)
    app15 += struct.pack('>H', len(pickle_data) + 2)
    app15 += pickle_data

    # 分段写入：SOI~DHT | APP15 | SOS~EOI
    # 避免 data[:sos] + app15 + data[sos:] 的大块 bytes 拼接
    sos = data.find(b'\xff\xda')
    with open(filepath, 'wb') as f:
        if sos > 0:
            f.write(data[:sos])
            f.write(app15)
            f.write(data[sos:])
        else:
            f.write(data)
            f.write(app15)


class DetectorFactory:
    """人脸检测器工厂类"""
    
    DETECTORS = {k: v for k, v in {
        'BlazeFace': BlazeFace,
        'CenterFace': CenterFace,
        'FastFaceAlign': FastFaceAlign,
        'RetinaFace_10g': RetinaFace,
        'RetinaFace_500m': RetinaFace,
        'S3FD': S3FD,
        'YoloV5Face': YoloV5Face,
        'YoloV8Face': YoloV8Face,
        'DamoFD': DamoFD,
        'TinyMog': TinyMog,
        'ULFD': ULFD,
        'MogFace': MogFace,
        'MTCNN': MTCNN,
        'LightweightFD': LightweightFD,
        'YoloV11nFace': YoloV11nFace,
    }.items() if v is not None}

    # RetinaFace 不同参数量的模型映射
    RETINAFACE_MODELS = {
        'RetinaFace_10g': 'det_10g',
        'RetinaFace_500m': 'det_500m',
    }

    @classmethod
    def create_detector(cls, detector_name: str, device_info):
        """Create face detector instance"""
        detector_class = cls.DETECTORS.get(detector_name)
        if detector_class is None:
            raise ValueError(f"Unsupported detector: {detector_name}")

        try:
            # 对 RetinaFace 变体传入对应的 model_name 参数
            if detector_name in cls.RETINAFACE_MODELS:
                detector = detector_class(device_info, model_name=cls.RETINAFACE_MODELS[detector_name])
            else:
                detector = detector_class(device_info)
            print(S('DETECTOR_LOADED', detector_name))
            return detector
        except Exception as e:
            print(S('LOAD_DETECTOR_FAILED', detector_name, e))
            raise


class LandmarkFactory:
    """特征点标记器工厂类"""
    
    LANDMARKS = {k: v for k, v in {
        'insightface-2d106det': InsightFace2D106,
        '2DFAN-4': FAN,       # 2D FAN (68 pts, 256x256)
        '3DFAN-4': FAN,       # 3D FAN (68 pts, 256x256)
        'insightface-3d68': InsightFace3D68,  # 1k3d68 (3309 pts)
        'Google-mediapipe': FaceMesh,
        'OpenSeeFace': OpenSeeFaceLandmark,
        'PFLD': PFLDLandmark,
        'MobileFaceNet': MobileFaceNetLandmark,
    }.items() if v is not None}
    
    @classmethod
    def create_landmarker(cls, landmark_name: str, device_info):
        """创建特征点标记器实例"""
        landmark_class = cls.LANDMARKS.get(landmark_name)
        if landmark_class is None:
            raise ValueError(f"不支持的特征点标记器: {landmark_name}")

        try:
            if landmark_name == '3DFAN-4':
                landmarker = landmark_class(device_info, landmarks_3D=True)
            elif landmark_name == '2DFAN-4':
                landmarker = landmark_class(device_info, landmarks_3D=False)
            else:
                landmarker = landmark_class(device_info)
            print(S('LANDMARKER_LOADED', landmark_name))
            return landmarker
        except Exception as e:
            print(S('LOAD_LANDMARKER_FAILED', landmark_name, e))
            raise


def detect_faces_multi_angle(detector, image: np.ndarray, angles: List[int] = [0],
                             input_mode: str = 'one_stage', resize_mode: str = 'letterbox', input_size: int = 640) -> List[Tuple[int, int, int, int]]:
    """
    Detect faces from multiple rotation angles and merge results

    Args:
        detector: Face detector instance
        image: Input image (BGR)
        angles: List of rotation angles in degrees (clockwise): [0, 90, 180, 270]
        input_mode: Detection input preprocessing ('letterbox' | 'warp' | 'mix')
        input_size: Square input size for letterbox/warp modes

    Returns:
        List of face rects with angle info: [(angle, l, t, r, b), ...]
    """
    all_detections = []
    h, w = image.shape[:2]
    
    for angle in angles:
        if angle == 0:
            rotated_img = image
        else:
            # Rotate image clockwise
            if angle == 90:
                rotated_img = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
            elif angle == 180:
                rotated_img = cv2.rotate(image, cv2.ROTATE_180)
            elif angle == 270:
                rotated_img = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
            else:
                print(f"WARNING: Unsupported angle {angle}, skipping")
                continue
        
        # Detect faces on rotated image
        try:
            # YoloV8Face doesn't support fixed_window parameter
            if isinstance(detector, YoloV8Face):
                results = detector.extract(rotated_img, input_mode=input_mode, resize_mode=resize_mode, input_size=input_size)
            else:
                results = detector.extract(rotated_img, fixed_window=0,
                                           input_mode=input_mode, resize_mode=resize_mode, input_size=input_size)
            if not results or len(results) == 0:
                continue
            
            faces = results[0] if isinstance(results[0], list) else results
            
            # Convert coordinates back to original image space
            for face_rect in faces:
                l, t, r, b = face_rect[:4]
                
                # Rotate coordinates back to original orientation
                if angle == 90:
                    # Rotated 90 CW: new_w=h, new_h=w
                    # Original: (x, y) -> Rotated: (y, new_w-x)
                    # Back: (x', y') -> (new_h-y', x')
                    orig_l = t
                    orig_t = w - r
                    orig_r = b
                    orig_b = w - l
                elif angle == 180:
                    # Rotated 180: same dimensions
                    # Back: (x', y') -> (w-x', h-y')
                    orig_l = w - r
                    orig_t = h - b
                    orig_r = w - l
                    orig_b = h - t
                elif angle == 270:
                    # Rotated 270 CW (90 CCW): new_w=h, new_h=w
                    # Back: (x', y') -> (y', new_w-x')
                    orig_l = h - b
                    orig_t = l
                    orig_r = h - t
                    orig_b = r
                else:
                    orig_l, orig_t, orig_r, orig_b = l, t, r, b
                
                # Ensure coordinates are valid
                orig_l = max(0, int(orig_l))
                orig_t = max(0, int(orig_t))
                orig_r = min(w, int(orig_r))
                orig_b = min(h, int(orig_b))
                
                if orig_r > orig_l and orig_b > orig_t:
                    all_detections.append((angle, orig_l, orig_t, orig_r, orig_b))
        except Exception as e:
            print(f"WARNING: Detection failed at angle {angle}: {e}")
    
    # Remove duplicate detections using IoU
    unique_detections = remove_duplicate_detections(all_detections, iou_threshold=0.5)
    
    return unique_detections


def remove_duplicate_detections(detections: List[Tuple[int, int, int, int, int]], 
                                iou_threshold: float = 0.5) -> List[Tuple[int, int, int, int, int]]:
    """
    Remove duplicate face detections based on IoU (Intersection over Union)
    
    Args:
        detections: List of (angle, l, t, r, b)
        iou_threshold: IoU threshold for considering duplicates
        
    Returns:
        Filtered list of unique detections
    """
    if not detections:
        return []
    
    def calculate_iou(box1, box2):
        """Calculate IoU between two boxes (angle, l, t, r, b)"""
        _, l1, t1, r1, b1 = box1
        _, l2, t2, r2, b2 = box2
        
        # Calculate intersection
        inter_l = max(l1, l2)
        inter_t = max(t1, t2)
        inter_r = min(r1, r2)
        inter_b = min(b1, b2)
        
        if inter_r <= inter_l or inter_b <= inter_t:
            return 0.0
        
        inter_area = (inter_r - inter_l) * (inter_b - inter_t)
        
        # Calculate union
        area1 = (r1 - l1) * (b1 - t1)
        area2 = (r2 - l2) * (b2 - t2)
        union_area = area1 + area2 - inter_area
        
        if union_area == 0:
            return 0.0
        
        return inter_area / union_area
    
    # Sort by confidence (use box area as proxy, larger boxes first)
    sorted_detections = sorted(detections, key=lambda x: (x[3]-x[1])*(x[4]-x[2]), reverse=True)
    
    keep = []
    for detection in sorted_detections:
        is_duplicate = False
        for kept in keep:
            iou = calculate_iou(detection, kept)
            if iou > iou_threshold:
                is_duplicate = True
                break
        
        if not is_duplicate:
            keep.append(detection)
    
    return keep


def detect_and_align_on_resized(detector, landmarker, image: np.ndarray, fixed_window: int = 0,
                                image_size_fixed: Optional[int] = None,
                                face_type_str: str = 'whole_face',
                                detection_angles: List[int] = None,
                                kps_align: bool = True,
                                input_mode: str = 'one_stage', resize_mode: str = 'letterbox', input_size: int = 640) -> List[Dict]:
    """
    Complete face processing pipeline on (possibly resized) image
    Returns normalized results that can be mapped to original image

    Args:
        detector: Face detector
        landmarker: Landmark detector
        image: Input image (will be resized if needed)
        fixed_window: Pre-resize width (0 = no resize)
        image_size_fixed: Fixed output size for alignment
        face_type_str: Face type string
        detection_angles: List of angles for multi-angle detection [0, 90, 180, 270]
        input_mode: Detection input preprocessing ('letterbox' | 'warp' | 'mix')
        input_size: Square input size for letterbox/warp modes

    Returns:
        List of dicts with normalized coordinates and transformation matrices
    """
    h_orig, w_orig = image.shape[:2]
    scale_factor = 1.0
    working_image = image
    
    # Step 1: Resize if needed
    # One-Stage 模式下检测器会把整图缩放到 input_size，预缩放无意义（4K 也会被 letterbox），跳过
    if input_mode != 'one_stage' and fixed_window > 0 and w_orig > fixed_window:
        scale_factor = w_orig / fixed_window
        new_h = int(h_orig / scale_factor)
        new_w = fixed_window
        working_image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    # Step 2: Detect faces on working image (with multi-angle support)
    if detection_angles is None:
        detection_angles = [0]  # Default: only 0 degree

    if len(detection_angles) == 1 and detection_angles[0] == 0:
        # Single angle detection (fast path)
        # Use extract_with_kps when kps_align enabled and detector supports it
        if kps_align and isinstance(detector, (RetinaFace, DamoFD, TinyMog)):
            raw = detector.extract_with_kps(working_image,
                                            input_mode=input_mode, resize_mode=resize_mode, input_size=input_size)
            if not raw:
                return []
            # raw is [(box, kps_or_none), ...]
            faces = [(0, *item[0][:4]) for item in raw]
            kps_list = [item[1] for item in raw]
        else:
            # Original path — no keypoints
            _yolo_retina = tuple(t for t in (YoloV8Face, RetinaFace) if t is not None)
            _fixed_win = tuple(t for t in (BlazeFace, YoloV5Face, CenterFace, S3FD) if t is not None)
            if _yolo_retina and isinstance(detector, _yolo_retina):
                results = detector.extract(working_image,
                                           input_mode=input_mode, resize_mode=resize_mode, input_size=input_size)
            elif _fixed_win and isinstance(detector, _fixed_win):
                results = detector.extract(working_image, fixed_window=0,
                                           input_mode=input_mode, resize_mode=resize_mode, input_size=input_size)
            else:
                results = detector.extract(working_image,
                                           input_mode=input_mode, resize_mode=resize_mode, input_size=input_size)
            if not results or len(results) == 0:
                return []
            faces_raw = results[0] if isinstance(results[0], list) else results
            faces = [(0, *face[:4]) for face in faces_raw]
            kps_list = [None] * len(faces)
    else:
        # Multi-angle detection — always uses original extract (no kps support)
        faces_with_angle = detect_faces_multi_angle(detector, working_image, detection_angles,
                                                    input_mode=input_mode, resize_mode=resize_mode, input_size=input_size)
        if not faces_with_angle:
            return []
        faces = faces_with_angle
        kps_list = [None] * len(faces)
    
    h_work, w_work = working_image.shape[:2]
    
    # Step 3: Extract landmarks on working image
    # Note: faces_with_angle contains coordinates in working_image space,
    # but we need to rotate the image back to the detection angle to extract landmarks correctly
    landmarks_list = []
    for face_idx, (face_info, kps) in enumerate(zip(faces, kps_list)):
        if len(face_info) == 5:
            angle, l, t, r, b = face_info
        else:
            angle, l, t, r, b = 0, *face_info[:4]

        if angle == 0:
            # Compute pre-rotation angle from detector keypoints (if available)
            rot_angle = 0.0
            if kps is not None and len(kps) >= 5:
                eye_cx = (kps[0][0] + kps[1][0]) / 2.0
                eye_cy = (kps[0][1] + kps[1][1]) / 2.0
                mc_x = (kps[3][0] + kps[4][0]) / 2.0
                mc_y = (kps[3][1] + kps[4][1]) / 2.0
                rot_angle = math.degrees(math.atan2(eye_cx - mc_x, -(eye_cy - mc_y)))

            if abs(rot_angle) > 30.0:
                # Pre-rotate face crop before landmark extraction
                margin = round((r - l) * 0.2)
                l_crop = max(0, int(l) - margin)
                t_crop = max(0, int(t) - margin)
                r_crop = min(w_work, int(r) + margin)
                b_crop = min(h_work, int(b) + margin)
                face_img = working_image[t_crop:b_crop, l_crop:r_crop]
                if face_img.size == 0:
                    landmarks_list.append(None)
                    continue
                h_f, w_f = face_img.shape[:2]
                center = (w_f // 2, h_f // 2)
                rot_mat = cv2.getRotationMatrix2D(center, rot_angle, 1.0)
                face_img_rot = cv2.warpAffine(face_img, rot_mat, (w_f, h_f),
                                              flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)
                # Extract landmarks from pre-rotated crop directly
                lmks = extract_landmarks(landmarker, face_img_rot, [(0, 0, w_f, h_f)])
                if lmks and lmks[0] is not None:
                    pts = lmks[0].copy()
                    # Reverse-rotate landmarks back
                    rot_inv = cv2.getRotationMatrix2D(center, -rot_angle, 1.0)
                    ones = np.ones((pts.shape[0], 1))
                    pts = (rot_inv @ np.concatenate([pts, ones], axis=1).T).T
                    # Add crop offset to get working-image coordinates
                    pts[:, 0] += l_crop
                    pts[:, 1] += t_crop
                    landmarks_list.append(pts)
                else:
                    landmarks_list.append(None)
            else:
                # Rotation small or no kps — original flow
                face_rect_for_landmark = (int(l), int(t), int(r), int(b))
                lmks = extract_landmarks(landmarker, working_image, [face_rect_for_landmark])
                if lmks and lmks[0] is not None:
                    landmarks_list.append(lmks[0])
                else:
                    landmarks_list.append(None)
        else:
            # Non-zero detection angle — original multi-angle flow (unchanged)
            if angle == 90:
                rotated_working = cv2.rotate(working_image, cv2.ROTATE_90_CLOCKWISE)
                rot_h, rot_w = rotated_working.shape[:2]
                rot_l, rot_t, rot_r, rot_b = t, w_work - r, b, w_work - l
            elif angle == 180:
                rotated_working = cv2.rotate(working_image, cv2.ROTATE_180)
                rot_h, rot_w = h_work, w_work
                rot_l, rot_t, rot_r, rot_b = w_work - r, h_work - b, w_work - l, h_work - t
            elif angle == 270:
                rotated_working = cv2.rotate(working_image, cv2.ROTATE_90_COUNTERCLOCKWISE)
                rot_h, rot_w = rotated_working.shape[:2]
                rot_l, rot_t, rot_r, rot_b = h_work - b, l, h_work - t, r
            else:
                rotated_working = working_image
                rot_h, rot_w = h_work, w_work
                rot_l, rot_t, rot_r, rot_b = l, t, r, b

            face_rect_for_landmark = (rot_l, rot_t, rot_r, rot_b)
            lmks = extract_landmarks(landmarker, rotated_working, [face_rect_for_landmark])

            if lmks and lmks[0] is not None:
                pts = lmks[0].copy()
                if angle == 90:
                    temp_x = pts[:, 0].copy()
                    pts[:, 0] = pts[:, 1]
                    pts[:, 1] = rot_w - temp_x
                elif angle == 180:
                    pts[:, 0] = rot_w - pts[:, 0]
                    pts[:, 1] = rot_h - pts[:, 1]
                elif angle == 270:
                    temp_x = pts[:, 0].copy()
                    pts[:, 0] = rot_h - pts[:, 1]
                    pts[:, 1] = temp_x
                landmarks_list.append(pts)
            else:
                landmarks_list.append(None)
    
    # Step 4: Process each face - get normalized transformation matrix
    results_list = []
    
    for face_idx, ((angle, l, t, r, b), landmarks) in enumerate(zip(faces, landmarks_list)):
        if landmarks is None:
            continue

        is_facemesh = isinstance(landmarker, FaceMesh)

        # Convert landmarks to standard 68 points (keep full FaceMesh)
        if is_facemesh:
            landmarks_for_align = landmarks  # keep all 468 points
        elif len(landmarks) == 106:
            landmarks_for_align = landmark106to68(landmarks)
        elif len(landmarks) == 468:
            landmarks_for_align = landmarks[:68]
        elif len(landmarks) > 68:
            landmarks_for_align = landmarks[:68]
        else:
            landmarks_for_align = landmarks
        
        # Calculate output size based on landmarks in original image
        if is_facemesh:
            out_size = 256  # FaceMesh: fixed output size
        elif image_size_fixed is not None and image_size_fixed > 0:
            out_size = image_size_fixed
        else:
            # Map landmarks to original image coordinates
            landmarks_orig_temp = landmarks_for_align.copy()
            landmarks_orig_temp[:, 0] *= scale_factor
            landmarks_orig_temp[:, 1] *= scale_factor
            
            # Calculate what size get_transform_mat will extract from original image
            # We need to replicate the logic inside get_transform_mat to predict the extracted region size
            import numpy.linalg as npla
            
            # Use same landmarks subset as get_transform_mat (points 17:49 and 54:55)
            lm_subset = np.concatenate([landmarks_orig_temp[17:49], landmarks_orig_temp[54:55]])
            
            # Estimate transform to unit space
            from facelib.LandmarksProcessor import umeyama, transform_points, landmarks_2D_new
            mat_unit = umeyama(lm_subset, landmarks_2D_new, True)[0:2]
            
            # Get corner points in original image space
            g_p = transform_points(np.float32([(0,0),(1,0),(1,1),(0,1),(0.5,0.5)]), mat_unit, True)
            
            # Calculate diagonal length
            diag_vec = g_p[2] - g_p[0]
            diag_len = npla.norm(diag_vec)
            
            # Get padding based on face type
            face_type_map = {
                'half_face': 0.0,
                'midfull_face': 0.0675,
                'full_face': 0.2109375,
                'whole_face': 0.40,
                'head': 0.70
            }
            padding = face_type_map.get(face_type_str, 0.40)  # Default to whole_face
            
            # Calculate mod (half-diagonal of the extracted square)
            mod = diag_len * (padding * np.sqrt(2.0) + 0.5)
            
            # The extracted square has diagonal = 2 * mod
            # So side length = (2 * mod) / sqrt(2) = mod * sqrt(2)
            extracted_size = mod * np.sqrt(2.0)
            
            # Use the predicted extracted size directly as output size
            out_size = int(extracted_size)
            out_size = (out_size // 2) * 2  # Ensure even number
        
        try:
            # Convert face_type string to FaceType enum
            face_type_enum_map = {
                'half_face': facelib.FaceType.HALF,
                'midfull_face': facelib.FaceType.MID_FULL,
                'full_face': facelib.FaceType.FULL,
                'whole_face': facelib.FaceType.WHOLE_FACE,
                'head': facelib.FaceType.HEAD
            }
            face_type_enum = face_type_enum_map.get(face_type_str, facelib.FaceType.WHOLE_FACE)
            
            # Get transformation matrix (in working image coordinates)
            if is_facemesh:
                mat_work = facemesh_to_align_mat(landmarks, (h_work, w_work))
                out_size = 256
            else:
                mat_work = get_transform_mat(landmarks_for_align, out_size, face_type_enum)
            
            # Normalize the transformation matrix to [0, 1] range relative to working image
            # The affine transform maps from source (working image) to destination (out_size x out_size)
            # We need to store it in a way that can be applied to original image
            
            # Store normalized data
            result = {
                'face_rect': (l, t, r, b),  # In working image coordinates
                'detection_angle': angle,  # Detection angle used
                'landmarks': landmarks_for_align,  # In working image coordinates
                'transform_mat': mat_work,  # Affine matrix for working image
                'out_size': out_size,
                'scale_factor': scale_factor,
                'orig_size': (w_orig, h_orig),
                'work_size': (w_work, h_work),
                'face_type': face_type_str,  # Store face type string for alignment
                'face_type_enum': face_type_enum,  # Store face type enum for metadata
                'is_facemesh': is_facemesh,  # True for FaceMesh landmarks
            }
            results_list.append(result)
            
        except Exception as e:
            print(S('ALIGN_SAVE_FAILED', f"face {face_idx}", 0, e))
    
    return results_list


def apply_alignment_to_original(image: np.ndarray, face_data: Dict) -> Tuple[np.ndarray, np.ndarray, List]:
    """
    Apply stored transformation to original resolution image
    
    Args:
        image: Original resolution image
        face_data: Normalized face data from detect_and_align_on_resized
        
    Returns:
        Tuple of (aligned_face, aligned_landmarks, source_rect_in_original)
    """
    scale_factor = face_data['scale_factor']
    orig_w, orig_h = face_data['orig_size']
    work_w, work_h = face_data['work_size']
    out_size = face_data['out_size']
    
    # Scale landmarks from working image to original image
    landmarks_work = face_data['landmarks']
    landmarks_orig = landmarks_work.copy()
    landmarks_orig[:, 0] *= scale_factor
    landmarks_orig[:, 1] *= scale_factor
    
    # Scale face rect from working image to original image (use round for better precision)
    l, t, r, b = face_data['face_rect']
    face_rect_orig = (
        round(l * scale_factor),
        round(t * scale_factor),
        round(r * scale_factor),
        round(b * scale_factor)
    )
    
    # Recompute transformation matrix using original image landmarks
    if face_data.get('is_facemesh', False):
        mat_orig = facemesh_to_align_mat(landmarks_orig, (orig_h, orig_w))
    else:
        face_type_str = face_data.get('face_type', 'whole_face')
        face_type_enum_map = {
            'half_face': facelib.FaceType.HALF,
            'midfull_face': facelib.FaceType.MID_FULL,
            'full_face': facelib.FaceType.FULL,
            'whole_face': facelib.FaceType.WHOLE_FACE,
            'head': facelib.FaceType.HEAD
        }
        face_type_enum = face_type_enum_map.get(face_type_str, facelib.FaceType.WHOLE_FACE)
        mat_orig = get_transform_mat(landmarks_orig, out_size, face_type_enum)

    # Apply affine transform on ORIGINAL image
    aligned_face = cv2.warpAffine(image, mat_orig, (out_size, out_size),
                                 flags=cv2.INTER_LANCZOS4)
    
    # Transform landmarks to aligned space
    aligned_landmarks = facelib.LandmarksProcessor.transform_points(landmarks_orig, mat_orig)
    
    return aligned_face, aligned_landmarks, face_rect_orig


def extract_landmarks(landmarker, image: np.ndarray, faces: List[Tuple[int, int, int, int]]) -> List[np.ndarray]:
    """
    Extract facial landmarks
    
    Args:
        landmarker: Landmark detector instance
        image: BGR image (original)
        faces: Face bounding box list (MUST be in original image coordinates)
        
    Returns:
        Landmark list, each element is (N, 2) array (original image coordinates)
    """
    landmarks_list = []
    
    for face_idx, face_rect in enumerate(faces):
        # Use float coordinates to preserve precision
        l, t, r, b = face_rect[:4]
        if not isinstance(l, int):
            l, t, r, b = int(l), int(t), int(r), int(b)
        
        # Verify face rect is within image bounds
        h, w = image.shape[:2]
        if l < 0 or t < 0 or r > w or b > h:
            l = max(0, l)
            t = max(0, t)
            r = min(w, r)
            b = min(h, b)
        
        # Crop face region with margin (only for landmark detector input)
        # Use round() instead of int() to reduce bias
        margin = round((r - l) * 0.2)
        l_crop = max(0, l - margin)
        t_crop = max(0, t - margin)
        r_crop = min(w, r + margin)
        b_crop = min(h, b + margin)
        
        face_img = image[t_crop:b_crop, l_crop:r_crop]
        
        if face_img.size == 0:
            print(f"WARNING: Face {face_idx} cropped region is empty")
            landmarks_list.append(None)
            continue
        
        try:
            if isinstance(landmarker, InsightFace2D106):
                # Model returns coordinates relative to face_img
                lmks = landmarker.extract(face_img)
                if lmks is not None and len(lmks) > 0:
                    # Convert to original image coordinates: add crop offset
                    pts = lmks[0].copy()  # (106, 2)
                    pts[:, 0] += l_crop
                    pts[:, 1] += t_crop
                    landmarks_list.append(pts)
                else:
                    landmarks_list.append(None)
                    
            elif isinstance(landmarker, FaceMesh):
                # Model returns 468 points, map to DFL 68 via lookup table
                lmks = landmarker.extract(face_img)
                if lmks is not None and len(lmks) > 0:
                    pts = lmks[0][:, :2].copy()  # (468, 2)
                    # Apply 468→68 mapping
                    from modelhub.onnx.FaceMesh.fm68_mapping import FM68_MAP
                    pts = pts[FM68_MAP]  # (68, 2)
                    pts[:, 0] += l_crop
                    pts[:, 1] += t_crop
                    landmarks_list.append(pts)
                else:
                    landmarks_list.append(None)
            elif type(landmarker).__name__ == 'FAN':
                # FAN returns heatmaps (1, 68, 64, 64), decode to landmarks
                hm = landmarker.extract(face_img)  # (1, 68, 64, 64)
                if hm is not None:
                    hm_sq = hm[0].reshape(68, -1)  # (68, 4096)
                    idx = hm_sq.argmax(axis=1)
                    pts = np.column_stack([idx % 64, idx // 64]).astype(np.float32)
                    H_f, W_f = face_img.shape[:2]
                    pts[:, 0] = pts[:, 0] / 64.0 * W_f + l_crop
                    pts[:, 1] = pts[:, 1] / 64.0 * H_f + t_crop
                    landmarks_list.append(pts)
                else:
                    landmarks_list.append(None)
            else:
                # Generic landmarker: assume extract() returns [(N,2)] list
                lmks = landmarker.extract(face_img)
                if lmks is not None and len(lmks) > 0:
                    pts = lmks[0].copy()
                    pts[:, 0] += l_crop
                    pts[:, 1] += t_crop
                    landmarks_list.append(pts)
                else:
                    landmarks_list.append(None)
        except Exception as e:
            print(S('LANDMARK_EXTRACT_ERROR', e))
            import traceback
            traceback.print_exc()
            landmarks_list.append(None)
    
    return landmarks_list


def sort_faces_by_distance(
    prev_faces: List[Tuple[int, int, int, int]],
    curr_faces: List[Tuple[int, int, int, int]]
) -> List[Tuple[int, int, int, int]]:
    """
    Sort faces based on Euclidean distance to maintain consistency with previous frame
    
    Args:
        prev_faces: Previous frame faces (sorted)
        curr_faces: Current frame faces (unsorted)
        
    Returns:
        Sorted current frame faces
    """
    if not prev_faces or not curr_faces:
        return curr_faces
    
    if len(curr_faces) == 1:
        return curr_faces
    
    # Calculate center point for each face
    def get_center(face):
        l, t, r, b = face[:4]
        return ((l + r) / 2, (t + b) / 2)
    
    prev_centers = [get_center(f) for f in prev_faces]
    curr_centers = [get_center(f) for f in curr_faces]
    
    # Greedy matching: find nearest previous face for each current face
    matched_indices = set()
    sorted_faces = []
    
    for i, curr_center in enumerate(curr_centers):
        best_idx = -1
        best_dist = float('inf')
        
        for j, prev_center in enumerate(prev_centers):
            if j in matched_indices:
                continue
            
            dist = np.sqrt((curr_center[0] - prev_center[0])**2 + 
                          (curr_center[1] - prev_center[1])**2)
            
            if dist < best_dist:
                best_dist = dist
                best_idx = j
        
        if best_idx != -1:
            matched_indices.add(best_idx)
            sorted_faces.append(curr_faces[i])
    
    # Append unmatched faces to the end
    for i, face in enumerate(curr_faces):
        if i not in [list(matched_indices).index(j) if j in matched_indices else -1 
                     for j in range(len(prev_faces))]:
            if face not in sorted_faces:
                sorted_faces.append(face)
    
    return sorted_faces if sorted_faces else curr_faces


def sort_faces_by_distance_for_data(
    prev_faces: List[Tuple[int, int, int, int]],
    curr_faces: List[Tuple[int, int, int, int]]
) -> List[int]:
    """
    Sort indices based on face distance for face_data list sorting
    
    Args:
        prev_faces: Previous frame faces (sorted)
        curr_faces: Current frame faces (unsorted)
        
    Returns:
        List of indices to reorder curr_faces to match prev_faces order
    """
    if not prev_faces or not curr_faces:
        return list(range(len(curr_faces)))
    
    if len(curr_faces) == 1:
        return [0]
    
    # Calculate center point for each face
    def get_center(face):
        l, t, r, b = face[:4]
        return ((l + r) / 2, (t + b) / 2)
    
    prev_centers = [get_center(f) for f in prev_faces]
    curr_centers = [get_center(f) for f in curr_faces]
    
    # Greedy matching: find nearest previous face for each current face
    matched_prev = set()
    sorted_indices = [-1] * len(curr_faces)
    used_curr = set()
    
    # Match each previous face to nearest current face
    for j, prev_center in enumerate(prev_centers):
        best_idx = -1
        best_dist = float('inf')
        
        for i, curr_center in enumerate(curr_centers):
            if i in used_curr:
                continue
            
            dist = np.sqrt((curr_center[0] - prev_center[0])**2 + 
                          (curr_center[1] - prev_center[1])**2)
            
            if dist < best_dist:
                best_dist = dist
                best_idx = i
        
        if best_idx != -1:
            sorted_indices[j] = best_idx
            used_curr.add(best_idx)
    
    # Fill in unmatched positions
    unused_curr = [i for i in range(len(curr_faces)) if i not in used_curr]
    fill_idx = 0
    for j in range(len(sorted_indices)):
        if sorted_indices[j] == -1 and fill_idx < len(unused_curr):
            sorted_indices[j] = unused_curr[fill_idx]
            fill_idx += 1
    
    return sorted_indices


# Global variables for worker processes (initialized once per process)
_worker_detector = None
_worker_landmarker = None
_worker_device_info = None
_worker_detector_name = None
_worker_landmark_name = None
_worker_fixed_window = 0
_worker_kps_align = True


def init_worker_process(detector_name, landmark_name, device_info, fixed_window=0, kps_align=True):
    """
    Initialize worker process with detector and landmarker (called once per process)
    """
    global _worker_detector, _worker_landmarker, _worker_device_info
    global _worker_detector_name, _worker_landmark_name, _worker_fixed_window, _worker_kps_align

    _worker_detector_name = detector_name
    _worker_landmark_name = landmark_name
    _worker_device_info = device_info
    _worker_fixed_window = fixed_window
    _worker_kps_align = kps_align
    
    try:
        _worker_detector = DetectorFactory.create_detector(detector_name, device_info)
        _worker_landmarker = LandmarkFactory.create_landmarker(landmark_name, device_info)
    except Exception as e:
        print(S('INIT_WORKER_FAILED', e))
        import traceback
        traceback.print_exc()


def process_single_frame(args):
    """
    Process single frame (for multiprocessing)
    Args:
        args: tuple (frame_idx, frame, image_size, output_path, input_path_name)
    Returns:
        tuple (frame_idx, saved_count, metadata_dict)
    """
    global _worker_detector, _worker_landmarker, _worker_fixed_window, _worker_kps_align

    frame_idx, frame, image_size, output_path, input_path_name = args[:5]
    kps_align = args[5] if len(args) > 5 else _worker_kps_align

    try:
        # Use pre-initialized detector and landmarker
        if _worker_detector is None or _worker_landmarker is None:
            print(S('WORKER_NOT_INIT', frame_idx))
            return (frame_idx, 0, {})

        # NEW APPROACH: Complete pipeline on resized image, then apply to original
        face_data_list = detect_and_align_on_resized(
            _worker_detector, _worker_landmarker, frame,
            _worker_fixed_window, image_size, kps_align=kps_align
        )
        
        if not face_data_list:
            return (frame_idx, 0, {})
        
        # Align and save each face using original resolution image
        saved_count = 0
        metadata_dict = {}
        
        for face_idx, face_data in enumerate(face_data_list):
            try:
                # Apply alignment to ORIGINAL image
                aligned_face, aligned_landmarks, face_rect_orig = apply_alignment_to_original(
                    frame, face_data
                )
                
                out_size = face_data['out_size']
                
                # Generate filename
                filename = f"{frame_idx:05d}_{face_idx}.jpg"
                filepath = output_path / filename
                
                # Store metadata (embedded as APP15 for DFL compat)
                metadata_dict[filename] = {
                    'face_type': facelib.FaceType.toString(facelib.FaceType.WHOLE_FACE),
                    'landmarks': aligned_landmarks.tolist(),
                    'source_landmarks': face_data['landmarks'].tolist(),  # In working image coords
                    'source_rect': face_rect_orig,  # In original image coords
                    'image_to_face_mat': get_transform_mat(
                        face_data['landmarks'] * face_data['scale_factor'], 
                        out_size, 
                        facelib.FaceType.WHOLE_FACE
                    ).tolist(),
                    'source_filename': str(input_path_name)
                }

                # Save DFL-compatible JPG with embedded metadata
                save_dfljpg(str(filepath), aligned_face, metadata_dict[filename])

                saved_count += 1
            except Exception as e:
                print(S('ALIGN_SAVE_FAILED', f"frame {frame_idx}", face_idx, e))
        
        return (frame_idx, saved_count, metadata_dict)
    except Exception as e:
        print(S('PROCESS_IMAGE_FAILED', f"frame {frame_idx}", e))
        import traceback
        traceback.print_exc()
        return (frame_idx, 0, {})


def visualize_extraction_stages(original_image: np.ndarray, face_data_list: List[Dict], 
                               debug_dir: Path, img_idx: int, fixed_window: int = 0):
    """
    Visualize all stages of face extraction for debugging
    
    Args:
        original_image: Original input image
        face_data_list: List of face data from detect_and_align_on_resized
        debug_dir: Directory to save debug images
        img_idx: Image index
        fixed_window: Pre-resize width used
    """
    import os
    debug_dir.mkdir(parents=True, exist_ok=True)
    
    h_orig, w_orig = original_image.shape[:2]
    
    # Stage 1: Show pre-resized image with detection boxes
    if fixed_window > 0 and w_orig > fixed_window:
        scale_factor = w_orig / fixed_window
        new_h = int(h_orig / scale_factor)
        resized_img = cv2.resize(original_image, (fixed_window, new_h), interpolation=cv2.INTER_AREA)
    else:
        resized_img = original_image.copy()
        scale_factor = 1.0
    
    # Draw detection boxes on resized image
    vis_resized = resized_img.copy()
    for face_data in face_data_list:
        l, t, r, b = face_data['face_rect']
        angle = face_data.get('detection_angle', 0)
        
        # Color based on detection angle
        color_map = {0: (0, 255, 0), 90: (255, 0, 0), 180: (0, 0, 255), 270: (255, 255, 0)}
        color = color_map.get(angle, (255, 255, 255))
        
        cv2.rectangle(vis_resized, (l, t), (r, b), color, 2)
        cv2.putText(vis_resized, f'{angle}°', (l, t-5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    stage1_path = debug_dir / f"{img_idx:05d}_stage1_detection.png"
    cv2.imwrite(str(stage1_path), vis_resized)
    print(f"[DEBUG] Saved stage 1 (detection): {stage1_path}")
    
    # Stage 2: Show landmarks on resized image
    vis_landmarks = resized_img.copy()
    for face_data in face_data_list:
        landmarks = face_data['landmarks']
        l, t, r, b = face_data['face_rect']
        angle = face_data.get('detection_angle', 0)
        
        color_map = {0: (0, 255, 0), 90: (255, 0, 0), 180: (0, 0, 255), 270: (255, 255, 0)}
        color = color_map.get(angle, (255, 255, 255))
        
        # Draw landmarks
        for i, (x, y) in enumerate(landmarks):
            cv2.circle(vis_landmarks, (int(x), int(y)), 2, color, -1)
            if i % 10 == 0:  # Label every 10th point
                cv2.putText(vis_landmarks, str(i), (int(x)+3, int(y)), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)
        
        # Draw face box
        cv2.rectangle(vis_landmarks, (l, t), (r, b), color, 2)
    
    stage2_path = debug_dir / f"{img_idx:05d}_stage2_landmarks.png"
    cv2.imwrite(str(stage2_path), vis_landmarks)
    print(f"[DEBUG] Saved stage 2 (landmarks): {stage2_path}")
    
    # Stage 3: Show aligned faces and transformation on original image
    vis_original = original_image.copy()
    for face_idx, face_data in enumerate(face_data_list):
        angle = face_data.get('detection_angle', 0)
        color_map = {0: (0, 255, 0), 90: (255, 0, 0), 180: (0, 0, 255), 270: (255, 255, 0)}
        color = color_map.get(angle, (255, 255, 255))
        
        # Get source rect in original image coordinates
        scale_factor_fd = face_data['scale_factor']
        l_work, t_work, r_work, b_work = face_data['face_rect']
        l_orig = round(l_work * scale_factor_fd)
        t_orig = round(t_work * scale_factor_fd)
        r_orig = round(r_work * scale_factor_fd)
        b_orig = round(b_work * scale_factor_fd)
        
        # Draw detection box on original image
        cv2.rectangle(vis_original, (l_orig, t_orig), (r_orig, b_orig), color, 3)
        cv2.putText(vis_original, f'Face {face_idx} ({angle}°)', 
                   (l_orig, t_orig-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # Draw landmarks on original image
        landmarks_work = face_data['landmarks']
        landmarks_orig = landmarks_work.copy()
        landmarks_orig[:, 0] *= scale_factor_fd
        landmarks_orig[:, 1] *= scale_factor_fd
        
        for x, y in landmarks_orig:
            cv2.circle(vis_original, (int(x), int(y)), 3, color, -1)
        
        # Draw the extracted region rectangle
        out_size = face_data['out_size']
        mat_orig = get_transform_mat(landmarks_orig, out_size, 
                                    face_data.get('face_type_enum', facelib.FaceType.WHOLE_FACE))
        
        # Calculate corners of the extracted region
        corners = np.array([[0, 0], [out_size, 0], [out_size, out_size], [0, out_size]], dtype=np.float32)
        inv_mat = cv2.invertAffineTransform(mat_orig)
        orig_corners = cv2.transform(corners.reshape(1, -1, 2), inv_mat).reshape(-1, 2)
        
        # Draw polygon showing extraction area
        pts = orig_corners.astype(np.int32)
        cv2.polylines(vis_original, [pts], True, (255, 255, 255), 2)
        
        # Save individual aligned face
        aligned_face, aligned_landmarks, _ = apply_alignment_to_original(original_image, face_data)
        aligned_path = debug_dir / f"{img_idx:05d}_face_{face_idx}_aligned.png"
        cv2.imwrite(str(aligned_path), aligned_face)
        
        # Draw landmarks on aligned face
        vis_aligned = aligned_face.copy()
        for x, y in aligned_landmarks:
            cv2.circle(vis_aligned, (int(x), int(y)), 2, (0, 0, 255), -1)
        aligned_lm_path = debug_dir / f"{img_idx:05d}_face_{face_idx}_aligned_lm.png"
        cv2.imwrite(str(aligned_lm_path), vis_aligned)
    
    stage3_path = debug_dir / f"{img_idx:05d}_stage3_original_with_boxes.png"
    cv2.imwrite(str(stage3_path), vis_original)
    print(f"[DEBUG] Saved stage 3 (original with boxes): {stage3_path}")
    print(f"[DEBUG] Total faces detected: {len(face_data_list)}")
    print(f"[DEBUG] Debug images saved to: {debug_dir}\n")


def landmark106to68(pt106: np.ndarray) -> np.ndarray:#这个我能确保是包对的
    """
    Convert 106 landmarks to standard 68 landmarks
    """
    if len(pt106) != 106:
        return pt106[:68] if len(pt106) >= 68 else pt106
    
    landmark106to68 = [
        1, 10, 12, 14, 16, 3, 5, 7, 0,  # Chin 9 points
        23, 21, 19, 32, 30, 28, 26, 17,  # Eyebrows 8 points (should be 17 total for chin+brows)
        43, 48, 49, 51, 50,  # Left eyebrow 5 points
        102, 103, 104, 105, 101,  # Right eyebrow 5 points
        72, 73, 74, 86, 78, 79, 80, 85, 84,  # Nose 9 points
        35, 41, 42, 39, 37, 36,  # Left eye 6 points
        89, 95, 96, 93, 91, 90,  # Right eye 6 points
        52, 64, 63, 71, 67, 68, 61, 58, 59, 53, 56, 55, 65, 66, 62, 70, 69, 57, 60, 54  # Mouth 20 points
    ]
    
    pts68 = np.array([pt106[i] for i in landmark106to68])
    return pts68



# ── FaceMesh 468 canonical landmarks（用于 umeyama 仿射对齐）──
_FACEMESH_CANONICAL = np.float32([
  [0.4999769926, 0.6525340080], [0.5000259876, 0.5474870205], [0.4999740124, 0.6023719907],
  [0.4821130037, 0.4719790220], [0.5001509786, 0.5271559954], [0.4999099970, 0.4982529879],
  [0.4995230138, 0.4010620117], [0.2897120118, 0.3807640076], [0.4999549985, 0.3123980165],
  [0.4999870062, 0.2699189782], [0.5000230074, 0.1070500016], [0.5000230074, 0.6662340164],
  [0.5000159740, 0.6792240143], [0.5000230074, 0.6923480034], [0.4999769926, 0.6952779889],
  [0.4999769926, 0.7059339881], [0.4999769926, 0.7193850279], [0.4999769926, 0.7370190024],
  [0.4999679923, 0.7813709974], [0.4998160005, 0.5629810095], [0.4737730026, 0.5739099979],
  [0.1049069986, 0.2541409731], [0.3659299910, 0.4095759988], [0.3387579918, 0.4130250216],
  [0.3111200035, 0.4094600081], [0.2746579945, 0.3891310096], [0.3933619857, 0.4037060142],
  [0.3452340066, 0.3440110087], [0.3700940013, 0.3460760117], [0.3193219900, 0.3472650051],
  [0.2979030013, 0.3535910249], [0.2477920055, 0.4108099937], [0.3968890011, 0.8427550197],
  [0.2800979912, 0.3755999804], [0.1063100025, 0.3999559879], [0.2099249959, 0.3913530111],
  [0.3558079898, 0.5344060063], [0.4717510045, 0.6504039764], [0.4741550088, 0.6801919937],
  [0.4397850037, 0.6572290063], [0.4146170020, 0.6665409803], [0.4503740072, 0.6808609962],
  [0.4287709892, 0.6826909781], [0.3749710023, 0.7278050184], [0.4867169857, 0.5476289988],
  [0.4853009880, 0.5273950100], [0.2577649951, 0.3144900203], [0.4012230039, 0.4551720023],
  [0.4298189878, 0.5486149788], [0.4213519990, 0.5337409973], [0.2768959999, 0.5320569873],
  [0.4833700061, 0.4995869994], [0.3372119963, 0.2828829885], [0.2963919938, 0.2932429910],
  [0.1692949980, 0.1938139796], [0.4475800097, 0.3026099801], [0.3923900127, 0.3538879752],
  [0.3544900119, 0.6967840195], [0.0673049986, 0.7301050425], [0.4427390099, 0.5728260279],
  [0.4570980072, 0.5847920179], [0.3819740117, 0.6947109699], [0.3923889995, 0.6942030191],
  [0.2770760059, 0.2719320059], [0.4225519896, 0.5632330179], [0.3859190047, 0.2813640237],
  [0.3831030130, 0.2558400035], [0.3314310014, 0.1197140217], [0.2299239933, 0.2320029736],
  [0.3645009995, 0.1891139746], [0.2296220064, 0.2995409966], [0.1732870042, 0.2787479758],
  [0.4728789926, 0.6661980152], [0.4468280077, 0.6685270071], [0.4227620065, 0.6738899946],
  [0.4453079998, 0.5800659657], [0.3881030083, 0.6939610243], [0.4030390084, 0.7065399885],
  [0.4036290050, 0.6939530373], [0.4600419998, 0.5571390390], [0.4311580062, 0.6923660040],
  [0.4521819949, 0.6923660040], [0.4753870070, 0.6923660040], [0.4658280015, 0.7791900039],
  [0.4723289907, 0.7362259626], [0.4730870128, 0.7178570032], [0.4731220007, 0.7046259642],
  [0.4730330110, 0.6952779889], [0.4279420078, 0.6952779889], [0.4264790118, 0.7035399675],
  [0.4231620133, 0.7118459940], [0.4183090031, 0.7200629711], [0.3900949955, 0.6395729780],
  [0.0139539996, 0.5600340366], [0.4999139905, 0.5801470280], [0.4131999910, 0.6953999996],
  [0.4096260071, 0.7018229961], [0.4680800140, 0.6015349627], [0.4227289855, 0.5859850049],
  [0.4630799890, 0.5937839746], [0.3721199930, 0.4734140038], [0.3345620036, 0.4960730076],
  [0.4116710126, 0.5469650030], [0.2421759963, 0.1476759911], [0.2907769978, 0.2014459968],
  [0.3273380101, 0.2565270066], [0.3995099962, 0.7489210367], [0.4417279959, 0.2616760135],
  [0.4297649860, 0.1878340244], [0.4121980071, 0.1089010239], [0.2889550030, 0.3989520073],
  [0.2189369947, 0.4354109764], [0.4127820134, 0.3989700079], [0.2571350038, 0.3554400206],
  [0.4276849926, 0.4379609823], [0.4483399987, 0.5369360447], [0.1785600036, 0.4575539827],
  [0.2473080009, 0.4571939707], [0.2862670124, 0.4676749706], [0.3328279853, 0.4607120156],
  [0.3687559962, 0.4472069740], [0.3989639878, 0.4326549768], [0.4764100015, 0.4058060050],
  [0.1892410070, 0.5239239931], [0.2289620042, 0.3489509821], [0.4907259941, 0.5624009967],
  [0.4046700001, 0.4851329923], [0.0194690004, 0.4015640020], [0.4262430072, 0.4204310179],
  [0.3969930112, 0.5487970114], [0.2664699852, 0.3769770265], [0.4391210079, 0.5189579725],
  [0.0323139988, 0.6443569660], [0.4190540016, 0.3871549964], [0.4627830088, 0.5057469606],
  [0.2389789969, 0.7797449827], [0.1982209980, 0.8319380283], [0.1075500026, 0.5407550335],
  [0.1836100072, 0.7402570248], [0.1344099939, 0.3336830139], [0.3857640028, 0.8831539750],
  [0.4909670055, 0.5793780088], [0.3823849857, 0.5085729957], [0.1743990034, 0.3976709843],
  [0.3187850118, 0.3962349892], [0.3433640003, 0.4005969763], [0.3961000144, 0.7102169991],
  [0.1878850013, 0.5885379910], [0.4309870005, 0.9440649748], [0.3189930022, 0.8982850313],
  [0.2662479877, 0.8697010279], [0.5000230074, 0.1905760169], [0.4999769926, 0.9544529915],
  [0.3661699891, 0.3988220096], [0.3932070136, 0.3955370188], [0.4103730023, 0.3910800219],
  [0.1949930042, 0.3421019912], [0.3886649907, 0.3622840047], [0.3659619987, 0.3559709787],
  [0.3433640003, 0.3553569913], [0.3187850118, 0.3583400249], [0.3014149964, 0.3631560206],
  [0.0581329986, 0.3190760016], [0.3014149964, 0.3874490261], [0.4999879897, 0.6184340119],
  [0.4158380032, 0.6241959929], [0.4456819892, 0.5660769939], [0.4658440053, 0.6206409931],
  [0.4999229908, 0.3515239954], [0.2887189984, 0.8199459910], [0.3352789879, 0.8528199792],
  [0.4405120015, 0.9024189711], [0.1282940060, 0.7919409871], [0.4087719917, 0.3738939762],
  [0.4556069970, 0.4518010020], [0.4998770058, 0.9089900255], [0.3754369915, 0.9241920114],
  [0.1142100021, 0.6150220037], [0.4486620128, 0.6952779889], [0.4480200112, 0.7046320438],
  [0.4471119940, 0.7158080339], [0.4448319972, 0.7307940125], [0.4300119877, 0.7668089867],
  [0.4067870080, 0.6856729984], [0.4007380009, 0.6810690165], [0.3923999965, 0.6777030230],
  [0.3678559959, 0.6639189720], [0.2479230016, 0.6013330221], [0.4527699947, 0.4208499789],
  [0.4363920093, 0.3598870039], [0.4161640108, 0.3687139750], [0.4133859873, 0.6923660040],
  [0.2280180007, 0.6835719943], [0.4682680070, 0.3526710272], [0.4113619924, 0.8043270111],
  [0.4999890029, 0.4698250294], [0.4791539907, 0.4426540136], [0.4999740124, 0.4396370053],
  [0.4321120083, 0.4935889840], [0.4998860061, 0.8669170141], [0.4999130070, 0.8217290044],
  [0.4565489888, 0.8192009926], [0.3445490003, 0.7454389930], [0.3789089918, 0.5740100145],
  [0.3742929995, 0.7801849842], [0.3196879923, 0.5707379580], [0.3571549952, 0.6042699814],
  [0.2952840030, 0.6215809584], [0.4477500021, 0.8624770045], [0.4109860063, 0.5087230206],
  [0.3139509857, 0.7753080130], [0.3541280031, 0.8125529885], [0.3245480061, 0.7039929628],
  [0.1890960038, 0.6462999582], [0.2797769904, 0.7146580219], [0.1338230073, 0.6827009916],
  [0.3367680013, 0.6447330117], [0.4298839867, 0.4665219784], [0.4555279911, 0.5486229658],
  [0.4371140003, 0.5588960052], [0.4672879875, 0.5299249887], [0.4147120118, 0.3352199793],
  [0.3770459890, 0.3227779865], [0.3441079855, 0.3201509714], [0.3128759861, 0.3223320246],
  [0.2835260034, 0.3331900239], [0.2412459999, 0.3827859759], [0.1029860005, 0.4687629938],
  [0.2676120102, 0.4245600104], [0.2978790104, 0.4331759810], [0.3334339857, 0.4338780046],
  [0.3664270043, 0.4261159897], [0.3960120082, 0.4166960120], [0.4201210141, 0.4102280140],
  [0.0075610001, 0.4807770252], [0.4329490066, 0.5695179701], [0.4586389959, 0.4790890217],
  [0.4734660089, 0.5457440019], [0.4760879874, 0.5638300180], [0.4684720039, 0.5550569892],
  [0.4339909852, 0.5823619962], [0.4835180044, 0.5629839897], [0.4824829996, 0.5778490305],
  [0.4264500141, 0.3897989988], [0.4389989972, 0.3964949846], [0.4500670135, 0.4004340172],
  [0.2897120118, 0.3682529926], [0.2766700089, 0.3633729815], [0.5178620219, 0.4719480276],
  [0.7102879882, 0.3807640076], [0.5262269974, 0.5739099979], [0.8950930238, 0.2541409731],
  [0.6340699792, 0.4095759988], [0.6612420082, 0.4130250216], [0.6888800263, 0.4094600081],
  [0.7253419757, 0.3891310096], [0.6066300273, 0.4037050009], [0.6547660232, 0.3440110087],
  [0.6299059987, 0.3460760117], [0.6806780100, 0.3472650051], [0.7020969987, 0.3535910249],
  [0.7522119880, 0.4108049870], [0.6029180288, 0.8428629637], [0.7199019790, 0.3755999804],
  [0.8936929703, 0.3999599814], [0.7900819778, 0.3913540244], [0.6439980268, 0.5344879627],
  [0.5282490253, 0.6504039764], [0.5258499980, 0.6801910400], [0.5602149963, 0.6572290063],
  [0.5853840113, 0.6665409803], [0.5496259928, 0.6808609962], [0.5712280273, 0.6826919913],
  [0.6248520017, 0.7280989885], [0.5130500197, 0.5472819805], [0.5150970221, 0.5272519588],
  [0.7422469854, 0.3145070076], [0.5986310244, 0.4549790025], [0.5703380108, 0.5485750437],
  [0.5786319971, 0.5336229801], [0.7230870128, 0.5320540071], [0.5164459944, 0.4996389747],
  [0.6628010273, 0.2829179764], [0.7036240101, 0.2932710052], [0.8307049870, 0.1938139796],
  [0.5523859859, 0.3025680184], [0.6076099873, 0.3538879752], [0.6454290152, 0.6967070103],
  [0.9326949716, 0.7301050425], [0.5572609901, 0.5728260279], [0.5429019928, 0.5847920179],
  [0.6180260181, 0.6947109699], [0.6075909734, 0.6942030191], [0.7229430079, 0.2719630003],
  [0.5774139762, 0.5631669760], [0.6140829921, 0.2813869715], [0.6169070005, 0.2558860183],
  [0.6685090065, 0.1199139953], [0.7700920105, 0.2320209742], [0.6355360150, 0.1892489791],
  [0.7703909874, 0.2995560169], [0.8267220259, 0.2787550092], [0.5271210074, 0.6661980152],
  [0.5531719923, 0.6685270071], [0.5772380233, 0.6738899946], [0.5546919703, 0.5800659657],
  [0.6118969917, 0.6939610243], [0.5969610214, 0.7065399885], [0.5963709950, 0.6939530373],
  [0.5399580002, 0.5571390390], [0.5688419938, 0.6923660040], [0.5478180051, 0.6923660040],
  [0.5246130228, 0.6923660040], [0.5340899825, 0.7791410089], [0.5276709795, 0.7362259626],
  [0.5269129872, 0.7178570032], [0.5268779993, 0.7046259642], [0.5269669890, 0.6952779889],
  [0.5720580220, 0.6952779889], [0.5735210180, 0.7035399675], [0.5768380165, 0.7118459940],
  [0.5816910267, 0.7200629711], [0.6099449992, 0.6399099827], [0.9860460162, 0.5600340366],
  [0.5867999792, 0.6953999996], [0.5903720260, 0.7018229961], [0.5319150090, 0.6015369892],
  [0.5772680044, 0.5859349966], [0.5369150043, 0.5937860012], [0.6275429726, 0.4733520150],
  [0.6655859947, 0.4959509969], [0.5883539915, 0.5468620062], [0.7578240037, 0.1476759911],
  [0.7092499733, 0.2015079856], [0.6726840138, 0.2565810084], [0.6004089713, 0.7490049601],
  [0.5582659841, 0.2616720200], [0.5703039765, 0.1878709793], [0.5881659985, 0.1090440154],
  [0.7110450268, 0.3989520073], [0.7810699940, 0.4354050159], [0.5872470140, 0.3989319801],
  [0.7428699732, 0.3554459810], [0.5721560121, 0.4376519918], [0.5518680215, 0.5365700126],
  [0.8214420080, 0.4575560093], [0.7527019978, 0.4571819901], [0.7137569785, 0.4676269889],
  [0.6671130061, 0.4606729746], [0.6311010122, 0.4471539855], [0.6008620262, 0.4324730039],
  [0.5234810114, 0.4056270123], [0.8107479811, 0.5239260197], [0.7710459828, 0.3489590287],
  [0.5091270208, 0.5627180338], [0.5952929854, 0.4850239754], [0.9805309772, 0.4015640020],
  [0.5734999776, 0.4200000167], [0.6029949784, 0.5486879945], [0.7335299850, 0.3769770265],
  [0.5606110096, 0.5190169811], [0.9676859975, 0.6443569660], [0.5809850097, 0.3871600032],
  [0.5377280116, 0.5053850412], [0.7609660029, 0.7797529697], [0.8017789721, 0.8319380283],
  [0.8924409747, 0.5407609940], [0.8163509965, 0.7402600050], [0.8655949831, 0.3336870074],
  [0.6140739918, 0.8832460046], [0.5089529753, 0.5794379711], [0.6179419756, 0.5083160400],
  [0.8256080151, 0.3976749778], [0.6812149882, 0.3962349892], [0.6566359997, 0.4005969763],
  [0.6039000154, 0.7102169991], [0.8120859861, 0.5885390043], [0.5680130124, 0.9445649981],
  [0.6810079813, 0.8982850313], [0.7337520123, 0.8697010279], [0.6338300109, 0.3988220096],
  [0.6067929864, 0.3955370188], [0.5896599889, 0.3910620213], [0.8050159812, 0.3421080112],
  [0.6113349795, 0.3622840047], [0.6340379715, 0.3559709787], [0.6566359997, 0.3553569913],
  [0.6812149882, 0.3583400249], [0.6985849738, 0.3631560206], [0.9418669939, 0.3190760016],
  [0.6985849738, 0.3874490261], [0.5841770172, 0.6241070032], [0.5543180108, 0.5660769939],
  [0.5341539979, 0.6206400394], [0.7112179995, 0.8199750185], [0.6646299958, 0.8528710008],
  [0.5590999722, 0.9026319981], [0.8717060089, 0.7919409871], [0.5912340283, 0.3738939762],
  [0.5443410277, 0.4515839815], [0.6245629787, 0.9241920114], [0.8857700229, 0.6150289774],
  [0.5513380170, 0.6952779889], [0.5519800186, 0.7046320438], [0.5528879762, 0.7158080339],
  [0.5551679730, 0.7307940125], [0.5699440241, 0.7670350075], [0.5932030082, 0.6856759787],
  [0.5992619991, 0.6810690165], [0.6075999737, 0.6777030230], [0.6319379807, 0.6635000110],
  [0.7520329952, 0.6013150215], [0.5472260118, 0.4203950167], [0.5635439754, 0.3598279953],
  [0.5838410258, 0.3687139750], [0.5866140127, 0.6923660040], [0.7719150186, 0.6835780144],
  [0.5315970182, 0.3524829745], [0.5883709788, 0.8044409752], [0.5207970142, 0.4425650239],
  [0.5679849982, 0.4934790134], [0.5432829857, 0.8192549944], [0.6553170085, 0.7455149889],
  [0.6210089922, 0.5740180016], [0.6255599856, 0.7803120017], [0.6801980138, 0.5707190037],
  [0.6427639723, 0.6043379903], [0.7046629786, 0.6215299964], [0.5520120263, 0.8625919819],
  [0.5890719891, 0.5086370111], [0.6859449744, 0.7753570080], [0.6457350254, 0.8126400113],
  [0.6753429770, 0.7039780021], [0.8108580112, 0.6463049650], [0.7201219797, 0.7146669626],
  [0.8661519885, 0.6827049851], [0.6631870270, 0.6445969939], [0.5700820088, 0.4663259983],
  [0.5445619822, 0.5483759642], [0.5627589822, 0.5587849617], [0.5319870114, 0.5301400423],
  [0.5852710009, 0.3351770043], [0.6229529977, 0.3227789998], [0.6558960080, 0.3201630116],
  [0.6871320009, 0.3223459721], [0.7164819837, 0.3332009912], [0.7587569952, 0.3827869892],
  [0.8970130086, 0.4687690139], [0.7323920131, 0.4245470166], [0.7021139860, 0.4331629872],
  [0.6665250063, 0.4338660240], [0.6335049868, 0.4260879755], [0.6038759947, 0.4165869951],
  [0.5796579719, 0.4099450111], [0.9924399853, 0.4807770252], [0.5671920180, 0.5694199800],
  [0.5413659811, 0.4788990021], [0.5265640020, 0.5461180210], [0.5239130259, 0.5638300180],
  [0.5315290093, 0.5550569892], [0.5660359859, 0.5823290348], [0.5163109899, 0.5630539656],
  [0.5174720287, 0.5778770447], [0.5735949874, 0.3898069859], [0.5606979728, 0.3953319788],
  [0.5497559905, 0.3997510076], [0.7102879882, 0.3682529926], [0.7233300209, 0.3633729815]
])

def facemesh_to_align_mat(landmarks_68: np.ndarray, image_size: tuple) -> np.ndarray:
    """Standard 68-point landmarks -> affine transform using 5 keypoints (eyes, nose, mouth).
    Works with both 68-point (after FM68_MAP) and FaceMesh 468-point landmarks.
    """
    pts = landmarks_68[:, :2] if landmarks_68.shape[1] >= 2 else landmarks_68
    if pts.shape[0] == 68:
        # Standard 68-point format
        left_eye = pts[36:42].mean(axis=0)   # avg of left eye contour
        right_eye = pts[42:48].mean(axis=0)  # avg of right eye contour
        nose = pts[30]                        # nose tip
        mouth_left = pts[48]                  # left mouth corner
        mouth_right = pts[54]                 # right mouth corner
    else:
        # FaceMesh 468-point format
        left_eye = (pts[33] + pts[133]) / 2
        right_eye = (pts[362] + pts[263]) / 2
        nose = pts[1]
        mouth_left = pts[61]
        mouth_right = pts[291]
    kps5 = np.array([left_eye, right_eye, nose, mouth_left, mouth_right], dtype=np.float32)
    from facelib.LandmarksProcessor import get_transform_mat_from_5
    import facelib
    return get_transform_mat_from_5(kps5, 256, facelib.FaceType.WHOLE_FACE)

def check_and_adjust_resize(input_path: Path, fixed_window: int) -> int:
    """
    Smart check: if media width <= resize value, disable pre-resize
    
    Args:
        input_path: Input file or directory path
        fixed_window: User-specified resize value
        
    Returns:
        Adjusted resize value (0 if should be disabled)
    """
    if fixed_window <= 0:
        return 0
    
    try:
        import cv2
        
        # Check if it's an image file
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}
        if input_path.is_file() and input_path.suffix.lower() in image_extensions:
            img = cv2.imread(str(input_path))
            if img is not None:
                width = img.shape[1]
                if width <= fixed_window:
                    print(S('RESIZE_AUTO_DISABLED', width, fixed_window))
                    return 0
                return fixed_window
        
        # Check if it's a video file
        video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv'}
        if input_path.is_file() and input_path.suffix.lower() in video_extensions:
            cap = cv2.VideoCapture(str(input_path))
            if cap.isOpened():
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                cap.release()
                if width <= fixed_window:
                    print(S('RESIZE_AUTO_DISABLED', width, fixed_window))
                    return 0
                return fixed_window
        
        # If it's a directory, check the first file
        if input_path.is_dir():
            # Try to find video files first
            for ext in ['.mp4', '.avi', '.mov', '.mkv']:
                video_files = list(input_path.glob(f'*{ext}'))
                if video_files:
                    cap = cv2.VideoCapture(str(video_files[0]))
                    if cap.isOpened():
                        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        cap.release()
                        if width <= fixed_window:
                            print(S('RESIZE_AUTO_DISABLED', width, fixed_window))
                            return 0
                        return fixed_window
            
            # Then try image files
            for ext in image_extensions:
                image_files = list(input_path.glob(f'*{ext}'))
                if image_files:
                    img = cv2.imread(str(image_files[0]))
                    if img is not None:
                        width = img.shape[1]
                        if width <= fixed_window:
                            print(S('RESIZE_AUTO_DISABLED', width, fixed_window))
                            return 0
                        return fixed_window
        
        # Cannot determine, keep original value
        return fixed_window
        
    except Exception as e:
        print(S('RESIZE_CHECK_ERROR', e))
        return fixed_window  # Keep original value on error


def process_single_image(
    img_path: Path,
    img_idx: int,
    detector,
    landmarker,
    output_path: Path,
    image_size_fixed: Optional[int] = None,
    debug: bool = False,
    debug_dir: Optional[Path] = None,
    fixed_window: int = 0,
    face_type: str = 'whole_face',
    detection_angles: List[int] = None,
    kps_align: bool = True,
    input_mode: str = 'one_stage', resize_mode: str = 'letterbox', input_size: int = 640
) -> int:
    """
    Process single image using new approach: detect+landmark on resized, align on original
    Returns number of faces saved
    
    Args:
        detection_angles: List of angles for multi-angle detection [0, 90, 180, 270]
    """
    try:
        # Read image
        image = cv2.imread(str(img_path))
        if image is None:
            return 0
        
        # NEW APPROACH: Complete pipeline on resized image
        face_data_list = detect_and_align_on_resized(
            detector, landmarker, image, fixed_window, image_size_fixed, face_type, detection_angles,
            kps_align=kps_align,
            input_mode=input_mode, resize_mode=resize_mode, input_size=input_size
        )
        
        if not face_data_list:
            return 0
        
        # Align and save each face using original resolution image
        saved_count = 0
        for face_idx, face_data in enumerate(face_data_list):
            try:
                # Apply alignment to ORIGINAL image
                aligned_face, aligned_landmarks, face_rect_orig = apply_alignment_to_original(
                    image, face_data
                )
                
                out_size = face_data['out_size']
                
                # Generate filename
                filename = f"{img_idx:05d}_{face_idx}.jpg"
                filepath = output_path / filename

                # Store metadata in global cache (旧版本，非线程安全)
                if not hasattr(process_single_image, 'metadata_cache'):
                    process_single_image.metadata_cache = {}

                process_single_image.metadata_cache[filename] = {
                    'face_type': facelib.FaceType.toString(face_data.get('face_type_enum', facelib.FaceType.WHOLE_FACE)),
                    'landmarks': aligned_landmarks.tolist(),
                    'source_landmarks': face_data['landmarks'].tolist(),
                    'source_rect': face_rect_orig,
                    'image_to_face_mat': get_transform_mat(
                        face_data['landmarks'] * face_data['scale_factor'],
                        out_size,
                        face_data.get('face_type_enum', facelib.FaceType.WHOLE_FACE)
                    ).tolist(),
                    'source_filename': str(img_path.name)
                }

                # Save DFL-compatible JPG with embedded metadata
                save_dfljpg(str(filepath), aligned_face, process_single_image.metadata_cache[filename])

                saved_count += 1
            except Exception as e:
                print(S('ALIGN_SAVE_FAILED', img_path.name, face_idx, e))

        # Debug visualization if enabled
        if debug and debug_dir and face_data_list:
            visualize_extraction_stages(image, face_data_list, debug_dir, img_idx, fixed_window)
        
        return saved_count
    except Exception as e:
        print(S('PROCESS_IMAGE_FAILED', img_path.name, e))
        import traceback
        traceback.print_exc()
        return 0


def process_single_image_threadsafe(
    img_path: Path,
    img_idx: int,
    detector,
    landmarker,
    output_path: Path,
    image_size_fixed: Optional[int] = None,
    debug: bool = False,
    debug_dir: Optional[Path] = None,
    fixed_window: int = 0,
    face_type: str = 'whole_face',
    detection_angles: List[int] = None,
    kps_align: bool = True,
    input_mode: str = 'one_stage', resize_mode: str = 'letterbox', input_size: int = 640,
    metadata_cache: Dict = None,
    metadata_lock = None
) -> int:
    """
    Thread-safe version of process_single_image
    使用共享字典和线程锁保护元数据写入
    """
    try:
        # Read image
        image = cv2.imread(str(img_path))
        if image is None:
            return 0
        
        # NEW APPROACH: Complete pipeline on resized image
        face_data_list = detect_and_align_on_resized(
            detector, landmarker, image, fixed_window, image_size_fixed, face_type, detection_angles,
            kps_align=kps_align,
            input_mode=input_mode, resize_mode=resize_mode, input_size=input_size
        )
        
        if not face_data_list:
            return 0
        
        # Align and save each face using original resolution image
        saved_count = 0
        local_metadata = {}  # 本地缓存，最后统一加锁写入
        
        for face_idx, face_data in enumerate(face_data_list):
            try:
                # Apply alignment to ORIGINAL image
                aligned_face, aligned_landmarks, face_rect_orig = apply_alignment_to_original(
                    image, face_data
                )
                
                out_size = face_data['out_size']
                
                # Generate filename
                filename = f"{img_idx:05d}_{face_idx}.jpg"
                filepath = output_path / filename
                
                # 存储到本地缓存（避免频繁加锁）
                local_metadata[filename] = {
                    'face_type': facelib.FaceType.toString(face_data.get('face_type_enum', facelib.FaceType.WHOLE_FACE)),
                    'landmarks': aligned_landmarks.tolist(),
                    'source_landmarks': face_data['landmarks'].tolist(),
                    'source_rect': face_rect_orig,
                    'image_to_face_mat': get_transform_mat(
                        face_data['landmarks'] * face_data['scale_factor'],
                        out_size,
                        face_data.get('face_type_enum', facelib.FaceType.WHOLE_FACE)
                    ).tolist(),
                    'source_filename': str(img_path.name)
                }

                # Save DFL-compatible JPG with embedded metadata
                save_dfljpg(str(filepath), aligned_face, local_metadata[filename])

                saved_count += 1
            except Exception as e:
                print(S('ALIGN_SAVE_FAILED', img_path.name, face_idx, e))

        # 一次性将本地缓存合并到共享缓存（加锁保护）
        if local_metadata and metadata_cache is not None and metadata_lock is not None:
            with metadata_lock:
                metadata_cache.update(local_metadata)
        
        # Debug visualization if enabled
        if debug and debug_dir and face_data_list:
            visualize_extraction_stages(image, face_data_list, debug_dir, img_idx, fixed_window)
        
        return saved_count
    except Exception as e:
        print(S('PROCESS_IMAGE_FAILED', img_path.name, e))
        import traceback
        traceback.print_exc()
        return 0



def process_images(
    input_path: Path,
    output_path: Path,
    detector_name: str,
    landmark_name: str,
    device_info,
    image_size: int = None,  # Changed default to None for dynamic size
    debug: bool = False,
    fixed_window: int = 0,  # Pre-resize parameter
    face_type: str = 'whole_face',  # Face type for extraction
    detection_angles: List[int] = None,  # Multi-angle detection
    kps_align: bool = True,
    input_mode: str = 'one_stage', resize_mode: str = 'letterbox', input_size: int = 640,
):
    """处理图片文件夹 - 并行处理"""
    print(S('PROCESSING_IMAGES', input_path))
    
    # 创建检测器和标记器
    detector = DetectorFactory.create_detector(detector_name, device_info)
    landmarker = LandmarkFactory.create_landmarker(landmark_name, device_info)
    
    # 获取所有图片文件
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}
    image_files = sorted([
        f for f in input_path.iterdir() 
        if f.suffix.lower() in image_extensions
    ])
    
    if not image_files:
        print(S('NO_IMAGES'))
        return
    
    if debug:
        image_files = image_files[:1]
        print(f"[DEBUG] Processing only first image: {image_files[0].name}\n")  # Keep DEBUG in English
    else:
        print(S('FOUND_IMAGES', len(image_files)))
        print(S('USING_THREADS', cpu_count()))
        print()
    
    # Create output directory
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Create debug directory
    debug_dir = output_path / "debug" if debug else None
    
    if debug:
        total_faces = process_single_image(
            image_files[0], 0, detector, landmarker, output_path,
            image_size,  # Pass image_size directly
            debug=True,
            debug_dir=debug_dir,
            fixed_window=fixed_window,
            face_type=face_type,
            detection_angles=detection_angles,
            kps_align=kps_align,
            input_mode=input_mode,
            input_size=input_size,
        )
        print(S('DEBUG_COMPLETE', debug_dir))
    else:
        from threading import Lock
        
        max_workers = cpu_count()
        total_faces = 0
        processed_files = 0
        metadata_cache = {}  # 共享元数据缓存
        metadata_lock = Lock()  # 线程锁保护元数据写入
        
        pbar = tqdm.tqdm(total=len(image_files), desc="Processing", unit="img", ascii=True)
        _t0 = time.time()
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_idx = {
                executor.submit(
                    process_single_image_threadsafe,
                    img_path, idx, detector, landmarker, output_path,
                    image_size,  # Always pass image_size, None means dynamic calculation
                    False,  # debug
                    None,   # debug_dir
                    fixed_window,  # Pre-resize parameter
                    face_type,  # Face type for extraction
                    detection_angles,  # Multi-angle detection
                    kps_align,  # kps_align
                    input_mode,  # input preprocessing mode
                    resize_mode,  # resize mode
                    input_size,  # input square size
                    metadata_cache,  # 共享缓存
                    metadata_lock  # 线程锁
                ): (img_path, idx)
                for idx, img_path in enumerate(image_files)
            }
            
            # 收集结果
            for future in concurrent.futures.as_completed(future_to_idx):
                img_path, idx = future_to_idx[future]
                try:
                    saved_count = future.result(timeout=60)  # 60秒超时
                    total_faces += saved_count
                    processed_files += 1
                    pbar.update(1)
                    pbar.set_postfix({"extracted": total_faces})
                except concurrent.futures.TimeoutError:
                    print(S('TIMEOUT', img_path.name))
                    processed_files += 1
                    pbar.update(1)
                except Exception as e:
                    print(S('FAILED', img_path.name, e))
                    processed_files += 1
                    pbar.update(1)
        
        pbar.close()
        _t1 = time.time()
        print(f'⏱ Total: {_t1-_t0:.1f}s')
        print(S('COMPLETE', processed_files, total_faces, output_path))
        
        # Save metadata cache to HDF5 file
        if metadata_cache:
            import h5py
            
            metadata_file = output_path / "metadata.h5"
            with h5py.File(metadata_file, 'w') as f:
                # Create groups for each image
                for filename, meta in metadata_cache.items():
                    # Replace invalid characters in filename for HDF5 group names
                    safe_name = filename.replace('/', '_SLASH_').replace('\\', '_BSLASH_')
                    grp = f.create_group(safe_name)
                    
                    # 存储原始文件名
                    grp.attrs['__original_filename__'] = filename
                    
                    # Store each metadata field
                    for key, value in meta.items():
                        if isinstance(value, (list, np.ndarray)):
                            grp.create_dataset(key, data=np.array(value), compression='gzip', compression_opts=4)
                        elif isinstance(value, (int, float, str)):
                            grp.attrs[key] = value
                        else:
                            grp.attrs[key] = str(value)
            
            print(S('METADATA_SAVED', metadata_file))
            print(S('METADATA_ENTRIES', len(metadata_cache)))


def process_video(
    input_path: Path,
    output_path: Path,
    detector_name: str,
    landmark_name: str,
    device_info,
    image_size: int = None,
    fixed_window: int = 0,
    face_type: str = 'whole_face',
    detection_angles: List[int] = None,
    debug: bool = False,
    skip_frames: int = 0,  # 帧跳跃：0=逐帧，1=隔1帧...
    kps_align: bool = True,
    input_mode: str = 'one_stage', resize_mode: str = 'letterbox', input_size: int = 640,
):
    """Process video file - single process with GPU acceleration"""
    print(S('PROCESSING_VIDEO', input_path))
    
    # Initialize detector and landmarker once
    detector = DetectorFactory.create_detector(detector_name, device_info)
    landmarker = LandmarkFactory.create_landmarker(landmark_name, device_info)
    
    # Open video (解码器线程内会重新打开，这里只做可用性检查)
    cap_check = cv2.VideoCapture(str(input_path))
    if not cap_check.isOpened():
        print(S('CANNOT_OPEN_VIDEO', input_path))
        return
    total_frames = int(cap_check.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap_check.get(cv2.CAP_PROP_FPS)
    width = int(cap_check.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap_check.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap_check.release()
    
    print(S('VIDEO_INFO_FULL', total_frames, fps, width, height))
    print()
    
    # Create output directory
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Create debug directory if needed
    debug_dir = output_path / "debug" if debug else None
    
    total_saved = 0
    processed_frames = 0
    all_metadata = {}
    prev_faces = None  # For inter-frame face sorting
    frame_idx = 0
    
    pbar = tqdm.tqdm(total=total_frames, desc="Processing", unit="frame", ascii=True, bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {postfix}]')
    
    # 用于计算人脸提取速率
    import time
    start_time = time.time()
    last_update_time = start_time
    # 消费者-生产者：后台线程预解码帧与 GPU 推理流水线重叠
    import threading as _th
    import queue as _q
    _frame_queue = _q.Queue(maxsize=4)
    _stop_event = _th.Event()
    def _decoder():
        _cap = cv2.VideoCapture(str(input_path))
        while not _stop_event.is_set():
            ret, frame = _cap.read()
            _frame_queue.put(frame if ret else None)
            if not ret:
                break
            if skip_frames > 0:
                for _ in range(skip_frames):
                    if not _cap.grab():
                        _stop_event.set()
                        break
        _cap.release()
    _th.Thread(target=_decoder, daemon=True).start()

    while True:
        frame = _frame_queue.get()
        if frame is None:
            break
        
        try:
            # NEW APPROACH: Complete pipeline on resized image
            face_data_list = detect_and_align_on_resized(
                detector, landmarker, frame, fixed_window, image_size, face_type, detection_angles,
                kps_align=kps_align,
                input_mode=input_mode, resize_mode=resize_mode, input_size=input_size
            )
            
            # Extract face rects for inter-frame sorting
            faces = [face_data['face_rect'] for face_data in face_data_list]
            
            # Inter-frame face sorting (on working image coordinates)
            if faces and prev_faces is not None:
                sorted_indices = sort_faces_by_distance_for_data(prev_faces, faces)
                face_data_list = [face_data_list[i] for i in sorted_indices]
                faces = [face_data['face_rect'] for face_data in face_data_list]
            prev_faces = faces.copy() if faces else None
            
            faces_in_this_frame = 0
            
            # Align and save each face using original resolution image
            for face_idx, face_data in enumerate(face_data_list):
                try:
                    # Apply alignment to ORIGINAL image
                    aligned_face, aligned_landmarks, face_rect_orig = apply_alignment_to_original(
                        frame, face_data
                    )
                    
                    out_size = face_data['out_size']
                    
                    # Generate filename
                    filename = f"{frame_idx:05d}_{face_idx}.jpg"
                    filepath = output_path / filename
                    

                    # Store metadata (embedded as APP15 for DFL compat)
                    all_metadata[filename] = {
                        'face_type': facelib.FaceType.toString(face_data.get('face_type_enum', facelib.FaceType.WHOLE_FACE)),
                        'landmarks': aligned_landmarks.tolist(),
                        'source_landmarks': face_data['landmarks'].tolist(),
                        'source_rect': face_rect_orig,
                        'image_to_face_mat': get_transform_mat(
                            face_data['landmarks'] * face_data['scale_factor'],
                            out_size,
                            face_data.get('face_type_enum', facelib.FaceType.WHOLE_FACE)
                        ).tolist(),
                        'source_filename': str(input_path.name)
                    }
                    
                    # Save DFL-compatible JPG with embedded metadata
                    save_dfljpg(str(filepath), aligned_face, all_metadata[filename])

                    total_saved += 1
                    faces_in_this_frame += 1
                except Exception as e:
                    print(S('ALIGN_SAVE_FAILED', f"frame {frame_idx}", face_idx, e))
            
            processed_frames += 1
            pbar.update(1)
            
            # Debug mode: visualize first frame and exit
            if debug and face_data_list and debug_dir:
                visualize_extraction_stages(frame, face_data_list, debug_dir, frame_idx, fixed_window)
                print(S('DEBUG_COMPLETE', debug_dir))
                _stop_event.set()
                break
            
            # 每秒更新一次速率显示（只显示人脸提取速率）
            current_time = time.time()
            if current_time - last_update_time >= 1.0:
                elapsed = current_time - start_time
                if elapsed > 0:
                    faces_per_sec = total_saved / elapsed
                    frames_per_sec = frame_idx / elapsed
                    pbar.set_postfix({"frames/s": f"{frames_per_sec:.1f}",
                                      "faces/s": f"{faces_per_sec:.1f}",
                                      "saved": total_saved})
                last_update_time = current_time
            
        except Exception as e:
            print(S('PROCESS_IMAGE_FAILED', f"frame {frame_idx}", e))
            import traceback
            traceback.print_exc()
        
        frame_idx += 1

        # 解码器已跳过 skip_frames 帧，主线程只需更新计数器
        if skip_frames > 0:
            frame_idx += skip_frames
            pbar.update(skip_frames)

    pbar.close()
    _stop_event.set()  # 确保解码线程退出

    print(S('COMPLETE', processed_frames, total_saved, output_path))
    
    # Save metadata to HDF5
    if all_metadata:
        import h5py
        
        metadata_file = output_path / "metadata.h5"
        with h5py.File(metadata_file, 'w') as f:
            # Create groups for each image
            for filename, meta in all_metadata.items():
                # Replace invalid characters in filename for HDF5 group names
                safe_name = filename.replace('/', '_SLASH_').replace('\\', '_BSLASH_')
                grp = f.create_group(safe_name)
                
                # 存储原始文件名
                grp.attrs['__original_filename__'] = filename
                
                # Store each metadata field
                for key, value in meta.items():
                    if isinstance(value, (list, np.ndarray)):
                        grp.create_dataset(key, data=np.array(value), compression='gzip', compression_opts=4)
                    elif isinstance(value, (int, float, str)):
                        grp.attrs[key] = value
                    else:
                        grp.attrs[key] = str(value)
        
        print(S('METADATA_SAVED', metadata_file))
        print(S('METADATA_ENTRIES', len(all_metadata)))



def _check_dependencies():
    """Verify that refactored-out functions exist before consumer_worker runs."""
    for name in ('detect_faces_in_image', 'calculate_face_size'):
        if name not in globals():
            raise NameError(
                f"{name}() was removed in a refactor. "
                f"Reimplement before using consumer_worker.")


def consumer_worker(worker_id, detector_name, landmark_name, device_info,
                   image_size, output_path, input_path_name,
                   frame_queue, result_queue):
    """
    Consumer worker process: extract faces from frames

    NOTE: This function is currently unused (dead code). The previous refactor
    deleted helper functions detect_faces_in_image() and calculate_face_size()
    that this worker depended on. Before activating, reimplement both helpers
    or inline their logic.
    """
    _check_dependencies()
    try:
        # Initialize detector and landmarker once per process
        detector = DetectorFactory.create_detector(detector_name, device_info)
        landmarker = LandmarkFactory.create_landmarker(landmark_name, device_info)
        
        while True:
            # Get frame from queue
            item = frame_queue.get()
            
            if item is None:
                # Stop signal
                result_queue.put(None)
                break
            
            frame_idx, frame = item
            
            try:
                # Face detection
                faces, scale_factor = detect_faces_in_image(detector, frame)
                
                if not faces:
                    result_queue.put((frame_idx, 0, {}))
                    continue
                
                # Landmark extraction
                landmarks_list = extract_landmarks(landmarker, frame, faces)
                
                # Align and save each face
                saved_count = 0
                metadata_dict = {}
                
                for face_idx, (face_rect, landmarks) in enumerate(zip(faces, landmarks_list)):
                    if landmarks is None:
                        continue
                    
                    # Convert landmarks to standard 68 points
                    if len(landmarks) == 106:
                        landmarks_for_align = landmark106to68(landmarks)
                    elif len(landmarks) == 468:
                        landmarks_for_align = landmarks[:68]
                    elif len(landmarks) > 68:
                        landmarks_for_align = landmarks[:68]
                    else:
                        landmarks_for_align = landmarks
                    
                    # Calculate output size
                    if image_size is not None and image_size > 0:
                        out_size = image_size
                    else:
                        out_size = calculate_face_size(face_rect)
                    
                    try:
                        # Get transformation matrix
                        mat = get_transform_mat(landmarks_for_align, out_size, facelib.FaceType.WHOLE_FACE)
                        
                        # Affine transform
                        aligned_face = cv2.warpAffine(frame, mat, (out_size, out_size), 
                                                    flags=cv2.INTER_LANCZOS4)
                        
                        # Transform landmarks
                        aligned_landmarks = facelib.LandmarksProcessor.transform_points(landmarks_for_align, mat)
                        
                        # Generate filename
                        filename = f"{frame_idx:05d}_{face_idx}.jpg"
                        filepath = output_path / filename
                        
                        # Store metadata (embedded as APP15 for DFL compat)
                        metadata_dict[filename] = {
                            'face_type': facelib.FaceType.toString(facelib.FaceType.WHOLE_FACE),
                            'landmarks': aligned_landmarks.tolist(),
                            'source_landmarks': landmarks_for_align.tolist(),
                            'source_rect': face_rect,
                            'image_to_face_mat': mat.tolist(),
                            'source_filename': str(input_path_name)
                        }

                        # Save DFL-compatible JPG with embedded metadata
                        save_dfljpg(str(filepath), aligned_face, metadata_dict[filename])

                        saved_count += 1
                    except Exception as e:
                        print(S('ALIGN_SAVE_FAILED', f"frame {frame_idx}", face_idx, e))

                result_queue.put((frame_idx, saved_count, metadata_dict))
            except Exception as e:
                print(S('FAILED', f"Worker {worker_id} on frame {frame_idx}", e))
                import traceback
                traceback.print_exc()
                result_queue.put((frame_idx, 0, {}))
    
    except Exception as e:
        print(S('INIT_WORKER_FAILED', f"Worker {worker_id}: {e}"))
        import traceback
        traceback.print_exc()


def process_batch_frames(batch_args_list):
    """
    Process a batch of frames in worker process
    Args:
        batch_args_list: list of args tuples for each frame
    Returns:
        list of (frame_idx, saved_count, metadata_dict)
    """
    results = []
    for args in batch_args_list:
        result = process_single_frame(args)
        results.append(result)
    return results


def process_video_ffmpeg_pipe(
    ffmpeg_cmd: list,
    frame_size: tuple,  # (width, height)
    output_path: Path,
    detector_name: str,
    landmark_name: str,
    device_info,
    image_size: int = None,
    fixed_window: int = 0,
    face_type: str = 'whole_face',
    detection_angles: List[int] = None,
    debug: bool = False,
    kps_align: bool = True,
    input_mode: str = 'one_stage', resize_mode: str = 'letterbox', input_size: int = 640,
):
    """从 FFmpeg 管道读取 rawvideo，内存中完成人脸提取，不写中间帧到磁盘。"""
    print(f"[FFmpeg Pipe] 启动: {' '.join(ffmpeg_cmd[:8])}...")
    print(f"[FFmpeg Pipe] 使用 rawvideo rgb24 管道")
    detector = DetectorFactory.create_detector(detector_name, device_info)
    landmarker = LandmarkFactory.create_landmarker(landmark_name, device_info)
    output_path.mkdir(parents=True, exist_ok=True)
    debug_dir = output_path / "debug" if debug else None

    w, h = frame_size
    frame_bytes = w * h * 3  # rgb24 = 3 bytes/pixel
    proc = subprocess.Popen(ffmpeg_cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=2**20)

    total_saved = 0
    processed_frames = 0
    all_metadata = {}
    prev_faces = None
    frame_idx = 0

    pbar = tqdm.tqdm(desc="Pipe", unit="frame", ascii=True, file=sys.stdout,
                     bar_format='{l_bar}{bar}| {n_fmt} [{elapsed}<{remaining}, {postfix}]')
    pbar.refresh()
    sys.stdout.flush()
    import time as _time
    _start = _time.time()
    _last_upd = _start

    while True:
        # 读取固定大小的 rawvideo 帧（rgb24 = W*H*3 字节）
        raw = bytearray()
        while len(raw) < frame_bytes:
            chunk = proc.stdout.read(frame_bytes - len(raw))
            if not chunk:
                break
            raw.extend(chunk)
        if len(raw) < frame_bytes:
            break
        frame = cv2.cvtColor(
            np.frombuffer(bytes(raw), dtype=np.uint8).reshape((h, w, 3)),
            cv2.COLOR_RGB2BGR)

        try:
            face_data_list = detect_and_align_on_resized(
                detector, landmarker, frame, fixed_window, image_size,
                face_type, detection_angles, kps_align=kps_align,
                input_mode=input_mode, resize_mode=resize_mode, input_size=input_size
            )
            faces = [fd['face_rect'] for fd in face_data_list]

            if faces and prev_faces is not None:
                sorted_indices = sort_faces_by_distance_for_data(prev_faces, faces)
                face_data_list = [face_data_list[i] for i in sorted_indices]
                faces = [fd['face_rect'] for fd in face_data_list]
            prev_faces = faces.copy() if faces else None

            for face_idx, face_data in enumerate(face_data_list):
                try:
                    aligned_face, aligned_landmarks, face_rect_orig = apply_alignment_to_original(frame, face_data)
                    out_size = face_data['out_size']
                    filename = f"{frame_idx:05d}_{face_idx}.jpg"
                    filepath = output_path / filename
                    all_metadata[filename] = {
                        'face_type': facelib.FaceType.toString(
                            face_data.get('face_type_enum', facelib.FaceType.WHOLE_FACE)),
                        'landmarks': aligned_landmarks.tolist(),
                        'source_landmarks': face_data['landmarks'].tolist(),
                        'source_rect': face_rect_orig,
                        'image_to_face_mat': get_transform_mat(
                            face_data['landmarks'] * face_data['scale_factor'],
                            out_size,
                            face_data.get('face_type_enum', facelib.FaceType.WHOLE_FACE)
                        ).tolist(),
                        'source_filename': 'ffmpeg_pipe',
                    }
                    save_dfljpg(str(filepath), aligned_face, all_metadata[filename])
                    total_saved += 1
                except Exception as e:
                    print(S('ALIGN_SAVE_FAILED', f"frame {frame_idx}", face_idx, e))

            processed_frames += 1
            frame_idx += 1
            pbar.update(1)
            _now = _time.time()
            if _now - _last_upd >= 1.0:
                pbar.set_postfix(fps=f"{processed_frames/(_now-_start):.1f}", faces=total_saved)
                _last_upd = _now
        except Exception as e:
            print(S('PROCESSING_ERROR'), e)

    proc.wait()
    if processed_frames == 0 and proc.returncode != 0:
        print(f"[FFmpeg Pipe] ⚠ FFmpeg 退出码={proc.returncode}（无输出）")
    pbar.close()
    print(S('COMPLETE', processed_frames, total_saved, output_path))

    # Save metadata to HDF5
    if all_metadata:
        import h5py
        metadata_file = output_path / "metadata.h5"
        with h5py.File(metadata_file, 'w') as f:
            for filename, meta in all_metadata.items():
                safe_name = filename.replace('/', '_SLASH_').replace('\\', '_BSLASH_')
                grp = f.create_group(safe_name)
                grp.attrs['__original_filename__'] = filename
                for key, value in meta.items():
                    if isinstance(value, (list, np.ndarray)):
                        grp.create_dataset(key, data=np.array(value), compression='gzip', compression_opts=4)
                    elif isinstance(value, (int, float, str)):
                        grp.attrs[key] = value
                    else:
                        grp.attrs[key] = str(value)
        print(S('METADATA_SAVED', metadata_file))


def process_video_directory(
    input_dir: Path,
    output_path: Path,
    detector_name: str,
    landmark_name: str,
    device_info,
    image_size: int = None,
    fixed_window: int = 0,
    face_type: str = 'whole_face',
    detection_angles: List[int] = None,
    debug: bool = False,
    skip_frames: int = 0,
    kps_align: bool = True,
    input_mode: str = 'one_stage', resize_mode: str = 'letterbox', input_size: int = 640,
):
    """Process all video files in a directory - batch mode"""
    print(S('PROCESSING_VIDEO_DIR', input_dir))
    
    # Find all video files
    video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv'}
    video_files = sorted([
        f for f in input_dir.iterdir()
        if f.is_file() and f.suffix.lower() in video_extensions
    ])
    
    if not video_files:
        print(S('NO_VIDEOS_FOUND'))
        return
    
    print(S('FOUND_VIDEOS', len(video_files)))
    print()
    
    # Debug mode: only process first video
    if debug:
        video_files = video_files[:1]
        print(f"[DEBUG] Processing only first video: {video_files[0].name}\n")
    
    # Create output directory
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Create debug directory if needed
    debug_dir = output_path / "debug" if debug else None
    
    total_videos_processed = 0
    total_frames_processed = 0
    total_faces_saved = 0
    all_metadata = {}
    
    # Track used base names to handle duplicates
    used_base_names = {}  # Maps base_name -> count
    
    for video_idx, video_path in enumerate(video_files, 1):
        print(f"\n[{video_idx}/{len(video_files)}] Processing: {video_path.name}")
        
        # Generate unique base name for this video
        base_name = video_path.stem  # filename without extension
        ext_lower = video_path.suffix.lower().lstrip('.')
        
        # Check for duplicate base names
        if base_name in used_base_names:
            # Duplicate found, use format: name_ext_frame_XXXXX_XX.jpg
            unique_base = f"{base_name}_{ext_lower}"
            used_base_names[base_name] += 1
        else:
            # First occurrence, check if extension is needed
            unique_base = base_name
            used_base_names[base_name] = 1
        
        try:
            # Initialize detector and landmarker once per video
            detector = DetectorFactory.create_detector(detector_name, device_info)
            landmarker = LandmarkFactory.create_landmarker(landmark_name, device_info)
            
            # Open video
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                print(S('CANNOT_OPEN_VIDEO', video_path))
                continue
            
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            print(S('VIDEO_INFO_FULL', total_frames, fps, width, height))
            
            faces_in_video = 0
            frames_in_video = 0
            prev_faces = None
            frame_idx = 0
            
            pbar = tqdm.tqdm(total=total_frames, desc=f"Video {video_idx}", unit="frame", ascii=True, bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {postfix}]')
            
            import time
            start_time = time.time()
            last_update_time = start_time
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                try:
                    # Complete pipeline on resized image
                    face_data_list = detect_and_align_on_resized(
                        detector, landmarker, frame, fixed_window, image_size, face_type, detection_angles,
                        kps_align=kps_align,
                        input_mode=input_mode, resize_mode=resize_mode, input_size=input_size
                    )
                    
                    # Extract face rects for inter-frame sorting
                    faces = [face_data['face_rect'] for face_data in face_data_list]
                    
                    # Inter-frame face sorting
                    if faces and prev_faces is not None:
                        sorted_indices = sort_faces_by_distance_for_data(prev_faces, faces)
                        face_data_list = [face_data_list[i] for i in sorted_indices]
                        faces = [face_data['face_rect'] for face_data in face_data_list]
                    prev_faces = faces.copy() if faces else None
                    
                    # Align and save each face
                    for face_idx, face_data in enumerate(face_data_list):
                        try:
                            # Apply alignment to ORIGINAL image
                            aligned_face, aligned_landmarks, face_rect_orig = apply_alignment_to_original(
                                frame, face_data
                            )
                            
                            out_size = face_data['out_size']
                            
                            # Generate filename with unique base
                            filename = f"{unique_base}_frame_{frame_idx:05d}_{face_idx}.jpg"
                            filepath = output_path / filename
                            
                            # Store metadata (embedded as APP15 for DFL compat)
                            all_metadata[filename] = {
                                'face_type': facelib.FaceType.toString(face_data.get('face_type_enum', facelib.FaceType.WHOLE_FACE)),
                                'landmarks': aligned_landmarks.tolist(),
                                'source_landmarks': face_data['landmarks'].tolist(),
                                'source_rect': face_rect_orig,
                                'image_to_face_mat': get_transform_mat(
                                    face_data['landmarks'] * face_data['scale_factor'],
                                    out_size,
                                    face_data.get('face_type_enum', facelib.FaceType.WHOLE_FACE)
                                ).tolist(),
                                'source_filename': str(video_path.name)
                            }

                            # Save DFL-compatible JPG with embedded metadata
                            save_dfljpg(str(filepath), aligned_face, all_metadata[filename])

                            faces_in_video += 1
                            total_faces_saved += 1
                        except Exception as e:
                            print(S('ALIGN_SAVE_FAILED', f"frame {frame_idx}", face_idx, e))

                    frames_in_video += 1
                    total_frames_processed += 1
                    pbar.update(1)
                    
                    # Debug mode: visualize first frame and exit
                    if debug and face_data_list and debug_dir:
                        visualize_extraction_stages(frame, face_data_list, debug_dir, frame_idx, fixed_window)
                        print(f"[DEBUG] Visualization saved to: {debug_dir}")
                        print(S('DEBUG_COMPLETE', debug_dir))
                        # Close video and return early
                        pbar.close()
                        cap.release()
                        print(f"  [DEBUG] Debug completed for video: {video_path.name}")
                        return  # Exit function after first frame visualization
                    
                    # Update rate display
                    current_time = time.time()
                    if current_time - last_update_time >= 1.0:
                        elapsed = current_time - start_time
                        if elapsed > 0:
                            faces_per_sec = faces_in_video / elapsed
                            frames_per_sec = frame_idx / elapsed
                            pbar.set_postfix({"frames/s": f"{frames_per_sec:.1f}",
                                              "faces/s": f"{faces_per_sec:.1f}",
                                              "faces": faces_in_video})
                        last_update_time = current_time
                    
                except Exception as e:
                    print(S('PROCESS_IMAGE_FAILED', f"frame {frame_idx}", e))
                    import traceback
                    traceback.print_exc()
                
                frame_idx += 1
                # 帧跳跃
                if skip_frames > 0 and frame_idx < total_frames:
                    to_skip = min(skip_frames, total_frames - frame_idx)
                    for _ in range(to_skip):
                        ret = cap.grab()
                        if not ret:
                            break
                        frame_idx += 1
                        pbar.update(1)

            pbar.close()
            cap.release()

            print(f"  Video completed: {frames_in_video} frames, {faces_in_video} faces extracted")
            total_videos_processed += 1
            
        except Exception as e:
            print(S('FAILED', video_path.name, e))
            import traceback
            traceback.print_exc()
    
    print(f"\n{'='*80}")
    print(S('BATCH_COMPLETE_SUMMARY', total_videos_processed, total_frames_processed, total_faces_saved, output_path))
    
    # Save metadata to HDF5
    if all_metadata:
        import h5py
        
        metadata_file = output_path / "metadata.h5"
        with h5py.File(metadata_file, 'w') as f:
            # Create groups for each image
            for filename, meta in all_metadata.items():
                # Replace invalid characters in filename for HDF5 group names
                safe_name = filename.replace('/', '_SLASH_').replace('\\', '_BSLASH_')
                grp = f.create_group(safe_name)
                
                # 存储原始文件名
                grp.attrs['__original_filename__'] = filename
                
                # Store each metadata field
                for key, value in meta.items():
                    if isinstance(value, (list, np.ndarray)):
                        grp.create_dataset(key, data=np.array(value), compression='gzip', compression_opts=4)
                    elif isinstance(value, (int, float, str)):
                        grp.attrs[key] = value
                    else:
                        grp.attrs[key] = str(value)
        
        print(S('METADATA_SAVED', metadata_file))
        print(S('METADATA_ENTRIES', len(all_metadata)))


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description='DeepFaceLab Torch - 人脸提取器',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  python Extractor.py --input ".\\workspace\\data_dst.mp4" --output ".\\workspace\\data_dst\\aligned"
  python Extractor.py -i ".\\workspace\\images" -o ".\\workspace\\faces" -d BlazeFace -l insightface-2d106det
  python Extractor.py  # 交互式模式
        """
    )
    
    parser.add_argument(
        '-i', '--input',
        type=str,
        help='输入路径（视频文件或图片文件夹）'
    )
    
    parser.add_argument(
        '-o', '--output',
        type=str,
        help='输出路径（保存对齐后的人脸）'
    )
    
    parser.add_argument(
        '-d', '--FaceDetector',
        type=str,
        choices=list(DetectorFactory.DETECTORS.keys()),
        help=f'人脸检测器: {", ".join(DetectorFactory.DETECTORS.keys())}'
    )
    
    parser.add_argument(
        '-l', '--FaceMarker',
        type=str,
        choices=list(LandmarkFactory.LANDMARKS.keys()),
        help=f'特征点标记器: {", ".join(LandmarkFactory.LANDMARKS.keys())}'
    )
    
    parser.add_argument(
        '-s', '--size',
        type=int,
        default=None,  # None means dynamic calculation based on bbox
        help='Output image size (default: None = dynamic based on face bbox)'
    )
    
    parser.add_argument(
        '-r', '--resize',
        type=int,
        default=0,
        help='Pre-resize input image width before face detection (0 = no resize, improves performance for high-res images)'
    )
    
    parser.add_argument(
        '-t', '--face-type',
        type=str,
        choices=['half_face', 'midfull_face', 'full_face', 'whole_face', 'head'],
        default='whole_face',
        help='Face type for extraction (controls padding): half_face, midfull_face, full_face, whole_face, head'
    )
    
    parser.add_argument(
        '-a', '--angles',
        type=str,
        default='0',
        help='Detection angles in degrees (comma-separated), e.g., "0,90,180,270". Default: "0"'
    )
    
    parser.add_argument(
        '--quick-test',
        action='store_true',
        help='Quick test mode: only process first image with debug visualization'
    )

    parser.add_argument(
        '--skip-frames',
        type=int,
        default=0,
        help='Video frame skip: 0=all frames, 1=every other frame, 2=every 3rd frame...'
    )

    parser.add_argument(
        '--kps-align',
        action=argparse.BooleanOptionalAction,
        default=True,
        help='Use detector 5-point keypoints to pre-rotate face crop before landmark extraction '
             '(default: enabled). Only affects DamoFD, RetinaFace, TinyMog detectors.'
    )

    parser.add_argument(
        '--input-mode',
        type=str,
        choices=['one_stage', 'sliding_window'],
        default='one_stage',
        help='Detection mode: one_stage (resize whole image to input size, fast, default), '
             'sliding_window (scan image in fixed-size windows, better for small faces in '
             'large images). See GUI tooltip for per-detector suitability.'
    )

    parser.add_argument(
        '--resize-mode',
        type=str,
        choices=['letterbox', 'warp'],
        default='letterbox',
        help='Resize behavior: letterbox (aspect+pad, default), warp (direct stretch, fastest). '
             'Applies to one_stage whole-image resize and sliding_window edge windows.'
    )

    parser.add_argument(
        '--input-size',
        type=int,
        default=640,
        help='Square input/window size (default 640).'
    )

    parser.add_argument(
        '-m', '--mode',
        type=str,
        choices=['auto', 'video', 'image'],
        default='auto',
        help='Processing mode: auto (detect from input), video (force video mode), image (force image mode)'
    )

    parser.add_argument(
        '--ffmpeg-cmd',
        nargs=argparse.REMAINDER,
        help='FFmpeg rawvideo pipe command parts (passed from GUI). When set, input is ignored and frames come from FFmpeg stdout.'
    )
    parser.add_argument(
        '--ffmpeg-frame-size',
        type=str, default='',
        help='Frame size WxH for rawvideo pipe, e.g. "1920x1080"'
    )

    return parser.parse_args()


def main():
    """主函数"""
    #雷霆大字
    print("""
====================================================================================================================
███████╗██╗  ██╗████████╗██████╗  █████╗  ██████╗████████╗ ██████╗ ██████╗     ████████╗ ██████╗  ██████╗ ██╗     
██╔════╝╚██╗██╔╝╚══██╔══╝██╔══██╗██╔══██╗██╔════╝╚══██╔══╝██╔═══██╗██╔══██╗    ╚══██╔══╝██╔═══██╗██╔═══██╗██║     
█████╗   ╚███╔╝    ██║   ██████╔╝███████║██║        ██║   ██║   ██║██████╔╝       ██║   ██║   ██║██║   ██║██║     
██╔══╝   ██╔██╗    ██║   ██╔══██╗██╔══██║██║        ██║   ██║   ██║██╔══██╗       ██║   ██║   ██║██║   ██║██║     
███████╗██╔╝ ██╗   ██║   ██║  ██║██║  ██║╚██████╗   ██║   ╚██████╔╝██║  ██║       ██║   ╚██████╔╝╚██████╔╝███████╗
╚══════╝╚═╝  ╚═╝   ╚═╝   ╚═╝  ╚═╝╚═╝  ╚═╝ ╚═════╝   ╚═╝    ╚═════╝ ╚═╝  ╚═╝       ╚═╝    ╚═════╝  ╚═════╝ ╚══════╝                                                                                                          
====================================================================================================================
    """)
    args = parse_args()

    print()
    
    # ── FFmpeg 管道模式（GUI 传来，无 -i 参数）──
    if hasattr(args, 'ffmpeg_cmd') and args.ffmpeg_cmd:
        output_path = Path(args.output)
        detector_name = args.FaceDetector
        landmark_name = args.FaceMarker
        image_size = args.size
        fixed_window = args.resize
        face_type = args.face_type if hasattr(args, 'face_type') else 'whole_face'
        if hasattr(args, 'angles') and args.angles:
            try: detection_angles = [int(x.strip()) for x in args.angles.split(',')]
            except: detection_angles = [0]
        else: detection_angles = [0]
        quick_test = False
        skip_frames = 0
        input_path = Path(args.output)  # dummy, not used in pipe mode
        kps_align = getattr(args, 'kps_align', True)
        print(f"[Pipe] FFmpeg rawvideo pipe mode, output: {output_path}")
    elif not args.input or not args.output:
        # Interactive mode for missing paths
        if not args.input:
            input_path_str = input(S('ENTER_INPUT_PATH')).strip()
            input_path = Path(input_path_str)
        else:
            input_path = Path(args.input)
        
        if not args.output:
            output_path_str = input(S('ENTER_OUTPUT_PATH')).strip()
            output_path = Path(output_path_str)
        else:
            output_path = Path(args.output)
        
        if not input_path.exists():
            print(S('PATH_NOT_EXIST', input_path))
            return
        
        # Ask for detector if not provided
        if not args.FaceDetector:
            print(S('AVAILABLE_DETECTORS'))
            detectors = list(DetectorFactory.DETECTORS.keys())
            for i, det in enumerate(detectors, 1):
                print(f"  {i}. {det}")
            
            det_choice = input(S('SELECT_DETECTOR', len(detectors))).strip()
            det_idx = int(det_choice) - 1 if det_choice.isdigit() else 4
            detector_name = detectors[det_idx] if 0 <= det_idx < len(detectors) else 'BlazeFace'
        else:
            detector_name = args.FaceDetector
        
        # Ask for landmarker if not provided
        if not args.FaceMarker:
            print(S('AVAILABLE_LANDMARKERS'))
            landmarks = list(LandmarkFactory.LANDMARKS.keys())
            for i, lm in enumerate(landmarks, 1):
                print(f"  {i}. {lm}")
            
            lm_choice = input(S('SELECT_LANDMARKER', len(landmarks))).strip()
            lm_idx = int(lm_choice) - 1 if lm_choice.isdigit() else 0
            landmark_name = landmarks[lm_idx] if 0 <= lm_idx < len(landmarks) else 'insightface-2d106det'
        else:
            landmark_name = args.FaceMarker
        
        # Ask for pre-resize if not provided
        if args.resize == 0 and not args.input:  # Only ask in fully interactive mode
            resize_input = input(S('ENTER_RESIZE_SIZE', '1920')).strip()
            fixed_window = int(resize_input) if resize_input.isdigit() else 0
        else:
            fixed_window = args.resize  # Pre-resize parameter
        
        # Ask for detection angles if not explicitly provided via command line
        import sys
        if '-a' not in sys.argv and '--angles' not in sys.argv:
            angles_input = input("Enter detection angles (comma-separated, e.g., 0,90,180,270) [default: 0]: ").strip()
            if angles_input:
                try:
                    detection_angles = [int(x.strip()) for x in angles_input.split(',')]
                except:
                    detection_angles = [0]
            else:
                detection_angles = [0]
        else:
            try:
                detection_angles = [int(x.strip()) for x in args.angles.split(',')]
            except:
                detection_angles = [0]
        
        # Quick test mode - 默认禁用
        if hasattr(args, 'quick_test') and args.quick_test:
            quick_test = True
        else:
            quick_test = False
        skip_frames = getattr(args, 'skip_frames', 0)

        # Get face type
        face_type = args.face_type if hasattr(args, 'face_type') else 'whole_face'
        
        image_size = args.size  # None means dynamic
        
        input_mode = getattr(args, 'input_mode', 'one_stage')
        resize_mode = getattr(args, 'resize_mode', 'letterbox')
        input_size = getattr(args, 'input_size', 640)
        print(S('CONFIGURATION'))
        print(f"  {S('INPUT_PATH')}: {input_path}")
        print(f"  {S('OUTPUT_PATH')}: {output_path}")
        print(f"  {S('DETECTOR')}: {detector_name}")
        print(f"  {S('LANDMARKER')}: {landmark_name}")
        print(f"  {S('OUTPUT_SIZE')}: {S('OUTPUT_SIZE_DYNAMIC') if image_size is None else f'{image_size}x{image_size}'}")
        print(f"  {S('PRE_RESIZE')}: {f'{fixed_window}px' if fixed_window > 0 else S('PRE_RESIZE_DISABLED')}")
        print(f"  Detection Angles: {detection_angles}")
        print(f"  Quick Test: {quick_test}")
        print(f"  Frame Skip: {skip_frames}")
        print(f"  KPS Align: {args.kps_align}")
        print(f"  Input Mode: {input_mode} | Resize Mode: {resize_mode} | Input Size: {input_size}")
        print()
        
        confirm = input(S('CONFIRM_START')).strip().lower()
        # 空输入或 y/yes 都视为确认
        if confirm == '' or confirm == 'y' or confirm == 'yes':
            pass  # 继续执行
        else:
            print(S('CANCELLED'))
            return
    elif not args.FaceDetector or not args.FaceMarker:
        # Paths provided but missing detector/landmarker - interactive selection
        input_path = Path(args.input)
        output_path = Path(args.output)
        
        if not input_path.exists():
            print(S('PATH_NOT_EXIST', input_path))
            return
        
        # Ask for detector if not provided
        if not args.FaceDetector:
            print(S('AVAILABLE_DETECTORS'))
            detectors = list(DetectorFactory.DETECTORS.keys())
            for i, det in enumerate(detectors, 1):
                print(f"  {i}. {det}")
            
            det_choice = input(S('SELECT_DETECTOR', len(detectors))).strip()
            det_idx = int(det_choice) - 1 if det_choice.isdigit() else 4
            detector_name = detectors[det_idx] if 0 <= det_idx < len(detectors) else 'BlazeFace'
        else:
            detector_name = args.FaceDetector
        
        # Ask for landmarker if not provided
        if not args.FaceMarker:
            print(S('AVAILABLE_LANDMARKERS'))
            landmarks = list(LandmarkFactory.LANDMARKS.keys())
            for i, lm in enumerate(landmarks, 1):
                print(f"  {i}. {lm}")
            
            lm_choice = input(S('SELECT_LANDMARKER', len(landmarks))).strip()
            lm_idx = int(lm_choice) - 1 if lm_choice.isdigit() else 0
            landmark_name = landmarks[lm_idx] if 0 <= lm_idx < len(landmarks) else 'insightface-2d106det'
        else:
            landmark_name = args.FaceMarker
        
        # Ask for pre-resize if not provided
        if args.resize == 0 and not args.FaceDetector:  # Only ask when detector was also selected interactively
            resize_input = input(S('ENTER_RESIZE_SIZE', '1920')).strip()
            fixed_window = int(resize_input) if resize_input.isdigit() else 0
        else:
            fixed_window = args.resize  # Pre-resize parameter
        
        # Ask for detection angles if not explicitly provided via command line
        import sys
        if '-a' not in sys.argv and '--angles' not in sys.argv:
            angles_input = input("Enter detection angles (comma-separated, e.g., 0,90,180,270) [default: 0]: ").strip()
            if angles_input:
                try:
                    detection_angles = [int(x.strip()) for x in angles_input.split(',')]
                except:
                    detection_angles = [0]
            else:
                detection_angles = [0]
        else:
            try:
                detection_angles = [int(x.strip()) for x in args.angles.split(',')]
            except:
                detection_angles = [0]
        
        # Quick test mode - 默认禁用
        if hasattr(args, 'quick_test') and args.quick_test:
            quick_test = True
        else:
            quick_test = False
        skip_frames = getattr(args, 'skip_frames', 0)

        # Get face type
        face_type = args.face_type if hasattr(args, 'face_type') else 'whole_face'

        image_size = args.size

        input_mode = getattr(args, 'input_mode', 'one_stage')
        resize_mode = getattr(args, 'resize_mode', 'letterbox')
        input_size = getattr(args, 'input_size', 640)
        print(S('CONFIGURATION'))
        print(f"  {S('INPUT_PATH')}: {input_path}")
        print(f"  {S('OUTPUT_PATH')}: {output_path}")
        print(f"  {S('DETECTOR')}: {detector_name}")
        print(f"  {S('LANDMARKER')}: {landmark_name}")
        print(f"  {S('OUTPUT_SIZE')}: {S('OUTPUT_SIZE_DYNAMIC') if image_size is None else f'{image_size}x{image_size}'}")
        print(f"  {S('PRE_RESIZE')}: {f'{fixed_window}px' if fixed_window > 0 else S('PRE_RESIZE_DISABLED')}")
        print(f"  Detection Angles: {detection_angles}")
        print(f"  Quick Test: {quick_test}")
        print(f"  Frame Skip: {skip_frames}")
        print(f"  KPS Align: {args.kps_align}")
        print(f"  Input Mode: {input_mode} | Resize Mode: {resize_mode} | Input Size: {input_size}")
        print()
        
        confirm = input(S('CONFIRM_START')).strip().lower()
        # 空输入或 y/yes 都视为确认
        if confirm == '' or confirm == 'y' or confirm == 'yes':
            pass  # 继续执行
        else:
            print(S('CANCELLED'))
            return
    else:
        # Full command line mode (all params provided)
        input_path = Path(args.input)
        output_path = Path(args.output)
        detector_name = args.FaceDetector
        landmark_name = args.FaceMarker
        image_size = args.size
        fixed_window = args.resize  # Pre-resize parameter
        face_type = args.face_type if hasattr(args, 'face_type') else 'whole_face'
        
        # Parse detection angles - ask if not provided
        if hasattr(args, 'angles') and args.angles:
            try:
                detection_angles = [int(x.strip()) for x in args.angles.split(',')]
            except:
                detection_angles = [0]
        else:
            # Not provided via command line, ask user
            angles_input = input("Enter detection angles (comma-separated, e.g., 0,90,180,270) [default: 0]: ").strip()
            if angles_input:
                try:
                    detection_angles = [int(x.strip()) for x in angles_input.split(',')]
                except:
                    detection_angles = [0]
            else:
                detection_angles = [0]
        
        # Quick test mode - UI模式强制禁用
        # 当所有参数都通过命令行提供时（UI模式），默认禁用快速测试
        import sys
        if hasattr(args, 'quick_test') and args.quick_test:
            # 显式指定了 --quick-test 参数
            quick_test = True
        else:
            # UI调用或命令行未指定：默认禁用
            quick_test = False
            skip_frames = getattr(args, 'skip_frames', 0)

        input_mode = getattr(args, 'input_mode', 'one_stage')
        resize_mode = getattr(args, 'resize_mode', 'letterbox')
        input_size = getattr(args, 'input_size', 640)
        print(S('CMD_MODE'))
        print(f"  {S('INPUT_PATH')}: {input_path}")
        print(f"  {S('OUTPUT_PATH')}: {output_path}")
        print(f"  {S('DETECTOR')}: {detector_name}")
        print(f"  {S('LANDMARKER')}: {landmark_name}")
        print(f"  {S('OUTPUT_SIZE')}: {S('OUTPUT_SIZE_DYNAMIC') if image_size is None else f'{image_size}x{image_size}'}")
        print(f"  {S('PRE_RESIZE')}: {f'{fixed_window}px' if fixed_window > 0 else S('PRE_RESIZE_DISABLED')}")
        print(f"  Detection Angles: {detection_angles}")
        print(f"  Quick Test: {quick_test}")
        print(f"  Frame Skip: {skip_frames}")
        print(f"  KPS Align: {args.kps_align}")
        print(f"  Input Mode: {input_mode} | Resize Mode: {resize_mode} | Input Size: {input_size}")
        print()
        
        if not input_path.exists():
            print(S('PATH_NOT_EXIST', input_path))
            return
    
    # 初始化设备（按优先级：CUDA > DX12 > CPU）
    print(S('INIT_DEVICE'))
    devices = get_available_devices_info()
    
    device_info = None
    selected_device_name = S('NO_DEVICE')
    
    # 优先级1: CUDA (GPU)
    for device in devices:
        if 'CUDA' in str(device).upper() or 'cuda' in str(device).lower():
            device_info = device
            selected_device_name = str(device)
            print(S('DEVICE_CUDA', selected_device_name))
            break
    
    # 优先级2: DirectML/DX12 (GPU)
    if device_info is None:
        for device in devices:
            if 'DML' in str(device).upper() or 'directml' in str(device).lower() or 'dx12' in str(device).lower():
                device_info = device
                selected_device_name = str(device)
                print(S('DEVICE_DIRECTML', selected_device_name))
                break
    
    # 优先级3: CPU
    if device_info is None:
        device_info = get_cpu_device_info()
        selected_device_name = str(device_info)
        print(S('DEVICE_CPU', selected_device_name))
    
    print()
    
    # Smart check and adjust resize parameter
    if fixed_window > 0:
        fixed_window = check_and_adjust_resize(input_path, fixed_window)
    
    # Process input
    try:
        # FFmpeg pipe mode (from GUI, zero disk writes for intermediate frames)
        if hasattr(args, 'ffmpeg_cmd') and args.ffmpeg_cmd:
            _fs = args.ffmpeg_frame_size
            _frame = tuple(map(int, _fs.split('x'))) if _fs and 'x' in _fs else (0, 0)
            process_video_ffmpeg_pipe(
                args.ffmpeg_cmd, _frame, output_path,
                detector_name, landmark_name, device_info,
                image_size, fixed_window, face_type, detection_angles,
                debug=quick_test, kps_align=args.kps_align,
                input_mode=args.input_mode, resize_mode=args.resize_mode, input_size=args.input_size
            )
            return

        # Determine processing mode
        processing_mode = args.mode if hasattr(args, 'mode') else 'auto'
        
        if input_path.is_file():
            # Single video file
            process_video(input_path, output_path, detector_name, landmark_name, device_info, image_size, fixed_window, face_type, detection_angles, debug=quick_test, skip_frames=skip_frames, kps_align=args.kps_align, input_mode=args.input_mode, resize_mode=args.resize_mode, input_size=args.input_size)
        elif input_path.is_dir():
            # Directory - check mode
            if processing_mode == 'video':
                # Force video directory mode
                process_video_directory(input_path, output_path, detector_name, landmark_name, device_info, image_size, fixed_window, face_type, detection_angles, debug=quick_test, skip_frames=skip_frames, kps_align=args.kps_align, input_mode=args.input_mode, resize_mode=args.resize_mode, input_size=args.input_size)
            elif processing_mode == 'image':
                # Force image mode
                process_images(input_path, output_path, detector_name, landmark_name, device_info,
                             image_size, debug=quick_test, fixed_window=fixed_window,
                             face_type=face_type, detection_angles=detection_angles,
                             kps_align=args.kps_align, input_mode=args.input_mode, resize_mode=args.resize_mode, input_size=args.input_size)
            else:
                # Auto mode - detect content type
                video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv'}
                image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}
                
                video_files = [f for f in input_path.iterdir() if f.is_file() and f.suffix.lower() in video_extensions]
                image_files = [f for f in input_path.iterdir() if f.is_file() and f.suffix.lower() in image_extensions]
                
                if video_files and not image_files:
                    # Only videos found
                    print(S('AUTO_DETECT_VIDEO_MODE'))
                    process_video_directory(input_path, output_path, detector_name, landmark_name, device_info, image_size, fixed_window, face_type, detection_angles, debug=quick_test, skip_frames=skip_frames, kps_align=args.kps_align, input_mode=args.input_mode, resize_mode=args.resize_mode, input_size=args.input_size)
                elif image_files and not video_files:
                    # Only images found
                    print(S('AUTO_DETECT_IMAGE_MODE'))
                    process_images(input_path, output_path, detector_name, landmark_name, device_info,
                                 image_size, debug=quick_test, fixed_window=fixed_window,
                                 face_type=face_type, detection_angles=detection_angles,
                                 kps_align=args.kps_align, input_mode=args.input_mode, resize_mode=args.resize_mode, input_size=args.input_size)
                elif video_files and image_files:
                    # Both found - ask user
                    print(S('MIXED_CONTENT_DETECTED', len(video_files), len(image_files)))
                    choice = input(S('SELECT_PROCESSING_MODE')).strip().lower()
                    if choice == 'v' or choice == 'video':
                        process_video_directory(input_path, output_path, detector_name, landmark_name, device_info, image_size, fixed_window, face_type, detection_angles, debug=quick_test, skip_frames=skip_frames, kps_align=args.kps_align, input_mode=args.input_mode, resize_mode=args.resize_mode, input_size=args.input_size)
                    else:
                        process_images(input_path, output_path, detector_name, landmark_name, device_info,
                                     image_size, debug=quick_test, fixed_window=fixed_window,
                                     face_type=face_type, detection_angles=detection_angles,
                                     kps_align=args.kps_align, input_mode=args.input_mode, resize_mode=args.resize_mode, input_size=args.input_size)
                else:
                    # No supported files
                    print(S('NO_SUPPORTED_FILES'))
        else:
            print(S('INVALID_PATH_TYPE', input_path))
    except Exception as e:
        print(S('PROCESSING_ERROR'))
        traceback.print_exc()


if __name__ == '__main__':
    main()
