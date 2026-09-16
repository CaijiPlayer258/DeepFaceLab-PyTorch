"""
DeepFaceLab Torch - FacesetProcessor Sorter Module
人脸集排序器模块：基于多种特征对人脸进行排序
支持多进程、可组合、易扩展的架构
"""

import sys
from pathlib import Path
import cv2
import numpy as np
import json
import argparse
from typing import Dict, List, Optional, Tuple, Callable
import tqdm
from multiprocessing import Pool, cpu_count
import operator
import logging

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 导入多语言支持
from strings import S

import warnings
warnings.filterwarnings('ignore', module='onnxruntime')

# 注：本模块【不需要】onnxruntime。它以前在模块顶部 import，但那会让任何
#     import 本模块的 GUI 页面在 Qt 之后崩在
#         WinError 1114 动态链接库(DLL)初始化例程失败
#     （原生扩展在 Qt 之后加载会失败，和 torch / c10.dll 同一机制）。
#     排序路径完全不碰 onnxruntime，所以直接去掉这个 import。

# 导入项目模块
from core import pathex
from core.cv2ex import cv2_imread
from facelib.LandmarksProcessor import get_image_hull_mask, estimate_pitch_yaw_roll
from DFLIMG import DFLIMG

# 导入基类
from FacesetBaseProcessor import FacesetBaseProcessor


class SortMethod:
    """排序方法枚举"""
    PHASH = "phash"              # 感知哈希排序
    HIST = "hist"                # 直方图排序
    BLUR = "blur"                # 模糊度排序
    FACE_POSE = "face_pose"      # 人脸姿态排序
    RESOLUTION = "resolution"    # 分辨率排序
    COLOR = "color"              # 颜色排序
    NAME = "name"                # 文件名排序
    ABS_DIFF = "absdiff"         # 绝对差异排序
    MOTION_BLUR = "motion_blur"  # 运动模糊排序
    OCCLUSION = "occlusion"      # 遮挡排序
    LANDMARKS_OCCLUSION = "landmarks_occlusion"  #  landmarks遮挡排序
    MIX_OCCLUSION = "mix_occlusion"  # 混合遮挡排序
    HIST_DISSIM = "hist_dissim"  # 直方图相似度排序
    
    # 所有可用的排序方法
    ALL_METHODS = [
        PHASH, HIST, BLUR, FACE_POSE, RESOLUTION, COLOR, NAME,
        ABS_DIFF, MOTION_BLUR, OCCLUSION, LANDMARKS_OCCLUSION,
        MIX_OCCLUSION, HIST_DISSIM
    ]


def _calculate_phash_worker(args):
    """
    多进程工作函数：计算单张图片的感知哈希
    
    Args:
        args: (image_path_str,)
        
    Returns:
        (filename, phash_value)
    """
    from imagehash import phash
    from PIL import Image
    
    image_path_str = args[0]
    
    try:
        image = Image.open(image_path_str)
        hash_value = phash(image)
        return (Path(image_path_str).name, int(str(hash_value), 16))
    except Exception as e:
        return (Path(image_path_str).name, None)


def _calculate_histogram_worker(args):
    """
    多进程工作函数：计算单张图片的直方图（缩略图加速）。
    排序只需要直方图分布相似度，缩略图到 64×64 计算 256-bin
    直方图，分布几乎不变但速度快 ~8×。
    """
    image_path_str = args[0]
    try:
        image = cv2_imread(image_path_str)
        if image is None:
            return (Path(image_path_str).name, None)
        h, w = image.shape[:2]
        if h > 64 or w > 64:
            fx = 64 / max(h, w)
            image = cv2.resize(image, None, fx=fx, fy=fx,
                               interpolation=cv2.INTER_AREA)
        hist_b = cv2.calcHist([image], [0], None, [256], [0, 256]).flatten()
        hist_g = cv2.calcHist([image], [1], None, [256], [0, 256]).flatten()
        hist_r = cv2.calcHist([image], [2], None, [256], [0, 256]).flatten()
        hist_b = (hist_b / hist_b.sum()).tolist()
        hist_g = (hist_g / hist_g.sum()).tolist()
        hist_r = (hist_r / hist_r.sum()).tolist()
        return (Path(image_path_str).name, {'b': hist_b, 'g': hist_g, 'r': hist_r})
    except Exception:
        return (Path(image_path_str).name, None)


_SHARP_TRAIN_RES = 256          # 与 Filter.py 的默认值保持一致（= 训练分辨率）
_SHARP_METRIC_FNS = None        # (score_fn, mask_fn) 惰性导入缓存


def _sharp_field(train_res: int) -> str:
    """元数据字段名。**按分辨率区分**，这样换 train_res 不会和旧缓存混用。

    ⚠️ 不用旧的 'sharpness' / 'blur' 字段：那是拉普拉斯方差，量纲不同。
    混用会让一部分图用旧尺子、一部分用新尺子，排出来的顺序毫无意义。
    """
    return f'sharpness_br{int(train_res)}'


def _sharp_metric():
    """惰性导入 Filter 里的清晰度指标，保证 Filter / Sorter 用的是同一份实现。"""
    global _SHARP_METRIC_FNS
    if _SHARP_METRIC_FNS is None:
        try:
            from Filter import face_sharpness_score, _load_face_mask
        except ImportError:
            sys.path.insert(0, str(Path(__file__).parent))
            from Filter import face_sharpness_score, _load_face_mask
        _SHARP_METRIC_FNS = (face_sharpness_score, _load_face_mask)
    return _SHARP_METRIC_FNS


def _calculate_blur_worker(args):
    """
    多进程工作函数：计算单张图片的清晰度 = E(b0)/E(b2)（高频/低频能量比），**分数越高越清晰**。

    与 Filter.face_sharpness_score 是同一份实现 —— 排序和过滤共用一把尺子。

    与旧版拉普拉斯方差的三个区别：
      1. 带比是【比值】而非绝对值，内容对比度自动约掉
         （实测内容变异 CV 0.45 -> 0.11，模糊分离度 1.49 -> 2.12）。
      2. 掩码只用于【选择统计像素】，不再 `image * mask` —— 那会在掩码边界
         制造强度极高的假边缘，主导拉普拉斯响应。
      3. 噪声补偿：噪声=高频=高分，不扣掉的话高 ISO / 强锐化的帧会排到最前面。

    Args:
        args: (image_path_str, landmarks, use_motion_blur[, train_res])

    Returns:
        (filename, sharpness)  分数越高越清晰
    """
    image_path_str, landmarks, use_motion_blur = args[0], args[1], args[2]
    train_res = args[3] if len(args) > 3 else _SHARP_TRAIN_RES

    try:
        image = cv2_imread(image_path_str)
        if image is None:
            return (Path(image_path_str).name, 0.0)

        score_fn, mask_fn = _sharp_metric()
        mask = mask_fn(image_path_str, image, landmarks)
        return (Path(image_path_str).name, float(score_fn(image, mask, train_res)))
    except Exception as e:
        return (Path(image_path_str).name, 0.0)


def _calculate_face_pose_worker(args):
    """
    多进程工作函数：计算单张图片的人脸姿态
    
    Args:
        args: (image_path_str, landmarks)
        
    Returns:
        (filename, pose_dict)
    """
    image_path_str, landmarks = args
    
    try:
        image = cv2_imread(image_path_str)
        if image is None or landmarks is None:
            return (Path(image_path_str).name, None)
        
        # 估计姿态
        pitch, yaw, roll = estimate_pitch_yaw_roll(np.array(landmarks), image.shape[1], image.shape[0])
        
        return (Path(image_path_str).name, {
            'pitch': float(pitch),
            'yaw': float(yaw),
            'roll': float(roll)
        })
    except Exception as e:
        return (Path(image_path_str).name, None)


def _calculate_resolution_worker(args):
    """
    多进程工作函数：计算单张图片的分辨率
    
    Args:
        args: (image_path_str,)
        
    Returns:
        (filename, resolution)
    """
    image_path_str = args[0]
    
    try:
        image = cv2_imread(image_path_str)
        if image is None:
            return (Path(image_path_str).name, 0)
        
        h, w = image.shape[:2]
        resolution = h * w
        
        return (Path(image_path_str).name, resolution)
    except Exception as e:
        return (Path(image_path_str).name, 0)



def _sort_hist_chunk(args):
    """
    多进程工作函数：对单个块进行直方图贪心排序

    Args:
        args: (chunk_indices, file_list, hist_arrays_dict)

    Returns:
        sorted_original_indices: 排序后的原始索引列表
    """
    chunk_indices, file_list, hist_arrays_dict = args

    chunk_files = [file_list[i] for i in chunk_indices]
    chunk_hist = {f: hist_arrays_dict[f] for f in chunk_files}

    # 块内贪心排序
    sorted_chunk = []
    remaining = set(range(len(chunk_files)))

    if not remaining:
        return []

    current_idx = 0
    sorted_chunk.append(current_idx)
    remaining.remove(current_idx)

    while remaining:
        current_file = chunk_files[current_idx]
        current_hist_b, current_hist_g, current_hist_r = chunk_hist[current_file]

        min_score = float("inf")
        min_index = -1

        for i in remaining:
            test_file = chunk_files[i]
            test_hist_b, test_hist_g, test_hist_r = chunk_hist[test_file]

            score = (cv2.compareHist(current_hist_b, test_hist_b, cv2.HISTCMP_BHATTACHARYYA) +
                     cv2.compareHist(current_hist_g, test_hist_g, cv2.HISTCMP_BHATTACHARYYA) +
                     cv2.compareHist(current_hist_r, test_hist_r, cv2.HISTCMP_BHATTACHARYYA))

            if score < min_score:
                min_score = score
                min_index = i

        if min_index >= 0:
            sorted_chunk.append(min_index)
            remaining.remove(min_index)
            current_idx = min_index

    # 返回原始索引
    return [chunk_indices[i] for i in sorted_chunk]


def _merge_hist_chunks(chunk_a, chunk_b, file_list, hist_arrays):
    """
    合并两个已排序的 hist 块：旋转 chunk_b 使其头部与 chunk_a 尾部对齐。

    Args:
        chunk_a: 已排序的块A（原始索引列表）
        chunk_b: 已排序的块B（原始索引列表）
        file_list: 文件名列表
        hist_arrays: {filename: (hist_b, hist_g, hist_r)}

    Returns:
        合并后的完整原始索引列表
    """
    if not chunk_a or not chunk_b:
        return chunk_a + chunk_b

    # 取 chunk_a 的最后一个元素作为锚点
    last_a_idx = chunk_a[-1]
    last_a_file = file_list[last_a_idx]
    last_a_hb, last_a_hg, last_a_hr = hist_arrays[last_a_file]

    # 在 chunk_b 中找到与锚点最相似的元素
    best_idx = 0
    best_dist = float('inf')
    for i, idx_b in enumerate(chunk_b):
        file_b = file_list[idx_b]
        hb, hg, hr = hist_arrays[file_b]
        dist = (cv2.compareHist(last_a_hb, hb, cv2.HISTCMP_BHATTACHARYYA) +
                cv2.compareHist(last_a_hg, hg, cv2.HISTCMP_BHATTACHARYYA) +
                cv2.compareHist(last_a_hr, hr, cv2.HISTCMP_BHATTACHARYYA))
        if dist < best_dist:
            best_dist = dist
            best_idx = i

    # 旋转 chunk_b，使最相似的元素成为新的开头
    chunk_b_rotated = chunk_b[best_idx:] + chunk_b[:best_idx]
    return chunk_a + chunk_b_rotated


class FaceSorter(FacesetBaseProcessor):
    """人脸排序器 - 基于多种特征对人脸进行排序
    
    继承自 FacesetBaseProcessor，自动获得：
    - HDF5 流式访问能力
    - 图像文件扫描
    - 元数据缓存管理
    - 完整性校验
    """
    
    def __init__(self, faceset_path: Path, metadata_file: Optional[Path] = None):
        """
        初始化排序器
        
        Args:
            faceset_path: 人脸数据集路径
            metadata_file: 元数据文件路径（可选，默认为 faceset_path/metadata.h5）
        """
        # 调用父类初始化（会自动初始化 HDF5 访问器和扫描文件）
        super().__init__(faceset_path, metadata_file)
    
    def _save_to_temp_h5(self, new_features: Dict[str, Dict], index: int) -> Optional[Path]:
        """
        将特征保存到临时HDF5文件
        
        Args:
            new_features: {filename: {feature_key: feature_value}}
            index: 临时文件索引
            
        Returns:
            临时文件路径，失败返回None
        """
        if not new_features:
            return None
        
        try:
            import h5py
            temp_file = self.faceset_path / f"_temp_sort_{index}.h5"
            
            with h5py.File(temp_file, 'w') as f:
                for filename, features in new_features.items():
                    safe_name = filename.replace('/', '_SLASH_').replace('\\', '_BSLASH_')
                    grp = f.create_group(safe_name)
                    grp.attrs['__original_filename__'] = filename
                    
                    for key, value in features.items():
                        if isinstance(value, (list, np.ndarray)):
                            grp.create_dataset(key, data=np.array(value), compression='gzip', compression_opts=4)
                        elif isinstance(value, (int, float, str)):
                            grp.attrs[key] = value
                        else:
                            grp.attrs[key] = str(value)
                
                f.flush()
            
            return temp_file
            
        except Exception as e:
            print(f"\n⚠ 保存临时文件失败: {e}")
            return None
    
    def _merge_temp_h5_files(self, temp_files: List[Path]):
        """
        合并所有临时HDF5文件到主数据库
        
        Args:
            temp_files: 临时HDF5文件列表
        """
        if not temp_files:
            return
        
        try:
            import h5py
            metadata_file = self.faceset_path / "metadata.h5"
            
            # 关闭当前访问器
            if self._h5_accessor is not None:
                self._h5_accessor.close()
            
            # 打开主文件（追加模式）
            with h5py.File(metadata_file, 'a') as main_f:
                # 逐个合并临时文件
                for temp_file in temp_files:
                    if not temp_file.exists():
                        continue
                    
                    with h5py.File(temp_file, 'r') as temp_f:
                        for key in temp_f.keys():
                            # 如果主文件中已存在，先删除
                            if key in main_f:
                                del main_f[key]
                            
                            # 复制到主文件
                            temp_f.copy(key, main_f)
                    
                    main_f.flush()
            
        except Exception as e:
            print(f"\n⚠ 合并临时文件失败: {e}")
            import traceback
            traceback.print_exc()
    
    def _save_new_features_to_hdf5(self, new_features: Dict[str, Dict]):
        """
        将新计算的特征增量保存到HDF5文件
        
        Args:
            new_features: {filename: {feature_key: feature_value}}
        """
        if not new_features:
            return
        
        try:
            import h5py
            metadata_file = self.faceset_path / "metadata.h5"
            
            # 关闭当前访问器
            if self._h5_accessor is not None:
                self._h5_accessor.close()
            
            # 以追加模式打开HDF5文件
            with h5py.File(metadata_file, 'a') as f:
                for filename, features in new_features.items():
                    safe_name = filename.replace('/', '_SLASH_').replace('\\', '_BSLASH_')
                    
                    # 如果组已存在，更新它；否则创建新组
                    if safe_name in f:
                        grp = f[safe_name]
                    else:
                        grp = f.create_group(safe_name)
                        grp.attrs['__original_filename__'] = filename
                    
                    # 写入新特征
                    for key, value in features.items():
                        # 删除旧数据（如果存在）
                        if key in grp:
                            del grp[key]
                        
                        if isinstance(value, (list, np.ndarray)):
                            grp.create_dataset(key, data=np.array(value), compression='gzip', compression_opts=4)
                        elif isinstance(value, (int, float, str)):
                            grp.attrs[key] = value
                        else:
                            grp.attrs[key] = str(value)
                
                # 强制flush确保数据写入磁盘
                f.flush()
            
        except Exception as e:
            print(f"\n⚠ 保存新特征失败: {e}")
            import traceback
            traceback.print_exc()
    
    @property
    def image_files(self) -> List[Path]:
        """获取图像文件列表（父类属性的别名）"""
        return self._image_files
    
    def sort_by_method(self, method: str, **kwargs):
        """
        根据指定方法排序
        
        Args:
            method: 排序方法
            **kwargs: 额外参数
            
        Returns:
            排序后的列表 [(filename, value), ...]
        """
        if method not in SortMethod.ALL_METHODS:
            raise ValueError(f"Unsupported sort method: {method}")
        
        print(S('SORTER_SORTING_BY', method))
        
        # 调用对应的排序方法
        if method == SortMethod.PHASH:
            return self._sort_by_phash(**kwargs)
        elif method == SortMethod.HIST:
            return self._sort_by_hist(**kwargs)
        elif method == SortMethod.BLUR:
            return self._sort_by_blur(**kwargs)
        elif method == SortMethod.FACE_POSE:
            return self._sort_by_face_pose(**kwargs)
        elif method == SortMethod.RESOLUTION:
            return self._sort_by_resolution(**kwargs)
        elif method == SortMethod.COLOR:
            return self._sort_by_color(**kwargs)
        elif method == SortMethod.NAME:
            return self._sort_by_name(**kwargs)
        else:
            raise NotImplementedError(f"Sort method '{method}' not implemented yet")
    
    def _sort_by_phash(self, workers: int = None) -> List[Tuple[str, int]]:
        """
        按感知哈希排序（相似的图片排在一起）
        优先从元数据中读取，缺失的才重新计算
        
        Args:
            workers: 工作进程数
            
        Returns:
            排序后的列表
        """
        if workers is None:
            workers = cpu_count()
        
        print(S('SORTER_CALCULATING_PHASH', len(self.image_files)))
        
        # 第一步：从元数据中收集已有的 phash 值
        cached_results = []
        missing_files = []
        
        for img_path in self.image_files:
            filename = img_path.name
            # 先检查 phash 是否存在（HDF5 组一次访问）
            if not self.has_fields(filename, {'phash'}).get('phash'):
                missing_files.append(img_path)
                continue
            phash_value = self.get_field(filename, 'phash')
            # 元数据中有 phash，直接使用
            try:
                if isinstance(phash_value, list):
                    phash_int = int(''.join(str(b) for b in phash_value), 2)
                elif isinstance(phash_value, str):
                    phash_int = int(phash_value, 16)
                else:
                    phash_int = int(phash_value)
                cached_results.append((filename, phash_int))
            except Exception as e:
                print(f"Warning: Invalid phash value for {filename}: {e}")
                missing_files.append(img_path)
        
        print(f"✓ From metadata: {len(cached_results)}, Need to calculate: {len(missing_files)}")
        
        # 第二步：计算缺失的 phash 值（每10000个保存到临时文件，最后合并）
        newly_calculated = []
        save_interval = 10000
        last_save_count = 0
        temp_h5_files = []  # 记录临时HDF5文件
        
        if missing_files:
            tasks = [(str(img_path),) for img_path in missing_files]
            
            with Pool(processes=workers) as pool:
                pbar = tqdm.tqdm(
                    pool.imap_unordered(_calculate_phash_worker, tasks),
                    total=len(tasks),
                    desc="Calculating missing phash",
                    unit="img"
                )
                
                for result in pbar:
                    if result[1] is not None:
                        newly_calculated.append(result)
                        
                        # 每计算save_interval个就保存到临时文件
                        if len(newly_calculated) - last_save_count >= save_interval:
                            batch_features = {}
                            for filename, phash_value in newly_calculated[last_save_count:]:
                                phash_hex = format(phash_value, 'x')
                                batch_features[filename] = {'phash': phash_hex}
                            
                            # 保存到临时HDF5文件
                            temp_file = self._save_to_temp_h5(batch_features, len(temp_h5_files))
                            if temp_file:
                                temp_h5_files.append(temp_file)
                            
                            last_save_count = len(newly_calculated)
                            pbar.set_description(f"Calculating missing phash [{last_save_count} saved to temp]")
            
            print(f"✓ Newly calculated: {len(newly_calculated)}")
            
            # 保存剩余的数据到临时文件
            if len(newly_calculated) > last_save_count:
                batch_features = {}
                for filename, phash_value in newly_calculated[last_save_count:]:
                    phash_hex = format(phash_value, 'x')
                    batch_features[filename] = {'phash': phash_hex}
                temp_file = self._save_to_temp_h5(batch_features, len(temp_h5_files))
                if temp_file:
                    temp_h5_files.append(temp_file)
            
            # 合并所有临时文件到主HDF5
            if temp_h5_files:
                print(f"\n  正在合并 {len(temp_h5_files)} 个临时文件到主数据库...")
                self._merge_temp_h5_files(temp_h5_files)
                print(f"  ✓ 合并完成")
                
                # 清理临时文件
                for temp_file in temp_h5_files:
                    try:
                        temp_file.unlink()
                    except:
                        pass
        
        # 第三步：合并结果
        all_results = cached_results + newly_calculated
        
        if not all_results:
            return [], {}
        
        print(S('SORTER_SORTING_SIMILARITY'))
        
        # 第四步：按 phash 整数值排序（phash 相似 → 数值相近）
        # 相比分块贪心算法：O(n log n)、确定性、没有末尾退化问题
        print(f"  总图片数: {len(all_results)}")
        print(f"  使用确定性排序（按 phash 值）...")

        all_results.sort(key=lambda x: x[1])  # 按 phash int 升序排列

        sorted_results = all_results

        print(f"✓ 排序完成，共 {len(sorted_results)} 张图片")

        # 构建新特征数据（将新计算的 phash 保存到元数据）
        new_features = {}
        for filename, phash_value in newly_calculated:
            # 将整数 phash 转换为十六进制字符串
            phash_hex = format(phash_value, 'x')
            new_features[filename] = {'phash': phash_hex}

        return sorted_results, new_features
    
    def _sort_by_hist(self, workers: int = None) -> List[Tuple[str, Dict]]:
        """
        按直方图相似度排序（相似的图片排在一起）
        优先从元数据中读取，缺失的才重新计算
        
        Args:
            workers: 工作进程数
            
        Returns:
            排序后的列表
        """
        if workers is None:
            workers = cpu_count()
        
        print(S('SORTER_CALCULATING_HIST', len(self.image_files)))
        
        # 第一步：从元数据中收集已有的 histogram 值
        cached_results = []
        missing_files = []
        
        for img_path in self.image_files:
            filename = img_path.name
            # 批量检查 histogram 字段（一次 HDF5 组访问替代两次 get_field）
            hist_keys = self.has_fields(filename, {'histogram_rgb', 'histogram'})
            if hist_keys.get('histogram_rgb'):
                hist_data = self.get_field(filename, 'histogram_rgb')
            elif hist_keys.get('histogram'):
                hist_data = self.get_field(filename, 'histogram')
            else:
                hist_data = None
            if hist_data is not None:
                # 如果 hist_data 是字符串，尝试解析为字典
                if isinstance(hist_data, str):
                    try:
                        import json
                        hist_data = json.loads(hist_data)
                    except:
                        # 解析失败，跳过该文件
                        missing_files.append(img_path)
                        continue
                
                cached_results.append((filename, hist_data))
            else:
                missing_files.append(img_path)
        
        print(f"✓ From metadata: {len(cached_results)}, Need to calculate: {len(missing_files)}")
        
        # 第二步：计算缺失的直方图（每10000个保存到临时文件，最后合并）
        newly_calculated = []
        save_interval = 10000
        last_save_count = 0
        temp_h5_files = []
        
        if missing_files:
            tasks = [(str(img_path),) for img_path in missing_files]
            
            with Pool(processes=workers) as pool:
                pbar = tqdm.tqdm(
                    pool.imap_unordered(_calculate_histogram_worker, tasks),
                    total=len(tasks),
                    desc="Calculating missing histogram",
                    unit="img"
                )
                
                for result in pbar:
                    if result[1] is not None:
                        newly_calculated.append(result)
                        
                        # 每计算save_interval个就保存到临时文件
                        if len(newly_calculated) - last_save_count >= save_interval:
                            batch_features = {}
                            for filename, hist_data in newly_calculated[last_save_count:]:
                                batch_features[filename] = {'histogram': hist_data}
                            
                            temp_file = self._save_to_temp_h5(batch_features, len(temp_h5_files))
                            if temp_file:
                                temp_h5_files.append(temp_file)
                            
                            last_save_count = len(newly_calculated)
                            pbar.set_description(f"Calculating missing histogram [{last_save_count} saved to temp]")
            
            print(f"✓ Newly calculated: {len(newly_calculated)}")
            
            # 保存剩余的数据到临时文件
            if len(newly_calculated) > last_save_count:
                batch_features = {}
                for filename, hist_data in newly_calculated[last_save_count:]:
                    batch_features[filename] = {'histogram': hist_data}
                temp_file = self._save_to_temp_h5(batch_features, len(temp_h5_files))
                if temp_file:
                    temp_h5_files.append(temp_file)
            
            # 合并所有临时文件到主HDF5
            if temp_h5_files:
                print(f"\n  正在合并 {len(temp_h5_files)} 个临时文件到主数据库...")
                self._merge_temp_h5_files(temp_h5_files)
                print(f"  ✓ 合并完成")
                
                # 清理临时文件
                for temp_file in temp_h5_files:
                    try:
                        temp_file.unlink()
                    except:
                        pass
        
        # 第三步：合并结果
        all_results = cached_results + newly_calculated
        
        if not all_results:
            return [], {}
        
        print(S('SORTER_SORTING_BY_HIST'))

        print(f"  总图片数: {len(all_results)}")
        print(f"  使用贪心最近邻算法（原版 DFL 逻辑）...")

        # 还原原版 DFL hist 排序：O(n²) 贪心最近邻
        # 使用 cv2.compareHist HISTCMP_BHATTACHARYYA 逐通道对比
        file_list = [r[0] for r in all_results]
        # 预构建直方图列表，避免循环中重复字典查找
        hist_list = []
        for _, hdata in all_results:
            hist_list.append([
                np.array(hdata['b'], dtype=np.float32),
                np.array(hdata['g'], dtype=np.float32),
                np.array(hdata['r'], dtype=np.float32),
            ])

        n_total = len(hist_list)
        result_idx = [0]
        remaining_mask = [True] * n_total
        remaining_mask[0] = False
        current = 0

        for _ in tqdm.tqdm(range(n_total - 1), desc="Sorting",
                      unit="img", ascii=True):
            h_b, h_g, h_r = hist_list[current]
            best_score = float('inf')
            best_i = -1
            # 在剩余中找最相似的
            for i in range(n_total):
                if not remaining_mask[i]:
                    continue
                r_b, r_g, r_r = hist_list[i]
                score = (cv2.compareHist(h_b, r_b, cv2.HISTCMP_BHATTACHARYYA) +
                         cv2.compareHist(h_g, r_g, cv2.HISTCMP_BHATTACHARYYA) +
                         cv2.compareHist(h_r, r_r, cv2.HISTCMP_BHATTACHARYYA))
                if score < best_score:
                    best_score = score
                    best_i = i
            result_idx.append(best_i)
            remaining_mask[best_i] = False
            current = best_i

        sorted_results = [(file_list[i], all_results[i][1]) for i in result_idx]

        print(f"✓ 排序完成，共 {len(sorted_results)} 张图片")

        # 构建新特征数据
        new_features = {}
        for filename, hist_data in newly_calculated:
            new_features[filename] = {'histogram': hist_data}

        return sorted_results, new_features
    
    def _sort_by_blur(self, workers: int = None, use_motion_blur: bool = False,
                      train_res: int = None) -> List[Tuple[str, float]]:
        """
        按清晰度排序：**分数越高 = 越清晰 = 序号越靠前**（重命名后 sorted_00000 最清晰）。

        指标 = E(b0)/E(b2) 高频/低频能量比，与 Filter.face_sharpness_score 是同一份实现。
        优先读元数据缓存，缺失的才重算 —— 第二次跑几乎是瞬间完成。

        Args:
            workers: 工作进程数
            use_motion_blur: 【已废弃】新指标对运动模糊同样敏感（运动模糊也是一条
                低通），不再需要单独的检测分支；保留参数只为兼容旧调用。
            train_res: 归一化分辨率，应与训练分辨率一致（默认 256）。
                字段名按分辨率区分，所以改分辨率不会和旧缓存混用。

        Returns:
            排序后的列表（从高到低）
        """
        if workers is None:
            workers = cpu_count()
        if train_res is None:
            train_res = _SHARP_TRAIN_RES
        field = _sharp_field(train_res)
        
        print(S('SORTER_CALCULATING_BLUR', len(self.image_files)))
        
        # 第一步：从元数据中收集已有的清晰度值
        cached_results = []
        missing_files = []
        
        for img_path in self.image_files:
            filename = img_path.name
            # ⚠️ 只读新指标字段。旧的 'sharpness'/'blur' 是拉普拉斯方差，量纲不同，
            #    混用会让一部分图用旧尺子、一部分用新尺子，排出来的顺序毫无意义。
            blur_value = self.get_field(filename, field)
            landmarks = self.get_field(filename, 'landmarks')
            
            if blur_value is not None:
                cached_results.append((filename, float(blur_value)))
            else:
                missing_files.append((img_path, landmarks))
        
        print(f"✓ From metadata [{field}]: {len(cached_results)}, Need to calculate: {len(missing_files)}")
        if missing_files and len(cached_results) == 0:
            print("  （首次跑该指标，需要全量计算；结果会写入元数据，之后直接读缓存）")
        
        # 第二步：计算缺失的模糊度（每10000个保存一次）
        newly_calculated = []
        save_interval = 10000
        last_save_count = 0
        
        if missing_files:
            tasks = [(str(img_path), landmarks, use_motion_blur, train_res)
                     for img_path, landmarks in missing_files]
            
            with Pool(processes=workers) as pool:
                pbar = tqdm.tqdm(
                    pool.imap_unordered(_calculate_blur_worker, tasks),
                    total=len(tasks),
                    desc="Calculating missing blur",
                    unit="img"
                )
                
                for result in pbar:
                    if result[1] > 0:
                        newly_calculated.append(result)
                        
                        # 每计算save_interval个就保存一次
                        if len(newly_calculated) - last_save_count >= save_interval:
                            batch_features = {}
                            for filename, blur_value in newly_calculated[last_save_count:]:
                                batch_features[filename] = {field: blur_value}
                            self._save_new_features_to_hdf5(batch_features)
                            last_save_count = len(newly_calculated)
                            pbar.set_description(f"Calculating missing blur [{last_save_count} saved]")
            
            print(f"✓ Newly calculated: {len(newly_calculated)}")
            
            # 保存剩余的数据
            if len(newly_calculated) > last_save_count:
                batch_features = {}
                for filename, blur_value in newly_calculated[last_save_count:]:
                    batch_features[filename] = {field: blur_value}
                self._save_new_features_to_hdf5(batch_features)
        
        # 第三步：合并结果
        all_results = cached_results + newly_calculated
        
        if not all_results:
            return [], {}
        
        print(S('SORTER_SORTING_BY_BLUR'))
        
        # 第四步：按模糊度从高到低排序
        sorted_results = sorted(all_results, key=operator.itemgetter(1), reverse=True)
        
        # 构建新特征数据
        new_features = {}
        for filename, blur_value in newly_calculated:
            new_features[filename] = {field: blur_value}
        
        return sorted_results, new_features
    
    def _sort_by_face_pose(self, workers: int = None, pose_type: str = 'yaw') -> List[Tuple[str, float]]:
        """
        按人脸姿态排序
        优先从元数据中读取，缺失的才重新计算
        
        Args:
            workers: 工作进程数
            pose_type: 姿态类型 ('pitch', 'yaw', 'roll')
            
        Returns:
            排序后的列表
        """
        if workers is None:
            workers = cpu_count()
        
        print(S('SORTER_CALCULATING_POSE', len(self.image_files)))
        
        # 第一步：从元数据中收集已有的 pose 值
        cached_results = []
        missing_files = []
        
        for img_path in self.image_files:
            filename = img_path.name
            # 批量检查所有 pose 相关字段（一次 HDF5 组访问）
            pose_fields = self.has_fields(filename, {'pose', 'pitch', 'yaw', 'roll', 'landmarks'})
            pose_data = None
            if pose_fields.get('pose'):
                pose_data = self.get_field(filename, 'pose')
            elif pose_fields.get('pitch') or pose_fields.get('yaw') or pose_fields.get('roll'):
                pose_data = {}
                for k in ('pitch', 'yaw', 'roll'):
                    if pose_fields.get(k):
                        pose_data[k] = self.get_field(filename, k)
            landmarks = self.get_field(filename, 'landmarks') if pose_fields.get('landmarks') else None
            
            if pose_data and pose_type in pose_data:
                cached_results.append((filename, float(pose_data[pose_type])))
            else:
                missing_files.append((img_path, landmarks))
        
        print(f"✓ From metadata: {len(cached_results)}, Need to calculate: {len(missing_files)}")
        
        # 第二步：计算缺失的姿态（每10000个保存一次）
        newly_calculated = []
        save_interval = 10000
        last_save_count = 0
        
        if missing_files:
            tasks = [(str(img_path), landmarks) for img_path, landmarks in missing_files]
            
            with Pool(processes=workers) as pool:
                pbar = tqdm.tqdm(
                    pool.imap_unordered(_calculate_face_pose_worker, tasks),
                    total=len(tasks),
                    desc="Calculating missing pose",
                    unit="img"
                )
                
                for result in pbar:
                    if result[1] is not None:
                        newly_calculated.append((result[0], result[1].get(pose_type, 0.0)))
                        
                        # 每计算save_interval个就保存一次
                        if len(newly_calculated) - last_save_count >= save_interval:
                            batch_features = {}
                            for filename, pose_value in newly_calculated[last_save_count:]:
                                batch_features[filename] = {'pose': {pose_type: pose_value}}
                            self._save_new_features_to_hdf5(batch_features)
                            last_save_count = len(newly_calculated)
                            pbar.set_description(f"Calculating missing pose [{last_save_count} saved]")
            
            print(f"✓ Newly calculated: {len(newly_calculated)}")
            
            # 保存剩余的数据
            if len(newly_calculated) > last_save_count:
                batch_features = {}
                for filename, pose_value in newly_calculated[last_save_count:]:
                    batch_features[filename] = {'pose': {pose_type: pose_value}}
                self._save_new_features_to_hdf5(batch_features)
        
        # 第三步：合并结果
        all_results = cached_results + newly_calculated
        
        if not all_results:
            return [], {}
        
        print(S('SORTER_SORTING_BY_POSE', pose_type))
        
        # 第四步：按姿态值排序
        sorted_results = sorted(all_results, key=operator.itemgetter(1))
        
        # 构建新特征数据
        new_features = {}
        for filename, pose_value in newly_calculated:
            # 需要重新获取完整的 pose 数据
            # 这里简化处理，只保存当前姿态类型
            new_features[filename] = {'pose': {pose_type: pose_value}}
        
        return sorted_results, new_features
    
    def _sort_by_resolution(self, workers: int = None) -> List[Tuple[str, int]]:
        """
        按分辨率排序
        优先从元数据中读取，缺失的才重新计算
        
        Args:
            workers: 工作进程数
            
        Returns:
            排序后的列表（从低到高）
        """
        if workers is None:
            workers = cpu_count()
        
        print(S('SORTER_CALCULATING_RESOLUTION', len(self.image_files)))
        
        # 第一步：从元数据中收集已有的 resolution 值
        cached_results = []
        missing_files = []
        
        for img_path in self.image_files:
            filename = img_path.name
            resolution = self.get_field(filename, 'resolution')
            
            if resolution is not None:
                cached_results.append((filename, int(resolution)))
            else:
                missing_files.append(img_path)
        
        print(f"✓ From metadata: {len(cached_results)}, Need to calculate: {len(missing_files)}")
        
        # 第二步：计算缺失的分辨率（每10000个保存一次）
        newly_calculated = []
        save_interval = 10000
        last_save_count = 0
        
        if missing_files:
            tasks = [(str(img_path),) for img_path in missing_files]
            
            with Pool(processes=workers) as pool:
                pbar = tqdm.tqdm(
                    pool.imap_unordered(_calculate_resolution_worker, tasks),
                    total=len(tasks),
                    desc="Calculating missing resolution",
                    unit="img"
                )
                
                for result in pbar:
                    if result[1] > 0:
                        newly_calculated.append(result)
                        
                        # 每计算save_interval个就保存一次
                        if len(newly_calculated) - last_save_count >= save_interval:
                            batch_features = {}
                            for filename, resolution in newly_calculated[last_save_count:]:
                                batch_features[filename] = {'resolution': resolution}
                            self._save_new_features_to_hdf5(batch_features)
                            last_save_count = len(newly_calculated)
                            pbar.set_description(f"Calculating missing resolution [{last_save_count} saved]")
            
            print(f"✓ Newly calculated: {len(newly_calculated)}")
            
            # 保存剩余的数据
            if len(newly_calculated) > last_save_count:
                batch_features = {}
                for filename, resolution in newly_calculated[last_save_count:]:
                    batch_features[filename] = {'resolution': resolution}
                self._save_new_features_to_hdf5(batch_features)
        
        # 第三步：合并结果
        all_results = cached_results + newly_calculated
        
        if not all_results:
            return [], {}
        
        print(S('SORTER_SORTING_BY_RESOLUTION'))
        
        # 第四步：按分辨率从低到高排序
        sorted_results = sorted(all_results, key=operator.itemgetter(1))
        
        # 构建新特征数据
        new_features = {}
        for filename, resolution in newly_calculated:
            new_features[filename] = {'resolution': resolution}
        
        return sorted_results, new_features
    
    def _sort_by_color(self, workers: int = None) -> List[Tuple[str, float]]:
        """
        按颜色排序（平均色相）
        优先从元数据中读取，缺失的才重新计算
        
        Args:
            workers: 工作进程数
            
        Returns:
            排序后的列表
        """
        if workers is None:
            workers = cpu_count()
        
        print(S('SORTER_CALCULATING_COLOR', len(self.image_files)))
        
        # 第一步：从元数据中收集已有的 histogram 值（用于计算颜色）
        cached_results = []
        missing_files = []
        
        for img_path in self.image_files:
            filename = img_path.name
            # 批量检查 histogram 字段（一次 HDF5 组访问替代两次 get_field）
            hist_keys = self.has_fields(filename, {'histogram_rgb', 'histogram'})
            if hist_keys.get('histogram_rgb'):
                hist_data = self.get_field(filename, 'histogram_rgb')
            elif hist_keys.get('histogram'):
                hist_data = self.get_field(filename, 'histogram')
            else:
                hist_data = None
            if hist_data is not None:
                # 计算平均亮度作为颜色指标
                avg_brightness = (
                    sum(hist_data['r']) + 
                    sum(hist_data['g']) + 
                    sum(hist_data['b'])
                ) / 3
                cached_results.append((filename, avg_brightness))
            else:
                missing_files.append(img_path)
        
        print(f"✓ From metadata: {len(cached_results)}, Need to calculate: {len(missing_files)}")
        
        # 第二步：计算缺失的颜色值
        newly_calculated = []
        if missing_files:
            tasks = [(str(img_path),) for img_path in missing_files]
            
            with Pool(processes=workers) as pool:
                for result in tqdm.tqdm(
                    pool.imap_unordered(_calculate_histogram_worker, tasks),
                    total=len(tasks),
                    desc="Calculating missing color",
                    unit="img"
                ):
                    if result[1] is not None:
                        hist_data = result[1]
                        avg_brightness = (
                            sum(hist_data['r']) + 
                            sum(hist_data['g']) + 
                            sum(hist_data['b'])
                        ) / 3
                        newly_calculated.append((result[0], avg_brightness))
            
            print(f"✓ Newly calculated: {len(newly_calculated)}")
        
        # 第三步：合并结果
        all_results = cached_results + newly_calculated
        
        if not all_results:
            return [], {}
        
        print(S('SORTER_SORTING_BY_COLOR'))
        
        # 第四步：按颜色排序
        sorted_results = sorted(all_results, key=operator.itemgetter(1))
        
        # color 方法不保存新特征，因为它使用的是 histogram 数据
        return sorted_results, {}
    
    def _sort_by_name(self) -> List[Tuple[str, str]]:
        """
        按文件名排序
        
        Returns:
            排序后的列表
        """
        print(S('SORTER_SORTING_BY_NAME'))
        
        results = [(img_path.name, img_path.name) for img_path in self.image_files]
        sorted_results = sorted(results, key=operator.itemgetter(0))
        
        return sorted_results, {}
    
    def rename_sorted_files(self, sorted_list: List[Tuple[str, any]], prefix: str = "sorted", new_features: Dict[str, Dict] = None):
        """
        重命名排序后的文件，并更新元数据库
        使用 序号+16位随机码 作为最终名称，彻底避免冲突
        
        Args:
            sorted_list: 排序后的列表 [(filename, value), ...]
            prefix: 文件名前缀
            new_features: 新计算的特征数据 {filename: {feature_key: feature_value}}
        """
        import random
        import string
        
        print(S('SORTER_RENAMING_FILES', len(sorted_list)))
        
        # 生成16位随机字符串的函数
        def generate_random_suffix(length=16):
            chars = string.ascii_lowercase + string.digits
            return ''.join(random.choices(chars, k=length))
        
        # 第一阶段：收集所有需要重命名的文件
        rename_plan = []  # [(old_path, final_name, index), ...]
        
        for i, (filename, _) in enumerate(sorted_list):
            old_path = self.faceset_path / filename
            
            if not old_path.exists():
                print(f"\n⚠ 跳过 {filename}: 文件不存在")
                continue
            
            # 直接生成最终名称：prefix_序号_16位随机码.扩展名
            random_suffix = generate_random_suffix()
            final_name = f"{prefix}_{i:05d}_{random_suffix}{old_path.suffix}"
            
            rename_plan.append((old_path, final_name, i))
        
        print(f"  计划重命名 {len(rename_plan)} 个文件")
        print(f"  命名格式: {prefix}_序号_16位随机码.jpg (例如: {prefix}_00000_a3f7b9c2d4e5f6g8.jpg)")
        
        # 第二阶段：执行重命名 + 增量保存元数据
        success_count = 0
        failed_count = 0
        failed_files = []
        saved_metadata_count = 0
        
        # 创建备份
        metadata_file = self.faceset_path / "metadata.h5"
        backup_file = self.faceset_path / "metadata_backup.h5"
        
        if metadata_file.exists():
            try:
                import shutil
                shutil.copy2(metadata_file, backup_file)
                print(f"  ✓ 已创建元数据备份: {backup_file.name}")
            except Exception as e:
                print(f"  ⚠ 无法创建备份: {e}")
        
        pbar = tqdm.tqdm(total=len(rename_plan), desc="重命名文件", unit="file", ascii=True)
        
        # 增量元数据字典
        incremental_metadata = {}
        save_interval = 10000
        
        for old_path, final_name, i in rename_plan:
            final_path = self.faceset_path / final_name
            
            try:
                # 直接重命名为最终名称（带随机后缀，不会冲突）
                old_path.rename(final_path)
                success_count += 1
                
                # 构建元数据
                original_filename = old_path.name
                meta = self.get_metadata(original_filename).copy() if self.has_metadata(original_filename) else {}
                
                # 添加新特征
                if new_features and original_filename in new_features:
                    meta.update(new_features[original_filename])
                
                incremental_metadata[final_name] = meta
                saved_metadata_count += 1
                
                # 定期保存元数据
                if saved_metadata_count % save_interval == 0 and incremental_metadata:
                    try:
                        import h5py
                        
                        if self._h5_accessor is not None:
                            self._h5_accessor.close()
                        
                        with h5py.File(metadata_file, 'a') as f:
                            for filename, meta in incremental_metadata.items():
                                safe_name = filename.replace('/', '_SLASH_').replace('\\', '_BSLASH_')
                                
                                # 如果组已存在，先删除（避免重复）
                                if safe_name in f:
                                    del f[safe_name]
                                
                                grp = f.create_group(safe_name)
                                grp.attrs['__original_filename__'] = filename
                                
                                for key, value in meta.items():
                                    if isinstance(value, (list, np.ndarray)):
                                        grp.create_dataset(key, data=np.array(value), compression='gzip', compression_opts=4)
                                    elif isinstance(value, (int, float, str)):
                                        grp.attrs[key] = value
                                    else:
                                        grp.attrs[key] = str(value)
                        
                        incremental_metadata.clear()
                        pbar.set_description(f"重命名文件 [已保存: {saved_metadata_count}]")
                        
                    except Exception as e:
                        print(f"\n⚠ 元数据保存失败: {e}")
                        import traceback
                        traceback.print_exc()
                
                pbar.update(1)
                
            except Exception as e:
                print(f"\n⚠ 重命名失败 {old_path.name}: {e}")
                failed_count += 1
                failed_files.append(old_path.name)
                pbar.update(1)
        
        pbar.close()
        
        print(f"\n✓ 成功重命名: {success_count}, 失败: {failed_count}")
        
        if failed_files:
            print(f"  失败的文件: {', '.join(failed_files[:5])}{'...' if len(failed_files) > 5 else ''}")
        
        # 第三阶段：保存剩余的元数据
        if incremental_metadata:
            print(f"  正在保存剩余的 {len(incremental_metadata)} 个元数据条目...")
            try:
                import h5py
                
                if self._h5_accessor is not None:
                    self._h5_accessor.close()
                
                with h5py.File(metadata_file, 'a') as f:
                    for filename, meta in incremental_metadata.items():
                        safe_name = filename.replace('/', '_SLASH_').replace('\\', '_BSLASH_')
                        
                        if safe_name in f:
                            del f[safe_name]
                        
                        grp = f.create_group(safe_name)
                        grp.attrs['__original_filename__'] = filename
                        
                        for key, value in meta.items():
                            if isinstance(value, (list, np.ndarray)):
                                grp.create_dataset(key, data=np.array(value), compression='gzip', compression_opts=4)
                            elif isinstance(value, (int, float, str)):
                                grp.attrs[key] = value
                            else:
                                grp.attrs[key] = str(value)
                
                print(f"  ✓ 元数据保存完成")
            except Exception as e:
                print(f"\n⚠ 最终元数据保存失败: {e}")
                import traceback
                traceback.print_exc()
        
        # 清理旧文件名的元数据（防止HDF5文件无限增长）
        print(f"  正在清理旧元数据条目...")
        try:
            import h5py
            
            if self._h5_accessor is not None:
                self._h5_accessor.close()
            
            # 收集所有需要保留的新文件名
            new_filenames = set(final_name for _, final_name, _ in rename_plan)
            
            with h5py.File(metadata_file, 'a') as f:
                # 找出所有需要删除的旧条目
                keys_to_delete = []
                for key in f.keys():
                    # 解码key（h5py返回的是bytes或str）
                    key_str = key if isinstance(key, str) else key.decode('utf-8')
                    
                    # 如果不是新文件名，且不是特殊属性，就删除
                    if key_str not in new_filenames and not key_str.startswith('__'):
                        keys_to_delete.append(key_str)
                
                # 删除旧条目
                for key in keys_to_delete:
                    del f[key]
                
                if keys_to_delete:
                    print(f"  ✓ 已清理 {len(keys_to_delete)} 个旧元数据条目")
                else:
                    print(f"  ✓ 无需清理")
                    
        except Exception as e:
            print(f"\n⚠ 清理旧元数据失败: {e}")
            # 不中断流程，继续执行
        
        # 第四阶段：删除备份（如果一切成功）
        if backup_file.exists() and failed_count == 0:
            try:
                backup_file.unlink()
                print(f"  ✓ 已删除备份文件（重命名全部成功）")
            except:
                pass
        elif backup_file.exists():
            print(f"  ⚠ 保留备份文件以备恢复: {backup_file.name}")
        
        print(S('SORTER_RENAME_COMPLETE'))


def main():
    """主函数 - 命令行入口"""
    parser = argparse.ArgumentParser(description='Faceset Sorter - 人脸集排序工具')
    parser.add_argument('--input', '-i', type=str, required=True, help='人脸集路径')
    parser.add_argument('--method', '-m', type=str, default='phash',
                       choices=SortMethod.ALL_METHODS,
                       help='排序方法')
    parser.add_argument('--rename', action='store_true', help='重命名排序后的文件')
    parser.add_argument('--prefix', type=str, default='sorted', help='重命名前缀')
    parser.add_argument('--workers', type=int, default=None, help='工作进程数')
    parser.add_argument('--pose-type', type=str, default='yaw',
                       choices=['pitch', 'yaw', 'roll'],
                       help='姿态类型（仅用于face_pose排序）')
    parser.add_argument('--motion-blur', action='store_true',
                        help='【已废弃】新指标对运动模糊同样敏感，此开关不再有作用')
    parser.add_argument('--train-res', type=int, default=256,
                        help='模糊排序：清晰度归一化分辨率，应与训练分辨率一致 (default: 256)')
    parser.add_argument('--reset', action='store_true', help='重置排序：将 sorted_XXXXX_随机码.jpg 还原为 XXXXX.jpg')
    
    args = parser.parse_args()
    
    # 创建排序器
    faceset_path = Path(args.input)
    if not faceset_path.exists():
        print(f"Error: Path not found: {faceset_path}")
        sys.exit(1)
    
    sorter = FaceSorter(faceset_path)
    
    # 如果指定了 --reset，执行重置操作
    if args.reset:
        import re
        
        prefix = args.prefix
        pattern = re.compile(rf'^{re.escape(prefix)}_(\d{{5}})_[a-z0-9]{{16}}(.+)$')
        
        sorted_files = [
            f for f in faceset_path.iterdir()
            if f.is_file() and pattern.match(f.name)
        ]
        
        if not sorted_files:
            print(f"未发现 {prefix}_XXXXX_随机码 格式的文件")
            sys.exit(0)
        
        print(f"发现 {len(sorted_files)} 个已排序的文件，开始重置...")
        print(f"  将还原为: XXXXX.jpg 格式（按序号重命名）")
        
        success_count = 0
        failed_count = 0
        metadata_updates = {}  # 记录需要更新的元数据 {新文件名: 元数据}
        
        pbar = tqdm.tqdm(total=len(sorted_files), desc="重置文件", unit="file", ascii=True)
        
        for sorted_file in sorted_files:
            match = pattern.match(sorted_file.name)
            if match:
                index = match.group(1)
                ext = match.group(2)
                new_name = f"{index}{ext}"
                new_path = faceset_path / new_name
                
                try:
                    # 如果目标文件已存在，先删除
                    if new_path.exists():
                        new_path.unlink()
                    
                    # 从HDF5读取旧文件名的元数据
                    old_meta = sorter.get_metadata(sorted_file.name)
                    if old_meta:
                        metadata_updates[new_name] = old_meta
                    
                    sorted_file.rename(new_path)
                    success_count += 1
                    pbar.update(1)
                except Exception as e:
                    print(f"\n⚠ 重置失败 {sorted_file.name}: {e}")
                    failed_count += 1
                    pbar.update(1)
            else:
                pbar.update(1)
        
        pbar.close()
        
        # 更新元数据库
        if metadata_updates:
            print(f"\n  正在更新元数据库 ({len(metadata_updates)} 条)...")
            try:
                import h5py
                metadata_file = faceset_path / "metadata.h5"
                
                if sorter._h5_accessor is not None:
                    sorter._h5_accessor.close()
                
                with h5py.File(metadata_file, 'a') as f:
                    # 第1步：删除所有 sorted_ 开头的旧条目
                    keys_to_delete = []
                    for key in f.keys():
                        key_str = key if isinstance(key, str) else key.decode('utf-8')
                        if key_str.startswith('sorted_'):
                            keys_to_delete.append(key_str)
                    
                    for key in keys_to_delete:
                        del f[key]
                    
                    if keys_to_delete:
                        print(f"    已删除 {len(keys_to_delete)} 个旧条目")
                    
                    # 第2步：创建新条目
                    for new_name, meta in metadata_updates.items():
                        safe_name = new_name.replace('/', '_SLASH_').replace('\\', '_BSLASH_')
                        
                        grp = f.create_group(safe_name)
                        grp.attrs['__original_filename__'] = new_name
                        
                        for key, value in meta.items():
                            if isinstance(value, (list, np.ndarray)):
                                grp.create_dataset(key, data=np.array(value), compression='gzip', compression_opts=4)
                            elif isinstance(value, (int, float, str)):
                                grp.attrs[key] = value
                            else:
                                grp.attrs[key] = str(value)
                
                print(f"  ✓ 元数据库已更新")
            except Exception as e:
                print(f"\n⚠ 元数据更新失败: {e}")
                import traceback
                traceback.print_exc()
        
        print(f"\n✓ 重置完成: 成功 {success_count}, 失败 {failed_count}")
        print(f"  文件已按序号重命名为: 00000.jpg, 00001.jpg, ...")
        print("\nDone!")
        sys.exit(0)
    
    # 执行排序
    kwargs = {'workers': args.workers}
    if args.method == SortMethod.FACE_POSE:
        kwargs['pose_type'] = args.pose_type
    elif args.method == SortMethod.BLUR:
        kwargs['use_motion_blur'] = args.motion_blur
        kwargs['train_res'] = args.train_res
    
    sorted_list = sorter.sort_by_method(args.method, **kwargs)
    
    # 处理返回的元组 (sorted_list, new_features)
    if isinstance(sorted_list, tuple):
        sorted_list, new_features = sorted_list
    else:
        new_features = {}
    
    if not sorted_list:
        print("No images to sort")
        sys.exit(1)
    
    print(f"\nSorted {len(sorted_list)} images by {args.method}")
    
    # 如果需要重命名（始终传递 new_features）
    if args.rename:
        sorter.rename_sorted_files(sorted_list, args.prefix, new_features)
    
    print("\nDone!")


if __name__ == "__main__":
    main()
