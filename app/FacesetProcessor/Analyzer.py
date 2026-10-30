"""
DeepFaceLab Torch - FacesetProcessor Analyzer Module
人脸集分析器模块：计算、校验和丰富元数据
"""

import sys
from pathlib import Path
import cv2
import numpy as np
import json
import traceback
import argparse
import pickle
import struct
from typing import Dict, List, Optional, Tuple
import tqdm
from multiprocessing import Pool, cpu_count

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 导入多语言支持
from strings import S

# 减少ONNX Runtime警告
import onnxruntime
onnxruntime.set_default_logger_severity(3)
import warnings
warnings.filterwarnings('ignore', module='onnxruntime')

# 导入项目模块
from facelib.LandmarksProcessor import estimate_pitch_yaw_roll
import imagehash
from PIL import Image

# 导入流式 HDF5 访问器和基类
from H5StreamingAccessor import H5StreamingAccessor
from FacesetBaseProcessor import FacesetBaseProcessor

# 导入 ArcFace ONNX 提取器 (用于计算 embedding)
try:
    sys.path.insert(0, str(project_root / "FacesetProcessor"))
    from Filter import ArcFaceONNXExtractor
    ARCFACE_AVAILABLE = True
except ImportError:
    ARCFACE_AVAILABLE = False


class MetadataAnalyzer(FacesetBaseProcessor):
    """元数据分析器 - 计算和校验人脸数据集的元数据
    
    继承自 FacesetBaseProcessor，自动获得：
    - HDF5 流式访问能力
    - 图像文件扫描
    - 元数据缓存管理
    - 完整性校验
    """
    
    # 定义可用的分析特征
    AVAILABLE_FEATURES = ['phash', 'histogram', 'hue', 'pose', 'embedding', 'landmark']
    
    def __init__(self, faceset_path: Path, metadata_file: Optional[Path] = None, 
                 features: Optional[List[str]] = None):
        """
        初始化分析器
        
        Args:
            faceset_path: 人脸数据集路径
            metadata_file: 元数据文件路径（可选，默认为 faceset_path/metadata.h5）
            features: 要分析的特征列表（可选，默认为全部）
        """
        # 先调用父类构造函数，它会初始化 _image_files 和 _h5_accessor
        super().__init__(faceset_path, metadata_file)
        
        self.arcface_extractor = None
        
        # 流式 HDF5 访问器已在父类中初始化，这里只需要缓存字典
        self._metadata_cache = {}  # 缓存已访问的元数据
        
        # 设置要分析的特征
        if features is None or 'all' in features:
            self.features = self.AVAILABLE_FEATURES.copy()
        else:
            # 验证特征选项
            valid_features = []
            for f in features:
                if f in self.AVAILABLE_FEATURES:
                    valid_features.append(f)
                else:
                    print(S('ANALYZER_INVALID_FEATURE', f))
            self.features = valid_features if valid_features else self.AVAILABLE_FEATURES.copy()
        
        print(S('ANALYZER_SELECTED_FEATURES', ', '.join(self.features)))
        print(f"\n实际将计算的特征: {self.features}")
        
        # 初始化 ArcFace 提取器 (用于计算 embedding)
        if 'embedding' in self.features and ARCFACE_AVAILABLE:
            try:
                model_dir = project_root / "modelhub" / "onnx" / "ArcFace"
                if model_dir.exists():
                    print(f"\n正在初始化 ArcFace 提取器...")
                    print(f"  模型目录: {model_dir}")

                    # 列出可用的模型文件
                    onnx_files = list(model_dir.glob("*.onnx"))
                    print(f"  找到的 ONNX 模型: {[f.name for f in onnx_files]}")
                    
                    self.arcface_extractor = ArcFaceONNXExtractor(model_dir, model_name="w600k_mbf")
                    print(f"✓ ArcFace 提取器初始化成功: {self.arcface_extractor.model_name_used}")
                    
                    # 测试提取功能
                    print(f"  测试提取功能...")
                    test_img = np.zeros((112, 112, 3), dtype=np.uint8)
                    test_emb = self.arcface_extractor.extract_embedding(test_img)
                    if test_emb is not None:
                        print(f"  ✓ 测试成功: embedding 维度 = {len(test_emb)}\n")
                    else:
                        print(f"  ⚠ 警告: 测试返回 None\n")
                else:
                    print(f"\n✗ 模型目录不存在: {model_dir}\n")
            except Exception as e:
                print(f"\n✗ Failed to initialize ArcFace extractor: {e}\n")
                import traceback
                traceback.print_exc()
        elif 'embedding' in self.features:
            print(f"\n⚠ Warning: ARCFACE_AVAILABLE = {ARCFACE_AVAILABLE}")
            print(f"  → 无法初始化 ArcFace 提取器\n")
        
        # 初始化 InsightFace 106pt 标记器 (用于 landmark 模式)
        self.landmarker = None
        if 'landmark' in self.features:
            try:
                from modelhub.onnx import InsightFace2D106
                from xlib.onnxruntime import get_available_devices_info, get_cpu_device_info
                
                print(f"\n正在初始化 InsightFace 106pt 标记器...")
                
                # 检测设备（优先级：CUDA > DX12 > CPU）
                devices = get_available_devices_info()
                device_info = None
                
                # 优先级1: CUDA
                for device in devices:
                    if 'CUDA' in str(device).upper() or 'cuda' in str(device).lower():
                        device_info = device
                        break
                
                # 优先级2: DirectML/DX12
                if device_info is None:
                    for device in devices:
                        if 'DML' in str(device).upper() or 'directml' in str(device).lower() or 'dx12' in str(device).lower():
                            device_info = device
                            break
                
                # 优先级3: CPU
                if device_info is None:
                    device_info = get_cpu_device_info()
                
                print(f"  使用设备: {device_info}")
                self.landmarker = InsightFace2D106(device_info)
                print(f"✓ InsightFace 106pt 标记器初始化成功\n")
            except Exception as e:
                print(f"\n✗ Failed to initialize InsightFace 106pt landmarker: {e}\n")
                import traceback
                traceback.print_exc()
    
    def _get_metadata_streaming(self, filename: str) -> Dict:
        """
        流式获取单个文件的元数据（带缓存）
        这是父类 get_metadata() 的别名，保持向后兼容
        
        Args:
            filename: 文件名
            
        Returns:
            元数据字典
        """
        return self.get_metadata(filename)
    
    @property
    def image_files(self) -> List[Path]:
        """获取图像文件列表（父类属性的别名）"""
        return self._image_files
    
    def _scan_image_files(self):
        """扫描人脸数据集中的图像文件"""
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
        self._image_files = sorted([
            f for f in self.faceset_path.iterdir()
            if f.suffix.lower() in image_extensions and f.is_file()
        ])
        print(S('ANALYZER_FOUND_IMAGES', len(self.image_files)))
    
    def verify_database_integrity(self) -> Dict:
        """
        校验数据库完整性（使用父类的实现）
        
        Returns:
            校验结果字典
        """
        return super().verify_database_integrity()
    
    def calculate_phash(self, image_path: Path) -> str:
        """
        计算图像的感知哈希值
        
        Args:
            image_path: 图像文件路径
            
        Returns:
            感知哈希字符串
        """
        try:
            image = Image.open(image_path)
            phash = imagehash.phash(image)
            return str(phash)
        except Exception as e:
            print(S('ANALYZER_PHASH_ERROR', image_path.name, e))
            return ""
    
    def calculate_histogram(self, image: np.ndarray) -> Dict[str, List[float]]:
        """
        计算图像的RGB和HSV直方图分布
        
        Args:
            image: BGR格式图像 (numpy array)
            
        Returns:
            直方图统计字典
        """
        try:
            # RGB直方图
            hist_b = cv2.calcHist([image], [0], None, [256], [0, 256]).flatten()
            hist_g = cv2.calcHist([image], [1], None, [256], [0, 256]).flatten()
            hist_r = cv2.calcHist([image], [2], None, [256], [0, 256]).flatten()
            
            # 归一化
            hist_b = (hist_b / hist_b.sum()).tolist()
            hist_g = (hist_g / hist_g.sum()).tolist()
            hist_r = (hist_r / hist_r.sum()).tolist()
            
            # HSV直方图
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            hist_h = cv2.calcHist([hsv], [0], None, [180], [0, 180]).flatten()
            hist_s = cv2.calcHist([hsv], [1], None, [256], [0, 256]).flatten()
            hist_v = cv2.calcHist([hsv], [2], None, [256], [0, 256]).flatten()
            
            hist_h = (hist_h / hist_h.sum()).tolist()
            hist_s = (hist_s / hist_s.sum()).tolist()
            hist_v = (hist_v / hist_v.sum()).tolist()
            
            return {
                'histogram_rgb': {
                    'b': hist_b,
                    'g': hist_g,
                    'r': hist_r
                },
                'histogram_hsv': {
                    'h': hist_h,
                    's': hist_s,
                    'v': hist_v
                }
            }
        except Exception as e:
            print(S('ANALYZER_HISTOGRAM_ERROR', e))
            return {}
    
    def calculate_hue_distribution(self, image: np.ndarray) -> Dict[str, float]:
        """
        计算色相分布统计
        
        Args:
            image: BGR格式图像
            
        Returns:
            色相分布统计（均值、标准差、主色相）
        """
        try:
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            hue = hsv[:, :, 0].astype(np.float32)
            
            # 基本统计
            hue_mean = float(np.mean(hue))
            hue_std = float(np.std(hue))
            
            return {
                'hue_mean': hue_mean,
                'hue_std': hue_std
            }
        except Exception as e:
            print(S('ANALYZER_HUE_ERROR', e))
            return {}
    
    def extract_face_embedding(self, image: np.ndarray) -> Optional[List[float]]:
        """
        提取人脸 embedding (使用 ArcFace)
        
        Args:
            image: BGR格式图像 (已对齐的人脸)
            
        Returns:
            embedding 向量列表 (128维)，失败返回 None
        """
        if self.arcface_extractor is None:
            return None
        
        try:
            embedding = self.arcface_extractor.extract_embedding(image)
            if embedding is not None:
                return embedding.tolist()
            return None
        except Exception as e:
            return None
    
    def estimate_face_pose(self, landmarks: List[List[float]], image_size: int = 256) -> Dict[str, float]:
        """
        估算人脸姿态角度（yaw, pitch, roll）
        
        Args:
            landmarks: 68个特征点坐标列表
            image_size: 图像尺寸
            
        Returns:
            姿态角度字典（弧度）
        """
        try:
            landmarks_array = np.array(landmarks, dtype=np.float32)
            pitch, yaw, roll = estimate_pitch_yaw_roll(landmarks_array, size=image_size)
            
            return {
                'pitch': float(pitch),
                'yaw': float(yaw),
                'roll': float(roll)
            }
        except Exception as e:
            print(S('ANALYZER_POSE_ERROR', e))
            return {
                'pitch': 0.0,
                'yaw': 0.0,
                'roll': 0.0
            }
    
    def analyze_single_image(self, filename: str) -> Dict:
        """
        分析单张图像，根据选择的特征计算
        
        Args:
            filename: 图像文件名
            
        Returns:
            完整的元数据字典
        """
        image_path = self.faceset_path / filename
        
        # 读取图像
        image = cv2.imread(str(image_path))
        if image is None:
            print(S('ANALYZER_READ_ERROR', filename))
            return {}
        
        # 获取现有元数据（流式读取）
        existing_meta = self._get_metadata_streaming(filename)
        
        # 根据选择的特征进行分析
        enhanced_metadata = {**existing_meta}  # 保留原有数据
        
        # 1. 感知哈希
        if 'phash' in self.features:
            phash = self.calculate_phash(image_path)
            enhanced_metadata['phash'] = phash
        
        # 2. 直方图分布
        if 'histogram' in self.features:
            histogram_data = self.calculate_histogram(image)
            enhanced_metadata.update(histogram_data)
        
        # 3. 色相分布
        if 'hue' in self.features:
            hue_data = self.calculate_hue_distribution(image)
            enhanced_metadata.update(hue_data)
        
        # 4. 人脸姿态（如果已有landmarks）
        if 'pose' in self.features and 'landmarks' in existing_meta:
            landmarks = existing_meta['landmarks']
            image_size = image.shape[0]  # 假设是正方形
            pose_data = self.estimate_face_pose(landmarks, image_size)
            enhanced_metadata.update(pose_data)
        
        # 5. 人脸 embedding (如果 ArcFace 可用)
        if 'embedding' in self.features and self.arcface_extractor is not None:
            embedding = self.extract_face_embedding(image)
            if embedding is not None:
                enhanced_metadata['embedding'] = embedding
        
        # 6. 人脸特征点检测 (landmark 模式)
        if 'landmark' in self.features and self.landmarker is not None:
            landmarks_106pt = self.detect_landmarks_106pt(image)
            if landmarks_106pt is not None:
                # 转换为 68 点
                from Extractor.Extractor import landmark106to68
                landmarks_68pt = landmark106to68(landmarks_106pt)
                
                # 保存 68pt landmarks
                enhanced_metadata['landmarks'] = landmarks_68pt.tolist()
                enhanced_metadata['landmarks_count'] = 68
                
                # 同时计算 pose（使用 68 点）
                if 'pose' in self.features:
                    image_size = image.shape[0]  # 假设是正方形
                    pose_data = self.estimate_face_pose(landmarks_68pt, image_size)
                    enhanced_metadata.update(pose_data)
        
        # 添加分析时间戳
        enhanced_metadata['analyzed_timestamp'] = str(int(image_path.stat().st_mtime))
        
        return enhanced_metadata
    
    def detect_landmarks_106pt(self, image: np.ndarray):
        """
        使用 InsightFace 检测 106 点人脸特征点
        输入图片会先缩放到512x512以加速推理
        
        Args:
            image: BGR 图像 (numpy array)
            
        Returns:
            landmarks_106pt: shape (106, 2) 或 None（原始图像坐标系）
        """
        if self.landmarker is None:
            return None
        
        try:
            # 记录原始尺寸
            original_h, original_w = image.shape[:2]
            
            # 缩放到512x512以加速推理
            target_size = 512
            if original_h != target_size or original_w != target_size:
                resized_image = cv2.resize(image, (target_size, target_size), interpolation=cv2.INTER_AREA)
                scale_x = original_w / target_size
                scale_y = original_h / target_size
            else:
                resized_image = image
                scale_x = 1.0
                scale_y = 1.0
            
            # 在缩放后的图像上检测
            results = self.landmarker.extract(resized_image)
            if results is not None and len(results) > 0:
                landmarks_106pt = results[0]  # shape: (106, 2)
                
                # 将landmarks坐标转换回原始图像坐标系
                landmarks_106pt[:, 0] *= scale_x
                landmarks_106pt[:, 1] *= scale_y
                
                return landmarks_106pt
            else:
                return None
        except Exception as e:
            print(f"Landmark 检测失败: {e}")
            return None
    
    def analyze_batch(self, batch_size: int = 100, force_reanalyze: bool = False, workers: int = None) -> int:
        """
        批量分析所有图像（支持多进程）
        
        Args:
            batch_size: 批处理大小（仅单进程模式有效）
            force_reanalyze: 是否强制重新分析（覆盖已有数据）
            workers: 工作进程数，None=自动检测CPU核心数，1=单进程
            
        Returns:
            成功分析的图像数量
        """
        print(S('ANALYZER_START_BATCH_ANALYSIS', len(self.image_files)))
        print(f"\n配置信息:")
        print(f"  - 数据集路径: {self.faceset_path}")
        print(f"  - 待分析图像数: {len(self.image_files)}")
        print(f"  - 要计算的特征: {self.features}")
        print(f"  - 强制重新分析: {'是' if force_reanalyze else '否'}")
        print(f"  - 工作进程数: {workers if workers else '自动检测'}")
        print()
        
        # 确定需要分析的文件
        files_to_analyze = []
        for img_file in self.image_files:
            filename = img_file.name
            
            if force_reanalyze:
                # 强制模式：重新分析所有
                files_to_analyze.append(filename)
            else:
                # 智能模式：检查是否缺少所选特征的数据（使用流式访问）
                existing_meta = self._get_metadata_streaming(filename)
                needs_analysis = False
                
                # 检查每个选中的特征是否缺失
                for feature in self.features:
                    if feature == 'phash' and 'phash' not in existing_meta:
                        needs_analysis = True
                        break
                    elif feature == 'histogram' and 'histogram_rgb' not in existing_meta:
                        needs_analysis = True
                        break
                    elif feature == 'hue' and 'hue_mean' not in existing_meta:
                        needs_analysis = True
                        break
                    elif feature == 'pose' and 'landmarks' in existing_meta and 'pitch' not in existing_meta:
                        needs_analysis = True
                        break
                    elif feature == 'embedding' and self.arcface_extractor is not None and 'embedding' not in existing_meta:
                        needs_analysis = True
                        break
                    elif feature == 'landmark' and 'landmarks' not in existing_meta:
                        needs_analysis = True
                        break
                
                if needs_analysis:
                    files_to_analyze.append(filename)
        
        if not files_to_analyze:
            print(S('ANALYZER_ALL_ANALYZED'))
            return 0
        
        print(S('ANALYZER_TO_ANALYZE', len(files_to_analyze)))
        
        # 如果选择了pose特征，提前检查是否有landmarks
        if 'pose' in self.features:
            # 扫描前100个文件检查 landmark 可用性，避免仅采第一个文件造成的漏报
            missing_landmarks = False
            for sample_file in files_to_analyze[:100]:
                sample_meta = self._get_metadata_streaming(sample_file)
                if 'landmarks' not in sample_meta:
                    missing_landmarks = True
                    break
            if missing_landmarks:
                    print(f"\n{'='*60}")
                    print(f"⚠ 警告: 缺少landmarks数据，无法计算pose")
                    print(f"{'='*60}")
                    print(f"原因: 元数据中没有 'landmarks' 字段")
                    print(f"解决方案:")
                    print(f"  1. 重新运行Extractor，选择支持landmarks的标记器:")
                    print(f"     - insightface-2d106det (推荐)")
                    print(f"     - 2DFAN-4")
                    print(f"     - Google-mediapipe")
                    print(f"  2. 或者不选择pose特征进行分析")
                    print(f"{'='*60}\n")
        
        # 根据workers参数选择单进程或多进程
        if workers is None:
            # 自动检测：如果文件数>100，使用多进程
            workers = min(cpu_count(), len(files_to_analyze)) if len(files_to_analyze) > 100 else 1
            # GPU 模型（landmark/embedding）每个进程独享一份 GPU 显存拷贝
            # 12 个 landmarker 同时加载可消耗 6GB+ VRAM → OOM 风险
            # 保持至少 4 个 worker 以确保 CPU-GPU 流水线饱和，不多于 cpu_count
            has_gpu = any(f in self.features for f in ('landmark', 'embedding'))
            if has_gpu:
                try:
                    import torch
                    gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
                except ImportError:
                    gpu_count = 0
                if gpu_count > 0:
                    capped = max(gpu_count * 2, 4)  # 至少 4 个
                    if workers > capped:
                        print(f"  → 检测到 GPU×{gpu_count}，worker 数已从 {workers} 降至 {capped}（节省显存）")
                        workers = min(workers, capped)
                else:
                    workers = 1
        
        if workers <= 1:
            # 单进程模式
            analyzed_count = self._analyze_batch_single(files_to_analyze)
            # 增量保存（只写 cache，避免全量读 HDF5 + 重写）
            self._save_metadata(incremental_only=True, reinit_accessor=True)
        else:
            # 多进程模式（每个进程独立初始化模型）
            analyzed_count = self._analyze_batch_multiprocess(files_to_analyze, workers)
            # 多进程模式已在内部保存，无需再次保存
        
        return analyzed_count
    
    def _analyze_batch_single(self, files_to_analyze: List[str]) -> int:
        """单进程批量分析"""
        analyzed_count = 0
        error_count = 0
        save_interval = 10000  # 每10000张图片保存一次
        last_save_count = 0
        
        pbar = tqdm.tqdm(total=len(files_to_analyze), desc=S('ANALYZER_PROGRESS'), unit="img", ascii=True)
        
        for filename in files_to_analyze:
            try:
                metadata = self.analyze_single_image(filename)
                if metadata:
                    # 合并到缓存 (保留已有数据，添加新计算的)
                    if filename in self._metadata_cache:
                        self._metadata_cache[filename].update(metadata)
                    else:
                        self._metadata_cache[filename] = metadata
                    analyzed_count += 1
                else:
                    error_count += 1
                
                # 每处理save_interval张图片就保存一次
                if analyzed_count - last_save_count >= save_interval:
                    self._save_metadata(incremental_only=True, reinit_accessor=False)  # 不重新初始化，避免冲突
                    last_save_count = analyzed_count
            except Exception as e:
                print(S('ANALYZER_ANALYZE_ERROR', filename, e))
                error_count += 1
            finally:
                pbar.update(1)
        
        pbar.close()
        
        print(S('ANALYZER_BATCH_COMPLETE', analyzed_count, error_count))
        return analyzed_count
    
    def _analyze_batch_multiprocess(self, files_to_analyze: List[str], workers: int) -> int:
        """多进程批量分析（分批收集结果，每10000条统一保存一次）"""
        from multiprocessing import Pool, Manager
        
        print(f"使用 {workers} 个工作进程进行并行分析")
        print(f"传递给工作进程的特征: {self.features}")
        
        # 创建共享管理器用于收集结果
        manager = Manager()
        result_dict = manager.dict()  # 共享字典存储结果
        
        # 准备任务参数
        tasks = [
            (str(self.faceset_path), filename, self._get_metadata_streaming(filename), self.features)
            for filename in files_to_analyze
        ]
        
        analyzed_count = 0
        error_count = 0
        embedding_saved_count = 0
        save_interval = 10000  # 每10000张图片保存一次
        batch_count = 0  # 当前批次计数
        
        # 使用多进程池
        with Pool(processes=workers) as pool:
            for result in tqdm.tqdm(
                pool.imap_unordered(_analyze_single_worker, tasks),
                total=len(tasks),
                desc=S('ANALYZER_PROGRESS'),
                unit="img",
                ascii=True
            ):
                filename, metadata = result
                if metadata:
                    # 存入共享字典
                    result_dict[filename] = metadata
                    
                    # 统计 embedding 保存情况
                    if 'embedding' in metadata and metadata['embedding'] is not None:
                        embedding_saved_count += 1
                    
                    analyzed_count += 1
                    batch_count += 1
                else:
                    error_count += 1
                
                # 检查是否达到保存阈值
                if batch_count >= save_interval:
                    print(f"\n已达到 {save_interval} 条记录，正在保存...")
                    
                    # 合并到缓存
                    self._metadata_cache.update(dict(result_dict))
                    
                    # 保存当前批次（不重新初始化访问器，避免冲突）
                    self._save_metadata(incremental_only=True, reinit_accessor=False)
                    
                    # 清空共享字典和批次计数
                    result_dict.clear()
                    batch_count = 0
                    
                    print(f"✓ 已保存 {analyzed_count} 条记录\n")
        
        print(S('ANALYZER_BATCH_COMPLETE', analyzed_count, error_count))
        print(f"✓ 成功保存 {embedding_saved_count} 条 embedding 数据")
        
        # 保存剩余的数据
        if len(result_dict) > 0:
            print(f"\n正在保存剩余的 {len(result_dict)} 条记录...")
            self._metadata_cache.update(dict(result_dict))
            self._save_metadata(incremental_only=True, reinit_accessor=True)  # 最后一次保存后重新初始化
            print(f"✓ 最终保存完成")
        
        return analyzed_count
    
    def _save_metadata(self, incremental_only: bool = False, reinit_accessor: bool = True):
        """
        保存元数据为 HDF5 格式
        
        Args:
            incremental_only: 如果为True，只保存缓存中的新数据（增量模式）
            reinit_accessor: 是否重新初始化HDF5访问器（默认True）
        """
        try:
            import h5py

            # 检查 HDF5 文件完整性，如果损坏则重新创建
            if self.metadata_file.exists() and not h5py.is_hdf5(str(self.metadata_file)):
                print(f"[!] HDF5 file is corrupted, recreating: {self.metadata_file.name}")
                self.metadata_file.unlink()

            # 关闭旧的 HDF5 访问器（仅在打开时，避免不必要的文件系统调用）
            if self._h5_accessor is not None:
                self.close()

            if incremental_only and self._metadata_cache:
                # 增量模式：只保存缓存中的数据
                with h5py.File(self.metadata_file, 'a') as f:
                    for filename, meta in self._metadata_cache.items():
                        safe_name = filename.replace('/', '_SLASH_').replace('\\', '_BSLASH_')
                        
                        # 如果组已存在，更新它；否则创建新组
                        if safe_name in f:
                            grp = f[safe_name]
                        else:
                            grp = f.create_group(safe_name)
                            grp.attrs['__original_filename__'] = filename
                        
                        # 写入/更新元数据字段
                        for key, value in meta.items():
                            # 删除旧数据（如果存在）
                            if key in grp:
                                del grp[key]
                            
                            if isinstance(value, (list, np.ndarray)):
                                grp.create_dataset(key, data=np.array(value), compression='gzip', compression_opts=4)
                            elif isinstance(value, (int, float, str)):
                                grp.attrs[key] = value
                            else:
                                grp.attrs[key] = str(value)
                
                # 清空缓存
                cache_count = len(self._metadata_cache)
                self._metadata_cache.clear()
                
            else:
                # 全量模式：保存所有元数据
                all_metadata = self.get_metadata_snapshot()

                # 合并缓存中的新数据（尚未保存到HDF5的）
                if self._metadata_cache:
                    all_metadata.update(self._metadata_cache)

                with h5py.File(self.metadata_file, 'w') as f:
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

                # 清空缓存（全量保存后数据已持久化）
                if self._metadata_cache:
                    self._metadata_cache.clear()

            # 重要：保存后重新初始化 HDF5 访问器，以便后续读取
            if reinit_accessor:
                self._init_h5_accessor()
            
        except Exception as e:
            print(S('ANALYZER_SAVE_ERROR', e))
            import traceback
            traceback.print_exc()
    
    def generate_report(self, output_file: Optional[Path] = None):
        """
        生成分析报告
        
        Args:
            output_file: 输出文件路径（可选）
        """
        print(S('ANALYZER_GENERATING_REPORT'))
        
        # 统计数据（使用流式访问，避免加载全部元数据）
        total_images = self.metadata_count
        phash_count = 0
        pose_count = 0
        histogram_count = 0
        embedding_count = 0
        landmark_count = 0
        
        # 流式统计各字段数量
        if self._h5_accessor:
            check_fields = {'phash', 'pitch', 'histogram_rgb', 'embedding', 'landmarks'}
            for filename in self._h5_accessor.filenames:
                fields = self._h5_accessor.has_fields(filename, check_fields)
                if fields.get('phash'):          phash_count += 1
                if fields.get('pitch'):          pose_count += 1
                if fields.get('histogram_rgb'):  histogram_count += 1
                if fields.get('embedding'):      embedding_count += 1
                if fields.get('landmarks'):      landmark_count += 1
        
        report = f"""
{'='*60}
人脸集分析报告
{'='*60}

数据集路径: {self.faceset_path}
总图像数: {total_images}

元数据统计:
  - 感知哈希 (phash): {phash_count}/{total_images}
  - 人脸姿态 (pose): {pose_count}/{total_images}
  - 直方图 (histogram): {histogram_count}/{total_images}
  - 人脸 Embedding (embedding): {embedding_count}/{total_images}
  - 人脸特征点 (landmarks): {landmark_count}/{total_images}

"""
        
        print(report)
        
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(report)
            print(S('ANALYZER_REPORT_SAVED', output_file))
    
    def merge_datasets(self, source_path: Path) -> Dict:
        """
        合并两个数据集（将 source_path 的图片迁移到当前 faceset_path）
        使用严格的 phash 匹配：只有完全相同的 phash 才认为重复
        
        Args:
            source_path: 源数据集路径（B目录）
            
        Returns:
            合并结果统计
        """
        print(S('MERGER_START_MERGE', source_path))
        
        source_path = Path(source_path)
        if not source_path.exists():
            print(S('PATH_NOT_EXIST', source_path))
            return {}
        
        # 加载源数据集元数据（HDF5 格式）
        source_metadata_file = source_path / "metadata.h5"
        
        source_metadata = {}
        if source_metadata_file.exists():
            try:
                import h5py
                with h5py.File(source_metadata_file, 'r') as f:
                    for safe_name in f.keys():
                        grp = f[safe_name]
                        meta = {}
                        for key in grp.keys():
                            dataset = grp[key]
                            if isinstance(dataset, h5py.Dataset):
                                data = dataset[:]
                                # 对于 landmarks 和 embedding，保持为 numpy 数组
                                if key in ['landmarks', 'embedding'] and isinstance(data, np.ndarray):
                                    meta[key] = data
                                elif isinstance(data, np.ndarray):
                                    if data.ndim == 0:
                                        meta[key] = data.item()
                                    else:
                                        meta[key] = data.tolist()
                                else:
                                    meta[key] = data
                        for key, value in grp.attrs.items():
                            # 跳过内部使用的属性
                            if key == '__original_filename__':
                                continue
                            meta[key] = value
                        # 使用存储的原始文件名
                        original_filename = grp.attrs.get('__original_filename__', safe_name)
                        source_metadata[original_filename] = meta
                print(S('MERGER_SOURCE_METADATA_LOADED', len(source_metadata)))
            except Exception as e:
                print(S('ANALYZER_METADATA_LOAD_ERROR', e))
        
        # 扫描源数据集图片
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
        source_files = sorted([
            f for f in source_path.iterdir()
            if f.suffix.lower() in image_extensions and f.is_file()
        ])
        
        print(S('MERGER_SOURCE_FOUND_IMAGES', len(source_files)))
        
        # 构建当前数据集的 phash 索引（用于去重）- 流式读取
        target_phash_index = {}
        if self._h5_accessor:
            for filename in self._h5_accessor.filenames:
                meta = self._get_metadata_streaming(filename)
                if 'phash' in meta and meta['phash']:
                    target_phash_index[meta['phash']] = filename
        
        print(S('MERGER_BUILDING_INDEX', len(target_phash_index)))
        
        # 合并统计
        stats = {
            'total_source': len(source_files),
            'duplicates_found': 0,
            'files_copied': 0,
            'files_renamed': 0,
            'metadata_merged': 0,
            'errors': 0,
            'duplicate_files': [],
            'copied_files': [],
            'renamed_files': []
        }
        
        # 处理每个源文件
        pbar = tqdm.tqdm(total=len(source_files), desc=S('MERGER_PROGRESS'), unit="img", ascii=True)
        
        for source_file in source_files:
            try:
                # 计算源文件的 phash
                source_phash = self.calculate_phash(source_file)
                
                # 检查是否重复（严格匹配：phash 必须完全相同）
                is_duplicate = False
                if source_phash and source_phash in target_phash_index:
                    is_duplicate = True
                    target_filename = target_phash_index[source_phash]
                    stats['duplicates_found'] += 1
                    stats['duplicate_files'].append({
                        'source': source_file.name,
                        'target': target_filename,
                        'phash': source_phash
                    })
                    print(S('MERGER_DUPLICATE_FOUND', source_file.name, target_filename, 0))
                
                if is_duplicate:
                    pbar.update(1)
                    continue
                
                # 非重复文件，需要复制
                # 检查文件名是否冲突
                target_filename = source_file.name
                target_filepath = self.faceset_path / target_filename
                
                if target_filepath.exists():
                    # 文件名冲突，生成新名称
                    base_name = source_file.stem
                    suffix = source_file.suffix
                    counter = 1
                    
                    while target_filepath.exists():
                        target_filename = f"{base_name}_merge{counter}{suffix}"
                        target_filepath = self.faceset_path / target_filename
                        counter += 1
                    
                    stats['files_renamed'] += 1
                    stats['renamed_files'].append({
                        'original': source_file.name,
                        'renamed': target_filename
                    })
                else:
                    stats['files_copied'] += 1
                
                # 复制文件
                import shutil
                shutil.copy2(str(source_file), str(target_filepath))
                stats['copied_files'].append(target_filename)
                
                # 合并元数据（如果有）
                if source_file.name in source_metadata:
                    self._metadata_cache[target_filename] = source_metadata[source_file.name]
                    stats['metadata_merged'] += 1
                
                # 更新 phash 索引
                if source_phash:
                    target_phash_index[source_phash] = target_filename
                
            except Exception as e:
                print(S('MERGER_COPY_ERROR', source_file.name, e))
                stats['errors'] += 1
            finally:
                pbar.update(1)
        
        pbar.close()
        
        # 保存合并后的元数据
        self._save_metadata()
        
        # 打印统计结果
        print(S('MERGER_COMPLETE'))
        print(f"  {S('MERGER_STAT_TOTAL')}: {stats['total_source']}")
        print(f"  {S('MERGER_STAT_DUPLICATES')}: {stats['duplicates_found']}")
        print(f"  {S('MERGER_STAT_COPIED')}: {stats['files_copied']}")
        print(f"  {S('MERGER_STAT_RENAMED')}: {stats['files_renamed']}")
        print(f"  {S('MERGER_STAT_METADATA')}: {stats['metadata_merged']}")
        print(f"  {S('MERGER_STAT_ERRORS')}: {stats['errors']}")
        
        return stats
    
    def write_metadata_to_jpg(self, filename: str) -> bool:
        """
        将 metadata.json 中的元数据写回 JPG 文件的 APP15 chunk
        参考 DFLJPG.py 的实现方式
        
        Args:
            filename: 图像文件名
            
        Returns:
            是否成功写入
        """
        try:
            image_path = self.faceset_path / filename
            
            # 检查文件是否存在
            if not image_path.exists():
                print(S('ANALYZER_FILE_NOT_FOUND', filename))
                return False
            
            # 获取该文件的元数据（流式读取）
            meta_data = self._get_metadata_streaming(filename)
            if not meta_data:
                print(S('ANALYZER_NO_METADATA_FOR_FILE', filename))
                return False
            
            # 读取原始 JPG 文件
            with open(image_path, 'rb') as f:
                data = f.read()
            
            # 解析 JPG chunks
            chunks = []
            data_counter = 0
            inst_length = len(data)
            
            while data_counter < inst_length:
                if data_counter + 2 > inst_length:
                    break
                    
                chunk_m_l, chunk_m_h = struct.unpack("BB", data[data_counter:data_counter+2])
                data_counter += 2
                
                if chunk_m_l != 0xFF:
                    print(S('ANALYZER_INVALID_JPG', filename))
                    return False
                
                chunk_name = None
                chunk_size = None
                chunk_data = None
                chunk_ex_data = None
                
                # 识别 chunk 类型
                if chunk_m_h & 0xF0 == 0xD0:
                    n = chunk_m_h & 0x0F
                    if n >= 0 and n <= 7:
                        chunk_name = "RST%d" % (n)
                        chunk_size = 0
                    elif n == 0x8:
                        chunk_name = "SOI"
                        chunk_size = 0
                    elif n == 0x9:
                        chunk_name = "EOI"
                        chunk_size = 0
                    elif n == 0xA:
                        chunk_name = "SOS"
                    elif n == 0xB:
                        chunk_name = "DQT"
                    elif n == 0xD:
                        chunk_name = "DRI"
                        chunk_size = 2
                elif chunk_m_h & 0xF0 == 0xC0:
                    n = chunk_m_h & 0x0F
                    if n == 0:
                        chunk_name = "SOF0"
                    elif n == 2:
                        chunk_name = "SOF2"
                    elif n == 4:
                        chunk_name = "DHT"
                elif chunk_m_h & 0xF0 == 0xE0:
                    n = chunk_m_h & 0x0F
                    chunk_name = "APP%d" % (n)
                
                # 解析 chunk 大小
                if chunk_size is None:  # variable size
                    if data_counter + 2 > inst_length:
                        break
                    chunk_size, = struct.unpack(">H", data[data_counter:data_counter+2])
                    chunk_size -= 2
                    data_counter += 2
                
                # 读取 chunk 数据
                if chunk_size > 0:
                    if data_counter + chunk_size > inst_length:
                        break
                    chunk_data = data[data_counter:data_counter+chunk_size]
                    data_counter += chunk_size
                
                # SOS chunk 有特殊处理：找 EOI（FF D9）。逐字节 Python 循环在
                # 36KB 的文件上要 3 万次迭代（1.3ms），133KB 要 13 万次（5.6ms）；
                # 换成 C 层的 find，结果一样。
                if chunk_name == "SOS":
                    c = data.find(b"\xff\xd9", data_counter)
                    if c < 0:
                        c = inst_length - 1
                    chunk_ex_data = data[data_counter:c]
                    data_counter = c
                
                chunks.append({
                    'name': chunk_name,
                    'm_h': chunk_m_h,
                    'data': chunk_data,
                    'ex_data': chunk_ex_data,
                })
            
            # 移除旧的 APP15 chunk
            chunks = [c for c in chunks if c['name'] != 'APP15']
            
            # 找到最后一个 APP chunk 的位置
            last_app_chunk = 0
            for i, chunk in enumerate(chunks):
                if chunk['m_h'] & 0xF0 == 0xE0:
                    last_app_chunk = i
            
            # 创建新的 APP15 chunk，使用 pickle 序列化元数据
            # 重要：将所有 numpy 数组转换为 list，避免 NumPy 版本不兼容问题
            def convert_numpy_to_list(obj):
                """递归转换 numpy 数组为 Python list"""
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, dict):
                    return {k: convert_numpy_to_list(v) for k, v in obj.items()}
                elif isinstance(obj, (list, tuple)):
                    return [convert_numpy_to_list(item) for item in obj]
                else:
                    return obj
            
            dict_data = {k: convert_numpy_to_list(v) for k, v in meta_data.items() if v is not None}
            dflchunk = {
                'name': 'APP15',
                'm_h': 0xEF,
                'data': pickle.dumps(dict_data),
                'ex_data': None,
            }
            chunks.insert(last_app_chunk + 1, dflchunk)
            
            # 重建 JPG 数据
            output_data = b""
            for chunk in chunks:
                output_data += struct.pack("BB", 0xFF, chunk['m_h'])
                chunk_data = chunk['data']
                if chunk_data is not None:
                    output_data += struct.pack(">H", len(chunk_data) + 2)
                    output_data += chunk_data
                
                chunk_ex_data = chunk['ex_data']
                if chunk_ex_data is not None:
                    output_data += chunk_ex_data
            
            # 写回文件
            with open(image_path, 'wb') as f:
                f.write(output_data)
            
            return True
            
        except Exception as e:
            print(S('ANALYZER_WRITE_METADATA_ERROR', filename, e))
            return False
    
    def batch_write_metadata_to_jpgs(self, workers: int = None) -> int:
        """
        批量将所有元数据写回 JPG 文件（支持多进程）
        
        Args:
            workers: 工作进程数，默认为 CPU 核心数
            
        Returns:
            成功写入的文件数量
        """
        if workers is None:
            workers = cpu_count()
        
        # 限制最大进程数，避免过多进程导致 I/O 瓶颈
        workers = min(workers, 8)
        
        print(S('ANALYZER_START_BATCH_WRITE', self._h5_accessor.file_count if self._h5_accessor else len(self._metadata_cache)))
        print(f"使用 {workers} 个工作进程")
        
        # 预先转换所有元数据为纯 Python 类型，避免在每个进程中重复转换
        print("预处理：转换元数据类型...")
        preprocessed_metadata = {}
        
        # 流式读取并转换
        filenames = self._h5_accessor.filenames if self._h5_accessor else list(self._metadata_cache.keys())
        pbar = tqdm.tqdm(total=len(filenames), desc="Loading metadata", unit="file", ascii=True)
        
        for filename in filenames:
            meta = self._get_metadata_streaming(filename)
            converted = self._convert_numpy_to_list(meta)
            
            # 重要：为了兼容原版 DFL 的 XSeg 功能，需要添加 source_landmarks 字段
            # XSegUtil.py 使用 get_source_landmarks() 而不是 get_landmarks()
            if 'landmarks' in converted and 'source_landmarks' not in converted:
                converted['source_landmarks'] = converted['landmarks']

            # 添加 source_filename（原帧文件名），导出流水线需要此字段定位原始帧
            if 'source_filename' not in converted:
                src = Path(filename).stem
                if '_' in src:
                    src = src.rsplit('_', 1)[0]
                converted['source_filename'] = src + '.jpg'
            
            preprocessed_metadata[filename] = converted
            pbar.update(1)
        
        pbar.close()
        print(f"✓ 预处理完成")
        
        # 准备参数列表（只传递文件名和已转换的元数据）
        tasks = [(str(self.faceset_path), filename, preprocessed_metadata[filename]) 
                 for filename in preprocessed_metadata.keys()]
        
        # 使用多进程池
        success_count = 0
        error_count = 0
        
        with Pool(processes=workers) as pool:
            results = list(tqdm.tqdm(
                pool.imap_unordered(_write_single_file_worker_optimized, tasks),
                total=len(tasks),
                desc=S('ANALYZER_WRITE_PROGRESS'),
                unit="img",
                ascii=True
            ))
        
        # 统计结果
        for success in results:
            if success:
                success_count += 1
            else:
                error_count += 1
        
        print(S('ANALYZER_BATCH_WRITE_COMPLETE', success_count, error_count))
        
        return success_count
    
    def _convert_numpy_to_list(self, obj, depth=0):
        """
        递归转换 numpy 数组和标量为 Python 原生类型
        
        Args:
            obj: 任意对象
            depth: 递归深度（用于调试）
            
        Returns:
            转换后的对象（numpy array/scalar → list/int/float）
        """
        # 处理 numpy 数组
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        # 处理 numpy 标量类型（int64, float64 等）
        elif isinstance(obj, np.generic):
            return obj.item()
        elif isinstance(obj, dict):
            result = {}
            for k, v in obj.items():
                converted_v = self._convert_numpy_to_list(v, depth+1)
                result[k] = converted_v
            return result
        elif isinstance(obj, (list, tuple)):
            result = []
            for item in obj:
                converted_item = self._convert_numpy_to_list(item, depth+1)
                result.append(converted_item)
            return result
        else:
            return obj


def _write_single_file_worker_optimized(args):
    """
    多进程工作函数：写入单个文件的元数据（优化版）
    假设 meta_data 已经是纯 Python 类型，无需再次转换
    
    Args:
        args: (faceset_path_str, filename, preprocessed_metadata) 元组
        
    Returns:
        是否成功
    """
    faceset_path_str, filename, meta_data = args
    
    try:
        from pathlib import Path
        import pickle
        import struct
        
        faceset_path = Path(faceset_path_str)
        image_path = faceset_path / filename
        
        # 检查文件是否存在
        if not image_path.exists():
            return False
        
        # 读取原始 JPG 文件
        with open(image_path, 'rb') as f:
            data = f.read()
        
        # 解析 JPG chunks
        chunks = []
        data_counter = 0
        inst_length = len(data)
        
        while data_counter < inst_length:
            if data_counter + 2 > inst_length:
                break
                
            chunk_m_l, chunk_m_h = struct.unpack("BB", data[data_counter:data_counter+2])
            data_counter += 2
            
            if chunk_m_l != 0xFF:
                return False
            
            chunk_name = None
            chunk_size = None
            chunk_data = None
            chunk_ex_data = None
            
            # 识别 chunk 类型
            if chunk_m_h & 0xF0 == 0xD0:
                n = chunk_m_h & 0x0F
                if n >= 0 and n <= 7:
                    chunk_name = "RST%d" % (n)
                    chunk_size = 0
                elif n == 0x8:
                    chunk_name = "SOI"
                    chunk_size = 0
                elif n == 0x9:
                    chunk_name = "EOI"
                    chunk_size = 0
                elif n == 0xA:
                    chunk_name = "SOS"
                elif n == 0xB:
                    chunk_name = "DQT"
                elif n == 0xD:
                    chunk_name = "DRI"
                    chunk_size = 2
            elif chunk_m_h & 0xF0 == 0xC0:
                n = chunk_m_h & 0x0F
                if n == 0:
                    chunk_name = "SOF0"
                elif n == 2:
                    chunk_name = "SOF2"
                elif n == 4:
                    chunk_name = "DHT"
            elif chunk_m_h & 0xF0 == 0xE0:
                n = chunk_m_h & 0x0F
                chunk_name = "APP%d" % (n)
            
            # 解析 chunk 大小
            if chunk_size is None:  # variable size
                if data_counter + 2 > inst_length:
                    break
                chunk_size, = struct.unpack(">H", data[data_counter:data_counter+2])
                chunk_size -= 2
                data_counter += 2
            
            # 读取 chunk 数据
            if chunk_size > 0:
                if data_counter + chunk_size > inst_length:
                    break
                chunk_data = data[data_counter:data_counter+chunk_size]
                data_counter += chunk_size
            
            # SOS chunk 有特殊处理：同上面那处，逐字节循环换成 find
            if chunk_name == "SOS":
                c = data.find(b"\xff\xd9", data_counter)
                if c < 0:
                    c = inst_length - 1
                chunk_ex_data = data[data_counter:c]
                data_counter = c
            
            chunks.append({
                'name': chunk_name,
                'm_h': chunk_m_h,
                'data': chunk_data,
                'ex_data': chunk_ex_data,
            })
        
        # 移除旧的 APP15 chunk
        chunks = [c for c in chunks if c['name'] != 'APP15']
        
        # 找到最后一个 APP chunk 的位置
        last_app_chunk = 0
        for i, chunk in enumerate(chunks):
            if chunk['m_h'] & 0xF0 == 0xE0:
                last_app_chunk = i
        
        # 创建新的 APP15 chunk（meta_data 已经是纯 Python 类型，直接序列化）
        dict_data = {k: v for k, v in meta_data.items() if v is not None}
        
        dflchunk = {
            'name': 'APP15',
            'm_h': 0xEF,
            'data': pickle.dumps(dict_data),
            'ex_data': None,
        }
        chunks.insert(last_app_chunk + 1, dflchunk)
        
        # 重建 JPG 数据（使用列表拼接，避免频繁的字符串 concatenation）
        output_parts = []
        for chunk in chunks:
            output_parts.append(struct.pack("BB", 0xFF, chunk['m_h']))
            chunk_data = chunk['data']
            if chunk_data is not None:
                output_parts.append(struct.pack(">H", len(chunk_data) + 2))
                output_parts.append(chunk_data)
            
            chunk_ex_data = chunk['ex_data']
            if chunk_ex_data is not None:
                output_parts.append(chunk_ex_data)
        
        output_data = b''.join(output_parts)
        
        # 写回文件
        with open(image_path, 'wb') as f:
            f.write(output_data)
        
        return True
        
    except Exception as e:
        return False


# 全局变量：进程级别的 ArcFace 提取器单例
_process_arcface_extractor = None
# 全局变量：进程级别的 InsightFace 106pt 标记器单例
_process_landmarker = None

def _analyze_single_worker(args):
    """
    多进程工作函数：分析单个图像
    
    Args:
        args: (faceset_path_str, filename, existing_metadata, features) 元组
        
    Returns:
        (filename, metadata) 元组
    """
    global _process_arcface_extractor, _process_landmarker
    
    faceset_path_str, filename, existing_meta, features = args
    
    try:
        import os
        from pathlib import Path
        import cv2
        import numpy as np
        import imagehash
        from PIL import Image
        
        faceset_path = Path(faceset_path_str)
        image_path = faceset_path / filename
        
        # 读取图像
        image = cv2.imread(str(image_path))
        if image is None:
            return (filename, {})
        
        # 根据选择的特征进行分析
        enhanced_metadata = {**existing_meta}  # 保留原有数据
        
        # 1. 感知哈希
        if 'phash' in features:
            try:
                pil_image = Image.open(image_path)
                phash = str(imagehash.phash(pil_image))
                enhanced_metadata['phash'] = phash
            except:
                pass
        
        # 2. 直方图分布
        if 'histogram' in features:
            try:
                # RGB直方图
                hist_b = cv2.calcHist([image], [0], None, [256], [0, 256]).flatten()
                hist_g = cv2.calcHist([image], [1], None, [256], [0, 256]).flatten()
                hist_r = cv2.calcHist([image], [2], None, [256], [0, 256]).flatten()
                
                # 归一化
                hist_b = (hist_b / hist_b.sum()).tolist()
                hist_g = (hist_g / hist_g.sum()).tolist()
                hist_r = (hist_r / hist_r.sum()).tolist()
                
                # HSV直方图
                hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
                hist_h = cv2.calcHist([hsv], [0], None, [180], [0, 180]).flatten()
                hist_s = cv2.calcHist([hsv], [1], None, [256], [0, 256]).flatten()
                hist_v = cv2.calcHist([hsv], [2], None, [256], [0, 256]).flatten()
                
                hist_h = (hist_h / hist_h.sum()).tolist()
                hist_s = (hist_s / hist_s.sum()).tolist()
                hist_v = (hist_v / hist_v.sum()).tolist()
                
                enhanced_metadata.update({
                    'histogram_rgb': {
                        'b': hist_b,
                        'g': hist_g,
                        'r': hist_r
                    },
                    'histogram_hsv': {
                        'h': hist_h,
                        's': hist_s,
                        'v': hist_v
                    }
                })
            except:
                pass
        
        # 3. 色相分布
        if 'hue' in features:
            try:
                hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
                hue = hsv[:, :, 0].astype(np.float32)
                
                hue_mean = float(np.mean(hue))
                hue_std = float(np.std(hue))
                
                enhanced_metadata.update({
                    'hue_mean': hue_mean,
                    'hue_std': hue_std
                })
            except:
                pass
        
        # 4. 人脸姿态（如果已有landmarks）
        if 'pose' in features and 'landmarks' in existing_meta:
            try:
                from facelib.LandmarksProcessor import estimate_pitch_yaw_roll
                landmarks = existing_meta['landmarks']
                landmarks_array = np.array(landmarks, dtype=np.float32)
                image_size = image.shape[0]  # 假设是正方形
                pitch, yaw, roll = estimate_pitch_yaw_roll(landmarks_array, size=image_size)
                
                enhanced_metadata.update({
                    'pitch': float(pitch),
                    'yaw': float(yaw),
                    'roll': float(roll)
                })
            except:
                pass
        
        # 5. 人脸 Embedding (ArcFace)
        if 'embedding' in features:
            try:
                global _process_arcface_extractor
                
                # 如果还没有初始化，则创建提取器（每个进程只创建一次）
                if _process_arcface_extractor is None:
                    import sys
                    from pathlib import Path as PathLib
                    project_root = PathLib(__file__).parent.parent
                    model_dir = project_root / "modelhub" / "onnx" / "ArcFace"
                    
                    print(f"\n[进程 {os.getpid()}] 正在初始化 ArcFace 提取器...", flush=True)
                    
                    if model_dir.exists():
                        try:
                            # 动态导入
                            sys.path.insert(0, str(project_root))
                            from FacesetProcessor.Filter import ArcFaceONNXExtractor
                            
                            # 创建提取器实例
                            _process_arcface_extractor = ArcFaceONNXExtractor(model_dir, model_name="w600k_mbf")
                            print(f"[进程 {os.getpid()}] ✓ ArcFace 提取器初始化成功: {_process_arcface_extractor.model_name_used}\n", flush=True)
                        except Exception as init_err:
                            print(f"[进程 {os.getpid()}] ✗ ArcFace 初始化失败: {init_err}", flush=True)
                            import traceback
                            traceback.print_exc()
                            _process_arcface_extractor = None  # 确保为 None
                    else:
                        print(f"[进程 {os.getpid()}] ✗ 模型目录不存在: {model_dir}\n", flush=True)
                
                # 使用提取器计算 embedding
                if _process_arcface_extractor is not None:
                    emb = _process_arcface_extractor.extract_embedding(image)
                    if emb is not None:
                        enhanced_metadata['embedding'] = emb.tolist()
                    else:
                        # 只在第一次失败时打印警告
                        if not hasattr(_analyze_single_worker, '_emb_warning_shown'):
                            print(f"[进程 {os.getpid()}] ⚠ Warning: ArcFace 返回 None for {filename}", flush=True)
                            _analyze_single_worker._emb_warning_shown = True
                else:
                    # 提取器未初始化，只在第一次打印
                    if not hasattr(_analyze_single_worker, '_extractor_none_shown'):
                        print(f"[进程 {os.getpid()}] ✗ ArcFace 提取器为 None，跳过 embedding 计算", flush=True)
                        _analyze_single_worker._extractor_none_shown = True
            except Exception as e:
                # embedding 提取失败，打印错误信息
                if not hasattr(_analyze_single_worker, '_error_shown'):
                    print(f"\n[进程 {os.getpid()}] ✗ Embedding 提取错误: {e}", flush=True)
                    import traceback
                    traceback.print_exc()
                    _analyze_single_worker._error_shown = True
        
        # 6. 人脸特征点检测 (landmark 模式)
        if 'landmark' in features:
            try:
                global _process_landmarker
                
                # 如果还没有初始化，则创建标记器（每个进程只创建一次）
                if _process_landmarker is None:
                    import sys
                    from pathlib import Path as PathLib
                    project_root = PathLib(__file__).parent.parent
                    
                    print(f"\n[进程 {os.getpid()}] 正在初始化 InsightFace 106pt 标记器...", flush=True)
                    
                    try:
                        # 检测设备
                        from xlib.onnxruntime import get_available_devices_info, get_cpu_device_info
                        devices = get_available_devices_info()
                        device_info = None
                        
                        # 优先级1: CUDA
                        for device in devices:
                            if 'CUDA' in str(device).upper() or 'cuda' in str(device).lower():
                                device_info = device
                                break
                        
                        # 优先级2: DirectML/DX12
                        if device_info is None:
                            for device in devices:
                                if 'DML' in str(device).upper() or 'directml' in str(device).lower() or 'dx12' in str(device).lower():
                                    device_info = device
                                    break
                        
                        # 优先级3: CPU
                        if device_info is None:
                            device_info = get_cpu_device_info()
                        
                        # 动态导入
                        sys.path.insert(0, str(project_root))
                        from modelhub.onnx import InsightFace2D106
                        
                        # 创建标记器实例
                        _process_landmarker = InsightFace2D106(device_info)
                        print(f"[进程 {os.getpid()}] ✓ InsightFace 106pt 标记器初始化成功 (设备: {device_info})\n", flush=True)
                    except Exception as init_err:
                        print(f"[进程 {os.getpid()}] ✗ Landmarker 初始化失败: {init_err}", flush=True)
                        import traceback
                        traceback.print_exc()
                        _process_landmarker = None  # 确保为 None
                
                # 使用标记器检测 landmarks
                if _process_landmarker is not None:
                    # 记录原始尺寸
                    original_h, original_w = image.shape[:2]
                    
                    # 缩放到512x512以加速推理
                    target_size = 512
                    if original_h != target_size or original_w != target_size:
                        resized_image = cv2.resize(image, (target_size, target_size), interpolation=cv2.INTER_AREA)
                        scale_x = original_w / target_size
                        scale_y = original_h / target_size
                    else:
                        resized_image = image
                        scale_x = 1.0
                        scale_y = 1.0
                    
                    # 在缩放后的图像上检测
                    results = _process_landmarker.extract(resized_image)
                    if results is not None and len(results) > 0:
                        landmarks_106pt = results[0]  # shape: (106, 2)
                        
                        # 将landmarks坐标转换回原始图像坐标系
                        landmarks_106pt[:, 0] *= scale_x
                        landmarks_106pt[:, 1] *= scale_y
                        
                        # 转换为 68 点
                        from Extractor.Extractor import landmark106to68
                        landmarks_68pt = landmark106to68(landmarks_106pt)
                        
                        enhanced_metadata['landmarks'] = landmarks_68pt.tolist()
                        enhanced_metadata['landmarks_count'] = 68
                        
                        # 如果同时选择了 pose 特征，计算姿态
                        if 'pose' in features:
                            from facelib.LandmarksProcessor import estimate_pitch_yaw_roll
                            landmarks_array = np.array(landmarks_68pt, dtype=np.float32)
                            image_size = image.shape[0]  # 使用原始图像尺寸
                            pitch, yaw, roll = estimate_pitch_yaw_roll(landmarks_array, size=image_size)
                            enhanced_metadata.update({
                                'pitch': float(pitch),
                                'yaw': float(yaw),
                                'roll': float(roll)
                            })
                    else:
                        # 只在第一次失败时打印警告
                        if not hasattr(_analyze_single_worker, '_landmark_warning_shown'):
                            print(f"[进程 {os.getpid()}] ⚠ Warning: 未检测到人脸特征点 for {filename}", flush=True)
                            _analyze_single_worker._landmark_warning_shown = True
                else:
                    # 标记器未初始化，只在第一次打印
                    if not hasattr(_analyze_single_worker, '_landmarker_none_shown'):
                        print(f"[进程 {os.getpid()}] ✗ InsightFace 106pt 标记器为 None，跳过 landmark 检测", flush=True)
                        _analyze_single_worker._landmarker_none_shown = True
            except Exception as e:
                # landmark 检测失败，打印错误信息
                if not hasattr(_analyze_single_worker, '_landmark_error_shown'):
                    print(f"\n[进程 {os.getpid()}] ✗ Landmark 检测错误: {e}", flush=True)
                    import traceback
                    traceback.print_exc()
                    _analyze_single_worker._landmark_error_shown = True
        
        # 添加分析时间戳
        enhanced_metadata['analyzed_timestamp'] = str(int(image_path.stat().st_mtime))
        
        return (filename, enhanced_metadata)
        
    except Exception as e:
        return (filename, {})


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description=S('ANALYZER_DESCRIPTION'),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  python Analyzer.py --input ".\\workspace\\data_dst\\aligned"
  python Analyzer.py -i ".\\workspace\\faces" --force
  python Analyzer.py -i ".\\workspace\\faces" --report report.txt
  python Analyzer.py -i ".\\workspace\\A" --merge ".\\workspace\\B"
  python Analyzer.py -i ".\\workspace\\faces" --features phash,pose
  python Analyzer.py -i ".\\workspace\\faces" --features embedding,histogram
  python Analyzer.py -i ".\\workspace\\faces" --features hue
  python Analyzer.py -i ".\\workspace\\faces" --features landmark  # 恢复人脸特征点
  python Analyzer.py -i ".\\workspace\\faces" --features landmark,pose  # 检测特征点并计算姿态
        """
    )
    
    parser.add_argument(
        '-i', '--input',
        type=str,
        required=True,
        help=S('ANALYZER_ARG_INPUT')
    )
    
    parser.add_argument(
        '--force',
        action='store_true',
        help=S('ANALYZER_ARG_FORCE')
    )
    
    parser.add_argument(
        '--report',
        type=str,
        default=None,
        help=S('ANALYZER_ARG_REPORT')
    )
    
    parser.add_argument(
        '--merge',
        type=str,
        default=None,
        help=S('ANALYZER_ARG_MERGE')
    )
    
    parser.add_argument(
        '--features',
        type=str,
        default='all',
        help=S('ANALYZER_ARG_FEATURES')
    )
    
    parser.add_argument(
        '--write-back',
        action='store_true',
        help=S('ANALYZER_ARG_WRITE_BACK')
    )
    
    parser.add_argument(
        '--skip-analysis',
        action='store_true',
        help='跳过分析步骤，直接执行后续操作（如回写元数据）'
    )
    
    parser.add_argument(
        '--workers',
        type=int,
        default=None,
        help=S('ANALYZER_ARG_WORKERS')
    )
    
    return parser.parse_args()


def main():
    """主函数"""
    print("""
==========================================================================
  █████╗ ███╗   ██╗ █████╗ ██╗  ██╗   ██╗███████╗███████╗██████╗ 
 ██╔══██╗████╗  ██║██╔══██╗██║  ╚██╗ ██╔╝╚══███╔╝██╔════╝██╔══██╗
 ███████║██╔██╗ ██║███████║██║   ╚████╔╝   ███╔╝ █████╗  ██████╔╝
 ██╔══██║██║╚██╗██║██╔══██║██║    ╚██╔╝   ███╔╝  ██╔══╝  ██╔══██╗
 ██║  ██║██║ ╚████║██║  ██║███████╗██║   ███████╗███████╗██║  ██║
 ╚═╝  ╚═╝╚═╝  ╚═══╝╚═╝  ╚═╝╚══════╝╚═╝   ╚══════╝╚══════╝╚═╝  ╚═╝
==========================================================================
    """)
    
    args = parse_args()
    
    # 打印用户输入参数
    print("\n" + "="*80)
    print("FacesetProcessor Analyzer - 参数配置")
    print("="*80)
    print(f"输入路径: {args.input}")
    print(f"强制重新分析: {'是' if args.force else '否'}")
    print(f"跳过分析: {'是' if args.skip_analysis else '否'}")
    print(f"回写元数据: {'是' if args.write_back else '否'}")
    print(f"合并模式: {args.merge if args.merge else '无'}")
    print(f"报告文件: {args.report if args.report else '控制台输出'}")
    print(f"工作进程数: {args.workers if args.workers else '自动检测'}")
    
    # 解析并显示特征选项
    features_list = [f.strip().lower() for f in args.features.split(',')]
    print(f"\n选择的特征: {', '.join(features_list)}")
    
    if 'all' in features_list:
        print("  → 将计算所有可用特征:")
        print("     - phash (感知哈希)")
        print("     - histogram (直方图分布)")
        print("     - hue (色相分布)")
        print("     - pose (人脸姿态)")
        print("     - embedding (人脸嵌入向量/FaceID)")
        print("     - landmark (人脸特征点检测/68pt)")
    else:
        print("  → 将计算以下特征:")
        feature_names = {
            'phash': '感知哈希',
            'histogram': '直方图分布',
            'hue': '色相分布',
            'pose': '人脸姿态',
            'embedding': '人脸嵌入向量/FaceID',
            'landmark': '人脸特征点检测/68pt'
        }
        for feat in features_list:
            name = feature_names.get(feat, feat)
            print(f"     - {feat} ({name})")
    
    print("="*80 + "\n")
    
    input_path = Path(args.input)
    if not input_path.exists():
        print(S('PATH_NOT_EXIST', input_path))
        return
    
    if not input_path.is_dir():
        print(S('INVALID_PATH_TYPE', input_path))
        return
    
    # 解析特征选项
    features_list = [f.strip().lower() for f in args.features.split(',')]
    
    # 创建分析器
    analyzer = MetadataAnalyzer(input_path, features=features_list)
    
    # 检查是否是合并模式
    if args.merge:
        source_path = Path(args.merge)
        print()
        
        # 执行合并
        merge_stats = analyzer.merge_datasets(source_path)
        print()
        
        # 合并后生成报告
        if args.report:
            analyzer.generate_report(Path(args.report))
        else:
            analyzer.generate_report()
    else:
        # 正常分析模式
        # 校验完整性
        integrity_result = analyzer.verify_database_integrity()
        print()
        
        # 如果指定了 --write-back，自动跳过分析步骤（只回写元数据）
        if args.write_back:
            print("检测到 --write-back 参数，跳过分析步骤，直接回写元数据")
            print()
        else:
            # 批量分析（支持多进程）
            analyzed_count = analyzer.analyze_batch(force_reanalyze=args.force, workers=args.workers)
            print()
        
        # 如果指定了 --write-back，将元数据写回 JPG 文件
        if args.write_back:
            print()
            written_count = analyzer.batch_write_metadata_to_jpgs(workers=args.workers)
            print()
        
        # 生成报告
        if args.report:
            analyzer.generate_report(Path(args.report))
        else:
            analyzer.generate_report()


if __name__ == '__main__':
    main()
