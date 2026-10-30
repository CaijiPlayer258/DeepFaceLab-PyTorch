"""
Faceset Base Processor - 人脸集处理器基类
提供通用的 HDF5 流式访问、文件扫描等基础功能
供 Analyzer、Filter、Sorter 等模块继承复用
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 导入多语言支持
from FacesetProcessor.strings import S

# 导入流式 HDF5 访问器
from FacesetProcessor.H5StreamingAccessor import H5StreamingAccessor


class FacesetBaseProcessor:
    """
    人脸集处理器基类
    
    提供以下通用功能：
    - HDF5 元数据流式访问
    - 图像文件扫描
    - 元数据缓存管理
    - 通用工具方法
    
    子类应继承此类以获得统一的元数据访问接口
    """
    
    def __init__(self, faceset_path: Path, metadata_file: Optional[Path] = None):
        """
        初始化处理器
        
        Args:
            faceset_path: 人脸数据集路径
            metadata_file: 元数据文件路径（可选，默认为 faceset_path/metadata.h5）
        """
        self.faceset_path = Path(faceset_path)
        self.metadata_file = metadata_file or (self.faceset_path / "metadata.h5")
        
        # 流式 HDF5 访问器
        self._h5_accessor = None
        self._image_files = []
        
        # 初始化
        self._init_h5_accessor()
        self._scan_image_files()
    
    def _init_h5_accessor(self):
        """初始化流式 HDF5 访问器"""
        if self.metadata_file.exists():
            # 检查 HDF5 文件完整性
            import h5py
            if not h5py.is_hdf5(str(self.metadata_file)):
                print(f"[!] HDF5 file is corrupted (invalid signature), recreating: {self.metadata_file.name}")
                backup = self.metadata_file.with_suffix('.h5.corrupted')
                self.metadata_file.rename(backup)
                print(f"  → 已备份到 {backup.name}，可尝试手动恢复")
                print(S('ANALYZER_NO_METADATA'))
                self._h5_accessor = None
                return

            file_size = self.metadata_file.stat().st_size
            file_size_mb = file_size / (1024 * 1024)
            print(f"\n{'='*80}")
            print(f"Initializing streaming HDF5 accessor...")
            print(f"  File: {self.metadata_file.name}")
            print(f"  Size: {file_size_mb:.2f} MB")
            print(f"{'='*80}")

            # 创建流式访问器（自动打开并构建索引）
            self._h5_accessor = H5StreamingAccessor(self.metadata_file)

            print(f"✓ Indexed {self._h5_accessor.file_count} files in HDF5 (streaming mode)\n")
        else:
            print(S('ANALYZER_NO_METADATA'))
            self._h5_accessor = None
    
    def _scan_image_files(self):
        """扫描人脸数据集中的图像文件"""
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
        self._image_files = sorted([
            f for f in self.faceset_path.iterdir()
            if f.suffix.lower() in image_extensions and f.is_file()
        ])
        print(S('ANALYZER_FOUND_IMAGES', len(self._image_files)))
    
    @property
    def h5_accessor(self) -> Optional[H5StreamingAccessor]:
        """获取 HDF5 访问器"""
        return self._h5_accessor
    
    @property
    def image_files(self) -> List[Path]:
        """获取图像文件列表"""
        return self._image_files
    
    def get_metadata(self, filename: str) -> Dict:
        """
        获取单个文件的元数据（流式读取，带缓存）
        
        Args:
            filename: 文件名
            
        Returns:
            元数据字典，如果文件不存在返回空字典
        """
        if self._h5_accessor is None:
            return {}
        
        return self._h5_accessor.get_metadata(filename, use_cache=True)
    
    def get_field(self, filename: str, field_name: str, default=None):
        """
        获取单个字段值（更高效，只读取需要的字段）
        
        Args:
            filename: 文件名
            field_name: 字段名（如 'phash', 'landmarks', 'embedding' 等）
            default: 默认值
            
        Returns:
            字段值或默认值
        """
        if self._h5_accessor is None:
            return default
        
        return self._h5_accessor.get_field(filename, field_name, default)
    
    def has_metadata(self, filename: str) -> bool:
        """检查文件是否有元数据"""
        if self._h5_accessor is None:
            return False
        return self._h5_accessor.has_file(filename)

    def has_fields(self, filename: str, field_names: set) -> dict:
        """
        批量检查多个字段是否存在（只需一次 HDF5 组访问）。

        Args:
            filename: 文件名
            field_names: 要检查的字段名集合

        Returns:
            {field_name: bool} 字典
        """
        if self._h5_accessor is None:
            return {f: False for f in field_names}
        return self._h5_accessor.has_fields(filename, field_names)
    
    @property
    def metadata_count(self) -> int:
        """获取元数据文件中的条目数量"""
        if self._h5_accessor is None:
            return 0
        return self._h5_accessor.file_count
    
    @property
    def metadata_filenames(self) -> List[str]:
        """获取所有有元数据的文件名列表"""
        if self._h5_accessor is None:
            return []
        return self._h5_accessor.filenames
    
    def clear_metadata_cache(self):
        """清空元数据缓存（释放内存）"""
        if self._h5_accessor is not None:
            self._h5_accessor.clear_cache()
    
    def preload_metadata(self, filenames: List[str]):
        """
        预加载指定文件的元数据到缓存
        
        Args:
            filenames: 要预加载的文件名列表
        """
        if self._h5_accessor is not None:
            self._h5_accessor.preload_files(filenames)
    
    def get_metadata_snapshot(self) -> Dict:
        """
        获取所有元数据的完整快照（大数据集慎用）
        
        Returns:
            完整的元数据字典 {filename: metadata}
        """
        if self._h5_accessor is None:
            return {}
        
        print("Building full metadata snapshot (this may take a while for large datasets)...")
        snapshot = {}
        total = self._h5_accessor.file_count
        processed = 0
        
        import tqdm
        pbar = tqdm.tqdm(total=total, desc="Loading all metadata", unit="file", ascii=True)
        
        for filename in self._h5_accessor.filenames:
            snapshot[filename] = self._h5_accessor.get_metadata(filename, use_cache=True)
            processed += 1
            pbar.update(1)
        
        pbar.close()
        print(f"✓ Loaded {processed} metadata entries\n")
        
        return snapshot
    
    def verify_database_integrity(self) -> Dict:
        """
        校验数据库完整性
        
        Returns:
            校验结果字典
        """
        print(S('ANALYZER_VERIFYING_INTEGRITY'))
        
        # 使用流式访问器的索引
        metadata_files = set(self.metadata_filenames) if self._h5_accessor else set()
        actual_files = set([f.name for f in self._image_files])
        
        # 找出缺失的文件
        missing_in_actual = metadata_files - actual_files
        missing_in_metadata = actual_files - metadata_files
        
        result = {
            'total_metadata_entries': len(metadata_files),
            'total_actual_files': len(actual_files),
            'missing_in_actual': list(missing_in_actual),
            'missing_in_metadata': list(missing_in_metadata),
            'is_consistent': len(missing_in_actual) == 0 and len(missing_in_metadata) == 0
        }
        
        print(S('ANALYZER_INTEGRITY_RESULT', 
                result['total_metadata_entries'],
                result['total_actual_files'],
                len(missing_in_actual),
                len(missing_in_metadata)))
        
        if not result['is_consistent']:
            if missing_in_actual:
                print(S('ANALYZER_WARNING_MISSING_FILES', len(missing_in_actual)))
            if missing_in_metadata:
                print(S('ANALYZER_WARNING_NEW_FILES', len(missing_in_metadata)))
        
        return result
    
    def close(self):
        """关闭资源（HDF5 文件句柄等）"""
        if self._h5_accessor is not None:
            self._h5_accessor.close()
    
    def __del__(self):
        """析构时关闭资源"""
        self.close()
    
    # ========== 兼容旧代码的属性 ==========
    
    @property
    def metadata(self) -> Dict:
        """
        兼容旧代码的 metadata 属性
        注意：这会返回所有元数据的快照，大数据集慎用
        建议使用 get_metadata() 或 get_field() 代替
        """
        return self.get_metadata_snapshot()
