"""
HDF5 Streaming Accessor - 流式 HDF5 元数据访问器
提供按需加载、内存高效的 HDF5 元数据访问
"""

import h5py
import numpy as np
from pathlib import Path
from typing import Dict, Optional


class H5StreamingAccessor:
    """
    HDF5 流式访问器

    特性：
    - 延迟加载：只在需要时读取数据
    - 缓存机制：已读取的数据会被缓存
    - 内存高效：不会一次性加载所有数据到内存
    """

    # 类级别计数器：跟踪 HDF5 损坏条目总数，避免重复警告刷屏
    _corrupt_count = 0
    _corrupt_threshold_warned = False

    def __init__(self, h5_file_path: Path):
        """
        初始化流式访问器

        Args:
            h5_file_path: HDF5 文件路径
        """
        self.h5_file_path = Path(h5_file_path)
        self._h5_file = None
        self._filename_index = {}  # original_filename -> safe_name
        self._cache = {}  # filename -> metadata dict
        self._corrupt_files = set()  # 已知损坏的文件名（避免重复尝试）

        if self.h5_file_path.exists():
            self._open()
    
    def _open(self):
        """打开 HDF5 文件并构建索引。重试一次以处理并发写入冲突。"""
        import time as _time
        for attempt in range(2):
            try:
                # 优先尝试读写模式（兼容其他读句柄），失败则回退只读
                mode = 'r+' if attempt == 0 else 'r'
                self._h5_file = h5py.File(self.h5_file_path, mode)

                # 构建文件名索引
                for safe_name in self._h5_file.keys():
                    grp = self._h5_file[safe_name]
                    original_filename = grp.attrs.get('__original_filename__', safe_name)
                    self._filename_index[original_filename] = safe_name
                return

            except Exception as e:
                err_str = str(e)
                if attempt == 0 and ('unable to open file' in err_str.lower() or 'permission' in err_str.lower()):
                    # 可能是并发写入导致锁冲突，等待后重试
                    _time.sleep(0.5)
                    continue
                if attempt == 1 or 'unable to synchronously open object' in err_str:
                    # B-tree 损坏等不可恢复错误，仅在第二次重试后打印
                    print(f"✗ Failed to open HDF5 file: {e}")
                self._h5_file = None
                self._filename_index = {}
    
    def close(self):
        """关闭 HDF5 文件"""
        if self._h5_file is not None:
            try:
                self._h5_file.close()
            except:
                pass
            self._h5_file = None
    
    def __del__(self):
        """析构时关闭文件"""
        self.close()
    
    @property
    def file_count(self) -> int:
        """获取文件数量（不加载实际数据）"""
        return len(self._filename_index)
    
    @property
    def filenames(self):
        """获取所有文件名列表"""
        return list(self._filename_index.keys())
    
    def get_metadata(self, filename: str, use_cache: bool = True) -> Dict:
        """
        获取单个文件的元数据（流式读取）
        
        Args:
            filename: 文件名
            use_cache: 是否使用缓存（默认True）
            
        Returns:
            元数据字典，如果文件不存在返回空字典
        """
        # 检查缓存
        if use_cache and filename in self._cache:
            return self._cache[filename]

        # 跳过已知损坏的条目
        if filename in self._corrupt_files:
            return {}

        # 从 HDF5 流式读取
        if self._h5_file is None or filename not in self._filename_index:
            return {}
        
        try:
            safe_name = self._filename_index[filename]
            grp = self._h5_file[safe_name]
            meta = {}
            
            # 读取 datasets（数组类型）
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
            
            # 读取 attributes（标量类型）
            for key, value in grp.attrs.items():
                if key == '__original_filename__':
                    continue
                meta[key] = value
            
            # 缓存结果
            if use_cache:
                self._cache[filename] = meta
            
            return meta
            
        except Exception as e:
            H5StreamingAccessor._corrupt_count += 1
            self._corrupt_files.add(filename)
            count = H5StreamingAccessor._corrupt_count
            if count <= 5:
                print(f"Warning: Failed to read metadata for {filename}: {e}")
                if count == 5:
                    print(f"  ... (further corruption warnings suppressed; run --repair to rebuild HDF5)")
            elif count % 500 == 0:
                print(f"  [HDF5 corruption] {count} entries failed so far. "
                      f"Tip: delete {self.h5_file_path.name} and re-run Analyzer to rebuild from JPG APP15 metadata.")
            return {}
    
    def has_file(self, filename: str) -> bool:
        """检查文件是否存在于 HDF5 中"""
        return filename in self._filename_index
    
    def get_field(self, filename: str, field_name: str, default=None):
        """
        获取单个字段值（更高效，只读取需要的字段）
        
        Args:
            filename: 文件名
            field_name: 字段名
            default: 默认值
            
        Returns:
            字段值或默认值
        """
        # 先尝试从缓存获取完整元数据
        if filename in self._cache:
            return self._cache[filename].get(field_name, default)
        
        # 跳过已知损坏的条目，从 HDF5 读取单个字段
        if self._h5_file is None or filename not in self._filename_index or filename in self._corrupt_files:
            return default
        
        try:
            safe_name = self._filename_index[filename]
            grp = self._h5_file[safe_name]
            
            # 先检查是否是 dataset
            if field_name in grp:
                dataset = grp[field_name]
                if isinstance(dataset, h5py.Dataset):
                    data = dataset[:]
                    if isinstance(data, np.ndarray):
                        if data.ndim == 0:
                            return data.item()
                        elif field_name in ['landmarks', 'embedding']:
                            return data
                        else:
                            return data.tolist()
                    return data
            
            # 再检查是否是 attribute
            if field_name in grp.attrs:
                return grp.attrs[field_name]
            
            return default

        except Exception as e:
            self._corrupt_files.add(filename)
            H5StreamingAccessor._corrupt_count += 1
            count = H5StreamingAccessor._corrupt_count
            if count <= 3:
                print(f"Warning: Failed to read field '{field_name}' for {filename}: {e}")
            return default
    
    def clear_cache(self):
        """清空缓存"""
        self._cache.clear()
    
    def has_fields(self, filename: str, field_names: set) -> dict:
        """
        批量检查多个字段是否存在。
        比多次 get_field 更高效（只需一次 HDF5 组访问）。

        Args:
            filename: 文件名
            field_names: 要检查的字段名集合

        Returns:
            {field_name: True/False} 字典
        """
        result = {f: False for f in field_names}
        # 先查缓存
        if filename in self._cache:
            meta = self._cache[filename]
            for f in field_names:
                result[f] = f in meta
            return result
        # 从 HDF5 读取
        if self._h5_file is None or filename not in self._filename_index:
            return result
        try:
            safe_name = self._filename_index[filename]
            grp = self._h5_file[safe_name]
            for f in field_names:
                result[f] = f in grp or f in grp.attrs
        except Exception:
            pass
        return result

    def preload_files(self, filenames: list):
        """
        预加载指定文件的元数据到缓存

        Args:
            filenames: 要预加载的文件名列表
        """
        for filename in filenames:
            if filename not in self._cache:
                self.get_metadata(filename, use_cache=True)
