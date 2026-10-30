"""
SampleLoader V4 - 流式加载 + 动态扫描 + 多进程验证 + 预取缓冲

设计要点：
1. 初始化时立即扫描文件列表，get_batch() 可即时返回（无需等待验证完成）
2. 多进程预加载：独立进程负责图片解码 + DFL 特征提取，消除 GIL 竞争
3. 主进程通过 multiprocessing.Queue 接收已解码图片，无 I/O 等待
4. WAL 模式 SQLite 支持并发读写
5. 缓存文件跨会话保留
"""
import sys
import time
import random
import collections
import cv2
import math
import numpy as np
import threading
import queue as std_queue
import multiprocessing as mp
from pathlib import Path

_HERE = Path(__file__).resolve().parent          # samplelib/
_BASE = _HERE.parent                              # 项目根目录
sys.path.insert(0, str(_BASE))
sys.path.insert(0, str(_HERE))

from DFLIMG import DFLIMG
from facelib import LandmarksProcessor

# PAK 支持：已打包的 faceset 文件由 PackedFaceset 管理
from samplelib.PackedFaceset import PackedFaceset, packed_faceset_filename


# ------------------------------------------------------------------
# 模块级函数（可被 multiprocessing pickle，在子进程中运行）
# ------------------------------------------------------------------

def _quick_validate_jpeg(file_path):
    """快速验证文件是否为有效 JPEG/PNG（不解析 DFL 数据，<0.1ms/文件）"""
    try:
        if not file_path.exists():
            return None
        stat = file_path.stat()
        if stat.st_size < 1000:
            return None
        with open(file_path, 'rb') as f:
            header = f.read(4)
        if header[:2] in (b'\xff\xd8',) or header[:4] in (b'\x89PNG',):
            return str(file_path)
        return None
    except:
        pass
    return None


def _load_one_process(fpath_str, resolution):
    """
    加载单个样本：文件读取 + 解码 + resize + DFL 特征提取。
    每次都从 DFLJPG 文件直接读取，无任何缓存。
    """
    fpath = Path(fpath_str)
    if not fpath.exists():
        return None

    try:
        dflimg = DFLIMG.load(fpath)
        if dflimg is None or not dflimg.has_data():
            return None

        landmarks = dflimg.get_landmarks()
        pyr = None
        xseg_mask_raw = dflimg.get_xseg_mask() if dflimg.has_xseg_mask() else None
    except:
        return None

    # 读取 + 解码 + resize
    try:
        img_array = np.fromfile(str(fpath), dtype=np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        if img is None or img.size == 0:
            return None

        original_h, original_w = img.shape[:2]
        if resolution > 0:
            img = cv2.resize(img, (resolution, resolution))

        # 偏航角
        yaw_bucket = None
        if landmarks is not None:
            try:
                pyr = LandmarksProcessor.estimate_pitch_yaw_roll(landmarks, size=original_w)
                if pyr is not None:
                    yaw_deg = math.degrees(pyr[1])
                    yaw_bucket = math.floor(yaw_deg / 5.0) * 5.0
            except:
                pass

        # 使用 XSeg 遮罩
        xseg_mask = None
        if xseg_mask_raw is not None:
            if xseg_mask_raw.ndim == 3 and xseg_mask_raw.shape[2] > 1:
                xseg_mask_raw = xseg_mask_raw[..., :1]
            oh_x, ow_x = xseg_mask_raw.shape[:2]
            if resolution > 0 and (oh_x != resolution or ow_x != resolution):
                xseg_mask = cv2.resize(xseg_mask_raw, (resolution, resolution),
                                        interpolation=cv2.INTER_LINEAR)
            else:
                xseg_mask = xseg_mask_raw
        if xseg_mask is not None and xseg_mask.ndim == 2:
            xseg_mask = xseg_mask[..., None]

        return {
            'image': img,
            'filename': fpath.name,
            'file_path': fpath_str,
            'landmarks': landmarks,
            'pitch_yaw_roll': pyr,
            'yaw_bucket': yaw_bucket,
            'original_shape': (original_h, original_w),
            'xseg_mask': xseg_mask,
        }
    except Exception:
        return None


def _load_one_pak_sample(sample, resolution):
    """
    从 PAK 加载单个样本 —— 直接从 sample.read_raw_file() 读原始字节，
    不经过 DFLJPG（PAK 打包时已校验过数据完整性）。

    返回与 _load_one_process 相同格式的 dict（批量 / yaw / buffer 三者共用）。
    对 yaw 做 lazy 计算（与普通文件路径一致的时机）。
    """
    try:
        raw_bytes = sample.read_raw_file()
        if not raw_bytes:
            return None
        img_array = np.frombuffer(raw_bytes, dtype=np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        if img is None or img.size == 0:
            return None

        original_h, original_w = img.shape[:2]
        if resolution > 0:
            img = cv2.resize(img, (resolution, resolution))

        # yaw —— 从 sample 已预提取的 landmarks 计算（懒加载）
        landmarks = sample.landmarks
        yaw_bucket = None
        pyr = None
        if landmarks is not None:
            try:
                pyr = LandmarksProcessor.estimate_pitch_yaw_roll(
                    landmarks, size=original_w)
                if pyr is not None:
                    yaw_deg = math.degrees(pyr[1])
                    yaw_bucket = math.floor(yaw_deg / 5.0) * 5.0
            except Exception:
                pass

        # XSeg mask —— sample.get_xseg_mask() 会解码压缩缓存
        xseg_mask = None
        if sample.has_xseg_mask():
            try:
                xseg_mask_raw = sample.get_xseg_mask()
                if xseg_mask_raw is not None:
                    if xseg_mask_raw.ndim == 3 and xseg_mask_raw.shape[2] > 1:
                        xseg_mask_raw = xseg_mask_raw[..., :1]
                    oh_x, ow_x = xseg_mask_raw.shape[:2]
                    if resolution > 0 and (oh_x != resolution or ow_x != resolution):
                        xseg_mask = cv2.resize(xseg_mask_raw, (resolution, resolution),
                                               interpolation=cv2.INTER_LINEAR)
                    else:
                        xseg_mask = xseg_mask_raw
                    if xseg_mask is not None and xseg_mask.ndim == 2:
                        xseg_mask = xseg_mask[..., None]
            except Exception:
                pass

        return {
            'image': img,
            'filename': sample.filename,
            'file_path': sample.filename,  # PAK 无独立文件路径
            'landmarks': landmarks,
            'pitch_yaw_roll': pyr,
            'yaw_bucket': yaw_bucket,
            'original_shape': (original_h, original_w),
            'xseg_mask': xseg_mask,
        }
    except Exception:
        return None


def _prefetch_process_main(queue, stop_event, cmd_queue, file_paths,
                           resolution, batch_size, use_yaw_sampling,
                           pak_samples=None):
    """
    预加载子进程入口。

    支持两种模式：
      - 普通模式（pak_samples=None）：直接读取 DFLJPG 文件
      - PAK 模式（pak_samples=List[Sample]）：从打包 faceset 读取
    """
    import random

    is_pak = pak_samples is not None

    if is_pak:
        total = len(pak_samples)
        if total == 0:
            return
        indices = list(range(total))
        shuffled = list(indices)
    else:
        valid_paths = list(file_paths)
        if not valid_paths:
            return
        total = len(valid_paths)
        shuffled = list(valid_paths)

    random.shuffle(shuffled)
    scan_idx = 0
    yaw_groups = {}
    yaw_buckets = []
    yaw_group_known = set()

    def put_item_safe(item):
        while not stop_event.is_set():
            try:
                queue.put(item, timeout=0.5)
                return True
            except std_queue.Full:
                continue
        return False

    # 有限大小 LRU 缓存（防止无限增长耗尽系统内存）
    _CACHE_MAX = 256
    item_cache = collections.OrderedDict()
    deleted_set = set()

    def _remove_path(fp):
        nonlocal total
        if fp in deleted_set:
            return
        deleted_set.add(fp)
        item_cache.pop(fp, None)
        if use_yaw_sampling:
            yaw_group_known.discard(fp)
            for bucket in list(yaw_groups.keys()):
                try:
                    yaw_groups[bucket].remove(fp)
                except ValueError:
                    pass
                if not yaw_groups[bucket]:
                    del yaw_groups[bucket]
        if not is_pak:
            try:
                valid_paths.remove(fp)
            except ValueError:
                pass
            total = len(valid_paths)
        else:
            try:
                indices.remove(fp)
            except ValueError:
                pass
            total = len(indices)

    def _get_item(fp):
        if fp in deleted_set:
            return None
        if fp in item_cache:
            item_cache.move_to_end(fp)
            return item_cache[fp]
        if is_pak:
            item = _load_one_pak_sample(pak_samples[fp], resolution)
        else:
            item = _load_one_process(fp, resolution)
        if item is not None:
            item_cache[fp] = item
            if len(item_cache) > _CACHE_MAX:
                item_cache.popitem(last=False)
        else:
            _remove_path(fp)
        return item

    # 处理来自主进程的命令
    def _process_commands():
        while not cmd_queue.empty():
            try:
                cmd = cmd_queue.get_nowait()
                if cmd.get('op') == 'remove':
                    _remove_path(cmd['path'])
                    item_cache.pop(cmd['path'], None)
            except std_queue.Empty:
                break

    # 主预加载循环
    while not stop_event.is_set():
        # 处理主进程命令（删除文件等）
        _process_commands()

        # 全部文件都被判无效（不是 DFLJPG / 文件坏了）时，valid_paths 会被清空，
        # total 变 0 —— 此时下面的 load_count 恒为 0、loaded 恒为 0，这个循环会
        # 一直空转，而主进程 get_batch() 的退出条件要求"子进程已死"，于是整程序挂死。
        # 直接把子进程收掉，让 get_batch() 走 not is_alive() 分支正常返回。
        if total == 0:
            print("[SampleLoaderV4] [WARN] 没有可用样本（全部被丢弃），预取进程退出", flush=True)
            return

        # 每次加载 batch_size 个（背压由 queue.put() 的 maxsize 提供）
        load_count = min(batch_size * 4, total)

        loaded = 0

        if use_yaw_sampling:
            # 偏航角均衡采样
            if yaw_buckets:
                n = min(load_count, batch_size)
                classified = len(yaw_group_known)
                do_explore = n > 1 and classified < total

                if do_explore:
                    n_bucket = n - 1
                    n_explore = 1
                else:
                    n_bucket = n
                    n_explore = 0

                if len(yaw_buckets) < n_bucket:
                    selected = random.choices(yaw_buckets, k=n_bucket)
                else:
                    selected = random.sample(yaw_buckets, n_bucket)

                pool = indices if is_pak else valid_paths

                for bucket in selected:
                    paths = yaw_groups.get(bucket, [])
                    if not paths:
                        fp = random.choice(pool)
                    else:
                        fp = random.choice(paths)
                    item = _get_item(fp)
                    if item is not None:
                        # 更新 yaw 分组
                        yb = item.get('yaw_bucket')
                        if yb is not None and fp not in yaw_group_known:
                            yaw_group_known.add(fp)
                            if yb not in yaw_groups:
                                yaw_groups[yb] = []
                                yaw_buckets = sorted(yaw_groups.keys())
                            yaw_groups[yb].append(fp)
                        if put_item_safe(item):
                            loaded += 1

                if n_explore > 0:
                    fp = random.choice(pool)
                    item = _get_item(fp)
                    if item is not None and put_item_safe(item):
                        loaded += 1
            else:
                # 角度未就绪时随机采样
                pool_yaw = indices if is_pak else valid_paths
                for _ in range(load_count):
                    fp = random.choice(pool_yaw)
                    item = _get_item(fp)
                    if item is not None:
                        if put_item_safe(item):
                            loaded += 1
        else:
            # 普通模式：顺序扫描
            for _ in range(load_count):
                if scan_idx >= total:
                    scan_idx = 0
                    shuffled = list(indices if is_pak else valid_paths)
                    random.shuffle(shuffled)
                fp = shuffled[scan_idx]
                scan_idx += 1

                item = _get_item(fp)
                if item is not None and put_item_safe(item):
                    loaded += 1

        if loaded == 0:
            time.sleep(0.002)


# ------------------------------------------------------------------
# 加载器
# ------------------------------------------------------------------

class SampleLoaderV4:
    """高性能动态样本加载器 - 多进程验证 + 预取缓冲"""

    def __init__(self, aligned_path, batch_size=4, resolution=256, use_yaw_sampling=False):
        self.aligned_path = Path(aligned_path).resolve()
        self.batch_size = batch_size
        self.resolution = resolution
        self.use_yaw_sampling = use_yaw_sampling

        # 文件列表（初始化时扫描，支持运行时删除）
        self.file_paths = []
        self.total_file_count = 0
        self.is_pak = False
        self.pak_samples = None

        # 控制标志
        self.is_running = True
        self.validation_complete = True  # PAK 模式下无需验证

        # 统计
        self.total_batches_produced = 0
        self.total_images_received = 0

        # 多进程通信
        self.stop_event = mp.Event()
        self.cmd_queue = mp.Queue()
        self.prefetch_queue = mp.Queue(maxsize=batch_size * 32)
        self.prefetch_process = None
        self.consumer_thread = None

        # 本地缓冲（consumer 线程从 queue 拉取到这里）
        self.buffer = collections.deque(maxlen=batch_size * 16)
        self.buffer_lock = threading.Lock()

        print(f"[SampleLoaderV4] 正在初始化 {self.aligned_path.name} ...")

        # 1) 立即扫描文件（普通文件 或 PAK）
        self._scan_files()

        if self.total_file_count > 0:
            # 2) 启动预加载子进程（所有 I/O + decode + 特征提取在该进程完成）
            prefetch_args = (
                self.prefetch_queue,
                self.stop_event,
                self.cmd_queue,
                list(self.file_paths),
                resolution, batch_size, use_yaw_sampling,
            )
            if self.is_pak:
                prefetch_args = prefetch_args + (self.pak_samples,)
            else:
                prefetch_args = prefetch_args + (None,)
            self.prefetch_process = mp.Process(
                target=_prefetch_process_main,
                args=prefetch_args,
                daemon=True,
            )
            self.prefetch_process.start()

            # 3) consumer 线程：将 queue 中的 item 拉取到本地 buffer
            self.consumer_thread = threading.Thread(
                target=self._consume_loop, daemon=True
            )
            self.consumer_thread.start()

            # 4) 监控线程：检测子进程验证完成
            self._validation_detected = False
        else:
            print(f"[SampleLoaderV4] [WARN] 未找到图片文件: {self.aligned_path}")

    # ------------------------------------------------------------------
    # 内部方法
    # ------------------------------------------------------------------

    def _scan_files(self):
        """扫描目录，立即填充文件列表（支持 PAK 打包文件）"""
        if not self.aligned_path.exists():
            print(f"[SampleLoaderV4] [ERROR] 路径不存在: {self.aligned_path}")
            return

        pak_path = self.aligned_path / packed_faceset_filename
        if pak_path.exists():
            # PAK 模式：直接加载打包样本，不逐个验证文件
            self.is_pak = True
            try:
                samples = PackedFaceset.load(self.aligned_path)
            except Exception as e:
                print(f"[SampleLoaderV4] [ERROR] PAK 加载失败: {e}")
                return
            if samples is None or len(samples) == 0:
                print(f"[SampleLoaderV4] [WARN] PAK 文件为空: {pak_path}")
                return
            self.pak_samples = samples
            # file_paths 用伪路径占位（实际 prefetch 用 pak_samples 索引）
            self.file_paths = [f"pak://{i}" for i in range(len(samples))]
            self.total_file_count = len(samples)
            print(f"[SampleLoaderV4] PAK 加载完成: {self.total_file_count} 个打包样本")
        else:
            # 普通模式：扫描目录下的 jpg/png 文件
            valid_extensions = {'.jpg', '.jpeg'}
            paths = []
            for f in self.aligned_path.iterdir():
                if f.suffix.lower() in valid_extensions:
                    paths.append(str(f))

            self.file_paths = paths
            self.total_file_count = len(paths)
            print(f"[SampleLoaderV4] 扫描完成: {self.total_file_count} 个文件")

    def _consume_loop(self):
        """consumer 线程：从多进程 Queue 拉取 item 到本地 buffer。"""
        while self.is_running:
            # 本地缓冲满时暂停消费 → 回压到 Queue → 子进程 put() 阻塞
            with self.buffer_lock:
                buf_len = len(self.buffer)
            if buf_len >= self.buffer.maxlen:
                time.sleep(0.005)
                continue

            try:
                item = self.prefetch_queue.get(timeout=0.01)
                with self.buffer_lock:
                    self.buffer.append(item)
                self.total_images_received += 1

                # 检查子进程是否已完成验证（队列中有数据说明验证已完成）
                if not self._validation_detected:
                    self._validation_detected = True
                    self.validation_complete = True

            except std_queue.Empty:
                # 子进程已退出时排空剩余队列
                if self.prefetch_process is not None \
                        and not self.prefetch_process.is_alive():
                    while self.is_running:
                        try:
                            item = self.prefetch_queue.get_nowait()
                            with self.buffer_lock:
                                self.buffer.append(item)
                        except std_queue.Empty:
                            break
                    continue

    # ------------------------------------------------------------------
    # 公开接口
    # ------------------------------------------------------------------

    def get_batch(self):
        """获取一个批次（从缓冲中随机采样，返回随机组成的批次）。"""
        if not self.file_paths:
            return None

        batch = []

        # 1) 从缓冲中随机采样（仅从 buffer 读取，queue 由 consumer 线程独占）
        with self.buffer_lock:
            buf_len = len(self.buffer)
            if buf_len >= self.batch_size:
                idxs = random.sample(range(buf_len), self.batch_size)
                buf_list = list(self.buffer)
                batch = [buf_list[i] for i in idxs]
                skip = set(idxs)
                remaining = [buf_list[i] for i in range(buf_len) if i not in skip]
                self.buffer.clear()
                self.buffer.extend(remaining)

        # 2) 缓冲不足时等待 consumer 线程填充
        while len(batch) < self.batch_size:
            with self.buffer_lock:
                buf_len = len(self.buffer)
                if buf_len > 0:
                    need = self.batch_size - len(batch)
                    k = min(need, buf_len)
                    idxs = random.sample(range(buf_len), k)
                    buf_list = list(self.buffer)
                    batch.extend([buf_list[i] for i in idxs])
                    skip = set(idxs)
                    remaining = [buf_list[i] for i in range(buf_len) if i not in skip]
                    self.buffer.clear()
                    self.buffer.extend(remaining)
            if len(batch) >= self.batch_size:
                break
            # consumer 线程已退出且缓冲为空 → 无更多数据
            if self.prefetch_process is not None \
                    and not self.prefetch_process.is_alive() \
                    and len(self.buffer) == 0:
                break
            time.sleep(0.002)

        if batch:
            self.total_batches_produced += 1
            return batch
        return None

    def get_stats(self):
        """返回统计信息字典"""
        return {
            'count': self.total_file_count,
            'total_batches_produced': self.total_batches_produced,
            'total_images_loaded': self.total_images_received,
        }

    def remove_file(self, file_path):
        """从训练集中移除指定文件（从文件列表 + 子进程缓存中删除）。"""
        fp_str = str(Path(file_path).resolve())
        # 从主进程文件列表中移除
        try:
            self.file_paths.remove(fp_str)
        except ValueError:
            pass
        # 通知子进程移除
        try:
            self.cmd_queue.put_nowait({'op': 'remove', 'path': fp_str})
        except std_queue.Full:
            pass

    def clear_cache(self):
        """清空本地缓冲，下次读取时立即向预取队列拉取新数据。"""
        with self.buffer_lock:
            self.buffer.clear()

    def shutdown(self):
        """释放资源"""
        if not getattr(self, '_shutdown', False):
            self._shutdown = True
        else:
            return
        self.is_running = False

        # 通知子进程退出
        self.stop_event.set()

        # 排空队列，让子进程可以退出（如果它在 put 时阻塞）
        time.sleep(0.1)
        while self.prefetch_process is not None and self.prefetch_process.is_alive():
            try:
                self.prefetch_queue.get_nowait()
            except std_queue.Empty:
                break
            except Exception:
                break

        if self.prefetch_process is not None:
            self.prefetch_process.join(timeout=5)
            if self.prefetch_process.is_alive():
                self.prefetch_process.terminate()

        if self.consumer_thread is not None and self.consumer_thread.is_alive():
            self.consumer_thread.join(timeout=2)

        print(f"[SampleLoaderV4] [OK] 已关闭 {self.aligned_path.name}")
