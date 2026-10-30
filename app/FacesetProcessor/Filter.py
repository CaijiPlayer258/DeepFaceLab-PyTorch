"""
DeepFaceLab Torch - FacesetProcessor Filter Module
人脸集过滤器模块：基于质量过滤和人脸ID分组
"""

import os
import sys
from pathlib import Path
import cv2
import numpy as np
import json
import shutil
import argparse
from typing import Dict, List, Optional, Tuple
import tqdm
import concurrent.futures
import multiprocessing
from multiprocessing import Pool
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_similarity

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 导入多语言支持
from strings import S

import warnings
warnings.filterwarnings('ignore', module='onnxruntime')

# ⚠️ onnxruntime 只能【惰性导入】，不能放模块顶部。
#    它带原生扩展，在 Qt 已经加载过 DLL 之后再 import 会以
#        WinError 1114 动态链接库(DLL)初始化例程失败
#    直接崩 —— 和 torch / c10.dll 是同一个机制（已实测复现）。
#    GUI 里只要有页面 import 本模块（例如将来做「按清晰度排序」），模块级 import
#    就会让整个 GUI 崩在启动阶段，而且报错信息跟 torch 那次完全不同，很难排查。
#    而排序 / 筛选两条路径都不需要 onnxruntime，只有 faceid（ArcFace 推理）才要。
_ORT_SILENCED = False


def _ort():
    """惰性导入 onnxruntime 并压低日志等级。只在真正要用推理会话时调用。"""
    global _ORT_SILENCED
    import onnxruntime
    if not _ORT_SILENCED:
        onnxruntime.set_default_logger_severity(3)
        _ORT_SILENCED = True
    return onnxruntime

# 导入项目模块
from facelib.LandmarksProcessor import get_image_hull_mask

# 导入基类
from FacesetBaseProcessor import FacesetBaseProcessor

# 工具函数
from core.pathex import batch_move_files

# InsightFace 相关 (可选,优先使用 ONNX)
try:
    from insightface.app import FaceAnalysis
    INSIGHTFACE_AVAILABLE = True
except ImportError:
    INSIGHTFACE_AVAILABLE = False


class ArcFaceONNXExtractor:
    """
    纯 ONNX 实现的 ArcFace embedding 提取器
    不需要安装 insightface 库，直接使用 .onnx 模型文件
    """
    
    def __init__(self, model_dir: Path, model_name: str = None):
        """
        初始化 ArcFace 提取器
        
        Args:
            model_dir: 模型目录路径
            model_name: 指定模型名称 (可选)，支持: w600k_mbf, mobilefacenet, w600k_r50
        """
        self.model_dir = Path(model_dir)
        self.model_name_used = None  # 记录实际使用的模型名称
        
        # 加载识别模型 (按优先级或指定名称)
        if model_name:
            # 使用指定的模型
            if not model_name.endswith('.onnx'):
                model_name += '.onnx'
            rec_model_path = self.model_dir / model_name
            if not rec_model_path.exists():
                raise FileNotFoundError(f"Specified model '{model_name}' not found in {self.model_dir}")
            self.model_name_used = model_name
        else:
            # 自动选择: w600k_mbf > w600k_r50
            rec_model_path = None
            for candidate_name in ["w600k_mbf.onnx", "w600k_r50.onnx"]:
                candidate = self.model_dir / candidate_name
                if candidate.exists():
                    rec_model_path = candidate
                    self.model_name_used = candidate_name
                    break
            
            if rec_model_path is None:
                raise FileNotFoundError(f"No recognition model found in {self.model_dir}")
        
        self.rec_session = _ort().InferenceSession(
            str(rec_model_path), 
            providers=['CPUExecutionProvider']
        )
        self.rec_input_name = self.rec_session.get_inputs()[0].name
        
        # 检测模型 (可选,已对齐的人脸不需要)
        # 注释掉检测模型加载,因为输入已经是裁剪好的人脸
    
    def detect_faces(self, img: np.ndarray) -> List[np.ndarray]:
        """
        检测人脸并返回边界框
        注意: 对于已对齐的人脸数据集，此方法不再使用
        
        Args:
            img: BGR 图像 (H, W, 3)
            
        Returns:
            人脸边界框列表 [(x1, y1, x2, y2), ...]
        """
        # 已对齐的人脸，返回整张图
        h, w = img.shape[:2]
        return [[0, 0, w, h]]
    
    def extract_embedding(self, face_img: np.ndarray) -> Optional[np.ndarray]:
        """
        从人脸图像提取 embedding
        
        Args:
            face_img: 人脸图像 (BGR格式)
            
        Returns:
            embedding 向量 (512维)，失败返回 None
        """
        try:
            # 预处理：resize 到 112x112
            face_resized = cv2.resize(face_img, (112, 112))
            
            # BGR -> RGB
            face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)
            
            # 归一化
            face_normalized = face_rgb.astype(np.float32) / 255.0
            
            # 标准化 (ArcFace 要求)
            mean = np.array([0.5, 0.5, 0.5], dtype=np.float32)
            std = np.array([0.5, 0.5, 0.5], dtype=np.float32)
            face_normalized = (face_normalized - mean) / std
            
            # HWC -> CHW
            face_transposed = np.transpose(face_normalized, (2, 0, 1))
            
            # 添加 batch 维度
            face_batch = np.expand_dims(face_transposed, axis=0)
            
            # 推理
            embedding = self.rec_session.run(
                None, 
                {self.rec_input_name: face_batch}
            )[0]
            
            # 归一化 embedding
            embedding = embedding.flatten()
            embedding = embedding / (np.linalg.norm(embedding) + 1e-8)
            
            return embedding
            
        except Exception as e:
            print(S('FILTER_EMBEDDING_ERROR', 'unknown', e))
            return None


# ============================================================================
# 清晰度指标：高频/低频能量比 E(b0)/E(b2)
# ============================================================================
# 旧实现用「拉普拉斯方差」，问题是它测的是【绝对值】，把内容对比度和锐度混在
# 一起：一张高对比的模糊图，分数会高过一张低对比的清晰图 —— 所以阈值在整批
# 图之间不可通用。实测（120 张对齐人脸 + 合成高斯模糊，256 分辨率）：
#
#     指标                    分离度@σ=2     内容变异系数 CV
#     拉普拉斯方差（缩到64）      1.49            0.45
#     高频/低频能量比 E(b0)/E(b2)      2.12            0.11     <-- 采用
#
# 原理：自然图像功率谱服从 1/f 律，在拉普拉斯金字塔上表现为「每个倍频程能量
# 大致相等」。模糊会砍掉最细的倍频程，于是 E(b0) 相对 E(b2) 塌陷。因为是
# 【比值】，内容对比度自动约掉，阈值才能在整批图之间通用。
#
# ⚠️ 掩码只在【求均值】时使用，绝不参与像素运算 —— 旧代码的
#    `image * mask` 会在掩码边界制造强度极高的假边缘，主导拉普拉斯响应。
#
# 分数方向与旧的拉普拉斯方差一致：越大越清晰。

_SHARP_LEVELS = 3
_DEFAULT_TRAIN_RES = 256


def _lap_bands(gray, levels=_SHARP_LEVELS):
    """严格可逆的拉普拉斯金字塔：down=INTER_AREA, up=NEAREST。"""
    bands, cur = [], gray
    for _ in range(levels - 1):
        dn = cv2.resize(cur, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
        up = cv2.resize(dn, (cur.shape[1], cur.shape[0]), interpolation=cv2.INTER_NEAREST)
        bands.append(cur - up)
        cur = dn
    bands.append(cur)
    return bands


# 噪声在金字塔各带里的方差系数（纯噪声实测 Var(b_k)=C_k*sigma^2）
#   Var(b0)/s^2 = 0.7500   Var(b1)/s^2 = 0.1875   Var(b2)/s^2 = 0.0625
_NOISE_COEF_B0 = 0.75
_NOISE_COEF_B2 = 0.0625
_NB_LEVELS = _SHARP_LEVELS


# 纯噪声下 |Laplacian 响应| 的 p10 与 σ 之比（实测标定，5 档 σ 稳定在 0.751）
_NOISE_P10_COEF = 0.751


def _noise_sigma(gray, mask=None):
    """稳健噪声估计：|Laplacian 响应| 的 p10 除以标定常数。

    为什么不用 Immerkær 的均值版：均值会被真实纹理抬高。实测在几乎没有噪声的
    人脸上，均值版估出 σ≈0.008，把整体中位分压低 18%、CV 从 0.251 抬到 0.271
    （等于引入了一层随内容变化的偏置）。p10 取的是最平滑那部分像素 —— 干净图上
    给出 0（完全不改动分数），有噪声的图上正常拾取：
        加噪 σ_n=0.04 时，无补偿虚高 4.96x → p10 补偿后 0.89x，且模糊单调性不变。
    """
    M = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], np.float32)
    r = np.abs(cv2.filter2D(gray.astype(np.float32), cv2.CV_32F, M,
                            borderType=cv2.BORDER_REPLICATE))
    v = r[mask > 0.5] if mask is not None else r.ravel()
    if not v.size:
        return 0.0
    return float(np.percentile(v, 10) / _NOISE_P10_COEF)


def _band_energy_ratio(gray, mask=None, noise_compensate=True):
    """高频 / 低频能量比 = E(b0) / E(b2)，越大越清晰。

    金字塔三层分解后：
        b0 = gray - up(avgpool2(gray))   256→128  带通，最细的细节
        b1 = d1   - up(avgpool2(d1))     128→64   带通
        b2 = d2                          64×64    低频残差（**不是**带通）

    分子 b0 是带通，均值恒为 0，所以 mean(x^2) 就等于方差。
    分母 b2 是低频残差，**它带着 DC（画面亮度）**，必须去掉 ——
    实测 DC 占了 mean(b2^2) 的 96.5%，用 mean(b2^2) 当分母的话，
    这个指标实际退化成「亮度归一化的高频能量」，而不是「高频/低频能量比」。
    去掉 DC 之后两项都变好：内容变异 CV 0.264 -> 0.240，模糊分离度 2.51 -> 2.96。

    noise_compensate=True 扣掉噪声对高频带的贡献。为什么必须扣：
    噪声本身是高频，实测给图加 sigma_n=0.04 的噪声会让这个分数虚高 **4.97 倍** ——
    而"多轮取 high"的筛法会把噪声图一路富集上来，最后留下的恰好是最噪的一批。
    扣掉之后同一批图是 0.89 倍（噪声确实在掩盖细节，方向正确），
    并且对真实模糊的单调性完全保留。
    """
    bands = _lap_bands(gray, _NB_LEVELS)
    b0, b2 = bands[0], bands[-1]
    done = False
    if mask is not None:
        m2 = cv2.resize(mask, (b2.shape[1], b2.shape[0]), interpolation=cv2.INTER_AREA)
        if int((mask > 0.5).sum()) >= 64 and int((m2 > 0.5).sum()) >= 16:
            v0 = (b0 * b0)[mask > 0.5]
            v2 = b2[m2 > 0.5]
            done = True
    if not done:
        v0 = (b0 * b0).ravel()
        v2 = b2.ravel()
    e0 = float(v0.mean())      # b0 是带通（均值 0），等价于方差
    e2 = float(v2.var())       # b2 是低频残差，必须去掉 DC
    if noise_compensate:
        ns2 = _noise_sigma(gray, mask) ** 2
        e0 = max(e0 - _NOISE_COEF_B0 * ns2, 0.0)
        e2 = max(e2 - _NOISE_COEF_B2 * ns2, 1e-12)
    return e0 / (e2 + 1e-12)


def prepare_gray_mask(image, mask=None, train_res=_DEFAULT_TRAIN_RES):
    """缩放到训练分辨率 + 灰度 + 掩码腐蚀。返回 (gray_float01, mask_or_None)

    归一化到统一分辨率很重要：降采样会等比缩小模糊半径，所以
    「相对模糊」在同一个 train_res 下才可比。
    """
    h, w = image.shape[:2]
    if max(h, w) != train_res:
        interp = cv2.INTER_AREA if max(h, w) > train_res else cv2.INTER_LINEAR
        image = cv2.resize(image, (train_res, train_res), interpolation=interp)
        if mask is not None:
            mask = cv2.resize(mask, (train_res, train_res), interpolation=cv2.INTER_NEAREST)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    if mask is not None:
        m = (mask > 0.5).astype(np.uint8)
        m = cv2.erode(m, np.ones((9, 9), np.uint8), iterations=1)   # 腐蚀 4px，避开掩码边界
        mask = m.astype(np.float32) if int(m.sum()) >= 256 else None
    return gray, mask


def face_sharpness_score(image, mask=None, train_res=_DEFAULT_TRAIN_RES,
                         noise_compensate=True):
    """人脸清晰度分数 = E(b0)/E(b2)。越大越清晰。"""
    gray, mask = prepare_gray_mask(image, mask, train_res)
    return _band_energy_ratio(gray, mask, noise_compensate)


def _load_face_mask(image_path, image, landmarks=None):
    """遮罩优先级：XSeg → hull(landmarks) → None（全图）。

    ⚠️ 标定和打分必须用同一套遮罩，否则两者量的不是同一个区域，阈值没有意义。
    """
    mask = None
    try:
        from DFLIMG.DFLJPG import DFLJPG as _J
        _inst = _J.load(image_path)
        if _inst is not None and _inst.has_data():
            xs = _inst.get_xseg_mask()
            if xs is not None:
                mask = (xs[:,:,0] > 0.5).astype(np.float32)
            if mask is not None and mask.shape[:2] != image.shape[:2]:
                mask = cv2.resize(mask, (image.shape[1], image.shape[0]),
                                  interpolation=cv2.INTER_NEAREST)
    except Exception:
        pass
    if mask is None and landmarks is not None:
        hull = get_image_hull_mask(image.shape, np.array(landmarks))
        mask = hull[:,:,0] if hull.ndim == 3 else hull
    return mask


def _calculate_sharpness_worker(args):
    """
    多进程工作函数：计算人脸区域清晰度（高频/低频能量比 E(b0)/E(b2)）。
    遮罩优先级：XSeg → hull(landmarks) → 全图。
    遮罩只用于【选择统计像素】，不参与像素运算。
    args = (path, landmarks, _[, train_res])
    """
    image_path_str, landmarks, _ = args[0], args[1], args[2]
    train_res = args[3] if len(args) > 3 else _DEFAULT_TRAIN_RES
    try:
        image = cv2.imread(image_path_str)
        if image is None:
            return (Path(image_path_str).name, 0.0, "Failed to read image")

        mask = _load_face_mask(image_path_str, image, landmarks)
        return (Path(image_path_str).name, face_sharpness_score(image, mask, train_res), None)
    except Exception as e:
        return (Path(image_path_str).name, 0.0, str(e))


# 标定用的合成模糊档位（单位：train_res 下的像素）
CAL_SIGMAS = (0.0, 0.15, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0)


def _calibrate_sharp_thresholds(image_files, train_res=_DEFAULT_TRAIN_RES, sample: int = 96):
    """用合成高斯模糊自动标定，产出一条【分数 -> 等效模糊 σ】的标定曲线。

    对同一批样本分别施加 CAL_SIGMAS 里的每一档高斯模糊，取分数的中位数，
    得到 (σ -> 中位分数) 的单调递减曲线。有了它，任何一个分数都能反查成
    「这张图相当于被 σ=X 高斯模糊过」—— 这才是可跨轮次、跨数据集比较的分级。

    同时给出三个绝对阈值（σ=2 / σ=1 / 清晰样本 60 分位）。
    返回 None 表示样本不足，调用方应退回分位数阈值。
    """
    if not image_files:
        return None
    rng = np.random.default_rng(0)
    idx = rng.choice(len(image_files), size=min(sample, len(image_files)), replace=False)
    # 先读样本（原图 + 与 worker 一致的遮罩 + 缩放比）
    samples = []
    for i in idx:
        img = cv2.imread(str(image_files[int(i)]))
        if img is None:
            continue
        mask = _load_face_mask(str(image_files[int(i)]), img)
        samples.append((img, mask, max(img.shape[:2]) / float(train_res)))
    if len(samples) < 8:
        return None

    # 曲线用 90 分位作锚：σ=0 的点代表"本数据集里最锐的那批"。
    # 用中位数当锚的话，一半的图会全部挤在 σ=0，分级就废了；
    # 用 90 分位则只有最锐的 ~10% 会读到 σ=0，四档都有区分度。
    curve = {}
    for sg in CAL_SIGMAS:
        vals = []
        for img, mask, scale in samples:
            g = img if sg == 0.0 else cv2.GaussianBlur(img, (0, 0), sg * scale)
            vals.append(face_sharpness_score(g, mask, train_res))
        curve[sg] = float(np.percentile(vals, 90))

    return {
        'curve': curve,
        'clean': [curve[0.0]],      # 兼容旧字段
        'thr_blurry': float(curve[2.0]),
        'thr_low':    float(curve[1.0]),
        'thr_medium': float(np.percentile([curve[0.0]], 60)),
    }


def equivalent_blur_sigma(score, curve):
    """把分数反查成"等效高斯模糊 σ"。曲线单调递减，线性插值。

    这是分级的可比较单位：第 N 轮的 high 组中位 σ 如果不再下降，就说明
    已经触到这批素材的清晰度天花板，可以停止继续筛了。
    """
    if not curve:
        return None
    sigs = sorted(curve.keys())
    vals = [curve[s] for s in sigs]
    if score >= vals[0]:
        return 0.0
    if score <= vals[-1]:
        return float(sigs[-1])
    for i in range(len(sigs) - 1):
        hi_v, lo_v = vals[i], vals[i + 1]
        if hi_v >= score >= lo_v:
            d = hi_v - lo_v
            t = 0.0 if d <= 1e-15 else (hi_v - score) / d
            return float(sigs[i] + t * (sigs[i + 1] - sigs[i]))
    return float(sigs[-1])


def _load_metadata_h5(metadata_file: Path) -> Dict:
    """
    通用函数：加载 HDF5 格式的元数据
    
    Args:
        metadata_file: 元数据文件路径
        
    Returns:
        元数据字典
    """
    import h5py
    
    if not metadata_file.exists():
        print(S('ANALYZER_NO_METADATA'))
        return {}
    
    try:
        # 显示文件大小
        file_size = metadata_file.stat().st_size
        file_size_mb = file_size / (1024 * 1024)
        print(f"\nLoading metadata from: {metadata_file.name}")
        print(f"File size: {file_size_mb:.2f} MB")
        
        print("Reading HDF5...", end=" ", flush=True)
        
        metadata_dict = {}
        with h5py.File(metadata_file, 'r') as f:
            # 遍历所有组（每个组对应一个文件）
            for safe_name in f.keys():
                grp = f[safe_name]
                meta = {}
                
                # 读取 datasets（数组类型）
                for key in grp.keys():
                    dataset = grp[key]
                    if isinstance(dataset, h5py.Dataset):
                        data = dataset[:]
                        # 对于 landmarks 和 embedding，保持为 numpy 数组
                        # 其他数组转换为 list
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
                    # 跳过内部使用的属性
                    if key == '__original_filename__':
                        continue
                    meta[key] = value
                
                # 使用存储的原始文件名
                original_filename = grp.attrs.get('__original_filename__', safe_name)
                metadata_dict[original_filename] = meta
        
        print(f"\r{' ' * 80}\r", end="", flush=True)
        print(S('FILTER_METADATA_LOADED', len(metadata_dict)))
        return metadata_dict
        
    except ImportError:
        print("\n✗ 错误: 需要安装 h5py 库来读取 HDF5 格式")
        print("请运行: pip install h5py\n")
        return {}
    except Exception as e:
        print(S('ANALYZER_METADATA_LOAD_ERROR', e))
        import traceback
        traceback.print_exc()
        return {}


class QualityFilter(FacesetBaseProcessor):
    """质量过滤器 - 基于拉普拉斯方差检测模糊
    
    继承自 FacesetBaseProcessor，自动获得流式 HDF5 访问能力
    """
    
    def __init__(self, faceset_path: Path, metadata_file: Optional[Path] = None):
        """
        初始化质量过滤器
        
        Args:
            faceset_path: 人脸数据集路径
            metadata_file: 元数据文件路径（可选）
        """
        # 调用父类初始化（会自动初始化 HDF5 访问器和扫描文件）
        super().__init__(faceset_path, metadata_file)
    
    def calculate_face_sharpness(self, image: np.ndarray, landmarks: List[List[float]] = None,
                                 train_res: int = _DEFAULT_TRAIN_RES) -> float:
        """
        计算人脸清晰度 = E(b0)/E(b2)（高频/低频能量比），越大越清晰。

        与旧版拉普拉斯方差的区别：这是【比值】而非绝对值，内容对比度自动约掉，
        所以阈值可以跨图片通用（实测分离度 1.49 -> 2.12，内容变异 0.45 -> 0.11）。

        Args:
            image: BGR格式图像
            landmarks: 68个特征点坐标（可选；给了就用 hull 遮罩限定统计区域）
            train_res: 归一化分辨率，应与训练分辨率一致（默认 256）

        Returns:
            E(b0)/E(b2)
        """
        try:
            mask = None
            if landmarks is not None:
                hull = get_image_hull_mask(image.shape, np.array(landmarks))
                mask = hull[:,:,0] if hull.ndim == 3 else hull
            return float(face_sharpness_score(image, mask, train_res))
        except Exception as e:
            print(S('FILTER_SHARPNESS_ERROR', e))
            return 0.0
    
    def filter_by_quality(self, threshold: float = 0.0, workers: int = None,
                          train_res: int = _DEFAULT_TRAIN_RES,
                          percentiles: Tuple[float, float, float] = (10, 30, 60),
                          absolute: bool = False) -> Dict:
        """
        按清晰度 E(b0)/E(b2)（高频/低频能量比）分 4 组：
          blurry / low_quality / medium_quality / high_quality

        阈值不再写死 —— 旧的 10/20/40 是【拉普拉斯方差】的量纲，对新指标无意义。
        现在提供两套机制：

          absolute=False（默认，相对分位）
              按本数据集自身的 percentiles 分位切（默认 10/30/60）。
              任何数据集都能分组，适合"从现有素材里淘汰最差的一批"。

          absolute=True（绝对基准）
              用合成高斯模糊定基准（对样本做 σ=1 / σ=2 高斯模糊）：
                  blurry < median(σ=2)    low < median(σ=1)    medium < 清晰样本 60 分位
              适合"我就要一个固定质量门槛，低于它一律不要"。
              注意：如果数据集整体都很锐，这个模式会正确地判定"没有模糊图"，
              此时分组可能是空的那就说明你的数据确实没问题。

        两种模式都会打印 σ=0/1/2 的基准分数，方便判断数据集的绝对水平。

        Args:
            threshold: 保留兼容，未使用
            workers: 工作进程数
            train_res: 归一化分辨率，应与训练分辨率一致（默认 256）
            percentiles: 相对分位模式的切点（分位越低越糊）
            absolute: 是否使用绝对基准阈值
        Returns:
            过滤统计结果
        """
        print(f"按清晰度分组（E(b0)/E(b2) 高频/低频能量比，归一化到 {train_res}px）")
        _dirs = {
            'blurry': self.faceset_path / "blurry",
            'low':    self.faceset_path / "low_quality",
            'medium': self.faceset_path / "medium_quality",
            'high':   self.faceset_path / "high_quality",
        }
        for d in _dirs.values():
            d.mkdir(exist_ok=True)

        stats = {'total': 0, 'errors': 0}
        for k in _dirs:
            stats[k] = 0
        
        # 扫描所有图片
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
        image_files = sorted([
            f for f in self.faceset_path.iterdir()
            if f.suffix.lower() in image_extensions and f.is_file()
        ])
        
        print(S('FILTER_FOUND_IMAGES', len(image_files)))
        
        # ========== 第一阶段：计算并标记（多进程） ==========
        print(S('FILTER_PHASE1_CALCULATING'))
        
        # 任务列表：只传路径，worker 内部自行读取 XSeg/landmarks
        tasks = [(str(p), None, None, train_res) for p in image_files]
        
        # 设置工作进程数
        if workers is None:
            workers = min(multiprocessing.cpu_count(), len(tasks))
        
        print(S('FILTER_USING_WORKERS', workers))
        
        # 多进程计算
        sharpness_results = {}
        if tasks:
            with Pool(processes=workers) as pool:
                pbar = tqdm.tqdm(total=len(tasks), desc=S('FILTER_PROGRESS_QUALITY'), unit="img", ascii=True)
                
                for filename, sharpness, error in pool.imap_unordered(_calculate_sharpness_worker, tasks):
                    if error:
                        print(S('FILTER_PROCESS_ERROR', filename, error))
                        stats['errors'] += 1
                    else:
                        sharpness_results[filename] = sharpness
                        stats['total'] += 1
                    pbar.update(1)
                
                pbar.close()
        
        # ========== 第二阶段：标定阈值 + 分类 ==========
        print(S('FILTER_PHASE2_CLASSIFYING'))

        # 合成高斯模糊标定：把"分数"翻译成"等效模糊 σ"，分级才有绝对含义
        cal = _calibrate_sharp_thresholds(image_files, train_res)
        curve = cal['curve'] if cal else None
        if curve:
            print("  标定曲线（合成高斯模糊 → 该档模糊下样本的中位分数）：")
            for _i in range(0, len(CAL_SIGMAS), 5):
                print("    " + "   ".join(f"σ={s:g}→{curve[s]:.5f}" for s in CAL_SIGMAS[_i:_i+5]))
        else:
            print("  ⚠️ 样本不足，跳过合成模糊标定（等效 σ 不可用）")
        m1 = curve[1.0] if curve else None

        vals = np.asarray(list(sharpness_results.values()), dtype=np.float64)
        if not vals.size:
            thr_blurry = thr_low = thr_medium = 0.0
        elif absolute and cal is not None and cal['thr_blurry'] < cal['thr_low'] < cal['thr_medium']:
            thr_blurry, thr_low, thr_medium = cal['thr_blurry'], cal['thr_low'], cal['thr_medium']
            print("  阈值模式：绝对基准（blurry = 比 σ=2 更糊；low = 比 σ=1 更糊）")
        else:
            p = np.percentile(vals, list(percentiles))
            thr_blurry, thr_low, thr_medium = float(p[0]), float(p[1]), float(p[2])
            print(f"  阈值模式：相对分位 {tuple(percentiles)}%（数据集自身中位 {float(np.median(vals)):.5f}）")
        print(f"  最终阈值：blurry<{thr_blurry:.5f}  low<{thr_low:.5f}  medium<{thr_medium:.5f}")
        if m1 is not None:
            n_vs1 = int((vals < m1).sum())
            print(f"  参考：全集中位 {float(np.median(vals)):.5f} vs σ=1 基准 {m1:.5f} —— "
                  f"有 {n_vs1}/{vals.size} 张比 σ=1 模糊更糊")

        move_list = []
        bucket_scores = {'blurry': [], 'low': [], 'medium': [], 'high': []}
        for p in image_files:
            v = sharpness_results.get(p.name)
            if v is None:
                continue
            k = ('blurry' if v < thr_blurry else 'low' if v < thr_low
                 else 'medium' if v < thr_medium else 'high')
            bucket_scores[k].append(v)
            move_list.append((p, _dirs[k] / p.name))
            stats[k] += 1
        if move_list:
            batch_move_files(move_list, self.faceset_path)
        stats['total'] = len(image_files)

        # ---------- 分组报告：张数 + 分数区间 + 中位等效模糊 σ ----------
        _notes = {'blurry': '明显糊', 'low': '偏软', 'medium': '可接受', 'high': '锐利'}
        print('清晰度分组完成：')
        print('  %-14s %6s  %-21s  %-14s %s' % ('分组', '张数', '分数区间', '中位等效 blur σ', '说明'))
        for k in ('blurry', 'low', 'medium', 'high'):
            vs = bucket_scores[k]
            if vs:
                rng_txt = '%.5f ~ %.5f' % (min(vs), max(vs))
                eq = equivalent_blur_sigma(float(np.median(vs)), curve)
                eq_txt = ('σ≈%.2f' % eq) if eq is not None else '—'
            else:
                rng_txt, eq_txt = '—', '—'
            print('  %-14s %6d  %-21s  %-14s %s' % (k, stats[k], rng_txt, eq_txt, _notes[k]))
        print(f"  错误: {stats['errors']} 张")

        # ---------- 多轮筛选的进入/停止判据 ----------
        if vals.size:
            all_med = float(np.median(vals))
            hi = bucket_scores['high']
            hi_med = float(np.median(hi)) if hi else None
            e_all = equivalent_blur_sigma(all_med, curve)
            line = f"  本轮全集：中位分 {all_med:.5f}"
            if e_all is not None:
                line += f"（等效 blur σ≈{e_all:.2f}）"
            print(line)
            if hi_med is not None:
                e_hi = equivalent_blur_sigma(hi_med, curve)
                line = f"  high 组：中位分 {hi_med:.5f}"
                if e_hi is not None:
                    line += f"（等效 blur σ≈{e_hi:.2f}）"
                print(line)
            print("  多轮筛：把 high/ 当输入再跑一次即可 —— 每轮都会砍掉当前分布较糊的一段。")
            print("  ⚠️ 等效 σ 是相对【本轮输入集】最锐的那批标的，跨轮次不可直接比；"
                  "判断该不该继续筛，看 high 组的【中位分】是否还在明显上升。")
        return stats


class FaceIDFilter(FacesetBaseProcessor):
    """人脸ID过滤器 - 基于ArcFace embedding分组
    
    继承自 FacesetBaseProcessor，自动获得流式 HDF5 访问能力
    """
    
    def __init__(self, faceset_path: Path, metadata_file: Optional[Path] = None, model_name: str = None, skip_model_init: bool = False):
        """
        初始化人脸ID过滤器
        
        Args:
            faceset_path: 人脸数据集路径
            metadata_file: 元数据文件路径（可选）
            model_name: 指定使用的模型名称 (可选)，支持: w600k_mbf, mobilefacenet, w600k_r50
            skip_model_init: 跳过模型初始化（用于 merge-only 模式）
        """
        # 调用父类初始化（会自动初始化 HDF5 访问器和扫描文件）
        super().__init__(faceset_path, metadata_file)
        
        # 如果跳过模型初始化（merge-only 模式），直接返回
        if skip_model_init:
            self.arcface_extractor = None
            self.use_insightface = False
            return
        
        # 判断是否需要模型：采样前 10 个文件检查 embedding 覆盖率
        # 全量扫描太慢，采样足够判断"全部缓存"或"全部缺失"的常见场景
        if self.metadata_count > 0:
            sample_missing = 0
            for filename in list(self.metadata_filenames)[:10]:
                if not self.has_fields(filename, {'embedding'}).get('embedding'):
                    sample_missing += 1
            if sample_missing == 0:
                print(f"✓ HDF5 中有 {self.metadata_count} 条元数据（采样前10均含 embedding）")
                print(f"  → 将直接使用缓存的 embedding，无需加载模型\n")
                self.arcface_extractor = None
                self.use_insightface = False
                return
            else:
                print(f"⚠ 采样 10 个文件中有 {sample_missing} 个缺少 embedding")
                print(f"  → 将加载模型并提取缺失的 embedding\n")
        
        # 初始化 ArcFace ONNX 提取器
        # 模型在 modelhub/onnx/ArcFace/ 下
        project_root = Path(__file__).parent.parent
        model_dir = project_root / "modelhub" / "onnx" / "ArcFace"
        
        if model_dir.exists():
            try:
                # 如果未指定模型且有多个可用，询问用户
                if model_name is None:
                    available_models = []
                    for candidate_name in ["w600k_mbf.onnx", "w600k_r50.onnx"]:
                        if (model_dir / candidate_name).exists():
                            available_models.append(candidate_name.replace('.onnx', ''))
                    
                    if len(available_models) > 1:
                        print(f"\n{S('FILTER_AVAILABLE_MODELS')}: {', '.join(available_models)}")
                        print(S('FILTER_SELECT_MODEL'))
                        for i, name in enumerate(available_models, 1):
                            marker = f" {S('FILTER_RECOMMENDED')}" if i == 1 else ""
                            print(f"  {i}. {name}{marker}")
                        
                        try:
                            choice = input(f"\n{S('FILTER_ENTER_CHOICE').format(1, len(available_models), 1)}").strip()
                            if choice.isdigit() and 1 <= int(choice) <= len(available_models):
                                model_name = available_models[int(choice) - 1]
                            elif choice == '':
                                model_name = available_models[0]  # 默认第一个
                            else:
                                print(S('FILTER_INVALID_CHOICE').format(available_models[0]))
                                model_name = available_models[0]
                        except (EOFError, KeyboardInterrupt):
                            print(f"\n{S('FILTER_USING_DEFAULT').format(available_models[0])}")
                            model_name = available_models[0]
                        print()
                
                self.arcface_extractor = ArcFaceONNXExtractor(model_dir, model_name=model_name)
                self.use_insightface = False
                # 显示使用的模型
                if self.arcface_extractor.model_name_used:
                    print(f"Using model: {self.arcface_extractor.model_name_used}\n")
            except Exception as e:
                print(S('FILTER_ARCFACE_ONNX_FAILED', e))
                self.arcface_extractor = None
                self.use_insightface = False
        else:
            self.arcface_extractor = None
            self.use_insightface = False
        
        # 如果 ONNX 失败，尝试使用 InsightFace
        if not self.arcface_extractor and INSIGHTFACE_AVAILABLE:
            print(S('FILTER_INIT_INSIGHTFACE'))
            try:
                self.app = FaceAnalysis(providers=['CPUExecutionProvider'])
                self.app.prepare(ctx_id=0, det_size=(640, 640))
                self.use_insightface = True
                print(S('FILTER_INSIGHTFACE_READY'))
            except Exception as e:
                print(S('FILTER_INSIGHTFACE_FAILED', e))
                self.use_insightface = False
        
        if not self.arcface_extractor and not self.use_insightface:
            # 不打印警告，因为可以从元数据中读取 embedding
            pass
    
    def extract_embedding(self, image_path: Path) -> Optional[np.ndarray]:
        """
        提取人脸 embedding 向量
        
        Args:
            image_path: 图像路径
            
        Returns:
            embedding 向量 (512维)，失败返回 None
        """
        try:
            # 读取图像
            img = cv2.imread(str(image_path))
            if img is None:
                print(S('FILTER_IMAGE_READ_ERROR', image_path.name))
                return None
            
            # 优先使用 ArcFace ONNX
            if self.arcface_extractor:
                # 对于已对齐的人脸，直接提取 embedding
                emb = self.arcface_extractor.extract_embedding(img)
                if emb is None:
                    print(S('FILTER_ARCFACE_RETURNED_NONE', image_path.name))
                return emb
            
            # 其次使用 InsightFace
            elif self.use_insightface and hasattr(self, 'app'):
                faces = self.app.get(img)
                if len(faces) == 0:
                    print(S('FILTER_INSIGHTFACE_NO_FACE', image_path.name))
                    return None
                best_face = max(faces, key=lambda x: x.det_score)
                return best_face.embedding
            
            else:
                # 没有任何可用的提取器
                if not hasattr(self, '_extraction_warning_shown'):
                    print(S('FILTER_NO_EXTRACTION_MODEL'))
                    print(f"  - ArcFace ONNX: {'✓' if self.arcface_extractor else '✗'}")
                    print(f"  - InsightFace: {'✓' if self.use_insightface else '✗'}")
                    print(f"\n{S('FILTER_CHECK_MODEL_PATH')}")
                    print(S('FILTER_SUPPORTED_MODELS'))
                    print(S('FILTER_CURRENT_MODEL_PATH', Path(__file__).parent.parent / 'modelhub'))
                    print()
                    self._extraction_warning_shown = True
                return None
            
        except Exception as e:
            print(S('FILTER_EMBEDDING_ERROR', image_path.name, e))
            import traceback
            traceback.print_exc()
            return None
    
    def calculate_embedding_similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """
        计算两个 embedding 的余弦相似度
        
        Args:
            emb1: 第一个 embedding 向量
            emb2: 第二个 embedding 向量
            
        Returns:
            余弦相似度 (0-1)，越接近1表示越相似
        """
        try:
            # 归一化
            emb1_norm = emb1 / (np.linalg.norm(emb1) + 1e-8)
            emb2_norm = emb2 / (np.linalg.norm(emb2) + 1e-8)
            
            # 余弦相似度
            similarity = np.dot(emb1_norm, emb2_norm)
            
            # 转换到 0-1 范围
            similarity = (similarity + 1.0) / 2.0
            
            return float(similarity)
        except Exception as e:
            print(S('FILTER_SIMILARITY_ERROR', e))
            return 0.0
    
    def calculate_landmark_similarity(self, landmarks1: List[List[float]], 
                                     landmarks2: List[List[float]]) -> float:
        """
        计算两组特征点的相似度（归一化欧氏距离）
        
        Args:
            landmarks1: 第一组68个特征点
            landmarks2: 第二组68个特征点
            
        Returns:
            相似度分数 (0-1)，越接近1表示越相似
        """
        try:
            pts1 = np.array(landmarks1, dtype=np.float32)
            pts2 = np.array(landmarks2, dtype=np.float32)
            
            # 计算欧氏距离
            distances = np.sqrt(np.sum((pts1 - pts2) ** 2, axis=1))
            
            # 归一化：除以人脸尺寸（使用两眼之间的距离作为参考）
            eye_dist = np.sqrt(np.sum((pts1[36] - pts1[45]) ** 2))
            if eye_dist == 0:
                eye_dist = 1.0
            
            normalized_dist = np.mean(distances) / eye_dist
            
            # 转换为相似度（距离越小，相似度越高）
            similarity = 1.0 / (1.0 + normalized_dist)
            
            return float(similarity)
        except Exception as e:
            print(S('FILTER_SIMILARITY_ERROR', e))
            return 0.0
    
    def filter_by_face_id(self, eps: float = 0.3, min_samples: int = 2, merge_back: bool = False, output_dir: str = None) -> Dict:
        """
        根据人脸ID分组，将相似的人脸移动到同一文件夹
        使用 DBSCAN 聚类算法进行准确的人脸识别分组
        
        Args:
            eps: DBSCAN 的邻域半径参数（余弦距离空间）
                 默认 0.3 表示余弦相似度 >= 0.7 视为相似
                 值越小，分组越严格；值越大，分组越宽松
            min_samples: DBSCAN 的最小样本数，默认2
            merge_back: 是否将分组后的文件合并回一个文件夹
            output_dir: 合并后的输出目录名称（默认: aligned）
            
        Returns:
            分组统计结果
        """
        print(S('FILTER_START_FACEID_FILTER', f'eps={eps}'))
        print(S('FILTER_DBSCLAN_INFO', eps, min_samples))
        
        # 扫描所有图片文件
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
        image_files = sorted([
            f for f in self.faceset_path.iterdir()
            if f.suffix.lower() in image_extensions and f.is_file()
        ])
        
        if not image_files:
            print(S('FILTER_NO_IMAGES_FOUND'))
            return {}
        
        print(S('FILTER_FOUND_IMAGES', len(image_files)))
        
        # ========== 第一阶段：提取 embedding（多进程并行）==========
        print(S('FILTER_PHASE1_EXTRACTING_EMBEDDINGS'))

        embeddings_dict = {}  # {filename: embedding}
        filename_list = []  # 保持顺序
        failed_files = []
        saved_count = 0

        # 第一遍：快速检查文件属性，分离有缓存的 vs 需要计算的
        missing_files = []
        for img_file in tqdm.tqdm(image_files, desc="检查缓存", unit="img", ascii=True):
            filename = img_file.name
            emb_list = self.get_field(filename, 'embedding')
            if emb_list is not None:
                emb = np.array(emb_list, dtype=np.float32)
                embeddings_dict[filename] = emb
                filename_list.append(filename)
            else:
                missing_files.append(img_file)

        # 第二遍：多线程并行提取缺失的 embedding
        if missing_files:
            print(f"\n需要提取 {len(missing_files)} 个 embedding（多线程并行）...")
            _some = min(len(missing_files), 12)
            with concurrent.futures.ThreadPoolExecutor(max_workers=_some) as _ex:
                _futs = {_ex.submit(self.extract_embedding, p): p for p in missing_files}
                for _f in tqdm.tqdm(
                    concurrent.futures.as_completed(_futs),
                    total=len(_futs), desc="提取 embedding", unit="img", ascii=True
                ):
                    _p = _futs[_f]
                    try:
                        emb = _f.result()
                        if emb is not None:
                            embeddings_dict[_p.name] = emb
                            filename_list.append(_p.name)
                            saved_count += 1
                        else:
                            failed_files.append(_p.name)
                    except Exception:
                        failed_files.append(_p.name)
            print(f"\n✓ 多线程提取完成：{saved_count} 成功，{len(failed_files)} 失败")
        
        # 保存新提取的 embedding 到 HDF5（主存储）和 JSON（互补备份）
        if saved_count > 0:
            # 写入 HDF5
            try:
                import h5py
                if self._h5_accessor is not None:
                    self._h5_accessor.close()
                with h5py.File(self.metadata_file, 'a') as f:
                    for fn in filename_list:
                        if fn not in embeddings_dict:
                            continue
                        emb_val = embeddings_dict[fn]
                        safe_name = fn.replace('/', '_SLASH_').replace('\\', '_BSLASH_')
                        grp = f.create_group(safe_name) if safe_name not in f else f[safe_name]
                        grp.attrs['__original_filename__'] = fn
                        if 'embedding' in grp:
                            del grp['embedding']
                        grp.create_dataset('embedding', data=np.array(emb_val, dtype=np.float32),
                                           compression='gzip', compression_opts=4)
                # 重新初始化 HDF5 访问器
                from FacesetProcessor.H5StreamingAccessor import H5StreamingAccessor
                self._h5_accessor = H5StreamingAccessor(self.metadata_file)
                print(f"\n✓ 已保存 {saved_count} 个 embedding 到 HDF5")
            except Exception as e:
                print(f"[!] 保存 embedding 到 HDF5 失败: {e}")
        
        # 统计缓存使用情况（直接用已处理的结果，避免再次遍历 HDF5）
        cached_count = len(embeddings_dict)
        if cached_count > 0 and saved_count < cached_count:
            print(S('FILTER_USING_CACHED_EMBEDDINGS'))

        print(S('FILTER_EMBEDDINGS_EXTRACTED', len(embeddings_dict), len(failed_files)))
        
        if not embeddings_dict:
            print(S('FILTER_NO_EMBEDDINGS'))
            return {}
        
        # ========== 第二阶段：DBSCAN 聚类 ==========
        print(S('FILTER_PHASE2_GROUPING'))
        print(S('FILTER_COMPUTING_SIMILARITY'))
        
        # 构建 embedding 矩阵
        embedding_matrix = np.array([embeddings_dict[f] for f in filename_list])
        print(f"  Embedding 矩阵形状: {embedding_matrix.shape}")
        print(f"  预计内存需求: {embedding_matrix.nbytes / (1024**2):.2f} MB")
        
        # 检查是否可以使用全量相似度矩阵（限制在 10GB 以内）
        n_samples = len(filename_list)
        estimated_memory_gb = (n_samples * n_samples * 4) / (1024**3)  # float32 = 4 bytes
        
        if estimated_memory_gb > 10:
            print(f"\n⚠ 全量相似度矩阵需要 {estimated_memory_gb:.2f} GB 内存")
            from sklearn.cluster import DBSCAN as _DBSCAN
            try:
                import faiss
                _HAVE_FAISS = True
            except ImportError:
                _HAVE_FAISS = False

            if _HAVE_FAISS:
                print(f"  → 使用 FAISS range_search + Union-Find\n")
            else:
                print(f"  → faiss 未安装，回退 sklearn ball_tree\n")

            # 归一化 embedding 到单位向量
            norms = np.linalg.norm(embedding_matrix, axis=1, keepdims=True)
            norms[norms == 0] = 1
            emb_norm = (embedding_matrix / norms).astype(np.float32)

            if _HAVE_FAISS:
                k = max(min_samples + 1, 50)
                index = faiss.IndexFlatIP(emb_norm.shape[1])
                index.add(emb_norm)
                # search 返回 top-k 近邻（含自身），dists 是内积
                k_actual = min(k, n_samples)
                faiss_dists, faiss_idxs = index.search(emb_norm, k_actual)
                threshold = 1.0 - eps

                parent = np.arange(n_samples, dtype=np.int32)
                def find(x):
                    while parent[x] != x:
                        parent[x] = parent[parent[x]]
                        x = parent[x]
                    return x
                def union(x, y):
                    px, py = find(x), find(y)
                    if px != py:
                        parent[px] = py

                core = np.zeros(n_samples, dtype=bool)
                for i in range(n_samples):
                    nbrs = int(np.sum(faiss_dists[i] > threshold))
                    nbrs -= 1  # 去掉自身（idx 0 总是自身）
                    core[i] = nbrs >= min_samples

                _eps_adjusted = False
                n_components = 0
                while n_components <= 1:
                    for i in range(n_samples):
                        if not core[i]:
                            continue
                        for jj in range(1, k_actual):
                            j = faiss_idxs[i, jj]
                            if core[j] and faiss_dists[i, jj] > threshold and j > i:
                                union(i, j)

                    labels = np.full(n_samples, -1, dtype=np.int32)
                    label_map = {}
                    current_label = 0
                    for i in range(n_samples):
                        if core[i]:
                            root = find(i)
                            if root not in label_map:
                                label_map[root] = current_label
                                current_label += 1
                            labels[i] = label_map[root]

                    for i in range(n_samples):
                        if not core[i]:
                            for jj in range(1, k_actual):
                                if faiss_dists[i, jj] > threshold and labels[faiss_idxs[i, jj]] != -1:
                                    labels[i] = labels[faiss_idxs[i, jj]]
                                    break

                    _labels_set = set(labels) - {-1}
                    n_components = len(_labels_set)
                    print(f"  {'✓' if not _eps_adjusted else '→'} DBSCAN 聚类：{n_components} 个簇")
                    if n_components <= 1 and eps > 0.05 and not _eps_adjusted:
                        eps *= 0.7
                        threshold = 1.0 - eps
                        _eps_adjusted = True
                        continue
                    break
            else:
                # FAISS 不可用时的回退
                eps_euc = (2.0 * eps) ** 0.5
                _tries = 0; _current_eps = eps_euc
                while True:
                    db = _DBSCAN(eps=_current_eps, min_samples=min_samples,
                                 metric='euclidean', algorithm='ball_tree', n_jobs=-1)
                    labels = db.fit_predict(emb_norm)
                    _labels_set = set(labels) - {-1}
                    n_components = len(_labels_set)
                    print(f"  {'✓' if _tries==0 else '→'} DBSCAN 聚类：{n_components} 个簇"
                          f"{'' if _tries==0 else f'（eps={_current_eps}）'}")
                    if n_components <= 1 and _current_eps > 0.05 and _tries < 2:
                        _current_eps *= 0.7
                        _tries += 1
                        continue
                    break
        else:
            # 原始的全量矩阵方法（小数据集）
            print(f"  使用全量相似度矩阵（预计 {estimated_memory_gb:.2f} GB）...")
            
            # 计算余弦相似度矩阵
            similarity_matrix = cosine_similarity(embedding_matrix)
            
            # 转换为距离矩阵（DBSCAN 需要距离）
            distance_matrix = 1.0 - similarity_matrix
            distance_matrix = np.clip(distance_matrix, 0.0, None)
            
            print(S('FILTER_RUNNING_DBSCLAN'))
            
            # 执行 DBSCAN 聚类
            db = DBSCAN(
                eps=eps,
                min_samples=min_samples,
                metric='precomputed'
            )
            
            labels = db.fit_predict(distance_matrix)
        
        # 统计聚类结果
        unique_labels = set(labels)
        n_clusters = len(unique_labels) - (1 if -1 in labels else 0)  # 排除噪声点
        n_noise = list(labels).count(-1)
        
        print(S('FILTER_CLUSTERING_RESULTS'))
        print(S('FILTER_TOTAL_CLUSTERS', n_clusters))
        print(S('FILTER_NOISE_POINTS', n_noise))
        
        # 构建分组字典
        groups = {}
        noise_files = []
        
        for i, label in enumerate(labels):
            filename = filename_list[i]
            if label == -1:
                # 噪声点（无法归类）
                noise_files.append(filename)
            else:
                if label not in groups:
                    groups[label] = []
                groups[label].append(filename)
        
        # 重新编号组（按成员数量降序排列，最多的组获得最小编号）
        print(f"  正在按组成员数量排序...")
        sorted_groups = sorted(groups.items(), key=lambda x: len(x[1]), reverse=True)
        
        renumbered_groups = {}
        for new_id, (old_id, filenames) in enumerate(sorted_groups):
            renumbered_groups[new_id] = filenames
        
        groups = renumbered_groups
        if groups:
            print(f"  ✓ 排序完成：最大的组有 {len(groups[0])} 张图片")
        else:
            print(f"  ⚠ 排序完成：所有面孔均为独立个体（无聚类组）")
        
        # ========== 第三阶段：最终确认保存（确保所有 embedding 已保存）==========
        
        # ========== 第四阶段：批量移动 ==========
        print(S('FILTER_MOVING_FILES', len(groups)))
        
        stats = {
            'total_groups': len(groups),
            'total_files': sum(len(g) for g in groups.values()),
            'failed_files': len(failed_files),
            'noise_files': len(noise_files),
            'groups': {}
        }
        
        # 构建移动列表
        move_list = []  # [(src_path, dst_path), ...]
        
        for group_id, filenames in groups.items():
            # 创建组文件夹
            group_dir = self.faceset_path / f"faceid_group_{group_id:03d}"
            group_dir.mkdir(exist_ok=True)
            
            # 添加到移动列表
            for filename in filenames:
                src = self.faceset_path / filename
                dst = group_dir / filename
                move_list.append((src, dst))
            
            stats['groups'][f"group_{group_id:03d}"] = len(filenames)
        
        # 处理噪声点（无法归类的图片）
        if noise_files:
            noise_dir = self.faceset_path / "faceid_noise"
            noise_dir.mkdir(exist_ok=True)
            
            for filename in noise_files:
                src = self.faceset_path / filename
                dst = noise_dir / filename
                move_list.append((src, dst))
            
            print(S('FILTER_MOVING_NOISE', len(noise_files)))
        
        # 批量移动：写 batch 文件 → 一次性提交给 cmd.exe
        # 避免每文件一次 os.replace 的 Python→C 切换开销
        if move_list:
            batch_move_files(move_list, self.faceset_path)
            stats['total_files'] = len(move_list)
            stats['errors'] = 0
        
        # ========== 第四阶段：可选的合并操作 ==========
        if merge_back:
            stats = self._merge_groups_to_aligned(stats, output_dir)
        
        print(S('FILTER_FACEID_COMPLETE'))
        print(f"  {S('FILTER_STAT_GROUPS')}: {stats['total_groups']}")
        print(f"  {S('FILTER_STAT_TOTAL')}: {stats['total_files']}")
        print(f"  {S('FILTER_STAT_NOISE')}: {stats['noise_files']}")
        print(f"  {S('FILTER_STAT_FAILED')}: {stats['failed_files']}")
        
        # 显示前10个组的详细信息
        for i, (group_name, count) in enumerate(list(stats['groups'].items())[:10]):
            print(f"    {group_name}: {count} 张")
        
        if len(stats['groups']) > 10:
            print(f"    ... 还有 {len(stats['groups']) - 10} 个组")
        
        return stats
    
    def _merge_groups_to_aligned(self, stats: Dict, output_dir: str = None) -> Dict:
        """
        将所有分组中的人脸图片合并回一个文件夹
        处理文件名冲突，保留最新的文件
        
        Args:
            stats: 统计字典
            output_dir: 输出目录名称（默认: aligned）
            
        Returns:
            更新后的统计字典
        """
        output_dir_name = output_dir or "aligned"
        output_path = self.faceset_path / output_dir_name
        output_path.mkdir(exist_ok=True)
        
        print(f"\nMerging all groups to '{output_dir_name}/'...")
        
        # 收集所有分组和噪声文件夹中的文件
        all_source_dirs = []
        
        # 添加所有分组目录
        for group_id in range(stats['total_groups']):
            group_dir = self.faceset_path / f"faceid_group_{group_id:03d}"
            if group_dir.exists():
                all_source_dirs.append(group_dir)
        
        # 添加噪声目录
        noise_dir = self.faceset_path / "faceid_noise"
        if noise_dir.exists():
            all_source_dirs.append(noise_dir)
        
        if not all_source_dirs:
            print("No groups to merge")
            return stats
        
        # 统计信息
        merged_count = 0
        conflict_count = 0
        skipped_count = 0
        error_count = 0
        
        # 遍历所有源目录
        total_files = sum(
            len(list(d.glob("*.jpg"))) + len(list(d.glob("*.jpeg"))) +
            len(list(d.glob("*.png"))) + len(list(d.glob("*.bmp")))
            for d in all_source_dirs
        )
        pbar = tqdm.tqdm(total=total_files, desc="Merging", unit="file", ascii=True)
        
        for source_dir in all_source_dirs:
            image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
            image_files = [f for f in source_dir.iterdir() if f.suffix.lower() in image_extensions and f.is_file()]
            
            for src_file in image_files:
                try:
                    dst_file = output_path / src_file.name
                    
                    # 检查是否存在冲突
                    if dst_file.exists():
                        # 比较元数据，保留较新的
                        if self._should_replace_file(src_file, dst_file):
                            # 目标文件存在但源文件更新，替换
                            os.replace(str(src_file), str(dst_file))
                            conflict_count += 1
                        else:
                            # 目标文件更新或相同，跳过
                            skipped_count += 1
                    else:
                        # 没有冲突，直接移动
                        os.replace(str(src_file), str(dst_file))
                        merged_count += 1
                    
                    pbar.update(1)
                    
                except Exception as e:
                    print(f"Error merging {src_file.name}: {e}")
                    error_count += 1
                    pbar.update(1)
        
        pbar.close()
        
        print(f"\nMerge complete:")
        print(f"  Merged: {merged_count}")
        print(f"  Conflicts (replaced): {conflict_count}")
        print(f"  Skipped (older): {skipped_count}")
        print(f"  Errors: {error_count}")
        
        # 更新统计
        stats['merged_files'] = merged_count
        stats['conflict_replaced'] = conflict_count
        stats['conflict_skipped'] = skipped_count
        stats['merge_errors'] = error_count
        
        return stats
    
    def _should_replace_file(self, src_file: Path, dst_file: Path) -> bool:
        """
        判断是否应该用源文件替换目标文件
        仅使用文件修改时间，不加载元数据（提高性能）
        
        Args:
            src_file: 源文件路径
            dst_file: 目标文件路径
            
        Returns:
            True 如果应该替换，False 否则
        """
        try:
            # 直接使用文件修改时间，避免加载元数据
            src_mtime = src_file.stat().st_mtime
            dst_mtime = dst_file.stat().st_mtime
            
            return src_mtime > dst_mtime
            
        except Exception as e:
            # 出错时默认不替换，保护已有数据
            print(f"Warning: Could not compare files {src_file.name}: {e}")
            return False
    
    def merge_subfolders_to_aligned(self, output_dir: str = "aligned") -> Dict:
        """
        将当前目录下所有一级子文件夹中的图片合并到指定文件夹
        安全限制：只遍历直接子文件夹，不递归到更深层
        
        Args:
            output_dir: 输出目录名称（默认: aligned）
                       注意：如果 faceset_path 本身就是 aligned 目录，
                       则直接合并到 faceset_path，不再创建子目录
            
        Returns:
            统计结果
        """
        # 判断 faceset_path 是否已经是目标目录
        # 如果路径以 'aligned' 结尾，说明用户已经指定了 aligned 目录
        path_name = self.faceset_path.name.lower()
        if path_name == output_dir.lower():
            # 用户输入的已经是 aligned 目录，直接使用该目录作为输出
            output_path = self.faceset_path
            print(f"\nMerging images from subfolders to current directory...")
        else:
            # 用户输入的是父目录，需要创建子目录
            output_path = self.faceset_path / output_dir
            output_path.mkdir(exist_ok=True)
            print(f"\nMerging images from subfolders to '{output_dir}/'...")
        
        print("Safety: Only scanning immediate subdirectories (not recursive)")
        
        # 获取所有一级子文件夹（排除输出目录本身）
        subdirs = [
            d for d in self.faceset_path.iterdir() 
            if d.is_dir() and d != output_path
        ]
        
        if not subdirs:
            print("No subdirectories found")
            return {'merged': 0, 'conflicts': 0, 'skipped': 0, 'errors': 0}
        
        print(f"Found {len(subdirs)} subdirectories")
        
        # 记录原本就是空的目录（合并后不删除）
        originally_empty_dirs = set()
        for subdir in subdirs:
            image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
            has_images = any(
                f.is_file() and f.suffix.lower() in image_extensions
                for f in subdir.iterdir()
            )
            if not has_images:
                originally_empty_dirs.add(subdir)
        
        if originally_empty_dirs:
            print(f"Found {len(originally_empty_dirs)} originally empty directories (will be preserved)")
        
        # 统计信息
        merged_count = 0
        conflict_count = 0
        skipped_count = 0
        error_count = 0
        
        # 收集所有要移动的文件
        all_files = []
        for subdir in subdirs:
            image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
            # 只获取直接子文件夹中的文件，不递归
            image_files = [
                f for f in subdir.iterdir() 
                if f.is_file() and f.suffix.lower() in image_extensions
            ]
            all_files.extend(image_files)
        
        print(f"Total images to process: {len(all_files)}")
        
        if not all_files:
            print("No images found in subdirectories")
            return {'merged': 0, 'conflicts': 0, 'skipped': 0, 'errors': 0}
        
        # batch 文件 + 超时保护（10 分钟内必须完成全部移动）
        import subprocess as _sp
        _bat = output_path / '_merge_temp.bat'
        with open(_bat, 'w', encoding='utf-8') as _f:
            for _src in all_files:
                _f.write(f'move /y "{_src}" "{output_path / _src.name}"\n')
        print(f"  批处理文件已生成，正在执行（{len(all_files)} 个文件）...")
        try:
            _sp.run([str(_bat)], stdout=_sp.DEVNULL, stderr=_sp.DEVNULL, timeout=600)
        except _sp.TimeoutExpired:
            print(f"  ⚠ 部分文件移动超时（10 分钟），剩余文件保留在原目录")
        _bat.unlink()
        merged_count = sum(1 for _ in output_path.iterdir() if _.is_file())
        conflict_count = error_count = 0

        print(f"\nMerge complete:")
        print(f"  Merged: {merged_count}")
        print(f"  Conflicts (replaced): {conflict_count}")
        print(f"  Errors: {error_count}")
        
        # 清理空目录：删除那些因为文件被移走而变空的目录
        print("\nCleaning up empty directories...")
        removed_dirs = []
        kept_empty_dirs = []
        
        for subdir in subdirs:
            # 跳过原本就是空的目录
            if subdir in originally_empty_dirs:
                kept_empty_dirs.append(subdir.name)
                continue
            
            # 检查目录是否为空（或只包含非图片文件）
            image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
            remaining_files = [
                f for f in subdir.iterdir()
                if f.is_file() and f.suffix.lower() in image_extensions
            ]
            
            if not remaining_files:
                # 目录已空，删除
                try:
                    shutil.rmtree(str(subdir))
                    removed_dirs.append(subdir.name)
                except Exception as e:
                    print(f"Warning: Could not remove directory {subdir.name}: {e}")
        
        if removed_dirs:
            print(f"Removed {len(removed_dirs)} empty directories: {', '.join(removed_dirs)}")
        if kept_empty_dirs:
            print(f"Preserved {len(kept_empty_dirs)} originally empty directories: {', '.join(kept_empty_dirs)}")
        if not removed_dirs and not kept_empty_dirs:
            print("No empty directories to clean up")
        
        return {
            'merged': merged_count,
            'conflicts': conflict_count,
            'skipped': skipped_count,
            'errors': error_count,
            'removed_empty_dirs': len(removed_dirs),
            'preserved_empty_dirs': len(kept_empty_dirs)
        }


class PositionFilter(FacesetBaseProcessor):
    """位置过滤器 - 基于人脸坐标聚类"""
    
    def __init__(self, faceset_path: Path, metadata_file: Optional[Path] = None):
        """
        初始化位置过滤器
        
        Args:
            faceset_path: 人脸数据集路径
            metadata_file: 元数据文件路径（可选）
        """
        super().__init__(faceset_path, metadata_file)
    
    def filter_by_position(self, eps: float = 50.0, min_samples: int = 2) -> Dict:
        """
        根据人脸在原始图像中的位置进行聚类分组
        适用于固定摄像头、视频会议等场景
        
        Args:
            eps: DBSCAN 邻域半径（像素），默认 50.0
                 表示位置距离小于 50 像素的人脸视为同一位置
            min_samples: DBSCAN 最小样本数，默认 2
            
        Returns:
            分组统计结果
        """
        print(S('FILTER_START_POSITION_FILTER', eps))
        
        # 扫描所有图片文件
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
        image_files = sorted([
            f for f in self.faceset_path.iterdir()
            if f.suffix.lower() in image_extensions and f.is_file()
        ])
        
        if not image_files:
            print(S('FILTER_NO_IMAGES_FOUND'))
            return {}
        
        print(S('FILTER_FOUND_IMAGES', len(image_files)))
        
        # ========== 第一阶段：提取位置信息 ==========
        print(S('FILTER_PHASE1_EXTRACTING_POSITIONS'))
        
        positions_dict = {}  # {filename: (center_x, center_y)}
        filename_list = []
        failed_files = []
        
        pbar = tqdm.tqdm(total=len(image_files), desc=S('FILTER_PROGRESS_EXTRACTING'), unit="img", ascii=True)
        
        for img_file in image_files:
            filename = img_file.name
            try:
                source_rect = self.get_field(filename, 'source_rect')
                
                if source_rect is None:
                    failed_files.append(filename)
                    pbar.update(1)
                    continue
                
                # 计算人脸中心点
                l, t, r, b = source_rect[:4]
                center_x = (l + r) / 2.0
                center_y = (t + b) / 2.0
                
                positions_dict[filename] = (center_x, center_y)
                filename_list.append(filename)
                
            except Exception as e:
                print(S('FILTER_PROCESS_ERROR', img_file.name, e))
                failed_files.append(filename)
            finally:
                pbar.update(1)
        
        pbar.close()
        
        print(S('FILTER_POSITIONS_EXTRACTED', len(positions_dict), len(failed_files)))
        
        if not positions_dict:
            print(S('FILTER_NO_POSITIONS'))
            return {}
        
        # ========== 第二阶段：DBSCAN 聚类 ==========
        print(S('FILTER_PHASE2_CLUSTERING_POSITIONS'))
        
        # 构建位置矩阵
        position_matrix = np.array([positions_dict[f] for f in filename_list])
        
        print(S('FILTER_RUNNING_DBSCLAN_POSITION', eps, min_samples))
        
        # 执行 DBSCAN 聚类
        db = DBSCAN(
            eps=eps,
            min_samples=min_samples,
            metric='euclidean'
        )
        
        labels = db.fit_predict(position_matrix)
        
        # 统计聚类结果
        unique_labels = set(labels)
        n_clusters = len(unique_labels) - (1 if -1 in labels else 0)
        n_noise = list(labels).count(-1)
        
        print(S('FILTER_CLUSTERING_RESULTS'))
        print(S('FILTER_TOTAL_CLUSTERS', n_clusters))
        print(S('FILTER_NOISE_POINTS', n_noise))
        
        # 构建分组字典
        groups = {}
        noise_files = []
        
        for i, label in enumerate(labels):
            filename = filename_list[i]
            if label == -1:
                noise_files.append(filename)
            else:
                if label not in groups:
                    groups[label] = []
                groups[label].append(filename)
        
        # 重新编号组
        renumbered_groups = {}
        for new_id, (old_id, filenames) in enumerate(sorted(groups.items())):
            renumbered_groups[new_id] = filenames
        
        groups = renumbered_groups
        
        # ========== 第三阶段：批量移动 ==========
        print(S('FILTER_MOVING_FILES', len(groups)))
        
        stats = {
            'total_groups': len(groups),
            'total_files': sum(len(g) for g in groups.values()),
            'failed_files': len(failed_files),
            'noise_files': len(noise_files),
            'groups': {}
        }
        
        # 构建移动列表
        move_list = []
        
        for group_id, filenames in groups.items():
            # 创建组文件夹
            group_dir = self.faceset_path / f"position_group_{group_id:03d}"
            group_dir.mkdir(exist_ok=True)
            
            # 添加到移动列表
            for filename in filenames:
                src = self.faceset_path / filename
                dst = group_dir / filename
                move_list.append((src, dst))
            
            stats['groups'][f"group_{group_id:03d}"] = len(filenames)
        
        # 处理噪声点
        if noise_files:
            noise_dir = self.faceset_path / "position_noise"
            noise_dir.mkdir(exist_ok=True)
            
            for filename in noise_files:
                src = self.faceset_path / filename
                dst = noise_dir / filename
                move_list.append((src, dst))
            
            print(S('FILTER_MOVING_NOISE', len(noise_files)))
        
        # 批量移动：写 batch 文件 → 一次性提交给 cmd.exe
        # 避免每文件一次 os.replace 的 Python→C 切换开销
        if move_list:
            batch_move_files(move_list, self.faceset_path)
            stats['total_files'] = len(move_list)
            stats['errors'] = 0
        
        print(S('FILTER_POSITION_COMPLETE'))
        print(f"  {S('FILTER_STAT_GROUPS')}: {stats['total_groups']}")
        print(f"  {S('FILTER_STAT_TOTAL')}: {stats['total_files']}")
        print(f"  {S('FILTER_STAT_NOISE')}: {stats['noise_files']}")
        print(f"  {S('FILTER_STAT_FAILED')}: {stats['failed_files']}")
        
        # 显示前10个组的详细信息
        for i, (group_name, count) in enumerate(list(stats['groups'].items())[:10]):
            print(f"    {group_name}: {count} 张")
        
        if len(stats['groups']) > 10:
            print(f"    ... 还有 {len(stats['groups']) - 10} 个组")
        
        return stats


def _calculate_phash_worker(args):
    """
    多进程工作函数：计算单张图片的 phash
    
    Args:
        args: (image_path_str, hash_method)
        
    Returns:
        (filename, hash_array_or_None, error_msg_or_None)
    """
    image_path_str, hash_method = args
    
    try:
        # 读取图像
        img = cv2.imread(image_path_str)
        if img is None:
            return (Path(image_path_str).name, None, "Failed to read image")
        
        # 根据方法计算哈希
        if hash_method == 'phash':
            hash_value = RepeatedFilter.phash(img)
        elif hash_method == 'ahash':
            hash_value = RepeatedFilter.ahash(img)
        elif hash_method == 'dhash':
            hash_value = RepeatedFilter.dhash(img)
        else:
            hash_value = RepeatedFilter.phash(img)
        
        return (Path(image_path_str).name, hash_value, None)
    except Exception as e:
        return (Path(image_path_str).name, None, str(e))


class RepeatedFilter(FacesetBaseProcessor):
    """重复图片过滤器 - 基于感知哈希去重"""

    def __init__(self, faceset_path: Path):
        super().__init__(faceset_path)

    @staticmethod
    def phash(img: np.ndarray, hash_size: int = 8) -> Optional[np.ndarray]:
        try:
            if img is None:
                return None
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            resized = cv2.resize(gray, (hash_size * 4, hash_size * 4), interpolation=cv2.INTER_AREA)
            dct = cv2.dct(np.float32(resized))
            dct_low_freq = dct[:hash_size, :hash_size]
            avg = np.mean(dct_low_freq)
            hash_value = (dct_low_freq > avg).astype(int)
            return hash_value.flatten()
        except Exception as e:
            return None

    @staticmethod
    def ahash(img: np.ndarray, hash_size: int = 8) -> Optional[np.ndarray]:
        try:
            if img is None:
                return None
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            resized = cv2.resize(gray, (hash_size, hash_size), interpolation=cv2.INTER_AREA)
            avg = np.mean(resized)
            hash_value = (resized > avg).astype(int)
            return hash_value.flatten()
        except Exception as e:
            return None

    @staticmethod
    def dhash(img: np.ndarray, hash_size: int = 8) -> Optional[np.ndarray]:
        try:
            if img is None:
                return None
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            resized = cv2.resize(gray, (hash_size + 1, hash_size), interpolation=cv2.INTER_AREA)
            hash_value = []
            for row in range(resized.shape[0]):
                for col in range(resized.shape[1] - 1):
                    if resized[row, col] > resized[row, col + 1]:
                        hash_value.append(1)
                    else:
                        hash_value.append(0)
            return np.array(hash_value)
        except Exception as e:
            return None

    @staticmethod
    def hamming_distance(hash1: np.ndarray, hash2: np.ndarray) -> int:
        return int(np.sum(hash1 != hash2))

    @staticmethod
    def similarity(hash1, hash2) -> float:
        """
        计算两个哈希值的相似度
        
        Args:
            hash1: 可以是 np.ndarray, list, 或其他可转换为数组的对象
            hash2: 同上
        
        Returns:
            相似度 (0-1)
        """
        # 确保转换为 numpy 数组
        try:
            if not isinstance(hash1, np.ndarray):
                hash1 = np.array(hash1)
            if not isinstance(hash2, np.ndarray):
                hash2 = np.array(hash2)
            
            # 检查是否为有效数组
            if hash1.ndim == 0 or hash2.ndim == 0:
                # 标量值，无法计算相似度
                return 0.0
            
            total_bits = len(hash1)
            if total_bits == 0:
                return 0.0
            
            distance = RepeatedFilter.hamming_distance(hash1, hash2)
            return 1.0 - (distance / total_bits)
        except Exception as e:
            print(f"Warning: similarity calculation error: {e}")
            print(f"  hash1 type: {type(hash1)}, shape: {getattr(hash1, 'shape', 'N/A')}")
            print(f"  hash2 type: {type(hash2)}, shape: {getattr(hash2, 'shape', 'N/A')}")
            return 0.0

    def filter_repeated(self, threshold: float = 0.98, hash_method: str = 'phash', workers: int = None) -> Dict:
        """
        使用顺序扫描策略找出重复图片
        算法逻辑：
        1. 第一张图片作为起点A
        2. 向后扫描，找到第一个不与A相似的图片B
        3. A和A后面所有相似的都保留，其他的标记为重复
        4. B成为新起点，继续向后扫描
        5. 复杂度 O(n)，只需记录当前起点
        """
        import multiprocessing
        from multiprocessing import Pool

        print(S('FILTER_START_REPEATED_FILTER', threshold, hash_method))
        hash_functions = {'phash': self.phash, 'ahash': self.ahash, 'dhash': self.dhash}
        hash_func = hash_functions.get(hash_method, self.phash)

        if hash_method == 'phash':
            print(S('FILTER_HASH_METHOD_PHASH'))
        elif hash_method == 'ahash':
            print(S('FILTER_HASH_METHOD_AHASH'))
        elif hash_method == 'dhash':
            print(S('FILTER_HASH_METHOD_DHASH'))

        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
        image_files = sorted(
            [f for f in self.faceset_path.iterdir() if f.suffix.lower() in image_extensions and f.is_file()])

        if not image_files:
            print(S('FILTER_NO_IMAGES_FOUND'))
            return {}

        total_images = len(image_files)
        print(S('FILTER_FOUND_IMAGES', total_images))

        repeated_dir = self.faceset_path / "repeated"
        repeated_dir.mkdir(exist_ok=True)

        if workers is None:
            workers = multiprocessing.cpu_count()

        # ========== 第一阶段：计算哈希并保存到元数据 ==========
        print(S('FILTER_PHASE1_CALCULATING_HASHES'))
        
        # 准备任务列表
        tasks = []
        valid_files = []
        
        for img_file in image_files:
            filename = img_file.name
            # 检查元数据中是否已有 phash（流式读取，避免加载整个 HDF5）
            if self.has_fields(filename, {'phash'}).get('phash'):
                # 已有 phash，跳过计算
                continue
            else:
                # 需要计算
                tasks.append((str(img_file), hash_method))
                valid_files.append(img_file)
        
        # 多进程计算 phash（只计算缺失的）
        phash_results = {}  # {filename: hash_array}
        new_phash_count = 0
        
        if tasks:
            with Pool(processes=workers) as pool:
                pbar = tqdm.tqdm(total=len(tasks), desc=S('FILTER_PROGRESS_HASHING'), unit="img", ascii=True)
                
                # 使用 imap_unordered 获取结果
                for filename, hash_value, error in pool.imap_unordered(_calculate_phash_worker, tasks):
                    if error:
                        print(S('FILTER_HASH_ERROR', filename, error))
                    else:
                        if hash_value is not None:
                            phash_results[filename] = hash_value
                            
                            # 检查是否需要保存到元数据
                            if not self.has_fields(filename, {'phash'}).get('phash'):
                                new_phash_count += 1
                    
                    pbar.update(1)
                
                pbar.close()
        
        # 保存新的 phash 到元数据
        if new_phash_count > 0:
            print(S('FILTER_SAVING_EMBEDDINGS').replace('embedding', 'phash'))
            try:
                import h5py
                with h5py.File(self.metadata_file, 'w') as f:
                    for filename, meta in self.metadata.items():
                        safe_name = filename.replace('/', '_SLASH_').replace('\\', '_BSLASH_')
                        grp = f.create_group(safe_name)
                        # 存储原始文件名
                        grp.attrs['__original_filename__'] = filename
                        for key, value in meta.items():
                            if isinstance(value, (list, np.ndarray)):
                                grp.create_dataset(key, data=np.array(value), compression='gzip', compression_opts=4)
                            elif isinstance(value, (int, float, str)):
                                grp.attrs[key] = value
                            else:
                                grp.attrs[key] = str(value)
                print(S('FILTER_EMBEDDINGS_SAVED', new_phash_count).replace('embedding', 'phash'))
            except Exception as e:
                print(S('FILTER_SAVE_METADATA_WARNING', e))
        else:
            print(S('FILTER_EMBEDDINGS_ALREADY_SAVED').replace('embedding', 'phash'))
        
        # 将所有图片（包括已有phash的）加入 valid_files
        valid_files = image_files
        
        # ========== 第二阶段：顺序扫描找出重复图片 ==========
        print(S('FILTER_PHASE2_SEQUENTIAL_SCAN'))
        
        # 收集所有图片的哈希值（按顺序）
        file_hash_list = []  # [(filename, hash_array, file_size), ...]
        
        pbar = tqdm.tqdm(total=len(valid_files), desc=S('FILTER_PROGRESS_LOADING_HASHES'), unit="img", ascii=True)
        
        for img_file in valid_files:
            filename = img_file.name
            
            # 从 phash_results 或元数据中获取 phash
            if filename in phash_results:
                hash_value = phash_results[filename]
            else:
                # 从元数据读取（流式，不加载整个 HDF5）
                phash_data = self.get_field(filename, 'phash')
                if phash_data is not None:
                    # 确保转换为 numpy 数组
                    if isinstance(phash_data, str):
                        # 字符串格式（十六进制），转换为二进制数组
                        try:
                            # 将十六进制字符串转换为整数，再转换为二进制数组
                            phash_int = int(phash_data, 16)
                            # 转换为 64 位二进制数组（phash 通常是 64 位）
                            hash_value = np.array([(phash_int >> i) & 1 for i in range(63, -1, -1)], dtype=np.int32)
                        except Exception as e:
                            print(f"Warning: Failed to convert phash string for {filename}: {e}")
                            pbar.update(1)
                            continue
                    elif isinstance(phash_data, list):
                        hash_value = np.array(phash_data, dtype=np.int32)
                    elif isinstance(phash_data, np.ndarray):
                        hash_value = phash_data
                    else:
                        # 其他类型（可能是标量），跳过
                        print(f"Warning: Invalid phash format for {filename}: {type(phash_data)}")
                        pbar.update(1)
                        continue
                else:
                    # 没有哈希值，跳过
                    pbar.update(1)
                    continue
            
            file_size = img_file.stat().st_size
            file_hash_list.append((filename, hash_value, file_size))
            pbar.update(1)
        
        pbar.close()
        
        if not file_hash_list:
            print("No valid images with hash found")
            return {}
        
        # 顺序扫描：O(n) 复杂度，直接移动重复文件
        print(S('FILTER_PHASE3_MARKING_DUPLICATES'))
        
        anchors = []  # 起点列表（保留的图片）
        moved_count = 0
        skipped_count = 0
        
        # 第一张图片作为第一个起点
        if file_hash_list:
            current_anchor_idx = 0
            anchors.append(file_hash_list[0])
        else:
            return {}
        
        # 从第二张图片开始扫描
        pbar = tqdm.tqdm(total=len(file_hash_list) - 1, desc=S('FILTER_PROGRESS_SCANNING'), unit="img", ascii=True)
        
        for i in range(1, len(file_hash_list)):
            current_file, current_hash, current_size = file_hash_list[i]
            
            # 只与当前起点对比
            anchor_file, anchor_hash, anchor_size = file_hash_list[current_anchor_idx]
            sim = self.similarity(current_hash, anchor_hash)
            
            if sim >= threshold:
                # 与当前起点相似，直接移动到 repeated 目录
                src = self.faceset_path / current_file
                dst = repeated_dir / current_file
                
                try:
                    if src.exists():
                        os.replace(str(src), str(dst))
                        moved_count += 1
                    else:
                        skipped_count += 1
                except Exception as e:
                    print(S('FILTER_MOVE_ERROR', current_file, e))
                    skipped_count += 1
            else:
                # 不相似，成为新起点
                current_anchor_idx = i
                anchors.append((current_file, current_hash, current_size))
            
            pbar.update(1)
        
        pbar.close()
        
        duplicate_count = moved_count
        anchor_count = len(anchors)
        print(S('FILTER_FOUND_DUPLICATES', duplicate_count))
        print(f"  Anchors (kept): {anchor_count}")
        
        if skipped_count > 0:
            print(f"  跳过 {skipped_count} 个不存在的文件")
        
        remaining_images = total_images - moved_count
        reduction_percentage = (moved_count / total_images) * 100 if total_images > 0 else 0
        print(S('FILTER_REPEATED_COMPLETE'))
        print(f"  {S('FILTER_STAT_TOTAL')}: {total_images}")
        print(f"  {S('FILTER_STAT_MOVED')}: {moved_count}")
        print(f"  {S('FILTER_STAT_REMAINING')}: {remaining_images}")
        print(f"  {S('FILTER_STAT_REDUCTION')}: {reduction_percentage:.2f}%")
        print(f"  Anchor groups: {anchor_count}")
        
        return {
            'total': total_images,
            'moved': moved_count,
            'remaining': remaining_images,
            'reduction_percentage': reduction_percentage,
            'duplicate_count': duplicate_count,
            'anchor_count': anchor_count
        }

    @staticmethod
    def _compare_with_group(args):
        hash_value, group, threshold = args
        representative_hash = group[0]['hash']
        sim = RepeatedFilter.similarity(hash_value, representative_hash)
        return sim >= threshold, sim

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description=S('FILTER_DESCRIPTION'),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  python Filter.py --input ".\\workspace\\faces" --mode blur
  python Filter.py --input ".\\workspace\\faces" --mode faceid --eps 0.3 --min-samples 2
  python Filter.py --input ".\\workspace\\faces" --mode position --threshold 50 --min-samples 2
  python Filter.py --input ".\\workspace\\faces" --mode repeated --threshold 0.98
        """
    )
    
    parser.add_argument(
        '-i', '--input',
        type=str,
        required=True,
        help=S('FILTER_ARG_INPUT')
    )
    
    parser.add_argument(
        '--mode',
        type=str,
        choices=['blur', 'faceid', 'position', 'repeated'],
        required=False,  # 改为可选，因为 merge-only 模式不需要
        default=None,
        help=S('FILTER_ARG_MODE')
    )
    
    parser.add_argument(
        '--threshold',
        type=float,
        default=None,
        help=S('FILTER_ARG_THRESHOLD')  # Used for blur and position modes only
    )
    
    parser.add_argument(
        '--eps',
        type=float,
        default=0.3,
        help=S('FILTER_ARG_EPS')
    )
    
    parser.add_argument(
        '--min-samples',
        type=int,
        default=2,
        help=S('FILTER_ARG_MIN_SAMPLES')
    )
    
    parser.add_argument(
        '-w', '--workers',
        type=int,
        default=None,
        help=S('FILTER_ARG_WORKERS')
    )
    
    parser.add_argument(
        '--model',
        type=str,
        choices=['w600k_mbf', 'w600k_r50'],
        default=None,
        help=S('FILTER_ARG_MODEL')
    )
    
    parser.add_argument(
        '--merge-back',
        action='store_true',
        help='Merge all groups back to a single folder after grouping'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='aligned',
        help='Output directory name for merged files (default: aligned)'
    )
    
    parser.add_argument(
        '--merge-only',
        action='store_true',
        help='Only merge subfolders without performing face ID grouping'
    )

    parser.add_argument(
        '--train-res',
        type=int,
        default=256,
        help='模糊过滤：清晰度归一化分辨率，应与训练分辨率一致 (default: 256)'
    )

    parser.add_argument(
        '--absolute',
        action='store_true',
        help='模糊过滤：用绝对基准阈值（比 σ=1/2 高斯模糊更糊才判低）而不是默认的相对分位'
    )

    return parser.parse_args()


def main():
    """主函数"""
    #依旧雷霆大字
    print("""
=======================================================================================================================
███████╗ █████╗  ██████╗███████╗███████╗██╗     ██╗████████╗███████╗██████╗     ████████╗ ██████╗  ██████╗ ██╗     
██╔════╝██╔══██╗██╔════╝██╔════╝██╔════╝██║     ██║╚══██╔══╝██╔════╝██╔══██╗    ╚══██╔══╝██╔═══██╗██╔═══██╗██║     
█████╗  ███████║██║     █████╗  █████╗  ██║     ██║   ██║   █████╗  ██████╔╝       ██║   ██║   ██║██║   ██║██║     
██╔══╝  ██╔══██║██║     ██╔══╝  ██╔══╝  ██║     ██║   ██║   ██╔══╝  ██╔══██╗       ██║   ██║   ██║██║   ██║██║     
██║     ██║  ██║╚██████╗███████╗██║     ███████╗██║   ██║   ███████╗██║  ██║       ██║   ╚██████╔╝╚██████╔╝███████╗
╚═╝     ╚═╝  ╚═╝ ╚═════╝╚══════╝╚═╝     ╚══════╝╚═╝   ╚═╝   ╚══════╝╚═╝  ╚═╝       ╚═╝    ╚═════╝  ╚═════╝ ╚══════╝
=======================================================================================================================
    """)
    
    args = parse_args()
    
    # 打印用户输入参数
    print("\n" + "="*80)
    print("FacesetProcessor Filter - 参数配置")
    print("="*80)
    print(f"输入路径: {args.input}")
    print(f"过滤模式: {args.mode if args.mode else 'merge-only'}")
    
    if args.mode == 'faceid':
        print(f"  - eps: {args.eps}")
        print(f"  - min_samples: {args.min_samples}")
        print(f"  - model: {args.model if args.model else 'auto-select'}")
        print(f"  - merge_back: {args.merge_back}")
        print(f"  - output_dir: {args.output_dir}")
    elif args.mode == 'blur':
        print(f"  - train_res: {args.train_res} (归一化分辨率)")
        print(f"  - 阈值模式: {'绝对基准(σ=1/2)' if args.absolute else '相对分位 10/30/60'}")
        print(f"  - workers: {args.workers if args.workers else 'auto'}")
    elif args.mode == 'position':
        print(f"  - eps (threshold): {args.threshold if args.threshold is not None else 50.0}")
        print(f"  - min_samples: {args.min_samples if args.min_samples else 2}")
    elif args.mode == 'repeated':
        print(f"  - threshold: {args.threshold if args.threshold is not None else 0.98}")
        print(f"  - hash_method: phash")
        print(f"  - workers: {args.workers if args.workers else 'auto'}")
    elif args.merge_only:
        print(f"  - output_dir: {args.output_dir}")
    
    print("="*80 + "\n")
    
    input_path = Path(args.input)
    if not input_path.exists():
        print(S('PATH_NOT_EXIST', input_path))
        return
    
    if not input_path.is_dir():
        print(S('INVALID_PATH_TYPE', input_path))
        return
    
    # 根据模式执行不同的过滤
    if args.merge_only:
        # 仅合并子文件夹模式（不需要加载模型）
        filter_obj = FaceIDFilter(input_path, skip_model_init=True)
        filter_obj.merge_subfolders_to_aligned(output_dir=args.output_dir)
    
    elif args.mode is None:
        # 如果没有指定 mode 且不是 merge-only，报错
        print("错误: 必须指定 --mode 参数或使用 --merge-only")
        return
    
    elif args.mode == 'blur':
        # 模糊过滤模式：清晰度 = E(b0)/E(b2) 高频/低频能量比
        filter_obj = QualityFilter(input_path)
        filter_obj.filter_by_quality(workers=args.workers,
                                     train_res=args.train_res,
                                     absolute=args.absolute)
    
    elif args.mode == 'faceid':
        # 人脸ID过滤模式
        filter_obj = FaceIDFilter(input_path, model_name=args.model)
        filter_obj.filter_by_face_id(
            eps=args.eps if args.eps is not None else 0.3,
            min_samples=args.min_samples if args.min_samples else 2,
            merge_back=args.merge_back,
            output_dir=args.output_dir
        )
    
    elif args.mode == 'position':
        # 位置过滤模式
        eps = args.threshold if args.threshold is not None else 50.0
        filter_obj = PositionFilter(input_path)
        filter_obj.filter_by_position(
            eps=eps,
            min_samples=args.min_samples if args.min_samples else 2
        )
    
    elif args.mode == 'repeated':
        # 重复图片过滤模式
        threshold = args.threshold if args.threshold is not None else 0.98
        hash_method = 'phash'  # 默认使用phash
        filter_obj = RepeatedFilter(input_path)
        filter_obj.filter_repeated(
            threshold=threshold,
            hash_method=hash_method,
            workers=args.workers
        )


if __name__ == '__main__':
    main()
