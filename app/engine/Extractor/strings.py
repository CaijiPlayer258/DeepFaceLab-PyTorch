"""
Extractor localization strings
"""
import locale

# Detect system language
def _get_system_language():
    try:
        sys_locale = locale.getdefaultlocale()[0]
        if sys_locale:
            lang = sys_locale[0:2].lower()
            if lang in ['zh', 'en']:
                return lang
    except:
        pass
    return 'en'

CURRENT_LANGUAGE = _get_system_language()

# English strings
STRINGS_EN = {
    'BANNER_TITLE': "DeepFaceLab Torch - Face Extractor",
    'CMD_MODE': "[Command Line Mode]",
    'INPUT_PATH': "Input path",
    'OUTPUT_PATH': "Output path",
    'DETECTOR': "Detector",
    'LANDMARKER': "Landmarker",
    'OUTPUT_SIZE': "Output size",
    'OUTPUT_SIZE_DYNAMIC': "Dynamic (based on face bbox)",
    'ENTER_RESIZE_SIZE': "Enter pre-resize width (0=disable, default={}): ",
    'ENTER_INPUT_PATH': "Enter input path (video file or image folder): ",
    'ENTER_OUTPUT_PATH': "Enter output path (save aligned faces): ",
    'PATH_NOT_EXIST': "Error: Path does not exist: {}",
    'INVALID_PATH_TYPE': "Error: Invalid path type: {}",
    'AVAILABLE_DETECTORS': "\nAvailable face detectors:",
    'SELECT_DETECTOR': "\nSelect detector (1-{}, default=5 BlazeFace): ",
    'AVAILABLE_LANDMARKERS': "\nAvailable landmarkers:",
    'SELECT_LANDMARKER': "\nSelect landmarker (1-{}, default=1 insightface-2d106det): ",
    'CONFIGURATION': "\nConfiguration:",
    'CONFIRM_START': "Confirm to start extraction? (y/n): ",
    'CANCELLED': "Cancelled",
    'INIT_DEVICE': "\nInitializing device...",
    'DEVICE_CUDA': "✓ Using CUDA GPU: {}",
    'DEVICE_DIRECTML': "✓ Using DirectML/DX12 GPU: {}",
    'DEVICE_CPU': "⚠ Using CPU: {}",
    'NO_DEVICE': "No available device found",
    'PROCESSING_IMAGES': "\nProcessing image folder: {}",
    'DETECTOR_LOADED': "✓ Detector loaded successfully: {}",
    'LANDMARKER_LOADED': "✓ Landmarker loaded successfully: {}",
    'FOUND_IMAGES': "Found {} images",
    'USING_THREADS': "Using {} threads for parallel processing",
    'NO_IMAGES': "No image files found!",
    'COMPLETE': "\n✓ Complete! Processed {} images, saved {} faces to: {}",
    'METADATA_SAVED': "✓ Metadata saved to: {}",
    'METADATA_ENTRIES': "  Total entries: {}",
    'PROCESSING_VIDEO': "\nProcessing video: {}",
    'VIDEO_INFO': "Video info: {} frames, {:.2f} FPS",
    'CANNOT_OPEN_VIDEO': "✗ Cannot open video file: {}",
    'STAGE_1_DETECT': "[Stage 1/5] Face detection...",
    'STAGE_2_SORT': "[Stage 2/5] Face sorting (inter-frame matching)...",
    'STAGE_3_LANDMARKS': "[Stage 3/5] Landmark extraction...",
    'STAGE_4_ALIGN': "[Stage 4/5] Face alignment...",
    'STAGE_5_SAVE': "[Stage 5/5] Saving output...",
    'TOTAL_DETECTED': "✓ Total detected {} faces",
    'SORT_COMPLETE': "✓ Face sorting complete",
    'TOTAL_LANDMARKS': "✓ Total extracted {} valid landmarks",
    'TOTAL_ALIGNED': "✓ Total aligned {} faces",
    'PROGRESS_DETECT': "Detecting faces",
    'PROGRESS_SORT': "Sorting faces",
    'PROGRESS_LANDMARKS': "Extracting landmarks",
    'PROGRESS_ALIGN': "Aligning faces",
    'PROGRESS_SAVE': "Saving files",
    'PROGRESS_PROCESSING': "Processing",
    'ALIGN_SAVE_FAILED': "Align save failed [{}, face {}]: {}",
    'PROCESS_IMAGE_FAILED': "Process image failed [{}]: {}",
    'LANDMARK_EXTRACT_ERROR': "Landmark extraction error: {}",
    'FACE_DETECT_ERROR': "Face detection error: {}",
    'LOAD_DETECTOR_FAILED': "✗ Failed to load detector [{}]: {}",
    'LOAD_LANDMARKER_FAILED': "✗ Failed to load landmarker [{}]: {}",
    'UNSUPPORTED_DETECTOR': "Unsupported detector: {}",
    'UNSUPPORTED_LANDMARKER': "Unsupported landmarker: {}",
    'PROCESSING_ERROR': "\nError during processing:",
    'WARNING_METADATA_FAILED': "Warning: Failed to add metadata for {}: {}",
    'TIMEOUT': "\n⚠ Timeout: {}",
    'FAILED': "\n✗ Failed [{}]: {}",
    'WORKER_NOT_INIT': "Worker not initialized for frame {}",
    'INIT_WORKER_FAILED': "Failed to initialize worker process: {}",
    'DEBUG_COMPLETE': "\n✓ Debug complete! Check {} for intermediate results",
    'VIDEO_INFO_FULL': "Video info: {} frames, {:.2f} FPS, Resolution: {}x{}",
    'PRE_RESIZE': "Pre-resize",
    'PRE_RESIZE_DISABLED': "Disabled",
    'RESIZE_AUTO_DISABLED': "ℹ️ Auto-disabled pre-resize: media width ({}px) <= resize value ({}px)",
    'RESIZE_CHECK_ERROR': "⚠️ Failed to check media resolution: {}",
    'PROCESSING_VIDEO_DIR': "\nProcessing video directory: {}",
    'FOUND_VIDEOS': "Found {} video files",
    'NO_VIDEOS_FOUND': "No video files found in directory!",
    'BATCH_COMPLETE_SUMMARY': "\n✓ Batch processing complete! Processed {} videos, {} frames, saved {} faces to: {}",
    'AUTO_DETECT_VIDEO_MODE': "ℹ️ Auto-detected video directory mode",
    'AUTO_DETECT_IMAGE_MODE': "ℹ️ Auto-detected image directory mode",
    'MIXED_CONTENT_DETECTED': "\n⚠ Mixed content detected: {} videos and {} images found",
    'SELECT_PROCESSING_MODE': "Select processing mode (v=video, i=image, default=v): ",
    'NO_SUPPORTED_FILES': "No supported video or image files found in directory!",
}

# Chinese strings
STRINGS_ZH = {
    'BANNER_TITLE': "DeepFaceLab Torch - 人脸提取器",
    'CMD_MODE': "[命令行模式]",
    'INPUT_PATH': "输入路径",
    'OUTPUT_PATH': "输出路径",
    'DETECTOR': "检测器",
    'LANDMARKER': "特征点标记器",
    'OUTPUT_SIZE': "输出尺寸",
    'OUTPUT_SIZE_DYNAMIC': "动态（基于人脸边界框）",
    'ENTER_RESIZE_SIZE': "请输入预缩放宽度（0=禁用，默认={}）: ",
    'ENTER_INPUT_PATH': "请输入输入路径（视频文件或图片文件夹）: ",
    'ENTER_OUTPUT_PATH': "请输入输出路径（保存对齐后的人脸）: ",
    'PATH_NOT_EXIST': "错误: 路径不存在: {}",
    'INVALID_PATH_TYPE': "错误: 无效的路径类型: {}",
    'AVAILABLE_DETECTORS': "\n可用的人脸检测器:",
    'SELECT_DETECTOR': "\n请选择检测器 (1-{}, 默认=5 BlazeFace): ",
    'AVAILABLE_LANDMARKERS': "\n可用的特征点标记器:",
    'SELECT_LANDMARKER': "\n请选择特征点标记器 (1-{}, 默认=1 insightface-2d106det): ",
    'CONFIGURATION': "\n配置确认:",
    'CONFIRM_START': "确认开始提取？(y/n): ",
    'CANCELLED': "已取消",
    'INIT_DEVICE': "\n初始化设备...",
    'DEVICE_CUDA': "✓ 使用 CUDA GPU: {}",
    'DEVICE_DIRECTML': "✓ 使用 DirectML/DX12 GPU: {}",
    'DEVICE_CPU': "⚠ 使用 CPU: {}",
    'NO_DEVICE': "未找到可用设备",
    'PROCESSING_IMAGES': "\n开始处理图片文件夹: {}",
    'DETECTOR_LOADED': "✓ 成功加载检测器: {}",
    'LANDMARKER_LOADED': "✓ 成功加载特征点标记器: {}",
    'FOUND_IMAGES': "找到 {} 个图像文件",
    'USING_THREADS': "使用 {} 个线程进行并行处理",
    'NO_IMAGES': "未找到图片文件！",
    'COMPLETE': "\n✓ 完成！处理了 {} 张图片，保存了 {} 个人脸到: {}",
    'METADATA_SAVED': "✓ 元数据已保存到: {}",
    'METADATA_ENTRIES': "  总条目数: {}",
    'PROCESSING_VIDEO': "\n开始处理视频: {}",
    'VIDEO_INFO': "视频信息: {} 帧, {:.2f} FPS",
    'CANNOT_OPEN_VIDEO': "✗ 无法打开视频文件: {}",
    'STAGE_1_DETECT': "[阶段1/5] 人脸检测...",
    'STAGE_2_SORT': "[阶段2/5] 人脸排序（帧间匹配）...",
    'STAGE_3_LANDMARKS': "[阶段3/5] 特征点提取...",
    'STAGE_4_ALIGN': "[阶段4/5] 人脸对齐...",
    'STAGE_5_SAVE': "[阶段5/5] 保存输出...",
    'TOTAL_DETECTED': "✓ 共检测到 {} 个人脸",
    'SORT_COMPLETE': "✓ 人脸排序完成",
    'TOTAL_LANDMARKS': "✓ 共提取 {} 个有效特征点",
    'TOTAL_ALIGNED': "✓ 共对齐 {} 个人脸",
    'PROGRESS_DETECT': "检测人脸",
    'PROGRESS_SORT': "排序人脸",
    'PROGRESS_LANDMARKS': "提取特征点",
    'PROGRESS_ALIGN': "对齐人脸",
    'PROGRESS_SAVE': "保存文件",
    'PROGRESS_PROCESSING': "处理中",
    'ALIGN_SAVE_FAILED': "对齐保存失败 [{}, 人脸{}]: {}",
    'PROCESS_IMAGE_FAILED': "处理图片失败 [{}]: {}",
    'LANDMARK_EXTRACT_ERROR': "特征点提取出错: {}",
    'FACE_DETECT_ERROR': "人脸检测出错: {}",
    'LOAD_DETECTOR_FAILED': "✗ 加载检测器失败 [{}]: {}",
    'LOAD_LANDMARKER_FAILED': "✗ 加载特征点标记器失败 [{}]: {}",
    'UNSUPPORTED_DETECTOR': "不支持的检测器: {}",
    'UNSUPPORTED_LANDMARKER': "不支持的特征点标记器: {}",
    'PROCESSING_ERROR': "\n处理过程中发生错误:",
    'WARNING_METADATA_FAILED': "警告: 无法为 {} 添加元数据: {}",
    'TIMEOUT': "\n⚠ 超时: {}",
    'FAILED': "\n✗ 失败 [{}]: {}",
    'WORKER_NOT_INIT': "工作进程未初始化，帧号 {}",
    'INIT_WORKER_FAILED': "工作进程初始化失败: {}",
    'DEBUG_COMPLETE': "\n✓ 调试完成！检查 {} 查看中间结果",
    'VIDEO_INFO_FULL': "视频信息: {} 帧, {:.2f} FPS, 分辨率: {}x{}",
    'PRE_RESIZE': "预缩放尺寸",
    'PRE_RESIZE_DISABLED': "禁用",
    'RESIZE_AUTO_DISABLED': "ℹ️ 自动禁用预缩放：媒体宽度({}px) ≤ 预缩放尺寸({}px)",
    'RESIZE_CHECK_ERROR': "⚠️ 无法检测媒体分辨率: {}",
    'PROCESSING_VIDEO_DIR': "\n处理视频目录: {}",
    'FOUND_VIDEOS': "找到 {} 个视频文件",
    'NO_VIDEOS_FOUND': "目录中未找到视频文件！",
    'BATCH_COMPLETE_SUMMARY': "\n✓ 批量处理完成！处理了 {} 个视频，{} 帧，保存了 {} 个人脸到: {}",
    'AUTO_DETECT_VIDEO_MODE': "ℹ️ 自动检测到视频目录模式",
    'AUTO_DETECT_IMAGE_MODE': "ℹ️ 自动检测到图片目录模式",
    'MIXED_CONTENT_DETECTED': "\n⚠ 检测到混合内容：找到 {} 个视频和 {} 张图片",
    'SELECT_PROCESSING_MODE': "请选择处理模式（v=视频，i=图片，默认=v）: ",
    'NO_SUPPORTED_FILES': "目录中未找到支持的视频或图片文件！",
}

# Language strings database
STRINGS_DB = {
    'en': STRINGS_EN,
    'zh': STRINGS_ZH,
}


def S(key, *args):
    """
    Get localized string
    Usage: S('KEY_NAME') or S('KEY_WITH_FORMAT', arg1, arg2)
    """
    db = STRINGS_DB.get(CURRENT_LANGUAGE, STRINGS_DB['en'])
    text = db.get(key, key)  # Fallback to key if not found
    if args:
        return text.format(*args)
    return text


def set_language(lang):
    """Set current language ('en' or 'zh')"""
    global CURRENT_LANGUAGE
    if lang in STRINGS_DB:
        CURRENT_LANGUAGE = lang
