from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QSizePolicy, QBoxLayout
import subprocess
import re
import threading
from contextlib import contextmanager
from pathlib import Path
from siui.components.combobox_ import SiCapsuleComboBox
from siui.components.container import SiDenseContainer, SiTriSectionPanelCard
from siui.components.editbox import SiLabeledLineEdit
from siui.components.page import SiPage
from siui.components.button import SiPushButtonRefactor
from siui.components import SiTitledWidgetGroup
from siui.components.widgets import SiCheckBox
@contextmanager
def createPanelCard(parent: SiTitledWidgetGroup, title: str) -> SiTriSectionPanelCard:
    """创建面板卡片的上下文管理器"""
    card = SiTriSectionPanelCard(parent)
    card.setTitle(title)
    try:
        yield card
    finally:
        card.adjustSize()
        parent.addWidget(card)
@contextmanager
def createDenseContainer(parent,
                         direction: QBoxLayout.Direction,
                         side: Qt.Edges = Qt.LeftEdge | Qt.TopEdge):
    """创建密集容器的上下文管理器"""
    from siui.components.container import SiDenseContainer
    container = SiDenseContainer(parent)
    container.layout().setDirection(direction)
    container.layout().setSpacing(12)
    container.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
    try:
        yield container
    finally:
        parent.addWidget(container, side)
class DataExtractionPage(SiPage):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setPadding(64)
        self.setScrollMaximumWidth(1000)
        self.setScrollAlignment(Qt.AlignLeft)
        self.setTitle("数据提取")
        # 创建滚动容器
        self.titled_widgets_group = SiTitledWidgetGroup(self)
        self.titled_widgets_group.setSpacing(32)
        self.titled_widgets_group.setAdjustWidgetsSize(True)
        # 第一组：帧提取相关
        with self.titled_widgets_group as group:
            group.addTitle("帧提取")
            # 原始帧提取卡片
            with createPanelCard(group, "原始帧提取") as card:
                with createDenseContainer(card.body(), QBoxLayout.TopToBottom) as container:
                    # 视频路径输入框
                    self.video_path_input = SiLabeledLineEdit(self)
                    self.video_path_input.setTitle("视频路径")
                    self.video_path_input.setPlaceholderText("请输入视频文件路径...")
                    project_root = Path(__file__).parent.parent.parent.parent
                    self.video_path_input.setText(str(project_root / "workspace" / "data_dst.mp4"))
                    self.video_path_input.resize(700, 64)
                    # 输出路径输入框
                    self.output_path_input = SiLabeledLineEdit(self)
                    self.output_path_input.setTitle("输出路径")
                    self.output_path_input.setPlaceholderText("请输入输出文件夹路径...")
                    self.output_path_input.setText(str(project_root / "workspace" / "data_dst"))
                    self.output_path_input.resize(700, 64)
                    container.addWidget(self.video_path_input)
                    container.addWidget(self.output_path_input)
                with createDenseContainer(card.body(), QBoxLayout.TopToBottom) as row2:
                    # FFmpeg 路径输入框
                    self.ffmpeg_path_input = SiLabeledLineEdit(self)
                    self.ffmpeg_path_input.setTitle("FFmpeg 目录")
                    project_root = Path(__file__).parent.parent.parent.parent
                    self.ffmpeg_path_input.setText(str(project_root / "ffmpeg"))
                    self.ffmpeg_path_input.setPlaceholderText("FFmpeg 可执行文件所在目录...")
                    self.ffmpeg_path_input.resize(700, 64)
                    self.ffmpeg_path_input.setToolTip("FFmpeg 目录路径，需包含 ffmpeg.exe")
                    row2.addWidget(self.ffmpeg_path_input)
                with createDenseContainer(card.body(), QBoxLayout.LeftToRight) as row3:
                    # 帧数选择输入框
                    self.frame_count_input = SiLabeledLineEdit(self)
                    self.frame_count_input.setTitle("提取帧数")
                    self.frame_count_input.setPlaceholderText("0=全部")
                    self.frame_count_input.setText("0")
                    self.frame_count_input.resize(150, 48)
                    self.frame_count_input.setToolTip("0=提取全部帧，输入数字提取指定帧数")
                    # 提取按钮
                    self.extract_frames_button = SiPushButtonRefactor(self)
                    self.extract_frames_button.setText("提取帧")
                    self.extract_frames_button.resize(150, 48)
                    self.extract_frames_button.setToolTip("使用 FFmpeg 从视频中提取帧图片")
                    self.extract_frames_button.clicked.connect(self.on_extract_frames_clicked)
                    row3.addWidget(self.frame_count_input)
                    row3.addWidget(self.extract_frames_button)
            # 导出格式选择卡片
            with createPanelCard(group, "导出设置") as card:
                with createDenseContainer(card.body(), QBoxLayout.LeftToRight) as container:
                    # 图片格式选择组合框
                    self.image_format_combobox = SiCapsuleComboBox(self)
                    self.image_format_combobox.setTitle("图片格式")
                    self.image_format_combobox.setMinimumHeight(36)
                    self.image_format_combobox.setEditable(False)
                    self.image_format_combobox.addItems(["png", "jpg"])
                    self.image_format_combobox.setToolTip("选择导出图片的格式")
                    self.image_format_combobox.currentIndexChanged.connect(self._update_extract_tooltip)
                    # 编码方式选择组合框
                    self.hardware_acceleration_combobox = SiCapsuleComboBox(self)
                    self.hardware_acceleration_combobox.setTitle("编码方式")
                    self.hardware_acceleration_combobox.setMinimumHeight(36)
                    self.hardware_acceleration_combobox.setEditable(False)
                    self.hardware_acceleration_combobox.addItems([
                        "h264_nvenc",
                        "hevc_nvenc",
                        "h264_amf",
                        "hevc_amf",
                        "h264_qsv",
                        "hevc_qsv",
                        "libx264",
                        "libx265",
                        "copy",
                    ])
                    self.hardware_acceleration_combobox.setToolTip("选择 FFmpeg 视频编码方式")
                    # 处理模式选择组合框
                    self.process_mode_extract = SiCapsuleComboBox(self)
                    self.process_mode_extract.setTitle("处理模式")
                    self.process_mode_extract.setMinimumHeight(36)
                    self.process_mode_extract.setEditable(False)
                    self.process_mode_extract.addItems(["处理单个文件", "处理目录"])
                    self.process_mode_extract.setToolTip("选择处理方式：单个文件或整个目录")
                    self.process_mode_extract.currentIndexChanged.connect(self._update_extract_tooltip)
                    container.addWidget(self.image_format_combobox)
                    container.addWidget(self.hardware_acceleration_combobox)
                    container.addWidget(self.process_mode_extract)
            # 杜比视界转码卡片
            with createPanelCard(group, "杜比视界转码") as card:
                with createDenseContainer(card.body(), QBoxLayout.TopToBottom) as container:
                    self.dovi_input_input = SiLabeledLineEdit(self)
                    self.dovi_input_input.setTitle("输入路径")
                    project_root = Path(__file__).parent.parent.parent.parent
                    self.dovi_input_input.setText(str(project_root / "workspace"))
                    self.dovi_input_input.setPlaceholderText("视频文件或文件夹路径...")
                    self.dovi_input_input.resize(700, 64)
                    container.addWidget(self.dovi_input_input)
                with createDenseContainer(card.body(), QBoxLayout.LeftToRight) as row1:
                    self.dovi_target_combobox = SiCapsuleComboBox(self)
                    self.dovi_target_combobox.setTitle("目标色彩空间")
                    self.dovi_target_combobox.setMinimumHeight(36)
                    self.dovi_target_combobox.addItems(["Rec.709 (SDR)", "Rec.2020 (HDR)"])
                    self.dovi_target_combobox.setCurrentText("Rec.709 (SDR)")
                    self.dovi_mode_combobox = SiCapsuleComboBox(self)
                    self.dovi_mode_combobox.setTitle("转码质量")
                    self.dovi_mode_combobox.setMinimumHeight(36)
                    self.dovi_mode_combobox.addItems(["硬件加速(快速)", "硬件加速(高画质)", "CPU(最慢)"])
                    self.dovi_mode_combobox.setCurrentText("硬件加速(快速)")
                    self.dovi_bitrate_combobox = SiCapsuleComboBox(self)
                    self.dovi_bitrate_combobox.setTitle("比特率策略")
                    self.dovi_bitrate_combobox.setMinimumHeight(36)
                    self.dovi_bitrate_combobox.addItems(["自动(同原视频)", "手动"])
                    self.dovi_bitrate_combobox.setCurrentText("自动(同原视频)")
                    self.dovi_bitrate_combobox.currentIndexChanged.connect(self._toggle_dovi_bitrate)
                    row1.addWidget(self.dovi_target_combobox)
                    row1.addWidget(self.dovi_mode_combobox)
                    row1.addWidget(self.dovi_bitrate_combobox)
                with createDenseContainer(card.body(), QBoxLayout.LeftToRight) as row2:
                    self.dovi_bitrate_input = SiLabeledLineEdit(self)
                    self.dovi_bitrate_input.setTitle("比特率(bps)")
                    self.dovi_bitrate_input.setPlaceholderText("如10000000")
                    self.dovi_bitrate_input.setText("10000000")
                    self.dovi_bitrate_input.resize(200, 48)
                    self.dovi_bitrate_input.setEnabled(False)
                    self.dovi_run_button = SiPushButtonRefactor(self)
                    self.dovi_run_button.setText("开始转码")
                    self.dovi_run_button.resize(150, 48)
                    self.dovi_run_button.clicked.connect(self._on_dovi_transcode)
                    row2.addWidget(self.dovi_bitrate_input)
                    row2.addWidget(self.dovi_run_button)

        # 第二组：人脸提取相关
        with self.titled_widgets_group as group:
            group.addTitle("人脸提取")
            # 人脸提取路径卡片
            with createPanelCard(group, "人脸提取路径") as card:
                with createDenseContainer(card.body(), QBoxLayout.TopToBottom) as container:
                    # 输入路径输入框
                    self.face_input_path_input = SiLabeledLineEdit(self)
                    self.face_input_path_input.setTitle("输入路径")
                    self.face_input_path_input.setPlaceholderText("请输入视频文件或文件夹路径...")
                    # 使用相对于项目根目录的路径
                    project_root = Path(__file__).parent.parent.parent.parent
                    default_input = project_root / "workspace" / "data_dst"
                    self.face_input_path_input.setText(str(default_input))
                    self.face_input_path_input.resize(700, 64)
                    # 输出路径输入框
                    self.face_output_path_input = SiLabeledLineEdit(self)
                    self.face_output_path_input.setTitle("输出路径")
                    self.face_output_path_input.setPlaceholderText("请输入输出文件夹路径...")
                    # 使用相对于项目根目录的路径
                    default_output = project_root / "workspace" / "data_dst" / "aligned"
                    self.face_output_path_input.setText(str(default_output))
                    self.face_output_path_input.resize(700, 64)
                    container.addWidget(self.face_input_path_input)
                    container.addWidget(self.face_output_path_input)
            # 人脸提取运行设置卡片
            with createPanelCard(group, "运行设置") as card:
                with createDenseContainer(card.body(), QBoxLayout.TopToBottom) as container:
                    # 第一行：人脸检测算法 + 特征点标记算法 + 输出格式
                    with createDenseContainer(container, QBoxLayout.LeftToRight) as row1:
                        # 人脸检测算法选择组合框（与Extractor.py保持一致）
                        self.face_detector_combobox = SiCapsuleComboBox(self)
                        self.face_detector_combobox.setTitle("人脸检测算法")
                        self.face_detector_combobox.setMinimumHeight(36)
                        self.face_detector_combobox.setEditable(False)
                        self.face_detector_combobox.setMaxVisibleItems(15)
                        self.face_detector_combobox.addItems([
                            "BlazeFace",
                            "CenterFace",
                            "DamoFD",
                            "FastFaceAlign",
                            "LightweightFD",
                            "MogFace",
                            "MTCNN",
                            "RetinaFace_10g",
                            "RetinaFace_500m",
                            "S3FD",
                            "TinyMog",
                            "ULFD",
                            "YoloV5Face",
                            "YoloV8Face",
                            "YoloV11nFace"
                        ])
                        self.face_detector_combobox.setCurrentText("TinyMog")
                        self.face_detector_combobox.setToolTip("选择人脸检测算法\n推荐: TinyMog (轻量高精度)\nDamoFD: ICLR 2023 轻量高精度\nMogFace: CVPR 2022 最高精度(大模型)\nMTCNN: 经典级联检测+关键点")
                        self.face_detector_combobox.currentIndexChanged.connect(self._update_face_tooltip)
                        # 特征点标记算法选择组合框（与Extractor.py保持一致）
                        self.landmark_detector_combobox = SiCapsuleComboBox(self)
                        self.landmark_detector_combobox.setTitle("特征点标记算法")
                        self.landmark_detector_combobox.setMinimumHeight(36)
                        self.landmark_detector_combobox.setEditable(False)
                        self.landmark_detector_combobox.addItems([
                            "insightface-2d106det",
                            "2DFAN-4",
                            "Google-mediapipe",
                            "MobileFaceNet",
                            "OpenSeeFace",
                            "PFLD"
                        ])
                        self.landmark_detector_combobox.setCurrentText("insightface-2d106det")
                        self.landmark_detector_combobox.setToolTip("选择特征点标记算法")
                        self.landmark_detector_combobox.currentIndexChanged.connect(self._update_face_tooltip)
                        # 输出格式选择组合框
                        self.face_output_format_combobox = SiCapsuleComboBox(self)
                        self.face_output_format_combobox.setTitle("输出格式")
                        self.face_output_format_combobox.setMinimumHeight(36)
                        self.face_output_format_combobox.setEditable(False)
                        self.face_output_format_combobox.addItems(["jpg", "png"])
                        self.face_output_format_combobox.setCurrentText("jpg")
                        self.face_output_format_combobox.currentIndexChanged.connect(self._update_face_tooltip)
                        self.face_output_format_combobox.setToolTip("选择人脸图片输出格式")
                        row1.addWidget(self.face_detector_combobox)
                        row1.addWidget(self.landmark_detector_combobox)
                        row1.addWidget(self.face_output_format_combobox)
                    # 第三行：脸型类型 + 处理模式 + 预缩放尺寸 + 检测角度
                    with createDenseContainer(container, QBoxLayout.LeftToRight) as row3:
                        # 脸型类型选择组合框（对应原版DeepFaceLab的face_type）
                        self.face_type_combobox = SiCapsuleComboBox(self)
                        self.face_type_combobox.setTitle("脸型类型")
                        self.face_type_combobox.setMinimumHeight(36)
                        self.face_type_combobox.setEditable(False)
                        self.face_type_combobox.addItems([
                            "half_face",
                            "midfull_face",
                            "full_face",
                            "whole_face",
                            "head"
                        ])
                        self.face_type_combobox.setCurrentText("whole_face")
                        self.face_type_combobox.currentIndexChanged.connect(self._update_face_tooltip)
                        self.face_type_combobox.setToolTip(
                            "选择提取的脸型范围:\n"
                            "- half_face: 半脸\n"
                            "- midfull_face: 中全脸\n"
                            "- full_face: 全脸\n"
                            "- whole_face: 整脸（包含更多头部区域）\n"
                            "- head: 头部（包含整个头部）"
                        )
                        # 处理模式选择组合框
                        self.process_mode_combobox = SiCapsuleComboBox(self)
                        self.process_mode_combobox.setTitle("处理模式")
                        self.process_mode_combobox.setMinimumHeight(36)
                        self.process_mode_combobox.setEditable(False)
                        self.process_mode_combobox.addItems(["单文件/图片", "视频目录批量"])
                        self.process_mode_combobox.setCurrentText("单文件/图片")
                        self.process_mode_combobox.setToolTip(
                            "选择处理模式:\n"
                            "- 单文件/图片: 处理单个视频文件或图片目录\n"
                            "- 视频目录批量: 遍历目录下所有视频，统一输出到同一文件夹"
                        )
                        self.process_mode_combobox.currentIndexChanged.connect(self._update_face_tooltip)
                        # 预缩放尺寸输入框
                        self.resize_input = SiLabeledLineEdit(self)
                        self.resize_input.setTitle("预缩放尺寸")
                        self.resize_input.setPlaceholderText("0=禁用，推荐720")
                        self.resize_input.setText("720")
                        self.resize_input.resize(200, 48)
                        self.resize_input.setToolTip("输入图像预缩放宽度（像素），0表示禁用。当原始分辨率小于此值时自动禁用。\n仅 Sliding Window 模式生效；One-Stage 模式整图缩放，此值无效。")
                        
                        # 检测角度输入框
                        self.detection_angles_input = SiLabeledLineEdit(self)
                        self.detection_angles_input.setTitle("检测角度")
                        self.detection_angles_input.setPlaceholderText("0,90,180,270")
                        self.detection_angles_input.setText("0")
                        self.detection_angles_input.resize(250, 48)
                        self.detection_angles_input.setToolTip(
                            "人脸检测角度（逗号分隔），例如:\n"
                            "- 0: 仅正常方向（最快）\n"
                            "- 0,90,180,270: 多方向检测（适合侧脸、倒置等场景）\n"
                            "默认: 0"
                        )

                        # 帧跳跃输入框（仅视频模式有效）
                        self.skip_frames_input = SiLabeledLineEdit(self)
                        self.skip_frames_input.setTitle("帧跳跃")
                        self.skip_frames_input.setPlaceholderText("0=逐帧")
                        self.skip_frames_input.setText("0")
                        self.skip_frames_input.resize(150, 48)
                        self.skip_frames_input.setToolTip(
                            "视频帧跳跃步长（仅视频模式有效）：\n"
                            "- 0: 逐帧提取（默认）\n"
                            "- 1: 每2帧提取1帧\n"
                            "- 2: 每3帧提取1帧\n"
                            "数值越大速度越快，但可能遗漏人脸"
                        )

                        row3.addWidget(self.face_type_combobox)
                        row3.addWidget(self.process_mode_combobox)
                        row3.addWidget(self.resize_input)
                        row3.addWidget(self.detection_angles_input)
                        row3.addWidget(self.skip_frames_input)
                    # 第三行B：检测模式 + 缩放方式
                    with createDenseContainer(container, QBoxLayout.LeftToRight) as row3b:
                        self.input_mode_combobox = SiCapsuleComboBox(self)
                        self.input_mode_combobox.setTitle("检测模式")
                        self.input_mode_combobox.setMinimumHeight(36)
                        self.input_mode_combobox.addItems([
                            "One-Stage (整图缩放·快)",
                            "Sliding Window (滑窗扫描)",
                        ])
                        self.input_mode_combobox.setCurrentText("One-Stage (整图缩放·快)")
                        self.input_mode_combobox.setToolTip(
                            "检测模式：\n"
                            "- One-Stage: 整图缩放到输入尺寸一次检测（快）\n"
                            "- Sliding Window: 固定窗口滑动扫描，逐窗口检测后合并（适合大图小脸）\n"
                            "适合滑窗的检测器：BlazeFace、MTCNN\n"
                            "适合 One-Stage 的检测器：RetinaFace、DamoFD、TinyMog、YOLO 系"
                        )
                        self.input_mode_combobox.currentIndexChanged.connect(self._on_input_mode_changed)
                        row3b.addWidget(self.input_mode_combobox)
                        # 缩放方式
                        self.resize_mode_combobox = SiCapsuleComboBox(self)
                        self.resize_mode_combobox.setTitle("缩放方式")
                        self.resize_mode_combobox.setMinimumHeight(36)
                        self.resize_mode_combobox.addItems([
                            "LetterBox (保纵横比)",
                            "WarpAffine (拉伸·最快)",
                        ])
                        self.resize_mode_combobox.setCurrentText("LetterBox (保纵横比)")
                        self.resize_mode_combobox.setToolTip(
                            "缩放方式（作用于 One-Stage 整图缩放 和 Sliding Window 边缘窗口）：\n"
                            "- LetterBox: 等比缩放+补边，不变形（默认）\n"
                            "- WarpAffine: 直接拉伸填满，速度最快，但变形"
                        )
                        row3b.addWidget(self.resize_mode_combobox)
                        # 输入尺寸
                        self.input_size_input = SiLabeledLineEdit(self)
                        self.input_size_input.setTitle("窗口/输入尺寸")
                        self.input_size_input.setPlaceholderText("默认640")
                        self.input_size_input.setText("640")
                        self.input_size_input.resize(160, 48)
                        self.input_size_input.setToolTip(
                            "One-Stage 的整图缩放尺寸 / Sliding Window 的扫描窗口边长（默认640）。\n"
                            "非 8 倍数会自动规整到 8 的倍数。"
                        )
                        row3b.addWidget(self.input_size_input)
                        row3b.layout().addStretch()
                    # 第四行：KPS对齐开关
                    with createDenseContainer(container, QBoxLayout.LeftToRight) as row4:
                        self.kps_align_checkbox = SiCheckBox(self)
                        self.kps_align_checkbox.setText("KPS对齐（利用检测器5点关键点预旋转歪脸）")
                        self.kps_align_checkbox.setMinimumHeight(36)
                        self.kps_align_checkbox.setChecked(True)
                        self.kps_align_checkbox.setMinimumWidth(400)
                        self.kps_align_checkbox.setToolTip(
                            "使用检测器5点关键点预旋转人脸，提高歪脸特征点标记精度\n"
                            "仅支持 RetinaFace、DamoFD、TinyMog 检测器\n"
                            "关闭后使用原始提取方式"
                        )
                        row4.addWidget(self.kps_align_checkbox)
                        row4.layout().addStretch()
                    # 第四行B：HDR 模式选择
                    with createDenseContainer(container, QBoxLayout.LeftToRight) as row4hdr:
                        self.face_hdr_mode_combobox = SiCapsuleComboBox(self)
                        self.face_hdr_mode_combobox.setTitle("HDR 模式")
                        self.face_hdr_mode_combobox.setMinimumHeight(36)
                        self.face_hdr_mode_combobox.addItems(["标准模式(快速,后处理色调映射)", "HDR精确模式(慢速,管道内色调映射)"])
                        self.face_hdr_mode_combobox.setCurrentText("标准模式(快速,后处理色调映射)")
                        self.face_hdr_mode_combobox.setToolTip("标准=bt709直出满速切脸+后处理色调映射 | HDR精确=管道内libplacebo色调映射~6fps")
                        row4hdr.addWidget(self.face_hdr_mode_combobox)
                        row4hdr.layout().addStretch()
                    # 第五行：运行按钮
                    with createDenseContainer(container, QBoxLayout.LeftToRight) as row5:
                        # 运行按钮
                        self.run_extractor_button = SiPushButtonRefactor(self)
                        self.run_extractor_button.setText("开始提取人脸")
                        self.run_extractor_button.resize(200, 48)
                        self.run_extractor_button.setToolTip("点击开始执行人脸提取任务")
                        self.run_extractor_button.clicked.connect(self.on_run_extractor_clicked)
                        
                        row5.addWidget(self.run_extractor_button)

        # 第三组：杜比视界后处理
        with self.titled_widgets_group as group:
            group.addTitle("杜比视界后处理")
            with createPanelCard(group, "色调映射") as card:
                with createDenseContainer(card.body(), QBoxLayout.LeftToRight) as container_tonemap:
                    self.tonemap_input = SiLabeledLineEdit(self)
                    self.tonemap_input.setTitle("aligned 目录")
                    project_root = Path(__file__).parent.parent.parent.parent
                    self.tonemap_input.setText(str(project_root / "workspace" / "data_dst" / "aligned"))
                    self.tonemap_input.resize(600, 64)
                    self.tonemap_input.setToolTip("已提取人脸目录，对该目录下的图片做 HDR->SDR 色调映射")
                    self.tonemap_button = SiPushButtonRefactor(self)
                    self.tonemap_button.setText("杜比视界色调映射")
                    self.tonemap_button.resize(180, 48)
                    self.tonemap_button.setToolTip("对已提取的人脸做批次色调映射(bat709参数)")
                    self.tonemap_button.clicked.connect(self._on_tonemap_clicked)
                    container_tonemap.addWidget(self.tonemap_input)
                    container_tonemap.addWidget(self.tonemap_button)

        # 添加页脚空白
        self.titled_widgets_group.addPlaceholder(64)
        # 设置为页面对象
        self.setAttachment(self.titled_widgets_group)

        # 初始化 tooltip（让默认选项也有详细注释）
        self._update_extract_tooltip()
        self._update_face_tooltip()


    # ── 杜比视界转码 ─────────────────────────────────────
    def _toggle_dovi_bitrate(self):
        self.dovi_bitrate_input.setEnabled(self.dovi_bitrate_combobox.currentText() == "手动")

    def _on_input_mode_changed(self):
        # One-Stage 模式整图缩放，预缩放无意义 → 禁用
        self.resize_input.setEnabled("Sliding Window" in self.input_mode_combobox.currentText())

    def _get_dovi_params(self, target: str):
        if '709' in target:
            return (1, 2, 1, "_Rec709", "Rec.709")
        return (9, 14, 9, "_Rec2020", "Rec.2020")

    def _on_dovi_transcode(self):
        import subprocess, threading
        input_path = self.dovi_input_input.text().strip()
        if not input_path:
            print("错误: 请输入输入路径")
            return
        project_root = Path(__file__).parent.parent.parent.parent
        input_path_obj = Path(input_path)
        if not input_path_obj.is_absolute():
            input_path_obj = project_root / input_path
        if not input_path_obj.exists():
            print(f"错误: 路径不存在: {input_path_obj}")
            return
        target_txt = self.dovi_target_combobox.currentText()
        cs, trc, prim, suffix, desc = self._get_dovi_params(target_txt)
        mode_txt = self.dovi_mode_combobox.currentText()
        bitrate_manual = self.dovi_bitrate_combobox.currentText() == "手动"
        manual_bps = self.dovi_bitrate_input.text().strip() if bitrate_manual else ""
        files = [input_path_obj] if input_path_obj.is_file() else (
            sorted(input_path_obj.glob("*.mkv")) + sorted(input_path_obj.glob("*.mp4")))
        if not files:
            print("错误: 未找到 .mkv/.mp4 视频文件")
            return
        ffmpeg_exe = project_root / "ffmpeg" / "ffmpeg.exe"
        if not ffmpeg_exe.exists():
            print(f"错误: FFmpeg 未找到: {ffmpeg_exe}")
            return
        print(f"杜比视界转码->{desc} | {len(files)}个文件 | {mode_txt}")
        self.dovi_run_button.setText("转码中...")
        self.dovi_run_button.setEnabled(False)
        def _run():
            try:
                for i, fp in enumerate(files):
                    out_path = fp.parent / (fp.stem + suffix + ".mp4")
                    if bitrate_manual and manual_bps:
                        bitrate = manual_bps
                    else:
                        bitrate = "10000000"
                    filter_str = (f"libplacebo=tonemapping=hable:colorspace={cs}:color_trc={trc}:"
                                  f"color_primaries={prim}:range=tv:dithering=blue:format=yuv420p10le")
                    # 尝试 NVENC 硬件编码，失败则回退 libx265
                    cmd = [str(ffmpeg_exe), "-y", "-i", str(fp),
                           "-filter_complex", filter_str,
                           "-c:v", "hevc_nvenc", "-preset", "p7", "-tier", "high",
                           "-b:v", bitrate, "-c:a", "copy", str(out_path)]
                    try:
                        subprocess.run(cmd, capture_output=True, timeout=5)
                    except:
                        pass
                    if not out_path.exists() or out_path.stat().st_size < 1000:
                        cmd = [str(ffmpeg_exe), "-i", str(fp),
                               "-filter_complex", filter_str,
                               "-c:v", "libx265", "-preset", "medium",
                               "-b:v", bitrate, "-c:a", "copy", str(out_path)]
                    cmd_str = " ".join(cmd) + " & echo. & echo 完成！按任意键关闭窗口... & pause"
                    if hasattr(subprocess, "CREATE_NEW_CONSOLE"):
                        subprocess.Popen(f"cmd /c {cmd_str}",
                            creationflags=subprocess.CREATE_NEW_CONSOLE).wait()
                    else:
                        subprocess.run(cmd)
            finally:
                self.dovi_run_button.setText("开始转码")
                self.dovi_run_button.setEnabled(True)
                if files: print("全部完成，输出目录: " + str(files[0].parent))
        threading.Thread(target=_run, daemon=True).start()

    # ── 杜比视界后处理 ───────────────────────────────────
    def _on_tonemap_clicked(self):
        import subprocess, threading, sys
        aligned_dir = self.tonemap_input.text().strip()
        if not aligned_dir:
            print("错误: 请输入 aligned 目录路径")
            return
        project_root = Path(__file__).parent.parent.parent.parent
        script = project_root / "tools" / "tonemap_aligned_faces.py"
        if not script.exists():
            print(f"错误: 脚本不存在: {script}")
            return
        cmd = [sys.executable, str(script), "-i", aligned_dir, "--inplace"]
        print("启动色调映射: " + " ".join(cmd))
        self.tonemap_button.setText("映射中...")
        self.tonemap_button.setEnabled(False)
        def _run():
            try:
                subprocess.run(cmd)
                print("色调映射完成")
            finally:
                self.tonemap_button.setText("杜比视界色调映射")
                self.tonemap_button.setEnabled(True)
        threading.Thread(target=_run, daemon=True).start()

    # ── 动态 tooltip 更新 ──────────────────────────────────────

    def _update_extract_tooltip(self):
        """根据当前选择的格式和模式更新提取帧按钮的提示"""
        fmt = self.image_format_combobox.currentText()
        mode = self.process_mode_extract.currentText()
        self.extract_frames_button.setToolTip(
            f"使用 FFmpeg 从视频中提取帧图片\n"
            f"格式: {fmt} | 模式: {mode}"
        )

    def _update_face_tooltip(self):
        """根据当前选择的检测/标记算法和模式更新人脸提取按钮的提示"""
        detector = self.face_detector_combobox.currentText()
        landmark = self.landmark_detector_combobox.currentText()

        # 附加说明
        detector_notes = {
            'DamoFD': ' [⚠精度偏低]',
        }
        landmark_notes = {
            'OpenSeeFace': ' [CPU only]',
            'Google-mediapipe': ' [FaceMesh ⚠左偏]',
        }

        # 每个检测器适合的检测模式（One-Stage / Sliding Window）
        detector_mode_notes = {
            'BlazeFace': ' [适合滑窗]',
            'MTCNN': ' [适合滑窗]',
            'FastFaceAlign': ' [适合One-Stage]',
            'RetinaFace_10g': ' [适合One-Stage]',
            'RetinaFace_500m': ' [适合One-Stage]',
            'DamoFD': ' [适合One-Stage]',
            'TinyMog': ' [适合One-Stage]',
            'MogFace': ' [适合One-Stage]',
            'YoloV5Face': ' [适合One-Stage]',
            'YoloV8Face': ' [适合One-Stage]',
            'YoloV11nFace': ' [适合One-Stage]',
            'CenterFace': ' [适合One-Stage]',
            'S3FD': ' [适合One-Stage]',
            'ULFD': ' [适合One-Stage]',
            'LightweightFD': ' [适合One-Stage]',
        }

        det_note = detector_notes.get(detector, '')
        lm_note = landmark_notes.get(landmark, '')
        mode_note = detector_mode_notes.get(detector, '')

        face_type = self.face_type_combobox.currentText()
        mode = self.process_mode_combobox.currentText()
        self.run_extractor_button.setToolTip(
            f"运行人脸提取\n"
            f"检测: {detector}{det_note}{mode_note} | 标记: {landmark}{lm_note} | 脸型: {face_type} | 模式: {mode}"
        )

    def _create_task_monitor(self, button, completed_description: str, interrupted_description: str = "用户强制中断了一个任务"):
        """
        创建任务监控器的工厂方法
        
        Args:
            button: 需要监控的按钮对象
            completed_description: 完成时的描述文本
            interrupted_description: 中断时的描述文本
            
        Returns:
            monitor_function: 监控函数
        """
        original_text = button.text()
        
        def monitor_process(process):
            """监控进程并更新UI"""
            try:
                process.wait()  # 等待进程结束
                # 正常完成 - 显示绿色通知
                parent_window = self.window()
                if hasattr(parent_window, 'show_task_completed_notification'):
                    parent_window.show_task_completed_notification(completed_description)
                print(f"✓ {completed_description}")
            except Exception as e:
                # 异常结束
                error_msg = str(e)
                parent_window = self.window()
                if hasattr(parent_window, 'show_task_interrupted_notification'):
                    if 'terminated' in error_msg.lower() or 'killed' in error_msg.lower():
                        parent_window.show_task_interrupted_notification(interrupted_description)
                    else:
                        parent_window.show_task_error_notification(f"任务异常结束: {error_msg}")
                print(f"✗ 任务异常: {error_msg}")
            finally:
                # 恢复按钮状态
                button.setText(original_text)
                button.setEnabled(True)
        
        return monitor_process
    def on_extract_frames_clicked(self):
        """FFmpeg 提取视频帧"""
        video_path = self.video_path_input.text().strip()
        if not video_path:
            print("错误: 请输入视频路径")
            return

        output_path = self.output_path_input.text().strip()
        if not output_path:
            print("错误: 请输入输出路径")
            return

        project_root = Path(__file__).parent.parent.parent.parent
        video_path_obj = Path(video_path)
        if not video_path_obj.is_absolute():
            video_path_obj = project_root / video_path
        if not video_path_obj.exists():
            print(f"错误: 视频文件不存在: {video_path_obj}")
            return

        output_path_obj = Path(output_path)
        if not output_path_obj.is_absolute():
            output_path_obj = project_root / output_path
        output_path_obj.mkdir(parents=True, exist_ok=True)

        # 帧数
        frame_count_str = self.frame_count_input.text().strip()
        try:
            frame_count = int(frame_count_str) if frame_count_str else 0
        except ValueError:
            frame_count = 0

        # 图片格式
        img_fmt = self.image_format_combobox.currentText()

        # FFmpeg 路径（可自定义）
        ffmpeg_dir_str = self.ffmpeg_path_input.text().strip()
        ffmpeg_dir = Path(ffmpeg_dir_str) if ffmpeg_dir_str else (project_root / "ffmpeg")
        if not ffmpeg_dir.is_absolute():
            ffmpeg_dir = project_root / ffmpeg_dir
        ffmpeg_exe = ffmpeg_dir / "ffmpeg.exe"
        if not ffmpeg_exe.exists():
            print(f"错误: FFmpeg 未找到: {ffmpeg_exe}")
            return

        # 构建命令
        output_pattern = output_path_obj / f"%06d.{img_fmt}"
        cmd = [
            str(ffmpeg_exe),
            "-i", str(video_path_obj),
            "-q:v", "2",
        ]
        if frame_count > 0:
            # -vframes 限制帧数
            cmd.extend(["-vframes", str(frame_count)])
        cmd.append(str(output_pattern))

        cmd_str = " ".join(cmd)
        print(f"\n执行命令:\n{cmd_str}\n")

        parent_window = self.window()
        count_info = f"全部帧" if frame_count <= 0 else f"{frame_count} 帧"
        if hasattr(parent_window, 'show_command_notification'):
            parent_window.show_command_notification(cmd_str, f"FFmpeg 提取帧 - {count_info} → {img_fmt}")

        try:
            monitor = self._create_task_monitor(
                self.extract_frames_button,
                f"帧提取完成 - {count_info}",
                "用户中断了帧提取任务"
            )
            self.extract_frames_button.setText("正在提取...")
            self.extract_frames_button.setEnabled(False)

            if hasattr(subprocess, 'CREATE_NEW_CONSOLE'):
                cmd_with_pause = " ".join(cmd) + " & echo. & echo 任务完成！按任意键关闭窗口... & pause"
                process = subprocess.Popen(
                    f"cmd /c {cmd_with_pause}",
                    creationflags=subprocess.CREATE_NEW_CONSOLE
                )
                threading.Thread(target=lambda p=process: monitor(p), daemon=True).start()
            else:
                process = subprocess.Popen(cmd)
                threading.Thread(target=lambda p=process: monitor(p), daemon=True).start()
            print("✓ 已启动 FFmpeg 帧提取进程")
        except Exception as e:
            print(f"✗ 启动失败: {e}")
            self.extract_frames_button.setText("提取帧")
            self.extract_frames_button.setEnabled(True)

    def on_run_extractor_clicked(self):
        """运行按钮点击事件"""
        # 获取输入路径
        input_path = self.face_input_path_input.text().strip()
        if not input_path:
            print("错误: 请输入有效的输入路径")
            return
        
        # 将相对路径转换为绝对路径（基于项目根目录）
        project_root = Path(__file__).parent.parent.parent.parent
        input_path_obj = Path(input_path)
        if not input_path_obj.is_absolute():
            input_path_obj = project_root / input_path
        
        if not input_path_obj.exists():
            print(f"错误: 输入路径不存在: {input_path_obj}")
            # 显示红色错误通知
            parent_window = self.window()
            if hasattr(parent_window, 'show_task_error_notification'):
                parent_window.show_task_error_notification("路径不存在")
            return
        
        # 获取输出路径
        output_path = self.face_output_path_input.text().strip()
        if not output_path:
            print("错误: 请输入有效的输出路径")
            return
        
        # 将相对路径转换为绝对路径（基于项目根目录）
        output_path_obj = Path(output_path)
        if not output_path_obj.is_absolute():
            output_path_obj = project_root / output_path
        
        # 获取检测器和标记器
        detector = self.face_detector_combobox.currentText()
        landmarker = self.landmark_detector_combobox.currentText()
        
        # 获取脸型类型
        face_type = self.face_type_combobox.currentText()
        
        # 获取预缩放参数
        resize_value_str = self.resize_input.text().strip()
        try:
            resize_value = int(resize_value_str) if resize_value_str else 0
            if resize_value < 0:
                resize_value = 0
        except ValueError:
            print("警告: 无效的预缩放尺寸，已重置为0（禁用）")
            resize_value = 0

        # 获取检测模式 / 缩放方式 / 尺寸
        _mode_map = {
            "One-Stage (整图缩放·快)": "one_stage",
            "Sliding Window (滑窗扫描)": "sliding_window",
        }
        input_mode = _mode_map.get(self.input_mode_combobox.currentText(), "one_stage")
        _resize_map = {
            "LetterBox (保纵横比)": "letterbox",
            "WarpAffine (拉伸·最快)": "warp",
        }
        resize_mode = _resize_map.get(self.resize_mode_combobox.currentText(), "letterbox")
        _size_str = self.input_size_input.text().strip()
        try:
            input_size = int(_size_str) if _size_str.isdigit() else 640
            if input_size < 64:
                input_size = 640
        except ValueError:
            input_size = 640
        
        # 获取处理模式
        process_mode = self.process_mode_combobox.currentText()
        is_video_batch = (process_mode == "视频目录批量")
        
        # 获取检测角度
        detection_angles_str = self.detection_angles_input.text().strip()
        if not detection_angles_str:
            detection_angles_str = "0"  # 默认为0度
        
        # 判断输入是否为视频文件
        _video_ext = {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv'}
        _is_video_input = input_path_obj.is_file() and input_path_obj.suffix.lower() in _video_ext
        extractor_script = project_root / "Extractor" / "Extractor.py"
        import sys
        python_exe = Path(sys.executable)

        if _is_video_input and not is_video_batch:
            # FFmpeg 管道模式
            ffmpeg_exe = project_root / "ffmpeg" / "ffmpeg.exe"
            if not ffmpeg_exe.exists():
                print(f"错误: FFmpeg 未找到: {ffmpeg_exe}")
                return
            _r = subprocess.run(
                [str(ffmpeg_exe), "-i", str(input_path_obj), "-vframes", "0", "-f", "null", "-"],
                capture_output=True, text=True, timeout=10)
            _dims = None
            for _line in _r.stderr.splitlines():
                if 'Stream' in _line and 'Video' in _line:
                    _m = re.search(r'(\d{3,})x(\d{3,})', _line)
                    if _m:
                        _dims = (_m.group(1), _m.group(2))
                        break
            if not _dims:
                print("错误: 无法获取视频尺寸")
                return
            _w, _h = int(_dims[0]), int(_dims[1])
            _frame_size = f"{_w}x{_h}"
            _hdr_mode = self.face_hdr_mode_combobox.currentText()
            _skip_str = self.skip_frames_input.text().strip()
            _skip = int(_skip_str) if _skip_str.isdigit() else 0
            if "精确" in _hdr_mode:
                _vf = "libplacebo=tonemapping=hable:colorspace=1:color_trc=2:color_primaries=1:range=tv:dithering=blue:format=yuv420p"
                if _skip > 0:
                    _vf = "select=not(mod(n,{}))".format(_skip+1) + "," + _vf
                _ffmpeg_cmd = [
                    str(ffmpeg_exe), "-i", str(input_path_obj),
                    "-vf", _vf, "-f", "rawvideo", "-pix_fmt", "rgb24", "-",
                ]
            else:
                _ffmpeg_cmd = [str(ffmpeg_exe), "-i", str(input_path_obj)]
                if _skip > 0:
                    _ffmpeg_cmd += ["-vf", "select=not(mod(n,{}))".format(_skip+1)]
                _ffmpeg_cmd += ["-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
            cmd = [
                str(python_exe), str(extractor_script),
                "-o", str(output_path_obj),
                "-d", detector, "-l", landmarker,
                "-t", face_type, "-a", detection_angles_str,
            ]
            if resize_value > 0:
                cmd.extend(["-r", str(resize_value)])
            if not self.kps_align_checkbox.isChecked():
                cmd.append("--no-kps-align")
            cmd.extend(["--input-mode", input_mode, "--resize-mode", resize_mode, "--input-size", str(input_size)])
            cmd.extend(["--ffmpeg-frame-size", _frame_size])
            cmd.extend(["--ffmpeg-cmd"] + _ffmpeg_cmd)
        else:
            # 传统模式
            cmd = [
                str(python_exe), str(extractor_script),
                "-i", str(input_path_obj),
                "-o", str(output_path_obj),
                "-d", detector, "-l", landmarker,
                "-t", face_type, "-a", detection_angles_str,
            ]
            if resize_value > 0:
                cmd.extend(["-r", str(resize_value)])
            skip_frames_str = self.skip_frames_input.text().strip()
            skip_frames = int(skip_frames_str) if skip_frames_str.isdigit() else 0
            if skip_frames > 0:
                cmd.extend(["--skip-frames", str(skip_frames)])
            if not self.kps_align_checkbox.isChecked():
                cmd.append("--no-kps-align")
            cmd.extend(["--input-mode", input_mode, "--resize-mode", resize_mode, "--input-size", str(input_size)])
            if is_video_batch:
                cmd.extend(["-m", "video"])
        
        # 打印命令
        cmd_str = " ".join(cmd)
        print(f"\n执行命令:\n{cmd_str}\n")
        
        # 显示通知
        parent_window = self.window()
        resize_info = f" (预缩放: {resize_value}px)" if resize_value > 0 else " (无预缩放)"
        batch_mode_info = " [视频批量模式]" if is_video_batch else ""
        angles_info = f", 角度: {detection_angles_str}" if detection_angles_str != "0" else ""
        if hasattr(parent_window, 'show_command_notification'):
            parent_window.show_command_notification(
                cmd_str,
                f"人脸提取 - {detector} + {landmarker}, 脸型: {face_type}{angles_info}{resize_info}{batch_mode_info}"
            )
        
        try:
            # 创建任务监控器（在修改按钮状态之前）
            monitor = self._create_task_monitor(
                self.run_extractor_button,
                f"人脸提取已完成 - {detector} + {landmarker}, 脸型: {face_type}{angles_info}{resize_info}{batch_mode_info}",
                "用户强制中断了人脸提取任务"
            )
            
            # 设置按钮为运行状态
            self.run_extractor_button.setText("正在运行...")
            self.run_extractor_button.setEnabled(False)
            
            # 在新窗口中运行命令，并在执行完成后等待用户按键
            if hasattr(subprocess, 'CREATE_NEW_CONSOLE'):
                # Windows 系统：使用 cmd /c 执行命令，完成后 pause 等待按键
                cmd_with_pause = " ".join(cmd) + " & echo. & echo 任务完成！按任意键关闭窗口... & pause"
                process = subprocess.Popen(
                    f"cmd /c {cmd_with_pause}",
                    creationflags=subprocess.CREATE_NEW_CONSOLE
                )
                
                # 启动监控线程（使用默认参数避免闭包问题）
                monitor_thread = threading.Thread(target=lambda p=process: monitor(p), daemon=True)
                monitor_thread.start()
            else:
                # 其他系统
                process = subprocess.Popen(cmd)
                monitor_thread = threading.Thread(target=lambda p=process: monitor(p), daemon=True)
                monitor_thread.start()
                
            print("✓ 已启动人脸提取进程")
        except Exception as e:
            print(f"✗ 启动失败: {e}")
            self.run_extractor_button.setText("开始提取人脸")
            self.run_extractor_button.setEnabled(True)
