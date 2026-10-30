from PyQt5.QtCore import QEasingCurve, QPropertyAnimation, QTimer, Qt
from PyQt5.QtWidgets import QApplication, QGraphicsOpacityEffect, QSizePolicy, QBoxLayout, QLabel
import subprocess
import threading
from contextlib import contextmanager
from pathlib import Path
from siui.components import SiDenseHContainer, SiDenseVContainer, SiTitledWidgetGroup
from siui.components.button import SiPushButtonRefactor
from siui.components.combobox_ import SiCapsuleComboBox
from siui.components.container import SiTriSectionPanelCard
from siui.components.editbox import SiLabeledLineEdit
from siui.components.page import SiPage
from siui.components.widgets import SiCheckBox, SiLabel
from siui.core import SiExpAnimation
@contextmanager
def createPanelCard(parent: SiTitledWidgetGroup, title: str) -> SiTriSectionPanelCard:
    """创建面板卡片的上下文管理器"""
    card = SiTriSectionPanelCard(parent)
    card.setTitle(title)
    # 设置卡片为自适应大小
    card.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.MinimumExpanding)
    try:
        yield card
    finally:
        # 确保内容完全添加后再调整大小
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
    # 设置容器为自适应大小，允许垂直扩展
    container.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.MinimumExpanding)
    try:
        yield container
    finally:
        parent.addWidget(container, side)
class DataProcessingPage(SiPage):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setPadding(64)
        self.setScrollMaximumWidth(1000)
        self.setScrollAlignment(Qt.AlignLeft)
        self.setTitle("数据处理")
        
        # 初始化卡片高度动画字典
        self.card_height_animations = {}
        # 创建滚动容器
        self.titled_widgets_group = SiTitledWidgetGroup(self)
        self.titled_widgets_group.setSpacing(32)
        self.titled_widgets_group.setAdjustWidgetsSize(True)
        # 第一组：Analyzer - 元数据分析
        with self.titled_widgets_group as group:
            group.addTitle("元数据分析器 (Analyzer)")
            with createPanelCard(group, "分析设置") as card:
                with createDenseContainer(card.body(), QBoxLayout.TopToBottom) as container:
                    # 输入路径输入框
                    self.analyzer_input_path_input = SiLabeledLineEdit(self)
                    self.analyzer_input_path_input.setTitle("人脸集路径")
                    self.analyzer_input_path_input.setPlaceholderText("请输入对齐后的人脸集文件夹路径...")
                    # 使用相对于项目根目录的路径
                    project_root = Path(__file__).parent.parent.parent.parent
                    default_input = project_root / "workspace" / "data_dst" / "aligned"
                    self.analyzer_input_path_input.setText(str(default_input))
                    self.analyzer_input_path_input.resize(700, 64)
                    container.addWidget(self.analyzer_input_path_input)
                    
                    # 特征选择区域（使用 siui 容器）
                    self.feature_selection_container = SiDenseVContainer(self)
                    self.feature_selection_container.setSpacing(12)
                    self.feature_selection_container.setAlignment(Qt.AlignLeft)
                    self.feature_selection_container.setMinimumHeight(350)  # 设置最小高度以容纳所有复选框和按钮
                    
                    # 标题标签
                    self.feature_title_label = SiLabel(self)
                    self.feature_title_label.setText("选择分析特征")
                    self.feature_title_label.setStyleSheet("font-size: 14px; font-weight: bold; color: #FFFFFF;")
                    self.feature_selection_container.addWidget(self.feature_title_label)
                    
                    # 创建特征复选框（使用 siui 的 SiCheckBox）
                    self.feature_checkboxes = {}
                    feature_options = [
                        ("phash", "感知哈希 (phash) - 用于重复检测"),
                        ("histogram", "直方图分布 (histogram) - RGB和HSV统计"),
                        ("hue", "色相分析 (hue) - 颜色分布统计"),
                        ("pose", "人脸姿态 (pose) - 需要landmarks"),
                        ("embedding", "人脸Embedding (embedding) - ArcFace特征向量"),
                        ("landmark", "人脸特征点 (landmark) - InsightFace 106pt检测"),
                    ]
                    
                    for feature_key, feature_desc in feature_options:
                        checkbox = SiCheckBox(self)
                        checkbox.setText(feature_desc)
                        checkbox.setChecked(True)  # 默认全选
                        checkbox.resize(400, 32)
                        checkbox.setToolTip(f"启用 {feature_key} 特征分析")
                        self.feature_checkboxes[feature_key] = checkbox
                        self.feature_selection_container.addWidget(checkbox)
                    
                    # 全选/取消全选按钮
                    with createDenseContainer(container, QBoxLayout.LeftToRight) as btn_row:
                        self.select_all_button = SiPushButtonRefactor(self)
                        self.select_all_button.setText("全选")
                        self.select_all_button.resize(120, 36)
                        self.select_all_button.setToolTip("勾选所有分析特征")
                        self.select_all_button.clicked.connect(self.on_select_all_features)

                        self.deselect_all_button = SiPushButtonRefactor(self)
                        self.deselect_all_button.setText("取消全选")
                        self.deselect_all_button.resize(120, 36)
                        self.deselect_all_button.setToolTip("取消勾选所有分析特征")
                        self.deselect_all_button.clicked.connect(self.on_deselect_all_features)
                        
                        btn_row.addWidget(self.select_all_button)
                        btn_row.addWidget(self.deselect_all_button)
                    
                    self.feature_selection_container.addWidget(btn_row)
                    container.addWidget(self.feature_selection_container)
            # 运行控制卡片
            with createPanelCard(group, "运行控制") as card:
                with createDenseContainer(card.body(), QBoxLayout.LeftToRight) as container:
                    # 强制重新分析开关（暂时用按钮代替）
                    self.analyzer_force_button = SiPushButtonRefactor(self)
                    self.analyzer_force_button.setText("强制重新分析")
                    self.analyzer_force_button.resize(180, 48)
                    self.analyzer_force_button.setToolTip("覆盖已有数据，重新计算所有特征")
                    self.analyzer_force_button.clicked.connect(lambda: self.on_run_analyzer_clicked(force_reanalyze=True))  # 绑定到相同的处理函数，但传递force=True
                    
                    # 写回源文件按钮
                    self.write_back_button = SiPushButtonRefactor(self)
                    self.write_back_button.setText("写回源文件")
                    self.write_back_button.resize(180, 48)
                    self.write_back_button.setToolTip("用于兼容原版DeepFaceLab，虽然我不知道为什么你还要用回老版本")
                    self.write_back_button.clicked.connect(self.on_write_back_clicked)
                    
                    # 运行按钮
                    self.run_analyzer_button = SiPushButtonRefactor(self)
                    self.run_analyzer_button.setText("开始分析")
                    self.run_analyzer_button.resize(180, 48)
                    self.run_analyzer_button.setToolTip("点击开始执行元数据分析任务")
                    self.run_analyzer_button.clicked.connect(self.on_run_analyzer_clicked)
                    
                    container.addWidget(self.analyzer_force_button)
                    container.addWidget(self.write_back_button)
                    container.addWidget(self.run_analyzer_button)
        # 第二组：Filter - 人脸集过滤
        with self.titled_widgets_group as group:
            group.addTitle("人脸集过滤器 (Filter)")
            with createPanelCard(group, "过滤设置") as card:
                with createDenseContainer(card.body(), QBoxLayout.TopToBottom) as container:
                    container.layout().setSpacing(16)  # 增加行间距
                    
                    # 输入路径输入框
                    self.filter_input_path_input = SiLabeledLineEdit(self)
                    self.filter_input_path_input.setTitle("人脸集路径")
                    self.filter_input_path_input.setPlaceholderText("请输入对齐后的人脸集文件夹路径...")
                    # 使用相对于项目根目录的路径
                    project_root = Path(__file__).parent.parent.parent.parent
                    default_input = project_root / "workspace" / "data_dst" / "aligned"
                    self.filter_input_path_input.setText(str(default_input))
                    self.filter_input_path_input.resize(700, 64)
                    container.addWidget(self.filter_input_path_input)
            # 过滤模式和控制卡片
            with createPanelCard(group, "过滤模式") as card:
                with createDenseContainer(card.body(), QBoxLayout.TopToBottom) as container:
                    # 第一行：过滤模式选择
                    with createDenseContainer(container, QBoxLayout.LeftToRight) as row1:
                        self.filter_mode_combobox = SiCapsuleComboBox(self)
                        self.filter_mode_combobox.setTitle("过滤模式")
                        self.filter_mode_combobox.setMinimumHeight(36)
                        self.filter_mode_combobox.setEditable(False)
                        self.filter_mode_combobox.addItems([
                            "blur",         # 模糊过滤（拉普拉斯方差）
                            "faceid",       # 人脸ID分组
                            "position",     # 位置聚类分组
                            "repeated"      # 重复图片去重
                        ])
                        self.filter_mode_combobox.setCurrentText("blur")
                        self.filter_mode_combobox.setToolTip(
                            "选择过滤模式\n\n"
                            "blur — 模糊过滤（拉普拉斯方差）\n"
                            "  剔除模糊的人脸图片，保留清晰的样本。\n"
                            "  使用拉普拉斯算子计算图像边缘清晰度，值越低越模糊。\n\n"
                            "faceid — 人脸ID分组\n"
                            "  基于ArcFace特征向量聚类，按人脸身份分组。\n"
                            "  同一组≈同一个人，方便筛选特定人物。\n\n"
                            "position — 位置聚类分组\n"
                            "  基于人脸在原始视频中的位置聚类。\n"
                            "  同一位置区域的人脸归为一组，适合场景筛选。\n\n"
                            "repeated — 重复图片去重\n"
                            "  使用感知哈希(phash)检测并剔除高度相似的重复图片。\n"
                            "  适合清理连续帧导出的几乎相同的脸。"
                        )
                        self.filter_mode_combobox.currentTextChanged.connect(self.on_filter_mode_changed)
                        self.filter_mode_combobox.currentTextChanged.connect(self._update_filter_tooltip)

                        row1.addWidget(self.filter_mode_combobox)
                    # 第二行：位置过滤参数（仅 position 模式显示）
                    with createDenseContainer(container, QBoxLayout.LeftToRight) as row_params:
                        self.filter_position_eps_input = SiLabeledLineEdit(self)
                        self.filter_position_eps_input.setTitle("邻域半径 (像素)")
                        self.filter_position_eps_input.setPlaceholderText("默认: 50")
                        self.filter_position_eps_input.setText("50")
                        self.filter_position_eps_input.resize(180, 48)
                        self.filter_position_eps_input.setToolTip(
                            "DBSCAN 聚类半径（单位：像素）\n\n"
                            "含义：两张人脸在原始画面中的位置距离 ≤ 此值时视为同一位置。\n"
                            "值越小 → 分组越细，只有很近的脸被归为一组。\n"
                            "值越大 → 分组越粗，同半张画面上的人都归为一组。\n\n"
                            "建议：标准视频（1920×1080）用 50-100，小尺寸视频适当减小。\n"
                            "如果你只想要正中间的脸，设 30；如果整段视频都在拍同一个人，可设 200+。"
                        )
                        self.filter_position_eps_input.setVisible(False)
                        
                        self.filter_position_min_samples_input = SiLabeledLineEdit(self)
                        self.filter_position_min_samples_input.setTitle("最小样本数")
                        self.filter_position_min_samples_input.setPlaceholderText("默认: 2")
                        self.filter_position_min_samples_input.setText("2")
                        self.filter_position_min_samples_input.resize(150, 48)
                        self.filter_position_min_samples_input.setToolTip(
                            "形成一组所需的最少人脸数量\n\n"
                            "含义：同一个位置区域内，至少有N张人脸才算一组。\n"
                            "值越小 → 越容易成组，少量出现也被保留。\n"
                            "值越大 → 需要更多同位置的人脸才保留该组。\n\n"
                            "示例：设 2 → 某位置只有1张人脸时被当成离群点剔除。\n"
                            "设 1 → 所有人的脸都会保留（不过滤位置）。\n\n"
                            "建议：通常 2-3，只保留出现2次以上的位置。"
                        )
                        self.filter_position_min_samples_input.setVisible(False)
                        
                        row_params.addWidget(self.filter_position_eps_input)
                        row_params.addWidget(self.filter_position_min_samples_input)
                    # 第三行：重复去重参数（仅 repeated 模式显示）
                    with createDenseContainer(container, QBoxLayout.LeftToRight) as row_repeated_params:
                        self.filter_repeated_threshold_input = SiLabeledLineEdit(self)
                        self.filter_repeated_threshold_input.setTitle("相似度阈值")
                        self.filter_repeated_threshold_input.setPlaceholderText("默认: 0.98")
                        self.filter_repeated_threshold_input.setText("0.98")
                        self.filter_repeated_threshold_input.resize(180, 48)
                        self.filter_repeated_threshold_input.setToolTip(
                            "感知哈希相似度阈值（0.0 ~ 1.0）\n\n"
                            "含义：两张人脸的phash对比，相似度 ≥ 此值时视为重复。\n"
                            "1.0 = 完全一样（像素级相同）\n"
                            "0.98 = 几乎一样（微小的光影/表情差异）\n"
                            "0.95 = 相似但不同帧（同角度但轻微变化）\n"
                            "0.8  = 同一个人不同表情\n\n"
                            "建议：\n"
                            "  - 0.99：仅剔除完全相同的帧（最保守）\n"
                            "  - 0.98（默认）：剔除连续帧中的高度相似图片\n"
                            "  - 0.95：更激进地去重，会丢失一些表情变化\n\n"
                            "工作原理: 使用感知哈希(phash)计算汉明距离后归一化。"
                        )
                        self.filter_repeated_threshold_input.setVisible(False)
                        
                        row_repeated_params.addWidget(self.filter_repeated_threshold_input)
                    # 第四行：人脸ID过滤参数（仅 faceid 模式显示）
                    with createDenseContainer(container, QBoxLayout.LeftToRight) as row_faceid_params:
                        self.filter_faceid_eps_input = SiLabeledLineEdit(self)
                        self.filter_faceid_eps_input.setTitle("聚类半径 (eps)")
                        self.filter_faceid_eps_input.setPlaceholderText("默认: 0.31415926")
                        self.filter_faceid_eps_input.setText("0.31415926")
                        self.filter_faceid_eps_input.resize(180, 48)
                        self.filter_faceid_eps_input.setToolTip(
                            "DBSCAN 聚类半径（余弦距离空间）\n\n"
                            "含义：ArcFace特征向量之间的距离阈值。\n"
                            "值越小 → 分组越严格，只有非常相似的人脸才归为一组。\n"
                            "值越大 → 分组越宽松，同个人不同表情/角度也能归为一组。\n\n"
                            "参考值：\n"
                            "  - 0.1：极严格，只有高度相似的同视角同表情\n"
                            "  - 0.114514（默认）：对大多数模型效果较好\n"
                            "  - 0.2：宽松，同一个人不同表情能分到一起\n"
                            "  - 0.3+：非常宽松，可能把不同人混在一起\n\n"
                            "建议：从 0.114514 开始，根据输出调整。"
                        )
                        self.filter_faceid_eps_input.setVisible(False)

                        self.filter_faceid_min_samples_input = SiLabeledLineEdit(self)
                        self.filter_faceid_min_samples_input.setTitle("最小样本数")
                        self.filter_faceid_min_samples_input.setPlaceholderText("默认: 2")
                        self.filter_faceid_min_samples_input.setText("2")
                        self.filter_faceid_min_samples_input.resize(150, 48)
                        self.filter_faceid_min_samples_input.setToolTip(
                            "形成一组所需的最少人脸数量\n\n"
                            "含义：同一个人（按特征向量聚类）至少有N张人脸才算有效组。\n\n"
                            "示例：\n"
                            "  - 设 2 → 某个只出现了1次的人脸被当作离群点剔除\n"
                            "  - 设 1 → 保留所有人的脸（不丢弃任何身份）\n"
                            "  - 设 5 → 只保留出现5次以上的人物\n\n"
                            "建议：\n"
                            "  - 主要人物用 2-3（默认2）\n"
                            "  - 如果你只想要主要角色，设 5-10 来过滤掉路人"
                        )
                        self.filter_faceid_min_samples_input.setVisible(False)
                        
                        row_faceid_params.addWidget(self.filter_faceid_eps_input)
                        row_faceid_params.addWidget(self.filter_faceid_min_samples_input)
                    # 第五行：运行按钮 + 合并按钮
                    with createDenseContainer(container, QBoxLayout.LeftToRight) as row2:
                        self.run_filter_button = SiPushButtonRefactor(self)
                        self.run_filter_button.setText("开始过滤")
                        self.run_filter_button.resize(180, 48)
                        self.run_filter_button.setToolTip(
                            "运行人脸集过滤\n\n"
                            "根据当前选择的过滤模式处理 aligned 目录中的人脸图片：\n"
                            "- blur：剔除模糊图片，保留清晰样本\n"
                            "- faceid：按人脸身份分组，同一组 = 同一个人\n"
                            "- position：按人脸在画面中的位置分组\n"
                            "- repeated：剔除高度相似的重复图片\n\n"
                            "处理后的图片会按组分到子文件夹中，方便筛选。"
                        )
                        self.run_filter_button.clicked.connect(self.on_run_filter_clicked)

                        self.merge_subfolders_button = SiPushButtonRefactor(self)
                        self.merge_subfolders_button.setText("合并子文件夹")
                        self.merge_subfolders_button.resize(180, 48)
                        self.merge_subfolders_button.setToolTip(
                            "将所有一级子文件夹中的图片合并到 aligned 目录\n\n"
                            "用途：过滤后的图片被分到了子文件夹中，\n"
                            "使用此功能将它们全部移回 aligned 根目录。\n\n"
                            "安全机制：\n"
                            "- 只遍历一级子文件夹（不递归）\n"
                            "- 不会删除任何原始文件\n"
                            "- 不会操作系统根目录\n\n"
                            "注意：如果有同名文件，会自动重命名避免覆盖。"
                        )
                        self.merge_subfolders_button.clicked.connect(self.on_merge_subfolders_clicked)
                        
                        row2.addWidget(self.run_filter_button)
                        row2.addWidget(self.merge_subfolders_button)
        # 第三组：Sorter - 智能排序
        with self.titled_widgets_group as group:
            group.addTitle("智能排序器 (Sorter)")
            with createPanelCard(group, "排序设置") as card:
                with createDenseContainer(card.body(), QBoxLayout.TopToBottom) as container:
                    # 输入路径输入框
                    self.sorter_input_path_input = SiLabeledLineEdit(self)
                    self.sorter_input_path_input.setTitle("人脸集路径")
                    self.sorter_input_path_input.setPlaceholderText("请输入对齐后的人脸集文件夹路径...")
                    # 使用相对于项目根目录的路径
                    project_root = Path(__file__).parent.parent.parent.parent
                    default_input = project_root / "workspace" / "data_dst" / "aligned"
                    self.sorter_input_path_input.setText(str(default_input))
                    self.sorter_input_path_input.resize(700, 64)
                    container.addWidget(self.sorter_input_path_input)
            # 排序模式和控制卡片
            with createPanelCard(group, "排序模式") as card:
                with createDenseContainer(card.body(), QBoxLayout.TopToBottom) as container:
                    # 第一行：排序方式选择 + 姿态类型（同一行）
                    with createDenseContainer(container, QBoxLayout.LeftToRight) as row1:
                        self.sorter_method_combobox = SiCapsuleComboBox(self)
                        self.sorter_method_combobox.setTitle("排序方式")
                        self.sorter_method_combobox.setMinimumHeight(36)
                        self.sorter_method_combobox.setEditable(False)
                        self.sorter_method_combobox.addItems([
                            "phash",              # 感知哈希排序
                            "hist",               # 直方图排序
                            "blur",               # 模糊度排序
                            "face_pose",          # 人脸姿态排序
                            "resolution",         # 分辨率排序
                            "color",              # 颜色排序
                            "name",               # 文件名排序
                        ])
                        self.sorter_method_combobox.setCurrentText("phash")
                        self.sorter_method_combobox.setToolTip(
                            "选择排序方式\n\n"
                            "phash — 感知哈希排序\n"
                            "  基于图片的感知哈希值排序，相似的排在一起。\n"
                            "  适合：浏览数据集时把重复/相似图片放一起方便对比删除。\n\n"
                            "hist — 直方图排序\n"
                            "  基于RGB直方图分布排序，亮度和色调相似的排一起。\n"
                            "  适合：按光照条件整理训练数据。\n\n"
                            "blur — 模糊度排序\n"
                            "  按拉普拉斯方差（清晰度得分）从清晰到模糊排序。\n"
                            "  适合：快速找出并剔除模糊样本。\n\n"
                            "face_pose — 人脸姿态排序\n"
                            "  按人脸朝向角度排序（pitch俯仰/yaw偏航/roll翻滚）。\n"
                            "  适合：按脸部角度整理数据，确保角度多样性。\n\n"
                            "resolution — 分辨率排序\n"
                            "  按人脸图片的分辨率从高到低排序。\n"
                            "  适合：保留高分辨率样本，剔除低分辨率。\n\n"
                            "color — 颜色排序\n"
                            "  按平均色调/饱和度排序，颜色相近的排一起。\n"
                            "  适合：按肤色或衣着颜色分组。\n\n"
                            "name — 文件名排序\n"
                            "  按文件名（数字/字母顺序）排序。\n"
                            "  适合：恢复原始文件顺序。"
                        )
                        self.sorter_method_combobox.currentTextChanged.connect(self.on_sorter_method_changed)
                        self.sorter_method_combobox.currentTextChanged.connect(self._update_sorter_tooltip)

                        # 姿态类型选择（仅 face_pose 时显示）
                        self.sorter_pose_type_combobox = SiCapsuleComboBox(self)
                        self.sorter_pose_type_combobox.setTitle("姿态类型")
                        self.sorter_pose_type_combobox.setMinimumHeight(36)
                        self.sorter_pose_type_combobox.setEditable(False)
                        self.sorter_pose_type_combobox.addItems(["pitch", "yaw", "roll"])
                        self.sorter_pose_type_combobox.setCurrentText("yaw")
                        self.sorter_pose_type_combobox.setToolTip(
                            "选择人脸姿态角度类型（仅用于 face_pose 排序）\n\n"
                            "pitch — 俯仰角（点头）\n"
                            "  头部上下转动，范围约 -90° ~ +90°。\n"
                            "  0° = 正面，正值 = 低头，负值 = 抬头。\n\n"
                            "yaw — 偏航角（摇头）\n"
                            "  头部左右转动，范围约 -90° ~ +90°。\n"
                            "  0° = 正面，正值 = 向右转，负值 = 向左转。\n"
                            "  推荐：最常用的角度，适合筛选不同侧脸角度。\n\n"
                            "roll — 翻滚角（歪头）\n"
                            "  头部左右倾斜，范围约 -90° ~ +90°。\n"
                            "  0° = 竖直，正值 = 向右歪，负值 = 向左歪。"
                        )
                        self.sorter_pose_type_combobox.setVisible(False)

                        row1.addWidget(self.sorter_method_combobox)
                        row1.addWidget(self.sorter_pose_type_combobox)
                    # 第二行：运动模糊检测开关（仅 blur 模式显示）
                    self.sorter_motion_blur_checkbox = SiCheckBox(self)
                    self.sorter_motion_blur_checkbox.setText("使用运动模糊检测")
                    self.sorter_motion_blur_checkbox.setMinimumHeight(36)
                    self.sorter_motion_blur_checkbox.setChecked(False)
                    self.sorter_motion_blur_checkbox.setToolTip(
                        "启用运动模糊检测（仅用于 blur 排序）\n\n"
                        "常规 blur 排序使用拉普拉斯方差检测静态模糊（对焦不准）。\n"
                        "启用此选项后额外检测运动模糊（物体/镜头移动导致的拖影）。\n\n"
                        "两种模糊的区别：\n"
                        "- 静态模糊：相机没对准焦，整张图软糊\n"
                        "- 运动模糊：有方向性的拖影，边缘出现条纹\n\n"
                        "建议：如果你主要想筛掉慢快门导致的运动模糊，开启此选项。"
                    )
                    self.sorter_motion_blur_checkbox.setVisible(False)
                    container.addWidget(self.sorter_motion_blur_checkbox)
                    # 第三行：运行按钮
                    with createDenseContainer(container, QBoxLayout.LeftToRight) as row2:
                        self.run_sorter_button = SiPushButtonRefactor(self)
                        self.run_sorter_button.setText("开始排序")
                        self.run_sorter_button.resize(180, 48)
                        self.run_sorter_button.setToolTip(
                            "运行人脸集排序\n\n"
                            "根据当前选择的排序方式，对 aligned 目录中的人脸图片进行排序：\n"
                            "- phash：按相似度分组\n"
                            "- hist：按直方图分布\n"
                            "- blur：按清晰度从高到低\n"
                            "- face_pose：按人脸朝向角度\n"
                            "- resolution：按分辨率\n"
                            "- color：按颜色相似度\n"
                            "- name：按文件名\n\n"
                            "排序完成后，图片会被重命名（sorted_001, sorted_002...），\n"
                            "方便在文件管理器中按顺序浏览。"
                        )
                        self.run_sorter_button.clicked.connect(self.on_run_sorter_clicked)
                        
                        row2.addWidget(self.run_sorter_button)
            # 初始化时确保控件状态正确（不触发动画）
            self.sorter_pose_type_combobox.setVisible(False)
            self.sorter_motion_blur_checkbox.setVisible(False)
        # 第四组：PAK 解包
        with self.titled_widgets_group as group:
            group.addTitle("PAK 文件解包")
            with createPanelCard(group, "解包设置") as card:
                with createDenseContainer(card.body(), QBoxLayout.TopToBottom) as container:
                    project_root = Path(__file__).parent.parent.parent.parent
                    self.pak_input_path = SiLabeledLineEdit(self)
                    self.pak_input_path.setTitle("人脸集路径（包含 faceset.pak）")
                    self.pak_input_path.setPlaceholderText("选择包含 faceset.pak 的目录...")
                    default_pak = str(project_root / "workspace" / "data_dst" / "aligned")
                    self.pak_input_path.setText(default_pak)
                    self.pak_input_path.resize(700, 64)
                    self.pak_input_path.setToolTip(
                        "包含 faceset.pak 文件的目录路径\n\n"
                        "例如: workspace/data_dst/aligned/faceset.pak\n"
                        "只需输入目录路径，工具会自动查找其中的 faceset.pak"
                    )
                    container.addWidget(self.pak_input_path)
                    self.pak_output_path = SiLabeledLineEdit(self)
                    self.pak_output_path.setTitle("输出目录（留空 = 与 pak 同目录）")
                    self.pak_output_path.setPlaceholderText("留空自动与 pak 文件同目录")
                    self.pak_output_path.resize(700, 64)
                    self.pak_output_path.setToolTip(
                        "解包后的图片输出目录\n\n"
                        "留空：图片输出到 faceset.pak 所在目录\n"
                        "指定：图片输出到指定目录\n\n"
                        "如果 pak 中有人物名称信息，会自动按人物分到子目录"
                    )
                    container.addWidget(self.pak_output_path)
                    with createDenseContainer(container, QBoxLayout.LeftToRight) as btn_row:
                        self.pak_unpack_btn = SiPushButtonRefactor(self)
                        self.pak_unpack_btn.setText("开始解包")
                        self.pak_unpack_btn.resize(180, 48)
                        self.pak_unpack_btn.setToolTip(
                            "将 faceset.pak 解包为原始图片\n\n"
                            "流程：\n"
                            "1. 读取 faceset.pak 文件\n"
                            "2. 解析包头和每张图片的配置信息\n"
                            "3. 提取所有图片到输出目录\n\n"
                            "支持：\n"
                            "- 按人物名称分目录存放\n"
                            "- 自动跳过已存在的文件\n\n"
                            "提示：解包后原 .pak 文件不会被删除"
                        )
                        self.pak_unpack_btn.clicked.connect(self._on_pak_unpack)
                        btn_row.addWidget(self.pak_unpack_btn)
        # 第五组：数据可视化
        with self.titled_widgets_group as group:
            group.addTitle("数据可视化")
            with createPanelCard(group, "角度分布") as card:
                with createDenseContainer(card.body(), QBoxLayout.LeftToRight) as container:
                    container.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
                    container.layout().addStretch()
                    self.viz_button = SiPushButtonRefactor(self)
                    self.viz_button.setText("启动可视化")
                    self.viz_button.resize(160, 48)
                    self.viz_button.setToolTip(
                        "启动角度分布可视化服务器，浏览器访问查看散点图\n\n"
                        "支持：\n"
                        "- 热力图 / 点图切换\n"
                        "- 手动输入目录路径\n"
                        "- 鼠标悬停查看文件名"
                    )
                    self.viz_button.clicked.connect(self._on_viz_launch)
                    container.addWidget(self.viz_button)
        # 添加页脚空白
        self.titled_widgets_group.addPlaceholder(64)
        # 设置为页面对象
        self.setAttachment(self.titled_widgets_group)

        # 初始化 tooltip（让默认选项也有详细注释）
        self._update_filter_tooltip()
        self._update_sorter_tooltip()

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
    def on_select_all_features(self):
        """全选所有特征"""
        for checkbox in self.feature_checkboxes.values():
            checkbox.setChecked(True)
    
    def on_deselect_all_features(self):
        """取消全选所有特征"""
        for checkbox in self.feature_checkboxes.values():
            checkbox.setChecked(False)
    
    def get_selected_features(self):
        """获取用户选择的特征列表"""
        selected = [key for key, checkbox in self.feature_checkboxes.items() if checkbox.isChecked()]
        return selected
    
    # ── 动态 tooltip 更新 ──────────────────────────────────────

    def _update_filter_tooltip(self):
        """根据当前过滤模式更新过滤按钮提示"""
        mode = self.filter_mode_combobox.currentText()
        tips = {
            "blur": (
                "运行人脸集过滤 — 模糊过滤\n\n"
                "剔除模糊的人脸图片（拉普拉斯方差法）：\n"
                "- 工作原理：对每张人脸计算拉普拉斯算子的方差\n"
                "- 方差越低 → 图片越模糊 → 被剔除\n"
                "- 方差越高 → 图片越清晰 → 被保留\n\n"
                "使用场景：\n"
                "- 清理运动模糊、失焦的样本\n"
                "- 提升训练数据整体质量\n\n"
                "参数：无额外参数，自动计算最佳阈值。"
            ),
            "faceid": (
                "运行人脸集过滤 — 人脸ID分组\n\n"
                "基于 ArcFace 特征向量聚类，按人脸身份分组：\n"
                "- 每个人物的脸被分到一个独立子文件夹\n"
                "- 出现次数少的离群点被剔除\n\n"
                "使用场景：\n"
                "- 从多人视频中筛选特定人物\n"
                "- 分离src和dst中不小心混入的不同人脸\n\n"
                "参数：\n"
                "- 聚类半径(eps)：值越小分组越严格\n"
                "- 最小样本数：同一人至少出现N次才保留"
            ),
            "position": (
                "运行人脸集过滤 — 位置聚类分组\n\n"
                "基于人脸在原始画面中的坐标位置聚类：\n"
                "- 同一区域的人脸被归为一组\n"
                "- 位置偏移大的离群点被剔除\n\n"
                "使用场景：\n"
                "- 画面中有固定位置的人物（如新闻主播）\n"
                "- 裁剪画面边缘偶尔出现的路人\n\n"
                "参数：\n"
                "- 邻域半径：位置距离多近算「同一位置」\n"
                "- 最小样本数：同一位置至少出现N次才保留"
            ),
            "repeated": (
                "运行人脸集过滤 — 重复图片去重\n\n"
                "使用感知哈希(phash)检测高度相似的重复图片：\n"
                "- 对比每张人脸的phash指纹\n"
                "- 相似度高于阈值的视为重复，仅保留一张\n\n"
                "使用场景：\n"
                "- 从连续视频帧中清理几乎相同的脸\n"
                "- 减少训练数据中的冗余样本\n\n"
                "参数：\n"
                "- 相似度阈值：0.98 = 几乎一样才去重，0.95 = 更激进去重"
            ),
        }
        desc = tips.get(mode, f"运行人脸集过滤\n当前模式: {mode}")
        self.run_filter_button.setToolTip(desc)

    def _update_sorter_tooltip(self):
        """根据当前排序方式更新排序按钮提示"""
        method = self.sorter_method_combobox.currentText()
        tips = {
            "phash": (
                "运行人脸集排序 — 感知哈希排序\n\n"
                "基于图片的感知哈希(phash)值排序。\n"
                "工作原理：将每张图片缩小 → 灰度化 → 计算DCT → 取低频 → 生成64位指纹。\n"
                "然后按指纹的数值排序，相似的图片会排在一起。\n\n"
                "适合场景：\n"
                "- 快速浏览数据集，方便对比删除重复的图片\n"
                "- 检查不同帧之间的人脸是否有较大变化\n\n"
                "注意：排序后文件会被重命名为 sorted_001, sorted_002..."
            ),
            "hist": (
                "运行人脸集排序 — 直方图排序\n\n"
                "基于RGB三通道直方图分布排序。\n"
                "工作原理：统计每张图片的RGB像素分布 → 计算直方图 → 按直方图相似度排序。\n\n"
                "适合场景：\n"
                "- 按光照条件整理训练数据\n"
                "- 将亮度/色调相似的图片排在一起方便对比\n\n"
                "注意：对黑白/灰度图片效果有限。"
            ),
            "blur": (
                "运行人脸集排序 — 模糊度排序\n\n"
                "基于拉普拉斯算子的方差检测图像清晰度，从清晰到模糊排序。\n"
                "得分越高 → 图片越清晰（边缘锐利）\n"
                "得分越低 → 图片越模糊（对焦不准/运动模糊）\n\n"
                "适合场景：\n"
                "- 快速找出模糊样本并剔除\n"
                "- 清洗训练数据集中的低质量图片\n\n"
                "可选：开启「运动模糊检测」可额外识别有方向性拖影的图片。"
            ),
            "face_pose": (
                "运行人脸集排序 — 人脸姿态排序\n\n"
                "基于人脸关键点检测，计算头部在三维空间中的朝向角度。\n"
                "可以选择按 pitch（俯仰）/ yaw（偏航）/ roll（翻滚）排序。\n\n"
                "适合场景：\n"
                "- 按脸部角度整理训练数据\n"
                "- 确保数据集包含多样化的角度\n"
                "- 找出极端角度（大侧脸、俯视等）\n\n"
                "当前姿态类型：可在下拉框中切换。"
            ),
            "resolution": (
                "运行人脸集排序 — 分辨率排序\n\n"
                "按人脸图片的像素分辨率从高到低排序。\n"
                "工作原理：读取每张图片的尺寸（宽×高），按面积大小排序。\n\n"
                "适合场景：\n"
                "- 优先查看高分辨率样本\n"
                "- 找出分辨率异常的图片\n"
                "- 检查对齐后的人脸尺寸是否一致"
            ),
            "color": (
                "运行人脸集排序 — 颜色排序\n\n"
                "基于图片的平均色调和饱和度进行排序。\n"
                "颜色相近的图片会被排在一起。\n\n"
                "适合场景：\n"
                "- 按肤色或衣着颜色分组\n"
                "- 将有特殊色调的图片分离出来\n"
                "- 检查色彩迁移效果"
            ),
            "name": (
                "运行人脸集排序 — 文件名排序\n\n"
                "按文件名字母/数字顺序排序。\n\n"
                "适合场景：\n"
                "- 恢复文件的原始顺序\n"
                "- 配合其他排序方式使用（先按某种方式排完，再用此方式回到文件顺序）\n\n"
                "注意：文件名排序后，文件仍会被重命名为 sorted_001...\n"
                "原始文件名会在元数据中保留。"
            ),
        }
        desc = tips.get(method, f"运行人脸集排序\n当前方式: {method}")
        self.run_sorter_button.setToolTip(desc)

    def fade_widget_in(self, widget, duration: int = 200):
        """
        淡入显示控件
        """
                        
        widget.setVisible(True)
        opacity_effect = QGraphicsOpacityEffect(widget)
        widget.setGraphicsEffect(opacity_effect)
        
        animation = QPropertyAnimation(opacity_effect, b"opacity")
        animation.setDuration(duration)
        animation.setStartValue(0.0)
        animation.setEndValue(1.0)
        animation.setEasingCurve(QEasingCurve.InOutQuad)
        animation.start()
        
        # 保存引用防止被回收
        if not hasattr(self, '_fade_animations'):
            self._fade_animations = []
        self._fade_animations.append(animation)
    
    def fade_widget_out(self, widget, duration: int = 200):
        """
        淡出隐藏控件
        """
                        
        opacity_effect = widget.graphicsEffect()
        if opacity_effect is None:
            opacity_effect = QGraphicsOpacityEffect(widget)
            widget.setGraphicsEffect(opacity_effect)
        
        animation = QPropertyAnimation(opacity_effect, b"opacity")
        animation.setDuration(duration)
        animation.setStartValue(1.0)
        animation.setEndValue(0.0)
        animation.setEasingCurve(QEasingCurve.InOutQuad)
        
        # 动画结束后隐藏
        animation.finished.connect(lambda: widget.setVisible(False))
        animation.start()
        
        # 保存引用
        if not hasattr(self, '_fade_animations'):
            self._fade_animations = []
        self._fade_animations.append(animation)
    
    def on_sorter_method_changed(self, method: str):
        """排序方式改变时的处理"""
        # 根据排序方法显示/隐藏相关控件（带动画）
        if method == "face_pose":
            # 显示姿态类型选择（同一行，不触发动画）
            self.fade_widget_in(self.sorter_pose_type_combobox)
            self.fade_widget_out(self.sorter_motion_blur_checkbox)
        elif method == "blur":
            # 显示运动模糊选项（第二行，需要触发高度动画）
            self.fade_widget_out(self.sorter_pose_type_combobox)
            self.fade_widget_in(self.sorter_motion_blur_checkbox)
        else:
            # 隐藏特殊选项
            self.fade_widget_out(self.sorter_pose_type_combobox)
            self.fade_widget_out(self.sorter_motion_blur_checkbox)
        
        # 只有当 motion_blur_checkbox 的可见性真正改变时才触发动画
        # 使用定时器等待 fade 动画完成后再调整卡片高度
        QTimer.singleShot(250, lambda: self._animate_sorter_card_height(method))
    
    def _animate_sorter_card_height(self, method: str):
        """在 fade 动画完成后调整卡片高度"""
        if not hasattr(self, 'sorter_method_combobox'):
            return
        
        # 找到包含排序模式的卡片
        card = self.sorter_method_combobox.parent()
        while card and not hasattr(card, 'setTitle'):
            card = card.parent()
        
        if not card:
            return
        
        # 保存当前宽度
        current_width = card.width()
        current_height = card.height()
        
        # 临时取消固定宽度限制
        card.setMinimumWidth(0)
        card.setMaximumWidth(16777215)
        
        # 强制刷新布局
        card.layout().invalidate()
        card.layout().activate()
        QApplication.processEvents()
        
        # 延迟计算目标高度
        QTimer.singleShot(10, lambda: self._calculate_sorter_card_animation(card, current_width, current_height))
    
    def _calculate_sorter_card_animation(self, card, current_width, current_height):
        """延迟计算排序卡片目标高度并启动动画"""
        # 计算目标高度
        card.adjustSize()
        target_height = card.sizeHint().height()
        
        # 恢复宽度限制并设置当前宽度
        card.setFixedWidth(current_width)
        
        # 如果目标高度与当前高度相同，不需要动画
        if abs(target_height - current_height) < 1:
            return
        
        # 创建或获取高度动画
        sorter_animation_key = f"sorter_{id(card)}"
        if sorter_animation_key not in self.card_height_animations:
            animation = SiExpAnimation(card)
            animation.setFactor(1/8)
            animation.setBias(0.2)
            animation.ticked.connect(lambda value, c=card: c.setFixedHeight(int(value)))
            self.card_height_animations[sorter_animation_key] = animation
        
        # 设置动画的起始和目标值
        animation = self.card_height_animations[sorter_animation_key]
        animation.setCurrent(current_height)
        animation.setTarget(target_height)
        animation.start()
    
    def on_write_back_clicked(self):
        """写回源文件按钮点击事件"""
        # 获取输入路径
        input_path = self.analyzer_input_path_input.text().strip()
        if not input_path:
            print("错误: 请输入有效的人脸集路径")
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
        
        # 构建命令
        analyzer_script = project_root / "FacesetProcessor" / "Analyzer.py"
        
        # 使用当前运行的 Python 解释器（支持 conda 环境）
        import sys
        python_exe = Path(sys.executable)
        
        cmd = [
            str(python_exe),
            str(analyzer_script),
            "--input", str(input_path_obj),
            "--write-back",
        ]
        
        # 打印命令
        cmd_str = " ".join(cmd)
        print(f"\n执行命令:\n{cmd_str}\n")
        print("提示: 用于兼容原版DeepFaceLab，虽然我不知道为什么你还要用回老版本")
        
        # 显示通知
        parent_window = self.window()
        if hasattr(parent_window, 'show_command_notification'):
            parent_window.show_command_notification(
                cmd_str,
                "写回源文件 - DFL兼容模式"
            )
        
        try:
            # 创建任务监控器（在修改按钮状态之前）
            monitor = self._create_task_monitor(
                self.write_back_button,
                "元数据写回已完成",
                "用户强制中断了元数据写回任务"
            )
            
            # 设置按钮为运行状态
            self.write_back_button.setText("正在写入...")
            self.write_back_button.setEnabled(False)
            
            # 在新窗口中运行命令
            if hasattr(subprocess, 'CREATE_NEW_CONSOLE'):
                cmd_with_pause = " ".join(cmd) + " & echo. & echo 任务完成！按任意键关闭窗口... & pause"
                process = subprocess.Popen(
                    f"cmd /c {cmd_with_pause}",
                    creationflags=subprocess.CREATE_NEW_CONSOLE
                )
                
                # 启动监控线程（使用默认参数避免闭包问题）
                monitor_thread = threading.Thread(target=lambda p=process: monitor(p), daemon=True)
                monitor_thread.start()
            else:
                process = subprocess.Popen(cmd)
                monitor_thread = threading.Thread(target=lambda p=process: monitor(p), daemon=True)
                monitor_thread.start()
                
            print("✓ 已启动元数据写回进程")
        except Exception as e:
            print(f"✗ 启动失败: {e}")
            self.write_back_button.setText("写回源文件")
            self.write_back_button.setEnabled(True)
    
    def on_run_analyzer_clicked(self, force_reanalyze: bool = False):
        """Analyzer 运行按钮点击事件
        
        Args:
            force_reanalyze: 是否强制重新分析（覆盖已有数据）
        """
        # 获取输入路径
        input_path = self.analyzer_input_path_input.text().strip()
        if not input_path:
            print("错误: 请输入有效的人脸集路径")
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
        
        # 获取选择的特征
        selected_features = self.get_selected_features()
        if not selected_features:
            print("错误: 请至少选择一个分析特征")
            return
        
        # 构建命令
        analyzer_script = project_root / "FacesetProcessor" / "Analyzer.py"
        
        # 使用当前运行的 Python 解释器（支持 conda 环境）
        import sys
        python_exe = Path(sys.executable)
        
        cmd = [
            str(python_exe),
            str(analyzer_script),
            "--input", str(input_path_obj),
            "--features", ",".join(selected_features),
        ]
        
        # 如果勾选了强制重新分析，添加 --force 参数
        if force_reanalyze:
            cmd.append("--force")
        
        # 打印命令
        cmd_str = " ".join(cmd)
        print(f"\n执行命令:\n{cmd_str}\n")
        print(f"已选择的特征: {', '.join(selected_features)}")
        print(f"强制重新分析: {'是' if force_reanalyze else '否'}")
        
        # 显示通知
        parent_window = self.window()
        if hasattr(parent_window, 'show_command_notification'):
            mode_text = "强制重新分析" if force_reanalyze else "增量分析"
            parent_window.show_command_notification(
                cmd_str,
                f"元数据分析 - {mode_text} - 特征: {', '.join(selected_features)}"
            )
        
        try:
            # 创建任务监控器（在修改按钮状态之前）
            monitor = self._create_task_monitor(
                self.run_analyzer_button,
                f"元数据分析已完成 - 特征: {', '.join(selected_features)}",
                "用户强制中断了元数据分析任务"
            )
            
            # 设置按钮为运行状态
            self.run_analyzer_button.setText("正在运行...")
            self.run_analyzer_button.setEnabled(False)
            
            # 在新窗口中运行命令
            if hasattr(subprocess, 'CREATE_NEW_CONSOLE'):
                cmd_with_pause = " ".join(cmd) + " & echo. & echo 任务完成！按任意键关闭窗口... & pause"
                process = subprocess.Popen(
                    f"cmd /c {cmd_with_pause}",
                    creationflags=subprocess.CREATE_NEW_CONSOLE
                )
                
                # 启动监控线程（使用默认参数避免闭包问题）
                monitor_thread = threading.Thread(target=lambda p=process: monitor(p), daemon=True)
                monitor_thread.start()
            else:
                process = subprocess.Popen(cmd)
                monitor_thread = threading.Thread(target=lambda p=process: monitor(p), daemon=True)
                monitor_thread.start()
                
            print("✓ 已启动元数据分析进程")
        except Exception as e:
            print(f"✗ 启动失败: {e}")
            self.run_analyzer_button.setText("开始分析")
            self.run_analyzer_button.setEnabled(True)
    def on_run_filter_clicked(self):
        """Filter 运行按钮点击事件"""
        # 获取输入路径
        input_path = self.filter_input_path_input.text().strip()
        if not input_path:
            print("错误: 请输入有效的人脸集路径")
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
        
        # 获取过滤模式
        filter_mode = self.filter_mode_combobox.currentText()
        
        # 构建命令
        filter_script = project_root / "FacesetProcessor" / "Filter.py"
        
        # 使用当前运行的 Python 解释器（支持 conda 环境）
        import sys
        python_exe = Path(sys.executable)
        
        cmd = [
            str(python_exe),
            str(filter_script),
            "--input", str(input_path_obj),
            "--mode", filter_mode,
        ]
        
        # 根据模式添加参数
        if filter_mode == "position":
            # 位置过滤模式：添加 eps 和 min-samples 参数
            try:
                eps = float(self.filter_position_eps_input.text().strip() or "50")
                min_samples = int(self.filter_position_min_samples_input.text().strip() or "2")
                cmd.extend(["--threshold", str(eps), "--min-samples", str(min_samples)])
            except ValueError:
                print("错误: 请输入有效的数字参数")
                return
        elif filter_mode == "faceid":
            # 人脸ID过滤模式：添加 eps, min-samples 参数
            try:
                eps = float(self.filter_faceid_eps_input.text().strip() or "0.3")
                min_samples = int(self.filter_faceid_min_samples_input.text().strip() or "2")
                cmd.extend([
                    "--eps", str(eps),
                    "--min-samples", str(min_samples)
                ])
            except ValueError:
                print("错误: 请输入有效的数字参数")
                return
        elif filter_mode == "repeated":
            # 重复去重模式：添加 threshold 参数（固定使用phash）
            try:
                threshold = float(self.filter_repeated_threshold_input.text().strip() or "0.98")
                cmd.extend([
                    "--threshold", str(threshold)
                ])
            except ValueError:
                print("错误: 请输入有效的阈值参数")
                return
        
        # 打印命令
        cmd_str = " ".join(cmd)
        print(f"\n执行命令:\n{cmd_str}\n")
        
        mode_desc = {
            "blur": "模糊过滤",
            "faceid": "人脸ID分组",
            "position": "位置聚类",
            "repeated": "重复去重"
        }
        
        # 显示通知
        parent_window = self.window()
        if hasattr(parent_window, 'show_command_notification'):
            parent_window.show_command_notification(
                cmd_str,
                f"{mode_desc.get(filter_mode, '过滤')} - {filter_mode} 模式"
            )
        
        try:
            # 创建任务监控器（在修改按钮状态之前）
            monitor = self._create_task_monitor(
                self.run_filter_button,
                f"人脸集过滤已完成 - 模式: {filter_mode}",
                "用户强制中断了人脸集过滤任务"
            )
            
            # 设置按钮为运行状态
            self.run_filter_button.setText("正在运行...")
            self.run_filter_button.setEnabled(False)
            
            # 在新窗口中运行命令
            if hasattr(subprocess, 'CREATE_NEW_CONSOLE'):
                cmd_with_pause = " ".join(cmd) + " & echo. & echo 任务完成！按任意键关闭窗口... & pause"
                process = subprocess.Popen(
                    f"cmd /c {cmd_with_pause}",
                    creationflags=subprocess.CREATE_NEW_CONSOLE
                )
                
                # 启动监控线程（使用默认参数避免闭包问题）
                monitor_thread = threading.Thread(target=lambda p=process: monitor(p), daemon=True)
                monitor_thread.start()
            else:
                process = subprocess.Popen(cmd)
                monitor_thread = threading.Thread(target=lambda p=process: monitor(p), daemon=True)
                monitor_thread.start()
                
            print("✓ 已启动人脸集过滤进程")
        except Exception as e:
            print(f"✗ 启动失败: {e}")
            self.run_filter_button.setText("开始过滤")
            self.run_filter_button.setEnabled(True)
    def on_filter_mode_changed(self, mode: str):
        """过滤模式切换事件"""
        if mode == "position":
            # 显示位置参数
            self.filter_position_eps_input.setVisible(True)
            self.filter_position_min_samples_input.setVisible(True)
            # 隐藏人脸ID参数
            self.filter_faceid_eps_input.setVisible(False)
            self.filter_faceid_min_samples_input.setVisible(False)
            # 隐藏重复去重参数
            self.filter_repeated_threshold_input.setVisible(False)
        elif mode == "faceid":
            # 显示人脸ID参数
            self.filter_faceid_eps_input.setVisible(True)
            self.filter_faceid_min_samples_input.setVisible(True)
            # 隐藏位置参数
            self.filter_position_eps_input.setVisible(False)
            self.filter_position_min_samples_input.setVisible(False)
            # 隐藏重复去重参数
            self.filter_repeated_threshold_input.setVisible(False)
        elif mode == "repeated":
            # 显示重复去重参数
            self.filter_repeated_threshold_input.setVisible(True)
            # 隐藏位置参数
            self.filter_position_eps_input.setVisible(False)
            self.filter_position_min_samples_input.setVisible(False)
            # 隐藏人脸ID参数
            self.filter_faceid_eps_input.setVisible(False)
            self.filter_faceid_min_samples_input.setVisible(False)
        else:
            # 隐藏所有参数
            self.filter_position_eps_input.setVisible(False)
            self.filter_position_min_samples_input.setVisible(False)
            self.filter_faceid_eps_input.setVisible(False)
            self.filter_faceid_min_samples_input.setVisible(False)
            self.filter_repeated_threshold_input.setVisible(False)
        
        # 动态调整卡片大小 - 只调整高度，保持宽度不变（带动画）
        if hasattr(self, 'filter_mode_combobox'):
            # 找到包含过滤模式的卡片
            card = self.filter_mode_combobox.parent()
            while card and not hasattr(card, 'setTitle'):
                card = card.parent()
            if card:
                # 保存当前宽度和高度
                current_width = card.width()
                current_height = card.height()
                
                # 临时取消固定宽度限制，让 adjustSize 能正确计算
                card.setMinimumWidth(0)
                card.setMaximumWidth(16777215)  # QWIDGETSIZE_MAX
                
                # 强制刷新布局
                card.layout().invalidate()
                card.layout().activate()
                QApplication.processEvents()
                
                # 使用定时器延迟计算目标高度，确保布局已完全更新
                QTimer.singleShot(10, lambda: self._calculate_and_animate(card, current_width, current_height))
    
    def _calculate_and_animate(self, card, current_width, current_height):
        """延迟计算目标高度并启动动画"""
        # 计算目标高度
        card.adjustSize()
        target_height = card.sizeHint().height()
        
        # 恢复宽度限制并设置当前宽度
        card.setFixedWidth(current_width)
        
        # 如果目标高度与当前高度相同，不需要动画
        if abs(target_height - current_height) < 1:
            return
        
        # 创建或获取高度动画
        if id(card) not in self.card_height_animations:
            animation = SiExpAnimation(card)
            animation.setFactor(1/8)
            animation.setBias(0.2)
            animation.ticked.connect(lambda value, c=card: c.setFixedHeight(int(value)))
            self.card_height_animations[id(card)] = animation
        
        # 设置动画的起始和目标值
        animation = self.card_height_animations[id(card)]
        animation.setCurrent(current_height)
        animation.setTarget(target_height)
        animation.start()
    def on_merge_subfolders_clicked(self):
        """合并子文件夹按钮点击事件"""
        # 获取输入路径
        input_path = self.filter_input_path_input.text().strip()
        if not input_path:
            print("错误: 请输入有效的人脸集路径")
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
        
        # 安全检查：确保不是系统根目录
        dangerous_paths = ['C:\\', 'D:\\', 'E:\\', 'F:\\', '/', '/home', '/usr']
        if str(input_path_obj) in dangerous_paths or len(str(input_path_obj)) <= 3:
            print(f"错误: 出于安全考虑，不允许在根目录执行此操作: {input_path_obj}")
            return
        
        # 构建命令
        filter_script = project_root / "FacesetProcessor" / "Filter.py"
        
        # 使用当前运行的 Python 解释器（支持 conda 环境）
        import sys
        python_exe = Path(sys.executable)
        
        cmd = [
            str(python_exe),
            str(filter_script),
            "--input", str(input_path_obj),
            "--merge-only",
            "--output-dir", "aligned"
        ]
        
        # 打印命令
        cmd_str = " ".join(cmd)
        print(f"\n执行命令:\n{cmd_str}\n")
        print("提示: 只遍历一级子文件夹，不会递归到更深层")
        
        # 显示通知
        parent_window = self.window()
        if hasattr(parent_window, 'show_command_notification'):
            parent_window.show_command_notification(
                cmd_str,
                "合并子文件夹 - 安全模式"
            )
        
        try:
            # 创建任务监控器（在修改按钮状态之前）
            monitor = self._create_task_monitor(
                self.merge_subfolders_button,
                "子文件夹合并已完成",
                "用户强制中断了子文件夹合并任务"
            )
            
            # 设置按钮为运行状态
            self.merge_subfolders_button.setText("正在合并...")
            self.merge_subfolders_button.setEnabled(False)
            
            # 在新窗口中运行命令
            if hasattr(subprocess, 'CREATE_NEW_CONSOLE'):
                cmd_with_pause = " ".join(cmd) + " & echo. & echo 任务完成！按任意键关闭窗口... & pause"
                process = subprocess.Popen(
                    f"cmd /c {cmd_with_pause}",
                    creationflags=subprocess.CREATE_NEW_CONSOLE
                )
                
                # 启动监控线程（使用默认参数避免闭包问题）
                monitor_thread = threading.Thread(target=lambda p=process: monitor(p), daemon=True)
                monitor_thread.start()
            else:
                process = subprocess.Popen(cmd)
                monitor_thread = threading.Thread(target=lambda p=process: monitor(p), daemon=True)
                monitor_thread.start()
                
            print("✓ 已启动子文件夹合并进程")
        except Exception as e:
            print(f"✗ 启动失败: {e}")
            self.merge_subfolders_button.setText("合并子文件夹")
            self.merge_subfolders_button.setEnabled(True)
    def on_run_sorter_clicked(self):
        """Sorter 运行按钮点击事件"""
        # 获取输入路径
        input_path = self.sorter_input_path_input.text().strip()
        if not input_path:
            print("错误: 请输入有效的人脸集路径")
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
        
        # 获取排序方式
        sort_method = self.sorter_method_combobox.currentText()
        
        # 构建命令
        sorter_script = project_root / "FacesetProcessor" / "Sorter.py"
        
        # 使用当前运行的 Python 解释器（支持 conda 环境）
        import sys
        python_exe = Path(sys.executable)
        
        cmd = [
            str(python_exe),
            str(sorter_script),
            "--input", str(input_path_obj),
            "--method", sort_method,
        ]
        
        # 根据排序方法添加额外参数
        description_parts = [f"方法: {sort_method}"]
        
        if sort_method == "face_pose":
            # 添加姿态类型参数
            pose_type = self.sorter_pose_type_combobox.currentText()
            cmd.extend(["--pose-type", pose_type])
            description_parts.append(f"姿态: {pose_type}")
        elif sort_method == "blur":
            # 添加运动模糊参数
            if self.sorter_motion_blur_checkbox.isChecked():
                cmd.append("--motion-blur")
                description_parts.append("运动模糊")
        
        # 始终重命名文件（排序的核心功能）
        cmd.extend(["--rename", "--prefix", "sorted"])
        description_parts.append("自动重命名+更新元数据")
        
        # 打印命令
        cmd_str = " ".join(cmd)
        print(f"\n执行命令:\n{cmd_str}\n")
        
        # 显示通知
        parent_window = self.window()
        if hasattr(parent_window, 'show_command_notification'):
            parent_window.show_command_notification(
                cmd_str,
                f"人脸集排序 - {', '.join(description_parts)}"
            )
        
        try:
            # 创建任务监控器（在修改按钮状态之前）
            monitor = self._create_task_monitor(
                self.run_sorter_button,
                f"人脸集排序已完成 - {', '.join(description_parts)}",
                "用户强制中断了人脸集排序任务"
            )
            
            # 设置按钮为运行状态
            self.run_sorter_button.setText("正在运行...")
            self.run_sorter_button.setEnabled(False)
            
            # 在新窗口中运行命令
            if hasattr(subprocess, 'CREATE_NEW_CONSOLE'):
                cmd_with_pause = " ".join(cmd) + " & echo. & echo 任务完成！按任意键关闭窗口... & pause"
                process = subprocess.Popen(
                    f"cmd /c {cmd_with_pause}",
                    creationflags=subprocess.CREATE_NEW_CONSOLE
                )
                
                # 启动监控线程（使用默认参数避免闭包问题）
                monitor_thread = threading.Thread(target=lambda p=process: monitor(p), daemon=True)
                monitor_thread.start()
            else:
                process = subprocess.Popen(cmd)
                monitor_thread = threading.Thread(target=lambda p=process: monitor(p), daemon=True)
                monitor_thread.start()
                
            print("✓ 已启动人脸集排序进程")
        except Exception as e:
            print(f"✗ 启动失败: {e}")
            self.run_sorter_button.setText("开始排序")
            self.run_sorter_button.setEnabled(True)

    # ── PAK 解包 ──────────────────────────────────────────────────

    def _on_pak_unpack(self):
        """解包 faceset.pak 文件（在新进程中执行，避免 PyTorch DLL 冲突）"""
        import sys, subprocess, threading
        from pathlib import Path

        input_dir = self.pak_input_path.text().strip()
        if not input_dir:
            print("错误: 请输入包含 faceset.pak 的目录路径")
            return
        pak_path = Path(input_dir) / "faceset.pak"
        if not pak_path.exists():
            print(f"错误: 未找到 faceset.pak: {pak_path}")
            return
        output_dir = self.pak_output_path.text().strip() or ""

        self.pak_unpack_btn.setText("正在解包...")
        self.pak_unpack_btn.setEnabled(False)
        QApplication.processEvents()

        def _run():
            try:
                cmd = [sys.executable, '-m', 'tools.unpack_pak', str(pak_path.parent)]
                if output_dir:
                    cmd.append(output_dir)
                r = subprocess.run(cmd, capture_output=True, text=True, timeout=120,
                                   cwd=str(Path(__file__).parent.parent.parent.parent))
                print(r.stdout)
                if r.returncode == 0:
                    from PyQt5.QtCore import QTimer
                    QTimer.singleShot(0, lambda: self._on_pak_done(True, str(pak_path.parent)))
                else:
                    print(f"✗ PAK 解包失败: {r.stderr}")
                    QTimer.singleShot(0, lambda: self._on_pak_done(False))
            except Exception as e:
                print(f"✗ PAK 解包失败: {e}")
                QTimer.singleShot(0, lambda: self._on_pak_done(False))
        threading.Thread(target=_run, daemon=True).start()

    def _on_pak_done(self, ok, output_dir=""):
        """PAK 解包完成回调"""
        self.pak_unpack_btn.setText("开始解包")
        self.pak_unpack_btn.setEnabled(True)
        if ok:
            # 删除源文件
            pak_path = Path(self.pak_input_path.text().strip()) / "faceset.pak"
            if pak_path.exists():
                try:
                    pak_path.unlink()
                    print(f"[PAK] 已删除源文件: {pak_path}")
                except Exception as e:
                    print(f"[PAK] 删除源文件失败: {e}")
            _pw = self.window()
            if hasattr(_pw, 'show_task_completed_notification'):
                _pw.show_task_completed_notification(f"PAK 解包完成 → {output_dir}")

    def _on_viz_launch(self):
        """启动角度可视化服务器"""
        import subprocess, sys, threading, time
        from pathlib import Path
        project_root = Path(__file__).parent.parent.parent.parent
        script = project_root / "samplelib" / "serve_angle_view.py"
        if not script.exists():
            print(f"[可视化] 脚本不存在: {script}")
            return
        python_exe = Path(sys.executable)
        self.viz_button.setText("启动中...")
        self.viz_button.setEnabled(False)
        def _run():
            try:
                p = subprocess.Popen(
                    [str(python_exe), str(script)],
                    cwd=str(project_root),
                    creationflags=subprocess.CREATE_NEW_CONSOLE
                )
            except Exception as e:
                print(f"[可视化] 启动失败: {e}")
            finally:
                self.viz_button.setText("数据可视化")
                self.viz_button.setEnabled(True)
        threading.Thread(target=_run, daemon=True).start()
        time.sleep(0.5)
        print("[可视化] 服务器已启动，请查看新窗口")
