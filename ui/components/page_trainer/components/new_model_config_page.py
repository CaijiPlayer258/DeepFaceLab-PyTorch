"""
新建模型配置子页面 - 点击新建模型按钮时弹出（仅模型架构）
"""
from pathlib import Path
from PyQt5.QtCore import pyqtSignal, QTimer
from siui.components.page.child_page import SiChildPage
from siui.components import SiTitledWidgetGroup, SiPushButton, SiSwitch
from siui.components.editbox import SiLabeledLineEdit
from siui.components.option_card import SiOptionCardLinear
from siui.components.combobox_ import SiCapsuleComboBox
from siui.core import SiGlobal


from core.leras.archis.archi_spec import (
    ARCHI_CHOICES, ARCHI_DESCRIPTIONS, SUBARCHI_CHOICES, SUBARCHI_DESCRIPTIONS,
    KERNEL_CHOICES, build_archi_string as _build_archi_string,
    kernel_desc_for, is_lite_display, default_kernel_label,
)


def safe_get_icon(name):
    """安全获取图标"""
    try:
        return SiGlobal.siui.iconpack.get(name)
    except KeyError:
        return None


class NewModelConfigChildPage(SiChildPage):
    """新建模型配置子页面 - 点击新建模型按钮时弹出（仅模型架构）"""
    
    # 定义信号：当用户点击创建模型按钮时发出
    model_created = pyqtSignal(dict)
    
    def __init__(self, parent=None, model_name="model_001", model_info=None):
        super().__init__(parent)
        self.model_info = model_info or {}
        
        self.view().setMinimumWidth(900)
        self.content().setTitle(f"新建模型配置 - {model_name}")
        self.content().setPadding(48)
        
        # 存储所有配置项的值
        self.config_data = {
            'model_name': model_name,
            'resolution': '256',
            'archi': 'DF',
            'subarchi': '-ud',
            'kernel': None,   # None -> default_kernel_label()，下面初始化时填
            'e_dims': '64',
            'ae_dims': '256',
            'd_dims': '128',
            'd_mask_dims': '32'
        }
        
        # 页面内容
        self.titled_widget_group = SiTitledWidgetGroup(self)
        
        with self.titled_widget_group as group:
            group.addTitle("模型架构参数")
            
            # 模型名称
            self.model_name_card = SiOptionCardLinear(self)
            self.model_name_card.setTitle("模型名称", "设置模型的名称")
            self.model_name_card.load(safe_get_icon("ic_fluent_textbox_filled"))
            self.model_name_input = SiLabeledLineEdit(self.model_name_card)
            self.model_name_input.setTitle("字符串")
            self.model_name_input.setText(model_name)
            self.model_name_input.setFixedHeight(48)
            self.model_name_input.resize(150, 48)
            self.model_name_input.textChanged.connect(lambda: self.update_config('model_name'))
            self.model_name_card.addWidget(self.model_name_input)
            self.model_name_card.adjustSize()
            
            # 分辨率
            self.resolution_card = SiOptionCardLinear(self)
            self.resolution_card.setTitle("分辨率", "模型输入输出的图像分辨率")
            self.resolution_card.load(safe_get_icon("ic_fluent_scan_filled"))
            self.resolution_input = SiLabeledLineEdit(self.resolution_card)
            self.resolution_input.setTitle("整数")
            self.resolution_input.setText("256")
            self.resolution_input.setFixedHeight(48)
            self.resolution_input.resize(150, 48)
            self.resolution_input.textChanged.connect(lambda: self.update_config('resolution'))
            self.resolution_card.addWidget(self.resolution_input)
            self.resolution_card.adjustSize()
            
            # archi - 模型架构（胶囊下拉框）
            self.archi_card = SiOptionCardLinear(self)
            self.archi_card.setTitle("架构", "选择你的模型架构")
            self.archi_card.load(safe_get_icon("ic_fluent_branch_fork_filled"))
            self.archi_combo = SiCapsuleComboBox(self.archi_card)
            self.archi_combo.setTitle("架构")
            self.archi_combo.setFixedWidth(200)
            self.archi_combo.setMinimumHeight(32)
            self.archi_combo.setMaximumHeight(32)
            self.archi_combo.setEditable(False)
            self.archi_combo.addItems([c[0] for c in ARCHI_CHOICES])
            self.archi_combo.setCurrentText(ARCHI_CHOICES[0][0])
            self.archi_combo.currentTextChanged.connect(lambda text: self.update_architecture(text))
            self.archi_card.addWidget(self.archi_combo)
            self.archi_card.adjustSize()
            
            # 子分支选项（胶囊下拉框）
            self.subarchi_card = SiOptionCardLinear(self)
            self.subarchi_card.setTitle("子分支", "选择架构的子分支变体")
            self.subarchi_card.load(safe_get_icon("ic_fluent_branch_compare_filled"))
            self.subarchi_combo = SiCapsuleComboBox(self.subarchi_card)
            self.subarchi_combo.setTitle("子分支")
            self.subarchi_combo.setFixedWidth(200)
            self.subarchi_combo.setMinimumHeight(32)
            self.subarchi_combo.setMaximumHeight(32)
            self.subarchi_combo.setEditable(False)
            self.subarchi_combo.addItems(list(SUBARCHI_CHOICES))
            self.subarchi_combo.setCurrentText("-ud")
            self.subarchi_combo.currentTextChanged.connect(lambda text: self.update_subarchitecture(text))
            self.subarchi_card.addWidget(self.subarchi_combo)
            self.subarchi_card.adjustSize()

            # 算子 - 仅对 Lite 架构生效（DF Lite / LIAE Lite）
            self.kernel_card = SiOptionCardLinear(self)
            self.kernel_card.setTitle("算子", "Lite 架构的卷积算子（决定速度/参数量/感受野）")
            self.kernel_card.load(safe_get_icon("ic_fluent_settings_filled"))
            self.kernel_combo = SiCapsuleComboBox(self.kernel_card)
            self.kernel_combo.setTitle("算子")
            self.kernel_combo.setFixedWidth(330)
            self.kernel_combo.setMinimumHeight(32)
            self.kernel_combo.setMaximumHeight(32)
            self.kernel_combo.setEditable(False)
            self.kernel_combo.addItems([c[0] for c in KERNEL_CHOICES])
            _default_kernel = default_kernel_label()
            self.kernel_combo.setCurrentText(_default_kernel)
            self.config_data['kernel'] = _default_kernel
            self.kernel_combo.currentTextChanged.connect(lambda text: self.update_kernel(text))
            self.kernel_card.addWidget(self.kernel_combo)
            self.kernel_card.adjustSize()
            
            # e_dims - 编码器维度大小
            self.e_dims_card = SiOptionCardLinear(self)
            self.e_dims_card.setTitle("e_dims", "编码器维度大小，用卷积提取输入的特征")
            self.e_dims_card.load(safe_get_icon("ic_fluent_box_filled"))
            self.e_dims_input = SiLabeledLineEdit(self.e_dims_card)
            self.e_dims_input.setTitle("整数")
            self.e_dims_input.setText("64")
            self.e_dims_input.setFixedHeight(48)
            self.e_dims_input.resize(150, 48)
            self.e_dims_input.setEnabled(True)
            self.e_dims_input.textChanged.connect(lambda: self.update_config('e_dims'))
            self.e_dims_card.addWidget(self.e_dims_input)
            self.e_dims_card.adjustSize()

            # ae_dims - 隐空间维度大小
            self.ae_dims_card = SiOptionCardLinear(self)
            self.ae_dims_card.setTitle("ae_dims", "隐空间维度大小，将编码器卷积拓张为张量\n这个值越大，理论能存的特征越多，但是超过了256大小有用的维度越少")
            self.ae_dims_card.load(safe_get_icon("ic_fluent_cube_filled"))
            self.ae_dims_input = SiLabeledLineEdit(self.ae_dims_card)
            self.ae_dims_input.setTitle("整数")
            self.ae_dims_input.setText("256")
            self.ae_dims_input.setFixedHeight(48)
            self.ae_dims_input.resize(150, 48)
            self.ae_dims_input.setEnabled(True)
            self.ae_dims_input.textChanged.connect(lambda: self.update_config('ae_dims'))
            self.ae_dims_card.addWidget(self.ae_dims_input)
            self.ae_dims_card.adjustSize()

            # d_dims - 解码器维度大小
            self.d_dims_card = SiOptionCardLinear(self)
            self.d_dims_card.setTitle("d_dims", "解码器维度大小，将中间层的输出进行上采样解码\n解码器越大，解码能力越强，模型恢复的图像更厉害")
            self.d_dims_card.load(safe_get_icon("ic_fluent_arrow_upload_filled"))
            self.d_dims_input = SiLabeledLineEdit(self.d_dims_card)
            self.d_dims_input.setTitle("整数")
            self.d_dims_input.setText("128")
            self.d_dims_input.setFixedHeight(48)
            self.d_dims_input.resize(150, 48)
            self.d_dims_input.setEnabled(True)
            self.d_dims_input.textChanged.connect(lambda: self.update_config('d_dims'))
            self.d_dims_card.addWidget(self.d_dims_input)
            self.d_dims_card.adjustSize()

            # d_mask_dims - 遮罩解码器维度大小
            self.d_mask_dims_card = SiOptionCardLinear(self)
            self.d_mask_dims_card.setTitle("d_mask_dims", "遮罩解码器维度大小，为pred提供一个遮罩\n过高并不能提高特别多精度")
            self.d_mask_dims_card.load(safe_get_icon("ic_fluent_eye_off_filled"))
            self.d_mask_dims_input = SiLabeledLineEdit(self.d_mask_dims_card)
            self.d_mask_dims_input.setTitle("整数")
            self.d_mask_dims_input.setText("32")
            self.d_mask_dims_input.setFixedHeight(48)
            self.d_mask_dims_input.resize(150, 48)
            self.d_mask_dims_input.setEnabled(True)
            self.d_mask_dims_input.textChanged.connect(lambda: self.update_config('d_mask_dims'))
            self.d_mask_dims_card.addWidget(self.d_mask_dims_input)
            self.d_mask_dims_card.adjustSize()
            
            group.addWidget(self.model_name_card)
            group.addWidget(self.resolution_card)
            group.addWidget(self.archi_card)
            group.addWidget(self.subarchi_card)
            group.addWidget(self.kernel_card)
            group.addWidget(self.e_dims_card)
            group.addWidget(self.ae_dims_card)
            group.addWidget(self.d_dims_card)
            group.addWidget(self.d_mask_dims_card)
        
        self.content().setAttachment(self.titled_widget_group)
        
        # 控制面板
        self.create_button = SiPushButton(self)
        self.create_button.resize(128, 32)
        self.create_button.attachment().setText("创建模型")
        self.create_button.clicked.connect(self.on_create_clicked)
        
        self.cancel_button = SiPushButton(self)
        self.cancel_button.resize(128, 32)
        self.cancel_button.attachment().setText("取消")
        self.cancel_button.clicked.connect(self.closeParentLayer)
        
        self.panel().addWidget(self.cancel_button, "left")
        self.panel().addWidget(self.create_button, "right")
        
        # 加载样式表
        SiGlobal.siui.reloadStyleSheetRecursively(self)
    
    def update_architecture(self, text):
        """更新架构选择并改变描述"""
        self.config_data['archi'] = text
        print(f"[CONFIG] archi = {text}")
        if text in ARCHI_DESCRIPTIONS:
            self.archi_card.setTitle("架构", ARCHI_DESCRIPTIONS[text])
        self._refresh_archi_str()

    def update_kernel(self, text):
        """更新算子选择（仅 Lite 架构生效）"""
        self.config_data['kernel'] = text
        print(f"[CONFIG] kernel = {text}")
        self.kernel_card.setTitle("算子", kernel_desc_for(text))
        self._refresh_archi_str()

    def _refresh_archi_str(self):
        """把「架构 + 子分支 + 算子」拼成后端可解析的 archi 串。

        显示名 'DF' 要变成 'df'；选了 Lite 架构才追加第三段算子修饰符。
        结果放进 config_data['archi_str']，page_trainer 直接取用。
        """
        try:
            self.config_data['archi_str'] = _build_archi_string(
                self.config_data.get('archi', 'DF'),
                self.config_data.get('subarchi', '-ud'),
                self.config_data.get('kernel') or default_kernel_label(),
            )
        except Exception as e:
            print(f"[CONFIG] build_archi_string 失败: {e}")
            self.config_data['archi_str'] = None
        print(f"[CONFIG] archi_str = {self.config_data['archi_str']}")
    
    def update_subarchitecture(self, text):
        """更新子分支选择并改变描述"""
        self.config_data['subarchi'] = text
        print(f"[CONFIG] subarchi = {text}")
        
        if text in SUBARCHI_DESCRIPTIONS:
            self.subarchi_card.setTitle("子分支", SUBARCHI_DESCRIPTIONS[text])
        self._refresh_archi_str()
    
    def update_config(self, key, value=None):
        """更新配置值"""
        # 检查是否是开关
        if hasattr(self, f'{key}_switch'):
            switch_widget = getattr(self, f'{key}_switch')
            self.config_data[key] = switch_widget.isChecked()
            print(f"[CONFIG] {key} = {self.config_data[key]}")
        # 检查是否是输入框
        elif hasattr(self, f'{key}_input'):
            input_widget = getattr(self, f'{key}_input')
            self.config_data[key] = input_widget.text()
            print(f"[CONFIG] {key} = {self.config_data[key]}")
        # 检查是否是下拉框（SiCapsuleComboBox）
        elif hasattr(self, f'{key}_combo'):
            combo_widget = getattr(self, f'{key}_combo')
            if value is not None:
                self.config_data[key] = value
            else:
                # 对于 SiCapsuleComboBox，使用 currentText() 获取当前值
                self.config_data[key] = combo_widget.currentText()
            print(f"[CONFIG] {key} = {self.config_data[key]}")
    
    def get_config(self):
        """获取当前配置"""
        return self.config_data.copy()
    
    def on_create_clicked(self):
        """点击创建模型按钮"""

        # 发出信号，传递配置数据
        self.model_created.emit(self.config_data)

        # 关闭子页面
        self.closeParentLayer()


class NewLargeModelConfigChildPage(SiChildPage):
    """
    新建 Large 系列模型配置子页面（DeepFakeLarge / LIAELarge）
    架构固定，只需设置名称和分辨率
    """

    model_created = pyqtSignal(dict)

    def __init__(self, parent=None, model_name="model_001",
                 model_class="DeepFakeLarge", default_options=None):
        super().__init__(parent)
        self.model_class = model_class
        self.default_options = default_options or {'ae_dims': 256, 'e_dims': 64, 'd_dims': 128}

        ae_def = self.default_options.get('ae_dims', 256)
        e_def = self.default_options.get('e_dims', 64)
        d_def = self.default_options.get('d_dims', 128)

        display_name = {"DeepFakeLarge": "DFLarge", "LIAELarge": "LIAELarge"}.get(model_class, model_class)

        self.view().setMinimumWidth(900)
        self.content().setTitle(f"新建 {display_name} 模型 - {model_name}")
        self.content().setPadding(48)

        self.config_data = {
            'model_name': model_name,
            'model_class': model_class,
            'resolution': '128',
            'ae_dims': str(ae_def),
            'e_dims': str(e_def),
            'd_dims': str(d_def),
        }

        self.titled_widget_group = SiTitledWidgetGroup(self)

        with self.titled_widget_group as group:
            group.addTitle("模型参数")

            # 模型名称
            self.model_name_card = SiOptionCardLinear(self)
            self.model_name_card.setTitle("模型名称", f"设置 {display_name} 模型的名称")
            self.model_name_card.load(safe_get_icon("ic_fluent_textbox_filled"))
            self.model_name_input = SiLabeledLineEdit(self.model_name_card)
            self.model_name_input.setTitle("字符串")
            self.model_name_input.setText(model_name)
            self.model_name_input.setFixedHeight(48)
            self.model_name_input.resize(150, 48)
            self.model_name_input.textChanged.connect(lambda: self._update_simple('model_name'))
            self.model_name_card.addWidget(self.model_name_input)
            self.model_name_card.adjustSize()

            # 分辨率
            self.resolution_card = SiOptionCardLinear(self)
            self.resolution_card.setTitle("分辨率", "模型输入输出的图像分辨率（自动对齐到16的倍数）")
            self.resolution_card.load(safe_get_icon("ic_fluent_scan_filled"))
            self.resolution_input = SiLabeledLineEdit(self.resolution_card)
            self.resolution_input.setTitle("整数")
            self.resolution_input.setText("128")
            self.resolution_input.setFixedHeight(48)
            self.resolution_input.resize(150, 48)
            self.resolution_input.textChanged.connect(lambda: self._update_simple('resolution'))
            self.resolution_card.addWidget(self.resolution_input)
            self.resolution_card.adjustSize()

            # ae_dims
            self.ae_dims_card = SiOptionCardLinear(self)
            self.ae_dims_card.setTitle("ae_dims", "自编码器维度（新建时可修改）")
            self.ae_dims_card.load(safe_get_icon("ic_fluent_cube_filled"))
            self.ae_dims_input = SiLabeledLineEdit(self.ae_dims_card)
            self.ae_dims_input.setTitle("整数")
            self.ae_dims_input.setText(str(ae_def))
            self.ae_dims_input.setFixedHeight(48)
            self.ae_dims_input.resize(150, 48)
            self.ae_dims_input.setEnabled(True)
            self.ae_dims_input.textChanged.connect(lambda: self._update_simple('ae_dims'))
            self.ae_dims_card.addWidget(self.ae_dims_input)
            self.ae_dims_card.adjustSize()

            # e_dims
            self.e_dims_card = SiOptionCardLinear(self)
            self.e_dims_card.setTitle("e_dims", "编码器维度（新建时可修改）")
            self.e_dims_card.load(safe_get_icon("ic_fluent_box_filled"))
            self.e_dims_input = SiLabeledLineEdit(self.e_dims_card)
            self.e_dims_input.setTitle("整数")
            self.e_dims_input.setText(str(e_def))
            self.e_dims_input.setFixedHeight(48)
            self.e_dims_input.resize(150, 48)
            self.e_dims_input.setEnabled(True)
            self.e_dims_input.textChanged.connect(lambda: self._update_simple('e_dims'))
            self.e_dims_card.addWidget(self.e_dims_input)
            self.e_dims_card.adjustSize()

            # d_dims
            self.d_dims_card = SiOptionCardLinear(self)
            self.d_dims_card.setTitle("d_dims", "解码器维度（新建时可修改）")
            self.d_dims_card.load(safe_get_icon("ic_fluent_arrow_upload_filled"))
            self.d_dims_input = SiLabeledLineEdit(self.d_dims_card)
            self.d_dims_input.setTitle("整数")
            self.d_dims_input.setText(str(d_def))
            self.d_dims_input.setFixedHeight(48)
            self.d_dims_input.resize(150, 48)
            self.d_dims_input.setEnabled(True)
            self.d_dims_input.textChanged.connect(lambda: self._update_simple('d_dims'))
            self.d_dims_card.addWidget(self.d_dims_input)
            self.d_dims_card.adjustSize()

            group.addWidget(self.model_name_card)
            group.addWidget(self.resolution_card)
            group.addWidget(self.ae_dims_card)
            group.addWidget(self.e_dims_card)
            group.addWidget(self.d_dims_card)

        self.content().setAttachment(self.titled_widget_group)

        # 控制面板
        self.create_button = SiPushButton(self)
        self.create_button.resize(128, 32)
        self.create_button.attachment().setText("创建模型")
        self.create_button.clicked.connect(self._on_create_clicked)

        self.cancel_button = SiPushButton(self)
        self.cancel_button.resize(128, 32)
        self.cancel_button.attachment().setText("取消")
        self.cancel_button.clicked.connect(self.closeParentLayer)

        self.panel().addWidget(self.cancel_button, "left")
        self.panel().addWidget(self.create_button, "right")

        SiGlobal.siui.reloadStyleSheetRecursively(self)

    def _update_simple(self, key):
        """更新简单配置字段"""
        if key == 'model_name':
            self.config_data['model_name'] = self.model_name_input.text()
        elif key == 'resolution':
            self.config_data['resolution'] = self.resolution_input.text()
        elif key == 'ae_dims':
            self.config_data['ae_dims'] = self.ae_dims_input.text()
        elif key == 'e_dims':
            self.config_data['e_dims'] = self.e_dims_input.text()
        elif key == 'd_dims':
            self.config_data['d_dims'] = self.d_dims_input.text()

    def _on_create_clicked(self):
        """点击创建模型按钮"""
        self.model_created.emit(self.config_data)
        self.closeParentLayer()


class XSegTrainingConfigChildPage(SiChildPage):
    """
    XSeg / XSegLite 训练配置子页面
    XSeg 的参数很少，不需要完整的 SAEHD 训练配置页
    """

    def __init__(self, parent=None, model_name="XSegLite",
                 model_info=None, model_dir=""):
        super().__init__(parent)
        self._model_name = model_name
        self._model_dir = model_dir
        _mdl = model_info or {}
        # 如果 model_info 为空，尝试从 _data.dat 读取已保存的配置
        if not _mdl:
            try:
                _dat = Path(model_dir) / "XSegLite_XSegLite_data.dat"
                if _dat.exists():
                    import pickle
                    _raw = pickle.loads(_dat.read_bytes())
                    _opts = _raw.get('options', {})
                    for _k, _v in _opts.items():
                        if not isinstance(_v, (bytes, bytearray)):
                            _mdl[_k] = _v
            except Exception:
                pass

        self.view().setMinimumWidth(900)
        self.content().setTitle(f"XSeg 训练配置 - {model_name}")
        self.content().setPadding(48)

        self.titled_widget_group = SiTitledWidgetGroup(self)

        with self.titled_widget_group as group:
            group.addTitle("数据集参数")

            # 人脸集路径（XSegLite 只用单数据集）
            src_card = SiOptionCardLinear(self)
            src_card.setTitle("人脸集路径", "已对齐的人脸集路径")
            src_card.load(safe_get_icon("ic_fluent_folder_filled"))
            self.src_input = SiLabeledLineEdit(src_card)
            self.src_input.setTitle("路径")
            self.src_input.setPlaceholderText("请输入 src 人脸集路径...")
            project_root = Path(__file__).parent.parent.parent.parent.parent
            self.src_input.setText(str(project_root / "workspace" / "data_src" / "aligned"))
            self.src_input.setFixedHeight(48)
            self.src_input.resize(500, 48)
            src_card.addWidget(self.src_input)
            src_card.adjustSize()

            # 脸型
            ft_card = SiOptionCardLinear(self)
            ft_card.setTitle("脸型", "训练用的人脸类型")
            ft_card.load(safe_get_icon("ic_fluent_person_filled"))
            self.face_type_combo = SiCapsuleComboBox(ft_card)
            self.face_type_combo.setTitle("脸型")
            self.face_type_combo.setFixedWidth(200)
            self.face_type_combo.setMinimumHeight(32)
            self.face_type_combo.setEditable(False)
            self.face_type_combo.addItems(["wf", "f", "head", "hf", "ff"])
            self.face_type_combo.setCurrentText(_mdl.get('face_type', 'wf'))
            ft_card.addWidget(self.face_type_combo)
            ft_card.adjustSize()

            group.addWidget(src_card)
            group.addWidget(ft_card)

            group.addTitle("训练参数")

            # 批次大小
            bs_card = SiOptionCardLinear(self)
            bs_card.setTitle("批次大小", "每次训练的样本数量")
            bs_card.load(safe_get_icon("ic_fluent_table_simple_filled"))
            self.batch_size_input = SiLabeledLineEdit(bs_card)
            self.batch_size_input.setTitle("整数")
            self.batch_size_input.setText(str(_mdl.get('batch_size', '4')))
            self.batch_size_input.setFixedHeight(48)
            self.batch_size_input.resize(150, 48)
            bs_card.addWidget(self.batch_size_input)
            bs_card.adjustSize()

            # BF16 开关
            bf16_card = SiOptionCardLinear(self)
            bf16_card.setTitle("BF16 混合精度", "使用 BF16 混合精度训练（需 Ampere+ GPU）")
            bf16_card.load(safe_get_icon("ic_fluent_number_row_filled"))
            self.use_bf16_switch = SiSwitch(bf16_card)
            self.use_bf16_switch.setChecked(bool(_mdl.get('use_bf16', False)))
            bf16_card.addWidget(self.use_bf16_switch)
            bf16_card.adjustSize()

            # 预训练模式暂时弃用
            pt_card = SiOptionCardLinear(self)
            pt_card.setTitle("预训练模式", "（暂不可用）使用通用人脸数据进行预训练")
            pt_card.load(safe_get_icon("ic_fluent_hat_graduation_filled"))
            self.pretrain_switch = SiSwitch(pt_card)
            self.pretrain_switch.setChecked(False)
            self.pretrain_switch.setEnabled(False)
            pt_card.addWidget(self.pretrain_switch)
            pt_card.adjustSize()

            # 快速加载器开关
            fast_card = SiOptionCardLinear(self)
            fast_card.setTitle("快速加载器", "跳过验证，直接加载全部文件（更快）")
            fast_card.load(safe_get_icon("ic_fluent_flash_filled"))
            self.fast_loader_switch = SiSwitch(fast_card)
            self.fast_loader_switch.setChecked(True)
            fast_card.addWidget(self.fast_loader_switch)
            fast_card.adjustSize()

            # 使用 torch.compile 开关
            compile_card = SiOptionCardLinear(self)
            compile_card.setTitle("使用 torch.compile", "编译模型加速训练（首次需编译耗时）")
            compile_card.load(safe_get_icon("ic_fluent_developer_board_filled"))
            self.use_compile_switch = SiSwitch(compile_card)
            self.use_compile_switch.setChecked(False)
            compile_card.addWidget(self.use_compile_switch)
            compile_card.adjustSize()

            group.addWidget(bs_card)
            group.addWidget(bf16_card)
            group.addWidget(pt_card)
            group.addWidget(fast_card)
            group.addWidget(compile_card)

        self.content().setAttachment(self.titled_widget_group)

        # 控制面板
        self.start_button = SiPushButton(self)
        self.start_button.resize(128, 32)
        self.start_button.attachment().setText("开始训练")
        self.start_button.clicked.connect(self._on_start_training)

        self.cancel_button = SiPushButton(self)
        self.cancel_button.resize(128, 32)
        self.cancel_button.attachment().setText("取消")
        self.cancel_button.clicked.connect(self.closeParentLayer)

        self.panel().addWidget(self.cancel_button, "left")
        self.panel().addWidget(self.start_button, "right")

        SiGlobal.siui.reloadStyleSheetRecursively(self)

    def _on_start_training(self):
        """启动 XSegLite 训练（单数据集，src=dst）"""
        import subprocess, sys
        from pathlib import Path

        src = self.src_input.text().strip()
        if not src:
            print("[ERROR] 人脸集路径不能为空")
            return

        project_root = Path(__file__).parent.parent.parent.parent.parent
        python_exe = sys.executable

        args = [
            python_exe, str(project_root / "main.py"), "train",
            "--model", "XSegLite",
            "--model-dir", self._model_dir,
            "--force-model-name", self._model_name,
            "--training-data-src-dir", src,
            "--training-data-dst-dir", src,
            "--no-preview",
            "--silent-start",
        ]

        # 写入用户配置到 _data.dat
        # XSegLite 的 force_model_class_name 使 model_name = 'XSegLite_XSegLite'
        dat_path = Path(self._model_dir) / "XSegLite_XSegLite_data.dat"
        if dat_path.exists():
            try:
                import pickle
                data = pickle.loads(dat_path.read_bytes())
                if 'options' in data:
                    data['options']['batch_size'] = int(self.batch_size_input.text() or '4')
                    data['options']['use_bf16'] = self.use_bf16_switch.isChecked()
                    data['options']['pretrain'] = False  # pretrain 已停用：强制 False
                    data['options']['face_type'] = self.face_type_combo.currentText()
                    data['options']['loader_skip'] = self.fast_loader_switch.isChecked()
                    data['options']['use_compile'] = self.use_compile_switch.isChecked()
                    tmp = dat_path.with_suffix('.dat.write_tmp')
                    tmp.write_bytes(pickle.dumps(data))
                    tmp.replace(dat_path)
            except Exception as e:
                print(f"[ERROR] 写入 XSeg 配置失败: {e}")

        self.closeParentLayer()

        _cmd_str = subprocess.list2cmdline(args)
        print(f"执行命令: {_cmd_str}")
        _pw = self.window()
        if hasattr(_pw, 'show_command_notification'):
            _pw.show_command_notification(_cmd_str, "训练 XSegLite")

        cmd_line = _cmd_str + " & pause"
        _p = subprocess.Popen(
            ['cmd', '/c', cmd_line],
            creationflags=subprocess.CREATE_NEW_CONSOLE if hasattr(subprocess, 'CREATE_NEW_CONSOLE') else 0,
            cwd=str(project_root),
        )
        def _wait_xseg():
            _p.wait()
            QTimer.singleShot(0, lambda: _done_xseg())
        def _done_xseg():
            try:
                _mw = self.window()
                if _mw and hasattr(_mw, 'show_task_completed_notification'):
                    _mw.show_task_completed_notification("XSegLite 训练已停止")
            except Exception as _ex:
                print(f"[WARN] 通知异常: {_ex}")
            print("✓ XSegLite 训练进程已关闭")
        import threading
        threading.Thread(target=_wait_xseg, daemon=True).start()
