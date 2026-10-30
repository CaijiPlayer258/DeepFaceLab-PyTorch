"""
DeepFaceLab Torch - MaskProcessor Page
遮罩绘制：启动 MaskProcessor Web 服务
"""

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QSizePolicy, QBoxLayout, QLabel, QApplication
from pathlib import Path
import json, os, subprocess, sys

from siui.components.page import SiPage
from siui.components.container import SiTriSectionPanelCard, SiDenseContainer
from siui.components.editbox import SiLabeledLineEdit
from siui.components.button import SiPushButtonRefactor
from siui.components import SiTitledWidgetGroup, SiSwitch
from siui.components.combobox_ import SiCapsuleComboBox
from siui.core import SiGlobal

# Theme colors matching main window
_BG = "#1C191F"
_BG_CARD = "#2a2733"
_TEXT = "#FFFFFF"
_TEXT_DIM = "#999999"
_THEME = "#855198"

def safe_get_icon(name):
    try:
        return SiGlobal.siui.iconpack.get(name)
    except KeyError:
        return None


_WEBUI_JSON = Path(__file__).parent.parent.parent.parent / "workspace" / "webui_settings.json"

def _load_webui_settings():
    if _WEBUI_JSON.exists():
        try:
            return json.loads(_WEBUI_JSON.read_text(encoding='utf-8'))
        except Exception:
            pass
    return {}

def _save_webui_setting(key, value):
    try:
        data = _load_webui_settings()
        data[key] = value
        _WEBUI_JSON.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
    except Exception as e:
        print(f"[WARN] 保存设置失败: {e}")


class MaskProcessorPage(SiPage):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.setPadding(64)
        self.setScrollMaximumWidth(1000)
        self.setScrollAlignment(Qt.AlignLeft)
        self.setTitle("遮罩绘制")

        self._project_root = Path(__file__).parent.parent.parent.parent
        self._project_root_s = str(self._project_root)

        self.titled_widgets_group = SiTitledWidgetGroup(self)
        self.titled_widgets_group.setSpacing(32)
        self.titled_widgets_group.setAdjustWidgetsSize(True)

        with self.titled_widgets_group as group:
            group.addTitle("WebUI 设置")

            from contextlib import contextmanager
            @contextmanager
            def _make_card(title):
                c = SiTriSectionPanelCard(group)
                c.setTitle(title)
                try:
                    yield c
                finally:
                    c.adjustSize()
                    group.addWidget(c)

            with _make_card("MaskProcessor 服务") as card:
                container = SiDenseContainer(card.body())
                container.layout().setDirection(QBoxLayout.TopToBottom)
                container.layout().setSpacing(12)
                container.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)

                port_input = SiLabeledLineEdit(card)
                port_input.setTitle("服务端口")
                port_input.setPlaceholderText("默认: 8000")
                saved_port = _load_webui_settings().get('maskprocessor_port', '8000')
                port_input.setText(saved_port)
                port_input.resize(700, 64)
                self._port = saved_port
                port_input.textChanged.connect(
                    lambda t: (_save_webui_setting('maskprocessor_port', t or '8000'),
                               setattr(self, '_port', t or '8000')))
                container.addWidget(port_input)

                btn_container = SiDenseContainer(card.body())
                btn_container.layout().setDirection(QBoxLayout.LeftToRight)
                btn_container.layout().setSpacing(12)

                self.launch_btn = SiPushButtonRefactor(card)
                self.launch_btn.setText("启动 MaskProcessor")
                self.launch_btn.setSvgIcon(safe_get_icon("ic_fluent_play_filled"))
                self.launch_btn.adjustSize()
                self.launch_btn.clicked.connect(self._on_launch)
                btn_container.addWidget(self.launch_btn)

                card.body().addWidget(container)
                card.body().addWidget(btn_container)

        # ═══════ ONNX 导出 ═══════
        with self.titled_widgets_group as group:
            group.addTitle("ONNX 导出")

            with _make_card("导出 XSeg / XSegLite 为 ONNX") as card:
                cont = SiDenseContainer(card.body())
                cont.layout().setDirection(QBoxLayout.LeftToRight)
                cont.layout().setSpacing(12)
                cont.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)

                btn_xseg = SiPushButtonRefactor.withText("导出 XSeg → ONNX", card)
                btn_xseg.clicked.connect(lambda: self._on_export_onnx('XSeg'))
                cont.addWidget(btn_xseg)

                btn_xseglite = SiPushButtonRefactor.withText("导出 XSegLite → ONNX", card)
                btn_xseglite.clicked.connect(lambda: self._on_export_onnx('XSegLite'))
                cont.addWidget(btn_xseglite)

                card.body().addWidget(cont)

        # ═══════ XSeg 遮罩应用 ═══════
        with self.titled_widgets_group as group:
            group.addTitle("XSeg 遮罩应用")

            with _make_card("遮罩应用") as card:
                container = SiDenseContainer(card.body())
                container.layout().setDirection(QBoxLayout.TopToBottom)
                container.layout().setSpacing(12)
                container.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)

                # 输入路径
                path_input = SiLabeledLineEdit(card)
                path_input.setTitle("人脸集路径")
                path_input.setPlaceholderText("请输入已对齐的人脸集目录路径...")
                path_input.setText(str(self._project_root / "workspace" / "data_src" / "aligned"))
                path_input.resize(700, 64)
                self._xseg_path = path_input
                container.addWidget(path_input)

                # 模型类型和分辨率（同一行）
                row = SiDenseContainer(card.body())
                row.layout().setDirection(QBoxLayout.LeftToRight)
                row.layout().setSpacing(12)

                from siui.components.combobox_ import SiCapsuleComboBox
                model_combo = SiCapsuleComboBox(card)
                model_combo.setTitle("模型类型")
                model_combo.setMinimumHeight(36)
                model_combo.setEditable(False)
                model_combo.addItems(["XSegLite", "XSeg"])
                model_combo.setCurrentText("XSegLite")
                model_combo.setToolTip("选择遮罩模型类型")
                model_combo.currentIndexChanged.connect(self._on_mask_model_changed)
                self._xseg_model = model_combo
                row.addWidget(model_combo)

                # XSegLite 模型文件选择（仅 XSegLite 时显示）
                self._xseg_file_combo = SiCapsuleComboBox(card)
                self._xseg_file_combo.setTitle("模型文件")
                self._xseg_file_combo.setMinimumHeight(36)
                self._xseg_file_combo.setEditable(False)
                self._refresh_xseg_file_list()
                self._xseg_file_combo.setToolTip("选择 XSegLite 模型文件\n.engine = TRT BF16 (最快)\n.onnx = ONNX")
                row.addWidget(self._xseg_file_combo)

                res_input = SiLabeledLineEdit(card)
                res_input.setTitle("分辨率")
                res_input.setPlaceholderText("默认 256")
                res_input.setText("256")
                res_input.setFixedHeight(48)
                res_input.resize(100, 48)
                self._xseg_res = res_input
                row.addWidget(res_input)

                container.addWidget(row)
                card.body().addWidget(container)

                # 反转 + 应用按钮（同一行）
                btn_row = SiDenseContainer(card.body())
                btn_row.layout().setDirection(QBoxLayout.LeftToRight)
                btn_row.layout().setSpacing(12)

                invert_combo = SiCapsuleComboBox(card)
                invert_combo.setTitle("反转遮罩")
                invert_combo.setMinimumHeight(36)
                invert_combo.setEditable(False)
                invert_combo.addItems(["否", "是"])
                invert_combo.setCurrentText("否")
                self._xseg_invert = invert_combo
                btn_row.addWidget(invert_combo)

                apply_btn = SiPushButtonRefactor.withText("应用遮罩", card)
                apply_btn.clicked.connect(self._on_apply_xseg)
                btn_row.addWidget(apply_btn)

                card.body().addWidget(btn_row)

        self.setAttachment(self.titled_widgets_group)

    def showEvent(self, event):
        super().showEvent(event)
        # 强制刷新布局，避免切换页面时显示旧内容
        self.titled_widgets_group.adjustSize()
        self.titled_widgets_group.updateGeometry()
        self.update()

    def _on_export_onnx(self, model_type):
        """在新控制台运行 ONNX 导出脚本"""
        script_map = {'XSeg': 'models/Model_XSeg/export_XSeg.py',
                      'XSegLite': 'models/Model_XSegLite/export_XSegLite.py'}
        script = script_map.get(model_type)
        if not script:
            return

        cmd_str = f'{sys.executable} {script}'
        print(f"执行命令: {cmd_str}")
        parent_window = self.window()
        if hasattr(parent_window, 'show_command_notification'):
            parent_window.show_command_notification(cmd_str, f"导出 {model_type} → ONNX")

        full_cmd = f'{cmd_str} & echo. & echo 导出完成！按任意键关闭... & pause'
        try:
            p = subprocess.Popen(
                ['cmd', '/c', full_cmd],
                creationflags=subprocess.CREATE_NEW_CONSOLE,
                cwd=str(self._project_root),
            )
            self._monitor_process(p, f"{model_type} ONNX 导出完成")
        except Exception as e:
            print(f"[ERROR] 启动导出失败: {e}")

    def _monitor_process(self, process, completed_msg, callback=None):
        """后台线程：等待进程结束，在主窗口显示完成通知"""
        import threading
        def _wait():
            try:
                process.wait()
            except Exception:
                pass
            from PyQt5.QtCore import QTimer
            QTimer.singleShot(0, lambda: _notify())
        def _notify():
            try:
                _mw = self.window()
                if _mw and hasattr(_mw, 'show_task_completed_notification'):
                    _mw.show_task_completed_notification(completed_msg)
            except Exception as _e:
                print(f"[WARN] 通知异常: {_e}")
            print(f"✓ {completed_msg}")
            if callback:
                try:
                    callback()
                except Exception as _ce:
                    print(f"[WARN] 回调异常: {_ce}")
        threading.Thread(target=_wait, daemon=True).start()

    def _refresh_xseg_file_list(self):
        """扫描 workspace/model/XSegLite/ 下的 .onnx 和 .engine 文件"""
        from siui.components.combobox_ import SiCapsuleComboBox
        xseg_dir = self._project_root / "workspace" / "model" / "XSegLite"
        files = []
        if xseg_dir.exists():
            for f in sorted(xseg_dir.iterdir()):
                if f.suffix in ('.onnx', '.engine'):
                    files.append(f.name)
        self._xseg_file_combo.clear()
        if files:
            self._xseg_file_combo.addItems(files)
            # 默认选第一个 .engine（如果有）或第一个 .onnx
            engine_idx = next((i for i, n in enumerate(files) if n.endswith('.engine')), 0)
            self._xseg_file_combo.setCurrentIndex(engine_idx)
        else:
            self._xseg_file_combo.addItems(["（未检测到模型文件）"])

    def _on_mask_model_changed(self, idx):
        """模型类型切换时显示/隐藏 XSegLite 模型文件选择"""
        is_xseglite = self._xseg_model.currentText() == "XSegLite"
        self._xseg_file_combo.setVisible(is_xseglite)
        if is_xseglite:
            self._refresh_xseg_file_list()

    def _on_apply_xseg(self):
        """在新控制台运行 XSeg 遮罩应用"""
        path = self._xseg_path.text().strip()
        model = self._xseg_model.currentText()
        res = self._xseg_res.text().strip() or "256"
        if not path:
            print("[ERROR] 请输入人脸集路径")
            return

        invert_flag = " --invert" if self._xseg_invert.currentText() == "是" else ""
        model_file = ""
        if model == "XSegLite":
            sel = self._xseg_file_combo.currentText()
            if sel and not sel.startswith("（"):
                model_file = sel

        trt_flag = ""
        if model_file.endswith('.engine'):
            trt_flag = " --trt"

        cmd_str = (
            f'{sys.executable} -m DataAugmenter -i "{path}" -m {model} -r {res}'
            f'{invert_flag}'
            f'{" --model-file " + model_file if model_file else ""}'
            f'{trt_flag}'
        )

        # 打印命令和右上角提示
        print(f"执行命令: {cmd_str}")
        parent_window = self.window()
        if hasattr(parent_window, 'show_command_notification'):
            parent_window.show_command_notification(cmd_str, f"XSeg 遮罩 - {model}")

        full_cmd = (
            f'{cmd_str}'
            f' & echo. & echo 遮罩应用完成！按任意键关闭... & pause'
        )
        try:
            p = subprocess.Popen(
                ['cmd', '/c', full_cmd],
                creationflags=subprocess.CREATE_NEW_CONSOLE,
                cwd=str(self._project_root),
            )
            self._monitor_process(p, "XSeg 遮罩应用完成")
        except Exception as e:
            print(f"[ERROR] 启动遮罩应用失败: {e}")

    def _on_launch(self):
        """在新控制台启动 MaskProcessor（自动安装依赖后运行）"""
        port = getattr(self, '_port', '8000')
        _cmd_short = f'{sys.executable} -m MaskProcessor (port {port})'
        print(f"执行命令: {_cmd_short}")
        _pw = self.window()
        if hasattr(_pw, 'show_command_notification'):
            _pw.show_command_notification(_cmd_short, "启动 MaskProcessor")
        pip_cmds = (
            f'{sys.executable} -m pip install -q segment-anything transformers onnxruntime onnx '
            f'hydra-core addict supervision timm yapf Pillow ipdb'
        )
        cmd = (
            f'{pip_cmds}'
            f' && {sys.executable} -m MaskProcessor'
            f' & echo. & echo MaskProcessor已在端口 {port} 启动，按任意键关闭... & pause'
        )
        try:
            p = subprocess.Popen(
                cmd,
                shell=True,
                creationflags=subprocess.CREATE_NEW_CONSOLE,
                cwd=str(self._project_root),
            )
            self._monitor_process(p, "MaskProcessor 已关闭")
        except Exception as e:
            print(f"[ERROR] 启动 MaskProcessor 失败: {e}")
