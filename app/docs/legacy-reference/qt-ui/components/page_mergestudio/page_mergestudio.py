"""
DeepFaceLab Torch - MergeStudio Page
合成工作室：启动 MergeStudio Web 服务，管理模型导出
"""

from PyQt5.QtCore import Qt, QTimer, QVariantAnimation, QEasingCurve
from PyQt5.QtWidgets import QLabel, QSizePolicy, QBoxLayout, QApplication
from pathlib import Path
import json, os, subprocess, sys
from datetime import datetime
from contextlib import contextmanager

from siui.components.page import SiPage
from siui.components.container import SiTriSectionPanelCard, SiDenseContainer
from siui.components.editbox import SiLabeledLineEdit
from siui.components.button import SiPushButtonRefactor
from siui.components import SiTitledWidgetGroup
from siui.components.option_card import SiOptionCardLinear
from siui.core import SiGlobal, Si


def safe_get_icon(name):
    try:
        return SiGlobal.siui.iconpack.get(name)
    except KeyError:
        return None


@contextmanager
def createPanelCard(parent: SiTitledWidgetGroup, title: str) -> SiTriSectionPanelCard:
    card = SiTriSectionPanelCard(parent)
    card.setTitle(title)
    try:
        yield card
    finally:
        card.adjustSize()
        parent.addWidget(card)


# ── WebUI 设置持久化 ────────────────────────────────────────────────

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
        print(f"[WARN] 保存 WebUI 设置失败: {e}")


# ── Model scanner ──────────────────────────────────────────────────

def _scan_all_models(model_dir):
    """扫描所有模型目录，返回模型信息列表"""
    results = []
    seen_names = set()

    def _base_name(stem, class_name):
        suffix = f"_{class_name}"
        if stem.endswith(suffix):
            return stem[:-len(suffix)]
        return stem

    def _read_dat(dat_path):
        """从 .meta_cache.json 读取（仅读 json，不碰 _data.dat）"""
        import json
        meta_path = dat_path.with_suffix('.meta_cache.json')
        if meta_path.exists():
            try:
                return json.loads(meta_path.read_text(encoding='utf-8'))
            except Exception:
                pass
        return {'iter': 0, 'resolution': '?', 'face_type': '?',
                'ae_dims': '?', 'e_dims': '?', 'd_dims': '?', 'precision': 'fp32'}

    base = Path(model_dir)

    # 1. .dfm 文件
    if base.exists():
        for f in sorted(base.glob("*.dfm")):
            name = f.stem
            if name not in seen_names:
                seen_names.add(name)
                results.append({
                    'name': name, 'type': 'DFM', 'file_type': 'dfm',
                    'path': str(f), 'iter': '-', 'resolution': '-',
                    'face_type': '-', 'mtime': datetime.fromtimestamp(f.stat().st_mtime),
                })

    # 2. SAEHD _data.dat
    if base.exists():
        for f in sorted(base.glob("*_SAEHD_data.dat")):
            stem = f.name[:-len('_data.dat')]
            parts = stem.rsplit('_', 1)
            if len(parts) != 2 or parts[1] != 'SAEHD':
                continue
            name = parts[0]
            if name in seen_names:
                continue
            seen_names.add(name)
            info = _read_dat(f)
            results.append({
                'name': name, 'type': 'SAEHD', 'file_type': 'pth',
                'path': str(f), **info,
                'mtime': datetime.fromtimestamp(f.stat().st_mtime),
            })

    # 3. DeepFakeLarge _data.dat
    dfl_dir = base / "DeepFakeLarge"
    if dfl_dir.exists():
        for f in sorted(dfl_dir.glob("*_DeepFakeLarge_data.dat")):
            stem = f.name[:-len('_data.dat')]
            parts = stem.rsplit('_', 1)
            if len(parts) != 2 or parts[1] != 'DeepFakeLarge':
                continue
            name = parts[0]
            if name in seen_names:
                continue
            seen_names.add(name)
            info = _read_dat(f)
            results.append({
                'name': name, 'type': 'DeepFakeLarge', 'file_type': 'pth',
                'path': str(f), **info,
                'mtime': datetime.fromtimestamp(f.stat().st_mtime),
            })

    # 4. XSeg / XSegLite — 提示去遮罩绘制页面
    for _xseg_dir, _xseg_type in [("XSeg", "XSeg"), ("XSegLite", "XSegLite")]:
        xd = base / _xseg_dir
        if xd.exists():
            for f in sorted(xd.glob("*.pth")):
                if '_opt' in f.stem or '_star' in f.stem:
                    continue
                name = f.stem
                if name in seen_names:
                    continue
                seen_names.add(name)
                results.append({
                    'name': name, 'type': _xseg_type, 'file_type': 'xseg_skip',
                    'path': str(xd), 'iter': '-', 'resolution': '-',
                    'face_type': '-', 'mtime': datetime.fromtimestamp(f.stat().st_mtime),
                })
            # 检查是否有文件放错了目录
            _other_type = "XSegLite" if _xseg_type == "XSeg" else "XSeg"
            _other_dir = base / _other_type
            if _other_dir.exists():
                for f in sorted(xd.glob("*.pth")):
                    if '_opt' in f.stem or '_star' in f.stem:
                        continue
                    name = f.stem
                    # Check if this model also exists in the other type's dir
                    other_patterns = list(_other_dir.glob(f"{name}_*.pth")) + list(_other_dir.glob(f"{name}.pth"))
                    if other_patterns:
                        print(f"[警告] {name} 同时存在于 {_xseg_dir} 和 {_other_type} 目录中，"
                              f"请确认正确的目录")

    # 5. LIAELarge _data.dat
    ll_dir = base / "LIAELarge"
    if ll_dir.exists():
        for f in sorted(ll_dir.glob("*_LIAELarge_data.dat")):
            stem = f.name[:-len('_data.dat')]
            parts = stem.rsplit('_', 1)
            if len(parts) != 2 or parts[1] != 'LIAELarge':
                continue
            name = parts[0]
            if name in seen_names:
                continue
            seen_names.add(name)
            info = _read_dat(f)
            results.append({
                'name': name, 'type': 'LIAELarge', 'file_type': 'pth',
                'path': str(f), **info,
                'mtime': datetime.fromtimestamp(f.stat().st_mtime),
            })

    # 5. .npy 文件（旧格式 legacy）— 一个模型有多个组件 .npy，只归为一条
    _NPY_MODULE_SUFFIXES = sorted((
        '_encoder', '_decoder_src', '_decoder_dst', '_inter',
        '_inter_AB', '_inter_B', '_inter_src', '_inter_dst',
        '_src_dst_opt', '_D_src', '_D_code_opt',
        '_GAN', '_GAN_opt', '_code_discriminator',
        '_encoder_opt', '_decoder_src_opt', '_decoder_dst_opt',
        '_decoder_src_mask', '_decoder_dst_mask',
    ), key=len, reverse=True)  # 长后缀优先匹配，避免 _inter 匹配到 _inter_AB
    seen_npy_bases = set()
    _npy_search_dirs = [base]
    for npy_dir in _npy_search_dirs:
        if npy_dir.exists():
            for f in sorted(npy_dir.glob("*.npy")):
                raw_stem = f.stem
                base_name = raw_stem
                for sfx in _NPY_MODULE_SUFFIXES:
                    if raw_stem.endswith(sfx):
                        base_name = raw_stem[:-len(sfx)]
                        break
                # 未匹配到模块后缀时，尝试剥离模型类后缀（_SAEHD / _DeepFakeLarge / _LIAELarge）
                _model_class_suffixes = ('_SAEHD', '_PIA', '_DeepFakeLarge', '_LIAELarge', '_DFSingle')
                for _cls_sfx in _model_class_suffixes:
                    if raw_stem.endswith(_cls_sfx):
                        _try_name = raw_stem[:-len(_cls_sfx)]
                        if _try_name in seen_names:
                            base_name = _try_name
                            break
                if base_name in seen_names or base_name in seen_npy_bases:
                    continue
                seen_npy_bases.add(base_name)
                # Detect model type from parent dir name
                parent_name = npy_dir.name  # 'DeepFakeLarge', 'LIAELarge', or 'model'
                if parent_name in ('DeepFakeLarge', 'LIAELarge'):
                    npy_model_type = parent_name
                else:
                    npy_model_type = 'SAEHD'
                results.append({
                    'name': base_name, 'type': npy_model_type, 'file_type': 'npy',
                    'path': str(npy_dir), 'iter': '-', 'resolution': '-',
                    'face_type': '-', 'mtime': datetime.fromtimestamp(max(
                        (f2.stat().st_mtime for f2 in npy_dir.glob(f"{base_name}_*.npy")),
                        default=0)),
                })

    results.sort(key=lambda x: x['mtime'], reverse=True)
    return results


# ── 页面主类 ────────────────────────────────────────────────────────

class MergeStudioPage(SiPage):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.setPadding(64)
        self.setScrollMaximumWidth(1000)
        self.setScrollAlignment(Qt.AlignLeft)
        self.setTitle("合成工作室")

        self.titled_widgets_group = SiTitledWidgetGroup(self)
        self.titled_widgets_group.setSpacing(32)
        self.titled_widgets_group.setAdjustWidgetsSize(True)

        # 项目根目录
        import inspect
        self._project_root = Path(__file__).parent.parent.parent.parent
        self._model_dir = str(self._project_root / "workspace" / "model")

        # ═══════ WebUI 设置 ═══════
        with self.titled_widgets_group as group:
            group.addTitle("WebUI 设置")

            with createPanelCard(group, "MergeStudio 服务") as card:
                container = SiDenseContainer(card.body())
                container.layout().setDirection(QBoxLayout.TopToBottom)
                container.layout().setSpacing(12)
                container.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)

                # 端口
                port_input = SiLabeledLineEdit(card)
                port_input.setTitle("服务端口")
                port_input.setPlaceholderText("默认: 8000")
                saved_port = _load_webui_settings().get('mergestudio_port', '8000')
                port_input.setText(saved_port)
                port_input.resize(700, 64)
                self._port = saved_port
                port_input.textChanged.connect(
                    lambda t: (_save_webui_setting('mergestudio_port', t or '8000'),
                               setattr(self, '_port', t or '8000')))
                container.addWidget(port_input)

                # 启动按钮
                btn_container = SiDenseContainer(card.body())
                btn_container.layout().setDirection(QBoxLayout.LeftToRight)
                btn_container.layout().setSpacing(12)

                self.launch_btn = SiPushButtonRefactor(card)
                self.launch_btn.setText("启动 MergeStudio")
                self.launch_btn.setSvgIcon(safe_get_icon("ic_fluent_play_filled"))
                self.launch_btn.adjustSize()
                self.launch_btn.clicked.connect(self._on_launch_mergestudio)
                btn_container.addWidget(self.launch_btn)

                card.body().addWidget(container)
                card.body().addWidget(btn_container)

        # ═══════ 模型列表 ═══════
        with self.titled_widgets_group as group:
            group.addTitle("模型列表")

            with createPanelCard(group, "模型（点击右侧按钮导出为 DFM）") as card:
                card.setMinimumWidth(1000)

                models_container = SiDenseContainer(card.body())
                models_container.layout().setDirection(QBoxLayout.TopToBottom)
                models_container.layout().setSpacing(16)

                self.models_container = models_container
                self.models_card = card

                # 在添加到 body 之前先填充数据（和训练器页面完全一样）
                self._rebuild_model_list()

                card.body().addWidget(models_container)

        self.setAttachment(self.titled_widgets_group)

        self._last_model_files = ''
        self._model_watch_timer = QTimer(self)
        self._model_watch_timer.setInterval(2000)
        self._model_watch_timer.timeout.connect(self._check_model_files)
        QTimer.singleShot(3000, self._start_model_watch)

    def showEvent(self, event):
        super().showEvent(event)
        self._check_model_files()
        self.titled_widgets_group.adjustSize()
        self.titled_widgets_group.updateGeometry()
        self.update()

    # ── 启动 MergeStudio ──────────────────────────────────────────────

    def _on_launch_mergestudio(self):
        """在新控制台启动 MergeStudio"""
        port = getattr(self, '_port', '8000')
        _cmd_short = f'python -m MergeStudio (port {port})'
        print(f"执行命令: {_cmd_short}")
        _pw = self.window()
        if hasattr(_pw, 'show_command_notification'):
            _pw.show_command_notification(_cmd_short, "启动 MergeStudio")
        cmd = (
            f'{sys.executable} -m MergeStudio'
            f' & echo. & echo MergeStudio 已在端口 {port} 启动，按任意键关闭... & pause'
        )
        try:
            import threading
            p = subprocess.Popen(
                ['cmd', '/c', cmd],
                creationflags=subprocess.CREATE_NEW_CONSOLE,
                cwd=str(self._project_root),
            )
            def _wait():
                p.wait()
                from PyQt5.QtCore import QTimer
                QTimer.singleShot(0, lambda: _done())
            def _done():
                try:
                    _mw = self.window()
                    if _mw and hasattr(_mw, 'show_task_completed_notification'):
                        _mw.show_task_completed_notification("MergeStudio 已关闭")
                except Exception as _e:
                    print(f"[WARN] 通知异常: {_e}")
                print("✓ MergeStudio 已关闭")
            threading.Thread(target=_wait, daemon=True).start()
        except Exception as e:
            print(f"[ERROR] 启动 MergeStudio 失败: {e}")

    # ── 文件变更检测 ────────────────────────────────────────────────

    def _start_model_watch(self):
        """延迟启动文件监控"""
        self._record_model_files()
        self._model_watch_timer.start()

    def _record_model_files(self):
        """用目录 mtime 做快照（极快）"""
        base = Path(self._model_dir)
        if base.exists():
            mtimes = []
            for p in [base, base/"DeepFakeLarge", base/"LIAELarge", base/"XSeg", base/"XSegLite"]:
                if p.exists():
                    mtimes.append(str(p.stat().st_mtime))
            self._last_model_files = '|'.join(mtimes)
        else:
            self._last_model_files = ''

    def _check_model_files(self):
        """快速比较目录 mtime，有变更时刷新列表"""
        if not hasattr(self, '_last_model_files'):
            return
        base = Path(self._model_dir)
        if not base.exists():
            return
        mtimes = []
        for p in [base, base/"DeepFakeLarge", base/"LIAELarge", base/"XSeg", base/"XSegLite"]:
            if p.exists():
                mtimes.append(str(p.stat().st_mtime))
        current = '|'.join(mtimes)
        if current == self._last_model_files:
            return
        self._last_model_files = current
        print("[MergeStudio] 检测到 model 目录变更，刷新列表")
        self._rebuild_model_list()
        self.titled_widgets_group.adjustSize()
        self.titled_widgets_group.updateGeometry()

    # ── 模型列表 ────────────────────────────────────────────────────

    def _rebuild_model_list(self, models=None):
        # 清除旧卡片（与训练器页面同样的方式）
        lay = self.models_container.layout()
        for i in range(lay.count() - 1, -1, -1):
            w = lay.itemAt(i).widget()
            if w and isinstance(w, (SiOptionCardLinear, QLabel)):
                lay.removeWidget(w)
                w.deleteLater()

        if not models:
            models = _scan_all_models(self._model_dir)

        if not models:
            placeholder = QLabel("未检测到模型\n请先在训练器中训练模型")
            placeholder.setAlignment(Qt.AlignCenter)
            placeholder.setStyleSheet("color: #888; padding: 32px;")
            self.models_container.addWidget(placeholder)
        else:
            for i, m in enumerate(models):
                card = self._create_model_card(m)
                lay.insertWidget(i, card)

        # 激活布局
        lay.invalidate()
        lay.activate()

    def _create_model_card(self, info):
        """根据模型类型创建卡片"""
        name = info.get('name', '?')
        mtype = info.get('type', '?')
        resolution = info.get('resolution', '-')
        face_type = info.get('face_type', '-')
        training_iter = info.get('iter', 0)
        file_type = info.get('file_type', 'pth')

        if file_type == 'dfm':
            subtitle = f"DFM 已导出 | {mtype}"
            btn_text = "逆导出为 pth"
            btn_icon = "ic_fluent_arrow_repeat_all_filled"
            btn_style = "dfm"
        elif file_type == 'xseg_skip':
            subtitle = f"XSeg 遮罩模型 | 请到遮罩绘制页面导出 DFM"
            btn_text = "前往遮罩绘制"
            btn_icon = "ic_fluent_eyedropper_filled"
            btn_style = "xseg_skip"
        elif file_type == 'npy':
            subtitle = f"NPY 旧权重 | 类型: {mtype}"
            btn_text = "转换并导出 DFM"
            btn_icon = "ic_fluent_arrow_export_filled"
            btn_style = "npy"
        else:
            subtitle = (f"分辨率: {resolution} | 脸型: {face_type} | "
                        f"精度: {info.get('precision', 'fp32')} | 迭代: {training_iter}")
            btn_text = "导出为 DFM"
            btn_icon = "ic_fluent_arrow_export_filled"
            btn_style = "export"

        card = SiOptionCardLinear(self.models_card)
        card.setTitle(f"{name}  ({mtype})", subtitle)
        card.load(safe_get_icon("ic_fluent_box_multiple_filled"))
        card.setGraphicsEffect(None)  # 防止 QGraphicsOpacityEffect 滚动区 bug

        # 右侧操作按钮
        action_btn = SiPushButtonRefactor(card)
        action_btn.setText(btn_text)
        if btn_icon:
            ico = safe_get_icon(btn_icon)
            if ico:
                action_btn.setSvgIcon(ico)
        action_btn.adjustSize()

        if btn_style == "disabled":
            action_btn.setEnabled(False)
        elif btn_style == "xseg_skip":
            _ico_xseg = safe_get_icon("ic_fluent_eyedropper_filled")
            if _ico_xseg:
                action_btn.setSvgIcon(_ico_xseg)
            action_btn.clicked.connect(self._on_go_to_mask_processor)
        elif btn_style == "dfm":
            action_btn.clicked.connect(lambda checked=False, i=info: self._on_reverse_export(i))
        elif btn_style == "npy":
            action_btn.clicked.connect(lambda checked=False, i=info: self._on_export_dfm(i))
        elif btn_style == "export":
            action_btn.clicked.connect(lambda checked=False, i=info: self._on_export_dfm(i))

        card.addWidget(action_btn)
        card.adjustSize()
        return card

    # ── 导出 DFM ────────────────────────────────────────────────────

    def _on_export_dfm(self, info):
        """导出模型为 DFM（ONNX）"""
        mtype = info.get('type', 'SAEHD')
        name = info.get('name', 'unknown')

        if mtype in ('DeepFakeLarge', 'LIAELarge'):
            model_dir = str(Path(self._model_dir) / mtype)
        else:
            model_dir = self._model_dir

        args = [sys.executable, 'tools/export_dfm.py',
                '--model', mtype, '--model-dir', model_dir, '--force-name', name]
        cmd_str = subprocess.list2cmdline(args) + " & echo. & echo 导出完成！按任意键关闭... & pause"
        print(f"\n导出 DFM: {name} ({mtype})")
        _pw = self.window()
        if hasattr(_pw, 'show_command_notification'):
            _pw.show_command_notification(f"{mtype} DFM 导出", f"{name} ({mtype})")

        try:
            import threading
            p = subprocess.Popen(
                ['cmd', '/c', cmd_str],
                creationflags=subprocess.CREATE_NEW_CONSOLE,
                cwd=str(self._project_root),
            )
            _mw = self.window()
            def _wait_dfm():
                p.wait()
                from PyQt5.QtCore import QTimer
                QTimer.singleShot(0, lambda: _done_dfm())
            def _done_dfm():
                try:
                    if _mw and hasattr(_mw, 'show_task_completed_notification'):
                        _mw.show_task_completed_notification(f"{name} DFM 导出完成")
                except Exception:
                    pass
                print(f"✓ {name} ({mtype}) DFM 导出完成")
            threading.Thread(target=_wait_dfm, daemon=True).start()
        except Exception as e:
            print(f"[ERROR] 导出失败: {e}")

        def _on_reverse_export(self, info):
            """DFM → PTH (已禁用)"""
            from PyQt5.QtWidgets import QMessageBox
            QMessageBox.information(self, "DFM → PTH",
                "由于某些原因，暂时不提供此功能，如需要请联系开发者处理——菜级玩家")
    
    def _on_go_to_mask_processor(self):
        """跳转到遮罩绘制页面"""
        try:
            from siui.core import SiGlobal
            main_win = SiGlobal.siui.windows.get("MAIN_WINDOW")
            if main_win and hasattr(main_win, 'layerMain'):
                sc = main_win.layerMain().page_view.stacked_container
                for i in range(sc.widgetsAmount()):
                    w = sc.widgets[i]
                    if 'MaskProcessor' in type(w).__name__:
                        sc.setCurrentIndex(i)
                        return
        except Exception as e:
            print(f"[ERROR] 跳转页面失败: {e}")
