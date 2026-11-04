"""
TRT 编译工具页面 — 全链路 TRT 引擎编译
"""
import sys, threading, subprocess, time, os
from pathlib import Path
from PyQt5.QtCore import Qt, QObject, pyqtSignal
from PyQt5.QtWidgets import QVBoxLayout, QHBoxLayout, QTextEdit, QWidget

from siui.components.page import SiPage
from siui.components import SiTitledWidgetGroup
from siui.components.container import SiTriSectionPanelCard
from siui.components.button import SiPushButtonRefactor
from siui.components.progress_bar import SiProgressBar
from siui.core import SiGlobal, SiColor

_BG = "#1C191F"
_TEXT = "#FFFFFF"
_TEXT_DIM = "#999999"
_ACCENT = "#7C6FF7"


class _CompileSignals(QObject):
    """跨线程信号转发"""
    log_line = pyqtSignal(str)
    progress_val = pyqtSignal(object)      # 0.0 ~ 1.0 float
    status_msg = pyqtSignal(str)           # 状态文本
    finished = pyqtSignal(bool)            # True=成功, False=失败
    model_done = pyqtSignal(str, bool)     # 模型名, 成功与否


_MODEL_LIST = [
    ("ArcFace",          "modelhub/onnx/ArcFace/w600k_mbf.onnx",           "w600k_mbf"),
    ("ArcFace R50",      "modelhub/onnx/ArcFace/w600k_r50.onnx",          "w600k_r50"),
    ("GenderAge",        "modelhub/onnx/GenderAge/genderage.onnx",        "genderage"),
    ("FAN 2D",           "modelhub/onnx/FAN/2DFAN.onnx",                  "2DFAN"),
    ("FAN 3D",           "modelhub/onnx/FAN/3DFAN.onnx",                  "3DFAN"),
    ("InsightFace3D68",  "modelhub/onnx/InsightFace3D68/1k3d68.onnx",    "1k3d68"),
    ("S3FD",             "modelhub/onnx/S3FD/S3FD.onnx",                  "S3FD"),
    ("CenterFace",       "modelhub/onnx/CenterFace/CenterFace.onnx",      "CenterFace"),
    ("FaceMesh",         "modelhub/onnx/FaceMesh/FaceMesh.onnx",          "FaceMesh"),
    ("InsightFace2D106", "modelhub/onnx/InsightFace2d106/InsightFace2D106.onnx", "InsightFace2D106"),
    ("PFLD",             "modelhub/onnx/PFLD/pfld.onnx",                  "pfld"),
    ("MobileFaceNet",    "modelhub/onnx/MobileFaceNet/landmark_detection_56_se_external.onnx", "landmark_detection_56_se_external"),
    ("LightweightFD",    "modelhub/onnx/LightweightFD/face_det_lite.onnx", "face_det_lite"),
    ("YoloV5Face",       "modelhub/onnx/YoloV5Face/YoloV5Face.onnx",     "YoloV5Face"),
    ("YoloV8Face",       "modelhub/onnx/YoloV8Face/yolov8n-face.onnx",   "yolov8n-face"),
    ("YoloV11nFace",     "modelhub/onnx/YoloV11nFace/yolov11n-face.onnx","yolov11n-face"),
    ("RetinaFace 10g",   "modelhub/onnx/RetinaFace/det_10g.onnx",        "det_10g"),
    ("RetinaFace 500m",  "modelhub/onnx/RetinaFace/det_500m.onnx",       "det_500m"),
    ("DamoFD",           "modelhub/onnx/DamoFD/DamoFD.onnx",              "DamoFD"),
    ("FastFaceAlign",    "modelhub/onnx/FastFaceAlign/FastFaceAlign.onnx","FastFaceAlign"),
    ("TinyMog",          "modelhub/onnx/TinyMog/TinyMog.onnx",            "TinyMog"),
    ("ULFD",             "modelhub/onnx/ULFD/ULFD.onnx",                  "ULFD"),
    ("FaceEnhancer",     "modelhub/onnx/FaceEnhancer/FaceEnhancer.onnx",  "FaceEnhancer"),
    ("LIA",              "modelhub/onnx/LIA/generator.onnx",              "generator"),
    ("BlazeFace",        "modelhub/onnx/BlazeFace/blaze.onnx",            "blaze"),
    ("MogFace",          "modelhub/onnx/MogFace/MogFace.onnx",            "MogFace"),
    ("MTCNN PNet",       "modelhub/onnx/MTCNN/pnet.onnx",                 "pnet"),
    ("MTCNN RNet",       "modelhub/onnx/MTCNN/rnet.onnx",                 "rnet"),
    ("MTCNN ONet",       "modelhub/onnx/MTCNN/onet.onnx",                 "onet"),
    ("FaceParser",       "modelhub/onnx/FaceParser/faceparser.onnx",      "faceparser"),
]


class TRTCompilePage(SiPage):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setPadding(64)
        self.setScrollMaximumWidth(1000)
        self.setScrollAlignment(Qt.AlignLeft)
        self.setTitle("TRT 编译工具")

        self._project_root = Path(__file__).parent.parent.parent.parent
        self._signals = _CompileSignals()
        self._running = False
        self._worker = None

        # ── 构建 UI ──
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(16)

        # 标题区
        _group = SiTitledWidgetGroup(self)
        with _group as g:
            g.addTitle("全链路 TRT 引擎编译")

            # 说明卡片
            _desc_card = SiTriSectionPanelCard(g)
            _desc_card.setTitle("说明")
            _desc = QTextEdit()
            _desc.setReadOnly(True)
            _desc.setMaximumHeight(100)
            _desc.setPlainText(
                "将所有 modelhub 中的 ONNX 模型编译为 BF16 TensorRT 引擎。\n"
                f"共 {len(_MODEL_LIST)} 个模型，需要 NVIDIA GPU + TensorRT 环境。\n"
                "编译完成的引擎会自动保存到各模型目录，应用会在下次启动时自动加载。"
            )
            _desc.setStyleSheet(f"QTextEdit {{ background: transparent; border: none; color: {_TEXT_DIM}; font-size: 12px; }}")
            _desc_card.body().addWidget(_desc)
            _desc_card.adjustSize()
            g.addWidget(_desc_card)

            # 进度条
            _progress_card = SiTriSectionPanelCard(g)
            _progress_card.setTitle("总体进度")

            self._progress = SiProgressBar(_progress_card)
            self._progress.setFixedHeight(20)
            self._progress.setValue(0.0)
            self._progress.setState("processing")
            self._progress.setHint("0%")
            _progress_card.body().addWidget(self._progress)

            self._status_label = QTextEdit()
            self._status_label.setReadOnly(True)
            self._status_label.setMaximumHeight(36)
            self._status_label.setPlainText("就绪")
            self._status_label.setStyleSheet(
                f"QTextEdit {{ background: transparent; border: none; color: {_TEXT}; font-size: 12px; }}")
            _progress_card.body().addWidget(self._status_label)
            _progress_card.adjustSize()
            g.addWidget(_progress_card)

            # 日志输出
            _log_card = SiTriSectionPanelCard(g)
            _log_card.setTitle("编译日志")
            self._log_edit = QTextEdit()
            self._log_edit.setReadOnly(True)
            self._log_edit.setMinimumHeight(300)
            self._log_edit.setStyleSheet(
                f"QTextEdit {{ background: {_BG}; border: none;"
                f"  color: {_TEXT}; font-size: 12px;"
                f"  font-family: 'Consolas', 'Courier New', monospace;"
                f"}}"
                f"QScrollBar:vertical {{ background: transparent; width: 6px; border: none; }}"
                f"QScrollBar::handle:vertical {{ background: #3a3a52; border-radius: 3px; min-height: 30px; }}"
                f"QScrollBar::handle:vertical:hover {{ background: #5a5a72; }}"
                f"QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; }}"
                f"QScrollBar:horizontal {{ height: 0; }}"
            )
            _log_card.body().addWidget(self._log_edit)
            _log_card.adjustSize()
            g.addWidget(_log_card)

            # 按钮
            _btn_widget = QWidget(g)
            _btn_widget.setFixedHeight(60)
            _btn_layout = QHBoxLayout(_btn_widget)
            _btn_layout.setContentsMargins(0, 8, 0, 8)

            self._start_btn = SiPushButtonRefactor(_btn_widget)
            self._start_btn.setText("▶  开始全链路编译")
            self._start_btn.setMinimumWidth(280)
            self._start_btn.setMinimumHeight(40)
            self._start_btn.clicked.connect(self._start_compile)

            self._stop_btn = SiPushButtonRefactor(_btn_widget)
            self._stop_btn.setText("■  停止")
            self._stop_btn.setMinimumWidth(280)
            self._stop_btn.setMinimumHeight(40)
            self._stop_btn.clicked.connect(self._stop_compile)
            self._stop_btn.setVisible(False)

            _btn_layout.addWidget(self._start_btn)
            _btn_layout.addWidget(self._stop_btn)
            _btn_layout.addStretch()
            g.addWidget(_btn_widget)

        self.setAttachment(_group)

        # Signal connections（必须在 widget 创建之后）
        self._signals.log_line.connect(self._append_log)
        self._signals.progress_val.connect(self._on_progress)
        self._signals.status_msg.connect(lambda t: self._status_label.setPlainText(t))
        self._signals.finished.connect(self._on_finished)

    def _on_progress(self, val):
        self._progress.setValue(float(val))

    def _append_log(self, text: str):
        self._log_edit.append(text)
        # 自动滚到底部
        sb = self._log_edit.verticalScrollBar()
        sb.setValue(sb.maximum())

    def _set_log(self, text: str):
        self._log_edit.setPlainText(text)

    def _start_compile(self):
        if self._running:
            return
        self._running = True
        self._start_btn.setVisible(False)
        self._stop_btn.setVisible(True)
        self._set_log("")
        self._progress.setValue(0.0)
        self._progress.setState("processing")
        self._status_label.setPlainText("启动编译...")

        self._worker = _CompileWorker(self._project_root, self._signals)
        self._worker.start()

    def _stop_compile(self):
        if self._worker and self._worker.is_alive():
            self._worker.stop()
            self._append_log("\n⚠ 用户中断编译")
            self._on_finished(False)

    def _on_finished(self, success: bool):
        self._running = False
        self._start_btn.setVisible(True)
        self._stop_btn.setVisible(False)
        if success:
            self._progress.setState("completing")
            self._status_label.setPlainText("✓ 所有模型编译完成")
        else:
            self._status_label.setPlainText("⚠ 编译已中断/失败")


class _CompileWorker(threading.Thread):
    """后台编译工作线程"""

    def __init__(self, project_root, signals: _CompileSignals):
        super().__init__(daemon=True)
        self._root = project_root
        self._signals = signals
        self._cancelled = False

    def stop(self):
        self._cancelled = True

    def run(self):
        total = len(_MODEL_LIST)
        python_exe = sys.executable
        export_tool = str(self._root / "tools" / "export_onnx_to_trt.py")

        for idx, (label, onnx_rel, name) in enumerate(_MODEL_LIST):
            if self._cancelled:
                self._signals.finished.emit(False)
                return

            onnx_path = self._root / onnx_rel
            if not onnx_path.exists():
                self._signals.log_line.emit(f"[{idx+1}/{total}] ⏭ {label}: ONNX 不存在 ({onnx_rel})")
                self._signals.progress_val.emit((idx + 1) / total)
                continue

            self._signals.log_line.emit(f"[{idx+1}/{total}] 🔧 编译 {label} ({name})...")
            self._signals.status_msg.emit(f"编译 {label} ({idx+1}/{total})")

            cmd = [python_exe, export_tool, str(onnx_path), "--name", name]
            try:
                proc = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    cwd=str(self._root),
                )
                for line in proc.stdout:
                    if self._cancelled:
                        proc.kill()
                        self._signals.finished.emit(False)
                        return
                    line = line.rstrip()
                    if line:
                        self._signals.log_line.emit(f"  {line}")

                proc.wait()
                if proc.returncode == 0:
                    self._signals.log_line.emit(f"  ✓ {label} 编译成功")
                else:
                    self._signals.log_line.emit(f"  ✗ {label} 编译失败 (exit={proc.returncode})")
            except Exception as e:
                self._signals.log_line.emit(f"  ✗ {label} 异常: {e}")

            self._signals.progress_val.emit((idx + 1) / total)

        if not self._cancelled:
            self._signals.log_line.emit("\n=== 全链路编译完成 ===")
            self._signals.finished.emit(True)
