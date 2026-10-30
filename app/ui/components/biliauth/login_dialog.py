"""
Bilibili QR code login dialog — shown before main window on first launch.
Styled to match the app's dark siui theme.
"""

import sys
import threading
from io import BytesIO
from PyQt5.QtCore import Qt, QTimer, QObject, pyqtSignal
from PyQt5.QtGui import QPixmap, QColor
from PyQt5.QtWidgets import (
    QApplication, QDialog, QLabel,
    QVBoxLayout, QHBoxLayout,
)
from siui.components.button import SiPushButtonRefactor

from core.biliauth.qrcode_gen import generate_qrcode
from core.biliauth.login_poll import (
    poll_login, NOT_SCANNED, SCANNED_PENDING,
    SCANNED_CONFIRMED, EXPIRED,
)
from core.biliauth.cookie_store import save_cookie, save_user_info
from core.biliauth.user_info import fetch_user_info
from core.biliauth.follow import follow_user

# Theme colors matching the main window
_BG = "#1C191F"
_BG_CARD = "#2a2733"
_BUTTON = "#2a2733"
_TEXT = "#FFFFFF"
_TEXT_DIM = "#999999"
_THEME = "#855198"
_SUCCESS = "#4CAF50"


class _QRWorker(QObject):
    """后台线程生成二维码，不阻塞 UI。"""
    finished = pyqtSignal(object, object, object)  # (url, key, pil_image)
    error = pyqtSignal(str)

    def run(self):
        try:
            url, key, img = generate_qrcode()
            self.finished.emit(url, key, img)
        except Exception as e:
            self.error.emit(str(e))


class _PollWorker(QObject):
    """后台线程轮询登录状态，不阻塞 UI。"""
    status = pyqtSignal(int, str)  # (status_code, message)
    done = pyqtSignal()

    def __init__(self, qrcode_key):
        super().__init__()
        self._key = qrcode_key
        self._running = True

    def stop(self):
        self._running = False

    def run(self):
        for code, msg in poll_login(self._key):
            if not self._running:
                break
            self.status.emit(code, msg)
            if code in (SCANNED_CONFIRMED, EXPIRED) or code < 0:
                break
        self.done.emit()


class LoginDialog(QDialog):
    """Modal dialog showing a Bilibili QR code for scan-to-login."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("深变:DFL-PyTorch 账号验证")
        self.setFixedSize(360, 460)
        self.setWindowFlags(Qt.WindowStaysOnTopHint | Qt.Dialog)
        self.setModal(True)
        self.setStyleSheet(
            f"background-color: {_BG};"
            "font-family: 'Inter', 'Microsoft YaHei', sans-serif;"
            "font-size: 13px;"
        )

        self._qrcode_key = None
        self._qr_generating = False
        self._poll_worker = None
        self._poll_thread = None

        self._build_ui()
        # 二维码在 showEvent 中延迟生成，避免窗口未显示时设置图片导致空指针

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(16)
        layout.setContentsMargins(32, 24, 32, 24)

        # Title
        title = QLabel("深变:DFL-PyTorch")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet(f"font-size: 18px; font-weight: bold; color: {_TEXT}; background: transparent;")
        layout.addWidget(title)

        subtitle = QLabel("请使用哔哩哔哩 App 扫码登录")
        subtitle.setAlignment(Qt.AlignCenter)
        subtitle.setStyleSheet(f"font-size: 13px; color: {_TEXT_DIM}; background: transparent;")
        layout.addWidget(subtitle)

        # QR code image
        self._qr_label = QLabel()
        self._qr_label.setAlignment(Qt.AlignCenter)
        self._qr_label.setFixedSize(260, 260)
        self._qr_label.setStyleSheet(
            f"border: 2px solid {_BG_CARD}; border-radius: 8px; background: white;"
        )
        layout.addWidget(self._qr_label, alignment=Qt.AlignCenter)

        # Status text
        self._status_label = QLabel("正在生成二维码...")
        self._status_label.setAlignment(Qt.AlignCenter)
        self._status_label.setWordWrap(True)
        self._status_label.setStyleSheet(f"font-size: 12px; color: {_TEXT_DIM}; background: transparent;")
        layout.addWidget(self._status_label)

        # Buttons
        btn_layout = QHBoxLayout()
        btn_layout.setSpacing(12)

        self._refresh_btn = SiPushButtonRefactor.withText("重新生成", self)
        self._refresh_btn.setCursor(Qt.PointingHandCursor)
        self._refresh_btn.clicked.connect(self._generate_new_qrcode)
        btn_layout.addWidget(self._refresh_btn)

        self._exit_btn = SiPushButtonRefactor.withText("退出", self)
        self._exit_btn.setCursor(Qt.PointingHandCursor)
        self._exit_btn.clicked.connect(self._on_exit)
        self._exit_btn.style_data.button_color = QColor("#8B0000")
        self._exit_btn.style_data.text_color = QColor("#FFFFFF")
        self._exit_btn.style_data.idle_color = QColor("#00FFFFFF")
        self._exit_btn.style_data.hover_color = QColor("#22FFFFFF")
        self._exit_btn.update()
        btn_layout.addWidget(self._exit_btn)

        layout.addLayout(btn_layout)

    def _generate_new_qrcode(self):
        """在后台线程生成二维码，不阻塞 UI。"""
        if self._qr_generating:
            return
        self._qr_generating = True
        self._cleanup()
        self._status_label.setText("正在生成二维码...")
        self._refresh_btn.setEnabled(False)

        self._worker = _QRWorker()
        self._worker.finished.connect(self._on_qr_ready)
        self._worker.error.connect(self._on_qr_error)
        t = threading.Thread(target=self._worker.run, daemon=True)
        t.start()

    def _on_qr_ready(self, qrcode_url, qrcode_key, qr_img):
        """二维码生成完成（主线程回调）。"""
        self._qr_generating = False
        self._qrcode_key = qrcode_key

        buf = BytesIO()
        qr_img.save(buf, format='PNG')
        buf.seek(0)
        pixmap = QPixmap()
        pixmap.loadFromData(buf.read())
        pixmap = pixmap.scaled(240, 240, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self._qr_label.setPixmap(pixmap)

        self._status_label.setText("等待扫码...")
        self._refresh_btn.setEnabled(True)

        # 启动后台轮询线程
        self._poll_worker = _PollWorker(qrcode_key)
        self._poll_worker.status.connect(self._on_poll_status)
        self._poll_worker.done.connect(self._on_poll_done)
        self._poll_thread = threading.Thread(target=self._poll_worker.run, daemon=True)
        self._poll_thread.start()

    def _on_qr_error(self, err_msg):
        """二维码生成失败（主线程回调）。"""
        self._qr_generating = False
        self._status_label.setText(f"生成二维码失败:\n{err_msg}")
        self._refresh_btn.setEnabled(True)

    def _on_poll_status(self, code, msg):
        """轮询状态更新（主线程回调）。"""
        if code == SCANNED_CONFIRMED:
            if '||' in msg:
                _, cookie_str = msg.split('||', 1)
                self._on_login_success(cookie_str)
            else:
                self._on_login_success(msg)
        elif code == EXPIRED:
            self._status_label.setText(msg)
        elif code < 0:
            self._status_label.setText(msg)
        else:
            self._status_label.setText(msg)

    def _on_poll_done(self):
        """轮询结束（主线程回调）。"""

    def showEvent(self, event):
        """窗口显示后再生二维码，避免空指针问题"""
        super().showEvent(event)
        if self._qrcode_key is None:
            QTimer.singleShot(0, self._generate_new_qrcode)

    def _on_login_success(self, cookie_str: str):
        """Handle successful login — save cookie, fetch user info, follow devs, then proceed."""
        self._cleanup()
        save_cookie(cookie_str)
        self._status_label.setText("✓ 登录成功！正在加载主页面...")
        QApplication.processEvents()

        # Fetch user info & follow developers in background
        import threading as _t
        def _fetch():
            info = fetch_user_info(cookie_str)
            if info:
                save_user_info(info)
            # 关注主开发者账号（UID 500398541）
            follow_user(500398541, cookie_str)
            QTimer.singleShot(0, self._accept_after_login)
        _t.Thread(target=_fetch, daemon=True).start()

    def _accept_after_login(self):
        self._status_label.setText("✓ 登录成功！正在启动...")
        self._status_label.setStyleSheet("font-size: 12px; color: #4CAF50; font-weight: bold;")
        QTimer.singleShot(800, self.accept)

    def _cleanup(self):
        """停止后台线程"""
        if self._poll_worker is not None:
            self._poll_worker.stop()

    def _on_exit(self):
        """Exit the application."""
        self._cleanup()
        self.reject()

    def closeEvent(self, event):
        self._cleanup()
        event.accept()
