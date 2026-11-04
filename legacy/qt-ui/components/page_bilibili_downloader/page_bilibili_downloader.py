"""
Bilibili 视频下载器 — 输入 UID 获取视频列表，支持单/批量下载
"""
import sys, threading, os, io
from pathlib import Path
from PyQt5.QtCore import Qt, QTimer, QObject, pyqtSignal
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QSizePolicy, QBoxLayout, QWidget, QVBoxLayout, QHBoxLayout

from siui.components.page import SiPage
from siui.components import SiTitledWidgetGroup, SiSwitch
from siui.components.container import SiTriSectionPanelCard, SiDenseContainer
from siui.components.editbox import SiLabeledLineEdit
from siui.components.button import SiPushButtonRefactor
from siui.components.combobox_ import SiCapsuleComboBox
from siui.components.widgets import SiLabel, SiCheckBox
from siui.components.progress_bar import SiProgressBar
from siui.components.spinbox.spinbox import SiIntSpinBox
from siui.core import SiGlobal, SiColor

from core.biliauth.cookie_store import load_cookie
from core.biliauth.video_api import fetch_user_info_by_mid, fetch_up_videos, download_single_video
import requests


class _BilibiliSignals(QObject):
    """跨线程信号中转"""
    user_result = pyqtSignal(object, str)         # info dict, mid
    videos_result = pyqtSignal(object)            # videos list
    avatar_data = pyqtSignal(object)             # QPixmap
    status_msg = pyqtSignal(str)                  # status text
    progress_val = pyqtSignal(float)             # 0.0 ~ 1.0
    # Download manager signals
    worker_progress = pyqtSignal(int, float, float, str)  # worker_id, fraction, speed_mbps, stage
    worker_state = pyqtSignal(int, str, bool)             # worker_id, bvid, is_active
    overall_progress = pyqtSignal(int, int)               # completed, total
    all_done = pyqtSignal()

_BG = "#1C191F"
_TEXT = "#FFFFFF"
_TEXT_DIM = "#999999"


class _WorkerCardWidget(QWidget):
    """单个下载进程的UI卡片: [Worker N] [BVID] [进度条] [速度] [重启]"""

    restart_clicked = pyqtSignal(int)

    def __init__(self, worker_id, parent=None):
        super().__init__(parent)
        self.worker_id = worker_id
        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 4, 8, 4)
        layout.setSpacing(8)

        self._label = SiLabel(self)
        self._label.setFixedWidth(70)
        self._label.setStyleSheet(f"color: {_TEXT}; font-size: 12px;")
        self._label.setText(f"进程 {worker_id + 1}")
        layout.addWidget(self._label)

        self._bvid_label = SiLabel(self)
        self._bvid_label.setFixedWidth(170)
        self._bvid_label.setStyleSheet(f"color: {_TEXT_DIM}; font-size: 11px;")
        self._bvid_label.setText("空闲")
        layout.addWidget(self._bvid_label)

        self._progress = SiProgressBar(self)
        self._progress.setFixedHeight(16)
        self._progress.setFixedWidth(180)
        self._progress.setValue(0.0)
        layout.addWidget(self._progress)

        self._speed_label = SiLabel(self)
        self._speed_label.setFixedWidth(80)
        self._speed_label.setStyleSheet(f"color: {_TEXT_DIM}; font-size: 11px;")
        self._speed_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self._speed_label.setText("0.0 MB/s")
        layout.addWidget(self._speed_label)

        self._restart_btn = SiPushButtonRefactor.withText("重启", self)
        self._restart_btn.setFixedWidth(50)
        self._restart_btn.clicked.connect(self._on_restart)
        self._restart_btn.setToolTip("强制重启当前视频的下载")
        layout.addWidget(self._restart_btn)

        self.set_idle()

    def _on_restart(self):
        self.restart_clicked.emit(self.worker_id)

    def update_progress(self, fraction, speed, stage):
        self._progress.setValue(min(fraction, 1.0))
        if fraction >= 1.0:
            self._progress.setState("completing")
        else:
            self._progress.setState("processing")
        self._speed_label.setText(f"{speed:.1f} MB/s")

    def set_active(self, bvid):
        self._bvid_label.setText(bvid[:20])
        self._bvid_label.setStyleSheet(f"color: {_TEXT}; font-size: 11px;")
        self._restart_btn.setEnabled(True)

    def set_idle(self):
        self._bvid_label.setText("空闲")
        self._bvid_label.setStyleSheet(f"color: {_TEXT_DIM}; font-size: 11px;")
        self._progress.setValue(0.0)
        self._speed_label.setText("0.0 MB/s")
        self._restart_btn.setEnabled(False)


class _DownloadManager:
    """管理并发下载队列和工作进程"""

    def __init__(self, signals: _BilibiliSignals, cookie: str, max_workers: int = 2,
                 ffmpeg_path: str = None):
        self._signals = signals
        self._cookie = cookie
        self._max_workers = max_workers
        self._ffmpeg_path = ffmpeg_path
        self._queue = []       # [(bvid, out_dir), ...]
        self._workers = []     # [_DownloadWorker, ...]
        self._lock = threading.Lock()
        self._completed = 0
        self._total = 0
        self._active = False
        self._pending = {}     # bvid -> out_dir

    def start(self, items, output_dir):
        """启动下载队列"""
        self._queue = [(bvid, output_dir) for bvid, _ in items]
        self._total = len(self._queue)
        self._completed = 0
        self._active = True
        self._pending = {bvid: output_dir for bvid, _ in items}
        actual = min(self._max_workers, self._total)
        self._workers = [
            _DownloadWorker(i, self._cookie, self._signals, self, self._ffmpeg_path)
            for i in range(actual)]
        for w in self._workers:
            w.start()
        self._signals.overall_progress.emit(0, self._total)

    def pop_next(self):
        with self._lock:
            if self._queue:
                bvid, out_dir = self._queue.pop(0)
                return (bvid, out_dir)
            return None

    def requeue(self, bvid, out_dir):
        with self._lock:
            self._queue.insert(0, (bvid, out_dir))

    def is_pending(self, bvid):
        with self._lock:
            return bvid in self._pending

    def notify_completed(self, bvid, success):
        with self._lock:
            self._completed += 1
            self._pending.pop(bvid, None)
        self._signals.overall_progress.emit(self._completed, self._total)
        self._signals.status_msg.emit(f"下载中: {self._completed}/{self._total}")
        if self._completed >= self._total:
            self._active = False
            self._signals.all_done.emit()

    def abort_worker(self, worker_id):
        for w in self._workers:
            if w.worker_id == worker_id:
                w.request_restart()
                break

    def is_active(self):
        return self._active


class _DownloadWorker(threading.Thread):
    """单个下载工作进程"""

    def __init__(self, worker_id, cookie, signals, manager, ffmpeg_path=None):
        super().__init__(daemon=True)
        self.worker_id = worker_id
        self._cookie = cookie
        self._signals = signals
        self._manager = manager
        self._ffmpeg_path = ffmpeg_path
        self._restart_flag = False
        self._lock = threading.Lock()

    def request_restart(self):
        with self._lock:
            self._restart_flag = True

    def _check_restart(self):
        with self._lock:
            if self._restart_flag:
                self._restart_flag = False
                return True
            return False

    def run(self):
        while self._manager.is_active():
            item = self._manager.pop_next()
            if item is None:
                break
            bvid, out_dir = item
            self._signals.worker_state.emit(self.worker_id, bvid, True)

            # Progress callback: emits signal, checks restart flag
            def _cb(fraction, speed, stage):
                if self._check_restart():
                    return False
                self._signals.worker_progress.emit(self.worker_id, fraction, speed, stage)
                return True

            try:
                ok = download_single_video(bvid, self._cookie, out_dir,
                                           progress_callback=_cb,
                                           ffmpeg_path=self._ffmpeg_path)
            except Exception as e:
                print(f"[Worker {self.worker_id}] 异常: {e}")
                ok = False

            if self._check_restart():
                # User requested restart → re-queue
                self._manager.requeue(bvid, out_dir)
                self._signals.worker_state.emit(self.worker_id, "", False)
                continue

            self._manager.notify_completed(bvid, ok)
            self._signals.worker_state.emit(self.worker_id, "", False)
            import time
            time.sleep(0.05)


class BilibiliDownloaderPage(SiPage):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setPadding(64)
        self.setScrollMaximumWidth(1000)
        self.setScrollAlignment(Qt.AlignLeft)
        self.setTitle("B站视频下载")

        self._project_root = Path(__file__).parent.parent.parent.parent
        self._cookie = load_cookie()
        self._videos = []         # [{bvid, title, is_coop, widget, cb}, ...]
        self._list_content = None  # replaced on each rebuild
        self._worker_cards = []    # download worker UI cards

        # Cross-thread signals (Qt.QueuedConnection auto for thread boundary)
        self._bs = _BilibiliSignals()
        self._bs.user_result.connect(self._on_user_result)
        self._bs.videos_result.connect(self._on_videos_result)
        self._bs.avatar_data.connect(self._set_avatar)
        self._bs.status_msg.connect(self._set_status)
        self._build_ui()
        self._bs.progress_val.connect(self._progress_bar.setValue)
        self._bs.worker_progress.connect(self._on_worker_progress)
        self._bs.worker_state.connect(self._on_worker_state)
        self._bs.overall_progress.connect(self._on_overall_progress)
        self._bs.all_done.connect(self._on_all_done)

    def _build_ui(self):
        _g = SiTitledWidgetGroup(self)
        self._titled_group = _g

        # ======== 用户搜索 ========
        with _g as g:
            g.addTitle("用户搜索")
            _card = SiTriSectionPanelCard(g)
            self._search_card = _card
            _card.setTitle("输入 UP 主 UID")
            _c = SiDenseContainer(_card.body())
            _c.layout().setDirection(QBoxLayout.TopToBottom)
            _c.layout().setSpacing(12)

            _row1 = QWidget()
            _row1_l = QHBoxLayout(_row1)
            _row1_l.setContentsMargins(0, 0, 0, 0)

            self._uid_input = SiLabeledLineEdit(_card)
            self._uid_input.setTitle("UID")
            self._uid_input.setPlaceholderText("请输入哔哩哔哩 UID（纯数字）...")
            self._uid_input.setFixedHeight(48)
            self._uid_input.textChanged.connect(self._on_uid_changed)
            # 过滤非数字字符
            self._uid_input.textChanged.connect(self._filter_uid_input)
            _row1_l.addWidget(self._uid_input, 1)
            _c.addWidget(_row1)

            self._user_info_widget = QWidget()
            _user_layout = QHBoxLayout(self._user_info_widget)
            _user_layout.setContentsMargins(0, 0, 0, 0)
            _user_layout.setSpacing(12)

            # Avatar
            self._avatar_label = SiLabel(self._user_info_widget)
            self._avatar_label.setFixedSize(56, 56)
            self._avatar_label.setStyleSheet(
                f"border-radius: 28px; background-color: #333; border: 2px solid #555;"
            )
            self._avatar_label.setAlignment(Qt.AlignCenter)
            _user_layout.addWidget(self._avatar_label)

            # Name + sub text
            _name_col = QVBoxLayout()
            _name_col.setSpacing(4)
            self._name_label = SiLabel(self._user_info_widget)
            self._name_label.setStyleSheet(f"color: {_TEXT}; font-size: 16px; font-weight: bold;")
            _name_col.addWidget(self._name_label)

            self._sub_label = SiLabel(self._user_info_widget)
            self._sub_label.setStyleSheet(f"color: {_TEXT_DIM}; font-size: 12px;")
            _name_col.addWidget(self._sub_label)

            _user_layout.addLayout(_name_col)
            _user_layout.addStretch()
            _c.addWidget(self._user_info_widget)

            self._status_label = SiLabel(_card)
            self._status_label.setVisible(False)
            self._status_label.setFixedHeight(28)
            self._status_label.setStyleSheet(f"color: {_TEXT_DIM}; font-size: 12px; padding: 4px 0;")
            _c.addWidget(self._status_label)

            self._fetch_videos_btn = SiPushButtonRefactor.withText("获取视频列表", _card)
            self._fetch_videos_btn.clicked.connect(self._on_fetch_videos)
            self._fetch_videos_btn.setVisible(False)
            self._fetch_videos_btn.setToolTip("开始获取该UP主的视频列表")
            _c.addWidget(self._fetch_videos_btn)

            _card.body().addWidget(_c)
            _card.adjustSize()
            g.addWidget(_card)

        # ======== 操作栏 ========
        with _g as g:
            g.addTitle("批量操作")
            _card2 = SiTriSectionPanelCard(g)
            self._batch_card = _card2
            _card2.setTitle("视频列表操作")
            _c2_body = SiDenseContainer(_card2.body())
            _c2_body.layout().setDirection(QBoxLayout.TopToBottom)
            _c2_body.layout().setSpacing(8)

            # Row 1: buttons
            _btn_row = QWidget()
            _btn_layout = QHBoxLayout(_btn_row)
            _btn_layout.setContentsMargins(0, 0, 0, 0)
            _btn_layout.setSpacing(8)
            _btn_specs = [
                ("全选", self._select_all, "勾选列表中所有视频"),
                ("取消全选", self._deselect_all, "取消勾选所有视频"),
                ("移除联合投稿", self._remove_coop, "取消勾选所有联合投稿视频（标签为橙色的视频）"),
                ("下载选中", self._download_selected, "下载所有已勾选的视频"),
            ]
            for _text, _slot, _tip in _btn_specs:
                _btn = SiPushButtonRefactor.withText(_text, _card2)
                _btn.clicked.connect(_slot)
                _btn.setToolTip(_tip)
                _btn_layout.addWidget(_btn)
            _btn_layout.addStretch()
            _c2_body.addWidget(_btn_row)

            # Row 2: download path
            _path_row = QWidget()
            _path_layout = QHBoxLayout(_path_row)
            _path_layout.setContentsMargins(0, 0, 0, 0)
            _path_layout.setSpacing(8)
            self._download_path_input = SiLabeledLineEdit(_card2)
            self._download_path_input.setTitle("下载保存路径")
            default_dl = str(self._project_root / "workspace" / "downloads")
            self._download_path_input.setText(default_dl)
            self._download_path_input.setFixedHeight(48)
            _path_layout.addWidget(self._download_path_input, 1)
            _c2_body.addWidget(_path_row)

            # Row 3: ffmpeg path
            _ffmpeg_row = QWidget()
            _ffmpeg_layout = QHBoxLayout(_ffmpeg_row)
            _ffmpeg_layout.setContentsMargins(0, 0, 0, 0)
            _ffmpeg_layout.setSpacing(8)
            self._ffmpeg_path_input = SiLabeledLineEdit(_card2)
            self._ffmpeg_path_input.setTitle("FFmpeg 路径")
            default_ffmpeg = str(self._project_root / "ffmpeg" / "ffmpeg.exe")
            self._ffmpeg_path_input.setText(default_ffmpeg)
            self._ffmpeg_path_input.setFixedHeight(48)
            _ffmpeg_layout.addWidget(self._ffmpeg_path_input, 1)
            _c2_body.addWidget(_ffmpeg_row)

            # Row 4: worker count + overall progress
            _dl_ctrl_row = QWidget()
            _dl_ctrl_layout = QHBoxLayout(_dl_ctrl_row)
            _dl_ctrl_layout.setContentsMargins(0, 0, 0, 0)
            _dl_ctrl_layout.setSpacing(8)
            self._max_workers_spin = SiIntSpinBox(_card2)
            self._max_workers_spin.setMinimum(1)
            self._max_workers_spin.setMaximum(8)
            self._max_workers_spin.setValue(2)
            self._max_workers_spin.setFixedHeight(48)
            self._max_workers_spin.setFixedWidth(150)
            _dl_ctrl_layout.addWidget(self._max_workers_spin)
            self._overall_bar = SiProgressBar(_card2)
            self._overall_bar.setFixedHeight(20)
            self._overall_bar.setValue(0.0)
            self._overall_bar.setVisible(False)
            _dl_ctrl_layout.addWidget(self._overall_bar, 1)
            self._overall_label = SiLabel(_card2)
            self._overall_label.setStyleSheet(f"color: {_TEXT_DIM}; font-size: 12px;")
            self._overall_label.setFixedWidth(70)
            self._overall_label.setAlignment(Qt.AlignCenter)
            self._overall_label.setText("0 / 0")
            self._overall_label.setVisible(False)
            _dl_ctrl_layout.addWidget(self._overall_label)
            _c2_body.addWidget(_dl_ctrl_row)

            # Row 5: worker cards container (dynamic)
            self._worker_cards_outer = QWidget()
            self._worker_cards_layout = QVBoxLayout(self._worker_cards_outer)
            self._worker_cards_layout.setContentsMargins(0, 0, 0, 0)
            self._worker_cards_layout.setSpacing(4)
            self._worker_cards_layout.addStretch()
            self._worker_cards_outer.setVisible(False)
            _c2_body.addWidget(self._worker_cards_outer)

            _card2.body().addWidget(_c2_body)
            _card2.adjustSize()
            g.addWidget(_card2)

        # ======== 视频列表容器 ========
        with _g as g:
            g.addTitle("视频列表")
            self._list_card = SiTriSectionPanelCard(g)
            self._list_card.setTitle("视频")
            self._list_body = self._list_card.body()
            self._list_body.setContentsMargins(6, 0, 6, 0)
            self._list_body.layout().setSpacing(4)

            # Progress bar
            self._progress_bar = SiProgressBar(self._list_body)
            self._progress_bar.setFixedHeight(24)
            self._progress_bar.setValue(0.0)
            self._progress_bar.setVisible(False)
            self._list_body.addWidget(self._progress_bar)

            # Empty state label
            self._list_empty_label = SiLabel(self._list_body)
            self._list_empty_label.setText("暂无视频数据，请先搜索 UID")
            self._list_empty_label.setStyleSheet(f"color: {_TEXT_DIM}; padding: 20px;")
            self._list_body.addWidget(self._list_empty_label)

            # Placeholder for the video list content (replaced on each rebuild)
            self._list_content = QWidget()
            self._list_content.setVisible(False)
            self._list_body.addWidget(self._list_content)

            self._list_card.adjustSize()
            g.addWidget(self._list_card)

        self.setAttachment(_g)

    def _filter_uid_input(self):
        """过滤非数字字符 — 不会触发递归（block信号）"""
        txt = self._uid_input.text()
        filtered = ''.join(c for c in txt if c.isdigit())
        if filtered != txt:
            # block signals to prevent recursion with _on_uid_changed
            self._uid_input.blockSignals(True)
            self._uid_input.setText(filtered)
            self._uid_input.setCursorPosition(len(filtered))
            self._uid_input.blockSignals(False)

    # ── 自动检测 UID ─────────────────────────────────────────

    def _on_uid_changed(self):
        # 停止之前的定时器
        if hasattr(self, '_uid_timer'):
            self._uid_timer.stop()
        from PyQt5.QtCore import QTimer
        self._uid_timer = QTimer(self)
        self._uid_timer.setSingleShot(True)
        self._uid_timer.timeout.connect(self._do_check_uid)
        self._uid_timer.start(800)  # 延迟 800ms，避免频繁请求

    def _do_check_uid(self):
        mid = self._uid_input.text().strip()
        if not mid or not mid.isdigit():
            self._status_label.setVisible(False)
            return
        if not self._cookie:
            self._set_status("请先登录（需要 Bilibili Cookie）")
            return

        self._set_status(f"正在查询 UID: {mid} ...")
        # Don't clear user info or resize card here — keeps layout stable during query

        def _do():
            try:
                info = fetch_user_info_by_mid(mid, self._cookie)
                print(f"[BilibiliAPI] 用户查询结果: mid={mid} info={info}")
            except Exception as e:
                print(f"[BilibiliAPI] 用户查询异常: {e}")
                info = {}
            self._bs.user_result.emit(info, mid)
        threading.Thread(target=_do, daemon=True).start()

    def _on_user_result(self, info, mid):
        if info.get('name'):
            name = info['name']
            face_url = info.get('face', '')
            self._name_label.setText(name)
            self._sub_label.setText(f"UID: {mid}")
            self._fetch_videos_btn.setVisible(True)
            self._current_mid = mid
            # Adjust card height to fit all content
            self._search_card.adjustSize()
            self._titled_group.adjustSize()
            self._refresh_scroll_area()
            self._bs.status_msg.emit(f"已找到用户 {name}，点击下方获取视频列表")

            # Download avatar in background
            if face_url:
                def _dl_face():
                    try:
                        r = requests.get(face_url, timeout=10)
                        r.raise_for_status()
                        pix = QPixmap()
                        pix.loadFromData(r.content)
                        if not pix.isNull():
                            pix = pix.scaled(56, 56, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                            self._bs.avatar_data.emit(pix)
                    except Exception as e:
                        print(f"[BilibiliAPI] 下载头像失败: {e}")
                threading.Thread(target=_dl_face, daemon=True).start()
        else:
            self._bs.status_msg.emit(f"未找到 UID {mid}，请检查输入")

    def _set_avatar(self, pixmap):
        """Set the avatar pixmap clipped to a circle."""
        from PyQt5.QtGui import QPainter, QPainterPath
        from PyQt5.QtCore import QRectF
        rounded = QPixmap(pixmap.size())
        rounded.fill(Qt.transparent)
        p = QPainter(rounded)
        p.setRenderHint(QPainter.Antialiasing)
        path = QPainterPath()
        path.addEllipse(QRectF(0, 0, pixmap.width(), pixmap.height()))
        p.setClipPath(path)
        p.drawPixmap(0, 0, pixmap)
        p.end()
        self._avatar_label.setPixmap(rounded)
        self._avatar_label.setStyleSheet("border: none; background: transparent;")
        self._refresh_scroll_area()

    def _refresh_scroll_area(self):
        """Force scroll area to recalculate scroll bar range after content height changes."""
        from PyQt5.QtGui import QResizeEvent
        old_sz = self.scroll_area.size()
        self.scroll_area.resizeEvent(QResizeEvent(old_sz, old_sz))

    def _set_status(self, msg):
        self._status_label.setText(msg)
        self._status_label.setVisible(bool(msg))

    # ── 获取视频 ──────────────────────────────────────────────

    def _on_fetch_videos(self):
        mid = getattr(self, '_current_mid', '')
        if not mid:
            return
        self._fetch_videos_btn.setEnabled(False)
        self._fetch_videos_btn.setText("正在获取...")
        self._set_status("正在请求 Bilibili API，请稍候...")
        self._progress_bar.setVisible(True)
        self._progress_bar.setValue(0.0)
        self._progress_bar.setState("processing")

        def on_page(count, page):
            # Fixed estimate, never goes backwards
            estimated_total = max(count, 180)
            p = min(count / (estimated_total * 1.05), 0.95)
            self._bs.progress_val.emit(p)

        def _do():
            try:
                videos = fetch_up_videos(mid, self._cookie, progress_callback=on_page)
                print(f"[BilibiliAPI] 获取视频列表: mid={mid} count={len(videos)}")
            except Exception as e:
                print(f"[BilibiliAPI] 获取视频列表异常: {e}")
                videos = []
            self._bs.videos_result.emit(videos)
        threading.Thread(target=_do, daemon=True).start()

    def _on_videos_result(self, videos):
        print(f"[UI] _on_videos_result 收到 {len(videos)} 个视频")  # debug
        self._fetch_videos_btn.setEnabled(True)
        self._fetch_videos_btn.setText("获取视频列表")
        self._progress_bar.setValue(1.0)
        self._progress_bar.setState("completing")
        self._videos = videos
        self._rebuild_table()
        # Hide progress bar after short delay so user sees 100%
        QTimer.singleShot(400, lambda: self._progress_bar.setVisible(False))
        self._set_status(f"共获取 {len(videos)} 个视频")

    def _rebuild_table(self):
        # Remove and delete old list content widget
        if self._list_content:
            self._list_content.setParent(None)
            self._list_content.deleteLater()
            self._list_content = None

        # Show/hide empty label
        has_videos = bool(self._videos)
        self._list_empty_label.setVisible(not has_videos)
        if not has_videos:
            return

        # Build new content widget with all video rows
        new_content = QWidget()
        new_layout = QVBoxLayout(new_content)
        new_layout.setContentsMargins(0, 0, 0, 0)
        new_layout.setSpacing(4)

        # Header
        hdr = QWidget()
        hdr_l = QHBoxLayout(hdr)
        hdr_l.setContentsMargins(8, 4, 8, 4)
        for _t, _w in [("选择", 50), ("标题", 400), ("BV号", 140), ("联合投稿", 70), ("操作", 120)]:
            _l = SiLabel(hdr)
            _l.setText(_t)
            _l.setStyleSheet(f"color: {_TEXT_DIM}; font-size: 12px; font-weight: bold;")
            _l.setFixedWidth(_w)
            hdr_l.addWidget(_l)
        hdr_l.addStretch()
        new_layout.addWidget(hdr)

        for i, v in enumerate(self._videos):
            row = QWidget()
            row_l = QHBoxLayout(row)
            row_l.setContentsMargins(8, 4, 8, 4)
            row_l.setSpacing(4)

            # Checkbox
            cb = SiCheckBox(row)
            cb.setFixedWidth(50)
            v['cb'] = cb
            row_l.addWidget(cb)

            # Title
            tl = SiLabel(row)
            tl.setText(v.get('title', '')[:50])
            tl.setStyleSheet(f"color: {_TEXT}; font-size: 12px;")
            tl.setFixedWidth(400)
            row_l.addWidget(tl)

            # BV
            bv = SiLabel(row)
            bv.setText(v.get('bvid', ''))
            bv.setStyleSheet(f"color: {_TEXT_DIM}; font-size: 11px;")
            bv.setFixedWidth(140)
            row_l.addWidget(bv)

            # Co-op
            cp = SiLabel(row)
            cp.setText("是" if v.get('is_coop') else "否")
            cp.setStyleSheet(f"color: {'#FFA500' if v.get('is_coop') else _TEXT_DIM}; font-size: 12px;")
            cp.setFixedWidth(70)
            row_l.addWidget(cp)

            # Download button
            dl_btn = SiPushButtonRefactor.withText("下载", row)
            dl_btn.setFixedWidth(60)
            dl_btn.clicked.connect(lambda checked=False, bvid=v['bvid']: self._on_download_one(bvid))
            dl_btn.setToolTip("单独下载该视频")
            row_l.addWidget(dl_btn)

            # Delete button
            del_btn = SiPushButtonRefactor.withText("×", row)
            del_btn.setFixedWidth(30)
            del_btn.clicked.connect(lambda checked=False, idx=i: self._on_delete_row(idx))
            del_btn.setToolTip("从列表中移除该视频")
            row_l.addWidget(del_btn)

            row_l.addStretch()
            new_layout.addWidget(row)
            # Update checkbox reference on the video dict
            v['cb'] = cb

        # Insert new content before stretch widget
        sw = self._list_body.stretchWidget()
        sw_idx = self._list_body.layout().indexOf(sw)
        if sw_idx >= 0:
            self._list_body.layout().insertWidget(sw_idx, new_content)
        else:
            self._list_body.layout().addWidget(new_content)
        self._list_content = new_content
        new_content.setVisible(True)
        new_content.update()
        # Force full layout chain update
        # Preserve card width, only adjust height
        saved_width = self._list_card.width()
        self._list_card.adjustSize()
        self._list_card.setMinimumWidth(saved_width)
        self._titled_group.adjustSize()
        self._titled_group.updateGeometry()
        self._refresh_scroll_area()

    # ── 操作 ──────────────────────────────────────────────────

    def _select_all(self):
        for v in self._videos:
            cb = v.get('cb')
            if cb:
                cb.setChecked(True)

    def _deselect_all(self):
        for v in self._videos:
            cb = v.get('cb')
            if cb:
                cb.setChecked(False)

    def _remove_coop(self):
        """取消勾选所有联合投稿视频，不移除"""
        for v in self._videos:
            if v.get('is_coop') and v.get('cb'):
                v['cb'].setChecked(False)

    def _download_selected(self):
        selected = [v for v in self._videos if v.get('cb') and v['cb'].isChecked()]
        if not selected:
            _n = SiLabel(self._list_card)
            _n.setText("请先勾选要下载的视频")
            _n.setStyleSheet(f"color: #FFA500; padding: 4px;")
            self._list_body.addWidget(_n)
            QTimer.singleShot(3000, _n.deleteLater)
            return
        out_dir = self._download_path_input.text().strip() or str(self._project_root / "workspace" / "downloads")

        max_workers = self._max_workers_spin.value()
        actual_workers = min(max_workers, len(selected))

        # Build worker cards
        for i in reversed(range(self._worker_cards_layout.count())):
            item = self._worker_cards_layout.itemAt(i)
            if item and item.widget():
                item.widget().deleteLater()
        self._worker_cards = []
        for i in range(actual_workers):
            card = _WorkerCardWidget(i, self._worker_cards_outer)
            card.restart_clicked.connect(self._on_worker_restart)
            self._worker_cards.append(card)
            self._worker_cards_layout.insertWidget(self._worker_cards_layout.count() - 1, card)

        self._worker_cards_outer.setVisible(True)
        self._overall_bar.setVisible(True)
        self._overall_label.setVisible(True)
        self._overall_bar.setValue(0.0)
        self._overall_bar.setState("processing")
        self._overall_label.setText("0 / 0")

        # Adjust card height to fit all controls
        self._batch_card.adjustSize()
        self._titled_group.adjustSize()
        self._refresh_scroll_area()

        ffmpeg_path = self._ffmpeg_path_input.text().strip() or None
        self._download_manager = _DownloadManager(self._bs, self._cookie, max_workers,
                                                   ffmpeg_path=ffmpeg_path)
        items = [(v['bvid'], v.get('title', '')) for v in selected]
        self._download_manager.start(items, out_dir)

    def _on_download_one(self, bvid):
        out_dir = self._download_path_input.text().strip() or str(self._project_root / "workspace" / "downloads")
        t = threading.Thread(target=lambda: self._do_download(bvid, out_dir), daemon=True)
        t.start()

    def _do_download(self, bvid, out_dir):
        ok = download_single_video(bvid, self._cookie, out_dir)
        msg = f"✓ {bvid} 下载完成" if ok else f"✗ {bvid} 下载失败"
        print(msg)
        _pw = self.window()
        if hasattr(_pw, 'show_task_completed_notification'):
            _pw.show_task_completed_notification(msg)

    def _on_delete_row(self, idx):
        if 0 <= idx < len(self._videos):
            self._videos.pop(idx)
            self._rebuild_table()

    # ── 下载管理器信号处理 ──────────────────────────────────

    def _on_worker_progress(self, worker_id, fraction, speed, stage):
        if 0 <= worker_id < len(self._worker_cards):
            self._worker_cards[worker_id].update_progress(fraction, speed, stage)

    def _on_worker_state(self, worker_id, bvid, active):
        if 0 <= worker_id < len(self._worker_cards):
            if active:
                self._worker_cards[worker_id].set_active(bvid)
            else:
                self._worker_cards[worker_id].set_idle()

    def _on_overall_progress(self, completed, total):
        self._overall_bar.setVisible(True)
        self._overall_label.setVisible(True)
        if total > 0:
            self._overall_bar.setValue(completed / total)
        self._overall_label.setText(f"{completed} / {total}")

    def _on_all_done(self):
        self._overall_bar.setState("completing")
        self._overall_label.setText("完成")
        _pw = self.window()
        if hasattr(_pw, 'show_task_completed_notification'):
            _pw.show_task_completed_notification("所有视频下载完成")
        QTimer.singleShot(3000, self._reset_download_ui)

    def _reset_download_ui(self):
        self._overall_bar.setVisible(False)
        self._overall_label.setVisible(False)
        self._worker_cards_outer.setVisible(False)
        for card in getattr(self, '_worker_cards', []):
            self._worker_cards_layout.removeWidget(card)
            card.deleteLater()
        self._worker_cards = []

    def _on_worker_restart(self, worker_id):
        if hasattr(self, '_download_manager') and self._download_manager.is_active():
            self._download_manager.abort_worker(worker_id)
            if 0 <= worker_id < len(self._worker_cards):
                self._worker_cards[worker_id]._bvid_label.setText("重启中...")
