import icons
import sys
from components.page_about import About
from components.page_changelog.page_changelog import ChangelogPage
from components.page_bilibili_downloader.page_bilibili_downloader import BilibiliDownloaderPage
from components.page_data_extraction import DataExtractionPage
from components.page_data_processing import DataProcessingPage
from components.page_homepage import ExampleHomepage
from components.page_mergestudio.page_mergestudio import MergeStudioPage
from components.page_maskprocessor.page_maskprocessor import MaskProcessorPage
from components.page_trtcompile.page_trtcompile import TRTCompilePage
from components.page_trainer import TrainerPage
from PyQt5.QtCore import QObject, pyqtSignal
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import QDesktopWidget, QApplication


class NotificationWorker(QObject):
    """通知工作器，用于从子线程安全地发送通知"""
    show_notification_signal = pyqtSignal(str, str, int, object, int)
    
    def __init__(self, sidebar_method):
        super().__init__()
        self.sidebar_method = sidebar_method
        self.show_notification_signal.connect(self._handle_notification)
    
    def _handle_notification(self, title: str, text: str, msg_type: int, icon, fold_after: int):
        """在主线程中处理通知"""
        try:
            if self.sidebar_method is not None:
                self.sidebar_method().send(
                    title=title,
                    text=text,
                    msg_type=msg_type,
                    icon=icon,
                    fold_after=fold_after,
                )
        except Exception as e:
            print(f"显示通知失败: {e}")

import siui
from siui.core import SiColor, SiGlobal
from siui.templates.application.application import SiliconApplication

# 载入图标
siui.core.globals.SiGlobal.siui.loadIcons(
    icons.IconDictionary(color=SiGlobal.siui.colors.fromToken(SiColor.SVG_NORMAL)).icons
)


class MySiliconApp(SiliconApplication):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        screen_geo = QDesktopWidget().screenGeometry()
        self.setMinimumSize(1024, 380)
        self.resize(1366, 916)
        self.move((screen_geo.width() - self.width()) // 2, (screen_geo.height() - self.height()) // 2)
        self.layerMain().setTitle("深变:DFL-PyTorch")
        self.setWindowTitle("深变:DFL-PyTorch")
        
        # 使用绝对路径加载窗口图标和任务栏图标
        import os
        import ctypes
        icon_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'img', 'logo_new.png')
        if os.path.exists(icon_path):
            # 设置窗口图标（标题栏）
            window_icon = QIcon(icon_path)
            self.setWindowIcon(window_icon)
            
            # 设置应用程序图标（任务栏）- Windows特殊处理
            QApplication.setWindowIcon(window_icon)
            
            # Windows特定：设置应用用户模型ID以正确显示任务栏图标
            if sys.platform == 'win32':
                try:
                    # 设置AppUserModelID，这样Windows任务栏会正确显示图标
                    ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID('DeepFaceLab.Torch.App')
                except Exception as e:
                    print(f'设置AppUserModelID失败: {e}')
            
            print(f'✓ 窗口图标和任务栏图标加载成功: {icon_path}')
        else:
            print(f'✗ 图标文件不存在: {icon_path}')

        # 初始化通知工作器
        self.notification_worker = NotificationWorker(self.LayerRightMessageSidebar)

        # 修正侧边栏顶部应用图标（siui 库用了相对路径 ./img/logo_new.png 导致不显示）
        _header_icon = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'img', 'logo_new.png')
        if os.path.exists(_header_icon):
            try:
                self.layerMain().app_icon.load(_header_icon)
            except Exception:
                pass

        # 注册所有页面到侧边栏
        self._setup_pages()

    def _setup_pages(self):
        """注册所有页面到侧边栏"""
        self.layerMain().addPage(ExampleHomepage(self),
                                 icon=self.safe_get_icon("ic_fluent_home_filled"),
                                 hint="主页", side="top")
        self.layerMain().addPage(DataExtractionPage(self),
                                 icon=self.safe_get_icon("ic_fluent_scan_person_filled"),
                                 hint="数据提取", side="top")
        self.layerMain().addPage(DataProcessingPage(self),
                                 icon=self.safe_get_icon("ic_fluent_data_scatter_filled"),
                                 hint="数据处理", side="top")
        self.layerMain().addPage(MaskProcessorPage(self),
                                 icon=self.safe_get_icon("ic_fluent_eyedropper_filled"),
                                 hint="遮罩绘制", side="top")
        self.layerMain().addPage(TrainerPage(self),
                                 icon=self.safe_get_icon("ic_fluent_fire_filled"),
                                 hint="训练器", side="top")
        self.layerMain().addPage(MergeStudioPage(self),
                                 icon=self.safe_get_icon("ic_fluent_box_arrow_up_filled"),
                                 hint="合成工作台", side="top")
        self.layerMain().addPage(BilibiliDownloaderPage(self),
                                 icon=self.safe_get_icon("ic_fluent_video_filled"),
                                 hint="B站下载", side="top")
        self.layerMain().addPage(TRTCompilePage(self),
                                 icon=self.safe_get_icon("fi-rr-settings"),
                                 hint="TRT编译", side="top")
        # 侧边栏排序规则：
        #   top: 先注册的在上方，后注册的往中间堆叠
        #   bottom: 先注册的在最下方，后注册的往中间堆叠
        self.layerMain().addPage(About(self),
                                 icon=self.safe_get_icon("ic_fluent_info_filled"),
                                 hint="关于", side="bottom")
        self.layerMain().addPage(ChangelogPage(self),
                                 icon=self.safe_get_icon("ic_fluent_history_filled"),
                                 hint="更新日志", side="bottom")
        self.layerMain().setPage(0)
        SiGlobal.siui.reloadAllWindowsStyleSheet()

    def safe_get_icon(self, name):
        """安全获取图标"""
        try:
            return SiGlobal.siui.iconpack.get(name)
        except KeyError:
            pass
        try:
            return SiGlobal.siui.icons[name]
        except KeyError:
            return None

    def show_command_notification(self, command: str, description: str = ""):
        """
        显示命令执行通知
        
        Args:
            command: 实际执行的命令
            description: 命令描述（可选）
        """
        try:
            icon = SiGlobal.siui.iconpack.get("ic_fluent_terminal_filled")
        except KeyError:
            icon = None
        
        # 格式化命令显示（截断过长的命令）
        if len(command) > 150:
            command_display = command[:147] + "..."
        else:
            command_display = command
        
        # 构建通知文本
        if description:
            text = f"<strong>{description}</strong><br>命令: {command_display}"
        else:
            text = f"执行命令:<br>{command_display}"
        
        # 使用信号槽机制，线程安全（持续5秒）
        self.notification_worker.show_notification_signal.emit(
            '正在执行', text, 1, icon, 5000
        )
    
    def show_task_completed_notification(self, description: str = "任务已完成"):
        """
        显示任务完成通知（绿色）
        
        Args:
            description: 任务描述
        """
        try:
            icon = SiGlobal.siui.iconpack.get("ic_fluent_checkmark_circle_filled")
        except KeyError:
            icon = None
        
        # 使用信号槽机制，线程安全（持续2秒）
        self.notification_worker.show_notification_signal.emit(
            '✓ 任务完成', description, 1, icon, 2000
        )
    
    def show_task_interrupted_notification(self, description: str = "用户强制中断了一个任务"):
        """
        显示任务被中断通知（黄色）
        
        Args:
            description: 中断描述
        """
        try:
            icon = SiGlobal.siui.iconpack.get("ic_fluent_dismiss_circle_filled")
        except KeyError:
            icon = None
        
        # 使用信号槽机制，线程安全（持续2秒）
        self.notification_worker.show_notification_signal.emit(
            '⚠ 任务中断', description, 2, icon, 2000
        )
    
    def show_task_error_notification(self, description: str = "任务异常结束"):
        """
        显示任务错误通知（红色）
        
        Args:
            description: 错误描述
        """
        try:
            icon = SiGlobal.siui.iconpack.get("ic_fluent_error_circle_filled")
        except KeyError:
            icon = None
        
        # 使用信号槽机制，线程安全（持续3秒，让用户看清错误信息）
        self.notification_worker.show_notification_signal.emit(
            '✗ 任务失败', description, 4, icon, 3000
        )
