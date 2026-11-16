import sys
import os
import ctypes

_DEBUG = '--debug' in sys.argv or os.environ.get('UI_DEBUG', '') == '1'

# 控制台/日志重定向时默认编码可能是 GBK，而 ui.py 里有 "✓ / ✗" 这类字符，
# 直接 print 会抛 UnicodeEncodeError 把整个启动打断。
# errors='replace' 只为兜住这类装饰性输出——正常控制台下无影响。
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(errors='replace')
    except Exception:
        pass
del _stream

def _dbg(msg):
    if _DEBUG:
        print(f'Debug: {msg}')

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.insert(0, project_root)  # 确保 core/ 等模块可导入

# 通过 PyQt5 模块定位 Qt DLL 目录（兼容任意 Python 环境布局）
try:
    import PyQt5
    _pyqt5_dir = os.path.dirname(PyQt5.__file__)
    qt_bin_path = os.path.join(_pyqt5_dir, 'Qt5', 'bin')
    if not os.path.exists(qt_bin_path):
        # 备选：pip 安装的 PyQt5 有时 Qt5 在上一层
        qt_bin_path = os.path.join(_pyqt5_dir, '..', 'Qt5', 'bin')
    if not os.path.exists(qt_bin_path):
        raise RuntimeError(f'Qt5/bin not found near PyQt5: {_pyqt5_dir}')
except ImportError:
    raise RuntimeError('PyQt5 is not installed')

qt_bin_path = os.path.abspath(qt_bin_path)
_dbg(f'Qt bin path: {qt_bin_path}')

if hasattr(os, 'add_dll_directory'):
    os.add_dll_directory(qt_bin_path)
os.environ['PATH'] = qt_bin_path + ';' + os.environ['PATH']

# Preload critical VC++ + Qt DLLs (only Qt5Core is mandatory)
vc_dlls = ['vcruntime140.dll', 'vcruntime140_1.dll', 'msvcp140.dll', 'concrt140.dll']
for dll_name in vc_dlls:
    dll_path = os.path.join(qt_bin_path, dll_name)
    if os.path.exists(dll_path):
        try:
            ctypes.CDLL(dll_path)
            _dbg(f'Loaded {dll_name}')
        except Exception as e:
            if _DEBUG:
                _dbg(f'Failed to load {dll_name}: {e}')
# Qt5Core is mandatory
qt5core_path = os.path.join(qt_bin_path, 'Qt5Core.dll')
if os.path.exists(qt5core_path):
    try:
        ctypes.CDLL(qt5core_path)
    except Exception as e:
        raise RuntimeError(f'Failed to load Qt5Core.dll: {e}') from e
else:
    raise RuntimeError(f'Qt5Core.dll not found in {qt_bin_path}')

# Other Qt DLLs load lazily — skip preloading
for dll_name in ['Qt5Gui.dll', 'Qt5Widgets.dll', 'Qt5Svg.dll', 'Qt5Network.dll']:
    dll_path = os.path.join(qt_bin_path, dll_name)
    if os.path.exists(dll_path):
        try:
            ctypes.CDLL(dll_path)
        except Exception:
            pass  # will be loaded on demand by Qt5Core

# Torch lib path — 只加 PATH，不 import（torch DLL 初始化需在 Qt 之后）
_torch_lib_candidates = [
    os.path.join(project_root, 'python', 'Lib', 'site-packages', 'torch', 'lib'),
]
# 也可能通过 PyQt5 同级的 site-packages
_pyqt_site = os.path.dirname(os.path.dirname(os.path.dirname(qt_bin_path)))  # e.g. .../site-packages
_torch_in_site = os.path.join(_pyqt_site, 'torch', 'lib')
if os.path.exists(_torch_in_site):
    _torch_lib_candidates.append(_torch_in_site)

for _p in _torch_lib_candidates:
    if os.path.exists(_p):
        os.environ['PATH'] = _p + ';' + os.environ['PATH']
        if hasattr(os, 'add_dll_directory'):
            os.add_dll_directory(_p)
        break

import subprocess
print("[start.py] 初始化...", flush=True)
# Auto-install Bilibili auth dependencies if missing
for _bili_pkg in ['qrcode', 'brotli']:
    try:
        __import__(_bili_pkg.replace('-', '_'))
    except ImportError:
        print(f"正在安装 {_bili_pkg}...")
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', _bili_pkg])

from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import QApplication, QDialog
from PyQt5.QtGui import QIcon

from PyQt5.QtCore import qInstallMessageHandler
def _silent_qt_msg_handler(msg_type, context, message):
    if 'QPixmap::scaled' in message or 'QPixmap::fromImage' in message:
        return
    if msg_type == 3 or msg_type == 4:
        sys.stderr.write(message + '\n')
qInstallMessageHandler(_silent_qt_msg_handler)

try:
    from ui.ui import MySiliconApp
except ModuleNotFoundError:
    from ui import MySiliconApp

import siui
from siui.core import SiGlobal


def show_version_message(window):
    try:
        icon = SiGlobal.siui.iconpack.get("ic_fluent_hand_wave_filled")
    except KeyError:
        icon = None
    from core.biliauth.cookie_store import get_user_name, load_user_info, is_dev_uid
    info = load_user_info()
    uid = info.get('uid', 0)
    name = info.get('name', '')
    display_name = get_user_name()
    if is_dev_uid(uid):
        greeting = f"欢迎回来！{display_name}"
    elif name:
        greeting = f"您好！{name}"
    else:
        greeting = "欢迎使用"
    window.LayerRightMessageSidebar().send(
        title=greeting,
        text="欢迎使用 深变:DFL-PyTorch",
        msg_type=1,
        icon=icon,
        fold_after=5000,
    )


if __name__ == "__main__":
    print("[start.py] 创建 QApplication...", flush=True)
    app = QApplication(sys.argv)

    icon_path = os.path.join(script_dir, 'img', 'logo_new.png')
    if os.path.exists(icon_path):
        app.setWindowIcon(QIcon(icon_path))
        if sys.platform == 'win32':
            try:
                ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID('DeepFaceLab.Torch.App')
            except Exception:
                pass

    # === Bilibili login gate (before main window) ===
    print("[start.py] 登录门禁...", flush=True)
    sys.path.insert(0, project_root)
    sys.path.insert(0, script_dir)
    from core.biliauth.cookie_store import is_logged_in

    if not is_logged_in():
        from components.biliauth.login_dialog import LoginDialog
        print("[start.py] 弹出登录对话框...", flush=True)
        _dlg = LoginDialog()
        try:
            _result = _dlg.exec_()
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"登录对话框异常: {e}", flush=True)
            _result = QDialog.Rejected
        if _result != QDialog.Accepted:
            print("[start.py] 登录被取消", flush=True)
            sys.exit(0)

    window = MySiliconApp()
    window.show()

    # 显示欢迎通知
    from PyQt5.QtCore import QTimer
    QTimer.singleShot(500, lambda: show_version_message(window))

    # 进入 Qt 事件循环
    sys.exit(app.exec_())

