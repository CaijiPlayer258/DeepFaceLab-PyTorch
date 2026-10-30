import os
import sys
import shutil
from pathlib import Path

if sys.platform[0:3] == 'win':
    from ctypes import windll
    from ctypes import wintypes

def set_process_lowest_prio():
    try:
        if sys.platform[0:3] == 'win':
            GetCurrentProcess = windll.kernel32.GetCurrentProcess
            GetCurrentProcess.restype = wintypes.HANDLE
            SetPriorityClass = windll.kernel32.SetPriorityClass
            SetPriorityClass.argtypes = (wintypes.HANDLE, wintypes.DWORD)
            SetPriorityClass ( GetCurrentProcess(), 0x00000040 )
        elif 'darwin' in sys.platform:
            os.nice(10)
        elif 'linux' in sys.platform:
            os.nice(20)
    except:
        print("Unable to set lowest process priority")

def set_process_dpi_aware():
    if sys.platform[0:3] == 'win':
        windll.user32.SetProcessDPIAware(True)

def get_screen_size():
    if sys.platform[0:3] == 'win':
        user32 = windll.user32
        return user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)
    elif 'darwin' in sys.platform:
        pass
    elif 'linux' in sys.platform:
        pass
        
    return (1366, 768)


def _workspace_root() -> Path:
    # core/osex.py -> core -> workspace root
    return Path(__file__).resolve().parents[1]


def _resolve_executable(env_var: str, bundled_relative_names: list, fallback_name: str):
    """Resolve an executable path.

    Priority:
    1) environment variable (absolute or relative)
    2) workspace bundled executable under tools/
    3) system PATH (shutil.which)
    """
    env_value = os.environ.get(env_var, '').strip().strip('"')
    if env_value:
        p = Path(env_value)
        if not p.is_absolute():
            p = (_workspace_root() / p).resolve()
        if p.exists():
            return str(p)

    root = _workspace_root()
    for rel in bundled_relative_names:
        p = (root / rel).resolve()
        if p.exists():
            return str(p)

    which = shutil.which(fallback_name)
    if which:
        return which

    return None


def get_ffmpeg_path():
    """Return resolved ffmpeg executable path or None."""
    is_win = sys.platform[0:3] == 'win'
    bundled = [
        'tools/ffmpeg.exe' if is_win else 'tools/ffmpeg',
        'tools/ffmpeg',
        'tools/ffmpeg.exe',
    ]
    return _resolve_executable('DFL_FFMPEG', bundled, 'ffmpeg')


def get_ffprobe_path():
    """Return resolved ffprobe executable path or None."""
    is_win = sys.platform[0:3] == 'win'
    bundled = [
        'tools/ffprobe.exe' if is_win else 'tools/ffprobe',
        'tools/ffprobe',
        'tools/ffprobe.exe',
    ]
    return _resolve_executable('DFL_FFPROBE', bundled, 'ffprobe')
        