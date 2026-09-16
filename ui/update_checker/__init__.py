"""
自动更新检查器
启动后在后台检查 GitHub 是否有新版本，询问用户是否拉取。
"""
import subprocess, sys, os, json, shutil, re
from pathlib import Path

# 测试开关：设为 True 可模拟有更新（即使本地已最新）
_FORCE_UPDATE_TEST = False

RAW_VERSION_URL = "https://raw.githubusercontent.com/CaijiPlayer258/DeepFaceLab-PyTorch/master/version.txt"
ZIP_URL = "https://github.com/CaijiPlayer258/DeepFaceLab-PyTorch/archive/master.zip"
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
VERSION_FILE = PROJECT_ROOT / "version.txt"

def _is_git_installed() -> bool:
    """检查系统是否安装了 git"""
    try:
        r = subprocess.run(["git", "--version"], capture_output=True, text=True, timeout=5)
        return r.returncode == 0
    except FileNotFoundError:
        return False


def _install_git() -> bool:
    """通过镜像下载 Git for Windows 并静默安装（备选 winget）"""
    import urllib.request
    import re as _re

    # 获取 winget 中已知的 Git 最新版本号（直接搜字节避免 GBK 编码问题）
    _ver = ""
    try:
        _r = subprocess.run(
            ["winget", "show", "--id", "Git.Git", "--exact", "--accept-source-agreements"],
            capture_output=True, timeout=20
        )
        _m = _re.search(rb'(\d+\.\d+\.\d+\.\d+)', _r.stdout)
        if _m:
            _ver = _m.group(1).decode('ascii')
    except Exception:
        pass

    if not _ver:
        print("[Update] 未获取到 Git 版本号，走 winget 安装...", flush=True)
        return _install_git_winget()

    # 从版本号解析 tag 和 exe 名
    # 版本 e.g. "2.55.0.3" → tag "v2.55.0.windows.3" → exe "Git-2.55.0.3-64-bit.exe"
    _parts = _ver.split(".")
    if len(_parts) == 4:
        _tag = f"v{_parts[0]}.{_parts[1]}.{_parts[2]}.windows.{_parts[3]}"
        _exe = f"Git-{_ver}-64-bit.exe"
    else:
        print(f"[Update] 无法解析版本号: {_ver}，走 winget 安装...", flush=True)
        return _install_git_winget()

    _urls = [
        f"https://registry.npmmirror.com/-/binary/git-for-windows/{_tag}/{_exe}",
        f"https://mirrors.huaweicloud.com/git-for-windows/{_tag}/{_exe}",
        f"https://ghproxy.com/https://github.com/git-for-windows/git/releases/download/{_tag}/{_exe}",
        f"https://github.com/git-for-windows/git/releases/download/{_tag}/{_exe}",
    ]

    _exe_path = os.path.join(os.environ.get("TEMP", "."), _exe)
    _ok = False
    for _url in _urls:
        print(f"[Update] 下载 {_url}", flush=True)
        try:
            _resp = urllib.request.urlopen(_url, timeout=60)
            _total = int(_resp.headers.get('Content-Length', 0))
            _dl = 0
            _chunk = 256 * 1024  # 256KB
            with open(_exe_path, 'wb') as _f:
                while True:
                    _buf = _resp.read(_chunk)
                    if not _buf:
                        break
                    _f.write(_buf)
                    _dl += len(_buf)
                    if _total > 0:
                        _pct = min(100, _dl * 100 // _total)
                        print(f"\r  {_pct}% ({_dl//1024**2}MB/{_total//1024**2}MB)", end="", flush=True)
                    else:
                        print(f"\r  {_dl//1024**2}MB", end="", flush=True)
            print("", flush=True)
            _ok = True
            break
        except Exception as _e:
            print(f"\n[Update] 下载失败: {_e}", flush=True)
            continue

    if not _ok:
        print("[Update] 所有下载源均失败，走 winget 安装...", flush=True)
        return _install_git_winget()

    print("[Update] 正在静默安装 Git...", flush=True)
    try:
        p = subprocess.Popen(
            [_exe_path, "/VERYSILENT", "/SUPPRESSMSGBOXES", "/NORESTART", "/CLOSEAPPLICATIONS"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        # InnoSetup 静默安装无输出，用旋转等待动画
        import itertools as _it
        for _c in _it.cycle('|/-\\'):
            if p.poll() is not None:
                break
            print(f"\r  安装中... {_c}", end="", flush=True)
            import time as _t
            _t.sleep(0.3)
        print("", flush=True)
        p.wait(timeout=120)
        if p.returncode == 0:
            print("[Update] ✅ Git 安装成功", flush=True)
            try:
                os.remove(_exe_path)
            except Exception:
                pass
            return True
        print(f"[Update] ❌ 安装失败 (exit={p.returncode})", flush=True)
        return False
    except Exception as _e:
        print(f"[Update] 安装异常: {_e}，走 winget...", flush=True)
        return _install_git_winget()


def _install_git_winget() -> bool:
    """winget 安装 git（备选方案）"""
    print("[Update] 正在通过 winget 安装 Git for Windows...", flush=True)
    try:
        p = subprocess.Popen(
            ["winget", "install", "--id", "Git.Git", "--accept-package-agreements", "--accept-source-agreements"],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding='utf-8', errors='replace'
        )
        for _line in p.stdout:
            print(f"  {_line.rstrip()}", flush=True)
        p.wait(timeout=120)
        if p.returncode == 0:
            print("[Update] ✅ Git 安装成功", flush=True)
            return True
        print(f"[Update] ❌ winget 安装失败 (exit={p.returncode})", flush=True)
        return False
    except Exception as e:
        print(f"[Update] 安装 Git 时出错: {e}", flush=True)
        return False


def _parse_versions(text: str):
    """解析 version.txt 格式，返回 [(ver, [notes...]), ...]

    version.txt 是 UTF-8 **带 BOM** 的：read_text(encoding='utf-8') 会把 BOM
    留在首行变成 '\ufeff4.0.8'，_ver_to_tuple 解析失败退化成 (0,)，版本比较
    随之完全失效（本地 4.0.8.1 对着远程 4.0.8 会被误判成「有新版本」）。
    统一在这里剥掉 BOM，所有调用点都不用再操心编码。
    """
    sections = []
    for block in text.lstrip("\ufeff").strip().split("\n\n"):
        # 关键：不能对每行 l.strip()。version.txt 用「- 」主条目 + 「  · 」子条目 +
        # 4 空格续行来表达层级，strip() 会把缩进一并吃掉，界面上就退化成一坨没有
        # 层次的文字（只剩行首那个 · / - 字符本身），这正是「更新信息没有分行」。
        lines = [l.rstrip() for l in block.strip("\n").split("\n") if l.strip()]
        if lines:
            sections.append((lines[0].strip(), lines[1:]))
    return sections


_BULLET_RE = re.compile(r'^(\s*)([-•·*])\s+(.*)$')
_ASCII_TAIL_RE = re.compile(r'[0-9A-Za-z_]$')
_ASCII_HEAD_RE = re.compile(r'^[0-9A-Za-z_]')


def _reflow_notes(notes):
    """把 version.txt 里为终端宽度做的「手工折行」还原成有层级的条目。

    version.txt 的版式是：
        - 主条目
          · 子条目
            续行（2 或 4 空格起）
    续行的缩进不代表新条目，只是上一句在同一段里被折到了下一行。直接塞进 QLabel
    会同时丢掉层级和折行的意义，所以这里先把续行并回上一条，再按缩进判层级。

    返回 [(level, text), ...]：level 0 = 主条目，level 1 = 子条目。
    """
    items = []
    for raw in notes:
        if not raw.strip():
            continue
        m = _BULLET_RE.match(raw)
        if m:
            indent = len(m.group(1).expandtabs(4))
            items.append([0 if indent == 0 else 1, m.group(3).strip()])
            continue
        indent = len(raw) - len(raw.lstrip())
        if indent >= 2 and items:
            prev, piece = items[-1][1], raw.strip()
            # 中英文混排：两边都是 ASCII 词时才补空格，否则直接接上
            sep = " " if (_ASCII_TAIL_RE.search(prev) and _ASCII_HEAD_RE.match(piece)) else ""
            items[-1][1] = prev + sep + piece
        else:
            items.append([0, raw.strip()])
    return [(lv, tx) for lv, tx in items]


def _ver_to_tuple(v: str) -> tuple:
    """"1.2.3" -> (1,2,3)，用于版本比较"""
    try:
        return tuple(int(x) for x in v.split("."))
    except Exception:
        return (0,)


def _print_network_tip():
    """网络访问失败时输出引导提示"""
    print("", flush=True)
    print("=" * 50, flush=True)
    print("  网络访问失败？试试 Watt Toolkit 加速！", flush=True)
    print("  下载地址: https://steampp.net/download", flush=True)
    print("=" * 50, flush=True)
    print("", flush=True)


def _get_remote_version() -> str:
    """获取远程 version.txt 的最新版本号"""
    print(f"[Update]  检查远程版本...", flush=True)
    try:
        import urllib.request
        r = urllib.request.urlopen(RAW_VERSION_URL, timeout=10)
        text = r.read().decode('utf-8')
        sections = _parse_versions(text)
        if sections:
            return sections[0][0]  # 第一个版本号
        return ""
    except Exception as _ex:
        print(f"[Update]    -> 失败: {_ex}", flush=True)
        _print_network_tip()
        return ""


def _get_local_version() -> str:
    """读取本地 version.txt 版本号（首行）"""
    try:
        sections = _parse_versions(VERSION_FILE.read_text(encoding='utf-8'))
        if sections:
            return sections[0][0]
        return ""
    except Exception:
        return ""


def _get_changelog_since(local_ver: str, remote_text: str) -> list:
    """获取比 local_ver 新的所有版本的更新日志"""
    sections = _parse_versions(remote_text)
    entries = []
    for ver, notes in sections:
        if _ver_to_tuple(ver) > _ver_to_tuple(local_ver):
            entries.append((ver, notes))
    return entries


def check_for_updates(callback=None):
    """检查更新"""
    try:
        if _FORCE_UPDATE_TEST:
            print("[Update] 测试模式：跳过检查", flush=True)
            return
        import urllib.request
        r = urllib.request.urlopen(RAW_VERSION_URL, timeout=10)
        remote_text = r.read().decode('utf-8')
        remote_sections = _parse_versions(remote_text)
        if not remote_sections:
            print("[Update] 获取远程版本失败")
            return
        remote_ver = remote_sections[0][0]
        local_ver = _get_local_version()
        if not local_ver:
            print("[Update] 无法获取本地版本")
            return
        print(f"[Update] 本地: v{local_ver}  远程: v{remote_ver}", flush=True)
        # 必须比大小而不是比字符串：本地 4.0.8.1 对着远程 4.0.8 时字符串不相等，
        # 比字符串会误报「有新版本」，而 _get_changelog_since 又只会返回空列表。
        if _ver_to_tuple(remote_ver) <= _ver_to_tuple(local_ver):
            print(f"[Update] 当前版本 v{local_ver}（已是最新）")
            return
        changelog = _get_changelog_since(local_ver, remote_text)
        info = {
            "local_ver": local_ver,
            "remote_ver": remote_ver,
            "changelog": changelog,
        }
        if callback:
            callback(True, info)
    except Exception as e:
        print(f"[Update] 检查异常: {e}")
        if callback:
            callback(False, str(e))


def _download_and_extract_zip():
    """下载 GitHub zip 包并解压覆盖"""
    import uuid, zipfile

    _tmp_zip = os.path.join(os.environ.get("TEMP", "."), f"dfl_update_{uuid.uuid4().hex[:8]}.zip")
    _tmp_dir = os.path.join(os.environ.get("TEMP", "."), f"dfl_update_{uuid.uuid4().hex[:8]}")

    print(f"[Update]  下载更新包...", flush=True)
    try:
        import urllib.request
        _resp = urllib.request.urlopen(ZIP_URL, timeout=60)
        _total = int(_resp.headers.get('Content-Length', 0))
        _dl = 0
        _chunk = 256 * 1024
        with open(_tmp_zip, 'wb') as _f:
            while True:
                _buf = _resp.read(_chunk)
                if not _buf:
                    break
                _f.write(_buf)
                _dl += len(_buf)
                if _total > 0:
                    _pct = min(100, _dl * 100 // _total)
                    print(f"\r  {_pct}% ({_dl//1024**2}MB/{_total//1024**2}MB)", end="", flush=True)
                else:
                    print(f"\r  {_dl//1024**2}MB", end="", flush=True)
        print("", flush=True)
    except Exception as _e:
        print(f"[Update]  下载失败: {_e}", flush=True)
        _print_network_tip()
        return False

    print("[Update]  正在解压更新...", flush=True)
    try:
        os.makedirs(_tmp_dir, exist_ok=True)
        with zipfile.ZipFile(_tmp_zip, 'r') as _zf:
            _zf.extractall(_tmp_dir)
        # zip 内第一层是目录（如 DeepFaceLab-PyTorch-master/），找到它
        _dirs = [d for d in os.listdir(_tmp_dir) if os.path.isdir(os.path.join(_tmp_dir, d))]
        _src = os.path.join(_tmp_dir, _dirs[0]) if _dirs else _tmp_dir

        # 覆盖项目文件
        _skip = {'.git', '.claude', 'workspace'}
        for _item in os.listdir(_src):
            if _item in _skip:
                continue
            _s = os.path.join(_src, _item)
            _d = os.path.join(str(PROJECT_ROOT), _item)
            if os.path.isdir(_s):
                for _root, _dirs, _files in os.walk(_s):
                    _rel = os.path.relpath(_root, _src)
                    _target = os.path.join(str(PROJECT_ROOT), _rel)
                    os.makedirs(_target, exist_ok=True)
                    for _f in _files:
                        shutil.copy2(os.path.join(_root, _f), os.path.join(_target, _f))
            else:
                shutil.copy2(_s, _d)
    except Exception as _e:
        print(f"[Update]  解压覆盖失败: {_e}", flush=True)
        return False
    finally:
        try:
            os.remove(_tmp_zip)
            shutil.rmtree(_tmp_dir, ignore_errors=True)
        except Exception:
            pass

    _new_ver = _get_local_version()
    print(f"[Update]  ✅ 更新完成（版本 v{_new_ver}）", flush=True)
    return True


def pull_updates(callback=None):
    """下载 zip 更新"""
    if _download_and_extract_zip():
        print("[Update] ✅ 更新成功，请重启应用", flush=True)
        if callback:
            callback(True, "")
    else:
        print("[Update] ❌ 更新失败", flush=True)
        _print_network_tip()
        if callback:
            callback(False, "failed")


def ensure_git_installed(callback=None):
    """检查并安装 Git"""
    if _is_git_installed():
        if callback:
            callback(True)
        return
    print("[Update] Git 未安装，尝试自动安装...")
    ok = _install_git()
    if ok:
        os.environ["PATH"] += os.pathsep + r"C:\Program Files\Gitin"
        os.environ["PATH"] += os.pathsep + r"C:\Program Files\Git\cmd"
    if callback:
        callback(ok)


def _truncate_to_height(text, font, width, max_h):
    """按实际渲染高度截断文本（对话框不能无限长，否则更新按钮会被挤出屏幕）"""
    from PyQt5.QtGui import QTextDocument
    _doc = QTextDocument()
    _doc.setDefaultFont(font)
    _doc.setPlainText(text)
    _doc.setTextWidth(width)
    if _doc.size().height() <= max_h:
        return text
    _tail = "\n…（内容过长已截断，完整更新说明见「更新日志」页）"
    lines = text.split("\n")
    while lines:
        lines = lines[:-1]
        cand = "\n".join(lines) + _tail
        _doc.setPlainText(cand)
        _doc.setTextWidth(width)
        if _doc.size().height() <= max_h:
            return cand
    return _tail.strip()


def show_update_dialog(window, title, text, btn_text, btn_callback):
    """显示更新对话框（SiModalDialog，失败回退 QMessageBox）"""
    try:
        from siui.templates.application.components.dialog.modal import SiModalDialog
        from siui.components.widgets.label import SiLabel
        from siui.components.widgets.button import SiPushButton
        from siui.core.globals import SiGlobal
        from siui.core import SiColor
        _layer = window.layerModalDialog()
        if _layer is None:
            raise RuntimeError("layerModalDialog is None")
        # 版式宽度：content_container 宽 = 对话框宽 - 2*body_padding_h(43)
        # 旧版对话框只有 480 宽、文本区只有 250px（约 18 个汉字一行），一条更新说明就被
        # 折成两三行，条目的层级完全看不出来，观感就是一坨。这里放宽到 620。
        _DLG_W, _PAD_R = 700, 80   # _PAD_R 让出右上角那个 32px 的 alert 图标
        _TXT_W = _DLG_W - 86
        _CONTENT_W = _TXT_W - _PAD_R
        _dlg = SiModalDialog(_layer)
        _dlg.setFixedWidth(_DLG_W)
        _dlg.resize(_DLG_W, 1)  # 让实际宽度立刻生效，否则后面 adjustSize 里 self.width() 是错的
        _dlg.icon().load(SiGlobal.siui.iconpack.get("ic_fluent_alert_filled"))
        _dlg.icon().setSvgSize(32, 32)
        _dlg.icon().resize(32, 32)
        _dlg.reloadStyleSheet()
        from PyQt5.QtCore import Qt
        from PyQt5.QtGui import QTextDocument, QFont
        _txt = SiLabel(_dlg.contentContainer())
        _txt.setWordWrap(True)
        _txt.setTextFormat(Qt.PlainText)
        _font = QFont(_txt.font())
        _font.setPixelSize(14)
        # 太长会把「更新」按钮挤出屏幕，超出高度就截断并指路更新日志页
        text = _truncate_to_height(text, _font, _CONTENT_W, 470)
        _txt.setText(text)
        # 用 QTextDocument 精确计算换行高度：宽度必须和实际渲染宽度一致，否则高度会算错
        _doc = QTextDocument()
        _doc.setDefaultFont(_font)
        _doc.setPlainText(text)
        _doc.setTextWidth(_CONTENT_W)
        _txt_height = int(_doc.size().height()) + 16  # +上下 padding
        _txt.setFixedSize(_TXT_W, _txt_height)
        _txt.setStyleSheet(f"font-size: 14px; padding: 8px {_PAD_R}px 8px 0; color: #FFFFFF;")
        _dlg.contentContainer().addWidget(_txt)
        # 更新按钮：绿色背景 + 白色文字
        _btn = SiPushButton(_dlg.buttonContainer())
        _btn.attachment().setText(btn_text)
        _btn.colorGroup().assign(SiColor.BUTTON_PANEL, "#519868")
        _btn.colorGroup().assign(SiColor.BUTTON_SHADOW, "#3a6e4f")
        _btn.reloadStyleSheet()
        _btn.label.setStyleSheet("color: #FFFFFF;")
        _dlg.buttonContainer().addWidget(_btn)
        # 取消按钮：标准面板色 + 白色文字
        _cancel = SiPushButton(_dlg.buttonContainer())
        _cancel.attachment().setText("取消")
        _cancel.reloadStyleSheet()
        _cancel.label.setStyleSheet("color: #FFFFFF;")
        _dlg.buttonContainer().addWidget(_cancel)
        _dlg.adjustSize()
        _dlg.reloadStyleSheet()
        def _on_confirm():
            _layer.closeLayer()
            btn_callback()
        def _on_cancel():
            _layer.closeLayer()
        _btn.clicked.connect(_on_confirm)
        _cancel.clicked.connect(_on_cancel)
        _layer.setDialog(_dlg)
    except Exception as _e:
        import traceback
        traceback.print_exc()
        print(f"[Update] SiModalDialog 失败: {_e}")
        from PyQt5.QtWidgets import QMessageBox
        _r = QMessageBox.question(window, title, text, QMessageBox.Yes | QMessageBox.No)
        if _r == QMessageBox.Yes:
            btn_callback()

