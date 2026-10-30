"""
Bilibili cookie & user info persistence — store in ui/user/.
Cookie is encrypted with machine-specific key to prevent use on other computers.
"""

import json, hashlib, base64, subprocess, os
from pathlib import Path

_USER_DIR = Path(__file__).parent.parent.parent / "ui" / "user"
COOKIE_PATH = _USER_DIR / "bilibili_cookie.txt"
INFO_PATH = _USER_DIR / "user_info.json"

_ENC_PREFIX = "ENC:"  # marks encrypted cookies


def _get_machine_key() -> bytes:
    """Derive a machine-specific encryption key from hardware UUID."""
    try:
        r = subprocess.run('wmic csproduct get uuid', shell=True, timeout=5,
                           capture_output=True, text=True)
        if r.returncode == 0:
            raw = r.stdout.strip().split('\n')[-1].strip()
            if raw:
                return hashlib.sha256(raw.encode()).digest()
    except Exception:
        pass
    # Fallback: use computer name only (wmic removed in Win11 24H2+)
    machine = os.environ.get('COMPUTERNAME', 'UNKNOWN')
    return hashlib.sha256(machine.encode()).digest()


def _encrypt(plaintext: str) -> str:
    """Encrypt with machine key, return 'ENC:' + base64."""
    key = _get_machine_key()
    data = plaintext.encode('utf-8')
    encrypted = bytes(data[i] ^ key[i % len(key)] for i in range(len(data)))
    return _ENC_PREFIX + base64.b64encode(encrypted).decode()


def _decrypt(ciphertext: str) -> str:
    """Decrypt; if no ENC: prefix, return as-is (plaintext fallback)."""
    if not ciphertext.startswith(_ENC_PREFIX):
        return ciphertext
    key = _get_machine_key()
    try:
        data = base64.b64decode(ciphertext[len(_ENC_PREFIX):])
        decrypted = bytes(data[i] ^ key[i % len(key)] for i in range(len(data)))
        return decrypted.decode('utf-8')
    except Exception:
        return ''  # decryption failed


def save_cookie(cookie_str: str):
    """Save Bilibili login cookie (encrypted with machine key)."""
    _USER_DIR.mkdir(parents=True, exist_ok=True)
    COOKIE_PATH.write_text(_encrypt(cookie_str.strip()), encoding='utf-8')


def load_cookie() -> str:
    """Load saved cookie, try decryption first, fallback to plaintext."""
    if COOKIE_PATH.exists():
        raw = COOKIE_PATH.read_text(encoding='utf-8').strip()
        return _decrypt(raw)
    return ''


def is_logged_in() -> bool:
    """Check if a non-empty cookie file exists with SESSDATA."""
    c = load_cookie()
    return bool(c) and 'SESSDATA' in c


def save_user_info(info: dict):
    """Save user info (uid, name, avatar_url, etc.) as JSON."""
    _USER_DIR.mkdir(parents=True, exist_ok=True)
    INFO_PATH.write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding='utf-8')


def load_user_info() -> dict:
    """Load saved user info, or empty dict."""
    if INFO_PATH.exists():
        try:
            return json.loads(INFO_PATH.read_text(encoding='utf-8'))
        except Exception:
            pass
    return {}


# 开发者 UID 列表 — 后续添加更多开发者在此处增加
DEV_UIDS = {500398541,50751743,
20686294}


def is_dev_uid(uid: int) -> bool:
    """Check if a UID belongs to a developer."""
    return uid in DEV_UIDS


def get_user_name() -> str:
    """Get display name for welcome message."""
    info = load_user_info()
    uid = info.get('uid', 0)
    name = info.get('name', '')
    if is_dev_uid(uid):
        return '开发者'
    if name:
        return name
    return '用户'


def clear_all():
    """Remove saved cookie and user info (for logout)."""
    if COOKIE_PATH.exists():
        COOKIE_PATH.unlink()
    if INFO_PATH.exists():
        INFO_PATH.unlink()
