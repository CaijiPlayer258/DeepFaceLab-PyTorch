# -*- coding: utf-8 -*-
"""B 站扫码登录：二维码生成 / 轮询 / 用户信息 / Cookie 持久化。

接口与 4.0 的 core/biliauth 完全一致：
  generate  https://passport.bilibili.com/x/passport-login/web/qrcode/generate
  poll      https://passport.bilibili.com/x/passport-login/web/qrcode/poll?qrcode_key=...
  nav       https://api.bilibili.com/x/web-interface/nav

BILI_API_BASE 可用环境变量覆盖，便于本地联调（指向内置 mock 服务）。
"""

import os
import json
import base64
import hashlib
import subprocess
from io import BytesIO
from pathlib import Path

import requests
import qrcode

# bilibili 的证书链在部分 Windows 机器上验不过（certifi 里缺对应根，
# requests 直接报 "unable to get local issuer certificate"，二维码就一直"获取失败"）。
# truststore 走系统证书存储：浏览器能打开的站，这里也能打开。
try:
    import truststore as _ts
    _ts.inject_into_ssl()
except Exception:
    pass

# 本机回环地址绝不走系统代理：否则本地 mock / 本地服务会被代理劫持成 404
_LOOPBACK = ("127.0.0.1", "localhost", "::1", "0.0.0.0")


def _get(url, **kw):
    host = url.split("//", 1)[-1].split("/", 1)[0].split(":")[0]
    if host in _LOOPBACK:
        kw["proxies"] = {"http": None, "https": None}
    return requests.get(url, **kw)

# ---- 接口地址（可被环境变量覆盖）----
BASE = os.environ.get("BILI_API_BASE", "https://passport.bilibili.com").rstrip("/")
API_BASE = os.environ.get("BILI_API_BASE_API", "https://api.bilibili.com").rstrip("/")

GEN_URL = BASE + "/x/passport-login/web/qrcode/generate"
POLL_URL = BASE + "/x/passport-login/web/qrcode/poll"
NAV_URL = API_BASE + "/x/web-interface/nav"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/120.0 Safari/537.36",
    "Referer": "https://www.bilibili.com/",
}

# 轮询状态码
NOT_SCANNED = 86101
SCANNED_PENDING = 86090
CONFIRMED = 0
EXPIRED = 86038

STATUS_TEXT = {
    NOT_SCANNED: "等待扫码",
    SCANNED_PENDING: "已扫描，请在手机上确认",
    CONFIRMED: "登录成功",
    EXPIRED: "二维码已过期",
}


# ---------------- Cookie 持久化 ----------------
def _machine_key() -> bytes:
    try:
        r = subprocess.run("wmic csproduct get uuid", shell=True, timeout=4,
                           capture_output=True, text=True)
        if r.returncode == 0:
            raw = r.stdout.strip().split("\n")[-1].strip()
            if raw:
                return hashlib.sha256(raw.encode()).digest()
    except Exception:
        pass
    return hashlib.sha256(os.environ.get("COMPUTERNAME", "UNKNOWN").encode()).digest()


def _xor(data: bytes) -> bytes:
    k = _machine_key()
    return bytes(data[i] ^ k[i % len(k)] for i in range(len(data)))


class CookieStore:
    """存 workspace/auth/bilibili_cookie.txt，机器密钥 XOR + base64。"""

    def __init__(self, root: Path):
        self.dir = root / "auth"
        self.cookie_path = self.dir / "bilibili_cookie.txt"
        self.info_path = self.dir / "bilibili_info.json"

    def save(self, cookie: str):
        self.dir.mkdir(parents=True, exist_ok=True)
        enc = base64.b64encode(_xor(cookie.strip().encode("utf-8"))).decode()
        self.cookie_path.write_text(enc, encoding="utf-8")

    def load(self) -> str:
        try:
            raw = self.cookie_path.read_text(encoding="utf-8").strip()
            return _xor(base64.b64decode(raw)).decode("utf-8", "ignore")
        except Exception:
            return ""

    def clear(self):
        for p in (self.cookie_path, self.info_path):
            try:
                p.unlink()
            except Exception:
                pass

    def save_info(self, info: dict):
        self.dir.mkdir(parents=True, exist_ok=True)
        self.info_path.write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")

    def load_info(self) -> dict:
        try:
            return json.loads(self.info_path.read_text(encoding="utf-8"))
        except Exception:
            return {}


# ---------------- 二维码 ----------------
def make_qr(url: str) -> str:
    """把登录 URL 编成二维码，返回 PNG 的 base64（不带 data: 前缀）。"""
    img = qrcode.make(url)
    buf = BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def generate():
    """返回 {url, key, qr_png}；失败抛 RuntimeError。"""
    resp = _get(GEN_URL, headers=HEADERS, timeout=15)
    resp.raise_for_status()
    data = resp.json()
    if data.get("code") != 0:
        raise RuntimeError(f"二维码接口错误：{data.get('message', 'unknown')}")
    d = data["data"]
    return {"url": d["url"], "key": d["qrcode_key"], "qr_png": make_qr(d["url"])}


def poll(qrcode_key: str):
    """轮询一次。返回 {code, status, message, cookie?}。"""
    resp = _get(POLL_URL, params={"qrcode_key": qrcode_key},
                headers=HEADERS, timeout=15)
    resp.raise_for_status()
    data = resp.json()
    if data.get("code") != 0:
        return {"code": -1, "status": "error", "message": data.get("message", "unknown")}

    d = data.get("data") or {}
    code = d.get("code", -1)

    if code == CONFIRMED:
        jar = resp.cookies
        cookie = "; ".join(f"{c.name}={c.value}" for c in jar) if jar else ""
        return {"code": code, "status": "ok", "message": STATUS_TEXT[CONFIRMED], "cookie": cookie}

    return {"code": code, "status": "pending" if code != EXPIRED else "expired",
            "message": STATUS_TEXT.get(code, f"未知状态 {code}")}


def fetch_user_info(cookie: str) -> dict:
    headers = dict(HEADERS)
    headers["Cookie"] = cookie
    try:
        r = _get(NAV_URL, headers=headers, timeout=10)
        r.raise_for_status()
        d = r.json()
        if d.get("code") == 0 and d.get("data"):
            info = d["data"]
            return {
                "uid": info.get("mid", 0),
                "name": info.get("uname", ""),
                "avatar": info.get("face", ""),
                "level": (info.get("level_info") or {}).get("current_level", 0),
            }
    except Exception:
        pass
    return {}
