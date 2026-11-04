"""
Silently follow a Bilibili user by UID using the login cookie.
"""

import re
import time
import hashlib
import requests
from functools import reduce
from urllib.parse import quote

_FOLLOW_URL = "https://api.bilibili.com/x/relation/modify"
_HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 '
                  '(KHTML, like Gecko) Chrome/120.0 Safari/537.36',
    'Referer': 'https://www.bilibili.com/',
    'Content-Type': 'application/x-www-form-urlencoded',
}

# WBI signing (same as qrcode_gen.py)
_mixinKeyEncTab = [
    46, 47, 18, 2, 53, 8, 23, 32, 15, 50, 10, 31, 58, 3, 45, 35, 27, 43, 5, 49,
    33, 9, 42, 19, 29, 28, 14, 39, 12, 38, 41, 13, 37, 48, 7, 16, 24, 55, 40,
    61, 26, 17, 0, 1, 60, 51, 30, 4, 22, 25, 54, 21, 56, 59, 6, 63, 57, 62, 11,
    36, 20, 34, 44, 52
]


def _getMixinKey(orig: str):
    return reduce(lambda s, i: s + orig[i], _mixinKeyEncTab, '')[:32]


def _encWbi(params: dict, img_key: str, sub_key: str):
    mixin_key = _getMixinKey(img_key + sub_key)
    params['wts'] = round(time.time())
    params = dict(sorted(params.items()))
    params = {k: ''.join(filter(lambda c: c not in "!'()*", str(v))) for k, v in params.items()}
    query = '&'.join(f'{k}={quote(str(v))}' for k, v in params.items())
    params['w_rid'] = hashlib.md5((query + mixin_key).encode()).hexdigest()
    return params


def _getWbiKeys(cookie_str: str):
    resp = requests.get(
        'https://api.bilibili.com/x/web-interface/nav',
        headers={**_HEADERS, 'Cookie': cookie_str},
        timeout=10,
    )
    resp.raise_for_status()
    j = resp.json()
    img_url = j['data']['wbi_img']['img_url']
    sub_url = j['data']['wbi_img']['sub_url']
    return (img_url.rsplit('/', 1)[1].split('.')[0],
            sub_url.rsplit('/', 1)[1].split('.')[0])


def follow_user(target_uid: int, cookie_str: str) -> bool:
    """
    Silently follow a Bilibili user by UID. Returns True on success.
    Uses the login cookie for authentication; no user interaction.
    """
    # Extract CSRF token from cookie
    m = re.search(r'bili_jct=([^;]+)', cookie_str)
    if not m:
        return False
    csrf = m.group(1)

    try:
        img_key, sub_key = _getWbiKeys(cookie_str)
        params = _encWbi({'fid': target_uid, 'act': 1, 're_src': 11, 'csrf': csrf}, img_key, sub_key)
        data = {k: str(v) for k, v in params.items()}
        resp = requests.post(_FOLLOW_URL, headers={**_HEADERS, 'Cookie': cookie_str}, data=data, timeout=10)
        resp.raise_for_status()
        return resp.json().get('code') == 0
    except Exception:
        return False
