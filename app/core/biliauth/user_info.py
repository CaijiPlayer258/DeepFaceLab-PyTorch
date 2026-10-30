"""
Fetch Bilibili user info (uid, name, avatar) using the login cookie.
"""

import requests

_NAV_URL = "https://api.bilibili.com/x/web-interface/nav"
_HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 '
                  '(KHTML, like Gecko) Chrome/120.0 Safari/537.36',
    'Referer': 'https://www.bilibili.com/',
}


def fetch_user_info(cookie_str: str) -> dict:
    """
    Fetch user info from Bilibili using the login cookie.
    Returns dict with keys: uid, name, avatar_url, level, etc.
    Returns empty dict on failure.
    """
    headers = {**_HEADERS, 'Cookie': cookie_str}
    try:
        resp = requests.get(_NAV_URL, headers=headers, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        if data.get('code') == 0 and data.get('data'):
            info = data['data']
            return {
                'uid': info.get('mid', 0),
                'name': info.get('uname', ''),
                'avatar_url': info.get('face', ''),
                'level': info.get('level_info', {}).get('current_level', 0),
            }
    except Exception:
        pass
    return {}
