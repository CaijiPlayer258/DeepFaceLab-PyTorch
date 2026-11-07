"""
Fetch Bilibili user info (uid, name, avatar) using the login cookie.

2026-05 修：以前所有异常都吞成 {}，前端只能显示「未登录」。现在走 bili_http，
失败抛 BiliError（401 未登录 / 502 网络 / 412 风控），由路由层转成明确状态码。
"""

from . import bili_http
from .bili_http import BiliError

_NAV_URL = 'https://api.bilibili.com/x/web-interface/nav'


def fetch_user_info(cookie_str: str) -> dict:
    """Return {uid, name, avatar, avatar_url, level}. Raises BiliError on failure."""
    j = bili_http.get_json(_NAV_URL, headers={'Cookie': cookie_str}, timeout=10,
                           what='nav（用户信息）')
    data = j.get('data') or {}
    if not data.get('isLogin', True) or not data.get('mid'):
        raise BiliError('账号未登录（nav 返回空用户）', kind='not_logged_in',
                        code=j.get('code'))
    return {
        'uid': data.get('mid', 0),
        'name': data.get('uname', ''),
        'avatar': data.get('face', ''),
        'avatar_url': data.get('face', ''),   # 两个键都留着，调用方读哪个都不会丢
        'level': (data.get('level_info') or {}).get('current_level', 0),
    }
