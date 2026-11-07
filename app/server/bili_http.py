# -*- coding: utf-8 -*-
"""B 站 HTTP 公共层：证书信任、错误归一化、Cookie 脱敏日志。

为什么要有这一层：
1) 证书。这台机器（以及不少国内 Windows 机器）的 HTTPS 被本机代理接管，
   根证书在 Windows 证书存储里，而 requests 自带的是 certifi，两边对不上，
   于是 api.bilibili.com 直接抛
   SSLError: unable to get local issuer certificate。
   4.0 里靠 truststore 把 ssl 切到系统存储，但打包环境不一定装了 truststore。
   这里按 truststore → Windows 根证书合并包 → certifi 的顺序挑一个能用的
   verify 参数，保证「请求真的发出去了」再谈业务错误。
2) 错误。B 站把所有失败都塞在 HTTP 200 的 code 字段里，以前上层
   except Exception: return [] 把 -101（未登录）和 -352（风控）都吞成
   空列表。现在统一升成 BiliError，带 kind/status，路由层原样回给前端。
"""
from __future__ import annotations

import os
import ssl
import tempfile
import time
from pathlib import Path

import requests

UA = ('Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 '
      '(KHTML, like Gecko) Chrome/120.0 Safari/537.36')

HEADERS = {
    'User-Agent': UA,
    'Referer': 'https://www.bilibili.com/',
}

# 本机回环地址绝不走系统代理：否则本地 mock / 本地服务会被代理劫持成 404
_LOOPBACK = ('127.0.0.1', 'localhost', '::1', '0.0.0.0')

# B 站业务码 → 错误类别（类别决定 HTTP 状态码）
_CODE_KIND = {
    -101: 'not_logged_in',   # 账号未登录 / cookie 失效
    -102: 'not_logged_in',
    -400: 'api',             # 请求错误
    -403: 'api',             # 权限不足
    -404: 'not_found',       # 视频 / 用户不存在
    -352: 'risk_control',    # 风控校验失败
    -412: 'risk_control',    # 请求被拦截
    -509: 'risk_control',    # 请求过于频繁
    -799: 'risk_control',    # 请求过于频繁
    -10403: 'risk_control',  # 地区限制
    87008: 'api',            # 该清晰度需要大会员
}

# kind → HTTP 状态码
_KIND_STATUS = {
    'bad_request': 400,
    'not_logged_in': 401,
    'risk_control': 412,
    'not_found': 404,
    'network': 502,
    'api': 502,
    'ffmpeg': 500,
}


class BiliError(RuntimeError):
    """B 站调用的统一异常：kind 决定 HTTP 状态，message 直接给用户看。"""

    def __init__(self, message: str, kind: str = 'api', code=None, http_status=None):
        super().__init__(message)
        self.kind = kind if kind in _KIND_STATUS else 'api'
        self.code = code
        self.http_status = http_status

    @property
    def status(self) -> int:
        return _KIND_STATUS.get(self.kind, 502)

    def payload(self) -> dict:
        return {
            'code': self.kind,
            'message': str(self),
            'bili_code': self.code,
            'http_status': self.http_status,
        }


def mask_cookie(cookie: str) -> str:
    """日志里只留 cookie 的名字和长度，绝不打印值。"""
    if not cookie:
        return '<empty>'
    parts = []
    for item in cookie.split(';'):
        name = item.split('=', 1)[0].strip()
        if name:
            parts.append(name + '=***')
    return ','.join(parts) + ' (len=%d)' % len(cookie)


def _windows_ca_bundle() -> str:
    """把 Windows 根证书存储导出成一个 PEM，供 requests 的 verify 使用。

    7 天刷新一次；导出失败返回空串（退化成 certifi）。
    """
    if os.name != 'nt' or not hasattr(ssl, 'enum_certificates'):
        return ''
    cache = Path(tempfile.gettempdir()) / 'dfl_win_ca_bundle.pem'
    try:
        st = cache.stat()
        if st.st_size > 4096 and (time.time() - st.st_mtime) < 7 * 86400:
            return str(cache)
    except OSError:
        pass
    try:
        pems = []
        for store in ('ROOT', 'CA'):
            for cert, enc, _trust in ssl.enum_certificates(store):
                if enc != 'x509_asn':
                    continue
                try:
                    pems.append(ssl.DER_cert_to_PEM_cert(cert))
                except Exception:
                    continue
        if not pems:
            return ''
        cache.write_text(''.join(pems), encoding='ascii')
        return str(cache)
    except Exception:
        return ''


_verify_cache = {'value': None}


def ssl_verify():
    """requests 的 verify 参数：证书链能验通的那个来源。"""
    if _verify_cache['value'] is not None:
        return _verify_cache['value']
    if os.environ.get('DFL_BILI_INSECURE') == '1':
        try:
            import urllib3
            urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        except Exception:
            pass
        _verify_cache['value'] = False
        return False
    bundle = _windows_ca_bundle()
    _verify_cache['value'] = bundle or True
    return _verify_cache['value']


def request(method: str, url: str, *, headers: dict = None, timeout: int = 15,
            stream: bool = False, **kw) -> requests.Response:
    """统一的 requests 调用：套公共头、定 verify、把网络层异常升成 BiliError。"""
    h = dict(HEADERS)
    if headers:
        h.update(headers)
    host = url.split('//', 1)[-1].split('/', 1)[0].split(':')[0]
    if host in _LOOPBACK:
        kw['proxies'] = {'http': None, 'https': None}
    try:
        return requests.request(method, url, headers=h, timeout=timeout,
                                stream=stream, verify=ssl_verify(), **kw)
    except requests.exceptions.SSLError as e:
        raise BiliError(
            'HTTPS 证书校验失败（%s）：本机根证书与 requests 的 certifi 不匹配，'
            '装 truststore 或设置 DFL_BILI_INSECURE=1' % type(e).__name__,
            kind='network') from e
    except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
        raise BiliError('网络错误：连不上 %s（%s）' % (host, type(e).__name__),
                        kind='network') from e


def get_json(url: str, *, headers: dict = None, timeout: int = 15, what: str = '接口') -> dict:
    """GET 一个 B 站 JSON 接口，code != 0 / 412 / 非 JSON 一律抛 BiliError。"""
    resp = request('GET', url, headers=headers, timeout=timeout)
    if resp.status_code == 412:
        raise BiliError('%s 触发风控（HTTP 412），请稍后重试或换出口 IP' % what,
                        kind='risk_control', http_status=412)
    if resp.status_code == 401:
        raise BiliError('%s 未授权（HTTP 401）' % what,
                        kind='not_logged_in', http_status=401)
    try:
        data = resp.json()
    except ValueError:
        raise BiliError('%s 返回的不是 JSON（HTTP %d）' % (what, resp.status_code),
                        kind='api', http_status=resp.status_code)

    code = data.get('code')
    if code not in (0, None):
        kind = _CODE_KIND.get(code, 'api')
        msg = data.get('message') or data.get('msg') or ''
        raise BiliError(('%s 失败：code=%s %s' % (what, code, msg)).strip(),
                        kind=kind, code=code, http_status=resp.status_code)
    return data
