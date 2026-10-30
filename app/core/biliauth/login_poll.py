"""
Poll Bilibili QR login status and extract cookie on success.
Adapted from CheckLoginAndGetCookie.py with improved cookie parsing.
"""

import time
import requests
from http.cookies import SimpleCookie

_HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 '
                  '(KHTML, like Gecko) Chrome/120.0 Safari/537.36',
    'Referer': 'https://www.bilibili.com/',
}

# Bilibili QR code status codes
NOT_SCANNED = 86101      # QR not scanned yet
SCANNED_PENDING = 86090  # Scanned, waiting for confirmation
SCANNED_CONFIRMED = 0    # Login success
EXPIRED = 86038          # QR code expired


def _extract_cookie(response: requests.Response) -> str:
    """
    Extract Bilibili login cookies from response Set-Cookie headers.
    Returns semicolon-separated cookie string (same format as browser cookies).
    """
    cookie_jar = response.cookies
    if not cookie_jar:
        return ''

    # Build semicolon-separated string from all cookies in the jar
    parts = []
    for domain_cookie in cookie_jar:
        parts.append(f"{domain_cookie.name}={domain_cookie.value}")
    return '; '.join(parts)


def poll_login(qrcode_key: str, interval: float = 2.0, max_retries: int = 45) -> tuple:
    """
    Poll Bilibili QR login status.

    Args:
        qrcode_key: The key from generate_qrcode()
        interval: Polling interval in seconds (default 2s)
        max_retries: Maximum number of poll attempts (default 45 = 90s timeout)

    Yields:
        (status_code: int, message: str) tuples for UI updates.
        Status codes: 0=success, 86101=waiting, 86090=scanned_pending,
                      86038=expired, -1=error

    Final yield from the generator (after loop ends):
        (final_code, result_string) where result_string is:
          - On success: the cookie string
          - On failure: error description
    """
    poll_url = "https://passport.bilibili.com/x/passport-login/web/qrcode/poll"

    for attempt in range(max_retries):
        try:
            resp = requests.get(
                poll_url,
                params={'qrcode_key': qrcode_key},
                headers=_HEADERS,
            )
            resp.raise_for_status()
            data = resp.json()

            api_code = data.get('code', -1)
            if api_code != 0:
                yield (-1, f"API 错误: {data.get('message', 'unknown')}")
                time.sleep(interval)
                continue

            poll_code = data['data'].get('code', -1)

            if poll_code == SCANNED_CONFIRMED:
                # Login successful — extract cookie and return in one yield
                cookie_str = _extract_cookie(resp)
                yield (SCANNED_CONFIRMED, f"登录成功||{cookie_str}")
                return

            elif poll_code == SCANNED_PENDING:
                yield (SCANNED_PENDING, "已扫描，请在手机上确认")

            elif poll_code == NOT_SCANNED:
                yield (NOT_SCANNED, "等待扫码...")

            elif poll_code == EXPIRED:
                yield (EXPIRED, "二维码已过期，请重新生成")
                return

            else:
                yield (poll_code, f"未知状态: {poll_code}")

        except requests.RequestException as e:
            yield (-1, f"网络错误: {e}")

        time.sleep(interval)

    yield (-1, "登录超时，请重新尝试")
