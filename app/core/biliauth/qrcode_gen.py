"""
Generate Bilibili QR code for login via the official API.
Adapted from GetQrcode.py — returns data instead of writing to files.
"""

import requests
import brotli
import json
import qrcode
from io import BytesIO
from PIL import Image

_HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 '
                  '(KHTML, like Gecko) Chrome/120.0 Safari/537.36',
    'Referer': 'https://www.bilibili.com/',
    'Accept-Encoding': 'gzip, deflate, br',
}


def _decode_response(resp: requests.Response) -> dict:
    """Handle Brotli-compressed Bilibili API responses."""
    raw = resp.content
    # Try direct JSON first
    try:
        return resp.json()
    except json.JSONDecodeError:
        pass
    # Fall back to brotli decompression
    try:
        decompressed = brotli.decompress(raw)
        return json.loads(decompressed.decode('utf-8'))
    except Exception as e:
        raise RuntimeError(f"Failed to decode Bilibili API response: {e}")


def generate_qrcode() -> tuple:
    """
    Call Bilibili QR generation API, return (qrcode_url, qrcode_key, qr_pil_image).

    Returns:
        (qrcode_url: str, qrcode_key: str, qr_image: PIL.Image.Image)

    Raises:
        RuntimeError if the API call fails or response is invalid.
    """
    url = "https://passport.bilibili.com/x/passport-login/web/qrcode/generate"
    resp = requests.get(url, headers=_HEADERS)
    resp.raise_for_status()

    data = _decode_response(resp)
    if data.get('code') != 0:
        raise RuntimeError(f"Bilibili QR API error: {data.get('message', 'unknown')}")

    qrcode_url = data['data']['url']
    qrcode_key = data['data']['qrcode_key']

    # Generate QR code PIL image
    qr = qrcode.make(qrcode_url)
    return qrcode_url, qrcode_key, qr
