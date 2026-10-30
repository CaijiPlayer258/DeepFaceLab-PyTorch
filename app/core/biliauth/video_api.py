"""
Bilibili video API — fetch user info, list videos, download.
Uses cookie from core.biliauth.cookie_store.
"""
import re, time, hashlib, os, subprocess, requests
from functools import reduce
from urllib.parse import quote, urlencode
from pathlib import Path

_HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 '
                  '(KHTML, like Gecko) Chrome/120.0 Safari/537.36',
    'Referer': 'https://www.bilibili.com/',
}

# WBI signing (shared with other modules)
_mixinKeyEncTab = [
    46, 47, 18, 2, 53, 8, 23, 32, 15, 50, 10, 31, 58, 3, 45, 35, 27, 43, 5, 49,
    33, 9, 42, 19, 29, 28, 14, 39, 12, 38, 41, 13, 37, 48, 7, 16, 24, 55, 40,
    61, 26, 17, 0, 1, 60, 51, 30, 4, 22, 25, 54, 21, 56, 59, 6, 63, 57, 62, 11,
    36, 20, 34, 44, 52
]

# WBI key cache with disk persistence (survives app restart)
_wbi_cache = {'img_key': '', 'sub_key': '', 'expires': 0}
_wbi_cache_file = None  # set on first use

def _get_wbi_cache_path():
    global _wbi_cache_file
    if _wbi_cache_file is None:
        _wbi_cache_file = Path(__file__).parent / '.wbi_cache.json'
    return _wbi_cache_file

def _load_wbi_cache():
    p = _get_wbi_cache_path()
    if p.exists():
        try:
            import json
            d = json.loads(p.read_text('utf-8'))
            _wbi_cache.update(d)
        except Exception:
            pass

def _save_wbi_cache():
    import json
    try:
        _get_wbi_cache_path().write_text(json.dumps(_wbi_cache), 'utf-8')
    except Exception:
        pass

# User info cache: {mid: {name, face, mid, _cached_at}}
_user_cache = {}

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

def _getWbiKeys(cookie_str: str = ''):
    now = time.time()
    if now < _wbi_cache['expires'] and _wbi_cache['img_key']:
        return (_wbi_cache['img_key'], _wbi_cache['sub_key'])
    # Load from disk if memory cache expired
    _load_wbi_cache()
    if now < _wbi_cache['expires'] and _wbi_cache['img_key']:
        return (_wbi_cache['img_key'], _wbi_cache['sub_key'])
    h = {**_HEADERS}
    if cookie_str:
        h['Cookie'] = cookie_str
    resp = requests.get('https://api.bilibili.com/x/web-interface/nav', headers=h, timeout=10)
    resp.raise_for_status()
    j = resp.json()
    if j.get('code') != 0:
        raise RuntimeError(f"WBI key fetch failed: code={j.get('code')}")
    img_url = j['data']['wbi_img']['img_url']
    sub_url = j['data']['wbi_img']['sub_url']
    _wbi_cache['img_key'] = img_url.rsplit('/', 1)[1].split('.')[0]
    _wbi_cache['sub_key'] = sub_url.rsplit('/', 1)[1].split('.')[0]
    _wbi_cache['expires'] = now + 600  # cache for 10 minutes
    _save_wbi_cache()  # persist to disk
    return (_wbi_cache['img_key'], _wbi_cache['sub_key'])


def _try_fetch_user_json(url: str, headers: dict) -> dict:
    """Make a GET request and return {name, face, mid} if successful."""
    try:
        resp = requests.get(url, headers=headers, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        if data.get('code') == 0 and data.get('data'):
            d = data['data']
            # Handle different response formats
            if 'card' in d:  # /x/web-interface/card format
                card = d['card']
                name = card.get('name', '')
                face = card.get('face', '')
                mid = card.get('mid', '')
            else:  # /x/space/acc/info format
                name = d.get('name', '')
                face = d.get('face', '')
                mid = d.get('mid', '')
            if name:
                return {'name': name, 'face': face, 'mid': str(mid)}
    except Exception as e:
        print(f"[BilibiliAPI] _try_fetch_user_json 失败: {e}")
    return {}


def fetch_user_info_by_mid(mid: str, cookie_str: str = '') -> dict:
    """Get user name and avatar URL by mid. Results cached in memory. Returns {} on failure."""
    # Check in-memory user cache first (5 min TTL)
    if mid in _user_cache:
        cached = _user_cache[mid]
        if time.time() - cached.get('_cached_at', 0) < 300:
            return {k: v for k, v in cached.items() if not k.startswith('_')}

    h = {**_HEADERS, 'Accept': 'application/json, text/plain, */*'}
    if cookie_str:
        h['Cookie'] = cookie_str

    # Strategy: try multiple endpoints with delays to work around -799 rate limit
    endpoints = []

    # 1. WBI-signed /x/space/acc/info
    try:
        img_key, sub_key = _getWbiKeys(cookie_str)
        signed = _encWbi({'mid': mid}, img_key, sub_key)
        endpoints.append(f"https://api.bilibili.com/x/space/acc/info?{urlencode(signed)}")
    except Exception:
        pass

    # 2. Plain (no WBI) /x/space/acc/info (might work when WBI version is rate-limited)
    endpoints.append(f"https://api.bilibili.com/x/space/acc/info?mid={mid}")

    # 3. /x/web-interface/card (with WBI)
    try:
        img_key2, sub_key2 = _getWbiKeys(cookie_str)
        card_signed = _encWbi({'mid': mid}, img_key2, sub_key2)
        endpoints.append(f"https://api.bilibili.com/x/web-interface/card?{urlencode(card_signed)}")
    except Exception:
        pass

    # 4. Plain /x/web-interface/card (no WBI)
    endpoints.append(f"https://api.bilibili.com/x/web-interface/card?mid={mid}")

    for i, url in enumerate(endpoints):
        if i > 0:
            time.sleep(0.5)  # delay between endpoint attempts
        result = _try_fetch_user_json(url, h)
        if result:
            # Cache it
            _user_cache[mid] = {**result, '_cached_at': time.time()}
            return result
        print(f"[BilibiliAPI] 端点 {i+1}/{len(endpoints)} 失败，尝试下一个...")

    return {}


def fetch_up_videos(mid: str, cookie_str: str = '', max_pages: int = None,
                    progress_callback=None) -> list:
    """Get all video list for a UP主. Returns [{bvid, title, is_coop}, ...].
    If progress_callback is provided, calls progress_callback(current_count, page) after each page.
    """
    h = {**_HEADERS, 'Accept': 'application/json, text/plain, */*'}
    if cookie_str:
        h['Cookie'] = cookie_str

    videos = []
    page = 1
    try:
        img_key, sub_key = _getWbiKeys(cookie_str)
    except Exception:
        return videos

    while True:
        if max_pages and page > max_pages:
            break
        try:
            params = _encWbi({'mid': mid, 'ps': 30, 'pn': page}, img_key, sub_key)
            url = f"https://api.bilibili.com/x/space/wbi/arc/search?{urlencode(params)}"
            resp = requests.get(url, headers=h, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            if data.get('code') != 0:
                break
            vlist = data.get('data', {}).get('list', {}).get('vlist', [])
            if not vlist:
                break
            for v in vlist:
                bvid = v.get('bvid', '')
                title = v.get('title', '').strip()
                # Detect co-op: check if title contains '合作' or API has staff list
                staff = v.get('staff')
                is_coop = '合作' in title or (isinstance(staff, list) and len(staff) > 0)
                videos.append({
                    'bvid': bvid,
                    'title': title,
                    'is_coop': is_coop,
                })
            page += 1
            if progress_callback:
                progress_callback(len(videos), page)
            if page > 1:  # after page 1, we know there's more data
                time.sleep(0.5)
        except Exception as _e:
            print(f"[BilibiliAPI] fetch_up_videos 第{page}页失败: {_e}")
            break
    return videos


def download_single_video(bvid: str, cookie_str: str, output_dir: str,
                          progress_callback=None, ffmpeg_path: str = None) -> bool:
    """Download a video by BVID. Returns True on success.
    If progress_callback is provided, calls progress_callback(fraction, speed_mbps, stage).
    Callback returning False aborts the download.
    """
    h = {**_HEADERS, 'Cookie': cookie_str}
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    try:
        # Get video info for title
        info_resp = requests.get(f'https://api.bilibili.com/x/web-interface/view?bvid={bvid}', headers=h, timeout=10)
        info_resp.raise_for_status()
        info_data = info_resp.json().get('data', {})
        title = re.sub(r'[\\/:*?"<>|]', '', info_data.get('title', bvid))
        cid = str(info_data.get('cid', info_data.get('pages', [{}])[0].get('cid', 0)))
        if not cid:
            return False

        img_key, sub_key = _getWbiKeys(cookie_str)
        play_params = _encWbi({'bvid': bvid, 'cid': cid, 'qn': '80', 'fnval': '4048', 'fnver': '0', 'fourk': '1'},
                              img_key, sub_key)
        play_url = f"https://api.bilibili.com/x/player/wbi/playurl?{urlencode(play_params)}"
        play_resp = requests.get(play_url, headers=h, timeout=10)
        play_resp.raise_for_status()
        play_data = play_resp.json().get('data', {})

        video_url = audio_url = None
        if 'dash' in play_data:
            video_url = play_data['dash']['video'][0]['baseUrl']
            audio_url = play_data['dash']['audio'][0]['baseUrl']
        elif 'durl' in play_data:
            video_url = play_data['durl'][0]['url']

        if not video_url:
            return False

        v_path = out / f"{title}_video.mp4"
        a_path = out / f"{title}_audio.mp4"
        m_path = out / f"{title}.mp4"

        # Pre-fetch sizes for progress calculation
        v_size = 0
        a_size = 0
        try:
            if video_url:
                vs = requests.head(video_url, headers=h, timeout=10)
                v_size = int(vs.headers.get('Content-Length', 0))
            if audio_url:
                a_s = requests.head(audio_url, headers=h, timeout=10)
                a_size = int(a_s.headers.get('Content-Length', 0))
        except Exception:
            pass
        total_bytes = v_size + a_size

        def _dl(url, path, stage_offset=0.0, stage_span=1.0, stage_name="downloading_video"):
            nonlocal v_size, a_size
            r = requests.get(url, headers=h, stream=True, timeout=120)
            r.raise_for_status()
            stream_size = int(r.headers.get('Content-Length', 0))
            downloaded = 0
            phase_start = time.time()
            with open(str(path), 'wb') as f:
                for chunk in r.iter_content(chunk_size=1024*1024):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        if progress_callback and total_bytes > 0:
                            # Fraction across total ops (video + audio + merge)
                            frac = stage_offset + (downloaded / max(stream_size, 1)) * stage_span
                            # Speed
                            elapsed = time.time() - phase_start
                            speed = downloaded / (1024 * 1024 * max(elapsed, 0.01))
                            if not progress_callback(min(frac, 0.99), speed, stage_name):
                                return False
            return True

        # Download video stream
        v_weight = (v_size / total_bytes) if total_bytes > 0 else 0.45
        a_weight = (a_size / total_bytes) if total_bytes > 0 else 0.45
        if not _dl(video_url, v_path, 0.0, v_weight, "downloading_video"):
            # Aborted
            _cleanup_temp(v_path, a_path)
            return False

        if audio_url:
            if not _dl(audio_url, a_path, v_weight, a_weight, "downloading_audio"):
                _cleanup_temp(v_path, a_path)
                return False
            # Report merge phase
            if progress_callback:
                progress_callback(0.99, 0, "merging")
            # Merge with ffmpeg
            try:
                _ffmpeg = ffmpeg_path or 'ffmpeg'
                subprocess.run([_ffmpeg, '-i', str(v_path), '-i', str(a_path),
                               '-c:v', 'copy', '-c:a', 'aac', '-strict', 'experimental',
                               '-shortest', str(m_path)],
                              capture_output=True, timeout=300)
                v_path.unlink(missing_ok=True)
                a_path.unlink(missing_ok=True)
            except Exception:
                m_path = v_path  # fallback: video-only
        else:
            v_path.rename(m_path)

        if progress_callback:
            progress_callback(1.0, 0, "done")
        return True
    except Exception as e:
        print(f"[ERROR] 下载失败 {bvid}: {e}")
        return False


def _cleanup_temp(v_path, a_path):
    """Remove partial download files on abort."""
    try:
        if v_path.exists():
            v_path.unlink()
    except Exception:
        pass
    try:
        if a_path.exists():
            a_path.unlink()
    except Exception:
        pass
