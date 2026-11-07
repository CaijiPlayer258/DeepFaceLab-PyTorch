"""B 站视频 API：列投稿、取播放地址、下载（从 4.0 的 core/biliauth/video_api.py 移植）。

和 4.0 的区别（2026-05 修）：
 · 任何接口失败都抛 bili_http.BiliError（401 未登录 / 412 风控 / 502 网络），
   不再 except Exception: return [] 把失败伪装成「这个 UP 没视频」。
 · 下载走 DASH：view 拿 cid → player/wbi/playurl 拿 dash 音视频分流地址 →
   分别拉取 → ffmpeg 合流；清晰度按 qn 选（120=4K/116=1080P60/80=1080P/64=720P）。
 · ffmpeg 找不到时明确报错，不生成任何占位文件。
"""
from __future__ import annotations

import hashlib
import os
import re
import shutil
import subprocess
import time
from functools import reduce
from pathlib import Path
from urllib.parse import quote, urlencode

from . import bili_http
from .bili_http import BiliError

_HEADERS = dict(bili_http.HEADERS)
_JSON_HEADERS = dict(_HEADERS, Accept='application/json, text/plain, */*')

VIEW_URL = 'https://api.bilibili.com/x/web-interface/view'
PLAYURL_URL = 'https://api.bilibili.com/x/player/wbi/playurl'
ARC_SEARCH_URL = 'https://api.bilibili.com/x/space/wbi/arc/search'
NAV_URL = 'https://api.bilibili.com/x/web-interface/nav'

# qn：B 站清晰度代码
QN_MAP = {'best': 120, '1080': 80, '720': 64, '480': 32, '360': 16}

# WBI signing
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


def _getMixinKey(orig: str):
    return reduce(lambda s, i: s + orig[i], _mixinKeyEncTab, '')[:32]


def _encWbi(params: dict, img_key: str, sub_key: str):
    mixin_key = _getMixinKey(img_key + sub_key)
    params['wts'] = round(time.time())
    params = dict(sorted(params.items()))
    params = {k: ''.join(filter(lambda c: c not in "!'()*", str(v))) for k, v in params.items()}
    query = '&'.join('%s=%s' % (k, quote(str(v))) for k, v in params.items())
    params['w_rid'] = hashlib.md5((query + mixin_key).encode()).hexdigest()
    return params


def _getWbiKeys(cookie_str: str = ''):
    """拿 WBI 签名用的 img_key / sub_key。内存 -> 磁盘 -> nav 接口三级缓存。"""
    now = time.time()
    if now < _wbi_cache['expires'] and _wbi_cache['img_key']:
        return (_wbi_cache['img_key'], _wbi_cache['sub_key'])
    _load_wbi_cache()
    if now < _wbi_cache['expires'] and _wbi_cache['img_key']:
        return (_wbi_cache['img_key'], _wbi_cache['sub_key'])

    h = {}
    if cookie_str:
        h['Cookie'] = cookie_str
    j = bili_http.get_json(NAV_URL, headers=h, timeout=10, what='nav（WBI 密钥）')
    data = j.get('data') or {}
    wbi = data.get('wbi_img') or {}
    if not wbi.get('img_url') or not wbi.get('sub_url'):
        raise BiliError('nav 没返回 WBI 密钥（通常是 cookie 失效）', kind='not_logged_in')
    _wbi_cache['img_key'] = wbi['img_url'].rsplit('/', 1)[1].split('.')[0]
    _wbi_cache['sub_key'] = wbi['sub_url'].rsplit('/', 1)[1].split('.')[0]
    _wbi_cache['expires'] = now + 600  # cache for 10 minutes
    _save_wbi_cache()
    return (_wbi_cache['img_key'], _wbi_cache['sub_key'])


# ============================================================
# 投稿列表
# ============================================================
def fetch_up_page(mid: str, cookie_str: str = '', page: int = 1, ps: int = 30) -> dict:
    """拉某一页投稿（真实接口 /x/space/wbi/arc/search，需要 WBI 签名）。

    返回 {items, page, ps, total, has_more}；任何失败抛 BiliError。
    """
    mid = str(mid or '').strip()
    if not re.fullmatch(r'\d+', mid):
        raise BiliError('mid 必须是纯数字 UID：%r' % mid, kind='bad_request')
    page = max(1, int(page or 1))
    ps = min(50, max(1, int(ps or 30)))

    h = dict(_JSON_HEADERS)
    if cookie_str:
        h['Cookie'] = cookie_str

    img_key, sub_key = _getWbiKeys(cookie_str)
    params = _encWbi({'mid': mid, 'ps': ps, 'pn': page}, img_key, sub_key)
    j = bili_http.get_json('%s?%s' % (ARC_SEARCH_URL, urlencode(params)),
                           headers=h, timeout=15,
                           what='投稿列表 /x/space/wbi/arc/search (mid=%s p%d)' % (mid, page))
    data = j.get('data') or {}
    pageinfo = data.get('page') or {}
    vlist = ((data.get('list') or {}).get('vlist') or [])

    items = []
    for v in vlist:
        title = (v.get('title') or '').strip()
        staff = v.get('staff')
        items.append({
            'bvid': v.get('bvid', ''),
            'title': title,
            'length': v.get('length', ''),      # "mm:ss" / "hh:mm:ss"
            'created': v.get('created', 0),
            'play': v.get('play', 0),
            'is_coop': ('合作' in title) or (isinstance(staff, list) and len(staff) > 0),
            'aid': v.get('aid', 0),
        })

    total = int(pageinfo.get('count') or 0)
    if page == 1 and total == 0 and not items:
        # 接口 code=0 但一条都没有：UID 不存在 / 没公开投稿。
        # 以前这种情况会返回空列表，前端显示「获取完成 0 个」，等于骗人。
        raise BiliError('UID %s 没有公开投稿（或该用户不存在）' % mid, kind='not_found')
    got = (page - 1) * ps + len(items)
    has_more = bool(items) and (got < total if total else len(items) >= ps)
    return {'items': items, 'page': page, 'ps': ps, 'total': total, 'has_more': has_more}


def fetch_up_videos(mid: str, cookie_str: str = '', max_pages: int = None,
                    progress_callback=None) -> list:
    """一次拉完某 UP 的全部投稿（逐页调用 fetch_up_page）。

    出错直接抛 BiliError —— 调用方不能再把失败当成「没有视频」。
    """
    videos = []
    page = 1
    while True:
        if max_pages and page > max_pages:
            break
        data = fetch_up_page(mid, cookie_str, page=page, ps=30)
        for v in data['items']:
            videos.append({'bvid': v['bvid'], 'title': v['title'],
                           'length': v['length'], 'is_coop': v['is_coop']})
        if progress_callback:
            progress_callback(len(videos), page)
        if not data['has_more']:
            break
        page += 1
        time.sleep(0.5)  # 限流：B 站对 arc/search 有频率限制
    return videos


def fetch_user_info_by_mid(mid: str, cookie_str: str = '') -> dict:
    """按 mid 取 UP 主名字/头像（WBI 签名 /x/space/wbi/acc/info，失败抛 BiliError）。"""
    h = dict(_JSON_HEADERS)
    if cookie_str:
        h['Cookie'] = cookie_str
    img_key, sub_key = _getWbiKeys(cookie_str)
    signed = _encWbi({'mid': str(mid)}, img_key, sub_key)
    j = bili_http.get_json('https://api.bilibili.com/x/space/wbi/acc/info?%s' % urlencode(signed),
                           headers=h, timeout=10, what='UP 主信息 /x/space/wbi/acc/info')
    d = j.get('data') or {}
    return {'name': d.get('name', ''), 'face': d.get('face', ''), 'mid': str(d.get('mid', mid))}


# ============================================================
# 下载
# ============================================================
def resolve_ffmpeg() -> str:
    """找 ffmpeg：imageio-ffmpeg -> DFL_FFMPEG -> PATH。找不到就抛 BiliError。"""
    try:
        import imageio_ffmpeg
        exe = imageio_ffmpeg.get_ffmpeg_exe()
        if exe and Path(exe).is_file():
            return exe
    except Exception:
        pass
    env = os.environ.get('DFL_FFMPEG', '').strip()
    if env and Path(env).is_file():
        return env
    which = shutil.which('ffmpeg')
    if which:
        return which
    raise BiliError('ffmpeg 不可用：装 imageio-ffmpeg、设置 DFL_FFMPEG，或把 ffmpeg 放进 PATH',
                    kind='ffmpeg')


def _stream_url(stream: dict) -> str:
    for k in ('baseUrl', 'base_url', 'url'):
        if stream.get(k):
            return stream[k]
    return ''


def _stream_urls(stream: dict) -> list:
    """一条流的候选地址：主地址 + B 站给的 backupUrl。

    mcdn / PCDN 节点经常 404（跨网、代理、节点下线），必须能换地址重试，
    否则就会把「这个节点挂了」错报成「下载失败」。
    """
    out = []
    for k in ('baseUrl', 'base_url', 'url'):
        if stream.get(k):
            out.append(stream[k])
    for k in ('backupUrl', 'backup_url'):
        for u in (stream.get(k) or []):
            if u:
                out.append(u)
    seen = set()
    uniq = []
    for u in out:
        if u not in seen:
            seen.add(u)
            uniq.append(u)
    return uniq


_CODEC_PREF = {'avc': 4, 'hev': 3, 'hvc': 3, 'av01': 2}


def _stream_score(s: dict):
    """画质优先，其次编码（avc 兼容性最好），再次带宽，最后躲开 mcdn 节点。"""
    codec = (s.get('codecs') or '').lower()
    pref = 0
    for c, p in _CODEC_PREF.items():
        if codec.startswith(c):
            pref = p
            break
    host = (_stream_url(s).split('/')[2] if _stream_url(s) else '')
    stable = 0 if 'mcdn.bilivideo' in host else 1
    return (int(s.get('id') or 0), pref, stable, int(s.get('bandwidth') or 0))


def _rank_streams(streams, qn: int) -> list:
    """返回按优先级排好序的候选流（同画质不同编码/节点都保留，失败可换）。"""
    cands = [s for s in (streams or []) if _stream_url(s)]
    if not cands:
        return []
    capped = [s for s in cands if int(s.get('id') or 0) <= qn] or cands
    return sorted(capped, key=_stream_score, reverse=True)


def _pick_stream(streams, qn: int) -> dict:
    ranked = _rank_streams(streams, qn)
    return ranked[0] if ranked else {}


class _Aborted(Exception):
    """progress_callback 返回 False：用户取消。"""


def _download_stream(urls, path: Path, cookie_str: str, on_bytes) -> int:
    """按候选地址依次尝试下载；on_bytes(got, total) 返回 False 表示取消。

    mcdn 节点 404/断流是常态，所以这里不是「一条地址失败就整体失败」，
    而是换下一个地址（含 backupUrl、同画质不同节点的流）继续。
    """
    last = None
    for url in urls:
        try:
            resp = bili_http.request('GET', url, headers={'Cookie': cookie_str},
                                     timeout=120, stream=True)
        except BiliError as e:
            last = e
            print('[bili] 地址失败（%s），换下一个' % e, flush=True)
            continue
        if resp.status_code not in (200, 206):
            last = BiliError('媒体流返回 HTTP %d（%s）'
                             % (resp.status_code, url.split('/')[2]),
                             kind='api', http_status=resp.status_code)
            resp.close()
            print('[bili] HTTP %d（%s），换下一个' % (resp.status_code, url.split('/')[2]),
                  flush=True)
            continue
        total = int(resp.headers.get('Content-Length') or 0)
        got = 0
        try:
            with open(str(path), 'wb') as f:
                for chunk in resp.iter_content(chunk_size=1024 * 1024):
                    if not chunk:
                        continue
                    f.write(chunk)
                    got += len(chunk)
                    if on_bytes and on_bytes(got, total) is False:
                        raise _Aborted()
            return got
        except _Aborted:
            raise
        except Exception as e:            # 断流 / 超时：换地址重试
            last = BiliError('下载中断：%s' % e, kind='network')
            print('[bili] 下载中断（%s），换下一个地址' % e, flush=True)
            continue
        finally:
            resp.close()
    raise last or BiliError('没有任何可用的媒体地址', kind='api')


def _merge_av(ffmpeg: str, v_path: Path, a_path: Path, out_path: Path) -> bool:
    """ffmpeg 合流；先 copy 音轨，不行再转 aac。成功要求退出码 0 且文件非空。"""
    attempts = (
        ['-c:v', 'copy', '-c:a', 'copy'],
        ['-c:v', 'copy', '-c:a', 'aac', '-strict', 'experimental'],
    )
    for extra in attempts:
        cmd = [ffmpeg, '-y', '-loglevel', 'error', '-i', str(v_path), '-i', str(a_path)]
        cmd += extra + ['-shortest', str(out_path)]
        try:
            r = subprocess.run(cmd, capture_output=True, timeout=600)
        except Exception as e:
            print('[bili] ffmpeg 调用失败: %s' % e, flush=True)
            continue
        if r.returncode == 0 and out_path.exists() and out_path.stat().st_size > 0:
            return True
        err = (r.stderr or b'').decode('utf-8', 'ignore').strip().splitlines()
        print('[bili] ffmpeg 合流失败（%s）: %s' % (' '.join(extra), err[-1] if err else ''),
              flush=True)
    return False


def _safe_name(name: str, limit: int = 80) -> str:
    name = re.sub(r'[\\/:*?"<>|\r\n\t]', '', name or '').strip()
    return name[:limit] or 'video'


def download_single_video(bvid: str, cookie_str: str, output_dir: str,
                          progress_callback=None, ffmpeg_path: str = None,
                          qn: int = 120) -> str:
    """按 BVID 真下一条视频，返回落盘的 mp4 路径。

    失败抛 BiliError；progress_callback(fraction, speed_mbps, stage) 返回 False 表示取消，
    此时抛 _Aborted（调用方按取消处理）。
    """
    bvid = str(bvid or '').strip()
    if not re.fullmatch(r'BV[0-9A-Za-z]{10}', bvid):
        raise BiliError('bvid 格式不对：%r' % bvid, kind='bad_request')

    ffmpeg = ffmpeg_path or resolve_ffmpeg()
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    h = {'Cookie': cookie_str} if cookie_str else {}

    info = bili_http.get_json('%s?bvid=%s' % (VIEW_URL, bvid), headers=h, timeout=15,
                              what='视频信息 /x/web-interface/view (%s)' % bvid)
    data = info.get('data') or {}
    title = _safe_name(data.get('title') or bvid)
    cid = str(data.get('cid') or ((data.get('pages') or [{}])[0].get('cid') or 0))
    if not cid or cid == '0':
        raise BiliError('视频信息里没有 cid（%s）' % bvid, kind='api')

    img_key, sub_key = _getWbiKeys(cookie_str)
    play_params = _encWbi({'bvid': bvid, 'cid': cid, 'qn': qn, 'fnval': 4048,
                           'fnver': 0, 'fourk': 1}, img_key, sub_key)
    play = bili_http.get_json('%s?%s' % (PLAYURL_URL, urlencode(play_params)),
                              headers=h, timeout=15,
                              what='播放地址 /x/player/wbi/playurl (%s)' % bvid)
    pdata = play.get('data') or {}

    video_urls = []
    audio_urls = []
    quality = 0
    if pdata.get('dash'):
        v_ranked = _rank_streams(pdata['dash'].get('video'), qn)
        a_ranked = _rank_streams(pdata['dash'].get('audio'), 9999)
        for s in v_ranked:
            video_urls.extend(_stream_urls(s))
        for s in a_ranked:
            audio_urls.extend(_stream_urls(s))
        quality = int(v_ranked[0].get('id') or 0) if v_ranked else 0
    elif pdata.get('durl'):
        video_urls = [d.get('url', '') for d in pdata['durl'] if d.get('url')]
        quality = int(pdata.get('quality') or 0)
    video_urls = [u for u in video_urls if u]
    audio_urls = [u for u in audio_urls if u]
    if not video_urls:
        raise BiliError('playurl 没给可用的视频流（%s）：可能需要大会员 / 版权 / 地区限制' % bvid,
                        kind='api')

    m_path = out / ('%s_%s.mp4' % (bvid, title))
    v_path = m_path.with_name(m_path.stem + '.video.m4s')
    a_path = m_path.with_name(m_path.stem + '.audio.m4s')
    print('[bili] 下载 %s cid=%s qn=%d 实际画质=%d dash=%s 音轨=%s 候选地址=%d/%d'
          % (bvid, cid, qn, quality, bool(pdata.get('dash')), bool(audio_urls),
             len(video_urls), len(audio_urls)), flush=True)

    total = 0
    for u in (video_urls[:1] + audio_urls[:1]):
        try:
            head = bili_http.request('HEAD', u, headers={'Cookie': cookie_str}, timeout=10)
            total += int(head.headers.get('Content-Length') or 0)
        except BiliError:
            pass

    state = {'done_before': 0, 'started': time.time()}

    def on_bytes(got, size):
        if not progress_callback:
            return True
        whole = total if total > 0 else max(size, 1)
        frac = min(0.98, (state['done_before'] + got) / max(whole, 1))
        speed = got / (1024 * 1024 * max(time.time() - state['started'], 0.01))
        stage = 'downloading_audio' if state['done_before'] else 'downloading_video'
        return progress_callback(frac, speed, stage)

    try:
        _download_stream(video_urls, v_path, cookie_str, on_bytes)
        if audio_urls:
            state['done_before'] = v_path.stat().st_size
            state['started'] = time.time()
            _download_stream(audio_urls, a_path, cookie_str, on_bytes)
    except (_Aborted, BiliError):
        _cleanup_temp(v_path, a_path)
        raise

    if not v_path.exists() or v_path.stat().st_size == 0:
        _cleanup_temp(v_path, a_path)
        raise BiliError('视频流下载后是空文件（%s）' % bvid, kind='api')

    if audio_urls and a_path.exists() and a_path.stat().st_size > 0:
        if progress_callback:
            progress_callback(0.99, 0, 'merging')
        if _merge_av(ffmpeg, v_path, a_path, m_path):
            v_path.unlink(missing_ok=True)
            a_path.unlink(missing_ok=True)
        else:
            # 合流失败：保留纯视频，明确告警，绝不假装成功
            if m_path.exists():
                m_path.unlink()
            v_path.replace(m_path)
            a_path.unlink(missing_ok=True)
            print('[bili] 警告：%s 音轨合并失败，已保留无声视频 %s' % (bvid, m_path), flush=True)
    else:
        v_path.replace(m_path)
        print('[bili] 警告：%s 没有下到音轨，输出为无声视频 %s' % (bvid, m_path), flush=True)

    if progress_callback:
        progress_callback(1.0, 0, 'done')
    return str(m_path)


def _cleanup_temp(*paths):
    """Remove partial download files on abort."""
    for p in paths:
        try:
            if p and Path(p).exists():
                Path(p).unlink()
        except Exception:
            pass
