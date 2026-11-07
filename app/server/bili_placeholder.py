# -*- coding: utf-8 -*-
"""显式 --mock 才启用的占位数据（默认关闭）。

历史：占位列表原来是写在前端 app/web/js/app.js 里的 biliMockFetch()，
点「获取视频列表」时纯本地随机生成 60~160 条假视频（假 bvid、假标题），
一个网络请求都没有 —— 用户看到 136 条就是这个原因。

现在占位只留在这里，且必须满足 DFL_BILI_MOCK=1（由 serve.py 的 --mock 参数设置）
才会被路由用到；返回体里固定带 mock=True，前端会打上「占位数据」标记。
"""
from __future__ import annotations

import os

_TITLES = [
    '【占位】深夜书房的一次实验', '【占位】第一次尝试这种转场', '【占位】三分钟讲完一个故事',
    '【占位】棚拍花絮', '【占位】用一盏灯拍完整支片子', '【占位】海边随手拍',
    '【占位】重制版', '【占位】这期的调色我改了七遍', '【占位】街头采访第 12 期',
    '【占位】器材开箱', '【占位】一场雨戏怎么拍', '【占位】五分钟看完整个项目',
    '【占位】关于构图的三件事', '【占位】夜景实拍', '【占位】剪辑复盘',
    '【占位】合作投稿', '【占位】拍到凌晨三点', '【占位】年度总结',
]


def enabled() -> bool:
    return os.environ.get('DFL_BILI_MOCK', '').strip().lower() in ('1', 'true', 'yes', 'on')


def placeholder_page(mid: str, page: int = 1, ps: int = 30, total: int = 136) -> dict:
    """生成一页假投稿，字段和 fetch_up_page 保持一致，便于前端同一套渲染。"""
    page = max(1, int(page or 1))
    ps = min(50, max(1, int(ps or 30)))
    start = (page - 1) * ps
    end = min(total, start + ps)
    items = []
    for n in range(start, end):
        items.append({
            'bvid': 'BV1mock%06d' % (n + 1),
            'title': _TITLES[n % len(_TITLES)] + ' · P%d' % (n + 1),
            'length': '%02d:%02d' % (n % 5, (n * 7) % 60),
            'created': 0,
            'play': 0,
            'is_coop': n % 7 == 0,
            'aid': 0,
        })
    return {'items': items, 'page': page, 'ps': ps, 'total': total,
            'has_more': end < total, 'mock': True}
