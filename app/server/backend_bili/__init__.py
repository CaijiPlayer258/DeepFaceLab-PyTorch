# -*- coding: utf-8 -*-
"""B 站下载器：扫码登录、投稿列表、DASH 音视频下载、批量队列。

旧路径 → 新名字（旧文件全部留 PEP 562 惰性代理，现有 import 不断）：
  bili_http.py        http.py         公共 HTTP 层：证书信任、错误归一化、cookie 脱敏
  biliauth.py         auth.py         扫码登录 / 轮询 / 登录态 / cookie 持久化
  bili_video.py       video.py        WBI 签名、投稿列表、playurl、下载与 ffmpeg 合流
  bili_queue.py       queue.py        批量下载线程池 + 进度状态
  bili_follow.py      follow.py       静默关注
  bili_user_info.py   user.py         nav 取 UID / 昵称 / 头像
  bili_placeholder.py placeholder.py  DFL_BILI_MOCK=1 时才启用的占位数据
  routes_bili.py      routes.py       /api/bili/* 的 APIRouter
  routes_account.py   account.py      /api/auth* 的 APIRouter（读写 workspace/auth.json）

**不在 __init__ 里 import 子模块**：会把 requests / qrcode 拖进启动路径，
而 routes.py 一被 import 就建 APIRouter 并读 WORKSPACE，等于把挂载时机提前 ——
一律走 `from .子模块 import 名字`。
"""
