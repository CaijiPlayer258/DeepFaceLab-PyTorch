# -*- coding: utf-8 -*-
"""本地 mock：模拟 B 站 passport 接口，供离线联调用。"""
import time
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

app = FastAPI()
STATE = {"polls": 0, "key": ""}

@app.get("/x/passport-login/web/qrcode/generate")
def gen():
    STATE["key"] = "MOCKKEY123"
    STATE["polls"] = 0
    return {"code": 0, "message": "0", "data": {
        "url": "https://www.bilibili.com/h5/mall/...mock",
        "qrcode_key": STATE["key"]}}

@app.get("/x/passport-login/web/qrcode/poll")
def poll(qrcode_key: str = ""):
    STATE["polls"] += 1
    n = STATE["polls"]
    if n <= 2:
        code = 86101          # 未扫码
    elif n <= 4:
        code = 86090          # 已扫未确认
    else:
        code = 0              # 确认成功
    resp = JSONResponse({"code": 0, "message": "0", "data": {"code": code, "message": ""}})
    if code == 0:
        resp.set_cookie("SESSDATA", "mock_sessdata_value", domain="127.0.0.1")
        resp.set_cookie("bili_jct", "mock_jct", domain="127.0.0.1")
        resp.set_cookie("DedeUserID", "258000111", domain="127.0.0.1")
    return resp

@app.get("/x/web-interface/nav")
def nav(request: Request):
    cookie = request.headers.get("cookie", "")
    if "SESSDATA=" not in cookie:
        return {"code": -101, "message": "账号未登录", "data": {}}
    return {"code": 0, "message": "0", "data": {
        "mid": 258000111, "uname": "菜鸡播放器258",
        "face": "https://example.invalid/avatar.jpg",
        "level_info": {"current_level": 5}}}
