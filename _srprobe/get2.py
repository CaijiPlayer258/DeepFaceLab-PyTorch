import json, urllib.request
op = urllib.request.build_opener(urllib.request.ProxyHandler({}))
print("proxies:", urllib.request.getproxies())
try:
    with op.open("http://127.0.0.1:8760/api/merge/videos", timeout=20) as r:
        print("no-proxy opener ->", r.status, r.read()[:160].decode("utf-8","replace"))
except Exception as e:
    print("no-proxy 失败:", type(e).__name__, e)
