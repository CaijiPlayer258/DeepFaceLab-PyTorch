import json, urllib.request, urllib.error
BASE="http://127.0.0.1:8760"
def get(p):
    try:
        with urllib.request.urlopen(BASE+p, timeout=30) as r:
            return r.status, r.read()[:400].decode("utf-8","replace")
    except urllib.error.HTTPError as e:
        return e.code, (e.read()[:400].decode("utf-8","replace") or "<空>")
    except Exception as e:
        return "ERR", str(e)
for p in ["/api/merge/preview/status", "/api/merge/videos", "/openapi.json", "/api/ping", "/docs", "/"]:
    print(p, "->", get(p))
