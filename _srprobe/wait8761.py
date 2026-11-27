import json, time, urllib.request, urllib.error
OP = urllib.request.build_opener(urllib.request.ProxyHandler({}))
for i in range(60):
    try:
        with OP.open("http://127.0.0.1:8761/api/merge/preview/status", timeout=5) as r:
            print("UP after %.1fs -> %s" % (i*2, r.status)); break
    except Exception as e:
        print("  ...waiting (%02d) %s" % (i, type(e).__name__), flush=True)
        time.sleep(2)
