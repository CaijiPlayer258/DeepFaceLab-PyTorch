"""python -m MaskProcessor — launch the FastAPI server with auto-open browser."""
import threading
import time
import webbrowser

import uvicorn


def _open_browser(host: str, port: int, delay: float = 1.5):
    """Open the WebUI in the default browser after the server starts."""
    # 0.0.0.0 means bind all interfaces; browser needs a concrete address
    if host == "0.0.0.0":
        host = "127.0.0.1"
    url = f"http://{host}:{port}"

    time.sleep(delay)
    try:
        webbrowser.open(url)
    except Exception as e:
        print(f"[auto-open] Failed to open browser: {e}")


if __name__ == "__main__":
    host = "127.0.0.1"
    port = 8000

    threading.Thread(target=_open_browser, args=(host, port), daemon=True).start()
    uvicorn.run("MaskProcessor.api.server:app", host=host, port=port)
