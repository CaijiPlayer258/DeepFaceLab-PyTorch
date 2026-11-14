# -*- coding: utf-8 -*-
"""全站 i18n 审计：切英文后把每个 .page 里的中文揪出来。

做法
  1. 临时往 index.html 注入一段探针（用完一定还原）
  2. 无头 Edge 打开页面，探针把所有 .page 都显示出来、
     逐个文本节点 / title / placeholder / value 收集含中文的字符串，
     写进 <pre id="__report">
  3. --dump-dom 拿回 DOM，解析报告打印

用法
  python tools/i18n_audit.py            # 全部页面
  python tools/i18n_audit.py p-merge    # 只看某一页

注意
  · 报告里的"中文"可能出现在英文译文里（例如规范文件名），需要人工判断
  · 运行时才拼出来的文案，探针只能看到当前状态 —— 需要触发一遍渲染才准
"""
import json
import os
import re
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path


def _roots() -> tuple:
    """(仓库根, app 目录)。index.html 就在 <仓库根>/app/web/index.html。

    以前的写法是 parents[1] 再拼 "app/web/index.html"，算出来是
    <仓库>/app/app/web/index.html —— 路径根本不存在，脚本开头就崩。
    这里优先问 server/_paths.py 的 ROOT（全站唯一的根目录来源），
    拿不到再按仓库布局退一级（parents[2]），最后才用文件自己的父目录兜底。
    """
    here = Path(__file__).resolve()
    app_dir = here.parents[1]                       # app/tools/xxx.py → app/
    try:
        if str(app_dir) not in sys.path:
            sys.path.insert(0, str(app_dir))
        from server._paths import ROOT as _root        # noqa: WPS433 —— 运行时才 import
        root = Path(_root).resolve()
        if (root / "app" / "web" / "index.html").is_file():
            return root, root / "app"
    except Exception:
        pass
    for root in (here.parents[2], here.parents[1]):     # 标准布局：<root>/app/tools/
        if (root / "app" / "web" / "index.html").is_file():
            return root, root / "app"
    return here.parents[2], here.parents[2] / "app"


ROOT, APP = _roots()
EDGE = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"
PYTHON = sys.executable

# 英文译文里有意保留的中文片段：出现在译文里是对的，别当成漏译
ALLOW = ["UI设计规范.md"]

PROBE = '''  <script id="__probe">
  window.addEventListener("load", function () {
    setTimeout(function () {
      try { window.I18N.apply("en"); } catch (e) {
        var b = document.querySelector('[data-lang="en"]'); if (b) b.click();
      }
      setTimeout(function () {
        var CJK = /[\\u4e00-\\u9fa5]/;
        var out = [];
        var pages = document.querySelectorAll(".page");
        Array.prototype.forEach.call(pages, function (p) { p.classList.add("on"); });
        Array.prototype.forEach.call(pages, function (p) {
          var id = p.id || "(no-id)";
          var w = document.createTreeWalker(p, NodeFilter.SHOW_TEXT, null, false), n;
          while ((n = w.nextNode())) {
            var t = (n.nodeValue || "").trim().replace(/\\s+/g, " ");
            if (!t || !CJK.test(t)) continue;
            var tag = n.parentNode && n.parentNode.tagName ? n.parentNode.tagName : "?";
            if (tag === "SCRIPT" || tag === "STYLE") continue;
            out.push(id + "|" + tag + "|" + t.slice(0, 90));
          }
          Array.prototype.forEach.call(p.querySelectorAll("[title],[placeholder]"), function (el) {
            ["title", "placeholder"].forEach(function (a) {
              var v = el.getAttribute(a);
              if (v && CJK.test(v)) out.push(id + "|@" + a + "|" + v.slice(0, 90));
            });
          });
          Array.prototype.forEach.call(p.querySelectorAll("input[value]"), function (el) {
            if (el.value && CJK.test(el.value)) out.push(id + "|@value|" + el.value.slice(0, 90));
          });
        });
        var pre = document.createElement("pre");
        pre.id = "__report";
        pre.textContent = out.join("\\n");
        document.body.appendChild(pre);
      }, 800);
    }, 1200);
  });
  </script>
</body>'''


# 本机调试口必须直连：系统代理（这里常开着 Watt Toolkit 的 127.0.0.1:26561）
# 会被 urllib 从注册表读出来，连 127.0.0.1 的请求也发给它，于是永远连不上。
_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def _http_json(url: str, timeout: float = 5.0, method: str = "GET"):
    req = urllib.request.Request(url, method=method)
    with _OPENER.open(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8", "replace"))


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _new_target(port: int, url: str) -> str:
    """让无头 Edge 新开一个 tab（新版本只认 PUT，老版本是 GET）。"""
    for method in ("PUT", "GET"):
        try:
            d = _http_json("http://127.0.0.1:%d/json/new?%s" % (port, url),
                           timeout=5, method=method)
            if d.get("webSocketDebuggerUrl"):
                return str(d["webSocketDebuggerUrl"])
        except Exception:
            continue
    return ""


def _report_via_cdp(url: str, timeout: float = 90.0) -> str:
    """起无头 Edge，用 CDP 等页面自己把 <pre id="__report"> 填好，再把文本取回来。

    为什么不用 `--dump-dom --virtual-time-budget`：这一页有定时器和长轮询，
    虚拟时间会被挂住的请求永远冻住，dump 一直不返回（实测卡 6 分钟以上，
    只能手动杀进程）。CDP 走真实时间，页面怎么异步都不影响。

    **必须给专用 --user-data-dir**：不给的话无头实例会去抢用户正在用的
    默认 profile 的锁，同样会卡死。
    """
    try:
        from websockets.sync.client import connect          # runtime/server 自带
    except Exception as ex:
        print("没装 websockets（%s）—— 跳过浏览器那一半" % ex)
        return ""

    port = _free_port()
    # 每次一个全新的 profile：复用同一个目录会撞上上次被杀的实例留下的锁，
    # Edge 直接退出，调试口永远不出现（第一版就是这么失败的）。
    profile = Path(tempfile.gettempdir()) / ("dfl_i18n_profile_%d" % os.getpid())
    if profile.is_dir():
        import shutil
        shutil.rmtree(profile, ignore_errors=True)
    proc = subprocess.Popen(
        [EDGE, "--headless=new", "--disable-gpu", "--window-size=1624,1200",
         "--no-first-run", "--no-default-browser-check", "--disable-extensions",
         "--remote-allow-origins=*",
         "--user-data-dir=%s" % profile, "--remote-debugging-port=%d" % port,
         url],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        creationflags=0x08000000,
    )
    try:
        ws_url = ""
        deadline = time.time() + 40
        while time.time() < deadline and not ws_url:
            try:
                tabs = _http_json("http://127.0.0.1:%d/json/list" % port, timeout=2)
                pages = [t for t in tabs if t.get("type") == "page" and t.get("webSocketDebuggerUrl")]
                # 优先选已经打开目标页的那个 target（Edge 起手可能还塞一个 edge:// 引导页）
                for t in pages:
                    if str(t.get("url") or "").startswith(url.rstrip("/")):
                        ws_url = str(t["webSocketDebuggerUrl"])
                        break
                if not ws_url and pages:
                    # 起手那个 target 可能是 edge:// 引导页：另开一个，别在上面导航
                    ws_url = _new_target(port, url) or str(pages[0]["webSocketDebuggerUrl"])
            except Exception:
                time.sleep(0.3)
        if not ws_url:
            print("连不上无头 Edge 的调试口（%d）" % port)
            return ""

        seq = [0]

        with connect(ws_url, max_size=None, open_timeout=20) as ws:
            def call(method: str, params: dict | None = None) -> dict:
                seq[0] += 1
                ws.send(json.dumps({"id": seq[0], "method": method, "params": params or {}}))
                while True:
                    msg = json.loads(ws.recv(timeout=30))
                    if msg.get("id") == seq[0]:
                        return msg.get("result") or {}

            call("Page.enable")
            call("Page.navigate", {"url": url})
            text = ""
            expr = ("(function(){var p=document.getElementById('__report');"
                    "return p?String(p.textContent):'';})()")
            deadline = time.time() + timeout
            while time.time() < deadline:
                time.sleep(0.4)
                try:
                    res = call("Runtime.evaluate",
                               {"expression": expr, "returnByValue": True})
                except Exception:
                    continue
                text = str((res.get("result") or {}).get("value") or "")
                if text.strip():
                    break
            try:
                call("Browser.close")
            except Exception:
                pass
            return text
    finally:
        try:
            proc.terminate()
            proc.wait(timeout=5)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass


def main() -> int:
    only = sys.argv[1] if len(sys.argv) > 1 else None
    index = APP / "web" / "index.html"
    print("仓库根：%s" % ROOT)
    print("页面：  %s" % index)
    if not index.is_file():
        print("找不到 index.html —— 路径解析错了（见 _roots()）")
        return 1
    if not Path(EDGE).is_file():
        print("找不到无头 Edge：%s —— 浏览器那一半跑不了（后端仍可单独验证）" % EDGE)
        return 1
    backup = index.read_text(encoding="utf-8")
    if "__probe" in backup:
        print("index.html 里已经有探针，先手动清掉再跑")
        return 1
    if backup.count("</body>") != 1:
        print("index.html 里 </body> 不止一处，探针位置不确定")
        return 1
    index.write_text(backup.replace("</body>", PROBE), encoding="utf-8")

    env = dict(os.environ)
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"      # 子进程按 utf-8 说话，下面也按 utf-8 收
    proc = subprocess.Popen(
        [PYTHON, "serve.py"], cwd=str(APP),
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, encoding="utf-8", errors="replace", bufsize=1,
        env=env, creationflags=0x08000000,
    )
    try:
        port = None
        for line in proc.stdout:
            if line.startswith("PORT="):
                port = int(line.strip().split("=")[1])
                break
        if port is None:
            print("后端没起来")
            return 1
        text = _report_via_cdp("http://127.0.0.1:%d/" % port)
        if not text.strip():
            print("没拿到报告 —— 页面可能报错了（读到的是空报告）")
            return 1
        # 后面按 <pre id="__report"> 解析，这里把文本包成同样的形状，下游一行不用改
        dom = ('<pre id="__report">%s</pre>'
               % text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))
    finally:
        index.write_text(backup, encoding="utf-8")
        proc.terminate()
        time.sleep(0.6)

    m = re.search(r'<pre id="__report">([\s\S]*?)</pre>', dom)
    if not m:
        print("没拿到报告 —— 页面可能报错了")
        return 1
    text = (m.group(1).replace("&lt;", "<").replace("&gt;", ">")
            .replace("&amp;", "&").replace("&quot;", '"'))
    rows = [ln for ln in text.split("\n") if ln.strip()]
    # 英文译文里有意保留的中文（例如规范文件名），不算漏译
    rows = [ln for ln in rows if not any(k in ln for k in ALLOW)]
    if only:
        rows = [ln for ln in rows if ln.startswith(only + "|")]
    by_page = {}
    for ln in rows:
        page, tag, rest = (ln.split("|", 2) + ["", ""])[:3]
        by_page.setdefault(page, []).append("%s: %s" % (tag, rest))

    # 控制台在中文 Windows 上会乱码，另外写一份 UTF-8 的全量清单
    try:
        dump = Path(tempfile.gettempdir()) / "dfl_i18n_audit.txt"
        dump.write_text("\n".join(rows), encoding="utf-8")
        print("清单已写到 " + str(dump))
    except OSError:
        pass

    print("未翻译条目：%d" % len(rows))
    for page, items in by_page.items():
        print("\n=== %s (%d) ===" % (page, len(items)))
        for it in items:
            print("  " + it)
    return 0 if not rows else 2


if __name__ == "__main__":
    raise SystemExit(main())
