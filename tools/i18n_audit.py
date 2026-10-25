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
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "app"
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


def main() -> int:
    only = sys.argv[1] if len(sys.argv) > 1 else None
    index = APP / "web" / "index.html"
    backup = index.read_text(encoding="utf-8")
    if "__probe" in backup:
        print("index.html 里已经有探针，先手动清掉再跑")
        return 1
    if backup.count("</body>") != 1:
        print("index.html 里 </body> 不止一处，探针位置不确定")
        return 1
    index.write_text(backup.replace("</body>", PROBE), encoding="utf-8")

    proc = subprocess.Popen(
        [PYTHON, "serve.py"], cwd=str(APP),
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
        env=dict(os.environ), creationflags=0x08000000,
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
        # 不用 TemporaryDirectory：Edge 退出后文件句柄还挂着，rmtree 会炸
        shot = Path(tempfile.gettempdir()) / "dfl_i18n_dom.html"
        with open(shot, "wb") as fh:
            subprocess.run(
                [EDGE, "--headless=new", "--disable-gpu", "--window-size=1624,1200",
                 "--virtual-time-budget=9000", "--dump-dom",
                 "http://127.0.0.1:%d/" % port],
                timeout=240, stdout=fh, stderr=subprocess.DEVNULL,
            )
        dom = shot.read_text(encoding="utf-8", errors="replace")
        try:
            shot.unlink()
        except OSError:
            pass
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

    print("未翻译条目：%d" % len(rows))
    for page, items in by_page.items():
        print("\n=== %s (%d) ===" % (page, len(items)))
        for it in items:
            print("  " + it)
    return 0 if not rows else 2


if __name__ == "__main__":
    raise SystemExit(main())
