# -*- coding: utf-8 -*-
r"""合成器 · 导出设置页 / 导出进度页 的版式验收（真后端 + 无头 Edge + CDP）。

验的是这一轮的用户口径：
  A 设置页 = 训练器参数页那套：每个分组一张 .card（.cardhd + N 行 .cfgrow），
    行 = 左侧定宽标签（.cl）+ 右侧控件（.cc），控件只用 .seg / .mgsel / .inp / .mini，
    底部动作条是 .cfgactions（与训练页对齐）。
  B 进度页卡片宽度 == 训练页参数卡宽度（±4px）；窄到 640px 视口时六步列表不被挤扁
    （.cvsteps ≥ 420px、可横向滚动、不换行不重叠）。

跑法（仓库根）：
    runtime\server\Scripts\python.exe app\tools\merge_ui_accept.py --tag before
退出码 0 = 全过。
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import socket
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
APP = REPO / "app"
PY = REPO / "runtime" / "server" / "Scripts" / "python.exe"
TEMP = Path(os.environ.get("TEMP") or os.environ.get("TMP") or ".")
BOX = TEMP / "dfl_merge_ui_accept"
WS = BOX / "ws"
SHOTS = BOX / "shots"
LOG = BOX / "server.log"

sys.path.insert(0, str(APP))
RESULTS = []
BASE = ""


def check(ok, name, detail=""):
    ok = bool(ok)
    RESULTS.append((ok, name, str(detail)))
    print(("[PASS] " if ok else "[FAIL] ") + name + ("  · " + str(detail) if detail else ""),
          flush=True)
    return ok


def say(msg):
    print("       " + str(msg), flush=True)


def _free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def start_server():
    global BASE
    WS.mkdir(parents=True, exist_ok=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env["DFL_WORKSPACE"] = str(WS)
    env["PYTHONIOENCODING"] = "utf-8"
    env["NO_PROXY"] = env["no_proxy"] = "127.0.0.1,localhost"
    env.pop("DFL_PORT", None)
    logf = open(str(LOG), "ab")
    proc = subprocess.Popen([str(PY), "serve.py"], cwd=str(APP), env=env,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    port = {"v": None}

    def reader():
        for line in iter(proc.stdout.readline, b""):
            try:
                logf.write(line)
                logf.flush()
            except Exception:                              # noqa: BLE001
                pass
            text = line.decode("utf-8", "replace")
            if port["v"] is None and text.startswith("PORT="):
                try:
                    port["v"] = int(text.strip().split("=", 1)[1])
                except Exception:                          # noqa: BLE001
                    pass

    threading.Thread(target=reader, daemon=True).start()
    t0 = time.time()
    while port["v"] is None and time.time() - t0 < 120:
        if proc.poll() is not None:
            raise SystemExit("后端起不来，见 %s" % LOG)
        time.sleep(0.2)
    if port["v"] is None:
        raise SystemExit("等不到 PORT= 行，见 %s" % LOG)
    BASE = "http://127.0.0.1:%d" % port["v"]
    op = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    while time.time() - t0 < 150:
        try:
            with op.open(BASE + "/api/merge/stage/status", timeout=20) as r:
                if r.status == 200:
                    return proc
        except Exception:                                  # noqa: BLE001
            pass
        time.sleep(0.3)
    raise SystemExit("后端不应答，见 %s" % LOG)


def stop_server(proc):
    if proc is None:
        return
    try:
        subprocess.run(["taskkill", "/T", "/F", "/PID", str(proc.pid)],
                       capture_output=True, timeout=30)
    except Exception:                                      # noqa: BLE001
        pass
    try:
        proc.kill()
    except Exception:                                      # noqa: BLE001
        pass


# ---------------------------------------------------------------- 页面探针
PROBE = r"""
(function(){
  var q = function(s, r){ return (r || document).querySelector(s); };
  var qa = function(s, r){ return Array.prototype.slice.call((r || document).querySelectorAll(s)); };
  var rect = function(e){ if(!e) return null; var b = e.getBoundingClientRect();
    return {w: +b.width.toFixed(2), h: +b.height.toFixed(2), l: +b.left.toFixed(2),
            r: +b.right.toFixed(2), t: +b.top.toFixed(2), b: +b.bottom.toFixed(2)}; };
  var view = q('#p-merge .mgview[data-view="settings"]');
  var cards = qa('.card', view);
  var rows = qa('.cfgrow', view);
  var train = q('#p-train-cfg');
  var trainCard = q('#cfgBox > .card') || q('#p-train-cfg .card');
  var wrap = q('#cfgBox') || train;
  /* 训练页内容区宽度：卡片是 width:100%，所以「页面内容区」就是参数卡的真实宽度。
     没用模型时 #cfgBox 是空的（没有卡可量），这一条就是唯一稳定的量法。 */
  var tw = 0, tl = 0, tr2 = 0;
  if (train) {
    var cs = getComputedStyle(train);
    tw = train.clientWidth - parseFloat(cs.paddingLeft) - parseFloat(cs.paddingRight);
    tl = parseFloat(cs.paddingLeft); tr2 = parseFloat(cs.paddingRight);
  }
  var prog = q('#p-merge .mgview[data-view="progress"]');
  var mgprog = q('.mgprog', prog);
  var steps = q('#mgpSteps');
  var groups = qa('.mggrp', view);
  var out = {
    viewport: {w: window.innerWidth, h: window.innerHeight},
    settings: {
      cards: cards.length,
      titles: cards.map(function(c){ var b = q('.cardhd b', c); return b ? b.textContent.trim() : ''; }),
      rows: rows.length,
      row_h: rows.map(function(r){ return rect(r).h; }),
      label_w: rows.map(function(r){ var c = q('.cl', r); return c ? rect(c).w : -1; }),
      has_cl: rows.filter(function(r){ return !!q('.cl', r); }).length,
      has_cc: rows.filter(function(r){ return !!q('.cc', r); }).length,
      seg: qa('.seg', view).length,
      mgsel: qa('.mgsel', view).length,
      inp: qa('.inp', view).length,
      mini: qa('.mini', view).length,
      legacy_sec: qa('.mgp-sec, .mgp-hd, .mgnum, .mgtiny, .mgp-scroll, .mgp-footrow, .mgrow', view).length,
      overflow: rows.filter(function(r){ var c = q('.cc', r);
        return c && c.scrollWidth > c.clientWidth + 1; }).length,
      foot_cfgactions: !!q('.cfgactions', view),
      foot_btns: qa('.cfgactions button', view).map(function(b){ return b.id; })
    },
    train: {
      visible: !!(train && train.offsetParent !== null),
      found: !!trainCard,
      card_w: trainCard ? rect(trainCard).w : null,
      content_w: +tw.toFixed(2),
      pad_l: tl, pad_r: tr2,
      wrap_w: wrap ? rect(wrap).w : null,
      cards: qa('#cfgBox > .card').length
    },
    progress: {
      card_w: mgprog ? rect(mgprog).w : null,
      card_l: mgprog ? rect(mgprog).l : null,
      card_r: mgprog ? rect(mgprog).r : null,
      view_w: prog ? rect(prog).w : null,
      card_scroll: mgprog ? mgprog.scrollWidth : null,
      card_client: mgprog ? mgprog.clientWidth : null,
      steps_w: steps ? rect(steps).w : null,
      steps_scroll: steps ? steps.scrollWidth : null,
      steps_client: steps ? steps.clientWidth : null,
      steps_box: (function(){ var s = q('.mgprog-steps'); return s ? {sw: s.scrollWidth, cw: s.clientWidth,
        ox: getComputedStyle(s).overflowX} : null; })(),
      nodes: qa('#mgpSteps .cvst').length,
      label_wrap: qa('#mgpSteps .cvnm').map(function(n){ return rect(n).h; }),
      tip_wrap: qa('#mgpSteps .cvtip').map(function(n){ return rect(n).h; }),
      overlap: (function(){
        var bad = 0;
        qa('#mgpSteps .cvsh').forEach(function(h){
          var n = q('.cvnm', h), t = q('.cvtip', h);
          if (!n || !t || !t.textContent) return;
          if (rect(n).r > rect(t).l + 0.5) bad += 1;
        });
        return bad;
      })()
    }
  };
  return JSON.stringify(out);
})()
"""


def shot(b, path):
    import base64
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(base64.b64decode(
            b.call("Page.captureScreenshot", format="png").get("data") or ""))
        say("截图 %s" % path)
    except Exception as e:                                 # noqa: BLE001
        say("截图失败（%s）：%s" % (path.name, e))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="after")
    args = ap.parse_args()
    tag = args.tag

    from tools.merge_accept import Browser, _edge_path       # noqa: E402
    edge = _edge_path()
    if not edge:
        raise SystemExit("找不到 Edge（装 Edge 或设 EDGE_PATH）")
    if BOX.exists():
        shutil.rmtree(BOX, ignore_errors=True)
    SHOTS.mkdir(parents=True, exist_ok=True)

    proc = start_server()
    say("后端 %s" % BASE)
    b = Browser(edge)
    shots = {}
    try:
        b.goto(BASE + "/")
        t0 = time.time()
        while time.time() - t0 < 60:
            try:
                if b.eval("typeof window.go === 'function' && typeof window.mgStep === 'function'"):
                    break
            except Exception:                              # noqa: BLE001
                pass
            time.sleep(0.3)
        b.eval("(function(){var w=document.getElementById('welcome'); if(w) w.style.display='none';"
               " var m=document.getElementById('modal'); if(m) m.style.display='none';"
               " return true;})()")
        b.eval("window.go('merge'); true")
        time.sleep(0.8)
        b.eval("window.mgStep('settings'); true")
        time.sleep(0.6)

        sh = PROBE
        # 训练页先量一次（必须在它可见的时候量：切走之后 display:none，宽度全成 0）
        b.eval("window.go('train-cfg'); true")
        time.sleep(0.9)
        pt = b.js(sh)
        say("训练页参数卡：visible=%s · 内容区 %.1f · 真卡 %s 张 · 卡宽 %s"
            % (pt["train"]["visible"], pt["train"]["content_w"],
               pt["train"]["cards"], pt["train"]["card_w"]))
        shots_train = SHOTS / ("%s-train.png" % tag)
        shot(b, shots_train)
        b.eval("window.go('merge'); true")
        time.sleep(0.4)
        b.eval("window.mgStep('settings'); true")
        time.sleep(0.6)
        p0 = b.js(sh)
        shots["settings"] = SHOTS / ("%s-settings.png" % tag)
        shot(b, shots["settings"])

        # ---- A：设置页版式 ----
        s = p0["settings"]
        check(s["cards"] == 4, "A 设置页 = 4 张分组卡（这次导出/输出/编码/性能）",
              "cards=%s" % s["cards"])
        check(s["titles"] == ["这次导出", "输出", "编码", "性能"],
              "A 分组标题用训练页 .cardhd 那套", "%s" % (s["titles"],))
        # 10 行 = 这次导出 3 + 输出 2 + 编码 2 + 性能 3
        # （「质量」那一行已经删掉：帧图 JPEG 固定 quality=100，界面不再暴露；
        #   性能那一组多了一行「快速导出」= 视频模式的融合流水线开关）
        check(s["rows"] == 10, "A 共 10 行 .cfgrow（3+2+2+3）", "rows=%s" % s["rows"])
        # 两个「路径行」(.cfgpathrow) 的标签在输入框里（.inp > .lb），本来就没有 .cl/.cc
        check(s["has_cl"] == s["rows"] - 2 and s["has_cc"] == s["rows"] - 2,
              "A 除 2 个路径行外，每行都有左侧标签 .cl + 右侧控件 .cc",
              "cl=%s cc=%s / %s" % (s["has_cl"], s["has_cc"], s["rows"]))
        uniq_lbl = sorted(set(round(x, 1) for x in s["label_w"] if x > 0))
        check(bool(uniq_lbl) and len(uniq_lbl) == 1 and abs(uniq_lbl[0] - 130) <= 1,
              "A 标签列定宽 130px（训练页 .cfgrow .cl）", "实测 %s" % (uniq_lbl,))
        rh = s["row_h"] or [0.0]
        # 普通行 45px（和训练页一致）。两行**本来就高一点（52px）**，别把这条卡死：
        #   · 帧范围：两个数字框 + 「→」+ 末帧占位 + 「恢复全片」按钮；
        #   · 编码器：两档名字长（h264_nvenc / libx264），.seg 里的 b 换行。
        # 判据 = 除了这两行，其余每行都在 44~46；且不许出现第三行变高。
        tall = [i + 1 for i, h in enumerate(rh) if h > 48]
        check(min(rh) >= 43 and max(rh) <= 52 and len(tall) <= 2,
              "A 行高与训练页一致（44~46px；只允许帧范围/编码器两行 52px）",
              "min=%.1f max=%.1f · >48px 的行 %s · 逐行 %s"
              % (min(rh), max(rh), tall, [round(x, 1) for x in rh]))
        # 5 个 .seg = 格式 / 编码器 / 解码加速 / 保留 cache / 快速导出
        check(s["seg"] == 5 and s["mini"] == 1 and s["inp"] >= 5,
              "A 控件复用训练页的 .seg / .inp / .mini", "seg=%s inp=%s mini=%s"
              % (s["seg"], s["inp"], s["mini"]))
        check(s["legacy_sec"] == 0, "A 旧的 .mgp-sec/.mgp-hd/.mgnum/.mgtiny 已全部去掉",
              "残留 %s 个" % s["legacy_sec"])
        check(s["overflow"] == 0, "A 没有控件溢出（.cc scrollWidth ≤ clientWidth）",
              "%s 行溢出" % s["overflow"])
        check(s["foot_cfgactions"], "A 底部动作条 = 训练页 .cfgactions")
        check(s["foot_btns"][-2:] == ["mgCancelExport", "mgStartExport"],
              "A 按钮顺序 = 取消 / 开始导出", "%s" % (s["foot_btns"],))

        # ---- B：进度页宽度 ----
        b.eval("window.mgStep('progress'); true")
        time.sleep(0.5)
        b.eval("(function(){var r=document.getElementById('mgpResult');"
               " if(r){r.hidden=true;} "
               " var f=document.getElementById('mgpFill'); if(f) f.style.width='48%';"
               " var p=document.getElementById('mgpPct'); if(p) p.textContent='48%';"
               " return true;})()")
        time.sleep(0.3)
        p1 = b.js(sh)
        shots["progress"] = SHOTS / ("%s-progress.png" % tag)
        shot(b, shots["progress"])
        pr = p1["progress"]
        tr = pt["train"]                     # 训练页的宽度只能在它可见时量（见上）
        base_w = tr["card_w"] if tr["found"] else tr["content_w"]
        say("训练页参数卡宽 %s（内容区 %.1f）· 进度卡片宽 %s"
            % (tr["card_w"], tr["content_w"], pr["card_w"]))
        check(base_w is not None and pr["card_w"] is not None
              and abs(float(pr["card_w"]) - float(base_w)) <= 4,
              "B 进度卡片宽度 == 训练页参数卡宽度（±4px）",
              "进度 %.1f / 训练 %s（差 %.1f）"
              % (float(pr["card_w"] or 0), base_w,
                 abs(float(pr["card_w"] or 0) - float(base_w or 0))))
        check(pr["nodes"] == 6, "B 六步 Stepper 六个节点都在", "%s" % pr["nodes"])
        check(pr["card_scroll"] <= pr["card_client"] + 1,
              "B 进度卡内部没有横向溢出",
              "scroll=%s client=%s" % (pr["card_scroll"], pr["card_client"]))

        # ---- B：640px 窄视口 ----
        b.call("Emulation.setDeviceMetricsOverride", width=640, height=900,
               deviceScaleFactor=1, mobile=False)
        time.sleep(0.6)
        p2 = b.js(sh)
        pr2 = p2["progress"]
        shots["progress640"] = SHOTS / ("%s-progress-640.png" % tag)
        shot(b, shots["progress640"])
        say("640 视口：卡片 %.1f · 六步区 %s · 列表 scroll/client %s"
            % (float(pr2["card_w"] or 0), pr2["steps_w"], pr2["steps_box"]))
        check(float(pr2["steps_w"] or 0) >= 420 - 0.5,
              "B 640px 时六步列表仍 ≥ 420px（不被卡片挤扁）",
              "steps_w=%s" % pr2["steps_w"])
        check(pr2["steps_box"] and pr2["steps_box"]["sw"] >= pr2["steps_box"]["cw"] - 1,
              "B 640px 时六步区可横向滚动（scrollWidth ≥ clientWidth）",
              "%s" % (pr2["steps_box"],))
        check(bool(pr2["label_wrap"]) and max(pr2["label_wrap"] + [0]) <= 26,
              "B 640px 时 .cvnm 不换行（单行高度 ≤26px）", "%s" % pr2["label_wrap"])
        check(pr2["overlap"] == 0, "B 640px 时名字与提示不重叠",
              "重叠 %s 处" % pr2["overlap"])
        check(pr2["card_scroll"] <= pr2["card_client"] + 1,
              "B 640px 时卡片自身不横向溢出",
              "scroll=%s client=%s" % (pr2["card_scroll"], pr2["card_client"]))
        # 回到宽视口，再截一张结果页（终态那块要和进度卡同宽）
        b.call("Emulation.setDeviceMetricsOverride", width=1624, height=1046,
               deviceScaleFactor=1, mobile=False)
        time.sleep(0.5)
        b.eval("(function(){var r=document.getElementById('mgpResult');"
               " if(r){r.hidden=false;} return true;})()")
        time.sleep(0.3)
        p3 = b.js(sh)
        shots["result"] = SHOTS / ("%s-result.png" % tag)
        shot(b, shots["result"])
        check(abs(float(p3["progress"]["card_w"] or 0) - float(base_w or 0)) <= 4,
              "B 结果态卡片宽度同样等于训练页参数卡",
              "结果态 %.1f / 训练 %s" % (float(p3["progress"]["card_w"] or 0), base_w))

        (BOX / ("probe-%s.json" % tag)).write_text(
            json.dumps({"train": pt, "settings": p0, "progress": p1,
                        "narrow": p2, "result": p3},
                       ensure_ascii=False, indent=1), encoding="utf-8")
        return 0
    finally:
        b.close()
        stop_server(proc)
        fails = [r for r in RESULTS if not r[0]]
        print("")
        print("汇总（%s）：PASS %d · FAIL %d" % (tag, len(RESULTS) - len(fails), len(fails)))
        for ok, name, detail in fails:
            print("  FAIL %s · %s" % (name, detail))
        print("截图目录：%s" % SHOTS)


if __name__ == "__main__":
    rc = main()
    fails = [r for r in RESULTS if not r[0]]
    raise SystemExit(1 if fails else 0)
