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
  var steps = q('#mgPipeSvg');        /* 进度页那张图（6 列 lane + 曲线） */
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
      /* 两个性能下拉（推理精度 / 合成后端）：id、data-cfg 记忆键、当前值、显示文字、尺寸。
         尺寸取**计算值**（getComputedStyle）而不是 getBoundingClientRect：后者在有滚动
         容器 / 无头视口下会给出 0×0（元素明明画出来了），断言会假红。计算值能直接证明
         「#p-merge .mgsel{width:100%} 被本页那条 152×30 的规则压住了」—— 那才是要证的。 */
      sels: qa('select.mgsel[data-cfg]', view).map(function(s){
        var o = s.options[s.selectedIndex];
        var cs = getComputedStyle(s);
        return {id: s.id, cfg: s.getAttribute('data-cfg'), v: String(s.value),
                txt: o ? String(o.textContent).trim() : '',
                cw: parseFloat(cs.width) || 0, ch: parseFloat(cs.height) || 0,
                ow: s.offsetWidth, oh: s.offsetHeight,
                w: rect(s) ? rect(s).w : -1, h: rect(s) ? rect(s).h : -1,
                disp: String(cs.display),
                ov: s.scrollWidth > s.clientWidth + 1};
      }),
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
      /* 流水线实况：6 列标题 + N 个 lane 方块（N 由后端 lanes 决定，这里只看形状） */
      pipe_cols: qa('#mgPipeSvg text.t-stage').length,
      pipe_lanes: qa('#mgPipeSvg rect.lane').length,
      label_wrap: qa('#mgPipeSvg text.t-stage').map(function(n){ return rect(n).h; }),
      tip_wrap: qa('#mgPipeSvg text.t-lane').map(function(n){ return rect(n).h; }),
      overlap: (function(){
        var bad = 0, ts = qa('#mgPipeSvg text.t-stage');
        for (var i = 1; i < ts.length; i++) {
          if (rect(ts[i - 1]).r > rect(ts[i]).l + 0.5) bad += 1;
        }
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

    # tools/ 按用途分了子目录（accept/bench/migrate/audit），这个 import 当年没跟着改，
    # 一直指向 app/tools/merge_accept.py（已经不在了）→ 脚本一开场就 ModuleNotFoundError。
    from tools.accept.merge_accept import Browser, _edge_path       # noqa: E402
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
        # 12 行 = 这次导出 3 + 输出 2 + 编码 2 + 性能 5
        # （「质量」那一行已经删掉：帧图 JPEG 固定 quality=100，界面不再暴露；
        #   性能那一组是 并行进程 / 保留 cache / 快速导出 + 这轮新增的两个下拉
        #   推理精度 / 合成后端 —— 37 个 DFL_* 里只有这两个值得给用户选，见 docs/27）
        check(s["rows"] == 12, "A 共 12 行 .cfgrow（3+2+2+5）", "rows=%s" % s["rows"])
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
        # 2 个 .mgsel = 推理精度 / 合成后端（>4 档用 select、≤4 档用 seg 的那条口径）
        check(s["seg"] == 5 and s["mini"] == 1 and s["inp"] >= 5 and s["mgsel"] == 2,
              "A 控件复用训练页的 .seg / .inp / .mini / .mgsel",
              "seg=%s inp=%s mini=%s mgsel=%s"
              % (s["seg"], s["inp"], s["mini"], s["mgsel"]))
        check(s["legacy_sec"] == 0, "A 旧的 .mgp-sec/.mgp-hd/.mgnum/.mgtiny 已全部去掉",
              "残留 %s 个" % s["legacy_sec"])
        check(s["overflow"] == 0, "A 没有控件溢出（.cc scrollWidth ≤ clientWidth）",
              "%s 行溢出" % s["overflow"])
        check(s["foot_cfgactions"], "A 底部动作条 = 训练页 .cfgactions")
        check(s["foot_btns"][-2:] == ["mgCancelExport", "mgStartExport"],
              "A 按钮顺序 = 取消 / 开始导出", "%s" % (s["foot_btns"],))

        # ---- C：两个性能下拉（推理精度 / 合成后端）----
        sels = p0["settings"]["sels"]
        check([x["id"] for x in sels] == ["mgDfmDtype", "mgCmpBackend"],
              "C1 性能区两颗下拉都在（#mgDfmDtype / #mgCmpBackend）",
              str([x["id"] for x in sels]))
        check([x["cfg"] for x in sels] == ["dfm_dtype", "compose_backend"],
              "C2 都带 data-cfg（与训练器参数页同一套记忆键）",
              str([x["cfg"] for x in sels]))
        check([x["v"] for x in sels] == ["auto", "auto"]
              and all("自动" in x["txt"] for x in sels),
              "C3 默认值 = auto，显示文字是「自动」（不是 auto 原文）",
              str([(x["v"], x["txt"]) for x in sels]))
        check(all(abs(float(x["cw"]) - 152) <= 2 and abs(float(x["ch"]) - 30) <= 2
                  and not x["ov"] for x in sels),
              "C4 尺寸与训练页下拉一致（152×30 · 不溢出）",
              str([(x["id"], x["cw"], x["ch"], x["ow"], x["oh"], x["disp"], x["ov"])
                   for x in sels]))
        # 档位齐全：auto/fp32/bf16/bf16w 与 auto/cpu/gpu
        opts = json.loads(b.eval(
            "(function(){var g=function(id){var s=document.getElementById(id);"
            "if(!s) return [];return Array.prototype.map.call(s.options,function(o){"
            "return String(o.value);});};"
            "return JSON.stringify({d:g('mgDfmDtype'),c:g('mgCmpBackend')});})()") or "{}")
        check(opts.get("d") == ["auto", "fp32", "bf16", "bf16w"]
              and opts.get("c") == ["auto", "cpu", "gpu"],
              "C5 选项值就是后端认的那几个（auto 不写成具体档）", str(opts))

        # ---- D：请求体（默认不带键 / 选了才带）+ data-cfg 记忆（刷新后保持）----
        js_settings = ("(function(){var s=window.MergePages.settings();"
                       "return JSON.stringify({dfm_dtype:s.dfm_dtype,"
                       "compose_backend:s.compose_backend});})()")
        set_sel = ("(function(id,v){var e=document.getElementById(id);e.value=v;"
                   "e.dispatchEvent(new Event('change',{bubbles:true}));return e.value;})")
        # 默认态：settings 报 auto，mgPerfBody 不往请求体里塞这两个键
        d0 = json.loads(b.eval(
            "(function(){var s=window.MergePages.settings();"
            "var body=window.mgPerfBody({video:'clip.mp4',fused:1},s);"
            "return JSON.stringify({s:s.dfm_dtype+'/'+s.compose_backend,body:body});})()") or "{}")
        check(d0.get("s") == "auto/auto", "D1 默认态 settings 报 auto/auto", str(d0.get("s")))
        check("dfm_dtype" not in (d0.get("body") or {})
              and "compose_gpu" not in (d0.get("body") or {}),
              "D2 默认（自动）时请求体**不带**这两个键",
              json.dumps(d0.get("body") or {}, ensure_ascii=False)[:160])
        # 选 fp32 / gpu：请求体带 dfm_dtype=fp32、compose_gpu=1
        b.eval(set_sel + "('mgDfmDtype','fp32'); true")
        b.eval(set_sel + "('mgCmpBackend','gpu'); true")
        time.sleep(0.2)
        d1 = json.loads(b.eval(
            "(function(){var s=window.MergePages.settings();"
            "var b1=window.mgPerfBody({video:'clip.mp4',fused:1},s);"
            "var b2=window.mgPerfBody({video:'clip.mp4'},{dfm_dtype:'fp32',compose_backend:'cpu'});"
            "return JSON.stringify({body:b1,cpu:b2});})()") or "{}")
        body1 = d1.get("body") or {}
        check(body1.get("dfm_dtype") == "fp32" and body1.get("compose_gpu") == 1,
              "D3 选 fp32 + gpu → dfm_dtype=fp32 / compose_gpu=1",
              json.dumps(body1, ensure_ascii=False)[:200])
        check((d1.get("cpu") or {}).get("compose_gpu") == 0,
              "D4 选 cpu → compose_gpu=0（「关」也要明确说出来）",
              json.dumps(d1.get("cpu") or {}, ensure_ascii=False)[:160])
        mem = json.loads(b.eval(
            "(function(){var m=window.__mgCfgMem?window.__mgCfgMem():null;"
            "return JSON.stringify(m||{});})()") or "{}")
        check(mem.get("dfm_dtype") == "fp32" and mem.get("compose_backend") == "gpu",
              "D5 改完写进记忆（localStorage · data-cfg 键）", str(mem))
        # 刷新页面：两个下拉必须是 fp32 / gpu（记忆回填）
        # 用 Page.navigate 而不是 location.reload()：reload 之后 CDP 的执行上下文会被销毁，
        # 紧接着的 Runtime.evaluate 有概率落到旧上下文上报个莫名其妙的异常。
        def reload_merge():
            """刷新一页并回到「导出设置」：记忆回填（mgCfgMemApply）+ 合成后端门控都跑完再读。"""
            b.goto(BASE + "/")
            t0 = time.time()
            while time.time() - t0 < 60:
                try:
                    if b.eval("typeof window.go === 'function'"):
                        break
                except Exception:                          # noqa: BLE001
                    pass
                time.sleep(0.3)
            time.sleep(0.5)
            b.eval("(function(){var w=document.getElementById('welcome');"
                   " if(w) w.style.display='none';"
                   " var m=document.getElementById('modal'); if(m) m.style.display='none';"
                   " return true;})()")
            b.eval("window.go('merge'); true")
            time.sleep(0.5)
            b.eval("window.mgStep('settings'); true")
            time.sleep(0.5)

        # 合成后端那颗下拉 + 融合方式的现状（gpu 档禁没禁用、显示的是哪一档）+ 记忆快照
        CMP_STATE = ("(function(){var s=document.getElementById('mgCmpBackend');var o=null;"
                     "for(var i=0;i<s.options.length;i++){"
                     "if(s.options[i].value==='gpu'){o=s.options[i];break;}}"
                     "var d=document.getElementById('mgDfmDtype');"
                     "var m=(window.__mgCfgMem?window.__mgCfgMem():null)||{};"
                     "var p=(window.__mgCmpPref?window.__mgCmpPref():'');"
                     "return JSON.stringify({mode:String((document.getElementById('mgMode')||{}).value||''),"
                     "v:String(s.value),dfm:String(d&&d.value),gpu_disabled:!!(o&&o.disabled),"
                     "mem_cb:String(m.compose_backend||''),mem_dt:String(m.dfm_dtype||''),pref:String(p)});})()")

        reload_merge()                                     # 第 1 次刷新
        back = b.js(sh)["settings"]["sels"]
        v6 = [x["v"] for x in back]
        mem6 = json.loads(b.eval(CMP_STATE) or "{}")
        # D6 的原话就是「刷新后还是 fp32 / gpu（**data-cfg 记忆**，不是回默认）」——
        # 这句的主语是记忆：用户存过的偏好必须还在。seamless 下界面暂时显示「自动」
        # 是另一件事（那是这一单确实用不上 GPU 合成，见 D7）：两者的分工写在这里，
        # 别再让门控顺手把记忆改成 auto —— 那才是这轮修的那个 bug。
        check(mem6.get("mem_dt") == "fp32" and mem6.get("mem_cb") == "gpu",
              "D6 刷新后还是 fp32 / gpu（data-cfg 记忆，不是回默认）",
              "记忆 dfm=%s/cmp=%s · 界面此刻 %s · 偏好 %s"
              % (mem6.get("mem_dt"), mem6.get("mem_cb"), v6, mem6.get("pref")))
        shot(b, SHOTS / ("%s-settings-picked.png" % tag))

        # ---- D7：seamless 下「这一单用不上」≠「记忆被改了」----
        # 两次刷新之后：gpu 仍可选、界面/记忆都是 gpu（seamless 已开放走 GPU）；改 overlay 不变，
        # 这条同时证明「门控不会覆盖用户偏好」。
        s7a = json.loads(b.eval(CMP_STATE) or "{}")
        reload_merge()                                     # 第 2 次刷新
        s7b = json.loads(b.eval(CMP_STATE) or "{}")
        # 改回 overlay：走用户那条路（#mgMode 的 change → mgSyncBlendSeg + mgGateCmpBackend）
        b.eval("(function(){var e=document.getElementById('mgMode');e.value='overlay';"
               "e.dispatchEvent(new Event('change',{bubbles:true}));return e.value;})()")
        time.sleep(0.3)
        s7c = json.loads(b.eval(CMP_STATE) or "{}")
        # seamless 现在**也走 GPU**（DFL_COMPOSE_GPU_SEAMLESS 默认 1），所以不再「显示 auto + 禁用 gpu」。
        # 这条断言改成：两次刷新后 gpu 仍可选、界面仍是 gpu、记忆仍是 gpu，改 overlay 也不变。
        check(s7a.get("v") == "gpu" and s7a.get("gpu_disabled") is False
              and s7b.get("v") == "gpu" and s7b.get("gpu_disabled") is False
              and s7b.get("mem_cb") == "gpu" and s7b.get("mem_dt") == "fp32"
              and s7c.get("v") == "gpu" and s7c.get("dfm") == "fp32",
              "D7 seamless 下刷两次：gpu 仍可选、界面/记忆都还是 gpu（seamless 已开放走 GPU）",
              "刷1 %s/%s · 刷2 %s/%s · 记忆 %s/%s · 改 overlay %s/%s(gpu禁用=%s)"
              % (s7a.get("v"), s7a.get("dfm"), s7b.get("v"), s7b.get("dfm"),
                 s7b.get("mem_dt"), s7b.get("mem_cb"),
                 s7c.get("dfm"), s7c.get("v"), s7c.get("gpu_disabled")))
        # 清理时把融合方式也还回默认（下一段 B 只看版式，但别把状态留给下一轮）
        b.eval("(function(){var e=document.getElementById('mgMode');e.value='seamless';"
               "e.dispatchEvent(new Event('change',{bubbles:true}));return true;})()")
        time.sleep(0.2)
        # 收拾干净：记忆清掉、界面回 auto（不把测试状态留给下一轮/下次人工看）
        b.eval("(function(){try{localStorage.removeItem('dfl-mg-cfg-memory');}catch(e){}"
               "document.getElementById('mgDfmDtype').value='auto';"
               "document.getElementById('mgCmpBackend').value='auto';"
               "return true;})()")
        time.sleep(0.2)

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
        check(pr["pipe_cols"] == 6, "B 流水线图六列标题都在", "%s" % pr["pipe_cols"])
        check(int(pr["pipe_lanes"] or 0) >= 6,
              "B 流水线图有 lane 方块（≥6，数量由后端 lanes 决定）",
              "%s 个" % pr["pipe_lanes"])
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
        say("640 视口：卡片 %.1f · 流水线图 %s · 容器 scroll/client %s"
            % (float(pr2["card_w"] or 0), pr2["steps_w"], pr2["steps_box"]))
        # 流水线图（SVG）是**等比缩放**的：它跟着卡片缩，不横向滚动、也不被挤扁到看不见
        # （竖排六步那版才需要「可横向滚动 + ≥420px」，那条口径随 UI 一起废掉）。
        check(float(pr2["steps_w"] or 0) >= 300 - 0.5,
              "B 640px 时流水线图仍 ≥ 300px（等比缩放，不横向溢出）",
              "steps_w=%s" % pr2["steps_w"])
        check(pr2["steps_box"] and pr2["steps_box"]["sw"] <= pr2["steps_box"]["cw"] + 1,
              "B 640px 时流水线容器自己不横向溢出（scrollWidth ≤ clientWidth）",
              "%s" % (pr2["steps_box"],))
        check(bool(pr2["label_wrap"]) and max(pr2["label_wrap"] + [0]) <= 26,
              "B 640px 时列标题不换行（单行高度 ≤26px）", "%s" % pr2["label_wrap"])
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
