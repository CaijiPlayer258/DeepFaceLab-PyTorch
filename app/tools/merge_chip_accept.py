# -*- coding: utf-8 -*-
r"""合成页「人脸框 chip」文案验收：单选 / 多选 / 0 选。

用法
    runtime\server\Scripts\python.exe app\tools\merge_chip_accept.py
    ... --video "D:\...\workspace\9月29日 (5).mp4"     # 换素材（真视频）
    ... --models "张元英,阿証証証証-416"                  # 指定用哪两个模型（默认取列表前两个）
    ... --frame 120                                    # 指定一帧（默认自动找一张有脸的）

判读同 merge_accept：PASS / FAIL / SKIP；退出码 0 = 没有 FAIL，1 = 有 FAIL，2 = 脚本自己炸了。

它跑的是**真链路**：真后端（app/serve.py，工作区 = 仓库自己的 workspace，模型列表里
就有张元英 / 阿証…）+ 真视频（/api/merge/open 打开 mp4，帧现场用 ffmpeg 取）+ 真检测
（/api/merge/detect）+ 真页面（无头 Edge，CDP 驱动）。断言只看 DOM 里
#mgFaceLayer .mgfc .ddtxt 的文本 —— 也就是用户截图里那个 chip。

三项断言（用户报的 bug：单选张元英，chip 却写着「选择模型」）
    A 单选 1 个模型      → 每一张脸的 chip 文本 == 该模型名（不是「选择模型」）
    B 多选 2 个模型      → chip 文本 == "第一个 +1"（含第一个模型名）
    C 一个都不选         → chip 文本 == 「选择模型」
另外记录合成 / 预览回包里真正生效的 model / dfm（chip 应该和它一致）。
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import merge_accept as MA                                     # noqa: E402  复用 CDP 外壳

ROOT = Path(MA.ROOT)
APP_DIR = ROOT / "app"
DEFAULT_VIDEO = ROOT / "workspace" / "9月29日 (5).mp4"
TMP = Path(MA.TMP)


# ---------------------------------------------------------------- 页面探针
CHIP_JS = r"""
(function(){
  var txt = function(x){ return String((x && x.textContent) || "").trim(); };
  var chips = Array.prototype.map.call(
    document.querySelectorAll("#mgFaceLayer .mgfc .ddtxt"), txt);
  var rows = Array.prototype.map.call(document.querySelectorAll("#mgModels .mgitem"), function(x){
    var nm = x.querySelector(".nm");
    return { n: txt(nm), on: x.classList.contains("on"),
             /* 「已导出 .dfm」的标记挂在行类名上（DFM 徽标按用户要求删了） */
             dfm: x.classList.contains("has-dfm") };
  });
  /* 选中态从 DOM 现算（和 mgSelNames() 同源），老版本页面里也能读 —— 改前/改后同一把尺子 */
  var sel = rows.filter(function(r){ return r.on; }).map(function(r){ return r.n; });
  var bodyModel = "";
  try { bodyModel = String((window.mgCmpBody ? window.mgCmpBody(0) : {}).model || ""); } catch (e) {}
  var net = (window.__chipAccNet || []).slice(-4).map(function(n){
    return { url: n.url, status: n.status, model: n.model, dfm: n.dfm,
             faces: n.faces, ms: n.ms };
  });
  return JSON.stringify({
    chips: chips, rows: rows, sel: sel, bodyModel: bodyModel, net: net,
    frame: (window.MergeTimeline && window.MergeTimeline.getFrame) ? window.MergeTimeline.getFrame() : null,
    frames: (window.__mv || {}).frames || 0,
    videoMode: !!(window.mgVideoMode && window.mgVideoMode()),
    toasts: Array.prototype.map.call(document.querySelectorAll("#toastroot .toast"), txt).slice(-3)
  });
})()
"""

# 记录合成 / 预览回包里真正生效的 model / dfm（失败时当证据，成功时用来核对 chip）
HOOK_JS = r"""
(function(){
  if (window.__chipAccHooked) return 1;
  window.__chipAccHooked = 1;
  window.__chipAccNet = [];
  var of = window.fetch;
  window.fetch = function(u, o){
    var url = String((u && u.url) ? u.url : u);
    var p = of.apply(this, arguments);
    if (url.indexOf("/api/merge/frame") === 0 || url.indexOf("/api/merge/preview") === 0) {
      try {
        p.then(function(r){
          r.clone().text().then(function(t){
            var d = null;
            try { d = JSON.parse(t); } catch (e) { d = null; }
            window.__chipAccNet.push({
              url: url, status: r.status,
              model: d ? String(d.model || "") : "",
              dfm: d ? String(d.dfm || "") : "",
              faces: (d && d.faces) ? d.faces.length : 0,
              ms: d ? d.ms : null
            });
            window.__chipAccNet = window.__chipAccNet.slice(-8);
          }, function(){});
        }, function(){});
      } catch (e) {}
    }
    return p;
  };
  return 1;
})()
"""

# 把左栏选中态改成「只选这几个」：点掉不该选的、点开该选的
PICK_JS = r"""function(names){
  var want = {};
  (names || []).forEach(function(n){ want[String(n)] = 1; });
  var clicked = 0;
  Array.prototype.forEach.call(document.querySelectorAll("#mgModels .mgitem"), function(r){
    var nm = r.querySelector(".nm");
    var name = String((nm && nm.textContent) || "").trim();
    var on = r.classList.contains("on");
    var should = !!want[name];
    if (on !== should) { r.click(); clicked++; }
  });
  return clicked;
}"""


def _short(d, n=800):
    try:
        s = json.dumps(d, ensure_ascii=False)
    except Exception:                                       # noqa: BLE001
        s = str(d)
    s = s.replace("\n", " ")
    return s if len(s) <= n else s[:n] + "…"


class Report:
    def __init__(self):
        self.items = []

    def _emit(self, status, what, detail="", ctx=None):
        line = "%-4s %s" % (status, what)
        if detail:
            line += " · " + detail
        print(line)
        if ctx is not None and status != "PASS":
            print("     实际：%s" % _short(ctx, 900))
        self.items.append({"status": status, "what": what, "detail": detail})

    def ok(self, flag, what, detail="", ctx=None):
        self._emit("PASS" if flag else "FAIL", what, detail,
                   None if flag else (ctx if ctx is not None else {}))
        return bool(flag)

    def skip(self, what, why, ctx=None):
        self._emit("SKIP", what, why, ctx)

    def fail(self, what, why="", ctx=None):
        self._emit("FAIL", what, why, ctx if ctx is not None else {})


def _wait(b, pred, timeout, interval=0.25):
    t0 = time.time()
    last = {}
    while time.time() - t0 < timeout:
        try:
            last = b.js(CHIP_JS)
        except Exception as e:                              # noqa: BLE001
            last = {"error": str(e)[:200]}
        if pred(last):
            return True, last, round(time.time() - t0, 2)
        time.sleep(interval)
    return False, last, round(time.time() - t0, 2)


def _pick(b, names):
    return b.eval("(" + PICK_JS + ")(%s)" % json.dumps(list(names), ensure_ascii=False))


def _pick_only(b, names, timeout=20):
    """点选到只剩 names（按列表顺序）；返回 (ok, 选中态快照)。"""
    _pick(b, names)
    t0 = time.time()
    last = {}
    want = sorted(names)
    while time.time() - t0 < timeout:
        last = b.js(CHIP_JS)
        if sorted(last.get("sel") or []) == want:
            return True, last
        time.sleep(0.2)
    return False, last


def _detect_frame(b, det, total, want=0):
    """找一张真有人脸的帧：先试 want，再按步长扫。返回 (帧号, 框数, 接口证据)。"""
    cands = []
    if want is not None:
        cands.append(int(want))
    step = max(1, int(total or 300) // 12)
    cands += [i * step for i in range(0, 12)]
    seen = set()
    for c in cands:
        if c in seen or c < 0 or (total and c > total - 1):
            continue
        seen.add(c)
        raw = b.js(MA._js_fetch("/api/merge/detect/%d?model=%s&thresh=0.5" % (c, det)))
        body = (raw or {}).get("body") or {}
        if (raw or {}).get("status") == 200 and (body.get("boxes") or []):
            return c, len(body["boxes"]), {"status": raw.get("status"), "boxes": len(body["boxes"])}
    return None, 0, {"detect": raw if seen else None}


def _draw_boxes(b, frame, timeout=60):
    """把这一帧的框 + chip 画出来（就是 detect 回包那一下做的事）。

    顺序照页面的真实链路：先把时间轴停到这一帧（mvDetect 会拿
    MergeTimeline.getFrame() 比对，不一致就把回包当 "outdated" 丢掉、不画框），
    再发那一发检测。
    """
    MA._set_frame(b, frame)
    b.eval("window.mvFrame && window.mvFrame(%d), 1" % frame)
    _wait(b, lambda s: s.get("frame") == frame, 20, 0.2)
    b.eval("window.mvDetect && window.mvDetect(%d, {force:true}), 1" % frame)
    return _wait(b, lambda s: len(s.get("chips") or []) > 0, timeout, 0.25)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default=str(DEFAULT_VIDEO))
    ap.add_argument("--models", default="")
    ap.add_argument("--frame", type=int, default=None)
    ap.add_argument("--scale", type=float, default=1.0)
    opt = ap.parse_args()

    r = Report()
    video = Path(opt.video)
    edge = MA._edge_path()
    py = MA._server_python()
    if not edge:
        r.skip("整轮", "找不到 Edge（装 Edge 或设 EDGE_PATH）")
        return r.items
    if not py or MA._ws_connect is None:
        r.skip("整轮", "当前解释器缺 fastapi/uvicorn/websockets（用 runtime\\server\\Scripts\\python.exe 跑）")
        return r.items
    if not video.is_file():
        r.skip("整轮", "素材视频不在：%s" % video)
        return r.items

    log_path = TMP / "merge_chip_accept.log"
    log_fh = open(str(log_path), "w", encoding="utf-8")
    proc = browser = None
    try:
        proc, port, tail = MA._server_start(py, ROOT / "workspace", log_fh)
        if not port:
            r.fail("后端启动", "serve.py 没打出 PORT=", {"server_log": tail[-10:]})
            return r.items
        print("[后端] port=%d · 工作区 %s" % (port, ROOT / "workspace"))

        browser = MA.Browser(edge)
        if not browser.goto("http://127.0.0.1:%d/" % port):
            r.fail("打开界面", "readyState 一直不到 complete", {"console": browser.console[-4:]})
            return r.items
        browser.eval('window.go && window.go("merge"), 1')
        ok, s, dt = _wait(browser, lambda x: bool(x.get("rows")),
                          MA.T_MERGE_UI, 0.3)
        browser.eval(HOOK_JS)

        # ---------- 打开真视频（/api/merge/open → 帧现场从 mp4 取） ----------
        browser.eval("window.mgOpenPrepared && window.mgOpenPrepared(%s), 1" % json.dumps(str(video)))
        ok, s, dt = _wait(browser, lambda x: (x.get("frames") or 0) > 0, 120, 0.4)
        if not r.ok(ok, "打开素材", "%s · %s 帧 · %s · %.1fs"
                    % (video.name, (s or {}).get("frames"), "video" if (s or {}).get("videoMode") else "frame",
                       dt),
                    ctx=None if ok else {"snap": s, "server_log": tail[-8:]}):
            return r.items
        total = int(s.get("frames") or 0)

        # ---------- 模型列表 ----------
        ok, s, dt = _wait(browser, lambda x: len(x.get("rows") or []) > 0, MA.T_MODELS, 0.3)
        rows = [x for x in (s.get("rows") or []) if x.get("dfm")]
        if not ok or not rows:
            r.skip("模型列表", "工作区里没有导出过 .dfm 的模型", {"snap": s})
            return r.items
        names = [x["n"] for x in rows]
        want = [n.strip() for n in str(opt.models or "").split(",") if n.strip()]
        want = [n for n in want if n in names]
        if len(want) < 2:
            want = names[:2]
        if len(want) < 2:
            r.skip("模型列表", "只有 %d 个带 .dfm 的模型（这项要 2 个才能验多选）" % len(want),
                   {"models": names})
            return r.items
        first, second = want[0], want[1]
        r.ok(True, "模型列表", "%d 个 · 本次用 %s / %s" % (len(names), first, second))

        # ---------- 检测器 + 一张有脸的帧 ----------
        det = browser.eval("(window.mgDetectorName ? window.mgDetectorName() : '') || ''") or "RetinaFace_10g"
        frame, nboxes, dev = _detect_frame(browser, det, total, opt.frame)
        if frame is None:
            r.skip("找一张有脸的帧", "扫了 12 帧都没人脸（--frame 指定一帧）", dev)
            return r.items
        r.ok(True, "找一张有脸的帧", "帧 %d · %d 个框 · 检测器 %s" % (frame, nboxes, det))

        ok, s, dt = _draw_boxes(browser, frame, 90 * opt.scale)
        if not r.ok(ok, "画出人脸框和 chip", "帧 %d · %d 个 chip · %.1fs"
                    % (frame, len((s or {}).get("chips") or []), dt),
                    ctx=None if ok else {"snap": s, "console": browser.console[-4:]}):
            return r.items

        # ---------- A 单选 1 个模型 ----------
        ok, s = _pick_only(browser, [first])
        chips = (s or {}).get("chips") or []
        if not ok:
            r.fail("A 单选：选中态", "没点成只选 %s" % first, {"snap": s})
        if not chips:
            ok2, s2, _ = _draw_boxes(browser, frame, 30)
            chips = (s2 or {}).get("chips") or []
        all_first = bool(chips) and all(c == first for c in chips)
        r.ok(all_first, "A 单选 1 个模型：chip == 模型名",
             "chip=%s · 选中=%s" % (json.dumps(chips, ensure_ascii=False),
                                  json.dumps((s or {}).get("sel"), ensure_ascii=False)),
             ctx=None if all_first else {"chips": chips, "sel": (s or {}).get("sel"), "snap": s})

        # 合成一次，核对后端真正生效的 model / dfm 和 chip 是否一致
        browser.eval("window.mgCmpNow && window.mgCmpNow(), 1")
        ok_net, s_net, _ = _wait(browser, lambda x: any(n.get("model") for n in (x.get("net") or [])),
                                 60 * opt.scale, 0.3)
        net = [n for n in ((s_net or {}).get("net") or []) if n.get("model")]
        be_model = net[-1]["model"] if net else ""
        chip0 = chips[0] if chips else ""
        r.ok((not be_model) or (chip0 == be_model), "A chip 文本 == 后端回包的 model",
             "chip=%s · 回包 model=%s · dfm=%s · %sms"
             % (json.dumps(chip0, ensure_ascii=False), be_model,
                (net[-1].get("dfm") if net else ""), (net[-1].get("ms") if net else None)),
             ctx=None if (not be_model or chip0 == be_model) else {"net": net, "chips": chips})

        # ---------- B 多选 2 个模型 ----------
        ok, s = _pick_only(browser, [first, second])
        chips = (s or {}).get("chips") or []
        want_b = first + " +1"
        all_b = bool(chips) and all(c == want_b for c in chips)
        r.ok(all_b, "B 多选 2 个模型：chip 含第一个模型名",
             "chip=%s · 期望 %s · 选中=%s" % (json.dumps(chips, ensure_ascii=False), want_b,
                                            json.dumps((s or {}).get("sel"), ensure_ascii=False)),
             ctx=None if all_b else {"chips": chips, "sel": (s or {}).get("sel"), "snap": s})

        # ---------- C 一个都不选 ----------
        ok, s = _pick_only(browser, [])
        chips = (s or {}).get("chips") or []
        all_c = bool(chips) and all(c == "选择模型" for c in chips)
        r.ok(all_c, "C 0 个模型：chip == 选择模型",
             "chip=%s · 选中=%s" % (json.dumps(chips, ensure_ascii=False),
                                  json.dumps((s or {}).get("sel"), ensure_ascii=False)),
             ctx=None if all_c else {"chips": chips, "sel": (s or {}).get("sel"), "snap": s})

        if (s or {}).get("toasts"):
            print("[toast] %s" % json.dumps(s["toasts"], ensure_ascii=False))
        return r.items
    finally:
        if browser is not None:
            browser.close()
        if proc is not None:
            try:
                subprocess.run(["taskkill", "/T", "/F", "/PID", str(proc.pid)],
                               capture_output=True, timeout=30)
            except Exception:                               # noqa: BLE001
                pass
        try:
            log_fh.write("log: %s\n" % log_path)
            log_fh.close()
        except Exception:                                   # noqa: BLE001
            pass


if __name__ == "__main__":
    try:
        items = main()
    except SystemExit:
        raise
    except Exception as e:                                  # noqa: BLE001
        import traceback
        traceback.print_exc()
        print("脚本自己炸了：%s" % e)
        raise SystemExit(2)
    bad = [x for x in items if x["status"] == "FAIL"]
    print("\n合计：PASS %d · FAIL %d · SKIP %d"
          % (sum(1 for x in items if x["status"] == "PASS"), len(bad),
             sum(1 for x in items if x["status"] == "SKIP")))
    raise SystemExit(1 if bad else 0)
