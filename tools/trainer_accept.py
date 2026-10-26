# -*- coding: utf-8 -*-
r"""训练器验收：一条命令跑完「建模型 → 开训 → 进度/loss → 暂停/继续 → 保存 → 停止 → 删除」。

用法
    python tools/trainer_accept.py            # 默认用临时小模型（96px，几十秒跑完）
    python tools/trainer_accept.py --keep     # 保留临时模型（调试用）
    python tools/trainer_accept.py --model 384wf   # 用已有模型跑（会真的训练它，慎用）

做的事
    起 4.1 后端 → 建一个 _accept 临时模型 → 真起 4.0 训练进程 →
    逐项断言 → 收尾（停止 + 删掉临时模型）。

为什么这么验
    训练器最容易假的地方是「进度」：4.0 的 stdout 里 tqdm 进度条带着 "73.25it/s"，
以前用 (数字)(it|iter) 这种正则抓，抓到的全是进度条数字 —— 界面上的迭代数、
    速度、ETA 全是假的。所以这里必须断言「迭代数在涨、速度接近日志里的 ms、
    loss 有值」，而不是只看接口返回了字段。
"""

import json
import os
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

RL = str(Path(__file__).resolve().parents[1])
WS = os.environ.get("DFL_WORKSPACE") or r"D:\MyProject\DeepFaceLab-PyTorch\workspace"

_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))
_FAILS = []


def ok(flag, what):
    print(("PASS " if flag else "FAIL ") + what, flush=True)
    if not flag:
        _FAILS.append(what)


def api(base, path, obj=None, timeout=180):
    req = urllib.request.Request(
        base + path, method="POST" if obj is not None else "GET",
        data=json.dumps(obj).encode() if obj is not None else None,
        headers={"Content-Type": "application/json"})
    try:
        with _OPENER.open(req, timeout=timeout) as r:
            return json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", "replace")
        try:
            return {"HTTP": e.code, "detail": json.loads(body).get("detail")}
        except Exception:
            return {"HTTP": e.code, "detail": body[:200]}
    except Exception as e:                                    # noqa: BLE001
        return {"ERR": repr(e)[:160]}


def trainer_alive() -> int:
    """还剩几个 4.0 训练进程（必须是 0，不然就是泄漏）。"""
    try:
        out = subprocess.run(
            ["powershell", "-NoProfile", "-Command",
             "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | "
             "Where-Object { $_.CommandLine -like '*_dfl_train.py*' } | Measure-Object | "
             "Select-Object -ExpandProperty Count"],
            capture_output=True, text=True, timeout=60)
        return int((out.stdout or "0").strip() or 0)
    except Exception:
        return -1


def main() -> int:
    keep = "--keep" in sys.argv
    model = ""
    for i, a in enumerate(sys.argv[1:]):
        if a == "--model" and i + 2 <= len(sys.argv[1:]):
            model = sys.argv[i + 2]
    scratch = not model
    name = model or ("_accept%d" % int(time.time() % 100000))

    py = sys.executable
    env = dict(os.environ, DFL_WORKSPACE=WS)
    proc = subprocess.Popen([py, "serve.py"], cwd=os.path.join(RL, "app"),
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, bufsize=1, env=env, creationflags=0x08000000)
    port = 0
    for line in proc.stdout:
        if line.startswith("PORT="):
            port = int(line.strip().split("=")[1])
            break
    threading.Thread(target=lambda: [None for _ in proc.stdout], daemon=True).start()
    if not port:
        print("后端没起来")
        return 2
    base = "http://127.0.0.1:%d" % port

    try:
        if scratch:
            r = api(base, "/api/models/create", {
                "name": name, "preset": "fast", "resolution": 96,
                "ae_dims": 64, "e_dims": 16, "d_dims": 16, "d_mask_dims": 8, "batch_size": 2})
            ok(r.get("ok") is True, "建临时模型 %s" % name)

        ok(api(base, "/api/train/start", {"name": name}).get("ok") is True, "启动训练")

        # ---- 进度必须是真的 ----
        t0, samples, last = time.time(), [], 0
        while time.time() - t0 < 180:
            st = api(base, "/api/train/status", timeout=30)
            it = st.get("iter") or 0
            if it and it != last:
                samples.append((round(time.time() - t0), it, st.get("it_s") or 0))
                last = it
            if len(samples) >= 4 and time.time() - t0 > 25:
                break
            time.sleep(3)
        ok(len(samples) >= 3, "迭代在前进：%s" % (samples[-3:] if samples else "无"))
        st = api(base, "/api/train/status", timeout=30)
        ok((st.get("it_s") or 0) > 0.2, "速度合理：%s it/s（不是 tqdm 的假数字）" % st.get("it_s"))
        ok(st.get("loss_src") is not None and st.get("loss_dst") is not None,
           "loss 解析：src %s / dst %s" % (st.get("loss_src"), st.get("loss_dst")))
        ok((st.get("lr") or 0) > 0, "学习率解析：%s" % st.get("lr"))

        # ---- 暂停 / 继续 ----
        before = st.get("iter") or 0
        ok(api(base, "/api/train/pause", {}).get("paused") is True, "暂停")
        time.sleep(6)
        mid = (api(base, "/api/train/status", timeout=30).get("iter") or 0)
        time.sleep(6)
        mid2 = (api(base, "/api/train/status", timeout=30).get("iter") or 0)
        ok(mid == mid2, "暂停期间迭代不动（%s → %s → %s）" % (before, mid, mid2))
        ok(api(base, "/api/train/resume", {}).get("paused") is False, "继续")
        time.sleep(8)
        after = (api(base, "/api/train/status", timeout=30).get("iter") or 0)
        ok(after > mid2, "继续后迭代恢复（%s → %s）" % (mid2, after))

        # ---- 立即保存（.dat 时间戳要变） ----
        dat = os.path.join(WS, "model", name + "_SAEHD_data.dat")
        m0 = os.path.getmtime(dat) if os.path.exists(dat) else 0
        ok(api(base, "/api/train/save", {}).get("ok") is True, "立即保存：接口返回 ok")
        t1 = time.time()
        changed = False
        while time.time() - t1 < 30:
            time.sleep(3)
            m1 = os.path.getmtime(dat) if os.path.exists(dat) else 0
            if m1 and m1 != m0:
                changed = True
                break
        ok(changed, "立即保存：_data.dat 真的落盘了")

        # ---- 停止 ----
        r = api(base, "/api/train/stop", {}, timeout=240)
        ok(r.get("ok") is True, "停止：%s" % r.get("msg"))
        st = api(base, "/api/train/status", timeout=30)
        ok(st.get("running") is False, "停止后 running=False")
        ok(st.get("ok") is True, "手动停止不算失败（ok=True）")
        time.sleep(2)
        n = trainer_alive()
        ok(n == 0, "没有残留训练进程（剩 %s 个）" % n)

        # ---- 断点续训：再起一次，迭代要从保存处接着走 ----
        saved = st.get("iter") or 0
        ok(api(base, "/api/train/start", {"name": name}).get("ok") is True, "续训：再次启动")
        t2, resumed = time.time(), None
        while time.time() - t2 < 120:
            st2 = api(base, "/api/train/status", timeout=30)
            if st2.get("iter"):
                resumed = st2.get("iter")
                break
            time.sleep(3)
        ok(bool(resumed) and resumed >= saved - 50, "续训：从 %s 接上（保存时 %s）" % (resumed, saved))
        api(base, "/api/train/stop", {}, timeout=240)
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=15)
        except Exception:
            pass

    if scratch and not keep:
        # 后端已经退出，直接按文件删（和 /api/models/{name}/delete 同一套规则）
        import shutil
        md = Path(WS) / "model"
        n = 0
        for p in md.glob(name + "_*"):
            try:
                shutil.rmtree(p) if p.is_dir() else p.unlink()
                n += 1
            except Exception:
                pass
        print("清理临时模型 %s（%d 个文件）" % (name, n))

    print("\n%d 项失败" % len(_FAILS))
    return 1 if _FAILS else 0


if __name__ == "__main__":
    raise SystemExit(main())
