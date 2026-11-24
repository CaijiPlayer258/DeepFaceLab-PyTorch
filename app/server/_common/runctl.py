# -*- coding: utf-8 -*-
"""运行控制：训练 / DFM 导出 / 训练进程清扫 / 应用启动退出钩子（从 app/server/app.py 切出）。

两件容易踩的事，和原文件一模一样：

  · global _train_proc 指的必须是**同一个**模块全局 —— 所以 _train_proc、_train_worker、
    _sweep_orphan_trainers、_kill_train_tree 全都在本文件里，别拆开。
  · Path(__file__).parent / "_dfl_train.py" 原来指 app/server/，本文件在 app/server/_common/，
    下标改成 parents[1]。

函数体 / 注释 / 日志文案逐字节照抄 app.py；只改上面那一处 __file__ 下标 + 补 import。
"""
from __future__ import annotations

import os
import threading
import time
from pathlib import Path

from .. import _load_events
from .bootstrap import _NO_WINDOW
from .model_info import _engine_has_flag, _engine_supports_cpu_only, _gpu_list, _model_dat
from .paths import WORKSPACE, _DFL_PY, _DFL_ROOT, _DFL_TOOL, engine_python as _engine_python


def _sweep_orphan_trainers(why: str = "") -> int:
    """杀掉不属于当前对话的训练进程（命令行里带 _dfl_train.py 的那些）。"""
    global _train_proc
    try:
        import psutil
    except Exception:
        return 0
    mine = _train_proc.pid if _train_proc is not None else 0
    killed = 0
    try:
        for p in psutil.process_iter(["pid", "cmdline"]):
            try:
                cmd = " ".join(p.info.get("cmdline") or [])
                if "_dfl_train.py" not in cmd or p.pid == mine:
                    continue
                # 别误杀另一个 4.1 实例正在用的训练进程：
                # 只有当它的父进程已经没了（或者父进程不是 serve.py）才算孤儿。
                managed = False
                try:
                    par = p.parent()
                    if par is not None and par.is_running():
                        pcmd = " ".join(par.cmdline() or [])
                        managed = ("serve.py" in pcmd) or ("app" in pcmd and "python" in pcmd.lower())
                except Exception:
                    managed = False
                if managed:
                    continue
                p.kill()
                killed += 1
            except Exception:
                continue
    except Exception:
        return killed
    if killed:
        with _train_job_lock:
            _train_job["running"] = False
            _train_job["paused"] = False
        print("[trainer] 清理遗留训练进程 %d 个 %s" % (killed, why), flush=True)
    return killed

def _startup_sweep():
    """上一次 4.1 没正常退出时，训练进程会留在这儿。开服先收掉。"""
    _sweep_orphan_trainers("(启动清扫)")
    # 进程内的人脸 embedding / 组均值缓存：**进程重启就清空**（不落盘复用）。
    # 这里就是"重启"那一次调用点；另一次在每次导出 job 结束（routes_merge 的
    # _vid_worker 的 finally）。启动时本来就该是空的，这次调用是口径上的保证，
    # 顺便把清理结果打进日志，验收可以直接断言「profiles=0 · files=0」。
    try:
        from .._face_group import clear_face_group_cache
        st = clear_face_group_cache("serve 启动")
        print("[face-cache] 启动清理：组均值 %s 条 · 其它 %s · 磁盘缓存文件 %s 个"
              % (st.get("profiles"), st.get("extra") or "无", len(st.get("files") or [])),
              flush=True)
    except Exception as e:                              # noqa: BLE001 - 缓存清理不该挡住开服
        print("[face-cache] 启动清理失败（不影响使用）：%s" % str(e)[:160], flush=True)

def _shutdown_cleanup():
    """4.1 退出时把训练进程一起收掉 —— 否则它会变成孤儿继续占显存。”"""
    with _train_job_lock:
        running = _train_job["running"]
    if not running:
        return
    try:
        _webui_post("/request-quit", timeout=2.0)
    except Exception:
        pass
    for _ in range(20):
        if _train_proc is None or _train_proc.poll() is not None:
            return
        time.sleep(0.5)
    _kill_train_tree()

# ============================================================
# DFM 导出（内置在合成器里）
# 用 4.0 的 python（有 torch 2.7 + onnx）跑 tools/migrate/export_dfm.py
# ============================================================
# 导出的三个阶段（前端步进器按它点亮）
_EXPORT_STEPS = [("load", "加载权重"), ("pack", "组装 DFM"), ("write", "写入文件")]

_export_job = {"running": False, "name": "", "log": [], "ok": None, "error": None,
               "step": "idle", "modules": [], "percent": 0}

# 4.0 训练进程自带的监控服务端口（4.1 只从它拉预览，不展示那个界面）
WEBUI_PORT = 6790
# 4.0 的 WebUI 默认密码是 caiji；4.1 自己指定一个，控制接口（保存/退出）要用它
WEBUI_PASSWORD = "dfl41-webui"

def _arch_of(name: str) -> str:
    """从 *_<ARCH>_data.dat 反推架构名。"""
    md = WORKSPACE / "model"
    for dat in md.glob(name + "_*_data.dat"):
        stem = dat.name[: -len("_data.dat")]
        _, _, arch = stem.rpartition("_")
        return arch or "SAEHD"
    return "SAEHD"


def _snapshot_dats(name: str):
    """记下这个模型的 _data.dat 原始字节，导出后要还原。"""
    md = WORKSPACE / "model"
    snap = {}
    for dat in md.glob(name + "_*_data.dat"):
        try:
            snap[dat] = dat.read_bytes()
        except Exception:
            pass
    return snap


def _restore_dats(snap: dict):
    """导出脚本会改 _data.dat（把 iter 归 1 / 重写文件），这里原样还原。"""
    import shutil
    restored = []
    for dat, raw in snap.items():
        try:
            if dat.exists() and dat.read_bytes() == raw:
                continue
            shutil.copy2(dat, str(dat) + ".before-export")
            dat.write_bytes(raw)
            restored.append(dat.name)
        except Exception:
            pass
    return restored


def _export_worker(name: str, arch: str):
    import subprocess
    snap = _snapshot_dats(name)
    cmd = [str(_DFL_PY), str(_DFL_TOOL),
           "--model", arch,
           "--model-dir", str(WORKSPACE / "model"),
           "--force-name", name]
    with _export_job_lock:
        _export_job.update({"running": True, "name": name, "log": [], "ok": None, "error": None,
                           "step": "load", "percent": 0,
                           "modules": [{"name": n, "state": ("run" if k == "load" else "wait")}
                                       for k, n in _EXPORT_STEPS]})
    try:
        proc = subprocess.Popen(cmd, cwd=str(_DFL_ROOT), creationflags=_NO_WINDOW,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, encoding="utf-8", errors="replace", bufsize=1)
        # 4.0 里那几行 WebUI 提示已经没意义了（服务器被摘掉了），别往日志里塞
        _TRAIN_LOG_SKIP = ("等待模型加载完成后启动 WebUI", "模型已加载，正在启动 WebUI", "[WebUI]")

        for line in proc.stdout:
            line = line.rstrip()
            if not line:
                continue
            if any(k in line for k in _TRAIN_LOG_SKIP):
                continue
            if line.startswith("[stage]"):
                _st = line.split()[1] if len(line.split()) > 1 else ""
                _keys = [k for k, _ in _EXPORT_STEPS]
                if _st in _keys:
                    _i = _keys.index(_st)
                    _mods = [{"name": n,
                              "state": ("ok" if _j < _i else ("run" if _j == _i else "wait"))}
                             for _j, (_k2, n) in enumerate(_EXPORT_STEPS)]
                    with _export_job_lock:
                        _export_job["step"] = _st
                        _export_job["modules"] = _mods
                        _export_job["percent"] = int(_i * 100 / len(_EXPORT_STEPS))
                continue
            with _export_job_lock:
                _export_job["log"].append(line)
                if len(_export_job["log"]) > 400:
                    _export_job["log"] = _export_job["log"][-400:]
        proc.wait()
        ok = proc.returncode == 0
        restored = _restore_dats(snap)
        size_note = ""
        dfm = WORKSPACE / "model" / (name + ".dfm")
        if dfm.is_file():
            mb = dfm.stat().st_size / 1048576
            if mb < 1:
                ok = False
                size_note = ("导出产物只有 %.2f MB，明显不完整（正常几百 MB）。"
                             "4.0 的 export_dfm.py 对 liae 架构可能只写了结构，"
                             "权重没进去 —— 这是导出脚本的问题，不是界面。") % mb
        with _export_job_lock:
            _export_job["running"] = False
            _export_job["ok"] = ok
            if ok:
                _export_job["step"] = "done"
                _export_job["percent"] = 100
                _export_job["modules"] = [{"name": n, "state": "ok"}
                                          for _k, n in _EXPORT_STEPS]
            if size_note:
                _export_job["error"] = size_note
                _export_job["log"].append("[!] " + size_note)
            if restored:
                _export_job["log"].append("[!] 导出脚本改动了 _data.dat，已还原: " + ", ".join(restored))
            if not ok:
                _export_job["error"] = _export_job.get("error") or ("导出进程退出码 %d" % proc.returncode)
                _export_job["step"] = "error"
                # 当前阶段标红，带上最后一行日志当原因
                _tail = [x for x in (_export_job.get("log") or []) if x.strip()][-1:]
                _why = (_tail[0] if _tail else _export_job["error"])[:220]
                _cur = next((i for i, m in enumerate(_export_job.get("modules") or [])
                             if m.get("state") == "run"), 0)
                _mods = _export_job.get("modules") or []
                if _mods:
                    _mods[_cur]["state"] = "bad"
                    _mods[_cur]["reason"] = _why
                    _export_job["modules"] = _mods
    except Exception as e:
        with _export_job_lock:
            _export_job["running"] = False
            _export_job["ok"] = False
            _export_job["error"] = str(e)

import threading as _threading
_export_job_lock = _threading.Lock()

# ============================================================
# 训练：调 4.0 的 main.py train（真起进程，日志回传）
# ============================================================
_train_job = {"running": False, "name": "", "log": [], "ok": None, "error": None, "fatal": None,
              "iter": 0, "it_s": 0.0, "iter_at": 0.0, "started_at": 0.0, "target": 0,
              "paused": False, "saving": False, "lr": 0.0,
              "loss_src": None, "loss_dst": None,
              # 模块加载失败的**结构化事件**（训练进程 stdout 里的 [DFL-LOAD-EVENT] 行）。
              # 它是状态不是一次性通知：只要训练还在跑，刷新页面 / 重进训练页
              # 都能再从 /api/train/status 拿到，于是横幅重新出现。
              "warnings": [],
              "history": [], "webui_ready": False, "stopping": False}
_train_proc = None        # 4.0 训练主进程（控制/清理都要它）
# 最后一次拿到的实时预览（4.0 的 /preview 是破坏性 pop，得自己留一份给前端反复读）
_last_preview_state = {"body": None, "iter": 0, "at": 0.0}


def _merge_warning(existing, ev):
    """把一条新事件并进 warnings：**同一个模块只留最后一条**。

    模型反复重启训练时同一个模块会反复失败，界面上不该堆成十几条横幅。
    """
    out = [w for w in (existing or []) if w.get("module") != ev.get("module")]
    out.append(ev)
    return out

def _train_worker(name: str, cpu_only: bool):
    import re
    import pickle as _pk
    import subprocess
    py = _engine_python()
    main_py = _DFL_ROOT / "main.py"
    env = dict(**__import__("os").environ)
    env["DFL_SILENT_INPUT"] = "1"
    # 不设这个，4.0 按系统代码页（GBK）往管道里写，我们按 UTF-8 解出来全是乱码：
    # 「正在创建备份」匹配不到、日志尾巴也没法看。
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUNBUFFERED"] = "1"

    # ── 设备选择：options["device"] → CUDA_VISIBLE_DEVICES / --force-gpu-idxs ──
    # 以前这里**从来不读 options.device**：用户在参数页选好 0 号卡，引擎照样走
    # nn.DeviceConfig.BestGPU()（显存最大的那张），而且不报错、不提示。
    # 现在：解析 → 校验 → 映射全部交给 _train_device.plan()，非法值在起进程之前就拒。
    from .. import _train_device as _tdev
    _dev_raw = None
    try:
        _dat_dev = _model_dat(name)
        if _dat_dev:
            _dev_raw = (_pk.loads(_dat_dev.read_bytes()).get("options") or {}).get("device")
    except Exception:
        _dev_raw = None
    try:
        _dev = _tdev.plan(_dev_raw, _gpu_list(), _engine_supports_cpu_only())
    except _tdev.DeviceOptionError as _e:
        # 启动前拒绝，绝不静默回退到"显存最大的那张卡"。任务状态也置成失败，
        # 这样界面上是红色错误而不是"训练中 + 永远 0 迭代"。
        _emsg = str(_e)
        print("[train] 设备选择无效，已拒绝启动：" + _emsg, flush=True)
        with _train_job_lock:
            _train_job.update({
                "running": False, "name": name, "ok": False, "error": _emsg,
                "log": ["[train] 设备选择无效，已拒绝启动：" + _emsg],
                "warnings": [], "history": [], "iter": 0, "iter_seen": False,
                "it_s": 0.0, "lr": 0.0, "loss_src": None, "loss_dst": None,
                "paused": False, "saving": False, "stopping": False, "pid": None,
                "device": {"raw": _tdev.raw_text(_dev_raw), "mode": "invalid",
                           "error": _emsg, "available": _gpu_list()}})
        return

    if _dev["cuda_visible"]:
        env["CUDA_VISIBLE_DEVICES"] = _dev["cuda_visible"]
        # 引擎 core/leras/device.py 的 initialize_main_env() 会主动 pop 掉
        # CUDA_VISIBLE_DEVICES（它假定那是父进程留下的脏值）。这个哨兵告诉它：
        # 这次的值是启动器算出来的，别动 —— 否则 pop 之后 CUDA 立刻看见全部卡，
        # 我设的 env 就成了纯装饰（实测 device_count 从 0 变回 1）。
        env["DFL_KEEP_CUDA_VISIBLE_DEVICES"] = "1"
    if _dev["cpu_only"]:
        cpu_only = True

    # src/dst 是必填：优先用模型 options 里记的路径，否则用工作区默认的 aligned 目录
    src_dir = dst_dir = ""
    try:
        dat = _model_dat(name)
        if dat:
            opts = (_pk.loads(dat.read_bytes()).get("options") or {})
            src_dir = str(opts.get("src") or "")
            dst_dir = str(opts.get("dst") or "")
    except Exception:
        pass
    if not src_dir:
        src_dir = str(WORKSPACE / "data_src" / "aligned") if (WORKSPACE / "data_src" / "aligned").is_dir() else str(WORKSPACE / "data_src")
    if not dst_dir:
        dst_dir = str(WORKSPACE / "data_dst" / "aligned") if (WORKSPACE / "data_dst" / "aligned").is_dir() else str(WORKSPACE / "data_dst")

    # 预训练集：以前这里**从来不给引擎传** --pretraining-data-dir，
    # 所以参数页里填的「预训练集」形同虚设（引擎只会退回到 dst）。
    # 现在：dat 里 pretrain=True 且路径非空 → 传下去；空着 → 不传，
    # 由引擎走它自己的默认（dst 训练集）。
    pt_dir = ""
    try:
        if dat:
            _po = (_pk.loads(dat.read_bytes()).get("options") or {})
            if _po.get("pretrain"):
                pt_dir = str(_po.get("pretrain_dataset_path") or "").strip()
    except Exception:
        pt_dir = ""

    # 走包装脚本：把 4.0 的 WebUI 和自动开浏览器摘掉，其余照旧
    boot = Path(__file__).resolve().parents[1] / "_dfl_train.py"
    entry = str(boot) if boot.is_file() else str(main_py)
    cmd = [str(py), entry, "train",
           "--training-data-src-dir", src_dir,
           "--training-data-dst-dir", dst_dir,
           "--model-dir", str(WORKSPACE / "model"),
           "--model", "SAEHD",
           "--force-model-name", name,
           "--no-preview",
           "--silent-start",
           "--webui-port", str(WEBUI_PORT),
           "--webui-password", WEBUI_PASSWORD]
    if pt_dir:
        cmd += ["--pretraining-data-dir", pt_dir]
    if cpu_only:
        cmd.append("--cpu-only")
    if _dev["force_gpu_idxs"]:
        # 逻辑序号（0..n-1），不是物理序号：CUDA_VISIBLE_DEVICES 已经把可见集合
        # 重排过了，传物理 0,2 会让引擎取不到 2 号逻辑卡，两卡退化成单卡。
        if _engine_has_flag("--force-gpu-idxs"):
            cmd += ["--force-gpu-idxs", _dev["force_gpu_idxs"]]
        elif len(_dev["physical"]) > 1:
            # env 已经把可见集合限住了，但引擎会自己挑一张 → 多卡会退化成单卡。
            # 与其让用户以为在双卡训练，不如直接说出来。
            _w = ("这个引擎的 train 子命令没有 --force-gpu-idxs，device=%s 的多卡选择"
                  "只会用其中一张（CUDA_VISIBLE_DEVICES=%s 已生效）"
                  % (_dev["raw"], _dev["cuda_visible"]))
            _dev = dict(_dev, warn=((_dev.get("warn") + "；") if _dev.get("warn") else "") + _w)

    # 梯度累计：4.1 界面会把 target_batch 拆成 batch_size / accum_steps，
    # 但引擎目前**没有实现梯度累计**（Model_SAEHD 里只有一句注释说明它没有）。
    # 与其让人以为等效 batch 生效，不如开局就说明白。
    try:
        _o = (__import__("pickle").loads(_model_dat(name).read_bytes()).get("options") or {}) if _model_dat(name) else {}
        _acc = int(_o.get("accum_steps") or 0)
        if _acc > 1:
            print("[trainer] 注意：accum_steps=%d（目标 batch %s）目前不会生效 —— "
                  "引擎尚未实现梯度累计，实际按 batch_size=%s 训练。"
                  % (_acc, _o.get("target_batch") or "?", _o.get("batch_size") or 1), flush=True)
    except Exception:
        pass

    # 目标迭代次数（模型 options 里记着，用来算 ETA）
    target = 0
    try:
        dat = _model_dat(name)
        if dat:
            target = int((_pk.loads(dat.read_bytes()).get("options") or {}).get("target_iter", 0) or 0)
    except Exception:
        target = 0

    with _train_job_lock:
        # 上一次的 iter / loss / history 必须清掉：否则第二次开训时界面立刻显示
        # "训练中 + 上次的迭代数"，而模型其实还在加载 —— 用户看到的就是假的进度。
        _train_job.update({"running": True, "name": name, "log": [], "ok": None,
                           "error": None, "webui_port": WEBUI_PORT,
                           "it_s": 0.0, "iter_at": 0.0, "started_at": time.time(),
                           "target": target,
                           "iter": 0, "iter_seen": False, "lr": 0.0,
                           "loss_src": None, "loss_dst": None, "history": [],
                           "paused": False, "saving": False, "stopping": False,
                           # 上一轮的模块加载失败事件不能留到这一轮：
                           # 否则新开的训练一开始就挂着上回那条横幅（状态是假的）。
                           "warnings": [],
                           "last_backup_iter": 0, "pid": None,
                           # 设备选择的结果本身是"状态"：/api/train/status 每次都能
                           # 回显 device=<值> → CUDA_VISIBLE_DEVICES=<env>，用户能核对。
                           "device": _dev})
        # 启动日志也要有这一行（否则只有引擎的 stdout，看不出到底选了哪张卡）
        _train_job["log"].append(_dev["banner"])
        if _dev.get("warn"):
            _train_job["log"].append("[train] " + _dev["warn"])
    print(_dev["banner"], flush=True)
    if _dev.get("warn"):
        print("[train] " + _dev["warn"], flush=True)
    global _train_proc
    # 4.0 训练进度行的真实格式：
    #   [08:42:26][#002012][0061ms][lr:5.00e-05][1.0574][1.8756]
    #   时间       迭代     每轮耗时   学习率        src_loss dst_loss
    # 以前用 r"(\d+)\s*(?:it|iter)" 抓，抓到的是 tqdm 进度条里的 "73.25it/s" —— 迭代数
    # 和速度全是假的。这里按行首的 [#数字] 精确匹配。
    ITER_RE = re.compile(r"^\[[\d:]+\]\[#(\d+)\]\[(\d+)ms\]\[lr:([0-9.eE+-]+)\]"
                         r"\[([0-9.]+)\]\[([0-9.]+)\]")
    try:
        proc = subprocess.Popen(cmd, cwd=str(_DFL_ROOT), env=env, creationflags=_NO_WINDOW,
                                stdin=subprocess.DEVNULL,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, encoding="utf-8", errors="replace", bufsize=1)
        _train_proc = proc
        with _train_job_lock:
            _train_job["pid"] = proc.pid
        for line in proc.stdout:
            line = line.rstrip()
            if not line:
                continue
            # 模块加载失败的结构化事件（引擎 core/leras/load_events.py 打的 JSON 行）。
            # 它同时是"状态"，所以要进 _train_job，由 /api/train/status 的 warnings[]
            # 交给界面 —— 刷新页面后横幅还能重新出现，靠的就是这里存住了。
            _lev = _load_events.parse_line(line)
            m = ITER_RE.match(line)
            with _train_job_lock:
                if _lev is not None:
                    _train_job["warnings"] = _merge_warning(
                        _train_job.get("warnings") or [], _lev)
                _train_job["log"].append(line[:400])
                if len(_train_job["log"]) > 300:
                    _train_job["log"] = _train_job["log"][-300:]
                # 显存不够是最常见的启动失败：4.0 只往 stdout 打一坨 Python traceback，
                # 用户看到的是 "CUDA out of memory" 原文，不知道该怎么办。这里翻译成人话，
                # 顺便给出几条真正能降显存的开关。
                # 训练数据为空是最常见的「新建模型直接开训」错误。4.0 抛完 ValueError
                # 之后并不退出（它会继续等），于是界面上一直是「加载模型…已等 N 分钟」，
                # 用户以为在加载、实际卡死。这里看到这行就立刻杀掉并给出能照做的提示。
                if not _train_job.get("fatal"):
                    _l2 = line.lower()
                    if ("no training data provided" in _l2
                            or ("training data" in _l2 and "valueerror" in _l2)):
                        _train_job["fatal"] = (
                            "训练数据是空的：data_src / data_dst 里没有当前分辨率的对齐脸。"
                            "\n1) 先跑「提取器」把素材对齐（注意对齐分辨率要和模型一致）"
                            "\n2) 或者换一个已经有对齐脸的模型（分辨率必须一致，4.0 不做自动缩放）")
                        _train_job["error"] = _train_job["fatal"]
                        _kill_train_tree()
                        break
                if not _train_job.get("oom_hint"):
                    _low = line.lower()
                    if ("out of memory" in _low or "cuda error: out of memory" in _low
                            or "cublas_status_alloc_failed" in _low):
                        _train_job["oom_hint"] = (
                            "显存不足（CUDA OutOfMemory）。可以按这个顺序试：\n"
                            "1) 把分辨率降一档，或把批次大小减半\n"
                            "2) 打开「梯度检查点」use_gradient_checkpointing（用时间换显存，通常省 30~50%）\n"
                            "3) 关掉「优化器GPU」models_opt_on_gpu（优化器状态挪回内存，约省 参数个数×8 字节）\n"
                            "4) 关掉预览（预览要额外做一次完整前向）\n"
                            "5) 关掉其它占显存的程序（浏览器硬件加速、视频播放器）"
                        )
                        _train_job["error"] = _train_job["oom_hint"]
                if "正在创建备份" in line:
                    # 4.0 打这行时正在备份，当前 iter 就是备份点
                    _train_job["last_backup_iter"] = _train_job["iter"]
                    _train_job["last_backup_at"] = time.time()
                if m:
                    it = int(m.group(1))
                    ms = max(1, int(m.group(2)))
                    _train_job["iter_seen"] = True
                    _train_job["iter"] = it
                    _train_job["iter_at"] = time.time()
                    _train_job["lr"] = float(m.group(3))
                    _train_job["loss_src"] = float(m.group(4))
                    _train_job["loss_dst"] = float(m.group(5))
                    inst = 1000.0 / ms
                    old_rate = _train_job["it_s"] or inst
                    _train_job["it_s"] = round(old_rate * 0.85 + inst * 0.15, 2)
                    hist = _train_job["history"]
                    hist.append([it, round(float(m.group(4)), 4), round(float(m.group(5)), 4)])
                    if len(hist) > 400:
                        del hist[:-400]
        proc.wait()
        ok = proc.returncode == 0
        with _train_job_lock:
            manual = bool(_train_job.get("stopping"))
            _train_job["running"] = False
            _train_job["paused"] = False
            _train_job["stopping"] = False
            # 用户点「停止」时进程可能是被强杀的（退出码非 0），不该显示成"训练失败"
            _train_job["ok"] = ok or manual
            if not ok and not manual:
                rc = proc.returncode
                # Windows 上被外部杀掉会报 4294967295（其实就是 -1），直接印出来没人看得懂
                if rc is not None and rc >= 0x80000000:
                    rc -= 0x100000000
                if rc is None or rc < 0:
                    _train_job["error"] = "训练进程被外部结束（退出码 %s）—— 是不是任务管理器里结束了它？" % rc
                else:
                    _train_job["error"] = "训练进程退出码 %d" % rc
            elif manual:
                _train_job["error"] = None
    except Exception as e:
        with _train_job_lock:
            _train_job["running"] = False
            _train_job["ok"] = False
            _train_job["error"] = str(e)

import threading as _th3
_train_job_lock = _th3.Lock()

def _repair_option_strings(name: str) -> list:
    """把 options 里 "1.0（关闭）" 这类被当数值写进去的字符串还原成数字。

    4.0 读 options 时直接 float()，带着中文说明就会崩：
        ValueError: could not convert string to float: '1.0（关闭）'
    这是 4.1 早先的参数页 bug 留下的坑，这里在启动前兜一次。
    """
    import pickle as _pk
    import re as _re
    dat = _model_dat(name)
    if dat is None:
        return []
    try:
        data = _pk.loads(dat.read_bytes())
    except Exception:
        return []
    opts = data.get("options") or {}
    fixed = []
    pat = _re.compile(r"^\s*([0-9]+(?:\.[0-9]+)?)\s*[（(].*[）)]\s*$")
    for k, v in list(opts.items()):
        if not isinstance(v, str):
            continue
        m = pat.match(v)
        if not m:
            continue
        num = float(m.group(1))
        opts[k] = int(num) if num.is_integer() and "." not in m.group(1) else num
        fixed.append(k)
    if not fixed:
        return []
    data["options"] = opts
    try:
        dat.with_suffix(".dat.bak").write_bytes(dat.read_bytes())
        tmp = dat.with_suffix(".dat.tmp")
        tmp.write_bytes(_pk.dumps(data))
        os.replace(str(tmp), str(dat))
    except Exception:
        return []
    return fixed




def _webui_post(path: str, payload: dict | None = None, timeout: float = 3.0):
    """给 4.0 训练进程里的 WebUI 发控制指令。没起来就抛异常。"""
    import json as _json
    import urllib.request
    body = dict(payload or {})
    body.setdefault("password", WEBUI_PASSWORD)
    req = urllib.request.Request(
        "http://127.0.0.1:%d%s" % (WEBUI_PORT, path),
        data=_json.dumps(body).encode(), method="POST",
        headers={"Content-Type": "application/json", "User-Agent": "dfl41"})
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(req, timeout=timeout) as r:
        return r.read().decode("utf-8", "replace")


def _kill_train_tree(force_after: float = 0.0):
    """把训练进程连同子进程收掉（psutil 可用时按树杀）。"""
    global _train_proc
    proc = _train_proc
    if proc is None:
        return False
    try:
        if force_after:
            time.sleep(force_after)
        if proc.poll() is None:
            try:
                import psutil
                parent = psutil.Process(proc.pid)
                for child in parent.children(recursive=True):
                    try:
                        child.kill()
                    except Exception:
                        pass
                parent.kill()
            except Exception:
                proc.kill()
    except Exception:
        pass
    try:
        proc.wait(timeout=10)
    except Exception:
        pass
    return True


def install_events(app):
    """把启动/退出钩子挂到 FastAPI 上（原来就是 @app.on_event 那两行装饰器）。

    等价于 app.on_event("startup")(_startup_sweep) / ("shutdown")(_shutdown_cleanup)：
    顺序、函数对象、异常口径都不变，只是钩子跟着它要收拾的 _train_proc 一起住在本模块。
    """
    app.on_event("startup")(_startup_sweep)
    app.on_event("shutdown")(_shutdown_cleanup)
