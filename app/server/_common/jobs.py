# -*- coding: utf-8 -*-
"""任务注册表 + 进程内共享运行时状态（从 app/server/app.py 原样切出）。

装的是「谁在跑」这一层，不含业务算法：

  Job / JOB / STAGE_FAILED   提取任务（抽帧 → 对齐 → 遮罩）的进度与失败收口
  JOBS                       多任务表（一次性操作按 id 轮询）
  _LAST_EXTRACT              最近一次 _align_stream 的实测数字（自检脚本读它，不解析 stdout）
  _merge_cache               「当前打开的视频/项目」缓存
  _merge_prep / _merge_prep_lock  「视频 → 项目」准备的进度
  _job_fail / _job_status    失败与快照的唯一口径

函数体 / 注释 / 日志文案逐字节照抄 app.py，只补 import（threading / time / sys / traceback）。
_LAST_EXTRACT 原来只是 app 模块的全局，现在归本模块所有：_align_stream 那边写的是
_jobs._LAST_EXTRACT = {...}，Job.fail 读的也是这份，双方仍是同一个对象。
"""
from __future__ import annotations

import sys
import threading
import time
import traceback


# 合成工作室「当前打开的视频/项目」：预览取帧、抽帧准备、导出都要读它
_merge_cache = {"path": None, "frames": 0, "fps": 30.0, "w": 0, "h": 0}
_merge_prep = {"running": False, "ok": None, "error": "", "pct": 0, "msg": "",
               "stage": "", "project": "", "frames": 0, "aligned": 0, "video": ""}
_merge_prep_lock = threading.Lock()
JOBS: dict = {}
# ---------------- 演示用任务状态（M0 占位，M1 换成真实提取器） ----------------
STAGE_FAILED = -1        # 失败收尾（0 抽帧 / 1 对齐 / 2 遮罩 / 3 完成 之外的第四种）
class Job:
    def __init__(self):
        self.lock = threading.Lock()
        self.reset()

    def reset(self):
        with self.lock:
            self.running = False
            self.pct = 0
            self.fps = 0.0
            self.faces = 0.0
            self.faces_new = 0.0   # 本次运行新落盘的脸（不含 base_faces），人脸速度的分子
            self.done = 0
            self.total = 0
            self.msg = "就绪"
            self.stage = 0          # 0 抽帧 / 1 对齐 / 2 应用遮罩 / 3 完成 / -1 失败
            self.t0 = 0.0
            self.ok = None          # None=没跑/在跑 · True=成功结束 · False=失败
            self.error = ""         # 失败原因（ok=False 时必定非空），前端据此出红条

    def fail(self, exc, where: str = "") -> str:
        """把异常落成「失败」态 —— 不许表现成「完成」。

        事故复盘：给 TinyMogDet 多传了一个参数 → TypeError。异常抛在抽取线程里
        没人接，JOB 停在 pct<100 的空转态；前端把「没在跑」当成「跑完了」，
        界面上是「提取完成 0 帧」。所以失败必须一次写完四件事：
            running=False / ok=False / error 非空（msg 同步成同一句）
            stage=STAGE_FAILED —— 前端看到它就不再显示「提取完成」
        """
        name = type(exc).__name__
        # PipelineError 只是外壳，报里面那个真异常才有用（类型名 + 200 字说明）
        inner = getattr(exc, "original", None)
        if inner is not None:
            name, exc = type(inner).__name__, inner
        detail = " ".join(str(exc).split())[:200] or "（没有异常说明）"
        text = "提取失败：%s: %s" % (name, detail)
        with self.lock:
            self.running = False
            self.ok = False
            self.error = text
            self.msg = text
            self.stage = STAGE_FAILED
            self.pct = min(int(self.pct or 0), 99)
        # 自检脚本读 _LAST_EXTRACT，别只把失败留在界面上
        try:
            _LAST_EXTRACT.update({"ok": False, "error": text, "where": where})
        except Exception:
            pass
        print("[extract] %s%s" % (text, ("  @ " + where) if where else ""), flush=True)
        return text

    def begin(self, stage: int, total: int, msg: str = ""):
        """进入一个新阶段：重置计时与计数，前端据此算速度和剩余时间。"""
        with self.lock:
            self.stage = stage
            self.t0 = time.time()
            self.done = 0
            self.total = int(total)
            self.pct = 0
            if msg:
                self.msg = msg

    def snapshot(self) -> dict:
        with self.lock:
            el = (time.time() - self.t0) if self.t0 else 0.0
            rate = (self.done / el) if (el > 0.3 and self.done) else 0.0
            left = max(0, self.total - self.done)
            eta = (left / rate) if (rate > 0 and left > 0) else 0.0
            # 「人脸速度」和「帧速度」同一个口径：分子都是**本次运行**的计数
            # （帧 = done / faces 用 faces_new，续跑时不含目录里原有的 base_faces），
            # 分母都是 el（== 前端拿到的 elapsed）。以前前端自己拿累计 faces / elapsed
            # 算，续跑会虚高；而真正的毛病是 faces 在跑的时候压根没更新过。
            face_rate = (self.faces_new / el) if (el > 0.3 and self.faces_new) else 0.0
            return {
                "running": self.running, "pct": self.pct, "fps": self.fps,
                "faces": self.faces, "faces_new": self.faces_new,
                "face_rate": round(face_rate, 2),
                "done": self.done, "total": self.total,
                "msg": self.msg, "stage": self.stage,
                "ok": self.ok, "error": self.error,
                "rate": round(rate, 2), "eta": round(eta, 1), "elapsed": round(el, 1),
            }
JOB = Job()
# 最近一次 _align_stream 的实测数字（模式/耗时/队列深度/取消）。自检脚本读它，
# 不解析 stdout —— 打印格式可以随便改，数字口径不变。
_LAST_EXTRACT = {}
def _job_fail(where: str, exc=None) -> str:
    """统一失败收口：JOB 落失败态 + 完整 traceback 进日志。

    以前异常只出现在 stderr（GUI 启动时根本没人看得见），界面拿到的却是「完成」。
    """
    if exc is None:
        exc = sys.exc_info()[1]
    text = JOB.fail(exc, where=where)
    tb = traceback.format_exc()
    if tb and "NoneType: None" not in tb:
        print(tb.rstrip(), flush=True)
    return text


def _job_status() -> dict:
    """抽取任务的一次性状态（前端轮询 / 自检脚本都读这一份）。"""
    snap = JOB.snapshot()
    snap["ok"] = JOB.ok
    snap["error"] = JOB.error
    return snap
def _demo_worker():
    total = 342
    with JOB.lock:
        JOB.running = True
        JOB.total = total
    for i in range(total + 1):
        if not JOB.running:
            break
        with JOB.lock:
            JOB.done = i
            JOB.pct = int(i * 100 / total)
            JOB.fps = 34 + (i % 7) * 0.9
            JOB.faces = 9 + (i % 5) * 0.2
            JOB.faces_new = JOB.faces
            JOB.msg = "处理中 %d / %d" % (i, total)
        time.sleep(0.02)
    with JOB.lock:
        JOB.running = False
        JOB.pct = 100
        JOB.msg = "完成"
