# -*- coding: utf-8 -*-
"""阶段计时器：_StageMeter / _StageTicks（导出进度那条流水线图的数据源）。

从 app/server/routes_merge.py 按关注点原样切出来的（16741 行的单文件 → 按域分家），
函数体 / 注释 / 日志文案逐字节照抄，只补 import；同包之间的 import 放在文件末尾
（见文末「环上的依赖」），另有一处 __file__ 相对路径下标跟着目录层级改了。
旧行号 → 现文件 的对照表：app/server/backend_merge/MAPPING.md。
"""
from __future__ import annotations

import threading


class _StageTicks:
    """阶段内计时（多线程安全）。profile 表就是从这里出去的。"""

    def __init__(self):
        self._d = {}
        self._lk = threading.Lock()

    def add(self, k: str, dt: float):
        with self._lk:
            self._d[k] = self._d.get(k, 0.0) + float(dt)

    def total(self, k: str) -> float:
        with self._lk:
            return float(self._d.get(k, 0.0))

    def ms(self, k: str, n: int) -> float:
        with self._lk:
            v = self._d.get(k, 0.0)
        return round(1000.0 * v / max(1, int(n)), 2)


# ============================================================
# 流水线实况：交接点计数（导出进度页那条 SVG 图的**唯一**数据源）
# ------------------------------------------------------------
# 进度页从「六个圆点」换成「6 列 lane + 贝塞尔曲线 + 光点」之后，前端要三样东西，
# 少一样图就是假的：
#   ticks     这个交接点一共过了多少帧/张脸（只增）—— 光点按**增量**放，列标题显示「N 帧」
#   inflight  这个交接点**下游那个队列**的当前长度（q.qsize()）—— 列间那句「在飞 N」
#   lanes     这个阶段**真的有**几条 lane（线程数）：合成池 8 线程就是 t0..t7，
#             编码 6 线程就是 e0..e5，dfm/xseg/det 各一个 session 就是 1 条（带 #0）
# 开销：一次 dict 自增 + 一次 time.monotonic()，一把锁（临界区就两行）。tick 里不拼字符串、
# 不进 JSON、不碰队列；只有 status 那一拍（500ms 一次）才 snapshot，qsize() 是 O(1)。
# 为什么不用「每帧发一次请求」：那样 20 帧/秒就是 20 个请求，界面比片还重。
# ============================================================
_STAGE_KEYS = ("frames", "det", "dfm", "xseg", "compose", "encode")
# 每段的耗时取自 _FusedTicks 的同名 key（per-frame ms，和 <成片>.log 里那份同一个口径）
_STAGE_MS_SRC = {"frames": "decode", "det": "detect", "dfm": "dfm",
                 "xseg": "xseg", "compose": "compose", "encode": "encode"}


class _StageMeter:
    """交接点计数器：{key: ticks} + 每段的 lane 名 + 下游队列。

    一个进程里同时只会跑一单导出（起单处有互斥），所以这里是**单例**：
    reset() 开一单，stop() 收一单（收单后保留最后一份快照，界面上那张图不会突然塌掉）。
    """

    def __init__(self):
        self.lock = threading.Lock()
        self.ticks = {k: 0 for k in _STAGE_KEYS}
        self.lanes = {k: [] for k in _STAGE_KEYS}
        self.queues = {}
        self.ms = {}
        self.batch = {}
        self.no_face = 0            # 无脸帧（切脸之后没脸可送去换脸的那些）
        self.armed = False          # 开过单没（没开过 → status 里干脆不出现 stages）
        self.on = False             # 现在还在跑没（收单后不再涨）

    def reset(self, lanes=None) -> None:
        with self.lock:
            self.ticks = {k: 0 for k in _STAGE_KEYS}
            self.lanes = {k: list((lanes or {}).get(k) or []) for k in _STAGE_KEYS}
            self.queues = {}
            self.ms = {}
            self.batch = {}
            self.no_face = 0
            self.armed = True
            self.on = True

    def bind(self, queues: dict) -> None:
        """key -> 下游那个 queue（inflight 就是它的 qsize()）。"""
        with self.lock:
            self.queues = dict(queues or {})

    def stop(self) -> None:
        """收单：不再涨，但保留 lanes/ticks 给界面画最后一帧（inflight 归零）。"""
        with self.lock:
            self.on = False
            self.queues = {}

    def begin(self) -> None:
        """一单开始（分段链 / 项目流水是**串行**跑的）：上一单收过单就清空重来。"""
        with self.lock:
            if not self.on:
                self.ticks = {k: 0 for k in _STAGE_KEYS}
                self.lanes = {k: [] for k in _STAGE_KEYS}
                self.queues = {}
                self.ms = {}
                self.batch = {}
                self.no_face = 0
            self.armed = True
            self.on = True

    def set_lanes(self, lanes: dict) -> None:
        """只补空的 lane：分段链一次只跑一段，lane 数每段才定，但不能把上一段的清掉
        （六段是**同一单**里串行跑的，tick 要一路累上去 —— 图上的「N 帧」才对得上）。"""
        with self.lock:
            for k, v in (lanes or {}).items():
                if v and not self.lanes.get(k):
                    self.lanes[k] = list(v)

    def force_lanes(self, lanes: dict) -> None:
        """覆盖式设置 lane 名（分趟时每趟换一个模型，名字必须跟着走）。"""
        with self.lock:
            for k, v in (lanes or {}).items():
                if v:
                    self.lanes[k] = list(v)

    def tick(self, key: str, n: int = 1) -> None:
        """交接点 +n（n>1 只给「一次回调跨了好几帧」那种地方用，比如 ffmpeg 编码进度）。"""
        with self.lock:
            if not self.on:
                return
            self.ticks[key] = int(self.ticks.get(key) or 0) + max(1, int(n))

    def note_no_face(self, n: int = 1) -> None:
        """一帧「没有脸可换」：切脸之后直接进缓存/编码，换脸/遮罩/合成都不碰它。

        进度页那条旁路线就是按它画的（「切脸 1439 / 换脸 1204」差的那 235 帧，
        不画出来用户以为丢帧）。一次自增，和 tick 同一个量级，不额外扫目录。
        """
        with self.lock:
            if not self.on:
                return
            self.no_face = int(self.no_face or 0) + max(0, int(n))

    def note_ms(self, ms: dict) -> None:
        with self.lock:
            for k, v in (ms or {}).items():
                if v is not None:
                    self.ms[k] = float(v)

    def note_batch(self, batch: dict) -> None:
        with self.lock:
            for k, v in (batch or {}).items():
                if v is not None:
                    self.batch[k] = int(v)

    def snapshot(self) -> dict:
        """status.stages 的那一块（没开过单就是空 dict，前端据此显示「等待流水线…」）。"""
        with self.lock:
            if not self.armed:
                return {}
            out = {}
            for k in _STAGE_KEYS:
                qq = self.queues.get(k)
                inflight = 0
                if qq is not None:
                    try:
                        inflight = int(qq.qsize())
                    except Exception:                          # noqa: BLE001
                        inflight = 0
                row = {"ticks": int(self.ticks.get(k) or 0),
                       "inflight": inflight,
                       "lanes": list(self.lanes.get(k) or [])}
                if k == "det":
                    # 切脸这一格的「无脸帧」计数：图上那条旁路线只在 >0 时出现
                    row["no_face_ticks"] = int(self.no_face or 0)
                if k in self.ms:
                    row["ms"] = round(float(self.ms[k]), 2)
                if k in self.batch:
                    row["batch"] = int(self.batch[k])
                out[k] = row
            return out


_STAGE_METER = _StageMeter()


def _stage_tick(key: str, n: int = 1) -> None:
    """交接点 +1 —— 在**入队/交给下一阶段之前**调用（六个 key 见 _STAGE_KEYS）。

    故意做成模块级函数：埋点处只写一行，读的人一眼知道「这里是交接点」。
    """
    _STAGE_METER.tick(key, n)


class _FusedTicks:
    """逐段累计耗时（和原型同一套口径：每段的总忙时间 / 帧数 = 每帧 ms）。"""

    def __init__(self):
        self.d = {}
        self.lock = threading.Lock()

    def add(self, k, dt):
        with self.lock:
            self.d[k] = float(self.d.get(k) or 0.0) + float(dt)

    def raw(self, k):
        with self.lock:
            return float(self.d.get(k) or 0.0)

    def dump(self, n):
        with self.lock:
            return {k: round(1000.0 * v / max(1, int(n)), 2) for k, v in self.d.items()}
