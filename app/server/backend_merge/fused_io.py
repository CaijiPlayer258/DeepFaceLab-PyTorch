# -*- coding: utf-8 -*-
"""融合流水线的 I/O 侧：跳过账本、raw 写者、ffmpeg 命令、cv2 线程。

从 pipeline_fused.py 搬出（原文件 3178 行，其中 _fused_run 一个函数 1697 行；
这几个是**独立**的类/函数，可以逐字搬，搬完改那一条流水线时少翻 400 行）。
注意：_fused_cfg / _fused_open_frames 没搬 —— 它们依赖 30 多个模块级常量，
搬走等于把这些常量也搬走，反而更乱。
"""
from __future__ import annotations

import os
import subprocess
import threading
import time
from pathlib import Path

import numpy as np

from .pipeline_fused import (
    _BadInput, _FUSED_CV2_DEFAULT, _FUSED_ENV_CV2, _SKIP_TOTAL_MIN, _app_mod, _esel,
    _kill_proc_tree, _vid_log,
)




class _FusedSkipBad:
    """坏帧账本：合成腿记「哪一帧坏、为什么」，写者腿记「连着坏了几帧」。

    两条腿都会写，所以加锁。**按帧号去重**：一帧在合成失败、又在写管道被顶替，
    账上仍然只是一帧（写者不新增记录，只查原因）。
    """

    __slots__ = ("lock", "cap_streak", "cap_total", "why", "streak", "worst", "closing")

    def __init__(self, cap_streak: int, cap_total: int):
        self.lock = threading.Lock()
        self.cap_streak = max(1, int(cap_streak))
        self.cap_total = max(1, int(cap_total))
        self.why = {}                     # 帧号 -> 原因摘要
        self.streak = 0                   # 写者腿看到的「连续坏」
        self.worst = 0
        self.closing = False              # 取消/收尾：从这里往后不算坏帧

    # ---- 合成腿 ----
    def note(self, idx: int, why: str) -> None:
        with self.lock:
            self.why.setdefault(int(idx), str(why)[:160])

    def reason(self, idx: int) -> str:
        with self.lock:
            return self.why.get(int(idx), "")

    def ids(self, limit: int = 2000) -> list:
        with self.lock:
            return sorted(self.why)[:int(limit)]

    def total(self) -> int:
        with self.lock:
            return len(self.why)

    # ---- 写者腿 ----
    def hit(self, idx: int, why: str) -> str:
        """这一帧被顶替了。返回非空 = 判定不可恢复（调用方立刻整单失败）。"""
        with self.lock:
            self.why.setdefault(int(idx), str(why)[:160])
            n = len(self.why)
            if self.closing:
                return ""
            self.streak += 1
            self.worst = max(self.worst, self.streak)
            if self.streak > self.cap_streak:
                return ("连续 %d 帧失败（上限 %d 帧）—— 零星坏帧能顶替，连着坏说明"
                        "这条线病了（模型/素材/编码器），判定不可恢复，整单作废"
                        % (self.streak, self.cap_streak))
            if n > self.cap_total:
                return ("本单已跳过 %d 帧（累计上限 %d 帧 = 5%%，下限 %d 帧）—— 这不是"
                        "零星坏帧，判定不可恢复，整单作废"
                        % (n, self.cap_total, _SKIP_TOTAL_MIN))
            return ""

    def ok(self) -> None:
        with self.lock:
            self.streak = 0

    def stop_skip(self) -> None:
        """取消/收尾：从这里往后不再记坏帧 —— 取消不是坏帧。"""
        with self.lock:
            self.closing = True




def _fused_pipe_cmd(part, video, fps, w, h, audio_ss=None, audio_t=None, limit=None) -> list:
    """rawvideo 版 _vid_cmd：同一个滤镜、同一个编码器、同一段音轨，只是帧从 pipe:0 来。

    三条口径必须和 _vid_cmd（app/server/_media/encoder.py）**一字不差**，否则成片就和老路径
    不一样了：crf16 + yuv420p + scale=out_range=tv；音轨 -c:a copy、**不带 -shortest**
    （带上会按较短的一边截断，帧范围的片子就少一截音）；奇数宽高先 pad 一像素。
    这里复制一份而不是参数化 _vid_cmd，是因为那个吃的是「帧序列文件模式」；改它的时候两边一起看，
    有 _accept 的成片 md5 对拍兜着。

    **format=rgb24 不能省**：rawvideo 声明 bgr24，而 scale 在 bgr24 输入下出来的像素和 PNG
    路径（png 解码器出的是 rgb24）**不是**逐位相同 —— 实测 12/12 帧 md5 全不同。中间插一层
    format=rgb24（ffmpeg 自己换通道，在它自己的线程里，约 1ms/帧）之后，成片文件 md5 与老路径
    **完全相同**（150 帧实测 8af9c0e8f81c / 321466B）。

    **-c:v 那一段不在这里写死**（_common/encoder_sel.py）：默认仍是 libx264 crf16，逐字节
    不变；DFL_FUSED_ENCODER=nvenc-h264|nvenc-hevc 才换。PNG 落盘那条出口
    （_media/encoder._vid_cmd）调的是**同一个** encoder_args() —— 两条路必须一起换，
    否则「两条路成片 md5 相同」这条不变量就破了。

    也**不给 -progress pipe:1**：那条进度走 stdout，没人读就会把 ffmpeg 堵在管道上；
    进度由写者线程逐帧上报（比解析 ffmpeg 的 frame= 更准，而且它本来就知道写到了第几帧）。

    **limit（-frames:v N）必须和 _vid_cmd 一样给**——这一条不是洁癖：实测（30 帧范围导出）
    不加它时音轨会多出**一个** AAC 包（43 → 44，成片 md5 因此不同、时长 1.00 → 1.02s）。
    原因：老路径有 -frames:v，视频一满 30 帧输出就收尾，那个还没来得及 mux 的音频包被丢掉；
    管道这条路没有 -frames:v，输出要等到 stdin EOF，音频已经把第 44 包 mux 进去了。
    加上之后两条路的包序列**逐字节相同**（30 视频包 + 43 音频包，包 md5 全等）。
    """
    pre_aud = []
    if audio_ss is not None:
        pre_aud += ["-ss", "%.6f" % float(audio_ss)]
    if audio_t is not None:
        pre_aud += ["-t", "%.6f" % float(audio_t)]
    cmd = [str(_app_mod._ffmpeg_exe()), "-y", "-hide_banner", "-loglevel", "error",
           "-nostats", "-nostdin",
           "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", "%dx%d" % (int(w), int(h)),
           "-framerate", "%.6g" % float(fps), "-i", "pipe:0"] + pre_aud + \
          ["-i", str(video),
           "-map", "0:v:0", "-map", "1:a?",
           "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2,format=rgb24,scale=out_range=tv"] + \
          _esel.encoder_args(int(w), int(h)) + \
          ["-pix_fmt", "yuv420p",
           "-c:a", "copy"]
    if Path(str(part)).suffix.lower() in (".mp4", ".mov", ".m4v"):
        cmd += ["-movflags", "+faststart"]
    if limit:
        cmd += ["-frames:v", str(int(limit))]
    cmd += [str(part)]
    return cmd




class _FusedRawWriter(threading.Thread):
    """rawvideo 管道的**唯一写者**：按帧号严格递增把帧塞进 ffmpeg 的 stdin。

    为什么必须单写者：编码有 3 条线程（乱序完成），而管道是**有序流** —— 谁先干完谁写，
    成片就串帧，而串帧肉眼几乎看不出来（几十毫秒的错位在 30fps 里像「抖了一下」），
    所以这里宁可断言也不静默：拿到的不是「下一个该写的帧号」就整单失败。

    为什么必须有一条 stderr 排空线程：ffmpeg 的 stderr 是管道，我们不读、它写满就**阻塞**；
    它一阻塞就不读 stdin，于是我们 write 也阻塞 —— 整条流水线抱死在两个管道上（很好复现）。

    背压：write 阻塞 = ffmpeg 比我们慢，那就是真的慢，等它，绝不丢帧（丢帧 = 成片少帧）。
    ffmpeg 提前退出（-vf 写坏 / 编码器不可用 / 磁盘满）：write 抛 BrokenPipeError →
    立刻停整条线，并把 stderr 尾巴交给调用方 → 用户拿到的是「为什么没出来」，不是一句「0 帧」。
    """

    def __init__(self, proc, total, sem, live, live_lock, ticks, stop, on_frame,
                 on_progress, skip=None, frame_bytes=0, on_skip=None):
        super().__init__(name="fused-raw", daemon=True)
        self.proc = proc
        self.total = int(total)
        self.sem = sem
        self.live = live
        self.live_lock = live_lock
        self.ticks = ticks
        self.stop = stop                 # 只用来让「上游」别再喂；写者自己照常把手里写完
        self.on_frame = on_frame
        self.on_progress = on_progress
        self.cv = threading.Condition()
        self.store = {}
        self.expect = 1
        self.closing = False             # 上游说「不会再有帧了」
        # ★ 乱序缓冲**不设上限**：整条流水线的在飞帧数已经被帧预算（sem，默认 30）卡住了，
        #   这里再压一层「只看当前手里有多少帧」的小上限会**死锁**（实测：全片导出跑到第 140 帧
        #   时 store 里是 141..148、而 140 还排在 FIFO 的 q_enc 里，3 条编码线程全卡在 put 上，
        #   谁也取不到 140 → 整条线不动、主线程等线程退出等到天荒地老）。
        #   上界由 sem 保证：一帧从解码进来到写进管道才 release，所以 store ≤ 预算帧数（30×6.2MB）。
        self.depth = 0
        self.n = 0                       # 已写帧数
        self.tail = []
        self.rc = None
        self.err = ""
        self.seconds = 0.0
        self.write_max = 0.0
        self._dead = False
        # ---- 坏帧降级（DFL_FUSED_SKIP_BAD）----
        # skip=None = 关（默认）：run() 里连一个判断都不做，与现状逐字节相同。
        # 开着时：这一帧没图 / 字节数不对 → 用**上一帧真的写出去的那份字节**顶替。
        # 为什么「上一帧」只能在这里算：这是全流程唯一的单写者，帧号严格递增 ——
        # 合成是 n_cmp 条并发线程，谁先完成谁没有次序，"上一帧"在那儿没有定义。
        self.skip = skip
        self.frame_bytes = int(frame_bytes or 0)
        self.on_skip = on_skip
        self.last = None                 # 上一帧真的写进管道的字节（一帧 6.2MB，只留一份）
        self.black = None                # 首帧就坏时的黑帧（懒分配，字节数与一帧严格相同）
        self.skipped = 0
        self.skip_abort = ""             # 非空 = 这一单是被「坏帧降级判死」的（错误文案要分开）

    # 上游：编码线程（可并发）
    def put(self, job, buf) -> bool:
        seq = int(job.n)
        with self.cv:
            while (self.depth > 0 and len(self.store) >= self.depth
                   and not self._dead and not self.closing):
                if self.stop.is_set():
                    return False
                self.cv.wait(0.05)
            if self._dead or self.stop.is_set():
                return False
            if seq in self.store:
                self.err = "同一帧被写了两遍（第 %d 帧）—— 帧号重复" % seq
                self._dead = True
                self.stop.set()
                return False
            if seq < self.expect:
                self.err = "第 %d 帧已经写出去了，又收到一次（帧号回退）" % seq
                self._dead = True
                self.stop.set()
                return False
            self.store[seq] = (job, buf)
            self.cv.notify_all()
            return True

    def _black(self):
        """一帧黑帧（字节数与一帧严格相同）——首帧就坏 / PNG 路顶替时用。

        必须和正常帧**同字节数**：管道是定长流，少一个字节后面就全错位。
        """
        n = int(self.frame_bytes or 0)
        if n <= 0:
            return None
        if self.black is None:
            self.black = memoryview(np.zeros(n, np.uint8))
        return self.black

    def _slot(self, job, buf):
        """这一帧到底写什么字节：好帧就是它自己，坏帧用**上一帧**顶替（首帧用黑帧）。

        返回 (要写的字节, 判定不可恢复的原因)。第二个非空 = 调用方立刻整单失败。
        **默认口径（skip=None）原样返回** —— 关掉开关时这条路径上一次都不会进。
        """
        if self.skip is None:
            return buf, ""
        why = ""
        # 账本的键一律用**源视频绝对帧号**（job.idx）：帧范围导出时 job.n 是段内序号，
        # 拿它当键会和合成腿（记 job.idx）对不上，同一帧被数两次。
        if buf is None:
            why = self.skip.reason(int(job.idx)) or "合成阶段失败（没给这一帧的图）"
        elif self.frame_bytes and len(buf) != self.frame_bytes:
            # 定长流里字节数不对 = 整条流从这一帧起错位，必须按坏帧处理（不是照样写进去）
            why = "这一帧字节数不对（%d，应为 %d）" % (len(buf), self.frame_bytes)
        if not why:
            self.last = buf                 # 只有真的写出去的帧才当「上一帧」
            self.skip.ok()
            return buf, ""
        fatal = self.skip.hit(int(job.idx), why)
        first = self.last is None
        if first:
            self.last = self._black()
            if self.last is None:
                return buf, ("坏帧顶替失败：拿不到一帧的字节数（probe 宽高缺失）"
                             "—— 这一帧没图也顶不了")
        self.skipped += 1
        _vid_log("快速导出：第 %d 帧坏 → 跳过（%s）；这一格用%s顶替（本单已跳过 %d 帧）"
                 % (int(job.idx), why, "黑帧（首帧没有上一帧）" if first else "上一帧",
                    self.skipped))
        if self.on_skip is not None:
            try:
                self.on_skip(int(job.n), why)
            except Exception:                              # noqa: BLE001
                pass
        return self.last, fatal

    def start(self):                     # noqa: D102 - 先起 stderr 排空线程，再起写者
        threading.Thread(target=self._drain, name="fused-raw-err", daemon=True).start()
        super().start()

    def _drain(self):
        try:
            for ln in iter(self.proc.stderr.readline, b""):
                s = ln.decode("utf-8", "replace").rstrip()
                if s:
                    self.tail.append(s)
                    if len(self.tail) > 60:
                        del self.tail[:20]
        except Exception:                                  # noqa: BLE001
            pass

    def run(self):
        t0 = time.perf_counter()
        try:
            while True:
                with self.cv:
                    while self.expect not in self.store and not self.closing:
                        self.cv.wait(0.1)
                    item = self.store.pop(self.expect, None)
                    if item is None:
                        # closing 且下一帧不在手里：收尾（正常 = 全写完；异常 = 中间有洞）
                        if not self.store:
                            return
                        self.err = self.err or (
                            "帧序有洞：等第 %d 帧，手里最小的是第 %d 帧"
                            % (self.expect, min(self.store)))
                        self._dead = True
                        return
                job, buf = item
                if int(job.n) != self.expect:              # 双保险：写之前断言 seq 递增
                    self.err = "帧序错位：期望第 %d 帧，拿到第 %d 帧" % (self.expect, int(job.n))
                    self._dead = True
                    self.stop.set()
                    return
                t = time.perf_counter()
                buf, _fatal = self._slot(job, buf)
                if _fatal:
                    self.err = "坏帧降级：" + _fatal
                    self.skip_abort = _fatal
                    self._dead = True
                    self.stop.set()
                    return
                try:
                    self.proc.stdin.write(buf)
                except (BrokenPipeError, ConnectionResetError, ValueError, OSError) as e:
                    self.err = ("ffmpeg 提前退出（第 %d 帧写不进去）：%s"
                                % (self.expect, str(e)[:160]))
                    self._dead = True
                    self.stop.set()
                    return
                dt = time.perf_counter() - t
                self.write_max = max(self.write_max, dt)
                self.ticks.add("write", dt)
                self.expect += 1
                self.n += 1
                try:
                    self.sem.release()
                except ValueError:
                    pass
                with self.live_lock:
                    self.live["n"] -= 1
                self.on_frame(job)
                self.on_progress(self.n)
        except BaseException as e:                              # noqa: BLE001
            # ★ 写者猝死必须让调用方看见。以前这里只有 finally（置 _dead），
            #   而 pipeline_fused.py:2557 是靠 `if raw_w.err` 判「管道坏了」的 ——
            #   err 永远是空 → 判不出来 → ffmpeg 拿到 EOF 出一支短片，
            #   再 os.replace 成成片名、状态写「完成」。正是本项目自己说的
            #   「静默产出坏片比失败更糟」。
            if not self.err:
                self.err = "%s: %s" % (e.__class__.__name__, str(e)[:300])
            raise
        finally:
            self.seconds = time.perf_counter() - t0
            self._dead = True

    def finish(self, force: bool = False):
        """收尾：让写者把手里写完 → 关 stdin（ffmpeg 见 EOF 才会把最后一批编完）→ 等它退出。

        force=True（失败/取消）：不再等它排空，直接杀掉 —— 半截成片由调用方删。
        """
        with self.cv:
            self.closing = True
            self.cv.notify_all()
        self.join(timeout=5.0 if force else 300.0)
        stuck = self.is_alive()
        # 关 stdin = 给 ffmpeg 一个 EOF：它才会把 lookahead 里剩下的帧编完（排空）。
        # 写者要是还卡在 write 里，这一下会让它拿到「write to closed file」并退出 ——
        # 比在这里干等一个可能永远不来的帧安全。
        try:
            self.proc.stdin.close()
        except Exception:                                  # noqa: BLE001
            pass
        if force or stuck:
            try:
                _kill_proc_tree(self.proc)
            except Exception:                              # noqa: BLE001
                pass
        try:
            self.rc = int(self.proc.wait(timeout=30 if (force or stuck) else 600))
        except Exception:                                  # noqa: BLE001
            try:
                _kill_proc_tree(self.proc)
            except Exception:                              # noqa: BLE001
                pass
            self.rc = -1
        return self.rc

    def tail_text(self, n: int = 12) -> str:
        return "\n".join((self.tail or [])[-int(n):])




def _fused_cv2_threads(cfg: dict) -> int:
    """fused 这条线给 OpenCV 留几条内部线程（**0 = 这一单不去动它**，用 OpenCV 自己的默认）。

    优先级：body.merge_cv2_threads / DFL_MERGE_CV2_THREADS（显式给了就用，还是那个老旋钮）
    → DFL_CV2_THREADS → 不动（默认）。

    默认为什么是「不动」而不是某个数字：三档都在真机上量过，**孤立基准和整条线给出
    相反的答案**（下面两组数是同一台机、同一份素材）：
      · 孤立 compose 基准（8 线程打满 · 真实 1080p 帧 · mask_mode8/seamless/mt · 5 轮中位）：
          cv2=4  22.7 帧/秒容量 · 8.03 核      cv2=1  25.3 帧/秒 · 7.45 核
          cv2=0（不动/OpenCV 默认 32） 26.6 帧/秒 · 7.38 核
        → 「少给 cv2 线程 = 同样的活少烧核、容量更高」在打满时成立。
      · 整条 fused（150 帧 416 tinymog，4 配置 × 4 轮交错中位）：
          cv2=4  18.62 帧/秒 · compose 107.3ms/帧
          cv2=1  18.18 帧/秒 · compose 125.8ms/帧   ← 整条**更慢**，compose 墙钟 +18ms
        → 生产 compose 只用到 ~25%，那 12% 容量换不成帧/秒；而 cv2 池退出并行会让
          每个合成任务变长（seamlessClone 300px 15.0 → 15.6ms、warp 0.94 → 3.04ms 这些
          单项差异在 25% 占用下被分摊掉，收益收不出来）。
    **结论：局部更快 ≠ 整条更快。** 默认保持「不动」（= 谁都不设时 OpenCV 自己的默认），
    要 A/B 就显式 DFL_CV2_THREADS=1 / 4。
    """
    raw = cfg.get("merge_cv2_threads")
    try:
        want = int(raw or 0)
    except (TypeError, ValueError):
        want = 0
    if want > 0:
        return min(32, want)
    env = str(os.environ.get(_FUSED_ENV_CV2) or "").strip()
    if env:
        try:
            return max(0, min(32, int(env, 10)))
        except ValueError:
            pass
    return max(0, min(32, int(_FUSED_CV2_DEFAULT)))
