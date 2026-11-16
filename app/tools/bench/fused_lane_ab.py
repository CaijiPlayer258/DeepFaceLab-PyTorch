# -*- coding: utf-8 -*-
r"""GPU 车道数 A/B：DFL_FUSED_GPU_LANES = 1 / 2 / 3（真 _fused_run 端到端，不是原型）。

问题：GPU 段（dfm + XSeg）和 host 段（攒批数组 + D2H + u8 量化）在**同一条线程**里严格交替，
所以每帧成本 = GPU + host。开第二条车道（第二条线程 + 第二个独立 dfm 会话）能不能让
「车道 A 的 host 段」和「车道 B 的 GPU 段」重叠，把每帧从 add 变成 max？

跑法（仓库根，一格一个**子进程**：显存、会话、缓存都不跨臂）：

    runtime\server\Scripts\python.exe app\tools\bench\fused_lane_ab.py
    ... --frames 150 --rounds 3 --arms 1 2 3
    ... --sample            # 同时用 nvidia-smi 采 SM%/显存（只给「利用率」那一列用）
    ... --parity            # 只跑逐位对拍（两份会话 + 两条线程并发）

口径（和 experiments/*.md 那几份一致）：
  · 同进程交替做不到（一条车道要多占 ~1GB 显存，3 臂叠在一个进程里就是 OOM），
    所以**臂 = 子进程**，轮次交替（r1: 1→2→3，r2: 3→2→1，r3: 2→1→3），中位取整。
  · fps / 分项 ms 一律取 `_fused_run` 自己的账（per_stage_ms，即 <成片>.log 里那行），
    不额外埋探针；--sample 那一列只采外部 nvidia-smi（100ms），不参与 fps 结论。
  · 成片 md5 在对拍那一段用 DFL_FUSED_ENCODER=x264 钉死（换编码器必然换 md5）。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
APP = REPO / 'app'
PY = REPO / 'runtime' / 'server' / 'Scripts' / 'python.exe'
WS = REPO / 'workspace'
MODEL = '阿証証証証-416'
# 和 experiments/dec_lane.md、tools/accept/det_align_src_accept.py 同一份 B 配置：
# seamless + mt + **mask6**（不要 dst 遮罩 = 每张脸只跑一次 xseg）。
# mask8 会让每张脸跑两次遮罩、合成也贵一截 —— 那组的墙是合成，量不出车道的事。
SETTINGS = {'blend': 'seamless', 'ct_mode': 'mt', 'mask_mode': 6,
            'mask_erode': 0, 'mask_blur': 4}
OUTDIR = Path(os.environ.get('TEMP', '.')) / 'dfl_lane_ab'
RESULT_TAG = '##LANE_RESULT##'


def say(s=''):
    print(s, flush=True)


def _boot():
    sys.path.insert(0, str(REPO))
    sys.path.insert(0, str(APP))
    for _s in (sys.stdout, sys.stderr):
        try:
            _s.reconfigure(encoding='utf-8', errors='replace')
        except Exception:                                  # noqa: BLE001
            pass


def md5_of(p: Path) -> str:
    h = hashlib.md5()
    with open(p, 'rb') as fh:
        for blk in iter(lambda: fh.read(1 << 20), b''):
            h.update(blk)
    return h.hexdigest()


# ---------------------------------------------------------------- 采样（可选）
class Sampler(threading.Thread):
    """nvidia-smi -lms 采样（只给「SM 利用率 / 显存峰值」两列）。"""

    def __init__(self, period_ms=100):
        super().__init__(daemon=True)
        self.period = max(20, int(period_ms))
        self.rows = []
        self.t0 = None
        self.stop_flag = threading.Event()

    def run(self):
        sub = subprocess.Popen(
            ['nvidia-smi', '--query-gpu=utilization.gpu,memory.used',
             '--format=csv,noheader,nounits', '-lms', str(self.period)],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        self.t0 = time.time()
        for line in iter(sub.stdout.readline, b''):
            if self.stop_flag.is_set():
                break
            parts = [x.strip() for x in line.decode('utf-8', 'replace').split(',')]
            if len(parts) >= 2:
                try:
                    self.rows.append((time.time() - self.t0, float(parts[0]), float(parts[1])))
                except ValueError:
                    pass
        try:
            sub.kill()
        except Exception:                              # noqa: BLE001
            pass

    def stop(self):
        self.stop_flag.set()
        self.join(timeout=3)

    def summary(self, warm_s=2.0):
        """稳态 = 丢掉开头 warm_s 秒（会话加载/NVML 滞后都在那段）。"""
        rows = [r for r in self.rows if r[0] >= warm_s] or self.rows
        if not rows:
            return {}
        u = sorted(r[1] for r in rows)
        m = [r[2] for r in rows]
        return {'n': len(rows), 'util_mean': round(statistics.mean([r[1] for r in rows]), 1),
                'util_p50': round(u[len(u) // 2], 1),
                'util_max': round(max(u), 1),
                'mem_peak_mb': round(max(m), 1),
                'mem_mean_mb': round(statistics.mean(m), 1),
                'idle_lt10_pct': round(100.0 * sum(1 for v in u if v < 10) / len(u), 1)}


# ---------------------------------------------------------------- 子进程：一次导出
def run_child(a):
    _boot()
    if a.fault_every:
        # 排障用：每 N 秒把所有线程的栈打到 stderr（死锁只有看栈才查得动）
        import faulthandler
        faulthandler.enable()
        faulthandler.dump_traceback_later(int(a.fault_every), repeat=True)
    import torch
    from server import app as app_mod
    from server.backend_merge import pipeline_fused as pf
    from server.backend_merge import pipeline_video as pv

    clip = Path(a.clip)
    outdir = Path(a.outdir) / ('lane%d_r%d' % (int(a.lanes), int(a.round)))
    outdir.mkdir(parents=True, exist_ok=True)
    out = outdir / ('lane%d.mp4' % int(a.lanes))
    if out.exists():
        out.unlink()
    body = {'video': str(clip), 'model': MODEL, 'settings': dict(SETTINGS),
            'output_dir': str(outdir), 'frame_budget': 30,
            'start_frame': int(a.start), 'end_frame': int(a.start) + int(a.frames) - 1}
    t_imp = time.time()
    free0 = total0 = 0
    try:
        free0, total0 = (int(x) // (1 << 20) for x in torch.cuda.mem_get_info())
    except Exception:                                  # noqa: BLE001
        pass
    smp = None
    try:
        cfg = pf._fused_cfg(body)
        pv._vid_open_log(out.with_suffix('.log'))
        t_load = time.time()
        ent = pf._dfm_session(str(WS / 'model' / (MODEL + '.dfm')))   # 预热/计时只为了让 t_load 有定义
        load_s = round(time.time() - t_load, 2)
        if a.sample:
            smp = Sampler(100)
            smp.start()
        t0 = time.time()
        res = pf._fused_run(cfg)
        wall = round(time.time() - t0, 2)
        if smp is not None:
            smp.stop()
        alloc = reserved = 0.0
        try:
            alloc = round(torch.cuda.max_memory_allocated() / 1048576.0, 1)
            reserved = round(torch.cuda.max_memory_reserved() / 1048576.0, 1)
        except Exception:                              # noqa: BLE001
            pass
        rec = {'ok': True, 'lanes': int(a.lanes), 'round': int(a.round),
               'frames': int(res.get('frames') or 0), 'fps': float(res.get('fps') or 0.0),
               'wall_s': wall, 'gpu_lanes': int(res.get('gpu_lanes') or 0),
               'gpu_lanes_want': int(res.get('gpu_lanes_want') or 0),
               'per_stage_ms': res.get('per_stage_ms') or {},
               'stage_util_pct': res.get('stage_util_pct') or {},
               'batch_actual': res.get('batch_actual') or {},
               'peak_live': res.get('peak_live'), 'frame_budget': res.get('frame_budget'),
               'sink': res.get('sink'), 'out': str(res.get('out') or out),
               'md5': (md5_of(Path(res.get('out') or out))
                       if Path(res.get('out') or out).is_file() else ''),
               'session_load_s': load_s, 'import_s': round(t_load - t_imp, 2),
               'vram_free0_mb': free0, 'vram_total_mb': total0,
               'torch_max_alloc_mb': alloc, 'torch_max_reserved_mb': reserved,
               'cpu_count': os.cpu_count(),
               'model': MODEL, 'encoder': str(os.environ.get('DFL_FUSED_ENCODER') or 'auto'),
               'sessions': (ent[6] if len(ent) > 6 else '?')}
        if smp is not None:
            rec['gpu'] = smp.summary(warm_s=float(a.sample_warm))
        return rec
    except BaseException as e:                             # noqa: BLE001
        return {'ok': False, 'lanes': int(a.lanes), 'round': int(a.round),
                'error': '%s: %s' % (type(e).__name__, str(e)[:400]),
                'trace': traceback.format_exc()[-1200:]}
    finally:
        try:
            pv._vid_close_log()
        except Exception:                              # noqa: BLE001
            pass
        if smp is not None:
            try:
                smp.stop()
            except Exception:                          # noqa: BLE001
                pass


# ---------------------------------------------------------------- 逐位对拍（两次会话/两条线程）
def run_parity(a):
    _boot()
    import numpy as np
    from server.backend_merge.pipeline_stages import _dfm_run_batch, _dfm_session

    dfm = str(WS / 'model' / (MODEL + '.dfm'))
    e0 = _dfm_session(dfm)
    t0 = time.time()
    e1 = _dfm_session(dfm, lane=1)
    say('两份会话就绪（第二份 %.1f s）· #0=%s · #1=%s'
        % (time.time() - t0, e0[6], e1[6]))
    sz = int(e0[2])
    rng = np.random.default_rng(20261116)
    rows = []
    for b in (1, 2, 4, 8):
        x = rng.random((b, sz, sz, 3), dtype=np.float32)
        a0 = _dfm_run_batch(e0, x)
        a1 = _dfm_run_batch(e1, x)
        a0b = _dfm_run_batch(e0, x)          # 控制组：同一份会话跑两次（确定性）
        for tag, ya, yb in (('lane0_vs_self', a0, a0b), ('lane0_vs_lane1', a0, a1)):
            for k, name in enumerate(('pred', 'prd', 'dst')):
                if ya[0][k] is None or yb[0][k] is None:
                    continue
                d = [np.abs(np.asarray(u[k], np.float32) - np.asarray(v[k], np.float32))
                     for u, v in zip(ya, yb)]
                mx = float(max(float(x2.max()) for x2 in d)) if d else 0.0
                same = sum(1 for x2 in d if not x2.any())
                rows.append({'batch': b, 'cmp': tag, 'tensor': name, 'max': mx,
                             'bit_same': int(same), 'n': len(d)})
    # 并发：两条线程各跑自己那份会话，输出和「单独跑」的逐位比
    job = []
    for b in (1, 2, 4, 8):
        x = rng.random((b, sz, sz, 3), dtype=np.float32)
        job.append((b, x))
    solo0 = [_dfm_run_batch(e0, x) for _b, x in job]
    solo1 = [_dfm_run_batch(e1, x) for _b, x in job]
    got = {'0': [], '1': []}
    err = []

    def _work(tag, ent, store):
        try:
            for b, x in job:
                store.append(_dfm_run_batch(ent, x))
        except BaseException as e:                     # noqa: BLE001
            err.append('%s: %r' % (tag, e))

    th0 = threading.Thread(target=_work, args=('0', e0, got['0']), daemon=True)
    th1 = threading.Thread(target=_work, args=('1', e1, got['1']), daemon=True)
    t0 = time.time()
    th0.start(); th1.start(); th0.join(); th1.join()
    say('并发跑完 %.2f s · 错误 %s' % (time.time() - t0, err or '无'))
    for tag, base, mine in (('#0', solo0, got['0']), ('#1', solo1, got['1'])):
        for i, (b, _x) in enumerate(job):
            for k, name in enumerate(('pred', 'prd', 'dst')):
                if base[i][0][k] is None or mine[i][0][k] is None:
                    continue
                d = [float(np.abs(np.asarray(u[k], np.float32) - np.asarray(v[k], np.float32)).max())
                     for u, v in zip(base[i], mine[i])]
                rows.append({'batch': b, 'cmp': tag + '_solo_vs_concurrent', 'tensor': name,
                             'max': float(max(d)) if d else 0.0,
                             'bit_same': sum(1 for v in d if v == 0.0), 'n': len(d)})
    bad = [r for r in rows if r['max'] > 0.0]
    say('')
    say('%-22s %-6s %-6s %8s %8s %6s' % ('比较', 'batch', 'tensor', 'max', '逐位同', 'n'))
    for r in rows:
        say('%-22s %-6d %-6s %8.0f %8d %6d'
            % (r['cmp'], r['batch'], r['tensor'], r['max'], r['bit_same'], r['n']))
    say('')
    say('结论：%s' % ('全部 max=0（两条车道逐位相同，会话之间没有状态泄漏）' if not bad
                     else ('有 %d 格 max>0 —— 见上表' % len(bad))))
    return {'rows': rows, 'bad': len(bad), 'lanes_ent': [e0[6], e1[6]]}


# ---------------------------------------------------------------- 父进程：轮流跑各臂
def spawn(a, lanes, rnd):
    cmd = [str(PY), str(Path(__file__).resolve()), '--child',
           '--clip', str(a.clip), '--outdir', str(a.outdir), '--frames', str(a.frames),
           '--start', str(a.start), '--lanes', str(lanes), '--round', str(rnd)]
    if a.fault_every:
        cmd += ['--fault-every', str(a.fault_every)]
    if a.dump_frames:
        cmd += ['--dump-frames', str(a.dump_frames)]
    if a.sample:
        cmd.append('--sample')
        cmd += ['--sample-warm', str(a.sample_warm)]
    env = dict(os.environ)
    env['DFL_FUSED_GPU_LANES'] = str(lanes)
    env['PYTHONIOENCODING'] = 'utf-8'
    for k in ('DFL_COMPOSE_GPU', 'DFL_FUSED_BUDGET', 'DFL_FUSED_VRAM_FREE_MB',
              'DFL_ALLOW_CPU_FALLBACK', 'DFL_FUSED_BATCH_WAIT_MS'):
        env.pop(k, None)
    if a.dump_frames:
        env['DFL_FUSED_DUMP'] = str(a.dump_frames)
    else:
        env.pop('DFL_FUSED_DUMP', None)
    if a.batch:
        # 钉批：攒批档位（1..8）是**按阶梯实测选的**，同一份代码两轮能选出 1 也能选出 8
        # （每张 19.5 vs 24.4ms），不钉住它，臂之间的差就不是车道、是批大小。
        env['DFL_FUSED_BATCH'] = str(int(a.batch))
    else:
        env.pop('DFL_FUSED_BATCH', None)
    if a.encoder:
        env['DFL_FUSED_ENCODER'] = str(a.encoder)
    t0 = time.time()
    pr = subprocess.run(cmd, cwd=str(REPO), env=env, capture_output=True)
    txt = (pr.stdout or b'').decode('utf-8', 'replace')
    out = {'lanes': lanes, 'round': rnd, 'ok': False,
           'error': '子进程退出码 %s' % pr.returncode}
    for line in txt.splitlines():
        if line.startswith(RESULT_TAG):
            try:
                out = json.loads(line[len(RESULT_TAG):])
            except Exception:                          # noqa: BLE001
                pass
    out['child_s'] = round(time.time() - t0, 2)
    if not out.get('ok'):
        out['stderr_tail'] = (pr.stderr or b'').decode('utf-8', 'replace')[-800:]
    return out


def med(vals):
    vals = [v for v in vals if v is not None]
    return round(statistics.median(vals), 2) if vals else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--clip', default=str(WS / '9月29日 (5).mp4'))
    ap.add_argument('--frames', type=int, default=150)
    ap.add_argument('--start', type=int, default=200)
    ap.add_argument('--rounds', type=int, default=3)
    ap.add_argument('--arms', nargs='*', type=int, default=[1, 2, 3])
    ap.add_argument('--encoder', default='auto')
    ap.add_argument('--batch', type=int, default=1,
                    help='钉住 DFL_FUSED_BATCH（0=不钉，让 _target_for 自己选）')
    ap.add_argument('--dump-frames', default='',
                    help='DFL_FUSED_DUMP：把这些帧的逐脸 pred/遮罩/合成落盘（对拍用）')
    ap.add_argument('--sample', action='store_true')
    ap.add_argument('--sample-warm', type=float, default=2.0)
    ap.add_argument('--parity', action='store_true')
    ap.add_argument('--child', action='store_true')
    ap.add_argument('--lanes', type=int, default=1)
    ap.add_argument('--round', type=int, default=0)
    ap.add_argument('--fault-every', type=int, default=0,
                    help='排障：每 N 秒 dump 全部线程栈（0=关）')
    ap.add_argument('--outdir', default=str(OUTDIR))
    ap.add_argument('--out', default=str(OUTDIR / 'lane_ab.json'))
    a = ap.parse_args()
    Path(a.outdir).mkdir(parents=True, exist_ok=True)

    if a.child:
        rec = run_child(a)
        say(RESULT_TAG + json.dumps(rec, ensure_ascii=False))
        return 0
    if a.parity:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        res = run_parity(a)
        Path(str(Path(a.out).with_suffix('.parity.json'))).write_text(
            json.dumps(res, indent=2, ensure_ascii=False), encoding='utf-8')
        return 0 if not res.get('bad') else 3

    arms = list(a.arms)
    recs = []
    for rnd in range(1, int(a.rounds) + 1):
        order = arms if rnd % 2 else list(reversed(arms))
        if rnd % 3 == 0:
            order = arms[1:] + arms[:1]
        for lanes in order:
            say('')
            say('=== 车道 %d · 第 %d 轮 · %d 帧%s ==='
                % (lanes, rnd, a.frames, ' · 采样' if a.sample else ''))
            rec = spawn(a, lanes, rnd)
            recs.append(rec)
            if rec.get('ok'):
                ps = rec.get('per_stage_ms') or {}
                say('  fps=%.2f  wall=%.2fs  dfm=%s u8=%s xseg=%s  GPU段=%s ms  显存peak=%s MB  md5=%s'
                    % (rec.get('fps') or 0, rec.get('wall_s') or 0,
                       ps.get('dfm'), ps.get('u8'), ps.get('xseg'),
                       round(sum(float(ps.get(k) or 0) for k in ('dfm', 'u8', 'xseg')), 1),
                       (rec.get('gpu') or {}).get('mem_peak_mb', rec.get('torch_max_alloc_mb')),
                       (rec.get('md5') or '')[:12]))
            else:
                say('  失败：%s' % str(rec.get('error'))[:300])
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(recs, indent=2, ensure_ascii=False), encoding='utf-8')

    say('')
    say('%-6s %8s %8s %8s %8s %8s %8s %8s %10s'
        % ('车道', 'fps', 'dfm', 'u8', 'xseg', 'gpu_wait', 'SM%', '显存MB', '成片md5'))
    for lanes in arms:
        g = [r for r in recs if r.get('ok') and int(r.get('lanes') or 0) == lanes]
        if not g:
            say('%-6d %s' % (lanes, '全部失败'))
            continue
        say('%-6d %8s %8s %8s %8s %8s %8s %8s %10s'
            % (lanes,
               med([r.get('fps') for r in g]),
               med([(r.get('per_stage_ms') or {}).get('dfm') for r in g]),
               med([(r.get('per_stage_ms') or {}).get('u8') for r in g]),
               med([(r.get('per_stage_ms') or {}).get('xseg') for r in g]),
               med([(r.get('per_stage_ms') or {}).get('gpu_wait') for r in g]),
               med([(r.get('gpu') or {}).get('util_p50') for r in g]),
               med([(r.get('gpu') or {}).get('mem_peak_mb') or r.get('torch_max_alloc_mb') for r in g]),
               (g[0].get('md5') or '')[:10]))
    base = [r for r in recs if r.get('ok') and int(r.get('lanes') or 0) == arms[0]]
    if base:
        b = med([r.get('fps') for r in base]) or 0.0
        for lanes in arms[1:]:
            g = [r for r in recs if r.get('ok') and int(r.get('lanes') or 0) == lanes]
            if g and b:
                say('车道 %d / 车道 %d = %+.1f%%'
                    % (lanes, arms[0], 100.0 * ((med([r.get('fps') for r in g]) or 0.0) / b - 1.0)))
    say('')
    say('[ok] %s' % a.out)
    return 0


if __name__ == '__main__':
    sys.exit(main())
