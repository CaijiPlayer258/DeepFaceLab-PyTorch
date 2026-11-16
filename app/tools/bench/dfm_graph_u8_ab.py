# -*- coding: utf-8 -*-
r"""DFL_DFM_GRAPH_U8 的生产 A/B：u8 量化留在 host（=0）vs 录进 CUDA Graph（=1）。

问题：现在 pred 的 u8 是**主机侧**做的（np.multiply + clip + astype，416×416×3 每张脸
2MB 的 float 走一遍），实测 GPU 车道上摊到 4.3ms/帧。把「×255 → clamp(0,255) → 截断」
录进 dfm 的 CUDA Graph 之后，图直接吐 uint8：D2H 4 倍小、host 侧一个 float 乘法都不剩。

跑法（仓库根；一臂一个**子进程**：会话/图/显存都不跨臂）：

    runtime\server\Scripts\python.exe app\tools\bench\dfm_graph_u8_ab.py
    ... --frames 150 --rounds 3 --arms 0 1
    ... --sample                       # 外部 nvidia-smi 采 SM%（不进 fps 结论）
    ... --dump-frames 41,80            # 把这几帧的逐脸 pred/遮罩落盘，两臂按 md5 比
    ... --encoder x264 --batch 1       # 成片 md5 对拍（NVENC 不稳，对拍必须钉 x264）

口径（与 experiments/gpu_lanes.md、fused_lane_ab.py 一致）：
  · fps / 分项 ms 取 _fused_run 自己的账（per_stage_ms），不额外埋探针；
  · 轮次交替（r1: 0→1，r2: 1→0，r3: 0→1），中位；
  · 帧预算 30、simeless + mt + mask6、编码 auto（= nvenc-h264）—— 与「现状 23.47 帧/秒」
    那一行同配置，所以只有 u8 这一段在动。
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
SETTINGS = {'blend': 'seamless', 'ct_mode': 'mt', 'mask_mode': 6,
            'mask_erode': 0, 'mask_blur': 4}
OUTDIR = Path(os.environ.get('TEMP', '.')) / 'dfl_u8_graph_ab'
RESULT_TAG = '##U8_RESULT##'


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


class Sampler(threading.Thread):
    """nvidia-smi -lms（外部采样，只给 SM%/显存两列）。"""

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
        except Exception:                                  # noqa: BLE001
            pass

    def stop(self):
        self.stop_flag.set()
        self.join(timeout=3)

    def summary(self, warm_s=2.0):
        rows = [r for r in self.rows if r[0] >= warm_s] or self.rows
        if not rows:
            return {}
        u = sorted(r[1] for r in rows)
        m = [r[2] for r in rows]
        return {'n': len(rows), 'util_mean': round(statistics.mean([r[1] for r in rows]), 1),
                'util_p50': round(u[len(u) // 2], 1), 'util_max': round(max(u), 1),
                'mem_peak_mb': round(max(m), 1),
                'idle_lt10_pct': round(100.0 * sum(1 for v in u if v < 10) / len(u), 1)}


def run_child(a):
    _boot()
    import torch
    from server.backend_merge import pipeline_fused as pf
    from server.backend_merge import pipeline_video as pv

    clip = Path(a.clip)
    outdir = Path(a.outdir) / ('u8_%d_r%d' % (int(a.arm), int(a.round)))
    outdir.mkdir(parents=True, exist_ok=True)
    out = outdir / ('u8_%d.mp4' % int(a.arm))
    if out.exists():
        out.unlink()
    body = {'video': str(clip), 'model': MODEL, 'settings': dict(SETTINGS),
            'output_dir': str(outdir), 'frame_budget': 30,
            'start_frame': int(a.start), 'end_frame': int(a.start) + int(a.frames) - 1}
    dump_dir = Path(a.outdir) / ('u8_%d_r%d.mp4.dump' % (int(a.arm), int(a.round)))
    smp = None
    try:
        cfg = pf._fused_cfg(body)
        pv._vid_open_log(out.with_suffix('.log'))
        t_load = time.time()
        ent = pf._dfm_session(str(WS / 'model' / (MODEL + '.dfm')))
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
        rec = {'ok': True, 'arm': int(a.arm), 'round': int(a.round),
               'frames': int(res.get('frames') or 0), 'fps': float(res.get('fps') or 0.0),
               'wall_s': wall, 'gpu_lanes': int(res.get('gpu_lanes') or 0),
               'per_stage_ms': res.get('per_stage_ms') or {},
               'stage_util_pct': res.get('stage_util_pct') or {},
               'batch_actual': res.get('batch_actual') or {},
               'peak_live': res.get('peak_live'), 'frame_budget': res.get('frame_budget'),
               'sink': res.get('sink'), 'out': str(res.get('out') or out),
               'md5': (md5_of(Path(res.get('out') or out))
                       if Path(res.get('out') or out).is_file() else ''),
               'session_load_s': load_s, 'frame_budget_cfg': res.get('frame_budget'),
               'torch_max_alloc_mb': alloc, 'torch_max_reserved_mb': reserved,
               'cpu_count': os.cpu_count(), 'model': MODEL,
               'encoder': str(os.environ.get('DFL_FUSED_ENCODER') or 'auto'),
               'graph_u8': str(os.environ.get('DFL_DFM_GRAPH_U8') or '0'),
               'sessions': (ent[6] if len(ent) > 6 else '?'),
               'dump': str(dump_dir) if dump_dir.is_dir() else ''}
        if smp is not None:
            rec['gpu'] = smp.summary(warm_s=float(a.sample_warm))
        return rec
    except BaseException as e:                             # noqa: BLE001
        return {'ok': False, 'arm': int(a.arm), 'round': int(a.round),
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


def spawn(a, arm, rnd):
    cmd = [str(PY), str(Path(__file__).resolve()), '--child',
           '--clip', str(a.clip), '--outdir', str(a.outdir), '--frames', str(a.frames),
           '--start', str(a.start), '--arm', str(arm), '--round', str(rnd)]
    if a.sample:
        cmd.append('--sample')
        cmd += ['--sample-warm', str(a.sample_warm)]
    env = dict(os.environ)
    env['DFL_DFM_GRAPH_U8'] = str(arm)
    env['DFL_FUSED_GPU_LANES'] = '1'
    env['PYTHONIOENCODING'] = 'utf-8'
    for k in ('DFL_COMPOSE_GPU', 'DFL_FUSED_BUDGET', 'DFL_FUSED_VRAM_FREE_MB',
              'DFL_ALLOW_CPU_FALLBACK', 'DFL_FUSED_BATCH_WAIT_MS'):
        env.pop(k, None)
    if a.dump_frames:
        env['DFL_FUSED_DUMP'] = str(a.dump_frames)
    else:
        env.pop('DFL_FUSED_DUMP', None)
    if a.batch:
        env['DFL_FUSED_BATCH'] = str(int(a.batch))
    else:
        env.pop('DFL_FUSED_BATCH', None)
    if a.encoder:
        env['DFL_FUSED_ENCODER'] = str(a.encoder)
    t0 = time.time()
    pr = subprocess.run(cmd, cwd=str(REPO), env=env, capture_output=True)
    txt = (pr.stdout or b'').decode('utf-8', 'replace')
    out = {'arm': arm, 'round': rnd, 'ok': False,
           'error': '子进程退出码 %s' % pr.returncode}
    for line in txt.splitlines():
        if line.startswith(RESULT_TAG):
            try:
                out = json.loads(line[len(RESULT_TAG):])
            except Exception:                          # noqa: BLE001
                pass
    out['child_s'] = round(time.time() - t0, 2)
    if not out.get('ok'):
        out['stderr_tail'] = (pr.stderr or b'').decode('utf-8', 'replace')[-900:]
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
    ap.add_argument('--arms', nargs='*', type=int, default=[0, 1])
    ap.add_argument('--encoder', default='auto')
    ap.add_argument('--batch', type=int, default=1,
                    help='钉住 DFL_FUSED_BATCH（0=不钉，让 _target_for 自己选）')
    ap.add_argument('--dump-frames', default='')
    ap.add_argument('--sample', action='store_true')
    ap.add_argument('--sample-warm', type=float, default=2.0)
    ap.add_argument('--child', action='store_true')
    ap.add_argument('--arm', type=int, default=0)
    ap.add_argument('--round', type=int, default=0)
    ap.add_argument('--outdir', default=str(OUTDIR))
    ap.add_argument('--out', default=str(OUTDIR / 'u8_ab.json'))
    a = ap.parse_args()
    Path(a.outdir).mkdir(parents=True, exist_ok=True)

    if a.child:
        rec = run_child(a)
        say(RESULT_TAG + json.dumps(rec, ensure_ascii=False))
        return 0

    arms = list(a.arms)
    recs = []
    for rnd in range(1, int(a.rounds) + 1):
        order = arms if rnd % 2 else list(reversed(arms))
        for arm in order:
            say('')
            say('=== DFL_DFM_GRAPH_U8=%d · 第 %d 轮 · %d 帧 ===' % (arm, rnd, a.frames))
            rec = spawn(a, arm, rnd)
            recs.append(rec)
            if rec.get('ok'):
                ps = rec.get('per_stage_ms') or {}
                say('  fps=%.2f  wall=%.2fs  dfm=%s u8=%s xseg=%s 等脸=%s  GPU段=%s ms  md5=%s'
                    % (rec.get('fps') or 0, rec.get('wall_s') or 0,
                       ps.get('dfm'), ps.get('u8'), ps.get('xseg'), ps.get('gpu_wait'),
                       round(sum(float(ps.get(k) or 0) for k in ('dfm', 'u8', 'xseg')), 1),
                       (rec.get('md5') or '')[:12]))
            else:
                say('  失败：%s' % str(rec.get('error'))[:300])
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(recs, indent=2, ensure_ascii=False), encoding='utf-8')

    say('')
    say('%-6s %8s %8s %8s %8s %8s %8s %8s %10s'
        % ('u8', 'fps', 'dfm', 'u8ms', 'xseg', 'gpu_wait', 'SM%', '显存MB', '成片md5'))
    for arm in arms:
        g = [r for r in recs if r.get('ok') and int(r.get('arm') or 0) == arm]
        if not g:
            say('%-6d %s' % (arm, '全部失败'))
            continue
        say('%-6d %8s %8s %8s %8s %8s %8s %8s %10s'
            % (arm,
               med([r.get('fps') for r in g]),
               med([(r.get('per_stage_ms') or {}).get('dfm') for r in g]),
               med([(r.get('per_stage_ms') or {}).get('u8') for r in g]),
               med([(r.get('per_stage_ms') or {}).get('xseg') for r in g]),
               med([(r.get('per_stage_ms') or {}).get('gpu_wait') for r in g]),
               med([(r.get('gpu') or {}).get('util_p50') for r in g]),
               med([(r.get('gpu') or {}).get('mem_peak_mb') or r.get('torch_max_alloc_mb') for r in g]),
               (g[0].get('md5') or '')[:10]))
    base = [r for r in recs if r.get('ok') and int(r.get('arm') or 0) == arms[0]]
    if base and len(arms) > 1:
        b = med([r.get('fps') for r in base]) or 0.0
        for arm in arms[1:]:
            g = [r for r in recs if r.get('ok') and int(r.get('arm') or 0) == arm]
            if g and b:
                say('u8=%d / u8=%d = %+.1f%%' % (arm, arms[0],
                    100.0 * ((med([r.get('fps') for r in g]) or 0.0) / b - 1.0)))
    say('')
    say('[ok] %s' % a.out)
    return 0


if __name__ == '__main__':
    sys.exit(main())
