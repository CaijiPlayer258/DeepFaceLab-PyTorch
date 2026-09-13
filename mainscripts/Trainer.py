import os
import sys
import traceback
import queue
import threading
import time
import numpy as np
import itertools
from pathlib import Path
from core import pathex
from core import imagelib
from core.imagelib.text import _get_pil_font
import cv2
import models
from core.interact import interact as io


def _make_chaotic_diffusion(shape, t):
    """FBM 体积云烟雾背景（H,W,3）float [0,1]。"""
    h, w = shape[:2]
    yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
    xx_n = xx.astype(np.float32) / max(w, 1)
    yy_n = yy.astype(np.float32) / max(h, 1)

    v = np.zeros((h, w), dtype=np.float32)
    amp = 1.0
    freq = 1.0
    n_octaves = 6
    for o in range(n_octaves):
        a = t * (0.020 + o * 0.010)
        cos_a = np.cos(a)
        sin_a = np.sin(a)
        xr = xx_n * cos_a + yy_n * sin_a
        yr = -xx_n * sin_a + yy_n * cos_a

        n = np.sin(xr * freq * 3.0 + t * (0.10 + o * 0.020))
        n += np.cos(yr * freq * 3.0 + t * (0.08 + o * 0.015))
        n += np.sin((xr + yr * 0.7) * freq * 2.5 + t * (0.12 + o * 0.012))
        n /= 3.0
        v += n * amp
        amp *= 0.50
        freq *= 2.0

    v = v / (2.0 - 0.5 ** (n_octaves - 1)) + 0.5
    v = np.clip(v, 0, 1)
    v = v * v * (3 - 2 * v)
    bg = 0.01 + v * 0.70
    return np.stack([bg] * 3, axis=-1).astype(np.float32)


def _draw_dynamic_text(cell, text, iter_val, position='bottom'):
    """在 cell (H,W,3) float [0,1] 上绘制动态蓝紫渐变文字，原地修改。
    position='bottom' 底左，'top' 顶左。渐变色沿 45°（左上→右下）随时间流动。

    === 这里是 CV2 预览窗口渐变文字的来源 ===
    不是 WebUI，不是 ModelBase，不是 onGetPreview。
    调用点：① Trainer 主循环 cv2 窗口显示 ② WebUI 预览图 JPEG 编码前烘焙。
    PIL 渲染 + 45° 渐变 + time.time() 驱动流动动画。
    """
    if not text:
        return
    h, w = cell.shape[:2]
    font_size = max(10, h // 25)
    pil_font = _get_pil_font(None, font_size)

    from PIL import Image, ImageDraw
    tmp = Image.new('RGB', (1, 1))
    bbox = ImageDraw.Draw(tmp).textbbox((0, 0), text, font=pil_font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]

    pad_x, pad_y = 3, 8
    if position == 'top':
        tx, ty = pad_x, pad_y
    else:
        tx, ty = pad_x, h - th - pad_y

    mask = Image.new('L', (w, h), 0)
    ImageDraw.Draw(mask).text((tx, ty), text, font=pil_font, fill=255)
    mask_np = np.asarray(mask, dtype=np.float32) / 255.0

    # 45° 梯度（相对于文字包围盒）
    yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
    d = ((xx - tx) + (yy - ty)).astype(np.float32) / max(1, tw + th)
    d = np.clip(d, 0, 1)

    t = (iter_val or 0) * 2.0

    hue = 0.62 + d * 0.23 + np.sin(t + d * np.pi * 2) * 0.04
    sat = 0.85 + np.sin(t * 0.7 + d) * 0.15
    val = 0.75 + np.sin(t * 0.5 + d * 1.5) * 0.25

    hsv_h = hue % 1.0
    hi = (hsv_h * 6).astype(np.int32) % 6
    f = hsv_h * 6 - np.floor(hsv_h * 6)
    p = val * (1.0 - sat)
    q = val * (1.0 - f * sat)
    t_ = val * (1.0 - (1.0 - f) * sat)

    r = np.select([hi==0, hi==1, hi==2, hi==3, hi==4, hi==5],
                  [val, q, p, p, t_, val])
    g = np.select([hi==0, hi==1, hi==2, hi==3, hi==4, hi==5],
                  [t_, val, val, q, p, p])
    b = np.select([hi==0, hi==1, hi==2, hi==3, hi==4, hi==5],
                  [p, p, t_, val, val, q])

    for c_idx, color in enumerate([b, g, r]):
        cell[:, :, c_idx] = np.where(mask_np > 0, color, cell[:, :, c_idx])

def trainerThread (s2c, c2s, e,
                    model_class_name = None,
                    saved_models_path = None,
                    training_data_src_path = None,
                    training_data_dst_path = None,
                    pretraining_data_path = None,
                    pretrained_model_path = None,
                    no_preview=False,
                    force_model_name=None,
                    force_gpu_idxs=None,
                    cpu_only=None,
                    silent_start=False,
                    model_ready_event=None,
                    execute_programs = None,
                    max_iters: int = 0,
                    debug=False,
                    **kwargs):
    _webui_trainer = None
    try:
        from WebUI import page_trainer as _webui_trainer
    except Exception:
        pass

    # ── 进程互斥锁：防止同一模型目录被启动第二个训练进程 ──────
    _lock_file = Path(saved_models_path) / '.train.lock' if saved_models_path else None
    if _lock_file is not None:
        try:
            if _lock_file.exists():
                _pid_str = _lock_file.read_text().strip()
                if _pid_str:
                    _pid = int(_pid_str)
                    # Windows 下 os.kill(pid, 0) 检测进程是否存活
                    try:
                        os.kill(_pid, 0)
                        raise RuntimeError(
                            f"模型目录 {saved_models_path} 已被进程 PID={_pid} 锁定，"
                            f"无法启动第二个训练进程。请先关闭已有训练进程，"
                            f"或删除锁文件 {_lock_file}。"
                        )
                    except OSError:
                        pass  # 进程已死，锁过期
            _lock_file.write_text(str(os.getpid()))
        except Exception as _lock_err:
            if '已被进程' in str(_lock_err):
                raise
            pass  # 非致命错误，继续启动
    # ── 锁结束 ──────────────────────────────────────────────
    def _lock_cleanup():
        try:
            if _lock_file and _lock_file.exists():
                _lock_file.unlink()
        except Exception:
            pass

    while True:
        try:
            start_time = time.time()

            if not training_data_src_path.exists():
                training_data_src_path.mkdir(exist_ok=True, parents=True)

            if not training_data_dst_path.exists():
                training_data_dst_path.mkdir(exist_ok=True, parents=True)

            if not saved_models_path.exists():
                saved_models_path.mkdir(exist_ok=True, parents=True)
                            
            model = models.import_model(model_class_name)(
                        is_training=True,
                        saved_models_path=saved_models_path,
                        training_data_src_path=training_data_src_path,
                        training_data_dst_path=training_data_dst_path,
                        pretraining_data_path=pretraining_data_path,
                        pretrained_model_path=pretrained_model_path,
                        no_preview=no_preview,
                        force_model_name=force_model_name,
                        force_gpu_idxs=force_gpu_idxs,
                        cpu_only=cpu_only,
                        silent_start=silent_start,
                        debug=debug)

            # Signal main() that model is loaded; WebUI can start now
            # (before sample loader initialization, so WebUI appears immediately)
            if model_ready_event is not None:
                model_ready_event.set()

            start_iter = model.get_iter()
            if max_iters is None:
                max_iters = 0
            try:
                max_iters = int(max_iters)
            except Exception:
                max_iters = 0

            is_reached_goal = model.is_reached_iter_goal()

            shared_state = { 'after_save' : False }
            loss_string = ""
            save_iter =  model.get_iter()
            def model_save():
                if not debug and not is_reached_goal:
                    # 保存前清理内存碎片，降低 save 期间 OOM 概率
                    import gc
                    gc.collect()
                    try:
                        import torch
                        torch.cuda.empty_cache()
                    except Exception:
                        pass
                    io.log_info ("正在保存……", end='\r')
                    model.save()
                    _persist_model_options()
                    shared_state['after_save'] = True
                    
            def model_backup():
                if not debug and not is_reached_goal:
                    model.create_backup()
            def _persist_model_options():
                import pickle
                try:
                    _dpath = Path(model.get_strpath_storage_for_file(model.get_model_name() + '_data.dat'))
                    if _dpath.exists():
                        _d = pickle.loads(_dpath.read_bytes())
                        _d['options'].update(dict(model.options))
                        _dpath.write_bytes(pickle.dumps(_d))
                except Exception as _e:
                    pass
            def send_preview():
                if not debug:
                    previews = model.get_previews()
                    n_samples_val = 0
                    n_cols_val = 5
                    if previews:
                        p_h, p_w = previews[0][1].shape[:2]
                        # Detect: assume square cells, n_cols = pw // (ph // 4_guess)
                        n_cols_guess = max(1, p_w // max(p_h // 4, 1))
                        if n_cols_guess <= 5:
                            n_cols_val = 5  # SAEHD compatibility
                        else:
                            n_cols_val = n_cols_guess
                        # n_samples from detected column count
                        cell_h = p_w // n_cols_val
                        n_samples_val = p_h // cell_h if cell_h > 0 else 0
                    payload = {'op':'show', 'previews': previews, 'iter':model.get_iter(),
                               'loss_history': model.get_loss_history().copy(),
                               'n_samples': n_samples_val, 'n_cols': n_cols_val}
                    if hasattr(model, 'last_filenames') and model.last_filenames:
                        payload['src_filenames'] = model.last_filenames[0] if len(model.last_filenames) > 0 else []
                        payload['dst_filenames'] = model.last_filenames[1] if len(model.last_filenames) > 1 else []
                    if model.loss_history:
                        last = model.loss_history[-1]
                        payload['src_loss'] = float(last[0]) if len(last) > 0 else 0.0
                        payload['dst_loss'] = float(last[1]) if len(last) > 1 else float(last[0])
                    # per-sample loss vectors for individual image labels
                    # XSegLite: per-sample loss for gradient text overlay
                    if hasattr(model, '_last_loss_per_sample') and model._last_loss_per_sample is not None:
                        v = model._last_loss_per_sample
                        payload['src_loss_vec'] = v.tolist() if hasattr(v, 'tolist') else list(v)
                    elif hasattr(model, '_last_src_loss_per_sample') and model._last_src_loss_per_sample is not None:
                        payload['src_loss_vec'] = model._last_src_loss_per_sample.tolist()
                    if hasattr(model, '_last_dst_loss_per_sample') and model._last_dst_loss_per_sample is not None:
                        payload['dst_loss_vec'] = model._last_dst_loss_per_sample.tolist()
                    pm = getattr(model, '_preview_masks', None)
                    if pm is not None:
                        payload['mask_data'] = pm
                    c2s.put(payload)
                else:
                    previews = [( 'debug, press update for new', model.debug_one_iter())]
                    c2s.put ( {'op':'show', 'previews': previews} )
                e.set() #Set the GUI Thread as Ready

            def _push_settings_to_cache(model, webui):
                """Read current generator settings and push to WebUI cache."""
                try:
                    gl = model.get_training_data_generators()
                    if not gl or len(gl) < 2:
                        return
                    def _read_gen(g):
                        sp = g.sample_process_options
                        sp_data = {}
                        for k in ('random_flip', 'rotation_range', 'scale_range', 'tx_range', 'ty_range'):
                            if hasattr(sp, k):
                                v = getattr(sp, k)
                                if isinstance(v, np.ndarray):
                                    v = v.tolist()
                                sp_data[k] = v
                        ot_data = {}
                        for out in g.output_sample_types:
                            if isinstance(out, dict) and ('warp' in out or 'ct_mode' in out):
                                for k in ('warp', 'transform', 'ct_mode', 'random_hsv_shift_amount'):
                                    if k in out:
                                        v = out[k]
                                        if isinstance(v, np.floating):
                                            v = float(v)
                                        elif isinstance(v, np.integer):
                                            v = int(v)
                                        elif isinstance(v, np.ndarray):
                                            v = v.tolist()
                                        elif k == 'ct_mode' and v is None:
                                            v = 'none'
                                        ot_data[k] = v
                                break
                        # Fill missing keys with effective defaults (V2 generator behaviour)
                        for k, d in {'warp':False, 'transform':True, 'ct_mode':None,
                                     'random_hsv_shift_amount':0}.items():
                            ot_data.setdefault(k, d)
                        ld_data = {}
                        if hasattr(g, 'loader') and g.loader is not None:
                            bs = getattr(g.loader, 'batch_size', None)
                            if bs is not None:
                                ld_data['batch_size'] = int(bs)
                        return {
                            'sample_process_options': sp_data,
                            'output_sample_types': ot_data,
                            'loader': ld_data,
                        }
                    mo = {}
                    for k in ('crash_threshold', 'backup_interval', 'max_backups',
                              'resolution', 'face_type', 'archi', 'ae_dims', 'e_dims', 'd_dims', 'd_mask_dims',
                              'adabelief', 'clipgrad', 'gan_power', 'gan_patch_size', 'gan_dims',
                              'lr', 'lr_cos', 'lr_dropout', 'use_bf16', 'models_opt_on_gpu',
                              'random_warp', 'random_hsv_power',
                              'random_src_flip', 'random_dst_flip',
                              'eyes_mouth_prio', 'uniform_yaw', 'blur_out_mask', 'use_fast_generator',
                              'gradient_checkpointing', 'pretrain', 'true_face_power', 'face_style_power',
                              'bg_style_power', 'vgg_perceptual_power', 'masked_training', 'write_preview_history', 'target_iter'):
                        if k in model.options:
                            mo[k] = model.options[k]
                    mo['model_name'] = model.get_model_name()
                    mo['model_dir'] = str(model.get_model_root_path())
                    webui.update_cache({
                        'settings_src': _read_gen(gl[0]),
                        'settings_dst': _read_gen(gl[1]),
                        'model_options': mo,
                    })
                except Exception:
                    pass

            def _apply_settings_update(model, params):
                """Apply settings update to model generators."""
                try:
                    gl = model.get_training_data_generators()
                    if not gl or len(gl) < 2:
                        return
                    logs = []
                    logged = set()
                    def _apply_to_gen(g, updates, side_name):
                        sp = updates.get('sample_process_options', {})
                        if sp and hasattr(g, 'sample_process_options'):
                            spo = g.sample_process_options
                            for k, v in sp.items():
                                if hasattr(spo, k):
                                    old = getattr(spo, k)
                                    if old != v:
                                        setattr(spo, k, v)
                                        logs.append(f'[WebUI] {side_name} SP.{k}: {old} -> {v}')
                        otypes = updates.get('output_sample_types', {})
                        if otypes:
                            key_order = ('warp', 'transform', 'ct_mode', 'random_hsv_shift_amount')
                            for out in g.output_sample_types:
                                if isinstance(out, dict):
                                    for k in key_order:
                                        if k in otypes:
                                            v = otypes[k]
                                            if k == 'ct_mode' and v == 'none':
                                                v = None
                                            old = out.get(k, {'warp':False, 'transform':True, 'ct_mode':None,
                                                               'random_hsv_shift_amount':0}.get(k))
                                            if old != v:
                                                out[k] = v
                                                lk = f'OT.{k}'
                                                if lk not in logged:
                                                    logged.add(lk)
                                                    logs.append(f'[WebUI] OT.{k}: {old} -> {v}')
                        ld = updates.get('loader', {})
                        if ld and hasattr(g, 'loader') and g.loader is not None:
                            for k, v in ld.items():
                                if hasattr(g.loader, k):
                                    old = getattr(g.loader, k)
                                    if old != v:
                                        setattr(g.loader, k, v)
                                        lk = f'LD.{k}'
                                        if lk not in logged:
                                            logged.add(lk)
                                            logs.append(f'[WebUI] LD.{k}: {old} -> {v}')
                    if 'src' in params:
                        _apply_to_gen(gl[0], params['src'], 'SRC')
                    if 'dst' in params:
                        _apply_to_gen(gl[1], params['dst'], 'DST')
                    for line in logs:
                        io.log_info(line)
                    # Clear loader buffers so fresh items reflect new settings
                    for g in gl:
                        try:
                            if hasattr(g, 'loader') and g.loader is not None:
                                g.loader.clear_cache()
                        except Exception:
                            pass
                except Exception:
                    pass

            _HOT_KEYS = {
                'crash_threshold', 'backup_interval', 'max_backups',
                'face_style_power', 'bg_style_power',
                'masked_training', 'blur_out_mask', 'eyes_mouth_prio', 'uniform_yaw',
                'lr', 'lr_cos', 'lr_dropout',
                'random_src_flip', 'random_dst_flip', 'random_hsv_power',
            }
            def _apply_model_options(model, updates):
                for k, v in updates.items():
                    if k not in _HOT_KEYS:
                        io.log_info(f'[WebUI] 跳过不可热更新参数: {k}')
                        continue
                    try:
                        old = model.options.get(k)
                        if old == v:
                            continue

                        # 学习率——先校验再保存
                        if k == 'lr':
                            try:
                                v = float(v)
                            except (ValueError, TypeError):
                                io.log_info(f'[WebUI] 忽略无效学习率: {v!r}')
                                continue
                            model.options[k] = v
                            io.log_info(f'[WebUI] Model option {k}: {old} -> {v}')
                            try:
                                model.src_dst_opt.lr = v
                            except Exception:
                                pass
                        else:
                            model.options[k] = v
                            io.log_info(f'[WebUI] Model option {k}: {old} -> {v}')

                        # 余弦退火·热重启周期（Cosine Annealing with Warm Restarts）
                        if k == 'lr_cos':
                            try:
                                model.src_dst_opt.lr_cos = int(v)
                            except Exception:
                                pass
                        elif k == 'max_backups':
                            model.max_backups = int(v)
                        # random_warp——更新生成器的 warp 输出配置
                        elif k == 'random_warp':
                            try:
                                _warp_v = bool(v)
                                for _gen in model.get_training_data_generators():
                                    if hasattr(_gen, 'output_sample_types') and len(_gen.output_sample_types) > 0:
                                        _gen.output_sample_types[0]['warp'] = _warp_v
                                io.log_info(f'[WebUI] random_warp -> {_warp_v}（已更新生成器）')
                            except Exception as _we:
                                io.log_info(f'[WebUI] random_warp 更新失败: {_we}')
                        # 翻转——更新对应生成器的 sample_process_options
                        elif k in ('random_src_flip', 'random_dst_flip'):
                            try:
                                gl = model.get_training_data_generators()
                                if gl:
                                    idx = 0 if k == 'random_src_flip' else 1
                                    if idx < len(gl) and hasattr(gl[idx], 'sample_process_options'):
                                        gl[idx].sample_process_options.random_flip = bool(v)
                            except Exception:
                                pass
                        elif hasattr(model, k):
                            setattr(model, k, bool(v) if isinstance(v, bool) else type(old)(v) if old is not None else v)
                    except Exception as e:
                        io.log_info(f'[WebUI] 更新参数 {k} 失败: {e}')

            # Push initial generator settings to WebUI cache
            if _webui_trainer is not None:
                try:
                    _push_settings_to_cache(model, _webui_trainer)
                except Exception:
                    pass

            if model.get_target_iter() != 0:
                if is_reached_goal:
                    io.log_info('模型已训练到目标迭代次数，可以使用预览。')
                else:
                    io.log_info('开始训练。目标迭代次数：%d。按“回车（Enter）”停止训练并保存模型。' % ( model.get_target_iter()  ) )
            else:
                io.log_info('开始训练。按“回车（Enter）”停止训练并保存模型。')

            last_backup_iter = model.get_iter()
            model.max_backups = int(model.options.get('max_backups', 24))
            backup_interval = int(model.options.get('backup_interval', 500))

            # execute_programs is optional; normalize to a safe iterable.
            if execute_programs is None:
                execute_programs = []
            elif not isinstance(execute_programs, (list, tuple)):
                execute_programs = []

            _normalized_execute_programs = []
            for x in execute_programs:
                try:
                    prog_time, prog = x[0], x[1]
                except Exception:
                    continue
                _normalized_execute_programs.append([prog_time, prog, time.time()])
            execute_programs = _normalized_execute_programs

            for i in itertools.count(0,1):
                if not debug:
                    cur_time = time.time()

                    for x in execute_programs:
                        prog_time, prog, last_time = x
                        exec_prog = False
                        if prog_time > 0 and (cur_time - start_time) >= prog_time:
                            x[0] = 0
                            exec_prog = True
                        elif prog_time < 0 and (cur_time - last_time)  >= -prog_time:
                            x[2] = cur_time
                            exec_prog = True

                        if exec_prog:
                            try:
                                exec(prog)
                            except Exception as e:
                                print("无法执行程序片段: %s" % (prog) )

                    if not is_reached_goal:

                        if model.get_iter() == 0:
                            io.log_info("")
                            io.log_info("正在尝试进行首次迭代。如果出现错误，请降低模型参数/配置。")
                            io.log_info("")
                            
                            if sys.platform[0:3] == 'win':
                                io.log_info("!!!")
                                io.log_info("Windows 10 用户重要提示：为保证正常运行，请按下图进行设置。")
                                io.log_info("https://i.imgur.com/B7cmDCB.jpg")
                                io.log_info("!!!")

                        iter, iter_time = model.train_one_iter()

                        # Push real-time iter/loss to WebUI (every iteration)
                        if _webui_trainer is not None:
                            try:
                                _webui_trainer.update_cache({
                                    'iter': iter,
                                    'loss_history': model.get_loss_history(),
                                })
                            except Exception:
                                pass

                        # Check for pending settings update from WebUI
                        try:
                            settings_update = _webui_trainer.get_settings_update()
                            if settings_update is not None:
                                _apply_settings_update(model, settings_update)
                                _push_settings_to_cache(model, _webui_trainer)
                        except Exception:
                            pass

                        # Check for pending model option updates from WebUI
                        try:
                            model_updates = _webui_trainer.get_model_options_update()
                            if model_updates is not None:
                                _apply_model_options(model, model_updates)
                                _push_settings_to_cache(model, _webui_trainer)
                        except Exception:
                            pass

                        if (not is_reached_goal) and (not debug) and max_iters > 0 and (iter - start_iter) >= max_iters:
                            io.log_info(f'\n已达到 max_iters={max_iters}。正在保存并停止。')
                            model_save()
                            is_reached_goal = True
                            send_preview()
                            i = -1
                            break

                        loss_history = model.get_loss_history()
                        time_str = time.strftime("[%H:%M:%S]")

                        # 当前实际学习率。调度器（余弦退火 / 平台期衰减）会让它随时变化，
                        # 只看起始 lr 没法判断"现在是多少"，所以直接打进日志。
                        # 格式 "lr:1.23e-05"；拿不到就显示 "lr:?"，不影响训练。
                        _opt = getattr(model, 'src_dst_opt', None)
                        try:
                            if _opt is not None and hasattr(_opt, 'get_lr'):
                                _lr_now = float(_opt.get_lr())
                                lr_txt = "[lr:{0:.2e}]".format(_lr_now)
                            else:
                                lr_txt = "[lr:?]"
                        except Exception:
                            lr_txt = "[lr:?]"

                        if iter_time >= 10:
                            loss_string = "{0}[#{1:06d}][{2:.5s}s]{3}".format ( time_str, iter, '{:0.4f}'.format(iter_time), lr_txt )
                        else:
                            loss_string = "{0}[#{1:06d}][{2:04d}ms]{3}".format ( time_str, iter, int(iter_time*1000), lr_txt )

                        if shared_state['after_save']:
                            shared_state['after_save'] = False
                            
                            mean_loss = np.mean ( loss_history[save_iter:iter], axis=0)

                            for loss_value in mean_loss:
                                loss_string += "[%.4f]" % (loss_value)

                            io.log_info (loss_string)

                            save_iter = iter
                        else:
                            for loss_value in loss_history[-1]:
                                loss_string += "[%.4f]" % (loss_value)

                            if io.is_colab():
                                io.log_info ('\r' + loss_string, end='')
                            else:
                                io.log_info (loss_string, end='\r')

                        if model.get_iter() == 1:
                            model_save()

                        if model.get_target_iter() != 0 and model.is_reached_iter_goal():
                            io.log_info ('已达到目标迭代次数。')
                            model_save()
                            is_reached_goal = True
                            io.log_info ('现在可以使用预览。')
                
                # 迭代间隔备份 (0 = 禁用自动备份)
                _bi = int(model.options.get('backup_interval', 500))
                if not is_reached_goal and _bi > 0 and (model.get_iter() - last_backup_iter) >= _bi:
                    last_backup_iter = model.get_iter()
                    model_save()
                    model_backup()
                    send_preview()

                if i==0:
                    if is_reached_goal:
                        model.pass_one_iter()
                    send_preview()

                if debug:
                    time.sleep(0.005)

                while not s2c.empty():
                    input = s2c.get()
                    op = input['op']
                    if op == 'save':
                        model_save()
                    elif op == 'backup':
                        model_backup()
                    elif op == 'preview':
                        if is_reached_goal:
                            model.pass_one_iter()
                        send_preview()
                    elif op == 'close':
                        model_save()
                        i = -1
                        break

                if i == -1:
                    break



            model.finalize()

        except Exception as e:
            print ('Error: %s' % (str(e)))
            traceback.print_exc()
        break
    # 清理进程锁
    try:
        _lock_cleanup()
    except Exception:
        pass
    c2s.put ( {'op':'close'} )



def main(**kwargs):
    io.log_info ("正在运行训练器。\r\n")

    no_preview = kwargs.get('no_preview', False)

    s2c = queue.Queue()
    c2s = queue.Queue()

    e = threading.Event()
    model_ready_event = threading.Event()
    kwargs['model_ready_event'] = model_ready_event
    thread = threading.Thread(target=trainerThread, args=(s2c, c2s, e), kwargs=kwargs )
    thread.start()

    # Wait until model is loaded before starting WebUI
    io.log_info("等待模型加载完成后启动 WebUI...")
    model_ready_event.wait()
    io.log_info("模型已加载，正在启动 WebUI...")

    # Start WebUI monitor (daemon thread, HTTP on webui_port)
    _webui = None
    try:
        from WebUI import page_trainer as _webui_module
        _webui_module.start_webui_server(
            port=int(kwargs.get('webui_port', 6789)),
            model_dir=str(kwargs.get('saved_models_path', '.')),
            password=kwargs.get('webui_password'),
        )
        _webui = _webui_module
    except Exception as exc:
        print(f'[WebUI] 启动失败: {exc}')

    # 自动在默认浏览器中打开 WebUI
    try:
        import webbrowser
        _port = int(kwargs.get('webui_port', 6789))
        _password = kwargs.get('webui_password', '')
        _url = f"http://127.0.0.1:{_port}"
        if _password:
            _url += f"?password={_password}"
        # 延迟 1.5s 等服务器完全就绪后再打开
        import threading as _thr
        _thr.Timer(1.5, lambda: webbrowser.open(_url)).start()
    except Exception:
        pass

    e.wait() #Wait for inital load to occur.

    if no_preview:
        _preview_pending = False
        while True:
            if not c2s.empty():
                input = c2s.get()
                op = input.get('op','')
                if op == 'close':
                    break
                if op == 'show' and _webui is not None:
                    _webui.update_cache({
                        'iter': input.get('iter', 0),
                        'loss_history': input.get('loss_history', []),
                    })
                    # Encode and cache previews when requested
                    if _preview_pending:
                        previews = input.get('previews')
                        if previews:
                            try:
                                encoded = {}
                                for name, bgr in previews:
                                    img = (np.clip(bgr, 0, 1) * 255).astype(np.uint8)
                                    ok, jpg = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 100])
                                    if ok:
                                        encoded[name] = jpg.tobytes()
                                # Pass filename labels and losses alongside previews
                                labels = {
                                    'src_fnames': input.get('src_filenames', []),
                                    'dst_fnames': input.get('dst_filenames', []),
                                    'n_samples': input.get('n_samples', 0),
                                    'n_cols': input.get('n_cols', 5),
                                    'src_loss': input.get('src_loss', 0),
                                    'dst_loss': input.get('dst_loss', 0),
                                    'src_loss_vec': input.get('src_loss_vec', []),
                                    'dst_loss_vec': input.get('dst_loss_vec', []),
                                }
                                _webui.update_cache({'previews': encoded, 'preview_labels': labels})
                            finally:
                                _preview_pending = False

            # WebUI 远程控制事件
            if _webui is not None:
                if _webui.is_preview_requested():
                    _webui.clear_preview_request()
                    _preview_pending = True
                    s2c.put({'op': 'preview'})
                if _webui.is_close_requested():
                    io.log_info('收到WebUI退出请求，正在保存并退出…')
                    _webui.clear_close_request()
                    _webui.clear_save_request()  # close already saves, avoid redundant save op
                    s2c.put({'op': 'close'})
                elif _webui.is_save_requested():
                    s2c.put({'op': 'save'})
                    _webui.clear_save_request()

            try:
                io.process_messages(0.1)
            except KeyboardInterrupt:
                s2c.put ( {'op': 'close'} )
    else:
        wnd_name = "Training preview"
        io.named_window(wnd_name)
        io.capture_keys(wnd_name)

        # Some OpenCV builds / Windows desktop setups may intermittently report a newly created
        # window as not visible, which would instantly trigger a clean shutdown.
        # Provide a debounce and an escape hatch via env vars.
        # 默认禁用自动关闭（OpenCV 窗口在无 GUI 环境下不可见时仍可正常训练）
        # 设置 DFL_ENABLE_TRAIN_PREVIEW_AUTOCLOSE=1 以恢复旧行为
        disable_preview_autoclose = str(os.environ.get('DFL_ENABLE_TRAIN_PREVIEW_AUTOCLOSE', '')).strip().lower() not in (
            '1', 'y', 'yes', 'true', 'on'
        )
        try:
            preview_autoclose_grace_sec = float(os.environ.get('DFL_TRAIN_PREVIEW_AUTOCLOSE_GRACE_SEC', '2.0'))
        except Exception:
            preview_autoclose_grace_sec = 2.0
        preview_loop_start_time = time.time()
        invisible_consecutive = 0

        previews = None
        loss_history = None
        selected_preview = 0
        update_preview = False
        is_showing = False
        is_waiting_preview = False
        show_last_history_iters_count = 10000
        iter = 0
        close_requested = False
        close_requested_time = 0.0
        selected_preview_name = ''

        # 缓存：基础图（无文字）、loss图、文件名、loss值
        cached_base = None
        cached_loss_img = None
        cached_loss_iter = -1
        cached_src_fnames = []
        cached_dst_fnames = []
        cached_src_loss = ''
        cached_dst_loss = ''
        cached_src_loss_val = 0.0
        cached_dst_loss_val = 0.0
        cached_src_loss_vec = []
        cached_dst_loss_vec = []
        cached_n_samples = 0
        cached_n_cols = 0
        cached_mask_data = None
        _webui_cached_previews = None
        _webui_preview_pending = False
        while True:
            # If user closed the window via OS controls, request a clean shutdown.
            if (not close_requested) and (not disable_preview_autoclose) and is_showing:
                try:
                    # OpenCV returns < 1 when window is closed/hidden.
                    if (time.time() - preview_loop_start_time) >= preview_autoclose_grace_sec:
                        if cv2.getWindowProperty(wnd_name, cv2.WND_PROP_VISIBLE) < 1:
                            invisible_consecutive += 1
                        else:
                            invisible_consecutive = 0

                        # Debounce: require a few consecutive reads to avoid false positives.
                        if invisible_consecutive >= 3:
                            io.log_info(
                                '检测到训练预览窗口已关闭/不可见，正在保存并退出… '
                                '(设置 DFL_ENABLE_TRAIN_PREVIEW_AUTOCLOSE=1 启用窗口检测自动退出)'
                            )
                            close_requested = True
                            close_requested_time = time.time()
                            s2c.put({'op': 'close'})
                except Exception:
                    pass

            if not c2s.empty():
                input = c2s.get()
                op = input['op']
                if op == 'show':
                    is_waiting_preview = False
                    loss_history = input['loss_history'] if 'loss_history' in input.keys() else None
                    previews = input['previews'] if 'previews' in input.keys() else None
                    iter = input['iter'] if 'iter' in input.keys() else 0
                    cached_src_fnames = input.get('src_filenames', [])
                    cached_dst_fnames = input.get('dst_filenames', [])
                    cached_src_loss_val = float(input['src_loss']) if 'src_loss' in input else 0.0
                    cached_dst_loss_val = float(input['dst_loss']) if 'dst_loss' in input else 0.0
                    cached_src_loss = f"{cached_src_loss_val:.4f}" if 'src_loss' in input else ''
                    cached_dst_loss = f"{cached_dst_loss_val:.4f}" if 'dst_loss' in input else ''
                    cached_src_loss_vec = input.get('src_loss_vec', [])
                    cached_dst_loss_vec = input.get('dst_loss_vec', [])
                    cached_n_cols = input.get('n_cols', 0)
                    cached_mask_data = input.get('mask_data')
                    cached_n_samples = input.get('n_samples', 0)

                    # Update WebUI and cache previews for on-demand encoding
                    if _webui is not None:
                        _webui.update_cache({
                            'iter': iter,
                            'loss_history': loss_history,
                        })
                        _webui_cached_previews = previews

                        # If WebUI requested a preview refresh, encode and push now
                        if _webui_preview_pending:
                            try:
                                encoded = {}
                                for name, bgr in previews:
                                    img = (np.clip(bgr, 0, 1) * 255).astype(np.uint8)
                                    ok, jpg = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 100])
                                    if ok:
                                        encoded[name] = jpg.tobytes()
                                labels = {
                                    'src_fnames': cached_src_fnames,
                                    'dst_fnames': cached_dst_fnames,
                                    'n_samples': cached_n_samples,
                                    'n_cols': cached_n_cols if cached_n_cols else 5,
                                    'src_loss': cached_src_loss_val,
                                    'dst_loss': cached_dst_loss_val,
                                    'src_loss_vec': cached_src_loss_vec,
                                    'dst_loss_vec': cached_dst_loss_vec,
                                }
                                _webui.update_cache({'previews': encoded, 'preview_labels': labels})
                            finally:
                                _webui_preview_pending = False

                    if previews is not None:
                        max_w = 0
                        max_h = 0
                        for (preview_name, preview_rgb) in previews:
                            (h, w, c) = preview_rgb.shape
                            max_h = max (max_h, h)
                            max_w = max (max_w, w)

                        max_size = 800
                        if max_h > max_size:
                            max_w = int( max_w / (max_h / max_size) )
                            max_h = max_size

                        #make all previews size equal
                        for preview in previews[:]:
                            (preview_name, preview_rgb) = preview
                            (h, w, c) = preview_rgb.shape
                            if h != max_h or w != max_w:
                                previews.remove(preview)
                                previews.append ( (preview_name, cv2.resize(preview_rgb, (max_w, max_h))) )
                        selected_preview = selected_preview % len(previews)
                        update_preview = True
                elif op == 'close':
                    break

            if update_preview:
                update_preview = False

                selected_preview_name = previews[selected_preview][0]
                selected_preview_rgb = previews[selected_preview][1]
                (h,w,c) = selected_preview_rgb.shape

                # GUI: 最多显示4行，WebUI不受限制
                _cell_w = w // 5
                _total_n = h // _cell_w
                if _total_n > 4:
                    selected_preview_rgb = selected_preview_rgb[:4 * _cell_w]

                # HEAD
                if close_requested:
                    elapsed = max(0.0, time.time() - close_requested_time)
                    head_lines = [
                        f'正在保存并退出…请稍候（{elapsed:0.1f}s）',
                        '[Ctrl+C]：强制退出',
                        ' ',
                    ]
                else:
                    head_lines = [
                        '[s]：保存  [b]：备份  [回车]：退出',
                        '[p]：刷新  [空格]：下一个预览  [l]：切换历史范围',
                        '预览："%s" [%d/%d]' % (selected_preview_name,selected_preview+1, len(previews) )
                        ]
                head_line_height = 15
                head_height = len(head_lines) * head_line_height
                head = np.ones ( (head_height,w,c) ) * 0.1

                for i in range(0, len(head_lines)):
                    t = i*head_line_height
                    b = (i+1)*head_line_height
                    head[t:b, 0:w] += imagelib.get_text_image (  (head_line_height,w,c) , head_lines[i], color=[0.8]*c )

                # 组装基础图（head + loss + 预览，不含文字覆盖）
                base = head
                if loss_history is not None:
                    if show_last_history_iters_count == 0:
                        loss_history_to_show = loss_history
                    else:
                        loss_history_to_show = loss_history[-show_last_history_iters_count:]
                    cached_loss_img = models.ModelBase.get_loss_history_preview(loss_history_to_show, iter, w, c)
                    base = np.concatenate([base, cached_loss_img], axis=0)
                base = np.concatenate([base, selected_preview_rgb], axis=0)
                cached_base = np.clip(base, 0, 1)

                # 记录样本数用于文字定位（来自 payload 或几何推算）
                if cached_n_samples == 0:
                    cached_n_samples = selected_preview_rgb.shape[0] // (selected_preview_rgb.shape[1] // 5)

            # 每帧：在缓存图上叠加动态文字
            if cached_base is not None:
                final = cached_base.copy()

                # 预览区域在组装图中的起始行（head固定45px，loss 100px）
                prev_start = 45 + (cached_loss_img.shape[0] if cached_loss_img is not None else 0)
                if prev_start >= final.shape[0]:
                    prev_start = 0

                # 在预览区域叠加文字
                if prev_start > 0 and cached_n_samples > 0:
                    preview_region = final[prev_start:, :, :]
                    ph, pw = preview_region.shape[:2]
                    n_cols = cached_n_cols if cached_n_cols else 5
                    cell_w = pw // n_cols
                    # rows = ph / cell_w (square cells), clamped to known sample count
                    actual_rows = ph // cell_w
                    n_rows = min(cached_n_samples, actual_rows) if actual_rows > 0 else cached_n_samples
                    cell_h = ph // n_rows

                    # 遮罩视图：模型 onGetPreview() 已乘好遮罩，不做额外合成

                    # 列标签已由模型 label strip 显示，不在图像上重复绘制
                    if '合并预览' in selected_preview_name:
                        pass
                    else:
                        for i in range(n_rows):
                            y0 = prev_start + i * cell_h

                            if n_cols == 5:
                                _fn_s = (cached_src_fnames[i] if i < len(cached_src_fnames) else '')
                                if _fn_s:
                                    _fn_s = os.path.basename(str(_fn_s))
                                    _draw_dynamic_text(final[y0:y0+cell_h, 0:cell_w], _fn_s, time.time() % 1000, 'bottom')
                                _fn_d = (cached_dst_fnames[i] if i < len(cached_dst_fnames) else '')
                                if _fn_d:
                                    _fn_d = os.path.basename(str(_fn_d))
                                    _draw_dynamic_text(final[y0:y0+cell_h, 2*cell_w:3*cell_w], _fn_d, time.time() % 1000, 'bottom')
                                if cached_src_loss:
                                    _draw_dynamic_text(final[y0:y0+cell_h, cell_w:2*cell_w], cached_src_loss, time.time() % 1000, 'bottom')
                                if cached_dst_loss:
                                    _draw_dynamic_text(final[y0:y0+cell_h, 3*cell_w:4*cell_w], cached_dst_loss, time.time() % 1000, 'bottom')
                            else:
                                # XSegLite: col1=filename, col4=per-sample loss
                                _fn = (cached_src_fnames[i] if i < len(cached_src_fnames) else '')
                                if _fn:
                                    _fn = os.path.basename(str(_fn))
                                    _draw_dynamic_text(final[y0:y0+cell_h, 0:cell_w], _fn, time.time() % 1000, 'bottom')
                                _cl = (cached_src_loss_vec[i] if i < len(cached_src_loss_vec) else None)
                                _sl = f'{_cl:.4f}' if _cl is not None else ''
                                if _sl:
                                    _draw_dynamic_text(final[y0:y0+cell_h, 3*cell_w:4*cell_w], _sl, time.time() % 1000, 'bottom')

                io.show_image(wnd_name, (final * 255).astype(np.uint8))
                is_showing = True

            key_events = io.get_key_events(wnd_name)
            key, chr_key, ctrl_pressed, alt_pressed, shift_pressed = key_events[-1] if len(key_events) > 0 else (0,0,False,False,False)

            if (key == ord('\n') or key == ord('\r') or key == 27) and not close_requested:
                # 回车 / Esc -> 请求保存并退出
                io.log_info('收到退出按键（Enter/Esc），正在保存并退出…')
                close_requested = True
                close_requested_time = time.time()
                update_preview = True
                s2c.put ( {'op': 'close'} )
            elif (not close_requested) and key == ord('s'):
                s2c.put ( {'op': 'save'} )
            elif (not close_requested) and key == ord('b'):
                s2c.put ( {'op': 'backup'} )
            elif (not close_requested) and key == ord('p'):
                if not is_waiting_preview:
                    is_waiting_preview = True
                    s2c.put ( {'op': 'preview'} )
            elif (not close_requested) and key == ord('l'):
                if show_last_history_iters_count == 10000:
                    show_last_history_iters_count = 50000
                elif show_last_history_iters_count == 50000:
                    show_last_history_iters_count = 100000
                elif show_last_history_iters_count == 100000:
                    show_last_history_iters_count = 0
                elif show_last_history_iters_count == 0:
                    show_last_history_iters_count = 10000
                update_preview = True
            elif (not close_requested) and key == ord(' '):
                selected_preview = (selected_preview + 1) % len(previews)
                update_preview = True

            # WebUI 远程控制事件
            if _webui is not None:
                if _webui.is_close_requested() and not close_requested:
                    io.log_info('收到WebUI退出请求，正在保存并退出…')
                    close_requested = True
                    close_requested_time = time.time()
                    update_preview = True
                    _webui.clear_close_request()
                    _webui.clear_save_request()  # close already saves, avoid redundant save op
                    s2c.put({'op': 'close'})
                elif _webui.is_save_requested():
                    s2c.put({'op': 'save'})
                    _webui.clear_save_request()
                # WebUI 请求预览 → 立即清掉请求标记，触发训练线程生成新预览
                if _webui.is_preview_requested():
                    _webui.clear_preview_request()
                    if not is_waiting_preview:
                        is_waiting_preview = True
                        _webui_preview_pending = True
                        s2c.put({'op': 'preview'})

            try:
                io.process_messages(0.03)
            except KeyboardInterrupt:
                s2c.put ( {'op': 'close'} )

        io.destroy_all_windows()