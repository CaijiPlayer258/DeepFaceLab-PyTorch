import colorsys
from copy import deepcopy
import inspect
import json
import multiprocessing
import operator
import os
import pickle
import shutil
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np

from core import imagelib, pathex
from core.cv2ex import *
from core.interact import interact as io
from core.leras import nn
from samplelib import SampleGeneratorBase


class ModelBase(object):
    @staticmethod
    def _disp_len(s):
        """Terminal display width of string s (CJK chars count as 2)."""
        w = 0
        for c in s:
            cp = ord(c)
            if cp >= 0x4E00 and cp <= 0x9FFF:            # CJK Unified Ideographs
                w += 2
            elif cp >= 0x3400 and cp <= 0x4DBF:           # CJK Extension A
                w += 2
            elif cp >= 0x20000 and cp <= 0x2FFFD:         # CJK Extension B
                w += 2
            elif cp >= 0xF900 and cp <= 0xFAFF:           # CJK Compatibility
                w += 2
            elif cp >= 0x3000 and cp <= 0x303F:           # CJK Symbols/Punctuation
                w += 2
            elif cp >= 0xFF01 and cp <= 0xFF60:           # Fullwidth Forms
                w += 2
            elif cp >= 0x2E80 and cp <= 0x2FDF:           # CJK Radicals / Kangxi
                w += 2
            else:
                w += 1
        return w

    def __init__(self, is_training=False,
                       is_exporting=False,
                       saved_models_path=None,
                       training_data_src_path=None,
                       training_data_dst_path=None,
                       pretraining_data_path=None,
                       pretrained_model_path=None,
                       no_preview=False,
                       force_model_name=None,
                       force_gpu_idxs=None,
                       cpu_only=False,
                       debug=False,
                       force_model_class_name=None,
                       silent_start=False,
                       **kwargs):
        self.is_training = is_training
        self.is_exporting = is_exporting
        self.saved_models_path = saved_models_path
        self.training_data_src_path = training_data_src_path
        self.training_data_dst_path = training_data_dst_path
        self.pretraining_data_path = pretraining_data_path
        self.pretrained_model_path = pretrained_model_path
        self.no_preview = no_preview
        self.silent_start = silent_start
        self.debug = debug

        self.model_class_name = model_class_name = Path(inspect.getmodule(self).__file__).parent.name.rsplit("_", 1)[1]

        def _extract_base_model_name_from_data_dat(filename: str):
            """Extract base model name from '<base>_<class>_data.dat'.

            Base names may contain underscores. Do NOT use split('_')[0].
            """
            suffix = f"_{model_class_name}_data.dat"
            if filename.endswith(suffix):
                return filename[: -len(suffix)]
            return None

        if force_model_class_name is None:
            if force_model_name is not None:
                self.model_name = force_model_name
            else:
                while True:
                    # gather all model dat files
                    saved_models_names = []
                    for filepath in pathex.get_file_paths(saved_models_path):
                        filepath_name = filepath.name
                        if filepath_name.endswith(f'{model_class_name}_data.dat'):
                            base = _extract_base_model_name_from_data_dat(filepath_name)
                            if base is not None:
                                saved_models_names += [(base, os.path.getmtime(filepath))]

                    # sort by modified datetime
                    saved_models_names = sorted(saved_models_names, key=operator.itemgetter(1), reverse=True)
                    saved_models_names = [x[0] for x in saved_models_names]

                    if len(saved_models_names) != 0:
                        if silent_start:
                            self.model_name = saved_models_names[0]
                            io.log_info(f'静默启动：已选择模型 "{self.model_name}"')
                        else:
                            io.log_info("请选择一个已保存的模型，或输入名称以创建新模型。")
                            io.log_info("[r]：重命名")
                            io.log_info("[d]：删除")
                            io.log_info("")
                            for i, model_name in enumerate(saved_models_names):
                                s = f"[{i}] : {model_name} "
                                if i == 0:
                                    s += "- 最新"
                                io.log_info(s)

                            inp = io.input_str(f"", "0", show_default_value=False)
                            model_idx = -1
                            try:
                                model_idx = np.clip(int(inp), 0, len(saved_models_names) - 1)
                            except Exception:
                                pass

                            if model_idx == -1:
                                if len(inp) == 1:
                                    is_rename = inp[0] == 'r'
                                    is_delete = inp[0] == 'd'

                                    if is_rename or is_delete:
                                        if len(saved_models_names) != 0:
                                            if is_rename:
                                                name = io.input_str(f"请输入要重命名的模型名称")
                                            elif is_delete:
                                                name = io.input_str(f"请输入要删除的模型名称")

                                            if name in saved_models_names:
                                                if is_rename:
                                                    new_model_name = io.input_str(f"请输入模型的新名称")

                                                old_prefix = f"{name}_{model_class_name}_"
                                                for filepath in pathex.get_paths(saved_models_path):
                                                    filepath_name = filepath.name
                                                    if not filepath_name.startswith(old_prefix):
                                                        continue

                                                    remain_filename = filepath_name[len(name) + 1:]
                                                    if is_rename:
                                                        new_filepath = filepath.parent / (new_model_name + '_' + remain_filename)
                                                        filepath.rename(new_filepath)
                                                    elif is_delete:
                                                        filepath.unlink()
                                        continue

                                self.model_name = inp
                            else:
                                self.model_name = saved_models_names[model_idx]

                    else:
                        self.model_name = io.input_str(f"未找到已保存的模型。请输入新模型名称", "new")
                        self.model_name = self.model_name.replace('_', ' ')
                    break

            # Users may input full name (e.g. MyModel_SAEHD). Avoid creating
            # MyModel_SAEHD_SAEHD which would look like "lost" progress.
            if not self.model_name.endswith('_' + self.model_class_name):
                self.model_name = self.model_name + '_' + self.model_class_name
        else:
            self.model_name = force_model_class_name

        self.iter = 0
        self.options = {}
        self.options_show_override = {}
        self.loss_history = []
        self.sample_for_preview = None
        self.choosed_gpu_indexes = None

        model_data = {}
        self.model_data_path = Path(self.get_strpath_storage_for_file('data.dat'))
        if self.model_data_path.exists():
            io.log_info(f"正在加载模型：{self.model_name} ...")
            model_data = pickle.loads(self.model_data_path.read_bytes())
            self.iter = model_data.get('iter', 0)
            if self.iter != 0:
                self.options = model_data['options']
                self.loss_history = model_data.get('loss_history', [])
                self.sample_for_preview = model_data.get('sample_for_preview', None)
                self.choosed_gpu_indexes = model_data.get('choosed_gpu_indexes', None)

        if self.is_first_run():
            io.log_info("\n模型首次运行。")

        if silent_start:
            self.device_config = nn.DeviceConfig.CPU() if cpu_only else nn.DeviceConfig.BestGPU()
            io.log_info(f"静默启动：已选择设备 {'CPU' if self.device_config.cpu_only else self.device_config.devices[0].name}")
        else:
            self.device_config = nn.DeviceConfig.GPUIndexes(force_gpu_idxs or nn.ask_choose_device_idxs(suggest_best_multi_gpu=True)) \
                                if not cpu_only else nn.DeviceConfig.CPU()

        nn.initialize(self.device_config)

        ####
        self.default_options_path = saved_models_path / f'{self.model_class_name}_default_options.dat'
        self.default_options = {}
        if self.default_options_path.exists():
            try:
                self.default_options = pickle.loads(self.default_options_path.read_bytes())
            except Exception:
                pass

        self.choose_preview_history = False
        self.batch_size = self.load_or_def_option('batch_size', 1)
        #####

        io.input_skip_pending()
        self.on_initialize_options()

        if self.is_first_run():
            # save as default options only for first run model initialize
            self.default_options_path.write_bytes(pickle.dumps(self.options))

        self.crash_threshold = self.options.get('crash_threshold', 0.0)
        self._crash_detected = False
        self.write_preview_history = self.options.get('write_preview_history', False)
        self.target_iter = self.options.get('target_iter', 0)
        self.random_flip = self.options.get('random_flip', True)
        self.random_src_flip = self.options.get('random_src_flip', False)
        self.random_dst_flip = self.options.get('random_dst_flip', True)

        # 模块信息列表，on_initialize 中由各模型填充
        # 每项: (名称, 参数量, 状态字符串)
        self._module_info_list = []

        self.on_initialize()
        self.options['batch_size'] = self.batch_size

        self.autobackups_path = self.saved_models_path / f'{self.get_model_name()}_autobackups'
        self.preview_history_writer = None
        if self.is_training:
            self.preview_history_path = self.saved_models_path / (f'{self.get_model_name()}_history')

            if self.write_preview_history or io.is_colab():
                if not self.preview_history_path.exists():
                    self.preview_history_path.mkdir(exist_ok=True)
                else:
                    if self.iter == 0:
                        for filename in pathex.get_image_paths(self.preview_history_path):
                            Path(filename).unlink()

            if self.generator_list is None:
                raise ValueError('You didnt set_training_data_generators()')
            else:
                for i, generator in enumerate(self.generator_list):
                    if not isinstance(generator, SampleGeneratorBase):
                        raise ValueError('training data generator is not subclass of SampleGeneratorBase')

            self.update_sample_for_preview(choose_preview_history=self.choose_preview_history)

        io.log_info(self.get_summary_text())

    def update_sample_for_preview(self, choose_preview_history=False, force_new=False):
        if self.sample_for_preview is None or choose_preview_history or force_new:
            if choose_preview_history and io.is_support_windows():
                wnd_name = "[p] - 下一张。[空格] - 切换预览类型。[回车] - 确认。"
                io.log_info(f"为预览历史选择图片。{wnd_name}")
                io.named_window(wnd_name)
                io.capture_keys(wnd_name)
                choosed = False
                preview_id_counter = 0
                while not choosed:
                    self.sample_for_preview = self.generate_next_samples()
                    previews = self.get_history_previews()

                    io.show_image(wnd_name, (previews[preview_id_counter % len(previews)][1] * 255).astype(np.uint8))

                    while True:
                        key_events = io.get_key_events(wnd_name)
                        key, chr_key, ctrl_pressed, alt_pressed, shift_pressed = key_events[-1] if len(key_events) > 0 else (0, 0, False, False, False)
                        if key == ord('\n') or key == ord('\r'):
                            choosed = True
                            break
                        elif key == ord(' '):
                            preview_id_counter += 1
                            break
                        elif key == ord('p'):
                            break

                        try:
                            io.process_messages(0.1)
                        except KeyboardInterrupt:
                            choosed = True

                io.destroy_window(wnd_name)
            else:
                self.sample_for_preview = self.generate_next_samples()

        try:
            self.get_history_previews()
        except Exception:
            self.sample_for_preview = self.generate_next_samples()

        self.last_sample = self.sample_for_preview

    def load_or_def_option(self, name, def_value):
        options_val = self.options.get(name, None)
        if options_val is not None:
            return options_val

        def_opt_val = self.default_options.get(name, None)
        if def_opt_val is not None:
            return def_opt_val

        return def_value

    def ask_override(self):
        if self.silent_start:
            return False
        return self.is_training and self.iter != 0 and io.input_in_time("请在 2 秒内按回车以覆盖模型设置。", 5 if io.is_colab() else 2)

    def ask_backup_interval(self, default_value=500):
        default_backup_interval = self.load_or_def_option('backup_interval', default_value)
        self.options['backup_interval'] = io.input_int("备份间隔（迭代次数）", default_backup_interval, add_info="0=关闭", help_message="每 N 次迭代自动保存并备份模型。0 表示禁用。")

    def ask_max_backups(self, default_value=24):
        default_max_backups = self.load_or_def_option('max_backups', default_value)
        self.options['max_backups'] = io.input_int("最大备份数量", default_max_backups, add_info="1..99", help_message="保留最近 N 份备份，超出则删除最旧的。")

    def ask_crash_threshold(self, default_value=4.0):
        default_crash = self.load_or_def_option('crash_threshold', default_value)
        self.options['crash_threshold'] = io.input_number(f"崩溃阈值", default_crash, add_info="0=关闭, 建议4.0", help_message="当 src 或 dst loss 超过此值时视为训练异常，自动禁用保存与备份。0 表示关闭。")

    def ask_write_preview_history(self, default_value=False):
        default_write_preview_history = self.load_or_def_option('write_preview_history', default_value)
        self.options['write_preview_history'] = io.input_bool(f"写入预览历史", default_write_preview_history, help_message="预览历史将写入 <ModelName>_history 文件夹。")

        if self.options['write_preview_history']:
            if io.is_support_windows():
                self.choose_preview_history = io.input_bool("为预览历史选择图片", False)
            elif io.is_colab():
                self.choose_preview_history = io.input_bool("为预览历史随机选择新图片", False, help_message="如果你用同一模型训练不同人物，预览历史可能一直停留在旧人脸上。除非你确实更换了 src/dst 人物，否则建议选择否。")

    def ask_target_iter(self, default_value=0):
        default_target_iter = self.load_or_def_option('target_iter', default_value)
        self.options['target_iter'] = max(0, io.input_int("目标迭代次数", default_target_iter))

    def ask_random_flip(self):
        default_random_flip = self.load_or_def_option('random_flip', True)
        self.options['random_flip'] = io.input_bool("随机翻转人脸", default_random_flip, help_message="关闭该选项可能使预测人脸更自然，但要求 src faceset 像 dst faceset 一样覆盖足够多的朝向。")

    def ask_random_src_flip(self):
        default_random_src_flip = self.load_or_def_option('random_src_flip', False)
        self.options['random_src_flip'] = io.input_bool("随机翻转 SRC 人脸", default_random_src_flip, help_message="对 SRC faceset 做随机水平翻转，以覆盖更多角度，但可能使人脸看起来稍不自然。")

    def ask_random_dst_flip(self):
        default_random_dst_flip = self.load_or_def_option('random_dst_flip', True)
        self.options['random_dst_flip'] = io.input_bool("随机翻转 DST 人脸", default_random_dst_flip, help_message="对 DST faceset 做随机水平翻转。在未启用 SRC 随机翻转时，有助于提升 src->dst 的泛化。")

    def ask_batch_size(self, suggest_batch_size=None, range=None):
        default_batch_size = self.load_or_def_option('batch_size', suggest_batch_size or self.batch_size)

        batch_size = max(0, io.input_int("Batch_size", default_batch_size, valid_range=range, help_message="更大的 batch size 通常更利于泛化，但可能导致显存不足（OOM）。请根据你的显卡手动调整。"))

        if range is not None:
            batch_size = np.clip(batch_size, range[0], range[1])

        self.options['batch_size'] = self.batch_size = batch_size

    def ask_lr(self, default_value=5e-5):
        default_lr = self.load_or_def_option('lr', default_value)
        self.options['lr'] = io.input_number(
            "学习率 (Learning Rate)",
            default_lr,
            add_info='1e-5 .. 1e-3',
            help_message='训练的学习率。较高的值（如 5e-4）收敛更快但可能不稳定，较低的值（如 5e-5）更稳定但收敛更慢。',
        )

    def ask_lr_scheduler(self, default_value=0):
        """余弦退火·热重启（Cosine Annealing with Warm Restarts, SGDR）。

        术语要分清楚（容易混）：
          余弦退火 (Cosine Annealing)                    —— lr 单调下降，不重启
          余弦退火·热重启 (Cosine Annealing + Warm Restarts) —— lr 周期性弹回峰值

        这里用的是**后者**：每 lr_cos 步把 lr 弹回峰值重开一轮。
        实测在训练后期会反复把模型踢出已收敛的极小值，质量比不开启更差，
        所以建议保持 0。想要单调下降请用 ask_lr_plateau()。
        """
        default_lr_cos = self.load_or_def_option('lr_cos', default_value)
        self.options['lr_cos'] = int(
            io.input_int(
                "余弦退火·热重启周期 (Cosine Annealing Warm Restarts)",
                default_lr_cos,
                add_info='0=关闭, >0=单轮长度',
                help_message='注意这是「热重启」：每 N 步把 lr 弹回峰值重新开始一轮，不是单调退火。'
                             '实测后期会拖累质量，建议保持 0。想要单调下降请用「平台期自动降 lr」。',
            )
        )

    def ask_lr_plateau(self):
        """平台期自动降 lr 的交互式设置。

        为什么需要它：真实训练通常**没有预定总步数**（一直挂着训），
        所以"提前声明退火总步数"这种接口不实用。平台期检测是唯一不需要
        预知总步数的自适应策略 —— 连续 N 步没有实质改善就降一次 lr。

        默认关闭（factor=1.0）：这是会改变训练动态的功能，不静默替用户开启。
        想用就在训练配置页把「平台期自动降 lr」的衰减系数选成 0.5 之类。
        """
        d_factor = self.load_or_def_option('lr_plateau_factor', 1.0)
        try:
            d_factor = float(d_factor)
        except (TypeError, ValueError):
            d_factor = 1.0
        if d_factor > 0.0 and io.input_bool(
                "启用平台期自动降 lr（连续无改善则衰减学习率）", d_factor < 1.0,
                help_message='不依赖预定总步数。连续 N 步 loss 没有实质改善时把 lr 乘以衰减系数。'
                             '适合"一直挂着训"的场景。'):
            self.options['lr_plateau_factor'] = float(
                io.input_number("衰减系数", d_factor if d_factor < 1.0 else 0.5, add_info='0.1..0.9'))
            self.options['lr_plateau_patience'] = int(
                io.input_int("耐心步数", self.load_or_def_option('lr_plateau_patience', 0),
                             add_info='0=自动'))
            self.options['lr_plateau_min_ratio'] = float(
                io.input_number("学习率下限（占初始值的比例）",
                                self.load_or_def_option('lr_plateau_min_ratio', 0.1),
                                add_info='0.01..0.5'))
        else:
            self.options['lr_plateau_factor'] = 1.0
            self.options['lr_plateau_patience'] = int(
                self.load_or_def_option('lr_plateau_patience', 0) or 0)
            self.options['lr_plateau_min_ratio'] = 0.1

    def ask_gradient_checkpointing(self, default_value=False):
        default_grad_ckpt = self.load_or_def_option('gradient_checkpointing', default_value)
        self.options['gradient_checkpointing'] = io.input_bool(
            "启用梯度检查点 (Gradient Checkpointing)",
            default_grad_ckpt,
            help_message='以更多计算量为代价减少显存占用。在显存不足时可启用，会略微降低训练速度。',
        )

    #overridable
    def on_initialize_options(self):
        pass

    #overridable
    def on_initialize(self):
        '''
        initialize your models

        store and retrieve your model options in self.options['']

        check example
        '''
        pass

    #overridable
    def onSave(self):
        #save your models here
        pass

    #overridable
    def onTrainOneIter(self, sample, generator_list):
        #train your models here

        #return array of losses
        return ( ('loss_src', 0), ('loss_dst', 0) )

    #overridable
    def onGetPreview(self, sample, for_history=False):
        #you can return multiple previews
        #return [ ('preview_name',preview_rgb), ... ]
        return []

    #overridable if you want model name differs from folder name
    def get_model_name(self):
        return self.model_name

    #overridable , return [ [model, filename],... ]  list
    def get_model_filename_list(self):
        return []

    #overridable
    def get_MergerConfig(self):
        #return predictor_func, predictor_input_shape, MergerConfig() for the model
        raise NotImplementedError

    def get_pretraining_data_path(self):
        return self.pretraining_data_path

    def get_target_iter(self):
        return self.target_iter

    def is_reached_iter_goal(self):
        return self.target_iter != 0 and self.iter >= self.target_iter

    def get_previews(self):
        return self.onGetPreview(self.last_sample)

    def get_history_previews(self):
        return self.onGetPreview(self.sample_for_preview, for_history=True)

    def get_preview_history_writer(self):
        if self.preview_history_writer is None:
            self.preview_history_writer = PreviewHistoryWriter()
        return self.preview_history_writer

    def _ensure_disk_space(self, min_free_gb=1.0):
        """若磁盘空间不足，删除最旧的备份直到空间达标。"""
        if not self.autobackups_path.exists():
            return
        for attempt in range(99):
            total, used, free = shutil.disk_usage(self.saved_models_path)
            if free >= min_free_gb * (1024**3):
                break
            backup_dirs = sorted([
                d for d in self.autobackups_path.iterdir()
                if d.is_dir() and d.name.isdigit()
            ])
            if not backup_dirs:
                break
            oldest = backup_dirs[-1]  # 最高编号 = 最旧备份
            io.log_info(f'磁盘空间不足 ({free/(1024**3):.1f}GB)，删除旧备份 {oldest.name}')
            pathex.delete_all_files(oldest)

    def save(self):
        if self._crash_detected:
            io.log_info(f'[Crash Detector] 检测到训练崩溃，取消本次自动备份')
            return

        self._ensure_disk_space()

        Path(self.get_summary_path()).write_text(self.get_summary_text())

        # Save .dat first — preserves current settings even if archiving fails
        model_data = {
            'iter': self.iter,
            'options': self.options,
            'loss_history': self.loss_history,
            'sample_for_preview': self.sample_for_preview,
            'choosed_gpu_indexes': self.choosed_gpu_indexes,
        }
        pathex.write_bytes_safe(self.model_data_path, pickle.dumps(model_data))

        # Write .meta_cache.json for fast GUI loading（不包含 loss_history）
        try:
            _meta = {
                'iter': self.iter,
                'options': {k: v for k, v in self.options.items()
                            if not k.startswith('_') and k not in ('loss_history',)},
            }
            _meta_path = self.model_data_path.with_suffix('.meta_cache.json')
            import json
            _meta_path.write_text(json.dumps(_meta, ensure_ascii=False, default=str), encoding='utf-8')
        except Exception:
            pass

        # Archive old .npy weight files to old/ after conversion to .pth
        model_root = self.get_model_root_path()
        if model_root.exists():
            npy_files = list(model_root.glob(f"{self.get_model_name()}_*.npy"))
            if npy_files:
                old_dir = model_root / "old"
                old_dir.mkdir(exist_ok=True)
                for npy in npy_files:
                    dest = old_dir / npy.name
                    if not dest.exists():
                        shutil.move(str(npy), str(dest))
                # 同时备份 _data.dat 到 old/
                dat_path = self.model_data_path
                if dat_path.exists():
                    dat_dest = old_dir / dat_path.name
                    if not dat_dest.exists():
                        shutil.copy2(str(dat_path), str(dat_dest))
                io.log_info(f"  {len(npy_files)} old .npy files archived to old/")

        self.onSave()

    def create_backup(self):
        if self._crash_detected:
            io.log_info(f'[Crash Detector] 检测到训练崩溃，取消本次自动备份')
            return

        self._ensure_disk_space()

        io.log_info("正在创建备份...", end='\r')

        if not self.autobackups_path.exists():
            self.autobackups_path.mkdir(exist_ok=True)

        n = getattr(self, 'max_backups', 24)

        bckp_filename_list = [self.get_strpath_storage_for_file(filename) for _, filename in self.get_model_filename_list()]
        bckp_filename_list += [str(self.get_summary_path()), str(self.model_data_path)]

        for i in range(n, 0, -1):
            idx_str = '%.2d' % i
            next_idx_str = '%.2d' % (i + 1)

            idx_backup_path = self.autobackups_path / idx_str
            next_idx_packup_path = self.autobackups_path / next_idx_str

            try:
                if idx_backup_path.exists():
                    if i == n:
                        pathex.delete_all_files(idx_backup_path)
                    else:
                        next_idx_packup_path.mkdir(exist_ok=True)
                        pathex.move_all_files(idx_backup_path, next_idx_packup_path)
            except (OSError, PermissionError):
                pass  # file locked by another process, skip this rotation step

            if i == 1:
                try:
                    idx_backup_path.mkdir(exist_ok=True)
                    for filename in bckp_filename_list:
                        if Path(filename).exists():
                            try:
                                shutil.copy(str(filename), str(idx_backup_path / Path(filename).name))
                            except (OSError, PermissionError):
                                pass  # source file locked, skip
                except (OSError, PermissionError):
                    pass  # can't create backup dir, skip

                try:
                    previews = self.get_previews()
                    plist = []
                    for i in range(len(previews)):
                        name, bgr = previews[i]
                        plist += [(bgr, idx_backup_path / ('preview_%s.jpg' % (name)))]
                    if len(plist) != 0:
                        self.get_preview_history_writer().post(plist, self.loss_history, self.iter)
                except Exception:
                    pass  # preview save failed, non-critical

    def debug_one_iter(self):
        images = []
        for generator in self.generator_list:
            for i, batch in enumerate(next(generator)):
                if len(batch.shape) == 4:
                    images.append(batch[0])

        return imagelib.equalize_and_stack_square(images)

    def generate_next_samples(self):
        sample = []
        filenames = []
        for i, generator in enumerate(self.generator_list):
            side = 'src' if i == 0 else 'dst'
            if not generator.is_initialized():
                raise RuntimeError(
                    f"训练数据加载失败（{side}）：生成器未就绪。\n"
                    f"请检查 {side} 人脸集路径是否存在且包含有效的对齐脸文件。"
                )
            gen_result = generator.generate_next()
            if gen_result is None or (isinstance(gen_result, (list, tuple)) and len(gen_result) == 0):
                raise RuntimeError(
                    f"训练数据加载失败（{side}）：生成器返回空数据。\n"
                    f"可能原因：\n"
                    f"  1. {side} 路径为空\n"
                    f"  2. 路径下无有效的 DFL 对齐脸 JPG 文件\n"
                    f"  3. 所有图片都被拒绝（参见前置错误日志）\n"
                    f"请检查 {side} 人脸集路径。"
                )
            sample.append(deepcopy(gen_result))
            if hasattr(generator, 'get_last_filenames'):
                filenames.append(generator.get_last_filenames())
            else:
                filenames.append([])
        self.last_sample = sample
        self.last_filenames = filenames
        return sample

    #overridable
    def should_save_preview_history(self):
        return (not io.is_colab() and self.iter % 10 == 0) or (io.is_colab() and self.iter % 100 == 0)

    def train_one_iter(self):
        iter_time = time.time()
        losses = self.onTrainOneIter()
        iter_time = time.time() - iter_time

        self.loss_history.append([float(loss[1]) for loss in losses])

        # crash detection: check last 100 iterations for loss spikes
        ct = self.options.get('crash_threshold', 0.0)
        if ct > 0.0:
            recent = self.loss_history[-100:]
            self._crash_detected = any(max(l) > ct for l in recent)

        if self.should_save_preview_history():
            plist = []

            if io.is_colab():
                previews = self.get_previews()
                for i in range(len(previews)):
                    name, bgr = previews[i]
                    plist += [(bgr, self.get_strpath_storage_for_file('preview_%s.jpg' % (name)))]

            if self.write_preview_history:
                previews = self.get_history_previews()
                for i in range(len(previews)):
                    name, bgr = previews[i]
                    path = self.preview_history_path / name
                    plist += [(bgr, str(path / (f'{self.iter:07d}.jpg')))]
                    if not io.is_colab():
                        plist += [(bgr, str(path / ('_last.jpg')))]

            if len(plist) != 0:
                self.get_preview_history_writer().post(plist, self.loss_history, self.iter)

        self.iter += 1

        return self.iter, iter_time

    def pass_one_iter(self):
        self.generate_next_samples()

    def finalize(self):
        # Stop sample generators that may own multiprocessing workers/queues.
        try:
            gl = getattr(self, 'generator_list', None)
            if gl is not None:
                for g in gl:
                    try:
                        if hasattr(g, 'close'):
                            g.close()
                    except Exception:
                        pass
        except Exception:
            pass
        nn.close_session()

    def is_first_run(self):
        return self.iter == 0

    def is_debug(self):
        return self.debug

    def set_batch_size(self, batch_size):
        self.batch_size = batch_size

    def get_batch_size(self):
        return self.batch_size

    def get_iter(self):
        return self.iter

    def set_iter(self, iter):
        self.iter = iter
        self.loss_history = self.loss_history[:iter]

    def get_loss_history(self):
        return self.loss_history

    def set_training_data_generators(self, generator_list):
        self.generator_list = generator_list

    def get_training_data_generators(self):
        return self.generator_list

    def get_model_root_path(self):
        return self.saved_models_path

    def get_strpath_storage_for_file(self, filename):
        return str(self.saved_models_path / (self.get_model_name() + '_' + filename))

    def get_summary_path(self):
        return self.get_strpath_storage_for_file('summary.txt')

    def get_summary_text(self):
        visible_options = self.options.copy()
        visible_options.update(self.options_show_override)

        # ---- display-width helpers (CJK-aware) ----
        def _dw(s):
            return ModelBase._disp_len(s)

        def _ljust(s, w):
            return s + ' ' * max(0, w - _dw(s))

        def _rjust(s, w):
            return ' ' * max(0, w - _dw(s)) + s

        def _center(s, w):
            p = max(0, w - _dw(s))
            return ' ' * (p // 2) + s + ' ' * (p - p // 2)

        # Wrap content in ==...==, padding to `w` display-width
        def _line(content='', w=None):
            if w is None:
                w = inner_w
            return f'=={content}{" " * max(0, w - _dw(content))}=='

        summary_text = []

        # ============================================================
        #   Option categories
        # ============================================================
        arch_opts = [
            ('model_name', '模型名称'),
            ('resolution', '分辨率'),
            ('face_type', '人脸类型'),
            ('archi', '架构'),
            ('ae_dims', 'AE维度'),
            ('e_dims', '编码器维'),
            ('d_dims', '解码器维'),
            ('d_mask_dims', '遮罩维'),
        ]
        aug_opts = [
            ('random_warp', '随机形变'),
            ('random_hsv_power', 'HSV强度'),
            ('ct_mode', '色彩迁移'),
            ('random_src_flip', 'SRC翻转'),
            ('random_dst_flip', 'DST翻转'),
            ('masked_training', '遮罩训练'),
            ('blur_out_mask', '遮罩羽化'),
            ('eyes_mouth_prio', '嘴眼优先'),
            ('uniform_yaw', '均匀偏航'),
        ]
        patch_opts = [
            ('gan_power', 'GAN强度'),
            ('true_face_power', '真脸强度'),
            ('face_style_power', '面部风格'),
            ('bg_style_power', '背景风格'),
            ('vgg_perceptual_power', 'VGG感知损失'),
            ('gan_patch_size', 'GAN块大小'),
            ('gan_dims', 'GAN维度'),
        ]
        train_opts = [
            ('optimizer', '优化器'),
            ('clipgrad', '梯度裁剪'),
            ('crash_threshold', '崩溃阈值'),
            ('use_bf16', 'BF16混合'),
            ('batch_size', '批次大小'),
            ('lr_dropout', '学习率策略'),
            ('models_opt_on_gpu', '优化器GPU'),
            ('use_fast_generator', '快速生成器'),
            ('pretrain', '预训练'),
            ('pretrain_single_decoder', '单解码器'),
            ('write_preview_history', '预览历史'),
            ('target_iter', '目标迭代'),
            ('backup_interval', '备份间隔（迭代）'),
            ('max_backups', '最大备份数'),
            ('lr', '学习率'),
            ('use_gradient_checkpointing', '梯度检查点'),
            ('lr_scheduler', 'LR调度器'),
            ('lr_cos_period', '余弦周期'),
        ]

        categories = [
            ('模型架构', arch_opts),
            ('数据增强', aug_opts),
            ('补丁/损失', patch_opts),
            ('训练设置', train_opts),
        ]

        col_contents = []
        for cat_name, opt_list in categories:
            items = []
            for key, label in opt_list:
                if key == 'model_name':
                    val = self.get_model_name()
                else:
                    val = visible_options.get(key)
                if val is not None:
                    items.append((label, str(val)))
            col_contents.append(items)

        # ---- column widths in display-width units ----
        MIN_COL_DW = 18
        col_dws = []
        for items in col_contents:
            if items:
                max_dw = max(_dw(f'{k}: {v}') for k, v in items)
                col_dws.append(max(max_dw + 2, MIN_COL_DW))
            else:
                col_dws.append(MIN_COL_DW)

        n_cols = len(categories)
        sep_dw = _dw(' | ')  # 3
        pad = '  '
        pad_dw = _dw(pad)    # 2
        inner_w = sum(col_dws) + sep_dw * (n_cols - 1) + pad_dw * 2
        inner_w = max(inner_w, 72)

        # ============================================================
        #   Header
        # ============================================================
        summary_text += [_line(_center(' 模型摘要 ', inner_w), inner_w)]
        hdr = f'模型名称: {self.get_model_name()}  |  当前迭代: {self.get_iter()}'
        summary_text += [_line(_center(hdr, inner_w), inner_w)]
        summary_text += [_line('', inner_w)]

        sep_heavy = '-' * inner_w
        summary_text += [_line(sep_heavy, inner_w)]
        summary_text += [_line('', inner_w)]

        # ============================================================
        #   4-column options table
        # ============================================================
        summary_text += [_line(sep_heavy, inner_w)]
        summary_text += [_line('', inner_w)]

        col_sep = ' | '
        # Category headers (centered in each column)
        cat_hdr_parts = []
        for i, (name, _) in enumerate(categories):
            cat_hdr_parts.append(_center(name, col_dws[i]))
        cat_hdr = col_sep.join(cat_hdr_parts)
        summary_text += [_line(pad + cat_hdr + pad, inner_w)]

        # Sub-header rules
        sub_parts = ['-' * dw for dw in col_dws]
        sub_rule = col_sep.join(sub_parts)
        summary_text += [_line(pad + sub_rule + pad, inner_w)]

        # Content rows
        max_rows = max(len(c) for c in col_contents)
        for row_idx in range(max_rows):
            parts = []
            for col_idx, items in enumerate(col_contents):
                if row_idx < len(items):
                    k, v = items[row_idx]
                    cell_text = f'{k}: {v}'
                    parts.append(_ljust(cell_text, col_dws[col_idx]))
                else:
                    parts.append(' ' * col_dws[col_idx])
            row_str = col_sep.join(parts)
            summary_text += [_line(pad + row_str + pad, inner_w)]

        summary_text += [_line('', inner_w)]
        summary_text += [_line(sep_heavy, inner_w)]
        summary_text += [_line('', inner_w)]

        # ============================================================
        #   Module status (key:value columns, same style as options)
        # ============================================================
        if self._module_info_list:
            # Separate modules from optimizers
            mods = [(n, p, s) for n, p, s in self._module_info_list if not n.endswith('_opt')]
            opts = [(n, p, s) for n, p, s in self._module_info_list if n.endswith('_opt')]

            col_data = []
            # Column 1: module name + param count
            c1 = []
            for name, params, status in mods:
                c1.append(f'名称: {name}')
            col_data.append(('模块', c1))

            # Column 2: parameters
            c2 = []
            for name, params, status in mods:
                c2.append(f'参数量: {params:,}')
            col_data.append(('参数', c2))

            # Column 3: precision
            c3 = []
            for name, params, status in mods:
                pl = 'FP32'
                c3.append(f'精度: {pl}')
            col_data.append(('精度', c3))

            # Column 4: status
            c4 = []
            for name, params, status in mods:
                c4.append(f'状态: {status}')
            col_data.append(('状态', c4))

            # Render as 4-column key:value table matching options style
            ncols = len(col_data)
            col_dws = []
            for _, items in col_data:
                dw = max(_dw(item) for item in items) if items else 10
                col_dws.append(dw)

            total_col_w = sum(col_dws) + (ncols - 1) * _dw(col_sep) + 2 * _dw(pad)
            mod_total_w = max(total_col_w, inner_w)
            # Distribute extra width so module table fills the full inner width
            if mod_total_w > total_col_w:
                avail = mod_total_w - (ncols - 1) * _dw(col_sep) - 2 * _dw(pad)
                extra = avail - sum(col_dws)
                if extra > 0:
                    for i in range(ncols):
                        col_dws[i] += extra // ncols + (1 if i < extra % ncols else 0)

            summary_text += [_line(_center(' 模块状态 ', mod_total_w), mod_total_w)]
            summary_text += [_line('', mod_total_w)]

            cat_hdr_parts = []
            for i, (name, _) in enumerate(col_data):
                cat_hdr_parts.append(_center(name, col_dws[i]))
            cat_hdr = col_sep.join(cat_hdr_parts)
            summary_text += [_line(pad + cat_hdr + pad, mod_total_w)]

            sub_parts = ['-' * dw for dw in col_dws]
            sub_rule = col_sep.join(sub_parts)
            summary_text += [_line(pad + sub_rule + pad, mod_total_w)]

            max_rows = max(len(items) for _, items in col_data)
            for row_idx in range(max_rows):
                parts = []
                for col_idx, (_, items) in enumerate(col_data):
                    if row_idx < len(items):
                        parts.append(_ljust(items[row_idx], col_dws[col_idx]))
                    else:
                        parts.append(' ' * col_dws[col_idx])
                row_str = col_sep.join(parts)
                summary_text += [_line(pad + row_str + pad, mod_total_w)]

            summary_text += [_line('', mod_total_w)]
            summary_text += [_line(sep_heavy, mod_total_w)]
            summary_text += [_line('', mod_total_w)]

            total_params = sum(p for _, p, _ in mods)
            opt_params = sum(p for _, p, _ in opts)
            precision_str = 'FP32'
            compute_mode = 'BF16' if getattr(self, 'use_bf16', False) else 'FP32'

            # VRAM estimation — per-module based on actual param counts
            batch_size = self.options.get('batch_size', 1)
            resolution = self.options.get('resolution', 128)

            model_total_params = 0
            w_bytes = 0
            g_bytes = 0
            o_bytes = 0
            for name, pcount, _ in self._module_info_list:
                if name.endswith('_opt'):
                    o_bytes += pcount * 4  # optimizer states (ms+vs) in FP32
                else:
                    model_total_params += pcount
                    w_bytes += pcount * 4  # model weights in FP32
                    g_bytes += pcount * 4  # gradients in FP32 (same dtype as weights)

            w_mb = w_bytes / (1048576.0)
            g_mb = g_bytes / (1048576.0)
            o_mb = o_bytes / (1048576.0)
            i_mb = 2 * batch_size * 3 * resolution * resolution * 4 / (1048576.0)
            # activation memory: intermediate feature maps retained for backward
            # Scales with model size × src+dst paths × batch × resolution
            gc_enabled = bool(self.options.get('use_gradient_checkpointing', False))
            # activation memory: scales with model size × src+dst paths × batch × resolution
            act_mult = 1.75 if gc_enabled else 3.5  # checkpoint ~50% less activation
            a_mb = w_mb * act_mult * (batch_size / 4) * (resolution / 256) ** 2
            # CUDA system overhead (context + allocator cache)
            c_mb = 900
            t_mb = w_mb + o_mb + g_mb + i_mb + a_mb + c_mb

            total_line = f'  总计: {model_total_params:,} 训练参数 | 存储: {precision_str} 计算: {compute_mode}'
            summary_text += [_line(total_line, mod_total_w)]
            vram_fmt = (f'  预估显存: ~{t_mb:.0f} MB '
                        f'(权重:{w_mb:.0f} + 优化器:{o_mb:.0f} + '
                        f'梯度:{g_mb:.0f} + 输入:{i_mb:.0f} + '
                        f'激活:{a_mb:.0f} + 系统:{c_mb:.0f})')
            summary_text += [_line(vram_fmt, mod_total_w)]
            if self._crash_detected:
                crash_warn = '  [崩溃] 自动保存与备份已禁用! loss 超过阈值'
                summary_text += [_line(crash_warn, mod_total_w)]
            summary_text += [_line('', mod_total_w)]

            if mod_total_w > inner_w:
                inner_w = mod_total_w

        # ============================================================
        #   Device info
        # ============================================================
        dev_w = max(inner_w, 72)
        summary_text += [_line(_center(' 运行设备 ', dev_w), dev_w)]
        summary_text += [_line('', dev_w)]
        if len(self.device_config.devices) == 0:
            summary_text += [_line('  使用设备: CPU', dev_w)]
        else:
            for device in self.device_config.devices:
                vram_str = f'{device.total_mem_gb:.2f}GB'
                dl = f'  设备: {device.name}  |  显存: {vram_str}  |  设备索引: {device.index}'
                summary_text += [_line(dl, dev_w)]
        summary_text += [_line('', dev_w)]
        summary_text += [_line('=' * dev_w, dev_w)]
        summary_text = "\n".join(summary_text)
        return summary_text

    @staticmethod
    def get_loss_history_preview(loss_history, iter, w, c):
        loss_history = np.array(loss_history.copy())

        lh_height = 100
        lh_img = np.ones((lh_height, w, c)) * 0.1

        if len(loss_history) != 0:
            loss_count = len(loss_history[0])
            lh_len = len(loss_history)

            l_per_col = lh_len / w

            # Compute mean value per column per loss series
            plist = np.zeros((w, loss_count), dtype=np.float32)
            for col in range(w):
                i_start = int(col * l_per_col)
                i_end = int((col + 1) * l_per_col)
                if i_start >= i_end:
                    idx = min(i_start, lh_len - 1)
                    for p in range(loss_count):
                        plist[col, p] = float(loss_history[idx][p])
                else:
                    for p in range(loss_count):
                        plist[col, p] = float(np.mean([loss_history[i][p]
                                                        for i in range(i_start, i_end)]))

            # Robust max scale
            recent = loss_history[max(0, lh_len // 5):]
            if len(recent) > 0:
                plist_abs_max = float(np.mean(recent)) * 2.0
            else:
                plist_abs_max = 1.0
            if plist_abs_max < 1e-8 or not np.isfinite(plist_abs_max):
                plist_abs_max = 1.0

            # Draw connected line plot for each loss series
            for p in range(loss_count):
                rgb = colorsys.hsv_to_rgb(p * (1.0 / loss_count), 1.0, 1.0)
                # cv2 uses BGR; convert
                line_color = (float(rgb[2]), float(rgb[1]), float(rgb[0]))
                if c > 3:
                    line_color = line_color + (1.0,)

                prev_y = None
                for col in range(w):
                    val = plist[col, p]
                    if not np.isfinite(val):
                        val = 0.0
                    ph = val / plist_abs_max * (lh_height - 1)
                    ph = np.clip(ph, 0, lh_height - 1)
                    y = int(lh_height - ph - 1)

                    if prev_y is not None:
                        cv2.line(lh_img, (col - 1, prev_y), (col, y), line_color, 1)
                    prev_y = y

        lh_lines = 5
        lh_line_height = (lh_height - 1) / lh_lines
        for i in range(0, lh_lines + 1):
            lh_img[int(i * lh_line_height), :] = (0.8,) * c

        last_line_t = int((lh_lines - 1) * lh_line_height)
        last_line_b = int(lh_lines * lh_line_height)

        lh_text = 'Iter: %d' % (iter) if iter != 0 else ''

        lh_img[last_line_t:last_line_b, 0:w] += imagelib.get_text_image((last_line_b - last_line_t, w, c), lh_text, color=[0.8] * c)
        return lh_img


class PreviewHistoryWriter():
    def __init__(self):
        self.sq = multiprocessing.Queue()
        self.p = multiprocessing.Process(target=self.process, args=(self.sq,))
        self.p.daemon = True
        self.p.start()

    def process(self, sq):
        while True:
            while not sq.empty():
                plist, loss_history, iter = sq.get()

                preview_lh_cache = {}
                for preview, filepath in plist:
                    filepath = Path(filepath)
                    i = (preview.shape[1], preview.shape[2])

                    preview_lh = preview_lh_cache.get(i, None)
                    if preview_lh is None:
                        preview_lh = ModelBase.get_loss_history_preview(loss_history, iter, preview.shape[1], preview.shape[2])
                        preview_lh_cache[i] = preview_lh

                    img = (np.concatenate([preview_lh, preview], axis=0) * 255).astype(np.uint8)

                    filepath.parent.mkdir(parents=True, exist_ok=True)
                    cv2_imwrite(filepath, img)

            time.sleep(0.01)

    def post(self, plist, loss_history, iter):
        self.sq.put((plist, loss_history, iter))

    # disable pickling
    def __getstate__(self):
        return dict()

    def __setstate__(self, d):
        self.__dict__.update(d)
