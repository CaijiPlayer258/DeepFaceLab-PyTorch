import colorsys
from copy import deepcopy
import csv
import inspect
import json
import multiprocessing
import operator
import os
import pickle
import shutil
import tempfile
import threading
import time
from pathlib import Path

import cv2
import numpy as np

from core import imagelib, pathex
from core.cv2ex import *
from core.interact import interact as io
from core.leras import nn
from samplelib import SampleGeneratorBase


# ============================================================
# data.dat 参数缓存
# ============================================================
# 原版把 iter / options / loss_history / sample_for_preview 全塞进
# <model>_data.dat，一次性 pickle。实测一个 300 万次迭代的模型：
#     文件 122.7 MB —— 读字节 0.06s + 反序列化 0.82s
#     其中 sample_for_preview 占 55 MB、loss_history 占 67 MB，
#     而启动时真正需要的只有 iter 和 options（几 KB）。
# 所以加一层缓存 <model>_data.dat.params（pickle，几 KB，毫秒级）：
#   · iter / options / choosed_gpu_indexes 从缓存读 → 启动不再碰大文件
#   · loss_history 交给后台线程从 data.dat 解析后并进来（8M 条约 2s，不挡启动）
#   · sample_for_preview 不缓存（55MB），置 None 让 update_sample_for_preview() 重取
# 与此同时 data.dat 仍按老格式【完整】写回（含完整 loss_history），
# 这样训练完把模型丢回上游原版也能直接读。
_PARAMS_CACHE_SUFFIX = '.params'

# 训练日志 CSV 的默认写盘间隔（步）。10 步一行足够画出收敛曲线，
# 又不至于让 open/append 变成训练线程上的可观测开销；
# options 里给了 log_interval 就按它来（见 get_train_log_interval）。
_TRAIN_LOG_DEFAULT_INTERVAL = 10


def _params_cache_path(dat_path):
    return Path(str(dat_path) + _PARAMS_CACHE_SUFFIX)


def _read_params_cache(dat_path):
    """缓存必须和 data.dat 一一对应：文件大小 + mtime 对不上就当没有。

    外部工具（tools/migrate/clear_Iter_str.py、转换器、ipe rov 原版）会直接改 data.dat，
    靠这对指纹保证我们不会拿一份过期的参数去覆盖它。
    """
    try:
        st = Path(dat_path).stat()
        data = pickle.loads(_params_cache_path(dat_path).read_bytes())
        if data.get('version') != 1:
            return None
        if data.get('src_size') != st.st_size or data.get('src_mtime_ns') != st.st_mtime_ns:
            return None
        return data
    except Exception:
        return None


def _write_params_cache(dat_path, model):
    try:
        st = Path(dat_path).stat()
        payload = {
            'version': 1,
            'iter': int(model.iter),
            'options': model.options,
            'choosed_gpu_indexes': model.choosed_gpu_indexes,
            'loss_count': len(model.loss_history),
            # 这份 dat 是不是 TF 时代的：缓存一命中就不再做老 dat 检测，但那条
            # 「只有参数、没有 torch 权重」的提示每轮都得说，标记存下来。
            'legacy': bool(getattr(model, '_legacy_dat', None)),
            'src_size': st.st_size,
            'src_mtime_ns': st.st_mtime_ns,
        }
        # 同时写一份 JSON 边车：4.1 的参数页/模型列表直接读它，不再解 122 MB 的 dat。
        # 带 (mtime, size) 指纹，dat 一动就失配，后端会重建。
        try:
            import json as _json
            _side = Path(dat_path).with_suffix('.opt_cache.json')
            _json_payload = {
                'ver': 1,
                'mtime': round(st.st_mtime, 3),
                'size': st.st_size,
                'iter': payload['iter'],
                'options': {k: v for k, v in (model.options or {}).items()
                            if not str(k).startswith('_') and k != 'loss_history'},
            }
            _tmpj = Path(str(_side) + '.tmp')
            _tmpj.write_text(_json.dumps(_json_payload, ensure_ascii=False, default=str), encoding='utf-8')
            os.replace(str(_tmpj), str(_side))
        except Exception:
            pass
        p = _params_cache_path(dat_path)
        tmp = Path(str(p) + '.tmp')
        tmp.write_bytes(pickle.dumps(payload, protocol=4))
        os.replace(str(tmp), str(p))
    except Exception:
        pass          # 缓存写不上不影响正确性，下次照样回退到解析 data.dat


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
            # 模型类名被写死（XSeg / XSegLite / FaceRestoreLite / DFSingle / PIA 都会
            # setdefault force_model_class_name），只有这时 model_name 才回落到类名。
            # 但 GUI 是用 --force-model-name 传"模型实例名"的（如 pia_004 / dfs_001），
            # 这里必须优先用它，否则同一类的所有实例都会塌缩成 <Class>_data.dat：
            #   1) 找不到 GUI 刚建好的 <name>_<Class>_data.dat，被当成首次运行，
            #      GUI 里设的分辨率 / 维度 / archi 全被丢掉；
            #   2) 训练时所有同类的模型都写进同一个文件，互相覆盖。
            # 注意保留 force_model_name == force_model_class_name 的旧行为
            # （XSeg / XSegLite 是单实例，dat 就一直叫 XSeg_data.dat / XSegLite_data.dat）。
            if force_model_name is not None and force_model_name != force_model_class_name:
                self.model_name = force_model_name
                if not self.model_name.endswith('_' + self.model_class_name):
                    self.model_name = self.model_name + '_' + self.model_class_name
            else:
                self.model_name = force_model_class_name

        self.iter = 0
        self.options = {}
        self.options_show_override = {}
        self.loss_history = []
        self.sample_for_preview = None
        # 训练第一轮之前 last_sample 还没有值；不初始化的话 get_previews() 会直接
        # AttributeError —— 异常被上层吞掉，表现就是「首次启动永远没有预览图」。
        self.last_sample = None
        self.choosed_gpu_indexes = None
        self._legacy_dat = None      # 非 None = 这份 dat 来自 TF 时代，值是迁移报告
        self.model_data_path = Path(self.get_strpath_storage_for_file('data.dat'))

        self._load_model_data()

        # TF 时代（DFL 1.x/2.0）的 dat 只有参数，没有 torch 权重：先把话说清楚，
        # 免得用户以为接着以前那次训练在跑。
        if self._legacy_dat is not None:
            self._announce_legacy_dat()

        if self.is_first_run():
            io.log_info("\n模型首次运行。")

        if silent_start:
            # 静默启动以前**完全忽略 force_gpu_idxs**：训练页选了「0,1」两卡，这里
            # 还是 BestGPU() 只取显存最大的那一张 —— 多卡选择等于没传，而且不提示。
            if cpu_only:
                self.device_config = nn.DeviceConfig.CPU()
            elif force_gpu_idxs:
                # 注意：这里的序号是**逻辑**序号（CUDA_VISIBLE_DEVICES 映射之后），
                # 由 4.1 的 _train_worker 算好再传下来。
                self.device_config = nn.DeviceConfig.GPUIndexes(force_gpu_idxs)
            else:
                self.device_config = nn.DeviceConfig.BestGPU()
            if self.device_config.cpu_only:
                io.log_info("静默启动：已选择设备 CPU")
            else:
                io.log_info("静默启动：已选择设备 " + ", ".join(
                    str(d.name) for d in self.device_config.devices))
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

    # ------------------------------------------------------------------
    #  TF 时代（DFL 1.x / 2.0）dat 的兼容层
    # ------------------------------------------------------------------
    # 老 dat 的 options 和 4.1 不是一套：face_type 短码/全名混用、learn_mask 改名成
    # masked_training、ed_ch_dims 一个值管编码器+解码器、archi 出现过 dfhd/liaech…
    # 规则、键映射和报告都在 core/legacy_dat.py；这里只负责「读完 dat、还没人用过
    # options」这一小段。新格式的 dat 必须原样通过（返回同一个 dict 对象）。
    def _adopt_loaded_options(self, options):
        if not isinstance(options, dict):
            return options
        try:
            from core import legacy_dat as _legacy
        except Exception as e:
            io.log_err(f"[LEGACY] 兼容层加载失败（{e}），这份 dat 按 4.1 原样读")
            return options

        reasons = _legacy.detect_legacy(options)
        if not reasons:
            return options

        try:
            model_py = Path(inspect.getmodule(type(self)).__file__)
        except Exception:
            model_py = None
        try:
            migrated, report = _legacy.migrate_options(options, model_py=model_py)
        except Exception as e:
            io.log_err(f"[LEGACY] 老 dat 迁移失败（{e}），按原样继续")
            return options

        counts = _legacy.report_counts(report)
        moved = counts['迁移'] + counts['改键']
        io.log_info(f"[LEGACY] 检测到 TF 时代 dat（{'；'.join(reasons)}），"
                    f"已迁移 {moved} 项，补默认 {counts['补默认']} 项，忽略 {counts['忽略']} 项")
        for line in report[:12]:
            io.log_info('  ' + line)
        if len(report) > 12:
            io.log_info(f"  …（其余 {len(report) - 12} 行报告："
                        f"python tools/migrate/import_legacy_dat.py --dat {self.model_data_path}）")
        self._legacy_dat = {'report': report, 'counts': counts, 'reasons': reasons}
        return migrated

    def _announce_legacy_dat(self):
        """老 dat 只有参数、没有可用权重 —— 训练/合成前把现状和下一步说清楚。"""
        name = self.get_model_name()
        try:
            pths = sorted(Path(self.saved_models_path).glob(name + '_*.pth'))
            npys = sorted(Path(self.saved_models_path).glob(name + '_*.npy'))
        except Exception:
            pths, npys = [], []
        if pths:
            io.log_info(f"[LEGACY] 已找到 {len(pths)} 个 torch 权重（*.pth），按常规加载。")
            return
        if self._legacy_dat.get('cached'):
            io.log_info("[LEGACY] 这份 dat 仍是 TF 时代的（options 上次已迁移过，本次走参数缓存）；"
                        "只有参数，没有可用的 torch 权重；新建模型会从零开始训练。")
        else:
            io.log_info("[LEGACY] 该 dat 来自 TF 版本，只有参数，没有可用的 torch 权重；"
                        "新建模型会从零开始训练。")
        dat_name = self.model_data_path.name
        stem = dat_name[:-len('_data.dat')] if dat_name.endswith('_data.dat') else name
        if npys:
            io.log_info(f"[LEGACY] 当前目录里有 {len(npys)} 个 TF 权重（.npy），可以转：")
        else:
            io.log_info("[LEGACY] 当前目录里连 TF 权重（.npy）都没有，这份 dat 只能当参数模板：")
        # 注意顺序：转换脚本是拿 dat 里的 options 建网络骨架的，老 dat（archi=dfhd 等）
        # 直接喂给它只会 ValueError: Unsupported SAEHD archi_type —— 先迁 dat 再转。
        io.log_info('        python tools/migrate/import_legacy_dat.py '
                    f'--dat "{self.model_data_path}" '
                    f'--apply --out "<目标模型目录>/{stem}_data.dat" --convert')

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

    def ask_write_train_log(self, default_value=False):
        """是否把每个采样点的训练指标追加到 <模型名>_train_log.csv。

        默认关闭：它是给外部做对比分析用的（不同 lr / 优化器 / 架构的收敛曲线），
        平时训练不需要多一个文件。开启后每 N 步只多一次小文件 append，代价可以忽略。
        """
        default_write_train_log = bool(self.load_or_def_option('write_train_log', default_value))
        self.options['write_train_log'] = io.input_bool(
            "写入训练日志 CSV",
            default_write_train_log,
            help_message='每 %d 步往模型目录下的 <模型名>_train_log.csv 追加一行：'
                         '迭代 / loss_src / loss_dst / 当前 lr / 单步耗时(ms)。'
                         'Excel 或 pandas 可直接读。' % _TRAIN_LOG_DEFAULT_INTERVAL,
        )

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

    def ask_lr_warmup(self, default_value=0):
        """LR warmup：前 N 步把有效 lr 从 1/N 线性抬到设定值（0 = 关闭）。

        为什么需要它：Adam/AdaBelief 的 m/v 都从 0 起步，最初几步的二阶矩估计严重偏小，
        实际步长远大于名义 lr —— 高分辨率 / 小 batch 时，训练刚开始那几步最容易炸成 NaN。
        与其为了「前期稳」把全局 lr 一直压低（后期又嫌慢），不如只在开头 N 步压住。
        """
        default_lr_warmup_steps = self.load_or_def_option('lr_warmup_steps', default_value)
        try:
            default_lr_warmup_steps = max(0, int(default_lr_warmup_steps or 0))
        except (TypeError, ValueError):
            default_lr_warmup_steps = 0
        self.options['lr_warmup_steps'] = max(0, int(io.input_int(
            "学习率预热步数 (LR warmup)",
            default_lr_warmup_steps,
            add_info='0=关闭',
            help_message='前 N 步的有效 lr 按 i/N 线性增长，第 N 步之后恢复设定值。'
                         '0 = 关闭。分辨率高 / batch 小 / 一开跑就崩的训练可以设 200~2000。',
        )))

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
        # 优先用「上一轮训练真正喂进去的样本」；还没跑过一轮时退回初始化时生成的
        # sample_for_preview —— 否则新模型/迁移过来的模型首次启动永远拿不到预览。
        s = getattr(self, 'last_sample', None) or getattr(self, 'sample_for_preview', None)
        if s is None:
            _now = time.time()
            if _now - getattr(self, '_preview_nosample_t', 0.0) >= 30.0:
                self._preview_nosample_t = _now
                io.log_info('[preview] 还没有样本可用（last_sample / sample_for_preview 都是空），这一轮不生成预览 —— 等训练跑过一轮就有了')
            return []
        try:
            return self.onGetPreview(s) or []
        except Exception as e:
            io.log_info(f'预览生成失败（这一轮跳过）：{type(e).__name__}: {e}')
            return []

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

    def _load_model_data(self):
        """读 data.dat —— 有参数缓存就走快路径，否则老路径全量解析。

        快路径下 loss_history 交给后台线程（解析大 pickled 要 0.8~2s+），
        sample_for_preview 也留到后面由 update_sample_for_preview() 重取。
        """
        self._loss_lock = threading.Lock()
        self._loss_partial = False
        self._loss_loader = None
        if not self.model_data_path.exists():
            return

        cache = _read_params_cache(self.model_data_path)
        if cache is not None:
            # 快路径：只读几 KB 的参数缓存，大文件留给后台线程
            t_cache = time.time()
            self.iter = int(cache.get('iter', 0))
            if self.iter != 0:
                self.options = self._adopt_loaded_options(cache.get('options', {}))
                if cache.get('legacy'):
                    self._legacy_dat = {'cached': True, 'report': [], 'counts': {}, 'reasons': []}
                self.choosed_gpu_indexes = cache.get('choosed_gpu_indexes', None)
                self.loss_history = []
                self._loss_partial = True
                # 8 秒后再读历史：那时权重已加载完、第一个迭代也跑起来了
                self._start_loss_loader(delay=8.0)
            io.log_info(f"正在加载模型：{self.model_name} ...（参数缓存，"
                        f"{(time.time() - t_cache) * 1000:.0f} ms"
                        f"{'，损失曲线后台加载中' if self._loss_partial else ''}）")
            return

        io.log_info(f"正在加载模型：{self.model_name} ...")
        model_data = pickle.loads(self.model_data_path.read_bytes())
        self.iter = model_data.get('iter', 0)
        if self.iter != 0:
            self.options = self._adopt_loaded_options(model_data['options'])
            self.loss_history = model_data.get('loss_history', [])
            self.sample_for_preview = model_data.get('sample_for_preview', None)
            self.choosed_gpu_indexes = model_data.get('choosed_gpu_indexes', None)
        _write_params_cache(self.model_data_path, self)   # 下次就走快路径

    # ---------- 模型卡片用的预览快照 ----------
    # 模型选择页要显示"这个模型练到什么样了"，所以最后一次刷新的预览图必须落盘。
    # 原版的 sample_for_preview（那个跨重启固定的预览样本）本身没用，但它
    # 55MB 的存在逼得任何想展示预览的程序都得先反序列化整个 data.dat —— 现在
    # 改成训练时把当前预览写成一张小 jpg + 一份 loss 尾巴的小 json：
    #     <models>/<model>_preview.jpg   ~40-120KB
    #     <models>/<model>_preview.json  约 40KB（最后 2500 个 loss 点）
    # 写盘规矩（服务器上跑就得按不可信环境来）：
    #   · 路径只由模型名拼出来，且 resolve() 之后必须正好落在模型根目录下
    #   · 名字里出现 / \ ? 或 . / .. 直接拒绝
    #   · 目标是符号链接时拒绝（否则等于让人拿模型名写到别处）
    #   · 先写 .tmp 再 os.replace，读的人永远看不到半个文件
    #   · 全程 try/except，写失败绝不影响训练
    _PREVIEW_SNAPSHOT_INTERVAL = 30.0
    _PREVIEW_SNAPSHOT_MAX_W = 1280

    def get_preview_snapshot_paths(self):
        root = Path(self.saved_models_path).resolve()
        name = str(self.get_model_name())
        if (not name) or name in ('.', '..') or any(ch in name for ch in ('/', chr(92), chr(0))):
            raise ValueError('模型名不能用于预览快照: %r' % (name,))
        jpg = (root / (name + '_preview.jpg')).resolve()
        js = (root / (name + '_preview.json')).resolve()
        for p in (jpg, js):
            if p.parent != root:
                raise ValueError('预览快照路径越界: %s' % p)
            if p.is_symlink():
                raise ValueError('预览快照目标是符号链接，拒绝写入: %s' % p)
        return jpg, js

    @staticmethod
    def _preview_to_bgr(img):
        """把 onGetPreview 给的 (C,H,W) 或 (H,W,C) float[0..1] 转成 uint8 BGR 图。"""
        a = np.asarray(img, dtype=np.float32)
        if a.ndim == 3 and a.shape[0] in (1, 3) and a.shape[-1] > 4:
            a = np.transpose(a, (1, 2, 0))
        if a.ndim == 2:
            a = np.repeat(a[..., None], 3, axis=2)
        if a.ndim != 3:
            return None
        if a.shape[2] == 1:
            a = np.repeat(a, 3, axis=2)
        if a.shape[2] > 3:
            a = a[..., :3]
        return (np.clip(a, 0.0, 1.0) * 255.0).astype(np.uint8)

    def write_preview_snapshot(self, previews=None):
        """把当前预览 / loss 尾巴落盘，给模型库用。失败静默。"""
        try:
            if previews is None:
                previews = self.get_previews()
            jpg_path, js_path = self.get_preview_snapshot_paths()

            imgs = []
            for item in previews or []:
                if not isinstance(item, (list, tuple)) or len(item) < 2:
                    continue
                bgr = self._preview_to_bgr(item[1])
                if bgr is not None and bgr.size:
                    imgs.append(bgr)
            if imgs:
                h = min(i.shape[0] for i in imgs)
                row = []
                for i in imgs:
                    if i.shape[0] != h:
                        w = max(1, int(round(i.shape[1] * h / max(1, i.shape[0]))))
                        i = cv2.resize(i, (w, h), interpolation=cv2.INTER_AREA)
                    row.append(i)
                strip = np.concatenate(row, axis=1)
                if strip.shape[1] > self._PREVIEW_SNAPSHOT_MAX_W:
                    s = self._PREVIEW_SNAPSHOT_MAX_W / float(strip.shape[1])
                    strip = cv2.resize(strip, (self._PREVIEW_SNAPSHOT_MAX_W,
                                                max(1, int(round(strip.shape[0] * s)))),
                                       interpolation=cv2.INTER_AREA)
                ok, buf = cv2.imencode('.jpg', strip, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
                if ok:
                    tmp = jpg_path.with_name(jpg_path.name + '.tmp')
                    tmp.write_bytes(buf.tobytes())
                    os.replace(str(tmp), str(jpg_path))

            # loss 尾巴：模型卡片上那条小曲线，不读 data.dat 就能画
            import json as _json
            tail = list(self.loss_history)[-2500:]
            payload = {
                'iter': int(self.iter),
                'loss_tail': [[float(x[0]), float(x[1])] for x in tail
                              if isinstance(x, (list, tuple)) and len(x) >= 2],
                'updated': int(time.time()),
            }
            tmp_js = js_path.with_name(js_path.name + '.tmp')
            tmp_js.write_text(_json.dumps(payload, ensure_ascii=False), encoding='utf-8')
            os.replace(str(tmp_js), str(js_path))
        except Exception as e:
            # 以前这里是裸 pass：预览快照写失败**完全无声**，界面就一直是「还没有预览图」，
            # 用户和我们都只能猜。快照失败确实不该影响训练，但必须留下痕迹 ——
            # 每次都打一行（30 秒最多一次，量很小），带异常类型和消息。
            io.log_info(f'[preview] 预览快照写入失败（不影响训练）：{type(e).__name__}: {e}')

    def _preview_snapshot_due(self):
        return (time.time() - getattr(self, '_preview_snapshot_t', 0.0)
                >= self._PREVIEW_SNAPSHOT_INTERVAL)

    def _maybe_write_preview_snapshot(self, previews=None, force=False):
        """按时间节流（默认 30s）在后台线程里写，不占训练线程。

        注意：previews 必须由训练线程算好再传进来（get_previews 会跑一次前向，
        丢到后台线程会跟训练抢 CUDA）。这里只负责编码和写盘。
        """
        now = time.time()
        if not force and (now - getattr(self, '_preview_snapshot_t', 0.0)
                          < self._PREVIEW_SNAPSHOT_INTERVAL):
            return
        self._preview_snapshot_t = now
        try:
            t = threading.Thread(target=self.write_preview_snapshot,
                                 kwargs={'previews': previews}, daemon=True,
                                 name='preview-snapshot')
            t.start()
        except Exception as e:
            io.log_info(f'[preview] 起不了快照线程（不影响训练）：{type(e).__name__}: {e}')

    def _start_loss_loader(self, delay=0.0):
        """后台把 data.dat 里的 loss_history 读出来并进来。

        启动时不等它：300 万条要 0.8s，800 万条约 2s+。期间训练照常开跑，
        预览曲线会短暂只用新点，解析完成后自动接上完整历史。

        delay>0 时把「开始读」往后推 —— pickle.loads 虽然是 C 实现，但会一直握着
        GIL，正好和启动时的权重加载 / CUDA 初始化 / 第一个迭代抢同一把锁。
        等首帧跑起来再读，117MB 的解析就不挡路了。
        """
        dat_path = self.model_data_path

        def _work():
            try:
                hist = pickle.loads(dat_path.read_bytes()).get('loss_history', [])
            except Exception:
                io.log_err(f"后台读取 {dat_path.name} 的 loss_history 失败；"
                           f"预览曲线暂时只有新点，保存时会同步重试。")
                return
            with self._loss_lock:
                # 解析期间训练已经在 append 新点，接在历史后面
                self.loss_history = hist + self.loss_history
                self._loss_partial = False

        def _spawn():
            if not self._loss_partial:
                return           # 期间已经有人同步补完了
            t = threading.Thread(target=_work, name='loss-history-loader', daemon=True)
            self._loss_loader = t
            t.start()

        if delay and delay > 0:
            timer = threading.Timer(float(delay), _spawn)
            timer.daemon = True
            timer.start()
            self._loss_timer = timer
        else:
            _spawn()

    def _ensure_loss_history_complete(self):
        """写 data.dat 之前必须成立：否则会把历史截断成"本次运行的点"。"""
        with self._loss_lock:
            if not self._loss_partial:
                return True
        t = self._loss_loader
        if t is None:
            # 延迟还没到（或者根本没起）就有人要保存：就地同步补完
            timer = getattr(self, '_loss_timer', None)
            if timer is not None:
                timer.cancel()
            try:
                hist = pickle.loads(self.model_data_path.read_bytes()).get('loss_history', [])
            except Exception:
                return False
            with self._loss_lock:
                self.loss_history = hist + self.loss_history
                self._loss_partial = False
            return True
        if t.is_alive():
            t.join()
        with self._loss_lock:
            return not self._loss_partial

    def _append_loss(self, row):
        with self._loss_lock:
            self.loss_history.append(row)

    def save(self):
        if self._crash_detected:
            io.log_info(f'[Crash Detector] 检测到训练崩溃，取消本次自动备份')
            return

        self._ensure_disk_space()

        # 先落权重、再写 dat。反过来的话，正好在两次写之间断电/崩溃，
        # dat 里的 iter 会比权重新一拍 —— 恢复后从更大的 iter 接着跑，
        # 曲线看着像“白跑了一段”，而且不会有任何报错。
        self.onSave()

        # encoding 必须显式给：Path.write_text 的默认值是这个进程的 locale（Windows 上
        # 是 cp936），摘要里只要有 cp936 编不出的字符就直接抛 UnicodeEncodeError，
        # 整个 save()（权重 + dat）跟着一起失败；而且同一份摘要换台机器读出来是乱码。
        Path(self.get_summary_path()).write_text(self.get_summary_text(), encoding='utf-8')

        # Save .dat first — preserves current settings even if archiving fails
        if self._ensure_loss_history_complete():
            model_data = {
                'iter': self.iter,
                'options': self.options,
                'loss_history': self.loss_history,
                'sample_for_preview': self.sample_for_preview,
                'choosed_gpu_indexes': self.choosed_gpu_indexes,
            }
            pathex.write_bytes_safe(self.model_data_path, pickle.dumps(model_data))
            _write_params_cache(self.model_data_path, self)
            # 保存时顺手刷新模型卡片预览（get_previews 必须在训练线程里调用）
            try:
                if self.generator_list:
                    self._maybe_write_preview_snapshot(self.get_previews(), force=True)
            except Exception:
                pass
        else:
            # 宁可这次不写 data.dat，也不能用"只有本次运行的点"把历史覆盖掉
            io.log_err("没能读出 data.dat 里的历史损失，本次跳过写 data.dat"
                       "（避免把曲线截断）；权重和预览照常保存。")

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
                    # 以前是「old/ 里有同名就跳过」—— 于是重复转换时源 .npy 永远留在
                    # model/ 里，看起来像「压根没归档」。这里改成覆盖：old/ 里留最新一份，
                    # model/ 里那批 .npy 一定要挪走。
                    try:
                        if dest.exists():
                            dest.unlink()
                        shutil.move(str(npy), str(dest))
                    except Exception as _e:
                        io.log_err(f"archive {npy.name} failed: {_e}")
                # 同时备份 _data.dat 到 old/
                dat_path = self.model_data_path
                if dat_path.exists():
                    dat_dest = old_dir / dat_path.name
                    if not dat_dest.exists():
                        shutil.copy2(str(dat_path), str(dat_dest))
                io.log_info(f"  {len(npy_files)} old .npy files archived to old/")

    def create_backup(self):
        if self._crash_detected:
            io.log_info(f'[Crash Detector] 检测到训练崩溃，取消本次自动备份')
            return

        self._ensure_disk_space()

        io.log_info("正在创建备份...", end='\r')

        if not self.autobackups_path.exists():
            self.autobackups_path.mkdir(exist_ok=True)

        # 「最大备份数量」界面上能改、也写进了 dat 的 options，但这里以前读的是
        # self.max_backups —— 那个属性**从来没有被赋过值**，getattr 永远返回 24。
        # 于是用户设 3 也是按 24 轮转：三份之后的备份既不会被删，编号还会一路涨
        # 到 24 才开始滚动。和 backup_interval 不同（Trainer 那边读的是 options，
        # 那个是生效的），这一项从 4.0 起就是死的。
        try:
            n = int(self.options.get('max_backups') or 24)
        except (TypeError, ValueError):
            n = 24
        n = max(1, min(99, n))      # 界面给的范围是 1..99，别让脏值把循环搞炸

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

    # ============================================================
    # 训练日志 CSV
    # ============================================================
    # 为什么要落盘：终端那行日志是用 \r 原地刷新的，没有稳定列宽、没有时间戳，
    # 外部要做基准对比（改前/改后、换优化器、换 batch）时只能靠人眼截图。
    # 每 N 步追加一行 CSV 就够分析了，代价是一次小文件 append。
    def get_train_log_path(self):
        # 复用 get_strpath_storage_for_file：落成 <模型名>_train_log.csv，和别的存档同级
        return Path(self.get_strpath_storage_for_file('train_log.csv'))

    def get_train_log_interval(self):
        try:
            n = int(self.options.get('log_interval', 0) or 0)
        except (TypeError, ValueError):
            n = 0
        return n if n > 0 else _TRAIN_LOG_DEFAULT_INTERVAL

    def get_train_log_lr(self):
        """当前实际生效的 lr（含余弦 / 平台期 / warmup）。取不到就退回初始 lr。"""
        for attr_name in ('src_dst_opt', 'G_opt', 'opt'):
            _opt = getattr(self, attr_name, None)
            _get_lr = getattr(_opt, 'get_lr', None)
            if callable(_get_lr):
                try:
                    return float(_get_lr())
                except Exception:
                    break
        try:
            return float(self.options.get('lr', 0.0) or 0.0)
        except (TypeError, ValueError):
            return 0.0

    def _write_train_log_row(self, losses, iter_time):
        """把本步的 iter / 两个 loss / lr / 耗时追加进 CSV。任何异常都吞掉。

        losses 的形态是 onTrainOneIter() 的返回值 [(name, value), ...]；
        SAEHD / DFSingle 的前两项就是 loss_src、loss_dst（列名固定，不跟随名字变）。
        """
        if not bool(self.options.get('write_train_log', False)):
            return
        interval = self.get_train_log_interval()
        if interval > 1 and (self.iter % interval) != 0:
            return
        try:
            values = []
            for item in losses:
                try:
                    values.append(float(item[1]))
                except (TypeError, ValueError, IndexError):
                    values.append(float('nan'))
            loss_src = values[0] if len(values) > 0 else float('nan')
            loss_dst = values[1] if len(values) > 1 else float('nan')

            path = self.get_train_log_path()
            need_header = (not path.exists()) or path.stat().st_size == 0
            # newline="" 让 csv 模块自己决定行尾（Windows 上就是 \r\n），否则会多出空行；
            # encoding 固定 utf-8，避免不同 locale 下的编码漂移（内容本身全是 ASCII）。
            with open(str(path), 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                if need_header:
                    writer.writerow(['iter', 'loss_src', 'loss_dst', 'lr', 'ms'])
                writer.writerow([
                    int(self.iter),
                    '{:.6f}'.format(loss_src),
                    '{:.6f}'.format(loss_dst),
                    '{:.6e}'.format(self.get_train_log_lr()),
                    int(round(float(iter_time) * 1000.0)),
                ])
        except Exception:
            pass          # 日志写不进去绝不能影响训练

    def train_one_iter(self):
        iter_time = time.time()
        losses = self.onTrainOneIter()
        iter_time = time.time() - iter_time

        self._append_loss([float(loss[1]) for loss in losses])

        # crash detection: check last 100 iterations for loss spikes
        ct = self.options.get('crash_threshold', 0.0)
        if ct > 0.0:
            recent = self.loss_history[-100:]
            self._crash_detected = any(max(l) > ct for l in recent)

        if self.should_save_preview_history():
            plist = []
            previews = None

            if io.is_colab() or self.write_preview_history:
                previews = self.get_previews()

            # 模型卡片快照：没开预览历史也要有图，按时间节流（30s 最多一次）
            if previews is None and self._preview_snapshot_due():
                previews = self.get_previews()
            if previews is not None:
                self._maybe_write_preview_snapshot(previews)

            if io.is_colab():
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
        self._write_train_log_row(losses, iter_time)

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
        self._ensure_loss_history_complete()      # 截断前必须先有完整历史，否则会把历史截成只剩新点
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

    # ------------------------------------------------------------------
    #  摘要排版表：中文名 / 分组 / 结构参数集合 / 分组顺序
    #
    #  这几张表**只决定排版**，不决定「打印哪些键」。
    #  打印集合完全来自 self.options 本身（见 get_summary_text 的 buckets）：
    #    * 引擎新增一个 option → 摘要自动多一行，不用改这里；
    #    * 这里漏了某个键的中文名 → 那一行照打，标签退化成键名本身。
    #  以前这里是 4 个手写的键名白名单（架构/增强/补丁/训练），dat 里不在这四张
    #  表里的键永远不会出现 —— freeze_* 全系列、lr_cos / lr_plateau_* /
    #  gradient_checkpointing / src / dst / device / target_batch / xseg_* …
    #  于是「摘要里看到的」和「实际训练时用的」对不上，把模型交给别人时尤其致命。
    # ------------------------------------------------------------------
    SUMMARY_STRUCT_KEYS = ('resolution', 'face_type', 'archi', 'e_dims',
                           'ae_dims', 'd_dims', 'd_mask_dims')

    SUMMARY_GROUP_ORDER = ('数据集', '训练设置', '优化器', '精度与显存', 'GAN',
                           '损失与补丁', '数据增强', '冻结层', '预训练',
                           '备份与日志', '遮罩/XSeg', '设备', '其他')

    SUMMARY_LABELS = {
        # 结构参数
        'resolution': '分辨率', 'face_type': '人脸类型', 'archi': '架构',
        'ae_dims': 'AE维度', 'e_dims': '编码器维', 'd_dims': '解码器维',
        'd_mask_dims': '遮罩维', 'inter_dims': '中间层维度',
        # 数据集
        'src': 'src人脸集', 'dst': 'dst人脸集', 'pretrain_dataset_path': '预训练集',
        # 训练设置
        'batch_size': '批次大小', 'target_batch': '目标批次',
        'accum_steps': '累积次数', 'target_iter': '目标迭代数',
        'masked_training': '遮罩训练', 'blur_out_mask': '遮罩羽化',
        'eyes_mouth_prio': '嘴眼优先', 'uniform_yaw': '均匀偏航',
        # 优化器
        'optimizer': '优化器', 'lr': '学习率', 'lr_policy': '学习率策略',
        'lr_cos': '余弦周期', 'lr_total_steps': '学习率总步数',
        'lr_plateau_factor': '平台衰减系数', 'lr_plateau_patience': '平台耐心',
        'lr_plateau_min_ratio': '平台最小比例', 'lr_dropout': 'LR Dropout',
        'lr_warmup_steps': 'LR 预热步数', 'lr_scheduler': 'LR调度器',
        'lr_cos_period': '余弦周期', 'clipgrad': '梯度裁剪',
        # 精度与显存
        'use_bf16': 'BF16混合', 'gradient_checkpointing': '梯度检查点',
        'models_opt_on_gpu': '优化器GPU', 'use_fast_generator': '快速生成器',
        'eager_mode': '即时模式',
        # GAN
        'gan_power': 'GAN强度', 'gan_patch_size': 'GAN块大小', 'gan_dims': 'GAN维度',
        # 损失与补丁
        'true_face_power': '真脸强度', 'face_style_power': '面部风格',
        'bg_style_power': '背景风格', 'vgg_perceptual_power': 'VGG感知损失',
        'edge_enhance_power': '边缘强化', 'ct_mode': '色彩迁移',
        'preview_grade_alg': '预览调色',
        # 数据增强
        'random_warp': '随机形变', 'random_occlusion': '随机遮挡',
        'random_noise': '随机噪点', 'random_color_power': '随机偏色',
        'random_color_alg': '偏色算法', 'random_hsv_power': 'HSV强度',
        'random_src_flip': 'SRC翻转', 'random_dst_flip': 'DST翻转',
        'random_flip': '随机翻转', 'rotation_range': '旋转范围',
        'scale_range': '缩放范围', 't_range': '偏移范围',
        'eyes_scale': '眼睛范围', 'mouth_scale': '嘴巴范围',
        'pia_cycle_power': 'PIA循环一致',
        # 冻结层
        'freeze_encoder': '冻结编码器', 'freeze_inter': '冻结中间',
        'freeze_inter_AB': '冻结inter_AB', 'freeze_inter_B': '冻结inter_B',
        'freeze_decoder_mask': '冻结遮罩解码器', 'freeze_decoder_dst': '冻结输出',
        'freeze_inter_src': '冻结源', 'freeze_inter_dst': '冻结目标',
        'freeze_bottleneck': '冻结瓶颈', 'freeze_decoderA_mask': '冻结A遮罩',
        'freeze_decoderB_mask': '冻结B遮罩', 'freeze_bottleneck_AB': '冻结共享',
        'freeze_bottleneck_B': '冻结瓶颈B',
        # 预训练
        'pretrain': '预训练', 'pretrain_single_decoder': '单解码器',
        # 备份与日志
        'max_backups': '最大备份数', 'backup_interval': '备份间隔',
        'crash_threshold': '崩溃阈值', 'write_preview_history': '预览历史',
        'write_train_log': '训练日志CSV', 'log_code_stats': 'Inter分布日志',
        'nan_guard': 'NaN梯度保护',
        # 设备 / 遮罩-XSeg
        'device': '设备', 'xseg_erode': 'XSeg腐蚀', 'xseg_blur': 'XSeg模糊',
    }

    SUMMARY_GROUP_OF = {
        'src': '数据集', 'dst': '数据集', 'pretrain_dataset_path': '数据集',
        'batch_size': '训练设置', 'target_batch': '训练设置',
        'accum_steps': '训练设置', 'target_iter': '训练设置',
        'masked_training': '训练设置', 'blur_out_mask': '训练设置',
        'eyes_mouth_prio': '训练设置', 'uniform_yaw': '训练设置',
        'optimizer': '优化器', 'lr': '优化器', 'lr_policy': '优化器',
        'lr_cos': '优化器', 'lr_total_steps': '优化器',
        'lr_plateau_factor': '优化器', 'lr_plateau_patience': '优化器',
        'lr_plateau_min_ratio': '优化器', 'lr_dropout': '优化器',
        'lr_warmup_steps': '优化器', 'lr_scheduler': '优化器',
        'lr_cos_period': '优化器', 'clipgrad': '优化器',
        'use_bf16': '精度与显存', 'gradient_checkpointing': '精度与显存',
        'models_opt_on_gpu': '精度与显存', 'use_fast_generator': '精度与显存',
        'eager_mode': '精度与显存',
        'gan_power': 'GAN', 'gan_patch_size': 'GAN', 'gan_dims': 'GAN',
        'true_face_power': '损失与补丁', 'face_style_power': '损失与补丁',
        'bg_style_power': '损失与补丁', 'vgg_perceptual_power': '损失与补丁',
        'edge_enhance_power': '损失与补丁', 'ct_mode': '损失与补丁',
        'preview_grade_alg': '损失与补丁',
        'random_warp': '数据增强', 'random_occlusion': '数据增强',
        'random_noise': '数据增强', 'random_color_power': '数据增强',
        'random_color_alg': '数据增强', 'random_hsv_power': '数据增强',
        'random_src_flip': '数据增强', 'random_dst_flip': '数据增强',
        'random_flip': '数据增强', 'rotation_range': '数据增强',
        'scale_range': '数据增强', 't_range': '数据增强',
        'eyes_scale': '数据增强', 'mouth_scale': '数据增强',
        'pia_cycle_power': '数据增强',
        'pretrain': '预训练', 'pretrain_single_decoder': '预训练',
        'max_backups': '备份与日志', 'backup_interval': '备份与日志',
        'crash_threshold': '备份与日志', 'write_preview_history': '备份与日志',
        'write_train_log': '备份与日志', 'log_code_stats': '备份与日志',
        'nan_guard': '备份与日志',
        'device': '设备',
    }

    @classmethod
    def _summary_group_of(cls, key):
        """一个 option 键该排在哪个小标题下。

        认识的走白表；不认识的按前缀猜（freeze_* / random_* / gan_* / lr* …）；
        实在猜不出来就进「其他」—— 无论走哪条路，这个键**一定会被打印**。
        """
        g = cls.SUMMARY_GROUP_OF.get(key)
        if g:
            return g
        k = str(key).lower()
        if k.startswith('freeze_'):
            return '冻结层'
        if k.startswith('xseg') or 'mask' in k:
            return '遮罩/XSeg'
        if k.startswith('random_') or k.startswith('ct_'):
            return '数据增强'
        if k.startswith('lr') or k in ('optimizer', 'clipgrad'):
            return '优化器'
        if k.startswith('gan_'):
            return 'GAN'
        return '其他'

    @staticmethod
    def _summary_fmt_value(v):
        """按 dat 里存的原始精度写出来，不做任何「好看化」。

        * float 用 repr：0.00005 存进 pickle 就是 5e-05，摘要里也写 5e-05；
          round / %g / f-string 都可能把它改成另一个写法。
        * numpy 标量（老 dat 里 gan_patch_size / resolution 是 np.int64）先收敛成
          Python 原生类型，否则会打印成 np.int64(16)。
        * bool 打印 True/False（和 pickle 里的字面量一致），不是 1/0。
        """
        if v is None:
            return 'None'
        if isinstance(v, bool):
            return 'True' if v else 'False'
        try:
            if isinstance(v, np.bool_):
                return 'True' if bool(v) else 'False'
            if isinstance(v, np.integer):
                return str(int(v))
            if isinstance(v, np.floating):
                return repr(float(v))
        except Exception:
            pass
        if isinstance(v, int):
            return str(v)
        if isinstance(v, float):
            return repr(v)
        if isinstance(v, (list, tuple)):
            return '[' + ', '.join(ModelBase._summary_fmt_value(x) for x in v) + ']'
        if isinstance(v, dict):
            return '{' + ', '.join('%s: %s' % (k, ModelBase._summary_fmt_value(x))
                                   for k, x in v.items()) + '}'
        if isinstance(v, str) and v == '':
            # 空串必须显式写成 ''：否则摘要里只剩一个「键名: 」，和「这个键没打印」
            # 长得一模一样（pretrain_dataset_path 的默认值就是空串）。
            return "''"
        return str(v)

    def get_summary_text(self):
        # ---- display-width helpers (CJK-aware) ----
        def _dw(s):
            return ModelBase._disp_len(s)

        def _ljust(s, w):
            return s + ' ' * max(0, w - _dw(s))

        def _center(s, w):
            p = max(0, w - _dw(s))
            return ' ' * (p // 2) + s + ' ' * (p - p // 2)

        # Wrap content in ==...==, padding to w display-width
        def _line(content='', w=None):
            if w is None:
                w = inner_w
            return f'=={content}{" " * max(0, w - _dw(content))}=='

        def _cell(key, val):
            """一格：「中文名 键名: 值」（没有中文名就只写键名）。"""
            lab = self.SUMMARY_LABELS.get(key) or ''
            body = self._summary_fmt_value(val)
            return f'{lab} {key}: {body}' if lab else f'{key}: {body}'

        # ============================================================
        #   1) 参数分区：结构参数 / 训练参数（全量，一个都不许漏）
        # ============================================================
        opts = dict(self.options or {})
        override = dict(getattr(self, 'options_show_override', None) or {})

        struct_keys = [k for k in self.SUMMARY_STRUCT_KEYS if k in opts]
        struct_cells = [_cell(k, opts[k]) for k in struct_keys]
        struct_set = set(struct_keys)

        buckets = {}
        for k in opts:
            if k in struct_set:
                continue
            buckets.setdefault(self._summary_group_of(k), []).append(k)
        for k in buckets:
            buckets[k].sort()          # 组内按键名排序：不依赖手写顺序，也不依赖 dat 的键序

        ordered_groups = [g for g in self.SUMMARY_GROUP_ORDER if buckets.get(g)]
        for g in sorted(buckets):
            if g not in ordered_groups:
                ordered_groups.append(g)          # 万一有新的组名，也排在后面照样打印

        # ---- 2 列宽度：所有区块共用同一套列宽，行与行才对得齐 ----
        all_cells = list(struct_cells)
        for g in ordered_groups:
            all_cells += [_cell(k, opts[k]) for k in buckets[g]]
        all_cells += [_cell(k, override[k]) for k in sorted(override)]
        MIN_INNER = 72
        COL_GAP = ' | '
        PAD = '  '
        col_w = max([_dw(c) + 2 for c in all_cells] + [24])
        inner_w = max(col_w * 2 + _dw(COL_GAP) + _dw(PAD) * 2, MIN_INNER)
        col_w = max(col_w, (inner_w - _dw(COL_GAP) - _dw(PAD) * 2) // 2)

        def _rows(cells):
            """把一串格子摊成 2 列的行（先读左列、再读右列）。"""
            out = []
            if not cells:
                return out
            per = (len(cells) + 1) // 2
            left, right = cells[:per], cells[per:]
            for i in range(per):
                a = _ljust(left[i], col_w)
                b = _ljust(right[i], col_w) if i < len(right) else ' ' * col_w
                out.append(_line(PAD + a + COL_GAP + b + PAD, inner_w))
            return out

        def _group_header(name):
            rule = '-' * max(0, inner_w - _dw(name) - 4)
            return _line(f'-- {name} {rule}', inner_w)

        summary_text = []

        # ============================================================
        #   Header
        # ============================================================
        summary_text += [_line(_center(' 模型摘要 ', inner_w), inner_w)]
        # 分隔符刻意不用 " | "：那一对竖线是格子之间的列分隔符，标题行用它会
        # 让「按列切分」的解析器把标题也当成一个参数格。
        hdr = (f'模型名称: {self.get_model_name()}  ·  当前迭代: {self.get_iter()}'
               f'  ·  参数项: {len(opts)}')
        summary_text += [_line(_center(hdr, inner_w), inner_w)]
        summary_text += [_line('', inner_w)]
        summary_text += [_line('-' * inner_w, inner_w)]

        # ============================================================
        #   结构参数（创建后不可改）
        # ============================================================
        if struct_cells:
            summary_text += [_group_header('结构参数（创建后不可改，换值只能新建模型）')]
            summary_text += _rows(struct_cells)
            summary_text += [_line('', inner_w)]

        # ============================================================
        #   训练参数（self.options 全量，按分组排序）
        # ============================================================
        summary_text += [_group_header('训练参数（dat 里 options 的全部 %d 项）'
                                       % (len(opts) - len(struct_set)))]
        for g in ordered_groups:
            summary_text += [_group_header('· ' + g)]
            summary_text += _rows([_cell(k, opts[k]) for k in buckets[g]])
        if not ordered_groups:
            summary_text += [_line('  （这份 dat 的 options 里只有结构参数）', inner_w)]
        summary_text += [_line('', inner_w)]

        # 显示覆盖值：预训练时引擎按"实际生效值"覆盖几项（见各模型的
        # options_show_override）。它们不是 dat 里的独立键，所以单独成块，
        # 不混进上面的全量列表 —— 上面那一区的键集合必须和 dat 严格一一对应。
        if override:
            summary_text += [_group_header('实际生效值（引擎覆盖，非 dat 条目）')]
            summary_text += _rows([_cell(k, override[k]) for k in sorted(override)])
            summary_text += [_line('', inner_w)]

        summary_text += [_line('-' * inner_w, inner_w)]
        summary_text += [_line('', inner_w)]
        # ============================================================
        #   Module status (key:value columns, same style as options)
        # ============================================================
        if self._module_info_list:
            # Separate modules from optimizers
            mods = [(n, p, s) for n, p, s in self._module_info_list if not n.endswith('_opt')]
            opts_m = [(n, p, s) for n, p, s in self._module_info_list if n.endswith('_opt')]

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

            total_col_w = sum(col_dws) + (ncols - 1) * _dw(COL_GAP) + 2 * _dw(PAD)
            mod_total_w = max(total_col_w, inner_w)
            # Distribute extra width so module table fills the full inner width
            if mod_total_w > total_col_w:
                avail = mod_total_w - (ncols - 1) * _dw(COL_GAP) - 2 * _dw(PAD)
                extra = avail - sum(col_dws)
                if extra > 0:
                    for i in range(ncols):
                        col_dws[i] += extra // ncols + (1 if i < extra % ncols else 0)

            summary_text += [_line(_center(' 模块状态 ', mod_total_w), mod_total_w)]
            summary_text += [_line('', mod_total_w)]

            cat_hdr_parts = []
            for i, (name, _) in enumerate(col_data):
                cat_hdr_parts.append(_center(name, col_dws[i]))
            cat_hdr = COL_GAP.join(cat_hdr_parts)
            summary_text += [_line(PAD + cat_hdr + PAD, mod_total_w)]

            sub_parts = ['-' * dw for dw in col_dws]
            sub_rule = COL_GAP.join(sub_parts)
            summary_text += [_line(PAD + sub_rule + PAD, mod_total_w)]

            max_rows = max(len(items) for _, items in col_data)
            for row_idx in range(max_rows):
                parts = []
                for col_idx, (_, items) in enumerate(col_data):
                    if row_idx < len(items):
                        parts.append(_ljust(items[row_idx], col_dws[col_idx]))
                    else:
                        parts.append(' ' * col_dws[col_idx])
                row_str = COL_GAP.join(parts)
                summary_text += [_line(PAD + row_str + PAD, mod_total_w)]

            summary_text += [_line('', mod_total_w)]
            summary_text += [_line('-' * mod_total_w, mod_total_w)]
            summary_text += [_line('', mod_total_w)]

            total_params = sum(p for _, p, _ in mods)
            opt_params = sum(p for _, p, _ in opts_m)
            precision_str = 'FP32'
            # 以前这里读 getattr(self, 'use_bf16')：模型对象上根本没有这个属性，
            # 于是永远打印 FP32。算力精度只看 dat 里那一位。
            compute_mode = 'BF16' if bool(opts.get('use_bf16', False)) else 'FP32'

            # VRAM estimation — per-module based on actual param counts
            batch_size = opts.get('batch_size', 1) or 1
            resolution = opts.get('resolution', 128) or 128

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
            # 键名是 gradient_checkpointing：以前这里读 use_gradient_checkpointing
            # （dat 里从来没有这个键），于是「预估显存」永远按"没开检查点"算。
            gc_enabled = bool(opts.get('gradient_checkpointing', False))
            act_mult = 1.75 if gc_enabled else 3.5  # checkpoint ~50% less activation
            a_mb = w_mb * act_mult * (batch_size / 4) * ((resolution / 256) ** 2)
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
        #   Device info（真实硬件，不是 dat 里的 device 选项）
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
