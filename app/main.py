if __name__ == "__main__":
    # Linux: multiprocessing spawn 兼容性修复
    import multiprocessing
    multiprocessing.set_start_method("spawn")

    from core.leras import nn
    nn.initialize_main_env()
    import os
    import sys
    import time
    import argparse

    from core import pathex
    from core import osex
    from pathlib import Path
    from core.interact import interact as io

    if sys.version_info[0] < 3 or (sys.version_info[0] == 3 and sys.version_info[1] < 12):
        raise Exception("本程序至少需要 Python 3.12 版本。")

    class fixPathAction(argparse.Action):
        def __call__(self, parser, namespace, values, option_string=None):
            setattr(namespace, self.dest, os.path.abspath(os.path.expanduser(values)))

    exit_code = 0
    
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()

    # ========== Qt
    qt_parser = subparsers.add_parser("qt", help="Qt 工具").add_subparsers()

    def process_qt_selftest(arguments):
        from core.qtex.qt_selftest import selftest

        print(selftest())

    p = qt_parser.add_parser("selftest", help="打印 Qt 绑定与版本信息")
    p.set_defaults(func=process_qt_selftest)

    def process_extract(arguments):
        osex.set_process_lowest_prio()
        from mainscripts import Extractor
        Extractor.main( detector                = arguments.detector,
                        input_path              = Path(arguments.input_dir),
                        output_path             = Path(arguments.output_dir),
                        output_debug            = arguments.output_debug,
                        manual_fix              = arguments.manual_fix,
                        manual_output_debug_fix = arguments.manual_output_debug_fix,
                        manual_window_size      = arguments.manual_window_size,
                        face_type               = arguments.face_type,
                        max_faces_from_image    = arguments.max_faces_from_image,
                        image_size              = arguments.image_size,
                        jpeg_quality            = arguments.jpeg_quality,
                        cpu_only                = arguments.cpu_only,
                        force_gpu_idxs          = [ int(x) for x in arguments.force_gpu_idxs.split(',') ] if arguments.force_gpu_idxs is not None else None,
                      )

    p = subparsers.add_parser( "extract", help="从图片/帧中提取人脸")
    p.add_argument('--detector', dest="detector", choices=['s3fd','manual'], default=None, help="检测器类型")
    p.add_argument('--input-dir', required=True, action=fixPathAction, dest="input_dir", help="输入目录：包含待处理文件的目录")
    p.add_argument('--output-dir', required=True, action=fixPathAction, dest="output_dir", help="输出目录：提取结果会写入该目录")
    p.add_argument('--output-debug', action="store_true", dest="output_debug", default=None, help="将调试图片写入 <output-dir>_debug 目录")
    p.add_argument('--no-output-debug', action="store_false", dest="output_debug", default=None, help="不写入调试图片到 <output-dir>_debug 目录")
    p.add_argument('--face-type', dest="face_type", choices=['half_face', 'full_face', 'whole_face', 'head', 'mark_only'], default=None)
    p.add_argument('--max-faces-from-image', type=int, dest="max_faces_from_image", default=None, help="每张图片最多提取多少张人脸")
    p.add_argument('--image-size', type=int, dest="image_size", default=None, help="输出人脸图像尺寸")
    p.add_argument('--jpeg-quality', type=int, dest="jpeg_quality", default=None, help="JPEG 质量")
    p.add_argument('--manual-fix', action="store_true", dest="manual_fix", default=False, help="开启手动修复：仅对未识别到人脸的帧进行手动提取")
    p.add_argument('--manual-output-debug-fix', action="store_true", dest="manual_output_debug_fix", default=False, help="手动重提取：对 [output_dir]_debug 目录中被删除的帧进行重新提取")
    p.add_argument('--manual-window-size', type=int, dest="manual_window_size", default=1368, help="手动修复窗口大小。默认：1368")
    p.add_argument('--cpu-only', action="store_true", dest="cpu_only", default=False, help="仅使用 CPU 进行提取")
    p.add_argument('--force-gpu-idxs', dest="force_gpu_idxs", default=None, help="强制选择 GPU idx（逗号分隔）")

    p.set_defaults (func=process_extract)

    def process_sort(arguments):
        osex.set_process_lowest_prio()
        from mainscripts import Sorter
        Sorter.main (input_path=Path(arguments.input_dir), sort_by_method=arguments.sort_by_method)

    p = subparsers.add_parser( "sort", help="对目录中的人脸进行排序")
    p.add_argument('--input-dir', required=True, action=fixPathAction, dest="input_dir", help="输入目录：包含待排序人脸文件的目录")
    p.add_argument('--by', dest="sort_by_method", default=None, choices=("blur", "motion-blur", "face-yaw", "face-pitch", "face-source-rect-size", "hist", "hist-dissim", "brightness", "hue", "black", "origname", "oneface", "final-by-blur", "final-by-size", "absdiff"), help="排序方式。'origname' 表示按原始文件名排序以恢复原始序列" )
    p.set_defaults (func=process_sort)

    def process_util(arguments):
        osex.set_process_lowest_prio()
        from mainscripts import Util

        if arguments.add_landmarks_debug_images:
            Util.add_landmarks_debug_images (input_path=arguments.input_dir)

        if arguments.recover_original_aligned_filename:
            Util.recover_original_aligned_filename (input_path=arguments.input_dir)

        if arguments.save_faceset_metadata:
            Util.save_faceset_metadata_folder (input_path=arguments.input_dir)

        if arguments.restore_faceset_metadata:
            Util.restore_faceset_metadata_folder (input_path=arguments.input_dir)

        if arguments.pack_faceset:
            io.log_info ("正在打包 faceset...\r\n")
            from samplelib import PackedFaceset
            PackedFaceset.pack( Path(arguments.input_dir) )

        if arguments.unpack_faceset:
            io.log_info ("正在解包 faceset...\r\n")
            from samplelib import PackedFaceset
            PackedFaceset.unpack( Path(arguments.input_dir) )
            
        if arguments.export_faceset_mask:
            io.log_info ("正在导出 faceset mask..\r\n")
            Util.export_faceset_mask( Path(arguments.input_dir) )

    p = subparsers.add_parser( "util", help="工具集")
    p.add_argument('--input-dir', required=True, action=fixPathAction, dest="input_dir", help="输入目录：包含待处理文件的目录")
    p.add_argument('--add-landmarks-debug-images', action="store_true", dest="add_landmarks_debug_images", default=False, help="为 aligned 人脸添加 landmarks 调试图")
    p.add_argument('--recover-original-aligned-filename', action="store_true", dest="recover_original_aligned_filename", default=False, help="恢复 aligned 文件的原始文件名")
    p.add_argument('--save-faceset-metadata', action="store_true", dest="save_faceset_metadata", default=False, help="保存 faceset 元数据到文件")
    p.add_argument('--restore-faceset-metadata', action="store_true", dest="restore_faceset_metadata", default=False, help="从文件恢复 faceset 元数据（文件名需与保存时一致）")
    p.add_argument('--pack-faceset', action="store_true", dest="pack_faceset", default=False, help="打包 faceset（PackedFaceset.pack）")
    p.add_argument('--unpack-faceset', action="store_true", dest="unpack_faceset", default=False, help="解包 faceset（PackedFaceset.unpack）")
    p.add_argument('--export-faceset-mask', action="store_true", dest="export_faceset_mask", default=False, help="导出 faceset mask")

    p.set_defaults (func=process_util)

    def process_train(arguments):
        osex.set_process_lowest_prio()


        kwargs = {'model_class_name'         : arguments.model_name,
                  'saved_models_path'        : Path(arguments.model_dir),
                  'training_data_src_path'   : Path(arguments.training_data_src_dir),
                  'training_data_dst_path'   : Path(arguments.training_data_dst_dir),
                  'pretraining_data_path'    : Path(arguments.pretraining_data_dir) if arguments.pretraining_data_dir is not None else None,
                  'pretrained_model_path'    : Path(arguments.pretrained_model_dir) if arguments.pretrained_model_dir is not None else None,
                  'no_preview'               : arguments.no_preview,
                  'force_model_name'         : arguments.force_model_name,
                  'force_gpu_idxs'           : [ int(x) for x in arguments.force_gpu_idxs.split(',') ] if arguments.force_gpu_idxs is not None else None,
                  'cpu_only'                 : arguments.cpu_only,
                  'silent_start'             : arguments.silent_start,
                  'execute_programs'         : [ [int(x[0]), x[1] ] for x in arguments.execute_program ],
                  'debug'                    : arguments.debug,
                  'webui_port'               : arguments.webui_port,
                  'webui_password'           : arguments.webui_password,
                  }
        from mainscripts import Trainer
        Trainer.main(**kwargs)

    p = subparsers.add_parser( "train", help="训练")
    p.add_argument('--training-data-src-dir', required=True, action=fixPathAction, dest="training_data_src_dir", help="SRC faceset（已提取并 aligned）的目录")
    p.add_argument('--training-data-dst-dir', required=True, action=fixPathAction, dest="training_data_dst_dir", help="DST faceset（已提取并 aligned）的目录")
    p.add_argument('--pretraining-data-dir', action=fixPathAction, dest="pretraining_data_dir", default=None, help="可选：用于 pretrain 的 faceset 目录")
    p.add_argument('--pretrained-model-dir', action=fixPathAction, dest="pretrained_model_dir", default=None, help="可选：预训练模型文件目录（当前仅 Quick96 使用）")
    p.add_argument('--model-dir', required=True, action=fixPathAction, dest="model_dir", help="模型保存目录（model_dir）")
    p.add_argument('--model', required=True, dest="model_name", choices=pathex.get_all_dir_names_startswith ( Path(__file__).parent / 'models' , 'Model_'), help="模型类型")
    p.add_argument('--debug', action="store_true", dest="debug", default=False, help="调试样本")
    p.add_argument('--no-preview', action="store_true", dest="no_preview", default=False, help="禁用预览窗口")
    p.add_argument('--force-model-name', dest="force_model_name", default=None, help="强制选择 model/ 目录中的模型实例名")
    p.add_argument('--cpu-only', action="store_true", dest="cpu_only", default=False, help="仅使用 CPU 训练")
    p.add_argument('--force-gpu-idxs', dest="force_gpu_idxs", default=None, help="强制选择 GPU idx（逗号分隔）")
    p.add_argument('--silent-start', action="store_true", dest="silent_start", default=False, help="静默启动：自动选择最佳 GPU 与最近使用的模型")
    p.add_argument('--webui-port', type=int, dest='webui_port', default=6789, help='WebUI 端口 (默认 6789)')
    p.add_argument('--webui-password', type=str, dest='webui_password', default=None, help='WebUI 访问密码 (默认 caiji)')
    p.add_argument('--execute-program', dest="execute_program", default=[], action='append', nargs='+')
    p.set_defaults (func=process_train)
    
    def process_exportdfm(arguments):
        osex.set_process_lowest_prio()
        from tools.export_dfm import export as _export_dfm
        _export_dfm(model_class_name = arguments.model_name,
                    saved_models_path = str(Path(arguments.model_dir)),
                    force_model_name = getattr(arguments, 'force_name', None))

    p = subparsers.add_parser( "exportdfm", help="导出供 DeepFaceLive 使用的模型")
    p.add_argument('--model-dir', required=True, action=fixPathAction, dest="model_dir", help="模型保存目录")
    p.add_argument('--model', required=True, dest="model_name", choices=pathex.get_all_dir_names_startswith ( Path(__file__).parent / 'models' , 'Model_'), help="模型类型")
    p.add_argument('--force-name', default=None, dest="force_name", help="强制模型名称")
    p.set_defaults (func=process_exportdfm)

    def process_merge(arguments):
        osex.set_process_lowest_prio()
        from mainscripts import Merger

        # Collect merge settings from CLI args if any
        merge_settings = {}
        for attr in ['mode','face_type','masked_hist_match','hist_match_threshold',
                     'mask_mode','erode_mask','blur_mask','motion_blur',
                     'output_face_scale','color_transfer','super_resolution',
                     'image_denoise','bicubic_degrade','color_degrade',
                     'sharpen_mode','sharpen_amount']:
            val = getattr(arguments, attr, None)
            if val is not None:
                merge_settings[attr] = val

        Merger.main ( model_class_name       = arguments.model_name,
                      saved_models_path      = Path(arguments.model_dir),
                      force_model_name       = arguments.force_model_name,
                      input_path             = Path(arguments.input_dir),
                      output_path            = Path(arguments.output_dir),
                      output_mask_path       = Path(arguments.output_mask_dir),
                      aligned_path           = Path(arguments.aligned_dir) if arguments.aligned_dir is not None else None,
                      force_gpu_idxs         = arguments.force_gpu_idxs,
                      cpu_only               = arguments.cpu_only,
                      dfm_onnx_path          = Path(arguments.dfm_onnx) if arguments.dfm_onnx else None,
                      xseg_onnx_path         = Path(arguments.xseg_onnx) if arguments.xseg_onnx else None,
                      xseg_onnx_res          = arguments.xseg_onnx_res,
                      pipeline               = arguments.pipeline,
                      merge_settings         = merge_settings if arguments.pipeline else None)

    p = subparsers.add_parser( "merge", help="合成（Merger）")
    p.add_argument('--input-dir', required=True, action=fixPathAction, dest="input_dir", help="输入目录：包含待合成的帧/图片")
    p.add_argument('--output-dir', required=True, action=fixPathAction, dest="output_dir", help="输出目录：合成结果会写入该目录")
    p.add_argument('--output-mask-dir', required=True, action=fixPathAction, dest="output_mask_dir", help="输出 mask 目录：mask 文件会写入该目录")
    p.add_argument('--aligned-dir', action=fixPathAction, dest="aligned_dir", default=None, help="aligned 目录：用于读取提取的人脸对齐信息")
    p.add_argument('--model-dir', required=True, action=fixPathAction, dest="model_dir", help="模型目录（model_dir）")
    p.add_argument('--model', required=True, dest="model_name", choices=pathex.get_all_dir_names_startswith ( Path(__file__).parent / 'models' , 'Model_'), help="模型类型")
    p.add_argument('--force-model-name', dest="force_model_name", default=None, help="强制选择 model/ 目录中的模型实例名")
    p.add_argument('--cpu-only', action="store_true", dest="cpu_only", default=False, help="仅使用 CPU 进行合成")
    p.add_argument('--force-gpu-idxs', dest="force_gpu_idxs", default=None, help="强制选择 GPU idx（逗号分隔）")
    p.add_argument('--dfm-onnx', action=fixPathAction, dest="dfm_onnx", default=None, help="ONNX DFM 模型路径（替代 pytorch 模型做换脸推理）")
    p.add_argument('--xseg-onnx', action=fixPathAction, dest="xseg_onnx", default=None, help="ONNX XSegLite/XSeg 遮罩模型路径（替代 pytorch XSeg 做遮罩提取）")
    p.add_argument('--xseg-onnx-res', type=int, dest="xseg_onnx_res", default=256, help="ONNX XSeg 输入分辨率（min=128, 推荐=256）")
    p.add_argument('--pipeline', action='store_true', dest='pipeline', default=False, help="流水线模式：跳过所有交互，直接合成")
    # 合成参数（--pipeline 时生效，跳过交互询问）
    p.add_argument('--mode', default=None, help="合成模式: overlay/hist-match/seamless/seamless-hist-match/raw-rgb/raw-predict")
    p.add_argument('--face-type', default=None, help="人脸类型: half_face/mid_full/full_face/whole_face/head")
    p.add_argument('--masked-hist-match', type=int, default=None, choices=[0,1], help="直方图匹配遮罩 0/1")
    p.add_argument('--hist-match-threshold', type=int, default=None, help="直方图匹配阈值 0-255")
    p.add_argument('--mask-mode', type=int, default=None, help="遮罩模式 0-9")
    p.add_argument('--erode-mask', type=int, default=None, help="遮罩侵蚀 -400-400")
    p.add_argument('--blur-mask', type=int, default=None, help="遮罩模糊 0-400")
    p.add_argument('--motion-blur', type=int, default=None, help="运动模糊 0-100")
    p.add_argument('--output-face-scale', type=int, default=None, help="输出人脸缩放 -50-50")
    p.add_argument('--color-transfer', default=None, help="色彩迁移: None/rct/lct/mkl/mkl-m/idt/idt-m/sot-m/mix-m")
    p.add_argument('--super-resolution', type=int, default=None, help="超分辨率 0-100")
    p.add_argument('--image-denoise', type=int, default=None, help="去噪 0-500")
    p.add_argument('--bicubic-degrade', type=int, default=None, help="双三次降质 0-100")
    p.add_argument('--color-degrade', type=int, default=None, help="颜色降质 0-100")
    p.add_argument('--sharpen-mode', type=int, default=None, help="锐化模式 0=无/1=方框/2=高斯")
    p.add_argument('--sharpen-amount', type=int, default=None, help="锐化强度 -100-100")
    p.set_defaults(func=process_merge)

    videoed_parser = subparsers.add_parser( "videoed", help="视频处理").add_subparsers()

    def process_videoed_extract_video(arguments):
        osex.set_process_lowest_prio()
        from mainscripts import VideoEd
        VideoEd.extract_video (arguments.input_file,
                               arguments.output_dir,
                               arguments.output_ext,
                               arguments.fps)
    p = videoed_parser.add_parser( "extract-video", help="从视频中提取图片帧")
    p.add_argument('--input-file', required=True, action=fixPathAction, dest="input_file", help="输入文件。可用 .*-扩展名表示匹配并选择第一个文件")
    p.add_argument('--output-dir', required=True, action=fixPathAction, dest="output_dir", help="输出目录：提取的图片会写入该目录")
    p.add_argument('--output-ext', dest="output_ext", default=None, help="输出图片格式（扩展名）")
    p.add_argument('--fps', type=int, dest="fps", default=None, help="每秒提取多少帧。0 表示全帧率")
    p.set_defaults(func=process_videoed_extract_video)

    def process_videoed_cut_video(arguments):
        osex.set_process_lowest_prio()
        from mainscripts import VideoEd
        VideoEd.cut_video (arguments.input_file,
                           arguments.from_time,
                           arguments.to_time,
                           arguments.audio_track_id,
                           arguments.bitrate)
    p = videoed_parser.add_parser( "cut-video", help="裁剪视频")
    p.add_argument('--input-file', required=True, action=fixPathAction, dest="input_file", help="输入文件。可用 .*-扩展名表示匹配并选择第一个文件")
    p.add_argument('--from-time', dest="from_time", default=None, help="起始时间，例如 00:00:00.000")
    p.add_argument('--to-time', dest="to_time", default=None, help="结束时间，例如 00:00:00.000")
    p.add_argument('--audio-track-id', type=int, dest="audio_track_id", default=None, help="指定音轨 ID")
    p.add_argument('--bitrate', type=int, dest="bitrate", default=None, help="输出码率（单位：Mbps）")
    p.set_defaults(func=process_videoed_cut_video)

    def process_videoed_denoise_image_sequence(arguments):
        osex.set_process_lowest_prio()
        from mainscripts import VideoEd
        VideoEd.denoise_image_sequence (arguments.input_dir, arguments.factor)
    p = videoed_parser.add_parser( "denoise-image-sequence", help="对图片序列降噪并保留锐利边缘（可减少预测脸的像素抖动）")
    p.add_argument('--input-dir', required=True, action=fixPathAction, dest="input_dir", help="输入目录：待处理图片序列")
    p.add_argument('--factor', type=int, dest="factor", default=None, help="降噪强度（1-20）")
    p.set_defaults(func=process_videoed_denoise_image_sequence)

    def process_videoed_video_from_sequence(arguments):
        osex.set_process_lowest_prio()
        from mainscripts import VideoEd
        VideoEd.video_from_sequence (input_dir      = arguments.input_dir,
                                     output_file    = arguments.output_file,
                                     reference_file = arguments.reference_file,
                                     ext      = arguments.ext,
                                     fps      = arguments.fps,
                                     bitrate  = arguments.bitrate,
                                     include_audio = arguments.include_audio,
                                     lossless = arguments.lossless)

    p = videoed_parser.add_parser( "video-from-sequence", help="由图片序列生成视频")
    p.add_argument('--input-dir', required=True, action=fixPathAction, dest="input_dir", help="输入目录：图片序列所在目录")
    p.add_argument('--output-file', required=True, action=fixPathAction, dest="output_file", help="输出视频文件路径")
    p.add_argument('--reference-file', action=fixPathAction, dest="reference_file", help="参考文件：用于确定正确 FPS 并从中拷贝音频。可用 .*-扩展名表示匹配并选择第一个文件")
    p.add_argument('--ext', dest="ext", default='png', help="输入图片格式（扩展名）")
    p.add_argument('--fps', type=int, dest="fps", default=None, help="输出视频 FPS（若设置 reference-file 将以其为准）")
    p.add_argument('--bitrate', type=int, dest="bitrate", default=None, help="输出码率（单位：Mbps）")
    p.add_argument('--include-audio', action="store_true", dest="include_audio", default=False, help="包含 reference-file 的音频")
    p.add_argument('--lossless', action="store_true", dest="lossless", default=False, help="无损（PNG 编码）")

    p.set_defaults(func=process_videoed_video_from_sequence)

    facesettool_parser = subparsers.add_parser( "facesettool", help="Faceset 工具").add_subparsers()

    def process_faceset_enhancer(arguments):
        osex.set_process_lowest_prio()
        from mainscripts import FacesetEnhancer
        FacesetEnhancer.process_folder ( Path(arguments.input_dir),
                                         cpu_only=arguments.cpu_only,
                                         force_gpu_idxs=arguments.force_gpu_idxs
                                       )

    p = facesettool_parser.add_parser ("enhance", help="增强 DFL faceset 细节")
    p.add_argument('--input-dir', required=True, action=fixPathAction, dest="input_dir", help="输入目录：aligned faces 所在目录")
    p.add_argument('--cpu-only', action="store_true", dest="cpu_only", default=False, help="仅使用 CPU 处理")
    p.add_argument('--force-gpu-idxs', dest="force_gpu_idxs", default=None, help="强制选择 GPU idx（逗号分隔）")

    p.set_defaults(func=process_faceset_enhancer)
    
    
    p = facesettool_parser.add_parser ("resize", help="重采样/调整 DFL faceset")
    p.add_argument('--input-dir', required=True, action=fixPathAction, dest="input_dir", help="输入目录：aligned faces 所在目录")

    def process_faceset_resizer(arguments):
        osex.set_process_lowest_prio()
        from mainscripts import FacesetResizer
        FacesetResizer.process_folder ( Path(arguments.input_dir) )
    p.set_defaults(func=process_faceset_resizer)

    def process_dev_test(arguments):
        osex.set_process_lowest_prio()
        from mainscripts import dev_misc
        dev_misc.dev_gen_mask_files( arguments.input_dir )

    p = subparsers.add_parser( "dev_test", help="开发/调试工具")
    p.add_argument('--input-dir', required=True, action=fixPathAction, dest="input_dir", help="输入目录")
    p.set_defaults (func=process_dev_test)
    
    # ========== XSeg
    xseg_parser = subparsers.add_parser( "xseg", help="XSeg 工具").add_subparsers()
    
    p = xseg_parser.add_parser( "editor", help="XSeg 编辑器")

    def process_xsegeditor(arguments):
        osex.set_process_lowest_prio()
        global exit_code
        try:
            from XSegEditor import XSegEditor
        except Exception as e:
            io.log_err(
                "XSeg 编辑器需要 Qt 绑定（PyQt5 或 PySide6），但当前环境未安装。\n"
                "提示：可先运行 `python main.py qt selftest` 诊断，然后在受支持的 Python 版本上安装 Qt 绑定（推荐 3.10-3.12）。\n"
                f"导入错误：{type(e).__name__}: {e}"
            )
            exit_code = 1
            return
        exit_code = XSegEditor.start (Path(arguments.input_dir))
        
    p.add_argument('--input-dir', required=True, action=fixPathAction, dest="input_dir")

    p.set_defaults (func=process_xsegeditor)
  
    p = xseg_parser.add_parser( "apply", help="将训练好的 XSeg 模型应用到已提取的人脸")

    def process_xsegapply(arguments):
        osex.set_process_lowest_prio()
        from mainscripts import XSegUtil
        XSegUtil.apply_xseg (Path(arguments.input_dir), Path(arguments.model_dir))
    p.add_argument('--input-dir', required=True, action=fixPathAction, dest="input_dir")
    p.add_argument('--model-dir', required=True, action=fixPathAction, dest="model_dir")
    p.set_defaults (func=process_xsegapply)

    p = xseg_parser.add_parser("train", help="训练 XSeg 模型（仅 PyTorch）")

    def process_xsegtrain(arguments):
        osex.set_process_lowest_prio()
        from mainscripts import Trainer

        kwargs = {
            'model_class_name': 'XSeg',
            'saved_models_path': Path(arguments.model_dir),
            'training_data_src_path': Path(arguments.training_data_src_dir),
            'training_data_dst_path': Path(arguments.training_data_dst_dir),
            'pretraining_data_path': Path(arguments.pretraining_data_dir) if arguments.pretraining_data_dir is not None else None,
            'pretrained_model_path': None,
            'no_preview': arguments.no_preview,
            'force_model_name': arguments.force_model_name,
            'force_gpu_idxs': [int(x) for x in arguments.force_gpu_idxs.split(',')] if arguments.force_gpu_idxs is not None else None,
            'cpu_only': arguments.cpu_only,
            'silent_start': arguments.silent_start,
            'execute_programs': [],
            'debug': arguments.debug,
            'webui_port': getattr(arguments, 'webui_port', 6789),
            'webui_password': getattr(arguments, 'webui_password', None),
        }
        Trainer.main(**kwargs)

    p.add_argument(
        '--training-data-src-dir',
        required=True,
        action=fixPathAction,
        dest='training_data_src_dir',
        help='SRC faceset（已提取并 aligned）的目录',
    )
    p.add_argument(
        '--training-data-dst-dir',
        required=True,
        action=fixPathAction,
        dest='training_data_dst_dir',
        help='DST faceset（已提取并 aligned）的目录',
    )
    p.add_argument(
        '--pretraining-data-dir',
        action=fixPathAction,
        dest='pretraining_data_dir',
        default=None,
        help='可选：用于 pretrain 的目录',
    )
    p.add_argument(
        '--model-dir',
        required=True,
        action=fixPathAction,
        dest='model_dir',
        help='模型保存目录（model_dir）',
    )
    p.add_argument('--debug', action='store_true', dest='debug', default=False, help='调试样本')
    p.add_argument('--no-preview', action='store_true', dest='no_preview', default=False, help='禁用预览窗口')
    p.add_argument('--force-model-name', dest='force_model_name', default=None, help='强制选择 model/ 目录中的模型实例名')
    p.add_argument('--cpu-only', action='store_true', dest='cpu_only', default=False, help='仅使用 CPU 训练')
    p.add_argument('--force-gpu-idxs', dest='force_gpu_idxs', default=None, help='强制选择 GPU idx（逗号分隔）')
    p.add_argument('--silent-start', action='store_true', dest='silent_start', default=False, help='静默启动：自动选择最佳 GPU 与最近使用的模型')
    p.set_defaults(func=process_xsegtrain)
    
    
    p = xseg_parser.add_parser( "remove", help="移除已应用的 XSeg mask")
    def process_xsegremove(arguments):
        osex.set_process_lowest_prio()
        from mainscripts import XSegUtil
        XSegUtil.remove_xseg (Path(arguments.input_dir) )
    p.add_argument('--input-dir', required=True, action=fixPathAction, dest="input_dir")
    p.set_defaults (func=process_xsegremove)
    
    
    p = xseg_parser.add_parser( "remove_labels", help="移除 XSeg 标签")
    def process_xsegremovelabels(arguments):
        osex.set_process_lowest_prio()
        from mainscripts import XSegUtil
        XSegUtil.remove_xseg_labels (Path(arguments.input_dir) )
    p.add_argument('--input-dir', required=True, action=fixPathAction, dest="input_dir")
    p.set_defaults (func=process_xsegremovelabels)
    
    
    p = xseg_parser.add_parser( "fetch", help="将包含 XSeg 多边形标注的人脸复制到 <input_dir>_xseg 目录")

    def process_xsegfetch(arguments):
        osex.set_process_lowest_prio()
        from mainscripts import XSegUtil
        XSegUtil.fetch_xseg (Path(arguments.input_dir) )
    p.add_argument('--input-dir', required=True, action=fixPathAction, dest="input_dir")
    p.set_defaults (func=process_xsegfetch)
    
    def bad_args(arguments):
        parser.print_help()
        exit(0)
    parser.set_defaults(func=bad_args)

    arguments = parser.parse_args()
    arguments.func(arguments)

    if exit_code == 0:
        print ("完成。")
        
    exit(exit_code)
    
'''
import code
code.interact(local=dict(globals(), **locals()))
'''
