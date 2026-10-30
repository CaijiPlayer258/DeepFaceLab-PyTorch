import subprocess
import numpy as np
import ffmpeg
from pathlib import Path
from core import pathex
from core import osex
from core.interact import interact as io


def extract_video(input_file, output_dir, output_ext=None, fps=None):
    input_file_path = Path(input_file)
    output_path = Path(output_dir)

    if not output_path.exists():
        output_path.mkdir(exist_ok=True)

    if input_file_path.suffix == '.*':
        input_file_path = pathex.get_first_file_by_stem(input_file_path.parent, input_file_path.stem)
    else:
        if not input_file_path.exists():
            input_file_path = None

    if input_file_path is None:
        io.log_err("未找到输入文件。")
        return

    if fps is None:
        fps = io.input_int("输入 FPS", 0, help_message="每秒提取多少帧。0 表示使用视频原始全帧率")

    if output_ext is None:
        output_ext = io.input_str(
            "输出图片格式",
            "png",
            ["png", "jpg"],
            help_message="png 为无损，但在机械硬盘上提取速度可能更慢，并且比 jpg 占用更多磁盘空间。",
        )

    for filename in pathex.get_image_paths(output_path, ['.' + output_ext]):
        Path(filename).unlink()

    job = ffmpeg.input(str(input_file_path))

    kwargs = {'pix_fmt': 'rgb24'}
    if fps != 0:
        kwargs.update({'r': str(fps)})

    if output_ext == 'jpg':
        kwargs.update({'q:v': '2'})  # highest quality for jpg

    job = job.output(str(output_path / ('%5d.' + output_ext)), **kwargs)

    ffmpeg_cmd = osex.get_ffmpeg_path()
    if ffmpeg_cmd is None:
        io.log_err('未找到 ffmpeg。请安装 ffmpeg，或将 ffmpeg 可执行文件放到 tools/ 目录（Windows：tools/ffmpeg.exe），或设置环境变量 DFL_FFMPEG 指向 ffmpeg。')
        return

    try:
        job = job.run(cmd=ffmpeg_cmd)
    except:
        io.log_err("ffmpeg 执行失败，命令行：" + str(job.compile(cmd=ffmpeg_cmd)))


def cut_video(input_file, from_time=None, to_time=None, audio_track_id=None, bitrate=None):
    input_file_path = Path(input_file)
    if input_file_path is None:
        io.log_err("未找到输入文件。")
        return

    output_file_path = input_file_path.parent / (input_file_path.stem + "_cut" + input_file_path.suffix)

    if from_time is None:
        from_time = io.input_str("起始时间", "00:00:00.000")

    if to_time is None:
        to_time = io.input_str("结束时间", "00:00:00.000")

    if audio_track_id is None:
        audio_track_id = io.input_int("指定音轨 ID", 0)

    if bitrate is None:
        bitrate = max(1, io.input_int("输出码率（Mbps）", 25))

    kwargs = {
        "c:v": "libx264",
        "b:v": "%dM" % (bitrate),
        "pix_fmt": "yuv420p",
    }

    job = ffmpeg.input(str(input_file_path), ss=from_time, to=to_time)

    job_v = job['v:0']
    job_a = job['a:' + str(audio_track_id) + '?']

    job = ffmpeg.output(job_v, job_a, str(output_file_path), **kwargs).overwrite_output()

    ffmpeg_cmd = osex.get_ffmpeg_path()
    if ffmpeg_cmd is None:
        io.log_err('未找到 ffmpeg。请安装 ffmpeg，或将 ffmpeg 可执行文件放到 tools/ 目录（Windows：tools/ffmpeg.exe），或设置环境变量 DFL_FFMPEG 指向 ffmpeg。')
        return

    try:
        job = job.run(cmd=ffmpeg_cmd)
    except:
        io.log_err("ffmpeg 执行失败，命令行：" + str(job.compile(cmd=ffmpeg_cmd)))


def denoise_image_sequence(input_dir, ext=None, factor=None):
    input_path = Path(input_dir)

    if not input_path.exists():
        io.log_err("未找到输入目录。")
        return

    image_paths = [Path(filepath) for filepath in pathex.get_image_paths(input_path)]

    # Check extension of all images
    image_paths_suffix = None
    for filepath in image_paths:
        if image_paths_suffix is None:
            image_paths_suffix = filepath.suffix
        else:
            if filepath.suffix != image_paths_suffix:
                io.log_err(f"{input_path.name} 目录内所有图片必须使用相同的扩展名。")
                return

    if factor is None:
        factor = np.clip(io.input_int("降噪强度？", 7, add_info="1-20"), 1, 20)

    # Rename to temporary filenames
    for i, filepath in io.progress_bar_generator(enumerate(image_paths), "重命名中", leave=False):
        src = filepath
        dst = filepath.parent / (f'{i+1:06}_{filepath.name}')
        try:
            src.rename(dst)
        except:
            io.log_error('重命名失败 %s' % (src.name))
            return

    # Rename to sequential filenames
    for i, filepath in io.progress_bar_generator(enumerate(image_paths), "重命名中", leave=False):
        src = filepath.parent / (f'{i+1:06}_{filepath.name}')
        dst = filepath.parent / (f'{i+1:06}{filepath.suffix}')
        try:
            src.rename(dst)
        except:
            io.log_error('重命名失败 %s' % (src.name))
            return

    # Process image sequence in ffmpeg
    kwargs = {}
    if image_paths_suffix == '.jpg':
        kwargs.update({'q:v': '2'})

    job = (
        ffmpeg
        .input(str(input_path / ('%6d' + image_paths_suffix)))
        .filter("hqdn3d", factor, factor, 5, 5)
        .output(str(input_path / ('%6d' + image_paths_suffix)), **kwargs)
    )

    ffmpeg_cmd = osex.get_ffmpeg_path()
    if ffmpeg_cmd is None:
        io.log_err('未找到 ffmpeg。请安装 ffmpeg，或将 ffmpeg 可执行文件放到 tools/ 目录（Windows：tools/ffmpeg.exe），或设置环境变量 DFL_FFMPEG 指向 ffmpeg。')
        return

    try:
        job = job.run(cmd=ffmpeg_cmd)
    except:
        io.log_err("ffmpeg 执行失败，命令行：" + str(job.compile(cmd=ffmpeg_cmd)))

    # Rename to temporary filenames
    for i, filepath in io.progress_bar_generator(enumerate(image_paths), "重命名中", leave=False):
        src = filepath.parent / (f'{i+1:06}{filepath.suffix}')
        dst = filepath.parent / (f'{i+1:06}_{filepath.name}')
        try:
            src.rename(dst)
        except:
            io.log_error('重命名失败 %s' % (src.name))
            return

    # Rename to initial filenames
    for i, filepath in io.progress_bar_generator(enumerate(image_paths), "重命名中", leave=False):
        src = filepath.parent / (f'{i+1:06}_{filepath.name}')
        dst = filepath

        try:
            src.rename(dst)
        except:
            io.log_error('重命名失败 %s' % (src.name))
            return


def video_from_sequence(input_dir, output_file, reference_file=None, ext=None, fps=None, bitrate=None, include_audio=False, lossless=None):
    input_path = Path(input_dir)
    output_file_path = Path(output_file)
    reference_file_path = Path(reference_file) if reference_file is not None else None

    if not input_path.exists():
        io.log_err("未找到输入目录。")
        return

    if not output_file_path.parent.exists():
        output_file_path.parent.mkdir(parents=True, exist_ok=True)
        return

    out_ext = output_file_path.suffix

    if ext is None:
        ext = io.input_str("输入图片格式（扩展名）", "png")

    if lossless is None:
        lossless = io.input_bool("是否使用无损编码", False)

    video_id = None
    audio_id = None
    ref_in_a = None
    if reference_file_path is not None:
        if reference_file_path.suffix == '.*':
            reference_file_path = pathex.get_first_file_by_stem(reference_file_path.parent, reference_file_path.stem)
        else:
            if not reference_file_path.exists():
                reference_file_path = None

        if reference_file_path is None:
            io.log_err("未找到参考文件。")
            return

        # probing reference file
        ffprobe_cmd = osex.get_ffprobe_path()
        if ffprobe_cmd is None:
            io.log_err('未找到 ffprobe。请安装 ffmpeg（通常会自带 ffprobe），或将 ffprobe 可执行文件放到 tools/ 目录（Windows：tools/ffprobe.exe），或设置环境变量 DFL_FFPROBE 指向 ffprobe。')
            return

        probe = ffmpeg.probe(str(reference_file_path), cmd=ffprobe_cmd)

        # getting first video and audio streams id with fps
        for stream in probe['streams']:
            if video_id is None and stream['codec_type'] == 'video':
                video_id = stream['index']
                fps = stream['r_frame_rate']

            if audio_id is None and stream['codec_type'] == 'audio':
                audio_id = stream['index']

        if audio_id is not None:
            # has audio track
            ref_in_a = ffmpeg.input(str(reference_file_path))[str(audio_id)]

    if fps is None:
        # if fps not specified and not overwritten by reference-file
        fps = max(1, io.input_int("请输入 FPS", 25))

    if not lossless and bitrate is None:
        bitrate = max(1, io.input_int("输出视频码率（MB/s）", 16))

    input_image_paths = pathex.get_image_paths(input_path)

    i_in = ffmpeg.input('pipe:', format='image2pipe', r=fps)

    output_args = [i_in]

    if include_audio and ref_in_a is not None:
        output_args += [ref_in_a]

    output_args += [str(output_file_path)]

    output_kwargs = {}

    if lossless:
        output_kwargs.update({
            "c:v": "libx264",
            "crf": "0",
            "pix_fmt": "yuv420p",
        })
    else:
        output_kwargs.update({
            "c:v": "libx264",
            "b:v": "%dM" % (bitrate),
            "pix_fmt": "yuv420p",
        })

    ffmpeg_cmd = osex.get_ffmpeg_path()
    if ffmpeg_cmd is None:
        io.log_err('未找到 ffmpeg。请安装 ffmpeg，或将 ffmpeg 可执行文件放到 tools/ 目录（Windows：tools/ffmpeg.exe），或设置环境变量 DFL_FFMPEG 指向 ffmpeg。')
        return

    if include_audio and ref_in_a is not None:
        output_kwargs.update({
            "c:a": "aac",
            "b:a": "192k",
            "ar": "48000",
            "strict": "experimental",
        })

    job = (ffmpeg.output(*output_args, **output_kwargs).overwrite_output())

    try:
        job_run = job.run_async(cmd=ffmpeg_cmd, pipe_stdin=True)

        for image_path in input_image_paths:
            with open(image_path, "rb") as f:
                image_bytes = f.read()
                job_run.stdin.write(image_bytes)

        job_run.stdin.close()
        job_run.wait()
    except:
        io.log_err("ffmpeg 失败，任务命令行：" + str(job.compile(cmd=ffmpeg_cmd)))
