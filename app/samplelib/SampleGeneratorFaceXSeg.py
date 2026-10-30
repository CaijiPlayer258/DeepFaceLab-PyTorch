import multiprocessing
import pickle
import time
import traceback
from enum import IntEnum

import cv2
import numpy as np
from pathlib import Path
from core import imagelib, mplib, pathex
from core.imagelib import sd
from core.cv2ex import *
from core.interact import interact as io
from core.joblib import Subprocessor, SubprocessGenerator, ThisThreadGenerator
from facelib import LandmarksProcessor
from samplelib import (SampleGeneratorBase, SampleLoader, SampleProcessor, SampleType, Sample)

class SampleGeneratorFaceXSeg(SampleGeneratorBase):
    def __init__ (self, paths, debug=False, batch_size=1, resolution=256, face_type=None,
                        generators_count=4, data_format="NHWC",
                        skip_filter=False, **kwargs):

        super().__init__(debug, batch_size)
        self.initialized = False

        self.skip_filter = skip_filter

        if skip_filter:
            import glob as _g, threading
            self._files = []
            for p in paths:
                self._files += _g.glob(str(p) + '/*.jpg') + _g.glob(str(p) + '/*.png')
            self._files = sorted(set(self._files))
            io.log_info(f"Fast loader: {len(self._files)} files (no validation).")
            self._resolution = resolution
            self._face_type = face_type
            self._data_format = data_format
            self._batch_size = batch_size
            self._last_fnames = []
            self._prefetched = None
            self._prefetch_lock = threading.Lock()
            self._prefetch_thread = None
            # Generate first batch immediately
            self._prefetched = self._skip_generate_next()
            self.initialized = True
            return

        samples = sum([ SampleLoader.load (SampleType.FACE, path) for path in paths ]  )
        seg_sample_idxs = list(range(len(samples)))
        io.log_info(f"Loaded {len(samples)} samples (runtime filter will skip empty masks).")

        if self.debug:
            self.generators_count = 1
        else:
            self.generators_count = max(1, generators_count)

        args = (samples, seg_sample_idxs, resolution, face_type, data_format)
        if self.debug:
            self.generators = [ThisThreadGenerator ( self.batch_func, args )]
        else:
            self.generators = [SubprocessGenerator ( self.batch_func, args, start_now=False ) for i in range(self.generators_count) ]

            SubprocessGenerator.start_in_parallel( self.generators )

        self.generator_counter = -1

        self.initialized = True

    #overridable
    def is_initialized(self):
        return self.initialized

    def __iter__(self):
        return self

    def __next__(self):
        if self.skip_filter:
            import threading
            # Wait for previous prefetch to finish (if any)
            if self._prefetch_thread is not None:
                self._prefetch_thread.join()
            result = self._prefetched
            if result is None:
                result = self._skip_generate_next()
            # Start loading next batch in background
            self._prefetch_thread = threading.Thread(target=self._prefetch_next, daemon=True)
            self._prefetch_thread.start()
            return result
        self.generator_counter += 1
        generator = self.generators[self.generator_counter % len(self.generators) ]
        result = next(generator)
        if len(result) >= 3:
            self._last_fnames = result[2]
            return result[:2]
        return result

    def _is_valid_mask(self, mask):
        """检查遮罩是否非空（至少覆盖 15% 像素，正常人脸遮罩在 30-70%）。"""
        if mask is None:
            return False
        if mask.max() < 0.5:
            return False
        return (mask > 0.5).sum() >= mask.size * 0.25

    def _load_image_mask(self, fpath):
        """Load image + XSeg mask + meta from a file. Returns (img, mask, face_type, landmarks) or None."""
        img = cv2.imread(fpath)
        if img is None:
            return None
        img = img.astype(np.float32) / 255.0
        face_type, landmarks = None, None
        mask = None
        try:
            from DFLIMG.DFLJPG import DFLJPG
            inst = DFLJPG.load(fpath)
            if inst is not None:
                face_type = inst.get_face_type()
                landmarks = inst.get_landmarks()
                if inst.has_xseg_mask():
                    mask = inst.get_xseg_mask()
                    mask = np.clip(mask, 0, 1)
                    mask[mask < 0.5] = 0.0
                    mask[mask >= 0.5] = 1.0
        except Exception:
            pass
        # 没有有效遮罩 → 返回 None 让上层跳过
        if not self._is_valid_mask(mask):
            return None
        if mask.ndim == 2:
            mask = mask[..., None]
        if mask.shape[:2] != img.shape[:2]:
            mask = cv2.resize(mask, (img.shape[1], img.shape[0]))
        return img, mask, face_type, landmarks

    def _prefetch_next(self):
        """Load next batch in background thread."""
        try:
            self._prefetched = self._skip_generate_next()
        except Exception as e:
            io.log_err(f'Prefetch failed: {e}')
            self._prefetched = None

    def _skip_generate_next(self):
        bs = self._batch_size
        res = self._resolution
        imgs = np.zeros((bs, 3, res, res), dtype=np.float32)
        masks = np.zeros((bs, 1, res, res), dtype=np.float32)
        fnames = []

        for i in range(bs):
            retries = 0
            while retries < 100:
                idx = np.random.randint(0, len(self._files))
                fpath = self._files[idx]
                loaded = self._load_image_mask(fpath)
                if loaded is not None:
                    break
                self._files.pop(idx)
                retries += 1
                if not self._files:
                    raise Exception('No valid files remain after filtering.')
            if retries >= 100:
                imgs[i] = 0.0; masks[i] = 0.0; fnames.append(''); continue

            img, mask, ft, lm = loaded
            fnames.append(Path(fpath).name)

            if ft is None or ft == self._face_type or lm is None:
                if img.shape[1] != res:
                    img = cv2.resize(img, (res, res), interpolation=cv2.INTER_LANCZOS4)
                    mask = cv2.resize(mask, (res, res), interpolation=cv2.INTER_LANCZOS4)
            else:
                mat = LandmarksProcessor.get_transform_mat(lm, res, self._face_type)
                img = cv2.warpAffine(img, mat, (res, res),
                                     borderMode=cv2.BORDER_CONSTANT, flags=cv2.INTER_LANCZOS4)
                mask = cv2.warpAffine(mask, mat, (res, res),
                                     borderMode=cv2.BORDER_CONSTANT, flags=cv2.INTER_LANCZOS4)

            if mask.ndim == 2:
                mask = mask[..., None]

            # ---- Data augmentation (same as original batch_func) ----
            # Random background from another file
            if np.random.randint(2) == 0 and len(self._files) > 1:
                bg_idx = np.random.randint(0, len(self._files))
                bg_loaded = self._load_image_mask(self._files[bg_idx])
                if bg_loaded is not None:
                    bg_img, bg_mask, _ft, _lm = bg_loaded
                    if bg_img.shape[0] != res:
                        bg_img = cv2.resize(bg_img, (res, res), interpolation=cv2.INTER_LANCZOS4)
                        bg_mask = cv2.resize(bg_mask, (res, res), interpolation=cv2.INTER_LANCZOS4)
                    bg_wp = imagelib.gen_warp_params(res, True,
                                                     rotation_range=[-180, 180],
                                                     scale_range=[-0.10, 0.10],
                                                     tx_range=[-0.10, 0.10],
                                                     ty_range=[-0.10, 0.10])
                    bg_img = imagelib.warp_by_params(bg_wp, bg_img,
                                                     can_warp=False, can_transform=True,
                                                     can_flip=True, border_replicate=True)
                    bg_mask = imagelib.warp_by_params(bg_wp, bg_mask,
                                                      can_warp=False, can_transform=True,
                                                      can_flip=True, border_replicate=False)
                    bg_img = bg_img * (1 - bg_mask)
                    if np.random.randint(2) == 0:
                        bg_img = imagelib.apply_random_hsv_shift(bg_img)
                    else:
                        bg_img = imagelib.apply_random_rgb_levels(bg_img)
                    c = 1.0 - (1 - bg_mask) * (1 - mask)
                    r = 0.25 + np.random.uniform() * 0.85
                    img = img * c + img * (1 - c) * r + bg_img * (1 - c) * (1 - r)

            # Warp augmentation
            wp = imagelib.gen_warp_params(res, True,
                                          rotation_range=[-10, 10],
                                          scale_range=[-0.05, 0.05],
                                          tx_range=[-0.05, 0.05],
                                          ty_range=[-0.05, 0.05])
            img = imagelib.warp_by_params(wp, img,
                                          can_warp=True, can_transform=True,
                                          can_flip=True, border_replicate=True)
            mask = imagelib.warp_by_params(wp, mask,
                                           can_warp=True, can_transform=True,
                                           can_flip=True, border_replicate=False)
            img = np.clip(img.astype(np.float32), 0, 1)
            mask[mask < 0.5] = 0.0
            mask[mask >= 0.5] = 1.0
            mask = np.clip(mask, 0, 1)

            # Random face flare / bg flare
            if np.random.randint(2) == 0:
                krn = np.random.randint(res // 4, res)
                krn = krn - krn % 2 + 1
                img = img + cv2.GaussianBlur(img * mask, (krn, krn), 0)
            if np.random.randint(2) == 0:
                krn = np.random.randint(res // 4, res)
                krn = krn - krn % 2 + 1
                img = img + cv2.GaussianBlur(img * (1 - mask), (krn, krn), 0)

            # HSV / RGB
            if np.random.randint(2) == 0:
                img = imagelib.apply_random_hsv_shift(img,
                        mask=sd.random_circle_faded([res, res]))
            else:
                img = imagelib.apply_random_rgb_levels(img,
                        mask=sd.random_circle_faded([res, res]))

            # Sharpen or blur
            if np.random.randint(2) == 0:
                img = imagelib.apply_random_sharpen(img, 25, 5,
                        mask=sd.random_circle_faded([res, res]))
            else:
                img = imagelib.apply_random_motion_blur(img, 25, 5,
                        mask=sd.random_circle_faded([res, res]))
                img = imagelib.apply_random_gaussian_blur(img, 25, 5,
                        mask=sd.random_circle_faded([res, res]))

            # Resize jitter
            if np.random.randint(2) == 0:
                img = imagelib.apply_random_nearest_resize(img, 25, 75,
                        mask=sd.random_circle_faded([res, res]))
            else:
                img = imagelib.apply_random_bilinear_resize(img, 25, 75,
                        mask=sd.random_circle_faded([res, res]))
            img = np.clip(img, 0, 1)

            # JPEG compression
            img = imagelib.apply_random_jpeg_compress(img, 25,
                    mask=sd.random_circle_faded([res, res]))

            if self._data_format == 'NCHW':
                imgs[i] = np.transpose(img, (2, 0, 1))
                masks[i] = np.transpose(mask, (2, 0, 1))
            else:
                imgs[i] = img
                masks[i] = mask

        self._last_fnames = fnames
        return [imgs, masks]

    def get_last_filenames(self):
        return getattr(self, '_last_fnames', [])

    def batch_func(self, param ):
        samples, seg_sample_idxs, resolution, face_type, data_format = param

        shuffle_idxs = []
        bg_shuffle_idxs = []

        random_flip = True
        rotation_range=[-10,10]
        scale_range=[-0.05, 0.05]
        tx_range=[-0.05, 0.05]
        ty_range=[-0.05, 0.05]

        random_bilinear_resize_chance, random_bilinear_resize_max_size_per = 25,75
        sharpen_chance, sharpen_kernel_max_size = 25, 5
        motion_blur_chance, motion_blur_mb_max_size = 25, 5
        gaussian_blur_chance, gaussian_blur_kernel_max_size = 25, 5
        random_jpeg_compress_chance = 25

        def gen_img_mask(sample):
            # skip_filter mode: minimal Sample, filename is the full path
            use_skip = (getattr(sample, 'landmarks', None) is None and
                        not hasattr(sample, 'seg_ie_polys') and
                        not sample.has_xseg_mask())
            if use_skip:
                img = cv2.imread(sample.filename)
                if img is None:
                    raise Exception(f'cannot read {sample.filename}')
                img = img.astype(np.float32) / 255.0
                h, w, c = img.shape
                try:
                    from DFLIMG.DFLJPG import DFLJPG
                    inst = DFLJPG.load(sample.filename)
                    if inst is not None and inst.has_xseg_mask():
                        mask = inst.get_xseg_mask()
                        mask[mask < 0.5] = 0.0
                        mask[mask >= 0.5] = 1.0
                    else:
                        raise Exception(f'no xseg mask in {sample.filename}')
                except Exception:
                    raise Exception(f'no xseg mask in {sample.filename}')
            else:
                img = sample.load_bgr()
                h,w,c = img.shape
                mask = None
                if sample.seg_ie_polys.has_polys():
                    mask = np.zeros ((h,w,1), dtype=np.float32)
                    sample.seg_ie_polys.overlay_mask(mask)
                elif sample.has_xseg_mask():
                    mask = sample.get_xseg_mask()
                    mask[mask < 0.5] = 0.0
                    mask[mask >= 0.5] = 1.0
                else:
                    raise Exception(f'no mask in sample {sample.filename}')
                # 运行时校验：加载后再次验证遮罩有效
                if mask is None or mask.max() < 0.5 or (mask > 0.5).sum() < mask.size * 0.25:
                    raise Exception(f'empty mask in {sample.filename}')

            if face_type == sample.face_type:
                if w != resolution:
                    img = cv2.resize( img, (resolution, resolution), interpolation=cv2.INTER_LANCZOS4 )
                    mask = cv2.resize( mask, (resolution, resolution), interpolation=cv2.INTER_LANCZOS4 )
            else:
                mat = LandmarksProcessor.get_transform_mat (sample.landmarks, resolution, face_type)
                img  = cv2.warpAffine( img,  mat, (resolution,resolution), borderMode=cv2.BORDER_CONSTANT, flags=cv2.INTER_LANCZOS4 )
                mask = cv2.warpAffine( mask, mat, (resolution,resolution), borderMode=cv2.BORDER_CONSTANT, flags=cv2.INTER_LANCZOS4 )

            if len(mask.shape) == 2:
                mask = mask[...,None]
            return img, mask

        bs = self.batch_size
        while True:
            batches = [ [], [], [] ]  # images, masks, filenames

            n_batch = 0
            while n_batch < bs:
                try:
                    if len(shuffle_idxs) == 0:
                        shuffle_idxs = seg_sample_idxs.copy()
                        np.random.shuffle(shuffle_idxs)
                    sample = samples[shuffle_idxs.pop()]
                    img, mask = gen_img_mask(sample)
                    fname = getattr(sample, 'filename', '')

                    if np.random.randint(2) == 0:
                        if len(bg_shuffle_idxs) == 0:
                            bg_shuffle_idxs = seg_sample_idxs.copy()
                            np.random.shuffle(bg_shuffle_idxs)
                        bg_sample = samples[bg_shuffle_idxs.pop()]

                        bg_img, bg_mask = gen_img_mask(bg_sample)

                        # ensure bg matches training resolution
                        if bg_img.shape[0] != resolution or bg_img.shape[1] != resolution:
                            bg_img = cv2.resize(bg_img, (resolution, resolution), interpolation=cv2.INTER_LANCZOS4)
                            bg_mask = cv2.resize(bg_mask, (resolution, resolution), interpolation=cv2.INTER_LANCZOS4)

                        bg_wp   = imagelib.gen_warp_params(resolution, True, rotation_range=[-180,180], scale_range=[-0.10, 0.10], tx_range=[-0.10, 0.10], ty_range=[-0.10, 0.10] )
                        bg_img  = imagelib.warp_by_params (bg_wp, bg_img,  can_warp=False, can_transform=True, can_flip=True, border_replicate=True)
                        bg_mask = imagelib.warp_by_params (bg_wp, bg_mask, can_warp=False, can_transform=True, can_flip=True, border_replicate=False)
                        bg_img = bg_img*(1-bg_mask)
                        if np.random.randint(2) == 0:
                            bg_img = imagelib.apply_random_hsv_shift(bg_img)
                        else:
                            bg_img = imagelib.apply_random_rgb_levels(bg_img)

                        c_mask = 1.0 - (1-bg_mask) * (1-mask)
                        rnd = 0.25 + np.random.uniform()*0.85
                        img = img*(c_mask) + img*(1-c_mask)*rnd + bg_img*(1-c_mask)*(1-rnd)

                    warp_params = imagelib.gen_warp_params(resolution, random_flip, rotation_range=rotation_range, scale_range=scale_range, tx_range=tx_range, ty_range=ty_range )
                    img   = imagelib.warp_by_params (warp_params, img,  can_warp=True, can_transform=True, can_flip=True, border_replicate=True)
                    mask  = imagelib.warp_by_params (warp_params, mask, can_warp=True, can_transform=True, can_flip=True, border_replicate=False)

                    img = np.clip(img.astype(np.float32), 0, 1)
                    mask[mask < 0.5] = 0.0
                    mask[mask >= 0.5] = 1.0
                    mask = np.clip(mask, 0, 1)
                    
                    if np.random.randint(2) == 0:
                        # random face flare
                        krn = np.random.randint( resolution//4, resolution )
                        krn = krn - krn % 2 + 1
                        img = img + cv2.GaussianBlur(img*mask, (krn,krn), 0)

                    if np.random.randint(2) == 0:
                        # random bg flare
                        krn = np.random.randint( resolution//4, resolution )
                        krn = krn - krn % 2 + 1
                        img = img + cv2.GaussianBlur(img*(1-mask), (krn,krn), 0)

                    if np.random.randint(2) == 0:
                        img = imagelib.apply_random_hsv_shift(img, mask=sd.random_circle_faded ([resolution,resolution]))
                    else:
                        img = imagelib.apply_random_rgb_levels(img, mask=sd.random_circle_faded ([resolution,resolution]))
                        
                    if np.random.randint(2) == 0:
                        img = imagelib.apply_random_sharpen( img, sharpen_chance, sharpen_kernel_max_size, mask=sd.random_circle_faded ([resolution,resolution]))
                    else:
                        img = imagelib.apply_random_motion_blur( img, motion_blur_chance, motion_blur_mb_max_size, mask=sd.random_circle_faded ([resolution,resolution]))
                        img = imagelib.apply_random_gaussian_blur( img, gaussian_blur_chance, gaussian_blur_kernel_max_size, mask=sd.random_circle_faded ([resolution,resolution]))
                        
                    if np.random.randint(2) == 0:
                        img = imagelib.apply_random_nearest_resize( img, random_bilinear_resize_chance, random_bilinear_resize_max_size_per, mask=sd.random_circle_faded ([resolution,resolution]))
                    else:
                        img = imagelib.apply_random_bilinear_resize( img, random_bilinear_resize_chance, random_bilinear_resize_max_size_per, mask=sd.random_circle_faded ([resolution,resolution]))
                    img = np.clip(img, 0, 1)

                    img = imagelib.apply_random_jpeg_compress( img, random_jpeg_compress_chance, mask=sd.random_circle_faded ([resolution,resolution]))

                    if data_format == "NCHW":
                        img = np.transpose(img, (2,0,1) )
                        mask = np.transpose(mask, (2,0,1) )

                    batches[0].append ( img )
                    batches[1].append ( mask )
                    batches[2].append ( fname )

                    n_batch += 1
                except:
                    pass

            yield [ np.array(batch) for batch in batches]

class SegmentedSampleFilterSubprocessor(Subprocessor):
    #override
    def __init__(self, samples, count_xseg_mask=False ):
        self.samples = samples
        self.samples_len = len(self.samples)
        self.count_xseg_mask = count_xseg_mask

        self.idxs = [*range(self.samples_len)]
        self.result = []
        super().__init__('SegmentedSampleFilterSubprocessor', SegmentedSampleFilterSubprocessor.Cli, 60)

    #override
    def process_info_generator(self):
        for i in range(multiprocessing.cpu_count()):
            yield 'CPU%d' % (i), {}, {'samples':self.samples, 'count_xseg_mask':self.count_xseg_mask}

    #override
    def on_clients_initialized(self):
        io.progress_bar ("Filtering", self.samples_len)

    #override
    def on_clients_finalized(self):
        io.progress_bar_close()

    #override
    def get_data(self, host_dict):
        if len (self.idxs) > 0:
            return self.idxs.pop(0)

        return None

    #override
    def on_data_return (self, host_dict, data):
        self.idxs.insert(0, data)

    #override
    def on_result (self, host_dict, data, result):
        idx, is_ok = result
        if is_ok:
            self.result.append(idx)
        io.progress_bar_inc(1)
    def get_result(self):
        return self.result

    class Cli(Subprocessor.Cli):
        #overridable optional
        def on_initialize(self, client_dict):
            self.samples = client_dict['samples']
            self.count_xseg_mask = client_dict['count_xseg_mask']

        def process_data(self, idx):
            if self.count_xseg_mask:
                if not self.samples[idx].has_xseg_mask():
                    return idx, False
                # 验证遮罩不是空的（至少 15% 像素被覆盖）
                try:
                    mask = self.samples[idx].get_xseg_mask()
                    if mask is None or mask.max() < 0.5:
                        return idx, False
                    if (mask > 0.5).sum() < mask.size * 0.25:
                        return idx, False
                except Exception:
                    return idx, False
                return idx, True
            elif self.samples[idx].seg_ie_polys.get_pts_count() != 0:
                # 手画遮罩：检查多边形面积是否足够
                try:
                    h, w = self.samples[idx].load_bgr().shape[:2]
                    mask = np.zeros((h, w, 1), dtype=np.float32)
                    self.samples[idx].seg_ie_polys.overlay_mask(mask)
                    if (mask > 0.5).sum() < mask.size * 0.25:
                        return idx, False
                except Exception:
                    return idx, False
                return idx, True
            else:
                return idx, self.samples[idx].seg_ie_polys.get_pts_count() != 0

"""
  bg_path = None
        for path in paths:
            bg_path = Path(path) / 'backgrounds'
            if bg_path.exists():

                break
        if bg_path is None:
            io.log_info(f'Random backgrounds will not be used. Place no face jpg images to aligned\backgrounds folder. ')
            bg_pathes = None
        else:
            bg_pathes = pathex.get_image_paths(bg_path, image_extensions=['.jpg'], return_Path_class=True)
            io.log_info(f'Using {len(bg_pathes)} random backgrounds from {bg_path}')

if bg_pathes is not None:
            bg_path = bg_pathes[ np.random.randint(len(bg_pathes)) ]

            bg_img = cv2_imread(bg_path)
            if bg_img is not None:
                bg_img = bg_img.astype(np.float32) / 255.0
                bg_img = imagelib.normalize_channels(bg_img, 3)

                bg_img = imagelib.random_crop(bg_img, resolution, resolution)
                bg_img = cv2.resize(bg_img, (resolution, resolution), interpolation=cv2.INTER_LINEAR)

            if np.random.randint(2) == 0:
                bg_img = imagelib.apply_random_hsv_shift(bg_img)
            else:
                bg_img = imagelib.apply_random_rgb_levels(bg_img)

            bg_wp   = imagelib.gen_warp_params(resolution, True, rotation_range=[-180,180], scale_range=[0,0], tx_range=[0,0], ty_range=[0,0])
            bg_img  = imagelib.warp_by_params (bg_wp, bg_img,  can_warp=False, can_transform=True, can_flip=True, border_replicate=True)

            bg = img*(1-mask)
            fg = img*mask

            c_mask = sd.random_circle_faded ([resolution,resolution])
            bg = ( bg_img*c_mask + bg*(1-c_mask) )*(1-mask)

            img = fg+bg

        else:
"""