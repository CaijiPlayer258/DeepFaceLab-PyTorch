"""Dynamic sample generator for XSegLite training.

Features:
- Zero startup: only scans filenames, no metadata loading
- Watchdog-based file system monitoring: detects new/deleted images in real-time
- On-demand loading: image + mask loaded only when picked for a batch
- Thread-safe FileIndex: supports concurrent add/remove while training
"""

import glob
import os
import random
import threading
import time
from pathlib import Path

import cv2
import numpy as np

from core import imagelib
from core.imagelib import sd
from core.interact import interact as io
from samplelib import SampleGeneratorBase

try:
    from watchdog.observers import Observer
    from watchdog.events import FileSystemEventHandler
except ImportError:
    Observer = None
    FileSystemEventHandler = object


_EXTENSIONS = ('*.jpg', '*.jpeg', '*.png', '*.bmp')


class FileIndex:
    """Thread-safe set of file paths. Supports concurrent add/remove while training iterates."""

    def __init__(self):
        self._lock = threading.Lock()
        self._files: list[Path] = []
        self._file_set: set[Path] = set()

    def scan(self, paths):
        """Initial scan: collect all image files from the given directories."""
        collected = set()
        for p in paths:
            p = Path(p)
            if not p.exists():
                continue
            for ext in _EXTENSIONS:
                for f in sorted(p.glob(ext)):
                    collected.add(f.resolve())
                for f in sorted(p.glob(ext.upper())):
                    collected.add(f.resolve())
        with self._lock:
            self._files = sorted(collected)
            self._file_set = collected
        if collected:
            io.log_info(f'[DynamicSampleGenerator] Scanned {len(collected)} images')

    def add(self, path: Path):
        """Add a single file (called by watchdog on_created)."""
        path = path.resolve()
        if not path.suffix.lower() in ('.jpg', '.jpeg', '.png', '.bmp'):
            return
        with self._lock:
            if path not in self._file_set:
                self._file_set.add(path)
                self._files.append(path)

    def remove(self, path: Path):
        """Remove a single file (called by watchdog on_deleted)."""
        path = path.resolve()
        with self._lock:
            self._file_set.discard(path)
            # _files list is rebuilt lazily; snapshot() builds from _file_set

    def random_sample(self):
        """Return a random file path. Thread-safe."""
        with self._lock:
            if not self._files:
                return None
            # Rebuild list if out of sync with set
            if len(self._files) != len(self._file_set):
                self._files = sorted(self._file_set)
            if not self._files:
                return None
            return random.choice(self._files)

    def snapshot(self):
        """Return a copy of current file paths (for background composite selection)."""
        with self._lock:
            return list(self._file_set)

    def __len__(self):
        with self._lock:
            return len(self._file_set)


class DirectoryWatcher(FileSystemEventHandler):
    """Watchdog event handler: forwards file create/delete events to FileIndex."""

    def __init__(self, file_index: FileIndex):
        super().__init__()
        self._file_index = file_index
        self._observer = None

    def start(self, paths):
        if Observer is None:
            io.log_info('[DynamicSampleGenerator] watchdog not installed, file watching disabled')
            return
        self._observer = Observer()
        for p in paths:
            p = Path(p)
            if p.exists():
                self._observer.schedule(self, str(p), recursive=False)
        self._observer.daemon = True
        self._observer.start()
        io.log_info('[DynamicSampleGenerator] File watcher started')

    def stop(self):
        if self._observer is not None:
            self._observer.stop()
            self._observer = None

    def on_created(self, event):
        if not event.is_directory:
            self._file_index.add(Path(event.src_path))

    def on_deleted(self, event):
        if not event.is_directory:
            self._file_index.remove(Path(event.src_path))

    def on_moved(self, event):
        if not event.is_directory:
            self._file_index.remove(Path(event.src_path))
            self._file_index.add(Path(event.dest_path))


def _load_image_mask(fpath: str, resolution: int):
    """Load image + XSeg mask from a file.

    Returns (img_hwc, mask_hw) as float32 [0,1] or None if mask is invalid.
    No face_type/landmarks alignment — assumes all images are at target face_type.
    """
    img = cv2.imread(fpath)
    if img is None:
        return None
    img = img.astype(np.float32) / 255.0
    h, w = img.shape[:2]

    try:
        from DFLIMG.DFLJPG import DFLJPG
        inst = DFLJPG.load(fpath)
        if inst is None or not inst.has_xseg_mask():
            return None
        mask = inst.get_xseg_mask().squeeze()  # (H,W) or (H,W,1) -> (H,W)
    except Exception:
        return None

    # Validate mask
    if mask is None or mask.max() < 0.5:
        return None
    if (mask > 0.5).sum() < mask.size * 0.001:  # 0.1% — 眼瞳遮罩覆盖率很小
        return None

    # Resize to target resolution
    if w != resolution or h != resolution:
        img = cv2.resize(img, (resolution, resolution), interpolation=cv2.INTER_LANCZOS4)
    if mask.shape[0] != resolution or mask.shape[1] != resolution:
        mask = cv2.resize(mask, (resolution, resolution), interpolation=cv2.INTER_LANCZOS4)

    mask[mask < 0.5] = 0.0
    mask[mask >= 0.5] = 1.0
    mask = np.clip(mask, 0, 1)

    if mask.ndim == 2:
        mask = mask[..., None]

    return img, mask


def _augment_image(img, mask, resolution, file_index: FileIndex, data_format: str):
    """Apply augmentation pipeline to a single image-mask pair.

    Matches the augmentation in SampleGeneratorFaceXSeg._skip_generate_next().
    Returns (img_nchw, mask_nchw) or (img_nhwc, mask_nhwc) depending on data_format.
    """
    # Random background from another file
    files = file_index.snapshot()
    if np.random.randint(2) == 0 and len(files) > 1:
        bg_fpath = random.choice(files)
        bg_loaded = _load_image_mask(str(bg_fpath), resolution)
        if bg_loaded is not None:
            bg_img, bg_mask = bg_loaded
            bg_wp = imagelib.gen_warp_params(resolution, True,
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
    wp = imagelib.gen_warp_params(resolution, True,
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

    # Face flare / BG flare
    if np.random.randint(2) == 0:
        krn = np.random.randint(resolution // 4, resolution)
        krn = krn - krn % 2 + 1
        img = img + cv2.GaussianBlur(img * mask, (krn, krn), 0)
    if np.random.randint(2) == 0:
        krn = np.random.randint(resolution // 4, resolution)
        krn = krn - krn % 2 + 1
        img = img + cv2.GaussianBlur(img * (1 - mask), (krn, krn), 0)

    # HSV / RGB
    if np.random.randint(2) == 0:
        img = imagelib.apply_random_hsv_shift(img, mask=sd.random_circle_faded([resolution, resolution]))
    else:
        img = imagelib.apply_random_rgb_levels(img, mask=sd.random_circle_faded([resolution, resolution]))

    # Sharpen or blur
    if np.random.randint(2) == 0:
        img = imagelib.apply_random_sharpen(img, 25, 5, mask=sd.random_circle_faded([resolution, resolution]))
    else:
        img = imagelib.apply_random_motion_blur(img, 25, 5, mask=sd.random_circle_faded([resolution, resolution]))
        img = imagelib.apply_random_gaussian_blur(img, 25, 5, mask=sd.random_circle_faded([resolution, resolution]))

    # Resize jitter
    if np.random.randint(2) == 0:
        img = imagelib.apply_random_nearest_resize(img, 25, 75, mask=sd.random_circle_faded([resolution, resolution]))
    else:
        img = imagelib.apply_random_bilinear_resize(img, 25, 75, mask=sd.random_circle_faded([resolution, resolution]))
    img = np.clip(img, 0, 1)

    # JPEG compression
    img = imagelib.apply_random_jpeg_compress(img, 25, mask=sd.random_circle_faded([resolution, resolution]))

    # Format
    if data_format == 'NCHW':
        img = np.transpose(img, (2, 0, 1))
        mask = np.transpose(mask, (2, 0, 1))

    return img, mask


class DynamicSampleGenerator(SampleGeneratorBase):
    """XSegLite dynamic sample generator.

    - Scans directory at init (filenames only)
    - Watchdog monitors for new/deleted files in real-time
    - Loads image+mask on-demand during batch generation
    - Prefetches next batch in background thread
    """

    def __init__(self, paths, batch_size=1, resolution=256,
                 data_format='NCHW', generators_count=4):
        super().__init__(debug=False, batch_size=batch_size)
        self._resolution = resolution
        self._data_format = data_format
        self._last_fnames = []

        # Thread-safe file index
        self._file_index = FileIndex()
        self._file_index.scan(paths)

        if len(self._file_index) == 0:
            io.log_info('[DynamicSampleGenerator] No images found.')

        # Start file system watcher
        self._watcher = DirectoryWatcher(self._file_index)
        self._watcher.start(paths)

        # Prefetch
        self._prefetched = None
        self._prefetch_thread: threading.Thread | None = None
        self._prefetched = self._generate_batch()

        self.initialized = True

    def is_initialized(self):
        return self.initialized

    def __next__(self):
        # Wait for prefetch to finish
        if self._prefetch_thread is not None:
            self._prefetch_thread.join()
        result = self._prefetched
        # Start next prefetch
        self._prefetch_thread = threading.Thread(target=self._prefetch, daemon=True)
        self._prefetch_thread.start()
        # Unpack: result is (imgs, masks, fnames), return imgs+masks but store fnames
        self._last_fnames = result[2] if result is not None and len(result) >= 3 else []
        return result[:2] if result is not None else None

    def _prefetch(self):
        try:
            self._prefetched = self._generate_batch()
        except Exception as e:
            io.log_err(f'[DynamicSampleGenerator] Prefetch failed: {e}')
            self._prefetched = None

    def _generate_batch(self):
        """Generate a single batch of (images, masks, fnames)."""
        bs = self.batch_size
        res = self._resolution
        imgs = np.zeros((bs, 3, res, res), dtype=np.float32)
        masks = np.zeros((bs, 1, res, res), dtype=np.float32)
        fnames = []

        for i in range(bs):
            retries = 0
            loaded = None
            while retries < 100 and loaded is None:
                fpath = self._file_index.random_sample()
                if fpath is None:
                    break
                loaded = _load_image_mask(str(fpath), res)
                retries += 1

            if loaded is None:
                imgs[i] = 0.0
                masks[i] = 0.0
                fnames.append('')
                continue

            img, mask = loaded
            fnames.append(fpath.name)

            # Apply augmentation
            aug_img, aug_mask = _augment_image(img, mask, res, self._file_index, self._data_format)
            imgs[i] = aug_img
            masks[i] = aug_mask

        return [imgs, masks, fnames]

    def get_last_filenames(self):
        return self._last_fnames

    def close(self):
        self._watcher.stop()
        super().close()
