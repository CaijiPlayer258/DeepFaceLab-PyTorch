"""Workspace manager for aligned faceset scanning and XSeg mask I/O."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import struct

from DFLIMG.DFLJPG import DFLJPG


class FileEntry:
    """Represents a single aligned face image and its XSeg mask status."""

    def __init__(self, path: Path, has_mask: bool = False):
        self.path = path
        self.name = path.name
        self.has_mask = has_mask  # whether this file has an XSeg mask
        self._dfljpg: Optional["DFLJPG"] = None  # lazy loaded

    @property
    def dfljpg(self) -> Optional["DFLJPG"]:
        """Lazy-load DFLJPG on first access and update mask flag."""
        if self._dfljpg is None:
            try:
                self._dfljpg = DFLJPG.load(str(self.path))
                if self._dfljpg is not None and self._dfljpg.has_xseg_mask():
                    self.has_mask = True
            except Exception:
                self._dfljpg = None
        return self._dfljpg


class Workspace:
    """Manages an aligned faceset directory with lazy DFLJPG access."""

    def __init__(self, path: str):
        self.root = Path(path)
        if not self.root.exists():
            raise FileNotFoundError(f"Workspace not found: {path}")
        self._files: list[FileEntry] = []
        self._current_index: Optional[int] = None
        self._scan()

    def _scan(self) -> None:
        """Scan for .jpg files. Mask detection is lazy (deferred to FileEntry.dfljpg)."""
        import glob as glob_mod
        pattern = str(self.root / "*.jpg")
        paths = sorted(glob_mod.glob(pattern))
        self._files = [FileEntry(Path(p)) for p in paths]

    @property
    def files(self) -> list[FileEntry]:
        """Return the list of scanned FileEntry objects."""
        return self._files

    def __len__(self) -> int:
        return len(self._files)

    def __getitem__(self, index: int) -> FileEntry:
        return self._files[index]

    def load_image(self, index: int) -> np.ndarray:
        """Load the image at *index* and return a BGR numpy array."""
        path = str(self._files[index].path)
        img = cv2.imread(path)
        if img is None:
            raise RuntimeError(f"Failed to load image: {path}")
        return img

    def save_mask(self, index: int, mask: np.ndarray) -> None:
        """Save a float32 [0,1] mask into the DFLJPG at *index*.
        Saves both raster (xseg_mask) and polygon (seg_ie_polys) formats."""
        entry = self._files[index]
        dfljpg = entry.dfljpg
        if dfljpg is None:
            raise RuntimeError(f"Cannot load DFLJPG for {entry.name}")
        dfljpg.set_xseg_mask(mask)
        # Also save polygon format
        try:
            from MaskProcessor.core.mask_ops import mask_to_polygons
            from core.imagelib.SegIEPolys import SegIEPolys, SegIEPoly, SegIEPolyType

            # Scale polygon coordinates from mask resolution to image resolution
            # (old DFL renders polys at image size then resizes to training res)
            shape = dfljpg.get_shape()
            img_h = int(shape[0]) if shape is not None else 256
            img_w = int(shape[1]) if shape is not None else 256
            mask_h, mask_w = mask.shape[:2]
            sx = img_w / float(mask_w)
            sy = img_h / float(mask_h)

            polys = SegIEPolys()
            poly_list = mask_to_polygons(mask)
            for pts in poly_list:
                poly = polys.add_poly(SegIEPolyType.INCLUDE)
                for pt in pts:
                    poly.add_pt(int(pt[0] * sx), int(pt[1] * sy))
            dfljpg.set_seg_ie_polys(polys)
        except Exception:
            pass  # polygon extraction is optional
        try:
            dfljpg.save()
        except struct.error:
            # Data too large for APP15 segment - retry without seg_ie_polys
            dfljpg.dfl_dict.pop('seg_ie_polys', None)
            dfljpg.save()
        entry.has_mask = True

    def get_mask(self, index: int) -> Optional[np.ndarray]:
        """Return the existing XSeg mask (float32, [0,1]) or None."""
        entry = self._files[index]
        dfljpg = entry.dfljpg
        if dfljpg is None:
            return None
        return dfljpg.get_xseg_mask()
