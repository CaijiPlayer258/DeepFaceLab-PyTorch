"""Common mask manipulation utilities used by all generators."""

from __future__ import annotations

from typing import List, Optional

import cv2
import numpy as np


def merge_masks(masks: list[np.ndarray], mode: str = "union") -> np.ndarray:
    """Merge multiple masks into a single mask.

    All input masks must have the same shape and be float32 in range [0, 1].

    Args:
        masks: List of (H, W) float32 mask arrays.
        mode:
            - ``"union"``: logical OR — the result is 1 wherever any mask is 1.
            - ``"diff"``: first mask minus the union of the rest — the result
              is 1 only in regions of the first mask not covered by any other
              mask.

    Returns:
        A single (H, W) float32 mask in [0, 1].

    Raises:
        ValueError: If *masks* is empty or *mode* is unknown.
    """
    if not masks:
        raise ValueError("masks list is empty")

    if mode == "union":
        return np.maximum.reduce(masks).astype(np.float32)
    elif mode == "diff":
        first = masks[0].astype(np.float32)
        if len(masks) == 1:
            return first
        rest = np.maximum.reduce(masks[1:]).astype(np.float32)
        result = np.clip(first - rest, 0.0, 1.0)
        return result
    else:
        raise ValueError(f"Unknown merge mode: {mode!r}. Supported: 'union', 'diff'.")


def apply_morphology(
    mask: np.ndarray,
    kernel_size: int = 3,
    op: str = "close",
) -> np.ndarray:
    """Apply a morphological operation to a mask.

    Uses an elliptical (ellipse-shaped) structuring element.

    Args:
        mask: Input (H, W) float32 mask in [0, 1].
        kernel_size: Diameter of the elliptical kernel (must be positive odd
            integer). Will be rounded up to the nearest odd integer if even.
        op:
            - ``"close"``: dilation followed by erosion — fills small holes.
            - ``"open"``: erosion followed by dilation — removes small
              speckles.
            - ``"dilate"``: expands mask boundaries.
            - ``"erode"``: shrinks mask boundaries.

    Returns:
        Cleaned (H, W) float32 mask in [0, 1].

    Raises:
        ValueError: If *op* is unknown.
    """
    # Ensure odd kernel size
    if kernel_size < 1:
        kernel_size = 1
    if kernel_size % 2 == 0:
        kernel_size += 1

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))

    # Convert to uint8 for OpenCV morphology (0 or 255)
    binary = (mask > 0.5).astype(np.uint8) * 255

    op_map = {
        "close": cv2.MORPH_CLOSE,
        "open": cv2.MORPH_OPEN,
        "dilate": cv2.MORPH_DILATE,
        "erode": cv2.MORPH_ERODE,
    }

    if op not in op_map:
        raise ValueError(
            f"Unknown morphology operation: {op!r}. "
            f"Supported: {', '.join(sorted(op_map.keys()))}."
        )

    result = cv2.morphologyEx(binary, op_map[op], kernel)

    # Convert back to float32 [0, 1]
    return (result > 0).astype(np.float32)


def polygon_to_mask(polygons: list, shape: tuple) -> np.ndarray:
    """Convert polygon vertex arrays to a binary mask.

    Args:
        polygons: List of polygon definitions. Each element can be:
            - A NumPy array of shape (N, 2) with dtype float32/int32,
              representing N vertices.
            - Any iterable of (x, y) coordinate pairs that can be converted
              to an ``np.int32`` array of shape (N, 1, 2).
        shape: Target mask dimensions ``(height, width)``.

    Returns:
        (H, W) float32 mask in [0, 1] where pixels inside any polygon are 1.
    """
    mask = np.zeros(shape, dtype=np.uint8)

    for poly in polygons:
        # Normalise to int32 array of shape (N, 1, 2)
        pts = np.array(poly, dtype=np.int32)
        if pts.ndim == 2 and pts.shape[1] == 2:
            pts = pts.reshape((-1, 1, 2))
        cv2.fillPoly(mask, [pts], 255)

    return (mask > 0).astype(np.float32)


def mask_to_polygons(
    mask: np.ndarray,
    simplify_epsilon: float = 1.0,
) -> list[np.ndarray]:
    """Extract simplified polygon contours from a binary mask.

    Args:
        mask: Input (H, W) float32 mask in [0, 1].
        simplify_epsilon: Maximum distance from the original contour to the
            approximated polygonal curve (``cv2.approxPolyDP`` epsilon).
            Higher values yield coarser (simpler) polygons.

    Returns:
        List of (N, 2) float32 vertex arrays. Contours with area < 10 pixels
        are filtered out.
    """
    binary = (mask > 0.5).astype(np.uint8) * 255

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    polygons: list[np.ndarray] = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 10.0:
            continue

        epsilon = simplify_epsilon
        approx = cv2.approxPolyDP(cnt, epsilon, closed=True)

        # Reshape from (N, 1, 2) to (N, 2)
        pts = approx.reshape(-1, 2).astype(np.float32)
        polygons.append(pts)

    return polygons
