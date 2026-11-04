"""Example: Using MaskProcessor core programmatically.

This script demonstrates how to use the core components of MaskProcessor
outside of the web UI -- useful for batch processing or integrating into
existing pipelines.
"""

import cv2
import numpy as np

from MaskProcessor.core.mask_ops import merge_masks, apply_morphology
from MaskProcessor.workspace import Workspace


def example_mask_ops():
    """Demonstrate basic mask operations."""
    # Create two simple masks
    m1 = np.ones((100, 100), dtype=np.float32)          # all foreground
    m2 = np.zeros((100, 100), dtype=np.float32)          # all background

    # Union: combine both masks
    merged = merge_masks([m1, m2], method="union")
    print(f"Union shape: {merged.shape}, dtype: {merged.dtype}")
    assert np.allclose(merged, m1), "Union of (ones, zeros) should equal ones"

    # Intersection: only where both agree
    merged = merge_masks([m1, m2], method="intersection")
    print(f"Intersection shape: {merged.shape}, dtype: {merged.dtype}")
    assert np.allclose(merged, m2), "Intersection of (ones, zeros) should equal zeros"

    # Difference: m1 minus m2
    merged = merge_masks([m1, m2], method="difference")
    print(f"Difference shape: {merged.shape}, dtype: {merged.dtype}")
    assert np.allclose(merged, m1), "Difference of (ones, zeros) should equal ones"

    # Morphological close (dilate then erode) to fill small holes
    noisy = np.zeros((100, 100), dtype=np.float32)
    cv2.circle(noisy, (50, 50), 20, 1.0, -1)            # solid circle
    cv2.circle(noisy, (50, 50), 8, 0.0, -1)              # hole in centre

    cleaned = apply_morphology(noisy, op="close", kernel_size=11)
    hole_value = cleaned[50, 50]
    print(f"Morphological close: hole filled = {hole_value:.2f} (expected ~1.0)")
    assert hole_value > 0.9, "Closing should fill the central hole"

    print("\nAll mask operations passed.")


def example_workspace():
    """Demonstrate workspace discovery."""
    ws = Workspace("C:/MyData/my_dfl_project/data_dst")
    print(f"Workspace root: {ws.root}")
    print(f"Aligned directory: {ws.aligned_dir}")

    faces = ws.list_faces()
    print(f"Found {len(faces)} aligned faces")
    if faces:
        first_face = faces[0]
        img = ws.load_face(first_face)
        if img is not None:
            print(f"First face image shape: {img.shape}")


if __name__ == "__main__":
    print("=== Mask Ops Example ===\n")
    example_mask_ops()

    print("\n=== Workspace Example ===")
    print("(Skipped -- requires a real DFL workspace directory)")
