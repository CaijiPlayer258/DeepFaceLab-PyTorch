"""
Landmark conversion utilities.
Reference: Extractor/Extractor.py landmark106to68
"""
import numpy as np


def landmark106to68(pt106: np.ndarray) -> np.ndarray:
    """
    Convert 106-point landmarks (InsightFace) to standard 68-point landmarks.
    """
    if len(pt106) != 106:
        return pt106[:68] if len(pt106) >= 68 else pt106

    indices = [
        1, 10, 12, 14, 16, 3, 5, 7, 0,
        23, 21, 19, 32, 30, 28, 26, 17,
        43, 48, 49, 51, 50,
        102, 103, 104, 105, 101,
        72, 73, 74, 86, 78, 79, 80, 85, 84,
        35, 41, 42, 39, 37, 36,
        89, 95, 96, 93, 91, 90,
        52, 64, 63, 71, 67, 68, 61, 58, 59, 53, 56, 55, 65, 66, 62, 70, 69, 57, 60, 54
    ]

    return np.array([pt106[i] for i in indices])
