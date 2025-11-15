# mosaic_generator/metrics.py

"""
Image quality metrics used to evaluate mosaic output. Includes MSE, PSNR,
SSIM, and simple color correlation.
"""

import numpy as np
import cv2
from skimage.metrics import structural_similarity as ssim


def mse(a: np.ndarray, b: np.ndarray) -> float:
    """Mean Squared Error."""
    return float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    """Peak Signal-to-Noise Ratio."""
    m = mse(a, b)
    if m == 0:
        return float("inf")
    return 20 * np.log10(255.0 / np.sqrt(m))


def ssim_score(a: np.ndarray, b: np.ndarray) -> float:
    """Structural Similarity Index."""
    try:
        return ssim(a, b, channel_axis=2, data_range=255)
    except Exception:
        return 0.0


def color_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Correlation between flattened RGB channels."""
    x = a.flatten().astype(np.float64)
    y = b.flatten().astype(np.float64)
    corr = np.corrcoef(x, y)[0, 1]
    return float(corr) if not np.isnan(corr) else 0.0


def compute_all(a: np.ndarray, b: np.ndarray) -> dict:
    """Compute MSE, PSNR, SSIM, and color correlation."""
    return {
        "mse": mse(a, b),
        "psnr": psnr(a, b),
        "ssim": ssim_score(a, b),
        "color_similarity": color_similarity(a, b),
    }