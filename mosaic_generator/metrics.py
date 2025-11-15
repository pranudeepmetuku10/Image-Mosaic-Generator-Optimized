"""
metrics.py

Compute MSE, PSNR, SSIM, and color similarity between images.
"""

import numpy as np
import cv2
from skimage.metrics import structural_similarity as ssim


def mse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean((a.astype("float64") - b.astype("float64")) ** 2))


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    error = mse(a, b)
    return float("inf") if error == 0 else 20 * np.log10(255.0 / np.sqrt(error))


def ssim_score(a: np.ndarray, b: np.ndarray) -> float:
    return ssim(a, b, channel_axis=2, data_range=255)


def color_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a_flat = a.flatten().astype("float64")
    b_flat = b.flatten().astype("float64")
    corr = np.corrcoef(a_flat, b_flat)[0, 1]
    return 0.0 if np.isnan(corr) else float(corr)


def compute_metrics(original: np.ndarray, mosaic: np.ndarray) -> dict:
    """
    Compute all metrics and return dictionary.
    """
    if original.shape != mosaic.shape:
        mosaic = cv2.resize(mosaic, (original.shape[1], original.shape[0]))

    return {
        "mse": mse(original, mosaic),
        "psnr": psnr(original, mosaic),
        "ssim": ssim_score(original, mosaic),
        "color_similarity": color_similarity(original, mosaic)
    }