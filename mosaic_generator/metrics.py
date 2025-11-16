"""
metrics.py

Compute MSE, PSNR, SSIM, and color similarity between images.
Robust and compatible with all Lab-5 modules.
"""

import numpy as np
import cv2
from skimage.metrics import structural_similarity as ssim

from .utils import ensure_uint8


def mse(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype("float64")
    b = b.astype("float64")
    return float(np.mean((a - b) ** 2))


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    error = mse(a, b)
    return float("inf") if error == 0 else 20 * np.log10(255.0 / np.sqrt(error))


def ssim_score(a: np.ndarray, b: np.ndarray) -> float:
    """
    SSIM may fail for small images or quantized outputs.
    Fallback to 0.0 instead of crashing the app.
    """
    try:
        return float(ssim(a, b, channel_axis=2, data_range=255))
    except Exception:
        return 0.0


def color_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a_flat = a.flatten().astype("float64")
    b_flat = b.flatten().astype("float64")
    corr = np.corrcoef(a_flat, b_flat)[0, 1]
    return 0.0 if np.isnan(corr) else float(corr)


def compute_metrics(original: np.ndarray, mosaic: np.ndarray) -> dict:
    """
    Compute all metrics and return dictionary.
    Ensures dtype correctness and shape alignment.
    """

    # Ensure uint8 inputs
    original = ensure_uint8(original)
    mosaic = ensure_uint8(mosaic)

    # Resize mosaic if needed
    if original.shape != mosaic.shape:
        mosaic = cv2.resize(mosaic, (original.shape[1], original.shape[0]))

    return {
        "mse": mse(original, mosaic),
        "psnr": psnr(original, mosaic),
        "ssim": ssim_score(original, mosaic),
        "color_similarity": color_similarity(original, mosaic)
    }