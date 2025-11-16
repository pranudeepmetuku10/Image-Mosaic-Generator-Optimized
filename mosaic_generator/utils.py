# mosaic_generator/utils.py

import numpy as np
import cv2
from PIL import Image


# ---------------------------------------------------------------
# Basic Array Utilities
# ---------------------------------------------------------------

def ensure_uint8(arr: np.ndarray) -> np.ndarray:
    """Clip to [0,255] and convert to uint8."""
    return np.clip(arr, 0, 255).astype(np.uint8)


def pil_to_numpy(img: Image.Image) -> np.ndarray:
    """Convert PIL → NumPy RGB."""
    return np.array(img.convert("RGB"))


def numpy_to_pil(arr: np.ndarray) -> Image.Image:
    """Convert RGB NumPy → PIL."""
    return Image.fromarray(ensure_uint8(arr))


# ---------------------------------------------------------------
# Validation
# ---------------------------------------------------------------

def validate_grid_size(image_shape, grid_size):
    h, w = image_shape[:2]
    rows, cols = grid_size
    if rows <= 0 or cols <= 0:
        raise ValueError("Grid size must be positive.")
    if h < rows or w < cols:
        raise ValueError(
            f"Grid {rows}×{cols} too fine for image {h}×{w}. "
            "Increase resolution or use fewer tiles."
        )


def validate_tile_array(tiles: np.ndarray):
    if tiles.ndim != 4 or tiles.shape[-1] != 3:
        raise ValueError("Tiles must have shape (N, H, W, 3).")


# ---------------------------------------------------------------
# Feature Extraction
# ---------------------------------------------------------------

def compute_average_color(arr: np.ndarray) -> np.ndarray:
    """Mean RGB vector."""
    return arr.reshape(-1, 3).mean(axis=0).astype(np.float32)


def compute_histogram(arr: np.ndarray, bins: int = 8) -> np.ndarray:
    """Normalized 3D RGB histogram."""
    hist = cv2.calcHist(
        [arr],
        [0, 1, 2],
        None,
        [bins, bins, bins],
        [0, 256, 0, 256, 0, 256]
    )
    cv2.normalize(hist, hist)
    return hist.flatten().astype(np.float32)


def fast_dominant_color(cell: np.ndarray, bins: int = 16) -> np.ndarray:
    """
    Fast approximate dominant color using HSV histogram (H channel only).
    No KMeans → HuggingFace-safe.
    """
    hsv = cv2.cvtColor(cell, cv2.COLOR_RGB2HSV)
    h = hsv[:, :, 0]

    hist = cv2.calcHist([h], [0], None, [bins], [0, 180])
    dom_bin = np.argmax(hist)

    # convert peak hue bin → actual hue value
    hue = int((dom_bin + 0.5) * (180 / bins))

    hsv_color = np.uint8([[[hue, 180, 200]]])
    rgb = cv2.cvtColor(hsv_color, cv2.COLOR_HSV2RGB)[0, 0]

    return rgb.astype(np.float32)