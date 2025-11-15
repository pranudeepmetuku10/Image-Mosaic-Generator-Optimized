# mosaic_generator/utils.py

"""
Utility helpers for image validation, type normalization, and safe
conversions between PIL and NumPy. These functions are shared across
the mosaic generator package.
"""

import numpy as np
import cv2
from PIL import Image


def ensure_uint8(arr: np.ndarray) -> np.ndarray:
    """Convert array to uint8 format while clipping to [0,255]."""
    return np.clip(arr, 0, 255).astype(np.uint8)


def pil_to_numpy(im: Image.Image) -> np.ndarray:
    """Convert a PIL Image to a NumPy RGB array."""
    return np.array(im.convert("RGB"))


def numpy_to_pil(arr: np.ndarray) -> Image.Image:
    """Convert a NumPy RGB array to a PIL Image."""
    return Image.fromarray(ensure_uint8(arr))


def validate_grid_size(image_shape, grid_size):
    """
    Validate that the requested grid size is compatible with the
    preprocessed image dimensions.
    """
    h, w = image_shape[:2]
    rows, cols = grid_size

    if rows <= 0 or cols <= 0:
        raise ValueError("Grid dimensions must be positive integers.")
    if h < rows or w < cols:
        raise ValueError(
            f"Grid {rows}×{cols} is too fine for image size {h}×{w}. "
            "Increase preprocessing resolution or use a coarser grid."
        )


def validate_tile_array(tiles: np.ndarray):
    """Ensure tile array is of shape (N, H, W, 3) with three RGB channels."""
    if tiles.ndim != 4 or tiles.shape[-1] != 3:
        raise ValueError(
            "Tile array must have shape (N, H, W, 3) representing RGB tiles."
        )