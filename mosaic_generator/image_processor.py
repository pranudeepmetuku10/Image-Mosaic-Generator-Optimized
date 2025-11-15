# mosaic_generator/image_processor.py

"""
Image processing utilities for preprocessing input images and performing
vectorized grid slicing. This module ensures the input image is resized,
optionally quantized, and converted into a grid representation suitable
for fast mosaic construction.
"""

import numpy as np
import cv2
from PIL import Image

from .config import PREPROCESS_RESOLUTION
from .utils import ensure_uint8


class ImageProcessor:
    """
    Handles preprocessing of input images and vectorized grid division.

    Methods
    -------
    preprocess(image, apply_quantization=False, n_colors=16)
        Resize, pad, and optionally quantize the image.
    divide_into_grid(image, grid_size)
        Vectorized split of image into (rows, cols, tile_h, tile_w, 3)
        without Python loops.
    """

    @staticmethod
    def preprocess(image: np.ndarray,
                   apply_quantization: bool = False,
                   n_colors: int = 16) -> np.ndarray:
        """
        Resize and optionally quantize the image. Aspect ratio is preserved
        via letterboxing before cropping to the target resolution.

        Parameters
        ----------
        image : np.ndarray
            Input RGB image.
        apply_quantization : bool
            Whether to apply K-means color quantization.
        n_colors : int
            Number of colors to quantize to.

        Returns
        -------
        np.ndarray
            Preprocessed RGB image of size PREPROCESS_RESOLUTION.
        """

        target_h, target_w = PREPROCESS_RESOLUTION
        h, w = image.shape[:2]

        scale = min(target_h / h, target_w / w)
        new_h, new_w = int(h * scale), int(w * scale)

        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)

        top = (target_h - new_h) // 2
        left = (target_w - new_w) // 2

        canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
        canvas[top:top + new_h, left:left + new_w] = resized
        output = canvas

        if apply_quantization:
            output = ImageProcessor._quantize(output, n_colors)

        return ensure_uint8(output)

    @staticmethod
    def _quantize(image: np.ndarray, n_colors: int) -> np.ndarray:
        """K-means quantization to reduce color complexity."""
        data = image.reshape((-1, 3)).astype(np.float32)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)

        _, labels, centers = cv2.kmeans(
            data, n_colors, None, criteria, 5, cv2.KMEANS_RANDOM_CENTERS
        )

        centers = centers.astype(np.uint8)
        quant = centers[labels.flatten()].reshape(image.shape)
        return quant

    @staticmethod
    def divide_into_grid(image: np.ndarray, grid_size: tuple[int, int]) -> np.ndarray:
        """
        Vectorized division of the preprocessed image into a grid.

        Parameters
        ----------
        image : np.ndarray
            Preprocessed RGB image.
        grid_size : (int, int)
            Number of rows and columns.

        Returns
        -------
        np.ndarray
            Grid of shape (rows, cols, tile_h, tile_w, 3).
        """

        rows, cols = grid_size
        h, w = image.shape[:2]

        tile_h = h // rows
        tile_w = w // cols

        image = image[: rows * tile_h, : cols * tile_w]

        grid = image.reshape(rows, tile_h, cols, tile_w, 3)
        grid = grid.transpose(0, 2, 1, 3, 4)

        return grid