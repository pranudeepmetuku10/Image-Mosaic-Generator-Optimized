"""
image_processor.py

Handles image loading, resizing, preprocessing, grid slicing, and optional
color quantization. Fully vectorized and optimized for Lab-5 performance requirements.
"""

import numpy as np
import cv2
from PIL import Image

from .config import PREPROCESS_RESOLUTION


class ImageProcessor:
    """
    Provides preprocessing utilities:
    - image resizing with aspect preservation
    - optional quantization
    - vectorized grid slicing
    """

    @staticmethod
    def preprocess(image: np.ndarray,
                   target_size: tuple[int, int] = PREPROCESS_RESOLUTION,
                   apply_quantization: bool = False,
                   n_colors: int = 8) -> np.ndarray:
        """
        Resize image with aspect ratio preserved and optional padding.
        Optionally apply color quantization using KMeans.
        """
        h, w = image.shape[:2]
        target_h, target_w = target_size

        scale = min(target_h / h, target_w / w)
        new_h, new_w = int(h * scale), int(w * scale)

        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)

        top = (target_h - new_h) // 2
        bottom = target_h - new_h - top
        left = (target_w - new_w) // 2
        right = target_w - new_w - left

        padded = cv2.copyMakeBorder(
            resized, top, bottom, left, right,
            cv2.BORDER_CONSTANT, value=(0, 0, 0)
        )

        final_img = padded

        if apply_quantization:
            final_img = ImageProcessor.quantize(final_img, n_colors=n_colors)

        return final_img

    @staticmethod
    def quantize(image: np.ndarray, n_colors: int = 8) -> np.ndarray:
        """
        Apply KMeans color quantization.
        """
        data = image.reshape(-1, 3).astype(np.float32)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)

        _, labels, centers = cv2.kmeans(
            data, n_colors, None, criteria, 5, cv2.KMEANS_RANDOM_CENTERS
        )

        centers = np.uint8(centers)
        quantized = centers[labels.flatten()]
        return quantized.reshape(image.shape)

    @staticmethod
    def slice_grid(image: np.ndarray, grid_size: tuple[int, int]) -> np.ndarray:
        """
        Vectorized slicing of the image into a uniform grid.
        Returns array of shape (rows, cols, tile_h, tile_w, 3).
        """
        rows, cols = grid_size
        h, w = image.shape[:2]

        tile_h, tile_w = h // rows, w // cols
        cropped = image[:rows * tile_h, :cols * tile_w]

        grid = cropped.reshape(rows, tile_h, cols, tile_w, 3)
        grid = grid.transpose(0, 2, 1, 3, 4)

        return grid