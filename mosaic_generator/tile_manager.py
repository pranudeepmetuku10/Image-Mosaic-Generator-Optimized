# mosaic_generator/tile_manager.py

"""
TileManager handles loading real-photo mosaic tiles and computing their
cached features, including average RGB vectors and normalized 3D color
histograms. The resulting arrays are optimized for vectorized matching.
"""

import os
import numpy as np
import cv2
from PIL import Image

from .config import (
    TILE_SIZE,
    HIST_BINS,
    NORMALIZE_HISTOGRAMS,
    MIN_TILE_COUNT,
)
from .utils import ensure_uint8


class TileManager:
    """
    Loads and caches real-photo tiles for mosaic generation.
    
    Attributes
    ----------
    tiles : np.ndarray
        Array of shape (N, H, W, 3) containing all resized RGB tiles.
    avg_colors : np.ndarray
        Array of shape (N, 3) representing each tile's mean RGB color.
    histograms : np.ndarray
        Array of shape (N, B^3) containing color histograms.
    """

    def __init__(self, tile_folder: str):
        self.tile_folder = tile_folder
        self.tiles = None
        self.avg_colors = None
        self.histograms = None
        self._load_tiles()

    def _load_tiles(self):
        """Load all tiles from directory and compute their features."""
        tiles, avg_colors, histograms = [], [], []

        for fname in os.listdir(self.tile_folder):
            path = os.path.join(self.tile_folder, fname)

            try:
                image = Image.open(path).convert("RGB")
                image = image.resize((TILE_SIZE, TILE_SIZE), Image.LANCZOS)
                arr = np.array(image)

                tiles.append(arr)
                avg_colors.append(arr.reshape(-1, 3).mean(axis=0))

                hist = cv2.calcHist(
                    [arr],
                    [0, 1, 2],
                    None,
                    [HIST_BINS] * 3,
                    [0, 256] * 3,
                )

                if NORMALIZE_HISTOGRAMS:
                    cv2.normalize(hist, hist)

                histograms.append(hist.flatten())

            except Exception as exc:
                print(f"Skipping {fname}: {exc}")

        if len(tiles) < MIN_TILE_COUNT:
            raise ValueError(
                f"Only {len(tiles)} usable tiles found. "
                f"A minimum of {MIN_TILE_COUNT} is required."
            )

        self.tiles = np.stack(tiles).astype(np.uint8)
        self.avg_colors = np.vstack(avg_colors).astype(np.float32)
        self.histograms = np.vstack(histograms).astype(np.float32)

        print(f"Loaded {len(self.tiles)} tiles from '{self.tile_folder}'.")
        print(f"Tile array:       {self.tiles.shape}")
        print(f"Avg colors:       {self.avg_colors.shape}")
        print(f"Histograms:       {self.histograms.shape}")

    # Public getters
    def get_tiles(self) -> np.ndarray:
        return self.tiles

    def get_avg_colors(self) -> np.ndarray:
        return self.avg_colors

    def get_histograms(self) -> np.ndarray:
        return self.histograms