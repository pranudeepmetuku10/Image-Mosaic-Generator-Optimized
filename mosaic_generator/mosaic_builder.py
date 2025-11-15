"""
mosaic_builder.py

Core mosaic construction. Handles:
- tile selection (avg color or histogram)
- vectorized matching
- assembly of final output mosaic
"""

import numpy as np
import cv2

from .utils import compute_average_color, compute_histogram
from .config import TILE_SIZE


class MosaicBuilder:
    """
    Build mosaics using preloaded tiles (from TileManager).
    """

    def __init__(self, tile_manager):
        self.tile_manager = tile_manager

    def match_tiles(self, grid: np.ndarray, method: str = "average_color") -> np.ndarray:
        """
        Match each grid cell to nearest tile using selected method:
        - average_color
        - histogram
        """
        rows, cols = grid.shape[:2]
        assignments = np.zeros((rows, cols), dtype=int)

        if method == "average_color":
            tile_features = self.tile_manager.avg_colors
            for i in range(rows):
                for j in range(cols):
                    cell_avg = compute_average_color(grid[i, j])
                    idx = np.argmin(np.linalg.norm(tile_features - cell_avg, axis=1))
                    assignments[i, j] = idx

        elif method == "histogram":
            tile_hists = self.tile_manager.normalized_hists
            for i in range(rows):
                for j in range(cols):
                    hist = compute_histogram(grid[i, j])
                    dists = np.array([
                        cv2.compareHist(hist, th, cv2.HISTCMP_CHISQR)
                        for th in tile_hists
                    ])
                    assignments[i, j] = np.argmin(dists)

        else:
            raise ValueError(f"Unsupported matching method: {method}")

        return assignments

    def build(self, assignments: np.ndarray, tile_h: int, tile_w: int) -> np.ndarray:
        """
        Construct the final mosaic using tile assignments.
        """
        rows, cols = assignments.shape
        mosaic = np.zeros((rows * tile_h, cols * tile_w, 3), dtype=np.uint8)

        for i in range(rows):
            for j in range(cols):
                tile_img = self.tile_manager.tiles[assignments[i, j]]
                if tile_img.shape[:2] != (tile_h, tile_w):
                    tile_img = cv2.resize(tile_img, (tile_w, tile_h))
                y0, y1 = i * tile_h, (i + 1) * tile_h
                x0, x1 = j * tile_w, (j + 1) * tile_w
                mosaic[y0:y1, x0:x1] = tile_img

        return mosaic