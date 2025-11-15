# mosaic_generator/mosaic_builder.py

"""
MosaicBuilder constructs the final mosaic image using vectorized
distance calculations for both average-color and histogram-based
matching. This module eliminates Python loops in the hot path.
"""

import numpy as np
import cv2

from .config import TILE_SIZE, TOP_K
from .utils import ensure_uint8, validate_grid_size


class MosaicBuilder:
    """
    Builds mosaics from preprocessed images and cached tile features.

    Parameters
    ----------
    avg_colors : np.ndarray
        Array of shape (N, 3) with average RGB for each tile.
    histograms : np.ndarray
        Array of shape (N, B^3) with histogram features.
    tiles : np.ndarray
        Array of shape (N, TILE_SIZE, TILE_SIZE, 3) with RGB tiles.
    """

    def __init__(self, avg_colors: np.ndarray,
                 histograms: np.ndarray,
                 tiles: np.ndarray):
        self.avg_colors = avg_colors
        self.histograms = histograms
        self.tiles = tiles

    # ------------------------------------------------------------------
    # Matching Strategies
    # ------------------------------------------------------------------
    def _match_by_average(self, cell_avgs: np.ndarray) -> np.ndarray:
        """
        Vectorized average-color matching.

        cell_avgs : (M, 3)
        returns    : (M,) array of tile indices
        """
        diff = self.avg_colors[None, :, :] - cell_avgs[:, None, :]
        dist = np.linalg.norm(diff, axis=2)
        return np.argmin(dist, axis=1)

    def _match_by_histogram(self, cell_hists: np.ndarray) -> np.ndarray:
        """
        Vectorized histogram matching using L2 distance.

        cell_hists : (M, H)
        returns    : (M,) array of tile indices
        """
        diff = self.histograms[None, :, :] - cell_hists[:, None, :]
        dist = np.linalg.norm(diff, axis=2)
        return np.argmin(dist, axis=1)

    # ------------------------------------------------------------------
    # Mosaic Construction
    # ------------------------------------------------------------------
    def build(self, grid: np.ndarray, method="histogram") -> np.ndarray:
        """
        Construct the mosaic from grid cells.

        Parameters
        ----------
        grid : np.ndarray
            Array of shape (rows, cols, tile_h, tile_w, 3)
        method : str
            'average' or 'histogram'

        Returns
        -------
        np.ndarray
            Final mosaic image.
        """

        rows, cols = grid.shape[:2]
        tile_h, tile_w = TILE_SIZE, TILE_SIZE

        # Compute per-cell features
        cells = grid.reshape(rows * cols, tile_h, tile_w, 3)

        if method == "average":
            cell_feats = cells.reshape(len(cells), -1, 3).mean(axis=1)
            tile_idx = self._match_by_average(cell_feats)

        elif method == "histogram":
            hist = []
            for cell in cells:
                h = cv2.calcHist(
                    [cell],
                    [0, 1, 2],
                    None,
                    [self.histograms.shape[-1] ** (1/3)] * 3,   # uses same bin count
                    [0, 256] * 3
                )
                cv2.normalize(h, h)
                hist.append(h.flatten())
            cell_feats = np.vstack(hist)
            tile_idx = self._match_by_histogram(cell_feats)

        else:
            raise ValueError("Unknown matching method.")

        # Tile placement
        mosaic = np.zeros((rows * tile_h, cols * tile_w, 3), dtype=np.uint8)
        for idx, t in enumerate(tile_idx):
            i = idx // cols
            j = idx % cols
            mosaic[i * tile_h:(i + 1) * tile_h,
                   j * tile_w:(j + 1) * tile_w] = self.tiles[t]

        return mosaic