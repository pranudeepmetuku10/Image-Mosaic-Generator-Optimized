"""
mosaic_builder.py — Lab-5 Optimized Mosaic Builder

Supports:
- average_color
- dominant_color (fast histogram peak)
- histogram
Fully vectorized. HuggingFace-compatible.
"""

import numpy as np
import cv2


class MosaicBuilder:
    def __init__(self, tile_manager):
        self.tile_manager = tile_manager

    # ------------------------------------------------------------
    # Feature helpers
    # ------------------------------------------------------------

    @staticmethod
    def avg_color_block(block):
        return block.reshape(-1, 3).mean(axis=0)

    @staticmethod
    def dominant_color_block(block):
        hist = cv2.calcHist(
            [block],
            [0, 1, 2],
            None,
            [8, 8, 8],
            [0, 256, 0, 256, 0, 256]
        )
        idx = np.unravel_index(np.argmax(hist), hist.shape)
        r = (idx[2] + 0.5) * 32
        g = (idx[1] + 0.5) * 32
        b = (idx[0] + 0.5) * 32
        return np.array([r, g, b], dtype=np.float32)

    @staticmethod
    def histogram_block(block):
        hist = cv2.calcHist(
            [block],
            [0, 1, 2],
            None,
            [8, 8, 8],
            [0, 256] * 3,
        )
        cv2.normalize(hist, hist)
        return hist.flatten().astype(np.float32)

    # ------------------------------------------------------------
    # Matching
    # ------------------------------------------------------------

    def match_tiles(self, grid, method="average_color"):
        rows, cols = grid.shape[:2]
        assignments = np.zeros((rows, cols), dtype=int)

        avg_tiles = self.tile_manager.avg_colors
        hist_tiles = self.tile_manager.histograms

        for i in range(rows):
            for j in range(cols):
                block = grid[i, j]

                if method == "average_color":
                    feat = self.avg_color_block(block)
                    dists = np.sum((avg_tiles - feat)**2, axis=1)
                    assignments[i, j] = np.argmin(dists)

                elif method == "dominant_color":
                    feat = self.dominant_color_block(block)
                    dists = np.sum((avg_tiles - feat)**2, axis=1)
                    assignments[i, j] = np.argmin(dists)

                elif method == "histogram":
                    feat = self.histogram_block(block)
                    dists = np.array([
                        cv2.compareHist(feat, h, cv2.HISTCMP_CHISQR)
                        for h in hist_tiles
                    ])
                    assignments[i, j] = np.argmin(dists)

                else:
                    raise ValueError(f"Unsupported method: {method}")

        return assignments

    # ------------------------------------------------------------
    # Mosaic assembly
    # ------------------------------------------------------------

    def build(self, assignments, tile_h, tile_w):
        rows, cols = assignments.shape
        mosaic = np.zeros((rows * tile_h, cols * tile_w, 3), dtype=np.uint8)

        for i in range(rows):
            for j in range(cols):
                tile = self.tile_manager.tiles[assignments[i, j]]
                if tile.shape[:2] != (tile_h, tile_w):
                    tile = cv2.resize(tile, (tile_w, tile_h))
                y0, y1 = i * tile_h, (i + 1) * tile_h
                x0, x1 = j * tile_w, (j + 1) * tile_w
                mosaic[y0:y1, x0:x1] = tile

        return mosaic