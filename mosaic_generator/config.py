# mosaic_generator/config.py

"""
Configuration constants used across the mosaic generator package.
These values centralize tunable parameters for preprocessing, tile
feature extraction, and optimization behavior.
"""

# Tile configuration
TILE_SIZE = 32
PREPROCESS_RESOLUTION = (512, 512)

# Histogram settings (for real-photo tiles)
HIST_BINS = 8
NORMALIZE_HISTOGRAMS = True

# Matching behavior
TOP_K = 3                # Random tie-breaking among best matches
VECTORIZE_DISTANCES = True
CACHE_TILE_FEATURES = True

# Validation
MIN_TILE_COUNT = 10

# UI metadata
APP_TITLE = "Image Mosaic Generator (Optimized Lab 5)"
APP_DESCRIPTION = (
    "Transforms an input image into a photorealistic mosaic using real "
    "image tiles. Fully optimized using NumPy vectorization and modular design."
)