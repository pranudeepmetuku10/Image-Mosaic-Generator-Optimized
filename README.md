# Image Mosaic Generator — Optimized Edition

A high-performance, modular Python application that transforms any image into a photorealistic mosaic by intelligently tiling it with real photographs. This optimized version uses NumPy vectorization and a clean, production-ready architecture built with Gradio for an intuitive web interface.

## Overview

The Image Mosaic Generator takes an input image and recreates it as a grid of smaller image tiles selected from a tile library. By analyzing the color or dominant features of each grid cell and matching them to the most similar tiles, the result is a detailed mosaic that reveals the original image when viewed from a distance.

This optimized edition focuses on:
- **Performance**: Vectorized NumPy operations eliminate loop-based tile matching
- **Modularity**: Cleanly separated concerns (image processing, tile management, matching, metrics)
- **Flexibility**: Multiple tile-matching strategies and preprocessing options
- **Quality**: Built-in metrics to evaluate mosaic fidelity

## Features

- **Real-photo tiling**: Create mosaics using actual photographs from a customizable tile library
- **Multiple matching strategies**: Choose between dominant color, average color, or histogram-based matching
- **Color quantization**: Optional preprocessing to reduce colors before tiling
- **Performance metrics**: Measure and visualize mosaic quality with MSE, PSNR, SSIM, and color similarity scores
- **Web interface**: Easy-to-use Gradio UI for non-technical users
- **Configurable grid sizes**: Generate mosaics at various resolutions (8×8 to 64×64 grids)
- **Batch processing ready**: Modular Python API for programmatic use

## System Architecture

```
mosaic_generator/
├── __init__.py
├── image_processor.py      # Image loading, resizing, grid division
├── tile_manager.py         # Tile loading, caching, feature extraction
├── mosaic_builder.py       # Main mosaic construction logic
├── metrics.py              # Similarity metrics (MSE, SSIM)
├── config.py               # Configuration constants
└── utils.py                # Helper functions

app.py                      # Gradio interface
requirements.txt            # Dependencies with versions
README.md                   # Documentation
```

Each module has a single responsibility and can be imported independently for programmatic use.

## Installation

### Prerequisites

- Python 3.8 or later
- pip package manager

### Steps

1. **Extract the project:**
   
   If you have a ZIP file:
   - Extract the ZIP folder to your desired location
   - Open a terminal and navigate to the extracted folder:
     ```bash
     cd /path/to/Image-Mosaic-Generator-Optimized
     ```
   
   Alternatively, clone from GitHub:
   ```bash
   git clone https://github.com/pranudeepmetuku10/Image-Mosaic-Generator-Optimized.git
   cd Image-Mosaic-Generator-Optimized
   ```

2. **Create a virtual environment (recommended):**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```
   On Windows:
   ```bash
   python -m venv venv
   venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

### Dependencies

- **numpy** (2.0.2): Vectorized numerical computations
- **opencv-python** (4.12.0): Image I/O and processing
- **gradio** (5.46.0): Web interface framework
- **matplotlib** (3.10.0): Visualization and plotting
- **Pillow** (11.3.0): Image manipulation
- **scikit-image** (0.25.2): Additional image processing algorithms
- **scikit-learn** (1.6.1): Machine learning utilities

## Usage

### Web Interface

The fastest way to generate mosaics is via the Gradio web app:

```bash
python app.py
```

This starts a local web server (typically at `http://localhost:7860`). Upload an image and adjust settings:

- **Grid Rows / Grid Columns**: Set the mosaic resolution (higher = smaller tiles, more detail)
- **Tile Matching Method**:
  - `dominant_color`: Fast, good for most use cases
  - `average_color`: Standard approach
  - `histogram`: More precise color matching (slower)
- **Apply Color Quantization**: Reduce image colors before tiling for stylized effects
- **Show Performance Visualization**: Display quality metrics after generation

### Programmatic API

Use the modular Python API for batch processing or integration into other applications:

```python
from mosaic_generator import TileManager, ImageProcessor, MosaicBuilder, compute_metrics
from PIL import Image
import numpy as np

# Load and setup
tile_manager = TileManager("tiles")
builder = MosaicBuilder(tile_manager)
image = Image.open("input.jpg")
image_np = np.array(image.convert("RGB"))

# Preprocess
processed = ImageProcessor.preprocess(
    image_np,
    target_size=(512, 512),
    apply_quantization=False,
    n_colors=12
)

# Create grid
grid = ImageProcessor.slice_grid(processed, (16, 16))

# Match tiles
assignments = builder.match_tiles(grid, method="dominant_color")

# Build mosaic
tile_h, tile_w = processed.shape[0] // 16, processed.shape[1] // 16
mosaic = builder.build(assignments, tile_h, tile_w)

# Evaluate quality
metrics = compute_metrics(processed, mosaic)
print(f"PSNR: {metrics['psnr']:.2f} dB, SSIM: {metrics['ssim']:.4f}")

# Save result
result = Image.fromarray(mosaic)
result.save("output_mosaic.png")
```

## Configuration

Global settings are centralized in `mosaic_generator/config.py`:

- `TILE_SIZE`: Default tile dimensions (pixels)
- `PREPROCESS_RESOLUTION`: Image preprocessing target size
- `HIST_BINS`: Histogram bin count for color matching
- `TOP_K`: Number of candidate tiles for random tie-breaking
- `CACHE_TILE_FEATURES`: Enable tile feature caching for faster matching

Modify these values to tune performance and output quality.

## Performance Metrics

The application evaluates mosaic quality using:

- **MSE** (Mean Squared Error): Lower is better; measures pixel-level difference
- **PSNR** (Peak Signal-to-Noise Ratio): Higher is better; logarithmic measure of image quality
- **SSIM** (Structural Similarity Index): Range [−1, 1]; measures perceived quality
- **Color Similarity**: Measures overall color distribution match between original and mosaic

## Recommended Settings

- **Artistic effect**: 64×64 grid, dominant_color matching, quantization enabled (8–12 colors)


## Tile Library

Place your image tiles in the `tiles/` folder. The application will:
1. Automatically discover all PNG, JPG, and JPEG files
2. Extract and cache color features from each tile
3. Use these features for efficient matching

For best results, use tiles of consistent size and quality.

## Optimization Details

This optimized version achieves significant performance improvements through:

- **NumPy vectorization**: Replaces nested loops with batch operations
- **Feature caching**: Tile colors/histograms computed once and reused
- **Modular design**: Each component can be optimized independently
- **Efficient data structures**: Proper use of NumPy arrays and indexing

Typical runtimes on modern hardware:
- 512×512 image, 16×16 grid, 100 tiles: ~1–3 seconds
- Same settings with histogram matching: ~3–5 seconds

## Development

To extend or modify the codebase:

1. Review `mosaic_generator/__init__.py` for the public API
2. Each module (tile_manager, image_processor, etc.) is self-contained
3. Add new matching methods in `mosaic_builder.py`
4. Update configuration in `config.py` as needed
5. Add tests to validate changes

## Troubleshooting

**Issue: "No tiles found in folder"**
- Ensure the `tiles/` directory exists and contains image files
- Supported formats: PNG, JPG, JPEG

**Issue: Slow performance on large images**
- Reduce grid size (e.g., use 8×8 instead of 32×32)
- Use `dominant_color` matching instead of `histogram`
- Increase `TILE_SIZE` in config.py

**Issue: Gradio won't start**
- Confirm all dependencies installed: `pip install -r requirements.txt`
- Check Python version (3.8+)
- Verify no other service is using port 7860

## License

This project is provided as-is. Refer to any LICENSE file in the repository for terms.

## Credits

Built and optimized as an enhancement of the Gradio Mosaic Image Generator project, with emphasis on performance and modularity.
