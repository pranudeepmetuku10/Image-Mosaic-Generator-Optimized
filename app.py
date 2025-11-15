"""
Optimized Gradio application for the Lab 5 Image Mosaic Generator.
Uses modular components:
- TileManager
- ImageProcessor
- MosaicBuilder
- Metrics

This file contains *no algorithmic logic* — only orchestration and UI,
as required by proper modular design.
"""

import time
import numpy as np
import gradio as gr
from PIL import Image

from mosaic_generator import (
    TileManager,
    ImageProcessor,
    MosaicBuilder,
    compute_all,
)
from mosaic_generator.config import (
    APP_TITLE,
    APP_DESCRIPTION,
    PREPROCESS_RESOLUTION,
)

# Load and cache real-photo tiles (fast after the first run)
tile_manager = TileManager(tile_folder="tiles")

builder = MosaicBuilder(
    avg_colors=tile_manager.get_avg_colors(),
    histograms=tile_manager.get_histograms(),
    tiles=tile_manager.get_tiles(),
)


# Main mosaic generation function
def generate_mosaic(
    image: Image.Image,
    grid_rows: int,
    grid_cols: int,
    method: str,
    apply_quantization: bool,
    n_colors: int,
):
    if image is None:
        return None, "Please upload an image first."

    try:
        start = time.perf_counter()

        # Convert to numpy
        img_np = np.array(image.convert("RGB"))

        # Preprocess image
        processed = ImageProcessor.preprocess(
            img_np,
            apply_quantization=apply_quantization,
            n_colors=n_colors,
        )

        # Vectorized grid slicing
        grid = ImageProcessor.divide_into_grid(processed, (grid_rows, grid_cols))

        # Build mosaic (vectorized)
        mosaic = builder.build(grid, method=method)

        # Compute metrics
        metrics = compute_all(processed, mosaic)

        elapsed = time.perf_counter() - start

        metrics_md = f"""
### Performance Metrics

**Processing Time:** {elapsed:.3f} seconds  
**Grid:** {grid_rows} × {grid_cols} = {grid_rows * grid_cols} cells  
**Preprocessed Resolution:** {PREPROCESS_RESOLUTION[0]} × {PREPROCESS_RESOLUTION[1]}

---

### Image Quality

- **MSE:** {metrics['mse']:.2f}
- **PSNR:** {metrics['psnr']:.2f} dB
- **SSIM:** {metrics['ssim']:.4f}
- **Color Similarity:** {metrics['color_similarity']:.4f}
"""

        return Image.fromarray(mosaic), metrics_md

    except Exception as exc:
        return None, f"Error: {exc}"


def create_interface():
    return gr.Interface(
        fn=generate_mosaic,
        inputs=[
            gr.Image(type="pil", label="Upload Image"),
            gr.Slider(8, 64, value=16, step=1, label="Grid Rows"),
            gr.Slider(8, 64, value=16, step=1, label="Grid Columns"),
            gr.Dropdown(
                choices=["average", "histogram"],
                value="histogram",
                label="Tile Matching Method"
            ),
            gr.Checkbox(value=False, label="Apply Color Quantization"),
            gr.Slider(4, 32, value=16, step=2, label="Number of Quantization Colors"),
        ],
        outputs=[
            gr.Image(type="pil", label="Mosaic Result"),
            gr.Markdown(label="Metrics"),
        ],
        title=APP_TITLE,
        description=APP_DESCRIPTION,
        allow_flagging="never",
        cache_examples=False,
    )


if __name__ == "__main__":
    demo = create_interface()
    demo.launch(server_name="0.0.0.0", server_port=7860, share=True)