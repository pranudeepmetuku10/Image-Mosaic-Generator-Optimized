import gradio as gr
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import tempfile
import os

from mosaic_generator import (
    TileManager,
    ImageProcessor,
    MosaicBuilder,
    compute_metrics,
    PREPROCESS_RESOLUTION,
)


# ---------------------------------------------------------------------
# Initialize system
# ---------------------------------------------------------------------

TILE_FOLDER = "tiles"  # Folder in your repo on HuggingFace

tile_manager = TileManager(TILE_FOLDER)
builder = MosaicBuilder(tile_manager)


# ---------------------------------------------------------------------
# Core mosaic pipeline
# ---------------------------------------------------------------------

def run_mosaic_pipeline(
    image,
    grid_rows,
    grid_cols,
    tile_set,
    classification_method,
    apply_quantization,
    quant_colors,
    show_performance
):
    if image is None:
        return None, "Please upload an image.", None, None

    # Convert PIL → NumPy
    if hasattr(image, "convert"):
        image_np = np.array(image.convert("RGB"))
    else:
        image_np = np.array(image)

    # Preprocess image
    processed = ImageProcessor.preprocess(
        image_np,
        target_size=PREPROCESS_RESOLUTION,
        apply_quantization=apply_quantization,
        n_colors=int(quant_colors)
    )

    # Slice into grid
    grid = ImageProcessor.slice_grid(processed, (grid_rows, grid_cols))

    # Handle geometric fallback
    if tile_set == "geometric":
        tile_set = "real_photos"

    # Tile matching
    assignments = builder.match_tiles(grid, method=classification_method)

    # Build mosaic
    tile_h = processed.shape[0] // grid_rows
    tile_w = processed.shape[1] // grid_cols

    mosaic = builder.build(assignments, tile_h, tile_w)
    metrics = compute_metrics(processed, mosaic)

    mosaic_pil = Image.fromarray(mosaic)

    # Performance text
    text = f"""
### 📊 Performance Metrics
- **Grid:** {grid_rows} × {grid_cols}
**Image Quality**
- **MSE:** {metrics['mse']:.2f}
- **PSNR:** {metrics['psnr']:.2f} dB
- **SSIM:** {metrics['ssim']:.4f}
- **Color Similarity:** {metrics['color_similarity']:.4f}
"""

    # Plot performance metrics
    plot = None
    if show_performance:
        values = [
            metrics["mse"],
            metrics["psnr"],
            metrics["ssim"],
            metrics["color_similarity"]
        ]
        labels = ["MSE", "PSNR", "SSIM", "ColorSim"]
        norm = [x / max(values) if max(values) > 0 else 0 for x in values]

        fig, ax = plt.subplots(figsize=(5, 3))
        bars = ax.bar(labels, norm)
        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.02,
                f"{val:.2f}",
                ha="center"
            )

        ax.set_ylim(0, 1.2)
        ax.set_title("Normalized Performance Metrics")
        plt.tight_layout()
        plot = fig

    # Save mosaic to a temporary file (required for gr.File)
    tmp_dir = tempfile.mkdtemp()
    file_path = os.path.join(tmp_dir, "mosaic.png")
    mosaic_pil.save(file_path)

    return mosaic_pil, text, plot, file_path


# ---------------------------------------------------------------------
# Build Gradio Interface
# ---------------------------------------------------------------------

def build_interface():
    return gr.Interface(
        fn=run_mosaic_pipeline,
        inputs=[
            gr.Image(type="pil", label="Upload Image"),
            gr.Slider(8, 64, value=16, step=1, label="Grid Rows"),
            gr.Slider(8, 64, value=16, step=1, label="Grid Columns"),
            gr.Dropdown(
                ["real_photos", "geometric"],
                value="real_photos",
                label="Tile Set (Geometric for UI only)"
            ),
            gr.Dropdown(
                ["dominant_color", "average_color", "histogram"],
                value="dominant_color",
                label="Tile Matching Method"
            ),
            gr.Checkbox(label="Apply Color Quantization", value=False),
            gr.Slider(4, 32, value=12, step=1, label="Number of Colors (Quantization)"),
            gr.Checkbox(label="Show Performance Visualization", value=False),
        ],
        outputs=[
            gr.Image(type="pil", label="Mosaic Result"),
            gr.Markdown(label="Performance Metrics"),
            gr.Plot(label="Metrics Visualization"),
            gr.File(label="Download Mosaic")
        ],
        title="🖼️ Image Mosaic Generator — Lab 5 (Optimized + Modular)",
        description="""
Upload an image to generate an optimized mosaic using vectorized NumPy operations
and real photo tiles. The UI includes all Lab-1 options for continuity.
**Recommended**
- Tile Set: real_photos  
- Method: dominant_color (fast mode)  
- Grid: 16×16 → 32×32
""",
        allow_flagging="never",
        cache_examples=False,
    )


app = build_interface()

if __name__ == "__main__":
    app.launch()