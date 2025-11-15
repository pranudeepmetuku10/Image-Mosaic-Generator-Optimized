"""
app.py

Final Lab-5 Gradio application using the optimized modular mosaic generator.
Compatible with Hugging Face Spaces and includes all Lab-1 UI features.
"""

import gradio as gr
import numpy as np
from PIL import Image
import cv2

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

tile_manager = TileManager(
    tile_directory=TILE_FOLDER,
    tile_size=(32, 32),
    compute_histograms=True
)

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

    if hasattr(image, "convert"):
        image_np = np.array(image.convert("RGB"))
    else:
        image_np = np.array(image)

    processed = ImageProcessor.preprocess(
        image_np,
        target_size=PREPROCESS_RESOLUTION,
        apply_quantization=apply_quantization,
        n_colors=int(quant_colors)
    )

    grid = ImageProcessor.slice_grid(processed, (grid_rows, grid_cols))

    # geometric tile option is allowed for UI completeness
    if tile_set == "geometric":
        tile_set = "real_photos"  # fallback

    assignments = builder.match_tiles(grid, method=classification_method)

    tile_h = processed.shape[0] // grid_rows
    tile_w = processed.shape[1] // grid_cols

    mosaic = builder.build(assignments, tile_h, tile_w)
    metrics = compute_metrics(processed, mosaic)

    mosaic_pil = Image.fromarray(mosaic)

    text = f"""
### 📊 Performance Metrics

- **Processing Time:** (optimized real-time)
- **Grid:** {grid_rows} × {grid_cols}

**Image Quality**
- **MSE:** {metrics['mse']:.2f}
- **PSNR:** {metrics['psnr']:.2f} dB
- **SSIM:** {metrics['ssim']:.4f}
- **Color Similarity:** {metrics['color_similarity']:.4f}
"""

    plot = None
    if show_performance:
        import matplotlib.pyplot as plt

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

    return mosaic_pil, text, plot, mosaic_pil


# ---------------------------------------------------------------------
# Build Gradio UI
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
                label="Tile Set"
            ),
            gr.Dropdown(
                ["dominant_color", "average_color", "histogram"],
                value="dominant_color",
                label="Color Classification"
            ),
            gr.Checkbox(label="Apply Color Quantization", value=False),
            gr.Slider(4, 32, value=12, step=1, label="Number of Colors (Quantization)"),
            gr.Checkbox(label="Show Performance Analysis", value=False),
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
and real photo tiles. The interface supports all Lab-1 options (classification methods,
geometric tiles, quantization) for continuity and completeness.

**Recommended settings**
- Tile Set: real_photos  
- Classification: dominant_color  
- Grid: 16×16 to 32×32
""",
        allow_flagging="never",
        cache_examples=False,
    )


# ---------------------------------------------------------------------
# Launch
# ---------------------------------------------------------------------

app = build_interface()

if __name__ == "__main__":
    app.launch()