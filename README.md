# Watermark Removal (Unified CLI)

Single CLI that detects and removes watermarks while preserving image quality.

## Features

- ✅ **Automatic Detection**: No manual mask creation needed
- ✅ **No Blurring**: Enhanced perceptual loss preserves image sharpness  
- ✅ **High Accuracy**: Multiple detection methods for precision
- ✅ **Quality Preservation**: Advanced algorithms maintain detail
- ✅ **Easy to Use**: Simple one-command execution

## Installation

### Installation

1. Create a virtual environment:
```bash
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

2. Install dependencies (fast path + utils):
```bash
pip install -r requirements.txt
```

## Usage

1. Place your watermarked image as `image.png` in this directory
2. Fast, high‑quality removal (recommended):
```bash
python ws.py image.png outputs/image_cleaned.png --save-mask --quality high --radius 2 --method telea --feather 12 --debug --debug-dir outputs/debug
```

3. Highest quality (AI engine — optional): install PyTorch and run
```bash
pip install --prefer-binary --index-url https://download.pytorch.org/whl/cpu torch torchvision torchaudio
python ws.py image.png outputs/image_cleaned_ai.png --engine ai --quality balanced --save-mask --debug --debug-dir outputs/debug
```

4. Simple UI (upload → result)
```bash
pip install gradio
python -m ui.app
# A local URL will open in the browser; upload an image and choose engine
```

4. Pretrained model (LaMa) — no per‑image training (requires plugin)
```bash
# Option A: Implement a plugin using a LaMa library (e.g., lama-cleaner or saicinpainting)
# 1) Copy plugins/lama_plugin_example.py to plugins/lama_plugin.py and implement `inpaint`
# 2) Provide your LaMa weights via --lama-model-dir if your plugin needs it
python ws.py image.png outputs/image_cleaned_lama.png --engine lama --lama-model-dir pretrained/lama --save-mask --debug --debug-dir outputs/debug

# If the plugin is not present or fails, ws.py falls back to the fast engine automatically.
```

## Output Files

After running, you'll get:
- `outputs/` - cleaned images and masks
- `outputs/debug/` - debug masks when `--debug` is enabled

## How It Works

1. **Automatic Detection**: Uses multiple detection methods to identify watermarks:
   - High contrast region detection
   - Color uniformity analysis
   - Position-based detection (corners, edges, center)
   - Edge density analysis

2. **Quality Preservation**: Uses enhanced perceptual loss functions to:
   - Prevent blurring with total variation loss
   - Preserve details with gradient-based loss
   - Maintain texture with frequency domain loss
   - Use multi-scale processing for sharpness

3. **Advanced Removal**: Optimized neural network training to remove watermarks while maintaining image quality

## Troubleshooting

- **Missing Dependencies**: Run `python install_helper.py` or use the setup script
- **Memory Issues**: The system automatically resizes large images to 768px max dimension
- **No image.png**: Make sure you have an `image.png` file in the project directory
- **Virtual Environment**: If using a virtual environment, make sure it's activated before running

## Cleaning

Use the cleaner script to remove generated artifacts while keeping your `.venv`:

```bash
python scripts/clean.py --dry-run    # preview what would be removed
python scripts/clean.py              # perform cleanup (safe: skips tracked files and .venv)
```

Options:
- `--include-tracked` delete even git-tracked files (use with care)
- `--remove-venv` also remove the `.venv` folder
- `--patterns PAT1 PAT2` add extra glob patterns

## Project Layout

- `ws.py` – unified CLI (fast inpaint + AI engine)
- `enhanced_perceptual_api.py` – enhanced Deep Image Prior remover
- `helper.py`, `model/` – model utilities
- `scripts/clean.py` – cleanup utility
- `outputs/` – results and debug artifacts

## System Requirements

- Python 3.7 or higher
- At least 4GB RAM (8GB recommended for large images)
- For GPU acceleration: CUDA-compatible GPU with appropriate PyTorch installation
