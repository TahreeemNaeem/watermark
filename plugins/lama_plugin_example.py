"""
Optional LaMa plugin example.

To use a pre-trained LaMa inpainting model (train once, reuse),
install a LaMa implementation and rename this file to `lama_plugin.py`.

Recommended options (choose one):
1) lama-cleaner (simple install)
   - pip install lama-cleaner onnxruntime
   - Download the 'lama' weights via lama-cleaner docs (e.g., big-lama)
   - Then implement `inpaint` below using lama-cleaner's API.

2) saicinpainting (original LaMa)
   - Follow the repo instructions to download checkpoints
   - Implement `inpaint` to load the checkpoint and run inference.

This plugin will be auto-detected by ws.py when you run with `--engine lama`.
If the plugin is not present, ws.py will fall back to the fast OpenCV path.
"""
from typing import Optional


def inpaint(image_path: str, mask_path: str, output_path: str, model_dir: Optional[str] = None, device: Optional[str] = None) -> bool:
    """
    Implement this using your preferred LaMa library.

    Args:
        image_path: input image path
        mask_path:  path to 0/255 mask (255 = region to remove)
        output_path: where to save the final image
        model_dir: directory with pre-trained LaMa weights (if needed by the lib)
        device: 'cuda', 'mps', or 'cpu'

    Returns:
        True on success, False otherwise.
    """
    raise NotImplementedError("Rename this file to lama_plugin.py and implement using a LaMa library.")

