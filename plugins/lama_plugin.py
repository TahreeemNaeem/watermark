"""
LaMa/SD Inpaint plugin for ws.py

This plugin provides a pre-trained, per-image-no-training inpainting engine.
It tries Stable Diffusion Inpainting (diffusers). If diffusers is not
installed, it raises a clear ImportError so you can install the deps.

Install (CPU-only example):
    pip install --upgrade pip
    pip install diffusers transformers accelerate safetensors torch

If you have a GPU, install a CUDA-enabled PyTorch before diffusers for speed.

Usage via ws.py:
    python ws.py image.png outputs/final_lama.png --engine lama --save-mask

Note: Despite the name, this plugin currently uses Stable Diffusion Inpainting
as a high-quality, pre-trained method. You can replace this with an actual
LaMa implementation later without changing ws.py.
"""
from typing import Optional
import os
import numpy as np
from PIL import Image
import cv2


def inpaint(image_path: str, mask_path: str, output_path: str, model_dir: Optional[str] = None, device: Optional[str] = None) -> bool:
    try:
        import torch
        from diffusers import StableDiffusionInpaintPipeline
    except Exception as e:
        raise ImportError(
            "Stable Diffusion Inpainting dependencies missing. Install with:\n"
            "  pip install diffusers transformers accelerate safetensors torch\n"
            f"Original error: {e}"
        )

    # Load image and mask
    image_pil = Image.open(image_path).convert("RGB")
    mask_pil = Image.open(mask_path).convert("L")

    # SD expects white = masked area to fill
    # Our mask is already 255 for to-remove; ensure binary
    mask_pil = mask_pil.point(lambda p: 255 if p > 0 else 0)

    # Optional mask refinement to avoid overpaint and reduce blur
    erode_px = int(os.environ.get("WMR_SD_ERODE", 1))
    feather_px = int(os.environ.get("WMR_SD_FEATHER", 12))
    mask_np = np.array(mask_pil)
    if erode_px > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask_np = cv2.erode(mask_np, k, iterations=erode_px)
    mask_bin = (mask_np > 0).astype(np.uint8) * 255
    mask_pil = Image.fromarray(mask_bin)

    # Choose device
    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        else:
            try:
                if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                    device = "mps"
                else:
                    device = "cpu"
            except Exception:
                device = "cpu"

    # Load pipeline
    # Model card: runwayml/stable-diffusion-inpainting (2.1 requires FP16 with GPU)
    # For broad compatibility, we use the 1.5 inpainting model.
    model_id = os.environ.get("WMR_SD_MODEL", "runwayml/stable-diffusion-inpainting")
    pipe = StableDiffusionInpaintPipeline.from_pretrained(model_id)
    pipe = pipe.to(device)

    # Conservative prompt setup to preserve content
    prompt = os.environ.get("WMR_SD_PROMPT", "clean background, same scene, remove text watermark")
    negative_prompt = os.environ.get(
        "WMR_SD_NEGATIVE",
        "low quality, blurry, deformed, distorted, extra digits, extra limbs, watermark, text",
    )

    # Guidance and steps — adjust via env if needed
    guidance_scale = float(os.environ.get("WMR_SD_GUIDANCE", 6.5))
    num_inference_steps = int(os.environ.get("WMR_SD_STEPS", 35))
    strength = float(os.environ.get("WMR_SD_STRENGTH", 0.75))

    result_pil = pipe(
        prompt=prompt,
        image=image_pil,
        mask_image=mask_pil,
        negative_prompt=negative_prompt,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        strength=strength,
    ).images[0]

    # Feathered composite with the original image to keep boundaries sharp
    try:
        img_np = np.array(image_pil)
        res_np = np.array(result_pil)
        m = (mask_bin > 0).astype(np.uint8) * 255
        if feather_px > 0:
            alpha = cv2.GaussianBlur(m.astype(np.float32) / 255.0, (0, 0), sigmaX=feather_px / 3)
        else:
            alpha = (m > 0).astype(np.float32)
        alpha = np.clip(alpha[..., None], 0.0, 1.0)
        comp = (res_np.astype(np.float32) * alpha + img_np.astype(np.float32) * (1.0 - alpha)).astype(np.uint8)
        Image.fromarray(comp).save(output_path)
    except Exception:
        # Fallback: save SD output directly
        result_pil.save(output_path)
    return True
