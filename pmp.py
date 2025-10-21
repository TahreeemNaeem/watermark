from diffusers import BrushNetModel, StableDiffusionBrushNetPipeline
import torch
from PIL import Image

# Load BrushNet (~1.5GB)
brushnet = BrushNetModel.from_pretrained(
    "TencentARC/brushnet-ckpt",
    torch_dtype=torch.float16
)

pipe = StableDiffusionBrushNetPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    brushnet=brushnet,
    torch_dtype=torch.float16
)

# Usage
image = Image.open("input.jpg")
mask = Image.open("mask.jpg")  # Still needs mask

result = pipe(
    prompt="remove watermark",
    image=image,
    mask_image=mask
).images[0]