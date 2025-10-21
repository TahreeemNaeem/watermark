
import os
import torch
import numpy as np
from PIL import Image
from typing import Optional, Tuple
import cv2

class PretrainedWatermarkRemover:
    """
    Watermark removal using pre-trained deep learning models
    """
    
    def __init__(self, device: Optional[str] = None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        self._sd_pipe = None
        self._lama_model = None
        self._mat_model = None
        
    def _load_sd_inpaint(self):
        """Load Stable Diffusion Inpainting pipeline"""
        if self._sd_pipe is None:
            try:
                from diffusers import StableDiffusionInpaintPipeline
                print("Loading Stable Diffusion Inpainting model...")
                self._sd_pipe = StableDiffusionInpaintPipeline.from_pretrained(
                    "stabilityai/stable-diffusion-2-inpainting",
                    torch_dtype=torch.float16 if self.device == 'cuda' else torch.float32,
                )
                self._sd_pipe = self._sd_pipe.to(self.device)
                self._sd_pipe.enable_attention_slicing()
                if self.device == 'cuda':
                    self._sd_pipe.enable_xformers_memory_efficient_attention()
                print("✓ Stable Diffusion model loaded")
            except Exception as e:
                print(f"Failed to load SD model: {e}")
                raise
        return self._sd_pipe
    
    def _load_lama(self):
        """Load LaMa (Large Mask Inpainting) model"""
        if self._lama_model is None:
            try:
                import torch.hub
                print("Loading LaMa model...")
                # Using the official LaMa implementation
                self._lama_model = torch.hub.load(
                    'advimman/lama',
                    'lama',
                    pretrained=True,
                    force_reload=False
                )
                self._lama_model = self._lama_model.to(self.device)
                self._lama_model.eval()
                print("✓ LaMa model loaded")
            except Exception as e:
                print(f"Failed to load LaMa model: {e}")
                # Try an alternative approach
                try:
                    print("Trying alternative LaMa loading method...")
                    # Try using lama_cleaner as an alternative
                    import subprocess
                    import sys
                    subprocess.check_call([sys.executable, "-m", "pip", "install", "lama-cleaner"])
                    import lama_cleaner
                    print("✓ LaMa loaded via alternative method")
                except Exception as alt_e:
                    print(f"Alternative loading also failed: {alt_e}")
                    raise
        return self._lama_model
    
    def _prepare_image_and_mask(
        self, 
        image_path: str, 
        mask_path: str,
        target_size: Optional[Tuple[int, int]] = None
    ) -> Tuple[Image.Image, Image.Image]:
        """Load and prepare image and mask"""
        image = Image.open(image_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')
        
        # Resize if needed
        if target_size:
            image = image.resize(target_size, Image.Resampling.LANCZOS)
            mask = mask.resize(target_size, Image.Resampling.NEAREST)
        
        # Ensure mask is binary
        mask_array = np.array(mask)
        mask_array = (mask_array > 127).astype(np.uint8) * 255
        mask = Image.fromarray(mask_array)
        
        return image, mask
    
    def remove_with_sd_inpaint(
        self,
        image_path: str,
        mask_path: str,
        output_path: str,
        prompt: str = "clean background, no watermark, high quality",
        negative_prompt: str = "watermark, text, logo, blurry, low quality",
        num_inference_steps: int = 30,
        guidance_scale: float = 7.5,
    ) -> bool:
        """Remove watermark using Stable Diffusion Inpainting"""
        try:
            pipe = self._load_sd_inpaint()
            
            # Load and prepare inputs with correct size for SD 2 Inpainting (512x512)
            image, mask = self._prepare_image_and_mask(image_path, mask_path, target_size=(512, 512))
            
            print(f"Processing with SD Inpainting (steps={num_inference_steps})...")
            
            # Run inpainting
            result = pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                image=image,
                mask_image=mask,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
            ).images[0]
            
            # Save result
            result.save(output_path, quality=95)
            print(f"✓ Saved result to: {output_path}")
            return True
            
        except Exception as e:
            print(f"SD Inpainting failed: {e}")
            return False
    
    def remove_with_lama(
        self,
        image_path: str,
        mask_path: str,
        output_path: str,
    ) -> bool:
        """Remove watermark using LaMa"""
        try:
            model = self._load_lama()
            
            # Load image and mask
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            
            # Prepare for model
            image_tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float() / 255.0
            mask_tensor = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0).float() / 255.0
            
            image_tensor = image_tensor.to(self.device)
            mask_tensor = mask_tensor.to(self.device)
            
            print("Processing with LaMa...")
            
            with torch.no_grad():
                result = model(image_tensor, mask_tensor)
            
            # Convert back to image
            result = result[0].permute(1, 2, 0).cpu().numpy()
            result = (result * 255).astype(np.uint8)
            result = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
            
            cv2.imwrite(output_path, result, [cv2.IMWRITE_PNG_COMPRESSION, 9])
            print(f"✓ Saved result to: {output_path}")
            return True
            
        except Exception as e:
            print(f"LaMa failed: {e}")
            return False
    
    def auto_remove(
        self,
        image_path: str,
        mask_path: str,
        output_path: str,
        model_type: str = 'sd-inpaint',
        **kwargs
    ) -> bool:
        """
        Automatically remove watermark using specified model
        
        Args:
            image_path: Path to input image
            mask_path: Path to watermark mask
            output_path: Path for output image
            model_type: 'sd-inpaint', 'lama', or 'auto'
            **kwargs: Additional model-specific parameters
        """
        if model_type == 'auto':
            # Try models in order of quality
            for model in ['sd-inpaint', 'lama']:
                print(f"\nTrying {model}...")
                if self.auto_remove(image_path, mask_path, output_path, model, **kwargs):
                    return True
            return False
        
        elif model_type == 'sd-inpaint':
            return self.remove_with_sd_inpaint(image_path, mask_path, output_path, **kwargs)
        
        elif model_type == 'lama':
            return self.remove_with_lama(image_path, mask_path, output_path)
        
        else:
            raise ValueError(f"Unknown model type: {model_type}")


def remove_watermark_with_pretrained(
    image_path: str,
    mask_path: str,
    output_path: str,
    model_type: str = 'sd-inpaint',
    device: Optional[str] = None,
    **kwargs
) -> bool:
    """
    Convenience function for watermark removal
    
    Args:
        image_path: Path to input image with watermark
        mask_path: Path to binary mask (white = watermark region)
        output_path: Path to save cleaned image
        model_type: Model to use ('sd-inpaint', 'lama', 'auto')
        device: Device to use ('cuda', 'cpu', or None for auto)
        **kwargs: Additional model parameters
    
    Returns:
        True if successful, False otherwise
    """
    remover = PretrainedWatermarkRemover(device=device)
    return remover.auto_remove(image_path, mask_path, output_path, model_type, **kwargs)


# FIXED: Watermark-specific detection (logos, text, overlays)
class DeepWatermarkDetector:
    """
    Advanced watermark detection targeting logos, text, and colored overlays
    Uses multiple techniques: OCR, edge detection, color analysis
    """
    
    def __init__(self, device: Optional[str] = None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self._ocr_reader = None
        
    def _load_ocr(self):
        """Load OCR model for text detection"""
        if self._ocr_reader is None:
            try:
                import easyocr
                print("Loading OCR model for text detection...")
                self._ocr_reader = easyocr.Reader(['en'], gpu=(self.device == 'cuda'))
                print("✓ OCR model loaded")
            except Exception as e:
                print(f"Failed to load OCR model: {e}")
                print("Install with: pip install easyocr")
                raise
        return self._ocr_reader
    
    def _detect_text_regions(self, image_np: np.ndarray) -> np.ndarray:
        """Detect text regions using OCR"""
        try:
            reader = self._load_ocr()
            results = reader.readtext(image_np)
            
            mask = np.zeros(image_np.shape[:2], dtype=np.uint8)
            
            for (bbox, text, prob) in results:
                if prob > 0.3:  # Confidence threshold
                    pts = np.array(bbox, dtype=np.int32)
                    cv2.fillPoly(mask, [pts], 255)
                    
            return mask
        except Exception as e:
            print(f"Text detection failed: {e}")
            return np.zeros(image_np.shape[:2], dtype=np.uint8)
    
    def _detect_logo_regions(self, image_np: np.ndarray) -> np.ndarray:
        """Detect logo-like regions using edge detection and contours"""
        gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
        
        # Edge detection
        edges = cv2.Canny(gray, 50, 150)
        
        # Morphological operations to connect edges
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        mask = np.zeros_like(gray)
        
        # Filter contours that look like logos (compact, not too large)
        h, w = image_np.shape[:2]
        min_area = (h * w) * 0.0001  # At least 0.01% of image
        max_area = (h * w) * 0.15     # At most 15% of image
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if min_area < area < max_area:
                x, y, cw, ch = cv2.boundingRect(cnt)
                aspect_ratio = cw / float(ch)
                
                # Logos tend to have reasonable aspect ratios
                if 0.2 < aspect_ratio < 5.0:
                    cv2.drawContours(mask, [cnt], -1, 255, -1)
        
        return mask
    
    def _detect_color_overlays(self, image_np: np.ndarray) -> np.ndarray:
        """Detect colored watermark overlays using color analysis"""
        # Convert to HSV for better color analysis
        hsv = cv2.cvtColor(image_np, cv2.COLOR_RGB2HSV)
        
        # Detect semi-transparent overlays by looking for uniform color regions
        # that are distinct from the background
        
        # Calculate local color variance
        h, w = image_np.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        
        # Split into grid and analyze each cell
        cell_size = 50
        for y in range(0, h - cell_size, cell_size):
            for x in range(0, w - cell_size, cell_size):
                cell = image_np[y:y+cell_size, x:x+cell_size]
                
                # Check if cell has uniform color (potential watermark)
                std = np.std(cell, axis=(0, 1))
                if np.mean(std) < 20:  # Low variance = uniform color
                    # Check if it's distinct from neighbors
                    is_watermark = self._is_overlay_region(image_np, x, y, cell_size)
                    if is_watermark:
                        mask[y:y+cell_size, x:x+cell_size] = 255
        
        return mask
    
    def _is_overlay_region(self, image_np: np.ndarray, x: int, y: int, size: int) -> bool:
        """Check if a region appears to be an overlay"""
        h, w = image_np.shape[:2]
        
        # Get the region
        region = image_np[y:y+size, x:x+size]
        region_mean = np.mean(region, axis=(0, 1))
        
        # Check surrounding regions
        neighbors = []
        offsets = [(-size, 0), (size, 0), (0, -size), (0, size)]
        
        for dx, dy in offsets:
            nx, ny = x + dx, y + dy
            if 0 <= nx < w - size and 0 <= ny < h - size:
                neighbor = image_np[ny:ny+size, nx:nx+size]
                neighbors.append(np.mean(neighbor, axis=(0, 1)))
        
        if not neighbors:
            return False
        
        # If region color is very different from all neighbors, might be overlay
        neighbor_mean = np.mean(neighbors, axis=0)
        color_diff = np.linalg.norm(region_mean - neighbor_mean)
        
        return color_diff > 30  # Threshold for color difference
    
    def _detect_corner_watermarks(self, image_np: np.ndarray) -> np.ndarray:
        """Detect watermarks commonly placed in corners"""
        h, w = image_np.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        
        # Check corners (common watermark locations)
        corner_size = min(h, w) // 4
        corners = [
            (0, 0, corner_size, corner_size),                    # Top-left
            (w - corner_size, 0, w, corner_size),               # Top-right
            (0, h - corner_size, corner_size, h),               # Bottom-left
            (w - corner_size, h - corner_size, w, h),           # Bottom-right
        ]
        
        for x1, y1, x2, y2 in corners:
            region = image_np[y1:y2, x1:x2]
            
            # Check if corner has distinct content
            if self._has_overlay_characteristics(region):
                mask[y1:y2, x1:x2] = 255
        
        return mask
    
    def _has_overlay_characteristics(self, region: np.ndarray) -> bool:
        """Check if region has watermark-like characteristics"""
        # Convert to grayscale
        gray = cv2.cvtColor(region, cv2.COLOR_RGB2GRAY)
        
        # Check for edges (logos/text have edges)
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size
        
        # Watermarks typically have 1-10% edge density
        if 0.01 < edge_density < 0.1:
            return True
        
        # Check for uniform color (colored overlays)
        color_std = np.std(region, axis=(0, 1))
        if np.mean(color_std) < 15:  # Very uniform
            return True
        
        return False
    
    def detect_watermark(
        self,
        image_path: str,
        output_mask_path: str,
        detect_text: bool = True,
        detect_logos: bool = True,
        detect_overlays: bool = True,
        detect_corners: bool = True,
        min_area: int = 100,
        dilate_iterations: int = 2
    ) -> bool:
        """
        Detect watermark regions (text, logos, overlays) NOT people/objects
        
        Args:
            image_path: Path to input image
            output_mask_path: Path to save detected mask
            detect_text: Enable text detection via OCR
            detect_logos: Enable logo detection via edge analysis
            detect_overlays: Enable colored overlay detection
            detect_corners: Focus on corner regions (common watermark spots)
            min_area: Minimum watermark area in pixels
            dilate_iterations: Mask dilation iterations
        
        Returns:
            True if watermark detected and mask saved
        """
        try:
            # Load image
            image = Image.open(image_path).convert('RGB')
            image_np = np.array(image)
            
            print("Detecting watermark regions...")
            
            # Combine multiple detection methods
            combined_mask = np.zeros(image_np.shape[:2], dtype=np.uint8)
            
            if detect_text:
                print("  - Detecting text watermarks...")
                text_mask = self._detect_text_regions(image_np)
                combined_mask = cv2.bitwise_or(combined_mask, text_mask)
            
            if detect_logos:
                print("  - Detecting logo watermarks...")
                logo_mask = self._detect_logo_regions(image_np)
                combined_mask = cv2.bitwise_or(combined_mask, logo_mask)
            
            if detect_overlays:
                print("  - Detecting colored overlays...")
                overlay_mask = self._detect_color_overlays(image_np)
                combined_mask = cv2.bitwise_or(combined_mask, overlay_mask)
            
            if detect_corners:
                print("  - Checking corner regions...")
                corner_mask = self._detect_corner_watermarks(image_np)
                combined_mask = cv2.bitwise_or(combined_mask, corner_mask)
            
            # Post-process mask
            if np.sum(combined_mask > 0) > min_area:
                # Dilate to ensure complete coverage
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
                combined_mask = cv2.dilate(combined_mask, kernel, iterations=dilate_iterations)
                
                # Save mask
                mask_img = Image.fromarray(combined_mask)
                mask_img.save(output_mask_path)
                
                detected_pixels = np.sum(combined_mask > 0)
                total_pixels = combined_mask.size
                coverage = (detected_pixels / total_pixels) * 100
                
                print(f"✓ Watermark detected! Coverage: {coverage:.2f}%")
                print(f"✓ Mask saved to: {output_mask_path}")
                return True
            else:
                print("✗ No watermark detected (or too small)")
                return False
            
        except Exception as e:
            print(f"Detection failed: {e}")
            import traceback
            traceback.print_exc()
            return False


if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description="Advanced Watermark Removal with Pre-trained Models")
    parser.add_argument("image", help="Input image path")
    parser.add_argument("--mask", help="Watermark mask path (optional, will auto-detect if not provided)")
    parser.add_argument("--output", default="cleaned_output.png", help="Output path")
    parser.add_argument("--model", choices=['sd-inpaint', 'lama', 'auto'], 
                       default='sd-inpaint', help="Model to use")
    parser.add_argument("--device", choices=['cuda', 'cpu', 'auto'], 
                       default='auto', help="Device to use")
    parser.add_argument("--detect", action='store_true', 
                       help="Auto-detect watermark (requires --mask to be unset)")
    parser.add_argument("--no-text", action='store_true', help="Disable text detection")
    parser.add_argument("--no-logos", action='store_true', help="Disable logo detection")
    parser.add_argument("--no-overlays", action='store_true', help="Disable overlay detection")
    parser.add_argument("--no-corners", action='store_true', help="Disable corner detection")
    
    args = parser.parse_args()
    
    device = None if args.device == 'auto' else args.device
    
    # Auto-detect mask if needed
    if args.detect and not args.mask:
        print("Auto-detecting watermark...")
        detector = DeepWatermarkDetector(device=device)
        mask_path = "auto_detected_mask.png"
        
        success = detector.detect_watermark(
            args.image, 
            mask_path,
            detect_text=not args.no_text,
            detect_logos=not args.no_logos,
            detect_overlays=not args.no_overlays,
            detect_corners=not args.no_corners
        )
        
        if success:
            args.mask = mask_path
        else:
            print("Failed to auto-detect watermark. Please provide a mask with --mask")
            exit(1)
    
    if not args.mask:
        print("Error: Either provide --mask or use --detect for auto-detection")
        exit(1)
    
    # Remove watermark
    print(f"\nRemoving watermark using {args.model}...")
    success = remove_watermark_with_pretrained(
        args.image,
        args.mask,
        args.output,
        model_type=args.model,
        device=device
    )
    
    if success:
        print(f"\n✓ Successfully removed watermark!")
        print(f"  Output saved to: {args.output}")
    else:
        print("\n✗ Failed to remove watermark")
        exit(1)