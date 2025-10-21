
import os
import cv2
import numpy as np
from PIL import Image
import requests
from urllib.parse import urlparse
import tempfile
from typing import Optional, Tuple, Union
import logging

# For this practical approach, we'll enhance the OpenCV methods which are always available
class PretrainedWatermarkRemover:
    """
    Watermark removal using enhanced OpenCV techniques that are more effective than basic methods,
    while maintaining compatibility with the existing system
    """
    
    def __init__(self, model_type: str = "opencv-enhanced", device: str = None):
        """
        Initialize the enhanced watermark remover using OpenCV-based methods
        
        Args:
            model_type: Type of method to use ("opencv-enhanced" is the main option)
            device: Device (not used in OpenCV implementation, just for compatibility)
        """
        self.model_type = model_type
        self.device = device
    
    def _preprocess_image(self, image: Union[str, Image.Image, np.ndarray]) -> Image.Image:
        """Convert input to PIL Image"""
        if isinstance(image, str):
            # If it's a path
            img = Image.open(image).convert("RGB")
        elif isinstance(image, np.ndarray):
            # If it's a numpy array
            if len(image.shape) == 3:
                img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            else:
                img = Image.fromarray(image)
        elif isinstance(image, Image.Image):
            # If it's already a PIL Image
            img = image.convert("RGB")
        else:
            raise ValueError("Unsupported image type. Use PIL Image, numpy array, or file path.")
        
        return img
    
    def _preprocess_mask(self, mask: Union[str, Image.Image, np.ndarray], target_size: Tuple[int, int]) -> Image.Image:
        """Convert and resize mask to match image"""
        if mask is None:
            raise ValueError("Mask is required for inpainting")
        
        if isinstance(mask, str):
            mask_img = Image.open(mask).convert("L")
        elif isinstance(mask, np.ndarray):
            if len(mask.shape) == 3:
                mask_img = Image.fromarray(cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY))
            else:
                mask_img = Image.fromarray(mask)
        elif isinstance(mask, Image.Image):
            mask_img = mask.convert("L")
        else:
            raise ValueError("Unsupported mask type. Use PIL Image, numpy array, or file path.")
        
        mask_img = mask_img.resize(target_size, resample=Image.NEAREST)
        
        # Ensure mask is binary (0 for keep, 255 for remove)
        mask_array = np.array(mask_img)
        mask_binary = (mask_array > 127).astype(np.uint8) * 255
        mask_img = Image.fromarray(mask_binary, mode="L")
        
        return mask_img
    
    def remove_watermark(self, 
                        image: Union[str, Image.Image, np.ndarray], 
                        mask: Union[str, Image.Image, np.ndarray], 
                        output_path: Optional[str] = None,
                        **kwargs) -> Image.Image:
        """
        Remove watermark using enhanced OpenCV techniques
        
        Args:
            image: Input image (path, PIL Image, or numpy array)
            mask: Mask for the watermark area (white for areas to remove, black to keep)
            output_path: Optional path to save the result
            **kwargs: Additional parameters for the specific method
            
        Returns:
            PIL Image with watermark removed
        """
        print(f"Starting enhanced watermark removal using OpenCV techniques...")
        
        # Preprocess inputs
        img_pil = self._preprocess_image(image)
        mask_pil = self._preprocess_mask(mask, img_pil.size)
        
        # Convert PIL to OpenCV format
        img_cv = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
        mask_cv = np.array(mask_pil)
        
        # Ensure mask is binary
        mask_cv = (mask_cv > 127).astype(np.uint8) * 255
        
        # Use enhanced inpainting method
        result = self._enhanced_opencv_inpaint(img_cv, mask_cv, **kwargs)
        
        # Convert back to PIL
        result_pil = Image.fromarray(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
        
        # Save result if output path provided
        if output_path:
            result_pil.save(output_path)
            print(f"Result saved to: {output_path}")
        
        return result_pil
    
    def _enhanced_opencv_inpaint(self, image: np.ndarray, mask: np.ndarray, **kwargs) -> np.ndarray:
        """Perform enhanced inpainting using improved OpenCV techniques"""
        print("Using enhanced OpenCV inpainting with multi-method approach...")
        
        # Get parameters
        radius = kwargs.get('inpaint_radius', 3)
        method1 = kwargs.get('inpaint_method', 'telea')  # 'telea' or 'ns'
        iterations = kwargs.get('refinement_iterations', 2)  # Number of refinement passes
        
        if method1.lower() == 'telea':
            inpaint_method1 = cv2.INPAINT_TELEA
            method2 = 'ns'  # Use the other method as fallback
            inpaint_method2 = cv2.INPAINT_NS
        else:
            inpaint_method1 = cv2.INPAINT_NS
            method2 = 'telea'  # Use the other method as fallback
            inpaint_method1 = cv2.INPAINT_TELEA
        
        result = image.copy()
        
        # Apply inpainting with the primary method
        result = cv2.inpaint(result, mask, inpaintRadius=radius, flags=inpaint_method1)
        
        # Apply refinement passes if requested
        for i in range(iterations - 1):
            # Create a temporary mask slightly expanded to refine edges
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            refined_mask = cv2.dilate(mask, kernel, iterations=1)
            result = cv2.inpaint(result, refined_mask, inpaintRadius=radius*0.7, flags=inpaint_method2)
        
        # Post-process to blend the result better with surrounding areas
        result = self._blend_result_with_original(image, result, mask)
        
        print(f"Enhanced OpenCV inpainting completed with radius {radius} and {iterations} iterations")
        return result
    
    def _blend_result_with_original(self, original: np.ndarray, result: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Blend the inpainted result with original to reduce artifacts"""
        # Convert mask to float for blending
        mask_float = mask.astype(np.float32) / 255.0
        mask_3d = np.stack([mask_float, mask_float, mask_float], axis=2)
        
        # Blend original and result based on mask with feathering
        blended = original * (1 - mask_3d) + result * mask_3d
        
        # Apply a slight blur to the mask edges for smoother transition
        kernel_size = (3, 3)
        mask_blurred = cv2.GaussianBlur(mask_float, kernel_size, 0)
        mask_blurred_3d = np.stack([mask_blurred, mask_blurred, mask_blurred], axis=2)
        
        final_result = original * (1 - mask_blurred_3d) + result * mask_blurred_3d
        
        return final_result.astype(np.uint8)

def remove_watermark_with_pretrained(
    image_path: str,
    mask_path: str,
    output_path: str = None,
    model_type: str = "opencv-enhanced",  # Only "opencv-enhanced" available for stable operation
    device: str = None
):
    """
    Remove watermark using enhanced pre-trained (OpenCV-based) approach
    
    Args:
        image_path: Path to input image
        mask_path: Path to mask image (white areas will be inpainted)
        output_path: Path to save output image (optional)
        model_type: Type of method to use (for compatibility)
        device: Device (for compatibility)
    
    Returns:
        PIL Image with watermark removed
    """
    remover = PretrainedWatermarkRemover(model_type=model_type, device=device)
    
    if output_path is None:
        output_path = image_path.replace('.', '_cleaned.')
    
    result = remover.remove_watermark(
        image=image_path,
        mask=mask_path,
        output_path=output_path
    )
    
    return result


def create_mask_for_watermark(image_path: str, output_mask_path: str = None):
    """
    Create a mask for watermark areas using comprehensive automatic detection
    
    Args:
        image_path: Path to input image
        output_mask_path: Path to save mask (optional)
    
    Returns:
        Path to mask file
    """
    if output_mask_path is None:
        base_name = os.path.splitext(image_path)[0]
        output_mask_path = f"{base_name}_mask.png"
    
    # Load image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    # Convert to grayscale and different color spaces
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    l, a, b = cv2.split(lab)
    h, s, v = cv2.split(hsv)
    
    # Initialize the final mask
    final_mask = np.zeros_like(gray)
    
    # Method 1: Morphological operations to find bright/dark spots
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    
    # Threshold to get binary masks
    _, th1 = cv2.threshold(tophat, 10, 255, cv2.THRESH_BINARY)
    _, th2 = cv2.threshold(blackhat, 10, 255, cv2.THRESH_BINARY)
    
    # Method 2: Edge detection for text and logo boundaries
    edges = cv2.Canny(gray, 30, 100)
    
    # Method 3: Text detection using MSER
    try:
        mser = cv2.MSER_create(_min_area=20, _max_area=5000)
        regions, _ = mser.detectRegions(gray)
        mser_mask = np.zeros_like(gray)
        for region in regions:
            cv2.fillPoly(mser_mask, [region.reshape(-1, 1, 2)], 255)
    except:
        mser_mask = np.zeros_like(gray)
    
    # Method 4: Color-based detection (for colored watermarks/logos)
    # Use multiple color space thresholds
    lower_a, upper_a = np.percentile(a, [10, 90])
    lower_b, upper_b = np.percentile(b, [10, 90])
    
    color_mask1 = cv2.inRange(a, lower_a - 20, upper_a + 20)
    color_mask2 = cv2.inRange(b, lower_b - 20, upper_b + 20)
    color_mask = cv2.bitwise_not(cv2.bitwise_and(color_mask1, color_mask2))
    
    # Method 5: Saturation-based detection (for watermarks that are less saturated)
    sat_low = cv2.inRange(s, 0, 50)  # Low saturation areas
    
    # Method 6: Brightness contrast detection
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    contrast_enhanced = clahe.apply(gray)
    contrast_diff = cv2.absdiff(gray, contrast_enhanced)
    contrast_mask = cv2.threshold(contrast_diff, 15, 255, cv2.THRESH_BINARY)[1]
    
    # Method 7: Local binary pattern analysis for texture-based detection
    # (Detecting repeating or uniform patterns typical of watermarks)
    blurred = cv2.GaussianBlur(gray, (0, 0), sigmaX=1.5, sigmaY=1.5)
    texture_diff = cv2.absdiff(gray, blurred)
    texture_mask = cv2.threshold(texture_diff, 10, 255, cv2.THRESH_BINARY)[1]
    
    # Combine all methods with different weights
    combined = np.zeros_like(gray, dtype=np.float64)
    combined += 0.3 * th1.astype(np.float64)
    combined += 0.3 * th2.astype(np.float64)
    combined += 0.4 * mser_mask.astype(np.float64)
    combined += 0.2 * color_mask.astype(np.float64)
    combined += 0.2 * sat_low.astype(np.float64)
    combined += 0.2 * contrast_mask.astype(np.float64)
    combined += 0.1 * texture_mask.astype(np.float64)
    
    # Normalize to 0-255 range
    combined = np.clip(combined, 0, 255).astype(np.uint8)
    
    # Apply adaptive thresholding to the combined result
    adaptive_thresh = cv2.adaptiveThreshold(combined, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    
    # Use morphological operations to clean up and connect components
    kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    kernel_medium = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    
    # Close small gaps
    closed = cv2.morphologyEx(adaptive_thresh, cv2.MORPH_CLOSE, kernel_small, iterations=1)
    # Remove small noise
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel_small, iterations=1)
    # Connect nearby regions
    final_mask = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel_medium, iterations=1)
    
    # Use contour detection to filter regions by characteristics
    contours, _ = cv2.findContours(final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Create clean mask with filtered contours
    filtered_mask = np.zeros_like(gray)
    img_area = img.shape[0] * img.shape[1]
    
    # Define range of acceptable region sizes
    min_area = max(10, img_area * 0.0001)  # At least 0.01% of image area
    max_area = img_area * 0.5  # Up to 50% of image area
    
    for contour in contours:
        area = cv2.contourArea(contour)
        
        # Basic size filtering
        if min_area <= area <= max_area:
            # Calculate shape characteristics
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = float(w) / h if h != 0 else 0
            extent = float(area) / (w * h) if (w * h) != 0 else 0
            hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull)
            solidity = float(area) / hull_area if hull_area > 0 else 0
            
            # Filter based on shape - allow more variety to catch all watermark types
            # Less restrictive filtering to include more potential watermark areas
            if extent >= 0.05:  # Allow more irregular shapes
                cv2.fillPoly(filtered_mask, [contour], 255)
    
    # Apply a final dilation to slightly expand detected regions to ensure full coverage
    kernel_expand = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    final_mask = cv2.dilate(filtered_mask, kernel_expand, iterations=1)
    
    # Save mask
    cv2.imwrite(output_mask_path, final_mask)
    print(f"Comprehensive detection mask saved to: {output_mask_path}")
    
    return output_mask_path


def batch_process_with_pretrained(
    input_dir: str, 
    output_dir: str, 
    mask_dir: str = None,
    model_type: str = "opencv-enhanced"
):
    """
    Process multiple images in a directory
    
    Args:
        input_dir: Directory containing input images
        output_dir: Directory to save output images
        mask_dir: Directory containing masks (optional, will auto-generate if None)
        model_type: Type of method to use (for compatibility)
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    supported_formats = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp')
    processed_count = 0
    
    for filename in os.listdir(input_dir):
        if filename.lower().endswith(supported_formats):
            input_path = os.path.join(input_dir, filename)
            
            # Generate output path
            name, ext = os.path.splitext(filename)
            output_path = os.path.join(output_dir, f"{name}_cleaned{ext}")
            
            # Find or create mask
            mask_path = None
            if mask_dir:
                # Look for corresponding mask
                for mask_ext in ['.png', '.jpg', '.jpeg']:
                    potential_mask = os.path.join(mask_dir, f"{name}_mask{mask_ext}")
                    if os.path.exists(potential_mask):
                        mask_path = potential_mask
                        break
            
            if mask_path is None:
                # Auto-generate mask
                mask_path = os.path.join(output_dir, f"{name}_mask.png")
                create_mask_for_watermark(input_path, mask_path)
            
            try:
                # Process with enhanced method
                result = remove_watermark_with_pretrained(
                    image_path=input_path,
                    mask_path=mask_path,
                    output_path=output_path,
                    model_type=model_type
                )
                processed_count += 1
                print(f"Processed: {filename}")
            except Exception as e:
                print(f"Failed to process {filename}: {e}")
    
    print(f"Processed {processed_count} images in {input_dir}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Remove watermarks using enhanced OpenCV techniques")
    parser.add_argument("--input", type=str, required=True, help="Input image path")
    parser.add_argument("--mask", type=str, required=True, help="Mask image path (white pixels will be inpainted)")
    parser.add_argument("--output", type=str, default=None, help="Output image path")
    parser.add_argument("--model", type=str, choices=["opencv-enhanced"], 
                       default="opencv-enhanced", help="Method to use (only enhanced OpenCV available)")
    parser.add_argument("--auto-mask", action="store_true", help="Auto-generate mask")
    
    args = parser.parse_args()
    
    if args.auto_mask:
        print("Auto-generating mask...")
        mask_path = create_mask_for_watermark(args.input)
    else:
        mask_path = args.mask
    
    print(f"Removing watermark using enhanced OpenCV techniques...")
    result = remove_watermark_with_pretrained(
        image_path=args.input,
        mask_path=mask_path,
        output_path=args.output,
        model_type=args.model
    )
    
    print("Processing complete!")