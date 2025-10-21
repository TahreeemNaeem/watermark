import sys
import os
import shutil

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

print("Using alternative watermark processing approach...")
print("The original FluxKontextPipeline approach is not compatible with current library versions.")
print("Using the improved watermark removal system instead...")

# Run the watermark detection and removal using the available tools
from improved_watermark_removal import WatermarkDetector, WatermarkRemover, ModelDownloader

def main():
    # Check if models are available
    if not ModelDownloader.ensure_models():
        print("Required models not available. Please download them first.")
        return
    
    # Set up detector and remover
    detector = WatermarkDetector()
    remover = WatermarkRemover()
    
    input_image_path = "image.png"
    mask_output_path = "detection_mask.png"
    output_image_path = "output.png"
    
    # Try to detect watermarks in the image
    success = detector.detect_watermark(
        input_image_path,
        mask_output_path,
        detection_mode='smart',
        refinement_method='grabcut',
        confidence_threshold=0.3,
        visualize=True
    )
    
    if success:
        print("Watermark detected. Removing...")
        # Remove the watermark using the detected mask
        try:
            removal_success = remover.remove(input_image_path, mask_output_path, output_image_path, method='auto')
            if removal_success:
                print(f"Watermark removal completed. Output saved to {output_image_path}")
            else:
                print("Watermark removal failed, copying original image to output.")
                shutil.copy(input_image_path, output_image_path)
        except Exception as e:
            print(f"Watermark removal failed with error: {e}")
            print("Copying original image to output.")
            shutil.copy(input_image_path, output_image_path)
    else:
        print("No watermarks detected in the image. Copying original image to output.")
        shutil.copy(input_image_path, output_image_path)

if __name__ == "__main__":
    main()
