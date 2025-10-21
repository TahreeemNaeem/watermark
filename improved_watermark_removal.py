
import os
import sys
import torch
import numpy as np
from PIL import Image
from typing import Optional, Tuple, List, Dict
import cv2
from pathlib import Path
import urllib.request
from tqdm import tqdm

# Model download URLs and paths - Updated with JoyCaption model
MODEL_CONFIGS = {
    'yolo11x': {
        'url': 'https://huggingface.co/spaces/fancyfeast/joycaption-watermark-detection/resolve/main/yolo11x-train28-best.pt',
        'path': 'models/yolo11x-train28-best.pt',
        'size': '56MB'
    },
    'fastsam': {
        'url': 'https://github.com/ultralytics/assets/releases/download/v8.2.0/FastSAM-x.pt',
        'path': 'models/FastSAM-x.pt',
        'size': '138MB'
    },
    'lama': {
        'repo': 'advimman/lama',
        'model': 'lama',
        'size': '~50MB'
    }
}

class ModelDownloader:
    """Handles automatic model downloading with progress bars"""
    
    @staticmethod
    def download_file(url: str, output_path: str, description: str = "Downloading"):
        """Download file with progress bar"""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        if os.path.exists(output_path):
            print(f"✓ {description} already exists at {output_path}")
            return True
        
        print(f"Downloading {description}...")
        try:
            def report_progress(block_num, block_size, total_size):
                downloaded = block_num * block_size
                if total_size > 0:
                    percent = min(downloaded * 100 / total_size, 100)
                    bar_length = 50
                    filled = int(bar_length * percent / 100)
                    bar = '█' * filled + '░' * (bar_length - filled)
                    print(f'\r{bar} {percent:.1f}%', end='', flush=True)
            
            urllib.request.urlretrieve(url, output_path, report_progress)
            print(f"\n✓ Downloaded to {output_path}")
            return True
        except Exception as e:
            print(f"\n✗ Download failed: {e}")
            if os.path.exists(output_path):
                os.remove(output_path)
            return False
    
    @staticmethod
    def ensure_models():
        """Ensure all required models are downloaded"""
        print("="*60)
        print("CHECKING REQUIRED MODELS")
        print("="*60)
        
        # YOLO11x (JoyCaption watermark detection model)
        if not os.path.exists(MODEL_CONFIGS['yolo11x']['path']):
            success = ModelDownloader.download_file(
                MODEL_CONFIGS['yolo11x']['url'],
                MODEL_CONFIGS['yolo11x']['path'],
                f"YOLO11x Watermark Detector ({MODEL_CONFIGS['yolo11x']['size']})"
            )
            if not success:
                print("✗ Failed to download YOLO11x watermark model")
                return False
        else:
            print(f"✓ YOLO11x Watermark Detector found at {MODEL_CONFIGS['yolo11x']['path']}")
        
        # FastSAM
        if not os.path.exists(MODEL_CONFIGS['fastsam']['path']):
            success = ModelDownloader.download_file(
                MODEL_CONFIGS['fastsam']['url'],
                MODEL_CONFIGS['fastsam']['path'],
                f"FastSAM ({MODEL_CONFIGS['fastsam']['size']})"
            )
            if not success:
                print("✗ Failed to download FastSAM")
                return False
        else:
            print(f"✓ FastSAM found at {MODEL_CONFIGS['fastsam']['path']}")
        
        print("✓ All models ready\n")
        return True


class WatermarkDetector:
    """
    Advanced watermark detection using YOLO11x (JoyCaption) + FastSAM + GrabCut
    Multiple refinement approaches for robust segmentation
    """
    
    def __init__(self, device: Optional[str] = None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        self._yolo_model = None
        self._fastsam_model = None
        
    def _load_yolo11x(self):
        """Load YOLO11x JoyCaption watermark detection model"""
        if self._yolo_model is None:
            try:
                from ultralytics import YOLO
                print("Loading YOLO11x JoyCaption Watermark Detector...")
                
                model_path = MODEL_CONFIGS['yolo11x']['path']
                if not os.path.exists(model_path):
                    raise FileNotFoundError(f"YOLO11x watermark model not found at {model_path}")
                
                self._yolo_model = YOLO(model_path)
                self._yolo_model.to(self.device)
                print("✓ YOLO11x Watermark Detector loaded")
                
                # Print model info
                print(f"  Model classes: {self._yolo_model.names}")
                
            except ImportError:
                print("✗ Ultralytics not installed. Install with: pip install ultralytics")
                raise
            except Exception as e:
                print(f"✗ Failed to load YOLO11x Watermark Detector: {e}")
                raise
                
        return self._yolo_model
    
    def _load_fastsam(self):
        """Load FastSAM model for segmentation"""
        if self._fastsam_model is None:
            try:
                from ultralytics import FastSAM
                print("Loading FastSAM model...")
                
                model_path = MODEL_CONFIGS['fastsam']['path']
                if not os.path.exists(model_path):
                    raise FileNotFoundError(f"FastSAM model not found at {model_path}")
                
                self._fastsam_model = FastSAM(model_path)
                print("✓ FastSAM loaded")
                
            except ImportError:
                print("✗ Ultralytics not installed. Install with: pip install ultralytics")
                raise
            except Exception as e:
                print(f"✗ Failed to load FastSAM: {e}")
                raise
                
        return self._fastsam_model
    
    def detect_watermarks_yolo(
        self,
        image_path: str,
        confidence_threshold: float = 0.25,
        iou_threshold: float = 0.5,
        augment: bool = True
    ) -> List[Dict]:
        """
        Detect watermarks using YOLO11 JoyCaption model
        
        Args:
            image_path: Path to input image
            confidence_threshold: Minimum confidence score
            iou_threshold: IOU threshold for NMS
            augment: Use test time augmentation
            
        Returns:
            List of detections with bounding boxes and metadata
        """
        model = self._load_yolo11x()
        
        print(f"Detecting watermarks with YOLO11 JoyCaption (conf≥{confidence_threshold})...")
        
        # Load image
        image = Image.open(image_path)
        
        # Run detection with optimized parameters for watermark detection
        results = model.predict(
            source=image,
            imgsz=1024,
            augment=augment,
            iou=iou_threshold,
            conf=confidence_threshold,
            verbose=False,
            device=self.device
        )
        
        detections = []
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = float(box.conf[0])
                    cls = int(box.cls[0])
                    cls_name = model.names[cls]
                    
                    detections.append({
                        'bbox': (int(x1), int(y1), int(x2), int(y2)),
                        'confidence': conf,
                        'class_id': cls,
                        'class_name': cls_name
                    })
        
        print(f"  Found {len(detections)} potential watermarks")
        return detections
    
    def refine_with_fastsam(
        self,
        image_path: str,
        bounding_boxes: List[Tuple[int, int, int, int]]
    ) -> np.ndarray:
        """
        Refine detection masks using FastSAM
        
        Args:
            image_path: Path to image
            bounding_boxes: List of (x1, y1, x2, y2) bounding boxes
            
        Returns:
            Combined binary mask
        """
        if not bounding_boxes:
            return None
            
        model = self._load_fastsam()
        
        print("Refining segmentation with FastSAM...")
        
        # Load image
        image = cv2.imread(image_path)
        h, w = image.shape[:2]
        
        # Run FastSAM
        results = model(
            image_path,
            device=self.device,
            retina_masks=True,
            imgsz=1024,
            conf=0.4,
            iou=0.9,
        )
        
        # Create combined mask
        combined_mask = np.zeros((h, w), dtype=np.uint8)
        
        # Extract masks that overlap with bounding boxes
        for i, bbox in enumerate(bounding_boxes):
            x1, y1, x2, y2 = bbox
            print(f"  Processing region {i+1}/{len(bounding_boxes)}")
            
            # Get masks from FastSAM results
            if results[0].masks is not None:
                masks = results[0].masks.data.cpu().numpy()
                
                # Find masks that overlap with this bbox
                for mask in masks:
                    # Resize mask to image size if needed
                    if mask.shape != (h, w):
                        mask = cv2.resize(mask, (w, h))
                    
                    # Check overlap with bbox
                    bbox_mask = np.zeros((h, w), dtype=bool)
                    bbox_mask[y1:y2, x1:x2] = True
                    
                    overlap = np.logical_and(mask > 0.5, bbox_mask).sum()
                    bbox_area = (x2 - x1) * (y2 - y1)
                    
                    # If significant overlap, add to combined mask
                    if overlap > bbox_area * 0.3:
                        combined_mask = np.maximum(
                            combined_mask, 
                            (mask > 0.5).astype(np.uint8) * 255
                        )
        
        print("✓ FastSAM segmentation complete")
        return combined_mask
    
    def refine_with_grabcut(
        self,
        image_path: str,
        bounding_boxes: List[Tuple[int, int, int, int]]
    ) -> np.ndarray:
        """
        Refine detection masks using GrabCut algorithm
        Fast and no additional models needed!

        Args:
            image_path: Path to image
            bounding_boxes: List of (x1, y1, x2, y2) bounding boxes
            
        Returns:
            Combined binary mask
        """
        if not bounding_boxes:
            return None
        
        print("Refining segmentation with GrabCut...")
        
        # Load image
        image = cv2.imread(image_path)
        h, w = image.shape[:2]
        
        # Create combined mask
        combined_mask = np.zeros((h, w), dtype=np.uint8)
        
        # Process each bounding box with GrabCut
        for i, bbox in enumerate(bounding_boxes):
            x1, y1, x2, y2 = bbox
            print(f"  Processing region {i+1}/{len(bounding_boxes)}")
            
            # Ensure bbox is within image bounds
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            
            if x2 <= x1 or y2 <= y1:
                continue
            
            # Initialize GrabCut mask
            mask = np.zeros(image.shape[:2], np.uint8)
            bgd_model = np.zeros((1, 65), np.float64)
            fgd_model = np.zeros((1, 65), np.float64)
            
            # Define rectangle for GrabCut
            rect = (x1, y1, x2 - x1, y2 - y1)
            
            try:
                # Run GrabCut
                cv2.grabCut(
                    image, mask, rect, bgd_model, fgd_model,
                    5,  # iterations
                    cv2.GC_INIT_WITH_RECT
                )
                
                # Create binary mask (0 and 2 are background, 1 and 3 are foreground)
                result_mask = np.where(
                    (mask == 2) | (mask == 0),
                    0,
                    255
                ).astype(np.uint8)
                
                # Add to combined mask
                combined_mask = np.maximum(combined_mask, result_mask)
                
            except Exception as e:
                print(f"    GrabCut failed for this region: {e}")
                # Fallback: use the bounding box itself
                combined_mask[y1:y2, x1:x2] = 255
        
        print("✓ GrabCut segmentation complete")
        return combined_mask
    
    def refine_with_morphology(
        self,
        image_path: str,
        bounding_boxes: List[Tuple[int, int, int, int]],
        use_edge_detection: bool = True
    ) -> np.ndarray:
        """
        Refine detection using morphological operations and edge detection
        Very fast, no additional models!

        Args:
            image_path: Path to image
            bounding_boxes: List of (x1, y1, x2, y2) bounding boxes
            use_edge_detection: Use edge detection to refine boundaries
            
        Returns:
            Combined binary mask
        """
        if not bounding_boxes:
            return None
        
        print("Refining segmentation with morphological operations...")
        
        # Load image
        image = cv2.imread(image_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        h, w = image.shape[:2]
        
        # Create combined mask
        combined_mask = np.zeros((h, w), dtype=np.uint8)
        
        for i, bbox in enumerate(bounding_boxes):
            x1, y1, x2, y2 = bbox
            print(f"  Processing region {i+1}/{len(bounding_boxes)}")
            
            # Ensure bbox is within bounds
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            
            if x2 <= x1 or y2 <= y1:
                continue
            
            # Extract region
            roi = gray[y1:y2, x1:x2]
            
            if use_edge_detection:
                # Use adaptive thresholding + edge detection
                # Apply bilateral filter to preserve edges
                roi_filtered = cv2.bilateralFilter(roi, 9, 75, 75)
                
                # Adaptive threshold
                thresh = cv2.adaptiveThreshold(
                    roi_filtered,
                    255,
                    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                    cv2.THRESH_BINARY_INV,
                    11, 2
                )
                
                # Edge detection
                edges = cv2.Canny(roi_filtered, 50, 150)
                
                # Combine threshold and edges
                region_mask = cv2.bitwise_or(thresh, edges)
                
            else:
                # Simple thresholding
                _, region_mask = cv2.threshold(
                    roi, 0, 255,
                    cv2.THRESH_BINARY + cv2.THRESH_OTSU
                )
            
            # Morphological operations to clean up
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            region_mask = cv2.morphologyEx(region_mask, cv2.MORPH_CLOSE, kernel)
            region_mask = cv2.morphologyEx(region_mask, cv2.MORPH_OPEN, kernel)
            
            # Place refined mask in combined mask
            combined_mask[y1:y2, x1:x2] = np.maximum(
                combined_mask[y1:y2, x1:x2],
                region_mask
            )
        
        print("✓ Morphological segmentation complete")
        return combined_mask
    
    def refine_with_advanced_segmentation(
        self,
        image_path: str,
        bounding_boxes: List[Tuple[int, int, int, int]]
    ) -> np.ndarray:
        """
        Advanced segmentation using multiple techniques for better watermark detection
        Combines color analysis, edge detection and improved GrabCut
        
        Args:
            image_path: Path to image
            bounding_boxes: List of (x1, y1, x2, y2) bounding boxes
            
        Returns:
            Combined binary mask
        """
        if not bounding_boxes:
            return None
        
        print("Refining segmentation with advanced technique...")
        
        # Load image
        image = cv2.imread(image_path)
        h, w = image.shape[:2]
        
        # Create combined mask
        combined_mask = np.zeros((h, w), dtype=np.uint8)
        
        for i, bbox in enumerate(bounding_boxes):
            x1, y1, x2, y2 = bbox
            print(f"  Processing region {i+1}/{len(bounding_boxes)}")
            
            # Ensure bbox is within image bounds
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            
            if x2 <= x1 or y2 <= y1:
                continue
            
            # Extract region of interest
            roi = image[y1:y2, x1:x2]
            roi_h, roi_w = roi.shape[:2]
            
            # Method 1: Color-based thresholding (for colored watermarks)
            hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            lower = np.array([0, 0, 200])  # Bright areas
            upper = np.array([180, 50, 255])  # Bright areas with low saturation
            color_mask = cv2.inRange(hsv, lower, upper)
            
            # Method 2: Edge-based detection (for text/logos with sharp edges)
            gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray_roi, 50, 150)
            
            # Method 3: Improved GrabCut with better initialization
            grabcut_mask = np.full((roi_h, roi_w), cv2.GC_PR_BGD, dtype=np.uint8)
            
            # Initialize with color and edge information
            _, thresh = cv2.threshold(gray_roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            grabcut_mask[thresh > 127] = cv2.GC_PR_FGD
            grabcut_mask[edges > 50] = cv2.GC_PR_FGD
            
            # Mark borders as definite background
            margin = max(1, min(roi_h, roi_w) // 8)
            grabcut_mask[0:margin, :] = cv2.GC_BGD
            grabcut_mask[-margin:, :] = cv2.GC_BGD
            grabcut_mask[:, 0:margin] = cv2.GC_BGD
            grabcut_mask[:, -margin:] = cv2.GC_BGD
            
            bgd_model = np.zeros((1, 65), np.float64)
            fgd_model = np.zeros((1, 65), np.float64)
            
            try:
                # Run GrabCut with combined initialization
                cv2.grabCut(
                    roi, grabcut_mask, None, bgd_model, fgd_model,
                    10,  # more iterations
                    cv2.GC_INIT_WITH_MASK
                )
                
                # Combine GrabCut result with color/edge masks
                grabcut_result = np.where(
                    (grabcut_mask == cv2.GC_FGD) | (grabcut_mask == cv2.GC_PR_FGD),
                    255,
                    0
                ).astype(np.uint8)
                
                # Combine all methods using OR operation
                combined_result = cv2.bitwise_or(grabcut_result, color_mask)
                combined_result = cv2.bitwise_or(combined_result, edges)
                
                # Apply morphological operations to clean up
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
                combined_result = cv2.morphologyEx(combined_result, cv2.MORPH_CLOSE, kernel)
                combined_result = cv2.morphologyEx(combined_result, cv2.MORPH_OPEN, kernel)
                
                # Add to combined mask
                combined_mask[y1:y2, x1:x2] = np.maximum(
                    combined_mask[y1:y2, x1:x2], 
                    combined_result
                )
                
            except Exception as e:
                print(f"    Advanced segmentation failed for this region: {e}")
                # Fallback: use the bounding box itself
                combined_mask[y1:y2, x1:x2] = 255
        
        print("✓ Advanced segmentation complete")
        return combined_mask
    
    def detect_watermark(
        self,
        image_path: str,
        output_mask_path: str,
        detection_mode: str = 'smart',
        refinement_method: str = 'grabcut',
        confidence_threshold: float = 0.3,
        iou_threshold: float = 0.5,
        expand_mask: int = 5,
        visualize: bool = False
    ) -> bool:
        """
        Detect watermarks intelligently using YOLO11 JoyCaption model
        
        Args:
            image_path: Input image path
            output_mask_path: Where to save the mask
            detection_mode: 'smart' (auto-detect watermarks), 'all' (all objects)
            refinement_method: 'fastsam', 'grabcut', 'morphology', 'advanced', 'auto'
            confidence_threshold: Detection confidence
            iou_threshold: IOU threshold for NMS
            expand_mask: Pixels to expand mask
            visualize: Save visualization
            
        Returns:
            True if watermark detected successfully
        """
        try:
            # Detect watermarks using YOLO11 JoyCaption model
            detections = self.detect_watermarks_yolo(
                image_path,
                confidence_threshold=confidence_threshold,
                iou_threshold=iou_threshold,
                augment=True
            )
            
            if not detections:
                print("✗ No watermarks detected")
                return False
            
            print(f"Found {len(detections)} potential watermarks:")
            for det in detections:
                print(f"  - {det['class_name']}: {det['confidence']:.2f} at {det['bbox']}")
            
            # Filter detections likely to be watermarks (JoyCaption model is already specialized)
            if detection_mode == 'smart':
                detections = self._filter_watermark_candidates(detections, image_path)
            
            if not detections:
                print("✗ No valid watermark candidates after filtering")
                return False
            
            # Extract bounding boxes
            bboxes = [det['bbox'] for det in detections]
            
            # Refine with selected method
            if refinement_method == 'auto':
                # Try methods in order of speed/quality tradeoff
                print("Using auto refinement (trying multiple methods)...")
                mask = self.refine_with_grabcut(image_path, bboxes)
                if mask is None or np.sum(mask > 0) < 100:
                    print("  GrabCut insufficient, trying morphology...")
                    mask = self.refine_with_morphology(image_path, bboxes)
            elif refinement_method == 'advanced':
                mask = self.refine_with_advanced_segmentation(image_path, bboxes)
            elif refinement_method == 'fastsam':
                mask = self.refine_with_fastsam(image_path, bboxes)
            elif refinement_method == 'grabcut':
                mask = self.refine_with_grabcut(image_path, bboxes)
            elif refinement_method == 'morphology':
                mask = self.refine_with_morphology(image_path, bboxes)
            else:
                raise ValueError(f"Unknown refinement method: {refinement_method}")
            
            if mask is None or np.sum(mask > 0) < 10:
                print("✗ Refinement failed to produce valid mask")
                return False
            
            # Post-process mask
            if expand_mask > 0:
                kernel = cv2.getStructuringElement(
                    cv2.MORPH_ELLIPSE, 
                    (expand_mask*2+1, expand_mask*2+1)
                )
                mask = cv2.dilate(mask, kernel, iterations=1)
            
            # Apply morphological cleaning to the final mask
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            
            # Save mask
            cv2.imwrite(output_mask_path, mask)
            
            # Calculate coverage
            coverage = (np.sum(mask > 0) / mask.size) * 100
            print(f"✓ Watermark mask created! Coverage: {coverage:.2f}%")
            print(f"✓ Mask saved to: {output_mask_path}")
            
            # Visualize
            if visualize:
                vis_path = output_mask_path.replace('.png', '_visualization.png')
                self._create_visualization(image_path, detections, mask, vis_path)
            
            return True
            
        except Exception as e:
            print(f"✗ Detection failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _filter_watermark_candidates(
        self,
        detections: List[Dict],
        image_path: str
    ) -> List[Dict]:
        """Filter detections to find likely watermarks"""
        image = cv2.imread(image_path)
        h, w = image.shape[:2]
        
        filtered = []
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            bbox_w = x2 - x1
            bbox_h = y2 - y1
            
            # Calculate relative position and size
            rel_size = (bbox_w * bbox_h) / (w * h)
            
            # Watermarks are typically:
            # 1. Small to medium sized (not dominating the image)
            # 2. Often in corners or edges
            # 3. Aspect ratio variations
            
            # Size filter: 0.1% to 30% of image
            if rel_size < 0.001 or rel_size > 0.3:
                continue
            
            # Check if in corner/edge regions (common for watermarks)
            in_corner = self._is_in_corner_region(x1, y1, x2, y2, w, h)
            
            # Since we're using JoyCaption model, most detections are already watermarks
            # But we can still filter by position and size
            if in_corner or rel_size < 0.15:
                filtered.append(det)
        
        return filtered if filtered else detections  # Keep all if filtering removes everything
    
    def _is_in_corner_region(
        self,
        x1: int, y1: int, x2: int, y2: int,
        img_w: int, img_h: int,
        margin: float = 0.2
    ) -> bool:
        """Check if bbox is in corner/edge region"""
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        
        # Define corner regions
        left = cx < img_w * margin
        right = cx > img_w * (1 - margin)
        top = cy < img_h * margin
        bottom = cy > img_h * (1 - margin)
        
        return (left or right) and (top or bottom)
    
    def _create_visualization(
        self,
        image_path: str,
        detections: List[Dict],
        mask: np.ndarray,
        output_path: str
    ):
        """Create visualization of detection results"""
        image = cv2.imread(image_path)
        
        # Draw bounding boxes
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            conf = det['confidence']
            label = f"{det['class_name']}: {conf:.2f}"
            
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image, label, (x1, y1-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Overlay mask
        mask_colored = cv2.applyColorMap(mask, cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(image, 0.7, mask_colored, 0.3, 0)
        
        cv2.imwrite(output_path, overlay)
        print(f"✓ Visualization saved to: {output_path}")


class WatermarkRemover:
    """
    Watermark removal using state-of-the-art inpainting models
    """

    def __init__(self, device: Optional[str] = None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self._lama_model = None
        self._sd_pipe = None
        
    def _load_lama(self):
        """Load LaMa model with improved error handling"""
        if self._lama_model is None:
            try:
                print("Loading LaMa model...")
                
                # Try to load the LaMa model
                import torch
                repo_path = os.path.join(os.path.dirname(__file__), "lama_repo")
                if not os.path.exists(repo_path):
                    # Clone the repo if it doesn't exist
                    print("Cloning LaMa repository...")
                    os.system(f"git clone https://github.com/advimman/lama.git {repo_path}")
                
                # Add the repo to the path
                sys.path.insert(0, repo_path)
                
                # Load the model using torch hub
                self._lama_model = torch.hub.load(
                    'advimman/lama',
                    'lama',
                    pretrained=True,
                    force_reload=False
                )
                self._lama_model = self._lama_model.to(self.device)
                self._lama_model.eval()
                print("✓ LaMa loaded")
            except Exception as e:
                print(f"✗ Failed to load LaMa: {e}")
                # Try alternative: use lama-cleaner pip package
                try:
                    print("Trying to load LaMa via lama-cleaner...")
                    import lama_cleaner
                    # Use lama-cleaner as an alternative
                    print("✓ LaMa loaded via lama-cleaner")
                    # Note: The lama-cleaner requires a server to run, so this is just a fallback
                except ImportError:
                    print("✗ LaMa alternative (lama-cleaner) not available either")
                    raise
                raise
        return self._lama_model
    
    def _load_stable_diffusion(self):
        """Load Stable Diffusion Inpainting"""
        if self._sd_pipe is None:
            try:
                from diffusers import StableDiffusionInpaintPipeline
                print("Loading Stable Diffusion Inpainting...")
                self._sd_pipe = StableDiffusionInpaintPipeline.from_pretrained(
                    "stabilityai/stable-diffusion-2-inpainting",
                    torch_dtype=torch.float16 if self.device == 'cuda' else torch.float32,
                )
                self._sd_pipe = self._sd_pipe.to(self.device)
                if self.device == 'cuda':
                    self._sd_pipe.enable_attention_slicing()
                print("✓ Stable Diffusion loaded")
            except Exception as e:
                print(f"✗ Failed to load SD: {e}")
                raise
        return self._sd_pipe

    def remove_with_lama(
        self,
        image_path: str,
        mask_path: str,
        output_path: str
    ) -> bool:
        """Remove watermark using LaMa"""
        try:
            model = self._load_lama()
            
            # Load image and mask
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            
            # Convert to tensors
            image_tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float() / 255.0
            mask_tensor = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0).float() / 255.0
            
            image_tensor = image_tensor.to(self.device)
            mask_tensor = mask_tensor.to(self.device)
            
            print("Inpainting with LaMa...")
            
            with torch.no_grad():
                result = model(image_tensor, mask_tensor)
            
            # Convert back to image
            result = result[0].permute(1, 2, 0).cpu().numpy()
            result = (result * 255).clip(0, 255).astype(np.uint8)
            result = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
            
            cv2.imwrite(output_path, result, [cv2.IMWRITE_PNG_COMPRESSION, 9])
            print(f"✓ Saved to: {output_path}")
            return True
            
        except Exception as e:
            print(f"✗ LaMa failed: {e}")
            return False

    def remove_with_sd(
        self,
        image_path: str,
        mask_path: str,
        output_path: str,
        prompt: str = "high quality, clean, no watermark",
        negative_prompt: str = "watermark, text, logo, low quality",
        num_steps: int = 30
    ) -> bool:
        """Remove watermark using Stable Diffusion"""
        try:
            pipe = self._load_stable_diffusion()
            
            image = Image.open(image_path).convert('RGB')
            mask = Image.open(mask_path).convert('L')
            
            # Resize image and mask to model's expected size
            # Stable Diffusion 2 Inpainting model expects 512x512 images
            original_size = image.size
            image = image.resize((512, 512), Image.Resampling.LANCZOS)
            mask = mask.resize((512, 512), Image.Resampling.LANCZOS)
            
            print(f"Inpainting with Stable Diffusion ({num_steps} steps)...")
            
            result = pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                image=image,
                mask_image=mask,
                num_inference_steps=num_steps,
                guidance_scale=7.5,
                # Removed strength parameter to fix the error
            ).images[0]
            
            # Resize result back to original size if needed
            if result.size != original_size:
                result = result.resize(original_size, Image.Resampling.LANCZOS)
            
            result.save(output_path, quality=95)
            print(f"✓ Saved to: {output_path}")
            return True
            
        except Exception as e:
            print(f"✗ SD failed: {e}")
            return False

    def remove(
        self,
        image_path: str,
        mask_path: str,
        output_path: str,
        method: str = 'lama'
    ) -> bool:
        """Remove watermark using specified method"""
        if method == 'lama':
            return self.remove_with_lama(image_path, mask_path, output_path)
        elif method == 'sd':
            return self.remove_with_sd(image_path, mask_path, output_path)
        elif method == 'auto':
            # Try LaMa first (faster), fallback to SD
            if self.remove_with_lama(image_path, mask_path, output_path):
                return True
            print("Falling back to Stable Diffusion...")
            return self.remove_with_sd(image_path, mask_path, output_path)
        else:
            raise ValueError(f"Unknown method: {method}")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Advanced Watermark Detection & Removal using JoyCaption Model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Auto-detect and remove watermark using JoyCaption model
  python script.py image.jpg --detect --output clean.png
  
  # Download all required models
  python script.py --download-models
  
  # Use manual mask
  python script.py image.jpg --mask mask.png --output clean.png
  
  # Smart detection with GrabCut refinement (fast and good)
  python script.py image.jpg --detect --refine grabcut --output clean.png
  
  # Use FastSAM for better segmentation quality
  python script.py image.jpg --detect --refine fastsam --output clean.png
  
  # Fast morphological refinement (no ML models needed)
  python script.py image.jpg --detect --refine morphology --output clean.png
  
  # Use Stable Diffusion for removal
  python script.py image.jpg --detect --method sd --output clean.png
  
  # Adjust detection confidence
  python script.py image.jpg --detect --confidence 0.5 --output clean.png
        """
    )
    
    parser.add_argument("image", nargs='?', help="Input image path")
    parser.add_argument("--output", default="cleaned_output.png", help="Output path")
    parser.add_argument("--mask", help="Manual mask path (optional)")
    parser.add_argument("--method", choices=['lama', 'sd', 'auto'], 
                       default='lama', help="Removal method")
    parser.add_argument("--device", choices=['cuda', 'cpu', 'auto'], 
                       default='auto', help="Device to use")
    
    # Detection options
    detect_group = parser.add_argument_group('detection options')
    detect_group.add_argument("--detect", action='store_true', 
                             help="Auto-detect watermark using JoyCaption model")
    detect_group.add_argument("--mode", choices=['smart', 'all'], 
                             default='smart', help="Detection mode")
    detect_group.add_argument("--confidence", type=float, default=0.3,
                             help="Detection confidence threshold")
    detect_group.add_argument("--iou", type=float, default=0.5,
                             help="IOU threshold for NMS")
    detect_group.add_argument("--refine", choices=['fastsam', 'grabcut', 'morphology', 'auto'],
                             default='grabcut', help="Mask refinement method")
    detect_group.add_argument("--visualize", action='store_true',
                             help="Save detection visualization")
    
    # Utility
    parser.add_argument("--download-models", action='store_true',
                       help="Download all required models and exit")
    
    args = parser.parse_args()
    
    # Download models if requested
    if args.download_models:
        if not ModelDownloader.ensure_models():
            sys.exit(1)
        print("\n✓ All models downloaded successfully!")
        return
    
    if not args.image:
        parser.print_help()
        return
    
    # Ensure models are available
    if not ModelDownloader.ensure_models():
        print("\n✗ Required models not available. Use --download-models first.")
        sys.exit(1)
    
    device = None if args.device == 'auto' else args.device
    
    # Auto-detect if requested
    if args.detect and not args.mask:
        print("\n" + "="*60)
        print("STEP 1: DETECTING WATERMARK (JoyCaption Model)")
        print("="*60)
        
        detector = WatermarkDetector(device=device)
        mask_path = "detected_mask.png"
        
        success = detector.detect_watermark(
            args.image,
            mask_path,
            detection_mode=args.mode,
            refinement_method=args.refine,
            confidence_threshold=args.confidence,
            iou_threshold=args.iou,
            visualize=args.visualize
        )
        
        if success:
            args.mask = mask_path
        else:
            print("\n✗ Detection failed")
            sys.exit(1)
    
    if not args.mask:
        print("Error: Provide --mask or use --detect")
        sys.exit(1)
    
    # Remove watermark
    print("\n" + "="*60)
    print(f"STEP 2: REMOVING WATERMARK ({args.method})")
    print("="*60)
    
    remover = WatermarkRemover(device=device)
    success = remover.remove(args.image, args.mask, args.output, args.method)
    
    if success:
        print("\n" + "="*60)
        print("✓ SUCCESS!")
        print("="*60)
        print(f"  Clean image: {args.output}")
        if args.detect:
            print(f"  Mask: {args.mask}")
            if args.visualize:
                print(f"  Visualization: {args.mask.replace('.png', '_visualization.png')}")
    else:
        print("\n✗ Removal failed")
        sys.exit(1)


if __name__ == "__main__":
    main()