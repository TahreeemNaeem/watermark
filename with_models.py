
import os
import sys
import shutil
import tempfile
import torch
import numpy as np
from PIL import Image
from typing import Optional, Tuple, List, Dict
import cv2
from pathlib import Path
import urllib.request
from tqdm import tqdm

# Model download URLs and paths - Updated with JoyCaption model
PEOPLE_MODEL_NAME = 'yolov8n.pt'
WATERMARK_CLASS_KEYWORDS = (
    'watermark',
    'copyright',
    'logo',
    'text',
    'stamp',
    'mark',
    'signature',
    'overlay',
    'badge',
    'label'
)
REQUIRED_DEPENDENCIES = {
    'torch': "pip install torch torchvision torchaudio",
    'ultralytics': "pip install ultralytics",
    'cv2': "pip install opencv-python",
    'diffusers': "pip install diffusers transformers accelerate safetensors",
    'transformers': "pip install transformers",
    'accelerate': "pip install accelerate",
    'safetensors': "pip install safetensors",
    'huggingface_hub': "pip install huggingface_hub",
}

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
        ModelDownloader.verify_dependencies()

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

        # Auxiliary people suppression model (optional but recommended)
        if not os.path.exists(PEOPLE_MODEL_NAME):
            print(f"⚠ Optional YOLOv8 person model '{PEOPLE_MODEL_NAME}' not found.")
            print("  Download from https://github.com/ultralytics/assets/releases and place it in the project root.")
        else:
            print(f"✓ Found auxiliary person suppression model at {PEOPLE_MODEL_NAME}")

        print("✓ All models ready\n")
        return True

    @staticmethod
    def verify_dependencies() -> bool:
        """Check that python packages required by models are present."""
        missing = []
        for module_name, install_hint in REQUIRED_DEPENDENCIES.items():
            try:
                if module_name == 'cv2':
                    import cv2  # noqa: F401
                else:
                    __import__(module_name)
            except Exception:
                missing.append((module_name, install_hint))

        if missing:
            print("="*60)
            print("⚠ WARNING: Missing Python dependencies detected")
            for module_name, install_hint in missing:
                print(f"  - {module_name} (install with `{install_hint}`)")
            print("  Some functionality may fail until these packages are installed.\n")
            return False
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
        self._person_model = None
        
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
    
    def _load_people_detector(self):
        """Load general object detector to suppress people/objects from masks."""
        if self._person_model is None:
            try:
                from ultralytics import YOLO
                print("Loading auxiliary YOLOv8 model for person suppression...")
                self._person_model = YOLO(PEOPLE_MODEL_NAME)
                self._person_model.to(self.device)
                print("✓ Person suppression model loaded")
            except ImportError:
                print("⚠ Ultralytics not installed; cannot enable person suppression.")
                self._person_model = False
            except Exception as e:
                print(f"⚠ Failed to load auxiliary YOLOv8 model: {e}")
                self._person_model = False
        return self._person_model if self._person_model not in (False, None) else None
    
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

    def refine_with_hybrid(
        self,
        image_path: str,
        bounding_boxes: List[Tuple[int, int, int, int]],
        margin: float = 0.18
    ) -> Optional[np.ndarray]:
        """
        Hybrid segmentation tuned for watermark text/logos by combining HSV/Lab heuristics,
        adaptive thresholds and edge cues inside each YOLO detection.
        """
        if not bounding_boxes:
            return None

        image = cv2.imread(image_path)
        if image is None:
            return None

        h, w = image.shape[:2]
        combined_mask = np.zeros((h, w), dtype=np.uint8)

        for idx, bbox in enumerate(bounding_boxes):
            x1, y1, x2, y2 = self._expand_bbox(bbox, w, h, margin=margin)
            if x2 <= x1 or y2 <= y1:
                continue

            roi = image[y1:y2, x1:x2]
            if roi.size == 0:
                continue
            roi_h, roi_w = roi.shape[:2]
            if roi_h < 8 or roi_w < 8:
                continue

            roi_hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            roi_lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)
            roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            roi_gray_blur = cv2.GaussianBlur(roi_gray, (3, 3), 0)

            H, S, V = cv2.split(roi_hsv)
            L, A, B = cv2.split(roi_lab)

            S_f = S.astype(np.float32) / 255.0
            V_f = V.astype(np.float32) / 255.0
            L_f = L.astype(np.float32) / 255.0

            whiteness = (1.0 - S_f) * (0.6 * V_f + 0.4 * L_f)
            whiteness = cv2.GaussianBlur(whiteness, (3, 3), 0)
            w_thresh = np.percentile(whiteness, 65)
            bright_mask = (whiteness > w_thresh).astype(np.uint8) * 255

            darkness = (1.0 - S_f) * (1.0 - V_f)
            darkness = cv2.GaussianBlur(darkness, (3, 3), 0)
            d_thresh = np.percentile(darkness, 75)
            dark_mask = (darkness > d_thresh).astype(np.uint8) * 255

            # Adaptive threshold for fine text lines
            adaptive = cv2.adaptiveThreshold(
                roi_gray_blur,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY_INV,
                31 if min(roi_h, roi_w) > 40 else 21,
                4
            )

            # Top-hat to emphasise thin strokes/lines
            k_size = max(3, int(min(roi_h, roi_w) * 0.06))
            if k_size % 2 == 0:
                k_size += 1
            line_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k_size, 1))
            tophat_h = cv2.morphologyEx(roi_gray_blur, cv2.MORPH_TOPHAT, line_kernel)
            line_kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, k_size))
            tophat_v = cv2.morphologyEx(roi_gray_blur, cv2.MORPH_TOPHAT, line_kernel_v)
            tophat = cv2.max(tophat_h, tophat_v)
            _, tophat_mask = cv2.threshold(tophat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            # Edge cues
            edges = cv2.Canny(roi_gray_blur, 40, 160)
            edges = cv2.dilate(edges, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=1)

            region_mask = cv2.max(bright_mask, dark_mask)
            region_mask = cv2.max(region_mask, adaptive)
            region_mask = cv2.max(region_mask, tophat_mask)
            region_mask = cv2.max(region_mask, edges)

            region_mask = cv2.morphologyEx(
                region_mask,
                cv2.MORPH_CLOSE,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
                iterations=1
            )
            region_mask = cv2.medianBlur(region_mask, 3)

            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(region_mask, connectivity=8)
            refined = np.zeros_like(region_mask)
            roi_area = roi_h * roi_w
            min_area = max(30, int(0.0002 * roi_area))
            max_area = int(0.6 * roi_area)

            for comp in range(1, num_labels):
                area = stats[comp, cv2.CC_STAT_AREA]
                if area < min_area or area > max_area:
                    continue

                cx = stats[comp, cv2.CC_STAT_LEFT] + stats[comp, cv2.CC_STAT_WIDTH] / 2.0
                cy = stats[comp, cv2.CC_STAT_TOP] + stats[comp, cv2.CC_STAT_HEIGHT] / 2.0
                cx_norm = cx / roi_w
                cy_norm = cy / roi_h

                comp_mask = (labels == comp)
                satur_mean = float(np.mean(S[comp_mask]))
                value_mean = float(np.mean(V[comp_mask]))
                l_mean = float(np.mean(L[comp_mask]))
                whiteness_mean = float(np.mean(whiteness[comp_mask]))

                keep = True
                if satur_mean > 170 and value_mean < 80:
                    keep = False  # saturated dark region
                if whiteness_mean < (w_thresh - 0.05):
                    keep = False
                if 0.35 < cx_norm < 0.65 and 0.35 < cy_norm < 0.65 and area > 0.12 * roi_area:
                    keep = False

                if keep:
                    refined[comp_mask] = 255

            if np.sum(refined > 0) < min_area:
                refined = region_mask

            combined_mask[y1:y2, x1:x2] = np.maximum(
                combined_mask[y1:y2, x1:x2],
                refined
            )

        if np.sum(combined_mask > 0) < 10:
            return None

        combined_mask = cv2.morphologyEx(
            combined_mask,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
            iterations=1
        )
        combined_mask = cv2.morphologyEx(
            combined_mask,
            cv2.MORPH_OPEN,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
            iterations=1
        )
        return combined_mask

    @staticmethod
    def _expand_bbox(
        bbox: Tuple[int, int, int, int],
        img_w: int,
        img_h: int,
        margin: float = 0.1
    ) -> Tuple[int, int, int, int]:
        """Expand bounding box by a percentage margin while staying inside image bounds."""
        x1, y1, x2, y2 = bbox
        bw = x2 - x1
        bh = y2 - y1
        expand_w = int(bw * margin)
        expand_h = int(bh * margin)
        x1 = max(0, x1 - expand_w)
        y1 = max(0, y1 - expand_h)
        x2 = min(img_w, x2 + expand_w)
        y2 = min(img_h, y2 + expand_h)
        return x1, y1, x2, y2

    def _classical_mask_refinement(
        self,
        image_path: str,
        bounding_boxes: Optional[List[Tuple[int, int, int, int]]] = None
    ) -> Optional[np.ndarray]:
        """
        Classical CV-based fallback to build a watermark mask when ML refinements fail.
        Mixes adaptive thresholding, morphological top/black-hat responses and edges.
        """
        image = cv2.imread(image_path)
        if image is None:
            return None

        h, w = image.shape[:2]
        rois = bounding_boxes if bounding_boxes else [(0, 0, w, h)]
        margin = 0.12 if bounding_boxes else 0.05

        combined_mask = np.zeros((h, w), dtype=np.uint8)
        base_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

        for bbox in rois:
            x1, y1, x2, y2 = self._expand_bbox(bbox, w, h, margin=margin)
            if x2 <= x1 or y2 <= y1:
                continue

            roi = image[y1:y2, x1:x2]
            roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(roi_gray, (3, 3), 0)

            # Highlight bright and dark watermark strokes
            tophat = cv2.morphologyEx(blur, cv2.MORPH_TOPHAT, base_kernel)
            blackhat = cv2.morphologyEx(blur, cv2.MORPH_BLACKHAT, base_kernel)
            _, th_top = cv2.threshold(
                tophat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )
            _, th_blk = cv2.threshold(
                blackhat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )

            # Adaptive threshold catches semi-transparent overlays
            adaptive = cv2.adaptiveThreshold(
                blur, 255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY_INV,
                25, 2
            )

            # Edges for crisp logos / typography
            edges = cv2.Canny(roi_gray, 45, 160)

            region_mask = cv2.bitwise_or(th_top, th_blk)
            region_mask = cv2.bitwise_or(region_mask, adaptive)

            # Dilate edges before combining so they cover the full glyph
            if np.any(edges > 0):
                edges_dilated = cv2.dilate(edges, base_kernel, iterations=1)
                region_mask = cv2.bitwise_or(region_mask, edges_dilated)

            # Clean noise and enforce contiguous blobs
            region_mask = cv2.morphologyEx(region_mask, cv2.MORPH_CLOSE, base_kernel, iterations=2)
            region_mask = cv2.morphologyEx(region_mask, cv2.MORPH_OPEN, base_kernel, iterations=1)

            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(region_mask, connectivity=8)
            filtered = np.zeros_like(region_mask)
            roi_area = (y2 - y1) * (x2 - x1)
            min_area = max(32, int(0.00005 * roi_area))
            max_area = int(0.5 * roi_area)
            for idx in range(1, num_labels):
                area = stats[idx, cv2.CC_STAT_AREA]
                if min_area <= area <= max_area:
                    filtered[labels == idx] = 255

            combined_mask[y1:y2, x1:x2] = np.maximum(
                combined_mask[y1:y2, x1:x2],
                filtered
            )

        if np.sum(combined_mask > 0) < 10:
            return None
    
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, base_kernel, iterations=1)
        return combined_mask
    

    def _boost_mask_with_edges(
        self,
        image_path: str,
        mask: np.ndarray,
        bounding_boxes: Optional[List[Tuple[int, int, int, int]]] = None
    ) -> np.ndarray:
        """Strengthen masks by adding edge cues and filling residual holes."""
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None or mask is None:
            return mask

        h, w = image.shape[:2]
        edges = cv2.Canny(image, 35, 130)
        edge_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        edges = cv2.dilate(edges, edge_kernel, iterations=1)

        region_mask = np.full((h, w), 255, dtype=np.uint8)
        if bounding_boxes:
            region_mask = np.zeros((h, w), dtype=np.uint8)
            for bbox in bounding_boxes:
                x1, y1, x2, y2 = self._expand_bbox(bbox, w, h, margin=0.18)
                region_mask[y1:y2, x1:x2] = 255

        support = cv2.dilate(mask, edge_kernel, iterations=2)
        allowed = cv2.max(region_mask, support)
        edges = cv2.bitwise_and(edges, allowed)

        boosted = cv2.max(mask, edges)

        close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        boosted = cv2.morphologyEx(boosted, cv2.MORPH_CLOSE, close_kernel, iterations=1)
        boosted = cv2.morphologyEx(boosted, cv2.MORPH_OPEN, edge_kernel, iterations=1)
        boosted = cv2.medianBlur(boosted, 3)
        _, boosted = cv2.threshold(boosted, 0, 255, cv2.THRESH_BINARY)
        return boosted

    def _detect_people_mask(self, image_path: str, mask_shape: Tuple[int, int]) -> np.ndarray:
        """Run auxiliary detector to find people/objects that should be excluded."""
        model = self._load_people_detector()
        if model is None:
            return np.zeros(mask_shape, dtype=np.uint8)

        try:
            results = model.predict(
                source=image_path,
                imgsz=640,
                conf=0.35,
                classes=[0],  # person class in COCO
                verbose=False,
                device=self.device
            )
        except Exception as e:
            print(f"⚠ Person suppression inference failed: {e}")
            return np.zeros(mask_shape, dtype=np.uint8)

        h, w = mask_shape
        suppression_mask = np.zeros((h, w), dtype=np.uint8)
        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                coords = box.xyxy[0].tolist()
                if len(coords) != 4:
                    continue
                x1, y1, x2, y2 = map(int, coords)
                x1 = max(0, min(w, x1))
                x2 = max(0, min(w, x2))
                y1 = max(0, min(h, y1))
                y2 = max(0, min(h, y2))
                if x2 <= x1 or y2 <= y1:
                    continue
                suppression_mask[y1:y2, x1:x2] = 255
        if np.any(suppression_mask):
            dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
            suppression_mask = cv2.dilate(suppression_mask, dilate_kernel, iterations=1)
        return suppression_mask

    def _finalize_mask(
        self,
        image_path: str,
        mask: np.ndarray,
        output_path: str,
        bounding_boxes: Optional[List[Tuple[int, int, int, int]]] = None
    ) -> Tuple[np.ndarray, float]:
        """Boost, binarize, save mask and return coverage statistic."""
        mask = self._boost_mask_with_edges(image_path, mask, bounding_boxes)
        mask = self._prune_mask_components(image_path, mask, bounding_boxes)
        mask = np.where(mask > 0, 255, 0).astype(np.uint8)
        cv2.imwrite(output_path, mask)
        coverage = (np.sum(mask > 0) / mask.size) * 100
        return mask, coverage

    def _prune_mask_components(
        self,
        image_path: str,
        mask: np.ndarray,
        bounding_boxes: Optional[List[Tuple[int, int, int, int]]] = None
    ) -> np.ndarray:
        """Remove mask blobs that look like people/objects rather than watermarks."""
        image = cv2.imread(image_path)
        if image is None or mask is None:
            return mask

        h, w = mask.shape[:2]
        if np.sum(mask > 0) < 20:
            return mask

        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 45, 150)
        person_mask = self._detect_people_mask(image_path, (h, w))

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        if num_labels <= 1:
            return mask

        def in_detection_bbox(x, y, bw, bh, area) -> bool:
            if not bounding_boxes:
                return False
            comp_rect = (x, y, x + bw, y + bh)
            comp_area = area
            for bx1, by1, bx2, by2 in bounding_boxes:
                inter_x1 = max(comp_rect[0], bx1)
                inter_y1 = max(comp_rect[1], by1)
                inter_x2 = min(comp_rect[2], bx2)
                inter_y2 = min(comp_rect[3], by2)
                if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
                    continue
                inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
                if inter_area / max(1, comp_area) > 0.55:
                    return True
            return False

        pruned = mask.copy()
        removed_components = 0
        for idx in range(1, num_labels):
            area = stats[idx, cv2.CC_STAT_AREA]
            if area < 25:
                pruned[labels == idx] = 0
                removed_components += 1
                continue

            x = stats[idx, cv2.CC_STAT_LEFT]
            y = stats[idx, cv2.CC_STAT_TOP]
            bw = stats[idx, cv2.CC_STAT_WIDTH]
            bh = stats[idx, cv2.CC_STAT_HEIGHT]

            rel_area = area / float(h * w)
            center_x = (x + bw / 2) / w
            center_y = (y + bh / 2) / h

            comp_mask = (labels == idx)
            roi_mask = comp_mask[y:y+bh, x:x+bw]
            roi_image = image[y:y+bh, x:x+bw]
            if roi_image.size == 0:
                pruned[labels == idx] = 0
                removed_components += 1
                continue

            hsv_roi = hsv[y:y+bh, x:x+bw]
            sat_vals = hsv_roi[..., 1][roi_mask]
            val_vals = hsv_roi[..., 2][roi_mask]
            sat_mean = float(np.mean(sat_vals)) if sat_vals.size else 0.0
            sat_std = float(np.std(sat_vals)) if sat_vals.size else 0.0
            val_mean = float(np.mean(val_vals)) if val_vals.size else 0.0
            val_std = float(np.std(val_vals)) if val_vals.size else 0.0

            roi_flat = roi_image.reshape(-1, 3)
            roi_color_std = float(np.mean(np.std(roi_flat, axis=0)))

            edges_roi = edges[y:y+bh, x:x+bw]
            edge_density = float(np.mean(edges_roi[roi_mask] > 0)) if np.any(roi_mask) else 0.0
            person_overlap = float(np.mean(person_mask[comp_mask] > 0)) if np.any(comp_mask) else 0.0
            supported_by_detection = in_detection_bbox(x, y, bw, bh, area)

            # Early keep for very watermark-like components
            if sat_mean < 35 and val_mean > 155 and rel_area < 0.05:
                continue  # Bright, nearly white translucent watermark
            if rel_area < 0.0015 and val_mean > 120 and sat_mean < 70:
                continue  # Small bright glyph, likely text

            remove = False
            reason = ""

            if person_overlap > 0.35:
                remove = True
                reason = f"person overlap ({person_overlap:.2f})"
            elif rel_area > 0.22:
                remove = True
                reason = f"huge area ({rel_area*100:.1f}%)"
            elif rel_area > 0.08 and 0.3 < center_x < 0.7 and 0.3 < center_y < 0.7:
                remove = True
                reason = f"large central blob ({rel_area*100:.1f}%)"
            elif sat_mean > 85 and roi_color_std > 25:
                remove = True
                reason = f"high saturation ({sat_mean:.1f})"
            elif roi_color_std > 65 and rel_area > 0.01:
                remove = True
                reason = f"high color variance ({roi_color_std:.1f})"
            elif edge_density > 0.45 and rel_area > 0.005:
                remove = True
                reason = f"edge-dense ({edge_density:.2f})"
            elif val_std > 60 and roi_color_std > 45:
                remove = True
                reason = f"high brightness variance ({val_std:.1f})"
            elif not supported_by_detection:
                # Compute a soft score to decide whether to keep non-detected components
                score = 0.0
                if sat_mean < 65:
                    score += 1.0
                if val_mean > 145 or val_mean < 85:
                    score += 0.6
                if 0.0003 <= rel_area <= 0.06:
                    score += 0.7
                if edge_density > 0.02 and edge_density < 0.32:
                    score += 0.5
                if roi_color_std < 45:
                    score += 0.4
                if sat_std < 25:
                    score += 0.3
                if val_std < 40:
                    score += 0.3
                # Corner preference
                if center_x < 0.22 or center_x > 0.78 or center_y < 0.22 or center_y > 0.78:
                    score += 0.4
                if rel_area < 0.00025:
                    score -= 0.3
                if score < 1.2:
                    remove = True
                    reason = f"low watermark score ({score:.2f})"

            if remove and supported_by_detection:
                remove = False  # keep if strongly supported by detector

            if remove:
                pruned[comp_mask] = 0
                removed_components += 1
                print(f"  - Pruned component {idx} ({reason})")

        if removed_components > 0:
            cleanup_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            pruned = cv2.morphologyEx(pruned, cv2.MORPH_OPEN, cleanup_kernel, iterations=1)

        if np.sum(pruned > 0) < 10 and np.sum(mask > 0) >= 10:
            return mask
        return pruned
    
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
                print("✗ No watermarks detected by YOLO. Falling back to classical analysis...")
                fallback_mask = self._classical_mask_refinement(image_path)
                if fallback_mask is None:
                    print("✗ Classical fallback also failed to find a watermark.")
                    return False
                
                fallback_mask, coverage = self._finalize_mask(
                    image_path, fallback_mask, output_mask_path, None
                )
                print(f"✓ Watermark mask created via classical fallback! Coverage: {coverage:.2f}%")
                if visualize:
                    h, w = fallback_mask.shape[:2]
                    fake_detections = [{
                        'bbox': (0, 0, w, h),
                        'confidence': 0.0,
                        'class_id': 0,
                        'class_name': 'classical_fallback'
                    }]
                    self._create_visualization(
                        image_path,
                        fake_detections,
                        fallback_mask,
                        output_mask_path.replace('.png', '_visualization.png')
                    )
                return True
            
            print(f"Found {len(detections)} potential watermarks:")
            for det in detections:
                print(f"  - {det['class_name']}: {det['confidence']:.2f} at {det['bbox']}")
            
            # Filter detections likely to be watermarks (JoyCaption model is already specialized)
            if detection_mode == 'smart':
                detections = self._filter_watermark_candidates(detections, image_path)
            
            if not detections:
                print("✗ No valid watermark candidates after filtering. Using classical fallback...")
                fallback_mask = self._classical_mask_refinement(image_path)
                if fallback_mask is None:
                    print("✗ Classical fallback also failed to isolate watermark regions.")
                    return False
                
                fallback_mask, coverage = self._finalize_mask(
                    image_path, fallback_mask, output_mask_path, None
                )
                print(f"✓ Watermark mask created via classical fallback! Coverage: {coverage:.2f}%")
                if visualize:
                    h, w = fallback_mask.shape[:2]
                    fake_detections = [{
                        'bbox': (0, 0, w, h),
                        'confidence': 0.0,
                        'class_id': 0,
                        'class_name': 'classical_fallback'
                    }]
                    self._create_visualization(
                        image_path,
                        fake_detections,
                        fallback_mask,
                        output_mask_path.replace('.png', '_visualization.png')
                    )
                return True
            
            # Extract bounding boxes
            bboxes = [det['bbox'] for det in detections]
            
            # Refine with selected method
            if refinement_method == 'auto':
                # Try methods in order of speed/quality tradeoff
                print("Using auto refinement (trying multiple methods)...")
                mask = self.refine_with_hybrid(image_path, bboxes)
                if mask is None or np.sum(mask > 0) < 200:
                    print("  Hybrid refinement insufficient, trying GrabCut...")
                    mask = self.refine_with_grabcut(image_path, bboxes)
                if mask is None or np.sum(mask > 0) < 100:
                    print("  GrabCut insufficient, trying morphology...")
                    mask = self.refine_with_morphology(image_path, bboxes)
            elif refinement_method == 'hybrid':
                mask = self.refine_with_hybrid(image_path, bboxes)
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

            # Calculate coverage and auto fallback if mask too small
            coverage = (np.sum(mask > 0) / mask.size) * 100
            bbox_area_ratio = sum(
                max(0, (x2 - x1)) * max(0, (y2 - y1))
                for (x1, y1, x2, y2) in bboxes
            ) / float(mask.size)
            expected_coverage = bbox_area_ratio * 120.0  # allow 20% slack over bbox sizes
            min_coverage = max(0.3, min(25.0, expected_coverage))
            if coverage < min_coverage:
                print(f"  Mask coverage ({coverage:.2f}%) below expected threshold ({min_coverage:.2f}%).")
                print("  Applying classical CV fallback to enlarge watermark region...")
                classical_mask = self._classical_mask_refinement(image_path, bboxes)
                if classical_mask is not None:
                    mask = np.maximum(mask, classical_mask)
                    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
                    coverage = (np.sum(mask > 0) / mask.size) * 100
                    print(f"  Coverage after fallback: {coverage:.2f}%")
                else:
                    print("  Classical fallback failed to improve mask coverage.")

            mask, coverage = self._finalize_mask(
                image_path, mask, output_mask_path, bboxes
            )
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
            class_name = str(det.get('class_name', '') or '').lower()
            has_valid_keyword = False
            if class_name:
                has_valid_keyword = any(keyword in class_name for keyword in WATERMARK_CLASS_KEYWORDS)
                if not has_valid_keyword:
                    continue
            
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

    @staticmethod
    def _prepare_mask(mask: np.ndarray, dilate_px: int = 2, feather_px: int = 4) -> np.ndarray:
        """Make mask robust before inpainting by dilating and feathering edges."""
        if mask is None:
            raise ValueError("Mask could not be loaded.")

        if mask.dtype != np.uint8:
            mask = mask.astype(np.uint8)

        _, mask_bin = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY)

        if dilate_px > 0:
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (dilate_px * 2 + 1, dilate_px * 2 + 1)
            )
            mask_bin = cv2.dilate(mask_bin, kernel, iterations=1)

        cleanup_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask_bin = cv2.morphologyEx(mask_bin, cv2.MORPH_CLOSE, cleanup_kernel, iterations=1)

        if feather_px > 0:
            sigma = max(0.5, feather_px / 2.0)
            blurred = cv2.GaussianBlur(mask_bin, (0, 0), sigma)
            max_val = blurred.max()
            if max_val > 0:
                mask_bin = np.clip((blurred / max_val) * 255.0, 0, 255).astype(np.uint8)

        return mask_bin
        
    def _load_lama(self):
        """Load LaMa model"""
        if self._lama_model is None:
            cache_dir = Path(torch.hub.get_dir()) / 'advimman_lama_main'
            hubconf_path = cache_dir / 'hubconf.py'
            force_reload = False
            if cache_dir.exists() and not hubconf_path.exists():
                print("Detected incomplete LaMa cache. Refreshing download...")
                try:
                    shutil.rmtree(cache_dir)
                except Exception as cleanup_error:
                    print(f"  Warning: could not clean LaMa cache: {cleanup_error}")
                force_reload = True

            def _load_repo(force: bool):
                load_kwargs = dict(pretrained=True, force_reload=force)
                try:
                    return torch.hub.load(
                        'advimman/lama',
                        'lama',
                        trust_repo=True,
                        **load_kwargs
                    )
                except TypeError:
                    # Older torch versions may not support trust_repo
                    return torch.hub.load(
                        'advimman/lama',
                        'lama',
                        **load_kwargs
                    )

            try:
                print("Loading LaMa model...")
                self._lama_model = _load_repo(force_reload)
                self._lama_model = self._lama_model.to(self.device)
                self._lama_model.eval()
                print("✓ LaMa loaded")
            except Exception as e:
                if not force_reload:
                    print(f"  Initial LaMa load failed ({e}). Retrying with a clean cache...")
                    try:
                        if cache_dir.exists():
                            shutil.rmtree(cache_dir)
                    except Exception as cleanup_error:
                        print(f"  Warning: could not clean LaMa cache: {cleanup_error}")
                    try:
                        self._lama_model = _load_repo(True)
                        self._lama_model = self._lama_model.to(self.device)
                        self._lama_model.eval()
                        print("✓ LaMa loaded after cache refresh")
                        return self._lama_model
                    except Exception as second_error:
                        print(f"✗ Failed to load LaMa after retry: {second_error}")
                        raise
                print(f"✗ Failed to load LaMa: {e}")
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

    def _fallback_diffusion_inpaint(
        self,
        image_path: str,
        mask_path: str,
        output_path: str,
        mask_dilate_px: int,
        mask_feather_px: int,
        prepared_mask: Optional[np.ndarray] = None
    ) -> bool:
        """Fallback using diffusers-based plugin if LaMa is unavailable."""
        try:
            from plugins import lama_plugin
        except Exception as plugin_import_error:
            print(f"✗ Diffusers fallback unavailable: {plugin_import_error}")
            return False

        if prepared_mask is None:
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                print(f"✗ Fallback failed: mask not found at {mask_path}")
                return False
            prepared_mask = self._prepare_mask(mask, dilate_px=mask_dilate_px, feather_px=mask_feather_px)

        if prepared_mask.dtype != np.uint8:
            prepared_mask = prepared_mask.astype(np.uint8)

        tmp_file = None
        try:
            with tempfile.NamedTemporaryFile(suffix="_prepared_mask.png", delete=False) as tmp:
                cv2.imwrite(tmp.name, prepared_mask)
                tmp_file = tmp.name
            print("→ Using diffusers-based fallback inpainting...")
            success = lama_plugin.inpaint(image_path, tmp_file, output_path, device=self.device)
            if success:
                print("✓ Diffusers fallback completed successfully.")
            return success
        except Exception as plugin_error:
            print(f"✗ Diffusers fallback failed: {plugin_error}")
            return False
        finally:
            if tmp_file and os.path.exists(tmp_file):
                try:
                    os.remove(tmp_file)
                except OSError:
                    pass
    
    def remove_with_lama(
        self,
        image_path: str,
        mask_path: str,
        output_path: str,
        mask_dilate_px: int = 2,
        mask_feather_px: int = 4
    ) -> bool:
        """Remove watermark using LaMa"""
        mask_gray = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask_gray is None:
            print(f"✗ Mask not found at {mask_path}")
            return False

        prepared_mask = self._prepare_mask(
            mask_gray,
            dilate_px=mask_dilate_px,
            feather_px=mask_feather_px
        )

        try:
            model = self._load_lama()
        except Exception as load_error:
            print(f"✗ LaMa load failed: {load_error}")
            return self._fallback_diffusion_inpaint(
                image_path,
                mask_path,
                output_path,
                mask_dilate_px,
                mask_feather_px,
                prepared_mask
            )

        try:
            image = cv2.imread(image_path)
            if image is None:
                raise FileNotFoundError(f"Image not found at {image_path}")
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            image_tensor = torch.from_numpy(image_rgb).permute(2, 0, 1).unsqueeze(0).float() / 255.0
            mask_tensor = torch.from_numpy(prepared_mask).unsqueeze(0).unsqueeze(0).float() / 255.0

            image_tensor = image_tensor.to(self.device)
            mask_tensor = mask_tensor.to(self.device)

            print("Inpainting with LaMa...")

            with torch.no_grad():
                result = model(image_tensor, mask_tensor)

            result = result[0].permute(1, 2, 0).cpu().numpy()
            result = (result * 255).clip(0, 255).astype(np.uint8)
            result = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)

            cv2.imwrite(output_path, result, [cv2.IMWRITE_PNG_COMPRESSION, 9])
            print(f"✓ Saved to: {output_path}")
            return True

        except Exception as e:
            print(f"✗ LaMa failed: {e}")
            return self._fallback_diffusion_inpaint(
                image_path,
                mask_path,
                output_path,
                mask_dilate_px,
                mask_feather_px,
                prepared_mask
            )
    
    def remove_with_sd(
        self,
        image_path: str,
        mask_path: str,
        output_path: str,
        prompt: str = "high quality, clean, no watermark",
        negative_prompt: str = "watermark, text, logo, low quality",
        num_steps: int = 30,
        mask_dilate_px: int = 2,
        mask_feather_px: int = 4
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

            mask_np = np.array(mask)
            mask_np = self._prepare_mask(mask_np, dilate_px=mask_dilate_px, feather_px=mask_feather_px)
            mask = Image.fromarray(mask_np)
            
            print(f"Inpainting with Stable Diffusion ({num_steps} steps)...")
            
            result = pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                image=image,
                mask_image=mask,
                num_inference_steps=num_steps,
                guidance_scale=7.5
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
        method: str = 'lama',
        mask_dilate_px: int = 2,
        mask_feather_px: int = 4
    ) -> bool:
        """Remove watermark using specified method"""
        if method == 'lama':
            return self.remove_with_lama(
                image_path,
                mask_path,
                output_path,
                mask_dilate_px=mask_dilate_px,
                mask_feather_px=mask_feather_px
            )
        elif method == 'sd':
            return self.remove_with_sd(
                image_path,
                mask_path,
                output_path,
                mask_dilate_px=mask_dilate_px,
                mask_feather_px=mask_feather_px
            )
        elif method == 'auto':
            # Try LaMa first (faster), fallback to SD
            if self.remove_with_lama(
                image_path,
                mask_path,
                output_path,
                mask_dilate_px=mask_dilate_px,
                mask_feather_px=mask_feather_px
            ):
                return True
            print("Falling back to Stable Diffusion...")
            return self.remove_with_sd(
                image_path,
                mask_path,
                output_path,
                mask_dilate_px=mask_dilate_px,
                mask_feather_px=mask_feather_px
            )
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
  
  # Precision hybrid refinement tuned for watermark text/logos
  python script.py image.jpg --detect --refine hybrid --output clean.png
  
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
    parser.add_argument("--mask-dilate", type=int, default=2,
                       help="Pixels to dilate the mask before inpainting")
    parser.add_argument("--mask-feather", type=int, default=4,
                       help="Gaussian blur (in px) to feather mask edges")
    
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
    detect_group.add_argument("--refine", choices=['hybrid', 'fastsam', 'grabcut', 'morphology', 'advanced', 'auto'],
                             default='hybrid', help="Mask refinement method")
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
    success = remover.remove(
        args.image,
        args.mask,
        args.output,
        args.method,
        mask_dilate_px=max(0, args.mask_dilate),
        mask_feather_px=max(0, args.mask_feather)
    )
    
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
