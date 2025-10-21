from __future__ import annotations

import json
import math
import warnings
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

try:
    import torch
    import torch.nn as nn
    _TORCH_AVAILABLE = True
except Exception:
    _TORCH_AVAILABLE = False
    torch = None
    nn = object

try:
    import easyocr
    _EASYOCR_AVAILABLE = True
except Exception:
    _EASYOCR_AVAILABLE = False
    easyocr = None

warnings.filterwarnings("ignore", category=RuntimeWarning)


@dataclass
class WatermarkRegion:
    bbox: Tuple[int, int, int, int]
    confidence: float
    area: int
    coverage: float
    classification: str
    votes: Dict[str, float]

    def to_dict(self) -> Dict:
        data = asdict(self)
        data["bbox"] = list(self.bbox)
        for key, value in data.items():
            if isinstance(value, (np.integer, np.floating, np.bool_)):
                data[key] = value.item()
            elif isinstance(value, np.ndarray):
                data[key] = value.tolist()
        return data


class MultiAlgorithmWatermarkDetector:
    """
    Enhanced watermark detector with improved person rejection.
    """

    WATERMARK_TYPES = [
        "text",
        "diagonal_text",
        "stamp",
        "signature",
        "logo",
        "pattern",
        "overlay",
    ]

    def __init__(
        self,
        saliency_weight: float = 0.2,  # Reduced from 0.6
        face_reject: bool = False,  # Changed default - we want watermarks ON faces
        device: Optional[str] = None,
        skin_reject: bool = True,  # NEW: Better approach than face rejection
    ):
        self.saliency_weight = np.clip(saliency_weight, 0.0, 1.0)
        self.face_reject = face_reject
        self.skin_reject = skin_reject
        self.device = device
        self._ocr_reader = None
        self._face_cascade = None

        if face_reject:
            self._load_face_cascade()

        if _TORCH_AVAILABLE and device:
            self.device = device
        elif _TORCH_AVAILABLE:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def _load_face_cascade(self):
        try:
            cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            self._face_cascade = cv2.CascadeClassifier(cascade_path)
            if self._face_cascade.empty():
                self._face_cascade = None
        except Exception:
            self._face_cascade = None

    def _load_ocr_reader(self):
        if not _EASYOCR_AVAILABLE:
            return None
        if self._ocr_reader is None:
            try:
                self._ocr_reader = easyocr.Reader(["en"], gpu=False)
            except Exception:
                self._ocr_reader = None
        return self._ocr_reader

    @staticmethod
    def _normalize(matrix: np.ndarray) -> np.ndarray:
        matrix = matrix.astype(np.float32)
        min_val, max_val = np.min(matrix), np.max(matrix)
        if max_val - min_val < 1e-6:
            return np.zeros_like(matrix, dtype=np.float32)
        return (matrix - min_val) / (max_val - min_val)

    @staticmethod
    def _ensure_gray(img: np.ndarray) -> np.ndarray:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img

    @staticmethod
    def _sigmoid(x: np.ndarray, gain: float = 10.0, threshold: float = 0.5) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-gain * (x - threshold)))

    @staticmethod
    def _bandpass_mask(shape: Tuple[int, int], radius: int = 20) -> np.ndarray:
        rows, cols = shape
        crow, ccol = rows // 2, cols // 2
        Y, X = np.ogrid[:rows, :cols]
        dist = np.sqrt((X - ccol) ** 2 + (Y - crow) ** 2)
        mask = np.logical_and(dist > radius, dist < 0.45 * max(rows, cols))
        return mask.astype(np.float32)

    # NEW: Skin detection to filter out person features
    def _detect_skin_mask(self, bgr: np.ndarray) -> np.ndarray:
        """Detect skin regions using YCrCb color space."""
        ycrcb = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
        
        # Skin detection thresholds in YCrCb
        lower_skin = np.array([0, 133, 77], dtype=np.uint8)
        upper_skin = np.array([255, 173, 127], dtype=np.uint8)
        
        skin_mask = cv2.inRange(ycrcb, lower_skin, upper_skin)
        
        # Also try HSV for better coverage
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        lower_hsv = np.array([0, 15, 0], dtype=np.uint8)
        upper_hsv = np.array([17, 170, 255], dtype=np.uint8)
        skin_mask_hsv = cv2.inRange(hsv, lower_hsv, upper_hsv)
        
        # Combine both masks
        skin_mask = cv2.bitwise_or(skin_mask, skin_mask_hsv)
        
        # Clean up the mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, kernel)
        skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, kernel)
        
        # Blur for soft rejection
        skin_mask = cv2.GaussianBlur(skin_mask, (15, 15), 0)
        
        return self._normalize(skin_mask)

    # IMPROVED: More selective brightness detection
    def _detect_brightness(self, gray: np.ndarray) -> Tuple[np.ndarray, str]:
        norm = self._normalize(gray)
        # More selective thresholds
        bright_mask = np.clip((norm - 0.75) * 4.0, 0, 1)  # Only very bright
        dark_mask = np.clip((0.35 - norm) * 4.0, 0, 1)  # Only very dark
        combined = np.maximum(bright_mask, dark_mask)
        return combined, "brightness_extremes"

    def _detect_adaptive(self, gray: np.ndarray) -> Tuple[np.ndarray, str]:
        block = 31 if min(gray.shape) > 300 else 21
        adaptive = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            block,
            5,
        )
        adaptive = self._normalize(adaptive)
        return adaptive, "adaptive_threshold"

    def _detect_gradient(self, gray: np.ndarray) -> Tuple[np.ndarray, str]:
        sobelx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        magnitude = np.sqrt(sobelx**2 + sobely**2)
        mag_norm = self._normalize(magnitude)
        # More aggressive threshold for sharper edges (watermarks are usually sharp)
        mag_norm = self._sigmoid(mag_norm, gain=8.0, threshold=0.35)
        return mag_norm, "gradient_edges"

    def _detect_top_hat(self, gray: np.ndarray) -> Tuple[np.ndarray, str]:
        kernel_size = max(5, int(min(gray.shape) * 0.02))
        if kernel_size % 2 == 0:
            kernel_size += 1
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
        tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)
        tophat_norm = self._normalize(tophat)
        return tophat_norm, "morph_tophat"

    def _detect_canny(self, gray: np.ndarray) -> Tuple[np.ndarray, str]:
        # Higher thresholds for more selective edge detection
        edges = cv2.Canny(gray, 60, 180)
        edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
        return self._normalize(edges), "canny_edges"

    def _detect_fft(self, gray: np.ndarray) -> Tuple[np.ndarray, str]:
        f = np.fft.fft2(gray)
        fshift = np.fft.fftshift(f)
        mask = self._bandpass_mask(gray.shape, radius=15)
        filtered = np.abs(fshift) * mask
        magnitude = np.log1p(filtered)
        magnitude = self._normalize(magnitude)
        return magnitude, "fft_highfreq"

    def _detect_dct(self, gray: np.ndarray) -> Tuple[np.ndarray, str]:
        h, w = gray.shape
        block = 32
        dct_map = np.zeros_like(gray, dtype=np.float32)
        for y in range(0, h, block):
            for x in range(0, w, block):
                block_img = gray[y : y + block, x : x + block]
                if block_img.size == 0:
                    continue
                block_float = block_img.astype(np.float32)
                dct = cv2.dct(block_float)
                high_freq = np.abs(dct[8:, 8:])
                score = np.mean(high_freq)
                dct_map[y : y + block, x : x + block] = score
        dct_norm = self._normalize(dct_map)
        return dct_norm, "dct_highfreq"

    def _detect_hsv_overlay(self, bgr: np.ndarray) -> Tuple[np.ndarray, str]:
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        h_channel, s_channel, v_channel = cv2.split(hsv)
        s_norm = self._normalize(s_channel)
        v_norm = self._normalize(v_channel)
        # Watermarks often have low saturation and high/medium value
        overlay = np.maximum(0, (0.7 - s_norm)) * np.clip(v_norm * 1.5, 0, 1)
        overlay = cv2.GaussianBlur(overlay, (9, 9), 0)
        return overlay, "low_saturation_overlay"

    def _detect_autocorrelation(self, gray: np.ndarray) -> Tuple[np.ndarray, str]:
        gray_float = self._normalize(gray)
        spectrum = np.fft.fft2(gray_float)
        power = spectrum * np.conj(spectrum)
        autocorr = np.fft.ifft2(power).real
        autocorr = self._normalize(autocorr)
        autocorr = np.fft.fftshift(autocorr)
        pattern = np.maximum(0, autocorr - 0.65)
        pattern = cv2.GaussianBlur(pattern, (5, 5), 0)
        return pattern, "autocorrelation_pattern"

    def _detect_text(self, bgr: np.ndarray, gray: np.ndarray) -> Tuple[np.ndarray, str]:
        reader = self._load_ocr_reader()
        mask = np.zeros(gray.shape, dtype=np.float32)
        
        if reader is not None:
            try:
                results = reader.readtext(bgr)
                for bbox, text, conf in results:
                    # Higher confidence threshold
                    if conf < 0.4:
                        continue
                    contour = np.array(bbox, dtype=np.int32)
                    cv2.fillPoly(mask, [contour], min(1.0, conf * 1.2))
            except Exception:
                pass

        if np.sum(mask) == 0:
            # Fallback morphological text detection
            grad = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
            grad = cv2.convertScaleAbs(grad)
            _, bw = cv2.threshold(grad, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
            bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, np.ones((3, 15), np.uint8))
            mask = self._normalize(bw)

        return mask, "text_detection"

    def _detect_saliency_inverse(self, gray: np.ndarray) -> Tuple[np.ndarray, str]:
        saliency = self._compute_saliency(gray)
        inverse = 1.0 - saliency
        inverse = cv2.GaussianBlur(inverse, (11, 11), 0)
        return inverse, "saliency_inverse"

    # IMPROVED: Better ML classifier with edge consistency check
    def _ml_classifier_map(
        self,
        masks: Dict[str, np.ndarray],
        gray: np.ndarray,
        confidence_map: np.ndarray,
    ) -> Tuple[np.ndarray, str]:
        features = [
            masks["brightness_extremes"],
            masks["adaptive_threshold"],
            masks["gradient_edges"],
            masks["morph_tophat"],
            masks["canny_edges"],
            masks["fft_highfreq"],
            masks["dct_highfreq"],
            masks["low_saturation_overlay"],
            masks["autocorrelation_pattern"],
            masks["text_detection"],
            masks["saliency_inverse"],
        ]
        feature_stack = np.stack(features, axis=2)
        feature_mean = np.mean(feature_stack, axis=2)
        feature_var = np.var(feature_stack, axis=2)
        
        # Watermarks have sharp edges, not gradual
        edge_weight = masks["gradient_edges"] * 0.7 + masks["canny_edges"] * 0.3
        freq_weight = masks["fft_highfreq"] * 0.5 + masks["dct_highfreq"] * 0.5

        # NEW: Edge consistency - watermarks have consistent edge patterns
        edge_consistency = cv2.GaussianBlur(edge_weight, (5, 5), 0)
        edge_consistency = np.where(edge_consistency > 0.3, 1.0, 0.0)

        ml_score = (
            0.25 * feature_mean
            + 0.10 * feature_var
            + 0.25 * edge_weight
            + 0.20 * freq_weight
            + 0.10 * confidence_map
            + 0.10 * edge_consistency
        )
        ml_score = self._sigmoid(ml_score, gain=5.0, threshold=0.4)
        return ml_score, "ml_classifier"

    def _compute_saliency(self, gray: np.ndarray) -> np.ndarray:
        gray_float = gray.astype(np.float32) / 255.0
        fft = np.fft.fft2(gray_float)
        log_amplitude = np.log(np.abs(fft) + 1e-8)
        spectral_residual = log_amplitude - cv2.blur(log_amplitude, (3, 3))
        saliency = np.abs(np.fft.ifft2(np.exp(spectral_residual + 1j * np.angle(fft)))) ** 2
        saliency = cv2.GaussianBlur(saliency, (9, 9), 0)
        return self._normalize(saliency)

    def _face_mask(self, bgr: np.ndarray) -> np.ndarray:
        if self._face_cascade is None:
            return np.zeros(bgr.shape[:2], dtype=np.float32)
        gray = self._ensure_gray(bgr)
        faces = self._face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)
        mask = np.zeros(gray.shape, dtype=np.float32)
        for (x, y, w, h) in faces:
            cv2.rectangle(mask, (x, y), (x + w, y + h), 1.0, thickness=-1)
        mask = cv2.GaussianBlur(mask, (15, 15), 0)
        return np.clip(mask, 0, 1)

    def detect(
        self,
        image_path: str,
        output_dir: Optional[str] = None,
        confidence_threshold: float = 0.45,  # Increased from 0.38
    ) -> Dict:
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(image_path)

        bgr = cv2.imread(str(image_path))
        if bgr is None:
            raise ValueError(f"Failed to load image: {image_path}")
        gray = self._ensure_gray(bgr)

        algorithm_masks: Dict[str, np.ndarray] = {}
        confidence_accumulator = np.zeros(gray.shape, dtype=np.float32)
        weight_accumulator = np.zeros(gray.shape, dtype=np.float32)

        def accumulate(mask_name: str, mask: np.ndarray, weight: float = 1.0):
            algorithm_masks[mask_name] = mask.astype(np.float32)
            confidence_accumulator[:] += mask * weight
            weight_accumulator[:] += weight

        detectors = [
            self._detect_brightness,
            self._detect_adaptive,
            self._detect_gradient,
            self._detect_top_hat,
            self._detect_canny,
            lambda g: self._detect_fft(g),
            lambda g: self._detect_dct(g),
            lambda g: self._detect_hsv_overlay(bgr),
            self._detect_autocorrelation,
            lambda g: self._detect_text(bgr, g),
            self._detect_saliency_inverse,
        ]

        # IMPROVED: Adjusted weights to favor text and sharp edges
        weights = {
            "brightness_extremes": 0.8,  # Reduced
            "adaptive_threshold": 1.0,
            "gradient_edges": 1.3,  # Increased
            "morph_tophat": 0.9,
            "canny_edges": 1.2,  # Increased
            "fft_highfreq": 1.1,
            "dct_highfreq": 1.0,
            "low_saturation_overlay": 1.1,
            "autocorrelation_pattern": 0.8,
            "text_detection": 1.5,  # Increased - text is common in watermarks
            "saliency_inverse": 0.5,  # Reduced
        }

        for detector in detectors:
            mask, name = detector(gray)
            accumulate(name, mask, weights.get(name, 1.0))

        with np.errstate(divide="ignore", invalid="ignore"):
            confidence_map = np.divide(
                confidence_accumulator,
                weight_accumulator,
                out=np.zeros_like(confidence_accumulator),
                where=weight_accumulator > 0,
            )

        # ML classifier
        ml_map, ml_name = self._ml_classifier_map(algorithm_masks, gray, confidence_map)
        accumulate(ml_name, ml_map, 1.5)
        confidence_map = np.divide(
            confidence_accumulator,
            weight_accumulator,
            out=np.zeros_like(confidence_accumulator),
            where=weight_accumulator > 0,
        )

        # IMPROVED: Minimal saliency rejection (watermarks can be on salient regions)
        saliency = self._compute_saliency(gray)
        confidence_map = np.clip(
            confidence_map * (1.0 - self.saliency_weight * saliency),
            0.0,
            1.0,
        )

        # NEW: Skin rejection instead of face rejection
        if self.skin_reject:
            skin_mask = self._detect_skin_mask(bgr)
            # Only reject if it's a large skin region without sharp edges
            edge_density = algorithm_masks["canny_edges"]
            skin_rejection = skin_mask * (1.0 - edge_density * 0.8)
            confidence_map *= (1.0 - 0.7 * skin_rejection)

        # Face rejection only if explicitly enabled (not recommended)
        if self.face_reject:
            face_mask = self._face_mask(bgr)
            confidence_map *= (1.0 - 0.5 * face_mask)  # Reduced from 0.9

        confidence_map = cv2.GaussianBlur(confidence_map, (7, 7), 0)

        # Binary mask with morphological cleanup
        mask_binary = (confidence_map >= confidence_threshold).astype(np.uint8)
        mask_binary = cv2.morphologyEx(mask_binary, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        mask_binary = cv2.morphologyEx(mask_binary, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

        # Remove very small regions (likely noise)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_binary, connectivity=8)
        for label_id in range(1, num_labels):
            area = stats[label_id, cv2.CC_STAT_AREA]
            if area < 50:  # Minimum area threshold
                mask_binary[labels == label_id] = 0

        regions = self._extract_regions(mask_binary, confidence_map, algorithm_masks, bgr, gray)

        result = {
            "mask": mask_binary,
            "confidence_map": confidence_map,
            "regions": [region.to_dict() for region in regions],
            "algorithm_votes": {k: float(np.mean(v)) for k, v in algorithm_masks.items()},
            "metadata": {
                "confidence_threshold": confidence_threshold,
                "image_path": str(image_path),
                "saliency_weight": self.saliency_weight,
                "face_reject": self.face_reject,
                "skin_reject": self.skin_reject,
                "ocr_used": _EASYOCR_AVAILABLE and self._ocr_reader is not None,
                "torch_available": _TORCH_AVAILABLE,
            },
        }

        if output_dir is not None:
            self._save_outputs(result, bgr, output_dir, image_path.name)

        return result

    def _extract_regions(
        self,
        mask_binary: np.ndarray,
        confidence_map: np.ndarray,
        algorithm_masks: Dict[str, np.ndarray],
        bgr: np.ndarray,
        gray: np.ndarray,
    ) -> List[WatermarkRegion]:
        regions: List[WatermarkRegion] = []
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_binary, connectivity=8)

        for label_id in range(1, num_labels):
            x, y, w, h, area = stats[label_id]
            if area < 50:  # Minimum area
                continue
            bbox = (x, y, x + w, y + h)
            region_mask = (labels == label_id).astype(np.uint8)

            region_conf = float(np.mean(confidence_map[labels == label_id]))
            coverage = float(area / mask_binary.size)
            votes = {name: float(np.mean(mask[labels == label_id])) for name, mask in algorithm_masks.items()}
            classification = self._classify_region(
                bbox, region_mask, votes, bgr, gray, region_conf
            )

            regions.append(
                WatermarkRegion(
                    bbox=bbox,
                    confidence=region_conf,
                    area=int(area),
                    coverage=coverage,
                    classification=classification,
                    votes=votes,
                )
            )
        return regions

    def _classify_region(
        self,
        bbox: Tuple[int, int, int, int],
        region_mask: np.ndarray,
        votes: Dict[str, float],
        bgr: np.ndarray,
        gray: np.ndarray,
        confidence: float,
    ) -> str:
        x1, y1, x2, y2 = bbox
        roi_gray = gray[y1:y2, x1:x2]
        roi_mask = region_mask[y1:y2, x1:x2]
        roi_bgr = bgr[y1:y2, x1:x2]

        h, w = roi_mask.shape
        aspect = w / float(h + 1e-6)
        diag = math.degrees(math.atan2(h, w))
        edge_density = votes.get("canny_edges", 0.0)
        text_vote = votes.get("text_detection", 0.0)
        pattern_vote = votes.get("autocorrelation_pattern", 0.0)
        freq_vote = (votes.get("fft_highfreq", 0.0) + votes.get("dct_highfreq", 0.0)) / 2.0
        overlay_vote = votes.get("low_saturation_overlay", 0.0)

        hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV) if roi_bgr.size else np.zeros((1, 1, 3), dtype=np.uint8)
        saturation_mean = float(np.mean(hsv[..., 1])) / 255.0
        value_mean = float(np.mean(hsv[..., 2])) / 255.0

        if min(w, h) < 10:
            return "unknown"

        # Text detection (highest priority for watermarks)
        if text_vote > 0.5:
            if abs(diag) > 25 or aspect > 4.0:
                return "diagonal_text"
            return "text"

        # Diagonal text
        if (abs(diag) > 25 or aspect > 4.0) and edge_density > 0.4:
            return "diagonal_text"

        # Stamp (circular/square with pattern)
        if 0.7 < aspect < 1.4 and pattern_vote > 0.3 and confidence > 0.5:
            return "stamp"

        # Signature (elongated with low saturation)
        if aspect > 3.0 and saturation_mean < 0.3 and edge_density > 0.35:
            return "signature"

        # Logo (high frequency, moderate edges)
        if freq_vote > 0.5 and edge_density > 0.4 and aspect < 2.5:
            return "logo"

        # Pattern
        if pattern_vote > 0.45:
            return "pattern"

        # Overlay
        if overlay_vote > 0.5 and saturation_mean < 0.25:
            return "overlay"

        return "text"  # Default to text for ambiguous cases

    @staticmethod
    def _looks_like_qr(mask: np.ndarray, gray: np.ndarray) -> bool:
        if mask.size == 0:
            return False
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 100:
                continue
            x, y, w, h = cv2.boundingRect(cnt)
            aspect = w / float(h + 1e-6)
            if 0.8 < aspect < 1.2:
                block = gray[y:y + h, x:x + w]
                if block.size == 0:
                    continue
                resized = cv2.resize(block, (21, 21), interpolation=cv2.INTER_LINEAR)
                grid = (resized > np.mean(resized)).astype(np.uint8)
                transitions = np.sum(np.abs(np.diff(grid, axis=0))) + np.sum(np.abs(np.diff(grid, axis=1)))
                if transitions > 150:
                    return True
        return False

    @staticmethod
    def _looks_like_border(bbox: Tuple[int, int, int, int], shape: Tuple[int, int, int]) -> Optional[str]:
        x1, y1, x2, y2 = bbox
        h, w = shape[:2]
        margin = int(min(h, w) * 0.05)
        if x1 < margin and x2 > w - margin and y2 - y1 < margin * 2:
            return "border"
        if y1 < margin and y2 > h - margin and x2 - x1 < margin * 2:
            return "border"
        return None

    def _save_outputs(self, result: Dict, bgr: np.ndarray, output_dir: str, stem: str):
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        mask_path = output_path / f"{stem}_mask.png"
        conf_path = output_path / f"{stem}_confidence.png"
        info_path = output_path / f"{stem}_regions.json"
        overlay_path = output_path / f"{stem}_overlay.png"

        mask_img = (result["mask"] * 255).astype(np.uint8)
        cv2.imwrite(str(mask_path), mask_img)

        conf_img = (np.clip(result["confidence_map"], 0, 1) * 255).astype(np.uint8)
        cv2.imwrite(str(conf_path), conf_img)

        overlay = bgr.copy()
        overlay[result["mask"] > 0] = [0, 0, 255]
        blended = cv2.addWeighted(bgr, 0.7, overlay, 0.3, 0.0)

        for region in result["regions"]:
            x1, y1, x2, y2 = region["bbox"]
            label = f"{region['classification']}:{region['confidence']:.2f}"
            cv2.rectangle(blended, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                blended,
                label,
                (x1, max(0, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1,
                cv2.LINE_AA,
            )
        cv2.imwrite(str(overlay_path), blended)

        def convert_numpy_types(obj):
            """Convert numpy types to native Python types for JSON serialization."""
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.integer, np.floating, np.bool_)):
                return obj.item()
            elif isinstance(obj, dict):
                return {key: convert_numpy_types(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            return obj

        json_payload = {
            "regions": result["regions"],
            "algorithm_votes": result["algorithm_votes"],
            "metadata": result["metadata"],
            "artifacts": {
                "mask": mask_path.name,
                "confidence_map": conf_path.name,
                "overlay": overlay_path.name,
            },
            "mask_coverage": float(np.sum(result["mask"] > 0) / result["mask"].size),
            "confidence_stats": {
                "min": float(np.min(result["confidence_map"])),
                "max": float(np.max(result["confidence_map"])),
                "mean": float(np.mean(result["confidence_map"])),
            },
        }

        # Convert numpy types to native Python types
        json_payload = convert_numpy_types(json_payload)

        with open(info_path, "w", encoding="utf-8") as f:
            json.dump(json_payload, f, indent=2)


# ----------------------------------------------------------------------
# Command line interface
# ----------------------------------------------------------------------
def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Comprehensive multi-algorithm watermark detector.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("image", help="Path to input image")
    parser.add_argument("--output-dir", default="outputs/detections", help="Directory to save results")
    parser.add_argument("--confidence", type=float, default=0.38, help="Confidence threshold for mask binarisation")
    parser.add_argument("--no-face-reject", action="store_true", help="Disable face/subject rejection")
    parser.add_argument("--saliency-weight", type=float, default=0.6, help="Weight of saliency suppression")
    args = parser.parse_args()

    detector = MultiAlgorithmWatermarkDetector(
        saliency_weight=args.saliency_weight,
        face_reject=not args.no_face_reject,
    )

    result = detector.detect(
        args.image,
        output_dir=args.output_dir,
        confidence_threshold=args.confidence,
    )

    print("\nDetection summary")
    print("=" * 60)
    print(f"Mask coverage: {np.sum(result['mask'] > 0) / result['mask'].size * 100:.2f}%")
    print(f"Regions detected: {len(result['regions'])}")

    for idx, region in enumerate(result["regions"], 1):
        bbox = region["bbox"]
        print(
            f"[{idx}] {region['classification']:<12} "
            f"conf={region['confidence']:.2f} "
            f"bbox={bbox} "
            f"coverage={region['coverage']*100:.2f}%"
        )

    print(f"\nOutputs saved under: {Path(args.output_dir).resolve()}")


if __name__ == "__main__":
    main()
