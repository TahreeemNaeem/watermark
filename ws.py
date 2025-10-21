import os
import argparse
from typing import List, Tuple, Optional
import logging
import importlib.util
import tempfile

# Fast path deps (optional)
import numpy as np
try:
    import cv2  # type: ignore
    _CV2_IMPORT_ERROR = None
except Exception as _e:  # OpenCV may be missing or have dylib issues
    cv2 = None  # type: ignore
    _CV2_IMPORT_ERROR = _e

class WatermarkRemover:
    """
    Watermark removal system with two engines:
    - fast: automatic mask + OpenCV inpainting (default)
    - ai: Deep Image Prior-based removal (slower, optional)
    """

    def __init__(self, debug: bool = False):
        self.debug = debug
        if debug:
            logging.basicConfig(level=logging.DEBUG)

    # -------------------- FAST PATH (OpenCV) --------------------
    def _resize_for_detection(self, img: np.ndarray, max_dim: int = 1200) -> Tuple[np.ndarray, float]:
        h, w = img.shape[:2]
        scale = 1.0
        max_wh = max(h, w)
        if max_wh > max_dim:
            scale = max_dim / float(max_wh)
            img_small = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
            return img_small, scale
        return img, scale

    def _auto_detect_mask(
        self,
        img_bgr: np.ndarray,
        debug: bool = False,
        whiteness_thr: float = None,
        darkness_thr: float = None,
        grad_percentile: float = 90.0,
        morph_frac: float = 0.015,
        multiscale: bool = True,
        use_mser: bool = True,
        use_gabor: bool = True,
        use_saliency: bool = True,
    ) -> np.ndarray:
        """
        Automatically detect likely watermark regions quickly.
        Returns a binary uint8 mask (255 = watermark to remove) at full resolution.
        """
        orig_h, orig_w = img_bgr.shape[:2]

        # Work on a downscaled copy for speed
        work_img, scale0 = self._resize_for_detection(img_bgr, max_dim=1200)
        h0, w0 = work_img.shape[:2]

        def _gabor_text_map(gray: np.ndarray) -> np.ndarray:
            # Bank of Gabor filters to capture stroke-like patterns at multiple orientations
            grayf = gray.astype(np.float32) / 255.0
            angles = [0, 30, 45, 60, 90, 120, 135]
            lam = 8.0  # wavelength
            sigma = 3.0
            gamma = 0.5
            ksize = 9
            acc = np.zeros_like(grayf)
            for theta_deg in angles:
                theta = np.deg2rad(theta_deg)
                kernel = cv2.getGaborKernel((ksize, ksize), sigma, theta, lam, gamma, 0, ktype=cv2.CV_32F)
                resp = cv2.filter2D(grayf, cv2.CV_32F, kernel)
                acc = np.maximum(acc, np.abs(resp))
            acc = (acc / (acc.max() + 1e-6))
            acc = (acc * 255).astype(np.uint8)
            # Threshold by percentile to keep only strong responses
            thr = np.percentile(acc, 85)
            return (acc > thr).astype(np.uint8) * 255

        def _spectral_residual_saliency(gray: np.ndarray) -> np.ndarray:
            # Simple spectral residual saliency implementation
            g = gray.astype(np.float32) / 255.0
            h, w = g.shape
            # Add small border to avoid wrap artifacts
            f = np.fft.fft2(g)
            log_amp = np.log(np.abs(f) + 1e-8)
            phase = np.angle(f)
            avg_log_amp = cv2.blur(log_amp, (3, 3))
            spectral_residual = log_amp - avg_log_amp
            sal = np.fft.ifft2(np.exp(spectral_residual + 1j * phase))
            sal = np.abs(sal) ** 2
            sal = cv2.GaussianBlur(sal, (0, 0), 2.0)
            sal = (sal - sal.min()) / (sal.max() - sal.min() + 1e-6)
            sal = (sal * 255).astype(np.uint8)
            thr = np.percentile(sal, 80)
            return (sal > thr).astype(np.uint8) * 255

        def detect_at_scale(im):
            h, w = im.shape[:2]
            lab = cv2.cvtColor(im, cv2.COLOR_BGR2LAB)
            hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
            L, A, B = cv2.split(lab)
            H, S, V = cv2.split(hsv)

            Lf = L.astype(np.float32)
            Sf = S.astype(np.float32)
            Vf = V.astype(np.float32)

            s_frac = morph_frac
            k = max(7, int(min(h, w) * s_frac))
            if k % 2 == 0:
                k += 1
            se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
            tophat = cv2.morphologyEx(L, cv2.MORPH_TOPHAT, se)
            blackhat = cv2.morphologyEx(L, cv2.MORPH_BLACKHAT, se)

            gx = cv2.Sobel(L, cv2.CV_32F, 1, 0, ksize=3)
            gy = cv2.Sobel(L, cv2.CV_32F, 0, 1, ksize=3)
            grad = cv2.magnitude(gx, gy)
            grad = cv2.convertScaleAbs(grad)

            # gating by desaturation
            S_norm = (Sf / 255.0)
            V_norm = (Vf / 255.0)
            whiteness = (1.0 - S_norm) * V_norm
            darkness = (1.0 - S_norm) * (1.0 - V_norm)

            # Adaptive thresholds if None
            w_thr = np.percentile(whiteness, 70) if whiteness_thr is None else whiteness_thr
            d_thr = np.percentile(darkness, 70) if darkness_thr is None else darkness_thr

            white_gate = (whiteness > w_thr).astype(np.uint8) * 255
            dark_gate = (darkness > d_thr).astype(np.uint8) * 255

            _, th_top = cv2.threshold(tophat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            _, th_blk = cv2.threshold(blackhat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            cand_white = cv2.bitwise_and(th_top, white_gate)
            cand_dark = cv2.bitwise_and(th_blk, dark_gate)

            grad_thr = np.percentile(grad, grad_percentile)
            grad_mask = (grad > max(10, grad_thr)).astype(np.uint8) * 255
            grad_gate = ((white_gate > 0) | (dark_gate > 0)).astype(np.uint8) * 255
            cand_grad = cv2.bitwise_and(grad_mask, grad_gate)

            combined = cv2.max(cv2.max(cand_white, cand_dark), cand_grad)

            # Add Gabor text response
            if use_gabor:
                gabor_mask = _gabor_text_map(L)
                # gate gabor by low saturation to avoid colorful textures
                gabor_mask = cv2.bitwise_and(gabor_mask, (Sf < 90).astype(np.uint8) * 255)
                combined = cv2.max(combined, gabor_mask)

            # Add spectral residual saliency
            if use_saliency:
                sal_mask = _spectral_residual_saliency(L)
                sal_mask = cv2.bitwise_and(sal_mask, (Sf < 110).astype(np.uint8) * 255)
                combined = cv2.max(combined, sal_mask)
            close_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (max(3, k // 3), max(3, k // 3)))
            open_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, close_k, iterations=1)
            combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, open_k, iterations=1)

            # Remove tiny noise
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(combined, connectivity=8)
            cleaned = np.zeros_like(combined)
            min_area = max(50, int(0.00005 * h * w))
            max_area = int(0.7 * h * w)
            for i in range(1, num_labels):
                area = stats[i, cv2.CC_STAT_AREA]
                if min_area <= area <= max_area:
                    cleaned[labels == i] = 255

            dil_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            cleaned = cv2.dilate(cleaned, dil_k, iterations=1)
            return cleaned

        # Multi-scale detection (union)
        masks = []
        masks.append(detect_at_scale(work_img))
        if multiscale:
            for s in (0.75, 0.5):
                small = cv2.resize(work_img, (int(w0 * s), int(h0 * s)), interpolation=cv2.INTER_AREA)
                m_small = detect_at_scale(small)
                m_up = cv2.resize(m_small, (w0, h0), interpolation=cv2.INTER_NEAREST)
                masks.append(m_up)
        combined = masks[0]
        for m in masks[1:]:
            combined = cv2.max(combined, m)

        # MSER text regions (useful for typographic watermarks)
        if use_mser:
            try:
                gray = cv2.cvtColor(work_img, cv2.COLOR_BGR2GRAY)
                mser = cv2.MSER_create(_min_area=60, _max_area=max(200, int(0.02 * w0 * h0)))
                regions, _ = mser.detectRegions(gray)
                mser_mask = np.zeros_like(gray, dtype=np.uint8)
                for pts in regions:
                    hull = cv2.convexHull(pts.reshape(-1, 1, 2))
                    cv2.fillConvexPoly(mser_mask, hull, 255)
                # gate MSER by low saturation to avoid natural text/edges
                hsv = cv2.cvtColor(work_img, cv2.COLOR_BGR2HSV)
                S = hsv[:, :, 1].astype(np.float32) / 255.0
                mser_mask = cv2.bitwise_and(mser_mask, (S < 0.4).astype(np.uint8) * 255)
                mser_mask = cv2.dilate(mser_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=1)
                combined = cv2.max(combined, mser_mask)
            except Exception:
                pass

        # Upscale to original size if we downscaled
        if scale0 != 1.0:
            cleaned = cv2.resize(combined, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
        else:
            cleaned = combined

        if debug:
            dbg_dir = os.environ.get('WMR_DEBUG_DIR', None)
            if dbg_dir:
                cv2.imwrite(os.path.join(dbg_dir, 'temp_mask_debug_small.png'), combined)
                cv2.imwrite(os.path.join(dbg_dir, 'temp_mask_debug_cleaned.png'), cleaned)
            else:
                cv2.imwrite('temp_mask_debug_small.png', combined)
                cv2.imwrite('temp_mask_debug_cleaned.png', cleaned)

        return cleaned.astype(np.uint8)

    def _shrink_mask(self, mask_bin: np.ndarray, iterations: int = 1) -> np.ndarray:
        if iterations <= 0:
            return mask_bin
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        return cv2.erode(mask_bin, k, iterations=iterations)

    def _limit_mask_coverage(self, mask_bin: np.ndarray, max_coverage: float = 0.33) -> np.ndarray:
        """Reduce mask area if automatic detection grew too large."""
        if cv2 is None:
            return mask_bin
        try:
            max_coverage = float(max_coverage)
        except Exception:
            return mask_bin
        if max_coverage <= 0 or mask_bin.size == 0:
            return mask_bin

        coverage = float(mask_bin.mean()) / 255.0
        if coverage <= max_coverage or coverage == 0.0:
            return mask_bin

        orig_cov = coverage
        shrunk = mask_bin.copy()
        iterations = 0
        while coverage > max_coverage and coverage > 0.0 and iterations < 8:
            shrunk = self._shrink_mask(shrunk, iterations=1)
            coverage = float(shrunk.mean()) / 255.0
            iterations += 1
            if coverage == 0.0:
                break

        if coverage == 0.0:
            return mask_bin
        if coverage < 0.02 and orig_cov > 0.4:
            return mask_bin

        if iterations > 0 and coverage < orig_cov:
            if self.debug:
                print(f"Mask coverage adjusted {orig_cov:.3f} -> {coverage:.3f} (limit {max_coverage:.3f})")
            return shrunk
        return mask_bin

    def _filter_mask_components(
        self,
        mask_bin: np.ndarray,
        img_bgr: Optional[np.ndarray],
        max_total_frac: float = 0.33,
        max_component_frac: float = 0.22,
        min_elongation: float = 0.18,
    ) -> np.ndarray:
        """Drop large, blob-like components to keep the mask focused on watermark strokes."""
        if os.environ.get('WMR_SKIP_COMPONENT_FILTER'):
            return mask_bin
        if cv2 is None or mask_bin is None:
            return mask_bin

        try:
            max_total_frac = float(max_total_frac)
        except Exception:
            max_total_frac = 0.33
        try:
            max_component_frac = float(max_component_frac)
        except Exception:
            max_component_frac = 0.22
        try:
            min_elongation = float(min_elongation)
        except Exception:
            min_elongation = 0.18

        if mask_bin.size == 0 or max_total_frac <= 0:
            return mask_bin

        total_px = float(mask_bin.shape[0] * mask_bin.shape[1])
        if total_px == 0:
            return mask_bin

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_bin, connectivity=8)
        if num_labels <= 1:
            return mask_bin

        components: List[Tuple[float, int, float]] = []
        img_lab = None
        median_L = None
        if img_bgr is not None:
            try:
                img_lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
                median_L = float(np.median(img_lab[..., 0]))
            except Exception:
                img_lab = None
                median_L = None

        for idx in range(1, num_labels):
            area = float(stats[idx, cv2.CC_STAT_AREA])
            if area <= 0:
                continue
            frac = area / total_px
            if frac < 5e-5:
                continue
            if frac > max_component_frac:
                continue

            ys, xs = np.nonzero(labels == idx)
            if len(xs) < 12:
                continue

            coords = np.vstack([xs.astype(np.float32), ys.astype(np.float32)])
            cov = np.cov(coords)
            if not np.isfinite(cov).all():
                continue
            try:
                eigvals, _ = np.linalg.eigh(cov + 1e-6 * np.eye(2))
            except Exception:
                continue
            major = float(np.sqrt(max(eigvals)))
            minor = float(np.sqrt(min(eigvals)))
            if major <= 1e-6:
                continue
            elong = minor / (major + 1e-6)
            if elong > min_elongation:
                continue

            score = 1.0 / (elong + 1e-3)

            if img_lab is not None and median_L is not None:
                try:
                    L_channel = img_lab[..., 0]
                    component_vals = L_channel[ys, xs]
                    contrast = float(np.abs(component_vals - median_L).mean())
                    score += 0.002 * contrast
                except Exception:
                    pass

            components.append((score, idx, frac))

        if not components:
            return mask_bin

        components.sort(reverse=True)
        filtered = np.zeros_like(mask_bin)
        accumulated = 0.0
        keep_count = 0
        coverage_limit = min(max_total_frac, 0.95)

        for score, idx, frac in components:
            if accumulated + frac > coverage_limit:
                continue
            filtered[labels == idx] = 255
            accumulated += frac
            keep_count += 1

        if accumulated == 0.0:
            # Ensure we keep at least the strongest component so the pipeline still runs
            top_idx = components[0][1]
            filtered[labels == top_idx] = 255
            accumulated = components[0][2]

        if self.debug:
            print(
                f"Mask component filter kept {keep_count} regions covering {accumulated:.3f} (limit {coverage_limit:.3f})"
            )

        return filtered

    def _load_external_mask(self, mask_path: str, reference_shape: Tuple[int, int]) -> np.ndarray:
        if cv2 is None:
            raise RuntimeError(
                "OpenCV is required to use external masks. Install opencv-python-headless.\n"
                f"Original import error: {_CV2_IMPORT_ERROR}"
            )
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise ValueError(f'Failed to read mask file: {mask_path}')
        ref_h, ref_w = reference_shape
        if mask.shape != (ref_h, ref_w):
            mask = cv2.resize(mask, (ref_w, ref_h), interpolation=cv2.INTER_NEAREST)
        mask_bin = (mask > 127).astype(np.uint8) * 255
        return mask_bin

    def _write_mask(self, mask_bin: np.ndarray, dest_path: str) -> str:
        if cv2 is None:
            raise RuntimeError(
                "OpenCV is required to save masks. Install opencv-python-headless.\n"
                f"Original import error: {_CV2_IMPORT_ERROR}"
            )
        mask_bin = (mask_bin > 0).astype(np.uint8) * 255
        os.makedirs(os.path.dirname(dest_path) or '.', exist_ok=True)
        cv2.imwrite(dest_path, mask_bin)
        return dest_path

    def _ensure_mask_file(self, mask_path: str, base_input: str, suffix: str = '_mask.png') -> str:
        if not mask_path:
            return mask_path
        if cv2 is None:
            return mask_path
        img = cv2.imread(base_input, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError('Failed to read input image to validate mask dimensions')
        mask_bin = self._load_external_mask(mask_path, img.shape[:2])
        base_name = os.path.splitext(os.path.basename(base_input))[0]
        temp_path = os.path.join(tempfile.gettempdir(), base_name + suffix)
        return self._write_mask(mask_bin, temp_path)

    def _feather_blend(self, original_bgr: np.ndarray, inpaint_bgr: np.ndarray, mask_bin: np.ndarray, feather_px: int = 10) -> np.ndarray:
        """
        Feather around mask boundaries to avoid halos and visible seams.
        """
        mask01 = (mask_bin > 0).astype(np.uint8)
        if feather_px <= 0:
            return inpaint_bgr
        # Distance transform for feathering outside the mask
        inv = 1 - mask01
        dist = cv2.distanceTransform(inv.astype(np.uint8), cv2.DIST_L2, 3)
        alpha = np.clip(mask01.astype(np.float32) + (1.0 - mask01) * np.clip(dist / float(feather_px), 0.0, 1.0), 0.0, 1.0)
        # Smooth alpha near boundary
        alpha = cv2.GaussianBlur(alpha, (0, 0), sigmaX=1.2)
        alpha = alpha[..., None]
        blend = (inpaint_bgr.astype(np.float32) * alpha + original_bgr.astype(np.float32) * (1.0 - alpha)).astype(np.uint8)
        return blend

    def _denoise_in_mask(self, img_bgr: np.ndarray, mask_bin: np.ndarray, h: int = 3) -> np.ndarray:
        try:
            den = cv2.fastNlMeansDenoisingColored(img_bgr, None, h, h, 7, 21)
            out = img_bgr.copy()
            out[mask_bin > 0] = den[mask_bin > 0]
            return out
        except Exception:
            return img_bgr

    def _two_stage_inpaint(self, img_bgr: np.ndarray, mask_bin: np.ndarray, r_inner: int, r_outer: int, method: str) -> np.ndarray:
        core = self._shrink_mask(mask_bin, iterations=1)
        first = self._inpaint(img_bgr, core, radius=max(2, r_inner), method=method)
        second = self._inpaint(first, mask_bin, radius=max(2, r_outer), method=method)
        return second

    def _unsharp(self, img_bgr: np.ndarray, amount: float = 0.3, sigma: float = 1.0) -> np.ndarray:
        if amount <= 0:
            return img_bgr
        blur = cv2.GaussianBlur(img_bgr, (0, 0), sigmaX=sigma)
        sharp = cv2.addWeighted(img_bgr, 1 + amount, blur, -amount, 0)
        return sharp

    def _apply_inside(self, base_bgr: np.ndarray, variant_bgr: np.ndarray, mask_bin: np.ndarray) -> np.ndarray:
        out = base_bgr.copy()
        m = (mask_bin > 0)
        out[m] = variant_bgr[m]
        return out

    def _inpaint(self, img_bgr: np.ndarray, mask_bin: np.ndarray, radius: int = 3, method: str = 'telea') -> np.ndarray:
        """
        Fast inpainting using OpenCV.
        mask_bin: 0/255 where 255 indicates watermark to remove.
        """
        if cv2 is None:
            raise RuntimeError(
                "OpenCV is not available. Install opencv-python-headless or use --engine ai.\n"
                f"Original import error: {_CV2_IMPORT_ERROR}"
            )
        inpaint_flag = cv2.INPAINT_TELEA if method.lower() == 'telea' else cv2.INPAINT_NS
        # OpenCV expects mask 0/1; convert 255->1
        mask01 = (mask_bin > 0).astype(np.uint8)
        result = cv2.inpaint(img_bgr, mask01, inpaintRadius=radius, flags=inpaint_flag)
        return result

    def process_image_fast(
        self,
        input_path: str,
        output_path: str,
        mask_path: Optional[str] = None,
        save_mask: bool = False,
        inpaint_radius: int = 3,
        inpaint_method: str = 'telea',
        whiteness_thr: float = 0.6,
        darkness_thr: float = 0.6,
        grad_percentile: float = 92.0,
        morph_frac: float = 0.012,
        multiscale: bool = True,
        use_mser: bool = True,
        use_gabor: bool = True,
        use_saliency: bool = True,
        shrink_iter: int = 1,
        feather_px: int = 10,
        two_stage: bool = True,
        sharpen: float = 0.2,
        denoise: bool = True,
    ) -> bool:
        try:
            if cv2 is None:
                raise RuntimeError(
                    "OpenCV import failed. Install opencv-python-headless==4.10.0.84 or use --engine ai.\n"
                    f"Original import error: {_CV2_IMPORT_ERROR}"
                )
            img = cv2.imread(input_path, cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError('Failed to read input image')

            external_mask = False
            if mask_path:
                mask = self._load_external_mask(mask_path, img.shape[:2])
                external_mask = True
            else:
                mask = self._auto_detect_mask(
                    img,
                    debug=self.debug,
                    whiteness_thr=whiteness_thr,
                    darkness_thr=darkness_thr,
                    grad_percentile=grad_percentile,
                    morph_frac=morph_frac,
                    multiscale=multiscale,
                    use_mser=use_mser,
                    use_gabor=use_gabor,
                    use_saliency=use_saliency,
                )
                mask = self._filter_mask_components(
                    mask,
                    img,
                    max_total_frac=float(os.environ.get('WMR_MAX_MASK_COVERAGE', 0.33)),
                    max_component_frac=float(os.environ.get('WMR_MAX_COMPONENT_FRAC', 0.24)),
                    min_elongation=float(os.environ.get('WMR_MIN_COMPONENT_ELONG', 0.18)),
                )
                mask = self._limit_mask_coverage(mask, max_coverage=float(os.environ.get('WMR_MAX_MASK_COVERAGE', 0.33)))

            # Shrink mask a bit to avoid over-inpainting
            shrink_iters = shrink_iter if not external_mask else max(0, shrink_iter)
            if shrink_iters > 0:
                mask = self._shrink_mask(mask, iterations=shrink_iters)
            if save_mask:
                mask_out = os.path.splitext(output_path)[0] + '_mask.png'
                cv2.imwrite(mask_out, mask)
                if self.debug:
                    print(f'Saved detected mask to: {mask_out}')

            if two_stage:
                cleaned = self._two_stage_inpaint(img, mask, r_inner=inpaint_radius, r_outer=inpaint_radius+1, method=inpaint_method)
            else:
                cleaned = self._inpaint(img, mask, radius=max(2, inpaint_radius), method=inpaint_method)

            # Feathered blend to reduce boundary artifacts
            cleaned = self._feather_blend(img, cleaned, mask, feather_px=feather_px)

            # Gentle unsharp mask to recover crispness
            if sharpen > 0:
                sharpened = self._unsharp(cleaned, amount=min(0.6, sharpen), sigma=1.0)
                cleaned = self._apply_inside(cleaned, sharpened, mask)

            # Light denoise inside mask to mitigate blotches
            if denoise:
                cleaned = self._denoise_in_mask(cleaned, mask, h=3)

            ok = cv2.imwrite(output_path, cleaned)
            if not ok:
                raise IOError('Failed to write output image')

            if self.debug:
                print(f"Fast inpaint completed: {input_path} -> {output_path}")
            return True
        except Exception as e:
            if self.debug:
                print(f"Fast path failed: {e}")
            return False

    # -------------------- AI PATH (DIP) --------------------
    def process_image_ai(
        self,
        input_path: str,
        output_path: str,
        mask_path: Optional[str] = None,
        ai_steps: int = 2200,
        ai_dim: int = 768,
        ai_lr: float = 0.008,
        ai_reg_noise: float = 0.02,
        ai_input_depth: int = 64,
        ai_show_step: int = 500,
        ai_loss: str = 'enhanced',
        ai_early_stop: int = 0,
    ) -> bool:
        """
        Process image using DIP-based neural approach (slower but can handle tough cases).
        """
        try:
            # Lazy import to avoid importing torch/numpy when using fast path
            from enhanced_perceptual_api import enhanced_remove_watermark  # noqa: WPS433
            tmp_mask = None
            if mask_path:
                tmp_mask = self._ensure_mask_file(mask_path, input_path, suffix='_ai_mask.png')
            if not tmp_mask or not os.path.exists(tmp_mask):
                tmp_mask = self.generate_temporary_mask(input_path)
            if not tmp_mask or not os.path.exists(tmp_mask):
                tmp_mask = self.create_mask(input_path, output_path)

            enhanced_remove_watermark(
                image_path=input_path,
                mask_path=tmp_mask,
                max_dim=ai_dim,
                show_step=ai_show_step,
                reg_noise=ai_reg_noise,
                input_depth=ai_input_depth,
                lr=ai_lr,
                training_steps=ai_steps,
                loss_type=ai_loss,
                early_stop_patience=ai_early_stop,
            )

            # enhanced_remove_watermark saves as <name>_enhanced_output.png
            output_file = os.path.splitext(os.path.basename(input_path))[0] + '_enhanced_output.png'
            if os.path.exists(output_file) and output_file != output_path:
                os.replace(output_file, output_path)

            if self.debug:
                print(f"AI path completed: {input_path} -> {output_path}")
            return True
        except Exception as e:
            if self.debug:
                print(f"AI path failed: {e}")
            return False

    # -------------------- LAMA PATH (PRETRAINED) --------------------
    def process_image_lama(
        self,
        input_path: str,
        output_path: str,
        lama_model_dir: str = None,
        lama_plugin: str = 'plugins.lama_plugin',
        mask_path: Optional[str] = None,
        inpaint_radius: int = 2,
        inpaint_method: str = 'telea',
        quality_preset: str = 'balanced',
    ) -> bool:
        """
        Use a pre-trained LaMa engine if available via plugin; otherwise fallback to fast inpaint.
        """
        try:
            if cv2 is None:
                raise RuntimeError(
                    "OpenCV import failed. Install opencv-python-headless==4.10.0.84 or install a LaMa plugin.\n"
                    f"Original import error: {_CV2_IMPORT_ERROR}"
                )

            img = cv2.imread(input_path, cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError('Failed to read input image')

            # Reuse detector presets
            presets = {
                'fast': dict(
                    whiteness_thr=None,
                    darkness_thr=None,
                    grad_percentile=85.0,
                    morph_frac=0.018,
                    multiscale=False,
                    use_mser=True,
                    use_gabor=True,
                    use_saliency=False,
                    shrink_iter=0,
                    feather_px=6,
                    max_coverage=0.26,
                    max_component_frac=0.18,
                    min_elongation=0.15,
                    ai_dim=768,
                    ai_lr=0.0075,
                    ai_reg_noise=0.012,
                    ai_input_depth=48,
                    ai_show_step=450,
                    ai_early_stop=80,
                ),
                'balanced': dict(
                    whiteness_thr=None,
                    darkness_thr=None,
                    grad_percentile=90.0,
                    morph_frac=0.015,
                    multiscale=True,
                    use_mser=True,
                    use_gabor=True,
                    use_saliency=True,
                    shrink_iter=1,
                    feather_px=9,
                    max_coverage=0.33,
                    max_component_frac=0.22,
                    min_elongation=0.18,
                    ai_dim=896,
                    ai_lr=0.0065,
                    ai_reg_noise=0.015,
                    ai_input_depth=64,
                    ai_show_step=600,
                    ai_early_stop=120,
                ),
                'high': dict(
                    whiteness_thr=None,
                    darkness_thr=None,
                    grad_percentile=92.0,
                    morph_frac=0.012,
                    multiscale=True,
                    use_mser=True,
                    use_gabor=True,
                    use_saliency=True,
                    shrink_iter=1,
                    feather_px=12,
                    max_coverage=0.36,
                    max_component_frac=0.26,
                    min_elongation=0.20,
                    ai_dim=960,
                    ai_lr=0.006,
                    ai_reg_noise=0.012,
                    ai_input_depth=64,
                    ai_show_step=650,
                    ai_early_stop=150,
                ),
            }
            conf = presets.get(quality_preset, presets['balanced'])

            # Build mask
            if mask_path:
                mask = self._load_external_mask(mask_path, img.shape[:2])
                mask_external = True
            else:
                mask = self._auto_detect_mask(
                    img,
                    debug=self.debug,
                    whiteness_thr=conf['whiteness_thr'],
                    darkness_thr=conf['darkness_thr'],
                    grad_percentile=conf['grad_percentile'],
                    morph_frac=conf['morph_frac'],
                    multiscale=conf['multiscale'],
                    use_mser=conf['use_mser'],
                    use_gabor=conf['use_gabor'],
                    use_saliency=conf['use_saliency'],
                )
                mask = self._filter_mask_components(
                    mask,
                    img,
                    max_total_frac=conf.get('max_coverage', 0.33),
                    max_component_frac=conf.get('max_component_frac', 0.22),
                    min_elongation=conf.get('min_elongation', 0.18),
                )
                mask = self._limit_mask_coverage(mask, max_coverage=conf.get('max_coverage', 0.33))
                mask_external = False

            tmp_mask = self._write_mask(mask, os.path.splitext(output_path)[0] + '_mask.png')

            # Try LaMa plugin
            try:
                import importlib
                plugin = importlib.import_module(lama_plugin)
                ok = bool(plugin.inpaint(input_path, tmp_mask, output_path, model_dir=lama_model_dir, device=None))
                if ok:
                    quality_ok, quality_reason = self._validate_pretrained_result(img, output_path, mask)
                    if quality_ok:
                        if self.debug:
                            print("LaMa plugin completed")
                        return True
                    else:
                        if self.debug:
                            print(f"LaMa plugin result rejected: {quality_reason}; falling back to fast inpaint")
                        ok = False
                else:
                    if self.debug:
                        print("LaMa plugin returned False; falling back to fast inpaint")
            except ModuleNotFoundError:
                if self.debug:
                    print("LaMa plugin not found; falling back to fast inpaint")
            except Exception as e:
                if self.debug:
                    print(f"LaMa plugin failed: {e}; falling back to fast inpaint")

            # Try new pre-trained approach with enhanced models
            if self._can_use_pretrained_models():
                try:
                    ok = self.process_image_pretrained(
                        input_path,
                        output_path,
                        mask_path=tmp_mask,
                        quality_preset=quality_preset
                    )
                    if ok:
                        if self.debug:
                            print("Pre-trained model approach succeeded")
                        return True
                    else:
                        if self.debug:
                            print("Pre-trained model approach returned False, trying AI engine...")
                except Exception as e:
                    if self.debug:
                        print(f"Pre-trained model approach unavailable: {e}; trying AI engine...")

            if self._can_use_ai_engine():
                try:
                    if self.process_image_ai(
                        input_path,
                        output_path,
                        mask_path=tmp_mask,
                        ai_steps=conf.get('ai_steps', 2000),
                        ai_dim=conf.get('ai_dim', 896),
                        ai_lr=conf.get('ai_lr', 0.0065),
                        ai_reg_noise=conf.get('ai_reg_noise', 0.015),
                        ai_input_depth=conf.get('ai_input_depth', 64),
                        ai_show_step=conf.get('ai_show_step', 600),
                        ai_loss='enhanced',
                        ai_early_stop=conf.get('ai_early_stop', 120),
                    ):
                        if self.debug:
                            print("AI fallback completed")
                        return True
                except Exception as e:
                    if self.debug:
                        print(f"AI fallback failed: {e}; proceeding to fast inpaint")

            # Fallback to fast inpaint with blending
            shrink_iters = conf['shrink_iter'] if not mask_external else max(0, conf['shrink_iter'])
            if shrink_iters > 0:
                mask = self._shrink_mask(mask, iterations=shrink_iters)
            cleaned = self._inpaint(img, mask, radius=max(2, inpaint_radius), method=inpaint_method)
            cleaned = self._feather_blend(img, cleaned, mask, feather_px=conf['feather_px'])
            cv2.imwrite(output_path, cleaned)
            return True
        except Exception as e:
            if self.debug:
                print(f"Lama path failed: {e}")
            return False

    # -------------------- NEW PRETRAINED PATH (State-of-the-art models) --------------------
    def _can_use_pretrained_models(self) -> bool:
        """Check if we can use enhanced pre-trained models"""
        try:
            return importlib.util.find_spec('torch') is not None
        except Exception:
            return False

    def process_image_pretrained(
        self,
        input_path: str,
        output_path: str,
        mask_path: Optional[str] = None,
        model_type: str = 'sd-inpaint',  # 'sd-inpaint', 'lama', 'opencv'
        quality_preset: str = 'balanced',
    ) -> bool:
        """
        Process image using state-of-the-art pre-trained models for watermark removal.
        """
        try:
            # Import the pre-trained model implementation
            from pretrained_watermark_removal import remove_watermark_with_pretrained
            
            # Get presets based on quality
            presets = {
                'fast': {
                    'num_inference_steps': 20,
                    'guidance_scale': 7.0,
                },
                'balanced': {
                    'num_inference_steps': 30,
                    'guidance_scale': 7.5,
                },
                'high': {
                    'num_inference_steps': 40,
                    'guidance_scale': 8.0,
                },
            }
            conf = presets.get(quality_preset, presets['balanced'])
            
            # Create temporary mask if not provided
            tmp_mask = None
            if mask_path:
                tmp_mask = self._ensure_mask_file(mask_path, input_path, suffix='_pretrained_mask.png')
            if not tmp_mask or not os.path.exists(tmp_mask):
                tmp_mask = self.generate_temporary_mask(input_path)
            if not tmp_mask or not os.path.exists(tmp_mask):
                tmp_mask = self.create_mask(input_path, output_path)
            
            if not os.path.exists(tmp_mask):
                raise ValueError(f"Could not generate mask for {input_path}")
            
            # Process with pre-trained model
            result = remove_watermark_with_pretrained(
                image_path=input_path,
                mask_path=tmp_mask,
                output_path=output_path,
                model_type=model_type,
            )
            
            if self.debug:
                print(f"Pre-trained model path completed: {input_path} -> {output_path}")
            return True
        except Exception as e:
            if self.debug:
                print(f"Pre-trained path failed: {e}")
            return False

    def _can_use_ai_engine(self) -> bool:
        if os.environ.get('WMR_SKIP_AI_FALLBACK'):
            return False
        try:
            return importlib.util.find_spec('torch') is not None
        except Exception:
            return False

    def _validate_pretrained_result(self, original: np.ndarray, result_path: str, mask: np.ndarray) -> Tuple[bool, str]:
        """Check that the pretrained output stayed sharp and localized to the mask."""
        if cv2 is None:
            return True, "opencv missing; skipping quality gate"

        result = cv2.imread(result_path, cv2.IMREAD_COLOR)
        if result is None:
            return False, "failed to read result image"

        if result.shape != original.shape:
            return False, "result shape mismatch"

        mask_bool = mask > 0
        if not mask_bool.any():
            return True, "mask empty"

        outside_tol = float(os.environ.get("WMR_PRETRAINED_OUTSIDE_L1", 6.0))
        sharp_ratio = float(os.environ.get("WMR_PRETRAINED_SHARPNESS_RATIO", 0.25))

        diff_gray = cv2.cvtColor(cv2.absdiff(result, original), cv2.COLOR_BGR2GRAY)
        inv_mask_bool = ~mask_bool

        if inv_mask_bool.any():
            outside_mean = float(diff_gray[inv_mask_bool].mean())
            if outside_mean > outside_tol:
                return False, f"outside difference {outside_mean:.2f} > {outside_tol}"

        gray_orig = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
        gray_res = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
        lap_orig = cv2.Laplacian(gray_orig, cv2.CV_64F)[mask_bool]
        lap_res = cv2.Laplacian(gray_res, cv2.CV_64F)[mask_bool]

        orig_var = float(lap_orig.var())
        res_var = float(lap_res.var())

        if orig_var > 1e-6 and res_var < sharp_ratio * orig_var:
            return False, f"sharpness dropped ({res_var:.2f} < {sharp_ratio} * {orig_var:.2f})"

        return True, "pass"
    
    def create_mask(self, image_path: str, output_path: str) -> str:
        """
        Create a mask for the watermark locations
        In a real implementation, this would use AI to detect watermarks
        """
        if cv2 is None:
            raise RuntimeError(
                "OpenCV is required to generate the mask. Install opencv-python-headless or use --engine ai.\n"
                f"Original import error: {_CV2_IMPORT_ERROR}"
            )

        img = cv2.imread(image_path)
        if img is None:
            return None

        # Use the fast detector and save a 3-channel mask for DIP compatibility
        mask_bin = self._auto_detect_mask(img, debug=self.debug)
        mask_bin = self._filter_mask_components(
            mask_bin,
            img,
            max_total_frac=float(os.environ.get('WMR_MAX_MASK_COVERAGE', 0.33)),
            max_component_frac=float(os.environ.get('WMR_MAX_COMPONENT_FRAC', 0.24)),
            min_elongation=float(os.environ.get('WMR_MIN_COMPONENT_ELONG', 0.18)),
        )
        mask_bin = self._limit_mask_coverage(mask_bin, max_coverage=float(os.environ.get('WMR_MAX_MASK_COVERAGE', 0.33)))
        mask_rgb = cv2.cvtColor(mask_bin, cv2.COLOR_GRAY2BGR)

        mask_path = (
            image_path.replace('.png', '_mask.png')
            .replace('.jpg', '_mask.jpg')
            .replace('.jpeg', '_mask.jpg')
        )
        cv2.imwrite(mask_path, mask_rgb)
        return mask_path
    
    def generate_temporary_mask(self, image_path: str) -> str:
        """
        Generate a temporary mask based on image analysis
        """
        if cv2 is None:
            raise RuntimeError(
                "OpenCV is required to generate the mask. Install opencv-python-headless or use --engine ai.\n"
                f"Original import error: {_CV2_IMPORT_ERROR}"
            )

        img = cv2.imread(image_path)
        if img is None:
            return None

        mask_bin = self._auto_detect_mask(img, debug=self.debug)
        mask_bin = self._filter_mask_components(
            mask_bin,
            img,
            max_total_frac=float(os.environ.get('WMR_MAX_MASK_COVERAGE', 0.33)),
            max_component_frac=float(os.environ.get('WMR_MAX_COMPONENT_FRAC', 0.24)),
            min_elongation=float(os.environ.get('WMR_MIN_COMPONENT_ELONG', 0.18)),
        )
        mask_bin = self._limit_mask_coverage(mask_bin, max_coverage=float(os.environ.get('WMR_MAX_MASK_COVERAGE', 0.33)))
        mask_rgb = cv2.cvtColor(mask_bin, cv2.COLOR_GRAY2BGR)

        temp_mask_path = (
            image_path.replace('.png', '_temp_mask.png')
            .replace('.jpg', '_temp_mask.jpg')
            .replace('.jpeg', '_temp_mask.jpg')
        )
        cv2.imwrite(temp_mask_path, mask_rgb)
        return temp_mask_path

    def batch_process(self, input_dir: str, output_dir: str, engine: str = 'fast') -> List[str]:
        """
        Process multiple images in a directory
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        supported_formats = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
        processed_files = []
        
        for filename in os.listdir(input_dir):
            if filename.lower().endswith(supported_formats):
                input_path = os.path.join(input_dir, filename)
                output_path = os.path.join(output_dir, f"cleaned_{filename}")
                
                if engine == 'ai':
                    ok = self.process_image_ai(input_path, output_path)
                else:
                    ok = self.process_image_fast(input_path, output_path, save_mask=False)
                if ok:
                    processed_files.append(output_path)
        
        return processed_files


def main():
    parser = argparse.ArgumentParser(description='Automatic Watermark Removal Tool')
    parser.add_argument('input', help='Input image path or directory', nargs='?', default='image.png')
    parser.add_argument('output', help='Output image path or directory', nargs='?', default='output.png')
    parser.add_argument('--batch', action='store_true', help='Batch process a directory')
    parser.add_argument('--engine', choices=['fast', 'ai', 'lama', 'pretrained'], default='fast', help='Removal engine to use')
    parser.add_argument('--mask', type=str, default=None, help='Path to external mask image (white pixels will be removed)')
    parser.add_argument('--save-mask', action='store_true', help='Save detected mask alongside output (fast engine)')
    parser.add_argument('--radius', type=int, default=3, help='Inpaint radius for fast engine')
    parser.add_argument('--method', choices=['telea', 'ns'], default='telea', help='Inpaint method for fast engine')
    parser.add_argument('--quality', choices=['fast', 'balanced', 'high'], default='high', help='Preset for detection aggressiveness and blending (fast engine)')
    parser.add_argument('--feather', type=int, default=None, help='Feather width in pixels at boundaries (fast engine)')
    parser.add_argument('--no-two-stage', action='store_true', help='Disable two-stage inpainting (fast engine)')
    parser.add_argument('--sharpen', type=float, default=None, help='Unsharp amount applied after inpainting (0-0.6)')
    parser.add_argument('--out-dir', type=str, default=None, help='Directory for outputs (overridden by explicit output path)')
    parser.add_argument('--debug-dir', type=str, default=None, help='Directory for debug masks when --debug is set')
    parser.add_argument('--no-multiscale', action='store_true', help='Disable multiscale detection (fast engine)')
    parser.add_argument('--no-mser', action='store_true', help='Disable MSER text detection (fast engine)')
    parser.add_argument('--no-denoise', action='store_true', help='Disable post-inpaint denoising (fast engine)')
    parser.add_argument('--no-gabor', action='store_true', help='Disable Gabor text detector (fast engine)')
    parser.add_argument('--no-saliency', action='store_true', help='Disable spectral saliency (fast engine)')
    parser.add_argument('--shrink', type=int, default=None, help='Erode mask N iterations before inpaint (fast engine)')
    parser.add_argument('--wthr', type=float, default=None, help='Whiteness threshold override 0..1 (fast engine)')
    parser.add_argument('--dthr', type=float, default=None, help='Darkness threshold override 0..1 (fast engine)')
    parser.add_argument('--gperc', type=float, default=None, help='Gradient percentile 70..98 (fast engine)')

    # AI engine hyperparameters
    parser.add_argument('--ai-steps', type=int, default=2200, help='Training steps for AI engine')
    parser.add_argument('--ai-dim', type=int, default=768, help='Max dimension for AI engine')
    parser.add_argument('--ai-lr', type=float, default=0.008, help='Learning rate for AI engine')
    parser.add_argument('--ai-reg-noise', type=float, default=0.02, help='Reg noise for AI engine')
    parser.add_argument('--ai-input-depth', type=int, default=64, help='Input depth for AI engine')
    parser.add_argument('--ai-show-step', type=int, default=500, help='Visualization step for AI engine')
    parser.add_argument('--ai-loss', choices=['enhanced','multiscale','mse'], default='enhanced', help='Loss type for AI engine')
    parser.add_argument('--ai-early-stop', type=int, default=0, help='Early stop patience (0 disables) for AI engine')

    # LaMa engine options
    parser.add_argument('--lama-model-dir', type=str, default=None, help='Directory containing LaMa weights (plugin dependent)')
    parser.add_argument('--lama-plugin', type=str, default='plugins.lama_plugin', help='Python module implementing LaMa plugin interface')
    parser.add_argument('--debug', action='store_true', help='Enable debug output')

    args = parser.parse_args()

    remover = WatermarkRemover(debug=args.debug)

    if args.batch:
        if not os.path.isdir(args.input):
            print('Input must be a directory when using --batch')
            return
        if not os.path.exists(args.output):
            os.makedirs(args.output)
        processed = remover.batch_process(args.input, args.output, engine=args.engine)
        print(f"Processed {len(processed)} images")
        for file in processed:
            print(f"  -> {file}")
        return

    # Single
    # If out-dir is provided and output is default, place result there
    output_path = args.output
    if args.out_dir and args.output == 'output.png':
        os.makedirs(args.out_dir, exist_ok=True)
        base = os.path.basename(args.input)
        name, _ = os.path.splitext(base)
        output_path = os.path.join(args.out_dir, f"{name}_cleaned.png")

    # Set debug dir for saving debug masks
    if args.debug and args.debug_dir:
        os.makedirs(args.debug_dir, exist_ok=True)
        # Redirect debug images into this folder via environment variable read in _auto_detect_mask
        os.environ['WMR_DEBUG_DIR'] = args.debug_dir

    if args.engine == 'pretrained':
        success = remover.process_image_pretrained(
            args.input,
            output_path,
            mask_path=args.mask,
            model_type='sd-inpaint',  # Default to SD inpaint
            quality_preset=args.quality,
        )
    elif args.engine == 'ai':
        success = remover.process_image_ai(
            args.input,
            output_path,
            mask_path=args.mask,
            ai_steps=args.ai_steps,
            ai_dim=args.ai_dim,
            ai_lr=args.ai_lr,
            ai_reg_noise=args.ai_reg_noise,
            ai_input_depth=args.ai_input_depth,
            ai_show_step=args.ai_show_step,
            ai_loss=args.ai_loss,
            ai_early_stop=args.ai_early_stop,
        )
    elif args.engine == 'lama':
        success = remover.process_image_lama(
            args.input,
            output_path,
            lama_model_dir=args.lama_model_dir,
            lama_plugin=args.lama_plugin,
            mask_path=args.mask,
            inpaint_radius=args.radius,
            inpaint_method=args.method,
            quality_preset=args.quality,
        )
    else:
        # Quality presets
        presets = {
            'fast': dict(whiteness_thr=None, darkness_thr=None, grad_percentile=85.0, morph_frac=0.018, multiscale=False, use_mser=True, use_gabor=True, use_saliency=False, shrink_iter=0, feather_px=6, two_stage=False, sharpen=0.12, denoise=False),
            'balanced': dict(whiteness_thr=None, darkness_thr=None, grad_percentile=90.0, morph_frac=0.015, multiscale=True, use_mser=True, use_gabor=True, use_saliency=True, shrink_iter=1, feather_px=9, two_stage=True, sharpen=0.18, denoise=True),
            'high': dict(whiteness_thr=None, darkness_thr=None, grad_percentile=92.0, morph_frac=0.012, multiscale=True, use_mser=True, use_gabor=True, use_saliency=True, shrink_iter=1, feather_px=12, two_stage=True, sharpen=0.22, denoise=True),
        }
        conf = presets.get(args.quality, presets['high'])
        # Allow manual feather override
        if args.feather is not None:
            conf['feather_px'] = args.feather
        if args.no_two_stage:
            conf['two_stage'] = False
        if args.sharpen is not None:
            conf['sharpen'] = args.sharpen
        if args.no_multiscale:
            conf['multiscale'] = False
        if args.no_mser:
            conf['use_mser'] = False
        if args.no_denoise:
            conf['denoise'] = False
        if args.no_gabor:
            conf['use_gabor'] = False
        if args.no_saliency:
            conf['use_saliency'] = False

        # Manual overrides
        if args.shrink is not None:
            conf['shrink_iter'] = max(0, int(args.shrink))
        if args.wthr is not None:
            conf['whiteness_thr'] = float(args.wthr)
        if args.dthr is not None:
            conf['darkness_thr'] = float(args.dthr)
        if args.gperc is not None:
            conf['grad_percentile'] = float(args.gperc)

        success = remover.process_image_fast(
            args.input,
            output_path,
            mask_path=args.mask,
            save_mask=args.save_mask,
            inpaint_radius=args.radius,
            inpaint_method=args.method,
            **conf,
        )
    if success:
        print(f"Successfully processed: {args.input} -> {output_path}")
    else:
        print(f"Failed to process: {args.input}")


if __name__ == "__main__":
    main()


def create_default_mask(image_path):
    """
    Back-compat utility: generate an auto mask using the fast detector.
    """
    if cv2 is None:
        raise RuntimeError(
            "OpenCV is required for default mask. Install opencv-python-headless or use --engine ai.\n"
            f"Original import error: {_CV2_IMPORT_ERROR}"
        )
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError('Failed to read image for default mask')
    remover = WatermarkRemover()
    mask_bin = remover._auto_detect_mask(img)
    mask_rgb = cv2.cvtColor(mask_bin, cv2.COLOR_GRAY2BGR)
    mask_path = 'mask.png'
    cv2.imwrite(mask_path, mask_rgb)
    return mask_path
