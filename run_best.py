#!/usr/bin/env python3
"""
One solid solution runner.

Tries the pre-trained engine (lama plugin using SD Inpaint) for best clarity.
If unavailable, falls back to the AI DIP engine. As a last resort, uses the
fast OpenCV inpainting path.

Usage:
  python run_best.py --input image.png --output outputs/final_best.png

Optional env knobs (for pretrained engine):
  WMR_SD_ERODE=2 WMR_SD_FEATHER=14 WMR_SD_STRENGTH=0.7 WMR_SD_GUIDANCE=6.5
"""
import argparse
import os
import sys

from ws import WatermarkRemover


def main():
    ap = argparse.ArgumentParser(description="Run best-quality watermark removal")
    ap.add_argument("--input", default="image.png", help="Input image path")
    ap.add_argument("--output", default="outputs/final_best.png", help="Output image path")
    ap.add_argument("--quality", choices=["fast", "balanced", "high"], default="balanced")
    ap.add_argument("--mask", help="Optional external mask (white = remove)")
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    remover = WatermarkRemover(debug=args.debug)

    # 1) Try pretrained (lama plugin / SD inpaint)
    try:
        ok = remover.process_image_lama(
            args.input,
            args.output,
            lama_model_dir=os.environ.get("WMR_SD_MODEL_DIR"),
            lama_plugin=os.environ.get("WMR_LAMA_PLUGIN", "plugins.lama_plugin"),
            mask_path=args.mask,
            quality_preset=args.quality,
        )
        if ok:
            print("✓ Pretrained engine succeeded")
            return 0
        else:
            print("… Pretrained engine returned False, trying AI engine…")
    except Exception as e:
        print(f"… Pretrained engine unavailable: {e}\nTrying AI engine…")

    # 2) Fall back to AI DIP
    try:
        ok = remover.process_image_ai(
            args.input,
            args.output,
            mask_path=args.mask,
            ai_steps=1800,
            ai_dim=896,
            ai_lr=0.0065,
            ai_reg_noise=0.015,
            ai_input_depth=64,
            ai_show_step=600,
            ai_loss="enhanced",
            ai_early_stop=200,
        )
        if ok:
            print("✓ AI DIP engine succeeded")
            return 0
        else:
            print("… AI engine returned False, trying fast engine…")
    except Exception as e:
        print(f"… AI engine unavailable: {e}\nTrying fast engine…")

    # 3) Last resort: fast inpainting
    ok = remover.process_image_fast(
        args.input,
        args.output,
        mask_path=args.mask,
        save_mask=True,
        inpaint_radius=2,
        inpaint_method="telea",
        whiteness_thr=None,
        darkness_thr=None,
        grad_percentile=90.0,
        morph_frac=0.015,
        multiscale=True,
        use_mser=True,
        use_gabor=False,
        use_saliency=False,
        shrink_iter=2,
        feather_px=10,
        two_stage=True,
        sharpen=0.18,
        denoise=False,
    )
    if ok:
        print("✓ Fast engine succeeded")
        return 0
    print("✗ All engines failed; check your environment and inputs.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
