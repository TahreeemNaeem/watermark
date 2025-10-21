
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
    # Add argument for specifying model type
    ap.add_argument("--model", choices=["lama", "sd-inpaint", "lama-standalone", "fast", "ai"], 
                   default="sd-inpaint", help="Model type to use for watermark removal")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    remover = WatermarkRemover(debug=args.debug)

    # 1) Try state-of-the-art pre-trained models (new approach)
    try:
        if args.model in ["sd-inpaint", "lama-standalone"]:
            ok = remover.process_image_pretrained(
                args.input,
                args.output,
                mask_path=args.mask,
                model_type=args.model if args.model != "lama-standalone" else "lama",
                quality_preset=args.quality,
            )
            if ok:
                print("✓ State-of-the-art pre-trained model succeeded")
                return 0
            else:
                print("… Pre-trained model returned False, trying lama plugin…")
        else:
            # If not using our new models, try lama plugin
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
                print("… Pretrained engine returned False, trying state-of-the-art pre-trained model…")
    except Exception as e:
        print(f"… Pretrained approach unavailable: {e}\nTrying AI engine…")

    # 2) Try state-of-the-art pre-trained models as fallback if not tried already
    if args.model != "sd-inpaint" and args.model != "lama-standalone":
        try:
            ok = remover.process_image_pretrained(
                args.input,
                args.output,
                mask_path=args.mask,
                model_type='sd-inpaint',
                quality_preset=args.quality,
            )
            if ok:
                print("✓ State-of-the-art pre-trained model succeeded")
                return 0
            else:
                print("… Pre-trained model returned False, trying AI engine…")
        except Exception as e:
            print(f"… State-of-the-art pre-trained model unavailable: {e}\nTrying AI engine…")

    # 3) Fall back to AI DIP
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

    # 4) Last resort: fast inpainting
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
