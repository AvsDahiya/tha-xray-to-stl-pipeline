#!/usr/bin/env python3
"""Step 8: Run full evaluation and generate metrics/figures."""

import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))

import argparse
from pathlib import Path

from thapipeline.config import PipelineConfig, get_device
from thapipeline.eval.evaluate_full_pipeline import evaluate_full_pipeline
from thapipeline.models.segmenter import ImplantSegmenter

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--tag", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--pairs-csv", type=str, default=None)
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument(
        "--segmentation-mode",
        choices=["combined", "classical", "mlp", "unet"],
        default="combined",
    )
    parser.add_argument(
        "--disable-reprojection-opt",
        action="store_true",
        help="Disable silhouette reprojection optimisation for ablation runs.",
    )
    args = parser.parse_args()

    config = PipelineConfig()
    device = args.device or get_device()
    pairs_csv = Path(args.pairs_csv) if args.pairs_csv else config.paths.pairing_table

    # Init segmenter
    seg_mlp = config.paths.segmenter_dir / "pixel_mlp.pt"
    seg_unet = config.paths.segmenter_dir / "unet_fallback.pt"
    segmenter = ImplantSegmenter(
        device=device,
        mlp_checkpoint=seg_mlp if seg_mlp.exists() else None,
        unet_checkpoint=seg_unet if seg_unet.exists() else None,
    )

    # Find best checkpoint
    if args.checkpoint:
        ckpt = Path(args.checkpoint)
    else:
        ckpt = config.paths.pix2pix_dir / "ssim_l10" / "best_model.pt"
        if not ckpt.exists():
            ckpt = config.paths.pix2pix_dir / "baseline" / "best_model.pt"

    tag = args.tag or f"test_evaluation_{ckpt.parent.name}"
    summary = evaluate_full_pipeline(
        config,
        ckpt,
        segmenter,
        device,
        output_name=tag,
        segmentation_mode=args.segmentation_mode,
        optimize_reprojection=not args.disable_reprojection_opt,
        pairs_csv=pairs_csv,
        evaluation_split=args.split,
    )
    print("\nEvaluation complete!")
