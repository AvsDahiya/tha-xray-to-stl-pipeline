#!/usr/bin/env python3
"""Step 10: Run leakage-safe 5-fold cross-validation."""

import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))

import argparse

from thapipeline.config import PipelineConfig, get_device
from thapipeline.eval.cross_validation import run_pix2pix_cross_validation


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run 5-fold Pix2Pix cross-validation")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--mode", choices=["baseline", "ssim", "both"], default="both")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--grad-accum-steps", type=int, default=None)
    parser.add_argument("--grad-clip-norm", type=float, default=None)
    parser.add_argument("--checkpoint-every", type=int, default=None)
    parser.add_argument("--lr-g", type=float, default=None)
    parser.add_argument("--lr-d", type=float, default=None)
    parser.add_argument("--lambda-ssim", type=float, default=None)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--tag", type=str, default="cv5")
    parser.add_argument("--fold-indices", type=int, nargs="*", default=None)
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument("--force-train", action="store_true")
    parser.add_argument("--notes", type=str, default="")
    parser.add_argument("--logic-change-note", type=str, default="")
    parser.add_argument(
        "--segmentation-mode",
        choices=["combined", "classical", "mlp", "unet"],
        default="combined",
    )
    parser.add_argument(
        "--disable-reprojection-opt",
        action="store_true",
        help="Disable silhouette reprojection optimisation for ablation CV runs.",
    )
    args = parser.parse_args()

    config = PipelineConfig()
    if args.epochs is not None:
        config.training.epochs = args.epochs
    if args.batch_size is not None:
        config.training.batch_size = args.batch_size
    if args.num_workers is not None:
        config.training.num_workers = args.num_workers
    if args.grad_accum_steps is not None:
        config.training.grad_accum_steps = max(1, args.grad_accum_steps)
    if args.grad_clip_norm is not None:
        config.training.grad_clip_norm = max(0.0, args.grad_clip_norm)
    if args.checkpoint_every is not None:
        config.training.checkpoint_every = max(1, args.checkpoint_every)
    if args.lr_g is not None:
        config.training.lr_G = args.lr_g
    if args.lr_d is not None:
        config.training.lr_D = args.lr_d
    if args.lambda_ssim is not None:
        config.training.lambda_SSIM = args.lambda_ssim
    if args.patience is not None:
        config.training.patience = args.patience

    device = args.device or get_device()
    if device == "mps" and args.num_workers is None:
        config.training.num_workers = 0

    overview = run_pix2pix_cross_validation(
        config,
        n_folds=args.folds,
        mode=args.mode,
        device=device,
        fold_indices=args.fold_indices,
        tag=args.tag,
        skip_train=args.skip_train,
        skip_eval=args.skip_eval,
        force_train=args.force_train,
        segmentation_mode=args.segmentation_mode,
        optimize_reprojection=not args.disable_reprojection_opt,
        notes=args.notes,
        logic_change_note=args.logic_change_note,
    )

    print("\nCross-validation complete.")
    print(overview)
