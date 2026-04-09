#!/usr/bin/env python3
"""Step 4: Train Pix2Pix cGAN.

Trains both baseline (L_cGAN + L1) and full (L_cGAN + L1 + L_SSIM) models.
Saves checkpoints, training curves, and sample images.
"""

import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))

import argparse
from pathlib import Path

from thapipeline.config import PipelineConfig, get_device
from thapipeline.training.train_pix2pix import Pix2PixTrainer
from thapipeline.utils.io import best_resume_checkpoint
from thapipeline.utils.vis import plot_training_curves


def main():
    parser = argparse.ArgumentParser(description="Train Pix2Pix cGAN")
    parser.add_argument("--mode", choices=["baseline", "ssim", "both"], default="both",
                        help="Training mode: baseline (no SSIM), ssim, or both")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override number of epochs")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from")
    parser.add_argument("--resume-latest", action="store_true",
                        help="Resume from the latest epoch checkpoint for the resolved run name.")
    parser.add_argument("--device", type=str, default=None,
                        help="Force device (cuda, mps, cpu)")
    parser.add_argument("--pairs-csv", type=str, default=None,
                        help="Optional alternate pairing_table.csv path.")
    parser.add_argument("--tag", type=str, default=None,
                        help="Optional experiment tag. If omitted, uses baseline/ssim_l<lambda>.")
    parser.add_argument("--notes", type=str, default="",
                        help="Free-text experiment note saved in the manifest and registry.")
    parser.add_argument("--logic-change-note", type=str, default="",
                        help="Free-text note describing the code or logic variation under test.")
    parser.add_argument("--lambda-ssim", type=float, default=None,
                        help="Override SSIM loss weight.")
    parser.add_argument("--lambda-l1", type=float, default=None,
                        help="Override L1 loss weight.")
    parser.add_argument("--lr-g", type=float, default=None,
                        help="Override generator learning rate.")
    parser.add_argument("--lr-d", type=float, default=None,
                        help="Override discriminator learning rate.")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Override batch size.")
    parser.add_argument("--num-workers", type=int, default=None,
                        help="Override dataloader workers.")
    parser.add_argument("--patience", type=int, default=None,
                        help="Override early stopping patience.")
    parser.add_argument("--grad-accum-steps", type=int, default=None,
                        help="Accumulate gradients across this many mini-batches.")
    parser.add_argument("--grad-clip-norm", type=float, default=None,
                        help="Optional gradient clipping norm (0 disables clipping).")
    parser.add_argument("--sample-every", type=int, default=None,
                        help="Override qualitative sample interval.")
    parser.add_argument("--checkpoint-every", type=int, default=None,
                        help="Override checkpoint interval.")
    parser.add_argument("--warmup-epochs", type=int, default=None,
                        help="Override LR warmup period before decay.")
    parser.add_argument("--decay-step", type=int, default=None,
                        help="Override LR decay step size.")
    parser.add_argument("--decay-factor", type=float, default=None,
                        help="Override LR decay factor.")
    args = parser.parse_args()

    config = PipelineConfig()
    if args.epochs:
        config.training.epochs = args.epochs
    if args.lambda_ssim is not None:
        config.training.lambda_SSIM = args.lambda_ssim
    if args.lambda_l1 is not None:
        config.training.lambda_L1 = args.lambda_l1
    if args.lr_g is not None:
        config.training.lr_G = args.lr_g
    if args.lr_d is not None:
        config.training.lr_D = args.lr_d
    if args.batch_size is not None:
        config.training.batch_size = args.batch_size
    if args.num_workers is not None:
        config.training.num_workers = args.num_workers
    if args.patience is not None:
        config.training.patience = args.patience
    if args.grad_accum_steps is not None:
        config.training.grad_accum_steps = max(1, args.grad_accum_steps)
    if args.grad_clip_norm is not None:
        config.training.grad_clip_norm = max(0.0, args.grad_clip_norm)
    if args.sample_every is not None:
        config.training.sample_every = max(1, args.sample_every)
    if args.checkpoint_every is not None:
        config.training.checkpoint_every = max(1, args.checkpoint_every)
    if args.warmup_epochs is not None:
        config.training.warmup_epochs = max(0, args.warmup_epochs)
    if args.decay_step is not None:
        config.training.decay_step = max(1, args.decay_step)
    if args.decay_factor is not None:
        config.training.decay_factor = args.decay_factor

    device = args.device or get_device()
    pairs_csv = Path(args.pairs_csv) if args.pairs_csv else None
    print(f"Using device: {device}")
    if device == "mps" and args.num_workers is None:
        config.training.num_workers = 0
        print("MPS detected: forcing num_workers=0 for safer dataloader behaviour.")

    if args.mode in ("baseline", "both"):
        print("\n" + "=" * 60)
        print("TRAINING BASELINE MODEL (L_cGAN + L1)")
        print("=" * 60)
        baseline_name = f"{args.tag}_baseline" if args.tag else "baseline"
        trainer = Pix2PixTrainer(
            config,
            use_ssim=False,
            device=device,
            experiment_name=baseline_name,
            notes=args.notes,
            logic_change_note=args.logic_change_note,
            pairs_csv=pairs_csv,
        )
        resume = Path(args.resume) if args.resume else None
        if resume is None and args.resume_latest:
            resume = best_resume_checkpoint(config.paths.pix2pix_dir / baseline_name, device=device)
            if resume is None:
                print(f"No valid checkpoint found for {baseline_name}; starting from scratch.")
            else:
                print(f"Resuming baseline from checkpoint: {resume}")
        history_baseline = trainer.train(resume_path=resume)
        plot_training_curves(
            history_baseline,
            config.paths.figures_dir / f"training_curves_{baseline_name}.png",
        )

    if args.mode in ("ssim", "both"):
        print("\n" + "=" * 60)
        print("TRAINING FULL MODEL (L_cGAN + L1 + L_SSIM)")
        print("=" * 60)
        lambda_label = str(int(config.training.lambda_SSIM)) if float(config.training.lambda_SSIM).is_integer() else str(config.training.lambda_SSIM).replace(".", "p")
        ssim_name = f"{args.tag}_ssim_l{lambda_label}" if args.tag else f"ssim_l{lambda_label}"
        trainer = Pix2PixTrainer(
            config,
            use_ssim=True,
            device=device,
            experiment_name=ssim_name,
            notes=args.notes,
            logic_change_note=args.logic_change_note,
            pairs_csv=pairs_csv,
        )
        resume = Path(args.resume) if args.resume else None
        if resume is None and args.resume_latest:
            resume = best_resume_checkpoint(config.paths.pix2pix_dir / ssim_name, device=device)
            if resume is None:
                print(f"No valid checkpoint found for {ssim_name}; starting from scratch.")
            else:
                print(f"Resuming SSIM run from checkpoint: {resume}")
        history_ssim = trainer.train(resume_path=resume)
        plot_training_curves(
            history_ssim,
            config.paths.figures_dir / f"training_curves_{ssim_name}.png",
        )


if __name__ == "__main__":
    main()
