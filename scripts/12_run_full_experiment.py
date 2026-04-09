#!/usr/bin/env python3
"""Step 12: Run the dissertation workflow as a restartable one-shot pipeline."""

import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))

import argparse

from thapipeline.config import PipelineConfig, get_device
from thapipeline.orchestration.full_experiment import run_full_experiment


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the dissertation workflow end to end")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--tag", type=str, default="d1_full")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--grad-accum-steps", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--checkpoint-every", type=int, default=None)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--skip-existing", dest="skip_existing", action="store_true", default=True)
    parser.add_argument("--no-skip-existing", dest="skip_existing", action="store_false")
    parser.add_argument(
        "--stop-after",
        choices=["curate", "pair", "preprocess", "gan", "evaluation", "segmentation", "reconstruction", "ablation"],
        default=None,
    )
    parser.add_argument(
        "--force-stage",
        action="append",
        choices=["curate", "pair", "preprocess", "train", "evaluate", "segment", "reconstruct", "ablation"],
        default=[],
    )
    parser.add_argument("--final-ssim-weights", type=float, nargs="*", default=[5, 10, 20])
    parser.add_argument("--disable-reprojection-opt", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--notes", type=str, default="")
    parser.add_argument("--logic-change-note", type=str, default="")
    parser.add_argument("--run-prefix", type=str, default=None)
    args = parser.parse_args()

    config = PipelineConfig()
    if args.batch_size is not None:
        config.training.batch_size = args.batch_size
    if args.grad_accum_steps is not None:
        config.training.grad_accum_steps = max(1, args.grad_accum_steps)
    if args.epochs is not None:
        config.training.epochs = args.epochs
    if args.checkpoint_every is not None:
        config.training.checkpoint_every = max(1, args.checkpoint_every)
    if args.patience is not None:
        config.training.patience = args.patience
    if args.num_workers is not None:
        config.training.num_workers = args.num_workers

    device = args.device or get_device()
    if device == "mps" and args.num_workers is None:
        config.training.num_workers = 0

    manifest = run_full_experiment(
        config,
        device=device,
        tag=args.tag,
        skip_existing=args.skip_existing,
        stop_after=args.stop_after,
        force_stages=args.force_stage,
        final_ssim_weights=args.final_ssim_weights,
        disable_reprojection_opt=args.disable_reprojection_opt,
        dry_run=args.dry_run,
        notes=args.notes,
        logic_change_note=args.logic_change_note,
        run_prefix=args.run_prefix,
    )

    print("One-shot workflow complete.")
    print(manifest["status"])
