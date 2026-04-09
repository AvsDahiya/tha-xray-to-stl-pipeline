#!/usr/bin/env python3
"""Step 9: Run ablation studies.

Compares model variants and generates comparison tables/charts:
  1. Loss ablation: baseline vs SSIM
  2. Segmentation method comparison
"""

import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))

from thapipeline.config import PipelineConfig
from thapipeline.eval.ablation_runner import (
    run_loss_ablation_analysis,
    run_segmentation_ablation_analysis,
    run_ssim_weight_ablation_analysis,
)
from thapipeline.utils.io import load_json

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--loss-tags",
        nargs=2,
        metavar=("BASELINE_TAG", "SSIM_TAG"),
        default=("test_evaluation_baseline", "test_evaluation_ssim_l10"),
        help="Evaluation tags for baseline and SSIM loss comparison.",
    )
    parser.add_argument(
        "--ssim-weight-tags",
        nargs="*",
        default=[],
        help="Pairs of TAG=WEIGHT for SSIM-weight ablations, e.g. eval_l0=0 eval_l5=5.",
    )
    parser.add_argument(
        "--segmentation-tags",
        nargs="*",
        default=[],
        help="Pairs of METHOD=TAG for segmentation ablations, e.g. classical=eval_classical.",
    )
    args = parser.parse_args()

    config = PipelineConfig()

    output_dir = config.paths.metrics_dir / "ablation"
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Loss Ablation ───────────────────────────────────────────────────
    baseline_eval = config.paths.metrics_dir / args.loss_tags[0] / "evaluation_summary.json"
    ssim_eval = config.paths.metrics_dir / args.loss_tags[1] / "evaluation_summary.json"
    baseline_cases = config.paths.metrics_dir / args.loss_tags[0] / "case_metrics.csv"
    ssim_cases = config.paths.metrics_dir / args.loss_tags[1] / "case_metrics.csv"

    if baseline_eval.exists() and ssim_eval.exists():
        baseline_results = load_json(baseline_eval)
        ssim_results = load_json(ssim_eval)
        case_metrics_paths = {}
        if baseline_cases.exists() and ssim_cases.exists():
            case_metrics_paths = {
                "L_cGAN + L1": baseline_cases,
                "L_cGAN + L1 + L_SSIM": ssim_cases,
            }
        run_loss_ablation_analysis(
            baseline_results,
            ssim_results,
            output_dir,
            case_metrics_paths=case_metrics_paths or None,
        )
        print("Loss ablation analysis complete.")
    else:
        print("Skipping loss ablation (need both baseline and SSIM evaluation results)")

    if args.ssim_weight_tags:
        by_weight = {}
        case_paths = {}
        for item in args.ssim_weight_tags:
            if "=" not in item:
                continue
            tag, weight = item.split("=", 1)
            summary_path = config.paths.metrics_dir / tag / "evaluation_summary.json"
            case_path = config.paths.metrics_dir / tag / "case_metrics.csv"
            if summary_path.exists():
                by_weight[float(weight)] = load_json(summary_path)
                if case_path.exists():
                    case_paths[f"λ₂={weight}"] = case_path
        if by_weight:
            run_ssim_weight_ablation_analysis(
                by_weight,
                output_dir,
                case_metrics_paths=case_paths or None,
            )
            print("SSIM-weight ablation analysis complete.")
        else:
            print("Skipping SSIM-weight ablation (no valid evaluation summaries found)")

    if args.segmentation_tags:
        by_method = {}
        case_paths = {}
        for item in args.segmentation_tags:
            if "=" not in item:
                continue
            method, tag = item.split("=", 1)
            summary_path = config.paths.metrics_dir / tag / "evaluation_summary.json"
            case_path = config.paths.metrics_dir / tag / "case_metrics.csv"
            if summary_path.exists():
                by_method[method] = load_json(summary_path)
                if case_path.exists():
                    case_paths[method] = case_path
        if by_method:
            run_segmentation_ablation_analysis(
                by_method,
                output_dir,
                case_metrics_paths=case_paths or None,
            )
            print("Segmentation ablation analysis complete.")
        else:
            print("Skipping segmentation ablation (no valid evaluation summaries found)")

    print("\nAblation studies complete!")
