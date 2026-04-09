"""Ablation study runner for systematic comparison of model variants.

Studies:
  1. Loss ablation: L_cGAN + L1 vs L_cGAN + L1 + L_SSIM
  2. SSIM weight ablation: λ₂ ∈ {0, 5, 10, 20}
  3. Segmentation comparison: Classical vs MLP vs U-Net vs Combined
  4. Reconstruction: With vs without reprojection optimisation
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from thapipeline.eval.statistics import paired_ttests_from_case_metrics


def generate_ablation_table(
    results: Dict[str, Dict],
    output_dir: Path,
    study_name: str,
) -> pd.DataFrame:
    """Generate comparison table for ablation results.

    Args:
        results: Dict mapping variant name to metrics dict.
        output_dir: Where to save table.
        study_name: Name for the output file.

    Returns:
        DataFrame with comparison.
    """
    rows = []
    for variant, metrics in results.items():
        row = {"Variant": variant}
        for metric_name, vals in metrics.items():
            if isinstance(vals, dict) and "mean" in vals:
                row[f"{metric_name}_mean"] = vals["mean"]
                row[f"{metric_name}_std"] = vals["std"]
                row[f"{metric_name}_ci95_low"] = vals.get("ci95_low")
                row[f"{metric_name}_ci95_high"] = vals.get("ci95_high")
            else:
                row[metric_name] = vals
        rows.append(row)

    df = pd.DataFrame(rows)
    csv_path = output_dir / f"ablation_{study_name}.csv"
    df.to_csv(csv_path, index=False)
    print(f"Ablation table saved to {csv_path}")
    return df


def plot_ablation_comparison(
    results: Dict[str, Dict],
    metric_key: str,
    output_dir: Path,
    study_name: str,
    ylabel: str = "",
) -> None:
    """Generate bar chart comparing variants on a specific metric."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    variants = list(results.keys())
    means = []
    stds = []

    for variant in variants:
        vals = results[variant].get(metric_key, {})
        if isinstance(vals, dict):
            means.append(vals.get("mean", 0))
            stds.append(vals.get("std", 0))
        else:
            means.append(float(vals) if vals else 0)
            stds.append(0)

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ["#4A90D9", "#E74C3C", "#2ECC71", "#F39C12", "#9B59B6"]
    bars = ax.bar(
        variants, means,
        yerr=stds,
        color=colors[:len(variants)],
        edgecolor="white",
        capsize=5,
        alpha=0.85,
    )

    # Add value labels
    for bar, mean in zip(bars, means):
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
            f"{mean:.3f}", ha="center", va="bottom", fontsize=10,
        )

    ax.set_ylabel(ylabel or metric_key)
    ax.set_title(f"Ablation Study: {study_name}")
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_dir / f"ablation_{study_name}_{metric_key}.png", dpi=150)
    plt.close(fig)
    print(f"Ablation plot saved: ablation_{study_name}_{metric_key}.png")


def run_loss_ablation_analysis(
    baseline_results: Dict,
    ssim_results: Dict,
    output_dir: Path,
    case_metrics_paths: Dict[str, Path] | None = None,
) -> pd.DataFrame:
    """Compare baseline (L1 only) vs full (L1 + SSIM) loss."""
    results = {
        "L_cGAN + L1": baseline_results,
        "L_cGAN + L1 + L_SSIM": ssim_results,
    }

    df = generate_ablation_table(results, output_dir, "loss_comparison")

    for metric in ["ssim", "psnr", "recon_dice"]:
        plot_ablation_comparison(
            results, metric, output_dir, "loss_comparison",
            ylabel=metric.upper(),
        )

    if case_metrics_paths:
        paired = paired_ttests_from_case_metrics(
            case_metrics_paths,
            metrics=["ssim", "psnr", "seg_dice", "recon_dice"],
        )
        if not paired.empty:
            paired.to_csv(output_dir / "paired_tests_loss_comparison.csv", index=False)

    return df


def run_ssim_weight_ablation_analysis(
    results_by_weight: Dict[float, Dict],
    output_dir: Path,
    case_metrics_paths: Dict[str, Path] | None = None,
) -> pd.DataFrame:
    """Compare different SSIM weights: λ₂ ∈ {0, 5, 10, 20}."""
    results = {f"λ₂={w}": m for w, m in results_by_weight.items()}

    df = generate_ablation_table(results, output_dir, "ssim_weight")

    for metric in ["ssim", "psnr"]:
        plot_ablation_comparison(
            results, metric, output_dir, "ssim_weight",
            ylabel=metric.upper(),
        )

    if case_metrics_paths:
        paired = paired_ttests_from_case_metrics(
            case_metrics_paths,
            metrics=["ssim", "psnr", "seg_dice", "recon_dice"],
        )
        if not paired.empty:
            paired.to_csv(output_dir / "paired_tests_ssim_weight.csv", index=False)

    return df


def run_segmentation_ablation_analysis(
    results_by_method: Dict[str, Dict],
    output_dir: Path,
    case_metrics_paths: Dict[str, Path] | None = None,
) -> pd.DataFrame:
    """Compare segmentation methods: Classical vs MLP vs U-Net vs Combined."""
    df = generate_ablation_table(results_by_method, output_dir, "segmentation")

    for metric in ["seg_dice", "recon_dice"]:
        if any(metric in v for v in results_by_method.values()):
            plot_ablation_comparison(
                results_by_method, metric, output_dir, "segmentation",
                ylabel="Dice Score",
            )

    if case_metrics_paths:
        paired = paired_ttests_from_case_metrics(
            case_metrics_paths,
            metrics=["seg_dice", "recon_dice"],
        )
        if not paired.empty:
            paired.to_csv(output_dir / "paired_tests_segmentation.csv", index=False)

    return df


def run_reconstruction_ablation_analysis(
    results_by_variant: Dict[str, Dict],
    output_dir: Path,
    case_metrics_paths: Dict[str, Path] | None = None,
) -> pd.DataFrame:
    """Compare reconstruction with and without reprojection optimisation."""
    df = generate_ablation_table(results_by_variant, output_dir, "reconstruction")

    for metric in ["recon_dice", "recon_hausdorff", "recon_chamfer"]:
        if any(metric in v for v in results_by_variant.values()):
            plot_ablation_comparison(
                results_by_variant,
                metric,
                output_dir,
                "reconstruction",
                ylabel=metric.replace("_", " ").title(),
            )

    if case_metrics_paths:
        paired = paired_ttests_from_case_metrics(
            case_metrics_paths,
            metrics=["recon_dice", "recon_hausdorff", "recon_chamfer"],
        )
        if not paired.empty:
            paired.to_csv(output_dir / "paired_tests_reconstruction.csv", index=False)

    return df
