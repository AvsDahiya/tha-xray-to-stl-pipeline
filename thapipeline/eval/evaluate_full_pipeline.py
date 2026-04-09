"""End-to-end evaluation orchestration for the THA pipeline."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch

from thapipeline.config import PipelineConfig, get_device
from thapipeline.data.datasets import RadiographPairDataset
from thapipeline.eval.metrics import (
    compute_chamfer_distance,
    compute_dice,
    compute_fid_from_paths,
    compute_hausdorff_distance,
    compute_kid_from_paths,
    compute_psnr,
    compute_reprojection_dice,
    compute_reprojection_error,
    compute_ssim,
)
from thapipeline.eval.statistics import summary_with_ci
from thapipeline.models.segmenter import ImplantSegmenter
from thapipeline.models.recon_3d import reconstruct_from_mask
from thapipeline.utils.io import load_image, save_json
from thapipeline.utils.mesh_utils import project_mesh_to_mask


def _load_mask(mask_dir: Path, case_id: str) -> Optional[np.ndarray]:
    mask_path = mask_dir / f"{case_id}.png"
    if not mask_path.exists():
        return None
    return load_image(mask_path)


def _write_print_validation_template(config: PipelineConfig, case_rows: List[Dict[str, object]]) -> None:
    fieldnames = [
        "case_id",
        "mesh_path",
        "print_method",
        "material",
        "measured_cup_radius_mm",
        "measured_stem_length_mm",
        "reference_cup_radius_mm",
        "reference_stem_length_mm",
        "absolute_error_mm",
        "notes",
    ]
    with open(config.paths.print_validation_template, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in case_rows[:5]:
            writer.writerow(
                {
                    "case_id": row.get("case_id", ""),
                    "mesh_path": row.get("mesh_path", ""),
                    "print_method": "",
                    "material": "",
                    "measured_cup_radius_mm": "",
                    "measured_stem_length_mm": "",
                    "reference_cup_radius_mm": row.get("cup_radius_mm", ""),
                    "reference_stem_length_mm": row.get("stem_length_mm", ""),
                    "absolute_error_mm": "",
                    "notes": "",
                }
            )


def _nested_metric_or_none(payload: Dict[str, object], key: str, metric: str) -> Optional[float]:
    section = payload.get(key)
    if not isinstance(section, dict):
        return None
    value = section.get(metric)
    return float(value) if value is not None else None


def evaluate_full_pipeline(
    config: PipelineConfig,
    generator_checkpoint: Path,
    segmenter: Optional[ImplantSegmenter],
    device: Optional[str] = None,
    output_name: str = "test_evaluation",
    segmentation_mode: str = "combined",
    optimize_reprojection: bool = True,
    pairs_csv: Path | None = None,
    evaluation_split: str = "test",
    run_downstream: bool = True,
) -> Dict[str, object]:
    from thapipeline.inference.gan_infer import load_generator

    dev = device or get_device()
    generator = load_generator(generator_checkpoint, config, dev)
    pairs_csv = pairs_csv or config.paths.pairing_table
    dataset = RadiographPairDataset(pairs_csv, split=evaluation_split, config=config, augment=False)

    output_dir = config.paths.metrics_dir / output_name
    generated_dir = output_dir / "generated_images"
    output_dir.mkdir(parents=True, exist_ok=True)
    generated_dir.mkdir(parents=True, exist_ok=True)

    all_metrics: Dict[str, List[object]] = {
        "ssim": [],
        "psnr": [],
        "seg_dice": [],
        "seg_iou": [],
        "seg_method": [],
        "recon_dice": [],
        "recon_hausdorff": [],
        "recon_chamfer": [],
        "recon_error_px": [],
        "watertight": [],
        "cup_radius_mm": [],
        "stem_length_mm": [],
    }

    case_rows: List[Dict[str, object]] = []
    real_paths: List[Path] = []
    fake_paths: List[Path] = []
    mask_dir = config.paths.implant_masks_dir

    for index in range(len(dataset)):
        sample = dataset[index]
        case_id = str(sample["pair_id"])
        pre_tensor = sample["pre"].unsqueeze(0).to(dev)
        target = load_image(Path(sample["post_processed_path"]))
        enhanced_pre = load_image(Path(sample["pre_processed_path"]))

        with torch.no_grad():
            generated_tensor = generator(pre_tensor)
        generated = ((generated_tensor.squeeze().cpu().numpy() + 1.0) * 127.5).clip(0, 255).astype(np.uint8)

        generated_path = generated_dir / f"{case_id}.png"
        from thapipeline.utils.io import save_image

        save_image(generated, generated_path)
        fake_paths.append(generated_path)
        real_paths.append(Path(sample["post_processed_path"]))

        ssim = compute_ssim(generated, target)
        psnr = compute_psnr(generated, target)
        seg_dice = None
        seg_iou = None
        recon_dice = None
        recon_hausdorff = None
        recon_chamfer = None
        recon_error_px = None
        all_metrics["ssim"].append(ssim)
        all_metrics["psnr"].append(psnr)

        seg_method = ""
        mesh_path = None
        recon_meta: Dict[str, object] = {}
        if run_downstream:
            if segmenter is None:
                raise ValueError("segmenter is required when run_downstream=True")

            seg_mask, seg_method = segmenter.segment(
                enhanced_pre,
                generated,
                force_mode=segmentation_mode,
            )
            all_metrics["seg_method"].append(seg_method)

            seg_gt = _load_mask(mask_dir, str(sample["post_id"]))
            if seg_gt is not None:
                seg_dice = compute_dice(seg_mask, seg_gt)
                from thapipeline.eval.metrics import compute_iou

                seg_iou = compute_iou(seg_mask, seg_gt)
                all_metrics["seg_dice"].append(seg_dice)
                all_metrics["seg_iou"].append(seg_iou)

            mesh, recon_meta = reconstruct_from_mask(
                seg_mask,
                dpi=config.reconstruction.default_dpi,
                magnification=config.reconstruction.magnification_factor,
                optimize=optimize_reprojection,
                smooth_iterations=config.reconstruction.laplacian_iterations,
            )

            if mesh is not None:
                mesh_case_dir = config.paths.meshes_dir / output_name / case_id
                mesh_case_dir.mkdir(parents=True, exist_ok=True)
                mesh_path = mesh_case_dir / "implant.stl"
                mesh.export(str(mesh_path))
                proj_mask = project_mesh_to_mask(mesh, seg_mask.shape, dpi=config.reconstruction.default_dpi)
                recon_dice = compute_reprojection_dice(seg_mask, proj_mask)
                recon_error_px = compute_reprojection_error(seg_mask, proj_mask)
                all_metrics["recon_dice"].append(recon_dice)
                all_metrics["recon_error_px"].append(recon_error_px)

                seg_edges = np.column_stack(np.where(seg_mask > 0))
                proj_edges = np.column_stack(np.where(proj_mask > 0))
                recon_hausdorff = compute_hausdorff_distance(seg_edges, proj_edges)
                recon_chamfer = compute_chamfer_distance(seg_edges, proj_edges)
                all_metrics["recon_hausdorff"].append(recon_hausdorff)
                all_metrics["recon_chamfer"].append(recon_chamfer)

            all_metrics["watertight"].append(bool(recon_meta.get("watertight", False)))
            if recon_meta.get("cup"):
                all_metrics["cup_radius_mm"].append(recon_meta["cup"]["radius_mm"])
            if recon_meta.get("stem"):
                all_metrics["stem_length_mm"].append(recon_meta["stem"]["length_mm"])

        case_rows.append(
            {
                "case_id": case_id,
                "post_id": str(sample["post_id"]),
                "mesh_path": str(mesh_path) if mesh_path else "",
                "ssim": ssim,
                "psnr": psnr,
                "seg_dice": seg_dice,
                "seg_iou": seg_iou,
                "seg_method": seg_method,
                "recon_dice": recon_dice,
                "recon_hausdorff": recon_hausdorff,
                "recon_chamfer": recon_chamfer,
                "recon_error_px": recon_error_px,
                "watertight": bool(recon_meta.get("watertight", False)),
                "cup_radius_mm": _nested_metric_or_none(recon_meta, "cup", "radius_mm"),
                "stem_length_mm": _nested_metric_or_none(recon_meta, "stem", "length_mm"),
            }
        )

    summary = _aggregate_metrics(all_metrics)
    unique_real_paths = sorted(set(real_paths))
    summary["fid"] = compute_fid_from_paths(unique_real_paths, fake_paths, device=dev)
    summary["kid"] = compute_kid_from_paths(unique_real_paths, fake_paths, device=dev)
    summary["generator_checkpoint"] = str(generator_checkpoint)
    summary["segmentation_mode"] = segmentation_mode
    summary["optimize_reprojection"] = bool(optimize_reprojection)
    summary["pairs_csv"] = str(pairs_csv)
    summary["evaluation_split"] = evaluation_split
    summary["run_downstream"] = bool(run_downstream)

    save_json(summary, output_dir / "evaluation_summary.json")
    pd.DataFrame(case_rows).to_csv(output_dir / "case_metrics.csv", index=False)
    _save_results_table(summary, output_dir)
    _plot_metrics_distributions(all_metrics, output_dir)
    _write_print_validation_template(config, case_rows)
    return summary


def _aggregate_metrics(metrics: Dict[str, List[object]]) -> Dict[str, object]:
    summary: Dict[str, object] = {}
    for key, values in metrics.items():
        if key == "seg_method":
            from collections import Counter

            summary[key] = dict(Counter(values))
            continue
        if key == "watertight":
            summary["watertight_count"] = int(sum(bool(v) for v in values))
            summary["watertight_rate"] = float(sum(bool(v) for v in values) / max(len(values), 1))
            continue

        clean = [v for v in values if v is not None]
        if clean:
            summary[key] = summary_with_ci(clean)
    summary["n_cases"] = max((len(v) for v in metrics.values()), default=0)
    return summary


def _save_results_table(summary: Dict[str, object], output_dir: Path) -> None:
    rows = []
    for metric, values in summary.items():
        if isinstance(values, dict) and "mean" in values:
            rows.append(
                {
                    "Metric": metric,
                    "Mean": values["mean"],
                    "Std": values["std"],
                    "CI95_Low": values["ci95_low"],
                    "CI95_High": values["ci95_high"],
                    "Min": values["min"],
                    "Max": values["max"],
                    "N": values["n"],
                }
            )
    if rows:
        pd.DataFrame(rows).to_csv(output_dir / "metrics_table.csv", index=False)


def _plot_metrics_distributions(metrics: Dict[str, List[object]], output_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plot_keys = ["ssim", "psnr", "seg_dice", "recon_dice", "recon_hausdorff", "recon_chamfer"]
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    for ax, key in zip(axes.flat, plot_keys):
        values = [v for v in metrics.get(key, []) if v is not None]
        if values:
            ax.hist(values, bins=20, color="#4A90D9", edgecolor="white")
            ax.axvline(np.mean(values), color="red", linestyle="--", label=f"{np.mean(values):.3f}")
            ax.legend()
        ax.set_title(key)
    fig.tight_layout()
    fig.savefig(output_dir / "metrics_distributions.png", dpi=150)
    plt.close(fig)
