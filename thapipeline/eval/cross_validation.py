"""Cross-validation orchestration for Pix2Pix THA experiments."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd

from thapipeline.config import PipelineConfig, get_device
from thapipeline.data.pairing import run_kfold_pairing_pipeline
from thapipeline.eval.evaluate_full_pipeline import evaluate_full_pipeline
from thapipeline.eval.statistics import summary_with_ci
from thapipeline.models.segmenter import ImplantSegmenter
from thapipeline.training.train_pix2pix import Pix2PixTrainer
from thapipeline.utils.io import best_resume_checkpoint, load_json, save_json
from thapipeline.utils.vis import plot_training_curves


def _metric_value(summary: Dict[str, object], key: str) -> Optional[float]:
    value = summary.get(key)
    if isinstance(value, dict) and "mean" in value:
        return float(value["mean"])
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return None


def _run_completed(run_dir: Path) -> bool:
    manifest_path = run_dir / "run_manifest.json"
    if not manifest_path.exists():
        return False
    try:
        manifest = load_json(manifest_path)
    except Exception:
        return False
    return manifest.get("status") == "completed" and (run_dir / "best_model.pt").exists()


def aggregate_fold_summaries(
    fold_results: Sequence[Dict[str, object]],
    output_dir: Path,
    experiment_name: str,
) -> Dict[str, object]:
    """Aggregate fold summaries into a dissertation-ready summary."""
    output_dir.mkdir(parents=True, exist_ok=True)

    per_fold_rows: List[Dict[str, object]] = []
    metric_keys = sorted(
        {
            key
            for result in fold_results
            for key in result.get("summary", {}).keys()
            if isinstance(result.get("summary", {}).get(key), dict) and "mean" in result["summary"][key]
        }
        | {
            key
            for result in fold_results
            for key in result.get("summary", {}).keys()
            if isinstance(result.get("summary", {}).get(key), (int, float))
            and not isinstance(result.get("summary", {}).get(key), bool)
        }
    )

    aggregate: Dict[str, object] = {
        "experiment_name": experiment_name,
        "n_folds_completed": len(fold_results),
        "folds": [],
        "metrics": {},
    }

    for result in fold_results:
        row = {
            "fold": result["fold"],
            "mode": result["mode"],
            "checkpoint": result["checkpoint"],
            "evaluation_tag": result["evaluation_tag"],
        }
        for key in metric_keys:
            row[key] = _metric_value(result["summary"], key)
        per_fold_rows.append(row)
        aggregate["folds"].append(
            {
                "fold": result["fold"],
                "mode": result["mode"],
                "checkpoint": result["checkpoint"],
                "evaluation_tag": result["evaluation_tag"],
                "summary": result["summary"],
            }
        )

    for key in metric_keys:
        values = [row[key] for row in per_fold_rows if row.get(key) is not None]
        if values:
            aggregate["metrics"][key] = summary_with_ci(values)

    if per_fold_rows:
        pd.DataFrame(per_fold_rows).to_csv(output_dir / f"{experiment_name}_fold_metrics.csv", index=False)
    save_json(aggregate, output_dir / f"{experiment_name}_summary.json")
    return aggregate


def run_pix2pix_cross_validation(
    config: PipelineConfig,
    n_folds: int = 5,
    mode: str = "both",
    device: Optional[str] = None,
    fold_indices: Optional[Iterable[int]] = None,
    tag: str = "cv5",
    skip_train: bool = False,
    skip_eval: bool = False,
    force_train: bool = False,
    segmentation_mode: str = "combined",
    optimize_reprojection: bool = True,
    notes: str = "",
    logic_change_note: str = "",
) -> Dict[str, object]:
    """Run 5-fold cross-validation for baseline and/or SSIM variants."""
    dev = device or get_device()
    fold_root = config.paths.metadata_dir / "cross_validation"
    fold_paths = run_kfold_pairing_pipeline(config, n_folds=n_folds, output_root=fold_root)

    if fold_indices is None:
        selected_folds = list(range(1, n_folds + 1))
    else:
        selected_folds = sorted({int(idx) for idx in fold_indices if 1 <= int(idx) <= n_folds})

    seg_mlp = config.paths.segmenter_dir / "pixel_mlp.pt"
    seg_unet = config.paths.segmenter_dir / "unet_fallback.pt"
    segmenter = ImplantSegmenter(
        device=dev,
        mlp_checkpoint=seg_mlp if seg_mlp.exists() else None,
        unet_checkpoint=seg_unet if seg_unet.exists() else None,
    )

    mode_map = {
        "baseline": [("baseline", False)],
        "ssim": [("ssim", True)],
        "both": [("baseline", False), ("ssim", True)],
    }
    experiment_results: Dict[str, List[Dict[str, object]]] = {"baseline": [], "ssim": []}

    for fold_number in selected_folds:
        pairs_csv = fold_paths[fold_number - 1]
        fold_label = f"fold{fold_number:02d}"
        print(f"\n{'=' * 72}\nCROSS-VALIDATION {fold_label} | pairs={pairs_csv}\n{'=' * 72}")

        for variant_name, use_ssim in mode_map[mode]:
            cfg = deepcopy(config)
            lambda_label = (
                str(int(cfg.training.lambda_SSIM))
                if float(cfg.training.lambda_SSIM).is_integer()
                else str(cfg.training.lambda_SSIM).replace(".", "p")
            )
            experiment_name = (
                f"{tag}_{fold_label}_ssim_l{lambda_label}" if use_ssim else f"{tag}_{fold_label}_baseline"
            )
            checkpoint = cfg.paths.pix2pix_dir / experiment_name / "best_model.pt"
            run_dir = cfg.paths.pix2pix_dir / experiment_name
            resume_path = best_resume_checkpoint(run_dir, device=dev)
            completed = _run_completed(run_dir)

            if not skip_train:
                if checkpoint.exists() and completed and not force_train:
                    print(f"Reusing existing checkpoint: {checkpoint}")
                else:
                    if resume_path is not None and not force_train:
                        print(f"Resuming from checkpoint: {resume_path}")
                    trainer = Pix2PixTrainer(
                        cfg,
                        use_ssim=use_ssim,
                        device=dev,
                        experiment_name=experiment_name,
                        notes=notes,
                        logic_change_note=logic_change_note,
                        pairs_csv=pairs_csv,
                    )
                    history = trainer.train(resume_path=resume_path if resume_path and not force_train else None)
                    plot_training_curves(
                        history,
                        cfg.paths.figures_dir / f"training_curves_{experiment_name}.png",
                    )

            if skip_eval:
                continue

            if not checkpoint.exists():
                raise FileNotFoundError(
                    f"Missing checkpoint for cross-validation evaluation: {checkpoint}"
                )

            evaluation_tag = f"{tag}_{fold_label}_{variant_name}_eval"
            summary = evaluate_full_pipeline(
                cfg,
                checkpoint,
                segmenter,
                device=dev,
                output_name=evaluation_tag,
                segmentation_mode=segmentation_mode,
                optimize_reprojection=optimize_reprojection,
                pairs_csv=pairs_csv,
                evaluation_split="test",
            )
            experiment_results[variant_name].append(
                {
                    "fold": fold_number,
                    "mode": variant_name,
                    "checkpoint": str(checkpoint),
                    "evaluation_tag": evaluation_tag,
                    "pairs_csv": str(pairs_csv),
                    "summary": summary,
                }
            )

    cv_output_dir = config.paths.metrics_dir / "cross_validation" / tag
    cv_output_dir.mkdir(parents=True, exist_ok=True)

    aggregate_outputs: Dict[str, object] = {
        "tag": tag,
        "device": dev,
        "n_folds": n_folds,
        "selected_folds": selected_folds,
        "mode": mode,
        "results": {},
    }

    for variant_name, fold_results in experiment_results.items():
        if not fold_results:
            continue
        aggregate_outputs["results"][variant_name] = aggregate_fold_summaries(
            fold_results,
            cv_output_dir,
            variant_name,
        )

    save_json(aggregate_outputs, cv_output_dir / "cross_validation_overview.json")
    return aggregate_outputs
