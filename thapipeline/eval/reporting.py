"""Consolidated reporting for dissertation experiments."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import pandas as pd

from thapipeline.config import PipelineConfig
from thapipeline.utils.io import load_json, save_json


def _flatten_metric_summary(tag: str, summary: Dict[str, object]) -> Dict[str, object]:
    row: Dict[str, object] = {
        "tag": tag,
        "generator_checkpoint": summary.get("generator_checkpoint", ""),
        "segmentation_mode": summary.get("segmentation_mode", ""),
        "optimize_reprojection": summary.get("optimize_reprojection", ""),
        "evaluation_split": summary.get("evaluation_split", ""),
        "n_cases": summary.get("n_cases", 0),
        "fid": summary.get("fid"),
        "kid": summary.get("kid"),
        "watertight_rate": summary.get("watertight_rate"),
    }
    for metric in (
        "ssim",
        "psnr",
        "seg_dice",
        "seg_iou",
        "recon_dice",
        "recon_hausdorff",
        "recon_chamfer",
        "recon_error_px",
        "cup_radius_mm",
        "stem_length_mm",
    ):
        value = summary.get(metric)
        if isinstance(value, dict) and "mean" in value:
            row[f"{metric}_mean"] = value.get("mean")
            row[f"{metric}_std"] = value.get("std")
            row[f"{metric}_ci95_low"] = value.get("ci95_low")
            row[f"{metric}_ci95_high"] = value.get("ci95_high")
            row[f"{metric}_n"] = value.get("n")
    return row


def compile_statistical_report(config: PipelineConfig) -> Dict[str, object]:
    """Compile evaluation, segmentation, ablation, and CV outputs into one report."""
    metrics_dir = config.paths.metrics_dir

    evaluation_rows: List[Dict[str, object]] = []
    evaluation_summaries: Dict[str, object] = {}
    for eval_summary_path in sorted(metrics_dir.glob("*/evaluation_summary.json")):
        tag = eval_summary_path.parent.name
        summary = load_json(eval_summary_path)
        evaluation_summaries[tag] = summary
        evaluation_rows.append(_flatten_metric_summary(tag, summary))

    cross_validation = {}
    for overview_path in sorted((metrics_dir / "cross_validation").glob("*/cross_validation_overview.json")):
        cross_validation[overview_path.parent.name] = load_json(overview_path)

    paired_tests = {}
    ablation_dir = metrics_dir / "ablation"
    if ablation_dir.exists():
        for csv_path in sorted(ablation_dir.glob("paired_tests_*.csv")):
            paired_tests[csv_path.name] = pd.read_csv(csv_path).to_dict(orient="records")

    segmentation_report = None
    if config.paths.segmentation_report_json.exists():
        segmentation_report = load_json(config.paths.segmentation_report_json)

    report = {
        "evaluations": evaluation_summaries,
        "cross_validation": cross_validation,
        "segmentation_report": segmentation_report,
        "paired_tests": paired_tests,
    }

    if evaluation_rows:
        pd.DataFrame(evaluation_rows).to_csv(config.paths.statistics_summary_csv, index=False)
    save_json(report, config.paths.statistics_summary_json)
    return report
