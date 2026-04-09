"""One-shot orchestration for the dissertation experiment pipeline."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set

import pandas as pd

from thapipeline.config import PipelineConfig, get_device
from thapipeline.data.curate import curate_all_datasets
from thapipeline.data.materialize import preprocess_all
from thapipeline.data.pairing import run_pairing_pipeline
from thapipeline.eval.ablation_runner import (
    run_loss_ablation_analysis,
    run_reconstruction_ablation_analysis,
    run_segmentation_ablation_analysis,
)
from thapipeline.eval.evaluate_full_pipeline import evaluate_full_pipeline
from thapipeline.eval.reporting import compile_statistical_report
from thapipeline.inference.gan_infer import infer_single, load_generator
from thapipeline.inference.segment_and_recon import process_single_case
from thapipeline.models.segmenter import ImplantSegmenter
from thapipeline.training.train_pix2pix import Pix2PixTrainer
from thapipeline.training.train_segmenter import prepare_training_records, train_and_evaluate_segmenter
from thapipeline.utils.experiment_log import utc_timestamp
from thapipeline.utils.io import best_resume_checkpoint, load_json, save_json
from thapipeline.utils.vis import plot_training_curves


def _weight_label(value: float) -> str:
    return str(int(value)) if float(value).is_integer() else str(value).replace(".", "p")


def resolve_run_prefix(config: PipelineConfig, tag: str, run_prefix: Optional[str] = None) -> str:
    """Resolve the GAN run prefix, reusing the existing `d1_*` family when present."""
    if run_prefix:
        return run_prefix
    if (config.paths.pix2pix_dir / "d1_baseline").exists():
        return "d1"
    if tag.endswith("_full"):
        return tag[: -len("_full")]
    return tag


def run_name_for_variant(run_prefix: str, variant: str, lambda_ssim: Optional[float] = None) -> str:
    if variant == "baseline":
        return f"{run_prefix}_baseline"
    if lambda_ssim is None:
        raise ValueError("lambda_ssim is required for SSIM variants")
    return f"{run_prefix}_ssim_l{_weight_label(lambda_ssim)}"


def load_run_state(run_dir: Path) -> Dict[str, object]:
    """Load run metadata for orchestration decisions."""
    manifest_path = run_dir / "run_manifest.json"
    history_path = run_dir / "history.json"
    manifest = load_json(manifest_path) if manifest_path.exists() else {}
    history_payload = load_json(history_path) if history_path.exists() else {}
    history = history_payload.get("history", {})
    latest_ckpt = best_resume_checkpoint(run_dir)
    best_ckpt = run_dir / "best_model.pt"
    best_ssim = manifest.get("best_val_ssim")
    if best_ssim is None and history.get("val_ssim"):
        best_ssim = max(history["val_ssim"])
    best_psnr = manifest.get("best_val_psnr")
    if best_psnr is None and history.get("val_psnr"):
        best_psnr = max(history["val_psnr"])
    completed = manifest.get("status") == "completed" and best_ckpt.exists()
    return {
        "name": run_dir.name,
        "run_dir": str(run_dir),
        "manifest": manifest,
        "latest_checkpoint": str(latest_ckpt) if latest_ckpt else None,
        "best_checkpoint": str(best_ckpt) if best_ckpt.exists() else None,
        "best_val_ssim": float(best_ssim) if best_ssim is not None else None,
        "best_val_psnr": float(best_psnr) if best_psnr is not None else None,
        "status": manifest.get("status", "missing"),
        "completed": completed,
    }


def select_best_ssim_run(run_states: Sequence[Dict[str, object]]) -> Dict[str, object]:
    """Select the final SSIM run by best validation SSIM, then PSNR, then name."""
    valid = [
        state for state in run_states
        if state.get("best_val_ssim") is not None and state.get("best_checkpoint")
    ]
    if not valid:
        raise RuntimeError("No completed SSIM runs are available for model selection.")
    return sorted(
        valid,
        key=lambda state: (
            -float(state["best_val_ssim"]),
            -float(state.get("best_val_psnr") or -1e9),
            str(state["name"]),
        ),
    )[0]


def build_blocked_summary(
    tag: str,
    device: str,
    resume_command: str,
    reason: str,
    annotations_dir: Path,
) -> Dict[str, object]:
    return {
        "status": "blocked_on_annotations",
        "blocked_at": utc_timestamp(),
        "tag": tag,
        "reason": reason,
        "annotations_dir": str(annotations_dir),
        "resume_command": resume_command,
        "device": device,
    }


def _write_json(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    save_json(payload, path)


def _make_resume_command(
    tag: str,
    device: str,
    config: PipelineConfig,
    skip_existing: bool,
    final_ssim_weights: Sequence[float],
) -> str:
    cmd = [
        "python3 scripts/12_run_full_experiment.py",
        f"--device {device}",
        f"--tag {tag}",
        f"--epochs {config.training.epochs}",
        f"--batch-size {config.training.batch_size}",
        f"--grad-accum-steps {config.training.grad_accum_steps}",
        f"--checkpoint-every {config.training.checkpoint_every}",
        f"--patience {config.training.patience}",
    ]
    if skip_existing:
        cmd.append("--skip-existing")
    if final_ssim_weights:
        weights = " ".join(_weight_label(weight) for weight in final_ssim_weights)
        cmd.append(f"--final-ssim-weights {weights}")
    return " \\\n  ".join(cmd)


def _stop_requested(stop_after: Optional[str], current_stage: str) -> bool:
    return stop_after == current_stage


def _write_single_split_summary(
    run_states: Sequence[Dict[str, object]],
    selected_state: Dict[str, object],
    output_dir: Path,
) -> Dict[str, object]:
    rows: List[Dict[str, object]] = []
    for state in run_states:
        rows.append(
            {
                "run_name": state["name"],
                "status": state["status"],
                "best_val_ssim": state.get("best_val_ssim"),
                "best_val_psnr": state.get("best_val_psnr"),
                "best_checkpoint": state.get("best_checkpoint"),
                "selected": state["name"] == selected_state["name"],
            }
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    csv_path = output_dir / "single_split_summary.csv"
    df.to_csv(csv_path, index=False)
    payload = {
        "generated_at": utc_timestamp(),
        "selected_model": selected_state,
        "runs": rows,
    }
    _write_json(output_dir / "single_split_summary.json", payload)
    _write_json(output_dir / "selected_model.json", selected_state)
    return payload


def _make_segmenter(config: PipelineConfig, device: str) -> ImplantSegmenter:
    seg_mlp = config.paths.segmenter_dir / "pixel_mlp.pt"
    seg_unet = config.paths.segmenter_dir / "unet_fallback.pt"
    return ImplantSegmenter(
        device=device,
        mlp_checkpoint=seg_mlp if seg_mlp.exists() else None,
        unet_checkpoint=seg_unet if seg_unet.exists() else None,
    )


def _run_training_variant(
    config: PipelineConfig,
    device: str,
    run_name: str,
    use_ssim: bool,
    lambda_ssim: Optional[float],
    skip_existing: bool,
    force_stage: bool,
    notes: str,
    logic_change_note: str,
    dry_run: bool,
) -> Dict[str, object]:
    cfg = deepcopy(config)
    if lambda_ssim is not None:
        cfg.training.lambda_SSIM = float(lambda_ssim)
    run_dir = cfg.paths.pix2pix_dir / run_name
    state = load_run_state(run_dir)

    if state["completed"] and skip_existing and not force_stage:
        state["action"] = "skipped_completed"
        return state
    if state["completed"] and force_stage:
        state["action"] = "skipped_completed_no_safe_force_rerun"
        return state

    resume_path = best_resume_checkpoint(run_dir, device=device)
    if dry_run:
        state["action"] = "would_resume" if resume_path else "would_start"
        state["resume_path"] = str(resume_path) if resume_path else None
        return state

    trainer = Pix2PixTrainer(
        cfg,
        use_ssim=use_ssim,
        device=device,
        experiment_name=run_name,
        notes=notes,
        logic_change_note=logic_change_note,
    )
    history = trainer.train(resume_path=resume_path)
    plot_training_curves(
        history,
        cfg.paths.figures_dir / f"training_curves_{run_name}.png",
    )
    updated_state = load_run_state(run_dir)
    updated_state["action"] = "trained"
    updated_state["resume_path"] = str(resume_path) if resume_path else None
    return updated_state


def _run_reconstruction_stage(
    config: PipelineConfig,
    device: str,
    checkpoint: Path,
    output_tag: str,
    skip_existing: bool,
    force_stage: bool,
    optimize_reprojection: bool,
    dry_run: bool,
) -> Dict[str, object]:
    output_dir = config.paths.outputs_dir / "reconstruction" / output_tag
    results_path = output_dir / "pipeline_results.json"
    if results_path.exists() and skip_existing and not force_stage:
        return {"status": "skipped_existing", "results_path": str(results_path)}
    if dry_run:
        return {"status": "would_run", "results_path": str(results_path)}

    segmenter = _make_segmenter(config, device)
    generator = load_generator(checkpoint, config, device)
    pairs = pd.read_csv(config.paths.pairing_table)
    test_pairs = pairs[pairs["split"] == "test"]
    output_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for _, row in test_pairs.iterrows():
        case_id = Path(row["pre_path"]).stem
        result = infer_single(
            generator,
            Path(row["pre_processed_path"]),
            config,
            device,
            preprocessed=True,
        )
        case_result = process_single_case(
            case_id=case_id,
            generated=result["generated"],
            enhanced=result["enhanced"],
            target=None,
            segmenter=segmenter,
            config=config,
            output_dir=output_dir,
            segmentation_mode="combined",
            optimize_reprojection=optimize_reprojection,
        )
        results.append(case_result)

    _write_json(results_path, {"results": results})
    return {
        "status": "completed",
        "results_path": str(results_path),
        "n_cases": len(results),
        "n_success": sum(1 for item in results if item.get("success")),
    }


def _run_ablation_stage(
    config: PipelineConfig,
    device: str,
    baseline_checkpoint: Path,
    selected_checkpoint: Path,
    output_tag: str,
    skip_existing: bool,
    force_stage: bool,
    dry_run: bool,
) -> Dict[str, object]:
    output_dir = config.paths.metrics_dir / "ablation" / output_tag
    summary_path = output_dir / "ablation_manifest.json"
    if summary_path.exists() and skip_existing and not force_stage:
        return load_json(summary_path)
    if dry_run:
        return {"status": "would_run", "output_dir": str(output_dir)}

    segmenter = _make_segmenter(config, device)
    output_dir.mkdir(parents=True, exist_ok=True)

    baseline_tag = f"{output_tag}_loss_baseline"
    selected_tag = f"{output_tag}_loss_selected"
    classical_tag = f"{output_tag}_seg_classical"
    no_reproj_tag = f"{output_tag}_recon_no_reproj"

    baseline_summary = evaluate_full_pipeline(
        config,
        baseline_checkpoint,
        segmenter,
        device=device,
        output_name=baseline_tag,
        segmentation_mode="combined",
        optimize_reprojection=True,
        run_downstream=True,
    )
    selected_summary = evaluate_full_pipeline(
        config,
        selected_checkpoint,
        segmenter,
        device=device,
        output_name=selected_tag,
        segmentation_mode="combined",
        optimize_reprojection=True,
        run_downstream=True,
    )
    classical_summary = evaluate_full_pipeline(
        config,
        selected_checkpoint,
        segmenter,
        device=device,
        output_name=classical_tag,
        segmentation_mode="classical",
        optimize_reprojection=True,
        run_downstream=True,
    )
    no_reproj_summary = evaluate_full_pipeline(
        config,
        selected_checkpoint,
        segmenter,
        device=device,
        output_name=no_reproj_tag,
        segmentation_mode="combined",
        optimize_reprojection=False,
        run_downstream=True,
    )

    run_loss_ablation_analysis(
        baseline_summary,
        selected_summary,
        output_dir,
        case_metrics_paths={
            "L_cGAN + L1": config.paths.metrics_dir / baseline_tag / "case_metrics.csv",
            "L_cGAN + L1 + L_SSIM": config.paths.metrics_dir / selected_tag / "case_metrics.csv",
        },
    )
    run_segmentation_ablation_analysis(
        {"classical": classical_summary, "combined": selected_summary},
        output_dir,
        case_metrics_paths={
            "classical": config.paths.metrics_dir / classical_tag / "case_metrics.csv",
            "combined": config.paths.metrics_dir / selected_tag / "case_metrics.csv",
        },
    )
    run_reconstruction_ablation_analysis(
        {"with_reprojection": selected_summary, "without_reprojection": no_reproj_summary},
        output_dir,
        case_metrics_paths={
            "with_reprojection": config.paths.metrics_dir / selected_tag / "case_metrics.csv",
            "without_reprojection": config.paths.metrics_dir / no_reproj_tag / "case_metrics.csv",
        },
    )

    payload = {
        "status": "completed",
        "output_dir": str(output_dir),
        "baseline_tag": baseline_tag,
        "selected_tag": selected_tag,
        "classical_tag": classical_tag,
        "no_reproj_tag": no_reproj_tag,
    }
    _write_json(summary_path, payload)
    return payload


def _contains_masks(mask_dir: Path) -> bool:
    return mask_dir.exists() and any(mask_dir.glob("*.png"))


def run_full_experiment(
    config: PipelineConfig,
    device: Optional[str] = None,
    tag: str = "d1_full",
    skip_existing: bool = True,
    stop_after: Optional[str] = None,
    force_stages: Optional[Iterable[str]] = None,
    final_ssim_weights: Sequence[float] = (5, 10, 20),
    disable_reprojection_opt: bool = False,
    dry_run: bool = False,
    notes: str = "",
    logic_change_note: str = "",
    run_prefix: Optional[str] = None,
) -> Dict[str, object]:
    """Run the dissertation workflow end to end with restartable stage control."""
    dev = device or get_device()
    force_set: Set[str] = set(force_stages or [])
    run_prefix = resolve_run_prefix(config, tag, run_prefix=run_prefix)

    full_run_dir = config.paths.experiments_dir / "full_run" / tag
    full_run_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = full_run_dir / "orchestration_manifest.json"
    blocked_path = full_run_dir / "blocked_on_annotations.json"

    manifest: Dict[str, object] = {
        "tag": tag,
        "device": dev,
        "run_prefix": run_prefix,
        "skip_existing": skip_existing,
        "stop_after": stop_after,
        "force_stages": sorted(force_set),
        "final_ssim_weights": list(final_ssim_weights),
        "disable_reprojection_opt": bool(disable_reprojection_opt),
        "started_at": utc_timestamp(),
        "stages": {},
    }

    def record_stage(stage: str, payload: Dict[str, object]) -> None:
        manifest["stages"][stage] = {**payload, "updated_at": utc_timestamp()}
        _write_json(manifest_path, manifest)

    # Stage 1: Curate
    if config.paths.catalogue_csv.exists() and config.paths.data_sources_json.exists() and skip_existing and "curate" not in force_set:
        record_stage("curate", {"status": "skipped_existing", "catalogue": str(config.paths.catalogue_csv)})
    elif dry_run:
        record_stage("curate", {"status": "would_run", "catalogue": str(config.paths.catalogue_csv)})
    else:
        catalogue = curate_all_datasets(config)
        record_stage("curate", {"status": "completed", "n_catalogue_rows": int(len(catalogue))})
    if _stop_requested(stop_after, "curate"):
        manifest["status"] = "stopped_after_curate"
        _write_json(manifest_path, manifest)
        return manifest

    # Stage 2: Pair
    if config.paths.pairing_table.exists() and config.paths.split_indices.exists() and skip_existing and "pair" not in force_set:
        record_stage("pair", {"status": "skipped_existing", "pairing_table": str(config.paths.pairing_table)})
    elif dry_run:
        record_stage("pair", {"status": "would_run", "pairing_table": str(config.paths.pairing_table)})
    else:
        pairs_df = run_pairing_pipeline(config)
        record_stage("pair", {"status": "completed", "n_pairs": int(len(pairs_df))})
    if _stop_requested(stop_after, "pair"):
        manifest["status"] = "stopped_after_pair"
        _write_json(manifest_path, manifest)
        return manifest

    # Stage 3: Preprocess
    if dry_run:
        record_stage("preprocess", {"status": "would_run"})
    else:
        preprocess_summary = preprocess_all(config)
        record_stage("preprocess", {"status": "completed", **preprocess_summary})
    if _stop_requested(stop_after, "preprocess"):
        manifest["status"] = "stopped_after_preprocess"
        _write_json(manifest_path, manifest)
        return manifest

    # Stage 4-5: GAN training
    run_states: List[Dict[str, object]] = []
    baseline_name = run_name_for_variant(run_prefix, "baseline")
    baseline_state = _run_training_variant(
        config,
        dev,
        baseline_name,
        use_ssim=False,
        lambda_ssim=None,
        skip_existing=skip_existing,
        force_stage="train" in force_set,
        notes=notes or "One-shot dissertation baseline",
        logic_change_note=logic_change_note or "Baseline L_cGAN + L1",
        dry_run=dry_run,
    )
    run_states.append(baseline_state)

    ssim_states: List[Dict[str, object]] = []
    for weight in final_ssim_weights:
        ssim_name = run_name_for_variant(run_prefix, "ssim", weight)
        state = _run_training_variant(
            config,
            dev,
            ssim_name,
            use_ssim=True,
            lambda_ssim=weight,
            skip_existing=skip_existing,
            force_stage="train" in force_set,
            notes=notes or "One-shot dissertation SSIM sweep",
            logic_change_note=logic_change_note or f"SSIM weight {weight}",
            dry_run=dry_run,
        )
        ssim_states.append(state)
        run_states.append(state)

    record_stage(
        "gan",
        {
            "status": "completed" if not dry_run else "would_run",
            "baseline": baseline_state,
            "ssim_runs": ssim_states,
        },
    )
    if _stop_requested(stop_after, "gan"):
        manifest["status"] = "stopped_after_gan"
        _write_json(manifest_path, manifest)
        return manifest

    selected_state = select_best_ssim_run(ssim_states)
    single_split_summary = _write_single_split_summary(run_states, selected_state, full_run_dir)
    record_stage("model_selection", {"status": "completed", "selected_model": selected_state})

    # Stage 6-7: GAN-only held-out evaluation
    eval_output_dir = config.paths.metrics_dir
    baseline_eval_tag = f"{tag}_gan_eval_baseline"
    selected_eval_tag = f"{tag}_gan_eval_selected"
    evaluation_results = {}
    for label, state, eval_tag in (
        ("baseline", baseline_state, baseline_eval_tag),
        ("selected", selected_state, selected_eval_tag),
    ):
        summary_path = eval_output_dir / eval_tag / "evaluation_summary.json"
        checkpoint_path = state.get("best_checkpoint")
        if checkpoint_path is None:
            raise RuntimeError(f"Missing best checkpoint for {label}: {state['name']}")
        if summary_path.exists() and skip_existing and "evaluate" not in force_set:
            summary = load_json(summary_path)
            status = "skipped_existing"
        elif dry_run:
            summary = {"status": "would_run", "checkpoint": checkpoint_path}
            status = "would_run"
        else:
            summary = evaluate_full_pipeline(
                config,
                Path(checkpoint_path),
                segmenter=None,
                device=dev,
                output_name=eval_tag,
                run_downstream=False,
            )
            status = "completed"
        evaluation_results[label] = {"status": status, "tag": eval_tag, "summary": summary}
    record_stage("evaluation", {"status": "completed" if not dry_run else "would_run", **evaluation_results})
    manifest["single_split_summary"] = single_split_summary
    _write_json(manifest_path, manifest)
    if _stop_requested(stop_after, "evaluation"):
        manifest["status"] = "stopped_after_evaluation"
        _write_json(manifest_path, manifest)
        return manifest

    # Stage 8: Annotation gate
    if not _contains_masks(config.paths.implant_masks_dir):
        blocked = build_blocked_summary(
            tag=tag,
            device=dev,
            resume_command=_make_resume_command(tag, dev, config, skip_existing, final_ssim_weights),
            reason="No HipXNet implant masks were found; downstream segmentation and reconstruction are paused.",
            annotations_dir=config.paths.implant_masks_dir,
        )
        blocked["selected_model"] = selected_state
        _write_json(blocked_path, blocked)
        record_stage("annotation_gate", {"blocked_state": blocked, "status": "blocked"})
        manifest["status"] = "blocked_on_annotations"
        _write_json(manifest_path, manifest)
        return manifest

    record_stage("annotation_gate", {"status": "passed", "annotations_dir": str(config.paths.implant_masks_dir)})

    # Stage 9: Segmentation training
    if config.paths.segmentation_report_json.exists() and skip_existing and "segment" not in force_set:
        segmentation_report = load_json(config.paths.segmentation_report_json)
        seg_status = "skipped_existing"
    elif dry_run:
        segmentation_report = {"status": "would_run"}
        seg_status = "would_run"
    else:
        records = prepare_training_records(config)
        segmentation_report = train_and_evaluate_segmenter(records, config, dev)
        seg_status = "completed"
    record_stage("segmentation", {"status": seg_status, "report": segmentation_report})
    if _stop_requested(stop_after, "segmentation"):
        manifest["status"] = "stopped_after_segmentation"
        _write_json(manifest_path, manifest)
        return manifest

    # Stage 10: Reconstruction
    reconstruction_status = _run_reconstruction_stage(
        config,
        dev,
        Path(selected_state["best_checkpoint"]),
        tag,
        skip_existing=skip_existing,
        force_stage="reconstruct" in force_set,
        optimize_reprojection=not disable_reprojection_opt,
        dry_run=dry_run,
    )
    record_stage("reconstruction", reconstruction_status)
    if _stop_requested(stop_after, "reconstruction"):
        manifest["status"] = "stopped_after_reconstruction"
        _write_json(manifest_path, manifest)
        return manifest

    # Stage 11: Ablations
    ablation_status = _run_ablation_stage(
        config,
        dev,
        Path(baseline_state["best_checkpoint"]),
        Path(selected_state["best_checkpoint"]),
        tag,
        skip_existing=skip_existing,
        force_stage="ablation" in force_set,
        dry_run=dry_run,
    )
    record_stage("ablation", ablation_status)
    if _stop_requested(stop_after, "ablation"):
        manifest["status"] = "stopped_after_ablation"
        _write_json(manifest_path, manifest)
        return manifest

    # Stage 12: Consolidated reporting
    if dry_run:
        report = {"status": "would_run"}
    else:
        report = compile_statistical_report(config)
    record_stage("report", {"status": "completed" if not dry_run else "would_run"})
    manifest["report"] = report
    manifest["status"] = "completed"
    manifest["completed_at"] = utc_timestamp()
    _write_json(manifest_path, manifest)
    return manifest
