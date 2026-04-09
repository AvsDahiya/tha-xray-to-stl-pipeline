"""Full pipeline: segmentation → 3D reconstruction → STL export."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np

from thapipeline.config import PipelineConfig
from thapipeline.models.segmenter import ImplantSegmenter
from thapipeline.models.recon_3d import reconstruct_from_mask
from thapipeline.eval.metrics import compute_all_metrics
from thapipeline.utils.io import save_image, save_json
from thapipeline.utils.mesh_utils import project_mesh_to_mask


def process_single_case(
    case_id: str,
    generated: np.ndarray,
    enhanced: np.ndarray,
    target: Optional[np.ndarray],
    segmenter: ImplantSegmenter,
    config: PipelineConfig,
    output_dir: Path,
    segmentation_mode: str = "combined",
    optimize_reprojection: bool = True,
) -> Dict:
    """Process a single case through segmentation → reconstruction.

    Args:
        case_id: Unique identifier for this case.
        generated: GAN-generated image (uint8).
        enhanced: CLAHE-enhanced pre-op image (uint8).
        target: Target post-op image (uint8) for evaluation, or None.
        segmenter: Trained segmenter instance.
        config: Pipeline configuration.
        output_dir: Where to save outputs.

    Returns:
        Result dictionary with metrics, paths, and metadata.
    """
    timings = {}
    result = {"case_id": case_id, "success": False}

    # ── Segmentation ────────────────────────────────────────────────────
    t0 = time.perf_counter()
    seg_mask, seg_method = segmenter.segment(
        enhanced,
        generated,
        force_mode=segmentation_mode,
    )
    timings["segmentation"] = time.perf_counter() - t0

    # Save segmentation mask
    case_dir = output_dir / case_id
    case_dir.mkdir(parents=True, exist_ok=True)
    save_image(seg_mask, case_dir / "segmentation_mask.png")

    # ── 3D Reconstruction ───────────────────────────────────────────────
    t1 = time.perf_counter()
    mesh, recon_meta = reconstruct_from_mask(
        seg_mask,
        dpi=config.reconstruction.default_dpi,
        magnification=config.reconstruction.magnification_factor,
        optimize=optimize_reprojection,
        smooth_iterations=config.reconstruction.laplacian_iterations,
    )
    timings["reconstruction"] = time.perf_counter() - t1

    # Save mesh
    if mesh is not None:
        stl_path = case_dir / "implant.stl"
        mesh.export(str(stl_path))
        result["mesh_path"] = str(stl_path)
        result["success"] = True

        # Project mesh for evaluation
        proj_mask = project_mesh_to_mask(
            mesh, seg_mask.shape,
            dpi=config.reconstruction.default_dpi,
        )
        save_image(proj_mask, case_dir / "reprojection_mask.png")
    else:
        proj_mask = None

    # ── Metrics ─────────────────────────────────────────────────────────
    metrics = {}
    if target is not None:
        metrics = compute_all_metrics(
            generated=generated,
            target=target,
            seg_pred=seg_mask if target is not None else None,
            seg_target=None,  # No ground-truth segmentation for synthetic pairs
            mesh_proj=proj_mask,
            seg_mask=seg_mask,
        )

    # ── Save overview visualisation ─────────────────────────────────────
    _save_case_overview(case_id, enhanced, generated, seg_mask, proj_mask, case_dir)

    result.update({
        "timings": timings,
        "metrics": metrics,
        "recon_metadata": recon_meta,
        "seg_method": seg_method,
        "watertight": recon_meta.get("watertight", False),
    })

    return result


def _save_case_overview(
    case_id: str,
    enhanced: np.ndarray,
    generated: np.ndarray,
    seg_mask: np.ndarray,
    proj_mask: Optional[np.ndarray],
    output_dir: Path,
) -> None:
    """Save 4/5-panel overview figure."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_panels = 5 if proj_mask is not None else 4
    fig, axes = plt.subplots(1, n_panels, figsize=(4 * n_panels, 4))

    axes[0].imshow(enhanced, cmap="gray")
    axes[0].set_title("CLAHE Enhanced")
    axes[0].axis("off")

    axes[1].imshow(generated, cmap="gray")
    axes[1].set_title("GAN Generated")
    axes[1].axis("off")

    axes[2].imshow(seg_mask, cmap="gray")
    axes[2].set_title("Segmentation")
    axes[2].axis("off")

    # Overlay segmentation on generated image
    overlay = cv2.cvtColor(generated, cv2.COLOR_GRAY2RGB)
    mask_color = np.zeros_like(overlay)
    mask_color[seg_mask > 0] = [0, 255, 0]
    overlay = cv2.addWeighted(overlay, 0.7, mask_color, 0.3, 0)
    axes[3].imshow(overlay)
    axes[3].set_title("Overlay")
    axes[3].axis("off")

    if proj_mask is not None:
        axes[4].imshow(proj_mask, cmap="gray")
        axes[4].set_title("3D Reprojection")
        axes[4].axis("off")

    fig.suptitle(f"Case {case_id}", fontsize=14)
    fig.tight_layout()
    fig.savefig(output_dir / f"overview.png", dpi=150)
    plt.close(fig)


def run_full_pipeline(
    results_from_gan: List[Dict],
    segmenter: ImplantSegmenter,
    config: PipelineConfig,
    output_dir: Path,
    segmentation_mode: str = "combined",
    optimize_reprojection: bool = True,
) -> List[Dict]:
    """Run full segmentation + reconstruction on all GAN outputs.

    Args:
        results_from_gan: List of dicts from gan_infer.batch_inference.
        segmenter: Trained segmenter.
        config: Pipeline configuration.
        output_dir: Output directory for meshes and visualisations.

    Returns:
        List of result dictionaries.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    all_results = []

    for i, gan_result in enumerate(results_from_gan):
        if "error" in gan_result:
            continue

        case_id = Path(gan_result["input_path"]).stem
        print(f"[{i+1}/{len(results_from_gan)}] Processing {case_id}...")

        result = process_single_case(
            case_id=case_id,
            generated=gan_result["generated"],
            enhanced=gan_result["enhanced"],
            target=None,  # Target loaded separately if available
            segmenter=segmenter,
            config=config,
            output_dir=output_dir,
            segmentation_mode=segmentation_mode,
            optimize_reprojection=optimize_reprojection,
        )
        all_results.append(result)

    # Summary
    success = sum(1 for r in all_results if r["success"])
    watertight = sum(1 for r in all_results if r.get("watertight", False))
    print(f"\nResults: {success}/{len(all_results)} successful, "
          f"{watertight}/{len(all_results)} watertight meshes")

    # Save aggregate results
    save_json(all_results, output_dir / "pipeline_results.json")

    return all_results
