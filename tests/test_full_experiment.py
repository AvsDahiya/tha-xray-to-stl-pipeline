import tempfile
import unittest
from pathlib import Path

import torch

from thapipeline.config import PathConfig, PipelineConfig
from thapipeline.orchestration.full_experiment import (
    build_blocked_summary,
    load_run_state,
    run_full_experiment,
    select_best_ssim_run,
)
from thapipeline.utils.io import save_json


def _create_run(
    run_dir: Path,
    *,
    status: str,
    best_val_ssim: float | None = None,
    best_val_psnr: float | None = None,
    latest_epoch: int | None = None,
    with_best_model: bool = False,
) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest = {"status": status}
    if best_val_ssim is not None:
        manifest["best_val_ssim"] = best_val_ssim
    if best_val_psnr is not None:
        manifest["best_val_psnr"] = best_val_psnr
    save_json(manifest, run_dir / "run_manifest.json")
    save_json(
        {
            "history": {
                "val_ssim": [best_val_ssim] if best_val_ssim is not None else [],
                "val_psnr": [best_val_psnr] if best_val_psnr is not None else [],
            }
        },
        run_dir / "history.json",
    )
    if latest_epoch is not None:
        torch.save({"epoch": latest_epoch}, run_dir / f"epoch_{latest_epoch:04d}.pt")
    if with_best_model:
        torch.save({"epoch": latest_epoch or 0}, run_dir / "best_model.pt")


class FullExperimentTests(unittest.TestCase):
    def test_load_run_state_detects_latest_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir) / "models" / "pix2pix" / "demo_ssim"
            _create_run(
                run_dir,
                status="running",
                best_val_ssim=0.61,
                best_val_psnr=12.4,
                latest_epoch=7,
                with_best_model=False,
            )

            state = load_run_state(run_dir)

        self.assertEqual(state["status"], "running")
        self.assertFalse(state["completed"])
        self.assertTrue(str(state["latest_checkpoint"]).endswith("epoch_0007.pt"))
        self.assertIsNone(state["best_checkpoint"])
        self.assertAlmostEqual(float(state["best_val_ssim"]), 0.61)

    def test_select_best_ssim_run_uses_ssim_then_psnr_then_name(self) -> None:
        states = [
            {
                "name": "run_b",
                "best_checkpoint": "/tmp/run_b/best_model.pt",
                "best_val_ssim": 0.60,
                "best_val_psnr": 12.1,
            },
            {
                "name": "run_a",
                "best_checkpoint": "/tmp/run_a/best_model.pt",
                "best_val_ssim": 0.60,
                "best_val_psnr": 12.1,
            },
            {
                "name": "run_c",
                "best_checkpoint": "/tmp/run_c/best_model.pt",
                "best_val_ssim": 0.60,
                "best_val_psnr": 12.5,
            },
            {
                "name": "run_d",
                "best_checkpoint": "/tmp/run_d/best_model.pt",
                "best_val_ssim": 0.58,
                "best_val_psnr": 13.0,
            },
        ]

        selected = select_best_ssim_run(states)
        self.assertEqual(selected["name"], "run_c")

        states[2]["best_val_psnr"] = 12.1
        selected = select_best_ssim_run(states)
        self.assertEqual(selected["name"], "run_a")

    def test_build_blocked_summary_fields(self) -> None:
        annotations_dir = Path("/tmp/hipxnet_masks")
        summary = build_blocked_summary(
            tag="d1_full",
            device="mps",
            resume_command="python3 scripts/12_run_full_experiment.py --device mps",
            reason="Masks are missing.",
            annotations_dir=annotations_dir,
        )
        self.assertEqual(summary["status"], "blocked_on_annotations")
        self.assertEqual(summary["annotations_dir"], str(annotations_dir))
        self.assertIn("scripts/12_run_full_experiment.py", summary["resume_command"])

    def test_run_full_experiment_dry_run_blocks_on_missing_masks(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            project_root = Path(tmpdir)
            config = PipelineConfig(paths=PathConfig(project_root=project_root))

            _create_run(
                config.paths.pix2pix_dir / "d1_baseline",
                status="completed",
                best_val_ssim=0.54,
                best_val_psnr=12.2,
                latest_epoch=20,
                with_best_model=True,
            )
            _create_run(
                config.paths.pix2pix_dir / "d1_ssim_l5",
                status="completed",
                best_val_ssim=0.57,
                best_val_psnr=12.6,
                latest_epoch=19,
                with_best_model=True,
            )
            _create_run(
                config.paths.pix2pix_dir / "d1_ssim_l10",
                status="running",
                best_val_ssim=0.55,
                best_val_psnr=12.4,
                latest_epoch=3,
                with_best_model=False,
            )

            manifest = run_full_experiment(
                config,
                device="cpu",
                tag="d1_full",
                skip_existing=True,
                dry_run=True,
                final_ssim_weights=(5, 10, 20),
                run_prefix="d1",
            )

            manifest_path = config.paths.experiments_dir / "full_run" / "d1_full" / "orchestration_manifest.json"
            blocked_path = config.paths.experiments_dir / "full_run" / "d1_full" / "blocked_on_annotations.json"

            self.assertEqual(manifest["status"], "blocked_on_annotations")
            self.assertTrue(manifest_path.exists())
            self.assertTrue(blocked_path.exists())
            self.assertEqual(manifest["stages"]["gan"]["baseline"]["action"], "skipped_completed")
            self.assertEqual(manifest["stages"]["gan"]["ssim_runs"][0]["name"], "d1_ssim_l5")
            self.assertEqual(manifest["stages"]["evaluation"]["baseline"]["status"], "would_run")
            self.assertEqual(manifest["stages"]["annotation_gate"]["status"], "blocked")


if __name__ == "__main__":
    unittest.main()
