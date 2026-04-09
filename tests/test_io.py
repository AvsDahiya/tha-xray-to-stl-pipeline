import tempfile
import unittest
from pathlib import Path

import torch

from thapipeline.utils.io import best_resume_checkpoint, prune_epoch_checkpoints


class IOTests(unittest.TestCase):
    def test_prune_epoch_checkpoints_keeps_latest_and_protected(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir)
            for idx in range(1, 6):
                (run_dir / f"epoch_{idx:04d}.pt").write_bytes(b"x")
            best_path = run_dir / "best_model.pt"
            best_path.write_bytes(b"best")

            removed = prune_epoch_checkpoints(
                run_dir,
                keep_last=2,
                keep_paths=[best_path],
            )

            remaining = sorted(path.name for path in run_dir.glob("epoch_*.pt"))
            self.assertEqual(remaining, ["epoch_0004.pt", "epoch_0005.pt"])
            self.assertEqual(len(removed), 3)
            self.assertTrue(best_path.exists())

    def test_best_resume_checkpoint_falls_back_from_corrupted_epoch_to_best_model(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir)
            torch.save({"epoch": 18, "best_ssim": 0.5}, run_dir / "best_model.pt")
            (run_dir / "epoch_0021.pt").write_bytes(b"corrupted")

            resume = best_resume_checkpoint(run_dir)

            self.assertEqual(resume, run_dir / "best_model.pt")

    def test_best_resume_checkpoint_prefers_latest_valid_epoch(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir)
            torch.save({"epoch": 10}, run_dir / "best_model.pt")
            torch.save({"epoch": 11}, run_dir / "epoch_0011.pt")
            torch.save({"epoch": 12}, run_dir / "epoch_0012.pt")

            resume = best_resume_checkpoint(run_dir)

            self.assertEqual(resume, run_dir / "epoch_0012.pt")


if __name__ == "__main__":
    unittest.main()
