import tempfile
import unittest
from pathlib import Path

from thapipeline.eval.cross_validation import _run_completed
from thapipeline.utils.io import save_json


class CrossValidationTests(unittest.TestCase):
    def test_run_completed_requires_completed_manifest_and_best_model(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir) / "run"
            run_dir.mkdir(parents=True, exist_ok=True)

            self.assertFalse(_run_completed(run_dir))

            save_json({"status": "running"}, run_dir / "run_manifest.json")
            (run_dir / "best_model.pt").write_bytes(b"x")
            self.assertFalse(_run_completed(run_dir))

            save_json({"status": "completed"}, run_dir / "run_manifest.json")
            self.assertTrue(_run_completed(run_dir))


if __name__ == "__main__":
    unittest.main()
