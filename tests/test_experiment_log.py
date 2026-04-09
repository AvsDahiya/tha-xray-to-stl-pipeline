from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from thapipeline.config import PipelineConfig
from thapipeline.utils.experiment_log import (
    append_jsonl,
    collect_dataset_summary,
    config_snapshot,
)
from thapipeline.utils.io import latest_epoch_checkpoint


class ExperimentLogTests(unittest.TestCase):
    def test_config_snapshot_contains_training(self):
        snapshot = config_snapshot(PipelineConfig())
        self.assertIn("training", snapshot)
        self.assertIn("paths", snapshot)

    def test_collect_dataset_summary_reads_pairing_table(self):
        with TemporaryDirectory() as tmpdir:
            pairing = Path(tmpdir) / "pairing_table.csv"
            pairing.write_text(
                "pair_id,split,post_id,post_reuse_count\n"
                "p1,train,a,1\n"
                "p2,train,a,2\n"
                "p3,val,b,1\n",
                encoding="utf-8",
            )
            summary = collect_dataset_summary(pairing)
            self.assertTrue(summary["available"])
            self.assertEqual(summary["total_pairs"], 3)
            self.assertEqual(summary["pairs_by_split"]["train"], 2)
            self.assertEqual(summary["max_post_reuse"], 2)

    def test_append_jsonl_writes_line(self):
        with TemporaryDirectory() as tmpdir:
            target = Path(tmpdir) / "registry.jsonl"
            append_jsonl(target, {"name": "exp1"})
            lines = target.read_text(encoding="utf-8").strip().splitlines()
            self.assertEqual(len(lines), 1)
            self.assertIn("exp1", lines[0])

    def test_latest_epoch_checkpoint(self):
        with TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir)
            (run_dir / "epoch_0001.pt").write_text("", encoding="utf-8")
            (run_dir / "epoch_0010.pt").write_text("", encoding="utf-8")
            latest = latest_epoch_checkpoint(run_dir)
            self.assertEqual(latest.name, "epoch_0010.pt")
