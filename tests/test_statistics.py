from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import pandas as pd

from thapipeline.eval.statistics import paired_ttests_from_case_metrics, summary_with_ci


class StatisticsTests(unittest.TestCase):
    def test_summary_with_ci_includes_interval(self):
        summary = summary_with_ci([0.8, 0.82, 0.78, 0.81, 0.79])
        self.assertIn("ci95_low", summary)
        self.assertIn("ci95_high", summary)
        self.assertEqual(summary["n"], 5)

    def test_paired_ttests_from_case_metrics(self):
        with TemporaryDirectory() as tmpdir:
            left = Path(tmpdir) / "left.csv"
            right = Path(tmpdir) / "right.csv"
            pd.DataFrame(
                {
                    "case_id": ["a", "b", "c"],
                    "ssim": [0.80, 0.82, 0.81],
                    "psnr": [24.0, 25.0, 24.5],
                }
            ).to_csv(left, index=False)
            pd.DataFrame(
                {
                    "case_id": ["a", "b", "c"],
                    "ssim": [0.83, 0.84, 0.85],
                    "psnr": [25.0, 25.5, 25.2],
                }
            ).to_csv(right, index=False)
            df = paired_ttests_from_case_metrics(
                {"left": left, "right": right},
                metrics=["ssim", "psnr"],
            )
            self.assertFalse(df.empty)
            self.assertIn("p_value", df.columns)
