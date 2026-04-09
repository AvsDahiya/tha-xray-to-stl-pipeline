import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from thapipeline.config import PathConfig, PipelineConfig
from thapipeline.data.datasets import RadiographPairDataset


class DatasetFallbackTests(unittest.TestCase):
    def test_pair_dataset_materializes_missing_processed_images(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            project_root = Path(tmpdir)
            config = PipelineConfig(paths=PathConfig(project_root=project_root))

            raw_dir = project_root / "raw"
            raw_dir.mkdir(parents=True, exist_ok=True)
            pre_raw = raw_dir / "pre.png"
            post_raw = raw_dir / "post.png"

            pre_image = np.zeros((600, 600), dtype=np.uint8)
            pre_image[:, 200:400] = 180
            post_image = np.zeros((600, 600), dtype=np.uint8)
            post_image[:, 180:420] = 210
            cv2.imwrite(str(pre_raw), pre_image)
            cv2.imwrite(str(post_raw), post_image)

            pre_processed = config.paths.data_processed / "train" / "pre" / "fracatlas__pre_case.png"
            post_processed = config.paths.data_processed / "train" / "post" / "hipxnet__post_case.png"

            pairs_csv = project_root / "pairs.csv"
            pd.DataFrame(
                [
                    {
                        "pair_id": "train_00001",
                        "split": "train",
                        "pre_path": str(pre_raw),
                        "post_path": str(post_raw),
                        "pre_processed_path": str(pre_processed),
                        "post_processed_path": str(post_processed),
                        "distance": 0.1,
                        "pre_source": "fracatlas",
                        "pre_id": "pre_case",
                        "post_id": "post_case",
                        "post_reuse_count": 1,
                    }
                ]
            ).to_csv(pairs_csv, index=False)

            dataset = RadiographPairDataset(pairs_csv, split="train", config=config, augment=False)
            sample = dataset[0]

            self.assertEqual(tuple(sample["pre"].shape), (1, 512, 512))
            self.assertEqual(tuple(sample["post"].shape), (1, 512, 512))
            self.assertTrue(pre_processed.exists())
            self.assertTrue(post_processed.exists())


if __name__ == "__main__":
    unittest.main()
