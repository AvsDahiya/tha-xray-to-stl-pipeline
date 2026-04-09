import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from thapipeline.config import PipelineConfig
from thapipeline.data.pairing import create_kfold_pairs, create_pairs


class PairingTests(unittest.TestCase):
    def test_split_safe_pairing_and_reuse_cap(self) -> None:
        config = PipelineConfig()
        rows = []
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            for idx in range(9):
                image = np.full((600, 600), 60 + idx * 10, dtype=np.uint8)
                image[:, 200:400] = 200
                path = root / f"pre_{idx}.png"
                cv2.imwrite(str(path), image)
                rows.append(
                    {
                        "source_id": f"pre_{idx}",
                        "canonical_source_id": f"pre_{idx}",
                        "source_dataset": "fracatlas",
                        "filepath": str(path),
                        "view": "AP",
                        "region": "hip",
                        "fracture_label": 0,
                        "postop_flag": 0,
                        "raw_split": "",
                        "has_label": False,
                        "label_path": "",
                        "notes": "",
                    }
                )

            for idx in range(6):
                image = np.full((600, 600), 70 + idx * 8, dtype=np.uint8)
                image[:, 180:420] = 210
                path = root / f"post_{idx}.png"
                cv2.imwrite(str(path), image)
                rows.append(
                    {
                        "source_id": f"post_{idx}",
                        "canonical_source_id": f"post_{idx}",
                        "source_dataset": "hipxnet",
                        "filepath": str(path),
                        "view": "AP",
                        "region": "hip_implant",
                        "fracture_label": 0,
                        "postop_flag": 1,
                        "raw_split": "",
                        "has_label": False,
                        "label_path": "",
                        "notes": "",
                    }
                )

            catalogue = pd.DataFrame(rows)
            pairs_df, split_indices = create_pairs(catalogue, config)

        self.assertGreater(len(pairs_df), 0)
        self.assertTrue({"pair_id", "pre_processed_path", "post_processed_path"}.issubset(pairs_df.columns))
        self.assertTrue((pairs_df["post_reuse_count"] <= config.split.max_postop_reuse).all())

        for split_name, groups in split_indices.items():
            pre_ids = set(groups["pre_ids"])
            post_ids = set(groups["post_ids"])
            self.assertTrue(pre_ids.isdisjoint(post_ids))

        pre_union = set()
        post_union = set()
        for groups in split_indices.values():
            current_pre = set(groups["pre_ids"])
            current_post = set(groups["post_ids"])
            self.assertTrue(pre_union.isdisjoint(current_pre))
            self.assertTrue(post_union.isdisjoint(current_post))
            pre_union |= current_pre
            post_union |= current_post

    def test_kfold_pairing_generation(self) -> None:
        config = PipelineConfig()
        rows = []
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            for idx in range(10):
                image = np.full((512, 512), 50 + idx * 5, dtype=np.uint8)
                image[:, 160:352] = 180
                path = root / f"pre_fold_{idx}.png"
                cv2.imwrite(str(path), image)
                rows.append(
                    {
                        "source_id": f"pre_fold_{idx}",
                        "canonical_source_id": f"pre_fold_{idx}",
                        "source_dataset": "fracatlas",
                        "filepath": str(path),
                        "view": "AP",
                        "region": "hip",
                        "fracture_label": 0,
                        "postop_flag": 0,
                        "raw_split": "",
                        "has_label": False,
                        "label_path": "",
                        "notes": "",
                    }
                )

            for idx in range(10):
                image = np.full((512, 512), 70 + idx * 4, dtype=np.uint8)
                image[:, 150:360] = 220
                path = root / f"post_fold_{idx}.png"
                cv2.imwrite(str(path), image)
                rows.append(
                    {
                        "source_id": f"post_fold_{idx}",
                        "canonical_source_id": f"post_fold_{idx}",
                        "source_dataset": "hipxnet",
                        "filepath": str(path),
                        "view": "AP",
                        "region": "hip_implant",
                        "fracture_label": 0,
                        "postop_flag": 1,
                        "raw_split": "",
                        "has_label": False,
                        "label_path": "",
                        "notes": "",
                    }
                )

            catalogue = pd.DataFrame(rows)
            fold_results = create_kfold_pairs(catalogue, config, n_folds=5)

        self.assertEqual(len(fold_results), 5)
        for pairs_df, split_indices in fold_results:
            self.assertGreater(len(pairs_df), 0)
            self.assertTrue((pairs_df["post_reuse_count"] <= config.split.max_postop_reuse).all())
            self.assertEqual(set(split_indices.keys()), {"train", "val", "test"})


if __name__ == "__main__":
    unittest.main()
