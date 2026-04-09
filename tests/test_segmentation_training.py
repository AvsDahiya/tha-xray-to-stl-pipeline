import unittest

import numpy as np

from thapipeline.config import PipelineConfig
from thapipeline.training.train_segmenter import split_segmentation_records


class SegmentationTrainingTests(unittest.TestCase):
    def test_split_segmentation_records_has_disjoint_case_ids(self):
        config = PipelineConfig()
        records = []
        for idx in range(12):
            records.append(
                {
                    "case_id": f"case_{idx:02d}",
                    "enhanced": np.zeros((8, 8), dtype=np.uint8),
                    "gan": np.zeros((8, 8), dtype=np.uint8),
                    "grad": np.zeros((8, 8), dtype=np.uint8),
                    "label": np.zeros((8, 8), dtype=np.float32),
                }
            )

        split_records, manifest = split_segmentation_records(records, config)
        train_ids = set(manifest["train"]["case_ids"])
        val_ids = set(manifest["val"]["case_ids"])
        test_ids = set(manifest["test"]["case_ids"])

        self.assertTrue(train_ids.isdisjoint(val_ids))
        self.assertTrue(train_ids.isdisjoint(test_ids))
        self.assertTrue(val_ids.isdisjoint(test_ids))
        self.assertEqual(
            len(split_records["train"]) + len(split_records["val"]) + len(split_records["test"]),
            len(records),
        )
