import unittest

import numpy as np
import trimesh

from thapipeline.eval.evaluate_full_pipeline import _nested_metric_or_none
from thapipeline.eval.metrics import compute_dice, compute_psnr, compute_ssim
from thapipeline.utils.mesh_utils import project_mesh_to_mask


class MetricTests(unittest.TestCase):
    def test_basic_image_metrics(self) -> None:
        image = np.zeros((64, 64), dtype=np.uint8)
        image[16:48, 16:48] = 255
        self.assertAlmostEqual(compute_dice(image, image), 1.0)
        self.assertTrue(compute_psnr(image, image) == float("inf"))
        self.assertAlmostEqual(compute_ssim(image, image), 1.0)

    def test_project_mesh_to_mask(self) -> None:
        mesh = trimesh.creation.box(extents=(10, 10, 10))
        mesh.apply_translation([20, 20, 0])
        mask = project_mesh_to_mask(mesh, (128, 128), dpi=150.0)
        self.assertEqual(mask.shape, (128, 128))
        self.assertGreater(int(mask.sum()), 0)

    def test_project_mesh_to_mask_ignores_nan_vertices(self) -> None:
        mesh = trimesh.creation.box(extents=(10, 10, 10))
        mesh.vertices[0, 0] = np.nan
        mask = project_mesh_to_mask(mesh, (128, 128), dpi=150.0)
        self.assertEqual(mask.shape, (128, 128))

    def test_nested_metric_or_none_handles_missing_sections(self) -> None:
        self.assertIsNone(_nested_metric_or_none({}, "stem", "length_mm"))
        self.assertIsNone(_nested_metric_or_none({"stem": None}, "stem", "length_mm"))
        self.assertEqual(_nested_metric_or_none({"stem": {"length_mm": 12.5}}, "stem", "length_mm"), 12.5)


if __name__ == "__main__":
    unittest.main()
