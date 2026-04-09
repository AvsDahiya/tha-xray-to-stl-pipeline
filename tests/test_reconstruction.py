import unittest

import cv2
import numpy as np

from thapipeline.models.recon_3d import reconstruct_from_mask
from thapipeline.models.segmenter import analyze_components, select_stem_component


class ReconstructionRegressionTests(unittest.TestCase):
    def test_select_stem_component_accepts_cup_geometry_dict(self) -> None:
        mask = np.zeros((256, 256), dtype=np.uint8)
        cv2.circle(mask, (80, 80), 28, 255, -1)
        cv2.rectangle(mask, (150, 90), (180, 220), 255, -1)

        components = analyze_components(mask, min_area=20)
        cup_geometry = {"center": (80.0, 80.0), "radius": 28.0}

        stem = select_stem_component(components, cup_geometry)

        self.assertIsNotNone(stem)
        self.assertIn("centroid", stem)

    def test_reconstruct_from_mask_handles_component_and_geometry_mix(self) -> None:
        mask = np.zeros((256, 256), dtype=np.uint8)
        cv2.circle(mask, (80, 80), 28, 255, -1)
        cv2.ellipse(mask, (160, 165), (20, 72), 8, 0, 360, 255, -1)

        mesh, metadata = reconstruct_from_mask(mask, optimize=False, smooth_iterations=1)

        self.assertIsNotNone(mesh)
        self.assertIn("cup", metadata)
        self.assertIn("stem", metadata)
        self.assertIsNotNone(metadata["cup"])
        self.assertIsNotNone(metadata["stem"])


if __name__ == "__main__":
    unittest.main()
