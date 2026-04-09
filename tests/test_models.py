import unittest

import torch

from thapipeline.models.patchgan import PatchGANDiscriminator
from thapipeline.models.pix2pix_unet import UNetGenerator


class ModelSmokeTests(unittest.TestCase):
    def test_unet_generator_forward_shape(self) -> None:
        model = UNetGenerator()
        x = torch.randn(1, 1, 512, 512)
        y = model(x)
        self.assertEqual(tuple(y.shape), (1, 1, 512, 512))

    def test_patchgan_forward_shape(self) -> None:
        model = PatchGANDiscriminator()
        x = torch.randn(1, 1, 512, 512)
        y = torch.randn(1, 1, 512, 512)
        out = model(x, y)
        self.assertEqual(tuple(out.shape), (1, 1, 30, 30))


if __name__ == "__main__":
    unittest.main()
