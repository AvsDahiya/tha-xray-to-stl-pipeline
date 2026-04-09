"""PatchGAN discriminator for paired radiograph realism scoring."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm


class PatchGANDiscriminator(nn.Module):
    """70x70 PatchGAN discriminator for `512x512` paired inputs."""

    def __init__(
        self,
        in_channels: int = 2,
        base_filters: int = 64,
        use_spectral_norm: bool = True,
    ):
        super().__init__()

        def _conv_block(
            in_c: int,
            out_c: int,
            stride: int = 2,
            use_bn: bool = True,
            use_sn: bool = True,
        ) -> nn.Sequential:
            """Create a discriminator convolutional block."""
            conv = nn.Conv2d(in_c, out_c, 4, stride=stride, padding=1, bias=not use_bn)
            if use_sn and use_spectral_norm:
                conv = spectral_norm(conv)
            layers = [conv]
            if use_bn:
                layers.append(nn.BatchNorm2d(out_c))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return nn.Sequential(*layers)

        self.model = nn.Sequential(
            # Block 1: No BatchNorm (D1 §3.7)
            _conv_block(in_channels, base_filters, stride=2, use_bn=False),

            # Block 2
            _conv_block(base_filters, base_filters * 2, stride=2),

            # Block 3
            _conv_block(base_filters * 2, base_filters * 4, stride=2),

            # Block 4
            _conv_block(base_filters * 4, base_filters * 8, stride=2),

            # Block 5: Final 1-channel output
            nn.Conv2d(base_filters * 8, 1, 4, stride=1, padding=1),
            nn.Sigmoid(),
        )

        # Apply spectral norm to final conv too
        if use_spectral_norm:
            # Last conv before Sigmoid
            self.model[-2] = spectral_norm(self.model[-2])

        # Weight initialisation
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m: nn.Module) -> None:
        """Initialise weights using Xavier/Glorot (D1 §3.7)."""
        if isinstance(m, nn.Conv2d):
            # spectral_norm wraps the module, check for weight_orig
            if hasattr(m, "weight_orig"):
                nn.init.xavier_normal_(m.weight_orig)
            else:
                nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Pre-operative image tensor (B, 1, H, W).
            y: Post-operative (real or generated) image tensor (B, 1, H, W).

        Returns:
            Probability map of shape (B, 1, ~30, ~30).
        """
        combined = torch.cat([x, y], dim=1)
        output = self.model(combined)

        # The dissertation text expects a 30x30 map; the 512px input geometry
        # yields 31x31 with the literal conv stack, so crop one border pixel.
        if output.shape[-2:] == (31, 31):
            output = output[..., :30, :30]
        return output
