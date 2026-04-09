"""Pix2Pix U-Net Generator (D1 §3.7)."""

from __future__ import annotations

import torch
import torch.nn as nn


class UNetEncoderBlock(nn.Module):
    """Single encoder block: Conv2d(4×4, s=2) → BN → LeakyReLU(0.2)."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        use_batchnorm: bool = True,
    ):
        super().__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, 4, stride=2, padding=1, bias=False),
        ]
        if use_batchnorm:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UNetDecoderBlock(nn.Module):
    """Single decoder block: ConvTranspose2d(4×4, s=2) → BN → Dropout → ReLU."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        use_dropout: bool = False,
        dropout_rate: float = 0.5,
    ):
        super().__init__()
        layers = [
            nn.ConvTranspose2d(in_channels, out_channels, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
        ]
        if use_dropout:
            layers.append(nn.Dropout(dropout_rate))
        layers.append(nn.ReLU(inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UNetGenerator(nn.Module):
    """8-downsample Pix2Pix generator for `512x512` single-channel inputs."""

    def __init__(self, in_channels: int = 1, out_channels: int = 1, base: int = 64):
        super().__init__()

        # ── Encoder ─────────────────────────────────────────────────────
        self.e1 = UNetEncoderBlock(in_channels, base, use_batchnorm=False)   # 512→256
        self.e2 = UNetEncoderBlock(base, base * 2)                            # 256→128
        self.e3 = UNetEncoderBlock(base * 2, base * 4)                        # 128→64
        self.e4 = UNetEncoderBlock(base * 4, base * 8)                        # 64→32
        self.e5 = UNetEncoderBlock(base * 8, base * 8)                        # 32→16
        self.e6 = UNetEncoderBlock(base * 8, base * 8)                        # 16→8
        self.e7 = UNetEncoderBlock(base * 8, base * 8)                        # 8→4
        self.e8 = UNetEncoderBlock(base * 8, base * 8)                        # 4→2

        # ── Bottleneck ──────────────────────────────────────────────────
        self.bottleneck = nn.Sequential(
            nn.Conv2d(base * 8, base * 8, 4, stride=2, padding=1, bias=False),  # 2→1
            nn.ReLU(inplace=True),
        )

        # ── Decoder ─────────────────────────────────────────────────────
        self.d8 = UNetDecoderBlock(base * 8, base * 8, use_dropout=True)       # 1→2
        self.d7 = UNetDecoderBlock(base * 16, base * 8, use_dropout=True)      # 2→4
        self.d6 = UNetDecoderBlock(base * 16, base * 8, use_dropout=True)      # 4→8
        self.d5 = UNetDecoderBlock(base * 16, base * 8)                        # 8→16
        self.d4 = UNetDecoderBlock(base * 16, base * 8)                        # 16→32
        self.d3 = UNetDecoderBlock(base * 16, base * 4)                        # 32→64
        self.d2 = UNetDecoderBlock(base * 8, base * 2)                         # 64→128
        self.d1 = UNetDecoderBlock(base * 4, base)                             # 128→256

        # ── Final ───────────────────────────────────────────────────────
        self.final = nn.Sequential(
            nn.ConvTranspose2d(base * 2, out_channels, 4, stride=2, padding=1),  # 256→512 (concat e1)
            nn.Tanh(),
        )

        # Weight initialisation (Xavier/Glorot)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m: nn.Module) -> None:
        """Initialise weights using Xavier/Glorot (D1 §3.7)."""
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (B, 1, 512, 512).

        Returns:
            Output tensor of shape (B, 1, 512, 512) in [-1, 1].
        """
        e1 = self.e1(x)
        e2 = self.e2(e1)
        e3 = self.e3(e2)
        e4 = self.e4(e3)
        e5 = self.e5(e4)
        e6 = self.e6(e5)
        e7 = self.e7(e6)
        e8 = self.e8(e7)
        b = self.bottleneck(e8)

        d8 = self.d8(b)
        d7 = self.d7(torch.cat([d8, e8], dim=1))
        d6 = self.d6(torch.cat([d7, e7], dim=1))
        d5 = self.d5(torch.cat([d6, e6], dim=1))
        d4 = self.d4(torch.cat([d5, e5], dim=1))
        d3 = self.d3(torch.cat([d4, e4], dim=1))
        d2 = self.d2(torch.cat([d3, e3], dim=1))
        d1 = self.d1(torch.cat([d2, e2], dim=1))
        return self.final(torch.cat([d1, e1], dim=1))
