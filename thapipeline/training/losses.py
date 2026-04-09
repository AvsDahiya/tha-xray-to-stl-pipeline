"""Composite loss functions for Pix2Pix training (D1 §3.7).

L_total = L_cGAN + λ₁·L_L1 + λ₂·L_SSIM

Components:
  - L_cGAN: Conditional adversarial loss (BCE)
  - L_L1: Pixel-wise L1 loss (λ₁=100)
  - L_SSIM: 1 - SSIM structural similarity loss (λ₂=10)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SSIMLoss(nn.Module):
    """Structural Similarity Index Measure loss (D1 §3.7).

    SSIM(y, ŷ) = (2μ_y·μ_ŷ + C1)(2σ_yŷ + C2) / (μ²_y + μ²_ŷ + C1)(σ²_y + σ²_ŷ + C2)

    Loss = 1 - SSIM(y, ŷ)

    Uses 11×11 Gaussian window for local statistics.
    """

    def __init__(self, window_size: int = 11, channel: int = 1):
        super().__init__()
        self.window_size = window_size
        self.channel = channel

        # Create Gaussian window
        sigma = 1.5
        coords = torch.arange(window_size, dtype=torch.float32) - window_size // 2
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g = g / g.sum()
        window_1d = g.unsqueeze(1)
        window_2d = window_1d @ window_1d.t()
        window_2d = window_2d.unsqueeze(0).unsqueeze(0)  # (1, 1, k, k)
        self.register_buffer("window", window_2d.expand(channel, 1, -1, -1).contiguous())

    def _ssim(
        self,
        img1: torch.Tensor,
        img2: torch.Tensor,
    ) -> torch.Tensor:
        """Compute SSIM between two images."""
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        pad = self.window_size // 2

        mu1 = F.conv2d(img1, self.window, padding=pad, groups=self.channel)
        mu2 = F.conv2d(img2, self.window, padding=pad, groups=self.channel)

        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu12 = mu1 * mu2

        sigma1_sq = F.conv2d(img1 ** 2, self.window, padding=pad, groups=self.channel) - mu1_sq
        sigma2_sq = F.conv2d(img2 ** 2, self.window, padding=pad, groups=self.channel) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, self.window, padding=pad, groups=self.channel) - mu12

        ssim_map = ((2 * mu12 + C1) * (2 * sigma12 + C2)) / \
                   ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

        return ssim_map.mean()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute SSIM loss = 1 - SSIM."""
        # Move window to same device as input
        if self.window.device != pred.device:
            self.window = self.window.to(pred.device)
        return 1.0 - self._ssim(pred, target)


class CompositeLoss(nn.Module):
    """Composite loss: L_total = L_cGAN + λ₁·L_L1 + λ₂·L_SSIM (D1 §3.7).

    Args:
        lambda_L1: Weight for L1 loss (default 100).
        lambda_SSIM: Weight for SSIM loss (default 10).
        label_smoothing: Smooth real labels to this value (default 0.9).
    """

    def __init__(
        self,
        lambda_L1: float = 100.0,
        lambda_SSIM: float = 10.0,
        label_smoothing: float = 0.9,
    ):
        super().__init__()
        self.lambda_L1 = lambda_L1
        self.lambda_SSIM = lambda_SSIM
        self.label_smoothing = label_smoothing

        self.bce = nn.BCELoss()
        self.l1 = nn.L1Loss()
        self.ssim = SSIMLoss(window_size=11, channel=1)

    def generator_loss(
        self,
        fake_pred: torch.Tensor,       # D(x, G(x))
        generated: torch.Tensor,        # G(x)
        target: torch.Tensor,           # y (real post-op)
    ) -> dict:
        """Compute generator loss components.

        Returns:
            Dictionary with 'total', 'adversarial', 'l1', 'ssim' keys.
        """
        # Adversarial loss: G wants D to classify fakes as real
        real_label = torch.ones_like(fake_pred) * self.label_smoothing
        loss_adv = self.bce(fake_pred, real_label)

        # L1 pixel loss
        loss_l1 = self.l1(generated, target)

        # SSIM loss  
        # Rescale from [-1,1] to [0,1] for SSIM computation
        gen_01 = (generated + 1.0) / 2.0
        tgt_01 = (target + 1.0) / 2.0
        loss_ssim = self.ssim(gen_01, tgt_01)

        # Total
        total = loss_adv + self.lambda_L1 * loss_l1 + self.lambda_SSIM * loss_ssim

        return {
            "total": total,
            "adversarial": loss_adv,
            "l1": loss_l1,
            "ssim": loss_ssim,
        }

    def discriminator_loss(
        self,
        real_pred: torch.Tensor,        # D(x, y)
        fake_pred: torch.Tensor,        # D(x, G(x))
    ) -> dict:
        """Compute discriminator loss.

        Returns:
            Dictionary with 'total', 'real', 'fake' keys.
        """
        # Real loss with label smoothing
        real_label = torch.ones_like(real_pred) * self.label_smoothing
        loss_real = self.bce(real_pred, real_label)

        # Fake loss
        fake_label = torch.zeros_like(fake_pred)
        loss_fake = self.bce(fake_pred, fake_label)

        total = (loss_real + loss_fake) / 2.0

        return {
            "total": total,
            "real": loss_real,
            "fake": loss_fake,
        }


class BaselineLoss(CompositeLoss):
    """Baseline loss without SSIM: L_total = L_cGAN + λ₁·L_L1.

    Used for ablation studies comparing with/without SSIM component.
    """

    def __init__(
        self,
        lambda_L1: float = 100.0,
        label_smoothing: float = 0.9,
    ):
        super().__init__(
            lambda_L1=lambda_L1,
            lambda_SSIM=0.0,
            label_smoothing=label_smoothing,
        )
