"""Visualisation utilities for training curves, sample grids, and overlays."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np


def plot_training_curves(
    history: Dict[str, list],
    output_path: Path,
) -> None:
    """Plot G/D loss curves, L1, SSIM, and validation metrics over epochs."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    epochs = range(1, len(history.get("G_loss", [])) + 1)

    # G and D loss
    ax = axes[0, 0]
    if "G_loss" in history:
        ax.plot(epochs, history["G_loss"], label="G Total", color="#2ECC71")
    if "D_loss" in history:
        ax.plot(epochs, history["D_loss"], label="D Total", color="#E74C3C")
    ax.set_title("Generator vs Discriminator Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()
    ax.grid(alpha=0.3)

    # L1 loss
    ax = axes[0, 1]
    if "G_l1" in history:
        ax.plot(epochs, history["G_l1"], color="#3498DB")
    ax.set_title("L1 Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("L1")
    ax.grid(alpha=0.3)

    # SSIM loss
    ax = axes[0, 2]
    if "G_ssim" in history:
        ax.plot(epochs, history["G_ssim"], color="#9B59B6")
    ax.set_title("SSIM Loss (1 - SSIM)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("SSIM Loss")
    ax.grid(alpha=0.3)

    # Validation SSIM
    ax = axes[1, 0]
    if "val_ssim" in history:
        ax.plot(epochs, history["val_ssim"], color="#2ECC71", linewidth=2)
        best_idx = int(np.argmax(history["val_ssim"]))
        ax.axvline(best_idx + 1, color="red", linestyle="--", alpha=0.5,
                    label=f"Best: {history['val_ssim'][best_idx]:.4f}")
        ax.legend()
    ax.set_title("Validation SSIM")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("SSIM")
    ax.grid(alpha=0.3)

    # Validation PSNR
    ax = axes[1, 1]
    if "val_psnr" in history:
        ax.plot(epochs, history["val_psnr"], color="#F39C12", linewidth=2)
    ax.set_title("Validation PSNR (dB)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("PSNR")
    ax.grid(alpha=0.3)

    # Learning rate
    ax = axes[1, 2]
    if "lr_G" in history:
        ax.plot(epochs, history["lr_G"], label="LR_G", color="#3498DB")
    if "lr_D" in history:
        ax.plot(epochs, history["lr_D"], label="LR_D", color="#E74C3C")
    ax.set_title("Learning Rates")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("LR")
    ax.legend()
    ax.set_yscale("log")
    ax.grid(alpha=0.3)

    fig.suptitle("Pix2Pix Training Progress", fontsize=16)
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Training curves saved to {output_path}")
